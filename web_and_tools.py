"""Gatekeeper Policy, Sandboxed Code Runner, Web Extraction, Agent-Reach, Market Data, & Parallel Tool DAG."""

import asyncio
import html
import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
import uuid
from typing import Awaitable, Callable

import aiohttp

from config import ARIES_DATA_DIR
from efficiency_engine import ToolCallRepairer, context_sandbox, tool_cache

GATEKEEPER_PATH = os.path.join(ARIES_DATA_DIR, "gatekeeper.json")


class GatekeeperPolicy:
    """Zero-trust per-capability permissions (allow / ask / deny) + audit log."""

    DEFAULT_RULES = {
        "web_search": "allow",
        "web_fetch": "allow",
        "code_sandbox": "allow",
        "open_app": "allow",
        "system_info": "allow",
        "memory": "allow",
        "market_data": "allow",
    }

    def __init__(self, path: str = GATEKEEPER_PATH):
        self.path = path
        self._rules = dict(self.DEFAULT_RULES)
        self._audit: list[dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                if isinstance(d, dict):
                    for k, v in d.items():
                        if k in self._rules and v in ("allow", "ask", "deny"):
                            self._rules[k] = v
            except Exception:
                pass

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._rules, fh, indent=2)

    def get_state(self) -> dict:
        return {"rules": dict(self._rules), "audit": self._audit[-40:]}

    def update(self, rules: dict) -> dict:
        for k, v in rules.items():
            if k in self._rules and v in ("allow", "ask", "deny"):
                self._rules[k] = v
        self._save()
        return self.get_state()

    def check(self, capability: str, tool_name: str) -> tuple[bool, str]:
        mode = self._rules.get(capability, "allow")
        self._audit.append({
            "ts": time.time(),
            "capability": capability,
            "tool": tool_name,
            "decision": mode,
        })
        if len(self._audit) > 100:
            self._audit = self._audit[-100:]
        if mode == "deny":
            return False, f"Blocked by Gatekeeper Policy: '{capability}' is set to Deny."
        return True, ""


class SandboxedCodeRunner:
    """Executes Python / Bash / JS snippets inside /private/tmp/aries_sandbox_<uuid> with a 10s timeout."""

    BLOCKED_PATTERNS = ("rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:", "shutdown", "reboot")

    @classmethod
    def run(cls, language: str, code: str, timeout: float = 10.0) -> dict:
        lang = (language or "python").strip().lower()
        low = code.lower()
        for pat in cls.BLOCKED_PATTERNS:
            if pat in low:
                return {"ok": False, "stdout": "", "stderr": f"Blocked unsafe pattern: {pat}"}

        sandbox_dir = f"/private/tmp/aries_sandbox_{uuid.uuid4().hex[:8]}"
        os.makedirs(sandbox_dir, exist_ok=True)
        try:
            if lang in ("py", "python", "python3"):
                script = os.path.join(sandbox_dir, "main.py")
                with open(script, "w", encoding="utf-8") as fh:
                    fh.write(code)
                cmd = ["python3", script]
            elif lang in ("sh", "bash", "zsh"):
                script = os.path.join(sandbox_dir, "run.sh")
                with open(script, "w", encoding="utf-8") as fh:
                    fh.write(code)
                cmd = ["zsh", script]
            elif lang in ("js", "javascript", "node"):
                script = os.path.join(sandbox_dir, "main.mjs")
                with open(script, "w", encoding="utf-8") as fh:
                    fh.write(code)
                cmd = ["node", script]
            else:
                return {"ok": False, "stdout": "", "stderr": f"Unsupported sandbox language: {language}"}

            res = subprocess.run(
                cmd,
                cwd=sandbox_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "ok": res.returncode == 0,
                "returncode": res.returncode,
                "stdout": res.stdout[:12_000],
                "stderr": res.stderr[:4_000],
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "stdout": "", "stderr": f"Execution timed out after {timeout}s."}
        except Exception as e:
            return {"ok": False, "stdout": "", "stderr": str(e)}
        finally:
            shutil.rmtree(sandbox_dir, ignore_errors=True)


class DOMMarkdownExtractor:
    """Strips HTML navbars/ads/scripts into clean Markdown."""

    @staticmethod
    def html_to_markdown(raw_html: str, max_chars: int = 12_000) -> str:
        text = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw_html, flags=re.S | re.I)
        text = re.sub(r"<h([1-4])[^>]*>(.*?)</h\1>", lambda m: f"\n{'#' * int(m.group(1))} {m.group(2)}\n", text, flags=re.S | re.I)
        text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.S | re.I)
        text = re.sub(r"<br\s*/?>|</p>|</div>|</tr>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
        cleaned = "\n".join(l for l in lines if l)
        return cleaned[:max_chars]


async def fetch_url_markdown(url: str) -> str:
    cached = tool_cache.get("read_url_markdown", url)
    if cached:
        return cached
    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "ARIES-Public-Preview/1.0"}) as sess:
        async with sess.get(url, allow_redirects=True) as resp:
            if resp.status >= 400:
                return f"HTTP {resp.status} fetching {url}"
            raw = await resp.text(errors="replace")
    md = DOMMarkdownExtractor.html_to_markdown(raw)
    tool_cache.set("read_url_markdown", url, md, ttl=900)
    return md


async def agent_reach_fetch(target: str) -> str:
    """Zero-API-key reader for GitHub READMEs, Reddit JSON, and YouTube pages."""
    cached = tool_cache.get("agent_reach_fetch", target)
    if cached:
        return cached
    url = target.strip()
    if "reddit.com" in url and not url.endswith(".json"):
        url = url.rstrip("/") + ".json"
    elif "github.com/" in url and "/blob/" not in url and url.count("/") >= 4:
        parts = url.rstrip("/").split("/")
        owner, repo = parts[3], parts[4]
        url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/README.md"

    timeout = aiohttp.ClientTimeout(total=12)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "ARIES-Public-Preview/1.0"}) as sess:
        async with sess.get(url, allow_redirects=True) as resp:
            raw = await resp.text(errors="replace")

    if url.endswith(".json"):
        try:
            data = json.loads(raw)
            out = json.dumps(data, indent=2)[:8000]
        except Exception:
            out = raw[:8000]
    elif "<html" in raw.lower():
        out = DOMMarkdownExtractor.html_to_markdown(raw, max_chars=8000)
    else:
        out = raw[:8000]

    tool_cache.set("agent_reach_fetch", target, out, ttl=900)
    return out


async def fetch_stock_quote(symbol: str) -> str:
    """Zero-API-key Yahoo Finance stock quote lookup."""
    sym = symbol.strip().upper()
    cached = tool_cache.get("get_stock_quote", sym)
    if cached:
        return cached
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}?interval=1d&range=5d"
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": "Mozilla/5.0"}) as sess:
        async with sess.get(url) as resp:
            if resp.status != 200:
                return f"Could not fetch quote for {sym} (HTTP {resp.status})."
            data = await resp.json()
    try:
        meta = data["chart"]["result"][0]["meta"]
        price = meta.get("regularMarketPrice")
        prev = meta.get("chartPreviousClose")
        curr = meta.get("currency", "USD")
        chg = ((price - prev) / prev * 100.0) if (price and prev) else 0.0
        out = f"{sym}: {price} {curr} ({chg:+.2f}% vs prev close {prev})"
        tool_cache.set("get_stock_quote", sym, out, ttl=300)
        return out
    except Exception:
        return f"No quote data found for {sym}."


gatekeeper = GatekeeperPolicy()


async def execute_tools_dag(
    tool_calls: list[dict],
    executor: Callable[[str, dict], Awaitable[str]],
) -> list[dict]:
    """Executes tool calls concurrently with repair, Gatekeeper check, and ContextMode sandboxing."""

    async def _run_one(tc: dict) -> dict:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args = ToolCallRepairer.parse_args(fn.get("arguments", {}))
        cap = "web_search" if "search" in name else ("web_fetch" if "url" in name or "reach" in name else ("code_sandbox" if "code" in name else "system_info"))
        allowed, reason = gatekeeper.check(cap, name)
        if not allowed:
            result = reason
        else:
            raw_res = await executor(name, args)
            result = context_sandbox.evaluate(name, str(raw_res))
        return {"tool_call_id": tc.get("id", ""), "name": name, "content": result}

    return await asyncio.gather(*(_run_one(tc) for tc in tool_calls))
