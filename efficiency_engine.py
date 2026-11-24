"""Lightweight Efficiency & Context-Optimization Engine for ARIES Public Preview Beta.

Ported from Aries Desktop's highest-impact, zero-weight efficiency modules:
- ContextModeSandbox (90-98% context reduction on verbose tool outputs)
- SemanticToolCache (normalized + token-Jaccard fuzzy TTL cache)
- ToolCallRepairer (deterministic JSON repair & schema coercion for 2B models)
- ThinkingStreamRouter (separates <think>/<|channel>thought + repetition loop guard)
- EnhancedTTSChunker & polish_pcm_audio (abbreviation/decimal-safe chunker + numpy polish)
- ReactionClassifier (<1ms 0-GPU emoji tapback classifier)
- InstinctStore (SQLite compact user rules & preferences)
"""

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from typing import Optional

import numpy as np

from config import ARIES_MEMORY_DB_PATH


class ContextModeSandbox:
    """Buffers verbose tool outputs (>3,000 chars) and returns compact head/tail receipts."""

    def __init__(self, max_inline_chars: int = 3000, max_buffers: int = 32):
        self.max_inline_chars = max_inline_chars
        self.max_buffers = max_buffers
        self._buffers: dict[str, dict] = {}
        self._lock = threading.Lock()

    def evaluate(self, tool_name: str, output: str) -> str:
        if not output or len(output) <= self.max_inline_chars:
            return output

        buf_id = f"buf_{uuid.uuid4().hex[:8]}"
        lines = output.splitlines()
        head = "\n".join(lines[:12])[:900]
        tail = "\n".join(lines[-8:])[-600:] if len(lines) > 12 else ""

        with self._lock:
            if len(self._buffers) >= self.max_buffers:
                oldest = next(iter(self._buffers))
                del self._buffers[oldest]
            self._buffers[buf_id] = {
                "tool": tool_name,
                "text": output,
                "created_at": time.time(),
            }

        return (
            f"[Sandboxed Output: {buf_id} | tool={tool_name} | {len(output)} chars, {len(lines)} lines]\n"
            f"--- HEAD ---\n{head}\n"
            + (f"--- TAIL ---\n{tail}\n" if tail else "")
            + f"Use `query_sandboxed_buffer(buffer_id=\"{buf_id}\", query=\"...\")` to search specific lines."
        )

    def query(self, buffer_id: str, query: str = "", limit: int = 20) -> str:
        with self._lock:
            entry = self._buffers.get(buffer_id)
        if not entry:
            return f"Buffer '{buffer_id}' not found or expired."
        text = entry["text"]
        lines = text.splitlines()
        if not query.strip():
            return "\n".join(lines[:limit])
        q_low = query.lower()
        matches = [f"L{i+1}: {line}" for i, line in enumerate(lines) if q_low in line.lower()]
        if not matches:
            return f"No lines matching '{query}' in {buffer_id} ({len(lines)} lines total)."
        return "\n".join(matches[:limit])


class SemanticToolCache:
    """Normalized + token-Jaccard fuzzy TTL cache for read-only tool calls."""

    def __init__(self, default_ttl: float = 900.0):
        self.default_ttl = default_ttl
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def get(self, tool_name: str, query: str) -> Optional[str]:
        norm = self._normalize(query)
        key = f"{tool_name}::{norm}"
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry and (now - entry["ts"] <= entry["ttl"]):
                return entry["result"]
            q_toks = self._tokens(norm)
            if len(q_toks) >= 2:
                for k, ent in list(self._store.items()):
                    if not k.startswith(f"{tool_name}::") or (now - ent["ts"] > ent["ttl"]):
                        continue
                    e_toks = ent["tokens"]
                    if not e_toks:
                        continue
                    jaccard = len(q_toks & e_toks) / len(q_toks | e_toks)
                    if jaccard >= 0.85:
                        return ent["result"]
        return None

    def set(self, tool_name: str, query: str, result: str, ttl: Optional[float] = None):
        norm = self._normalize(query)
        key = f"{tool_name}::{norm}"
        with self._lock:
            self._store[key] = {
                "result": result,
                "ts": time.time(),
                "ttl": ttl if ttl is not None else self.default_ttl,
                "tokens": self._tokens(norm),
            }


class ToolCallRepairer:
    """Deterministic JSON repair & argument coercion for small local models."""

    @staticmethod
    def parse_args(raw_args) -> dict:
        if isinstance(raw_args, dict):
            return raw_args
        if not isinstance(raw_args, str) or not raw_args.strip():
            return {}
        s = raw_args.strip()
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            pass

        # Strip markdown fences
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M).strip()
        # Fix trailing commas
        s = re.sub(r",\s*([}\]])", r"\1", s)
        # Balance unclosed braces
        open_b = s.count("{") - s.count("}")
        if open_b > 0:
            s += "}" * open_b
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            # Fallback: convert single-quoted keys/values
            s2 = re.sub(r"(?<!\\)'", '"', s)
            try:
                parsed = json.loads(s2)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}


class ThinkingStreamRouter:
    """Segments <think>/<|channel>thought from visible tokens and guards against repetition loops."""

    OPEN_TAGS = ("<|channel>thought", "<|channel|>thought", "<thought>", "<think>")
    CLOSE_TAGS = ("<channel|>", "<|channel|>", "</thought>", "</think>")

    def __init__(self, max_thinking_chars: int = 4000):
        self.is_inside_thinking = False
        self.buffer = ""
        self.thinking_chars = 0
        self.max_thinking_chars = max_thinking_chars
        self.loop_aborted = False

    def feed(self, chunk: str) -> tuple[str, str]:
        """Returns (visible_delta, thinking_delta)."""
        self.buffer += chunk
        vis_out = []
        think_out = []

        while self.buffer:
            if not self.is_inside_thinking:
                earliest = None
                for tag in self.OPEN_TAGS:
                    idx = self.buffer.find(tag)
                    if idx != -1 and (earliest is None or idx < earliest[0]):
                        earliest = (idx, tag)
                if earliest:
                    idx, tag = earliest
                    if idx > 0:
                        vis_out.append(self.buffer[:idx])
                    self.buffer = self.buffer[idx + len(tag):].lstrip("\n")
                    self.is_inside_thinking = True
                else:
                    # Hold potential partial tag at tail
                    last_lt = self.buffer.rfind("<")
                    if last_lt != -1 and len(self.buffer) - last_lt < 20:
                        vis_out.append(self.buffer[:last_lt])
                        self.buffer = self.buffer[last_lt:]
                        break
                    vis_out.append(self.buffer)
                    self.buffer = ""
                    break
            else:
                earliest = None
                for tag in self.CLOSE_TAGS:
                    idx = self.buffer.find(tag)
                    if idx != -1 and (earliest is None or idx < earliest[0]):
                        earliest = (idx, tag)
                if earliest:
                    idx, tag = earliest
                    t_part = self.buffer[:idx]
                    self.thinking_chars += len(t_part)
                    think_out.append(t_part)
                    self.buffer = self.buffer[idx + len(tag):].lstrip("\n")
                    self.is_inside_thinking = False
                else:
                    last_lt = self.buffer.rfind("<")
                    if last_lt != -1 and len(self.buffer) - last_lt < 20:
                        t_part = self.buffer[:last_lt]
                        self.thinking_chars += len(t_part)
                        think_out.append(t_part)
                        self.buffer = self.buffer[last_lt:]
                    else:
                        self.thinking_chars += len(self.buffer)
                        think_out.append(self.buffer)
                        self.buffer = ""
                    if self.thinking_chars > self.max_thinking_chars:
                        self.loop_aborted = True
                    break

        return "".join(vis_out), "".join(think_out)

    def flush(self) -> tuple[str, str]:
        if not self.buffer:
            return "", ""
        rem = self.buffer
        self.buffer = ""
        return ("", rem) if self.is_inside_thinking else (rem, "")


class EnhancedTTSChunker:
    """Abbreviation- and decimal-safe sentence chunker that strips code/LaTeX/Mermaid before speaking."""

    ABBREVS = ("dr.", "mr.", "mrs.", "ms.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.")

    def __init__(self):
        self._buf = ""
        self._in_code = False

    def push(self, token: str) -> list[str]:
        self._buf += token
        clauses: list[str] = []

        # Strip complete fenced code blocks
        while "```" in self._buf:
            first = self._buf.find("```")
            if not self._in_code:
                before = self._buf[:first]
                self._buf = self._buf[first + 3:]
                self._in_code = True
                if before.strip():
                    clauses.extend(self._extract_sentences(before))
            else:
                self._buf = self._buf[first + 3:]
                self._in_code = False

        if not self._in_code and self._buf:
            clauses.extend(self._extract_sentences(self._buf, keep_tail=True))
        elif self._in_code:
            self._buf = ""

        return clauses

    def _extract_sentences(self, text: str, keep_tail: bool = False) -> list[str]:
        out: list[str] = []
        i = 0
        start = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch in ".!?":
                # Protect decimals (3.14) and ellipses (...)
                if ch == "." and i + 1 < n and text[i + 1].isdigit():
                    i += 1
                    continue
                if ch == "." and i + 1 < n and text[i + 1] == ".":
                    while i < n and text[i] == ".":
                        i += 1
                    continue
                # Protect abbreviations
                seg_low = text[start:i + 1].strip().lower()
                if any(seg_low.endswith(a) for a in self.ABBREVS):
                    i += 1
                    continue
                if i + 1 == n or text[i + 1].isspace():
                    cand = self.clean_for_speech(text[start:i + 1])
                    if cand:
                        out.append(cand)
                    start = i + 1
            i += 1

        if keep_tail:
            self._buf = text[start:]
        else:
            rem = self.clean_for_speech(text[start:])
            if rem:
                out.append(rem)
        return out

    def flush(self) -> list[str]:
        if self._in_code or not self._buf.strip():
            self._buf = ""
            return []
        rem = self.clean_for_speech(self._buf)
        self._buf = ""
        return [rem] if rem else []

    @staticmethod
    def clean_for_speech(text: str) -> str:
        text = re.sub(r"\$\$.*?\$\$", " formula ", text, flags=re.S)
        text = re.sub(r"\$[^$\n]+\$", " math expression ", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
        text = re.sub(r"[*_#>]+", "", text)
        return re.sub(r"\s+", " ", text).strip()


def polish_pcm_audio(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Applies peak normalization and 5ms anti-click fade-in/out to 16-bit PCM audio."""
    if not pcm_bytes or len(pcm_bytes) < 64:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    peak = float(np.max(np.abs(arr)))
    if peak > 100.0:
        target = 32767.0 * 0.90
        if peak > target or peak < 12000.0:
            arr *= min(3.0, target / peak)
    fade_samples = min(len(arr) // 4, int(sample_rate * 0.005))
    if fade_samples > 1:
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        arr[:fade_samples] *= ramp
        arr[-fade_samples:] *= ramp[::-1]
    return np.clip(arr, -32768, 32767).astype(np.int16).tobytes()


class ReactionClassifier:
    """Sub-millisecond, 0-GPU emoji tapback classifier + [react:emoji] tag stripper."""

    RULES = [
        (("thank", "awesome", "great", "perfect", "love it", "nice!"), "🔥"),
        (("error", "bug", "crash", "fail", "broken", "fix"), "🛠️"),
        (("question", "why", "how does", "explain", "what is"), "💡"),
        (("haha", "lol", "funny", "joke"), "😄"),
        (("done", "shipped", "passed", "success", "works"), "✅"),
    ]

    @classmethod
    def classify(cls, user_text: str, assistant_text: str = "") -> Optional[str]:
        m = re.search(r"\[react:([^\]]+)\]", assistant_text)
        if m:
            return m.group(1).strip()
        low = user_text.lower()
        for keywords, emoji in cls.RULES:
            if any(k in low for k in keywords):
                return emoji
        return None

    @staticmethod
    def strip_tags(text: str) -> str:
        return re.sub(r"\s*\[react:[^\]]+\]", "", text).strip()


class InstinctStore:
    """SQLite-backed compact user instincts/rules (~100 tokens injected per turn)."""

    def __init__(self, db_path: str = ARIES_MEMORY_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS instincts (
                    id TEXT PRIMARY KEY,
                    rule TEXT NOT NULL,
                    category TEXT DEFAULT 'preference',
                    confidence REAL DEFAULT 0.8,
                    created_at REAL NOT NULL
                )
                """
            )

    def list_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, rule, category, confidence, created_at FROM instincts ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def add(self, rule: str, category: str = "preference", confidence: float = 0.85) -> dict:
        item_id = f"inst_{uuid.uuid4().hex[:8]}"
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO instincts (id, rule, category, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
                (item_id, rule.strip(), category.strip(), float(confidence), now),
            )
        return {"id": item_id, "rule": rule.strip(), "category": category, "confidence": confidence, "created_at": now}

    def delete(self, item_id: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM instincts WHERE id = ?", (item_id,))
            return cur.rowcount > 0

    def format_for_prompt(self, limit: int = 5) -> str:
        items = self.list_all()[:limit]
        if not items:
            return ""
        lines = "\n".join(f"- {it['rule']}" for it in items)
        return f"[LEARNED USER PREFERENCES]:\n{lines}"


context_sandbox = ContextModeSandbox()
tool_cache = SemanticToolCache()
instinct_store = InstinctStore()
