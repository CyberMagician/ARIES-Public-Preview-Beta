"""Local OpenAI & Anthropic API Bridge + Universal Benchmark Observer for ARIES Public Preview Beta.

Exposes /v1/models, /v1/chat/completions, /v1/embeddings, and /v1/messages
proxying directly to the active ARIES backend (llama-server or MLX), while
recording structured per-request observability events for Benchmark Observer Mode.
"""

from collections import deque
import json
import time
import aiohttp
from aiohttp import web

from config import LLAMA_SERVER_URL
from llama_engine import llama_manager
from prompt_settings import prompt_settings

OBSERVER_EVENTS: deque[dict] = deque(maxlen=200)
_OBSERVER_SEQ = 0


def _next_seq() -> int:
    global _OBSERVER_SEQ
    _OBSERVER_SEQ += 1
    return _OBSERVER_SEQ


def _is_allowed(request: web.Request) -> bool:
    if request.remote in ("127.0.0.1", "::1", "localhost"):
        return True
    return bool(prompt_settings.get().get("local_api_bridge_enabled", False))


def _extract_prompt_preview(messages: list) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, list):
                parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"]
                c = " ".join(parts)
            return str(c).strip()
    return ""


async def handle_v1_models(request: web.Request) -> web.Response:
    if not _is_allowed(request):
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(f"{LLAMA_SERVER_URL}/models", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.read()
                return web.Response(body=data, status=resp.status, content_type="application/json")
    except Exception:
        return web.json_response({
            "object": "list",
            "data": [{"id": "local-gemma-4-e2b", "object": "model", "owned_by": "aries-local"}],
        })


async def handle_v1_chat_completions(request: web.Request) -> web.Response:
    if not _is_allowed(request):
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    llama_manager.set_active(True)
    try:
        return await _proxy_chat_completions(request)
    finally:
        llama_manager.set_active(False)


async def _proxy_chat_completions(request: web.Request) -> web.Response:
    body = await request.json()
    seq = _next_seq()
    prompt_text = _extract_prompt_preview(body.get("messages", []))
    model_name = str(body.get("model") or "aries-local").split("/")[-1]
    is_stream = bool(body.get("stream", False))
    t_start = time.perf_counter()

    if is_stream:
        resp_out = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp_out.prepare(request)
        t_first: float | None = None
        thinking_parts: list[str] = []
        answer_parts: list[str] = []
        comp_tokens = 0
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(f"{LLAMA_SERVER_URL}/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                    async for raw_line in resp.content:
                        await resp_out.write(raw_line)
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        try:
                            chunk = json.loads(line[5:].strip())
                            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                            rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            ct = delta.get("content") or ""
                            if (rc or ct) and t_first is None:
                                t_first = time.perf_counter()
                            if rc:
                                thinking_parts.append(rc)
                            if ct:
                                answer_parts.append(ct)
                            if rc or ct:
                                comp_tokens += 1
                            usage = chunk.get("usage")
                            if isinstance(usage, dict) and usage.get("completion_tokens"):
                                comp_tokens = int(usage["completion_tokens"])
                        except Exception:
                            pass
        finally:
            try:
                await resp_out.write_eof()
            except Exception:
                pass
            t_end = time.perf_counter()
            dur_s = max(0.001, t_end - t_start)
            ttft_ms = round(((t_first or t_end) - t_start) * 1000.0, 1)
            tps = round(comp_tokens / max(0.001, (t_end - (t_first or t_start))), 1) if comp_tokens else 0.0
            OBSERVER_EVENTS.append({
                "seq": seq,
                "kind": "chat",
                "model": model_name,
                "prompt": prompt_text,
                "thinking": "".join(thinking_parts).strip(),
                "answer": "".join(answer_parts).strip(),
                "tokens": comp_tokens,
                "ttft_ms": ttft_ms,
                "duration_ms": round(dur_s * 1000.0, 1),
                "tps": tps,
                "timestamp": time.strftime("%H:%M:%S"),
            })
        return resp_out

    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"{LLAMA_SERVER_URL}/chat/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            raw = await resp.read()
            t_end = time.perf_counter()
            try:
                data = json.loads(raw)
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                thinking = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                answer = (msg.get("content") or "").strip()
                usage = data.get("usage") or {}
                comp_tokens = int(usage.get("completion_tokens") or max(1, len(answer.split())))
            except Exception:
                thinking, answer, comp_tokens = "", "", 0
            dur_s = max(0.001, t_end - t_start)
            OBSERVER_EVENTS.append({
                "seq": seq,
                "kind": "chat",
                "model": model_name,
                "prompt": prompt_text,
                "thinking": thinking,
                "answer": answer,
                "tokens": comp_tokens,
                "ttft_ms": round(dur_s * 1000.0, 1),
                "duration_ms": round(dur_s * 1000.0, 1),
                "tps": round(comp_tokens / dur_s, 1) if comp_tokens else 0.0,
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return web.Response(body=raw, status=resp.status, content_type="application/json")


async def handle_v1_embeddings(request: web.Request) -> web.Response:
    if not _is_allowed(request):
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    body = await request.json()
    seq = _next_seq()
    inp = body.get("input", "")
    batch = len(inp) if isinstance(inp, list) else 1
    preview = (inp[0] if isinstance(inp, list) and inp else str(inp))[:160]
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"{LLAMA_SERVER_URL}/embeddings", json=body, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            raw = await resp.read()
            dur_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            dims = 0
            try:
                d = json.loads(raw)
                emb = (d.get("data") or [{}])[0].get("embedding") or []
                dims = len(emb)
            except Exception:
                pass
            OBSERVER_EVENTS.append({
                "seq": seq,
                "kind": "embeddings",
                "model": str(body.get("model") or "embeddings").split("/")[-1],
                "prompt": f"[Batch {batch}] {preview}",
                "thinking": "",
                "answer": f"Embedded {batch} item(s) -> {dims}d vector",
                "tokens": batch,
                "ttft_ms": dur_ms,
                "duration_ms": dur_ms,
                "tps": round(batch / max(0.001, dur_ms / 1000.0), 1),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            return web.Response(body=raw, status=resp.status, content_type="application/json")


async def handle_observer_events(request: web.Request) -> web.Response:
    return web.json_response({"events": list(OBSERVER_EVENTS)})


async def handle_observer_clear(request: web.Request) -> web.Response:
    OBSERVER_EVENTS.clear()
    return web.json_response({"ok": True})


async def handle_v1_messages(request: web.Request) -> web.Response:
    """Minimal Anthropic /v1/messages adapter forwarding to OpenAI-compatible backend."""
    if not _is_allowed(request):
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    req_json = await request.json()
    messages = []
    if req_json.get("system"):
        messages.append({"role": "system", "content": req_json["system"]})
    for m in req_json.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
        messages.append({"role": m.get("role", "user"), "content": content})

    payload = {
        "messages": messages,
        "max_tokens": req_json.get("max_tokens", 1024),
        "temperature": req_json.get("temperature", 0.7),
        "stream": False,
    }
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"{LLAMA_SERVER_URL}/chat/completions", json=payload) as resp:
            data = await resp.json()
    text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return web.json_response({
        "id": data.get("id", "msg_aries"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "local-gemma-4-e2b",
        "stop_reason": "end_turn",
    })
