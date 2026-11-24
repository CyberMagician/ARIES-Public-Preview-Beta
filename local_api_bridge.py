"""Optional Local OpenAI & Anthropic API Bridge for ARIES Public Preview Beta.

When enabled in Settings, exposes /v1/models, /v1/chat/completions, and /v1/messages
proxying directly to the already-running embedded Apple Silicon Metal llama-server.
"""

import json
import aiohttp
from aiohttp import web

from config import LLAMA_SERVER_URL
from prompt_settings import prompt_settings


def _is_enabled() -> bool:
    return bool(prompt_settings.get().get("local_api_bridge_enabled", False))


async def handle_v1_models(request: web.Request) -> web.Response:
    if not _is_enabled():
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    return web.json_response({
        "object": "list",
        "data": [{"id": "local-gemma-4-e2b", "object": "model", "owned_by": "aries-local"}],
    })


async def handle_v1_chat_completions(request: web.Request) -> web.Response:
    if not _is_enabled():
        return web.json_response({"error": "Local API Bridge is disabled in ARIES Settings."}, status=403)
    body = await request.json()
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"{LLAMA_SERVER_URL}/chat/completions", json=body) as resp:
            data = await resp.read()
            return web.Response(body=data, status=resp.status, content_type="application/json")


async def handle_v1_messages(request: web.Request) -> web.Response:
    """Minimal Anthropic /v1/messages adapter forwarding to OpenAI-compatible llama-server."""
    if not _is_enabled():
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
