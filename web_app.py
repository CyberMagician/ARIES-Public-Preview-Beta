"""Unified Desktop App & WebSocket Bridge for Gemma Live Voice Agent.

Architecture: 100% Self-Contained Open-Source Google AI Edge Pipeline
- Consolidated ComfyUI-style Python Launcher + Voice Chat in a single frameless window
- True Rounded Apple macOS Corners (Win32 CreateRoundRectRgn + CSS Glass Rim)
- Live Integrated Server Console Drawer streaming [CPU], [LLM], [ASR], [TTS], [WARMUP], and [LATENCY] logs
- ASR: NVIDIA Parakeet TDT (0.6B / 110M FastConformer) via Sherpa-ONNX
- LLM: Bundled Vulkan llama-server.exe (Gemma 4 E2B IT QAT + MTP Speculative Decoding on Radeon Pro 570 GPU)
- TTS: Google AI Edge Phonon 31M (Freya) & Kokoro 82M (Lewis) (.litertlm, 24kHz Apache 2.0)
"""

import asyncio
from collections import deque
import ctypes
from datetime import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import psutil
from aiohttp import web, ClientSession, ClientTimeout, TCPConnector
from dotenv import load_dotenv
import numpy as np

if sys.platform == "win32":
    from ctypes import wintypes

# Ensure WebView2 auto-grants media stream (microphone) and audio autoplay without popups
os.environ.setdefault(
    "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
    "--autoplay-policy=no-user-gesture-required --use-fake-ui-for-media-stream --disable-features=Translate",
)

from config import (
    APP_DIR,
    ARIES_DATA_DIR,
    DEFAULT_MODEL,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    AVAILABLE_MODELS,
    DEFAULT_PORT,
    LM_STUDIO_URL,
    SYSTEM_INSTRUCTION,
    VISION_SYSTEM_INSTRUCTION,
)
from llama_engine import llama_manager
from rag_memory import memory_store, detect_query_task_type
from tools import get_openai_tools, execute_tool_async
from tracker import UsageTracker

load_dotenv()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

tracker = UsageTracker()

# -------------------------------------------------------------------------
# ComfyUI-Style Live Console Log Tee & WebSocket Broadcaster
# -------------------------------------------------------------------------
CONSOLE_LOGS = deque(maxlen=400)
ACTIVE_WEBSOCKETS: set[web.WebSocketResponse] = set()
MAIN_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
ENGINES_READY = False
ENGINE_STATUS_TEXT = "Initializing ARIES engines..."


class ConsoleTeeStream:
    """Intercepts stdout/stderr prints, timestamps them, and broadcasts live to the unified UI console."""

    def __init__(self, original_stream):
        self.original_stream = original_stream
        self._line_buf = ""

    def write(self, data: str):
        if self.original_stream:
            try:
                self.original_stream.write(data)
                self.original_stream.flush()
            except Exception:
                pass
        if not data:
            return
        self._line_buf += data
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            cleaned = line.rstrip("\r")
            if cleaned.strip():
                self._emit_line(cleaned)

    def _emit_line(self, line: str):
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        entry = f"[{ts}] {line}"
        CONSOLE_LOGS.append(entry)
        if MAIN_EVENT_LOOP and ACTIVE_WEBSOCKETS:
            MAIN_EVENT_LOOP.call_soon_threadsafe(
                lambda msg=entry: asyncio.create_task(_broadcast_console_line(msg))
            )

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


async def _broadcast_console_line(entry: str):
    payload = {
        "type": "console_log",
        "line": entry,
        "ready": ENGINES_READY,
        "status_text": ENGINE_STATUS_TEXT,
    }
    for ws in list(ACTIVE_WEBSOCKETS):
        if not ws.closed:
            try:
                await ws.send_json(payload)
            except Exception:
                pass


sys.stdout = ConsoleTeeStream(sys.__stdout__)
sys.stderr = ConsoleTeeStream(sys.__stderr__)

# -------------------------------------------------------------------------
# Set CPU Process Affinity, Priority & 1ms Multimedia Timer Resolution (Windows)
# -------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        kernel32 = ctypes.windll.kernel32
        winmm = ctypes.windll.winmm
        current_proc = kernel32.GetCurrentProcess()
        # 0x00008000 = HIGH_PRIORITY_CLASS
        kernel32.SetPriorityClass(current_proc, 0x00008000)
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        kernel32.SetProcessAffinityMask(current_proc, 15)
        winmm.timeBeginPeriod(1)
        print("[CPU] Pinned affinity (0b1111), HIGH priority, and 1ms timer resolution active!")
    except Exception as e:
        print(f"[CPU] Process scheduling note: {e}")

# -------------------------------------------------------------------------
# Global AI Engine Handles (Initialized asynchronously so window opens instantly)
# -------------------------------------------------------------------------
parakeet_recognizer = None
phonon_tts = None
mp_tts = None
_kokoro_lock = threading.Lock()


def _resolve_model_root() -> str:
    data_models = os.path.join(ARIES_DATA_DIR, "models")
    if os.path.exists(data_models):
        return data_models
    return os.path.join(APP_DIR, "models")


def init_ai_engines():
    """Loads embedded Vulkan llama-server, Parakeet TDT ASR, Phonon 31M, and Kokoro 82M in the background."""
    global parakeet_recognizer, phonon_tts, mp_tts, ENGINES_READY, ENGINE_STATUS_TEXT

    try:
        t_boot = time.perf_counter()
        print("[LAUNCHER] Starting Self-Contained Google AI Edge + Vulkan LLM Pipeline initialization...")

        # Launch embedded Vulkan llama-server & EmbeddingGemma 2 RAG concurrently while ASR & TTS load on CPU
        def _boot_vulkan_llm():
            if llama_manager.start():
                llama_manager.warmup_prompt_cache()

        llm_thread = threading.Thread(target=_boot_vulkan_llm, daemon=True)
        llm_thread.start()

        rag_thread = threading.Thread(target=memory_store.warmup_and_seed, daemon=True)
        rag_thread.start()

        ENGINE_STATUS_TEXT = "Loading ARIES Speech & Neural Engines..."

        import sherpa_onnx
        from mediapipe.tasks.python.audio import text_to_speech
        from mediapipe.tasks.python.core import base_options

        models_root = _resolve_model_root()
        parakeet_06b_dir = os.path.join(models_root, "parakeet_06b")
        parakeet_110m_dir = os.path.join(models_root, "parakeet_110m")
        model_dir = parakeet_06b_dir if os.path.exists(parakeet_06b_dir) else parakeet_110m_dir

        print(f"[ASR] Initializing NVIDIA Parakeet TDT from {model_dir}...")
        try:
            parakeet_recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=os.path.join(model_dir, "encoder.int8.onnx"),
                decoder=os.path.join(model_dir, "decoder.int8.onnx"),
                joiner=os.path.join(model_dir, "joiner.int8.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=6 if sys.platform == "darwin" else 4,
                sample_rate=16000,
                feature_dim=80,
                model_type="nemo_transducer",
            )
            print("[ASR] Parakeet TDT 0.6B INT8 loaded successfully!")
        except Exception as asr_err:
            parakeet_recognizer = None
            print(f"[ASR] Parakeet TDT load warning: {asr_err}")

        ENGINE_STATUS_TEXT = "Loading ARIES Voice Synthesizer..."
        print("[TTS] Initializing Google Edge Phonon 31M (INT8, Apache 2.0)...")
        phonon_model_path = os.path.join(models_root, "phonon", "phonon_int8.litertlm")
        if os.path.exists(phonon_model_path):
            try:
                phonon_options = text_to_speech.TextToSpeechOptions(
                    base_options=base_options.BaseOptions(model_asset_path=phonon_model_path),
                    default_voice_id="freya",
                    speaking_rate=1.0,
                )
                phonon_tts = text_to_speech.TextToSpeech.create_from_options(phonon_options)
                print("[TTS] Phonon 31M (Freya, Apache 2.0) loaded successfully at 24000 Hz!")
            except Exception as e:
                print(f"[TTS] Phonon loading notice: {e}")

        mp_tts = None
        print("[TTS] Kokoro-82M configured for on-demand lazy loading (Phonon 31M Freya is active default).")

        ENGINE_STATUS_TEXT = "Warming up ARIES neural execution arenas..."
        t_warm = time.perf_counter()
        if parakeet_recognizer is not None:
            dummy_stream = parakeet_recognizer.create_stream()
            dummy_stream.accept_waveform(16000, np.zeros(3200, dtype=np.float32))
            parakeet_recognizer.decode_stream(dummy_stream)
        if phonon_tts:
            phonon_tts.synthesize("Ready.", voice_id="freya")
        print(f"[WARMUP] Pre-warmed Parakeet TDT & Phonon 31M XNNPACK arenas in {(time.perf_counter() - t_warm)*1000:.0f} ms!")

        # Wait for the concurrent LLM server + prompt warmup to finish
        ENGINE_STATUS_TEXT = "Warming up ARIES reasoning engine..."
        llm_thread.join(timeout=35.0)

        ENGINES_READY = True
        if MAIN_EVENT_LOOP:
            async def _warm_http_pool():
                try:
                    sess = await get_http_session()
                    async with sess.get(f"{ llama_manager.base_url }/health") as r:
                        await r.read()
                except Exception:
                    pass
            MAIN_EVENT_LOOP.call_soon_threadsafe(lambda: asyncio.create_task(_warm_http_pool()))
        total_boot_s = time.perf_counter() - t_boot
        ENGINE_STATUS_TEXT = "Ready - Click microphone or hold Spacebar to speak"
        backend_tag = "Metal" if sys.platform == "darwin" else "Vulkan"
        print(f"[READY] All Embedded Engines (Parakeet + {backend_tag} Gemma 4 + Phonon/Kokoro) online in {total_boot_s:.2f}s!")
    except Exception as exc:
        ENGINE_STATUS_TEXT = f"Engine initialization error: {exc}"
        print(f"[ERROR] Failed to initialize AI engines: {exc}")


# -------------------------------------------------------------------------
# Audio Trimming & Synthesis Helpers
# -------------------------------------------------------------------------
def trim_tts_leading_silence(samples: np.ndarray, sample_rate: int = 24000, threshold: float = 0.0025) -> np.ndarray:
    """Trims only leading digital silence on the first sentence chunk; preserves natural inter-sentence pauses."""
    if len(samples) == 0:
        return samples
    active = np.where(np.abs(samples) > threshold)[0]
    if len(active) == 0:
        return samples
    start_idx = max(0, int(active[0]) - int(sample_rate * 0.008))
    return samples[start_idx:]


def transcribe_pcm16(audio_bytes: bytes) -> str:
    """Transcribes 16-bit 16kHz mono PCM audio bytes using Parakeet TDT with silence trimming."""
    if not audio_bytes or parakeet_recognizer is None:
        return ""
    int16_data = np.frombuffer(audio_bytes, dtype=np.int16)
    float32_data = int16_data.astype(np.float32) / 32768.0
    if len(float32_data) < 1600:
        return ""

    active = np.where(np.abs(float32_data) > 0.008)[0]
    if len(active) == 0:
        return ""
    start_idx = max(0, int(active[0]) - 960)
    end_idx = min(len(float32_data), int(active[-1]) + 960)
    trimmed_audio = float32_data[start_idx:end_idx]
    if len(trimmed_audio) < 800:
        return ""

    stream = parakeet_recognizer.create_stream()
    stream.accept_waveform(16000, trimmed_audio)
    parakeet_recognizer.decode_stream(stream)
    return stream.result.text.strip()


_RE_LONG_INLINE_CODE = re.compile(r'`[^`\n]{40,}`')
_RE_MARKDOWN = re.compile(r'[*_~`#]')
_RE_URL = re.compile(r'https?://\S+')
_RE_EMOJI = re.compile(r'[\U00010000-\U0010ffff]')
_RE_SPACES = re.compile(r'\s+')
_RE_UNUSED_TOKEN = re.compile(r"<unused\d+>")


def strip_unused_tokens(text: str) -> str:
    return _RE_UNUSED_TOKEN.sub("", text) if "<unused" in text else text


def clean_for_tts(text: str) -> str:
    """Cleans text for natural spoken TTS: removes long inline code, markdown, URLs, emojis, and collapses spaces."""
    if not text:
        return ""
    text = _RE_LONG_INLINE_CODE.sub('', text)
    text = _RE_MARKDOWN.sub('', text)
    text = _RE_URL.sub('', text)
    text = _RE_EMOJI.sub('', text)
    text = re.sub(r'\bARIES\b', 'Aries', text)
    text = _RE_SPACES.sub(' ', text)
    return text.strip()


def _get_or_load_kokoro_tts():
    """Lazily loads Kokoro-82M only if a Kokoro/Lewis voice is explicitly requested."""
    global mp_tts
    if mp_tts is not None:
        return mp_tts
    with _kokoro_lock:
        if mp_tts is not None:
            return mp_tts
        kokoro_model_path = os.path.join(_resolve_model_root(), "kokoro", "kokoro_int8_fp16.litertlm")
        if os.path.exists(kokoro_model_path):
            try:
                from mediapipe.tasks.python.audio import text_to_speech
                from mediapipe.tasks.python.core import base_options
                print("[TTS] Lazy-loading Google MediaPipe Kokoro-82M (Lewis, Apache 2.0)...")
                mp_tts_options = text_to_speech.TextToSpeechOptions(
                    base_options=base_options.BaseOptions(model_asset_path=kokoro_model_path),
                    default_voice_id="bm_lewis",
                    speaking_rate=1.0,
                )
                mp_tts = text_to_speech.TextToSpeech.create_from_options(mp_tts_options)
                print("[TTS] Kokoro-82M (Lewis, Apache 2.0) loaded successfully at 24000 Hz!")
            except Exception as e:
                print(f"[TTS] MediaPipe lazy-load notice: {e}")
    return mp_tts


def synthesize_speech_pcm(text: str, voice: str = "phonon_freya", is_first_chunk: bool = False) -> bytes:
    """Synthesizes text into 24kHz 16-bit PCM bytes using Phonon 31M (Freya) or lazy-loaded Kokoro 82M (Lewis)."""
    clean_text = clean_for_tts(text)
    if not clean_text:
        return b""

    # 1. Kokoro 82M Lewis (bm_lewis, loaded on demand only)
    if "lewis" in voice.lower() or "kokoro" in voice.lower() or voice.startswith("bm_"):
        kokoro = _get_or_load_kokoro_tts()
        if kokoro:
            try:
                res = kokoro.synthesize(clean_text, voice_id="bm_lewis")
                samples = trim_tts_leading_silence(res.samples, sample_rate=24000) if is_first_chunk else res.samples
                pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
                return pcm16.tobytes()
            except Exception as e:
                print(f"[Kokoro Lewis Synthesize Error]: {e}")

    # 2. Default: Phonon 31M Freya (en-US, Apache 2.0, 4.13x real-time)
    if phonon_tts:
        try:
            res = phonon_tts.synthesize(clean_text, voice_id="freya")
            samples = trim_tts_leading_silence(res.samples, sample_rate=24000) if is_first_chunk else res.samples
            pcm16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
            return pcm16.tobytes()
        except Exception as e:
            print(f"[Phonon Freya Synthesize Error]: {e}")

    return b""


# -------------------------------------------------------------------------
# Fast-Path Chit-Chat Bypass (Skips EmbeddingGemma 2 RAG on Greetings / Filler)
# -------------------------------------------------------------------------
CHITCHAT_PATTERN = re.compile(
    r"^(hi|hello|hey|howdy|yo|greetings|good\s+(morning|afternoon|evening|night|day)|"
    r"how\s+are\s+you|how\s+are\s+you\s+doing|how's\s+it\s+going|what's\s+up|"
    r"who\s+are\s+you|what\s+is\s+your\s+name|"
    r"thanks|thank\s+you|thx|cool|great|awesome|ok|okay|sure|fine|nice|"
    r"bye|goodbye|see\s+ya|catch\s+you\s+later|"
    r"yes|no|yep|nope)$",
    re.IGNORECASE,
)


EXPLICIT_MEMORY_QUERY_PATTERN = re.compile(
    r"\b(remember|memory|memories|knowledge|saved\s+note|notes|wifi|wi-fi|password|vpn|credential|aries\s+architecture)\b",
    re.IGNORECASE,
)

# -------------------------------------------------------------------------
# Smart Tool-Schema Gating (Skips 1,200-Token Tool Payload on Pure Chat Turns)
# -------------------------------------------------------------------------
TOOL_TRIGGER_PATTERN = re.compile(
    r'('
    r'\b(?:open|launch)\s+(?:the\s+)?(?:app|application|spotify|calculator|calc|notepad|notes|paint|preview|task\s*manager|activity\s*monitor|explorer|finder|files|terminal|powershell|cmd|safari|chrome|edge|browser)\b|'
    r'\b(?:open|go\s+to|visit|browse)\s+(?:https?://|\S+\.(?:com|org|net|io|dev|edu|gov)\b|(?:google|youtube|github)\b)|https?://\S+|'
    r'\b(?:search\s+(?:the\s+)?web|search\s+online|look\s+up\s+online|latest\s+news|current\s+events|stock\s+price)\b|'
    r'\b(?:weather|forecast|temperature\s+(?:in|outside|today)|is\s+it\s+raining)\b|'
    r'\b(?:system\s+status|mac\s+status|pc\s+status|my\s+(?:mac\'?s\s+)?(?:ram|memory|cpu|disk|battery|storage|status)|how\s+much\s+(?:ram|memory|disk|storage)|battery\s+(?:level|percent|status)|what\s+time\s+is\s+it|today\'?s\s+date|what\s+day\s+is\s+(?:it|today))\b|'
    r'\b(?:create\s+(?:a\s+)?calendar|schedule\s+(?:a\s+)?(?:meeting|event|appointment)|set\s+(?:a\s+)?(?:timer|reminder|alarm)|remind\s+me\s+(?:in|to)|wake\s+me\s+up)\b|'
    r'\b(?:save\s+(?:a\s+)?(?:note|memory)|remember\s+that|search\s+(?:my\s+)?(?:knowledge|notes|memory))\b'
    r')',
    re.IGNORECASE,
)


def should_include_tools(prompt_text: str) -> bool:
    if not prompt_text:
        return False
    return bool(TOOL_TRIGGER_PATTERN.search(prompt_text))


http_session: ClientSession | None = None
LLM_STREAM_TIMEOUT = ClientTimeout(total=None, connect=10.0, sock_connect=10.0, sock_read=120.0)


async def get_http_session() -> ClientSession:
    global http_session
    if http_session is None or http_session.closed:
        connector = TCPConnector(limit=10, keepalive_timeout=300, force_close=False)
        http_session = ClientSession(connector=connector)
    return http_session


# -------------------------------------------------------------------------
# Strict Sentence-Level Chunker (Ends Chunks Exclusively at Periods / Sentence Endings)
# -------------------------------------------------------------------------
ABBREVIATIONS = (
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "d.c."
)


class StrictSentenceChunker:
    """Strict eager sentence chunker that skips fenced code blocks (```...```) so TTS doesn't read code aloud."""

    def __init__(self):
        self.buffer = ""
        self.in_fence = False
        self._pending_ticks = ""

    def _extract_spoken_text(self, token: str, flush: bool = False) -> str:
        raw = self._pending_ticks + token
        self._pending_ticks = ""
        if not flush and raw.endswith("``") and not raw.endswith("```"):
            self._pending_ticks = "``"
            raw = raw[:-2]
        elif not flush and raw.endswith("`") and not raw.endswith("``"):
            self._pending_ticks = "`"
            raw = raw[:-1]

        parts = raw.split("```")
        spoken_parts = []
        for i, part in enumerate(parts):
            if not self.in_fence:
                spoken_parts.append(part)
            if i < len(parts) - 1:
                self.in_fence = not self.in_fence
        return "".join(spoken_parts)

    def push(self, token: str):
        spoken = self._extract_spoken_text(token, flush=False)
        if not spoken:
            return []
        self.buffer += spoken
        sentences = []

        while True:
            match = re.search(r'([.!?]+|\n+)(\s+|$)', self.buffer)
            if not match:
                break

            punct = match.group(1)
            p_start = match.start()
            if punct == "." and p_start > 0 and p_start < len(self.buffer) - 1:
                if self.buffer[p_start - 1].isdigit() and self.buffer[p_start + 1].isdigit():
                    break

            end_pos = match.end()
            candidate = self.buffer[:match.start() + len(punct)].strip()

            # If punctuation is at end-of-buffer without trailing space yet, require >= 2 words to avoid initials
            if match.group(2) == "":
                words = candidate.lower().split()
                if len(words) < 2 or words[-1] in ABBREVIATIONS:
                    break
            else:
                words = candidate.lower().split()
                if words and words[-1] in ABBREVIATIONS:
                    break

            chunk = self.buffer[:end_pos]
            self.buffer = self.buffer[end_pos:]
            cleaned = clean_for_tts(chunk)
            if cleaned:
                sentences.append(cleaned)

        return sentences

    def flush(self):
        spoken_tail = self._extract_spoken_text("", flush=True)
        self.buffer += spoken_tail
        cleaned = clean_for_tts(self.buffer)
        self.buffer = ""
        if cleaned:
            return [cleaned]
        return []


# -------------------------------------------------------------------------
# Concurrent Async TTS Pipeline (Producer-Consumer Audio Queue)
# -------------------------------------------------------------------------
class TTSStreamPipeline:
    """Async producer-consumer pipeline decoupling LLM streaming from TTS audio generation."""

    def __init__(self, ws, voice: str, turn_start_time: float | None = None):
        self.ws = ws
        self.voice = voice
        self.turn_start_time = turn_start_time or time.perf_counter()
        self.first_audio_sent = False
        self.first_tts_ms: int | None = None
        self.total_ttfa_ms: int | None = None
        self._chunk_index = 0
        self.queue = asyncio.Queue()
        self.llm_done_event = asyncio.Event()
        self._cancelled = False
        self.worker_task = asyncio.create_task(self._worker())

    def mark_llm_done(self):
        self.llm_done_event.set()

    async def _worker(self):
        try:
            while not self._cancelled:
                clause = await self.queue.get()
                if clause is None:
                    self.queue.task_done()
                    break
                if not self._cancelled and not self.ws.closed:
                    sent_idx = self._chunk_index
                    is_first = sent_idx == 0
                    self._chunk_index += 1
                    # After Sentence #1 is already playing in the user's speakers, avoid running heavy CPU
                    # XNNPACK TTS synthesis concurrently with llama-server's remaining token generation (up to 450ms)
                    if not is_first and not self.llm_done_event.is_set():
                        try:
                            await asyncio.wait_for(self.llm_done_event.wait(), timeout=0.45)
                        except asyncio.TimeoutError:
                            pass
                    t_tts = time.perf_counter()
                    pcm_bytes = await asyncio.to_thread(synthesize_speech_pcm, clause, self.voice, is_first)
                    tts_ms = (time.perf_counter() - t_tts) * 1000.0
                    if pcm_bytes and not self._cancelled and not self.ws.closed:
                        chunk_payload = {
                            "type": "tts_sentence_chunk",
                            "sentence_index": sent_idx,
                            "text": clause,
                            "duration_s": round(len(pcm_bytes) / 48000.0, 4),
                        }
                        if not self.first_audio_sent:
                            self.first_audio_sent = True
                            total_ttfa_ms = (time.perf_counter() - self.turn_start_time) * 1000.0
                            self.first_tts_ms = round(tts_ms)
                            self.total_ttfa_ms = round(total_ttfa_ms)
                            chunk_payload["tts_ms"] = self.first_tts_ms
                            chunk_payload["ttfa_ms"] = self.total_ttfa_ms
                            print(f"[LATENCY] First TTS Chunk ('{clause}'): {tts_ms:.0f} ms | Total Time-to-First-Audio: {total_ttfa_ms:.0f} ms")
                        await self.ws.send_json(chunk_payload)
                        await self.ws.send_bytes(pcm_bytes)
                self.queue.task_done()
        except asyncio.CancelledError:
            pass

    async def push_clause(self, clause: str):
        if not self._cancelled and clause:
            await self.queue.put(clause)

    async def finish(self):
        if not self._cancelled:
            await self.queue.put(None)
            await self.worker_task

    def cancel(self):
        self._cancelled = True
        self.worker_task.cancel()


# -------------------------------------------------------------------------
# Web Handlers
# -------------------------------------------------------------------------
async def handle_index(request):
    return web.FileResponse(INDEX_PATH)


async def handle_voices(request):
    return web.json_response(AVAILABLE_VOICES)


async def handle_logs(request):
    return web.json_response({
        "logs": list(CONSOLE_LOGS),
        "ready": ENGINES_READY,
        "status_text": ENGINE_STATUS_TEXT,
    })


async def handle_licenses(request):
    licenses_path = os.path.join(os.path.dirname(__file__), "LICENSES.md")
    content = ""
    if os.path.exists(licenses_path):
        try:
            with open(licenses_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            pass
    return web.json_response({
        "title": "ARIES Live for Mac - Open Source Licenses" if sys.platform == "darwin" else "ARIES Live for Windows - Open Source Licenses",
        "markdown": content,
    })


async def handle_models(request):
    local_running = llama_manager.is_running() or await asyncio.to_thread(llama_manager.is_healthy)

    models_copy = []
    for m in AVAILABLE_MODELS:
        m_dict = dict(m)
        if m_dict.get("is_local"):
            m_dict["active"] = local_running
        else:
            m_dict["active"] = True
        models_copy.append(m_dict)

    return web.json_response(models_copy)


async def handle_local_models(request):
    models = llama_manager.list_available_models()
    return web.json_response({
        "active_model": llama_manager.active_model_id,
        "active_engine": llama_manager.active_engine,
        "endpoint": llama_manager.v1_url,
        "models": models,
    })


async def handle_select_model(request):
    try:
        data = await request.json()
        model_id = (data.get("model_id") or "").strip()
        if not model_id:
            return web.json_response({"ok": False, "error": "model_id required"}, status=400)
        ok = await asyncio.to_thread(llama_manager.switch_model, model_id)
        return web.json_response({
            "ok": bool(ok),
            "active_model": llama_manager.active_model_id,
            "active_engine": llama_manager.active_engine,
        })
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)


async def handle_backend_logs(request):
    return web.json_response({
        "endpoint": llama_manager.v1_url,
        "active_model": llama_manager.active_model_id,
        "active_engine": llama_manager.active_engine,
        "running": llama_manager.is_running() or await asyncio.to_thread(llama_manager.is_healthy),
        "logs": list(llama_manager.backend_logs),
    })


def get_system_ram_metrics():
    """Reads system physical memory metrics."""
    if sys.platform != "win32":
        vm = psutil.virtual_memory()
        tot = vm.total / (1024**3)
        avail = vm.available / (1024**3)
        used = tot - avail
        return {
            "percent": int(round(vm.percent)),
            "used_gb": round(used, 1),
            "free_gb": round(avail, 1),
            "total_gb": round(tot, 1),
        }
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        tot = ms.ullTotalPhys / (1024**3)
        avail = ms.ullAvailPhys / (1024**3)
        used = tot - avail
        return {
            "percent": int(ms.dwMemoryLoad),
            "used_gb": round(used, 1),
            "free_gb": round(avail, 1),
            "total_gb": round(tot, 1),
        }
    except Exception:
        return {"percent": 0, "used_gb": 0.0, "free_gb": 0.0, "total_gb": 64.0}


async def handle_ram_status(request):
    return web.json_response(get_system_ram_metrics())


async def handle_audio_devices(request):
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = [
            {"deviceId": str(idx), "label": dev.get("name", f"Mic {idx}")}
            for idx, dev in enumerate(devices)
            if dev.get("max_input_channels", 0) > 0
        ]
        return web.json_response(inputs)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def handle_screenshot(request):
    if sys.platform != "darwin":
        return web.json_response({"error": "Native screenshot only supported on macOS"}, status=400)
    import base64
    import tempfile
    from PIL import Image
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        proc = await asyncio.create_subprocess_exec(
            "screencapture", "-x", "-t", "jpg", tmp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        with Image.open(tmp_path) as img:
            w, h = img.size
            if w > 1024:
                new_h = max(1, int(round(1024 * (h / w))))
                img = img.resize((1024, new_h), Image.Resampling.LANCZOS)
                w, h = img.size
            from io import BytesIO
            buf = BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return web.json_response({
            "data_url": f"data:image/jpeg;base64,{b64}",
            "base64": b64,
            "width": w,
            "height": h,
        })
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


async def handle_status(request):
    model = request.query.get("model", DEFAULT_MODEL)
    return web.json_response(tracker.get_status(model=model))


async def handle_sessions(request):
    if request.method == "GET":
        sessions = await asyncio.to_thread(memory_store.list_sessions)
        return web.json_response(sessions)
    elif request.method == "POST":
        data = await request.json() if request.can_read_body else {}
        title = data.get("title", "New Conversation")
        session = await asyncio.to_thread(memory_store.create_session, title)
        return web.json_response(session)


async def handle_session_detail(request):
    session_id = request.match_info.get("session_id")
    if request.method == "GET":
        msgs = await asyncio.to_thread(memory_store.get_session_messages, session_id)
        return web.json_response({"session_id": session_id, "messages": msgs})
    elif request.method == "DELETE":
        deleted = await asyncio.to_thread(memory_store.delete_session, session_id)
        return web.json_response({"session_id": session_id, "deleted": deleted})


async def handle_memory(request):
    if request.method == "GET":
        stats = await asyncio.to_thread(memory_store.get_stats)
        chunks = await asyncio.to_thread(memory_store.list_knowledge_chunks, include_chat_memory=True, limit=60)
        return web.json_response({"stats": stats, "chunks": chunks})
    elif request.method == "POST":
        data = await request.json()
        title = data.get("title", "none")
        content = data.get("content", "").strip()
        domain = data.get("domain", "qa")
        if not content:
            return web.json_response({"error": "Content is required"}, status=400)
        ids = await asyncio.to_thread(memory_store.add_knowledge_document, title, content, domain)
        return web.json_response({"status": "indexed", "ids": ids, "title": title})


async def handle_memory_chunk(request):
    chunk_id = int(request.match_info.get("chunk_id", "0"))
    deleted = await asyncio.to_thread(memory_store.delete_rag_chunk, chunk_id)
    return web.json_response({"chunk_id": chunk_id, "deleted": deleted})


async def handle_memory_search(request):
    data = await request.json()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 3))
    results = await asyncio.to_thread(memory_store.search_rag, query, top_k=top_k, min_score=0.5)
    return web.json_response({"query": query, "results": results})


async def handle_profile(request):
    """Get or update the structured User Profile (Memory Summary)."""
    if request.method == "GET":
        profile = await asyncio.to_thread(memory_store.get_user_profile)
        return web.json_response(profile)
    data = await request.json() if request.can_read_body else {}
    sections = data.get("sections") if "sections" in data else None
    enabled = data.get("enabled") if "enabled" in data else None
    full_text = data.get("full_text") if "full_text" in data else None
    updated = await asyncio.to_thread(
        memory_store.save_user_profile,
        sections=sections,
        enabled=enabled,
        full_text=full_text,
    )
    threading.Thread(target=llama_manager.warmup_prompt_cache, daemon=True).start()
    return web.json_response({"status": "ok", "profile": updated, **updated})


async def handle_profile_reset(request):
    """Reset the User Profile (Memory Summary) back to defaults."""
    updated = await asyncio.to_thread(memory_store.reset_user_profile)
    threading.Thread(target=llama_manager.warmup_prompt_cache, daemon=True).start()
    return web.json_response({"status": "ok", "profile": updated, **updated})


async def handle_profile_distill(request):
    """Distill durable user facts from the latest/specified chat session into Learned Notes using local Gemma 4."""
    data = await request.json() if request.can_read_body else {}
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        sessions = await asyncio.to_thread(memory_store.list_sessions, limit=1)
        if sessions:
            session_id = sessions[0]["id"]
    if not session_id:
        return web.json_response({
            "status": "no_sessions",
            "added": 0,
            "profile": await asyncio.to_thread(memory_store.get_user_profile),
        })

    msgs = await asyncio.to_thread(memory_store.get_session_messages, session_id)
    if not msgs:
        return web.json_response({
            "status": "empty_session",
            "added": 0,
            "profile": await asyncio.to_thread(memory_store.get_user_profile),
        })

    transcript = "\n".join(f"{m['role'].upper()}: {m['content'][:800]}" for m in msgs[-16:])
    current = await asyncio.to_thread(memory_store.get_user_profile)
    existing_text = current["full_prompt"]

    extract_prompt = (
        "Extract ONLY new, durable long-term facts, preferences, hardware, or projects revealed by USER in this conversation "
        "that are NOT already covered in the existing User Profile.\n"
        "Return strict JSON: {\"bullets\": [\"short factual sentence in third person ('The user ...')\", ...]} "
        "or {\"bullets\": []} if nothing durable was added.\n\n"
        f"EXISTING PROFILE:\n{existing_text}\n\nRECENT CHAT:\n{transcript}"
    )

    if not llama_manager.is_healthy():
        await asyncio.to_thread(llama_manager.ensure_started)

    session = await get_http_session()
    req_body = {
        "model": "google/gemma-4-e2b-it-qat",
        "messages": [
            {"role": "system", "content": "You maintain a concise, factual user Memory Summary JSON list."},
            {"role": "user", "content": extract_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }
    try:
        async with session.post(f"{LM_STUDIO_URL}/chat/completions", json=req_body, timeout=60) as resp:
            raw_json = await resp.json()
        raw_content = raw_json["choices"][0]["message"]["content"]
        match = re.search(r"\{[\s\S]*\}", raw_content)
        parsed = json.loads(match.group(0)) if match else {}
        bullets = [str(b).strip() for b in parsed.get("bullets", []) if str(b).strip()]
    except Exception as e:
        return web.json_response({"status": "error", "error": str(e), "added": 0, "profile": current}, status=500)

    if not bullets:
        return web.json_response({"status": "no_new_facts", "added": 0, "profile": current})

    sections = list(current["sections"])
    notes_idx = next((i for i, s in enumerate(sections) if s.get("id") == "learned_notes"), None)
    addition = "\n".join(bullets)
    if notes_idx is not None:
        prev = sections[notes_idx].get("content", "").strip()
        sections[notes_idx]["content"] = f"{prev}\n{addition}".strip() if prev else addition
    else:
        sections.append({
            "id": "learned_notes",
            "title": "Learned Preferences & Notes",
            "subtitle": "Durable facts distilled from your local conversations.",
            "content": addition,
        })
    updated = await asyncio.to_thread(memory_store.save_user_profile, sections=sections, enabled=current["enabled"])
    threading.Thread(target=llama_manager.warmup_prompt_cache, daemon=True).start()
    return web.json_response({"status": "updated", "added": len(bullets), "bullets": bullets, "profile": updated})


async def websocket_handler(request):
    """Full-duplex WebSocket bridge: raw PCM audio streaming + Parakeet ASR + Gemma + Streaming TTS + Console Logs."""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    ACTIVE_WEBSOCKETS.add(ws)

    model_name = request.query.get("model", DEFAULT_MODEL)
    voice_name = request.query.get("voice", DEFAULT_VOICE)
    req_session_id = request.query.get("session_id")
    current_session_id = await asyncio.to_thread(memory_store.ensure_session, req_session_id)

    tracker.record_session(model=model_name)
    await ws.send_json({"type": "session_active", "session_id": current_session_id})
    await ws.send_json({"type": "quota_update", "status": tracker.get_status(model=model_name)})
    await ws.send_json({
        "type": "console_init",
        "logs": list(CONSOLE_LOGS),
        "ready": ENGINES_READY,
        "status_text": ENGINE_STATUS_TEXT,
    })

    audio_buffer = bytearray()
    native_mic_stream = None
    native_mic_lock = threading.Lock()
    tools = get_openai_tools()
    active_tts_pipeline = None
    active_generation_task: asyncio.Task | None = None
    tts_muted = False
    last_asr_ms: int | None = None

    def stop_native_mic():
        global _PTT_ACTIVE
        _PTT_ACTIVE = False
        nonlocal native_mic_stream
        with native_mic_lock:
            if native_mic_stream is not None:
                try:
                    native_mic_stream.stop()
                except Exception:
                    pass
                try:
                    native_mic_stream.close()
                except Exception:
                    pass
                native_mic_stream = None

    def cancel_active_tts():
        nonlocal active_tts_pipeline
        if active_tts_pipeline:
            active_tts_pipeline.cancel()
            active_tts_pipeline = None

    def cancel_active_generation():
        nonlocal active_generation_task
        cancel_active_tts()
        if active_generation_task and not active_generation_task.done():
            active_generation_task.cancel()
        active_generation_task = None

    async def run_gemma_and_speak(prompt_text: str, images: list[str] | None = None, turn_start_time: float | None = None):
        nonlocal active_tts_pipeline
        cancel_active_tts()
        t_turn = turn_start_time or time.perf_counter()
        t_llm_start = time.perf_counter()
        first_sentence_logged = False

        # 1. EmbeddingGemma 2 Asymmetric RAG Retrieval (Bypass on visual, chit-chat, action/tool commands, and pure coding turns)
        rag_matches = []
        has_images = bool(images)
        is_visual = has_images or any(
            k in prompt_text.lower() for k in ("this image", "the image", "this picture", "this photo", "on my screen", "attached", "attachment")
        )
        is_chitchat = len(prompt_text.strip()) < 35 and bool(CHITCHAT_PATTERN.match(prompt_text.strip().rstrip(".!?")))
        explicit_memory_query = bool(EXPLICIT_MEMORY_QUERY_PATTERN.search(prompt_text))
        is_action_or_code = (
            should_include_tools(prompt_text) or detect_query_task_type(prompt_text) == "code"
        ) and not explicit_memory_query
        if memory_store.has_rag_chunks() and not is_visual and not is_chitchat and not is_action_or_code:
            rag_matches = await asyncio.to_thread(
                memory_store.search_rag,
                prompt_text,
                top_k=2,
                min_score=0.72,
                exclude_session_id=current_session_id,
                include_chat_memory=explicit_memory_query,
            )

        sampling_temp = 0.0
        grounded_prefix = ""
        if rag_matches:
            rag_context = "\n\n".join(
                f"• [{m['title']} (similarity {m['score']:.2f})]: {m['content']}" for m in rag_matches
            )
            grounded_prefix = (
                f"[RELEVANT GROUNDED KNOWLEDGE (EmbeddingGemma 2 RAG)]:\n"
                f"{rag_context}\n\n"
                f"Instruction: Use the above grounded facts if relevant, answering directly in spoken sentences without markdown in spoken words format.\n\n"
            )
            await ws.send_json({"type": "rag_retrieval", "matches": rag_matches})

        # 2. Multi-turn history following Google's Gemma 4 Multi-Turn History Rule (O(1) in-memory cache hit)
        if current_session_id in memory_store._session_history_cache:
            history_msgs = memory_store.build_gemma4_llm_history(current_session_id, max_messages=24)
        else:
            history_msgs = await asyncio.to_thread(
                memory_store.build_gemma4_llm_history,
                current_session_id,
                max_messages=24,
            )

        # 3. Format user message (supports text and multimodal SigLIP vision attachments)
        user_text = f"{grounded_prefix}{prompt_text}" if grounded_prefix else prompt_text
        if images:
            user_content = [{"type": "text", "text": user_text}]
            for img_b64 in images:
                url = img_b64 if img_b64.startswith("data:") else f"data:image/jpeg;base64,{img_b64}"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
        else:
            user_content = user_text

        # Visual and pure conversational turns omit tool schemas for minimal prompt tokens and max throughput
        # Keep system + User Profile prompt strictly invariant across turns for 100% KV prefix cache reuse
        include_tools = False if has_images else should_include_tools(prompt_text)
        sys_instruction = VISION_SYSTEM_INSTRUCTION if has_images else SYSTEM_INSTRUCTION
        profile_prompt = memory_store.format_user_profile_prompt()
        if profile_prompt:
            sys_instruction = f"{sys_instruction}\n\n{profile_prompt}" if sys_instruction else profile_prompt
        messages = [{"role": "system", "content": sys_instruction}] if sys_instruction else []
        messages.extend(history_msgs)
        messages.append({"role": "user", "content": user_content})
        if llama_manager.active_engine == "mlx":
            req_data = {
                "model": llama_manager.active_model_id or "default",
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.4 if has_images else sampling_temp,
                "stream": True,
            }
        else:
            req_data = {
                "model": "google/gemma-4-e2b-it-qat",
                "messages": messages,
                "max_tokens": -1,
                "temperature": 0.4 if has_images else sampling_temp,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
                "id_slot": 0,
                "cache_prompt": True,
                "timings_per_token": True,
                "stream": True,
            }
        if include_tools:
            req_data["tools"] = tools

        full_agent_text = ""
        tool_call_accum = None
        chunker = StrictSentenceChunker()
        active_tts_pipeline = None if tts_muted else TTSStreamPipeline(ws, voice=voice_name, turn_start_time=t_turn)
        t_first_token: float | None = None
        stream_chunks = 0
        last_timings: dict = {}
        first_sentence_ms: int | None = None

        try:
            # Wake up / ensure llama-server is in VRAM (with idle auto-offload)
            if not llama_manager.is_running():
                if not llama_manager.is_healthy():
                    await ws.send_json({"type": "agent_status", "status": "Waking up ARIES neural engine..."})
                await asyncio.to_thread(llama_manager.ensure_started)

            llama_manager.set_active(True)
            session = await get_http_session()
            t_llm_start = time.perf_counter()
            async with session.post(f"{LM_STUDIO_URL}/chat/completions", json=req_data, timeout=LLM_STREAM_TIMEOUT) as resp:
                if resp.status != 200:
                    err_msg = f"ARIES Engine error: HTTP {resp.status}"
                    await ws.send_json({"type": "agent_transcript", "text": err_msg})
                    cancel_active_tts()
                    return

                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line_str[6:])
                        if chunk.get("timings"):
                            last_timings = chunk["timings"]
                        delta = chunk["choices"][0].get("delta", {})

                        if delta.get("tool_calls"):
                            tc = delta["tool_calls"][0]
                            if not tool_call_accum:
                                tool_call_accum = {
                                    "id": tc.get("id", "call_1"),
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "")
                                }
                            else:
                                if tc.get("function", {}).get("name"):
                                    tool_call_accum["name"] = tc["function"]["name"]
                                tool_call_accum["arguments"] += tc.get("function", {}).get("arguments", "")

                        content = delta.get("content")
                        if content:
                            clean_chunk = strip_unused_tokens(content)
                            if not clean_chunk:
                                continue
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            stream_chunks += 1
                            ttft_ms = round(last_timings.get("prompt_ms") or ((t_first_token - t_llm_start) * 1000.0))
                            elapsed_gen = max(0.015, time.perf_counter() - t_first_token)
                            live_tok_s = round(last_timings.get("predicted_per_second") or (stream_chunks / elapsed_gen), 1)
                            live_tokens = int(last_timings.get("predicted_n") or stream_chunks)
                            full_agent_text += clean_chunk
                            await ws.send_json({
                                "type": "agent_transcript_chunk",
                                "chunk": clean_chunk,
                                "full_text": full_agent_text,
                                "tok_s": live_tok_s,
                                "tokens": live_tokens,
                                "ttft_ms": ttft_ms,
                            })

                            for clause in chunker.push(clean_chunk):
                                if not first_sentence_logged:
                                    first_sentence_logged = True
                                    llm_1st_ms = (time.perf_counter() - t_llm_start) * 1000.0
                                    first_sentence_ms = round(llm_1st_ms)
                                    print(f"[LATENCY] LLM 1st Sentence Ready (tools={include_tools}): {llm_1st_ms:.0f} ms -> '{clause}'")
                                if active_tts_pipeline:
                                    await active_tts_pipeline.push_clause(clause)

                    except Exception:
                        continue

            for clause in chunker.flush():
                if not first_sentence_logged:
                    first_sentence_logged = True
                    llm_1st_ms = (time.perf_counter() - t_llm_start) * 1000.0
                    first_sentence_ms = round(llm_1st_ms)
                    print(f"[LATENCY] LLM 1st Sentence Ready (tools={include_tools}): {llm_1st_ms:.0f} ms -> '{clause}'")
                if active_tts_pipeline:
                    await active_tts_pipeline.push_clause(clause)
            if active_tts_pipeline:
                active_tts_pipeline.mark_llm_done()
                await active_tts_pipeline.finish()

            if tool_call_accum and tool_call_accum.get("name"):
                tool_name = tool_call_accum["name"]
                try:
                    args = json.loads(tool_call_accum["arguments"]) if tool_call_accum["arguments"] else {}
                except Exception:
                    args = {}

                print(f"[TOOL] Executing {tool_name}({args})...")
                await ws.send_json({"type": "tool_call", "name": tool_name, "args": args})
                result = await execute_tool_async(tool_name, args)
                await ws.send_json({"type": "tool_result", "name": tool_name, "result": result})

                follow_up = list(messages)
                follow_up.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call_accum["id"],
                        "type": "function",
                        "function": {"name": tool_name, "arguments": tool_call_accum["arguments"]}
                    }]
                })
                follow_up.append({
                    "role": "tool",
                    "tool_call_id": tool_call_accum["id"],
                    "name": tool_name,
                    "content": json.dumps(result)
                })

                follow_req = {
                    "model": "google/gemma-4-e2b-it-qat",
                    "messages": follow_up,
                    "tools": tools,
                    "max_tokens": -1,
                    "temperature": 0.0,
                    "reasoning_effort": "none",
                    "chat_template_kwargs": {"enable_thinking": False},
                    "id_slot": 0,
                    "cache_prompt": True,
                    "timings_per_token": True,
                    "stream": True,
                }

                full_follow_text = ""
                follow_chunker = StrictSentenceChunker()
                active_tts_pipeline = None if tts_muted else TTSStreamPipeline(ws, voice=voice_name, turn_start_time=time.perf_counter())

                session = await get_http_session()
                async with session.post(f"{LM_STUDIO_URL}/chat/completions", json=follow_req, timeout=LLM_STREAM_TIMEOUT) as resp:
                    async for line in resp.content:
                        line_str = line.decode("utf-8").strip()
                        if not line_str.startswith("data: ") or line_str == "data: [DONE]":
                            continue
                        try:
                            chunk = json.loads(line_str[6:])
                            if chunk.get("timings"):
                                last_timings = chunk["timings"]
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                clean_chunk = strip_unused_tokens(content)
                                if not clean_chunk:
                                    continue
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                stream_chunks += 1
                                ttft_ms = round(last_timings.get("prompt_ms") or ((t_first_token - t_llm_start) * 1000.0))
                                elapsed_gen = max(0.015, time.perf_counter() - t_first_token)
                                live_tok_s = round(last_timings.get("predicted_per_second") or (stream_chunks / elapsed_gen), 1)
                                live_tokens = int(last_timings.get("predicted_n") or stream_chunks)
                                full_follow_text += clean_chunk
                                await ws.send_json({
                                    "type": "agent_transcript_chunk",
                                    "chunk": clean_chunk,
                                    "full_text": full_follow_text,
                                    "tok_s": live_tok_s,
                                    "tokens": live_tokens,
                                    "ttft_ms": ttft_ms,
                                })
                                for clause in follow_chunker.push(clean_chunk):
                                    if active_tts_pipeline:
                                        await active_tts_pipeline.push_clause(clause)
                        except Exception:
                            continue

                for clause in follow_chunker.flush():
                    if active_tts_pipeline:
                        await active_tts_pipeline.push_clause(clause)
                if active_tts_pipeline:
                    active_tts_pipeline.mark_llm_done()
                    await active_tts_pipeline.finish()

                full_agent_text = full_follow_text

            clean_text_for_db = strip_unused_tokens(full_agent_text).strip()
            # Update in-memory history cache immediately and persist to SQLite asynchronously off the hot path
            asyncio.create_task(asyncio.to_thread(
                memory_store.record_turn_and_index_async,
                current_session_id,
                prompt_text,
                clean_text_for_db,
                raw_assistant_text=clean_text_for_db,
                has_tool_call=bool(tool_call_accum),
                tool_name=tool_call_accum.get("name", "") if tool_call_accum else "",
            ))

            status = tracker.record_turn(
                prompt_tokens=len(prompt_text) // 4,
                response_tokens=len(full_agent_text) // 4,
                count_as_request=False,
                model=model_name,
            )
            turn_metrics = {
                "tok_s": round(last_timings.get("predicted_per_second") or (stream_chunks / max(0.015, time.perf_counter() - (t_first_token or t_llm_start))), 1),
                "tokens": int(last_timings.get("predicted_n") or stream_chunks),
                "ttft_ms": round(last_timings.get("prompt_ms") or (((t_first_token or time.perf_counter()) - t_llm_start) * 1000.0)),
                "asr_ms": last_asr_ms,
                "llm_1st_ms": first_sentence_ms,
                "ttfa_ms": active_tts_pipeline.total_ttfa_ms if active_tts_pipeline else None,
                "draft_accepted": int(last_timings.get("draft_n_accepted") or 0),
                "draft_total": int(last_timings.get("draft_n") or 0),
            }
            await ws.send_json({"type": "turn_complete", "session_id": current_session_id, "metrics": turn_metrics})
            await ws.send_json({"type": "quota_update", "status": status})

        except asyncio.CancelledError:
            cancel_active_tts()
            clean_partial = strip_unused_tokens(full_agent_text).strip()
            if clean_partial:
                await asyncio.to_thread(
                    memory_store.record_turn_and_index_async,
                    current_session_id,
                    prompt_text,
                    clean_partial,
                    raw_assistant_text=clean_partial,
                    has_tool_call=bool(tool_call_accum),
                    tool_name=tool_call_accum.get("name", "") if tool_call_accum else "",
                )
            if not ws.closed:
                await ws.send_json({"type": "turn_complete", "stopped": True, "session_id": current_session_id})
            raise

        except Exception as e:
            err = f"ARIES Engine Error: {str(e) or type(e).__name__}"
            print(f"[ERROR] {err}")
            await ws.send_json({"type": "agent_transcript", "text": err})
            await ws.send_json({"type": "turn_complete"})
        finally:
            active_tts_pipeline = None
            llama_manager.set_active(False)

    async def _process_audio_buffer():
        nonlocal active_generation_task, last_asr_ms
        if len(audio_buffer) > 1600:
            t_turn_start = time.perf_counter()
            raw_pcm = bytes(audio_buffer)
            audio_buffer.clear()

            transcription = await asyncio.to_thread(transcribe_pcm16, raw_pcm)
            asr_ms = (time.perf_counter() - t_turn_start) * 1000.0
            print(f"[LATENCY] Parakeet TDT ASR: {asr_ms:.0f} ms -> '{transcription}'")
            if transcription:
                last_asr_ms = round(asr_ms)
                await ws.send_json({"type": "user_transcript", "text": transcription, "asr_ms": last_asr_ms})
                cancel_active_generation()
                active_generation_task = asyncio.create_task(run_gemma_and_speak(transcription, turn_start_time=t_turn_start))
            else:
                await ws.send_json({"type": "no_speech_detected"})
        else:
            audio_buffer.clear()
            await ws.send_json({"type": "no_speech_detected"})

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                audio_buffer.extend(msg.data)

            elif msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")

                    if msg_type == "audio_stream_end":
                        await _process_audio_buffer()

                    elif msg_type == "native_mic_start":
                        stop_native_mic()
                        audio_buffer.clear()
                        try:
                            global _PTT_ACTIVE
                            _PTT_ACTIVE = True
                            import sounddevice as sd
                            dev_raw = str(data.get("device_id", "")).strip()
                            dev_idx = int(dev_raw) if dev_raw.isdigit() else None
                            last_level_ts = [0.0]

                            def _make_cb(src_rate: float):
                                def _cb(indata, frames, time_info, status):
                                    if indata is None or len(indata) == 0:
                                        return
                                    mono = indata[:, 0] if indata.ndim > 1 else indata.reshape(-1)
                                    if int(round(src_rate)) != 16000:
                                        x_old = np.linspace(0.0, 1.0, len(mono), endpoint=False)
                                        n_new = max(1, int(round(len(mono) * 16000.0 / src_rate)))
                                        x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
                                        resampled = np.interp(x_new, x_old, mono.astype(np.float32))
                                        pcm16 = np.clip(resampled, -32768, 32767).astype(np.int16)
                                    else:
                                        pcm16 = mono.astype(np.int16, copy=False)
                                    audio_buffer.extend(pcm16.tobytes())

                                    now = time.perf_counter()
                                    if now - last_level_ts[0] >= 0.06 and MAIN_EVENT_LOOP and not ws.closed:
                                        last_level_ts[0] = now
                                        norm = np.abs(pcm16.astype(np.float32)) / 32768.0
                                        chunks = np.array_split(norm, 8)
                                        levels = [int(min(255, np.sqrt(np.mean(c ** 2)) * 900.0)) if len(c) else 0 for c in chunks]
                                        MAIN_EVENT_LOOP.call_soon_threadsafe(
                                            lambda lv=levels: asyncio.create_task(ws.send_json({"type": "mic_levels", "levels": lv}))
                                        )
                                return _cb

                            try:
                                stream = sd.InputStream(
                                    samplerate=16000,
                                    channels=1,
                                    dtype="int16",
                                    blocksize=1024,
                                    device=dev_idx,
                                    callback=_make_cb(16000.0),
                                )
                                stream.start()
                            except Exception:
                                info = sd.query_devices(dev_idx, "input") if dev_idx is not None else sd.query_devices(kind="input")
                                def_sr = float(info.get("default_samplerate", 48000.0))
                                stream = sd.InputStream(
                                    samplerate=def_sr,
                                    channels=1,
                                    dtype="int16",
                                    blocksize=max(256, int(def_sr * 0.064)),
                                    device=dev_idx,
                                    callback=_make_cb(def_sr),
                                )
                                stream.start()

                            with native_mic_lock:
                                native_mic_stream = stream
                        except Exception as mic_err:
                            stop_native_mic()
                            await ws.send_json({"type": "mic_error", "error": str(mic_err)})

                    elif msg_type == "native_mic_stop":
                        stop_native_mic()
                        await _process_audio_buffer()

                    elif msg_type in ("text_prompt", "multimodal_input"):
                        last_asr_ms = None
                        prompt_text = data.get("text", "").strip()
                        images = data.get("images", [])
                        if not prompt_text and images:
                            prompt_text = "Describe this visual attachment in detail."
                        if prompt_text or images:
                            t_turn_start = time.perf_counter()
                            print(f"[USER] Text/multimodal prompt: '{prompt_text}' (attachments: {len(images)})")
                            cancel_active_generation()
                            active_generation_task = asyncio.create_task(run_gemma_and_speak(prompt_text, images=images, turn_start_time=t_turn_start))

                    elif msg_type in ("interrupt", "stop_response"):
                        stop_native_mic()
                        cancel_active_generation()
                        audio_buffer.clear()

                    elif msg_type in ("set_mute", "mute_audio"):
                        tts_muted = bool(data.get("muted", True))
                        if tts_muted:
                            cancel_active_tts()
                        await ws.send_json({"type": "mute_state", "muted": tts_muted})

                    elif msg_type == "switch_session":
                        target_id = data.get("session_id")
                        if target_id:
                            stop_native_mic()
                            cancel_active_generation()
                            audio_buffer.clear()
                            current_session_id = await asyncio.to_thread(memory_store.ensure_session, target_id)
                            msgs = await asyncio.to_thread(memory_store.get_session_messages, current_session_id)
                            await ws.send_json({
                                "type": "session_loaded",
                                "session_id": current_session_id,
                                "messages": msgs,
                            })

                    elif msg_type in ("new_chat", "clear_context"):
                        cancel_active_generation()
                        audio_buffer.clear()
                        new_s = await asyncio.to_thread(memory_store.create_session, "New Conversation")
                        current_session_id = new_s["id"]
                        print(f"[SESSION] New chat session initiated: {current_session_id}!")
                        await ws.send_json({"type": "session_active", "session_id": current_session_id})
                        await ws.send_json({"type": "context_cleared"})

                except Exception as e:
                    print(f"[WebSocket Error]: {e}")

            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break
    finally:
        stop_native_mic()
        ACTIVE_WEBSOCKETS.discard(ws)
        cancel_active_generation()

    return ws


# -------------------------------------------------------------------------
# Win32 Apple Rounded Window Helper & JS Window Control Bridge
# -------------------------------------------------------------------------
if sys.platform == "win32":
    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    gdi32.CreateRoundRectRgn.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    gdi32.CreateRoundRectRgn.restype = wintypes.HRGN

    user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN, wintypes.BOOL]
    user32.SetWindowRgn.restype = ctypes.c_int

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def apply_apple_rounded_corners(form_hwnd: int, corner_diameter: int = 40):
        """Applies smooth macOS rounded corners to the top-level Form and all full-size WebView2 child HWNDs."""
        if not form_hwnd:
            return
        rect = RECT()
        if not user32.GetWindowRect(form_hwnd, ctypes.byref(rect)):
            return
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return

        # Check if window is maximized (if maximized, clear rounded region so it fills screen cleanly)
        if user32.IsZoomed(form_hwnd):
            user32.SetWindowRgn(form_hwnd, None, True)

            def _clear_child(child_hwnd, _lparam):
                user32.SetWindowRgn(child_hwnd, None, True)
                return True

            user32.EnumChildWindows(form_hwnd, WNDENUMPROC(_clear_child), 0)
            return

        # Scale corner diameter by window DPI for crisp Retina 5K / QHD corners
        try:
            dpi = user32.GetDpiForWindow(form_hwnd)
            scale = dpi / 96.0 if dpi > 0 else 1.0
        except Exception:
            scale = 1.0
        diam = int(corner_diameter * scale)

        hrgn = gdi32.CreateRoundRectRgn(0, 0, w + 1, h + 1, diam, diam)
        user32.SetWindowRgn(form_hwnd, hrgn, True)

        def _round_child(child_hwnd, _lparam):
            crect = RECT()
            if user32.GetWindowRect(child_hwnd, ctypes.byref(crect)):
                cw = crect.right - crect.left
                ch = crect.bottom - crect.top
                if abs(cw - w) <= 4 and abs(ch - h) <= 4:
                    chrgn = gdi32.CreateRoundRectRgn(0, 0, cw + 1, ch + 1, diam, diam)
                    user32.SetWindowRgn(child_hwnd, chrgn, True)
            return True

        user32.EnumChildWindows(form_hwnd, WNDENUMPROC(_round_child), 0)
else:
    def apply_apple_rounded_corners(form_hwnd: int, corner_diameter: int = 40):
        pass


_PTT_ACTIVE = False
_MACOS_KEY_MONITOR = None


class DesktopWindowAPI:
    """JS API exposed as `window.pywebview.api` for macOS Traffic Light controls & native dragging."""

    def __init__(self):
        self._window = None
        self._is_maximized = False

    def set_ptt_active(self, active: bool):
        global _PTT_ACTIVE
        _PTT_ACTIVE = bool(active)

    def attach(self, window):
        self._window = window

    def minimize_window(self):
        if self._window:
            self._window.minimize()

    def toggle_maximize(self):
        if not self._window:
            return
        if sys.platform == "darwin":
            self._window.toggle_fullscreen()
            self._is_maximized = not self._is_maximized
            return self._is_maximized
        if self._is_maximized:
            self._window.restore()
            self._is_maximized = False
        else:
            self._window.maximize()
            self._is_maximized = True
        try:
            hwnd = self._window.native.Handle.ToInt32()
            apply_apple_rounded_corners(hwnd)
        except Exception:
            pass
        return self._is_maximized

    def close_window(self):
        memory_store.embedder.stop()
        llama_manager.stop()
        if self._window:
            self._window.destroy()
        os._exit(0)


async def start_web_server():
    global MAIN_EVENT_LOOP
    MAIN_EVENT_LOOP = asyncio.get_running_loop()

    from model_downloader import ModelDownloadManager
    from prompt_settings import PERSONA_PRESETS, prompt_settings
    from efficiency_engine import instinct_store
    from document_ingestion import doc_ingestor
    from web_and_tools import SandboxedCodeRunner, gatekeeper
    from local_api_bridge import handle_v1_chat_completions, handle_v1_messages, handle_v1_models

    download_mgr = ModelDownloadManager()

    async def handle_setup_status(request):
        return web.json_response(download_mgr.get_status())

    async def handle_setup_start(request):
        def _on_done():
            threading.Thread(target=init_ai_engines, daemon=True).start()
        started = download_mgr.start_download(on_complete=_on_done)
        return web.json_response({"started": started, "status": download_mgr.get_status()})

    async def handle_setup_cancel(request):
        download_mgr.cancel()
        return web.json_response({"cancelled": True, "status": download_mgr.get_status()})

    async def handle_setup_link_local(request):
        body = await request.json()
        res = download_mgr.import_from_directory(body.get("path", ""), symlink=True)
        if res.get("ok") and download_mgr.all_models_ready():
            threading.Thread(target=init_ai_engines, daemon=True).start()
        return web.json_response(res)

    async def handle_settings(request):
        if request.method == "GET":
            return web.json_response(prompt_settings.get())
        body = await request.json()
        return web.json_response(prompt_settings.update(body))

    async def handle_settings_reset(request):
        return web.json_response(prompt_settings.reset())

    async def handle_presets(request):
        return web.json_response({"presets": list(PERSONA_PRESETS.values())})

    async def handle_instincts(request):
        if request.method == "GET":
            return web.json_response({"instincts": instinct_store.list_all()})
        if request.method == "POST":
            body = await request.json()
            item = instinct_store.add(body.get("rule", ""), body.get("category", "preference"))
            return web.json_response(item)
        return web.json_response({"error": "Method not allowed"}, status=405)

    async def handle_instinct_delete(request):
        item_id = request.match_info.get("item_id", "")
        return web.json_response({"deleted": instinct_store.delete(item_id)})

    async def handle_gatekeeper(request):
        if request.method == "GET":
            return web.json_response(gatekeeper.get_state())
        body = await request.json()
        return web.json_response(gatekeeper.update(body.get("rules", body)))

    async def handle_upload(request):
        reader = await request.multipart()
        field = await reader.next()
        if not field:
            return web.json_response({"error": "No file uploaded"}, status=400)
        filename = field.filename or "upload.txt"
        data = await field.read()
        res = doc_ingestor.ingest_bytes(filename, data)
        return web.json_response(res)

    async def handle_sandbox_run(request):
        body = await request.json()
        res = SandboxedCodeRunner.run(body.get("language", "python"), body.get("code", ""))
        return web.json_response(res)

    async def handle_lan_info(request):
        import socket
        ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
        return web.json_response({"ip": ip, "port": DEFAULT_PORT, "url": f"http://{ip}:{DEFAULT_PORT}"})

    app = web.Application(client_max_size=50 * 1024 * 1024)
    app.router.add_get("/", handle_index)
    app.router.add_get("/api/voices", handle_voices)
    app.router.add_get("/api/models", handle_models)
    app.router.add_get("/api/local-models", handle_local_models)
    app.router.add_post("/api/models/select", handle_select_model)
    app.router.add_get("/api/backend-logs", handle_backend_logs)
    app.router.add_get("/api/ram", handle_ram_status)
    app.router.add_get("/api/audio_devices", handle_audio_devices)
    app.router.add_get("/api/screenshot", handle_screenshot)
    app.router.add_get("/api/status", handle_status)
    app.router.add_get("/api/logs", handle_logs)
    app.router.add_get("/api/licenses", handle_licenses)
    app.router.add_get("/api/sessions", handle_sessions)
    app.router.add_post("/api/sessions", handle_sessions)
    app.router.add_get("/api/sessions/{session_id}", handle_session_detail)
    app.router.add_delete("/api/sessions/{session_id}", handle_session_detail)
    app.router.add_get("/api/memory", handle_memory)
    app.router.add_post("/api/memory", handle_memory)
    app.router.add_delete("/api/memory/{chunk_id}", handle_memory_chunk)
    app.router.add_post("/api/memory/search", handle_memory_search)
    app.router.add_get("/api/profile", handle_profile)
    app.router.add_post("/api/profile", handle_profile)
    app.router.add_post("/api/profile/reset", handle_profile_reset)
    app.router.add_post("/api/profile/distill", handle_profile_distill)
    app.router.add_get("/api/setup/status", handle_setup_status)
    app.router.add_post("/api/setup/start", handle_setup_start)
    app.router.add_post("/api/setup/cancel", handle_setup_cancel)
    app.router.add_post("/api/setup/link-local", handle_setup_link_local)
    app.router.add_get("/api/settings", handle_settings)
    app.router.add_post("/api/settings", handle_settings)
    app.router.add_post("/api/settings/reset", handle_settings_reset)
    app.router.add_get("/api/presets", handle_presets)
    app.router.add_get("/api/instincts", handle_instincts)
    app.router.add_post("/api/instincts", handle_instincts)
    app.router.add_delete("/api/instincts/{item_id}", handle_instinct_delete)
    app.router.add_get("/api/gatekeeper", handle_gatekeeper)
    app.router.add_post("/api/gatekeeper", handle_gatekeeper)
    app.router.add_post("/api/upload", handle_upload)
    app.router.add_post("/api/sandbox/run", handle_sandbox_run)
    app.router.add_get("/api/lan-info", handle_lan_info)
    app.router.add_get("/v1/models", handle_v1_models)
    app.router.add_post("/v1/chat/completions", handle_v1_chat_completions)
    app.router.add_post("/v1/messages", handle_v1_messages)
    app.router.add_get("/ws", websocket_handler)
    app.router.add_static("/static", STATIC_DIR)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", DEFAULT_PORT)
    await site.start()

    print("======================================================================")
    if sys.platform == "darwin":
        print("  ARIES LIVE FOR macOS — SELF-CONTAINED METAL + AI EDGE LAUNCHER")
    else:
        print("  ARIES LIVE FOR WINDOWS — SELF-CONTAINED VULKAN + AI EDGE LAUNCHER")
    print("======================================================================")
    print(f"  Server URL : http://127.0.0.1:{DEFAULT_PORT}")
    print("  ASR Engine : NVIDIA Parakeet TDT 0.6B FastConformer (INT8)")
    if sys.platform == "darwin":
        print("  LLM Brain  : Bundled Metal llama-server (Gemma 4 E2B + MTP Draft)")
    else:
        print("  LLM Brain  : Bundled Vulkan llama-server (Gemma 4 E2B + MTP Draft)")
    print("  TTS Engine : Google Edge Phonon 31M (Freya) & Kokoro 82M (Lewis)")
    print("======================================================================")

    while True:
        await asyncio.sleep(3600)


def _run_server_in_background():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_web_server())


def launch_unified_desktop_app():
    """Starts the aiohttp backend + AI engine loader in background threads and opens the unified Apple-rounded window."""
    server_thread = threading.Thread(target=_run_server_in_background, daemon=True)
    server_thread.start()

    # Wait briefly for port 8766 to accept connections (<100ms)
    for _ in range(30):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{DEFAULT_PORT}/api/voices", timeout=0.2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(0.04)

    # Start loading Parakeet TDT + Phonon 31M + Kokoro 82M in background so window opens immediately
    engine_thread = threading.Thread(target=init_ai_engines, daemon=True)
    engine_thread.start()

    try:
        import webview

        api = DesktopWindowAPI()
        window = webview.create_window(
            title="Aries Live Beta" if sys.platform == "darwin" else "ARIES Live for Windows",
            url=f"http://127.0.0.1:{DEFAULT_PORT}",
            width=600,
            height=890,
            min_size=(480, 680),
            frameless=True,
            easy_drag=False,
            shadow=True,
            background_color="#04130f",
            js_api=api,
        )
        api.attach(window)

        def on_shown():
            if sys.platform != "win32":
                if sys.platform == "darwin":
                    global _MACOS_KEY_MONITOR
                    try:
                        import AppKit
                        from Foundation import NSBundle

                        bundle = NSBundle.mainBundle()
                        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                        if info:
                            info["CFBundleName"] = "Aries Live Beta"
                        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "icon.png")
                        if os.path.exists(icon_path):
                            ns_img = AppKit.NSImage.alloc().initWithContentsOfFile_(icon_path)
                            if ns_img:
                                AppKit.NSApplication.sharedApplication().setApplicationIconImage_(ns_img)

                        def _on_macos_key_down(event):
                            try:
                                if _PTT_ACTIVE and event.keyCode() == 49 and event.isARepeat():
                                    return None
                            except Exception:
                                pass
                            return event

                        _MACOS_KEY_MONITOR = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                            AppKit.NSEventMaskKeyDown, _on_macos_key_down
                        )
                        print("[WINDOW] macOS PTT Spacebar-repeat suppressor active.")
                    except Exception as e:
                        print(f"[WINDOW] macOS key monitor note: {e}")
                print("[WINDOW] Unified desktop window active (native macOS WKWebView).")
                return
            try:
                from Microsoft.Web.WebView2.Core import CoreWebView2PermissionState
                form = window.native
                hwnd = form.Handle.ToInt32()
                apply_apple_rounded_corners(hwnd, corner_diameter=40)

                def _on_form_resize(sender, args):
                    try:
                        apply_apple_rounded_corners(form.Handle.ToInt32(), corner_diameter=40)
                    except Exception:
                        pass

                form.Resize += _on_form_resize

                # Auto-grant microphone / camera / clipboard permissions in WebView2
                def _attach_permissions(sender, args):
                    try:
                        def _on_perm(s, perm_args):
                            perm_args.State = CoreWebView2PermissionState.Allow
                        sender.CoreWebView2.PermissionRequested += _on_perm
                    except Exception:
                        pass

                if form.webview and form.webview.CoreWebView2:
                    def _on_perm(s, perm_args):
                        perm_args.State = CoreWebView2PermissionState.Allow
                    form.webview.CoreWebView2.PermissionRequested += _on_perm
                elif form.webview:
                    form.webview.CoreWebView2InitializationCompleted += _attach_permissions
                print("[WINDOW] Unified Apple-rounded desktop window active (frameless WebView2).")
            except Exception as e:
                print(f"[WINDOW] Setup note: {e}")

        def on_loaded():
            try:
                hwnd = window.native.Handle.ToInt32()
                apply_apple_rounded_corners(hwnd, corner_diameter=40)
            except Exception:
                pass

        def on_closed():
            memory_store.embedder.stop()
            llama_manager.stop()
            os._exit(0)

        window.events.shown += on_shown
        window.events.loaded += on_loaded
        window.events.closed += on_closed

        if sys.platform == "win32":
            webview.start(gui="edgechromium", debug=False)
        else:
            webview.start(debug=False)
    except Exception as e:
        print(f"[WINDOW] Fallback to Edge app mode ({e})...")
        webbrowser.open(f"http://127.0.0.1:{DEFAULT_PORT}")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    try:
        if "--no-window" in sys.argv or "--headless" in sys.argv:
            threading.Thread(target=init_ai_engines, daemon=True).start()
            asyncio.run(start_web_server())
        else:
            launch_unified_desktop_app()
    except Exception as e:
        crash_log = os.path.join(APP_DIR, "crash.log")
        with open(crash_log, "w", encoding="utf-8") as f:
            import traceback
            traceback.print_exc(file=f)
        raise
