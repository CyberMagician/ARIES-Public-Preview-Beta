"""Configuration module for ARIES Public Preview Beta for macOS.

Powered by Google Gemma 4 E2B (Embedded Apple Silicon Metal llama.cpp Server + MTP Speculative Decoding)
paired with Parakeet-TDT 0.6B ASR, Phonon 31M TTS, and EmbeddingGemma 2 local RAG.
"""

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "ARIES Public Preview Beta"

# Web Server Port (8776 avoids colliding with other local AI services)
DEFAULT_PORT = int(os.getenv("ARIES_PORT", "8776"))

# Embedded Apple Silicon Metal llama-server Port & Endpoints
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8777"))
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", f"http://127.0.0.1:{LLAMA_SERVER_PORT}/v1")
LM_STUDIO_URL = LLAMA_SERVER_URL

# Embedded EmbeddingGemma 2 Server Port & Endpoints (256d MRL float32 RAG)
EMBEDDING_SERVER_PORT = int(os.getenv("EMBEDDING_SERVER_PORT", "8778"))
EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL", f"http://127.0.0.1:{EMBEDDING_SERVER_PORT}/v1")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ARIES_DATA_DIR = os.path.expanduser("~/Library/Application Support/ARIES Public Preview Beta")
os.makedirs(ARIES_DATA_DIR, exist_ok=True)

ARIES_MEMORY_DB_PATH = os.getenv(
    "ARIES_MEMORY_DB_PATH",
    os.path.join(ARIES_DATA_DIR, "aries_memory.db"),
)

LLAMA_ENGINE_DIR = os.path.join(APP_DIR, "engine", "llama-metal")
_BUNDLED_SERVER_EXE = os.path.join(LLAMA_ENGINE_DIR, "llama-server")
LLAMA_SERVER_EXE = (
    os.getenv("LLAMA_SERVER_EXE")
    or (_BUNDLED_SERVER_EXE if os.path.exists(_BUNDLED_SERVER_EXE) else None)
    or shutil.which("llama-server")
    or _BUNDLED_SERVER_EXE
)
GEMMA4_JINJA_TEMPLATE = os.path.join(LLAMA_ENGINE_DIR, "gemma-4-chat-template.jinja")


def resolve_model_paths() -> dict[str, str]:
    """Dynamically resolves all local model paths so newly downloaded/linked models are detected immediately."""
    models_root = os.path.join(ARIES_DATA_DIR, "models")
    local_models = os.path.join(APP_DIR, "models")
    if not os.path.exists(models_root) and os.path.exists(local_models) and os.listdir(local_models):
        models_root = local_models

    def _first_existing(env_var: str, rel_candidates: list[str]) -> str:
        env_val = os.getenv(env_var)
        if env_val and os.path.exists(env_val):
            return env_val
        for rel in rel_candidates:
            p = os.path.join(models_root, rel)
            if os.path.exists(p):
                return p
        return os.path.join(models_root, rel_candidates[0])

    return {
        "gemma4_llm": _first_existing(
            "GEMMA4_E2B_GGUF_PATH",
            [
                os.path.join("gemma-4", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
                os.path.join("gemma-4", "gemma-4-E2B_q4_0-it.gguf"),
            ],
        ),
        "gemma4_vision": _first_existing(
            "GEMMA4_MMPROJ_GGUF_PATH",
            [
                os.path.join("gemma-4", "mmproj-BF16.gguf"),
                os.path.join("gemma-4", "mmproj-gemma-4-E2B-it-Q8_0.gguf"),
            ],
        ),
        "gemma4_mtp": _first_existing(
            "GEMMA4_MTP_DRAFT_GGUF_PATH",
            [os.path.join("gemma-4", "mtp-gemma-4-E2B-it-Q4_0.gguf")],
        ),
        "embeddinggemma2": _first_existing(
            "EMBEDDING_GEMMA2_GGUF_PATH",
            [os.path.join("embeddinggemma-2", "embeddinggemma-2-UD-Q4_K_XL.gguf")],
        ),
        "parakeet_dir": os.path.join(models_root, "parakeet_06b"),
        "phonon_tts": _first_existing(
            "PHONON_TTS_PATH",
            [os.path.join("phonon", "phonon_int8.litertlm")],
        ),
    }


_INITIAL_PATHS = resolve_model_paths()
GEMMA4_E2B_GGUF_PATH = _INITIAL_PATHS["gemma4_llm"]
GEMMA4_MMPROJ_GGUF_PATH = _INITIAL_PATHS["gemma4_vision"]
GEMMA4_MTP_DRAFT_GGUF_PATH = _INITIAL_PATHS["gemma4_mtp"]
EMBEDDING_GEMMA2_GGUF_PATH = _INITIAL_PATHS["embeddinggemma2"]

# Edge Voice Profile: 100% Apache 2.0 Open-Weight Model (Phonon 31M Freya)
AVAILABLE_VOICES = [
    {
        "name": "Phonon-Freya",
        "id": "phonon_freya",
        "engine": "phonon",
        "style": "Phonon 31M Freya (en-US - Apache 2.0)",
        "gender": "Female",
    },
]
DEFAULT_VOICE = "phonon_freya"

DEFAULT_MODEL = "local-gemma-4-e2b"
AVAILABLE_MODELS = [
    {
        "id": "local-gemma-4-e2b",
        "name": "Gemma 4 E2B IT (Apple Silicon Metal + MTP)",
        "label": "Gemma 4 E2B IT (Apple Silicon Metal + MTP)",
        "limit": 0,
        "is_local": True,
        "description": "Running locally on embedded Apple Silicon Metal llama-server with MTP speculative decoding.",
    },
]

SIMPLE_SYSTEM_INSTRUCTION = (
    "You are ARIES, a fast, intelligent AI assistant running locally on macOS. "
    "Be clear, concise, and helpful."
)

FULL_SYSTEM_INSTRUCTION = """You are ARIES, an ultra-fast, intelligent, and natural AI assistant running locally on Apple Silicon.
When Voice mode is active, speak clearly, concisely, and conversationally in 1 to 2 spoken sentences without markdown.
When Chat mode is active, format code with fenced language blocks, math with LaTeX ($...$ / $$...$$), and diagrams with Mermaid when helpful.
"""

SYSTEM_INSTRUCTION = SIMPLE_SYSTEM_INSTRUCTION
VISION_SYSTEM_INSTRUCTION = SYSTEM_INSTRUCTION
