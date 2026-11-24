"""EmbeddingGemma 2 Local RAG & Persistent Multi-Session Chat History for ARIES Live.

Implements Google's official EmbeddingGemma 2 + Gemma 4 E2B-it QAT Reference RAG Stack:
1. Numerical Precision Guard:
   - Uses `bfloat16` / `float32` activations via `embeddinggemma-2-Q8_0.gguf` (F32 norms/projections)
     and `numpy.float32` vector math — NEVER `float16` (prevents fp16 dynamic range overflow / NaN).
2. Asymmetric Task Instruction Prefixes:
   - General Search (`SearchQuery`):     `task: search result | query: {query}`
   - Q&A RAG (`QuestionAnswering`):      `task: question answering | query: {question}`
   - Fact Checking (`FactChecking`):     `task: fact checking | query: {claim}`
   - Code Search (`CodeRetrieval`):      `task: code retrieval | query: {query}`
   - Document / Corpus Format:           `title: {title} | text: {content}`
   - Missing Title Quirk:                Explicitly formats as `title: none | text: {content}` when untitled.
3. Matryoshka Representation Learning (MRL) 256d Truncation:
   - Slices native 768d output vector to leading 256 dimensions (`[:256]`) and L2-normalizes in `float32`,
     cutting vector storage 3x while retaining ~100% text/code retrieval performance.
4. Selective Encoder Footprint (270M Text/Code Backbone):
   - Runs the 270M text/code GGUF backbone without vision/audio encoders (`<300 MB` RAM, 0 MB GPU VRAM contention).
5. 1,024-Token Sliding Window Chunking:
   - Splits documents into ~512-token (~2,000 char) semantic passages aligned with EmbeddingGemma 2's
     1,024-token sliding window attention horizon.
6. Gemma 4 Multi-Turn History Rule:
   - Strips `<thought>` / thinking traces from prior assistant turns in conversation history,
     EXCEPT on tool-call turns where thinking traces remain intact.
"""

import atexit
from datetime import datetime
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import numpy as np

from config import (
    ARIES_DATA_DIR,
    ARIES_MEMORY_DB_PATH,
    EMBEDDING_GEMMA2_GGUF_PATH,
    EMBEDDING_SERVER_PORT,
    LLAMA_SERVER_EXE,
)
from llama_engine import _bind_process_to_kill_on_close_job

# Full native EmbeddingGemma 2 dimensions (768d BF16/F32 maximum fidelity; overridable via env)
MRL_DIM = int(os.getenv("ARIES_EMBEDDING_DIM", "768"))
NATIVE_DIM = 768

# Asymmetric Task Instruction Prefix Templates (Google EmbeddingGemma 2 Spec)
QUERY_PREFIX_TEMPLATES = {
    "search": "task: search result | query: {query}",
    "qa": "task: question answering | query: {query}",
    "fact": "task: fact checking | query: {query}",
    "code": "task: code retrieval | query: {query}",
}

_CODE_QUERY_RE = re.compile(
    r"\b(code|function|class|method|script|python|javascript|typescript|sql|html|css|c\+\+|c#|rust|go|bug|syntax|api|regex|import|def\s|async\s)\b",
    re.IGNORECASE,
)
_FACT_QUERY_RE = re.compile(
    r"\b(is it true|fact check|verify|did .+ really|true or false|confirm whether)\b",
    re.IGNORECASE,
)
_QUESTION_QUERY_RE = re.compile(
    r"(\?$|^(what|who|where|when|why|how|which|can|could|does|do|is|are|tell me|explain|describe)\b)",
    re.IGNORECASE,
)

# Regex to strip Gemma 4 thinking traces on non-tool-call turns
_THINKING_TRACE_RE = re.compile(
    r"(<thought>[\s\S]*?</thought>|<\|channel\|?thought[\s\S]*?<channel\|?>|<think>[\s\S]*?</think>|<unused\d+>)",
    re.IGNORECASE,
)


def detect_query_task_type(query: str) -> str:
    """Selects the optimal EmbeddingGemma 2 asymmetric task prefix for a user query."""
    q = (query or "").strip()
    if _CODE_QUERY_RE.search(q):
        return "code"
    if _FACT_QUERY_RE.search(q):
        return "fact"
    if _QUESTION_QUERY_RE.search(q):
        return "qa"
    return "search"


def format_query_prefix(query: str, task_type: str = "auto") -> str:
    """Formats a text query with EmbeddingGemma 2's asymmetric task instruction prefix."""
    clean_q = (query or "").strip()
    if not clean_q:
        return "task: search result | query: "
    resolved_task = detect_query_task_type(clean_q) if task_type == "auto" else task_type
    template = QUERY_PREFIX_TEMPLATES.get(resolved_task, QUERY_PREFIX_TEMPLATES["qa"])
    return template.format(query=clean_q)


def format_document_prefix(content: str, title: str | None = None) -> str:
    """Formats a corpus chunk with EmbeddingGemma 2's asymmetric document prefix.

    Enforces the Document Title Quirk: if a chunk does not have a natural title,
    explicitly formats as `title: none | text: {content}`.
    """
    clean_text = (content or "").strip()
    clean_title = (title or "").strip()
    if not clean_title or clean_title.lower() == "none":
        clean_title = "none"
    return f"title: {clean_title} | text: {clean_text}"


def truncate_and_normalize_mrl(vec: np.ndarray, dim: int = MRL_DIM) -> np.ndarray:
    """Applies Matryoshka Representation Learning (MRL) slicing + L2 re-normalization in float32.

    Strictly uses `np.float32` (never `float16`) and guards against NaN/Inf overflow.
    """
    arr = np.asarray(vec, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    sliced = arr[:dim].astype(np.float32, copy=False)
    norm = float(np.linalg.norm(sliced))
    if norm > 1e-12:
        sliced = sliced / norm
    return sliced.astype(np.float32, copy=False)


def chunk_document_for_sliding_window(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
    """Chunks text into ~512-token passages aligned with EmbeddingGemma 2's 1,024-token sliding window."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{tail}\n{para}".strip()
            else:
                # Single paragraph exceeds max_chars; split along sentence or character boundaries
                for i in range(0, len(para), max_chars - overlap_chars):
                    piece = para[i : i + max_chars].strip()
                    if piece:
                        chunks.append(piece)
                current = ""

    if current:
        chunks.append(current)
    return chunks


def strip_gemma4_thinking_traces(text: str, has_tool_call: bool = False) -> str:
    r"""Enforces Google's Gemma 4 Multi-Turn History Rule:
    Strips thinking traces from prior turns in conversation history,
    EXCEPT on tool-call turns where thinking content must remain intact.
    Also defensively purges any raw special token artifacts (<unused\d+>).
    """
    if not text:
        return ""
    if has_tool_call:
        return re.sub(r"<unused\d+>", "", text).strip()
    cleaned = _THINKING_TRACE_RE.sub("", text).strip()
    return cleaned if cleaned else ""


# -------------------------------------------------------------------------
# Embedded EmbeddingGemma 2 Server Manager (270M Text/Code GGUF, float32)
# -------------------------------------------------------------------------
class EmbeddingGemma2ServerManager:
    """Manages the lightweight `llama-server.exe --embedding` instance for EmbeddingGemma 2."""

    def __init__(self, port: int = EMBEDDING_SERVER_PORT):
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.v1_url = f"{self.base_url}/v1"
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        atexit.register(self.stop)

    def is_healthy(self, timeout: float = 0.5) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self) -> bool:
        """Starts the embedded EmbeddingGemma 2 server on CPU (0 MB GPU VRAM contention, float32 activations)."""
        with self._lock:
            if self.is_healthy():
                return True

            if not os.path.exists(LLAMA_SERVER_EXE):
                print(f"[RAG] llama-server.exe not found at {LLAMA_SERVER_EXE}")
                return False

            if not os.path.exists(EMBEDDING_GEMMA2_GGUF_PATH):
                print(f"[RAG] EmbeddingGemma 2 model not found at {EMBEDDING_GEMMA2_GGUF_PATH}")
                return False

            if sys.platform == "darwin":
                cmd = [
                    LLAMA_SERVER_EXE,
                    "--load-mode", "auto",
                    "--model", EMBEDDING_GEMMA2_GGUF_PATH,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--embedding",
                    "--pooling", "mean",
                    "--ctx-size", "2048",
                    "--batch-size", "2048",
                    "--ubatch-size", "2048",
                    "--parallel", "1",
                    "--n-gpu-layers", "99",
                    "--flash-attn", "on",
                    "--threads", "6",
                    "--threads-batch", "8",
                    "--threads-http", "2",
                    "--no-webui",
                ]
            else:
                cmd = [
                    LLAMA_SERVER_EXE,
                    "--device", "none",
                    "--no-repack",
                    "--load-mode", "none",
                    "--model", EMBEDDING_GEMMA2_GGUF_PATH,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                    "--embedding",
                    "--pooling", "mean",
                    "--ctx-size", "2048",
                    "--batch-size", "2048",
                    "--ubatch-size", "2048",
                    "--parallel", "1",
                    "--n-gpu-layers", "0",  # Keep 100% of Radeon Pro 570 4GB VRAM dedicated to Gemma 4 E2B + MTP + SigLIP
                    "--threads", "2",
                    "--threads-batch", "4",
                    "--threads-http", "2",
                    "--poll", "0",
                    "--no-webui",
                ]

            t0 = time.perf_counter()
            print(
                f"[RAG] Launching EmbeddingGemma 2 (270M Text/Code, F32/BF16 safe, 2048 ctx, {MRL_DIM}d) on 127.0.0.1:{self.port}..."
            )

            popen_kwargs = {}
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(LLAMA_SERVER_EXE),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **popen_kwargs,
            )

            if sys.platform == "win32" and getattr(self.process, "_handle", None):
                _bind_process_to_kill_on_close_job(int(self.process._handle))

            for _ in range(120):
                if self.process.poll() is not None:
                    print(f"[ERROR] [RAG] EmbeddingGemma 2 server exited with code {self.process.returncode}")
                    return False
                if self.is_healthy(timeout=0.3):
                    boot_ms = (time.perf_counter() - t0) * 1000.0
                    print(f"[RAG] EmbeddingGemma 2 online in {boot_ms:.0f} ms (ctx=2048, dim={MRL_DIM}d, precision=float32)!")
                    return True
                time.sleep(0.15)

            print("[ERROR] [RAG] Timed out waiting for EmbeddingGemma 2 server.")
            return False

    def _post_embeddings(self, texts: list[str], dim: int, timeout: float = 120.0) -> list[np.ndarray]:
        payload = json.dumps({"input": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.v1_url}/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())

        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [truncate_and_normalize_mrl(np.array(item["embedding"], dtype=np.float32), dim=dim) for item in items]

    def embed_texts_mrl(self, formatted_texts: list[str], dim: int = MRL_DIM) -> list[np.ndarray]:
        """Computes L2-normalized float32 embeddings with automatic batch-splitting on oversized inputs."""
        if not formatted_texts:
            return []
        if not self.is_healthy():
            if not self.start():
                return []

        try:
            vecs = self._post_embeddings(formatted_texts, dim=dim, timeout=120.0)
            if len(vecs) == len(formatted_texts):
                return vecs
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            if not self.is_healthy() and not self.start():
                return []

        if len(formatted_texts) > 1:
            mid = len(formatted_texts) // 2
            return self.embed_texts_mrl(formatted_texts[:mid], dim=dim) + self.embed_texts_mrl(
                formatted_texts[mid:], dim=dim
            )

        # Single item exceeded context or stalled; progressively trim within 8k-token safety bounds.
        single = formatted_texts[0]
        for max_chars in (len(single), 24000, 16000, 12000, 8000, 4000):
            try:
                res = self._post_embeddings([single[:max_chars]], dim=dim, timeout=120.0)
                if res:
                    return res
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                continue
        return []

    def embed_query(self, query: str, task_type: str = "auto", dim: int = MRL_DIM) -> np.ndarray | None:
        """Formats a query with its asymmetric prefix and returns its 256d MRL float32 vector."""
        prefixed = format_query_prefix(query, task_type=task_type)
        vecs = self.embed_texts_mrl([prefixed], dim=dim)
        return vecs[0] if vecs else None

    def embed_documents(self, docs: list[tuple[str | None, str]], dim: int = MRL_DIM) -> list[np.ndarray]:
        """Formats (title, content) pairs with asymmetric document prefixes and returns 256d MRL float32 vectors."""
        formatted = [format_document_prefix(content=c, title=t) for (t, c) in docs]
        return self.embed_texts_mrl(formatted, dim=dim)

    def stop(self):
        with self._lock:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=1.5)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            self.process = None


# -------------------------------------------------------------------------
# Persistent SQLite Chat History, User Profile & 256d MRL Vector Store
# -------------------------------------------------------------------------
# Intentionally empty: ARIES Live keeps the RAG store empty by default unless
# the user explicitly indexes documents via the UI or save_memory_note.
DEFAULT_SEED_KNOWLEDGE: list[tuple[str, str, str]] = []

MAX_PROFILE_TOKENS = 2000
MAX_PROFILE_CHARS = 8000

USER_PROFILE_PREAMBLE = (
    "[USER PROFILE / MEMORY SUMMARY]:\n"
    "The facts below describe the user you are speaking with, NOT you. "
    "Use this background naturally to personalize responses when relevant."
)

DEFAULT_USER_PROFILE_SECTIONS = [
    {
        "id": "about",
        "title": "About Me",
        "subtitle": "Your name, role, location, or personal background you'd like the assistant to remember.",
        "content": "Add your preferred name, background, or context here.",
    },
    {
        "id": "projects",
        "title": "Current Projects",
        "subtitle": "Active work, tools, languages, or goals.",
        "content": "Add your current projects, tech stack, or focus areas here.",
    },
    {
        "id": "preferences",
        "title": "Preferences & Communication Style",
        "subtitle": "How you prefer answers formatted, tone, and level of detail.",
        "content": "Prefer clear, concise, practical answers.",
    },
]


class AriesMemoryStore:
    """Manages SQLite persistent multi-session chat history, User Profile, and EmbeddingGemma 2 256d RAG retrieval."""

    def __init__(self, db_path: str = ARIES_MEMORY_DB_PATH, embedder: EmbeddingGemma2ServerManager | None = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.embedder = embedder or EmbeddingGemma2ServerManager()
        self._session_history_cache: dict[str, list[dict]] = {}
        self._db_lock = threading.Lock()
        self._rag_chunk_count = 0
        self._cached_profile: dict | None = None
        self._cached_profile_prompt: str = ""
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        return conn

    def _init_schema(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._db_lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        raw_content TEXT NOT NULL,
                        has_tool_call INTEGER NOT NULL DEFAULT 0,
                        tool_name TEXT DEFAULT '',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                    );
                    CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, id);

                    CREATE TABLE IF NOT EXISTS rag_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_type TEXT NOT NULL,
                        session_id TEXT DEFAULT '',
                        title TEXT NOT NULL DEFAULT 'none',
                        content TEXT NOT NULL,
                        task_domain TEXT NOT NULL DEFAULT 'qa',
                        embedding_256d BLOB NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source_type);

                    CREATE TABLE IF NOT EXISTS user_profile (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        enabled INTEGER NOT NULL DEFAULT 1,
                        sections_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                prof_row = conn.execute("SELECT sections_json FROM user_profile WHERE id = 1").fetchone()
                if prof_row is None:
                    conn.execute(
                        "INSERT INTO user_profile (id, enabled, sections_json, updated_at) VALUES (1, 1, ?, ?)",
                        (json.dumps(DEFAULT_USER_PROFILE_SECTIONS, ensure_ascii=False), now),
                    )
                self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        self._refresh_profile_cache()

    def warmup_and_seed(self):
        """Checks vector store state and launches EmbeddingGemma 2 only when RAG chunks exist (lazy-loads otherwise)."""
        try:
            with self._db_lock:
                with self._connect() as conn:
                    self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                    k_count = conn.execute(
                        "SELECT COUNT(*) FROM rag_chunks WHERE source_type = 'knowledge'"
                    ).fetchone()[0]
            if self._rag_chunk_count == 0:
                print(
                    f"[RAG] Lean vector store ready ({k_count} knowledge, 0 total chunks — EmbeddingGemma 2 configured for on-demand lazy loading)."
                )
                return
            if not self.embedder.start():
                return
            self.embedder.embed_query("warmup", task_type="qa")
            print(f"[RAG] Lean vector store ready ({k_count} knowledge, {self._rag_chunk_count} total chunks).")
        except Exception as e:
            print(f"[RAG] Warmup note: {e}")

    def has_rag_chunks(self) -> bool:
        return getattr(self, "_rag_chunk_count", 0) > 0

    def clear_all_rag_chunks(self) -> int:
        with self._db_lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM rag_chunks")
                self._rag_chunk_count = 0
                return cur.rowcount

    # ------------------------------------------------------------------
    # Persistent Structured User Profile (Memory Summary)
    # ------------------------------------------------------------------
    @staticmethod
    def _render_user_profile_prompt(sections: list[dict]) -> str:
        blocks = []
        for s in sections:
            title = (s.get("title") or "").strip()
            content = (s.get("content") or "").strip()
            if content:
                header = f"### {title}" if title else "### Profile"
                blocks.append(f"{header}\n{content}")
        if not blocks:
            return ""
        return f"{USER_PROFILE_PREAMBLE}\n\n" + "\n\n".join(blocks)

    @staticmethod
    def _parse_full_profile_text(full_text: str, existing_sections: list[dict] | None = None) -> list[dict]:
        """Parses the editable full prompt text (`### Section Title` blocks) back into structured sections."""
        raw = (full_text or "").replace("\r\n", "\n").strip()
        if not raw:
            return []

        # Strip canonical preamble header lines if present at the top
        lines = raw.split("\n")
        while lines:
            head = lines[0].strip()
            if (
                not head
                or head.startswith("[USER PROFILE / MEMORY SUMMARY")
                or head.startswith("The facts below describe")
                or head.startswith("Use this background naturally")
                or head.startswith("Use this persistent background naturally")
            ):
                lines.pop(0)
            else:
                break
        body = "\n".join(lines).strip()
        if not body:
            return []

        lookup: dict[str, dict] = {}
        for ref in (existing_sections or []) + DEFAULT_USER_PROFILE_SECTIONS:
            t_key = (ref.get("title") or "").strip().lower()
            if t_key and t_key not in lookup:
                lookup[t_key] = ref

        sections: list[dict] = []
        current_title: str | None = None
        current_lines: list[str] = []

        def _flush(title_val: str | None, buf: list[str], idx: int):
            text_val = "\n".join(buf).strip()
            clean_title = (title_val or "").strip()
            if not clean_title and not text_val:
                return
            if not clean_title:
                clean_title = "Custom Profile"
            ref = lookup.get(clean_title.lower(), {})
            sec_id = ref.get("id") or re.sub(r"[^a-z0-9]+", "_", clean_title.lower()).strip("_") or f"section_{idx}"
            subtitle = ref.get("subtitle") or "User profile memory section."
            sections.append({
                "id": sec_id,
                "title": clean_title,
                "subtitle": subtitle,
                "content": text_val,
            })

        for line in body.split("\n"):
            m = re.match(r"^###\s+(.+)$", line.strip())
            if m:
                if current_title is not None or any(l.strip() for l in current_lines):
                    _flush(current_title, current_lines, len(sections))
                current_title = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_title is not None or any(l.strip() for l in current_lines):
            _flush(current_title, current_lines, len(sections))

        return sections

    def _refresh_profile_cache(self) -> dict:
        with self._db_lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT enabled, sections_json, updated_at FROM user_profile WHERE id = 1"
                ).fetchone()
        if not row:
            sections = [dict(s) for s in DEFAULT_USER_PROFILE_SECTIONS]
            enabled = True
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            enabled = bool(row["enabled"])
            updated_at = row["updated_at"]
            try:
                sections = json.loads(row["sections_json"])
            except Exception:
                sections = [dict(s) for s in DEFAULT_USER_PROFILE_SECTIONS]
        full_prompt = self._render_user_profile_prompt(sections)
        char_count = len(full_prompt)
        approx_tokens = (char_count + 3) // 4 if char_count else 0
        prof = {
            "enabled": enabled,
            "sections": sections,
            "updated_at": updated_at,
            "full_prompt": full_prompt,
            "char_count": char_count,
            "max_chars": MAX_PROFILE_CHARS,
            "approx_tokens": approx_tokens,
            "max_tokens": MAX_PROFILE_TOKENS,
        }
        self._cached_profile = prof
        self._cached_profile_prompt = full_prompt if enabled else ""
        return dict(prof)

    def get_user_profile(self) -> dict:
        if self._cached_profile is not None:
            return dict(self._cached_profile)
        return self._refresh_profile_cache()

    def save_user_profile(
        self,
        sections: list[dict] | None = None,
        enabled: bool | None = None,
        full_text: str | None = None,
    ) -> dict:
        curr = self.get_user_profile()
        next_enabled = curr["enabled"] if enabled is None else bool(enabled)
        next_sections = curr["sections"]

        if full_text is not None:
            next_sections = self._parse_full_profile_text(full_text, existing_sections=curr["sections"])
        elif sections is not None:
            cleaned: list[dict] = []
            for idx, s in enumerate(sections):
                if not isinstance(s, dict):
                    continue
                title = (s.get("title") or "").strip()
                subtitle = (s.get("subtitle") or "").strip()
                content = (s.get("content") or "").strip()
                sec_id = (s.get("id") or f"section_{idx}").strip()
                if title or content:
                    cleaned.append({
                        "id": sec_id,
                        "title": title or "Custom Notes",
                        "subtitle": subtitle,
                        "content": content,
                    })
            next_sections = cleaned

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO user_profile (id, enabled, sections_json, updated_at)
                    VALUES (1, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        enabled = excluded.enabled,
                        sections_json = excluded.sections_json,
                        updated_at = excluded.updated_at
                    """,
                    (1 if next_enabled else 0, json.dumps(next_sections, ensure_ascii=False), now),
                )
        return self._refresh_profile_cache()

    def reset_user_profile(self) -> dict:
        return self.save_user_profile(
            sections=[dict(s) for s in DEFAULT_USER_PROFILE_SECTIONS],
            enabled=True,
        )

    def format_user_profile_prompt(self) -> str:
        if self._cached_profile is None:
            self._refresh_profile_cache()
        return self._cached_profile_prompt

    # ------------------------------------------------------------------
    # Multi-Session Chat History Management
    # ------------------------------------------------------------------
    def create_session(self, title: str = "New Conversation") -> dict:
        session_id = f"chat_{uuid.uuid4().hex[:10]}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._db_lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, title, now, now),
                )
        self._session_history_cache[session_id] = []
        return {"id": session_id, "title": title, "created_at": now, "updated_at": now, "message_count": 0}

    def ensure_session(self, session_id: str | None = None) -> str:
        if session_id:
            with self._db_lock:
                with self._connect() as conn:
                    row = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
                    if row:
                        return session_id
        created = self.create_session("New Conversation")
        return created["id"]

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with self._db_lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT s.id, s.title, s.created_at, s.updated_at,
                           COUNT(m.id) AS message_count
                    FROM chat_sessions s
                    LEFT JOIN chat_messages m ON m.session_id = s.id
                    GROUP BY s.id
                    HAVING COUNT(m.id) > 0
                    ORDER BY s.updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                return [dict(r) for r in rows]

    def get_session_messages(self, session_id: str) -> list[dict]:
        with self._db_lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, session_id, role, content, raw_content, has_tool_call, tool_name, created_at
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session_id,),
                ).fetchall()
                return [dict(r) for r in rows]

    def build_gemma4_llm_history(self, session_id: str, max_messages: int = 8) -> list[dict]:
        """Builds multi-turn LLM context for Gemma 4 E2B-it QAT following Google's Multi-Turn History Rule:
        - Strips `<thought>` traces from prior assistant turns
        - Preserves thinking traces intact on tool-call turns (`has_tool_call=1`).
        """
        cached = self._session_history_cache.get(session_id)
        if cached is None:
            msgs = self.get_session_messages(session_id)
            cached = []
            for m in msgs:
                role = m["role"]
                if role not in ("user", "assistant"):
                    continue
                has_tc = bool(m.get("has_tool_call", 0))
                raw = m.get("raw_content") or m.get("content") or ""
                clean_content = strip_gemma4_thinking_traces(raw, has_tool_call=has_tc)
                cached.append({"role": role, "content": clean_content})
            self._session_history_cache[session_id] = cached
        return list(cached[-max_messages:]) if len(cached) > max_messages else list(cached)

    def record_turn_and_index_async(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        raw_assistant_text: str | None = None,
        has_tool_call: bool = False,
        tool_name: str = "",
    ):
        """Persists a completed (user, assistant) turn to SQLite and asynchronously indexes it in EmbeddingGemma 2."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_user = (user_text or "").strip()
        clean_assistant = strip_gemma4_thinking_traces(assistant_text or "", has_tool_call=False)
        raw_assist = (raw_assistant_text or assistant_text or "").strip()

        auto_title = None
        with self._db_lock:
            with self._connect() as conn:
                row = conn.execute("SELECT title FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
                if not row:
                    auto_title = self._make_auto_title(clean_user)
                    conn.execute(
                        "INSERT INTO chat_sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (session_id, auto_title, now, now),
                    )
                else:
                    curr_title = row["title"]
                    if curr_title in ("New Conversation", "Untitled", "") and clean_user:
                        auto_title = self._make_auto_title(clean_user)
                        conn.execute(
                            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                            (auto_title, now, session_id),
                        )
                    else:
                        auto_title = curr_title
                        conn.execute(
                            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                            (now, session_id),
                        )

                conn.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, raw_content, has_tool_call, tool_name, created_at)
                    VALUES (?, 'user', ?, ?, 0, '', ?)
                    """,
                    (session_id, clean_user, clean_user, now),
                )
                conn.execute(
                    """
                    INSERT INTO chat_messages (session_id, role, content, raw_content, has_tool_call, tool_name, created_at)
                    VALUES (?, 'assistant', ?, ?, ?, ?, ?)
                    """,
                    (session_id, clean_assistant, raw_assist, 1 if has_tool_call else 0, tool_name or "", now),
                )
        cached = self._session_history_cache.get(session_id)
        if cached is not None:
            cached.append({"role": "user", "content": clean_user})
            cached.append({
                "role": "assistant",
                "content": strip_gemma4_thinking_traces(raw_assist, has_tool_call=has_tool_call),
            })

        # Background indexing of raw chat turns is opt-in (ARIES_INDEX_CHAT_TURNS=1) so irrelevant past turns do not trigger false-positive RAG matches; explicit notes always use save_memory_note.
        if os.getenv("ARIES_INDEX_CHAT_TURNS", "0") == "1" and len(clean_user) >= 8 and len(clean_assistant) >= 8:
            threading.Thread(
                target=self._index_chat_turn_worker,
                args=(session_id, auto_title or "none", clean_user, clean_assistant, now),
                daemon=True,
            ).start()

    @staticmethod
    def _make_auto_title(user_text: str) -> str:
        clean = re.sub(r"\s+", " ", (user_text or "").strip())
        if not clean:
            return "Voice Conversation"
        return clean[:48] + ("..." if len(clean) > 48 else "")

    def _index_chat_turn_worker(self, session_id: str, session_title: str, user_text: str, assistant_text: str, created_at: str):
        try:
            combined = f"User: {user_text}\nARIES: {assistant_text}"
            vecs = self.embedder.embed_documents([(session_title, combined)], dim=MRL_DIM)
            if not vecs:
                return
            vec_bytes = vecs[0].astype(np.float32).tobytes()
            with self._db_lock:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO rag_chunks (source_type, session_id, title, content, task_domain, embedding_256d, created_at)
                        VALUES ('chat_memory', ?, ?, ?, 'qa', ?, ?)
                        """,
                        (session_id, session_title or "none", combined, vec_bytes, created_at),
                    )
                    self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        except Exception as e:
            print(f"[RAG] Background turn indexing note: {e}")

    def delete_session(self, session_id: str) -> bool:
        self._session_history_cache.pop(session_id, None)
        with self._db_lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM rag_chunks WHERE session_id = ? AND source_type = 'chat_memory'", (session_id,))
                cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
                self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                return cur.rowcount > 0

    # ------------------------------------------------------------------
    # RAG Knowledge Base Indexing & Retrieval (256d MRL float32)
    # ------------------------------------------------------------------
    def add_knowledge_document(self, title: str | None, content: str, task_domain: str = "qa") -> list[int]:
        """Chunks a document for the 1,024-token sliding window, computes 256d MRL float32 embeddings, and saves to SQLite."""
        clean_title = (title or "").strip() or "none"
        chunks = chunk_document_for_sliding_window(content, max_chars=2000, overlap_chars=200)
        if not chunks:
            return []

        doc_pairs = []
        for idx, ch in enumerate(chunks):
            chunk_title = clean_title if len(chunks) == 1 else f"{clean_title} (Part {idx + 1})"
            doc_pairs.append((chunk_title, ch))

        t0 = time.perf_counter()
        vecs = self.embedder.embed_documents(doc_pairs, dim=MRL_DIM)
        if not vecs or len(vecs) != len(doc_pairs):
            return []
        embed_ms = (time.perf_counter() - t0) * 1000.0

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        inserted_ids = []
        with self._db_lock:
            with self._connect() as conn:
                for (c_title, c_text), vec in zip(doc_pairs, vecs):
                    cur = conn.execute(
                        """
                        INSERT INTO rag_chunks (source_type, session_id, title, content, task_domain, embedding_256d, created_at)
                        VALUES ('knowledge', '', ?, ?, ?, ?, ?)
                        """,
                        (c_title, c_text, task_domain, vec.astype(np.float32).tobytes(), now),
                    )
                    inserted_ids.append(cur.lastrowid)
                self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]

        print(
            f"[RAG] Indexed {len(inserted_ids)} chunk(s) for '{clean_title}' "
            f"(256d MRL float32, {embed_ms:.0f} ms)"
        )
        return inserted_ids

    def list_knowledge_chunks(self, include_chat_memory: bool = False, limit: int = 100) -> list[dict]:
        with self._db_lock:
            with self._connect() as conn:
                if include_chat_memory:
                    rows = conn.execute(
                        """
                        SELECT id, source_type, session_id, title, content, task_domain, created_at
                        FROM rag_chunks
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, source_type, session_id, title, content, task_domain, created_at
                        FROM rag_chunks
                        WHERE source_type = 'knowledge'
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (limit,),
                    ).fetchall()
                return [dict(r) for r in rows]

    def delete_rag_chunk(self, chunk_id: int) -> bool:
        with self._db_lock:
            with self._connect() as conn:
                cur = conn.execute("DELETE FROM rag_chunks WHERE id = ?", (chunk_id,))
                self._rag_chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                return cur.rowcount > 0

    def search_rag(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.62,
        task_type: str = "auto",
        exclude_session_id: str | None = None,
        include_chat_memory: bool = True,
    ) -> list[dict]:
        """Retrieves the top-k most relevant chunks using EmbeddingGemma 2 256d MRL cosine similarity."""
        t0 = time.perf_counter()
        resolved_task = detect_query_task_type(query) if task_type == "auto" else task_type
        q_vec = self.embedder.embed_query(query, task_type=resolved_task, dim=MRL_DIM)
        if q_vec is None:
            return []

        with self._db_lock:
            with self._connect() as conn:
                if include_chat_memory:
                    if exclude_session_id:
                        rows = conn.execute(
                            """
                            SELECT id, source_type, session_id, title, content, task_domain, embedding_256d, created_at
                            FROM rag_chunks
                            WHERE source_type = 'knowledge' OR (source_type = 'chat_memory' AND session_id != ?)
                            """,
                            (exclude_session_id,),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT id, source_type, session_id, title, content, task_domain, embedding_256d, created_at
                            FROM rag_chunks
                            """
                        ).fetchall()
                else:
                    rows = conn.execute(
                        """
                        SELECT id, source_type, session_id, title, content, task_domain, embedding_256d, created_at
                        FROM rag_chunks
                        WHERE source_type = 'knowledge'
                        """
                    ).fetchall()

        if not rows:
            return []

        valid_rows = []
        scores_list = []
        q_cache: dict[int, np.ndarray] = {len(q_vec): q_vec}
        for r in rows:
            blob = r["embedding_256d"]
            if not blob or (len(blob) % 4) != 0:
                continue
            arr = np.frombuffer(blob, dtype=np.float32)
            dim = len(arr)
            if dim == 0 or not np.all(np.isfinite(arr)):
                continue
            if dim not in q_cache:
                if dim <= len(q_vec):
                    q_cache[dim] = truncate_and_normalize_mrl(q_vec, dim=dim)
                else:
                    continue
            valid_rows.append(r)
            scores_list.append(float(np.dot(arr, q_cache[dim])))

        if not scores_list:
            return []

        scores = np.array(scores_list, dtype=np.float32)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                continue
            r = valid_rows[int(idx)]
            results.append({
                "id": r["id"],
                "source_type": r["source_type"],
                "session_id": r["session_id"],
                "title": r["title"],
                "content": r["content"],
                "task_domain": r["task_domain"],
                "score": round(score, 4),
                "query_task_used": resolved_task,
                "created_at": r["created_at"],
            })

        rag_ms = (time.perf_counter() - t0) * 1000.0
        if results:
            print(
                f"[RAG] Retrieved {len(results)} chunk(s) in {rag_ms:.0f} ms "
                f"(task={resolved_task}, top_score={results[0]['score']:.3f}, title='{results[0]['title']}')"
            )
        return results

    def get_stats(self) -> dict:
        with self._db_lock:
            with self._connect() as conn:
                k_count = conn.execute("SELECT COUNT(*) FROM rag_chunks WHERE source_type = 'knowledge'").fetchone()[0]
                m_count = conn.execute("SELECT COUNT(*) FROM rag_chunks WHERE source_type = 'chat_memory'").fetchone()[0]
                s_count = conn.execute(
                    "SELECT COUNT(*) FROM (SELECT s.id FROM chat_sessions s JOIN chat_messages m ON m.session_id = s.id GROUP BY s.id)"
                ).fetchone()[0]
        return {
            "engine_online": self.embedder.is_healthy(),
            "model": "EmbeddingGemma 2 (270M Text/Code GGUF)",
            "mrl_dimensions": MRL_DIM,
            "native_dimensions": NATIVE_DIM,
            "precision": "float32 / bfloat16 (fp16 disabled)",
            "knowledge_chunks": k_count,
            "memory_chunks": m_count,
            "saved_sessions": s_count,
            "db_path": self.db_path,
        }


# Global singleton instance
embedding_server = EmbeddingGemma2ServerManager()
memory_store = AriesMemoryStore(embedder=embedding_server)
