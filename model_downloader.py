"""First-Launch Graphical Model Auto-Downloader & Local Model Importer.

Streams required Apache 2.0 / CC-BY-4.0 models from Hugging Face with HTTP Range
resume (.part files), atomic rename, per-model & aggregate progress telemetry,
and instant symlink/copy import from an existing local models/ directory.
"""

import asyncio
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp

from config import ARIES_DATA_DIR

MODELS_ROOT = os.path.join(ARIES_DATA_DIR, "models")


@dataclass
class ModelFileSpec:
    url: str
    rel_path: str
    expected_bytes: int


@dataclass
class ModelPackageSpec:
    id: str
    name: str
    role: str
    license: str
    total_bytes: int
    files: list[ModelFileSpec]
    optional: bool = False


MODEL_MANIFEST: list[ModelPackageSpec] = [
    ModelPackageSpec(
        id="gemma4_llm",
        name="Gemma 4 E2B IT (4-bit QAT)",
        role="Primary On-Device LLM",
        license="Apache 2.0",
        total_bytes=2_580_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
                rel_path=os.path.join("gemma-4", "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
                expected_bytes=2_580_000_000,
            )
        ],
    ),
    ModelPackageSpec(
        id="gemma4_vision",
        name="Gemma 4 Vision Projector",
        role="Multimodal Image & Document Vision",
        license="Apache 2.0",
        total_bytes=987_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/mmproj-BF16.gguf",
                rel_path=os.path.join("gemma-4", "mmproj-BF16.gguf"),
                expected_bytes=987_000_000,
            )
        ],
    ),
    ModelPackageSpec(
        id="gemma4_mtp",
        name="Gemma 4 MTP Speculative Draft",
        role="Multi-Token Prediction Speedup",
        license="Apache 2.0",
        total_bytes=59_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF/resolve/main/MTP/mtp-gemma-4-E2B-it-Q4_0.gguf",
                rel_path=os.path.join("gemma-4", "mtp-gemma-4-E2B-it-Q4_0.gguf"),
                expected_bytes=59_000_000,
            )
        ],
    ),
    ModelPackageSpec(
        id="embeddinggemma2",
        name="EmbeddingGemma 2 (270M)",
        role="Local SQLite Vector RAG",
        license="Apache 2.0",
        total_bytes=176_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/unsloth/embeddinggemma-2-GGUF/resolve/main/embeddinggemma-2-UD-Q4_K_XL.gguf",
                rel_path=os.path.join("embeddinggemma-2", "embeddinggemma-2-UD-Q4_K_XL.gguf"),
                expected_bytes=176_000_000,
            )
        ],
    ),
    ModelPackageSpec(
        id="parakeet_asr",
        name="Parakeet-TDT 0.6B v2 (INT8)",
        role="Real-Time Speech Recognition",
        license="CC-BY-4.0",
        total_bytes=664_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/encoder.int8.onnx",
                rel_path=os.path.join("parakeet_06b", "encoder.int8.onnx"),
                expected_bytes=652_000_000,
            ),
            ModelFileSpec(
                url="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/decoder.int8.onnx",
                rel_path=os.path.join("parakeet_06b", "decoder.int8.onnx"),
                expected_bytes=7_200_000,
            ),
            ModelFileSpec(
                url="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/joiner.int8.onnx",
                rel_path=os.path.join("parakeet_06b", "joiner.int8.onnx"),
                expected_bytes=1_800_000,
            ),
            ModelFileSpec(
                url="https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/resolve/main/tokens.txt",
                rel_path=os.path.join("parakeet_06b", "tokens.txt"),
                expected_bytes=10_000,
            ),
        ],
    ),
    ModelPackageSpec(
        id="phonon_tts",
        name="Phonon 31M Freya (INT8)",
        role="Neural Text-to-Speech",
        license="Apache 2.0",
        total_bytes=87_000_000,
        files=[
            ModelFileSpec(
                url="https://huggingface.co/litert-community/phonon-31m-en/resolve/main/phonon_int8.litertlm",
                rel_path=os.path.join("phonon", "phonon_int8.litertlm"),
                expected_bytes=87_000_000,
            )
        ],
    ),
]


@dataclass
class ModelProgressState:
    id: str
    name: str
    role: str
    license: str
    total_bytes: int
    downloaded_bytes: int = 0
    status: str = "pending"  # pending | downloading | ready | error | cancelled
    error: str = ""


class ModelDownloadManager:
    def __init__(self, models_root: str = MODELS_ROOT):
        self.models_root = models_root
        os.makedirs(self.models_root, exist_ok=True)
        self._task: Optional[asyncio.Task] = None
        self._cancelled = False
        self._is_downloading = False
        self._speed_bps: float = 0.0
        self._eta_seconds: float = 0.0
        self._states: dict[str, ModelProgressState] = {}
        self.refresh_states()

    def _is_package_present(self, pkg: ModelPackageSpec) -> tuple[bool, int]:
        present = True
        total_on_disk = 0
        for f in pkg.files:
            full = os.path.join(self.models_root, f.rel_path)
            if os.path.exists(full) and os.path.getsize(full) > 0:
                total_on_disk += os.path.getsize(full)
            else:
                present = False
                part = full + ".part"
                if os.path.exists(part):
                    total_on_disk += os.path.getsize(part)
        return present, total_on_disk

    def refresh_states(self):
        for pkg in MODEL_MANIFEST:
            present, on_disk = self._is_package_present(pkg)
            prev = self._states.get(pkg.id)
            if self._is_downloading and prev and prev.status == "downloading":
                continue
            self._states[pkg.id] = ModelProgressState(
                id=pkg.id,
                name=pkg.name,
                role=pkg.role,
                license=pkg.license,
                total_bytes=pkg.total_bytes,
                downloaded_bytes=pkg.total_bytes if present else on_disk,
                status="ready" if present else "pending",
            )

    def all_models_ready(self) -> bool:
        self.refresh_states()
        return all(self._states[p.id].status == "ready" for p in MODEL_MANIFEST if not p.optional)

    def get_status(self) -> dict:
        self.refresh_states()
        items = []
        total_bytes = 0
        downloaded_bytes = 0
        for pkg in MODEL_MANIFEST:
            st = self._states[pkg.id]
            total_bytes += st.total_bytes
            downloaded_bytes += min(st.downloaded_bytes, st.total_bytes)
            pct = (st.downloaded_bytes / st.total_bytes * 100.0) if st.total_bytes > 0 else 0.0
            items.append({
                "id": st.id,
                "name": st.name,
                "role": st.role,
                "license": st.license,
                "total_bytes": st.total_bytes,
                "downloaded_bytes": st.downloaded_bytes,
                "percent": round(min(100.0, pct), 1),
                "status": st.status,
                "error": st.error,
            })
        agg_pct = (downloaded_bytes / total_bytes * 100.0) if total_bytes > 0 else 0.0
        return {
            "all_ready": all(st["status"] == "ready" for st in items),
            "is_downloading": self._is_downloading,
            "total_bytes": total_bytes,
            "downloaded_bytes": downloaded_bytes,
            "percent": round(min(100.0, agg_pct), 1),
            "speed_mbps": round(self._speed_bps / (1024 * 1024), 2),
            "eta_seconds": int(self._eta_seconds),
            "models": items,
        }

    def start_download(self, on_complete: Optional[Callable[[], None]] = None) -> bool:
        if self._is_downloading:
            return False
        self._cancelled = False
        self._is_downloading = True
        self._task = asyncio.create_task(self._download_all(on_complete))
        return True

    def cancel(self):
        self._cancelled = True
        if self._task and not self._task.done():
            self._task.cancel()
        self._is_downloading = False
        self._speed_bps = 0.0
        self._eta_seconds = 0.0

    async def _download_all(self, on_complete: Optional[Callable[[], None]] = None):
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=60)
        headers = {"User-Agent": "ARIES-Public-Preview-Beta/1.0"}
        sem = asyncio.Semaphore(3)
        t_start_all = time.monotonic()
        session_bytes = 0

        async def _download_pkg(session: aiohttp.ClientSession, pkg: ModelPackageSpec):
            nonlocal session_bytes
            if self._cancelled:
                return
            present, _ = self._is_package_present(pkg)
            if present:
                self._states[pkg.id].status = "ready"
                self._states[pkg.id].downloaded_bytes = pkg.total_bytes
                return

            async with sem:
                if self._cancelled:
                    return
                st = self._states[pkg.id]
                st.status = "downloading"
                st.error = ""
                pkg_downloaded = 0

                try:
                    for f in pkg.files:
                        if self._cancelled:
                            return
                        dest = os.path.join(self.models_root, f.rel_path)
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        if os.path.exists(dest) and os.path.getsize(dest) > 0:
                            pkg_downloaded += os.path.getsize(dest)
                            st.downloaded_bytes = pkg_downloaded
                            continue

                        part = dest + ".part"
                        resume_pos = os.path.getsize(part) if os.path.exists(part) else 0
                        req_headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}

                        async with session.get(f.url, headers=req_headers, allow_redirects=True) as resp:
                            if resp.status == 416 and os.path.exists(part):
                                os.replace(part, dest)
                                pkg_downloaded += os.path.getsize(dest)
                                st.downloaded_bytes = pkg_downloaded
                                continue
                            if resp.status not in (200, 206):
                                raise RuntimeError(f"HTTP {resp.status} for {f.rel_path}")

                            mode = "ab" if resp.status == 206 and resume_pos > 0 else "wb"
                            if mode == "wb":
                                resume_pos = 0
                            file_bytes = resume_pos

                            with open(part, mode) as fh:
                                async for chunk in resp.content.iter_chunked(262_144):
                                    if self._cancelled:
                                        raise asyncio.CancelledError()
                                    fh.write(chunk)
                                    n = len(chunk)
                                    file_bytes += n
                                    session_bytes += n
                                    st.downloaded_bytes = pkg_downloaded + file_bytes
                                    elapsed = max(0.05, time.monotonic() - t_start_all)
                                    self._speed_bps = session_bytes / elapsed
                                    remaining = max(0, sum(s.total_bytes for s in self._states.values()) - sum(s.downloaded_bytes for s in self._states.values()))
                                    self._eta_seconds = (remaining / self._speed_bps) if self._speed_bps > 1024 else 0

                        os.replace(part, dest)
                        pkg_downloaded += os.path.getsize(dest)
                        st.downloaded_bytes = pkg_downloaded

                    if not self._cancelled:
                        st.status = "ready"
                        st.downloaded_bytes = pkg.total_bytes
                except asyncio.CancelledError:
                    st.status = "cancelled"
                    raise
                except Exception as e:
                    st.status = "error"
                    st.error = str(e)

        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                await asyncio.gather(*(_download_pkg(session, pkg) for pkg in MODEL_MANIFEST), return_exceptions=True)
        except asyncio.CancelledError:
            for st in self._states.values():
                if st.status == "downloading":
                    st.status = "cancelled"
        finally:
            self._is_downloading = False
            self._speed_bps = 0.0
            self._eta_seconds = 0.0
            if not self._cancelled and self.all_models_ready() and on_complete:
                try:
                    on_complete()
                except Exception:
                    pass

    def import_from_directory(self, src_dir: str, symlink: bool = True) -> dict:
        """Links or copies existing model subdirectories from a user-supplied models/ directory."""
        src_dir = os.path.expanduser(src_dir.strip())
        if not os.path.isdir(src_dir):
            return {"ok": False, "error": f"Directory not found: {src_dir}"}

        linked = []
        for sub in ("gemma-4", "embeddinggemma-2", "parakeet_06b", "phonon"):
            src_sub = os.path.join(src_dir, sub)
            dst_sub = os.path.join(self.models_root, sub)
            if os.path.exists(src_sub):
                if os.path.islink(dst_sub) or os.path.isdir(dst_sub):
                    if os.path.islink(dst_sub):
                        os.unlink(dst_sub)
                    else:
                        shutil.rmtree(dst_sub)
                if symlink:
                    os.symlink(src_sub, dst_sub)
                else:
                    shutil.copytree(src_sub, dst_sub)
                linked.append(sub)

        self.refresh_states()
        return {"ok": len(linked) > 0, "linked": linked, "status": self.get_status()}
