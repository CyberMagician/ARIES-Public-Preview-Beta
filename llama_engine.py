"""Native Embedded Vulkan llama.cpp Server Manager for Gemma Live Voice Agent.

Replaces LM Studio & Ollama completely by bundling and managing the standalone
`llama-server.exe` Vulkan AVX2 backend directly inside the app:
- 100% GPU Offloading (--n-gpu-layers 99) on AMD Radeon Pro 570 (4 GB VRAM)
- Multi-Token Prediction (MTP) Speculative Decoding (--model-draft + --spec-type draft-mtp)
- Single-User Voice Slot (--parallel 1, --ctx-size 8192) to eliminate 4-slot VRAM waste
- Canonical Gemma 4 Jinja Template for native tool-calling and reasoning_effort=none
- Automatic Win32 Job Object binding so llama-server.exe closes cleanly when the app exits
"""

import atexit
from collections import deque
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

if sys.platform == "win32":
    from ctypes import wintypes

from config import (
    GEMMA4_JINJA_TEMPLATE,
    LLAMA_ENGINE_DIR,
    LLAMA_SERVER_EXE,
    LLAMA_SERVER_PORT,
    SYSTEM_INSTRUCTION,
    resolve_model_paths,
)

# -------------------------------------------------------------------------
# Win32 Job Object Helper (Ensures llama-server.exe dies if parent Python exits)
# -------------------------------------------------------------------------
_job_handle = None


def _bind_process_to_kill_on_close_job(proc_handle: int):
    """Binds child process to a Win32 Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (no-op on non-Windows)."""
    if sys.platform != "win32":
        return
    global _job_handle
    try:
        kernel32 = ctypes.windll.kernel32
        if _job_handle is None:
            _job_handle = kernel32.CreateJobObjectW(None, None)
            if _job_handle:
                class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD),
                    ]

                class IO_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("ReadOperationCount", ctypes.c_uint64),
                        ("WriteOperationCount", ctypes.c_uint64),
                        ("OtherOperationCount", ctypes.c_uint64),
                        ("ReadTransferCount", ctypes.c_uint64),
                        ("WriteTransferCount", ctypes.c_uint64),
                        ("OtherTransferCount", ctypes.c_uint64),
                    ]

                class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                    _fields_ = [
                        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t),
                    ]

                info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
                info.BasicLimitInformation.LimitFlags = 0x00002000
                # JobObjectExtendedLimitInformation = 9
                kernel32.SetInformationJobObject(
                    _job_handle, 9, ctypes.byref(info), ctypes.sizeof(info)
                )
        if _job_handle and proc_handle:
            kernel32.AssignProcessToJobObject(_job_handle, int(proc_handle))
    except Exception:
        pass


class VulkanLlamaServerManager:
    """Manages the embedded Vulkan llama-server.exe subprocess and streams telemetry to the UI console."""

    def __init__(self, port: int = LLAMA_SERVER_PORT, idle_timeout: float = 1800.0):
        self.port = port
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.v1_url = f"{self.base_url}/v1"
        self.process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._is_ready = False
        self.idle_timeout = idle_timeout
        self.last_activity = time.time()
        self._is_active_request = False
        self._watchdog_thread: threading.Thread | None = None
        self._running_watchdog = True
        self.backend_logs: deque[str] = deque(maxlen=300)
        self.active_model_id: str | None = None
        self.active_engine: str = "llama-server"
        self._start_watchdog()
        atexit.register(self.stop)

    def _start_watchdog(self):
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return
        self._watchdog_thread = threading.Thread(target=self._idle_watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    def touch(self):
        """Marks active use so the idle timer resets."""
        self.last_activity = time.time()

    def set_active(self, active: bool):
        """Marks whether an inference or stream is actively processing."""
        self._is_active_request = active
        self.touch()

    def is_running(self) -> bool:
        """Fast zero-I/O check whether the managed llama-server subprocess is alive and ready."""
        return bool(self._is_ready and self.process is not None and self.process.poll() is None)

    def _idle_watchdog_loop(self):
        while self._running_watchdog:
            time.sleep(2.0)
            if self._is_active_request:
                continue
            with self._lock:
                if self.process and self.process.poll() is None:
                    idle_secs = time.time() - self.last_activity
                    if idle_secs >= self.idle_timeout:
                        print(f"[LLM] Idle for {idle_secs:.0f}s (>= {self.idle_timeout:.0f}s). Offloading llama-server from Radeon Pro 570 VRAM...")
                        self._is_ready = False
                        try:
                            self.process.terminate()
                            self.process.wait(timeout=2.0)
                        except Exception:
                            try:
                                self.process.kill()
                            except Exception:
                                pass
                        self.process = None
                        print("[LLM] VRAM freed! GPU entering ultra-low power state.")

    def ensure_started(self) -> bool:
        """Wakes up / starts the server on-demand before processing queries."""
        self.touch()
        if self.is_running() or self.is_healthy():
            self._is_ready = True
            return True
        print("[LLM] Waking up Vulkan llama-server on-demand...")
        ok = self.start()
        if ok:
            self.warmup_prompt_cache()
        return ok

    def _log_backend(self, level: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.backend_logs.append(f"[{ts}] [{level}] {msg}")

    def list_available_models(self) -> list[dict]:
        """Scans ARIES models/gemma-4 (*.gguf) and models/mlx (*) for switchable local models."""
        from config import ARIES_DATA_DIR
        ARIES_MODELS_DIR = os.path.join(ARIES_DATA_DIR, "models")
        models: list[dict] = []
        gguf_dir = os.path.join(ARIES_MODELS_DIR, "gemma-4")
        if os.path.isdir(gguf_dir):
            for fname in sorted(os.listdir(gguf_dir)):
                low = fname.lower()
                if low.endswith(".gguf") and not low.startswith("mmproj") and not low.startswith("mtp"):
                    fpath = os.path.join(gguf_dir, fname)
                    models.append({
                        "id": fpath,
                        "name": fname,
                        "engine": "llama-server",
                        "active": self.active_model_id == fpath or (self.active_model_id is None and self.active_engine == "llama-server"),
                    })
        mlx_dir = os.path.join(ARIES_MODELS_DIR, "mlx")
        if os.path.isdir(mlx_dir):
            for dname in sorted(os.listdir(mlx_dir)):
                dpath = os.path.join(mlx_dir, dname)
                if os.path.isdir(dpath) and os.path.exists(os.path.join(dpath, "config.json")):
                    models.append({
                        "id": dpath,
                        "name": f"{dname} (MLX)",
                        "engine": "mlx",
                        "active": self.active_model_id == dpath,
                    })
        return models

    def switch_model(self, model_id: str) -> bool:
        """Stops the active backend and starts the selected GGUF (llama-server) or MLX (mlx_lm.server) model."""
        with self._lock:
            self._is_ready = False
            if self.process and self.process.poll() is None:
                self._log_backend("INFO", f"Stopping {self.active_engine} before switching to {os.path.basename(model_id)}...")
                try:
                    self.process.terminate()
                    self.process.wait(timeout=4.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
                self.process = None

            if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "config.json")):
                self.active_engine = "mlx"
            else:
                self.active_engine = "llama-server"
            self.active_model_id = model_id
        return self.start()

    def is_healthy(self, timeout: float = 0.6) -> bool:
        endpoints = (f"{self.v1_url}/models", f"{self.base_url}/health") if self.active_engine == "mlx" else (f"{self.base_url}/health",)
        for url in endpoints:
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        return False

    def _free_conflicting_vram_processes(self):
        """Stops leftover LM Studio / Ollama processes if they are holding Radeon Pro 570 VRAM."""
        if sys.platform != "win32":
            return
        try:
            # Check if LM Studio or Ollama is running and holding GPU memory
            check = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq LM Studio.exe", "/FI", "IMAGENAME eq ollama.exe", "/NH"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            out = check.stdout or ""
            if "LM Studio.exe" in out or "ollama.exe" in out:
                print("[LLM] Reclaiming Radeon Pro 570 VRAM from background LM Studio / Ollama processes...")
                for img in ("LM Studio.exe", "ollama app.exe", "ollama.exe"):
                    subprocess.run(
                        ["taskkill", "/F", "/IM", img, "/T"],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
        except Exception:
            pass

    def start(self, status_callback=None) -> bool:
        """Starts the bundled Vulkan llama-server.exe if not already running and waits until healthy."""
        with self._lock:
            if self.is_healthy():
                self._is_ready = True
                print(f"[LLM] Embedded Vulkan llama-server already online at {self.base_url}!")
                return True

            paths = resolve_model_paths()
            if self.active_engine == "mlx" and self.active_model_id and os.path.isdir(self.active_model_id):
                model_path = self.active_model_id
                is_vlm = False
                try:
                    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as cf:
                        cfg = json.load(cf)
                    is_vlm = "vision_config" in cfg
                except Exception:
                    pass
                server_mod = "mlx_vlm.server" if is_vlm else "mlx_lm.server"
                cmd = [
                    sys.executable, "-m", server_mod,
                    "--model", model_path,
                    "--host", "127.0.0.1",
                    "--port", str(self.port),
                ]
                self._log_backend("INFO", f"Launching {server_mod}: {os.path.basename(model_path)} on 127.0.0.1:{self.port}")
                print(f"[LLM] Launching {server_mod} ({os.path.basename(model_path)}) on 127.0.0.1:{self.port}...")
                t0 = time.perf_counter()
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                threading.Thread(target=self._stream_server_logs, daemon=True).start()
                for _ in range(240):
                    if self.process.poll() is not None:
                        self._log_backend("ERROR", f"MLX server exited with code {self.process.returncode}")
                        return False
                    if self.is_healthy(timeout=0.5):
                        self._is_ready = True
                        boot_s = time.perf_counter() - t0
                        self._log_backend("INFO", f"MLX model {os.path.basename(model_path)} ready in {boot_s:.2f}s")
                        return True
                    time.sleep(0.25)
                self._log_backend("ERROR", "Timed out waiting for MLX server.")
                return False

            model_path = self.active_model_id if (self.active_model_id and os.path.isfile(self.active_model_id)) else paths["gemma4_llm"]
            self.active_model_id = model_path
            self.active_engine = "llama-server"
            mmproj_path = paths["gemma4_vision"]
            draft_path = paths["gemma4_mtp"]

            if not os.path.exists(LLAMA_SERVER_EXE):
                print(f"[ERROR] Bundled llama-server not found at {LLAMA_SERVER_EXE}")
                return False

            if not os.path.exists(model_path):
                print(f"[ERROR] Gemma 4 E2B GGUF model not found at {model_path}")
                return False

            self._free_conflicting_vram_processes()

            has_mmproj = bool(mmproj_path and os.path.exists(mmproj_path))
            default_mmproj_dev = "auto" if sys.platform == "darwin" else "Vulkan0"
            mmproj_device = os.getenv("ARIES_MMPROJ_DEVICE", default_mmproj_dev).strip()
            use_mmproj = has_mmproj and mmproj_device.lower() != "off"
            has_draft = bool(draft_path and os.path.exists(draft_path))
            default_ple_cpu = "0" if sys.platform == "darwin" else "1"
            ple_cpu_offload = os.environ.get("ARIES_PLE_CPU_OFFLOAD", default_ple_cpu).strip().lower() not in ("0", "false", "no")
            # Offloading Per-Layer Embeddings (per_layer_token_embd.weight, 1,260 MiB) to CPU RAM
            # shrinks active Gemma 4 E2B Vulkan0 weights to ~1,224 MiB so 8192 ctx + Vulkan0 SigLIP +
            # Vulkan0 MTP draft all fit simultaneously inside the 4 GB Radeon Pro 570 VRAM budget.
            is_mac = sys.platform == "darwin"
            default_ctx = "8192" if is_mac else ("4096" if (not ple_cpu_offload and use_mmproj and mmproj_device.lower() != "none") else "8192")
            ctx_size = os.getenv("ARIES_CTX_SIZE", default_ctx).strip()
            if not is_mac and not ple_cpu_offload and use_mmproj and has_draft and mmproj_device.lower() != "none":
                try:
                    if int(ctx_size) > 4096:
                        mmproj_device = "none"
                except ValueError:
                    pass
            default_load_mode = "none" if is_mac else ("none" if ple_cpu_offload else "mmap")
            load_mode = os.getenv("ARIES_LOAD_MODE", default_load_mode).strip()

            kv_type = "f16" if is_mac else "q8_0"
            mac_p_cores = "4"
            if is_mac:
                try:
                    p = int(subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True).strip())
                    mac_p_cores = str(max(2, p))
                except Exception:
                    mac_p_cores = "4"
            cmd = [
                LLAMA_SERVER_EXE,
                "--model", model_path,
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--no-webui",
                "--jinja",
                "--chat-template-file", GEMMA4_JINJA_TEMPLATE,
                "--ctx-size", ctx_size,
                "--n-gpu-layers", "99",
                "--parallel", "1",
                "--threads", "2",
                "--threads-batch", mac_p_cores if is_mac else "4",
                "--threads-http", "4" if is_mac else "2",
                "--batch-size", "2048",
                "--ubatch-size", "2048" if is_mac else "1024",
                "--cache-type-k", kv_type,
                "--cache-type-v", kv_type,
                "--fit", "off",
                "--load-mode", load_mode,
                "--prio", "3" if is_mac else "0",
                "--prio-batch", "3" if is_mac else "2",
            ]
            if is_mac:
                cmd.extend(["--flash-attn", "on"])
            else:
                cmd.extend(["--cache-reuse", "64", "--poll", "0"])

            if ple_cpu_offload:
                cmd.extend(["-ot", r"per_layer_token_embd\.weight=CPU"])

            if use_mmproj:
                cmd.extend(["--mmproj", mmproj_path])
                if mmproj_device.lower() != "auto":
                    cmd.extend(["--mmproj-device", mmproj_device])
                if mmproj_device.lower() == "none":
                    cmd.append("--no-mmproj-offload")
            if has_draft:
                cmd.extend([
                    "--model-draft", draft_path,
                    "--spec-type", "draft-mtp",
                    "--n-gpu-layers-draft", "99",
                    "--cache-type-k-draft", kv_type,
                    "--cache-type-v-draft", kv_type,
                    "--spec-draft-n-max", "3",
                    "--spec-draft-p-min", "0.15" if is_mac else "0.05",
                    "--spec-draft-threads", "2",
                    "--spec-draft-threads-batch", mac_p_cores if is_mac else "4",
                    "--spec-draft-prio", "3" if is_mac else "0",
                    "--spec-draft-prio-batch", "3" if is_mac else "2",
                ])

            if use_mmproj and has_draft:
                mode_desc = f"Gemma 4 E2B + SigLIP ({mmproj_device}) + MTP Speculative"
            elif use_mmproj:
                mode_desc = f"Gemma 4 E2B + SigLIP ({mmproj_device})"
            elif has_draft:
                mode_desc = "Gemma 4 E2B + MTP Speculative"
            else:
                mode_desc = "Gemma 4 E2B"
            backend_label = "Apple Silicon Metal" if sys.platform == "darwin" else "Vulkan"
            gpu_label = "Apple Silicon GPU" if sys.platform == "darwin" else "Radeon Pro 570"
            if status_callback:
                status_callback(f"Starting {backend_label} llama-server ({mode_desc})...")

            print(
                f"[LLM] Launching bundled {backend_label} llama-server on 127.0.0.1:{self.port} "
                f"(GPU={gpu_label}, ngl=99, ctx={ctx_size}, kv={kv_type}, ple={'CPU' if ple_cpu_offload else 'GPU'}, load={load_mode}, vision={use_mmproj}:{mmproj_device if use_mmproj else 'off'}, spec={'draft-mtp' if has_draft else 'none'})..."
            )

            t0 = time.perf_counter()
            env = os.environ.copy()
            if sys.platform == "win32":
                env["VK_LOADER_LAYERS_DISABLE"] = "~all~"

            popen_kwargs = {}
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            self.process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(LLAMA_SERVER_EXE),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **popen_kwargs,
            )

            if sys.platform == "win32" and getattr(self.process, "_handle", None):
                _bind_process_to_kill_on_close_job(int(self.process._handle))

            # Background reader thread to surface key llama-server telemetry in the unified console
            threading.Thread(target=self._stream_server_logs, daemon=True).start()

            # Poll /health until model is loaded in VRAM (~4 to 7 seconds)
            for _ in range(120):
                if self.process.poll() is not None:
                    print(f"[ERROR] Vulkan llama-server exited early with code {self.process.returncode}")
                    return False
                if self.is_healthy(timeout=0.4):
                    self._is_ready = True
                    boot_s = time.perf_counter() - t0
                    backend_name = "Apple Silicon Metal GPU" if sys.platform == "darwin" else "Vulkan GPU"
                    print(f"[LLM] Gemma 4 E2B IT QAT + MTP Draft loaded on {backend_name} in {boot_s:.2f}s!")
                    return True
                time.sleep(0.25)

            print("[ERROR] Timed out waiting for Vulkan llama-server to become healthy.")
            return False

    def _stream_server_logs(self):
        """Filters llama-server stdout/stderr and prints concise [LLM] telemetry to the UI console."""
        if not self.process or not self.process.stdout:
            return
        try:
            for raw_line in self.process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                self._log_backend(self.active_engine.upper(), line)
                if "listening on http" in line:
                    print(f"[LLM] {line.split('srv')[-1].strip()}")
                elif "common_speculative_init_result: loading draft model" in line:
                    print("[LLM] Loading MTP speculative draft model (mtp-gemma-4-E2B-it-Q4_0.gguf)...")
                elif "print_timing:" in line and ("eval time =" in line or "draft acceptance =" in line):
                    cleaned = line.split("print_timing:")[-1].strip()
                    print(f"[LLM] {cleaned}")
                elif "ErrorOutOfDeviceMemory" in line or "failed to load model" in line:
                    print(f"[ERROR] [LLM] {line}")
        except Exception:
            pass

    def warmup_prompt_cache(self):
        """Pre-evaluates default SYSTEM_INSTRUCTION + User Profile into slot 0 KV cache for instant first-turn TTFT."""
        if not self.is_healthy():
            return
        try:
            t_warm = time.perf_counter()
            from rag_memory import memory_store
            profile_prompt = memory_store.format_user_profile_prompt()
            if SYSTEM_INSTRUCTION and profile_prompt:
                sys_content = f"{SYSTEM_INSTRUCTION}\n\n{profile_prompt}"
            else:
                sys_content = SYSTEM_INSTRUCTION or profile_prompt
            warm_messages = ([{"role": "system", "content": sys_content}] if sys_content else []) + [
                {"role": "user", "content": "Hi."},
            ]
            payload = json.dumps({
                "model": "google/gemma-4-e2b-it-qat",
                "messages": warm_messages,
                "max_tokens": 2,
                "temperature": 0.0,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
                "id_slot": 0,
                "cache_prompt": True,
                "stream": False,
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.v1_url}/chat/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=25.0) as resp:
                resp.read()
            warm_ms = (time.perf_counter() - t_warm) * 1000.0
            print(f"[WARMUP] Pre-cached Gemma 4 default prompt in Vulkan KV slot 0 in {warm_ms:.0f} ms!")
        except Exception as e:
            print(f"[WARMUP] LLM prompt cache warmup note: {e}")

    def stop(self):
        """Cleanly terminates the bundled llama-server.exe process."""
        with self._lock:
            self._is_ready = False
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=2.0)
                except Exception:
                    try:
                        self.process.kill()
                    except Exception:
                        pass
            self.process = None


# Singleton instance managed by web_app.py
llama_manager = VulkanLlamaServerManager()
