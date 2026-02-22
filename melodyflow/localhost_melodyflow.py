from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torchaudio
import time
import tempfile
import os
import json
import inspect
import threading
import subprocess
import shutil
import select
import base64
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional
from audiocraft.models import MelodyFlow
import gc
from variations import VARIATIONS
import logging
from contextlib import contextmanager
from huggingface_hub import hf_hub_download

try:
    import redis
except Exception:
    redis = None  # type: ignore

# Use a consistent shared temp directory
SHARED_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gary4juce_shared")
os.makedirs(SHARED_TEMP_DIR, exist_ok=True)

_redis_disabled = os.environ.get("MELODYFLOW_DISABLE_REDIS", "0") == "1"
redis_client = None
if redis is not None and not _redis_disabled:
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

PROGRESS_FILE_PREFIX = "melodyflow_progress_"
PROGRESS_TTL_SECONDS = 3600
_progress_lock = threading.Lock()
_last_progress_percent = {}

def _progress_file_path(session_id: str) -> str:
    safe_session = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in ("-", "_"))
    if not safe_session:
        safe_session = "unknown"
    return os.path.join(SHARED_TEMP_DIR, f"{PROGRESS_FILE_PREFIX}{safe_session}.json")

def _build_queue_status(status: str, message: str, progress_percent: int = None) -> dict:
    payload = {
        "status": status,
        "message": message,
        "position": 0,
        "total_queued": 0,
        "estimated_time": None,
        "estimated_seconds": 0,
        "source": "melodyflow-localhost",
        "phase": "transform",
    }
    if progress_percent is not None:
        payload["progress"] = progress_percent
    return payload

def _write_progress_snapshot(session_id: str, *, progress: int, status: str, message: str, error: str = None) -> None:
    if not session_id:
        return

    queue_status = _build_queue_status(status, message, progress)
    snapshot = {
        "session_id": session_id,
        "progress": progress,
        "status": status,
        "queue_status": queue_status,
        "updated_at": time.time(),
    }
    if error:
        snapshot["error"] = error
        queue_status["error"] = error

    # Always write to shared temp so g4l_localhost can relay progress even without Redis.
    progress_path = _progress_file_path(session_id)
    tmp_path = f"{progress_path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f)
        os.replace(tmp_path, progress_path)
    except Exception as e:
        print(f"Progress snapshot write failed: {e}")

    # Best-effort Redis writes for compatibility if Redis is installed/running.
    if redis_client is None:
        return

    try:
        redis_client.setex(f"progress:{session_id}", PROGRESS_TTL_SECONDS, str(progress))
        redis_client.setex(f"status:{session_id}", PROGRESS_TTL_SECONDS, json.dumps({"status": status, "error": error} if error else {"status": status}))
        redis_client.setex(f"queue_status:{session_id}", PROGRESS_TTL_SECONDS, json.dumps(queue_status))
    except Exception as e:
        print(f"Redis progress update failed: {e}")

def _set_progress(session_id: str, progress_percent: int, status: str = "processing", message: str = None, error: str = None) -> None:
    if not session_id:
        return
    progress_percent = max(0, min(100, int(progress_percent)))
    if message is None:
        message = f"transforming... {progress_percent}%"
    _write_progress_snapshot(
        session_id,
        progress=progress_percent,
        status=status,
        message=message,
        error=error,
    )

def init_progress_tracking(session_id: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent[session_id] = -1
    _set_progress(session_id, 0, status="processing", message="transforming... 0%")

def finalize_progress_tracking(session_id: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent[session_id] = 100
    _set_progress(session_id, 100, status="processing", message="transforming... 100%")

def fail_progress_tracking(session_id: str, error_message: str) -> None:
    if not session_id:
        return
    with _progress_lock:
        _last_progress_percent.pop(session_id, None)
    _write_progress_snapshot(
        session_id,
        progress=0,
        status="failed",
        message=str(error_message),
        error=str(error_message),
    )

# Add progress callback that writes to shared snapshot + Redis (best effort).
def redis_progress_callback(session_id, current, total):
    if not session_id or not total:
        return

    try:
        progress_percent = int((float(current) / float(total)) * 100)
    except Exception:
        return

    # Keep completion ownership in the JUCE session worker.
    progress_percent = max(0, min(99, progress_percent))
    with _progress_lock:
        last_percent = _last_progress_percent.get(session_id)
        if last_percent == progress_percent:
            return
        _last_progress_percent[session_id] = progress_percent

    _set_progress(session_id, progress_percent, status="processing")

JUCE_SESSION_TTL_SECONDS = 3600
_juce_session_lock = threading.Lock()
_juce_sessions = {}
MODEL_PREDOWNLOAD_TTL_SECONDS = 3600
_model_predownload_lock = threading.Lock()
_model_predownload_sessions = {}

MELODYFLOW_MODEL_REPO = os.environ.get("MELODYFLOW_MODEL_REPO", "facebook/melodyflow-t24-30secs")
MELODYFLOW_REQUIRED_FILES = (
    "state_dict.bin",
    "compression_state_dict.bin",
)
HF_DOWNLOADER_PYTHON_VERSION = "3.11"
HF_DOWNLOADER_PACKAGES = [
    "huggingface_hub==1.4.1",
    "hf_xet==1.2.0",
    "tqdm==4.67.2",
]
HF_DOWNLOADER_VENV_DIR = os.environ.get(
    "G4L_HF_DOWNLOADER_VENV_DIR",
    os.path.join(
        os.path.expanduser("~"),
        "Library",
        "Application Support",
        "GaryLocalhost",
        "venvs",
        "hf-downloader",
    ),
)
HF_DOWNLOADER_MARKER_FILE = ".g4l_hf_downloader_spec.json"
HF_DOWNLOADER_SPEC_VERSION = 1
_hf_downloader_lock = threading.Lock()
_hf_downloader_python_path: Optional[str] = None
HF_DOWNLOADER_XET_MODE = str(os.environ.get("G4L_HF_DOWNLOADER_XET_MODE", "adaptive")).strip().lower()
HF_DOWNLOADER_XET_HIGH_PERFORMANCE = os.environ.get("G4L_HF_DOWNLOADER_XET_HIGH_PERFORMANCE", "1")
HF_DOWNLOADER_XET_NUM_CONCURRENT_RANGE_GETS = str(
    os.environ.get("G4L_HF_DOWNLOADER_XET_NUM_CONCURRENT_RANGE_GETS", "64")
).strip()
try:
    HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS = max(
        5.0, float(str(os.environ.get("G4L_HF_XET_FIRST_BYTE_TIMEOUT_SECONDS", "25")).strip())
    )
except Exception:
    HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS = 25.0
try:
    HF_DOWNLOADER_XET_SLOW_SPEED_BPS = max(
        64 * 1024,
        int(str(os.environ.get("G4L_HF_XET_SLOW_SPEED_BPS", str(1 * 1024 * 1024))).strip()),
    )
except Exception:
    HF_DOWNLOADER_XET_SLOW_SPEED_BPS = 1 * 1024 * 1024
try:
    HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS = max(
        5.0, float(str(os.environ.get("G4L_HF_XET_SLOW_SPEED_GRACE_SECONDS", "45")).strip())
    )
except Exception:
    HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS = 45.0

try:
    _HF_HUB_DOWNLOAD_SUPPORTS_TQDM_CLASS = (
        "tqdm_class" in inspect.signature(hf_hub_download).parameters
    )
except Exception:
    _HF_HUB_DOWNLOAD_SUPPORTS_TQDM_CLASS = False


class _NullTqdmStream:
    def write(self, message: str) -> int:
        return len(message) if message is not None else 0

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False


_NULL_TQDM_STREAM = _NullTqdmStream()


def _fmt_bytes(n: int) -> str:
    f = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024.0
    return f"{f:.1f}TB"


def _resolve_existing_uv_path() -> Optional[str]:
    candidates = []
    found_in_path = shutil.which("uv")
    if found_in_path:
        candidates.append(found_in_path)

    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, ".local", "bin", "uv"),
        os.path.join(home, "Library", "Application Support", "gary4local", "tools", "uv", "uv"),
        os.path.join(home, "Library", "Application Support", "gary4local", "tools", "uv", "bin", "uv"),
        os.path.join(home, "Library", "Application Support", "GaryLocalhost", "tools", "uv", "uv"),
        os.path.join(home, "Library", "Application Support", "GaryLocalhost", "tools", "uv", "bin", "uv"),
        "/opt/homebrew/bin/uv",
        "/usr/local/bin/uv",
    ])

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _bootstrap_uv_if_needed() -> str:
    existing = _resolve_existing_uv_path()
    if existing:
        return existing

    home = os.path.expanduser("~")
    install_dir = os.path.join(
        home,
        "Library",
        "Application Support",
        "gary4local",
        "tools",
        "uv",
    )
    os.makedirs(install_dir, exist_ok=True)

    env = os.environ.copy()
    env["UV_UNMANAGED_INSTALL"] = install_dir
    env["PATH"] = env.get("PATH") or "/usr/bin:/bin:/usr/sbin:/sbin"

    proc = subprocess.run(
        ["/bin/sh", "-lc", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"uv bootstrap failed (exit {proc.returncode}): {(proc.stderr or proc.stdout or '').strip()}"
        )

    resolved = _resolve_existing_uv_path()
    if not resolved:
        raise RuntimeError("uv bootstrap completed but uv executable was not found.")
    logger.info("HF downloader bootstrap: uv installed at %s", resolved)
    return resolved


def _run_command_checked(args: list[str], *, env: Optional[dict] = None) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, env=env)
    if proc.returncode == 0:
        return
    message = (proc.stderr or proc.stdout or "").strip()
    raise RuntimeError(
        f"command failed (exit {proc.returncode}): {' '.join(args)}"
        + (f" | {message}" if message else "")
    )


def _ensure_hf_downloader_python() -> str:
    global _hf_downloader_python_path

    with _hf_downloader_lock:
        if _hf_downloader_python_path and os.path.exists(_hf_downloader_python_path):
            return _hf_downloader_python_path

        uv_path = _bootstrap_uv_if_needed()
        venv_dir = HF_DOWNLOADER_VENV_DIR
        marker_path = os.path.join(venv_dir, HF_DOWNLOADER_MARKER_FILE)
        desired_marker = {
            "version": HF_DOWNLOADER_SPEC_VERSION,
            "python": HF_DOWNLOADER_PYTHON_VERSION,
            "packages": HF_DOWNLOADER_PACKAGES,
        }
        venv_python = os.path.join(venv_dir, "bin", "python")

        os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
        if not os.path.exists(venv_python):
            _run_command_checked([uv_path, "python", "install", HF_DOWNLOADER_PYTHON_VERSION])
            _run_command_checked([
                uv_path,
                "venv",
                "--python",
                HF_DOWNLOADER_PYTHON_VERSION,
                "--seed",
                venv_dir,
            ])

        needs_install = True
        if os.path.exists(marker_path):
            try:
                with open(marker_path, "r", encoding="utf-8") as f:
                    marker_payload = json.load(f)
                needs_install = marker_payload != desired_marker
            except Exception:
                needs_install = True

        if needs_install:
            _run_command_checked([
                uv_path, "pip", "install", "--python", venv_python,
                "--upgrade", "pip", "setuptools", "wheel"
            ])
            _run_command_checked(
                [uv_path, "pip", "install", "--python", venv_python, "--upgrade"]
                + HF_DOWNLOADER_PACKAGES
            )
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(desired_marker, f)

        _hf_downloader_python_path = venv_python
        logger.info("HF downloader env ready: %s", venv_python)
        return venv_python


def _download_with_shared_hf_downloader_env(
    *,
    model_name: str,
    filename: str,
    on_progress: Callable[[dict], None],
) -> str:
    def resolve_mode() -> str:
        mode = HF_DOWNLOADER_XET_MODE
        if mode in {"on", "off", "adaptive"}:
            return mode
        return "adaptive"

    def terminate_process(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def run_worker(
        *,
        use_xet: bool,
        force_download: bool = False,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        backend_label = "shared downloader env (xet)" if use_xet else "shared downloader env (http)"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if use_xet:
            env["HF_HUB_DISABLE_XET"] = "0"
            env["HF_XET_HIGH_PERFORMANCE"] = HF_DOWNLOADER_XET_HIGH_PERFORMANCE
            if HF_DOWNLOADER_XET_NUM_CONCURRENT_RANGE_GETS:
                env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = HF_DOWNLOADER_XET_NUM_CONCURRENT_RANGE_GETS
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        else:
            env["HF_HUB_DISABLE_XET"] = "1"
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        worker_cmd = [downloader_python, worker_path, "--repo-id", model_name, "--filename", filename]
        if force_download:
            worker_cmd.append("--force-download")

        process = subprocess.Popen(
            worker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        if process.stdout is None:
            terminate_process(process)
            raise RuntimeError("failed to capture downloader worker output.")

        worker_error = None
        started_at = time.time()
        first_byte_at = None
        slow_since = None

        while True:
            if use_xet:
                now = time.time()
                if first_byte_at is None and (now - started_at) > HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS:
                    terminate_process(process)
                    logger.warning(
                        "shared downloader env xet timeout: no first byte for %s after %.1fs",
                        filename,
                        HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS,
                    )
                    return False, "xet_no_first_byte", backend_label
                if first_byte_at is not None and slow_since is not None and (
                    now - slow_since
                ) > HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS:
                    terminate_process(process)
                    logger.warning(
                        "shared downloader env xet slow throughput for %s: < %s B/s for %.1fs",
                        filename,
                        HF_DOWNLOADER_XET_SLOW_SPEED_BPS,
                        HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS,
                    )
                    return False, "xet_slow_throughput", backend_label

            ready, _, _ = select.select([process.stdout], [], [], 0.4)
            if not ready:
                if process.poll() is not None:
                    break
                continue

            raw_line = process.stdout.readline()
            if raw_line == "":
                if process.poll() is not None:
                    break
                continue

            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except Exception:
                continue

            event = str(payload.get("event") or "").lower()
            if event == "progress":
                on_progress(payload)
                downloaded = int(payload.get("downloaded_bytes") or 0)
                speed_bps = float(payload.get("speed_bps") or 0.0)
                if downloaded > 0 and first_byte_at is None:
                    first_byte_at = time.time()
                if use_xet and downloaded > 0:
                    if speed_bps >= HF_DOWNLOADER_XET_SLOW_SPEED_BPS:
                        slow_since = None
                    else:
                        if slow_since is None:
                            slow_since = time.time()
            elif event == "error":
                worker_error = str(payload.get("error") or "unknown downloader error")

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(worker_error or f"downloader worker failed with exit {return_code}.")
        return True, None, backend_label

    downloader_python = _ensure_hf_downloader_python()
    worker_path = os.path.join(os.path.dirname(__file__), "hf_predownload_worker.py")
    if not os.path.exists(worker_path):
        raise RuntimeError(f"downloader worker is missing: {worker_path}")

    mode = resolve_mode()
    if mode == "off":
        success, _, backend_label = run_worker(use_xet=False)
        if not success:
            raise RuntimeError("shared downloader env failed in HTTP mode.")
        return backend_label or "shared downloader env (http)"

    if mode == "on":
        success, _, backend_label = run_worker(use_xet=True)
        if not success:
            raise RuntimeError("shared downloader env failed in forced XET mode.")
        return backend_label or "shared downloader env (xet)"

    # adaptive (default): try xet first, then fallback to http on bad TTFB/speed
    success, fallback_reason, backend_label = run_worker(use_xet=True)
    if success:
        return backend_label or "shared downloader env (xet)"

    logger.warning(
        "shared downloader env adaptive fallback for %s/%s: %s -> http",
        model_name,
        filename,
        fallback_reason or "unknown reason",
    )
    success_http, _, backend_label_http = run_worker(
        use_xet=False,
        force_download=True,
    )
    if not success_http:
        raise RuntimeError("shared downloader env failed in HTTP fallback mode.")
    return backend_label_http or "shared downloader env (http fallback)"


def _make_hf_tqdm_class(
    *,
    model_name: str,
    filename: str,
    stage_index: int,
    stage_total: int,
    on_progress: Optional[Callable[[dict], None]],
):
    if on_progress is None:
        return None

    try:
        from huggingface_hub.utils import tqdm as hf_tqdm
    except Exception:
        return None

    class _CallbackTqdm(hf_tqdm):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            self._last_emit_t = 0.0
            self._last_emit_pct = -1
            self._last_emit_bytes = 0
            kwargs.setdefault("file", _NULL_TQDM_STREAM)
            kwargs.setdefault("leave", False)
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._emit(force=True)

        def update(self, n=1):
            out = super().update(n)
            self._emit()
            return out

        def refresh(self, *args, **kwargs):
            out = super().refresh(*args, **kwargs)
            self._emit()
            return out

        def set_description(self, desc=None, refresh=True):
            out = super().set_description(desc, refresh=refresh)
            self._emit(force=True)
            return out

        def close(self):
            self._emit(force=True)
            return super().close()

        def display(self, msg=None, pos=None):
            return None

        def _emit(self, force: bool = False) -> None:
            now = time.time()
            total = int(getattr(self, "total", 0) or 0)
            downloaded = int(getattr(self, "n", 0) or 0)
            pct = int((downloaded / total) * 100) if total > 0 else 0

            if not force:
                if (now - self._last_emit_t) < 0.25:
                    return
                pct_delta_ok = self._last_emit_pct < 0 or abs(pct - self._last_emit_pct) >= 1
                bytes_delta_ok = (downloaded - self._last_emit_bytes) >= (1 * 1024 * 1024)
                if not (pct_delta_ok or bytes_delta_ok):
                    return

            self._last_emit_t = now
            self._last_emit_pct = pct
            self._last_emit_bytes = downloaded
            rate = None
            try:
                rate = float((getattr(self, "format_dict", {}) or {}).get("rate") or 0.0)
            except Exception:
                rate = 0.0
            on_progress({
                "repo_id": model_name,
                "filename": filename,
                "stage_index": stage_index,
                "stage_total": stage_total,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "download_percent": max(0, min(100, pct)),
                "speed_bps": max(0.0, float(rate or 0.0)),
            })

    return _CallbackTqdm


@contextmanager
def _patched_hf_download_tqdm(tqdm_class):
    if tqdm_class is None:
        yield
        return
    try:
        import huggingface_hub.file_download as hf_file_download
    except Exception:
        yield
        return

    original_tqdm = getattr(hf_file_download, "tqdm", None)
    if original_tqdm is None:
        yield
        return

    hf_file_download.tqdm = tqdm_class
    try:
        yield
    finally:
        hf_file_download.tqdm = original_tqdm


@contextmanager
def _temporary_env(overrides: Optional[dict[str, str]]):
    if not overrides:
        yield
        return

    previous = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _hf_hub_download_safe(**kwargs):
    try:
        return hf_hub_download(**kwargs)
    except TypeError:
        kwargs.pop("force_download", None)
        return hf_hub_download(**kwargs)


def _is_checkpoint_like_filename(filename: str) -> bool:
    lower = str(filename or "").strip().lower()
    return lower.endswith((".bin", ".ckpt", ".pt", ".pth", ".safetensors"))


def _looks_like_corrupt_checkpoint_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    markers = (
        "pytorchstreamreader failed",
        "invalid header",
        "archive is corrupted",
        "filename 'storages' not found",
        "cannot use ``weights_only=true``",
        "legacy .tar format",
        "unable to parse checkpoint",
    )
    return any(marker in text for marker in markers)


def _repair_melodyflow_model_cache() -> None:
    runtime_profiles = [
        {},
        {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
        {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"},
    ]

    for filename in MELODYFLOW_REQUIRED_FILES:
        downloaded = False
        errors = []
        for profile in runtime_profiles:
            profile_label = (
                "default"
                if not profile
                else ", ".join(f"{k}={v}" for k, v in profile.items())
            )
            try:
                with _temporary_env(profile):
                    _hf_hub_download_safe(
                        repo_id=MELODYFLOW_MODEL_REPO,
                        filename=filename,
                        force_download=True,
                    )
                downloaded = True
                break
            except Exception as download_error:
                errors.append(f"{profile_label}: {download_error}")

        if not downloaded:
            raise RuntimeError(
                f"failed to repair melodyflow cache for {filename}: {' | '.join(errors)}"
            )


def _cleanup_expired_juce_sessions(now_ts: float) -> None:
    expired = [
        sid for sid, payload in _juce_sessions.items()
        if float(payload.get("expires_at", 0)) <= now_ts
    ]
    for sid in expired:
        _juce_sessions.pop(sid, None)


def _upsert_juce_session(session_id: str, **updates) -> None:
    if not session_id:
        return
    now_ts = time.time()
    with _juce_session_lock:
        _cleanup_expired_juce_sessions(now_ts)
        payload = dict(_juce_sessions.get(session_id) or {})
        payload.update(updates)
        payload["updated_at"] = now_ts
        payload["expires_at"] = now_ts + JUCE_SESSION_TTL_SECONDS
        _juce_sessions[session_id] = payload


def _get_juce_session(session_id: str):
    if not session_id:
        return None
    now_ts = time.time()
    with _juce_session_lock:
        _cleanup_expired_juce_sessions(now_ts)
        payload = _juce_sessions.get(session_id)
        return dict(payload) if payload else None


def _cleanup_expired_model_predownload_sessions(now_ts: float) -> None:
    expired = [
        sid for sid, payload in _model_predownload_sessions.items()
        if float(payload.get("expires_at", 0)) <= now_ts
    ]
    for sid in expired:
        _model_predownload_sessions.pop(sid, None)


def _upsert_model_predownload_session(session_id: str, **updates) -> None:
    if not session_id:
        return
    now_ts = time.time()
    with _model_predownload_lock:
        _cleanup_expired_model_predownload_sessions(now_ts)
        payload = dict(_model_predownload_sessions.get(session_id) or {})
        payload.update(updates)
        payload["updated_at"] = now_ts
        payload["expires_at"] = now_ts + MODEL_PREDOWNLOAD_TTL_SECONDS
        _model_predownload_sessions[session_id] = payload


def _get_model_predownload_session(session_id: str):
    if not session_id:
        return None
    now_ts = time.time()
    with _model_predownload_lock:
        _cleanup_expired_model_predownload_sessions(now_ts)
        payload = _model_predownload_sessions.get(session_id)
        return dict(payload) if payload else None


def _model_catalog_payload() -> dict:
    return {
        "small": [],
        "medium": [],
        "large": [
            {
                "name": "melodyflow-t24-30secs",
                "path": MELODYFLOW_MODEL_REPO,
                "type": "single",
            }
        ],
    }


def _model_download_status_for(model_name: str) -> dict:
    missing = []
    for filename in MELODYFLOW_REQUIRED_FILES:
        try:
            hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_files_only=True,
            )
        except Exception:
            missing.append(filename)

    return {
        "downloaded": len(missing) == 0,
        "missing": missing,
    }


def _build_model_queue_status(
    *,
    status: str,
    message: str,
    model_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
) -> dict:
    payload = {
        "status": status,
        "message": message,
        "position": 0,
        "total_queued": 0,
        "estimated_time": None,
        "estimated_seconds": 0,
        "source": "melodyflow-localhost",
        "phase": "download",
        "repo_id": model_name,
        "download_percent": max(0, min(100, int(download_percent))),
        "stage_index": max(0, int(stage_index)),
        "stage_total": max(0, int(stage_total)),
        "unit": "files",
    }
    return payload


def _run_model_predownload(session_id: str, model_name: str) -> None:
    def worker():
        stage_total = len(MELODYFLOW_REQUIRED_FILES)
        try:
            _upsert_model_predownload_session(
                session_id,
                model_name=model_name,
                status="warming",
                progress=0,
                queue_status=_build_model_queue_status(
                    status="warming",
                    message=f"preparing download for {model_name}",
                    model_name=model_name,
                    stage_index=0,
                    stage_total=stage_total,
                    download_percent=0,
                ),
                error=None,
            )

            for stage_index, filename in enumerate(MELODYFLOW_REQUIRED_FILES, start=1):
                stage_start_progress = int(((stage_index - 1) / max(stage_total, 1)) * 100)
                stage_status = "warming" if stage_index == 1 else "processing"
                download_backend_label = "shared downloader env"
                _upsert_model_predownload_session(
                    session_id,
                    status=stage_status,
                    progress=stage_start_progress,
                    queue_status=_build_model_queue_status(
                        status=stage_status,
                        message=f"stage {stage_index}/{stage_total}: downloading {filename}",
                        model_name=model_name,
                        stage_index=stage_index,
                        stage_total=stage_total,
                        download_percent=0,
                    ),
                    error=None,
                )

                def on_progress(evt: dict) -> None:
                    stage_percent = int(evt.get("download_percent") or evt.get("percent") or 0)
                    downloaded = int(evt.get("downloaded_bytes") or 0)
                    total = int(evt.get("total_bytes") or 0)
                    speed_bps = float(evt.get("speed_bps") or 0.0)
                    overall_progress_raw = (
                        (
                            (stage_index - 1)
                            + (max(0, min(100, stage_percent)) / 100.0)
                        )
                        / max(stage_total, 1)
                        * 100
                    )
                    overall_progress = int(overall_progress_raw + 0.9999)
                    if downloaded > 0:
                        overall_progress = max(stage_start_progress + 1, overall_progress)
                    overall_progress = max(0, min(99, overall_progress))
                    message = (
                        f"stage {stage_index}/{stage_total}: downloading {filename} "
                        f"via {download_backend_label} "
                        f"({_fmt_bytes(downloaded)}/{_fmt_bytes(total)} • {stage_percent}%"
                        f"{(' • ' + _fmt_bytes(int(speed_bps)) + '/s') if speed_bps > 0 else ''})"
                    )
                    _upsert_model_predownload_session(
                        session_id,
                        status=stage_status,
                        progress=max(stage_start_progress, overall_progress),
                        queue_status={
                            **_build_model_queue_status(
                                status=stage_status,
                                message=message,
                                model_name=model_name,
                                stage_index=stage_index,
                                stage_total=stage_total,
                                download_percent=stage_percent,
                            ),
                            "downloaded_bytes": downloaded,
                            "total_bytes": total,
                            "speed_bps": speed_bps,
                        },
                        error=None,
                    )

                prefer_runtime_downloader = _is_checkpoint_like_filename(filename)
                downloaded_with_shared_env = False
                if prefer_runtime_downloader:
                    download_backend_label = "runtime checkpoint path"
                    logger.info(
                        "model predownload session %s: forcing runtime downloader for %s",
                        session_id,
                        filename,
                    )
                else:
                    try:
                        _upsert_model_predownload_session(
                            session_id,
                            status=stage_status,
                            progress=stage_start_progress,
                            queue_status=_build_model_queue_status(
                                status=stage_status,
                                message=f"stage {stage_index}/{stage_total}: preparing shared downloader env for {filename}",
                                model_name=model_name,
                                stage_index=stage_index,
                                stage_total=stage_total,
                                download_percent=0,
                            ),
                            error=None,
                        )
                        selected_backend_label = _download_with_shared_hf_downloader_env(
                            model_name=model_name,
                            filename=filename,
                            on_progress=on_progress,
                        )
                        if selected_backend_label:
                            download_backend_label = selected_backend_label
                            logger.info(
                                "model predownload session %s: %s selected for %s",
                                session_id,
                                selected_backend_label,
                                filename,
                            )
                        downloaded_with_shared_env = True
                    except Exception as shared_download_error:
                        download_backend_label = "runtime fallback"
                        logger.warning(
                            "Shared downloader env failed for %s/%s; falling back to runtime env: %s",
                            model_name,
                            filename,
                            shared_download_error,
                        )
                        _upsert_model_predownload_session(
                            session_id,
                            status=stage_status,
                            progress=stage_start_progress,
                            queue_status=_build_model_queue_status(
                                status=stage_status,
                                message=(
                                    f"stage {stage_index}/{stage_total}: shared downloader failed, "
                                    f"falling back to runtime env for {filename}"
                                ),
                                model_name=model_name,
                                stage_index=stage_index,
                                stage_total=stage_total,
                                download_percent=0,
                            ),
                            error=None,
                        )
                        logger.info(
                            "model predownload session %s: runtime fallback selected for %s",
                            session_id,
                            filename,
                        )

                if not downloaded_with_shared_env:
                    tqdm_class = _make_hf_tqdm_class(
                        model_name=model_name,
                        filename=filename,
                        stage_index=stage_index,
                        stage_total=stage_total,
                        on_progress=on_progress,
                    )

                    runtime_profiles = (
                        [
                            {},
                            {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "1"},
                            {"HF_HUB_DISABLE_XET": "1", "HF_HUB_ENABLE_HF_TRANSFER": "0"},
                        ]
                        if prefer_runtime_downloader
                        else [{}]
                    )
                    attempt_errors = []
                    downloaded = False
                    for attempt_index, env_profile in enumerate(runtime_profiles, start=1):
                        profile_label = (
                            "default"
                            if not env_profile
                            else ", ".join(f"{k}={v}" for k, v in env_profile.items())
                        )
                        try:
                            if len(runtime_profiles) > 1:
                                _upsert_model_predownload_session(
                                    session_id,
                                    status=stage_status,
                                    progress=stage_start_progress,
                                    queue_status=_build_model_queue_status(
                                        status=stage_status,
                                        message=(
                                            f"stage {stage_index}/{stage_total}: runtime download attempt "
                                            f"{attempt_index}/{len(runtime_profiles)} ({profile_label})"
                                        ),
                                        model_name=model_name,
                                        stage_index=stage_index,
                                        stage_total=stage_total,
                                        download_percent=0,
                                    ),
                                    error=None,
                                )

                            with _temporary_env(env_profile):
                                kwargs = {
                                    "repo_id": model_name,
                                    "filename": filename,
                                }
                                if prefer_runtime_downloader:
                                    kwargs["force_download"] = True

                                if _HF_HUB_DOWNLOAD_SUPPORTS_TQDM_CLASS:
                                    if tqdm_class is not None:
                                        kwargs["tqdm_class"] = tqdm_class
                                    _hf_hub_download_safe(**kwargs)
                                else:
                                    # huggingface_hub<0.20 has no `tqdm_class`; patch module-level
                                    # progress bar class during this call so we still get callbacks.
                                    with _patched_hf_download_tqdm(tqdm_class):
                                        _hf_hub_download_safe(**kwargs)

                            downloaded = True
                            break
                        except Exception as runtime_download_error:
                            attempt_errors.append(
                                f"attempt {attempt_index} ({profile_label}): {runtime_download_error}"
                            )
                            logger.warning(
                                "Runtime downloader attempt %s failed for %s/%s: %s",
                                attempt_index,
                                model_name,
                                filename,
                                runtime_download_error,
                            )

                    if not downloaded:
                        raise RuntimeError(
                            f"runtime downloader failed for {model_name}/{filename}: "
                            f"{' | '.join(attempt_errors)}"
                        )

                stage_end_progress = int((stage_index / max(stage_total, 1)) * 100)
                completed = stage_index >= stage_total
                _upsert_model_predownload_session(
                    session_id,
                    status="completed" if completed else "processing",
                    progress=100 if completed else stage_end_progress,
                    queue_status=_build_model_queue_status(
                        status="completed" if completed else "processing",
                        message=(
                            f"{model_name} is ready for offline use."
                            if completed
                            else f"stage {stage_index}/{stage_total} complete."
                        ),
                        model_name=model_name,
                        stage_index=stage_index,
                        stage_total=stage_total,
                        download_percent=100,
                    ),
                    error=None,
                )
        except Exception as e:
            error_message = str(e)
            logger.error("Melodyflow predownload failed: %s", error_message)
            _upsert_model_predownload_session(
                session_id,
                status="failed",
                progress=0,
                queue_status=_build_model_queue_status(
                    status="failed",
                    message=error_message,
                    model_name=model_name,
                    stage_index=0,
                    stage_total=stage_total,
                    download_percent=0,
                ),
                error=error_message,
            )

    threading.Thread(target=worker, daemon=True).start()


def _delete_progress_snapshot(session_id: str) -> None:
    progress_path = _progress_file_path(session_id)
    try:
        if os.path.exists(progress_path):
            os.remove(progress_path)
    except Exception as e:
        logger.warning("Failed to remove stale progress snapshot %s: %s", progress_path, e)


def _read_progress_snapshot(session_id: str):
    progress_path = _progress_file_path(session_id)
    if not os.path.exists(progress_path):
        return None
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _sync_juce_session_from_snapshot(session_id: str, session_data: dict) -> dict:
    if session_data.get("status") in {"completed", "failed"}:
        return session_data

    snapshot = _read_progress_snapshot(session_id)
    if not snapshot:
        return session_data

    updates = {}
    snap_status = snapshot.get("status")
    if isinstance(snap_status, str) and snap_status:
        updates["status"] = snap_status

    snap_progress = snapshot.get("progress")
    if isinstance(snap_progress, (int, float)):
        updates["progress"] = max(0, min(100, int(snap_progress)))

    snap_queue = snapshot.get("queue_status")
    if isinstance(snap_queue, dict):
        updates["queue_status"] = snap_queue

    snap_error = snapshot.get("error")
    if snap_error:
        updates["error"] = str(snap_error)

    if not updates:
        return session_data

    _upsert_juce_session(session_id, **updates)
    merged = dict(session_data)
    merged.update(updates)
    return merged


def _shared_audio_file(prefix: str, session_id: str) -> str:
    safe_session = "".join(ch for ch in str(session_id) if ch.isalnum() or ch in ("-", "_"))
    if not safe_session:
        safe_session = "unknown"
    return os.path.join(
        SHARED_TEMP_DIR,
        f"{prefix}_{safe_session}_{uuid.uuid4().hex[:8]}.wav",
    )


def _write_base64_audio_to_file(audio_base64: str, session_id: str) -> str:
    file_path = _shared_audio_file("input", session_id)
    audio_bytes = base64.b64decode(audio_base64)
    with open(file_path, "wb") as f:
        f.write(audio_bytes)
    return file_path


def _cleanup_file(file_path: str) -> None:
    if not file_path:
        return
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        logger.warning("Failed to remove temp file %s: %s", file_path, e)


def _queue_transform_job(
    *,
    session_id: str,
    audio_data: str,
    variation: str,
    flowstep: float,
    solver: str,
    custom_prompt: str,
    backend_engine: str,
) -> None:
    def worker():
        input_path = None
        output_path = None
        input_waveform = None
        processed_waveform = None
        try:
            _delete_progress_snapshot(session_id)

            warm_message = "loading terry (first run / model warmup)"
            with _progress_lock:
                _last_progress_percent[session_id] = -1
            _set_progress(session_id, 0, status="warming", message=warm_message)

            _upsert_juce_session(
                session_id,
                status="warming",
                progress=0,
                error=None,
                queue_status=_build_queue_status("warming", warm_message, 0),
                original_audio=audio_data,
            )

            input_path = _write_base64_audio_to_file(audio_data, session_id)

            _set_progress(session_id, 0, status="processing", message="transforming... 0%")
            _upsert_juce_session(
                session_id,
                status="processing",
                progress=0,
                error=None,
                queue_status=_build_queue_status("processing", "transforming...", 0),
            )

            input_waveform = load_audio_from_file(input_path)
            processed_waveform = process_audio(
                input_waveform,
                variation,
                flowstep,
                solver,
                custom_prompt,
                session_id=session_id,
                progress_callback=redis_progress_callback,
                backend_engine=backend_engine,
            )

            output_path = _shared_audio_file("output", session_id)
            torchaudio.save(output_path, processed_waveform.cpu(), 32000)
            with open(output_path, "rb") as f:
                result_audio = base64.b64encode(f.read()).decode("utf-8")

            finalize_progress_tracking(session_id)
            _upsert_juce_session(
                session_id,
                status="completed",
                progress=100,
                error=None,
                queue_status=_build_queue_status("completed", "done", 100),
                result_audio=result_audio,
            )
        except AudioProcessingError as e:
            fail_progress_tracking(session_id, str(e))
            _upsert_juce_session(
                session_id,
                status="failed",
                progress=0,
                error=str(e),
                queue_status=_build_queue_status("failed", str(e), 0),
            )
        except Exception as e:
            error_message = f"Unexpected error: {e}"
            fail_progress_tracking(session_id, error_message)
            logger.error("Unexpected JUCE transform error: %s", e)
            _upsert_juce_session(
                session_id,
                status="failed",
                progress=0,
                error=error_message,
                queue_status=_build_queue_status("failed", error_message, 0),
            )
        finally:
            _cleanup_file(input_path)
            _cleanup_file(output_path)
            if input_waveform is not None:
                del input_waveform
            if processed_waveform is not None:
                del processed_waveform
            if os.environ.get("MELODYFLOW_UNLOAD_EACH_REQUEST", "1") == "1":
                unload_model()
            gc.collect()

    threading.Thread(target=worker, daemon=True).start()


app = Flask(__name__)
CORS(app)

# Global variables
model = None
_mlx_runtime = None
_mlx_runtime_model_id = None
_mlx_runtime_module = None

BACKEND_MPS = "mps"
BACKEND_MLX_NATIVE_TORCH_CODEC = "mlx_native_torch_codec"
BACKEND_MLX_NATIVE_MLX_CODEC = "mlx_native_mlx_codec"
_VALID_BACKEND_ENGINES = {
    BACKEND_MPS,
    BACKEND_MLX_NATIVE_TORCH_CODEC,
    BACKEND_MLX_NATIVE_MLX_CODEC,
}


def _normalize_backend_engine(raw_backend) -> str:
    if raw_backend is None:
        return BACKEND_MPS
    backend = str(raw_backend).strip().lower()
    if backend in _VALID_BACKEND_ENGINES:
        return backend
    if backend in ("", "torch"):
        return BACKEND_MPS
    raise ValueError(
        "backend_engine must be one of 'mps', 'mlx_native_torch_codec', or 'mlx_native_mlx_codec'"
    )


def _default_backend_engine() -> str:
    configured = os.environ.get("MELODYFLOW_BACKEND_ENGINE", BACKEND_MPS)
    try:
        return _normalize_backend_engine(configured)
    except ValueError:
        logger.warning(
            "Invalid MELODYFLOW_BACKEND_ENGINE='%s'; falling back to '%s'",
            configured,
            BACKEND_MPS,
        )
        return BACKEND_MPS


def _resolve_backend_engine_from_request(raw_backend) -> str:
    if raw_backend is None:
        return _default_backend_engine()
    if isinstance(raw_backend, str) and raw_backend.strip() == "":
        return _default_backend_engine()
    return _normalize_backend_engine(raw_backend)


def _mlx_dtype_name() -> str:
    raw = os.environ.get("MELODYFLOW_MLX_DTYPE", "float32").strip().lower()
    if raw in {"float16", "bfloat16", "float32"}:
        return raw
    logger.warning("Invalid MELODYFLOW_MLX_DTYPE='%s'; falling back to 'float32'", raw)
    return "float32"


def _mlx_native_prompt_mode() -> str:
    raw = os.environ.get("MELODYFLOW_MLX_NATIVE_PROMPT_MODE", "mean").strip().lower()
    if raw in {"mean", "stochastic"}:
        return raw
    logger.warning(
        "Invalid MELODYFLOW_MLX_NATIVE_PROMPT_MODE='%s'; falling back to 'mean'",
        raw,
    )
    return "mean"


def _require_mlx_runtime_module():
    global _mlx_runtime_module
    if _mlx_runtime_module is None:
        try:
            import melodyflow_mlx_edit as mlx_runtime_module
        except Exception as exc:
            raise RuntimeError(
                "Failed to import MLX runtime support. Ensure MLX dependencies are installed."
            ) from exc
        _mlx_runtime_module = mlx_runtime_module
    return _mlx_runtime_module


def _get_mlx_runtime(current_model, *, need_native_codec: bool):
    global _mlx_runtime
    global _mlx_runtime_model_id

    mlx_module = _require_mlx_runtime_module()
    runtime_model_id = id(current_model)
    if _mlx_runtime is None or _mlx_runtime_model_id != runtime_model_id:
        _mlx_runtime = mlx_module.build_runtime(current_model, dtype_name=_mlx_dtype_name())
        _mlx_runtime_model_id = runtime_model_id

    if need_native_codec:
        mlx_module.ensure_native_codec(_mlx_runtime, current_model)

    return _mlx_runtime, mlx_module


def _pick_device() -> str:
    forced_device = os.environ.get("MELODYFLOW_DEVICE", "").strip().lower()
    require_mps = os.environ.get("MELODYFLOW_REQUIRE_MPS", "0") == "1"

    if forced_device:
        if forced_device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MELODYFLOW_DEVICE=mps but MPS is not available.")
        if forced_device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("MELODYFLOW_DEVICE=cuda but CUDA is not available.")
        if forced_device == "cpu":
            return "cpu"
        raise RuntimeError(f"Unsupported MELODYFLOW_DEVICE='{forced_device}'.")

    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    if require_mps:
        raise RuntimeError("MELODYFLOW_REQUIRE_MPS=1 but no MPS device is available.")
    return 'cpu'

device = _pick_device()
DEVICE = device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('melodyflow')
logger.info("Default MelodyFlow backend engine: %s", _default_backend_engine())

from flask import after_this_request

def unload_model():
    global model
    global _mlx_runtime
    global _mlx_runtime_model_id
    if model is not None:
        try:
            del model
        except:
            pass
        model = None
    _mlx_runtime = None
    _mlx_runtime_model_id = None
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except:
            pass
        torch.cuda.empty_cache()
        # Optional extra cleanup on some systems:
        try:
            torch.cuda.ipc_collect()
        except:
            pass
    if torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except:
            pass
        try:
            torch.mps.empty_cache()
        except:
            pass
    gc.collect()

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except:
                pass
            try:
                torch.mps.empty_cache()
            except:
                pass
        gc.collect()

def load_model():
    """Initialize the MelodyFlow model."""
    global model
    if model is None:
        print("Loading MelodyFlow model...")
        try:
            model = MelodyFlow.get_pretrained(
                "facebook/melodyflow-t24-30secs", device=DEVICE
            )
        except Exception as model_error:
            if not _looks_like_corrupt_checkpoint_error(model_error):
                raise
            logger.warning(
                "MelodyFlow checkpoint load failed; repairing cache and retrying once: %s",
                model_error,
            )
            _repair_melodyflow_model_cache()
            model = MelodyFlow.get_pretrained(
                "facebook/melodyflow-t24-30secs", device=DEVICE
            )
    return model

def load_audio_from_file(file_path: str, target_sr: int = 32000) -> torch.Tensor:
    """Load and preprocess audio from file path."""
    try:
        if not os.path.exists(file_path):
            raise AudioProcessingError(f"Audio file not found: {file_path}")
        
        waveform, sr = torchaudio.load(file_path)

        # Resample if needed
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # Ensure stereo
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        return waveform.unsqueeze(0).to(device)

    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio from file: {str(e)}")

def find_max_duration(model: MelodyFlow, waveform: torch.Tensor, sr: int = 32000, max_token_length: int = 750) -> tuple:
    """Binary search to find maximum duration that produces tokens under the limit."""
    min_seconds = 1.0
    max_seconds = waveform.shape[-1] / sr
    best_duration = min_seconds
    best_tokens = None

    if max_seconds <= min_seconds:
        tokens = model.encode_audio(waveform)
        return max_seconds, tokens

    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2
        samples = max(1, int(mid_seconds * sr))
        test_waveform = waveform[..., :samples]

        try:
            tokens = model.encode_audio(test_waveform)
            token_length = tokens.shape[-1]

            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds

        except Exception as e:
            max_seconds = mid_seconds

    if best_tokens is None:
        fallback_samples = max(1, int(min_seconds * sr))
        best_tokens = model.encode_audio(waveform[..., :fallback_samples])
        best_duration = min_seconds

    if int(best_tokens.shape[-1]) > max_token_length:
        raise RuntimeError(
            f"Prompt token length {int(best_tokens.shape[-1])} exceeds max_token_length={max_token_length}"
        )

    return best_duration, best_tokens


def _normalize_waveform_for_save(tensor: torch.Tensor) -> torch.Tensor:
    x = tensor
    if x.dim() >= 3:
        x = x[0]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() != 2:
        raise RuntimeError(f"Expected output waveform with 2 dims [C,T], got shape={tuple(x.shape)}")
    return x

def process_audio(
    waveform: torch.Tensor,
    variation_name: str,
    custom_flowstep: float = None,
    solver: str = "euler",
    custom_prompt: str = None,
    session_id: str = None,
    progress_callback=None,
    backend_engine: str = None,
) -> torch.Tensor:
    """Process audio with selected variation."""

    try:
        if variation_name not in VARIATIONS:
            raise AudioProcessingError(f"Unknown variation: {variation_name}")

        backend = _resolve_backend_engine_from_request(backend_engine)
        config = VARIATIONS[variation_name].copy()
        flowstep = custom_flowstep if custom_flowstep is not None else config['default_flowstep']

        if custom_prompt is not None:
            config['prompt'] = custom_prompt

        with resource_cleanup():
            current_model = load_model()

            codec_mode = "native" if backend == BACKEND_MLX_NATIVE_MLX_CODEC else "torch"

            if backend == BACKEND_MPS:
                max_valid_duration, tokens = find_max_duration(current_model, waveform)
            else:
                need_native_codec = codec_mode == "native"
                mlx_runtime, mlx_module = _get_mlx_runtime(
                    current_model,
                    need_native_codec=need_native_codec,
                )
                if codec_mode == "native":
                    max_valid_duration, tokens = mlx_module.find_max_duration_with_encoder(
                        waveform,
                        encode_audio_fn=lambda w: mlx_module.encode_audio_with_mlx_codec(
                            mlx_runtime,
                            w,
                            current_model.device,
                        ),
                        sr=current_model.sample_rate,
                        max_token_length=750,
                    )
                else:
                    max_valid_duration, tokens = mlx_module.find_max_duration_with_encoder(
                        waveform,
                        encode_audio_fn=lambda w: current_model.encode_audio(w.to(current_model.device)),
                        sr=current_model.sample_rate,
                        max_token_length=750,
                    )

            config['duration'] = max_valid_duration

            if solver.lower() == "midpoint":
                steps = 64
                use_regularize = False
            else:
                steps = config['steps']
                use_regularize = True

            current_model.set_generation_params(
                solver=solver.lower(),
                steps=steps,
                duration=config['duration'],
            )

            current_model.set_editing_params(
                solver=solver.lower(),
                steps=steps,
                target_flowstep=flowstep,
                regularize=use_regularize,
                regularize_iters=2 if use_regularize else 0,
                keep_last_k_iters=1 if use_regularize else 0,
                lambda_kl=0.2 if use_regularize else 0.0,
            )

            model_progress_callback = None
            if progress_callback and session_id:
                def model_progress_callback(elapsed_steps: int, total_steps: int):
                    progress_callback(session_id, elapsed_steps, total_steps)

            if backend == BACKEND_MPS:
                if model_progress_callback is not None:
                    current_model._progress_callback = model_progress_callback
                edited_audio = current_model.edit(
                    prompt_tokens=tokens,
                    descriptions=[config['prompt']],
                    src_descriptions=[""],
                    progress=True,
                    return_tokens=True
                )
                return _normalize_waveform_for_save(edited_audio[0][0])

            native_prompt_mode = _mlx_native_prompt_mode()
            edited_audio = mlx_module.edit_with_mlx(
                current_model,
                mlx_runtime,
                prompt_tokens=tokens,
                prompt_text=config['prompt'],
                src_prompt_text="",
                codec_mode=codec_mode,
                native_prompt_mode=native_prompt_mode,
                progress_callback=model_progress_callback,
            )
            return _normalize_waveform_for_save(edited_audio)

    except AudioProcessingError:
        raise
    except Exception as e:
        raise AudioProcessingError(f"Failed to process audio: {str(e)}", status_code=500)

@app.route('/transform', methods=['POST'])
def transform_audio():
    """Handle audio transformation requests - simplified for localhost."""
    output_file_path = None  # <-- declare early so cleanup hook can see it

    try:
        # Aggressive localhost cleanup: unload model after each request
        AGGRESSIVE_UNLOAD = os.environ.get("MELODYFLOW_UNLOAD_EACH_REQUEST", "1") == "1"

        if AGGRESSIVE_UNLOAD:
            @after_this_request
            def _cleanup(response):
                # Remove generated WAV
                try:
                    if output_file_path and os.path.exists(output_file_path):
                        os.remove(output_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete output wav: {e}")

                # Unload model + clear GPU
                unload_model()
                return response

        session_id = None
        request_backend_engine = None

        # ---------------------------------------------------------------------
        # Handle both multipart form-data and JSON
        # ---------------------------------------------------------------------
        if request.content_type and 'multipart/form-data' in request.content_type:
            # File upload mode
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({'error': 'No audio file selected'}), 400
            
            variation = request.form.get('transformation_type', request.form.get('variation'))
            custom_flowstep = request.form.get('flowstep')
            solver = request.form.get('solver', 'euler')
            custom_prompt = request.form.get('prompt', request.form.get('custom_prompt'))
            session_id = request.form.get('session_id')
            request_backend_engine = request.form.get('backend_engine')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                audio_file.save(tmp_file.name)
                temp_file_path = tmp_file.name
            
            try:
                input_waveform = load_audio_from_file(temp_file_path)
            finally:
                os.unlink(temp_file_path)

        else:
            # JSON mode
            data = request.get_json()
            if not data or 'variation' not in data:
                return jsonify({'error': 'Missing required data'}), 400
            
            variation = data['variation']
            custom_flowstep = data.get('flowstep')
            solver = data.get('solver', 'euler')
            custom_prompt = data.get('custom_prompt')
            session_id = data.get('session_id')
            request_backend_engine = data.get('backend_engine')

            if 'audio_file_path' in data and data['audio_file_path']:
                audio_file_path = data['audio_file_path']
                input_waveform = load_audio_from_file(audio_file_path)
            else:
                return jsonify({'error': 'No audio data provided'}), 400

        # ---------------------------------------------------------------------
        # Validate parameters
        # ---------------------------------------------------------------------
        if custom_flowstep is not None:
            try:
                custom_flowstep = float(custom_flowstep)
                if custom_flowstep <= 0:
                    return jsonify({'error': 'Flowstep must be positive'}), 400
            except ValueError:
                return jsonify({'error': 'Invalid flowstep value'}), 400
        
        if solver not in ['euler', 'midpoint']:
            return jsonify({'error': 'Invalid solver. Must be "euler" or "midpoint"'}), 400

        # Publish transform progress snapshots for g4l_localhost poll relay.
        if session_id:
            init_progress_tracking(session_id)

        # ---------------------------------------------------------------------
        # Process audio
        # ---------------------------------------------------------------------
        processed_waveform = process_audio(
            input_waveform,
            variation,
            custom_flowstep,
            solver,
            custom_prompt,
            session_id=session_id,
            progress_callback=redis_progress_callback,
            backend_engine=request_backend_engine,
        )

        output_filename = f"output_{session_id}_{uuid.uuid4().hex[:8]}.wav"
        output_file_path = os.path.join(SHARED_TEMP_DIR, output_filename)

        torchaudio.save(output_file_path, processed_waveform.cpu(), 32000)

        # Explicit tensor cleanup (before model unload)
        del input_waveform
        del processed_waveform

        if session_id:
            finalize_progress_tracking(session_id)

        return send_file(
            output_file_path,
            as_attachment=True,
            download_name='transformed_audio.wav',
            mimetype='audio/wav'
        )

    except AudioProcessingError as e:
        if session_id:
            fail_progress_tracking(session_id, str(e))
        return jsonify({'error': str(e)}), e.status_code

    except Exception as e:
        if session_id:
            fail_progress_tracking(session_id, str(e))
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500


@app.route('/api/juce/transform_audio', methods=['POST'])
def juce_transform_audio():
    """JUCE-compatible transform endpoint for localhost Terry service."""
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('session_id') or uuid.uuid4())
    variation = str(data.get('variation') or '').strip()
    audio_data = data.get('audio_data')

    if not variation:
        return jsonify({'success': False, 'error': 'variation is required'}), 400
    if not audio_data:
        return jsonify({'success': False, 'error': 'audio_data is required on localhost'}), 400

    flowstep = data.get('flowstep')
    if flowstep is not None:
        try:
            flowstep = float(flowstep)
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid flowstep value'}), 400
        if flowstep <= 0:
            return jsonify({'success': False, 'error': 'Flowstep must be positive'}), 400

    solver = str(data.get('solver') or 'euler').strip().lower()
    if solver not in ('euler', 'midpoint'):
        return jsonify({'success': False, 'error': 'Invalid solver. Must be "euler" or "midpoint"'}), 400

    custom_prompt = data.get('custom_prompt')
    backend_engine = data.get('backend_engine')

    _upsert_juce_session(
        session_id,
        type='transform',
        created_at=time.time(),
        variation=variation,
        status='queued',
        progress=0,
        queue_status=_build_queue_status('queued', 'queued for transform', 0),
    )

    _queue_transform_job(
        session_id=session_id,
        audio_data=audio_data,
        variation=variation,
        flowstep=flowstep,
        solver=solver,
        custom_prompt=custom_prompt,
        backend_engine=backend_engine,
    )

    return jsonify({
        'success': True,
        'session_id': session_id,
        'message': 'Audio transform started',
        'note': 'Poll /api/juce/poll_status/{session_id} for progress and results',
    })


@app.route('/api/juce/poll_status/<session_id>', methods=['GET'])
def juce_poll_status(session_id):
    session_data = _get_juce_session(session_id)
    if not session_data:
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    session_data = _sync_juce_session_from_snapshot(session_id, session_data)
    status = str(session_data.get('status') or 'unknown')
    progress = max(0, min(100, int(session_data.get('progress') or 0)))
    queue_status = session_data.get('queue_status')
    if not isinstance(queue_status, dict):
        queue_status = {}

    response = {
        'success': True,
        'status': status,
        'progress': progress,
        'queue_status': queue_status,
    }

    if status in ('queued', 'warming', 'processing'):
        response['generation_in_progress'] = False
        response['transform_in_progress'] = True
    elif status == 'completed':
        response['generation_in_progress'] = False
        response['transform_in_progress'] = False
        audio_data = session_data.get('result_audio')
        if audio_data:
            response['audio_data'] = audio_data
    elif status == 'failed':
        response['generation_in_progress'] = False
        response['transform_in_progress'] = False
        response['error'] = str(session_data.get('error') or 'transform failed')
    else:
        response['generation_in_progress'] = False
        response['transform_in_progress'] = False

    return jsonify(response)


@app.route('/api/juce/undo_transform', methods=['POST'])
def juce_undo_transform():
    """Restore original pre-transform audio for a Terry session."""
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('session_id') or '').strip()
    if not session_id:
        return jsonify({'success': False, 'error': 'Session ID required'}), 400

    session_data = _get_juce_session(session_id)
    if not session_data:
        return jsonify({'success': False, 'error': 'Session not found'}), 404

    original_audio = session_data.get('original_audio')
    if not original_audio:
        return jsonify({'success': False, 'error': 'No original audio found for undo'}), 404

    return jsonify({
        'success': True,
        'audio_data': original_audio,
        'message': 'Transform undone successfully',
    })

@app.route('/variations', methods=['GET'])
def get_variations():
    """Return list of available variations."""
    try:
        variations_list = list(VARIATIONS.keys())
        variations_with_details = {
            name: {
                'prompt': VARIATIONS[name]['prompt'],
                'flowstep': VARIATIONS[name]['default_flowstep']
            } for name in variations_list
        }
        return jsonify({
            'variations': variations_with_details
        })
    except Exception as e:
        return jsonify({'error': f'Failed to fetch variations: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_available_models():
    try:
        return jsonify({
            "success": True,
            "models": _model_catalog_payload(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/models/download_status', methods=['GET'])
def get_models_download_status():
    try:
        requested_model = request.args.get("model_name", type=str)
        model_name = MELODYFLOW_MODEL_REPO
        if requested_model and requested_model != model_name:
            return jsonify({
                "success": False,
                "error": f"Unknown model '{requested_model}'",
            }), 404

        return jsonify({
            "success": True,
            "models": {
                model_name: _model_download_status_for(model_name),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/models/predownload', methods=['POST'])
def start_model_predownload():
    try:
        payload = request.get_json(silent=True) or {}
        model_name = str(payload.get("model_name") or "").strip()
        if not model_name:
            return jsonify({
                "success": False,
                "error": "model_name is required",
            }), 400
        if model_name != MELODYFLOW_MODEL_REPO:
            return jsonify({
                "success": False,
                "error": f"Unknown model '{model_name}'",
            }), 404

        session_id = str(uuid.uuid4())
        _upsert_model_predownload_session(
            session_id,
            model_name=model_name,
            status="queued",
            progress=0,
            queue_status=_build_model_queue_status(
                status="queued",
                message="queued for download",
                model_name=model_name,
                stage_index=0,
                stage_total=len(MELODYFLOW_REQUIRED_FILES),
                download_percent=0,
            ),
            error=None,
        )
        _run_model_predownload(session_id, model_name)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "model_name": model_name,
            "message": f"Started pre-download for {model_name}",
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/models/predownload_status/<session_id>', methods=['GET'])
def get_model_predownload_status(session_id: str):
    try:
        session_data = _get_model_predownload_session(session_id)
        if not session_data:
            return jsonify({
                "success": False,
                "error": "Session not found",
            }), 404

        response = {
            "success": True,
            "session_id": session_id,
            "model_name": session_data.get("model_name"),
            "status": str(session_data.get("status") or "unknown"),
            "progress": max(0, min(100, int(session_data.get("progress") or 0))),
            "queue_status": session_data.get("queue_status") if isinstance(session_data.get("queue_status"), dict) else {},
        }
        if response["status"] == "failed":
            response["error"] = str(session_data.get("error") or "Unknown error")
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_loaded = model is not None
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        accelerator = 'cuda' if cuda_available else 'mps' if mps_available else 'cpu'
        backend_engine = _default_backend_engine()
        requires_mps = backend_engine == BACKEND_MPS
        accelerator_available = accelerator != 'cpu'
        backend_ready = mps_available if requires_mps else True
        
        status = {
            'status': 'healthy' if accelerator_available and backend_ready else 'degraded',
            'accelerator': accelerator,
            'cuda_available': cuda_available,
            'mps_available': mps_available,
            'model_loaded': model_loaded,
            'backend_engine': backend_engine,
            'backend_requires_mps': requires_mps,
        }
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 503
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8002, debug=False)
