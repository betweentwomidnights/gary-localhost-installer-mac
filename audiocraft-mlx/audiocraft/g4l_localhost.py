"""
g4l_localhost.py - Simplified localhost backend for gary4juce
Removes WebSocket, Go service, MongoDB complexity while maintaining JUCE plugin compatibility
"""

import os
# Faster Hugging Face downloads (Xet high-performance transfer).
# NOTE: Must be set BEFORE importing libraries that might import `huggingface_hub`
# (e.g. `transformers`) so the Hub picks it up at import time.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import json
import time
import uuid
import gc
import base64
import traceback
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel, ValidationError

# Import our audio processing functions (MLX MusicGen)
from g4laudio_mlx import (
    continue_music,
    get_model_download_status,
    predownload_model,
    process_audio,
)
from g4l_models import MODEL_CATALOG

# =============================================================================
# CONFIGURATION
# =============================================================================

app = Flask(__name__)
CORS(app)

# -----------------------------------------------------------------------------
# Redis (optional): fall back to in-memory store for easy local installs.
# -----------------------------------------------------------------------------

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


class _InMemoryKV:
    """Minimal Redis-like store with TTL support (setex/get/ping)."""

    def __init__(self):
        self._data: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def ping(self) -> bool:
        return True

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        expires_at = time.time() + int(ttl_seconds)
        with self._lock:
            self._data[key] = (expires_at, value)

    def get(self, key: str):
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._data.pop(key, None)
                return None
            return value


def _create_kv_store():
    if redis is None:
        print("[WARN] redis-py not installed; using in-memory session store.")
        return _InMemoryKV()
    try:
        client = redis.StrictRedis(  # type: ignore[attr-defined]
            host="localhost", port=6379, db=0, decode_responses=True
        )
        client.ping()
        return client
    except Exception as e:
        print(f"[WARN] Redis not available; using in-memory session store. ({e})")
        return _InMemoryKV()


redis_client = _create_kv_store()

# =============================================================================
# PYDANTIC MODELS (Keep existing models for validation)
# =============================================================================

class AudioRequest(BaseModel):
    audio_data: str
    model_name: str
    prompt_duration: int
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None
    quantization_mode: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None
    quantization_mode: Optional[str] = None

class ContinueMusicRequest(BaseModel):
    session_id: Optional[str] = None
    model_name: Optional[str] = None
    prompt_duration: Optional[int] = None
    audio_data: Optional[str] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    cfg_coef: Optional[float] = None
    description: Optional[str] = None
    quantization_mode: Optional[str] = None

class ModelPredownloadRequest(BaseModel):
    model_name: str

# =============================================================================
# GPU CLEANUP UTILITIES (Keep for memory management)
# =============================================================================

@contextmanager
def force_gpu_cleanup():
    """Enhanced GPU cleanup context manager."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

def clean_gpu_memory():
    """Utility function to force GPU memory cleanup."""
    if torch.cuda.is_available():
        devices = range(torch.cuda.device_count())
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

# =============================================================================
# SESSION MANAGEMENT (Redis-based, same format as remote backend)
# =============================================================================

def generate_session_id():
    """Generate a unique session ID."""
    return str(uuid.uuid4())

def store_session_data(session_id: str, data: dict):
    """Store session data in Redis with 1 hour expiration."""
    redis_client.setex(f"session:{session_id}", 3600, json.dumps(data))

def get_session_data(session_id: str):
    """Retrieve session data from Redis."""
    data = redis_client.get(f"session:{session_id}")
    return json.loads(data) if data else None

def store_session_progress(session_id: str, progress: int):
    """Store generation progress for polling."""
    redis_client.setex(f"progress:{session_id}", 3600, str(progress))

def get_session_progress(session_id: str):
    """Get current generation progress."""
    progress = redis_client.get(f"progress:{session_id}")
    return int(progress) if progress else 0

def store_session_status(session_id: str, status: str, error: str = None):
    """Store session status (processing, completed, failed)."""
    status_data = {"status": status}
    if error:
        status_data["error"] = error
    redis_client.setex(f"status:{session_id}", 3600, json.dumps(status_data))

def get_session_status(session_id: str):
    """Get session status."""
    status_data = redis_client.get(f"status:{session_id}")
    return json.loads(status_data) if status_data else {"status": "unknown"}

def store_audio_result(session_id: str, audio_base64: str):
    """Store generated audio result."""
    redis_client.setex(f"result:{session_id}", 3600, audio_base64)

def get_audio_result(session_id: str):
    """Get generated audio result."""
    return redis_client.get(f"result:{session_id}")

def store_queue_status_update(session_id: str, payload: dict):
    """Store queue/warmup/processing hints for the poller."""
    redis_client.setex(f"queue_status:{session_id}", 3600, json.dumps(payload))

def get_stored_queue_status(session_id: str):
    """Retrieve last queue/warmup/processing hint."""
    data = redis_client.get(f"queue_status:{session_id}")
    return json.loads(data) if data else None


def _derive_progress_from_queue_status(
    progress: int,
    status: str,
    qstatus: dict,
) -> int:
    if status == "completed":
        return 100

    normalized = max(0, min(100, int(progress)))
    if not isinstance(qstatus, dict):
        return normalized

    try:
        stage_total = int(qstatus.get("stage_total") or 0)
        stage_index = int(qstatus.get("stage_index") or 0)
        stage_percent = int(qstatus.get("download_percent") or 0)
    except Exception:
        return normalized

    if stage_total <= 0 or stage_index <= 0:
        return normalized

    stage_percent = max(0, min(100, stage_percent))

    # Keep global progress aligned to the dominant checkpoint transfer.
    # For current predownload flow (5 stages), stage 1 is almost all bytes.
    if stage_total == 5:
        primary_stage_weight = 0.96
        if stage_index <= 1:
            derived_raw = (stage_percent / 100.0) * primary_stage_weight * 100.0
        else:
            secondary_stage_weight = (1.0 - primary_stage_weight) / 4.0
            completed_secondary_stages = max(0, stage_index - 2)
            derived_raw = (
                primary_stage_weight
                + (completed_secondary_stages * secondary_stage_weight)
                + ((stage_percent / 100.0) * secondary_stage_weight)
            ) * 100.0
        derived = int(derived_raw + 0.9999)
    else:
        derived = int(
            (((stage_index - 1) + (stage_percent / 100.0)) / stage_total) * 100
        )

    if stage_percent > 0:
        derived = max(1, derived)
    return max(normalized, min(99, derived))

# =============================================================================
# ADDITIONAL REDIS FUNCTIONS FOR LAST INPUT AUDIO
# =============================================================================

def store_last_input_audio(session_id: str, audio_base64: str):
    """Store last input audio for retry functionality."""
    redis_client.setex(f"last_input:{session_id}", 3600, audio_base64)

def get_last_input_audio(session_id: str):
    """Get last input audio for retry."""
    return redis_client.get(f"last_input:{session_id}")


def all_catalog_models() -> list[str]:
    ordered: list[str] = []
    for size in ("small", "medium", "large"):
        ordered.extend(MODEL_CATALOG.get(size, []))
    return ordered


def _fmt_bytes(n: int) -> str:
    f = float(max(0, int(n)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024.0 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024.0
    return f"{f:.1f}TB"


def run_model_predownload(session_id: str, model_name: str):
    """Background task to pre-download all assets required for offline model usage."""
    stage_started_at: dict[tuple[int, str, str], float] = {}

    def progress_callback(evt: dict):
        try:
            stage_name = str(evt.get("stage_name") or "download")
            repo_id = str(evt.get("repo_id") or model_name)
            downloaded = int(evt.get("downloaded_bytes") or 0)
            total = int(evt.get("total_bytes") or 0)
            stage_percent = int(evt.get("stage_percent") or 0)
            stage_total = int(evt.get("stage_total") or 0)
            stage_index = int(evt.get("stage_index") or 0)
            unit = str(evt.get("unit") or "").strip().lower()
            progress_name = str(evt.get("progress_name") or "").strip()
            speed_bps = float(evt.get("speed_bps") or 0.0)
            progress = int(evt.get("percent") or 0)
            if progress <= 0 and stage_total > 0 and stage_index > 0:
                derived = int(
                    (((stage_index - 1) + (max(0, min(100, stage_percent)) / 100.0))
                     / max(stage_total, 1))
                    * 100
                )
                if stage_percent > 0:
                    derived = max(1, derived)
                progress = max(progress, derived)
            progress = max(0, min(100, progress))
            store_session_progress(session_id, progress)

            stage_prefix = f"Stage {stage_index}/{stage_total} {stage_name}" if stage_total > 0 else stage_name
            speed_suffix = f" • {_fmt_bytes(int(speed_bps))}/s" if speed_bps > 0 else ""
            stage_key = (stage_index, stage_name, repo_id)
            now = time.time()
            started_at = stage_started_at.setdefault(stage_key, now)
            prep_seconds = int(max(0, now - started_at))
            if unit in {"it", "item", "items", "file", "files"} and total > 0:
                message = (
                    f"{stage_prefix}: {repo_id} "
                    f"({downloaded}/{total} files • {stage_percent}%)"
                )
            elif stage_percent <= 0 and downloaded <= 0 and prep_seconds >= 3:
                message = (
                    f"{stage_prefix}: {repo_id} "
                    f"(preparing transfer... {prep_seconds}s)"
                )
            elif total >= 4 * 1024 or downloaded >= 4 * 1024:
                message = (
                    f"{stage_prefix}: {repo_id} "
                    f"({_fmt_bytes(downloaded)}/{_fmt_bytes(total)} • {stage_percent}%{speed_suffix})"
                )
            elif progress_name:
                message = f"{stage_prefix}: {repo_id} ({stage_percent}% • {progress_name})"
            else:
                message = f"{stage_prefix}: {repo_id} ({stage_percent}%)"

            store_session_status(session_id, "warming" if progress < 100 else "processing")
            store_queue_status_update(session_id, {
                "status": "warming" if progress < 100 else "processing",
                "message": message,
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost",
                "phase": "download",
                "repo_id": repo_id,
                "download_percent": stage_percent,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "speed_bps": speed_bps,
                "stage_name": stage_name,
                "stage_index": stage_index,
                "stage_total": stage_total,
                "unit": unit,
                "progress_name": progress_name,
            })
        except Exception:
            # Progress updates should never fail the actual download.
            return

    def worker():
        try:
            store_session_status(session_id, "warming")
            store_session_progress(session_id, 0)
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": f"Preparing download for {model_name}",
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost",
                "phase": "download",
            })

            predownload_model(
                model_name=model_name,
                download_progress_callback=progress_callback,
            )

            store_session_status(session_id, "completed")
            store_session_progress(session_id, 100)
            store_queue_status_update(session_id, {
                "status": "completed",
                "message": f"{model_name} is ready for offline use.",
                "source": "localhost",
                "phase": "download",
                "model_name": model_name,
            })
        except Exception as e:
            error_message = str(e)
            print(f"[ERROR] Model predownload failed for {model_name}: {error_message}")
            print(traceback.format_exc())
            store_session_status(session_id, "failed", error_message)
            store_queue_status_update(session_id, {
                "status": "failed",
                "message": error_message,
                "source": "localhost",
                "phase": "download",
                "model_name": model_name,
            })

    threading.Thread(target=worker, daemon=True).start()

# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def run_audio_processing(session_id: str, audio_data: str, model_name: str,
                        prompt_duration: int, **kwargs):
    generation_started = False

    def progress_callback(current, total):
        nonlocal generation_started
        progress_percent = int((current / total) * 100)
        store_session_progress(session_id, progress_percent)

        # FIRST nonzero progress => leave warming
        if progress_percent > 0:
            generation_started = True
            store_session_status(session_id, "processing")
            store_queue_status_update(session_id, {
                "status": "processing",
                "message": "generating…",
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

    def _fmt_bytes(n: int) -> str:
        f = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if f < 1024.0 or unit == "TB":
                return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
            f /= 1024.0
        return f"{f:.1f}TB"

    def download_progress_callback(evt: dict) -> None:
        # Ignore late download updates once generation begins.
        if generation_started:
            return
        try:
            repo_id = str(evt.get("repo_id") or model_name)
            pct = int(evt.get("percent") or 0)
            downloaded = int(evt.get("downloaded_bytes") or 0)
            total = int(evt.get("total_bytes") or 0)
            unit = evt.get("unit")
            progress_name = evt.get("progress_name")
            if os.environ.get("G4L_DEBUG_DOWNLOADS") == "1":
                print(
                    f"[DL] {repo_id} {pct}% ({downloaded}/{total}) unit={unit} name={progress_name}"
                )
            # Only surface byte-based download progress in UI
            if unit not in (None, "B") and total <= 0:
                return
            if total > 0:
                msg = f"downloading {repo_id} ({_fmt_bytes(downloaded)}/{_fmt_bytes(total)} • {pct}%)"
            else:
                msg = f"downloading {repo_id} ({pct}%)"
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": msg,
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost",
                "download_percent": pct,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "repo_id": repo_id,
                "phase": "download",
            })
        except Exception:
            # Never allow download UX to break generation.
            return

    def processing_thread():
        try:
            # Tell poller we are warming (this is BEFORE model load / HF download)
            store_session_status(session_id, "warming")
            store_session_progress(session_id, 0)
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

            with force_gpu_cleanup():
                result_base64 = process_audio(
                    audio_data,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=kwargs.get('top_k', 250),
                    temperature=kwargs.get('temperature', 1.0),
                    cfg_coef=kwargs.get('cfg_coef', 3.0),
                    description=kwargs.get('description', ''),
                    quantization_mode=kwargs.get('quantization_mode'),
                    download_progress_callback=download_progress_callback,
                )

                store_audio_result(session_id, result_base64)
                store_session_status(session_id, "completed")
                store_session_progress(session_id, 100)
                store_queue_status_update(session_id, {
                    "status": "completed",
                    "message": "done",
                    "source": "localhost"
                })
                print(f"[SUCCESS] Audio processing completed for {session_id}")

        except Exception as e:
            print(f"[ERROR] Audio processing failed for {session_id}: {e}")
            print(traceback.format_exc())
            store_session_status(session_id, "failed", str(e))
            store_queue_status_update(session_id, {
                "status": "failed",
                "message": str(e),
                "source": "localhost"
            })
        finally:
            clean_gpu_memory()

    threading.Thread(target=processing_thread, daemon=True).start()

def run_continue_processing(session_id: str, audio_data: str, model_name: str,
                            prompt_duration: int, **kwargs):
    generation_started = False

    def progress_callback(current, total):
        nonlocal generation_started
        progress_percent = int((current / total) * 100)
        store_session_progress(session_id, progress_percent)
        if progress_percent > 0:
            generation_started = True
            store_session_status(session_id, "processing")
            store_queue_status_update(session_id, {
                "status": "processing",
                "message": "generating…",
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

    def _fmt_bytes(n: int) -> str:
        f = float(n)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if f < 1024.0 or unit == "TB":
                return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
            f /= 1024.0
        return f"{f:.1f}TB"

    def download_progress_callback(evt: dict) -> None:
        if generation_started:
            return
        try:
            repo_id = str(evt.get("repo_id") or model_name)
            pct = int(evt.get("percent") or 0)
            downloaded = int(evt.get("downloaded_bytes") or 0)
            total = int(evt.get("total_bytes") or 0)
            unit = evt.get("unit")
            progress_name = evt.get("progress_name")
            if os.environ.get("G4L_DEBUG_DOWNLOADS") == "1":
                print(
                    f"[DL] {repo_id} {pct}% ({downloaded}/{total}) unit={unit} name={progress_name}"
                )
            if unit not in (None, "B") and total <= 0:
                return
            if total > 0:
                msg = f"downloading {repo_id} ({_fmt_bytes(downloaded)}/{_fmt_bytes(total)} • {pct}%)"
            else:
                msg = f"downloading {repo_id} ({pct}%)"
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": msg,
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost",
                "download_percent": pct,
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "repo_id": repo_id,
                "phase": "download",
            })
        except Exception:
            return

    def processing_thread():
        try:
            store_session_status(session_id, "warming")
            store_session_progress(session_id, 0)
            store_last_input_audio(session_id, audio_data)
            store_queue_status_update(session_id, {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "localhost"
            })

            with force_gpu_cleanup():
                result_base64 = continue_music(
                    audio_data,
                    model_name,
                    progress_callback,
                    prompt_duration=prompt_duration,
                    top_k=kwargs.get('top_k', 250),
                    temperature=kwargs.get('temperature', 1.0),
                    cfg_coef=kwargs.get('cfg_coef', 3.0),
                    description=kwargs.get('description', ''),
                    quantization_mode=kwargs.get('quantization_mode'),
                    download_progress_callback=download_progress_callback,
                )

                store_audio_result(session_id, result_base64)
                store_session_status(session_id, "completed")
                store_session_progress(session_id, 100)
                store_queue_status_update(session_id, {
                    "status": "completed",
                    "message": "done",
                    "source": "localhost"
                })

        except Exception as e:
            print(f"[ERROR] Continue processing failed for {session_id}: {e}")
            print(traceback.format_exc())
            store_session_status(session_id, "failed", str(e))
            store_queue_status_update(session_id, {
                "status": "failed",
                "message": str(e),
                "source": "localhost"
            })
        finally:
            clean_gpu_memory()

    threading.Thread(target=processing_thread, daemon=True).start()


# =============================================================================
# HTTP ENDPOINTS (Same interface as remote backend)
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint (matches remote backend format)."""
    health_status = {"status": "live", "service": "gary4juce-localhost"}

    # Check Redis (essential for session storage)
    try:
        redis_client.ping()
        health_status['redis'] = 'live'
    except Exception as e:
        health_status['redis'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check MLX (primary backend on Apple Silicon)
    try:
        import mlx.core as mx  # noqa: F401
        health_status['mlx'] = 'live'
    except Exception as e:
        health_status['mlx'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check PyTorch (used for loading HF weights, even in MLX mode)
    try:
        health_status['torch'] = 'live'
        health_status['cuda'] = bool(torch.cuda.is_available())
        health_status['mps'] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as e:
        health_status['torch'] = f'down: {str(e)}'
        health_status['status'] = 'degraded'

    # Check backend import (essential for Gary functionality)
    try:
        from g4laudio_mlx import process_audio  # noqa: F401
        health_status['backend'] = 'live'
    except Exception as e:
        health_status['backend'] = f'import error: {str(e)}'
        health_status['status'] = 'degraded'

    # Return appropriate status code
    status_code = 200 if health_status['status'] == 'live' else 500
    return jsonify(health_status), status_code

@app.route('/api/juce/process_audio', methods=['POST'])
def juce_process_audio():
    """Process audio - direct processing instead of queueing."""
    try:
        # Validate request
        request_data = AudioRequest(**request.json)
        session_id = generate_session_id()
        
        # Store session data (same format as remote)
        session_data = {
            'session_id': session_id,
            'model_name': request_data.model_name,
            'prompt_duration': request_data.prompt_duration,
            'quantization_mode': request_data.quantization_mode,
            'parameters': {
                'top_k': request_data.top_k or 250,
                'temperature': request_data.temperature or 1.0,
                'cfg_coef': request_data.cfg_coef or 3.0,
            },
            'description': request_data.description or '',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        store_session_data(session_id, session_data)
        
        # Start processing immediately (no queue)
        run_audio_processing(
            session_id,
            request_data.audio_data,
            request_data.model_name,
            request_data.prompt_duration,
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description,
            quantization_mode=request_data.quantization_mode,
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Audio processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/continue_music', methods=['POST'])
def juce_continue_music():
    """Continue music generation."""
    try:
        request_data = ContinueMusicRequest(**request.json)
        session_id = generate_session_id()
        
        # Store session data
        session_data = {
            'session_id': session_id,
            'model_name': request_data.model_name,
            'prompt_duration': request_data.prompt_duration,
            'quantization_mode': request_data.quantization_mode,
            'type': 'continue',
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        store_session_data(session_id, session_data)
        
        # Start continue processing
        run_continue_processing(
            session_id,
            request_data.audio_data,
            request_data.model_name,
            request_data.prompt_duration,
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description,
            quantization_mode=request_data.quantization_mode,
        )
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Continue processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/juce/retry_music', methods=['POST'])
def juce_retry_music():
    """Retry music generation with new parameters - FIXED VERSION."""
    try:
        request_data = SessionRequest(**request.json)
        old_session_id = request_data.session_id
        new_session_id = generate_session_id()
        
        # Get original session data
        old_session_data = get_session_data(old_session_id)
        if not old_session_data:
            return jsonify({'success': False, 'error': 'Original session not found'}), 404
        
        # FIXED: Get last INPUT audio (not result audio!)
        last_input_audio = get_last_input_audio(old_session_id)
        if not last_input_audio:
            return jsonify({'success': False, 'error': 'No last input audio found for retry'}), 404
        
        # Create new session with updated parameters
        new_session_data = old_session_data.copy()
        new_session_data['session_id'] = new_session_id
        new_session_data['type'] = 'retry'
        new_session_data['original_session'] = old_session_id
        
        # Update with new parameters if provided
        if request_data.model_name:
            new_session_data['model_name'] = request_data.model_name
        if request_data.prompt_duration:
            new_session_data['prompt_duration'] = request_data.prompt_duration
        if request_data.quantization_mode:
            new_session_data['quantization_mode'] = request_data.quantization_mode

        store_session_data(new_session_id, new_session_data)
        
        # FIXED: Start retry processing using last INPUT audio (not result audio)
        run_continue_processing(
            new_session_id,
            last_input_audio,  # Use the input audio that was used in the previous continuation
            new_session_data['model_name'],
            new_session_data['prompt_duration'],
            top_k=request_data.top_k,
            temperature=request_data.temperature,
            cfg_coef=request_data.cfg_coef,
            description=request_data.description,
            quantization_mode=new_session_data.get('quantization_mode'),
        )
        
        return jsonify({
            'success': True,
            'session_id': new_session_id,
            'message': 'Retry processing started'
        })
        
    except ValidationError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/api/juce/poll_status/<session_id>', methods=['GET'])
def juce_poll_status(session_id):
    try:
        status_data = get_session_status(session_id)      # {"status": "...", "error": "...?"}
        progress = get_session_progress(session_id)
        qstatus = get_stored_queue_status(session_id)     # may be None

        # Synthesize a warming hint if we look idle-but-working and nothing is stored yet
        if (status_data.get('status') in ('warming', 'processing') and progress == 0 and not qstatus):
            sess = get_session_data(session_id) or {}
            model_name = (sess.get('model_name') or 'model')
            qstatus = {
                "status": "warming",
                "message": f'loading {model_name} (first run / hub download)',
                "position": 0,
                "total_queued": 0,
                "estimated_time": None,
                "estimated_seconds": 0,
                "source": "synthetic-localhost"
            }

        response = {
            "success": True,
            "status": status_data.get("status", "unknown"),
            "progress": progress,
            "queue_status": qstatus or {}
        }

        # Keep JUCE flags consistent with remote
        if status_data.get("status") in ("warming", "processing"):
            response["generation_in_progress"] = True
            response["transform_in_progress"] = False
        elif status_data.get("status") == "completed":
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False
            audio_result = get_audio_result(session_id)
            if audio_result:
                response["audio_data"] = audio_result
        elif status_data.get("status") == "failed":
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False
            response["error"] = status_data.get("error", "Unknown error")
        else:
            response["generation_in_progress"] = False
            response["transform_in_progress"] = False

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
@app.route('/api/models', methods=['GET'])
def get_available_models():
    """
    Return available models organized by size with automatic checkpoint grouping.
    Models following the pattern 'name-size-epoch' are automatically grouped.
    """
    try:
        models = MODEL_CATALOG
        
        def parse_model_info(model_path):
            """Extract base name and checkpoint info from model path"""
            # Remove the 'thepatch/' prefix
            name = model_path.split('/')[-1]
            
            # Try to extract checkpoint number from end (e.g., 'bleeps-large-6' -> 6)
            parts = name.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return {
                    'full_path': model_path,
                    'display_name': name,
                    'base_name': parts[0],
                    'checkpoint': int(parts[1]),
                    'has_checkpoint': True
                }
            else:
                # Legacy models without checkpoint numbers
                return {
                    'full_path': model_path,
                    'display_name': name,
                    'base_name': name,
                    'checkpoint': None,
                    'has_checkpoint': False
                }
        
        def group_models(model_list):
            """Group models by base name, with checkpoints as nested items"""
            parsed = [parse_model_info(m) for m in model_list]
            
            # Group by base_name
            grouped = {}
            for model in parsed:
                base = model['base_name']
                if base not in grouped:
                    grouped[base] = []
                grouped[base].append(model)
            
            # Build result structure
            result = []
            for base_name, models_group in grouped.items():
                if len(models_group) == 1 and not models_group[0]['has_checkpoint']:
                    # Single model without checkpoint - don't nest
                    result.append({
                        'name': models_group[0]['display_name'],
                        'path': models_group[0]['full_path'],
                        'type': 'single'
                    })
                else:
                    # Multiple checkpoints - create group.
                    # Single-checkpoint groups are flattened to avoid client-side
                    # dropdowns skipping group nodes with one child.
                    checkpoints = sorted(
                        [m for m in models_group if m['has_checkpoint']], 
                        key=lambda x: x['checkpoint']
                    )
                    if len(checkpoints) == 1:
                        c = checkpoints[0]
                        result.append({
                            'name': c['display_name'],
                            'path': c['full_path'],
                            'type': 'single',
                            'epoch': c['checkpoint'],
                        })
                    else:
                        result.append({
                            'name': base_name,
                            'type': 'group',
                            'checkpoints': [
                                {
                                    'name': f"{base_name}-{c['checkpoint']}",
                                    'path': c['full_path'],
                                    'epoch': c['checkpoint']
                                }
                                for c in checkpoints
                            ]
                        })
            
            return result
        
        # Process each size category
        response = {
            'small': group_models(models['small']),
            'medium': group_models(models['medium']),
            'large': group_models(models['large'])
        }
        
        return jsonify({
            'success': True,
            'models': response,
            'updated_at': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/models/download_status', methods=['GET'])
def get_models_download_status():
    """Return offline-download availability for all catalog models (or one model)."""
    try:
        requested_model = request.args.get("model_name", type=str)
        catalog_models = all_catalog_models()
        catalog_set = set(catalog_models)

        if requested_model:
            if requested_model not in catalog_set:
                return jsonify({
                    "success": False,
                    "error": f"Unknown model '{requested_model}'",
                }), 404
            model_paths = [requested_model]
        else:
            model_paths = catalog_models

        status_payload = {
            model_path: get_model_download_status(model_path)
            for model_path in model_paths
        }

        return jsonify({
            "success": True,
            "models": status_payload,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/models/predownload', methods=['POST'])
def start_model_predownload():
    """Start background pre-download for a selected model."""
    try:
        payload = request.json or {}
        req = ModelPredownloadRequest(**payload)

        catalog_set = set(all_catalog_models())
        if req.model_name not in catalog_set:
            return jsonify({
                "success": False,
                "error": f"Unknown model '{req.model_name}'",
            }), 404

        session_id = generate_session_id()
        store_session_data(session_id, {
            "session_id": session_id,
            "model_name": req.model_name,
            "task": "model_predownload",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        run_model_predownload(session_id, req.model_name)

        return jsonify({
            "success": True,
            "session_id": session_id,
            "model_name": req.model_name,
            "message": f"Started pre-download for {req.model_name}",
        })
    except ValidationError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route('/api/models/predownload_status/<session_id>', methods=['GET'])
def get_model_predownload_status(session_id: str):
    """Poll status for an active/completed pre-download task."""
    try:
        status_data = get_session_status(session_id)
        progress = get_session_progress(session_id)
        qstatus = get_stored_queue_status(session_id) or {}
        session_data = get_session_data(session_id) or {}
        status_value = status_data.get("status", "unknown")
        progress = _derive_progress_from_queue_status(progress, status_value, qstatus)

        response = {
            "success": True,
            "session_id": session_id,
            "model_name": session_data.get("model_name"),
            "status": status_value,
            "progress": progress,
            "queue_status": qstatus,
        }
        if status_value == "failed":
            response["error"] = status_data.get("error", "Unknown error")
        return jsonify(response)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == '__main__':
    print("🎵 Starting gary4juce localhost backend...")
    print("🔧 Redis connection:", "OK" if redis_client.ping() else "FAILED")
    print("🎯 CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Start Flask app
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
