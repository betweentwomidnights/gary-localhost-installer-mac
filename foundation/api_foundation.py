#!/usr/bin/env python3
"""
Foundation-1 API Service (macOS / Apple Silicon)

Structured text-to-sample inference for the RoyalCities Foundation-1 model.
Uses the saomlx (Stable Audio MLX) pipeline for Apple Silicon acceleration.

Async generation with polling -- matches the gary4juce poll_status contract.

Endpoints:
  POST /generate                   Submit a generation request -> returns session_id
  POST /audio2audio                Submit audio-to-audio request -> returns session_id
  GET  /poll_status/<session_id>   Poll generation progress/completion
  POST /randomize                  Generate random preset via RC prompt engine
  GET  /api/models                 Model catalog for gary4local predownload UI
  GET  /api/models/download_status Foundation-1 cache status
  POST /api/models/predownload     Start Foundation-1 predownload
  GET  /api/models/predownload_status/<session_id>
  GET  /health                     Wrapper + model readiness
  GET  /ready                      Simple readiness probe
"""

from __future__ import annotations

import base64
import gc
import inspect
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
import wave
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from flask import Flask, request, jsonify

# ---------------------------------------------------------------------------
# saomlx import -- the stable-audio-tools directory must be on PYTHONPATH
# (set by the service manifest or manually via FOUNDATION_SAOMLX_ROOT).
# ---------------------------------------------------------------------------

_saomlx_root = os.getenv("FOUNDATION_SAOMLX_ROOT")
if _saomlx_root and _saomlx_root not in sys.path:
    sys.path.insert(0, _saomlx_root)

try:
    from saomlx import generate_diffusion_cond_mlx, clear_runtime_caches
    _HAS_SAOMLX = True
except ImportError:
    _HAS_SAOMLX = False
    print("[foundation] WARNING: saomlx not found -- MLX generation unavailable")
    print("[foundation] Set FOUNDATION_SAOMLX_ROOT to the stable-audio-tools directory")

# ---------------------------------------------------------------------------
# RC prompt engine (optional -- /randomize endpoint)
# ---------------------------------------------------------------------------

_rc_prompt = None


def _load_rc_prompt():
    global _rc_prompt
    if _rc_prompt is not None:
        return _rc_prompt
    import importlib

    # Try installed package first
    try:
        mod = importlib.import_module(
            "stable_audio_tools.interface.prompts.master_prompt_map"
        )
        _rc_prompt = mod
        return mod
    except ModuleNotFoundError:
        pass

    # Prefer the vendored RC prompt engine we ship with the Foundation runtime.
    # Keep the old local checkout path as a fallback for developer worktrees
    # that still have foundation/RC-stable-audio-tools around.
    import importlib.util
    candidates = [
        Path(__file__).parent / "third_party" / "rc-stable-audio-tools" / "stable_audio_tools" / "interface" / "prompts" / "master_prompt_map.py",
        Path(__file__).parent / "RC-stable-audio-tools" / "stable_audio_tools" / "interface" / "prompts" / "master_prompt_map.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("master_prompt_map", str(candidate))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _rc_prompt = mod
            return mod

    tried = ", ".join(str(candidate) for candidate in candidates)
    raise ImportError(f"RC prompt engine not found. Tried {tried}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FOUNDATION_MODEL_DIR = os.environ.get(
    "FOUNDATION_MODEL_DIR",
    os.path.join(
        os.environ.get("HOME", ""),
        "Library", "Application Support", "GaryLocalhost", "cache", "foundation-1",
    ),
)
FOUNDATION_CKPT_PATH = os.environ.get(
    "FOUNDATION_CKPT_PATH",
    os.path.join(FOUNDATION_MODEL_DIR, "Foundation_1.safetensors"),
)
FOUNDATION_CONFIG_PATH = os.environ.get(
    "FOUNDATION_CONFIG_PATH",
    os.path.join(FOUNDATION_MODEL_DIR, "model_config.json"),
)
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(FOUNDATION_MODEL_DIR, "outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)
FOUNDATION_HF_REPO = os.environ.get("FOUNDATION_HF_REPO", "RoyalCities/Foundation-1")

PORT = int(os.environ.get("PORT", "8015"))

SUPPORTED_BPMS = [100, 110, 120, 128, 130, 140, 150]
SUPPORTED_BARS = [4, 8]
VALID_KEY_ROOTS = [
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
]
VALID_KEY_MODES = ["major", "minor"]

FALLBACK_PROMPT = "Synth, Pad, Warm, Wide, Chord Progression, 4 Bars, 120 BPM, C minor"
FOUNDATION_MODEL_DISPLAY_NAME = os.environ.get("FOUNDATION_MODEL_DISPLAY_NAME", "foundation-1")


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


def foundation_required_model_files() -> list[tuple[str, Path]]:
    return [
        ("Foundation_1.safetensors", Path(FOUNDATION_CKPT_PATH)),
        ("model_config.json", Path(FOUNDATION_CONFIG_PATH)),
    ]


def foundation_model_download_status() -> dict:
    missing = [
        filename
        for filename, path in foundation_required_model_files()
        if not path.exists() or path.stat().st_size <= 0
    ]
    return {
        "downloaded": len(missing) == 0,
        "missing": missing,
    }


def _make_hf_tqdm_class(on_progress):
    from huggingface_hub.utils import tqdm as hf_tqdm

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
            percent = int((downloaded / total) * 100) if total > 0 else 0

            if not force:
                if (now - self._last_emit_t) < 0.25:
                    return
                pct_delta_ok = self._last_emit_pct < 0 or abs(percent - self._last_emit_pct) >= 1
                bytes_delta_ok = (downloaded - self._last_emit_bytes) >= (1 * 1024 * 1024)
                if not (pct_delta_ok or bytes_delta_ok):
                    return

            self._last_emit_t = now
            self._last_emit_pct = percent
            self._last_emit_bytes = downloaded

            speed_bps = 0.0
            try:
                speed_bps = float((getattr(self, "format_dict", {}) or {}).get("rate") or 0.0)
            except Exception:
                speed_bps = 0.0

            on_progress({
                "downloaded_bytes": downloaded,
                "total_bytes": total,
                "download_percent": max(0, min(100, percent)),
                "speed_bps": max(0.0, speed_bps),
            })

    return _CallbackTqdm


@contextmanager
def _patched_hf_download_tqdm_if_needed(tqdm_class):
    from huggingface_hub import hf_hub_download

    supports_tqdm_class = "tqdm_class" in inspect.signature(hf_hub_download).parameters
    if supports_tqdm_class:
        yield {"tqdm_class": tqdm_class}
        return

    try:
        import huggingface_hub.file_download as file_download
    except Exception:
        yield {}
        return

    original_tqdm = getattr(file_download, "tqdm", None)
    if original_tqdm is None:
        yield {}
        return

    file_download.tqdm = tqdm_class
    try:
        yield {}
    finally:
        file_download.tqdm = original_tqdm


def _download_hf_file_with_progress(
    *,
    repo_id: str,
    filename: str,
    repo_type: str = "model",
    on_progress=None,
) -> str:
    from huggingface_hub import hf_hub_download

    kwargs = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": repo_type,
    }

    if on_progress is not None:
        tqdm_class = _make_hf_tqdm_class(on_progress)
        with _patched_hf_download_tqdm_if_needed(tqdm_class) as extra_kwargs:
            kwargs_with_progress = dict(kwargs)
            kwargs_with_progress.update(extra_kwargs)
            return hf_hub_download(**kwargs_with_progress)

    return hf_hub_download(**kwargs)


def ensure_foundation_model_files(progress_callback=None) -> tuple[Path, Path]:
    """Download missing Foundation-1 files into FOUNDATION_MODEL_DIR."""
    model_dir = Path(FOUNDATION_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(FOUNDATION_CKPT_PATH)
    cfg = Path(FOUNDATION_CONFIG_PATH)

    def ensure_file(filename: str, destination: Path, stage_index: int, stage_total: int) -> Path:
        def emit_progress(download_percent: int, message: str, *, downloaded_bytes: int = 0, total_bytes: int = 0, speed_bps: float = 0.0):
            if progress_callback:
                progress_callback(
                    filename,
                    stage_index,
                    stage_total,
                    download_percent,
                    message,
                    downloaded_bytes,
                    total_bytes,
                    speed_bps,
                )

        if destination.exists() and destination.stat().st_size > 0:
            size_bytes = int(destination.stat().st_size)
            emit_progress(
                100,
                f"{filename} already downloaded",
                downloaded_bytes=size_bytes,
                total_bytes=size_bytes,
            )
            return destination
        if destination.exists():
            destination.unlink()
        emit_progress(0, f"downloading {filename}")

        def on_progress(evt: dict) -> None:
            downloaded = int(evt.get("downloaded_bytes") or 0)
            total = int(evt.get("total_bytes") or 0)
            stage_percent = int(evt.get("download_percent") or evt.get("percent") or 0)
            speed_bps = float(evt.get("speed_bps") or 0.0)
            speed_suffix = f" • {_fmt_bytes(int(speed_bps))}/s" if speed_bps > 0 else ""
            total_display = _fmt_bytes(total) if total > 0 else "?"
            message = (
                f"downloading {filename} "
                f"({_fmt_bytes(downloaded)}/{total_display} • {stage_percent}%{speed_suffix})"
            )
            emit_progress(
                stage_percent,
                message,
                downloaded_bytes=downloaded,
                total_bytes=total,
                speed_bps=speed_bps,
            )

        downloaded = Path(
            _download_hf_file_with_progress(
                repo_id=FOUNDATION_HF_REPO,
                filename=filename,
                repo_type="model",
                on_progress=on_progress,
            )
        )
        if downloaded.resolve() != destination.resolve():
            shutil.copy2(downloaded, destination)
        final_size = int(destination.stat().st_size) if destination.exists() else 0
        emit_progress(
            100,
            f"downloaded {filename}",
            downloaded_bytes=final_size,
            total_bytes=final_size,
        )
        return destination

    ensure_file("Foundation_1.safetensors", ckpt, 1, 2)
    ensure_file("model_config.json", cfg, 2, 2)
    return ckpt, cfg


def load_foundation_runtime_profile() -> dict[str, Optional[str]]:
    """Resolve the effective diffusion objective + sampler for this config."""
    cfg = Path(FOUNDATION_CONFIG_PATH)
    if not cfg.exists():
        return {
            "diffusion_objective": None,
            "sampler_type": None,
        }

    with cfg.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    diffusion_cfg = model_config.get("model", {}).get("diffusion", {})
    diffusion_objective = diffusion_cfg.get("diffusion_objective") or "v"

    if diffusion_objective == "rf_denoiser":
        sampler_type = "pingpong"
    elif diffusion_objective == "rectified_flow":
        sampler_type = "euler"
    else:
        sampler_type = "dpmpp-3m-sde"

    return {
        "diffusion_objective": diffusion_objective,
        "sampler_type": sampler_type,
    }

# ---------------------------------------------------------------------------
# Session store (session_id -> job state)
# ---------------------------------------------------------------------------

sessions = {}
sessions_lock = threading.Lock()
generation_semaphore = threading.Semaphore(1)

model_download_sessions = {}
model_download_lock = threading.Lock()

model_ready = threading.Event()


def create_session(session_id: str, meta: dict):
    with sessions_lock:
        sessions[session_id] = {
            "status": "queued",
            "generation_in_progress": True,
            "transform_in_progress": False,
            "progress": 0,
            "step": 0,
            "total_steps": meta.get("steps", 100),
            "audio_data": None,
            "error": None,
            "status_message": "waiting for GPU",
            "queue_status": {
                "status": "queued",
                "position": 1,
                "message": "waiting for GPU",
                "estimated_time": "~5s",
                "estimated_seconds": 5,
            },
            "meta": meta,
            "created_at": time.time(),
        }


def update_session(session_id: str, **kwargs):
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id].update(kwargs)


def get_session(session_id: str) -> Optional[dict]:
    with sessions_lock:
        return sessions.get(session_id, {}).copy() if session_id in sessions else None


def cleanup_old_sessions(max_age: float = 600.0):
    now = time.time()
    with sessions_lock:
        expired = [
            sid for sid, s in sessions.items()
            if now - s.get("created_at", 0) > max_age
        ]
        for sid in expired:
            del sessions[sid]
    with model_download_lock:
        expired_downloads = [
            sid for sid, s in model_download_sessions.items()
            if now - s.get("created_at", 0) > max_age
        ]
        for sid in expired_downloads:
            del model_download_sessions[sid]


def update_model_download_session(session_id: str, **kwargs):
    with model_download_lock:
        if session_id in model_download_sessions:
            model_download_sessions[session_id].update(kwargs)


def get_model_download_session(session_id: str) -> Optional[dict]:
    with model_download_lock:
        if session_id not in model_download_sessions:
            return None
        return model_download_sessions[session_id].copy()


def foundation_download_queue_status(
    *,
    status: str,
    message: str,
    stage_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
    speed_bps: float = 0.0,
) -> dict:
    return {
        "status": status,
        "message": message,
        "repo_id": FOUNDATION_HF_REPO,
        "stage_name": stage_name,
        "stage_index": stage_index,
        "stage_total": stage_total,
        "download_percent": max(0, min(100, int(download_percent))),
        "downloaded_bytes": max(0, int(downloaded_bytes)),
        "total_bytes": max(0, int(total_bytes)),
        "speed_bps": max(0.0, float(speed_bps)),
    }


def update_generation_download_status(
    session_id: str,
    *,
    status: str,
    message: str,
    stage_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
    speed_bps: float = 0.0,
):
    stage_progress = (
        (max(0, stage_index - 1) + (max(0, min(100, int(download_percent))) / 100.0))
        / max(1, stage_total)
    )
    update_session(
        session_id,
        status="downloading_model",
        generation_in_progress=True,
        transform_in_progress=False,
        progress=max(0, min(99, int(stage_progress * 100))),
        queue_status=foundation_download_queue_status(
            status=status,
            message=message,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_total=stage_total,
            download_percent=download_percent,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
        ),
        status_message=message,
    )


def ensure_foundation_model_files_for_session(session_id: str):
    update_generation_download_status(
        session_id,
        status="warming",
        message=f"preparing download for {FOUNDATION_HF_REPO}",
        stage_name="prepare",
        stage_index=0,
        stage_total=2,
        download_percent=0,
    )

    def progress_callback(
        filename: str,
        stage_index: int,
        stage_total: int,
        download_percent: int,
        message: str,
        downloaded_bytes: int = 0,
        total_bytes: int = 0,
        speed_bps: float = 0.0,
    ):
        update_generation_download_status(
            session_id,
            status="processing",
            message=message,
            stage_name=filename,
            stage_index=stage_index,
            stage_total=stage_total,
            download_percent=download_percent,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
        )

    ensure_foundation_model_files(progress_callback=progress_callback)


def run_foundation_model_predownload(session_id: str):
    def progress_callback(
        filename: str,
        stage_index: int,
        stage_total: int,
        download_percent: int,
        message: str,
        downloaded_bytes: int = 0,
        total_bytes: int = 0,
        speed_bps: float = 0.0,
    ):
        stage_progress = (
            (max(0, stage_index - 1) + (max(0, min(100, int(download_percent))) / 100.0))
            / max(1, stage_total)
        )
        update_model_download_session(
            session_id,
            status="processing",
            progress=max(1, min(99, int(stage_progress * 100))),
            queue_status=foundation_download_queue_status(
                status="processing",
                message=message,
                stage_name=filename,
                stage_index=stage_index,
                stage_total=stage_total,
                download_percent=download_percent,
                downloaded_bytes=downloaded_bytes,
                total_bytes=total_bytes,
                speed_bps=speed_bps,
            ),
        )

    def worker():
        try:
            update_model_download_session(
                session_id,
                status="warming",
                progress=0,
                queue_status=foundation_download_queue_status(
                    status="warming",
                    message=f"preparing download for {FOUNDATION_HF_REPO}",
                    stage_name="prepare",
                    stage_index=0,
                    stage_total=2,
                    download_percent=0,
                ),
            )
            ensure_foundation_model_files(progress_callback=progress_callback)
            update_model_download_session(
                session_id,
                status="completed",
                progress=100,
                queue_status=foundation_download_queue_status(
                    status="completed",
                    message="foundation-1 model files downloaded",
                    stage_name="completed",
                    stage_index=2,
                    stage_total=2,
                    download_percent=100,
                ),
                error=None,
            )
        except Exception as exc:
            update_model_download_session(
                session_id,
                status="failed",
                progress=0,
                queue_status=foundation_download_queue_status(
                    status="failed",
                    message=str(exc),
                    stage_name="failed",
                    stage_index=0,
                    stage_total=2,
                    download_percent=0,
                ),
                error=str(exc),
            )

    threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------------------
# BPM / duration helpers
# ---------------------------------------------------------------------------

def nearest_foundation_bpm(host_bpm: float) -> int:
    return min(SUPPORTED_BPMS, key=lambda b: abs(b - host_bpm))


def derive_duration(bars: int, bpm: float) -> float:
    return bars * 4 * 60.0 / bpm


def time_stretch_ratio(host_bpm: float, foundation_bpm: int) -> float:
    return host_bpm / foundation_bpm


# ---------------------------------------------------------------------------
# Audio shape normalization
# ---------------------------------------------------------------------------

def coerce_channels_first_audio(audio: torch.Tensor | np.ndarray, *, context: str) -> torch.Tensor:
    """Normalize audio tensors to [channels, samples]."""
    if isinstance(audio, torch.Tensor):
        tensor = audio.detach().to(torch.float32)
    else:
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    if tensor.ndim == 3:
        if int(tensor.shape[0]) != 1:
            raise ValueError(f"{context}: expected batch size 1, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]

    if tensor.ndim != 2:
        raise ValueError(f"{context}: expected 1D/2D/3D audio, got shape {tuple(tensor.shape)}")

    dim0, dim1 = int(tensor.shape[0]), int(tensor.shape[1])

    # Foundation's MLX pipeline returns [time, channels]. torchaudio and our
    # time-stretch path expect [channels, samples].
    if dim0 <= 8 and dim1 > dim0:
        return tensor.contiguous()
    if dim1 <= 8 and dim0 > dim1:
        return tensor.transpose(0, 1).contiguous()

    raise ValueError(
        f"{context}: ambiguous audio shape {tuple(tensor.shape)}; "
        "could not determine channel axis"
    )


# ---------------------------------------------------------------------------
# Time-stretch
# ---------------------------------------------------------------------------

def apply_time_stretch(audio_tensor: torch.Tensor, ratio: float, sample_rate: int) -> torch.Tensor:
    if abs(ratio - 1.0) < 0.001:
        return audio_tensor

    try:
        import pyrubberband as pyrb
        np_audio = audio_tensor.cpu().numpy()
        np_audio_t = np_audio.T
        stretched = pyrb.time_stretch(np_audio_t, sample_rate, ratio)
        return torch.from_numpy(stretched.T.copy()).to(audio_tensor.dtype)
    except ImportError:
        pass

    # Fallback: resample-based stretch
    virtual_sr = int(sample_rate * ratio)
    resampler_down = torchaudio.transforms.Resample(sample_rate, virtual_sr)
    resampler_up = torchaudio.transforms.Resample(virtual_sr, sample_rate)
    stretched = resampler_up(resampler_down(audio_tensor.cpu()))
    return stretched.to(audio_tensor.dtype)


def encode_audio_to_wav_bytes(
    audio_tensor: torch.Tensor,
    sample_rate: int,
    output_path: str | Path,
) -> tuple[bytes, str]:
    """Encode a torch audio tensor to a 16-bit WAV file and base64."""
    audio_cpu = coerce_channels_first_audio(audio_tensor, context="wav encoding").cpu()

    # The wave module expects interleaved frames. Convert from [channels, samples]
    # to [samples, channels], clamp to [-1, 1], and quantize to signed 16-bit PCM.
    audio_np = audio_cpu.transpose(0, 1).contiguous().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    pcm16 = np.rint(audio_np * 32767.0).astype(np.int16)
    pcm16 = np.ascontiguousarray(pcm16)
    num_channels = pcm16.shape[1]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    audio_bytes = output_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return audio_bytes, audio_b64


# ---------------------------------------------------------------------------
# Prompt builder (mirrors Windows build_prompt exactly)
# ---------------------------------------------------------------------------

def build_prompt(data: dict) -> str:
    parts = []

    family = (data.get("family") or "").strip()
    subfamily = (data.get("subfamily") or "").strip()
    if family:
        parts.append(family)
    if subfamily:
        parts.append(subfamily)

    for knob in ["descriptor_knob_a", "descriptor_knob_b", "descriptor_knob_c"]:
        val = (data.get(knob) or "").strip()
        if val:
            parts.append(val)

    for extra in (data.get("descriptors_extra") or []):
        val = (extra if isinstance(extra, str) else "").strip()
        if val:
            parts.append(val)

    for tag_key in ["spatial_tags", "band_tags", "wave_tech_tags", "style_tags"]:
        for tag in (data.get(tag_key) or []):
            val = (tag if isinstance(tag, str) else "").strip()
            if val:
                parts.append(val)

    behavior = data.get("behavior_tags") or []
    if isinstance(behavior, str):
        behavior = [b.strip() for b in behavior.split(",") if b.strip()]
    parts.extend(behavior)

    if data.get("reverb_enabled") and data.get("reverb_amount"):
        parts.append(str(data["reverb_amount"]))
    if data.get("delay_enabled") and data.get("delay_type"):
        parts.append(str(data["delay_type"]))
    if data.get("distortion_enabled") and data.get("distortion_amount"):
        parts.append(str(data["distortion_amount"]))
    if data.get("phaser_enabled") and data.get("phaser_amount"):
        parts.append(str(data["phaser_amount"]))
    if data.get("bitcrush_enabled") and data.get("bitcrush_amount"):
        parts.append(str(data["bitcrush_amount"]))

    bars = data.get("bars", 4)
    foundation_bpm = data.get("_foundation_bpm", 120)
    key_root = (data.get("key_root") or "C").strip()
    key_mode = (data.get("key_mode") or "minor").strip().lower()

    parts.append(f"{bars} Bars")
    parts.append(f"{foundation_bpm} BPM")
    parts.append(f"{key_root} {key_mode}")

    seen = set()
    deduped = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)

    prompt = ", ".join(deduped)
    return prompt if prompt else FALLBACK_PROMPT


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_request(data: dict) -> list:
    errors = []
    bars = data.get("bars")
    if bars is not None and bars not in SUPPORTED_BARS:
        errors.append(f"bars must be one of {SUPPORTED_BARS}")

    key_root = data.get("key_root")
    if key_root and key_root not in VALID_KEY_ROOTS:
        errors.append(f"key_root must be one of {VALID_KEY_ROOTS}")

    key_mode = data.get("key_mode")
    if key_mode and key_mode.lower() not in VALID_KEY_MODES:
        errors.append(f"key_mode must be one of {VALID_KEY_MODES}")

    host_bpm = data.get("host_bpm")
    if host_bpm is not None and (host_bpm < 40 or host_bpm > 300):
        errors.append("host_bpm must be between 40 and 300")

    return errors


# ---------------------------------------------------------------------------
# Model warmup
# ---------------------------------------------------------------------------

def warmup():
    """Validate that model files exist and mark ready.

    On macOS with saomlx, the model is loaded lazily on first generation
    (cached internally by the pipeline). We just verify files are present.
    """
    ckpt = Path(FOUNDATION_CKPT_PATH)
    cfg = Path(FOUNDATION_CONFIG_PATH)

    if not ckpt.exists():
        print(f"[foundation] WARNING: checkpoint not found: {ckpt}")
        print("[foundation] Download from HuggingFace: RoyalCities/Foundation-1")
        print("[foundation] Service will start but generation will fail until model is downloaded")
        # Still mark ready so health check doesn't 503 forever
        model_ready.set()
        return

    if not cfg.exists():
        print(f"[foundation] WARNING: model config not found: {cfg}")
        model_ready.set()
        return

    if not _HAS_SAOMLX:
        print("[foundation] WARNING: saomlx unavailable -- MLX generation will fail")
        model_ready.set()
        return

    with open(cfg, encoding="utf-8") as f:
        config = json.load(f)
    runtime_profile = load_foundation_runtime_profile()

    sr = config.get("sample_rate", 44100)
    ss = config.get("sample_size", 882000)
    print(f"[foundation] Model config validated: sample_rate={sr}, sample_size={ss}")
    print(
        "[foundation] Runtime profile: "
        f"objective={runtime_profile['diffusion_objective']} "
        f"default_sampler={runtime_profile['sampler_type']}"
    )
    print(f"[foundation] Checkpoint: {ckpt}")
    model_ready.set()
    print("[foundation] Ready.")


# ---------------------------------------------------------------------------
# Background generation worker
# ---------------------------------------------------------------------------

def generation_worker(session_id: str, data: dict):
    """Runs in a background thread. Updates session progress as it goes."""
    t_start = time.time()

    acquired = generation_semaphore.acquire(timeout=30)
    if not acquired:
        update_session(session_id,
                       status="failed",
                       generation_in_progress=False,
                       error="GPU busy -- another generation is in progress")
        return

    try:
        if not _HAS_SAOMLX:
            raise RuntimeError("saomlx not available -- cannot generate on this machine")
        if not Path(FOUNDATION_CKPT_PATH).exists() or not Path(FOUNDATION_CONFIG_PATH).exists():
            print(f"[{session_id}] Downloading Foundation-1 model files from {FOUNDATION_HF_REPO}...")
            ensure_foundation_model_files_for_session(session_id)

        seed = data["_seed"]
        bars = data["_bars"]
        host_bpm = data["_host_bpm"]
        foundation_bpm = data["_foundation_bpm"]
        gen_duration = data["_gen_duration"]
        prompt = data["_prompt"]
        guidance_scale = data["_guidance_scale"]
        steps = data["_steps"]
        stretch_ratio_val = data["_stretch_ratio"]
        runtime_profile = load_foundation_runtime_profile()
        sampler_type = runtime_profile["sampler_type"]

        update_session(
            session_id,
            status="generating",
            progress=0,
            queue_status={"status": "ready"},
            status_message="generating foundation-1 audio",
        )

        print(f"[{session_id}] Generate request:")
        print(f"  seed={seed} bars={bars} host_bpm={host_bpm}")
        print(f"  foundation_bpm={foundation_bpm} duration={gen_duration:.2f}s")
        print(f"  stretch_ratio={stretch_ratio_val:.4f}")
        print(f"  steps={steps} guidance={guidance_scale}")
        print(
            "  runtime_profile="
            f"{runtime_profile['diffusion_objective']} / {sampler_type}"
        )
        print(f"  prompt: {prompt}")

        # Progress callback for saomlx
        def on_step(payload):
            if not isinstance(payload, dict):
                return
            step = payload.get("i", 0)
            if isinstance(step, (int, float)):
                step = int(step) + 1
                pct = int(step / steps * 90)
                update_session(
                    session_id,
                    status="generating",
                    step=step,
                    progress=pct,
                    queue_status={"status": "ready"},
                    status_message=f"generating step {step}/{steps}",
                )

        # Run MLX generation
        result = generate_diffusion_cond_mlx(
            model_config_path=FOUNDATION_CONFIG_PATH,
            model_ckpt_path=FOUNDATION_CKPT_PATH,
            prompt=prompt,
            seed=seed,
            steps=steps,
            seconds=gen_duration,
            cfg_scale=guidance_scale,
            batch_size=1,
            sampler_type=sampler_type,
            out_dir=None,  # don't write intermediate files
            step_callback=on_step,
        )

        sample_rate = result["sample_rate"]
        audio_np = result["audio"]  # numpy array

        update_session(
            session_id,
            progress=92,
            status="stretching" if abs(stretch_ratio_val - 1.0) >= 0.001 else "encoding",
            queue_status={
                "status": "processing",
                "message": "stretching audio..." if abs(stretch_ratio_val - 1.0) >= 0.001 else "encoding audio...",
            },
            status_message="stretching audio..." if abs(stretch_ratio_val - 1.0) >= 0.001 else "encoding audio...",
        )

        # saomlx returns [batch, time, channels]. Normalize once here so
        # downstream stretch + WAV encoding consistently see [channels, samples].
        audio = coerce_channels_first_audio(audio_np, context="foundation generation output")
        print(
            f"[{session_id}] foundation output shape {tuple(np.shape(audio_np))} "
            f"-> normalized {tuple(audio.shape)}"
        )

        # Trim to exact duration
        expected_samples = int(gen_duration * sample_rate)
        if audio.shape[-1] > expected_samples:
            audio = audio[:, :expected_samples]

        # Time-stretch to host BPM if needed
        if abs(stretch_ratio_val - 1.0) >= 0.001:
            update_session(
                session_id,
                transform_in_progress=True,
                progress=94,
                queue_status={"status": "processing", "message": "stretching audio..."},
                status_message="stretching audio...",
            )
            audio = apply_time_stretch(audio, stretch_ratio_val, sample_rate)
            host_duration = derive_duration(bars, host_bpm)
            host_samples = int(host_duration * sample_rate)
            if audio.shape[-1] > host_samples:
                audio = audio[:, :host_samples]
            update_session(
                session_id,
                transform_in_progress=False,
                progress=96,
                queue_status={"status": "processing", "message": "encoding audio..."},
                status_message="encoding audio...",
            )

        audio = audio.clamp(-1.0, 1.0)
        final_duration = audio.shape[-1] / sample_rate

        # Encode to base64 WAV
        update_session(
            session_id,
            progress=97,
            queue_status={"status": "processing", "message": "encoding audio..."},
            status_message="encoding audio...",
        )
        filename = f"foundation_{session_id}_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        audio_bytes, audio_b64 = encode_audio_to_wav_bytes(audio, sample_rate, output_path)

        gen_time = time.time() - t_start
        print(f"[{session_id}] Done in {gen_time:.2f}s -> {output_path}")

        update_session(
            session_id,
            status="completed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=100,
            audio_data=audio_b64,
            queue_status={"status": "completed", "message": "foundation-1 generation complete"},
            status_message="foundation-1 generation complete",
            meta={
                **data.get("_original_request", {}),
                "session_id": session_id,
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "gen_duration": round(gen_duration, 4),
                "stretch_ratio": round(stretch_ratio_val, 4),
                "final_duration": round(final_duration, 4),
                "key": f"{data.get('key_root', 'C')} {data.get('key_mode', 'minor')}",
                "prompt": prompt,
                "generation_time": round(gen_time, 2),
                "output_path": output_path,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{session_id}] Error: {e}")
        update_session(
            session_id,
            status="failed",
            generation_in_progress=False,
            transform_in_progress=False,
            error=str(e),
            queue_status={"status": "failed", "message": str(e)},
            status_message=str(e),
        )
    finally:
        generation_semaphore.release()
        gc.collect()


# ---------------------------------------------------------------------------
# Audio2Audio worker
# ---------------------------------------------------------------------------

def audio2audio_worker(
    session_id: str,
    *,
    input_waveform: torch.Tensor,
    input_sr: int,
    prompt: str,
    seed: int,
    bars: int,
    host_bpm: float,
    foundation_bpm: int,
    gen_duration: float,
    stretch_ratio: float,
    init_noise_level: float,
    steps: int,
    guidance_scale: float,
    key_root: str,
    key_mode: str,
):
    """Background worker for audio-to-audio generation."""
    t_start = time.time()

    acquired = generation_semaphore.acquire(timeout=30)
    if not acquired:
        update_session(session_id,
                       status="failed",
                       generation_in_progress=False,
                       error="GPU busy -- another generation is in progress")
        return

    try:
        if not _HAS_SAOMLX:
            raise RuntimeError("saomlx not available -- cannot generate on this machine")
        if not Path(FOUNDATION_CKPT_PATH).exists() or not Path(FOUNDATION_CONFIG_PATH).exists():
            print(f"[{session_id}] Downloading Foundation-1 model files from {FOUNDATION_HF_REPO}...")
            ensure_foundation_model_files_for_session(session_id)
        runtime_profile = load_foundation_runtime_profile()
        sampler_type = runtime_profile["sampler_type"]

        update_session(
            session_id,
            status="generating",
            progress=0,
            queue_status={"status": "ready"},
            status_message="generating foundation-1 audio2audio",
        )

        print(f"[{session_id}] Audio2Audio request:")
        print(f"  seed={seed} bars={bars} host_bpm={host_bpm}")
        print(f"  foundation_bpm={foundation_bpm} duration={gen_duration:.2f}s")
        print(f"  init_noise_level={init_noise_level} steps={steps} guidance={guidance_scale}")
        print(
            "  runtime_profile="
            f"{runtime_profile['diffusion_objective']} / {sampler_type}"
        )
        print(f"  prompt: {prompt}")

        def on_step(payload):
            if not isinstance(payload, dict):
                return
            step = payload.get("i", 0)
            if isinstance(step, (int, float)):
                step = int(step) + 1
                pct = int(step / steps * 90)
                update_session(
                    session_id,
                    status="generating",
                    step=step,
                    progress=pct,
                    queue_status={"status": "ready"},
                    status_message=f"generating step {step}/{steps}",
                )

        update_session(
            session_id,
            status="preparing_init_audio",
            progress=5,
            queue_status={"status": "processing", "message": "preparing init audio..."},
            status_message="preparing init audio...",
        )
        result = generate_diffusion_cond_mlx(
            model_config_path=FOUNDATION_CONFIG_PATH,
            model_ckpt_path=FOUNDATION_CKPT_PATH,
            prompt=prompt,
            seed=seed,
            steps=steps,
            seconds=gen_duration,
            cfg_scale=guidance_scale,
            batch_size=1,
            init_audio=(input_sr, input_waveform),
            init_noise_level=init_noise_level,
            sampler_type=sampler_type,
            out_dir=None,
            step_callback=on_step,
        )

        sample_rate = result["sample_rate"]
        audio_np = result["audio"]

        update_session(
            session_id,
            progress=92,
            status="stretching" if abs(stretch_ratio - 1.0) >= 0.001 else "encoding",
            queue_status={
                "status": "processing",
                "message": "stretching audio..." if abs(stretch_ratio - 1.0) >= 0.001 else "encoding audio...",
            },
            status_message="stretching audio..." if abs(stretch_ratio - 1.0) >= 0.001 else "encoding audio...",
        )

        audio = coerce_channels_first_audio(audio_np, context="foundation audio2audio output")
        print(
            f"[{session_id}] foundation audio2audio output shape {tuple(np.shape(audio_np))} "
            f"-> normalized {tuple(audio.shape)}"
        )

        expected_samples = int(gen_duration * sample_rate)
        if audio.shape[-1] > expected_samples:
            audio = audio[:, :expected_samples]

        if abs(stretch_ratio - 1.0) >= 0.001:
            update_session(
                session_id,
                transform_in_progress=True,
                progress=94,
                queue_status={"status": "processing", "message": "stretching audio..."},
                status_message="stretching audio...",
            )
            audio = apply_time_stretch(audio, stretch_ratio, sample_rate)
            host_duration = derive_duration(bars, host_bpm)
            host_samples = int(host_duration * sample_rate)
            if audio.shape[-1] > host_samples:
                audio = audio[:, :host_samples]
            update_session(
                session_id,
                transform_in_progress=False,
                progress=96,
                queue_status={"status": "processing", "message": "encoding audio..."},
                status_message="encoding audio...",
            )

        audio = audio.clamp(-1.0, 1.0)
        final_duration = audio.shape[-1] / sample_rate

        update_session(
            session_id,
            progress=97,
            queue_status={"status": "processing", "message": "encoding audio..."},
            status_message="encoding audio...",
        )
        filename = f"foundation_a2a_{session_id}_{seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        audio_bytes, audio_b64 = encode_audio_to_wav_bytes(audio, sample_rate, output_path)

        gen_time = time.time() - t_start
        print(f"[{session_id}] Audio2Audio done in {gen_time:.2f}s -> {output_path}")

        update_session(
            session_id,
            status="completed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=100,
            audio_data=audio_b64,
            queue_status={"status": "completed", "message": "foundation-1 audio2audio complete"},
            status_message="foundation-1 audio2audio complete",
            meta={
                "session_id": session_id,
                "mode": "audio2audio",
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "init_noise_level": init_noise_level,
                "gen_duration": round(gen_duration, 4),
                "stretch_ratio": round(stretch_ratio, 4),
                "final_duration": round(final_duration, 4),
                "key": f"{key_root} {key_mode}",
                "prompt": prompt,
                "generation_time": round(gen_time, 2),
                "output_path": output_path,
            },
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{session_id}] Audio2Audio error: {e}")
        update_session(
            session_id,
            status="failed",
            generation_in_progress=False,
            transform_in_progress=False,
            error=str(e),
            queue_status={"status": "failed", "message": str(e)},
            status_message=str(e),
        )
    finally:
        generation_semaphore.release()
        gc.collect()


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/models", methods=["GET"])
def get_available_models():
    return jsonify({
        "success": True,
        "models": {
            "small": [],
            "medium": [],
            "large": [
                {
                    "name": FOUNDATION_MODEL_DISPLAY_NAME,
                    "path": FOUNDATION_HF_REPO,
                    "type": "single",
                }
            ],
        },
        "updated_at": time.time(),
    })


@app.route("/api/models/download_status", methods=["GET"])
def get_models_download_status():
    requested_model = (request.args.get("model_name") or "").strip()
    if requested_model and requested_model not in {FOUNDATION_HF_REPO, FOUNDATION_MODEL_DISPLAY_NAME}:
        return jsonify({
            "success": False,
            "error": f"Unknown model '{requested_model}'",
        }), 404

    return jsonify({
        "success": True,
        "models": {
            FOUNDATION_HF_REPO: foundation_model_download_status(),
        },
        "updated_at": time.time(),
    })


@app.route("/api/models/predownload", methods=["POST"])
def start_model_predownload():
    payload = request.get_json(silent=True) or {}
    model_name = str(payload.get("model_name") or "").strip()
    if not model_name:
        return jsonify({
            "success": False,
            "error": "model_name is required",
        }), 400
    if model_name not in {FOUNDATION_HF_REPO, FOUNDATION_MODEL_DISPLAY_NAME}:
        return jsonify({
            "success": False,
            "error": f"Unknown model '{model_name}'",
        }), 404

    session_id = str(uuid.uuid4())
    with model_download_lock:
        model_download_sessions[session_id] = {
            "session_id": session_id,
            "model_name": FOUNDATION_HF_REPO,
            "status": "queued",
            "progress": 0,
            "queue_status": foundation_download_queue_status(
                status="queued",
                message="queued for download",
                stage_name="queued",
                stage_index=0,
                stage_total=2,
                download_percent=0,
            ),
            "error": None,
            "created_at": time.time(),
        }
    run_foundation_model_predownload(session_id)

    return jsonify({
        "success": True,
        "session_id": session_id,
        "model_name": FOUNDATION_HF_REPO,
        "message": f"Started pre-download for {FOUNDATION_MODEL_DISPLAY_NAME}",
    })


@app.route("/api/models/predownload_status/<session_id>", methods=["GET"])
def get_model_predownload_status(session_id: str):
    session = get_model_download_session(session_id)
    if session is None:
        return jsonify({
            "success": False,
            "error": "predownload session not found",
        }), 404

    response = {
        "success": True,
        "session_id": session_id,
        "model_name": session.get("model_name", FOUNDATION_HF_REPO),
        "status": session.get("status", "unknown"),
        "progress": max(0, min(100, int(session.get("progress", 0)))),
        "queue_status": session.get("queue_status") or {},
    }
    if response["status"] == "failed":
        response["error"] = session.get("error", "unknown error")
    return jsonify(response)


@app.route("/health", methods=["GET"])
def health():
    if model_ready.is_set():
        ckpt_exists = Path(FOUNDATION_CKPT_PATH).exists()
        runtime_profile = load_foundation_runtime_profile()
        return jsonify({
            "status": "healthy" if ckpt_exists else "no_model",
            "model_loaded": ckpt_exists,
            "saomlx_available": _HAS_SAOMLX,
            "checkpoint": FOUNDATION_CKPT_PATH,
            "config": FOUNDATION_CONFIG_PATH,
            "diffusion_objective": runtime_profile["diffusion_objective"],
            "default_sampler_type": runtime_profile["sampler_type"],
            "uses_pingpong": runtime_profile["sampler_type"] == "pingpong",
        })
    return jsonify({"status": "starting", "model_loaded": False}), 503


@app.route("/ready", methods=["GET"])
def ready():
    if model_ready.is_set():
        return jsonify({"ready": True}), 200
    return jsonify({"ready": False}), 503


@app.route("/generate", methods=["POST"])
def generate():
    cleanup_old_sessions()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        errors = validate_request(data)
        if errors:
            return jsonify({"success": False, "errors": errors}), 400

        if not model_ready.is_set():
            return jsonify({"success": False, "error": "loading model -- warming up"}), 503

        seed = int(data.get("seed", -1))
        bars = int(data.get("bars", 4))
        host_bpm = float(data.get("host_bpm", 120.0))
        guidance_scale = float(data.get("guidance_scale", 7.0))
        steps = int(data.get("steps", 100))
        custom_override = (data.get("custom_prompt_override") or "").strip()
        key_root = data.get("key_root", "C")
        key_mode = data.get("key_mode", "minor")

        foundation_bpm = nearest_foundation_bpm(host_bpm)
        data["_foundation_bpm"] = foundation_bpm
        gen_duration = derive_duration(bars, foundation_bpm)

        if custom_override:
            prompt = custom_override
            if "Bars" not in prompt:
                prompt += f", {bars} Bars"
            if "BPM" not in prompt:
                prompt += f", {foundation_bpm} BPM"
            if key_root.lower() not in prompt.lower():
                prompt += f", {key_root} {key_mode}"
        else:
            prompt = build_prompt(data)

        if seed == -1:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        stretch_ratio_val = time_stretch_ratio(host_bpm, foundation_bpm)

        session_id = str(uuid.uuid4())[:12]

        data["_seed"] = seed
        data["_bars"] = bars
        data["_host_bpm"] = host_bpm
        data["_foundation_bpm"] = foundation_bpm
        data["_gen_duration"] = gen_duration
        data["_prompt"] = prompt
        data["_guidance_scale"] = guidance_scale
        data["_steps"] = steps
        data["_stretch_ratio"] = stretch_ratio_val
        data["_original_request"] = {
            k: v for k, v in data.items() if not k.startswith("_")
        }

        create_session(session_id, {
            "steps": steps,
            "prompt": prompt,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
        })

        thread = threading.Thread(
            target=generation_worker,
            args=(session_id, data),
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
            "gen_duration": round(gen_duration, 4),
            "stretch_ratio": round(stretch_ratio_val, 4),
            "prompt": prompt,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/audio2audio", methods=["POST"])
def audio2audio():
    cleanup_old_sessions()

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "JSON body required"}), 400

        audio_b64 = (data.get("audio_data") or "").strip()
        if not audio_b64:
            return jsonify({"success": False, "error": "audio_data (base64 WAV) is required"}), 400

        host_bpm = data.get("host_bpm")
        if host_bpm is None:
            return jsonify({"success": False, "error": "host_bpm is required"}), 400
        host_bpm = float(host_bpm)

        if not model_ready.is_set():
            return jsonify({"success": False, "error": "loading model -- warming up"}), 503

        bars = int(data.get("bars", 8))
        init_noise_level = float(data.get("init_noise_level", 0.25))
        init_noise_level = max(0.01, min(1.0, init_noise_level))
        seed = int(data.get("seed", -1))
        steps = int(data.get("steps", 75))
        guidance_scale = float(data.get("guidance_scale", 7.0))
        key_root = data.get("key_root", "C")
        key_mode = data.get("key_mode", "minor")

        foundation_bpm = nearest_foundation_bpm(host_bpm)
        stretch_ratio_val = time_stretch_ratio(host_bpm, foundation_bpm)
        gen_duration = derive_duration(bars, foundation_bpm)

        custom_override = (data.get("custom_prompt_override") or data.get("prompt") or "").strip()
        data["_foundation_bpm"] = foundation_bpm
        if custom_override:
            prompt_text = custom_override
            if "Bars" not in prompt_text:
                prompt_text += f", {bars} Bars"
            if "BPM" not in prompt_text:
                prompt_text += f", {foundation_bpm} BPM"
            if key_root.lower() not in prompt_text.lower():
                prompt_text += f", {key_root} {key_mode}"
        else:
            prompt_text = build_prompt(data)

        # Decode input audio
        audio_bytes = base64.b64decode(audio_b64)
        audio_buf = io.BytesIO(audio_bytes)
        input_waveform, input_sr = torchaudio.load(audio_buf)

        # Pre-stretch: host_bpm -> foundation_bpm
        pre_stretch_ratio = foundation_bpm / host_bpm
        if abs(pre_stretch_ratio - 1.0) >= 0.001:
            input_waveform = apply_time_stretch(input_waveform, pre_stretch_ratio, int(input_sr))

        if seed == -1:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        session_id = str(uuid.uuid4())[:12]

        create_session(session_id, {
            "steps": steps,
            "prompt": prompt_text,
            "seed": seed,
            "bars": bars,
            "host_bpm": host_bpm,
            "foundation_bpm": foundation_bpm,
            "mode": "audio2audio",
        })

        thread = threading.Thread(
            target=audio2audio_worker,
            args=(session_id,),
            kwargs={
                "input_waveform": input_waveform,
                "input_sr": int(input_sr),
                "prompt": prompt_text,
                "seed": seed,
                "bars": bars,
                "host_bpm": host_bpm,
                "foundation_bpm": foundation_bpm,
                "gen_duration": gen_duration,
                "stretch_ratio": stretch_ratio_val,
                "init_noise_level": init_noise_level,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "key_root": key_root,
                "key_mode": key_mode,
            },
            daemon=True,
        )
        thread.start()

        return jsonify({
            "success": True,
            "session_id": session_id,
            "seed": seed,
            "foundation_bpm": foundation_bpm,
            "init_noise_level": init_noise_level,
            "prompt": prompt_text,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/randomize", methods=["POST"])
def randomize():
    import random as stdlib_random

    try:
        rc = _load_rc_prompt()
    except Exception as e:
        return jsonify({"success": False, "error": f"RC prompt engine unavailable: {e}"}), 500

    data = request.get_json(silent=True) or {}

    seed = int(data.get("seed", -1))
    if seed == -1:
        seed = stdlib_random.randint(0, 2**31 - 1)

    mode = (data.get("mode") or "standard").strip().lower()
    variant = (data.get("variant") or "auto").strip()
    family_hint = (data.get("family_hint") or "").strip() or None

    # Run RC's engine to get the structured anchor
    vt = rc.choose_variant_type(mode=mode, variant=variant)
    profile = rc.normalize_mode_to_profile(mode)
    if vt == "T1":
        profile = "mix"

    base_rng = stdlib_random.Random(seed)
    anchor = rc.build_anchor(base_rng, profile=profile, family_hint=family_hint)

    family = str(anchor["family"])
    subfamily = str(anchor["sub"])
    tags = list(anchor["tags"])
    fx_tokens = list(anchor["fx"])
    melody_str = str(anchor["melody"])

    # Decompose FX tokens
    reverb_value = ""
    delay_value = ""
    distortion_value = ""
    phaser_value = ""
    bitcrush_value = ""

    reverb_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["reverb"])}
    delay_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["delay"])}
    distortion_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["distortion"])}
    phaser_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["phaser"])}
    bitcrush_items = {t.lower() for t, _ in zip(*rc.FX_BY_CAT["bitcrush"])}

    for tok in fx_tokens:
        tok_lower = tok.lower()
        if tok_lower in reverb_items:
            reverb_value = tok
        elif tok_lower in delay_items:
            delay_value = tok
        elif tok_lower in distortion_items:
            distortion_value = tok
        elif tok_lower in phaser_items:
            phaser_value = tok
        elif tok_lower in bitcrush_items:
            bitcrush_value = tok

    # Decompose tags into descriptor buckets
    timbre_set = {t.lower() for t in rc.TIMBRE_TAGS}
    spatial_set = {t.lower() for t in rc.SPATIAL_TAGS}
    band_set = {t.lower() for t in rc.BAND_TAGS}
    wave_set = {t.lower() for t in rc.WAVE_TECH_TAGS}
    style_set = {t.lower() for t in rc.STYLE_TAGS}

    descriptors = []
    spatial = []
    band = []
    wave_tech = []
    style = []

    for t in tags:
        tl = t.lower()
        if tl in timbre_set:
            descriptors.append(t)
        elif tl in spatial_set:
            spatial.append(t)
        elif tl in band_set:
            band.append(t)
        elif tl in wave_set:
            wave_tech.append(t)
        elif tl in style_set:
            style.append(t)
        else:
            descriptors.append(t)

    # Decompose melody string
    melody_parts = [m.strip() for m in melody_str.split(",") if m.strip()]

    speed_set = {s.lower() for s in rc.SPEED}
    rhythm_set = {r.lower() for r in rc.RHYTHM}
    contour_set = {c.lower() for c in rc.CONTOUR}
    density_set = {d.lower() for d in rc.DENSITY}
    structure_set = {s.lower() for s in rc.STRUCTURE_GENERIC + ["bassline"]}

    speed_val = ""
    structure_val = ""
    rhythm_vals = []
    contour_vals = []
    density_vals = []

    for part in melody_parts:
        pl = part.lower()
        if pl in speed_set:
            speed_val = part
        elif pl in structure_set:
            structure_val = part
        elif pl in rhythm_set:
            rhythm_vals.append(part)
        elif pl in contour_set:
            contour_vals.append(part)
        elif pl in density_set:
            density_vals.append(part)

    # Build the full prompt string
    full_prompt = rc.prompt_generator_variants(
        seed=seed, mode=mode, variant=variant,
        allow_timbre_mix=True, family_hint=family_hint,
    )

    return jsonify({
        "success": True,
        "seed": seed,
        "mode": mode,
        "variant": vt,
        "family": family,
        "subfamily": subfamily,
        "descriptor_knob_a": descriptors[0] if len(descriptors) > 0 else "",
        "descriptor_knob_b": descriptors[1] if len(descriptors) > 1 else "",
        "descriptor_knob_c": descriptors[2] if len(descriptors) > 2 else "",
        "descriptors_extra": descriptors[3:],
        "reverb_enabled": reverb_value != "",
        "reverb_amount": reverb_value,
        "delay_enabled": delay_value != "",
        "delay_type": delay_value,
        "distortion_enabled": distortion_value != "",
        "distortion_amount": distortion_value,
        "phaser_enabled": phaser_value != "",
        "phaser_amount": phaser_value,
        "bitcrush_enabled": bitcrush_value != "",
        "bitcrush_amount": bitcrush_value,
        "behavior_tags": melody_parts,
        "speed": speed_val,
        "structure": structure_val,
        "rhythm": rhythm_vals,
        "contour": contour_vals,
        "density": density_vals,
        "spatial_tags": spatial,
        "band_tags": band,
        "wave_tech_tags": wave_tech,
        "style_tags": style,
        "all_tags": tags,
        "prompt": full_prompt,
    })


@app.route("/poll_status/<session_id>", methods=["GET"])
def poll_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        return jsonify({
            "success": False,
            "error": f"unknown session: {session_id}",
        }), 404

    status = session["status"]
    gen_in_progress = session["generation_in_progress"]
    xform_in_progress = session["transform_in_progress"]
    progress = session["progress"]

    queue_status = session.get("queue_status") or {}
    if not queue_status and status == "queued":
        queue_status = {
            "status": "queued",
            "position": 1,
            "message": "waiting for GPU",
            "estimated_time": "~5s",
            "estimated_seconds": 5,
        }
    elif not queue_status and status == "downloading_model":
        queue_status = {"status": "warming", "message": "downloading model"}
    elif not queue_status and status == "preparing_init_audio":
        queue_status = {"status": "ready", "message": "preparing init audio"}
    elif not queue_status and status in ("generating", "stretching", "encoding"):
        queue_status = {"status": "ready"}

    response = {
        "success": True,
        "generation_in_progress": gen_in_progress,
        "transform_in_progress": xform_in_progress,
        "progress": progress,
        "status": status,
        "queue_status": queue_status,
        "status_message": session.get("status_message", queue_status.get("message", "")),
    }

    if status == "completed":
        response["audio_data"] = session.get("audio_data", "")
        response["meta"] = session.get("meta", {})
    elif status == "failed":
        response["success"] = False
        response["error"] = session.get("error", "unknown error")

    return jsonify(response)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warmup_thread = threading.Thread(target=warmup, daemon=True)
    warmup_thread.start()

    print(f"[foundation] Starting Foundation-1 API on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
