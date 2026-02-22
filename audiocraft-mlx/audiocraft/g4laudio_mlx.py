from __future__ import annotations

import os

# Prefer predictable HTTP transfer path for large model checkpoints on macOS.
# These must be set before importing any library that may import
# `huggingface_hub` (and therefore cache env-derived constants).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
# Retain as opt-in capability if users override `HF_HUB_DISABLE_XET=0`.
# On some macOS networks, hf_xet high-performance mode can dramatically delay
# time-to-first-byte, so default to the safer non-HP profile.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "0")
# Empirically improves xet time-to-first-byte on large checkpoints on macOS.
os.environ.setdefault("HF_XET_NUM_RANGE_IN_SEGMENT_BASE", "4")

import base64
import io
import json
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from mlx.utils import tree_flatten
from scipy.signal import resample_poly

from g4l_models import (
    OUTPUT_DURATION_S,
    get_base_model_for_finetune,
    has_explicit_base_model_override,
    get_model_description,
)
from mlx_continuation.encodec import preprocess_audio as encodec_preprocess_audio
from mlx_continuation.mlx_musicgen import MusicGenContinuation


class AudioProcessingError(Exception):
    """Localhost-compatible error wrapper (mirrors g4laudio.py)."""


_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, Optional[str], str], MusicGenContinuation] = {}
_HF_HUB_MODE_LOCK = threading.Lock()
_HF_XET_PROBE_LOCK = threading.Lock()
_HF_XET_PROBE_RESULTS: dict[tuple[str, str], bool] = {}
_HF_XET_GLOBAL_HEALTHY: Optional[bool] = None
_HF_DOWNLOADER_LOCK = threading.Lock()
_HF_DOWNLOADER_PYTHON_PATH: Optional[str] = None

# For now, we serialize generation to keep MLX state/threading simple.
_GEN_LOCK = threading.Lock()

_MODEL_SNAPSHOT_PATTERNS = ["*.json", "state_dict.bin"]
_CONFIG_SNAPSHOT_PATTERNS = ["config.json"]
_COMPONENT_SNAPSHOT_PATTERNS = ["*.json", "*.safetensors", "*.model"]
_TOKENIZER_SNAPSHOT_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "*.model",
    "*.json",
]

DEFAULT_QUANTIZATION_MODE = os.environ.get(
    "G4L_MLX_QUANTIZATION_DEFAULT",
    "q4_decoder_linears",
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


def _resolve_snapshot_max_workers() -> int:
    raw = os.environ.get("G4L_HF_SNAPSHOT_MAX_WORKERS", "64")
    try:
        parsed = int(str(raw).strip())
    except Exception:
        return 8
    return max(1, min(parsed, 64))


_HF_SNAPSHOT_MAX_WORKERS = _resolve_snapshot_max_workers()
_HF_DOWNLOAD_RUNTIME_LOGGED = False


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
    print(f"[HF] shared downloader bootstrap: uv installed at {resolved}")
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
    global _HF_DOWNLOADER_PYTHON_PATH

    with _HF_DOWNLOADER_LOCK:
        if _HF_DOWNLOADER_PYTHON_PATH and os.path.exists(_HF_DOWNLOADER_PYTHON_PATH):
            return _HF_DOWNLOADER_PYTHON_PATH

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
                uv_path,
                "pip",
                "install",
                "--python",
                venv_python,
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ])
            _run_command_checked(
                [uv_path, "pip", "install", "--python", venv_python, "--upgrade"]
                + HF_DOWNLOADER_PACKAGES
            )
            with open(marker_path, "w", encoding="utf-8") as f:
                json.dump(desired_marker, f)

        _HF_DOWNLOADER_PYTHON_PATH = venv_python
        print(f"[HF] shared downloader env ready: {venv_python}")
        return venv_python


def _download_with_shared_hf_downloader_env(
    *,
    repo_id: str,
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

        worker_cmd = [downloader_python, worker_path, "--repo-id", repo_id, "--filename", filename]
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
            raise RuntimeError("failed to capture downloader worker output")

        worker_error = None
        started_at = time.time()
        first_byte_at = None
        slow_since = None

        while True:
            if use_xet:
                now = time.time()
                if first_byte_at is None and (now - started_at) > HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS:
                    terminate_process(process)
                    print(
                        f"[HF] shared downloader xet timeout: no first byte for {repo_id}/{filename} "
                        f"after {HF_DOWNLOADER_XET_FIRST_BYTE_TIMEOUT_SECONDS:.1f}s"
                    )
                    return False, "xet_no_first_byte", backend_label
                if first_byte_at is not None and slow_since is not None and (
                    now - slow_since
                ) > HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS:
                    terminate_process(process)
                    print(
                        f"[HF] shared downloader xet slow throughput for {repo_id}/{filename}: "
                        f"< {HF_DOWNLOADER_XET_SLOW_SPEED_BPS} B/s for {HF_DOWNLOADER_XET_SLOW_SPEED_GRACE_SECONDS:.1f}s"
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
                    elif slow_since is None:
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
            raise RuntimeError("shared downloader env failed in HTTP mode")
        return backend_label or "shared downloader env (http)"

    if mode == "on":
        success, _, backend_label = run_worker(use_xet=True)
        if not success:
            raise RuntimeError("shared downloader env failed in forced XET mode")
        return backend_label or "shared downloader env (xet)"

    # adaptive (default): try XET first, fallback to HTTP on poor startup or throughput
    success, fallback_reason, backend_label = run_worker(use_xet=True)
    if success:
        return backend_label or "shared downloader env (xet)"

    print(
        f"[HF] shared downloader adaptive fallback for {repo_id}/{filename}: "
        f"{fallback_reason or 'unknown reason'} -> http"
    )
    success_http, _, backend_label_http = run_worker(
        use_xet=False,
        force_download=True,
    )
    if not success_http:
        raise RuntimeError("shared downloader env failed in HTTP fallback mode")
    return backend_label_http or "shared downloader env (http fallback)"


def _resolve_xet_mode() -> str:
    raw = str(os.environ.get("G4L_HF_XET_MODE", "adaptive")).strip().lower()
    if raw in {"adaptive", "on", "off"}:
        return raw
    return "adaptive"


def _resolve_xet_probe_timeout_seconds() -> float:
    raw = os.environ.get("G4L_HF_XET_PROBE_TIMEOUT_SECONDS", "20")
    try:
        parsed = float(str(raw).strip())
    except Exception:
        return 20.0
    return max(3.0, min(parsed, 180.0))


_HF_XET_MODE = _resolve_xet_mode()
_HF_XET_PROBE_TIMEOUT_SECONDS = _resolve_xet_probe_timeout_seconds()

_QUANTIZATION_PRESETS: dict[str, dict[str, Any]] = {
    "none": {"enabled": False},
    "q8_decoder_linears": {
        "enabled": True,
        "scope": "decoder-linears",
        "bits": 8,
        "group_size": 64,
        "mode": "affine",
    },
    "q4_decoder_linears": {
        "enabled": True,
        "scope": "decoder-linears",
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    },
    "q4_decoder_linears_emb": {
        "enabled": True,
        "scope": "decoder-linears+emb",
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    },
}

_QUANTIZATION_ALIASES: dict[str, str] = {
    "off": "none",
    "baseline": "none",
    "no_quant": "none",
    "false": "none",
    "0": "none",
    "q8": "q8_decoder_linears",
    "q8_decoder": "q8_decoder_linears",
    "q8_linears": "q8_decoder_linears",
    "q4": "q4_decoder_linears",
    "q4_decoder": "q4_decoder_linears",
    "q4_linears": "q4_decoder_linears",
    "q4_decoder_linears_emb": "q4_decoder_linears_emb",
    "q4_decoder_linears_embedding": "q4_decoder_linears_emb",
    "q4_decoder_linears_and_emb": "q4_decoder_linears_emb",
    "q4_decoder_linears_embs": "q4_decoder_linears_emb",
    "q4_linears_emb": "q4_decoder_linears_emb",
    "q4_linears_embeddings": "q4_decoder_linears_emb",
}


def _canonicalize_quantization_mode(mode: str) -> str:
    return (
        mode.strip()
        .lower()
        .replace("-", "_")
        .replace("+", "_")
        .replace(" ", "_")
    )


def _normalize_quantization_mode(requested: Optional[str]) -> str:
    raw = (
        requested
        if requested is not None and str(requested).strip() != ""
        else DEFAULT_QUANTIZATION_MODE
    )
    candidate = _canonicalize_quantization_mode(str(raw))
    mode = _QUANTIZATION_ALIASES.get(candidate, candidate)
    if mode in _QUANTIZATION_PRESETS:
        return mode

    if requested is None:
        print(
            f"[WARN] Invalid G4L_MLX_QUANTIZATION_DEFAULT='{raw}', using 'none'. "
            f"Valid modes: {sorted(_QUANTIZATION_PRESETS.keys())}"
        )
        return "none"

    raise AudioProcessingError(
        f"Invalid quantization_mode '{requested}'. "
        f"Valid modes: {', '.join(sorted(_QUANTIZATION_PRESETS.keys()))}"
    )


def _make_quantization_predicate(scope: str, group_size: int, bits: int, mode: str):
    params = {"group_size": group_size, "bits": bits, "mode": mode}

    def _matches_scope(path: str, module: nn.Module) -> bool:
        if scope == "decoder-linears":
            return isinstance(module, nn.Linear) and (
                path.startswith("layers.") or path.startswith("linears.")
            )
        if scope == "decoder-linears+emb":
            if isinstance(module, nn.Linear) and (
                path.startswith("layers.") or path.startswith("linears.")
            ):
                return True
            return isinstance(module, nn.Embedding) and path.startswith("emb.")
        return False

    def _supports_group_size(module: nn.Module) -> bool:
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) < 2:
            return False
        return int(weight.shape[-1]) % group_size == 0

    def _predicate(path: Optional[str], module: nn.Module):
        p = path or ""
        if not _matches_scope(p, module):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if not _supports_group_size(module):
            return False
        return params

    return _predicate


def _eval_model_parameters(model: MusicGenContinuation) -> None:
    params = [v for _, v in tree_flatten(model.parameters()) if hasattr(v, "shape")]
    if params:
        mx.eval(*params)


def _apply_quantization_preset(model: MusicGenContinuation, quantization_mode: str) -> None:
    preset = _QUANTIZATION_PRESETS[quantization_mode]
    if not bool(preset.get("enabled", False)):
        return

    predicate = _make_quantization_predicate(
        scope=str(preset["scope"]),
        group_size=int(preset["group_size"]),
        bits=int(preset["bits"]),
        mode=str(preset["mode"]),
    )
    nn.quantize(model, class_predicate=predicate)
    _eval_model_parameters(model)


def _load_audio_base64(audio_b64: str) -> Tuple[np.ndarray, int]:
    try:
        wav_bytes = base64.b64decode(audio_b64)
        audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio, int(sr)
    except Exception as e:
        raise AudioProcessingError(f"Failed to load audio: {e}") from e


def _save_audio_base64(audio: np.ndarray, sr: int) -> str:
    try:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        audio = np.nan_to_num(audio)
        audio = np.clip(audio, -1.0, 1.0)
        buf = io.BytesIO()
        # Keep float WAVs (matches typical torchaudio.save float32 behavior).
        sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        raise AudioProcessingError(f"Failed to encode output audio: {e}") from e


def _resample(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return audio.astype(np.float32, copy=False)
    return resample_poly(audio, out_sr, in_sr, axis=0).astype(np.float32)


def _convert_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == target_channels:
        return audio
    if target_channels == 1:
        return audio.mean(axis=1, keepdims=True).astype(np.float32)
    if audio.shape[1] == 1:
        return np.repeat(audio, target_channels, axis=1).astype(np.float32)
    return audio[:, :target_channels].astype(np.float32)


def _flatten_codes(encoded_frames: mx.array) -> mx.array:
    # encoded_frames: (chunks, B, K, frames)
    if encoded_frames.ndim != 4:
        raise ValueError("Expected encoded_frames with 4 dims (chunks, B, K, frames)")
    codes = mx.transpose(encoded_frames, (1, 2, 0, 3))
    bsz, num_codebooks, num_chunks, frames = codes.shape
    return codes.reshape(bsz, num_codebooks, num_chunks * frames)


def _samples_to_frames(num_samples: int, sample_rate: int, frame_rate: float) -> int:
    return int(round((num_samples / sample_rate) * frame_rate))


def _decoder_num_codebooks(model: MusicGenContinuation) -> int:
    quantizer = getattr(model._audio_decoder, "quantizer", None)
    layers = getattr(quantizer, "layers", None)
    if layers is not None:
        return int(len(layers))
    return int(model.num_codebooks)


def _select_prompt_encode_bandwidth(
    model: MusicGenContinuation,
    target_codebooks: Optional[int] = None,
) -> Optional[float]:
    quantizer = getattr(model._audio_decoder, "quantizer", None)
    config = getattr(model._audio_decoder, "config", None)
    target_bandwidths = list(getattr(config, "target_bandwidths", []) or [])
    if quantizer is None or not target_bandwidths:
        return None

    if target_codebooks is None:
        target_codebooks = int(model.num_codebooks)
    best_bandwidth: Optional[float] = None
    best_distance: Optional[int] = None

    for bw in target_bandwidths:
        bw_float = float(bw)
        num_quantizers = int(quantizer.get_num_quantizers_for_bandwidth(bw_float))
        distance = abs(num_quantizers - target_codebooks)
        if distance == 0:
            return bw_float
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_bandwidth = bw_float

    return best_bandwidth


def _encode_prompt_with_decoder(
    model: MusicGenContinuation,
    audio: np.ndarray,
    encode_bandwidth: Optional[float],
) -> mx.array:
    audio_mx = mx.array(audio)
    inputs, masks = encodec_preprocess_audio(
        audio_mx,
        sampling_rate=model.sampling_rate,
        chunk_length=model._audio_decoder.chunk_length,
        chunk_stride=model._audio_decoder.chunk_stride,
    )
    encoded_frames, _ = model._audio_decoder.encode(
        inputs,
        masks,
        bandwidth=encode_bandwidth,
    )
    return _flatten_codes(encoded_frames)


def _align_prompt_codebooks(model: MusicGenContinuation, prompt_tokens: mx.array) -> mx.array:
    target_codebooks = int(model.num_codebooks)
    current_codebooks = int(prompt_tokens.shape[1])
    if current_codebooks == target_codebooks:
        return prompt_tokens

    frames = int(prompt_tokens.shape[-1])
    if current_codebooks > target_codebooks:
        print(
            f"[WARN] Prompt codebooks ({current_codebooks}) > model codebooks ({target_codebooks}); truncating."
        )
        return prompt_tokens[:, :target_codebooks, :]

    print(
        f"[WARN] Prompt codebooks ({current_codebooks}) < model codebooks ({target_codebooks}); padding with BOS."
    )
    bos_pad = mx.full(
        (int(prompt_tokens.shape[0]), target_codebooks - current_codebooks, frames),
        model.bos_token_id,
        dtype=prompt_tokens.dtype,
    )
    return mx.concatenate([prompt_tokens, bos_pad], axis=1)


def _encode_prompt_tokens(model: MusicGenContinuation, prompt_audio: np.ndarray, prompt_sr: int) -> mx.array:
    """
    Convert prompt waveform (T, C) at prompt_sr to prompt tokens (1, K, T_frames).
    """
    audio = _resample(prompt_audio, prompt_sr, model.sampling_rate)
    num_samples = audio.shape[0]
    prompt_frames_target = _samples_to_frames(num_samples, model.sampling_rate, model.frame_rate)

    decoder_channels = int(getattr(model._audio_decoder, "channels", 1))
    decoder_codebooks = _decoder_num_codebooks(model)
    model_codebooks = int(model.num_codebooks)

    # Stereo fine-tunes can expose doubled LM codebooks while the EnCodec decoder
    # remains mono. Mirror AudioCraft's interleaving by encoding L/R independently.
    is_stereo_interleaved = (
        decoder_channels == 1 and model_codebooks == decoder_codebooks * 2
    )

    if is_stereo_interleaved:
        audio = _convert_channels(audio, 2)
        encode_bandwidth = _select_prompt_encode_bandwidth(
            model,
            target_codebooks=decoder_codebooks,
        )
        left_tokens = _encode_prompt_with_decoder(
            model,
            audio[:, 0:1],
            encode_bandwidth,
        )
        right_tokens = _encode_prompt_with_decoder(
            model,
            audio[:, 1:2],
            encode_bandwidth,
        )
        pairwise = mx.stack([left_tokens, right_tokens], axis=2)
        prompt_tokens = pairwise.reshape(
            int(pairwise.shape[0]),
            int(pairwise.shape[1]) * int(pairwise.shape[2]),
            int(pairwise.shape[3]),
        )
    else:
        audio = _convert_channels(audio, decoder_channels)
        encode_bandwidth = _select_prompt_encode_bandwidth(
            model,
            target_codebooks=model_codebooks,
        )
        prompt_tokens = _encode_prompt_with_decoder(
            model,
            audio,
            encode_bandwidth,
        )

    # Trim tokens to match the prompt length in frames.
    prompt_frames_target = max(1, min(prompt_frames_target, prompt_tokens.shape[-1]))
    prompt_tokens = prompt_tokens[:, :, :prompt_frames_target]
    prompt_tokens = _align_prompt_codebooks(model, prompt_tokens)
    return prompt_tokens


def _get_model(
    model_name: str,
    quantization_mode: Optional[str] = None,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> MusicGenContinuation:
    """
    Load + cache MLX MusicGen. If a finetune repo lacks config.json, we infer a base model.
    """
    resolved_quantization_mode = _normalize_quantization_mode(quantization_mode)

    with _MODEL_LOCK:
        # Return any cached instance for this model_name + quantization mode.
        for (name, _base, qmode), cached in _MODEL_CACHE.items():
            if name == model_name and qmode == resolved_quantization_mode:
                return cached

    # Load outside the lock if possible, but keep it simple for now: serialize loads.
    with _MODEL_LOCK:
        for (name, _base, qmode), cached in _MODEL_CACHE.items():
            if name == model_name and qmode == resolved_quantization_mode:
                return cached
        base_model = None
        prefer_base_config = False
        if has_explicit_base_model_override(model_name):
            # Some fine-tunes include config.json variants that don't decode reliably on MLX.
            # For explicit overrides we always use the known base config.
            base_model = get_base_model_for_finetune(model_name)
            prefer_base_config = True
            print(
                f"[INFO] Loading '{model_name}' with forced base config '{base_model}'."
            )
        def _load_model_once(
            selected_base_model: Optional[str],
            selected_prefer_base_config: bool,
        ) -> MusicGenContinuation:
            return MusicGenContinuation.from_pretrained(
                model_name,
                base_model=selected_base_model,
                prefer_base_config=selected_prefer_base_config,
                download_progress_callback=download_progress_callback,
            )

        def _load_with_base_fallback() -> MusicGenContinuation:
            nonlocal base_model
            try:
                return _load_model_once(base_model, prefer_base_config)
            except FileNotFoundError:
                base_model = get_base_model_for_finetune(model_name)
                return _load_model_once(base_model, True)

        try:
            model = _load_with_base_fallback()
        except Exception as load_error:
            if not _looks_like_checkpoint_load_error(load_error):
                raise

            print(
                f"[HF] checkpoint load failed for {model_name}; "
                "forcing checkpoint redownload and retrying once."
            )
            print(f"[HF] checkpoint load error: {load_error}")
            if not Path(model_name).exists():
                _predownload_model_checkpoint_assets(
                    model_name,
                    download_progress_callback=download_progress_callback,
                )
            model = _load_with_base_fallback()

        _apply_quantization_preset(model, resolved_quantization_mode)
        _MODEL_CACHE[(model_name, base_model, resolved_quantization_mode)] = model
        return model


def _snapshot_repo(
    repo_or_path: str,
    allow_patterns: list[str],
    *,
    local_files_only: bool,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> Path:
    path = Path(repo_or_path)
    if path.exists():
        return path

    from huggingface_hub import snapshot_download
    from huggingface_hub import constants as hf_constants

    global _HF_DOWNLOAD_RUNTIME_LOGGED
    if not _HF_DOWNLOAD_RUNTIME_LOGGED:
        _HF_DOWNLOAD_RUNTIME_LOGGED = True
        try:
            from huggingface_hub.utils._runtime import (
                is_package_available,
                is_xet_available,
            )

            xet_available = bool(is_xet_available())
            xet_package_available = bool(is_package_available("hf_xet"))
        except Exception:
            xet_available = False
            xet_package_available = False
        print(
            "[HF] download runtime:"
            f" xet_package_available={xet_package_available}"
            f" xet_available={xet_available}"
            f" xet_high_performance={bool(hf_constants.HF_XET_HIGH_PERFORMANCE)}"
            f" disable_xet={bool(hf_constants.HF_HUB_DISABLE_XET)}"
            f" xet_num_concurrent_range_gets={os.environ.get('HF_XET_NUM_CONCURRENT_RANGE_GETS', '<default>')}"
            f" xet_num_range_in_segment_base={os.environ.get('HF_XET_NUM_RANGE_IN_SEGMENT_BASE', '<default>')}"
            f" max_workers={_HF_SNAPSHOT_MAX_WORKERS}"
            f" HF_HOME={os.environ.get('HF_HOME', '')}"
            f" HUGGINGFACE_HUB_CACHE={os.environ.get('HUGGINGFACE_HUB_CACHE', '')}"
        )

    tqdm_class = None
    if (not local_files_only) and download_progress_callback is not None:
        from mlx_continuation.hf_progress import make_hf_tqdm_class

        tqdm_class = make_hf_tqdm_class(
            repo_id=repo_or_path,
            on_progress=download_progress_callback,
        )

    snapshot_path = snapshot_download(
        repo_id=repo_or_path,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
        max_workers=_HF_SNAPSHOT_MAX_WORKERS,
        tqdm_class=tqdm_class,
    )
    return Path(snapshot_path)


@contextmanager
def _override_hf_xet_disabled(disabled: bool):
    """
    Temporarily override HF Hub Xet mode in-process.

    huggingface_hub caches `HF_HUB_DISABLE_XET` in module constants at import
    time, so we patch both env + constants for the duration of a single call.
    """
    from huggingface_hub import constants as hf_constants

    with _HF_HUB_MODE_LOCK:
        previous_env = os.environ.get("HF_HUB_DISABLE_XET")
        previous_const = bool(hf_constants.HF_HUB_DISABLE_XET)
        new_value = bool(disabled)

        os.environ["HF_HUB_DISABLE_XET"] = "1" if new_value else "0"
        hf_constants.HF_HUB_DISABLE_XET = new_value
        try:
            yield
        finally:
            if previous_env is None:
                os.environ.pop("HF_HUB_DISABLE_XET", None)
            else:
                os.environ["HF_HUB_DISABLE_XET"] = previous_env
            hf_constants.HF_HUB_DISABLE_XET = previous_const


def _probe_xet_first_byte(repo_id: str, filename: str) -> bool:
    """
    Returns True if Xet reports first-byte progress quickly for this file.

    Probe runs in a short-lived subprocess so we can hard-timeout without
    risking a stuck in-process downloader.
    """
    timeout_s = _HF_XET_PROBE_TIMEOUT_SECONDS
    probe_code = r"""
import os
import sys
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import tqdm as hf_tqdm

repo_id = os.environ.get("G4L_XET_PROBE_REPO_ID", "")
filename = os.environ.get("G4L_XET_PROBE_FILENAME", "")

class _ProbeTqdm(hf_tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = False
        kwargs.setdefault("leave", False)
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        return None

    def update(self, n=1):
        out = super().update(n)
        if int(getattr(self, "n", 0) or 0) > 0:
            print("FIRST_BYTE", flush=True)
            os._exit(0)
        return out

hf_hub_download(repo_id=repo_id, filename=filename, force_download=True, tqdm_class=_ProbeTqdm)
sys.exit(3)
"""

    with tempfile.TemporaryDirectory(prefix="g4l_xet_probe_") as tmp_dir:
        probe_home = Path(tmp_dir) / "hf_home"
        probe_cache = Path(tmp_dir) / "hf_hub"
        probe_xet_cache = Path(tmp_dir) / "hf_xet"
        probe_home.mkdir(parents=True, exist_ok=True)
        probe_cache.mkdir(parents=True, exist_ok=True)
        probe_xet_cache.mkdir(parents=True, exist_ok=True)

        probe_env = os.environ.copy()
        # Preserve active HF_XET tuning values from this process, while
        # sandboxing probe caches to temporary directories.
        probe_env.update(
            {
                "HF_HUB_DISABLE_XET": "0",
                "HF_HOME": str(probe_home),
                "HUGGINGFACE_HUB_CACHE": str(probe_cache),
                "HF_XET_CACHE": str(probe_xet_cache),
                "G4L_XET_PROBE_REPO_ID": repo_id,
                "G4L_XET_PROBE_FILENAME": filename,
            }
        )

        try:
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                env=probe_env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            print(
                f"[HF] xet probe timeout ({timeout_s:.0f}s): "
                f"{repo_id}/{filename} -> fallback HTTP"
            )
            return False

        if completed.returncode == 0 and "FIRST_BYTE" in completed.stdout:
            return True
        print(
            f"[HF] xet probe failed (rc={completed.returncode}): "
            f"{repo_id}/{filename} -> fallback HTTP"
        )
        return False


def _should_use_xet(repo_id: str, filename: str) -> bool:
    global _HF_XET_GLOBAL_HEALTHY

    mode = _HF_XET_MODE
    if mode == "off":
        return False
    if mode == "on":
        return True

    # adaptive mode: only probe large checkpoint downloads where throughput
    # matters most.
    if filename != "state_dict.bin":
        return False

    with _HF_XET_PROBE_LOCK:
        if _HF_XET_GLOBAL_HEALTHY is True:
            return True

    try:
        # Do not use `is_xet_available()` here because it also checks the
        # runtime `HF_HUB_DISABLE_XET` flag, which we intentionally override per
        # request in `_download_repo_file`.
        from huggingface_hub.utils._runtime import is_package_available

        if not bool(is_package_available("hf_xet")):
            return False
    except Exception:
        return False

    probe_key = (repo_id, filename)
    with _HF_XET_PROBE_LOCK:
        cached = _HF_XET_PROBE_RESULTS.get(probe_key)
    if cached is not None:
        return cached

    result = _probe_xet_first_byte(repo_id, filename)
    with _HF_XET_PROBE_LOCK:
        _HF_XET_PROBE_RESULTS[probe_key] = result
        if result:
            _HF_XET_GLOBAL_HEALTHY = True
    return result


def _is_checkpoint_like_filename(filename: str) -> bool:
    lower = str(filename or "").strip().lower()
    return lower.endswith((".bin", ".ckpt", ".pt", ".pth", ".safetensors"))


def _looks_like_checkpoint_load_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    markers = (
        "pytorchstreamreader failed",
        "invalid header",
        "archive is corrupted",
        "filename 'storages' not found",
        "cannot use ``weights_only=true``",
        "legacy .tar format",
    )
    return any(marker in text for marker in markers)


def _download_repo_file(
    repo_id: str,
    filename: str,
    *,
    local_files_only: bool,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> Path:
    global _HF_XET_GLOBAL_HEALTHY

    from huggingface_hub import hf_hub_download

    prefer_runtime_downloader = _is_checkpoint_like_filename(filename)
    force_download = (not local_files_only) and prefer_runtime_downloader

    if not local_files_only and not prefer_runtime_downloader:
        def _shared_progress(evt: dict) -> None:
            if download_progress_callback is None:
                return
            payload = {
                "repo_id": repo_id,
                "filename": filename,
                "downloaded_bytes": int(evt.get("downloaded_bytes") or 0),
                "total_bytes": int(evt.get("total_bytes") or 0),
                "percent": int(evt.get("percent") or 0),
                "speed_bps": float(evt.get("speed_bps") or 0.0),
            }
            try:
                download_progress_callback(payload)
            except Exception:
                return

        try:
            selected_backend_label = _download_with_shared_hf_downloader_env(
                repo_id=repo_id,
                filename=filename,
                on_progress=_shared_progress,
            )
            print(
                f"[HF] shared downloader selected for {repo_id}/{filename}: "
                f"{selected_backend_label}"
            )
            local_file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=True,
            )
            return Path(local_file_path)
        except Exception as shared_download_error:
            print(
                f"[HF] shared downloader failed for {repo_id}/{filename}; "
                f"falling back to runtime hub path: {shared_download_error}"
            )
    elif force_download:
        print(
            f"[HF] runtime checkpoint path forced for {repo_id}/{filename} "
            "(reliability over shared worker)"
        )

    use_xet = (not local_files_only) and _should_use_xet(repo_id, filename)

    tqdm_class = None
    if (not local_files_only) and download_progress_callback is not None:
        from mlx_continuation.hf_progress import make_hf_tqdm_class

        tqdm_class = make_hf_tqdm_class(
            repo_id=repo_id,
            on_progress=download_progress_callback,
        )

    disable_xet = not use_xet
    if not local_files_only:
        print(
            f"[HF] transport decision: {repo_id}/{filename} -> "
            f"{'xet' if use_xet else 'http'} "
            f"(mode={_HF_XET_MODE}, probe_timeout={_HF_XET_PROBE_TIMEOUT_SECONDS:.0f}s)"
        )
    try:
        with _override_hf_xet_disabled(disable_xet):
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=local_files_only,
                force_download=force_download,
                tqdm_class=tqdm_class,
            )
        if use_xet:
            with _HF_XET_PROBE_LOCK:
                _HF_XET_GLOBAL_HEALTHY = True
                _HF_XET_PROBE_RESULTS[(repo_id, filename)] = True
    except Exception:
        if local_files_only or disable_xet:
            raise
        with _HF_XET_PROBE_LOCK:
            _HF_XET_PROBE_RESULTS[(repo_id, filename)] = False
            _HF_XET_GLOBAL_HEALTHY = False
        print(
            f"[HF] xet download failed for {repo_id}/{filename}; "
            "retrying via HTTP"
        )
        with _override_hf_xet_disabled(True):
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_files_only=local_files_only,
                force_download=force_download,
                tqdm_class=tqdm_class,
            )
    return Path(file_path)


def _predownload_model_checkpoint_assets(
    model_name: str,
    *,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> Path:
    """
    Prefer direct file downloads for model checkpoints to avoid slow repo-tree
    listing paths in `snapshot_download` on large/noisy repos.
    """
    from huggingface_hub.errors import (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
    )

    missing_entry_errors = (
        EntryNotFoundError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
    )

    required_files = ["state_dict.bin"]
    optional_files = [
        "compression_state_dict.bin",
        "config.json",
    ]

    for filename in required_files:
        try:
            _download_repo_file(
                model_name,
                filename,
                local_files_only=False,
                download_progress_callback=download_progress_callback,
            )
        except missing_entry_errors:
            # Preserve compatibility with repos that don't follow this exact
            # layout by falling back to the previous snapshot path.
            _emit_download_event(
                download_progress_callback,
                {
                    "model_name": model_name,
                    "phase": "download",
                    "desc": "Falling back to snapshot download layout",
                    "repo_id": model_name,
                    "unit": "B",
                    "downloaded_bytes": 0,
                    "total_bytes": 0,
                    "percent": 0,
                    "done": False,
                },
            )
            return _snapshot_repo(
                model_name,
                _MODEL_SNAPSHOT_PATTERNS,
                local_files_only=False,
                download_progress_callback=download_progress_callback,
            )

    for filename in optional_files:
        try:
            _download_repo_file(
                model_name,
                filename,
                local_files_only=False,
                download_progress_callback=download_progress_callback,
            )
        except missing_entry_errors:
            continue

    # Resolve local snapshot folder for downstream config/component inference.
    return _snapshot_repo(
        model_name,
        _MODEL_SNAPSHOT_PATTERNS,
        local_files_only=True,
        download_progress_callback=None,
    )


def _emit_download_event(
    callback: Optional[Callable[[dict], None]],
    payload: dict[str, Any],
) -> None:
    if callback is None:
        return
    try:
        callback(payload)
    except Exception:
        # Download status callbacks are best-effort only.
        return


def _make_stage_progress_callback(
    *,
    model_name: str,
    stage_index: int,
    stage_total: int,
    stage_name: str,
    on_progress: Optional[Callable[[dict], None]],
) -> Callable[[dict], None]:
    last_stage_percent = 0
    last_downloaded_bytes = 0
    heuristic_bytes_accumulator = 0
    last_speed_t: Optional[float] = None
    last_speed_downloaded = 0

    def _stage_progress(evt: dict) -> None:
        nonlocal last_stage_percent
        nonlocal last_downloaded_bytes
        nonlocal heuristic_bytes_accumulator
        nonlocal last_speed_t
        nonlocal last_speed_downloaded

        downloaded = int(evt.get("downloaded_bytes") or 0)
        total = int(evt.get("total_bytes") or 0)
        reported_percent = int(evt.get("percent") or 0)
        done = bool(evt.get("done") or False)
        unit = str(evt.get("unit") or "").strip().lower()
        is_item_counter = unit in {"it", "item", "items", "file", "files"}
        is_byte_counter = not is_item_counter
        byte_delta = 0

        if is_byte_counter:
            if downloaded >= last_downloaded_bytes:
                byte_delta = downloaded - last_downloaded_bytes
            elif downloaded > 0:
                # `tqdm` counters can reset between files inside one snapshot.
                byte_delta = downloaded
            if downloaded >= 0:
                last_downloaded_bytes = downloaded

        speed_bps = 0.0
        if is_byte_counter and downloaded >= 0:
            now = time.monotonic()
            if last_speed_t is not None and now > last_speed_t and downloaded >= last_speed_downloaded:
                speed_bps = (downloaded - last_speed_downloaded) / (now - last_speed_t)
            last_speed_t = now
            last_speed_downloaded = downloaded

        reliable_total_bytes = 8 * 1024 * 1024

        if done:
            stage_percent = 100
        elif (
            is_byte_counter
            and total >= reliable_total_bytes
            and downloaded <= total
        ):
            # Treat this as authoritative only for byte counters. Item counters
            # from snapshot orchestration can misrepresent large file progress.
            stage_percent = max(0, min(100, reported_percent))
        elif is_byte_counter and byte_delta > 0:
            stage_percent = max(1, last_stage_percent)
            heuristic_bytes_accumulator += byte_delta
            heuristic_step = 32 * 1024 * 1024
            while heuristic_bytes_accumulator >= heuristic_step and stage_percent < 90:
                stage_percent += 1
                heuristic_bytes_accumulator -= heuristic_step
        else:
            stage_percent = last_stage_percent

        if done:
            stage_percent = 100
        else:
            stage_percent = min(99, max(last_stage_percent, stage_percent))

        last_stage_percent = stage_percent

        overall_raw = (
            ((stage_index - 1) + (stage_percent / 100.0)) / max(stage_total, 1)
        ) * 100.0
        if done:
            overall_percent = 100
        else:
            capped_raw = max(0.0, min(99.0, overall_raw))
            if stage_percent > 0 or byte_delta > 0 or downloaded > 0:
                # Avoid long-lived 0% when total bytes are not reported.
                overall_percent = max(1, int(capped_raw + 0.9999))
            else:
                overall_percent = int(capped_raw)
        payload = {
            "model_name": model_name,
            "phase": "download",
            "stage_index": stage_index,
            "stage_total": stage_total,
            "stage_name": stage_name,
            "stage_percent": stage_percent,
            "percent": max(0, min(100, overall_percent)),
            "repo_id": str(evt.get("repo_id") or ""),
            "downloaded_bytes": downloaded,
            "total_bytes": total,
            "speed_bps": speed_bps,
            "desc": str(evt.get("desc") or ""),
            "done": done,
            "unit": str(evt.get("unit") or ""),
            "progress_name": str(evt.get("progress_name") or ""),
        }
        _emit_download_event(on_progress, payload)

    return _stage_progress


def _infer_model_component_repos(config_path: Path) -> tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    text_encoder = config_data.get("text_encoder", {}) or {}
    audio_encoder = config_data.get("audio_encoder", {}) or {}

    text_repo = str(text_encoder.get("_name_or_path") or "").strip()
    audio_repo_raw = str(audio_encoder.get("_name_or_path") or "").strip()
    if not text_repo:
        raise AudioProcessingError(
            f"Missing text_encoder._name_or_path in config: {config_path}"
        )
    if not audio_repo_raw:
        raise AudioProcessingError(
            f"Missing audio_encoder._name_or_path in config: {config_path}"
        )

    encodec_name = audio_repo_raw.split("/")[-1].replace("_", "-")
    encodec_repo = f"mlx-community/{encodec_name}-float32"
    return text_repo, encodec_repo


def _resolve_model_config_path(
    model_name: str,
    *,
    local_files_only: bool,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> tuple[Path, str]:
    base_model = get_base_model_for_finetune(model_name)
    prefer_base_config = has_explicit_base_model_override(model_name)

    model_path = _snapshot_repo(
        model_name,
        _MODEL_SNAPSHOT_PATTERNS,
        local_files_only=local_files_only,
        download_progress_callback=download_progress_callback,
    )

    model_config_path = model_path / "config.json"
    if (not prefer_base_config) and model_config_path.exists():
        return model_config_path, base_model

    base_path = _snapshot_repo(
        base_model,
        _CONFIG_SNAPSHOT_PATTERNS,
        local_files_only=local_files_only,
        download_progress_callback=download_progress_callback,
    )
    base_config_path = base_path / "config.json"
    if not base_config_path.exists():
        raise AudioProcessingError(
            f"Missing config.json for base model '{base_model}'."
        )
    return base_config_path, base_model


def predownload_model(
    model_name: str,
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> dict[str, Any]:
    """
    Download all assets needed for offline generation for a given finetune:
    - finetune checkpoint/config snapshot
    - base config snapshot when required
    - text encoder weights
    - EnCodec decoder weights
    - tokenizer assets used by runtime tokenizer ("t5-base")
    """
    base_model = get_base_model_for_finetune(model_name)
    prefer_base_config = has_explicit_base_model_override(model_name)

    # Step 1: model checkpoint snapshot (always required).
    step_count = 5

    step_index = 1
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Model checkpoint",
            "stage_percent": 0,
            "percent": 0,
            "repo_id": model_name,
            "desc": "Starting model checkpoint download",
            "done": False,
        },
    )
    model_snapshot_path = _predownload_model_checkpoint_assets(
        model_name,
        download_progress_callback=_make_stage_progress_callback(
            model_name=model_name,
            stage_index=step_index,
            stage_total=step_count,
            stage_name="Model checkpoint",
            on_progress=download_progress_callback,
        ),
    )
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Model checkpoint",
            "stage_percent": 100,
            "percent": int((step_index / step_count) * 100),
            "repo_id": model_name,
            "desc": "Model checkpoint ready",
            "done": True,
        },
    )

    step_index += 1

    # Step 2 (optional): base config stage.
    needs_base_config = prefer_base_config or not (model_snapshot_path / "config.json").exists()
    if needs_base_config:
        _emit_download_event(
            download_progress_callback,
            {
                "model_name": model_name,
                "phase": "download",
                "stage_index": step_index,
                "stage_total": step_count,
                "stage_name": "Base model config",
                "stage_percent": 0,
                "percent": int(((step_index - 1) / step_count) * 100),
                "repo_id": base_model,
                "desc": "Starting base config download",
                "done": False,
            },
        )
        _snapshot_repo(
            base_model,
            _CONFIG_SNAPSHOT_PATTERNS,
            local_files_only=False,
            download_progress_callback=_make_stage_progress_callback(
                model_name=model_name,
                stage_index=step_index,
                stage_total=step_count,
                stage_name="Base model config",
                on_progress=download_progress_callback,
            ),
        )
        _emit_download_event(
            download_progress_callback,
            {
                "model_name": model_name,
                "phase": "download",
                "stage_index": step_index,
                "stage_total": step_count,
                "stage_name": "Base model config",
                "stage_percent": 100,
                "percent": int((step_index / step_count) * 100),
                "repo_id": base_model,
                "desc": "Base model config ready",
                "done": True,
            },
        )
    else:
        _emit_download_event(
            download_progress_callback,
            {
                "model_name": model_name,
                "phase": "download",
                "stage_index": step_index,
                "stage_total": step_count,
                "stage_name": "Base model config",
                "stage_percent": 100,
                "percent": int((step_index / step_count) * 100),
                "repo_id": base_model,
                "desc": "Base model config not required",
                "done": True,
            },
        )
    step_index += 1

    # Resolve config path (downloads base config when finetune config is absent).
    config_path, _ = _resolve_model_config_path(
        model_name,
        local_files_only=True,
        download_progress_callback=None,
    )
    text_repo, encodec_repo = _infer_model_component_repos(config_path)

    # Step: text encoder.
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Text encoder",
            "stage_percent": 0,
            "percent": int(((step_index - 1) / step_count) * 100),
            "repo_id": text_repo,
            "desc": "Starting text encoder download",
            "done": False,
        },
    )
    _snapshot_repo(
        text_repo,
        _COMPONENT_SNAPSHOT_PATTERNS,
        local_files_only=False,
        download_progress_callback=_make_stage_progress_callback(
            model_name=model_name,
            stage_index=step_index,
            stage_total=step_count,
            stage_name="Text encoder",
            on_progress=download_progress_callback,
        ),
    )
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Text encoder",
            "stage_percent": 100,
            "percent": int((step_index / step_count) * 100),
            "repo_id": text_repo,
            "desc": "Text encoder ready",
            "done": True,
        },
    )
    step_index += 1

    # Step: EnCodec audio decoder.
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Audio decoder",
            "stage_percent": 0,
            "percent": int(((step_index - 1) / step_count) * 100),
            "repo_id": encodec_repo,
            "desc": "Starting audio decoder download",
            "done": False,
        },
    )
    _snapshot_repo(
        encodec_repo,
        _COMPONENT_SNAPSHOT_PATTERNS,
        local_files_only=False,
        download_progress_callback=_make_stage_progress_callback(
            model_name=model_name,
            stage_index=step_index,
            stage_total=step_count,
            stage_name="Audio decoder",
            on_progress=download_progress_callback,
        ),
    )
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Audio decoder",
            "stage_percent": 100,
            "percent": int((step_index / step_count) * 100),
            "repo_id": encodec_repo,
            "desc": "Audio decoder ready",
            "done": True,
        },
    )
    step_index += 1

    # Step: tokenizer assets used by Tokenizer(..., "t5-base").
    tokenizer_repo = "t5-base"
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Tokenizer",
            "stage_percent": 0,
            "percent": int(((step_index - 1) / step_count) * 100),
            "repo_id": tokenizer_repo,
            "desc": "Starting tokenizer download",
            "done": False,
        },
    )
    _snapshot_repo(
        tokenizer_repo,
        _TOKENIZER_SNAPSHOT_PATTERNS,
        local_files_only=False,
        download_progress_callback=_make_stage_progress_callback(
            model_name=model_name,
            stage_index=step_index,
            stage_total=step_count,
            stage_name="Tokenizer",
            on_progress=download_progress_callback,
        ),
    )
    _emit_download_event(
        download_progress_callback,
        {
            "model_name": model_name,
            "phase": "download",
            "stage_index": step_index,
            "stage_total": step_count,
            "stage_name": "Tokenizer",
            "stage_percent": 100,
            "percent": 100,
            "repo_id": tokenizer_repo,
            "desc": "Tokenizer ready",
            "done": True,
        },
    )

    return {
        "success": True,
        "model_name": model_name,
        "base_model": base_model,
        "text_repo": text_repo,
        "encodec_repo": encodec_repo,
        "tokenizer_repo": tokenizer_repo,
    }


def get_model_download_status(model_name: str) -> dict[str, Any]:
    """
    Check whether a model can run fully offline by verifying required snapshots
    are already present in local HF cache.
    """
    missing: list[str] = []

    def _require_snapshot(
        repo_or_path: str,
        allow_patterns: list[str],
        label: str,
    ) -> Optional[Path]:
        try:
            return _snapshot_repo(
                repo_or_path,
                allow_patterns,
                local_files_only=True,
                download_progress_callback=None,
            )
        except Exception:
            missing.append(label)
            return None

    model_path = _require_snapshot(
        model_name,
        _MODEL_SNAPSHOT_PATTERNS,
        f"{model_name} (model checkpoint)",
    )

    base_model = None
    config_path: Optional[Path] = None
    if model_path is not None:
        try:
            base_model = get_base_model_for_finetune(model_name)
            prefer_base_config = has_explicit_base_model_override(model_name)
            model_config_path = model_path / "config.json"
            if (not prefer_base_config) and model_config_path.exists():
                config_path = model_config_path
            else:
                base_path = _require_snapshot(
                    base_model,
                    _CONFIG_SNAPSHOT_PATTERNS,
                    f"{base_model} (base config)",
                )
                if base_path is not None:
                    candidate = base_path / "config.json"
                    if candidate.exists():
                        config_path = candidate
                    else:
                        missing.append(f"{base_model} (base config)")
        except Exception as e:
            missing.append(f"{model_name} (config resolution: {e})")

    text_repo = None
    encodec_repo = None
    if config_path is not None:
        try:
            text_repo, encodec_repo = _infer_model_component_repos(config_path)
        except Exception as e:
            missing.append(f"{model_name} (config parse: {e})")

    if text_repo:
        _require_snapshot(
            text_repo,
            _COMPONENT_SNAPSHOT_PATTERNS,
            f"{text_repo} (text encoder)",
        )
    if encodec_repo:
        _require_snapshot(
            encodec_repo,
            _COMPONENT_SNAPSHOT_PATTERNS,
            f"{encodec_repo} (audio decoder)",
        )
    _require_snapshot(
        "t5-base",
        _TOKENIZER_SNAPSHOT_PATTERNS,
        "t5-base (tokenizer)",
    )

    unique_missing = sorted(set(missing))
    return {
        "downloaded": len(unique_missing) == 0,
        "missing": unique_missing,
        "base_model": base_model,
        "text_repo": text_repo,
        "encodec_repo": encodec_repo,
        "tokenizer_repo": "t5-base",
    }


def _generate_with_prompt(
    model: MusicGenContinuation,
    prompt_audio: np.ndarray,
    prompt_sr: int,
    prompt_duration_s: float,
    top_k: int,
    temperature: float,
    cfg_coef: float,
    text: str,
    progress_callback: Optional[Callable[[int, int], None]],
) -> np.ndarray:
    """
    Returns full generated audio (prompt reconstruction + continuation) at model.sampling_rate.
    """
    prompt_tokens = _encode_prompt_tokens(model, prompt_audio, prompt_sr)
    prompt_frames = int(prompt_tokens.shape[-1])
    total_frames = int(round(OUTPUT_DURATION_S * model.frame_rate))
    if prompt_frames >= total_frames:
        raise AudioProcessingError(
            f"prompt_duration ({prompt_duration_s}s) must be < output duration ({OUTPUT_DURATION_S}s)."
        )
    max_new_steps = max(1, total_frames - prompt_frames)

    # Serialize generation for stability.
    with _GEN_LOCK:
        full_audio, _ = model.generate_continuation(
            prompt_tokens=prompt_tokens,
            max_new_steps=max_new_steps,
            text=text,
            top_k=top_k,
            temp=temperature,
            guidance_coef=cfg_coef,
            progress=False,
            progress_callback=progress_callback,
            return_tokens=False,
        )
        mx.eval(full_audio)

    full_np = np.array(full_audio[0]).astype(np.float32)
    return full_np


def process_audio(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None,
    quantization_mode: Optional[str] = None,
    device_id: int = 0,  # unused (kept for API compatibility)
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    MLX-backed equivalent of g4laudio.process_audio:
    - uses FIRST `prompt_duration` seconds as prompt
    - returns a 30s generation (prompt reconstruction + continuation)
    """
    try:
        audio, sr = _load_audio_base64(input_data_base64)
        out_channels = audio.shape[1]

        prompt_samples = max(1, int(round(prompt_duration * sr)))
        if audio.shape[0] < prompt_samples:
            pad = prompt_samples - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant").astype(np.float32)

        prompt_audio = audio[:prompt_samples]

        model = _get_model(
            model_name,
            quantization_mode=quantization_mode,
            download_progress_callback=download_progress_callback,
        )

        final_text = get_model_description(model_name, description) or ""
        full_np = _generate_with_prompt(
            model=model,
            prompt_audio=prompt_audio,
            prompt_sr=sr,
            prompt_duration_s=float(prompt_duration),
            top_k=int(top_k),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
            text=final_text,
            progress_callback=progress_callback,
        )

        # Resample back to input sample rate and match channels.
        if model.sampling_rate != sr:
            full_np = _resample(full_np, model.sampling_rate, sr)
        full_np = _convert_channels(full_np, out_channels)

        return _save_audio_base64(full_np, sr)
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Audio processing failed: {e}") from e


def continue_music(
    input_data_base64: str,
    model_name: str,
    progress_callback: Optional[Callable] = None,
    prompt_duration: int = 6,
    top_k: int = 250,
    temperature: float = 1.0,
    cfg_coef: float = 3.0,
    description: Optional[str] = None,
    quantization_mode: Optional[str] = None,
    device_id: int = 0,  # unused (kept for API compatibility)
    download_progress_callback: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    MLX-backed equivalent of g4laudio.continue_music:
    - uses LAST `prompt_duration` seconds as prompt
    - returns (original_without_prompt) + (30s generation which includes prompt reconstruction)
    """
    try:
        audio, sr = _load_audio_base64(input_data_base64)
        out_channels = audio.shape[1]

        prompt_samples = max(1, int(round(prompt_duration * sr)))
        if audio.shape[0] < prompt_samples:
            pad = prompt_samples - audio.shape[0]
            audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant").astype(np.float32)

        original_minus_prompt = audio[:-prompt_samples]
        prompt_audio = audio[-prompt_samples:]

        model = _get_model(
            model_name,
            quantization_mode=quantization_mode,
            download_progress_callback=download_progress_callback,
        )

        final_text = get_model_description(model_name, description) or ""
        full_np = _generate_with_prompt(
            model=model,
            prompt_audio=prompt_audio,
            prompt_sr=sr,
            prompt_duration_s=float(prompt_duration),
            top_k=int(top_k),
            temperature=float(temperature),
            cfg_coef=float(cfg_coef),
            text=final_text,
            progress_callback=progress_callback,
        )

        # Resample back to input sample rate and match channels.
        if model.sampling_rate != sr:
            full_np = _resample(full_np, model.sampling_rate, sr)
        full_np = _convert_channels(full_np, out_channels)

        combined = (
            np.concatenate([original_minus_prompt, full_np], axis=0)
            if original_minus_prompt.shape[0] > 0
            else full_np
        )

        return _save_audio_base64(combined, sr)
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Music continuation failed: {e}") from e
