#!/usr/bin/env python3
"""
Stable Audio 3 localhost API.

MLX-backed gary4local service that keeps the Windows HTTP contract while using
runtime torch-to-MLX conversion for Apple Silicon inference, including the
experimental pingpong-based latent-prefix continuation path used by the JUCE
companion.
"""

from __future__ import annotations

import base64
import gc
import io
import json
import math
import numpy as np
import os
import random
import re
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable

import soundfile as sf
import torch
import torchaudio
from einops import rearrange
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    from huggingface_hub import hf_hub_download, login, snapshot_download, try_to_load_from_cache
except Exception:  # pragma: no cover - import diagnostics are surfaced at load time
    hf_hub_download = None
    login = None
    snapshot_download = None
    try_to_load_from_cache = None

from stable_audio_3.mlx.pipeline import StableAudioMLXPipeline, adapt_sample_size
from stable_audio_3.mlx.runtime import (
    MLXRuntimeUnavailableError,
    import_mlx_core,
    mlx_runtime_available,
)


os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

app = Flask(__name__)
CORS(app)


MODEL_NAME = os.environ.get("SA3_MODEL", "medium")
DEFAULT_STEPS = int(os.environ.get("SA3_DEFAULT_STEPS", "8"))
DEFAULT_CFG = float(os.environ.get("SA3_DEFAULT_CFG", "1.0"))
DEFAULT_NEGATIVE = os.environ.get("SA3_DEFAULT_NEGATIVE", "low quality")
DEFAULT_DURATION = float(os.environ.get("SA3_DEFAULT_DURATION", "30"))
MAX_DURATION = float(os.environ.get("SA3_MAX_DURATION", "300"))
DEFAULT_SAMPLER = os.environ.get("SA3_DEFAULT_SAMPLER", "pingpong")
DEFAULT_LOOP_BARS = int(os.environ.get("SA3_DEFAULT_LOOP_BARS", "8"))
LOOP_PAD_SECONDS = float(os.environ.get("SA3_LOOP_PAD_SECONDS", "2.0"))
DEFAULT_CONTINUATION_SECONDS = float(os.environ.get("SA3_DEFAULT_CONTINUATION_SECONDS", "8.0"))
CONTINUE_TAIL_MODE = os.environ.get("SA3_CONTINUE_TAIL_MODE", "regen_past").lower()
CONTINUE_TAIL_PAD = float(os.environ.get("SA3_CONTINUE_TAIL_PAD", "6.0"))
CONTINUE_TAIL_PAD_MAX = float(os.environ.get("SA3_CONTINUE_TAIL_PAD_MAX", "60.0"))
OUTPUT_SAMPLE_RATE = int(os.environ.get("SA3_SAMPLE_RATE", "44100"))
GENERATION_PROGRESS_LOADING = 2
GENERATION_PROGRESS_PREPARING = 8
GENERATION_PROGRESS_CONDITIONING = 10
GENERATION_SAMPLING_PROGRESS_START = 12
GENERATION_SAMPLING_PROGRESS_END = 80
GENERATION_DECODE_PREP_PROGRESS = 84
GENERATION_DECODING_PROGRESS = 91
GENERATION_POSTPROCESS_PROGRESS = 95
GENERATION_ENCODING_PROGRESS = 98

OUTPUT_DIR = os.environ.get("OUTPUT_DIR") or os.path.join(os.getcwd(), "outputs")
PROMPTS_DIR = os.environ.get("SA3_PROMPTS_DIR") or os.path.join(os.getcwd(), "prompts")
LORA_DIR = os.environ.get("SA3_LORA_DIR") or os.path.join(os.getcwd(), "loras")
BUNDLED_DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent / "prompts" / "defaults.json"
LORA_REGISTRY_PATH = os.environ.get("SA3_LORA_REGISTRY") or os.path.join(
    Path(PROMPTS_DIR).resolve().parent, "lora_registry.json"
)
DEFAULT_LORA_NAME = os.environ.get("SA3_DEFAULT_LORA", "").strip()
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

VALID_SHIFTS = {"default", "none", "logsnr", "flux", "full"}
VALID_LOOP_BARS = {4, 8, 16, 32}
VALID_CONTINUATION_MODES = {"inpaint", "latent_prefix"}
LORA_EXTS = (".ckpt", ".safetensors")
_BPM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*bpm", re.IGNORECASE)


def env_optional_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw or raw.lower() in {"off", "none", "disable", "disabled"}:
        return None
    return float(raw)


LATENT_DIAG = os.environ.get("SA3_LATENT_DIAG", "0") != "0"
LATENT_RESCALE = float(os.environ.get("SA3_LATENT_RESCALE", "1.0"))
LATENT_SHIFT = float(os.environ.get("SA3_LATENT_SHIFT", "0.0"))
LATENT_TARGET_STD = env_optional_float("SA3_LATENT_TARGET_STD")
LATENT_ADAPT_MIN = float(os.environ.get("SA3_LATENT_ADAPT_MIN", "0.9"))
LATENT_ADAPT_MAX = float(os.environ.get("SA3_LATENT_ADAPT_MAX", "1.0"))
PEAK_NORM_DB = env_optional_float("SA3_PEAK_NORMALIZE_DB", 2.0)
LIMITER_CEILING_DB = env_optional_float("SA3_LIMITER_CEILING_DB", -0.3)
if LIMITER_CEILING_DB is not None and LIMITER_CEILING_DB > 0.0:
    LIMITER_CEILING_DB = None
LIMITER_KNEE = float(os.environ.get("SA3_LIMITER_KNEE", "0.8"))

# A ceiling, not a fixed allocation: the pipeline adapts this down to the
# requested duration. It prevents the upstream 120s default cap from clipping
# legitimate longer local requests.
MAX_SAMPLE_SIZE = int((MAX_DURATION + CONTINUE_TAIL_PAD_MAX + 40.0) * OUTPUT_SAMPLE_RATE)

SA3_MODEL_LINKS = {
    "stable-audio-3-medium": "https://huggingface.co/stabilityai/stable-audio-3-medium",
    "t5gemma-b-b-ul2": "https://huggingface.co/stabilityai/stable-audio-3-medium/tree/main/t5gemma-b-b-ul2",
}

VALID_MLX_DTYPES = {"float16", "float32", "bfloat16"}
VALID_MLX_ATTENTION = {"sliding", "full"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def env_mlx_dtype(name: str, default: str) -> str:
    raw = (os.environ.get(name) or default).strip().lower()
    if raw not in VALID_MLX_DTYPES:
        raise ValueError(
            f"{name} must be one of {sorted(VALID_MLX_DTYPES)}, got {raw!r}"
        )
    return raw


def env_mlx_attention(name: str, default: str = "sliding") -> str:
    raw = (os.environ.get(name) or default).strip().lower()
    if raw not in VALID_MLX_ATTENTION:
        raise ValueError(
            f"{name} must be one of {sorted(VALID_MLX_ATTENTION)}, got {raw!r}"
        )
    return raw


MLX_DTYPE = env_mlx_dtype("SA3_MLX_DTYPE", "float16")
MLX_DIT_DTYPE = env_mlx_dtype(
    "SA3_MLX_DIT_DTYPE",
    os.environ.get("SA3_MLX_DTYPE", MLX_DTYPE),
)
MLX_TEXT_DTYPE = env_mlx_dtype(
    "SA3_MLX_TEXT_DTYPE",
    os.environ.get("SA3_MLX_DTYPE", MLX_DTYPE),
)
MLX_NUMBER_DTYPE = env_mlx_dtype(
    "SA3_MLX_NUMBER_DTYPE",
    os.environ.get("SA3_MLX_DTYPE", MLX_DTYPE),
)
MLX_AUTOENCODER_DTYPE = env_mlx_dtype(
    "SA3_MLX_AUTOENCODER_DTYPE",
    os.environ.get("SA3_MLX_DTYPE", MLX_DTYPE),
)
MLX_ATTENTION = env_mlx_attention("SA3_MLX_ATTENTION", "sliding")
# Writing the MLX conversion cache costs ~5 GB of disk to avoid ~20 s of torch
# conversion on every service start. Reads always happen when an entry exists;
# this only controls whether a miss writes one.
SA3_MLX_CACHE_ENABLED = env_bool("SA3_MLX_CACHE", True)
TORCH_DEVICE = (os.environ.get("SA3_TORCH_DEVICE") or "auto").strip().lower() or "auto"
TORCH_MODEL_HALF = env_bool(
    "SA3_TORCH_MODEL_HALF",
    env_bool("SA3_MODEL_HALF", False),
)
CHUNKED_DECODE = env_bool("SA3_CHUNKED_DECODE", False)
DECODE_CHUNK_SIZE = int(os.environ.get("SA3_DECODE_CHUNK_SIZE", "128"))
DECODE_OVERLAP = int(os.environ.get("SA3_DECODE_OVERLAP", "32"))
DECODE_CHUNK_BATCH_SIZE = int(os.environ.get("SA3_DECODE_CHUNK_BATCH_SIZE", "1"))
if DECODE_CHUNK_SIZE < 1:
    raise ValueError(f"SA3_DECODE_CHUNK_SIZE must be positive, got {DECODE_CHUNK_SIZE}")
if DECODE_OVERLAP < 0 or DECODE_OVERLAP >= DECODE_CHUNK_SIZE:
    raise ValueError(
        "SA3_DECODE_OVERLAP must be >= 0 and smaller than SA3_DECODE_CHUNK_SIZE, "
        f"got overlap={DECODE_OVERLAP} size={DECODE_CHUNK_SIZE}"
    )
if DECODE_CHUNK_BATCH_SIZE < 1:
    raise ValueError(
        f"SA3_DECODE_CHUNK_BATCH_SIZE must be positive, got {DECODE_CHUNK_BATCH_SIZE}"
    )


sessions: dict[str, dict[str, Any]] = {}
sessions_lock = threading.Lock()
model_lock = threading.Lock()
generation_lock = threading.Lock()
predownload_sessions: dict[str, dict[str, Any]] = {}
predownload_sessions_lock = threading.Lock()

pipe: StableAudioMLXPipeline | None = None
model_loaded = False
model_loading = False
model_error: str | None = None
last_load_seconds = 0.0
model_sample_rate = OUTPUT_SAMPLE_RATE
model_device: str | None = None
lora_registry: list[tuple[str, str]] = []
lora_name_to_index: dict[str, int] = {}
PREDOWNLOAD_TTL_SECONDS = 3600

SA3_REQUIRED_MODEL_FILES: dict[str, dict[str, Any]] = {
    "small-music": {
        "repo_id": "stabilityai/stable-audio-3-small-music",
        "label": "stable-audio-3-small-music",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
    "small-music-base": {
        "repo_id": "stabilityai/stable-audio-3-small-music-base",
        "label": "stable-audio-3-small-music-base",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
    "small-sfx": {
        "repo_id": "stabilityai/stable-audio-3-small-sfx",
        "label": "stable-audio-3-small-sfx",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
    "small-sfx-base": {
        "repo_id": "stabilityai/stable-audio-3-small-sfx-base",
        "label": "stable-audio-3-small-sfx-base",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
    "medium": {
        "repo_id": "stabilityai/stable-audio-3-medium",
        "label": "stable-audio-3-medium",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
    "medium-base": {
        "repo_id": "stabilityai/stable-audio-3-medium-base",
        "label": "stable-audio-3-medium-base",
        "files": [
            "model_config.json",
            "model.safetensors",
        ],
    },
}

SA3_TEXT_ENCODER_SUBFOLDER = "t5gemma-b-b-ul2"
SA3_TEXT_ENCODER_MARKER_NAMES = [
    "config.json",
    "model.safetensors",
    "tokenizer.model",
    "tokenizer_config.json",
]


def current_sa3_model_target() -> dict[str, Any] | None:
    return SA3_REQUIRED_MODEL_FILES.get(MODEL_NAME)


def current_sa3_text_encoder_target() -> dict[str, Any]:
    model_target = current_sa3_model_target()
    repo_id = (
        str(model_target["repo_id"])
        if model_target is not None
        else "stabilityai/stable-audio-3-medium"
    )
    return {
        "repo_id": repo_id,
        "label": SA3_TEXT_ENCODER_SUBFOLDER,
        "display_label": f"{repo_id}/{SA3_TEXT_ENCODER_SUBFOLDER}",
        "subfolder": SA3_TEXT_ENCODER_SUBFOLDER,
    }


def current_sa3_text_encoder_markers() -> list[str]:
    subfolder = SA3_TEXT_ENCODER_SUBFOLDER
    return [f"{subfolder}/{name}" for name in SA3_TEXT_ENCODER_MARKER_NAMES]


def cuda_mem_mb() -> dict[str, float] | None:
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 1),
        "free_mb": round(free / 1048576, 1),
        "total_mb": round(total / 1048576, 1),
    }


def cleanup_cuda() -> None:
    for _ in range(2):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def hf_token_configured() -> bool:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return bool(token and token.strip())


def configure_hf_auth() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    if login is None:
        return
    try:
        login(token=token, add_to_git_credential=False)
    except TypeError:
        login(token=token)


def normalize_lora_name(raw: str) -> str:
    return raw.strip().lower()


def lora_path_is_valid(path: str) -> bool:
    return path.lower().endswith(LORA_EXTS) and os.path.isfile(path)


def scan_lora_dir() -> list[tuple[str, str]]:
    if not os.path.isdir(LORA_DIR):
        return []
    entries = []
    for filename in sorted(os.listdir(LORA_DIR)):
        if not filename.lower().endswith(LORA_EXTS):
            continue
        path = os.path.join(LORA_DIR, filename)
        if os.path.isfile(path):
            entries.append((os.path.splitext(filename)[0].lower(), path))
    return entries


def read_lora_registry_file() -> list[tuple[str, str]]:
    if not os.path.isfile(LORA_REGISTRY_PATH):
        return []
    with open(LORA_REGISTRY_PATH, encoding="utf-8") as handle:
        raw = json.load(handle)

    entries: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        for name, value in raw.items():
            if isinstance(value, dict):
                path = value.get("path")
            else:
                path = value
            if not isinstance(path, str):
                continue
            clean_name = normalize_lora_name(str(name))
            if clean_name and lora_path_is_valid(path):
                entries.append((clean_name, path))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = normalize_lora_name(str(item.get("name") or ""))
            path = item.get("path")
            if name and isinstance(path, str) and lora_path_is_valid(path):
                entries.append((name, path))

    seen = set()
    deduped = []
    for name, path in sorted(entries, key=lambda item: item[0]):
        if name in seen:
            continue
        seen.add(name)
        deduped.append((name, path))
    return deduped


def configured_loras() -> list[tuple[str, str]]:
    registry_exists = os.path.isfile(LORA_REGISTRY_PATH)
    try:
        registry_entries = read_lora_registry_file()
    except Exception as exc:
        print(f"[sa3] could not read LoRA registry {LORA_REGISTRY_PATH}: {exc}")
        return []
    if registry_exists:
        return registry_entries
    return scan_lora_dir()


def lora_payload(entries: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [{"index": i, "name": name, "path": path} for i, (name, path) in enumerate(entries)]


def friendly_load_error(error: Exception) -> str:
    raw = str(error)
    lower = raw.lower()
    if isinstance(error, MLXRuntimeUnavailableError) or "mlx is not installed" in lower:
        return (
            "MLX runtime is unavailable in this environment. "
            "Run sa3 on an Apple Silicon Mac with the MLX package installed."
        )
    if not hf_token_configured():
        return (
            "HF_TOKEN is not configured. Save a Hugging Face read token in "
            "gary4local, then accept the model terms for Stable Audio 3 Medium."
        )
    if any(marker in lower for marker in ("401", "403", "gated", "restricted", "access")):
        return (
            "Hugging Face token is configured, but this account may not have "
            "accepted all gated model terms for SA3. Open the Stable Audio 3 "
            "Medium model page, confirm access, then retry."
        )
    return raw


def hf_cache_root() -> Path:
    cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache and cache.strip():
        return Path(cache).expanduser().resolve()

    hf_home = os.environ.get("HF_HOME")
    if hf_home and hf_home.strip():
        return Path(hf_home).expanduser().resolve() / "hub"

    return Path.home() / ".cache" / "huggingface" / "hub"


def repo_snapshot_root(repo_id: str) -> Path:
    return hf_cache_root() / f"models--{repo_id.replace('/', '--')}" / "snapshots"


def repo_has_any_snapshot(repo_id: str) -> bool:
    root = repo_snapshot_root(repo_id)
    return root.exists() and any(path.is_dir() for path in root.iterdir())


def cached_repo_missing_files(repo_id: str, filenames: list[str]) -> list[str]:
    missing: list[str] = []
    if try_to_load_from_cache is None:
        return list(filenames)
    for filename in filenames:
        cached = try_to_load_from_cache(repo_id, filename)
        if not isinstance(cached, str):
            missing.append(filename)
    return missing


def sa3_required_download_targets() -> list[dict[str, Any]]:
    model_target = current_sa3_model_target()
    targets: list[dict[str, Any]] = []
    if model_target is not None:
        targets.append(
            {
                "type": "files",
                "repo_id": model_target["repo_id"],
                "label": model_target["label"],
                "display_label": f"{model_target['repo_id']} ({model_target['label']})",
                "files": list(model_target["files"]),
            }
        )
    text_target = current_sa3_text_encoder_target()
    targets.append(
        {
            "type": "subfolder",
            **text_target,
        }
    )
    return targets


def sa3_inventory_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_target = current_sa3_model_target()
    if model_target is not None:
        missing = cached_repo_missing_files(model_target["repo_id"], list(model_target["files"]))
        rows.append(
            {
                "repo_id": model_target["repo_id"],
                "label": f"{model_target['label']} required files",
                "downloaded": not missing,
                "missing": missing,
            }
        )

    text_target = current_sa3_text_encoder_target()
    text_markers = current_sa3_text_encoder_markers()
    text_missing = cached_repo_missing_files(text_target["repo_id"], text_markers)
    rows.append(
        {
            "repo_id": text_target["repo_id"],
            "label": f"{text_target['label']} bundled files",
            "downloaded": not text_missing,
            "missing": text_missing,
        }
    )
    return rows


def build_predownload_queue_status(
    *,
    status: str,
    message: str,
    target: str,
    stage_name: str,
    stage_index: int,
    stage_total: int,
    download_percent: int,
) -> dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "target": target,
        "repo_id": target,
        "stage_name": stage_name,
        "stage_index": stage_index,
        "stage_total": stage_total,
        "download_percent": max(0, min(100, int(download_percent))),
    }


def upsert_predownload_session(
    session_id: str,
    *,
    status: str,
    progress: int,
    queue_status: dict[str, Any],
    error: str | None,
    target: str,
) -> None:
    now = time.time()
    with predownload_sessions_lock:
        predownload_sessions[session_id] = {
            "status": status,
            "progress": max(0, min(100, int(progress))),
            "queue_status": queue_status,
            "error": error,
            "target": target,
            "updated_at": now,
        }

        stale_before = now - PREDOWNLOAD_TTL_SECONDS
        stale_ids = [
            key
            for key, value in predownload_sessions.items()
            if value.get("updated_at", 0.0) < stale_before
        ]
        for key in stale_ids:
            predownload_sessions.pop(key, None)


def read_predownload_session(session_id: str) -> dict[str, Any] | None:
    with predownload_sessions_lock:
        session = predownload_sessions.get(session_id)
        return None if session is None else session.copy()


def require_hf_download_support() -> None:
    if hf_hub_download is None or snapshot_download is None:
        raise RuntimeError(
            "huggingface_hub download helpers are unavailable in this environment."
        )


def perform_predownload_target(target: dict[str, Any], *, token: str) -> None:
    require_hf_download_support()
    repo_id = str(target["repo_id"])
    if target["type"] == "files":
        for filename in target["files"]:
            hf_hub_download(repo_id=repo_id, filename=str(filename), token=token)
        return

    if target["type"] == "subfolder":
        subfolder = str(target["subfolder"]).strip().strip("/")
        snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=[f"{subfolder}/*"],
            ignore_patterns=[
                "*.h5",
                "*.msgpack",
                "*.onnx",
                "flax_model.msgpack",
                "tf_model.h5",
            ],
        )
        return

    if target["type"] == "snapshot":
        snapshot_download(
            repo_id=repo_id,
            token=token,
            ignore_patterns=[
                "*.h5",
                "*.msgpack",
                "*.onnx",
                "flax_model.msgpack",
                "tf_model.h5",
            ],
        )
        return

    raise ValueError(f"unknown predownload target type: {target['type']!r}")


def run_sa3_predownload_task(session_id: str, payload: dict[str, Any]) -> None:
    target_type = str(payload.get("target_type", "required")).strip().lower()
    if target_type not in {"required", "repo"}:
        upsert_predownload_session(
            session_id,
            status="failed",
            progress=0,
            queue_status=build_predownload_queue_status(
                status="failed",
                message="target_type must be 'required' or 'repo'",
                target="sa3",
                stage_name="prepare",
                stage_index=0,
                stage_total=0,
                download_percent=0,
            ),
            error="target_type must be 'required' or 'repo'",
            target="sa3",
        )
        return

    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not token:
        error = "HF_TOKEN is not configured. Save your Hugging Face token in gary4local first."
        upsert_predownload_session(
            session_id,
            status="failed",
            progress=0,
            queue_status=build_predownload_queue_status(
                status="failed",
                message=error,
                target="sa3",
                stage_name="prepare",
                stage_index=0,
                stage_total=0,
                download_percent=0,
            ),
            error=error,
            target="sa3",
        )
        return

    if target_type == "repo":
        repo_id = str(payload.get("repo_id", "")).strip()
        if not repo_id:
            error = "repo_id is required when target_type is 'repo'."
            upsert_predownload_session(
                session_id,
                status="failed",
                progress=0,
                queue_status=build_predownload_queue_status(
                    status="failed",
                    message=error,
                    target="sa3",
                    stage_name="prepare",
                    stage_index=0,
                    stage_total=0,
                    download_percent=0,
                ),
                error=error,
                target="sa3",
            )
            return
        targets = [
            {
                "type": "snapshot",
                "repo_id": repo_id,
                "label": repo_id,
                "display_label": repo_id,
            }
        ]
        target_label = repo_id
    else:
        targets = sa3_required_download_targets()
        target_label = "required sa3 models"

    try:
        configure_hf_auth()
        stage_total = len(targets)
        for offset, target in enumerate(targets, start=1):
            label = str(target["display_label"])
            upsert_predownload_session(
                session_id,
                status="processing",
                progress=max(1, int(((offset - 1) / max(1, stage_total)) * 100)),
                queue_status=build_predownload_queue_status(
                    status="processing",
                    message=f"downloading {label}...",
                    target=str(target["repo_id"]),
                    stage_name="download",
                    stage_index=offset,
                    stage_total=stage_total,
                    download_percent=0,
                ),
                error=None,
                target=target_label,
            )
            perform_predownload_target(target, token=token)
            upsert_predownload_session(
                session_id,
                status="processing",
                progress=min(99, int((offset / max(1, stage_total)) * 100)),
                queue_status=build_predownload_queue_status(
                    status="processing",
                    message=f"downloaded {label}",
                    target=str(target["repo_id"]),
                    stage_name="download",
                    stage_index=offset,
                    stage_total=stage_total,
                    download_percent=100,
                ),
                error=None,
                target=target_label,
            )

        upsert_predownload_session(
            session_id,
            status="completed",
            progress=100,
            queue_status=build_predownload_queue_status(
                status="completed",
                message=f"downloaded {target_label}",
                target=target_label,
                stage_name="complete",
                stage_index=len(targets),
                stage_total=len(targets),
                download_percent=100,
            ),
            error=None,
            target=target_label,
        )
    except Exception as exc:
        error = friendly_load_error(exc)
        upsert_predownload_session(
            session_id,
            status="failed",
            progress=0,
            queue_status=build_predownload_queue_status(
                status="failed",
                message=error,
                target=target_label,
                stage_name="failed",
                stage_index=0,
                stage_total=len(targets),
                download_percent=0,
            ),
            error=error,
            target=target_label,
        )


def load_pipeline(force: bool = False) -> StableAudioMLXPipeline:
    global pipe, model_loaded, model_loading, model_error, last_load_seconds
    global model_sample_rate, model_device
    global lora_registry, lora_name_to_index

    with model_lock:
        if pipe is not None and not force:
            return pipe

        model_loading = True
        model_error = None
        if force:
            pipe = None
            model_loaded = False
            cleanup_cuda()

        started = time.time()
        try:
            configure_hf_auth()
            print(
                "[sa3] loading Stable Audio 3 MLX pipeline "
                f"model={MODEL_NAME} attention={MLX_ATTENTION} "
                f"dtype={MLX_DTYPE} dit_dtype={MLX_DIT_DTYPE} "
                f"text_dtype={MLX_TEXT_DTYPE} number_dtype={MLX_NUMBER_DTYPE} "
                f"autoencoder_dtype={MLX_AUTOENCODER_DTYPE} "
                f"torch_device={TORCH_DEVICE} torch_model_half={TORCH_MODEL_HALF}"
            )
            loaded = StableAudioMLXPipeline.from_pretrained_cached(
                MODEL_NAME,
                torch_device=TORCH_DEVICE,
                dtype=MLX_DTYPE,
                dit_dtype=MLX_DIT_DTYPE,
                text_dtype=MLX_TEXT_DTYPE,
                number_dtype=MLX_NUMBER_DTYPE,
                autoencoder_dtype=MLX_AUTOENCODER_DTYPE,
                attention=MLX_ATTENTION,
                model_half=TORCH_MODEL_HALF,
                write_cache=SA3_MLX_CACHE_ENABLED,
            )

            registry = configured_loras()
            if registry:
                paths = [path for _, path in registry]
                names = [name for name, _ in registry]
                print(f"[sa3] preloading {len(paths)} LoRA(s): {[name for name, _ in registry]}")
                loaded.load_lora(paths, names=names)
            else:
                print(f"[sa3] no LoRA files configured")

            lora_registry = registry
            lora_name_to_index = {name: i for i, (name, _) in enumerate(registry)}
            pipe = loaded
            model_loaded = True
            model_sample_rate = int(getattr(loaded, "sample_rate", OUTPUT_SAMPLE_RATE))
            model_device = f"mlx/{MLX_ATTENTION}"
            last_load_seconds = round(time.time() - started, 2)
            print(
                f"[sa3] model ready in {last_load_seconds}s "
                f"sr={model_sample_rate} device={model_device} "
                f"runtime_available={mlx_runtime_available()} mem={cuda_mem_mb()}"
            )
            return loaded
        except Exception as exc:
            model_loaded = False
            model_error = friendly_load_error(exc)
            print(f"[sa3] model load failed: {model_error}")
            traceback.print_exc()
            raise
        finally:
            model_loading = False


def unload_pipeline() -> dict[str, Any]:
    global pipe, model_loaded, model_error
    with model_lock:
        before = cuda_mem_mb()
        pipe = None
        model_loaded = False
        model_error = None
        cleanup_cuda()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        after = cuda_mem_mb()
        freed = None
        if before and after:
            freed = round(before["allocated_mb"] - after["allocated_mb"], 1)
        return {"status": "unloaded", "freed_mb": freed, "before": before, "after": after}


def create_session(meta: dict[str, Any]) -> str:
    session_id = str(uuid.uuid4())[:12]
    with sessions_lock:
        sessions[session_id] = {
            "status": "queued",
            "phase": "queued",
            "generation_in_progress": True,
            "transform_in_progress": meta.get("mode") == "transform",
            "progress": 0,
            "step": 0,
            "total_steps": meta.get("steps", DEFAULT_STEPS),
            "audio_data": None,
            "error": None,
            "meta": meta,
            "created_at": time.time(),
        }
    return session_id


def update_session(session_id: str, **updates: Any) -> None:
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id].update(updates)


def _resolve_completed_sampling_step(
    payload: dict[str, Any],
    *,
    total_steps: int,
    last_completed_step: int,
) -> int:
    raw_step = payload.get("step")
    if isinstance(raw_step, (int, float)):
        completed_step = int(raw_step)
    else:
        raw_index = payload.get("i")
        if not isinstance(raw_index, (int, float)):
            return last_completed_step
        step_index = int(raw_index)
        if step_index < 0:
            return last_completed_step
        candidates = []
        if 0 <= step_index < total_steps:
            candidates.append(step_index + 1)
        if 1 <= step_index <= total_steps:
            candidates.append(step_index)
        viable = [candidate for candidate in candidates if candidate >= last_completed_step]
        completed_step = min(viable) if viable else (max(candidates) if candidates else last_completed_step)
    return max(last_completed_step, min(max(total_steps, 1), completed_step))


def _sampling_progress_from_step(completed_step: int, total_steps: int) -> int:
    fraction = max(0.0, min(1.0, completed_step / float(max(total_steps, 1))))
    progress = GENERATION_SAMPLING_PROGRESS_START + int(
        round((GENERATION_SAMPLING_PROGRESS_END - GENERATION_SAMPLING_PROGRESS_START) * fraction)
    )
    return max(
        GENERATION_SAMPLING_PROGRESS_START,
        min(GENERATION_SAMPLING_PROGRESS_END, progress),
    )


def get_session(session_id: str) -> dict[str, Any] | None:
    with sessions_lock:
        value = sessions.get(session_id)
        return value.copy() if value else None


def cleanup_old_sessions(max_age_seconds: float = 1800.0) -> None:
    now = time.time()
    with sessions_lock:
        expired = [sid for sid, s in sessions.items() if now - s.get("created_at", 0) > max_age_seconds]
        for sid in expired:
            sessions.pop(sid, None)


def get_json_body() -> dict[str, Any] | None:
    data = request.get_json(silent=True)
    if data is not None:
        return data
    raw = request.get_data(as_text=True)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def extract_bpm(prompt: str) -> float | None:
    match = _BPM_RE.search(prompt or "")
    return float(match.group(1)) if match else None


def parse_float(data: dict[str, Any], key: str, default: float) -> float:
    raw = data.get(key, default)
    if raw in (None, ""):
        return default
    return float(raw)


def parse_int(data: dict[str, Any], key: str, default: int) -> int:
    raw = data.get(key, default)
    if raw in (None, ""):
        return default
    return int(raw)


def parse_optional_float(data: dict[str, Any], key: str, default: float | None) -> float | None:
    raw = data.get(key, default)
    if raw in (None, ""):
        return default
    if isinstance(raw, str) and raw.strip().lower() in {"off", "none", "disable", "disabled"}:
        return None
    return float(raw)


def parse_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    raw = data.get(key, default)
    if raw in (None, ""):
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean, got {raw!r}")


def parse_peak_normalize_db(data: dict[str, Any]) -> float | None:
    return parse_optional_float(data, "peak_normalize_db", PEAK_NORM_DB)


def parse_limiter_ceiling_db(data: dict[str, Any]) -> float | None:
    value = parse_optional_float(data, "limiter_ceiling_db", LIMITER_CEILING_DB)
    return None if value is not None and value > 0.0 else value


def float_or_default(value: float | None, default: float) -> float:
    return default if value is None else value


def loudness_params(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "latent_rescale": float_or_default(
            parse_optional_float(data, "latent_rescale", LATENT_RESCALE),
            1.0,
        ),
        "latent_shift": float_or_default(
            parse_optional_float(data, "latent_shift", LATENT_SHIFT),
            0.0,
        ),
        "latent_target_std": parse_optional_float(data, "latent_target_std", LATENT_TARGET_STD),
        "latent_adapt_min": float_or_default(
            parse_optional_float(data, "latent_adapt_min", LATENT_ADAPT_MIN),
            LATENT_ADAPT_MIN,
        ),
        "latent_adapt_max": float_or_default(
            parse_optional_float(data, "latent_adapt_max", LATENT_ADAPT_MAX),
            LATENT_ADAPT_MAX,
        ),
        "peak_normalize_db": parse_peak_normalize_db(data),
        "limiter_ceiling_db": parse_limiter_ceiling_db(data),
        "limiter_knee": float_or_default(
            parse_optional_float(data, "limiter_knee", LIMITER_KNEE),
            LIMITER_KNEE,
        ),
    }


def decode_params(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunked_decode": parse_bool(data, "chunked_decode", CHUNKED_DECODE),
        "decode_chunk_size": parse_int(data, "decode_chunk_size", DECODE_CHUNK_SIZE),
        "decode_overlap": parse_int(data, "decode_overlap", DECODE_OVERLAP),
        "decode_chunk_batch_size": parse_int(
            data,
            "decode_chunk_batch_size",
            DECODE_CHUNK_BATCH_SIZE,
        ),
    }


def resolve_seed(data: dict[str, Any]) -> int:
    seed = parse_int(data, "seed", -1)
    return random.randint(0, 99999) if seed == -1 else seed


def resolve_dist_shift(shift: str):
    shift = (shift or "default").lower()
    if shift == "default":
        return None
    return shift


def to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def to_torch_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    array = to_numpy_array(value)
    if array.dtype != np.float32 and dtype == torch.float32:
        array = array.astype(np.float32, copy=False)
    return torch.from_numpy(array).to(dtype=dtype)


def to_mlx_array(value: Any, *, dtype_name: str | None = None):
    mx = import_mlx_core(required=True)
    array = value if type(value).__module__.startswith("mlx") else None
    if array is None:
        array = mx.array(to_numpy_array(value).astype(np.float32, copy=False))
    if dtype_name is not None:
        target_dtype = getattr(mx, dtype_name)
        if array.dtype != target_dtype:
            array = array.astype(target_dtype)
    return array


def decode_mlx_latents_to_audio(
    local_pipe: StableAudioMLXPipeline,
    latents: Any,
    *,
    chunked_decode: bool = False,
    decode_chunk_size: int = DECODE_CHUNK_SIZE,
    decode_overlap: int = DECODE_OVERLAP,
    decode_chunk_batch_size: int = DECODE_CHUNK_BATCH_SIZE,
) -> torch.Tensor:
    latents_mx = to_mlx_array(latents, dtype_name=local_pipe.autoencoder_dtype_name)
    audio_mx = local_pipe.autoencoder.decode_audio(
        latents_mx,
        chunked=bool(chunked_decode),
        chunk_size=int(decode_chunk_size),
        overlap=int(decode_overlap),
        chunk_batch_size=int(decode_chunk_batch_size),
        add_bottleneck_noise=False,
    )
    mx = import_mlx_core(required=True)
    mx.eval(audio_mx)
    return to_torch_tensor(np.asarray(audio_mx, dtype=np.float32))


def validate_common(data: dict[str, Any], require_duration: bool = True) -> list[str]:
    errors: list[str] = []
    if not (data.get("prompt") or "").strip():
        errors.append("prompt is required")

    if require_duration:
        try:
            duration = parse_float(data, "duration", DEFAULT_DURATION)
            if duration <= 0 or duration > MAX_DURATION:
                errors.append(f"duration must be in (0, {MAX_DURATION}] seconds")
        except (TypeError, ValueError):
            errors.append("duration must be a number")

    try:
        steps = parse_int(data, "steps", DEFAULT_STEPS)
        if steps < 1 or steps > 200:
            errors.append("steps must be in [1, 200]")
    except (TypeError, ValueError):
        errors.append("steps must be an integer")

    try:
        cfg = parse_float(data, "cfg_scale", DEFAULT_CFG)
        if cfg < 0 or cfg > 25:
            errors.append("cfg_scale must be in [0, 25]")
    except (TypeError, ValueError):
        errors.append("cfg_scale must be a number")

    shift = (data.get("shift") or "default").lower()
    if shift not in VALID_SHIFTS:
        errors.append(f"shift must be one of {sorted(VALID_SHIFTS)}")

    if data.get("loras") is not None and not isinstance(data.get("loras"), list):
        errors.append("loras must be a list")

    try:
        loudness = loudness_params(data)
        target_std = loudness["latent_target_std"]
        if loudness["latent_rescale"] < 0:
            errors.append("latent_rescale must be >= 0")
        if target_std is not None and target_std <= 0:
            errors.append("latent_target_std must be > 0 or off")
        if loudness["latent_adapt_min"] < 0 or loudness["latent_adapt_max"] < loudness["latent_adapt_min"]:
            errors.append("latent_adapt_min/max must satisfy 0 <= min <= max")
        if loudness["limiter_knee"] <= 0 or loudness["limiter_knee"] > 1:
            errors.append("limiter_knee must be in (0, 1]")
    except (TypeError, ValueError):
        errors.append("loudness fields must be numbers, empty, or off")

    try:
        decode = decode_params(data)
        if decode["decode_chunk_size"] < 1:
            errors.append("decode_chunk_size must be >= 1")
        if decode["decode_overlap"] < 0:
            errors.append("decode_overlap must be >= 0")
        if decode["decode_overlap"] >= decode["decode_chunk_size"]:
            errors.append("decode_overlap must be smaller than decode_chunk_size")
        if decode["decode_chunk_batch_size"] < 1:
            errors.append("decode_chunk_batch_size must be >= 1")
    except (TypeError, ValueError):
        errors.append("decode fields must be valid numbers and booleans")

    return errors


def resolve_loras(data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = data.get("loras")
    if entries is None:
        selected = normalize_lora_name(str(data.get("lora") or ""))
        if not selected or selected == "none":
            return []
        if selected == "default":
            selected = DEFAULT_LORA_NAME or (lora_registry[0][0] if lora_registry else "")
        if not selected:
            return []
        entries = [{"name": selected, "strength": data.get("lora_strength", 1.0)}]

    resolved = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each loras entry must be an object")
        name = normalize_lora_name(str(entry.get("name") or ""))
        if not name:
            raise ValueError("each loras entry needs a name")
        if name not in lora_name_to_index:
            raise ValueError(f"unknown LoRA '{name}'. available: {list(lora_name_to_index)}")

        interval_min = float(entry.get("interval_min", 0.0))
        interval_max = float(entry.get("interval_max", 1.0))
        if not (0.0 <= interval_min <= interval_max <= 1.0):
            raise ValueError(f"LoRA '{name}': require 0 <= interval_min <= interval_max <= 1")

        resolved.append(
            {
                "lora_index": lora_name_to_index[name],
                "name": name,
                "strength": float(entry.get("strength", 1.0)),
                "interval": (interval_min, interval_max),
                "layer_filter": str(entry.get("layer_filter", "") or ""),
            }
        )
    return resolved


def common_params(data: dict[str, Any], duration: float | None = None) -> dict[str, Any]:
    return {
        "prompt": data["prompt"].strip(),
        "negative_prompt": (data.get("negative_prompt", DEFAULT_NEGATIVE) or "").strip(),
        "duration": duration if duration is not None else parse_float(data, "duration", DEFAULT_DURATION),
        "steps": parse_int(data, "steps", DEFAULT_STEPS),
        "cfg_scale": parse_float(data, "cfg_scale", DEFAULT_CFG),
        "shift": (data.get("shift") or "default").lower(),
        "sampler_type": data.get("sampler_type", DEFAULT_SAMPLER),
        "seed": resolve_seed(data),
        "target_samples": None,
        "mode": "generate",
        "loras_request": data.get("loras"),
        "lora": data.get("lora"),
        "lora_strength": data.get("lora_strength"),
        **loudness_params(data),
        **decode_params(data),
    }


def decode_audio_data(data: dict[str, Any]) -> tuple[int, torch.Tensor]:
    encoded = data.get("audio_data")
    if not encoded:
        raise ValueError("audio_data (base64 WAV) is required")
    if isinstance(encoded, str) and encoded.startswith("data:") and "," in encoded:
        encoded = encoded.split(",", 1)[1]
    raw = base64.b64decode(encoded)
    waveform, sample_rate = torchaudio.load(io.BytesIO(raw))
    return sample_rate, waveform


def encode_wav_base64(audio: Any, sample_rate: int) -> str:
    # audio is [batch, channels, samples]. The API returns one rendered sequence.
    audio = to_torch_tensor(audio)
    if audio.dim() == 3:
        audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).clamp(-1, 1).cpu()
    wav = io.BytesIO()
    sf.write(wav, audio.transpose(0, 1).numpy(), sample_rate, format="WAV", subtype="PCM_16")
    wav.seek(0)
    return base64.b64encode(wav.read()).decode("ascii")


def apply_target_length(audio: Any, target_samples: int | None) -> torch.Tensor:
    audio = to_torch_tensor(audio)
    if target_samples is None:
        return audio
    current = audio.shape[-1]
    if current > target_samples:
        return audio[..., :target_samples]
    if current < target_samples:
        return torch.nn.functional.pad(audio, (0, target_samples - current))
    return audio


def loudness_meta_from_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "latent_rescale": params["latent_rescale"],
        "latent_shift": params["latent_shift"],
        "latent_target_std": params["latent_target_std"],
        "latent_adapt_min": params["latent_adapt_min"],
        "latent_adapt_max": params["latent_adapt_max"],
        "latent_factor": 1.0,
        "latent_std": None,
        "peak_normalize_db": params["peak_normalize_db"],
        "peak_normalize_gain": None,
        "limiter_ceiling_db": params["limiter_ceiling_db"],
        "limiter_knee": params["limiter_knee"],
        "limiter_limited_fraction": None,
        "decoded_peak": None,
        "final_peak": None,
    }


def should_use_loudness_latent_path(params: dict[str, Any]) -> bool:
    return (
        LATENT_DIAG
        or params["latent_rescale"] != 1.0
        or params["latent_shift"] != 0.0
        or params["latent_target_std"] is not None
        or params["peak_normalize_db"] is not None
        or params["limiter_ceiling_db"] is not None
    )


def apply_loudness_chain(
    local_pipe: StableAudioMLXPipeline,
    latents: Any,
    params: dict[str, Any],
    sample_rate: int,
    session_id: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mx = import_mlx_core(required=True)
    meta = loudness_meta_from_params(params)
    target_std = params["latent_target_std"]
    adaptive = target_std is not None and target_std > 0.0
    latent_factor = params["latent_rescale"]
    shift = params["latent_shift"]
    latents_mx = to_mlx_array(latents)
    latents_np = None
    if LATENT_DIAG or adaptive:
        latents_np = np.asarray(latents_mx, dtype=np.float32)
    if progress_callback is not None:
        progress_callback(GENERATION_DECODE_PREP_PROGRESS, "decode_prepare")

    if LATENT_DIAG:
        if latents_np is None:
            latents_np = np.asarray(latents_mx, dtype=np.float32)
        seg = max(1, latents_np.shape[-1] // 5)
        head = latents_np[..., :seg]
        tail = latents_np[..., -seg:]
        print(
            f"[{session_id}] LATENT diag shape={tuple(latents_np.shape)} "
            f"min={float(latents_np.min()):.4f} max={float(latents_np.max()):.4f} "
            f"mean={float(latents_np.mean()):.4f} std={float(latents_np.std()):.4f} "
            f"absmax={float(np.abs(latents_np).max()):.4f} "
            f"head_std={float(head.std()):.4f} tail_std={float(tail.std()):.4f}"
        )
        del head, tail

    if adaptive:
        if latents_np is None:
            latents_np = np.asarray(latents_mx, dtype=np.float32)
        cur_std = float(latents_np.std())
        meta["latent_std"] = round(cur_std, 6)
        if cur_std > 1e-6:
            latent_factor = target_std / cur_std
            latent_factor = min(params["latent_adapt_max"], max(params["latent_adapt_min"], latent_factor))
        else:
            latent_factor = 1.0
        print(
            f"[{session_id}] adaptive latent rescale std={cur_std:.4f} "
            f"target={target_std} factor={latent_factor:.4f}"
        )
    elif LATENT_DIAG:
        if latents_np is None:
            latents_np = np.asarray(latents_mx, dtype=np.float32)
        meta["latent_std"] = round(float(latents_np.std()), 6)

    if latent_factor != 1.0 or shift != 0.0:
        if latents_mx.dtype != mx.float32:
            latents_mx = latents_mx.astype(mx.float32)
        latents_mx = latents_mx * latent_factor + shift
    meta["latent_factor"] = round(float(latent_factor), 6)

    if progress_callback is not None:
        progress_callback(GENERATION_DECODING_PROGRESS, "decoding")
    audio = decode_mlx_latents_to_audio(
        local_pipe,
        latents_mx,
        chunked_decode=params["chunked_decode"],
        decode_chunk_size=params["decode_chunk_size"],
        decode_overlap=params["decode_overlap"],
        decode_chunk_batch_size=params["decode_chunk_batch_size"],
    ).float()

    if not params.get("target_samples"):
        keep = int(params["duration"] * sample_rate)
        if audio.shape[-1] > keep:
            audio = audio[..., :keep]

    decoded_peak = audio.detach().abs().max().item()
    meta["decoded_peak"] = round(decoded_peak, 6)
    norm_db = params["peak_normalize_db"]
    if norm_db is not None and decoded_peak > 1e-6:
        gain = (10.0 ** (norm_db / 20.0)) / decoded_peak
        audio = audio * gain
        meta["peak_normalize_gain"] = round(gain, 6)

    lim_db = params["limiter_ceiling_db"]
    if lim_db is not None:
        ceiling = 10.0 ** (lim_db / 20.0)
        knee_fraction = max(1e-6, min(1.0, params["limiter_knee"]))
        knee = ceiling * knee_fraction
        mag = audio.abs()
        over = mag > knee
        limited = int(over.sum().item())
        if limited:
            if knee >= ceiling:
                limited_mag = torch.minimum(mag, mag.new_tensor(ceiling))
            else:
                limited_mag = knee + (ceiling - knee) * torch.tanh((mag - knee) / (ceiling - knee))
            audio = torch.where(over, torch.sign(audio) * limited_mag, audio)
        fraction = limited / max(1, audio.numel())
        meta["limiter_limited_fraction"] = round(float(fraction), 8)
        if LATENT_DIAG:
            print(
                f"[{session_id}] limiter ceiling={lim_db}dB knee={knee_fraction} "
                f"limited={limited}/{audio.numel()} ({100.0 * fraction:.4f}%)"
            )

    final_peak = audio.detach().abs().max().item()
    meta["final_peak"] = round(final_peak, 6)
    if LATENT_DIAG:
        clip_fraction = (audio.detach().abs() > 1.0).float().mean().item()
        print(
            f"[{session_id}] DECODED diag peak={final_peak:.4f} "
            f"clip>1.0={clip_fraction:.6f} norm_db={norm_db}"
        )

    if progress_callback is not None:
        progress_callback(GENERATION_POSTPROCESS_PROGRESS, "postprocessing")

    return audio, meta


def prepare_latent_prefix_inputs(
    local_pipe: StableAudioMLXPipeline,
    params: dict[str, Any],
    continuation: dict[str, Any],
):
    mx = import_mlx_core(required=True)
    conditioning = [{"prompt": params["prompt"], "seconds_total": params["duration"]}]
    audio_sample_size = adapt_sample_size(
        local_pipe.model_config,
        conditioning,
        MAX_SAMPLE_SIZE,
        6.0,
        sample_rate=local_pipe.sample_rate,
        downsampling_ratio=local_pipe.downsampling_ratio,
    )
    downsampling_ratio = int(local_pipe.downsampling_ratio)
    latent_sample_size = int(math.ceil(audio_sample_size / downsampling_ratio))
    fixed_prefix_data = local_pipe._encode_audio_input(
        params["inpaint_audio"],
        audio_sample_size=audio_sample_size,
        latent_length=latent_sample_size,
        batch_size=1,
        mlx_dtype=getattr(mx, local_pipe.dtype_name),
        autoencoder_dtype=getattr(mx, local_pipe.autoencoder_dtype_name),
    )
    prefix_samples = round(float(continuation["source_duration"]) * int(local_pipe.sample_rate))
    prefix_tokens = min(
        latent_sample_size,
        max(1, int(round(prefix_samples / downsampling_ratio))),
    )
    fixed_prefix_mask = np.zeros((1, 1, latent_sample_size), dtype=np.float32)
    fixed_prefix_mask[:, :, :prefix_tokens] = 1.0
    fixed_prefix_mask = mx.array(fixed_prefix_mask).astype(fixed_prefix_data.dtype)
    mx.eval(fixed_prefix_data, fixed_prefix_mask)
    continuation["prefix_latent_tokens"] = int(prefix_tokens)
    continuation["latent_sample_size"] = int(latent_sample_size)
    return fixed_prefix_data, fixed_prefix_mask


def generation_worker(session_id: str, params: dict[str, Any]) -> None:
    started = time.time()
    local_pipe: StableAudioMLXPipeline | None = None
    try:
        update_session(
            session_id,
            status="generating",
            phase="loading_model",
            progress=GENERATION_PROGRESS_LOADING,
        )

        with generation_lock:
            local_pipe = load_pipeline()
            sr = int(getattr(local_pipe, "sample_rate", OUTPUT_SAMPLE_RATE))
            loras = resolve_loras(
                {
                    "loras": params.get("loras_request"),
                    "lora": params.get("lora"),
                    "lora_strength": params.get("lora_strength"),
                }
            )
            update_session(
                session_id,
                status="generating",
                phase="preparing",
                progress=GENERATION_PROGRESS_PREPARING,
            )
            loaded_lora_count = len(getattr(local_pipe, "lora_paths", ()))
            requested_loras = {config["lora_index"]: config for config in loras}
            for idx in range(loaded_lora_count):
                strength = requested_loras.get(idx, {}).get("strength", 0.0)
                local_pipe.set_lora_strength(strength, lora_index=idx)
            lora_configs = [
                {
                    "lora_index": idx,
                    "interval": requested_loras.get(idx, {}).get("interval", (0.0, 1.0)),
                    "layer_filter": requested_loras.get(idx, {}).get("layer_filter", ""),
                }
                for idx in range(loaded_lora_count)
            ] if loaded_lora_count else None
            sampling_state = {
                "last_completed_step": 0,
                "last_progress": GENERATION_SAMPLING_PROGRESS_START - 1,
                "started_at": 0.0,
                "first_step_seconds": None,
            }

            def on_step(info: dict[str, Any]) -> None:
                idx = _resolve_completed_sampling_step(
                    info,
                    total_steps=params["steps"],
                    last_completed_step=sampling_state["last_completed_step"],
                )
                if idx > 0 and sampling_state["first_step_seconds"] is None and sampling_state["started_at"] > 0.0:
                    sampling_state["first_step_seconds"] = round(
                        time.time() - sampling_state["started_at"],
                        3,
                    )
                progress = _sampling_progress_from_step(idx, params["steps"])
                if idx <= sampling_state["last_completed_step"] and progress <= sampling_state["last_progress"]:
                    return
                sampling_state["last_completed_step"] = idx
                sampling_state["last_progress"] = max(sampling_state["last_progress"], progress)
                update_session(
                    session_id,
                    status="generating",
                    phase="sampling",
                    step=idx,
                    total_steps=params["steps"],
                    progress=progress,
                )

            gen_kwargs = {
                "prompt": params["prompt"],
                "negative_prompt": params["negative_prompt"] or None,
                "duration": params["duration"],
                "sample_size": MAX_SAMPLE_SIZE,
                "steps": params["steps"],
                "cfg_scale": params["cfg_scale"],
                "seed": params["seed"],
                "dist_shift": resolve_dist_shift(params["shift"]),
                "sampler_type": params["sampler_type"],
                "chunked_decode": params["chunked_decode"],
                "decode_chunk_size": params["decode_chunk_size"],
                "decode_overlap": params["decode_overlap"],
                "decode_chunk_batch_size": params["decode_chunk_batch_size"],
                "callback": on_step,
            }
            if lora_configs is not None:
                gen_kwargs["lora_configs"] = lora_configs

            if params.get("init_audio") is not None:
                gen_kwargs["init_audio"] = params["init_audio"]
                gen_kwargs["init_noise_level"] = params["init_noise_level"]

            cont = params.get("continue")
            if params.get("inpaint_audio") is not None and cont and cont.get("mode") == "latent_prefix":
                fixed_prefix_data, fixed_prefix_mask = prepare_latent_prefix_inputs(
                    local_pipe,
                    params,
                    cont,
                )
                gen_kwargs["fixed_prefix_data"] = fixed_prefix_data
                gen_kwargs["fixed_prefix_mask"] = fixed_prefix_mask
            elif params.get("inpaint_audio") is not None:
                gen_kwargs["inpaint_audio"] = params["inpaint_audio"]
                gen_kwargs["inpaint_mask_start_seconds"] = params["inpaint_mask_start_seconds"]
                gen_kwargs["inpaint_mask_end_seconds"] = params["inpaint_mask_end_seconds"]

            generation_timings: dict[str, float | None] = {
                "time_to_first_sampling_step": None,
                "generate_call_seconds": None,
                "decode_seconds": None,
                "postprocess_seconds": None,
                "encode_seconds": None,
                "total_seconds": None,
            }
            update_session(
                session_id,
                status="generating",
                phase="conditioning",
                step=0,
                total_steps=params["steps"],
                progress=GENERATION_PROGRESS_CONDITIONING,
            )
            sampling_state["started_at"] = time.time()
            if should_use_loudness_latent_path(params):
                gen_kwargs["return_latents"] = True
                generate_started = time.time()
                latents = local_pipe.generate(**gen_kwargs)
                generation_timings["generate_call_seconds"] = round(time.time() - generate_started, 3)
                generation_timings["time_to_first_sampling_step"] = sampling_state["first_step_seconds"]
                update_session(
                    session_id,
                    status="generating",
                    phase="decode_prepare",
                    step=params["steps"],
                    total_steps=params["steps"],
                    progress=GENERATION_DECODE_PREP_PROGRESS,
                )
                decode_started = time.time()
                audio, loudness_meta = apply_loudness_chain(
                    local_pipe,
                    latents,
                    params,
                    sr,
                    session_id,
                    progress_callback=lambda progress, phase: update_session(
                        session_id,
                        status="generating",
                        phase=phase,
                        step=params["steps"],
                        total_steps=params["steps"],
                        progress=progress,
                    ),
                )
                generation_timings["decode_seconds"] = round(time.time() - decode_started, 3)
            else:
                generate_started = time.time()
                audio = to_torch_tensor(local_pipe.generate(**gen_kwargs))
                generation_timings["generate_call_seconds"] = round(time.time() - generate_started, 3)
                generation_timings["time_to_first_sampling_step"] = sampling_state["first_step_seconds"]
                update_session(
                    session_id,
                    status="generating",
                    phase="postprocessing",
                    step=params["steps"],
                    total_steps=params["steps"],
                    progress=GENERATION_POSTPROCESS_PROGRESS,
                )
                loudness_meta = loudness_meta_from_params(params)
                loudness_meta["decoded_peak"] = round(audio.detach().abs().max().item(), 6)
                loudness_meta["final_peak"] = loudness_meta["decoded_peak"]
            postprocess_started = time.time()
            audio = apply_target_length(audio, params.get("target_samples"))
            generation_timings["postprocess_seconds"] = round(time.time() - postprocess_started, 3)

        update_session(
            session_id,
            status="encoding",
            phase="encoding",
            step=params["steps"],
            total_steps=params["steps"],
            progress=GENERATION_ENCODING_PROGRESS,
        )
        encode_started = time.time()
        audio_data = encode_wav_base64(audio, int(model_sample_rate or OUTPUT_SAMPLE_RATE))
        generation_timings["encode_seconds"] = round(time.time() - encode_started, 3)
        generation_timings["total_seconds"] = round(time.time() - started, 3)
        print(
            f"[{session_id}] generation timings "
            f"first_step={generation_timings['time_to_first_sampling_step']}s "
            f"generate={generation_timings['generate_call_seconds']}s "
            f"decode={generation_timings['decode_seconds']}s "
            f"postprocess={generation_timings['postprocess_seconds']}s "
            f"encode={generation_timings['encode_seconds']}s "
            f"total={generation_timings['total_seconds']}s"
        )

        meta = {
            "mode": params["mode"],
            "prompt": params["prompt"],
            "negative_prompt": params["negative_prompt"],
            "duration": params["duration"],
            "steps": params["steps"],
            "cfg_scale": params["cfg_scale"],
            "shift": params["shift"],
            "sampler_type": params["sampler_type"],
            "seed": params["seed"],
            "sample_rate": int(model_sample_rate or OUTPUT_SAMPLE_RATE),
            "generation_seconds": round(time.time() - started, 3),
            "timings": generation_timings,
            "loras": [
                {
                    "name": config["name"],
                    "strength": config["strength"],
                    "interval": list(config["interval"]),
                    "layer_filter": config["layer_filter"],
                }
                for config in loras
            ],
            "loudness": loudness_meta,
        }
        for key in ("loop", "transform", "continue"):
            if params.get(key):
                meta[key] = params[key]

        update_session(
            session_id,
            status="completed",
            phase="completed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=100,
            audio_data=audio_data,
            meta=meta,
        )
    except Exception as exc:
        error = friendly_load_error(exc)
        update_session(
            session_id,
            status="failed",
            phase="failed",
            generation_in_progress=False,
            transform_in_progress=False,
            progress=0,
            error=error,
        )
    finally:
        del local_pipe
        cleanup_cuda()


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "sa3",
            "backend": "mlx",
            "model": MODEL_NAME,
            "model_loaded": model_loaded,
            "model_loading": model_loading,
            "model_error": model_error,
            "last_load_seconds": last_load_seconds,
            "loras": lora_payload(lora_registry if model_loaded else configured_loras()),
            "hf_token_configured": hf_token_configured(),
            "gate_links": SA3_MODEL_LINKS,
            "device": model_device,
            "mlx_runtime_available": mlx_runtime_available(),
            "mlx_precision": {
                "dtype": MLX_DTYPE,
                "dit_dtype": MLX_DIT_DTYPE,
                "text_dtype": MLX_TEXT_DTYPE,
                "number_dtype": MLX_NUMBER_DTYPE,
                "autoencoder_dtype": MLX_AUTOENCODER_DTYPE,
                "attention": MLX_ATTENTION,
                "torch_device": TORCH_DEVICE,
                "torch_model_half": TORCH_MODEL_HALF,
            },
            "decode_defaults": {
                "chunked_decode": CHUNKED_DECODE,
                "decode_chunk_size": DECODE_CHUNK_SIZE,
                "decode_overlap": DECODE_OVERLAP,
                "decode_chunk_batch_size": DECODE_CHUNK_BATCH_SIZE,
            },
            "cuda_available": torch.cuda.is_available(),
            "cuda_mem": cuda_mem_mb(),
            "sample_rate": model_sample_rate,
            "loudness_defaults": {
                "latent_rescale": LATENT_RESCALE,
                "latent_shift": LATENT_SHIFT,
                "latent_target_std": LATENT_TARGET_STD,
                "latent_adapt_min": LATENT_ADAPT_MIN,
                "latent_adapt_max": LATENT_ADAPT_MAX,
                "peak_normalize_db": PEAK_NORM_DB,
                "limiter_ceiling_db": LIMITER_CEILING_DB,
                "limiter_knee": LIMITER_KNEE,
                "continuation_tail_mode": CONTINUE_TAIL_MODE,
                "continuation_tail_pad": CONTINUE_TAIL_PAD,
            },
        }
    )


@app.route("/ready", methods=["GET"])
def ready():
    if model_loaded:
        return jsonify({"ready": True, "model": MODEL_NAME})
    return jsonify({"ready": False, "loading": model_loading, "error": model_error}), 503


@app.route("/load", methods=["POST"])
def load():
    try:
        already_loaded = model_loaded
        load_pipeline()
        return jsonify(
            {
                "success": True,
                "status": "already_loaded" if already_loaded else "loaded",
                "load_seconds": 0.0 if already_loaded else last_load_seconds,
                "sample_rate": model_sample_rate,
                "device": model_device,
                "backend": "mlx",
                "mlx_runtime_available": mlx_runtime_available(),
                "cuda_mem": cuda_mem_mb(),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": friendly_load_error(exc), "gate_links": SA3_MODEL_LINKS}), 503


@app.route("/unload", methods=["POST"])
def unload():
    if generation_lock.locked():
        return jsonify({"success": False, "error": "generation in progress - retry when idle"}), 409
    return jsonify({"success": True, **unload_pipeline()})


@app.route("/models/predownload_inventory", methods=["POST"])
@app.route("/api/models/predownload_inventory", methods=["POST"])
def predownload_inventory():
    return jsonify(
        {
            "success": True,
            "known_models": sa3_inventory_rows(),
            "gate_links": SA3_MODEL_LINKS,
        }
    )


@app.route("/models/predownload", methods=["POST"])
@app.route("/api/models/predownload", methods=["POST"])
def start_model_predownload():
    if not hf_token_configured():
        return jsonify(
            {
                "success": False,
                "error": "HF_TOKEN is not configured. Save your Hugging Face token first.",
                "gate_links": SA3_MODEL_LINKS,
            }
        ), 400

    payload = request.get_json(silent=True) or {}
    target_type = str(payload.get("target_type", "required")).strip().lower()
    target_label = (
        str(payload.get("repo_id", "")).strip()
        if target_type == "repo"
        else "required sa3 models"
    )
    if not target_label:
        target_label = "required sa3 models"

    session_id = str(uuid.uuid4())
    upsert_predownload_session(
        session_id,
        status="warming",
        progress=0,
        queue_status=build_predownload_queue_status(
            status="warming",
            message=f"preparing download for {target_label}",
            target=target_label,
            stage_name="prepare",
            stage_index=0,
            stage_total=0,
            download_percent=0,
        ),
        error=None,
        target=target_label,
    )
    threading.Thread(
        target=run_sa3_predownload_task,
        args=(session_id, payload),
        daemon=True,
    ).start()
    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "model_name": target_label,
            "message": f"started pre-download for {target_label}",
        }
    )


@app.route("/models/predownload_status/<session_id>", methods=["GET"])
@app.route("/api/models/predownload_status/<session_id>", methods=["GET"])
def get_model_predownload_status(session_id: str):
    session = read_predownload_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": "predownload session not found"}), 404

    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "model_name": session.get("target"),
            "status": session.get("status", "unknown"),
            "progress": int(session.get("progress", 0)),
            "queue_status": session.get("queue_status", {}),
            "error": session.get("error"),
        }
    )


@app.route("/loras", methods=["GET"])
def loras():
    entries = lora_registry if model_loaded else configured_loras()
    return jsonify(
        {
            "loras": lora_payload(entries),
            "default_lora": DEFAULT_LORA_NAME or None,
            "lora_dir": LORA_DIR,
            "registry_path": LORA_REGISTRY_PATH,
            "model_loaded": model_loaded,
        }
    )


@app.route("/reload", methods=["POST"])
def reload_loras():
    if not generation_lock.acquire(blocking=False):
        return jsonify({"success": False, "error": "generation in progress - retry when idle"}), 409
    try:
        previous = [name for name, _ in lora_registry]
        load_pipeline(force=True)
        return jsonify(
            {
                "success": True,
                "previous": previous,
                "loras": lora_payload(lora_registry),
            }
        )
    except Exception as exc:
        return jsonify({"success": False, "error": friendly_load_error(exc), "gate_links": SA3_MODEL_LINKS}), 503
    finally:
        generation_lock.release()


def read_json_file(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def ensure_default_prompt_pool() -> str:
    destination = Path(PROMPTS_DIR) / "defaults.json"
    if destination.exists():
        return str(destination)
    if not BUNDLED_DEFAULT_PROMPTS_PATH.exists():
        return str(destination)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(BUNDLED_DEFAULT_PROMPTS_PATH, destination)
    except Exception as exc:
        print(f"[sa3] failed to seed default prompts: {exc}")
    return str(destination)


@app.route("/prompts", methods=["GET"])
def prompts():
    defaults_path = ensure_default_prompt_pool()
    data = read_json_file(defaults_path) or {
        "version": 1,
        "dice": {"generic": [], "instrumental": [], "drums": []},
    }
    dice = {
        key: list(value)
        for key, value in (data.get("dice") or {}).items()
        if isinstance(value, list)
    }
    available_loras = []
    if os.path.isdir(PROMPTS_DIR):
        available_loras = sorted(
            os.path.splitext(filename)[0]
            for filename in os.listdir(PROMPTS_DIR)
            if filename.endswith(".json") and filename != "defaults.json"
        )

    selected_loras: list[str] = []
    seen_loras = set()
    for raw in request.args.getlist("lora"):
        for name in (piece.strip().lower() for piece in raw.split(",")):
            if name and name not in seen_loras:
                selected_loras.append(name)
                seen_loras.add(name)

    source: dict[str, Any] = {
        "generic": "defaults.json" if os.path.exists(defaults_path) else "empty"
    }
    bucket_seen: dict[str, set[Any]] = {}
    bucket_replaced = set()
    missing_loras = []
    for name in selected_loras:
        lora_data = read_json_file(os.path.join(PROMPTS_DIR, f"{name}.json"))
        lora_dice = lora_data.get("dice") if lora_data else None
        if not isinstance(lora_dice, dict):
            missing_loras.append(name)
            continue
        for bucket, items in lora_dice.items():
            if not isinstance(items, list):
                continue
            if bucket not in bucket_replaced:
                dice[bucket] = []
                bucket_seen[bucket] = set()
                bucket_replaced.add(bucket)
                source[bucket] = []
            for item in items:
                key = item.lower() if isinstance(item, str) else item
                if key in bucket_seen[bucket]:
                    continue
                bucket_seen[bucket].add(key)
                dice[bucket].append(item)
            if f"{name}.json" not in source[bucket]:
                source[bucket].append(f"{name}.json")

    if missing_loras:
        source["_note"] = f"no prompt file for: {', '.join(missing_loras)}"

    return jsonify(
        {
            "success": True,
            "loras": selected_loras,
            "missing_loras": missing_loras,
            "available_loras": available_loras,
            "prompts": {
                "version": data.get("version", 1),
                "dice": dice,
                "source": source,
            },
        }
    )


@app.route("/generate", methods=["POST"])
def generate():
    cleanup_old_sessions()
    data = get_json_body()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400
    errors = validate_common(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    params = common_params(data)
    session_id = create_session(
        {
            "mode": "generate",
            "prompt": params["prompt"],
            "steps": params["steps"],
            "duration": params["duration"],
        }
    )
    threading.Thread(target=generation_worker, args=(session_id, params), daemon=True).start()
    return jsonify(
        {
            "success": True,
            "session_id": session_id,
            "seed": params["seed"],
            "prompt": params["prompt"],
            "duration": params["duration"],
        }
    )


@app.route("/generate/loop", methods=["POST"])
def generate_loop():
    cleanup_old_sessions()
    data = get_json_body()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400
    errors = validate_common(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    prompt = data["prompt"].strip()
    bpm = data.get("bpm")
    bpm = float(bpm) if bpm not in (None, "") else extract_bpm(prompt)
    if not bpm or bpm <= 0:
        return jsonify({"success": False, "error": "BPM required in prompt or bpm field"}), 400

    bars = parse_int(data, "bars", DEFAULT_LOOP_BARS)
    if bars not in VALID_LOOP_BARS:
        return jsonify({"success": False, "error": f"bars must be one of {sorted(VALID_LOOP_BARS)}"}), 400

    seconds_per_bar = (60.0 / bpm) * 4.0
    loop_duration = seconds_per_bar * bars
    gen_duration = loop_duration + LOOP_PAD_SECONDS
    if gen_duration > MAX_DURATION:
        return jsonify(
            {
                "success": False,
                "error": f"{bars} bars at {bpm} bpm exceeds max {MAX_DURATION}s with pad",
            }
        ), 400

    target_samples = round(loop_duration * OUTPUT_SAMPLE_RATE)
    params = common_params(data, duration=gen_duration)
    params["mode"] = "loop"
    params["target_samples"] = target_samples
    params["loop"] = {
        "bpm": bpm,
        "bars": bars,
        "seconds_per_bar": round(seconds_per_bar, 6),
        "loop_duration": round(loop_duration, 6),
        "gen_duration": round(gen_duration, 6),
        "target_samples": target_samples,
    }

    session_id = create_session({"mode": "loop", "prompt": params["prompt"], "steps": params["steps"], "duration": gen_duration})
    threading.Thread(target=generation_worker, args=(session_id, params), daemon=True).start()
    return jsonify({"success": True, "session_id": session_id, "seed": params["seed"], **params["loop"]})


@app.route("/transform", methods=["POST"])
def transform():
    cleanup_old_sessions()
    data = get_json_body()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400
    errors = validate_common(data, require_duration=False)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400
    try:
        input_sr, waveform = decode_audio_data(data)
    except Exception as exc:
        return jsonify({"success": False, "error": f"could not decode audio_data: {exc}"}), 400

    input_duration = waveform.shape[-1] / float(input_sr)
    if input_duration <= 0 or input_duration > MAX_DURATION:
        return jsonify({"success": False, "error": f"input duration must be in (0, {MAX_DURATION}] seconds"}), 400

    strength = max(0.01, min(1.0, parse_float(data, "strength", 0.9)))
    target_samples = round(input_duration * OUTPUT_SAMPLE_RATE)
    params = common_params(data, duration=input_duration + 0.5)
    params["mode"] = "transform"
    params["target_samples"] = target_samples
    params["init_audio"] = (input_sr, waveform)
    params["init_noise_level"] = strength
    params["transform"] = {
        "strength": strength,
        "input_duration": round(input_duration, 6),
        "input_sr": input_sr,
        "input_channels": int(waveform.shape[0]),
        "target_samples": target_samples,
    }

    session_id = create_session({"mode": "transform", "prompt": params["prompt"], "steps": params["steps"], "duration": input_duration})
    threading.Thread(target=generation_worker, args=(session_id, params), daemon=True).start()
    return jsonify({"success": True, "session_id": session_id, "seed": params["seed"], **params["transform"]})


@app.route("/continue", methods=["POST"])
def continue_audio():
    cleanup_old_sessions()
    data = get_json_body()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400
    errors = validate_common(data, require_duration=False)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    mode = (data.get("continuation_mode") or "inpaint").lower()
    if mode not in VALID_CONTINUATION_MODES:
        return jsonify(
            {
                "success": False,
                "error": f"continuation_mode must be one of {sorted(VALID_CONTINUATION_MODES)}",
            }
        ), 400

    try:
        input_sr, waveform = decode_audio_data(data)
    except Exception as exc:
        return jsonify({"success": False, "error": f"could not decode audio_data: {exc}"}), 400

    source_duration = waveform.shape[-1] / float(input_sr)
    continuation_seconds = parse_float(data, "continuation_seconds", DEFAULT_CONTINUATION_SECONDS)
    tail_pad = min(CONTINUE_TAIL_PAD_MAX, max(0.0, parse_float(data, "continuation_tail_pad", CONTINUE_TAIL_PAD)))
    total_duration = source_duration + continuation_seconds
    if source_duration <= 0 or continuation_seconds <= 0 or total_duration > MAX_DURATION:
        return jsonify({"success": False, "error": f"source + continuation must be in (0, {MAX_DURATION}] seconds"}), 400

    if CONTINUE_TAIL_MODE == "exact":
        gen_duration = total_duration
        mask_end = total_duration
    elif CONTINUE_TAIL_MODE == "regen_past":
        gen_duration = total_duration + tail_pad
        mask_end = gen_duration
    else:
        gen_duration = total_duration + 0.5
        mask_end = total_duration

    target_samples = round(total_duration * OUTPUT_SAMPLE_RATE)
    params = common_params(data, duration=gen_duration)
    requested_sampler_type = None
    if mode == "latent_prefix" and params["sampler_type"] != "pingpong":
        requested_sampler_type = params["sampler_type"]
        params["sampler_type"] = "pingpong"
    params["mode"] = "continue"
    params["target_samples"] = target_samples
    params["inpaint_audio"] = (input_sr, waveform)
    params["inpaint_mask_start_seconds"] = source_duration
    params["inpaint_mask_end_seconds"] = mask_end
    params["continue"] = {
        "mode": mode,
        "source_duration": source_duration,
        "continuation_seconds": round(continuation_seconds, 6),
        "total_duration": round(total_duration, 6),
        "tail_mode": CONTINUE_TAIL_MODE,
        "tail_pad": round(tail_pad, 6),
        "gen_duration": round(gen_duration, 6),
        "mask_start_seconds": round(source_duration, 6),
        "mask_end_seconds": round(mask_end, 6),
        "sampler_type": params["sampler_type"],
        "requested_sampler_type": requested_sampler_type,
        "input_sr": input_sr,
        "input_channels": int(waveform.shape[0]),
        "target_samples": target_samples,
    }

    session_id = create_session({"mode": "continue", "prompt": params["prompt"], "steps": params["steps"], "duration": total_duration})
    threading.Thread(target=generation_worker, args=(session_id, params), daemon=True).start()
    return jsonify({"success": True, "session_id": session_id, "seed": params["seed"], **params["continue"]})


@app.route("/poll_status/<session_id>", methods=["GET"])
def poll_status(session_id: str):
    session = get_session(session_id)
    if session is None:
        return jsonify({"success": False, "error": f"unknown session: {session_id}"}), 404

    status = session["status"]
    queue_status: dict[str, Any] = {}
    if status == "queued":
        queue_status = {
            "status": "queued",
            "position": 1,
            "total_queued": 1,
            "message": "Task queued locally.",
            "estimated_seconds": 5,
        }
    elif status in ("generating", "encoding"):
        queue_status = {"status": "ready"}

    response = {
        "success": status != "failed",
        "generation_in_progress": session["generation_in_progress"],
        "transform_in_progress": session["transform_in_progress"],
        "progress": session["progress"],
        "step": session.get("step", 0),
        "total_steps": session.get("total_steps", 0),
        "status": status,
        "phase": session.get("phase", status),
        "queue_status": queue_status,
    }
    if status == "completed":
        response["audio_data"] = session.get("audio_data", "")
        response["meta"] = session.get("meta", {})
    if status == "failed":
        response["error"] = session.get("error", "unknown error")
        response["gate_links"] = SA3_MODEL_LINKS
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8006"))
    ensure_default_prompt_pool()
    app.run(host="0.0.0.0", port=port, threaded=True)
