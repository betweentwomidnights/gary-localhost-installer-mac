from __future__ import annotations

import base64
import io
import json
import os
import threading
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
        try:
            model = MusicGenContinuation.from_pretrained(
                model_name,
                base_model=base_model,
                prefer_base_config=prefer_base_config,
                download_progress_callback=download_progress_callback,
            )
        except FileNotFoundError:
            base_model = get_base_model_for_finetune(model_name)
            model = MusicGenContinuation.from_pretrained(
                model_name,
                base_model=base_model,
                prefer_base_config=True,
                download_progress_callback=download_progress_callback,
            )
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
        tqdm_class=tqdm_class,
    )
    return Path(snapshot_path)


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

    def _stage_progress(evt: dict) -> None:
        nonlocal last_stage_percent, last_downloaded_bytes, heuristic_bytes_accumulator

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
    model_snapshot_path = _snapshot_repo(
        model_name,
        _MODEL_SNAPSHOT_PATTERNS,
        local_files_only=False,
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
