#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from stable_audio_3.mlx.pipeline import StableAudioMLXPipeline  # noqa: E402
from stable_audio_3.mlx.sampling import (  # noqa: E402
    distribution_shift_spec_to_jsonable,
    make_distribution_shift_spec,
    shift_timestep_values,
    training_distribution_shift_spec_from_model_config,
)
from mlx_lora_dataset import (  # noqa: E402
    LoraDatasetExample,
    compose_trigger_prompt,
    discover_dataset_examples,
)
from mlx_lora_loudness import normalize_audio_to_latent_rms  # noqa: E402
from mlx_training_assets import (  # noqa: E402
    OPTIMIZED_MEDIUM_BASE_FILENAME,
    OPTIMIZED_MODEL_REPO,
    TRAINING_MODEL_NAME,
    TRAINING_MODEL_REPO,
    resolve_hosted_medium_base_npz,
    resolve_medium_base_assets,
    validate_medium_base_assets,
)
from stable_audio_3.mlx.dit import StableAudioMLXDiT  # noqa: E402
from stable_audio_3.mlx.dit_medium_official import (  # noqa: E402
    load_official_medium_dit,
)
from stable_audio_3.mlx.training import (  # noqa: E402
    LORA_LAYER_SCOPE_CHOICES,
    LORA_LAYER_SCOPE_DEFAULT,
    inject_trainable_lora,
    iter_trainable_lora_layers,
    layer_scope_exclusions,
    rectified_flow_loss,
    sample_training_timesteps,
    save_trainable_lora,
)


DEFAULT_PROMPT = (
    "garysmoke, warm experimental electronic music, textured synths, rhythmic, stereo"
)
ENCODE_CHUNK_THRESHOLD_SECONDS = 30.0
ENCODE_CHUNK_SIZE = 128
ENCODE_CHUNK_OVERLAP = 32
FULL_TRACK_CROP_SECONDS = 285.35
FULL_TRACK_LATENT_FRAMES = 3072
FULL_TRACK_BUCKET_FRAMES = 512
FULL_TRACK_SILENCE_RESERVE_SECONDS = 4.0
SA3_ADAMW_BETAS = (0.9, 0.95)
SA3_ADAMW_EPS = 1e-8
SA3_ADAMW_WEIGHT_DECAY = 0.01
SECONDS_CONDITIONER_CHECKPOINT_NAME = (
    "conditioners.seconds_total.embedder.embedding.1"
)
DIT_ENGINE_GARY = "gary-generic"
DIT_ENGINE_OFFICIAL = "official-specialized"
DIT_ENGINE_CHOICES = (DIT_ENGINE_GARY, DIT_ENGINE_OFFICIAL)


class _SA3TrainingBundle(nn.Module):
    """Keep the DiT and trainable seconds conditioner in one grad tree."""

    def __init__(self, dit, seconds_conditioner, *, dit_engine: str):
        super().__init__()
        self.dit = dit
        self.seconds_conditioner = seconds_conditioner
        self.dit_engine = dit_engine

    def __call__(
        self,
        x,
        t,
        *,
        prompt_cond,
        seconds_total,
        local_add_cond,
        padding_mask,
        cfg_dropout_prob,
    ):
        seconds_token, _ = self.seconds_conditioner(seconds_total)
        cross_attn_cond = mx.concatenate(
            [
                prompt_cond.astype(mx.float32),
                seconds_token.astype(mx.float32),
            ],
            axis=1,
        )
        if cfg_dropout_prob > 0:
            keep = (
                mx.random.uniform(shape=(cross_attn_cond.shape[0], 1, 1))
                < (1.0 - float(cfg_dropout_prob))
            )
            cross_attn_cond = mx.where(
                keep,
                cross_attn_cond,
                mx.zeros_like(cross_attn_cond),
            )
        global_embed = seconds_token[:, 0, :]
        if self.dit_engine == DIT_ENGINE_OFFICIAL:
            return self.dit(
                x,
                t.astype(x.dtype),
                cross_attn_cond.astype(x.dtype),
                global_embed.astype(x.dtype),
                local_add_cond=local_add_cond.transpose(0, 2, 1),
            )
        return self.dit(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            global_embed=global_embed,
            local_add_cond=local_add_cond,
            padding_mask=padding_mask,
            cfg_scale=1.0,
            cfg_dropout_prob=0.0,
        )


def _create_sa3_adamw(learning_rate: float):
    """Match the official SA3 PyTorch LoRA optimizer, including bias correction."""

    return optim.AdamW(
        learning_rate=learning_rate,
        betas=list(SA3_ADAMW_BETAS),
        eps=SA3_ADAMW_EPS,
        weight_decay=SA3_ADAMW_WEIGHT_DECAY,
        bias_correction=True,
    )


def _initialize_adapter_factors_by_name(model, *, seed: int) -> None:
    """Make adapter initialization independent of module traversal order."""

    for layer in iter_trainable_lora_layers(model):
        lora_a = getattr(layer, "lora_A", None)
        if lora_a is None:
            continue
        digest = hashlib.sha256(
            f"{int(seed)}:{layer.source_name}".encode("utf-8")
        ).digest()
        layer_seed = int.from_bytes(digest[:4], "little")
        fan_in = int(lora_a.shape[1])
        init_scale = 1.0 / math.sqrt(fan_in)
        layer.lora_A = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=lora_a.shape,
            dtype=mx.float32,
            key=mx.random.key(layer_seed),
        )


def _enable_gradient_checkpointing(layer) -> None:
    """Recompute every block of this class during backward to save memory."""

    layer_type = type(layer)
    if getattr(layer_type, "_gary_gradient_checkpointed", False):
        return
    original_call = layer_type.__call__

    def checkpointed_call(model, *args, **kwargs):
        def inner_call(params, *inner_args, **inner_kwargs):
            model.update(params)
            return original_call(model, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner_call)(
            model.trainable_parameters(),
            *args,
            **kwargs,
        )

    layer_type.__call__ = checkpointed_call
    layer_type._gary_gradient_checkpointed = True


def _configure_mlx_wired_memory_limit() -> int | None:
    """Use Metal's recommended pinned working set to avoid paging stalls."""

    if not hasattr(mx, "set_wired_limit") or not hasattr(mx, "device_info"):
        return None
    recommended = mx.device_info().get("max_recommended_working_set_size")
    if not recommended:
        return None
    wired_limit = int(recommended)
    mx.set_wired_limit(wired_limit)
    return wired_limit


def _mlx_memory_snapshot() -> dict[str, float]:
    snapshot = {}
    for label, getter_name in (
        ("active_gib", "get_active_memory"),
        ("cache_gib", "get_cache_memory"),
        ("peak_gib", "get_peak_memory"),
    ):
        getter = getattr(mx, getter_name, None)
        if getter is not None:
            snapshot[label] = float(getter()) / (1024**3)
    return snapshot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLX Stable Audio 3 LoRA on an audio file or folder."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-path", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument(
        "--model-name",
        choices=(TRAINING_MODEL_NAME,),
        default=TRAINING_MODEL_NAME,
        help=(
            "Base model used for training. Gary trains on medium-base and "
            "applies the adapter to medium at inference."
        ),
    )
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument(
        "--dit-engine",
        choices=DIT_ENGINE_CHOICES,
        default=DIT_ENGINE_GARY,
        help=(
            "Training DiT implementation. Both choices load the same official "
            "hosted medium-base FP16 NPZ; the specialized engine is an "
            "experimental A/B path for fixed random crops."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path.home()
            / "Library"
            / "Application Support"
            / "GaryLocalhost"
            / "training"
            / "mlx-one-file-smoke"
        ),
    )
    parser.add_argument(
        "--latent-cache-dir",
        type=Path,
        help=(
            "Persistent content-addressed latent cache. Defaults to Gary's "
            "shared SA3 training cache so repeated datasets are not re-encoded."
        ),
    )
    parser.add_argument("--prompt", default="")
    parser.add_argument("--trigger-text", default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--crop-seconds", type=float, default=47.0)
    parser.add_argument(
        "--full-tracks",
        action="store_true",
        help=(
            "Train the first 285.35 seconds from 0:00, padding shorter tracks "
            "with encoded silence. Without this flag, choose a random crop "
            "offset on every step."
        ),
    )
    parser.add_argument(
        "--full-track-buckets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For full-track training, round each song to a compiled 512-frame "
            "bucket instead of computing all 3072 frames for short tracks."
        ),
    )
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float)
    parser.add_argument(
        "--adapter-type",
        choices=(
            "lora",
            "dora",
            "dora-rows",
            "bora",
            "lora-xs",
            "dora-rows-xs",
            "dora-cols-xs",
            "bora-xs",
        ),
        default="dora",
        help=(
            "Train LoRA, DoRA, BoRA, or their extra-small SVD variants. "
            "Defaults to DoRA for Gary parity."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.1)
    parser.add_argument(
        "--timestep-sampler",
        choices=(
            "uniform",
            "logit_normal",
            "trunc_logit_normal",
            "log_snr",
            "log_snr_uniform",
        ),
        default="trunc_logit_normal",
    )
    parser.add_argument(
        "--distribution-shift",
        choices=("model", "none", "full"),
        default="model",
        help=(
            "Training-time timestep shift. 'model' uses distribution_shift_options "
            "from the Stable Audio 3 model config."
        ),
    )
    parser.add_argument("--shift-base", type=float, default=0.5)
    parser.add_argument("--shift-max", type=float, default=1.15)
    parser.add_argument("--shift-min-length", type=int, default=256)
    parser.add_argument("--shift-max-length", type=int, default=4096)
    parser.add_argument("--shift-use-sine", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compile the fixed-shape MLX loss/backward/optimizer graph.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recompute transformer-block activations during backward. Off by "
            "default. On this hardware it costs roughly 45%% step time while "
            "reclaiming very little memory (2.0 GiB of 30.2 GiB at 2048 "
            "latents), so it is not a useful trade at any measured window. "
            "Kept as an escape hatch pending a fix to activation retention."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--dit-dtype", default="float16")
    parser.add_argument("--autoencoder-dtype", default="float16")
    parser.add_argument(
        "--per-track-target-latent-rms",
        type=float,
        default=0.0,
        help=(
            "If greater than zero, iteratively re-encode each track to approach "
            "this latent RMS. The base-model target is 0.90."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        help=(
            "Layer-name substring to adapt. Repeat for multiple filters. "
            "Defaults to every eligible DiT Linear/Conv1d layer plus the "
            "seconds_total conditioner Linear layer."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        help=(
            "Layer-name substring to exclude. Repeat for multiple filters. "
            "No layers are excluded by default."
        ),
    )
    parser.add_argument(
        "--layer-scope",
        choices=LORA_LAYER_SCOPE_CHOICES,
        default=LORA_LAYER_SCOPE_DEFAULT,
        help=(
            "Which DiT projections receive adapters. 'all-projections' adapts "
            "every eligible Linear/Conv1d layer (228 on medium-base). "
            "'attention-feedforward' applies the official standalone trainer's "
            "product-default exclusions and adapts 168. Any explicit --exclude "
            "filters are applied on top of the scope."
        ),
    )
    return parser.parse_args()


def _resolve_grad_checkpoint(requested: bool | None) -> bool:
    """Gradient checkpointing is off unless explicitly requested.

    It used to default on for full-track training. Benchmarking on a base M4
    Air showed it is not a useful trade at any measured window: at 256 latents
    it costs ~45% step time and reclaims 0.05 GiB, and at 2,048 latents it
    still only reclaims 2.0 GiB of a 30.2 GiB peak. Until activation retention
    is fixed, the flag stays available but off by default.
    """

    return False if requested is None else bool(requested)


def _resolve_lora_filters(
    include: list[str] | None,
    exclude: list[str] | None,
    layer_scope: str = LORA_LAYER_SCOPE_DEFAULT,
) -> tuple[list[str] | None, list[str]]:
    resolved_include = [value for value in (include or []) if value.strip()] or None
    resolved_exclude = [value for value in (exclude or []) if value.strip()]
    for pattern in layer_scope_exclusions(layer_scope):
        if pattern not in resolved_exclude:
            resolved_exclude.append(pattern)
    return resolved_include, resolved_exclude


def _filter_selects_seconds_conditioner(
    include: list[str] | None,
    exclude: list[str],
) -> bool:
    names = (
        "proj",
        "embedder.embedding.1",
        SECONDS_CONDITIONER_CHECKPOINT_NAME,
    )
    if include and not any(
        pattern in name for pattern in include for name in names
    ):
        return False
    return not any(
        pattern in name for pattern in exclude for name in names
    )


def _conditioning_seconds_for_example(
    *,
    source_duration_seconds: float,
    aligned_crop_seconds: float,
    full_tracks: bool,
) -> float:
    if full_tracks:
        return min(source_duration_seconds, aligned_crop_seconds)
    # The official pre-encoded dataset preserves the full source duration
    # after choosing a random crop. The seconds token describes the song,
    # not merely the visible training window.
    return source_duration_seconds


def _effective_sequence_length(
    *,
    conditioning_seconds: float,
    crop_latents: int,
    sample_rate: int,
    downsampling_ratio: int,
    use_effective_length: bool,
) -> int:
    if not use_effective_length:
        return int(crop_latents)
    return int(
        math.ceil(
            int(float(conditioning_seconds) * sample_rate)
            / downsampling_ratio
        )
    )


def _resolve_training_distribution_shift(args, model_config):
    if args.distribution_shift == "none":
        return None
    if args.distribution_shift == "model":
        return training_distribution_shift_spec_from_model_config(model_config)
    return make_distribution_shift_spec(
        "full",
        base_shift=args.shift_base,
        max_shift=args.shift_max,
        min_length=args.shift_min_length,
        max_length=args.shift_max_length,
        use_sine=args.shift_use_sine,
    )


def _load_audio(path: Path, *, sample_rate: int, channels: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.T
    if int(source_rate) != int(sample_rate):
        import torch
        import torchaudio.functional as audio_f

        audio = audio_f.resample(
            torch.from_numpy(audio),
            int(source_rate),
            int(sample_rate),
        ).numpy()

    if audio.shape[0] == 1 and channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > channels:
        audio = audio[:channels]
    elif audio.shape[0] != channels:
        raise ValueError(
            f"Cannot adapt {audio.shape[0]} input channels to {channels} target channels."
        )
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def _encode_or_load_latents(
    pipeline: StableAudioMLXPipeline,
    *,
    audio_path: Path,
    output_dir: Path,
    sample_rate: int,
    channels: int,
    target_latent_rms: float,
    latent_rms_correction_rounds: int = 4,
    latent_rms_tolerance: float = 0.03,
) -> tuple[Path, dict[str, object], bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_path = output_dir / "source_latents.npy"
    metadata_path = output_dir / "source_latents.json"
    source_stat = audio_path.stat()
    loudness_config = {
        "enabled": target_latent_rms > 0,
        "target_rms": target_latent_rms if target_latent_rms > 0 else None,
        "max_correction_rounds": latent_rms_correction_rounds,
        "tolerance": latent_rms_tolerance,
    }
    encoding_config = {
        "autoencoder_dtype": pipeline.autoencoder_dtype_name,
        "chunk_threshold_seconds": ENCODE_CHUNK_THRESHOLD_SECONDS,
        "chunk_size": ENCODE_CHUNK_SIZE,
        "chunk_overlap": ENCODE_CHUNK_OVERLAP,
    }
    if latent_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("audio_path") == str(audio_path)
            and int(metadata.get("sample_rate", 0)) == sample_rate
            and int(metadata.get("source_size", -1)) == source_stat.st_size
            and int(metadata.get("source_modified_ns", -1))
            == source_stat.st_mtime_ns
            and metadata.get("encoding_config") == encoding_config
            and all(
                metadata.get("loudness_fix", {}).get(key) == value
                for key, value in loudness_config.items()
            )
        ):
            return latent_path, metadata, True

    audio = _load_audio(audio_path, sample_rate=sample_rate, channels=channels)
    source_samples = int(audio.shape[-1])
    alignment = int(pipeline.autoencoder.downsampling_ratio) * 16
    aligned_samples = int(math.ceil(source_samples / alignment) * alignment)
    if aligned_samples != source_samples:
        audio = np.pad(audio, ((0, 0), (0, aligned_samples - source_samples)))

    def encode(value: np.ndarray) -> np.ndarray:
        chunked = value.shape[-1] > int(
            ENCODE_CHUNK_THRESHOLD_SECONDS * sample_rate
        )
        encoded = pipeline.autoencoder.encode_audio(
            mx.array(value[None]).astype(
                getattr(mx, pipeline.autoencoder_dtype_name)
            ),
            chunked=chunked,
            chunk_size=ENCODE_CHUNK_SIZE,
            overlap=ENCODE_CHUNK_OVERLAP,
        )
        latents = encoded.astype(getattr(mx, pipeline.dtype_name))
        mx.eval(latents)
        return np.asarray(latents, dtype=np.float32)

    if target_latent_rms > 0:
        normalization = normalize_audio_to_latent_rms(
            audio,
            encode=encode,
            actual_samples=source_samples,
            target_latent_rms=target_latent_rms,
            max_correction_rounds=latent_rms_correction_rounds,
            tolerance=latent_rms_tolerance,
        )
        latent_values = normalization.latents
        loudness_result = {
            **loudness_config,
            **{
                "gain": normalization.gain,
                "pre_normalization_rms": normalization.pre_normalization_rms,
                "achieved_rms": normalization.achieved_rms,
                "passes": normalization.passes,
                "converged": normalization.converged,
            },
        }
    else:
        latent_values = encode(audio)
        loudness_result = loudness_config

    np.save(latent_path, latent_values.astype(np.float16))
    metadata = {
        "audio_path": str(audio_path),
        "sample_rate": sample_rate,
        "channels": channels,
        "source_size": source_stat.st_size,
        "source_modified_ns": source_stat.st_mtime_ns,
        "encoding_config": encoding_config,
        "chunked_encoding": source_samples
        > int(ENCODE_CHUNK_THRESHOLD_SECONDS * sample_rate),
        "source_samples": source_samples,
        "aligned_samples": aligned_samples,
        "latent_shape": [int(value) for value in latent_values.shape],
        "downsampling_ratio": int(pipeline.autoencoder.downsampling_ratio),
        "loudness_fix": loudness_result,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return latent_path, metadata, False


def _resolve_training_window(
    *,
    full_tracks: bool,
    crop_seconds: float,
    sample_rate: int,
    downsampling_ratio: int,
) -> tuple[int, float]:
    if full_tracks:
        crop_latents = FULL_TRACK_LATENT_FRAMES
    else:
        crop_latents = int(
            math.ceil(crop_seconds * sample_rate / downsampling_ratio)
        )
        crop_latents = int(math.ceil(crop_latents / 16) * 16)
    aligned_seconds = crop_latents * downsampling_ratio / sample_rate
    return crop_latents, aligned_seconds


def _full_track_bucket_latents(
    *,
    valid_frames: int,
    maximum_frames: int,
    sample_rate: int,
    downsampling_ratio: int,
    bucket_frames: int = FULL_TRACK_BUCKET_FRAMES,
    silence_reserve_seconds: float = FULL_TRACK_SILENCE_RESERVE_SECONDS,
) -> int:
    """Choose a bounded static shape while retaining a short silence tail."""

    if maximum_frames <= 0 or bucket_frames <= 0:
        raise ValueError("maximum_frames and bucket_frames must be positive.")
    valid_frames = min(maximum_frames, max(0, int(valid_frames)))
    silence_reserve_frames = int(
        math.ceil(
            max(0.0, float(silence_reserve_seconds))
            * sample_rate
            / downsampling_ratio
        )
    )
    required_frames = min(
        maximum_frames,
        valid_frames + silence_reserve_frames,
    )
    bucket = max(
        bucket_frames,
        int(math.ceil(required_frames / bucket_frames) * bucket_frames),
    )
    return min(maximum_frames, bucket)


def _encode_silence_latents(
    pipeline: StableAudioMLXPipeline,
    *,
    sample_rate: int,
    channels: int,
    required_latent_frames: int,
) -> np.ndarray:
    downsampling_ratio = int(pipeline.autoencoder.downsampling_ratio)
    thirty_second_frames = int(
        round(ENCODE_CHUNK_THRESHOLD_SECONDS * sample_rate / downsampling_ratio)
    )
    silence_frames = min(
        required_latent_frames,
        max(16, int(thirty_second_frames // 16 * 16)),
    )
    silence_samples = silence_frames * downsampling_ratio
    silence = mx.zeros(
        (1, channels, silence_samples),
        dtype=getattr(mx, pipeline.autoencoder_dtype_name),
    )
    encoded = pipeline.autoencoder.encode_audio(silence, chunked=False)
    encoded = encoded.astype(getattr(mx, pipeline.dtype_name))
    mx.eval(encoded)
    values = np.asarray(encoded, dtype=np.float16)
    if values.shape[-1] <= 0:
        raise RuntimeError("The autoencoder returned an empty silence latent.")
    return values


def _crop_or_pad_latents(
    latents: np.ndarray,
    *,
    crop_latents: int,
    offset: int,
    valid_frames: int | None = None,
    padding_latents: np.ndarray | None = None,
) -> np.ndarray:
    source = np.asarray(latents)
    if source.ndim != 3:
        raise ValueError(f"Expected [batch, channels, frames] latents, got {source.shape}.")
    if crop_latents <= 0:
        raise ValueError("crop_latents must be positive.")
    source_valid_frames = (
        source.shape[-1]
        if valid_frames is None
        else min(source.shape[-1], max(0, int(valid_frames)))
    )
    if offset < 0 or offset >= max(1, source_valid_frames):
        raise ValueError(
            f"Invalid latent crop offset {offset} for {source_valid_frames} valid frames."
        )

    cropped = source[..., offset : min(offset + crop_latents, source_valid_frames)]
    if cropped.shape[-1] == crop_latents:
        return np.asarray(cropped, dtype=np.float16)

    padded = np.zeros((*source.shape[:-1], crop_latents), dtype=np.float16)
    padded[..., : cropped.shape[-1]] = cropped
    pad_needed = crop_latents - cropped.shape[-1]
    if padding_latents is not None and pad_needed > 0:
        silence = np.asarray(padding_latents, dtype=np.float16)
        if (
            silence.ndim != 3
            or silence.shape[0] not in (1, source.shape[0])
            or silence.shape[1] != source.shape[1]
            or silence.shape[-1] <= 0
        ):
            raise ValueError(
                "padding_latents must have shape [1 or batch, channels, frames] "
                f"compatible with {source.shape}, got {silence.shape}."
            )
        if silence.shape[0] == 1 and source.shape[0] != 1:
            silence = np.repeat(silence, source.shape[0], axis=0)
        repeats = int(math.ceil(pad_needed / silence.shape[-1]))
        silence_pad = np.tile(silence, (1, 1, repeats))[..., :pad_needed]
        padded[..., cropped.shape[-1] :] = silence_pad
    return padded


def _audio_content_digest(audio_path: Path) -> str:
    digest = hashlib.sha256()
    with audio_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_directory(cache_root: Path, example: LoraDatasetExample) -> Path:
    # Content-only keys allow renamed, moved, or duplicated tracks to reuse the
    # same encoding. Encoding and loudness settings remain validated by the
    # metadata inside the directory.
    return cache_root / _audio_content_digest(example.audio_path)


def _default_latent_cache_directory() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "GaryLocalhost"
        / "sa3"
        / "training"
        / "latent-cache"
    )


def _augment_padding_mask(
    *,
    valid_frames: int,
    crop_latents: int,
    sample_rate: int,
    downsampling_ratio: int,
    rng: np.random.Generator,
    silence_extension_scale_seconds: float = 4.0,
) -> np.ndarray:
    valid_frames = min(crop_latents, max(0, int(valid_frames)))
    augmented_frames = valid_frames
    if valid_frames < crop_latents and silence_extension_scale_seconds > 0:
        scale_tokens = (
            silence_extension_scale_seconds
            * sample_rate
            / downsampling_ratio
        )
        augmented_frames = min(
            crop_latents,
            valid_frames + int(rng.exponential(scale_tokens)),
        )
    mask = np.zeros((1, crop_latents), dtype=np.bool_)
    mask[:, :augmented_frames] = True
    return mask


def _sample_inpaint_mask(
    padding_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    max_mask_segments: int = 10,
) -> tuple[np.ndarray, str]:
    """Port the official PyTorch 10/80/10 inpainting-mask policy.

    Mask semantics are 0=generate and 1=provided context. The three modes are
    random segments, full generation, and causal continuation.
    """

    values = np.asarray(padding_mask, dtype=np.bool_)
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError(
            f"Expected a [1, frames] padding mask, got {values.shape}."
        )
    sequence_length = int(values.shape[1])
    real_sequence_length = int(values.sum())
    choice = int(rng.choice(3, p=(0.1, 0.8, 0.1)))
    mask = np.ones((1, 1, sequence_length), dtype=np.float32)

    if choice == 1:
        return np.zeros_like(mask), "full"

    if real_sequence_length > 0 and choice == 0:
        num_segments = int(rng.integers(1, max_mask_segments + 1))
        max_length = max(1, real_sequence_length // num_segments)
        for _ in range(num_segments):
            segment_length = int(rng.integers(1, max_length + 1))
            mask_start = int(
                rng.integers(0, real_sequence_length - segment_length + 1)
            )
            mask[
                :,
                :,
                mask_start : mask_start + segment_length,
            ] = 0
        mode = "random_segments"
    else:
        if real_sequence_length > 0:
            unmasked_prefix = int(
                rng.integers(0, real_sequence_length + 1)
            )
            mask[:, :, unmasked_prefix:real_sequence_length] = 0
        mode = "causal"

    mask[:, :, real_sequence_length:] = 0
    return mask, mode


def _training_examples(args: argparse.Namespace) -> tuple[Path | None, list[LoraDatasetExample]]:
    if args.dataset_dir is not None:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        examples = discover_dataset_examples(
            dataset_dir,
            trigger_text=args.trigger_text,
        )
        return dataset_dir, examples

    audio_path = args.audio_path.expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    prompt = compose_trigger_prompt(args.trigger_text, args.prompt or DEFAULT_PROMPT)
    return None, [
        LoraDatasetExample(
            audio_path=audio_path,
            relative_path=audio_path.name,
            sidecar_path=None,
            sidecar_kind=None,
            source_prompt=args.prompt or DEFAULT_PROMPT,
            prompt=prompt,
        )
    ]


def _log_prompt_policy(
    trigger_text: str,
    examples: list[LoraDatasetExample],
) -> None:
    trigger = trigger_text.strip()
    if trigger:
        print(
            f"[prompts] shared trigger {json.dumps(trigger, ensure_ascii=False)} "
            "is prepended to every caption at training time; sidecars and dice "
            "prompts remain unchanged",
            flush=True,
        )
    else:
        print(
            "[prompts] no shared trigger set; captions are used as-is",
            flush=True,
        )

    sample = next(
        (
            " ".join(example.prompt.split())
            for example in examples
            if example.prompt.strip()
        ),
        "",
    )
    if sample:
        if len(sample) > 300:
            sample = sample[:297].rstrip() + "..."
        print(
            f"[prompts] example conditioning="
            f"{json.dumps(sample, ensure_ascii=False)}",
            flush=True,
        )


def _save_checkpoint(
    model,
    *,
    output_dir: Path,
    step: int,
    rank: int,
    alpha: float,
    include: list[str] | None,
    exclude: list[str],
    args: argparse.Namespace,
    final: bool = False,
) -> Path:
    suffix = "final" if final else f"step-{step:06d}"
    return save_trainable_lora(
        model,
        output_dir / f"gary-mlx-lora-{suffix}.safetensors",
        rank=rank,
        alpha=alpha,
        include=include,
        exclude=exclude,
        adapter_type=args.adapter_type,
        extra_metadata={
            "step": step,
            "base_model": TRAINING_MODEL_REPO,
            "dit_engine": args.dit_engine,
            "dit_weights_repo": OPTIMIZED_MODEL_REPO,
            "dit_weights_filename": OPTIMIZED_MEDIUM_BASE_FILENAME,
            "dataset_path": str(
                args.dataset_dir.expanduser().resolve()
                if args.dataset_dir is not None
                else args.audio_path.expanduser().resolve()
            ),
            "trigger_text": args.trigger_text,
            "full_tracks": args.full_tracks,
            "full_track_buckets": args.full_track_buckets,
            "crop_seconds": args.crop_seconds,
            "learning_rate": args.learning_rate,
            "cfg_dropout_prob": args.cfg_dropout_prob,
            "timestep_sampler": args.timestep_sampler,
            "distribution_shift": args.distribution_shift,
            "compile": args.compile,
            "gradient_checkpointing": args.grad_checkpoint,
            "inpaint_mask_policy": "upstream-pytorch-0.1-0.8-0.1",
            "seed": args.seed,
        },
    )


def main() -> None:
    args = _parse_args()
    wired_memory_limit = _configure_mlx_wired_memory_limit()
    if args.full_tracks:
        args.crop_seconds = FULL_TRACK_CROP_SECONDS
    if args.dit_engine == DIT_ENGINE_OFFICIAL and args.full_tracks:
        raise ValueError(
            "The experimental official-specialized DiT currently supports "
            "fixed random crops only; disable full-track training for this A/B."
        )
    args.grad_checkpoint = _resolve_grad_checkpoint(args.grad_checkpoint)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.crop_seconds <= 0:
        raise ValueError("--crop-seconds must be positive.")
    if args.per_track_target_latent_rms < 0:
        raise ValueError("--per-track-target-latent-rms must be zero or greater.")
    if args.shift_min_length <= 0 or args.shift_max_length <= args.shift_min_length:
        raise ValueError("--shift-max-length must be greater than --shift-min-length.")

    # Adapter factors are initialized during LoRA injection. Seed before model
    # construction so repeated runs actually start from identical adapters;
    # the training PRNG is reset again immediately before the step loop.
    mx.random.seed(args.seed)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_cache_dir = (
        args.latent_cache_dir.expanduser().resolve()
        if args.latent_cache_dir is not None
        else _default_latent_cache_directory()
    )
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir, examples = _training_examples(args)
    if not examples:
        raise FileNotFoundError(
            f"No supported audio files found in {dataset_dir or args.audio_path}."
        )

    if (args.config_path is None) != (args.checkpoint_path is None):
        raise ValueError(
            "--config-path and --checkpoint-path must be supplied together."
        )
    print(f"training_base_model={args.model_name}", flush=True)
    print(f"training_base_repo={TRAINING_MODEL_REPO}", flush=True)
    print(f"dit_engine={args.dit_engine}", flush=True)
    if wired_memory_limit is not None:
        print(
            f"mlx_wired_memory_limit_bytes={wired_memory_limit} "
            f"mlx_wired_memory_limit_gib="
            f"{wired_memory_limit / (1024**3):.2f}",
            flush=True,
        )
    if args.config_path is None:
        print(
            f"resolving_training_assets={TRAINING_MODEL_REPO}",
            flush=True,
        )
        config_path, checkpoint_path = resolve_medium_base_assets()
    else:
        config_path, checkpoint_path = validate_medium_base_assets(
            args.config_path,
            args.checkpoint_path,
        )
    include, exclude = _resolve_lora_filters(
        args.include,
        args.exclude,
        args.layer_scope,
    )
    alpha = float(args.rank if args.alpha is None else args.alpha)

    print(f"config={config_path}", flush=True)
    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"dataset={dataset_dir or examples[0].audio_path}", flush=True)
    print(f"dataset_examples={len(examples)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"latent_cache_dir={latent_cache_dir}", flush=True)
    _log_prompt_policy(args.trigger_text, examples)
    print("Loading torch checkpoint and converting runtime modules to MLX...", flush=True)
    load_started = time.perf_counter()
    pipeline = StableAudioMLXPipeline.from_torch_checkpoint(
        config_path,
        checkpoint_path,
        torch_device="cpu",
        dit_dtype=args.dit_dtype,
        text_dtype=args.dit_dtype,
        number_dtype=args.dit_dtype,
        autoencoder_dtype=args.autoencoder_dtype,
        attention="sliding",
        model_half=False,
    )
    print(f"conversion_seconds={time.perf_counter() - load_started:.2f}", flush=True)

    sample_rate = int(pipeline.model_config["sample_rate"])
    channels = int(pipeline.autoencoder.io_channels)
    downsampling_ratio = int(pipeline.autoencoder.downsampling_ratio)
    pipeline.torch_pipeline = None
    gc.collect()
    print("torch_runtime_released=true", flush=True)

    crop_latents, aligned_crop_seconds = _resolve_training_window(
        full_tracks=args.full_tracks,
        crop_seconds=args.crop_seconds,
        sample_rate=sample_rate,
        downsampling_ratio=downsampling_ratio,
    )
    if args.full_tracks:
        print(
            "[training-window] mode=full-track start_seconds=0.00 "
            f"requested_seconds={FULL_TRACK_CROP_SECONDS:.2f} "
            f"aligned_seconds={aligned_crop_seconds:.3f} "
            f"latent_frames={crop_latents}",
            flush=True,
        )
    else:
        print(
            "[training-window] mode=random-crop "
            f"requested_seconds={args.crop_seconds:.2f} "
            f"aligned_seconds={aligned_crop_seconds:.3f} "
            f"latent_frames={crop_latents}",
            flush=True,
        )
    diffusion_config = pipeline.model_config.get("model", {}).get("diffusion", {})
    use_effective_length_for_schedule = bool(
        diffusion_config.get("use_effective_length_for_schedule", False)
    )
    distribution_shift = _resolve_training_distribution_shift(
        args,
        pipeline.model_config,
    )
    prepared_examples = []
    for index, example in enumerate(examples, start=1):
        progress = f"{index}/{len(examples)}"
        print(
            f"encoding_example={progress} path={example.relative_path}",
            flush=True,
        )
        encode_started = time.perf_counter()
        latent_path, latent_metadata, cache_hit = _encode_or_load_latents(
            pipeline,
            audio_path=example.audio_path,
            output_dir=_cache_directory(latent_cache_dir, example),
            sample_rate=sample_rate,
            channels=channels,
            target_latent_rms=args.per_track_target_latent_rms,
        )
        encode_seconds = time.perf_counter() - encode_started
        latent_frames = int(latent_metadata["latent_shape"][-1])
        source_duration_seconds = (
            int(latent_metadata["source_samples"]) / sample_rate
        )
        valid_latent_frames = min(
            latent_frames,
            int(
                math.ceil(
                    int(latent_metadata["source_samples"])
                    / downsampling_ratio
                )
            ),
        )
        valid_crop_frames = min(crop_latents, valid_latent_frames)
        training_latent_frames = (
            _full_track_bucket_latents(
                valid_frames=valid_crop_frames,
                maximum_frames=crop_latents,
                sample_rate=sample_rate,
                downsampling_ratio=downsampling_ratio,
            )
            if args.full_tracks and args.full_track_buckets
            else crop_latents
        )
        padding_mask_values = np.zeros(
            (1, training_latent_frames),
            dtype=np.bool_,
        )
        padding_mask_values[:, :valid_crop_frames] = True
        print(
            f"encoded_audio={progress} path={example.relative_path} "
            f"encode_elapsed_seconds={encode_seconds:.2f} "
            f"cached={str(cache_hit).lower()} "
            f"source_duration_seconds="
            f"{source_duration_seconds:.2f} "
            f"chunked={str(latent_metadata['chunked_encoding']).lower()}",
            flush=True,
        )
        if args.full_tracks:
            used_seconds = min(source_duration_seconds, aligned_crop_seconds)
            print(
                f"[training-window] path={example.relative_path} "
                "start_seconds=0.00 "
                f"used_audio_seconds={used_seconds:.2f} "
                f"encoded_silence_padding_seconds="
                f"{max(0.0, aligned_crop_seconds - source_duration_seconds):.2f} "
                f"truncated_tail_seconds="
                f"{max(0.0, source_duration_seconds - aligned_crop_seconds):.2f}",
                flush=True,
            )
            print(
                f"[training-bucket] path={example.relative_path} "
                f"valid_latent_frames={valid_crop_frames} "
                f"bucket_latent_frames={training_latent_frames} "
                f"bucket_seconds="
                f"{training_latent_frames * downsampling_ratio / sample_rate:.3f}",
                flush=True,
            )

        conditioning_started = time.perf_counter()
        conditioning_seconds = _conditioning_seconds_for_example(
            source_duration_seconds=source_duration_seconds,
            aligned_crop_seconds=aligned_crop_seconds,
            full_tracks=args.full_tracks,
        )
        effective_seq_len = _effective_sequence_length(
            conditioning_seconds=conditioning_seconds,
            crop_latents=crop_latents,
            sample_rate=sample_rate,
            downsampling_ratio=downsampling_ratio,
            use_effective_length=use_effective_length_for_schedule,
        )
        prompt_embeddings, _ = pipeline.text_conditioner([example.prompt])
        prompt_cond = mx.stop_gradient(
            prompt_embeddings.astype(mx.float32)
        )
        mx.eval(prompt_cond)
        prepared_examples.append(
            {
                "example": example,
                "latent_path": latent_path,
                "latent_frames": latent_frames,
                "valid_latent_frames": valid_latent_frames,
                "valid_crop_frames": valid_crop_frames,
                "training_latent_frames": training_latent_frames,
                "conditioning_seconds": conditioning_seconds,
                "effective_seq_len": effective_seq_len,
                "latent_metadata": latent_metadata,
                "prompt_cond": prompt_cond,
                "padding_mask_values": padding_mask_values,
            }
        )
        print(
            f"conditioned_example={progress} path={example.relative_path} "
            f"conditioning_elapsed_seconds="
            f"{time.perf_counter() - conditioning_started:.2f}",
            flush=True,
        )
        print(
            f"encoded_example={progress} path={example.relative_path} "
            f"total_elapsed_seconds={time.perf_counter() - encode_started:.2f}",
            flush=True,
        )
        loudness = latent_metadata["loudness_fix"]
        if loudness["enabled"]:
            print(
                f"loudness_fix={index}/{len(examples)} "
                f"path={example.relative_path} "
                f"target={loudness['target_rms']:.4f} "
                f"pre_rms={loudness['pre_normalization_rms']:.4f} "
                f"achieved={loudness['achieved_rms']:.4f} "
                f"gain={loudness['gain']:.4f} "
                f"passes={loudness['passes']} "
                f"converged={str(loudness['converged']).lower()}",
                flush=True,
            )

    silence_latents = None
    if any(item["valid_latent_frames"] < crop_latents for item in prepared_examples):
        silence_started = time.perf_counter()
        silence_latents = _encode_silence_latents(
            pipeline,
            sample_rate=sample_rate,
            channels=channels,
            required_latent_frames=crop_latents,
        )
        print(
            "encoded_silence_padding=true "
            f"latent_frames={silence_latents.shape[-1]} "
            f"elapsed_seconds={time.perf_counter() - silence_started:.2f}",
            flush=True,
        )

    if args.dit_engine == DIT_ENGINE_OFFICIAL:
        padded = [
            item["example"].relative_path
            for item in prepared_examples
            if item["valid_crop_frames"] < item["training_latent_frames"]
        ]
        if padded:
            raise ValueError(
                "The experimental official-specialized DiT does not consume "
                "Gary's padding mask. Use a shorter crop; padded examples: "
                + ", ".join(padded[:3])
            )

    # Dataset preparation still uses Gary's converted VAE and conditioners.
    # Replace the converted DiT before training so both A/B engines load the
    # exact same official hosted medium-base FP16 tensors.
    pipeline.mlx_dit = None
    gc.collect()
    mx.clear_cache()
    hosted_dit_path = resolve_hosted_medium_base_npz()
    hosted_load_started = time.perf_counter()
    dit_dtype = getattr(mx, args.dit_dtype)
    if args.dit_engine == DIT_ENGINE_OFFICIAL:
        model = load_official_medium_dit(
            hosted_dit_path,
            T_lat=crop_latents,
            dtype=dit_dtype,
        )
    else:
        model = StableAudioMLXDiT.from_hosted_medium_npz(
            pipeline.model_config,
            str(hosted_dit_path),
            param_dtype=dit_dtype,
        )
    print(
        f"hosted_dit={hosted_dit_path} "
        f"engine={args.dit_engine} "
        f"load_seconds={time.perf_counter() - hosted_load_started:.2f}",
        flush=True,
    )
    dit_injection = inject_trainable_lora(
        model,
        rank=args.rank,
        alpha=alpha,
        include=include,
        exclude=exclude,
        adapter_type=args.adapter_type,
    )
    if args.dit_engine == DIT_ENGINE_OFFICIAL:
        for layer in iter_trainable_lora_layers(model):
            layer.source_name = layer.source_name.replace(
                ".to_local_embed.seq.",
                ".to_local_embed.",
            )
    if args.grad_checkpoint:
        transformer_layers = model.transformer.layers
        if not transformer_layers:
            raise RuntimeError(
                "Gradient checkpointing requested, but the DiT has no "
                "transformer blocks."
            )
        _enable_gradient_checkpointing(transformer_layers[0])
        print(
            "gradient_checkpointing=true "
            f"transformer_blocks={len(transformer_layers)}",
            flush=True,
        )
    number_conditioner = pipeline.number_conditioner
    conditioner_injection = None
    if _filter_selects_seconds_conditioner(include, exclude):
        conditioner_injection = inject_trainable_lora(
            number_conditioner,
            rank=args.rank,
            alpha=alpha,
            include=["proj"],
            exclude=[],
            adapter_type=args.adapter_type,
        )
        number_conditioner.proj.source_name = (
            SECONDS_CONDITIONER_CHECKPOINT_NAME
        )
    else:
        number_conditioner.freeze()
        print(
            "seconds_conditioner_adapter=false reason=lora_filters",
            flush=True,
        )
    bundle = _SA3TrainingBundle(
        model,
        number_conditioner,
        dit_engine=args.dit_engine,
    )
    _initialize_adapter_factors_by_name(bundle, seed=args.seed)
    mx.eval(bundle.parameters())
    adapted_layer_names = [
        name.replace(".to_local_embed.seq.", ".to_local_embed.")
        for name in dit_injection.layer_names
    ]
    if conditioner_injection is not None:
        adapted_layer_names.append(SECONDS_CONDITIONER_CHECKPOINT_NAME)
    trainable_parameters = (
        dit_injection.trainable_parameters
        + (
            conditioner_injection.trainable_parameters
            if conditioner_injection is not None
            else 0
        )
    )

    pipeline.torch_pipeline = None
    pipeline.autoencoder = None
    pipeline.text_conditioner = None
    pipeline.number_conditioner = None
    gc.collect()
    mx.clear_cache()

    run_config = {
        "dataset_path": str(dataset_dir or examples[0].audio_path),
        "training_base_model": args.model_name,
        "training_base_repo": TRAINING_MODEL_REPO,
        "dit_engine": args.dit_engine,
        "dit_weights_repo": OPTIMIZED_MODEL_REPO,
        "dit_weights_filename": OPTIMIZED_MEDIUM_BASE_FILENAME,
        "dit_weights_path": str(hosted_dit_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "latent_cache_dir": str(latent_cache_dir),
        "trigger_text": args.trigger_text,
        "examples": [
            {
                "audio_path": str(item["example"].audio_path),
                "relative_path": item["example"].relative_path,
                "sidecar_path": (
                    str(item["example"].sidecar_path)
                    if item["example"].sidecar_path is not None
                    else None
                ),
                "sidecar_kind": item["example"].sidecar_kind,
                "source_prompt": item["example"].source_prompt,
                "prompt": item["example"].prompt,
                "conditioning_seconds": item["conditioning_seconds"],
                "training_latent_frames": item["training_latent_frames"],
                "latent_metadata": item["latent_metadata"],
            }
            for item in prepared_examples
        ],
        "steps": args.steps,
        "full_tracks": args.full_tracks,
        "full_track_buckets": args.full_track_buckets,
        "training_latent_buckets": sorted(
            {
                item["training_latent_frames"]
                for item in prepared_examples
            }
        ),
        "crop_seconds": args.crop_seconds,
        "aligned_crop_seconds": aligned_crop_seconds,
        "crop_latents": crop_latents,
        "rank": args.rank,
        "alpha": alpha,
        "adapter_type": dit_injection.adapter_type,
        "layer_scope": args.layer_scope,
        "include": include,
        "exclude": exclude,
        "learning_rate": args.learning_rate,
        "cfg_dropout_prob": args.cfg_dropout_prob,
        "timestep_sampler": args.timestep_sampler,
        "distribution_shift": {
            "requested": args.distribution_shift,
            "resolved": distribution_shift_spec_to_jsonable(distribution_shift),
            "effective_seq_len": [
                item["effective_seq_len"] for item in prepared_examples
            ],
            "uses_effective_length": use_effective_length_for_schedule,
        },
        "autoencoder_dtype": args.autoencoder_dtype,
        "per_track_target_latent_rms": args.per_track_target_latent_rms,
        "seed": args.seed,
        "compile": args.compile,
        "gradient_checkpointing": args.grad_checkpoint,
        "mlx_wired_memory_limit_bytes": wired_memory_limit,
        "adapted_layers": adapted_layer_names,
        "trainable_parameters": trainable_parameters,
        "optimizer": {
            "type": "AdamW",
            "betas": list(SA3_ADAMW_BETAS),
            "eps": SA3_ADAMW_EPS,
            "weight_decay": SA3_ADAMW_WEIGHT_DECAY,
            "bias_correction": True,
        },
        "local_add_conditioning": {
            "mode": "upstream-pytorch-mixed-inpainting",
            "mask_semantics": "zero=generate, one=context",
            "mask_type_probabilities": {
                "random_segments": 0.1,
                "full_generation": 0.8,
                "causal": 0.1,
            },
            "context_reconstruction_loss_weight": 1.0,
            "silence_extension_scale_seconds": 4.0,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(
        f"training_mode={'full-track' if args.full_tracks else 'random-crop'} "
        f"crop_latents={crop_latents} "
        f"aligned_crop_seconds={aligned_crop_seconds:.3f}",
        flush=True,
    )
    if args.full_tracks:
        print(
            f"full_track_buckets={str(args.full_track_buckets).lower()} "
            "bucket_latent_frames="
            f"{sorted({item['training_latent_frames'] for item in prepared_examples})}",
            flush=True,
        )
    print(f"timestep_sampler={args.timestep_sampler}", flush=True)
    print(
        "distribution_shift="
        f"{json.dumps(distribution_shift_spec_to_jsonable(distribution_shift), sort_keys=True)} "
        "effective_seq_len=per-example "
        f"range={min(item['effective_seq_len'] for item in prepared_examples)}-"
        f"{max(item['effective_seq_len'] for item in prepared_examples)}",
        flush=True,
    )
    print(f"adapter_type={dit_injection.adapter_type}", flush=True)
    print(
        "training_local_add_conditioning=upstream-pytorch-mixed "
        "mask_semantics=zero-generate,one-context "
        "probabilities=random_segments:0.1,full:0.8,causal:0.1 "
        "context_loss_weight=1.0 silence_extension_seconds=4.0",
        flush=True,
    )
    print(f"compiled_training_step={str(args.compile).lower()}", flush=True)
    print(
        f"gradient_checkpointing={str(args.grad_checkpoint).lower()}",
        flush=True,
    )
    print(
        "lora_filters="
        f"{json.dumps({'include': include, 'exclude': exclude}, sort_keys=True)}",
        flush=True,
    )
    print(f"lora_layer_scope={args.layer_scope}", flush=True)
    print(
        f"adapted_layers={len(adapted_layer_names)} "
        f"dit_layers={dit_injection.layer_count} "
        f"seconds_conditioner_layers="
        f"{conditioner_injection.layer_count if conditioner_injection else 0}",
        flush=True,
    )
    print(f"trainable_parameters={trainable_parameters}", flush=True)
    print(
        "optimizer=AdamW "
        f"betas={SA3_ADAMW_BETAS} eps={SA3_ADAMW_EPS} "
        f"weight_decay={SA3_ADAMW_WEIGHT_DECAY} bias_correction=true",
        flush=True,
    )

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    timestep_rng = np.random.default_rng(args.seed + 1)
    mask_rng = np.random.default_rng(args.seed + 2)
    optimizer = _create_sa3_adamw(args.learning_rate)

    latent_arrays = [
        np.load(item["latent_path"], mmap_mode="r") for item in prepared_examples
    ]

    def loss_fn(
        local_bundle,
        clean,
        timesteps,
        generation_loss_mask,
        context_loss_mask,
        prompt_cond,
        seconds_total,
        local_add_cond,
        padding_mask,
    ):
        return rectified_flow_loss(
            local_bundle,
            clean,
            timesteps,
            loss_mask=generation_loss_mask,
            context_loss_mask=context_loss_mask,
            context_loss_weight=1.0,
            model_kwargs={
                "prompt_cond": prompt_cond,
                "seconds_total": seconds_total,
                "local_add_cond": local_add_cond,
                "padding_mask": padding_mask,
                "cfg_dropout_prob": float(args.cfg_dropout_prob),
            },
    )

    loss_and_grad = nn.value_and_grad(bundle, loss_fn)
    optimizer.init(bundle.trainable_parameters())
    compiled_state = [bundle.state, optimizer.state, mx.random.state]

    def train_step(
        clean,
        timesteps,
        generation_loss_mask,
        context_loss_mask,
        prompt_cond,
        seconds_total,
        local_add_cond,
        padding_mask,
    ):
        loss, grads = loss_and_grad(
            bundle,
            clean,
            timesteps,
            generation_loss_mask,
            context_loss_mask,
            prompt_cond,
            seconds_total,
            local_add_cond,
            padding_mask,
        )
        optimizer.update(bundle, grads)
        return loss

    uncompiled_train_step = train_step
    compiled_train_steps = {}

    def train_step_for_frames(latent_frames: int):
        if not args.compile:
            return uncompiled_train_step
        if latent_frames not in compiled_train_steps:
            compiled_train_steps[latent_frames] = partial(
                mx.compile,
                inputs=compiled_state,
                outputs=compiled_state,
            )(uncompiled_train_step)
        return compiled_train_steps[latent_frames]

    loss_log_path = output_dir / "loss.jsonl"
    if hasattr(mx, "reset_peak_memory"):
        mx.reset_peak_memory()
    started = time.perf_counter()
    mask_type_counts = {
        "random_segments": 0,
        "full": 0,
        "causal": 0,
    }
    with loss_log_path.open("a") as loss_log:
        for step in range(1, args.steps + 1):
            example_index = int(rng.integers(0, len(prepared_examples)))
            prepared = prepared_examples[example_index]
            latents = latent_arrays[example_index]
            step_crop_latents = int(prepared["training_latent_frames"])
            max_offset = int(prepared["valid_latent_frames"]) - crop_latents
            offset = (
                int(rng.integers(0, max_offset + 1))
                if not args.full_tracks and max_offset > 0
                else 0
            )
            clean = mx.array(
                _crop_or_pad_latents(
                    latents,
                    crop_latents=step_crop_latents,
                    offset=offset,
                    valid_frames=int(prepared["valid_latent_frames"]),
                    padding_latents=silence_latents,
                )
            ).astype(getattr(mx, pipeline.dtype_name))
            timestep_values = sample_training_timesteps(
                args.timestep_sampler,
                1,
                rng=timestep_rng,
            )
            timestep_values = np.asarray(
                shift_timestep_values(
                    timestep_values,
                    dist_shift=distribution_shift,
                    effective_seq_len=int(prepared["effective_seq_len"]),
                ),
                dtype=np.float32,
            )
            t = mx.array(timestep_values).astype(mx.float32)
            seconds_total = mx.array(
                [float(prepared["conditioning_seconds"])],
                dtype=mx.float32,
            )
            augmented_padding_values = _augment_padding_mask(
                valid_frames=int(prepared["valid_crop_frames"]),
                crop_latents=step_crop_latents,
                sample_rate=sample_rate,
                downsampling_ratio=downsampling_ratio,
                rng=mask_rng,
            )
            inpaint_mask_values, inpaint_mode = _sample_inpaint_mask(
                augmented_padding_values,
                rng=mask_rng,
            )
            mask_type_counts[inpaint_mode] += 1
            inpaint_mask = mx.array(inpaint_mask_values).astype(clean.dtype)
            local_add_cond = mx.concatenate(
                [inpaint_mask, clean * inpaint_mask],
                axis=1,
            )
            padding_mask = mx.array(augmented_padding_values)
            generation_loss_mask = mx.array(
                augmented_padding_values
                & (inpaint_mask_values[:, 0, :] == 0)
            )
            context_loss_mask = mx.array(
                augmented_padding_values
                & (inpaint_mask_values[:, 0, :] == 1)
            )

            step_started = time.perf_counter()
            first_compiled_bucket_use = (
                args.compile
                and step_crop_latents not in compiled_train_steps
            )
            loss = train_step_for_frames(step_crop_latents)(
                clean,
                t,
                generation_loss_mask,
                context_loss_mask,
                prepared["prompt_cond"],
                seconds_total,
                local_add_cond,
                padding_mask,
            )
            mx.eval(compiled_state, loss)
            step_seconds = time.perf_counter() - step_started
            if first_compiled_bucket_use:
                print(
                    "[compile] "
                    f"latent_frames={step_crop_latents} "
                    f"first_step_seconds={step_seconds:.2f}",
                    flush=True,
                )
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss_value}")

            record = {
                "step": step,
                "loss": loss_value,
                "t": float(t[0]),
                "crop_offset": offset,
                "full_tracks": args.full_tracks,
                "training_latent_frames": step_crop_latents,
                "conditioning_seconds": prepared["conditioning_seconds"],
                "effective_seq_len": prepared["effective_seq_len"],
                "inpaint_mode": inpaint_mode,
                "inpaint_context_fraction": float(
                    inpaint_mask_values[:, :, :].mean()
                ),
                "valid_latent_frames_with_silence": int(
                    augmented_padding_values.sum()
                ),
                "dataset_index": example_index,
                "audio_path": str(prepared["example"].audio_path),
                "prompt": prepared["example"].prompt,
                "step_seconds": step_seconds,
                "elapsed_seconds": time.perf_counter() - started,
            }
            loss_log.write(json.dumps(record) + "\n")
            loss_log.flush()

            if step == 1 or step % args.log_every == 0 or step == args.steps:
                average = (time.perf_counter() - started) / step
                memory = _mlx_memory_snapshot()
                memory_log = " ".join(
                    f"mlx_{name}={value:.2f}"
                    for name, value in memory.items()
                )
                print(
                    f"step={step}/{args.steps} loss={loss_value:.8f} "
                    f"step_seconds={step_seconds:.2f} average_seconds={average:.2f} "
                    f"latent_frames={step_crop_latents} "
                    f"inpaint_mode={inpaint_mode} "
                    f"mask_counts={json.dumps(mask_type_counts, sort_keys=True)} "
                    f"{memory_log}",
                    flush=True,
                )

            if args.save_every > 0 and step % args.save_every == 0:
                saved = _save_checkpoint(
                    bundle,
                    output_dir=output_dir,
                    step=step,
                    rank=args.rank,
                    alpha=alpha,
                    include=include,
                    exclude=exclude,
                    args=args,
                )
                print(f"checkpoint={saved}", flush=True)

    final_checkpoint = _save_checkpoint(
        bundle,
        output_dir=output_dir,
        step=args.steps,
        rank=args.rank,
        alpha=alpha,
        include=include,
        exclude=exclude,
        args=args,
        final=True,
    )
    print(f"final_checkpoint={final_checkpoint}", flush=True)
    print(f"total_seconds={time.perf_counter() - started:.2f}", flush=True)
    print(
        f"mlx_memory={json.dumps(_mlx_memory_snapshot(), sort_keys=True)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
