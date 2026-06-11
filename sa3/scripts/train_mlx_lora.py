#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from stable_audio_3.mlx.conditioning import (  # noqa: E402
    assemble_conditioning_inputs_from_tensors,
    build_mlx_conditioning_tensors,
)
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
from stable_audio_3.mlx.training import (  # noqa: E402
    inject_trainable_lora,
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


def _cached_rf_assets() -> tuple[Path, Path]:
    snapshots = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--stabilityai--stable-audio-3-medium"
        / "snapshots"
    )
    configs = sorted(
        snapshots.glob("*/stable-audio-3-medium-RF.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for config in configs:
        checkpoint = config.with_suffix(".safetensors")
        if checkpoint.is_file():
            return config.absolute(), checkpoint.absolute()
    raise FileNotFoundError(
        "No cached stable-audio-3-medium RF config/checkpoint pair was found. "
        "Pass --config-path and --checkpoint-path explicitly."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLX Stable Audio 3 LoRA on an audio file or folder."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-path", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
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
    parser.add_argument("--prompt", default="")
    parser.add_argument("--trigger-text", default="")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--crop-seconds", type=float, default=47.0)
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
            "Defaults to all Linear layers in transformer layers 20-23."
        ),
    )
    return parser.parse_args()


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


def _crop_or_pad_latents(
    latents: np.ndarray,
    *,
    crop_latents: int,
    offset: int,
) -> np.ndarray:
    source = np.asarray(latents)
    if source.ndim != 3:
        raise ValueError(f"Expected [batch, channels, frames] latents, got {source.shape}.")
    if crop_latents <= 0:
        raise ValueError("crop_latents must be positive.")
    if offset < 0 or offset >= max(1, source.shape[-1]):
        raise ValueError(f"Invalid latent crop offset {offset} for {source.shape[-1]} frames.")

    cropped = source[..., offset : offset + crop_latents]
    if cropped.shape[-1] == crop_latents:
        return np.asarray(cropped, dtype=np.float16)

    padded = np.zeros((*source.shape[:-1], crop_latents), dtype=np.float16)
    padded[..., : cropped.shape[-1]] = cropped
    return padded


def _cache_directory(output_dir: Path, example: LoraDatasetExample) -> Path:
    digest = hashlib.sha1(str(example.audio_path).encode("utf-8")).hexdigest()[:12]
    stem = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in example.audio_path.stem.lower()
    ).strip("-")
    return output_dir / "encoded" / f"{stem or 'audio'}-{digest}"


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


def _save_checkpoint(
    model,
    *,
    output_dir: Path,
    step: int,
    rank: int,
    alpha: float,
    include: list[str],
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
        adapter_type=args.adapter_type,
        extra_metadata={
            "step": step,
            "base_model": "stable-audio-3-medium-RF",
            "dataset_path": str(
                args.dataset_dir.expanduser().resolve()
                if args.dataset_dir is not None
                else args.audio_path.expanduser().resolve()
            ),
            "trigger_text": args.trigger_text,
            "crop_seconds": args.crop_seconds,
            "learning_rate": args.learning_rate,
            "cfg_dropout_prob": args.cfg_dropout_prob,
            "timestep_sampler": args.timestep_sampler,
            "distribution_shift": args.distribution_shift,
            "seed": args.seed,
        },
    )


def main() -> None:
    args = _parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.crop_seconds <= 0:
        raise ValueError("--crop-seconds must be positive.")
    if args.per_track_target_latent_rms < 0:
        raise ValueError("--per-track-target-latent-rms must be zero or greater.")
    if args.shift_min_length <= 0 or args.shift_max_length <= args.shift_min_length:
        raise ValueError("--shift-max-length must be greater than --shift-min-length.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir, examples = _training_examples(args)
    if not examples:
        raise FileNotFoundError(
            f"No supported audio files found in {dataset_dir or args.audio_path}."
        )

    if args.config_path is None or args.checkpoint_path is None:
        cached_config, cached_checkpoint = _cached_rf_assets()
    config_path = (
        args.config_path.expanduser().absolute()
        if args.config_path is not None
        else cached_config
    )
    checkpoint_path = (
        args.checkpoint_path.expanduser().absolute()
        if args.checkpoint_path is not None
        else cached_checkpoint
    )
    include = args.include or ["transformer.layers.[20-23]"]
    alpha = float(args.rank if args.alpha is None else args.alpha)

    print(f"config={config_path}", flush=True)
    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"dataset={dataset_dir or examples[0].audio_path}", flush=True)
    print(f"dataset_examples={len(examples)}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
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
    pipeline.torch_pipeline = None
    gc.collect()
    print("torch_runtime_released=true", flush=True)

    crop_latents = int(
        math.ceil(args.crop_seconds * sample_rate / pipeline.autoencoder.downsampling_ratio)
    )
    crop_latents = int(math.ceil(crop_latents / 16) * 16)
    diffusion_config = pipeline.model_config.get("model", {}).get("diffusion", {})
    use_effective_length_for_schedule = bool(
        diffusion_config.get("use_effective_length_for_schedule", False)
    )
    effective_seq_len = (
        int(
            math.ceil(
                args.crop_seconds
                * sample_rate
                / pipeline.autoencoder.downsampling_ratio
            )
        )
        if use_effective_length_for_schedule
        else crop_latents
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
            output_dir=_cache_directory(output_dir, example),
            sample_rate=sample_rate,
            channels=channels,
            target_latent_rms=args.per_track_target_latent_rms,
        )
        encode_seconds = time.perf_counter() - encode_started
        latent_frames = int(latent_metadata["latent_shape"][-1])
        valid_crop_frames = min(crop_latents, latent_frames)
        padding_mask_values = np.zeros((1, crop_latents), dtype=np.bool_)
        padding_mask_values[:, :valid_crop_frames] = True
        loss_mask = mx.array(padding_mask_values)
        print(
            f"encoded_audio={progress} path={example.relative_path} "
            f"seconds={encode_seconds:.2f} cached={str(cache_hit).lower()} "
            f"source_seconds={latent_metadata['source_samples'] / sample_rate:.2f} "
            f"chunked={str(latent_metadata['chunked_encoding']).lower()}",
            flush=True,
        )

        conditioning_started = time.perf_counter()
        conditioning_tensors = build_mlx_conditioning_tensors(
            pipeline.model_config,
            [{"prompt": example.prompt, "seconds_total": args.crop_seconds}],
            text_conditioners={"prompt": pipeline.text_conditioner},
            number_conditioners={"seconds_total": pipeline.number_conditioner},
        )
        conditioning = assemble_conditioning_inputs_from_tensors(
            pipeline.model_config,
            conditioning_tensors,
            latent_length=crop_latents,
            dtype_name=pipeline.dtype_name,
        )
        conditioning.update(
            {
                "padding_mask": loss_mask,
                "cfg_scale": 1.0,
                "cfg_dropout_prob": float(args.cfg_dropout_prob),
            }
        )
        mx.eval(
            *[
                value
                for value in conditioning.values()
                if isinstance(value, mx.array)
            ]
        )
        prepared_examples.append(
            {
                "example": example,
                "latent_path": latent_path,
                "latent_frames": latent_frames,
                "valid_crop_frames": valid_crop_frames,
                "latent_metadata": latent_metadata,
                "conditioning": conditioning,
                "loss_mask": loss_mask,
            }
        )
        print(
            f"conditioned_example={progress} path={example.relative_path} "
            f"seconds={time.perf_counter() - conditioning_started:.2f}",
            flush=True,
        )
        print(
            f"encoded_example={progress} path={example.relative_path} "
            f"seconds={time.perf_counter() - encode_started:.2f}",
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

    model = pipeline.mlx_dit
    injection = inject_trainable_lora(
        model,
        rank=args.rank,
        alpha=alpha,
        include=include,
        adapter_type=args.adapter_type,
    )
    mx.eval(model.parameters())

    pipeline.torch_pipeline = None
    pipeline.autoencoder = None
    pipeline.text_conditioner = None
    pipeline.number_conditioner = None
    gc.collect()
    mx.clear_cache()

    run_config = {
        "dataset_path": str(dataset_dir or examples[0].audio_path),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
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
                "latent_metadata": item["latent_metadata"],
            }
            for item in prepared_examples
        ],
        "steps": args.steps,
        "crop_seconds": args.crop_seconds,
        "crop_latents": crop_latents,
        "rank": args.rank,
        "alpha": alpha,
        "adapter_type": injection.adapter_type,
        "include": include,
        "learning_rate": args.learning_rate,
        "cfg_dropout_prob": args.cfg_dropout_prob,
        "timestep_sampler": args.timestep_sampler,
        "distribution_shift": {
            "requested": args.distribution_shift,
            "resolved": distribution_shift_spec_to_jsonable(distribution_shift),
            "effective_seq_len": effective_seq_len,
            "uses_effective_length": use_effective_length_for_schedule,
        },
        "autoencoder_dtype": args.autoencoder_dtype,
        "per_track_target_latent_rms": args.per_track_target_latent_rms,
        "seed": args.seed,
        "adapted_layers": list(injection.layer_names),
        "trainable_parameters": injection.trainable_parameters,
    }
    (output_dir / "run.json").write_text(json.dumps(run_config, indent=2) + "\n")

    print(f"crop_latents={crop_latents}", flush=True)
    print(f"timestep_sampler={args.timestep_sampler}", flush=True)
    print(
        "distribution_shift="
        f"{json.dumps(distribution_shift_spec_to_jsonable(distribution_shift), sort_keys=True)} "
        f"effective_seq_len={effective_seq_len}",
        flush=True,
    )
    print(f"adapter_type={injection.adapter_type}", flush=True)
    print(f"adapted_layers={injection.layer_count}", flush=True)
    print(f"trainable_parameters={injection.trainable_parameters}", flush=True)

    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    timestep_rng = np.random.default_rng(args.seed + 1)
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=0.0,
    )

    latent_arrays = [
        np.load(item["latent_path"], mmap_mode="r") for item in prepared_examples
    ]

    def make_loss_and_grad(conditioning, loss_mask):
        def loss_fn(local_model, clean, timesteps):
            return rectified_flow_loss(
                local_model,
                clean,
                timesteps,
                loss_mask=loss_mask,
                model_kwargs=conditioning,
            )

        return nn.value_and_grad(model, loss_fn)

    loss_and_grads = [
        make_loss_and_grad(item["conditioning"], item["loss_mask"])
        for item in prepared_examples
    ]
    loss_log_path = output_dir / "loss.jsonl"
    started = time.perf_counter()
    with loss_log_path.open("a") as loss_log:
        for step in range(1, args.steps + 1):
            example_index = int(rng.integers(0, len(prepared_examples)))
            prepared = prepared_examples[example_index]
            latents = latent_arrays[example_index]
            max_offset = int(prepared["latent_frames"]) - crop_latents
            offset = int(rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            clean = mx.array(
                _crop_or_pad_latents(
                    latents,
                    crop_latents=crop_latents,
                    offset=offset,
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
                    effective_seq_len=effective_seq_len,
                ),
                dtype=np.float32,
            )
            t = mx.array(timestep_values).astype(mx.float32)

            step_started = time.perf_counter()
            loss, grads = loss_and_grads[example_index](model, clean, t)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            step_seconds = time.perf_counter() - step_started
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss_value}")

            record = {
                "step": step,
                "loss": loss_value,
                "t": float(t[0]),
                "crop_offset": offset,
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
                print(
                    f"step={step}/{args.steps} loss={loss_value:.8f} "
                    f"step_seconds={step_seconds:.2f} average_seconds={average:.2f}",
                    flush=True,
                )

            if args.save_every > 0 and step % args.save_every == 0:
                saved = _save_checkpoint(
                    model,
                    output_dir=output_dir,
                    step=step,
                    rank=args.rank,
                    alpha=alpha,
                    include=include,
                    args=args,
                )
                print(f"checkpoint={saved}", flush=True)

    final_checkpoint = _save_checkpoint(
        model,
        output_dir=output_dir,
        step=args.steps,
        rank=args.rank,
        alpha=alpha,
        include=include,
        args=args,
        final=True,
    )
    print(f"final_checkpoint={final_checkpoint}", flush=True)
    print(f"total_seconds={time.perf_counter() - started:.2f}", flush=True)


if __name__ == "__main__":
    main()
