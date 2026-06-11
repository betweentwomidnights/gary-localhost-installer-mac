#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from stable_audio_3.mlx.pipeline import StableAudioMLXPipeline  # noqa: E402


DEFAULT_PROMPT = (
    "garysmoke, warm experimental electronic music, textured synths, rhythmic, stereo"
)


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
        "No cached stable-audio-3-medium RF config/checkpoint pair was found."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched base and MLX LoRA Stable Audio 3 generations."
    )
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--strength", type=float, default=1.0)
    return parser.parse_args()


def _write_audio(path: Path, audio, sample_rate: int) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)[0].T
    sf.write(path, np.clip(array, -1.0, 1.0), sample_rate, subtype="PCM_16")
    return array


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_path = args.lora_path.expanduser().resolve()
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

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

    print("Loading base checkpoint and converting runtime modules to MLX...", flush=True)
    pipeline = StableAudioMLXPipeline.from_torch_checkpoint(
        config_path,
        checkpoint_path,
        torch_device="cpu",
        dit_dtype="float16",
        text_dtype="float16",
        number_dtype="float16",
        autoencoder_dtype="float32",
        attention="sliding",
        model_half=False,
    )
    generation_args = {
        "prompt": args.prompt,
        "duration": args.duration,
        "duration_padding_sec": 0.0,
        "steps": args.steps,
        "cfg_scale": 1.0,
        "seed": args.seed,
        "return_dict": True,
    }

    print("Rendering base comparison...", flush=True)
    base = pipeline.generate(**generation_args)
    sample_rate = int(base.report["sample_rate"])
    base_path = output_dir / "base.wav"
    base_array = _write_audio(base_path, base.audio, sample_rate)

    print(f"Loading LoRA: {lora_path}", flush=True)
    pipeline.load_lora([lora_path], names=["gary-mlx-smoke"], strength=args.strength)
    print("Rendering LoRA comparison...", flush=True)
    adapted = pipeline.generate(**generation_args)
    adapted_path = output_dir / "lora.wav"
    adapted_array = _write_audio(adapted_path, adapted.audio, sample_rate)

    delta = adapted_array - base_array
    comparison = {
        "prompt": args.prompt,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "strength": args.strength,
        "base_path": str(base_path),
        "lora_path": str(adapted_path),
        "checkpoint_path": str(lora_path),
        "base_report": base.report,
        "lora_report": adapted.report,
        "comparison": {
            "base_rms": float(np.sqrt(np.mean(base_array**2))),
            "lora_rms": float(np.sqrt(np.mean(adapted_array**2))),
            "delta_rms": float(np.sqrt(np.mean(delta**2))),
            "max_abs_delta": float(np.max(np.abs(delta))),
            "correlation": float(
                np.corrcoef(base_array.reshape(-1), adapted_array.reshape(-1))[0, 1]
            ),
        },
    }
    report_path = output_dir / "comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"base={base_path}", flush=True)
    print(f"lora={adapted_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(json.dumps(comparison["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
