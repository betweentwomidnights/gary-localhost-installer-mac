#!/usr/bin/env python3
"""
Benchmark MLX MusicGen continuation with optional in-memory quantization.

This script is intentionally isolated from the localhost API path so we can
measure speed/quality tradeoffs safely before integrating anything into
`audiocraft/g4laudio_mlx.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import soundfile as sf
from mlx.utils import tree_flatten, tree_map
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIOCRAFT_ROOT = REPO_ROOT / "audiocraft"
if str(AUDIOCRAFT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIOCRAFT_ROOT))

from mlx_continuation.encodec import preprocess_audio as encodec_preprocess_audio
from mlx_continuation.mlx_musicgen import MusicGenContinuation


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, int(sr)


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
        raise ValueError("Expected encoded_frames with shape (chunks, B, K, frames).")
    codes = mx.transpose(encoded_frames, (1, 2, 0, 3))
    bsz, num_codebooks, num_chunks, frames = codes.shape
    return codes.reshape(bsz, num_codebooks, num_chunks * frames)


def _samples_to_frames(num_samples: int, sample_rate: int, frame_rate: float) -> int:
    return int(round((num_samples / sample_rate) * frame_rate))


def _eval_tree(tree) -> None:
    arrays = [v for _, v in tree_flatten(tree) if hasattr(v, "shape")]
    if arrays:
        mx.eval(*arrays)


def _cast_model_floats(model: MusicGenContinuation, dtype: mx.Dtype) -> None:
    float_dtypes = {mx.float16, mx.float32, mx.bfloat16}
    casted = tree_map(
        lambda x: x.astype(dtype) if getattr(x, "dtype", None) in float_dtypes else x,
        model.parameters(),
    )
    model.update(casted)


def _make_quant_predicate(scope: str, group_size: int, bits: int, mode: str):
    params = {"group_size": group_size, "bits": bits, "mode": mode}

    def _matches_scope(path: str, module: nn.Module) -> bool:
        if scope == "all-quantizable":
            return True
        if scope == "all-linears":
            return isinstance(module, nn.Linear)
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


def _count_quantized_modules(model: MusicGenContinuation) -> tuple[int, int]:
    n_q_linear = 0
    n_q_embed = 0
    for _, module in tree_flatten(model.leaf_modules()):
        if isinstance(module, nn.QuantizedLinear):
            n_q_linear += 1
        elif isinstance(module, nn.QuantizedEmbedding):
            n_q_embed += 1
    return n_q_linear, n_q_embed


def _encode_prompt_tokens(
    model: MusicGenContinuation,
    prompt_audio: np.ndarray,
    prompt_sr: int,
) -> mx.array:
    """
    Convert prompt waveform (T, C) at prompt_sr to prompt tokens (1, K, T_frames).
    """
    audio = _resample(prompt_audio, prompt_sr, model.sampling_rate)
    audio = _convert_channels(audio, model._audio_decoder.channels)
    num_samples = audio.shape[0]

    audio_mx = mx.array(audio)
    inputs, masks = encodec_preprocess_audio(
        audio_mx,
        sampling_rate=model.sampling_rate,
        chunk_length=model._audio_decoder.chunk_length,
        chunk_stride=model._audio_decoder.chunk_stride,
    )
    encoded_frames, _ = model._audio_decoder.encode(inputs, masks)
    prompt_tokens = _flatten_codes(encoded_frames)

    prompt_frames_target = _samples_to_frames(
        num_samples, model.sampling_rate, model.frame_rate
    )
    prompt_frames_target = max(1, min(prompt_frames_target, int(prompt_tokens.shape[-1])))
    return prompt_tokens[:, :, :prompt_frames_target]


def _slice_prompt(audio: np.ndarray, sr: int, seconds: float, use_last: bool) -> np.ndarray:
    prompt_samples = max(1, int(round(seconds * sr)))
    if audio.shape[0] < prompt_samples:
        pad = prompt_samples - audio.shape[0]
        audio = np.pad(audio, ((0, pad), (0, 0)), mode="constant").astype(np.float32)
    return audio[-prompt_samples:] if use_last else audio[:prompt_samples]


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="FLOAT")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLX MusicGen continuation with optional quantization."
    )
    parser.add_argument("--model", required=True, help="HF repo or local model path.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model for finetunes without config.json.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt WAV path (uses start of file by default, like process_audio).",
    )
    parser.add_argument(
        "--prompt-seconds",
        type=float,
        default=6.0,
        help="Prompt length to encode.",
    )
    parser.add_argument(
        "--output-seconds",
        type=float,
        default=30.0,
        help="Target total output duration in seconds.",
    )
    parser.add_argument(
        "--prompt-from-start",
        dest="prompt_from_start",
        action="store_true",
        help="Use first N seconds from prompt audio (default).",
    )
    parser.add_argument(
        "--prompt-from-end",
        dest="prompt_from_start",
        action="store_false",
        help="Use last N seconds from prompt audio (continue-style behavior).",
    )
    parser.set_defaults(prompt_from_start=True)
    parser.add_argument("--text", default="", help="Text conditioning.")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-coef", type=float, default=3.0)
    parser.add_argument(
        "--allow-empty-text-cfg",
        action="store_true",
        help=(
            "Keep CFG active when --text is empty. "
            "By default, empty text disables CFG for continuation quality."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic sampling (best-effort).",
    )
    parser.add_argument("--no-progress", action="store_true")

    parser.add_argument(
        "--cast-dtype",
        default="none",
        choices=["none", "float16", "bfloat16", "float32"],
        help="Cast model float params before quantization/loading benchmark.",
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--quant-scope",
        default="decoder-linears",
        choices=[
            "decoder-linears",
            "decoder-linears+emb",
            "all-linears",
            "all-quantizable",
        ],
    )
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--q-mode",
        default="affine",
        choices=["affine", "mxfp4", "mxfp8", "nvfp4"],
    )

    parser.add_argument(
        "--save-prefix",
        default=None,
        help="Optional output prefix (writes <prefix>_full.wav and <prefix>_cont.wav).",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional path to write benchmark summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    prompt_path = Path(args.prompt).expanduser().resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    if args.seed is not None:
        mx.random.seed(int(args.seed))

    try:
        mx.metal.reset_peak_memory()
    except Exception:
        pass

    timings: dict[str, float] = {}
    run_start = time.perf_counter()

    t0 = time.perf_counter()
    model = MusicGenContinuation.from_pretrained(args.model, base_model=args.base_model)
    _eval_tree(model.parameters())
    timings["model_load_s"] = time.perf_counter() - t0

    if args.cast_dtype != "none":
        t0 = time.perf_counter()
        cast_dtype = getattr(mx, args.cast_dtype)
        _cast_model_floats(model, cast_dtype)
        _eval_tree(model.parameters())
        timings["cast_dtype_s"] = time.perf_counter() - t0
    else:
        timings["cast_dtype_s"] = 0.0

    if args.quantize:
        t0 = time.perf_counter()
        predicate = _make_quant_predicate(
            scope=args.quant_scope,
            group_size=args.q_group_size,
            bits=args.q_bits,
            mode=args.q_mode,
        )
        nn.quantize(model, class_predicate=predicate)
        _eval_tree(model.parameters())
        timings["quantize_s"] = time.perf_counter() - t0
    else:
        timings["quantize_s"] = 0.0

    q_linear, q_embed = _count_quantized_modules(model)

    t0 = time.perf_counter()
    audio, sr = _load_audio(prompt_path)
    prompt_audio = _slice_prompt(
        audio=audio,
        sr=sr,
        seconds=args.prompt_seconds,
        use_last=not bool(args.prompt_from_start),
    )
    prompt_tokens = _encode_prompt_tokens(model, prompt_audio, sr)
    mx.eval(prompt_tokens)
    timings["prompt_encode_s"] = time.perf_counter() - t0

    prompt_frames = int(prompt_tokens.shape[-1])
    target_total_frames = int(round(float(args.output_seconds) * float(model.frame_rate)))
    if prompt_frames >= target_total_frames:
        raise ValueError(
            f"Prompt consumes all target frames ({prompt_frames} >= {target_total_frames}). "
            "Increase --output-seconds or reduce --prompt-seconds."
        )
    max_new_steps = target_total_frames - prompt_frames

    text = args.text or ""
    text_is_empty = text.strip() == ""
    cfg_enabled = (float(args.guidance_coef) != 0.0) and (
        (not text_is_empty) or bool(args.allow_empty_text_cfg)
    )
    effective_guidance_coef = float(args.guidance_coef) if cfg_enabled else 0.0

    t0 = time.perf_counter()
    tokens = model._generate_tokens_with_prompt(
        text=text,
        prompt_tokens=prompt_tokens,
        max_new_steps=max_new_steps,
        top_k=args.top_k,
        temp=args.temperature,
        guidance_coef=args.guidance_coef,
        allow_empty_text_cfg=bool(args.allow_empty_text_cfg),
        progress=not args.no_progress,
    )
    mx.eval(tokens)
    timings["token_generate_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    full_audio = model.decode_tokens(tokens)
    mx.eval(full_audio)
    full_np = np.array(full_audio[0]).astype(np.float32)
    timings["decode_full_s"] = time.perf_counter() - t0

    cont_np: Optional[np.ndarray] = None
    if args.save_prefix is not None:
        t0 = time.perf_counter()
        cont_tokens = tokens[:, prompt_frames:, :]
        cont_audio = model.decode_tokens(cont_tokens)
        mx.eval(cont_audio)
        cont_np = np.array(cont_audio[0]).astype(np.float32)
        timings["decode_cont_for_save_s"] = time.perf_counter() - t0
    else:
        timings["decode_cont_for_save_s"] = 0.0

    timings["total_s"] = time.perf_counter() - run_start

    generated_audio_s = max_new_steps / float(model.frame_rate)
    speed_x_realtime = (
        generated_audio_s / timings["token_generate_s"]
        if timings["token_generate_s"] > 0
        else float("inf")
    )
    wall_s_per_generated_audio_s = (
        timings["token_generate_s"] / generated_audio_s
        if generated_audio_s > 0
        else float("inf")
    )

    try:
        peak_mem_gb = float(mx.metal.get_peak_memory()) / (1024**3)
    except Exception:
        peak_mem_gb = None

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "prompt": str(prompt_path),
        "prompt_seconds": float(args.prompt_seconds),
        "prompt_slice": "start" if bool(args.prompt_from_start) else "end",
        "output_seconds": float(args.output_seconds),
        "text_len": len(args.text),
        "seed": args.seed,
        "sampling": {
            "top_k": int(args.top_k),
            "temperature": float(args.temperature),
            "requested_guidance_coef": float(args.guidance_coef),
            "cfg_enabled": bool(cfg_enabled),
            "effective_guidance_coef": float(effective_guidance_coef),
            "allow_empty_text_cfg": bool(args.allow_empty_text_cfg),
        },
        "quantization": {
            "enabled": bool(args.quantize),
            "scope": args.quant_scope if args.quantize else None,
            "bits": int(args.q_bits) if args.quantize else None,
            "group_size": int(args.q_group_size) if args.quantize else None,
            "mode": args.q_mode if args.quantize else None,
            "quantized_linear_modules": q_linear,
            "quantized_embedding_modules": q_embed,
        },
        "cast_dtype": args.cast_dtype,
        "shape": {
            "prompt_frames": prompt_frames,
            "target_total_frames": target_total_frames,
            "max_new_frames": max_new_steps,
            "frame_rate": float(model.frame_rate),
            "sample_rate": int(model.sampling_rate),
            "num_codebooks": int(model.num_codebooks),
        },
        "timings_s": timings,
        "throughput": {
            "generated_audio_seconds": generated_audio_s,
            "speed_x_realtime_for_generation": speed_x_realtime,
            "wall_seconds_per_generated_audio_second": wall_s_per_generated_audio_s,
        },
        "peak_memory_gb": peak_mem_gb,
    }

    print(json.dumps(summary, indent=2))

    if args.save_prefix is not None:
        out_prefix = Path(args.save_prefix).expanduser().resolve()
        full_path = out_prefix.parent / f"{out_prefix.name}_full.wav"
        cont_path = out_prefix.parent / f"{out_prefix.name}_cont.wav"
        _write_wav(full_path, full_np, model.sampling_rate)
        if cont_np is not None:
            _write_wav(cont_path, cont_np, model.sampling_rate)
        print(f"Wrote {full_path}")
        if cont_np is not None:
            print(f"Wrote {cont_path}")

    if args.report_json is not None:
        report_path = Path(args.report_json).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
