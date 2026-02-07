#!/usr/bin/env python3

import argparse
import os
import time
from typing import Tuple

import mlx.core as mx
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .encodec import preprocess_audio
from .mlx_musicgen import MusicGenContinuation


def _load_audio(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    return audio, sr


def _resample_audio(audio: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return audio
    return resample_poly(audio, out_sr, in_sr, axis=0).astype(np.float32)


def _convert_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == target_channels:
        return audio
    if target_channels == 1:
        return audio.mean(axis=1, keepdims=True)
    if audio.shape[1] == 1:
        return np.repeat(audio, target_channels, axis=1)
    return audio[:, :target_channels]


def _save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr)


def _flatten_codes(encoded_frames: mx.array) -> mx.array:
    if encoded_frames.ndim != 4:
        raise ValueError("Expected encoded_frames with 4 dims (chunks, B, K, frames)")
    codes = mx.transpose(encoded_frames, (1, 2, 0, 3))
    bsz, num_codebooks, num_chunks, frames = codes.shape
    return codes.reshape(bsz, num_codebooks, num_chunks * frames)


def _samples_to_frames(num_samples: int, sample_rate: int, frame_rate: float) -> int:
    return int(round((num_samples / sample_rate) * frame_rate))


def _encode_prompt_mlx(
    model: MusicGenContinuation,
    prompt_path: str,
    prompt_seconds: float,
    trim_to_frame_rate: bool,
) -> Tuple[mx.array, int]:
    audio, sr = _load_audio(prompt_path)
    audio = _resample_audio(audio, sr, model.sampling_rate)
    audio = _convert_channels(audio, model._audio_decoder.channels)

    if prompt_seconds is not None:
        max_samples = max(1, int(prompt_seconds * model.sampling_rate))
        audio = audio[:max_samples]

    num_samples = audio.shape[0]

    audio_mx = mx.array(audio)
    inputs, masks = preprocess_audio(
        audio_mx,
        sampling_rate=model.sampling_rate,
        chunk_length=model._audio_decoder.chunk_length,
        chunk_stride=model._audio_decoder.chunk_stride,
    )
    encoded_frames, _ = model._audio_decoder.encode(inputs, masks)
    prompt_tokens = _flatten_codes(encoded_frames)

    if trim_to_frame_rate:
        prompt_frames_target = _samples_to_frames(
            num_samples, model.sampling_rate, model.frame_rate
        )
        prompt_frames_target = max(1, min(prompt_frames_target, prompt_tokens.shape[-1]))
        prompt_tokens = prompt_tokens[:, :, :prompt_frames_target]

    return prompt_tokens, num_samples


def _encode_prompt_torch(
    model: MusicGenContinuation,
    prompt_path: str,
    prompt_seconds: float,
    trim_to_frame_rate: bool,
) -> Tuple[mx.array, int]:
    # Backward-compatible alias: historically this mode used audiocraft's torch
    # Encodec path. We now route to HF Encodec so this script no longer depends
    # on the full audiocraft package checkout.
    print("[INFO] --prompt-encodec torch uses HF Encodec backend in this build.")
    return _encode_prompt_hf(
        model=model,
        prompt_path=prompt_path,
        prompt_seconds=prompt_seconds,
        trim_to_frame_rate=trim_to_frame_rate,
    )


def _encode_prompt_hf(
    model: MusicGenContinuation,
    prompt_path: str,
    prompt_seconds: float,
    trim_to_frame_rate: bool,
) -> Tuple[mx.array, int]:
    import torch
    from transformers import EncodecModel

    audio, sr = _load_audio(prompt_path)
    audio = _resample_audio(audio, sr, model.sampling_rate)
    audio = _convert_channels(audio, model._audio_decoder.channels)

    if prompt_seconds is not None:
        max_samples = max(1, int(prompt_seconds * model.sampling_rate))
        audio = audio[:max_samples]

    num_samples = audio.shape[0]

    encodec = EncodecModel.from_pretrained("facebook/encodec_32khz")
    encodec.eval()

    audio_t = torch.from_numpy(audio).transpose(0, 1).unsqueeze(0).contiguous()
    audio_t = audio_t.to(dtype=torch.float32)
    padding_mask = torch.ones_like(audio_t, dtype=torch.bool)

    with torch.no_grad():
        enc_out = encodec.encode(audio_t, padding_mask=padding_mask, return_dict=True)

    codes = enc_out.audio_codes
    prompt_tokens = mx.array(codes.cpu().numpy())
    prompt_tokens = _flatten_codes(prompt_tokens)

    if trim_to_frame_rate:
        prompt_frames_target = _samples_to_frames(
            num_samples, model.sampling_rate, model.frame_rate
        )
        prompt_frames_target = max(1, min(prompt_frames_target, prompt_tokens.shape[-1]))
        prompt_tokens = prompt_tokens[:, :, :prompt_frames_target]

    return prompt_tokens, num_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/musicgen-small")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Fallback base model for config.json when finetune repo is missing it.",
    )
    parser.add_argument("--prompt", required=False, help="Path to prompt wav")
    parser.add_argument("--prompt-seconds", type=float, default=4.0)
    parser.add_argument("--continuation-seconds", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override continuation steps (frames). If set, overrides --continuation-seconds.")
    parser.add_argument("--text", default="", help="Optional text prompt")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Generate audio from text only (no prompt).",
    )
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--guidance-coef", type=float, default=3.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--prompt-encodec",
        choices=["mlx", "hf", "torch"],
        default="mlx",
        help="Which Encodec encoder to use for prompt tokenization. 'torch' is an alias of 'hf'.",
    )
    parser.add_argument(
        "--dump-prompt-recon",
        action="store_true",
        help="Write a reconstruction of the prompt tokens using the MLX decoder.",
    )
    parser.add_argument(
        "--no-trim-prompt",
        action="store_true",
        help="Do not trim prompt tokens to the target frame rate length.",
    )
    parser.add_argument(
        "--bos-stats",
        action="store_true",
        help="Print the BOS token ratio in the continuation tokens.",
    )
    parser.add_argument(
        "--token-stats",
        action="store_true",
        help="Print token debug stats for prompt/continuation.",
    )
    parser.add_argument(
        "--prepend-bos",
        action="store_true",
        help="Prepend a BOS frame to prompt tokens before continuation (debug).",
    )
    parser.add_argument(
        "--allow-empty-text-cfg",
        action="store_true",
        help="Allow CFG when text is empty (debug).",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a self-consistency check between text-only and continuation tokens.",
    )
    parser.add_argument(
        "--compare-encodec",
        action="store_true",
        help="Compare MLX vs HF Encodec prompt tokens (debug).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for self-check.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading MLX MusicGen model: {args.model}")
    model = MusicGenContinuation.from_pretrained(args.model, base_model=args.base_model)

    if args.compare_encodec:
        if not args.prompt:
            parser.error("--prompt is required for --compare-encodec.")
        print("Comparing prompt tokens: MLX vs HF Encodec")
        prompt_mlx, _ = _encode_prompt_mlx(
            model, args.prompt, args.prompt_seconds, trim_to_frame_rate=not args.no_trim_prompt
        )
        prompt_hf, _ = _encode_prompt_hf(
            model, args.prompt, args.prompt_seconds, trim_to_frame_rate=not args.no_trim_prompt
        )
        len_mlx = prompt_mlx.shape[-1]
        len_hf = prompt_hf.shape[-1]
        min_len = min(len_mlx, len_hf)
        print(f"MLX tokens: shape={prompt_mlx.shape} dtype={prompt_mlx.dtype}")
        print(f"HF tokens:  shape={prompt_hf.shape} dtype={prompt_hf.dtype}")
        if len_mlx != len_hf:
            print(f"Length mismatch: mlx={len_mlx} hf={len_hf} (comparing first {min_len})")
        if min_len == 0:
            print("No tokens to compare.")
            return
        prompt_mlx = prompt_mlx[:, :, :min_len]
        prompt_hf = prompt_hf[:, :, :min_len]
        match = mx.mean((prompt_mlx == prompt_hf).astype(mx.float32))
        mx.eval(match)
        print(f"Match ratio (all): {float(match):.4f}")
        for k in range(prompt_mlx.shape[1]):
            mk = mx.mean((prompt_mlx[:, k, :] == prompt_hf[:, k, :]).astype(mx.float32))
            mx.eval(mk)
            print(f"Match ratio (codebook {k}): {float(mk):.4f}")
        return

    if args.self_check:
        prompt_frames = int(round(args.prompt_seconds * model.frame_rate))
        cont_frames = args.max_steps or int(round(args.continuation_seconds * model.frame_rate))
        prompt_frames = max(1, prompt_frames)
        cont_frames = max(1, cont_frames)
        total_frames = prompt_frames + cont_frames
        max_steps_full = total_frames + model.num_codebooks - 1

        print(
            f"Self-check: prompt_frames={prompt_frames} "
            f"cont_frames={cont_frames} total_frames={total_frames}"
        )

        mx.random.seed(args.seed)
        full_tokens = model.generate_tokens(
            text=args.text,
            max_steps=max_steps_full,
            top_k=args.top_k,
            temp=args.temperature,
            guidance_coef=args.guidance_coef,
            allow_empty_text_cfg=args.allow_empty_text_cfg,
            progress=not args.no_progress,
        )
        mx.eval(full_tokens)

        prompt_tokens = mx.swapaxes(full_tokens[:, :prompt_frames, :], -1, -2)

        mx.random.seed(args.seed)
        _, _, cont_tokens = model.generate_continuation(
            prompt_tokens=prompt_tokens,
            max_new_steps=cont_frames,
            text=args.text,
            top_k=args.top_k,
            temp=args.temperature,
            guidance_coef=args.guidance_coef,
            allow_empty_text_cfg=args.allow_empty_text_cfg,
            progress=not args.no_progress,
            return_tokens=True,
            prepend_bos=args.prepend_bos,
        )
        mx.eval(cont_tokens)

        match_all = mx.mean((cont_tokens == full_tokens).astype(mx.float32))
        match_cont = mx.mean(
            (cont_tokens[:, prompt_frames:, :] == full_tokens[:, prompt_frames:, :]).astype(mx.float32)
        )
        mx.eval(match_all, match_cont)
        print(f"Match ratio (all): {float(match_all):.4f}")
        print(f"Match ratio (continuation): {float(match_cont):.4f}")
        return

    if args.text_only:
        text_steps = args.max_steps or int(round(args.continuation_seconds * model.frame_rate))
        text_steps = max(1, text_steps)
        print(f"Text-only steps: {text_steps}")
        start = time.perf_counter()
        audio = model.generate(
            text=args.text,
            max_steps=text_steps,
            top_k=args.top_k,
            temp=args.temperature,
            guidance_coef=args.guidance_coef,
            allow_empty_text_cfg=args.allow_empty_text_cfg,
        )
        mx.eval(audio)
        elapsed = time.perf_counter() - start
        audio_np = np.array(audio)
        out_path = os.path.join(args.out_dir, "out_text.wav")
        _save_audio(out_path, audio_np, model.sampling_rate)
        seconds = audio_np.shape[0] / model.sampling_rate
        print(f"Wrote: {out_path}")
        print(f"Audio seconds: {seconds:.2f}")
        print(f"Wall time: {elapsed:.2f}s | sec/sec: {seconds / elapsed:.2f}")
        return

    if not args.prompt:
        parser.error("--prompt is required unless --text-only is set.")

    trim_prompt = not args.no_trim_prompt
    if args.prompt_encodec == "torch":
        prompt_tokens, prompt_samples = _encode_prompt_torch(
            model, args.prompt, args.prompt_seconds, trim_prompt
        )
    elif args.prompt_encodec == "hf":
        prompt_tokens, prompt_samples = _encode_prompt_hf(
            model, args.prompt, args.prompt_seconds, trim_prompt
        )
    else:
        prompt_tokens, prompt_samples = _encode_prompt_mlx(
            model, args.prompt, args.prompt_seconds, trim_prompt
        )

    if args.max_steps is not None:
        cont_frames = args.max_steps
    else:
        cont_frames = int(round(args.continuation_seconds * model.frame_rate))

    cont_frames = max(1, cont_frames)

    prompt_seconds_actual = prompt_samples / model.sampling_rate
    print(
        "Prompt: "
        f"{prompt_seconds_actual:.2f}s | "
        f"tokens={prompt_tokens.shape[-1]} | "
        f"frame_rate={model.frame_rate:.2f}"
    )
    print(f"Continuation frames: {cont_frames}")

    if args.dump_prompt_recon:
        prompt_tokens_bt = mx.swapaxes(prompt_tokens, -1, -2)
        prompt_audio = model.decode_tokens(prompt_tokens_bt)
        mx.eval(prompt_audio)
        prompt_audio_np = np.array(prompt_audio[0])
        prompt_recon_path = os.path.join(args.out_dir, "out_prompt_recon.wav")
        _save_audio(prompt_recon_path, prompt_audio_np, model.sampling_rate)
        print(f"Wrote: {prompt_recon_path}")

    start = time.perf_counter()
    if args.bos_stats or args.token_stats:
        full_audio, cont_audio, all_tokens = model.generate_continuation(
            prompt_tokens=prompt_tokens,
            max_new_steps=cont_frames,
            text=args.text,
            top_k=args.top_k,
            temp=args.temperature,
            guidance_coef=args.guidance_coef,
            allow_empty_text_cfg=args.allow_empty_text_cfg,
            progress=not args.no_progress,
            return_tokens=True,
            prepend_bos=args.prepend_bos,
        )
    else:
        full_audio, cont_audio = model.generate_continuation(
            prompt_tokens=prompt_tokens,
            max_new_steps=cont_frames,
            text=args.text,
            top_k=args.top_k,
            temp=args.temperature,
            guidance_coef=args.guidance_coef,
            allow_empty_text_cfg=args.allow_empty_text_cfg,
            progress=not args.no_progress,
            prepend_bos=args.prepend_bos,
        )
    mx.eval(full_audio, cont_audio)
    elapsed = time.perf_counter() - start

    if args.bos_stats or args.token_stats:
        cont_tokens = all_tokens[:, prompt_tokens.shape[-1] :]
        if args.bos_stats:
            bos_ratio = mx.mean((cont_tokens == model.bos_token_id).astype(mx.float32))
            mx.eval(bos_ratio)
            print(f"BOS ratio (continuation): {float(bos_ratio):.4f}")
        if args.token_stats:
            prompt_bt = mx.swapaxes(prompt_tokens, -1, -2)
            prompt_out = all_tokens[:, : prompt_tokens.shape[-1]]
            match_ratio = mx.mean((prompt_out == prompt_bt).astype(mx.float32))
            mx.eval(match_ratio)
            print(f"Prompt match ratio: {float(match_ratio):.4f}")
            cont_np = np.array(cont_tokens)
            # cont_np shape: (B, T, K)
            for k in range(cont_np.shape[-1]):
                flat = cont_np[:, :, k].reshape(-1)
                uniq, counts = np.unique(flat, return_counts=True)
                top_idx = counts.argmax()
                top_token = int(uniq[top_idx])
                top_frac = float(counts[top_idx]) / float(flat.size)
                print(
                    f"Codebook {k}: unique={len(uniq)} | top_token={top_token} | top_frac={top_frac:.3f}"
                )
        total_frames = all_tokens.shape[1]
        print(
            "Token counts: "
            f"prompt={prompt_tokens.shape[-1]} | "
            f"total={total_frames} | "
            f"continuation={total_frames - prompt_tokens.shape[-1]}"
        )

    full_audio_np = np.array(full_audio[0])
    cont_audio_np = np.array(cont_audio[0])

    full_path = os.path.join(args.out_dir, "out_full.wav")
    cont_path = os.path.join(args.out_dir, "out_continuation_only.wav")
    _save_audio(full_path, full_audio_np, model.sampling_rate)
    _save_audio(cont_path, cont_audio_np, model.sampling_rate)

    full_seconds = full_audio_np.shape[0] / model.sampling_rate
    cont_seconds = cont_audio_np.shape[0] / model.sampling_rate

    print(f"Wrote: {full_path}")
    print(f"Wrote: {cont_path}")
    print(f"Full audio seconds: {full_seconds:.2f}")
    print(f"Continuation seconds: {cont_seconds:.2f}")
    print(f"Wall time: {elapsed:.2f}s | sec/sec (cont): {cont_seconds / elapsed:.2f}")


if __name__ == "__main__":
    main()
