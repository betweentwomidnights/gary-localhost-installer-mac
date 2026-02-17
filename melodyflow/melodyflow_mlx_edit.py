#!/usr/bin/env python3
"""Minimal MLX edit runtime for MelodyFlow localhost backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import torch
from omegaconf import OmegaConf

from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout
from audiocraft.utils.renoise import noise_regularization
from audiocraft.utils.utils import vae_sample

from melodyflow_mlx_codec import build_mlx_melodyflow_codec
from melodyflow_mlx_flow import build_mlx_flow_from_state

ProgressCallback = Callable[[int, int], None]


@dataclass
class MelodyFlowMlxRuntime:
    flow: Any
    mlx_dtype: Any
    dtype_name: str
    native_codec: Any | None = None


def build_runtime(model: Any, dtype_name: str = "float32") -> MelodyFlowMlxRuntime:
    lm = model.lm
    cfg = OmegaConf.create(
        {
            "transformer_lm": {
                "num_heads": int(lm.transformer.layers[0].self_attn.num_heads),
                "skip_connections": bool(lm.transformer.skip_connections),
            }
        }
    )
    mlx_dtype = getattr(mx, dtype_name)
    flow = build_mlx_flow_from_state(lm.state_dict(), cfg, dtype=mlx_dtype)
    return MelodyFlowMlxRuntime(flow=flow, mlx_dtype=mlx_dtype, dtype_name=dtype_name)


def ensure_native_codec(runtime: MelodyFlowMlxRuntime, model: Any) -> None:
    if runtime.native_codec is not None:
        return
    runtime.native_codec = build_mlx_melodyflow_codec(model.compression_model, dtype=runtime.mlx_dtype)


def encode_audio_with_mlx_codec(
    runtime: MelodyFlowMlxRuntime,
    waveform: torch.Tensor,
    out_device: torch.device,
) -> torch.Tensor:
    if runtime.native_codec is None:
        raise RuntimeError("Native MLX codec has not been initialized.")
    wave_np = waveform.detach().cpu().float().numpy()
    wave_mx = mx.array(wave_np).astype(runtime.mlx_dtype)
    codes, _ = runtime.native_codec.encode(wave_mx)
    mx.eval(codes)
    latent = np.asarray(mx.squeeze(codes, axis=1), dtype=np.float32)
    return torch.from_numpy(latent).to(device=out_device)


def decode_audio_with_mlx_codec(
    model: Any,
    runtime: MelodyFlowMlxRuntime,
    gen_tokens: torch.Tensor,
) -> torch.Tensor:
    if runtime.native_codec is None:
        raise RuntimeError("Native MLX codec has not been initialized.")
    assert gen_tokens.dim() == 3
    with torch.no_grad():
        if model.lm.latent_mean.shape[1] != gen_tokens.shape[1]:
            mean, scale = gen_tokens.chunk(2, dim=1)
            latent = vae_sample(mean, scale)
        else:
            latent = gen_tokens * (model.lm.latent_std + 1e-5) + model.lm.latent_mean
    latent_np = latent.detach().cpu().float().numpy()
    latent_mx = mx.array(latent_np).astype(runtime.mlx_dtype)
    audio_mx = runtime.native_codec.decode(latent_mx, None)
    mx.eval(audio_mx)
    audio_np = np.asarray(audio_mx, dtype=np.float32)
    return torch.from_numpy(audio_np)


def prompt_tokens_to_mean_latent(prompt_tokens: torch.Tensor, lm: torch.nn.Module) -> torch.Tensor:
    if prompt_tokens.dim() != 3:
        raise ValueError(f"Expected prompt tokens [B, C, T], got shape={tuple(prompt_tokens.shape)}")
    if int(prompt_tokens.shape[1]) == int(lm.latent_mean.shape[1]):
        return prompt_tokens
    mean, _ = prompt_tokens.chunk(2, dim=1)
    return (mean - lm.latent_mean) / (lm.latent_std + 1e-5)


def _mx_scalar(x: mx.array) -> float:
    return float(np.asarray(x).item())


def _mx_repeat_batch(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    reps = [1] * x.ndim
    reps[0] = repeats
    return mx.tile(x, reps)


def _build_schedule_numpy(
    *,
    source_flowstep: float,
    target_flowstep: float,
    steps: int,
    sway_coefficient: float,
) -> np.ndarray:
    if target_flowstep > source_flowstep:
        schedule = np.arange(0.0, 1.0 + 1e-5, 1.0 / steps, dtype=np.float32)
    else:
        schedule = np.arange(1.0, 0.0 - 1e-5, -1.0 / steps, dtype=np.float32)
    schedule = schedule + sway_coefficient * (np.cos(np.pi * 0.5 * schedule) - 1.0 + schedule)
    if target_flowstep > source_flowstep:
        schedule = schedule * (target_flowstep - source_flowstep) + source_flowstep
    else:
        schedule = schedule * (source_flowstep - target_flowstep) + target_flowstep
    return schedule.astype(np.float32, copy=False)


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    elapsed: int,
    offset: int,
    total_steps: int,
) -> None:
    if callback is None:
        return
    callback(offset + elapsed, total_steps)


def run_generate_mlx_native(
    lm: torch.nn.Module,
    *,
    mlx_model: Any,
    attributes: list[Any],
    prompt_tokens: torch.Tensor | None,
    solver: str,
    steps: int,
    max_gen_len: int,
    source_flowstep: float,
    target_flowstep: float,
    regularize: bool,
    regularize_iters: int,
    keep_last_k_iters: int,
    lambda_kl: float,
    progress_callback: ProgressCallback | None = None,
    progress_step_offset: int = 0,
    progress_total_steps: int | None = None,
    sway_coefficient: float = -0.8,
) -> torch.Tensor:
    assert solver in {"euler", "midpoint"}, "Supported ODE solvers are either euler or midpoint!"
    assert not (regularize and solver == "midpoint"), "Latent regularization is only supported with euler solver!"
    assert keep_last_k_iters <= regularize_iters
    lm_device = next(iter(lm.parameters())).device

    if prompt_tokens is not None and lm.latent_mean.shape[1] != prompt_tokens.shape[1]:
        mean, scale = prompt_tokens.chunk(2, dim=1)
        sampled = vae_sample(mean, scale)
        prompt_tokens = (sampled - lm.latent_mean) / (lm.latent_std + 1e-5)
    prompt_tokens_torch = prompt_tokens.detach().to(lm_device) if prompt_tokens is not None else None

    if attributes:
        null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(attributes)
        tokenized = lm.condition_provider.tokenize(attributes + null_conditions)
        cfg_conditions = lm.condition_provider(tokenized)
    else:
        raise RuntimeError("Expected non-empty attributes for edit loop.")

    if "description" not in cfg_conditions:
        raise RuntimeError("Missing 'description' conditioning in condition provider output.")
    cond_src_t, cond_mask_t = cfg_conditions["description"]

    if prompt_tokens is None:
        gen_sequence_t = torch.randn(1, lm.latent_dim, max_gen_len, device=lm_device)
    else:
        gen_sequence_t = prompt_tokens

    bsz = int(gen_sequence_t.shape[0])
    mx_dtype = mlx_model.in_proj_w.dtype

    gen_sequence = mx.array(gen_sequence_t.detach().float().cpu().numpy()).astype(mx_dtype)
    prompt_mx = None
    if prompt_tokens is not None:
        prompt_mx = mx.array(prompt_tokens.detach().float().cpu().numpy()).astype(mx_dtype)

    cond_src_all = mx.array(cond_src_t.detach().float().cpu().numpy()).astype(mx_dtype)
    cond_mask_log_all = mx.array(
        torch.log(cond_mask_t.unsqueeze(1).unsqueeze(1)).detach().float().cpu().numpy()
    ).astype(mx_dtype)
    cond_src_cond = cond_src_all[:bsz]
    cond_mask_log_cond = cond_mask_log_all[:bsz]

    if solver == "midpoint":
        assert steps % 2 == 0, "Midpoint solver can only run with even number of steps"
        next_sequence: mx.array | None = None

    if not regularize:
        regularize_iters = 1
        keep_last_k_iters = 0
    else:
        avg_regularized_velocity = mx.zeros_like(gen_sequence)

    regularization_iters_threshold = regularize_iters - keep_last_k_iters
    cfg_coef = 0.0 if target_flowstep < source_flowstep else float(lm.cfg_coef)
    schedule = _build_schedule_numpy(
        source_flowstep=float(source_flowstep),
        target_flowstep=float(target_flowstep),
        steps=int(steps),
        sway_coefficient=float(sway_coefficient),
    )

    regularize_weight_denominator = float(
        keep_last_k_iters * regularization_iters_threshold + sum(range(keep_last_k_iters))
    )
    total_callback_steps = int(steps * regularize_iters)
    report_total = total_callback_steps if progress_total_steps is None else int(progress_total_steps)

    for idx, current_flowstep in enumerate(schedule[:-1]):
        delta_t = float(schedule[idx + 1] - current_flowstep)
        input_sequence = gen_sequence
        if solver == "midpoint" and idx % 2 == 1:
            assert next_sequence is not None
            input_sequence = next_sequence

        for jdx in range(regularize_iters):
            should_compute_kl = jdx >= regularization_iters_threshold
            if should_compute_kl:
                if prompt_mx is None:
                    raise RuntimeError("Regularization path requires prompt tokens.")
                if prompt_tokens_torch is None:
                    raise RuntimeError("Missing torch prompt tokens for regularization.")
                torch_shape = tuple(int(d) for d in input_sequence.shape)
                regularizing_sequence_torch = (
                    (1.0 - float(schedule[idx + 1]))
                    * torch.randn(torch_shape, device=lm_device, dtype=prompt_tokens_torch.dtype)
                    + float(schedule[idx + 1]) * prompt_tokens_torch
                    + 1e-5 * torch.randn(torch_shape, device=lm_device, dtype=prompt_tokens_torch.dtype)
                )
                regularizing_sequence = mx.array(
                    regularizing_sequence_torch.detach().float().cpu().numpy()
                ).astype(mx_dtype)
                input_sequence = mx.concatenate([input_sequence, regularizing_sequence], axis=0)

            effective_t = float(schedule[idx + 1] if regularize else current_flowstep)
            regularizing_velocity: mx.array | None = None

            if cfg_coef == 0.0:
                rep = 1 + int(should_compute_kl)
                cond_src = _mx_repeat_batch(cond_src_cond, rep)
                cond_mask_log = _mx_repeat_batch(cond_mask_log_cond, rep)
                t_batch = mx.full((bsz * rep, 1), effective_t, dtype=mx_dtype)
                predicted_velocity = mlx_model.forward(
                    input_sequence,
                    t_batch,
                    cond_src,
                    cond_mask_log,
                )
                mx.eval(predicted_velocity)
                velocity = predicted_velocity[:bsz]
                if should_compute_kl:
                    regularizing_velocity = predicted_velocity[bsz:]
            else:
                rep = 1 + int(should_compute_kl)
                input_repeated = _mx_repeat_batch(input_sequence, 2)
                cond_src = mx.repeat(cond_src_all, rep, axis=0)
                cond_mask_log = mx.repeat(cond_mask_log_all, rep, axis=0)
                t_batch = mx.full((bsz * 2 * rep, 1), effective_t, dtype=mx_dtype)
                predicted_velocity = mlx_model.forward(
                    input_repeated,
                    t_batch,
                    cond_src,
                    cond_mask_log,
                )
                mx.eval(predicted_velocity)
                if should_compute_kl:
                    velocity = (1.0 + cfg_coef) * predicted_velocity[:bsz] - cfg_coef * predicted_velocity[2 * bsz : 3 * bsz]
                    regularizing_velocity = (
                        (1.0 + cfg_coef) * predicted_velocity[bsz : 2 * bsz]
                        - cfg_coef * predicted_velocity[3 * bsz : 4 * bsz]
                    )
                else:
                    velocity = (1.0 + cfg_coef) * predicted_velocity[:bsz] - cfg_coef * predicted_velocity[bsz : 2 * bsz]

            if should_compute_kl:
                if regularizing_velocity is None:
                    raise RuntimeError("Expected regularizing velocity when should_compute_kl is true.")
                velocity_cpu = torch.from_numpy(np.asarray(velocity, dtype=np.float32))
                regularizing_cpu = torch.from_numpy(np.asarray(regularizing_velocity, dtype=np.float32))
                regularized_velocity_t = noise_regularization(
                    velocity_cpu,
                    regularizing_cpu,
                    lambda_kl=float(lambda_kl),
                    lambda_ac=0.0,
                    num_reg_steps=4,
                    num_ac_rolls=5,
                )
                regularized_velocity = mx.array(regularized_velocity_t.detach().float().cpu().numpy()).astype(mx_dtype)
                if regularize_weight_denominator > 0:
                    avg_regularized_velocity = avg_regularized_velocity + (
                        regularized_velocity * (float(jdx) / regularize_weight_denominator)
                    )
                input_sequence = gen_sequence + regularized_velocity * delta_t
            else:
                input_sequence = gen_sequence + velocity * delta_t

            elapsed = 1 + idx * regularize_iters + jdx
            _emit_progress(
                progress_callback,
                elapsed=elapsed,
                offset=progress_step_offset,
                total_steps=report_total,
            )

        if regularize:
            velocity = avg_regularized_velocity
            avg_regularized_velocity = mx.zeros_like(gen_sequence)

        if solver == "midpoint":
            if idx % 2 == 0:
                next_sequence = gen_sequence + velocity * delta_t
            else:
                gen_sequence = gen_sequence + velocity * float(schedule[idx + 1] - schedule[idx - 1])
        else:
            gen_sequence = gen_sequence + velocity * delta_t

    out_np = np.asarray(gen_sequence, dtype=np.float32)
    return torch.from_numpy(out_np).to(device=lm_device)


def run_edit_loop_native(
    model: Any,
    *,
    prompt_tokens: torch.Tensor,
    source_attributes: list[Any],
    edit_attributes: list[Any],
    mlx_model: Any,
    progress_callback: ProgressCallback | None = None,
) -> torch.Tensor:
    inversion_params = model.editing_params.copy()

    inversion_steps = int(inversion_params["steps"])
    inversion_regularize = bool(inversion_params["regularize"])
    inversion_regularize_iters = int(inversion_params["regularize_iters"])
    inversion_total = inversion_steps * (inversion_regularize_iters if inversion_regularize else 1)
    forward_total = inversion_steps
    overall_total = inversion_total + forward_total

    intermediate_tokens = run_generate_mlx_native(
        model.lm,
        mlx_model=mlx_model,
        attributes=source_attributes,
        prompt_tokens=prompt_tokens,
        source_flowstep=1.0,
        target_flowstep=float(inversion_params["target_flowstep"]),
        solver=str(inversion_params["solver"]),
        steps=inversion_steps,
        regularize=inversion_regularize,
        regularize_iters=inversion_regularize_iters,
        keep_last_k_iters=int(inversion_params["keep_last_k_iters"]),
        lambda_kl=float(inversion_params["lambda_kl"]),
        max_gen_len=int(prompt_tokens.shape[-1]),
        progress_callback=progress_callback,
        progress_step_offset=0,
        progress_total_steps=overall_total,
    )

    if intermediate_tokens.shape[0] < len(edit_attributes):
        intermediate_tokens = intermediate_tokens.repeat(len(edit_attributes) // intermediate_tokens.shape[0], 1, 1)

    forward_params = inversion_params.copy()
    forward_params.pop("regularize")
    source_flowstep = float(forward_params.pop("target_flowstep"))

    final_tokens = run_generate_mlx_native(
        model.lm,
        mlx_model=mlx_model,
        attributes=edit_attributes,
        prompt_tokens=intermediate_tokens,
        source_flowstep=source_flowstep,
        target_flowstep=1.0,
        solver=str(forward_params["solver"]),
        steps=int(forward_params["steps"]),
        regularize=False,
        regularize_iters=int(forward_params["regularize_iters"]),
        keep_last_k_iters=int(forward_params["keep_last_k_iters"]),
        lambda_kl=float(forward_params["lambda_kl"]),
        max_gen_len=int(intermediate_tokens.shape[-1]),
        progress_callback=progress_callback,
        progress_step_offset=inversion_total,
        progress_total_steps=overall_total,
    )

    return final_tokens


def edit_with_mlx(
    model: Any,
    runtime: MelodyFlowMlxRuntime,
    *,
    prompt_tokens: torch.Tensor,
    prompt_text: str,
    src_prompt_text: str,
    codec_mode: str,
    native_prompt_mode: str,
    progress_callback: ProgressCallback | None = None,
) -> torch.Tensor:
    if codec_mode not in {"torch", "native"}:
        raise ValueError("codec_mode must be 'torch' or 'native'")
    if native_prompt_mode not in {"mean", "stochastic"}:
        raise ValueError("native_prompt_mode must be 'mean' or 'stochastic'")

    source_attributes, _ = model._prepare_tokens_and_attributes([src_prompt_text], None)
    edit_attributes, _ = model._prepare_tokens_and_attributes([prompt_text], None)

    prompt_tokens_for_flow = prompt_tokens
    if codec_mode == "native" and native_prompt_mode == "mean":
        prompt_tokens_for_flow = prompt_tokens_to_mean_latent(prompt_tokens=prompt_tokens, lm=model.lm)

    final_tokens = run_edit_loop_native(
        model,
        prompt_tokens=prompt_tokens_for_flow,
        source_attributes=source_attributes,
        edit_attributes=edit_attributes,
        mlx_model=runtime.flow,
        progress_callback=progress_callback,
    )

    if codec_mode == "native":
        return decode_audio_with_mlx_codec(model=model, runtime=runtime, gen_tokens=final_tokens)
    return model.generate_audio(final_tokens)


def find_max_duration_with_encoder(
    waveform: torch.Tensor,
    *,
    encode_audio_fn: Callable[[torch.Tensor], torch.Tensor],
    sr: int = 32000,
    max_token_length: int = 750,
) -> tuple[float, torch.Tensor]:
    min_seconds = 1.0
    max_seconds = waveform.shape[-1] / sr

    if max_seconds <= min_seconds:
        tokens = encode_audio_fn(waveform)
        return max_seconds, tokens

    best_duration = min_seconds
    best_tokens: torch.Tensor | None = None
    while max_seconds - min_seconds > 0.1:
        mid_seconds = (min_seconds + max_seconds) / 2.0
        samples = max(1, int(mid_seconds * sr))
        test_waveform = waveform[..., :samples]
        try:
            tokens = encode_audio_fn(test_waveform)
            token_length = int(tokens.shape[-1])
            if token_length <= max_token_length:
                best_duration = mid_seconds
                best_tokens = tokens
                min_seconds = mid_seconds
            else:
                max_seconds = mid_seconds
        except Exception:
            max_seconds = mid_seconds

    if best_tokens is None:
        fallback_samples = max(1, int(min_seconds * sr))
        best_tokens = encode_audio_fn(waveform[..., :fallback_samples])
        best_duration = min_seconds

    if int(best_tokens.shape[-1]) > max_token_length:
        raise RuntimeError(
            f"Prompt token length {int(best_tokens.shape[-1])} exceeds max_token_length={max_token_length}"
        )
    return best_duration, best_tokens
