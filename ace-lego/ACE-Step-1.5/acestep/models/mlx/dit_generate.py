# MLX diffusion generation loop for AceStep DiT decoder.
#
# Replicates the timestep scheduling and ODE/SDE stepping from
# ``AceStepConditionGenerationModel.generate_audio`` using pure MLX arrays.

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Pre-defined timestep schedules (from modeling_acestep_v15_turbo.py)
VALID_SHIFTS = [1.0, 2.0, 3.0]

VALID_TIMESTEPS = [
    1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
    0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
    0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
    0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125,
]

SHIFT_TIMESTEPS = {
    1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
    2.0: [1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
          0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222],
    3.0: [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
          0.6428571428571429, 0.5, 0.3],
}


def _mlx_apg_forward(
    pred_cond,
    pred_uncond,
    guidance_scale: float,
    momentum_state: Optional[Dict[str, object]] = None,
    norm_threshold: float = 2.5,
):
    """Approximate the PyTorch APG guidance path in MLX.

    The torch implementation projects along dim 1 and keeps a small negative
    momentum buffer. We mirror that behavior closely enough to recover the
    quality characteristics of the native torch path without pulling in the
    broader upstream sampler rewrite.
    """
    import mlx.core as mx

    diff = pred_cond - pred_uncond
    if momentum_state is not None:
        running_average = momentum_state.get("running_average", 0)
        momentum = float(momentum_state.get("momentum", -0.75))
        running_average = diff + momentum * running_average
        momentum_state["running_average"] = running_average
        diff = running_average

    if norm_threshold > 0.0:
        diff_norm = mx.sqrt((diff * diff).sum(axis=1, keepdims=True))
        scale_factor = mx.minimum(mx.ones_like(diff_norm), norm_threshold / (diff_norm + 1e-8))
        diff = diff * scale_factor

    pred_cond_norm = mx.sqrt((pred_cond * pred_cond).sum(axis=1, keepdims=True))
    pred_cond_unit = pred_cond / (pred_cond_norm + 1e-8)
    diff_parallel = (diff * pred_cond_unit).sum(axis=1, keepdims=True) * pred_cond_unit
    diff_orthogonal = diff - diff_parallel
    return pred_cond + (guidance_scale - 1.0) * diff_orthogonal


def get_timestep_schedule(
    shift: float = 3.0,
    timesteps: Optional[list] = None,
    infer_steps: int = 8,
    is_turbo: bool = True,
) -> List[float]:
    """Compute the timestep schedule for diffusion sampling.

    For turbo variants, this keeps the fast fixed schedule behavior.
    For base variants, this mirrors the PyTorch base schedule:
    linspace(1 -> 0, infer_steps + 1) with optional shift transform.

    Args:
        shift: Diffusion timestep shift (1, 2, or 3).
        timesteps: Optional custom list of timesteps.
        infer_steps: Requested inference step count for base variants.
        is_turbo: Whether the loaded DiT config is turbo.

    Returns:
        List of timestep values (descending, without trailing 0).
    """
    t_schedule_list = None

    if timesteps is not None:
        ts_list = [float(t) for t in list(timesteps)]
        # Remove trailing zeros
        while ts_list and abs(ts_list[-1]) < 1e-12:
            ts_list.pop()
        if len(ts_list) < 1:
            logger.warning("timesteps empty after removing zeros; using default shift=%s", shift)
        else:
            if is_turbo:
                if len(ts_list) > 20:
                    logger.warning("timesteps length=%d > 20; truncating", len(ts_list))
                    ts_list = ts_list[:20]
                # Turbo supports a discrete valid schedule set.
                mapped = [min(VALID_TIMESTEPS, key=lambda x, t=t: abs(x - t)) for t in ts_list]
                t_schedule_list = mapped
            else:
                t_schedule_list = ts_list

    if t_schedule_list is None:
        if is_turbo:
            original_shift = shift
            shift = min(VALID_SHIFTS, key=lambda x: abs(x - shift))
            if original_shift != shift:
                logger.warning("shift=%.2f rounded to nearest valid shift=%.1f", original_shift, shift)
            t_schedule_list = SHIFT_TIMESTEPS[shift]
        else:
            # Base model schedule parity with PyTorch generate_audio():
            # t = linspace(1.0, 0.0, infer_steps + 1), then optional shift.
            infer_steps = max(1, int(infer_steps))
            t = np.linspace(1.0, 0.0, infer_steps + 1, dtype=np.float32)
            if shift != 1.0:
                t = shift * t / (1 + (shift - 1) * t)
            t_schedule_list = [float(x) for x in t[:-1]]

    return t_schedule_list


def mlx_generate_diffusion(
    mlx_decoder,
    encoder_hidden_states_np: np.ndarray,
    context_latents_np: np.ndarray,
    src_latents_shape: Tuple[int, ...],
    src_latents_np: Optional[np.ndarray] = None,
    seed: Optional[Union[int, List[int]]] = None,
    infer_method: str = "ode",
    infer_steps: int = 8,
    shift: float = 3.0,
    timesteps: Optional[list] = None,
    is_turbo: bool = True,
    guidance_scale: float = 1.0,
    null_condition_emb_np: Optional[np.ndarray] = None,
    cfg_interval_start: float = 0.0,
    cfg_interval_end: float = 1.0,
    audio_cover_strength: float = 1.0,
    cover_noise_strength: float = 0.0,
    use_adg: bool = False,
    encoder_hidden_states_non_cover_np: Optional[np.ndarray] = None,
    context_latents_non_cover_np: Optional[np.ndarray] = None,
    compile_model: bool = False,
    disable_tqdm: bool = False,
) -> Dict[str, object]:
    """Run the complete MLX diffusion loop.

    This is the core generation function.  It accepts numpy arrays (converted
    from PyTorch tensors by the handler) and returns numpy arrays that the
    handler converts back to PyTorch.

    Args:
        mlx_decoder: ``MLXDiTDecoder`` instance with loaded weights.
        encoder_hidden_states_np: [B, enc_L, D] from prepare_condition (numpy).
        context_latents_np: [B, T, C] from prepare_condition (numpy).
        src_latents_shape: shape tuple [B, T, 64] for noise generation.
        src_latents_np: Optional source latents [B, T, C] for cover renoise start.
        seed: random seed (int, list[int], or None).
        infer_method: "ode" or "sde".
        infer_steps: Requested diffusion step count.
        shift: timestep shift factor.
        timesteps: optional custom timestep list.
        is_turbo: Whether the loaded DiT config is turbo.
        guidance_scale: CFG guidance strength (>1 enables guidance).
        null_condition_emb_np: [1, 1, D] unconditional embedding for CFG.
        cfg_interval_start: CFG start timestep ratio.
        cfg_interval_end: CFG end timestep ratio.
        audio_cover_strength: cover strength (0-1).
        cover_noise_strength: cover renoise strength (0-1, 1 = closest to source).
        use_adg: When requested, MLX falls back to plain CFG because the
            runtime-specific ADG path is not implemented here.
        encoder_hidden_states_non_cover_np: optional [B, enc_L, D] for non-cover.
        context_latents_non_cover_np: optional [B, T, C] for non-cover.
        compile_model: If True, compile the decoder step with ``mx.compile``
            for kernel fusion. Cross-attention KV caching is disabled under
            compilation; the fused kernels typically offset this cost.
            Falls back to uncompiled path on failure.
        disable_tqdm: If True, suppress the diffusion progress bar.

    Returns:
        Dict with ``"target_latents"`` (numpy) and ``"time_costs"`` dict.
    """
    import mlx.core as mx
    from .dit_model import MLXCrossAttentionCache

    time_costs = {}
    total_start = time.time()

    # Convert numpy arrays to MLX
    enc_hs = mx.array(encoder_hidden_states_np)
    ctx = mx.array(context_latents_np)

    enc_hs_nc = mx.array(encoder_hidden_states_non_cover_np) if encoder_hidden_states_non_cover_np is not None else None
    ctx_nc = mx.array(context_latents_non_cover_np) if context_latents_non_cover_np is not None else None

    bsz = src_latents_shape[0]
    T = src_latents_shape[1]
    C = src_latents_shape[2]

    do_cfg = guidance_scale > 1.0 and null_condition_emb_np is not None
    if guidance_scale > 1.0 and null_condition_emb_np is None:
        logger.warning(
            "[MLX-DiT] guidance_scale=%.3f requested but null_condition_emb is unavailable; sampling without CFG.",
            float(guidance_scale),
        )
    if use_adg and do_cfg:
        logger.warning(
            "[MLX-DiT] use_adg=True requested on the native MLX path; falling back to basic CFG guidance.",
        )

    if do_cfg:
        null_cond = mx.array(null_condition_emb_np)
        enc_hs = mx.concatenate([enc_hs, mx.broadcast_to(null_cond, enc_hs.shape)], axis=0)
        ctx = mx.concatenate([ctx, ctx], axis=0)
        if enc_hs_nc is not None:
            enc_hs_nc = mx.concatenate([enc_hs_nc, mx.broadcast_to(null_cond, enc_hs_nc.shape)], axis=0)
        if ctx_nc is not None:
            ctx_nc = mx.concatenate([ctx_nc, ctx_nc], axis=0)
    momentum_state: Optional[Dict[str, object]] = {"momentum": -0.75} if do_cfg and not use_adg else None

    # ---- Noise preparation ----
    if seed is None:
        noise = mx.random.normal((bsz, T, C))
    elif isinstance(seed, list):
        parts = []
        for s in seed:
            if s is None or s < 0:
                parts.append(mx.random.normal((1, T, C)))
            else:
                key = mx.random.key(int(s))
                parts.append(mx.random.normal((1, T, C), key=key))
        noise = mx.concatenate(parts, axis=0)
    else:
        key = mx.random.key(int(seed))
        noise = mx.random.normal((bsz, T, C), key=key)

    # ---- Timestep schedule ----
    t_schedule_list = get_timestep_schedule(
        shift=shift,
        timesteps=timesteps,
        infer_steps=infer_steps,
        is_turbo=is_turbo,
    )
    num_steps = len(t_schedule_list)
    if num_steps > 0:
        logger.info(
            "[MLX-DiT] Timestep schedule: steps=%d requested=%d turbo=%s shift=%.3f first=%.6f last=%.6f",
            num_steps,
            int(infer_steps),
            bool(is_turbo),
            float(shift),
            float(t_schedule_list[0]),
            float(t_schedule_list[-1]),
        )
    else:
        logger.warning(
            "[MLX-DiT] Empty timestep schedule (requested=%d, turbo=%s, shift=%.3f).",
            int(infer_steps),
            bool(is_turbo),
            float(shift),
        )

    cover_steps = int(num_steps * audio_cover_strength)

    # Cover noise initialization parity with PyTorch generate_audio():
    # - cover_noise_strength=0.0 -> pure noise start (default)
    # - cover_noise_strength=1.0 -> closest start to source latents
    if cover_noise_strength > 0.0 and src_latents_np is not None and num_steps > 0:
        effective_noise_level = 1.0 - float(cover_noise_strength)
        nearest_t = min(t_schedule_list, key=lambda x: abs(float(x) - effective_noise_level))
        start_idx = t_schedule_list.index(nearest_t)
        src_latents_mx = mx.array(src_latents_np)
        xt = nearest_t * noise + (1.0 - nearest_t) * src_latents_mx
        t_schedule_list = t_schedule_list[start_idx:]
        num_steps = len(t_schedule_list)
        cover_steps = int(num_steps * audio_cover_strength)
        logger.info(
            "[MLX-DiT] Cover mode: cover_noise_strength=%.4f effective_noise_level=%.4f nearest_t=%.6f remaining_steps=%d",
            float(cover_noise_strength),
            float(effective_noise_level),
            float(nearest_t),
            int(num_steps),
        )
    else:
        xt = noise
        if cover_noise_strength > 0.0 and src_latents_np is None:
            logger.warning(
                "[MLX-DiT] cover_noise_strength=%.4f requested but src_latents_np missing; using pure-noise start.",
                float(cover_noise_strength),
            )

    # ---- Prepare decoder step (compiled or plain with KV cache) ----
    # When compiled, we wrap the decoder call in mx.compile for kernel fusion.
    # Cross-attention KV caching is incompatible with mx.compile graph tracing
    # (cache uses Python dicts with dynamic branching), so it is disabled.
    # The kernel fusion from compilation typically offsets this cost.
    _compiled_step = None
    if compile_model:
        def _raw_step(xt, t, tr, enc, ctx):
            vt, _ = mlx_decoder(
                hidden_states=xt, timestep=t, timestep_r=tr,
                encoder_hidden_states=enc, context_latents=ctx,
                cache=None, use_cache=False,
            )
            return vt

        try:
            _compiled_step = mx.compile(_raw_step)
            logger.info("[MLX-DiT] Diffusion step compiled with mx.compile().")
        except Exception as exc:
            logger.warning(
                "[MLX-DiT] mx.compile() failed (%s); using uncompiled path.", exc
            )

    cache = MLXCrossAttentionCache() if (_compiled_step is None and not do_cfg) else None
    diff_start = time.time()
    switched_to_non_cover = False

    for step_idx in tqdm(range(num_steps), desc="MLX DiT diffusion", disable=disable_tqdm):
        current_t = t_schedule_list[step_idx]
        t_curr = mx.full((bsz,), current_t)

        # Switch to non-cover conditions when appropriate
        if step_idx >= cover_steps and not switched_to_non_cover:
            switched_to_non_cover = True
            if enc_hs_nc is not None:
                enc_hs = enc_hs_nc
                ctx = ctx_nc
            if cache is not None:
                cache = MLXCrossAttentionCache()

        model_xt = mx.concatenate([xt, xt], axis=0) if do_cfg else xt

        if _compiled_step is not None:
            model_t = mx.full((model_xt.shape[0],), current_t)
            vt = _compiled_step(model_xt, model_t, model_t, enc_hs, ctx)
        else:
            vt, cache = mlx_decoder(
                hidden_states=model_xt,
                timestep=mx.full((model_xt.shape[0],), current_t),
                timestep_r=mx.full((model_xt.shape[0],), current_t),
                encoder_hidden_states=enc_hs,
                context_latents=ctx,
                cache=cache,
                use_cache=cache is not None,
            )

        # Evaluate to ensure computation is complete before next step
        mx.eval(vt)

        if do_cfg:
            pred_cond = vt[:bsz]
            pred_uncond = vt[bsz:]
            apply_cfg = cfg_interval_start <= float(current_t) <= cfg_interval_end
            if apply_cfg:
                if use_adg:
                    vt = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    vt = _mlx_apg_forward(
                        pred_cond,
                        pred_uncond,
                        guidance_scale=guidance_scale,
                        momentum_state=momentum_state,
                    )
            else:
                vt = pred_cond
            mx.eval(vt)

        # Final step: compute x0
        if step_idx == num_steps - 1:
            t_unsq = mx.expand_dims(mx.expand_dims(t_curr, axis=-1), axis=-1)
            xt = xt - vt * t_unsq
            mx.eval(xt)
        else:
            # ODE / SDE update
            next_t = t_schedule_list[step_idx + 1]
            if infer_method == "sde":
                t_unsq = mx.expand_dims(mx.expand_dims(t_curr, axis=-1), axis=-1)
                pred_clean = xt - vt * t_unsq
                # Re-noise with next timestep
                new_noise = mx.random.normal(xt.shape)
                xt = next_t * new_noise + (1.0 - next_t) * pred_clean
            else:
                # ODE Euler step: x_{t+1} = x_t - v_t * dt
                dt = current_t - next_t
                dt_arr = mx.full((bsz, 1, 1), dt)
                xt = xt - vt * dt_arr

            mx.eval(xt)

    diff_end = time.time()
    total_end = time.time()

    time_costs["diffusion_time_cost"] = diff_end - diff_start
    time_costs["diffusion_per_step_time_cost"] = time_costs["diffusion_time_cost"] / max(num_steps, 1)
    time_costs["num_steps"] = int(num_steps)
    time_costs["total_time_cost"] = total_end - total_start

    # Convert result back to numpy
    result_np = np.array(xt)
    return {
        "target_latents": result_np,
        "time_costs": time_costs,
    }
