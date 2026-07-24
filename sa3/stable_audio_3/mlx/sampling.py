from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

from stable_audio_3.mlx.runtime import import_mlx_core


@dataclass(frozen=True)
class RFSchedulePreview:
    objective: str
    sampler_type: str
    steps: int
    sigma_max: float
    values: tuple[float, ...]


@dataclass(frozen=True)
class DistributionShiftSpec:
    kind: str
    params: dict[str, float | int | bool]


def default_sampler_type_for_objective(diffusion_objective: str) -> str:
    if diffusion_objective == "rf_denoiser":
        return "pingpong"
    if diffusion_objective == "rectified_flow":
        return "euler"
    if diffusion_objective == "v":
        return "dpmpp-3m-sde"
    return "unsupported"


def make_distribution_shift_spec(kind: str, **params) -> DistributionShiftSpec | None:
    kind = str(kind).lower()
    if kind in {"default", ""}:
        raise ValueError("'default' needs a model config or torch object to resolve a concrete shift.")
    if kind in {"none", "identity"}:
        return DistributionShiftSpec("none", {})
    if kind == "full":
        return DistributionShiftSpec(
            "full",
            {
                "base_shift": float(params.get("base_shift", 0.5)),
                "max_shift": float(params.get("max_shift", 1.15)),
                "min_length": int(params.get("min_length", 256)),
                "max_length": int(params.get("max_length", 4096)),
                "use_sine": bool(params.get("use_sine", False)),
            },
        )
    if kind == "flux":
        return DistributionShiftSpec(
            "flux",
            {
                "min_length": int(params.get("min_length", 256)),
                "max_length": int(params.get("max_length", 4096)),
                "alpha_min": float(params.get("alpha_min", 1.0)),
                "alpha_max": float(params.get("alpha_max", 1.0)),
            },
        )
    if kind == "logsnr":
        return DistributionShiftSpec(
            "logsnr",
            {
                "anchor_length": int(params.get("anchor_length", 2000)),
                "anchor_logsnr": float(params.get("anchor_logsnr", -6.2)),
                "rate": float(params.get("rate", 1.0)),
                "logsnr_end": float(params.get("logsnr_end", 2.0)),
            },
        )
    raise ValueError(f"Unknown distribution shift kind: {kind!r}")


def distribution_shift_spec_from_options(options: dict[str, tp.Any] | None) -> DistributionShiftSpec | None:
    if options is None:
        return make_distribution_shift_spec("logsnr", rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)
    kind = str(options.get("type", "full")).lower()
    params = {key: value for key, value in options.items() if key != "type"}
    return make_distribution_shift_spec(kind, **params)


def distribution_shift_spec_from_model_config(model_config: dict[str, tp.Any]) -> DistributionShiftSpec | None:
    diffusion = model_config.get("model", {}).get("diffusion", {})
    if diffusion.get("sampling_distribution_shift_options") is not None:
        return distribution_shift_spec_from_options(diffusion.get("sampling_distribution_shift_options"))
    if diffusion.get("distribution_shift_options") is not None:
        return distribution_shift_spec_from_options(diffusion.get("distribution_shift_options"))
    return distribution_shift_spec_from_options(None)


def sampling_distribution_shift_spec_from_model_config(
    model_config: dict[str, tp.Any],
) -> DistributionShiftSpec | None:
    """Reproduce the torch model's inference-time ``sampling_dist_shift``.

    ``StableAudioDiffusionCond`` uses ``sampling_distribution_shift_options``
    when present and otherwise falls back to a sequence-length-invariant LogSNR
    shift. It deliberately does **not** fall back to
    ``distribution_shift_options``, which is the *training* schedule.

    ``distribution_shift_spec_from_model_config`` does fall through to the
    training options, so it is the wrong helper for inference defaults: for
    SA3 medium, whose ``sampling_distribution_shift_options`` is null, the two
    disagree ('logsnr' versus 'full').
    """

    diffusion = model_config.get("model", {}).get("diffusion", {})
    options = diffusion.get("sampling_distribution_shift_options")
    if options is not None:
        return distribution_shift_spec_from_options(options)
    return make_distribution_shift_spec(
        "logsnr",
        anchor_length=2000,
        anchor_logsnr=-6.2,
        rate=0,
        logsnr_end=2.0,
    )


def training_distribution_shift_spec_from_model_config(
    model_config: dict[str, tp.Any],
) -> DistributionShiftSpec | None:
    diffusion = model_config.get("model", {}).get("diffusion", {})
    options = diffusion.get("distribution_shift_options")
    if options is None:
        return None
    return distribution_shift_spec_from_options(options)


def distribution_shift_spec_from_object(value: tp.Any) -> DistributionShiftSpec | None:
    if value is None:
        return None
    if isinstance(value, DistributionShiftSpec):
        return value
    if isinstance(value, str):
        return make_distribution_shift_spec(value)
    if isinstance(value, dict):
        return distribution_shift_spec_from_options(value)

    class_name = value.__class__.__name__
    if class_name == "IdentityDistributionShift":
        return make_distribution_shift_spec("none")
    if class_name == "DistributionShift":
        return make_distribution_shift_spec(
            "full",
            base_shift=getattr(value, "base_shift", 0.5),
            max_shift=getattr(value, "max_shift", 1.15),
            min_length=getattr(value, "min_length", 256),
            max_length=getattr(value, "max_length", 4096),
            use_sine=getattr(value, "use_sine", False),
        )
    if class_name == "FluxDistributionShift":
        return make_distribution_shift_spec(
            "flux",
            min_length=getattr(value, "min_length", 256),
            max_length=getattr(value, "max_length", 4096),
            alpha_min=getattr(value, "alpha_min", 1.0),
            alpha_max=getattr(value, "alpha_max", 1.0),
        )
    if class_name == "LogSNRShift":
        return make_distribution_shift_spec(
            "logsnr",
            anchor_length=getattr(value, "anchor_length", 2000),
            anchor_logsnr=getattr(value, "anchor_logsnr", -6.2),
            rate=getattr(value, "rate", 1.0),
            logsnr_end=getattr(value, "logsnr_end", 2.0),
        )
    raise TypeError(f"Cannot convert {class_name!r} to an MLX distribution shift spec.")


def distribution_shift_spec_to_jsonable(spec: DistributionShiftSpec | None) -> dict[str, tp.Any] | None:
    if spec is None:
        return None
    return {"kind": spec.kind, "params": dict(spec.params)}


def _coerce_shift_spec(value: tp.Any) -> DistributionShiftSpec | None:
    return distribution_shift_spec_from_object(value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(float(value), float(minimum)), float(maximum))


def _as_seq_len_list(effective_seq_len) -> list[float] | None:
    if effective_seq_len is None:
        return None
    if isinstance(effective_seq_len, (list, tuple)):
        return [float(value) for value in effective_seq_len]
    try:
        import numpy as np

        if isinstance(effective_seq_len, np.ndarray):
            return [float(value) for value in effective_seq_len.reshape(-1).tolist()]
    except ModuleNotFoundError:
        pass
    if hasattr(effective_seq_len, "tolist"):
        value = effective_seq_len.tolist()
        if isinstance(value, list):
            return [float(item) for item in value]
        return [float(value)]
    return [float(effective_seq_len)]


def _shift_scalar_timestep(t: float, seq_len: float, spec: DistributionShiftSpec) -> float:
    kind = spec.kind
    if kind == "none":
        return float(t)
    if kind == "full":
        min_length = float(spec.params["min_length"])
        max_length = float(spec.params["max_length"])
        seq_len = _clamp(seq_len, min_length, max_length)
        base_shift = float(spec.params["base_shift"])
        max_shift = float(spec.params["max_shift"])
        mu = -(
            base_shift
            + (max_shift - base_shift)
            * (seq_len - min_length)
            / (max_length - min_length)
        )
        if t <= 0.0:
            shifted = 0.0
        elif t >= 1.0:
            shifted = 1.0
        else:
            shifted = 1.0 - math.exp(mu) / (
                math.exp(mu) + (1.0 / (1.0 - t) - 1.0)
            )
        if bool(spec.params.get("use_sine", False)):
            shifted = math.sin(shifted * math.pi / 2.0)
        return float(shifted)
    if kind == "flux":
        min_length = float(spec.params["min_length"])
        max_length = float(spec.params["max_length"])
        seq_len = _clamp(seq_len, min_length, max_length)
        alpha_min = max(float(spec.params["alpha_min"]), 1e-8)
        alpha_max = max(float(spec.params["alpha_max"]), 1e-8)
        log_min_seq = math.log(min_length)
        log_max_seq = math.log(max_length)
        if log_max_seq == log_min_seq:
            log_max_seq += 1e-8
        frac = (math.log(seq_len) - log_min_seq) / (log_max_seq - log_min_seq)
        log_alpha = math.log(alpha_min) + frac * (math.log(alpha_max) - math.log(alpha_min))
        alpha = math.exp(log_alpha)
        return float(alpha * t / (1.0 + (alpha - 1.0) * t))
    if kind == "logsnr":
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        anchor_length = float(spec.params["anchor_length"])
        anchor_logsnr = float(spec.params["anchor_logsnr"])
        rate = float(spec.params["rate"])
        logsnr_end = float(spec.params["logsnr_end"])
        logsnr_start = anchor_logsnr - rate * math.log2(float(seq_len) / anchor_length)
        logsnr = logsnr_end - t * (logsnr_end - logsnr_start)
        return float(1.0 / (1.0 + math.exp(logsnr)))
    raise ValueError(f"Unsupported distribution shift kind: {kind!r}")


def shift_schedule_values(
    values: tp.Sequence[float],
    *,
    dist_shift: tp.Any,
    effective_seq_len=None,
    fallback_seq_len: int | None = None,
    sigma_max: float | None = None,
) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
    spec = _coerce_shift_spec(dist_shift)
    base_values = tuple(float(value) for value in values)
    if spec is None or spec.kind == "none":
        return base_values

    seq_lens = _as_seq_len_list(effective_seq_len)
    if seq_lens is None:
        if fallback_seq_len is None:
            raise ValueError("fallback_seq_len is required when effective_seq_len is not provided.")
        seq_lens = [float(fallback_seq_len)]

    sigma_max_value = float(base_values[0] if sigma_max is None else sigma_max)
    shifted_rows = []
    for seq_len in seq_lens:
        row = [_shift_scalar_timestep(value, seq_len, spec) for value in base_values]
        row[0] = sigma_max_value
        shifted_rows.append(tuple(row))

    if len(shifted_rows) == 1 and _as_seq_len_list(effective_seq_len) is None:
        return shifted_rows[0]
    if len(shifted_rows) == 1 and effective_seq_len is not None and not isinstance(effective_seq_len, (list, tuple)):
        return shifted_rows[0]
    return tuple(shifted_rows)


def shift_timestep_values(
    values: tp.Sequence[float],
    *,
    dist_shift: tp.Any,
    effective_seq_len: float | tp.Sequence[float],
) -> tuple[float, ...]:
    spec = _coerce_shift_spec(dist_shift)
    base_values = tuple(float(value) for value in values)
    if spec is None or spec.kind == "none":
        return base_values

    seq_lens = _as_seq_len_list(effective_seq_len)
    if seq_lens is None:
        raise ValueError("effective_seq_len is required for timestep distribution shifting.")
    if len(seq_lens) == 1:
        seq_lens *= len(base_values)
    if len(seq_lens) != len(base_values):
        raise ValueError(
            "effective_seq_len must contain one value or match the timestep batch size."
        )
    return tuple(
        _shift_scalar_timestep(value, seq_len, spec)
        for value, seq_len in zip(base_values, seq_lens, strict=True)
    )


def make_rf_schedule_values(
    steps: int,
    sigma_max: float = 1.0,
    *,
    full_noise_anchor_logsnr: float = -6.0,
    logsnr_end: float = 2.0,
) -> tuple[float, ...]:
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    sigma_max = float(sigma_max)
    if sigma_max <= 0.0:
        raise ValueError(f"sigma_max must be > 0, got {sigma_max}")

    sigma_max = min(sigma_max, 1.0)
    if sigma_max < 1.0:
        logsnr_max = math.log(((1.0 - sigma_max) / sigma_max) + 1e-6)
    else:
        logsnr_max = float(full_noise_anchor_logsnr)

    out = [sigma_max]
    for step_index in range(1, steps):
        alpha = step_index / steps
        logsnr = logsnr_max + (logsnr_end - logsnr_max) * alpha
        value = 1.0 / (1.0 + math.exp(logsnr))
        out.append(float(value))
    out.append(0.0)
    return tuple(out)


def make_rf_schedule_mlx(
    steps: int,
    sigma_max: float = 1.0,
    *,
    dtype_name: str = "float32",
):
    mx = import_mlx_core(required=True)
    dtype = getattr(mx, dtype_name)
    values = make_rf_schedule_values(steps=steps, sigma_max=sigma_max)
    return mx.array(values, dtype=dtype)


def make_linear_schedule_values(
    steps: int,
    sigma_max: float = 1.0,
    *,
    include_endpoint: bool = True,
) -> tuple[float, ...]:
    """Match the torch RF sampler's default unshifted timestep schedule."""
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    sigma_max = float(sigma_max)
    if sigma_max <= 0.0:
        raise ValueError(f"sigma_max must be > 0, got {sigma_max}")

    n_points = steps + 1 if include_endpoint else steps
    denominator = steps if include_endpoint else steps + 1
    return tuple(
        sigma_max + (0.0 - sigma_max) * (index / denominator)
        for index in range(n_points)
    )


def make_shifted_linear_schedule_values(
    steps: int,
    sigma_max: float = 1.0,
    *,
    dist_shift: tp.Any = None,
    effective_seq_len=None,
    fallback_seq_len: int | None = None,
    include_endpoint: bool = True,
) -> tuple[float, ...] | tuple[tuple[float, ...], ...]:
    values = make_linear_schedule_values(
        steps=steps,
        sigma_max=sigma_max,
        include_endpoint=include_endpoint,
    )
    return shift_schedule_values(
        values,
        dist_shift=dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=fallback_seq_len,
        sigma_max=sigma_max,
    )


def make_linear_schedule_mlx(
    steps: int,
    sigma_max: float = 1.0,
    *,
    dtype_name: str = "float32",
    include_endpoint: bool = True,
):
    mx = import_mlx_core(required=True)
    dtype = getattr(mx, dtype_name)
    values = make_linear_schedule_values(
        steps=steps,
        sigma_max=sigma_max,
        include_endpoint=include_endpoint,
    )
    return mx.array(values, dtype=dtype)


def make_shifted_linear_schedule_mlx(
    steps: int,
    sigma_max: float = 1.0,
    *,
    dist_shift: tp.Any = None,
    effective_seq_len=None,
    fallback_seq_len: int | None = None,
    dtype_name: str = "float32",
    include_endpoint: bool = True,
):
    mx = import_mlx_core(required=True)
    dtype = getattr(mx, dtype_name)
    values = make_shifted_linear_schedule_values(
        steps=steps,
        sigma_max=sigma_max,
        dist_shift=dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=fallback_seq_len,
        include_endpoint=include_endpoint,
    )
    return mx.array(values, dtype=dtype)


def effective_latent_lengths_from_durations(
    durations: tp.Sequence[float],
    *,
    sample_rate: int,
    downsampling_ratio: int,
) -> tuple[int, ...]:
    return tuple(
        int(math.ceil(int(float(duration) * int(sample_rate)) / int(downsampling_ratio)))
        for duration in durations
    )


def padding_lengths_from_effective_lengths(
    effective_seq_len,
    latent_length: int,
    *,
    headroom_tokens: int = 0,
) -> tuple[int, ...] | None:
    seq_lens = _as_seq_len_list(effective_seq_len)
    if seq_lens is None:
        return None

    latent_length = int(latent_length)
    headroom_tokens = max(int(headroom_tokens), 0)
    return tuple(
        min(max(int(math.ceil(seq_len)) + headroom_tokens, 0), latent_length)
        for seq_len in seq_lens
    )


def create_padding_mask_from_lengths(valid_lengths: tp.Sequence[int], latent_length: int):
    mx = import_mlx_core(required=True)
    latent_length = int(latent_length)
    lengths = mx.array([int(length) for length in valid_lengths], dtype=mx.int32)
    positions = mx.arange(latent_length, dtype=mx.int32)[None, :]
    return positions < lengths[:, None]


def create_padding_mask_for_effective_lengths(
    effective_seq_len,
    latent_length: int,
    *,
    headroom_tokens: int = 0,
):
    valid_lengths = padding_lengths_from_effective_lengths(
        effective_seq_len,
        latent_length,
        headroom_tokens=headroom_tokens,
    )
    if valid_lengths is None:
        return None
    return create_padding_mask_from_lengths(valid_lengths, latent_length)


def _as_batch_timestep(t, x):
    mx = import_mlx_core(required=True)
    return t * mx.ones((int(x.shape[0]),), dtype=x.dtype)


def _broadcast_timestep(t):
    if getattr(t, "ndim", 0) == 0:
        return t
    return t[:, None, None]


def sample_rf_euler_mlx(
    model: tp.Callable[..., tp.Any],
    x,
    sigmas,
    *,
    callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    **extra_args: tp.Any,
):
    """Euler RF latent sampler in MLX, for latent-space smoke tests."""
    mx = import_mlx_core(required=True)
    per_element_schedule = int(sigmas.ndim) == 2
    num_steps = int(sigmas.shape[-1]) - 1

    for i in range(num_steps):
        if per_element_schedule:
            t_curr = sigmas[:, i].astype(x.dtype)
            t_next = sigmas[:, i + 1].astype(x.dtype)
            dt = (t_next - t_curr)[:, None, None]
            t_curr_tensor = t_curr
            t_next_tensor = t_next
        else:
            t_curr = sigmas[i].astype(x.dtype)
            t_next = sigmas[i + 1].astype(x.dtype)
            dt = t_next - t_curr
            t_curr_tensor = _as_batch_timestep(t_curr, x)
            t_next_tensor = _as_batch_timestep(t_next, x)

        velocity = model(x, t_curr_tensor, **extra_args)
        x = x + dt * velocity
        if callback is not None:
            denoised = x - _broadcast_timestep(t_next_tensor) * velocity
            mx.eval(x, denoised)
            callback(
                {
                    "x": x,
                    "t": t_curr_tensor,
                    "sigma": t_curr_tensor,
                    "i": i,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "denoised": denoised,
                }
            )

    return x


def sample_rf_rk4_mlx(
    model: tp.Callable[..., tp.Any],
    x,
    sigmas,
    *,
    callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    **extra_args: tp.Any,
):
    """4th-order Runge-Kutta RF latent sampler in MLX."""
    mx = import_mlx_core(required=True)
    per_element_schedule = int(sigmas.ndim) == 2
    num_steps = int(sigmas.shape[-1]) - 1

    for i in range(num_steps):
        if per_element_schedule:
            t_curr = sigmas[:, i].astype(x.dtype)
            t_next = sigmas[:, i + 1].astype(x.dtype)
            dt = t_next - t_curr
            dt_broadcast = dt[:, None, None]
            t_curr_tensor = t_curr
            t_mid_tensor = t_curr + dt / 2.0
            t_next_eval = mx.maximum(t_next, mx.array(1e-5, dtype=x.dtype))
        else:
            t_curr = sigmas[i].astype(x.dtype)
            t_next = sigmas[i + 1].astype(x.dtype)
            dt = t_next - t_curr
            dt_broadcast = dt
            t_curr_tensor = _as_batch_timestep(t_curr, x)
            t_mid_tensor = _as_batch_timestep(t_curr + dt / 2.0, x)
            t_next_eval = _as_batch_timestep(mx.maximum(t_next, mx.array(1e-5, dtype=x.dtype)), x)

        k1 = model(x, t_curr_tensor, **extra_args)
        k2 = model(x + dt_broadcast / 2.0 * k1, t_mid_tensor, **extra_args)
        k3 = model(x + dt_broadcast / 2.0 * k2, t_mid_tensor, **extra_args)
        k4 = model(x + dt_broadcast * k3, t_next_eval, **extra_args)
        x = x + dt_broadcast / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if callback is not None:
            t_next_tensor = t_next if per_element_schedule else _as_batch_timestep(t_next, x)
            denoised = x - _broadcast_timestep(t_next_tensor) * k4
            mx.eval(x, denoised)
            callback(
                {
                    "x": x,
                    "t": t_curr_tensor,
                    "sigma": t_curr_tensor,
                    "i": i,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "denoised": denoised,
                }
            )

    return x


def sample_rf_dpmpp_mlx(
    model: tp.Callable[..., tp.Any],
    x,
    sigmas,
    *,
    callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    **extra_args: tp.Any,
):
    """DPM-Solver++ RF latent sampler in MLX."""
    mx = import_mlx_core(required=True)
    per_element_schedule = int(sigmas.ndim) == 2
    num_steps = int(sigmas.shape[-1]) - 1
    old_denoised = None

    def log_snr(t):
        return mx.log(mx.maximum(1.0 - t, 1e-10) / mx.maximum(t, 1e-10))

    for i in range(num_steps):
        if per_element_schedule:
            t_curr = sigmas[:, i].astype(x.dtype)
            t_next = sigmas[:, i + 1].astype(x.dtype)
            t_prev = sigmas[:, i - 1].astype(x.dtype) if i > 0 else None
            t_curr_broadcast = t_curr[:, None, None]
            t_next_broadcast = t_next[:, None, None]
            t_curr_tensor = t_curr
            t_prev_broadcast = t_prev[:, None, None] if t_prev is not None else None
        else:
            t_curr = sigmas[i].astype(x.dtype)
            t_next = sigmas[i + 1].astype(x.dtype)
            t_prev = sigmas[i - 1].astype(x.dtype) if i > 0 else None
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = _as_batch_timestep(t_curr, x)
            t_prev_broadcast = t_prev

        model_output = model(x, t_curr_tensor, **extra_args)
        denoised = x - t_curr_broadcast * model_output

        alpha_t = 1.0 - t_next_broadcast
        dt = t_next_broadcast - t_curr_broadcast
        dpmpp_coeff = dt / (
            mx.maximum(1.0 - t_next_broadcast, 1e-10) * mx.maximum(t_curr_broadcast, 1e-10)
        )

        is_first_step = old_denoised is None
        is_last_step = i == num_steps - 1
        if is_first_step or is_last_step:
            x = (
                t_next_broadcast / mx.maximum(t_curr_broadcast, 1e-10)
            ) * x - alpha_t * dpmpp_coeff * denoised
        else:
            h = log_snr(t_next_broadcast) - log_snr(t_curr_broadcast)
            h_last = log_snr(t_curr_broadcast) - log_snr(t_prev_broadcast)
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (
                1.0 / (2.0 * r)
            ) * old_denoised
            x = (
                t_next_broadcast / mx.maximum(t_curr_broadcast, 1e-10)
            ) * x - alpha_t * dpmpp_coeff * denoised_d

        old_denoised = denoised
        if callback is not None:
            mx.eval(x, denoised)
            callback(
                {
                    "x": x,
                    "i": i,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "t": t_curr_tensor,
                    "sigma": t_curr_tensor,
                    "denoised": denoised,
                }
            )

    return x


def sample_rf_pingpong_mlx(
    model: tp.Callable[..., tp.Any],
    x,
    sigmas,
    *,
    callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    **extra_args: tp.Any,
):
    """Ping-pong RF latent sampler in MLX, matching the distilled torch path.

    Optional latent-prefix continuation extras (popped from ``extra_args``):
        fixed_prefix_data: [B, C, T] clean latent values to pin under the mask.
        fixed_prefix_mask: [B, 1, T] 1 = fixed prefix token, 0 = generated.
        fixed_prefix_noise: [B, C, T] optional fixed noise path; randn_like if omitted.
    """
    mx = import_mlx_core(required=True)
    fixed_prefix_data = extra_args.pop("fixed_prefix_data", None)
    fixed_prefix_mask = extra_args.pop("fixed_prefix_mask", None)
    fixed_prefix_noise = extra_args.pop("fixed_prefix_noise", None)

    if fixed_prefix_data is None:
        if fixed_prefix_mask is not None or fixed_prefix_noise is not None:
            raise ValueError("fixed_prefix_mask/fixed_prefix_noise require fixed_prefix_data.")
    else:
        if fixed_prefix_mask is None:
            raise ValueError("fixed_prefix_data requires fixed_prefix_mask.")
        if fixed_prefix_data.dtype != x.dtype:
            fixed_prefix_data = fixed_prefix_data.astype(x.dtype)
        if fixed_prefix_noise is None:
            fixed_prefix_noise = mx.random.normal(fixed_prefix_data.shape, dtype=x.dtype)
        elif fixed_prefix_noise.dtype != x.dtype:
            fixed_prefix_noise = fixed_prefix_noise.astype(x.dtype)
        if fixed_prefix_mask.dtype != x.dtype:
            fixed_prefix_mask = fixed_prefix_mask.astype(x.dtype)
        mx.eval(fixed_prefix_data, fixed_prefix_noise, fixed_prefix_mask)

    def impose_prefix(cur_x, t_value):
        if fixed_prefix_data is None:
            return cur_x
        if int(getattr(t_value, "ndim", 0)) >= 1:
            t_broadcast = t_value.reshape((-1, 1, 1)).astype(cur_x.dtype)
        else:
            t_broadcast = t_value.astype(cur_x.dtype)
        prefix_x = fixed_prefix_data * (1.0 - t_broadcast) + fixed_prefix_noise * t_broadcast
        return cur_x * (1.0 - fixed_prefix_mask) + prefix_x * fixed_prefix_mask

    per_element_schedule = int(sigmas.ndim) == 2
    num_steps = int(sigmas.shape[-1]) - 1

    if fixed_prefix_data is not None:
        t0 = sigmas[:, 0] if per_element_schedule else sigmas[0]
        x = impose_prefix(x, t0)
        mx.eval(x)

    for i in range(num_steps):
        if per_element_schedule:
            t_curr = sigmas[:, i].astype(x.dtype)
            t_next = sigmas[:, i + 1].astype(x.dtype)
            t_curr_broadcast = t_curr[:, None, None]
            t_next_broadcast = t_next[:, None, None]
            t_curr_tensor = t_curr
        else:
            t_curr = sigmas[i].astype(x.dtype)
            t_next = sigmas[i + 1].astype(x.dtype)
            t_curr_broadcast = t_curr
            t_next_broadcast = t_next
            t_curr_tensor = _as_batch_timestep(t_curr, x)

        denoised = x - t_curr_broadcast * model(x, t_curr_tensor, **extra_args)
        if fixed_prefix_data is not None:
            denoised = denoised * (1.0 - fixed_prefix_mask) + fixed_prefix_data * fixed_prefix_mask
        x = (1.0 - t_next_broadcast) * denoised + t_next_broadcast * mx.random.normal(
            x.shape,
            dtype=x.dtype,
        )
        if fixed_prefix_data is not None:
            x = impose_prefix(x, t_next)
        if callback is not None:
            mx.eval(x, denoised)
            callback(
                {
                    "x": x,
                    "i": i,
                    "step": i + 1,
                    "total_steps": num_steps,
                    "t": t_curr_tensor,
                    "sigma": t_curr_tensor,
                    "sigma_hat": t_curr_tensor,
                    "denoised": denoised,
                }
            )

    return x


def sample_rf_latents_mlx(
    model: tp.Callable[..., tp.Any],
    noise,
    *,
    cond_inputs: dict[str, tp.Any],
    diffusion_objective: str,
    steps: int,
    cfg_scale: float = 1.0,
    sampler_type: str | None = None,
    init_data=None,
    init_noise_level: float = 1.0,
    dist_shift: tp.Any = None,
    effective_seq_len=None,
    seed: int | None = None,
    callback: tp.Callable[[dict[str, tp.Any]], None] | None = None,
    **model_kwargs: tp.Any,
):
    """Sample RF latents in MLX without decoding through SAME.

    This covers the SA3 Medium RF-family latent sampler paths with precomputed
    conditioning. SAME-L decode is handled by the caller.
    """
    mx = import_mlx_core(required=True)
    if diffusion_objective not in {"rf_denoiser", "rectified_flow"}:
        raise ValueError(f"MLX RF latent sampler only supports RF objectives, got {diffusion_objective!r}")

    if seed is not None:
        mx.random.seed(int(seed))

    sampler_type = sampler_type or default_sampler_type_for_objective(diffusion_objective)
    if model_kwargs.get("fixed_prefix_data") is not None and sampler_type != "pingpong":
        raise ValueError("fixed_prefix_data requires sampler_type='pingpong'.")
    sigma_max = float(init_noise_level) if init_data is not None else 1.0
    x = noise
    if init_data is not None:
        x = init_data * (1.0 - sigma_max) + noise * sigma_max

    sigma_values = make_shifted_linear_schedule_values(
        steps=steps,
        sigma_max=sigma_max,
        dist_shift=dist_shift,
        effective_seq_len=effective_seq_len,
        fallback_seq_len=int(x.shape[-1]),
    )
    sigmas = mx.array(sigma_values, dtype=x.dtype)
    extra_args = {
        **cond_inputs,
        "cfg_scale": cfg_scale,
        **model_kwargs,
    }

    if sampler_type == "pingpong":
        return sample_rf_pingpong_mlx(
            model,
            x,
            sigmas,
            callback=callback,
            **extra_args,
        )
    if sampler_type == "euler":
        return sample_rf_euler_mlx(
            model,
            x,
            sigmas,
            callback=callback,
            **extra_args,
        )
    if sampler_type == "rk4":
        return sample_rf_rk4_mlx(
            model,
            x,
            sigmas,
            callback=callback,
            **extra_args,
        )
    if sampler_type == "dpmpp":
        return sample_rf_dpmpp_mlx(
            model,
            x,
            sigmas,
            callback=callback,
            **extra_args,
        )

    raise ValueError(f"Unsupported MLX RF sampler type: {sampler_type!r}")


def preview_rf_schedule(
    diffusion_objective: str,
    *,
    steps: int,
    sigma_max: float = 1.0,
) -> RFSchedulePreview:
    return RFSchedulePreview(
        objective=diffusion_objective,
        sampler_type=default_sampler_type_for_objective(diffusion_objective),
        steps=int(steps),
        sigma_max=float(sigma_max),
        values=make_rf_schedule_values(steps=steps, sigma_max=sigma_max),
    )
