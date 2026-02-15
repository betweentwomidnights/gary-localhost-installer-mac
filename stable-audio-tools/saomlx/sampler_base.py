"""Sampler interfaces used by the MLX pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import numpy as np


@dataclass
class SamplerState:
    latents: object
    step: int


def make_rf_schedule(steps: int, sigma_max: float = 1.0, *, dtype=mx.float32) -> mx.array:
    """SAT-compatible rectified-flow schedule used by `sample_rf`."""
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    sigma_max = min(float(sigma_max), 1.0)
    if sigma_max < 1.0:
        logsnr_max = float(mx.log(((1.0 - sigma_max) / sigma_max) + 1e-6).item())
    else:
        logsnr_max = -6.0

    logsnr = mx.linspace(logsnr_max, 2.0, steps + 1, dtype=mx.float32)
    t = 1.0 / (1.0 + mx.exp(logsnr))
    t = t.astype(dtype)
    # Match SAT endpoint clamping.
    t = mx.concatenate([mx.array([sigma_max], dtype=dtype), t[1:-1], mx.array([0.0], dtype=dtype)], axis=0)
    return t


def sample_rf_euler_mlx(
    model_fn,
    x: mx.array,
    *,
    steps: int,
    sigma_max: float = 1.0,
    callback=None,
    **extra_args,
) -> mx.array:
    """MLX Euler solver for rectified-flow / rf_denoiser models."""
    t = make_rf_schedule(steps=steps, sigma_max=sigma_max, dtype=x.dtype)
    ts = mx.ones((x.shape[0],), dtype=x.dtype)
    h = x

    for i in range(int(t.shape[0]) - 1):
        t_curr = t[i]
        t_prev = t[i + 1]
        dt = t_prev - t_curr
        t_curr_tensor = t_curr * ts

        v = model_fn(h, t_curr_tensor, **extra_args)
        h = h + dt * v

        if callback is not None:
            denoised = h - t_prev * v
            callback(
                {
                    "x": h,
                    "t": t_curr,
                    "sigma": t_curr,
                    "i": i + 1,
                    "denoised": denoised,
                }
            )

    return h


def sample_rf_dpmpp_mlx(
    model_fn,
    x: mx.array,
    *,
    steps: int,
    sigma_max: float = 1.0,
    callback=None,
    **extra_args,
) -> mx.array:
    """DPM-Solver++ sampler for rectified-flow / rf_denoiser models."""
    t = make_rf_schedule(steps=steps, sigma_max=sigma_max, dtype=x.dtype)
    old_denoised = None

    # sample_flow_dpmpp from SAT:
    # log_snr(t) = ln((1 - t) / t)
    def log_snr(val: mx.array) -> mx.array:
        return mx.log((1.0 - val) / val)

    h = x
    for i in range(int(t.shape[0]) - 1):
        t_curr = t[i]
        t_next = t[i + 1]

        v = model_fn(h, t_curr * mx.ones((h.shape[0],), dtype=h.dtype), **extra_args)
        denoised = h - t_curr * v
        if callback is not None:
            callback({"x": h, "i": i, "t": t_curr, "sigma": t_curr, "denoised": denoised})

        alpha_t = 1.0 - t_next
        h_step = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or float(t_next.item()) == 0.0:
            h = (t_next / t_curr) * h - alpha_t * mx.expm1(-h_step) * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h_step
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
            h = (t_next / t_curr) * h - alpha_t * mx.expm1(-h_step) * denoised_d
        old_denoised = denoised
    return h


def _get_alphas_sigmas(t: mx.array) -> tuple[mx.array, mx.array]:
    return mx.cos(t * mx.pi / 2.0), mx.sin(t * mx.pi / 2.0)


def _append_dims(x: mx.array, target_dims: int) -> mx.array:
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}")
    if dims_to_append == 0:
        return x
    return x.reshape((*x.shape, *([1] * dims_to_append)))


def make_k_sigmas_polyexponential(
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    *,
    dtype=mx.float32,
) -> mx.array:
    """k-diffusion polyexponential schedule with final zero sigma."""
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError("sigma_min and sigma_max must be > 0")
    ramp = mx.linspace(1.0, 0.0, steps, dtype=mx.float32) ** float(rho)
    sigmas = mx.exp(ramp * (mx.log(float(sigma_max)) - mx.log(float(sigma_min))) + mx.log(float(sigma_min)))
    sigmas = sigmas.astype(dtype)
    return mx.concatenate([sigmas, mx.zeros((1,), dtype=dtype)], axis=0)


def _sigma_to_t(sigma: mx.array) -> mx.array:
    # Matches k-diffusion VDenoiser.sigma_to_t.
    return mx.arctan(sigma) / mx.pi * 2.0


def _to_d(x: mx.array, sigma: mx.array, denoised: mx.array) -> mx.array:
    """Convert denoiser output to Karras ODE derivative."""
    return (x - denoised) / _append_dims(sigma, x.ndim)


def _sigma_fn(t: mx.array) -> mx.array:
    return mx.exp(-t)


def _t_fn(sigma: mx.array) -> mx.array:
    return -mx.log(sigma)


def _get_ancestral_step(sigma_from: float, sigma_to: float, eta: float = 1.0) -> tuple[float, float]:
    """k-diffusion ancestral step helper."""
    if eta == 0.0:
        return float(sigma_to), 0.0
    numer = sigma_to * sigma_to * max(sigma_from * sigma_from - sigma_to * sigma_to, 0.0)
    denom = max(sigma_from * sigma_from, 1e-20)
    sigma_up = min(float(sigma_to), float(eta) * math.sqrt(max(numer / denom, 0.0)))
    sigma_down = math.sqrt(max(sigma_to * sigma_to - sigma_up * sigma_up, 0.0))
    return sigma_down, sigma_up


def _linear_multistep_coeff(order: int, sigmas: np.ndarray, i: int, j: int) -> float:
    """Numerically integrate LMS coefficient (matches k-diffusion)."""
    from scipy import integrate

    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau: float) -> float:
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - sigmas[i - k]) / (sigmas[i - j] - sigmas[i - k])
        return prod

    return float(integrate.quad(fn, float(sigmas[i]), float(sigmas[i + 1]), epsrel=1e-4)[0])


def _v_denoiser_forward(model_fn, x: mx.array, sigma: mx.array, **extra_args) -> mx.array:
    """k-diffusion VDenoiser wrapper around a v-objective model function."""
    sigma_data = 1.0
    sigma2 = sigma * sigma
    c_skip = (sigma_data**2) / (sigma2 + sigma_data**2)
    c_out = -sigma * sigma_data / mx.sqrt(sigma2 + sigma_data**2)
    c_in = 1.0 / mx.sqrt(sigma2 + sigma_data**2)

    c_skip_e = _append_dims(c_skip, x.ndim)
    c_out_e = _append_dims(c_out, x.ndim)
    c_in_e = _append_dims(c_in, x.ndim)
    t = _sigma_to_t(sigma)

    inner = model_fn(x * c_in_e, t, **extra_args)
    return inner * c_out_e + x * c_skip_e


def sample_k_heun_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    callback=None,
    **extra_args,
) -> mx.array:
    """Heun sampler (Algorithm 2 from Karras et al.) for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_i_val = float(sigma_i.item())
        gamma = (
            min(float(s_churn) / (int(sigmas.shape[0]) - 1), math.sqrt(2.0) - 1.0)
            if s_tmin <= sigma_i_val <= s_tmax
            else 0.0
        )
        sigma_hat = sigma_i * (1.0 + gamma)
        if gamma > 0.0:
            eps = mx.random.normal(x.shape, dtype=x.dtype) * float(s_noise)
            x = x + eps * mx.sqrt(mx.maximum(sigma_hat**2 - sigma_i**2, 0.0))

        denoised = _v_denoiser_forward(model_fn, x, sigma_hat * s_in, **extra_args)
        d = _to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})

        dt = sigma_next - sigma_hat
        if float(sigma_next.item()) == 0.0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = _v_denoiser_forward(model_fn, x_2, sigma_next * s_in, **extra_args)
            d_2 = _to_d(x_2, sigma_next, denoised_2)
            x = x + ((d + d_2) * 0.5) * dt
    return x


def sample_k_lms_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    order: int = 4,
    callback=None,
    **extra_args,
) -> mx.array:
    """Linear multistep sampler for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    sigmas_np = np.asarray(sigmas, dtype=np.float64)

    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)
    ds: list[mx.array] = []

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        denoised = _v_denoiser_forward(model_fn, x, sigma_i * s_in, **extra_args)
        d = _to_d(x, sigma_i, denoised)
        ds.append(d)
        if len(ds) > int(order):
            ds.pop(0)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})

        cur_order = min(i + 1, int(order))
        coeffs = [_linear_multistep_coeff(cur_order, sigmas_np, i, j) for j in range(cur_order)]
        update = mx.zeros_like(x)
        for coeff, d_term in zip(coeffs, reversed(ds)):
            update = update + d_term * float(coeff)
        x = x + update

    return x


def sample_k_dpm_2_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    callback=None,
    **extra_args,
) -> mx.array:
    """DPM-Solver-2 inspired sampler for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_i_val = float(sigma_i.item())
        gamma = (
            min(float(s_churn) / (int(sigmas.shape[0]) - 1), math.sqrt(2.0) - 1.0)
            if s_tmin <= sigma_i_val <= s_tmax
            else 0.0
        )
        sigma_hat = sigma_i * (1.0 + gamma)
        if gamma > 0.0:
            eps = mx.random.normal(x.shape, dtype=x.dtype) * float(s_noise)
            x = x + eps * mx.sqrt(mx.maximum(sigma_hat**2 - sigma_i**2, 0.0))

        denoised = _v_denoiser_forward(model_fn, x, sigma_hat * s_in, **extra_args)
        d = _to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_hat, "denoised": denoised})

        if float(sigma_next.item()) == 0.0:
            dt = sigma_next - sigma_hat
            x = x + d * dt
        else:
            sigma_mid = mx.exp((mx.log(sigma_hat) + mx.log(sigma_next)) * 0.5)
            dt_1 = sigma_mid - sigma_hat
            dt_2 = sigma_next - sigma_hat
            x_2 = x + d * dt_1
            denoised_2 = _v_denoiser_forward(model_fn, x_2, sigma_mid * s_in, **extra_args)
            d_2 = _to_d(x_2, sigma_mid, denoised_2)
            x = x + d_2 * dt_2
    return x


def sample_k_dpmpp_2s_ancestral_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    eta: float = 1.0,
    s_noise: float = 1.0,
    sde_noise_backend: str = "gaussian",
    seed: int | None = None,
    callback=None,
    **extra_args,
) -> mx.array:
    """DPM-Solver++(2S) ancestral sampler for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)
    eta = float(eta)
    s_noise = float(s_noise)

    if sde_noise_backend not in {"gaussian", "brownian_torch"}:
        raise ValueError(f"Unknown sde_noise_backend '{sde_noise_backend}'")
    brownian_noises = None
    if eta != 0.0 and sde_noise_backend == "brownian_torch":
        brownian_noises = _precompute_brownian_noises_torch(
            shape=tuple(noise.shape),
            sigmas=np.asarray(sigmas),
            seed=seed,
            dtype=noise.dtype,
        )

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _v_denoiser_forward(model_fn, x, sigma_i * s_in, **extra_args)
        sigma_down, sigma_up = _get_ancestral_step(
            float(sigma_i.item()),
            float(sigma_next.item()),
            eta=eta,
        )

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})

        if sigma_down == 0.0:
            d = _to_d(x, sigma_i, denoised)
            x = x + d * (sigma_down - sigma_i)
        else:
            t = _t_fn(sigma_i)
            t_next = _t_fn(mx.array(sigma_down, dtype=x.dtype))
            r = 0.5
            h = t_next - t
            s = t + r * h
            sigma_s = _sigma_fn(s)
            x_2 = (sigma_s / sigma_i) * x - mx.expm1(-h * r) * denoised
            denoised_2 = _v_denoiser_forward(model_fn, x_2, sigma_s * s_in, **extra_args)
            x = (mx.array(sigma_down, dtype=x.dtype) / sigma_i) * x - mx.expm1(-h) * denoised_2

        if float(sigma_next.item()) > 0.0 and sigma_up > 0.0:
            if brownian_noises is not None and brownian_noises[i] is not None:
                noise_step = brownian_noises[i]
            else:
                noise_step = mx.random.normal(x.shape, dtype=x.dtype)
            x = x + noise_step * (s_noise * float(sigma_up))

    return x


def sample_k_dpmpp_2m_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    callback=None,
    **extra_args,
) -> mx.array:
    """DPM-Solver++(2M) sampler for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)
    old_denoised = None

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _v_denoiser_forward(model_fn, x, sigma_i * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})

        if float(sigma_next.item()) == 0.0:
            x = denoised
        elif old_denoised is None:
            t = _t_fn(sigma_i)
            t_next = _t_fn(sigma_next)
            h = t_next - t
            x = (sigma_next / sigma_i) * x - mx.expm1(-h) * denoised
        else:
            t = _t_fn(sigma_i)
            t_next = _t_fn(sigma_next)
            h = t_next - t
            h_last = t - _t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
            x = (sigma_next / sigma_i) * x - mx.expm1(-h) * denoised_d
        old_denoised = denoised
    return x


def sample_k_dpmpp_2m_sde_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    eta: float = 1.0,
    s_noise: float = 1.0,
    sde_noise_backend: str = "gaussian",
    seed: int | None = None,
    solver_type: str = "midpoint",
    callback=None,
    **extra_args,
) -> mx.array:
    """DPM-Solver++(2M) SDE sampler for v-objective models."""
    if solver_type not in {"heun", "midpoint"}:
        raise ValueError("solver_type must be one of {'heun', 'midpoint'}")
    if sde_noise_backend not in {"gaussian", "brownian_torch"}:
        raise ValueError(f"Unknown sde_noise_backend '{sde_noise_backend}'")

    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)
    eta = float(eta)
    s_noise = float(s_noise)

    brownian_noises = None
    if eta != 0.0 and sde_noise_backend == "brownian_torch":
        brownian_noises = _precompute_brownian_noises_torch(
            shape=tuple(noise.shape),
            sigmas=np.asarray(sigmas),
            seed=seed,
            dtype=noise.dtype,
        )

    old_denoised = None
    h_last = None

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = _v_denoiser_forward(model_fn, x, sigma_i * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma_i, "sigma_hat": sigma_i, "denoised": denoised})

        if float(sigma_next.item()) == 0.0:
            x = denoised
            old_denoised = denoised
            continue

        t = -mx.log(sigma_i)
        s = -mx.log(sigma_next)
        h = s - t
        eta_h = eta * h

        x = (sigma_next / sigma_i) * mx.exp(-eta_h) * x + (-mx.expm1(-h - eta_h)) * denoised

        if old_denoised is not None and h_last is not None:
            r = h_last / h
            if solver_type == "heun":
                numer = -mx.expm1(-h - eta_h)
                denom = mx.where(mx.abs(-h - eta_h) > 1e-12, -h - eta_h, mx.ones_like(h))
                x = x + ((numer / denom) + 1.0) * (1.0 / r) * (denoised - old_denoised)
            else:
                x = x + 0.5 * (-mx.expm1(-h - eta_h)) * (1.0 / r) * (denoised - old_denoised)

        if eta != 0.0:
            if brownian_noises is not None and brownian_noises[i] is not None:
                noise_step = brownian_noises[i]
            else:
                noise_step = mx.random.normal(x.shape, dtype=x.dtype)
            noise_scale = sigma_next * mx.sqrt(mx.maximum(-mx.expm1(-2.0 * eta_h), 0.0)) * s_noise
            x = x + noise_step * noise_scale

        old_denoised = denoised
        h_last = h

    return x


def sample_k_dpmpp_3m_sde_mlx(
    model_fn,
    noise: mx.array,
    *,
    steps: int,
    sigma_min: float = 0.01,
    sigma_max: float = 100.0,
    rho: float = 1.0,
    eta: float = 1.0,
    s_noise: float = 1.0,
    sde_noise_backend: str = "gaussian",
    seed: int | None = None,
    callback=None,
    **extra_args,
) -> mx.array:
    """MLX approximation of k-diffusion `sample_dpmpp_3m_sde` for v-objective models."""
    sigmas = make_k_sigmas_polyexponential(
        steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        dtype=noise.dtype,
    )

    # SAT sample_k behavior: scale initial noise by the first sigma.
    x = noise * sigmas[0]
    s_in = mx.ones((x.shape[0],), dtype=x.dtype)

    denoised_1 = None
    denoised_2 = None
    h_1 = None
    h_2 = None
    eta = float(eta)
    s_noise = float(s_noise)

    if sde_noise_backend not in {"gaussian", "brownian_torch"}:
        raise ValueError(f"Unknown sde_noise_backend '{sde_noise_backend}'")

    brownian_noises = None
    if eta != 0.0 and sde_noise_backend == "brownian_torch":
        brownian_noises = _precompute_brownian_noises_torch(
            shape=tuple(noise.shape),
            sigmas=np.asarray(sigmas),
            seed=seed,
            dtype=noise.dtype,
        )

    for i in range(int(sigmas.shape[0]) - 1):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_vec = sigma_i * s_in
        h = h_1 if h_1 is not None else mx.array(0.0, dtype=x.dtype)

        denoised = _v_denoiser_forward(model_fn, x, sigma_vec, **extra_args)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma_i,
                    "sigma_hat": sigma_i,
                    "denoised": denoised,
                }
            )

        sigma_next_val = float(sigma_next.item())
        if sigma_next_val == 0.0:
            x = denoised
        else:
            t = -mx.log(sigma_i)
            s = -mx.log(sigma_next)
            h = s - t
            h_eta = h * (eta + 1.0)

            x = mx.exp(-h_eta) * x + (-mx.expm1(-h_eta)) * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = mx.expm1(-h_eta) / h_eta + 1.0
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = mx.expm1(-h_eta) / h_eta + 1.0
                x = x + phi_2 * d

            if eta != 0.0:
                if brownian_noises is not None and brownian_noises[i] is not None:
                    noise_step = brownian_noises[i]
                else:
                    noise_step = mx.random.normal(x.shape, dtype=x.dtype)
                noise_scale = sigma_next * mx.sqrt(-mx.expm1(-2.0 * h * eta)) * s_noise
                x = x + noise_step * noise_scale

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1

    return x


def _precompute_brownian_noises_torch(
    *,
    shape: tuple[int, ...],
    sigmas: np.ndarray,
    seed: int | None,
    dtype,
) -> list[mx.array | None]:
    """
    Precompute Brownian-tree normalized noise increments using torch/k-diffusion.

    Returns a list aligned to sampler steps; entries for terminal sigma=0 are None.
    """
    import torch
    import k_diffusion as K

    sigmas = np.asarray(sigmas, dtype=np.float64)
    positive = sigmas[sigmas > 0]
    if positive.size == 0:
        return [None] * (len(sigmas) - 1)

    sigma_min = float(np.min(positive))
    sigma_max = float(np.max(positive))
    sigma_min_tree = float(np.nextafter(sigma_min, -np.inf))
    sigma_max_tree = float(np.nextafter(sigma_max, np.inf))
    x_ref = torch.zeros(shape, device="cpu", dtype=torch.float32)
    sampler = K.sampling.BrownianTreeNoiseSampler(
        x_ref,
        sigma_min=sigma_min_tree,
        sigma_max=sigma_max_tree,
        seed=seed,
    )

    out: list[mx.array | None] = []
    for i in range(len(sigmas) - 1):
        sigma_i = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        if sigma_next == 0.0:
            out.append(None)
            continue
        # Clamp to tree bounds to avoid tiny float edge violations in torchsde.
        sigma_i = min(max(sigma_i, sigma_min_tree), sigma_max_tree)
        sigma_next = min(max(sigma_next, sigma_min_tree), sigma_max_tree)
        step_noise = sampler(sigma_i, sigma_next).detach().cpu().numpy().astype(np.float32, copy=False)
        out.append(mx.array(step_noise).astype(dtype))
    return out


def sample_v_ddim_mlx(
    model_fn,
    x: mx.array,
    *,
    steps: int,
    eta: float = 0.0,
    sigma_max: float = 1.0,
    callback=None,
    cfg_pp: bool = False,
    **extra_args,
) -> mx.array:
    """MLX v-diffusion DDIM sampler (SAT `sample(..., eta=0)` analog)."""

    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")

    sigma_max = min(float(sigma_max), 1.0)
    ts = mx.ones((x.shape[0],), dtype=x.dtype)

    t = mx.linspace(sigma_max, 0.0, steps + 1, dtype=x.dtype)[:-1]
    alphas, sigmas = _get_alphas_sigmas(t)

    h = x
    pred = None
    eta = float(eta)

    for i in range(steps):
        ti = t[i]
        if cfg_pp:
            v, info = model_fn(h, ts * ti, return_info=True, **extra_args)
            v_eps = info.get("uncond_output", v)
        else:
            v = model_fn(h, ts * ti, **extra_args)
            v_eps = v

        pred = h * alphas[i] - v * sigmas[i]
        eps = h * sigmas[i] + v_eps * alphas[i]

        if i < steps - 1:
            # This follows SAT's `sample()` implementation.
            ddim_sigma = (
                eta
                * mx.sqrt((sigmas[i + 1] ** 2) / (sigmas[i] ** 2))
                * mx.sqrt(1.0 - (alphas[i] ** 2) / (alphas[i + 1] ** 2))
            )
            adjusted_sigma = mx.sqrt(mx.maximum(sigmas[i + 1] ** 2 - ddim_sigma**2, 0.0))

            h = pred * alphas[i + 1] + eps * adjusted_sigma
            if eta > 0.0:
                h = h + mx.random.normal(h.shape, dtype=h.dtype) * ddim_sigma

        if callback is not None:
            callback(
                {
                    "x": h if i < steps - 1 else pred,
                    "t": ti,
                    "sigma": sigmas[i],
                    "i": i,
                    "denoised": pred,
                }
            )

    if pred is None:
        raise RuntimeError("DDIM sampling did not execute any steps.")
    return pred
