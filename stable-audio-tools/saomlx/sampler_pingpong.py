"""Ping-pong sampler scaffold for rf_denoiser models."""

from __future__ import annotations

import mlx.core as mx

from .sampler_base import make_rf_schedule


class PingPongSampler:
    def __init__(self, steps: int, *, sigma_max: float = 1.0):
        self.steps = int(steps)
        self.sigma_max = float(sigma_max)

    def sample(self, model_fn, latents: mx.array, *, callback=None, **kwargs) -> mx.array:
        t = make_rf_schedule(self.steps, sigma_max=self.sigma_max, dtype=latents.dtype)
        ts = mx.ones((latents.shape[0],), dtype=latents.dtype)
        x = latents

        for i in range(int(t.shape[0]) - 1):
            t_i = t[i]
            t_next = t[i + 1]
            denoised = x - t_i * model_fn(x, t_i * ts, **kwargs)

            if callback is not None:
                callback(
                    {
                        "x": x,
                        "i": i,
                        "t": t_i,
                        "sigma": t_i,
                        "sigma_hat": t_i,
                        "denoised": denoised,
                    }
                )

            x = (1.0 - t_next) * denoised + t_next * mx.random.normal(x.shape, dtype=x.dtype)

        return x
