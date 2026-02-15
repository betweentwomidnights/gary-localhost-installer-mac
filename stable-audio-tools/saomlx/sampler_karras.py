"""Karras-style sampler scaffold for v-diffusion models."""

from __future__ import annotations


class KarrasSampler:
    def __init__(self, steps: int):
        self.steps = steps

    def sample(self, model_fn, latents, **kwargs):
        raise NotImplementedError
