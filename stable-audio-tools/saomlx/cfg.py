"""Classifier-free guidance helpers."""

from __future__ import annotations


def apply_cfg(cond, uncond, scale: float):
    return uncond + (cond - uncond) * scale
