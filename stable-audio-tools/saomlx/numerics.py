"""Numerical policy helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DTypePolicy:
    model_dtype: str = "float32"
    sampler_dtype: str = "float32"
    decode_dtype: str = "float32"


FP32_POLICY = DTypePolicy()
BF16_POLICY = DTypePolicy(model_dtype="bfloat16", sampler_dtype="float32", decode_dtype="float32")
