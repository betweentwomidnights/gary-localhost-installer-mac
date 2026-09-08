"""Feature-gated runtime quantization for native MLX DiT experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


_QUANTIZATION_ENV = "ACESTEP_MLX_DIT_QUANTIZATION"


@dataclass(frozen=True)
class DiTQuantization:
    """Describe an affine MLX weight-only quantization configuration."""

    bits: int
    group_size: int = 64


def requested_dit_quantization() -> DiTQuantization | None:
    """Parse the experimental DiT quantization setting or return ``None``."""
    raw = os.getenv(_QUANTIZATION_ENV, "").strip().lower()
    if raw in {"", "0", "off", "false", "none"}:
        return None
    aliases = {"4": 4, "4bit": 4, "int4": 4, "8": 8, "8bit": 8, "int8": 8}
    if raw not in aliases:
        raise ValueError(
            f"{_QUANTIZATION_ENV} must be one of off, 4bit, or 8bit"
        )
    return DiTQuantization(bits=aliases[raw])


def quantize_dit(decoder: Any, setting: DiTQuantization) -> None:
    """Quantize decoder linear weights in place and materialize packed arrays."""
    import mlx.core as mx
    import mlx.nn as nn

    nn.quantize(
        decoder,
        group_size=setting.group_size,
        bits=setting.bits,
        mode="affine",
        class_predicate=lambda _path, module: isinstance(module, nn.Linear),
    )
    mx.eval(decoder.parameters())
    decoder.mlx_weight_quantization = setting


def dequantize_dit_linears(decoder: Any) -> DiTQuantization | None:
    """Replace packed linear modules with FP16 linears and return their setting."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    setting = getattr(decoder, "mlx_weight_quantization", None)
    if setting is None:
        return None
    replacements = []
    for path, module in decoder.named_modules():
        if not isinstance(module, nn.QuantizedLinear):
            continue
        linear = nn.Linear(
            module.weight.shape[1] * 32 // module.bits,
            module.scales.shape[0],
            bias="bias" in module,
        )
        linear.weight = mx.dequantize(
            module.weight,
            module.scales,
            module.get("biases"),
            module.group_size,
            module.bits,
            mode=module.mode,
            dtype=mx.float16,
        )
        if "bias" in module:
            linear.bias = module.bias
        replacements.append((path, linear))
    decoder.update_modules(tree_unflatten(replacements))
    mx.eval(decoder.parameters())
    decoder.mlx_weight_quantization = None
    return setting
