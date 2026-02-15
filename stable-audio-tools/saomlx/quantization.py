"""Quantization helpers for SAO MLX models."""

from __future__ import annotations

import typing as tp
from dataclasses import asdict, dataclass

import mlx.nn as nn


@dataclass(frozen=True)
class QuantizationReport:
    enabled: bool
    scope: str
    bits: int
    group_size: int
    eligible_linear_modules: int
    skipped_linear_modules: int
    quantized_linear_modules: int
    skipped_examples: list[dict[str, tp.Any]]

    def to_dict(self) -> dict[str, tp.Any]:
        return asdict(self)


def _linear_quant_candidate(
    module: nn.Module,
    *,
    group_size: int,
) -> tuple[bool, str]:
    if not isinstance(module, nn.Linear):
        return False, "not_linear"
    if not hasattr(module, "to_quantized"):
        return False, "missing_to_quantized"
    weight = getattr(module, "weight", None)
    if weight is None:
        return False, "missing_weight"
    if len(weight.shape) != 2:
        return False, f"weight_rank_{len(weight.shape)}"
    input_dim = int(weight.shape[1])
    if input_dim % int(group_size) != 0:
        return False, f"input_dim_{input_dim}_not_divisible_by_{group_size}"
    return True, "ok"


def quantize_dit(
    dit_model: nn.Module,
    *,
    bits: int = 8,
    group_size: int = 64,
    scope: str = "transformer",
) -> QuantizationReport:
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}")
    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if scope not in {"transformer", "all"}:
        raise ValueError(f"Unknown quantization scope '{scope}'. Expected one of ['transformer', 'all'].")

    target = dit_model.transformer if scope == "transformer" else dit_model

    skipped_examples: list[dict[str, tp.Any]] = []
    eligible_count = 0
    skipped_count = 0

    def predicate(name: str, module: nn.Module) -> bool:
        nonlocal eligible_count, skipped_count
        ok, reason = _linear_quant_candidate(module, group_size=group_size)
        if ok:
            eligible_count += 1
            return True
        if isinstance(module, nn.Linear):
            skipped_count += 1
            if len(skipped_examples) < 12:
                shape = getattr(getattr(module, "weight", None), "shape", None)
                skipped_examples.append(
                    {
                        "name": name,
                        "reason": reason,
                        "weight_shape": list(shape) if shape is not None else None,
                    }
                )
        return False

    nn.quantize(target, group_size=group_size, bits=bits, class_predicate=predicate)

    quantized_linear_cls = getattr(nn, "QuantizedLinear", None)
    quantized_count = 0
    if quantized_linear_cls is not None:
        for _, module in target.named_modules():
            if isinstance(module, quantized_linear_cls):
                quantized_count += 1

    return QuantizationReport(
        enabled=True,
        scope=scope,
        bits=int(bits),
        group_size=int(group_size),
        eligible_linear_modules=int(eligible_count),
        skipped_linear_modules=int(skipped_count),
        quantized_linear_modules=int(quantized_count),
        skipped_examples=skipped_examples,
    )
