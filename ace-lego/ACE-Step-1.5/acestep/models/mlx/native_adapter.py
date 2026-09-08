"""Apply PEFT LoRA and DoRA adapters directly to an MLX decoder."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

_KEY = re.compile(
    r"^(?:base_model\.model\.)?(?P<layer>.+)\."
    r"(?:"
    r"(?P<ab_kind>lora_A|lora_B)(?:\.(?P<ab_adapter>[^.]+))?\.weight"
    r"|"
    r"(?P<magnitude_kind>lora_magnitude_vector)"
    r"(?:\.(?P<magnitude_adapter>[^.]+))?(?:\.weight)?"
    r")$"
)


@dataclass(frozen=True)
class _Target:
    """Hold one resolved adapter target and its low-rank arrays."""

    name: str
    module: Any
    lora_a: Any
    lora_b: Any
    magnitude: Any | None
    scaling: float


class NativeMLXAdapter:
    """Merge one PEFT-compatible adapter into MLX weights without PyTorch."""

    def __init__(
        self,
        decoder: Any,
        adapter_dir: str | Path,
        restore_base: Callable[[], None],
        expected_variant: str | None = None,
    ) -> None:
        """Resolve adapter targets against an unquantized native MLX decoder."""
        self.decoder = decoder
        if getattr(decoder, "mlx_weight_quantization", None) is not None:
            raise ValueError(
                "Native MLX LoRA/DoRA requires the FP16 decoder; disable "
                "ACESTEP_MLX_DIT_QUANTIZATION before loading an adapter"
            )
        self.restore_base = restore_base
        self.path = Path(adapter_dir).expanduser().resolve()
        self.config = self._read_config()
        self._validate_base_variant(expected_variant)
        self.use_dora = bool(self.config.get("use_dora", False))
        self.targets = self._resolve_targets()
        self.eval_batch_size = self._read_eval_batch_size()
        self.scale = 0.0

    @property
    def target_count(self) -> int:
        """Return the number of decoder projections controlled by the adapter."""
        return len(self.targets)

    def apply(self, strength: float) -> None:
        """Apply ``strength`` in [0, 1], restoring exact base arrays at zero."""
        strength = float(strength)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("Native MLX adapter strength must be finite and in [0, 1]")
        if self.scale != 0.0:
            self.restore_base()
        if strength == 0.0:
            self.scale = 0.0
            return
        pending = []
        for target in self.targets:
            base = target.module.weight
            delta = (target.lora_b @ target.lora_a) * target.scaling
            if target.magnitude is None:
                value = base.astype(mx.float32) + delta * strength
            else:
                value = self._dora_weight(base, target, delta, strength)
            target.module.weight = value.astype(base.dtype)
            pending.append(target.module.weight)
            if len(pending) >= self.eval_batch_size:
                mx.eval(pending)
                pending.clear()
        if pending:
            mx.eval(pending)
        self.scale = strength

    def unload(self) -> None:
        """Restore every original decoder array and release adapter state."""
        if self.scale != 0.0:
            self.restore_base()
        self.scale = 0.0

    def _dora_weight(self, base: Any, target: _Target, delta: Any, strength: float) -> Any:
        """Return the PEFT-compatible row-wise DoRA weight at ``strength``."""
        value = base.astype(mx.float32) + delta * strength
        norm = mx.sqrt(mx.sum(value**2, axis=1, keepdims=True))
        return value / mx.maximum(norm, 1e-12) * target.magnitude[:, None]

    def _read_config(self) -> dict[str, Any]:
        """Read and validate the adapter's PEFT configuration."""
        path = self.path / "adapter_config.json"
        weights = self.path / "adapter_model.safetensors"
        if not path.is_file() or not weights.is_file():
            raise FileNotFoundError("Native MLX loading requires adapter_config.json and adapter_model.safetensors")
        config = json.loads(path.read_text(encoding="utf-8"))
        if str(config.get("peft_type", "LORA")).upper() != "LORA":
            raise ValueError("Native MLX loading currently supports PEFT LoRA/DoRA only")
        if config.get("fan_in_fan_out", False):
            raise ValueError("fan_in_fan_out LoRA adapters are not supported by the MLX decoder")
        return config

    @staticmethod
    def _read_eval_batch_size() -> int:
        """Return a bounded synchronization batch for lazy MLX adapter merges."""
        raw = os.getenv("ACESTEP_MLX_ADAPTER_EVAL_BATCH_SIZE", "").strip() or "32"
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "ACESTEP_MLX_ADAPTER_EVAL_BATCH_SIZE must be an integer"
            ) from exc
        if value < 1 or value > 1024:
            raise ValueError(
                "ACESTEP_MLX_ADAPTER_EVAL_BATCH_SIZE must be between 1 and 1024"
            )
        return value

    def _validate_base_variant(self, expected_variant: str | None) -> None:
        """Reject an adapter that declares a different ACE source family."""
        declared = str(self.config.get("base_model_name_or_path") or "").strip()
        if not expected_variant or not declared:
            return
        declared_variant = declared.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if declared_variant != expected_variant:
            raise ValueError(
                f"Adapter base variant {declared_variant!r} does not match {expected_variant!r}"
            )

    def _resolve_targets(self) -> tuple[_Target, ...]:
        """Match adapter tensors strictly to MLX decoder projection weights."""
        arrays = dict(mx.load(str(self.path / "adapter_model.safetensors")))
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for key, value in arrays.items():
            match = _KEY.match(key)
            if match:
                layer = match.group("layer").removeprefix("decoder.")
                kind = match.group("ab_kind") or match.group("magnitude_kind")
                adapter_name = (
                    match.group("ab_adapter")
                    or match.group("magnitude_adapter")
                    or "default"
                )
                grouped.setdefault((layer, adapter_name), {})[
                    kind
                ] = value.astype(mx.float32)
        parameters = dict(tree_flatten(self.decoder.parameters()))
        modules = dict(self.decoder.named_modules())
        adapter_names = {adapter_name for _, adapter_name in grouped}
        if len(adapter_names) > 1:
            raise ValueError("Native MLX adapter files must contain exactly one adapter name")
        targets = []
        for (layer, _adapter_name), values in grouped.items():
            name = f"{layer}.weight"
            if name not in parameters:
                raise ValueError(f"Adapter target is absent from MLX decoder: {name}")
            if layer not in modules or not hasattr(modules[layer], "weight"):
                raise ValueError(f"Adapter target is not an MLX weighted module: {layer}")
            if "lora_A" not in values or "lora_B" not in values:
                raise ValueError(f"Adapter target has incomplete A/B weights: {layer}")
            magnitude = values.get("lora_magnitude_vector")
            if self.use_dora != (magnitude is not None):
                raise ValueError(f"DoRA magnitude/config mismatch for {layer}")
            self._validate_shapes(name, parameters[name], values, magnitude)
            rank = int(values["lora_A"].shape[0])
            expected_rank = int(self._pattern_value("rank_pattern", layer, self.config.get("r", rank)))
            if rank != expected_rank:
                raise ValueError(f"Adapter rank/config mismatch for {layer}")
            alpha = float(self._pattern_value("alpha_pattern", layer, self.config.get("lora_alpha", rank)))
            divisor = math.sqrt(rank) if self.config.get("use_rslora", False) else rank
            targets.append(_Target(name, modules[layer], values["lora_A"], values["lora_B"], magnitude, alpha / divisor))
        if not targets:
            raise ValueError("No PEFT LoRA tensors were found in adapter_model.safetensors")
        return tuple(sorted(targets, key=lambda target: target.name))

    def _pattern_value(self, field: str, layer: str, fallback: Any) -> Any:
        """Resolve PEFT exact-or-suffix rank/alpha pattern semantics."""
        patterns = self.config.get(field) or {}
        matches = [(key, value) for key, value in patterns.items() if layer == key or layer.endswith(f".{key}")]
        return max(matches, key=lambda item: len(item[0]))[1] if matches else fallback

    @staticmethod
    def _validate_shapes(name: str, base: Any, values: dict[str, Any], magnitude: Any | None) -> None:
        """Reject incompatible matrices before mutating the decoder."""
        a, b = values["lora_A"], values["lora_B"]
        if len(base.shape) != 2 or tuple(a.shape)[1] != base.shape[1] or b.shape[0] != base.shape[0] or b.shape[1] != a.shape[0]:
            raise ValueError(f"Adapter shape mismatch for {name}")
        if magnitude is not None and tuple(magnitude.shape) != (base.shape[0],):
            raise ValueError(f"DoRA magnitude shape mismatch for {name}")
