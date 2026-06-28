"""Trainable LoRA/DoRA helpers for the ACE-Step MLX DiT decoder."""

from __future__ import annotations

import json
import math
import typing as tp
from dataclasses import dataclass
from pathlib import Path

try:  # Keep profile helpers importable on systems without MLX installed.
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
except ModuleNotFoundError:  # pragma: no cover - exercised by non-MLX dev shells.
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    tree_unflatten = None  # type: ignore[assignment]


ATTENTION_PROFILE = "attention"
BALANCED_PROFILE = "balanced"
MODULE_PROFILE_CHOICES = (ATTENTION_PROFILE, BALANCED_PROFILE)

DEFAULT_ADAPTER_NAME = "default"
PEFT_ADAPTER_CONFIG = "adapter_config.json"
PEFT_ADAPTER_WEIGHTS = "adapter_model.safetensors"

_ATTENTION_TARGETS = (
    "cross_attn.k_proj",
    "cross_attn.o_proj",
    "cross_attn.q_proj",
    "cross_attn.v_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)

# Relative to the user-selected reference rank. At rank 64 these become:
# self Q/K/V/O = 16/24/80/56, cross Q/K/V/O = 64/40/32/48,
# MLP gate/up/down = 40/48/48.
_BALANCED_RANK_MULTIPLIERS = {
    "cross_attn.k_proj": 5 / 8,
    "cross_attn.o_proj": 3 / 4,
    "cross_attn.q_proj": 1.0,
    "cross_attn.v_proj": 1 / 2,
    "mlp.down_proj": 3 / 4,
    "mlp.gate_proj": 5 / 8,
    "mlp.up_proj": 3 / 4,
    "self_attn.k_proj": 3 / 8,
    "self_attn.o_proj": 7 / 8,
    "self_attn.q_proj": 1 / 4,
    "self_attn.v_proj": 5 / 4,
}


@dataclass(frozen=True)
class ACELoRALayerSpec:
    """Resolved per-projection adapter budget for an ACE MLX Linear layer."""

    layer_name: str
    family: str
    rank: int
    alpha: int


@dataclass(frozen=True)
class ACELoRAInjectionReport:
    """Summary returned after injecting trainable ACE MLX adapters."""

    layer_names: tuple[str, ...]
    trainable_parameters: int
    adapter_type: str
    module_profile: str
    rank_pattern: dict[str, int]
    alpha_pattern: dict[str, int]

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)


class ACEMLXLoRALinear(nn.Module if nn is not None else object):  # type: ignore[misc]
    """MLX Linear wrapper implementing ACE LoRA and row-wise DoRA."""

    def __init__(
        self,
        base: tp.Any,
        *,
        rank: int,
        alpha: int,
        source_name: str,
        adapter_type: str = "lora",
    ):
        _require_mlx()
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scaling = float(self.alpha) / float(self.rank)
        self.source_name = str(source_name)
        self.adapter_type = _canonical_adapter_type(adapter_type)

        fan_out, fan_in = (int(value) for value in base.weight.shape)
        _validate_rank(self.rank, fan_out=fan_out, fan_in=fan_in, source_name=source_name)
        init_scale = 1.0 / math.sqrt(fan_in)
        self.lora_A = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(self.rank, fan_in),
            dtype=mx.float32,
        )
        self.lora_B = mx.zeros((fan_out, self.rank), dtype=mx.float32)
        if self.adapter_type == "dora":
            self.magnitude = _row_norms(base.weight)

    def __call__(self, x):
        if self.adapter_type == "dora":
            adapted_weight = _dora_weight_2d(
                self.base.weight.astype(mx.float32),
                lora_A=self.lora_A,
                lora_B=self.lora_B,
                magnitude=self.magnitude,
                scaling=self.scaling,
            )
            out = x.astype(mx.float32) @ adapted_weight.T
            bias = getattr(self.base, "bias", None)
            if bias is not None:
                out = out + bias.astype(mx.float32)
            return out.astype(x.dtype)

        base_out = self.base(x)
        adapter_in = x.astype(mx.float32)
        adapter_out = (adapter_in @ self.lora_A.T) @ self.lora_B.T
        return base_out + (adapter_out * self.scaling).astype(base_out.dtype)


def build_balanced_projection_profile(
    base_rank: int,
    base_alpha: int,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Build target, rank, and alpha patterns for the balanced ACE profile."""
    if base_rank <= 0 or base_alpha <= 0:
        raise ValueError("base rank and alpha must be greater than zero")

    rank_pattern = {
        name: max(1, round(base_rank * multiplier))
        for name, multiplier in _BALANCED_RANK_MULTIPLIERS.items()
    }
    alpha_pattern = {
        name: max(1, round(base_alpha * rank / base_rank))
        for name, rank in rank_pattern.items()
    }
    return list(_BALANCED_RANK_MULTIPLIERS), rank_pattern, alpha_pattern


def build_ace_projection_profile(
    *,
    rank: int,
    alpha: int | None = None,
    module_profile: str = BALANCED_PROFILE,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Resolve the ACE adapter target families and per-family budgets."""
    rank = int(rank)
    alpha = int(rank * 2 if alpha is None else alpha)
    if rank <= 0 or alpha <= 0:
        raise ValueError("rank and alpha must be greater than zero")

    module_profile = _canonical_module_profile(module_profile)
    if module_profile == BALANCED_PROFILE:
        return build_balanced_projection_profile(rank, alpha)

    rank_pattern = {name: rank for name in _ATTENTION_TARGETS}
    alpha_pattern = {name: alpha for name in _ATTENTION_TARGETS}
    return list(_ATTENTION_TARGETS), rank_pattern, alpha_pattern


def resolve_ace_lora_layer_specs(
    model: tp.Any,
    *,
    rank: int,
    alpha: int | None = None,
    module_profile: str = BALANCED_PROFILE,
) -> tuple[ACELoRALayerSpec, ...]:
    """Find ACE MLX Linear projections selected by an adapter profile."""
    _require_mlx()
    targets, rank_pattern, alpha_pattern = build_ace_projection_profile(
        rank=rank,
        alpha=alpha,
        module_profile=module_profile,
    )
    specs: list[ACELoRALayerSpec] = []
    for name, layer in model.named_modules():
        if not name or not isinstance(layer, nn.Linear):
            continue
        family = _matched_projection_family(name, targets)
        if family is None:
            continue
        specs.append(
            ACELoRALayerSpec(
                layer_name=name,
                family=family,
                rank=rank_pattern[family],
                alpha=alpha_pattern[family],
            )
        )

    if not specs:
        raise ValueError(
            f"No ACE MLX Linear layers matched module_profile={module_profile!r}."
        )
    return tuple(specs)


def inject_trainable_lora(
    model: tp.Any,
    *,
    rank: int = 64,
    alpha: int | None = None,
    module_profile: str = BALANCED_PROFILE,
    adapter_type: str = "dora",
) -> ACELoRAInjectionReport:
    """Freeze ``model`` and replace selected ACE projections with LoRA wrappers."""
    _require_mlx()
    adapter_type = _canonical_adapter_type(adapter_type)
    module_profile = _canonical_module_profile(module_profile)
    targets, rank_pattern, alpha_pattern = build_ace_projection_profile(
        rank=rank,
        alpha=alpha,
        module_profile=module_profile,
    )

    model.freeze()
    replacements: list[tuple[str, ACEMLXLoRALinear]] = []
    for name, layer in model.named_modules():
        if not name or not isinstance(layer, nn.Linear):
            continue
        family = _matched_projection_family(name, targets)
        if family is None:
            continue
        replacements.append(
            (
                name,
                ACEMLXLoRALinear(
                    layer,
                    rank=rank_pattern[family],
                    alpha=alpha_pattern[family],
                    source_name=name,
                    adapter_type=adapter_type,
                ),
            )
        )

    if not replacements:
        raise ValueError(
            f"No ACE MLX Linear layers matched module_profile={module_profile!r}."
        )

    model.update_modules(tree_unflatten(replacements))
    trainable_count = sum(
        int(value.size)
        for _, value in tree_flatten(model.trainable_parameters())
    )
    return ACELoRAInjectionReport(
        layer_names=tuple(name for name, _ in replacements),
        trainable_parameters=trainable_count,
        adapter_type=adapter_type,
        module_profile=module_profile,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )


def iter_trainable_lora_layers(model: tp.Any) -> tp.Iterator[ACEMLXLoRALinear]:
    """Yield trainable ACE LoRA wrappers from an MLX model."""
    _require_mlx()
    for _, layer in model.named_modules():
        if isinstance(layer, ACEMLXLoRALinear):
            yield layer


def save_trainable_lora_adapter(
    model: tp.Any,
    output_dir: str | Path,
    *,
    rank: int,
    alpha: int | None = None,
    module_profile: str = BALANCED_PROFILE,
    adapter_type: str | None = None,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
    base_model_name_or_path: str | None = None,
    extra_config: dict[str, tp.Any] | None = None,
) -> Path:
    """Save injected ACE MLX adapters as a PEFT-style LoRA directory."""
    _require_mlx()
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    adapter_layers = list(iter_trainable_lora_layers(model))
    if not adapter_layers:
        raise ValueError("The model has no trainable ACE MLX LoRA layers to save.")

    resolved_adapter_type = _canonical_adapter_type(
        adapter_type or _adapter_type_from_layers(adapter_layers)
    )
    targets, rank_pattern, alpha_pattern = build_ace_projection_profile(
        rank=rank,
        alpha=alpha,
        module_profile=module_profile,
    )

    state_dict: dict[str, tp.Any] = {}
    adapter_name = (adapter_name or DEFAULT_ADAPTER_NAME).strip() or DEFAULT_ADAPTER_NAME
    for layer in adapter_layers:
        if layer.adapter_type != resolved_adapter_type:
            raise ValueError(
                f"Cannot save layer {layer.source_name!r} as {resolved_adapter_type}; "
                f"layer adapter type is {layer.adapter_type!r}."
            )
        prefix = f"base_model.model.{layer.source_name}"
        state_dict[f"{prefix}.lora_A.{adapter_name}.weight"] = layer.lora_A.astype(mx.float16)
        state_dict[f"{prefix}.lora_B.{adapter_name}.weight"] = layer.lora_B.astype(mx.float16)
        if resolved_adapter_type == "dora":
            state_dict[
                f"{prefix}.lora_magnitude_vector.{adapter_name}.weight"
            ] = layer.magnitude.astype(mx.float16)

    mx.save_safetensors(
        str(output_path / PEFT_ADAPTER_WEIGHTS),
        state_dict,
        metadata={
            "gary_adapter_config": json.dumps(
                {
                    "format": "peft-lora",
                    "adapter_type": resolved_adapter_type,
                    "module_profile": _canonical_module_profile(module_profile),
                    "rank_pattern": rank_pattern,
                    "alpha_pattern": alpha_pattern,
                }
            )
        },
    )

    config = _build_peft_adapter_config(
        rank=int(rank),
        alpha=int(rank * 2 if alpha is None else alpha),
        target_modules=targets,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        use_dora=resolved_adapter_type == "dora",
        base_model_name_or_path=base_model_name_or_path,
        extra_config=extra_config,
    )
    with (output_path / PEFT_ADAPTER_CONFIG).open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def _build_peft_adapter_config(
    *,
    rank: int,
    alpha: int,
    target_modules: tp.Sequence[str],
    rank_pattern: dict[str, int],
    alpha_pattern: dict[str, int],
    use_dora: bool,
    base_model_name_or_path: str | None,
    extra_config: dict[str, tp.Any] | None,
) -> dict[str, tp.Any]:
    config: dict[str, tp.Any] = {
        "alpha_pattern": alpha_pattern,
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path or "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": int(alpha),
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": int(rank),
        "rank_pattern": rank_pattern,
        "revision": None,
        "target_modules": list(target_modules),
        "task_type": None,
        "use_dora": bool(use_dora),
        "use_rslora": False,
    }
    if extra_config:
        config.update(extra_config)
    return config


def _require_mlx() -> None:
    if mx is None or nn is None or tree_flatten is None or tree_unflatten is None:
        raise RuntimeError("MLX is required for ACE MLX LoRA training helpers.")


def _canonical_module_profile(module_profile: str) -> str:
    value = str(module_profile or BALANCED_PROFILE).strip().lower()
    if value in MODULE_PROFILE_CHOICES:
        return value
    raise ValueError(
        f"Unsupported ACE module profile {module_profile!r}; "
        f"expected one of {MODULE_PROFILE_CHOICES}."
    )


def _canonical_adapter_type(adapter_type: str) -> str:
    value = str(adapter_type or "lora").strip().lower()
    if value == "dora-rows":
        return "dora"
    if value in {"lora", "dora"}:
        return value
    raise ValueError(
        "ACE MLX training currently supports adapter_type='lora' and "
        f"adapter_type='dora'; got {adapter_type!r}."
    )


def _matched_projection_family(name: str, targets: tp.Sequence[str]) -> str | None:
    for target in targets:
        if name == target or name.endswith(f".{target}"):
            return target
    return None


def _adapter_type_from_layers(layers: tp.Sequence[ACEMLXLoRALinear]) -> str:
    adapter_types = {layer.adapter_type for layer in layers}
    if not adapter_types:
        return "lora"
    if len(adapter_types) != 1:
        raise ValueError(
            "Cannot save a mixed-adapter ACE MLX checkpoint from adapter types "
            f"{sorted(adapter_types)}."
        )
    return next(iter(adapter_types))


def _validate_rank(rank: int, *, fan_out: int, fan_in: int, source_name: str) -> None:
    max_rank = min(fan_out, fan_in)
    if rank > max_rank:
        raise ValueError(
            f"Adapter rank {rank} exceeds maximum rank {max_rank} for "
            f"{source_name!r} with shape ({fan_out}, {fan_in})."
        )


def _row_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=1)).astype(mx.float32)


def _dora_weight_2d(weight_2d, *, lora_A, lora_B, magnitude, scaling: float):
    v = weight_2d.astype(mx.float32) + (lora_B @ lora_A) * float(scaling)
    norms = mx.sqrt(mx.sum(v**2, axis=1, keepdims=True))
    v_hat = v / mx.maximum(norms, 1e-12)
    return v_hat * magnitude.astype(mx.float32)[:, None]
