from __future__ import annotations

import re
import typing as tp
from dataclasses import dataclass, field
from itertools import product as _product
from pathlib import Path

import numpy as np

from stable_audio_3.mlx.runtime import (
    MLXRuntimeUnavailableError,
    import_mlx_core,
)

try:
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError as exc:
    raise MLXRuntimeUnavailableError(
        "MLX is not installed in this environment. "
        "Install the Apple Silicon MLX runtime before attempting MLX inference."
    ) from exc

mx = import_mlx_core(required=True)

_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.+)\.parametrizations\.weight\.(?P<index>\d+)\."
    r"(?P<param>lora_A|lora_B|M_xs|magnitude|magnitude_r|magnitude_c|U|V)$"
)
_XS_ADAPTERS = {"lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs"}
_SUPPORTED_ADAPTERS = {
    "lora",
    "dora",
    "dora-rows",
    "dora-cols",
    "bora",
    *_XS_ADAPTERS,
}


@dataclass(frozen=True)
class MLXLoRAApplyReport:
    target_label: str
    paths: tuple[str, ...]
    names: tuple[str, ...]
    strength: float | None
    strengths: tuple[float, ...]
    loaded_layers: int
    applied_layers: int
    loaded_layers_by_lora: tuple[tuple[str, int], ...] = ()
    applied_layers_by_lora: tuple[tuple[str, int], ...] = ()
    active_layers_by_lora: tuple[tuple[str, int], ...] = ()
    intervals: tuple[tuple[float, float], ...] = ()
    layer_filters: tuple[str, ...] = ()
    filtered_layers: tuple[str, ...] = ()
    skipped_layers: tuple[str, ...] = ()
    missing_targets: tuple[str, ...] = ()
    unsupported_adapters: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, tp.Any]:
        return {
            "target_label": self.target_label,
            "paths": list(self.paths),
            "names": list(self.names),
            "strength": None if self.strength is None else float(self.strength),
            "strengths": [float(strength) for strength in self.strengths],
            "loaded_layers": int(self.loaded_layers),
            "applied_layers": int(self.applied_layers),
            "loaded_layers_by_lora": [
                {"name": name, "layers": int(count)}
                for name, count in self.loaded_layers_by_lora
            ],
            "applied_layers_by_lora": [
                {"name": name, "layers": int(count)}
                for name, count in self.applied_layers_by_lora
            ],
            "active_layers_by_lora": [
                {"name": name, "layers": int(count)}
                for name, count in self.active_layers_by_lora
            ],
            "intervals": [
                {"min": float(interval[0]), "max": float(interval[1])}
                for interval in self.intervals
            ],
            "layer_filters": list(self.layer_filters),
            "filtered_layers": list(self.filtered_layers),
            "skipped_layers": list(self.skipped_layers),
            "missing_targets": list(self.missing_targets),
            "unsupported_adapters": list(self.unsupported_adapters),
        }


@dataclass(frozen=True)
class _MLXLoRALayer:
    lora_index: int
    lora_name: str
    source_name: str
    target_key: str
    adapter_type: str
    alpha: float
    rank: int
    params: dict[str, np.ndarray]

    def apply(self, target_weight: np.ndarray, *, strength: float) -> np.ndarray:
        if float(strength) == 0.0:
            return target_weight
        if self.adapter_type in _SUPPORTED_ADAPTERS:
            return self._apply_lora_family(target_weight, strength=float(strength))
        raise ValueError(f"Unsupported MLX LoRA adapter type: {self.adapter_type!r}")

    def _apply_lora_family(self, target_weight: np.ndarray, *, strength: float) -> np.ndarray:
        if self.adapter_type in _XS_ADAPTERS:
            source_shape = _source_shape_for_xs(tuple(target_weight.shape), self.params)
        else:
            delta_2d, _ = _lora_delta_2d(self.params)
            source_shape = _source_shape_for_delta(tuple(target_weight.shape), delta_2d.shape)
        base_source = _target_to_source_weight(target_weight, source_shape)
        base_2d = base_source.reshape(source_shape[0], -1).astype(np.float32, copy=False)
        if self.adapter_type in _XS_ADAPTERS:
            delta_2d, rank = _xs_delta_2d(self.params, base_2d)
        else:
            delta_2d, rank = _lora_delta_2d(self.params)

        scaling = float(self.alpha) / float(rank)
        v = base_2d + (scaling * strength) * delta_2d.astype(np.float32, copy=False)

        if self.adapter_type in {"lora", "lora-xs"}:
            adapted_2d = v
        elif self.adapter_type in {"dora", "dora-rows", "dora-rows-xs"}:
            magnitude = _require_param(self.params, "magnitude", self.source_name).reshape(-1)
            adapted_2d = _dora_scale(v, magnitude, norm_dim=1)
        elif self.adapter_type in {"dora-cols", "dora-cols-xs"}:
            magnitude = _require_param(self.params, "magnitude", self.source_name).reshape(-1)
            adapted_2d = _dora_scale(v, magnitude, norm_dim=0)
        elif self.adapter_type in {"bora", "bora-xs"}:
            magnitude_r = _require_param(self.params, "magnitude_r", self.source_name).reshape(-1)
            magnitude_c = _require_param(self.params, "magnitude_c", self.source_name).reshape(-1)
            adapted_2d = _bora_scale(v, magnitude_r, magnitude_c)
        else:
            raise ValueError(f"Unsupported MLX LoRA adapter type: {self.adapter_type!r}")

        adapted_source = adapted_2d.reshape(source_shape)
        return _source_to_target_weight(adapted_source, tuple(target_weight.shape))


@dataclass
class MLXLoRASet:
    target_label: str
    paths: tuple[str, ...]
    names: tuple[str, ...]
    layers: list[_MLXLoRALayer]
    missing_targets: tuple[str, ...] = ()
    unsupported_adapters: tuple[str, ...] = ()
    skipped_layers: tuple[str, ...] = ()
    _base_params: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_checkpoints(
        cls,
        paths: tp.Sequence[str | Path],
        module,
        *,
        target_label: str,
        names: tp.Sequence[str] | None = None,
    ) -> "MLXLoRASet":
        from stable_audio_3.models.lora import (
            infer_global_rank,
            load_lora_checkpoint,
            prepare_dora_state_dict,
            resolve_adapter_type,
        )

        target_params = dict(tree_flatten(module.parameters()))
        target_keys = tuple(target_params.keys())
        resolved_paths = tuple(str(Path(path).expanduser().resolve()) for path in paths)
        display_names = _normalize_lora_names(names, len(resolved_paths))
        layers: list[_MLXLoRALayer] = []
        missing_targets: list[str] = []
        unsupported_adapters: list[str] = []
        skipped_layers: list[str] = []
        resolved_names: list[str] = []

        for lora_index, path in enumerate(resolved_paths):
            state_dict, config = load_lora_checkpoint(path)
            prepare_dora_state_dict(state_dict)
            lora_name = display_names[lora_index] if display_names else Path(path).stem
            resolved_names.append(lora_name)
            raw_adapter_type = str(config.get("adapter_type", "lora"))
            adapter_type = _adapter_type_from_state(
                resolve_adapter_type(raw_adapter_type, state_dict),
                state_dict,
            )

            if adapter_type not in _SUPPORTED_ADAPTERS:
                unsupported_adapters.append(f"{Path(path).name}: {adapter_type}")
                continue

            if config.get("rank") is not None:
                global_rank = int(config["rank"])
            else:
                try:
                    global_rank = infer_global_rank(state_dict)
                except ValueError:
                    global_rank = 0
            alpha_value = config.get("alpha", config.get("lora_alpha"))
            alpha = float(alpha_value if alpha_value is not None else (global_rank or 1))
            grouped = _group_lora_state_dict(state_dict)
            for source_name, params in grouped.items():
                target_key = _resolve_target_key(f"{source_name}.weight", target_keys)
                if target_key is None:
                    missing_targets.append(source_name)
                    continue
                if adapter_type in _XS_ADAPTERS:
                    if "M_xs" not in params:
                        skipped_layers.append(f"{source_name}: missing M_xs")
                        continue
                elif "lora_A" not in params or "lora_B" not in params:
                    skipped_layers.append(f"{source_name}: missing lora_A/lora_B")
                    continue
                rank = _rank_from_lora_params(params) or global_rank
                if rank <= 0:
                    skipped_layers.append(f"{source_name}: unable to infer rank")
                    continue
                if adapter_type in _XS_ADAPTERS and (
                    "U" not in params or "V" not in params
                ):
                    target_weight = np.asarray(
                        target_params[target_key],
                        dtype=np.float32,
                    )
                    source_shape = _source_shape_for_xs(
                        tuple(target_weight.shape),
                        params,
                    )
                    base_source = _target_to_source_weight(
                        target_weight,
                        source_shape,
                    )
                    base_2d = base_source.reshape(source_shape[0], -1)
                    u, v = _svd_bases(base_2d, rank)
                    params = {**params, "U": u, "V": v}
                layers.append(
                    _MLXLoRALayer(
                        lora_index=lora_index,
                        lora_name=lora_name,
                        source_name=source_name,
                        target_key=target_key,
                        adapter_type=adapter_type,
                        alpha=alpha,
                        rank=rank,
                        params=params,
                    )
                )

        return cls(
            target_label=target_label,
            paths=resolved_paths,
            names=tuple(resolved_names),
            layers=layers,
            missing_targets=tuple(sorted(set(missing_targets))),
            unsupported_adapters=tuple(unsupported_adapters),
            skipped_layers=tuple(skipped_layers),
        )

    def apply_to(
        self,
        module,
        *,
        strength: float | tp.Sequence[float] = 1.0,
        lora_configs: tp.Sequence[dict[str, tp.Any]] | None = None,
        sigma: float | None = None,
    ) -> MLXLoRAApplyReport:
        params = dict(tree_flatten(module.parameters()))
        target_keys = sorted({layer.target_key for layer in self.layers})
        strengths = _normalize_lora_strengths(strength, len(self.paths))
        configs = _normalize_lora_configs(lora_configs, len(self.paths))
        if self._base_params is None:
            self._base_params = {
                key: np.asarray(params[key], dtype=np.float32).copy()
                for key in target_keys
                if key in params
            }

        adapted_by_key = {key: value.copy() for key, value in self._base_params.items()}
        applied = 0
        loaded_by_index = [0 for _ in self.paths]
        applied_by_index = [0 for _ in self.paths]
        active_by_index = [0 for _ in self.paths]
        filtered_layers: list[str] = []
        skipped = list(self.skipped_layers)
        for layer in self.layers:
            loaded_by_index[layer.lora_index] += 1
        for layer in self.layers:
            if layer.target_key not in adapted_by_key:
                skipped.append(f"{layer.source_name}: target missing at apply time")
                continue
            config = configs[layer.lora_index]
            interval = config["interval"]
            if sigma is not None and not (interval[0] <= float(sigma) <= interval[1]):
                continue
            layer_filter = str(config.get("layer_filter", "") or "")
            if _layer_matches_filter(layer, layer_filter):
                filtered_layers.append(layer.source_name)
                continue
            try:
                adapted_by_key[layer.target_key] = layer.apply(
                    adapted_by_key[layer.target_key],
                    strength=float(strengths[layer.lora_index]),
                )
            except ValueError as exc:
                skipped.append(f"{layer.source_name}: {exc}")
                continue
            applied += 1
            applied_by_index[layer.lora_index] += 1
            active_by_index[layer.lora_index] += 1

        updates = []
        for key, adapted in adapted_by_key.items():
            target_dtype = params[key].dtype
            arr = mx.array(adapted.astype(np.float32, copy=False))
            if arr.dtype != target_dtype:
                arr = arr.astype(target_dtype)
            updates.append((key, arr))

        if updates:
            module.update(tree_unflatten(updates))
            mx.eval(module.parameters())

        scalar_strength = strengths[0] if len(set(strengths)) == 1 else None
        loaded_layers_by_lora = tuple(zip(self.names, loaded_by_index, strict=True))
        applied_layers_by_lora = tuple(zip(self.names, applied_by_index, strict=True))
        active_layers_by_lora = tuple(zip(self.names, active_by_index, strict=True))
        return MLXLoRAApplyReport(
            target_label=self.target_label,
            paths=self.paths,
            names=self.names,
            strength=scalar_strength,
            strengths=strengths,
            loaded_layers=len(self.layers),
            applied_layers=applied,
            loaded_layers_by_lora=loaded_layers_by_lora,
            applied_layers_by_lora=applied_layers_by_lora,
            active_layers_by_lora=active_layers_by_lora,
            intervals=tuple(config["interval"] for config in configs),
            layer_filters=tuple(str(config.get("layer_filter", "") or "") for config in configs),
            filtered_layers=tuple(sorted(set(filtered_layers))),
            skipped_layers=tuple(skipped),
            missing_targets=self.missing_targets,
            unsupported_adapters=self.unsupported_adapters,
        )


class MLXLoRAScheduledModule:
    """Apply DiT LoRAs lazily as the sampler moves through sigma intervals."""

    def __init__(
        self,
        module,
        lora_set: MLXLoRASet,
        *,
        strength: float | tp.Sequence[float] = 1.0,
        lora_configs: tp.Sequence[dict[str, tp.Any]] | None = None,
    ) -> None:
        self.module = module
        self.lora_set = lora_set
        self.strengths = _normalize_lora_strengths(strength, len(lora_set.paths))
        self.lora_configs = _normalize_lora_configs(lora_configs, len(lora_set.paths))
        self._last_signature: tuple[tp.Any, ...] | None = None
        self.reports: list[MLXLoRAApplyReport] = []

    def __call__(self, x, t, **kwargs):
        sigma = _first_scalar(t)
        signature = _lora_schedule_signature(
            self.strengths,
            self.lora_configs,
            sigma=sigma,
        )
        if signature != self._last_signature:
            report = self.lora_set.apply_to(
                self.module,
                strength=self.strengths,
                lora_configs=self.lora_configs,
                sigma=sigma,
            )
            self.reports.append(report)
            self._last_signature = signature
        return self.module(x, t, **kwargs)


def apply_mlx_loras(
    module,
    paths: tp.Sequence[str | Path],
    *,
    target_label: str,
    strength: float | tp.Sequence[float] = 1.0,
    lora_configs: tp.Sequence[dict[str, tp.Any]] | None = None,
    sigma: float | None = None,
    names: tp.Sequence[str] | None = None,
) -> MLXLoRAApplyReport:
    loras = MLXLoRASet.from_checkpoints(
        paths,
        module,
        target_label=target_label,
        names=names,
    )
    return loras.apply_to(
        module,
        strength=strength,
        lora_configs=lora_configs,
        sigma=sigma,
    )


def _normalize_lora_strengths(
    strength: float | tp.Sequence[float],
    count: int,
) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if isinstance(strength, (int, float)):
        return tuple(float(strength) for _ in range(count))

    values = tuple(float(value) for value in strength)
    if len(values) == 1:
        return tuple(values[0] for _ in range(count))
    if len(values) != count:
        raise ValueError(f"Expected 1 or {count} LoRA strengths, got {len(values)}")
    return values


def _normalize_lora_names(names: tp.Sequence[str] | None, count: int) -> tuple[str, ...] | None:
    if names is None:
        return None
    if len(names) != count:
        raise ValueError(f"Expected {count} LoRA names, got {len(names)}")
    return tuple(str(name) for name in names)


def _normalize_lora_configs(
    lora_configs: tp.Sequence[dict[str, tp.Any]] | None,
    count: int,
) -> tuple[dict[str, tp.Any], ...]:
    if count <= 0:
        return ()
    if lora_configs is None:
        return tuple({"interval": (0.0, 1.0), "layer_filter": ""} for _ in range(count))
    if len(lora_configs) != count:
        raise ValueError(f"Expected {count} LoRA configs, got {len(lora_configs)}")

    normalized = []
    for index, config in enumerate(lora_configs):
        interval = config.get("interval", (0.0, 1.0))
        if len(interval) != 2:
            raise ValueError(f"LoRA config {index} interval must have two values.")
        interval_min = float(interval[0])
        interval_max = float(interval[1])
        if interval_min > interval_max:
            raise ValueError(f"LoRA config {index} interval min must be <= max.")
        normalized.append(
            {
                "lora_index": int(config.get("lora_index", index)),
                "interval": (interval_min, interval_max),
                "layer_filter": str(config.get("layer_filter", "") or ""),
            }
        )
    return tuple(normalized)


def _first_scalar(value: tp.Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    arr = np.asarray(value, dtype=np.float32)
    return float(arr.reshape(-1)[0])


def _lora_schedule_signature(
    strengths: tuple[float, ...],
    lora_configs: tuple[dict[str, tp.Any], ...],
    *,
    sigma: float,
) -> tuple[tp.Any, ...]:
    signature = []
    for strength, config in zip(strengths, lora_configs, strict=True):
        interval = config["interval"]
        active = interval[0] <= float(sigma) <= interval[1]
        signature.append(
            (
                float(strength) if active else 0.0,
                str(config.get("layer_filter", "") or ""),
            )
        )
    return tuple(signature)


def _layer_matches_filter(layer: _MLXLoRALayer, layer_filter: str) -> bool:
    layer_filter = (layer_filter or "").strip().lower()
    if not layer_filter:
        return False
    layer_names = (
        layer.source_name.lower(),
        layer.target_key.lower(),
        layer.target_key.lower().removesuffix(".weight"),
    )
    filters = [item.strip() for item in layer_filter.split(",") if item.strip()]
    for item in filters:
        for expanded in _expand_filter(item):
            if any(expanded in layer_name for layer_name in layer_names):
                return True
    return False


def _expand_filter(value: str) -> tuple[str, ...]:
    match_parts = re.split(r"\[(\d+)-(\d+)\]", value)
    if len(match_parts) == 1:
        return (value,)

    literals = match_parts[0::3]
    starts = match_parts[1::3]
    ends = match_parts[2::3]
    pools = []
    for start, end in zip(starts, ends, strict=True):
        start_i = int(start)
        end_i = int(end)
        step = 1 if end_i >= start_i else -1
        pools.append([str(value) for value in range(start_i, end_i + step, step)])

    out = []
    for combo in _product(*pools):
        pieces = []
        for index, literal in enumerate(literals):
            pieces.append(literal)
            if index < len(combo):
                pieces.append(combo[index])
        out.append("".join(pieces))
    return tuple(out)


def _tensor_to_numpy(value: tp.Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _group_lora_state_dict(state_dict: dict[str, tp.Any]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for key, value in state_dict.items():
        match = _LORA_KEY_RE.match(key)
        if match is None:
            continue
        prefix = match.group("prefix")
        param = match.group("param")
        grouped.setdefault(prefix, {})[param] = _tensor_to_numpy(value)
    return grouped


def _adapter_type_from_state(adapter_type: str, state_dict: dict[str, tp.Any]) -> str:
    keys = tuple(state_dict.keys())
    has_xs = any(key.endswith(".M_xs") for key in keys)
    if has_xs:
        if adapter_type in {"bora", "bora-xs"} or any(
            key.endswith(".magnitude_r") or key.endswith(".magnitude_c")
            for key in keys
        ):
            return "bora-xs"
        if adapter_type in {"dora-cols", "dora-cols-xs"}:
            return "dora-cols-xs"
        if adapter_type in {"dora", "dora-rows", "dora-rows-xs"} or any(
            key.endswith(".magnitude") for key in keys
        ):
            return "dora-rows-xs"
        return "lora-xs"
    if adapter_type == "lora":
        if any(key.endswith(".magnitude_r") or key.endswith(".magnitude_c") for key in keys):
            return "bora"
        if any(key.endswith(".magnitude") for key in keys):
            return "dora-rows"
    return adapter_type


def _resolve_target_key(source_weight_key: str, target_keys: tuple[str, ...]) -> str | None:
    target_key_set = set(target_keys)
    candidates = _target_key_candidates(source_weight_key)
    for candidate in candidates:
        if candidate in target_key_set:
            return candidate

    suffix_matches = []
    for candidate in candidates:
        suffix_matches.extend(key for key in target_keys if key.endswith(candidate))
    suffix_matches = sorted(set(suffix_matches))
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def _target_key_candidates(source_weight_key: str) -> tuple[str, ...]:
    prefixes = (
        "model.model.",
        "model.",
        "conditioner.",
        "encoder.",
        "conditioners.prompt.model.encoder.",
        "conditioners.prompt.model.",
        "conditioners.prompt.",
        "conditioners.seconds_total.",
    )
    candidates = [source_weight_key]
    for prefix in prefixes:
        if source_weight_key.startswith(prefix):
            candidates.append(source_weight_key[len(prefix) :])

    if source_weight_key.endswith("embedder.embedding.1.weight"):
        candidates.append("proj.weight")
    return tuple(dict.fromkeys(candidates))


def _rank_from_lora_params(params: dict[str, np.ndarray]) -> int:
    core = params.get("M_xs")
    if core is not None and core.ndim == 2 and core.shape[0] == core.shape[1]:
        return int(core.shape[0])
    a = params.get("lora_A")
    b = params.get("lora_B")
    if a is None or b is None:
        return 0
    if b.shape[-1] == a.shape[0]:
        return int(a.shape[0])
    if a.shape[-1] == b.shape[0]:
        return int(a.shape[-1])
    return 0


def _lora_delta_2d(params: dict[str, np.ndarray]) -> tuple[np.ndarray, int]:
    a = _require_param(params, "lora_A", "LoRA layer")
    b = _require_param(params, "lora_B", "LoRA layer")
    a_merge = a.astype(np.float64, copy=False)
    b_merge = b.astype(np.float64, copy=False)
    if b.shape[-1] == a.shape[0]:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            delta = b_merge @ a_merge
        if not np.isfinite(delta).all():
            raise ValueError(f"LoRA delta contains non-finite values for shapes A={a.shape}, B={b.shape}")
        return delta.astype(np.float32, copy=False), int(a.shape[0])
    if a.shape[-1] == b.shape[0]:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            delta = a_merge @ b_merge
        if not np.isfinite(delta).all():
            raise ValueError(f"LoRA delta contains non-finite values for shapes A={a.shape}, B={b.shape}")
        return delta.astype(np.float32, copy=False), int(a.shape[-1])
    raise ValueError(f"Unable to multiply LoRA matrices with shapes A={a.shape}, B={b.shape}")


def _xs_delta_2d(
    params: dict[str, np.ndarray],
    base_2d: np.ndarray,
) -> tuple[np.ndarray, int]:
    core = _require_param(params, "M_xs", "LoRA-XS layer")
    if core.ndim != 2 or core.shape[0] != core.shape[1]:
        raise ValueError(f"LoRA-XS core must be square, got shape {core.shape}")
    rank = int(core.shape[0])
    if rank > min(base_2d.shape):
        raise ValueError(
            f"LoRA-XS rank {rank} exceeds base weight shape {base_2d.shape}"
        )

    u = params.get("U")
    v = params.get("V")
    if u is None or v is None:
        u, v = _svd_bases(base_2d, rank)
    else:
        u = u.astype(np.float32, copy=False)
        v = v.astype(np.float32, copy=False)
        if u.shape != (base_2d.shape[0], rank):
            raise ValueError(
                f"LoRA-XS U shape {u.shape} does not match "
                f"{(base_2d.shape[0], rank)}"
            )
        if v.shape != (base_2d.shape[1], rank):
            raise ValueError(
                f"LoRA-XS V shape {v.shape} does not match "
                f"{(base_2d.shape[1], rank)}"
            )
    delta = u.astype(np.float64) @ core.astype(np.float64) @ v.astype(np.float64).T
    if not np.isfinite(delta).all():
        raise ValueError("LoRA-XS delta contains non-finite values")
    return delta.astype(np.float32, copy=False), rank


def _require_param(params: dict[str, np.ndarray], key: str, layer_name: str) -> np.ndarray:
    value = params.get(key)
    if value is None:
        raise ValueError(f"{layer_name} is missing {key}")
    return value.astype(np.float32, copy=False)


def _source_shape_for_delta(
    target_shape: tuple[int, ...],
    delta_shape: tuple[int, int],
) -> tuple[int, ...]:
    if len(target_shape) == 2:
        candidates = (target_shape, (target_shape[1], target_shape[0]))
    elif len(target_shape) == 3:
        candidates = (target_shape, (target_shape[0], target_shape[2], target_shape[1]))
    else:
        candidates = (target_shape,)

    for candidate in candidates:
        if candidate[0] == delta_shape[0] and int(np.prod(candidate[1:])) == delta_shape[1]:
            return candidate
    raise ValueError(f"Unable to map LoRA delta {delta_shape} to target shape {target_shape}")


def _source_shape_for_xs(
    target_shape: tuple[int, ...],
    params: dict[str, np.ndarray],
) -> tuple[int, ...]:
    u = params.get("U")
    v = params.get("V")
    if u is not None and v is not None:
        return _source_shape_for_delta(
            target_shape,
            (int(u.shape[0]), int(v.shape[0])),
        )
    if len(target_shape) == 3:
        return (target_shape[0], target_shape[2], target_shape[1])
    return target_shape


def _target_to_source_weight(target: np.ndarray, source_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(target.shape) == source_shape:
        return target
    if target.ndim == 2 and tuple(target.T.shape) == source_shape:
        return target.T
    if target.ndim == 3:
        candidate = np.transpose(target, (0, 2, 1))
        if tuple(candidate.shape) == source_shape:
            return candidate
    raise ValueError(f"Unable to map target shape {target.shape} to source shape {source_shape}")


def _source_to_target_weight(source: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    if tuple(source.shape) == target_shape:
        return source
    if source.ndim == 2:
        candidate = source.T
        if tuple(candidate.shape) == target_shape:
            return candidate
    if source.ndim == 3:
        candidate = np.transpose(source, (0, 2, 1))
        if tuple(candidate.shape) == target_shape:
            return candidate
    raise ValueError(f"Unable to map source shape {source.shape} to target shape {target_shape}")


def _dora_scale(v: np.ndarray, magnitude: np.ndarray, *, norm_dim: int) -> np.ndarray:
    norms = np.linalg.norm(v, axis=norm_dim, keepdims=True)
    v_hat = v / np.maximum(norms, 1e-12)
    if norm_dim == 1:
        if magnitude.shape[0] != v.shape[0]:
            raise ValueError(f"DoRA row magnitude {magnitude.shape} does not match {v.shape}")
        return v_hat * magnitude[:, None]
    if magnitude.shape[0] != v.shape[1]:
        raise ValueError(f"DoRA column magnitude {magnitude.shape} does not match {v.shape}")
    return v_hat * magnitude[None, :]


def _bora_scale(v: np.ndarray, magnitude_r: np.ndarray, magnitude_c: np.ndarray) -> np.ndarray:
    if magnitude_r.shape[0] != v.shape[0]:
        raise ValueError(f"BoRA row magnitude {magnitude_r.shape} does not match {v.shape}")
    if magnitude_c.shape[0] != v.shape[1]:
        raise ValueError(f"BoRA column magnitude {magnitude_c.shape} does not match {v.shape}")
    row_normed = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    intermediate = magnitude_r[:, None] * row_normed
    col_normed = intermediate / np.maximum(np.linalg.norm(intermediate, axis=0, keepdims=True), 1e-12)
    return col_normed * magnitude_c[None, :]


def _svd_bases(weight_2d: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    u, _, vh = np.linalg.svd(
        weight_2d.astype(np.float32, copy=False),
        full_matrices=False,
    )
    u, vh = _canonicalize_svd_signs(u, vh)
    return (
        u[:, :rank].astype(np.float32, copy=False),
        vh[:rank, :].T.astype(np.float32, copy=False),
    )


def _canonicalize_svd_signs(
    u: np.ndarray,
    vh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    max_abs_indices = np.abs(u).argmax(axis=0)
    signs = np.sign(u[max_abs_indices, np.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs[None, :], vh * signs[:, None]
