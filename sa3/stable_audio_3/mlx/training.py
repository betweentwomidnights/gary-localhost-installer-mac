from __future__ import annotations

import json
import math
import re
import typing as tp
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import NormalDist

import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from stable_audio_3.mlx.runtime import import_mlx_core, import_mlx_nn

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)

_XS_ADAPTER_TYPES = {"lora-xs", "dora-rows-xs", "dora-cols-xs", "bora-xs"}
_FULL_WEIGHT_ADAPTER_TYPES = {
    "dora-rows",
    "bora",
    *_XS_ADAPTER_TYPES,
}
_DORA_ROW_ADAPTER_TYPES = {"dora-rows", "dora-rows-xs"}
_DORA_COL_ADAPTER_TYPES = {"dora-cols-xs"}
_TIMESTEP_SAMPLERS = {
    "uniform",
    "logit_normal",
    "trunc_logit_normal",
    "log_snr",
    "log_snr_uniform",
}
_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class MLXLoRAInjectionReport:
    layer_names: tuple[str, ...]
    trainable_parameters: int
    adapter_type: str

    @property
    def layer_count(self) -> int:
        return len(self.layer_names)


def sample_training_timesteps(
    sampler: str,
    batch_size: int,
    *,
    rng: np.random.Generator,
    options: dict[str, float] | None = None,
) -> np.ndarray:
    sampler = str(sampler).strip().lower()
    if sampler not in _TIMESTEP_SAMPLERS:
        raise ValueError(f"Unsupported timestep sampler: {sampler!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    options = options or {}
    if sampler == "uniform":
        values = rng.random(batch_size)
    elif sampler == "logit_normal":
        values = _sigmoid(rng.standard_normal(batch_size))
    elif sampler == "trunc_logit_normal":
        values = 1.0 - _truncated_logistic_normal_rescaled(
            batch_size,
            rng=rng,
        )
    elif sampler == "log_snr":
        mean = float(options.get("mean_logsnr", -1.2))
        std = float(options.get("std_logsnr", 2.0))
        logsnr = rng.standard_normal(batch_size) * std + mean
        values = np.clip(_sigmoid(-logsnr), 1e-4, 1.0 - 1e-4)
    else:
        minimum = float(options.get("min_logsnr", -6.0))
        maximum = float(options.get("max_logsnr", 5.0))
        if maximum <= minimum:
            raise ValueError("max_logsnr must be greater than min_logsnr.")
        logsnr = rng.uniform(minimum, maximum, batch_size)
        values = np.clip(_sigmoid(-logsnr), 1e-4, 1.0 - 1e-4)
    return np.asarray(values, dtype=np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=np.float64)))


def _truncated_logistic_normal_rescaled(
    size: int,
    *,
    rng: np.random.Generator,
    left_trunc: float = 0.075,
    right_trunc: float = 1.0,
) -> np.ndarray:
    if not 0.0 < left_trunc < right_trunc <= 1.0:
        raise ValueError("Expected 0 < left_trunc < right_trunc <= 1.")

    lower_logit = math.log(left_trunc / (1.0 - left_trunc))
    lower_cdf = _STANDARD_NORMAL.cdf(lower_logit)
    upper_cdf = (
        1.0
        if right_trunc == 1.0
        else _STANDARD_NORMAL.cdf(math.log(right_trunc / (1.0 - right_trunc)))
    )
    uniforms = lower_cdf + (upper_cdf - lower_cdf) * rng.random(size)
    epsilon = np.finfo(np.float64).eps
    logits = np.asarray(
        [
            _STANDARD_NORMAL.inv_cdf(float(np.clip(value, epsilon, 1.0 - epsilon)))
            for value in uniforms
        ],
        dtype=np.float64,
    )
    samples = _sigmoid(logits)
    return (samples - left_trunc) / (right_trunc - left_trunc)


class MLXLoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.adapter_type = _canonical_adapter_type(adapter_type)

        fan_out, fan_in = (int(value) for value in base.weight.shape)
        source_weight = _linear_source_weight_2d(base.weight)
        _validate_rank(self.rank, fan_out=fan_out, fan_in=fan_in, source_name=source_name)
        if self.adapter_type in _XS_ADAPTER_TYPES:
            self.U, self.V = _svd_bases(source_weight, self.rank)
            self.M_xs = mx.zeros((self.rank, self.rank), dtype=mx.float32)
            self.freeze(keys=["U", "V"], recurse=False)
        else:
            init_scale = 1.0 / math.sqrt(fan_in)
            self.lora_A = mx.random.uniform(
                low=-init_scale,
                high=init_scale,
                shape=(self.rank, fan_in),
                dtype=mx.float32,
            )
            self.lora_B = mx.zeros((fan_out, self.rank), dtype=mx.float32)
        if self.adapter_type == "dora-rows":
            self.magnitude = _row_norms(source_weight)
        elif self.adapter_type == "dora-rows-xs":
            self.magnitude = _row_norms(source_weight)
        elif self.adapter_type == "dora-cols-xs":
            self.magnitude = _column_norms(source_weight)
        elif self.adapter_type == "bora":
            self.magnitude_r = _row_norms(source_weight)
            self.magnitude_c = _column_norms(source_weight)
        elif self.adapter_type == "bora-xs":
            self.magnitude_r = _row_norms(source_weight)
            self.magnitude_c = _column_norms(source_weight)
        # Cache the frozen base energy used by the algebraically reformulated
        # DoRA norm. Leading-underscore attributes are not MLX parameters and
        # therefore do not alter checkpoint contents.
        if self.adapter_type in _DORA_ROW_ADAPTER_TYPES:
            self._w0_sq = mx.sum(source_weight * source_weight, axis=1)
        elif self.adapter_type in _DORA_COL_ADAPTER_TYPES:
            self._w0_sq = mx.sum(source_weight * source_weight, axis=0)

    def __call__(self, x):
        if self.adapter_type in (
            _DORA_ROW_ADAPTER_TYPES | _DORA_COL_ADAPTER_TYPES
        ):
            return _reformulated_dora_linear_forward(self, x)
        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            adapted_weight = _adapted_weight_2d(
                _linear_source_weight_2d(self.base.weight),
                adapter_type=self.adapter_type,
                layer=self,
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


class MLXLoRAConv1d(nn.Module):
    def __init__(
        self,
        base: nn.Conv1d,
        *,
        rank: int,
        alpha: float,
        source_name: str,
        adapter_type: str = "lora",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")

        self.base = base
        self.base.freeze()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.source_name = str(source_name)
        self.adapter_type = _canonical_adapter_type(adapter_type)

        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in base.weight.shape
        )
        fan_in = fan_in_per_group * kernel_size
        source_weight = _conv1d_source_weight_2d(base.weight)
        _validate_rank(self.rank, fan_out=fan_out, fan_in=fan_in, source_name=source_name)
        if self.adapter_type in _XS_ADAPTER_TYPES:
            self.U, self.V = _svd_bases(source_weight, self.rank)
            self.M_xs = mx.zeros((self.rank, self.rank), dtype=mx.float32)
            self.freeze(keys=["U", "V"], recurse=False)
        else:
            init_scale = 1.0 / math.sqrt(fan_in)
            self.lora_A = mx.random.uniform(
                low=-init_scale,
                high=init_scale,
                shape=(self.rank, fan_in),
                dtype=mx.float32,
            )
            self.lora_B = mx.zeros((fan_out, self.rank), dtype=mx.float32)
        if self.adapter_type == "dora-rows":
            self.magnitude = _row_norms(source_weight)
        elif self.adapter_type == "dora-rows-xs":
            self.magnitude = _row_norms(source_weight)
        elif self.adapter_type == "dora-cols-xs":
            self.magnitude = _column_norms(source_weight)
        elif self.adapter_type == "bora":
            self.magnitude_r = _row_norms(source_weight)
            self.magnitude_c = _column_norms(source_weight)
        elif self.adapter_type == "bora-xs":
            self.magnitude_r = _row_norms(source_weight)
            self.magnitude_c = _column_norms(source_weight)

    def __call__(self, x):
        fan_out, kernel_size, fan_in_per_group = (
            int(value) for value in self.base.weight.shape
        )

        if self.adapter_type in _FULL_WEIGHT_ADAPTER_TYPES:
            adapted_source = _adapted_weight_2d(
                _conv1d_source_weight_2d(self.base.weight),
                adapter_type=self.adapter_type,
                layer=self,
            )
            adapted_weight = _conv1d_weight_from_source_2d(
                adapted_source,
                fan_out=fan_out,
                fan_in_per_group=fan_in_per_group,
                kernel_size=kernel_size,
            )
            out = mx.conv1d(
                x.astype(mx.float32),
                adapted_weight,
                self.base.stride,
                self.base.padding,
                self.base.dilation,
                self.base.groups,
            )
            bias = getattr(self.base, "bias", None)
            if bias is not None:
                out = out + bias.astype(mx.float32)
            return out.astype(x.dtype)

        base_out = self.base(x)
        delta_weight = _conv1d_weight_from_source_2d(
            self.lora_B @ self.lora_A,
            fan_out=fan_out,
            fan_in_per_group=fan_in_per_group,
            kernel_size=kernel_size,
        )
        adapter_out = mx.conv1d(
            x.astype(mx.float32),
            delta_weight,
            self.base.stride,
            self.base.padding,
            self.base.dilation,
            self.base.groups,
        )
        return base_out + (adapter_out * self.scaling).astype(base_out.dtype)


MLXTrainableLoRALayer = MLXLoRALinear | MLXLoRAConv1d


DEFAULT_SA3_TRAINING_LORA_EXCLUDE = (
    "to_timestep_embed",
    "to_cond_embed",
    "to_global_embed",
    "to_local_embed",
    "global_cond_embedder",
    "project_in",
    "project_out",
    "preprocess_conv",
    "postprocess_conv",
)


LORA_LAYER_SCOPE_ALL = "all-projections"
LORA_LAYER_SCOPE_ATTENTION_FF = "attention-feedforward"
LORA_LAYER_SCOPE_CHOICES = (
    LORA_LAYER_SCOPE_ALL,
    LORA_LAYER_SCOPE_ATTENTION_FF,
)
# Defaults to the reduced scope: a paired 2,000-step A/B against the full 228
# DiT layers showed no measurable quality difference (mean loss delta +0.0005)
# while running ~5% faster and producing a smaller adapter.
LORA_LAYER_SCOPE_DEFAULT = LORA_LAYER_SCOPE_ATTENTION_FF


def layer_scope_exclusions(scope: str) -> tuple[str, ...]:
    """Return the DiT layer-name exclusions implied by a training layer scope.

    ``all-projections`` is Gary's historical behaviour and adapts every eligible
    DiT Linear/Conv1d layer (228 on medium-base). ``attention-feedforward``
    applies the official standalone trainer's product-default exclusions, which
    drop the embedding/projection layers and leave 168. Both scopes leave the
    seconds_total conditioner adapter untouched, so the two differ only in DiT
    layer count.
    """

    if scope == LORA_LAYER_SCOPE_ALL:
        return ()
    if scope == LORA_LAYER_SCOPE_ATTENTION_FF:
        return DEFAULT_SA3_TRAINING_LORA_EXCLUDE
    raise ValueError(
        f"Unknown LoRA layer scope {scope!r}; "
        f"expected one of {', '.join(LORA_LAYER_SCOPE_CHOICES)}."
    )


def inject_trainable_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    adapter_type: str = "lora",
) -> MLXLoRAInjectionReport:
    alpha = float(rank if alpha is None else alpha)
    adapter_type = _canonical_adapter_type(adapter_type)
    model.freeze()

    replacements: list[tuple[str, MLXTrainableLoRALayer]] = []
    for name, layer in model.named_modules():
        if not name or not _name_is_selected(name, include=include, exclude=exclude):
            continue
        if isinstance(layer, nn.Linear):
            replacement = MLXLoRALinear(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
            )
        elif isinstance(layer, nn.Conv1d):
            replacement = MLXLoRAConv1d(
                layer,
                rank=rank,
                alpha=alpha,
                source_name=name,
                adapter_type=adapter_type,
            )
        else:
            continue
        replacements.append((name, replacement))

    if not replacements:
        raise ValueError("No MLX Linear or Conv1d layers matched the LoRA filters.")

    model.update_modules(tree_unflatten(replacements))
    trainable_count = sum(
        int(value.size)
        for _, value in tree_flatten(model.trainable_parameters())
    )
    return MLXLoRAInjectionReport(
        layer_names=tuple(name for name, _ in replacements),
        trainable_parameters=trainable_count,
        adapter_type=adapter_type,
    )


def save_trainable_lora(
    model: nn.Module,
    path: str | Path,
    *,
    rank: int,
    alpha: float | None = None,
    include: tp.Sequence[str] | None = None,
    exclude: tp.Sequence[str] | None = None,
    adapter_type: str | None = None,
    extra_metadata: dict[str, tp.Any] | None = None,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict: dict[str, tp.Any] = {}
    adapter_layers = list(iter_trainable_lora_layers(model))
    adapter_type = _canonical_adapter_type(
        adapter_type or _adapter_type_from_layers(adapter_layers)
    )
    for layer in adapter_layers:
        prefix = f"{layer.source_name}.parametrizations.weight.0"
        if adapter_type in _XS_ADAPTER_TYPES:
            if layer.adapter_type != adapter_type:
                raise ValueError(
                    f"Cannot save layer {layer.source_name!r} as {adapter_type}; "
                    f"layer adapter type is {layer.adapter_type!r}."
                )
            state_dict[f"{prefix}.M_xs"] = layer.M_xs.astype(mx.float16)
        else:
            state_dict[f"{prefix}.lora_A"] = layer.lora_A.astype(mx.float16)
            state_dict[f"{prefix}.lora_B"] = layer.lora_B.astype(mx.float16)
        if adapter_type == "dora-rows":
            if layer.adapter_type != "dora-rows":
                raise ValueError(
                    f"Cannot save layer {layer.source_name!r} as DoRA; "
                    f"layer adapter type is {layer.adapter_type!r}."
                )
            state_dict[f"{prefix}.magnitude"] = layer.magnitude.astype(mx.float16)
        elif adapter_type == "bora":
            if layer.adapter_type != "bora":
                raise ValueError(
                    f"Cannot save layer {layer.source_name!r} as BoRA; "
                    f"layer adapter type is {layer.adapter_type!r}."
            )
            state_dict[f"{prefix}.magnitude_r"] = layer.magnitude_r.astype(mx.float16)
            state_dict[f"{prefix}.magnitude_c"] = layer.magnitude_c.astype(mx.float16)
        elif adapter_type in {"dora-rows-xs", "dora-cols-xs"}:
            state_dict[f"{prefix}.magnitude"] = layer.magnitude.astype(mx.float16)
        elif adapter_type == "bora-xs":
            state_dict[f"{prefix}.magnitude_r"] = layer.magnitude_r.astype(mx.float16)
            state_dict[f"{prefix}.magnitude_c"] = layer.magnitude_c.astype(mx.float16)

    if not state_dict:
        raise ValueError("The model has no trainable MLX LoRA layers to save.")

    config: dict[str, tp.Any] = {
        "rank": int(rank),
        "alpha": float(rank if alpha is None else alpha),
        "adapter_type": adapter_type,
        "include": list(include) if include else None,
        "exclude": list(exclude) if exclude else None,
    }
    if extra_metadata:
        config.update(extra_metadata)
    mx.save_safetensors(
        str(output_path),
        state_dict,
        metadata={"lora_config": json.dumps(config)},
    )
    return output_path


def iter_trainable_lora_layers(model: nn.Module) -> tp.Iterator[MLXTrainableLoRALayer]:
    for _, layer in model.named_modules():
        if isinstance(layer, (MLXLoRALinear, MLXLoRAConv1d)):
            yield layer


def rectified_flow_loss(
    model: nn.Module,
    clean,
    t,
    *,
    noise=None,
    loss_mask=None,
    context_loss_mask=None,
    context_loss_weight: float = 1.0,
    model_kwargs: dict[str, tp.Any] | None = None,
):
    if noise is None:
        noise = mx.random.normal(clean.shape, dtype=clean.dtype)
    t = t.astype(mx.float32)
    alpha = (1.0 - t)[:, None, None].astype(clean.dtype)
    sigma = t[:, None, None].astype(clean.dtype)
    noised = clean * alpha + noise * sigma
    target = noise - clean
    prediction = model(noised, t, **(model_kwargs or {}))
    mse = (prediction.astype(mx.float32) - target.astype(mx.float32)) ** 2

    if loss_mask is None:
        loss = mx.mean(mse)
    else:
        mask = loss_mask[:, None, :].astype(mx.float32)
        loss = mx.sum(mse * mask) / mx.maximum(
            mx.sum(mask) * mse.shape[1],
            1.0,
        )

    if context_loss_mask is not None and context_loss_weight > 0:
        context_mask = context_loss_mask[:, None, :].astype(mx.float32)
        context_loss = mx.sum(mse * context_mask) / mx.maximum(
            mx.sum(context_mask) * mse.shape[1],
            1.0,
        )
        loss = loss + context_loss * float(context_loss_weight)
    return loss


def _name_is_selected(
    name: str,
    *,
    include: tp.Sequence[str] | None,
    exclude: tp.Sequence[str] | None,
) -> bool:
    if include and not _matches_any(name, include):
        return False
    return not (exclude and _matches_any(name, exclude))


def _canonical_adapter_type(adapter_type: str) -> str:
    adapter_type = str(adapter_type or "lora").strip().lower()
    if adapter_type == "dora":
        return "dora-rows"
    if adapter_type == "dora-xs":
        return "dora-rows-xs"
    if adapter_type == "xs":
        return "lora-xs"
    if adapter_type in {
        "lora",
        "dora-rows",
        "bora",
        "lora-xs",
        "dora-rows-xs",
        "dora-cols-xs",
        "bora-xs",
    }:
        return adapter_type
    raise ValueError(
        "MLX training currently supports adapter_type='lora', "
        "adapter_type='dora'/'dora-rows', adapter_type='bora', "
        "and XS variants 'lora-xs', 'dora-rows-xs', 'dora-cols-xs', "
        "and 'bora-xs'; "
        f"got {adapter_type!r}."
    )


def _adapter_type_from_layers(layers: tp.Sequence[MLXTrainableLoRALayer]) -> str:
    adapter_types = {layer.adapter_type for layer in layers}
    if not adapter_types:
        return "lora"
    if len(adapter_types) != 1:
        raise ValueError(
            "Cannot save a mixed-adapter MLX LoRA checkpoint from adapter types "
            f"{sorted(adapter_types)}."
        )
    return next(iter(adapter_types))


def _linear_source_weight_2d(weight):
    return weight.astype(mx.float32)


def _conv1d_source_weight_2d(weight):
    fan_out, kernel_size, fan_in_per_group = (int(value) for value in weight.shape)
    return weight.astype(mx.float32).transpose(0, 2, 1).reshape(
        fan_out,
        fan_in_per_group * kernel_size,
    )


def _conv1d_weight_from_source_2d(
    source,
    *,
    fan_out: int,
    fan_in_per_group: int,
    kernel_size: int,
):
    return source.reshape(
        fan_out,
        fan_in_per_group,
        kernel_size,
    ).transpose(0, 2, 1)


def _row_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=1)).astype(mx.float32)


def _column_norms(weight_2d):
    return mx.sqrt(mx.sum(weight_2d.astype(mx.float32) ** 2, axis=0)).astype(mx.float32)


def _adapted_weight_2d(weight_2d, *, adapter_type: str, layer):
    delta = _adapter_delta_2d(layer)
    v = weight_2d.astype(mx.float32) + delta * float(layer.scaling)
    if adapter_type == "lora-xs":
        return v
    if adapter_type in {"dora-rows", "dora-rows-xs"}:
        return _dora_weight_2d(v, magnitude=layer.magnitude, norm_dim=1)
    if adapter_type == "dora-cols-xs":
        return _dora_weight_2d(v, magnitude=layer.magnitude, norm_dim=0)
    if adapter_type in {"bora", "bora-xs"}:
        return _bora_weight_2d(
            v,
            magnitude_r=layer.magnitude_r,
            magnitude_c=layer.magnitude_c,
        )
    raise ValueError(f"Unsupported normalized MLX adapter type: {adapter_type!r}.")


def _adapter_delta_2d(layer):
    if layer.adapter_type in _XS_ADAPTER_TYPES:
        return layer.U @ layer.M_xs.astype(mx.float32) @ layer.V.T
    return layer.lora_B @ layer.lora_A


def _effective_low_rank_factors(layer):
    """Return fp32 (A, B) such that the adapter delta is B @ A."""

    if layer.adapter_type in _XS_ADAPTER_TYPES:
        return layer.V.T, layer.U @ layer.M_xs.astype(mx.float32)
    return layer.lora_A, layer.lora_B


def _reformulated_dora_linear_forward(layer, x):
    """Apply linear DoRA without materializing its full fp32 adapted weight.

    For V = W0 + scaling * B @ A, row DoRA scales the combined base and
    low-rank output, while column DoRA scales the input. The expanded norm
    below is mathematically equivalent to constructing V, but only rank-sized
    products touch the frozen base projection.
    """

    a, b = _effective_low_rank_factors(layer)
    weight = layer.base.weight
    bias = getattr(layer.base, "bias", None)
    scaling = float(layer.scaling)

    x32 = x.astype(mx.float32)
    if layer.adapter_type in _DORA_COL_ADAPTER_TYPES:
        x32 = x32 * _dora_scale_no_materialize(
            layer,
            a,
            b,
            weight,
            norm_dim=0,
        )
        base_output = x32.astype(x.dtype) @ weight.T
    else:
        base_output = x @ weight.T
    output = base_output.astype(mx.float32) + ((x32 @ a.T) @ b.T) * scaling
    if layer.adapter_type in _DORA_ROW_ADAPTER_TYPES:
        output = output * _dora_scale_no_materialize(
            layer,
            a,
            b,
            weight,
            norm_dim=1,
        )
    if bias is not None:
        output = output + bias.astype(mx.float32)
    return output.astype(x.dtype)


def _dora_scale_no_materialize(layer, a, b, weight, *, norm_dim: int):
    return layer.magnitude.astype(mx.float32) / _v_norm_no_materialize(
        layer,
        a,
        b,
        weight,
        norm_dim=norm_dim,
    )


def _v_norm_no_materialize(layer, a, b, weight, *, norm_dim: int):
    """Compute norm(W0 + scaling * B @ A) from rank-sized products."""

    scaling = float(layer.scaling)
    if norm_dim == 1:
        cross_lhs = mx.matmul(weight, a.T.astype(weight.dtype)).astype(
            mx.float32
        )
        cross = mx.sum(cross_lhs * b, axis=1)
        gram = a @ a.T
        quad = mx.sum((b @ gram) * b, axis=1)
    else:
        cross_lhs = mx.matmul(weight.T, b.astype(weight.dtype)).astype(
            mx.float32
        )
        cross = mx.sum(cross_lhs * a.T, axis=1)
        gram = b.T @ b
        quad = mx.sum((gram @ a) * a, axis=0)
    norm_sq = layer._w0_sq + 2.0 * scaling * cross + scaling * scaling * quad
    norm = mx.sqrt(mx.maximum(norm_sq, 0.0))
    return mx.maximum(norm, 1e-12)


def _dora_weight_2d(v, *, magnitude, norm_dim: int):
    norms = mx.sqrt(mx.sum(v**2, axis=norm_dim, keepdims=True))
    v_hat = v / mx.maximum(norms, 1e-12)
    if norm_dim == 1:
        return v_hat * magnitude.astype(mx.float32)[:, None]
    return v_hat * magnitude.astype(mx.float32)[None, :]


def _bora_weight_2d(v, *, magnitude_r, magnitude_c):
    row_norms = mx.sqrt(mx.sum(v**2, axis=1, keepdims=True))
    row_scaled = (
        v / mx.maximum(row_norms, 1e-12)
    ) * magnitude_r.astype(mx.float32)[:, None]
    column_norms = mx.sqrt(mx.sum(row_scaled**2, axis=0, keepdims=True))
    return (
        row_scaled / mx.maximum(column_norms, 1e-12)
    ) * magnitude_c.astype(mx.float32)[None, :]


def _validate_rank(rank: int, *, fan_out: int, fan_in: int, source_name: str) -> None:
    max_rank = min(fan_out, fan_in)
    if rank > max_rank:
        raise ValueError(
            f"Adapter rank {rank} exceeds maximum rank {max_rank} for "
            f"{source_name!r} with shape ({fan_out}, {fan_in})."
        )


def _svd_bases(weight_2d, rank: int):
    source = np.asarray(weight_2d, dtype=np.float32)
    u, _, vh = np.linalg.svd(source, full_matrices=False)
    u, vh = _canonicalize_svd_signs(u, vh)
    return (
        mx.array(u[:, :rank].copy(), dtype=mx.float32),
        mx.array(vh[:rank, :].T.copy(), dtype=mx.float32),
    )


def _canonicalize_svd_signs(u: np.ndarray, vh: np.ndarray):
    max_abs_indices = np.abs(u).argmax(axis=0)
    signs = np.sign(u[max_abs_indices, np.arange(u.shape[1])])
    signs[signs == 0] = 1
    return u * signs[None, :], vh * signs[:, None]


def _matches_any(name: str, patterns: tp.Sequence[str]) -> bool:
    return any(expanded in name for pattern in patterns for expanded in _expand(pattern))


def _expand(pattern: str) -> list[str]:
    parts = re.split(r"\[(\d+)-(\d+)\]", pattern)
    if len(parts) == 1:
        return [pattern]

    literals = parts[0::3]
    starts = parts[1::3]
    ends = parts[2::3]
    ranges = []
    for start, end in zip(starts, ends, strict=True):
        start_value = int(start)
        end_value = int(end)
        step = 1 if end_value >= start_value else -1
        ranges.append(
            [str(value) for value in range(start_value, end_value + step, step)]
        )

    expanded = []
    for values in product(*ranges):
        pieces = []
        for index, literal in enumerate(literals):
            pieces.append(literal)
            if index < len(values):
                pieces.append(values[index])
        expanded.append("".join(pieces))
    return expanded
