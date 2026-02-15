"""MLX-native conditioning path for Stable Audio Open models."""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .t5_embedder import T5Embedder


def _to_numpy(value: tp.Any) -> np.ndarray:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _mx_dtype_name(dtype: mx.Dtype) -> str:
    if dtype == mx.float16:
        return "float16"
    if dtype == mx.bfloat16:
        return "bfloat16"
    return "float32"


@dataclass(frozen=True)
class LinearLayer:
    """Simple frozen linear layer weights in MLX layout."""

    weight_t: mx.array
    bias: mx.array | None

    @classmethod
    def from_torch_module(cls, module: tp.Any, *, dtype: mx.Dtype = mx.float32) -> "LinearLayer | None":
        if module is None:
            return None
        if not hasattr(module, "weight"):
            return None
        weight = _to_numpy(module.weight).astype(np.float32, copy=False)
        bias = None
        if hasattr(module, "bias") and module.bias is not None:
            bias = _to_numpy(module.bias).astype(np.float32, copy=False)
        return cls(
            weight_t=mx.array(weight.T).astype(dtype),
            bias=(None if bias is None else mx.array(bias).astype(dtype)),
        )

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight_t
        if self.bias is not None:
            y = y + self.bias
        return y


class T5ConditionerMLX:
    """MLX implementation of SAT T5Conditioner (encode + optional projection)."""

    def __init__(
        self,
        *,
        t5_model_name: str,
        max_length: int,
        proj_out: LinearLayer | None = None,
        dtype: mx.Dtype = mx.float32,
    ):
        self.dtype = dtype
        self.embedder = T5Embedder(
            model_name=t5_model_name,
            max_length=max_length,
            dtype=_mx_dtype_name(dtype),
        )
        self.proj_out = proj_out

    @classmethod
    def from_torch_conditioner(
        cls,
        *,
        config: dict[str, tp.Any],
        torch_conditioner: tp.Any | None,
        dtype: mx.Dtype = mx.float32,
    ) -> "T5ConditionerMLX":
        model_name = str(config.get("t5_model_name", "t5-base"))
        max_length = int(config.get("max_length", 64))
        proj_out = None
        if torch_conditioner is not None and hasattr(torch_conditioner, "proj_out"):
            proj_out = LinearLayer.from_torch_module(torch_conditioner.proj_out, dtype=dtype)
        return cls(
            t5_model_name=model_name,
            max_length=max_length,
            proj_out=proj_out,
            dtype=dtype,
        )

    def __call__(self, texts: tp.Sequence[str]) -> tuple[mx.array, mx.array]:
        embeddings, mask = self.embedder.encode(texts)
        embeddings = embeddings.astype(self.dtype)
        if self.proj_out is not None:
            embeddings = self.proj_out(embeddings)
        embeddings = embeddings * mask[..., None].astype(embeddings.dtype)
        return embeddings, mask.astype(mx.bool_)


class NumberConditionerMLX:
    """MLX implementation of SAT NumberConditioner with converted torch weights."""

    def __init__(
        self,
        *,
        min_val: float,
        max_val: float,
        frequencies: mx.array,
        linear: LinearLayer,
        proj_out: LinearLayer | None = None,
        dtype: mx.Dtype = mx.float32,
    ):
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.dtype = dtype
        self.frequencies = frequencies.astype(dtype)
        self.linear = linear
        self.proj_out = proj_out

    @classmethod
    def from_torch_conditioner(
        cls,
        torch_conditioner: tp.Any,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> "NumberConditionerMLX":
        if torch_conditioner is None:
            raise ValueError("NumberConditionerMLX requires a torch number conditioner.")

        state = torch_conditioner.embedder.state_dict()
        frequencies = _to_numpy(state["embedding.0.weights"]).astype(np.float32, copy=False)
        linear_weight = _to_numpy(state["embedding.1.weight"]).astype(np.float32, copy=False)
        linear_bias = _to_numpy(state["embedding.1.bias"]).astype(np.float32, copy=False)

        linear = LinearLayer(
            weight_t=mx.array(linear_weight.T).astype(dtype),
            bias=mx.array(linear_bias).astype(dtype),
        )

        proj_out = None
        if hasattr(torch_conditioner, "proj_out"):
            proj_out = LinearLayer.from_torch_module(torch_conditioner.proj_out, dtype=dtype)

        return cls(
            min_val=float(torch_conditioner.min_val),
            max_val=float(torch_conditioner.max_val),
            frequencies=mx.array(frequencies),
            linear=linear,
            proj_out=proj_out,
            dtype=dtype,
        )

    def __call__(self, values: tp.Sequence[tp.Any]) -> tuple[mx.array, mx.array]:
        numbers = np.asarray([float(v) for v in values], dtype=np.float32)
        numbers = np.clip(numbers, self.min_val, self.max_val)
        denom = self.max_val - self.min_val
        if abs(denom) < 1e-12:
            normalized = np.zeros_like(numbers)
        else:
            normalized = (numbers - self.min_val) / denom

        x = mx.array(normalized).astype(self.dtype)  # [B]
        freqs = x[:, None] * self.frequencies[None, :] * (2.0 * math.pi)  # [B, H]
        fouriered = mx.concatenate([x[:, None], mx.sin(freqs), mx.cos(freqs)], axis=-1)  # [B, 1+2H]

        embeddings = self.linear(fouriered)
        if self.proj_out is not None:
            embeddings = self.proj_out(embeddings)

        embeddings = embeddings[:, None, :]  # [B, 1, D]
        mask = mx.ones((embeddings.shape[0], 1), dtype=mx.bool_)
        return embeddings, mask


class SAOMLXConditioner:
    """Subset of SAT MultiConditioner + get_conditioning_inputs for SAO."""

    def __init__(
        self,
        *,
        conditioners: dict[str, tp.Callable[[tp.Sequence[tp.Any]], tuple[mx.array, mx.array | None]]],
        default_keys: dict[str, str] | None = None,
        cross_attn_cond_ids: tp.Sequence[str] | None = None,
        global_cond_ids: tp.Sequence[str] | None = None,
        input_concat_ids: tp.Sequence[str] | None = None,
        prepend_cond_ids: tp.Sequence[str] | None = None,
    ):
        self.conditioners = dict(conditioners)
        self.default_keys = dict(default_keys or {})
        self.cross_attn_cond_ids = list(cross_attn_cond_ids or [])
        self.global_cond_ids = list(global_cond_ids or [])
        self.input_concat_ids = list(input_concat_ids or [])
        self.prepend_cond_ids = list(prepend_cond_ids or [])

    @classmethod
    def from_torch_model(
        cls,
        *,
        torch_model: tp.Any,
        model_config: dict[str, tp.Any],
        dtype: mx.Dtype = mx.float32,
    ) -> "SAOMLXConditioner":
        conditioning_cfg = model_config.get("model", {}).get("conditioning", {})
        diffusion_cfg = model_config.get("model", {}).get("diffusion", {})
        cond_infos = conditioning_cfg.get("configs", [])

        torch_conditioners = {}
        if hasattr(torch_model, "conditioner") and hasattr(torch_model.conditioner, "conditioners"):
            torch_conditioners = dict(torch_model.conditioner.conditioners)

        conditioners: dict[str, tp.Callable[[tp.Sequence[tp.Any]], tuple[mx.array, mx.array | None]]] = {}
        for info in cond_infos:
            cond_id = info["id"]
            cond_type = info["type"]
            config = dict(info.get("config", {}))
            torch_conditioner = torch_conditioners.get(cond_id)

            if cond_type == "t5":
                conditioners[cond_id] = T5ConditionerMLX.from_torch_conditioner(
                    config=config,
                    torch_conditioner=torch_conditioner,
                    dtype=dtype,
                )
            elif cond_type == "number":
                conditioners[cond_id] = NumberConditionerMLX.from_torch_conditioner(
                    torch_conditioner=torch_conditioner,
                    dtype=dtype,
                )
            else:
                raise NotImplementedError(
                    f"Conditioner type '{cond_type}' is not implemented in MLX path yet."
                )

        return cls(
            conditioners=conditioners,
            default_keys=conditioning_cfg.get("default_keys", {}),
            cross_attn_cond_ids=diffusion_cfg.get("cross_attention_cond_ids"),
            global_cond_ids=diffusion_cfg.get("global_cond_ids"),
            input_concat_ids=diffusion_cfg.get("input_concat_ids"),
            prepend_cond_ids=diffusion_cfg.get("prepend_cond_ids"),
        )

    def _collect_conditioner_inputs(
        self,
        *,
        key: str,
        batch_metadata: list[dict[str, tp.Any]],
    ) -> list[tp.Any]:
        out: list[tp.Any] = []
        for item in batch_metadata:
            lookup = key
            if lookup not in item:
                if lookup in self.default_keys:
                    lookup = self.default_keys[lookup]
                else:
                    raise ValueError(f"Conditioner key '{lookup}' not found in batch metadata")
            value = item[lookup]
            if isinstance(value, list) or (isinstance(value, tuple) and len(value) == 1):
                value = value[0]
            out.append(value)
        return out

    def condition(self, batch_metadata: list[dict[str, tp.Any]]) -> dict[str, tuple[mx.array, mx.array | None]]:
        tensors: dict[str, tuple[mx.array, mx.array | None]] = {}
        for key, conditioner in self.conditioners.items():
            inputs = self._collect_conditioner_inputs(key=key, batch_metadata=batch_metadata)
            tensors[key] = conditioner(inputs)
        return tensors

    def _assemble_inputs(
        self,
        conditioning_tensors: dict[str, tuple[mx.array, mx.array | None]],
        *,
        negative: bool,
    ) -> dict[str, mx.array | None]:
        cross_attention_input = None
        cross_attention_masks = None
        global_cond = None
        input_concat_cond = None
        prepend_cond = None
        prepend_cond_mask = None

        if self.cross_attn_cond_ids:
            cross_inputs: list[mx.array] = []
            cross_masks: list[mx.array] = []
            for key in self.cross_attn_cond_ids:
                cross_in, cross_mask = conditioning_tensors[key]
                if cross_in.ndim == 2:
                    cross_in = cross_in[:, None, :]
                if cross_mask is None:
                    cross_mask = mx.ones((cross_in.shape[0], cross_in.shape[1]), dtype=mx.bool_)
                elif cross_mask.ndim == 1:
                    cross_mask = cross_mask[:, None]
                cross_inputs.append(cross_in)
                cross_masks.append(cross_mask.astype(mx.bool_))

            cross_attention_input = mx.concatenate(cross_inputs, axis=1)
            cross_attention_masks = mx.concatenate(cross_masks, axis=1)

        if self.global_cond_ids:
            global_conds = [conditioning_tensors[key][0] for key in self.global_cond_ids]
            global_cond = mx.concatenate(global_conds, axis=-1)
            if global_cond.ndim == 3 and global_cond.shape[1] == 1:
                global_cond = mx.squeeze(global_cond, axis=1)

        if self.input_concat_ids:
            input_concat_cond = mx.concatenate(
                [conditioning_tensors[key][0] for key in self.input_concat_ids],
                axis=1,
            )

        if self.prepend_cond_ids:
            prepend_cond = mx.concatenate(
                [conditioning_tensors[key][0] for key in self.prepend_cond_ids],
                axis=1,
            )
            prepend_cond_mask = mx.concatenate(
                [
                    conditioning_tensors[key][1]
                    if conditioning_tensors[key][1] is not None
                    else mx.ones((conditioning_tensors[key][0].shape[0], conditioning_tensors[key][0].shape[1]), dtype=mx.bool_)
                    for key in self.prepend_cond_ids
                ],
                axis=1,
            ).astype(mx.bool_)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond,
            }

        return {
            "cross_attn_cond": cross_attention_input,
            "cross_attn_mask": cross_attention_masks,
            "global_cond": global_cond,
            "input_concat_cond": input_concat_cond,
            "prepend_cond": prepend_cond,
            "prepend_cond_mask": prepend_cond_mask,
        }

    def get_conditioning_inputs(
        self,
        batch_metadata: list[dict[str, tp.Any]],
        *,
        negative: bool = False,
    ) -> dict[str, mx.array | None]:
        conditioning_tensors = self.condition(batch_metadata)
        return self._assemble_inputs(conditioning_tensors, negative=negative)
