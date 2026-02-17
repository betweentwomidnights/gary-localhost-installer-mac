#!/usr/bin/env python3
"""MLX FlowModel runtime conversion helpers for MelodyFlow localhost backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np


def linear(x: mx.array, w_out_in: mx.array, b: mx.array | None = None) -> mx.array:
    y = x @ mx.transpose(w_out_in, (1, 0))
    if b is not None:
        y = y + b
    return y


def layer_norm(x: mx.array, gamma: mx.array, beta: mx.array, eps: float = 1e-5) -> mx.array:
    return mx.fast.layer_norm(x, gamma, beta, eps)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = mx.stack((-x2, x1), axis=-1)
    return stacked.reshape(x.shape)


def apply_rope(q: mx.array, k: mx.array, frequencies: mx.array) -> tuple[mx.array, mx.array]:
    t = q.shape[2]
    idx = mx.arange(t, dtype=mx.float32)
    angles = idx[:, None] * frequencies[None, :]
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    cos = mx.repeat(cos, 2, axis=-1)[None, None, :, :]
    sin = mx.repeat(sin, 2, axis=-1)[None, None, :, :]
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


@dataclass
class MlxAttention:
    in_proj_weight: mx.array
    out_proj_weight: mx.array
    num_heads: int
    add_zero_attn: bool
    rope_frequencies: mx.array | None = None

    def _project_self(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        proj = linear(x, self.in_proj_weight)
        q, k, v = mx.split(proj, 3, axis=-1)
        return q, k, v

    def _project_cross(self, q_in: mx.array, kv_in: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        d = self.in_proj_weight.shape[1]
        wq = self.in_proj_weight[:d, :]
        wk = self.in_proj_weight[d : 2 * d, :]
        wv = self.in_proj_weight[2 * d :, :]
        q = linear(q_in, wq)
        k = linear(kv_in, wk)
        v = linear(kv_in, wv)
        return q, k, v

    def _reshape_heads(self, x: mx.array) -> mx.array:
        b, t, d = x.shape
        hd = d // self.num_heads
        x = x.reshape(b, t, self.num_heads, hd)
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: mx.array) -> mx.array:
        b, h, t, d = x.shape
        return mx.transpose(x, (0, 2, 1, 3)).reshape(b, t, h * d)

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        *,
        attn_mask: mx.array | None = None,
        use_rope: bool = False,
        is_cross: bool = False,
    ) -> mx.array:
        if is_cross:
            q, k, v = self._project_cross(query, key)
        else:
            q, k, v = self._project_self(query)

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        if use_rope and self.rope_frequencies is not None:
            q, k = apply_rope(q, k, self.rope_frequencies)

        if self.add_zero_attn:
            zero_k = mx.zeros((k.shape[0], k.shape[1], 1, k.shape[3]), dtype=k.dtype)
            zero_v = mx.zeros((v.shape[0], v.shape[1], 1, v.shape[3]), dtype=v.dtype)
            k = mx.concatenate([zero_k, k], axis=2)
            v = mx.concatenate([zero_v, v], axis=2)
            if attn_mask is not None:
                zero_col = mx.zeros(
                    (attn_mask.shape[0], attn_mask.shape[1], attn_mask.shape[2], 1), dtype=attn_mask.dtype
                )
                attn_mask = mx.concatenate([zero_col, attn_mask], axis=-1)

        scale = 1.0 / math.sqrt(float(q.shape[-1]))
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale
        if attn_mask is not None:
            if attn_mask.shape[0] == 1 and scores.shape[0] > 1:
                attn_mask = mx.repeat(attn_mask, scores.shape[0], axis=0)
            scores = scores + attn_mask.astype(scores.dtype)
        probs = mx.softmax(scores, axis=-1)
        out = probs @ v
        out = self._merge_heads(out)
        return linear(out, self.out_proj_weight)


@dataclass
class MlxFlowLayer:
    norm1_w: mx.array
    norm1_b: mx.array
    norm_cross_w: mx.array
    norm_cross_b: mx.array
    norm2_w: mx.array
    norm2_b: mx.array
    self_attn: MlxAttention
    cross_attn: MlxAttention
    ff1_w: mx.array
    ff2_w: mx.array

    def __call__(
        self,
        x: mx.array,
        *,
        cross_src: mx.array,
        cross_mask: mx.array | None,
        timestep_embedding: mx.array | None,
    ) -> mx.array:
        xn = layer_norm(x, self.norm1_w, self.norm1_b)
        if timestep_embedding is not None:
            xn = xn + timestep_embedding
        x = x + self.self_attn(xn, xn, xn, use_rope=True, is_cross=False)

        xn = layer_norm(x, self.norm_cross_w, self.norm_cross_b)
        if timestep_embedding is not None:
            xn = xn + timestep_embedding
        x = x + self.cross_attn(
            xn,
            cross_src,
            cross_src,
            attn_mask=cross_mask,
            use_rope=False,
            is_cross=True,
        )

        xn = layer_norm(x, self.norm2_w, self.norm2_b)
        ff = linear(nn_gelu(linear(xn, self.ff1_w)), self.ff2_w)
        x = x + ff
        return x


def nn_gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def nn_silu(x: mx.array) -> mx.array:
    return x * (1.0 / (1.0 + mx.exp(-x)))


@dataclass
class MlxFlowModel:
    in_proj_w: mx.array
    out_proj_w: mx.array
    out_norm_w: mx.array
    out_norm_b: mx.array
    t_mlp0_w: mx.array
    t_mlp2_w: mx.array
    t_freq_dim: int
    layers: list[MlxFlowLayer]
    skip_projections: list[mx.array]
    skip_connections: bool

    def timestep_embedding(self, t: mx.array) -> mx.array:
        if t.ndim == 1:
            t = t[:, None]
        dim = self.t_freq_dim
        half = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(start=0, stop=half, dtype=mx.float32) / float(half))
        args = t.astype(mx.float32)[:, :, None] * freqs[None, None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :, :1])], axis=-1)
        return emb

    def timestep_embedder(self, t: mx.array) -> mx.array:
        x = self.timestep_embedding(t)
        x = linear(x, self.t_mlp0_w)
        x = nn_silu(x)
        x = linear(x, self.t_mlp2_w)
        return x

    def forward(
        self,
        z: mx.array,
        t: mx.array,
        condition_src: mx.array,
        condition_mask_log: mx.array | None,
    ) -> mx.array:
        denom = mx.sqrt(mx.power(t[..., None], 2) + mx.power(1.0 - t[..., None], 2))
        x = z / denom
        x = mx.transpose(x, (0, 2, 1))
        x = linear(x, self.in_proj_w)

        t_emb = self.timestep_embedder(t)

        states: list[mx.array] = []
        n_layers = len(self.layers)
        for idx, layer in enumerate(self.layers):
            if self.skip_connections and idx > n_layers / 2:
                skip = states.pop()
                x = mx.concatenate([x, skip], axis=-1)
                proj = self.skip_projections[idx % len(self.skip_projections)]
                x = linear(x, proj)
            x = layer(
                x,
                cross_src=condition_src,
                cross_mask=condition_mask_log,
                timestep_embedding=t_emb,
            )
            if self.skip_connections and idx < n_layers / 2 - 1:
                states.append(x)

        x = layer_norm(x, self.out_norm_w, self.out_norm_b)
        x = linear(x, self.out_proj_w)
        x = mx.transpose(x, (0, 2, 1)) * math.sqrt(2.0)
        return x


def state_np(state_dict: dict[str, Any], key: str, dtype: Any) -> mx.array:
    t = state_dict[key].detach().cpu().numpy().astype(np.float32, copy=False)
    arr = mx.array(t)
    if arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


def build_mlx_flow_from_state(state_dict: dict[str, Any], cfg: Any, dtype: Any) -> MlxFlowModel:
    layer_indices = []
    for k in state_dict.keys():
        if k.startswith("transformer.layers.") and ".self_attn.in_proj_weight" in k:
            idx = int(k.split(".")[2])
            layer_indices.append(idx)
    n_layers = max(layer_indices) + 1

    layers: list[MlxFlowLayer] = []
    for i in range(n_layers):
        prefix = f"transformer.layers.{i}"
        layers.append(
            MlxFlowLayer(
                norm1_w=state_np(state_dict, f"{prefix}.norm1.weight", dtype),
                norm1_b=state_np(state_dict, f"{prefix}.norm1.bias", dtype),
                norm_cross_w=state_np(state_dict, f"{prefix}.norm_cross.weight", dtype),
                norm_cross_b=state_np(state_dict, f"{prefix}.norm_cross.bias", dtype),
                norm2_w=state_np(state_dict, f"{prefix}.norm2.weight", dtype),
                norm2_b=state_np(state_dict, f"{prefix}.norm2.bias", dtype),
                self_attn=MlxAttention(
                    in_proj_weight=state_np(state_dict, f"{prefix}.self_attn.in_proj_weight", dtype),
                    out_proj_weight=state_np(state_dict, f"{prefix}.self_attn.out_proj.weight", dtype),
                    num_heads=int(cfg.transformer_lm.num_heads),
                    add_zero_attn=True,
                    rope_frequencies=state_np(state_dict, f"{prefix}.self_attn.rope.frequencies", dtype),
                ),
                cross_attn=MlxAttention(
                    in_proj_weight=state_np(state_dict, f"{prefix}.cross_attention.in_proj_weight", dtype),
                    out_proj_weight=state_np(state_dict, f"{prefix}.cross_attention.out_proj.weight", dtype),
                    num_heads=int(cfg.transformer_lm.num_heads),
                    add_zero_attn=True,
                    rope_frequencies=None,
                ),
                ff1_w=state_np(state_dict, f"{prefix}.linear1.weight", dtype),
                ff2_w=state_np(state_dict, f"{prefix}.linear2.weight", dtype),
            )
        )

    skip_projections: list[mx.array] = []
    i = 0
    while f"transformer.skip_projections.{i}.weight" in state_dict:
        skip_projections.append(state_np(state_dict, f"transformer.skip_projections.{i}.weight", dtype))
        i += 1

    return MlxFlowModel(
        in_proj_w=state_np(state_dict, "in_proj.weight", dtype),
        out_proj_w=state_np(state_dict, "out_proj.weight", dtype),
        out_norm_w=state_np(state_dict, "out_norm.weight", dtype),
        out_norm_b=state_np(state_dict, "out_norm.bias", dtype),
        t_mlp0_w=state_np(state_dict, "timestep_embedder.mlp.0.weight", dtype),
        t_mlp2_w=state_np(state_dict, "timestep_embedder.mlp.2.weight", dtype),
        t_freq_dim=256,
        layers=layers,
        skip_projections=skip_projections,
        skip_connections=bool(cfg.transformer_lm.get("skip_connections", False)),
    )
