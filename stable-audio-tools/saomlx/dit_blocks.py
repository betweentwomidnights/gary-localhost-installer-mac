"""DiT block components for SAO models (MLX implementation)."""

from __future__ import annotations

import math
import typing as tp

import mlx.core as mx
import mlx.nn as nn


class Identity(nn.Module):
    def __call__(self, x):
        return x


def run_layers(layers: list[tp.Any], x):
    h = x
    for layer in layers:
        h = layer(h)
    return h


class LayerNormGammaBeta(nn.Module):
    """LayerNorm variant matching SAT key names (`gamma`/`beta`)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = mx.ones((dim,), dtype=mx.float32)
        self.beta = mx.zeros((dim,), dtype=mx.float32)
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.layer_norm(x, self.gamma, self.beta, self.eps)


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")
        self.weight = mx.random.normal((out_features // 2, in_features), dtype=mx.float32) * float(std)

    def __call__(self, x: mx.array) -> mx.array:
        f = (2.0 * math.pi) * (x @ mx.transpose(self.weight, (1, 0)))
        return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        use_xpos: bool = False,
        interpolation_factor: float = 1.0,
        base: float = 10000.0,
        base_rescale_factor: float = 1.0,
    ):
        super().__init__()
        if use_xpos:
            raise NotImplementedError("xPos rotary scaling is not implemented in MLX DiT blocks.")
        if interpolation_factor < 1.0:
            raise ValueError("interpolation_factor must be >= 1.0")

        base = float(base) * float(base_rescale_factor) ** (dim / (dim - 2))
        freqs = mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)
        self.inv_freq = 1.0 / (base ** freqs)
        self.interpolation_factor = float(interpolation_factor)

    def forward_from_seq_len(self, seq_len: int) -> tuple[mx.array, float]:
        t = mx.arange(seq_len, dtype=mx.float32)
        return self(t)

    def __call__(self, t: mx.array) -> tuple[mx.array, float]:
        t = t.astype(mx.float32) / self.interpolation_factor
        freqs = t[:, None] * self.inv_freq[None, :]
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs, 1.0


def rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(t: mx.array, freqs: mx.array, scale: float | mx.array = 1.0) -> mx.array:
    rot_dim = min(freqs.shape[-1], t.shape[-1])
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :rot_dim]

    # Broadcast freqs to [B, H, N, D] when attention tensors include batch/head dims.
    while freqs.ndim < t.ndim:
        freqs = freqs[None, ...]

    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * mx.cos(freqs) * scale) + (rotate_half(t_rot) * mx.sin(freqs) * scale)
    return mx.concatenate([t_rot, t_pass], axis=-1)


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x_proj = self.proj(x)
        x_main, x_gate = mx.split(x_proj, 2, axis=-1)
        return x_main * nn.silu(x_gate)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        no_bias: bool = False,
        zero_init_output: bool = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out

        linear_in = GLU(dim, inner_dim)
        linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias)
        if zero_init_output:
            linear_out.weight = mx.zeros_like(linear_out.weight)
            if linear_out.bias is not None:
                linear_out.bias = mx.zeros_like(linear_out.bias)

        # Keep list-based structure so key names align with SAT (`ff.ff.0...`, `ff.ff.2...`).
        self.ff = [linear_in, Identity(), linear_out, Identity()]

    def __call__(self, x: mx.array) -> mx.array:
        return run_layers(self.ff, x)


def _reshape_heads(x: mx.array, num_heads: int) -> mx.array:
    b, n, d = x.shape
    if d % num_heads != 0:
        raise ValueError(f"Embedding dim {d} is not divisible by num_heads {num_heads}")
    dh = d // num_heads
    x = x.reshape(b, n, num_heads, dh)
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    b, h, n, d = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return x.reshape(b, n, h * d)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        dim_context: int | None = None,
        causal: bool = False,
        zero_init_output: bool = True,
        qk_norm: str = "none",
    ):
        super().__init__()
        self.dim = int(dim)
        self.dim_heads = int(dim_heads)
        if self.dim % self.dim_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by dim_heads ({self.dim_heads})")

        dim_kv = int(dim_context) if dim_context is not None else self.dim
        if dim_kv % self.dim_heads != 0:
            raise ValueError(f"dim_kv ({dim_kv}) must be divisible by dim_heads ({self.dim_heads})")

        self.num_heads = self.dim // self.dim_heads
        self.kv_heads = dim_kv // self.dim_heads
        self.causal = bool(causal)

        if dim_context is not None:
            self.to_q = nn.Linear(self.dim, self.dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)

        self.to_out = nn.Linear(self.dim, self.dim, bias=False)
        if zero_init_output:
            self.to_out.weight = mx.zeros_like(self.to_out.weight)

        if qk_norm not in {"none", "l2", "ln"}:
            raise ValueError(f"Unsupported qk_norm '{qk_norm}'")
        self.qk_norm = qk_norm
        if qk_norm == "ln":
            self.q_norm = nn.LayerNorm(self.dim_heads, eps=1e-6)
            self.k_norm = nn.LayerNorm(self.dim_heads, eps=1e-6)

    def __call__(
        self,
        x: mx.array,
        *,
        context: mx.array | None = None,
        rotary_pos_emb: tuple[mx.array, float] | None = None,
        causal: bool | None = None,
    ) -> mx.array:
        kv_input = context if context is not None else x

        if hasattr(self, "to_q"):
            q = self.to_q(x)
            k, v = mx.split(self.to_kv(kv_input), 2, axis=-1)
        else:
            q, k, v = mx.split(self.to_qkv(x), 3, axis=-1)

        q = _reshape_heads(q, self.num_heads)
        k = _reshape_heads(k, self.kv_heads)
        v = _reshape_heads(v, self.kv_heads)

        if self.qk_norm == "l2":
            q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + 1e-12)
            k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + 1e-12)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            if q.shape[-2] >= k.shape[-2]:
                ratio = float(q.shape[-2]) / float(k.shape[-2])
                q_freqs, k_freqs = freqs, ratio * freqs
            else:
                ratio = float(k.shape[-2]) / float(q.shape[-2])
                q_freqs, k_freqs = ratio * freqs, freqs
            q = apply_rotary_pos_emb(q, q_freqs)
            k = apply_rotary_pos_emb(k, k_freqs)

        use_causal = self.causal if causal is None else causal
        mask: str | None = "causal" if use_causal else None
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.dim_heads**-0.5,
            mask=mask,
        )
        out = _merge_heads(out)
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: int | None = None,
        global_cond_dim: int | None = None,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        add_rope: bool = False,
        attn_kwargs: dict[str, tp.Any] | None = None,
        ff_kwargs: dict[str, tp.Any] | None = None,
        norm_kwargs: dict[str, tp.Any] | None = None,
    ):
        super().__init__()
        attn_kwargs = attn_kwargs or {}
        ff_kwargs = ff_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        if global_cond_dim is not None:
            raise NotImplementedError("adaLN/global_cond path is not implemented in MLX TransformerBlock yet.")

        self.dim = int(dim)
        self.dim_heads = int(min(dim_heads, dim))
        self.cross_attend = bool(cross_attend)
        self.causal = bool(causal)
        self.add_rope = bool(add_rope)

        self.pre_norm = LayerNormGammaBeta(self.dim, **norm_kwargs)
        self.self_attn = Attention(
            self.dim,
            dim_heads=self.dim_heads,
            causal=self.causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.self_attn_scale = Identity()

        if self.cross_attend:
            if dim_context is None:
                raise ValueError("dim_context must be set when cross_attend=True")
            self.cross_attend_norm = LayerNormGammaBeta(self.dim, **norm_kwargs)
            self.cross_attn = Attention(
                self.dim,
                dim_heads=self.dim_heads,
                dim_context=dim_context,
                causal=self.causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = Identity()

        self.ff_norm = LayerNormGammaBeta(self.dim, **norm_kwargs)
        self.ff = FeedForward(self.dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = Identity()

        self.rope = RotaryEmbedding(self.dim_heads // 2) if self.add_rope else None

    def __call__(
        self,
        x: mx.array,
        *,
        context: mx.array | None = None,
        rotary_pos_emb: tuple[mx.array, float] | None = None,
        global_cond: mx.array | None = None,
    ) -> mx.array:
        if global_cond is not None:
            raise NotImplementedError("global_cond path is not implemented in MLX TransformerBlock yet.")

        if rotary_pos_emb is None and self.add_rope and self.rope is not None:
            rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

        x = x + self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb))

        if context is not None and self.cross_attend:
            x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context=context))

        x = x + self.ff_scale(self.ff(self.ff_norm(x)))
        return x


class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        dim_in: int | None = None,
        dim_out: int | None = None,
        dim_heads: int = 64,
        cross_attend: bool = False,
        cond_token_dim: int | None = None,
        final_cross_attn_ix: int = -1,
        global_cond_dim: int | None = None,
        causal: bool = False,
        rotary_pos_emb: bool = True,
        zero_init_branch_outputs: bool = True,
        attn_kwargs: dict[str, tp.Any] | None = None,
        ff_kwargs: dict[str, tp.Any] | None = None,
        norm_kwargs: dict[str, tp.Any] | None = None,
        **_: tp.Any,
    ):
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.causal = bool(causal)

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else Identity()
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb else None

        self.layers = []
        for i in range(self.depth):
            should_cross_attend = bool(cross_attend) and (final_cross_attn_ix == -1 or i <= final_cross_attn_ix)
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads=dim_heads,
                    cross_attend=should_cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    add_rope=False,
                    attn_kwargs=attn_kwargs,
                    ff_kwargs=ff_kwargs,
                    norm_kwargs=norm_kwargs,
                )
            )

    def __call__(
        self,
        x: mx.array,
        *,
        prepend_embeds: mx.array | None = None,
        context: mx.array | None = None,
        global_cond: mx.array | None = None,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
    ):
        info: dict[str, tp.Any] = {"hidden_states": []}

        x = self.project_in(x)

        if prepend_embeds is not None:
            x = mx.concatenate([prepend_embeds, x], axis=1)

        rotary = self.rotary_pos_emb.forward_from_seq_len(x.shape[1]) if self.rotary_pos_emb is not None else None

        for layer_ix, layer in enumerate(self.layers):
            x = layer(x, context=context, global_cond=global_cond, rotary_pos_emb=rotary)
            if return_info:
                info["hidden_states"].append(x)
            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                if return_info:
                    return x, info
                return x

        x = self.project_out(x)
        if return_info:
            return x, info
        return x
