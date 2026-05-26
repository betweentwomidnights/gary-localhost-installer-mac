from __future__ import annotations

import math
import typing as tp

from stable_audio_3.mlx.runtime import import_mlx_core, import_mlx_nn

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)

_ATTENTION_MASK_CACHE: dict[tuple[int, int, bool, tuple[int, int] | None, str], tp.Any] = {}


class Identity(nn.Module):
    def __call__(self, x):
        return x


def run_layers(layers: list[tp.Any], x):
    h = x
    for layer in layers:
        h = layer(h)
    return h


def _left_pad_to_match(emb, target_len: int):
    emb_len = int(emb.shape[-2])
    if emb_len < target_len:
        pad = mx.zeros(
            (emb.shape[0], target_len - emb_len, emb.shape[-1]),
            dtype=emb.dtype,
        )
        return mx.concatenate([pad, emb], axis=-2)
    if emb_len > target_len:
        return emb[:, -target_len:, :]
    return emb


class FourierFeatures(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError(f"out_features must be even, got {out_features}")
        self.weight = mx.random.normal(
            (out_features // 2, in_features),
            dtype=mx.float32,
        ) * float(std)

    def __call__(self, x):
        f = (2.0 * math.pi) * (x @ mx.transpose(self.weight, (1, 0)))
        return mx.concatenate([mx.cos(f), mx.sin(f)], axis=-1)


class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim: int, min_freq: float = 0.5, max_freq: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = int(dim)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)

    def __call__(self, t):
        in_dtype = t.dtype
        t = t.astype(mx.float32)
        if t.ndim == 1:
            t = t[:, None]

        half_dim = self.dim // 2
        ramp = mx.linspace(0.0, 1.0, half_dim, dtype=mx.float32)
        log_min = math.log(self.min_freq)
        log_max = math.log(self.max_freq)
        freqs = mx.exp(ramp * (log_max - log_min) + log_min)
        args = t * freqs[None, :] * (2.0 * math.pi)
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return embedding.astype(in_dtype)


class DynamicTanh(nn.Module):
    def __init__(self, dim: int, init_alpha: float = 4.0, **_: tp.Any):
        super().__init__()
        self.alpha = mx.ones((1,), dtype=mx.float32) * float(init_alpha)
        self.gamma = mx.ones((dim,), dtype=mx.float32)
        self.beta = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        return self.gamma * mx.tanh(self.alpha * x) + self.beta


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        bias: bool = False,
        fix_scale: bool = False,
        force_fp32: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.gamma = mx.ones((dim,), dtype=mx.float32)
        if fix_scale:
            self.gamma = mx.ones((dim,), dtype=mx.float32)
        self.beta = mx.zeros((dim,), dtype=mx.float32)
        self.use_bias = bool(bias)
        self.force_fp32 = bool(force_fp32)
        self.eps = float(eps)

    def __call__(self, x):
        if not self.use_bias:
            beta = mx.zeros_like(self.beta)
        else:
            beta = self.beta

        if self.force_fp32:
            out = mx.fast.layer_norm(
                x.astype(mx.float32),
                self.gamma.astype(mx.float32),
                beta.astype(mx.float32),
                self.eps,
            )
            return out.astype(x.dtype)

        return mx.fast.layer_norm(x, self.gamma, beta, self.eps)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        fix_scale: bool = False,
        force_fp32: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.gamma = mx.ones((dim,), dtype=mx.float32)
        if fix_scale:
            self.gamma = mx.ones((dim,), dtype=mx.float32)
        self.force_fp32 = bool(force_fp32)
        self.eps = float(eps)

    def __call__(self, x):
        if self.force_fp32:
            x_fp32 = x.astype(mx.float32)
            gamma = self.gamma.astype(mx.float32)
            denom = 1.0 / mx.sqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
            out = x_fp32 * denom * gamma
            return out.astype(x.dtype)

        denom = 1.0 / mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * denom * self.gamma


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_val: float = 1e-5):
        super().__init__()
        self.scale = mx.full((dim,), float(init_val), dtype=mx.float32)

    def __call__(self, x):
        return x * self.scale


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
            raise NotImplementedError("xPos rotary scaling is not implemented in the MLX DiT.")
        if interpolation_factor < 1.0:
            raise ValueError("interpolation_factor must be >= 1.0")

        base = float(base) * float(base_rescale_factor) ** (dim / max(dim - 2, 1))
        freqs = mx.arange(0, dim, 2, dtype=mx.float32) / float(dim)
        self.inv_freq = 1.0 / (base ** freqs)
        self.interpolation_factor = float(interpolation_factor)

    def forward_from_seq_len(self, seq_len: int):
        t = mx.arange(seq_len, dtype=mx.float32)
        return self(t)

    def __call__(self, t):
        t = t.astype(mx.float32) / self.interpolation_factor
        freqs = t[:, None] * self.inv_freq[None, :]
        freqs = mx.concatenate([freqs, freqs], axis=-1)
        return freqs, 1.0


def rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(t, freqs, scale: float | tp.Any = 1.0):
    rot_dim = min(int(freqs.shape[-1]), int(t.shape[-1]))
    seq_len = int(t.shape[-2])
    freqs = freqs[-seq_len:, :rot_dim]

    while freqs.ndim < t.ndim:
        freqs = freqs[None, ...]

    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    t_rot = (t_rot * mx.cos(freqs) * scale) + (rotate_half(t_rot) * mx.sin(freqs) * scale)
    return mx.concatenate([t_rot, t_pass], axis=-1)


class GLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, activation: str = "silu"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=True)
        self.activation = str(activation)

    def __call__(self, x):
        x_proj = self.proj(x)
        x_main, x_gate = mx.split(x_proj, 2, axis=-1)
        if self.activation == "silu":
            gate = nn.silu(x_gate)
        elif self.activation == "sin":
            gate = mx.sin(math.pi * x_gate)
        else:
            raise ValueError(f"Unsupported GLU activation: {self.activation!r}")
        return x_main * gate


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: float = 4.0,
        no_bias: bool = False,
        glu: bool = True,
        use_conv: bool = False,
        zero_init_output: bool = True,
        sinusoidal: bool = False,
        **_: tp.Any,
    ):
        super().__init__()
        if use_conv:
            raise NotImplementedError("Convolutional feed-forward blocks are not implemented in the MLX DiT.")
        if not glu:
            raise NotImplementedError("Non-GLU feed-forward blocks are not implemented in the MLX DiT.")

        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out

        linear_in = GLU(dim, inner_dim, activation="sin" if sinusoidal else "silu")
        linear_out = nn.Linear(inner_dim, dim_out, bias=not no_bias)
        if zero_init_output:
            linear_out.weight = mx.zeros_like(linear_out.weight)
            if linear_out.bias is not None:
                linear_out.bias = mx.zeros_like(linear_out.bias)

        self.ff = [linear_in, Identity(), linear_out, Identity()]

    def __call__(self, x):
        return run_layers(self.ff, x)


def _reshape_heads(x, num_heads: int):
    b, n, d = x.shape
    if d % num_heads != 0:
        raise ValueError(f"Embedding dim {d} is not divisible by num_heads {num_heads}")
    dh = d // num_heads
    x = x.reshape(b, n, num_heads, dh)
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x):
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
        qk_norm_eps: float = 1e-6,
        qk_norm: str = "none",
        differential: bool = False,
        feat_scale: bool = False,
    ):
        super().__init__()
        self.dim = int(dim)
        self.dim_heads = int(dim_heads)
        self.causal = bool(causal)
        self.differential = bool(differential)
        self.feat_scale = bool(feat_scale)
        self.qk_norm = str(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)

        dim_kv = int(dim_context) if dim_context is not None else self.dim
        if self.dim % self.dim_heads != 0:
            raise ValueError(f"dim ({self.dim}) must be divisible by dim_heads ({self.dim_heads})")
        if dim_kv % self.dim_heads != 0:
            raise ValueError(f"dim_kv ({dim_kv}) must be divisible by dim_heads ({self.dim_heads})")

        self.num_heads = self.dim // self.dim_heads
        self.kv_heads = dim_kv // self.dim_heads

        if dim_context is not None:
            if self.differential:
                self.to_q = nn.Linear(self.dim, self.dim * 2, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 3, bias=False)
            else:
                self.to_q = nn.Linear(self.dim, self.dim, bias=False)
                self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            if self.differential:
                self.to_qkv = nn.Linear(self.dim, self.dim * 5, bias=False)
            else:
                self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)

        self.to_out = nn.Linear(self.dim, self.dim, bias=False)
        if zero_init_output:
            self.to_out.weight = mx.zeros_like(self.to_out.weight)

        if self.qk_norm not in {"l2", "ln", "rms", "dyt", "none"}:
            raise ValueError(
                'qk_norm must be one of ["l2", "ln", "rms", "dyt", "none"], '
                f"got {self.qk_norm!r}"
            )

        if self.qk_norm == "ln":
            self.q_norm = LayerNorm(self.dim_heads, bias=True, eps=self.qk_norm_eps)
            self.k_norm = LayerNorm(self.dim_heads, bias=True, eps=self.qk_norm_eps)
        elif self.qk_norm == "rms":
            self.q_norm = RMSNorm(self.dim_heads, eps=self.qk_norm_eps)
            self.k_norm = RMSNorm(self.dim_heads, eps=self.qk_norm_eps)
        elif self.qk_norm == "dyt":
            self.q_norm = DynamicTanh(self.dim_heads)
            self.k_norm = DynamicTanh(self.dim_heads)

        if self.feat_scale:
            self.lambda_dc = mx.zeros((self.dim,), dtype=mx.float32)
            self.lambda_hf = mx.zeros((self.dim,), dtype=mx.float32)

    def apply_qk_norm(self, q, k):
        return self.q_norm(q), self.k_norm(k)

    def _attention_mask(self, q_len: int, k_len: int, *, causal: bool, sliding_window=None, dtype=mx.float32):
        if sliding_window is None and not causal:
            return None

        window_key = None
        if sliding_window is not None:
            left, right = sliding_window
            window_key = (int(left), int(right))
        key = (int(q_len), int(k_len), bool(causal), window_key, str(dtype))
        cached = _ATTENTION_MASK_CACHE.get(key)
        if cached is not None:
            return cached

        q_pos = mx.arange(q_len, dtype=mx.int32)[:, None]
        k_pos = mx.arange(k_len, dtype=mx.int32)[None, :]
        valid = mx.ones((q_len, k_len), dtype=mx.bool_)

        if window_key is not None:
            left, right = window_key
            valid = valid & (k_pos >= (q_pos - left)) & (k_pos <= (q_pos + right))

        if causal:
            valid = valid & (k_pos <= q_pos)

        zeros = mx.zeros((q_len, k_len), dtype=dtype)
        mask = mx.where(valid, zeros, -mx.inf)
        _ATTENTION_MASK_CACHE[key] = mask
        return mask

    def apply_attn(
        self,
        q,
        k,
        v,
        *,
        causal: bool | None = None,
        padding_mask=None,
        sliding_window=None,
    ):
        if self.num_heads != self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            k = mx.repeat(k, heads_per_kv_head, axis=1)
            v = mx.repeat(v, heads_per_kv_head, axis=1)

        if padding_mask is not None and k.shape[-2] == padding_mask.shape[-1]:
            mask_expanded = padding_mask[:, None, :, None].astype(v.dtype)
            v = v * mask_expanded

        use_causal = self.causal if causal is None else causal
        attn_mask = self._attention_mask(
            int(q.shape[-2]),
            int(k.shape[-2]),
            causal=bool(use_causal),
            sliding_window=sliding_window,
            dtype=q.dtype,
        )
        return mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.dim_heads**-0.5,
            mask=attn_mask,
        )

    def __call__(
        self,
        x,
        *,
        context=None,
        rotary_pos_emb=None,
        rotary_pos_emb_k=None,
        causal: bool | None = None,
        padding_mask=None,
        sliding_window=None,
    ):
        h = self.num_heads
        kv_h = self.kv_heads
        has_context = context is not None
        q_diff = None
        k_diff = None
        kv_input = context if has_context else x

        if hasattr(self, "to_q"):
            if self.differential:
                q, q_diff = mx.split(self.to_q(x), 2, axis=-1)
                q = _reshape_heads(q, h)
                q_diff = _reshape_heads(q_diff, h)

                k, k_diff, v = mx.split(self.to_kv(kv_input), 3, axis=-1)
                k = _reshape_heads(k, kv_h)
                k_diff = _reshape_heads(k_diff, kv_h)
                v = _reshape_heads(v, kv_h)
            else:
                q = _reshape_heads(self.to_q(x), h)
                k, v = mx.split(self.to_kv(kv_input), 2, axis=-1)
                k = _reshape_heads(k, kv_h)
                v = _reshape_heads(v, kv_h)
        else:
            if self.differential:
                q, k, v, q_diff, k_diff = mx.split(self.to_qkv(x), 5, axis=-1)
                q = _reshape_heads(q, h)
                k = _reshape_heads(k, h)
                v = _reshape_heads(v, h)
                q_diff = _reshape_heads(q_diff, h)
                k_diff = _reshape_heads(k_diff, h)
            else:
                q, k, v = mx.split(self.to_qkv(x), 3, axis=-1)
                q = _reshape_heads(q, h)
                k = _reshape_heads(k, h)
                v = _reshape_heads(v, h)

        if self.qk_norm == "l2":
            q = q / mx.sqrt(mx.sum(q * q, axis=-1, keepdims=True) + self.qk_norm_eps)
            k = k / mx.sqrt(mx.sum(k * k, axis=-1, keepdims=True) + self.qk_norm_eps)
            if self.differential:
                q_diff = q_diff / mx.sqrt(
                    mx.sum(q_diff * q_diff, axis=-1, keepdims=True) + self.qk_norm_eps
                )
                k_diff = k_diff / mx.sqrt(
                    mx.sum(k_diff * k_diff, axis=-1, keepdims=True) + self.qk_norm_eps
                )
        elif self.qk_norm != "none":
            q, k = self.apply_qk_norm(q, k)
            if self.differential:
                q_diff, k_diff = self.apply_qk_norm(q_diff, k_diff)

        if rotary_pos_emb is not None:
            freqs, _ = rotary_pos_emb
            q_freqs = freqs
            if rotary_pos_emb_k is not None:
                k_freqs, _ = rotary_pos_emb_k
            else:
                k_freqs = q_freqs
                if q.shape[-2] >= k.shape[-2]:
                    ratio = float(q.shape[-2]) / float(k.shape[-2])
                    q_freqs, k_freqs = freqs, ratio * freqs
                else:
                    ratio = float(k.shape[-2]) / float(q.shape[-2])
                    q_freqs, k_freqs = ratio * freqs, freqs

            q = apply_rotary_pos_emb(q.astype(mx.float32), q_freqs.astype(mx.float32)).astype(v.dtype)
            k = apply_rotary_pos_emb(k.astype(mx.float32), k_freqs.astype(mx.float32)).astype(v.dtype)
            if self.differential:
                q_diff = apply_rotary_pos_emb(
                    q_diff.astype(mx.float32),
                    q_freqs.astype(mx.float32),
                ).astype(v.dtype)
                k_diff = apply_rotary_pos_emb(
                    k_diff.astype(mx.float32),
                    k_freqs.astype(mx.float32),
                ).astype(v.dtype)

        use_causal = self.causal if causal is None else causal
        if int(q.shape[-2]) == 1 and use_causal:
            use_causal = False

        if self.differential:
            out = self.apply_attn(
                q,
                k,
                v,
                causal=use_causal,
                padding_mask=padding_mask,
                sliding_window=sliding_window,
            )
            out_diff = self.apply_attn(
                q_diff,
                k_diff,
                v,
                causal=use_causal,
                padding_mask=padding_mask,
                sliding_window=sliding_window,
            )
            out = out - out_diff
        else:
            out = self.apply_attn(
                q,
                k,
                v,
                causal=use_causal,
                padding_mask=padding_mask,
                sliding_window=sliding_window,
            )

        out = _merge_heads(out)
        out = self.to_out(out)

        if self.feat_scale:
            if padding_mask is not None:
                mask = padding_mask[:, :, None].astype(out.dtype)
                denom = mx.maximum(mx.sum(mask, axis=-2, keepdims=True), 1.0)
                out_dc = mx.sum(out * mask, axis=-2, keepdims=True) / denom
                out_hf = out - out_dc
                out = out + (self.lambda_dc * out_dc + self.lambda_hf * out_hf) * mask
            else:
                out_dc = mx.mean(out, axis=-2, keepdims=True)
                out_hf = out - out_dc
                out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_heads: int = 64,
        cross_attend: bool = False,
        dim_context: int | None = None,
        global_cond_dim: int | None = None,
        local_add_cond_dim: int | None = None,
        modular_local_cond_configs: list[dict[str, tp.Any]] | None = None,
        causal: bool = False,
        zero_init_branch_outputs: bool = True,
        conformer: bool = False,
        layer_ix: int = -1,
        add_rope: bool = False,
        layer_scale: bool = False,
        norm_type: str = "layer_norm",
        attn_kwargs: dict[str, tp.Any] | None = None,
        ff_kwargs: dict[str, tp.Any] | None = None,
        norm_kwargs: dict[str, tp.Any] | None = None,
    ):
        super().__init__()
        if conformer:
            raise NotImplementedError("Conformer blocks are not implemented in the MLX DiT.")
        if modular_local_cond_configs:
            raise NotImplementedError("Modular local conditioning is not implemented in the MLX DiT.")

        attn_kwargs = attn_kwargs or {}
        ff_kwargs = ff_kwargs or {}
        norm_kwargs = norm_kwargs or {}

        if layer_scale and zero_init_branch_outputs:
            zero_init_branch_outputs = False

        norm_layer_map = {
            "layer_norm": LayerNorm,
            "rms_norm": RMSNorm,
            "dyt": DynamicTanh,
        }
        if norm_type not in norm_layer_map:
            raise ValueError(
                'norm_type must be one of ["layer_norm", "rms_norm", "dyt"], '
                f"got {norm_type!r}"
            )
        norm_layer = norm_layer_map[norm_type]

        self.dim = int(dim)
        self.dim_heads = int(min(dim_heads, dim))
        self.cross_attend = bool(cross_attend)
        self.global_cond_dim = global_cond_dim
        self.local_add_cond_dim = local_add_cond_dim
        self.add_rope = bool(add_rope)
        self.layer_ix = int(layer_ix)

        self.pre_norm = norm_layer(self.dim, **norm_kwargs)
        self.self_attn = Attention(
            self.dim,
            dim_heads=self.dim_heads,
            causal=causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs,
        )
        self.self_attn_scale = LayerScale(self.dim) if layer_scale else Identity()

        if self.cross_attend:
            if dim_context is None:
                raise ValueError("dim_context must be set when cross_attend=True")
            self.cross_attend_norm = norm_layer(self.dim, **norm_kwargs)
            self.cross_attn = Attention(
                self.dim,
                dim_heads=self.dim_heads,
                dim_context=dim_context,
                causal=causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs,
            )
            self.cross_attn_scale = LayerScale(self.dim) if layer_scale else Identity()

        self.ff_norm = norm_layer(self.dim, **norm_kwargs)
        self.ff = FeedForward(self.dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)
        self.ff_scale = LayerScale(self.dim) if layer_scale else Identity()

        if self.global_cond_dim is not None:
            self.to_scale_shift_gate = (
                mx.random.normal((6 * self.dim,), dtype=mx.float32) / math.sqrt(self.dim)
            )

        if self.local_add_cond_dim is not None:
            linear_out = nn.Linear(self.dim, self.dim, bias=True)
            linear_out.weight = mx.zeros_like(linear_out.weight)
            linear_out.bias = mx.zeros_like(linear_out.bias)
            self.to_local_embed = [
                nn.Linear(self.local_add_cond_dim, self.dim, bias=True),
                nn.silu,
                linear_out,
            ]
        else:
            self.to_local_embed = None

        self.rope = RotaryEmbedding(self.dim_heads // 2) if self.add_rope else None

    def _apply_local_conditioning(self, x, local_add_cond=None):
        if local_add_cond is not None and self.to_local_embed is not None:
            local_emb = run_layers(self.to_local_embed, local_add_cond)
            x = x + _left_pad_to_match(local_emb, int(x.shape[-2]))
        return x

    def __call__(
        self,
        x,
        *,
        context=None,
        global_cond=None,
        local_add_cond=None,
        rotary_pos_emb=None,
        cross_attn_rotary_pos_emb=None,
        padding_mask=None,
        self_attention_sliding_window=None,
    ):
        if rotary_pos_emb is None and self.add_rope and self.rope is not None:
            rotary_pos_emb = self.rope.forward_from_seq_len(int(x.shape[-2]))

        if self.global_cond_dim is not None and global_cond is not None:
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = mx.split(
                (self.to_scale_shift_gate[None, :] + global_cond)[:, None, :],
                6,
                axis=-1,
            )

            residual = x
            x = self.pre_norm(x)
            x = x * (1.0 + scale_self) + shift_self
            x = self.self_attn(
                x,
                rotary_pos_emb=rotary_pos_emb,
                padding_mask=padding_mask,
                sliding_window=self_attention_sliding_window,
            )
            x = x * mx.sigmoid(1.0 - gate_self)
            x = self.self_attn_scale(x)
            x = x + residual

            if context is not None and self.cross_attend:
                cross_kwargs = {"context": context}
                if cross_attn_rotary_pos_emb is not None:
                    cross_kwargs["rotary_pos_emb"] = rotary_pos_emb
                    cross_kwargs["rotary_pos_emb_k"] = cross_attn_rotary_pos_emb
                x = x + self.cross_attn_scale(
                    self.cross_attn(self.cross_attend_norm(x), **cross_kwargs)
                )

            x = self._apply_local_conditioning(x, local_add_cond=local_add_cond)

            residual = x
            x = self.ff_norm(x)
            x = x * (1.0 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * mx.sigmoid(1.0 - gate_ff)
            x = self.ff_scale(x)
            x = x + residual
            return x

        x = x + self.self_attn_scale(
                self.self_attn(
                    self.pre_norm(x),
                    rotary_pos_emb=rotary_pos_emb,
                    padding_mask=padding_mask,
                    sliding_window=self_attention_sliding_window,
                )
            )

        if context is not None and self.cross_attend:
            cross_kwargs = {"context": context}
            if cross_attn_rotary_pos_emb is not None:
                cross_kwargs["rotary_pos_emb"] = rotary_pos_emb
                cross_kwargs["rotary_pos_emb_k"] = cross_attn_rotary_pos_emb
            x = x + self.cross_attn_scale(
                self.cross_attn(self.cross_attend_norm(x), **cross_kwargs)
            )

        x = self._apply_local_conditioning(x, local_add_cond=local_add_cond)
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
        local_add_cond_dim: int | None = None,
        modular_local_cond_configs: list[dict[str, tp.Any]] | None = None,
        causal: bool = False,
        rotary_pos_emb: bool = True,
        cross_attn_rotary_pos_emb: bool = False,
        zero_init_branch_outputs: bool = True,
        num_memory_tokens: int = 0,
        sliding_window: tp.Any = None,
        attn_kwargs: dict[str, tp.Any] | None = None,
        ff_kwargs: dict[str, tp.Any] | None = None,
        norm_kwargs: dict[str, tp.Any] | None = None,
        **kwargs: tp.Any,
    ):
        super().__init__()
        if sliding_window is not None:
            raise NotImplementedError("Sliding-window attention is not implemented in the MLX DiT.")

        self.dim = int(dim)
        self.depth = int(depth)
        self.causal = bool(causal)
        self.num_memory_tokens = int(num_memory_tokens)
        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else Identity()
        self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32)) if rotary_pos_emb else None
        self.cross_attn_rotary_pos_emb = (
            RotaryEmbedding(max(dim_heads // 2, 32)) if cross_attn_rotary_pos_emb else None
        )

        if self.num_memory_tokens > 0:
            self.memory_tokens = mx.random.normal(
                (self.num_memory_tokens, self.dim),
                dtype=mx.float32,
            )

        if global_cond_dim is not None:
            self.global_cond_embedder = [
                nn.Linear(global_cond_dim, self.dim, bias=True),
                nn.silu,
                nn.Linear(self.dim, self.dim * 6, bias=True),
            ]
        else:
            self.global_cond_embedder = None

        self.layers = []
        for layer_ix in range(self.depth):
            should_cross_attend = bool(cross_attend) and (
                final_cross_attn_ix == -1 or layer_ix <= final_cross_attn_ix
            )
            self.layers.append(
                TransformerBlock(
                    self.dim,
                    dim_heads=dim_heads,
                    cross_attend=should_cross_attend,
                    dim_context=cond_token_dim,
                    global_cond_dim=global_cond_dim,
                    local_add_cond_dim=local_add_cond_dim,
                    modular_local_cond_configs=modular_local_cond_configs,
                    causal=causal,
                    zero_init_branch_outputs=zero_init_branch_outputs,
                    layer_ix=layer_ix,
                    attn_kwargs=attn_kwargs,
                    ff_kwargs=ff_kwargs,
                    norm_kwargs=norm_kwargs,
                    **kwargs,
                )
            )

    def __call__(
        self,
        x,
        *,
        prepend_embeds=None,
        context=None,
        global_cond=None,
        local_add_cond=None,
        return_info: bool = False,
        exit_layer_ix: int | None = None,
        padding_mask=None,
    ):
        info = {"hidden_states": []}
        batch = int(x.shape[0])

        x = self.project_in(x)

        prepend_length = 0
        if prepend_embeds is not None:
            prepend_length = int(prepend_embeds.shape[1])
            x = mx.concatenate([prepend_embeds, x], axis=1)

        if self.num_memory_tokens > 0:
            memory_tokens = mx.broadcast_to(
                self.memory_tokens[None, :, :],
                (batch, self.num_memory_tokens, self.dim),
            )
            x = mx.concatenate([memory_tokens, x], axis=1)

        rotary = (
            self.rotary_pos_emb.forward_from_seq_len(int(x.shape[1]))
            if self.rotary_pos_emb is not None
            else None
        )
        cross_rotary = None
        if self.cross_attn_rotary_pos_emb is not None and context is not None:
            cross_rotary = self.cross_attn_rotary_pos_emb.forward_from_seq_len(int(context.shape[1]))

        if global_cond is not None and self.global_cond_embedder is not None:
            global_cond = run_layers(self.global_cond_embedder, global_cond)

        extended_padding_mask = padding_mask
        if padding_mask is not None:
            prepend_valid = self.num_memory_tokens + prepend_length
            if prepend_valid > 0:
                pad_prefix = mx.ones((batch, prepend_valid), dtype=mx.bool_)
                extended_padding_mask = mx.concatenate([pad_prefix, padding_mask], axis=-1)

        for layer_ix, layer in enumerate(self.layers):
            x = layer(
                x,
                context=context,
                global_cond=global_cond,
                local_add_cond=local_add_cond,
                rotary_pos_emb=rotary,
                cross_attn_rotary_pos_emb=cross_rotary,
                padding_mask=extended_padding_mask,
            )
            if return_info:
                info["hidden_states"].append(x)
            if exit_layer_ix is not None and layer_ix == exit_layer_ix:
                x = x[:, self.num_memory_tokens :, :]
                if return_info:
                    return x, info
                return x

        x = x[:, self.num_memory_tokens :, :]
        x = self.project_out(x)

        if return_info:
            return x, info
        return x
