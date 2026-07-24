"""sa3-medium DiT — MLX implementation.

Mirrors `dit_torch_medium.py` exactly using MLX primitives:
  - mx.fast.rope (traditional=False — half-half pairing, matches PyTorch port)
  - mx.fast.scaled_dot_product_attention
  - nn.RMSNorm  (weight only, no bias)
  - mx.compile  (optional, for fusion)

Differential attention pattern: to_qkv(x).chunk(5) → q, k, v, q_diff, k_diff;
two SDPAs, subtract.

Conv1d weights stored as [out, k, in] (MLX layout). Linear weights as [out, in]
(same as PyTorch). The weight loader converts from PyTorch safetensors.

Forward signature matches PyTorch (channels-first I/O):
    DiT(x, t, cross_attn_cond, global_cond) -> v
      x:     [1, 256, T_lat]
      t:     [1]
      cross: [1, 257, 768]
      gcond: [1, 768]
      v:     [1, 256, T_lat]
"""

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


# Constants matching sa3-medium DiT
IO_CHANNELS = 256
EMBED_DIM = 1536
DEPTH = 24
NUM_HEADS = 24
HEAD_DIM = 64
ROPE_DIMS = 32
COND_TOKEN_DIM = 768
GLOBAL_COND_DIM = 768
LOCAL_ADD_COND_DIM = 257
NUM_MEMORY_TOKENS = 64
FF_INNER = 6144
TIMESTEP_FEAT_DIM = 256

NORM_EPS = 1e-5
QK_NORM_EPS = 1e-6


# ---- Modules ----

class DifferentialSelfAttention(nn.Module):
    """Self-attn, fused 5x QKV, qk_rms_norm, partial RoPE, two SDPAs - subtract."""

    def __init__(self):
        super().__init__()
        self.to_qkv = nn.Linear(EMBED_DIM, 5 * EMBED_DIM, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x):
        B, T, _ = x.shape
        H, D = NUM_HEADS, HEAD_DIM

        qkv = self.to_qkv(x)                                  # [B, T, 5E]
        q, k, v, q_diff, k_diff = mx.split(qkv, 5, axis=-1)   # each [B, T, E]

        # [B, T, E] → [B, H, T, D]
        def to_heads(t):
            return t.reshape(B, T, H, D).transpose(0, 2, 1, 3)
        q      = to_heads(q)
        k      = to_heads(k)
        v      = to_heads(v)
        q_diff = to_heads(q_diff)
        k_diff = to_heads(k_diff)

        q      = self.q_norm(q)
        k      = self.k_norm(k)
        q_diff = self.q_norm(q_diff)
        k_diff = self.k_norm(k_diff)

        # RoPE on first 32 of 64 head dims (half-half pairing)
        q      = mx.fast.rope(q,      ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k      = mx.fast.rope(k,      ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        q_diff = mx.fast.rope(q_diff, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)
        k_diff = mx.fast.rope(k_diff, ROPE_DIMS, traditional=False, base=10000.0, scale=1.0, offset=0)

        out_main = mx.fast.scaled_dot_product_attention(q,      k,      v, scale=self.scale)
        out_diff = mx.fast.scaled_dot_product_attention(q_diff, k_diff, v, scale=self.scale)
        out = out_main - out_diff

        out = out.transpose(0, 2, 1, 3).reshape(B, T, EMBED_DIM)
        return self.to_out(out)


class DifferentialCrossAttention(nn.Module):
    """Cross-attn, separate to_q (2x), to_kv (3x); no RoPE."""

    def __init__(self):
        super().__init__()
        self.to_q  = nn.Linear(EMBED_DIM, 2 * EMBED_DIM, bias=False)
        self.to_kv = nn.Linear(EMBED_DIM, 3 * EMBED_DIM, bias=False)
        self.to_out = nn.Linear(EMBED_DIM, EMBED_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=QK_NORM_EPS)
        self.scale = HEAD_DIM ** -0.5

    def __call__(self, x, context):
        B, Tx, _ = x.shape
        Tc = context.shape[1]
        H, D = NUM_HEADS, HEAD_DIM

        q, q_diff = mx.split(self.to_q(x), 2, axis=-1)
        k, k_diff, v = mx.split(self.to_kv(context), 3, axis=-1)

        def to_q_heads(t):
            return t.reshape(B, Tx, H, D).transpose(0, 2, 1, 3)

        def to_c_heads(t):
            return t.reshape(B, Tc, H, D).transpose(0, 2, 1, 3)

        q      = to_q_heads(q)
        q_diff = to_q_heads(q_diff)
        k      = to_c_heads(k)
        k_diff = to_c_heads(k_diff)
        v      = to_c_heads(v)

        q      = self.q_norm(q)
        k      = self.k_norm(k)
        q_diff = self.q_norm(q_diff)
        k_diff = self.k_norm(k_diff)

        out_main = mx.fast.scaled_dot_product_attention(q,      k,      v, scale=self.scale)
        out_diff = mx.fast.scaled_dot_product_attention(q_diff, k_diff, v, scale=self.scale)
        out = out_main - out_diff

        out = out.transpose(0, 2, 1, 3).reshape(B, Tx, EMBED_DIM)
        return self.to_out(out)


class _GLUWrap(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(EMBED_DIM, 2 * FF_INNER, bias=True)

    def __call__(self, x):
        x = self.proj(x)
        x, gate = mx.split(x, 2, axis=-1)
        return x * nn.silu(gate)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # ff is a list (state dict keys ff.0.proj.weight, ff.2.weight/bias)
        self.ff = [_GLUWrap(), None, nn.Linear(FF_INNER, EMBED_DIM, bias=True), None]

    def __call__(self, x):
        x = self.ff[0](x)
        x = self.ff[2](x)
        return x


class LocalEmbedSeq(nn.Module):
    """to_local_embed.seq.{0,2}.{weight,bias} matches PyTorch port."""

    def __init__(self):
        super().__init__()
        self.seq = [
            nn.Linear(LOCAL_ADD_COND_DIM, EMBED_DIM, bias=True),
            None,  # SiLU
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=True),
        ]

    def __call__(self, x):
        x = self.seq[0](x)
        x = nn.silu(x)
        x = self.seq[2](x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.self_attn = DifferentialSelfAttention()
        self.cross_attend_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.cross_attn = DifferentialCrossAttention()
        self.ff_norm = nn.RMSNorm(EMBED_DIM, eps=NORM_EPS)
        self.ff = FeedForward()
        self.to_scale_shift_gate = mx.zeros((6 * EMBED_DIM,))
        self.to_local_embed = LocalEmbedSeq()

    def __call__(self, x, context, global_cond, local_emb_padded):
        ss = (self.to_scale_shift_gate + global_cond)[:, None, :]   # [B, 1, 6E]
        scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = mx.split(ss, 6, axis=-1)

        residual = x
        h = self.pre_norm(x)
        h = h * (1 + scale_self) + shift_self
        h = self.self_attn(h)
        h = h * mx.sigmoid(1 - gate_self)
        x = h + residual

        x = x + self.cross_attn(self.cross_attend_norm(x), context)

        x = x + local_emb_padded

        residual = x
        h = self.ff_norm(x)
        h = h * (1 + scale_ff) + shift_ff
        h = self.ff(h)
        h = h * mx.sigmoid(1 - gate_ff)
        x = h + residual
        return x


class ContinuousTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_in = nn.Linear(IO_CHANNELS, EMBED_DIM, bias=False)
        self.project_out = nn.Linear(EMBED_DIM, IO_CHANNELS, bias=False)
        self.memory_tokens = mx.zeros((NUM_MEMORY_TOKENS, EMBED_DIM))
        self.global_cond_embedder = [
            nn.Linear(EMBED_DIM, EMBED_DIM),
            None,  # SiLU
            nn.Linear(EMBED_DIM, 6 * EMBED_DIM),
        ]
        self.layers = [TransformerBlock() for _ in range(DEPTH)]

    def __call__(self, x, context, global_embed, local_add_cond_zeros):
        B, T, _ = x.shape
        x = self.project_in(x)

        mem = mx.broadcast_to(self.memory_tokens[None], (B, NUM_MEMORY_TOKENS, EMBED_DIM))
        x = mx.concatenate([mem, x], axis=1)

        # global_cond_embedder: Linear → SiLU → Linear
        g = self.global_cond_embedder[0](global_embed)
        g = nn.silu(g)
        g = self.global_cond_embedder[2](g)                         # [B, 6E]
        global_cond = g

        # Pre-compute per-block local embeddings (each block has its own MLP)
        # Pad with zeros for memory tokens on the left along seq axis
        for layer in self.layers:
            local_emb = layer.to_local_embed(local_add_cond_zeros)  # [B, T, E]
            # Pad left with zeros for memory tokens
            pad = mx.zeros((B, NUM_MEMORY_TOKENS, EMBED_DIM), dtype=local_emb.dtype)
            local_emb_padded = mx.concatenate([pad, local_emb], axis=1)
            x = layer(x, context, global_cond, local_emb_padded)

        x = x[:, NUM_MEMORY_TOKENS:, :]
        x = self.project_out(x)
        return x


class ExpoFourierFeatures(nn.Module):
    def __init__(self, dim=TIMESTEP_FEAT_DIM, min_freq=0.5, max_freq=10000.0):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        ramp = mx.linspace(0.0, 1.0, half)
        log_min, log_max = math.log(min_freq), math.log(max_freq)
        freqs = mx.exp(ramp * (log_max - log_min) + log_min)
        self.freqs = freqs * 2 * math.pi  # buffer

    def __call__(self, t):
        args = t[:, None] * self.freqs
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class DiT(nn.Module):
    """Top-level sa3-medium DiT (MLX)."""

    def __init__(self, T_lat=320):
        super().__init__()
        self.T_lat = T_lat

        # 1x1 convs in MLX: nn.Conv1d expects (in, out, k); inputs [B, T, C].
        # We use them on channels-first [B, C, T], so wrap with transposes inside.
        self.preprocess_conv  = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, kernel_size=1, bias=False)
        self.postprocess_conv = nn.Conv1d(IO_CHANNELS, IO_CHANNELS, kernel_size=1, bias=False)

        self.to_cond_embed = [
            nn.Linear(COND_TOKEN_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_global_embed = [
            nn.Linear(GLOBAL_COND_DIM, EMBED_DIM, bias=False),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=False),
        ]
        self.to_timestep_embed = [
            nn.Linear(TIMESTEP_FEAT_DIM, EMBED_DIM, bias=True),
            None,
            nn.Linear(EMBED_DIM, EMBED_DIM, bias=True),
        ]
        self.timestep_features = ExpoFourierFeatures(TIMESTEP_FEAT_DIM, 0.5, 10000.0)
        self.transformer = ContinuousTransformer()

        # Buffer: local_add_cond_zeros (constant, doesn't depend on B)
        self._local_zeros_1 = mx.zeros((1, T_lat, LOCAL_ADD_COND_DIM))

    def __call__(self, x, t, cross_attn_cond_raw, global_cond_raw, local_add_cond=None):
        """x: [B, 256, T_lat], t: [B], cross: [B, 257, 768], gcond: [B, 768].

        local_add_cond: optional [B, T_lat, 257] — inpaint_mask + inpaint_masked_input
            (in batch-last-channel layout). Pass None for vanilla text-to-audio.
        """
        B = x.shape[0]

        # cond projections
        c = self.to_cond_embed[0](cross_attn_cond_raw)
        c = nn.silu(c)
        c = self.to_cond_embed[2](c)
        context = c

        g = self.to_global_embed[0](global_cond_raw)
        g = nn.silu(g)
        g = self.to_global_embed[2](g)
        global_pre = g

        # Timestep features + MLP
        tf = self.timestep_features(t)
        tf = self.to_timestep_embed[0](tf)
        tf = nn.silu(tf)
        tf = self.to_timestep_embed[2](tf)
        t_embed = tf

        global_embed = global_pre + t_embed

        # preprocess_conv + residual on channels-first input
        # nn.Conv1d in MLX takes [B, T, C] so we transpose
        x_lc = x.transpose(0, 2, 1)           # [B, T, 256]
        x_pp = self.preprocess_conv(x_lc) + x_lc

        if local_add_cond is None:
            # Zero local-add-cond at the INPUT length (not the baked self.T_lat),
            # so one loaded model runs any sequence length — text-to-audio
            # inference always loads T_lat == the gen length (identical result),
            # but training demos reuse the crop-length model for longer clips.
            local = mx.zeros((B, x.shape[-1], LOCAL_ADD_COND_DIM))
        else:
            local = local_add_cond

        h = self.transformer(x_pp, context, global_embed, local)  # [B, T, 256]

        out = self.postprocess_conv(h) + h
        return out.transpose(0, 2, 1)         # back to [B, 256, T_lat]




def load_official_medium_dit(weights_path, *, T_lat=320, dtype=mx.float16):
    """Load the official specialized medium DiT from a hosted MLX NPZ."""

    weights = {
        key: value
        for key, value in dict(mx.load(str(weights_path))).items()
        if not key.startswith("cond.")
        and key != "transformer.rotary_pos_emb.inv_freq"
    }
    model = DiT(T_lat=T_lat)
    model.load_weights(
        [(key, value.astype(dtype)) for key, value in weights.items()],
        # Mirrors upstream: the model generates timestep_features.freqs and
        # does not materialize rotary_pos_emb.inv_freq as a parameter.
        strict=False,
    )
    del weights
    mx.eval(model.parameters())
    return model
