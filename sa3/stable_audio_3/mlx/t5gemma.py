from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

import numpy as np

from stable_audio_3.mlx.runtime import (
    MLXRuntimeUnavailableError,
    import_mlx_core,
    import_mlx_nn,
)

try:
    from mlx.utils import tree_flatten, tree_unflatten
except ImportError as exc:
    raise MLXRuntimeUnavailableError(
        "MLX is not installed in this environment. "
        "Install the Apple Silicon MLX runtime before attempting MLX inference."
    ) from exc

mx = import_mlx_core(required=True)
nn = import_mlx_nn(required=True)


@dataclass(frozen=True)
class T5GemmaConversionReport:
    missing_keys: list[str]
    unexpected_keys: list[str]
    transposed_keys: list[str]


def gelu_pytorch_tanh(x):
    coeff = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + mx.tanh(coeff * (x + 0.044715 * x * x * x)))


def rotate_half(x):
    half = int(x.shape[-1]) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class T5GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.weight = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x):
        x_fp32 = x.astype(mx.float32)
        output = x_fp32 * mx.rsqrt(mx.mean(x_fp32 * x_fp32, axis=-1, keepdims=True) + self.eps)
        output = output * (1.0 + self.weight.astype(mx.float32))
        return output.astype(x.dtype)


class T5GemmaRotaryEmbedding(nn.Module):
    def __init__(self, *, head_dim: int, max_position_embeddings: int, rope_theta: float = 10000.0):
        super().__init__()
        self.max_position_embeddings = int(max_position_embeddings)
        freqs = mx.arange(0, int(head_dim), 2, dtype=mx.float32) / float(head_dim)
        self.inv_freq = 1.0 / (float(rope_theta) ** freqs)

    def __call__(self, x, position_ids):
        freqs = position_ids.astype(mx.float32)[:, :, None] * self.inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return cos.astype(x.dtype), sin.astype(x.dtype)


class T5GemmaSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        query_pre_attn_scalar: int,
        attn_logit_softcapping: float,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = float(query_pre_attn_scalar) ** -0.5
        self.attn_logit_softcapping = float(attn_logit_softcapping)

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)

    def _reshape_heads(self, x, heads: int):
        b, n, _ = x.shape
        x = x.reshape(b, n, heads, self.head_dim)
        return mx.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x):
        b, h, n, d = x.shape
        x = mx.transpose(x, (0, 2, 1, 3))
        return x.reshape(b, n, h * d)

    def __call__(self, hidden_states, position_embeddings, attention_mask=None):
        input_shape = hidden_states.shape[:-1]
        q = self._reshape_heads(self.q_proj(hidden_states), self.num_attention_heads)
        k = self._reshape_heads(self.k_proj(hidden_states), self.num_key_value_heads)
        v = self._reshape_heads(self.v_proj(hidden_states), self.num_key_value_heads)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.num_key_value_groups != 1:
            k = mx.repeat(k, self.num_key_value_groups, axis=1)
            v = mx.repeat(v, self.num_key_value_groups, axis=1)

        attn_weights = mx.matmul(q.astype(mx.float32), mx.swapaxes(k.astype(mx.float32), -1, -2))
        attn_weights = attn_weights * self.scaling
        if self.attn_logit_softcapping:
            attn_weights = mx.tanh(attn_weights / self.attn_logit_softcapping) * self.attn_logit_softcapping
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask.astype(mx.float32)

        attn_weights = mx.softmax(attn_weights, axis=-1).astype(q.dtype)
        attn_output = mx.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output).reshape(*input_shape, -1)
        return self.o_proj(attn_output)


class T5GemmaMLP(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        hidden_states = gelu_pytorch_tanh(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(hidden_states)


class T5GemmaEncoderLayer(nn.Module):
    def __init__(self, config: dict[str, tp.Any]):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        self.self_attn = T5GemmaSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=int(config["num_attention_heads"]),
            num_key_value_heads=int(config["num_key_value_heads"]),
            head_dim=int(config["head_dim"]),
            query_pre_attn_scalar=int(config["query_pre_attn_scalar"]),
            attn_logit_softcapping=float(config["attn_logit_softcapping"]),
        )
        self.pre_self_attn_layernorm = T5GemmaRMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
        self.post_self_attn_layernorm = T5GemmaRMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
        self.mlp = T5GemmaMLP(
            hidden_size=hidden_size,
            intermediate_size=int(config["intermediate_size"]),
        )
        self.pre_feedforward_layernorm = T5GemmaRMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))
        self.post_feedforward_layernorm = T5GemmaRMSNorm(hidden_size, eps=float(config["rms_norm_eps"]))

    def __call__(self, hidden_states, position_embeddings, attention_mask=None):
        residual = hidden_states
        hidden_states = self.pre_self_attn_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return residual + hidden_states


class T5GemmaEncoder(nn.Module):
    def __init__(
        self,
        config: dict[str, tp.Any],
        *,
        param_dtype=mx.float32,
    ):
        super().__init__()
        self.config = dict(config)
        self.param_dtype = param_dtype
        self.hidden_size = int(config["hidden_size"])
        self.vocab_size = int(config["vocab_size"])
        self.pad_token_id = int(config.get("pad_token_id", 0))

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.norm = T5GemmaRMSNorm(self.hidden_size, eps=float(config["rms_norm_eps"]))
        self.layers = [T5GemmaEncoderLayer(config) for _ in range(int(config["num_hidden_layers"]))]
        self.rotary_emb = T5GemmaRotaryEmbedding(
            head_dim=int(config["head_dim"]),
            max_position_embeddings=int(config["max_position_embeddings"]),
            rope_theta=float(config["rope_parameters"]["rope_theta"]),
        )

    def _attention_mask(self, attention_mask, *, dtype):
        if attention_mask is None:
            return None
        valid = attention_mask.astype(mx.bool_)
        zeros = mx.zeros(attention_mask.shape, dtype=dtype)
        additive = mx.where(valid[:, None, None, :], zeros[:, None, None, :], -mx.inf)
        return additive

    def __call__(self, input_ids, *, attention_mask=None, position_ids=None):
        hidden_states = self.embed_tokens(input_ids)
        if position_ids is None:
            positions = mx.arange(int(hidden_states.shape[1]), dtype=mx.int32)
            position_ids = mx.broadcast_to(positions[None, :], input_ids.shape)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).astype(mx.int32)

        hidden_states = hidden_states * math.sqrt(self.hidden_size)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        additive_mask = self._attention_mask(attention_mask, dtype=mx.float32)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                attention_mask=additive_mask,
            )

        return self.norm(hidden_states)

    def load_torch_state_dict(self, torch_state_dict: dict[str, tp.Any]) -> T5GemmaConversionReport:
        params = dict(tree_flatten(self.parameters()))
        missing: list[str] = []
        used_keys: set[str] = set()
        transposed: list[str] = []
        updates: list[tuple[str, tp.Any]] = []

        for key, target in params.items():
            if key == "rotary_emb.inv_freq":
                continue
            source_key = _resolve_torch_state_key(key, torch_state_dict)
            if source_key is None:
                missing.append(key)
                continue
            used_keys.add(source_key)
            src = torch_state_dict[source_key].detach().cpu().float().numpy()
            src, did_transpose = _convert_weight_to_mlx_shape(src, tuple(target.shape))
            if did_transpose:
                transposed.append(key)
            arr = mx.array(src.astype(np.float32, copy=False))
            if arr.dtype != self.param_dtype:
                arr = arr.astype(self.param_dtype)
            updates.append((key, arr))

        if missing:
            raise RuntimeError(f"Missing {len(missing)} keys for MLX T5Gemma load, e.g. {missing[:5]}")

        self.update(tree_unflatten(updates))
        unexpected = sorted(k for k in torch_state_dict if k not in used_keys)
        return T5GemmaConversionReport(
            missing_keys=missing,
            unexpected_keys=unexpected,
            transposed_keys=transposed,
        )

    @classmethod
    def from_torch_encoder(
        cls,
        torch_encoder_model,
        *,
        mlx_dtype=mx.float32,
    ) -> tuple["T5GemmaEncoder", T5GemmaConversionReport]:
        config = torch_encoder_model.config.encoder.to_dict()
        mlx_model = cls(config, param_dtype=mlx_dtype)
        report = mlx_model.load_torch_state_dict(torch_encoder_model.state_dict())
        mx.eval(mlx_model.parameters())
        return mlx_model, report


def _resolve_torch_state_key(key: str, torch_state_dict: dict[str, tp.Any]) -> str | None:
    for candidate in (key, f"encoder.{key}", f"model.encoder.{key}"):
        if candidate in torch_state_dict:
            return candidate
    return None


def _convert_weight_to_mlx_shape(arr: np.ndarray, target_shape: tuple[int, ...]) -> tuple[np.ndarray, bool]:
    if arr.shape == target_shape:
        return arr, False

    if arr.ndim == 2:
        candidate = arr.T
        if candidate.shape == target_shape:
            return candidate, True

    raise ValueError(f"Unable to map tensor with shape {arr.shape} to target {target_shape}")


class T5GemmaTextConditioner:
    MODEL_ALIASES = {
        "stabilityai/t5gemma-b-b-ul2": "google/t5gemma-b-b-ul2",
    }

    def __init__(
        self,
        *,
        encoder: T5GemmaEncoder,
        tokenizer,
        max_length: int,
        padding_mode: str = "zero",
        padding_embedding=None,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.padding_mode = str(padding_mode)
        self.padding_embedding = padding_embedding

    @classmethod
    def from_torch_conditioner(
        cls,
        torch_conditioner,
        *,
        mlx_dtype=mx.float32,
    ) -> tuple["T5GemmaTextConditioner", T5GemmaConversionReport]:
        if not hasattr(torch_conditioner, "model"):
            raise ValueError("torch_conditioner must expose a `.model` T5Gemma encoder.")
        if torch_conditioner.proj_out.__class__.__name__ != "Identity":
            raise NotImplementedError("MLX T5Gemma conditioner currently expects an identity projection.")

        encoder, report = T5GemmaEncoder.from_torch_encoder(
            torch_conditioner.model,
            mlx_dtype=mlx_dtype,
        )
        padding_embedding = None
        if hasattr(torch_conditioner, "padding_embedding"):
            padding_embedding = mx.array(
                torch_conditioner.padding_embedding.detach().cpu().numpy().astype(np.float32, copy=False)
            )
            if padding_embedding.dtype != mlx_dtype:
                padding_embedding = padding_embedding.astype(mlx_dtype)

        return (
            cls(
                encoder=encoder,
                tokenizer=torch_conditioner.tokenizer,
                max_length=int(torch_conditioner.max_length),
                padding_mode=str(torch_conditioner.padding_mode),
                padding_embedding=padding_embedding,
            ),
            report,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "google/t5gemma-b-b-ul2",
        *,
        max_length: int = 256,
        padding_mode: str = "zero",
        padding_embedding=None,
        mlx_dtype=mx.float32,
    ) -> tuple["T5GemmaTextConditioner", T5GemmaConversionReport]:
        import torch
        from transformers import AutoConfig, AutoTokenizer, T5GemmaEncoderModel

        resolved_name = cls.MODEL_ALIASES.get(model_name, model_name)
        tokenizer = AutoTokenizer.from_pretrained(resolved_name)
        config = AutoConfig.from_pretrained(resolved_name)
        config.is_encoder_decoder = False
        torch_model = T5GemmaEncoderModel.from_pretrained(
            resolved_name,
            config=config,
            attn_implementation="eager",
        ).to(torch.float32).eval()
        encoder, report = T5GemmaEncoder.from_torch_encoder(torch_model, mlx_dtype=mlx_dtype)
        if padding_embedding is not None and not hasattr(padding_embedding, "shape"):
            padding_embedding = mx.array(padding_embedding)
        if padding_embedding is not None and padding_embedding.dtype != mlx_dtype:
            padding_embedding = padding_embedding.astype(mlx_dtype)

        return (
            cls(
                encoder=encoder,
                tokenizer=tokenizer,
                max_length=max_length,
                padding_mode=padding_mode,
                padding_embedding=padding_embedding,
            ),
            report,
        )

    def _apply_padding(self, embeddings, attention_mask):
        mode = self.padding_mode
        if mode == "none":
            return embeddings
        mask = attention_mask.astype(mx.bool_)[:, :, None]
        if mode == "zero":
            return embeddings * mask.astype(embeddings.dtype)
        if mode == "learned":
            if self.padding_embedding is None:
                raise ValueError("padding_mode='learned' requires a padding_embedding.")
            learned = mx.broadcast_to(
                self.padding_embedding[None, None, :].astype(embeddings.dtype),
                embeddings.shape,
            )
            return mx.where(mask, embeddings, learned)
        raise ValueError(f"Unknown padding mode: {mode!r}")

    def __call__(self, prompts: list[str]):
        encoded = self.tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"]).astype(mx.int32)
        attention_mask = mx.array(encoded["attention_mask"]).astype(mx.int32)
        embeddings = self.encoder(input_ids, attention_mask=attention_mask)
        embeddings = self._apply_padding(embeddings, attention_mask)
        return embeddings, attention_mask.astype(mx.bool_)
