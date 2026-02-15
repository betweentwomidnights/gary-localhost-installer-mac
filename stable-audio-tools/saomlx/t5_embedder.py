"""T5 embedding wrapper backed by third_party/mlx-examples/t5."""

from __future__ import annotations

import json
import sys
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .hf_download import download_file


@dataclass(frozen=True)
class SAOT5Spec:
    conditioner_id: str
    model_name: str
    max_length: int
    cond_dim: int


def _load_t5_class():
    repo_root = Path(__file__).resolve().parents[2]
    t5_dir = repo_root / "third_party" / "mlx-examples" / "t5"
    if not t5_dir.exists():
        raise FileNotFoundError(
            f"Could not find MLX T5 implementation at {t5_dir}. "
            "Run ./scripts/pin_deps.sh first."
        )

    t5_path = str(t5_dir)
    if t5_path not in sys.path:
        sys.path.insert(0, t5_path)

    from t5 import T5  # type: ignore

    return T5


class T5Embedder:
    """Minimal prompt -> embeddings wrapper with explicit mask output."""

    def __init__(
        self,
        model_name: str = "t5-base",
        dtype: str = "float32",
        max_length: int = 64,
    ):
        import mlx.core as mx

        dtype_map = {
            "float16": mx.float16,
            "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype '{dtype}'. Expected one of {sorted(dtype_map)}")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")

        self._mx = mx
        self.model_name = model_name
        self.dtype = dtype
        self.max_length = max_length

        t5_cls = _load_t5_class()
        self.model, self.tokenizer = t5_cls.from_pretrained(model_name, dtype=dtype_map[dtype])

    @classmethod
    def from_sao_model_config(
        cls,
        model_config: dict[str, tp.Any],
        *,
        conditioner_id: str = "prompt",
        dtype: str = "float32",
    ) -> "T5Embedder":
        spec = extract_sao_t5_spec(model_config, conditioner_id=conditioner_id)
        return cls(model_name=spec.model_name, dtype=dtype, max_length=spec.max_length)

    @classmethod
    def from_sao_hf_repo(
        cls,
        repo_id: str,
        *,
        conditioner_id: str = "prompt",
        dtype: str = "float32",
    ) -> "T5Embedder":
        config_path = download_file(repo_id=repo_id, filename="model_config.json", repo_type="model")
        with config_path.open("r", encoding="utf-8") as handle:
            model_config = json.load(handle)
        return cls.from_sao_model_config(model_config, conditioner_id=conditioner_id, dtype=dtype)

    def encode(self, prompts: Iterable[str]):
        prompts = list(prompts)
        if not prompts:
            raise ValueError("prompts must contain at least one string")

        token_rows: list[np.ndarray] = []
        for prompt in prompts:
            token_ids = np.asarray(self.tokenizer.encode(prompt), dtype=np.int32)
            if token_ids.ndim == 2:
                token_ids = token_ids[0]
            token_rows.append(token_ids)

        # The mlx-examples T5 encoder does not expose an attention-mask argument.
        # Encode each sequence at its true length to avoid pad-token attention pollution,
        # then pad encoded outputs back to [B, max_length, D].
        encoded_rows: list[np.ndarray] = []
        hidden_dim = None
        for row in token_rows:
            n_tokens = min(len(row), self.max_length)
            ids_mx = self._mx.array(row[:n_tokens][None, :])
            encoded = self.model.encode(ids_mx)[0]
            encoded_np = np.asarray(encoded.astype(self._mx.float32))
            encoded_rows.append(encoded_np)
            if hidden_dim is None:
                hidden_dim = int(encoded_np.shape[-1])

        if hidden_dim is None:
            raise RuntimeError("Failed to encode any prompts.")

        attention_mask = np.zeros((len(token_rows), self.max_length), dtype=np.bool_)
        embeddings = np.zeros((len(token_rows), self.max_length, hidden_dim), dtype=np.float32)

        for idx, encoded in enumerate(encoded_rows):
            n_tokens = min(encoded.shape[0], self.max_length)
            attention_mask[idx, :n_tokens] = True
            embeddings[idx, :n_tokens, :] = encoded[:n_tokens]

        mask_mx = self._mx.array(attention_mask)
        embeddings_mx = self._mx.array(embeddings).astype(self.model.wte.weight.dtype)
        embeddings_mx = embeddings_mx * mask_mx[..., None].astype(embeddings_mx.dtype)
        return embeddings_mx, mask_mx

    def encode_prompt(self, prompt: str):
        embeddings, mask = self.encode([prompt])
        return embeddings, mask


def extract_sao_t5_spec(
    model_config: dict[str, tp.Any],
    *,
    conditioner_id: str = "prompt",
) -> SAOT5Spec:
    conditioning = model_config.get("model", {}).get("conditioning", {})
    configs = conditioning.get("configs", [])
    cond_dim = int(conditioning["cond_dim"])

    for cond in configs:
        if cond.get("id") != conditioner_id:
            continue
        if cond.get("type") != "t5":
            raise ValueError(
                f"Conditioner '{conditioner_id}' is type '{cond.get('type')}', expected 't5'."
            )
        config = cond.get("config", {})
        model_name = config.get("t5_model_name", "t5-base")
        max_length = int(config.get("max_length", 64))
        return SAOT5Spec(
            conditioner_id=conditioner_id,
            model_name=model_name,
            max_length=max_length,
            cond_dim=cond_dim,
        )

    raise ValueError(f"No conditioner with id '{conditioner_id}' found in model config.")
