# Copyright © 2024 Apple Inc.

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback if tqdm is not installed.
    def tqdm(x):  # type: ignore
        return x

from .encodec import EncodecModel
from .t5 import T5


def _load_torch_best_state(state_dict_path: Path):
    import torch

    def _unwrap_best_state(payload):
        if isinstance(payload, dict) and "best_state" in payload:
            return payload["best_state"]
        raise RuntimeError(
            f"Unexpected checkpoint payload for {state_dict_path}: missing 'best_state'"
        )

    try:
        return _unwrap_best_state(torch.load(state_dict_path, weights_only=True))
    except TypeError:
        # Older torch versions may not support the `weights_only` keyword.
        return _unwrap_best_state(torch.load(state_dict_path))
    except Exception as first_error:
        text = str(first_error or "").lower()
        legacy_markers = (
            "cannot use ``weights_only=true``",
            "legacy .tar format",
            "weights_only",
            "filename 'storages' not found",
        )
        if not any(marker in text for marker in legacy_markers):
            raise
        return _unwrap_best_state(torch.load(state_dict_path, weights_only=False))


class TextConditioner(nn.Module):
    def __init__(
        self,
        t5_name,
        input_dim,
        output_dim,
        download_progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__()
        self._t5, self.tokenizer = T5.from_pretrained(
            t5_name, download_progress_callback=download_progress_callback
        )
        self.output_proj = nn.Linear(input_dim, output_dim)

    def __call__(self, text):
        x = self.tokenizer.encode(text)
        x = self._t5.encode(x)
        return self.output_proj(x)


class KVCache:
    def __init__(self, head_dim, n_kv_heads):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys = None
        self.values = None
        self.offset = 0
        self.step = 256

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    @property
    def state(self):
        return self.keys, self.values


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()

        self.n_heads = n_heads

        head_dim = dim // n_heads

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L_q, D = queries.shape
        L_k = keys.shape[1]

        queries, keys, values = (
            self.q_proj(queries),
            self.k_proj(keys),
            self.v_proj(values),
        )

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L_q, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.decoder.num_attention_heads
        self.hidden_size = config.decoder.hidden_size
        self.self_attn = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.cross_attn = MultiHeadAttention(self.hidden_size, self.num_attention_heads)
        self.linear1 = nn.Linear(self.hidden_size, config.decoder.ffn_dim, bias=False)
        self.linear2 = nn.Linear(config.decoder.ffn_dim, self.hidden_size, bias=False)

        self.norm1 = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.norm_cross = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.hidden_size, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        xn = self.norm1(x)
        x += self.self_attn(xn, xn, xn, mask, cache)
        xn = self.norm_cross(x)
        x += self.cross_attn(xn, conditioning, conditioning, mask)
        xn = self.norm2(x)
        x += self.linear2(nn.gelu(self.linear1(xn)))
        return x


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logits: mx.array, top_k: float, temperature: float, axis: int = -1
) -> mx.array:
    """
    Apply top-k sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_k: Sample from the top k logits.
        temperature: Temperature parameter for softmax distribution reshaping.
        axis: Axis along which to sample.
    Returns:
        token selected based on the top-k criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits * (1 / temperature), axis=axis)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)
    prob_threshold = mx.take(sorted_probs, mx.array(-top_k), axis=axis)

    # select the top K tokens in probability
    top_probs = mx.where(
        sorted_probs > prob_threshold,
        sorted_probs,
        0,
    )

    sorted_token = mx.random.categorical(mx.log(top_probs), axis=axis)
    token = mx.take_along_axis(
        sorted_indices, mx.expand_dims(sorted_token, axis), axis=axis
    )

    return token


def create_sin_embedding(positions: mx.array, dim: int, max_period: float = 10000):
    assert dim % 2 == 0
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


class MusicGen(nn.Module):
    def __init__(
        self,
        config,
        download_progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.num_codebooks = config.decoder.num_codebooks
        self.codebook_size = config.audio_encoder.codebook_size
        self.bos_token_id = config.decoder.bos_token_id
        self.hidden_size = config.decoder.hidden_size
        self.num_attention_heads = config.decoder.num_attention_heads
        self.sampling_rate = config.audio_encoder.sampling_rate

        self.text_conditioner = TextConditioner(
            config.text_encoder._name_or_path,
            config.text_encoder.d_model,
            self.hidden_size,
            download_progress_callback=download_progress_callback,
        )
        self.emb = [
            nn.Embedding(self.codebook_size + 1, self.hidden_size)
            for _ in range(self.num_codebooks)
        ]
        self.layers = [
            TransformerBlock(config) for _ in range(config.decoder.num_hidden_layers)
        ]
        self.out_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.linears = [
            nn.Linear(self.hidden_size, self.codebook_size, bias=False)
            for _ in range(self.num_codebooks)
        ]
        encodec_name = config.audio_encoder._name_or_path.split("/")[-1]
        encodec_name = encodec_name.replace("_", "-")
        self._audio_decoder, _ = EncodecModel.from_pretrained(
            f"mlx-community/{encodec_name}-float32",
            download_progress_callback=download_progress_callback,
        )

    def __call__(
        self,
        audio_tokens: mx.array,
        conditioning: mx.array,
        cache: list[KVCache] = None,
    ):

        if cache is None:
            cache = [None] * len(self.layers)

        x = sum([self.emb[k](audio_tokens[..., k]) for k in range(self.num_codebooks)])

        offset = cache[0].offset if cache[0] is not None else 0
        pos_emb = create_sin_embedding(offset, self.hidden_size)
        x += pos_emb.astype(x.dtype)

        for layer, c in zip(self.layers, cache):
            x = layer(x, conditioning, cache=c)

        x = self.out_norm(x)
        x = mx.stack([self.linears[k](x) for k in range(self.num_codebooks)], axis=-1)
        return x

    def generate(
        self,
        text: str,
        max_steps: int = 200,
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
    ) -> mx.array:
        """
        Generates a waveform conditioned on `text`.

        Args:
            text (str): The text to condition generation on.
            max_steps (int): Max steps to generate.
            top_k (int): Top k used in sampling.
            temp (float): Sampling softmax temperature.
            guidance_coef (float): Classifier free guidance coefficent.
                Used to combine conditional and unconditional logits.

        Returns:
            An mx.array of audio samples of shape ``(num_samples,)``.
        """
        # Assuming no audio prompt we start with all bos token for the codebooks
        audio_shape = (1, max_steps + 1, self.num_codebooks)
        audio_seq = mx.full(audio_shape, self.bos_token_id)

        text_tokens = self.text_conditioner(text)
        # Compute conditional and unconditional logits in one batch
        text_tokens = mx.concatenate([text_tokens, mx.zeros_like(text_tokens)], axis=0)

        head_dim = self.hidden_size // self.num_attention_heads
        cache = [
            KVCache(head_dim, self.num_attention_heads) for _ in range(len(self.layers))
        ]
        for offset in tqdm(range(max_steps)):
            audio_input = mx.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
            audio_logits = self(audio_input, text_tokens, cache)
            cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
            audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = top_k_sampling(audio_logits, top_k, temp, axis=-2)
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens
            mx.eval(audio_seq)

        # Undo delay
        for i in range(self.num_codebooks):
            audio_seq[:, : -self.num_codebooks, i] = audio_seq[
                :, i : -self.num_codebooks + i, i
            ]
        audio_seq = audio_seq[:, 1 : -self.num_codebooks + 1]

        audio_seq = mx.swapaxes(audio_seq, -1, -2)[:, mx.newaxis]
        audio = self._audio_decoder.decode(audio_seq, audio_scales=[None])
        return audio[0]

    @classmethod
    def sanitize(cls, weights):
        out_weights = {}
        for k, arr in weights.items():
            if k.startswith("transformer."):
                k = k[len("transformer.") :]

            if "cross_attention" in k:
                k = k.replace("cross_attention", "cross_attn")

            if "condition_provider" in k:
                k = k.replace(
                    "condition_provider.conditioners.description", "text_conditioner"
                )

            if "in_proj_weight" in k:
                dim = arr.shape[0] // 3
                name = "in_proj_weight"
                out_weights[k.replace(name, "q_proj.weight")] = arr[:dim]
                out_weights[k.replace(name, "k_proj.weight")] = arr[dim : dim * 2]
                out_weights[k.replace(name, "v_proj.weight")] = arr[dim * 2 :]
                continue

            out_weights[k] = arr
        return out_weights

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        download_progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            if download_progress_callback is not None:
                # Signal the UI that a download is about to start, even before the first byte tick.
                download_progress_callback(
                    {
                        "repo_id": path_or_repo,
                        "downloaded_bytes": 0,
                        "total_bytes": 0,
                        "percent": 0,
                        "desc": "Starting download",
                        "done": False,
                    }
                )
            tqdm_class = None
            if download_progress_callback is not None:
                from .hf_progress import make_hf_tqdm_class

                tqdm_class = make_hf_tqdm_class(
                    repo_id=path_or_repo,
                    on_progress=download_progress_callback,
                )
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "state_dict.bin"],
                    tqdm_class=tqdm_class,
                )
            )

        with open(path / "config.json", "r") as f:
            config = SimpleNamespace(**json.load(f))
            config.text_encoder = SimpleNamespace(**config.text_encoder)
            config.audio_encoder = SimpleNamespace(**config.audio_encoder)
            config.decoder = SimpleNamespace(**config.decoder)

        weights = _load_torch_best_state(path / "state_dict.bin")
        weights = {k: mx.array(v) for k, v in weights.items()}
        weights = cls.sanitize(weights)

        model = MusicGen(config, download_progress_callback=download_progress_callback)
        model.load_weights(list(weights.items()))
        return model


def _apply_delay_pattern(tokens: mx.array, bos_token_id: int) -> mx.array:
    """
    Convert non-delayed tokens (B, K, T) to delayed sequence (B, T + K, K).
    Position 0 is BOS for all codebooks, and token (k, t) appears at pos 1 + k + t.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens should have shape (B, K, T)")
    bsz, num_codebooks, num_frames = tokens.shape
    seq_len = num_frames + num_codebooks
    seq = mx.full((bsz, seq_len, num_codebooks), bos_token_id)
    for k in range(num_codebooks):
        seq[:, 1 + k : 1 + k + num_frames, k] = tokens[:, k, :]
    return seq


def _remove_delay_pattern(audio_seq: mx.array, num_codebooks: int) -> mx.array:
    """Undo MusicGen delay pattern to recover tokens (B, T, K)."""
    for i in range(num_codebooks):
        audio_seq[:, : -num_codebooks, i] = audio_seq[
            :, i : -num_codebooks + i, i
        ]
    audio_seq = audio_seq[:, 1 : -num_codebooks + 1]
    return audio_seq


def _build_prompt_sequence(
    prompt_tokens: mx.array,
    total_frames: int,
    bos_token_id: int,
):
    """Create delayed sequence with prompt tokens and a fill mask."""
    if prompt_tokens.ndim != 3:
        raise ValueError("prompt_tokens should have shape (B, K, T)")
    bsz, num_codebooks, prompt_frames = prompt_tokens.shape
    seq_len = total_frames + num_codebooks
    seq = mx.full((bsz, seq_len, num_codebooks), bos_token_id, dtype=mx.int32)
    prompt_tokens = prompt_tokens.astype(seq.dtype)
    fill_mask = mx.zeros((seq_len, num_codebooks), dtype=mx.bool_)
    for k in range(num_codebooks):
        start = 1 + k
        end = start + total_frames
        fill_mask[start:end, k] = True
        if prompt_frames > 0:
            seq[:, start : start + prompt_frames, k] = prompt_tokens[:, k, :]
            fill_mask[start : start + prompt_frames, k] = False
    return seq, fill_mask


class MusicGenContinuation(MusicGen):
    @staticmethod
    def _safe_progress_callback(
        callback: Optional[Callable[[int, int], None]],
        current: int,
        total: int,
    ) -> None:
        if callback is None:
            return
        try:
            callback(current, total)
        except Exception as e:  # pragma: no cover
            # Progress reporting should never crash generation.
            print(f"[WARN] progress_callback failed: {e}")

    def _prepare_text_tokens(
        self,
        text: str,
        guidance_coef: float,
        allow_empty_text_cfg: bool = False,
    ):
        text = "" if text is None else text
        text_tokens = self.text_conditioner(text)
        use_cfg = guidance_coef != 0.0
        if text.strip() == "" and not allow_empty_text_cfg:
            # Empty text behaves better as unconditional; avoid CFG in this case.
            text_tokens = mx.zeros_like(text_tokens)
            use_cfg = False
        return text_tokens, use_cfg

    @property
    def frame_rate(self) -> float:
        return self._audio_decoder.quantizer.frame_rate

    def _decoder_num_codebooks(self) -> int:
        quantizer = getattr(self._audio_decoder, "quantizer", None)
        layers = getattr(quantizer, "layers", None)
        if layers is not None:
            return int(len(layers))
        return int(self.num_codebooks)

    def decode_tokens(self, audio_tokens: mx.array) -> mx.array:
        """Decode tokens of shape (B, T, K) to audio (B, T_audio, C)."""
        if audio_tokens.ndim != 3:
            raise ValueError("audio_tokens should have shape (B, T, K)")
        if int(audio_tokens.shape[1]) == 0 or int(audio_tokens.shape[2]) == 0:
            channels = int(getattr(self._audio_decoder, "channels", 1))
            return mx.zeros(
                (int(audio_tokens.shape[0]), 0, channels),
                dtype=mx.float32,
            )
        token_codebooks = int(audio_tokens.shape[2])
        decoder_codebooks = self._decoder_num_codebooks()
        decoder_channels = int(getattr(self._audio_decoder, "channels", 1))

        # Stereo fine-tunes can interleave two mono-codebook streams:
        # [k0_left, k0_right, k1_left, k1_right, ...].
        if (
            decoder_channels == 1
            and token_codebooks == decoder_codebooks * 2
        ):
            left_tokens = audio_tokens[:, :, 0::2]
            right_tokens = audio_tokens[:, :, 1::2]
            left_seq = mx.swapaxes(left_tokens, -1, -2)[:, mx.newaxis]
            right_seq = mx.swapaxes(right_tokens, -1, -2)[:, mx.newaxis]
            left_audio = self._audio_decoder.decode(left_seq, audio_scales=[None])
            right_audio = self._audio_decoder.decode(right_seq, audio_scales=[None])
            return mx.concatenate([left_audio, right_audio], axis=-1)

        if token_codebooks > decoder_codebooks:
            raise ValueError(
                "Token codebook count exceeds decoder capacity "
                f"({token_codebooks} > {decoder_codebooks}) and does not match stereo interleaving."
            )

        audio_seq = mx.swapaxes(audio_tokens, -1, -2)[:, mx.newaxis]
        audio = self._audio_decoder.decode(audio_seq, audio_scales=[None])
        return audio

    def generate_tokens(
        self,
        text: str,
        max_steps: int = 200,
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
        allow_empty_text_cfg: bool = False,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> mx.array:
        """Generate tokens (B, T, K) without decoding."""
        audio_shape = (1, max_steps + 1, self.num_codebooks)
        audio_seq = mx.full(audio_shape, self.bos_token_id)

        text_tokens, use_cfg = self._prepare_text_tokens(
            text, guidance_coef, allow_empty_text_cfg
        )
        if use_cfg:
            text_tokens = mx.concatenate([text_tokens, mx.zeros_like(text_tokens)], axis=0)

        head_dim = self.hidden_size // self.num_attention_heads
        cache = [
            KVCache(head_dim, self.num_attention_heads) for _ in range(len(self.layers))
        ]

        total_iters = range(max_steps)
        if progress:
            total_iters = tqdm(total_iters)

        last_pct = 0
        for offset in total_iters:
            audio_input = (
                mx.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
                if use_cfg
                else audio_seq[:, offset : offset + 1]
            )
            audio_logits = self(audio_input, text_tokens, cache)
            if use_cfg:
                cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
                audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = top_k_sampling(audio_logits, top_k, temp, axis=-2)
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.bos_token_id
            audio_seq[:, offset + 1 : offset + 2] = audio_tokens
            mx.eval(audio_seq)
            if progress_callback is not None:
                step = offset + 1
                pct = int((step / max_steps) * 100) if max_steps else 100
                if pct > last_pct:
                    self._safe_progress_callback(progress_callback, step, max_steps)
                    last_pct = pct

        audio_seq = _remove_delay_pattern(audio_seq, self.num_codebooks)
        return audio_seq

    def _generate_tokens_with_prompt(
        self,
        text: str,
        prompt_tokens: mx.array,
        max_new_steps: int,
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
        allow_empty_text_cfg: bool = False,
        progress: bool = True,
        prepend_bos: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> mx.array:
        if prompt_tokens.ndim != 3:
            raise ValueError("prompt_tokens should have shape (B, K, T)")
        if prompt_tokens.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for prompt_tokens.")
        if int(prompt_tokens.shape[1]) != int(self.num_codebooks):
            target_codebooks = int(self.num_codebooks)
            current_codebooks = int(prompt_tokens.shape[1])
            if current_codebooks > target_codebooks:
                print(
                    "[WARN] prompt_tokens codebook dimension exceeds model expectation; truncating."
                )
                prompt_tokens = prompt_tokens[:, :target_codebooks, :]
            else:
                print(
                    "[WARN] prompt_tokens codebook dimension below model expectation; padding BOS."
                )
                bos_pad = mx.full(
                    (int(prompt_tokens.shape[0]), target_codebooks - current_codebooks, int(prompt_tokens.shape[-1])),
                    self.bos_token_id,
                    dtype=prompt_tokens.dtype,
                )
                prompt_tokens = mx.concatenate([prompt_tokens, bos_pad], axis=1)

        prompt_frames = prompt_tokens.shape[-1]
        if prepend_bos:
            bos_frame = mx.full(
                (prompt_tokens.shape[0], prompt_tokens.shape[1], 1),
                self.bos_token_id,
                dtype=prompt_tokens.dtype,
            )
            prompt_tokens = mx.concatenate([bos_frame, prompt_tokens], axis=-1)
        total_frames = prompt_tokens.shape[-1] + max_new_steps
        max_steps = total_frames + self.num_codebooks - 1

        audio_seq, fill_mask = _build_prompt_sequence(
            prompt_tokens=prompt_tokens,
            total_frames=total_frames,
            bos_token_id=self.bos_token_id,
        )
        text_tokens, use_cfg = self._prepare_text_tokens(
            text, guidance_coef, allow_empty_text_cfg
        )
        if use_cfg:
            text_tokens = mx.concatenate([text_tokens, mx.zeros_like(text_tokens)], axis=0)

        head_dim = self.hidden_size // self.num_attention_heads
        cache = [
            KVCache(head_dim, self.num_attention_heads) for _ in range(len(self.layers))
        ]

        total_iters = range(max_steps)
        if progress:
            total_iters = tqdm(total_iters)

        last_pct = 0
        for offset in total_iters:
            audio_input = (
                mx.tile(audio_seq[:, offset : offset + 1], [2, 1, 1])
                if use_cfg
                else audio_seq[:, offset : offset + 1]
            )
            audio_logits = self(audio_input, text_tokens, cache)
            if use_cfg:
                cond_logits, uncond_logits = audio_logits[:1], audio_logits[1:2]
                audio_logits = uncond_logits + (cond_logits - uncond_logits) * guidance_coef
            audio_tokens = top_k_sampling(audio_logits, top_k, temp, axis=-2)
            # "delay" pattern
            audio_tokens[..., offset + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + offset] = self.bos_token_id
            next_slice = audio_seq[:, offset + 1 : offset + 2]
            valid_here = fill_mask[offset + 1 : offset + 2]
            valid_here = mx.expand_dims(valid_here, axis=0)
            audio_seq[:, offset + 1 : offset + 2] = mx.where(
                valid_here,
                audio_tokens,
                next_slice,
            )
            mx.eval(audio_seq)
            if progress_callback is not None:
                step = offset + 1
                pct = int((step / max_steps) * 100) if max_steps else 100
                if pct > last_pct:
                    self._safe_progress_callback(progress_callback, step, max_steps)
                    last_pct = pct

        audio_seq = _remove_delay_pattern(audio_seq, self.num_codebooks)
        if prepend_bos:
            audio_seq = audio_seq[:, 1:, :]
        return audio_seq

    def generate_continuation(
        self,
        prompt_tokens: mx.array,
        max_new_steps: int = 200,
        text: str = "",
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
        allow_empty_text_cfg: bool = False,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        return_tokens: bool = False,
        prepend_bos: bool = False,
    ):
        tokens = self._generate_tokens_with_prompt(
            text=text,
            prompt_tokens=prompt_tokens,
            max_new_steps=max_new_steps,
            top_k=top_k,
            temp=temp,
            guidance_coef=guidance_coef,
            allow_empty_text_cfg=allow_empty_text_cfg,
            progress=progress,
            prepend_bos=prepend_bos,
            progress_callback=progress_callback,
        )
        prompt_frames = prompt_tokens.shape[-1]
        cont_tokens = tokens[:, prompt_frames:]
        full_audio = self.decode_tokens(tokens)
        cont_audio = (
            self.decode_tokens(cont_tokens)
            if int(cont_tokens.shape[1]) > 0
            else mx.zeros(
                (int(full_audio.shape[0]), 0, int(full_audio.shape[-1])),
                dtype=full_audio.dtype,
            )
        )
        if return_tokens:
            return full_audio, cont_audio, tokens
        return full_audio, cont_audio

    def generate(
        self,
        text: str,
        max_steps: int = 200,
        top_k: int = 250,
        temp: float = 1.0,
        guidance_coef: float = 3.0,
        allow_empty_text_cfg: bool = False,
    ) -> mx.array:
        tokens = self.generate_tokens(
            text=text,
            max_steps=max_steps,
            top_k=top_k,
            temp=temp,
            guidance_coef=guidance_coef,
            allow_empty_text_cfg=allow_empty_text_cfg,
        )
        audio_seq = mx.swapaxes(tokens, -1, -2)[:, mx.newaxis]
        audio = self._audio_decoder.decode(audio_seq, audio_scales=[None])
        return audio[0]
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo: str,
        base_model: Optional[str] = None,
        prefer_base_config: bool = False,
        download_progress_callback: Optional[Callable[[dict], None]] = None,
    ):
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            if download_progress_callback is not None:
                download_progress_callback(
                    {
                        "repo_id": path_or_repo,
                        "downloaded_bytes": 0,
                        "total_bytes": 0,
                        "percent": 0,
                        "desc": "Starting download",
                        "done": False,
                    }
                )
            tqdm_class = None
            if download_progress_callback is not None:
                from .hf_progress import make_hf_tqdm_class

                tqdm_class = make_hf_tqdm_class(
                    repo_id=path_or_repo,
                    on_progress=download_progress_callback,
                )
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "state_dict.bin"],
                    tqdm_class=tqdm_class,
                )
            )

        config_path = path / "config.json"
        if prefer_base_config and base_model is not None:
            config_path = Path("__force_base_config__")

        if not config_path.exists():
            if base_model is None:
                raise FileNotFoundError(
                    f"config.json not found in {path}. "
                    "Provide --base-model to load a base config."
                )
            base_path = Path(base_model)
            if not base_path.exists():
                if download_progress_callback is not None:
                    download_progress_callback(
                        {
                            "repo_id": base_model,
                            "downloaded_bytes": 0,
                            "total_bytes": 0,
                            "percent": 0,
                            "desc": "Starting download",
                            "done": False,
                        }
                    )
                tqdm_class = None
                if download_progress_callback is not None:
                    from .hf_progress import make_hf_tqdm_class

                    tqdm_class = make_hf_tqdm_class(
                        repo_id=base_model,
                        on_progress=download_progress_callback,
                    )
                base_path = Path(
                    snapshot_download(
                        repo_id=base_model,
                        allow_patterns=["config.json"],
                        tqdm_class=tqdm_class,
                    )
                )
            config_path = base_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"config.json not found in base model {base_model}."
                )

        with open(config_path, "r") as f:
            config = SimpleNamespace(**json.load(f))
            config.text_encoder = SimpleNamespace(**config.text_encoder)
            config.audio_encoder = SimpleNamespace(**config.audio_encoder)
            config.decoder = SimpleNamespace(**config.decoder)

        weights = _load_torch_best_state(path / "state_dict.bin")
        weights = {k: mx.array(v) for k, v in weights.items()}
        weights = cls.sanitize(weights)

        model = cls(config, download_progress_callback=download_progress_callback)
        model.load_weights(list(weights.items()))
        return model
