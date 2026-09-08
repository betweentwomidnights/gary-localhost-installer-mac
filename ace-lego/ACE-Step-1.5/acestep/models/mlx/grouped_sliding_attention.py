"""Exact grouped sliding-window attention for the native MLX ACE decoder."""

from __future__ import annotations

import os

import mlx.core as mx


_ENABLED_ENV = "ACESTEP_MLX_GROUPED_SLIDING_ATTENTION"
_BLOCK_SIZE_ENV = "ACESTEP_MLX_GROUPED_SLIDING_BLOCK_SIZE"


def grouped_sliding_attention_enabled() -> bool:
    """Return whether the experimental grouped MLX attention path is enabled."""
    return os.getenv(_ENABLED_ENV, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def grouped_sliding_block_size(window_size: int) -> int:
    """Resolve a positive query-group size, defaulting to the model window."""
    raw = os.getenv(_BLOCK_SIZE_ENV, "").strip()
    if not raw:
        return int(window_size)
    try:
        block_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_BLOCK_SIZE_ENV} must be an integer") from exc
    if block_size < 1:
        raise ValueError(f"{_BLOCK_SIZE_ENV} must be positive")
    return block_size


def _grouped_attention_mask(
    *,
    seq_len: int,
    num_groups: int,
    block_size: int,
    window_size: int,
    dtype: mx.Dtype,
) -> mx.array:
    """Build exact per-group bidirectional masks, including padded boundaries."""
    group = mx.arange(num_groups, dtype=mx.int32)[:, None, None]
    query_offset = mx.arange(block_size, dtype=mx.int32)[None, :, None]
    key_width = block_size + 2 * window_size
    key_offset = mx.arange(key_width, dtype=mx.int32)[None, None, :]

    query_position = group * block_size + query_offset
    key_position = group * block_size - window_size + key_offset
    valid_query = query_position < seq_len
    valid_key = (key_position >= 0) & (key_position < seq_len)
    in_window = mx.abs(query_position - key_position) <= window_size
    valid = valid_query & valid_key & in_window

    # Padded query rows are cropped from the result, but still need one valid
    # key to avoid an all-masked softmax producing NaNs inside the fused kernel.
    padded_query_fallback = (~valid_query) & (key_offset == window_size)
    valid = valid | padded_query_fallback
    zeros = mx.zeros(valid.shape, dtype=dtype)
    negative = mx.full(valid.shape, -1e9, dtype=dtype)
    return mx.where(valid, zeros, negative)


def grouped_sliding_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    window_size: int,
    block_size: int,
) -> mx.array:
    """Compute exact bidirectional local attention without a dense sequence mask.

    Inputs and output use ``[batch, heads, sequence, head_dim]`` layout. Query
    groups attend to a strided K/V view containing the group plus both window
    halos. Arbitrary sequence lengths are padded internally and cropped.
    """
    batch, heads, seq_len, head_dim = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError("grouped sliding attention requires matching Q/K/V shapes")
    if window_size < 0:
        raise ValueError("window_size must be non-negative")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if seq_len == 0:
        return q

    num_groups = (seq_len + block_size - 1) // block_size
    padded_len = num_groups * block_size
    query_padding = padded_len - seq_len
    if query_padding:
        pad = [(0, 0), (0, 0), (0, query_padding), (0, 0)]
        q = mx.pad(q, pad)

    kv_pad = [
        (0, 0),
        (0, 0),
        (window_size, window_size + query_padding),
        (0, 0),
    ]
    k_padded = mx.pad(k, kv_pad)
    v_padded = mx.pad(v, kv_pad)
    padded_kv_len = padded_len + 2 * window_size
    key_width = block_size + 2 * window_size
    view_shape = (batch, heads, num_groups, key_width, head_dim)
    view_strides = (
        heads * padded_kv_len * head_dim,
        padded_kv_len * head_dim,
        block_size * head_dim,
        head_dim,
        1,
    )
    k_grouped = mx.as_strided(k_padded, shape=view_shape, strides=view_strides)
    v_grouped = mx.as_strided(v_padded, shape=view_shape, strides=view_strides)

    q_grouped = q.reshape(batch, heads, num_groups, block_size, head_dim)
    q_grouped = q_grouped.transpose(0, 2, 1, 3, 4).reshape(
        batch * num_groups, heads, block_size, head_dim
    )
    k_grouped = k_grouped.transpose(0, 2, 1, 3, 4).reshape(
        batch * num_groups, heads, key_width, head_dim
    )
    v_grouped = v_grouped.transpose(0, 2, 1, 3, 4).reshape(
        batch * num_groups, heads, key_width, head_dim
    )

    mask = _grouped_attention_mask(
        seq_len=seq_len,
        num_groups=num_groups,
        block_size=block_size,
        window_size=window_size,
        dtype=q.dtype,
    )
    mask = mx.broadcast_to(mask[None, :, None, :, :], (
        batch,
        num_groups,
        1,
        block_size,
        key_width,
    )).reshape(batch * num_groups, 1, block_size, key_width)

    output = mx.fast.scaled_dot_product_attention(
        q_grouped,
        k_grouped,
        v_grouped,
        scale=scale,
        mask=mask,
    )
    output = output.reshape(batch, num_groups, heads, block_size, head_dim)
    output = output.transpose(0, 2, 1, 3, 4).reshape(
        batch,
        heads,
        padded_len,
        head_dim,
    )
    return output[:, :, :seq_len, :]
