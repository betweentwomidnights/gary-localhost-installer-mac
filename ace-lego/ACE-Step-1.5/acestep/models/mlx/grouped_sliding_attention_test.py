"""Tests for exact grouped MLX sliding-window attention."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import mlx.core as mx

from acestep.models.mlx.grouped_sliding_attention import (
    grouped_sliding_attention,
    grouped_sliding_attention_enabled,
    grouped_sliding_block_size,
)


def _dense_sliding_attention(q, k, v, *, scale: float, window_size: int):
    """Return the dense masked reference for bidirectional local attention."""
    seq_len = q.shape[2]
    positions = mx.arange(seq_len)
    valid = mx.abs(positions[:, None] - positions[None, :]) <= window_size
    mask = mx.where(
        valid,
        mx.array(0.0, dtype=q.dtype),
        mx.array(-1e9, dtype=q.dtype),
    )[None, None, :, :]
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


class GroupedSlidingAttentionTests(unittest.TestCase):
    """Validate grouped attention configuration, boundaries, and numerics."""

    def test_feature_flag_defaults_off_and_accepts_true(self):
        """The experimental runtime path must remain opt-in until full validation."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(grouped_sliding_attention_enabled())
        with patch.dict(os.environ, {"ACESTEP_MLX_GROUPED_SLIDING_ATTENTION": "true"}):
            self.assertTrue(grouped_sliding_attention_enabled())

    def test_block_size_defaults_to_window_and_validates_override(self):
        """Block configuration should be predictable and reject invalid values."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(grouped_sliding_block_size(128), 128)
        with patch.dict(os.environ, {"ACESTEP_MLX_GROUPED_SLIDING_BLOCK_SIZE": "64"}):
            self.assertEqual(grouped_sliding_block_size(128), 64)
        with patch.dict(os.environ, {"ACESTEP_MLX_GROUPED_SLIDING_BLOCK_SIZE": "0"}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                grouped_sliding_block_size(128)

    def test_grouped_matches_dense_across_block_boundaries(self):
        """Grouped FP32 output should match dense SDPA for arbitrary lengths."""
        scale = 8**-0.5
        for seq_len in (1, 15, 16, 17, 31, 32, 33, 65, 129):
            with self.subTest(seq_len=seq_len):
                mx.random.seed(1000 + seq_len)
                q = mx.random.normal((2, 4, seq_len, 8)).astype(mx.float32)
                k = mx.random.normal((2, 4, seq_len, 8)).astype(mx.float32)
                v = mx.random.normal((2, 4, seq_len, 8)).astype(mx.float32)
                expected = _dense_sliding_attention(
                    q, k, v, scale=scale, window_size=16
                )
                actual = grouped_sliding_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    window_size=16,
                    block_size=16,
                )
                mx.eval(expected, actual)
                error = float(mx.max(mx.abs(expected - actual)))
                self.assertLessEqual(error, 3e-6)

    def test_grouped_matches_dense_when_block_differs_from_window(self):
        """The exact mask must hold when tuning query groups independently."""
        mx.random.seed(42)
        q = mx.random.normal((1, 2, 73, 16)).astype(mx.float16)
        k = mx.random.normal((1, 2, 73, 16)).astype(mx.float16)
        v = mx.random.normal((1, 2, 73, 16)).astype(mx.float16)
        scale = 16**-0.5
        expected = _dense_sliding_attention(q, k, v, scale=scale, window_size=12)
        actual = grouped_sliding_attention(
            q,
            k,
            v,
            scale=scale,
            window_size=12,
            block_size=20,
        )
        mx.eval(expected, actual)
        self.assertLessEqual(float(mx.max(mx.abs(expected - actual))), 2e-3)

    def test_rejects_mismatched_qkv_shapes(self):
        """Mismatched Q/K/V shapes should fail instead of broadcasting silently."""
        q = mx.zeros((1, 2, 8, 4))
        with self.assertRaisesRegex(ValueError, "matching Q/K/V shapes"):
            grouped_sliding_attention(
                q,
                mx.zeros((1, 2, 7, 4)),
                q,
                scale=0.5,
                window_size=4,
                block_size=4,
            )


if __name__ == "__main__":
    unittest.main()
