"""Tests for native MLX DiT quantization configuration."""

import os
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn

from acestep.models.mlx.dit_quantization import (
    DiTQuantization,
    dequantize_dit_linears,
    quantize_dit,
    requested_dit_quantization,
)


class _TinyDecoder(nn.Module):
    """Provide a quantizable decoder-shaped module."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(64, 32)


class DiTQuantizationTest(unittest.TestCase):
    """Verify parsing without requiring model allocation."""

    def test_quantization_is_disabled_by_default(self) -> None:
        """Keep the shipping FP16 decoder unchanged without an explicit flag."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(requested_dit_quantization())

    def test_four_bit_alias_is_accepted(self) -> None:
        """Map the public four-bit alias to affine group quantization."""
        with patch.dict(os.environ, {"ACESTEP_MLX_DIT_QUANTIZATION": "4bit"}):
            setting = requested_dit_quantization()
        self.assertEqual(setting.bits, 4)
        self.assertEqual(setting.group_size, 64)

    def test_invalid_quantization_is_rejected(self) -> None:
        """Reject ambiguous modes instead of silently changing precision."""
        with patch.dict(os.environ, {"ACESTEP_MLX_DIT_QUANTIZATION": "3bit"}):
            with self.assertRaisesRegex(ValueError, "off, 4bit, or 8bit"):
                requested_dit_quantization()

    def test_dequantize_replaces_packed_linears(self) -> None:
        """Restore ordinary FP16 modules so native adapters can merge weights."""
        decoder = _TinyDecoder()
        setting = DiTQuantization(bits=8)
        quantize_dit(decoder, setting)
        self.assertIsInstance(decoder.proj, nn.QuantizedLinear)
        restored = dequantize_dit_linears(decoder)
        self.assertEqual(restored, setting)
        self.assertIsInstance(decoder.proj, nn.Linear)
        self.assertEqual(decoder.proj.weight.dtype, mx.float16)
        self.assertIsNone(decoder.mlx_weight_quantization)


if __name__ == "__main__":
    unittest.main()
