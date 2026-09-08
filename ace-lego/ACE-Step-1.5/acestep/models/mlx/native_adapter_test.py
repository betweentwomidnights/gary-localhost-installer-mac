"""Tests for direct PEFT LoRA/DoRA application to MLX decoder weights."""

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from acestep.models.mlx.native_adapter import NativeMLXAdapter


class _TinyDecoder(nn.Module):
    """Small decoder-shaped module used to exercise MLX parameter updates."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(3, 2, bias=False)]


class NativeMLXAdapterTests(unittest.TestCase):
    """Validate native merge math, restoration, and strict input checks."""

    def setUp(self) -> None:
        self.decoder = _TinyDecoder()
        self.base = mx.array([[1.0, 2.0, 3.0], [2.0, -1.0, 0.5]], dtype=mx.float16)
        self.a = mx.array([[0.5, -1.0, 2.0]], dtype=mx.float16)
        self.b = mx.array([[2.0], [-0.5]], dtype=mx.float16)
        self.decoder.load_weights([("layers.0.weight", self.base)])

    def _write_adapter(
        self, root: Path, *, dora: bool = False, unnamed: bool = False
    ) -> Path:
        """Write one tiny PEFT-compatible adapter directory."""
        config = {
            "peft_type": "LORA",
            "r": 1,
            "lora_alpha": 2,
            "rank_pattern": {"layers.0": 1},
            "alpha_pattern": {"layers.0": 3},
            "use_dora": dora,
            "use_rslora": False,
            "fan_in_fan_out": False,
        }
        (root / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        prefix = "base_model.model.layers.0"
        adapter_segment = "" if unnamed else ".default"
        weights = {
            f"{prefix}.lora_A{adapter_segment}.weight": self.a,
            f"{prefix}.lora_B{adapter_segment}.weight": self.b,
        }
        if dora:
            magnitude_suffix = "" if unnamed else ".default.weight"
            weights[f"{prefix}.lora_magnitude_vector{magnitude_suffix}"] = mx.array(
                [4.0, 2.5], dtype=mx.float16
            )
        mx.save_safetensors(str(root / "adapter_model.safetensors"), weights)
        return root

    def _weight(self):
        """Return the current tiny decoder weight."""
        return self.decoder.layers[0].weight

    def _restore(self) -> None:
        """Restore the test decoder as the pinned artifact loader would."""
        self.decoder.load_weights([("layers.0.weight", self.base)])

    def test_quantized_decoder_is_rejected_before_adapter_io(self):
        """Give an actionable error because packed weights cannot be merged."""
        self.decoder.mlx_weight_quantization = object()
        with self.assertRaisesRegex(ValueError, "requires the FP16 decoder"):
            NativeMLXAdapter(self.decoder, "/missing", self._restore)

    def test_lora_uses_suffix_alpha_pattern_and_restores_base(self):
        """LoRA should use alpha/rank scaling and unload to exact FP16 base."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = NativeMLXAdapter(
                self.decoder, self._write_adapter(Path(tmp)), self._restore
            )
            adapter.apply(0.5)
            expected = self.base.astype(mx.float32) + (self.b @ self.a) * 3.0 * 0.5
            self.assertTrue(mx.allclose(self._weight(), expected.astype(mx.float16)))
            adapter.unload()
            self.assertTrue(mx.array_equal(self._weight(), self.base))

    def test_dora_matches_row_normalized_peft_merge(self):
        """DoRA at full strength should normalize each row to learned magnitude."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = NativeMLXAdapter(
                self.decoder, self._write_adapter(Path(tmp), dora=True), self._restore
            )
            adapter.apply(1.0)
            value = self.base.astype(mx.float32) + (self.b @ self.a) * 3.0
            norm = mx.sqrt(mx.sum(value**2, axis=1, keepdims=True))
            expected = value / norm * mx.array([4.0, 2.5])[:, None]
            self.assertTrue(mx.allclose(self._weight(), expected.astype(mx.float16)))
            adapter.apply(0.0)
            self.assertTrue(mx.array_equal(self._weight(), self.base))

    def test_unnamed_peft_dora_tensor_layout_is_supported(self):
        """PEFT 0.18 may omit `.default` and `.weight` on magnitude vectors."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = NativeMLXAdapter(
                self.decoder,
                self._write_adapter(Path(tmp), dora=True, unnamed=True),
                self._restore,
            )
            self.assertEqual(adapter.target_count, 1)
            adapter.apply(1.0)
            value = self.base.astype(mx.float32) + (self.b @ self.a) * 3.0
            norm = mx.sqrt(mx.sum(value**2, axis=1, keepdims=True))
            expected = value / norm * mx.array([4.0, 2.5])[:, None]
            self.assertTrue(mx.allclose(self._weight(), expected.astype(mx.float16)))

    def test_eval_batch_size_is_validated(self):
        """Invalid synchronization batch configuration should fail clearly."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ", {"ACESTEP_MLX_ADAPTER_EVAL_BATCH_SIZE": "0"}
            ):
                with self.assertRaisesRegex(ValueError, "between 1 and 1024"):
                    NativeMLXAdapter(
                        self.decoder, self._write_adapter(Path(tmp)), self._restore
                    )

    def test_default_eval_batch_size_is_thirty_two(self):
        """The default should amortize synchronization without a huge lazy graph."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ", {"ACESTEP_MLX_ADAPTER_EVAL_BATCH_SIZE": ""}
            ):
                adapter = NativeMLXAdapter(
                    self.decoder, self._write_adapter(Path(tmp)), self._restore
                )
                self.assertEqual(adapter.eval_batch_size, 32)

    def test_dora_config_requires_magnitude_vectors(self):
        """A DoRA config without magnitude tensors must fail before mutation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_adapter(Path(tmp))
            config_path = root / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["use_dora"] = True
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DoRA magnitude/config mismatch"):
                NativeMLXAdapter(self.decoder, root, self._restore)

    def test_declared_base_variant_must_match(self):
        """Adapters declaring another ACE family must fail before mutation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_adapter(Path(tmp))
            config_path = root / "adapter_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["base_model_name_or_path"] = "/models/acestep-v15-base"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                NativeMLXAdapter(
                    self.decoder,
                    root,
                    self._restore,
                    expected_variant="acestep-v15-xl-base",
                )


if __name__ == "__main__":
    unittest.main()
