"""Tests for native MLX adapter state transitions on the Carey handler."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from acestep.core.generation.handler.lora.native_mlx import (
    add_native_mlx_lora,
    scale_native_mlx_lora,
    toggle_native_mlx_lora,
    unload_native_mlx_lora,
)


class NativeMLXHandlerTests(unittest.TestCase):
    """Verify native adapter bookkeeping remains compatible with the public API."""

    def _host(self):
        """Return a minimal conditioning-shell handler stub."""
        return SimpleNamespace(
            _conditioning_shell_active=True,
            _mlx_native_adapter=None,
            _mlx_artifact_dir="artifact",
            _mlx_null_condition_emb=None,
            mlx_decoder=object(),
            lora_loaded=False,
            use_lora=False,
            lora_scale=1.0,
            _active_loras={},
            _lora_active_adapter=None,
            _adapter_type=None,
        )

    def test_add_populates_native_public_state(self):
        """A successful native load should populate legacy status fields."""
        host = self._host()
        adapter = SimpleNamespace(use_dora=True, target_count=12, apply=Mock())
        with patch(
            "acestep.core.generation.handler.lora.native_mlx.NativeMLXAdapter",
            return_value=adapter,
        ):
            message = add_native_mlx_lora(host, "adapter", "voice")
        self.assertTrue(message.startswith("✅"))
        adapter.apply.assert_called_once_with(1.0)
        self.assertEqual(host._active_loras, {"voice": 1.0})
        self.assertEqual(host._adapter_type, "dora")

    def test_scale_uses_zero_while_disabled_but_remembers_requested_value(self):
        """Scaling a disabled adapter should keep base weights resident."""
        host = self._host()
        adapter = SimpleNamespace(apply=Mock())
        host._mlx_native_adapter = adapter
        host._lora_active_adapter = "main"
        host._active_loras = {"main": 1.0}
        message = scale_native_mlx_lora(host, "main", 0.4)
        self.assertIn("disabled", message)
        adapter.apply.assert_called_once_with(0.0)
        self.assertEqual(host._active_loras["main"], 0.4)

    def test_toggle_reapplies_remembered_scale(self):
        """Enabling should merge the last requested adapter scale."""
        host = self._host()
        adapter = SimpleNamespace(apply=Mock())
        host._mlx_native_adapter = adapter
        host._lora_active_adapter = "main"
        host._active_loras = {"main": 0.25}
        self.assertTrue(toggle_native_mlx_lora(host, True).startswith("✅"))
        adapter.apply.assert_called_once_with(0.25)

    def test_unload_restores_and_clears_state(self):
        """Unload should restore base weights and clear all adapter state."""
        host = self._host()
        adapter = SimpleNamespace(unload=Mock())
        host._mlx_native_adapter = adapter
        host.lora_loaded = host.use_lora = True
        host._active_loras = {"main": 1.0}
        host._lora_active_adapter = "main"
        self.assertTrue(unload_native_mlx_lora(host).startswith("✅"))
        adapter.unload.assert_called_once_with()
        self.assertIsNone(host._mlx_native_adapter)
        self.assertFalse(host.lora_loaded)
        self.assertEqual(host._active_loras, {})


if __name__ == "__main__":
    unittest.main()
