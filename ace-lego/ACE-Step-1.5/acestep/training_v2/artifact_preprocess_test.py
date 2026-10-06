"""Regression tests for artifact-only Mac training preprocessing."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from acestep.training_v2 import artifact_preprocess as assets
from acestep.training_v2.model_loader import _resolve_silence_latent_path


class ArtifactPreprocessTests(unittest.TestCase):
    """Keep Mac conditioning-shell reuse separate from full Torch training."""

    def test_without_mac_gate_no_artifact_lookup_occurs(self):
        """Ordinary CPU/CUDA preprocessing retains its original loading path."""
        with patch.dict(os.environ, {"ACESTEP_TRAINING_USE_MLX_ARTIFACTS": "0"}):
            with patch("acestep.core.generation.handler.optimized_init.resolve_optimized_artifact") as resolve:
                self.assertIsNone(assets.resolve_preprocessing_artifact("base"))
        resolve.assert_not_called()

    def test_mac_gate_normalizes_regular_base_alias(self):
        """The Mac wrapper's base alias resolves only verified local artifacts."""
        with patch.dict(os.environ, {"ACESTEP_TRAINING_USE_MLX_ARTIFACTS": "1"}):
            with patch("acestep.core.generation.handler.optimized_init.resolve_optimized_artifact", return_value=Path("/artifact")) as resolve:
                self.assertEqual(assets.resolve_preprocessing_artifact("base"), Path("/artifact"))
        resolve.assert_called_once_with(variant="acestep-v15-base", use_mlx_dit=True, resolved_device="cpu")

    def test_conditioner_loads_without_legacy_model_directory(self):
        """Only conditioning weights are needed for pass two, not a Torch decoder."""
        model = MagicMock()
        with patch.object(assets, "resolve_preprocessing_artifact", return_value=Path("/artifact")):
            with patch("acestep.core.generation.handler.conditioning_shell.load_conditioning_shell", return_value=model) as load:
                assets.load_preprocessing_conditioner("/missing", "base", "cpu", "fp32")
        load.assert_called_once_with(
            variant="acestep-v15-base", config_root=Path("/artifact"),
            weights_root=Path("/artifact/conditioning"), dtype=torch.float32,
        )
        model.requires_grad_.assert_called_once_with(False)
        model.to.assert_called_once_with(device="cpu")

    def test_missing_artifact_uses_legacy_conditioner(self):
        """Legacy installations still preprocess from their full source model."""
        with patch.object(assets, "resolve_preprocessing_artifact", return_value=None):
            with patch("acestep.training_v2.model_loader.load_decoder_for_training", return_value="legacy") as load:
                self.assertEqual(assets.load_preprocessing_conditioner("/checkpoints", "base", "cpu", "fp32"), "legacy")
        load.assert_called_once_with("/checkpoints", "base", "cpu", "fp32")

    def test_silence_latent_comes_from_the_same_artifact(self):
        """Artifact-only preprocessing must not borrow another variant's latent."""
        with patch.object(assets, "resolve_preprocessing_artifact", return_value=Path("/artifact")):
            self.assertEqual(_resolve_silence_latent_path("/missing", "base"), Path("/artifact/conditioning/silence_latent.pt"))


if __name__ == "__main__":
    unittest.main()
