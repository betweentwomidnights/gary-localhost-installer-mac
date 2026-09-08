"""Tests for precision-safe MLX artifact selection during training."""

import os
import unittest
from unittest.mock import patch

from acestep.models.mlx.training_artifact import (
    load_training_artifact_if_compatible,
)


class TrainingArtifactTests(unittest.TestCase):
    """Ensure training never silently changes its requested base precision."""

    def test_matching_fp16_artifact_is_loaded(self):
        """An enabled FP16 job should use the registered FP16 XL artifact."""
        with patch.dict(os.environ, {"ACESTEP_MLX_OPTIMIZED_LOAD": "1"}):
            with patch(
                "acestep.models.mlx.training_artifact.resolve_artifact_dir",
                return_value="/artifact",
            ):
                with patch(
                    "acestep.models.mlx.training_artifact.load_decoder_artifact",
                    return_value="null",
                ) as load:
                    result = load_training_artifact_if_compatible(
                        variant="acestep-v15-xl-base",
                        training_dtype="fp16",
                        decoder="decoder",
                    )
        self.assertEqual(result, ("null", "/artifact"))
        load.assert_called_once_with("/artifact", "decoder")

    def test_bf16_job_does_not_use_fp16_artifact(self):
        """The default BF16 trainer must retain its source-checkpoint loader."""
        with patch.dict(os.environ, {"ACESTEP_MLX_OPTIMIZED_LOAD": "1"}):
            with patch(
                "acestep.models.mlx.training_artifact.resolve_artifact_dir"
            ) as resolve:
                result = load_training_artifact_if_compatible(
                    variant="acestep-v15-xl-base",
                    training_dtype="bf16",
                    decoder="decoder",
                )
        self.assertIsNone(result)
        resolve.assert_not_called()

    def test_bf16_job_can_explicitly_upcast_fp16_artifact(self):
        """The one-download canary may opt into FP16 storage with BF16 math."""
        with patch.dict(os.environ, {"ACESTEP_MLX_OPTIMIZED_LOAD": "1"}):
            with patch(
                "acestep.models.mlx.training_artifact.resolve_artifact_dir",
                return_value="/artifact",
            ) as resolve:
                with patch(
                    "acestep.models.mlx.training_artifact.load_decoder_artifact",
                    return_value="null",
                ):
                    result = load_training_artifact_if_compatible(
                        variant="acestep-v15-xl-base",
                        training_dtype="bf16",
                        decoder="decoder",
                        allow_storage_upcast=True,
                    )

        self.assertEqual(result, ("null", "/artifact"))
        resolve.assert_called_once_with(
            "acestep-v15-xl-base", local_files_only=True
        )


if __name__ == "__main__":
    unittest.main()
