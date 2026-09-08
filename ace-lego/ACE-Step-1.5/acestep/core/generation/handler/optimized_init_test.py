"""Tests for guarded optimized initialization and legacy-checkpoint bypass."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.core.generation.handler.init_service_downloads import (
    InitServiceDownloadsMixin,
)
from acestep.core.generation.handler.optimized_init import resolve_optimized_artifact


class OptimizedInitTests(unittest.TestCase):
    """Verify opt-in resolution and safe shared-only prechecks."""

    def test_resolver_is_disabled_without_opt_in(self):
        """The optimized route should remain guarded by its environment flag."""
        with patch.dict(os.environ, {"ACESTEP_MLX_OPTIMIZED_LOAD": "0"}):
            self.assertIsNone(
                resolve_optimized_artifact(
                    variant="acestep-v15-xl-base",
                    use_mlx_dit=True,
                    resolved_device="mps",
                )
            )

    def test_verified_artifact_skips_legacy_dit_precheck(self):
        """A complete artifact should require shared VAE/Qwen files, not old DiT files."""
        host = InitServiceDownloadsMixin()
        with patch.object(host, "_missing_required_files", return_value=[]):
            with patch(
                "acestep.core.generation.handler.init_service_downloads.check_model_exists"
            ) as check_model:
                result = host._ensure_models_present(
                    checkpoint_path=Path("/missing/checkpoints"),
                    config_path="acestep-v15-xl-base",
                    prefer_source=None,
                    require_dit=False,
                )
        self.assertIsNone(result)
        check_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
