"""Tests for native MLX DiT cross-attention cache policy."""

import os
import unittest
from unittest.mock import patch

from acestep.models.mlx.dit_cache_policy import (
    cfg_kv_cache_enabled,
    should_create_cross_attention_cache,
)


class DiTCachePolicyTest(unittest.TestCase):
    """Verify cache selection without loading an MLX model."""

    def test_cfg_cache_is_opt_in(self) -> None:
        """Keep CFG caching disabled unless its feature flag is enabled."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(cfg_kv_cache_enabled())
            self.assertFalse(
                should_create_cross_attention_cache(compiled=False, do_cfg=True)
            )

    def test_cfg_cache_accepts_true_flag(self) -> None:
        """Enable uncompiled CFG caching for a recognized true value."""
        with patch.dict(os.environ, {"ACESTEP_MLX_CFG_KV_CACHE": "true"}):
            self.assertTrue(
                should_create_cross_attention_cache(compiled=False, do_cfg=True)
            )

    def test_non_cfg_cache_behavior_is_unchanged(self) -> None:
        """Continue caching non-CFG cross-attention without an opt-in flag."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(
                should_create_cross_attention_cache(compiled=False, do_cfg=False)
            )

    def test_compilation_disables_all_python_caches(self) -> None:
        """Never introduce the Python cache into a compiled decoder graph."""
        with patch.dict(os.environ, {"ACESTEP_MLX_CFG_KV_CACHE": "1"}):
            self.assertFalse(
                should_create_cross_attention_cache(compiled=True, do_cfg=True)
            )
            self.assertFalse(
                should_create_cross_attention_cache(compiled=True, do_cfg=False)
            )


if __name__ == "__main__":
    unittest.main()
