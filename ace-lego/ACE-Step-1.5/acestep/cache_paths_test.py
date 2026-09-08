"""Tests for writable ACE-Step cache path resolution."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep.cache_paths import get_acestep_cache_root


class CachePathsTest(unittest.TestCase):
    def test_explicit_cache_dir_wins(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "explicit"
            with patch.dict(
                os.environ,
                {"ACESTEP_CACHE_DIR": str(explicit), "XDG_CACHE_HOME": "/ignored"},
                clear=False,
            ):
                self.assertEqual(get_acestep_cache_root("/project"), str(explicit.resolve()))

    def test_xdg_cache_is_used_for_packaged_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                os.environ,
                {"ACESTEP_CACHE_DIR": "", "XDG_CACHE_HOME": temp_dir},
                clear=False,
            ):
                expected = (Path(temp_dir) / "acestep").resolve()
                self.assertEqual(get_acestep_cache_root("/project"), str(expected))

    def test_source_checkout_fallback_is_unchanged(self):
        with patch.dict(
            os.environ,
            {"ACESTEP_CACHE_DIR": "", "XDG_CACHE_HOME": ""},
            clear=False,
        ):
            expected = (Path("/project") / ".cache" / "acestep").resolve()
            self.assertEqual(get_acestep_cache_root("/project"), str(expected))


if __name__ == "__main__":
    unittest.main()
