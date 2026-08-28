"""Tests for training model/asset resolution.

The silence latent is the one that bites: it used to fall back to scanning
every known variant subdirectory, so a partial install trained against another
model's latent instead of failing.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ACE_ROOT = Path(__file__).resolve().parents[2]
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from acestep.training_v2.model_loader import _resolve_silence_latent_path  # noqa: E402


class ResolveSilenceLatentPathTests(unittest.TestCase):
    def make_checkpoints(self, temp: str, *variants: str) -> Path:
        checkpoints = Path(temp)
        for name in variants:
            (checkpoints / name).mkdir(parents=True)
            (checkpoints / name / "silence_latent.pt").write_bytes(name.encode())
        return checkpoints

    def test_xl_literal_variant_resolves_its_own_silence_latent(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = self.make_checkpoints(
                temp, "acestep-v15-base", "acestep-v15-xl-base"
            )

            resolved = _resolve_silence_latent_path(
                checkpoints, "acestep-v15-xl-base"
            )

            self.assertEqual(
                resolved, checkpoints / "acestep-v15-xl-base" / "silence_latent.pt"
            )

    def test_base_alias_resolves_through_the_variant_map(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = self.make_checkpoints(temp, "acestep-v15-base")

            resolved = _resolve_silence_latent_path(checkpoints, "base")

            self.assertEqual(
                resolved, checkpoints / "acestep-v15-base" / "silence_latent.pt"
            )

    def test_missing_xl_assets_raise_instead_of_borrowing_base(self):
        # The regression. _VARIANT_DIR has no xl-base entry, so the old
        # last-resort scan matched acestep-v15-base and trained silently
        # against the wrong latent.
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = self.make_checkpoints(temp, "acestep-v15-base")

            with self.assertRaises(FileNotFoundError):
                _resolve_silence_latent_path(checkpoints, "acestep-v15-xl-base")

    def test_root_latent_wins_for_custom_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = self.make_checkpoints(temp, "acestep-v15-base")
            (checkpoints / "silence_latent.pt").write_bytes(b"root")

            resolved = _resolve_silence_latent_path(checkpoints, "base")

            self.assertEqual(resolved, checkpoints / "silence_latent.pt")

    def test_no_variant_scans_known_subdirectories(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = self.make_checkpoints(temp, "acestep-v15-sft")

            resolved = _resolve_silence_latent_path(checkpoints, None)

            self.assertEqual(
                resolved, checkpoints / "acestep-v15-sft" / "silence_latent.pt"
            )


if __name__ == "__main__":
    unittest.main()
