"""Tests for pinned decoder and conditioning artifact identity validation."""

import json
import tempfile
import unittest
from pathlib import Path

from acestep.models.mlx.artifact_export import FORMAT_NAME, FORMAT_VERSION
from acestep.models.mlx.artifact_runtime import (
    ARTIFACTS,
    ARTIFACT_REVISION,
    ArtifactSpec,
    _validate_identity,
)
from acestep.models.mlx.artifact_source import sha256_file


class ArtifactRuntimeTests(unittest.TestCase):
    """Exercise strict metadata checks without allocating real model weights."""

    def _artifact(self, root: Path) -> ArtifactSpec:
        """Write a tiny structurally complete decoder and conditioning artifact."""
        spec = ArtifactSpec("variant", "MLX/variant", "source-revision")
        config = root / "config.json"
        index = root / "model.safetensors.index.json"
        shard = root / "model.safetensors"
        config.write_text("{}\n", encoding="utf-8")
        index.write_text('{"weight_map": {}}\n', encoding="utf-8")
        shard.write_bytes(b"decoder")
        provenance = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "variant": spec.variant,
            "source": {"revision": spec.source_revision},
            "artifact": {
                "config_sha256": sha256_file(config),
                "index_sha256": sha256_file(index),
                "files": [
                    {"filename": shard.name, "size": shard.stat().st_size}
                ],
            },
        }
        (root / "provenance.json").write_text(
            json.dumps(provenance), encoding="utf-8"
        )

        conditioning = root / "conditioning"
        conditioning.mkdir()
        cond_index = conditioning / "model.safetensors.index.json"
        cond_shard = conditioning / "model.safetensors"
        silence = conditioning / "silence_latent.pt"
        cond_index.write_text('{"weight_map": {}}\n', encoding="utf-8")
        cond_shard.write_bytes(b"conditioning")
        silence.write_bytes(b"silence")
        cond_provenance = {
            "format": "thepatch-ace-step-conditioning",
            "format_version": 1,
            "variant": spec.variant,
            "source": {"revision": spec.source_revision},
            "artifact": {
                "index_sha256": sha256_file(cond_index),
                "files": [
                    {"filename": cond_shard.name, "size": cond_shard.stat().st_size}
                ],
                "silence_latent": {"size": silence.stat().st_size},
            },
        }
        (conditioning / "provenance.json").write_text(
            json.dumps(cond_provenance), encoding="utf-8"
        )
        return spec

    def test_complete_artifact_passes(self):
        """Matching decoder and conditioning metadata should validate."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _validate_identity(root, self._artifact(root))

    def test_xl_turbo_is_pinned_in_runtime_catalog(self):
        """Resolve XL-turbo only through its immutable public source revision."""
        self.assertEqual(
            ARTIFACT_REVISION,
            "218ca5e51809821177bdfed70bf500020df6e958",
        )
        spec = ARTIFACTS["acestep-v15-xl-turbo"]
        self.assertEqual(spec.subdir, "MLX/acestep-v15-xl-turbo-f16")
        self.assertEqual(
            spec.source_revision,
            "d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
        )

    def test_all_six_variants_are_pinned(self):
        """Every selectable ACE-Step 1.5 checkpoint has one optimized artifact."""
        self.assertEqual(
            set(ARTIFACTS),
            {
                "acestep-v15-base",
                "acestep-v15-sft",
                "acestep-v15-turbo",
                "acestep-v15-xl-base",
                "acestep-v15-xl-sft",
                "acestep-v15-xl-turbo",
            },
        )

    def test_conditioning_index_corruption_is_rejected(self):
        """A changed conditioning index must fail before model construction."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self._artifact(root)
            (root / "conditioning/model.safetensors.index.json").write_text(
                "corrupt", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Conditioning artifact index"):
                _validate_identity(root, spec)


if __name__ == "__main__":
    unittest.main()
