"""Unit tests for immutable ACE artifact source resolution."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from acestep.models.mlx.artifact_source import (
    SourceFileRecord,
    checkpoint_files,
    hash_checkpoint_files,
    verify_huggingface_source,
)


class ArtifactSourceTests(unittest.TestCase):
    """Exercise local hashing and pinned remote identity checks."""

    def test_hashes_single_file_checkpoint_in_reproducible_order(self) -> None:
        """A single-file checkpoint includes config followed by its weights."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            save_file(
                {"decoder.weight": np.array([[1.0, 2.0]], dtype=np.float32)},
                root / "model.safetensors",
            )

            records = hash_checkpoint_files(root)

            self.assertEqual(
                [record.filename for record in records],
                ["config.json", "model.safetensors"],
            )
            self.assertTrue(all(len(record.sha256) == 64 for record in records))

    def test_rejects_index_that_references_a_missing_shard(self) -> None:
        """A sharded checkpoint fails before conversion when a shard is absent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"decoder.weight": "missing.safetensors"}}),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                checkpoint_files(root)

    def test_verifies_lfs_hash_without_downloading_weights(self) -> None:
        """A matching LFS SHA is copied into the returned provenance record."""
        record = SourceFileRecord("model.safetensors", 12, "a" * 64)
        sibling = SimpleNamespace(
            rfilename=record.filename,
            size=record.size,
            blob_id="blob-id",
            lfs=SimpleNamespace(sha256=record.sha256),
        )
        api = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                sha="b" * 40,
                siblings=[sibling],
            )
        )
        with patch("huggingface_hub.HfApi", return_value=api), patch(
            "huggingface_hub.hf_hub_download"
        ) as download:
            verified = verify_huggingface_source(
                (record,),
                repo_id="ACE-Step/example",
                revision="b" * 40,
            )

        download.assert_not_called()
        self.assertEqual(verified[0].remote_blob_id, "blob-id")
        self.assertEqual(verified[0].remote_lfs_sha256, record.sha256)

    def test_rejects_remote_lfs_hash_mismatch(self) -> None:
        """A local file cannot be attributed to a different remote LFS object."""
        record = SourceFileRecord("model.safetensors", 12, "a" * 64)
        sibling = SimpleNamespace(
            rfilename=record.filename,
            size=record.size,
            lfs=SimpleNamespace(sha256="c" * 64),
        )
        api = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                sha="b" * 40,
                siblings=[sibling],
            )
        )
        with patch("huggingface_hub.HfApi", return_value=api):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_huggingface_source(
                    (record,),
                    repo_id="ACE-Step/example",
                    revision="b" * 40,
                )

    def test_verifies_checkpoint_inside_remote_subdirectory(self) -> None:
        """Monorepo checkpoints retain their exact remote filename in provenance."""
        record = SourceFileRecord("model.safetensors", 12, "a" * 64)
        sibling = SimpleNamespace(
            rfilename="acestep-v15-turbo/model.safetensors",
            size=record.size,
            blob_id="blob-id",
            lfs=SimpleNamespace(sha256=record.sha256),
        )
        api = SimpleNamespace(
            model_info=lambda *_args, **_kwargs: SimpleNamespace(
                sha="b" * 40,
                siblings=[sibling],
            )
        )
        with patch("huggingface_hub.HfApi", return_value=api):
            verified = verify_huggingface_source(
                (record,),
                repo_id="ACE-Step/Ace-Step1.5",
                revision="b" * 40,
                source_subdir="acestep-v15-turbo",
            )

        self.assertEqual(
            verified[0].remote_filename,
            "acestep-v15-turbo/model.safetensors",
        )


if __name__ == "__main__":
    unittest.main()
