"""Unit tests for ACE decoder artifact conversion and packaging."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx

from acestep.models.mlx import artifact_export, artifact_validate, dit_convert
from acestep.models.mlx.artifact_source import hash_checkpoint_files, sha256_file


def _write_source(root: Path) -> dict[str, tuple[int, ...]]:
    """Write a tiny representative checkpoint and return expected MLX shapes."""
    root.joinpath("config.json").write_text("{}\n", encoding="utf-8")
    weights = {
        "decoder.proj_in.1.weight": mx.arange(24, dtype=mx.float32).reshape(2, 3, 4),
        "decoder.proj_out.1.weight": mx.arange(24, dtype=mx.float32).reshape(3, 2, 4),
        "decoder.layer.weight": mx.arange(6, dtype=mx.float32).reshape(2, 3),
        "decoder.null_condition_emb": mx.ones((1, 2, 3)),
        "encoder.ignored.weight": mx.zeros((2, 2)),
    }
    mx.save_safetensors(str(root / "model.safetensors"), weights)
    return {
        "proj_in.weight": (2, 4, 3),
        "proj_out.weight": (2, 4, 3),
        "layer.weight": (2, 3),
    }


class ArtifactExportTests(unittest.TestCase):
    """Verify tensor conversion and atomic artifact metadata."""

    def test_prepares_decoder_tensors_with_mlx_layout_and_dtype(self) -> None:
        """Convolutions are transposed while unrelated component tensors are skipped."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = _write_source(root)

            weights, parameter_count, tensor_bytes = artifact_export._prepare_shard(
                root / "model.safetensors",
                expected=expected,
                dtype_name="float16",
            )
            mx.eval(weights)

            self.assertEqual(
                set(weights),
                {*expected, artifact_export.NULL_CONDITION_KEY},
            )
            self.assertEqual(tuple(weights["proj_in.weight"].shape), (2, 4, 3))
            self.assertEqual(tuple(weights["proj_out.weight"].shape), (2, 4, 3))
            self.assertEqual(weights["layer.weight"].dtype, mx.float16)
            self.assertEqual(parameter_count, 54)
            self.assertEqual(tensor_bytes, 120)

    def test_exports_self_hashing_artifact_with_standard_index(self) -> None:
        """A completed artifact pins source identity and hashes every output shard."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            expected = _write_source(source)
            output = root / "artifact"

            with patch.object(dit_convert, "load_model_config", return_value=object()), patch.object(
                artifact_export,
                "expected_decoder_specs",
                return_value=expected,
            ):
                artifact_export.export_decoder_artifact(
                    model_dir=source,
                    output_dir=output,
                    variant="tiny-test",
                    source_repo="ACE-Step/example",
                    source_revision="b" * 40,
                    source_files=hash_checkpoint_files(source),
                )

            provenance = json.loads((output / "provenance.json").read_text())
            index = json.loads((output / "model.safetensors.index.json").read_text())
            artifact_file = provenance["artifact"]["files"][0]
            self.assertEqual(provenance["decoder_tensor_count"], 3)
            self.assertEqual(provenance["artifact_tensor_count"], 4)
            self.assertEqual(index["metadata"]["total_size"], 120)
            self.assertEqual(artifact_file["sha256"], sha256_file(output / artifact_file["filename"]))
            self.assertIn(artifact_export.NULL_CONDITION_KEY, index["weight_map"])
            self.assertFalse(any(output.parent.glob(".artifact.partial-*")))

            with patch.object(dit_convert, "load_model_config", return_value=object()), patch.object(
                artifact_validate,
                "expected_decoder_specs",
                return_value=expected,
            ):
                structural = artifact_validate.validate_artifact(output)
                parity = artifact_validate.validate_source_parity(source, output)
            self.assertEqual(structural["artifact_tensors"], 4)
            self.assertEqual(parity["compared_tensors"], 4)

    def test_refuses_to_overwrite_an_existing_artifact(self) -> None:
        """Existing artifact paths are never mutated by a conversion attempt."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                artifact_export.export_decoder_artifact(
                    model_dir=temporary,
                    output_dir=output,
                    variant="tiny-test",
                    source_repo="ACE-Step/example",
                    source_revision="b" * 40,
                    source_files=(),
                )


if __name__ == "__main__":
    unittest.main()
