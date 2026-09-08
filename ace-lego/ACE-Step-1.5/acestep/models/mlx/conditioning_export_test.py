"""Tests for deterministic conditioning artifact shard planning."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from acestep.models.mlx.artifact_source import sha256_file
from acestep.models.mlx.conditioning_export import _load_group, _shard_groups
from acestep.models.mlx.conditioning_validate import validate_conditioning_parity


class ConditioningExportTests(unittest.TestCase):
    """Validate bounded deterministic shell sharding."""

    def test_groups_are_sorted_and_bounded(self):
        """Tensor names should pack deterministically without exceeding the limit."""
        specs = {
            "z": ((3,), torch.float32),
            "a": ((2,), torch.float32),
            "m": ((1,), torch.float32),
        }
        self.assertEqual(_shard_groups(specs, 12), [["a", "m"], ["z"]])

    def test_oversized_single_tensor_remains_its_own_group(self):
        """A single tensor larger than the target should not create an empty shard."""
        specs = {"large": ((10,), torch.float32), "small": ((1,), torch.float32)}
        self.assertEqual(_shard_groups(specs, 8), [["large"], ["small"]])

    def test_source_floating_dtype_is_preserved(self):
        """BF16 source tensors are valid when meta construction reports FP32."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file(
                {"bias": torch.ones(4, dtype=torch.bfloat16)},
                root / "model.safetensors",
            )
            values = _load_group(
                root,
                {"bias": "model.safetensors"},
                ["bias"],
                {"bias": ((4,), torch.float32)},
            )

        self.assertEqual(values["bias"].dtype, torch.bfloat16)

    def test_conditioning_parity_accepts_exact_source(self):
        """The validator checks tensors, source files, and the silence latent."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, artifact = root / "source", root / "artifact"
            conditioning = artifact / "conditioning"
            source.mkdir()
            conditioning.mkdir(parents=True)
            (source / "config.json").write_text("{}\n", encoding="utf-8")
            (source / "silence_latent.pt").write_bytes(b"silence")
            save_file({"bias": torch.ones(4)}, source / "model.safetensors")
            save_file({"bias": torch.ones(4)}, conditioning / "part.safetensors")
            index = conditioning / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {"bias": "part.safetensors"}}),
                encoding="utf-8",
            )
            source_files = []
            for name in ("config.json", "model.safetensors", "silence_latent.pt"):
                path = source / name
                source_files.append(
                    {"filename": name, "size": path.stat().st_size, "sha256": sha256_file(path)}
                )
            part = conditioning / "part.safetensors"
            silence = conditioning / "silence_latent.pt"
            silence.write_bytes(b"silence")
            provenance = {
                "tensor_count": 1,
                "source": {"files": source_files},
                "artifact": {
                    "index_sha256": sha256_file(index),
                    "files": [{"filename": part.name, "size": part.stat().st_size, "sha256": sha256_file(part)}],
                    "silence_latent": {"size": silence.stat().st_size, "sha256": sha256_file(silence)},
                },
            }
            (conditioning / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")

            report = validate_conditioning_parity(source, artifact)

        self.assertEqual(report["compared_tensors"], 1)


if __name__ == "__main__":
    unittest.main()
