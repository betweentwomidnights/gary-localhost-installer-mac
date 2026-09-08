"""Write indexes and provenance records for ACE MLX artifacts."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .artifact_source import SourceFileRecord, sha256_file


def write_artifact_metadata(
    staging: Path,
    *,
    format_name: str,
    format_version: int,
    variant: str,
    dtype_name: str,
    tensor_count: int,
    parameter_count: int,
    tensor_bytes: int,
    source_repo: str,
    source_revision: str,
    source_files: tuple[SourceFileRecord, ...],
    artifact_files: list[dict[str, Any]],
    artifact_map: dict[str, str],
    converter: dict[str, Any] | None,
) -> None:
    """Write the standard safetensors index and immutable provenance manifest."""
    index_path = staging / "model.safetensors.index.json"
    index_payload = {
        "metadata": {"total_size": tensor_bytes},
        "weight_map": dict(sorted(artifact_map.items())),
    }
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "format": format_name,
        "format_version": format_version,
        "variant": variant,
        "dtype": dtype_name,
        "decoder_tensor_count": tensor_count,
        "artifact_tensor_count": tensor_count + 1,
        "auxiliary_tensors": ["__acestep__.null_condition_emb"],
        "parameter_count": parameter_count,
        "tensor_bytes": tensor_bytes,
        "source": {
            "repo_id": source_repo,
            "revision": source_revision,
            "files": [asdict(record) for record in source_files],
        },
        "artifact": {
            "files": artifact_files,
            "config_sha256": sha256_file(staging / "config.json"),
            "index_sha256": sha256_file(index_path),
        },
        "converter": {
            "mlx_version": importlib.metadata.version("mlx"),
            **(converter or {}),
        },
    }
    (staging / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
