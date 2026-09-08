"""Validate ACE MLX artifact integrity and source tensor parity."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .artifact_export import NULL_CONDITION_KEY, _prepare_shard, expected_decoder_specs
from .artifact_source import checkpoint_weight_map, hash_checkpoint_files, sha256_file


def _json(path: Path) -> dict[str, Any]:
    """Load one JSON object or raise a descriptive format error."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _artifact_records(provenance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return provenance artifact file records keyed by safe basenames."""
    records = {}
    for record in provenance.get("artifact", {}).get("files", []):
        filename = record.get("filename")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Unsafe artifact filename: {filename!r}")
        records[filename] = record
    if not records:
        raise ValueError("Provenance contains no artifact files")
    return records


def validate_artifact(artifact_dir: str | Path) -> dict[str, int]:
    """Verify hashes, safetensors headers, indexes, dtypes, and decoder shapes."""
    from .dit_convert import load_model_config

    root = Path(artifact_dir).expanduser().resolve()
    provenance = _json(root / "provenance.json")
    index_path = root / "model.safetensors.index.json"
    index = _json(index_path)
    config_path = root / "config.json"
    if sha256_file(config_path) != provenance["artifact"]["config_sha256"]:
        raise ValueError("Artifact config hash mismatch")
    if sha256_file(index_path) != provenance["artifact"]["index_sha256"]:
        raise ValueError("Artifact index hash mismatch")

    records = _artifact_records(provenance)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Artifact index has no weight_map object")
    actual_shapes = {}
    dtype_label = {"float16": "F16", "bfloat16": "BF16", "float32": "F32"}[
        provenance["dtype"]
    ]
    for filename, record in records.items():
        path = root / filename
        if path.stat().st_size != record["size"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Artifact file identity mismatch: {filename}")
        with safe_open(str(path), framework="np") as handle:
            metadata = handle.metadata() or {}
            if metadata.get("format") != provenance["format"]:
                raise ValueError(f"Artifact format metadata mismatch: {filename}")
            for key in handle.keys():
                if weight_map.get(key) != filename:
                    raise ValueError(f"Index mismatch for tensor {key!r}")
                tensor = handle.get_slice(key)
                if tensor.get_dtype() != dtype_label:
                    raise ValueError(f"Dtype mismatch for tensor {key!r}")
                if key in actual_shapes:
                    raise ValueError(f"Duplicate artifact tensor {key!r}")
                actual_shapes[key] = tuple(tensor.get_shape())

    if set(actual_shapes) != set(weight_map):
        raise ValueError("Artifact safetensors keys do not match the index")
    expected = expected_decoder_specs(load_model_config(root))
    if {key: actual_shapes.get(key) for key in expected} != expected:
        raise ValueError("Artifact decoder shapes do not match config.json")
    if NULL_CONDITION_KEY not in actual_shapes:
        raise ValueError("Artifact is missing the null condition embedding")
    if provenance["decoder_tensor_count"] != len(expected):
        raise ValueError("Provenance decoder tensor count mismatch")
    return {"decoder_tensors": len(expected), "artifact_tensors": len(actual_shapes)}


def validate_source_parity(
    source_dir: str | Path,
    artifact_dir: str | Path,
) -> dict[str, int]:
    """Reconvert each source shard and require exact equality with its artifact tensors."""
    import mlx.core as mx

    from .dit_convert import load_model_config

    source = Path(source_dir).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()
    provenance = _json(artifact / "provenance.json")
    pinned = {record["filename"]: record for record in provenance["source"]["files"]}
    for record in hash_checkpoint_files(source):
        expected = pinned.get(record.filename)
        if expected is None or record.size != expected["size"] or record.sha256 != expected["sha256"]:
            raise ValueError(f"Source identity mismatch: {record.filename}")

    index = _json(artifact / "model.safetensors.index.json")["weight_map"]
    specs = expected_decoder_specs(load_model_config(source))
    compared = 0
    source_shards = sorted(set(checkpoint_weight_map(source).values()))
    for shard_name in source_shards:
        converted, _parameters, _bytes = _prepare_shard(
            source / shard_name,
            expected=specs,
            dtype_name=provenance["dtype"],
        )
        mx.eval(converted)
        output_names = {index[key] for key in converted}
        if len(output_names) != 1:
            raise ValueError(f"Source shard {shard_name} maps to multiple artifact shards")
        artifact_weights = dict(mx.load(str(artifact / output_names.pop())))
        if set(converted) != set(artifact_weights):
            raise ValueError(f"Tensor set mismatch for source shard {shard_name}")
        for key, value in converted.items():
            if not bool(mx.array_equal(value, artifact_weights[key])):
                raise ValueError(f"Tensor parity mismatch: {key}")
            compared += 1
        del converted, artifact_weights
        gc.collect()
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    return {"source_shards": len(source_shards), "compared_tensors": compared}
