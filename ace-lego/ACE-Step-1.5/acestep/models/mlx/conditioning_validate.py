"""Validate decoder-free conditioning artifacts against their exact source."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from .artifact_source import (
    SourceFileRecord,
    checkpoint_weight_map,
    hash_checkpoint_files,
    sha256_file,
)


def _source_records(source: Path) -> tuple[SourceFileRecord, ...]:
    """Hash checkpoint inputs plus the conditioning silence latent."""
    records = list(hash_checkpoint_files(source))
    silence = source / "silence_latent.pt"
    records.append(
        SourceFileRecord(silence.name, silence.stat().st_size, sha256_file(silence))
    )
    return tuple(records)


def validate_conditioning_parity(
    source_dir: str | Path,
    artifact_dir: str | Path,
) -> dict[str, int]:
    """Require exact identity for every conditioning tensor and the silence latent."""
    source = Path(source_dir).expanduser().resolve()
    root = Path(artifact_dir).expanduser().resolve() / "conditioning"
    provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    if sha256_file(index_path) != provenance["artifact"]["index_sha256"]:
        raise ValueError("Conditioning index identity mismatch")

    pinned = {record["filename"]: record for record in provenance["source"]["files"]}
    for record in _source_records(source):
        expected = pinned.get(record.filename)
        if expected is None or record.size != expected["size"] or record.sha256 != expected["sha256"]:
            raise ValueError(f"Conditioning source identity mismatch: {record.filename}")
    silence = provenance["artifact"]["silence_latent"]
    silence_path = root / "silence_latent.pt"
    if silence_path.stat().st_size != silence["size"] or sha256_file(silence_path) != silence["sha256"]:
        raise ValueError("Conditioning silence latent identity mismatch")

    source_map = checkpoint_weight_map(source)
    if set(index) - set(source_map):
        raise ValueError("Conditioning index references tensors absent from source")
    files = {record["filename"]: record for record in provenance["artifact"]["files"]}
    compared = 0
    for artifact_name in sorted(set(index.values())):
        artifact_path = root / artifact_name
        record = files.get(artifact_name)
        if record is None or artifact_path.stat().st_size != record["size"]:
            raise ValueError(f"Conditioning shard identity mismatch: {artifact_name}")
        if sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Conditioning shard hash mismatch: {artifact_name}")
        names = sorted(name for name, filename in index.items() if filename == artifact_name)
        with safe_open(str(artifact_path), framework="pt", device="cpu") as artifact:
            for name in names:
                with safe_open(str(source / source_map[name]), framework="pt", device="cpu") as original:
                    if not torch.equal(artifact.get_tensor(name), original.get_tensor(name)):
                        raise ValueError(f"Conditioning tensor parity mismatch: {name}")
                compared += 1
    if compared != provenance["tensor_count"]:
        raise ValueError("Conditioning tensor count mismatch")
    return {"compared_tensors": compared, "artifact_shards": len(files)}
