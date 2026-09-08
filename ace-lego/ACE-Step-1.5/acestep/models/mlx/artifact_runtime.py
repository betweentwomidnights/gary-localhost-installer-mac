"""Resolve and load pinned ACE-Step MLX decoder artifacts at runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_export import FORMAT_NAME, FORMAT_VERSION, NULL_CONDITION_KEY
from .artifact_source import sha256_file

ARTIFACT_REPO_ID = "thepatch/acestep-1.5-mlx"
ARTIFACT_REVISION = "218ca5e51809821177bdfed70bf500020df6e958"


@dataclass(frozen=True)
class ArtifactSpec:
    """Describe one supported optimized artifact at an immutable Hub revision."""

    variant: str
    subdir: str
    source_revision: str
    dtype_name: str = "float16"


ARTIFACTS = {
    "acestep-v15-base": ArtifactSpec(
        variant="acestep-v15-base",
        subdir="MLX/acestep-v15-base-f16",
        source_revision="e432212fec32b8965a14ffa57ae653438d6abd14",
    ),
    "acestep-v15-sft": ArtifactSpec(
        variant="acestep-v15-sft",
        subdir="MLX/acestep-v15-sft-f16",
        source_revision="c410d249e71ea9385a7b586865e65b1473e1098d",
    ),
    "acestep-v15-turbo": ArtifactSpec(
        variant="acestep-v15-turbo",
        subdir="MLX/acestep-v15-turbo-f16",
        source_revision="19671f406d603126926c1b7e2adc169acbcade22",
    ),
    "acestep-v15-xl-base": ArtifactSpec(
        variant="acestep-v15-xl-base",
        subdir="MLX/acestep-v15-xl-base-f16",
        source_revision="220c1166efbdd9583eafcb12eb160594bbfcb241",
    ),
    "acestep-v15-xl-sft": ArtifactSpec(
        variant="acestep-v15-xl-sft",
        subdir="MLX/acestep-v15-xl-sft-f16",
        source_revision="d06de46b4622f781cf07f4a013a67d591ca52819",
    ),
    "acestep-v15-xl-turbo": ArtifactSpec(
        variant="acestep-v15-xl-turbo",
        subdir="MLX/acestep-v15-xl-turbo-f16",
        source_revision="d4a0b288b83ebb7e25a8c0b32c573c22e134e8ee",
    ),
}


def optimized_load_requested() -> bool:
    """Return whether the guarded optimized Carey load path is enabled."""
    value = os.getenv("ACESTEP_MLX_OPTIMIZED_LOAD", "0")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_artifact_dir(variant: str) -> Path:
    """Return a verified local artifact directory, downloading the pinned snapshot if needed."""
    spec = ARTIFACTS.get(variant)
    if spec is None:
        raise ValueError(f"No optimized MLX artifact is registered for {variant!r}")
    override = os.getenv("ACESTEP_MLX_ARTIFACT_DIR", "").strip()
    if override:
        root = Path(override).expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(
            repo_id=ARTIFACT_REPO_ID,
            revision=ARTIFACT_REVISION,
            allow_patterns=[f"{spec.subdir}/*", f"{spec.subdir}/conditioning/*"],
            token=False,
        )
        root = Path(snapshot) / spec.subdir
    _validate_identity(root, spec)
    return root


def _validate_identity(root: Path, spec: ArtifactSpec) -> None:
    """Reject missing, stale, or incompatible artifact metadata before model construction."""
    provenance_path = root / "provenance.json"
    index_path = root / "model.safetensors.index.json"
    config_path = root / "config.json"
    if not provenance_path.is_file() or not index_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Incomplete ACE MLX artifact directory: {root}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "variant": spec.variant,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Artifact {key} mismatch: {provenance.get(key)!r} != {value!r}")
    if provenance.get("source", {}).get("revision") != spec.source_revision:
        raise ValueError("Artifact source revision does not match the runtime catalog")
    if sha256_file(index_path) != provenance["artifact"]["index_sha256"]:
        raise ValueError("Artifact index SHA-256 mismatch")
    if sha256_file(config_path) != provenance["artifact"]["config_sha256"]:
        raise ValueError("Artifact config SHA-256 mismatch")
    records = {record["filename"]: record for record in provenance["artifact"]["files"]}
    for filename, record in records.items():
        path = root / filename
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Artifact shard identity mismatch: {filename}")
    _validate_conditioning(root, spec)


def _validate_conditioning(root: Path, spec: ArtifactSpec) -> None:
    """Reject a missing or mismatched decoder-free conditioning component."""
    conditioning = root / "conditioning"
    provenance_path = conditioning / "provenance.json"
    index_path = conditioning / "model.safetensors.index.json"
    silence_path = conditioning / "silence_latent.pt"
    if not provenance_path.is_file() or not index_path.is_file() or not silence_path.is_file():
        raise FileNotFoundError(f"Incomplete ACE conditioning artifact: {conditioning}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "format": "thepatch-ace-step-conditioning",
        "format_version": 1,
        "variant": spec.variant,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"Conditioning artifact {key} mismatch")
    if provenance.get("source", {}).get("revision") != spec.source_revision:
        raise ValueError("Conditioning source revision does not match runtime catalog")
    artifact = provenance.get("artifact", {})
    if sha256_file(index_path) != artifact.get("index_sha256"):
        raise ValueError("Conditioning artifact index SHA-256 mismatch")
    for record in artifact.get("files", []):
        path = conditioning / record["filename"]
        if not path.is_file() or path.stat().st_size != record["size"]:
            raise ValueError(f"Conditioning shard identity mismatch: {record['filename']}")
    silence = artifact.get("silence_latent", {})
    if silence_path.stat().st_size != silence.get("size"):
        raise ValueError("Conditioning silence latent size mismatch")


def load_decoder_artifact(root: str | Path, decoder: Any) -> Any:
    """Load an MLX decoder strictly from native shards and return its null embedding."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    artifact = Path(root).expanduser().resolve()
    index = json.loads(
        (artifact / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    expected = {name: tuple(value.shape) for name, value in tree_flatten(decoder.parameters())}
    loaded: set[str] = set()
    null_condition = None
    for shard_name in sorted(set(index.values())):
        values = dict(mx.load(str(artifact / shard_name)))
        if NULL_CONDITION_KEY in values:
            null_condition = values.pop(NULL_CONDITION_KEY)
            mx.eval(null_condition)
        unknown = set(values) - set(expected)
        if unknown:
            raise ValueError(f"Unknown MLX artifact tensor(s): {sorted(unknown)[:5]}")
        for name, value in values.items():
            if tuple(value.shape) != expected[name]:
                raise ValueError(f"MLX artifact shape mismatch for {name}")
        decoder.load_weights(list(values.items()), strict=False)
        mx.eval(list(values.values()))
        loaded.update(values)
    missing = set(expected) - loaded
    if missing:
        raise ValueError(f"MLX artifact is missing {len(missing)} decoder tensor(s)")
    if null_condition is None:
        raise ValueError("MLX artifact is missing the null condition embedding")
    return null_condition
