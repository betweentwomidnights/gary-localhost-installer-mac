"""Export the decoder-free ACE conditioning shell with immutable provenance."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file

from acestep.models.common.configuration_acestep_v15 import AceStepConfig

from .artifact_source import SourceFileRecord, checkpoint_weight_map, sha256_file

FORMAT_NAME = "thepatch-ace-step-conditioning"
FORMAT_VERSION = 1
DEFAULT_SHARD_BYTES = 2_000_000_000


def expected_conditioning_specs(
    variant: str, config_root: str | Path
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Return exact decoder-free state shapes and dtypes for a bundled variant."""
    from acestep.core.generation.handler.conditioning_shell import _model_class

    config = AceStepConfig.from_pretrained(Path(config_root), local_files_only=True)
    with init_empty_weights():
        model = _model_class(variant)(config)
    model.decoder = None
    specs = {
        name: (tuple(value.shape), value.dtype)
        for name, value in model.state_dict().items()
    }
    del model
    return specs


def _shard_groups(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], max_bytes: int
) -> list[list[str]]:
    """Pack sorted tensors into deterministic bounded-size shard groups."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name, (shape, dtype) in sorted(specs.items()):
        size = int(torch.tensor([], dtype=dtype).element_size())
        for dimension in shape:
            size *= dimension
        if current and current_bytes + size > max_bytes:
            groups.append(current)
            current, current_bytes = [], 0
        current.append(name)
        current_bytes += size
    if current:
        groups.append(current)
    return groups


def _load_group(
    source: Path,
    mapping: dict[str, str],
    names: list[str],
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> dict[str, torch.Tensor]:
    """Load and strictly validate one bounded group from source safetensors."""
    values: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(mapping[name], []).append(name)
    for filename, keys in by_file.items():
        with safe_open(str(source / filename), framework="pt", device="cpu") as handle:
            for name in keys:
                value = handle.get_tensor(name)
                shape, dtype = specs[name]
                dtype_mismatch = value.dtype != dtype and not value.is_floating_point()
                if tuple(value.shape) != shape or dtype_mismatch:
                    raise ValueError(f"Conditioning source tensor mismatch for {name}")
                values[name] = value
    return values


def export_conditioning_artifact(
    *,
    model_dir: str | Path,
    artifact_dir: str | Path,
    variant: str,
    source_repo: str,
    source_revision: str,
    source_files: tuple[SourceFileRecord, ...],
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> Path:
    """Atomically add the slim conditioning component to an MLX artifact."""
    source = Path(model_dir).expanduser().resolve()
    artifact = Path(artifact_dir).expanduser().resolve()
    destination = artifact / "conditioning"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite conditioning artifact: {destination}")
    mapping = checkpoint_weight_map(source)
    specs = expected_conditioning_specs(variant, artifact)
    absent = set(specs) - set(mapping)
    if absent:
        raise ValueError(f"Source is missing {len(absent)} conditioning tensor(s)")
    groups = _shard_groups(specs, max_shard_bytes)
    staging = Path(tempfile.mkdtemp(prefix=".conditioning.partial-", dir=artifact))
    files = []
    weight_map: dict[str, str] = {}
    tensor_bytes = 0
    try:
        for number, names in enumerate(groups, start=1):
            filename = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
            values = _load_group(source, mapping, names, specs)
            output = staging / filename
            save_file(
                values,
                str(output),
                metadata={
                    "format": FORMAT_NAME,
                    "format_version": str(FORMAT_VERSION),
                    "variant": variant,
                    "source_repo": source_repo,
                    "source_revision": source_revision,
                },
            )
            weight_map.update({name: filename for name in names})
            tensor_bytes += sum(value.numel() * value.element_size() for value in values.values())
            files.append(
                {
                    "filename": filename,
                    "size": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            )
            del values
        silence_source = source / "silence_latent.pt"
        if not silence_source.is_file():
            raise FileNotFoundError(f"Missing conditioning silence latent: {silence_source}")
        shutil.copy2(silence_source, staging / silence_source.name)
        index = staging / "model.safetensors.index.json"
        index.write_text(
            json.dumps(
                {"metadata": {"total_size": tensor_bytes}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "variant": variant,
            "tensor_count": len(specs),
            "tensor_bytes": tensor_bytes,
            "source": {
                "repo_id": source_repo,
                "revision": source_revision,
                "files": [asdict(record) for record in source_files],
            },
            "artifact": {
                "files": files,
                "index_sha256": sha256_file(index),
                "silence_latent": {
                    "size": (staging / silence_source.name).stat().st_size,
                    "sha256": sha256_file(staging / silence_source.name),
                },
            },
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
