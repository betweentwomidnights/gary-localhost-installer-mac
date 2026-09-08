"""Export reproducible, sharded ACE-Step decoder artifacts for native MLX."""

from __future__ import annotations

import gc
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .artifact_source import SourceFileRecord, checkpoint_weight_map, sha256_file
from .artifact_manifest import write_artifact_metadata

FORMAT_NAME = "thepatch-ace-step-mlx"
FORMAT_VERSION = 1
NULL_CONDITION_KEY = "__acestep__.null_condition_emb"


def _dtype(name: str):
    """Resolve a supported artifact dtype by stable CLI name."""
    import mlx.core as mx

    supported = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    try:
        return supported[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported MLX artifact dtype {name!r}") from exc


def expected_decoder_specs(config: Any) -> dict[str, tuple[int, ...]]:
    """Return the exact target parameter shapes for one ACE decoder config."""
    from mlx.utils import tree_flatten

    from .dit_model import MLXDiTDecoder

    model = MLXDiTDecoder.from_config(config)
    specs = {name: tuple(value.shape) for name, value in tree_flatten(model.parameters())}
    del model
    return specs


def _prepare_shard(
    source_path: Path,
    *,
    expected: dict[str, tuple[int, ...]],
    dtype_name: str,
) -> tuple[dict[str, Any], int, int]:
    """Convert decoder tensors in one source shard to MLX names and layouts."""
    import mlx.core as mx

    from .dit_convert import _convert_checkpoint_array, _local_decoder_key

    raw = dict(mx.load(str(source_path)))
    converted: dict[str, Any] = {}
    parameter_count = 0
    target_dtype = _dtype(dtype_name)
    for source_name, value in raw.items():
        if source_name.endswith("null_condition_emb"):
            converted[NULL_CONDITION_KEY] = value.astype(target_dtype)
            continue
        local_name = _local_decoder_key(source_name)
        if local_name is None:
            continue
        target_name, array = _convert_checkpoint_array(
            local_name,
            value,
            dtype=target_dtype,
        )
        target_shape = expected.get(target_name)
        if target_shape is None:
            raise ValueError(f"Unknown MLX decoder target {target_name!r}")
        if tuple(array.shape) != target_shape:
            raise ValueError(
                f"Shape mismatch for {target_name}: {tuple(array.shape)} != {target_shape}"
            )
        if target_name in converted:
            raise ValueError(f"Duplicate MLX decoder target {target_name!r}")
        converted[target_name] = array
        parameter_count += int(array.size)
    del raw
    tensor_bytes = sum(int(array.nbytes) for array in converted.values())
    return converted, parameter_count, tensor_bytes


def _save_shard(path: Path, weights: dict[str, Any], metadata: dict[str, str]) -> None:
    """Materialize and save one converted shard with string metadata."""
    import mlx.core as mx

    mx.save_safetensors(str(path), weights, metadata=metadata)
    del weights
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def export_decoder_artifact(
    *,
    model_dir: str | Path,
    output_dir: str | Path,
    variant: str,
    source_repo: str,
    source_revision: str,
    source_files: tuple[SourceFileRecord, ...],
    dtype_name: str = "float16",
    converter: dict[str, Any] | None = None,
) -> Path:
    """Atomically export one variant and return its completed artifact directory."""
    from .dit_convert import load_model_config

    source = Path(model_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite artifact directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    weight_map = checkpoint_weight_map(source)
    source_shards = sorted(set(weight_map.values()))
    config = load_model_config(source)
    expected = expected_decoder_specs(config)
    loaded: set[str] = set()
    artifact_files = []
    parameter_count = 0
    tensor_bytes = 0
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.partial-", dir=destination.parent))
    try:
        shutil.copy2(source / "config.json", staging / "config.json")
        artifact_map: dict[str, str] = {}
        for index, shard_name in enumerate(source_shards, start=1):
            output_name = (
                "model.safetensors"
                if len(source_shards) == 1
                else f"model-{index:05d}-of-{len(source_shards):05d}.safetensors"
            )
            weights, shard_parameters, shard_bytes = _prepare_shard(
                source / shard_name,
                expected=expected,
                dtype_name=dtype_name,
            )
            overlap = loaded.intersection(key for key in weights if key != NULL_CONDITION_KEY)
            if overlap:
                raise ValueError(f"Duplicate targets across shards: {sorted(overlap)[:5]}")
            loaded.update(key for key in weights if key != NULL_CONDITION_KEY)
            parameter_count += shard_parameters
            tensor_bytes += shard_bytes
            for key in weights:
                artifact_map[key] = output_name
            output_path = staging / output_name
            _save_shard(
                output_path,
                weights,
                {
                    "format": FORMAT_NAME,
                    "format_version": str(FORMAT_VERSION),
                    "variant": variant,
                    "dtype": dtype_name,
                    "source_repo": source_repo,
                    "source_revision": source_revision,
                },
            )
            del weights
            artifact_files.append(
                {
                    "filename": output_name,
                    "size": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                }
            )
        missing = set(expected) - loaded
        if missing:
            raise ValueError(f"Artifact is missing {len(missing)} decoder tensors")
        if NULL_CONDITION_KEY not in artifact_map:
            raise ValueError("Artifact is missing the null condition embedding")
        write_artifact_metadata(
            staging,
            format_name=FORMAT_NAME,
            format_version=FORMAT_VERSION,
            variant=variant,
            dtype_name=dtype_name,
            tensor_count=len(expected),
            parameter_count=parameter_count,
            tensor_bytes=tensor_bytes,
            source_repo=source_repo,
            source_revision=source_revision,
            source_files=source_files,
            artifact_files=artifact_files,
            artifact_map=artifact_map,
            converter=converter,
        )
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
