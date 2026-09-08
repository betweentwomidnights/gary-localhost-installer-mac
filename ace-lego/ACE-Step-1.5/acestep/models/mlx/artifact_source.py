"""Resolve and verify immutable source files for ACE MLX artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFileRecord:
    """Describe one local source file and its verified content identity."""

    filename: str
    size: int
    sha256: str
    remote_filename: str | None = None
    remote_blob_id: str | None = None
    remote_lfs_sha256: str | None = None


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 hex digest for ``path`` using bounded reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_weight_map(model_dir: str | Path) -> dict[str, str]:
    """Return source tensor-to-shard mappings for sharded or single checkpoints."""
    directory = Path(model_dir).expanduser().resolve()
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index: {index_path}")
        return {str(key): str(value) for key, value in weight_map.items()}

    model_path = directory / "model.safetensors"
    if not model_path.is_file():
        raise FileNotFoundError(f"No ACE safetensors checkpoint under {directory}")
    from safetensors import safe_open

    with safe_open(str(model_path), framework="np") as handle:
        return {key: model_path.name for key in handle.keys()}


def checkpoint_files(model_dir: str | Path) -> tuple[Path, ...]:
    """Return config, optional index, and ordered weight files for a checkpoint."""
    directory = Path(model_dir).expanduser().resolve()
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"ACE config.json not found: {config_path}")
    weight_map = checkpoint_weight_map(directory)
    weight_files = tuple(directory / name for name in sorted(set(weight_map.values())))
    missing = [path for path in weight_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ACE checkpoint shard(s): {missing}")
    index_path = directory / "model.safetensors.index.json"
    support = (config_path, index_path) if index_path.is_file() else (config_path,)
    return support + weight_files


def hash_checkpoint_files(model_dir: str | Path) -> tuple[SourceFileRecord, ...]:
    """Hash every source file required to reproduce an ACE MLX artifact."""
    directory = Path(model_dir).expanduser().resolve()
    return tuple(
        SourceFileRecord(
            filename=str(path.relative_to(directory)),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in checkpoint_files(directory)
    )


def verify_huggingface_source(
    records: tuple[SourceFileRecord, ...],
    *,
    repo_id: str,
    revision: str,
    source_subdir: str = "",
) -> tuple[SourceFileRecord, ...]:
    """Verify local files against one immutable public Hugging Face revision."""
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).model_info(
        repo_id,
        revision=revision,
        files_metadata=True,
        token=False,
    )
    if info.sha != revision:
        raise ValueError(f"Hugging Face resolved {revision!r} to unexpected {info.sha!r}")
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    prefix = source_subdir.strip("/")
    verified = []
    for record in records:
        remote_filename = f"{prefix}/{record.filename}" if prefix else record.filename
        sibling = siblings.get(remote_filename)
        if sibling is None:
            raise ValueError(f"{remote_filename} is absent from {repo_id}@{revision}")
        if sibling.size is not None and int(sibling.size) != record.size:
            raise ValueError(f"Size mismatch for {record.filename}")
        lfs = getattr(sibling, "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_sha and lfs_sha != record.sha256:
            raise ValueError(f"SHA-256 mismatch for {record.filename}")
        if not lfs_sha:
            remote = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_filename,
                    revision=revision,
                    token=False,
                )
            )
            if sha256_file(remote) != record.sha256:
                raise ValueError(f"Content mismatch for {record.filename}")
        verified.append(
            SourceFileRecord(
                **{
                    **asdict(record),
                    "remote_filename": remote_filename,
                    "remote_blob_id": getattr(sibling, "blob_id", None),
                    "remote_lfs_sha256": lfs_sha,
                }
            )
        )
    return tuple(verified)
