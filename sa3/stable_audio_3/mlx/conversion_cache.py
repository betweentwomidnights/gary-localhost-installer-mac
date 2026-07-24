"""On-disk cache of PyTorch -> MLX converted components.

Converting the SA3 checkpoint at load time costs roughly 20-25 seconds, almost
all of it reading the 8.6 GB PyTorch checkpoint and building torch modules that
are discarded immediately after conversion. On a machine where SA3 and
ACE-Step cannot be resident at the same time, services get stopped and started
constantly, so that cost is paid far more often than "once per session".

This module caches the converted MLX weights so a warm start can construct every
component from ``model_config`` and load MLX tensors directly, never touching
torch or the original checkpoint.

The cache is keyed on everything that can change the converted output. A miss on
any of them reconverts and rewrites:

* ``CONVERSION_FORMAT_VERSION`` -- bump whenever conversion logic changes
* the source checkpoint's identity (resolved path, size, mtime)
* the four component dtypes
* the attention mode, which changes autoencoder construction

Writes land in a temporary directory and are renamed into place, so an
interrupted first run cannot leave a partially written cache that a later run
would trust.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import typing as tp
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

# Bump when the conversion produces different tensors for the same inputs.
CONVERSION_FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"
DIT_WEIGHTS = "dit.safetensors"
TEXT_WEIGHTS = "text_encoder.safetensors"
NUMBER_WEIGHTS = "number_conditioner.safetensors"
AUTOENCODER_WEIGHTS = "autoencoder.safetensors"
TOKENIZER_DIR = "tokenizer"
PADDING_EMBEDDING_KEY = "__conditioner__.padding_embedding"

_REQUIRED_FILES = (
    MANIFEST_NAME,
    DIT_WEIGHTS,
    TEXT_WEIGHTS,
    NUMBER_WEIGHTS,
    AUTOENCODER_WEIGHTS,
)


def default_cache_root() -> Path:
    """Cache location, overridable with ``SA3_MLX_CACHE_DIR``."""

    override = os.environ.get("SA3_MLX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "GaryLocalhost"
        / "sa3"
        / "mlx-cache"
    )


def _checkpoint_identity(resolved_config_path: Path | None) -> dict[str, tp.Any]:
    """Identify the source weights so a changed checkpoint invalidates the cache."""

    if resolved_config_path is None:
        return {"config_path": None}
    config_path = Path(resolved_config_path)
    identity: dict[str, tp.Any] = {
        # The parent directory is the Hugging Face snapshot hash for cached
        # repos, which already pins the revision.
        "snapshot": config_path.parent.name,
        "config_name": config_path.name,
    }
    try:
        stat = config_path.stat()
        identity["config_size"] = int(stat.st_size)
        identity["config_mtime_ns"] = int(stat.st_mtime_ns)
    except OSError:
        pass
    checkpoint = config_path.parent / "model.safetensors"
    try:
        stat = checkpoint.stat()
        identity["checkpoint_size"] = int(stat.st_size)
        identity["checkpoint_mtime_ns"] = int(stat.st_mtime_ns)
    except OSError:
        identity["checkpoint_size"] = None
    return identity


def cache_key(
    *,
    source_name: str,
    resolved_config_path: Path | None,
    dit_dtype: str,
    text_dtype: str,
    number_dtype: str,
    autoencoder_dtype: str,
    attention: str,
) -> str:
    payload = {
        "format_version": CONVERSION_FORMAT_VERSION,
        "source_name": str(source_name),
        "checkpoint": _checkpoint_identity(resolved_config_path),
        "dtypes": {
            "dit": dit_dtype,
            "text": text_dtype,
            "number": number_dtype,
            "autoencoder": autoencoder_dtype,
        },
        # Changes MLXSAMEEncoder/Decoder construction, so it changes the graph
        # the cached weights are meant for.
        "attention": attention,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    return f"{source_name}-{digest}"


def cache_dir_for(cache_root: Path | str, key: str) -> Path:
    return Path(cache_root).expanduser() / key


def is_complete(cache_dir: Path | str) -> bool:
    directory = Path(cache_dir)
    if not directory.is_dir():
        return False
    if not all((directory / name).is_file() for name in _REQUIRED_FILES):
        return False
    try:
        manifest = read_manifest(directory)
    except Exception:
        return False
    return int(manifest.get("format_version", -1)) == CONVERSION_FORMAT_VERSION


def read_manifest(cache_dir: Path | str) -> dict[str, tp.Any]:
    with (Path(cache_dir) / MANIFEST_NAME).open(encoding="utf-8") as handle:
        return json.load(handle)


def _dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _split_empty_tensors(weights: dict) -> tuple[dict, dict]:
    """Separate zero-element tensors, which MLX refuses to serialize.

    SA3 medium has ``bottleneck.noise_scaling_factor`` with shape (1, 0, 1)
    because ``noise_augment_dim`` is 0. It carries no data and is unused at
    runtime, but ``load_weights(strict=True)`` still requires it to be present,
    so its shape and dtype are recorded and the tensor is rebuilt on load.
    """

    dense: dict = {}
    empty: dict = {}
    for key, value in weights.items():
        if int(value.size) == 0:
            empty[key] = {
                "shape": [int(d) for d in value.shape],
                "dtype": _dtype_name(value.dtype),
            }
        else:
            dense[key] = value
    return dense, empty


def _restore_empty_tensors(weights: dict, empty: dict | None) -> dict:
    for key, spec in (empty or {}).items():
        weights[key] = mx.zeros(
            tuple(spec["shape"]),
            dtype=getattr(mx, spec["dtype"]),
        )
    return weights


def save_module_weights(module, path: Path | str) -> dict:
    """Write module parameters; returns metadata for any zero-element tensors."""

    dense, empty = _split_empty_tensors(dict(tree_flatten(module.parameters())))
    mx.save_safetensors(str(path), dense)
    return empty


def load_module_weights(module, path: Path | str, empty: dict | None = None) -> None:
    weights = _restore_empty_tensors(dict(mx.load(str(path))), empty)
    module.load_weights(list(weights.items()), strict=True)
    mx.eval(module.parameters())


def save_text_encoder(encoder, padding_embedding, path: Path | str) -> dict:
    """Store the encoder parameters plus the learned padding embedding."""

    weights = dict(tree_flatten(encoder.parameters()))
    if PADDING_EMBEDDING_KEY in weights:
        raise ValueError(
            f"Encoder unexpectedly contains reserved key {PADDING_EMBEDDING_KEY!r}."
        )
    if padding_embedding is not None:
        weights[PADDING_EMBEDDING_KEY] = padding_embedding
    dense, empty = _split_empty_tensors(weights)
    mx.save_safetensors(str(path), dense)
    return empty


def load_text_encoder(encoder, path: Path | str, empty: dict | None = None):
    """Restore encoder parameters and return the padding embedding, if any."""

    weights = _restore_empty_tensors(dict(mx.load(str(path))), empty)
    padding_embedding = weights.pop(PADDING_EMBEDDING_KEY, None)
    encoder.load_weights(list(weights.items()), strict=True)
    mx.eval(encoder.parameters())
    return padding_embedding


class CacheWriter:
    """Write a cache entry atomically.

    Everything is staged in a sibling temporary directory and renamed into place
    only once every file is present, so a crash or a killed service cannot leave
    a half-written entry that ``is_complete`` would accept.
    """

    def __init__(self, cache_root: Path | str, key: str):
        self.cache_root = Path(cache_root).expanduser()
        self.key = key
        self.final_dir = cache_dir_for(self.cache_root, key)
        self._tmp_dir: Path | None = None

    def __enter__(self) -> Path:
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = Path(
            tempfile.mkdtemp(prefix=f".{self.key}.partial-", dir=self.cache_root)
        )
        return self._tmp_dir

    def __exit__(self, exc_type, exc, tb) -> bool:
        tmp_dir = self._tmp_dir
        self._tmp_dir = None
        if tmp_dir is None:
            return False
        if exc_type is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return False
        missing = [n for n in _REQUIRED_FILES if not (tmp_dir / n).is_file()]
        if missing:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Refusing to publish incomplete MLX cache; missing {missing}."
            )
        if self.final_dir.exists():
            stale = self.final_dir.with_name(self.final_dir.name + ".stale")
            shutil.rmtree(stale, ignore_errors=True)
            self.final_dir.rename(stale)
            shutil.rmtree(stale, ignore_errors=True)
        tmp_dir.rename(self.final_dir)
        return False


def write_manifest(cache_dir: Path | str, manifest: dict[str, tp.Any]) -> None:
    payload = dict(manifest)
    payload["format_version"] = CONVERSION_FORMAT_VERSION
    with (Path(cache_dir) / MANIFEST_NAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def entry_size_bytes(cache_dir: Path | str) -> int:
    total = 0
    for path in Path(cache_dir).rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def purge_other_entries(cache_root: Path | str, keep_key: str) -> list[str]:
    """Drop cache entries other than ``keep_key``.

    Entries are only invalidated wholesale (dtype, attention, checkpoint, or
    format change), so keeping stale ones just consumes several GB each.
    """

    root = Path(cache_root).expanduser()
    if not root.is_dir():
        return []
    removed = []
    for child in root.iterdir():
        if child.name == keep_key:
            continue
        if not child.is_dir():
            continue
        shutil.rmtree(child, ignore_errors=True)
        removed.append(child.name)
    return removed
