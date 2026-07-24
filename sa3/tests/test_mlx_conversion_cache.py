"""Cache keying, atomicity, and zero-element tensor handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from stable_audio_3.mlx import conversion_cache as cache_mod


def _key(**overrides):
    kwargs = {
        "source_name": "medium",
        "resolved_config_path": None,
        "dit_dtype": "float16",
        "text_dtype": "float16",
        "number_dtype": "float16",
        "autoencoder_dtype": "float16",
        "attention": "sliding",
    }
    kwargs.update(overrides)
    return cache_mod.cache_key(**kwargs)


def test_cache_key_is_stable_for_identical_inputs():
    assert _key() == _key()
    assert _key().startswith("medium-")


@pytest.mark.parametrize(
    "override",
    [
        {"dit_dtype": "float32"},
        {"text_dtype": "float32"},
        {"number_dtype": "float32"},
        {"autoencoder_dtype": "float32"},
        # Attention changes autoencoder construction, so cached weights are
        # built for a different graph.
        {"attention": "full"},
        {"source_name": "medium-base"},
    ],
)
def test_cache_key_changes_when_conversion_inputs_change(override):
    assert _key(**override) != _key()


def test_cache_key_tracks_checkpoint_identity(tmp_path: Path):
    snapshot = tmp_path / "abc123"
    snapshot.mkdir()
    config = snapshot / "model_config.json"
    config.write_text("{}")

    before = _key(resolved_config_path=config)

    other = tmp_path / "def456"
    other.mkdir()
    other_config = other / "model_config.json"
    other_config.write_text("{}")

    # A different snapshot means different weights, so a different key.
    assert _key(resolved_config_path=other_config) != before


def _write_entry(directory: Path, *, format_version=None):
    directory.mkdir(parents=True, exist_ok=True)
    for name in (
        cache_mod.DIT_WEIGHTS,
        cache_mod.TEXT_WEIGHTS,
        cache_mod.NUMBER_WEIGHTS,
        cache_mod.AUTOENCODER_WEIGHTS,
    ):
        (directory / name).write_bytes(b"x")
    manifest = {"format_version": (
        cache_mod.CONVERSION_FORMAT_VERSION if format_version is None else format_version
    )}
    (directory / cache_mod.MANIFEST_NAME).write_text(json.dumps(manifest))


def test_is_complete_requires_every_file(tmp_path: Path):
    entry = tmp_path / "entry"
    _write_entry(entry)
    assert cache_mod.is_complete(entry)

    (entry / cache_mod.DIT_WEIGHTS).unlink()
    assert not cache_mod.is_complete(entry)


def test_is_complete_rejects_a_stale_format_version(tmp_path: Path):
    entry = tmp_path / "entry"
    _write_entry(entry, format_version=cache_mod.CONVERSION_FORMAT_VERSION - 1)
    assert not cache_mod.is_complete(entry)


def test_is_complete_is_false_for_missing_directory(tmp_path: Path):
    assert not cache_mod.is_complete(tmp_path / "nope")


def test_cache_writer_publishes_atomically(tmp_path: Path):
    with cache_mod.CacheWriter(tmp_path, "entry") as staging:
        _write_entry(staging)
        # Nothing is visible at the final location until the block exits.
        assert not (tmp_path / "entry").exists()
    assert cache_mod.is_complete(tmp_path / "entry")


def test_cache_writer_discards_staging_on_error(tmp_path: Path):
    with pytest.raises(RuntimeError, match="boom"):
        with cache_mod.CacheWriter(tmp_path, "entry") as staging:
            _write_entry(staging)
            raise RuntimeError("boom")
    assert not (tmp_path / "entry").exists()
    assert list(tmp_path.iterdir()) == []


def test_cache_writer_refuses_to_publish_an_incomplete_entry(tmp_path: Path):
    with pytest.raises(RuntimeError, match="incomplete"):
        with cache_mod.CacheWriter(tmp_path, "entry") as staging:
            (staging / cache_mod.DIT_WEIGHTS).write_bytes(b"x")
    assert not (tmp_path / "entry").exists()


def test_purge_other_entries_keeps_only_the_current_key(tmp_path: Path):
    for name in ("keep", "old-a", "old-b"):
        _write_entry(tmp_path / name)

    removed = cache_mod.purge_other_entries(tmp_path, "keep")

    assert sorted(removed) == ["old-a", "old-b"]
    assert cache_mod.is_complete(tmp_path / "keep")
    assert not (tmp_path / "old-a").exists()


def test_zero_element_tensors_round_trip_through_the_manifest(tmp_path: Path):
    """SA3 medium has a (1, 0, 1) bottleneck parameter MLX cannot serialize."""

    weights = {
        "dense": mx.ones((2, 3), dtype=mx.float16),
        "bottleneck.noise_scaling_factor": mx.ones((1, 0, 1), dtype=mx.float32),
    }

    dense, empty = cache_mod._split_empty_tensors(weights)

    assert set(dense) == {"dense"}
    assert empty == {
        "bottleneck.noise_scaling_factor": {
            "shape": [1, 0, 1],
            "dtype": "float32",
        }
    }

    # The serializable half survives a real save/load, and the empty tensor is
    # rebuilt so strict loading still sees a complete parameter set.
    path = tmp_path / "w.safetensors"
    mx.save_safetensors(str(path), dense)
    restored = cache_mod._restore_empty_tensors(dict(mx.load(str(path))), empty)

    assert set(restored) == set(weights)
    rebuilt = restored["bottleneck.noise_scaling_factor"]
    assert rebuilt.shape == (1, 0, 1)
    assert rebuilt.dtype == mx.float32
    assert int(rebuilt.size) == 0


def test_restore_tolerates_no_empty_tensors():
    assert cache_mod._restore_empty_tensors({"a": mx.ones((1,))}, None).keys() == {"a"}
