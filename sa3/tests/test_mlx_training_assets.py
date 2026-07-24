from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from mlx_training_assets import (  # noqa: E402
    TRAINING_MODEL_REPO,
    validate_medium_base_assets,
)


def _asset_pair(
    root: Path,
    *,
    repo_cache_name: str,
    arc: bool,
) -> tuple[Path, Path]:
    snapshot = (
        root
        / repo_cache_name
        / "snapshots"
        / "test-snapshot"
    )
    snapshot.mkdir(parents=True)
    config = snapshot / "model_config.json"
    checkpoint = snapshot / "model.safetensors"
    training = {"use_ema": True}
    if arc:
        training["arc"] = {"use_model_as_discriminator": True}
    config.write_text(json.dumps({"training": training}), encoding="utf-8")
    checkpoint.write_bytes(b"weights")
    return config, checkpoint


def test_accepts_medium_base_snapshot_assets(tmp_path: Path) -> None:
    config, checkpoint = _asset_pair(
        tmp_path,
        repo_cache_name="models--stabilityai--stable-audio-3-medium-base",
        arc=False,
    )

    assert validate_medium_base_assets(config, checkpoint) == (
        config.absolute(),
        checkpoint.absolute(),
    )


def test_rejects_inference_medium_snapshot_assets(tmp_path: Path) -> None:
    config, checkpoint = _asset_pair(
        tmp_path,
        repo_cache_name="models--stabilityai--stable-audio-3-medium",
        arc=True,
    )

    with pytest.raises(ValueError, match=TRAINING_MODEL_REPO):
        validate_medium_base_assets(config, checkpoint)


def test_rejects_arc_config_even_under_base_cache_path(
    tmp_path: Path,
) -> None:
    config, checkpoint = _asset_pair(
        tmp_path,
        repo_cache_name="models--stabilityai--stable-audio-3-medium-base",
        arc=True,
    )

    with pytest.raises(ValueError, match="ARC/distilled medium"):
        validate_medium_base_assets(config, checkpoint)


@pytest.mark.parametrize(
    "manifest_name",
    ("services.production.json", "services.dev.json"),
)
def test_inference_service_remains_on_medium(manifest_name: str) -> None:
    manifest_path = (
        REPOSITORY_ROOT
        / "control-center"
        / "manifest"
        / manifest_name
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sa3_service = next(
        service
        for service in manifest["services"]
        if service["id"] == "sa3"
    )

    assert sa3_service["environment"]["SA3_MODEL"] == "medium"
