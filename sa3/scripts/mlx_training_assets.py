from __future__ import annotations

import json
from pathlib import Path


TRAINING_MODEL_NAME = "medium-base"
TRAINING_MODEL_REPO = "stabilityai/stable-audio-3-medium-base"
TRAINING_CONFIG_FILENAME = "model_config.json"
TRAINING_CHECKPOINT_FILENAME = "model.safetensors"
OPTIMIZED_MODEL_REPO = "stabilityai/stable-audio-3-optimized"
OPTIMIZED_MEDIUM_BASE_FILENAME = "MLX/dit_medium-base_f16.npz"
# Pinned so an upstream re-upload cannot change training weights silently.
# Verified byte-identical to Gary's runtime PyTorch->MLX conversion: all 522 DiT
# tensors match with max abs diff 0.0. Bump deliberately, and re-run that
# comparison before trusting a new revision.
OPTIMIZED_MODEL_REVISION = "08c64b96b1e59942aade69759f60fb88c58c90c4"
_EXPECTED_CACHE_COMPONENT = (
    "models--stabilityai--stable-audio-3-medium-base"
)


def validate_medium_base_assets(
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[Path, Path]:
    """Fail closed unless both assets are from the SA3 medium-base snapshot."""
    config = Path(config_path).expanduser().absolute()
    checkpoint = Path(checkpoint_path).expanduser().absolute()
    if not config.is_file():
        raise FileNotFoundError(f"Medium-base model config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Medium-base model checkpoint not found: {checkpoint}"
        )
    if config.name != TRAINING_CONFIG_FILENAME:
        raise ValueError(
            "SA3 training requires medium-base model_config.json; "
            f"received {config.name}."
        )
    if checkpoint.name != TRAINING_CHECKPOINT_FILENAME:
        raise ValueError(
            "SA3 training requires medium-base model.safetensors; "
            f"received {checkpoint.name}."
        )
    if config.parent != checkpoint.parent:
        raise ValueError(
            "SA3 medium-base config and checkpoint must come from the same "
            "Hugging Face snapshot."
        )
    if _EXPECTED_CACHE_COMPONENT not in config.parts:
        raise ValueError(
            "Refusing to train SA3 from unverified model assets. Training must "
            f"use {TRAINING_MODEL_REPO}, never the inference-time medium model."
        )

    with config.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    training = payload.get("training")
    if not isinstance(training, dict) or "arc" in training:
        raise ValueError(
            "Refusing to train SA3 from ARC/distilled medium configuration. "
            f"Training must use {TRAINING_MODEL_REPO}."
        )
    return config, checkpoint


def resolve_medium_base_assets() -> tuple[Path, Path]:
    """Download or reuse the one supported base model for Gary SA3 training."""
    from stable_audio_3.model_configs import models

    config_path, checkpoint_path = models[TRAINING_MODEL_NAME].resolve()
    return validate_medium_base_assets(config_path, checkpoint_path)


def resolve_hosted_medium_base_npz() -> Path:
    """Download or reuse the official FP16 MLX medium-base DiT."""

    from huggingface_hub import hf_hub_download

    resolved = Path(
        hf_hub_download(
            repo_id=OPTIMIZED_MODEL_REPO,
            filename=OPTIMIZED_MEDIUM_BASE_FILENAME,
            revision=OPTIMIZED_MODEL_REVISION,
        )
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Hosted medium-base MLX checkpoint not found: {resolved}"
        )
    return resolved
