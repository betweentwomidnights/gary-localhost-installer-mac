"""Guarded initialization helpers for the pinned native MLX artifact path."""

from pathlib import Path
from typing import Any

import torch
from loguru import logger


def resolve_optimized_artifact(
    *, variant: str, use_mlx_dit: bool, resolved_device: str
) -> Path | None:
    """Resolve a complete pinned artifact when the opt-in runtime flag is set."""
    if not use_mlx_dit or resolved_device not in ("mps", "cpu"):
        return None
    from acestep.models.mlx.artifact_runtime import (
        optimized_load_requested,
        resolve_artifact_dir,
    )

    if not optimized_load_requested():
        return None
    try:
        return resolve_artifact_dir(variant)
    except Exception as exc:
        logger.warning(
            "[initialize_service] Optimized MLX artifact unavailable; "
            "using full PyTorch fallback: {}",
            exc,
        )
        return None


def load_optimized_conditioning(
    host: Any,
    *,
    variant: str,
    artifact_path: Path,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Load the decoder-free shell and silence latent from one verified artifact."""
    from .conditioning_shell import load_conditioning_shell

    conditioning = artifact_path / "conditioning"
    host.model = load_conditioning_shell(
        variant=variant,
        config_root=artifact_path,
        weights_root=conditioning,
        dtype=dtype,
    )
    host.config = host.model.config
    host.silence_latent = (
        torch.load(conditioning / "silence_latent.pt", weights_only=True)
        .transpose(1, 2)
        .to(device)
        .to(dtype)
    )
    host._conditioning_shell_active = True
    logger.info(
        "[initialize_service] Loaded decoder-free conditioning shell from {}",
        conditioning,
    )
