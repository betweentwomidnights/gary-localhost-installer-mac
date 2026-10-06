"""Mac-only preprocessing assets from the pinned decoder-free MLX artifact."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_preprocessing_artifact(variant: str | None) -> Path | None:
    """Resolve local assets only for the Mac wrapper's explicitly enabled path."""
    if variant is None or os.getenv("ACESTEP_TRAINING_USE_MLX_ARTIFACTS") != "1":
        return None
    from acestep.core.generation.handler.optimized_init import resolve_optimized_artifact

    canonical = variant if variant.startswith("acestep-v15-") else f"acestep-v15-{variant}"
    return resolve_optimized_artifact(
        variant=canonical, use_mlx_dit=True, resolved_device="cpu"
    )


def load_preprocessing_conditioner(checkpoint_dir, variant, device, precision):
    """Load only conditioning weights on Mac; preserve the full legacy fallback."""
    from acestep.training_v2.model_loader import _resolve_dtype, load_decoder_for_training

    artifact = resolve_preprocessing_artifact(variant)
    if artifact is None:
        return load_decoder_for_training(checkpoint_dir, variant, device, precision)
    from acestep.core.generation.handler.conditioning_shell import load_conditioning_shell

    canonical = variant if variant.startswith("acestep-v15-") else f"acestep-v15-{variant}"
    model = load_conditioning_shell(
        variant=canonical,
        config_root=artifact,
        weights_root=artifact / "conditioning",
        dtype=_resolve_dtype(precision),
    )
    model.requires_grad_(False)
    return model.to(device=device).eval()
