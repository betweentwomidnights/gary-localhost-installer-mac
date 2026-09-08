"""Select pinned MLX decoder artifacts for precision-compatible training."""

from __future__ import annotations

from typing import Any

from loguru import logger

from .artifact_runtime import (
    ARTIFACTS,
    ARTIFACT_REVISION,
    load_decoder_artifact,
    optimized_load_requested,
    resolve_artifact_dir,
    weight_source_policy,
)

_TRAINING_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
}


def load_training_artifact_if_compatible(
    *,
    variant: str,
    training_dtype: str,
    decoder: Any,
    allow_storage_upcast: bool = False,
) -> tuple[Any, str] | None:
    """Load a compatible artifact, optionally allowing FP16 storage for BF16 math."""
    spec = ARTIFACTS.get(variant)
    requested_dtype = _TRAINING_DTYPES.get(training_dtype)
    storage_upcast = (
        allow_storage_upcast
        and requested_dtype == "bfloat16"
        and spec is not None
        and spec.dtype_name == "float16"
    )
    if (
        not optimized_load_requested()
        or spec is None
        or (requested_dtype != spec.dtype_name and not storage_upcast)
    ):
        return None
    try:
        artifact = resolve_artifact_dir(variant, local_files_only=True)
        null_condition = load_decoder_artifact(artifact, decoder)
    except Exception as exc:
        if weight_source_policy() == "optimized":
            raise RuntimeError(
                f"Pinned Apple MLX weights for {variant} are not downloaded or valid"
            ) from exc
        logger.warning(
            "Pinned MLX training artifact unavailable for {}; auto policy is using source checkpoint: {}",
            variant,
            exc,
        )
        return None
    logger.info(
        "Loaded MLX training decoder from pinned artifact {}@{}",
        variant,
        ARTIFACT_REVISION,
    )
    return null_condition, str(artifact)
