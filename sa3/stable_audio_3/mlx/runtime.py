from __future__ import annotations


class MLXRuntimeUnavailableError(RuntimeError):
    """Raised when MLX-backed code is requested without MLX installed."""


def import_mlx_core(*, required: bool = False):
    try:
        import mlx.core as mx  # type: ignore
    except ImportError as exc:
        if required:
            raise MLXRuntimeUnavailableError(
                "MLX is not installed in this environment. "
                "Install the Apple Silicon MLX runtime before attempting MLX inference."
            ) from exc
        return None
    return mx


def import_mlx_nn(*, required: bool = False):
    try:
        import mlx.nn as nn  # type: ignore
    except ImportError as exc:
        if required:
            raise MLXRuntimeUnavailableError(
                "MLX is not installed in this environment. "
                "Install the Apple Silicon MLX runtime before attempting MLX inference."
            ) from exc
        return None
    return nn


def mlx_runtime_available() -> bool:
    return import_mlx_core(required=False) is not None


def require_mlx_runtime() -> None:
    import_mlx_core(required=True)
