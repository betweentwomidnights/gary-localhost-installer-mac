"""Small helpers for run stats and JSON logging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal environments
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TensorStats:
    shape: tuple[int, ...]
    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
        }


def as_numpy(value: Any) -> np.ndarray:
    """Best-effort conversion for torch/mlx/numpy arrays."""
    if np is None:
        raise RuntimeError("numpy is not available")

    if isinstance(value, np.ndarray):
        return value

    # torch tensors
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except Exception:
            pass

    try:
        return np.asarray(value)
    except Exception:
        # MLX bfloat16 arrays may not expose a direct PEP-3118 buffer.
        cast_attempts: list[Any] = [np.float32, "float32"]
        try:
            import mlx.core as mx  # type: ignore

            cast_attempts = [mx.float32, *cast_attempts]
        except Exception:
            pass

        if hasattr(value, "astype"):
            for cast_dtype in cast_attempts:
                try:
                    casted = value.astype(cast_dtype)
                    if hasattr(casted, "numpy"):
                        try:
                            return np.asarray(casted.numpy())
                        except Exception:
                            pass
                    return np.asarray(casted)
                except Exception:
                    continue
        raise


def tensor_stats(value: Any) -> TensorStats:
    if np is not None:
        arr = as_numpy(value).astype(np.float64, copy=False)
        return TensorStats(
            shape=tuple(arr.shape),
            mean=float(arr.mean()),
            std=float(arr.std()),
            min=float(arr.min()),
            max=float(arr.max()),
        )

    flat = _flatten_to_floats(value)
    if not flat:
        raise ValueError("Cannot compute stats for an empty value")
    mean = sum(flat) / len(flat)
    var = sum((v - mean) ** 2 for v in flat) / len(flat)
    return TensorStats(
        shape=_shape(value),
        mean=float(mean),
        std=float(var**0.5),
        min=float(min(flat)),
        max=float(max(flat)),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        try:
            return tuple(int(v) for v in value.shape)
        except Exception:
            pass

    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        return (len(value),) + _shape(value[0])
    return ()


def _flatten_to_floats(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(_flatten_to_floats(item))
        return out
    return [float(value)]
