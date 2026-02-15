"""Optional wrapper for MLX EnCodec 48k checkpoint.

Not used by Stable Audio Open Small/1.0 baseline generation, which relies on
the SAO Oobleck/VAE pretransform.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _load_mlx_encodec_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    encodec_file = repo_root / "third_party" / "mlx-examples" / "encodec" / "encodec.py"
    if not encodec_file.exists():
        raise FileNotFoundError(
            f"Could not find MLX EnCodec implementation at {encodec_file}. "
            "Run ./scripts/pin_deps.sh first."
        )

    module_name = "saomlx_mlxexamples_encodec"
    spec = importlib.util.spec_from_file_location(module_name, str(encodec_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {encodec_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Encodec48k:
    def __init__(
        self,
        repo_id: str = "mlx-community/encodec-48khz-float32",
        bandwidth: float = 3.0,
    ):
        self.repo_id = repo_id
        self.bandwidth = bandwidth
        self._mx = None
        self._module: ModuleType | None = None
        self.model = None
        self.processor = None

    def load(self) -> None:
        if self.model is not None:
            return

        import mlx.core as mx

        self._mx = mx
        self._module = _load_mlx_encodec_module()
        self.model, self.processor = self._module.EncodecModel.from_pretrained(self.repo_id)

    @property
    def sampling_rate(self) -> int:
        self.load()
        return int(self.model.sampling_rate)

    @property
    def channels(self) -> int:
        self.load()
        return int(self.model.channels)

    def preprocess(self, audio: np.ndarray | Any):
        self.load()
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"Expected audio shape (samples,) or (samples, channels), got {arr.shape}")
        if arr.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {arr.shape[1]}")

        return self.processor(self._mx.array(arr))

    def encode(
        self,
        audio: np.ndarray | Any,
        *,
        bandwidth: float | None = None,
    ):
        self.load()
        feats, mask = self.preprocess(audio)
        codes, scales = self.model.encode(feats, mask, bandwidth=bandwidth or self.bandwidth)
        return codes, scales, mask

    def decode(self, codes, scales, mask):
        self.load()
        return self.model.decode(codes, scales, mask)

    def roundtrip(
        self,
        audio: np.ndarray | Any,
        *,
        bandwidth: float | None = None,
    ) -> np.ndarray:
        codes, scales, mask = self.encode(audio, bandwidth=bandwidth)
        recon = self.decode(codes, scales, mask)
        recon = np.asarray(recon[0], dtype=np.float32)
        source_len = np.asarray(audio).shape[0]
        return recon[:source_len]
