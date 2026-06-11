import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from mlx_lora_loudness import (  # noqa: E402
    latent_rms,
    normalize_audio_to_latent_rms,
    valid_latent_length,
)


def test_valid_latent_length_rounds_up_and_clamps() -> None:
    assert valid_latent_length(3, 8, 4) == 2
    assert valid_latent_length(99, 8, 4) == 4


def test_latent_rms_ignores_padded_region() -> None:
    latents = np.array([[[1.0, 1.0, 50.0, 50.0]]], dtype=np.float32)

    measured = latent_rms(
        latents,
        actual_samples=2,
        padded_samples=4,
    )

    assert measured == pytest.approx(1.0)


def test_iterative_normalization_hits_target() -> None:
    audio = np.ones((1, 4), dtype=np.float32)
    calls = []

    def encode(value: np.ndarray) -> np.ndarray:
        calls.append(value.copy())
        return value[None] * 0.5

    result = normalize_audio_to_latent_rms(
        audio,
        encode=encode,
        actual_samples=4,
        target_latent_rms=0.9,
    )

    assert len(calls) == 2
    assert result.passes == 2
    assert result.pre_normalization_rms == pytest.approx(0.5)
    assert result.achieved_rms == pytest.approx(0.9)
    assert result.gain == pytest.approx(1.8)
    assert result.converged


def test_normalization_stops_after_four_correction_rounds() -> None:
    audio = np.ones((1, 4), dtype=np.float32)

    def encode(value: np.ndarray) -> np.ndarray:
        gain = float(value[0, 0])
        return np.full((1, 1, 4), 0.5 * np.sqrt(gain), dtype=np.float32)

    result = normalize_audio_to_latent_rms(
        audio,
        encode=encode,
        actual_samples=4,
        target_latent_rms=0.9,
        max_correction_rounds=4,
        tolerance=1e-6,
    )

    assert result.passes == 5
    assert not result.converged
