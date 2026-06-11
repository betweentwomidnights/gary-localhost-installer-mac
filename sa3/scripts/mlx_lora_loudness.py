from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class LatentRMSNormalization:
    latents: np.ndarray
    gain: float
    pre_normalization_rms: float
    achieved_rms: float
    passes: int
    converged: bool


def valid_latent_length(
    actual_samples: int,
    padded_samples: int,
    latent_length: int,
) -> int:
    if padded_samples <= 0:
        return latent_length
    return max(
        1,
        min(
            latent_length,
            int(np.ceil(actual_samples * latent_length / padded_samples)),
        ),
    )


def latent_rms(
    latents: np.ndarray,
    *,
    actual_samples: int,
    padded_samples: int,
) -> float:
    latent_length = int(latents.shape[-1])
    valid_length = valid_latent_length(
        actual_samples,
        padded_samples,
        latent_length,
    )
    valid = np.asarray(latents[..., :valid_length], dtype=np.float32)
    return max(float(np.sqrt(np.mean(np.square(valid), dtype=np.float64))), 1e-6)


def normalize_audio_to_latent_rms(
    audio: np.ndarray,
    *,
    encode: Callable[[np.ndarray], np.ndarray],
    actual_samples: int,
    target_latent_rms: float,
    max_correction_rounds: int = 4,
    tolerance: float = 0.03,
) -> LatentRMSNormalization:
    if target_latent_rms <= 0:
        raise ValueError("target_latent_rms must be positive.")
    if max_correction_rounds < 0:
        raise ValueError("max_correction_rounds must be zero or greater.")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between 0 and 1.")

    gain = 1.0
    pre_normalization_rms = 0.0
    padded_samples = int(audio.shape[-1])

    for correction_round in range(max_correction_rounds + 1):
        pass_index = correction_round + 1
        latents = np.asarray(encode(audio * gain), dtype=np.float32)
        measured = latent_rms(
            latents,
            actual_samples=actual_samples,
            padded_samples=padded_samples,
        )
        if pass_index == 1:
            pre_normalization_rms = measured

        relative_error = abs(measured - target_latent_rms) / target_latent_rms
        converged = relative_error <= tolerance
        if converged or correction_round == max_correction_rounds:
            return LatentRMSNormalization(
                latents=latents,
                gain=gain,
                pre_normalization_rms=pre_normalization_rms,
                achieved_rms=measured,
                passes=pass_index,
                converged=converged,
            )

        gain *= target_latent_rms / measured

    raise RuntimeError("latent RMS normalization did not produce latents")
