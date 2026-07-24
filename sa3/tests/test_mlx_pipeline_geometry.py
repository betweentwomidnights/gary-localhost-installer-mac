"""Torch-free pipeline geometry and length math.

These cover the scalars that generation used to read off ``torch_pipeline``.
Values were verified against the live torch pipeline for SA3 medium: all four
geometry scalars, 27 ``_adapt_sample_size`` cases, and the sampling
distribution shift matched exactly.
"""

from __future__ import annotations

import pytest

from stable_audio_3.mlx.pipeline import (
    adapt_sample_size,
    model_geometry_from_config,
)
from stable_audio_3.mlx.sampling import (
    distribution_shift_spec_from_model_config,
    sampling_distribution_shift_spec_from_model_config,
)


def _medium_like_config(**overrides):
    config = {
        "sample_rate": 44100,
        "model": {
            "io_channels": 256,
            "diffusion": {
                "config": {
                    "cond_token_dim": 768,
                    "global_cond_dim": 768,
                    "local_add_cond_dim": 257,
                },
                "sampling_distribution_shift_options": None,
                "distribution_shift_options": {
                    "min_length": 256,
                    "max_length": 4096,
                },
            },
            "pretransform": {
                "config": {
                    "io_channels": 2,
                    "latent_dim": 256,
                    "downsampling_ratio": 4096,
                    "encoder": {"config": {"strides": [16]}},
                }
            },
        },
    }
    config.update(overrides)
    return config


def test_model_geometry_separates_latent_and_audio_channels():
    geometry = model_geometry_from_config(_medium_like_config())

    assert geometry == {
        "sample_rate": 44100,
        "downsampling_ratio": 4096,
        "latent_channels": 256,
        "audio_channels": 2,
    }
    # These are different values and must never be conflated: the DiT operates
    # on 256 latent channels while audio input preparation wants 2.
    assert geometry["latent_channels"] != geometry["audio_channels"]


def test_model_geometry_requires_sample_rate():
    config = _medium_like_config()
    del config["sample_rate"]
    with pytest.raises(KeyError, match="sample_rate"):
        model_geometry_from_config(config)


def _adapt(seconds, sample_size=44100 * 190, padding=6.0):
    config = _medium_like_config()
    conditioning = (
        [{"prompt": "x"}]
        if seconds is None
        else [{"prompt": "x", "seconds_total": seconds}]
    )
    return adapt_sample_size(
        config,
        conditioning,
        sample_size,
        padding,
        sample_rate=44100,
        downsampling_ratio=4096,
    )


def test_adapt_sample_size_falls_back_to_sample_size_without_seconds():
    assert _adapt(None) == 44100 * 190
    assert _adapt(0) == 44100 * 190


def test_adapt_sample_size_aligns_to_encoder_chunk_grid():
    # chunk_size defaults to 32 with stride 16, so latent_align is 2 and the
    # audio length must land on a 2 * 4096 = 8192 sample grid.
    for seconds in (1, 5.5, 23.777, 47, 95):
        assert _adapt(seconds) % 8192 == 0


def test_adapt_sample_size_clamps_to_sample_size():
    assert _adapt(10_000) == 44100 * 190


def test_adapt_sample_size_accounts_for_duration_padding():
    assert _adapt(30, padding=6.0) >= _adapt(30, padding=0.0)


def test_sampling_shift_does_not_fall_back_to_the_training_schedule():
    """The inference default is LogSNR, not the training 'full' shift.

    ``distribution_shift_spec_from_model_config`` falls through to
    ``distribution_shift_options``; the torch model does not. Using the wrong
    helper silently changes the inference sampling schedule.
    """

    config = _medium_like_config()

    sampling = sampling_distribution_shift_spec_from_model_config(config)
    assert sampling.kind == "logsnr"
    assert sampling.params == {
        "anchor_length": 2000,
        "anchor_logsnr": -6.2,
        "rate": 0,
        "logsnr_end": 2.0,
    }

    training = distribution_shift_spec_from_model_config(config)
    assert training.kind == "full"
    assert sampling != training


def test_sampling_shift_honors_explicit_options():
    config = _medium_like_config()
    config["model"]["diffusion"]["sampling_distribution_shift_options"] = {
        "min_length": 512,
        "max_length": 2048,
    }

    spec = sampling_distribution_shift_spec_from_model_config(config)
    assert spec.kind == "full"
    assert spec.params["min_length"] == 512
    assert spec.params["max_length"] == 2048
