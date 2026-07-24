from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

from stable_audio_3.mlx.conditioning import (
    assemble_conditioning_inputs_from_tensors,
)


def _inpaint_model_config() -> dict:
    return {
        "model_type": "diffusion_cond",
        "model": {
            "diffusion": {
                "type": "dit",
                "diffusion_objective": "rectified_flow",
                "local_add_cond_ids": [
                    "inpaint_mask",
                    "inpaint_masked_input",
                ],
                "config": {
                    "io_channels": 256,
                    "local_add_cond_dim": 257,
                },
            },
            "conditioning": {"configs": []},
            "pretransform": {
                "config": {
                    "io_channels": 2,
                    "latent_dim": 256,
                    "downsampling_ratio": 4096,
                    "encoder": {"config": {}},
                    "decoder": {"config": {}},
                }
            },
        },
    }


def test_inference_default_local_conditioning_remains_all_zeros() -> None:
    inputs = assemble_conditioning_inputs_from_tensors(
        _inpaint_model_config(),
        {"placeholder": mx.zeros((2, 1, 4), dtype=mx.float16)},
        latent_length=6,
        dtype_name="float16",
    )

    local_add = np.asarray(inputs["local_add_cond"])
    assert local_add.shape == (2, 257, 6)
    np.testing.assert_array_equal(local_add, np.zeros_like(local_add))


def test_full_generation_training_local_conditioning_uses_zero_mask() -> None:
    inputs = assemble_conditioning_inputs_from_tensors(
        _inpaint_model_config(),
        {"placeholder": mx.zeros((2, 1, 4), dtype=mx.float16)},
        latent_length=6,
        dtype_name="float16",
        default_inpaint_mode="training",
    )

    local_add = np.asarray(inputs["local_add_cond"])
    assert local_add.shape == (2, 257, 6)
    np.testing.assert_array_equal(local_add, np.zeros_like(local_add))
