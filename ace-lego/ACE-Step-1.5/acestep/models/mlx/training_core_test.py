"""Tests for ACE MLX flow-matching training primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("mlx.core")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from acestep.models.mlx import training_core as ace_training  # noqa: E402


def test_flow_min_snr_weights_match_pc_reference_cases() -> None:
    low_snr = ace_training.flow_min_snr_weights(
        mx.array([0.5, 0.9], dtype=mx.float32),
        gamma=5.0,
    )
    high_snr = ace_training.flow_min_snr_weights(
        mx.array([0.1], dtype=mx.float32),
        gamma=5.0,
    )
    extreme = ace_training.flow_min_snr_weights(
        mx.array([0.0, 1.0], dtype=mx.float32),
        gamma=5.0,
    )

    assert mx.allclose(low_snr, mx.ones_like(low_snr))
    assert mx.allclose(high_snr, mx.array([5.0 / 81.0], dtype=mx.float32))
    assert bool(mx.all(mx.isfinite(extreme)))
    assert bool(mx.all((extreme >= 0.0) & (extreme <= 1.0)))
    with pytest.raises(ValueError, match="greater than zero"):
        ace_training.flow_min_snr_weights(mx.array([0.5]), gamma=0.0)


def test_sample_timesteps_uses_t_as_r_when_meanflow_disabled() -> None:
    mx.random.seed(3)
    t, r = ace_training.sample_timesteps(
        16,
        dtype=mx.float32,
        timestep_mu=-0.4,
        timestep_sigma=1.0,
        use_meanflow=False,
    )

    assert t.shape == (16,)
    assert bool(mx.all((t > 0.0) & (t < 1.0)))
    assert mx.allclose(t, r)


def test_sample_timesteps_orders_t_and_r_when_meanflow_enabled() -> None:
    mx.random.seed(4)
    t, r = ace_training.sample_timesteps(
        16,
        dtype=mx.float32,
        data_proportion=0.0,
        use_meanflow=True,
    )

    assert bool(mx.all(t >= r))
    assert not bool(mx.allclose(t, r))


def test_cfg_dropout_extremes_keep_or_replace_conditions() -> None:
    encoder_hidden_states = mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)
    null_condition_emb = mx.full((1, 3, 4), -1.0, dtype=mx.float32)

    kept = ace_training.apply_cfg_dropout(
        encoder_hidden_states,
        null_condition_emb,
        cfg_ratio=0.0,
    )
    dropped = ace_training.apply_cfg_dropout(
        encoder_hidden_states,
        null_condition_emb,
        cfg_ratio=1.0,
    )

    assert mx.array_equal(kept, encoder_hidden_states)
    assert mx.array_equal(
        dropped,
        mx.broadcast_to(null_condition_emb, encoder_hidden_states.shape),
    )


class _FakeDecoder:
    def __init__(self, prediction):
        self.prediction = prediction
        self.last_hidden_states = None
        self.last_timestep = None
        self.last_timestep_r = None
        self.last_encoder_attention_mask = None

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        cache=None,
        use_cache=False,
    ):
        self.last_hidden_states = hidden_states
        self.last_timestep = timestep
        self.last_timestep_r = timestep_r
        self.last_encoder_attention_mask = encoder_attention_mask
        return self.prediction, None


def test_ace_flow_matching_loss_uses_xt_and_flow_target() -> None:
    target_latents = mx.array(
        [
            [[1.0, -1.0], [0.5, 0.25]],
            [[-0.5, 1.5], [1.0, -2.0]],
        ],
        dtype=mx.float32,
    )
    noise = mx.array(
        [
            [[0.0, 1.0], [1.5, -0.25]],
            [[0.5, -1.5], [0.0, 2.0]],
        ],
        dtype=mx.float32,
    )
    timesteps = mx.array([0.25, 0.75], dtype=mx.float32)
    flow = noise - target_latents
    decoder = _FakeDecoder(prediction=mx.zeros_like(flow))
    batch = {
        "target_latents": target_latents,
        "encoder_hidden_states": mx.zeros((2, 1, 3), dtype=mx.float32),
        "encoder_attention_mask": mx.array([[1.0], [1.0]], dtype=mx.float32),
        "context_latents": mx.zeros((2, 2, 2), dtype=mx.float32),
    }

    loss = ace_training.ace_flow_matching_loss(
        decoder,
        batch,
        config=ace_training.ACEFlowMatchingConfig(
            cfg_ratio=0.0,
            loss_weighting="none",
        ),
        noise=noise,
        timesteps=timesteps,
    )
    expected_noisy = timesteps[:, None, None] * noise + (
        1.0 - timesteps[:, None, None]
    ) * target_latents
    expected_loss = mx.mean(flow**2)

    assert mx.allclose(decoder.last_hidden_states, expected_noisy)
    assert mx.allclose(decoder.last_timestep, timesteps)
    assert mx.allclose(decoder.last_timestep_r, timesteps)
    assert mx.array_equal(decoder.last_encoder_attention_mask, batch["encoder_attention_mask"])
    assert mx.allclose(loss, expected_loss)


def test_ace_flow_matching_loss_applies_min_snr_per_sample() -> None:
    target_latents = mx.zeros((2, 2, 2), dtype=mx.float32)
    noise = mx.ones_like(target_latents)
    prediction = mx.zeros_like(target_latents)
    timesteps = mx.array([0.1, 0.9], dtype=mx.float32)
    decoder = _FakeDecoder(prediction=prediction)
    batch = {
        "target_latents": target_latents,
        "encoder_hidden_states": mx.zeros((2, 1, 3), dtype=mx.float32),
        "context_latents": mx.zeros((2, 2, 2), dtype=mx.float32),
    }

    loss = ace_training.ace_flow_matching_loss(
        decoder,
        batch,
        config=ace_training.ACEFlowMatchingConfig(
            cfg_ratio=0.0,
            loss_weighting="min_snr",
            snr_gamma=5.0,
        ),
        noise=noise,
        timesteps=timesteps,
    )
    weights = ace_training.flow_min_snr_weights(timesteps, gamma=5.0)

    assert mx.allclose(loss, mx.mean(weights))


class _TrainableConstantDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = mx.array([0.0], dtype=mx.float32)

    def __call__(
        self,
        *,
        hidden_states,
        timestep,
        timestep_r,
        encoder_hidden_states,
        encoder_attention_mask=None,
        context_latents=None,
        cache=None,
        use_cache=False,
    ):
        return mx.ones_like(hidden_states) * self.bias, None


def test_ace_adamw_update_step_reduces_flow_loss() -> None:
    model = _TrainableConstantDecoder()
    optimizer = ace_training.create_adamw_optimizer(
        ace_training.ACEAdamWConfig(learning_rate=0.2, weight_decay=0.0)
    )
    target_latents = mx.zeros((1, 2, 2), dtype=mx.float32)
    noise = mx.ones_like(target_latents)
    timesteps = mx.array([0.5], dtype=mx.float32)
    batch = {
        "target_latents": target_latents,
        "encoder_hidden_states": mx.zeros((1, 1, 3), dtype=mx.float32),
        "context_latents": mx.zeros((1, 2, 2), dtype=mx.float32),
    }
    config = ace_training.ACEFlowMatchingConfig(
        cfg_ratio=0.0,
        loss_weighting="none",
    )

    initial_loss = ace_training.ace_flow_matching_loss(
        model,
        batch,
        config=config,
        noise=noise,
        timesteps=timesteps,
    )
    for _ in range(20):
        loss = ace_training.ace_adamw_update_step(
            model,
            optimizer,
            batch,
            config=config,
            noise=noise,
            timesteps=timesteps,
        )
    final_loss = ace_training.ace_flow_matching_loss(
        model,
        batch,
        config=config,
        noise=noise,
        timesteps=timesteps,
    )

    assert float(loss) < float(initial_loss)
    assert float(final_loss) < float(initial_loss) * 0.1
