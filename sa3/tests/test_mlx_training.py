from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from stable_audio_3.models.lora import load_lora_checkpoint
from stable_audio_3.mlx.lora import MLXLoRASet
from stable_audio_3.mlx.sampling import (
    make_distribution_shift_spec,
    shift_timestep_values,
    training_distribution_shift_spec_from_model_config,
)
from stable_audio_3.mlx.training import (
    inject_trainable_lora,
    sample_training_timesteps,
    save_trainable_lora,
)


class TinyMLXRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(3, 4, bias=False)
        self.output = nn.Linear(4, 2, bias=False)
        self.output.weight = mx.zeros_like(self.output.weight)

    def __call__(self, x):
        return self.output(nn.silu(self.input(x)))


class TinyMLXLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2, bias=False)
        self.layer.weight = mx.array(
            [[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]],
            dtype=mx.float32,
        )

    def __call__(self, x):
        return self.layer(x)


class TinyMLXConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv1d(2, 3, kernel_size=3, padding=1)
        self.layer.weight = mx.arange(18, dtype=mx.float32).reshape(3, 3, 2) / 20
        self.layer.bias = mx.array([0.1, -0.2, 0.3], dtype=mx.float32)

    def __call__(self, x):
        return self.layer(x)


def test_truncated_logit_normal_training_sampler_matches_upstream_shape():
    values = sample_training_timesteps(
        "trunc_logit_normal",
        20_000,
        rng=np.random.default_rng(17),
    )

    assert values.dtype == np.float32
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert 0.52 < float(values.mean()) < 0.55
    assert 0.52 < float(np.median(values)) < 0.56


def test_model_training_shift_uses_full_distribution_defaults():
    model_config = {
        "model": {
            "diffusion": {
                "distribution_shift_options": {
                    "min_length": 256,
                    "max_length": 4096,
                }
            }
        }
    }
    spec = training_distribution_shift_spec_from_model_config(model_config)

    assert spec == make_distribution_shift_spec(
        "full",
        min_length=256,
        max_length=4096,
    )
    shifted = shift_timestep_values(
        [0.5],
        dist_shift=spec,
        effective_seq_len=507,
    )
    assert shifted[0] == pytest.approx(0.6323907626)


def test_training_shift_can_be_disabled():
    assert shift_timestep_values(
        [0.25, 0.75],
        dist_shift=None,
        effective_seq_len=[256, 4096],
    ) == (0.25, 0.75)


def test_trainable_lora_updates_adapters_without_updating_base_weights():
    mx.random.seed(7)
    model = TinyMLXRegressor()
    report = inject_trainable_lora(model, rank=2, alpha=2, include=["output"])
    base_before = mx.array(model.output.base.weight)

    x = mx.array([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]], dtype=mx.float32)
    target = mx.array([[0.5, -1.0], [-0.25, 0.75]], dtype=mx.float32)

    def loss_fn(local_model, inputs, expected):
        return mx.mean((local_model(inputs) - expected) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.1)
    initial_loss = float(loss_fn(model, x, target))
    for _ in range(40):
        loss, grads = loss_and_grad(model, x, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    final_loss = float(loss_fn(model, x, target))

    assert report.layer_names == ("output",)
    assert report.trainable_parameters == 12
    assert [name for name, _ in tree_flatten(model.trainable_parameters())] == [
        "output.lora_A",
        "output.lora_B",
    ]
    assert mx.array_equal(model.output.base.weight, base_before)
    assert final_loss < initial_loss * 0.1


def test_trainable_dora_rows_updates_adapters_without_updating_base_weights():
    mx.random.seed(11)
    model = TinyMLXLinear()
    report = inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="dora",
    )
    base_before = mx.array(model.layer.base.weight)

    x = mx.array([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]], dtype=mx.float32)
    target = mx.zeros((2, 2), dtype=mx.float32)

    def loss_fn(local_model, inputs, expected):
        return mx.mean((local_model(inputs) - expected) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.05, weight_decay=0.0)
    initial_loss = float(loss_fn(model, x, target))
    for _ in range(40):
        loss, grads = loss_and_grad(model, x, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    final_loss = float(loss_fn(model, x, target))

    assert report.adapter_type == "dora-rows"
    assert report.trainable_parameters == 7
    assert [name for name, _ in tree_flatten(model.trainable_parameters())] == [
        "layer.lora_A",
        "layer.lora_B",
        "layer.magnitude",
    ]
    assert mx.array_equal(model.layer.base.weight, base_before)
    assert final_loss < initial_loss * 0.1


def test_trainable_dora_rows_conv1d_matches_full_adapted_weight():
    model = TinyMLXConv1d()
    report = inject_trainable_lora(model, rank=1, alpha=1, adapter_type="dora")
    x = mx.arange(16, dtype=mx.float32).reshape(1, 8, 2) / 10

    assert mx.allclose(model(x), model.layer.base(x), atol=1e-6)

    model.layer.lora_A = mx.array(
        [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]],
        dtype=mx.float32,
    )
    model.layer.lora_B = mx.array([[0.5], [-0.25], [0.75]], dtype=mx.float32)
    model.layer.magnitude = mx.array([1.0, 1.5, 2.0], dtype=mx.float32)

    base_source = model.layer.base.weight.transpose(0, 2, 1).reshape(3, 6)
    v = base_source + model.layer.lora_B @ model.layer.lora_A
    adapted_source = v / mx.maximum(
        mx.sqrt(mx.sum(v**2, axis=1, keepdims=True)),
        1e-12,
    )
    adapted_source *= model.layer.magnitude[:, None]
    adapted_weight = adapted_source.reshape(3, 2, 3).transpose(0, 2, 1)
    expected = mx.conv1d(x, adapted_weight, stride=1, padding=1)
    expected += model.layer.base.bias

    assert report.trainable_parameters == 12
    assert mx.allclose(model(x), expected, atol=1e-6)


def test_trainable_bora_updates_adapters_without_updating_base_weights():
    mx.random.seed(13)
    model = TinyMLXLinear()
    report = inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="bora",
    )
    base_before = mx.array(model.layer.base.weight)

    x = mx.array([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]], dtype=mx.float32)
    target = mx.zeros((2, 2), dtype=mx.float32)

    def loss_fn(local_model, inputs, expected):
        return mx.mean((local_model(inputs) - expected) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.05, weight_decay=0.0)
    initial_loss = float(loss_fn(model, x, target))
    for _ in range(40):
        loss, grads = loss_and_grad(model, x, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    final_loss = float(loss_fn(model, x, target))

    assert report.adapter_type == "bora"
    assert report.trainable_parameters == 10
    assert sorted(name for name, _ in tree_flatten(model.trainable_parameters())) == [
        "layer.lora_A",
        "layer.lora_B",
        "layer.magnitude_c",
        "layer.magnitude_r",
    ]
    assert mx.array_equal(model.layer.base.weight, base_before)
    assert final_loss < initial_loss * 0.1


def test_trainable_bora_conv1d_matches_full_adapted_weight():
    model = TinyMLXConv1d()
    report = inject_trainable_lora(model, rank=1, alpha=1, adapter_type="bora")
    x = mx.arange(16, dtype=mx.float32).reshape(1, 8, 2) / 10

    assert mx.allclose(model(x), model.layer.base(x), atol=1e-6)

    model.layer.lora_A = mx.array(
        [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]],
        dtype=mx.float32,
    )
    model.layer.lora_B = mx.array([[0.5], [-0.25], [0.75]], dtype=mx.float32)
    model.layer.magnitude_r = mx.array([1.0, 1.5, 2.0], dtype=mx.float32)
    model.layer.magnitude_c = mx.array(
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        dtype=mx.float32,
    )

    base_source = model.layer.base.weight.transpose(0, 2, 1).reshape(3, 6)
    v = base_source + model.layer.lora_B @ model.layer.lora_A
    row_scaled = v / mx.maximum(
        mx.sqrt(mx.sum(v**2, axis=1, keepdims=True)),
        1e-12,
    )
    row_scaled *= model.layer.magnitude_r[:, None]
    adapted_source = row_scaled / mx.maximum(
        mx.sqrt(mx.sum(row_scaled**2, axis=0, keepdims=True)),
        1e-12,
    )
    adapted_source *= model.layer.magnitude_c[None, :]
    adapted_weight = adapted_source.reshape(3, 2, 3).transpose(0, 2, 1)
    expected = mx.conv1d(x, adapted_weight, stride=1, padding=1)
    expected += model.layer.base.bias

    assert report.trainable_parameters == 18
    assert mx.allclose(model(x), expected, atol=1e-6)


def test_trainable_lora_xs_updates_core_without_updating_base_or_bases():
    model = TinyMLXLinear()
    report = inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="lora-xs",
    )
    base_before = mx.array(model.layer.base.weight)
    u_before = mx.array(model.layer.U)
    v_before = mx.array(model.layer.V)

    x = mx.array([[1.0, -2.0, 0.5], [-1.0, 0.5, 2.0]], dtype=mx.float32)
    desired_core = mx.array([[0.75]], dtype=mx.float32)
    desired_weight = base_before + model.layer.U @ desired_core @ model.layer.V.T
    target = x @ desired_weight.T

    def loss_fn(local_model, inputs, expected):
        return mx.mean((local_model(inputs) - expected) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=0.05, weight_decay=0.0)
    initial_loss = float(loss_fn(model, x, target))
    for _ in range(80):
        loss, grads = loss_and_grad(model, x, target)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
    final_loss = float(loss_fn(model, x, target))

    assert report.adapter_type == "lora-xs"
    assert report.trainable_parameters == 1
    assert [name for name, _ in tree_flatten(model.trainable_parameters())] == [
        "layer.M_xs",
    ]
    assert mx.array_equal(model.layer.base.weight, base_before)
    assert mx.array_equal(model.layer.U, u_before)
    assert mx.array_equal(model.layer.V, v_before)
    assert final_loss < initial_loss * 0.001


def test_trainable_lora_xs_conv1d_matches_full_adapted_weight():
    model = TinyMLXConv1d()
    report = inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type="lora-xs",
    )
    x = mx.arange(16, dtype=mx.float32).reshape(1, 8, 2) / 10

    assert mx.allclose(model(x), model.layer.base(x), atol=1e-6)

    model.layer.M_xs = mx.array([[0.75]], dtype=mx.float32)
    base_source = model.layer.base.weight.transpose(0, 2, 1).reshape(3, 6)
    adapted_source = base_source + model.layer.U @ model.layer.M_xs @ model.layer.V.T
    adapted_weight = adapted_source.reshape(3, 2, 3).transpose(0, 2, 1)
    expected = mx.conv1d(x, adapted_weight, stride=1, padding=1)
    expected += model.layer.base.bias

    assert report.trainable_parameters == 1
    assert mx.allclose(model(x), expected, atol=1e-6)


def test_saved_mlx_training_checkpoint_loads_through_existing_lora_runtime(
    tmp_path: Path,
):
    model = TinyMLXLinear()
    inject_trainable_lora(model, rank=1, alpha=1)
    model.layer.lora_A = mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32)
    model.layer.lora_B = mx.array([[0.5], [-0.25]], dtype=mx.float32)

    checkpoint = save_trainable_lora(
        model,
        tmp_path / "tiny.safetensors",
        rank=1,
        alpha=1,
        extra_metadata={"step": 12, "base_model": "sa3-medium"},
    )
    state_dict, config = load_lora_checkpoint(checkpoint)

    fresh = TinyMLXLinear()
    lora_set = MLXLoRASet.from_checkpoints([checkpoint], fresh, target_label="tiny")
    report = lora_set.apply_to(fresh)

    assert sorted(state_dict) == [
        "layer.parametrizations.weight.0.lora_A",
        "layer.parametrizations.weight.0.lora_B",
    ]
    assert config["adapter_type"] == "lora"
    assert config["step"] == 12
    assert report.applied_layers == 1
    expected = model.layer.base.weight + model.layer.lora_B @ model.layer.lora_A
    assert mx.allclose(fresh.layer.weight, expected)


def test_saved_mlx_dora_checkpoint_loads_through_existing_lora_runtime(
    tmp_path: Path,
):
    model = TinyMLXLinear()
    inject_trainable_lora(model, rank=1, alpha=1, adapter_type="dora")
    model.layer.lora_A = mx.array([[0.25, -0.5, 1.0]], dtype=mx.float32)
    model.layer.lora_B = mx.array([[0.5], [-0.25]], dtype=mx.float32)
    model.layer.magnitude = mx.array([3.0, 2.0], dtype=mx.float32)

    checkpoint = save_trainable_lora(
        model,
        tmp_path / "tiny-dora.safetensors",
        rank=1,
        alpha=1,
        extra_metadata={"step": 24, "base_model": "sa3-medium"},
    )
    state_dict, config = load_lora_checkpoint(checkpoint)

    fresh = TinyMLXLinear()
    lora_set = MLXLoRASet.from_checkpoints(
        [checkpoint],
        fresh,
        target_label="tiny-dora",
    )
    report = lora_set.apply_to(fresh)

    assert sorted(state_dict) == [
        "layer.parametrizations.weight.0.lora_A",
        "layer.parametrizations.weight.0.lora_B",
        "layer.parametrizations.weight.0.magnitude",
    ]
    assert config["adapter_type"] == "dora-rows"
    assert config["step"] == 24
    assert report.applied_layers == 1

    base = np.asarray(model.layer.base.weight, dtype=np.float32)
    delta = np.asarray(model.layer.lora_B @ model.layer.lora_A, dtype=np.float32)
    v = base + delta
    expected = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    expected *= np.asarray(model.layer.magnitude, dtype=np.float32)[:, None]
    assert mx.allclose(fresh.layer.weight, mx.array(expected), atol=1e-3)


def test_saved_mlx_bora_checkpoint_loads_through_existing_lora_runtime(
    tmp_path: Path,
):
    model = TinyMLXLinear()
    inject_trainable_lora(model, rank=1, alpha=1, adapter_type="bora")
    model.layer.lora_A = mx.array([[0.25, -0.5, 1.0]], dtype=mx.float32)
    model.layer.lora_B = mx.array([[0.5], [-0.25]], dtype=mx.float32)
    model.layer.magnitude_r = mx.array([3.0, 2.0], dtype=mx.float32)
    model.layer.magnitude_c = mx.array([1.5, 2.5, 3.5], dtype=mx.float32)

    checkpoint = save_trainable_lora(
        model,
        tmp_path / "tiny-bora.safetensors",
        rank=1,
        alpha=1,
        extra_metadata={"step": 36, "base_model": "sa3-medium"},
    )
    state_dict, config = load_lora_checkpoint(checkpoint)

    fresh = TinyMLXLinear()
    lora_set = MLXLoRASet.from_checkpoints(
        [checkpoint],
        fresh,
        target_label="tiny-bora",
    )
    report = lora_set.apply_to(fresh)

    assert sorted(state_dict) == [
        "layer.parametrizations.weight.0.lora_A",
        "layer.parametrizations.weight.0.lora_B",
        "layer.parametrizations.weight.0.magnitude_c",
        "layer.parametrizations.weight.0.magnitude_r",
    ]
    assert config["adapter_type"] == "bora"
    assert config["step"] == 36
    assert report.applied_layers == 1

    base = np.asarray(model.layer.base.weight, dtype=np.float32)
    delta = np.asarray(model.layer.lora_B @ model.layer.lora_A, dtype=np.float32)
    v = base + delta
    row_scaled = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    row_scaled *= np.asarray(model.layer.magnitude_r, dtype=np.float32)[:, None]
    expected = row_scaled / np.maximum(
        np.linalg.norm(row_scaled, axis=0, keepdims=True),
        1e-12,
    )
    expected *= np.asarray(model.layer.magnitude_c, dtype=np.float32)[None, :]
    assert mx.allclose(fresh.layer.weight, mx.array(expected), atol=1e-3)


@pytest.mark.parametrize(
    ("adapter_type", "expected_suffixes"),
    [
        ("lora-xs", ["M_xs"]),
        ("dora-rows-xs", ["M_xs", "magnitude"]),
        ("dora-cols-xs", ["M_xs", "magnitude"]),
        ("bora-xs", ["M_xs", "magnitude_c", "magnitude_r"]),
    ],
)
def test_saved_mlx_xs_checkpoints_load_through_existing_lora_runtime(
    tmp_path: Path,
    adapter_type: str,
    expected_suffixes: list[str],
):
    model = TinyMLXLinear()
    inject_trainable_lora(
        model,
        rank=1,
        alpha=1,
        adapter_type=adapter_type,
    )
    model.layer.M_xs = mx.array([[0.5]], dtype=mx.float32)
    if adapter_type == "dora-rows-xs":
        model.layer.magnitude = mx.array([3.0, 2.0], dtype=mx.float32)
    elif adapter_type == "dora-cols-xs":
        model.layer.magnitude = mx.array([1.5, 2.5, 3.5], dtype=mx.float32)
    elif adapter_type == "bora-xs":
        model.layer.magnitude_r = mx.array([3.0, 2.0], dtype=mx.float32)
        model.layer.magnitude_c = mx.array([1.5, 2.5, 3.5], dtype=mx.float32)

    checkpoint = save_trainable_lora(
        model,
        tmp_path / f"tiny-{adapter_type}.safetensors",
        rank=1,
        alpha=1,
        extra_metadata={"step": 48, "base_model": "sa3-medium"},
    )
    state_dict, config = load_lora_checkpoint(checkpoint)

    fresh = TinyMLXLinear()
    lora_set = MLXLoRASet.from_checkpoints(
        [checkpoint],
        fresh,
        target_label=f"tiny-{adapter_type}",
    )
    report = lora_set.apply_to(fresh)

    assert sorted(key.rsplit(".", 1)[-1] for key in state_dict) == expected_suffixes
    assert config["adapter_type"] == adapter_type
    assert config["step"] == 48
    assert report.applied_layers == 1
    assert report.unsupported_adapters == ()
    assert report.skipped_layers == ()

    base = np.asarray(model.layer.base.weight, dtype=np.float32)
    delta = np.asarray(
        model.layer.U @ model.layer.M_xs @ model.layer.V.T,
        dtype=np.float32,
    )
    v = base + delta
    if adapter_type == "lora-xs":
        expected = v
    elif adapter_type == "dora-rows-xs":
        expected = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        expected *= np.asarray(model.layer.magnitude, dtype=np.float32)[:, None]
    elif adapter_type == "dora-cols-xs":
        expected = v / np.maximum(np.linalg.norm(v, axis=0, keepdims=True), 1e-12)
        expected *= np.asarray(model.layer.magnitude, dtype=np.float32)[None, :]
    else:
        row_scaled = v / np.maximum(
            np.linalg.norm(v, axis=1, keepdims=True),
            1e-12,
        )
        row_scaled *= np.asarray(model.layer.magnitude_r, dtype=np.float32)[:, None]
        expected = row_scaled / np.maximum(
            np.linalg.norm(row_scaled, axis=0, keepdims=True),
            1e-12,
        )
        expected *= np.asarray(model.layer.magnitude_c, dtype=np.float32)[None, :]
    assert mx.allclose(fresh.layer.weight, mx.array(expected), atol=1e-3)
