"""Tests for ACE MLX LoRA/DoRA training helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from acestep.models.mlx import lora_training as ace_lora  # noqa: E402


def test_balanced_profile_matches_pc_rank_64_family_budget() -> None:
    targets, ranks, alphas = ace_lora.build_balanced_projection_profile(64, 128)

    assert len(targets) == 11
    assert ranks == {
        "cross_attn.k_proj": 40,
        "cross_attn.o_proj": 48,
        "cross_attn.q_proj": 64,
        "cross_attn.v_proj": 32,
        "mlp.down_proj": 48,
        "mlp.gate_proj": 40,
        "mlp.up_proj": 48,
        "self_attn.k_proj": 24,
        "self_attn.o_proj": 56,
        "self_attn.q_proj": 16,
        "self_attn.v_proj": 80,
    }
    assert all(alphas[name] == rank * 2 for name, rank in ranks.items())


def test_attention_profile_targets_attention_projections_only() -> None:
    targets, ranks, alphas = ace_lora.build_ace_projection_profile(
        rank=12,
        alpha=24,
        module_profile="attention",
    )

    assert targets == [
        "cross_attn.k_proj",
        "cross_attn.o_proj",
        "cross_attn.q_proj",
        "cross_attn.v_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]
    assert set(ranks) == set(targets)
    assert set(alphas) == set(targets)
    assert all(rank == 12 for rank in ranks.values())
    assert all(alpha == 24 for alpha in alphas.values())
    assert not any(target.startswith("mlp.") for target in targets)


@pytest.mark.parametrize("rank,alpha", [(0, 128), (64, 0), (-1, 128)])
def test_balanced_profile_rejects_invalid_reference_values(
    rank: int,
    alpha: int,
) -> None:
    with pytest.raises(ValueError):
        ace_lora.build_balanced_projection_profile(rank, alpha)


def _mlx_modules():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    tree_flatten = pytest.importorskip("mlx.utils").tree_flatten
    return mx, nn, tree_flatten


def _make_tiny_ace_decoder(num_layers: int = 1):
    _mx, nn, _tree_flatten = _mlx_modules()

    class _TinyAttention(nn.Module):
        def __init__(self, hidden_size: int):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
            self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    class _TinyMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    class _TinyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _TinyAttention(96)
            self.cross_attn = _TinyAttention(96)
            self.mlp = _TinyMLP(96, 128)

    class _TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_TinyBlock() for _ in range(num_layers)]

    return _TinyDecoder()


def test_balanced_injection_uses_attention_and_mlp_rank_pattern() -> None:
    _mx, _nn, tree_flatten = _mlx_modules()
    model = _make_tiny_ace_decoder()

    report = ace_lora.inject_trainable_lora(
        model,
        rank=64,
        alpha=128,
        module_profile="balanced",
        adapter_type="dora",
    )
    layers = {
        layer.source_name: layer
        for layer in ace_lora.iter_trainable_lora_layers(model)
    }
    trainable_count = sum(
        int(value.size)
        for _name, value in tree_flatten(model.trainable_parameters())
    )
    trainable_names = [
        name for name, _value in tree_flatten(model.trainable_parameters())
    ]

    assert report.layer_count == 11
    assert report.adapter_type == "dora"
    assert report.module_profile == "balanced"
    assert layers["layers.0.self_attn.q_proj"].rank == 16
    assert layers["layers.0.self_attn.v_proj"].rank == 80
    assert layers["layers.0.cross_attn.q_proj"].rank == 64
    assert layers["layers.0.mlp.gate_proj"].rank == 40
    assert hasattr(layers["layers.0.self_attn.q_proj"], "magnitude")
    assert report.trainable_parameters == trainable_count
    assert all(".base." not in name for name in trainable_names)


def test_attention_injection_excludes_mlp_layers() -> None:
    _mx, _nn, _tree_flatten = _mlx_modules()
    model = _make_tiny_ace_decoder()

    report = ace_lora.inject_trainable_lora(
        model,
        rank=8,
        alpha=16,
        module_profile="attention",
        adapter_type="lora",
    )
    layers = list(ace_lora.iter_trainable_lora_layers(model))

    assert report.layer_count == 8
    assert all(".mlp." not in name for name in report.layer_names)
    assert all(layer.rank == 8 for layer in layers)
    assert all(layer.alpha == 16 for layer in layers)
    assert all(layer.adapter_type == "lora" for layer in layers)


def test_save_trainable_lora_adapter_writes_peft_style_directory(tmp_path: Path) -> None:
    mx, _nn, _tree_flatten = _mlx_modules()
    model = _make_tiny_ace_decoder()
    ace_lora.inject_trainable_lora(
        model,
        rank=64,
        alpha=128,
        module_profile="balanced",
        adapter_type="dora",
    )

    output_dir = ace_lora.save_trainable_lora_adapter(
        model,
        tmp_path / "adapter",
        rank=64,
        alpha=128,
        module_profile="balanced",
        adapter_type="dora",
    )

    weights = mx.load(str(output_dir / ace_lora.PEFT_ADAPTER_WEIGHTS))
    config = json.loads((output_dir / ace_lora.PEFT_ADAPTER_CONFIG).read_text())

    assert output_dir.is_dir()
    assert (
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        in weights
    )
    assert (
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        in weights
    )
    assert (
        "base_model.model.layers.0.self_attn.q_proj."
        "lora_magnitude_vector.default.weight"
        in weights
    )
    assert config["peft_type"] == "LORA"
    assert config["use_dora"] is True
    assert config["target_modules"] == [
        "cross_attn.k_proj",
        "cross_attn.o_proj",
        "cross_attn.q_proj",
        "cross_attn.v_proj",
        "mlp.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "self_attn.k_proj",
        "self_attn.o_proj",
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]
    assert config["rank_pattern"]["self_attn.v_proj"] == 80
    assert config["alpha_pattern"]["self_attn.v_proj"] == 160
