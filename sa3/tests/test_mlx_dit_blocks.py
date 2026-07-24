from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx

import stable_audio_3.mlx.dit_blocks as dit_blocks
from stable_audio_3.mlx.dit import _hosted_medium_source_key
from stable_audio_3.mlx.dit_blocks import (
    Attention,
    RotaryEmbedding,
    apply_fast_rotary_pos_emb,
    apply_rotary_pos_emb,
)


@pytest.mark.parametrize(
    ("dtype", "atol"),
    [
        (mx.float32, 2e-5),
        (mx.float16, 2e-3),
    ],
)
def test_fast_rope_matches_existing_half_half_rotation(dtype, atol: float):
    mx.random.seed(17)
    values = mx.random.normal((2, 3, 128, 64)).astype(dtype)
    freqs, _ = RotaryEmbedding(32).forward_from_seq_len(128)

    expected = apply_rotary_pos_emb(
        values.astype(mx.float32),
        freqs.astype(mx.float32),
    ).astype(dtype)
    actual = apply_fast_rotary_pos_emb(
        values.astype(mx.float32),
        freqs,
    ).astype(dtype)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, atol=atol, rtol=atol)


def test_fast_rope_matches_existing_rotation_gradient():
    mx.random.seed(19)
    values = mx.random.normal((1, 4, 128, 64)).astype(mx.float32)
    coefficients = mx.random.normal(values.shape).astype(mx.float32)
    freqs, _ = RotaryEmbedding(32).forward_from_seq_len(128)

    def existing_loss(inputs):
        rotated = apply_rotary_pos_emb(inputs, freqs)
        return mx.sum(rotated * coefficients)

    def fused_loss(inputs):
        rotated = apply_fast_rotary_pos_emb(inputs, freqs)
        return mx.sum(rotated * coefficients)

    expected = mx.grad(existing_loss)(values)
    actual = mx.grad(fused_loss)(values)
    mx.eval(expected, actual)

    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_fast_rope_matches_existing_differential_attention_forward_and_gradient(
    monkeypatch: pytest.MonkeyPatch,
):
    mx.random.seed(23)
    attention = Attention(
        64,
        dim_heads=64,
        differential=True,
        qk_norm="rms",
        zero_init_output=False,
    )
    values = mx.random.normal((1, 48, 64)).astype(mx.float32)
    coefficients = mx.random.normal(values.shape).astype(mx.float32)
    rotary = RotaryEmbedding(32).forward_from_seq_len(48)

    def attention_loss(inputs):
        output = attention(inputs, rotary_pos_emb=rotary)
        return mx.sum(output * coefficients)

    monkeypatch.setattr(dit_blocks, "_NAIVE_ROPE", True)
    expected_output = attention(values, rotary_pos_emb=rotary)
    expected_gradient = mx.grad(attention_loss)(values)
    monkeypatch.setattr(dit_blocks, "_NAIVE_ROPE", False)
    actual_output = attention(values, rotary_pos_emb=rotary)
    actual_gradient = mx.grad(attention_loss)(values)
    mx.eval(
        expected_output,
        actual_output,
        expected_gradient,
        actual_gradient,
    )

    assert mx.allclose(
        actual_output,
        expected_output,
        atol=3e-5,
        rtol=3e-5,
    )
    assert mx.allclose(
        actual_gradient,
        expected_gradient,
        atol=3e-5,
        rtol=3e-5,
    )


def test_hosted_medium_weight_key_mapping():
    assert (
        _hosted_medium_source_key("transformer.layers.0.pre_norm.gamma")
        == "transformer.layers.0.pre_norm.weight"
    )
    assert (
        _hosted_medium_source_key(
            "transformer.layers.0.to_local_embed.2.weight"
        )
        == "transformer.layers.0.to_local_embed.seq.2.weight"
    )
    assert (
        _hosted_medium_source_key(
            "transformer.layers.0.self_attn.to_qkv.weight"
        )
        == "transformer.layers.0.self_attn.to_qkv.weight"
    )
