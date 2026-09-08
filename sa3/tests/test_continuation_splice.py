import math
from pathlib import Path

import torch

from api import (
    cached_hf_snapshot_file,
    continuation_mask_bounds,
    continuation_splice_params,
    splice_continuation_source,
)


def test_finds_decoder_file_in_commit_pinned_snapshot(tmp_path: Path):
    checkpoint = (
        tmp_path
        / "models--thepatch--same-l-decoder-lora"
        / "snapshots"
        / "commit-sha"
        / "squeakfix_v3.safetensors"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"adapter")

    assert cached_hf_snapshot_file(
        "thepatch/same-l-decoder-lora",
        "squeakfix_v3.safetensors",
        tmp_path,
    ) == str(checkpoint.resolve())


def test_request_boolean_controls_accept_explicit_off():
    parsed = continuation_splice_params(
        {
            "splice_source": "off",
            "splice_xfade": "0.125",
            "splice_gain_match": 0,
            "mask_overlap": "0.2",
        }
    )
    assert parsed == {
        "splice_source": False,
        "splice_xfade": 0.125,
        "splice_gain_match": False,
        "mask_overlap": 0.2,
    }


def test_mask_overlap_moves_regeneration_into_source_and_clamps_short_inputs():
    assert continuation_mask_bounds(8.0, 0.2) == (0.2, 7.8)
    overlap, mask_start = continuation_mask_bounds(0.1, 0.2)
    assert math.isclose(overlap, 0.05)
    assert math.isclose(mask_start, 0.05)
    assert continuation_mask_bounds(0.03, 0.2) == (0.0, 0.03)


def test_restores_source_head_and_leaves_generated_tail_untouched():
    generated = torch.arange(12, dtype=torch.float32).view(1, 1, -1)
    source = torch.full((1, 8), -2.0)
    output, meta = splice_continuation_source(
        generated,
        (10, source),
        mask_start_seconds=0.8,
        sample_rate=10,
        xfade_seconds=0.2,
        gain_match=False,
    )
    torch.testing.assert_close(output[..., :6], source[..., :6].unsqueeze(0))
    torch.testing.assert_close(output[..., 8:], generated[..., 8:])
    assert meta == {
        "splice_applied": True,
        "splice_end_seconds": 0.8,
        "splice_xfade_applied": 0.2,
        "splice_gain": 1.0,
    }


def test_gain_match_is_clamped_and_mono_source_expands_to_stereo():
    generated = torch.ones(1, 2, 8)
    source = torch.full((1, 8), 0.01)
    output, meta = splice_continuation_source(
        generated,
        (8, source),
        mask_start_seconds=1.0,
        sample_rate=8,
        xfade_seconds=0,
        gain_match=True,
    )
    torch.testing.assert_close(output, torch.full_like(generated, 0.04))
    assert meta["splice_gain"] == 4.0


def test_equal_power_fade_meets_generated_audio_at_seam():
    sample_rate = 1000
    splice_end = 800
    timeline = torch.arange(1200, dtype=torch.float32) / sample_rate
    generated = torch.sin(2 * math.pi * 7 * timeline).view(1, 1, -1)
    source = (generated[..., :splice_end] * 0.5).squeeze(0)
    output, meta = splice_continuation_source(
        generated,
        (sample_rate, source),
        mask_start_seconds=splice_end / sample_rate,
        sample_rate=sample_rate,
        xfade_seconds=0.03,
        gain_match=True,
    )
    assert meta["splice_xfade_applied"] == 0.03
    assert math.isclose(meta["splice_gain"], 2.0, abs_tol=1e-5)
    assert math.isclose(
        output[0, 0, splice_end - 1].item(),
        generated[0, 0, splice_end - 1].item(),
        abs_tol=1e-5,
    )
    torch.testing.assert_close(output[..., splice_end:], generated[..., splice_end:])
