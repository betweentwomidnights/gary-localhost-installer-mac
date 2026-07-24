import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_mlx_lora import (  # noqa: E402
    FULL_TRACK_LATENT_FRAMES,
    SA3_ADAMW_BETAS,
    SA3_ADAMW_WEIGHT_DECAY,
    _augment_padding_mask,
    _cache_directory,
    _conditioning_seconds_for_example,
    _create_sa3_adamw,
    _crop_or_pad_latents,
    _effective_sequence_length,
    _full_track_bucket_latents,
    _log_prompt_policy,
    _resolve_lora_filters,
    _resolve_training_window,
    _sample_inpaint_mask,
)
from mlx_lora_dataset import (  # noqa: E402
    compose_trigger_prompt,
    dice_prompt_from_caption,
    discover_dataset_examples,
    prompt_pool,
)


def test_discovers_recursive_audio_and_honors_sidecar_precedence(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    first = tmp_path / "one.wav"
    second = nested / "two.flac"
    ignored = tmp_path / "notes.md"
    for path in (first, second, ignored):
        path.write_bytes(b"content")
    first.with_suffix(".txt").write_text("bright bells, BPM: 145\n")
    second.with_suffix(".txt").write_text("ignored text\n")
    second.with_suffix(".json").write_text('{"genre": "ambient", "mood": "glassy"}\n')

    examples = discover_dataset_examples(tmp_path, trigger_text="garybell")

    assert [example.relative_path for example in examples] == ["nested/two.flac", "one.wav"]
    assert examples[0].sidecar_kind == "json"
    assert examples[0].prompt == "garybell, Genre: ambient, Mood: glassy"
    assert examples[1].prompt == "garybell, bright bells, BPM: 145"
    assert prompt_pool(examples) == [
        "Genre: ambient, Mood: glassy",
        "bright bells",
    ]
    assert first.with_suffix(".txt").read_text() == "bright bells, BPM: 145\n"


def test_trigger_is_not_duplicated() -> None:
    assert compose_trigger_prompt("garybell", "garybell, bright bells") == (
        "garybell, bright bells"
    )
    assert compose_trigger_prompt("", "bright bells") == "bright bells"
    assert compose_trigger_prompt("garybell", "") == "garybell"
    assert dice_prompt_from_caption("bright bells; 145 BPM") == "bright bells"
    assert dice_prompt_from_caption("liquid dnb, 174 bpm, A minor") == "liquid dnb"
    assert dice_prompt_from_caption("post-rock, Key: F# major") == "post-rock"


def test_prompt_pool_removes_legacy_trigger_prefix(tmp_path: Path) -> None:
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"content")
    audio.with_suffix(".txt").write_text("garybell, bright bells, 145 bpm\n")
    examples = discover_dataset_examples(tmp_path, trigger_text="garybell")

    assert prompt_pool(examples, trigger_text="garybell") == ["bright bells"]


def test_prompt_policy_log_shows_rendered_conditioning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio = tmp_path / "one.wav"
    audio.write_bytes(b"content")
    audio.with_suffix(".txt").write_text("bright bells, 145 bpm\n")
    examples = discover_dataset_examples(tmp_path, trigger_text="garybell")

    _log_prompt_policy("garybell", examples)

    output = capsys.readouterr().out
    assert 'shared trigger "garybell" is prepended' in output
    assert "sidecars and dice prompts remain unchanged" in output
    assert 'example conditioning="garybell, bright bells, 145 bpm"' in output


def test_short_latents_are_padded_after_encoding() -> None:
    latents = np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    cropped = _crop_or_pad_latents(latents, crop_latents=5, offset=0)

    assert cropped.dtype == np.float16
    np.testing.assert_array_equal(cropped[..., :3], latents.astype(np.float16))
    np.testing.assert_array_equal(cropped[..., 3:], np.zeros((1, 2, 2)))


def test_short_latents_use_encoded_silence_padding() -> None:
    latents = np.arange(8, dtype=np.float32).reshape(1, 2, 4)
    silence = np.array(
        [[[100, 101], [200, 201]]],
        dtype=np.float32,
    )

    cropped = _crop_or_pad_latents(
        latents,
        crop_latents=7,
        offset=0,
        valid_frames=3,
        padding_latents=silence,
    )

    np.testing.assert_array_equal(cropped[..., :3], latents[..., :3])
    np.testing.assert_array_equal(
        cropped[..., 3:],
        np.array([[[100, 101, 100, 101], [200, 201, 200, 201]]]),
    )


def test_full_track_window_is_exactly_3072_latent_frames() -> None:
    crop_latents, aligned_seconds = _resolve_training_window(
        full_tracks=True,
        crop_seconds=47,
        sample_rate=44_100,
        downsampling_ratio=4_096,
    )

    assert crop_latents == FULL_TRACK_LATENT_FRAMES == 3_072
    assert aligned_seconds == pytest.approx(285.3268, abs=0.0001)


@pytest.mark.parametrize(
    ("valid_frames", "expected_bucket"),
    [
        (1_286, 1_536),  # 119-second song plus a short encoded-silence tail
        (1_865, 2_048),  # 173-second song
        (2_381, 2_560),  # 221-second song
        (3_072, 3_072),  # maximum full-track window
    ],
)
def test_full_track_buckets_avoid_computing_the_entire_padded_window(
    valid_frames: int,
    expected_bucket: int,
) -> None:
    assert _full_track_bucket_latents(
        valid_frames=valid_frames,
        maximum_frames=FULL_TRACK_LATENT_FRAMES,
        sample_rate=44_100,
        downsampling_ratio=4_096,
    ) == expected_bucket


def test_random_crop_window_keeps_existing_alignment() -> None:
    crop_latents, aligned_seconds = _resolve_training_window(
        full_tracks=False,
        crop_seconds=47,
        sample_rate=44_100,
        downsampling_ratio=4_096,
    )

    assert crop_latents == 512
    assert aligned_seconds == pytest.approx(47.5545, abs=0.0001)


def test_default_lora_filters_adapt_every_eligible_layer() -> None:
    include, exclude = _resolve_lora_filters(None, None)

    assert include is None
    assert exclude == []


def test_explicit_lora_filters_override_defaults() -> None:
    include, exclude = _resolve_lora_filters(
        ["transformer.layers.[0-3]"],
        ["project_out"],
    )

    assert include == ["transformer.layers.[0-3]"]
    assert exclude == ["project_out"]


def test_random_crop_conditioning_preserves_full_source_duration() -> None:
    seconds = _conditioning_seconds_for_example(
        source_duration_seconds=221.10,
        aligned_crop_seconds=23.78,
        full_tracks=False,
    )

    assert seconds == pytest.approx(221.10)
    assert _effective_sequence_length(
        conditioning_seconds=seconds,
        crop_latents=256,
        sample_rate=44_100,
        downsampling_ratio=4_096,
        use_effective_length=True,
    ) == 2_381


def test_full_track_conditioning_caps_duration_at_training_window() -> None:
    seconds = _conditioning_seconds_for_example(
        source_duration_seconds=380.0,
        aligned_crop_seconds=285.3268,
        full_tracks=True,
    )

    assert seconds == pytest.approx(285.3268)


def test_sa3_optimizer_uses_official_pytorch_lora_contract() -> None:
    optimizer = _create_sa3_adamw(1e-4)

    assert optimizer.betas == list(SA3_ADAMW_BETAS) == [0.9, 0.95]
    assert optimizer.weight_decay == SA3_ADAMW_WEIGHT_DECAY == 0.01
    assert optimizer.eps == 1e-8
    assert optimizer.bias_correction is True


def test_latent_cache_is_addressed_by_audio_content(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "renamed.wav"
    second = second_dir / "copy.wav"
    different = tmp_path / "different.wav"
    first.write_bytes(b"same audio bytes")
    second.write_bytes(b"same audio bytes")
    different.write_bytes(b"different audio bytes")

    first_example = discover_dataset_examples(first_dir)[0]
    second_example = discover_dataset_examples(second_dir)[0]
    different_example = discover_dataset_examples(tmp_path)[0]
    cache_root = tmp_path / "cache"

    assert _cache_directory(cache_root, first_example) == _cache_directory(
        cache_root,
        second_example,
    )
    assert _cache_directory(
        cache_root,
        first_example,
    ) != _cache_directory(cache_root, different_example)


def test_upstream_inpaint_policy_samples_10_80_10_and_uses_zero_for_generation() -> None:
    rng = np.random.default_rng(90210)
    padding = np.ones((1, 64), dtype=np.bool_)
    counts = {"random_segments": 0, "full": 0, "causal": 0}

    for _ in range(10_000):
        mask, mode = _sample_inpaint_mask(padding, rng=rng)
        counts[mode] += 1
        assert mask.shape == (1, 1, 64)
        if mode == "full":
            np.testing.assert_array_equal(mask, np.zeros_like(mask))
        elif mode == "random_segments":
            assert np.any(mask == 0)

    assert 850 <= counts["random_segments"] <= 1_150
    assert 7_700 <= counts["full"] <= 8_300
    assert 850 <= counts["causal"] <= 1_150


def test_padding_augmentation_only_extends_into_encoded_silence() -> None:
    mask = _augment_padding_mask(
        valid_frames=32,
        crop_latents=64,
        sample_rate=44_100,
        downsampling_ratio=4_096,
        rng=np.random.default_rng(7),
    )

    assert mask.shape == (1, 64)
    assert 32 <= int(mask.sum()) <= 64
    np.testing.assert_array_equal(mask[:, :32], np.ones((1, 32), dtype=np.bool_))


def test_long_latents_are_cropped_from_requested_offset() -> None:
    latents = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

    cropped = _crop_or_pad_latents(latents, crop_latents=3, offset=2)

    np.testing.assert_array_equal(
        cropped,
        latents[..., 2:5].astype(np.float16),
    )
