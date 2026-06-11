import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from train_mlx_lora import _crop_or_pad_latents  # noqa: E402
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
        "garybell, Genre: ambient, Mood: glassy",
        "garybell, bright bells",
    ]


def test_trigger_is_not_duplicated() -> None:
    assert compose_trigger_prompt("garybell", "garybell, bright bells") == (
        "garybell, bright bells"
    )
    assert compose_trigger_prompt("garybell", "") == "garybell"
    assert dice_prompt_from_caption("bright bells; 145 BPM") == "bright bells"


def test_short_latents_are_padded_after_encoding() -> None:
    latents = np.arange(6, dtype=np.float32).reshape(1, 2, 3)

    cropped = _crop_or_pad_latents(latents, crop_latents=5, offset=0)

    assert cropped.dtype == np.float16
    np.testing.assert_array_equal(cropped[..., :3], latents.astype(np.float16))
    np.testing.assert_array_equal(cropped[..., 3:], np.zeros((1, 2, 2)))


def test_long_latents_are_cropped_from_requested_offset() -> None:
    latents = np.arange(12, dtype=np.float32).reshape(1, 2, 6)

    cropped = _crop_or_pad_latents(latents, crop_latents=3, offset=2)

    np.testing.assert_array_equal(
        cropped,
        latents[..., 2:5].astype(np.float16),
    )
