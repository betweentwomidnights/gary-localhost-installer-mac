from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_EXTENSIONS = {
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
}

TAG_LABELS = {
    "track_type": "TrackType",
    "vocal_type": "VocalType",
    "title": "Title",
    "artist": "Artist",
    "album": "Album",
    "genre": "Genre",
    "mood": "Mood",
    "composer": "Composer",
    "bpm": "BPM",
}


@dataclass(frozen=True)
class LoraDatasetExample:
    audio_path: Path
    relative_path: str
    sidecar_path: Path | None
    sidecar_kind: str | None
    source_prompt: str
    prompt: str


def compose_trigger_prompt(trigger_text: str, prompt: str) -> str:
    trigger = trigger_text.strip()
    caption = prompt.strip()
    if not trigger:
        return caption
    if not caption:
        return trigger
    if caption.casefold() == trigger.casefold():
        return caption
    normalized_caption = caption.casefold()
    normalized_trigger = trigger.casefold()
    if normalized_caption.startswith(f"{normalized_trigger},") or normalized_caption.startswith(
        f"{normalized_trigger} "
    ):
        return caption
    return f"{trigger}, {caption}"


def dice_prompt_from_caption(text: str) -> str:
    return re.sub(
        r"(?:[,;]\s*)?(?:BPM\s*:\s*\d+(?:\.\d+)?|\d+(?:\.\d+)?\s*BPM)\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip(" ,;")


def discover_audio_files(root: Path) -> list[Path]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise NotADirectoryError(f"Dataset folder not found: {resolved_root}")
    return sorted(
        (
            path.resolve()
            for path in resolved_root.rglob("*")
            if path.is_file()
            and not path.name.startswith("._")
            and path.suffix.lower() in AUDIO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(resolved_root).as_posix().casefold(),
    )


def discover_dataset_examples(
    root: Path,
    *,
    trigger_text: str = "",
) -> list[LoraDatasetExample]:
    resolved_root = root.expanduser().resolve()
    examples = []
    for audio_path in discover_audio_files(resolved_root):
        sidecar_path, sidecar_kind, source_prompt = _read_sidecar_prompt(
            resolved_root,
            audio_path,
        )
        examples.append(
            LoraDatasetExample(
                audio_path=audio_path,
                relative_path=audio_path.relative_to(resolved_root).as_posix(),
                sidecar_path=sidecar_path,
                sidecar_kind=sidecar_kind,
                source_prompt=source_prompt,
                prompt=compose_trigger_prompt(trigger_text, source_prompt),
            )
        )
    return examples


def prompt_pool(examples: list[LoraDatasetExample]) -> list[str]:
    prompts = []
    seen = set()
    for example in examples:
        prompt = dice_prompt_from_caption(example.prompt)
        normalized = prompt.casefold()
        if not prompt or normalized in seen:
            continue
        seen.add(normalized)
        prompts.append(prompt)
    return prompts


def _read_sidecar_prompt(
    root: Path,
    audio_path: Path,
) -> tuple[Path | None, str | None, str]:
    for candidate in _sidecar_candidates(root, audio_path, ".json"):
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        prompt = _prompt_from_json(payload)
        if prompt:
            return candidate, "json", prompt

    for candidate in _sidecar_candidates(root, audio_path, ".txt"):
        if not candidate.is_file():
            continue
        try:
            prompt = candidate.read_text(encoding="utf-8-sig", errors="replace").strip()
        except OSError:
            continue
        if prompt:
            return candidate, "txt", prompt
    return None, None, ""


def _sidecar_candidates(root: Path, audio_path: Path, suffix: str) -> list[Path]:
    candidates = [audio_path.with_suffix(suffix)]
    parent = audio_path.parent
    if parent != root and parent.parent.is_relative_to(root):
        candidates.append(parent.parent / suffix.lstrip(".") / f"{audio_path.stem}{suffix}")
    return candidates


def _prompt_from_json(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    explicit = payload.get("prompt")
    if isinstance(explicit, (str, int, float)) and str(explicit).strip():
        return str(explicit).strip()

    pieces = []
    for key, label in TAG_LABELS.items():
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            pieces.append(f"{label}: {str(value).strip()}")
    return ", ".join(pieces)
