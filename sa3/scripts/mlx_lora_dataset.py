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


_NOTE = r"[A-G][#b♯♭]?"
_MODE = r"(?:maj(?:or)?|min(?:or)?)"
_TRAILING_METADATA = re.compile(
    r"[,;]?\s*(?:"
    r"bpm\s*[:=]?\s*\d+(?:\.\d+)?"
    r"|\d+(?:\.\d+)?\s*bpm"
    rf"|(?:key|scale)\s*[:=]\s*{_NOTE}\s+{_MODE}"
    rf"|(?<![A-Za-z]){_NOTE}\s+(?:major|minor)"
    r")\s*$",
    re.IGNORECASE,
)


def dice_prompt_from_caption(text: str) -> str:
    """Remove trailing BPM/key metadata that Gary supplies at inference."""
    prompt = text.strip()
    while True:
        stripped = _TRAILING_METADATA.sub("", prompt).strip(" ,;\t\r\n")
        if stripped == prompt:
            return prompt
        prompt = stripped


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


def _without_trigger(text: str, trigger_text: str) -> str:
    prompt = text.strip()
    trigger = trigger_text.strip()
    if not trigger:
        return prompt
    if prompt.casefold() == trigger.casefold():
        return ""
    if prompt.casefold().startswith(f"{trigger.casefold()},"):
        return prompt[len(trigger) + 1 :].lstrip()
    if prompt.casefold().startswith(f"{trigger.casefold()} "):
        return prompt[len(trigger) :].lstrip()
    return prompt


def prompt_pool(
    examples: list[LoraDatasetExample],
    *,
    trigger_text: str = "",
) -> list[str]:
    prompts = []
    seen = set()
    for example in examples:
        # The trigger belongs only to the composed training prompt. Dice choices
        # come from literal sidecar content so a user's magic word is never emitted
        # unexpectedly by the plugin.
        source = _without_trigger(example.source_prompt, trigger_text)
        prompt = dice_prompt_from_caption(source)
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
