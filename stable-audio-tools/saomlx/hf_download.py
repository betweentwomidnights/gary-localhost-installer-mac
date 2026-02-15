"""Hugging Face download helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download, snapshot_download


def download_file(repo_id: str, filename: str, repo_type: str = "model") -> Path:
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type))


def snapshot(repo_id: str, allow_patterns: Iterable[str] | None = None, repo_type: str = "model") -> Path:
    return Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=list(allow_patterns) if allow_patterns is not None else None,
            repo_type=repo_type,
        )
    )
