"""Resolve writable ACE-Step cache paths outside packaged application code."""

from __future__ import annotations

import os
from pathlib import Path


def get_acestep_cache_root(project_root: str | os.PathLike[str]) -> str:
    """Return the writable cache root, honoring app-managed cache locations."""
    configured = os.getenv("ACESTEP_CACHE_DIR", "").strip()
    if configured:
        return str(Path(configured).expanduser().resolve())

    xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return str((Path(xdg_cache).expanduser() / "acestep").resolve())

    return str((Path(project_root).expanduser().resolve() / ".cache" / "acestep"))
