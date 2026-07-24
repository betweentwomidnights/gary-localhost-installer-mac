#!/usr/bin/env python3
"""Suggest BPM and musical key for one SA3 dataset track.

The estimators live with Carey's training wrapper because both training paths use
the same reconciliation rules. This script runs them in the lightweight SA3
environment and reserves stdout for one JSON response consumed by the macOS UI.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

WRAPPER_DIR = Path(__file__).resolve().parents[2] / "ace-lego" / "wrapper"
SCIPY_REQUIREMENT = "scipy>=1.14"
if str(WRAPPER_DIR) not in sys.path:
    sys.path.insert(0, str(WRAPPER_DIR))


def build_suggestion(bpm: int | None, keyscale: str) -> str:
    parts: list[str] = []
    if bpm is not None:
        parts.append(f"{bpm} bpm")
    if keyscale:
        parts.append(keyscale)
    return ", ".join(parts)


def ensure_scipy() -> None:
    if importlib.util.find_spec("scipy") is not None:
        return
    print(
        f"[environment-setup] Installing analysis dependency: {SCIPY_REQUIREMENT}",
        file=sys.stderr,
        flush=True,
    )
    if importlib.util.find_spec("pip") is not None:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            SCIPY_REQUIREMENT,
        ]
    else:
        uv = shutil.which("uv")
        if not uv:
            for candidate in (
                Path.home() / ".local/bin/uv",
                Path("/opt/homebrew/bin/uv"),
                Path("/usr/local/bin/uv"),
            ):
                if candidate.is_file():
                    uv = str(candidate)
                    break
        if not uv:
            raise RuntimeError(
                "SciPy is missing and neither pip nor uv is available. "
                "Rebuild the SA3 environment and try again."
            )
        command = [
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            SCIPY_REQUIREMENT,
        ]
    try:
        subprocess.check_call(
            command,
            stdout=sys.stderr,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not install {SCIPY_REQUIREMENT} automatically ({exc}). "
            "Rebuild the SA3 environment and try again."
        ) from exc
    importlib.invalidate_caches()
    if importlib.util.find_spec("scipy") is None:
        raise RuntimeError(
            f"{SCIPY_REQUIREMENT} is unavailable after installation. "
            "Rebuild the SA3 environment and try again."
        )


def analyze(audio_path: Path) -> dict[str, object]:
    from bpm_analysis import choose_bpm, estimate_bpm
    from key_analysis import choose_key, estimate_key

    bpm_estimate = estimate_bpm(audio_path)
    key_estimate = estimate_key(audio_path)
    bpm_decision = choose_bpm(local_estimate=bpm_estimate)
    # This is an explicit suggestion the user can correct, so return the best key
    # guess even when it would be too uncertain for unattended caption generation.
    key_decision = choose_key(
        local_estimate=key_estimate,
        minimum_local_confidence=0.0,
    )
    return {
        "ok": True,
        "bpm": bpm_decision.bpm,
        "keyscale": key_decision.keyscale,
        "bpm_source": bpm_decision.source,
        "key_source": key_decision.source,
        "bpm_confidence": (
            round(bpm_estimate.confidence, 4) if bpm_estimate else None
        ),
        "key_confidence": (
            round(key_estimate.confidence, 4) if key_estimate else None
        ),
        "suggestion": build_suggestion(
            bpm_decision.bpm,
            key_decision.keyscale,
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Suggest BPM/key for an SA3 track.")
    parser.add_argument("audio_path")
    args = parser.parse_args(argv)
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.is_file():
        print(json.dumps({"ok": False, "error": f"file not found: {audio_path}"}))
        return 1
    if not WRAPPER_DIR.is_dir():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Carey audio-analysis helpers are missing from the runtime. "
                        "Reinstall or update gary4local."
                    ),
                }
            )
        )
        return 1

    try:
        ensure_scipy()
        result = analyze(audio_path)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
