#!/usr/bin/env python3
"""Validate an ACE-Step MLX artifact and optionally prove exact source parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acestep.models.mlx.artifact_validate import (  # noqa: E402
    validate_artifact,
    validate_source_parity,
)
from acestep.models.mlx.conditioning_validate import validate_conditioning_parity  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse artifact validation command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Also hash the pinned checkpoint and compare every converted tensor.",
    )
    return parser.parse_args()


def main() -> int:
    """Run structural validation and optional exact source tensor comparison."""
    args = parse_args()
    report = {"artifact": validate_artifact(args.artifact_dir)}
    if args.source_dir is not None:
        report["source_parity"] = validate_source_parity(
            args.source_dir,
            args.artifact_dir,
        )
        if (args.artifact_dir / "conditioning").is_dir():
            report["conditioning_parity"] = validate_conditioning_parity(
                args.source_dir,
                args.artifact_dir,
            )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
