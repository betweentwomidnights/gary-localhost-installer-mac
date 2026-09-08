#!/usr/bin/env python3
"""Export one provenance-pinned ACE-Step MLX decoder artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acestep.models.mlx.artifact_export import export_decoder_artifact  # noqa: E402
from acestep.models.mlx.artifact_source import (  # noqa: E402
    hash_checkpoint_files,
    sha256_file,
    verify_huggingface_source,
)


def _git_output(*args: str) -> str:
    """Return stripped git output, or an empty string outside a repository."""
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _converter_metadata() -> dict[str, object]:
    """Describe both git state and exact conversion source files."""
    files = (
        REPO_ROOT / "acestep/models/mlx/artifact_export.py",
        REPO_ROOT / "acestep/models/mlx/artifact_manifest.py",
        REPO_ROOT / "acestep/models/mlx/artifact_source.py",
        REPO_ROOT / "acestep/models/mlx/dit_convert.py",
    )
    combined = hashlib.sha256()
    file_hashes = {}
    for path in files:
        digest = sha256_file(path)
        relative = str(path.relative_to(REPO_ROOT))
        file_hashes[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(bytes.fromhex(digest))
    return {
        "git_commit": _git_output("rev-parse", "HEAD") or None,
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "source_sha256": combined.hexdigest(),
        "source_files": file_hashes,
    }


def parse_args() -> argparse.Namespace:
    """Parse the standalone artifact export command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-subdir",
        default="",
        help="Remote checkpoint directory when the source is a monorepo.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--skip-hf-verification",
        action="store_true",
        help="Record local hashes without comparing the public source revision.",
    )
    return parser.parse_args()


def main() -> int:
    """Hash, verify, export, and print the completed provenance record."""
    args = parse_args()
    records = hash_checkpoint_files(args.model_dir)
    if not args.skip_hf_verification:
        records = verify_huggingface_source(
            records,
            repo_id=args.source_repo,
            revision=args.source_revision,
            source_subdir=args.source_subdir,
        )
    output = export_decoder_artifact(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        variant=args.variant,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        source_files=records,
        dtype_name=args.dtype,
        converter=_converter_metadata(),
    )
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
