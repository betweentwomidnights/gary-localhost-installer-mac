#!/usr/bin/env python3
"""Export a provenance-pinned decoder-free ACE conditioning component."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from acestep.models.mlx.artifact_source import (  # noqa: E402
    SourceFileRecord,
    hash_checkpoint_files,
    sha256_file,
    verify_huggingface_source,
)
from acestep.models.mlx.conditioning_export import export_conditioning_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse conditioning export arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--source-subdir",
        default="",
        help="Remote checkpoint directory when the source is a monorepo.",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    return parser.parse_args()


def main() -> int:
    """Verify the immutable source, export the component, and print its path."""
    args = parse_args()
    records = list(hash_checkpoint_files(args.model_dir))
    silence = args.model_dir.expanduser().resolve() / "silence_latent.pt"
    records.append(
        SourceFileRecord(
            filename=silence.name,
            size=silence.stat().st_size,
            sha256=sha256_file(silence),
        )
    )
    verified = verify_huggingface_source(
        tuple(records),
        repo_id=args.source_repo,
        revision=args.source_revision,
        source_subdir=args.source_subdir,
    )
    output = export_conditioning_artifact(
        model_dir=args.model_dir,
        artifact_dir=args.artifact_dir,
        variant=args.variant,
        source_repo=args.source_repo,
        source_revision=args.source_revision,
        source_files=verified,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
