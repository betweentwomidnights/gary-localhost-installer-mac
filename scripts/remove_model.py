#!/usr/bin/env python3
"""Remove a downloaded model, and say honestly what that gives back.

Models live in three shapes here: a repo directory in the shared Hugging Face
cache, a subdirectory of Carey's checkpoints tree, or Foundation's own weights
directory. All three are just directories, so removal is the same operation --
the care is in refusing to do it to the wrong one, and in not promising space
that will not appear.

The promise matters because of cloning. A Carey checkpoint is usually a
copy-on-write clone of a cache blob, so deleting the checkpoint alone frees
almost nothing: the bytes stay, held by the blob. Reporting the directory's
size would repeat exactly the overstatement this codebase already fixed once.
So a dry run indexes the physical extents of everything that will REMAIN and
counts only the target bytes that nothing else holds.

Safety rules, all of them refusals rather than best-effort:
  * the target must resolve strictly inside one of the allowed roots
  * the target may not BE a root -- no removing the whole cache by passing it
  * the target may not be a symlink, and symlinks inside it are unlinked, never
    followed out of the tree
  * a target that has vanished is reported, not treated as success

Whether it is safe to remove a model *right now* -- no service holding it open,
no download writing into it, no training job reading it -- is the caller's
question, not this script's. The app checks that before offering the button.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apfs_extents import first_extent_offset  # noqa: E402

import hashlib

LARGE_FILE_BYTES = 1 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_references(hub: Path) -> dict:
    """blob path -> the snapshot symlinks pointing at it, keyed by repo.

    macOS symlinks its snapshots into blobs/, so following each link is how you
    learn which logical file a blob actually serves.
    """
    refs = {}
    if not hub.is_dir():
        return refs
    for repo in hub.iterdir():
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        for dirpath, _dirnames, filenames in os.walk(snapshots):
            for name in filenames:
                link = Path(dirpath) / name
                if not link.is_symlink():
                    continue
                try:
                    target = link.resolve()
                except OSError:
                    continue
                refs.setdefault(str(target), []).append(link)
    return refs


def backing_cache_entries(model_dir: Path, hub: Path) -> tuple:
    """Cache blobs that exist only to serve this model, and their symlinks.

    A blob is claimed only when every snapshot link pointing at it names a file
    this model owns -- either at the repo root (a dedicated repo) or under a
    folder named after the model (a repo shared with other models, which is how
    ACE-Step/Ace-Step1.5 carries Qwen, the VAE and v15-turbo together). Anything
    else is left alone, because removing it would take another model's files
    with it.
    """
    model_name = model_dir.name
    owned = set()
    by_extent = {}
    by_hash = {}
    for path, size in iter_files(model_dir):
        rel = str(path.relative_to(model_dir))
        owned.add(rel)
        if size >= LARGE_FILE_BYTES:
            offset = first_extent_offset(str(path))
            if offset is not None:
                by_extent.setdefault(offset, []).append(rel)
        else:
            try:
                by_hash.setdefault(sha256_file(path), []).append(rel)
            except OSError:
                continue

    refs = snapshot_references(hub)
    blobs = []
    links = []
    for repo in (hub.iterdir() if hub.is_dir() else []):
        blob_dir = repo / "blobs"
        if not blob_dir.is_dir():
            continue
        for blob in blob_dir.iterdir():
            if blob.is_symlink() or not blob.is_file():
                continue
            try:
                size = blob.stat().st_size
            except OSError:
                continue

            claimed = False
            if size >= LARGE_FILE_BYTES:
                offset = first_extent_offset(str(blob))
                claimed = offset is not None and offset in by_extent
            else:
                claimed = blob.name in by_hash

            if not claimed:
                continue

            # Every link to it must name a file this model owns.
            blob_links = refs.get(str(blob.resolve()), [])
            exclusive = True
            for link in blob_links:
                parts = link.parts
                try:
                    snap_index = len(parts) - 1 - parts[::-1].index("snapshots")
                except ValueError:
                    exclusive = False
                    break
                logical = Path(*parts[snap_index + 2:])
                as_str = str(logical)
                if as_str in owned:
                    continue
                if logical.parts and logical.parts[0] == model_name \
                        and str(Path(*logical.parts[1:])) in owned:
                    continue
                exclusive = False
                break

            if exclusive:
                blobs.append(blob)
                links.extend(blob_links)
    return blobs, links


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def iter_files(root: Path, skip: Path = None):
    """Files under root, optionally pruning one subtree.

    The pruning matters: a root usually CONTAINS the target, and indexing the
    target's own extents as "kept" would make every removal look like it frees
    nothing.
    """
    skip_resolved = skip.resolve() if skip is not None else None
    for dirpath, dirnames, filenames in os.walk(root):
        if skip_resolved is not None:
            dirnames[:] = [
                d for d in dirnames
                if Path(dirpath, d).resolve() != skip_resolved
            ]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                yield path, path.stat().st_size
            except OSError:
                continue


def extents_of(root: Path, skip: Path = None) -> set:
    """Physical offsets of every large file under root, excluding `skip`."""
    offsets = set()
    for path, size in iter_files(root, skip=skip):
        if size < LARGE_FILE_BYTES:
            continue
        offset = first_extent_offset(str(path))
        if offset is not None:
            offsets.add(offset)
    return offsets


def measure_target(target: Path, keep_roots: list) -> tuple:
    """Bytes the target holds, and how many of them nothing else holds.

    Small files are counted in full: they are rarely cloned and indexing them
    would cost more than the answer is worth.
    """
    kept_extents = set()
    for root in keep_roots:
        kept_extents |= extents_of(root, skip=target)

    logical = 0
    freeable = 0
    files = 0
    for path, size in iter_files(target):
        logical += size
        files += 1
        if size >= LARGE_FILE_BYTES:
            offset = first_extent_offset(str(path))
            if offset is not None and offset in kept_extents:
                continue
        freeable += size
    return logical, freeable, files


def resolve_target(target: str, roots: list) -> Path:
    resolved = Path(target).expanduser()
    if resolved.is_symlink():
        raise ValueError("refusing to remove a symlink")
    resolved = resolved.resolve()

    for root in roots:
        root_resolved = root.resolve()
        if resolved == root_resolved:
            raise ValueError(f"refusing to remove the root itself: {resolved}")
        if root_resolved in resolved.parents:
            return resolved
    raise ValueError(f"target is not inside an allowed root: {resolved}")


def prune_empty_dirs(paths: list) -> None:
    """Drop directories left empty by the files we just removed.

    Stops at blobs/ and snapshots/ so a repo that still holds other models
    keeps its shape.
    """
    stops = {"blobs", "snapshots"}
    seen = set()
    for path in paths:
        parent = path.parent
        while parent not in seen:
            seen.add(parent)
            if parent.name in stops or not parent.is_dir():
                break
            try:
                if any(parent.iterdir()):
                    break
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def free_bytes(path: Path) -> int:
    stat = os.statvfs(str(path))
    return stat.f_bavail * stat.f_frsize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], required=True,
                        help="An allowed root. Repeatable.")
    parser.add_argument("--target", required=True, help="Directory to remove.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--hub",
        default=None,
        help=(
            "Hugging Face cache. When given, cache files that exist only to "
            "serve this model are removed with it -- otherwise deleting a "
            "checkpoint that was cloned from the cache frees nothing."
        ),
    )
    args = parser.parse_args()

    roots = [Path(r).expanduser() for r in args.root]
    existing_roots = [r for r in roots if r.is_dir()]

    try:
        target = resolve_target(args.target, existing_roots)
    except ValueError as exc:
        emit({"event": "error", "message": str(exc)})
        return 1

    if not target.exists():
        emit({"event": "error", "message": f"nothing to remove at {target}"})
        return 1
    if not target.is_dir():
        emit({"event": "error", "message": f"not a directory: {target}"})
        return 1

    # Everything that will still be here afterwards, so shared bytes are not
    # counted as recoverable.
    # Every root is indexed; iter_files prunes the target subtree out of each,
    # so the target's own bytes never count as "held by something else".
    keep_roots = [r for r in existing_roots if r.resolve() != target]

    emit({"event": "measuring", "target": str(target)})

    # Cache copies that serve only this model go with it. Without this, removing
    # a Carey checkpoint frees nothing at all: it is a clone, and the bytes stay
    # behind held by the blob it was cloned from.
    hub = Path(args.hub).expanduser() if args.hub else None
    cache_blobs, cache_links = ([], [])
    if hub is not None and hub.is_dir() and target != hub and hub not in target.parents:
        cache_blobs, cache_links = backing_cache_entries(target, hub)

    logical, freeable, files = measure_target(target, keep_roots)
    cache_bytes = 0
    for blob in cache_blobs:
        try:
            cache_bytes += blob.stat().st_size
        except OSError:
            continue

    # Once the cache copy goes too, the shared bytes are genuinely recoverable.
    total_logical = logical + cache_bytes
    total_freeable = min(total_logical, freeable + cache_bytes)

    emit({
        "event": "measured",
        "target": str(target),
        "logical_bytes": total_logical,
        "freeable_bytes": total_freeable,
        "shared_bytes": max(0, total_logical - total_freeable),
        "files": files + len(cache_blobs),
        "cache_blobs": len(cache_blobs),
        "cache_bytes": cache_bytes,
    })

    if args.dry_run:
        emit({"event": "done", "dry_run": True, "removed": False,
              "logical_bytes": total_logical, "freeable_bytes": total_freeable,
              "freed_bytes": 0})
        return 0

    before = free_bytes(target.parent)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        emit({"event": "error", "message": f"could not remove {target}: {exc}"})
        return 1

    # Symlinks first: a blob removed while a link still points at it leaves a
    # dangling entry that the hub reads as a corrupt cache rather than a missing
    # model.
    for link in cache_links:
        try:
            link.unlink()
        except OSError:
            continue
    for blob in cache_blobs:
        try:
            blob.unlink()
        except OSError:
            continue
    prune_empty_dirs(cache_links + cache_blobs)

    freed = free_bytes(target.parent) - before

    emit({
        "event": "done",
        "dry_run": False,
        "removed": True,
        "logical_bytes": total_logical,
        "freeable_bytes": total_freeable,
        "cache_blobs": len(cache_blobs),
        # What the volume actually gave back. Differs from freeable_bytes when
        # something else wrote to the disk while this ran.
        "freed_bytes": freed,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
