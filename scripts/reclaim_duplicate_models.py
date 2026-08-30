#!/usr/bin/env python3
"""Reclaim space taken by Carey model files stored twice.

The downloader fetches into the Hugging Face cache and then places the file in
the checkpoints tree. Until it learned to clone, that placement was a full byte
copy, so every Carey model cost twice its size: once as a cache blob, once as a
checkpoint. This pass finds those pairs and replaces the checkpoint copy with an
APFS clone of the blob, which is the same bytes sharing the same storage.

Nothing is deleted. A checkpoint file is only ever replaced by a clone of a blob
that is byte-identical to it, so both trees are exactly as complete afterwards
as they were before -- the space is recovered from the duplication itself.

Identity is established by content hash, not by size. Hugging Face names each
LFS blob after the sha256 of its contents, so a checkpoint file whose sha256
matches a blob's filename is that blob, with no chance of a same-size mismatch.

This is macOS/APFS specific: clonefile(2) is what makes a copy free. On a
filesystem without it, or with the two trees on different volumes, nothing is
cloned and the pass reports zero.

Note this is NOT the same problem the Windows build reclaims. There the hub
cannot symlink without developer mode, so it writes a file twice inside one repo
cache -- blobs/ and snapshots/ both real. macOS symlinks fine, so that never
happens here; the duplication is between the cache and the checkpoints tree.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apfs_extents import first_extent_offset  # noqa: E402

# Below this, a duplicate is not worth a hash pass.
MIN_INTERESTING_BYTES = 1 * 1024 * 1024
HASH_CHUNK = 8 * 1024 * 1024
SKIP_SUFFIXES = (".incomplete", ".tmp", ".clone.tmp", ".lock")


def emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def clonefile(src: Path, dst: Path) -> bool:
    """COW-clone src to dst. dst must not exist. False if the fs cannot."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        fn = libc.clonefile
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        return fn(os.fsencode(str(src)), os.fsencode(str(dst)), 0) == 0
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    stat = os.statvfs(str(path))
    return stat.f_bavail * stat.f_frsize


def index_blobs(hub_dir: Path) -> dict:
    """Map blob sha256 (its filename) -> path, for every cached repo.

    Only 64-hex names are indexed. Small files are stored under a git sha1
    instead, and those are not worth reclaiming anyway.
    """
    blobs: dict = {}
    if not hub_dir.is_dir():
        return blobs
    for repo in hub_dir.iterdir():
        blob_dir = repo / "blobs"
        if not blob_dir.is_dir():
            continue
        for blob in blob_dir.iterdir():
            name = blob.name
            if len(name) != 64:
                continue
            try:
                int(name, 16)
            except ValueError:
                continue
            try:
                if blob.is_symlink() or not blob.is_file():
                    continue
                if blob.stat().st_size < MIN_INTERESTING_BYTES:
                    continue
            except OSError:
                continue
            blobs[name] = blob
    return blobs


def candidate_files(checkpoints_dir: Path) -> list:
    found = []
    for root, _dirs, files in os.walk(checkpoints_dir):
        for name in files:
            if name.endswith(SKIP_SUFFIXES):
                continue
            path = Path(root) / name
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            if size < MIN_INTERESTING_BYTES:
                continue
            found.append((size, path))
    found.sort(key=lambda item: -item[0])
    return found


def contained_in(path: Path, root: Path) -> bool:
    """Refuse to touch anything that resolves outside the checkpoints tree."""
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    return root_resolved == resolved or root_resolved in resolved.parents


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True)
    parser.add_argument("--hf-cache", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be reclaimed without changing anything.",
    )
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir).expanduser()
    hub_dir = Path(args.hf_cache).expanduser()

    if not checkpoints_dir.is_dir():
        emit({"event": "error", "message": f"checkpoints directory not found: {checkpoints_dir}"})
        return 1

    blobs = index_blobs(hub_dir)
    files = candidate_files(checkpoints_dir)
    emit({
        "event": "start",
        "candidate_files": len(files),
        "indexed_blobs": len(blobs),
        "dry_run": bool(args.dry_run),
    })

    # Match by size first so we only hash files that could possibly pair up.
    blob_sizes: dict = {}
    for name, path in blobs.items():
        try:
            blob_sizes.setdefault(path.stat().st_size, []).append(name)
        except OSError:
            continue

    free_before = free_bytes(checkpoints_dir)
    matched_bytes = 0
    reclaimed_files = 0
    reclaimed_bytes = 0
    already_shared_files = 0
    already_shared_bytes = 0
    skipped = 0
    # Only size-matched files are ever hashed, so this is the real denominator
    # for how long the pass will take -- not the total file count.
    checkable = [(size, path) for size, path in files if size in blob_sizes]
    checkable_bytes = sum(size for size, _ in checkable)
    emit({
        "event": "checking",
        "files": len(checkable),
        "bytes": checkable_bytes,
    })

    for index, (size, path) in enumerate(checkable):
        emit({
            "event": "progress",
            "index": index + 1,
            "total": len(checkable),
            "file": str(path.relative_to(checkpoints_dir)),
            "bytes": size,
            "found_bytes": matched_bytes,
            "reclaimed_bytes": reclaimed_bytes,
        })

        # Before reading a byte: if this file already starts at the same
        # physical extent as one of the same-size blobs, it is already a clone
        # of it and there is nothing to recover. This is what makes a repeat
        # run cost a few syscalls per file instead of hashing everything again.
        own_offset = first_extent_offset(path)
        already = None
        if own_offset is not None:
            for name in blob_sizes[size]:
                if first_extent_offset(blobs[name]) == own_offset:
                    already = name
                    break
        if already is not None:
            matched_bytes += size
            already_shared_files += 1
            already_shared_bytes += size
            emit({
                "event": "already_shared",
                "file": str(path.relative_to(checkpoints_dir)),
                "bytes": size,
            })
            continue

        try:
            digest = sha256_file(path)
        except OSError as exc:
            emit({"event": "skip", "file": str(path), "reason": str(exc)})
            skipped += 1
            continue

        blob = blobs.get(digest)
        if blob is None:
            continue

        matched_bytes += size
        if args.dry_run:
            emit({
                "event": "would_reclaim",
                "file": str(path.relative_to(checkpoints_dir)),
                "bytes": size,
                "blob": blob.name,
            })
            continue

        if not contained_in(path, checkpoints_dir):
            emit({"event": "skip", "file": str(path), "reason": "outside checkpoints root"})
            skipped += 1
            continue

        tmp_path = path.with_name(path.name + ".clone.tmp")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass

        if not clonefile(blob, tmp_path):
            emit({"event": "skip", "file": str(path), "reason": "clonefile unavailable"})
            skipped += 1
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
            continue

        try:
            # Measure what the clone actually bought. A file that already
            # shares storage with its blob -- because a previous pass cloned
            # it -- is still byte-identical and still "matches", but replacing
            # it frees nothing. Counting those as reclaimable is what made a
            # second run promise the same space over again.
            free_before_file = free_bytes(checkpoints_dir)
            shutil.copystat(path, tmp_path)
            os.replace(str(tmp_path), str(path))
            gained = free_bytes(checkpoints_dir) - free_before_file

            # Well under one file's worth: nothing was really recovered.
            if gained < size // 2:
                already_shared_files += 1
                already_shared_bytes += size
                emit({
                    "event": "already_shared",
                    "file": str(path.relative_to(checkpoints_dir)),
                    "bytes": size,
                })
            else:
                reclaimed_files += 1
                reclaimed_bytes += size
                emit({
                    "event": "reclaimed",
                    "file": str(path.relative_to(checkpoints_dir)),
                    "bytes": size,
                    "reclaimed_bytes": reclaimed_bytes,
                })
        except OSError as exc:
            emit({"event": "skip", "file": str(path), "reason": str(exc)})
            skipped += 1
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

    free_after = free_bytes(checkpoints_dir)
    emit({
        "event": "done",
        "dry_run": bool(args.dry_run),
        # matched_bytes is "content that is duplicated", which stays true after
        # a successful pass. reclaimed_* is what this run actually recovered.
        "matched_bytes": matched_bytes,
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed_files": reclaimed_files,
        "already_shared_files": already_shared_files,
        "already_shared_bytes": already_shared_bytes,
        "skipped": skipped,
        # The measured figure is what the volume actually gained. It can differ
        # from matched_bytes when a file was already sharing storage, or when
        # something else wrote to the disk while this ran.
        "free_delta_bytes": free_after - free_before,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
