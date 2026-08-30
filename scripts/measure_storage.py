#!/usr/bin/env python3
"""Measure what gary4local actually occupies, not what it appears to occupy.

`du` counts a cloned file in full on both sides, so after the duplicate pass the
cache and the checkpoints tree each still report their whole size even though
they share most of their bytes. That produced a total far larger than the disk
the install is really using, which is worse than no number at all.

APFS can answer the real question. F_LOG2PHYS reports the physical device offset
of a file's first extent, and two files that share storage report the same one.
Counting each distinct offset once gives a figure that moves when space is
actually recovered.

Only files at or above LARGE_FILE_BYTES are probed. Cloning is something that
happens to model weights, and paying an open/fcntl on every small file in a
Python environment would cost more than it could ever explain.

Caveats worth knowing:
  - Only the first extent is compared. Two files sharing a first extent but
    diverging later would be counted once. Whole-file clones are what we make,
    so this holds in practice.
  - A file with no addressable extent (empty, or stored inline) simply counts in
    full.
  - Sharing is detected across everything measured in one run. A file sharing
    with something outside these paths is counted here in full, which is the
    honest answer for "what would I get back".
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import sys

F_LOG2PHYS = 49
# The kernel struct is packed: u_int32_t + off_t + off_t, no alignment padding.
LOG2PHYS_FMT = "=IQQ"
LARGE_FILE_BYTES = 1 * 1024 * 1024


def first_extent_offset(path: str):
    """Physical device offset of a file's first extent, or None."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        packed = fcntl.fcntl(fd, F_LOG2PHYS, struct.pack(LOG2PHYS_FMT, 0, 0, 0))
        _flags, _contig, devoffset = struct.unpack(LOG2PHYS_FMT, packed)
        return devoffset
    except OSError:
        return None
    finally:
        os.close(fd)


def measure(roots: dict) -> dict:
    """Walk each root once, reporting logical and actual bytes.

    `seen` spans every root, so a blob and the checkpoint cloned from it are
    counted once overall even though they live under different roots.
    """
    seen_extents = set()
    seen_inodes = set()
    results = {}
    # A root that contains another (the install root contains the cache) must
    # not walk into it, or every byte would read as already-seen and the
    # containing root would report almost nothing.
    all_roots = {os.path.realpath(r) for r in roots.values()}

    for label, root in roots.items():
        root_real = os.path.realpath(root)
        nested = {r for r in all_roots if r != root_real and r.startswith(root_real + os.sep)}
        logical = 0
        actual = 0
        files = 0
        shared = 0
        if not os.path.isdir(root):
            results[label] = {"logical": 0, "actual": 0, "files": 0, "shared_files": 0}
            continue

        for dirpath, dirnames, filenames in os.walk(root):
            if nested:
                dirnames[:] = [
                    d for d in dirnames
                    if os.path.realpath(os.path.join(dirpath, d)) not in nested
                ]
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    stat = os.lstat(path)
                except OSError:
                    continue
                # Symlinks cost nothing; the hub's snapshots are all symlinks.
                if not os.path.isfile(path) or os.path.islink(path):
                    continue
                size = stat.st_size
                logical += size
                files += 1

                # Hard links: same inode, one copy on disk.
                if stat.st_nlink > 1:
                    if stat.st_ino in seen_inodes:
                        shared += 1
                        continue
                    seen_inodes.add(stat.st_ino)

                if size >= LARGE_FILE_BYTES:
                    offset = first_extent_offset(path)
                    if offset is not None:
                        if offset in seen_extents:
                            shared += 1
                            continue
                        seen_extents.add(offset)
                actual += size

        results[label] = {
            "logical": logical,
            "actual": actual,
            "files": files,
            "shared_files": shared,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="A labelled directory to measure. Repeatable.",
    )
    args = parser.parse_args()

    roots = {}
    for entry in args.path:
        label, _, path = entry.partition("=")
        if not path:
            print(json.dumps({"error": f"expected LABEL=PATH, got {entry!r}"}))
            return 1
        roots[label] = os.path.expanduser(path)

    if not roots:
        print(json.dumps({"error": "no paths given"}))
        return 1

    # Walked in the order given. Where two paths share storage, the first one
    # walked is charged for it and the second reports it as shared -- so pass
    # the tree you would keep before the one you might remove.
    results = measure(roots)

    any_path = next(iter(roots.values()))
    stat = os.statvfs(any_path if os.path.exists(any_path) else os.path.expanduser("~"))
    print(json.dumps({
        "paths": {label: results[label] for label in roots},
        "free_bytes": stat.f_bavail * stat.f_frsize,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
