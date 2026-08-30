"""Ask APFS where a file's bytes physically live.

Two files that share storage -- an APFS clone and its source -- report the same
physical device offset for their first extent. That makes "are these already the
same bytes on disk" a couple of syscalls instead of reading gigabytes.

Getting this right took two false starts worth recording:

  * ctypes cannot make variadic calls correctly on arm64, and fcntl is variadic.
    Every attempt through ctypes returns EFAULT on Apple Silicon. Python's own
    fcntl module uses the right convention.
  * struct log2phys is packed -- u_int32_t followed by two off_t with no
    alignment padding, 20 bytes. Unpacking it as naturally aligned ("IQQ", 24
    bytes) reads fields that straddle two real ones, which produces values that
    still differ between unrelated files and so look like they work.

Only the first extent is compared. Two files sharing a first extent but
diverging later would look identical here; whole-file clones are what this
codebase creates, so that holds in practice. Treat a match as "these already
share storage", never as proof of equal content.
"""

from __future__ import annotations

import fcntl
import os
import struct

F_LOG2PHYS = 49
LOG2PHYS_FMT = "=IQQ"  # packed: l2p_flags, l2p_contigbytes, l2p_devoffset


def first_extent_offset(path):
    """Physical device offset of a file's first extent, or None.

    None means "cannot tell" -- an empty file, one stored inline in its inode,
    an unreadable path, or a filesystem that does not implement F_LOG2PHYS.
    Callers must treat None as "not known to be shared".
    """
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


def shares_storage(left, right) -> bool:
    """True when both files start at the same physical extent."""
    a = first_extent_offset(left)
    if a is None:
        return False
    return a == first_extent_offset(right)
