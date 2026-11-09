"""Return freed heap pages to the OS between large batch phases (no src imports)."""

from __future__ import annotations

import ctypes
import ctypes.util


def release_free_memory() -> None:
    """Hand freed Arrow-pool and glibc-heap pages back to the OS.

    Both allocators keep freed pages for reuse, so a process that just dropped a multi-GB
    intermediate still holds it as RSS and the next phase peaks on top of it. Values in live
    objects are untouched; on a non-glibc platform the heap step is skipped.
    """
    import pyarrow as pa

    pa.default_memory_pool().release_unused()
    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        return
    try:
        trim = ctypes.CDLL(libc_name).malloc_trim
    except (OSError, AttributeError):
        return
    trim(0)
