#!/usr/bin/env python3
"""Page-cache controls for the CP0-D warm and cold model-load baselines."""

from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path
import time
from typing import Any


class CacheError(RuntimeError):
    pass


def resident_pages(path: Path) -> dict[str, int]:
    size = path.stat().st_size
    if size <= 0:
        raise CacheError(f"{path}: empty file")
    page_size = os.sysconf("SC_PAGE_SIZE")
    page_count = (size + page_size - 1) // page_size
    with path.open("rb") as stream:
        mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
        try:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            vector = (ctypes.c_ubyte * page_count)()
            libc = ctypes.CDLL(None, use_errno=True)
            result = libc.mincore(
                ctypes.c_void_p(address),
                ctypes.c_size_t(size),
                ctypes.byref(vector),
            )
            if result != 0:
                error = ctypes.get_errno()
                raise CacheError(f"mincore failed for {path}: errno={error}")
            resident = sum(1 for value in vector if value & 1)
        finally:
            mapping.close()
    return {
        "bytes": size,
        "page_size": page_size,
        "pages": page_count,
        "resident_pages": resident,
        "resident_ppm": resident * 1_000_000 // page_count,
    }


def warm_file(path: Path, block_bytes: int = 32 * 1024 * 1024) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    total = 0
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(block_bytes)
            if not block:
                break
            total += len(block)
    if total != path.stat().st_size:
        raise CacheError(f"{path}: incomplete warm read")
    result = resident_pages(path)
    result.update({
        "elapsed_ns": time.monotonic_ns() - started_ns,
        "method": "COMPLETE_SEQUENTIAL_READ",
    })
    return result


def evict_file(path: Path) -> dict[str, Any]:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise CacheError("POSIX_FADV_DONTNEED is unavailable")
    started_ns = time.monotonic_ns()
    with path.open("rb", buffering=0) as stream:
        os.posix_fadvise(
            stream.fileno(), 0, path.stat().st_size, os.POSIX_FADV_DONTNEED
        )
    time.sleep(0.1)
    result = resident_pages(path)
    result.update({
        "elapsed_ns": time.monotonic_ns() - started_ns,
        "method": "POSIX_FADV_DONTNEED_COMPLETE_FILE",
    })
    return result
