#!/usr/bin/python3 -I
"""Capture the V2.3 runtime-bundle fresh readiness snapshot."""

import os
from pathlib import Path
import stat
import sys


sys.dont_write_bytecode = True


def _load_entry():
    try:
        index = sys.argv.index("--entry-support")
        path = Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit(
            "V23_RUNTIME_BUNDLE_REFUSED: missing --entry-support"
        ) from error
    del sys.argv[index:index + 2]
    if not path.is_absolute():
        raise SystemExit(
            "V23_RUNTIME_BUNDLE_REFUSED: invalid --entry-support"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SystemExit(
            "V23_RUNTIME_BUNDLE_REFUSED: cannot read --entry-support"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(
                "V23_RUNTIME_BUNDLE_REFUSED: invalid --entry-support"
            )
        raw = bytearray()
        while block := os.read(descriptor, 1024 * 1024):
            raw.extend(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_mode,
    )
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise SystemExit(
            "V23_RUNTIME_BUNDLE_REFUSED: changed --entry-support"
        )
    try:
        source = bytes(raw).decode("ascii")
    except UnicodeDecodeError as error:
        raise SystemExit(
            "V23_RUNTIME_BUNDLE_REFUSED: non-ASCII --entry-support"
        ) from error
    namespace = {"__file__": str(path), "__name__": "s39_v23_source_entry_v2"}
    exec(compile(source, str(path), "exec", dont_inherit=True), namespace)
    return namespace["load_support"]()


if __name__ == "__main__":
    raise SystemExit(_load_entry().fresh_main())
