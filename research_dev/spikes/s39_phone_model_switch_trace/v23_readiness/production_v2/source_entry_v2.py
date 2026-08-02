#!/usr/bin/python3 -I
"""Load captured readiness-driver sources without consulting bytecode caches."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import types


sys.dont_write_bytecode = True


def _take_path(name: str) -> Path:
    try:
        index = sys.argv.index(name)
        path = Path(sys.argv[index + 1])
    except (ValueError, IndexError) as error:
        raise SystemExit(f"V23_RUNTIME_BUNDLE_REFUSED: missing {name}") from error
    del sys.argv[index:index + 2]
    if not path.is_absolute():
        raise SystemExit(f"V23_RUNTIME_BUNDLE_REFUSED: invalid {name}")
    return path


def _read_source(path: Path, field: str) -> str:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SystemExit(
            f"V23_RUNTIME_BUNDLE_REFUSED: cannot read {field}: {error}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SystemExit(
                f"V23_RUNTIME_BUNDLE_REFUSED: invalid {field}"
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
            f"V23_RUNTIME_BUNDLE_REFUSED: changed {field}"
        )
    try:
        return bytes(raw).decode("ascii")
    except UnicodeDecodeError as error:
        raise SystemExit(
            f"V23_RUNTIME_BUNDLE_REFUSED: non-ASCII {field}"
        ) from error


def _load_source(
    path: Path,
    name: str,
    injected: dict[str, object] | None = None,
) -> types.ModuleType:
    source = _read_source(path, name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    if injected:
        module.__dict__.update(injected)
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    exec(code, module.__dict__)
    return module


def load_support():
    base_path = _take_path("--base-support")
    support_path = _take_path("--support")
    base = _load_source(base_path, "s39_v23_driver_common_v1_captured")
    return _load_source(
        support_path,
        "s39_v23_driver_common_v2_captured",
        {"BASE": base},
    )
