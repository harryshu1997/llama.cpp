#!/usr/bin/env python3
"""Capture cgroup memory and swap enforcement for a process tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def read_int(path: Path) -> int | str:
    value = path.read_text(encoding="ascii").strip()
    return value if value == "max" else int(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pid = args.pid or os.getpid()
    if (
        pid <= 0
        or not args.output.is_absolute()
        or args.output.exists()
    ):
        parser.error("invalid process or output")

    rows = Path(f"/proc/{pid}/cgroup").read_text(
        encoding="ascii"
    ).splitlines()
    unified = [line.partition("::")[2] for line in rows if "::" in line]
    if len(unified) != 1 or not unified[0].startswith("/"):
        raise RuntimeError("process has no unique unified cgroup")
    relative = unified[0]
    root = Path("/sys/fs/cgroup").joinpath(relative.removeprefix("/"))
    events: dict[str, int] = {}
    for line in root.joinpath("memory.events").read_text(
        encoding="ascii"
    ).splitlines():
        name, value = line.split()
        events[name] = int(value)
    output = {
        "cgroup": relative,
        "memory": {
            "current_bytes": read_int(root / "memory.current"),
            "events": events,
            "peak_bytes": read_int(root / "memory.peak"),
            "swap_current_bytes": read_int(root / "memory.swap.current"),
            "swap_max_bytes": read_int(root / "memory.swap.max"),
        },
        "pid": pid,
        "schema": "s42-cgroup-memory-receipt-v1",
        "status": "PASS",
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    args.output.write_bytes(canonical(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
