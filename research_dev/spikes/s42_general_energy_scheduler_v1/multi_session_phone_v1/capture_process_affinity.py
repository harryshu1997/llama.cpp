#!/usr/bin/env python3
"""Capture and validate thread affinities for persistent scheduler processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time


SCHEMA = "s42-process-affinity-receipt-v1"


def parse_cpu_list(text: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in text.split(","):
        fields = item.split("-", 1)
        try:
            first = int(fields[0])
            last = int(fields[-1])
        except (IndexError, ValueError) as exc:
            raise argparse.ArgumentTypeError("invalid CPU list") from exc
        if first < 0 or last < first:
            raise argparse.ArgumentTypeError("invalid CPU list bounds")
        values.update(range(first, last + 1))
    if not values:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    return tuple(sorted(values))


def parse_entry(text: str) -> tuple[str, int, tuple[int, ...]]:
    fields = text.split(":", 2)
    if len(fields) != 3 or re.fullmatch(r"[a-z][a-z0-9-]*", fields[0]) is None:
        raise argparse.ArgumentTypeError("invalid process affinity entry")
    try:
        pid = int(fields[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid process PID") from exc
    if pid <= 0:
        raise argparse.ArgumentTypeError("invalid process PID")
    return fields[0], pid, parse_cpu_list(fields[2])


def format_cpu_list(cpus: set[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def process_receipt(pid: int, expected: tuple[int, ...]) -> dict[str, object]:
    root = Path(f"/proc/{pid}")
    counts: dict[str, int] = {}
    try:
        tasks = sorted(root.joinpath("task").iterdir(), key=lambda row: int(row.name))
        command = root.joinpath("cmdline").read_bytes()
        stat = root.joinpath("stat").read_text(encoding="ascii")
    except (FileNotFoundError, OSError, UnicodeError) as exc:
        raise RuntimeError("affinity process is unavailable") from exc
    for task in tasks:
        try:
            cpus = set(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        key = format_cpu_list(cpus)
        counts[key] = counts.get(key, 0) + 1
    expected_set = set(expected)
    if not counts or any(
        not set(parse_cpu_list(cpu_text)) <= expected_set for cpu_text in counts
    ):
        raise RuntimeError("process thread affinity is outside its assignment")
    fields = stat.rsplit(") ", 1)
    if len(fields) != 2 or len(fields[1].split()) < 20:
        raise RuntimeError("process identity is invalid")
    return {
        "cmdline_sha256": hashlib.sha256(command).hexdigest(),
        "cpu_sets": counts,
        "expected_cpus": list(expected),
        "pid": pid,
        "start_ticks": int(fields[1].split()[19]),
        "thread_count": sum(counts.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", action="append", type=parse_entry, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = [entry[0] for entry in args.entry]
    if len(names) != len(set(names)):
        parser.error("process affinity names must be unique")
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or not args.output.parent.is_dir()
    ):
        parser.error("output must be an unused absolute path")
    try:
        processes = {
            name: process_receipt(pid, cpus)
            for name, pid, cpus in args.entry
        }
    except (RuntimeError, ValueError) as exc:
        parser.exit(2, f"process affinity capture failed: {exc}\n")
    value = {
        "captured_ns": time.monotonic_ns(),
        "processes": processes,
        "schema": SCHEMA,
        "status": "PASS",
    }
    args.output.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
