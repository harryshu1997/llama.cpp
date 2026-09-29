#!/usr/bin/env python3
"""Record peak RSS and swap for named processes until a stop file appears."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


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


def process_memory(pid: int) -> dict[str, int] | None:
    path = Path(f"/proc/{pid}/status")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except FileNotFoundError:
        return None
    fields: dict[str, int] = {}
    for line in lines:
        name, separator, value = line.partition(":")
        if separator and name in {"VmRSS", "VmSwap"}:
            words = value.strip().split()
            if len(words) != 2 or words[1] != "kB":
                raise RuntimeError("invalid process memory field")
            fields[name] = int(words[0]) * 1024
    if set(fields) != {"VmRSS", "VmSwap"}:
        return None
    return {
        "rss_bytes": fields["VmRSS"],
        "swap_bytes": fields["VmSwap"],
    }


def system_swap_free() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("SwapFree:"):
            words = line.partition(":")[2].strip().split()
            if len(words) == 2 and words[1] == "kB":
                return int(words[0]) * 1024
    raise RuntimeError("missing system SwapFree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--process", action="append", required=True, metavar="NAME:PID"
    )
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=50)
    args = parser.parse_args()
    if (
        not args.stop_file.is_absolute()
        or args.stop_file.exists()
        or not args.output.is_absolute()
        or args.output.exists()
        or args.interval_ms <= 0
        or args.interval_ms > 1000
    ):
        parser.error("invalid monitor path or interval")

    identities: dict[str, int] = {}
    for item in args.process:
        name, separator, pid_text = item.partition(":")
        if (
            separator != ":"
            or not name
            or name in identities
            or not pid_text.isdecimal()
            or int(pid_text) <= 0
        ):
            parser.error("invalid process identity")
        identities[name] = int(pid_text)

    started_ns = time.monotonic_ns()
    initial_swap_free = system_swap_free()
    minimum_swap_free = initial_swap_free
    samples = 0
    observed = {name: 0 for name in identities}
    maxima = {
        name: {"rss_bytes": 0, "swap_bytes": 0}
        for name in identities
    }
    while not args.stop_file.exists():
        minimum_swap_free = min(minimum_swap_free, system_swap_free())
        for name, pid in identities.items():
            row = process_memory(pid)
            if row is None:
                continue
            observed[name] += 1
            maxima[name]["rss_bytes"] = max(
                maxima[name]["rss_bytes"], row["rss_bytes"]
            )
            maxima[name]["swap_bytes"] = max(
                maxima[name]["swap_bytes"], row["swap_bytes"]
            )
        samples += 1
        time.sleep(args.interval_ms / 1000)

    completed_ns = time.monotonic_ns()
    final_swap_free = system_swap_free()
    if any(count == 0 for count in observed.values()):
        raise RuntimeError("a monitored process was never observed")
    output = {
        "completed_ns": completed_ns,
        "interval_ms": args.interval_ms,
        "processes": {
            name: {
                "observed_samples": observed[name],
                "pid": identities[name],
                **maxima[name],
            }
            for name in sorted(identities)
        },
        "samples": samples,
        "schema": "s42-process-memory-monitor-v1",
        "started_ns": started_ns,
        "status": "PASS",
        "system_swap": {
            "final_free_bytes": final_swap_free,
            "initial_free_bytes": initial_swap_free,
            "minimum_free_bytes": minimum_swap_free,
        },
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    args.output.write_bytes(canonical(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
