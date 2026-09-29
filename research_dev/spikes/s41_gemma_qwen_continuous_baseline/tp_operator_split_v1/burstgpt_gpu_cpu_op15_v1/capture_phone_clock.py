#!/usr/bin/env python3
"""Capture one bounded desktop-monotonic to phone-uptime clock anchor."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import run_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_trace.require(args.output.is_absolute() and not args.output.exists(), "output")
    command = [
        "adb", "-P", str(args.adb_port), "-s", args.serial,
        "shell", "cat /proc/uptime; cat /proc/sys/kernel/random/boot_id",
    ]
    before_ns = time.monotonic_ns()
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=5, check=False
    )
    after_ns = time.monotonic_ns()
    run_trace.require(
        process.returncode == 0 and not process.stderr.strip(),
        "ADB clock capture",
    )
    lines = process.stdout.splitlines()
    run_trace.require(len(lines) == 2, "phone clock fields")
    fields = lines[0].split()
    run_trace.require(len(fields) == 2, "phone uptime")
    phone_uptime_ns = int(round(float(fields[0]) * 1e9))
    run_trace.require(
        phone_uptime_ns > 0 and after_ns - before_ns <= 500_000_000,
        "clock anchor bounds",
    )
    run_trace.write_json(args.output, {
        "adb_port": args.adb_port,
        "boot_id": lines[1],
        "host_after_ns": after_ns,
        "host_before_ns": before_ns,
        "host_midpoint_ns": (before_ns + after_ns) // 2,
        "phone_uptime_ns": phone_uptime_ns,
        "round_trip_ns": after_ns - before_ns,
        "schema": "s41-phone-clock-anchor-v1",
        "serial": args.serial,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
