#!/usr/bin/env python3
"""Capture one fail-closed GPU thermal bracket from both phones."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


THERMAL_SCRIPT = r"""
boot=$(cat /proc/sys/kernel/random/boot_id) || exit 11
printf 'BOOT\t%s\n' "$boot"
found=0
for zone in /sys/class/thermal/thermal_zone*; do
    type=$(cat "$zone/type" 2>/dev/null) || continue
    case "$type" in
        gpuss-*)
            temp=$(cat "$zone/temp" 2>/dev/null) || exit 12
            printf 'ZONE\t%s\t%s\n' "$type" "$temp"
            found=1
            ;;
    esac
done
test "$found" -eq 1
"""


def parse_device(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("device must be NAME=SERIAL")
    name, serial = value.split("=", 1)
    if not name or not serial:
        raise argparse.ArgumentTypeError("device name and serial must be nonempty")
    return name, serial


def parse_thermal(raw: bytes) -> dict:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("thermal output is not ASCII") from error
    boot_id: str | None = None
    zones: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0] == "BOOT":
            if boot_id is not None or not fields[1]:
                raise ValueError("invalid thermal boot record")
            boot_id = fields[1]
        elif len(fields) == 3 and fields[0] == "ZONE":
            name = fields[1]
            if not name or name in zones:
                raise ValueError("invalid thermal zone name")
            try:
                temp_millic = int(fields[2])
            except ValueError as error:
                raise ValueError("invalid thermal temperature") from error
            if not 0 < temp_millic < 200000:
                raise ValueError("thermal temperature is out of range")
            zones[name] = temp_millic
        elif line:
            raise ValueError("unknown thermal output record")
    if boot_id is None or not zones:
        raise ValueError("thermal output is incomplete")
    return {
        "device_boot_id": boot_id,
        "gpu_max_millic": max(zones.values()),
        "gpu_zones_millic": zones,
    }


def write_atomic(path: Path, value: dict) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--adb-port", type=int, default=5038)
    parser.add_argument("--device", action="append", type=parse_device, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.adb_port <= 65535:
        parser.error("ADB port is out of range")
    if not args.label:
        parser.error("label must be nonempty")
    devices = dict(args.device)
    if len(devices) != len(args.device):
        parser.error("device names must be unique")

    samples = {}
    for name in sorted(devices):
        process = subprocess.run(
            [
                args.adb,
                "-P",
                str(args.adb_port),
                "-s",
                devices[name],
                "shell",
                "sh",
                "-c",
                THERMAL_SCRIPT,
            ],
            check=False,
            capture_output=True,
            timeout=20,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"thermal capture failed for {name}: rc={process.returncode}"
            )
        sample = parse_thermal(process.stdout)
        sample["serial"] = devices[name]
        samples[name] = sample

    report = {
        "schema": "s39-phone-thermal-bracket-v1",
        "captured_utc_ns": time.time_ns(),
        "label": args.label,
        "samples": samples,
    }
    write_atomic(args.output, report)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        print(
            json.dumps(
                {"error": str(error), "verdict": "FAIL"},
                sort_keys=True,
            )
        )
        raise SystemExit(2)
