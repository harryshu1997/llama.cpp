#!/usr/bin/env python3
"""Capture live GPU, host, and OP15 memory for placement selection."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    PhoneResidencyPlan,
    RuntimePlacementSnapshot,
)


EXPECTED_GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
EXPECTED_GPU_CAPACITY_BYTES = 17_175_674_880


class CaptureError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CaptureError(message)


def run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CaptureError(f"command failed ({result.returncode}): {detail}")
    return result.stdout


def parse_meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.replace("\r", "").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ("MemTotal:", "MemAvailable:"):
            try:
                values[fields[0]] = int(fields[1]) * 1024
            except ValueError as exc:
                raise CaptureError("memory value is not an integer") from exc
    require(set(values) == {"MemTotal:", "MemAvailable:"}, "memory fields")
    total = values["MemTotal:"]
    available = values["MemAvailable:"]
    require(0 < available <= total, "memory capacity")
    return total, available


def parse_gpu(
    text: str,
    *,
    expected_uuid: str,
    expected_name: str,
) -> tuple[int, int]:
    matches = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 4:
            raise CaptureError("unexpected nvidia-smi row")
        uuid, name, total_mib, used_mib = (item.strip() for item in row)
        if uuid != expected_uuid:
            continue
        require(name == expected_name, "GPU name differs from certificate")
        try:
            total = int(total_mib) * 1024 * 1024
            used = int(used_mib) * 1024 * 1024
        except ValueError as exc:
            raise CaptureError("GPU memory value is not an integer") from exc
        require(0 <= used < total, "GPU memory capacity")
        matches.append((total, used))
    require(len(matches) == 1, "certified GPU UUID is not uniquely present")
    return matches[0]


def read_residency_plan(path: Path) -> PhoneResidencyPlan:
    value = json.loads(path.read_text(encoding="ascii"))
    return PhoneResidencyPlan.from_json(value)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb-port", type=int, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--residency-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=EXPECTED_GPU_UUID)
    parser.add_argument("--gpu-name", default=EXPECTED_GPU_NAME)
    parser.add_argument(
        "--gpu-capacity-bytes",
        type=int,
        default=EXPECTED_GPU_CAPACITY_BYTES,
    )
    parser.add_argument("--gpu-reserve-bytes", type=int, default=536_870_912)
    parser.add_argument("--host-reserve-bytes", type=int, default=8_589_934_592)
    parser.add_argument("--valid-for-us", type=int, default=300_000_000)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        residency = read_residency_plan(args.residency_plan)
        gpu_text = run([
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        gpu_total, gpu_used = parse_gpu(
            gpu_text,
            expected_uuid=args.gpu_uuid,
            expected_name=args.gpu_name,
        )
        require(
            gpu_total == args.gpu_capacity_bytes,
            "GPU capacity differs from certificate",
        )
        host_total, host_available = parse_meminfo(
            Path("/proc/meminfo").read_text(encoding="ascii")
        )
        phone_text = run([
            "adb",
            "-P",
            str(args.adb_port),
            "-s",
            args.serial,
            "shell",
            "cat",
            "/proc/meminfo",
        ])
        phone_total, phone_available = parse_meminfo(phone_text)
        require(args.serial == residency.phone_serial, "phone serial")
        require(
            abs(phone_total - residency.memory_capacity_bytes)
            <= 64 * 1024 * 1024,
            "phone capacity differs from residency plan",
        )
        captured_at_us = time.monotonic_ns() // 1000
        snapshot = RuntimePlacementSnapshot(
            snapshot_id=(
                f"full-fp16-live-{args.gpu_uuid}-{args.serial}-"
                f"{captured_at_us}"
            ),
            captured_at_us=captured_at_us,
            valid_until_us=captured_at_us + args.valid_for_us,
            capacities={
                f"cuda:{args.gpu_uuid}:vram": DeviceMemoryCapacity(
                    resource_id=f"cuda:{args.gpu_uuid}:vram",
                    capacity_bytes=gpu_total,
                    occupied_bytes=gpu_used,
                    reserve_bytes=args.gpu_reserve_bytes,
                ),
                "desktop-host-ram": DeviceMemoryCapacity(
                    resource_id="desktop-host-ram",
                    capacity_bytes=host_total,
                    occupied_bytes=host_total - host_available,
                    reserve_bytes=args.host_reserve_bytes,
                ),
                f"phone:{args.serial}:dram": DeviceMemoryCapacity(
                    resource_id=f"phone:{args.serial}:dram",
                    capacity_bytes=phone_total,
                    occupied_bytes=phone_total - phone_available,
                    reserve_bytes=residency.minimum_available_bytes,
                ),
            },
        )
        args.output.write_bytes(canonical(snapshot.to_json()))
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"runtime placement snapshot failed: {exc}\n")
    print(json.dumps({
        "available_bytes": {
            resource_id: capacity.available_bytes
            for resource_id, capacity in snapshot.capacities.items()
        },
        "output": str(args.output),
        "snapshot_id": snapshot.snapshot_id,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
