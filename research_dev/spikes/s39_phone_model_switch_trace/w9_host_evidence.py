#!/usr/bin/env python3
"""Capture W9 page-cache and selected-CUDA evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import w9_profiled_cutover as w9


GPU_FIELDS = (
    "index",
    "name",
    "uuid",
    "pci.bus_id",
    "driver_version",
    "memory.total",
    "power.limit",
    "temperature.gpu",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "memory.used",
    "utilization.gpu",
)


def run_checked(arguments: list[str]) -> bytes:
    process = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if process.returncode != 0:
        raise w9.W9Error(
            f"command failed ({process.returncode}): {' '.join(arguments)}"
        )
    return process.stdout


def parse_csv_line(raw: bytes, expected: int, field: str) -> list[str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise w9.W9Error(f"{field}: non-ASCII output") from exc
    rows = list(csv.reader(text.splitlines()))
    w9.require(len(rows) == 1 and len(rows[0]) == expected, f"{field}: shape")
    return [item.strip() for item in rows[0]]


def integer(value: str, field: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise w9.W9Error(f"{field}: invalid integer") from exc
    w9.require(result >= 0, f"{field}: negative value")
    return result


def decimal_milli(value: str, field: str) -> int:
    try:
        result = int(Decimal(value) * 1000)
    except (InvalidOperation, ValueError) as exc:
        raise w9.W9Error(f"{field}: invalid decimal") from exc
    w9.require(result >= 0, f"{field}: negative value")
    return result


def query_gpu(uuid: str) -> tuple[dict[str, object], str]:
    checked_uuid = uuid
    w9.require(
        type(checked_uuid) is str and checked_uuid.startswith("GPU-"),
        "gpu: invalid UUID",
    )
    raw = run_checked([
        "nvidia-smi",
        "-i",
        checked_uuid,
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ])
    fields = parse_csv_line(raw, len(GPU_FIELDS), "gpu")
    value = dict(zip(GPU_FIELDS, fields))
    w9.require(value["uuid"] == uuid, "gpu: UUID mismatch")
    sample = {
        "clock_memory_mhz": integer(
            value["clocks.current.memory"],
            "gpu.clock_memory",
        ),
        "clock_sm_mhz": integer(value["clocks.current.sm"], "gpu.clock_sm"),
        "index": integer(value["index"], "gpu.index"),
        "memory_total_mib": integer(value["memory.total"], "gpu.memory_total"),
        "memory_used_mib": integer(value["memory.used"], "gpu.memory_used"),
        "name": value["name"],
        "pci_bus_id": value["pci.bus_id"],
        "power_limit_mw": decimal_milli(
            value["power.limit"],
            "gpu.power_limit",
        ),
        "power_mw": decimal_milli(value["power.draw"], "gpu.power"),
        "pstate": value["pstate"],
        "temperature_c": integer(
            value["temperature.gpu"],
            "gpu.temperature",
        ),
        "utilization_gpu_pct": integer(
            value["utilization.gpu"],
            "gpu.utilization",
        ),
        "uuid": value["uuid"],
        "driver_version": value["driver_version"],
    }
    w9.require(
        sample["name"]
        and sample["pci_bus_id"]
        and sample["pstate"].startswith("P")
        and sample["driver_version"],
        "gpu: incomplete identity",
    )
    return sample, raw.decode("ascii")


def query_compute_processes(
    uuid: str,
) -> tuple[list[dict[str, object]], str]:
    raw = run_checked([
        "nvidia-smi",
        "-i",
        uuid,
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise w9.W9Error("gpu processes: non-ASCII output") from exc
    result = []
    for index, row in enumerate(csv.reader(text.splitlines())):
        if not row:
            continue
        w9.require(len(row) == 4, f"gpu processes.{index}: shape")
        gpu_uuid, pid, name, used = [item.strip() for item in row]
        w9.require(gpu_uuid == uuid and name, "gpu processes: identity")
        result.append({
            "gpu_uuid": gpu_uuid,
            "pid": integer(pid, "gpu processes.pid"),
            "process_name": name,
            "used_memory_mib": integer(used, "gpu processes.memory"),
        })
    return result, text


def capture_gpu(
    uuid: str,
    *,
    samples: int,
    span_us: int,
    require_idle: bool,
) -> dict[str, object]:
    w9.require(samples >= 1 and span_us >= 0, "gpu capture: invalid timing")
    interval_s = span_us / 1_000_000 / max(1, samples - 1)
    values = []
    started_ns = time.monotonic_ns()
    for index in range(samples):
        sample_started_ns = time.monotonic_ns()
        gpu, raw_gpu = query_gpu(uuid)
        processes, raw_processes = query_compute_processes(uuid)
        completed_ns = time.monotonic_ns()
        values.append({
            "completed_ns": completed_ns,
            "compute_processes": processes,
            "gpu": gpu,
            "raw_compute_processes_csv": raw_processes,
            "raw_gpu_query_csv": raw_gpu,
            "sample_index": index,
            "started_ns": sample_started_ns,
        })
        if index + 1 < samples:
            target = started_ns + int((index + 1) * interval_s * 1_000_000_000)
            delay = (target - time.monotonic_ns()) / 1_000_000_000
            if delay > 0:
                time.sleep(delay)
    ended_ns = time.monotonic_ns()
    identity = {
        key: values[0]["gpu"][key]
        for key in (
            "driver_version",
            "index",
            "memory_total_mib",
            "name",
            "pci_bus_id",
            "power_limit_mw",
            "uuid",
        )
    }
    for value in values:
        w9.require(
            all(value["gpu"][key] == expected for key, expected in identity.items()),
            "gpu capture: identity changed",
        )
    idle = all(
        value["gpu"]["utilization_gpu_pct"] == 0
        and value["compute_processes"] == []
        for value in values
    )
    w9.require(
        not require_idle or (idle and ended_ns - started_ns >= span_us * 1000),
        "gpu capture: idle gate",
    )
    return {
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ended_ns": ended_ns,
        "host_boot_id": Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="ascii").strip(),
        "identity": identity,
        "idle": idle,
        "require_idle": require_idle,
        "samples": values,
        "started_ns": started_ns,
    }


def warm_file(path: Path) -> dict[str, object]:
    w9.require(path.is_file() and not path.is_symlink(), "page cache: unsafe file")
    digest = hashlib.sha256()
    count = 0
    started_ns = time.monotonic_ns()
    with path.open("rb") as source:
        while True:
            block = source.read(16 * 1024 * 1024)
            if not block:
                break
            count += len(block)
            digest.update(block)
    ended_ns = time.monotonic_ns()
    stat = path.stat()
    w9.require(count == stat.st_size and count > 0, "page cache: byte count")
    return {
        "bytes_read": count,
        "ended_ns": ended_ns,
        "file_sha256": digest.hexdigest(),
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "started_ns": started_ns,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    warm = subparsers.add_parser("warm")
    warm.add_argument("--path", type=Path, required=True)
    warm.add_argument("--label", required=True)
    warm.add_argument("--output", type=Path, required=True)
    gpu = subparsers.add_parser("gpu")
    gpu.add_argument("--uuid", required=True)
    gpu.add_argument("--samples", type=int, required=True)
    gpu.add_argument("--span-us", type=int, required=True)
    gpu.add_argument("--require-idle", action="store_true")
    gpu.add_argument("--label", required=True)
    gpu.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.label:
        parser.error("invalid output or label")
    try:
        if args.command == "warm":
            payload = warm_file(args.path)
            schema = "s39-page-cache-precondition-v1"
        else:
            payload = capture_gpu(
                args.uuid,
                samples=args.samples,
                span_us=args.span_us,
                require_idle=args.require_idle,
            )
            schema = "s39-selected-gpu-bracket-v1"
        report = {
            "label": args.label,
            "payload": payload,
            "schema": schema,
            "status": "PASS",
        }
        w9.write_atomic(args.output, report)
        print(
            __import__("json").dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        w9.W9Error,
    ) as exc:
        print(
            __import__("json").dumps(
                {"error": str(exc), "status": "FAIL"},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
