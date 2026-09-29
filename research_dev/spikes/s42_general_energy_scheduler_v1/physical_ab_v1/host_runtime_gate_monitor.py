#!/usr/bin/env python3
"""Sample host topology, thermal gates, USB state, and bridge progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
HOT_MODEL = "/home/zhihao/models/Qwen3-14B-Q4_K_M.gguf"
COLD_MODEL = "/home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf"
BRIDGE = "/home/zhihao/s41-dynamic-ffn-v1/ffn_dmabuf_bridge-reset-qualified"
CPU_THROTTLE = Path(
    "/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count"
)


def canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def processes() -> dict[str, list[int]]:
    result = {"bridge": [], "cold": [], "hot": []}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "cmdline").read_bytes().split(b"\0")
            args = [field.decode("utf-8") for field in fields if field]
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeError):
            continue
        if not args:
            continue
        executable = args[0]
        pid = int(entry.name)
        if executable == BRIDGE:
            result["bridge"].append(pid)
        elif executable.endswith("/llama-server") and HOT_MODEL in args:
            result["hot"].append(pid)
        elif executable.endswith("/llama-server") and COLD_MODEL in args:
            result["cold"].append(pid)
    return {key: sorted(value) for key, value in result.items()}


def gpu() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,temperature.gpu,power.draw,utilization.gpu,"
        "memory.used,pstate,clocks_event_reasons.sw_thermal_slowdown,"
        "clocks_event_reasons.hw_thermal_slowdown",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, capture_output=True, check=True, text=True, timeout=5
    )
    fields = [field.strip() for field in completed.stdout.strip().split(",")]
    if len(fields) != 8:
        raise ValueError("unexpected nvidia-smi output")
    return {
        "hardware_thermal_slowdown": fields[7],
        "memory_used_mib": int(fields[4]),
        "power_w": float(fields[2]),
        "pstate": fields[5],
        "software_thermal_slowdown": fields[6],
        "temperature_c": int(fields[1]),
        "utilization_pct": int(fields[3]),
        "uuid": fields[0],
    }


def usb_devices() -> list[dict[str, object]]:
    rows = []
    for entry in Path("/sys/bus/usb/devices").iterdir():
        try:
            vendor = (entry / "idVendor").read_text().strip()
            product = (entry / "idProduct").read_text().strip()
            speed = int(float((entry / "speed").read_text().strip()))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        if (vendor, product) in {("18d1", "2d00"), ("22d9", "2772")}:
            rows.append({
                "path": entry.name,
                "speed_mbps": speed,
                "vendor_product": f"{vendor}:{product}",
            })
    return sorted(rows, key=lambda row: str(row["path"]))


def bridge_progress(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    calls = [int(value) for value in re.findall(r"\bcalls=([0-9]+)\b", text)]
    recoveries = [
        int(value) for value in re.findall(r"\brecovery=([0-9]+)\b", text)
    ]
    stat = path.stat()
    return {
        "calls": max(calls, default=0),
        "file_mtime_ns": stat.st_mtime_ns,
        "file_size": stat.st_size,
        "reset_recoveries": max(recoveries, default=0),
    }


def monitor(
    mode: str,
    result_root: Path,
    stop_file: Path,
    output: Path,
    timeout_s: int,
) -> None:
    bridge_log = result_root / "dmabuf-bridge.stderr"
    started = time.monotonic()
    next_sample = started
    last_bridge_signature: tuple[int, int] | None = None
    with output.open("x", encoding="ascii", buffering=1) as stream:
        while not stop_file.exists():
            now = time.monotonic()
            if now - started >= timeout_s:
                stream.write(canonical({
                    "kind": "monitor_timeout",
                    "monotonic_ns": time.monotonic_ns(),
                    "schema": "s42-stage6-host-gate-v1",
                }) + "\n")
                raise TimeoutError("host gate monitor timed out")
            if bridge_log.exists():
                try:
                    progress = bridge_progress(bridge_log)
                    signature = (
                        int(progress["file_mtime_ns"]),
                        int(progress["file_size"]),
                    )
                    if signature != last_bridge_signature:
                        stream.write(canonical({
                            **progress,
                            "kind": "bridge_progress",
                            "monotonic_ns": time.monotonic_ns(),
                            "schema": "s42-stage6-host-gate-v1",
                            "wall_time_ns": time.time_ns(),
                        }) + "\n")
                        last_bridge_signature = signature
                except (OSError, UnicodeError, ValueError):
                    pass
            if now >= next_sample:
                row: dict[str, object] = {
                    "cpu_package_throttle_count": None,
                    "gpu": None,
                    "kind": "telemetry",
                    "mode": mode,
                    "monotonic_ns": time.monotonic_ns(),
                    "processes": processes(),
                    "schema": "s42-stage6-host-gate-v1",
                    "usb": usb_devices(),
                    "wall_time_ns": time.time_ns(),
                }
                try:
                    row["cpu_package_throttle_count"] = int(
                        CPU_THROTTLE.read_text().strip()
                    )
                except (FileNotFoundError, PermissionError, ValueError):
                    pass
                try:
                    row["gpu"] = gpu()
                except (OSError, subprocess.SubprocessError, ValueError):
                    pass
                stream.write(canonical(row) + "\n")
                next_sample = now + 1.0
            time.sleep(0.05)
        stream.write(canonical({
            "gpu_uuid": GPU_UUID,
            "kind": "monitor_complete",
            "mode": mode,
            "monotonic_ns": time.monotonic_ns(),
            "schema": "s42-stage6-host-gate-v1",
        }) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=int, default=2400)
    args = parser.parse_args()
    if args.timeout_s < 1:
        parser.error("timeout must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    monitor(
        args.mode,
        args.result_root,
        args.stop_file,
        args.output,
        args.timeout_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
