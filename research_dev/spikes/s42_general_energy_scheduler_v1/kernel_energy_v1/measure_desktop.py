#!/usr/bin/env python3
"""Measure one desktop command with synchronized RAPL and NVML samples."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from typing import Any

import energy_common


GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"


def rapl_root() -> Path:
    candidates = (
        Path("/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0"),
        Path("/sys/class/powercap/intel-rapl:0"),
    )
    for candidate in candidates:
        if candidate.joinpath("energy_uj").is_file():
            energy_common.require(
                candidate.joinpath("name").read_text().strip() == "package-0",
                "RAPL package identity",
            )
            return candidate
    raise energy_common.EnergyError("RAPL package is unavailable")


def gpu_sample() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,power.draw,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    energy_common.require(len(fields) == 4, "GPU sample")
    energy_common.require(fields[0] == GPU_UUID, "GPU identity")
    return {
        "gpu_memory_used_mib": int(fields[3]),
        "gpu_power_mw": int(round(float(fields[1]) * 1000)),
        "gpu_utilization_pct": int(fields[2]),
        "gpu_uuid": fields[0],
    }


class Sampler:
    def __init__(self, interval_s: float):
        self.interval_s = interval_s
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread: threading.Thread | None = None
        self.rapl = rapl_root()

    def start(self) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                monotonic_before = time.monotonic_ns()
                realtime_before = time.time_ns()
                gpu = gpu_sample()
                rapl_energy = int(self.rapl.joinpath("energy_uj").read_text())
                rapl_max = int(
                    self.rapl.joinpath("max_energy_range_uj").read_text()
                )
                monotonic_after = time.monotonic_ns()
                realtime_after = time.time_ns()
                self.rows.append({
                    **gpu,
                    "monotonic_ns": (monotonic_before + monotonic_after) // 2,
                    "rapl_energy_uj": rapl_energy,
                    "rapl_max_energy_range_uj": rapl_max,
                    "realtime_ns": (realtime_before + realtime_after) // 2,
                })
                self.stop_event.wait(self.interval_s)
        except BaseException as error:
            self.error = error

    def stop(self) -> None:
        self.stop_event.set()
        energy_common.require(self.thread is not None, "sampler thread")
        self.thread.join(timeout=15)
        energy_common.require(not self.thread.is_alive(), "sampler stop")
        if self.error is not None:
            raise self.error


def parse_environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        energy_common.require(
            bool(separator) and bool(key) and "\x00" not in value,
            "invalid environment entry",
        )
        result[key] = item
    return result


def nested_int(value: Any, path: str) -> int:
    current = value
    for component in path.split("."):
        energy_common.require(type(current) is dict, "window JSON path")
        current = current.get(component)
    energy_common.require(type(current) is int, "window JSON integer")
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--sample-interval-s", type=float, default=0.1)
    parser.add_argument(
        "--window",
        choices=("stdout-realtime", "process-monotonic", "json-monotonic"),
        default="stdout-realtime",
    )
    parser.add_argument("--window-json", type=Path)
    parser.add_argument("--start-field", default="campaign_start_monotonic_ns")
    parser.add_argument("--end-field", default="campaign_end_monotonic_ns")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    energy_common.require(args.output.is_absolute(), "absolute output path")
    energy_common.require(not args.output.exists(), "output already exists")
    energy_common.require(args.command, "missing command")
    energy_common.require(
        0.02 <= args.sample_interval_s <= 2.0, "sample interval"
    )
    if args.window == "json-monotonic":
        energy_common.require(args.window_json is not None, "window JSON")

    sampler = Sampler(args.sample_interval_s)
    sampler.start()
    time.sleep(max(0.5, 2 * args.sample_interval_s))
    process_start_monotonic_ns = time.monotonic_ns()
    process_start_realtime_ns = time.time_ns()
    completed = subprocess.run(
        args.command,
        env={**os.environ, **parse_environment(args.env)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    process_end_monotonic_ns = time.monotonic_ns()
    process_end_realtime_ns = time.time_ns()
    time.sleep(max(0.5, 2 * args.sample_interval_s))
    sampler.stop()

    if args.window == "stdout-realtime":
        start_ns, end_ns = energy_common.parse_desktop_window(completed.stdout)
        clock = "realtime_ns"
    elif args.window == "process-monotonic":
        start_ns = process_start_monotonic_ns
        end_ns = process_end_monotonic_ns
        clock = "monotonic_ns"
    else:
        raw_window = json.loads(args.window_json.read_text())
        start_ns = nested_int(raw_window, args.start_field)
        end_ns = nested_int(raw_window, args.end_field)
        clock = "monotonic_ns"
    energy_common.require(end_ns > start_ns, "paid window")

    gpu_energy_j = energy_common.integrate_power(
        [
            (int(row[clock]), int(row["gpu_power_mw"]) / 1000.0)
            for row in sampler.rows
        ],
        start_ns,
        end_ns,
    )
    cpu_energy_j = energy_common.integrate_rapl(
        sampler.rows, start_ns, end_ns, clock
    )
    duration_s = (end_ns - start_ns) / 1e9
    result = {
        "case_id": args.case_id,
        "clock": clock,
        "command": args.command,
        "cpu_package_average_power_w": cpu_energy_j / duration_s,
        "cpu_package_energy_j": cpu_energy_j,
        "duration_s": duration_s,
        "end_ns": end_ns,
        "gpu_board_average_power_w": gpu_energy_j / duration_s,
        "gpu_board_energy_j": gpu_energy_j,
        "gpu_uuid": GPU_UUID,
        "hostname": socket.gethostname(),
        "process_end_monotonic_ns": process_end_monotonic_ns,
        "process_end_realtime_ns": process_end_realtime_ns,
        "process_returncode": completed.returncode,
        "process_start_monotonic_ns": process_start_monotonic_ns,
        "process_start_realtime_ns": process_start_realtime_ns,
        "sample_count": len(sampler.rows),
        "samples": sampler.rows,
        "schema": "s42-desktop-energy-case-v1",
        "start_ns": start_ns,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "stdout": completed.stdout,
        "window": args.window,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(energy_common.canonical(result))
    return 0 if completed.returncode == 0 else completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except energy_common.EnergyError as error:
        print(f"S42_DESKTOP_ENERGY_ERROR: {error}")
        raise SystemExit(2)
