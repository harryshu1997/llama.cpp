#!/usr/bin/env python3
"""Compatibility entry point for the unified 84-request physical runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
BURST_DIR = HERE.parent / "burstgpt_gpu_cpu_op15_v1"
if not BURST_DIR.exists():
    BURST_DIR = Path(os.environ.get(
        "S41_BURST_DIR",
        HERE.parent / "input",
    ))
for path in (REPO_ROOT, BURST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_trace  # noqa: E402


def sleep_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return
        time.sleep(min(remaining_ns / 1e9, 0.01))


class DynamicSampler:
    """Historical GPU/RAPL probe retained for physical-only callers."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.pids = {"hot": 0, "cold_cpu": 0, "cold_gpu": 0}
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_pid(self, name: str, pid: int) -> None:
        with self.lock:
            self.pids[name] = pid

    def start(self) -> None:
        self.thread.start()

    def latest_gpu_snapshot(self, max_age_ns: int) -> dict[str, Any]:
        run_trace.require(max_age_ns > 0, "GPU sampler maximum age")
        with self.lock:
            run_trace.require(self.rows, "GPU sampler has no snapshot")
            gpu = dict(self.rows[-1]["gpu"])
        run_trace.require(
            time.monotonic_ns() - int(gpu["sample_t_ns"]) <= max_age_ns,
            "GPU sampler snapshot is stale",
        )
        return gpu

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        run_trace.require(not self.thread.is_alive(), "sampler stop")
        run_trace.require(self.error is None, f"sampler: {self.error}")

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    pids = dict(self.pids)
                before_ns = time.monotonic_ns()
                gpu = run_trace.gpu_snapshot()
                after_ns = time.monotonic_ns()
                gpu["sample_t_ns"] = (before_ns + after_ns) // 2
                try:
                    rapl = run_trace.rapl_package_snapshot()
                except (FileNotFoundError, PermissionError,
                        run_trace.RunError, ValueError):
                    rapl = None
                row = {
                    "gpu": gpu,
                    "pids": {
                        name: run_trace.proc_status_or_zero(pid)
                        for name, pid in pids.items()
                    },
                    "rapl_package": rapl,
                    "schema": "s41-hierarchical-resource-v1",
                    "system": run_trace.system_memory(),
                    "t_ns": time.monotonic_ns(),
                }
                with self.lock:
                    self.rows.append(row)
                self.stop_event.wait(0.2)
            with (self.output / "resource-samples.jsonl").open("xb") as stream:
                for row in self.rows:
                    stream.write(run_trace.canonical(row))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"


def wait_for_resident_release(path: Path, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="ascii"))
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
                continue
            run_trace.require(
                type(value) is dict
                and value.get("schema") == "s42-resident-release-v1"
                and value.get("status") == "OVERLAY_COMPLETE"
                and type(value.get("released_at_ns")) is int
                and value["released_at_ns"] > 0,
                "resident release receipt",
            )
            return value
        time.sleep(0.1)
    raise run_trace.RunError("resident release timeout")


def route_metrics(
    rows: list[dict[str, Any]], paid_start_ns: int
) -> dict[str, Any]:
    result = {}
    for route_name in sorted({row["route"] for row in rows}):
        route_rows = [row for row in rows if row["route"] == route_name]
        result[route_name] = {
            "completed": len(route_rows),
            "decode_s": run_trace.stats([
                row["predicted_ms"] / 1000 for row in route_rows
            ]),
            "end_s": max(
                (row["completion_ns"] - paid_start_ns) / 1e9
                for row in route_rows
            ),
            "hold_s": run_trace.stats([
                (row["dispatch_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in route_rows
            ]),
            "output_tokens": sum(
                row["output_tokens"] for row in route_rows
            ),
            "prefill_s": run_trace.stats([
                row["prompt_ms"] / 1000 for row in route_rows
            ]),
            "service_s": run_trace.stats([
                (row["completion_ns"] - row["dispatch_ns"]) / 1e9
                for row in route_rows
            ]),
        }
    return result


def main() -> int:
    from research_dev.scheduler.campaigns.burstgpt.runner import (
        main as unified_main,
    )

    return unified_main()


if __name__ == "__main__":
    raise SystemExit(main())
