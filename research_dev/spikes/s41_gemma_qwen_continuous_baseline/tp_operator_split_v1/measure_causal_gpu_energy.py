#!/usr/bin/env python3
"""Measure GPU-board energy for a corrected causal layer probe.

The phone contribution is deliberately out of scope. The runner arrival-stamps
one continuous nvidia-smi stream, integrates power only between markers emitted
by the host, and alternates isolated monolithic and split executions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
from typing import Any


MARKER_RE = re.compile(
    r"ENERGY_WINDOW_(START|END) unix_ns=(\d+) mode=(\w+) n=(\d+)"
)
RESULT_RE = re.compile(
    r"^(?:RESULT|CUDA_ONLY|SPLIT_ENERGY) .*$", re.MULTILINE
)


class PowerSampler:
    def __init__(self, gpu: int, period_ms: int) -> None:
        self.gpu = gpu
        self.period_ms = period_ms
        self.rows: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def _read(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                fields = [field.strip() for field in line.strip().split(",")]
                if len(fields) != 6:
                    continue
                try:
                    row = {
                        "arrival_ns": time.time_ns(),
                        "power_w": float(fields[0]),
                        "power_limit_w": float(fields[1]),
                        "utilization_pct": float(fields[2]),
                        "graphics_mhz": float(fields[3]),
                        "memory_mhz": float(fields[4]),
                        "pstate": fields[5],
                    }
                except ValueError:
                    continue
                self.rows.append(row)
        except OSError as exc:
            self.error = str(exc)

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu),
                "--query-gpu=power.draw,power.limit,utilization.gpu,"
                "clocks.current.graphics,clocks.current.memory,pstate",
                "--format=csv,noheader,nounits",
                "-lms",
                str(self.period_ms),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RuntimeError("power sampler thread did not terminate")
        if self.error is not None:
            raise RuntimeError(f"power sampler failed: {self.error}")

    def integrate(self, start_ns: int, end_ns: int) -> dict[str, Any]:
        if end_ns <= start_ns:
            raise RuntimeError("non-positive energy window")
        rows = sorted(self.rows, key=lambda row: row["arrival_ns"])
        if not rows or rows[0]["arrival_ns"] > start_ns:
            raise RuntimeError("power samples do not bracket window start")
        if rows[-1]["arrival_ns"] < end_ns:
            raise RuntimeError("power samples do not bracket window end")

        energy_j = 0.0
        max_gap_s = 0.0
        for first, second in zip(rows, rows[1:]):
            lo = max(first["arrival_ns"], start_ns)
            hi = min(second["arrival_ns"], end_ns)
            if hi <= lo:
                continue
            duration_s = (hi - lo) / 1e9
            energy_j += first["power_w"] * duration_s
            max_gap_s = max(
                max_gap_s,
                (second["arrival_ns"] - first["arrival_ns"]) / 1e9,
            )

        window = [
            row for row in rows
            if start_ns <= row["arrival_ns"] <= end_ns
        ]
        if len(window) < 20:
            raise RuntimeError(f"too few in-window power samples: {len(window)}")
        duration_s = (end_ns - start_ns) / 1e9
        return {
            "duration_s": duration_s,
            "energy_j": energy_j,
            "average_power_w": energy_j / duration_s,
            "sample_count": len(window),
            "max_sample_gap_s": max_gap_s,
            "average_utilization_pct": statistics.fmean(
                row["utilization_pct"] for row in window
            ),
            "graphics_mhz_min": min(row["graphics_mhz"] for row in window),
            "graphics_mhz_max": max(row["graphics_mhz"] for row in window),
            "memory_mhz_min": min(row["memory_mhz"] for row in window),
            "memory_mhz_max": max(row["memory_mhz"] for row in window),
            "power_limit_w_values": sorted(
                {row["power_limit_w"] for row in window}
            ),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_window(output: str, expected_mode: str, expected_n: int) -> tuple[int, int]:
    markers: dict[str, tuple[int, str, int]] = {}
    for kind, timestamp, mode, count in MARKER_RE.findall(output):
        if kind in markers:
            raise RuntimeError(f"duplicate {kind} marker")
        markers[kind] = (int(timestamp), mode, int(count))
    if set(markers) != {"START", "END"}:
        raise RuntimeError("missing energy-window marker")
    for kind in ("START", "END"):
        _, mode, count = markers[kind]
        if mode != expected_mode or count != expected_n:
            raise RuntimeError(f"{kind} marker does not match command")
    return markers["START"][0], markers["END"][0]


def run_probe(
    *,
    binary: Path,
    output_dir: Path,
    index: int,
    mode: str,
    iterations: int,
    n_kv: int,
    phone_columns: int,
    weight_type: str,
    shape: str,
    timeout_s: int,
    sampler: PowerSampler,
    environment: dict[str, str],
    activation_return: bool,
) -> dict[str, Any]:
    if activation_return:
        paid_mode = "cuda_energy" if mode == "monolithic" else "split_energy"
        late_path = "f32_async" if mode == "monolithic" else "f32_dual"
        marker_mode = "cuda" if mode == "monolithic" else "split"
        command = [
            str(binary),
            str(iterations),
            str(phone_columns),
            weight_type,
            shape,
            "f16",
            paid_mode,
            "f16",
            late_path,
        ]
    else:
        marker_mode = mode
        command = [
            str(binary),
            str(n_kv),
            str(iterations),
            str(phone_columns if mode == "split" else 0),
            weight_type,
            mode,
            shape,
        ]
    started_ns = time.time_ns()
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        timeout=timeout_s,
        check=False,
    )
    ended_ns = time.time_ns()
    raw_path = output_dir / f"energy_{index:02d}_{mode}.log"
    raw_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} command failed with {completed.returncode}: {raw_path}"
        )
    start_ns, end_ns = parse_window(
        completed.stdout, marker_mode, iterations
    )
    time.sleep(0.3)
    energy = sampler.integrate(start_ns, end_ns)
    result_lines = RESULT_RE.findall(completed.stdout)
    if len(result_lines) != 1:
        raise RuntimeError(f"expected one RESULT line in {raw_path}")
    energy.update(
        {
            "index": index,
            "mode": mode,
            "iterations": iterations,
            "phone_columns": phone_columns if mode == "split" else 0,
            "command": command,
            "process_started_ns": started_ns,
            "window_started_ns": start_ns,
            "window_ended_ns": end_ns,
            "process_ended_ns": ended_ns,
            "energy_per_iteration_j": energy["energy_j"] / iterations,
            "result_line": result_lines[0],
            "raw_log": str(raw_path),
            "raw_log_sha256": sha256(raw_path),
        }
    )
    return energy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--period-ms", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--monolithic-iters", type=int, default=12000)
    parser.add_argument("--split-iters", type=int, default=5000)
    parser.add_argument("--n-kv", type=int, default=512)
    parser.add_argument("--phone-columns", type=int, default=256)
    parser.add_argument("--weight-type", default="q4_0")
    parser.add_argument(
        "--shape",
        choices=("qwen3_14b", "gemma4_12b"),
        default="qwen3_14b",
    )
    parser.add_argument("--aoa-bus", required=True)
    parser.add_argument("--aoa-address", required=True)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--expected-graphics-mhz", type=float)
    parser.add_argument("--expected-memory-mhz", type=float)
    parser.add_argument("--activation-return", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    environment["S41_AOA_BUS"] = args.aoa_bus
    environment["S41_AOA_ADDRESS"] = args.aoa_address

    schedule: list[str] = []
    for repeat in range(args.repeats):
        pair = ["monolithic", "split"]
        if repeat % 2:
            pair.reverse()
        schedule.extend(pair)

    sampler = PowerSampler(args.gpu, args.period_ms)
    sampler.start()
    time.sleep(1.0)
    runs: list[dict[str, Any]] = []
    try:
        for index, mode in enumerate(schedule, start=1):
            iterations = (
                args.monolithic_iters if mode == "monolithic"
                else args.split_iters
            )
            run = run_probe(
                binary=args.binary,
                output_dir=args.output_dir,
                index=index,
                mode=mode,
                iterations=iterations,
                n_kv=args.n_kv,
                phone_columns=args.phone_columns,
                weight_type=args.weight_type,
                shape=args.shape,
                timeout_s=args.timeout_s,
                sampler=sampler,
                environment=environment,
                activation_return=args.activation_return,
            )
            runs.append(run)
            if (
                args.expected_graphics_mhz is not None
                and (
                    run["graphics_mhz_min"] != args.expected_graphics_mhz
                    or run["graphics_mhz_max"] != args.expected_graphics_mhz
                )
            ):
                raise RuntimeError("graphics clock left the expected lock")
            if (
                args.expected_memory_mhz is not None
                and (
                    run["memory_mhz_min"] != args.expected_memory_mhz
                    or run["memory_mhz_max"] != args.expected_memory_mhz
                )
            ):
                raise RuntimeError("memory clock left the expected lock")
            print(
                f"{mode}: {run['duration_s']:.3f} s, "
                f"{run['average_power_w']:.3f} W, "
                f"{run['energy_per_iteration_j']:.6f} J/iteration",
                flush=True,
            )
            time.sleep(0.7)
    finally:
        sampler.stop()

    samples_path = args.output_dir / "GPU_POWER_SAMPLES.jsonl"
    samples_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in sampler.rows
        ),
        encoding="utf-8",
    )
    by_mode = {
        mode: [run for run in runs if run["mode"] == mode]
        for mode in ("monolithic", "split")
    }
    medians = {
        mode: statistics.median(
            run["energy_per_iteration_j"] for run in mode_runs
        )
        for mode, mode_runs in by_mode.items()
    }
    output = {
        "schema": (
            "s41-activation-gpu-energy-v1"
            if args.activation_return
            else "s41-causal-gpu-energy-v1"
        ),
        "scope": "GPU_BOARD_ONLY",
        "method": "100ms nvidia-smi arrival timestamps with ZOH integration",
        "schedule": schedule,
        "shape": args.shape,
        "weight_type": args.weight_type,
        "runs": runs,
        "median_energy_per_iteration_j": medians,
        "split_gpu_energy_change_fraction": (
            medians["split"] / medians["monolithic"] - 1.0
        ),
        "binary": str(args.binary),
        "binary_sha256": sha256(args.binary),
        "power_samples": str(samples_path),
        "power_samples_sha256": sha256(samples_path),
        "source_sha256": {
            name: sha256(Path(__file__).resolve().parent / name)
            for name in (
                (
                    "causal_activation_worker.cpp"
                    if args.activation_return
                    else "causal_ffn_worker.cpp"
                ),
                (
                    "causal_activation_host.cpp"
                    if args.activation_return
                    else "causal_layer_host.cpp"
                ),
                "causal_ffn_protocol.h",
                "causal_quantized_weights.h",
                "measure_causal_gpu_energy.py",
            )
        },
    }
    result_path = args.output_dir / "GPU_ENERGY.json"
    result_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
