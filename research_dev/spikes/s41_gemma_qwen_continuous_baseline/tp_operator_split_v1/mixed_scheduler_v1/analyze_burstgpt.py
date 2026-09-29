#!/usr/bin/env python3
"""Analyze static and hierarchical BurstGPT routes from measured results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hierarchical_scheduler import (
    Request,
    RouteJob,
    RouteProfile,
    lower_bound_makespan_ms,
    optimize_task_assignment,
    schedule_jobs,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise ValueError(f"non-passing input: {path}")
    return value


def cold_rows(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = [
        row for row in result["request_results"]
        if row.get("role", "cold") == "cold"
    ]
    values = {row["request_index"]: row for row in rows}
    if len(values) != 17:
        raise ValueError("expected 17 cold requests")
    return values


def service_ms(rows: dict[int, dict[str, Any]]) -> dict[int, float]:
    return {
        index: row["prompt_ms"] + row["predicted_ms"]
        for index, row in rows.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-op15", type=Path, required=True)
    parser.add_argument("--gpu-fifo", type=Path, required=True)
    parser.add_argument("--gpu-lpt", type=Path)
    parser.add_argument("--hierarchical", type=Path)
    args = parser.parse_args()

    static = load(args.static_op15)
    gpu_fifo = load(args.gpu_fifo)
    gpu_lpt = load(args.gpu_lpt) if args.gpu_lpt else None
    hierarchical = load(args.hierarchical) if args.hierarchical else None
    static_rows = cold_rows(static)
    fifo_rows = cold_rows(gpu_fifo)
    if set(static_rows) != set(fifo_rows):
        raise ValueError("request identity mismatch")
    t0 = static["paid_start_ns"]
    hot_end_ms = max(
        (row["completion_ns"] - t0) / 1e6
        for row in static["request_results"] if row["role"] == "hot"
    )
    load_warm_ms = gpu_fifo["load_ms"] + gpu_fifo["warm_ms"]
    gpu_ready_ms = hot_end_ms + load_warm_ms
    requests = []
    for index, row in sorted(static_rows.items(), key=lambda item: (
        item[1]["scheduled_arrival_ns"], item[0]
    )):
        arrival_ms = (row["scheduled_arrival_ns"] - t0) / 1e6
        requests.append(Request(
            request_id=index,
            model_id="gemma-4-12b-it-q4_0",
            arrival_ms=arrival_ms,
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            deadline_ms=arrival_ms + row["slo_us"] / 1000,
            priority=0,
        ))

    cpu_profile = RouteProfile(
        "gemma_cpu_op15",
        "gemma-4-12b-it-q4_0",
        frozenset({"cpu", "usb", "op15_htp"}),
        8,
        0,
    )
    gpu_profile = RouteProfile(
        "gemma_cuda_full",
        "gemma-4-12b-it-q4_0",
        frozenset({"cuda"}),
        8,
        gpu_ready_ms,
    )
    cpu_service = service_ms(static_rows)
    gpu_service = service_ms(fifo_rows)
    cpu_jobs = [RouteJob(request, cpu_service[request.request_id])
                for request in requests]
    gpu_jobs = [RouteJob(request, gpu_service[request.request_id])
                for request in requests]
    cpu_reconstructed_ms = max(
        job.completion_ms for job in schedule_jobs(cpu_profile, cpu_jobs)
    )
    gpu_fifo_reconstructed_ms = max(
        job.completion_ms for job in schedule_jobs(gpu_profile, gpu_jobs)
    )
    gpu_lpt_reconstructed_ms = max(
        job.completion_ms for job in schedule_jobs(gpu_profile, gpu_jobs, "lpt")
    )
    assignment = optimize_task_assignment(
        requests,
        cpu_profile,
        gpu_profile,
        cpu_service,
        gpu_service,
        target_order="fifo",
        makespan_first=True,
    )
    static_makespan_ms = static["metrics"]["by_role"]["cold"][
        "makespan_s"
    ] * 1000
    switched_fifo_ms = gpu_ready_ms + gpu_fifo["metrics"]["makespan_s"] * 1000
    route_lower_bound_ms = lower_bound_makespan_ms(gpu_jobs, gpu_profile)
    result = {
        "measured": {
            "gpu_fifo_backlog_ms": gpu_fifo["metrics"]["makespan_s"] * 1000,
            "gpu_fifo_output_tokens_s": gpu_fifo["metrics"]
                ["output_throughput_tokens_s"],
            "gpu_load_warm_ms": load_warm_ms,
            "gpu_lpt_backlog_ms": (
                gpu_lpt["metrics"]["makespan_s"] * 1000
                if gpu_lpt is not None else None
            ),
            "hot_end_ms": hot_end_ms,
            "static_op15_makespan_ms": static_makespan_ms,
            "switched_fifo_composed_ms": switched_fifo_ms,
            "switched_speedup_x": static_makespan_ms / switched_fifo_ms,
            "switched_time_reduction_pct": 100 * (
                1 - switched_fifo_ms / static_makespan_ms
            ),
        },
        "profile_model": {
            "cpu_reconstructed_ms": cpu_reconstructed_ms,
            "cpu_reconstruction_error_pct": 100 * (
                cpu_reconstructed_ms / static_makespan_ms - 1
            ),
            "fifo_reconstructed_ms": gpu_fifo_reconstructed_ms,
            "fifo_reconstruction_error_pct": 100 * (
                gpu_fifo_reconstructed_ms / switched_fifo_ms - 1
            ),
            "lpt_reconstructed_ms": gpu_lpt_reconstructed_ms,
            "profile_lower_bound_ms": route_lower_bound_ms,
            "measured_fifo_gap_to_profile_lower_bound_pct": 100 * (
                switched_fifo_ms / route_lower_bound_ms - 1
            ),
        },
        "fixed_queue_assignment_oracle": {
            "cpu_request_indices": list(assignment.source_request_ids),
            "gpu_request_indices": list(assignment.target_request_ids),
            "makespan_ms": assignment.makespan_ms,
            "saving_vs_all_gpu_fifo_model_ms": (
                gpu_fifo_reconstructed_ms - assignment.makespan_ms
            ),
        },
        "schema": "s41-hierarchical-burstgpt-analysis-v1",
    }
    if hierarchical is not None:
        physical_ms = hierarchical["metrics"]["duration_s"] * 1000
        switch = hierarchical["switch"]
        physical_rows = cold_rows(hierarchical)
        physical_jobs = [
            RouteJob(
                Request(
                    request_id=index,
                    model_id="gemma-4-12b-it-q4_0",
                    arrival_ms=switch["gpu_ready_s"] * 1000,
                    input_tokens=row["input_tokens"],
                    output_tokens=row["output_tokens"],
                    deadline_ms=float("inf"),
                ),
                row["prompt_ms"] + row["predicted_ms"],
            )
            for index, row in physical_rows.items()
        ]
        physical_profile = RouteProfile(
            "gemma_cuda_full",
            "gemma-4-12b-it-q4_0",
            frozenset({"cuda"}),
            8,
            switch["gpu_ready_s"] * 1000,
        )
        physical_lb_ms = lower_bound_makespan_ms(
            physical_jobs, physical_profile
        )
        static_energy = static.get("server_energy", {}).get(
            "server_compute_device_energy_j"
        )
        physical_energy = hierarchical.get("server_energy", {}).get(
            "server_compute_device_energy_j"
        )
        result["physical_hierarchical"] = {
            "energy_reduction_pct": (
                100 * (1 - physical_energy / static_energy)
                if static_energy is not None and physical_energy is not None
                else None
            ),
            "gap_to_same_run_profile_lower_bound_pct": 100 * (
                physical_ms / physical_lb_ms - 1
            ),
            "makespan_ms": physical_ms,
            "profile_lower_bound_ms": physical_lb_ms,
            "speedup_x": static_makespan_ms / physical_ms,
            "time_reduction_pct": 100 * (
                1 - physical_ms / static_makespan_ms
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
