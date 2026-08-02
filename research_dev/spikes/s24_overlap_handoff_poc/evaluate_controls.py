#!/usr/bin/env python3
"""Evaluate the frozen S24 equal-work C0/C1/C2/C3 benefit gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "s24-benefit-controls-v1"
RUN_SCHEMA = "s24-fixed-diamond-physical-v1"
VALIDATION_SCHEMA = "s24-physical-session-validation-v1"
CONTROL_IDS = ("C0", "C1", "C2", "C3")


class ControlError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def parse_pair(value: str) -> tuple[Path, Path]:
    run, separator, validation = value.partition("::")
    if not separator or not run or not validation:
        raise argparse.ArgumentTypeError("control pair must be RUN::VALIDATION")
    return Path(run), Path(validation)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ControlError(f"artifact is not an object: {path}")
    return value


def load_pair(
    run_path: Path,
    validation_path: Path,
    control: str,
) -> dict[str, Any]:
    run = load_json(run_path)
    validation = load_json(validation_path)
    run_sha = "sha256:" + sha256_file(run_path)
    if (
        run.get("schema") != RUN_SCHEMA
        or run.get("status") != "RUN_COMPLETE"
        or run.get("control") != control
        or validation.get("schema") != VALIDATION_SCHEMA
        or validation.get("status") != "PHYSICAL_SESSION_PASS"
        or validation.get("runtime", {}).get("sha256") != run_sha
        or validation.get("placement_gate") != "PASS"
    ):
        raise ControlError(f"{control} input is not placement-validated: {run_path}")
    return {
        "path": str(run_path),
        "sha256": run_sha,
        "validation_path": str(validation_path),
        "validation_sha256": "sha256:" + sha256_file(validation_path),
        "report": run,
    }


def request_work(
    report: dict[str, Any],
) -> dict[int, tuple[int, int, int, int, int]]:
    return {
        int(row["request_id"]): (
            int(row["prompt_length"]),
            int(row["output_steps"]),
            int(row["observed_input_tokens"]),
            int(row["observed_output_tokens"]),
            int(row["scheduled_arrival_ns"]),
        )
        for row in report["runtime"]["requests"]
    }


def pooled_op15_sizes(rows: Sequence[dict[str, Any]]) -> list[int]:
    return [
        int(size)
        for row in rows
        for size in row["report"]["summary"]["op15_pooled"]["batch_sizes"]
    ]


def priority_values(
    rows: Sequence[dict[str, Any]], priority: int, field: str,
) -> list[int]:
    return [
        int(request[field])
        for row in rows
        for request in row["report"]["runtime"]["requests"]
        if request["priority"] == priority
    ]


def summarize_control(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reports = [row["report"] for row in rows]
    makespans = [int(report["summary"]["makespan_us"]) for report in reports]
    cuda_compute = [
        int(report["summary"]["summed_cuda_island_compute_us"])
        for report in reports
    ]
    energies = [
        report["summary"].get("gpu_board_energy", {}).get("energy_j")
        if isinstance(report["summary"].get("gpu_board_energy"), dict)
        else None
        for report in reports
    ]
    route_counts = {}
    for report in reports:
        for route_id, count in report["summary"]["route_distribution"].items():
            route_counts[route_id] = route_counts.get(route_id, 0) + int(count)
    op15_sizes = pooled_op15_sizes(rows)
    return {
        "repetitions": len(rows),
        "completed_requests": sum(
            int(report["summary"]["completed_requests"]) for report in reports
        ),
        "rejected_requests": sum(
            int(report["summary"]["rejected_requests"]) for report in reports
        ),
        "slo_misses": sum(int(report["summary"]["slo_misses"]) for report in reports),
        "route_distribution": dict(sorted(route_counts.items())),
        "makespan_us": {
            "samples": makespans,
            "median": statistics.median(makespans),
        },
        "op15_batches": {
            "sizes": op15_sizes,
            "count": len(op15_sizes),
            "mean": statistics.fmean(op15_sizes) if op15_sizes else 0.0,
        },
        "cuda_island_compute_us": {
            "samples": cuda_compute,
            "median": statistics.median(cuda_compute),
            "sum": sum(cuda_compute),
        },
        "priority_0": {
            "ttft_p95_us": nearest_rank(priority_values(rows, 0, "ttft_us"), 0.95),
            "latency_p95_us": nearest_rank(priority_values(rows, 0, "latency_us"), 0.95),
            "slo_misses": sum(
                not request["slo_met"]
                for report in reports
                for request in report["runtime"]["requests"]
                if request["priority"] == 0
            ),
            "completed": len(priority_values(rows, 0, "latency_us")),
        },
        "gpu_board_energy_j": {
            "samples": energies,
            "complete": all(value is not None for value in energies),
            "median": (
                statistics.median(float(value) for value in energies)
                if energies and all(value is not None for value in energies)
                else None
            ),
        },
    }


def evaluate_controls(
    rows_by_control: dict[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    if set(rows_by_control) != set(CONTROL_IDS):
        raise ControlError("controls must define C0, C1, C2, and C3")
    if any(len(rows_by_control[control]) < 3 for control in CONTROL_IDS):
        raise ControlError("every control requires at least three repetitions")

    first = rows_by_control["C0"][0]["report"]
    expected_trace = first["trace"]["trace_hash"]
    expected_work = request_work(first)
    expected_count = len(expected_work)
    for control, rows in rows_by_control.items():
        for row in rows:
            report = row["report"]
            if report["trace"]["trace_hash"] != expected_trace:
                raise ControlError(f"{control} trace hash differs")
            if request_work(report) != expected_work:
                raise ControlError(f"{control} completed work differs")
            if report["summary"]["completed_requests"] != expected_count:
                raise ControlError(f"{control} completion denominator differs")
            if report["summary"]["rejected_requests"] != 0:
                raise ControlError(f"{control} rejected equal work")
    if any(
        set(row["report"]["summary"]["route_distribution"]) != {"R0"}
        for row in rows_by_control["C0"]
    ):
        raise ControlError("C0 executed a non-R0 route")
    c1_distributions = [
        row["report"]["summary"]["route_distribution"]
        for row in rows_by_control["C1"]
    ]
    c2_distributions = [
        row["report"]["summary"]["route_distribution"]
        for row in rows_by_control["C2"]
    ]
    if c1_distributions != c2_distributions:
        raise ControlError("C1 and C2 route distributions differ")

    summaries = {
        control: summarize_control(rows_by_control[control])
        for control in CONTROL_IDS
    }
    c1 = summaries["C1"]
    c2 = summaries["C2"]
    c0 = summaries["C0"]
    c3 = summaries["C3"]
    batch_improved = c2["op15_batches"]["mean"] > c1["op15_batches"]["mean"]
    makespan_improved = (
        c2["makespan_us"]["median"] < c1["makespan_us"]["median"]
    )
    c1_priority = c1["priority_0"]
    c2_priority = c2["priority_0"]
    priority_metrics_present = all(
        value is not None
        for value in (
            c1_priority["ttft_p95_us"],
            c1_priority["latency_p95_us"],
            c2_priority["ttft_p95_us"],
            c2_priority["latency_p95_us"],
        )
    )
    priority_gate = priority_metrics_present and (
        c2_priority["ttft_p95_us"] <= 1.05 * c1_priority["ttft_p95_us"]
        and c2_priority["latency_p95_us"] <= 1.05 * c1_priority["latency_p95_us"]
        and c2_priority["slo_misses"] <= c1_priority["slo_misses"]
    )
    cuda_gate = (
        c3["cuda_island_compute_us"]["sum"]
        < c0["cuda_island_compute_us"]["sum"]
    )
    energy_complete = all(
        summaries[control]["gpu_board_energy_j"]["complete"]
        for control in CONTROL_IDS
    )
    gates = {
        "shared_op15_batch_or_makespan": {
            "pass": batch_improved or makespan_improved,
            "batch_mean_strictly_increased": batch_improved,
            "median_makespan_strictly_reduced": makespan_improved,
            "c1_mean_batch": c1["op15_batches"]["mean"],
            "c2_mean_batch": c2["op15_batches"]["mean"],
            "c1_median_makespan_us": c1["makespan_us"]["median"],
            "c2_median_makespan_us": c2["makespan_us"]["median"],
        },
        "priority_0_regression_limit": {
            "pass": priority_gate,
            "limit_ratio": 1.05,
            "c1": c1_priority,
            "c2": c2_priority,
        },
        "c3_cuda_compute_reduction": {
            "pass": cuda_gate,
            "comparison": "strict summed compute over equal repetition counts",
            "c0_sum_us": c0["cuda_island_compute_us"]["sum"],
            "c3_sum_us": c3["cuda_island_compute_us"]["sum"],
        },
        "gpu_board_energy_reported": {
            "pass": energy_complete,
            "improvement_required": False,
            "c0_median_j": c0["gpu_board_energy_j"]["median"],
            "c3_median_j": c3["gpu_board_energy_j"]["median"],
        },
    }
    passed = all(gate["pass"] for gate in gates.values())
    return {
        "schema": SCHEMA,
        "status": "BENEFIT_GATES_PASS" if passed else "BENEFIT_GATE_FAIL",
        "trace_hash": expected_trace,
        "equal_work_request_count_per_run": expected_count,
        "inputs": {
            control: [
                {
                    "path": row["path"],
                    "sha256": row["sha256"],
                    "validation_path": row["validation_path"],
                    "validation_sha256": row["validation_sha256"],
                }
                for row in rows_by_control[control]
            ]
            for control in CONTROL_IDS
        },
        "controls": summaries,
        "gates": gates,
        "energy_scope": "RTX_4060_TI_GPU_BOARD_ONLY",
        "phone_energy": "UNKNOWN",
        "network_energy": "UNKNOWN",
        "a6000_host_energy": "UNKNOWN",
        "total_system_energy": "UNKNOWN",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for control in CONTROL_IDS:
        parser.add_argument(
            f"--{control.lower()}-pair",
            type=parse_pair,
            action="append",
            default=[],
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        rows = {
            control: [
                load_pair(run, validation, control)
                for run, validation in getattr(args, f"{control.lower()}_pair")
            ]
            for control in CONTROL_IDS
        }
        report = evaluate_controls(rows)
    except (ControlError, OSError, ValueError) as exc:
        print(json.dumps({
            "status": "FAIL", "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
