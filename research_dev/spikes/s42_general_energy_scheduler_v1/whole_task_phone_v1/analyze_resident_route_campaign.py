#!/usr/bin/env python3
"""Fit task-route costs and exercise the scheduler on a physical campaign."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT)]

from research_dev.scheduler import (  # noqa: E402
    ProfileBundle,
    Request,
    UnifiedScheduler,
    decision_to_json,
)


RESULT_SCHEMA = "s42-whole-task-resident-route-result-v1"
PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"
ANALYSIS_SCHEMA = "s42-whole-task-resident-route-analysis-v2"
MODEL_ID = "llama-3.2-1b-instruct-q4_0"
WORKLOAD_ID = "llama-1b-resident-task"
ROUTES = ("desktop-cuda", "desktop-cpu", "phone-adreno")
BOUNDARY_ID = (
    "incremental-cpu-package-gpu-board-whole-phone-above-resident-idle-v1"
)


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    require(bool(values), "empty mean")
    return sum(values) / len(values)


def percentile(values: list[float], fraction: float) -> float:
    require(bool(values), "empty percentile")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def solve_linear(
    matrix: list[list[float]], vector: list[float]
) -> list[float] | None:
    size = len(vector)
    augmented = [
        matrix[index][:] + [vector[index]] for index in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1.0e-12:
            return None
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        scale = augmented[column][column]
        augmented[column] = [
            value / scale for value in augmented[column]
        ]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index]
                - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[index][-1] for index in range(size)]


def fit_nonnegative(
    rows: list[dict[str, Any]], target: Callable[[dict[str, Any]], float]
) -> dict[str, Any]:
    features = [
        (1.0, float(row["input_tokens"]), float(row["output_tokens"]))
        for row in rows
    ]
    targets = [target(row) for row in rows]
    require(len({(row[1], row[2]) for row in features}) >= 3, "fit geometry")
    best: tuple[float, tuple[float, ...]] | None = None
    for count in range(1, 4):
        for active in itertools.combinations(range(3), count):
            gram = [
                [
                    sum(row[left] * row[right] for row in features)
                    for right in active
                ]
                for left in active
            ]
            rhs = [
                sum(row[index] * value for row, value in zip(features, targets))
                for index in active
            ]
            solved = solve_linear(gram, rhs)
            if solved is None or any(value < -1e-9 for value in solved):
                continue
            coefficients = [0.0, 0.0, 0.0]
            for index, value in zip(active, solved):
                coefficients[index] = max(0.0, value)
            squared_error = sum(
                (sum(a * b for a, b in zip(coefficients, row)) - value) ** 2
                for row, value in zip(features, targets)
            )
            candidate = (squared_error, tuple(coefficients))
            if best is None or candidate < best:
                best = candidate
    require(best is not None, "nonnegative fit")
    coefficients = tuple(max(0, round(value)) for value in best[1])

    def predict(row: dict[str, Any]) -> int:
        return max(
            1,
            coefficients[0]
            + coefficients[1] * row["input_tokens"]
            + coefficients[2] * row["output_tokens"],
        )

    details = []
    lower_error_ppm = 0
    upper_error_ppm = 0
    upper_add = 0
    for row, actual_float in zip(rows, targets):
        actual = max(1, round(actual_float))
        predicted = predict(row)
        if predicted > actual:
            lower_error_ppm = max(
                lower_error_ppm,
                math.ceil(1_000_000 * (predicted - actual) / predicted),
            )
        if actual > predicted:
            upper_error_ppm = max(
                upper_error_ppm,
                math.ceil(1_000_000 * (actual - predicted) / predicted),
            )
            upper_add = max(upper_add, actual - predicted)
        details.append({
            "actual": actual,
            "error_pct": 100.0 * (predicted - actual) / actual,
            "mixed_request_index": row["mixed_request_index"],
            "predicted": predicted,
        })
    return {
        "coefficients": {
            "fixed": coefficients[0],
            "input_token": coefficients[1],
            "output_token": coefficients[2],
        },
        "details": details,
        "lower_error_ppm": min(1_000_000, lower_error_ppm),
        "max_abs_error_pct": max(abs(row["error_pct"]) for row in details),
        "upper_add": upper_add,
        "upper_error_ppm": min(1_000_000, upper_error_ppm),
    }


def affine(coefficients: dict[str, int]) -> dict[str, object]:
    return {
        "kind": "affine_tokens_v1",
        "fixed": coefficients["fixed"],
        "input_token": coefficients["input_token"],
        "output_token": coefficients["output_token"],
    }


def route_profile(
    route: str,
    latency: dict[str, Any],
    energy: dict[str, Any],
    evidence_ids: tuple[str, ...],
    quality: str,
) -> dict[str, Any]:
    if route == "desktop-cpu":
        resources = {"desktop-cpu": 1}
        leases: list[dict[str, Any]] = []
        baseline = True
        server_busy_ppm = 1_000_000
        server_memory_bytes = 770_928_288
    elif route == "desktop-cuda":
        resources = {"cuda0": 1}
        leases = []
        baseline = False
        server_busy_ppm = 100_000
        server_memory_bytes = 770_928_288
    else:
        resources = {"op15-adreno": 1, "usb-token-rpc": 1}
        service = affine(latency["coefficients"])
        leases = [
            {
                "lease_id": "phone-full-task",
                "resource_id": "op15-adreno",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": service,
                "duration_ucb_add_us": latency["upper_add"],
            },
            {
                "lease_id": "token-dispatch",
                "resource_id": "usb-token-rpc",
                "slots": 1,
                "start_offset_us": 0,
                "duration_us": 1_000,
            },
        ]
        baseline = False
        server_busy_ppm = 20_000
        server_memory_bytes = 0
    overlap: dict[str, object] = {"status": "not_applicable"}
    return {
        "baseline": baseline,
        "energy": {
            "boundary_id": BOUNDARY_ID,
            "cost_uj": affine(energy["coefficients"]),
            "lower_error_ppm": energy["lower_error_ppm"],
            "status": "measured",
            "upper_error_ppm": energy["upper_error_ppm"],
        },
        "evidence_ids": list(evidence_ids),
        "granularity": "task",
        "latency": {
            "cost_us": affine(latency["coefficients"]),
            "measured": True,
            "sample_count": len(latency["details"]),
            "ucb_add_us": latency["upper_add"],
        },
        "overlap": overlap,
        "placement_verified": True,
        "quality_class": quality,
        "resident": True,
        "resource_leases": leases,
        "resource_slots": resources,
        "route_id": route,
        "server_busy_ppm": server_busy_ppm,
        "server_memory_bytes": server_memory_bytes,
        "workload_id": WORKLOAD_ID,
    }


def build_profile(
    fits: dict[str, dict[str, Any]],
    evidence_ids: tuple[str, ...],
    quality: str,
    cuda_tail_uj: int = 0,
) -> dict[str, Any]:
    route_fits = json.loads(json.dumps(fits))
    route_fits["desktop-cuda"]["energy"]["coefficients"][
        "fixed"
    ] += cuda_tail_uj
    return {
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 20_000_000,
            "max_exposed_join_wait_ppm": 50_000,
        },
        "profile_id": (
            "s42-op15-whole-task-llama1b-resident-isolated-v2"
            if cuda_tail_uj
            else "s42-op15-whole-task-llama1b-resident-tail-reused-v2"
        ),
        "resources": [
            {
                "capacity": 1,
                "identity": "i9-12900k-package0",
                "kind": "cpu",
                "ready": True,
                "resource_id": "desktop-cpu",
            },
            {
                "capacity": 1,
                "identity": "rtx4060ti-gpu-3d43c513",
                "kind": "gpu",
                "ready": True,
                "resource_id": "cuda0",
            },
            {
                "capacity": 1,
                "identity": "op15-adreno840-3c15au002cl00000",
                "kind": "phone-gpu",
                "ready": True,
                "resource_id": "op15-adreno",
            },
            {
                "capacity": 1,
                "identity": "adb-token-transport-3c15au002cl00000",
                "kind": "transport",
                "ready": True,
                "resource_id": "usb-token-rpc",
            },
        ],
        "routes": [
            route_profile(
                route,
                route_fits[route]["latency"],
                route_fits[route]["energy"],
                evidence_ids,
                quality,
            )
            for route in ("desktop-cpu", "desktop-cuda", "phone-adreno")
        ],
        "schema": PROFILE_SCHEMA,
        "trace_workload_map": {MODEL_ID: WORKLOAD_ID},
    }


def composite_usb_profile(profile: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(profile))
    value["profile_id"] = "s42-op15-whole-task-llama1b-composite-usb-v2"
    cpu = next(
        row for row in value["resources"]
        if row["resource_id"] == "desktop-cpu"
    )
    cpu["capacity"] = 4
    value["resources"] = [
        row for row in value["resources"]
        if row["resource_id"] != "usb-token-rpc"
    ]
    value["resources"].extend((
        {
            "capacity": 1,
            "identity": "op15-ncm-3c15au002cl00000",
            "kind": "transport",
            "ready": True,
            "resource_id": "op15-ncm",
        },
        {
            "capacity": 1,
            "identity": "op15-functionfs-3c15au002cl00000",
            "kind": "transport",
            "ready": True,
            "resource_id": "op15-functionfs",
        },
        {
            "capacity": 2,
            "identity": "4060ti-op15-composite-usb-3c15au002cl00000",
            "kind": "usb",
            "ready": True,
            "resource_id": "desktop-usb-root",
        },
    ))
    phone = next(
        row for row in value["routes"]
        if row["route_id"] == "phone-adreno"
    )
    service = phone["latency"]["cost_us"]
    upper_add = phone["latency"]["ucb_add_us"]
    phone["resource_slots"] = {
        "desktop-usb-root": 1,
        "op15-adreno": 1,
        "op15-ncm": 1,
    }
    phone["resource_leases"] = [
        {
            "duration_ucb_add_us": upper_add,
            "duration_us": service,
            "lease_id": lease_id,
            "resource_id": resource_id,
            "slots": 1,
            "start_offset_us": 0,
        }
        for lease_id, resource_id in (
            ("phone-full-task", "op15-adreno"),
            ("ncm-full-task", "op15-ncm"),
            ("composite-usb-full-task", "desktop-usb-root"),
        )
    ]
    ProfileBundle.from_json(value)
    return value


def scheduler_decisions(
    profile: dict[str, Any], representative: dict[str, Any]
) -> dict[str, Any]:
    def request(request_id: str) -> Request:
        return Request(
            request_id=request_id,
            workload_id=WORKLOAD_ID,
            arrival_us=0,
            deadline_us=30_000_000,
            input_tokens=representative["input_tokens"],
            output_tokens=representative["output_tokens"],
            quality_requirement=profile["routes"][0]["quality_class"],
        )

    bundle = ProfileBundle.from_json(profile)
    idle = UnifiedScheduler((bundle,), "enforce")
    idle_decision = idle.schedule(request("gpu-idle"))

    busy = UnifiedScheduler((bundle,), "enforce")
    busy.reserve_external_resource(
        "cuda0", "analysis-gpu-busy", 0, 25_000_000
    )
    busy_decision = busy.schedule(request("gpu-busy"))

    full = UnifiedScheduler((bundle,), "enforce")
    full.set_resource_ready("cuda0", False, 0)
    full_decision = full.schedule(request("gpu-memory-unavailable"))
    return {
        "gpu_busy": decision_to_json(busy_decision),
        "gpu_idle": decision_to_json(idle_decision),
        "gpu_memory_unavailable": decision_to_json(full_decision),
    }


def cuda_tail_summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(
        value.get("schema") == "s42-cuda-resident-tail-result-v1"
        and value.get("status") == "PASS"
        and value.get("model_id") == MODEL_ID,
        "CUDA tail result",
    )
    cases = value.get("cases")
    require(type(cases) is list and len(cases) >= 3, "CUDA tail coverage")
    energies = [row["tail_gpu_dynamic_energy_j"] for row in cases]
    durations = [row["tail_duration_s"] for row in cases]
    return {
        "dynamic_energy_j": {
            "max": max(energies),
            "mean": mean(energies),
            "min": min(energies),
        },
        "duration_s": {
            "max": max(durations),
            "mean": mean(durations),
            "min": min(durations),
        },
        "path": str(path),
        "sample_count": len(cases),
        "sha256": digest_file(path),
    }


def analyze(
    result_path: Path, cuda_tail_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="ascii"))
    require(
        result.get("schema") == RESULT_SCHEMA and result.get("status") == "PASS",
        "campaign result",
    )
    cases = result.get("cases")
    require(type(cases) is list and cases, "campaign cases")
    idle = [row for row in cases if row.get("route") == "idle"]
    route_rows = {
        route: [row for row in cases if row.get("route") == route]
        for route in ROUTES
    }
    require(idle and all(route_rows.values()), "route coverage")
    route_windows = sorted(
        (
            row.get("paid_start_monotonic_ns"),
            row.get("paid_end_monotonic_ns"),
        )
        for rows in route_rows.values()
        for row in rows
    )
    require(
        all(
            type(start) is int and type(end) is int and end > start
            for start, end in route_windows
        )
        and all(
            left[1] <= right[0]
            for left, right in zip(route_windows, route_windows[1:])
        ),
        "isolated non-overlapping request windows",
    )
    idle_power = {
        "cpu_package_w": mean([
            row["cpu_package_average_power_w"] for row in idle
        ]),
        "gpu_board_w": mean([
            row["gpu_board_average_power_w"] for row in idle
        ]),
        "phone_w": mean([
            row["phone"]["whole_phone_average_power_w"] for row in idle
        ]),
    }
    idle_power["accounted_fleet_w"] = sum(idle_power.values())

    quality = "exact"
    by_request: dict[int, set[str]] = {}
    for row in cases:
        if row.get("route") not in ROUTES:
            continue
        by_request.setdefault(row["mixed_request_index"], set()).add(
            row["response"]["content_sha256"]
        )
    if any(len(values) != 1 for values in by_request.values()):
        quality = "bounded_numeric"

    fits: dict[str, dict[str, Any]] = {}
    response_window_fits: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for route, rows in route_rows.items():
        latency = fit_nonnegative(
            rows, lambda row: row["duration_s"] * 1_000_000
        )
        response_window_energy = fit_nonnegative(
            rows, lambda row: row["accounted_fleet_energy_j"] * 1_000_000
        )
        energy = fit_nonnegative(rows, lambda row: max(
            1.0,
            (
                row["accounted_fleet_energy_j"]
                - idle_power["accounted_fleet_w"] * row["duration_s"]
            ) * 1_000_000,
        ))
        fits[route] = {"energy": energy, "latency": latency}
        response_window_fits[route] = {
            "energy": response_window_energy,
            "latency": latency,
        }
        marginal = [
            row["accounted_fleet_energy_j"]
            - idle_power["accounted_fleet_w"] * row["duration_s"]
            for row in rows
        ]
        summaries[route] = {
            "accounted_fleet_energy_j": {
                "mean": mean([row["accounted_fleet_energy_j"] for row in rows]),
                "p90": percentile(
                    [row["accounted_fleet_energy_j"] for row in rows], 0.9
                ),
            },
            "duration_s": {
                "mean": mean([row["duration_s"] for row in rows]),
                "p90": percentile([row["duration_s"] for row in rows], 0.9),
            },
            "marginal_energy_above_resident_idle_j": {
                "mean": mean(marginal),
                "p90": percentile(marginal, 0.9),
            },
            "sample_count": len(rows),
        }

    evidence_ids = [f"sha256:{digest_file(result_path)}"]
    tail = None
    cuda_tail_uj = 0
    if cuda_tail_path is not None:
        tail = cuda_tail_summary(cuda_tail_path)
        evidence_ids.append(f"sha256:{tail['sha256']}")
        cuda_tail_uj = round(tail["dynamic_energy_j"]["mean"] * 1_000_000)
    response_only_profile = build_profile(
        fits, tuple(evidence_ids), quality
    )
    profile = build_profile(
        fits,
        tuple(evidence_ids),
        quality,
        cuda_tail_uj=cuda_tail_uj,
    )
    representative = min(
        route_rows["desktop-cpu"],
        key=lambda row: abs(row["input_tokens"] - 915)
        + abs(row["output_tokens"] - 292),
    )
    decisions = scheduler_decisions(profile, representative)
    response_only_decisions = scheduler_decisions(
        response_only_profile, representative
    )
    cpu_mean = summaries["desktop-cpu"]
    cuda_mean = summaries["desktop-cuda"]
    phone_mean = summaries["phone-adreno"]

    def saving(reference: float, candidate: float) -> float:
        return 100.0 * (reference - candidate) / reference

    cuda_response_energy_j = cuda_mean["accounted_fleet_energy_j"]["mean"]
    cuda_lifecycle_energy_j = cuda_response_energy_j
    if tail is not None:
        cuda_lifecycle_energy_j += tail["dynamic_energy_j"]["mean"]
    analysis = {
        "campaign_result": {
            "path": str(result_path),
            "sha256": digest_file(result_path),
        },
        "comparisons": {
            "phone_vs_cpu": {
                "accounted_fleet_energy_saving_pct": saving(
                    cpu_mean["accounted_fleet_energy_j"]["mean"],
                    phone_mean["accounted_fleet_energy_j"]["mean"],
                ),
                "latency_reduction_pct": saving(
                    cpu_mean["duration_s"]["mean"],
                    phone_mean["duration_s"]["mean"],
                ),
            },
            "phone_vs_cuda": {
                "cuda_lifecycle_energy_j": cuda_lifecycle_energy_j,
                "cuda_response_window_energy_j": cuda_response_energy_j,
                "lifecycle_energy_saving_pct": saving(
                    cuda_lifecycle_energy_j,
                    phone_mean["accounted_fleet_energy_j"]["mean"],
                ),
                "phone_response_window_energy_j": (
                    phone_mean["accounted_fleet_energy_j"]["mean"]
                ),
                "response_latency_ratio": (
                    phone_mean["duration_s"]["mean"]
                    / cuda_mean["duration_s"]["mean"]
                ),
            },
        },
        "cuda_post_response_tail": tail,
        "energy_accounting": {
            "cuda_post_response": (
                "dynamic GPU board energy above measured resident P8 idle"
            ),
            "request_window": (
                "diagnostic CPU package plus GPU board plus whole-phone energy"
            ),
            "resident_idle_after_response": (
                "not assigned to a completed request"
            ),
            "route_selection": (
                "incremental request energy above measured resident idle"
            ),
        },
        "evidence_ids": evidence_ids,
        "fits": fits,
        "idle_power": idle_power,
        "model_id": MODEL_ID,
        "profile_variants": {
            "cuda_epoch_open": profile["profile_id"],
            "cuda_epoch_reused": response_only_profile["profile_id"],
        },
        "quality_class": quality,
        "qualification": {
            "idle_baseline_sample_count": len(idle),
            "idle_baseline_subtracted": True,
            "max_concurrent_measured_requests": 1,
            "request_windows_non_overlapping": True,
            "status": "PASS",
        },
        "response_window_fits": response_window_fits,
        "route_summaries": summaries,
        "scheduler_decisions": decisions,
        "scheduler_lifecycle_policy": {
            "cuda_epoch_open": (
                "use when no active or queued CUDA epoch has already been "
                "charged its tail"
            ),
            "cuda_epoch_reused": (
                "use when the request joins a CUDA epoch whose tail is "
                "already accounted"
            ),
            "default_if_unknown": "cuda_epoch_open",
            "selection_point": "atomic precommit runtime snapshot",
        },
        "scheduler_tail_reused_counterfactual": response_only_decisions,
        "schema": ANALYSIS_SCHEMA,
        "status": "PASS",
    }
    return analysis, profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cuda-tail-result", type=Path)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--tail-reused-profile-output", type=Path)
    parser.add_argument("--composite-usb-profile-output", type=Path)
    args = parser.parse_args()
    output_paths = [args.analysis_output, args.profile_output]
    if args.tail_reused_profile_output is not None:
        output_paths.append(args.tail_reused_profile_output)
    if args.composite_usb_profile_output is not None:
        output_paths.append(args.composite_usb_profile_output)
    require(
        all(path.is_absolute() and not path.exists() for path in output_paths),
        "output paths",
    )
    analysis, profile = analyze(args.result, args.cuda_tail_result)
    args.analysis_output.write_bytes(canonical(analysis))
    args.profile_output.write_bytes(canonical(profile))
    if args.tail_reused_profile_output is not None:
        tail_reused_profile = build_profile(
            analysis["fits"],
            tuple(analysis["evidence_ids"]),
            analysis["quality_class"],
        )
        args.tail_reused_profile_output.write_bytes(
            canonical(tail_reused_profile)
        )
    if args.composite_usb_profile_output is not None:
        args.composite_usb_profile_output.write_bytes(
            canonical(composite_usb_profile(profile))
        )
    print(json.dumps({
        "analysis": str(args.analysis_output),
        "composite_usb_profile": (
            None
            if args.composite_usb_profile_output is None
            else str(args.composite_usb_profile_output)
        ),
        "gpu_busy_route": analysis["scheduler_decisions"]["gpu_busy"]["route_id"],
        "gpu_idle_route": analysis["scheduler_decisions"]["gpu_idle"]["route_id"],
        "gpu_memory_unavailable_route": analysis["scheduler_decisions"]["gpu_memory_unavailable"]["route_id"],
        "profile": str(args.profile_output),
        "tail_reused_profile": (
            None
            if args.tail_reused_profile_output is None
            else str(args.tail_reused_profile_output)
        ),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as error:
        print(f"S42_WHOLE_TASK_ANALYSIS_ERROR: {error}")
        raise SystemExit(2)
