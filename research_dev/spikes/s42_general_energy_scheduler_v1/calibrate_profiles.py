#!/usr/bin/env python3
"""Build an S42 route profile from paired physical trace results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    PROFILE_SCHEMA,
    SOURCE_TRACE_SCHEMA as SOURCE_SCHEMA,
    sha256_file,
)


class CalibrationError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="ascii") as source:
            value = json.load(source, object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise CalibrationError(f"{path} must contain an object")
    return value


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_trace_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="ascii") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line, object_pairs_hook=_no_duplicates)
                if type(row) is not dict or row.get("schema") != SOURCE_SCHEMA:
                    raise CalibrationError(f"trace line {line_number}: schema mismatch")
                event_id = row.get("event_id")
                if type(event_id) is not str or event_id in rows:
                    raise CalibrationError(f"trace line {line_number}: invalid event id")
                if (
                    type(row.get("input_tokens")) is not int
                    or row["input_tokens"] <= 0
                    or type(row.get("output_tokens")) is not int
                    or row["output_tokens"] <= 0
                ):
                    raise CalibrationError(f"trace line {line_number}: invalid shape")
                rows[event_id] = row
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read trace: {exc}") from exc
    if not rows:
        raise CalibrationError("trace is empty")
    return rows


def solve_linear(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1.0e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[index][-1] for index in range(size)]


def fit_nonnegative_affine(
    observations: list[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], dict[str, int]]:
    if len(observations) < 3:
        raise CalibrationError("at least three observations are required")
    rows = [(1.0, float(inp), float(out)) for inp, out, _ in observations]
    targets = [float(value) for _, _, value in observations]
    best: tuple[float, tuple[float, float, float]] | None = None
    for count in range(1, 4):
        for active in itertools.combinations(range(3), count):
            gram = [
                [
                    sum(row[left] * row[right] for row in rows)
                    for right in active
                ]
                for left in active
            ]
            rhs = [
                sum(row[index] * target for row, target in zip(rows, targets))
                for index in active
            ]
            solved = solve_linear(gram, rhs)
            if solved is None or any(value < -1.0e-8 for value in solved):
                continue
            coefficients = [0.0, 0.0, 0.0]
            for index, value in zip(active, solved):
                coefficients[index] = max(0.0, value)
            squared = sum(
                (
                    sum(coefficient * feature for coefficient, feature in zip(coefficients, row))
                    - target
                ) ** 2
                for row, target in zip(rows, targets)
            )
            key = (squared, tuple(coefficients))
            if best is None or key < best:
                best = key
    if best is None:
        raise CalibrationError("cannot fit a nonnegative affine model")
    rounded = tuple(max(0, int(round(value))) for value in best[1])
    predictions = [
        rounded[0] + rounded[1] * inp + rounded[2] * out
        for inp, out, _ in observations
    ]
    residuals = [
        actual - predicted
        for predicted, (_, _, actual) in zip(predictions, observations)
    ]
    absolute = sorted(abs(value) for value in residuals)
    p90_index = max(0, math.ceil(0.9 * len(absolute)) - 1)
    metrics = {
        "sample_count": len(observations),
        "mae_us": int(round(sum(absolute) / len(absolute))),
        "rmse_us": int(round(math.sqrt(sum(value * value for value in residuals) / len(residuals)))),
        "p90_abs_error_us": absolute[p90_index],
        "max_abs_error_us": max(absolute),
        "max_positive_error_us": max(0, max(residuals)),
        "actual_total_us": sum(actual for _, _, actual in observations),
        "predicted_total_us": sum(predictions),
    }
    return rounded, metrics


def result_observations(
    result: dict[str, Any],
    trace: dict[str, dict[str, Any]],
    role: str,
) -> list[tuple[int, int, int]]:
    if result.get("status") != "PASS":
        raise CalibrationError("physical result did not pass")
    rows = result.get("request_results")
    if type(rows) is not list:
        raise CalibrationError("physical result lacks request rows")
    observations = []
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict or row.get("role") != role:
            continue
        event_id = row.get("event_id")
        if type(event_id) is not str or event_id in seen or event_id not in trace:
            raise CalibrationError("physical result event identity mismatch")
        seen.add(event_id)
        if role == "cold":
            latency_us = row.get("route_wall_us")
        else:
            completion_ns = row.get("completion_ns")
            dispatch_ns = row.get("dispatch_ns")
            if type(completion_ns) is not int or type(dispatch_ns) is not int:
                raise CalibrationError("hot result lacks timing stamps")
            latency_us = (completion_ns - dispatch_ns + 999) // 1000
        if type(latency_us) is not int or latency_us <= 0:
            raise CalibrationError("physical result latency is invalid")
        shape = trace[event_id]
        observations.append((shape["input_tokens"], shape["output_tokens"], latency_us))
    if not observations:
        raise CalibrationError(f"physical result has no {role} observations")
    return observations


def affine_json(coefficients: tuple[int, int, int]) -> dict[str, object]:
    return {
        "kind": "affine_tokens_v1",
        "fixed": coefficients[0],
        "input_token": coefficients[1],
        "output_token": coefficients[2],
    }


def latency_json(
    observations: list[tuple[int, int, int]],
) -> tuple[dict[str, object], dict[str, int]]:
    coefficients, metrics = fit_nonnegative_affine(observations)
    return {
        "cost_us": affine_json(coefficients),
        "ucb_add_us": metrics["max_positive_error_us"],
        "sample_count": metrics["sample_count"],
        "measured": True,
    }, metrics


def unknown_energy() -> dict[str, object]:
    return {
        "status": "unknown",
        "cost_uj": None,
        "lower_error_ppm": 0,
        "upper_error_ppm": 0,
    }


def route_json(
    *,
    route_id: str,
    workload_id: str,
    granularity: str,
    baseline: bool,
    resource_slots: dict[str, int],
    latency: dict[str, object],
    quality_class: str,
    server_memory_bytes: int,
    evidence_ids: Iterable[str],
) -> dict[str, object]:
    return {
        "route_id": route_id,
        "workload_id": workload_id,
        "granularity": granularity,
        "baseline": baseline,
        "resource_slots": resource_slots,
        "latency": latency,
        "energy": unknown_energy(),
        "quality_class": quality_class,
        "placement_verified": True,
        "resident": True,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": server_memory_bytes,
        "evidence_ids": list(evidence_ids),
    }


def campaign_summary(result: dict[str, Any]) -> dict[str, object]:
    metrics = result.get("metrics")
    if type(metrics) is not dict:
        raise CalibrationError("physical result lacks campaign metrics")
    by_role = metrics.get("by_role")
    if type(by_role) is not dict or type(by_role.get("cold")) is not dict:
        raise CalibrationError("physical result lacks cold metrics")
    cold = by_role["cold"]
    service = cold.get("service_s")
    if type(service) is not dict:
        raise CalibrationError("physical result lacks cold service metrics")
    duration_s = metrics.get("duration_s")
    throughput = metrics.get("throughput_tokens_s")
    if (
        not isinstance(duration_s, (int, float))
        or isinstance(duration_s, bool)
        or duration_s <= 0
        or not isinstance(throughput, (int, float))
        or isinstance(throughput, bool)
        or throughput <= 0
        or type(metrics.get("completed")) is not int
        or type(metrics.get("slo_met")) is not int
        or not isinstance(service.get("p50"), (int, float))
        or isinstance(service.get("p50"), bool)
    ):
        raise CalibrationError("physical campaign metric is invalid")
    samples = result.get("resources", {}).get("samples", {})
    gpu_power_mw = samples.get("gpu_power_mean_mw")
    gpu_utilization = samples.get("gpu_utilization_mean_pct")
    return {
        "completed": metrics["completed"],
        "slo_met": metrics["slo_met"],
        "makespan_us": int(round(float(duration_s) * 1_000_000)),
        "throughput_tokens_s": float(throughput),
        "cold_service_p50_us": int(round(float(service["p50"]) * 1_000_000)),
        "gpu_power_mean_mw_diagnostic": gpu_power_mw,
        "gpu_utilization_mean_pct_diagnostic": gpu_utilization,
    }


def build_profile(
    trace_path: Path,
    cpu_path: Path,
    op15_path: Path,
    cpu_evidence_path: str | None = None,
    op15_evidence_path: str | None = None,
) -> dict[str, Any]:
    trace = load_trace_rows(trace_path)
    cpu = load_json(cpu_path)
    op15 = load_json(op15_path)
    cpu_sha = sha256_file(cpu_path)
    op15_sha = sha256_file(op15_path)
    trace_sha = sha256_file(trace_path)
    hot_observations = [
        *result_observations(cpu, trace, "hot"),
        *result_observations(op15, trace, "hot"),
    ]
    cpu_observations = result_observations(cpu, trace, "cold")
    op15_observations = result_observations(op15, trace, "cold")
    hot_latency, hot_fit = latency_json(hot_observations)
    cpu_latency, cpu_fit = latency_json(cpu_observations)
    op15_latency, op15_fit = latency_json(op15_observations)
    cpu_samples = cpu.get("resources", {}).get("samples", {})
    op15_samples = op15.get("resources", {}).get("samples", {})
    cpu_rss = cpu_samples.get("cold_rss_max_bytes")
    op15_rss = op15_samples.get("cold_rss_max_bytes")
    if type(cpu_rss) is not int or cpu_rss <= 0 or type(op15_rss) is not int or op15_rss <= 0:
        raise CalibrationError("physical result lacks cold RSS evidence")
    source_id = hashlib.sha256(
        (trace_sha + cpu_sha + op15_sha).encode("ascii")
    ).hexdigest()[:16]
    profile: dict[str, Any] = {
        "schema": PROFILE_SCHEMA,
        "profile_id": f"physical-4060ti-op15-{source_id}",
        "resources": [
            {
                "resource_id": "cuda0",
                "kind": "gpu",
                "capacity": 8,
                "ready": True,
                "identity": "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
            },
            {
                "resource_id": "cpu-cold",
                "kind": "cpu",
                "capacity": 1,
                "ready": True,
                "identity": "i9-12900K",
            },
            {
                "resource_id": "op15-htp",
                "kind": "npu",
                "capacity": 1,
                "ready": True,
                "identity": "3C15AU002CL00000:hexagon-v81",
            },
            {
                "resource_id": "op15-usb",
                "kind": "transport",
                "capacity": 1,
                "ready": True,
                "identity": "usb-5gbps-direct-dmabuf",
            },
        ],
        "trace_workload_map": {
            "gemma-4-12b-it-q8_0": "hot-generation",
            "qwen3-14b-q4_k_m": "cold-generation",
        },
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 1_050_000,
        },
        "routes": [
            route_json(
                route_id="hot-cuda-task",
                workload_id="hot-generation",
                granularity="task",
                baseline=True,
                resource_slots={"cuda0": 1},
                latency=hot_latency,
                quality_class="exact",
                server_memory_bytes=13_737_394_176,
                evidence_ids=(cpu_sha, op15_sha),
            ),
            route_json(
                route_id="cold-cpu-task",
                workload_id="cold-generation",
                granularity="task",
                baseline=True,
                resource_slots={"cpu-cold": 1},
                latency=cpu_latency,
                quality_class="exact",
                server_memory_bytes=cpu_rss,
                evidence_ids=(cpu_sha,),
            ),
            route_json(
                route_id="cold-cpu-op15-ffn",
                workload_id="cold-generation",
                granularity="operator",
                baseline=False,
                resource_slots={"cpu-cold": 1, "op15-htp": 1, "op15-usb": 1},
                latency=op15_latency,
                quality_class="approximate",
                server_memory_bytes=op15_rss,
                evidence_ids=(op15_sha,),
            ),
        ],
        "calibration": {
            "trace": {"path": str(trace_path), "sha256": trace_sha},
            "cpu_result": {
                "path": cpu_evidence_path or str(cpu_path),
                "sha256": cpu_sha,
            },
            "op15_result": {
                "path": op15_evidence_path or str(op15_path),
                "sha256": op15_sha,
            },
            "fit": {
                "hot-cuda-task": hot_fit,
                "cold-cpu-task": cpu_fit,
                "cold-cpu-op15-ffn": op15_fit,
            },
            "observed_campaigns": {
                "control": campaign_summary(cpu),
                "cpu_op15": campaign_summary(op15),
            },
            "energy_scope": "UNKNOWN_NO_SYNCHRONIZED_CPU_PHONE_FLEET_BOUNDARY",
            "quality_scope": "PERFORMANCE_TRACE_APPROXIMATE_OUTPUT_ALLOWED",
        },
    }
    profile["profile_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(profile)
    ).hexdigest()
    return profile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cpu-result", type=Path, required=True)
    parser.add_argument("--op15-result", type=Path, required=True)
    parser.add_argument("--cpu-evidence-path")
    parser.add_argument("--op15-evidence-path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        profile = build_profile(
            args.trace,
            args.cpu_result,
            args.op15_result,
            args.cpu_evidence_path,
            args.op15_evidence_path,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(profile))
    except CalibrationError as exc:
        parser.exit(2, f"calibration failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "profile_id": profile["profile_id"],
        "profile_hash": profile["profile_hash"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
