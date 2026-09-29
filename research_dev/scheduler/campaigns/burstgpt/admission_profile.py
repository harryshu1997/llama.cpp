#!/usr/bin/env python3
"""Fit causal resident-endpoint queue costs from repeated F16 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s41-hierarchical-burstgpt-result-v1"
PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"
CALIBRATION_SCHEMA = "s42-fp16-request-admission-calibration-v1"
INGRESS_CAPACITY = 128
FEATURES = (
    "input_tokens",
    "output_tokens",
    "active_model_requests",
    "active_model_input_tokens",
    "active_model_output_tokens",
)
MODEL_ROWS = {
    "hot": {
        "artifact_sha256": (
            "d89e9e823744222e595e0b3c8fd5436c"
            "e5d3a6a446fa42492ebce6064dfa9718"
        ),
        "effective_model_id": "qwen3-14b-q4_k_m",
        "model_id": "qwen3-14b-q4km-dequant-f16",
        "model_bytes": 29_543_423_360,
        "prefix": "qwen",
        "workload_id": "qwen3-14b-fp16-request",
    },
    "cold": {
        "artifact_sha256": (
            "ed76f2183d2d1d65091986033023e6c7"
            "8d27f6276c1b0c5826cc92acf73538cf"
        ),
        "effective_model_id": "gemma-4-12b-it-q4_0",
        "model_id": "gemma-4-12b-q40-dequant-f16",
        "model_bytes": 23_832_065_056,
        "prefix": "gemma",
        "workload_id": "gemma4-12b-fp16-request",
    },
}
ROUTE_FAMILIES = {
    "cpu": "cpu_f16",
    "gpu": "cuda_full_f16",
    "phone": "phone_f16",
    "gpu-cpu": "cuda_cpu_f16",
    "gpu-phone": "cuda_cpu_op15_f16",
    "cpu-phone": "cpu_op15_f16",
}


class FitError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FitError(message)


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


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_result(path: Path, arm: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    scheduler = value.get("fp16_resident_scheduler", {})
    metrics = value.get("metrics", {})
    rows = value.get("request_results")
    require(
        type(value) is dict
        and value.get("schema") == RESULT_SCHEMA
        and value.get("status") == "PASS"
        and value.get("mode") == "fp16-switch"
        and scheduler.get("arm") == arm
        and metrics.get("completed") == 74
        and metrics.get("output_tokens") == 11_605
        and type(rows) is list
        and len(rows) == 74,
        f"F16 result identity: {path}",
    )
    expected_suffix = (
        "cuda_cpu_op15_f16" if arm == "op15" else "cuda_cpu_f16"
    )
    require(
        all(
            type(row) is dict
            and row.get("role") in MODEL_ROWS
            and row.get("effective_model_id")
                == MODEL_ROWS[row["role"]]["effective_model_id"]
            and row.get("route")
                == f"{MODEL_ROWS[row['role']]['prefix']}_{expected_suffix}"
            and type(row.get("tokens")) is list
            and len(row["tokens"]) == row.get("output_tokens")
            for row in rows
        ),
        f"F16 physical route and output identity: {path}",
    )
    return value


def observations(
    value: dict[str, Any],
    role: str,
    split: str,
) -> list[dict[str, Any]]:
    rows = sorted(
        (
            row for row in value["request_results"]
            if type(row) is dict and row.get("role") == role
        ),
        key=lambda row: row.get("dispatch_ns", -1),
    )
    expected = 57 if role == "hot" else 17
    require(len(rows) == expected, f"{role} request count")
    result = []
    prior: list[dict[str, Any]] = []
    for row in rows:
        dispatch_ns = row.get("dispatch_ns")
        completion_ns = row.get("completion_ns")
        require(
            type(dispatch_ns) is int
            and type(completion_ns) is int
            and completion_ns > dispatch_ns
            and type(row.get("request_index")) is int
            and type(row.get("event_id")) is str
            and type(row.get("input_tokens")) is int
            and row["input_tokens"] > 0
            and type(row.get("output_tokens")) is int
            and row["output_tokens"] > 0,
            f"{role} physical request receipt",
        )
        active = [
            previous for previous in prior
            if previous["completion_ns"] > dispatch_ns
        ]
        features = {
            "active_model_input_tokens": sum(
                previous["input_tokens"] for previous in active
            ),
            "active_model_output_tokens": sum(
                previous["output_tokens"] for previous in active
            ),
            "active_model_requests": len(active),
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }
        result.append({
            "event_id": row["event_id"],
            "features": features,
            "latency_us": (completion_ns - dispatch_ns + 999) // 1000,
            "request_index": row["request_index"],
            "split": split,
        })
        prior.append(row)
    return result


def fit_nonnegative_affine(
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, int]]:
    columns = ("fixed", *FEATURES)
    matrix = [
        [1.0, *(float(row["features"][name]) for name in FEATURES)]
        for row in rows
    ]
    targets = [float(row["latency_us"]) for row in rows]
    scales = [
        max(vector[index] for vector in matrix)
        for index in range(len(columns))
    ]
    scaled = [
        [
            0.0 if scales[index] == 0 else value / scales[index]
            for index, value in enumerate(vector)
        ]
        for vector in matrix
    ]
    beta = [0.0] * len(columns)
    for _ in range(20_000):
        largest_change = 0.0
        for feature_index in range(len(columns)):
            column = [row[feature_index] for row in scaled]
            denominator = sum(value * value for value in column)
            if denominator == 0:
                continue
            numerator = 0.0
            for row_index, vector in enumerate(scaled):
                other = sum(
                    vector[index] * beta[index]
                    for index in range(len(columns))
                    if index != feature_index
                )
                numerator += column[row_index] * (
                    targets[row_index] - other
                )
            updated = max(0.0, numerator / denominator)
            largest_change = max(
                largest_change,
                abs(updated - beta[feature_index]),
            )
            beta[feature_index] = updated
        if largest_change < 1.0e-7:
            break
    coefficients = [
        0.0 if scale == 0 else value / scale
        for value, scale in zip(beta, scales)
    ]
    return max(0, round(coefficients[0])), {
        name: max(0, round(coefficients[index + 1]))
        for index, name in enumerate(FEATURES)
    }


def predict(
    row: dict[str, Any],
    fixed: int,
    coefficients: dict[str, int],
) -> int:
    return max(1, fixed + sum(
        coefficients[name] * row["features"][name]
        for name in FEATURES
    ))


def fit_role(
    train: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixed, coefficients = fit_nonnegative_affine(train)
    residuals = [
        row["latency_us"] - predict(row, fixed, coefficients)
        for row in train
    ]
    guard_us = max(
        250_000,
        math.ceil(max(row["latency_us"] for row in train) * 0.25),
    )
    ucb_add_us = max(0, max(residuals)) + guard_us
    evaluated = []
    for row in (*train, *holdout):
        estimate = predict(row, fixed, coefficients)
        upper = estimate + ucb_add_us
        evaluated.append({
            "event_id": row["event_id"],
            "latency_us": row["latency_us"],
            "predicted_us": estimate,
            "request_index": row["request_index"],
            "split": row["split"],
            "upper_us": upper,
            "upper_violation": row["latency_us"] > upper,
        })
    holdout_violations = sum(
        row["upper_violation"]
        for row in evaluated if row["split"] == "holdout"
    )
    require(holdout_violations == 0, "holdout upper-bound violation")
    latency = {
        "cost_us": {
            "coefficients": coefficients,
            "fixed": fixed,
            "kind": "affine_features_v1",
        },
        "measured": True,
        "sample_count": len(train),
        "ucb_add_us": ucb_add_us,
    }
    rmse_us = round(math.sqrt(sum(
        (row["latency_us"] - predict(row, fixed, coefficients)) ** 2
        for row in train
    ) / len(train)))
    audit = {
        "evaluated": evaluated,
        "feature_names": list(FEATURES),
        "holdout_count": len(holdout),
        "holdout_upper_violations": holdout_violations,
        "target": "controller_wall_us_including_endpoint_queue",
        "train_count": len(train),
        "train_rmse_us": rmse_us,
        "ucb_add_us": ucb_add_us,
    }
    return latency, audit


def route(
    model: dict[str, Any],
    suffix: str,
    latency: dict[str, Any],
    selected_suffix: str,
    evidence_ids: list[str],
) -> dict[str, Any]:
    route_id = f"{model['prefix']}_{suffix}"
    selected = suffix == selected_suffix
    return {
        "baseline": selected,
        "energy": {"status": "unknown"},
        "evidence_ids": evidence_ids,
        "granularity": "task",
        "latency": {
            **latency,
            "measured": selected,
        },
        "overlap": {"status": "not_applicable"},
        "placement_verified": selected,
        "quality_class": "approximate",
        "resident": selected,
        "resource_slots": {f"{route_id}-endpoint": 1},
        "route_id": route_id,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": model["model_bytes"],
        "workload_id": model["workload_id"],
    }


def build(
    train_path: Path,
    holdout_path: Path,
    arm: str,
) -> dict[str, Any]:
    train_digest = digest(train_path)
    holdout_digest = digest(holdout_path)
    require(
        train_path.resolve() != holdout_path.resolve()
        and train_digest != holdout_digest,
        "train and holdout must be distinct physical results",
    )
    train_result = load_result(train_path, arm)
    holdout_result = load_result(holdout_path, arm)
    train_rows = train_result["request_results"]
    holdout_rows = holdout_result["request_results"]
    train_identity = sorted(
        (
            row["request_index"],
            row["event_id"],
            row["role"],
            row["input_tokens"],
            row["output_tokens"],
        )
        for row in train_rows
    )
    holdout_identity = sorted(
        (
            row["request_index"],
            row["event_id"],
            row["role"],
            row["input_tokens"],
            row["output_tokens"],
        )
        for row in holdout_rows
    )
    require(train_identity == holdout_identity, "train/holdout work identity")
    work_identity_sha256 = hashlib.sha256(
        canonical(train_identity)
    ).hexdigest()
    latencies = {}
    audits = {}
    for role in ("hot", "cold"):
        latencies[role], audits[role] = fit_role(
            observations(train_result, role, "train"),
            observations(holdout_result, role, "holdout"),
        )
    selected_family = "gpu-phone" if arm == "op15" else "gpu-cpu"
    selected_suffix = ROUTE_FAMILIES[selected_family]
    evidence_ids = [
        "sha256:" + train_digest,
        "sha256:" + holdout_digest,
    ]
    routes = []
    for role in ("hot", "cold"):
        model = MODEL_ROWS[role]
        routes.extend(
            route(
                model,
                suffix,
                latencies[role],
                selected_suffix,
                evidence_ids,
            )
            for suffix in ROUTE_FAMILIES.values()
        )
    source_key = hashlib.sha256(
        (evidence_ids[0] + evidence_ids[1]).encode("ascii")
    ).hexdigest()[:16]
    profile: dict[str, Any] = {
        "calibration": {
            "applicability": {
                "active_set_observation": "physical_dispatch_time_proxy",
                "endpoint_queue_target": (
                    "controller_wall_us_including_endpoint_queue"
                ),
                "ingress_capacity": INGRESS_CAPACITY,
                "model_artifacts": {
                    role: {
                        "artifact_sha256": model["artifact_sha256"],
                        "bytes": model["model_bytes"],
                        "model_id": model["model_id"],
                    }
                    for role, model in MODEL_ROWS.items()
                },
                "same_work_repeated_holdout": True,
                "work_identity_sha256": work_identity_sha256,
            },
            "arm": arm,
            "audits": audits,
            "future_request_data": "not_accepted",
            "holdout_result": {
                "path": str(holdout_path),
                "sha256": evidence_ids[1],
            },
            "schema": CALIBRATION_SCHEMA,
            "status": "PASS",
            "train_result": {
                "path": str(train_path),
                "sha256": evidence_ids[0],
            },
        },
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 20_000_000,
        },
        "profile_id": f"fp16-request-admission-{arm}-{source_key}",
        "resources": [
            {
                "capacity": INGRESS_CAPACITY,
                "identity": row["resource_id"],
                "kind": "composite_executor_ingress_queue",
                "ready": True,
                "resource_id": row["resource_id"],
            }
            for row in (
                {"resource_id": f"{item['route_id']}-endpoint"}
                for item in routes
            )
        ],
        "routes": routes,
        "schema": PROFILE_SCHEMA,
        "trace_workload_map": {
            model["model_id"]: model["workload_id"]
            for model in MODEL_ROWS.values()
        },
    }
    return profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-result", type=Path, required=True)
    parser.add_argument("--holdout-result", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "op15"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        value = build(args.train_result, args.holdout_result, args.arm)
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"F16 request admission fit failed: {exc}\n")
    print(json.dumps({
        "arm": args.arm,
        "output": str(args.output),
        "profile_sha256": "sha256:" + hashlib.sha256(
            canonical(value)
        ).hexdigest(),
        "profile_id": value["profile_id"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
