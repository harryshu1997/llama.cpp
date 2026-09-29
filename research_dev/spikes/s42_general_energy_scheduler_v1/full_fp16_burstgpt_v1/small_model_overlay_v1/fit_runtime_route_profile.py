#!/usr/bin/env python3
"""Fit runtime route latency from repeated natural physical traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"
AUDIT_SCHEMA = "s42-fp16-llama1b-natural-route-calibration-v1"
ROUTE_FEATURES = {
    "desktop-cpu": (
        "input_tokens",
        "output_tokens",
        "actual_batch_size",
        "active_cpu_slots",
        "memory_bandwidth_pressure_basis_points",
    ),
    "phone-adreno": ("input_tokens", "output_tokens"),
}
PHASE_LABELS = {0: "idle", 1: "qwen", 2: "switching", 3: "gemma"}
MINIMUM_TRAIN = 6
MINIMUM_HOLDOUT = 4


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


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def contention_class(phase_id: int, op15: int) -> int:
    require(phase_id in PHASE_LABELS, "large phase id")
    require(op15 in {0, 1}, "large OP15 flag")
    return (4 if phase_id == 0 else phase_id) + 4 * op15


def normalized_features(row: dict[str, Any]) -> dict[str, int]:
    context = row.get("runtime_context")
    require(type(context) is dict, "runtime context")
    raw = context.get("cost_features")
    require(type(raw) is dict, "runtime cost features")
    phase_id = raw.get("large_phase_id")
    op15 = raw.get("large_model_op15")
    class_id = contention_class(phase_id, op15)
    recorded_class = raw.get("contention_class_id")
    require(
        recorded_class is None or recorded_class == class_id,
        "contention class identity",
    )
    if type(raw.get("active_cpu_slots")) is int:
        active_cpu_slots = raw["active_cpu_slots"]
        actual_batch_size = raw.get("actual_batch_size")
        memory_pressure = raw.get(
            "memory_bandwidth_pressure_basis_points"
        )
        source_contract = 2
    else:
        phase_capacity = {0: 0, 1: 4, 2: 8, 3: 8}[phase_id]
        active_cpu_slots = min(
            raw.get("active_large_cpu_requests", 0), phase_capacity
        ) + min(raw.get("active_small_cpu_requests", 0), 4)
        actual_batch_size = min(row.get("input_tokens", 0), 512)
        memory_pressure = raw.get("memory_stall_avg10_basis_points")
        source_contract = 1
    values = {
        "active_cpu_slots": active_cpu_slots,
        "actual_batch_size": actual_batch_size,
        "contention_class_id": class_id,
        "input_tokens": row.get("input_tokens"),
        "memory_bandwidth_pressure_basis_points": memory_pressure,
        "output_tokens": row.get("output_tokens"),
        "source_feature_contract": source_contract,
    }
    require(
        all(type(value) is int and value >= 0 for value in values.values())
        and values["input_tokens"] > 0
        and values["output_tokens"] > 0,
        "normalized cost features",
    )
    return values


def physical_observation(
    row: dict[str, Any],
    split: str,
    source_sha256: str,
) -> dict[str, Any]:
    route = row.get("route")
    prompt_ms = row.get("prompt_ms")
    predicted_ms = row.get("predicted_ms")
    dispatch_ns = row.get("dispatch_ns")
    completion_ns = row.get("completion_ns")
    require(
        split in {"train", "holdout"}
        and route in ROUTE_FEATURES
        and type(prompt_ms) in {int, float}
        and not isinstance(prompt_ms, bool)
        and prompt_ms >= 0
        and type(predicted_ms) in {int, float}
        and not isinstance(predicted_ms, bool)
        and predicted_ms > 0
        and type(dispatch_ns) is int
        and type(completion_ns) is int
        and completion_ns > dispatch_ns
        and type(row.get("endpoint_task_id")) is int
        and type(row.get("stream_sha256")) is str
        and len(row["stream_sha256"]) == 64,
        "physical route receipt",
    )
    features = normalized_features(row)
    latency_us = max(1, math.ceil((prompt_ms + predicted_ms) * 1000))
    controller_wall_us = (completion_ns - dispatch_ns) // 1000
    return {
        "class_id": features["contention_class_id"],
        "controller_wall_us": controller_wall_us,
        "endpoint_queue_us": max(0, controller_wall_us - latency_us),
        "event_id": row.get("event_id"),
        "features": features,
        "latency_us": latency_us,
        "overlay_request_index": row.get("overlay_request_index"),
        "route_id": route,
        "source_sha256": source_sha256,
        "split": split,
    }


def result_observations(
    path: Path,
    split: str,
    large_model_policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = load(path)
    rows = result.get("request_results")
    policy = result.get("policy")
    metrics = result.get("metrics")
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and type(policy) is dict
        and policy.get("large_model_policy") == large_model_policy
        and type(metrics) is dict
        and type(metrics.get("small_model")) is dict
        and type(rows) is list
        and len(rows) == metrics["small_model"].get("completed")
        and len(rows) > 0,
        f"natural physical result identity: {path}",
    )
    source_sha256 = digest(path)
    observations = []
    for row in rows:
        require(
            type(row) is dict
            and row.get("calibration_split") is None
            and row.get("target_large_phase") is None,
            "natural trace arrival",
        )
        observations.append(physical_observation(
            row, split, source_sha256
        ))
    identity = {
        "base_trace": result.get("input_sha256", {}).get("base_trace"),
        "manifest": result.get("input_sha256", {}).get("manifest"),
        "model_identities": result.get("model_identities"),
        "overlay_trace": result.get("input_sha256", {}).get(
            "overlay_trace"
        ),
    }
    require(
        all(
            type(identity[name]) is str and len(identity[name]) == 64
            for name in ("base_trace", "manifest", "overlay_trace")
        )
        and type(identity["model_identities"]) is dict,
        "trace and model identity",
    )
    return observations, identity


def fit_nonnegative_affine(
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    columns = ("fixed", *feature_names)
    matrix = [[
        1.0,
        *(float(row["features"][name]) for name in feature_names),
    ] for row in rows]
    targets = [float(row["latency_us"]) for row in rows]
    scales = [
        max(vector[index] for vector in matrix)
        for index in range(len(columns))
    ]
    scaled = [[
        0.0 if scales[index] == 0 else value / scales[index]
        for index, value in enumerate(vector)
    ] for vector in matrix]
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
                largest_change, abs(updated - beta[feature_index])
            )
            beta[feature_index] = updated
        if largest_change < 1e-7:
            break
    coefficients = [
        0.0 if scale == 0 else value / scale
        for value, scale in zip(beta, scales)
    ]
    return max(0, round(coefficients[0])), {
        name: max(0, round(coefficients[index + 1]))
        for index, name in enumerate(feature_names)
    }


def predict(
    row: dict[str, Any],
    fixed: int,
    coefficients: dict[str, int],
) -> int:
    return max(1, fixed + sum(
        coefficient * row["features"][name]
        for name, coefficient in coefficients.items()
    ))


def fit_variant(
    route_id: str,
    class_id: int,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    require(
        len(train) >= MINIMUM_TRAIN and len(holdout) >= MINIMUM_HOLDOUT,
        f"{route_id} contention class {class_id} sample count",
    )
    feature_names = ROUTE_FEATURES[route_id]
    fixed, coefficients = fit_nonnegative_affine(train, feature_names)
    selected = tuple(
        name for name in feature_names
        if coefficients[name] > 0
        or name in {"input_tokens", "output_tokens"}
    )
    if selected != feature_names:
        fixed, coefficients = fit_nonnegative_affine(train, selected)
    train_residuals = [
        row["latency_us"] - predict(row, fixed, coefficients)
        for row in train
    ]
    guard_us = max(
        250_000,
        math.ceil(max(row["latency_us"] for row in train) * 0.25),
    )
    ucb_add_us = max(0, max(train_residuals)) + guard_us
    evaluated = []
    holdout_violations = 0
    for row in rows:
        estimate = predict(row, fixed, coefficients)
        upper = estimate + ucb_add_us
        violation = row["latency_us"] > upper
        if row["split"] == "holdout":
            holdout_violations += int(violation)
        evaluated.append({
            "controller_wall_us": row["controller_wall_us"],
            "endpoint_queue_us": row["endpoint_queue_us"],
            "event_id": row["event_id"],
            "latency_us": row["latency_us"],
            "overlay_request_index": row["overlay_request_index"],
            "predicted_us": estimate,
            "source_sha256": row["source_sha256"],
            "split": row["split"],
            "upper_us": upper,
            "upper_violation": violation,
        })
    op15 = int(class_id > 4)
    phase_id = class_id - 4 * op15
    measured = holdout_violations == 0
    label = (
        f"{PHASE_LABELS[phase_id]}-"
        f"{'op15-assistance' if op15 else 'cpu-overflow'}-natural"
    )
    cost = {
        "coefficients": coefficients,
        "fixed": fixed,
        "kind": "affine_features_v1",
    }
    variant = {
        "cost_us": cost,
        "label": label,
        "measured": measured,
        "sample_count": len(train),
        "selector_value": class_id,
        "ucb_add_us": ucb_add_us,
    }
    rmse = math.sqrt(sum(
        (row["latency_us"] - predict(row, fixed, coefficients)) ** 2
        for row in train
    ) / len(train))
    audit = {
        "class_id": class_id,
        "evaluated": evaluated,
        "feature_coefficients": coefficients,
        "holdout_count": len(holdout),
        "holdout_upper_violations": holdout_violations,
        "label": label,
        "measured": measured,
        "route_id": route_id,
        "source_feature_contracts": sorted({
            row["features"]["source_feature_contract"] for row in rows
        }),
        "train_count": len(train),
        "train_rmse_us": round(rmse),
        "ucb_add_us": ucb_add_us,
    }
    return variant, audit


def update_full_route_leases(
    route: dict[str, Any],
    variants: list[dict[str, Any]],
) -> None:
    leases = route.get("resource_leases")
    if not leases:
        return
    require(
        len(variants) == 1
        and all(
            type(lease) is dict
            and lease.get("start_offset_us") == 0
            for lease in leases
        ),
        "conditioned full-route leases",
    )
    variant = variants[0]
    for lease in leases:
        lease["duration_us"] = variant["cost_us"]
        lease["duration_ucb_add_us"] = variant["ucb_add_us"]


def fit(
    train_paths: list[Path],
    holdout_paths: list[Path],
    base_profile_path: Path,
    large_model_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(train_paths and holdout_paths, "training and holdout results")
    train_hashes = {digest(path) for path in train_paths}
    holdout_hashes = {digest(path) for path in holdout_paths}
    require(not train_hashes & holdout_hashes, "disjoint calibration inputs")
    observations = []
    identities = []
    for split, paths in (("train", train_paths), ("holdout", holdout_paths)):
        for path in paths:
            rows, identity = result_observations(
                path, split, large_model_policy
            )
            observations.extend(rows)
            identities.append(identity)
    require(
        identities and all(identity == identities[0] for identity in identities),
        "identical trace and model identities",
    )
    profile = load(base_profile_path)
    require(profile.get("schema") == PROFILE_SCHEMA, "base profile schema")
    audits = []
    fitted_routes = []
    for route_id in ROUTE_FEATURES:
        route = next(
            (row for row in profile.get("routes", [])
             if row.get("route_id") == route_id),
            None,
        )
        require(type(route) is dict, f"base route: {route_id}")
        route_rows = [
            row for row in observations if row["route_id"] == route_id
        ]
        classes = sorted({row["class_id"] for row in route_rows})
        require(classes, f"route calibration rows: {route_id}")
        variants = []
        for class_id in classes:
            variant, audit = fit_variant(
                route_id,
                class_id,
                [row for row in route_rows if row["class_id"] == class_id],
            )
            variants.append(variant)
            audits.append(audit)
        route["latency"] = {
            "kind": "conditioned_affine_features_v1",
            "selector_feature": "contention_class_id",
            "variants": variants,
        }
        update_full_route_leases(route, variants)
        route["evidence_ids"] = sorted(set(
            route.get("evidence_ids", [])
            + ["sha256:" + value for value in train_hashes | holdout_hashes]
        ))
        fitted_routes.append(route_id)
    profile["profile_id"] = profile["profile_id"] + "-natural-runtime-v3"
    all_measured = all(row["measured"] for row in audits)
    audit = {
        "all_variants_measured": all_measured,
        "fitted_routes": fitted_routes,
        "input_identity": identities[0],
        "input_sha256": {
            "base_profile": digest(base_profile_path),
            "holdout_results": sorted(holdout_hashes),
            "train_results": sorted(train_hashes),
        },
        "large_model_policy": large_model_policy,
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if all_measured else "FAIL",
        "variants": audits,
    }
    audit["profile_sha256"] = hashlib.sha256(canonical(profile)).hexdigest()
    return profile, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-result", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--holdout-result", type=Path, action="append", required=True
    )
    parser.add_argument("--base-profile", type=Path, required=True)
    parser.add_argument(
        "--large-model-policy",
        choices=("cpu-overflow", "op15-assistance"),
        required=True,
    )
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output_profile.is_absolute()
        and args.output_audit.is_absolute()
        and not args.output_profile.exists()
        and not args.output_audit.exists(),
        "new absolute output paths",
    )
    profile, audit = fit(
        args.train_result,
        args.holdout_result,
        args.base_profile,
        args.large_model_policy,
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_bytes(canonical(profile))
    args.output_audit.write_bytes(canonical(audit))
    print(json.dumps({
        "audit": str(args.output_audit),
        "profile": str(args.output_profile),
        "status": audit["status"],
    }, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
