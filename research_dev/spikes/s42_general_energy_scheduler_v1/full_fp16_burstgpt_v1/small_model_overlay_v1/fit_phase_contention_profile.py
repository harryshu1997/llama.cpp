#!/usr/bin/env python3
"""Fit phase and large-arm conditioned CPU latency from physical runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"
AUDIT_SCHEMA = "s42-fp16-llama1b-contention-calibration-v2"
PHASE_LABELS = {
    1: "qwen",
    2: "switching",
    3: "gemma",
    4: "idle",
}
FEATURES = (
    "input_tokens",
    "output_tokens",
    "actual_batch_size",
    "active_cpu_slots",
    "memory_bandwidth_pressure_basis_points",
)
MINIMUM_TRAIN = 14
MINIMUM_HOLDOUT = 6
MINIMUM_NATURAL_PER_RUN = 40
MINIMUM_NATURAL_PER_POLICY = 80
MINIMUM_NATURAL_PER_CLASS = 2


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
    require(phase_id in {0, 1, 2, 3}, "large phase id")
    require(op15 in {0, 1}, "large OP15 flag")
    return (4 if phase_id == 0 else phase_id) + 4 * op15


def physical_observation(
    row: dict[str, Any],
    split: str,
    expected_route: str = "desktop-cpu",
) -> dict[str, Any]:
    context = row.get("runtime_context", {})
    features = context.get("cost_features", {})
    power = context.get("phase_power_observation", {})
    prompt_ms = row.get("prompt_ms")
    predicted_ms = row.get("predicted_ms")
    require(
        split in {"train", "holdout", "natural-holdout"}
        and row.get("route") == expected_route
        and type(features) is dict
        and all(type(features.get(name)) is int for name in FEATURES[2:])
        and type(power) is dict
        and power.get("status") == "MEASURED"
        and type(power.get("total_server_power_mw")) is int
        and type(row.get("dispatch_ns")) is int
        and type(row.get("completion_ns")) is int
        and row["completion_ns"] > row["dispatch_ns"]
        and type(prompt_ms) in {int, float}
        and not isinstance(prompt_ms, bool)
        and prompt_ms >= 0
        and type(predicted_ms) in {int, float}
        and not isinstance(predicted_ms, bool)
        and predicted_ms > 0,
        "physical calibration receipt",
    )
    phase_id = features.get("large_phase_id")
    op15 = features.get("large_model_op15")
    class_id = contention_class(phase_id, op15)
    controller_wall_us = (
        row["completion_ns"] - row["dispatch_ns"]
    ) // 1000
    endpoint_service_us = max(
        1, math.ceil((prompt_ms + predicted_ms) * 1000)
    )
    return {
        "class_id": class_id,
        "event_id": row["event_id"],
        "features": {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            **{name: features[name] for name in FEATURES[2:]},
        },
        "controller_wall_us": controller_wall_us,
        "endpoint_queue_us": max(
            0, controller_wall_us - endpoint_service_us
        ),
        "latency_us": endpoint_service_us,
        "overlay_request_index": row["overlay_request_index"],
        "phase_power_mw": power["total_server_power_mw"],
        "split": split,
    }


def calibration_observations(
    path: Path,
    expected_large_policy: str,
    expected_route: str = "desktop-cpu",
) -> list[dict[str, Any]]:
    result = load(path)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and result.get("policy", {}).get("large_model_policy")
            == expected_large_policy
        and result.get("policy", {}).get("small_model_policy")
            == "static-cpu"
        and result.get("policy", {}).get("static_route")
            == expected_route
        and result.get("scheduler_runtime", {}).get("route_counts")
            == {
                expected_route: result.get("metrics", {})
                    .get("small_model", {}).get("completed")
            },
        f"static calibration identity: {path}",
    )
    rows = result.get("request_results")
    require(type(rows) is list and len(rows) == 40, "calibration rows")
    output = []
    for row in rows:
        split = row.get("calibration_split")
        require(split in {"train", "holdout"}, "calibration split")
        output.append(physical_observation(row, split, expected_route))
    return output


def natural_observations(
    path: Path,
    expected_large_policy: str,
    expected_route: str = "desktop-cpu",
    minimum_rows: int = 10,
) -> list[dict[str, Any]]:
    result = load(path)
    rows = result.get("request_results")
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and result.get("policy", {}).get("large_model_policy")
            == expected_large_policy
        and result.get("policy", {}).get("small_model_policy")
            == "static-cpu"
        and result.get("policy", {}).get("static_route")
            == expected_route
        and type(rows) is list
        and type(minimum_rows) is int
        and minimum_rows > 0
        and len(rows) >= minimum_rows
        and result.get("scheduler_runtime", {}).get("route_counts")
            == {expected_route: len(rows)},
        f"natural validation identity: {path}",
    )
    output = []
    for row in rows:
        require(
            row.get("calibration_split") is None
            and row.get("target_large_phase") is None,
            "natural validation arrival",
        )
        output.append(physical_observation(
            row, "natural-holdout", expected_route
        ))
    return output


def fit_nonnegative_affine(
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, int]]:
    columns = ("fixed", *FEATURES)
    matrix = []
    targets = []
    for row in rows:
        matrix.append([
            1.0,
            *(float(row["features"][name]) for name in FEATURES),
        ])
        targets.append(float(row["latency_us"]))
    scales = [max(vector[index] for vector in matrix) for index in range(
        len(columns)
    )]
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
                largest_change, abs(updated - beta[feature_index])
            )
            beta[feature_index] = updated
        if largest_change < 1e-7:
            break
    coefficients = [
        0.0 if scale == 0 else value / scale
        for value, scale in zip(beta, scales)
    ]
    fixed = max(0, round(coefficients[0]))
    return fixed, {
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


def variant(
    class_id: int,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train = [row for row in rows if row["split"] == "train"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    natural = [
        row for row in rows if row["split"] == "natural-holdout"
    ]
    require(
        len(train) >= MINIMUM_TRAIN and len(holdout) >= MINIMUM_HOLDOUT,
        f"contention class {class_id} sample count",
    )
    fixed, coefficients = fit_nonnegative_affine(train)
    train_predictions = [predict(row, fixed, coefficients) for row in train]
    train_residuals = [
        row["latency_us"] - estimate
        for row, estimate in zip(train, train_predictions)
    ]
    guard_us = max(
        250_000,
        math.ceil(max(row["latency_us"] for row in train) * 0.25),
    )
    ucb_add_us = max(0, max(train_residuals)) + guard_us
    evaluated = []
    violations = 0
    natural_violations = 0
    for row in rows:
        estimate = predict(row, fixed, coefficients)
        upper = estimate + ucb_add_us
        violation = row["latency_us"] > upper
        if row["split"] == "holdout":
            violations += int(violation)
        elif row["split"] == "natural-holdout":
            natural_violations += int(violation)
        evaluated.append({
            "controller_wall_us": row.get(
                "controller_wall_us", row["latency_us"]
            ),
            "endpoint_queue_us": row.get("endpoint_queue_us", 0),
            "event_id": row["event_id"],
            "latency_us": row["latency_us"],
            "predicted_us": estimate,
            "split": row["split"],
            "upper_us": upper,
            "upper_violation": violation,
        })
    op15 = int(class_id > 4)
    phase_id = class_id - 4 * op15
    label = (
        f"{PHASE_LABELS[phase_id]}-"
        f"{'op15-assistance' if op15 else 'cpu-overflow'}"
    )
    measured = violations == 0 and natural_violations == 0
    phase_power_values = [row["phase_power_mw"] for row in rows]
    profile = {
        "cost_us": {
            "coefficients": coefficients,
            "fixed": fixed,
            "kind": "affine_features_v1",
        },
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
        "holdout_count": len(holdout),
        "holdout_upper_violations": violations,
        "label": label,
        "measured": measured,
        "natural_holdout_count": len(natural),
        "natural_holdout_upper_violations": natural_violations,
        "observed_endpoint_queue_us": {
            "maximum": max(row.get("endpoint_queue_us", 0) for row in rows),
            "mean": round(sum(
                row.get("endpoint_queue_us", 0) for row in rows
            ) / len(rows)),
            "samples": len(rows),
        },
        "phase_power_mw": {
            "maximum": max(phase_power_values),
            "mean": round(sum(phase_power_values) / len(phase_power_values)),
            "minimum": min(phase_power_values),
            "samples": len(phase_power_values),
        },
        "train_count": len(train),
        "train_rmse_us": rmse,
        "ucb_add_us": ucb_add_us,
    }
    return profile, audit


def fit(
    cpu_overflow_paths: list[Path],
    op15_paths: list[Path],
    cpu_overflow_natural_paths: list[Path],
    op15_natural_paths: list[Path],
    base_profile_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    require(
        len(cpu_overflow_paths) >= 2
        and len(op15_paths) >= 2
        and len(cpu_overflow_natural_paths) >= 2
        and len(op15_natural_paths) >= 2,
        "two repeated calibration and natural runs per large-model policy",
    )
    calibration_rows = [
        row
        for path in cpu_overflow_paths
        for row in calibration_observations(path, "cpu-overflow")
    ] + [
        row
        for path in op15_paths
        for row in calibration_observations(path, "op15-assistance")
    ]
    natural_rows = [
        row
        for path in cpu_overflow_natural_paths
        for row in natural_observations(
            path, "cpu-overflow", minimum_rows=MINIMUM_NATURAL_PER_RUN
        )
    ] + [
        row
        for path in op15_natural_paths
        for row in natural_observations(
            path,
            "op15-assistance",
            minimum_rows=MINIMUM_NATURAL_PER_RUN,
        )
    ]
    natural_class_counts = {
        class_id: sum(
            row["class_id"] == class_id for row in natural_rows
        )
        for class_id in range(1, 9)
    }
    for op15 in (0, 1):
        natural_policy_count = sum(
            (row["class_id"] > 4) == bool(op15)
            for row in natural_rows
        )
        require(
            natural_policy_count >= MINIMUM_NATURAL_PER_POLICY,
            "natural validation sample count",
        )
    require(
        all(
            count >= MINIMUM_NATURAL_PER_CLASS
            for count in natural_class_counts.values()
        ),
        "natural validation phase coverage",
    )
    rows = calibration_rows + natural_rows
    variants = []
    audits = []
    for class_id in range(1, 9):
        selected = [row for row in rows if row["class_id"] == class_id]
        profile_variant, audit_variant = variant(class_id, selected)
        variants.append(profile_variant)
        audits.append(audit_variant)

    profile = load(base_profile_path)
    require(profile.get("schema") == PROFILE_SCHEMA, "base profile schema")
    cpu_route = next(
        route for route in profile["routes"]
        if route.get("route_id") == "desktop-cpu"
    )
    cpu_route["latency"] = {
        "kind": "conditioned_affine_features_v1",
        "selector_feature": "contention_class_id",
        "variants": variants,
    }
    cpu_route["evidence_ids"] = sorted(set(
        cpu_route.get("evidence_ids", [])
        + ["sha256:" + digest(path) for path in (
            cpu_overflow_paths
            + op15_paths
            + cpu_overflow_natural_paths
            + op15_natural_paths
        )]
    ))
    profile["profile_id"] = (
        "s42-op15-whole-task-llama1b-phase-contention-v2"
    )
    all_variants_measured = all(row["measured"] for row in audits)
    audit = {
        "all_variants_measured": all_variants_measured,
        "feature_contract": list(FEATURES),
        "input_sha256": {
            "base_profile": digest(base_profile_path),
            "cpu_overflow_calibration": [
                digest(path) for path in cpu_overflow_paths
            ],
            "cpu_overflow_natural": [
                digest(path) for path in cpu_overflow_natural_paths
            ],
            "op15_assistance_calibration": [
                digest(path) for path in op15_paths
            ],
            "op15_assistance_natural": [
                digest(path) for path in op15_natural_paths
            ],
        },
        "natural_validation_count": len(natural_rows),
        "natural_validation_count_by_class": {
            str(class_id): count
            for class_id, count in natural_class_counts.items()
        },
        "schema": AUDIT_SCHEMA,
        "status": "PASS" if all_variants_measured else "FAIL",
        "variants": audits,
    }
    audit["profile_sha256"] = hashlib.sha256(canonical(profile)).hexdigest()
    return profile, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-overflow-result", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--op15-result", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--cpu-overflow-natural-result",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--op15-natural-result",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--base-profile", type=Path, required=True)
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
        args.cpu_overflow_result,
        args.op15_result,
        args.cpu_overflow_natural_result,
        args.op15_natural_result,
        args.base_profile,
    )
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_bytes(canonical(profile))
    args.output_audit.write_bytes(canonical(audit))
    print(json.dumps({
        "all_variants_measured": audit["all_variants_measured"],
        "audit": str(args.output_audit),
        "profile": str(args.output_profile),
        "status": audit["status"],
    }, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
