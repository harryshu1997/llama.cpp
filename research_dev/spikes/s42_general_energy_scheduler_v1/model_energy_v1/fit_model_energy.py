#!/usr/bin/env python3
"""Fit a bounded cohort energy profile from S42 single-model results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from calibrate_profiles import solve_linear  # noqa: E402


FEATURES = (
    "duration_ms",
    "input_token_rows",
    "output_token_rows",
    "decode_steps",
    "prefill_ubatches",
)


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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FitError(f"cannot read {path}: {error}") from error
    require(type(value) is dict, f"{path}: object")
    return value


def finite_positive(value: object, label: str) -> float:
    require(
        isinstance(value, (int, float)) and
        not isinstance(value, bool) and
        math.isfinite(float(value)) and
        float(value) > 0,
        label,
    )
    return float(value)


def assumed_phone_energy_uj(power_w: float, duration_s: float) -> int:
    require(math.isfinite(power_w) and power_w > 0, "assumed phone power")
    require(math.isfinite(duration_s) and duration_s > 0, "phone duration")
    return int(round(power_w * duration_s * 1e6))


def energy_components(case: dict[str, Any]) -> dict[str, int]:
    energy = case.get("server_energy")
    require(type(energy) is dict, "server energy")
    cpu_j = finite_positive(energy.get("cpu_package_energy_j"), "CPU energy")
    gpu_j = finite_positive(energy.get("gpu_board_energy_j"), "GPU energy")
    server_j = finite_positive(
        energy.get("server_compute_device_energy_j"), "server energy"
    )
    require(
        energy.get("boundary") == "paid_trace_interval" and
        math.isclose(server_j, cpu_j + gpu_j, rel_tol=1e-9),
        "server energy boundary",
    )
    return {
        "cpu_package": int(round(cpu_j * 1e6)),
        "gpu_board": int(round(gpu_j * 1e6)),
        "server_total": int(round(server_j * 1e6)),
    }


def feature_row(case: dict[str, Any]) -> dict[str, int]:
    duration_s = finite_positive(case.get("duration_s"), "case duration")
    raw = case.get("features")
    require(type(raw) is dict, "case features")
    result = {"duration_ms": max(1, int(round(duration_s * 1000.0)))}
    for name in FEATURES[1:]:
        value = raw.get(name)
        require(type(value) is int and value >= 0, f"feature {name}")
        result[name] = value
    return result


def load_phone_receipt(result_path: Path) -> dict[str, Any] | None:
    path = result_path.parent / "PHONE_ENERGY.json"
    if not path.exists():
        return None
    value = load_json(path)
    require(
        value.get("schema") == "s42-model-energy-phone-v1" and
        value.get("status") == "PASS" and
        value.get("result_sha256") == digest_file(result_path),
        f"{path}: identity",
    )
    cases = value.get("cases")
    require(type(cases) is list and cases, f"{path}: cases")
    by_id: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("case_id") if type(case) is dict else None
        require(
            type(case_id) is str and case_id not in by_id and
            finite_positive(case.get("whole_phone_energy_j"), "phone energy") > 0,
            f"{path}: phone case",
        )
        by_id[case_id] = case
    return {"by_id": by_id, "path": str(path), "sha256": digest_file(path)}


def observations(
    result_paths: list[Path],
    assumed_phone_power_w: float | None = None,
) -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    rows: list[dict[str, Any]] = []
    identity = None
    evidence = []
    repeat_indices: set[int] = set()
    phone_statuses: set[str] = set()
    for result_path in result_paths:
        value = load_json(result_path)
        require(
            value.get("schema") == "s42-model-energy-result-v1" and
            value.get("status") == "PASS",
            f"{result_path}: result",
        )
        repeat_index = value.get("repeat_index")
        require(
            type(repeat_index) is int and repeat_index >= 1 and
            repeat_index not in repeat_indices,
            f"{result_path}: repeat index",
        )
        repeat_indices.add(repeat_index)
        preflight = value.get("preflight")
        require(type(preflight) is dict, f"{result_path}: preflight")
        current_identity = {
            key: preflight.get(key)
            for key in (
                "case_plan_sha256", "gpu", "model_file_bytes", "model_id",
                "model_catalog_sha256", "model_sha256", "placement", "route",
                "runtime_manifest", "server_sha256",
            )
        }
        if identity is None:
            identity = current_identity
        else:
            prior_gpu = identity["gpu"]
            current_gpu = current_identity["gpu"]
            require(
                type(prior_gpu) is dict and type(current_gpu) is dict and
                prior_gpu.get("uuid") == current_gpu.get("uuid"),
                "GPU identity differs",
            )
            for key in current_identity:
                if key != "gpu":
                    require(identity[key] == current_identity[key], f"{key} differs")
        phone = load_phone_receipt(result_path)
        cases = value.get("cases")
        require(type(cases) is list and cases, f"{result_path}: cases")
        seen: set[str] = set()
        for case in cases:
            case_spec = case.get("case") if type(case) is dict else None
            case_id = case_spec.get("case_id") if type(case_spec) is dict else None
            require(
                type(case_id) is str and case_id not in seen and
                case.get("target_duration_met") is True,
                f"{result_path}: case identity",
            )
            seen.add(case_id)
            components = energy_components(case)
            if phone is not None:
                phone_case = phone["by_id"].get(case_id)
                require(type(phone_case) is dict, f"{result_path}: phone coverage")
                phone_uj = int(round(
                    finite_positive(
                        phone_case.get("whole_phone_energy_j"), "phone energy"
                    ) * 1e6
                ))
                components["phone"] = phone_uj
                components["accounted_fleet"] = components["server_total"] + phone_uj
                phone_statuses.add("measured")
            elif (
                assumed_phone_power_w is not None and
                current_identity["route"] == "gemma-op15"
            ):
                phone_uj = assumed_phone_energy_uj(
                    assumed_phone_power_w,
                    finite_positive(case.get("duration_s"), "case duration"),
                )
                components["phone"] = phone_uj
                components["accounted_fleet"] = components["server_total"] + phone_uj
                phone_statuses.add("estimated")
            rows.append({
                "case_id": case_id,
                "components_uj": components,
                "features": feature_row(case),
                "holdout": case_spec.get("holdout"),
                "kind": case_spec.get("kind"),
                "repeat_index": repeat_index,
            })
        evidence.append({
            "phone": (
                {"status": "not_included"}
                if phone is None and not (
                    assumed_phone_power_w is not None and
                    current_identity["route"] == "gemma-op15"
                )
                else (
                    {
                        "assumed_power_w": assumed_phone_power_w,
                        "status": "estimated",
                    }
                    if phone is None
                    else {
                        "path": phone["path"],
                        "sha256": phone["sha256"],
                        "status": "measured",
                    }
                )
            ),
            "result_path": str(result_path),
            "result_sha256": digest_file(result_path),
        })
    require(identity is not None, "missing identity")
    route = identity["route"]
    require(
        assumed_phone_power_w is None or route == "gemma-op15",
        "assumed phone power applies only to gemma-op15",
    )
    require(len(phone_statuses) <= 1, "mixed measured and estimated phone energy")
    require(
        route != "gemma-op15" or all("phone" in row["components_uj"] for row in rows),
        "Gemma OP15 requires whole-phone energy",
    )
    phone_status = next(iter(phone_statuses), "not_included")
    component_status = {
        "cpu_package": "measured",
        "gpu_board": "measured",
        "server_total": "measured",
    }
    if phone_status != "not_included":
        component_status.update({
            "accounted_fleet": phone_status,
            "phone": phone_status,
        })
    return rows, {
        "component_status": component_status,
        "evidence": evidence,
        "identity": identity,
        "phone_assumption_w": (
            assumed_phone_power_w if phone_status == "estimated" else None
        ),
        "repeat_count": len(repeat_indices),
    }


def fit_nonnegative(
    rows: list[dict[str, Any]], component: str
) -> dict[str, Any]:
    training = [row for row in rows if row["holdout"] is False]
    holdout = [row for row in rows if row["holdout"] is True]
    require(len(training) >= len(FEATURES), f"{component}: training coverage")
    require(holdout, f"{component}: held-out coverage")
    matrix = [
        [float(row["features"][name]) for name in FEATURES]
        for row in training
    ]
    targets = [float(row["components_uj"][component]) for row in training]
    best: tuple[float, tuple[float, ...]] | None = None
    for count in range(1, len(FEATURES) + 1):
        for active in itertools.combinations(range(len(FEATURES)), count):
            gram = [
                [
                    sum(row[left] * row[right] for row in matrix)
                    for right in active
                ]
                for left in active
            ]
            rhs = [
                sum(row[index] * target for row, target in zip(matrix, targets))
                for index in active
            ]
            solved = solve_linear(gram, rhs)
            if solved is None or any(value < -1e-8 for value in solved):
                continue
            coefficients = [0.0] * len(FEATURES)
            for index, value in zip(active, solved):
                coefficients[index] = max(0.0, value)
            squared = sum(
                (sum(a * b for a, b in zip(coefficients, row)) - target) ** 2
                for row, target in zip(matrix, targets)
            )
            candidate = (squared, tuple(coefficients))
            if best is None or candidate < best:
                best = candidate
    require(best is not None, f"{component}: fit")
    rounded = tuple(max(0, int(round(value))) for value in best[1])

    def predict(row: dict[str, Any]) -> int:
        return max(1, sum(
            rounded[index] * row["features"][name]
            for index, name in enumerate(FEATURES)
        ))

    def evaluate(group: list[dict[str, Any]]) -> dict[str, Any]:
        details = []
        for row in group:
            actual = row["components_uj"][component]
            predicted = predict(row)
            error = predicted - actual
            details.append({
                "actual_uj": actual,
                "case_id": row["case_id"],
                "error_pct": 100.0 * error / actual,
                "predicted_uj": predicted,
                "repeat_index": row["repeat_index"],
            })
        absolute_pct = [abs(row["error_pct"]) for row in details]
        positive_ppm = [
            max(0, math.ceil(1_000_000 * (row["actual_uj"] - row["predicted_uj"])
                                     / row["predicted_uj"]))
            for row in details
        ]
        return {
            "details": details,
            "mae_pct": sum(absolute_pct) / len(absolute_pct),
            "max_abs_error_pct": max(absolute_pct),
            "max_actual_over_prediction_ppm": max(positive_ppm),
        }

    train_metrics = evaluate(training)
    holdout_metrics = evaluate(holdout)
    return {
        "coefficients_uj_per_unit": dict(zip(FEATURES, rounded)),
        "feature_units": {
            "decode_steps": "cohort decode steps",
            "duration_ms": "paid milliseconds",
            "input_token_rows": "input token rows",
            "output_token_rows": "output token rows",
            "prefill_ubatches": "estimated 512-row prefill microbatches",
        },
        "holdout": holdout_metrics,
        "training": train_metrics,
    }


def build_profile(
    result_paths: list[Path], assumed_phone_power_w: float | None = None
) -> dict[str, Any]:
    rows, metadata = observations(result_paths, assumed_phone_power_w)
    component_names = sorted(set.intersection(*(
        set(row["components_uj"]) for row in rows
    )))
    models = {
        component: fit_nonnegative(rows, component)
        for component in component_names
    }
    heldout_pass = all(
        models[component]["holdout"]["max_abs_error_pct"] <= 10.0
        for component, status in metadata["component_status"].items()
        if status == "measured"
    )
    repeat_count = metadata["repeat_count"]
    if not heldout_pass:
        verdict = "FAIL_HELDOUT_ERROR"
    elif repeat_count < 3:
        verdict = "PILOT_PASS_REPETITIONS_PENDING"
    elif "estimated" in metadata["component_status"].values():
        verdict = "PASS_SERVER_MEASURED_PHONE_ESTIMATED"
    else:
        verdict = "PASS"
    ranges = {
        name: {
            "max": max(row["features"][name] for row in rows),
            "min": min(row["features"][name] for row in rows),
        }
        for name in FEATURES
    }
    result = {
        "component_models": models,
        "component_status": metadata["component_status"],
        "evidence": metadata["evidence"],
        "extrapolation_allowed": False,
        "feature_ranges": ranges,
        "identity": metadata["identity"],
        "phone_assumption_w": metadata["phone_assumption_w"],
        "repeat_count": repeat_count,
        "schema": "s42-model-energy-profile-v1",
        "verdict": verdict,
    }
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--assumed-phone-power-w", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "output path")
    profile = build_profile(args.result, args.assumed_phone_power_w)
    args.output.write_bytes(canonical(profile))
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": profile["record_sha256"],
        "verdict": profile["verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
