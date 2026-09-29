#!/usr/bin/env python3
"""Fit pooled marginal fleet costs from the repeated 2x2 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
BASE_RESULT_SCHEMA = "s41-hierarchical-burstgpt-result-v1"
PHONE_SCHEMA = "s41-phone-energy-v3"
PROFILE_SCHEMA = "s42-fp16-overlay-marginal-system-profile-v1"
RUNS = {
    "cpu-overflow": {
        "static": ("cpu-static-r1", "cpu-static-r2"),
        "runtime": ("cpu-runtime-r1", "cpu-runtime-r2"),
    },
    "op15-assistance": {
        "static": ("op15-static-r1", "op15-static-r2"),
        "runtime": ("op15-runtime-r1", "op15-runtime-r2"),
    },
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


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def load_run(
    root: Path,
    name: str,
    large_policy: str,
    small_policy: str,
) -> dict[str, Any]:
    result_path = root / name / "combined/RESULT.json"
    base_result_path = root / name / "base/RESULT.json"
    phone_path = root / name / "capture/PHONE_ENERGY.json"
    samples_path = root / name / "combined/resource-samples.jsonl"
    result = load(result_path)
    base_result = load(base_result_path)
    phone = load(phone_path)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and result.get("policy", {}).get("large_model_policy")
            == large_policy
        and result.get("policy", {}).get("small_model_policy")
            == small_policy
        and phone.get("schema") == PHONE_SCHEMA
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval"
        and base_result.get("schema") == BASE_RESULT_SCHEMA
        and base_result.get("status") == "PASS"
        and base_result.get("mode") == "fp16-switch"
        and result.get("base", {}).get("result_sha256")
            == digest(base_result_path)
        and samples_path.is_file(),
        f"physical run identity: {name}",
    )
    return {
        "name": name,
        "base_result": base_result,
        "base_result_path": base_result_path,
        "phone": phone,
        "phone_path": phone_path,
        "resource_samples_path": samples_path,
        "result": result,
        "result_path": result_path,
    }


def active_cpu_service_by_phase_us(
    run: dict[str, Any],
) -> dict[str, int]:
    rows = run["result"].get("request_results")
    require(type(rows) is list and rows, "request results")
    totals = {phase: 0 for phase in ("gemma", "qwen", "switching")}
    for row in rows:
        context = row.get("runtime_context", {})
        phase = context.get("large_model", {}).get("phase")
        if row.get("route") not in {"desktop-cpu", "cpu-phone-ffn-split"}:
            continue
        if phase == "idle":
            continue
        prompt_ms = row.get("prompt_ms")
        predicted_ms = row.get("predicted_ms")
        require(
            type(prompt_ms) in {int, float}
            and not isinstance(prompt_ms, bool)
            and prompt_ms >= 0
            and type(predicted_ms) in {int, float}
            and not isinstance(predicted_ms, bool)
            and predicted_ms > 0,
            "CPU execution interval",
        )
        require(phase in totals, "active large-model phase")
        totals[phase] += math.ceil((prompt_ms + predicted_ms) * 1000)
    return totals


def active_cpu_service_us(run: dict[str, Any]) -> int:
    return sum(active_cpu_service_by_phase_us(run).values())


def large_phase_durations_us(run: dict[str, Any]) -> dict[str, int]:
    base = run.get("base_result")
    require(type(base) is dict, "large-model base result")
    switch = base.get("switch")
    require(type(switch) is dict, "large-model switch receipt")
    hot_end_s = switch.get("hot_end_s")
    gpu_ready_s = switch.get("gpu_ready_s")
    paid_start_ns = base.get("paid_start_ns")
    paid_end_ns = base.get("paid_end_ns")
    require(
        type(hot_end_s) in {int, float}
        and not isinstance(hot_end_s, bool)
        and type(gpu_ready_s) in {int, float}
        and not isinstance(gpu_ready_s, bool)
        and type(paid_start_ns) is int
        and type(paid_end_ns) is int
        and paid_end_ns > paid_start_ns,
        "large-model phase timing receipt",
    )
    qwen_us = round(hot_end_s * 1_000_000)
    gpu_ready_us = round(gpu_ready_s * 1_000_000)
    duration_us = (paid_end_ns - paid_start_ns) // 1000
    require(
        0 < qwen_us <= gpu_ready_us <= duration_us,
        "ordered large-model phase timing",
    )
    return {
        "gemma": duration_us - gpu_ready_us,
        "qwen": qwen_us,
        "switching": gpu_ready_us - qwen_us,
    }


def large_model_duration_us(run: dict[str, Any]) -> int:
    return sum(large_phase_durations_us(run).values())


def gpu_idle_power_mw(run: dict[str, Any]) -> int:
    result = run["result"]
    start_ns = result["paid_start_ns"]
    end_ns = result["paid_end_ns"]
    values = []
    for raw in run["resource_samples_path"].read_bytes().splitlines():
        row = json.loads(raw)
        gpu = row.get("gpu", {})
        sample_ns = gpu.get("sample_t_ns")
        utilization = gpu.get("utilization_pct")
        power_mw = gpu.get("power_mw")
        if (
            type(sample_ns) is int
            and start_ns <= sample_ns <= end_ns
            and type(utilization) is int
            and utilization <= 5
            and type(power_mw) is int
        ):
            values.append(power_mw)
    require(len(values) >= 10, "GPU idle power sample count")
    return round(sum(values) / len(values))


def pair_interference_ppm(
    static: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[int, dict[str, int | str]]:
    static_service_us = active_cpu_service_us(static)
    runtime_service_us = active_cpu_service_us(runtime)
    service_delta_us = static_service_us - runtime_service_us
    static_duration_us = large_model_duration_us(static)
    runtime_duration_us = large_model_duration_us(runtime)
    duration_delta_us = static_duration_us - runtime_duration_us
    measured = service_delta_us > 0 and duration_delta_us > 0
    ppm = (
        min(
            1_000_000,
            round(duration_delta_us * 1_000_000 / service_delta_us),
        )
        if measured
        else 0
    )
    return ppm, {
        "duration_delta_us": duration_delta_us,
        "runtime_large_model_duration_us": runtime_duration_us,
        "runtime": runtime["name"],
        "runtime_active_cpu_service_us": runtime_service_us,
        "service_delta_us": service_delta_us,
        "static": static["name"],
        "static_active_cpu_service_us": static_service_us,
        "static_large_model_duration_us": static_duration_us,
        "status": "MEASURED" if measured else "UNMEASURED_NONPOSITIVE_DELTA",
    }


def pair_phase_interference_ppm(
    static: dict[str, Any],
    runtime: dict[str, Any],
    phase: str,
) -> tuple[int, dict[str, int | str]]:
    static_by_phase = active_cpu_service_by_phase_us(static)
    runtime_by_phase = active_cpu_service_by_phase_us(runtime)
    require(phase in static_by_phase, "marginal active phase")
    service_delta_us = static_by_phase[phase] - runtime_by_phase[phase]
    static_phase_duration_us = large_phase_durations_us(static)[phase]
    runtime_phase_duration_us = large_phase_durations_us(runtime)[phase]
    duration_delta_us = (
        static_phase_duration_us - runtime_phase_duration_us
    )
    measured = service_delta_us > 0 and duration_delta_us > 0
    coefficient = (
        min(
            1_000_000,
            round(duration_delta_us * 1_000_000 / service_delta_us),
        )
        if measured else 0
    )
    return coefficient, {
        "duration_delta_us": duration_delta_us,
        "phase": phase,
        "runtime_active_cpu_service_us": runtime_by_phase[phase],
        "runtime_phase_duration_us": runtime_phase_duration_us,
        "service_delta_us": service_delta_us,
        "static_active_cpu_service_us": static_by_phase[phase],
        "static_phase_duration_us": static_phase_duration_us,
        "status": "MEASURED" if measured else "UNMEASURED_NONPOSITIVE_DELTA",
    }


def mean_int(values: list[float | int]) -> int:
    require(bool(values), "nonempty mean")
    return round(sum(values) / len(values))


def fit_arm(
    large_policy: str,
    static_runs: list[dict[str, Any]],
    runtime_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    require(
        len(static_runs) == len(runtime_runs) == 2,
        "two matched marginal pairs",
    )
    coefficients = []
    pairs = []
    for static, runtime in zip(static_runs, runtime_runs):
        coefficient, receipt = pair_interference_ppm(static, runtime)
        coefficients.append(coefficient)
        pairs.append(receipt)
    measured_coefficients = [
        coefficient
        for coefficient, receipt in zip(coefficients, pairs)
        if receipt["status"] == "MEASURED"
    ]
    pooled_measured = len(measured_coefficients) == len(coefficients)
    pooled_coefficient = (
        mean_int(measured_coefficients) if measured_coefficients else 0
    )
    phase_coefficients: dict[str, int] = {}
    phase_receipts: dict[str, list[dict[str, int | str]]] = {}
    phase_measured: dict[str, bool] = {}
    for phase in ("gemma", "qwen", "switching"):
        values = []
        receipts = []
        for static, runtime in zip(static_runs, runtime_runs):
            value, receipt = pair_phase_interference_ppm(
                static, runtime, phase
            )
            values.append(value)
            receipts.append(receipt)
        observed = [
            value for value, receipt in zip(values, receipts)
            if receipt["status"] == "MEASURED"
        ]
        phase_measured[phase] = len(observed) == 2
        phase_coefficients[phase] = (
            mean_int(observed) if phase_measured[phase] else 0
        )
        phase_receipts[phase] = receipts
    measured_values = [
        phase_coefficients[phase]
        for phase in phase_coefficients
        if phase_measured[phase]
    ]
    measured = pooled_measured and bool(measured_values)
    reference_coefficient = (
        mean_int(measured_values) if measured_values else pooled_coefficient
    )
    relative_error_ppm = 0
    if reference_coefficient > 0:
        relative_error_ppm = math.ceil(
            max(
                abs(value - reference_coefficient)
                for value in (measured_values or measured_coefficients)
            )
            * 1_000_000
            / reference_coefficient
        )
    error_ppm = min(1_000_000, max(250_000, relative_error_ppm))
    runs = static_runs + runtime_runs
    cpu_power_mw = mean_int([
        run["result"]["server_energy"]["cpu_package_average_power_w"]
            * 1000
        for run in runs
    ])
    phone_power_mw = mean_int([
        run["phone"]["whole_phone_average_power_w"] * 1000
        for run in runs
    ])
    phase_server_power_mw = mean_int([
        (
            run["result"]["server_energy"]["cpu_package_average_power_w"]
            + run["result"]["server_energy"]["gpu_board_average_power_w"]
        ) * 1000
        for run in runs
    ])
    idle_gpu_power_mw = mean_int([
        gpu_idle_power_mw(run) for run in runs
    ])
    phases = {}
    for phase in ("gemma", "qwen", "switching"):
        coefficient = phase_coefficients[phase]
        phases[phase] = {
            "measured": phase_measured[phase],
            "route_cpu_interference_ppm": {
                "cpu-phone-ffn-split": coefficient,
                "desktop-cpu": coefficient,
                "desktop-cuda": 0,
                "phone-adreno": 0,
            },
        }
    phases["idle"] = {
        "measured": True,
        "route_cpu_interference_ppm": {
            route: 0 for route in (
                "cpu-phone-ffn-split",
                "desktop-cpu",
                "desktop-cuda",
                "phone-adreno",
            )
        },
    }
    duration_upper_us = math.ceil(
        max(run["result"]["metrics"]["duration_s"] for run in runs)
        * 1_050_000
    )
    return {
        "causal_tail_power_mw": cpu_power_mw + phone_power_mw,
        "evidence": {
            "interference_ppm_pairs": coefficients,
            "pair_receipts": pairs,
            "phase_interference_ppm": phase_coefficients,
            "phase_pair_receipts": phase_receipts,
            "phase_resolution": "per_observed_active_phase",
            "split_interference": (
                "conservative desktop-CPU bound until split-specific "
                "repeated qualification"
            ),
        },
        "gpu_idle_power_mw": idle_gpu_power_mw,
        "lower_error_ppm": error_ppm,
        "measured": measured,
        "phase_power_mw": phase_server_power_mw,
        "phone_phase_power_mw": phone_power_mw,
        "phases": phases,
        "sample_count": max(1, len(measured_coefficients)),
        "trace_duration_upper_us": duration_upper_us,
        "upper_error_ppm": error_ppm,
    }


def fit(root: Path) -> dict[str, Any]:
    arms = {}
    input_sha256 = {}
    for policy, names in RUNS.items():
        static = [
            load_run(root, name, policy, "static-cpu")
            for name in names["static"]
        ]
        runtime = [
            load_run(root, name, policy, "runtime-scheduler")
            for name in names["runtime"]
        ]
        arms[policy] = fit_arm(policy, static, runtime)
        for run in static + runtime:
            input_sha256[run["name"]] = {
                "base_result": digest(run["base_result_path"]),
                "phone_energy": digest(run["phone_path"]),
                "resource_samples": digest(run["resource_samples_path"]),
                "result": digest(run["result_path"]),
            }
    arm_qualification = {
        policy: (
            arm.get("measured") is True
            and all(
                phase.get("measured") is True
                for phase in arm.get("phases", {}).values()
            )
            and set(arm.get("phases", {}))
                == {"gemma", "idle", "qwen", "switching"}
        )
        for policy, arm in arms.items()
    }
    return {
        "arms": arms,
        "input_sha256": input_sha256,
        "profile_id": "s42-fp16-overlay-marginal-system-2x2-abba-v1",
        "qualification": {
            "arms": arm_qualification,
            "required_arms": ["cpu-overflow", "op15-assistance"],
            "status": (
                "PASS" if all(arm_qualification.values()) else "FAIL"
            ),
        },
        "schema": PROFILE_SCHEMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.experiment_root.is_absolute()
        and args.experiment_root.is_dir()
        and args.output.is_absolute()
        and not args.output.exists(),
        "existing absolute experiment root and new absolute output",
    )
    value = fit(args.experiment_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "output": str(args.output),
        "profile_id": value["profile_id"],
        "status": value["qualification"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
