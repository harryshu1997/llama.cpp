#!/usr/bin/env python3
"""Compare static CPU and runtime scheduling under one large-model policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
PHONE_SCHEMA = "s41-phone-energy-v3"
COMPARISON_SCHEMA = "s42-full-fp16-llama1b-comparison-v2"
QUALIFICATION_SCHEMA = "s42-fp16-llama1b-physical-qualification-v1"


class ComparisonError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


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


def stable_work_identity(result: dict[str, Any]) -> dict[str, Any]:
    inputs = {
        name: value
        for name, value in result["input_sha256"].items()
        if name not in {"automated_catalog", "resident_release"}
    }
    require(
        set(inputs) == {
            "base_trace", "manifest", "overlay_trace", "runtime_profile"
        },
        "stable input identity",
    )
    return {
        "input_sha256": inputs,
        "marginal_system_profile": result.get("scheduler_runtime", {}).get(
            "marginal_system_profile"
        ),
        "model_identities": result["model_identities"],
    }


def stable_work_receipts(result: dict[str, Any]) -> dict[str, Any]:
    receipts = result.get("work_receipts", {})
    stable = {}
    for name in ("large_model_outputs", "small_model_outputs"):
        receipt = receipts.get(name, {})
        request_count = receipt.get("request_count")
        output_tokens = receipt.get("output_tokens")
        actual_output_tokens = receipt.get(
            "actual_output_tokens", output_tokens
        )
        require(
            type(request_count) is int
            and request_count > 0
            and type(output_tokens) is int
            and output_tokens > 0
            and actual_output_tokens == output_tokens,
            "stable output shape receipt",
        )
        stable[name] = {
            "actual_output_tokens": actual_output_tokens,
            "output_tokens": output_tokens,
            "request_count": request_count,
        }
    return stable


def output_shape_receipts(result: dict[str, Any]) -> dict[str, str]:
    shapes = {}
    for name, receipt in result.get("work_receipts", {}).items():
        shape_sha256 = receipt.get("shape_sha256")
        if shape_sha256 is not None:
            require(
                type(shape_sha256) is str and len(shape_sha256) == 64,
                "output shape digest",
            )
            shapes[name] = shape_sha256
    return shapes


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def validate(
    result_path: Path,
    phone_path: Path,
    expected_small_model_policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = load(result_path)
    phone = load(phone_path)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS"
        and result.get("policy", {}).get("small_model_policy")
            == expected_small_model_policy
        and result.get("policy", {}).get("large_model_policy")
            in {"cpu-overflow", "op15-assistance"},
        "combined result status or policy",
    )
    require(
        phone.get("schema") == PHONE_SCHEMA
        and phone.get("status") == "PASS"
        and phone.get("boundary") == "paid_trace_interval",
        "phone energy status",
    )
    metrics = result["metrics"]
    base = result.get("base", {})
    small = metrics.get("small_model", {})
    small_count = small.get("completed")
    small_input_tokens = small.get("input_tokens")
    small_output_tokens = small.get("output_tokens")
    require(
        base.get("request_count") == 74
        and base.get("output_tokens") == 11_605
        and type(small_count) is int
        and small_count > 0
        and type(small_input_tokens) is int
        and small_input_tokens > 0
        and type(small_output_tokens) is int
        and small_output_tokens > 0
        and metrics.get("completed") == base["request_count"] + small_count
        and metrics.get("input_tokens") == 33_843 + small_input_tokens
        and metrics.get("output_tokens")
            == base["output_tokens"] + small_output_tokens,
        "combined work conservation",
    )
    receipts = result.get("work_receipts", {})
    require(
        receipts.get("large_model_outputs", {}).get("request_count")
            == base["request_count"]
        and receipts.get("large_model_outputs", {}).get("output_tokens")
            == base["output_tokens"]
        and receipts.get("small_model_outputs", {}).get("request_count")
            == small_count
        and receipts.get("small_model_outputs", {}).get("output_tokens")
            == small_output_tokens,
        "output work receipts",
    )
    require(
        abs(phone["duration_s"] - metrics["duration_s"]) < 1e-6,
        "phone and server paid intervals differ",
    )
    require(
        result["resources"]["llama1_cpu_swap_max_bytes"] == 0,
        "Llama CPU process swapped",
    )
    if expected_small_model_policy == "static-cpu":
        require(
            result["scheduler_runtime"]["enabled"] is False
            and result["scheduler_runtime"]["route_counts"]
                == {"desktop-cpu": small_count},
            "static CPU binding",
        )
    else:
        require(
            result["scheduler_runtime"]["enabled"] is True
            and len(result["scheduler_runtime"]["decisions"])
                == small_count
            and result["scheduler_runtime"]["overhead"] is not None,
            "runtime scheduler binding",
        )
    require(
        sum(result["scheduler_runtime"]["route_counts"].values())
            == small_count,
        "physical route count",
    )
    expected_arm = (
        "control"
        if result["policy"]["large_model_policy"] == "cpu-overflow"
        else "op15"
    )
    require(
        result["base"]["arm"] == expected_arm
        and result["policy"]["large_model_arm"] == expected_arm,
        "large-model policy binding",
    )
    return result, phone


def summary(result: dict[str, Any], phone: dict[str, Any]) -> dict[str, Any]:
    server = result["server_energy"]
    server_j = server["server_compute_device_energy_j"]
    phone_j = phone["whole_phone_energy_j"]
    return {
        "cpu_package_energy_j": server["cpu_package_energy_j"],
        "duration_s": result["metrics"]["duration_s"],
        "fleet_energy_j": server_j + phone_j,
        "gpu_board_energy_j": server["gpu_board_energy_j"],
        "gpu_utilization_mean_pct": result["resources"]
            ["gpu_utilization_pct"]["mean"],
        "phone_energy_j": phone_j,
        "route_counts": result["scheduler_runtime"]["route_counts"],
        "server_energy_j": server_j,
        "small_model": result["metrics"]["small_model"],
        "slo_met": result["metrics"]["slo_met"],
        "throughput_tokens_s": result["metrics"]
            ["output_throughput_tokens_s"],
    }


def validate_qualification(
    path: Path,
    result_path: Path,
    result: dict[str, Any],
    require_nonbaseline: bool,
) -> dict[str, Any]:
    value = load(path)
    gates = value.get("gates", {})
    required_gates = {
        "all_endpoint_tasks_in_selected_server_log",
        "all_leases_cover_physical_execution",
        "all_original_latency_upper_bounds_met",
        "all_runtime_snapshots_live",
        "decision_matches_physical_endpoint",
        "large_request_receipts_causal",
        "nonbaseline_physical_execution",
        "overlay_request_receipts_causal",
        "phase_transport_leases_released",
        "resident_release_after_final_phone_snapshot",
        "trace_model_energy_identity",
    }
    optional_gates = {"selected_split_route_has_phone_ffn_calls"}
    split_selected = result.get("scheduler_runtime", {}).get(
        "route_counts", {}
    ).get("cpu-phone-ffn-split", 0) > 0
    require(
        value.get("schema") == QUALIFICATION_SCHEMA
        and value.get("status") == "PASS"
        and required_gates.issubset(gates)
        and set(gates).issubset(required_gates | optional_gates)
        and (
            not split_selected
            or gates.get("selected_split_route_has_phone_ffn_calls") is True
        )
        and value.get("input_sha256", {}).get("result")
            == digest(result_path)
        and value.get("policy") == result.get("policy")
        and all(
            passed
            for name, passed in gates.items()
            if name != "nonbaseline_physical_execution"
        )
        and (
            not require_nonbaseline
            or value.get("nonbaseline_physical_executions", 0) > 0
        ),
        "physical route qualification",
    )
    return value


def compare(
    static_result_path: Path,
    static_phone_path: Path,
    runtime_result_path: Path,
    runtime_phone_path: Path,
    static_qualification_path: Path | None = None,
    runtime_qualification_path: Path | None = None,
) -> dict[str, Any]:
    static_result, static_phone = validate(
        static_result_path, static_phone_path, "static-cpu"
    )
    runtime_result, runtime_phone = validate(
        runtime_result_path, runtime_phone_path, "runtime-scheduler"
    )
    if static_qualification_path is not None:
        static_qualification = validate_qualification(
            static_qualification_path,
            static_result_path,
            static_result,
            False,
        )
    else:
        static_qualification = None
    if runtime_qualification_path is not None:
        runtime_qualification = validate_qualification(
            runtime_qualification_path,
            runtime_result_path,
            runtime_result,
            True,
        )
    else:
        runtime_qualification = None
    require(
        static_result["policy"]["large_model_policy"]
            == runtime_result["policy"]["large_model_policy"]
        and static_result["base"]["arm"] == runtime_result["base"]["arm"],
        "large-model policy differs",
    )
    static_work_identity = stable_work_identity(static_result)
    runtime_work_identity = stable_work_identity(runtime_result)
    static_work_receipts = stable_work_receipts(static_result)
    runtime_work_receipts = stable_work_receipts(runtime_result)
    static_output_shapes = output_shape_receipts(static_result)
    runtime_output_shapes = output_shape_receipts(runtime_result)
    require(
        static_work_identity == runtime_work_identity
        and static_work_receipts == runtime_work_receipts
        and static_output_shapes == runtime_output_shapes,
        "work or model identity differs",
    )
    require(
        static_result["server_energy"].get("boundary")
            == runtime_result["server_energy"].get("boundary")
            == static_phone.get("boundary")
            == runtime_phone.get("boundary")
            == "paid_trace_interval",
        "energy boundary differs",
    )
    static = summary(static_result, static_phone)
    runtime = summary(runtime_result, runtime_phone)
    fleet_saving_j = static["fleet_energy_j"] - runtime["fleet_energy_j"]
    duration_saving_s = static["duration_s"] - runtime["duration_s"]
    return {
        "claim_scope": "small-model scheduler at fixed large-model policy",
        "delta": {
            "cpu_package_energy_saving_j": (
                static["cpu_package_energy_j"]
                - runtime["cpu_package_energy_j"]
            ),
            "duration_saving_pct": (
                100 * duration_saving_s / static["duration_s"]
            ),
            "duration_saving_s": duration_saving_s,
            "fleet_energy_saving_j": fleet_saving_j,
            "fleet_energy_saving_pct": (
                100 * fleet_saving_j / static["fleet_energy_j"]
            ),
            "gpu_board_energy_saving_j": (
                static["gpu_board_energy_j"]
                - runtime["gpu_board_energy_j"]
            ),
            "phone_energy_change_j": (
                runtime["phone_energy_j"] - static["phone_energy_j"]
            ),
            "slo_gain": runtime["slo_met"] - static["slo_met"],
            "throughput_gain_pct": 100 * (
                runtime["throughput_tokens_s"]
                / static["throughput_tokens_s"]
                - 1
            ),
        },
        "input_sha256": {
            "runtime_phone": digest(runtime_phone_path),
            "runtime_result": digest(runtime_result_path),
            "static_phone": digest(static_phone_path),
            "static_result": digest(static_result_path),
        },
        "execution_output_receipts": {
            "runtime_scheduler": runtime_result["work_receipts"],
            "static_cpu": static_result["work_receipts"],
        },
        "large_model_policy": static_result["policy"]["large_model_policy"],
        "outcome": {
            "duration_improved": duration_saving_s > 0,
            "fleet_energy_improved": fleet_saving_j > 0,
            "slo_not_regressed": runtime["slo_met"] >= static["slo_met"],
        },
        "output_shape_receipts": static_output_shapes,
        "runtime_scheduler": runtime,
        "physical_qualification": {
            "runtime_scheduler": runtime_qualification,
            "static_cpu": static_qualification,
        },
        "scheduling_overhead": runtime_result["scheduler_runtime"]
            ["overhead"],
        "schema": COMPARISON_SCHEMA,
        "static_cpu": static,
        "status": "PASS",
        "work_identity": static_work_identity,
        "work_receipts": static_work_receipts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-result", type=Path, required=True)
    parser.add_argument("--static-phone", type=Path, required=True)
    parser.add_argument("--runtime-result", type=Path, required=True)
    parser.add_argument("--runtime-phone", type=Path, required=True)
    parser.add_argument("--static-qualification", type=Path, required=True)
    parser.add_argument("--runtime-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "output must be a new absolute path",
    )
    result = compare(
        args.static_result,
        args.static_phone,
        args.runtime_result,
        args.runtime_phone,
        args.static_qualification,
        args.runtime_qualification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    print(json.dumps({
        "delta": result["delta"],
        "output": str(args.output),
        "scheduling_overhead": result["scheduling_overhead"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
