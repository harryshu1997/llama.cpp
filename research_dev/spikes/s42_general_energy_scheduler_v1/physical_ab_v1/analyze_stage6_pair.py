#!/usr/bin/env python3
"""Strictly compare one monitored Stage 6 CPU/OP15 physical pair."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SCHEMA = "s42-stage6-physical-ab-v1"


class PairError(ValueError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PairError(f"cannot read {path}: {exc}") from exc
    if type(value) is not dict:
        raise PairError(f"{path} must contain an object")
    return value


def validate_gate(path: Path, mode: str, repeat_index: int,
                  result_sha256: str) -> dict[str, Any]:
    value = load_object(path)
    supplied = value.get("receipt_sha256")
    unhashed = dict(value)
    unhashed.pop("receipt_sha256", None)
    expected = hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if not (
        value.get("schema") == "s42-stage6-runtime-gate-receipt-v1"
        and value.get("mode") == mode
        and value.get("repeat_index") == repeat_index
        and value.get("result", {}).get("sha256") == result_sha256
        and supplied == expected
    ):
        raise PairError(f"{path}: gate receipt binding")
    return value


def validate_scheduler_binding(
    gate: dict[str, Any], path: Path
) -> dict[str, str] | None:
    scheduler = gate.get("scheduler")
    if scheduler is None:
        return None
    if not (
        type(scheduler) is dict
        and type(gate.get("gates")) is dict
        and gate["gates"].get("unified_scheduler_plan_bound") is True
        and all(
            isinstance(scheduler.get(key), str) and scheduler[key]
            for key in ("decision_reason", "plan_sha256", "route_id")
        )
        and scheduler["plan_sha256"].startswith("sha256:")
    ):
        raise PairError(f"{path}: unified scheduler binding")
    return {
        key: scheduler[key]
        for key in ("decision_reason", "plan_sha256", "route_id")
    }


def percent(treatment: float, control: float) -> float:
    return 100.0 * (treatment / control - 1.0)


def analyze(
    campaign_tools: Path,
    trace_path: Path,
    control_root: Path,
    treatment_root: Path,
    repeat_index: int,
    mmlu_control: Path,
    mmlu_treatment: Path,
) -> dict[str, Any]:
    if str(campaign_tools) not in sys.path:
        sys.path.insert(0, str(campaign_tools))
    try:
        i3 = importlib.import_module("analyze_i3_campaign")
    except ImportError as exc:
        raise PairError(f"cannot import strict I3 validator: {exc}") from exc

    trace = i3.validate_trace(trace_path)
    control = i3.validate_result(control_root, "cpu", repeat_index, trace)
    treatment = i3.validate_result(
        treatment_root, "op15", repeat_index, trace
    )
    quality_control = i3.validate_mmlu(mmlu_control, "cpu")
    quality_treatment = i3.validate_mmlu(mmlu_treatment, "op15")
    if control["runtime_identity"] != treatment["runtime_identity"]:
        raise PairError("runtime identity differs between arms")
    if (
        quality_control["runtime_manifest_signature"]
        != quality_treatment["runtime_manifest_signature"]
    ):
        raise PairError("quality runtime identity differs between arms")

    control_gate_root = Path(f"{control_root}.runtime-gates")
    treatment_gate_root = Path(f"{treatment_root}.runtime-gates")
    control_gate_path = control_gate_root / "RUNTIME_GATE_RECEIPT.json"
    treatment_gate_path = treatment_gate_root / "RUNTIME_GATE_RECEIPT_V2.json"
    if not treatment_gate_path.exists():
        treatment_gate_path = treatment_gate_root / "RUNTIME_GATE_RECEIPT.json"
    control_gate = validate_gate(
        control_gate_path,
        "cpu",
        repeat_index,
        control["result_sha256"],
    )
    treatment_gate = validate_gate(
        treatment_gate_path,
        "op15",
        repeat_index,
        treatment["result_sha256"],
    )
    control_scheduler = validate_scheduler_binding(
        control_gate, control_gate_path
    )
    treatment_scheduler = validate_scheduler_binding(
        treatment_gate, treatment_gate_path
    )
    if (control_scheduler is None) != (treatment_scheduler is None):
        raise PairError("only one arm is bound to the unified scheduler")

    duration_change = percent(treatment["duration_s"], control["duration_s"])
    fleet_change = percent(treatment["fleet_j"], control["fleet_j"])
    server_change = percent(
        treatment["server"]["server_j"], control["server"]["server_j"]
    )
    quality_delta = quality_treatment["correct"] - quality_control["correct"]
    wait_fraction = treatment["work"]["exposed_join_wait_fraction"]
    gates = {
        "accounted_fleet_energy_saving_at_least_10_percent": fleet_change <= -10.0,
        "approximate_quality_noninferior": (
            quality_treatment["correct"] >= 25 and quality_delta >= -1
        ),
        "control_runtime_gates": control_gate.get("status") == "PASS",
        "equal_completed_work": True,
        "join_wait_at_most_5_percent": wait_fraction <= 0.05,
        "makespan_not_regressed": duration_change <= 0.0,
        "slo_count_not_lower": treatment["slo_met"] >= control["slo_met"],
        "treatment_runtime_gates": treatment_gate.get("status") == "PASS",
        "zero_bridge_reset_recoveries": (
            treatment["work"]["reset_recoveries"] == 0
        ),
    }
    result: dict[str, Any] = {
        "comparison": {
            "control": {
                "cpu_package_j": control["server"]["cpu_package_j"],
                "duration_s": control["duration_s"],
                "fleet_j": control["fleet_j"],
                "gpu_board_j": control["server"]["gpu_board_j"],
                "phone_j": control["phone"]["energy_j"],
                "server_j": control["server"]["server_j"],
                "slo_met": control["slo_met"],
            },
            "fleet_energy_change_pct": fleet_change,
            "makespan_change_pct": duration_change,
            "server_energy_change_pct": server_change,
            "treatment": {
                "cpu_package_j": treatment["server"]["cpu_package_j"],
                "duration_s": treatment["duration_s"],
                "fleet_j": treatment["fleet_j"],
                "gpu_board_j": treatment["server"]["gpu_board_j"],
                "phone_j": treatment["phone"]["energy_j"],
                "server_j": treatment["server"]["server_j"],
                "slo_met": treatment["slo_met"],
            },
        },
        "gates": gates,
        "phone_work": {
            **treatment["work"],
            "executor_heartbeat_max_gap_s": treatment_gate["host"][
                "executor_heartbeat_max_gap_s"
            ],
            "rpc_progress_max_gap_s": treatment_gate["transport"][
                "progress_max_gap_s"
            ],
            "phone_temperature_max_millic": treatment_gate["phone"][
                "max_temperature_millic"
            ],
        },
        "quality": {
            "authority": "pinned_mmlu64_same_runtime_manifest",
            "control_correct": quality_control["correct"],
            "delta": quality_delta,
            "treatment_correct": quality_treatment["correct"],
        },
        "repeat_index": repeat_index,
        "repetition_scope": (
            "one successor validation pair; existing I3 three-pair certificate "
            "remains the promotion authority"
        ),
        "runtime_gate_receipts": {
            "control": control_gate["receipt_sha256"],
            "treatment": treatment_gate["receipt_sha256"],
        },
        "schema": SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "trace": {
            "input_tokens": 33843,
            "output_tokens": 11605,
            "requests": 74,
            "sha256": i3.TRACE_SHA256,
        },
    }
    if control_scheduler is not None and treatment_scheduler is not None:
        result["scheduler_plans"] = {
            "control": control_scheduler,
            "treatment": treatment_scheduler,
        }
    result["record_sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-tools", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--treatment-root", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--mmlu-control", type=Path, required=True)
    parser.add_argument("--mmlu-treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat_index < 1:
        parser.error("repeat index must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = analyze(
            args.campaign_tools,
            args.trace,
            args.control_root,
            args.treatment_root,
            args.repeat_index,
            args.mmlu_control,
            args.mmlu_treatment,
        )
        args.output.write_bytes(canonical_bytes(result))
    except (PairError, ValueError) as exc:
        parser.exit(2, f"stage6 pair analysis failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": result["record_sha256"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
