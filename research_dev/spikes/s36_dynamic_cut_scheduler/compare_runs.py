#!/usr/bin/env python3
"""Apply the frozen S36 control/treatment gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profiles import canonical_bytes


SCHEMA = "s36-dynamic-cut-comparison-v1"
RUN_SCHEMA = "s36-dynamic-cut-physical-run-v1"


class ComparisonError(RuntimeError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_run(path: Path, mode: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot load run: {exc}") from exc
    if (
        type(value) is not dict
        or value.get("schema") != RUN_SCHEMA
        or value.get("status") != "RUN_COMPLETE"
        or value.get("mode") != mode
    ):
        raise ComparisonError(f"{mode} run is not complete")
    return value


def compare(control: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    problems = []
    if control["trace"]["trace_hash"] != treatment["trace"]["trace_hash"]:
        problems.append("TRACE_MISMATCH")
    if control["profiles"]["profile_hash"] != treatment["profiles"]["profile_hash"]:
        problems.append("PROFILE_MISMATCH")
    control_rows = {row["request_id"]: row for row in control["runtime"]["requests"]}
    treatment_rows = {
        row["request_id"]: row for row in treatment["runtime"]["requests"]
    }
    if len(control_rows) != 60 or set(control_rows) != set(treatment_rows):
        problems.append("REQUEST_CONSERVATION")
    elif any(
        control_rows[request_id]["output_tokens"]
        != treatment_rows[request_id]["output_tokens"]
        for request_id in control_rows
    ):
        problems.append("TOKEN_MISMATCH")
    if set(control["summary"]["route_distribution"]) != {"cuda-c4"}:
        problems.append("CONTROL_ROUTE_MISMATCH")
    treatment_devices = set(treatment["summary"]["device_distribution"])
    if not {"op12", "op15"}.issubset(treatment_devices):
        problems.append("BOTH_PHONES_NOT_USED")
    if not {"4", "8"}.issubset(treatment["summary"]["cut_distribution"]):
        problems.append("BOTH_CUTS_NOT_USED")
    if treatment["summary"]["phone_mixed_phase_batches"] < 1:
        problems.append("NO_PHONE_MIXED_PHASE_BATCH")

    control_p0 = control["summary"]["priority"]["0"]
    treatment_p0 = treatment["summary"]["priority"]["0"]
    if treatment_p0["slo_misses"] > control_p0["slo_misses"]:
        problems.append("PRIORITY_ZERO_SLO_REGRESSION")
    control_p95 = control_p0["latency_us"]["p95"]
    treatment_p95 = treatment_p0["latency_us"]["p95"]
    if (
        type(control_p95) is not int
        or type(treatment_p95) is not int
        or treatment_p95 * 100 > control_p95 * 105
    ):
        problems.append("PRIORITY_ZERO_P95_REGRESSION")
    control_cuda = control["summary"]["selected_cuda_compute_us"]
    treatment_cuda = treatment["summary"]["selected_cuda_compute_us"]
    if (
        type(control_cuda) is not int
        or type(treatment_cuda) is not int
        or treatment_cuda >= control_cuda
    ):
        problems.append("NO_SELECTED_CUDA_COMPUTE_RELIEF")
    for name, run in (("control", control), ("treatment", treatment)):
        state = run["final_state"]
        if (
            any(state["active_counts"].values())
            or state["tail_active"] != 0
            or state["route_pins"]
            or any(state["software_leases"].values())
        ):
            problems.append(f"{name.upper()}_STATE_NOT_DRAINED")

    return {
        "schema": SCHEMA,
        "verdict": "PASS" if not problems else "FAIL",
        "problems": problems,
        "gates": {
            "request_count": len(treatment_rows),
            "token_equal": "TOKEN_MISMATCH" not in problems,
            "both_phones": "BOTH_PHONES_NOT_USED" not in problems,
            "both_cuts": "BOTH_CUTS_NOT_USED" not in problems,
            "phone_mixed_phase_batches": treatment["summary"][
                "phone_mixed_phase_batches"
            ],
            "priority_zero_control_p95_us": control_p95,
            "priority_zero_treatment_p95_us": treatment_p95,
            "priority_zero_control_misses": control_p0["slo_misses"],
            "priority_zero_treatment_misses": treatment_p0["slo_misses"],
            "control_selected_cuda_compute_us": control_cuda,
            "treatment_selected_cuda_compute_us": treatment_cuda,
            "selected_cuda_compute_relief_percent": (
                (control_cuda - treatment_cuda) * 100.0 / control_cuda
                if type(control_cuda) is int and control_cuda > 0
                else None
            ),
        },
        "scope": {
            "physical_scheduler_mechanics": "MEASURED",
            "selected_cuda_stage_wall": "MEASURED",
            "gpu_board_energy": "NOT_MEASURED",
            "phone_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        report = compare(
            load_run(args.control, "ALL_CUDA_CONTROL"),
            load_run(args.treatment, "DYNAMIC_CUT_TREATMENT"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0 if report["verdict"] == "PASS" else 2
    except Exception as exc:
        print(json.dumps({
            "verdict": "FAIL", "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
