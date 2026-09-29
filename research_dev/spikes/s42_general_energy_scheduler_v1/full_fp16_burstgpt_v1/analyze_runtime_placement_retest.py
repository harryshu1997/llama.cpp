#!/usr/bin/env python3
"""Validate one runtime-selected full FP16 BurstGPT retest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from analyze_full_fp16_abba import (
    TRACE_SHA256,
    canonical,
    compact,
    read_object,
    require,
    sha256,
    validate_result,
)


CERTIFICATE_SCHEMA = "s42-full-fp16-burstgpt-abba-v1"
OUTPUT_SCHEMA = "s42-runtime-placement-retest-v1"
METRIC_KEYS = (
    "duration_s",
    "cpu_package_j",
    "gpu_board_j",
    "server_j",
    "phone_j",
    "fleet_j",
    "switch_s",
)


def change(reference: float, observed: float) -> float:
    return 100.0 * (observed / reference - 1.0)


def validate_certificate(path: Path) -> dict[str, Any]:
    value = read_object(path)
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == CERTIFICATE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("energy_verdict") == "SAVING"
        and value.get("trace") == {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "requests": 74,
            "source_sha256": TRACE_SHA256,
        }
        and type(claimed_hash) is str
        and claimed_hash == hashlib.sha256(canonical(unsigned)).hexdigest(),
        f"baseline certificate identity: {path}",
    )
    for name in ("control_mean", "treatment_mean"):
        row = value.get(name)
        require(
            type(row) is dict
            and all(
                type(row.get(key)) in (int, float) and row[key] > 0
                for key in METRIC_KEYS
            ),
            f"baseline certificate {name}: {path}",
        )
    return value


def comparison(
    reference: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, float]:
    return {
        key + "_change_pct": change(reference[key], observed[key])
        for key in METRIC_KEYS
    }


def analyze(
    run_root: Path,
    certificate_path: Path,
) -> dict[str, Any]:
    certificate = validate_certificate(certificate_path)
    run = validate_result(run_root, "op15", 1)
    capture = Path(str(run_root) + ".phone-capture")
    plan = read_object(capture / "EXECUTION_PLAN.json")
    result = read_object(run_root / "RESULT.json")
    require(
        plan.get("schema") == "s42-full-fp16-burstgpt-plan-v2",
        f"runtime-selected plan: {capture}",
    )
    decision = plan["runtime_placement"]
    selected = decision["selected"]
    baseline = certificate["control_mean"]
    prior_treatment = certificate["treatment_mean"]
    baseline_changes = comparison(baseline, run)
    prior_changes = comparison(prior_treatment, run)
    runtime_resources = result.get("resources")
    require(type(runtime_resources) is dict, f"runtime resources: {run_root}")
    output = {
        "baseline": {
            "certificate_file_sha256": sha256(certificate_path),
            "certificate_record_sha256": certificate["record_sha256"],
            "gpu_cpu_mean": baseline,
            "prior_op15_mean": prior_treatment,
        },
        "boundary": certificate["boundary"],
        "comparison_vs_gpu_cpu_mean": baseline_changes,
        "comparison_vs_prior_op15_mean": prior_changes,
        "outcome_gates": {
            "fleet_energy_below_gpu_cpu_mean": (
                baseline_changes["fleet_j_change_pct"] < 0
            ),
            "makespan_below_gpu_cpu_mean": (
                baseline_changes["duration_s_change_pct"] < 0
            ),
        },
        "qualification": (
            "single exact-work retest; the existing ABBA certificate remains "
            "the placement qualification evidence"
        ),
        "run": compact(run),
        "runtime_decision": {
            "baseline_candidate_id": decision["baseline"]["candidate_id"],
            "conservative_energy_saving_pct": (
                decision["conservative_energy_saving_ppm"] / 10_000.0
            ),
            "conservative_latency_change_pct": (
                decision["conservative_latency_change_ppm"] / 10_000.0
            ),
            "decision_sha256": decision["decision_sha256"],
            "mean_energy_saving_pct": (
                decision["mean_energy_saving_ppm"] / 10_000.0
            ),
            "mean_latency_change_pct": (
                decision["mean_latency_change_ppm"] / 10_000.0
            ),
            "placement": plan["placement"],
            "rejected": decision["rejected"],
            "selected_candidate_id": selected["candidate_id"],
            "snapshot": decision["snapshot"],
        },
        "runtime_observation": {
            "gpu": runtime_resources,
            "phase_metrics": result["metrics"]["by_role"],
        },
        "schema": OUTPUT_SCHEMA,
        "status": "PASS",
        "trace": certificate["trace"],
        "validity_gates": {
            "capacity_snapshot_bound": True,
            "completed_exact_trace": True,
            "qualified_phone_shapes_only": True,
            "runtime_decision_hash_bound": True,
            "runtime_values_bound_to_execution": True,
            "synchronized_whole_phone_energy": True,
            "three_resident_phone_sessions": True,
            "zero_process_swap": True,
            "zero_reset_recoveries": True,
        },
    }
    require(
        all(output["outcome_gates"].values()),
        "runtime placement did not beat the GPU plus CPU baseline",
    )
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an absolute new path")
    try:
        value = analyze(args.run_root, args.baseline_certificate)
        args.output.write_bytes(canonical(value))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"runtime placement analysis failed: {exc}\n")
    print(json.dumps({
        "fleet_energy_change_pct": value[
            "comparison_vs_gpu_cpu_mean"
        ]["fleet_j_change_pct"],
        "makespan_change_pct": value[
            "comparison_vs_gpu_cpu_mean"
        ]["duration_s_change_pct"],
        "record_sha256": value["record_sha256"],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
