#!/usr/bin/env python3
"""Compile the measured Qwen screen through the unified scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    MetricEstimate,
    PhoneArmGroup,
    PhoneOffloadCandidate,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    PhoneSessionReceipt,
    ProfileBundle,
    UnifiedScheduler,
)


SCHEMA = "s42-unified-qwen-full-ffn-screen-decision-v1"
WORKER_SHA256 = (
    "sha256:4b1db032034bd9e2ff863df6f5350a604e3cfed5e32855f2d8e29a681891007c"
)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"),
                   sort_keys=True)
        + "\n"
    ).encode("ascii")


def metric(values: list[float], scale: int) -> MetricEstimate:
    return MetricEstimate(
        mean=round(sum(values) * scale / len(values)),
        upper=math.ceil(max(values) * scale),
        lower=math.floor(min(values) * scale),
        sample_count=len(values),
        measured=True,
    )


def profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "qwen-full-ffn-screen-v1",
        "resources": [
            {
                "capacity": 1,
                "identity": resource_id,
                "kind": kind,
                "ready": True,
                "resource_id": resource_id,
            }
            for resource_id, kind in (
                ("desktop-cpu", "cpu"),
                ("op15-htp", "phone_accelerator"),
                ("op15-functionfs", "phone_transport"),
                ("desktop-usb-root", "usb_root"),
            )
        ],
        "routes": [{
            "route_id": "qwen-gpu-cpu-control",
            "workload_id": "qwen-screen",
            "granularity": "task",
            "baseline": True,
            "resource_slots": {"desktop-cpu": 1},
            "latency": {
                "cost_us": {
                    "kind": "affine_tokens_v1",
                    "fixed": 32_291_419,
                    "input_token": 0,
                    "output_token": 0,
                },
                "ucb_add_us": 12_380,
                "sample_count": 2,
                "measured": True,
            },
            "energy": {
                "status": "measured",
                "boundary_id": "cpu-package+gpu-board+whole-phone",
                "cost_uj": {
                    "kind": "affine_tokens_v1",
                    "fixed": 4_037_354_264,
                    "input_token": 0,
                    "output_token": 0,
                },
                "lower_error_ppm": 4_066,
                "upper_error_ppm": 4_066,
            },
            "overlap": {"status": "not_applicable"},
            "quality_class": "exact",
            "placement_verified": True,
            "resident": True,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 1,
            "evidence_ids": ["qwen-full-ffn-energy-screen-r1-r2"],
        }],
        "trace_workload_map": {"qwen-screen": "qwen-screen"},
        "policy": {
            "energy_saving_ppm": 100_000,
            "latency_limit_ppm": 1_000_000,
        },
    })


def snapshot(plan: PhoneResidencyPlan) -> PhoneResidencySnapshot:
    receipts = {
        session.session_id: PhoneSessionReceipt(
            session_id=session.session_id,
            compute_backend=session.compute_backend,
            state="WARM",
            generation=index,
            reset_generation=plan.reset_generation,
            worker_hash=WORKER_SHA256,
            allocated_bytes=session.resident_bytes,
            slice_weight_hashes={
                row.slice_id: row.weight_hash for row in session.slices
            },
            last_transition_us=900_000,
        )
        for index, session in enumerate(plan.sessions, start=1)
    }
    return PhoneResidencySnapshot(
        snapshot_id="op15-warm-three-session-screen-v1",
        plan_id=plan.plan_id,
        captured_at_us=900_000,
        mem_available_bytes=2_315_308 * 1024,
        sessions=receipts,
    )


def compile_decision(result: dict[str, Any],
                     plan: PhoneResidencyPlan) -> dict[str, Any]:
    runs = result["runs"]
    control = [runs["control_r1"], runs["control_r2"]]
    treatment = [runs["treatment_r1"], runs["treatment_r2"]]
    candidate = PhoneOffloadCandidate(
        candidate_id="qwen-full-ffn-layers-0-11-m1",
        slice_id="qwen-ffn-layers-0-5-full",
        additional_slice_ids=("qwen-ffn-layers-6-11-full",),
        offload_units=12,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="matched-three-request-burstgpt-screen",
        baseline_latency_us=metric(
            [row["duration_s"] for row in control], 1_000_000),
        host_remainder_us=MetricEstimate(
            mean=1, upper=1, lower=1, sample_count=2, measured=True),
        phone_path_us=metric(
            [row["duration_s"] for row in treatment], 1_000_000),
        baseline_energy_uj=metric(
            [row["fleet_j"] for row in control], 1_000_000),
        split_energy_uj=metric(
            [row["fleet_j"] for row in treatment], 1_000_000),
        evidence_ids=("qwen-full-ffn-energy-screen-r1-r2",),
        execution_mode="full_replacement",
    )
    scheduler = UnifiedScheduler(
        (profile(),),
        "enforce",
        phone_residency_plan=plan,
        phone_residency_snapshot=snapshot(plan),
    )
    schedule = scheduler.schedule_phone_offload(
        (candidate,),
        request_id="qwen-screen-cohort",
        route_id="qwen-gpu-cpu-op15-full-ffn",
        physical_m=1,
        now_us=1_000_000,
        deadline_us=34_000_000,
        minimum_energy_saving_ppm=100_000,
    )
    decision = schedule.decision
    if not isinstance(decision.arm_signal, PhoneArmGroup):
        raise RuntimeError("unified scheduler did not produce a composite arm")
    if decision.candidate_id != candidate.candidate_id:
        raise RuntimeError("unified scheduler rejected the measured candidate")
    output: dict[str, Any] = {
        "arm": decision.arm_signal.to_json(),
        "candidate_id": decision.candidate_id,
        "decision_reason": decision.reason,
        "energy_saving_ppm_conservative": decision.energy_saving_ppm,
        "execution_mode": candidate.execution_mode,
        "leases": [
            {
                "reserved_until_us": lease.reserved_until_us,
                "resource_id": lease.resource_id,
                "start_us": lease.start_us,
            }
            for lease in schedule.leases
        ],
        "offload_units": decision.offload_units,
        "residency_plan_id": plan.plan_id,
        "result_record_sha256": result["record_sha256"],
        "schema": SCHEMA,
        "scope": result["scope"],
        "split_latency_upper_us": decision.split_latency_upper_us,
        "status": "PASS",
    }
    output["decision_sha256"] = hashlib.sha256(canonical_bytes(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--residency-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    result = json.loads(args.result.read_text(encoding="ascii"))
    plan = PhoneResidencyPlan.from_json(json.loads(
        args.residency_plan.read_text(encoding="ascii")))
    output = compile_decision(result, plan)
    args.output.write_bytes(canonical_bytes(output))
    print(json.dumps({
        "candidate_id": output["candidate_id"],
        "decision_sha256": output["decision_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
