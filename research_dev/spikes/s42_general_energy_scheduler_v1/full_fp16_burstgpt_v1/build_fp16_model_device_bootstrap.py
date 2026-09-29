#!/usr/bin/env python3
"""Build a two-model residency bootstrap without reading a request trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    MetricEstimate,
    PhoneArmGroup,
    PhoneOffloadCandidate,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    PhoneSessionReceipt,
    ProfileBundle,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
)


SCHEMA = "s42-fp16-model-device-bootstrap-v1"
QWEN_SHA256 = (
    "d89e9e823744222e595e0b3c8fd5436c"
    "e5d3a6a446fa42492ebce6064dfa9718"
)
GEMMA_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
WORKER_SHA256 = (
    "sha256:4b1db032034bd9e2ff863df6f5350a604"
    "e3cfed5e32855f2d8e29a681891007c"
)
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
GPU_MEMORY_RESOURCE = f"cuda:{GPU_UUID}:vram"
HOST_MEMORY_RESOURCE = "desktop-host-ram"
GPU_REQUIRED_BYTES = 15_168_700_416
HOST_REQUIRED_BYTES = 19_734_474_752


class BootstrapError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapError(message)


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
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def metric(values: list[float], scale: int) -> MetricEstimate:
    require(
        values and all(math.isfinite(value) and value > 0 for value in values),
        "metric samples must be positive",
    )
    return MetricEstimate(
        mean=round(sum(values) * scale / len(values)),
        upper=math.ceil(max(values) * scale),
        lower=math.floor(min(values) * scale),
        sample_count=len(values),
        measured=True,
    )


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "fp16-model-device-bootstrap-v1",
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
            "baseline": True,
            "energy": {
                "boundary_id": "cpu-package+gpu-board+whole-phone",
                "cost_uj": {
                    "fixed": 1,
                    "input_token": 0,
                    "kind": "affine_tokens_v1",
                    "output_token": 0,
                },
                "lower_error_ppm": 0,
                "status": "measured",
                "upper_error_ppm": 0,
            },
            "evidence_ids": ["model-device-bootstrap-control-v1"],
            "granularity": "task",
            "latency": {
                "cost_us": {
                    "fixed": 1,
                    "input_token": 0,
                    "kind": "affine_tokens_v1",
                    "output_token": 0,
                },
                "measured": True,
                "sample_count": 1,
                "ucb_add_us": 0,
            },
            "overlap": {"status": "not_applicable"},
            "placement_verified": True,
            "quality_class": "approximate",
            "resident": True,
            "resource_slots": {"desktop-cpu": 1},
            "route_id": "model-device-bootstrap-control",
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 1,
            "workload_id": "model-device-bootstrap",
        }],
        "trace_workload_map": {
            "model-device-bootstrap": "model-device-bootstrap"
        },
        "policy": {
            "energy_saving_ppm": 100_000,
            "latency_limit_ppm": 1_000_000,
        },
    })


def residency_snapshot(
    plan: PhoneResidencyPlan,
    available_bytes: int,
) -> PhoneResidencySnapshot:
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
                item.slice_id: item.weight_hash for item in session.slices
            },
            last_transition_us=0,
        )
        for index, session in enumerate(plan.sessions, start=1)
    }
    return PhoneResidencySnapshot(
        snapshot_id="model-device-bootstrap-residency",
        plan_id=plan.plan_id,
        captured_at_us=0,
        mem_available_bytes=available_bytes,
        sessions=receipts,
    )


def decision_json(schedule: Any) -> dict[str, Any]:
    arm = schedule.decision.arm_signal
    require(arm is not None, "phone route did not produce an arm signal")
    return {
        "arm": arm.to_json(),
        "candidate_id": schedule.decision.candidate_id,
        "decision_reason": schedule.decision.reason,
        "energy_saving_ppm_conservative": (
            schedule.decision.energy_saving_ppm
        ),
        "exposed_join_wait_upper_us": (
            schedule.decision.exposed_join_wait_upper_us
        ),
        "offload_units": schedule.decision.offload_units,
        "split_latency_upper_us": schedule.decision.split_latency_upper_us,
    }


def schedule_qwen(
    source: dict[str, Any],
    residency: PhoneResidencyPlan,
    available_bytes: int,
) -> dict[str, Any]:
    require(
        source.get("schema") == "s42-qwen-full-ffn-m1-m4-energy-screen-v1"
        and source.get("status") == "PASS"
        and all(source.get("gates", {}).values()),
        "Qwen operator evidence",
    )
    runs = source["runs"]
    control = [runs["control_r1"], runs["control_r2"]]
    treatment = [runs["treatment_r1"], runs["treatment_r2"]]
    scheduler = UnifiedScheduler(
        (scheduler_profile(),),
        "enforce",
        phone_residency_plan=residency,
        phone_residency_snapshot=residency_snapshot(
            residency, available_bytes
        ),
    )
    candidate = PhoneOffloadCandidate(
        candidate_id="qwen-full-ffn-layers-0-11-m1-m4",
        slice_id="qwen-ffn-layers-0-5-full",
        additional_slice_ids=("qwen-ffn-layers-6-11-full",),
        offload_units=12,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="matched-qwen-m1-m4-screen",
        baseline_latency_us=metric(
            [item["duration_s"] for item in control], 1_000_000
        ),
        host_remainder_us=MetricEstimate(
            mean=1,
            upper=1,
            lower=1,
            sample_count=2,
            measured=True,
        ),
        phone_path_us=metric(
            [item["duration_s"] for item in treatment], 1_000_000
        ),
        baseline_energy_uj=metric(
            [item["fleet_j"] for item in control], 1_000_000
        ),
        split_energy_uj=metric(
            [item["fleet_j"] for item in treatment], 1_000_000
        ),
        evidence_ids=("qwen-full-ffn-m1-m4-energy-screen-abba-v2",),
        execution_mode="full_replacement",
    )
    schedule = scheduler.schedule_phone_offload(
        (candidate,),
        request_id="bootstrap-qwen-model",
        route_id="qwen-f16-cuda18-cpu-op15-full-ffn",
        physical_m=4,
        now_us=1_000_000,
        deadline_us=2_000_000_000,
        minimum_energy_saving_ppm=100_000,
    )
    require(
        isinstance(schedule.decision.arm_signal, PhoneArmGroup),
        "Qwen phone arm must cover both resident sessions",
    )
    return decision_json(schedule)


def schedule_gemma(
    source: dict[str, Any],
    residency: PhoneResidencyPlan,
    available_bytes: int,
) -> dict[str, Any]:
    require(
        source.get("schema") == "s42-gpu-overflow-physical-pair-v1"
        and source.get("status") == "PASS"
        and all(source.get("gates", {}).values()),
        "Gemma operator evidence",
    )
    control = source["comparison"]["control"]
    treatment = source["comparison"]["treatment"]
    duration = metric([treatment["duration_s"]], 1_000_000)
    scheduler = UnifiedScheduler(
        (scheduler_profile(),),
        "enforce",
        phone_residency_plan=residency,
        phone_residency_snapshot=residency_snapshot(
            residency, available_bytes
        ),
    )
    candidate = PhoneOffloadCandidate(
        candidate_id="gemma-suffix-layers-0-22-m1-m16",
        slice_id="gemma-ffn-layers-0-22-suffix-6144",
        offload_units=23,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="matched-gemma-cold17-overflow",
        baseline_latency_us=metric(
            [control["duration_s"]], 1_000_000
        ),
        host_remainder_us=duration,
        phone_path_us=duration,
        baseline_energy_uj=metric([control["fleet_j"]], 1_000_000),
        split_energy_uj=metric([treatment["fleet_j"]], 1_000_000),
        evidence_ids=("gemma-gpu-overflow-physical-pair-r1",),
        execution_mode="parallel_split",
    )
    schedule = scheduler.schedule_phone_offload(
        (candidate,),
        request_id="bootstrap-gemma-model",
        route_id="gemma-f16-cuda25-cpu-op15-suffix",
        physical_m=16,
        now_us=1_000_000,
        deadline_us=2_000_000_000,
        minimum_energy_saving_ppm=100_000,
    )
    return decision_json(schedule)


def capacity_row(
    snapshot: RuntimePlacementSnapshot,
    resource_id: str,
    required_bytes: int,
) -> dict[str, object]:
    capacity = snapshot.capacities.get(resource_id)
    if capacity is None:
        return {
            "admitted": False,
            "available_bytes": None,
            "reason": "RESOURCE_ABSENT",
            "required_bytes": required_bytes,
            "resource_id": resource_id,
        }
    admitted = required_bytes <= capacity.available_bytes
    return {
        "admitted": admitted,
        "available_bytes": capacity.available_bytes,
        "reason": "ADMITTED" if admitted else "CAPACITY",
        "required_bytes": required_bytes,
        "resource_id": resource_id,
    }


def compile_bootstrap(
    qwen: dict[str, Any],
    gemma: dict[str, Any],
    residency: PhoneResidencyPlan,
    runtime_snapshot: RuntimePlacementSnapshot,
    *,
    qwen_hash: str,
    gemma_hash: str,
    residency_hash: str,
    snapshot_hash: str,
    additional_host_bytes: int = 0,
    additional_phone_bytes: int = 0,
) -> dict[str, Any]:
    require(
        type(additional_host_bytes) is int and additional_host_bytes >= 0,
        "additional host bytes",
    )
    require(
        type(additional_phone_bytes) is int and additional_phone_bytes >= 0,
        "additional phone bytes",
    )
    phone_resource = f"phone:{residency.phone_serial}:dram"
    phone_required = sum(
        session.resident_bytes for session in residency.sessions
    ) + additional_phone_bytes
    capacity = {
        resource_id: capacity_row(
            runtime_snapshot, resource_id, required_bytes
        )
        for resource_id, required_bytes in (
            (GPU_MEMORY_RESOURCE, GPU_REQUIRED_BYTES),
            (
                HOST_MEMORY_RESOURCE,
                HOST_REQUIRED_BYTES + additional_host_bytes,
            ),
            (phone_resource, phone_required),
        )
    }
    require(
        capacity[GPU_MEMORY_RESOURCE]["admitted"] is True
        and capacity[HOST_MEMORY_RESOURCE]["admitted"] is True,
        "server placement capacity",
    )
    phone_capacity_admitted = capacity[phone_resource]["admitted"] is True
    qwen_decision = None
    gemma_decision = None
    phone_reason = "PHONE_CAPACITY_UNAVAILABLE"
    if phone_capacity_admitted:
        available = runtime_snapshot.capacities[
            phone_resource
        ].available_bytes
        qwen_decision = schedule_qwen(qwen, residency, available)
        gemma_decision = schedule_gemma(gemma, residency, available)
        phone_reason = "MODEL_OPERATOR_ROUTES_ENERGY_POSITIVE"
    arm = (
        "op15"
        if qwen_decision is not None and gemma_decision is not None
        else "control"
    )
    phases = {
        "gemma": {
            "control_route": "gemma-f16-cuda25-cpu",
            "decision": gemma_decision,
            "execution_mode": "parallel_split",
            "layer_mask": "0x7fffff",
            "n_embd": 3840,
            "phone_columns": 6144,
            "phone_policy": "1:6144,16:6144,512:0",
            "physical_m_max": 16,
        },
        "qwen": {
            "control_route": "qwen-f16-cuda18-cpu",
            "decision": qwen_decision,
            "execution_mode": "full_replacement",
            "layer_mask": "0x0fff",
            "n_embd": 5120,
            "phone_columns": 17408,
            "phone_policy": "4:17408,512:0",
            "physical_m_max": 4,
        },
    }
    output = {
        "artifacts": {
            "gemma": {
                "bytes": 23_832_065_056,
                "sha256": GEMMA_SHA256,
            },
            "qwen": {
                "bytes": 29_543_423_360,
                "sha256": QWEN_SHA256,
            },
        },
        "capacity": capacity,
        "decision_basis": {
            "future_request_data": "not_accepted",
            "inputs": [
                "live_device_memory_snapshot",
                "model_artifact_identity",
                "measured_model_operator_routes",
                "resident_phone_session_contract",
            ],
            "phone_reason": phone_reason,
        },
        "evidence": {
            "gemma_operator_file_sha256": gemma_hash,
            "qwen_operator_file_sha256": qwen_hash,
            "residency_plan_file_sha256": residency_hash,
            "runtime_snapshot_file_sha256": snapshot_hash,
        },
        "execution_arm": arm,
        "execution_arm_source": "model_device_runtime_bootstrap",
        "phases": phases,
        "placement": {
            "gemma_gpu_layers": 25,
            "qwen_gpu_layers": 18,
        },
        "residency_plan_id": residency.plan_id,
        "runtime_snapshot": runtime_snapshot.to_json(),
        "schema": SCHEMA,
        "status": "PASS",
    }
    output["plan_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--gemma-result", type=Path, required=True)
    parser.add_argument("--residency-plan", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--additional-host-bytes", type=int, default=0)
    parser.add_argument("--additional-phone-bytes", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        qwen = read_object(args.qwen_result)
        gemma = read_object(args.gemma_result)
        residency = PhoneResidencyPlan.from_json(
            read_object(args.residency_plan)
        )
        snapshot = RuntimePlacementSnapshot.from_json(
            read_object(args.runtime_snapshot)
        )
        output = compile_bootstrap(
            qwen,
            gemma,
            residency,
            snapshot,
            qwen_hash=digest(args.qwen_result),
            gemma_hash=digest(args.gemma_result),
            residency_hash=digest(args.residency_plan),
            snapshot_hash=digest(args.runtime_snapshot),
            additional_host_bytes=args.additional_host_bytes,
            additional_phone_bytes=args.additional_phone_bytes,
        )
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"F16 model/device bootstrap failed: {exc}\n")
    print(json.dumps({
        "execution_arm": output["execution_arm"],
        "output": str(args.output),
        "plan_sha256": output["plan_sha256"],
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
