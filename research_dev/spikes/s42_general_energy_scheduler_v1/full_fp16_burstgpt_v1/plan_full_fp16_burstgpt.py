#!/usr/bin/env python3
"""Compile the two-phase F16 BurstGPT experiment through the scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
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
    RuntimePlacementCandidate,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
)


SCHEMA = "s42-full-fp16-burstgpt-plan-v2"
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
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
QUALIFIED_GPU_INCREMENTAL_BYTES = 15_168_700_416
QUALIFIED_HOST_INCREMENTAL_BYTES = 19_734_474_752
FULL_GPU_INCREMENTAL_BYTES = 30_809_054_592
STAGED_GPU_INCREMENTAL_BYTES = 15_228_469_248
STAGED_HOST_INCREMENTAL_BYTES = 31_086_264_320


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def binding_int(bindings: Any, key: str, minimum: int = 0) -> int:
    value = bindings.get(key)
    require(type(value) is int and value >= minimum, f"runtime binding {key}")
    return value


def binding_text(bindings: Any, key: str) -> str:
    value = bindings.get(key)
    require(type(value) is str and bool(value), f"runtime binding {key}")
    return value


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def metric(values: list[float], scale: int) -> MetricEstimate:
    require(values and all(math.isfinite(row) and row > 0 for row in values),
            "positive metric samples")
    return MetricEstimate(
        mean=round(sum(values) * scale / len(values)),
        upper=math.ceil(max(values) * scale),
        lower=math.floor(min(values) * scale),
        sample_count=len(values),
        measured=True,
    )


def unmeasured_metric() -> MetricEstimate:
    return MetricEstimate(
        mean=1,
        upper=1,
        lower=1,
        sample_count=0,
        measured=False,
    )


def fixed_bindings(residency: PhoneResidencyPlan) -> dict[str, int | str]:
    return {
        "gemma_execution_mode": "parallel_split",
        "gemma_gpu_layers": 25,
        "gemma_n_embd": 3840,
        "gemma_phone_columns": 6144,
        "gemma_phone_layer_mask": "0x7fffff",
        "gemma_phone_layers": 23,
        "gemma_phone_policy": "1:6144,16:6144,512:0",
        "gemma_physical_m_max": 16,
        "phone_residency_plan_id": residency.plan_id,
        "placement_mode": "sequential-partial-model-switch",
        "qwen_execution_mode": "full_replacement",
        "qwen_gpu_layers": 18,
        "qwen_n_embd": 5120,
        "qwen_phone_columns": 17408,
        "qwen_phone_layer_mask": "0x0fff",
        "qwen_phone_layers": 12,
        "qwen_phone_policy": "4:17408,512:0",
        "qwen_physical_m_max": 4,
    }


def runtime_candidates(
    result: dict[str, Any],
    residency: PhoneResidencyPlan,
    additional_phone_resident_bytes: int = 0,
    additional_host_resident_bytes: int = 0,
) -> tuple[RuntimePlacementCandidate, ...]:
    require(
        result.get("schema") == "s42-full-fp16-burstgpt-abba-v1"
        and result.get("status") == "PASS"
        and all(result.get("validity_gates", {}).values())
        and all(result.get("outcome_gates", {}).values())
        and result.get("trace", {}).get("source_sha256") == TRACE_SHA256,
        "full-trace placement qualification",
    )
    runs = result.get("runs")
    require(type(runs) is dict, "full-trace placement runs")
    control = [runs["control_r1"], runs["control_r2"]]
    treatment = [runs["treatment_r1"], runs["treatment_r2"]]
    work_set = "sha256:" + TRACE_SHA256
    boundary = "cpu-package+gpu-board+whole-phone"
    phone_memory_resource = f"phone:{residency.phone_serial}:dram"
    phone_resident_bytes = sum(
        session.resident_bytes for session in residency.sessions
    ) + additional_phone_resident_bytes
    fixed = fixed_bindings(residency)
    if additional_phone_resident_bytes:
        fixed = {
            **fixed,
            "additional_phone_resident_bytes": additional_phone_resident_bytes,
        }
    if additional_host_resident_bytes:
        fixed = {
            **fixed,
            "additional_host_resident_bytes": additional_host_resident_bytes,
        }
    baseline = RuntimePlacementCandidate(
        candidate_id="fp16-server-gpu-cpu-switch-v1",
        workload_id="full-fp16-burstgpt-74",
        work_set_sha256=work_set,
        energy_boundary_id=boundary,
        additional_bytes={
            GPU_MEMORY_RESOURCE: QUALIFIED_GPU_INCREMENTAL_BYTES,
            HOST_MEMORY_RESOURCE: (
                QUALIFIED_HOST_INCREMENTAL_BYTES
                + additional_host_resident_bytes
            ),
        },
        runtime_bindings={**fixed, "phone_execution": "disabled"},
        latency_us=metric(
            [row["duration_s"] for row in control], 1_000_000
        ),
        fleet_energy_uj=metric(
            [row["fleet_j"] for row in control], 1_000_000
        ),
        status="measured",
        placement_verified=True,
        workload_verified=True,
        evidence_ids=(
            "full-fp16-burstgpt-control-r1",
            "full-fp16-burstgpt-control-r2",
            "rtx4060ti-qwen18-gemma25-capacity-v1",
        ),
    )
    treatment_candidate = RuntimePlacementCandidate(
        candidate_id="fp16-server-gpu-cpu-op15-switch-v1",
        workload_id=baseline.workload_id,
        work_set_sha256=work_set,
        energy_boundary_id=boundary,
        additional_bytes={
            GPU_MEMORY_RESOURCE: QUALIFIED_GPU_INCREMENTAL_BYTES,
            HOST_MEMORY_RESOURCE: (
                QUALIFIED_HOST_INCREMENTAL_BYTES
                + additional_host_resident_bytes
            ),
            phone_memory_resource: phone_resident_bytes,
        },
        runtime_bindings={**fixed, "phone_execution": "qualified"},
        latency_us=metric(
            [row["duration_s"] for row in treatment], 1_000_000
        ),
        fleet_energy_uj=metric(
            [row["fleet_j"] for row in treatment], 1_000_000
        ),
        status="measured",
        placement_verified=True,
        workload_verified=True,
        evidence_ids=(
            "full-fp16-burstgpt-op15-r1",
            "full-fp16-burstgpt-op15-r2",
            "op15-three-session-residency-v1",
        ),
    )
    full_gpu = RuntimePlacementCandidate(
        candidate_id="fp16-sequential-full-gpu-v1",
        workload_id=baseline.workload_id,
        work_set_sha256=work_set,
        energy_boundary_id=boundary,
        additional_bytes={
            GPU_MEMORY_RESOURCE: FULL_GPU_INCREMENTAL_BYTES,
            HOST_MEMORY_RESOURCE: (
                4_294_967_296 + additional_host_resident_bytes
            ),
        },
        runtime_bindings={
            "gemma_gpu_layers": 49,
            "placement_mode": "sequential-full-model-switch",
            "qwen_gpu_layers": 41,
        },
        latency_us=unmeasured_metric(),
        fleet_energy_uj=unmeasured_metric(),
        status="estimated",
        placement_verified=False,
        workload_verified=False,
        evidence_ids=("fp16-model-file-sizes-v1",),
    )
    staged = RuntimePlacementCandidate(
        candidate_id="fp16-qwen15-gemma1-wavefront-v1",
        workload_id=baseline.workload_id,
        work_set_sha256=work_set,
        energy_boundary_id=boundary,
        additional_bytes={
            GPU_MEMORY_RESOURCE: STAGED_GPU_INCREMENTAL_BYTES,
            HOST_MEMORY_RESOURCE: (
                STAGED_HOST_INCREMENTAL_BYTES
                + additional_host_resident_bytes
            ),
            phone_memory_resource: phone_resident_bytes,
        },
        runtime_bindings={
            "gemma_gpu_layers": 1,
            "placement_mode": "concurrent-prefix-wavefront",
            "qwen_gpu_layers": 15,
        },
        latency_us=unmeasured_metric(),
        fleet_energy_uj=unmeasured_metric(),
        status="estimated",
        placement_verified=True,
        workload_verified=False,
        evidence_ids=(
            "rtx4060ti-qwen15-gemma1-capacity-v1",
            "m8-staged-cohort-r42-r45",
        ),
    )
    return baseline, treatment_candidate, full_gpu, staged


def scheduler_profile() -> ProfileBundle:
    return ProfileBundle.from_json({
        "schema": "s42-general-scheduler-profile-v1",
        "profile_id": "full-fp16-burstgpt-two-phase-v1",
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
            "route_id": "full-fp16-control",
            "workload_id": "full-fp16-burstgpt",
            "granularity": "task",
            "baseline": True,
            "resource_slots": {"desktop-cpu": 1},
            "latency": {
                "cost_us": {
                    "kind": "affine_tokens_v1",
                    "fixed": 1,
                    "input_token": 0,
                    "output_token": 0,
                },
                "ucb_add_us": 0,
                "sample_count": 1,
                "measured": True,
            },
            "energy": {
                "status": "measured",
                "boundary_id": "cpu-package+gpu-board+whole-phone",
                "cost_uj": {
                    "kind": "affine_tokens_v1",
                    "fixed": 1,
                    "input_token": 0,
                    "output_token": 0,
                },
                "lower_error_ppm": 0,
                "upper_error_ppm": 0,
            },
            "overlap": {"status": "not_applicable"},
            "quality_class": "approximate",
            "placement_verified": True,
            "resident": True,
            "server_busy_ppm": 1_000_000,
            "server_memory_bytes": 1,
            "evidence_ids": ["full-fp16-burstgpt-phase-planner-v1"],
        }],
        "trace_workload_map": {
            "full-fp16-burstgpt": "full-fp16-burstgpt"
        },
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
        snapshot_id="op15-warm-three-session-full-trace-v1",
        plan_id=plan.plan_id,
        captured_at_us=900_000,
        mem_available_bytes=2_199_748 * 1024,
        sessions=receipts,
    )


def decision_json(schedule: Any) -> dict[str, Any]:
    decision = schedule.decision
    arm = decision.arm_signal
    require(arm is not None, "scheduler did not arm phone route")
    return {
        "arm": arm.to_json(),
        "candidate_id": decision.candidate_id,
        "decision_reason": decision.reason,
        "energy_saving_ppm_conservative": decision.energy_saving_ppm,
        "exposed_join_wait_upper_us": decision.exposed_join_wait_upper_us,
        "leases": [{
            "reserved_until_us": lease.reserved_until_us,
            "resource_id": lease.resource_id,
            "start_us": lease.start_us,
        } for lease in schedule.leases],
        "offload_units": decision.offload_units,
        "split_latency_upper_us": decision.split_latency_upper_us,
    }


def compile_plan(
    qwen: dict[str, Any],
    gemma: dict[str, Any],
    placement_result: dict[str, Any],
    residency: PhoneResidencyPlan,
    runtime_snapshot: RuntimePlacementSnapshot,
    *,
    qwen_result_hash: str,
    gemma_result_hash: str,
    placement_result_hash: str,
    residency_hash: str,
    runtime_snapshot_hash: str,
    now_us: int | None = None,
    overlay_manifest: dict[str, Any] | None = None,
    overlay_manifest_hash: str | None = None,
) -> dict[str, Any]:
    require(
        qwen.get("schema") == "s42-qwen-full-ffn-m1-m4-energy-screen-v1"
        and qwen.get("status") == "PASS"
        and all(qwen.get("gates", {}).values()),
        "Qwen M=1..4 evidence",
    )
    require(
        gemma.get("schema") == "s42-gpu-overflow-physical-pair-v1"
        and gemma.get("status") == "PASS"
        and all(gemma.get("gates", {}).values()),
        "Gemma overflow evidence",
    )
    qwen_runs = qwen["runs"]
    qwen_control = [qwen_runs["control_r1"], qwen_runs["control_r2"]]
    qwen_treatment = [
        qwen_runs["treatment_r1"], qwen_runs["treatment_r2"]
    ]
    gemma_control = gemma["comparison"]["control"]
    gemma_treatment = gemma["comparison"]["treatment"]
    scheduler = UnifiedScheduler(
        (scheduler_profile(),),
        "enforce",
        phone_residency_plan=residency,
        phone_residency_snapshot=snapshot(residency),
    )
    additional_phone_resident_bytes = 0
    overlay_record = None
    if overlay_manifest is not None:
        overlay_trace = overlay_manifest.get("overlay_trace", {})
        overlay_count = overlay_trace.get("record_count")
        overlay_input_tokens = overlay_trace.get("input_tokens")
        overlay_output_tokens = overlay_trace.get("output_tokens")
        require(
            overlay_manifest.get("schema")
                in {
                    "s42-full-fp16-llama1b-natural-validation-manifest-v1",
                    "s42-full-fp16-llama1b-overlay-manifest-v1",
                    "s42-full-fp16-llama1b-phase-overlay-manifest-v1",
                }
            and overlay_manifest.get("base_trace", {}).get("sha256")
                == TRACE_SHA256
            and type(overlay_count) is int
            and overlay_count > 0
            and type(overlay_input_tokens) is int
            and overlay_input_tokens > 0
            and type(overlay_output_tokens) is int
            and overlay_output_tokens > 0
            and overlay_manifest.get("combined_work") == {
                "input_tokens": 33_843 + overlay_input_tokens,
                "output_tokens": 11_605 + overlay_output_tokens,
                "record_count": 74 + overlay_count,
            }
            and type(overlay_manifest_hash) is str
            and len(overlay_manifest_hash) == 64,
            "F16 small-model overlay identity",
        )
        llama = overlay_manifest.get("model_inventory", {}).get(
            "llama-3.2-1b-instruct-q4_0"
        )
        require(
            type(llama) is dict
            and llama.get("artifact_bytes") == 770_928_288
            and llama.get("artifact_sha256")
                == (
                    "4b90b1d7ae7324676194755a6dfce11c"
                    "b6e457982c4c01a1db2857be1ed064ad"
                ),
            "F16 small-model artifact identity",
        )
        additional_phone_resident_bytes = llama["artifact_bytes"]
        overlay_record = {
            "artifact_bytes": llama["artifact_bytes"],
            "artifact_sha256": llama["artifact_sha256"],
            "manifest_file_sha256": overlay_manifest_hash,
            "overlay_record_count": overlay_count,
            "overlay_trace_sha256": overlay_trace["sha256"],
        }
    runtime_placement = scheduler.select_runtime_placement(
        runtime_candidates(
            placement_result,
            residency,
            additional_phone_resident_bytes,
            additional_phone_resident_bytes,
        ),
        baseline_candidate_id="fp16-server-gpu-cpu-switch-v1",
        snapshot=runtime_snapshot,
        now_us=(time.monotonic_ns() // 1000 if now_us is None else now_us),
        minimum_energy_saving_ppm=200_000,
        maximum_latency_ppm=999_999,
        minimum_samples=2,
    )
    bindings = runtime_placement.selected.runtime_bindings
    phone_execution = binding_text(bindings, "phone_execution")
    require(
        phone_execution in {"disabled", "qualified"}
        and bindings.get("phone_residency_plan_id") == residency.plan_id,
        "runtime placement execution binding",
    )
    execution_arm = "op15" if phone_execution == "qualified" else "control"
    qwen_candidate = PhoneOffloadCandidate(
        candidate_id="qwen-full-ffn-layers-0-11-m1-m4",
        slice_id="qwen-ffn-layers-0-5-full",
        additional_slice_ids=("qwen-ffn-layers-6-11-full",),
        offload_units=binding_int(bindings, "qwen_phone_layers", 1),
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="matched-qwen-m1-m4-burstgpt-screen",
        baseline_latency_us=metric(
            [row["duration_s"] for row in qwen_control], 1_000_000
        ),
        host_remainder_us=MetricEstimate(
            mean=1, upper=1, lower=1, sample_count=2, measured=True
        ),
        phone_path_us=metric(
            [row["duration_s"] for row in qwen_treatment], 1_000_000
        ),
        baseline_energy_uj=metric(
            [row["fleet_j"] for row in qwen_control], 1_000_000
        ),
        split_energy_uj=metric(
            [row["fleet_j"] for row in qwen_treatment], 1_000_000
        ),
        evidence_ids=("qwen-full-ffn-m1-m4-energy-screen-abba-v2",),
        execution_mode="full_replacement",
    )
    qwen_schedule = scheduler.schedule_phone_offload(
        (qwen_candidate,),
        request_id="full-trace-qwen-phase",
        route_id=(
            "qwen-f16-cuda"
            f"{binding_int(bindings, 'qwen_gpu_layers', 1)}"
            "-cpu-op15-full-ffn"
        ),
        physical_m=binding_int(bindings, "qwen_physical_m_max", 1),
        now_us=1_000_000,
        deadline_us=2_000_000_000,
        minimum_energy_saving_ppm=100_000,
    )
    require(
        isinstance(qwen_schedule.decision.arm_signal, PhoneArmGroup),
        "Qwen decision is not composite",
    )

    gemma_duration = metric([gemma_treatment["duration_s"]], 1_000_000)
    gemma_candidate = PhoneOffloadCandidate(
        candidate_id="gemma-suffix-layers-0-22-m1-m16",
        slice_id="gemma-ffn-layers-0-22-suffix-6144",
        offload_units=binding_int(bindings, "gemma_phone_layers", 1),
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="matched-gemma-cold17-burstgpt-overflow",
        baseline_latency_us=metric(
            [gemma_control["duration_s"]], 1_000_000
        ),
        host_remainder_us=gemma_duration,
        phone_path_us=gemma_duration,
        baseline_energy_uj=metric(
            [gemma_control["fleet_j"]], 1_000_000
        ),
        split_energy_uj=metric(
            [gemma_treatment["fleet_j"]], 1_000_000
        ),
        evidence_ids=("gemma-gpu-overflow-physical-pair-r1",),
        execution_mode="parallel_split",
    )
    gemma_schedule = scheduler.schedule_phone_offload(
        (gemma_candidate,),
        request_id="full-trace-gemma-phase",
        route_id=(
            "gemma-f16-cuda"
            f"{binding_int(bindings, 'gemma_gpu_layers', 1)}"
            "-cpu-op15-suffix"
        ),
        physical_m=binding_int(bindings, "gemma_physical_m_max", 1),
        now_us=2_100_000_000,
        deadline_us=3_600_000_000,
        minimum_energy_saving_ppm=100_000,
    )
    output: dict[str, Any] = {
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
        "evidence": {
            "gemma_pair_file_sha256": gemma_result_hash,
            "placement_result_file_sha256": placement_result_hash,
            "qwen_m1_m4_file_sha256": qwen_result_hash,
            "residency_plan_file_sha256": residency_hash,
            "runtime_snapshot_file_sha256": runtime_snapshot_hash,
        },
        "phases": {
            "gemma": {
                "control_route": (
                    "gemma-f16-cuda"
                    f"{binding_int(bindings, 'gemma_gpu_layers', 1)}-cpu"
                ),
                "decision": decision_json(gemma_schedule),
                "execution_mode": binding_text(
                    bindings, "gemma_execution_mode"
                ),
                "layer_mask": binding_text(
                    bindings, "gemma_phone_layer_mask"
                ),
                "n_embd": binding_int(bindings, "gemma_n_embd", 1),
                "phone_columns": binding_int(
                    bindings, "gemma_phone_columns", 1
                ),
                "phone_policy": binding_text(
                    bindings, "gemma_phone_policy"
                ),
                "physical_m_max": binding_int(
                    bindings, "gemma_physical_m_max", 1
                ),
            },
            "qwen": {
                "control_route": (
                    "qwen-f16-cuda"
                    f"{binding_int(bindings, 'qwen_gpu_layers', 1)}-cpu"
                ),
                "decision": decision_json(qwen_schedule),
                "execution_mode": binding_text(
                    bindings, "qwen_execution_mode"
                ),
                "layer_mask": binding_text(
                    bindings, "qwen_phone_layer_mask"
                ),
                "n_embd": binding_int(bindings, "qwen_n_embd", 1),
                "phone_columns": binding_int(
                    bindings, "qwen_phone_columns", 1
                ),
                "phone_policy": binding_text(
                    bindings, "qwen_phone_policy"
                ),
                "physical_m_max": binding_int(
                    bindings, "qwen_physical_m_max", 1
                ),
            },
        },
        "placement": {
            "gemma_gpu_layers": binding_int(
                bindings, "gemma_gpu_layers", 1
            ),
            "qwen_gpu_layers": binding_int(
                bindings, "qwen_gpu_layers", 1
            ),
        },
        "execution_arm": execution_arm,
        "execution_arm_source": "runtime_placement",
        "residency_plan_id": residency.plan_id,
        "runtime_placement": runtime_placement.to_json(),
        "schema": SCHEMA,
        "status": "PASS",
        "trace": {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "requests": 74,
            "sha256": TRACE_SHA256,
        },
    }
    if overlay_record is not None:
        output["small_model_overlay"] = overlay_record
    output["plan_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--gemma-result", type=Path, required=True)
    parser.add_argument("--placement-result", type=Path, required=True)
    parser.add_argument("--residency-plan", type=Path, required=True)
    parser.add_argument("--runtime-snapshot", type=Path, required=True)
    parser.add_argument("--overlay-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        qwen = read_object(args.qwen_result)
        gemma = read_object(args.gemma_result)
        placement_result = read_object(args.placement_result)
        residency = PhoneResidencyPlan.from_json(
            read_object(args.residency_plan)
        )
        runtime_snapshot = RuntimePlacementSnapshot.from_json(
            read_object(args.runtime_snapshot)
        )
        overlay_manifest = (
            None
            if args.overlay_manifest is None
            else read_object(args.overlay_manifest)
        )
        output = compile_plan(
            qwen,
            gemma,
            placement_result,
            residency,
            runtime_snapshot,
            qwen_result_hash=sha256(args.qwen_result),
            gemma_result_hash=sha256(args.gemma_result),
            placement_result_hash=sha256(args.placement_result),
            residency_hash=sha256(args.residency_plan),
            runtime_snapshot_hash=sha256(args.runtime_snapshot),
            overlay_manifest=overlay_manifest,
            overlay_manifest_hash=(
                None
                if args.overlay_manifest is None
                else sha256(args.overlay_manifest)
            ),
        )
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"full F16 plan failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "plan_sha256": output["plan_sha256"],
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
