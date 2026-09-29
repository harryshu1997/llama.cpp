#!/usr/bin/env python3
"""Replay dynamic residency gates against the measured F16 BurstGPT result."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    DeviceMemoryCapacity,
    DynamicResidencyCandidate,
    DynamicResidencySnapshot,
    DynamicWeightPlacement,
    DynamicWeightPlacementSpec,
    MetricEstimate,
    PhoneResidencyPlan,
    canonical_sha256,
    select_dynamic_residency_transition,
)


SCHEMA = "s42-dynamic-residency-shadow-v1"
TRACE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
AGGREGATE_SCHEMA = "s42-full-fp16-burstgpt-abba-v1"
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
EFFECTIVE_MODELS = {
    "gemma-4-12b-it-q8_0": "qwen3-14b-f16-proxy",
    "qwen3-14b-q4_k_m": "gemma-4-12b-f16-proxy",
}
DEFAULT_AGGREGATE = (
    HERE.parent
    / "full_fp16_burstgpt_v1/results/FULL_FP16_BURSTGPT_ABBA_V2.json"
)
DEFAULT_TRACE = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline/"
      "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1/"
      "REQUESTS_SEMANTIC_SOURCE.jsonl"
)
DEFAULT_PHONE_PLAN = (
    HERE.parent
    / "multi_session_phone_v1/results/"
      "OP15_THREE_SESSION_RESIDENCY_PLAN_V1.json"
)
DEFAULT_GPU_CAPACITY = (
    HERE
    / "results/RTX4060TI_QWEN15_GEMMA1_CAPACITY_V1/RESULT.json"
)
DEFAULT_GPU_DIAGNOSTIC = (
    HERE / "results/DUAL_RESIDENCY_SERVICE_ABBA_V1.json"
)
DEFAULT_GPU_TENSOR_BUNDLE = (
    HERE / "results/GPU_TENSOR_MANIFEST_BUNDLE_V1.json"
)
DEFAULT_ADOPTION_ENERGY_SCREEN = (
    HERE
    / "results/OP15_ADOPTION_ENERGY_SCREEN_ABBA_V1/"
      "ADOPTION_ENERGY_SCREEN_ABBA_V1.json"
)


class ShadowReplayError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ShadowReplayError(message)


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
    try:
        with path.open("rb") as source:
            while block := source.read(4 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise ShadowReplayError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ShadowReplayError(f"cannot read {path}: {exc}") from exc
    require(type(value) is dict, f"object: {path}")
    return value


def read_trace(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ShadowReplayError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ShadowReplayError(f"malformed JSONL: {path}") from exc
        require(type(value) is dict, f"trace object: {path}")
        rows.append(value)
    require(rows, f"nonempty trace: {path}")
    return rows


def validate_aggregate(value: dict[str, Any]) -> None:
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == AGGREGATE_SCHEMA
        and value.get("status") == "PASS"
        and value.get("energy_verdict") == "SAVING"
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        "F16 aggregate identity",
    )
    require(
        all(value.get("validity_gates", {}).values())
        and all(value.get("outcome_gates", {}).values()),
        "F16 aggregate gates",
    )
    require(
        value.get("trace") == {
            "input_tokens": 33_843,
            "output_tokens": 11_605,
            "requests": 74,
            "source_sha256": TRACE_SHA256,
        },
        "F16 aggregate work identity",
    )


def validate_trace(rows: list[dict[str, Any]], trace_hash: str) -> None:
    require(trace_hash == TRACE_SHA256, "source trace SHA-256")
    require(
        len(rows) == 74
        and [row.get("request_index") for row in rows] == list(range(74))
        and all(row.get("schema") == TRACE_SCHEMA for row in rows)
        and sum(row.get("input_tokens", 0) for row in rows) == 33_843
        and sum(row.get("output_tokens", 0) for row in rows) == 11_605,
        "source trace geometry",
    )
    require(
        all(row.get("model_id") in EFFECTIVE_MODELS for row in rows),
        "source trace model mapping",
    )


def validate_gpu_capacity(value: dict[str, Any]) -> None:
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == "s42-dynamic-gpu-residency-capacity-v1"
        and value.get("status") == "CAPACITY_PASS"
        and value.get("measurement_scope")
            == "CAPACITY_ONLY_NO_ENERGY_OR_SERVICE_CLAIM"
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        "GPU capacity artifact identity",
    )
    configuration = value.get("configuration", {})
    baseline = value.get("baseline_gpu", {})
    cleanup = value.get("cleanup", {})
    stages = value.get("stages", [])
    require(
        configuration == {
            "gemma_gpu_layers": 1,
            "gpu_reserve_bytes": 536_870_912,
            "qwen_gpu_layers": 15,
        }
        and baseline.get("name") == "NVIDIA GeForce RTX 4060 Ti"
        and baseline.get("uuid")
            == "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
        and cleanup.get("passed") is True
        and cleanup.get("compute_apps") == []
        and type(stages) is list
        and len(stages) == 2,
        "GPU capacity geometry",
    )
    qwen, gemma = stages
    require(
        qwen.get("allocations", {}).get("offloaded_layers") == 15
        and gemma.get("allocations", {}).get("offloaded_layers") == 1
        and qwen.get("process_memory", {}).get("swap_bytes") == 0
        and gemma.get("process_memory", {}).get("swap_bytes") == 0
        and gemma.get("gpu", {}).get("memory_free_bytes", 0)
            >= configuration["gpu_reserve_bytes"],
        "GPU capacity gates",
    )


def validate_gpu_diagnostic(value: dict[str, Any]) -> None:
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == "s42-dual-residency-service-abba-v1"
        and value.get("status")
            == "REPEATED_SERVICE_DIRECTION_PASS_NO_ADMISSION"
        and value.get("boundary")
            == "cpu-package+gpu-board-concurrent-two-request-service-v1"
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        "GPU diagnostic artifact identity",
    )
    admission = value.get("admission", {})
    require(
        admission.get("eligible") is False
        and admission.get("dynamic_energy_claim") is None
        and admission.get("decision")
            == "COLLECT_TRANSITION_PHONE_AND_FENCE_RECEIPTS"
        and all(value.get("validity_gates", {}).values())
        and value.get("claim_gates", {}).get("service_abba_repeated") is True
        and not any(
            passed for gate, passed in value.get("claim_gates", {}).items()
            if gate != "service_abba_repeated"
        ),
        "GPU diagnostic claim gates",
    )
    require(
        value.get("workload_per_run") == {
            "input_tokens": 287,
            "models": 2,
            "output_tokens": 50,
            "requests": 2,
            "trace_source_sha256": TRACE_SHA256,
        }
        and value.get("runs", {}).get("control_r1", {}).get("configuration")
            == {
                "gemma_gpu_layers": 0,
                "gpu_reserve_bytes": 536_870_912,
                "qwen_gpu_layers": 18,
            }
        and value.get("runs", {}).get("treatment_r1", {}).get(
            "configuration"
        ) == {
            "gemma_gpu_layers": 1,
            "gpu_reserve_bytes": 536_870_912,
            "qwen_gpu_layers": 15,
        },
        "GPU diagnostic workload and placement",
    )
    changes = value.get("changes", {})
    quality = value.get("quality", {})
    screen = value.get("screen_gates", {})
    require(
        changes.get("cpu_package_energy_pct", 0) < 0
        and changes.get("gpu_board_energy_pct", 0) > 0
        and changes.get("server_energy_pct", 0) < 0
        and changes.get("wall_service_pct", 0) < 0
        and screen.get("each_pair_server_energy_lower") is True
        and screen.get("each_pair_wall_service_lower") is True
        and screen.get("qwen_first_token_not_regressed") is False
        and quality.get("qwen", {}).get("exact") is True
        and quality.get("gemma", {}).get("exact") is False,
        "GPU diagnostic observations",
    )


def validate_gpu_tensor_bundle(value: dict[str, Any]) -> None:
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == "s42-gpu-tensor-manifest-bundle-v1"
        and value.get("status")
            == "EXACT_MANIFEST_PASS_DIRECT_ATOMIC_STAGING_BLOCKED"
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        "GPU tensor bundle identity",
    )
    gates = value.get("claim_gates", {})
    require(
        gates == {
            "dynamic_energy_admission": False,
            "exact_gpu_tensor_slice_manifests": True,
            "measured_atomic_stage_before_evict": False,
            "measured_final_placement_capacity": True,
            "measured_second_epoch_atomic_staging_capacity": True,
            "runtime_model_buffer_bound_to_raw_tensor_bytes": True,
            "service_abba_revalidated_from_raw_runs": True,
        },
        "GPU tensor bundle gates",
    )
    placements = value.get("placements", {})
    expected_placements = {
        "gemma_gpu_0": (0, 0, 0),
        "gemma_gpu_1": (1, 2_013_281_280, 2),
        "qwen_gpu_15": (15, 10_804_873_216, 156),
        "qwen_gpu_18": (18, 12_786_807_808, 189),
    }
    require(
        set(placements) == set(expected_placements)
        and all(
            (
                placements[name].get("n_gpu_layers"),
                placements[name].get("selected_materialized_raw_bytes"),
                placements[name].get("selected_tensor_count"),
            ) == expected
            for name, expected in expected_placements.items()
        ),
        "GPU tensor bundle placements",
    )
    transition = value.get("transition_slice", {})
    staging = value.get("atomic_staging", {})
    require(
        transition.get("from") == "qwen18+gemma0"
        and transition.get("to") == "qwen15+gemma1"
        and transition.get("qwen_evicted_layer_ids") == [23, 24, 25]
        and transition.get("qwen_evicted_tensor_count") == 33
        and transition.get("qwen_evicted_raw_tensor_bytes")
            == 1_981_934_592
        and transition.get("gemma_added_raw_tensor_bytes")
            == 2_013_281_280
        and transition.get("net_added_raw_tensor_bytes") == 31_346_688
        and staging.get("final_measured_placement_fits") is True
        and staging.get("atomic_stage_before_evict_fits") is False
        and staging.get("stageable_bytes_beyond_reserve") == 205_520_896
        and staging.get("staging_shortfall_bytes") == 2_238_709_760
        and staging.get("qwen15_stageable_bytes_beyond_reserve")
            == 2_624_585_728
        and staging.get("gemma_after_qwen15_atomic_stage_fits") is True
        and staging.get("gemma_after_qwen15_staging_margin_bytes")
            == 180_355_072
        and staging.get("required_transition_sequence") == [
            {
                "from": "qwen18+gemma0",
                "mode": (
                    "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK"
                ),
                "step": 1,
                "to": "qwen15+gemma0",
            },
            {
                "from": "qwen15+gemma0",
                "mode": "ATOMIC_STAGE_BEFORE_EVICT",
                "step": 2,
                "to": "qwen15+gemma1",
            },
        ]
        and staging.get("required_transition_mode")
            == "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK",
        "GPU tensor transition and staging geometry",
    )


def validate_adoption_energy_screen(value: dict[str, Any]) -> None:
    claimed_hash = value.get("record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    require(
        value.get("schema") == "s42-adoption-energy-screen-abba-v1"
        and value.get("status") == "PASS"
        and value.get("admission")
            == "ENERGY_SCREEN_FAIL_FULL_TRACE_BLOCKED"
        and value.get("full_trace_authorized") is False
        and type(claimed_hash) is str
        and hashlib.sha256(canonical(unsigned)).hexdigest() == claimed_hash,
        "adoption energy screen identity",
    )
    require(
        all(value.get("validity_gates", {}).values())
        and not any(value.get("outcome_gates", {}).values())
        and value.get("changes", {}).get("fleet_j_change_pct", 0) > 0
        and value.get("changes", {}).get("duration_s_change_pct", 0) > 0
        and all(
            saving < 0
            for saving in value.get("pair_fleet_energy_saving_pct", [])
        )
        and value.get("next_gate")
            == "DO_NOT_RUN_FULL_TRACE_DIAGNOSE_ENERGY_OR_LATENCY",
        "adoption energy screen outcome",
    )
    require(
        value.get("workload") == {
            "gemma_input_tokens": 271,
            "gemma_output_tokens": 41,
            "qwen_input_tokens": 936,
            "qwen_output_tokens": 117,
            "requests": 7,
            "source_trace_sha256": TRACE_SHA256,
        }
        and value.get("run_order") == [
            "control_r1", "dynamic_r1", "dynamic_r2", "control_r2"
        ],
        "adoption energy screen workload",
    )


def metric(
    mean: int,
    upper: int,
    lower: int,
    *,
    measured: bool,
) -> MetricEstimate:
    return MetricEstimate(
        mean=mean,
        upper=upper,
        lower=lower,
        sample_count=3 if measured else 0,
        measured=measured,
    )


def workload_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effective = [EFFECTIVE_MODELS[row["model_id"]] for row in rows]
    runs: list[dict[str, Any]] = []
    for row, model_id in zip(rows, effective, strict=True):
        if runs and runs[-1]["model_id"] == model_id:
            runs[-1]["requests"] += 1
            runs[-1]["arrival_last_us"] = row["arrival_us"]
        else:
            runs.append({
                "arrival_first_us": row["arrival_us"],
                "arrival_last_us": row["arrival_us"],
                "model_id": model_id,
                "requests": 1,
            })
    counts = {
        model_id: effective.count(model_id)
        for model_id in sorted(set(effective))
    }
    return {
        "arrival_first_us": rows[0]["arrival_us"],
        "arrival_last_us": rows[-1]["arrival_us"],
        "arrival_model_bursts": runs,
        "arrival_model_transitions": len(runs) - 1,
        "effective_model_requests": counts,
        "held_for_static_gemma_phase": counts["gemma-4-12b-f16-proxy"],
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "requests": len(rows),
    }


def phone_snapshot(
    plan: PhoneResidencyPlan,
    measured_available_bytes: int,
) -> tuple[DynamicResidencySnapshot, dict[str, str]]:
    placements: dict[str, DynamicWeightPlacement] = {}
    placement_by_slice: dict[str, str] = {}
    for session in plan.sessions:
        for row in session.slices:
            placement_id = f"{session.session_id}:{row.slice_id}"
            placement_by_slice[row.slice_id] = placement_id
            spec = DynamicWeightPlacementSpec(
                placement_id=placement_id,
                slice_id=row.slice_id,
                model_id=row.model_id,
                model_hash=row.model_hash,
                weight_hash=row.weight_hash,
                resource_id=plan.memory_resource_id,
                resident_bytes=row.resident_bytes,
                execution_resource_ids=(plan.shared_compute_resource_id,),
                runtime_binding_ids=(
                    session.compute_backend,
                    session.session_id,
                ),
                evidence_ids=row.evidence_ids,
            )
            placements[placement_id] = DynamicWeightPlacement(
                spec=spec,
                generation=1,
                resident_since_us=0,
                minimum_resident_until_us=0,
            )
    occupied_bytes = plan.memory_capacity_bytes - measured_available_bytes
    snapshot = DynamicResidencySnapshot(
        snapshot_id="op15-f16-static-shadow-v1",
        epoch_key=canonical_sha256({
            "plan_id": plan.plan_id,
            "schema": "s42-phone-shadow-source-v1",
        }),
        generation=1,
        captured_at_us=0,
        valid_until_us=1_000_000_000,
        memory={
            plan.memory_resource_id: DeviceMemoryCapacity(
                resource_id=plan.memory_resource_id,
                capacity_bytes=plan.memory_capacity_bytes,
                occupied_bytes=occupied_bytes,
                reserve_bytes=plan.minimum_available_bytes,
            ),
        },
        placements=placements,
    )
    return snapshot, placement_by_slice


def phone_rotation_diagnostic(
    source: DynamicResidencySnapshot,
) -> dict[str, Any]:
    original = min(
        source.placements.values(),
        key=lambda row: (row.spec.resident_bytes, row.placement_id),
    )
    target = DynamicWeightPlacementSpec(
        placement_id="htp3-shadow-copy:" + original.spec.slice_id,
        slice_id=original.spec.slice_id,
        model_id=original.spec.model_id,
        model_hash=original.spec.model_hash,
        weight_hash=original.spec.weight_hash,
        resource_id=original.spec.resource_id,
        resident_bytes=original.spec.resident_bytes,
        execution_resource_ids=("op15-htp",),
        runtime_binding_ids=("HTP3", "htp3"),
        evidence_ids=original.spec.evidence_ids,
    )
    candidate = DynamicResidencyCandidate(
        candidate_id="rotate-smallest-known-slice-to-htp3",
        source_snapshot_id=source.snapshot_id,
        source_snapshot_sha256=canonical_sha256(source.to_json()),
        source_generation=source.generation,
        source_epoch_key=source.epoch_key,
        target=target,
        evict_placement_ids=(original.placement_id,),
        transition_resource_ids=(
            "desktop-usb-root",
            "op15-functionfs",
            "op15-htp3",
        ),
        expected_reuse_count=2,
        minimum_reuse_count=2,
        minimum_residency_us=1,
        latest_ready_us=900_000_000,
        energy_boundary_id="cpu-package+gpu-board+whole-phone",
        accounting_scope="full-f16-burstgpt-shadow",
        baseline_latency_us=metric(1_000, 1_000, 1_000, measured=False),
        resident_latency_us=metric(500, 500, 500, measured=False),
        load_latency_us=metric(100, 100, 100, measured=False),
        eviction_latency_us=metric(10, 10, 10, measured=False),
        baseline_energy_uj=metric(1_000, 1_000, 1_000, measured=False),
        resident_energy_uj=metric(500, 500, 500, measured=False),
        load_energy_uj=metric(100, 100, 100, measured=False),
        eviction_energy_uj=metric(10, 10, 10, measured=False),
        evidence_ids=("shadow-only-no-transition-measurement",),
    )
    strict = select_dynamic_residency_transition(
        source,
        (candidate,),
        now_us=1,
        transition_resource_ready_us=1,
        require_measured=True,
    )
    capacity_only = select_dynamic_residency_transition(
        source,
        (candidate,),
        now_us=1,
        transition_resource_ready_us=1,
        require_measured=False,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "diagnostic_only": True,
        "strict_policy_reason": dict(strict.rejected)[candidate.candidate_id],
        "capacity_only_reason": dict(capacity_only.rejected)[
            candidate.candidate_id
        ],
        "target_resident_bytes": target.resident_bytes,
    }


def build_shadow(
    aggregate: dict[str, Any],
    rows: list[dict[str, Any]],
    phone_plan: PhoneResidencyPlan,
    gpu_capacity: dict[str, Any],
    gpu_diagnostic: dict[str, Any],
    gpu_tensor_bundle: dict[str, Any],
    adoption_energy_screen: dict[str, Any],
    *,
    aggregate_hash: str,
    trace_hash: str,
    phone_plan_hash: str,
    gpu_capacity_hash: str,
    gpu_diagnostic_hash: str,
    gpu_tensor_bundle_hash: str,
    adoption_energy_screen_hash: str,
) -> dict[str, Any]:
    validate_aggregate(aggregate)
    validate_trace(rows, trace_hash)
    validate_gpu_capacity(gpu_capacity)
    validate_gpu_diagnostic(gpu_diagnostic)
    validate_gpu_tensor_bundle(gpu_tensor_bundle)
    validate_adoption_energy_screen(adoption_energy_screen)
    require(
        phone_plan.memory_resource_id == "op15-dram"
        and phone_plan.shared_compute_resource_id == "op15-htp"
        and len(phone_plan.sessions) == 3,
        "OP15 residency identity",
    )
    treatment_runs = [
        row for row in aggregate["runs"].values()
        if row.get("arm") == "op15"
    ]
    require(len(treatment_runs) == 2, "two measured treatment repeats")
    require(
        all(
            row.get("resources", {}).get("process_swap_max_bytes") == 0
            for row in treatment_runs
        ),
        "zero process swap",
    )
    measured_available_bytes = min(
        row["resident"]["mem_available_kib"] * 1024
        for row in treatment_runs
    )
    source, _ = phone_snapshot(phone_plan, measured_available_bytes)
    memory = source.memory[phone_plan.memory_resource_id]
    known_slice_bytes = sorted(
        placement.spec.resident_bytes
        for placement in source.placements.values()
    )
    rotation = phone_rotation_diagnostic(source)
    require(
        rotation["strict_policy_reason"] == "MEASUREMENT_REQUIRED"
        and rotation["capacity_only_reason"]
            == "ATOMIC_STAGING_MEMORY",
        "phone shadow diagnostic",
    )
    phone_headroom = memory.available_bytes
    smallest_slice = known_slice_bytes[0]
    gpu_baseline = gpu_capacity["baseline_gpu"]
    qwen_stage, dual_stage = gpu_capacity["stages"]
    gpu_reserve_bytes = gpu_capacity["configuration"][
        "gpu_reserve_bytes"
    ]
    qwen_gpu_bytes = (
        qwen_stage["gpu"]["memory_used_bytes"]
        - gpu_baseline["memory_used_bytes"]
    )
    gemma_incremental_gpu_bytes = (
        dual_stage["gpu"]["memory_used_bytes"]
        - qwen_stage["gpu"]["memory_used_bytes"]
    )
    dual_gpu_bytes = (
        dual_stage["gpu"]["memory_used_bytes"]
        - gpu_baseline["memory_used_bytes"]
    )
    baseline = {
        "accounted_fleet_energy_j": aggregate["treatment_mean"]["fleet_j"],
        "accounted_fleet_energy_saving_pct_vs_cpu_overflow": -aggregate[
            "changes"
        ]["fleet_j_change_pct"],
        "makespan_s": aggregate["treatment_mean"]["duration_s"],
        "makespan_saving_pct_vs_cpu_overflow": -aggregate["changes"][
            "duration_s_change_pct"
        ],
        "policy": "static-qwen18-then-gemma25-with-three-resident-phone-slices",
        "status": "MEASURED_FALLBACK",
    }
    output: dict[str, Any] = {
        "baseline": baseline,
        "evidence": {
            "adoption_energy_screen_file_sha256": (
                adoption_energy_screen_hash
            ),
            "aggregate_file_sha256": aggregate_hash,
            "gpu_capacity_file_sha256": gpu_capacity_hash,
            "gpu_diagnostic_file_sha256": gpu_diagnostic_hash,
            "gpu_tensor_bundle_file_sha256": gpu_tensor_bundle_hash,
            "phone_plan_file_sha256": phone_plan_hash,
            "trace_file_sha256": trace_hash,
        },
        "gates": {
            "adoption_energy_screen_exact_output": True,
            "adoption_energy_screen_positive": False,
            "adoption_energy_screen_repeated": True,
            "adoption_transition_full_boundary": True,
            "dynamic_incremental_energy_claim": False,
            "dual_residency_abba_repeated": True,
            "dual_residency_capacity": True,
            "dual_residency_exact_output": False,
            "dual_residency_repeated_service_directional": True,
            "exact_gpu_tensor_slice_manifests": True,
            "executable_gpu_backfill": False,
            "fallback_backed_transition_contract": True,
            "fallback_ready_route_receipt": False,
            "fallback_result_bound": True,
            "phone_2_gib_reserve_preserved": True,
            "phone_fourth_known_slice_fits": smallest_slice <= phone_headroom,
            "source_equal_work": True,
            "source_zero_process_swap": True,
            "strict_atomic_gpu_staging": False,
        },
        "gpu_shadow": {
            "admitted_backfills": 0,
            "capacity_candidate": {
                "combined_gpu_bytes": dual_gpu_bytes,
                "free_beyond_reserve_bytes": (
                    dual_stage["gpu"]["memory_free_bytes"]
                    - gpu_reserve_bytes
                ),
                "gemma_gpu_layers": 1,
                "gemma_incremental_gpu_bytes": (
                    gemma_incremental_gpu_bytes
                ),
                "gemma_load_us": dual_stage["load_us"],
                "gpu_free_bytes": dual_stage["gpu"][
                    "memory_free_bytes"
                ],
                "gpu_reserve_bytes": gpu_reserve_bytes,
                "process_swap_max_bytes": 0,
                "qwen_gpu_bytes": qwen_gpu_bytes,
                "qwen_gpu_layers": 15,
                "qwen_load_us": qwen_stage["load_us"],
                "status": "CAPACITY_ONLY",
            },
            "exact_tensor_slice_bytes": {
                "gemma_gpu_0_raw_bytes": gpu_tensor_bundle["placements"][
                    "gemma_gpu_0"
                ]["selected_materialized_raw_bytes"],
                "gemma_gpu_1_raw_bytes": gpu_tensor_bundle["placements"][
                    "gemma_gpu_1"
                ]["selected_materialized_raw_bytes"],
                "qwen_gpu_15_raw_bytes": gpu_tensor_bundle["placements"][
                    "qwen_gpu_15"
                ]["selected_materialized_raw_bytes"],
                "qwen_gpu_18_raw_bytes": gpu_tensor_bundle["placements"][
                    "qwen_gpu_18"
                ]["selected_materialized_raw_bytes"],
                "transition": gpu_tensor_bundle["transition_slice"],
            },
            "physical_transition_constraint": gpu_tensor_bundle[
                "atomic_staging"
            ],
            "incremental_energy_saving_j": None,
            "predicted_bubble_coverage_ppm": None,
            "request_path_bubble_receipts": 0,
            "service_energy_abba": {
                "admission_decision": gpu_diagnostic["admission"][
                    "decision"
                ],
                "changes": gpu_diagnostic["changes"],
                "control_mean_of_two": gpu_diagnostic[
                    "control_mean_of_two"
                ],
                "pair_server_energy_saving_pct": gpu_diagnostic[
                    "pair_server_energy_saving_pct"
                ],
                "pair_wall_service_saving_pct": gpu_diagnostic[
                    "pair_wall_service_saving_pct"
                ],
                "quality": gpu_diagnostic["quality"],
                "screen_gates": gpu_diagnostic["screen_gates"],
                "screen_outcome": gpu_diagnostic["screen_outcome"],
                "status": gpu_diagnostic["status"],
                "treatment_mean_of_two": gpu_diagnostic[
                    "treatment_mean_of_two"
                ],
            },
            "adoption_energy_screen_abba": {
                "admission": adoption_energy_screen["admission"],
                "changes": adoption_energy_screen["changes"],
                "control_mean": adoption_energy_screen["control_mean"],
                "dynamic_mean": adoption_energy_screen["dynamic_mean"],
                "next_gate": adoption_energy_screen["next_gate"],
                "pair_fleet_energy_saving_pct": adoption_energy_screen[
                    "pair_fleet_energy_saving_pct"
                ],
                "status": adoption_energy_screen["status"],
                "validity_gates": adoption_energy_screen[
                    "validity_gates"
                ],
            },
            "status": "BLOCKED_CONTENTION_ENERGY_REGRESSION",
            "unavailable_inputs": [
                "fallback-backed drain/reload transition receipts",
                "protected GPU-ready lower-bound fence timestamps",
                "candidate restore upper latency",
                "contention-adjusted micro-filler latency and energy",
                "full-trace equal-work and output-quality receipts",
            ],
        },
        "mode": "SHADOW_ONLY",
        "next_measurements": [
            "measure fallback-backed GPU drain/reload transitions",
            "replace whole-request overlap with a bounded micro-filler",
            "measure shared CPU and CUDA contention plus restore bounds",
            "capture protected GPU fences and idle intervals",
            "run the full trace only after the conservative repeated bound passes",
        ],
        "phone_memory": {
            "atomic_rotation_diagnostic": rotation,
            "capacity_bytes": memory.capacity_bytes,
            "known_resident_slice_bytes": known_slice_bytes,
            "mandatory_reserve_bytes": memory.reserve_bytes,
            "measured_available_bytes": measured_available_bytes,
            "occupied_bytes": memory.occupied_bytes,
            "resident_weight_bytes": phone_plan.resident_bytes,
            "smallest_known_new_slice_bytes": smallest_slice,
            "stageable_bytes_beyond_reserve": phone_headroom,
            "stageable_shortfall_for_smallest_slice_bytes": max(
                0, smallest_slice - phone_headroom
            ),
        },
        "policy": {
            "fast_loop": "backfill-only-with-ready-resident-slice",
            "slow_loop": "one-receipt-bound-placement-transition-per-epoch",
            "static_fallback": baseline["policy"],
            "transition_modes": [
                "ATOMIC_STAGE_BEFORE_EVICT",
                "DRAIN_OR_EVICT_BEFORE_STAGE_WITH_READY_FALLBACK",
            ],
            "unified_scheduler_package": "research_dev.scheduler",
        },
        "schema": SCHEMA,
        "shadow_accounting": {
            "admitted_bytes": 0,
            "evicted_bytes": 0,
            "incremental_energy_saving_j": None,
            "proposed_phone_bytes": smallest_slice,
            "useful_prefetch_bytes": 0,
        },
        "status": "BLOCKED",
        "workload": workload_summary(rows),
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate", type=Path, default=DEFAULT_AGGREGATE
    )
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument(
        "--phone-plan", type=Path, default=DEFAULT_PHONE_PLAN
    )
    parser.add_argument(
        "--gpu-capacity", type=Path, default=DEFAULT_GPU_CAPACITY
    )
    parser.add_argument(
        "--gpu-diagnostic", type=Path, default=DEFAULT_GPU_DIAGNOSTIC
    )
    parser.add_argument(
        "--gpu-tensor-bundle", type=Path,
        default=DEFAULT_GPU_TENSOR_BUNDLE,
    )
    parser.add_argument(
        "--adoption-energy-screen", type=Path,
        default=DEFAULT_ADOPTION_ENERGY_SCREEN,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    try:
        aggregate = read_object(args.aggregate)
        trace_rows = read_trace(args.trace)
        phone_plan = PhoneResidencyPlan.from_json(
            read_object(args.phone_plan)
        )
        gpu_capacity = read_object(args.gpu_capacity)
        gpu_diagnostic = read_object(args.gpu_diagnostic)
        gpu_tensor_bundle = read_object(args.gpu_tensor_bundle)
        adoption_energy_screen = read_object(args.adoption_energy_screen)
        output = build_shadow(
            aggregate,
            trace_rows,
            phone_plan,
            gpu_capacity,
            gpu_diagnostic,
            gpu_tensor_bundle,
            adoption_energy_screen,
            aggregate_hash=sha256(args.aggregate),
            trace_hash=sha256(args.trace),
            phone_plan_hash=sha256(args.phone_plan),
            gpu_capacity_hash=sha256(args.gpu_capacity),
            gpu_diagnostic_hash=sha256(args.gpu_diagnostic),
            gpu_tensor_bundle_hash=sha256(args.gpu_tensor_bundle),
            adoption_energy_screen_hash=sha256(
                args.adoption_energy_screen
            ),
        )
        args.output.write_bytes(canonical(output))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"dynamic residency shadow failed: {exc}\n")
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": output["record_sha256"],
        "status": output["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
