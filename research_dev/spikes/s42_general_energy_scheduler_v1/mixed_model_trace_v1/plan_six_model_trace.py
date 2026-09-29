#!/usr/bin/env python3
"""Plan the six-model trace from measured residency and route evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
S41_ROOT = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1"
)
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT)]

from research_dev.scheduler import (  # noqa: E402
    CohortRoute,
    QueueRequest,
    ResidencyResource,
    optimize_parallel_assignment,
    parallel_assignment_to_json,
    schedule_route,
)
from verify_mixed_trace import validate as validate_trace  # noqa: E402


SCHEMA = "s42-six-model-residency-plan-v1"
TRACE_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
BASELINE_SCHEMA = "s42-six-model-physical-result-v1"
OP15_SCHEMA = "s41-burstgpt-llama-server-result-v1"

QWEN14 = "qwen3-14b-q4_k_m"
QWEN8 = "qwen3-8b-q8_0"
GEMMA12 = "gemma-4-12b-it-q4_0"
QWEN06 = "qwen3-0.6b-q8_0"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
GEMMA_E2B = "gemma-4-e2b-it-q8_0-vlm"

GPU_ROUTE = f"{GEMMA12}-cuda"
PHONE_ROUTE = f"{GEMMA12}-cpu-op15-ffn"


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            value = json.loads(line)
            require(type(value) is dict, f"row {line_number}")
            result.append(value)
    return result


def promotion_record(
    baseline: dict[str, Any], model_id: str
) -> dict[str, Any]:
    values = [
        record for record in baseline["server_records"]
        if record.get("stage") == "promotion"
        and record.get("model_id") == model_id
    ]
    require(len(values) == 1, f"promotion record: {model_id}")
    return values[0]


def request_end_us(
    baseline: dict[str, Any], model_id: str
) -> int:
    paid_start_ns = baseline["paid_start_ns"]
    values = [
        row["completion_ns"] - paid_start_ns
        for row in baseline["request_results"]
        if row.get("execution_model_id") == model_id
    ]
    require(values, f"request results: {model_id}")
    return round(max(values) / 1000)


def timing_service_us(row: dict[str, Any]) -> int:
    value = row.get("prompt_ms", 0) + row.get("predicted_ms", 0)
    require(type(value) in (int, float) and value > 0, "service timing")
    return round(value * 1000)


def validate_baseline(
    baseline: dict[str, Any], rows: list[dict[str, Any]], trace_sha256: str
) -> None:
    require(
        baseline.get("schema") == BASELINE_SCHEMA
        and baseline.get("status") == "PASS",
        "baseline status",
    )
    require(baseline.get("trace_sha256") == trace_sha256, "baseline trace")
    request_results = baseline.get("request_results")
    require(type(request_results) is list, "baseline request results")
    require(
        sorted(row["mixed_request_index"] for row in request_results)
        == list(range(len(rows))),
        "baseline request conservation",
    )
    require(
        baseline.get("policy", {}).get("gpu_sequence")
        == [QWEN14, QWEN8, GEMMA12],
        "baseline GPU sequence",
    )


def validate_op15(
    op15: dict[str, Any],
    op15_sha256: str,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    require(
        op15.get("schema") == OP15_SCHEMA and op15.get("status") == "PASS",
        "OP15 cohort status",
    )
    require(
        receipt.get("status") == "PASS"
        and receipt.get("result", {}).get("sha256") == op15_sha256,
        "OP15 runtime-gate receipt",
    )
    preflight = op15.get("preflight", {})
    require(
        preflight.get("requests_sha256")
        == manifest["source"]["sha256"],
        "OP15 source trace",
    )
    require(
        preflight.get("cold_model_sha256")
        == manifest["model_inventory"][GEMMA12]["artifact_sha256"],
        "OP15 model identity",
    )
    split = preflight.get("split_policy", {})
    require(
        preflight.get("mode") == "op15"
        and split.get("id") == "i3-hidden-wait"
        and split.get("io") == "f16"
        and split.get("max_columns") == 11136,
        "OP15 route identity",
    )
    phone = op15.get("phone", {})
    require(
        phone.get("bridge", {}).get("reset_recoveries") == 0
        and phone.get("ffn", {}).get("status") == "ok",
        "OP15 transport qualification",
    )


def build_plan(
    requests_path: Path,
    manifest_path: Path,
    baseline_path: Path,
    op15_path: Path,
    receipt_path: Path,
    hardware_profile_path: Path,
) -> dict[str, Any]:
    validate_trace(requests_path, manifest_path)
    rows = read_jsonl(requests_path)
    manifest = read_object(manifest_path)
    baseline = read_object(baseline_path)
    op15 = read_object(op15_path)
    receipt = read_object(receipt_path)
    hardware_profile = read_object(hardware_profile_path)

    require(
        len(rows) == 114
        and all(row.get("schema") == TRACE_SCHEMA for row in rows),
        "trace geometry",
    )
    trace_sha256 = digest_file(requests_path)
    baseline_sha256 = digest_file(baseline_path)
    op15_sha256 = digest_file(op15_path)
    validate_baseline(baseline, rows, trace_sha256)
    validate_op15(op15, op15_sha256, receipt, manifest)

    hardware = hardware_profile.get("hardware", {})
    gpu_capacity = hardware.get("gpu_vram_bytes")
    phone_allocation_limit = hardware.get(
        "phone_htp_resident_weight_budget_bytes"
    )
    require(
        type(gpu_capacity) is int and gpu_capacity > 0,
        "GPU VRAM profile",
    )
    require(
        type(phone_allocation_limit) is int
        and phone_allocation_limit > 0,
        "phone HTP profile",
    )

    gemma_rows = [
        row for row in rows if row["execution_model_id"] == GEMMA12
    ]
    requests = tuple(
        QueueRequest(
            request_id=str(row["mixed_request_index"]),
            model_id=GEMMA12,
            arrival_us=row["arrival_us"],
            deadline_us=row["arrival_us"] + row["slo_us"],
        )
        for row in gemma_rows
    )
    mixed_by_source = {
        row["stream_request_index"]: str(row["mixed_request_index"])
        for row in gemma_rows
    }

    gpu_service = {
        str(row["mixed_request_index"]): timing_service_us(row)
        for row in baseline["request_results"]
        if row["execution_model_id"] == GEMMA12
    }
    phone_results = [
        row for row in op15["request_results"] if row["role"] == "cold"
    ]
    require(
        set(row["request_index"] for row in phone_results)
        == set(mixed_by_source),
        "OP15 Gemma cohort coverage",
    )
    phone_service = {
        mixed_by_source[row["request_index"]]: timing_service_us(row)
        for row in phone_results
    }

    gpu_ready_us = round(
        promotion_record(baseline, GEMMA12)["ready_s"] * 1_000_000
    )
    phone_resource = ResidencyResource(
        "op15-phone-dram",
        10 * 1024 * 1024 * 1024,
        phone_allocation_limit,
    )
    gpu_resource = ResidencyResource("cuda0-vram", gpu_capacity)
    phone_route = CohortRoute(
        route_id=PHONE_ROUTE,
        model_id=GEMMA12,
        resource_id=phone_resource.resource_id,
        slots=2,
        service_us=phone_service,
        residency_id="gemma4-q4-ffn-htp0",
        resident_bytes=phone_allocation_limit,
        preloaded=True,
    )
    gpu_route = CohortRoute(
        route_id=GPU_ROUTE,
        model_id=GEMMA12,
        resource_id=gpu_resource.resource_id,
        slots=8,
        service_us=gpu_service,
        residency_id="gemma4-q4-full-cuda0",
        resident_bytes=11_512_315_904,
        preloaded=False,
    )
    assignment = optimize_parallel_assignment(
        requests,
        phone_route,
        phone_resource,
        0,
        gpu_route,
        gpu_resource,
        gpu_ready_us,
    )
    gpu_only = schedule_route(requests, gpu_route, gpu_resource, gpu_ready_us)
    predicted_reduction_us = gpu_only.finish_us - assignment.makespan_us
    baseline_duration_us = round(baseline["metrics"]["duration_s"] * 1_000_000)

    phone_ids = {int(value) for value in assignment.source_request_ids}
    routes: dict[str, str] = {}
    for row in rows:
        index = row["mixed_request_index"]
        model_id = row["execution_model_id"]
        if model_id == GEMMA12:
            route = PHONE_ROUTE if index in phone_ids else GPU_ROUTE
        elif model_id in (QWEN14, QWEN8):
            route = f"{model_id}-cuda"
        else:
            route = f"{model_id}-cpu"
        routes[str(index)] = route

    qwen14_end_us = request_end_us(baseline, QWEN14)
    qwen8_ready_us = round(
        promotion_record(baseline, QWEN8)["ready_s"] * 1_000_000
    )
    qwen8_end_us = request_end_us(baseline, QWEN8)
    transitions = [
        {
            "source_model_id": QWEN14,
            "target_model_id": QWEN8,
            "source_finish_us": qwen14_end_us,
            "target_ready_us": qwen8_ready_us,
            "measured_transition_us": qwen8_ready_us - qwen14_end_us,
        },
        {
            "source_model_id": QWEN8,
            "target_model_id": GEMMA12,
            "source_finish_us": qwen8_end_us,
            "target_ready_us": gpu_ready_us,
            "measured_transition_us": gpu_ready_us - qwen8_end_us,
        },
    ]
    require(
        all(value["measured_transition_us"] > 0 for value in transitions),
        "measured GPU transitions",
    )

    selected = sorted(phone_ids)
    require(selected == [70, 78, 94], "qualified OP15 assignment drift")
    require(len(routes) == 114, "route conservation")

    result = {
        "evidence": {
            "baseline_result": {
                "path": str(baseline_path),
                "sha256": baseline_sha256,
            },
            "op15_result": {
                "path": str(op15_path),
                "sha256": op15_sha256,
            },
            "op15_runtime_gate": {
                "path": str(receipt_path),
                "sha256": digest_file(receipt_path),
            },
        },
        "gpu": {
            "capacity_bytes": gpu_capacity,
            "sequence": [QWEN14, QWEN8, GEMMA12],
            "transitions": transitions,
        },
        "objective": {
            "order": [
                "deadline_misses",
                "weighted_tardiness_us",
                "makespan_us",
                "weighted_completion_us",
            ],
            "predicted_baseline_duration_us": baseline_duration_us,
            "predicted_duration_us": baseline_duration_us - predicted_reduction_us,
            "predicted_reduction_us": predicted_reduction_us,
            "status": "latency_profiled_energy_pending_concurrent_measurement",
        },
        "parallel_assignment": parallel_assignment_to_json(assignment),
        "phone": {
            "active_htp_mapping_limit_bytes": phone_allocation_limit,
            "device_dram_capacity_bytes": phone_resource.capacity_bytes,
            "model_id": GEMMA12,
            "preloaded_before_paid_start": True,
            "desktop_companion_retire_when_idle_before_cuda": GEMMA12,
            "resident_shard_id": phone_route.residency_id,
            "route_id": PHONE_ROUTE,
            "scheduler_effective_lanes": phone_route.slots,
            "selected_mixed_request_indices": selected,
            "split_policy": {
                "id": "i3-hidden-wait",
                "io": "f16",
                "layer_mask": "0x0000ffffffffffff",
                "max_columns": 11136,
                "timeout_ms": 35000,
                "table": "1:9664,3:8192,8:4096,128:8192,512:11136",
            },
        },
        "rejected_phone_models": {
            model_id: "no exact-model qualified OP15 composed route"
            for model_id in (QWEN14, QWEN8, QWEN06, LLAMA1, GEMMA_E2B)
        },
        "request_routes": routes,
        "schema": SCHEMA,
        "status": "PASS",
        "trace_sha256": trace_sha256,
    }
    result["plan_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--op15-result", type=Path, required=True)
    parser.add_argument(
        "--op15-runtime-gate",
        type=Path,
        default=(
            S42_ROOT
            / "physical_ab_v1/STAGE6_TREATMENT_RUNTIME_GATE_V2.json"
        ),
    )
    parser.add_argument(
        "--hardware-profile",
        type=Path,
        default=S41_ROOT / "mixed_scheduler_v1/current_profile.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require(args.output.is_absolute() and not args.output.exists(), "output")
    plan = build_plan(
        args.requests,
        args.manifest,
        args.baseline_result,
        args.op15_result,
        args.op15_runtime_gate,
        args.hardware_profile,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(plan))
    print(json.dumps({
        "phone_requests": plan["phone"]["selected_mixed_request_indices"],
        "plan_sha256": plan["plan_sha256"],
        "predicted_duration_s": (
            plan["objective"]["predicted_duration_us"] / 1_000_000
        ),
        "predicted_reduction_s": (
            plan["objective"]["predicted_reduction_us"] / 1_000_000
        ),
        "status": plan["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
