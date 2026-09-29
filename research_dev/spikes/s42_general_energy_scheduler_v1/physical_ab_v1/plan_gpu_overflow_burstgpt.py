#!/usr/bin/env python3
"""Build a UnifiedScheduler plan for BurstGPT GPU overflow."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    ProfileBundle,
    Request,
    UnifiedScheduler,
    decision_to_json,
)


SCHEMA = "s42-burstgpt-gpu-overflow-plan-v1"
TRACE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"
RESULT_SCHEMA = "s41-hierarchical-burstgpt-result-v1"
HOT_SOURCE_MODEL = "gemma-4-12b-it-q8_0"
COLD_SOURCE_MODEL = "qwen3-14b-q4_k_m"
GPU_EXECUTOR = "gemma_cuda_full"
HELPER_EXECUTOR = "gemma_cpu_op15"
GPU_RESOURCE = "cuda0"
CPU_RESOURCE = "desktop-cpu"
PHONE_RESOURCE = "op15-htp"
USB_RESOURCE = "op15-usb"
SWITCH_FEATURE = "gpu_switch_start_us"


class PlanError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
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


def read_trace(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    require(
        len(rows) == 74
        and all(type(row) is dict and row.get("schema") == TRACE_SCHEMA for row in rows),
        "trace geometry",
    )
    require(
        Counter(row["model_id"] for row in rows)
        == Counter({HOT_SOURCE_MODEL: 57, COLD_SOURCE_MODEL: 17}),
        "trace model roles",
    )
    require(
        [row["request_index"] for row in rows] == list(range(74)),
        "trace request indices",
    )
    return rows


def result_rows(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = result.get("request_results")
    require(type(rows) is list, "result request rows")
    mapped = {row.get("request_index"): row for row in rows if type(row) is dict}
    require(
        len(mapped) == len(rows)
        and all(type(index) is int for index in mapped),
        "result request identities",
    )
    return mapped


def service_us(row: dict[str, Any]) -> int:
    dispatch_ns = row.get("dispatch_ns")
    completion_ns = row.get("completion_ns")
    require(
        type(dispatch_ns) is int
        and type(completion_ns) is int
        and completion_ns > dispatch_ns,
        "request service interval",
    )
    return max(1, round((completion_ns - dispatch_ns) / 1000))


def energy_unknown() -> dict[str, object]:
    return {
        "cost_uj": None,
        "lower_error_ppm": 0,
        "status": "unknown",
        "upper_error_ppm": 0,
    }


def latency(value_us: int) -> dict[str, object]:
    return {
        "cost_us": {
            "fixed": value_us,
            "input_token": 0,
            "kind": "affine_tokens_v1",
            "output_token": 0,
        },
        "measured": True,
        "sample_count": 1,
        "ucb_add_us": max(250_000, value_us // 5),
    }


def route(
    *,
    route_id: str,
    workload_id: str,
    baseline: bool,
    value_us: int,
    evidence_id: str,
) -> dict[str, object]:
    if baseline:
        resources = {GPU_RESOURCE: 1}
        granularity = "task"
        overlap = {"status": "not_applicable"}
    else:
        resources = {CPU_RESOURCE: 1, PHONE_RESOURCE: 1, USB_RESOURCE: 1}
        granularity = "operator"
        overlap = {"status": "unknown"}
    result: dict[str, object] = {
        "baseline": baseline,
        "energy": energy_unknown(),
        "evidence_ids": [evidence_id],
        "granularity": granularity,
        "latency": latency(value_us),
        "overlap": overlap,
        "placement_verified": True,
        "quality_class": "approximate",
        "resident": True,
        "resource_leases": [],
        "resource_slots": resources,
        "route_id": route_id,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": 6_975_878_176,
        "workload_id": workload_id,
    }
    if not baseline:
        result["finish_before_feature"] = SWITCH_FEATURE
    return result


def scheduler_request(
    row: dict[str, Any], workload_id: str, switch_start_us: int
) -> Request:
    return Request(
        request_id=row["event_id"],
        workload_id=workload_id,
        arrival_us=row["arrival_us"],
        deadline_us=row["arrival_us"] + row["slo_us"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        quality_requirement="approximate",
        features={SWITCH_FEATURE: switch_start_us},
    )


def build_plan(
    trace_path: Path,
    gpu_control_path: Path,
    helper_path: Path | None,
    mode: str,
    switch_guard_us: int,
    gpu_ready_guard_us: int,
) -> dict[str, Any]:
    require(mode in {"control", "shadow"}, "scheduler mode")
    require(type(switch_guard_us) is int and switch_guard_us >= 0, "switch guard")
    require(
        type(gpu_ready_guard_us) is int and gpu_ready_guard_us >= 0,
        "GPU ready guard",
    )
    require((mode == "control") == (helper_path is None), "helper evidence mode")

    rows = read_trace(trace_path)
    trace_hash = digest_file(trace_path)
    control = read_object(gpu_control_path)
    require(
        control.get("schema") == RESULT_SCHEMA and control.get("status") == "PASS",
        "GPU control result",
    )
    control_rows = result_rows(control)
    require(set(control_rows) == set(range(74)), "GPU control request coverage")
    cold_rows = [row for row in rows if row["model_id"] == COLD_SOURCE_MODEL]
    cold_indices = {row["request_index"] for row in cold_rows}
    require(
        all(
            control_rows[index].get("event_id") == rows[index]["event_id"]
            and control_rows[index].get("route") == GPU_EXECUTOR
            for index in cold_indices
        ),
        "GPU baseline route coverage",
    )

    switch = control.get("switch")
    require(type(switch) is dict, "GPU switch evidence")
    hot_end_s = switch.get("hot_end_s")
    gpu_ready_s = switch.get("gpu_ready_s")
    require(
        not isinstance(hot_end_s, bool)
        and isinstance(hot_end_s, (int, float))
        and not isinstance(gpu_ready_s, bool)
        and isinstance(gpu_ready_s, (int, float))
        and gpu_ready_s > hot_end_s > 0,
        "GPU switch timing",
    )
    switch_start_us = round(hot_end_s * 1_000_000) - switch_guard_us
    protected_until_us = round(gpu_ready_s * 1_000_000) + gpu_ready_guard_us
    require(
        switch_start_us > max(row["arrival_us"] for row in rows)
        and protected_until_us > switch_start_us,
        "GPU protected window",
    )

    helper_rows: dict[int, dict[str, Any]] = {}
    helper_hash = None
    if helper_path is not None:
        helper = read_object(helper_path)
        require(
            helper.get("schema") == RESULT_SCHEMA and helper.get("status") == "PASS",
            "helper result",
        )
        helper_all = result_rows(helper)
        require(
            set(helper_all) == set(range(74))
            and all(helper_all[index].get("event_id") == rows[index]["event_id"] for index in range(74)),
            "helper result request coverage",
        )
        helper_rows = {
            index: record
            for index, record in helper_all.items()
            if record.get("route") == HELPER_EXECUTOR
        }
        require(helper_rows and set(helper_rows) <= cold_indices, "helper route coverage")
        helper_hash = digest_file(helper_path)

    resources = [{
        "capacity": 8,
        "identity": "rtx-4060ti-cuda0",
        "kind": "gpu",
        "ready": True,
        "resource_id": GPU_RESOURCE,
    }]
    if mode == "shadow":
        resources.extend([
            {
                "capacity": 1,
                "identity": "i9-12900k-gemma-companion",
                "kind": "cpu",
                "ready": True,
                "resource_id": CPU_RESOURCE,
            },
            {
                "capacity": 1,
                "identity": "op15-htp0-gemma-ffn",
                "kind": "npu",
                "ready": True,
                "resource_id": PHONE_RESOURCE,
            },
            {
                "capacity": 1,
                "identity": "op15-functionfs-dmabuf",
                "kind": "transport",
                "ready": True,
                "resource_id": USB_RESOURCE,
            },
        ])

    control_evidence_id = "sha256:" + digest_file(gpu_control_path)
    helper_evidence_id = None if helper_hash is None else "sha256:" + helper_hash
    routes = []
    executors = {}
    workloads = {}
    for row in cold_rows:
        index = row["request_index"]
        workload_id = f"burstgpt-cold-{index}"
        workloads[str(index)] = workload_id
        gpu_route_id = f"r{index}-gemma-cuda-full"
        routes.append(route(
            route_id=gpu_route_id,
            workload_id=workload_id,
            baseline=True,
            value_us=service_us(control_rows[index]),
            evidence_id=control_evidence_id,
        ))
        executors[gpu_route_id] = GPU_EXECUTOR
        if index in helper_rows:
            helper_route_id = f"r{index}-gemma-cpu-op15"
            routes.append(route(
                route_id=helper_route_id,
                workload_id=workload_id,
                baseline=False,
                value_us=service_us(helper_rows[index]),
                evidence_id=helper_evidence_id,
            ))
            executors[helper_route_id] = HELPER_EXECUTOR

    profile = {
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 200_000_000,
            "max_exposed_join_wait_ppm": 50_000,
            "offload_min_finish_saving_us": 1_000,
            "offload_requires_baseline_queue": True,
        },
        "profile_id": f"burstgpt-gpu-overflow-{mode}-4060ti-op15-v1",
        "resources": resources,
        "routes": routes,
        "schema": "s42-general-scheduler-profile-v1",
        "trace_workload_map": {},
    }
    bundle = ProfileBundle.from_json(profile)
    scheduler = UnifiedScheduler((bundle,), mode)
    reservation = scheduler.reserve_external_resource(
        GPU_RESOURCE,
        "protected-hot-and-switch",
        0,
        protected_until_us,
        slots=8,
    )
    decisions = []
    for row in cold_rows:
        decision = scheduler.schedule(scheduler_request(
            row, workloads[str(row["request_index"])], switch_start_us
        ))
        value = decision_to_json(decision)
        value["executor_route"] = executors[decision.route_id]
        value["request_index"] = row["request_index"]
        decisions.append(value)
    helper_indices = sorted(
        decision["request_index"]
        for decision in decisions
        if decision["executor_route"] == HELPER_EXECUTOR
    )

    result = {
        "evidence": {
            "gpu_control_result": {
                "path": str(gpu_control_path),
                "sha256": digest_file(gpu_control_path),
            },
            "helper_result": None if helper_path is None else {
                "path": str(helper_path),
                "sha256": helper_hash,
            },
        },
        "gpu": {
            "protected_reservation": [
                {
                    "resource_id": item.resource_id,
                    "slots": len(item.lanes),
                    "start_us": item.start_us,
                    "token": item.token,
                    "until_us": item.reserved_until_us,
                }
                for item in reservation
            ],
            "protected_until_us": protected_until_us,
            "switch_start_us": switch_start_us,
        },
        "mode": mode,
        "phone": {
            "enabled": mode == "shadow",
            "executor_route": HELPER_EXECUTOR if mode == "shadow" else None,
            "predicted_selected_request_indices": helper_indices,
            "profiled_request_indices": sorted(helper_rows),
            "retire_before_gpu_switch": True,
        },
        "prediction": {
            "decisions": decisions,
            "finish_max_us": max(decision["finish_us"] for decision in decisions),
            "helper_count": len(helper_indices),
            "request_count": len(decisions),
        },
        "scheduler": {
            "policy_mode": mode,
            "profile": profile,
            "request_workloads": workloads,
            "route_executors": executors,
        },
        "schema": SCHEMA,
        "status": "PASS",
        "trace_sha256": trace_hash,
    }
    result["plan_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--gpu-control-result", type=Path, required=True)
    parser.add_argument("--helper-result", type=Path)
    parser.add_argument("--mode", choices=("control", "shadow"), required=True)
    parser.add_argument("--switch-guard-us", type=int, default=1_000_000)
    parser.add_argument("--gpu-ready-guard-us", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(args.output.is_absolute() and not args.output.exists(), "output")
        plan = build_plan(
            args.trace,
            args.gpu_control_result,
            args.helper_result,
            args.mode,
            args.switch_guard_us,
            args.gpu_ready_guard_us,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(plan))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"GPU overflow planning failed: {exc}\n")
    print(json.dumps({
        "helper_count": plan["prediction"]["helper_count"],
        "mode": plan["mode"],
        "output": str(args.output),
        "plan_sha256": plan["plan_sha256"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
