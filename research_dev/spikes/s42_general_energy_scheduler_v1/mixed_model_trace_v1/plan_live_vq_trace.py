#!/usr/bin/env python3
"""Build a live virtual-queue plan for the six-model mixed trace."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
from math import lcm
from pathlib import Path
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT)]

from research_dev.scheduler import (  # noqa: E402
    ProfileBundle,
    RESIDENCY_PROBLEM_SCHEMA,
    Request,
    UnifiedScheduler,
    decision_to_json,
    residency_sequence_to_json,
    solve_residency_problem,
)
from verify_mixed_trace import validate as validate_trace  # noqa: E402


SCHEMA = "s42-six-model-live-vq-plan-v3"
RESULT_SCHEMA = "s42-six-model-physical-result-v1"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
QWEN14 = "qwen3-14b-q4_k_m"
QWEN8 = "qwen3-8b-q8_0"
GEMMA12 = "gemma-4-12b-it-q4_0"
QWEN06 = "qwen3-0.6b-q8_0"
GEMMA_E2B = "gemma-4-e2b-it-q8_0-vlm"
UNIFIED_ROUTE = "unified-scheduler"
PHONE_ROUTE = f"{LLAMA1}-phone-adreno"


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutorBinding:
    model_id: str
    executor_route: str
    backend: str
    total_slots: int
    resource_id: str
    resource_slots: int
    resident_bytes: int
    residency_id: str
    preloaded: bool


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


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    require(all(type(row) is dict for row in rows), "trace rows")
    return rows


def trace_model_ids(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row["execution_model_id"] for row in rows))


def manifest_model_bytes(manifest: dict[str, Any]) -> dict[str, int]:
    inventory = manifest.get("model_inventory")
    require(type(inventory) is dict and inventory, "manifest model inventory")
    result: dict[str, int] = {}
    for model_id, raw in inventory.items():
        require(type(raw) is dict, "manifest model entry")
        artifact_bytes = raw.get("artifact_bytes")
        projector_bytes = raw.get("projector_bytes", 0)
        require(
            type(artifact_bytes) is int
            and artifact_bytes > 0
            and type(projector_bytes) is int
            and projector_bytes >= 0,
            "manifest model bytes",
        )
        result[model_id] = artifact_bytes + projector_bytes
    return result


def executor_bindings(
    rows: list[dict[str, Any]],
    control: dict[str, Any],
    model_bytes: Mapping[str, int],
) -> dict[str, ExecutorBinding]:
    model_ids = trace_model_ids(rows)
    records: dict[str, dict[str, Any]] = {}
    for record in control.get("server_records", []):
        model_id = record.get("model_id")
        if (
            model_id not in model_ids
            or record.get("stage") not in {"initial", "promotion"}
        ):
            continue
        require(model_id not in records, "duplicate control server binding")
        records[model_id] = record
    require(set(records) == set(model_ids), "control server binding coverage")
    control_routes: dict[str, set[str]] = {
        model_id: {
            record["route"]
            for record in control["request_results"]
            if record["execution_model_id"] == model_id
        }
        for model_id in model_ids
    }
    require(
        all(len(routes) == 1 for routes in control_routes.values()),
        "control executor route coverage",
    )

    raw: dict[str, tuple[str, str, int, bool]] = {}
    for model_id in model_ids:
        record = records[model_id]
        backend = record.get("backend")
        props = record.get("props")
        slots = None if type(props) is not dict else props.get("total_slots")
        executor = next(iter(control_routes[model_id]))
        require(
            backend in {"cpu", "cuda"}
            and type(slots) is int
            and slots > 0
            and executor == f"{model_id}-{backend}",
            "control executor binding",
        )
        require(
            model_id in model_bytes
            and type(model_bytes[model_id]) is int
            and model_bytes[model_id] > 0,
            "model residency bytes",
        )
        raw[model_id] = (
            executor,
            backend,
            slots,
            record["stage"] == "initial",
        )

    gpu_slots = [value[2] for value in raw.values() if value[1] == "cuda"]
    require(gpu_slots, "control CUDA binding")
    cuda_capacity = lcm(*gpu_slots)
    result: dict[str, ExecutorBinding] = {}
    for model_id, (executor, backend, slots, preloaded) in raw.items():
        resource_id = "cuda0" if backend == "cuda" else f"cpu:{model_id}"
        result[model_id] = ExecutorBinding(
            model_id=model_id,
            executor_route=executor,
            backend=backend,
            total_slots=slots,
            resource_id=(
                "desktop-cpu" if model_id == LLAMA1 else resource_id
            ),
            resource_slots=(cuda_capacity // slots if backend == "cuda" else 1),
            resident_bytes=model_bytes[model_id],
            residency_id=f"weights:{model_id}",
            preloaded=preloaded,
        )
    return result


def llama_executor(
    route_id: str, desktop_executor: str
) -> str:
    if route_id.endswith("-desktop-cpu"):
        return desktop_executor
    if route_id.endswith("-phone-adreno"):
        return PHONE_ROUTE
    raise PlanError(f"unknown generated route: {route_id}")


def request_workload_id(index: int) -> str:
    return f"mixed-request-{index}"


def request_from_row(row: dict[str, Any]) -> Request:
    index = row["mixed_request_index"]
    return Request(
        request_id=f"mixed-{index}",
        workload_id=request_workload_id(index),
        arrival_us=row["arrival_us"],
        deadline_us=row["arrival_us"] + row["slo_us"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        quality_requirement="bounded_numeric",
    )


def scheduler_routes(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["mixed_request_index"]): UNIFIED_ROUTE for row in rows
    }


def measured_service_us(record: dict[str, Any]) -> int:
    values = (record.get("prompt_ms"), record.get("predicted_ms"))
    require(
        all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for value in values
        ),
        "control request service measurements",
    )
    service_us = round(sum(values) * 1000)
    require(service_us > 0, "positive control request service")
    return service_us


def fixed_route(
    row: dict[str, Any],
    record: dict[str, Any],
    evidence_id: str,
    binding: ExecutorBinding,
) -> tuple[dict[str, Any], str]:
    index = row["mixed_request_index"]
    model_id = row["execution_model_id"]
    require(binding.model_id == model_id, "fixed route binding")
    service_us = measured_service_us(record)
    route_id = f"q{index}-{binding.executor_route}"
    return {
        "baseline": True,
        "energy": {
            "cost_uj": None,
            "lower_error_ppm": 0,
            "status": "unknown",
            "upper_error_ppm": 0,
        },
        "evidence_ids": [evidence_id],
        "granularity": "task",
        "latency": {
            "cost_us": {
                "fixed": service_us,
                "input_token": 0,
                "kind": "affine_tokens_v1",
                "output_token": 0,
            },
            "measured": True,
            "sample_count": 1,
            "ucb_add_us": max(250_000, service_us // 5),
        },
        "overlap": {"status": "not_applicable"},
        "placement_verified": True,
        "quality_class": "bounded_numeric",
        "resident": True,
        "resource_leases": [],
        "resource_slots": {binding.resource_id: binding.resource_slots},
        "route_id": route_id,
        "server_busy_ppm": 1_000_000,
        "server_memory_bytes": binding.resident_bytes,
        "workload_id": request_workload_id(index),
    }, binding.executor_route


def make_runtime_profile(
    rows: list[dict[str, Any]],
    control: dict[str, Any],
    source_profile: dict[str, Any],
    control_hash: str,
    bindings: Mapping[str, ExecutorBinding],
    mode: str = "adaptive",
) -> tuple[dict[str, Any], dict[str, str]]:
    require(mode in {"control", "adaptive"}, "profile mode")
    source_routes = {
        route["route_id"]: route for route in source_profile["routes"]
    }
    require(
        set(source_routes) == {
            "desktop-cpu", "desktop-cuda", "phone-adreno"
        },
        "whole-task route set",
    )
    records = {
        row["mixed_request_index"]: row
        for row in control["request_results"]
    }
    require(
        len(records) == len(control["request_results"])
        and set(records) == {
            row["mixed_request_index"] for row in rows
        },
        "control request coverage",
    )

    resources_by_id = {
        resource["resource_id"]: copy.deepcopy(resource)
        for resource in source_profile["resources"]
    }
    require(
        set(resources_by_id)
        == {"desktop-cpu", "cuda0", "op15-adreno", "usb-token-rpc"},
        "whole-task resource set",
    )
    require(set(bindings) == set(trace_model_ids(rows)), "executor bindings")
    resources_by_id["desktop-cpu"]["capacity"] = bindings[
        LLAMA1
    ].total_slots
    cuda_capacities = {
        binding.total_slots * binding.resource_slots
        for binding in bindings.values()
        if binding.backend == "cuda"
    }
    require(len(cuda_capacities) == 1, "shared CUDA lane capacity")
    resources_by_id["cuda0"]["capacity"] = next(iter(cuda_capacities))
    for binding in bindings.values():
        if binding.backend != "cpu" or binding.model_id == LLAMA1:
            continue
        resources_by_id[binding.resource_id] = {
            "capacity": binding.total_slots,
            "identity": f"desktop-cpu-worker:{binding.model_id}",
            "kind": "cpu",
            "ready": True,
            "resource_id": binding.resource_id,
        }
    if mode == "control":
        resources_by_id.pop("op15-adreno")
        resources_by_id.pop("usb-token-rpc")
    resources = [
        resources_by_id[resource_id]
        for resource_id in sorted(resources_by_id)
    ]

    generated_routes: list[dict[str, Any]] = []
    executors: dict[str, str] = {}
    evidence_id = "sha256:" + control_hash
    source_ids = (
        ("desktop-cpu",)
        if mode == "control"
        else ("desktop-cpu", "phone-adreno")
    )
    for row in rows:
        index = row["mixed_request_index"]
        workload_id = request_workload_id(index)
        model_id = row["execution_model_id"]
        if model_id != LLAMA1:
            generated, executor = fixed_route(
                row, records[index], evidence_id, bindings[model_id]
            )
            generated_routes.append(generated)
            executors[generated["route_id"]] = executor
            continue
        for source_id in source_ids:
            generated = copy.deepcopy(source_routes[source_id])
            generated["route_id"] = f"q{index}-{source_id}"
            generated["workload_id"] = workload_id
            generated["baseline"] = source_id == "desktop-cpu"
            generated["evidence_ids"] = sorted(set([
                *generated.get("evidence_ids", []), evidence_id,
            ]))
            if source_id == "desktop-cpu":
                record = records[index]
                service_us = measured_service_us(record)
                generated["latency"] = {
                    "cost_us": {
                        "fixed": service_us,
                        "input_token": 0,
                        "kind": "affine_tokens_v1",
                        "output_token": 0,
                    },
                    "measured": True,
                    "sample_count": 1,
                    "ucb_add_us": max(250_000, service_us // 5),
                }
            generated_routes.append(generated)
            executors[generated["route_id"]] = llama_executor(
                generated["route_id"], bindings[LLAMA1].executor_route
            )

    profile = {
        "profile_id": f"s42-six-model-live-vq-{mode}-4060ti-op15-v3",
        "policy": {
            "energy_saving_ppm": 50_000,
            "latency_limit_ppm": 200_000_000,
            "max_exposed_join_wait_ppm": 50_000,
            "offload_min_finish_saving_us": 1_000,
            "offload_requires_baseline_queue": True,
        },
        "resources": resources,
        "routes": generated_routes,
        "schema": "s42-general-scheduler-profile-v1",
        "trace_workload_map": {},
    }
    ProfileBundle.from_json(profile)
    return profile, executors


def simulate(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    executors: dict[str, str],
    residency_plan: dict[str, Any],
    mode: str = "adaptive",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scheduler = UnifiedScheduler(
        (ProfileBundle.from_json(profile),), mode
    )
    decisions: list[dict[str, Any]] = []

    def schedule(row: dict[str, Any], runtime_now_us: int | None = None) -> None:
        decision = scheduler.schedule(
            request_from_row(row), runtime_now_us=runtime_now_us
        )
        value = decision_to_json(decision)
        value["executor_route"] = executors[decision.route_id]
        value["mixed_request_index"] = row["mixed_request_index"]
        decisions.append(value)

    model_order = residency_plan["model_order"]
    require(type(model_order) is list and model_order, "GPU model order")
    deferred = set(model_order[1:])
    held: dict[str, list[dict[str, Any]]] = {
        model_id: [] for model_id in deferred
    }
    for row in rows:
        model_id = row["execution_model_id"]
        if model_id in held:
            held[model_id].append(row)
        else:
            schedule(row)
    phases = {
        phase["model_id"]: phase for phase in residency_plan["phases"]
    }
    require(set(phases) == set(model_order), "GPU residency phase coverage")
    for model_id in model_order[1:]:
        ready_us = phases[model_id]["ready_us"]
        for row in held[model_id]:
            schedule(row, ready_us)

    decisions.sort(key=lambda row: row["mixed_request_index"])
    counts = Counter(row["executor_route"] for row in decisions)
    known_energy = [
        row["energy_uj"] for row in decisions if row["energy_uj"] is not None
    ]
    deadlines = {
        row["mixed_request_index"]: row["arrival_us"] + row["slo_us"]
        for row in rows
    }
    return decisions, {
        "conservative_slo_met": sum(
            row["finish_upper_us"] <= deadlines[row["mixed_request_index"]]
            for row in decisions
        ),
        "decision_count": len(decisions),
        "energy_request_count": len(known_energy),
        "energy_uj": sum(known_energy),
        "energy_unknown_request_count": len(decisions) - len(known_energy),
        "finish_max_us": max(row["finish_us"] for row in decisions),
        "route_counts": dict(sorted(counts.items())),
    }


def plan_gpu_residency(
    rows: list[dict[str, Any]],
    control: dict[str, Any],
    bindings: Mapping[str, ExecutorBinding],
    evidence_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu_bindings = {
        model_id: binding
        for model_id, binding in bindings.items()
        if binding.backend == "cuda"
    }
    initial = [
        binding for binding in gpu_bindings.values() if binding.preloaded
    ]
    require(len(initial) == 1, "one preloaded CUDA residency")
    initial_binding = initial[0]
    request_rows = [
        row for row in rows if row["execution_model_id"] in gpu_bindings
    ]
    control_rows = {
        record["mixed_request_index"]: record
        for record in control["request_results"]
        if record["execution_model_id"] in gpu_bindings
    }
    require(
        set(control_rows)
        == {row["mixed_request_index"] for row in request_rows},
        "CUDA service coverage",
    )
    promotions = [
        record
        for record in control["server_records"]
        if record.get("stage") == "promotion"
        and record.get("model_id") in gpu_bindings
    ]
    require(
        {record["model_id"] for record in promotions}
        == set(gpu_bindings) - {initial_binding.model_id},
        "CUDA transition coverage",
    )
    transitions: list[dict[str, Any]] = []
    prior = initial_binding
    for record in promotions:
        target = gpu_bindings[record["model_id"]]
        load_ms = record.get("load_ms")
        warm_ms = record.get("warm_ms")
        require(
            all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and value >= 0
                for value in (load_ms, warm_ms)
            ),
            "measured CUDA transition latency",
        )
        transitions.append({
            "energy_uj": None,
            "evidence_ids": [evidence_id],
            "latency_us": max(1, round((load_ms + warm_ms) * 1000)),
            "measured": True,
            "source_residency_id": prior.residency_id,
            "target_residency_id": target.residency_id,
        })
        prior = target
    problem = {
        "initial_ready_us": 0,
        "initial_residency_id": initial_binding.residency_id,
        "requests": [
            {
                "arrival_us": row["arrival_us"],
                "deadline_us": row["arrival_us"] + row["slo_us"],
                "model_id": row["execution_model_id"],
                "request_id": f"mixed-{row['mixed_request_index']}",
            }
            for row in request_rows
        ],
        "resource": {
            "capacity_bytes": max(
                binding.resident_bytes for binding in gpu_bindings.values()
            ),
            "resource_id": "cuda0-residency",
        },
        "routes": [
            {
                "energy_uj": None,
                "model_id": model_id,
                "preloaded": binding.preloaded,
                "resident_bytes": binding.resident_bytes,
                "residency_id": binding.residency_id,
                "route_id": binding.executor_route,
                "service_us": {
                    f"mixed-{row['mixed_request_index']}": measured_service_us(
                        control_rows[row["mixed_request_index"]]
                    )
                    for row in request_rows
                    if row["execution_model_id"] == model_id
                },
                "slots": binding.total_slots,
            }
            for model_id, binding in gpu_bindings.items()
        ],
        "schema": RESIDENCY_PROBLEM_SCHEMA,
        "switches": transitions,
    }
    return problem, residency_sequence_to_json(
        solve_residency_problem(problem)
    )


def build_plan(
    requests_path: Path,
    manifest_path: Path,
    control_path: Path,
    whole_task_profile_path: Path,
    mode: str,
) -> dict[str, Any]:
    require(mode in {"control", "adaptive"}, "mode")
    validate_trace(requests_path, manifest_path)
    rows = read_rows(requests_path)
    control = read_object(control_path)
    trace_hash = digest_file(requests_path)
    control_hash = digest_file(control_path)
    require(
        control.get("schema") == RESULT_SCHEMA
        and control.get("status") == "PASS"
        and control.get("trace_sha256") == trace_hash,
        "control result identity",
    )
    require(
        sorted(row["mixed_request_index"] for row in control["request_results"])
        == list(range(len(rows))),
        "control request conservation",
    )
    source_profile = read_object(whole_task_profile_path)
    source_profile.pop("profile_hash", None)
    ProfileBundle.from_json(source_profile)
    manifest = read_object(manifest_path)
    bindings = executor_bindings(
        rows, control, manifest_model_bytes(manifest)
    )
    profile, executors = make_runtime_profile(
        rows, control, source_profile, control_hash, bindings, mode
    )
    paid_start_ns = control["paid_start_ns"]
    control_gpu_finish_us = max(
        (record["completion_ns"] - paid_start_ns) // 1000
        for record in control["request_results"]
        if record["route"].endswith("-cuda")
    )
    residency_problem, residency_plan = plan_gpu_residency(
        rows, control, bindings, "sha256:" + control_hash
    )
    predicted, prediction = simulate(
        rows, profile, executors, residency_plan, mode
    )
    bundle = ProfileBundle.from_json(profile)
    routes_by_workload = {
        route.workload_id: [
            candidate
            for candidate in bundle.routes
            if candidate.workload_id == route.workload_id
        ]
        for route in bundle.routes
    }
    control_cpu_energy_uj = 0
    for row in rows:
        if row["execution_model_id"] != LLAMA1:
            continue
        request = request_from_row(row)
        cpu_route = next(
            route
            for route in routes_by_workload[request.workload_id]
            if executors[route.route_id] == f"{LLAMA1}-cpu"
        )
        service_us = cpu_route.latency.predict_us(request)
        energy_uj = cpu_route.energy.predict_uj(request, service_us)
        require(energy_uj is not None, "control CPU energy")
        control_cpu_energy_uj += energy_uj
    scheduled_energy_uj = prediction["energy_uj"]
    control_llama = [
        row for row in control["request_results"]
        if row["execution_model_id"] == LLAMA1
    ]
    require(
        prediction["decision_count"] == len(rows)
        and prediction["energy_request_count"] == len(control_llama),
        "prediction accounting coverage",
    )
    prediction["control_cpu_energy_uj"] = control_cpu_energy_uj
    prediction["route_energy_saving_ppm"] = round(
        (control_cpu_energy_uj - scheduled_energy_uj)
        * 1_000_000
        / control_cpu_energy_uj
    )
    selected_phone = sorted(
        row["mixed_request_index"]
        for row in predicted
        if row["executor_route"] == PHONE_ROUTE
    )
    result = {
        "evidence": {
            "control_result": {
                "path": str(control_path),
                "sha256": control_hash,
            },
            "whole_task_profile": {
                "path": str(whole_task_profile_path),
                "sha256": digest_file(whole_task_profile_path),
            },
        },
        "gpu": {
            "control_finish_us": control_gpu_finish_us,
            "residency_plan": residency_plan,
            "residency_problem": residency_problem,
            "sequence": residency_plan["model_order"],
        },
        "mode": mode,
        "phone": {
            "model_id": LLAMA1 if mode == "adaptive" else None,
            "preloaded_before_paid_start": mode == "adaptive",
            "predicted_selected_mixed_request_indices": (
                selected_phone if mode == "adaptive" else []
            ),
            "route_id": PHONE_ROUTE if mode == "adaptive" else None,
            "route_kind": "whole-task-adreno" if mode == "adaptive" else "none",
            "scheduler_effective_lanes": 1 if mode == "adaptive" else 0,
        },
        "prediction": {
            **prediction,
            "control_llama_finish_max_us": max(
                (row["completion_ns"] - paid_start_ns) // 1000
                for row in control_llama
            ),
            "control_llama_slo_met": sum(
                row["slo_met"] for row in control_llama
            ),
            "decisions": predicted,
            "scope": (
                "all requests scheduled; route energy is available only for "
                "Llama CPU and phone placements"
            ),
        },
        "request_routes": scheduler_routes(rows),
        "scheduler": {
            "alternative_model_ids": [LLAMA1],
            "deferred_until_ready_model_ids": residency_plan[
                "model_order"
            ][1:],
            "enabled": True,
            "policy_mode": mode,
            "profile": profile,
            "route_executors": executors,
            "scheduled_model_ids": list(trace_model_ids(rows)),
            "single_route_model_ids": [
                model_id
                for model_id in trace_model_ids(rows)
                if model_id != LLAMA1
            ],
        },
        "schema": SCHEMA,
        "status": "PASS",
        "trace_sha256": trace_hash,
    }
    result["plan_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--whole-task-profile", type=Path, required=True)
    parser.add_argument("--mode", choices=("control", "adaptive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.is_absolute() and not args.output.exists(), "output")
    plan = build_plan(
        args.requests,
        args.manifest,
        args.control_result,
        args.whole_task_profile,
        args.mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(plan))
    print(json.dumps({
        "mode": args.mode,
        "output": str(args.output),
        "predicted_routes": plan["prediction"]["route_counts"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
