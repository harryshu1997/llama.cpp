#!/usr/bin/env python3
"""Build a request-by-request scheduling audit from a completed physical run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "s42-all-request-scheduling-audit-v1"
SUMMARY_SCHEMA = "s42-all-request-scheduling-audit-summary-v1"
COMBINED_RESULT_SCHEMAS = {
    "s42-full-fp16-llama1b-combined-result-v1",
    "s42-full-fp16-llama1b-combined-result-v2",
}
BASE_RESULT_SCHEMA = "s41-hierarchical-burstgpt-result-v1"
PLAN_SCHEMA = "s42-full-fp16-burstgpt-plan-v2"
BOOTSTRAP_SCHEMA = "s42-fp16-model-device-bootstrap-v1"
F16_MODEL_BY_PHASE = {
    "gemma": "gemma-4-12b-q40-dequant-f16",
    "qwen": "qwen3-14b-q4km-dequant-f16",
}


class AuditError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot parse object: {path}") from error
    require(type(value) is dict, f"object expected: {path}")
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot parse rows: {path}") from error
    require(all(type(row) is dict for row in rows), f"row object expected: {path}")
    return rows


def unique_by(
    rows: list[dict[str, Any]],
    kind: str,
    key: str,
) -> dict[int, dict[str, Any]]:
    result = {
        row[key]: row
        for row in rows
        if row.get("kind") == kind
    }
    require(
        len(result) == sum(row.get("kind") == kind for row in rows),
        f"duplicate {kind} event",
    )
    return result


def grouped_by(
    rows: list[dict[str, Any]],
    kind: str,
    key: str,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") == kind:
            result.setdefault(row[key], []).append(row)
    return result


def decision_attempt(event: dict[str, Any]) -> dict[str, Any]:
    attempt = event.get("attempt", event)
    require(type(attempt) is dict, "runtime decision attempt")
    return attempt


def offset_us(timestamp_ns: int, paid_start_ns: int) -> int:
    return (timestamp_ns - paid_start_ns) // 1000


def mib(value: int) -> float:
    return round(value / 1024**2, 3)


def phase_name(route: str) -> str:
    if route.startswith("qwen_"):
        return "qwen"
    if route.startswith("gemma_"):
        return "gemma"
    raise AuditError(f"unknown F16 route: {route}")


def phase_devices(
    phase: str,
    placement: dict[str, Any],
    phase_plan: dict[str, Any],
) -> dict[str, Any]:
    gpu_layers = placement[f"{phase}_gpu_layers"]
    decision = phase_plan.get("decision")
    arm = None if decision is None else decision.get("arm")
    signals = [] if arm is None else arm.get("signals", [arm])
    return {
        "cpu": {
            "resource_id": "desktop-cpu",
            "role": "host model work outside the GPU and phone FFN slices",
        },
        "gpu": {
            "layers": gpu_layers,
            "resource_id": "cuda0",
        },
        "phone": {
            "execution_mode": phase_plan["execution_mode"],
            "layer_mask": phase_plan["layer_mask"],
            "phone_columns": phase_plan["phone_columns"],
            "resource_id": "op15-htp",
            "session_ids": [signal["session_id"] for signal in signals],
        },
    }


def timing_record(
    row: dict[str, Any],
    arrival_event: dict[str, Any],
    paid_start_ns: int,
) -> dict[str, Any]:
    scheduled_us = offset_us(row["scheduled_arrival_ns"], paid_start_ns)
    completion_us = offset_us(row["completion_ns"], paid_start_ns)
    return {
        "completion_from_arrival_us": completion_us - scheduled_us,
        "completion_offset_us": completion_us,
        "dispatch_from_arrival_us": offset_us(
            row["dispatch_ns"], paid_start_ns
        ) - scheduled_us,
        "dispatch_offset_us": offset_us(row["dispatch_ns"], paid_start_ns),
        "first_token_from_arrival_us": offset_us(
            row["first_token_ns"], paid_start_ns
        ) - scheduled_us,
        "first_token_offset_us": offset_us(
            row["first_token_ns"], paid_start_ns
        ),
        "observed_arrival_offset_us": offset_us(
            arrival_event["t_ns"], paid_start_ns
        ),
        "scheduled_arrival_offset_us": scheduled_us,
        "slo_met": row["completion_ns"]
            <= row["scheduled_arrival_ns"] + row["slo_us"] * 1000,
        "slo_us": row["slo_us"],
    }


def build_large_record(
    row: dict[str, Any],
    combined_index: int,
    arrival_event: dict[str, Any],
    paid_start_ns: int,
    plan: dict[str, Any],
    base_scheduler: dict[str, Any],
) -> dict[str, Any]:
    phase = phase_name(row["route"])
    phase_plan = plan["phases"][phase]
    online_costs = base_scheduler.get("request_level_cost_estimates", {})
    online_receipts = base_scheduler.get("request_level_online_receipts", {})
    online_decisions = base_scheduler.get("request_level_decisions", {})
    online_observations = base_scheduler.get("request_level_observations", {})
    online_overheads = base_scheduler.get("request_level_overheads", {})
    online_receipt = online_receipts.get(str(row["request_index"]))
    if online_receipt is not None:
        decision = row.get("scheduler_decision")
        cost_set = online_costs.get(str(row["request_index"]))
        overhead = online_overheads.get(str(row["request_index"]))
        require(
            type(decision) is dict
            and type(cost_set) is dict
            and cost_set.get("request_id") == row["event_id"]
            and type(cost_set.get("snapshot")) is dict
            and decision == online_decisions.get(str(row["request_index"]))
            and online_observations.get(str(row["request_index"]))
                in {"arrival", "resident_endpoint_ready"}
            and online_receipt.get("selected_route_id") == row["route"]
            and type(overhead) is dict
            and type(overhead.get("total_core_ns")) is int
            and overhead["total_core_ns"] > 0
            and type(overhead.get("total_controller_ns")) is int
            and overhead["total_controller_ns"]
                >= overhead["total_core_ns"],
            "F16 online decision and physical route mismatch",
        )
        family_estimates = [
            {
                "admitted_by_estimator": item["admitted"],
                "energy_upper_uj": item["fleet_energy_upper_uj"],
                "estimator_reason": item["reason"],
                "executor_id": item["executor_id"],
                "family": item["family"],
                "route_id": item["route_id"],
                "service_upper_us": item["service_upper_us"],
            }
            for item in online_receipt["family_estimates"]
        ]
        return {
            "combined_request_index": combined_index,
            "event_id": row["event_id"],
            "execution": {
                "recovery": None,
                "release": row.get("scheduler_release"),
                "route": row["route"],
            },
            "model_id": F16_MODEL_BY_PHASE[phase],
            "request_shape": {
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
            },
            "runtime_scheduling": {
                "causal_input_sha256": online_receipt[
                    "causal_input_sha256"
                ],
                "decision_reason": decision["reason"],
                "decision_source": "causal_request_level_runtime_cost",
                "devices": phase_devices(
                    phase, plan["placement"], phase_plan
                ),
                "per_request_cost_estimates": family_estimates,
                "per_request_overhead_ns": overhead,
                "phase": phase,
                "placement_observation": online_observations[
                    str(row["request_index"])
                ],
                "prefix_sha256": online_receipt["prefix_sha256"],
                "scheduler_invoked_at_arrival": (
                    online_observations[str(row["request_index"])]
                    == "arrival"
                ),
                "scope": "causal_request_level",
                "selected_family": online_receipt["selected_family"],
                "selected_route": row["route"],
                "snapshot": cost_set["snapshot"],
            },
            "schema": AUDIT_SCHEMA,
            "stream": "fp16-burstgpt",
            "stream_request_index": row["request_index"],
            "timing": timing_record(row, arrival_event, paid_start_ns),
            "trace_model_identity": {
                "effective_model_id": row["effective_model_id"],
                "source_model_id": row["source_model_id"],
            },
        }
    runtime_placement = plan["runtime_placement"]
    selected = runtime_placement["selected"]
    require(
        row.get("scheduler_decision") is None
        and row.get("scheduler_release") is None,
        "unexpected per-request scheduler data on F16 request",
    )
    return {
        "combined_request_index": combined_index,
        "event_id": row["event_id"],
        "execution": {
            "recovery": None,
            "release": None,
            "route": row["route"],
        },
        "model_id": F16_MODEL_BY_PHASE[phase],
        "request_shape": {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        },
        "runtime_scheduling": {
            "decision_reason": phase_plan["decision"]["decision_reason"],
            "decision_source": "inherited_runtime_placement",
            "devices": phase_devices(
                phase,
                plan["placement"],
                phase_plan,
            ),
            "per_request_cost_estimates": None,
            "per_request_overhead_ns": None,
            "phase": phase,
            "phase_candidate_id": phase_plan["decision"]["candidate_id"],
            "placement_candidate_id": selected["candidate_id"],
            "placement_decision_sha256": runtime_placement["decision_sha256"],
            "placement_snapshot_id": runtime_placement["snapshot"]["snapshot_id"],
            "scheduler_invoked_at_arrival": False,
            "scope": "runtime_placement_inherited",
            "selected_route": row["route"],
        },
        "schema": AUDIT_SCHEMA,
        "stream": "fp16-burstgpt",
        "stream_request_index": row["request_index"],
        "timing": timing_record(row, arrival_event, paid_start_ns),
        "trace_model_identity": {
            "effective_model_id": row["effective_model_id"],
            "source_model_id": row["source_model_id"],
        },
    }


def merged_route_estimates(
    estimate_set: dict[str, Any],
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    rejected = {
        row["route_id"]: row["reason"]
        for row in decision["rejected"]
    }
    result = []
    for estimate in estimate_set["estimates"]:
        route_id = estimate["route_id"]
        if route_id == decision["route_id"]:
            policy_outcome = "SELECTED"
        else:
            policy_outcome = rejected.get(route_id, "NOT_SELECTED")
        result.append({
            "admitted_by_estimator": estimate["admitted"],
            "energy_upper_uj": estimate["fleet_energy_upper_uj"],
            "energy_uj": estimate["fleet_energy_uj"],
            "estimator_reason": estimate["reason"],
            "executor_id": estimate["executor_id"],
            "policy_outcome": policy_outcome,
            "route_id": route_id,
            "service_upper_us": estimate["service_upper_us"],
            "service_us": estimate["service_us"],
        })
    return result


def build_overlay_record(
    row: dict[str, Any],
    arrival_event: dict[str, Any],
    decision_events: list[dict[str, Any]],
    paid_start_ns: int,
    external_reservations: list[dict[str, Any]],
) -> dict[str, Any]:
    decision = row["scheduler_final_decision"]
    attempts = [decision_attempt(event) for event in decision_events]
    require(
        attempts
        and attempts[-1]["decision"] == row["scheduler_decision"]
        and attempts[-1]["decision"] == decision,
        "overlay decision attempt history mismatch",
    )
    decision_event = attempts[-1]
    estimate_set = decision_event["cost_estimates"]
    require(
        decision_event["decision"]["request_id"] == decision["request_id"]
        and row["scheduler_recovery"] is None,
        "overlay decision or recovery mismatch",
    )
    capacities = {
        item["resource_id"]: {
            "available_bytes": item["available_bytes"],
            "capacity_bytes": item["capacity_bytes"],
            "occupied_bytes": item["occupied_bytes"],
            "reserve_bytes": item["reserve_bytes"],
        }
        for item in estimate_set["snapshot"]["capacities"]
    }
    return {
        "combined_request_index": row["combined_request_index"],
        "event_id": row["event_id"],
        "execution": {
            "recovery": row["scheduler_recovery"],
            "release": row["scheduler_release"],
            "route": row["route"],
        },
        "model_id": row["execution_model_id"],
        "request_shape": {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        },
        "runtime_scheduling": {
            "active_external_reservations": external_reservations,
            "attempt_history": [
                {
                    "attempt_kind": attempt.get("attempt_kind", "arrival"),
                    "causal_input_sha256": (
                        attempt.get("online_placement_receipt") or {}
                    ).get("causal_input_sha256"),
                    "decision": attempt["decision"],
                    "overhead_ns": attempt["overhead_ns"],
                    "prefix_sha256": (
                        attempt.get("online_placement_receipt") or {}
                    ).get("prefix_sha256"),
                    "status": attempt.get("status", "DISPATCH"),
                }
                for attempt in attempts
            ],
            "decision_event_offset_us": offset_us(
                decision_events[-1]["t_ns"], paid_start_ns
            ),
            "decision_reason": decision["reason"],
            "decision_source": "per_request_runtime_cost_and_lease",
            "lease": decision["leases"],
            "per_request_cost_estimates": merged_route_estimates(
                estimate_set, decision
            ),
            "per_request_overhead_ns": decision_event["overhead_ns"],
            "predicted_queue_by_resource_us": decision["queue_by_resource_us"],
            "predicted_queue_us": decision["queue_us"],
            "scheduler_invoked_at_arrival": True,
            "scope": "per_request_runtime_cost_and_lease",
            "selected_route": decision["route_id"],
            "snapshot": {
                "capacities": capacities,
                "captured_at_us": estimate_set["snapshot"]["captured_at_us"],
                "snapshot_id": estimate_set["snapshot"]["snapshot_id"],
                "valid_until_us": estimate_set["snapshot"]["valid_until_us"],
            },
        },
        "schema": AUDIT_SCHEMA,
        "stream": "llama1b-overlay",
        "stream_request_index": row["overlay_request_index"],
        "timing": timing_record(row, arrival_event, paid_start_ns),
    }


def csv_value(record: dict[str, Any]) -> dict[str, object]:
    scheduling = record["runtime_scheduling"]
    timing = record["timing"]
    estimates = scheduling.get("per_request_cost_estimates") or []
    estimate_by_route = {row["route_id"]: row for row in estimates}
    snapshot = scheduling.get("snapshot", {})
    raw_capacities = snapshot.get("capacities", {})
    capacities = (
        {
            capacity["resource_id"]: capacity
            for capacity in raw_capacities
        }
        if type(raw_capacities) is list else raw_capacities
    )
    require(type(capacities) is dict, "runtime capacity snapshot")
    overhead = scheduling.get("per_request_overhead_ns") or {}
    lease = scheduling.get("lease") or []
    attempts = scheduling.get("attempt_history") or []
    if record["stream"] == "fp16-burstgpt":
        devices = scheduling["devices"]
        device_binding = (
            f"cuda0:{devices['gpu']['layers']}layers+desktop-cpu+"
            f"op15:{','.join(devices['phone']['session_ids'])}"
        )
    else:
        device_binding = scheduling["selected_route"]

    def route_field(route_id: str, field: str) -> object:
        return estimate_by_route.get(route_id, {}).get(field, "")

    return {
        "combined_index": record["combined_request_index"],
        "stream": record["stream"],
        "stream_index": record["stream_request_index"],
        "event_id": record["event_id"],
        "model_id": record["model_id"],
        "input_tokens": record["request_shape"]["input_tokens"],
        "output_tokens": record["request_shape"]["output_tokens"],
        "arrival_s": timing["scheduled_arrival_offset_us"] / 1e6,
        "observed_arrival_s": timing["observed_arrival_offset_us"] / 1e6,
        "scheduling_scope": scheduling["scope"],
        "scheduler_invoked_at_arrival": scheduling[
            "scheduler_invoked_at_arrival"
        ],
        "snapshot_id": snapshot.get(
            "snapshot_id", scheduling.get("placement_snapshot_id", "")
        ),
        "gpu_available_mib": mib(
            capacities.get("cuda0-vram", {}).get("available_bytes", 0)
        ) if capacities else "",
        "host_available_mib": mib(
            capacities.get("host-ram", {}).get("available_bytes", 0)
        ) if capacities else "",
        "phone_available_mib": mib(
            capacities.get("op15-ram", {}).get("available_bytes", 0)
        ) if capacities else "",
        "cpu_estimator": route_field("desktop-cpu", "estimator_reason"),
        "cpu_policy": route_field("desktop-cpu", "policy_outcome"),
        "cuda_estimator": route_field("desktop-cuda", "estimator_reason"),
        "cuda_policy": route_field("desktop-cuda", "policy_outcome"),
        "phone_estimator": route_field("phone-adreno", "estimator_reason"),
        "phone_policy": route_field("phone-adreno", "policy_outcome"),
        "selected_route": scheduling["selected_route"],
        "device_binding": device_binding,
        "decision_reason": scheduling["decision_reason"],
        "decision_attempts": len(attempts) if attempts else 1,
        "replan_count": max(0, len(attempts) - 1),
        "lease_token": lease[0]["token"] if lease else "",
        "predicted_queue_ms": scheduling.get("predicted_queue_us", "") / 1000
            if scheduling.get("predicted_queue_us") is not None else "",
        "dispatch_from_arrival_s": timing["dispatch_from_arrival_us"] / 1e6,
        "first_token_from_arrival_s": timing[
            "first_token_from_arrival_us"
        ] / 1e6,
        "completion_from_arrival_s": timing[
            "completion_from_arrival_us"
        ] / 1e6,
        "slo_s": timing["slo_us"] / 1e6,
        "slo_met": timing["slo_met"],
        "core_overhead_ms": overhead.get("total_core_ns", "") / 1e6
            if overhead else "",
        "controller_overhead_ms": overhead.get("total_controller_ns", "") / 1e6
            if overhead else "",
        "recovery": "none" if record["execution"]["recovery"] is None else "yes",
        "release_status": (
            record["execution"]["release"] or {}
        ).get("status", ""),
    }


def markdown(records: list[dict[str, Any]]) -> bytes:
    fp16_online = any(
        record["stream"] == "fp16-burstgpt"
        and record["runtime_scheduling"]["scope"]
            == "causal_request_level"
        for record in records
    )
    lines = [
        "# All-request runtime scheduling audit",
        "",
        (
            "All 84 requests have causal request-level scheduling receipts. "
            "The model/device bootstrap accepts no future request sequence."
            if fp16_online else
            "The 74 FP16 requests inherit the placement selected from a live "
            "startup snapshot. The ten Llama 1B requests invoke the cost "
            "estimator and lease policy independently at arrival."
        ),
        "",
        "| # | Request | Arrival s | Model | Scheduling scope | Selected binding | "
        "Candidate result | Decision cost | Completion s | SLO |",
        "| ---: | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        row = csv_value(record)
        if record["runtime_scheduling"].get(
            "per_request_cost_estimates"
        ) is not None:
            if record["stream"] == "fp16-burstgpt":
                candidate_result = "; ".join(
                    f"{item['family']} {item['estimator_reason']}"
                    for item in record["runtime_scheduling"][
                        "per_request_cost_estimates"
                    ]
                )
                cost = f"{row['controller_overhead_ms']:.3f} ms"
            else:
                candidate_result = (
                    f"CPU {row['cpu_policy']}; CUDA {row['cuda_policy']}; "
                    f"phone {row['phone_policy']}"
                )
                cost = f"{row['controller_overhead_ms']:.3f} ms"
        else:
            candidate_result = (
                f"inherited {record['runtime_scheduling']['phase_candidate_id']}"
            )
            cost = "n/a"
        request = f"{record['stream']}:{record['stream_request_index']}"
        slo = "met" if row["slo_met"] else "miss"
        lines.append(
            f"| {row['combined_index']} | {request} | {row['arrival_s']:.3f} | "
            f"{row['model_id']} | {row['scheduling_scope']} | "
            f"{row['device_binding']} | {candidate_result} | {cost} | "
            f"{row['completion_from_arrival_s']:.3f} | {slo} |"
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def artifact_paths(run_dir: Path) -> dict[str, Path]:
    flat = {
        "base_events": run_dir / "BASE_EVENTS.jsonl",
        "base_result": run_dir / "BASE_RESULT.json",
        "combined_result": run_dir / "RESULT.json",
        "execution_plan": run_dir / "EXECUTION_PLAN.json",
        "overlay_events": run_dir / "OVERLAY_EVENTS.jsonl",
    }
    if all(path.is_file() for path in flat.values()):
        return flat
    launcher = {
        "base_events": run_dir / "base/events.jsonl",
        "base_result": run_dir / "base/RESULT.json",
        "combined_result": run_dir / "combined/RESULT.json",
        "execution_plan": run_dir / "capture/EXECUTION_PLAN.json",
        "overlay_events": run_dir / "combined/events.jsonl",
    }
    require(
        all(path.is_file() for path in launcher.values()),
        "run directory has neither the flat nor launcher artifact layout",
    )
    return launcher


def build(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = artifact_paths(run_dir)
    base = load_object(paths["base_result"])
    combined = load_object(paths["combined_result"])
    plan = load_object(paths["execution_plan"])
    base_events = load_rows(paths["base_events"])
    overlay_events = load_rows(paths["overlay_events"])
    require(
        base.get("schema") == BASE_RESULT_SCHEMA
        and base.get("status") == "PASS"
        and len(base.get("request_results", [])) == 74,
        "base result identity",
    )
    require(
        combined.get("schema") in COMBINED_RESULT_SCHEMAS
        and combined.get("status") == "PASS"
        and combined.get("metrics", {}).get("completed") == 84
        and len(combined.get("request_results", [])) == 10
        and combined.get("scheduler_runtime", {}).get("enabled") is True,
        "combined result identity",
    )
    require(
        plan.get("schema") in {PLAN_SCHEMA, BOOTSTRAP_SCHEMA}
        and plan.get("status") == "PASS"
        and (
            plan["schema"] == BOOTSTRAP_SCHEMA
            and plan.get("decision_basis", {}).get("future_request_data")
                == "not_accepted"
            or plan["schema"] == PLAN_SCHEMA
            and plan["runtime_placement"]["selected"]["candidate_id"]
                == "fp16-server-gpu-cpu-op15-switch-v1"
        ),
        "execution plan identity",
    )
    paid_start_ns = combined["paid_start_ns"]
    require(base["paid_start_ns"] == paid_start_ns, "paid start mismatch")
    base_arrivals = unique_by(
        base_events, "request_arrival", "request_index"
    )
    overlay_arrivals = unique_by(
        overlay_events, "overlay_request_arrival", "overlay_request_index"
    )
    overlay_decisions = grouped_by(
        overlay_events, "runtime_scheduler_decision", "overlay_request_index"
    )
    require(
        set(base_arrivals) == set(range(74))
        and set(overlay_arrivals) == set(range(10))
        and set(overlay_decisions) == set(range(10)),
        "event coverage",
    )
    overlay_positions = {
        row["combined_request_index"]
        for row in combined["request_results"]
    }
    require(
        len(overlay_positions) == 10
        and all(0 <= index < 84 for index in overlay_positions),
        "combined overlay positions",
    )
    base_positions = [
        index for index in range(84) if index not in overlay_positions
    ]
    records = []
    base_scheduler = base.get("fp16_resident_scheduler", {})
    for row, combined_index in zip(
        sorted(base["request_results"], key=lambda item: item["request_index"]),
        base_positions,
        strict=True,
    ):
        records.append(build_large_record(
            row,
            combined_index,
            base_arrivals[row["request_index"]],
            paid_start_ns,
            plan,
            base_scheduler,
        ))
    external_reservations = combined["scheduler_runtime"][
        "external_reservations"
    ]
    for row in combined["request_results"]:
        index = row["overlay_request_index"]
        records.append(build_overlay_record(
            row,
            overlay_arrivals[index],
            overlay_decisions[index],
            paid_start_ns,
            external_reservations,
        ))
    records.sort(key=lambda row: row["combined_request_index"])
    require(
        [row["combined_request_index"] for row in records] == list(range(84))
        and sum(
            row["request_shape"]["input_tokens"] for row in records
        ) == 38_948
        and sum(
            row["request_shape"]["output_tokens"] for row in records
        ) == 13_132,
        "combined request conservation",
    )
    scope_counts = Counter(
        row["runtime_scheduling"]["scope"] for row in records
    )
    route_counts = Counter(row["execution"]["route"] for row in records)
    qualification_path = run_dir / "capture/QUALIFICATION.json"
    qualification = (
        load_object(qualification_path)
        if qualification_path.is_file() else None
    )
    summary = {
        "audit": {
            "per_request_runtime_scheduler_count": sum(
                row["runtime_scheduling"].get(
                    "per_request_cost_estimates"
                ) is not None
                for row in records
            ),
            "request_count": len(records),
            "route_counts": dict(sorted(route_counts.items())),
            "scope_counts": dict(sorted(scope_counts.items())),
        },
        "input_sha256": {
            name: digest(path) for name, path in sorted(paths.items())
        },
        "interpretation": {
            "audit_status_scope": (
                "artifact consistency only; performance qualification is "
                "reported separately"
            ),
            "fp16_requests": (
                "causal request-level costs and physical route binding"
                if plan["schema"] == BOOTSTRAP_SCHEMA
                else "runtime placement selected once, then inherited per request"
            ),
            "llama1b_requests": (
                "runtime costs, route feasibility, and lease committed at arrival"
            ),
            "not_yet_implemented": (
                "physical executors for unavailable six-family routes"
                if plan["schema"] == BOOTSTRAP_SCHEMA
                else "per-request cost re-estimation for the 74 FP16 requests"
            ),
        },
        "physical_qualification": (
            {"status": "NOT_PROVIDED"}
            if qualification is None else {
                "error": qualification.get("error"),
                "path": str(qualification_path),
                "sha256": digest(qualification_path),
                "status": qualification.get("status"),
            }
        ),
        "placement": (
            {
                "bootstrap_sha256": plan["plan_sha256"],
                "candidate_id": "model-device-runtime-bootstrap",
                "snapshot_id": plan["runtime_snapshot"]["snapshot_id"],
            }
            if plan["schema"] == BOOTSTRAP_SCHEMA else
            {
                "candidate_id": plan["runtime_placement"]["selected"][
                    "candidate_id"
                ],
                "decision_sha256": plan["runtime_placement"][
                    "decision_sha256"
                ],
                "snapshot_id": plan["runtime_placement"]["snapshot"][
                    "snapshot_id"
                ],
            }
        ),
        "schema": SUMMARY_SCHEMA,
        "status": "PASS",
    }
    return records, summary


def write_outputs(
    run_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    jsonl = b"".join(canonical(record) for record in records)
    csv_rows = [csv_value(record) for record in records]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    outputs = {
        run_dir / "ALL_REQUEST_SCHEDULING_LOG.csv": buffer.getvalue().encode(
            "ascii"
        ),
        run_dir / "ALL_REQUEST_SCHEDULING_LOG.jsonl": jsonl,
        run_dir / "ALL_REQUEST_SCHEDULING_LOG.md": markdown(records),
        run_dir / "ALL_REQUEST_SCHEDULING_SUMMARY.json": canonical(summary),
    }
    for path, content in outputs.items():
        path.write_bytes(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    require(args.run_dir.is_dir(), "run directory does not exist")
    records, summary = build(args.run_dir)
    write_outputs(args.run_dir, records, summary)
    print(json.dumps({
        "output_dir": str(args.run_dir),
        "per_request_runtime_scheduler_count": summary["audit"][
            "per_request_runtime_scheduler_count"
        ],
        "physical_qualification": summary["physical_qualification"][
            "status"
        ],
        "request_count": summary["audit"]["request_count"],
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
