#!/usr/bin/env python3
"""Validate physical route execution and runtime scheduler receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    ProfileBundle,
    canonical_sha256,
)
from research_dev.scheduler._internal.decision_log import (  # noqa: E402
    DecisionLogError,
    RuntimeDecisionLog,
)


RESULT_SCHEMA = "s42-full-fp16-llama1b-combined-result-v2"
QUALIFICATION_SCHEMA = "s42-fp16-llama1b-physical-qualification-v1"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
SPLIT_ROUTE = "cpu-phone-ffn-split"
FP16_REQUEST_PROFILE_CALIBRATION_SCHEMA = (
    "s42-fp16-request-admission-calibration-v1"
)
FP16_MODEL_ARTIFACTS = {
    "cold": {
        "artifact_sha256": (
            "ed76f2183d2d1d65091986033023e6c7"
            "8d27f6276c1b0c5826cc92acf73538cf"
        ),
        "bytes": 23_832_065_056,
        "model_id": "gemma-4-12b-q40-dequant-f16",
    },
    "hot": {
        "artifact_sha256": (
            "d89e9e823744222e595e0b3c8fd5436c"
            "e5d3a6a446fa42492ebce6064dfa9718"
        ),
        "bytes": 29_543_423_360,
        "model_id": "qwen3-14b-q4km-dequant-f16",
    },
}


class QualificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


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


def valid_causal_receipt(receipt: object) -> bool:
    if type(receipt) is not dict:
        return False
    causal_input = receipt.get("causal_input")
    causal_hash = receipt.get("causal_input_sha256")
    scheduler_state = (
        causal_input.get("scheduler_state")
        if type(causal_input) is dict else None
    )
    timeline = (
        scheduler_state.get("resource_timeline")
        if type(scheduler_state) is dict else None
    )
    if not (
        receipt.get("schema")
            == "research-scheduler-online-placement-v2"
        and type(causal_input) is dict
        and type(causal_hash) is str
        and hashlib.sha256(canonical(causal_input)).hexdigest()
            == causal_hash
        and causal_input.get("observed_at_us")
            == receipt.get("observed_at_us")
        and causal_input.get("previous_prefix_sha256")
            == receipt.get("previous_prefix_sha256")
        and causal_input.get("sequence_index")
            == receipt.get("sequence_index")
        and type(causal_input.get("request")) is dict
        and causal_input["request"].get("request_id")
            == receipt.get("request_id")
        and causal_input["request"].get("workload_id")
            == receipt.get("workload_id")
        and type(causal_input.get("snapshot")) is dict
        and causal_input["snapshot"].get("snapshot_id")
            == receipt.get("snapshot_id")
        and type(scheduler_state) is dict
        and scheduler_state.get("schema")
            == "research-scheduler-online-state-v1"
        and scheduler_state.get("mode") in {
            "adaptive", "capacity", "control", "enforce", "shadow"
        }
        and type(scheduler_state.get("profile_id")) is str
        and type(scheduler_state.get("profile_sha256")) is str
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            scheduler_state["profile_sha256"],
        ) is not None
        and type(timeline) is dict
        and timeline.get("schema")
            == "research-scheduler-resource-timeline-state-v1"
        and type(timeline.get("next_token")) is int
        and timeline["next_token"] >= 1
        and type(timeline.get("resources")) is dict
    ):
        return False
    receipt_body = {
        "causal_input_sha256": causal_hash,
        "decision": receipt.get("decision"),
        "family_estimates": receipt.get("family_estimates"),
        "observed_at_us": receipt.get("observed_at_us"),
        "previous_prefix_sha256": receipt.get(
            "previous_prefix_sha256"
        ),
        "request_id": receipt.get("request_id"),
        "selected_family": receipt.get("selected_family"),
        "selected_route_id": receipt.get("selected_route_id"),
        "sequence_index": receipt.get("sequence_index"),
        "snapshot_id": receipt.get("snapshot_id"),
        "workload_id": receipt.get("workload_id"),
    }
    return (
        hashlib.sha256(canonical(receipt_body)).hexdigest()
        == receipt.get("prefix_sha256")
    )


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    require(type(value) is dict, f"object expected: {path}")
    return value


def task_in_log(content: bytes, task_id: int) -> bool:
    value = str(task_id).encode("ascii")
    return any(re.search(pattern, content) is not None for pattern in (
        rb"id_task\s*=\s*" + value + rb"\b",
        rb"task id\s*=\s*" + value + rb"\b",
        rb"\|\s*task\s+" + value + rb"\s*\|",
    ))


def physical_log_route(row: dict[str, Any]) -> str | None:
    route = row.get("route")
    if route in {"desktop-cpu", "desktop-cuda", "phone-adreno", SPLIT_ROUTE}:
        return route
    binding = row.get("scheduler_final_binding")
    if type(binding) is not dict or binding.get("route_id") != route:
        return None
    return {
        "cpu": "desktop-cpu",
        "cuda": "desktop-cuda",
        "phone-adreno-ncm": "phone-adreno",
    }.get(binding.get("backend"))


def endpoint_log_hash(
    row: dict[str, Any], log_hashes: dict[str, str]
) -> str:
    physical_route = physical_log_route(row)
    if physical_route is None or physical_route not in log_hashes:
        raise QualificationError("physical route log hash is missing")
    return log_hashes[physical_route]


def split_summary(content: bytes) -> dict[str, Any] | None:
    prefix = b"S41SERVERFFN "
    rows = []
    for line in content.splitlines():
        position = line.find(prefix)
        if position < 0:
            continue
        try:
            value = json.loads(line[position + len(prefix):])
        except json.JSONDecodeError:
            continue
        if type(value) is dict:
            rows.append(value)
    return None if not rows else rows[-1]


def attempt_has_live_snapshot(
    attempt: object,
    overlay_request_index: int,
) -> bool:
    if type(attempt) is not dict:
        return False
    context = attempt.get("runtime_context")
    availability = attempt.get("executor_availability")
    estimates = attempt.get("cost_estimates")
    if not all(type(value) is dict for value in (
        context, availability, estimates
    )):
        return False
    features = context.get("cost_features")
    gpu_sample = context.get("gpu_capacity_sample")
    large_model = context.get("large_model")
    transport = context.get("transport_ownership")
    if not (
        context.get("overlay_request_index") == overlay_request_index
        and type(context.get("captured_ns")) is int
        and context["captured_ns"] > 0
        and type(features) is dict
        and all(type(value) is int for value in features.values())
        and type(gpu_sample) is dict
        and type(gpu_sample.get("age_us")) is int
        and 0 <= gpu_sample["age_us"] <= 2_000_000
        and type(gpu_sample.get("captured_at_us")) is int
        and type(gpu_sample.get("sample_t_ns")) is int
        and type(large_model) is dict
        and large_model.get("phase") in {"qwen", "switching", "gemma", "idle"}
        and type(large_model.get("phase_start_ns")) is int
        and type(transport) is dict
        and transport.get("status") in {"AVAILABLE", "LEASED"}
        and type(transport.get("resource_ids")) is list
    ):
        return False
    phone = availability.get("phone_adreno")
    phone_live = availability.get("phone_live_snapshot")
    cpu = availability.get("desktop_cpu")
    cuda = availability.get("desktop_cuda")
    if not (
        all(type(value) is dict for value in (phone, phone_live, cpu, cuda))
        and phone.get("health") in {"healthy", "unhealthy"}
        and cpu.get("health") in {"healthy", "unhealthy"}
        and cuda.get("health") in {"healthy", "unhealthy", "not_started"}
        and type(phone.get("free_slots")) is int
        and type(cpu.get("free_slots")) is int
        and type(cuda.get("free_slots")) is int
        and type(phone_live.get("sample_age_s")) in {int, float}
        and 0 <= phone_live["sample_age_s"] <= 5
        and type(phone_live.get("controller_cache_age_s")) in {int, float}
        and phone_live["controller_cache_age_s"] >= 0
        and type(phone_live.get("effective_sample_age_s")) in {int, float}
        and 0 <= phone_live["effective_sample_age_s"] <= 5
        and type(phone_live.get("available_bytes")) is int
        and type(phone_live.get("capacity_bytes")) is int
        and type(phone_live.get("task_server_alive")) is bool
        and type(phone_live.get("thermal_qualified")) is bool
    ):
        return False
    snapshot = estimates.get("snapshot")
    if type(snapshot) is not dict:
        return False
    captured_at_us = snapshot.get("captured_at_us")
    valid_until_us = snapshot.get("valid_until_us")
    capacities = snapshot.get("capacities")
    if not (
        type(captured_at_us) is int
        and type(valid_until_us) is int
        and 0 <= captured_at_us < valid_until_us
        and captured_at_us == gpu_sample["captured_at_us"]
        and valid_until_us - captured_at_us == 2_500_000
        and type(capacities) is list
    ):
        return False
    by_resource = {
        row.get("resource_id"): row
        for row in capacities
        if type(row) is dict
    }
    if set(by_resource) != {"cuda0-vram", "host-ram", "op15-ram"}:
        return False
    for capacity in by_resource.values():
        values = (
            capacity.get("capacity_bytes"),
            capacity.get("occupied_bytes"),
            capacity.get("reserve_bytes"),
            capacity.get("available_bytes"),
        )
        if not (
            all(type(value) is int and value >= 0 for value in values)
            and values[0] > 0
            and values[0] - values[1] - values[2] == values[3]
        ):
            return False
    return True


def runtime_selected_large_nonbaseline_count(
    base_result: dict[str, Any],
) -> int:
    scheduler = base_result.get("fp16_resident_scheduler", {})
    if not (
        scheduler.get("arm") == "op15"
        and scheduler.get("planned_arm") == "op15"
        and scheduler.get("arm_source") in {
            "model_device_runtime_bootstrap",
            "runtime_placement",
        }
    ):
        return 0
    rows = base_result.get("request_results")
    if type(rows) is not list:
        return 0
    return sum(
        type(row) is dict
        and type(row.get("route")) is str
        and "op15" in row["route"]
        for row in rows
    )


def causal_large_request_receipts(
    base_result: dict[str, Any],
) -> bool:
    scheduler = base_result.get("fp16_resident_scheduler", {})
    if scheduler.get("scheduler_scope") != "causal_request_level":
        return False
    cost_estimates = scheduler.get("request_level_cost_estimates")
    decisions = scheduler.get("request_level_decisions")
    observations = scheduler.get("request_level_observations")
    overheads = scheduler.get("request_level_overheads")
    receipts = scheduler.get("request_level_online_receipts")
    results = base_result.get("request_results")
    if not (
        type(cost_estimates) is dict
        and type(decisions) is dict
        and type(observations) is dict
        and type(overheads) is dict
        and type(receipts) is dict
        and type(results) is list
        and len(cost_estimates) == len(decisions) == len(observations)
            == len(overheads) == len(receipts) == len(results) == 74
    ):
        return False
    by_index = {
        str(row.get("request_index")): row
        for row in results
        if type(row) is dict
    }
    if (
        set(cost_estimates) != set(receipts)
        or set(decisions) != set(receipts)
        or set(observations) != set(receipts)
        or set(overheads) != set(receipts)
        or set(receipts) != set(by_index)
    ):
        return False
    profile_receipt = scheduler.get("request_profile")
    if type(profile_receipt) is not dict:
        return False
    profile_path_raw = profile_receipt.get("path")
    if type(profile_path_raw) is not str:
        return False
    profile_path = Path(profile_path_raw)
    try:
        profile = load(profile_path)
    except (OSError, ValueError):
        return False
    calibration = profile.get("calibration")
    applicability = (
        calibration.get("applicability")
        if type(calibration) is dict else None
    )
    audits = (
        calibration.get("audits")
        if type(calibration) is dict else None
    )
    routes = profile.get("routes")
    arm = scheduler.get("arm")
    expected_suffix = (
        "cuda_cpu_op15_f16" if arm == "op15" else "cuda_cpu_f16"
    )
    try:
        scheduler_profile_sha256 = canonical_sha256(
            ProfileBundle.from_json(profile)
        )
    except (TypeError, ValueError):
        return False
    if not (
        profile_receipt.get("profile_id") == profile.get("profile_id")
        and profile_receipt.get("sha256") == digest(profile_path)
        and profile_receipt.get("scheduler_profile_sha256")
            == scheduler_profile_sha256
        and profile.get("schema") == "s42-general-scheduler-profile-v1"
        and type(calibration) is dict
        and calibration.get("schema")
            == FP16_REQUEST_PROFILE_CALIBRATION_SCHEMA
        and calibration.get("status") == "PASS"
        and calibration.get("arm") == arm
        and calibration.get("future_request_data") == "not_accepted"
        and type(applicability) is dict
        and applicability.get("active_set_observation")
            == "physical_dispatch_time_proxy"
        and applicability.get("endpoint_queue_target")
            == "controller_wall_us_including_endpoint_queue"
        and applicability.get("ingress_capacity") == 128
        and applicability.get("same_work_repeated_holdout") is True
        and applicability.get("model_artifacts") == FP16_MODEL_ARTIFACTS
        and re.fullmatch(
            r"[0-9a-f]{64}",
            str(applicability.get("work_identity_sha256")),
        ) is not None
        and type(audits) is dict
        and set(audits) == {"cold", "hot"}
        and all(
            type(audit) is dict
            and audit.get("target")
                == "controller_wall_us_including_endpoint_queue"
            and type(audit.get("train_count")) is int
            and audit["train_count"] >= 17
            and type(audit.get("holdout_count")) is int
            and audit["holdout_count"] >= 17
            and audit.get("holdout_upper_violations") == 0
            for audit in audits.values()
        )
        and type(routes) is list
        and len(routes) == 12
        and {
            route.get("route_id")
            for route in routes
            if type(route) is dict and route.get("baseline") is True
        } == {
            f"qwen_{expected_suffix}",
            f"gemma_{expected_suffix}",
        }
    ):
        return False
    ordered = sorted(
        receipts.values(),
        key=lambda item: item.get("sequence_index", -1)
        if type(item) is dict else -1,
    )
    previous = "0" * 64
    prior_results: list[dict[str, Any]] = []
    for sequence_index, receipt in enumerate(ordered):
        if type(receipt) is not dict:
            return False
        families = receipt.get("family_estimates")
        request_id = receipt.get("request_id")
        result = next(
            (
                row for row in results
                if row.get("event_id") == request_id
            ),
            None,
        )
        overhead = (
            overheads.get(str(result.get("request_index")))
            if type(result) is dict else None
        )
        estimate_set = (
            cost_estimates.get(str(result.get("request_index")))
            if type(result) is dict else None
        )
        snapshot = (
            estimate_set.get("snapshot")
            if type(estimate_set) is dict else None
        )
        captured_at_us = (
            snapshot.get("captured_at_us")
            if type(snapshot) is dict else None
        )
        valid_until_us = (
            snapshot.get("valid_until_us")
            if type(snapshot) is dict else None
        )
        observed_at_us = receipt.get("observed_at_us")
        causal_input = receipt.get("causal_input")
        scheduler_state = (
            causal_input.get("scheduler_state")
            if type(causal_input) is dict else None
        )
        causal_request = (
            causal_input.get("request")
            if type(causal_input) is dict else None
        )
        expected_active = [
            prior for prior in prior_results
            if type(result) is dict
            and prior.get("role") == result.get("role")
            and (
                prior.get("completion_ns", 0)
                > base_result.get("paid_start_ns", 0)
                    + observed_at_us * 1000
            )
        ] if type(observed_at_us) is int else []
        expected_features = {
            "active_model_input_tokens": sum(
                prior["input_tokens"] for prior in expected_active
            ),
            "active_model_output_tokens": sum(
                prior["output_tokens"] for prior in expected_active
            ),
            "active_model_requests": len(expected_active),
        }
        release = (
            result.get("scheduler_release")
            if type(result) is dict else None
        )
        decision = receipt.get("decision")
        if not (
            valid_causal_receipt(receipt)
            and receipt.get("sequence_index") == sequence_index
            and receipt.get("previous_prefix_sha256") == previous
            and type(receipt.get("prefix_sha256")) is str
            and len(receipt["prefix_sha256"]) == 64
            and type(families) is list
            and [row.get("family") for row in families]
                == [
                    "cpu",
                    "gpu",
                    "phone",
                    "gpu-cpu",
                    "gpu-phone",
                    "cpu-phone",
                ]
            and type(result) is dict
            and type(causal_request) is dict
            and type(scheduler_state) is dict
            and scheduler_state.get("profile_id")
                == profile.get("profile_id")
            and scheduler_state.get("profile_sha256")
                == scheduler_profile_sha256
            and causal_request.get("features") == expected_features
            and causal_request.get("input_tokens")
                == result.get("input_tokens")
            and causal_request.get("output_tokens")
                == result.get("output_tokens")
            and type(estimate_set) is dict
            and estimate_set.get("request_id") == request_id
            and type(estimate_set.get("estimates")) is list
            and len(estimate_set["estimates"]) == 6
            and type(snapshot) is dict
            and snapshot.get("snapshot_id")
                == receipt.get("snapshot_id")
            and type(captured_at_us) is int
            and type(valid_until_us) is int
            and type(observed_at_us) is int
            and captured_at_us <= observed_at_us < valid_until_us
            and observed_at_us - captured_at_us <= 2_000_000
            and observations[str(result["request_index"])]
                == (
                    "arrival"
                    if str(result.get("route", "")).startswith("qwen_")
                    else "resident_endpoint_ready"
                )
            and receipt.get("selected_route_id") == result.get("route")
            and decisions[str(result["request_index"])].get("route_id")
                == result.get("route")
            and result.get("scheduler_decision")
                == decisions[str(result["request_index"])]
            and type(overhead) is dict
            and type(overhead.get("total_core_ns")) is int
            and overhead["total_core_ns"] > 0
            and type(overhead.get("total_controller_ns")) is int
            and overhead["total_controller_ns"]
                >= overhead["total_core_ns"]
            and type(release) is dict
            and release.get("status") == "released"
            and type(decision) is dict
            and type(release.get("actual_end_us")) is int
            and type(decision.get("finish_upper_us")) is int
            and release["actual_end_us"] <= decision["finish_upper_us"]
        ):
            return False
        prior_results.append(result)
        previous = receipt["prefix_sha256"]
    return True


def causal_overlay_request_receipts(
    result: dict[str, Any],
) -> bool:
    runtime = result.get("scheduler_runtime", {})
    receipts = runtime.get("online_placement_receipts")
    rows = result.get("request_results")
    if not (
        type(receipts) is dict
        and type(rows) is list
        and len(receipts) == len(rows)
    ):
        return False
    by_index = {
        str(row.get("overlay_request_index")): row
        for row in rows
        if type(row) is dict
    }
    if set(receipts) != set(by_index):
        return False
    ordered = sorted(
        receipts.items(),
        key=lambda item: item[1].get("sequence_index", -1)
        if type(item[1]) is dict else -1,
    )
    previous = "0" * 64
    for sequence_index, (index, receipt) in enumerate(ordered):
        if type(receipt) is not dict:
            return False
        families = receipt.get("family_estimates")
        row = by_index[index]
        final_decision = row.get("scheduler_final_decision")
        if not (
            valid_causal_receipt(receipt)
            and receipt.get("sequence_index") == sequence_index
            and receipt.get("previous_prefix_sha256") == previous
            and type(receipt.get("prefix_sha256")) is str
            and len(receipt["prefix_sha256"]) == 64
            and type(families) is list
            and [item.get("family") for item in families]
                == [
                    "cpu",
                    "gpu",
                    "phone",
                    "gpu-cpu",
                    "gpu-phone",
                    "cpu-phone",
                ]
            and type(final_decision) is dict
            and receipt.get("selected_route_id") == row.get("route")
            and final_decision.get("route_id") == row.get("route")
        ):
            return False
        previous = receipt["prefix_sha256"]
    return True


def release_has_final_lease_coverage(
    row: dict[str, Any],
    runtime: dict[str, Any],
) -> bool:
    release = row.get("scheduler_release")
    if type(release) is not dict:
        return False
    coverage = release.get("lease_coverage")
    if type(coverage) is dict:
        final = coverage.get("final_reserved_until_us")
        uncovered = coverage.get("uncovered_tokens")
        return (
            coverage.get("covered") is True
            and coverage.get("status") == "COVERED"
            and type(final) is dict
            and final
            and all(type(value) is int for value in final.values())
            and uncovered == []
        )

    decision = row.get("scheduler_final_decision")
    leases = decision.get("leases") if type(decision) is dict else None
    actual_end_us = release.get("actual_end_us")
    if type(leases) is not list or type(actual_end_us) is not int:
        return False
    final_ends = {
        lease.get("token"): lease.get("reserved_until_us")
        for lease in leases
        if type(lease) is dict
        and type(lease.get("token")) is str
        and type(lease.get("reserved_until_us")) is int
    }
    if len(final_ends) != len(leases) or not final_ends:
        return False
    renewals = runtime.get("renewal_receipts", {}).get(
        str(row.get("overlay_request_index")), []
    )
    if type(renewals) is not list:
        return False
    for renewal in renewals:
        extended = (
            renewal.get("extended_leases")
            if type(renewal) is dict else None
        )
        if type(extended) is not list:
            return False
        for extension in extended:
            if type(extension) is not dict:
                return False
            token = extension.get("token")
            reserved_until_us = extension.get("reserved_until_us")
            if (
                token not in final_ends
                or type(reserved_until_us) is not int
                or reserved_until_us < final_ends[token]
            ):
                return False
            final_ends[token] = reserved_until_us
    return all(actual_end_us <= value for value in final_ends.values())


def release_meets_original_latency_upper_bound(
    row: dict[str, Any],
) -> bool:
    release = row.get("scheduler_release")
    decision = row.get("scheduler_final_decision")
    if type(release) is not dict or type(decision) is not dict:
        return False
    latency = release.get("latency_upper_bound")
    if type(latency) is dict:
        return (
            latency.get("met") is True
            and latency.get("status") == "MET"
            and latency.get("overrun_us") == 0
            and latency.get("prediction_finish_upper_us")
                == decision.get("finish_upper_us")
        )
    return (
        release.get("upper_bound_violation") is False
        and type(release.get("actual_end_us")) is int
        and type(decision.get("finish_upper_us")) is int
        and release["actual_end_us"] <= decision["finish_upper_us"]
    )


def qualify(
    result_path: Path,
    cpu_log: Path,
    phone_log: Path | None,
    cuda_log: Path | None,
    split_log: Path | None,
    resident_release_path: Path,
    require_nonbaseline: bool,
    required_route: str | None = None,
) -> dict[str, Any]:
    result = load(result_path)
    require(
        result.get("schema") == RESULT_SCHEMA
        and result.get("status") == "PASS",
        "combined result",
    )
    resident_release = load(resident_release_path)
    base_result = load(Path(result["base"]["result_path"]))
    release_handshake = base_result.get(
        "fp16_resident_scheduler", {}
    ).get("release_handshake", {})
    recorded_release_sha256 = result.get("execution_sha256", {}).get(
        "resident_release"
    )
    if recorded_release_sha256 is None:
        recorded_release_sha256 = result.get("input_sha256", {}).get(
            "resident_release"
        )
    require(
        resident_release.get("schema") == "s42-resident-release-v1"
        and resident_release.get("status") == "OVERLAY_COMPLETE"
        and resident_release.get("overlay_completed")
            == result["metrics"]["small_model"]["completed"]
        and recorded_release_sha256 == digest(resident_release_path)
        and result.get("phone_residency_snapshot", {}).get(
            "final_observed_before_resident_release"
        ) is True
        and release_handshake.get("sha256")
            == digest(resident_release_path)
        and release_handshake.get("receipt") == resident_release,
        "resident release handshake",
    )
    runtime = result.get("scheduler_runtime", {})
    enabled = runtime.get("enabled") is True
    require(enabled == (result["policy"]["small_model_policy"]
        == "runtime-scheduler"), "scheduler policy identity")
    requires_overlay_receipts = (
        runtime.get("online_placement_required") is True
    )
    overlay_request_receipts_causal = causal_overlay_request_receipts(
        result
    )
    if requires_overlay_receipts:
        require(
            overlay_request_receipts_causal,
            "overlay causal request scheduling receipts",
        )
    log_paths = {
        "desktop-cpu": cpu_log,
        "desktop-cuda": cuda_log,
        "phone-adreno": phone_log,
        SPLIT_ROUTE: split_log,
    }
    log_contents: dict[str, bytes] = {}
    log_hashes: dict[str, str] = {}
    selected_routes = {
        row["route"] for row in result.get("request_results", [])
    }
    selected_physical_routes = {
        physical_log_route(row) for row in result.get("request_results", [])
    }
    require(None not in selected_physical_routes, "physical route binding")
    for route in sorted(selected_physical_routes):
        assert route is not None
        path = log_paths.get(route)
        require(path is not None and path.is_file(), f"route log: {route}")
        log_contents[route] = path.read_bytes()
        log_hashes[route] = digest(path)

    requires_execution_receipts = (
        runtime.get("physical_execution_receipt_required") is True
    )
    terminal_by_request = {}
    if requires_execution_receipts:
        decision_log = runtime.get("decision_log")
        try:
            RuntimeDecisionLog.validate(decision_log)
        except DecisionLogError as exc:
            raise QualificationError(
                f"scheduler decision log: {exc}"
            ) from exc
        terminal_by_request = {
            record["request_ids"][0]: record
            for record in decision_log["records"]
            if record["event_kind"] in {"COMPLETED", "FAILED", "CANCELLED"}
        }

    endpoint_receipts = []
    for row in result.get("request_results", []):
        route = row.get("route")
        physical_route = physical_log_route(row)
        decision = row.get("scheduler_final_decision")
        binding = row.get("scheduler_final_binding")
        task_id = row.get("endpoint_task_id")
        require(
            physical_route in log_contents
            and row.get("endpoint_model_alias") == LLAMA1
            and type(task_id) is int
            and task_id >= 0
            and type(row.get("endpoint_slot_id")) is int
            and type(row.get("stream_sha256")) is str
            and len(row["stream_sha256"]) == 64
            and (
                not enabled
                or type(decision) is dict
                and decision.get("route_id") == route
            ),
            "decision and physical endpoint identity",
        )
        require(
            task_in_log(log_contents[physical_route], task_id),
            f"endpoint task missing from {physical_route} log: {task_id}",
        )
        if requires_execution_receipts:
            request_id = decision.get("request_id")
            terminal = terminal_by_request.get(request_id)
            selected = (
                None if terminal is None else terminal.get("selected")
            )
            receipt = (
                None
                if type(selected) is not dict
                else selected.get("execution_receipt")
            )
            plan = row.get("scheduler_final_execution_plan")
            participants = (
                []
                if type(binding) is not dict
                else binding.get("participants", [])
            )
            require(
                type(binding) is dict
                and type(plan) is dict
                and type(receipt) is dict
                and selected.get("executor") == binding
                and selected.get("operator_plan") == plan
                and receipt.get("status") == "COMPLETED"
                and receipt.get("request_id") == request_id
                and receipt.get("executor_id")
                    == binding.get("executor_id")
                and receipt.get("endpoint") == binding.get("endpoint")
                and receipt.get("operator_plan_protocol")
                    == binding.get("operator_plan_protocol")
                and receipt.get("operator_plan_sha256")
                    == plan.get("plan_sha256")
                and receipt.get("participant_executor_ids")
                    == sorted(
                        participant["executor_id"]
                        for participant in participants
                    )
                and receipt.get("output_sha256")
                    == "sha256:" + row["stream_sha256"],
                "scheduler-bound physical execution receipt",
            )
        endpoint_receipts.append({
            "endpoint_slot_id": row["endpoint_slot_id"],
            "endpoint_task_id": task_id,
            "event_id": row["event_id"],
            "log_sha256": endpoint_log_hash(row, log_hashes),
            "overlay_request_index": row["overlay_request_index"],
            "physical_route": physical_route,
            "route": route,
            "stream_sha256": row["stream_sha256"],
        })

    small_count = result["metrics"]["small_model"]["completed"]
    route_counts = runtime.get("route_counts", {})
    if requires_execution_receipts:
        nonbaseline_count = sum(
            row.get("scheduler_final_candidate_baseline") is False
            for row in result.get("request_results", [])
        )
    else:
        nonbaseline_count = sum(
            count for route, count in route_counts.items()
            if route != "desktop-cpu"
        )
    physical_route_counts = {}
    for row in result.get("request_results", []):
        route = physical_log_route(row)
        physical_route_counts[route] = physical_route_counts.get(route, 0) + 1
    large_nonbaseline_count = runtime_selected_large_nonbaseline_count(
        base_result
    )
    large_request_receipts_causal = causal_large_request_receipts(
        base_result
    )
    requires_large_request_receipts = (
        base_result.get("fp16_resident_scheduler", {}).get("arm_source")
        == "model_device_runtime_bootstrap"
    )
    if requires_large_request_receipts:
        require(
            large_request_receipts_causal,
            "large-model causal request scheduling receipts",
        )
    require(
        sum(route_counts.values()) == small_count
        and (
            not require_nonbaseline
            or nonbaseline_count > 0
            or large_nonbaseline_count > 0
        )
        and (
            required_route is None
            or physical_route_counts.get(required_route, 0) > 0
        ),
        "physical route diversity",
    )
    split_execution = None
    if SPLIT_ROUTE in selected_physical_routes:
        split_execution = split_summary(log_contents[SPLIT_ROUTE])
        require(
            split_execution is not None
            and split_execution.get("status") == "ok"
            and type(split_execution.get("calls")) is int
            and split_execution["calls"] > 0,
            "physical FFN split callback receipt",
        )
    release_receipts = runtime.get("release_receipts", {})
    if enabled:
        require(
            len(release_receipts) == small_count
            and all(
                release_has_final_lease_coverage(row, runtime)
                for row in result.get("request_results", [])
            ),
            "final renewed lease coverage",
        )
        require(
            all(
                release_meets_original_latency_upper_bound(row)
                for row in result.get("request_results", [])
            ),
            "latency upper-bound qualification",
        )
        attempts = runtime.get("decision_attempts", {})
        require(
            len(attempts) == small_count
            and all(
                type(rows) is list
                and rows
                and all(
                    attempt_has_live_snapshot(row, int(index))
                    for row in rows
                )
                for index, rows in attempts.items()
            ),
            "live phone snapshots",
        )
    phone_residency = result.get("phone_residency_snapshot", {})
    require(
        phone_residency.get("startup") is not None
        and phone_residency.get("final") is not None,
        "phone boundary snapshots",
    )
    reservations = runtime.get("external_reservations", [])
    require(
        all(
            row.get("status") == "RELEASED"
            and type(row.get("released_at_us")) is int
            and row["released_at_us"] >= row["started_at_us"]
            for row in reservations
        ),
        "external transport lease release",
    )
    if result["policy"]["large_model_policy"] == "op15-assistance":
        require(
            {row.get("phase") for row in reservations}
                == {"qwen", "gemma"},
            "OP15 phase transport coverage",
        )
    else:
        require(not reservations, "CPU arm has FunctionFS reservations")

    input_hashes = result.get("input_sha256", {})
    execution_hashes = result.get("execution_sha256", {})
    require(
        input_hashes
        and all(
            type(value) is str
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in input_hashes.values()
        )
        and (
            not execution_hashes
            or all(
                type(value) is str
                and len(value) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in value
                )
                for value in execution_hashes.values()
            )
        )
        and all(
            type(model.get("artifact_sha256")) is str
            and len(model["artifact_sha256"]) == 64
            for model in result.get("model_identities", {}).values()
        )
        and result.get("server_energy", {}).get("boundary")
            == "paid_trace_interval",
        "trace, model, and energy identity",
    )
    return {
        "endpoint_receipts": endpoint_receipts,
        "gates": {
            "all_endpoint_tasks_in_selected_server_log": True,
            "all_leases_cover_physical_execution": (
                not enabled
                or all(
                    release_has_final_lease_coverage(row, runtime)
                    for row in result.get("request_results", [])
                )
            ),
            "all_original_latency_upper_bounds_met": (
                not enabled
                or all(
                    release_meets_original_latency_upper_bound(row)
                    for row in result.get("request_results", [])
                )
            ),
            "all_runtime_snapshots_live": True,
            "decision_matches_physical_endpoint": True,
            "large_request_receipts_causal": (
                not requires_large_request_receipts
                or large_request_receipts_causal
            ),
            "overlay_request_receipts_causal": (
                not requires_overlay_receipts
                or overlay_request_receipts_causal
            ),
            "nonbaseline_physical_execution": (
                nonbaseline_count > 0 or large_nonbaseline_count > 0
            ),
            "selected_split_route_has_phone_ffn_calls": (
                SPLIT_ROUTE not in selected_routes
                and SPLIT_ROUTE not in selected_physical_routes
                or split_execution is not None
            ),
            "phase_transport_leases_released": True,
            "resident_release_after_final_phone_snapshot": True,
            "trace_model_energy_identity": True,
        },
        "input_sha256": {
            "result": digest(result_path),
            "resident_release": digest(resident_release_path),
            **{
                f"{route}_log": value
                for route, value in sorted(log_hashes.items())
            },
        },
        "nonbaseline_physical_executions": nonbaseline_count,
        "runtime_selected_large_nonbaseline_executions": (
            large_nonbaseline_count
        ),
        "policy": result["policy"],
        "required_route": required_route,
        "route_counts": route_counts,
        "physical_route_counts": dict(sorted(physical_route_counts.items())),
        "schema": QUALIFICATION_SCHEMA,
        "status": "PASS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--cpu-log", type=Path, required=True)
    parser.add_argument("--phone-log", type=Path)
    parser.add_argument("--cuda-log", type=Path)
    parser.add_argument("--split-log", type=Path)
    parser.add_argument("--resident-release", type=Path, required=True)
    parser.add_argument("--require-nonbaseline", action="store_true")
    parser.add_argument(
        "--require-route",
        choices=("desktop-cpu", "desktop-cuda", "phone-adreno", SPLIT_ROUTE),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        args.output.is_absolute() and not args.output.exists(),
        "new absolute output path",
    )
    try:
        value = qualify(
            args.result,
            args.cpu_log,
            args.phone_log,
            args.cuda_log,
            args.split_log,
            args.resident_release,
            args.require_nonbaseline,
            args.require_route,
        )
    except QualificationError as error:
        value = {
            "error": str(error),
            "input_sha256": {
                name: digest(path)
                for name, path in {
                    "result": args.result,
                    "resident_release": args.resident_release,
                }.items()
                if path.is_file()
            },
            "required_route": args.require_route,
            "schema": QUALIFICATION_SCHEMA,
            "status": "FAIL",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(value))
    print(json.dumps({
        "error": value.get("error"),
        "nonbaseline_physical_executions": value.get(
            "nonbaseline_physical_executions"
        ),
        "output": str(args.output),
        "status": value["status"],
    }, sort_keys=True))
    return 0 if value["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
