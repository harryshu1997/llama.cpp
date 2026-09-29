#!/usr/bin/env python3
"""Replay a trace through the S42 general scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    POLICY_MODES,
    ProfileBundle,
    ProfileCatalogError,
    UnifiedScheduler,
    TraceError,
    decision_to_json,
    load_trace,
    load_profile_bundle,
)


OUTPUT_SCHEMA = "s42-general-scheduler-replay-v1"


class ReplayError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def load_profile(path: Path) -> tuple[dict[str, Any], ProfileBundle]:
    try:
        return load_profile_bundle(path)
    except ProfileCatalogError as exc:
        raise ReplayError(str(exc)) from exc


def summarize(
    mode: str,
    profile: ProfileBundle,
    requests: Iterable[Any],
    output_tokens: int,
    ready_overrides: dict[str, bool] | None = None,
) -> dict[str, Any]:
    scheduler = UnifiedScheduler((profile,), mode)
    for resource_id, ready in sorted((ready_overrides or {}).items()):
        scheduler.set_resource_ready(resource_id, ready, 0)
    request_list = list(requests)
    decisions = [scheduler.schedule(request) for request in request_list]
    routes = Counter(decision.route_id for decision in decisions)
    reasons = Counter(decision.reason for decision in decisions)
    rejections = Counter(
        reason for decision in decisions for _, reason in decision.rejected
    )
    blocking_resources = Counter(
        resource_id
        for decision in decisions
        for resource_id in decision.blocking_resources
    )
    queue_by_resource_us: Counter[str] = Counter()
    lease_predicted_us: Counter[str] = Counter()
    lease_reserved_us: Counter[str] = Counter()
    for decision in decisions:
        queue_by_resource_us.update(decision.queue_by_resource_us)
        for lease in decision.leases:
            lane_count = len(lease.lanes)
            lease_predicted_us[lease.resource_id] += (
                lease.predicted_end_us - lease.start_us
            ) * lane_count
            lease_reserved_us[lease.resource_id] += (
                lease.reserved_until_us - lease.start_us
            ) * lane_count
    by_workload: dict[str, list[Any]] = defaultdict(list)
    for decision in decisions:
        by_workload[decision.workload_id].append(decision)
    makespan_us = max(decision.finish_us for decision in decisions)
    conservative_slo = sum(
        decision.finish_upper_us <= request.deadline_us
        for decision, request in zip(decisions, request_list)
    )
    predicted_slo = sum(
        decision.finish_us <= request.deadline_us
        for decision, request in zip(decisions, request_list)
    )
    known_energy = [
        decision.energy_uj for decision in decisions if decision.energy_uj is not None
    ]
    energy_complete = len(known_energy) == len(decisions)
    return {
        "mode": mode,
        "request_count": len(decisions),
        "output_tokens": output_tokens,
        "makespan_us": makespan_us,
        "throughput_tokens_s": output_tokens * 1_000_000.0 / makespan_us,
        "predicted_slo_met": predicted_slo,
        "conservative_slo_met": conservative_slo,
        "route_counts": dict(sorted(routes.items())),
        "decision_reasons": dict(sorted(reasons.items())),
        "rejection_counts": dict(sorted(rejections.items())),
        "resource_queue": {
            "blocking_decision_counts": dict(sorted(blocking_resources.items())),
            "delay_total_us": dict(sorted(queue_by_resource_us.items())),
        },
        "resource_leases": {
            "predicted_lane_us": dict(sorted(lease_predicted_us.items())),
            "reserved_lane_us": dict(sorted(lease_reserved_us.items())),
        },
        "energy": {
            "complete": energy_complete,
            "known_request_count": len(known_energy),
            "total_uj": sum(known_energy) if energy_complete else None,
            "claim": (
                "TOTAL_FLEET_ENERGY_MODEL_COMPLETE"
                if energy_complete
                else "NO_ENERGY_CLAIM_INCOMPLETE_BOUNDARY"
            ),
        },
        "workloads": {
            workload: {
                "request_count": len(rows),
                "finish_max_us": max(row.finish_us for row in rows),
                "service_total_us": sum(row.service_us for row in rows),
                "queue_total_us": sum(row.queue_us for row in rows),
                "queue_by_resource_us": dict(sorted(
                    sum(
                        (Counter(row.queue_by_resource_us) for row in rows),
                        Counter(),
                    ).items()
                )),
                "server_busy_total_us": sum(row.server_busy_us for row in rows),
                "route_counts": dict(sorted(Counter(row.route_id for row in rows).items())),
            }
            for workload, rows in sorted(by_workload.items())
        },
        "decisions": [decision_to_json(decision) for decision in decisions],
    }


def replay(
    profile_path: Path,
    trace_path: Path,
    modes: list[str],
    quality_requirement: str,
    runtime_state_path: Path | None = None,
) -> dict[str, Any]:
    raw_profile, profile = load_profile(profile_path)
    trace = load_trace(
        trace_path,
        profile.trace_workload_map,
        quality_requirement=quality_requirement,
    )
    ready_overrides: dict[str, bool] | None = None
    runtime_state: dict[str, object] | None = None
    if runtime_state_path is not None:
        try:
            with runtime_state_path.open("r", encoding="ascii") as source:
                state = json.load(source, object_pairs_hook=_no_duplicates)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReplayError(f"cannot read runtime state: {exc}") from exc
        if type(state) is not dict or state.get("schema") != "s42-live-probe-v1":
            raise ReplayError("runtime state schema mismatch")
        supplied = state.get("probe_hash")
        unhashed_state = dict(state)
        unhashed_state.pop("probe_hash", None)
        expected = "sha256:" + hashlib.sha256(
            canonical_bytes(unhashed_state)
        ).hexdigest()
        if supplied != expected:
            raise ReplayError("runtime state hash mismatch")
        raw_ready = state.get("scheduler_resource_ready")
        if type(raw_ready) is not dict or any(
            type(key) is not str or type(value) is not bool
            for key, value in raw_ready.items()
        ):
            raise ReplayError("runtime readiness map is invalid")
        ready_overrides = dict(raw_ready)
        runtime_state = {
            "path": str(runtime_state_path),
            "probe_hash": supplied,
            "mutation_scope": state.get("mutation_scope"),
        }
    result: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "profile": {
            "path": str(profile_path),
            "profile_id": profile.profile_id,
            "profile_hash": raw_profile["profile_hash"],
        },
        "trace": {
            "path": trace.source_path,
            "sha256": trace.source_sha256,
            "quality_requirement": quality_requirement,
        },
        "runtime_state": runtime_state,
        "modes": [
            summarize(
                mode,
                profile,
                trace.requests,
                trace.output_tokens,
                ready_overrides,
            )
            for mode in modes
        ],
    }
    result["replay_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(result)
    ).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--modes", default="control,enforce,shadow,capacity"
    )
    parser.add_argument("--quality-requirement", default="approximate")
    parser.add_argument("--runtime-state", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    modes = args.modes.split(",")
    if not modes or any(mode not in POLICY_MODES for mode in modes):
        parser.error(
            "modes must be a comma-separated subset of "
            "control,enforce,shadow,capacity,adaptive"
        )
    if len(set(modes)) != len(modes):
        parser.error("modes must not repeat")
    if args.output is not None and args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        result = replay(
            args.profile,
            args.trace,
            modes,
            args.quality_requirement,
            args.runtime_state,
        )
        payload = canonical_bytes(result)
        if args.output is None:
            print(payload.decode("ascii"), end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(payload)
            print(json.dumps({
                "output": str(args.output),
                "replay_hash": result["replay_hash"],
            }, sort_keys=True, separators=(",", ":")))
    except (ReplayError, SchedulerError, TraceError) as exc:
        parser.exit(2, f"replay failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
