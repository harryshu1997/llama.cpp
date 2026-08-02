#!/usr/bin/env python3
"""Run the frozen S36 trace on two phones and one selected A6000."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dynamic_cut_policy import DeviceState, DynamicCutPolicy, RequestWork
from dynamic_route_runtime import DynamicOutcome, DynamicRequest
from physical_topology import WORKERS, PhysicalTopology, build_topology
from profile_routes import parse_endpoint
from profiles import canonical_bytes, load_profile_bundle
from trace_adapter import load_built_trace


SCHEMA = "s36-dynamic-cut-physical-run-v1"


class RunError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def nearest_rank(values: list[int], numerator: int, denominator: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def value_summary(values: list[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": nearest_rank(values, 95, 100),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def outcome_record(
    outcome: DynamicOutcome,
    source: dict[str, Any],
    origin_ns: int,
    admitted_ns: int,
) -> dict[str, Any]:
    return {
        "request_id": outcome.request_id,
        "route_epoch": outcome.route_epoch,
        "route_id": outcome.route_id,
        "device": outcome.device,
        "cut": outcome.cut,
        "priority": outcome.priority,
        "prompt_length": outcome.prompt_length,
        "output_tokens": list(outcome.output_tokens),
        "arrival_us": source["arrival_us"],
        "admitted_us": (admitted_ns - origin_ns) // 1000,
        "call_us": (outcome.call_ns - origin_ns) // 1000,
        "first_token_us": (outcome.first_token_ns - origin_ns) // 1000,
        "completed_us": (outcome.completed_ns - origin_ns) // 1000,
        "admission_queue_us": (
            admitted_ns - outcome.scheduled_arrival_ns
        ) // 1000,
        "lease_queue_us": (
            outcome.lease_ready_ns - outcome.call_ns
        ) // 1000,
        "ttft_us": outcome.ttft_us,
        "latency_us": outcome.latency_us,
        "slo_us": outcome.slo_us,
        "slo_met": outcome.slo_met,
        "observed_input_tokens": source["observed_input_tokens"],
        "observed_output_tokens": source["observed_output_tokens"],
    }


def validate_events(topology: PhysicalTopology) -> None:
    for name, batcher in topology.batchers.items():
        for event in batcher.events:
            if event.get("status") != "OK":
                raise RunError(f"{name} emitted a failed physical batch")
            cut = event.get("cut")
            if cut not in batcher.ranges:
                raise RunError(f"{name} emitted an unknown cut")
            if event.get("active_range") != list(batcher.ranges[cut]):
                raise RunError(f"{name} emitted a mismatched active range")
            priorities = event.get("priorities")
            phases = event.get("phases")
            if (
                type(priorities) is not list
                or not priorities
                or any(type(value) is not int or value < 0 for value in priorities)
                or type(phases) is not list
                or len(phases) != len(priorities)
                or any(value not in ("prefill", "decode") for value in phases)
            ):
                raise RunError(f"{name} emitted invalid batch lineage")
            if 0 in priorities and any(value != 0 for value in priorities):
                raise RunError(f"{name} mixed priority zero with background work")
            if event.get("priority_band") != (0 if 0 in priorities else 1):
                raise RunError(f"{name} priority band evidence is inconsistent")
            if event.get("mixed_phase") != (len(set(phases)) > 1):
                raise RunError(f"{name} mixed-phase evidence is inconsistent")


def summarize(
    records: list[dict[str, Any]], topology: PhysicalTopology, duration_us: int,
) -> dict[str, Any]:
    priority = {}
    for value in (0, 1, 2):
        rows = [row for row in records if row["priority"] == value]
        priority[str(value)] = {
            "completed": len(rows),
            "slo_misses": sum(not row["slo_met"] for row in rows),
            "latency_us": value_summary([row["latency_us"] for row in rows]),
            "ttft_us": value_summary([row["ttft_us"] for row in rows]),
        }
    workers = {}
    for name in WORKERS:
        events = topology.batchers[name].events
        sizes = [int(event["batch_size"]) for event in events]
        workers[name] = {
            "batch_count": len(events),
            "batch_sizes": sizes,
            "mean_batch": statistics.fmean(sizes) if sizes else 0.0,
            "max_batch": max(sizes, default=0),
            "mixed_phase_batches": sum(bool(event["mixed_phase"]) for event in events),
            "summed_compute_us": sum(int(event["compute_us"]) for event in events),
            "cuts": dict(sorted(Counter(str(event["cut"]) for event in events).items())),
            "release_reasons": dict(sorted(Counter(
                str(event["release_reason"]) for event in events
            ).items())),
        }
    return {
        "completed_requests": len(records),
        "slo_misses": sum(not row["slo_met"] for row in records),
        "duration_us": duration_us,
        "route_distribution": dict(sorted(Counter(
            row["route_id"] for row in records
        ).items())),
        "device_distribution": dict(sorted(Counter(
            row["device"] for row in records
        ).items())),
        "cut_distribution": dict(sorted(Counter(
            str(row["cut"]) for row in records
        ).items())),
        "priority": priority,
        "workers": workers,
        "selected_cuda_compute_us": (
            workers["cuda"]["summed_compute_us"]
            + workers["tail"]["summed_compute_us"]
        ),
        "phone_mixed_phase_batches": (
            workers["op12"]["mixed_phase_batches"]
            + workers["op15"]["mixed_phase_batches"]
        ),
    }


def run_workload(
    topology: PhysicalTopology,
    trace: dict[str, Any],
    policy: DynamicCutPolicy,
    force_control: bool,
    gather_us: int,
    request_timeout_s: float,
    expected_tokens: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not expected_tokens or any(type(token) is not int for token in expected_tokens):
        raise RunError("profile token signature is invalid")
    source_rows = list(trace["requests"])
    source_by_id = {row["request_id"]: row for row in source_rows}
    profile_by_id = policy.profiles
    route_by_id = topology.runner.routes
    head_capacity = {
        name: topology.stages[name].slots.capacity for name in ("cuda", "op12", "op15")
    }
    tail_capacity = topology.stages["tail"].slots.capacity
    active_counts = {name: 0 for name in head_capacity}
    active_tail = 0
    active: dict[int, tuple[threading.Thread, str]] = {}
    pending: list[dict[str, Any]] = []
    completions: queue.Queue[dict[str, Any]] = queue.Queue()
    decisions = []
    records = []
    next_arrival = 0
    next_epoch = 1
    fatal: BaseException | None = None
    origin_ns = time.monotonic_ns()

    def elapsed_us() -> int:
        return max(0, (time.monotonic_ns() - origin_ns) // 1000)

    def execute(
        source: dict[str, Any], request: DynamicRequest, admitted_ns: int,
    ) -> None:
        try:
            outcome = topology.runner.run(
                request,
                request_timeout_s,
                origin_ns + source["arrival_us"] * 1000,
            )
            completions.put({
                "kind": "ok",
                "request_id": source["request_id"],
                "device": outcome.device,
                "record": outcome_record(outcome, source, origin_ns, admitted_ns),
            })
        except BaseException as exc:
            completions.put({
                "kind": "error",
                "request_id": source["request_id"],
                "device": route_by_id[request.route_id].head.name,
                "error": exc,
            })

    def consume(item: dict[str, Any]) -> None:
        nonlocal active_tail, fatal
        request_id = item["request_id"]
        owned = active.pop(request_id, None)
        if owned is None:
            fatal = RunError("completion has no active owner")
            return
        device = item["device"]
        active_counts[device] -= 1
        active_tail -= 1
        if active_counts[device] < 0 or active_tail < 0:
            fatal = RunError("physical credits underflowed")
            return
        if item["kind"] == "error":
            fatal = item["error"]
        else:
            records.append(item["record"])

    while len(records) < len(source_rows):
        now_us = elapsed_us()
        while (
            next_arrival < len(source_rows)
            and source_rows[next_arrival]["arrival_us"] <= now_us
        ):
            pending.append(source_rows[next_arrival])
            next_arrival += 1

        while True:
            try:
                consume(completions.get_nowait())
            except queue.Empty:
                break
        if fatal is not None:
            break

        pending.sort(key=lambda row: (
            row["priority"],
            row["arrival_us"] + row["slo_us"],
            row["arrival_us"],
            row["request_id"],
        ))
        admitted_any = False
        while pending and active_tail < tail_capacity:
            source = pending[0]
            now_us = elapsed_us()
            work = RequestWork(
                source["request_id"],
                source["priority"],
                source["arrival_us"],
                source["arrival_us"] + source["slo_us"],
            )
            states = {
                name: DeviceState(
                    active_requests=active_counts[name],
                    capacity=head_capacity[name],
                    estimated_queue_us=0,
                    ready=True,
                    queued_rows_by_cut=topology.batchers[name].pending_by_cut(),
                )
                for name in ("op12", "op15")
            }
            decision = policy.choose(work, now_us, states, force_control)
            route = route_by_id[decision.route_id]
            device = route.head.name
            if active_counts[device] >= head_capacity[device]:
                break
            profile = profile_by_id[decision.route_id]
            predicted_us = profile.predicted_p95_us + profile.safety_margin_us
            remaining_us = work.deadline_us - now_us
            batch_wait_us = min(gather_us, max(0, remaining_us - predicted_us))
            if source["priority"] == 0:
                batch_wait_us = 0
            pending.pop(0)
            active_counts[device] += 1
            active_tail += 1
            route_epoch = next_epoch
            next_epoch += 1
            admitted_ns = time.monotonic_ns()
            request = DynamicRequest(
                request_id=source["request_id"],
                route_epoch=route_epoch,
                route_id=decision.route_id,
                prompt_tokens=tuple(source["prompt_tokens"]),
                output_steps=source["output_steps"],
                priority=source["priority"],
                slo_us=source["slo_us"],
                batch_wait_us=batch_wait_us,
            )
            decisions.append({
                **asdict(decision),
                "route_epoch": route_epoch,
                "admitted_us": (admitted_ns - origin_ns) // 1000,
                "batch_wait_us": batch_wait_us,
                "active_after_admit": dict(active_counts),
                "tail_active_after_admit": active_tail,
                "queue_model": "B32_PROFILE_BOUNDS_ONE_RESIDENT_CREDIT_EPOCH",
            })
            thread = threading.Thread(
                target=execute,
                args=(source, request, admitted_ns),
                name=f"s36-{source['request_id']}",
            )
            active[source["request_id"]] = (thread, device)
            thread.start()
            admitted_any = True

        if len(records) == len(source_rows):
            break
        if not admitted_any:
            wait_s = 0.005
            now_us = elapsed_us()
            if next_arrival < len(source_rows):
                wait_s = min(
                    wait_s,
                    max(0.0, (source_rows[next_arrival]["arrival_us"] - now_us) / 1e6),
                )
            if wait_s <= 0:
                wait_s = 0.001
            try:
                consume(completions.get(timeout=wait_s))
            except queue.Empty:
                pass
            if fatal is not None:
                break

    for thread, _device in list(active.values()):
        thread.join(request_timeout_s)
    if any(thread.is_alive() for thread, _device in active.values()):
        raise TimeoutError("physical request thread did not terminate")
    while True:
        try:
            consume(completions.get_nowait())
        except queue.Empty:
            break
    if fatal is not None:
        raise RunError("physical request execution failed") from fatal
    if pending or next_arrival != len(source_rows) or len(records) != len(source_rows):
        raise RunError("request conservation failed")
    if active or any(active_counts.values()) or active_tail:
        raise RunError("runtime retained physical credits")
    if topology.runner.pins():
        raise RunError("runtime retained route pins")
    if any(stage.slots.leased() for stage in topology.stages.values()):
        raise RunError("runtime retained software sequence leases")
    if {row["request_id"] for row in records} != set(source_by_id):
        raise RunError("terminal request ownership differs from trace")
    if any(tuple(row["output_tokens"]) != expected_tokens for row in records):
        raise RunError("runtime token output differs from the physical profile")
    records.sort(key=lambda row: row["request_id"])
    return {
        "origin_monotonic_ns": origin_ns,
        "duration_us": elapsed_us(),
        "decisions": decisions,
        "requests": records,
    }, {
        "active_counts": active_counts,
        "tail_active": active_tail,
        "route_pins": topology.runner.pins(),
        "software_leases": {
            name: topology.stages[name].slots.leased() for name in WORKERS
        },
        "profile_signature_checked": list(expected_tokens),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--tail", type=parse_endpoint, required=True)
    result.add_argument("--trace", type=Path, required=True)
    result.add_argument("--profiles", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--control", action="store_true")
    result.add_argument("--gather-us", type=int, default=20_000)
    result.add_argument("--queue-depth", type=int, default=4096)
    result.add_argument("--phone-knee", type=int, default=32)
    result.add_argument("--cuda-knee", type=int, default=32)
    result.add_argument("--tail-knee", type=int, default=32)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--request-timeout", type=float, default=120.0)
    result.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    return result


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.output.exists():
        arg_parser.error(f"output already exists: {args.output}")
    if (
        args.gather_us < 0
        or args.queue_depth <= 0
        or min(args.phone_knee, args.cuda_knee, args.tail_knee) <= 0
        or args.timeout <= 0
        or args.request_timeout <= 0
    ):
        arg_parser.error("runtime bounds are invalid")
    topology = None
    try:
        trace = load_built_trace(args.trace)
        profile_bundle, profiles = load_profile_bundle(args.profiles)
        policy = DynamicCutPolicy(profiles)
        topology = build_topology(
            {name: getattr(args, name) for name in WORKERS},
            args.timeout,
            args.gather_us,
            args.queue_depth,
            args.phone_knee,
            args.cuda_knee,
            args.tail_knee,
        )
        runtime, final_state = run_workload(
            topology,
            trace,
            policy,
            args.control,
            args.gather_us,
            args.request_timeout,
            tuple(profile_bundle.get("token_signature", ())),
        )
        validate_events(topology)
        summary = summarize(runtime["requests"], topology, runtime["duration_us"])
        topology.stop_batchers(args.timeout)
        final_workers = topology.end_sessions(args.session_end)
        report = {
            "schema": SCHEMA,
            "status": "RUN_COMPLETE",
            "mode": "ALL_CUDA_CONTROL" if args.control else "DYNAMIC_CUT_TREATMENT",
            "trace": {
                "path": str(args.trace),
                "file_sha256": sha256_file(args.trace),
                "trace_hash": trace["trace_hash"],
                "scope": trace["scope"],
            },
            "profiles": {
                "path": str(args.profiles),
                "file_sha256": sha256_file(args.profiles),
                "profile_hash": profile_bundle["profile_hash"],
            },
            "configuration": {
                "gather_us": args.gather_us,
                "queue_depth": args.queue_depth,
                "phone_knee_rows": args.phone_knee,
                "cuda_knee_rows": args.cuda_knee,
                "tail_knee_rows": args.tail_knee,
                "session_end": args.session_end,
                "endpoints": {
                    name: list(getattr(args, name)) for name in WORKERS
                },
                "second_a6000": "EXCLUDED",
            },
            "workers": {
                name: asdict(topology.hellos[name]) for name in WORKERS
            },
            "runtime": runtime,
            "batch_events": {
                name: list(topology.batchers[name].events) for name in WORKERS
            },
            "summary": summary,
            "final_state": final_state,
            "final_workers": final_workers,
            "claim_scope": {
                "selected_cuda_compute": "MEASURED_HOST_WALL_AROUND_BLOCKING_STAGE_CALLS",
                "gpu_board_energy": "NOT_MEASURED",
                "phone_energy": "UNKNOWN",
                "usb_wifi_energy": "UNKNOWN",
                "total_system_energy": "UNKNOWN",
                "whole_file_model_identity": "DIFFERS_FOR_SHARDS",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": report["status"],
            "mode": report["mode"],
            "output": str(args.output),
            "route_distribution": summary["route_distribution"],
            "slo_misses": summary["slo_misses"],
            "selected_cuda_compute_us": summary["selected_cuda_compute_us"],
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "status": "RUN_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical_bytes(failure))
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")))
        return 2
    finally:
        if topology is not None:
            if not topology.batchers_stopped:
                try:
                    topology.stop_batchers(args.timeout)
                except BaseException:
                    pass
            if not topology.session_ended:
                try:
                    if all(
                        topology.clients[name].status().active_sequences == 0
                        for name in WORKERS
                    ):
                        for name in WORKERS:
                            topology.clients[name].drain()
                            topology.clients[name].detach()
                        topology.session_ended = True
                except BaseException:
                    pass
            topology.close()


if __name__ == "__main__":
    raise SystemExit(main())
