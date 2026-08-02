#!/usr/bin/env python3
"""Run the S26 priority controller over the physical S24 StageNet routes."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S24 = HERE.parent / "s24_overlap_handoff_poc"
for dependency in (HERE, S24):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from physical_adapter import (
    WORKER_NAMES,
    PhysicalRuntimeError,
    build_topology,
    canonical_bytes,
    load_trace,
    outcome_record,
    parse_endpoint,
    physical_capacities,
    sha256_file,
    summarize_run,
)
from priority_policy import (
    DispatchGroup,
    PriorityAdmissionController,
    PriorityPolicyError,
    PriorityWork,
    RejectDecision,
    WaitDecision,
)
from priority_profiles import PriorityProfileError, load_bundle
from route_runtime import RouteRequest


SCHEMA = "s26-priority-physical-v1"
EXPECTED_TRACE_HASH = "sha256:9b4e84bdd5d6bf38bd8d951dd043ad5d7a1b22520f746ec2953ab547db81c8be"


@dataclass(frozen=True)
class ScheduledWork:
    source: dict[str, Any]
    arrival_us: int

    @property
    def request_id(self) -> int:
        return int(self.source["request_id"])


def prepare_trace(trace: dict[str, Any]) -> list[ScheduledWork]:
    if trace.get("trace_hash") != EXPECTED_TRACE_HASH:
        raise PhysicalRuntimeError("S26 requires the frozen deterministic trace")
    result = []
    seen = set()
    for source in trace["requests"]:
        request_id = source["request_id"]
        if request_id in seen:
            raise PhysicalRuntimeError("trace request id is duplicated")
        seen.add(request_id)
        result.append(ScheduledWork(source, source["arrival_us"]))
    result.sort(key=lambda row: (row.arrival_us, row.request_id))
    if not result:
        raise PhysicalRuntimeError("trace contains no work")
    return result


def _work(row: ScheduledWork) -> PriorityWork:
    source = row.source
    return PriorityWork(
        request_id=row.request_id,
        arrival_us=row.arrival_us,
        deadline_us=row.arrival_us + source["slo_us"],
        priority=source["priority"],
        input_tokens=source["input_tokens"],
        output_steps=source["output_steps"],
    )


def validate_conservation(
    rows: list[ScheduledWork],
    decisions: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    allowed_batches: dict[str, frozenset[int]] | None = None,
) -> None:
    if allowed_batches is None:
        allowed_batches = {
            "R0": frozenset((1, 4)),
            "R2": frozenset((1, 4)),
        }
    if (
        type(allowed_batches) is not dict
        or set(allowed_batches) != {"R0", "R2"}
        or any(
            type(values) is not frozenset
            or not values
            or any(type(value) is not int or value <= 0 for value in values)
            for values in allowed_batches.values()
        )
    ):
        raise PhysicalRuntimeError("allowed route batch points are invalid")
    expected = {row.request_id for row in rows}
    source_by_id = {row.request_id: row.source for row in rows}
    decided = [
        request_id
        for decision in decisions
        for request_id in decision["request_ids"]
    ]
    completed_ids = [row["request_id"] for row in completed]
    rejected_ids = [row["request_id"] for row in rejected]
    if len(decided) != len(set(decided)):
        raise PhysicalRuntimeError("a request appears in multiple dispatch groups")
    if len(completed_ids) != len(set(completed_ids)):
        raise PhysicalRuntimeError("a request completed more than once")
    if len(rejected_ids) != len(set(rejected_ids)):
        raise PhysicalRuntimeError("a request was rejected more than once")
    if set(completed_ids) & set(rejected_ids):
        raise PhysicalRuntimeError("a request has multiple terminal outcomes")
    if set(decided) != set(completed_ids):
        raise PhysicalRuntimeError("dispatch and completion ownership differ")
    if set(completed_ids) | set(rejected_ids) != expected:
        raise PhysicalRuntimeError("request conservation failed")

    by_request = {}
    for decision in decisions:
        request_ids = decision["request_ids"]
        route_epochs = decision["route_epochs"]
        priorities = decision["priorities"]
        if (
            len(request_ids) != decision["batch_size"]
            or len(route_epochs) != len(request_ids)
            or len(priorities) != len(request_ids)
        ):
            raise PhysicalRuntimeError("dispatch group cardinality changed")
        if decision["route_id"] == "R0":
            if decision["batch_size"] not in allowed_batches["R0"]:
                raise PhysicalRuntimeError("R0 used an unmeasured policy point")
        elif decision["route_id"] == "R2":
            if decision["batch_size"] not in allowed_batches["R2"]:
                raise PhysicalRuntimeError("R2 used an unmeasured policy point")
            if decision["predicted_cuda_relief_us"] <= 0:
                raise PhysicalRuntimeError("R2 dispatch has no predicted CUDA relief")
            if any(priority == 0 for priority in priorities):
                raise PhysicalRuntimeError("urgent work was sent to R2")
        else:
            raise PhysicalRuntimeError("unknown route in dispatch record")
        for request_id, route_epoch, priority in zip(
            request_ids, route_epochs, priorities,
        ):
            source = source_by_id.get(request_id)
            if source is None or priority != source["priority"]:
                raise PhysicalRuntimeError("dispatch priority differs from the trace")
            by_request[request_id] = (
                decision["route_id"], route_epoch, priority,
            )
    for row in completed:
        source = source_by_id[row["request_id"]]
        if by_request.get(row["request_id"]) != (
            row["route_id"], row["route_epoch"], row["priority"],
        ):
            raise PhysicalRuntimeError("completion lineage differs from dispatch")
        if (
            row["priority"] != source["priority"]
            or row["prompt_length"] != source["input_tokens"]
            or row["output_steps"] != source["output_steps"]
            or row["slo_us"] != source["slo_us"]
            or row["scheduled_arrival_ns"] != source["arrival_us"] * 1000
        ):
            raise PhysicalRuntimeError("completion workload differs from the trace")
        if row["slo_met"] is not (row["latency_us"] <= row["slo_us"]):
            raise PhysicalRuntimeError("stored SLO result is inconsistent")


def run_priority_workload(
    topology: Any,
    trace: dict[str, Any],
    controller: PriorityAdmissionController,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = prepare_trace(trace)
    source_by_id = {row.request_id: row for row in rows}
    origin_ns = time.monotonic_ns()
    next_arrival = 0
    completion_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    active: dict[int, threading.Thread] = {}
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    fatal: BaseException | None = None
    wait_decision = WaitDecision(None, "READY_QUEUE_EMPTY")

    def elapsed_us() -> int:
        return max(0, (time.monotonic_ns() - origin_ns) // 1000)

    def execute_group(
        group_rows: list[ScheduledWork],
        decision: DispatchGroup,
        batch_wait_us: int,
        admitted_ns: int,
    ) -> None:
        requests = tuple(
            RouteRequest(
                request_id=row.request_id,
                route_epoch=route_epoch,
                route_id=decision.route_id,
                prompt_tokens=tuple(
                    [row.source["synthetic_token"]] * row.source["input_tokens"]
                ),
                output_steps=row.source["output_steps"],
                slo_us=row.source["slo_us"],
                priority=row.source["priority"],
                batch_wait_us=batch_wait_us,
                prefill_chunk=args.prefill_chunk,
            )
            for row, route_epoch in zip(group_rows, decision.route_epochs)
        )
        request_ids = tuple(row.request_id for row in group_rows)
        try:
            outcomes = topology.runner.run_group(
                requests,
                args.request_timeout,
                scheduled_arrival_ns=tuple(
                    origin_ns + row.arrival_us * 1000 for row in group_rows
                ),
                capture_boundaries=False,
            )
            completion_queue.put({
                "kind": "completed",
                "request_ids": request_ids,
                "route_epochs": tuple(decision.route_epochs),
                "records": [
                    outcome_record(
                        outcome, row.source, origin_ns, admitted_ns,
                    )
                    for row, outcome in zip(group_rows, outcomes)
                ],
            })
        except BaseException as exc:
            completion_queue.put({
                "kind": "fatal",
                "request_ids": request_ids,
                "route_epochs": tuple(decision.route_epochs),
                "clean": all(
                    request_id not in topology.runner.pinned()
                    for request_id in request_ids
                ),
                "error": exc,
            })

    def consume(item: dict[str, Any]) -> None:
        nonlocal fatal
        request_ids = item["request_ids"]
        route_epochs = item["route_epochs"]
        for request_id in request_ids:
            active.pop(request_id, None)
        if item["kind"] == "fatal":
            if item["clean"]:
                try:
                    controller.complete_group(request_ids, route_epochs)
                except BaseException as exc:
                    fatal = PhysicalRuntimeError(
                        f"request failure plus reservation release failure: {exc}"
                    )
                    return
            fatal = item["error"]
            return
        try:
            controller.complete_group(request_ids, route_epochs)
        except BaseException as exc:
            fatal = exc
            return
        completed.extend(item["records"])

    while len(completed) + len(rejected) < len(rows):
        now_us = elapsed_us()
        while next_arrival < len(rows) and rows[next_arrival].arrival_us <= now_us:
            controller.enqueue(_work(rows[next_arrival]), now_us)
            next_arrival += 1

        while True:
            try:
                consume(completion_queue.get_nowait())
            except queue.Empty:
                break
        if fatal is not None:
            break

        admitted_any = False
        while True:
            now_us = elapsed_us()
            decision = controller.decide(now_us)
            if isinstance(decision, RejectDecision):
                source = source_by_id[decision.request_id].source
                rejected.append({
                    "request_id": decision.request_id,
                    "arrival_us": source["arrival_us"],
                    "rejected_us": now_us,
                    "priority": source["priority"],
                    "reason": decision.reason,
                })
                admitted_any = True
                continue
            if isinstance(decision, WaitDecision):
                wait_decision = decision
                break
            if not isinstance(decision, DispatchGroup):
                raise PhysicalRuntimeError("priority policy returned an unknown decision")

            admitted_ns = time.monotonic_ns()
            group_rows = [source_by_id[request_id] for request_id in decision.request_ids]
            decisions.append({
                "route_id": decision.route_id,
                "request_ids": list(decision.request_ids),
                "route_epochs": list(decision.route_epochs),
                "priorities": [row.source["priority"] for row in group_rows],
                "batch_size": decision.batch_size,
                "admitted_us": (admitted_ns - origin_ns) // 1000,
                "predicted_start_us": decision.predicted_start_us,
                "predicted_finish_us": decision.predicted_finish_us,
                "batch_wait_us": decision.batch_wait_us,
                "predicted_cuda_relief_us": decision.predicted_cuda_relief_us,
                "reason": decision.reason,
            })
            thread = threading.Thread(
                target=execute_group,
                args=(
                    group_rows,
                    decision,
                    decision.batch_wait_us,
                    admitted_ns,
                ),
                name="s26-group-" + "-".join(
                    str(row.request_id) for row in group_rows
                ),
            )
            for row in group_rows:
                active[row.request_id] = thread
            thread.start()
            admitted_any = True

        if len(completed) + len(rejected) == len(rows):
            break
        if not admitted_any:
            wait_s = 0.01
            now_us = elapsed_us()
            if next_arrival < len(rows):
                wait_s = min(
                    wait_s,
                    max(0.0, (rows[next_arrival].arrival_us - now_us) / 1e6),
                )
            if wait_decision.next_wake_us is not None:
                wait_s = min(
                    wait_s,
                    max(0.0, (wait_decision.next_wake_us - now_us) / 1e6),
                )
            if wait_s <= 0:
                wait_s = 0.001
            try:
                consume(completion_queue.get(timeout=wait_s))
            except queue.Empty:
                pass
            if fatal is not None:
                break

    for thread in set(active.values()):
        thread.join(timeout=args.request_timeout)
    if any(thread.is_alive() for thread in active.values()):
        raise TimeoutError("request thread did not terminate")
    while True:
        try:
            consume(completion_queue.get_nowait())
        except queue.Empty:
            break
    if fatal is not None:
        raise PhysicalRuntimeError("physical request failed") from fatal

    completed.sort(key=lambda row: row["request_id"])
    rejected.sort(key=lambda row: row["request_id"])
    validate_conservation(rows, decisions, completed, rejected)
    if controller.pending() or controller.active():
        raise PhysicalRuntimeError("priority controller retained request ownership")
    resources = controller.resource_state()
    if any(resources[name].get(resource, 0) for name in ("active", "low_priority_active") for resource in resources[name]):
        raise PhysicalRuntimeError("priority controller retained resource credits")
    if topology.runner.pinned():
        raise PhysicalRuntimeError("route runner retained a request pin")
    if any(pool.leased() for pool in topology.slots.values()):
        raise PhysicalRuntimeError("software sequence leases remain live")

    duration_ns = time.monotonic_ns() - origin_ns
    runtime = {
        "origin_monotonic_ns": origin_ns,
        "duration_ns": duration_ns,
        "selected_request_count": len(rows),
        "completed_count": len(completed),
        "rejected_count": len(rejected),
        "decisions": decisions,
        "requests": completed,
        "rejected": rejected,
        "energy": None,
        "nvml_samples": [],
        "nvml_errors": [],
    }
    state = {
        "runner_pins": topology.runner.pinned(),
        "software_leases": {
            name: pool.leased() for name, pool in topology.slots.items()
        },
        "priority_resources": resources,
        "priority_pending": [asdict(work) for work in controller.pending()],
        "priority_active": {
            str(request_id): asdict(reservation)
            for request_id, reservation in controller.active().items()
        },
    }
    return runtime, state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    result.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    result.add_argument("--trace", type=Path, required=True)
    result.add_argument("--profiles", type=Path, default=HERE / "profiles.json")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--control-mode", choices=("all-cuda", "priority"), default="priority",
    )
    result.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    result.add_argument("--allow-numeric-uncertified", action="store_true")
    result.add_argument("--prefill-chunk", type=int)
    result.add_argument("--queue-depth", type=int, default=1024)
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    return result


def validate_args(args: argparse.Namespace, arg_parser: argparse.ArgumentParser) -> None:
    if args.control_mode == "priority" and not args.allow_numeric_uncertified:
        arg_parser.error("phone routes require --allow-numeric-uncertified")
    if args.output.exists():
        arg_parser.error(f"output already exists: {args.output}")
    if args.queue_depth <= 0 or args.timeout <= 0 or args.request_timeout <= 0:
        arg_parser.error("runtime bounds must be positive")
    if args.prefill_chunk is not None and args.prefill_chunk <= 0:
        arg_parser.error("prefill chunk must be positive")


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    validate_args(args, arg_parser)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topology = None
    failure: BaseException | None = None
    try:
        trace = load_trace(args.trace)
        routes, capacities, reserve, profile_bundle = load_bundle(args.profiles)
        controller = PriorityAdmissionController(
            routes,
            capacities,
            reserve,
            offload_enabled=args.control_mode == "priority",
        )
        topology = build_topology(args)
        if physical_capacities(topology) != capacities:
            raise PhysicalRuntimeError("profile and physical capacities differ")
        runtime, software_state = run_priority_workload(
            topology, trace, controller, args,
        )
        topology.stop_batchers(args.timeout)
        batch_events = {
            name: list(batcher.events) for name, batcher in topology.batchers.items()
        }
        final_workers = topology.end_sessions(args.session_end)
        report = {
            "schema": SCHEMA,
            "status": "RUN_COMPLETE",
            "numeric_scope": (
                "Q8_CUDA_ONLY"
                if args.control_mode == "all-cuda"
                else "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED"
            ),
            "trace": {
                "path": str(args.trace),
                "sha256": "sha256:" + sha256_file(args.trace),
                "trace_hash": trace["trace_hash"],
                "scope": trace["scope"],
            },
            "profile": {
                "path": str(args.profiles),
                "sha256": "sha256:" + sha256_file(args.profiles),
                "schema": profile_bundle["schema"],
                "sources": profile_bundle["sources"],
            },
            "configuration": {
                "control_mode": args.control_mode,
                "session_end": args.session_end,
                "prefill_chunk": args.prefill_chunk,
                "queue_depth": args.queue_depth,
                "knees": {
                    "cuda-prefix": 4,
                    "cuda-mid": 4,
                    "op12-prefix": 4,
                    "op15-mid": 4,
                    "cuda-tail": 8,
                },
                "gather_us": {
                    "cuda-prefix": 5000,
                    "cuda-mid": 5000,
                    "op12-prefix": 5000,
                    "op15-mid": 5000,
                "cuda-tail-r0": 5000,
                    "cuda-tail-r2": 5000,
                },
                "tail_priority_queues": {
                    "cuda-tail-r0": {"batch_knee": 4, "gather_us": 5000},
                    "cuda-tail-r2": {"batch_knee": 4, "gather_us": 5000},
                },
                "endpoints": {
                    "cuda-prefix": list(args.cuda_prefix),
                    "cuda-mid": list(args.cuda_mid),
                    "op12-prefix": list(args.op12),
                    "op15-mid": list(args.op15),
                    "cuda-tail": list(args.cuda_tail),
                },
            },
            "workers": {name: asdict(hello) for name, hello in topology.hellos.items()},
            "runtime": runtime,
            "batch_events": batch_events,
            "summary": summarize_run(runtime, batch_events),
            "final_workers": final_workers,
            "final_software_state": software_state,
            "energy_exclusions": {
                "gpu_board_energy": "NOT_MEASURED_IN_S26_CP3",
                "phone_energy": "UNKNOWN",
                "network_energy": "UNKNOWN",
                "total_system_energy": "UNKNOWN",
            },
        }
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": report["status"],
            "completed": report["summary"]["completed_requests"],
            "slo_misses": report["summary"]["slo_misses"],
            "output": str(args.output),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, PriorityPolicyError, PriorityProfileError, BaseException) as exc:
        failure = exc
        failure_report = {
            "schema": SCHEMA,
            "status": "RUN_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.write_bytes(canonical_bytes(failure_report))
        except OSError:
            pass
        print(json.dumps(failure_report, sort_keys=True, separators=(",", ":")))
        return 2
    finally:
        if topology is not None:
            if not topology.batchers_stopped:
                try:
                    topology.stop_batchers(args.timeout)
                except BaseException:
                    pass
            if not topology.ended:
                try:
                    if all(
                        topology.clients[name].status().active_sequences == 0
                        for name in WORKER_NAMES
                    ):
                        topology.end_sessions("detach")
                except BaseException:
                    pass
            topology.close()
        if failure is not None:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
