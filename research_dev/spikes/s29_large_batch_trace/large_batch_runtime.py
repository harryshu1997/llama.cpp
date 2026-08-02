#!/usr/bin/env python3
"""Run the frozen 60-request trace with measured large route batches."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S24 = HERE.parent / "s24_overlap_handoff_poc"
S26 = HERE.parent / "s26_priority_scheduler"
S28 = HERE.parent / "s28_priority_shared_tail"
for dependency in (HERE, S28, S26, S24, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from async_pipeline import parse_endpoint
from large_batch_adapter import build_topology, knees_from_args, validate_shared_pair
from large_batch_profiles import load_bundle
from physical_adapter import (
    WORKER_NAMES,
    PhysicalRuntimeError,
    canonical_bytes,
    load_trace,
    physical_capacities,
    sha256_file,
    summarize_run,
)
from priority_policy import PriorityAdmissionController, PriorityWork, WaitDecision
from priority_shared_runtime import (
    run_priority_workload,
    validate_priority_events,
)


SCHEMA = "s29-large-batch-full-trace-v1"
MIN_PHONE_BATCH = 24


class LargeBatchAdmissionController(PriorityAdmissionController):
    """Use a useful phone batch or drain a finite trace to the CUDA route."""

    def __init__(
        self,
        *args: Any,
        finite_arrival_end_us: int,
        urgent_quiet_us: int,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if type(finite_arrival_end_us) is not int or finite_arrival_end_us < 0:
            raise ValueError("finite arrival end must be a nonnegative integer")
        if type(urgent_quiet_us) is not int or urgent_quiet_us < 0:
            raise ValueError("urgent quiet interval must be a nonnegative integer")
        self.finite_arrival_end_us = finite_arrival_end_us
        self.urgent_quiet_us = urgent_quiet_us
        self.last_urgent_arrival_us: int | None = None

    def enqueue(self, work: PriorityWork, now_us: int) -> None:
        super().enqueue(work, now_us)
        if work.priority <= self.urgent_priority_max:
            self.last_urgent_arrival_us = work.arrival_us

    def decide(self, now_us: int):
        ordered = list(self.pending())
        if (
            self.offload_enabled
            and ordered
            and not self._has_urgent()
            and self.last_urgent_arrival_us is not None
            and now_us < self.last_urgent_arrival_us + self.urgent_quiet_us
        ):
            return WaitDecision(
                self.last_urgent_arrival_us + self.urgent_quiet_us,
                "URGENT_QUIET_GUARD",
            )
        if (
            self.offload_enabled
            and now_us >= self.finite_arrival_end_us
            and ordered
            and not self._has_urgent()
        ):
            route = self.routes[self.offload_route_id]
            target = route.target()
            if len(ordered) < target.batch_size:
                current = route.largest_at_most(len(ordered))
                selected = ordered[:current.batch_size]
                relief = self._relief(current)
                if (
                    current.batch_size >= MIN_PHONE_BATCH
                    and relief > 0
                    and self._fits_deadlines(selected, now_us, current)
                    and self._can_reserve(route, len(selected), selected[0].priority)
                ):
                    return self._reserve(
                        route,
                        selected,
                        current,
                        now_us,
                        relief,
                        "FINITE_TRACE_LARGE_BATCH_DRAIN",
                    )
                return self._fallback(
                    ordered[0], now_us, "FINITE_TRACE_SMALL_REMAINDER_R0",
                )
        return super().decide(now_us)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    result.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    result.add_argument("--trace", type=Path, required=True)
    result.add_argument("--profiles", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--control-mode", choices=("all-cuda", "priority"), default="priority")
    result.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    result.add_argument("--allow-numeric-uncertified", action="store_true")
    result.add_argument("--prefill-chunk", type=int)
    result.add_argument("--queue-depth", type=int, default=4096)
    result.add_argument("--gather-us", type=int, default=5000)
    result.add_argument("--urgent-quiet-us", type=int, default=1_000_000)
    result.add_argument("--cuda-prefix-knee", type=int, default=32)
    result.add_argument("--cuda-mid-knee", type=int, default=32)
    result.add_argument("--op12-prefix-knee", type=int, default=32)
    result.add_argument("--op15-mid-knee", type=int, default=32)
    result.add_argument("--cuda-tail-knee", type=int, default=32)
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    return result


def validate_args(args: argparse.Namespace, arg_parser: argparse.ArgumentParser) -> None:
    if args.control_mode == "priority" and not args.allow_numeric_uncertified:
        arg_parser.error("phone routes require --allow-numeric-uncertified")
    if args.output.exists():
        arg_parser.error(f"output already exists: {args.output}")
    if (
        args.queue_depth <= 0
        or args.gather_us < 0
        or args.urgent_quiet_us < 0
        or args.timeout <= 0
        or args.request_timeout <= 0
    ):
        arg_parser.error("runtime bounds are invalid")
    if args.prefill_chunk is not None and args.prefill_chunk <= 0:
        arg_parser.error("prefill chunk must be positive")


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    validate_args(args, arg_parser)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topology = None
    try:
        trace = load_trace(args.trace)
        routes, capacities, reserve, profile_bundle = load_bundle(args.profiles)
        arrival_end_us = max(int(row["arrival_us"]) for row in trace["requests"])
        controller = LargeBatchAdmissionController(
            routes,
            capacities,
            reserve,
            offload_enabled=args.control_mode == "priority",
            finite_arrival_end_us=arrival_end_us,
            urgent_quiet_us=args.urgent_quiet_us,
        )
        topology = build_topology(args)
        validate_shared_pair(topology.routes)
        if physical_capacities(topology) != capacities:
            raise PhysicalRuntimeError("profile and physical capacities differ")
        runtime, software_state = run_priority_workload(
            topology,
            trace,
            controller,
            args,
            {
                route.route_id: frozenset(
                    point.batch_size for point in route.points
                )
                for route in routes
            },
        )
        topology.stop_batchers(args.timeout)
        batch_events = {
            name: list(batcher.events)
            for name, batcher in topology.batchers.items()
        }
        validate_priority_events(batch_events)
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
                "source": profile_bundle["source"],
            },
            "configuration": {
                "control_mode": args.control_mode,
                "session_end": args.session_end,
                "prefill_chunk": args.prefill_chunk,
                "queue_depth": args.queue_depth,
                "shared_tail": True,
                "urgent_background_batch_isolation": True,
                "minimum_phone_batch": MIN_PHONE_BATCH,
                "knees": knees_from_args(args),
                "gather_us": args.gather_us,
                "urgent_quiet_us": args.urgent_quiet_us,
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
                "gpu_board_energy": "NOT_MEASURED_IN_S29",
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
            "route_distribution": report["summary"]["route_distribution"],
            "output": str(args.output),
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
            args.output.write_bytes(canonical_bytes(failure))
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
