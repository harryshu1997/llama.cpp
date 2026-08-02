#!/usr/bin/env python3
"""Measure one synchronized R0 or R2 B4 point on resident StageNet workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
    outcome_record,
    parse_endpoint,
    sha256_file,
    summarize_run,
)
from route_runtime import RouteRequest


SCHEMA = "s24-fixed-diamond-physical-v1"
TRACE = S24 / "deterministic-three-class.json"
TRACE_HASH = "sha256:9b4e84bdd5d6bf38bd8d951dd043ad5d7a1b22520f746ec2953ab547db81c8be"
REQUEST_IDS = {
    "R0": tuple(range(26401, 26405)),
    "R2": tuple(range(26411, 26415)),
}


def build_requests(route_id: str, batch_wait_us: int) -> tuple[RouteRequest, ...]:
    if route_id not in REQUEST_IDS:
        raise PhysicalRuntimeError("measurement route must be R0 or R2")
    priority = 0 if route_id == "R0" else 2
    slo_us = 2_000_000 if route_id == "R0" else 15_000_000
    return tuple(
        RouteRequest(
            request_id=request_id,
            route_epoch=index,
            route_id=route_id,
            prompt_tokens=(2,),
            output_steps=4,
            slo_us=slo_us,
            priority=priority,
            batch_wait_us=batch_wait_us,
            prefill_chunk=None,
        )
        for index, request_id in enumerate(REQUEST_IDS[route_id], 1)
    )


def flatten_batch_events(
    topology: Any, route_id: str,
) -> dict[str, list[dict[str, Any]]]:
    selected_tail = "cuda-tail-r0" if route_id == "R0" else "cuda-tail-r2"
    idle_tail = "cuda-tail-r2" if route_id == "R0" else "cuda-tail-r0"
    if topology.batchers[idle_tail].events:
        raise PhysicalRuntimeError("idle CUDA tail queue executed work")
    return {
        "cuda-prefix": list(topology.batchers["cuda-prefix"].events),
        "cuda-mid": list(topology.batchers["cuda-mid"].events),
        "op12-prefix": list(topology.batchers["op12-prefix"].events),
        "op15-mid": list(topology.batchers["op15-mid"].events),
        "cuda-tail": list(topology.batchers[selected_tail].events),
    }


def validate_measurement_shape(report: dict[str, Any]) -> None:
    runtime = report.get("runtime", {})
    requests = runtime.get("requests", [])
    route_id = report.get("route_point", {}).get("route_id")
    active = {
        "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
        "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
    }
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "RUN_COMPLETE"
        or route_id not in active
        or len(requests) != 4
        or runtime.get("completed_count") != 4
        or runtime.get("rejected_count") != 0
    ):
        raise PhysicalRuntimeError("R0 B4 measurement accounting changed")
    for row in requests:
        if (
            row.get("route_id") != route_id
            or row.get("prompt_length") != 1
            or row.get("output_steps") != 4
            or row.get("slo_met") is not True
        ):
            raise PhysicalRuntimeError("R0 B4 request shape changed")
    expected = {
        name: [4, 4, 4, 4] if name in active[route_id] else []
        for name in WORKER_NAMES
    }
    observed = {
        name: [event.get("batch_size") for event in events]
        for name, events in report.get("batch_events", {}).items()
    }
    if observed != expected:
        raise PhysicalRuntimeError(
            f"{route_id} B4 did not execute as four lockstep B4 steps"
        )
    if any(
        event.get("status") != "OK"
        for events in report["batch_events"].values()
        for event in events
    ):
        raise PhysicalRuntimeError("R0 B4 contains a failed physical batch")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    result.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    result.add_argument("--route", choices=("R0", "R2"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--session-end", choices=("detach", "stop"), default="stop")
    result.add_argument("--queue-depth", type=int, default=1024)
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    result.add_argument("--batch-wait-us", type=int, default=5000)
    return result


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    if args.output.exists():
        arg_parser.error(f"output already exists: {args.output}")
    if (
        args.queue_depth <= 0
        or args.timeout <= 0
        or args.request_timeout <= 0
        or args.batch_wait_us < 0
    ):
        arg_parser.error("measurement bounds are invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    topology = None
    try:
        topology = build_topology(args)
        origin_ns = time.monotonic_ns()
        outcomes = topology.runner.run_group(
            build_requests(args.route, args.batch_wait_us),
            args.request_timeout,
            scheduled_arrival_ns=(origin_ns,) * 4,
            capture_boundaries=False,
        )
        completed_ns = time.monotonic_ns()
        priority = 0 if args.route == "R0" else 2
        slo_us = 2_000_000 if args.route == "R0" else 15_000_000
        sources = [{
            "request_id": request_id,
            "input_tokens": 1,
            "output_steps": 4,
            "observed_input_tokens": 1,
            "observed_output_tokens": 4,
            "priority": priority,
            "slo_us": slo_us,
        } for request_id in REQUEST_IDS[args.route]]
        requests = [
            outcome_record(outcome, source, origin_ns, origin_ns)
            for outcome, source in zip(outcomes, sources)
        ]
        decisions = [{
            "request_id": request_id,
            "route_epoch": index,
            "route_id": args.route,
            "priority": priority,
        } for index, request_id in enumerate(REQUEST_IDS[args.route], 1)]
        runtime = {
            "origin_monotonic_ns": origin_ns,
            "duration_ns": completed_ns - origin_ns,
            "selected_request_count": 4,
            "completed_count": 4,
            "rejected_count": 0,
            "route_delays_us": {"R0": 0, "R1": 0, "R2": 0},
            "decisions": decisions,
            "requests": requests,
            "rejected": [],
            "energy": None,
            "nvml_samples": [],
            "nvml_errors": [],
        }
        topology.stop_batchers(args.timeout)
        batch_events = flatten_batch_events(topology, args.route)
        final_workers = topology.end_sessions(args.session_end)
        final_state = {
            "runner_pins": topology.runner.pinned(),
            "software_leases": {
                name: pool.leased() for name, pool in topology.slots.items()
            },
            "capacity_ledger": None,
            "policy_active": None,
        }
        report = {
            "schema": SCHEMA,
            "status": "RUN_COMPLETE",
            "control": "C0" if args.route == "R0" else "C2",
            "numeric_scope": (
                "Q8_CUDA_ONLY"
                if args.route == "R0"
                else "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED"
            ),
            "route_point": {
                "route_id": args.route,
                "batch_size": 4,
                "input_tokens": 1,
                "output_steps": 4,
            },
            "trace": {
                "path": str(TRACE),
                "sha256": "sha256:" + sha256_file(TRACE),
                "trace_hash": TRACE_HASH,
                "scope": "SYNTHETIC_THREE_CLASS_FIXED_ROUTE_PROOF",
            },
            "configuration": {
                "session_end": args.session_end,
                "route_filter": [args.route],
                "limit_per_route": 4,
                "batch_wait_us": args.batch_wait_us,
                "group_execution": f"LOCKSTEP_{args.route}_B4",
                "endpoints": {
                    "cuda-prefix": list(args.cuda_prefix),
                    "cuda-mid": list(args.cuda_mid),
                    "op12-prefix": list(args.op12),
                    "op15-mid": list(args.op15),
                    "cuda-tail": list(args.cuda_tail),
                },
            },
            "workers": {
                name: asdict(hello) for name, hello in topology.hellos.items()
            },
            "runtime": runtime,
            "batch_events": batch_events,
            "summary": summarize_run(runtime, batch_events),
            "final_workers": final_workers,
            "final_software_state": final_state,
            "energy_exclusions": {
                "gpu_board_energy": "NOT_MEASURED",
                "phone_energy": "UNKNOWN",
                "network_energy": "UNKNOWN",
                "total_system_energy": "UNKNOWN",
            },
        }
        validate_measurement_shape(report)
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": f"{args.route}_B4_MEASURED",
            "outputs": [list(outcome.output_tokens) for outcome in outcomes],
            "makespan_us": report["summary"]["makespan_us"],
            "cuda_work_us": report["summary"]["summed_cuda_island_compute_us"],
            "output": str(args.output),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as exc:
        try:
            args.output.write_bytes(canonical_bytes({
                "schema": SCHEMA,
                "status": "RUN_FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }))
        except OSError:
            pass
        print(json.dumps({
            "status": "RUN_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
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
                    topology.end_sessions("stop")
                except BaseException:
                    pass
            topology.close()


if __name__ == "__main__":
    raise SystemExit(main())
