#!/usr/bin/env python3
"""Measure exact S29 R0/R2 batch points on resident physical workers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S24 = HERE.parent / "s24_overlap_handoff_poc"
S26 = HERE.parent / "s26_priority_scheduler"
for dependency in (HERE, S26, S24, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from async_pipeline import parse_endpoint
from large_batch_adapter import build_topology, knees_from_args
from physical_adapter import (
    WORKER_NAMES,
    PhysicalRuntimeError,
    canonical_bytes,
    physical_capacities,
)
from route_runtime import RouteRequest


SCHEMA = "s29-route-calibration-v1"
ACTIVE = {
    "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
    "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
}


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        batches = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from exc
    if not batches or any(item <= 0 for item in batches) or len(set(batches)) != len(batches):
        raise argparse.ArgumentTypeError("batches must be positive and unique")
    return tuple(sorted(batches))


def add_topology_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    parser.add_argument("--op12", type=parse_endpoint, required=True)
    parser.add_argument("--op15", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-prefix-knee", type=int, default=32)
    parser.add_argument("--cuda-mid-knee", type=int, default=32)
    parser.add_argument("--op12-prefix-knee", type=int, default=32)
    parser.add_argument("--op15-mid-knee", type=int, default=32)
    parser.add_argument("--cuda-tail-knee", type=int, default=32)
    parser.add_argument("--gather-us", type=int, default=5000)
    parser.add_argument("--queue-depth", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)


def _events_since(topology: Any, starts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    return {
        name: list(topology.batchers[name].events[starts[name]:])
        for name in WORKER_NAMES
    }


def _validate_events(route_id: str, batch: int, events: dict[str, list[dict[str, Any]]]) -> None:
    for worker in WORKER_NAMES:
        rows = events[worker]
        expected = 4 if worker in ACTIVE[route_id] else 0
        if len(rows) != expected:
            raise PhysicalRuntimeError(
                f"{route_id} B{batch} emitted {len(rows)} batches on {worker}, expected {expected}"
            )
        for row in rows:
            if row.get("status") != "OK" or row.get("batch_size") != batch:
                raise PhysicalRuntimeError(
                    f"{route_id} B{batch} emitted an invalid {worker} batch"
                )


def _measure_one(
    topology: Any,
    route_id: str,
    batch: int,
    rep: int,
    next_identity: int,
    timeout_s: float,
) -> tuple[dict[str, Any], int]:
    starts = {name: len(topology.batchers[name].events) for name in WORKER_NAMES}
    requests = []
    for offset in range(batch):
        identity = next_identity + offset
        requests.append(RouteRequest(
            request_id=identity,
            route_epoch=identity,
            route_id=route_id,
            prompt_tokens=(2,),
            output_steps=4,
            slo_us=int(timeout_s * 1e6),
            priority=1,
            batch_wait_us=5000,
        ))
    start_ns = time.monotonic_ns()
    outcomes = topology.runner.run_group(
        tuple(requests), timeout_s, capture_boundaries=False,
    )
    end_ns = time.monotonic_ns()
    events = _events_since(topology, starts)
    _validate_events(route_id, batch, events)
    token_rows = [list(outcome.output_tokens) for outcome in outcomes]
    if len(token_rows) != batch or len({tuple(row) for row in token_rows}) != 1:
        raise PhysicalRuntimeError(f"{route_id} B{batch} output rows are inconsistent")
    cuda_workers = ACTIVE[route_id] & {"cuda-prefix", "cuda-mid", "cuda-tail"}
    cuda_work_us = sum(
        int(row["compute_us"])
        for worker in cuda_workers
        for row in events[worker]
    )
    return ({
        "route_id": route_id,
        "batch_size": batch,
        "rep": rep,
        "wall_us": (end_ns - start_ns) // 1000,
        "max_latency_us": max(outcome.latency_us for outcome in outcomes),
        "cuda_work_us": cuda_work_us,
        "output_tokens": token_rows[0],
        "events": events,
    }, next_identity + batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_topology_args(parser)
    parser.add_argument("--batches", type=parse_batches, default=(1, 4, 24, 32))
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.reps <= 0 or args.gather_us < 0:
        parser.error("reps must be positive and gather-us cannot be negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    topology = None
    try:
        topology = build_topology(args)
        capacities = physical_capacities(topology)
        if max(args.batches) > min(capacities["op12-prefix"], capacities["op15-mid"]):
            raise PhysicalRuntimeError("requested phone batch exceeds physical capacity")
        points = []
        next_identity = 1
        # Exercise the phone route before an otherwise idle HTP session can age.
        for route_id in ("R2", "R0"):
            for batch in args.batches:
                for rep in range(args.reps):
                    row, next_identity = _measure_one(
                        topology,
                        route_id,
                        batch,
                        rep,
                        next_identity,
                        args.request_timeout,
                    )
                    points.append(row)
                    print(
                        f"{route_id} B{batch} rep={rep} wall_us={row['wall_us']} "
                        f"cuda_work_us={row['cuda_work_us']}",
                        flush=True,
                    )
        if topology.runner.pinned() or any(pool.leased() for pool in topology.slots.values()):
            raise PhysicalRuntimeError("calibration retained request state")
        topology.stop_batchers(args.timeout)
        final_workers = topology.end_sessions(args.session_end)
        report = {
            "schema": SCHEMA,
            "status": "CALIBRATION_COMPLETE",
            "shape": {"input_tokens": 1, "output_steps": 4, "context": 16},
            "batches": list(args.batches),
            "reps": args.reps,
            "knees": knees_from_args(args),
            "capacities": capacities,
            "workers": {name: asdict(hello) for name, hello in topology.hellos.items()},
            "points": points,
            "final_workers": final_workers,
            "numeric_scope": "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        }
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": report["status"],
            "points": len(points),
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "status": "CALIBRATION_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.write_bytes(canonical_bytes(failure))
        except OSError:
            pass
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
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
                    topology.end_sessions("detach")
                except BaseException:
                    pass
            topology.close()


if __name__ == "__main__":
    raise SystemExit(main())
