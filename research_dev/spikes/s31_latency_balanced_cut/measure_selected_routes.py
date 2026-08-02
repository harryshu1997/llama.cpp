#!/usr/bin/env python3
"""Calibrate R0 and the selected S31 R2 route on physical workers."""

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
from cut_selector import canonical_bytes, load_json, sha256_file
from dynamic_cut_adapter import build_topology, knees_from_args
from measure_cut import validate_launch_evidence
from physical_adapter import WORKER_NAMES, PhysicalRuntimeError, physical_capacities
from route_runtime import RouteRequest
from selected_profiles import CALIBRATION_SCHEMA, EXPECTED_BATCHES


ACTIVE = {
    "R0": {"cuda-prefix", "cuda-mid", "cuda-tail"},
    "R2": {"op12-prefix", "op15-mid", "cuda-tail"},
}


def parse_batches(value: str) -> tuple[int, ...]:
    try:
        result = tuple(sorted(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batches must be comma-separated integers") from exc
    if result != EXPECTED_BATCHES:
        raise argparse.ArgumentTypeError("selected calibration requires 1,4,24,32")
    return result


def add_args(parser: argparse.ArgumentParser) -> None:
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


def _measure(topology: Any, route: str, batch: int, rep: int, identity: int, timeout: float):
    starts = {name: len(topology.batchers[name].events) for name in WORKER_NAMES}
    requests = tuple(RouteRequest(
        request_id=identity + offset,
        route_epoch=identity + offset,
        route_id=route,
        prompt_tokens=(2,),
        output_steps=4,
        slo_us=int(timeout * 1e6),
        priority=1,
        batch_wait_us=5000,
    ) for offset in range(batch))
    start_ns = time.monotonic_ns()
    outcomes = topology.runner.run_group(requests, timeout, capture_boundaries=False)
    end_ns = time.monotonic_ns()
    events = {
        name: list(topology.batchers[name].events[starts[name]:])
        for name in WORKER_NAMES
    }
    for name in WORKER_NAMES:
        expected = 4 if name in ACTIVE[route] else 0
        if len(events[name]) != expected or any(
            row.get("status") != "OK" or row.get("batch_size") != batch
            for row in events[name]
        ):
            raise PhysicalRuntimeError(f"{route} B{batch} invalid events on {name}")
    tokens = [list(outcome.output_tokens) for outcome in outcomes]
    if len(tokens) != batch or len({tuple(row) for row in tokens}) != 1:
        raise PhysicalRuntimeError(f"{route} B{batch} output rows are inconsistent")
    cuda_work = sum(
        int(row["compute_us"])
        for name in ACTIVE[route] & {"cuda-prefix", "cuda-mid", "cuda-tail"}
        for row in events[name]
    )
    return ({
        "route_id": route,
        "batch_size": batch,
        "rep": rep,
        "wall_us": (end_ns - start_ns) // 1000,
        "max_latency_us": max(outcome.latency_us for outcome in outcomes),
        "cuda_work_us": cuda_work,
        "output_tokens": tokens[0],
        "events": events,
    }, identity + batch)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_args(parser)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--phone-session-env", type=Path, required=True)
    parser.add_argument("--desktop-session-env", type=Path, required=True)
    parser.add_argument("--batches", type=parse_batches, default=EXPECTED_BATCHES)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.reps < 2 or args.gather_us != 5000:
        parser.error("selected calibration requires at least two reps and gather-us=5000")
    topology = None
    try:
        selection = load_json(args.selection)
        cut = selection.get("selected", {}).get("cut_layer")
        if selection.get("status") != "CUT_SELECTED" or type(cut) is not int:
            raise PhysicalRuntimeError("selection is invalid")
        args.phone_cut = cut
        launch = validate_launch_evidence(
            args.phone_session_env, args.desktop_session_env, cut,
        )
        topology = build_topology(args)
        capacities = physical_capacities(topology)
        if min(capacities.values()) < 32:
            raise PhysicalRuntimeError("worker capacity is below B32")
        points = []
        identity = 1
        measurement_order = []
        for route in ("R2", "R0"):
            for batch in reversed(args.batches):
                measurement_order.append(f"{route}:B{batch}")
                for rep in range(args.reps):
                    row, identity = _measure(
                        topology, route, batch, rep, identity, args.request_timeout,
                    )
                    points.append(row)
                    print(
                        f"{route} B{batch} rep={rep} wall_us={row['wall_us']} "
                        f"cuda_work_us={row['cuda_work_us']}",
                        flush=True,
                    )
        if topology.runner.pinned() or any(pool.leased() for pool in topology.slots.values()):
            raise PhysicalRuntimeError("calibration retained request state")
        topology.stop_batchers(args.timeout)
        final_workers = topology.end_sessions(args.session_end)
        report = {
            "schema": CALIBRATION_SCHEMA,
            "status": "CALIBRATION_COMPLETE",
            "selected_cut": cut,
            "selection": {
                "path": args.selection.name,
                "sha256": "sha256:" + sha256_file(args.selection),
            },
            "launch_evidence": launch,
            "shape": {"input_tokens": 1, "output_steps": 4, "context": 16},
            "batches": list(args.batches),
            "reps": args.reps,
            "gather_us": args.gather_us,
            "measurement_order": measurement_order,
            "knees": knees_from_args(args),
            "capacities": capacities,
            "workers": {name: asdict(hello) for name, hello in topology.hellos.items()},
            "points": points,
            "final_workers": final_workers,
            "numeric_scope": "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        }
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({"status": report["status"], "cut": cut, "points": len(points)}))
        return 0
    except BaseException as exc:
        failure = {
            "schema": CALIBRATION_SCHEMA,
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
