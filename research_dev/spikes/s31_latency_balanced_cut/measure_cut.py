#!/usr/bin/env python3
"""Measure one S31 phone cut with resident physical workers."""

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
from cut_selector import SCHEMA, canonical_bytes, sha256_file
from dynamic_cut_adapter import build_topology, knees_from_args, phone_cut
from physical_adapter import WORKER_NAMES, PhysicalRuntimeError, physical_capacities
from route_runtime import RouteRequest


ACTIVE = {"op12-prefix", "op15-mid", "cuda-tail"}


def add_topology_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    parser.add_argument("--op12", type=parse_endpoint, required=True)
    parser.add_argument("--op15", type=parse_endpoint, required=True)
    parser.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    parser.add_argument("--phone-cut", type=int, required=True)
    parser.add_argument("--cuda-prefix-knee", type=int, default=32)
    parser.add_argument("--cuda-mid-knee", type=int, default=32)
    parser.add_argument("--op12-prefix-knee", type=int, default=32)
    parser.add_argument("--op15-mid-knee", type=int, default=32)
    parser.add_argument("--cuda-tail-knee", type=int, default=32)
    parser.add_argument("--gather-us", type=int, default=50000)
    parser.add_argument("--queue-depth", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)


def parse_env(path: Path) -> dict[str, str]:
    result = {}
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise PhysicalRuntimeError(f"cannot read session metadata: {exc}") from exc
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            raise PhysicalRuntimeError("session metadata is malformed")
        result[key] = value
    return result


def validate_launch_evidence(
    phone_path: Path, desktop_path: Path, cut: int,
) -> dict[str, str]:
    phone = parse_env(phone_path)
    desktop = parse_env(desktop_path)
    if (
        phone.get("schema") != "s24-a6000-phone-session-v1"
        or phone.get("runtime_activation_relay") != "DESKTOP_DIRECT_WIFI"
        or phone.get("max_streams") != "32"
        or phone.get("context") != "16"
        or phone.get("op12_layer_range") != f"0:{cut}"
        or phone.get("op15_layer_range") != f"{cut}:8"
    ):
        raise PhysicalRuntimeError("phone session metadata differs from the candidate")
    if (
        desktop.get("schema") != "s24-desktop-cuda-session-v1"
        or desktop.get("context") != "16"
        or desktop.get("prefix_streams") != "32"
        or desktop.get("mid_streams") != "32"
        or desktop.get("tail_streams") != "32"
        or desktop.get("prefix_layer_range") != "0:6"
        or desktop.get("mid_layer_range") != "6:8"
        or desktop.get("tail_layer_range") != "8:48"
    ):
        raise PhysicalRuntimeError("desktop session metadata differs from S31")
    for key in ("op12_head_sha256", "op15_mid_sha256"):
        value = phone.get(key, "")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise PhysicalRuntimeError(f"{key} is not a SHA-256 digest")
    desktop_model_sha256 = desktop.get("model_sha256", "")
    if (
        len(desktop_model_sha256) != 64
        or any(char not in "0123456789abcdef" for char in desktop_model_sha256)
    ):
        raise PhysicalRuntimeError("desktop model hash is not a SHA-256 digest")
    if len({
        phone["op12_head_sha256"],
        phone["op15_mid_sha256"],
        desktop_model_sha256,
    }) != 1:
        raise PhysicalRuntimeError("desktop and phone model identities differ")
    return {
        "activation_relay": phone["runtime_activation_relay"],
        "phone_session_env": "sha256:" + sha256_file(phone_path),
        "desktop_session_env": "sha256:" + sha256_file(desktop_path),
        "op12_model_sha256": phone["op12_head_sha256"],
        "op15_model_sha256": phone["op15_mid_sha256"],
        "desktop_model_sha256": desktop_model_sha256,
        "phone_binary_sha256": phone.get("phone_binary_sha256", "UNKNOWN"),
    }


def events_since(topology: Any, starts: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    return {
        name: list(topology.batchers[name].events[starts[name]:])
        for name in WORKER_NAMES
    }


def validate_events(events: dict[str, list[dict[str, Any]]]) -> None:
    for worker in WORKER_NAMES:
        rows = events[worker]
        expected = 4 if worker in ACTIVE else 0
        if len(rows) != expected:
            raise PhysicalRuntimeError(
                f"B32 emitted {len(rows)} batches on {worker}, expected {expected}"
            )
        if any(
            row.get("status") != "OK"
            or row.get("batch_size") != 32
            or type(row.get("compute_us")) is not int
            or row["compute_us"] <= 0
            for row in rows
        ):
            raise PhysicalRuntimeError(f"B32 emitted an invalid {worker} batch")


def measure_one(topology: Any, rep: int, next_identity: int, timeout_s: float):
    starts = {name: len(topology.batchers[name].events) for name in WORKER_NAMES}
    requests = tuple(RouteRequest(
        request_id=next_identity + offset,
        route_epoch=next_identity + offset,
        route_id="R2",
        prompt_tokens=(2,),
        output_steps=4,
        slo_us=int(timeout_s * 1e6),
        priority=2,
        batch_wait_us=None,
    ) for offset in range(32))
    start_ns = time.monotonic_ns()
    outcomes = topology.runner.run_group(requests, timeout_s, capture_boundaries=False)
    end_ns = time.monotonic_ns()
    events = events_since(topology, starts)
    validate_events(events)
    tokens = [list(outcome.output_tokens) for outcome in outcomes]
    if len(tokens) != 32 or len({tuple(row) for row in tokens}) != 1:
        raise PhysicalRuntimeError("B32 output rows are inconsistent")
    return ({
        "rep": rep,
        "wall_us": (end_ns - start_ns) // 1000,
        "max_latency_us": max(outcome.latency_us for outcome in outcomes),
        "completed_requests": len(outcomes),
        "output_tokens": tokens[0],
        "events": events,
    }, next_identity + 32)


def end_sessions(topology: Any, cuda_mode: str) -> dict[str, Any]:
    final = {}
    for name in WORKER_NAMES:
        status = topology.clients[name].status()
        if status.active_sequences != 0:
            raise PhysicalRuntimeError(f"{name} has live KV before drain")
        drained = topology.clients[name].drain()
        if drained.active_sequences != 0 or not drained.draining:
            raise PhysicalRuntimeError(f"{name} drain failed")
        final[name] = {
            "before_drain": asdict(status),
            "after_drain": asdict(drained),
            "session_end": "STOP" if name.startswith("op") else cuda_mode.upper(),
        }
    for name in WORKER_NAMES:
        mode = "stop" if name.startswith("op") else cuda_mode
        getattr(topology.clients[name], mode)()
    topology.ended = True
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    add_topology_args(parser)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--phone-session-env", type=Path, required=True)
    parser.add_argument("--desktop-session-env", type=Path, required=True)
    parser.add_argument("--cuda-session-end", choices=("detach", "stop"), default="detach")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.reps < 2 or args.gather_us < 0:
        parser.error("at least two reps are required and gather-us cannot be negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    topology = None
    try:
        cut = phone_cut(args)
        launch = validate_launch_evidence(
            args.phone_session_env, args.desktop_session_env, cut,
        )
        topology = build_topology(args)
        capacities = physical_capacities(topology)
        if min(capacities.values()) < 32:
            raise PhysicalRuntimeError("a worker cannot execute B32")
        repetitions = []
        next_identity = 1
        for rep in range(args.reps):
            row, next_identity = measure_one(
                topology, rep, next_identity, args.request_timeout,
            )
            repetitions.append(row)
            print(
                f"cut={cut} rep={rep} wall_us={row['wall_us']} "
                f"op12_us={[e['compute_us'] for e in row['events']['op12-prefix']]} "
                f"op15_us={[e['compute_us'] for e in row['events']['op15-mid']]}",
                flush=True,
            )
        if topology.runner.pinned() or any(pool.leased() for pool in topology.slots.values()):
            raise PhysicalRuntimeError("measurement retained request state")
        software_state = {
            "runner_pins": topology.runner.pinned(),
            "software_leases": {
                name: pool.leased() for name, pool in topology.slots.items()
            },
        }
        topology.stop_batchers(args.timeout)
        final_workers = end_sessions(topology, args.cuda_session_end)
        report = {
            "schema": SCHEMA,
            "status": "MEASUREMENT_COMPLETE",
            "candidate": {
                "cut_layer": cut,
                "op12_layers": [0, cut],
                "op15_layers": [cut, 8],
                "cuda_tail_layers": [8, 48],
            },
            "shape": {
                "input_tokens": 1,
                "output_steps": 4,
                "context": 16,
                "batch_size": 32,
            },
            "reps": args.reps,
            "gather_us": args.gather_us,
            "knees": knees_from_args(args),
            "capacities": capacities,
            "launch_evidence": launch,
            "workers": {name: asdict(hello) for name, hello in topology.hellos.items()},
            "repetitions": repetitions,
            "final_workers": final_workers,
            "final_software_state": software_state,
            "numeric_scope": "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED",
        }
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": report["status"],
            "cut_layer": cut,
            "output": str(args.output),
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        failure = {
            "schema": SCHEMA,
            "status": "MEASUREMENT_FAILED",
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
                    end_sessions(topology, "detach")
                except BaseException:
                    pass
            topology.close()


if __name__ == "__main__":
    raise SystemExit(main())
