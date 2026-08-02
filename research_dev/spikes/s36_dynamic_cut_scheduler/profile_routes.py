#!/usr/bin/env python3
"""Measure the five finite S36 routes on the physical workers."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dynamic_route_runtime import DynamicRequest, DynamicRoute
from physical_topology import WORKERS, PhysicalTopology, build_topology
from profiles import SCHEMA, canonical_bytes, with_profile_hash


class ProfilerError(RuntimeError):
    pass


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("endpoint port is out of range")
    return host, port


def nearest_rank(values: list[int], numerator: int, denominator: int) -> int:
    if not values:
        raise ProfilerError("cannot summarize an empty measurement")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def run_group(
    topology: PhysicalTopology,
    route: DynamicRoute,
    batch_size: int,
    repetition: int,
    request_base: int,
    timeout_s: float,
    gather_us: int,
) -> dict[str, object]:
    barrier = threading.Barrier(batch_size + 1)
    results: queue.Queue[tuple[str, object]] = queue.Queue()
    threads = []

    def execute(index: int) -> None:
        request_id = request_base + index
        request = DynamicRequest(
            request_id=request_id,
            route_epoch=repetition + 1,
            route_id=route.route_id,
            prompt_tokens=(2, 2, 2, 2),
            output_steps=4,
            priority=1,
            slo_us=int(timeout_s * 1e6),
            batch_wait_us=gather_us,
        )
        try:
            barrier.wait(timeout=timeout_s)
            arrival_ns = time.monotonic_ns()
            outcome = topology.runner.run(
                request, timeout_s, scheduled_arrival_ns=arrival_ns,
            )
            results.put(("ok", outcome))
        except BaseException as exc:
            results.put(("error", exc))

    before = {
        name: len(topology.batchers[name].events) for name in WORKERS
    }
    for index in range(batch_size):
        thread = threading.Thread(
            target=execute,
            args=(index,),
            name=f"profile-{route.route_id}-{batch_size}-{index}",
        )
        thread.start()
        threads.append(thread)
    started_ns = time.monotonic_ns()
    barrier.wait(timeout=timeout_s)
    for thread in threads:
        thread.join(timeout_s)
    if any(thread.is_alive() for thread in threads):
        raise TimeoutError("profile request thread did not terminate")
    records = [results.get_nowait() for _thread in threads]
    errors = [value for kind, value in records if kind == "error"]
    if errors:
        raise ProfilerError(
            f"{route.route_id} B{batch_size} failed with {type(errors[0]).__name__}: {errors[0]}"
        ) from errors[0]
    outcomes = [value for kind, value in records if kind == "ok"]
    outcomes.sort(key=lambda outcome: outcome.request_id)
    tokens = {outcome.output_tokens for outcome in outcomes}
    if len(tokens) != 1:
        raise ProfilerError("identical profile requests returned different tokens")
    events = {
        name: topology.batchers[name].events[before[name]:]
        for name in WORKERS
    }
    latencies = [outcome.latency_us for outcome in outcomes]
    return {
        "batch_size": batch_size,
        "repetition": repetition,
        "request_ids": [outcome.request_id for outcome in outcomes],
        "latency_us": latencies,
        "p95_latency_us": nearest_rank(latencies, 95, 100),
        "max_latency_us": max(latencies),
        "makespan_us": (time.monotonic_ns() - started_ns) // 1000,
        "tokens": list(next(iter(tokens))),
        "events": events,
    }


def profile(
    topology: PhysicalTopology,
    repetitions: int,
    timeout_s: float,
    gather_us: int,
) -> dict[str, object]:
    if repetitions < 2:
        raise ValueError("at least two profile repetitions are required")
    route_rows = []
    token_signature: tuple[int, ...] | None = None
    request_base = 1_000_000
    for route in topology.routes:
        measurements = []
        for batch_size in (8, 32):
            for repetition in range(repetitions):
                measurement = run_group(
                    topology,
                    route,
                    batch_size,
                    repetition,
                    request_base,
                    timeout_s,
                    gather_us,
                )
                request_base += batch_size + 100
                signature = tuple(measurement["tokens"])
                if token_signature is None:
                    token_signature = signature
                elif signature != token_signature:
                    raise ProfilerError("finite routes are not token-consistent")
                measurements.append(measurement)
        points = []
        for batch_size in (8, 32):
            selected = [
                row for row in measurements if row["batch_size"] == batch_size
            ]
            values = [
                value for row in selected for value in row["latency_us"]
            ]
            points.append({
                "batch_size": batch_size,
                "repetitions": repetitions,
                "sample_count": len(values),
                "median_latency_us": int(statistics.median(values)),
                "p95_latency_us": nearest_rank(values, 95, 100),
                "max_latency_us": max(values),
                "token_consistent": True,
            })
        predicted = max(point["max_latency_us"] for point in points)
        route_rows.append({
            "route_id": route.route_id,
            "device": route.head.name,
            "cut": route.cut,
            "batch_points": points,
            "predicted_p95_us": predicted,
            "safety_margin_us": (predicted * 20 + 99) // 100,
            "measured": True,
            "eligible": True,
            "measurements": measurements,
        })
    return {
        "schema": SCHEMA,
        "physical": True,
        "model_scope": "GEMMA4_12B_F16_LOGICAL_MODEL",
        "prediction_rule": "MAX_OBSERVED_B8_B32_PLUS_20_PERCENT",
        "token_signature": list(token_signature or ()),
        "workers": {
            name: asdict(topology.hellos[name]) for name in WORKERS
        },
        "routes": route_rows,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--tail", type=parse_endpoint, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--repetitions", type=int, default=2)
    result.add_argument("--gather-us", type=int, default=20_000)
    result.add_argument("--queue-depth", type=int, default=4096)
    result.add_argument("--phone-knee", type=int, default=32)
    result.add_argument("--cuda-knee", type=int, default=32)
    result.add_argument("--tail-knee", type=int, default=32)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.output.exists():
        parser().error(f"output already exists: {args.output}")
    topology = None
    try:
        topology = build_topology(
            {name: getattr(args, name) for name in WORKERS},
            args.timeout,
            args.gather_us,
            args.queue_depth,
            args.phone_knee,
            args.cuda_knee,
            args.tail_knee,
        )
        bundle = with_profile_hash(profile(
            topology, args.repetitions, args.timeout, args.gather_us,
        ))
        topology.stop_batchers(args.timeout)
        final_workers = topology.end_sessions(args.session_end)
        bundle_without_hash = dict(bundle)
        del bundle_without_hash["profile_hash"]
        bundle_without_hash["final_workers"] = final_workers
        bundle = with_profile_hash(bundle_without_hash)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(bundle))
        print(json.dumps({
            "status": "PASS",
            "output": str(args.output),
            "profile_hash": bundle["profile_hash"],
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as exc:
        print(json.dumps({
            "status": "FAIL",
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

