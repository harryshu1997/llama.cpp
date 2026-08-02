#!/usr/bin/env python3
"""Run every legal request-level handoff on the real S36 workers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
S36 = HERE.parent / "s36_dynamic_cut_scheduler"
if str(S36) not in sys.path:
    sys.path.insert(0, str(S36))

from physical_topology import WORKERS  # noqa: E402
from profile_routes import parse_endpoint, run_group  # noqa: E402

from arbitrary_topology import build_arbitrary_topology  # noqa: E402
from contract import SCHEMA, canonical_bytes, validate, with_digest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in WORKERS:
        parser.add_argument(f"--{name}", type=parse_endpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--gather-us", type=int, default=20_000)
    parser.add_argument("--queue-depth", type=int, default=4096)
    parser.add_argument("--knee", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    topology = None
    try:
        topology = build_arbitrary_topology(
            {name: getattr(args, name) for name in WORKERS},
            args.timeout,
            args.gather_us,
            args.queue_depth,
            args.knee,
        )
        rows = []
        signature: list[int] | None = None
        request_base = 2_000_000
        for route in topology.routes:
            repetitions = []
            for repetition in range(2):
                measurement = run_group(
                    topology,
                    route,
                    8,
                    repetition,
                    request_base,
                    args.timeout,
                    args.gather_us,
                )
                request_base += 100
                tokens = list(measurement["tokens"])
                if signature is None:
                    signature = tokens
                elif tokens != signature:
                    raise RuntimeError("arbitrary cuts returned different tokens")
                repetitions.append(measurement)
            rows.append({
                "route_id": route.route_id,
                "phone": route.head.name,
                "cut": route.cut,
                "repetitions": repetitions,
            })

        topology.stop_batchers(args.timeout)
        active_sequences = {
            name: topology.clients[name].status().active_sequences
            for name in WORKERS
        }
        final_state = {
            "route_pins": topology.runner.pins(),
            "software_leases": {
                name: topology.stages[name].slots.leased() for name in WORKERS
            },
            "active_sequences": active_sequences,
        }
        final_workers = topology.end_sessions("detach")
        result = with_digest({
            "schema": SCHEMA,
            "status": "PASS",
            "physical": True,
            "token_signature": signature or [],
            "workers": {
                name: asdict(topology.hellos[name]) for name in WORKERS
            },
            "routes": rows,
            "final_state": final_state,
            "final_workers": final_workers,
        })
        validate(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_bytes(result))
        print(json.dumps({
            "status": "PASS",
            "result_hash": result["result_hash"],
            "routes": len(rows),
            "output": str(args.output),
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
