#!/usr/bin/env python3
"""Measure one StageNet V3 head-to-tail route with a fixed live cohort."""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import asdict
from pathlib import Path

from async_pipeline import (
    DeviceBatcher,
    RequestSpec,
    parse_endpoint,
    parse_tokens,
    run_request,
    summarize_batches,
)
from stage_v3_client import (
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


def nearest_rank(values: list[float], numerator: int, denominator: int) -> float:
    if not values or numerator <= 0 or numerator > denominator or denominator <= 0:
        raise ValueError("invalid nearest-rank input")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--head-name", required=True)
    parser.add_argument("--head", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--gather-us", type=int, default=5000)
    parser.add_argument("--queue-depth", type=int, default=32)
    parser.add_argument("--slo-ms", type=float, default=30000.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=parse_tokens)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.route_id or not args.head_name:
        parser.error("route identifiers must be nonempty")
    if args.requests <= 0 or args.steps <= 0 or args.gather_us < 0:
        parser.error("request, step, and gather bounds are invalid")
    if args.queue_depth <= 0 or args.slo_ms <= 0 or args.timeout <= 0:
        parser.error("queue, SLO, and timeout bounds are invalid")
    if args.prefill_chunk is not None and args.prefill_chunk <= 0:
        parser.error("prefill chunk must be positive")

    clients: dict[str, StageV3Client] = {}
    batchers: dict[str, DeviceBatcher] = {}
    try:
        clients["head"] = StageV3Client.connect(*args.head, args.timeout)
        clients["tail"] = StageV3Client.connect(*args.tail, args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        head_hello = hellos["head"]
        tail_hello = hellos["tail"]
        if head_hello.layer_start != 0 or head_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("probe head is not a prefix stage")
        if not tail_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("probe tail is not terminal")
        if head_hello.layer_end != tail_hello.layer_start:
            raise ProtocolError("probe join boundary mismatch")
        if head_hello.max_streams < args.requests or tail_hello.max_streams < args.requests:
            raise ProtocolError("probe cohort exceeds sequence capacity")

        for name, client in clients.items():
            hello = hellos[name]
            batchers[name] = DeviceBatcher(
                name, client,
                min(hello.n_batch, hello.n_ubatch),
                args.gather_us, args.queue_depth,
            )

        specs = [
            RequestSpec(2001 + index, 1, args.head_name, index, index,
                        args.steps, args.slo_ms, None, args.prompt_tokens,
                        args.prefill_chunk)
            for index in range(args.requests)
        ]
        barrier = threading.Barrier(len(specs))
        outcomes: list[dict | None] = [None] * len(specs)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def target(index: int, spec: RequestSpec) -> None:
            try:
                outcomes[index] = run_request(
                    spec, batchers["head"], batchers["tail"], args.token,
                    barrier, args.timeout,
                )
            except BaseException as exc:
                with errors_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=target, args=(index, spec),
                             name=f"probe-{spec.request_id}")
            for index, spec in enumerate(specs)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.timeout * args.steps)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("probe request threads did not finish")
        if errors:
            raise RuntimeError("probe route failed") from errors[0]
        completed = [outcome for outcome in outcomes if outcome is not None]
        if len(completed) != len(specs):
            raise RuntimeError("probe request conservation failed")

        for batcher in batchers.values():
            batcher.stop(args.timeout)
        events = {name: list(batcher.events) for name, batcher in batchers.items()}
        batchers.clear()

        for spec in specs:
            clients["head"].remove(spec.head_seq, spec.request_id, spec.route_epoch)
            clients["tail"].remove(spec.tail_seq, spec.request_id, spec.route_epoch)
        statuses = {name: client.drain() for name, client in clients.items()}
        if any(status.active_sequences != 0 for status in statuses.values()):
            raise ProtocolError("probe left live sequences")

        latencies = [outcome["elapsed_ms"] for outcome in completed]
        tokens = {tuple(outcome["tokens"]) for outcome in completed}
        verdict = "PASS" if len(tokens) == 1 and all(
            outcome["slo_met"] for outcome in completed
        ) else "FAIL"
        report = {
            "schema": "s22-route-probe-v1",
            "verdict": verdict,
            "route_id": args.route_id,
            "head_name": args.head_name,
            "configuration": {
                "gather_us": args.gather_us,
                "requests": args.requests,
                "slo_ms": args.slo_ms,
                "steps": args.steps,
                "token": args.token,
                "prompt_tokens": list(args.prompt_tokens) if args.prompt_tokens else None,
                "prefill_chunk": args.prefill_chunk,
            },
            "workers": {name: asdict(hello) for name, hello in hellos.items()},
            "latency_ms": {
                "p50": nearest_rank(latencies, 1, 2),
                "p95": nearest_rank(latencies, 19, 20),
                "max": max(latencies),
            },
            "requests": completed,
            "batches": {name: summarize_batches(rows) for name, rows in events.items()},
            "batch_events": events,
            "tokens_equal": len(tokens) == 1,
        }
        for client in clients.values():
            if args.session_end == "stop":
                client.stop()
            else:
                client.detach()
        data = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="ascii")
        print(data, end="")
        return 0 if verdict == "PASS" else 2
    finally:
        for batcher in batchers.values():
            try:
                batcher.stop(args.timeout)
            except BaseException:
                pass
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
