#!/usr/bin/env python3
"""Measure a host-scheduled route with direct phone-to-phone activations."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from async_pipeline import DeviceBatcher, parse_endpoint, parse_tokens
from stage_v3_client import (
    BatchResult,
    BatchRow,
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


def run_request(
    request_id: int,
    seq_id: int,
    steps: int,
    slo_ms: float,
    prompt_tokens: tuple[int, ...],
    prefill_chunk: int,
    batcher: DeviceBatcher,
    barrier: threading.Barrier,
    timeout_s: float,
) -> dict:
    barrier.wait(timeout=timeout_s)
    start_ns = time.monotonic_ns()
    terminal_results: list[BatchResult] = []
    for start in range(0, len(prompt_tokens), prefill_chunk):
        stop = min(start + prefill_chunk, len(prompt_tokens))
        futures: list[Future[BatchResult]] = [
            batcher.submit(
                BatchRow(
                    request_id,
                    1,
                    seq_id,
                    position,
                    prompt_tokens[position],
                ),
                timeout_s,
            )
            for position in range(start, stop)
        ]
        terminal_results.extend(
            future.result(timeout=timeout_s) for future in futures
        )
    if any(
        result.token is None or result.hidden is not None
        for result in terminal_results
    ):
        raise ProtocolError("direct route returned a nonterminal prefill result")
    next_token = terminal_results[-1].token
    if next_token is None:
        raise ProtocolError("direct route omitted first generated token")
    generated = [next_token]

    for output_index in range(1, steps):
        position = len(prompt_tokens) + output_index - 1
        result = batcher.submit(
            BatchRow(request_id, 1, seq_id, position, next_token),
            timeout_s,
        ).result(timeout=timeout_s)
        if result.token is None or result.hidden is not None:
            raise ProtocolError("direct route returned a nonterminal decode result")
        next_token = result.token
        generated.append(next_token)

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    return {
        "elapsed_ms": elapsed_ms,
        "prefill_chunk": prefill_chunk,
        "prompt_length": len(prompt_tokens),
        "request_id": request_id,
        "route_epoch": 1,
        "slo_met": elapsed_ms <= slo_ms,
        "slo_ms": slo_ms,
        "tokens": generated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--relay", type=parse_endpoint, required=True)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--gather-us", type=int, default=20000)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--slo-ms", type=float, default=120000.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--prompt-tokens", type=parse_tokens, required=True)
    parser.add_argument("--prefill-chunk", type=int, required=True)
    parser.add_argument("--expected-tokens", type=parse_tokens, required=True)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.route_id:
        parser.error("route ID must be nonempty")
    if args.requests <= 0 or args.steps <= 0 or args.gather_us < 0:
        parser.error("request, step, and gather bounds are invalid")
    if args.queue_depth <= 0 or args.slo_ms <= 0 or args.timeout <= 0:
        parser.error("queue, SLO, and timeout bounds are invalid")
    if args.prefill_chunk <= 0 or args.prefill_chunk > len(args.prompt_tokens):
        parser.error("prefill chunk is outside the prompt")
    if len(args.expected_tokens) != args.steps:
        parser.error("expected token count must equal steps")

    client: StageV3Client | None = None
    batcher: DeviceBatcher | None = None
    try:
        client = StageV3Client.connect(*args.relay, args.timeout)
        hello = client.hello()
        if (
            hello.layer_start != 0
            or hello.layer_end != hello.n_layer
            or not hello.capabilities & STAGE_V3_CAP_TERMINAL
        ):
            raise ProtocolError("direct relay does not present a terminal route")
        if hello.max_streams < args.requests:
            raise ProtocolError("direct relay lacks sequence capacity")
        batcher = DeviceBatcher(
            "direct",
            client,
            min(hello.n_batch, hello.n_ubatch),
            args.gather_us,
            args.queue_depth,
        )
        barrier = threading.Barrier(args.requests)
        outcomes: list[dict | None] = [None] * args.requests
        errors: list[BaseException] = []
        error_lock = threading.Lock()

        def target(index: int) -> None:
            try:
                outcomes[index] = run_request(
                    3001 + index,
                    index,
                    args.steps,
                    args.slo_ms,
                    args.prompt_tokens,
                    args.prefill_chunk,
                    batcher,
                    barrier,
                    args.timeout,
                )
            except BaseException as error:
                with error_lock:
                    errors.append(error)

        threads = [
            threading.Thread(target=target, args=(index,), name=f"direct-{index}")
            for index in range(args.requests)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.timeout * args.steps)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("direct route request threads did not finish")
        if errors:
            raise RuntimeError("direct route failed") from errors[0]
        completed = [result for result in outcomes if result is not None]
        if len(completed) != args.requests:
            raise RuntimeError("direct route request conservation failed")

        batcher.stop(args.timeout)
        events = list(batcher.events)
        batcher = None
        for index in range(args.requests):
            client.remove(index, 3001 + index, 1)
        status = client.drain()
        if status.active_sequences != 0:
            raise ProtocolError("direct route left live sequences")

        expected = list(args.expected_tokens)
        token_exact = all(result["tokens"] == expected for result in completed)
        latencies = [result["elapsed_ms"] for result in completed]
        verdict = (
            "PASS"
            if token_exact and all(result["slo_met"] for result in completed)
            else "FAIL"
        )
        report = {
            "schema": "s39-direct-route-probe-v1",
            "verdict": verdict,
            "route_id": args.route_id,
            "relay_endpoint": {
                "host": args.relay[0],
                "port": args.relay[1],
            },
            "configuration": {
                "expected_tokens": expected,
                "gather_us": args.gather_us,
                "prefill_chunk": args.prefill_chunk,
                "prompt_tokens": list(args.prompt_tokens),
                "requests": args.requests,
                "slo_ms": args.slo_ms,
                "steps": args.steps,
            },
            "worker": asdict(hello),
            "latency_ms": {
                "p50": nearest_rank(latencies, 1, 2),
                "p95": nearest_rank(latencies, 19, 20),
                "max": max(latencies),
            },
            "requests": completed,
            "batches": {
                "batch_count": len(events),
                "batch_sizes": [event["batch_size"] for event in events],
                "max_batch": max(event["batch_size"] for event in events),
            },
            "batch_events": events,
            "tokens_exact": token_exact,
            "transport": {
                "direct_activation_payload_bytes": sum(
                    event["batch_size"] for event in events
                )
                * hello.n_embd
                * 4,
                "host_activation_payload_bytes": 0,
                "mode": "OP15_TO_OP12_DIRECT_WIFI",
            },
        }
        if args.session_end == "stop":
            client.stop()
        else:
            client.detach()
        client.close()
        client = None
        raw = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(raw, encoding="ascii")
        print(raw, end="")
        return 0 if verdict == "PASS" else 2
    finally:
        if batcher is not None:
            try:
                batcher.stop(args.timeout)
            except BaseException:
                pass
        if client is not None:
            client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as error:
        print(
            json.dumps(
                {"error": str(error), "verdict": "FAIL"},
                sort_keys=True,
            )
        )
        raise SystemExit(2)
