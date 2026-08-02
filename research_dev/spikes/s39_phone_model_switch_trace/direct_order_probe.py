#!/usr/bin/env python3
"""Compare canonical and permuted row order on one direct phone route."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import Future
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from mixed_phase_batcher import PHASE_DECODE, PHASE_PREFILL, MixedPhaseBatcher, PhaseRow
from stage_v3_client import (
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)
from async_pipeline import parse_endpoint, parse_tokens


REQUEST_BASE = 6001
ROUTE_EPOCH = 1
FROZEN_REQUESTS = 32
FROZEN_STEPS = 8
SHUFFLE_MULTIPLIER = 17
SHUFFLE_OFFSET = 11


def nearest_rank(
    values: Sequence[float],
    numerator: int,
    denominator: int,
) -> float:
    if not values or numerator <= 0 or numerator > denominator:
        raise ValueError("invalid nearest-rank input")
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[rank - 1]


def sequence_order(kind: str, requests: int = FROZEN_REQUESTS) -> list[int]:
    if kind == "sorted":
        return list(range(requests))
    if kind != "shuffled":
        raise ValueError("order must be sorted or shuffled")
    order = [
        (SHUFFLE_MULTIPLIER * index + SHUFFLE_OFFSET) % requests
        for index in range(requests)
    ]
    if sorted(order) != list(range(requests)):
        raise ValueError("shuffle parameters do not form a permutation")
    return order


def wait_group(
    batcher: MixedPhaseBatcher,
    entries: Sequence[PhaseRow],
    timeout_s: float,
) -> tuple[BatchResult, ...]:
    futures: tuple[Future[BatchResult], ...] = batcher.submit_many(
        entries,
        timeout_s,
    )
    results = tuple(future.result(timeout=timeout_s) for future in futures)
    if len(results) != len(entries):
        raise ProtocolError("order probe result count mismatch")
    for entry, result in zip(entries, results):
        row = entry.row
        if (
            result.request_id != row.request_id
            or result.route_epoch != row.route_epoch
            or result.seq_id != row.seq_id
            or result.position != row.position
            or result.hidden is not None
            or result.token is None
        ):
            raise ProtocolError("order probe returned invalid terminal lineage")
    return results


def expected_batch_sizes(
    requests: int,
    prompt_length: int,
    steps: int,
) -> list[int]:
    return [requests * prompt_length] + [requests] * (steps - 1)


def validate_order_events(
    events: Sequence[dict],
    order: Sequence[int],
    prompt_length: int,
    steps: int,
    batch_knee: int | None = None,
) -> None:
    requests = len(order)
    sizes = expected_batch_sizes(requests, prompt_length, steps)
    if batch_knee is None:
        batch_knee = requests * prompt_length
    if batch_knee <= 0:
        raise ValueError("batch knee must be positive")
    if [event.get("batch_size") for event in events] != sizes:
        raise ProtocolError("order probe batch sequence differs from plan")

    prefill_sequences = [
        sequence_id
        for sequence_id in order
        for _ in range(prompt_length)
    ]
    prefill_requests = [
        REQUEST_BASE + sequence_id
        for sequence_id in order
        for _ in range(prompt_length)
    ]
    for index, event in enumerate(events):
        prefill = index == 0
        expected_sequences = prefill_sequences if prefill else list(order)
        expected_requests = prefill_requests if prefill else [
            REQUEST_BASE + sequence_id for sequence_id in order
        ]
        expected_positions = (
            list(range(prompt_length)) * requests
            if prefill
            else [prompt_length + index - 1] * requests
        )
        expected_phase = PHASE_PREFILL if prefill else PHASE_DECODE
        expected_reason = (
            "BATCH_KNEE" if sizes[index] >= batch_knee else "DEADLINE"
        )
        if (
            event.get("sequence_ids") != expected_sequences
            or event.get("request_ids") != expected_requests
            or event.get("positions") != expected_positions
            or event.get("phases") != [expected_phase] * sizes[index]
            or event.get("decode_rows") != (0 if prefill else requests)
            or event.get("prefill_rows") != (sizes[index] if prefill else 0)
            or event.get("mixed_phase") is not False
            or event.get("release_reason") != expected_reason
        ):
            raise ProtocolError("order probe physical row order differs from plan")


def write_atomic(path: Path, value: dict) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--relay", type=parse_endpoint, required=True)
    parser.add_argument("--order", choices=("sorted", "shuffled"), required=True)
    parser.add_argument("--requests", type=int, default=FROZEN_REQUESTS)
    parser.add_argument("--steps", type=int, default=FROZEN_STEPS)
    parser.add_argument("--batch-knee", type=int, default=160)
    parser.add_argument("--gather-us", type=int, default=50000)
    parser.add_argument("--queue-depth", type=int, default=256)
    parser.add_argument("--slo-ms", type=float, default=300000.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--prompt-tokens", type=parse_tokens, required=True)
    parser.add_argument("--expected-tokens", type=parse_tokens, required=True)
    parser.add_argument(
        "--session-end",
        choices=("stop", "detach"),
        default="stop",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_length = len(args.prompt_tokens)
    if not args.route_id:
        parser.error("route ID must be nonempty")
    if args.requests != FROZEN_REQUESTS or args.steps != FROZEN_STEPS:
        parser.error("W3 requires exactly 32 requests and 8 generated tokens")
    if args.gather_us <= 0 or args.queue_depth <= 0:
        parser.error("gather and queue bounds are invalid")
    if args.slo_ms <= 0 or args.timeout <= 0:
        parser.error("SLO and timeout must be positive")
    if len(args.expected_tokens) != args.steps:
        parser.error("expected token count must equal steps")
    planned_knee = args.requests * prompt_length
    if args.batch_knee != planned_knee:
        parser.error("batch knee must equal the prefill row count")
    if args.queue_depth < planned_knee:
        parser.error("queue depth cannot hold the atomic prefill cohort")

    order = sequence_order(args.order, args.requests)
    client: StageV3Client | None = None
    batcher: MixedPhaseBatcher | None = None
    client_closed = False
    try:
        client = StageV3Client.connect(*args.relay, args.timeout)
        hello = client.hello()
        if (
            hello.layer_start != 0
            or hello.layer_end != hello.n_layer
            or not hello.capabilities & STAGE_V3_CAP_TERMINAL
        ):
            raise ProtocolError("direct relay does not present a terminal route")
        route_capacity = min(hello.n_batch, hello.n_ubatch)
        if hello.max_streams < args.requests:
            raise ProtocolError("direct relay lacks sequence capacity")
        if route_capacity < args.batch_knee:
            raise ProtocolError("direct relay lacks row capacity")

        batcher = MixedPhaseBatcher(
            f"direct-order-{args.order}",
            client,
            route_capacity,
            args.batch_knee,
            args.gather_us,
            args.queue_depth,
        )
        generated: dict[int, list[int]] = {
            sequence_id: [] for sequence_id in range(args.requests)
        }
        started_ns = time.monotonic_ns()
        prefill = [
            PhaseRow(
                BatchRow(
                    REQUEST_BASE + sequence_id,
                    ROUTE_EPOCH,
                    sequence_id,
                    position,
                    token,
                ),
                PHASE_PREFILL,
                0,
            )
            for sequence_id in order
            for position, token in enumerate(args.prompt_tokens)
        ]
        results = wait_group(batcher, prefill, args.timeout)
        for order_index, sequence_id in enumerate(order):
            token = results[(order_index + 1) * prompt_length - 1].token
            if token is None:
                raise ProtocolError("prefill omitted terminal token")
            generated[sequence_id].append(token)

        for output_index in range(1, args.steps):
            entries = [
                PhaseRow(
                    BatchRow(
                        REQUEST_BASE + sequence_id,
                        ROUTE_EPOCH,
                        sequence_id,
                        prompt_length + output_index - 1,
                        generated[sequence_id][-1],
                    ),
                    PHASE_DECODE,
                    0,
                )
                for sequence_id in order
            ]
            results = wait_group(batcher, entries, args.timeout)
            for sequence_id, result in zip(order, results):
                if result.token is None:
                    raise ProtocolError("decode omitted terminal token")
                generated[sequence_id].append(result.token)
        completed_ns = time.monotonic_ns()

        batcher.stop(args.timeout)
        events = list(batcher.events)
        batcher = None
        validate_order_events(
            events,
            order,
            prompt_length,
            args.steps,
            args.batch_knee,
        )

        expected = list(args.expected_tokens)
        elapsed_ms = (completed_ns - started_ns) / 1e6
        requests = [
            {
                "elapsed_ms": elapsed_ms,
                "request_id": REQUEST_BASE + sequence_id,
                "route_epoch": ROUTE_EPOCH,
                "sequence_id": sequence_id,
                "slo_met": elapsed_ms <= args.slo_ms,
                "tokens": generated[sequence_id],
            }
            for sequence_id in range(args.requests)
        ]
        tokens_exact = all(
            request["tokens"] == expected for request in requests
        )
        all_slo_met = all(request["slo_met"] for request in requests)

        for sequence_id in range(args.requests):
            client.remove(
                sequence_id,
                REQUEST_BASE + sequence_id,
                ROUTE_EPOCH,
            )
        status = client.drain()
        if status.active_sequences != 0:
            raise ProtocolError("order probe left live sequences")

        if args.session_end == "stop":
            client.stop()
        else:
            client.detach()
        client.close()
        client_closed = True

        latencies = [request["elapsed_ms"] for request in requests]
        verdict = "PASS" if tokens_exact and all_slo_met else "FAIL"
        report = {
            "schema": "s39-direct-order-route-v1",
            "verdict": verdict,
            "status": (
                "DIRECT_ORDER_MECHANICS_PASS"
                if verdict == "PASS"
                else "DIRECT_ORDER_FAIL"
            ),
            "route_id": args.route_id,
            "order": args.order,
            "permutation": order,
            "configuration": {
                "batch_knee": args.batch_knee,
                "expected_tokens": expected,
                "gather_us": args.gather_us,
                "prompt_tokens": list(args.prompt_tokens),
                "queue_depth": args.queue_depth,
                "requests": args.requests,
                "slo_ms": args.slo_ms,
                "steps": args.steps,
            },
            "worker": asdict(hello),
            "batch_events": events,
            "batch_summary": {
                "batch_count": len(events),
                "batch_sizes": [
                    event["batch_size"] for event in events
                ],
                "compute_us_total": sum(
                    event["compute_us"] for event in events
                ),
                "max_batch": max(
                    event["batch_size"] for event in events
                ),
            },
            "latency_ms": {
                "max": max(latencies),
                "p50": nearest_rank(latencies, 1, 2),
                "p95": nearest_rank(latencies, 19, 20),
            },
            "requests": requests,
            "token_checks": len(requests) * args.steps,
            "tokens_exact": tokens_exact,
            "transport": {
                "direct_activation_payload_bytes": sum(
                    event["batch_size"] for event in events
                )
                * hello.n_embd
                * 4,
                "host_activation_payload_bytes": 0,
                "mode": "OP15_TO_OP12_DIRECT_WIFI",
                "weight_provisioning": "USB_BEFORE_SERVICE",
            },
            "scope": {
                "energy": "NOT_MEASURED",
                "inter_stage_overlap": "NOT_IMPLEMENTED",
                "variable": "ROW_ORDER_ONLY",
            },
        }
        write_atomic(args.output, report)
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0 if verdict == "PASS" else 2
    finally:
        if batcher is not None:
            if client is not None and not client_closed:
                try:
                    client.close()
                    client_closed = True
                except OSError:
                    pass
            try:
                batcher.abort(RuntimeError("order probe aborted"), 5.0)
            except BaseException:
                pass
        if client is not None and not client_closed:
            try:
                client.close()
            except OSError:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ProtocolError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {"error": str(error), "verdict": "FAIL"},
                sort_keys=True,
            )
        )
        raise SystemExit(2)
