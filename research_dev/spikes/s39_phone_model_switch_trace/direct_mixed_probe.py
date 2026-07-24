#!/usr/bin/env python3
"""Run one mixed decode/prefill cohort through the direct phone chain."""

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

from mixed_phase_batcher import (
    PHASE_DECODE,
    PHASE_PREFILL,
    MixedPhaseBatcher,
    PhaseRow,
)
from stage_v3_client import (
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)
from async_pipeline import parse_endpoint, parse_tokens


SEED_REQUEST_BASE = 4001
NEW_REQUEST_BASE = 5001
ROUTE_EPOCH = 1


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
        raise ProtocolError("mixed route result count mismatch")
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
            raise ProtocolError("mixed route returned invalid terminal lineage")
    return results


def expected_batch_sizes(
    cohort_size: int,
    prompt_length: int,
    steps: int,
) -> list[int]:
    return (
        [cohort_size * prompt_length, cohort_size * (prompt_length + 1)]
        + [2 * cohort_size] * (steps - 2)
        + [cohort_size]
    )


def validate_mixed_events(
    events: Sequence[dict],
    cohort_size: int,
    prompt_length: int,
    steps: int,
) -> dict:
    expected_sizes = expected_batch_sizes(
        cohort_size,
        prompt_length,
        steps,
    )
    if [event.get("batch_size") for event in events] != expected_sizes:
        raise ProtocolError("mixed route batch sequence differs from plan")
    mixed = [event for event in events if event.get("mixed_phase") is True]
    if len(mixed) != 1:
        raise ProtocolError("mixed route requires exactly one mixed batch")
    event = mixed[0]
    decode_rows = cohort_size
    prefill_rows = cohort_size * prompt_length
    if (
        event.get("release_reason") != "BATCH_KNEE"
        or event.get("decode_rows") != decode_rows
        or event.get("prefill_rows") != prefill_rows
        or event.get("phases")
        != [PHASE_DECODE] * decode_rows
        + [PHASE_PREFILL] * prefill_rows
        or event.get("positions")[:decode_rows]
        != [prompt_length] * decode_rows
        or event.get("positions")[decode_rows:]
        != list(range(prompt_length)) * cohort_size
        or event.get("request_ids")[:decode_rows]
        != [SEED_REQUEST_BASE + index for index in range(cohort_size)]
        or event.get("request_ids")[decode_rows:]
        != [
            NEW_REQUEST_BASE + index
            for index in range(cohort_size)
            for _ in range(prompt_length)
        ]
    ):
        raise ProtocolError("mixed route phase ordering differs from plan")
    return {
        "batch_index": list(events).index(event),
        "decode_rows": decode_rows,
        "prefill_rows": prefill_rows,
        "release_reason": event["release_reason"],
    }


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
    parser.add_argument("--cohort-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-knee", type=int, default=96)
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
    if args.cohort_size <= 0 or args.steps < 2:
        parser.error("cohort size must be positive and steps at least two")
    if args.gather_us < 0 or args.queue_depth <= 0:
        parser.error("gather and queue bounds are invalid")
    if args.slo_ms <= 0 or args.timeout <= 0:
        parser.error("SLO and timeout must be positive")
    if len(args.expected_tokens) != args.steps:
        parser.error("expected token count must equal steps")
    planned_knee = args.cohort_size * (prompt_length + 1)
    if args.batch_knee != planned_knee:
        parser.error("batch knee must equal the mixed cohort row count")
    if args.queue_depth < max(planned_knee, args.cohort_size * prompt_length):
        parser.error("queue depth cannot hold the largest atomic cohort")

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
        if hello.max_streams < 2 * args.cohort_size:
            raise ProtocolError("direct relay lacks sequence capacity")
        if route_capacity < args.batch_knee:
            raise ProtocolError("direct relay lacks mixed-row capacity")

        batcher = MixedPhaseBatcher(
            "direct-mixed",
            client,
            route_capacity,
            args.batch_knee,
            args.gather_us,
            args.queue_depth,
        )
        seed_generated = [[] for _ in range(args.cohort_size)]
        new_generated = [[] for _ in range(args.cohort_size)]

        seed_start_ns = time.monotonic_ns()
        seed_prefill = [
            PhaseRow(
                BatchRow(
                    SEED_REQUEST_BASE + index,
                    ROUTE_EPOCH,
                    index,
                    position,
                    token,
                ),
                PHASE_PREFILL,
                2,
            )
            for index in range(args.cohort_size)
            for position, token in enumerate(args.prompt_tokens)
        ]
        seed_results = wait_group(
            batcher,
            seed_prefill,
            args.timeout,
        )
        for index in range(args.cohort_size):
            token = seed_results[(index + 1) * prompt_length - 1].token
            if token is None:
                raise ProtocolError("seed prefill omitted terminal token")
            seed_generated[index].append(token)

        new_start_ns = time.monotonic_ns()
        new_prefill = [
            PhaseRow(
                BatchRow(
                    NEW_REQUEST_BASE + index,
                    ROUTE_EPOCH,
                    args.cohort_size + index,
                    position,
                    token,
                ),
                PHASE_PREFILL,
                2,
            )
            for index in range(args.cohort_size)
            for position, token in enumerate(args.prompt_tokens)
        ]
        seed_decode = [
            PhaseRow(
                BatchRow(
                    SEED_REQUEST_BASE + index,
                    ROUTE_EPOCH,
                    index,
                    prompt_length,
                    seed_generated[index][-1],
                ),
                PHASE_DECODE,
                0,
            )
            for index in range(args.cohort_size)
        ]

        # Submit prefill first. The batcher must still place decode rows first.
        mixed_entries = new_prefill + seed_decode
        mixed_results = wait_group(
            batcher,
            mixed_entries,
            args.timeout,
        )
        new_results = mixed_results[: len(new_prefill)]
        seed_results = mixed_results[len(new_prefill):]
        for index, result in enumerate(seed_results):
            if result.token is None:
                raise ProtocolError("seed decode omitted terminal token")
            seed_generated[index].append(result.token)
        for index in range(args.cohort_size):
            token = new_results[(index + 1) * prompt_length - 1].token
            if token is None:
                raise ProtocolError("new prefill omitted terminal token")
            new_generated[index].append(token)

        for _round in range(args.steps - 2):
            entries: list[PhaseRow] = []
            for index, generated in enumerate(seed_generated):
                entries.append(
                    PhaseRow(
                        BatchRow(
                            SEED_REQUEST_BASE + index,
                            ROUTE_EPOCH,
                            index,
                            prompt_length + len(generated) - 1,
                            generated[-1],
                        ),
                        PHASE_DECODE,
                        0,
                    )
                )
            for index, generated in enumerate(new_generated):
                entries.append(
                    PhaseRow(
                        BatchRow(
                            NEW_REQUEST_BASE + index,
                            ROUTE_EPOCH,
                            args.cohort_size + index,
                            prompt_length + len(generated) - 1,
                            generated[-1],
                        ),
                        PHASE_DECODE,
                        0,
                    )
                )
            results = wait_group(batcher, entries, args.timeout)
            for index in range(args.cohort_size):
                token = results[index].token
                if token is None:
                    raise ProtocolError("seed continuation omitted token")
                seed_generated[index].append(token)
            for index in range(args.cohort_size):
                token = results[args.cohort_size + index].token
                if token is None:
                    raise ProtocolError("new continuation omitted token")
                new_generated[index].append(token)

        seed_completed_ns = time.monotonic_ns()
        final_entries = [
            PhaseRow(
                BatchRow(
                    NEW_REQUEST_BASE + index,
                    ROUTE_EPOCH,
                    args.cohort_size + index,
                    prompt_length + len(generated) - 1,
                    generated[-1],
                ),
                PHASE_DECODE,
                0,
            )
            for index, generated in enumerate(new_generated)
        ]
        final_results = wait_group(
            batcher,
            final_entries,
            args.timeout,
        )
        for index, result in enumerate(final_results):
            if result.token is None:
                raise ProtocolError("new final continuation omitted token")
            new_generated[index].append(result.token)
        new_completed_ns = time.monotonic_ns()

        batcher.stop(args.timeout)
        events = list(batcher.events)
        batcher = None
        mixed_gate = validate_mixed_events(
            events,
            args.cohort_size,
            prompt_length,
            args.steps,
        )

        expected = list(args.expected_tokens)
        requests: list[dict] = []
        for cohort, base, started_ns, completed_ns, generated_rows in (
            (
                "seed_decode",
                SEED_REQUEST_BASE,
                seed_start_ns,
                seed_completed_ns,
                seed_generated,
            ),
            (
                "new_prefill",
                NEW_REQUEST_BASE,
                new_start_ns,
                new_completed_ns,
                new_generated,
            ),
        ):
            for index, generated in enumerate(generated_rows):
                elapsed_ms = (completed_ns - started_ns) / 1e6
                requests.append(
                    {
                        "cohort": cohort,
                        "elapsed_ms": elapsed_ms,
                        "request_id": base + index,
                        "route_epoch": ROUTE_EPOCH,
                        "sequence_id": (
                            index
                            if cohort == "seed_decode"
                            else args.cohort_size + index
                        ),
                        "slo_met": elapsed_ms <= args.slo_ms,
                        "tokens": generated,
                    }
                )

        token_exact = all(
            request["tokens"] == expected for request in requests
        )
        all_slo_met = all(request["slo_met"] for request in requests)
        for index in range(2 * args.cohort_size):
            request_id = (
                SEED_REQUEST_BASE + index
                if index < args.cohort_size
                else NEW_REQUEST_BASE + index - args.cohort_size
            )
            client.remove(index, request_id, ROUTE_EPOCH)
        status = client.drain()
        if status.active_sequences != 0:
            raise ProtocolError("mixed route left live sequences")

        if args.session_end == "stop":
            client.stop()
        else:
            client.detach()
        client.close()
        client_closed = True

        latencies = [request["elapsed_ms"] for request in requests]
        verdict = "PASS" if token_exact and all_slo_met else "FAIL"
        report = {
            "schema": "s39-direct-mixed-route-v1",
            "verdict": verdict,
            "status": (
                "DIRECT_MIXED_BATCH_MECHANICS_PASS"
                if verdict == "PASS"
                else "DIRECT_MIXED_BATCH_FAIL"
            ),
            "route_id": args.route_id,
            "configuration": {
                "batch_knee": args.batch_knee,
                "cohort_size": args.cohort_size,
                "expected_tokens": expected,
                "gather_us": args.gather_us,
                "prompt_tokens": list(args.prompt_tokens),
                "queue_depth": args.queue_depth,
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
                "max_batch": max(
                    event["batch_size"] for event in events
                ),
                "mixed_batch": mixed_gate,
            },
            "latency_ms": {
                "max": max(latencies),
                "p50": nearest_rank(latencies, 1, 2),
                "p95": nearest_rank(latencies, 19, 20),
            },
            "requests": requests,
            "token_checks": len(requests) * args.steps,
            "tokens_exact": token_exact,
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
                "continuous_admission": "MECHANICS_ONLY",
                "inter_stage_overlap": "NOT_IMPLEMENTED",
                "phone_energy": "UNKNOWN",
                "throughput_gain": "NOT_CLAIMED",
            },
        }
        write_atomic(args.output, report)
        print(
            json.dumps(report, sort_keys=True, separators=(",", ":")),
        )
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
                batcher.abort(RuntimeError("mixed probe aborted"), 5.0)
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
