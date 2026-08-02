#!/usr/bin/env python3
"""End-to-end phone-to-CUDA request routes with selectable layer exits."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from stage_v3_client import (  # noqa: E402
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_RANGE,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)

from mixed_prefill_decode import (  # noqa: E402
    PROMPT,
    GateError,
    parse_endpoint,
    write_report,
)


def _activation(result: BatchResult) -> tuple[float, ...]:
    if result.hidden is None or result.token is not None:
        raise GateError("head returned a terminal result")
    if not result.hidden or not all(math.isfinite(value) for value in result.hidden):
        raise GateError("head returned an invalid activation")
    return result.hidden


def _token(result: BatchResult) -> int:
    if result.hidden is not None or type(result.token) is not int or result.token < 0:
        raise GateError("tail returned an invalid token")
    return result.token


def _tail_rows(
    source: list[BatchRow], results: tuple[BatchResult, ...],
) -> list[BatchRow]:
    if len(source) != len(results):
        raise GateError("head and tail row counts differ")
    return [
        BatchRow(
            row.request_id,
            row.route_epoch,
            row.seq_id,
            row.position,
            row.token,
            _activation(result),
        )
        for row, result in zip(source, results)
    ]


def execute_route(
    head: StageV3Client,
    tail: StageV3Client,
    cut: int,
    request_id: int,
    route_epoch: int,
    generated: int,
) -> dict[str, object]:
    prompt_rows = [
        BatchRow(request_id, route_epoch, 0, position, token)
        for position, token in enumerate(PROMPT)
    ]
    events = []
    started_ns = time.monotonic_ns()
    head_prompt = head.range_batch(prompt_rows, 0, cut)
    head_us = (time.monotonic_ns() - started_ns) // 1000
    started_ns = time.monotonic_ns()
    tail_prompt = tail.range_batch(
        _tail_rows(prompt_rows, head_prompt), cut, 48,
    )
    tail_us = (time.monotonic_ns() - started_ns) // 1000
    if len(tail_prompt) != len(prompt_rows):
        raise GateError("tail prompt returned the wrong row count")
    tokens = [_token(tail_prompt[-1])]
    events.append({
        "phase": "prefill",
        "position_start": 0,
        "rows": len(prompt_rows),
        "head_us": head_us,
        "tail_us": tail_us,
    })

    for offset in range(1, generated):
        position = len(PROMPT) + offset - 1
        source = [BatchRow(
            request_id, route_epoch, 0, position, tokens[-1],
        )]
        started_ns = time.monotonic_ns()
        head_result = head.range_batch(source, 0, cut)
        head_us = (time.monotonic_ns() - started_ns) // 1000
        started_ns = time.monotonic_ns()
        tail_result = tail.range_batch(
            _tail_rows(source, head_result), cut, 48,
        )
        tail_us = (time.monotonic_ns() - started_ns) // 1000
        if len(tail_result) != 1:
            raise GateError("tail decode returned the wrong row count")
        tokens.append(_token(tail_result[0]))
        events.append({
            "phase": "decode",
            "position_start": position,
            "rows": 1,
            "head_us": head_us,
            "tail_us": tail_us,
        })

    head_status = head.remove(0, request_id, route_epoch)
    tail_status = tail.remove(0, request_id, route_epoch)
    if head_status.active_sequences != 0 or tail_status.active_sequences != 0:
        raise GateError("route retained sequence state")
    return {
        "cut": cut,
        "tokens": tokens,
        "events": events,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    head = StageV3Client.connect(*args.head, args.timeout)
    tail = StageV3Client.connect(*args.tail, args.timeout)
    success = False
    try:
        head_hello = head.hello()
        tail_hello = tail.hello()
        if (head_hello.layer_start, head_hello.layer_end) != (0, 8):
            raise GateError("head residency must be [0,8)")
        if (tail_hello.layer_start, tail_hello.layer_end) != (4, 48):
            raise GateError("tail residency must be [4,48)")
        if head_hello.n_layer != 48 or tail_hello.n_layer != 48:
            raise GateError("worker layer counts differ from Gemma-4 12B")
        if head_hello.n_embd != tail_hello.n_embd:
            raise GateError("worker embedding widths differ")
        if not head_hello.capabilities & STAGE_V3_CAP_RANGE:
            raise GateError("head omitted dynamic-cut capability")
        if not tail_hello.capabilities & STAGE_V3_CAP_RANGE:
            raise GateError("tail omitted dynamic-cut capability")
        if not tail_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise GateError("tail omitted terminal capability")

        cut4 = execute_route(head, tail, 4, 401, 7, args.generated)
        cut8 = execute_route(head, tail, 8, 402, 8, args.generated)
        tokens_equal = cut4["tokens"] == cut8["tokens"]
        if head.status().active_sequences != 0 or tail.status().active_sequences != 0:
            raise GateError("pipeline ended with live sequences")
        head.stop()
        tail.stop()
        success = True
        return {
            "schema": "s35-dynamic-exit-pipeline-v1",
            "verdict": "PASS" if tokens_equal else "TOKEN_SCREEN_FAIL",
            "mechanics_pass": True,
            "token_screen_pass": tokens_equal,
            "head": asdict(head_hello),
            "tail": asdict(tail_hello),
            "routes": [cut4, cut8],
            "proofs": {
                "same_resident_workers": True,
                "request_level_cuts": [[0, 4, 48], [0, 8, 48]],
                "prefill_and_decode_use_same_cut": True,
                "sequence_state_removed": True,
                "energy": "NOT_MEASURED",
            },
        }
    finally:
        if not success:
            for client in (head, tail):
                try:
                    client.stop()
                except BaseException:
                    pass
        head.close()
        tail.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--head", required=True, type=parse_endpoint)
    result.add_argument("--tail", required=True, type=parse_endpoint)
    result.add_argument("--generated", type=int, default=3)
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.generated < 1 or args.timeout <= 0:
        parser().error("generated and timeout must be positive")
    try:
        report = run(args)
        write_report(args.output, report)
    except (FileExistsError, GateError, OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({
            "schema": "s35-dynamic-exit-pipeline-v1",
            "verdict": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }, sort_keys=True, separators=(",", ":")))
        return 2
    print(json.dumps({
        "verdict": report["verdict"],
        "output": str(args.output),
    }, sort_keys=True, separators=(",", ":")))
    return 0 if report["verdict"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
