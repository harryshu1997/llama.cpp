#!/usr/bin/env python3
"""Physical gate for request-pinned StageNet dynamic layer cuts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from stage_v3_client import (  # noqa: E402
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_RANGE,
    StageV3Client,
)

from mixed_prefill_decode import (  # noqa: E402
    DECODE_TOKEN,
    PROMPT,
    REL_L2_LIMIT,
    GateError,
    activation_metrics,
    parse_endpoint,
    write_report,
)


MIN_CUT_SEPARATION_REL_L2 = 0.01


def _hidden(result: BatchResult) -> tuple[float, ...]:
    if result.hidden is None or result.token is not None or not result.hidden:
        raise GateError("dynamic cut did not return a nonterminal activation")
    if not all(math.isfinite(value) for value in result.hidden):
        raise GateError("dynamic cut returned a non-finite activation")
    return result.hidden


def _digest(values: Sequence[float]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def execute_request(
    client: StageV3Client,
    request_id: int,
    route_epoch: int,
    seq_id: int,
    layer_end: int,
) -> tuple[tuple[float, ...], list[dict[str, int]]]:
    events = []
    started_ns = time.monotonic_ns()
    prompt = client.range_batch([
        BatchRow(request_id, route_epoch, seq_id, position, token)
        for position, token in enumerate(PROMPT)
    ], 0, layer_end)
    events.append({
        "layer_start": 0,
        "layer_end": layer_end,
        "rows": 4,
        "elapsed_us": (time.monotonic_ns() - started_ns) // 1000,
    })
    if len(prompt) != 4:
        raise GateError("dynamic prompt returned the wrong row count")

    started_ns = time.monotonic_ns()
    decoded = client.range_batch([
        BatchRow(request_id, route_epoch, seq_id, 4, DECODE_TOKEN),
    ], 0, layer_end)
    events.append({
        "layer_start": 0,
        "layer_end": layer_end,
        "rows": 1,
        "elapsed_us": (time.monotonic_ns() - started_ns) // 1000,
    })
    if len(decoded) != 1:
        raise GateError("dynamic decode returned the wrong row count")
    output = _hidden(decoded[0])
    status = client.remove(seq_id, request_id, route_epoch)
    if status.active_sequences != 0:
        raise GateError("dynamic request retained sequence state")
    return output, events


def run(args: argparse.Namespace) -> dict[str, object]:
    client = StageV3Client.connect(*args.endpoint, args.timeout)
    success = False
    try:
        hello = client.hello()
        if (hello.layer_start, hello.layer_end) != (0, 2):
            raise GateError("frozen gate requires resident range [0,2)")
        if not hello.capabilities & STAGE_V3_CAP_RANGE:
            raise GateError("worker did not advertise dynamic layer cuts")

        shallow_a, events_a = execute_request(client, 301, 3, 0, 1)

        client.range_batch([
            BatchRow(302, 4, 0, 0, PROMPT[0]),
        ], 0, 1)
        rejected_live_change = False
        try:
            client.range_batch([
                BatchRow(302, 4, 0, 1, PROMPT[1]),
            ], 0, 2)
        except ProtocolError as exc:
            if "rejected batch" not in str(exc):
                raise
            rejected_live_change = True
        if not rejected_live_change:
            raise GateError("worker accepted a live-sequence cut change")
        client.range_batch([
            BatchRow(302, 4, 0, 1, PROMPT[1]),
        ], 0, 1)
        client.remove(0, 302, 4)

        deep, events_deep = execute_request(client, 303, 5, 0, 2)
        shallow_b, events_b = execute_request(client, 304, 6, 0, 1)

        repeat = activation_metrics(shallow_a, shallow_b)
        separation = activation_metrics(shallow_a, deep)
        repeat_pass = repeat["rel_l2"] <= REL_L2_LIMIT
        separation_pass = (
            separation["rel_l2"] >= MIN_CUT_SEPARATION_REL_L2
            and _digest(shallow_a) != _digest(deep)
        )
        if client.status().active_sequences != 0:
            raise GateError("dynamic gate ended with live sequences")
        client.stop()
        success = True
        return {
            "schema": "s35-dynamic-layer-cut-v1",
            "device": args.device,
            "verdict": "PASS" if repeat_pass and separation_pass else "NUMERIC_FAIL",
            "hello": asdict(hello),
            "resident_range": [0, 2],
            "executed_ranges": [[0, 1], [0, 2], [0, 1]],
            "live_cut_change_rejected": rejected_live_change,
            "repeat": {**repeat, "limit": REL_L2_LIMIT, "pass": repeat_pass},
            "cut_separation": {
                **separation,
                "minimum_rel_l2": MIN_CUT_SEPARATION_REL_L2,
                "pass": separation_pass,
            },
            "activation_sha256": {
                "shallow_a": _digest(shallow_a),
                "deep": _digest(deep),
                "shallow_b": _digest(shallow_b),
            },
            "events": events_a + events_deep + events_b,
            "claims": {
                "weights_reloaded": False,
                "request_cut_pinned": True,
                "arbitrary_within_residency": True,
                "energy": "NOT_MEASURED",
            },
        }
    finally:
        if not success:
            try:
                client.stop()
            except BaseException:
                pass
        client.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--endpoint", required=True, type=parse_endpoint)
    result.add_argument("--device", required=True)
    result.add_argument("--timeout", type=float, default=300.0)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.timeout <= 0:
        parser().error("timeout must be positive")
    try:
        report = run(args)
        write_report(args.output, report)
    except (FileExistsError, GateError, OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({
            "schema": "s35-dynamic-layer-cut-v1",
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
