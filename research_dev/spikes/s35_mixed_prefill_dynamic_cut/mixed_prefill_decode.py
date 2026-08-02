#!/usr/bin/env python3
"""Real StageNet V3 mixed-prefill/decode proof."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from stage_v3_client import (  # noqa: E402
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    StageV3Client,
)


REL_L2_LIMIT = 0.005
PROMPT = (2, 532, 236772, 564)
DECODE_TOKEN = 496


class GateError(RuntimeError):
    pass


def _hidden(result: BatchResult) -> tuple[float, ...]:
    if result.hidden is None or result.token is not None or not result.hidden:
        raise GateError("worker did not return a nonterminal activation")
    if not all(math.isfinite(value) for value in result.hidden):
        raise GateError("worker returned a non-finite activation")
    return result.hidden


def activation_metrics(
    reference: Sequence[float], treatment: Sequence[float],
) -> dict[str, float]:
    if not reference or len(reference) != len(treatment):
        raise GateError("activation shapes differ")
    if not all(math.isfinite(value) for value in reference):
        raise GateError("reference activation is non-finite")
    if not all(math.isfinite(value) for value in treatment):
        raise GateError("treatment activation is non-finite")
    diff_sq = sum((lhs - rhs) ** 2 for lhs, rhs in zip(reference, treatment))
    reference_sq = sum(value * value for value in reference)
    treatment_sq = sum(value * value for value in treatment)
    dot = sum(lhs * rhs for lhs, rhs in zip(reference, treatment))
    if reference_sq <= 0.0 or treatment_sq <= 0.0:
        raise GateError("activation norm is zero")
    return {
        "rel_l2": math.sqrt(diff_sq / reference_sq),
        "cosine": dot / math.sqrt(reference_sq * treatment_sq),
        "max_abs": max(abs(lhs - rhs) for lhs, rhs in zip(reference, treatment)),
    }


def _rows(
    request_id: int,
    route_epoch: int,
    seq_id: int,
    start_pos: int,
    tokens: Sequence[int],
) -> list[BatchRow]:
    return [
        BatchRow(request_id, route_epoch, seq_id, start_pos + offset, token)
        for offset, token in enumerate(tokens)
    ]


def run_serial(client: StageV3Client) -> dict[tuple[str, int], tuple[float, ...]]:
    hello = client.hello()
    if hello.layer_start != 0 or hello.layer_end == hello.n_layer:
        raise GateError("mixed gate requires a nonterminal head worker")
    if hello.max_streams < 2 or min(hello.n_batch, hello.n_ubatch) < 5:
        raise GateError("worker lacks B=5 and two-stream capacity")

    output: dict[tuple[str, int], tuple[float, ...]] = {}
    rows_a = _rows(101, 1, 0, 0, PROMPT)
    for result in client.batch(rows_a):
        output[("A", result.position)] = _hidden(result)
    decode = client.batch([BatchRow(101, 1, 0, 4, DECODE_TOKEN)])
    if len(decode) != 1:
        raise GateError("serial decode returned the wrong row count")
    output[("A", 4)] = _hidden(decode[0])
    client.remove(0, 101, 1)

    rows_b = _rows(102, 1, 1, 0, PROMPT)
    for result in client.batch(rows_b):
        output[("B", result.position)] = _hidden(result)
    client.remove(1, 102, 1)
    if client.status().active_sequences != 0:
        raise GateError("serial oracle left live sequences")
    return output


def run_mixed(client: StageV3Client) -> tuple[
    dict[tuple[str, int], tuple[float, ...]], dict[str, object], Hello
]:
    hello = client.hello()
    if hello.layer_start != 0 or hello.layer_end == hello.n_layer:
        raise GateError("mixed gate requires a nonterminal head worker")
    if hello.max_streams < 2 or min(hello.n_batch, hello.n_ubatch) < 5:
        raise GateError("worker lacks B=5 and two-stream capacity")

    seed = client.batch(_rows(201, 2, 0, 0, PROMPT))
    if len(seed) != 4 or client.status().active_sequences != 1:
        raise GateError("request A seed state differs from contract")

    mixed_rows = [BatchRow(201, 2, 0, 4, DECODE_TOKEN)]
    mixed_rows.extend(_rows(202, 2, 1, 0, PROMPT))
    started_ns = time.monotonic_ns()
    mixed = client.batch(mixed_rows)
    elapsed_us = (time.monotonic_ns() - started_ns) // 1000
    if len(mixed) != 5 or client.status().active_sequences != 2:
        raise GateError("mixed batch state differs from contract")

    output = {("A", 4): _hidden(mixed[0])}
    for result in mixed[1:]:
        output[("B", result.position)] = _hidden(result)

    client.remove(0, 201, 2)
    client.remove(1, 202, 2)
    if client.status().active_sequences != 0:
        raise GateError("mixed treatment left live sequences")
    event = {
        "physical_batch_size": 5,
        "elapsed_us": elapsed_us,
        "rows": [
            {
                "phase": "decode" if index == 0 else "prefill",
                "request_id": row.request_id,
                "seq_id": row.seq_id,
                "position": row.position,
            }
            for index, row in enumerate(mixed_rows)
        ],
    }
    return output, event, hello


def evaluate(
    reference: Mapping[tuple[str, int], Sequence[float]],
    treatment: Mapping[tuple[str, int], Sequence[float]],
) -> tuple[list[dict[str, object]], bool]:
    expected = {("A", 4), *(('B', position) for position in range(4))}
    if set(treatment) != expected or not expected.issubset(reference):
        raise GateError("treatment output lineage differs from contract")
    metrics = []
    passed = True
    for key in sorted(expected):
        row_metrics = activation_metrics(reference[key], treatment[key])
        row_pass = row_metrics["rel_l2"] <= REL_L2_LIMIT
        passed = passed and row_pass
        metrics.append({
            "request": key[0],
            "position": key[1],
            **row_metrics,
            "pass": row_pass,
        })
    return metrics, passed


def run(args: argparse.Namespace) -> dict[str, object]:
    serial_client = StageV3Client.connect(*args.endpoint, args.timeout)
    try:
        reference = run_serial(serial_client)
        serial_client.detach()
    finally:
        serial_client.close()

    mixed_client = StageV3Client.connect(*args.endpoint, args.timeout)
    try:
        treatment, event, hello = run_mixed(mixed_client)
        metrics, numeric_pass = evaluate(reference, treatment)
        mixed_client.stop()
    finally:
        mixed_client.close()

    return {
        "schema": "s35-mixed-prefill-decode-v1",
        "device": args.device,
        "verdict": "PASS" if numeric_pass else "NUMERIC_FAIL",
        "mechanics_pass": True,
        "numeric_pass": numeric_pass,
        "rel_l2_limit": REL_L2_LIMIT,
        "hello": asdict(hello),
        "mixed_event": event,
        "metrics": metrics,
        "claims": {
            "one_physical_batch": True,
            "decode_rows": 1,
            "prefill_rows": 4,
            "continuous_sequences": True,
            "energy": "NOT_MEASURED",
        },
    }


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        parsed_port = int(port)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid endpoint port") from exc
    if not 1 <= parsed_port <= 65535:
        raise argparse.ArgumentTypeError("invalid endpoint port")
    return host, parsed_port


def write_report(path: Path, report: object) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    payload = (
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
            "schema": "s35-mixed-prefill-decode-v1",
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
