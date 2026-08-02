#!/usr/bin/env python3
"""Measure and select a bounded StageNet V3 batch knee."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
if str(S22) not in sys.path:
    sys.path.insert(0, str(S22))

from async_pipeline import parse_endpoint
from stage_v3_client import (
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


def parse_candidates(value: str) -> tuple[int, ...]:
    try:
        candidates = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidates must be comma-separated integers"
        ) from exc
    if (
        not candidates
        or any(candidate <= 0 for candidate in candidates)
        or tuple(sorted(set(candidates))) != candidates
    ):
        raise argparse.ArgumentTypeError(
            "candidates must be unique, positive, and increasing"
        )
    return candidates


def select_knee(
    samples_us: dict[int, Sequence[int]],
    fraction_of_peak: float = 0.95,
) -> tuple[int, dict[int, float]]:
    if not 0.0 < fraction_of_peak <= 1.0 or not samples_us:
        raise ValueError("knee inputs are invalid")
    throughputs = {}
    for batch, samples in samples_us.items():
        if batch <= 0 or not samples or any(sample <= 0 for sample in samples):
            raise ValueError("knee samples are invalid")
        throughputs[batch] = (
            batch * 1_000_000.0 / statistics.median(samples)
        )
    threshold = max(throughputs.values()) * fraction_of_peak
    knee = min(
        batch for batch, throughput in throughputs.items()
        if throughput >= threshold
    )
    return knee, throughputs


def validate_results(results, terminal: bool) -> None:
    for result in results:
        if terminal:
            if result.token is None or result.hidden is not None:
                raise ProtocolError("terminal knee probe returned hidden state")
        else:
            if result.hidden is None or result.token is not None:
                raise ProtocolError("nonterminal knee probe returned a token")
            if not result.hidden or not all(
                math.isfinite(value) for value in result.hidden
            ):
                raise ProtocolError("knee probe returned non-finite hidden state")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    parser.add_argument("--endpoint", type=parse_endpoint, required=True)
    parser.add_argument(
        "--candidates", type=parse_candidates, default=(1, 2, 4, 8),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument(
        "--session-end", choices=("detach", "stop"), default="detach",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.worker
        or args.warmup < 0
        or args.rounds <= 0
        or args.timeout <= 0
        or args.token < 0
    ):
        parser.error("probe bounds are invalid")

    client = StageV3Client.connect(*args.endpoint, args.timeout)
    ended = False
    try:
        hello = client.hello()
        terminal = bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
        candidates = tuple(
            candidate for candidate in args.candidates
            if candidate <= hello.max_streams
            and candidate <= min(hello.n_batch, hello.n_ubatch)
        )
        if not candidates:
            raise ProtocolError("worker supports no requested knee candidate")
        samples: dict[int, list[int]] = {
            candidate: [] for candidate in candidates
        }
        trial_id = 0
        for batch in candidates:
            for trial in range(args.warmup + args.rounds):
                trial_id += 1
                request_base = 24_000_000 + trial_id * 100
                hidden = (
                    tuple(0.0 for _index in range(hello.n_embd))
                    if hello.layer_start > 0 else None
                )
                rows = [
                    BatchRow(
                        request_base + seq_id,
                        1,
                        seq_id,
                        0,
                        args.token,
                        hidden,
                    )
                    for seq_id in range(batch)
                ]
                started_ns = time.monotonic_ns()
                results = client.batch(rows)
                elapsed_us = max(
                    1, (time.monotonic_ns() - started_ns) // 1000,
                )
                validate_results(results, terminal)
                for row in rows:
                    client.remove(
                        row.seq_id, row.request_id, row.route_epoch,
                    )
                if client.status().active_sequences != 0:
                    raise ProtocolError("knee probe left live worker KV")
                if trial >= args.warmup:
                    samples[batch].append(elapsed_us)
        knee, throughputs = select_knee(samples)
        report = {
            "schema": "s24-batch-knee-v1",
            "worker": args.worker,
            "endpoint": {
                "host": args.endpoint[0], "port": args.endpoint[1],
            },
            "hello": asdict(hello),
            "candidates": list(candidates),
            "warmup": args.warmup,
            "rounds": args.rounds,
            "selection": {
                "rule": "smallest-batch-at-least-95-percent-of-peak-rows-per-second",
                "fraction_of_peak": 0.95,
                "batch_knee": knee,
            },
            "samples_us": {
                str(batch): values for batch, values in samples.items()
            },
            "median_us": {
                str(batch): statistics.median(values)
                for batch, values in samples.items()
            },
            "rows_per_second": {
                str(batch): throughputs[batch] for batch in samples
            },
            "final_status": asdict(client.status()),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        if args.session_end == "detach":
            client.detach()
        else:
            client.stop()
        ended = True
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    finally:
        if not ended:
            try:
                client.stop()
            except BaseException:
                pass
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({
            "verdict": "FAIL", "error": str(exc),
        }, sort_keys=True))
        raise SystemExit(2)
