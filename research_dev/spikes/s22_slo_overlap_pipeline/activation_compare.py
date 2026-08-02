#!/usr/bin/env python3
"""Compare sequential and chunked StageNet V3 boundary activations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path
from typing import Sequence

from async_pipeline import parse_endpoint, parse_tokens
from stage_v3_client import BatchResult, BatchRow, ProtocolError, StageV3Client


def row_bytes(values: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def compare_vectors(reference: Sequence[float], candidate: Sequence[float]) -> dict:
    if not reference or len(reference) != len(candidate):
        raise ValueError("activation vectors must have the same nonzero width")
    ref_sq = 0.0
    candidate_sq = 0.0
    diff_sq = 0.0
    dot = 0.0
    max_abs = 0.0
    for ref, value in zip(reference, candidate):
        if not math.isfinite(ref) or not math.isfinite(value):
            raise ValueError("activation vector contains a non-finite value")
        delta = value - ref
        ref_sq += ref * ref
        candidate_sq += value * value
        diff_sq += delta * delta
        dot += ref * value
        max_abs = max(max_abs, abs(delta))
    if ref_sq == 0.0 or candidate_sq == 0.0:
        raise ValueError("activation vector has zero norm")
    ref_raw = row_bytes(reference)
    candidate_raw = row_bytes(candidate)
    return {
        "byte_equal": ref_raw == candidate_raw,
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "cosine": dot / math.sqrt(ref_sq * candidate_sq),
        "max_abs": max_abs,
        "reference_sha256": hashlib.sha256(ref_raw).hexdigest(),
        "rel_l2": math.sqrt(diff_sq / ref_sq),
    }


def capture_trial(
    client: StageV3Client,
    tokens: Sequence[int],
    request_id: int,
    route_epoch: int,
    seq_id: int,
    chunked: bool,
) -> tuple[list[tuple[float, ...]], float]:
    started_ns = time.monotonic_ns()
    results: list[BatchResult] = []
    if chunked:
        rows = [
            BatchRow(request_id, route_epoch, seq_id, position, token)
            for position, token in enumerate(tokens)
        ]
        results.extend(client.batch(rows))
    else:
        for position, token in enumerate(tokens):
            results.extend(client.batch([
                BatchRow(request_id, route_epoch, seq_id, position, token),
            ]))
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
    hidden = []
    for index, result in enumerate(results):
        if result.position != index or result.hidden is None or result.token is not None:
            raise ProtocolError("head trial returned an invalid activation row")
        hidden.append(result.hidden)
    if len(hidden) != len(tokens):
        raise ProtocolError("head trial returned the wrong row count")
    status = client.remove(seq_id, request_id, route_epoch)
    if status.active_sequences != 0:
        raise ProtocolError("head trial did not release its sequence")
    return hidden, elapsed_ms


def compare_trials(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> list[dict]:
    if not reference or len(reference) != len(candidate):
        raise ValueError("activation trials must have the same nonzero row count")
    return [
        {"position": position, **compare_vectors(ref, value)}
        for position, (ref, value) in enumerate(zip(reference, candidate))
    ]


def classify_trials(repeat: Sequence[dict], mode: Sequence[dict], max_rel_l2: float, min_cosine: float) -> str:
    if not repeat or len(repeat) != len(mode):
        raise ValueError("comparison sets must have the same nonzero row count")
    if not 0.0 <= max_rel_l2 <= 1.0 or not 0.0 <= min_cosine <= 1.0:
        raise ValueError("numeric thresholds are out of range")
    if not all(row["byte_equal"] for row in repeat):
        return "NONDETERMINISTIC"
    if all(row["byte_equal"] for row in mode):
        return "EXACT"
    if all(
        row["rel_l2"] <= max_rel_l2 and row["cosine"] >= min_cosine
        for row in mode
    ):
        return "NUMERIC_PASS"
    return "NUMERIC_FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=parse_endpoint, required=True)
    parser.add_argument("--prompt-tokens", type=parse_tokens, required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-rel-l2", type=float, default=5e-3)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    client = StageV3Client.connect(*args.head, args.timeout)
    try:
        hello = client.hello()
        if hello.layer_start != 0 or hello.layer_end == hello.n_layer:
            raise ProtocolError("activation comparison requires a nonterminal head")
        if len(args.prompt_tokens) > min(hello.n_batch, hello.n_ubatch):
            raise ProtocolError("prompt exceeds worker batch capacity")
        if len(args.prompt_tokens) > hello.n_ctx_seq:
            raise ProtocolError("prompt exceeds worker sequence context")

        sequential_a, sequential_a_ms = capture_trial(
            client, args.prompt_tokens, 4101, 1, 0, False,
        )
        chunked, chunked_ms = capture_trial(
            client, args.prompt_tokens, 4102, 1, 0, True,
        )
        sequential_b, sequential_b_ms = capture_trial(
            client, args.prompt_tokens, 4103, 1, 0, False,
        )
        repeat = compare_trials(sequential_a, sequential_b)
        mode = compare_trials(sequential_a, chunked)
        verdict = classify_trials(
            repeat, mode, args.max_rel_l2, args.min_cosine,
        )
        report = {
            "schema": "s22-activation-compare-v1",
            "verdict": verdict,
            "worker": {
                "layer_start": hello.layer_start,
                "layer_end": hello.layer_end,
                "n_embd": hello.n_embd,
                "n_batch": hello.n_batch,
                "n_ubatch": hello.n_ubatch,
            },
            "prompt_tokens": list(args.prompt_tokens),
            "thresholds": {
                "max_rel_l2": args.max_rel_l2,
                "min_cosine": args.min_cosine,
            },
            "elapsed_ms": {
                "sequential_a": sequential_a_ms,
                "chunked": chunked_ms,
                "sequential_b": sequential_b_ms,
            },
            "sequential_repeat": repeat,
            "sequential_vs_chunked": mode,
        }
        data = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="ascii")
        print(data, end="")
        if args.session_end == "stop":
            client.stop()
        else:
            client.detach()
        return 0 if verdict in ("EXACT", "NUMERIC_PASS") else 2
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
