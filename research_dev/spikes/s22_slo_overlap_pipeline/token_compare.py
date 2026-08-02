#!/usr/bin/env python3
"""Compare sequential and chunked tokens from one terminal V3 worker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from async_pipeline import parse_endpoint, parse_tokens
from stage_v3_client import (
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


def capture_tokens(
    client: StageV3Client,
    prompt: Sequence[int],
    request_id: int,
    seq_id: int,
    chunked: bool,
) -> tuple[list[int], float]:
    started_ns = time.monotonic_ns()
    results: list[BatchResult] = []
    if chunked:
        results.extend(client.batch([
            BatchRow(request_id, 1, seq_id, position, token)
            for position, token in enumerate(prompt)
        ]))
    else:
        for position, token in enumerate(prompt):
            results.extend(client.batch([
                BatchRow(request_id, 1, seq_id, position, token),
            ]))
    elapsed_ms = (time.monotonic_ns() - started_ns) / 1_000_000.0
    tokens = []
    for position, result in enumerate(results):
        if result.position != position or result.token is None or result.hidden is not None:
            raise ProtocolError("terminal trial returned an invalid token row")
        tokens.append(result.token)
    if len(tokens) != len(prompt):
        raise ProtocolError("terminal trial returned the wrong row count")
    status = client.remove(seq_id, request_id, 1)
    if status.active_sequences != 0:
        raise ProtocolError("terminal trial did not release its sequence")
    return tokens, elapsed_ms


def classify_tokens(
    sequential_a: Sequence[int],
    chunked: Sequence[int],
    sequential_b: Sequence[int],
) -> str:
    if not sequential_a or len(sequential_a) != len(chunked) or len(chunked) != len(sequential_b):
        raise ValueError("token trials must have the same nonzero length")
    if list(sequential_a) != list(sequential_b):
        return "NONDETERMINISTIC"
    if list(sequential_a) != list(chunked):
        return "MODE_DIVERGENCE"
    return "EXACT"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=parse_endpoint, required=True)
    parser.add_argument("--prompt-tokens", type=parse_tokens, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")

    client = StageV3Client.connect(*args.worker, args.timeout)
    try:
        hello = client.hello()
        if not hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("token comparison requires a terminal worker")
        if len(args.prompt_tokens) > min(hello.n_batch, hello.n_ubatch):
            raise ProtocolError("prompt exceeds worker batch capacity")
        if len(args.prompt_tokens) > hello.n_ctx_seq:
            raise ProtocolError("prompt exceeds worker sequence context")

        sequential_a, sequential_a_ms = capture_tokens(
            client, args.prompt_tokens, 4201, 0, False,
        )
        chunked, chunked_ms = capture_tokens(
            client, args.prompt_tokens, 4202, 0, True,
        )
        sequential_b, sequential_b_ms = capture_tokens(
            client, args.prompt_tokens, 4203, 0, False,
        )
        verdict = classify_tokens(sequential_a, chunked, sequential_b)
        report = {
            "schema": "s22-token-compare-v1",
            "verdict": verdict,
            "worker": {
                "layer_start": hello.layer_start,
                "layer_end": hello.layer_end,
                "n_batch": hello.n_batch,
                "n_ubatch": hello.n_ubatch,
            },
            "prompt_tokens": list(args.prompt_tokens),
            "elapsed_ms": {
                "sequential_a": sequential_a_ms,
                "chunked": chunked_ms,
                "sequential_b": sequential_b_ms,
            },
            "sequential_a": sequential_a,
            "chunked": chunked,
            "sequential_b": sequential_b,
        }
        data = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="ascii")
        print(data, end="")
        if args.session_end == "stop":
            client.stop()
        else:
            client.detach()
        return 0 if verdict == "EXACT" else 2
    finally:
        client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
