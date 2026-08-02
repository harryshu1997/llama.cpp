#!/usr/bin/env python3
"""Compare a physical three-stage Q8 B32 route with a one-GPU reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    BatchResult,
    BatchRow,
    Hello,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


SCHEMA = "s32-q8-chain-token-proof-v2"


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def validate_topology(head: Hello, middle: Hello, tail: Hello, reference: Hello, batch: int) -> None:
    if (head.layer_start, middle.layer_start, tail.layer_start, reference.layer_start) != (0, 4, 16, 0):
        raise ProtocolError("worker start layers differ from the frozen Q8 route")
    if (head.layer_end, middle.layer_end, tail.layer_end, reference.layer_end) != (4, 16, 48, 16):
        raise ProtocolError("worker end layers differ from the frozen Q8 route")
    for hello in (head, middle, tail, reference):
        if hello.n_layer != 48 or hello.n_embd != 3840:
            raise ProtocolError("worker model shape mismatch")
        if hello.max_streams < batch or min(hello.n_batch, hello.n_ubatch) < batch:
            raise ProtocolError("worker cannot execute the requested physical batch")
    if head.capabilities & STAGE_V3_CAP_TERMINAL or middle.capabilities & STAGE_V3_CAP_TERMINAL:
        raise ProtocolError("head or middle worker is terminal")
    if not tail.capabilities & STAGE_V3_CAP_TERMINAL:
        raise ProtocolError("tail worker is not terminal")
    if reference.capabilities & STAGE_V3_CAP_TERMINAL:
        raise ProtocolError("reference head is terminal")


def require_hidden(results: Sequence[BatchResult], batch: int, label: str) -> None:
    if len(results) != batch:
        raise ProtocolError(f"{label} result count mismatch")
    for result in results:
        if result.hidden is None or result.token is not None or len(result.hidden) != 3840:
            raise ProtocolError(f"{label} returned an invalid activation")


def require_tokens(results: Sequence[BatchResult], batch: int, label: str) -> list[int]:
    if len(results) != batch:
        raise ProtocolError(f"{label} result count mismatch")
    tokens: list[int] = []
    for result in results:
        if result.hidden is not None or type(result.token) is not int or result.token < 0:
            raise ProtocolError(f"{label} returned an invalid token")
        tokens.append(result.token)
    return tokens


def rows_for_tokens(tokens: Sequence[int], position: int, identity_base: int) -> list[BatchRow]:
    return [
        BatchRow(identity_base + seq_id, identity_base + seq_id, seq_id, position, token)
        for seq_id, token in enumerate(tokens)
    ]


def initial_tokens(batch: int) -> list[int]:
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be a positive integer")
    return [2 + seq_id for seq_id in range(batch)]


def rows_for_hidden(
    upstream: Sequence[BatchResult], tokens: Sequence[int], position: int,
) -> list[BatchRow]:
    if len(upstream) != len(tokens):
        raise ProtocolError("upstream activation count mismatch")
    return [
        BatchRow(
            result.request_id,
            result.route_epoch,
            result.seq_id,
            position,
            tokens[index],
            result.hidden,
        )
        for index, result in enumerate(upstream)
    ]


def timed_batch(client: StageV3Client, rows: Sequence[BatchRow]) -> tuple[tuple[BatchResult, ...], int]:
    started_ns = time.monotonic_ns()
    results = client.batch(rows)
    return results, (time.monotonic_ns() - started_ns) // 1000


def run_chain(
    head: StageV3Client,
    middle: StageV3Client,
    tail: StageV3Client,
    batch: int,
    steps: int,
    identity_base: int,
) -> tuple[list[list[int]], list[dict[str, int]]]:
    current = initial_tokens(batch)
    outputs = [[] for _ in range(batch)]
    timings: list[dict[str, int]] = []
    for position in range(steps):
        head_results, head_us = timed_batch(
            head, rows_for_tokens(current, position, identity_base),
        )
        require_hidden(head_results, batch, "head")
        middle_results, middle_us = timed_batch(
            middle, rows_for_hidden(head_results, current, position),
        )
        require_hidden(middle_results, batch, "middle")
        tail_results, tail_us = timed_batch(
            tail, rows_for_hidden(middle_results, current, position),
        )
        current = require_tokens(tail_results, batch, "tail")
        for index, token in enumerate(current):
            outputs[index].append(token)
        timings.append({"head_us": head_us, "middle_us": middle_us, "tail_us": tail_us})
    return outputs, timings


def run_reference(
    reference: StageV3Client,
    tail: StageV3Client,
    batch: int,
    steps: int,
    identity_base: int,
) -> tuple[list[list[int]], list[dict[str, int]]]:
    current = initial_tokens(batch)
    outputs = [[] for _ in range(batch)]
    timings: list[dict[str, int]] = []
    for position in range(steps):
        hidden, head_us = timed_batch(
            reference, rows_for_tokens(current, position, identity_base),
        )
        require_hidden(hidden, batch, "reference head")
        results, tail_us = timed_batch(
            tail, rows_for_hidden(hidden, current, position),
        )
        current = require_tokens(results, batch, "reference tail")
        for index, token in enumerate(current):
            outputs[index].append(token)
        timings.append({"head_us": head_us, "tail_us": tail_us})
    return outputs, timings


def clear_sequences(client: StageV3Client, batch: int, identity_base: int) -> None:
    for seq_id in range(batch):
        status = client.remove(seq_id, identity_base + seq_id, identity_base + seq_id)
    if status.active_sequences != 0:
        raise ProtocolError("worker retained sequence state")


def finish(client: StageV3Client) -> None:
    status = client.status()
    if status.active_sequences != 0:
        raise ProtocolError("worker has live state before drain")
    drained = client.drain()
    if drained.active_sequences != 0 or not drained.draining:
        raise ProtocolError("worker drain failed")
    client.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=parse_endpoint, required=True)
    parser.add_argument("--middle", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--reference", type=parse_endpoint, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.batch != 32
        or args.steps < 1
        or args.timeout <= 0
        or re.fullmatch(r"[0-9a-f]{64}", args.model_sha256) is None
    ):
        parser.error("invalid Q8 chain configuration")

    clients: dict[str, StageV3Client] = {}
    report: dict[str, object]
    try:
        for name in ("head", "middle", "tail", "reference"):
            clients[name] = StageV3Client.connect(*getattr(args, name), args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        validate_topology(
            hellos["head"], hellos["middle"], hellos["tail"], hellos["reference"], args.batch,
        )
        chain_tokens, chain_us = run_chain(
            clients["head"], clients["middle"], clients["tail"],
            args.batch, args.steps, 10000,
        )
        for name in ("head", "middle", "tail"):
            clear_sequences(clients[name], args.batch, 10000)
        reference_tokens, reference_us = run_reference(
            clients["reference"], clients["tail"],
            args.batch, args.steps, 20000,
        )
        clear_sequences(clients["reference"], args.batch, 20000)
        clear_sequences(clients["tail"], args.batch, 20000)
        token_matches = sum(left == right for left, right in zip(chain_tokens, reference_tokens))
        all_match = token_matches == args.batch
        report = {
            "schema": SCHEMA,
            "status": "TOKEN_EXACT_PASS" if all_match else "TOKEN_EXACT_FAIL",
            "scheduler_eligible": False,
            "model_sha256": args.model_sha256,
            "batch": args.batch,
            "steps": args.steps,
            "initial_tokens": initial_tokens(args.batch),
            "route": [["OP12", 0, 4], ["OP15", 4, 16], ["CUDA", 16, 48]],
            "reference_route": [["CUDA", 0, 16], ["CUDA", 16, 48]],
            "hellos": {name: asdict(hello) for name, hello in hellos.items()},
            "matching_requests": token_matches,
            "chain_tokens_sha256": hashlib.sha256(canonical(chain_tokens)).hexdigest(),
            "reference_tokens_sha256": hashlib.sha256(canonical(reference_tokens)).hexdigest(),
            "chain_tokens": chain_tokens,
            "reference_tokens": reference_tokens,
            "chain_stage_us": chain_us,
            "reference_stage_us": reference_us,
            "chain_step_median_us": statistics.median(
                row["head_us"] + row["middle_us"] + row["tail_us"] for row in chain_us
            ),
            "reference_step_median_us": statistics.median(
                row["head_us"] + row["tail_us"] for row in reference_us
            ),
            "scope": "B32_FOUR_STEP_TOKEN_CORRECTNESS_ONLY",
        }
        for client in clients.values():
            finish(client)
        for client in clients.values():
            client.close()
        clients.clear()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical(report))
        print(canonical(report).decode("ascii"), end="")
        return 0 if all_match else 3
    except BaseException as exc:
        report = {
            "schema": SCHEMA,
            "status": "PROBE_FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(canonical(report))
        except OSError:
            pass
        print(canonical(report).decode("ascii"), end="", file=sys.stderr)
        return 2
    finally:
        for client in clients.values():
            try:
                client.close()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
