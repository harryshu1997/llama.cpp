#!/usr/bin/env python3
"""Serve one BurstGPT Gemma request from an adopted GPU placement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_request(path: Path, request_index: int) -> dict[str, Any]:
    matches = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("request_index") == request_index:
                matches.append(row)
    if len(matches) != 1:
        raise RuntimeError("Gemma request identity is not unique")
    return matches[0]


def process_command(pid: int) -> list[str]:
    fields = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(
        b"\0"
    )
    if not fields or any(not field for field in fields):
        raise RuntimeError("invalid Gemma server command")
    return [field.decode("utf-8") for field in fields]


def process_memory(pid: int) -> dict[str, int]:
    fields: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"VmRSS", "VmSwap"}:
            fields[name] = int(value.strip().split()[0]) * 1024
    if set(fields) != {"VmRSS", "VmSwap"}:
        raise RuntimeError("incomplete Gemma process memory")
    return {
        "rss_bytes": fields["VmRSS"],
        "swap_bytes": fields["VmSwap"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-index", type=int, default=50)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.output.is_absolute()
        or args.output.exists()
        or not args.raw_output.is_absolute()
        or args.raw_output.exists()
        or args.port <= 0
        or args.port > 65535
        or args.server_pid <= 0
    ):
        parser.error("invalid output or server identity")

    sys.path.insert(0, str(args.burst_dir))
    import run_server_trace  # pylint: disable=import-error,import-outside-toplevel

    row = load_request(args.requests, args.request_index)
    started_ns = time.monotonic_ns()
    first_token: list[int] = []
    value = run_server_trace.server_completion(
        args.port,
        row,
        args.raw_output,
        first_token.append,
    )
    completed_ns = time.monotonic_ns()
    if len(first_token) != 1:
        raise RuntimeError("Gemma request has no unique first token")
    tokens = value["tokens"]
    output: dict[str, Any] = {
        "completed_ns": completed_ns,
        "first_token_ns": first_token[0],
        "input_tokens": row["input_tokens"],
        "output_tokens": len(tokens),
        "predicted_ms": value["predicted_ms"],
        "process_memory": process_memory(args.server_pid),
        "prompt_ms": value["prompt_ms"],
        "request_index": args.request_index,
        "schema": "s42-adopted-gemma-execution-v1",
        "server_command": process_command(args.server_pid),
        "service_s": (completed_ns - started_ns) / 1e9,
        "started_ns": started_ns,
        "status": "PASS",
        "tokens": tokens,
        "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
    }
    output["record_sha256"] = hashlib.sha256(canonical(output)).hexdigest()
    args.output.write_bytes(canonical(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
