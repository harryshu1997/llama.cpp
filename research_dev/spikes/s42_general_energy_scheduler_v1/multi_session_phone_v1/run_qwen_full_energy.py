#!/usr/bin/env python3
"""Measure one matched Qwen BurstGPT cohort through an existing server."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


def parse_indices(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("indices must be non-negative and unique")
    return values


def load_requests(path: Path, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            index = row.get("request_index")
            if index in indices:
                selected[index] = row
    if set(selected) != set(indices):
        raise RuntimeError("requested BurstGPT rows are missing")
    return [selected[index] for index in indices]


def process_command(pid: int) -> list[str]:
    payload = Path(f"/proc/{pid}/cmdline").read_bytes()
    fields = payload.removesuffix(b"\0").split(b"\0")
    if not fields or any(not field for field in fields):
        raise RuntimeError("invalid server command receipt")
    return [field.decode("utf-8") for field in fields]


def validate_receipt_path(
    parser: argparse.ArgumentParser, path: Path | None, name: str
) -> None:
    if path is not None and (
        not path.is_absolute() or path.exists() or not path.parent.is_dir()
    ):
        parser.error(f"{name} must be an unused absolute path")


def await_monotonic_receipt(
    path: Path, timeout_s: float, lower_bound_ns: int, name: str
) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            text = path.read_text(encoding="ascii")
        except FileNotFoundError:
            text = ""
        if text:
            try:
                receipt_ns = int(text.strip())
            except ValueError as exc:
                raise RuntimeError(f"invalid {name} receipt") from exc
            now_ns = time.monotonic_ns()
            if receipt_ns < lower_bound_ns or receipt_ns > now_ns:
                raise RuntimeError(f"{name} receipt is outside its interval")
            return receipt_ns
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {name} receipt")
        time.sleep(0.01)


def write_monotonic_receipt(path: Path, value_ns: int) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        temporary.write_text(f"{value_ns}\n", encoding="ascii")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--burst-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--indices", type=parse_indices, default=(52, 53, 31))
    parser.add_argument(
        "--dispatch",
        choices=("sequential", "concurrent"),
        default="sequential",
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--arm", choices=("control", "op15"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-arm-file", type=Path)
    parser.add_argument("--paid-ready-file", type=Path)
    parser.add_argument("--paid-release-file", type=Path)
    parser.add_argument("--qwen-complete-file", type=Path)
    parser.add_argument("--paid-tail-file", type=Path)
    parser.add_argument("--paid-tail-timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    if not args.output.is_absolute() or args.output.exists():
        parser.error("output must be an unused absolute path")
    if args.port <= 0 or args.port > 65535 or args.server_pid <= 0:
        parser.error("invalid server identity")
    validate_receipt_path(parser, args.prefetch_arm_file, "prefetch arm file")
    validate_receipt_path(parser, args.paid_ready_file, "paid ready file")
    validate_receipt_path(parser, args.paid_release_file, "paid release file")
    validate_receipt_path(parser, args.qwen_complete_file, "Qwen complete file")
    validate_receipt_path(parser, args.paid_tail_file, "paid-tail file")
    if (args.qwen_complete_file is None) != (args.paid_tail_file is None):
        parser.error("Qwen complete and paid-tail files must be provided together")
    if (args.paid_ready_file is None) != (args.paid_release_file is None):
        parser.error("paid ready and release files must be provided together")
    if args.paid_tail_timeout_s <= 0:
        parser.error("paid-tail timeout must be positive")
    args.output.mkdir()

    sys.path.insert(0, str(args.burst_dir))
    import run_server_trace  # pylint: disable=import-error,import-outside-toplevel
    import run_trace  # pylint: disable=import-error,import-outside-toplevel

    rows = load_requests(args.requests, args.indices)
    server_command = process_command(args.server_pid)
    for row in rows:
        run_server_trace.server_completion(
            args.port,
            row,
            args.output / f"warm-{row['request_index']:03d}.raw",
            lambda _: None,
        )

    sampler = run_trace.ResourceSampler(args.output, args.server_pid, 0)
    sampler.start()
    time.sleep(0.6)
    paid_barrier_ready_ns = None
    paid_barrier_release_ns = None
    if args.paid_ready_file is not None:
        paid_barrier_ready_ns = time.monotonic_ns()
        write_monotonic_receipt(args.paid_ready_file, paid_barrier_ready_ns)
        paid_barrier_release_ns = await_monotonic_receipt(
            args.paid_release_file,
            args.paid_tail_timeout_s,
            paid_barrier_ready_ns,
            "paid release",
        )

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        first_token: list[int] = []
        dispatch_ns = time.monotonic_ns()
        value = run_server_trace.server_completion(
            args.port,
            row,
            args.output / f"stream-{row['request_index']:03d}.raw",
            first_token.append,
        )
        completion_ns = time.monotonic_ns()
        if len(first_token) != 1:
            raise RuntimeError("request did not produce one first-token receipt")
        return {
            "completion_ns": completion_ns,
            "dispatch_ns": dispatch_ns,
            "first_token_ns": first_token[0],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "predicted_ms": value["predicted_ms"],
            "prompt_ms": value["prompt_ms"],
            "request_index": row["request_index"],
            "tokens": value["tokens"],
            "wall_s": (completion_ns - dispatch_ns) / 1e9,
        }

    prefetch_arm_ns = None

    def arm_prefetch() -> None:
        nonlocal prefetch_arm_ns
        if args.prefetch_arm_file is None:
            return
        if args.prefetch_arm_file.exists():
            prefetch_arm_ns = await_monotonic_receipt(
                args.prefetch_arm_file, 1.0, 0, "prefetch arm"
            )
        else:
            prefetch_arm_ns = time.monotonic_ns()
            write_monotonic_receipt(args.prefetch_arm_file, prefetch_arm_ns)

    if args.dispatch == "sequential":
        paid_start_ns = time.monotonic_ns()
        arm_prefetch()
        results = [execute(row) for row in rows]
    else:
        start = threading.Barrier(len(rows) + 1)

        def execute_after_barrier(row: dict[str, Any]) -> dict[str, Any]:
            start.wait()
            return execute(row)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(rows), thread_name_prefix="qwen-shape"
        ) as pool:
            futures = [pool.submit(execute_after_barrier, row) for row in rows]
            paid_start_ns = time.monotonic_ns()
            arm_prefetch()
            start.wait()
            results = [future.result(timeout=3600) for future in futures]
    qwen_end_ns = max(row["completion_ns"] for row in results)
    paid_tail_end_ns = None
    if args.qwen_complete_file is not None:
        write_monotonic_receipt(args.qwen_complete_file, qwen_end_ns)
        paid_tail_end_ns = await_monotonic_receipt(
            args.paid_tail_file,
            args.paid_tail_timeout_s,
            paid_start_ns,
            "paid tail",
        )
    paid_end_ns = max(qwen_end_ns, paid_tail_end_ns or qwen_end_ns)
    time.sleep(0.6)
    sampler.stop()
    resource_rows = list(sampler.rows)
    server_energy = run_trace.server_energy_summary(
        resource_rows, paid_start_ns, paid_end_ns
    )
    result = {
        "arm": args.arm,
        "dispatch": args.dispatch,
        "indices": list(args.indices),
        "metrics": {
            "duration_s": (paid_end_ns - paid_start_ns) / 1e9,
            "requests": len(results),
        },
        "paid_end_ns": paid_end_ns,
        "paid_barrier_ready_ns": paid_barrier_ready_ns,
        "paid_barrier_release_ns": paid_barrier_release_ns,
        "paid_start_ns": paid_start_ns,
        "paid_tail_end_ns": paid_tail_end_ns,
        "prefetch_arm_ns": prefetch_arm_ns,
        "qwen_end_ns": qwen_end_ns,
        "repeat_index": args.repeat_index,
        "request_results": results,
        "schema": "s41-burstgpt-llama-server-result-v1",
        "server_command": server_command,
        "server_energy": server_energy,
        "status": "PASS",
    }
    run_trace.write_json(args.output / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
