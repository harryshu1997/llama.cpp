#!/usr/bin/env python3
"""Run a bounded Gemma CPU versus OP15 token-quality probe."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import threading
import time
from typing import Any

import run_server_trace
import run_trace


CONFIRMATION = "RUN_GEMMA_OP15_QUALITY_PROBE"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--split-io", choices=("f16", "f32"), default="f16")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--request-indices", default="0,2,32,43")
    parser.add_argument(
        "--cohort-sizes",
        help="comma-separated synchronized cohort sizes; uses the first request index",
    )
    parser.add_argument("--n-predict", type=int, default=8)
    parser.add_argument("--cold-server", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--cold-port", type=int, default=18481)
    parser.add_argument("--cold-ctx-size", type=int, default=4096)
    parser.add_argument("--cold-batch-size", type=int, default=2048)
    parser.add_argument("--cold-ubatch-size", type=int, default=512)
    parser.add_argument("--cold-threads", type=int, default=-1)
    parser.add_argument("--cold-repack", choices=("default", "off"), default="default")
    parser.add_argument("--max-columns", type=int, default=11136)
    parser.add_argument("--m1-columns", type=int, default=9664)
    parser.add_argument("--small-m-columns", type=int, default=8192)
    parser.add_argument("--large-m-columns", type=int, default=11136)
    parser.add_argument("--small-m-max", type=int, default=128)
    parser.add_argument(
        "--policy-id",
        choices=tuple(run_server_trace.SPLIT_POLICIES),
        default="i1-balanced",
    )
    parser.add_argument(
        "--split-policy", default=run_server_trace.BALANCED_SPLIT_POLICY
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    run_trace.require(args.execute and args.confirm == CONFIRMATION, "confirmation")
    run_trace.require(args.output.is_absolute() and not args.output.exists(), "output")
    run_trace.require(args.n_predict > 0, "n_predict")
    run_server_trace.validate_split_policy(
        args.policy_id, args.max_columns, args.split_policy
    )
    for path in (args.requests, args.cold_server, args.cold_model, args.cold_lib_dir):
        run_trace.require(path.exists(), f"missing path: {path}")
    if args.mode == "op15":
        run_trace.require(args.bridge is not None and args.bridge.exists(), "bridge")
    run_trace.require(
        run_trace.digest_file(args.requests) == run_trace.SOURCE_REQUESTS_SHA256,
        "source trace identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176 and
        run_trace.digest_file(args.cold_model) == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )
    try:
        selected_indices = [int(value) for value in args.request_indices.split(",")]
    except ValueError as error:
        raise run_trace.RunError("request indices") from error
    run_trace.require(len(selected_indices) == len(set(selected_indices)), "duplicate index")
    try:
        cohort_sizes = (
            [int(value) for value in args.cohort_sizes.split(",")]
            if args.cohort_sizes else []
        )
    except ValueError as error:
        raise run_trace.RunError("cohort sizes") from error
    run_trace.require(
        not cohort_sizes or (
            len(selected_indices) == 1 and
            len(cohort_sizes) == len(set(cohort_sizes)) and
            all(1 <= size <= 8 for size in cohort_sizes)
        ),
        "cohort geometry",
    )
    requests = {
        row["request_index"]: row for row in run_trace.read_jsonl(args.requests)
        if run_trace.role(row) == "cold"
    }
    run_trace.require(all(index in requests for index in selected_indices), "cold indices")

    args.cold_parallel = max(cohort_sizes, default=1)
    args.control_cpus = None
    args.hot_cpus = None
    args.cold_cpus = None
    args.bridge_cpus = None
    args.require_server_energy = False
    args.output.mkdir(parents=True)
    bridge = None
    cold = None
    result: dict[str, Any] | None = None
    failure = None
    try:
        if args.mode == "op15":
            bridge = run_trace.start_bridge(args, args.output)
        cold = run_server_trace.start_cold_server(args, args.output)
        warm = dict(min(requests.values(), key=lambda row: row["input_tokens"]))
        warm["output_tokens"] = 2
        run_server_trace.server_completion(
            args.cold_port, warm, args.output / "warm.raw", lambda _: None
        )

        rows = []
        paid_start_ns = time.monotonic_ns()
        if cohort_sizes:
            source_index = selected_indices[0]
            for cohort_size in cohort_sizes:
                barrier = threading.Barrier(cohort_size)

                def run_member(member_index: int) -> dict[str, Any]:
                    request = dict(requests[source_index])
                    request["output_tokens"] = args.n_predict
                    request["request_index"] = (
                        1_000_000 + cohort_size * 100 + member_index
                    )
                    barrier.wait(timeout=30)
                    started_ns = time.monotonic_ns()
                    first: list[int] = []
                    value = run_server_trace.server_completion(
                        args.cold_port,
                        request,
                        args.output / (
                            f"stream-c{cohort_size}-m{member_index}.raw"
                        ),
                        first.append,
                    )
                    completed_ns = time.monotonic_ns()
                    run_trace.require(len(first) == 1, "first-token accounting")
                    return {
                        "cohort_size": cohort_size,
                        "completion_ns": completed_ns,
                        "decode_ms": value["predicted_ms"],
                        "first_token_ns": first[0],
                        "input_tokens": request["input_tokens"],
                        "member_index": member_index,
                        "output_tokens": args.n_predict,
                        "prefill_ms": value["prompt_ms"],
                        "request_index": source_index,
                        "started_ns": started_ns,
                        "tokens": value["tokens"],
                    }

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=cohort_size
                ) as pool:
                    futures = [
                        pool.submit(run_member, member_index)
                        for member_index in range(cohort_size)
                    ]
                    rows.extend(future.result(timeout=3600) for future in futures)
        else:
            for index in selected_indices:
                request = dict(requests[index])
                request["output_tokens"] = args.n_predict
                started_ns = time.monotonic_ns()
                first: list[int] = []
                value = run_server_trace.server_completion(
                    args.cold_port,
                    request,
                    args.output / f"stream-{index:03d}.raw",
                    first.append,
                )
                completed_ns = time.monotonic_ns()
                run_trace.require(len(first) == 1, "first-token accounting")
                rows.append({
                    "completion_ns": completed_ns,
                    "decode_ms": value["predicted_ms"],
                    "first_token_ns": first[0],
                    "input_tokens": request["input_tokens"],
                    "output_tokens": args.n_predict,
                    "prefill_ms": value["prompt_ms"],
                    "request_index": index,
                    "started_ns": started_ns,
                    "tokens": value["tokens"],
                })
        paid_end_ns = time.monotonic_ns()
        server_runtime_manifest = run_trace.process_runtime_manifest(
            cold.pid, [args.cold_lib_dir, args.cold_server.parent]
        )

        cold.terminate()
        ffn_lines = [
            line for line in cold.stderr_lines
            if line.startswith("S41SERVERFFN {")
        ]
        run_trace.require(
            len(ffn_lines) == (1 if args.mode == "op15" else 0),
            "server FFN summary count",
        )
        ffn_summary = (
            json.loads(ffn_lines[0].split(" ", 1)[1]) if ffn_lines else None
        )
        shape_summaries = [
            json.loads(line[len("S41SERVERFFNSHAPE "):])
            for line in cold.stderr_lines
            if line.startswith("S41SERVERFFNSHAPE ")
        ]
        if bridge is not None:
            bridge.terminate()
        bridge_summary = run_server_trace.prefixed_json(
            bridge.stderr_lines if bridge is not None else [],
            "FFNDMABUF ", args.mode == "op15",
        )
        if args.mode == "op15":
            run_trace.require(
                ffn_summary is not None and bridge_summary is not None and
                ffn_summary["status"] == "ok" and bridge_summary["status"] == "ok" and
                ffn_summary["calls"] == bridge_summary["calls"] and
                bridge_summary["reset_recoveries"] == 0,
                "phone summary",
            )
        result = {
            "cold_model_sha256": run_trace.COLD_MODEL_SHA256,
            "cohort_sizes": cohort_sizes,
            "duration_s": (paid_end_ns - paid_start_ns) / 1e9,
            "mode": args.mode,
            "phone": {
                "bridge": bridge_summary,
                "ffn": ffn_summary,
                "shapes": shape_summaries,
            },
            "request_results": rows,
            "schema": "s41-gemma-op15-quality-probe-v1",
            "server_runtime_manifest": server_runtime_manifest,
            "split_policy": {
                "id": args.policy_id,
                "table": args.split_policy,
            },
            "split_io": args.split_io if args.mode == "op15" else None,
            "status": "PASS",
        }
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if cold is not None:
            cold.terminate()
        if bridge is not None:
            bridge.terminate()

    if failure is not None:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": failure,
            "mode": args.mode,
            "schema": "s41-gemma-op15-quality-probe-failure-v1",
            "status": "FAIL",
        })
        return 2
    run_trace.require(result is not None, "missing result")
    run_trace.write_json(args.output / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
