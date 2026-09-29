#!/usr/bin/env python3
"""Test dual CUDA weight residency with host or compressed KV caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any


HERE = Path(__file__).resolve().parent
BURST_DIR = HERE.parent / "burstgpt_gpu_cpu_op15_v1"
if not BURST_DIR.exists():
    BURST_DIR = Path(os.environ.get(
        "S41_BURST_DIR",
        (HERE.parent / "server_trace_v2"
         if (HERE.parent / "server_trace_v2").exists()
         else HERE.parent / "input"),
    ))
sys.path.insert(0, str(BURST_DIR))
sys.path.insert(0, str(HERE))

import run_gpu_cold_trace as gpu_trace  # noqa: E402
import run_hierarchical_trace as hierarchical_trace  # noqa: E402
import run_server_trace as server_trace  # noqa: E402
import run_trace  # noqa: E402


CONFIRMATION = "RUN_DUAL_CUDA_BURSTGPT_TRACE"


def start_qwen(
    args: argparse.Namespace,
) -> tuple[run_trace.CapturedProcess, float]:
    command = [
        str(args.server),
        "--model", str(args.hot_model),
        "--alias", run_trace.HOT_MODEL,
        "--fit", "off",
        "--ctx-size", str(args.hot_ctx_size),
        "--parallel", str(args.hot_parallel),
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--flash-attn", "on",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", args.hot_cache_type_k,
        "--cache-type-v", args.hot_cache_type_v,
        "--split-mode", "none",
        "--n-gpu-layers", "all",
        "--main-gpu", "0",
        "--device", "CUDA0",
        "--host", "127.0.0.1",
        "--port", str(args.hot_port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-timestamps",
        "--verbose",
    ]
    if args.hot_kv_location == "host":
        command.append("--no-kv-offload")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = (
        f"{args.lib_dir}:{args.server.parent}:"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    process = run_trace.CapturedProcess(
        command, environment, args.output, "dual-hot-server"
    )
    started = time.monotonic()
    process.start()
    gpu_trace.wait_server(process, args.hot_port)
    load_ms = (time.monotonic() - started) * 1000
    matches = re.findall(
        r"offloaded ([0-9]+)/([0-9]+) layers to GPU",
        "\n".join(process.stderr_lines),
    )
    run_trace.require(
        any(int(loaded) == int(total) and int(total) > 0
            for loaded, total in matches),
        "Qwen model is not fully GPU offloaded",
    )
    return process, load_ms


def cold_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        batch_size=args.cold_batch_size,
        cache_type_k=args.cold_cache_type_k,
        cache_type_v=args.cold_cache_type_v,
        ctx_size=args.cold_ctx_size,
        kv_offload=args.cold_kv_location == "gpu",
        lib_dir=args.lib_dir,
        model=args.cold_model,
        n_gpu_layers=args.cold_gpu_layers,
        output=args.output,
        parallel=args.cold_parallel,
        port=args.cold_port,
        server=args.server,
        ubatch_size=args.cold_ubatch_size,
    )


def validate(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    for path in (
        args.requests, args.server, args.hot_model, args.cold_model, args.lib_dir
    ):
        run_trace.require(path.exists(), f"missing: {path}")
    run_trace.require(
        run_trace.digest_file(args.requests)
        == run_trace.SOURCE_REQUESTS_SHA256,
        "source trace identity",
    )
    run_trace.require(
        args.hot_model.stat().st_size == 9_001_752_960
        and run_trace.digest_file(args.hot_model) == run_trace.HOT_MODEL_SHA256,
        "hot model identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176
        and run_trace.digest_file(args.cold_model) == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )
    requests = run_trace.read_jsonl(args.requests)
    run_trace.require(
        len(requests) == 74
        and sum(run_trace.role(row) == "hot" for row in requests) == 57
        and sum(run_trace.role(row) == "cold" for row in requests) == 17,
        "trace geometry",
    )
    for role_name, parallel, ctx_size in (
        ("hot", args.hot_parallel, args.hot_ctx_size),
        ("cold", args.cold_parallel, args.cold_ctx_size),
    ):
        lengths = sorted(
            (
                row["input_tokens"] + row["output_tokens"]
                for row in requests if run_trace.role(row) == role_name
            ),
            reverse=True,
        )
        run_trace.require(
            ctx_size >= sum(lengths[:parallel]),
            f"{role_name} context does not cover the largest active slots",
        )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--admission", choices=("hot-first", "concurrent"),
        default="hot-first",
    )
    parser.add_argument("--kv-location", choices=("host", "gpu"), default="host")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--hot-kv-location", choices=("host", "gpu"))
    parser.add_argument("--cold-kv-location", choices=("host", "gpu"))
    parser.add_argument("--hot-cache-type-k")
    parser.add_argument("--hot-cache-type-v")
    parser.add_argument("--cold-cache-type-k")
    parser.add_argument("--cold-cache-type-v")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hot-port", type=int, default=18480)
    parser.add_argument("--cold-port", type=int, default=18482)
    parser.add_argument("--hot-ctx-size", type=int, default=24576)
    parser.add_argument("--hot-parallel", type=int, default=4)
    parser.add_argument("--cold-ctx-size", type=int, default=32768)
    parser.add_argument("--cold-parallel", type=int, default=8)
    parser.add_argument("--cold-batch-size", type=int, default=4096)
    parser.add_argument("--cold-ubatch-size", type=int, default=512)
    parser.add_argument("--cold-gpu-layers", default="all")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    args.hot_kv_location = args.hot_kv_location or args.kv_location
    args.cold_kv_location = args.cold_kv_location or args.kv_location
    args.hot_cache_type_k = args.hot_cache_type_k or args.cache_type_k
    args.hot_cache_type_v = args.hot_cache_type_v or args.cache_type_v
    args.cold_cache_type_k = args.cold_cache_type_k or args.cache_type_k
    args.cold_cache_type_v = args.cold_cache_type_v or args.cache_type_v

    requests = validate(args)
    args.output.mkdir(parents=True)
    hot = None
    cold = None
    sampler = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()
    try:
        hot, hot_load_ms = start_qwen(args)
        cold, cold_load_ms = gpu_trace.start_server(cold_args(args))
        dual_resident_gpu = run_trace.gpu_snapshot()

        hot_rows = [row for row in requests if run_trace.role(row) == "hot"]
        cold_rows = [row for row in requests if run_trace.role(row) == "cold"]
        hot_warm = dict(min(
            hot_rows, key=lambda row: row["input_tokens"] + row["output_tokens"]
        ))
        cold_warm = dict(min(
            cold_rows, key=lambda row: row["input_tokens"] + row["output_tokens"]
        ))
        hot_warm["output_tokens"] = 2
        cold_warm["output_tokens"] = 2
        warm_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as warm_pool:
            hot_future = warm_pool.submit(
                server_trace.server_completion,
                args.hot_port,
                hot_warm,
                args.output / "warm-hot.raw",
                lambda _: None,
            )
            cold_future = warm_pool.submit(
                server_trace.server_completion,
                args.cold_port,
                cold_warm,
                args.output / "warm-cold.raw",
                lambda _: None,
            )
            hot_future.result(timeout=3600)
            cold_future.result(timeout=3600)
        warm_ms = (time.monotonic() - warm_started) * 1000

        if args.probe_only:
            post_warm_gpu = run_trace.gpu_snapshot()
            cold.terminate()
            cold = None
            hot.terminate()
            hot = None
            result = {
                "cold_cache_type_k": args.cold_cache_type_k,
                "cold_cache_type_v": args.cold_cache_type_v,
                "cold_gpu_layers": args.cold_gpu_layers,
                "cold_kv_location": args.cold_kv_location,
                "dual_resident_gpu": dual_resident_gpu,
                "hot_cache_type_k": args.hot_cache_type_k,
                "hot_cache_type_v": args.hot_cache_type_v,
                "hot_kv_location": args.hot_kv_location,
                "load_ms": {"cold": cold_load_ms, "hot": hot_load_ms},
                "post_warm_gpu": post_warm_gpu,
                "schema": "s41-dual-cuda-capacity-probe-v1",
                "status": "PASS",
                "warm_ms": warm_ms,
            }
            run_trace.write_json(args.output / "RESULT.json", result)
            print(json.dumps(result, sort_keys=True))
            return 0

        sampler = hierarchical_trace.DynamicSampler(args.output)
        sampler.set_pid("hot", hot.pid)
        sampler.set_pid("cold_gpu", cold.pid)
        sampler.start()
        time.sleep(0.5)
        paid_start_ns = time.monotonic_ns()

        def execute_request(row: dict[str, Any], port: int, route: str) -> None:
            dispatch_ns = time.monotonic_ns()
            first = []
            try:
                value = server_trace.server_completion(
                    port,
                    row,
                    args.output / f"stream-{route}-{row['request_index']:03d}.raw",
                    first.append,
                )
                run_trace.require(len(first) == 1, "first-token accounting")
                record = {
                    "completion_ns": time.monotonic_ns(),
                    "dispatch_ns": dispatch_ns,
                    "effective_model_id": (
                        run_trace.HOT_MODEL if run_trace.role(row) == "hot"
                        else run_trace.COLD_MODEL
                    ),
                    "event_id": row["event_id"],
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "request_index": row["request_index"],
                    "role": run_trace.role(row),
                    "route": route,
                    "scheduled_arrival_ns": paid_start_ns + row["arrival_us"] * 1000,
                    "slo_us": row["slo_us"],
                    "source_model_id": row["model_id"],
                    "tokens": value["tokens"],
                }
                with lock:
                    results.append(record)
            except BaseException as error:
                with lock:
                    errors.append(
                        f"{row['request_index']}: "
                        f"{type(error).__name__}: {error}"
                    )

        held = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=74) as pool:
            hot_futures = []
            cold_futures = []
            for row in requests:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while time.monotonic_ns() < target_ns:
                    time.sleep(min(
                        (target_ns - time.monotonic_ns()) / 1e9, 0.01
                    ))
                if run_trace.role(row) == "hot":
                    hot_futures.append(pool.submit(
                        execute_request, row, args.hot_port,
                        "qwen_cuda_dual_resident",
                    ))
                elif args.admission == "concurrent":
                    cold_futures.append(pool.submit(
                        execute_request, row, args.cold_port,
                        "gemma_cuda_dual_resident",
                    ))
                else:
                    held.append(row)
            for future in hot_futures:
                future.result(timeout=3600)
            run_trace.require(
                any(row["role"] == "hot" for row in results),
                "hot route failed: " + "; ".join(errors),
            )
            hot_end_ns = max(
                row["completion_ns"] for row in results
                if row["role"] == "hot"
            )
            if held:
                cold_futures.extend(
                    pool.submit(
                        execute_request, row, args.cold_port,
                        "gemma_cuda_dual_resident",
                    )
                    for row in held
                )
            for future in cold_futures:
                future.result(timeout=3600)

        run_trace.require(not errors, "request errors: " + "; ".join(errors))
        run_trace.require(len(results) == len(requests), "request conservation")
        paid_end_ns = max(row["completion_ns"] for row in results)
        time.sleep(0.5)
        sampler.stop()
        samples = list(sampler.rows)
        sampler = None
        cold.terminate()
        cold = None
        hot.terminate()
        hot = None
        result = {
            "admission": args.admission,
            "cache_type_k": (
                args.hot_cache_type_k
                if args.hot_cache_type_k == args.cold_cache_type_k
                else "mixed"
            ),
            "cache_type_v": (
                args.hot_cache_type_v
                if args.hot_cache_type_v == args.cold_cache_type_v
                else "mixed"
            ),
            "cold_cache_type_k": args.cold_cache_type_k,
            "cold_cache_type_v": args.cold_cache_type_v,
            "cold_kv_location": args.cold_kv_location,
            "dual_resident_gpu": dual_resident_gpu,
            "hot_cache_type_k": args.hot_cache_type_k,
            "hot_cache_type_v": args.hot_cache_type_v,
            "hot_kv_location": args.hot_kv_location,
            "kv_location": (
                args.hot_kv_location
                if args.hot_kv_location == args.cold_kv_location
                else "mixed"
            ),
            "load_ms": {"cold": cold_load_ms, "hot": hot_load_ms},
            "metrics": server_trace.trace_metrics(results, paid_start_ns),
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "request_results": sorted(results, key=lambda row: row["request_index"]),
            "resources": {
                "gpu_memory_used_max_bytes": max(
                    row["gpu"]["memory_used_bytes"] for row in samples
                ),
                "gpu_power_w": run_trace.stats([
                    row["gpu"]["power_mw"] / 1000 for row in samples
                ]),
                "gpu_utilization_pct": run_trace.stats([
                    row["gpu"]["utilization_pct"] for row in samples
                ]),
                "system_available_min_bytes": min(
                    row["system"]["available_bytes"] for row in samples
                ),
            },
            "route_metrics": hierarchical_trace.route_metrics(results, paid_start_ns),
            "schema": "s41-dual-cuda-burstgpt-result-v1",
            "status": "PASS",
            "warm_ms": warm_ms,
        }
        if all(row.get("rapl_package") is not None for row in samples):
            result["server_energy"] = run_trace.server_energy_summary(
                samples, paid_start_ns, paid_end_ns
            )
            if args.admission == "hot-first":
                result["phase_energy"] = {
                    "protected_hot": run_trace.server_energy_summary(
                        samples, paid_start_ns, hot_end_ns
                    ),
                    "cold": run_trace.server_energy_summary(
                        samples, hot_end_ns, paid_end_ns
                    ),
                }
        run_trace.write_json(args.output / "RESULT.json", result)
        print(json.dumps({
            "cold_makespan_s": result["metrics"]["by_role"]["cold"]["makespan_s"],
            "duration_s": result["metrics"]["duration_s"],
            "hot_makespan_s": result["metrics"]["by_role"]["hot"]["makespan_s"],
            "status": "PASS",
        }, sort_keys=True))
        return 0
    except BaseException as error:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s41-dual-cuda-burstgpt-failure-v1",
            "status": "FAIL",
        })
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        for process in (cold, hot):
            if process is not None:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
