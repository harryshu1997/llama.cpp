#!/usr/bin/env python3
"""Run the source-length BurstGPT trace through two llama-server instances."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import run_trace

from research_dev.scheduler import (  # noqa: E402
    ExecutionPlan,
    ExecutionPlanError,
    OperatorSplitPolicy,
    load_execution_plan,
)


CONFIRMATION = "RUN_BURSTGPT_LLAMA_SERVER_TRACE"
BALANCED_SPLIT_POLICY = "1:9664,2:8192,4:6144,128:8192,512:11136"
DECODE_REBALANCE_SPLIT_POLICY = (
    "1:8192,3:7168,6:6144,7:7168,8:6144,128:8192,512:11136"
)
DECODE_REBALANCE_R1_SPLIT_POLICY = (
    "1:9664,3:8192,6:6144,7:7168,8:6144,128:8192,512:11136"
)
DECODE_REBALANCE_R2_SPLIT_POLICY = (
    "1:9664,3:8192,8:6144,128:8192,512:11136"
)
HIDDEN_WAIT_SPLIT_POLICY = (
    "1:9664,3:8192,8:4096,128:8192,512:11136"
)
SPLIT_POLICIES = {
    "i1-balanced": BALANCED_SPLIT_POLICY,
    "i2-decode-rebalance": DECODE_REBALANCE_SPLIT_POLICY,
    "i2-r1-decode-rebalance": DECODE_REBALANCE_R1_SPLIT_POLICY,
    "i2-r2-decode-rebalance": DECODE_REBALANCE_R2_SPLIT_POLICY,
    "i3-hidden-wait": HIDDEN_WAIT_SPLIT_POLICY,
}


def validate_split_policy(policy_id: str, max_columns: int, table: str) -> None:
    run_trace.require(policy_id in SPLIT_POLICIES, "split policy identity")
    run_trace.require(
        max_columns == 11136 and table == SPLIT_POLICIES[policy_id],
        "split shape policy",
    )


def validate_execution_plan(
    args: argparse.Namespace,
    plan: ExecutionPlan,
) -> None:
    run_trace.require(plan.execution_mode == args.mode, "scheduler execution mode")
    run_trace.require(
        plan.trace_sha256.removeprefix("sha256:")
        == run_trace.SOURCE_REQUESTS_SHA256,
        "scheduler trace identity",
    )
    run_trace.require(
        plan.model_hashes == {
            "cold": "sha256:" + run_trace.COLD_MODEL_SHA256,
            "hot": "sha256:" + run_trace.HOT_MODEL_SHA256,
        },
        "scheduler model identity",
    )
    plan.validate_runtime_bindings({
        "cold_batch_size": args.cold_batch_size,
        "cold_context": args.cold_ctx_size,
        "cold_parallel": args.cold_parallel,
        "cold_repack": args.cold_repack,
        "cold_threads": args.cold_threads,
        "cold_ubatch_size": args.cold_ubatch_size,
        "hot_context": args.hot_ctx_size,
        "hot_parallel": args.hot_parallel,
        "mode": args.mode,
        "phone_dense_ffn_split": args.mode == "op15",
        "request_workers": args.request_workers,
        "split_io": args.split_io,
        "split_max_columns": args.max_columns,
        "split_policy_id": args.policy_id,
        "split_table": args.split_policy,
    })
    plan.validate_artifact(
        "hot_model", str(args.hot_model), "sha256:" + run_trace.HOT_MODEL_SHA256
    )
    plan.validate_artifact(
        "cold_model", str(args.cold_model), "sha256:" + run_trace.COLD_MODEL_SHA256
    )
    plan.validate_artifact(
        "hot_server", str(args.hot_server), run_trace.digest_file(args.hot_server)
    )
    plan.validate_artifact(
        "cold_server", str(args.cold_server), run_trace.digest_file(args.cold_server)
    )
    if args.mode == "op15":
        run_trace.require(plan.offload is not None, "scheduler offload contract")
        plan.validate_artifact(
            "bridge", str(args.bridge), run_trace.digest_file(args.bridge)
        )
        run_trace.require(
            plan.offload.split
            == OperatorSplitPolicy.from_table(
                policy_id=args.policy_id,
                operator_family=plan.offload.split.operator_family,
                layer_ids=plan.offload.split.layer_ids,
                n_embd=plan.offload.split.n_embd,
                eligible_columns=args.max_columns,
                max_tokens=args.cold_ubatch_size,
                column_quantum=plan.offload.split.column_quantum,
                alternate_columns=plan.offload.split.alternate_columns,
                io_type=args.split_io,
                weight_layout=plan.offload.split.weight_layout,
                table=args.split_policy,
            ),
            "scheduler split contract",
        )
        args.bridge_allocator = plan.offload.transport.allocator
        args.bridge_bind = plan.offload.backend.bridge_bind
        args.bridge_port = plan.offload.backend.bridge_port
        args.offload_contract = plan.offload
    else:
        run_trace.require(plan.offload is None, "control plan cannot contain offload")
        args.offload_contract = None


def parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for item in text.split(","):
        bounds = item.split("-", 1)
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise run_trace.RunError("CPU list integer") from error
        run_trace.require(first >= 0 and last >= first, "CPU list bounds")
        result.update(range(first, last + 1))
    run_trace.require(result, "empty CPU list")
    return result


def wait_server(process: run_trace.CapturedProcess, port: int) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        run_trace.require(
            process.process is not None and process.process.poll() is None,
            f"{process.label}: exited during load",
        )
        try:
            if run_trace.http_json(
                    f"http://127.0.0.1:{port}/health", 1
            ).get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise run_trace.RunError(f"{process.label}: readiness timeout")


def start_cold_server(
    args: argparse.Namespace,
    output: Path,
) -> run_trace.CapturedProcess:
    n_gpu_layers = getattr(args, "cold_n_gpu_layers", 0)
    command = [
        str(args.cold_server),
        "--model", str(args.cold_model),
        "--alias", run_trace.COLD_MODEL,
        "--fit", "off",
        "--ctx-size", str(args.cold_ctx_size),
        "--parallel", str(args.cold_parallel),
        "--batch-size", str(args.cold_batch_size),
        "--ubatch-size", str(args.cold_ubatch_size),
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--n-gpu-layers", str(n_gpu_layers),
        "--host", "127.0.0.1",
        "--port", str(args.cold_port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-timestamps",
        "--verbose",
    ]
    if int(n_gpu_layers) > 0:
        command.extend([
            "--flash-attn", "on",
            "--split-mode", "none",
            "--main-gpu", "0",
            "--device", "CUDA0",
        ])
    if args.cold_threads > 0:
        command.extend([
            "--threads", str(args.cold_threads),
            "--threads-batch", str(args.cold_threads),
        ])
    if args.cold_repack == "off":
        command.append("--no-repack")
    if args.cold_cpus:
        command = ["taskset", "--cpu-list", args.cold_cpus, *command]

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cold_lib_dir}:" + environment.get("LD_LIBRARY_PATH", "")
    )
    split_names = (
        "S41_SERVER_FFN_HOST",
        "S41_SERVER_FFN_PORT",
        "S41_SERVER_FFN_N_EMBD",
        "S41_SERVER_FFN_LAYER_MASK",
        "S41_SERVER_FFN_COLUMNS",
        "S41_SERVER_FFN_F16_IO",
        "S41_SERVER_FFN_M1_COLUMNS",
        "S41_SERVER_FFN_SMALL_M_COLUMNS",
        "S41_SERVER_FFN_LARGE_M_COLUMNS",
        "S41_SERVER_FFN_SMALL_M_MAX",
        "S41_SERVER_FFN_POLICY",
        "S41_SERVER_FFN_TIMEOUT_MS",
        "LLAMA_FFN_SPLIT_LAYER_MASK",
        "LLAMA_FFN_SPLIT_COLUMNS",
        "LLAMA_FFN_SPLIT_M1_COLUMNS",
        "LLAMA_FFN_SPLIT_SMALL_M_COLUMNS",
        "LLAMA_FFN_SPLIT_LARGE_M_COLUMNS",
        "LLAMA_FFN_SPLIT_SMALL_M_MAX",
        "LLAMA_FFN_SPLIT_POLICY",
        "LLAMA_FFN_SPLIT_VIEW_SAFE_WEIGHTS",
    )
    for name in split_names:
        environment.pop(name, None)
    if args.mode == "op15":
        offload = getattr(args, "offload_contract", None)
        if offload is not None:
            environment.update(offload.server_environment())
        else:
            environment.update({
                "S41_SERVER_FFN_HOST": "127.0.0.1",
                "S41_SERVER_FFN_PORT": str(args.bridge_port),
                "S41_SERVER_FFN_N_EMBD": "3840",
                "S41_SERVER_FFN_LAYER_MASK": getattr(
                    args, "ffn_layer_mask", "0x0000ffffffffffff"
                ),
                "S41_SERVER_FFN_COLUMNS": str(args.max_columns),
                "S41_SERVER_FFN_F16_IO": (
                    "1" if args.split_io == "f16" else "0"
                ),
                "S41_SERVER_FFN_POLICY": args.split_policy,
                "S41_SERVER_FFN_TIMEOUT_MS": str(
                    getattr(args, "ffn_timeout_ms", 35000)
                ),
            })

    process = run_trace.CapturedProcess(
        command, environment, output, "cold-server"
    )
    process.start()
    wait_server(process, args.cold_port)
    split_ready = [
        line for line in process.stderr_lines
        if line.startswith("S41SERVERFFN ready ")
    ]
    run_trace.require(
        len(split_ready) == (1 if args.mode == "op15" else 0),
        "cold server split readiness",
    )
    if int(n_gpu_layers) > 0:
        matches = re.findall(
            r"offloaded ([0-9]+)/([0-9]+) layers to GPU",
            "\n".join(process.stderr_lines),
        )
        run_trace.require(
            any(int(loaded) == int(n_gpu_layers) and int(total) > 0
                for loaded, total in matches),
            "cold model GPU layer placement",
        )
    return process


def server_completion(
    port: int,
    row: dict[str, Any],
    stream_path: Path,
    on_first: Callable[[int], None],
) -> dict[str, Any]:
    body = {
        "cache_prompt": False,
        "ignore_eos": True,
        "n_predict": row["output_tokens"],
        "prompt": row["prompt_tokens"],
        "return_tokens": True,
        "seed": row["request_index"],
        "stream": True,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=run_trace.canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final = None
    tokens: list[int] = []
    with stream_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                payload_line = raw_line.decode("utf-8").strip()
                if not payload_line.startswith("data:"):
                    continue
                payload = payload_line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                value = json.loads(payload)
                run_trace.require(
                    type(value) is dict and "error" not in value,
                    "server completion chunk",
                )
                chunk = value.get("tokens", [])
                run_trace.require(
                    type(chunk) is list and
                    all(type(token) is int for token in chunk),
                    "server completion tokens",
                )
                if chunk and not tokens:
                    on_first(time.monotonic_ns())
                tokens.extend(chunk)
                if value.get("stop", False):
                    final = value
    run_trace.require(final is not None, "server completion final")
    timings = final.get("timings")
    run_trace.require(
        type(timings) is dict and
        timings.get("prompt_n") == row["input_tokens"] and
        timings.get("predicted_n") == row["output_tokens"] and
        len(tokens) == row["output_tokens"],
        "server completion accounting",
    )
    return {
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"),
        "tokens": tokens,
    }


def concurrency_peak(rows: list[dict[str, Any]]) -> int:
    events = []
    for row in rows:
        events.append((row["dispatch_ns"], 1))
        events.append((row["completion_ns"], -1))
    active = 0
    peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def trace_metrics(
    results: list[dict[str, Any]],
    paid_start_ns: int,
) -> dict[str, Any]:
    by_role = {}
    for route in ("hot", "cold"):
        rows = [row for row in results if row["role"] == route]
        run_trace.require(rows, f"missing {route} results")
        makespan_s = (
            max(row["completion_ns"] for row in rows) - paid_start_ns
        ) / 1e9
        by_role[route] = {
            "completed": len(rows),
            "completion_s": run_trace.stats([
                (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in rows
            ]),
            "client_in_flight_peak": concurrency_peak(rows),
            "decode_s": run_trace.stats([
                row["predicted_ms"] / 1000 for row in rows
            ]),
            "makespan_s": makespan_s,
            "output_throughput_tokens_s": (
                sum(len(row["tokens"]) for row in rows) / makespan_s
            ),
            "output_tokens": sum(len(row["tokens"]) for row in rows),
            "prefill_s": run_trace.stats([
                row["prompt_ms"] / 1000 for row in rows
            ]),
            "service_s": run_trace.stats([
                (row["completion_ns"] - row["dispatch_ns"]) / 1e9
                for row in rows
            ]),
            "slo_met": sum(
                row["completion_ns"] - row["scheduled_arrival_ns"]
                <= row["slo_us"] * 1000
                for row in rows
            ),
            "ttft_s": run_trace.stats([
                (row["first_token_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in rows
            ]),
        }
    paid_end_ns = max(row["completion_ns"] for row in results)
    duration_s = (paid_end_ns - paid_start_ns) / 1e9
    return {
        "by_role": by_role,
        "completed": len(results),
        "duration_s": duration_s,
        "output_throughput_tokens_s": (
            sum(len(row["tokens"]) for row in results) / duration_s
        ),
        "output_tokens": sum(len(row["tokens"]) for row in results),
        "slo_met": sum(
            row["completion_ns"] - row["scheduled_arrival_ns"]
            <= row["slo_us"] * 1000
            for row in results
        ),
    }


def prefixed_json(
    lines: list[str], prefix: str, required: bool
) -> dict[str, Any] | None:
    matches = [line for line in lines if line.startswith(prefix)]
    run_trace.require(
        len(matches) == (1 if required else 0),
        f"summary count for {prefix.strip()}",
    )
    return json.loads(matches[0][len(prefix):]) if matches else None


def validate(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    paths = [
        args.requests,
        args.hot_server,
        args.hot_model,
        args.cuda_lib_dir,
        args.cold_server,
        args.cold_model,
        args.cold_lib_dir,
    ]
    if args.scheduler_plan is not None:
        paths.append(args.scheduler_plan)
    if args.mode == "op15":
        paths.append(args.bridge)
    for path in paths:
        run_trace.require(path is not None and path.exists(), f"missing path: {path}")
    run_trace.require(
        run_trace.digest_file(args.requests) == run_trace.SOURCE_REQUESTS_SHA256,
        "source trace identity",
    )
    run_trace.require(
        args.hot_model.stat().st_size == 9_001_752_960 and
        run_trace.digest_file(args.hot_model) == run_trace.HOT_MODEL_SHA256,
        "hot model identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176 and
        run_trace.digest_file(args.cold_model) == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )
    run_trace.require(
        args.cold_parallel > 0 and args.hot_parallel > 0 and
        args.request_workers >= 74 and
        0 < args.cold_ubatch_size <= 512 and
        args.cold_batch_size >= args.cold_ubatch_size and
        (args.cold_threads == -1 or args.cold_threads > 0),
        "server bounds",
    )
    if args.require_server_energy:
        try:
            run_trace.rapl_package_snapshot()
        except (FileNotFoundError, PermissionError, run_trace.RunError, ValueError) as error:
            raise run_trace.RunError("readable CPU package RAPL is required") from error
    validate_split_policy(args.policy_id, args.max_columns, args.split_policy)
    args.execution_plan = None
    args.offload_contract = None
    if args.scheduler_plan is not None:
        try:
            args.execution_plan = load_execution_plan(args.scheduler_plan)
            validate_execution_plan(args, args.execution_plan)
        except ExecutionPlanError as exc:
            raise run_trace.RunError(
                f"scheduler execution plan: {exc}"
            ) from exc
    available = set(os.sched_getaffinity(0))
    for value in (
        args.control_cpus, args.hot_cpus, args.cold_cpus, args.bridge_cpus
    ):
        if value:
            run_trace.require(
                parse_cpu_list(value) <= available,
                "requested CPU is unavailable",
            )

    requests = run_trace.read_jsonl(args.requests)
    run_trace.require(
        len(requests) == 74 and
        sum(run_trace.role(row) == "hot" for row in requests) == 57 and
        sum(run_trace.role(row) == "cold" for row in requests) == 17 and
        all(
            row["schema"] == "s41-gemma-qwen-request-semantic-source-v1" and
            row["input_tokens"] == row["source_input_tokens"] and
            row["output_tokens"] == row["source_output_tokens"] and
            len(row["prompt_tokens"]) == row["input_tokens"]
            for row in requests
        ),
        "source trace geometry",
    )
    run_trace.require(
        args.cold_ctx_size >= max(
            row["input_tokens"] + row["output_tokens"]
            for row in requests if run_trace.role(row) == "cold"
        ) and
        args.hot_ctx_size >= max(
            row["input_tokens"] + row["output_tokens"]
            for row in requests if run_trace.role(row) == "hot"
        ),
        "context coverage",
    )
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--hot-server", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cold-server", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--hot-port", type=int, default=18480)
    parser.add_argument("--cold-port", type=int, default=18481)
    parser.add_argument("--hot-ctx-size", type=int, default=24576)
    parser.add_argument("--hot-parallel", type=int, default=4)
    parser.add_argument("--cold-ctx-size", type=int, default=32768)
    parser.add_argument("--cold-parallel", type=int, default=8)
    parser.add_argument("--cold-batch-size", type=int, default=4096)
    parser.add_argument("--cold-ubatch-size", type=int, default=512)
    parser.add_argument("--cold-threads", type=int, default=-1)
    parser.add_argument("--cold-repack", choices=("default", "off"), default="default")
    parser.add_argument("--split-io", choices=("f16", "f32"), default="f16")
    parser.add_argument("--require-server-energy", action="store_true")
    parser.add_argument("--request-workers", type=int, default=74)
    parser.add_argument("--control-cpus")
    parser.add_argument("--hot-cpus")
    parser.add_argument("--cold-cpus")
    parser.add_argument("--bridge-cpus")
    parser.add_argument("--hot-nice", type=int, default=0)
    parser.add_argument("--max-columns", type=int, default=11136)
    parser.add_argument("--m1-columns", type=int, default=9664)
    parser.add_argument("--small-m-columns", type=int, default=8192)
    parser.add_argument("--large-m-columns", type=int, default=11136)
    parser.add_argument("--small-m-max", type=int, default=128)
    parser.add_argument(
        "--policy-id", choices=tuple(SPLIT_POLICIES), default="i1-balanced"
    )
    parser.add_argument("--split-policy", default=BALANCED_SPLIT_POLICY)
    parser.add_argument("--scheduler-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    requests = validate(args)
    if args.control_cpus:
        os.sched_setaffinity(0, parse_cpu_list(args.control_cpus))
    args.hot_workers = args.request_workers
    args.output.mkdir(parents=True)
    events = run_trace.EventWriter(args.output / "events.jsonl")
    hot = None
    bridge = None
    cold = None
    sampler = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    result_value = None
    failure = None
    try:
        hot = run_trace.start_hot(args, args.output)
        if args.mode == "op15":
            bridge = run_trace.start_bridge(args, args.output)
        cold = start_cold_server(args, args.output)

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            hot_future = pool.submit(
                server_completion, args.hot_port, hot_warm,
                args.output / "warm-hot.raw", lambda _: None,
            )
            cold_future = pool.submit(
                server_completion, args.cold_port, cold_warm,
                args.output / "warm-cold.raw", lambda _: None,
            )
            hot_future.result(timeout=3600)
            cold_future.result(timeout=3600)

        run_trace.require(
            run_trace.proc_status(hot.pid)["swap_bytes"] == 0 and
            run_trace.proc_status(cold.pid)["swap_bytes"] == 0,
            "model process swap before trace",
        )
        preflight = {
            "cold_model_sha256": run_trace.COLD_MODEL_SHA256,
            "cold_runtime": {
                "batch_size": args.cold_batch_size,
                "context": args.cold_ctx_size,
                "parallel": args.cold_parallel,
                "repack": args.cold_repack,
                "thread_selection": (
                    "llama_server_default" if args.cold_threads == -1
                    else "explicit"
                ),
                "threads": args.cold_threads,
                "ubatch_size": args.cold_ubatch_size,
            },
            "cold_server_sha256": run_trace.digest_file(args.cold_server),
            "cold_runtime_manifest": run_trace.process_runtime_manifest(
                cold.pid, [args.cold_lib_dir, args.cold_server.parent]
            ),
            "cpu_affinity": {
                "bridge": args.bridge_cpus,
                "cold": args.cold_cpus,
                "control": args.control_cpus,
                "hot": args.hot_cpus,
            },
            "gpu": run_trace.gpu_snapshot(),
            "hot_model_sha256": run_trace.HOT_MODEL_SHA256,
            "hot_runtime": {
                "context": args.hot_ctx_size,
                "parallel": args.hot_parallel,
            },
            "hot_server_sha256": run_trace.digest_file(args.hot_server),
            "hot_runtime_manifest": run_trace.process_runtime_manifest(
                hot.pid, [args.cuda_lib_dir, args.hot_server.parent]
            ),
            "mode": args.mode,
            "requests_sha256": run_trace.SOURCE_REQUESTS_SHA256,
            "schema": "s41-burstgpt-llama-server-preflight-v1",
            "scheduler": (
                None
                if args.execution_plan is None
                else {
                    "admission_phase": args.execution_plan.admission_phase,
                    "decision_reason": args.execution_plan.decision.reason,
                    "plan_id": args.execution_plan.plan_id,
                    "plan_sha256": args.execution_plan.plan_sha256,
                    "route_id": args.execution_plan.decision.route_id,
                    "work_set_hash": args.execution_plan.decision.work_set_hash,
                }
            ),
            "split_policy": {
                "id": args.policy_id,
                "io": args.split_io,
                "max_columns": args.max_columns,
                "table": args.split_policy,
                "weight_layout": (
                    "view_safe_dense_ffn" if args.mode == "op15"
                    else "not_applicable"
                ),
            },
            "system_memory": run_trace.system_memory(),
        }
        run_trace.write_json(args.output / "preflight.json", preflight)

        sampler = run_trace.ResourceSampler(args.output, hot.pid, cold.pid)
        sampler.start()
        time.sleep(0.6)
        paid_start_ns = time.monotonic_ns()
        events.write({
            "kind": "trace_start",
            "mode": args.mode,
            "repeat_index": args.repeat_index,
            "schema": "s41-burstgpt-llama-server-event-v1",
            "t_ns": paid_start_ns,
        })

        def run_request(row: dict[str, Any]) -> None:
            dispatch_ns = time.monotonic_ns()
            first: list[int] = []
            route = run_trace.role(row)
            port = args.hot_port if route == "hot" else args.cold_port
            try:
                value = server_completion(
                    port,
                    row,
                    args.output / f"stream-{row['request_index']:03d}.raw",
                    first.append,
                )
                run_trace.require(len(first) == 1, "first-token accounting")
                record = {
                    "completion_ns": time.monotonic_ns(),
                    "dispatch_ns": dispatch_ns,
                    "effective_model_id": (
                        run_trace.HOT_MODEL if route == "hot"
                        else run_trace.COLD_MODEL
                    ),
                    "event_id": row["event_id"],
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "request_index": row["request_index"],
                    "role": route,
                    "scheduled_arrival_ns": (
                        paid_start_ns + row["arrival_us"] * 1000
                    ),
                    "schema": "s41-burstgpt-llama-server-request-v1",
                    "slo_us": row["slo_us"],
                    "source_model_id": row["model_id"],
                    "tokens": value["tokens"],
                }
                with result_lock:
                    results.append(record)
                events.write({"kind": "request_complete", **record})
            except BaseException as error:
                with result_lock:
                    errors.append(
                        f"{row['request_index']}: "
                        f"{type(error).__name__}: {error}"
                    )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.request_workers,
            thread_name_prefix="trace-request",
        ) as pool:
            futures = []
            for row in requests:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while True:
                    remaining_ns = target_ns - time.monotonic_ns()
                    if remaining_ns <= 0:
                        break
                    time.sleep(min(remaining_ns / 1e9, 0.01))
                actual_ns = time.monotonic_ns()
                events.write({
                    "actual_t_ns": actual_ns,
                    "event_id": row["event_id"],
                    "kind": "request_arrival",
                    "request_index": row["request_index"],
                    "role": run_trace.role(row),
                    "scheduled_t_ns": target_ns,
                    "schema": "s41-burstgpt-llama-server-event-v1",
                })
                futures.append(pool.submit(run_request, row))
            for future in futures:
                future.result(timeout=7200)

        run_trace.require(not errors, "request errors: " + "; ".join(errors))
        run_trace.require(len(results) == len(requests), "request conservation")
        paid_end_ns = max(row["completion_ns"] for row in results)
        events.write({
            "kind": "trace_end",
            "schema": "s41-burstgpt-llama-server-event-v1",
            "t_ns": paid_end_ns,
        })
        time.sleep(0.6)
        sampler.stop()
        resource_rows = list(sampler.rows)
        sampler = None

        cold.terminate()
        if bridge is not None:
            bridge.terminate()
        hot.terminate()

        ffn_lines = [
            line for line in cold.stderr_lines
            if line.startswith("S41SERVERFFN {")
        ]
        run_trace.require(
            len(ffn_lines) == (1 if args.mode == "op15" else 0),
            "server FFN summary count",
        )
        ffn_summary = (
            json.loads(ffn_lines[0].split(" ", 1)[1])
            if ffn_lines else None
        )
        shape_summaries = [
            json.loads(line[len("S41SERVERFFNSHAPE "):])
            for line in cold.stderr_lines
            if line.startswith("S41SERVERFFNSHAPE ")
        ]
        bridge_summary = prefixed_json(
            bridge.stderr_lines if bridge is not None else [],
            "FFNDMABUF ", args.mode == "op15",
        )
        if args.mode == "op15":
            expected_columns = {
                columns for _, columns
                in run_trace.parse_prefill_policy(args.split_policy)
                if columns > 0
            }
            run_trace.require(
                ffn_summary is not None and bridge_summary is not None and
                ffn_summary["status"] == "ok" and
                bridge_summary["status"] == "ok" and
                ffn_summary["calls"] == bridge_summary["calls"] and
                bridge_summary["reset_recoveries"] == 0 and
                {shape["columns"] for shape in shape_summaries}
                    == expected_columns,
                "phone split summary",
            )
        run_trace.require(resource_rows, "resource samples")
        server_energy = None
        if args.require_server_energy:
            server_energy = run_trace.server_energy_summary(
                resource_rows, paid_start_ns, paid_end_ns
            )
        resource_summary = {
            "cold_rss_max_bytes": max(
                row["cold"]["rss_bytes"] for row in resource_rows
            ),
            "cold_swap_max_bytes": max(
                row["cold"]["swap_bytes"] for row in resource_rows
            ),
            "gpu_power_w": run_trace.stats([
                row["gpu"]["power_mw"] / 1000 for row in resource_rows
            ]),
            "gpu_utilization_pct": run_trace.stats([
                row["gpu"]["utilization_pct"] for row in resource_rows
            ]),
            "hot_rss_max_bytes": max(
                row["hot"]["rss_bytes"] for row in resource_rows
            ),
            "hot_swap_max_bytes": max(
                row["hot"]["swap_bytes"] for row in resource_rows
            ),
            "sample_count": len(resource_rows),
            "system_available_min_bytes": min(
                row["system"]["available_bytes"] for row in resource_rows
            ),
        }
        run_trace.require(
            resource_summary["cold_swap_max_bytes"] == 0 and
            resource_summary["hot_swap_max_bytes"] == 0,
            "model process swap during trace",
        )
        result_value = {
            "metrics": trace_metrics(results, paid_start_ns),
            "mode": args.mode,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "phone": {
                "bridge": bridge_summary,
                "ffn": ffn_summary,
                "shapes": shape_summaries,
            },
            "preflight": preflight,
            "repeat_index": args.repeat_index,
            "request_results": sorted(
                results, key=lambda row: row["request_index"]
            ),
            "resources": resource_summary,
            "server_energy": server_energy,
            "schema": "s41-burstgpt-llama-server-result-v1",
            "scheduler_plan_sha256": (
                None
                if args.execution_plan is None
                else args.execution_plan.plan_sha256
            ),
            "status": "PASS",
        }
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except BaseException:
                pass
        if cold is not None:
            cold.terminate()
        if bridge is not None:
            bridge.terminate()
        if hot is not None:
            hot.terminate()
        events.close()

    if failure is not None:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": failure,
            "mode": args.mode,
            "repeat_index": args.repeat_index,
            "schema": "s41-burstgpt-llama-server-failure-v1",
            "status": "FAIL",
        })
        return 2
    run_trace.require(result_value is not None, "missing result")
    run_trace.write_json(args.output / "RESULT.json", result_value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
