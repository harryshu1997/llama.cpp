#!/usr/bin/env python3
"""Measure the cold BurstGPT requests on a CUDA or CUDA+CPU model."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error


HERE = Path(__file__).resolve().parent
UNIFIED_REPO_ROOT = Path(os.environ.get(
    "S42_UNIFIED_REPO_ROOT", HERE.parents[4]
))
if str(UNIFIED_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIFIED_REPO_ROOT))
BURST_DIR = HERE.parent / "burstgpt_gpu_cpu_op15_v1"
if not BURST_DIR.exists():
    BURST_DIR = Path(os.environ.get(
        "S41_BURST_DIR",
        (HERE.parent / "server_trace_v2"
         if (HERE.parent / "server_trace_v2").exists()
         else HERE.parent / "input"),
    ))
sys.path.insert(0, str(BURST_DIR))

import run_server_trace as server_trace  # noqa: E402
import run_trace  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    ExecutionPlan,
    ExecutionPlanError,
    load_execution_plan,
)


CONFIRMATION = "RUN_GEMMA_GPU_TRACE"
FP16_COLD_MODEL_BYTES = 23_832_065_056
FP16_COLD_MODEL_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
SPLIT_ENV_NAMES = (
    "S41_SERVER_FFN_HOST",
    "S41_SERVER_FFN_PORT",
    "S41_SERVER_FFN_N_EMBD",
    "S41_SERVER_FFN_LAYER_MASK",
    "S41_SERVER_FFN_COLUMNS",
    "S41_SERVER_FFN_F16_IO",
    "S41_SERVER_FFN_ACTIVATION",
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


def cold_work_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(run_trace.canonical(row))
    return digest.hexdigest()


def bridge_supports_allocator(path: Path, allocator: str) -> bool:
    result = subprocess.run(
        [str(path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 2 and allocator in (
        result.stdout + "\n" + result.stderr
    )


def validate_phone_work(
    plan: ExecutionPlan,
    bridge_summary: dict[str, Any],
    ffn_summary: dict[str, Any],
    shape_summaries: list[dict[str, Any]],
) -> None:
    offload = plan.offload
    run_trace.require(offload is not None, "phone offload contract")
    split = offload.split
    layer_count = len(split.layer_ids)
    shape_calls: list[tuple[int, int]] = []
    for row in shape_summaries:
        tokens = row.get("tokens")
        calls = row.get("calls")
        columns = row.get("columns")
        run_trace.require(
            type(tokens) is int
            and type(calls) is int
            and calls > 0
            and calls % layer_count == 0
            and columns == split.columns_for(tokens)
            and columns > 0,
            "phone split shape accounting",
        )
        shape_calls.append((tokens, calls))
    work = split.summarize(shape_calls)
    expected = plan.expected_work
    run_trace.require(
        shape_calls
        and work.token_rows % layer_count == 0
        and bridge_summary.get("status") == "ok"
        and ffn_summary.get("status") == "ok"
        and bridge_summary.get("allocator") == offload.transport.allocator
        and bridge_summary.get("max_wire_bytes")
            == offload.transport.max_wire_bytes
        and bridge_summary.get("calls") == work.calls
        and ffn_summary.get("calls") == work.calls
        and bridge_summary.get("upload_bytes") == work.upload_bytes
        and bridge_summary.get("download_bytes") == work.download_bytes
        and ffn_summary.get("upload_bytes") == work.upload_bytes
        and ffn_summary.get("download_bytes") == work.download_bytes
        and bridge_summary.get("reset_recoveries") == 0
        and expected["phone_calls_min"] <= work.calls
            <= expected["phone_calls_max"]
        and expected["phone_transfer_bytes_min"] <= work.upload_bytes
            <= expected["phone_transfer_bytes_max"]
        and expected["phone_macs_min"] <= work.phone_macs
            <= expected["phone_macs_max"]
        and expected["phone_token_rows_per_layer_min"]
            <= work.token_rows // layer_count
            <= expected["phone_token_rows_per_layer_max"],
        "phone split work summary",
    )


def validate_execution_plan(
    args: argparse.Namespace,
    plan: ExecutionPlan,
    requests: list[dict[str, Any]],
) -> None:
    expected_mode = args.mode
    run_trace.require(
        expected_mode in {"cuda-cpu", "cuda-cpu-op15"}
        and plan.execution_mode == expected_mode,
        "scheduler execution mode",
    )
    run_trace.require(
        plan.trace_sha256.removeprefix("sha256:")
        == run_trace.SOURCE_REQUESTS_SHA256,
        "scheduler source trace identity",
    )
    run_trace.require(
        plan.model_hashes
        == {"cold": "sha256:" + FP16_COLD_MODEL_SHA256},
        "scheduler model identity",
    )
    try:
        n_gpu_layers = int(args.n_gpu_layers)
    except ValueError as exc:
        raise run_trace.RunError("scheduler GPU layer count") from exc
    plan.validate_runtime_bindings({
        "arrival_mode": args.arrival_mode,
        "batch_size": args.batch_size,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "context": args.ctx_size,
        "cold_work_sha256": "sha256:" + cold_work_sha256(requests),
        "dispatch_order": args.dispatch_order,
        "kv_offload": args.kv_offload,
        "lib_dir": str(args.lib_dir),
        "mode": args.mode,
        "n_gpu_layers": n_gpu_layers,
        "parallel": args.parallel,
        "port": args.port,
        "source_trace_requests": 74,
        "trace_filter": "cold",
        "trace_path": str(args.requests),
        "ubatch_size": args.ubatch_size,
    })
    run_trace.require(
        plan.request_count == len(requests)
        and plan.input_tokens == sum(row["input_tokens"] for row in requests)
        and plan.output_tokens == sum(row["output_tokens"] for row in requests),
        "scheduler cold cohort work",
    )
    placement = plan.layer_placement
    run_trace.require(
        placement is not None
        and placement.selected.total_layers == 48
        and placement.selected.runtime_gpu_layers == n_gpu_layers
        and placement.cpu_layer_spec == "0-22"
        and placement.gpu_layer_spec == "23-47",
        "scheduler layer placement",
    )
    gpu_state = run_trace.gpu_snapshot()
    host_state = run_trace.system_memory()
    run_trace.require(
        gpu_state["memory_total_bytes"] == placement.gpu_capacity.capacity_bytes
        and gpu_state["memory_free_bytes"]
            >= placement.selected.gpu_total_bytes
                + placement.gpu_capacity.reserve_bytes,
        "fresh GPU capacity gate",
    )
    run_trace.require(
        host_state["available_bytes"]
            >= placement.selected.cpu_total_bytes
                + placement.cpu_capacity.reserve_bytes,
        "fresh host capacity gate",
    )
    plan.validate_artifact(
        "cold_model", str(args.model), "sha256:" + FP16_COLD_MODEL_SHA256
    )
    plan.validate_artifact(
        "cold_server", str(args.server), run_trace.digest_file(args.server)
    )
    if args.mode == "cuda-cpu-op15":
        run_trace.require(
            plan.offload is not None and args.bridge is not None,
            "scheduler phone offload contract",
        )
        plan.validate_artifact(
            "bridge", str(args.bridge), run_trace.digest_file(args.bridge)
        )
        run_trace.require(
            bridge_supports_allocator(
                args.bridge, plan.offload.transport.allocator
            ),
            "bridge allocator capability",
        )
        run_trace.require(
            plan.offload.backend.layer_spec == placement.cpu_layer_spec
            and plan.offload.split.eligible_columns == 6144
            and plan.offload.split.max_tokens == 512
            and plan.offload.split.column_quantum in {512, 1024}
            and all(
                row.phone_columns == 0
                or (
                    row.phone_columns <= 6144
                    and (
                        row.phone_columns == 6144
                        or row.phone_columns
                            in plan.offload.split.alternate_columns
                        or row.phone_columns
                            % plan.offload.split.column_quantum == 0
                    )
                )
                for row in plan.offload.split.buckets
            )
            and all(
                plan.offload.split.columns_for(tokens) > 0
                for tokens in range(1, 17)
            )
            and plan.offload.split.columns_for(17) == 0
            and plan.offload.split.io_type == "f16",
            "scheduler phone split placement",
        )
        args.bridge_allocator = plan.offload.transport.allocator
        args.bridge_bind = plan.offload.backend.bridge_bind
        args.bridge_port = plan.offload.backend.bridge_port
        args.offload_contract = plan.offload
    else:
        run_trace.require(plan.offload is None, "control plan contains phone offload")
        args.offload_contract = None


class Sampler:
    def __init__(self, process: run_trace.CapturedProcess) -> None:
        self.process = process
        self.rows: list[dict[str, Any]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        run_trace.require(not self.thread.is_alive(), "sampler stop")

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                gpu_before_ns = time.monotonic_ns()
                gpu = run_trace.gpu_snapshot()
                gpu_after_ns = time.monotonic_ns()
                gpu["sample_t_ns"] = (gpu_before_ns + gpu_after_ns) // 2
                try:
                    rapl_package = run_trace.rapl_package_snapshot()
                except (FileNotFoundError, PermissionError,
                        run_trace.RunError, ValueError):
                    rapl_package = None
                self.rows.append({
                    "gpu": gpu,
                    "process": run_trace.proc_status(self.process.pid),
                    "rapl_package": rapl_package,
                    "system": run_trace.system_memory(),
                    "t_ns": time.monotonic_ns(),
                })
            except (FileNotFoundError, ProcessLookupError):
                return
            self.stop_event.wait(0.2)


def wait_server(process: run_trace.CapturedProcess, port: int) -> None:
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        run_trace.require(
            process.process is not None and process.process.poll() is None,
            "GPU server exited during load",
        )
        try:
            health = run_trace.http_json(
                f"http://127.0.0.1:{port}/health", 1
            )
            if health.get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    raise run_trace.RunError("GPU server readiness timeout")


def start_server(
    args: argparse.Namespace,
) -> tuple[run_trace.CapturedProcess, float]:
    command = [
        str(args.server),
        "--model", str(args.model),
        "--alias", run_trace.COLD_MODEL,
        "--fit", "off",
        "--ctx-size", str(args.ctx_size),
        "--parallel", str(args.parallel),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--flash-attn", "on",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", getattr(args, "cache_type_k", "f16"),
        "--cache-type-v", getattr(args, "cache_type_v", "f16"),
        "--split-mode", "none",
        "--n-gpu-layers", str(getattr(args, "n_gpu_layers", "all")),
        "--main-gpu", "0",
        "--device", "CUDA0",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-timestamps",
        "--verbose",
    ]
    if not getattr(args, "kv_offload", True):
        command.append("--no-kv-offload")
    environment = os.environ.copy()
    for name in SPLIT_ENV_NAMES:
        environment.pop(name, None)
    offload = getattr(args, "offload_contract", None)
    if offload is not None:
        environment.update(offload.server_environment())
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = (
        f"{args.lib_dir}:{args.server.parent}:"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    process = run_trace.CapturedProcess(
        command, environment, args.output, "cold-gpu-server"
    )
    started = time.monotonic()
    process.start()
    wait_server(process, args.port)
    load_ms = (time.monotonic() - started) * 1000
    log = "\n".join(process.stderr_lines)
    split_ready = [
        line for line in process.stderr_lines
        if line.startswith("S41SERVERFFN ready ")
    ]
    run_trace.require(
        len(split_ready) == (1 if offload is not None else 0),
        "cold server split readiness",
    )
    matches = re.findall(r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log)
    requested_layers = getattr(args, "n_gpu_layers", "all")
    if str(requested_layers) == "all":
        matched = any(
            int(loaded) == int(total) and int(total) > 0
            for loaded, total in matches
        )
    else:
        matched = any(
            int(loaded) == int(requested_layers) and int(total) > 0
            for loaded, total in matches
        )
    run_trace.require(matched, "cold model GPU layer placement")
    return process, load_ms


def metrics(
    rows: list[dict[str, Any]], paid_start_ns: int
) -> dict[str, Any]:
    makespan_s = (
        max(row["completion_ns"] for row in rows) - paid_start_ns
    ) / 1e9
    return {
        "completed": len(rows),
        "completion_s": run_trace.stats([
            (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        ]),
        "decode_s": run_trace.stats([
            row["predicted_ms"] / 1000 for row in rows
        ]),
        "makespan_s": makespan_s,
        "output_throughput_tokens_s": (
            sum(row["output_tokens"] for row in rows) / makespan_s
        ),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "prefill_s": run_trace.stats([
            row["prompt_ms"] / 1000 for row in rows
        ]),
        "service_s": run_trace.stats([
            (row["completion_ns"] - row["dispatch_ns"]) / 1e9
            for row in rows
        ]),
        "ttft_s": run_trace.stats([
            (row["first_token_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        ]),
    }


def validate(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    paths = [args.requests, args.server, args.model, args.lib_dir]
    if args.scheduler_plan is not None:
        paths.append(args.scheduler_plan)
    if args.mode == "cuda-cpu-op15":
        paths.append(args.bridge)
    for path in paths:
        run_trace.require(path is not None and path.exists(), f"missing path: {path}")
    run_trace.require(
        args.mode == "standalone" or args.scheduler_plan is not None,
        "unified scheduler plan is required",
    )
    for path in (args.requests, args.server, args.model, args.lib_dir):
        run_trace.require(path.exists(), f"missing path: {path}")
    run_trace.require(
        run_trace.digest_file(args.requests)
        == run_trace.SOURCE_REQUESTS_SHA256,
        "source trace identity",
    )
    if args.model_profile == "fp16-proxy":
        expected_size = FP16_COLD_MODEL_BYTES
        expected_sha256 = FP16_COLD_MODEL_SHA256
    else:
        expected_size = 6_975_878_176
        expected_sha256 = run_trace.COLD_MODEL_SHA256
    run_trace.require(
        args.model.stat().st_size == expected_size
        and run_trace.digest_file(args.model) == expected_sha256,
        "cold model identity",
    )
    requests = [
        row for row in run_trace.read_jsonl(args.requests)
        if run_trace.role(row) == "cold"
    ]
    run_trace.require(len(requests) == 17, "cold request count")
    run_trace.require(
        sum(row["input_tokens"] for row in requests) == 11_476
        and sum(row["output_tokens"] for row in requests) == 6_919,
        "cold request work",
    )
    run_trace.require(
        args.ctx_size >= max(
            row["input_tokens"] + row["output_tokens"] for row in requests
        ),
        "context coverage",
    )
    if args.require_server_energy:
        try:
            run_trace.rapl_package_snapshot()
        except (
            FileNotFoundError,
            PermissionError,
            run_trace.RunError,
            ValueError,
        ) as exc:
            raise run_trace.RunError(
                "readable CPU package RAPL is required"
            ) from exc
    args.execution_plan = None
    args.offload_contract = None
    if args.scheduler_plan is not None:
        try:
            args.execution_plan = load_execution_plan(args.scheduler_plan)
            validate_execution_plan(args, args.execution_plan, requests)
        except ExecutionPlanError as exc:
            raise run_trace.RunError(
                f"scheduler execution plan: {exc}"
            ) from exc
    return requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-profile", choices=("q4", "fp16-proxy"), default="q4"
    )
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("standalone", "cuda-cpu", "cuda-cpu-op15"),
        default="standalone",
    )
    parser.add_argument("--repeat-index", type=int, default=1)
    parser.add_argument("--scheduler-plan", type=Path)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--require-server-energy", action="store_true")
    parser.add_argument("--port", type=int, default=18482)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--n-gpu-layers", default="all")
    parser.add_argument("--cache-type-k", default="f16")
    parser.add_argument("--cache-type-v", default="f16")
    parser.add_argument("--no-kv-offload", dest="kv_offload", action="store_false")
    parser.set_defaults(kv_offload=True)
    parser.add_argument(
        "--arrival-mode", choices=("source", "backlog"), default="backlog"
    )
    parser.add_argument(
        "--dispatch-order", choices=("source", "lpt"), default="source"
    )
    parser.add_argument("--order-profile", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    requests = validate(args)
    if args.dispatch_order == "lpt":
        run_trace.require(
            args.arrival_mode == "backlog"
            and args.order_profile is not None
            and args.order_profile.exists(),
            "LPT requires a backlog and an existing order profile",
        )
        profile = json.loads(args.order_profile.read_text())
        service_ms = {
            row["request_index"]: row["prompt_ms"] + row["predicted_ms"]
            for row in profile["request_results"]
        }
        run_trace.require(
            set(service_ms) == {row["request_index"] for row in requests},
            "order profile request coverage",
        )
        requests.sort(
            key=lambda row: (-service_ms[row["request_index"]],
                             row["request_index"])
        )
    args.output.mkdir(parents=True)
    process = None
    bridge = None
    sampler = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()
    try:
        if args.mode == "cuda-cpu-op15":
            bridge = run_trace.start_bridge(args, args.output)
        process, load_ms = start_server(args)
        warm = dict(min(
            requests,
            key=lambda row: row["input_tokens"] + row["output_tokens"],
        ))
        warm["output_tokens"] = 2
        warm_started = time.monotonic()
        server_trace.server_completion(
            args.port,
            warm,
            args.output / "warm.raw",
            lambda _: None,
        )
        warm_ms = (time.monotonic() - warm_started) * 1000

        sampler = Sampler(process)
        sampler.start()
        time.sleep(0.5)
        paid_start_ns = time.monotonic_ns()
        first_source_arrival_us = min(row["arrival_us"] for row in requests)

        def run_request(row: dict[str, Any], scheduled_ns: int) -> None:
            dispatch_ns = time.monotonic_ns()
            first: list[int] = []
            try:
                value = server_trace.server_completion(
                    args.port,
                    row,
                    args.output / f"stream-{row['request_index']:03d}.raw",
                    first.append,
                )
                run_trace.require(len(first) == 1, "first-token accounting")
                result = {
                    "completion_ns": time.monotonic_ns(),
                    "dispatch_ns": dispatch_ns,
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "output_tokens": row["output_tokens"],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "request_index": row["request_index"],
                    "route": (
                        "gemma_cuda_cpu_op15_f16"
                        if args.mode == "cuda-cpu-op15"
                        else (
                            "gemma_cuda_cpu_f16"
                            if args.model_profile == "fp16-proxy"
                            else "gemma_cuda_full"
                        )
                    ),
                    "scheduled_arrival_ns": scheduled_ns,
                    "tokens": value["tokens"],
                }
                with lock:
                    results.append(result)
            except BaseException as error:
                with lock:
                    errors.append(
                        f"{row['request_index']}: "
                        f"{type(error).__name__}: {error}"
                    )

        with concurrent.futures.ThreadPoolExecutor(max_workers=17) as pool:
            futures = []
            for row in requests:
                offset_us = 0
                if args.arrival_mode == "source":
                    offset_us = row["arrival_us"] - first_source_arrival_us
                target_ns = paid_start_ns + offset_us * 1000
                while time.monotonic_ns() < target_ns:
                    time.sleep(min((target_ns - time.monotonic_ns()) / 1e9, 0.01))
                futures.append(pool.submit(run_request, row, target_ns))
            for future in futures:
                future.result(timeout=3600)

        run_trace.require(not errors, "request errors: " + "; ".join(errors))
        run_trace.require(len(results) == len(requests), "request conservation")
        paid_end_ns = max(row["completion_ns"] for row in results)
        time.sleep(0.5)
        sampler.stop()
        samples = list(sampler.rows)
        sampler = None
        with (args.output / "resource-samples.jsonl").open("xb") as stream:
            for row in samples:
                stream.write(run_trace.canonical(row))
        process.terminate()
        if bridge is not None:
            bridge.terminate()

        ffn_lines = [
            line for line in process.stderr_lines
            if line.startswith("S41SERVERFFN {")
        ]
        run_trace.require(
            len(ffn_lines) == (1 if args.mode == "cuda-cpu-op15" else 0),
            "server FFN summary count",
        )
        ffn_summary = (
            json.loads(ffn_lines[0].split(" ", 1)[1])
            if ffn_lines else None
        )
        shape_summaries = [
            json.loads(line[len("S41SERVERFFNSHAPE "):])
            for line in process.stderr_lines
            if line.startswith("S41SERVERFFNSHAPE ")
        ]
        bridge_summary = server_trace.prefixed_json(
            bridge.stderr_lines if bridge is not None else [],
            "FFNDMABUF ",
            args.mode == "cuda-cpu-op15",
        )
        plan = args.execution_plan
        if args.mode == "cuda-cpu-op15":
            run_trace.require(
                ffn_summary is not None and bridge_summary is not None,
                "phone split summaries",
            )
            validate_phone_work(
                plan, bridge_summary, ffn_summary, shape_summaries
            )
        run_trace.require(samples, "resource samples")
        resource_summary = {
            "gpu_memory_total_bytes": samples[0]["gpu"]["memory_total_bytes"],
            "gpu_memory_used_max_bytes": max(
                row["gpu"]["memory_used_bytes"] for row in samples
            ),
            "gpu_power_w": run_trace.stats([
                row["gpu"]["power_mw"] / 1000 for row in samples
            ]),
            "gpu_utilization_pct": run_trace.stats([
                row["gpu"]["utilization_pct"] for row in samples
            ]),
            "process_rss_max_bytes": max(
                row["process"]["rss_bytes"] for row in samples
            ),
            "process_swap_max_bytes": max(
                row["process"]["swap_bytes"] for row in samples
            ),
            "sample_count": len(samples),
            "system_available_min_bytes": min(
                row["system"]["available_bytes"] for row in samples
            ),
        }
        run_trace.require(
            resource_summary["process_swap_max_bytes"] == 0,
            "model process swapped during trace",
        )
        server_energy = None
        if args.require_server_energy:
            run_trace.require(
                all(row["rapl_package"] is not None for row in samples),
                "CPU package energy samples",
            )
            server_energy = run_trace.server_energy_summary(
                samples, paid_start_ns, paid_end_ns
            )
        result = {
            "arrival_mode": args.arrival_mode,
            "dispatch_order": args.dispatch_order,
            "kv_offload": args.kv_offload,
            "load_ms": load_ms,
            "mode": args.mode,
            "model": {
                "path": str(args.model),
                "profile": args.model_profile,
                "sha256": (
                    FP16_COLD_MODEL_SHA256
                    if args.model_profile == "fp16-proxy"
                    else run_trace.COLD_MODEL_SHA256
                ),
                "size_bytes": args.model.stat().st_size,
            },
            "n_gpu_layers": args.n_gpu_layers,
            "metrics": metrics(results, paid_start_ns),
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "phone": {
                "bridge": bridge_summary,
                "ffn": ffn_summary,
                "shapes": shape_summaries,
            },
            "preflight": {
                "cold_work_sha256": cold_work_sha256(requests),
                "scheduler": (
                    None
                    if plan is None
                    else {
                        "cpu_layer_spec": plan.layer_placement.cpu_layer_spec,
                        "decision_reason": plan.decision.reason,
                        "gpu_layer_spec": plan.layer_placement.gpu_layer_spec,
                        "plan_sha256": plan.plan_sha256,
                        "route_id": plan.decision.route_id,
                    }
                ),
                "source_trace_sha256": run_trace.SOURCE_REQUESTS_SHA256,
            },
            "repeat_index": args.repeat_index,
            "request_results": sorted(
                results, key=lambda row: row["request_index"]
            ),
            "resources": resource_summary,
            "schema": "s41-gemma-gpu-trace-v2",
            "scheduler_plan_sha256": None if plan is None else plan.plan_sha256,
            "server_energy": server_energy,
            "status": "PASS",
            "warm_ms": warm_ms,
        }
        run_trace.write_json(args.output / "RESULT.json", result)
        print(json.dumps({
            "load_ms": load_ms,
            "makespan_s": result["metrics"]["makespan_s"],
            "output_throughput_tokens_s": result["metrics"]
                ["output_throughput_tokens_s"],
            "status": "PASS",
            "warm_ms": warm_ms,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s41-gemma-gpu-trace-failure-v1",
            "status": "FAIL",
        })
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        if process is not None:
            process.terminate()
        if bridge is not None:
            bridge.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
