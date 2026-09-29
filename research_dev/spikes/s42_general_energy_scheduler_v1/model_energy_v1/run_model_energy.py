#!/usr/bin/env python3
"""Measure resident single-model cohort energy on the 4060 Ti desktop."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any


HERE = Path(__file__).resolve().parent
S41_SUPPORT = (
    HERE.parents[1]
    / "s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1"
    / "burstgpt_gpu_cpu_op15_v1"
)
if S41_SUPPORT.is_dir() and str(S41_SUPPORT) not in sys.path:
    sys.path.insert(0, str(S41_SUPPORT))

import run_server_trace  # noqa: E402
import run_trace  # noqa: E402


CONFIRMATION = "RUN_S42_MODEL_ENERGY_V1"
CASES_SHA256 = "f2a48c02dff82da08e6382cc2d6758e234cf7f268c586d0f6f30681969ee93c0"
CATALOG_SHA256 = "9d7464b6cac747abe85f4701e312000fec0b65aa644bcbcb41aba789f5cf58bc"
GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
I3_POLICY = "1:9664,3:8192,8:4096,128:8192,512:11136"
ROUTES = {
    "gemma-cpu": {
        "file_bytes": 6_975_878_176,
        "model_id": "gemma4-12b-q4_0",
        "model_sha256": run_trace.COLD_MODEL_SHA256,
        "placement": "CPU",
    },
    "gemma-op15": {
        "file_bytes": 6_975_878_176,
        "model_id": "gemma4-12b-q4_0",
        "model_sha256": run_trace.COLD_MODEL_SHA256,
        "placement": "CPU+OP15-HTP",
    },
    "qwen-cuda": {
        "file_bytes": 9_001_752_960,
        "model_id": "qwen3-14b-q4_k_m",
        "model_sha256": run_trace.HOT_MODEL_SHA256,
        "placement": "CUDA0",
    },
}


def strict_int(value: object, label: str, minimum: int = 0) -> int:
    run_trace.require(type(value) is int and value >= minimum, label)
    return int(value)


def strict_number(value: object, label: str, minimum: float = 0.0) -> float:
    run_trace.require(
        isinstance(value, (int, float)) and
        not isinstance(value, bool) and
        math.isfinite(float(value)) and
        float(value) >= minimum,
        label,
    )
    return float(value)


def load_cases(path: Path) -> list[dict[str, Any]]:
    run_trace.require(run_trace.digest_file(path) == CASES_SHA256, "case-plan hash")
    rows = run_trace.read_jsonl(path)
    seen: set[str] = set()
    idle_count = 0
    for index, row in enumerate(rows):
        case_id = row.get("case_id")
        kind = row.get("kind")
        run_trace.require(
            row.get("schema") == "s42-model-energy-case-v1" and
            type(case_id) is str and case_id and case_id not in seen and
            type(row.get("holdout")) is bool and
            kind in {"idle", "inference"},
            f"cases[{index}]: identity",
        )
        seen.add(case_id)
        if kind == "idle":
            idle_count += 1
            run_trace.require(
                set(row) == {
                    "case_id", "duration_s", "holdout", "kind", "schema"
                } and
                strict_number(row.get("duration_s"), "idle duration", 5.0) >= 5.0,
                f"cases[{index}]: idle",
            )
        else:
            run_trace.require(
                set(row) == {
                    "case_id", "cohort_size", "holdout", "input_tokens",
                    "kind", "max_cohorts", "min_cohorts", "min_duration_s",
                    "output_tokens", "schema",
                },
                f"cases[{index}]: inference fields",
            )
            input_tokens = strict_int(row.get("input_tokens"), "input tokens", 1)
            output_tokens = strict_int(row.get("output_tokens"), "output tokens", 1)
            cohort_size = strict_int(row.get("cohort_size"), "cohort size", 1)
            minimum = strict_int(row.get("min_cohorts"), "minimum cohorts", 1)
            maximum = strict_int(row.get("max_cohorts"), "maximum cohorts", 1)
            run_trace.require(
                cohort_size <= 8 and input_tokens + output_tokens <= 4096 and
                minimum <= maximum and
                strict_number(row.get("min_duration_s"), "minimum duration", 2.0)
                >= 2.0,
                f"cases[{index}]: inference bounds",
            )
    run_trace.require(len(rows) >= 5 and idle_count == 1, "case-plan coverage")
    return rows


def parse_cuda_pids(text: str) -> set[int]:
    result: set[int] = set()
    for line in text.splitlines():
        value = line.strip()
        if not value or value == "No running processes found":
            continue
        try:
            pid = int(value)
        except ValueError as error:
            raise run_trace.RunError("CUDA process list") from error
        run_trace.require(pid > 0, "CUDA process PID")
        result.add(pid)
    return result


def foreign_inference_processes() -> list[dict[str, Any]]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw_command = entry.joinpath("cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        argv0 = raw_command.split(b"\0", 1)[0]
        executable = argv0.rsplit(b"/", 1)[-1]
        if executable in {
            b"llama-server",
            b"ffn_dmabuf_bridge",
            b"llama-ffn-split-worker",
        }:
            command = raw_command.replace(b"\0", b" ")
            result.append({
                "command_sha256": hashlib.sha256(command).hexdigest(),
                "pid": int(entry.name),
            })
    return sorted(result, key=lambda row: row["pid"])


def require_quiet_host() -> None:
    foreign = foreign_inference_processes()
    run_trace.require(not foreign, f"foreign inference processes: {foreign}")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    run_trace.require(not parse_cuda_pids(completed.stdout), "foreign CUDA process")


def validate_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output path",
    )
    for path in (args.cases, args.catalog, args.server, args.model, args.lib_dir):
        run_trace.require(path.exists(), f"missing path: {path}")
    run_trace.require(
        run_trace.digest_file(args.catalog) == CATALOG_SHA256,
        "model catalog hash",
    )
    if args.route == "gemma-op15":
        run_trace.require(args.bridge is not None and args.bridge.exists(), "bridge")
    else:
        run_trace.require(args.bridge is None, "unexpected bridge")
    route = ROUTES[args.route]
    run_trace.require(
        args.model.stat().st_size == route["file_bytes"] and
        run_trace.digest_file(args.model) == route["model_sha256"],
        "model identity",
    )
    run_trace.rapl_package_snapshot()
    gpu = run_trace.gpu_snapshot()
    run_trace.require(gpu["uuid"] == GPU_UUID, "GPU identity")
    require_quiet_host()
    run_trace.require(args.repeat_index >= 1, "repeat index")
    return load_cases(args.cases)


def configure_runtime(args: argparse.Namespace) -> None:
    args.hot_server = args.server
    args.hot_model = args.model
    args.cuda_lib_dir = args.lib_dir
    args.hot_ctx_size = 24576
    args.hot_parallel = 8
    args.hot_port = args.port
    args.hot_nice = 0
    args.hot_cpus = None

    args.cold_server = args.server
    args.cold_model = args.model
    args.cold_lib_dir = args.lib_dir
    args.cold_ctx_size = 32768
    args.cold_parallel = 8
    args.cold_batch_size = 4096
    args.cold_ubatch_size = 512
    args.cold_threads = -1
    args.cold_repack = "default"
    args.cold_port = args.port
    args.control_cpus = None
    args.cold_cpus = None
    args.bridge_cpus = None
    args.bridge_port = args.bridge_port
    args.max_columns = 11136
    args.split_io = "f16"
    args.split_policy = I3_POLICY
    args.mode = "op15" if args.route == "gemma-op15" else "cpu"


def start_runtime(
    args: argparse.Namespace,
) -> tuple[run_trace.CapturedProcess, run_trace.CapturedProcess | None]:
    bridge = None
    if args.route == "qwen-cuda":
        server = run_trace.start_hot(args, args.output)
    else:
        if args.route == "gemma-op15":
            bridge = run_trace.start_bridge(args, args.output)
        server = run_server_trace.start_cold_server(args, args.output)
        log = "\n".join(server.stderr_lines)
        placements = re.findall(r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log)
        run_trace.require(
            not placements or all(int(loaded) == 0 for loaded, _ in placements),
            "Gemma placement is not CPU",
        )
    return server, bridge


def request_row(request_index: int, input_tokens: int,
                output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_tokens": [1] * input_tokens,
        "request_index": request_index,
    }


def one_request(
    *,
    case_dir: Path,
    cohort_index: int,
    member_index: int,
    request_index: int,
    input_tokens: int,
    output_tokens: int,
    port: int,
    gate: threading.Event,
) -> dict[str, Any]:
    gate.wait()
    dispatch_ns = time.monotonic_ns()
    first: list[int] = []
    value = run_server_trace.server_completion(
        port,
        request_row(request_index, input_tokens, output_tokens),
        case_dir / f"stream-c{cohort_index:04d}-m{member_index:02d}.raw",
        first.append,
    )
    completion_ns = time.monotonic_ns()
    run_trace.require(len(first) == 1, "first-token accounting")
    prompt_ms = strict_number(value.get("prompt_ms"), "prompt timing")
    predicted_ms = strict_number(value.get("predicted_ms"), "decode timing")
    return {
        "completion_ns": completion_ns,
        "dispatch_ns": dispatch_ns,
        "first_token_ns": first[0],
        "predicted_ms": predicted_ms,
        "prompt_ms": prompt_ms,
        "request_index": request_index,
        "token_count": len(value["tokens"]),
        "token_sha256": hashlib.sha256(
            run_trace.canonical(value["tokens"])
        ).hexdigest(),
    }


def resource_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_trace.require(len(rows) >= 2, "resource sample count")
    return {
        "gpu_power_w": run_trace.stats([
            row["gpu"]["power_mw"] / 1000.0 for row in rows
        ]),
        "gpu_utilization_pct": run_trace.stats([
            row["gpu"]["utilization_pct"] for row in rows
        ]),
        "model_rss_max_bytes": max(row["hot"]["rss_bytes"] for row in rows),
        "model_swap_max_bytes": max(row["hot"]["swap_bytes"] for row in rows),
        "sample_count": len(rows),
        "system_available_min_bytes": min(
            row["system"]["available_bytes"] for row in rows
        ),
    }


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    server: run_trace.CapturedProcess,
    request_index: int,
) -> tuple[dict[str, Any], int]:
    case_dir = args.output / case["case_id"]
    case_dir.mkdir()
    sampler = run_trace.ResourceSampler(case_dir, server.pid, server.pid)
    sampler.start()
    time.sleep(0.7)
    paid_start_ns = time.monotonic_ns()
    requests: list[dict[str, Any]] = []
    cohorts = 0
    try:
        if case["kind"] == "idle":
            time.sleep(float(case["duration_s"]))
        else:
            cohort_size = int(case["cohort_size"])
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=cohort_size,
                thread_name_prefix="model-energy",
            ) as pool:
                while True:
                    gate = threading.Event()
                    futures = []
                    for member_index in range(cohort_size):
                        request_index += 1
                        futures.append(pool.submit(
                            one_request,
                            case_dir=case_dir,
                            cohort_index=cohorts,
                            member_index=member_index,
                            request_index=request_index,
                            input_tokens=case["input_tokens"],
                            output_tokens=case["output_tokens"],
                            port=args.port,
                            gate=gate,
                        ))
                    gate.set()
                    requests.extend(future.result(timeout=3600) for future in futures)
                    cohorts += 1
                    elapsed_s = (time.monotonic_ns() - paid_start_ns) / 1e9
                    if (
                        cohorts >= case["min_cohorts"] and
                        elapsed_s >= case["min_duration_s"]
                    ):
                        break
                    if cohorts >= case["max_cohorts"]:
                        break
        paid_end_ns = time.monotonic_ns()
        time.sleep(0.7)
    finally:
        sampler.stop()
    rows = list(sampler.rows)
    energy = run_trace.server_energy_summary(rows, paid_start_ns, paid_end_ns)
    resources = resource_summary(rows)
    run_trace.require(resources["model_swap_max_bytes"] == 0, "model swap")
    duration_s = (paid_end_ns - paid_start_ns) / 1e9
    if case["kind"] == "idle":
        features = {
            "cohorts": 0,
            "decode_steps": 0,
            "input_token_rows": 0,
            "output_token_rows": 0,
            "prefill_ubatches": 0,
            "requests": 0,
        }
        latency = None
        target_met = True
    else:
        expected_requests = cohorts * case["cohort_size"]
        run_trace.require(
            len(requests) == expected_requests and
            all(row["token_count"] == case["output_tokens"] for row in requests),
            "case request conservation",
        )
        features = {
            "cohorts": cohorts,
            "decode_steps": cohorts * case["output_tokens"],
            "input_token_rows": expected_requests * case["input_tokens"],
            "output_token_rows": expected_requests * case["output_tokens"],
            "prefill_ubatches": cohorts * math.ceil(
                case["cohort_size"] * case["input_tokens"] / 512
            ),
            "requests": expected_requests,
        }
        latency = {
            "request_wall_ms": run_trace.stats([
                (row["completion_ns"] - row["dispatch_ns"]) / 1e6
                for row in requests
            ]),
            "server_decode_ms": run_trace.stats([
                row["predicted_ms"] for row in requests
            ]),
            "server_prefill_ms": run_trace.stats([
                row["prompt_ms"] for row in requests
            ]),
            "ttft_ms": run_trace.stats([
                (row["first_token_ns"] - row["dispatch_ns"]) / 1e6
                for row in requests
            ]),
        }
        target_met = (
            cohorts >= case["min_cohorts"] and duration_s >= case["min_duration_s"]
        )
    return {
        "case": case,
        "duration_s": duration_s,
        "features": features,
        "latency": latency,
        "paid_end_ns": paid_end_ns,
        "paid_start_ns": paid_start_ns,
        "request_evidence": requests,
        "resources": resources,
        "server_energy": energy,
        "target_duration_met": target_met,
    }, request_index


def runtime_preflight(
    args: argparse.Namespace,
    server: run_trace.CapturedProcess,
) -> dict[str, Any]:
    route = ROUTES[args.route]
    return {
        "case_plan_sha256": CASES_SHA256,
        "model_catalog_sha256": CATALOG_SHA256,
        "cpu_affinity": None,
        "gpu": run_trace.gpu_snapshot(),
        "model_file_bytes": route["file_bytes"],
        "model_id": route["model_id"],
        "model_sha256": route["model_sha256"],
        "placement": route["placement"],
        "route": args.route,
        "runtime_manifest": run_trace.process_runtime_manifest(
            server.pid, [args.lib_dir, args.server.parent]
        ),
        "schema": "s42-model-energy-preflight-v1",
        "server_sha256": run_trace.digest_file(args.server),
        "system_memory": run_trace.system_memory(),
    }


def phone_summary(
    args: argparse.Namespace,
    server: run_trace.CapturedProcess,
    bridge: run_trace.CapturedProcess | None,
) -> dict[str, Any] | None:
    if args.route != "gemma-op15":
        return None
    ffn = run_server_trace.prefixed_json(
        server.stderr_lines, "S41SERVERFFN ", True
    )
    shapes = [
        json.loads(line[len("S41SERVERFFNSHAPE "):])
        for line in server.stderr_lines
        if line.startswith("S41SERVERFFNSHAPE ")
    ]
    run_trace.require(bridge is not None, "missing bridge process")
    bridge_value = run_server_trace.prefixed_json(
        bridge.stderr_lines, "FFNDMABUF ", True
    )
    run_trace.require(
        ffn is not None and ffn.get("status") == "ok" and
        bridge_value is not None and bridge_value.get("status") == "ok" and
        bridge_value.get("reset_recoveries") == 0 and
        ffn.get("calls") == bridge_value.get("calls") and
        ffn.get("upload_bytes") == bridge_value.get("upload_bytes") and
        ffn.get("download_bytes") == bridge_value.get("download_bytes") and
        shapes,
        "phone work summary",
    )
    return {"bridge": bridge_value, "ffn": ffn, "shapes": shapes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--port", type=int, default=19100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    cases = validate_args(args)
    configure_runtime(args)
    args.output.mkdir(parents=True)
    server = None
    bridge = None
    result = None
    failure = None
    try:
        server, bridge = start_runtime(args)
        warm_first: list[int] = []
        run_server_trace.server_completion(
            args.port,
            request_row(0, 32, 2),
            args.output / "warm.raw",
            warm_first.append,
        )
        run_trace.require(
            len(warm_first) == 1 and
            run_trace.proc_status(server.pid)["swap_bytes"] == 0,
            "warmup and swap",
        )
        preflight = runtime_preflight(args, server)
        run_trace.write_json(args.output / "preflight.json", preflight)
        case_results = []
        request_index = 0
        for case in cases:
            case_result, request_index = run_case(
                args, case, server, request_index
            )
            case_results.append(case_result)
        server.terminate()
        phone = None
        if bridge is not None:
            bridge.terminate()
        phone = phone_summary(args, server, bridge)
        run_trace.require(
            all(case["target_duration_met"] for case in case_results),
            "case duration target",
        )
        result = {
            "cases": case_results,
            "phone": phone,
            "preflight": preflight,
            "repeat_index": args.repeat_index,
            "route": args.route,
            "schema": "s42-model-energy-result-v1",
            "status": "PASS",
        }
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if server is not None:
            server.terminate()
        if bridge is not None:
            bridge.terminate()

    if failure is not None:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": failure,
            "repeat_index": args.repeat_index,
            "route": args.route,
            "schema": "s42-model-energy-failure-v1",
            "status": "FAIL",
        })
        return 2
    run_trace.require(result is not None, "missing result")
    run_trace.write_json(args.output / "RESULT.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
