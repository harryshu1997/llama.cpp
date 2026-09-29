#!/usr/bin/env python3
"""Measure matched late versus early Gemma weight adoption."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.request


TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
QWEN_SHA256 = (
    "d89e9e823744222e595e0b3c8fd5436c"
    "e5d3a6a446fa42492ebce6064dfa9718"
)
GEMMA_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
STAGE_INDICES = (52, 53, 31)
TAIL_INDICES = (54, 55, 47)
GEMMA_INDEX = 50
STAGE_OFFSET = 15_838_752
STAGE_BYTES = 2_013_265_920
CHUNK_BYTES = 4_194_304
CHUNKS_PER_WINDOW = 9
GPU_RESERVE_BYTES = 536_870_912


class ScreenError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_requests(path: Path) -> dict[int, dict[str, Any]]:
    wanted = set(STAGE_INDICES + TAIL_INDICES + (GEMMA_INDEX,))
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="ascii") as stream:
        for line in stream:
            row = json.loads(line)
            index = row.get("request_index")
            if index in wanted:
                rows[index] = row
    require(set(rows) == wanted, "screen request identity")
    require(sha256(path) == TRACE_SHA256, "source trace identity")
    return rows


def process_command(pid: int) -> list[str]:
    fields = Path(f"/proc/{pid}/cmdline").read_bytes().rstrip(b"\0").split(
        b"\0"
    )
    require(bool(fields) and all(fields), "Qwen process command")
    return [field.decode("utf-8") for field in fields]


def wait_log(path: Path, prefix: str, process: subprocess.Popen[bytes]) -> None:
    for _ in range(1800):
        if path.exists() and any(
            line.startswith(prefix)
            for line in path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        ):
            return
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise ScreenError(f"missing log marker: {prefix}")


def wait_health(port: int, process: subprocess.Popen[bytes]) -> int:
    for _ in range(1800):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=1
            ) as response:
                if response.status == 200:
                    return time.monotonic_ns()
        except (OSError, urllib.error.URLError):
            pass
        if process.poll() is not None:
            break
        time.sleep(0.1)
    raise ScreenError("Gemma server did not publish READY")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        require(process.returncode == 0, "Gemma server exit status")
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    require(process.returncode == 0, "Gemma server shutdown")


class DynamicSampler:
    def __init__(self, output: Path, qwen_pid: int) -> None:
        self.output = output
        self.pids = {"qwen": qwen_pid, "gemma": 0}
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_gemma_pid(self, pid: int) -> None:
        with self.lock:
            self.pids["gemma"] = pid

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), "resource sampler stop")
        require(self.error is None, f"resource sampler: {self.error}")

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    pids = dict(self.pids)
                before_ns = time.monotonic_ns()
                gpu = run_trace.gpu_snapshot()
                after_ns = time.monotonic_ns()
                gpu["sample_t_ns"] = (before_ns + after_ns) // 2
                try:
                    rapl = run_trace.rapl_package_snapshot()
                except (FileNotFoundError, PermissionError,
                        run_trace.RunError, ValueError):
                    rapl = None
                self.rows.append({
                    "gpu": gpu,
                    "pids": {
                        name: run_trace.proc_status_or_zero(pid)
                        for name, pid in pids.items()
                    },
                    "rapl_package": rapl,
                    "schema": "s42-adoption-screen-resource-v1",
                    "system": run_trace.system_memory(),
                    "t_ns": time.monotonic_ns(),
                })
                self.stop_event.wait(0.2)
            with (self.output / "resource-samples.jsonl").open("xb") as stream:
                for row in self.rows:
                    stream.write(canonical(row))
        except BaseException as exc:
            self.error = f"{type(exc).__name__}: {exc}"


def execute_request(
    port: int,
    row: dict[str, Any],
    raw_path: Path,
    route: str,
) -> dict[str, Any]:
    first_token: list[int] = []
    started_ns = time.monotonic_ns()
    value = run_server_trace.server_completion(
        port, row, raw_path, first_token.append
    )
    completed_ns = time.monotonic_ns()
    require(len(first_token) == 1, "request first-token receipt")
    tokens = value["tokens"]
    require(len(tokens) == row["output_tokens"], "request output length")
    return {
        "completed_ns": completed_ns,
        "first_token_ns": first_token[0],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "predicted_ms": value["predicted_ms"],
        "prompt_ms": value["prompt_ms"],
        "request_index": row["request_index"],
        "route": route,
        "started_ns": started_ns,
        "tokens": tokens,
        "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
    }


def parse_result(path: Path) -> dict[str, Any]:
    rows = []
    for line in path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.startswith("S42_FENCED_TENSOR_RESULT "):
            continue
        fields = dict(
            field.split("=", 1)
            for field in line.split()[1:]
            if "=" in field
        )
        rows.append(fields)
    require(len(rows) == 1, "one fenced tensor result")
    row = rows[0]
    require(
        row.get("status") == "PASS"
        and row.get("tensor") == "token_embd.weight"
        and int(row.get("source_offset", -1)) == STAGE_OFFSET
        and int(row.get("bytes", -1)) == STAGE_BYTES
        and int(row.get("chunk_bytes", -1)) == CHUNK_BYTES
        and int(row.get("chunks_per_window", -1)) == CHUNKS_PER_WINDOW
        and int(row.get("copied_chunks", -1)) == 480
        and row.get("verified") == "true"
        and row.get("adoptable") == "true"
        and int(row.get("gpu_free_min_bytes", -1)) >= GPU_RESERVE_BYTES,
        "fenced tensor qualification",
    )
    return {
        "armed_calls": int(row["armed_calls"]),
        "bytes": int(row["bytes"]),
        "copied_chunks": int(row["copied_chunks"]),
        "copy_windows": int(row["copy_windows"]),
        "fence_calls": int(row["fence_calls"]),
        "gpu_free_min_bytes": int(row["gpu_free_min_bytes"]),
        "gpu_reserve_bytes": int(row["gpu_reserve_bytes"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_requests(args.requests)
    require(sha256(args.qwen_model) == QWEN_SHA256, "Qwen model identity")
    require(sha256(args.gemma_model) == GEMMA_SHA256, "Gemma model identity")
    qwen_command = process_command(args.qwen_pid)
    require(
        "--n-gpu-layers" in qwen_command
        and qwen_command[qwen_command.index("--n-gpu-layers") + 1] == "15",
        "Qwen-15 placement",
    )

    gemma_stdout_path = args.output / "gemma.stdout"
    gemma_stderr_path = args.output / "gemma.stderr"
    gemma_stdout = gemma_stdout_path.open("xb")
    gemma_stderr = gemma_stderr_path.open("xb")
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cuda_lib_dir}:{args.gemma_server.parent}"
    )
    environment.update({
        "S42_FENCED_TENSOR_SOCKET": str(args.fence_socket),
        "S42_FENCED_TENSOR_ARM_FILE": str(args.arm_file),
        "S42_FENCED_TENSOR_NAME": "token_embd.weight",
        "S42_FENCED_TENSOR_EXPECTED_OFFSET": str(STAGE_OFFSET),
        "S42_FENCED_TENSOR_EXPECTED_BYTES": str(STAGE_BYTES),
        "S42_FENCED_TENSOR_CHUNK_BYTES": str(CHUNK_BYTES),
        "S42_FENCED_TENSOR_CHUNKS_PER_WINDOW": str(CHUNKS_PER_WINDOW),
        "S42_FENCED_TENSOR_GPU_RESERVE_BYTES": str(GPU_RESERVE_BYTES),
    })
    command = [
        str(args.gemma_server),
        "--model", str(args.gemma_model),
        "--alias", "gemma-adoption-energy-screen",
        "--fit", "off",
        "--ctx-size", "32768",
        "--parallel", "8",
        "--batch-size", "4096",
        "--ubatch-size", "512",
        "--flash-attn", "on",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--split-mode", "none",
        "--n-gpu-layers", "1",
        "--main-gpu", "0",
        "--device", "CUDA0",
        "--host", "127.0.0.1",
        "--port", str(args.gemma_port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-timestamps",
        "--log-verbosity", "1",
    ]

    sampler = DynamicSampler(args.output, args.qwen_pid)
    sampler.start()
    time.sleep(0.6)
    paid_start_ns = time.monotonic_ns()
    gemma = subprocess.Popen(
        command,
        env=environment,
        stdout=gemma_stdout,
        stderr=gemma_stderr,
    )
    sampler.set_gemma_pid(gemma.pid)
    request_results: list[dict[str, Any]] = []
    gemma_ready_ns = None
    try:
        wait_log(gemma_stderr_path, "S42_FENCED_TENSOR_READY ", gemma)
        source_ready_ns = time.monotonic_ns()

        warm = dict(rows[52])
        warm["output_tokens"] = 2
        execute_request(
            args.qwen_port,
            warm,
            args.output / "warm-qwen.raw",
            "qwen15-cpu-op15-warm",
        )

        def execute_gemma() -> dict[str, Any]:
            nonlocal gemma_ready_ns
            gemma_ready_ns = wait_health(args.gemma_port, gemma)
            return execute_request(
                args.gemma_port,
                rows[GEMMA_INDEX],
                args.output / "gemma-request-050.raw",
                "gemma1-cpu",
            )

        gemma_future: concurrent.futures.Future[dict[str, Any]] | None = None
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gemma-adopted"
        ) as pool:
            if args.arm == "dynamic":
                args.arm_file.write_text(
                    f"{time.monotonic_ns()}\n", encoding="ascii"
                )
                gemma_future = pool.submit(execute_gemma)

            for index in STAGE_INDICES:
                request_results.append(execute_request(
                    args.qwen_port,
                    rows[index],
                    args.output / f"qwen-{index:03d}.raw",
                    "qwen15-cpu-op15",
                ))

            if args.arm == "control":
                args.arm_file.write_text(
                    f"{time.monotonic_ns()}\n", encoding="ascii"
                )

            for index in TAIL_INDICES:
                request_results.append(execute_request(
                    args.qwen_port,
                    rows[index],
                    args.output / f"qwen-{index:03d}.raw",
                    "qwen15-cpu-op15",
                ))

            if args.arm == "control":
                gemma_future = pool.submit(execute_gemma)
            require(gemma_future is not None, "Gemma future")
            request_results.append(gemma_future.result(timeout=900))

        paid_end_ns = max(
            row["completed_ns"] for row in request_results
        )
        time.sleep(0.6)
        sampler.stop()
        samples = list(sampler.rows)
        server_energy = run_trace.server_energy_summary(
            samples, paid_start_ns, paid_end_ns
        )
        transition = parse_result(gemma_stderr_path)
        process_swap_max = max(
            process["swap_bytes"]
            for sample in samples
            for process in sample["pids"].values()
        )
        require(process_swap_max == 0, "process swap")
        result: dict[str, Any] = {
            "arm": args.arm,
            "artifacts": {
                "gemma_model_sha256": GEMMA_SHA256,
                "gemma_server_sha256": sha256(args.gemma_server),
                "qwen_model_sha256": QWEN_SHA256,
                "source_trace_sha256": TRACE_SHA256,
            },
            "gemma_ready_ns": gemma_ready_ns,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "placement": {
                "gemma_gpu_layers": 1,
                "qwen_gpu_layers": 15,
            },
            "process_swap_max_bytes": process_swap_max,
            "qwen_command": qwen_command,
            "repeat_index": args.repeat_index,
            "request_results": sorted(
                request_results, key=lambda row: row["request_index"]
            ),
            "resources": {
                "gpu_memory_free_min_bytes": min(
                    sample["gpu"]["memory_free_bytes"] for sample in samples
                ),
                "gpu_utilization_pct": run_trace.stats([
                    sample["gpu"]["utilization_pct"] for sample in samples
                ]),
                "samples": len(samples),
                "system_available_min_bytes": min(
                    sample["system"]["available_bytes"] for sample in samples
                ),
            },
            "schema": "s42-adoption-energy-screen-run-v1",
            "server_energy": server_energy,
            "source_ready_ns": source_ready_ns,
            "status": "PASS",
            "transition": transition,
            "warmup_inside_boundary": True,
            "workload": {
                "gemma_input_tokens": rows[GEMMA_INDEX]["input_tokens"],
                "gemma_output_tokens": rows[GEMMA_INDEX]["output_tokens"],
                "gemma_request_index": GEMMA_INDEX,
                "qwen_indices": list(STAGE_INDICES + TAIL_INDICES),
                "qwen_input_tokens": sum(
                    rows[index]["input_tokens"]
                    for index in STAGE_INDICES + TAIL_INDICES
                ),
                "qwen_output_tokens": sum(
                    rows[index]["output_tokens"]
                    for index in STAGE_INDICES + TAIL_INDICES
                ),
                "requests": 7,
            },
        }
        result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
        return result
    finally:
        if sampler.thread.is_alive():
            sampler.stop()
        stop_process(gemma)
        gemma_stdout.close()
        gemma_stderr.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("control", "dynamic"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--burst-dir", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-port", type=int, required=True)
    parser.add_argument("--qwen-pid", type=int, required=True)
    parser.add_argument("--gemma-server", type=Path, required=True)
    parser.add_argument("--gemma-model", type=Path, required=True)
    parser.add_argument("--gemma-port", type=int, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--fence-socket", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.repeat_index <= 0
        or not args.output.is_absolute()
        or args.output.exists()
        or args.qwen_pid <= 0
        or not 0 < args.qwen_port <= 65535
        or not 0 < args.gemma_port <= 65535
        or not args.fence_socket.is_absolute()
        or not args.arm_file.is_absolute()
        or args.arm_file.exists()
    ):
        parser.error("invalid screen arguments")
    for path in (
        args.burst_dir,
        args.requests,
        args.qwen_model,
        args.gemma_server,
        args.gemma_model,
        args.cuda_lib_dir,
    ):
        if not path.exists():
            parser.error(f"missing dependency: {path}")
    args.output.mkdir()
    sys.path.insert(0, str(args.burst_dir))
    global run_server_trace, run_trace
    import run_server_trace  # type: ignore[no-redef]  # noqa: E402
    import run_trace  # type: ignore[no-redef]  # noqa: E402
    try:
        value = run(args)
        (args.output / "RESULT.json").write_bytes(canonical(value))
        boundary = {
            "paid_end_ns": value["paid_end_ns"],
            "paid_start_ns": value["paid_start_ns"],
            "schema": "s41-burstgpt-llama-server-result-v1",
            "status": "PASS",
        }
        (args.output / "PHONE_BOUNDARY.json").write_bytes(canonical(boundary))
    except (OSError, ScreenError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"adoption energy screen failed: {exc}\n")
    print(json.dumps({
        "arm": value["arm"],
        "duration_s": (
            value["paid_end_ns"] - value["paid_start_ns"]
        ) / 1e9,
        "server_energy_j": value["server_energy"][
            "server_compute_device_energy_j"
        ],
        "status": value["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
