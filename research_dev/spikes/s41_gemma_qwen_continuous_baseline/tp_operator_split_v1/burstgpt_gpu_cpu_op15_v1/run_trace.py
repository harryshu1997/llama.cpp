#!/usr/bin/env python3
"""Run the bounded semantic BurstGPT trace with GPU-hot and CPU-cold routes."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler import (  # noqa: E402
    OperatorSplitError,
    ShapeBucket,
    parse_split_table,
    select_split_columns,
)


SOURCE_HOT = "gemma-4-12b-it-q8_0"
SOURCE_COLD = "qwen3-14b-q4_k_m"
HOT_MODEL = "qwen3-14b-q4_k_m"
COLD_MODEL = "gemma-4-12b-it-q4_0"
LONG_REQUESTS_SHA256 = (
    "ccde6e3e53dee4547e4eb80f9f090032"
    "afb04b1d3fec3fd07f0f961bed60cf8f"
)
SOURCE_REQUESTS_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
HOT_MODEL_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
COLD_MODEL_SHA256 = (
    "494518c2262a26e2a607af0e40bca11c"
    "4de5a0b108e7c21308906dbbfb1c6f8c"
)


class RunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def canonical(value: Any) -> bytes:
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


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True)):
        require(raw.endswith(b"\n"), f"requests[{index}]: framing")
        row = json.loads(raw)
        require(type(row) is dict, f"requests[{index}]: object")
        require(canonical(row) == raw, f"requests[{index}]: canonical")
        rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical(value))


def percentile(values: list[float], fraction: float) -> float:
    require(values, "empty percentile")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def parse_prefill_policy(text: str) -> list[tuple[int, int]]:
    try:
        return [
            (bucket.max_tokens, bucket.phone_columns)
            for bucket in parse_split_table(text)
        ]
    except OperatorSplitError as exc:
        raise RunError(str(exc)) from exc


def policy_columns(policy: list[tuple[int, int]], tokens: int) -> int:
    try:
        return select_split_columns(
            tuple(ShapeBucket(limit, columns) for limit, columns in policy),
            tokens,
        )
    except OperatorSplitError as exc:
        raise RunError(str(exc)) from exc


def proc_status(pid: int) -> dict[str, int]:
    wanted = {"VmRSS": "rss_bytes", "VmSwap": "swap_bytes"}
    result: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            fields = value.strip().split()
            require(len(fields) == 2 and fields[1] == "kB", "proc status")
            result[wanted[key]] = int(fields[0]) * 1024
    require(set(result) == set(wanted.values()), "proc status fields")
    return result


def proc_status_or_zero(pid: int) -> dict[str, int]:
    try:
        return proc_status(pid)
    except (FileNotFoundError, RunError):
        return {"rss_bytes": 0, "swap_bytes": 0}


def process_runtime_manifest(pid: int, roots: list[Path]) -> list[dict[str, Any]]:
    resolved_roots = [root.resolve(strict=True) for root in roots]
    paths = {Path(f"/proc/{pid}/exe").resolve(strict=True)}
    for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path_text = fields[5]
        deleted = path_text.endswith(" (deleted)")
        mapped_path = Path(
            path_text.removesuffix(" (deleted)") if deleted else path_text
        )
        candidate = mapped_path.resolve(strict=False)
        relevant = any(
            candidate == root or candidate.is_relative_to(root)
            for root in resolved_roots
        )
        if not relevant:
            continue
        require(not deleted, f"deleted runtime mapping: {mapped_path}")
        paths.add(mapped_path.resolve(strict=True))
    result = []
    for path in sorted(paths):
        require(path.is_file(), "runtime mapping is not a file")
        result.append({
            "path": str(path),
            "sha256": digest_file(path),
            "size_bytes": path.stat().st_size,
        })
    require(result, "empty runtime manifest")
    return result


def system_memory() -> dict[str, int]:
    wanted = {
        "MemAvailable": "available_bytes",
        "SwapFree": "swap_free_bytes",
        "SwapTotal": "swap_total_bytes",
    }
    result: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            fields = value.strip().split()
            require(len(fields) == 2 and fields[1] == "kB", "meminfo")
            result[wanted[key]] = int(fields[0]) * 1024
    require(set(result) == set(wanted.values()), "meminfo fields")
    return result


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.used,memory.free,"
            "utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    fields = [field.strip() for field in result.stdout.strip().split(",")]
    require(len(fields) == 7, "GPU snapshot")
    return {
        "memory_free_bytes": int(fields[4]) * 1024 * 1024,
        "memory_total_bytes": int(fields[2]) * 1024 * 1024,
        "memory_used_bytes": int(fields[3]) * 1024 * 1024,
        "name": fields[0],
        "power_mw": int(round(float(fields[6]) * 1000)),
        "utilization_pct": int(fields[5]),
        "uuid": fields[1],
    }


def rapl_package_snapshot() -> dict[str, Any]:
    root = Path("/sys/class/powercap/intel-rapl:0")
    require(root.joinpath("name").read_text().strip() == "package-0", "RAPL package")
    before_ns = time.monotonic_ns()
    energy_uj = int(root.joinpath("energy_uj").read_text().strip())
    after_ns = time.monotonic_ns()
    max_energy_range_uj = int(
        root.joinpath("max_energy_range_uj").read_text().strip()
    )
    require(
        0 <= energy_uj < max_energy_range_uj and max_energy_range_uj > 0,
        "RAPL package counter",
    )
    return {
        "energy_uj": energy_uj,
        "max_energy_range_uj": max_energy_range_uj,
        "name": "package-0",
        "sample_t_ns": (before_ns + after_ns) // 2,
    }


def _interpolate(points: list[tuple[int, float]], target_ns: int) -> float:
    require(len(points) >= 2, "energy sample count")
    require(points[0][0] <= target_ns <= points[-1][0], "energy sample boundary")
    for index in range(1, len(points)):
        left_t, left_value = points[index - 1]
        right_t, right_value = points[index]
        require(right_t > left_t, "energy sample order")
        if target_ns <= right_t:
            fraction = (target_ns - left_t) / (right_t - left_t)
            return left_value + fraction * (right_value - left_value)
    raise RunError("energy interpolation coverage")


def integrate_power_samples(
    rows: list[dict[str, Any]], start_ns: int, end_ns: int
) -> float:
    require(end_ns > start_ns, "GPU energy interval")
    points = sorted(
        (
            row["gpu"]["sample_t_ns"],
            row["gpu"]["power_mw"] / 1000.0,
        )
        for row in rows
    )
    start_power = _interpolate(points, start_ns)
    end_power = _interpolate(points, end_ns)
    bounded = [(start_ns, start_power)]
    bounded.extend(point for point in points if start_ns < point[0] < end_ns)
    bounded.append((end_ns, end_power))
    energy_j = 0.0
    for left, right in zip(bounded, bounded[1:]):
        duration_s = (right[0] - left[0]) / 1e9
        energy_j += duration_s * (left[1] + right[1]) / 2.0
    return energy_j


def integrate_rapl_samples(
    rows: list[dict[str, Any]], start_ns: int, end_ns: int
) -> float:
    require(end_ns > start_ns, "RAPL energy interval")
    samples = sorted(
        (
            row["rapl_package"]["sample_t_ns"],
            row["rapl_package"]["energy_uj"],
            row["rapl_package"]["max_energy_range_uj"],
        )
        for row in rows
        if row.get("rapl_package") is not None
    )
    require(len(samples) == len(rows), "missing RAPL sample")
    require(len(samples) >= 2, "RAPL sample count")
    max_range = samples[0][2]
    require(
        all(sample[2] == max_range for sample in samples),
        "RAPL range changed",
    )
    unwrapped: list[tuple[int, float]] = [(samples[0][0], 0.0)]
    prior = samples[0][1]
    total = 0
    for sample_t_ns, energy_uj, _ in samples[1:]:
        delta = energy_uj - prior
        if delta < 0:
            delta += max_range
        require(0 <= delta < max_range, "RAPL counter delta")
        total += delta
        unwrapped.append((sample_t_ns, float(total)))
        prior = energy_uj
    start_uj = _interpolate(unwrapped, start_ns)
    end_uj = _interpolate(unwrapped, end_ns)
    require(end_uj >= start_uj, "RAPL energy direction")
    return (end_uj - start_uj) / 1e6


def server_energy_summary(
    rows: list[dict[str, Any]], start_ns: int, end_ns: int
) -> dict[str, Any]:
    gpu_energy_j = integrate_power_samples(rows, start_ns, end_ns)
    cpu_energy_j = integrate_rapl_samples(rows, start_ns, end_ns)
    duration_s = (end_ns - start_ns) / 1e9
    return {
        "boundary": "paid_trace_interval",
        "cpu_package_average_power_w": cpu_energy_j / duration_s,
        "cpu_package_energy_j": cpu_energy_j,
        "gpu_board_average_power_w": gpu_energy_j / duration_s,
        "gpu_board_energy_j": gpu_energy_j,
        "method": "RAPL package delta plus trapezoidal NVML 1-second average power",
        "server_compute_device_energy_j": cpu_energy_j + gpu_energy_j,
        "unaccounted": [
            "AC conversion",
            "DRAM outside package RAPL",
            "fans",
            "motherboard",
            "storage",
        ],
    }


def http_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        value = json.loads(response.read())
    require(type(value) is dict, "HTTP object")
    return value


class EventWriter:
    def __init__(self, path: Path):
        self.stream = path.open("xb", buffering=0)
        self.lock = threading.Lock()

    def write(self, value: dict[str, Any]) -> None:
        with self.lock:
            self.stream.write(canonical(value))

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


class CapturedProcess:
    def __init__(
        self,
        command: list[str],
        environment: dict[str, str],
        output: Path,
        label: str,
        stdin: bool = False,
        capture_stdout: bool = False,
    ):
        self.command = command
        self.environment = environment
        self.output = output
        self.label = label
        self.stdin_enabled = stdin
        self.capture_stdout = capture_stdout
        self.process: subprocess.Popen[str] | None = None
        self.stderr_lines: list[str] = []
        self.stderr_thread: threading.Thread | None = None
        self.stderr_condition = threading.Condition()
        self.stderr_file = None
        self.stdout_file = None

    @property
    def pid(self) -> int:
        require(self.process is not None, f"{self.label}: not started")
        return self.process.pid

    def start(self) -> None:
        self.stderr_file = (self.output / f"{self.label}.stderr").open(
            "x", encoding="utf-8"
        )
        stdout: Any
        if self.capture_stdout:
            stdout = subprocess.PIPE
        else:
            self.stdout_file = (self.output / f"{self.label}.stdout").open(
                "x", encoding="utf-8"
            )
            stdout = self.stdout_file
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE if self.stdin_enabled else subprocess.DEVNULL,
            stdout=stdout,
            stderr=subprocess.PIPE,
            env=self.environment,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            bufsize=1,
            start_new_session=True,
        )
        self.stderr_thread = threading.Thread(
            target=self._read_stderr,
            name=f"{self.label}-stderr",
            daemon=True,
        )
        self.stderr_thread.start()

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        assert self.stderr_file is not None
        for line in self.process.stderr:
            self.stderr_file.write(line)
            self.stderr_file.flush()
            with self.stderr_condition:
                self.stderr_lines.append(line.rstrip("\n"))
                self.stderr_condition.notify_all()

    def wait_stderr(self, prefix: str, timeout: float) -> str:
        deadline = time.monotonic() + timeout
        with self.stderr_condition:
            while True:
                for line in self.stderr_lines:
                    if line.startswith(prefix):
                        return line
                require(
                    self.process is not None and self.process.poll() is None,
                    f"{self.label}: exited before {prefix}",
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RunError(f"{self.label}: timeout waiting for {prefix}")
                self.stderr_condition.wait(min(remaining, 0.2))

    def terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        if self.stderr_thread is not None:
            self.stderr_thread.join(timeout=5)
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None
        if self.stdout_file is not None:
            self.stdout_file.close()
            self.stdout_file = None


class ColdDriver(CapturedProcess):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, stdin=True, capture_stdout=True, **kwargs)
        self.lock = threading.Lock()

    def exchange(
        self,
        launch_id: int,
        row: dict[str, Any],
        session_end: str,
        prefill_columns: int,
        decode_columns: int,
    ) -> dict[str, Any]:
        command = {
            "ffn_decode_columns": decode_columns,
            "ffn_prefill_columns": prefill_columns,
            "launch_id": launch_id,
            "n_gen": row["output_tokens"],
            "prompt_tokens": row["prompt_tokens"],
            "request_count": 1,
            "schema": "layersplit-persistent-command-v3",
            "session_end": session_end,
        }
        with self.lock:
            require(self.process is not None, "cold driver not started")
            require(
                self.process.stdin is not None and self.process.stdout is not None,
                "cold driver pipes",
            )
            self.process.stdin.write(canonical(command).decode("ascii"))
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        require(line, "cold driver reply EOF")
        value = json.loads(line)
        require(
            type(value) is dict
            and value.get("schema") == "layersplit-persistent-result-v3"
            and value.get("launch_id") == launch_id
            and value.get("request_count") == 1
            and value.get("outcome") == "completed"
            and value.get("session_end") == session_end,
            "cold driver reply identity",
        )
        require(
            value.get("ffn_prefill_columns") == prefill_columns
            and value.get("ffn_decode_columns") == decode_columns,
            "cold driver split identity",
        )
        require(
            type(value.get("prefill_us")) is int
            and value["prefill_us"] > 0
            and type(value.get("decode_us")) is int
            and value["decode_us"] > 0
            and value["prefill_us"] + value["decode_us"]
                <= value["route_wall_us"],
            "cold driver timing",
        )
        tokens = value.get("token_ids")
        require(
            type(tokens) is list
            and len(tokens) == 1
            and type(tokens[0]) is list
            and len(tokens[0]) == row["output_tokens"]
            and all(type(token) is int for token in tokens[0]),
            "cold driver tokens",
        )
        return value


class ResourceSampler:
    def __init__(self, output: Path, hot_pid: int, cold_pid: int):
        self.output = output
        self.hot_pid = hot_pid
        self.cold_pid = cold_pid
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                gpu_before_ns = time.monotonic_ns()
                gpu = gpu_snapshot()
                gpu_after_ns = time.monotonic_ns()
                gpu["sample_t_ns"] = (gpu_before_ns + gpu_after_ns) // 2
                try:
                    rapl_package = rapl_package_snapshot()
                except (FileNotFoundError, PermissionError, RunError, ValueError):
                    rapl_package = None
                row = {
                    "cold": proc_status_or_zero(self.cold_pid),
                    "gpu": gpu,
                    "hot": proc_status_or_zero(self.hot_pid),
                    "rapl_package": rapl_package,
                    "schema": "s41-burstgpt-cpu-op15-resource-v1",
                    "system": system_memory(),
                    "t_ns": time.monotonic_ns(),
                }
                self.rows.append(row)
                self.stop_event.wait(0.5)
            with (self.output / "resource-samples.jsonl").open("xb") as stream:
                for row in self.rows:
                    stream.write(canonical(row))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"

    def stop(self) -> None:
        self.stop_event.set()
        require(self.thread is not None, "resource sampler thread")
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), "resource sampler did not stop")
        require(self.error is None, f"resource sampler: {self.error}")


def hot_completion(
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
        data=canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final = None
    tokens: list[int] = []
    with stream_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                payload_line = raw_line.decode("utf-8").strip()
                if not payload_line.startswith("data:"):
                    continue
                payload = payload_line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                value = json.loads(payload)
                require(type(value) is dict and "error" not in value, "hot chunk")
                chunk = value.get("tokens", [])
                require(
                    type(chunk) is list
                    and all(type(token) is int for token in chunk),
                    "hot tokens",
                )
                if chunk and not tokens:
                    on_first(time.monotonic_ns())
                tokens.extend(chunk)
                if value.get("stop", False):
                    final = value
    require(final is not None, "hot completion final")
    timings = final.get("timings")
    require(
        type(timings) is dict
        and timings.get("prompt_n") == row["input_tokens"]
        and timings.get("predicted_n") == row["output_tokens"]
        and len(tokens) == row["output_tokens"],
        "hot token accounting",
    )
    return {
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"),
        "tokens": tokens,
    }


def start_hot(args: argparse.Namespace, output: Path) -> CapturedProcess:
    n_gpu_layers = getattr(args, "hot_n_gpu_layers", "all")
    command = [
        str(args.hot_server),
        "--model", str(args.hot_model),
        "--alias", HOT_MODEL,
        "--fit", "off",
        "--ctx-size", str(args.hot_ctx_size),
        "--parallel", str(args.hot_parallel),
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--flash-attn", "on",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--split-mode", "none",
        "--n-gpu-layers", str(n_gpu_layers),
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
    hot_nice = getattr(args, "hot_nice", 0)
    if hot_nice:
        command = ["nice", "-n", str(hot_nice), *command]
    hot_cpus = getattr(args, "hot_cpus", None)
    if hot_cpus:
        command = ["taskset", "--cpu-list", hot_cpus, *command]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cuda_lib_dir}:{args.hot_server.parent}:"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    split_names = (
        "S41_SERVER_FFN_HOST",
        "S41_SERVER_FFN_PORT",
        "S41_SERVER_FFN_N_EMBD",
        "S41_SERVER_FFN_LAYER_MASK",
        "S41_SERVER_FFN_COLUMNS",
        "S41_SERVER_FFN_F16_IO",
        "S41_SERVER_FFN_ACTIVATION",
        "S41_SERVER_FFN_POLICY",
        "S41_SERVER_FFN_TIMEOUT_MS",
        "LLAMA_FFN_SPLIT_LAYER_MASK",
        "LLAMA_FFN_SPLIT_COLUMNS",
        "LLAMA_FFN_SPLIT_POLICY",
        "LLAMA_FFN_SPLIT_VIEW_SAFE_WEIGHTS",
    )
    for name in split_names:
        environment.pop(name, None)
    hot_ffn_environment = getattr(args, "hot_ffn_environment", None)
    if hot_ffn_environment is not None:
        require(type(hot_ffn_environment) is dict,
                "hot FFN environment")
        environment.update(hot_ffn_environment)
    process = CapturedProcess(command, environment, output, "hot-server")
    process.start()
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        require(
            process.process is not None and process.process.poll() is None,
            "hot server exited during load",
        )
        try:
            if http_json(
                    f"http://127.0.0.1:{args.hot_port}/health", 1
            ).get("status") == "ok":
                break
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.1)
    else:
        raise RunError("hot server readiness timeout")
    log = "\n".join(process.stderr_lines)
    matches = re.findall(r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log)
    if str(n_gpu_layers) == "all":
        placement_ok = any(
            int(loaded) == int(total) and int(total) > 0
            for loaded, total in matches
        )
    else:
        placement_ok = any(
            int(loaded) == int(n_gpu_layers) and int(total) > 0
            for loaded, total in matches
        )
    require(placement_ok, "hot model GPU layer placement")
    return process


def start_bridge(args: argparse.Namespace, output: Path) -> CapturedProcess:
    allocator = getattr(args, "bridge_allocator", "devmem")
    bind = getattr(args, "bridge_bind", "127.0.0.1")
    command = [
        str(args.bridge),
        bind,
        str(args.bridge_port),
        allocator,
    ]
    bridge_cpus = getattr(args, "bridge_cpus", None)
    if bridge_cpus:
        command = ["taskset", "--cpu-list", bridge_cpus, *command]
    process = CapturedProcess(
        command,
        os.environ.copy(),
        output,
        "dmabuf-bridge",
    )
    process.start()
    process.wait_stderr("[ffn-dmabuf-bridge] ready", 30)
    return process


def start_cold(
    args: argparse.Namespace,
    output: Path,
) -> ColdDriver:
    mode = "overlapdriver" if args.mode == "op15" else "monodriver"
    command = [
        str(args.cold_driver),
        "-m", str(args.cold_model),
        "-ngl", "0",
        "-t", "8",
        "-tb", "8",
        "--mode", mode,
        "--persistent-jsonl",
        "--driver-batch", "1",
        "--driver-context", str(args.driver_context),
        "--driver-max-prefill", str(args.driver_max_prefill),
        "--driver-warmup", "0",
        "--driver-requests", "1",
        "--no-repack",
        "-n", str(args.max_n_gen),
    ]
    if args.mode == "op15":
        command.extend([
            "--host", "127.0.0.1",
            "--port", str(args.bridge_port),
            "--ffn-layers", "0-47",
            "--ffn-columns", str(args.max_columns),
            "--ffn-f16-io",
        ])
    cold_cpus = getattr(args, "cold_cpus", None)
    if cold_cpus:
        command = ["taskset", "--cpu-list", cold_cpus, *command]
    environment = os.environ.copy()
    environment["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cold_lib_dir}:" + environment.get("LD_LIBRARY_PATH", "")
    )
    process = ColdDriver(
        command,
        environment,
        output,
        "cold-driver",
    )
    process.start()
    ready = process.wait_stderr("PERSISTENT_DRIVER_READY ", 300)
    value = json.loads(ready.split(" ", 1)[1])
    require(
        value.get("batch_size") == 1
        and value.get("max_n_gen") == args.max_n_gen,
        "cold driver readiness identity",
    )
    return process


def role(row: dict[str, Any]) -> str:
    if row["model_id"] == SOURCE_HOT:
        return "hot"
    if row["model_id"] == SOURCE_COLD:
        return "cold"
    raise RunError("unexpected source model")


def trace_metrics(
    results: list[dict[str, Any]],
    paid_start_ns: int,
) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    for route in ("hot", "cold"):
        rows = [row for row in results if row["role"] == route]
        if not rows:
            by_role[route] = {"completed": 0}
            continue
        completion = [
            (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        ]
        service = [
            (row["completion_ns"] - row["dispatch_ns"]) / 1e9
            for row in rows
        ]
        queue_s = [
            (row["dispatch_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        ]
        record = {
            "completed": len(rows),
            "completion_s": stats(completion),
            "queue_s": stats(queue_s),
            "service_s": stats(service),
            "slo_met": sum(
                row["completion_ns"] - row["scheduled_arrival_ns"]
                <= row["slo_us"] * 1000
                for row in rows
            ),
        }
        if route == "hot":
            record["ttft_s"] = stats([
                (row["first_token_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in rows
            ])
        else:
            record["route_wall_s"] = stats([
                row["route_wall_us"] / 1e6 for row in rows
            ])
            record["prefill_s"] = stats([
                row["prefill_us"] / 1e6 for row in rows
            ])
            record["decode_s"] = stats([
                row["decode_us"] / 1e6 for row in rows
            ])
        by_role[route] = record
    paid_end_ns = max(row["completion_ns"] for row in results)
    duration_s = (paid_end_ns - paid_start_ns) / 1e9
    return {
        "by_role": by_role,
        "completed": len(results),
        "duration_s": duration_s,
        "slo_met": sum(
            row["completion_ns"] - row["scheduled_arrival_ns"]
            <= row["slo_us"] * 1000
            for row in results
        ),
        "throughput_tokens_s": (
            sum(len(row["tokens"]) for row in results) / duration_s
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu", "op15"), required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--hot-server", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cold-driver", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--hot-port", type=int, default=18480)
    parser.add_argument(
        "--trace-profile", choices=("long", "source"), default="long"
    )
    parser.add_argument(
        "--hot-load", choices=("absent", "trace"), default="trace"
    )
    parser.add_argument("--driver-context", type=int, default=1024)
    parser.add_argument("--driver-max-prefill", type=int, default=512)
    parser.add_argument("--max-n-gen", type=int, default=32)
    parser.add_argument("--hot-ctx-size", type=int, default=4096)
    parser.add_argument("--hot-parallel", type=int, default=8)
    parser.add_argument("--hot-workers", type=int, default=8)
    parser.add_argument("--cold-cpus")
    parser.add_argument("--hot-cpus")
    parser.add_argument("--bridge-cpus")
    parser.add_argument("--max-columns", type=int, default=9664)
    parser.add_argument("--decode-columns", type=int, default=9664)
    parser.add_argument(
        "--prefill-policy",
        default="64:8192,128:8192,320:9664,512:9664",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    require(
        args.execute
        and args.confirm == "RUN_BURSTGPT_GPU_CPU_OP15_TRACE",
        "confirmation",
    )
    require(args.output.is_absolute() and not args.output.exists(), "output")
    prefill_policy = parse_prefill_policy(args.prefill_policy)
    require(
        args.driver_context > 0
        and 0 < args.driver_max_prefill <= 4096
        and args.max_n_gen > 0
        and args.hot_ctx_size > 0
        and args.hot_parallel > 0
        and args.hot_workers > 0,
        "runtime bounds",
    )
    supported_width = lambda columns: (
        columns == 0
        or columns == args.max_columns
        or (columns > 0 and columns % 512 == 0)
    )
    require(
        args.max_columns == 9664
        and supported_width(args.decode_columns)
        and args.decode_columns <= args.max_columns
        and all(
            supported_width(columns) and columns <= args.max_columns
            for _, columns in prefill_policy
        ),
        "split policy",
    )
    if args.mode == "cpu":
        require(
            args.decode_columns == 0
            and all(columns == 0 for _, columns in prefill_policy),
            "CPU control must disable phone work",
        )
    for path in (
        args.requests,
        args.hot_server,
        args.hot_model,
        args.cuda_lib_dir,
        args.cold_driver,
        args.cold_model,
        args.cold_lib_dir,
    ):
        require(path.exists(), f"missing path: {path}")
    if args.mode == "op15":
        require(args.bridge is not None and args.bridge.exists(), "bridge")

    requests = read_jsonl(args.requests)
    requests_sha256 = digest_file(args.requests)
    if args.trace_profile == "long":
        trace_identity = (
            requests_sha256 == LONG_REQUESTS_SHA256
            and all(
                row["schema"] == "s41-gemma-qwen-request-semantic-long-v1"
                and row["input_tokens"]
                    == min(row["source_input_tokens"], 512)
                and row["output_tokens"]
                    == min(row["source_output_tokens"], 32)
                for row in requests
            )
        )
    else:
        trace_identity = (
            requests_sha256 == SOURCE_REQUESTS_SHA256
            and all(
                row["schema"]
                    == "s41-gemma-qwen-request-semantic-source-v1"
                and row["input_tokens"] == row["source_input_tokens"]
                and row["output_tokens"] == row["source_output_tokens"]
                for row in requests
            )
        )
    require(
        trace_identity
        and len(requests) == 74
        and sum(role(row) == "hot" for row in requests) == 57
        and sum(role(row) == "cold" for row in requests) == 17
        and all(
            len(row["prompt_tokens"]) == row["input_tokens"]
            and row["slo_us"] == 30_000_000
            and row["prompt_tokenizer_model"] == (
                HOT_MODEL if role(row) == "hot" else COLD_MODEL
            )
            and all(
                type(token) is int
                and 0 <= token < (
                    151_936 if role(row) == "hot" else 262_144
                )
                for token in row["prompt_tokens"]
            )
            for row in requests
        ),
        "trace identity",
    )
    cold_requests = [row for row in requests if role(row) == "cold"]
    hot_requests = [row for row in requests if role(row) == "hot"]
    require(
        args.driver_max_prefill
            >= max(row["input_tokens"] for row in cold_requests)
        and args.max_n_gen
            >= max(row["output_tokens"] for row in cold_requests)
        and args.driver_context
            >= max(
                row["input_tokens"] + row["output_tokens"]
                for row in cold_requests
            )
        and prefill_policy[-1][0]
            >= max(row["input_tokens"] for row in cold_requests),
        "cold shape coverage",
    )
    if args.hot_load == "trace":
        require(
            args.hot_ctx_size
                >= max(
                    row["input_tokens"] + row["output_tokens"]
                    for row in hot_requests
                ),
            "hot context coverage",
        )
    require(
        args.hot_model.stat().st_size == 9_001_752_960
        and digest_file(args.hot_model) == HOT_MODEL_SHA256,
        "hot model identity",
    )
    require(
        args.cold_model.stat().st_size == 6_975_878_176
        and digest_file(args.cold_model) == COLD_MODEL_SHA256,
        "cold model identity",
    )

    args.output.mkdir(parents=True)
    events = EventWriter(args.output / "events.jsonl")
    hot = None
    bridge = None
    cold = None
    sampler = None
    errors: list[str] = []
    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    try:
        preflight = {
            "cold_driver": {
                "path": str(args.cold_driver),
                "sha256": digest_file(args.cold_driver),
            },
            "cold_model": {
                "path": str(args.cold_model),
                "sha256": COLD_MODEL_SHA256,
            },
            "gpu": gpu_snapshot(),
            "hot_model": {
                "path": str(args.hot_model),
                "sha256": HOT_MODEL_SHA256,
            },
            "hot_server": {
                "path": str(args.hot_server),
                "sha256": digest_file(args.hot_server),
            },
            "hot_load": args.hot_load,
            "hot_runtime": {
                "context": args.hot_ctx_size,
                "parallel": args.hot_parallel,
                "workers": args.hot_workers,
            },
            "mode": args.mode,
            "offload": {
                "decode_columns": args.decode_columns,
                "max_columns": args.max_columns,
                "prefill_policy": [
                    {"max_tokens": limit, "columns": columns}
                    for limit, columns in prefill_policy
                ],
            },
            "requests_sha256": requests_sha256,
            "trace_profile": args.trace_profile,
            "cold_runtime": {
                "context": args.driver_context,
                "max_n_gen": args.max_n_gen,
                "max_prefill": args.driver_max_prefill,
            },
            "role_remap": {
                SOURCE_COLD: COLD_MODEL,
                SOURCE_HOT: HOT_MODEL,
            },
            "schema": "s41-burstgpt-cpu-op15-preflight-v1",
            "system_memory": system_memory(),
        }
        write_json(args.output / "preflight.json", preflight)

        if args.hot_load == "trace":
            hot = start_hot(args, args.output)
        if args.mode == "op15":
            bridge = start_bridge(args, args.output)
        cold = start_cold(args, args.output)
        if hot is not None:
            require(proc_status(hot.pid)["swap_bytes"] == 0, "hot process swap")
        require(proc_status(cold.pid)["swap_bytes"] == 0, "cold process swap")

        cold_warm = min(
            cold_requests,
            key=lambda row: row["input_tokens"] + row["output_tokens"],
        )
        if hot is None:
            cold.exchange(
                1,
                cold_warm,
                "DETACH",
                policy_columns(prefill_policy, cold_warm["input_tokens"]),
                args.decode_columns,
            )
        else:
            hot_warm = min(
                hot_requests,
                key=lambda row: row["input_tokens"] + row["output_tokens"],
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as warm_pool:
                hot_future = warm_pool.submit(
                    hot_completion,
                    args.hot_port,
                    hot_warm,
                    args.output / "warm-hot.raw",
                    lambda _: None,
                )
                cold_future = warm_pool.submit(
                    cold.exchange,
                    1,
                    cold_warm,
                    "DETACH",
                    policy_columns(prefill_policy, cold_warm["input_tokens"]),
                    args.decode_columns,
                )
                hot_future.result(timeout=3600)
                cold_future.result(timeout=3600)

        sampler = ResourceSampler(
            args.output, hot.pid if hot is not None else 0, cold.pid
        )
        sampler.start()
        time.sleep(0.6)
        paid_start_ns = time.monotonic_ns()
        events.write({
            "kind": "trace_start",
            "mode": args.mode,
            "repeat_index": args.repeat_index,
            "schema": "s41-burstgpt-cpu-op15-event-v1",
            "t_ns": paid_start_ns,
        })

        cold_rows = [row for row in requests if role(row) == "cold"]
        last_cold_index = cold_rows[-1]["request_index"]
        cold_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        launch_id = 1

        def run_cold_queue() -> None:
            nonlocal launch_id
            while True:
                row = cold_queue.get()
                if row is None:
                    cold_queue.task_done()
                    return
                dispatch_ns = time.monotonic_ns()
                try:
                    launch_id += 1
                    prefill_columns = policy_columns(
                        prefill_policy, row["input_tokens"]
                    )
                    value = cold.exchange(
                        launch_id,
                        row,
                        "STOP" if row["request_index"] == last_cold_index
                        else "DETACH",
                        prefill_columns,
                        args.decode_columns,
                    )
                    completion_ns = time.monotonic_ns()
                    record = {
                        "completion_ns": completion_ns,
                        "dispatch_ns": dispatch_ns,
                        "effective_model_id": COLD_MODEL,
                        "event_id": row["event_id"],
                        "ffn_decode_columns": args.decode_columns,
                        "ffn_prefill_columns": prefill_columns,
                        "first_token_ns": None,
                        "decode_us": value["decode_us"],
                        "prefill_us": value["prefill_us"],
                        "request_index": row["request_index"],
                        "role": "cold",
                        "route_wall_us": value["route_wall_us"],
                        "scheduled_arrival_ns":
                            paid_start_ns + row["arrival_us"] * 1000,
                        "schema": "s41-burstgpt-cpu-op15-request-v1",
                        "slo_us": row["slo_us"],
                        "source_model_id": row["model_id"],
                        "tokens": value["token_ids"][0],
                    }
                    with results_lock:
                        results.append(record)
                    events.write({"kind": "request_complete", **record})
                except BaseException as error:
                    with results_lock:
                        errors.append(
                            f"{row['request_index']}: "
                            f"{type(error).__name__}: {error}"
                        )
                finally:
                    cold_queue.task_done()

        cold_thread = threading.Thread(
            target=run_cold_queue,
            name="cold-queue",
            daemon=True,
        )
        cold_thread.start()

        def run_hot(row: dict[str, Any]) -> None:
            dispatch_ns = time.monotonic_ns()
            first: list[int] = []
            try:
                require(hot is not None, "hot server unavailable")
                value = hot_completion(
                    args.hot_port,
                    row,
                    args.output / f"stream-{row['request_index']:03d}.raw",
                    first.append,
                )
                require(len(first) == 1, "hot first token")
                completion_ns = time.monotonic_ns()
                record = {
                    "completion_ns": completion_ns,
                    "dispatch_ns": dispatch_ns,
                    "effective_model_id": HOT_MODEL,
                    "event_id": row["event_id"],
                    "first_token_ns": first[0],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "request_index": row["request_index"],
                    "role": "hot",
                    "scheduled_arrival_ns":
                        paid_start_ns + row["arrival_us"] * 1000,
                    "schema": "s41-burstgpt-cpu-op15-request-v1",
                    "slo_us": row["slo_us"],
                    "source_model_id": row["model_id"],
                    "tokens": value["tokens"],
                }
                with results_lock:
                    results.append(record)
                events.write({"kind": "request_complete", **record})
            except BaseException as error:
                with results_lock:
                    errors.append(
                        f"{row['request_index']}: "
                        f"{type(error).__name__}: {error}"
                    )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.hot_workers,
            thread_name_prefix="hot",
        ) as hot_pool:
            hot_futures = []
            for row in requests:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while True:
                    remaining_ns = target_ns - time.monotonic_ns()
                    if remaining_ns <= 0:
                        break
                    time.sleep(min(remaining_ns / 1e9, 0.01))
                events.write({
                    "actual_t_ns": time.monotonic_ns(),
                    "event_id": row["event_id"],
                    "kind": "request_arrival",
                    "request_index": row["request_index"],
                    "role": role(row),
                    "scheduled_t_ns": target_ns,
                    "schema": "s41-burstgpt-cpu-op15-event-v1",
                })
                if role(row) == "cold":
                    cold_queue.put(row)
                elif hot is not None:
                    hot_futures.append(hot_pool.submit(run_hot, row))
            for future in hot_futures:
                future.result(timeout=3600)

        cold_queue.put(None)
        cold_queue.join()
        cold_thread.join(timeout=10)
        require(not cold_thread.is_alive(), "cold queue did not stop")
        require(not errors, "request errors: " + "; ".join(errors))
        expected_results = 74 if hot is not None else 17
        require(len(results) == expected_results, "request conservation")

        time.sleep(0.6)
        sampler.stop()
        resource_rows = list(sampler.rows)
        sampler = None
        metrics = trace_metrics(results, paid_start_ns)
        paid_end_ns = max(row["completion_ns"] for row in results)
        events.write({
            "kind": "trace_end",
            "schema": "s41-burstgpt-cpu-op15-event-v1",
            "t_ns": paid_end_ns,
        })

        require(cold.process is not None, "cold process")
        cold.process.wait(timeout=60)
        require(cold.process.returncode == 0, "cold process status")
        if bridge is not None:
            require(bridge.process is not None, "bridge process")
            bridge.process.wait(timeout=60)
            require(bridge.process.returncode == 0, "bridge process status")

        ffn_lines = [
            line for line in cold.stderr_lines if line.startswith("FFNSPLIT ")
        ]
        bridge_lines = [
            line for line in (bridge.stderr_lines if bridge else [])
            if line.startswith("FFNDMABUF ")
        ]
        if args.mode == "op15":
            require(
                len(ffn_lines) == 1 and len(bridge_lines) == 1,
                "offload summaries",
            )
        resource = {
            "cold_after": proc_status(cold.pid)
                if Path(f"/proc/{cold.pid}").exists() else None,
            "cold_log_ffn": (
                json.loads(ffn_lines[0].split(" ", 1)[1])
                if ffn_lines else None
            ),
            "dmabuf": (
                json.loads(bridge_lines[0].split(" ", 1)[1])
                if bridge_lines else None
            ),
            "gpu_after": gpu_snapshot(),
            "hot_after": proc_status(hot.pid) if hot is not None else None,
            "samples": {
                "cold_rss_max_bytes": max(
                    row["cold"]["rss_bytes"] for row in resource_rows
                ),
                "cold_swap_max_bytes": max(
                    row["cold"]["swap_bytes"] for row in resource_rows
                ),
                "gpu_power_mean_mw": round(
                    sum(row["gpu"]["power_mw"] for row in resource_rows)
                    / len(resource_rows)
                ),
                "gpu_utilization_mean_pct": (
                    sum(
                        row["gpu"]["utilization_pct"]
                        for row in resource_rows
                    ) / len(resource_rows)
                ),
                "hot_rss_max_bytes": max(
                    row["hot"]["rss_bytes"] for row in resource_rows
                ),
                "hot_swap_max_bytes": max(
                    row["hot"]["swap_bytes"] for row in resource_rows
                ),
                "sample_count": len(resource_rows),
                "system_available_min_bytes": min(
                    row["system"]["available_bytes"]
                    for row in resource_rows
                ),
                "system_swap_free_min_bytes": min(
                    row["system"]["swap_free_bytes"]
                    for row in resource_rows
                ),
            },
            "system_after": system_memory(),
        }
        result = {
            "metrics": metrics,
            "hot_load": args.hot_load,
            "mode": args.mode,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "preflight": preflight,
            "repeat_index": args.repeat_index,
            "request_results": sorted(
                results, key=lambda row: row["request_index"]
            ),
            "resources": resource,
            "schema": "s41-burstgpt-cpu-op15-result-v1",
            "status": "PASS",
        }
        write_json(args.output / "RESULT.json", result)
        return 0
    except BaseException as error:
        if args.output.exists():
            write_json(args.output / "FAILURE.json", {
                "error": f"{type(error).__name__}: {error}",
                "mode": args.mode,
                "repeat_index": args.repeat_index,
                "schema": "s41-burstgpt-cpu-op15-failure-v1",
                "status": "FAIL",
            })
        return 2
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


if __name__ == "__main__":
    raise SystemExit(main())
