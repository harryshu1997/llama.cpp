#!/usr/bin/env python3
"""Run CP0-D qualification, non-co-residency, and swap replay phases."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import cache_control
import validate_inputs


HERE = Path(__file__).resolve().parent


class RunError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n").encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_bytes().splitlines()):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunError(f"{path}:{index + 1}: invalid JSON") from exc
        if type(value) is not dict:
            raise RunError(f"{path}:{index + 1}: row is not an object")
        rows.append(value)
    return rows


def http_json(
        url: str, body: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = None if body is None else canonical(body)
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self._stream = path.open("xb")
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._stream.write(canonical(row))
            self._stream.flush()

    def close(self) -> None:
        with self._lock:
            self._stream.close()


def parse_kib(value: str) -> int:
    fields = value.split()
    if len(fields) != 2 or fields[1] != "kB":
        raise RunError(f"invalid /proc memory value: {value!r}")
    return int(fields[0]) * 1024


def proc_status(pid: int) -> dict[str, int]:
    values = {
        "process_rss_bytes": 0,
        "process_swap_bytes": 0,
    }
    try:
        lines = Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        if line.startswith("VmRSS:"):
            values["process_rss_bytes"] = parse_kib(line.split(":", 1)[1].strip())
        elif line.startswith("VmSwap:"):
            values["process_swap_bytes"] = parse_kib(line.split(":", 1)[1].strip())
    return values


def system_memory() -> dict[str, int]:
    wanted = {
        "MemAvailable": "system_mem_available_bytes",
        "SwapFree": "system_swap_free_bytes",
        "SwapTotal": "system_swap_total_bytes",
    }
    output: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        key, _, value = line.partition(":")
        if key in wanted:
            output[wanted[key]] = parse_kib(value.strip())
    if set(output) != set(wanted.values()):
        raise RunError("missing /proc/meminfo fields")
    return output


def gpu_snapshot(gpu_index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi", "-i", str(gpu_index),
        "--query-gpu=name,uuid,memory.total,memory.used,memory.free,"
        "utilization.gpu,pstate,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RunError(f"nvidia-smi snapshot failed: {result.stderr.strip()}")
    parts = [part.strip() for part in result.stdout.strip().split(",")]
    if len(parts) != 9:
        raise RunError(f"nvidia-smi snapshot has {len(parts)} fields")
    return {
        "gpu_name": parts[0],
        "gpu_uuid": parts[1],
        "gpu_memory_total_bytes": int(parts[2]) * 1024 * 1024,
        "gpu_memory_used_bytes": int(parts[3]) * 1024 * 1024,
        "gpu_memory_free_bytes": int(parts[4]) * 1024 * 1024,
        "gpu_utilization_milli_pct": int(parts[5]) * 1000,
        "gpu_pstate": parts[6],
        "gpu_power_mw": int(round(float(parts[7]) * 1000)),
        "gpu_power_limit_mw": int(round(float(parts[8]) * 1000)),
    }


def compute_processes() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise RunError(f"nvidia-smi process query failed: {result.stderr.strip()}")
    output: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RunError("malformed nvidia-smi process row")
        output.append({
            "pid": int(fields[0]),
            "used_memory_bytes": int(fields[1]) * 1024 * 1024,
        })
    return output


class PowerSampler:
    def __init__(self, path: Path, gpu_index: int,
                 pid_getter: Callable[[], int | None]) -> None:
        self.writer = JsonlWriter(path)
        self.gpu_index = gpu_index
        self.pid_getter = pid_getter
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi", "-i", str(self.gpu_index),
                "--query-gpu=power.draw.instant,power.draw,power.limit,"
                "memory.used,memory.free,utilization.gpu,pstate",
                "--format=csv,noheader,nounits", "-lms", "100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        previous_ns = -1
        try:
            for line in self.process.stdout:
                now_ns = time.monotonic_ns()
                if now_ns <= previous_ns:
                    now_ns = previous_ns + 1
                previous_ns = now_ns
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 7 or any(part in {"N/A", "[N/A]"} for part in parts):
                    raise RunError(f"malformed power row: {line.strip()}")
                pid = self.pid_getter()
                row = {
                    "gpu_memory_free_bytes": int(parts[4]) * 1024 * 1024,
                    "gpu_memory_used_bytes": int(parts[3]) * 1024 * 1024,
                    "gpu_power_average_mw": int(round(float(parts[1]) * 1000)),
                    "gpu_power_instant_mw": int(round(float(parts[0]) * 1000)),
                    "gpu_power_limit_mw": int(round(float(parts[2]) * 1000)),
                    "gpu_pstate": parts[6],
                    "gpu_utilization_milli_pct": int(parts[5]) * 1000,
                    "schema": "s39-cp0d-resource-sample-v1",
                    "server_pid": pid,
                    "t_ns": now_ns,
                    **system_memory(),
                }
                if pid is not None:
                    row.update(proc_status(pid))
                else:
                    row.update({
                        "process_rss_bytes": 0,
                        "process_swap_bytes": 0,
                    })
                self.rows.append(row)
                self.writer.write(row)
        except (RunError, ValueError, OSError) as exc:
            self.error = str(exc)

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                raise RunError("power sampler did not terminate")
        self.writer.close()
        if self.error is not None:
            raise RunError(f"power sampler failed: {self.error}")

    def integrate(self, start_ns: int, end_ns: int) -> dict[str, int]:
        if end_ns <= start_ns:
            raise RunError("energy window is empty")
        rows = sorted(self.rows, key=lambda row: row["t_ns"])
        if not rows or rows[0]["t_ns"] > start_ns or rows[-1]["t_ns"] < end_ns:
            raise RunError("power samples do not bracket paid window")
        energy_nj = 0
        maximum_gap_ns = 0
        in_window = 0
        changes = 0
        previous_power: int | None = None
        for row in rows:
            if start_ns <= row["t_ns"] <= end_ns:
                in_window += 1
                power = row["gpu_power_instant_mw"]
                if previous_power is not None and power != previous_power:
                    changes += 1
                previous_power = power
        for left, right in zip(rows, rows[1:]):
            gap = right["t_ns"] - left["t_ns"]
            lo = max(start_ns, left["t_ns"])
            hi = min(end_ns, right["t_ns"])
            if hi > lo:
                energy_nj += left["gpu_power_instant_mw"] * (hi - lo) // 1_000
                maximum_gap_ns = max(maximum_gap_ns, gap)
        if in_window < 20 or changes < 5 or maximum_gap_ns > 1_000_000_000:
            raise RunError(
                f"power quality failed: rows={in_window} changes={changes} "
                f"max_gap_ns={maximum_gap_ns}"
            )
        return {
            "energy_nj": energy_nj,
            "independent_power_changes": changes,
            "maximum_sample_gap_ns": maximum_gap_ns,
            "sample_count": in_window,
            "window_end_ns": end_ns,
            "window_start_ns": start_ns,
        }


class ServerProcess:
    def __init__(self, runner: "Runner", model_id: str, port: int,
                 label: str) -> None:
        self.runner = runner
        self.model_id = model_id
        self.port = port
        self.label = label
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout = None
        self.stderr = None
        self.stdout_path: Path | None = None
        self.stderr_path: Path | None = None
        self.command: list[str] = []
        self.started_ns = 0
        self.ready_ns = 0

    @property
    def pid(self) -> int | None:
        return None if self.process is None or self.process.poll() is not None \
            else self.process.pid

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def build_command(self) -> list[str]:
        model = self.runner.models[self.model_id]
        command = [
            str(self.runner.server),
            "--model", str(model["path"]),
            "--alias", self.model_id,
        ]
        if self.runner.serving_profile == "fixed_full_cuda":
            command.extend([
                "--n-gpu-layers", "all",
                "--split-mode", "none",
                "--main-gpu", "0",
                "--device", "CUDA0",
                "--fit", "off",
                "--ctx-size", "4096",
                "--parallel", "8",
                "--batch-size", "2048",
                "--ubatch-size", "512",
                "--flash-attn", "on",
                "--cont-batching",
                "--kv-unified",
                "--no-cache-idle-slots",
                "--cache-type-k", "f16",
                "--cache-type-v", "f16",
            ])
        elif self.runner.serving_profile != "stock_default":
            raise RunError(
                f"unknown serving profile {self.runner.serving_profile}"
            )
        command.extend([
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--metrics",
            "--slots",
            "--no-webui",
            "--log-colors", "off",
            "--log-timestamps",
            "--verbose",
        ])
        return command

    def start(self, require_headroom: bool = True) -> dict[str, Any]:
        index = self.runner.next_server_index()
        self.command = self.build_command()
        self.stdout_path = self.runner.output / f"server-{index:02d}-{self.label}.stdout"
        self.stderr_path = self.runner.output / f"server-{index:02d}-{self.label}.stderr"
        self.stdout = self.stdout_path.open("xb")
        self.stderr = self.stderr_path.open("xb")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.runner.gpu_index)
        environment["LD_LIBRARY_PATH"] = (
            f"{self.runner.cuda_lib_dir}:{self.runner.server.parent}:"
            + environment.get("LD_LIBRARY_PATH", "")
        )
        self.started_ns = time.monotonic_ns()
        self.runner.events.write({
            "command": self.command,
            "kind": "model_load_start",
            "label": self.label,
            "model_id": self.model_id,
            "schema": "s39-cp0d-server-event-v1",
            "t_ns": self.started_ns,
        })
        self.process = subprocess.Popen(
            self.command,
            stdout=self.stdout,
            stderr=self.stderr,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.runner.server_timeout_s
        last_error = "no response"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RunError(
                    f"{self.label}: server exited during load rc={self.process.returncode}"
                )
            try:
                health = http_json(self.base_url + "/health", timeout=1.0)
                if health.get("status") == "ok":
                    break
                last_error = repr(health)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.1)
        else:
            raise RunError(f"{self.label}: readiness timeout: {last_error}")
        self.ready_ns = time.monotonic_ns()
        assert self.stderr_path is not None
        load_log = self.stderr_path.read_text(encoding="utf-8", errors="replace")
        offload_matches = re.findall(
            r"offloaded ([0-9]+)/([0-9]+) layers to GPU", load_log
        )
        if not offload_matches:
            raise RunError(f"{self.label}: CUDA layer placement not observed")
        if self.runner.serving_profile == "fixed_full_cuda" and not any(
                int(loaded) == int(total) and int(total) > 0
                for loaded, total in offload_matches):
            raise RunError(f"{self.label}: full CUDA layer placement not proven")
        snapshot = gpu_snapshot(self.runner.gpu_index)
        props = http_json(self.base_url + "/props", timeout=5.0)
        status = proc_status(self.process.pid)
        if require_headroom and snapshot["gpu_memory_free_bytes"] < 536_870_912:
            raise RunError(f"{self.label}: insufficient post-load VRAM headroom")
        if status["process_swap_bytes"] != 0:
            raise RunError(f"{self.label}: process swap is nonzero")
        record = {
            "elapsed_ns": self.ready_ns - self.started_ns,
            "gpu": snapshot,
            "kind": "model_ready",
            "label": self.label,
            "model_id": self.model_id,
            "pid": self.process.pid,
            "process": status,
            "props": props,
            "serving_profile": self.runner.serving_profile,
            "full_cuda_offload_matches": offload_matches,
            "schema": "s39-cp0d-server-event-v1",
            "t_ns": self.ready_ns,
        }
        self.runner.events.write(record)
        return record

    def healthy(self) -> bool:
        if self.process is None or self.process.poll() is not None:
            return False
        try:
            return http_json(self.base_url + "/health", timeout=2.0).get("status") == "ok"
        except Exception:
            return False

    def stop(self) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        pid = None if self.process is None else self.process.pid
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        ended_ns = time.monotonic_ns()
        record = {
            "elapsed_ns": ended_ns - started_ns,
            "kind": "model_unloaded",
            "label": self.label,
            "model_id": self.model_id,
            "pid": pid,
            "returncode": None if self.process is None else self.process.returncode,
            "schema": "s39-cp0d-server-event-v1",
            "t_ns": ended_ns,
        }
        self.runner.events.write(record)
        if self.stdout is not None:
            self.stdout.close()
            self.stdout = None
        if self.stderr is not None:
            self.stderr.close()
            self.stderr = None
        return record


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.server = args.server.resolve()
        self.cuda_lib_dir = args.cuda_lib_dir.resolve()
        self.output = args.output.resolve()
        self.gpu_index = args.gpu_index
        self.server_timeout_s = args.server_timeout_s
        self.serving_profile = args.serving_profile
        self.contract = json.loads((HERE / "DESKTOP_BASELINE_CONTRACT.json").read_text())
        self.requests = read_jsonl(HERE / "DESKTOP_REQUESTS.jsonl")
        self.switches = read_jsonl(HERE / "DESKTOP_SWITCHES.jsonl")
        self.models = {
            model_id: {
                **record,
                "path": Path(record["path"]),
            }
            for model_id, record in self.contract["models"].items()
        }
        self.events: JsonlWriter
        self.server_counter = 0
        self.current_server: ServerProcess | None = None
        self.current_pid_lock = threading.Lock()

    def next_server_index(self) -> int:
        index = self.server_counter
        self.server_counter += 1
        return index

    def current_pid(self) -> int | None:
        with self.current_pid_lock:
            return None if self.current_server is None else self.current_server.pid

    def set_server(self, server: ServerProcess | None) -> None:
        with self.current_pid_lock:
            self.current_server = server

    def preflight(self) -> dict[str, Any]:
        validate_inputs.validate(HERE)
        if not self.server.is_file() or not os.access(self.server, os.X_OK):
            raise RunError("llama-server is missing or not executable")
        if not self.cuda_lib_dir.is_dir():
            raise RunError("CUDA library directory is missing")
        if self.output.exists():
            raise RunError(f"output already exists: {self.output}")
        self.output.mkdir(parents=True)
        self.events = JsonlWriter(self.output / "events.jsonl")
        model_records: dict[str, Any] = {}
        for model_id, record in self.models.items():
            path = record["path"]
            if not path.is_file() or path.stat().st_size != record["bytes"]:
                raise RunError(f"{model_id}: model size mismatch")
            actual = digest_file(path)
            if actual != record["sha256"]:
                raise RunError(f"{model_id}: model digest mismatch")
            model_records[model_id] = {
                "bytes": path.stat().st_size,
                "path": str(path),
                "sha256": actual,
            }
        gpu = gpu_snapshot(self.gpu_index)
        device = self.contract["device"]
        hostname = socket.gethostname()
        if hostname != device["host"] \
                or gpu["gpu_name"] != device["gpu_name"] \
                or gpu["gpu_uuid"] != device["gpu_uuid"] \
                or gpu["gpu_memory_total_bytes"] != device["gpu_memory_total_bytes"]:
            raise RunError("desktop or GPU identity mismatch")
        foreign = [
            row for row in compute_processes()
            if row["pid"] != os.getpid() and row["used_memory_bytes"] > 128 * 1024 * 1024
        ]
        if foreign:
            raise RunError(f"foreign CUDA compute processes present: {foreign}")
        report = {
            "contract_sha256": digest_file(HERE / "DESKTOP_BASELINE_CONTRACT.json"),
            "gpu": gpu,
            "hostname": hostname,
            "input_manifest_sha256": digest_file(HERE / "INPUT_MANIFEST.json"),
            "models": model_records,
            "server_path": str(self.server),
            "server_sha256": digest_file(self.server),
            "serving_profile": self.serving_profile,
            "system_memory": system_memory(),
        }
        (self.output / "preflight.json").write_bytes(canonical(report))
        return report

    def cache_prepare(self, model_id: str, regime: str) -> dict[str, Any]:
        path = self.models[model_id]["path"]
        started_ns = time.monotonic_ns()
        if regime == "WARM_CACHE":
            result = cache_control.resident_pages(path)
            if result["resident_ppm"] < 950_000:
                raise RunError(
                    f"{model_id}: warm cache fell below 95 percent: "
                    f"{result['resident_ppm']} ppm"
                )
            result["method"] = "VERIFY_PREWARMED_NO_REFILL"
        elif regime == "COLD_NVME":
            result = cache_control.evict_file(path)
            if result["resident_ppm"] > 50_000:
                raise RunError(
                    f"{model_id}: cold cache exceeds 5 percent: "
                    f"{result['resident_ppm']} ppm"
                )
        else:
            raise RunError(f"unknown cache regime {regime}")
        record = {
            **result,
            "kind": "cache_prepared",
            "model_id": model_id,
            "regime": regime,
            "schema": "s39-cp0d-cache-event-v1",
            "started_ns": started_ns,
            "t_ns": time.monotonic_ns(),
        }
        self.events.write(record)
        return record

    def warm_both(self) -> list[dict[str, Any]]:
        output = []
        for model_id in sorted(self.models):
            record = cache_control.warm_file(self.models[model_id]["path"])
            if record["resident_ppm"] < 950_000:
                raise RunError(f"{model_id}: prewarm did not reach 95 percent")
            output.append({"model_id": model_id, **record})
        for model_id in sorted(self.models):
            probe = cache_control.resident_pages(self.models[model_id]["path"])
            if probe["resident_ppm"] < 950_000:
                raise RunError(f"{model_id}: pair prewarm does not coexist in RAM")
            output.append({"model_id": model_id, "pair_probe": True, **probe})
        return output

    def close(self) -> None:
        try:
            if self.current_server is not None:
                self.current_server.stop()
                self.set_server(None)
        finally:
            self.events.close()


def stream_completion(
        server: ServerProcess, row: dict[str, Any], stream_path: Path,
        timeout_s: float, on_first: Callable[[int], None]) -> dict[str, Any]:
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
        server.base_url + "/completion",
        data=canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final: dict[str, Any] | None = None
    tokens: list[int] = []
    with stream_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                raw_stream.flush()
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    continue
                value = json.loads(payload)
                if type(value) is not dict:
                    raise RunError("completion chunk is not an object")
                if "error" in value:
                    raise RunError(f"completion error: {value['error']}")
                chunk_tokens = value.get("tokens", [])
                if type(chunk_tokens) is not list \
                        or any(type(token) is not int for token in chunk_tokens):
                    raise RunError("completion tokens malformed")
                if chunk_tokens and not tokens:
                    on_first(time.monotonic_ns())
                tokens.extend(chunk_tokens)
                if value.get("stop", False):
                    final = value
    if final is None:
        raise RunError("completion has no final record")
    timings = final.get("timings")
    if type(timings) is not dict \
            or timings.get("prompt_n") != row["input_tokens"] \
            or timings.get("predicted_n") != row["output_tokens"]:
        raise RunError("completion token accounting mismatch")
    if len(tokens) != row["output_tokens"]:
        raise RunError(
            f"completion returned {len(tokens)} token IDs, expected {row['output_tokens']}"
        )
    return {
        "prompt_n": timings["prompt_n"],
        "predicted_n": timings["predicted_n"],
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_ms": timings.get("predicted_ms"),
        "tokens": tokens,
    }


def run_qualification(runner: Runner, model_id: str) -> dict[str, Any]:
    preflight = runner.preflight()
    runner.warm_both()
    server = ServerProcess(runner, model_id, runner.args.port, f"qualify-{model_id}")
    runner.set_server(server)
    before = system_memory()
    ready = server.start()
    rows = [row for row in runner.requests if row["model_id"] == model_id][:8]
    if len(rows) != 8:
        raise RunError("qualification cohort is not B8")
    barrier = threading.Barrier(9)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    lock = threading.Lock()

    def worker(row: dict[str, Any]) -> None:
        try:
            barrier.wait()
            started_ns = time.monotonic_ns()
            first_ns: list[int] = []
            result = stream_completion(
                server,
                row,
                runner.output / f"stream-{row['request_index']:03d}.raw",
                runner.args.request_timeout_s,
                first_ns.append,
            )
            if len(first_ns) != 1:
                raise RunError("qualification first-token count mismatch")
            with lock:
                results.append({
                    "completion_ns": time.monotonic_ns(),
                    "event_id": row["event_id"],
                    "first_token_ns": first_ns[0],
                    "request_index": row["request_index"],
                    "started_ns": started_ns,
                    **result,
                })
        except Exception as exc:
            with lock:
                errors.append(f"{row['event_id']}: {exc}")

    threads = [threading.Thread(target=worker, args=(row,)) for row in rows]
    for thread in threads:
        thread.start()
    cohort_start_ns = time.monotonic_ns()
    barrier.wait()
    for thread in threads:
        thread.join(runner.args.request_timeout_s + 10)
    if any(thread.is_alive() for thread in threads):
        raise RunError("qualification worker did not terminate")
    if errors or len(results) != 8:
        raise RunError(f"qualification failed: {errors}")
    cohort_end_ns = max(row["completion_ns"] for row in results)
    after = system_memory()
    unload = server.stop()
    runner.set_server(None)
    if after["system_swap_free_bytes"] < before["system_swap_free_bytes"]:
        raise RunError("qualification changed system swap")
    report = {
        "cohort_end_ns": cohort_end_ns,
        "cohort_start_ns": cohort_start_ns,
        "model_id": model_id,
        "preflight": preflight,
        "ready": ready,
        "request_count": 8,
        "results": sorted(results, key=lambda row: row["request_index"]),
        "schema": "s39-cp0d-qualification-v1",
        "status": "INDEPENDENT_B8_QUALIFICATION_PASS",
        "system_memory_after": after,
        "system_memory_before": before,
        "unload": unload,
    }
    (runner.output / "qualification.json").write_bytes(canonical(report))
    return report


def run_noncoresidency(runner: Runner) -> dict[str, Any]:
    preflight = runner.preflight()
    runner.warm_both()
    orders = [
        ["qwen3-14b-q4_k_m", "qwen3-8b-q8_0"],
        ["qwen3-8b-q8_0", "qwen3-14b-q4_k_m"],
    ]
    attempts: list[dict[str, Any]] = []
    for order_index, (first_id, second_id) in enumerate(orders):
        first = ServerProcess(
            runner, first_id, runner.args.port + order_index * 2,
            f"pair-{order_index}-first",
        )
        runner.set_server(first)
        first_ready = first.start()
        second = ServerProcess(
            runner, second_id, runner.args.port + order_index * 2 + 1,
            f"pair-{order_index}-second",
        )
        second_ready: dict[str, Any] | None = None
        second_error: str | None = None
        try:
            second_ready = second.start()
        except Exception as exc:
            second_error = str(exc)
        first_healthy = first.healthy()
        snapshot = gpu_snapshot(runner.gpu_index)
        second_eligible = (
            second_ready is not None
            and snapshot["gpu_memory_free_bytes"] >= 536_870_912
        )
        second.stop()
        first.stop()
        runner.set_server(None)
        if not first_healthy or second_eligible:
            raise RunError(
                f"non-co-residency failed for order {first_id},{second_id}"
            )
        attempts.append({
            "first_healthy_after_attempt": first_healthy,
            "first_model_id": first_id,
            "first_ready": first_ready,
            "gpu_after_attempt": snapshot,
            "second_eligible": second_eligible,
            "second_error": second_error,
            "second_model_id": second_id,
            "second_ready": second_ready,
        })
    report = {
        "attempts": attempts,
        "preflight": preflight,
        "schema": "s39-cp0d-non-coresidency-v1",
        "status": "PHYSICAL_NON_CORESIDENCY_PASS",
    }
    (runner.output / "non_coresidency.json").write_bytes(canonical(report))
    return report


def run_replay(runner: Runner, regime: str, repeat_index: int) -> dict[str, Any]:
    preflight = runner.preflight()
    if regime == "WARM_CACHE":
        warm_records = runner.warm_both()
    elif regime == "COLD_NVME":
        warm_records = []
    else:
        raise RunError("invalid replay regime")

    initial_model = runner.contract["replay"]["initial_model_id"]
    runner.cache_prepare(initial_model, regime)
    server = ServerProcess(runner, initial_model, runner.args.port, "initial")
    runner.set_server(server)
    initial_ready = server.start()

    sampler = PowerSampler(
        runner.output / "resource_samples.jsonl",
        runner.gpu_index,
        runner.current_pid,
    )
    sampler.start()
    time.sleep(0.3)

    pending = {model_id: deque() for model_id in runner.models}
    condition = threading.Condition()
    active: dict[int, threading.Thread] = {}
    results: list[dict[str, Any]] = []
    request_errors: list[str] = []
    arrival_index = 0
    arrival_done = False
    current_model = initial_model
    admission_open = True
    paid_start_ns = time.monotonic_ns()
    runner.events.write({
        "kind": "replay_start",
        "regime": regime,
        "repeat_index": repeat_index,
        "schema": "s39-cp0d-replay-event-v1",
        "t_ns": paid_start_ns,
    })

    def arrival_producer() -> None:
        nonlocal arrival_index, arrival_done
        for index, row in enumerate(runner.requests):
            target_ns = paid_start_ns + row["arrival_us"] * 1000
            while True:
                remaining = target_ns - time.monotonic_ns()
                if remaining <= 0:
                    break
                time.sleep(min(remaining / 1e9, 0.01))
            actual_ns = time.monotonic_ns()
            with condition:
                pending[row["model_id"]].append(row)
                arrival_index = index + 1
                runner.events.write({
                    "actual_t_ns": actual_ns,
                    "event_id": row["event_id"],
                    "kind": "request_arrival",
                    "model_id": row["model_id"],
                    "request_index": row["request_index"],
                    "schema": "s39-cp0d-request-event-v1",
                    "scheduled_t_ns": target_ns,
                })
                condition.notify_all()
        with condition:
            arrival_done = True
            condition.notify_all()

    def request_worker(row: dict[str, Any], dispatch_ns: int,
                       dispatch_model: str, dispatch_server: ServerProcess) -> None:
        first_ns: list[int] = []
        try:
            result = stream_completion(
                dispatch_server,
                row,
                runner.output / f"stream-{row['request_index']:03d}.raw",
                runner.args.request_timeout_s,
                first_ns.append,
            )
            if len(first_ns) != 1:
                raise RunError("first-token event count mismatch")
            completion_ns = time.monotonic_ns()
            record = {
                "completion_ns": completion_ns,
                "dispatch_model_id": dispatch_model,
                "dispatch_ns": dispatch_ns,
                "event_id": row["event_id"],
                "first_token_ns": first_ns[0],
                "model_id": row["model_id"],
                "request_index": row["request_index"],
                "schema": "s39-cp0d-request-result-v1",
                "scheduled_arrival_ns": paid_start_ns + row["arrival_us"] * 1000,
                "slo_us": row["slo_us"],
                **result,
            }
            with condition:
                results.append(record)
                runner.events.write({
                    **record,
                    "kind": "request_complete",
                    "schema": "s39-cp0d-request-event-v1",
                })
        except Exception as exc:
            with condition:
                request_errors.append(f"{row['event_id']}: {exc}")
                runner.events.write({
                    "error": str(exc),
                    "event_id": row["event_id"],
                    "kind": "request_error",
                    "request_index": row["request_index"],
                    "schema": "s39-cp0d-request-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
        finally:
            with condition:
                active.pop(row["request_index"], None)
                condition.notify_all()

    arrival_thread = threading.Thread(target=arrival_producer, name="arrival-producer")
    arrival_thread.start()
    switch_records: list[dict[str, Any]] = []
    switch_index = 0
    try:
        while True:
            switch_due = (
                switch_index < len(runner.switches)
                and time.monotonic_ns()
                >= paid_start_ns + runner.switches[switch_index]["t_us"] * 1000
            )
            if switch_due:
                switch = runner.switches[switch_index]
                with condition:
                    while arrival_index < len(runner.requests) \
                            and runner.requests[arrival_index]["arrival_us"] \
                            <= switch["t_us"]:
                        condition.wait(timeout=0.05)
                    admission_open = False
                    switch_started_ns = time.monotonic_ns()
                    runner.events.write({
                        "from_model_id": current_model,
                        "intent_index": switch_index,
                        "kind": "switch_started",
                        "schema": "s39-cp0d-switch-event-v1",
                        "scheduled_t_ns": paid_start_ns + switch["t_us"] * 1000,
                        "t_ns": switch_started_ns,
                        "to_model_id": switch["to_model_id"],
                    })
                    while active:
                        condition.wait(timeout=0.1)
                drain_ended_ns = time.monotonic_ns()
                unload = server.stop()
                runner.set_server(None)
                cache_record = runner.cache_prepare(switch["to_model_id"], regime)
                server = ServerProcess(
                    runner,
                    switch["to_model_id"],
                    runner.args.port,
                    f"switch-{switch_index:02d}",
                )
                runner.set_server(server)
                ready = server.start()
                current_model = switch["to_model_id"]
                published_ns = time.monotonic_ns()
                record = {
                    "cache": cache_record,
                    "drain_elapsed_ns": drain_ended_ns - switch_started_ns,
                    "drain_ended_ns": drain_ended_ns,
                    "from_model_id": switch["from_model_id"],
                    "intent_index": switch_index,
                    "load": ready,
                    "publication_gap_ns": (
                        published_ns - (paid_start_ns + switch["t_us"] * 1000)
                    ),
                    "published_ns": published_ns,
                    "scheduled_t_ns": paid_start_ns + switch["t_us"] * 1000,
                    "schema": "s39-cp0d-switch-result-v1",
                    "started_ns": switch_started_ns,
                    "to_model_id": current_model,
                    "unload": unload,
                }
                switch_records.append(record)
                runner.events.write({
                    **record,
                    "kind": "model_published",
                    "schema": "s39-cp0d-switch-event-v1",
                })
                switch_index += 1
                with condition:
                    admission_open = True
                    condition.notify_all()
                continue

            dispatched = False
            with condition:
                while admission_open and len(active) < 8 and pending[current_model]:
                    row = pending[current_model].popleft()
                    dispatch_ns = time.monotonic_ns()
                    runner.events.write({
                        "event_id": row["event_id"],
                        "kind": "request_dispatched",
                        "model_id": current_model,
                        "request_index": row["request_index"],
                        "schema": "s39-cp0d-request-event-v1",
                        "t_ns": dispatch_ns,
                    })
                    thread = threading.Thread(
                        target=request_worker,
                        args=(row, dispatch_ns, current_model, server),
                        name=f"request-{row['request_index']:03d}",
                    )
                    active[row["request_index"]] = thread
                    thread.start()
                    dispatched = True

                if request_errors:
                    raise RunError("; ".join(request_errors))
                all_pending = sum(len(queue) for queue in pending.values())
                if arrival_done and switch_index == len(runner.switches) \
                        and not active and all_pending == 0:
                    break
                if arrival_done and switch_index == len(runner.switches) \
                        and not active and not pending[current_model] and all_pending:
                    stranded = {
                        model_id: len(queue)
                        for model_id, queue in pending.items() if queue
                    }
                    raise RunError(f"final target strands requests: {stranded}")
                if not dispatched:
                    timeout = 0.02
                    if switch_index < len(runner.switches):
                        due_ns = (
                            paid_start_ns
                            + runner.switches[switch_index]["t_us"] * 1000
                        )
                        timeout = max(
                            0.001,
                            min(timeout, (due_ns - time.monotonic_ns()) / 1e9),
                        )
                    condition.wait(timeout=timeout)
        arrival_thread.join(timeout=5)
        if arrival_thread.is_alive():
            raise RunError("arrival producer did not terminate")
        if len(results) != 74 \
                or {row["request_index"] for row in results} != set(range(74)):
            raise RunError("request conservation failed")
        paid_end_ns = max(row["completion_ns"] for row in results)
        runner.events.write({
            "kind": "replay_end",
            "schema": "s39-cp0d-replay-event-v1",
            "t_ns": paid_end_ns,
        })
        time.sleep(0.3)
        sampler.stop()
        energy = sampler.integrate(paid_start_ns, paid_end_ns)
        system_after = system_memory()
        if system_after["system_swap_free_bytes"] \
                < preflight["system_memory"]["system_swap_free_bytes"]:
            raise RunError("replay changed system swap")
        report = {
            "energy": energy,
            "initial_ready": initial_ready,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "preflight": preflight,
            "regime": regime,
            "repeat_index": repeat_index,
            "request_results": sorted(results, key=lambda row: row["request_index"]),
            "schema": "s39-cp0d-replay-v1",
            "status": "RAW_DESKTOP_REPLAY_PASS_ANALYSIS_PENDING",
            "switch_results": switch_records,
            "system_memory_after": system_after,
            "warm_records": warm_records,
        }
        (runner.output / "replay.json").write_bytes(canonical(report))
        return report
    finally:
        if arrival_thread.is_alive():
            arrival_thread.join(timeout=2)
        if sampler.process is not None and sampler.process.poll() is None:
            try:
                sampler.stop()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=["qualify", "noncoresidency", "replay"],
    )
    parser.add_argument("--model-id", choices=[
        "qwen3-8b-q8_0", "qwen3-14b-q4_k_m",
    ])
    parser.add_argument("--regime", choices=["WARM_CACHE", "COLD_NVME"])
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18150)
    parser.add_argument("--server-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--serving-profile",
        choices=["fixed_full_cuda", "stock_default"],
        default="fixed_full_cuda",
    )
    args = parser.parse_args()
    if args.phase == "qualify" and args.model_id is None:
        parser.error("--model-id is required for qualify")
    if args.phase == "replay" \
            and (args.regime is None or args.repeat_index is None):
        parser.error("--regime and --repeat-index are required for replay")
    return args


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    try:
        if args.phase == "qualify":
            run_qualification(runner, args.model_id)
        elif args.phase == "noncoresidency":
            run_noncoresidency(runner)
        else:
            run_replay(runner, args.regime, args.repeat_index)
        return 0
    finally:
        if hasattr(runner, "events"):
            runner.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RunError, cache_control.CacheError, validate_inputs.ValidationError) as exc:
        print(f"CP0D_RUN_ERROR: {exc}", flush=True)
        raise SystemExit(2)
