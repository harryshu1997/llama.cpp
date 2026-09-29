#!/usr/bin/env python3
"""Measure a bounded Qwen/Gemma dual-residency GPU configuration."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import Any
import urllib.error
import urllib.request


SCHEMA = "s42-dynamic-gpu-residency-capacity-v1"
CONFIRMATION = "PROFILE_DYNAMIC_GPU_RESIDENCY"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4060 Ti"
EXPECTED_GPU_UUID = "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08"
QWEN_BYTES = 29_543_423_360
QWEN_SHA256 = (
    "d89e9e823744222e595e0b3c8fd5436c"
    "e5d3a6a446fa42492ebce6064dfa9718"
)
GEMMA_BYTES = 23_832_065_056
GEMMA_SHA256 = (
    "ed76f2183d2d1d65091986033023e6c7"
    "8d27f6276c1b0c5826cc92acf73538cf"
)
MIB = 1024 * 1024
TRACE_SHA256 = (
    "b20a9ba66ee3558d835a0e19ed3cfa4"
    "c31a4a9e8b4f9c085b29a14f80250a0ff"
)
TRACE_SCHEMA = "s41-gemma-qwen-request-semantic-source-v1"


class CapacityProbeError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityProbeError(message)


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
    with path.open("rb") as source:
        while block := source.read(8 * MIB):
            digest.update(block)
    return digest.hexdigest()


def command_output(argv: list[str]) -> str:
    completed = subprocess.run(
        argv,
        capture_output=True,
        check=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def gpu_snapshot() -> dict[str, object]:
    output = command_output([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,memory.used,memory.free,"
        "utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    rows = output.splitlines()
    require(len(rows) == 1, "capacity probe requires exactly one visible GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    require(len(fields) == 7, "unexpected nvidia-smi GPU row")
    return {
        "memory_free_bytes": int(fields[4]) * MIB,
        "memory_total_bytes": int(fields[2]) * MIB,
        "memory_used_bytes": int(fields[3]) * MIB,
        "name": fields[0],
        "power_mw": int(round(float(fields[6]) * 1000)),
        "utilization_pct": int(fields[5]),
        "uuid": fields[1],
    }


def compute_apps() -> list[dict[str, object]]:
    output = command_output([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ])
    if not output:
        return []
    result = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(len(fields) == 3, "unexpected nvidia-smi process row")
        result.append({
            "pid": int(fields[0]),
            "process_name": fields[1],
            "used_gpu_memory_bytes": int(fields[2]) * MIB,
        })
    return result


def process_memory(pid: int) -> dict[str, int]:
    wanted = {
        "VmRSS": "rss_bytes",
        "VmSwap": "swap_bytes",
    }
    result: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key not in wanted:
            continue
        fields = value.strip().split()
        require(len(fields) == 2 and fields[1] == "kB", "process status")
        result[wanted[key]] = int(fields[0]) * 1024
    require(set(result) == set(wanted.values()), "process memory fields")
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
        if key not in wanted:
            continue
        fields = value.strip().split()
        require(len(fields) == 2 and fields[1] == "kB", "system memory")
        result[wanted[key]] = int(fields[0]) * 1024
    require(set(result) == set(wanted.values()), "system memory fields")
    return result


def rapl_snapshot() -> dict[str, int]:
    root = Path("/sys/class/powercap/intel-rapl:0")
    require(root.joinpath("name").read_text().strip() == "package-0", "RAPL")
    return {
        "energy_uj": int(root.joinpath("energy_uj").read_text().strip()),
        "max_energy_range_uj": int(
            root.joinpath("max_energy_range_uj").read_text().strip()
        ),
        "monotonic_ns": time.monotonic_ns(),
    }


def rapl_delta_uj(before: dict[str, int], after: dict[str, int]) -> int:
    require(
        before["max_energy_range_uj"] == after["max_energy_range_uj"]
        and after["monotonic_ns"] >= before["monotonic_ns"],
        "RAPL interval",
    )
    delta = after["energy_uj"] - before["energy_uj"]
    if delta < 0:
        delta += before["max_energy_range_uj"]
    require(0 <= delta < before["max_energy_range_uj"], "RAPL delta")
    return delta


def integrate_gpu_energy_uj(
    rows: list[dict[str, object]],
    start_ns: int,
    finish_ns: int,
) -> int:
    require(rows and finish_ns > start_ns, "GPU energy interval")
    ordered = sorted(rows, key=lambda row: row["monotonic_ns"])
    total_mw_ns = 0
    cursor_ns = start_ns
    power_mw = ordered[0]["power_mw"]
    for row in ordered:
        sample_ns = min(finish_ns, max(start_ns, row["monotonic_ns"]))
        if sample_ns > cursor_ns:
            total_mw_ns += power_mw * (sample_ns - cursor_ns)
            cursor_ns = sample_ns
        power_mw = row["power_mw"]
        if cursor_ns >= finish_ns:
            break
    if cursor_ns < finish_ns:
        total_mw_ns += power_mw * (finish_ns - cursor_ns)
    return (total_mw_ns + 999_999) // 1_000_000


def read_trace(path: Path) -> list[dict[str, Any]]:
    require(path.is_file() and sha256(path) == TRACE_SHA256, "trace identity")
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        value = json.loads(line)
        require(type(value) is dict, "trace row")
        rows.append(value)
    require(
        len(rows) == 74
        and all(row.get("schema") == TRACE_SCHEMA for row in rows),
        "trace geometry",
    )
    return rows


def select_smoke_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_index = {row.get("request_index"): row for row in rows}
    qwen = by_index.get(52)
    gemma = by_index.get(50)
    require(
        qwen is not None
        and qwen.get("model_id") == "gemma-4-12b-it-q8_0"
        and qwen.get("input_tokens") == 16
        and qwen.get("output_tokens") == 9,
        "Qwen smoke row",
    )
    require(
        gemma is not None
        and gemma.get("model_id") == "qwen3-14b-q4_k_m"
        and gemma.get("input_tokens") == 271
        and gemma.get("output_tokens") == 41,
        "Gemma smoke row",
    )
    return qwen, gemma


def completion(
    port: int,
    row: dict[str, Any],
    raw_path: Path,
    *,
    output_tokens: int | None = None,
) -> dict[str, object]:
    predicted_tokens = (
        row["output_tokens"] if output_tokens is None else output_tokens
    )
    body = {
        "cache_prompt": False,
        "ignore_eos": True,
        "n_predict": predicted_tokens,
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
    started_ns = time.monotonic_ns()
    first_token_ns = None
    final = None
    tokens: list[int] = []
    with raw_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                value = json.loads(payload)
                require(type(value) is dict and "error" not in value, "chunk")
                chunk = value.get("tokens", [])
                require(
                    type(chunk) is list
                    and all(type(token) is int for token in chunk),
                    "completion tokens",
                )
                if chunk and first_token_ns is None:
                    first_token_ns = time.monotonic_ns()
                tokens.extend(chunk)
                if value.get("stop") is True:
                    final = value
    completed_ns = time.monotonic_ns()
    require(final is not None and first_token_ns is not None, "completion final")
    timings = final.get("timings", {})
    require(
        timings.get("prompt_n") == row["input_tokens"]
        and timings.get("predicted_n") == predicted_tokens
        and len(tokens) == predicted_tokens,
        "completion work",
    )
    return {
        "completed_monotonic_ns": completed_ns,
        "first_token_us": (first_token_ns - started_ns) // 1000,
        "output_tokens": predicted_tokens,
        "predicted_ms": timings.get("predicted_ms"),
        "prompt_ms": timings.get("prompt_ms"),
        "request_index": row["request_index"],
        "service_us": (completed_ns - started_ns) // 1000,
        "started_monotonic_ns": started_ns,
        "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
    }


class GpuSampler:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []
        self.error: BaseException | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                row = gpu_snapshot()
                row["monotonic_ns"] = time.monotonic_ns()
                self.rows.append(row)
                self.stop_event.wait(0.1)
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        self.thread.start()
        deadline = time.monotonic() + 30
        while (
            not self.rows
            and self.error is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        require(
            self.error is None and self.rows,
            "GPU sampler did not start",
        )

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        require(not self.thread.is_alive(), "GPU sampler did not stop")
        if self.error is not None:
            raise self.error
        require(self.rows, "GPU sampler is empty")


def run_service_smoke(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    qwen: RunningServer,
    gemma: RunningServer,
) -> dict[str, object]:
    qwen_row, gemma_row = select_smoke_rows(rows)
    completion(
        args.qwen_port,
        qwen_row,
        args.output / "qwen-warm.raw",
        output_tokens=2,
    )
    completion(
        args.gemma_port,
        gemma_row,
        args.output / "gemma-warm.raw",
        output_tokens=2,
    )
    sampler = GpuSampler()
    sampler.start()
    rapl_before = rapl_snapshot()
    wall_started_ns = time.monotonic_ns()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            qwen_future = pool.submit(
                completion,
                args.qwen_port,
                qwen_row,
                args.output / "qwen-smoke.raw",
            )
            gemma_future = pool.submit(
                completion,
                args.gemma_port,
                gemma_row,
                args.output / "gemma-smoke.raw",
            )
            results = {
                "gemma": gemma_future.result(),
                "qwen": qwen_future.result(),
            }
        wall_completed_ns = time.monotonic_ns()
        rapl_after = rapl_snapshot()
    finally:
        sampler.stop()
    process_memory_rows = {
        "gemma": process_memory(gemma.pid),
        "qwen": process_memory(qwen.pid),
    }
    reserve_bytes = args.reserve_mib * MIB
    minimum_free_bytes = min(
        row["memory_free_bytes"] for row in sampler.rows
    )
    status = "EXECUTION_SMOKE_PASS"
    if any(row["swap_bytes"] for row in process_memory_rows.values()):
        status = "REJECTED_PROCESS_SWAP"
    elif minimum_free_bytes < reserve_bytes:
        status = "REJECTED_GPU_RESERVE"
    return {
        "energy": {
            "accounting_scope": "CPU_PACKAGE_PLUS_GPU_BOARD",
            "claim": "DIAGNOSTIC_SINGLE_PROCESS_NOT_PROFILE",
            "cpu_package_uj": rapl_delta_uj(rapl_before, rapl_after),
            "gpu_board_uj": integrate_gpu_energy_uj(
                sampler.rows, wall_started_ns, wall_completed_ns
            ),
        },
        "gpu": {
            "maximum_used_bytes": max(
                row["memory_used_bytes"] for row in sampler.rows
            ),
            "minimum_free_bytes": minimum_free_bytes,
            "samples": len(sampler.rows),
        },
        "process_memory": process_memory_rows,
        "requests": results,
        "status": status,
        "wall_service_us": (wall_completed_ns - wall_started_ns) // 1000,
    }


def parse_allocations(log: str, expected_layers: int) -> dict[str, object]:
    offloads = re.findall(r"offloaded ([0-9]+)/([0-9]+) layers to GPU", log)
    if expected_layers > 0:
        require(
            any(int(loaded) == expected_layers for loaded, _ in offloads),
            "GPU layer placement mismatch",
        )
        total_layers = max(int(total) for _, total in offloads)
    else:
        require(
            not offloads or all(int(loaded) == 0 for loaded, _ in offloads),
            "GPU layer placement mismatch",
        )
        total_layers = (
            None if not offloads else max(int(total) for _, total in offloads)
        )
    patterns = {
        "compute_buffer_mib": r"CUDA0 compute buffer size =\s+([0-9.]+) MiB",
        "gpu_kv_buffer_mib": r"CUDA0 KV buffer size =\s+([0-9.]+) MiB",
        "model_buffer_mib": r"CUDA0 model buffer size =\s+([0-9.]+) MiB",
    }
    result: dict[str, object] = {
        "offloaded_layers": expected_layers,
        "total_layers": total_layers,
    }
    for key, pattern in patterns.items():
        values = [float(value) for value in re.findall(pattern, log)]
        result[key] = sum(values)
    return result


def server_command(
    server: Path,
    model: Path,
    *,
    alias: str,
    layers: int,
    ctx_size: int,
    parallel: int,
    batch_size: int,
    port: int,
) -> list[str]:
    command = [
        str(server),
        "--model", str(model),
        "--alias", alias,
        "--fit", "off",
        "--ctx-size", str(ctx_size),
        "--parallel", str(parallel),
        "--batch-size", str(batch_size),
        "--ubatch-size", "512",
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--n-gpu-layers", str(layers),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-timestamps",
        "--verbose",
    ]
    if layers > 0:
        command.extend([
            "--flash-attn", "on",
            "--split-mode", "none",
            "--main-gpu", "0",
            "--device", "CUDA0",
        ])
    else:
        command.extend(["--device", "none"])
    return command


@dataclass
class RunningServer:
    name: str
    process: subprocess.Popen[bytes]
    stdout_path: Path
    stderr_path: Path
    stdout_stream: Any
    stderr_stream: Any

    @property
    def pid(self) -> int:
        return self.process.pid

    def stop(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=30)
        self.stdout_stream.close()
        self.stderr_stream.close()


def start_server(
    name: str,
    command: list[str],
    environment: dict[str, str],
    output: Path,
    port: int,
    layers: int,
) -> tuple[RunningServer, dict[str, object]]:
    stdout_path = output / f"{name}.stdout.log"
    stderr_path = output / f"{name}.stderr.log"
    stdout_stream = stdout_path.open("xb")
    stderr_stream = stderr_path.open("xb")
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=True,
        stdout=stdout_stream,
        stderr=stderr_stream,
    )
    running = RunningServer(
        name,
        process,
        stdout_path,
        stderr_path,
        stdout_stream,
        stderr_stream,
    )
    deadline = time.monotonic() + 360
    try:
        while time.monotonic() < deadline:
            require(process.poll() is None, f"{name} exited during load")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=1
                ) as response:
                    health = json.loads(response.read())
                if health.get("status") == "ok":
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise CapacityProbeError(f"{name} readiness timeout")
        stderr_stream.flush()
        log = stderr_path.read_text(encoding="utf-8", errors="replace")
        allocations = parse_allocations(log, layers)
        ready_ns = time.monotonic_ns()
        return running, {
            "allocations": allocations,
            "command": command,
            "load_us": (ready_ns - started_ns) // 1000,
            "pid": process.pid,
            "process_memory": process_memory(process.pid),
            "ready_monotonic_ns": ready_ns,
        }
    except BaseException:
        running.stop()
        raise


def capacity_status(
    stages: list[dict[str, object]], reserve_bytes: int
) -> str:
    if any(
        stage["process_memory"]["swap_bytes"] != 0
        for stage in stages
    ):
        return "REJECTED_PROCESS_SWAP"
    if stages[-1]["gpu"]["memory_free_bytes"] < reserve_bytes:
        return "REJECTED_GPU_RESERVE"
    return "CAPACITY_PASS"


def validate_file(path: Path, size: int, digest: str, name: str) -> None:
    require(path.is_file(), f"missing {name}")
    require(path.stat().st_size == size, f"{name} size mismatch")
    require(sha256(path) == digest, f"{name} SHA-256 mismatch")


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    require(args.confirm == CONFIRMATION, "confirmation")
    require(args.output.is_absolute() and not args.output.exists(), "output")
    require(args.qwen_layers > 0 and args.gemma_layers >= 0, "GPU layers")
    require(args.reserve_mib >= 0, "GPU reserve")
    validate_file(args.qwen_model, QWEN_BYTES, QWEN_SHA256, "Qwen model")
    validate_file(args.gemma_model, GEMMA_BYTES, GEMMA_SHA256, "Gemma model")
    trace_rows = None if args.trace is None else read_trace(args.trace)
    require(args.server.is_file(), "missing server")
    require(args.cuda_lib_dir.is_dir(), "missing CUDA library directory")
    require(not compute_apps(), "target GPU already has a compute process")
    baseline_gpu = gpu_snapshot()
    require(
        baseline_gpu["name"] == EXPECTED_GPU_NAME
        and baseline_gpu["uuid"] == EXPECTED_GPU_UUID,
        "target GPU identity",
    )

    args.output.mkdir(parents=True)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = (
        f"{args.cuda_lib_dir}:{args.server.parent}:"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    for name in tuple(environment):
        if name.startswith("S41_SERVER_FFN_") or name.startswith(
            "LLAMA_FFN_SPLIT_"
        ):
            del environment[name]

    running: list[RunningServer] = []
    stages: list[dict[str, object]] = []
    service_smoke = None
    try:
        qwen, qwen_stage = start_server(
            "qwen",
            server_command(
                args.server,
                args.qwen_model,
                alias="qwen3-14b-f16-proxy",
                layers=args.qwen_layers,
                ctx_size=24576,
                parallel=4,
                batch_size=2048,
                port=args.qwen_port,
            ),
            environment,
            args.output,
            args.qwen_port,
            args.qwen_layers,
        )
        running.append(qwen)
        qwen_stage["gpu"] = gpu_snapshot()
        qwen_stage["system_memory"] = system_memory()
        stages.append(qwen_stage)

        gemma, gemma_stage = start_server(
            "gemma",
            server_command(
                args.server,
                args.gemma_model,
                alias="gemma4-12b-f16-proxy",
                layers=args.gemma_layers,
                ctx_size=32768,
                parallel=8,
                batch_size=4096,
                port=args.gemma_port,
            ),
            environment,
            args.output,
            args.gemma_port,
            args.gemma_layers,
        )
        running.append(gemma)
        for stage in stages:
            stage["process_memory"] = process_memory(stage["pid"])
        gemma_stage["process_memory"] = process_memory(gemma.pid)
        gemma_stage["gpu"] = gpu_snapshot()
        gemma_stage["system_memory"] = system_memory()
        stages.append(gemma_stage)
        status = capacity_status(stages, args.reserve_mib * MIB)
        if trace_rows is not None and status == "CAPACITY_PASS":
            service_smoke = run_service_smoke(
                args, trace_rows, qwen, gemma
            )
            for stage in stages:
                stage["process_memory"] = process_memory(stage["pid"])
            status = service_smoke["status"]
    finally:
        for process in reversed(running):
            process.stop()

    final_gpu = gpu_snapshot()
    cleanup_apps = compute_apps()
    cleanup_ok = (
        not cleanup_apps
        and final_gpu["memory_used_bytes"]
            <= baseline_gpu["memory_used_bytes"] + 64 * MIB
    )
    if not cleanup_ok:
        status = "REJECTED_CLEANUP"
    inputs = {
        "gemma_model": {
            "path": str(args.gemma_model),
            "sha256": GEMMA_SHA256,
            "size_bytes": GEMMA_BYTES,
        },
        "qwen_model": {
            "path": str(args.qwen_model),
            "sha256": QWEN_SHA256,
            "size_bytes": QWEN_BYTES,
        },
        "probe_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "size_bytes": Path(__file__).stat().st_size,
        },
        "server": {
            "path": str(args.server),
            "sha256": sha256(args.server),
            "size_bytes": args.server.stat().st_size,
        },
    }
    if args.trace is not None:
        inputs["trace"] = {
            "path": str(args.trace),
            "sha256": TRACE_SHA256,
            "size_bytes": args.trace.stat().st_size,
        }
    result: dict[str, object] = {
        "baseline_gpu": baseline_gpu,
        "cleanup": {
            "compute_apps": cleanup_apps,
            "gpu": final_gpu,
            "passed": cleanup_ok,
        },
        "configuration": {
            "gemma_gpu_layers": args.gemma_layers,
            "gpu_reserve_bytes": args.reserve_mib * MIB,
            "qwen_gpu_layers": args.qwen_layers,
        },
        "inputs": inputs,
        "measurement_scope": (
            "CAPACITY_ONLY_NO_ENERGY_OR_SERVICE_CLAIM"
            if service_smoke is None
            else "EXECUTION_SMOKE_NO_ENERGY_OR_PROFILE_CLAIM"
        ),
        "schema": SCHEMA,
        "service_smoke": service_smoke,
        "stages": stages,
        "status": status,
    }
    result["record_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    (args.output / "RESULT.json").write_bytes(canonical(result))
    hashes = [("RESULT.json", sha256(args.output / "RESULT.json"))]
    hashes.extend(
        (path.name, sha256(path))
        for path in sorted(args.output.iterdir())
        if path.suffix in {".log", ".raw"}
    )
    (args.output / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes),
        encoding="ascii",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--gemma-model", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--qwen-layers", type=int, required=True)
    parser.add_argument("--gemma-layers", type=int, required=True)
    parser.add_argument("--qwen-port", type=int, default=18981)
    parser.add_argument("--gemma-port", type=int, default=18982)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    try:
        result = run_probe(args)
    except (CapacityProbeError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(args.output),
        "record_sha256": result["record_sha256"],
        "status": result["status"],
    }, sort_keys=True))
    return 0 if result["status"] in {
        "CAPACITY_PASS", "EXECUTION_SMOKE_PASS"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
