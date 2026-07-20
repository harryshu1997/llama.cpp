#!/usr/bin/env python3
"""Run the real S18 P0/P4 two-phone R1 mixed-workload acquisition."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENERGY = HERE.parent / "s14_mixed_streaming_scheduler/energy"
TYPED = HERE.parent / "s15_persistent_typed_gate"
S16 = HERE.parent / "s16_mixed_persistent_energy"
for path in (ENERGY, TYPED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cp_d_priority as CPD  # noqa: E402
import cp_e_live_priority as CPE  # noqa: E402
import physical_mux as MUX  # noqa: E402
from experiment_contract import (  # noqa: E402
    BATCHES_PER_ROUND, CONTROL_BATCH, FULL_ROUNDS, FULL_SCHEDULE, N_GEN,
    OP12_EXCHANGE_CREDIT, PHONE_BATCH, SCREEN_ROUNDS, SCREEN_SCHEDULE,
    ContractError, canonical, digest_file, integrate_power, parse_power_jsonl,
    strict_object, summarize, validate_result_record,
)


SCHEMA = "s18-two-phone-r1-mixed-v1"
GPU_INDEX = 0
GPU_UUID = CPD.SELECTED_GPU_UUID
SECOND_GPU_UUID = CPD.SECOND_GPU_UUID
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
HOST_BIN = ROOT / "build-cuda/bin/llama-layersplit"
HOST_LIB = HOST_BIN.parent
ANDROID_BIN = ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit"
CONTEXT = 16
MAX_PREFILL = 8
COHORT = HERE.parent / "s15_burst_cohort/cohort.json"
INPUT_MANIFEST = HERE.parent / "s15_burst_cohort/input_manifest.json"
REFERENCE_REPORT = HERE.parent / "s15_persistent_typed_gate/results/report.json"
POWER_PERIOD_MS = 100


@dataclass(frozen=True)
class LaneConfig:
    name: str
    serial: str
    port: int
    layer_end: int
    remote: str
    shard: str
    shard_sha256: str
    mbuf: int
    slo_us: int


OP15 = LaneConfig(
    name="op15",
    serial="3C15AU002CL00000",
    port=5981,
    layer_end=8,
    remote="/data/local/tmp/ls-s15-persistent-typed",
    shard="/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf",
    shard_sha256="a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8",
    mbuf=4192,
    slo_us=5_000_000,
)
OP12 = LaneConfig(
    name="op12",
    serial="5ae7a43d",
    port=5982,
    layer_end=6,
    remote="/data/local/tmp/ls-s16-op12",
    shard="/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf",
    shard_sha256="d507b7bb453242dff12ba1ce0add53189755b8a9a2b960743ead1578d8f1a6b5",
    mbuf=3336,
    slo_us=12_000_000,
)


class ExperimentError(RuntimeError):
    pass


def checked(argv: list[str], timeout: int = 30) -> bytes:
    result = subprocess.run(argv, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise ExperimentError(
            f"command failed rc={result.returncode}: {' '.join(argv)}: "
            f"{result.stderr.decode('utf-8', errors='replace')[-1000:]}"
        )
    return result.stdout


def adb(lane: LaneConfig, *args: str, timeout: int = 30,
        check: bool = True) -> bytes:
    result = subprocess.run(
        ["adb", "-s", lane.serial, *args], capture_output=True, timeout=timeout,
    )
    if check and result.returncode != 0:
        raise ExperimentError(
            result.stderr.decode("utf-8", errors="replace")[-1000:])
    return result.stdout


def parse_after(line: bytes, prefix: bytes, label: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ExperimentError(f"missing {label} prefix")
    return strict_object(line[len(prefix):], label, False)


def load_frozen_input() -> tuple[str, list[int], dict[str, str]]:
    manifest = strict_object(INPUT_MANIFEST.read_bytes(), "input manifest")
    report = strict_object(REFERENCE_REPORT.read_bytes(), "reference report", False)
    prompt = manifest.get("prompt_text")
    reference = report.get("reference_tokens")
    if type(prompt) is not str or not prompt or type(reference) is not list \
            or len(reference) != N_GEN or any(type(value) is not int for value in reference):
        raise ExperimentError("frozen prompt or token reference is invalid")
    return prompt, reference, {
        "cohort": digest_file(COHORT),
        "input_manifest": digest_file(INPUT_MANIFEST),
        "reference_report": digest_file(REFERENCE_REPORT),
    }


def validate_server_placement(value: dict[str, Any], pid: int) -> None:
    if value.get("schema") != "layersplit-scheduled-placement-v2" \
            or value.get("role") != "monodriver" \
            or value.get("mode") != "monodriver" \
            or value.get("layer_start") != 0 or value.get("layer_end") != 48 \
            or value.get("n_layer") != 48 or value.get("pid") != pid \
            or value.get("run_rc") != 0 \
            or value.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or type(value.get("compute_nodes")) is not int or value["compute_nodes"] <= 0:
        raise ExperimentError("server placement identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ExperimentError("server placement has no operation tally")
    for op_name, buffers in by_op.items():
        if type(buffers) is not dict or not buffers:
            raise ExperimentError("server placement operation tally is malformed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ExperimentError("server placement node count is invalid")
            if backend != "CUDA0" and not (
                    op_name == "GET_ROWS" and backend == "CUDA_Host"):
                raise ExperimentError(f"undeclared server fallback {op_name}@{backend}")


def validate_phone_session(value: dict[str, Any], lane: LaneConfig,
                           launch_id: int, session_end: str) -> None:
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != launch_id \
            or value.get("session_end") != session_end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 \
            or value.get("layer_end") != lane.layer_end \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (session_end == "DETACH"):
        raise ExperimentError(f"{lane.name} phone session identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ExperimentError(f"{lane.name} phone session has no operation tally")
    for op_name, buffers in by_op.items():
        if type(buffers) is not dict or not buffers:
            raise ExperimentError(f"{lane.name} phone placement is malformed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ExperimentError(f"{lane.name} phone node count is invalid")
            if backend != "HTP0" and not (
                    op_name == "GET_ROWS" and backend == "CPU"):
                raise ExperimentError(
                    f"undeclared {lane.name} fallback {op_name}@{backend}")


class PowerSampler:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.memory_mib: list[int] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi", "-i", str(GPU_INDEX),
                "--query-gpu=power.draw,power.limit,utilization.gpu,pstate,memory.used",
                "--format=csv,noheader,nounits", "-lms", str(POWER_PERIOD_MS),
            ],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        previous = -1
        try:
            for line in self.process.stdout:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 5:
                    raise ExperimentError("nvidia-smi emitted a malformed row")
                timestamp = time.time_ns() // 1000
                if timestamp <= previous:
                    timestamp = previous + 1
                previous = timestamp
                self.rows.append({
                    "t_us": timestamp,
                    "power_mw": int(round(float(parts[0]) * 1000)),
                    "power_limit_mw": int(round(float(parts[1]) * 1000)),
                    "util_milli_pct": int(round(float(parts[2]) * 1000)),
                    "pstate": parts[3],
                })
                self.memory_mib.append(int(parts[4]))
        except (ExperimentError, ValueError) as exc:
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
                raise ExperimentError("power sampler reader did not terminate")
        if self.error is not None:
            raise ExperimentError(self.error)


class ServerLane:
    def __init__(self, row_dir: Path, prompt: str, reference: list[int],
                 rounds: int) -> None:
        self.row_dir = row_dir
        self.prompt = prompt
        self.reference = reference
        self.rounds = rounds
        self.process: MUX.MonitoredProcess | None = None
        self.results: list[dict[str, Any]] = []
        self.placements: list[dict[str, Any]] = []
        self.windows_us: list[list[int]] = []

    def start(self) -> None:
        command = (
            str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
            "--mode", "monodriver", "-n", str(N_GEN),
            "--driver-batch", str(CONTROL_BATCH),
            "--driver-context", str(CONTEXT),
            "--driver-max-prefill", str(MAX_PREFILL), "--persistent-jsonl",
        )
        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "LD_LIBRARY_PATH": str(HOST_LIB),
            "LAYERSPLIT_PLACEMENT_CERT": "1",
        })
        environment.pop("LLAMA_LAYER_START", None)
        environment.pop("LLAMA_LAYER_END", None)
        self.process = MUX.MonitoredProcess(command, environment)
        line = self.process.wait_line("stderr", MUX.HOST_READY_PREFIX, 360)
        ready = strict_object(line[len(MUX.HOST_READY_PREFIX):], "server readiness")
        if ready.get("schema") != "layersplit-persistent-driver-v1" \
                or ready.get("batch_size") != CONTROL_BATCH \
                or ready.get("max_n_gen") != N_GEN:
            raise ExperimentError("server readiness failed")

    def exchange(self, launch_id: int) -> None:
        if self.process is None:
            raise ExperimentError("server exchange before readiness")
        exchanges = self.rounds * BATCHES_PER_ROUND
        session_end = "STOP" if launch_id == exchanges else "DETACH"
        command = canonical({
            "schema": "layersplit-persistent-command-v1",
            "launch_id": launch_id,
            "prompt": self.prompt,
            "n_gen": N_GEN,
            "request_count": CONTROL_BATCH,
            "session_end": session_end,
        })
        start = self.process.snapshot()
        start_us = time.time_ns() // 1000
        self.process.send(command)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            stdout = [line for line in self.process.lines_since("stdout", start[0]) if line.strip()]
            stderr = self.process.lines_since("stderr", start[1])
            markers = [line for line in stderr if line.startswith(MUX.HOST_END_PREFIX)]
            placements = [line for line in stderr if line.startswith(b"PLACEMENTCERT ")]
            if len(stdout) == len(markers) == len(placements) == 1:
                end_us = time.time_ns() // 1000
                result = strict_object(stdout[0], "server result")
                validate_result_record(
                    result, self.reference, CONTROL_BATCH, launch_id, session_end)
                placement = parse_after(
                    placements[0], b"PLACEMENTCERT ", "server placement")
                validate_server_placement(placement, result["host_pid"])
                marker = parse_after(
                    markers[0], MUX.HOST_END_PREFIX, "server marker")
                if marker != {"launch_id": launch_id}:
                    raise ExperimentError("server exchange marker failed")
                self.results.append(result)
                self.placements.append(placement)
                self.windows_us.append([start_us, end_us])
                if session_end == "STOP" and self.process.wait(20) != 0:
                    raise ExperimentError("server process did not stop cleanly")
                return
            if len(stdout) > 1 or len(markers) > 1 or len(placements) > 1:
                raise ExperimentError("duplicate server exchange evidence")
            if self.process.error is not None:
                raise ExperimentError(self.process.error)
            if self.process.process.poll() is not None and not stdout:
                raise ExperimentError("server process exited without a result")
            time.sleep(0.01)
        raise ExperimentError("server exchange timed out")

    def validate(self) -> None:
        if len(self.results) != self.rounds * BATCHES_PER_ROUND \
                or len({item["host_pid"] for item in self.results}) != 1:
            raise ExperimentError("server persistence failed")

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.persist(self.row_dir, "control")


class PhoneLane:
    def __init__(self, config: LaneConfig, row_dir: Path, prompt: str,
                 reference: list[int], rounds: int) -> None:
        self.config = config
        self.row_dir = row_dir
        self.prompt = prompt
        self.reference = reference
        self.rounds = rounds
        self.phone: MUX.MonitoredProcess | None = None
        self.host: MUX.MonitoredProcess | None = None
        self.results: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self.windows_us: list[list[int]] = []

    def start(self) -> None:
        lane = self.config
        adb(lane, "shell", "pkill -9 llama-layersplit >/dev/null 2>&1 || true",
            check=False)
        adb(lane, "forward", "--remove", f"tcp:{lane.port}", check=False)
        adb(lane, "forward", f"tcp:{lane.port}", f"tcp:{lane.port}")
        phone_shell = (
            f"cd {lane.remote} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
            f"GGML_HEXAGON_MBUF={lane.mbuf} LLAMA_LAYER_END={lane.layer_end} "
            "LAYERSPLIT_PLACEMENT_CERT=1 "
            f"./llama-layersplit -m {lane.shard} --devices HTP0 -ngl 99 "
            f"--mode stagenet --port {lane.port} -n {N_GEN} "
            f"--driver-batch {PHONE_BATCH} --driver-context {CONTEXT} "
            f"--driver-max-prefill {MAX_PREFILL}"
        )
        phone_command = ("adb", "-s", lane.serial, "shell", phone_shell)
        host_command = (
            str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
            "--mode", "pipedriver", "--host", "127.0.0.1",
            "--port", str(lane.port), "-n", str(N_GEN),
            "--driver-batch", str(PHONE_BATCH),
            "--driver-context", str(CONTEXT),
            "--driver-max-prefill", str(MAX_PREFILL), "--persistent-jsonl",
        )
        host_env = dict(os.environ)
        host_env.update({
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "LD_LIBRARY_PATH": str(HOST_LIB),
            "LLAMA_LAYER_START": str(lane.layer_end),
            "LAYERSPLIT_PLACEMENT_CERT": "1",
        })
        self.phone = MUX.MonitoredProcess(phone_command, None)
        self.phone.wait_line("stderr", MUX.PHONE_READY_PREFIX, 300)
        self.host = MUX.MonitoredProcess(host_command, host_env)
        ready_line = self.host.wait_line("stderr", MUX.HOST_READY_PREFIX, 360)
        ready = strict_object(
            ready_line[len(MUX.HOST_READY_PREFIX):], f"{lane.name} host readiness")
        if ready.get("schema") != "layersplit-persistent-driver-v1" \
                or ready.get("batch_size") != PHONE_BATCH \
                or ready.get("max_n_gen") != N_GEN:
            raise ExperimentError(f"{lane.name} host readiness failed")

    def exchange(self, launch_id: int) -> None:
        lane = self.config
        if self.host is None or self.phone is None:
            raise ExperimentError(f"{lane.name} exchange before readiness")
        session_end = "STOP" if launch_id == self.rounds else "DETACH"
        payload = canonical({
            "schema": "layersplit-persistent-command-v1",
            "launch_id": launch_id,
            "prompt": self.prompt,
            "n_gen": N_GEN,
            "request_count": PHONE_BATCH,
            "session_end": session_end,
        })
        host_start = self.host.snapshot()
        phone_start = self.phone.snapshot()
        start_us = time.time_ns() // 1000
        self.host.send(payload)
        result_line, _, phone_stderr, _ = MUX.wait_exchange(
            self.host, self.phone, host_start, phone_start, launch_id,
            30_000_000, self.host.process.pid, lane.layer_end, 48, "CUDA0",
        )
        end_us = time.time_ns() // 1000
        result = strict_object(result_line, f"{lane.name} result")
        validate_result_record(
            result, self.reference, PHONE_BATCH, launch_id, session_end)
        if result["route_wall_us"] > lane.slo_us:
            raise ExperimentError(f"{lane.name} SLO failed")
        certs = [line for line in phone_stderr if line.startswith(MUX.SESSION_PREFIX)]
        if len(certs) != 1:
            raise ExperimentError(f"{lane.name} SESSIONCERT count failed")
        session = parse_after(
            certs[0], MUX.SESSION_PREFIX, f"{lane.name} phone session")
        validate_phone_session(session, lane, launch_id, session_end)
        self.results.append(result)
        self.sessions.append(session)
        self.windows_us.append([start_us, end_us])
        if session_end == "STOP":
            if self.host.wait(20) != 0 or self.phone.wait(20) != 0:
                raise ExperimentError(f"{lane.name} STOP failed")

    def validate(self) -> None:
        if len(self.results) != self.rounds or len(self.sessions) != self.rounds \
                or len({item["host_pid"] for item in self.results}) != 1 \
                or len({item["worker_pid"] for item in self.sessions}) != 1 \
                or len({item["worker_boot_nonce"] for item in self.sessions}) != 1:
            raise ExperimentError(f"{self.config.name} persistence failed")

    def close(self) -> None:
        if self.host is not None:
            self.host.terminate()
            self.host.persist(self.row_dir, f"{self.config.name}-host")
        if self.phone is not None:
            self.phone.terminate()
            self.phone.persist(self.row_dir, f"{self.config.name}-phone")
        adb(self.config, "forward", "--remove", f"tcp:{self.config.port}", check=False)
        adb(self.config, "shell", "pkill -9 llama-layersplit >/dev/null 2>&1 || true",
            check=False)


def preflight() -> dict[str, Any]:
    for path in (FULL_MODEL, HOST_BIN, ANDROID_BIN, COHORT, INPUT_MANIFEST,
                 REFERENCE_REPORT, TYPED / "physical_mux.py"):
        if not path.is_file():
            raise ExperimentError(f"missing required artifact: {path}")
    devices = checked(["adb", "devices"]).decode("ascii", errors="replace")
    artifacts: dict[str, Any] = {
        "host_binary": digest_file(HOST_BIN),
        "android_binary": digest_file(ANDROID_BIN),
        "full_model": digest_file(FULL_MODEL),
        "physical_mux": digest_file(TYPED / "physical_mux.py"),
        "phones": {},
    }
    for lane in (OP15, OP12):
        if f"{lane.serial}\tdevice" not in devices:
            raise ExperimentError(f"{lane.name} is not connected")
        binary = adb(lane, "shell", f"sha256sum {lane.remote}/llama-layersplit").decode().split()[0]
        shard = adb(lane, "shell", f"sha256sum {lane.shard}", timeout=180).decode().split()[0]
        if "sha256:" + binary != artifacts["android_binary"]:
            raise ExperimentError(f"{lane.name} worker differs from the certified binary")
        if shard != lane.shard_sha256:
            raise ExperimentError(f"{lane.name} shard digest mismatch")
        artifacts["phones"][lane.name] = {
            "binary": "sha256:" + binary,
            "shard": "sha256:" + shard,
            "boot_id": adb(
                lane, "shell", "cat /proc/sys/kernel/random/boot_id").decode().strip(),
        }
    gpu_rows = checked([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ]).decode().splitlines()
    if not any(row.replace(" ", "") == f"{GPU_INDEX},{GPU_UUID}" for row in gpu_rows):
        raise ExperimentError("selected GPU index/UUID mapping changed")
    if not CPD.second_gpu_idle():
        raise ExperimentError("second A6000 has an active compute process")
    return artifacts


def write_power(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(canonical(row))


def persist_high(row_dir: Path, high: CPE.ReadyProcess) -> None:
    (row_dir / "high.stdout").write_text(
        high.stdout_text(), encoding="ascii", errors="replace")
    (row_dir / "high.stderr").write_text(
        high.stderr_text(), encoding="ascii", errors="replace")


def artifact_digests(row_dir: Path) -> dict[str, str]:
    result = {}
    for path in sorted(row_dir.iterdir()):
        if path.is_file() and path.name != "power.jsonl" and path.stat().st_size > 0:
            result[path.name] = digest_file(path)
    return result


def run_row(label: str, pair: int, row_dir: Path, prompt: str, reference: list[int],
            bge_batch: int, bge_words: int, bge_exact: int, bge_reps: int,
            rounds: int, full: bool) -> dict[str, Any]:
    row_dir.mkdir(parents=True)
    server: ServerLane | None = None
    phone_lanes: dict[str, PhoneLane] = {}
    high: CPE.ReadyProcess | None = None
    sampler = PowerSampler()
    thermal_start = thermal_end = None
    try:
        if label == "P0":
            server = ServerLane(row_dir, prompt, reference, rounds)
            server.start()
        else:
            thermal_start = {
                lane.name: CPE.wait_for_phone_thermal(lane.serial, 300)
                for lane in (OP15, OP12)
            }
            phone_lanes = {
                "op15": PhoneLane(
                    OP15, row_dir, prompt, reference,
                    rounds * BATCHES_PER_ROUND - min(rounds, OP12_EXCHANGE_CREDIT)),
                "op12": PhoneLane(
                    OP12, row_dir, prompt, reference,
                    min(rounds, OP12_EXCHANGE_CREDIT)),
            }
            phone_lanes["op15"].start()
            phone_lanes["op12"].start()
        high = CPE.bge_command(bge_batch, bge_words, bge_exact, bge_reps)
        high.start()
        high.wait_ready(360)
        if not CPD.second_gpu_idle():
            raise ExperimentError("second A6000 became active before GO")
        sampler.start()
        time.sleep(1.2)
        high.go()
        time.sleep(0.2)
        overlaps = []
        group_completion = []
        route_assignments = []
        phone_launch_ids = {"op15": 0, "op12": 0}
        for launch_id in range(1, rounds + 1):
            if label == "P0":
                assert server is not None
                first = (launch_id - 1) * BATCHES_PER_ROUND + 1
                for control_launch_id in range(first, first + BATCHES_PER_ROUND):
                    server.exchange(control_launch_id)
            else:
                round_start_us = time.time_ns() // 1000
                if launch_id <= min(rounds, OP12_EXCHANGE_CREDIT):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        futures = {}
                        for name in ("op15", "op12"):
                            phone_launch_ids[name] += 1
                            futures[name] = pool.submit(
                                phone_lanes[name].exchange, phone_launch_ids[name])
                        for future in futures.values():
                            future.result()
                    first_window = phone_lanes["op15"].windows_us[-1]
                    second_window = phone_lanes["op12"].windows_us[-1]
                    overlap = min(first_window[1], second_window[1]) \
                        - max(first_window[0], second_window[0])
                    if overlap <= 0:
                        raise ExperimentError("phone lane exchanges did not overlap")
                    overlaps.append(overlap)
                    route_assignments.append(["op15", "op12"])
                else:
                    phone_launch_ids["op15"] += 1
                    phone_lanes["op15"].exchange(phone_launch_ids["op15"])
                    first_window = phone_lanes["op15"].windows_us[-1]
                    phone_launch_ids["op15"] += 1
                    phone_lanes["op15"].exchange(phone_launch_ids["op15"])
                    second_window = phone_lanes["op15"].windows_us[-1]
                    route_assignments.append(["op15", "op15"])
                completion = [
                    first_window[1] - round_start_us,
                    second_window[1] - round_start_us,
                ]
                if completion[0] > OP15.slo_us or completion[1] > OP12.slo_us:
                    raise ExperimentError("phone class SLO including queueing failed")
                group_completion.append(completion)
        if server is not None:
            server.validate()
        for lane in phone_lanes.values():
            lane.validate()
        high_rc = high.wait(300)
        if high_rc != 0:
            raise ExperimentError(f"BGE process failed with rc={high_rc}")
        time.sleep(1.2)
        sampler.stop()
        bge = CPE.parse_bge(high, bge_batch, bge_exact, bge_reps)
        paid_start_us = int(round(float(bge["paid_start_s"]) * 1_000_000))
        paid_end_us = int(round(float(bge["paid_end_s"]) * 1_000_000))
        power_path = row_dir / "power.jsonl"
        write_power(power_path, sampler.rows)
        power = integrate_power(
            parse_power_jsonl(power_path.read_bytes()), paid_start_us, paid_end_us, full)
        if label == "P4":
            thermal_end = {
                lane.name: CPE.CPB.thermal_snapshot(lane.serial)
                for lane in (OP15, OP12)
            }
            for lane in (OP15, OP12):
                CPE.validate_end_thermal(thermal_end[lane.name], lane.serial)
        windows = server.windows_us if server is not None else [
            window
            for name in ("op15", "op12")
            for window in phone_lanes[name].windows_us
        ]
        all_inside = all(
            start >= paid_start_us and end <= paid_end_us for start, end in windows)
        if not all_inside:
            raise ExperimentError("low work escaped the paid BGE window")
        if not sampler.memory_mib:
            raise ExperimentError("selected-GPU memory was not sampled")
        persist_high(row_dir, high)
        if server is not None:
            server.close()
        for lane in phone_lanes.values():
            lane.close()
        row: dict[str, Any] = {
            "label": label,
            "pair": pair,
            "bge": bge,
            "rounds": rounds,
            "gemma_requests": rounds * BATCHES_PER_ROUND * CONTROL_BATCH,
            "gemma_tokens": rounds * BATCHES_PER_ROUND * CONTROL_BATCH * N_GEN,
            "all_tokens_match": True,
            "all_low_inside_bge": all_inside,
            "selected_gpu_peak_memory_mib": max(sampler.memory_mib),
            "power": power,
            "power_artifact": str(power_path.relative_to(HERE)),
            "power_sha256": digest_file(power_path),
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
        }
        if server is not None:
            row.update({
                "control_route_wall_us": [item["route_wall_us"] for item in server.results],
                "control_windows_us": server.windows_us,
                "control_results": server.results,
                "control_placements": server.placements,
            })
        else:
            row.update({
                "phone_route_wall_us": {
                    name: [item["route_wall_us"] for item in phone_lanes[name].results]
                    for name in ("op15", "op12")
                },
                "phone_windows_us": {
                    name: phone_lanes[name].windows_us for name in ("op15", "op12")
                },
                "phone_results": {
                    name: phone_lanes[name].results for name in ("op15", "op12")
                },
                "phone_sessions": {
                    name: phone_lanes[name].sessions for name in ("op15", "op12")
                },
                "round_overlap_us": overlaps,
                "group_completion_us": group_completion,
                "route_assignments": route_assignments,
            })
        row["raw_artifact_sha256"] = artifact_digests(row_dir)
        return row
    finally:
        try:
            sampler.stop()
        except ExperimentError:
            pass
        if high is not None:
            high.terminate()
            if not (row_dir / "high.stdout").exists():
                persist_high(row_dir, high)
        if server is not None:
            server.close()
        for lane in phone_lanes.values():
            lane.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--screen", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    full = bool(args.run)
    output_root = args.output or HERE / ("screen-results" if args.screen else "results")
    if not output_root.is_absolute():
        output_root = (HERE / output_root).resolve()
    if output_root.exists():
        raise ExperimentError(f"output directory already exists: {output_root}")
    if HERE.resolve() not in output_root.parents:
        raise ExperimentError("output directory must remain inside the spike")
    output_root.mkdir()
    prompt, reference, input_digests = load_frozen_input()
    artifacts = preflight()
    bge_points, knee, exact, bge_profile = CPD.load_bge_profile(CPD.BGE_PROFILE, 32)
    if knee != 16:
        raise ExperimentError("measured BGE knee is no longer B16")
    p50_us = next(point.duration_us for point in bge_points if point.batch_size == knee)
    window_s = 80 if full else 30
    bge_reps = math.ceil(window_s * 1_000_000 / p50_us)
    schedule = FULL_SCHEDULE if full else SCREEN_SCHEDULE
    rounds = FULL_ROUNDS if full else SCREEN_ROUNDS
    rows = []
    report_path = output_root / "report.json"
    try:
        for index, (label, pair) in enumerate(schedule):
            print(
                f"S18 row {index + 1}/{len(schedule)} {label} pair={pair}",
                flush=True,
            )
            rows.append(run_row(
                label, pair, output_root / f"row-{index:02d}-{label.lower()}",
                prompt, reference, knee, 25, exact, bge_reps, rounds, full,
            ))
        summary = summarize(rows, full)
        status = "S18_R1_FLEET_SCREEN_PASS" if not full and summary["screen_pass"] else \
            "S18_R1_FLEET_GPU_BOARD_PASS" if full and summary["overall_pass"] else \
            "S18_R1_FLEET_MECHANICS_PASS_RELIEF_INSUFFICIENT" if full \
            and summary["high_priority_gate"] else "S18_FAIL_GATE"
        report = {
            "schema": SCHEMA,
            "status": status,
            "mode": "full" if full else "screen",
            "scope": "SELECTED_A6000_GPU_BOARD_ONLY_PHONE_USB_TOTAL_UNKNOWN",
            "selected_gpu_uuid": GPU_UUID,
            "second_gpu_uuid": SECOND_GPU_UUID,
            "second_gpu_idle_at_end": CPD.second_gpu_idle(),
            "priority_provenance": "synthetic",
            "slo_provenance": "synthetic",
            "payload_provenance": "synthetic repetition of one observed BurstGPT cohort",
            "workload": {
                "bge_batch": knee,
                "bge_exact_tokens": exact,
                "bge_reps": bge_reps,
                "control_gemma_batch": CONTROL_BATCH,
                "control_batches_per_round": BATCHES_PER_ROUND,
                "phone_gemma_batch": PHONE_BATCH,
                "gemma_n_gen": N_GEN,
                "rounds": rounds,
            },
            "reference_tokens": reference,
            "input_digests": input_digests,
            "bge_profile_sha256": bge_profile,
            "artifacts": artifacts,
            "summary": summary,
            "rows": rows,
            "known_limitations": [
                "P4 holds two overlapping CUDA tail weight images",
                "priority SLO and repeated payload are synthetic",
            ],
            "formal_total_energy_claim": "NONE",
            "phone_energy": "UNKNOWN",
            "usb_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        }
        report_path.write_bytes(canonical(report))
        print(json.dumps({"status": status, "summary": summary}, indent=2, sort_keys=True))
        return 0 if status != "S18_FAIL_GATE" else 2
    except Exception as exc:
        report_path.write_bytes(canonical({
            "schema": SCHEMA,
            "status": "S18_FAIL_MEASUREMENT",
            "mode": "full" if full else "screen",
            "error": str(exc),
            "completed_rows": rows,
            "formal_total_energy_claim": "NONE",
        }))
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ExperimentError, CPE.LiveGateError, CPD.CPDError,
            MUX.MuxError, OSError, ValueError) as exc:
        print(f"S18_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
