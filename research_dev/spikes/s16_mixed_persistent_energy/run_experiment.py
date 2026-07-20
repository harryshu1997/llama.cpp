#!/usr/bin/env python3
"""Run the real S16 P0/P2 mixed persistent acquisition."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ENERGY = HERE.parent / "s14_mixed_streaming_scheduler/energy"
TYPED = HERE.parent / "s15_persistent_typed_gate"
for path in (HERE, ENERGY, TYPED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cp_d_priority as CPD  # noqa: E402
import cp_e_live_priority as CPE  # noqa: E402
import physical_mux as MUX  # noqa: E402
from experiment_contract import (  # noqa: E402
    BATCH, FULL_SCHEDULE, LOW_COHORTS, N_GEN, ContractError, canonical,
    digest_file, integrate_power, parse_power_jsonl, strict_object, summarize,
    validate_result_record,
)


SCHEMA = "s16-mixed-persistent-energy-v1"
SERIAL = "3C15AU002CL00000"
GPU_INDEX = 0
GPU_UUID = CPD.SELECTED_GPU_UUID
SECOND_GPU_UUID = CPD.SECOND_GPU_UUID
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
HOST_BIN = ROOT / "build-cuda/bin/llama-layersplit"
HOST_LIB = HOST_BIN.parent
ANDROID_BIN = ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin/llama-layersplit"
REMOTE = "/data/local/tmp/ls-s15-persistent-typed"
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf"
SHARD_SHA256 = "a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8"
PORT = 5964
CONTEXT = 512
MAX_PREFILL = 64
COHORT = HERE.parent / "s15_burst_cohort/cohort.json"
INPUT_MANIFEST = HERE.parent / "s15_burst_cohort/input_manifest.json"
REFERENCE_REPORT = TYPED / "results/report.json"
POWER_PERIOD_MS = 100


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


def adb(*args: str, timeout: int = 30) -> bytes:
    return checked(["adb", "-s", SERIAL, *args], timeout=timeout)


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


def parse_json_after(line: bytes, prefix: bytes, label: str) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ExperimentError(f"missing {label} prefix")
    return strict_object(line[len(prefix):], label, False)


def validate_host_placement(value: dict[str, Any], label: str, pid: int) -> None:
    expected_role = "monodriver" if label == "P0" else "host_tail"
    expected_mode = "monodriver" if label == "P0" else "pipedriver"
    expected_start = 0 if label == "P0" else 8
    if value.get("schema") != "layersplit-scheduled-placement-v2" \
            or value.get("role") != expected_role or value.get("mode") != expected_mode \
            or value.get("layer_start") != expected_start or value.get("layer_end") != 48 \
            or value.get("n_layer") != 48 or value.get("pid") != pid \
            or value.get("run_rc") != 0 or value.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0:
        raise ExperimentError(f"{label} host placement identity failed")
    if type(value.get("compute_nodes")) is not int or value["compute_nodes"] <= 0:
        raise ExperimentError(f"{label} host placement has no compute")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ExperimentError(f"{label} host placement has no operation tally")
    for op_name, buffers in by_op.items():
        if type(buffers) is not dict or not buffers:
            raise ExperimentError(f"{label} host placement has an invalid operation tally")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ExperimentError(f"{label} host placement has an invalid node count")
            if backend != "CUDA0" and not (
                    label == "P0" and op_name == "GET_ROWS" and backend == "CUDA_Host"):
                raise ExperimentError(f"{label} undeclared host fallback {op_name}@{backend}")


def validate_phone_session(value: dict[str, Any], launch_id: int, session_end: str) -> None:
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != launch_id \
            or value.get("session_end") != session_end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 or value.get("layer_end") != 8 \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (session_end == "DETACH"):
        raise ExperimentError("phone session certificate identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise ExperimentError("phone session certificate has no operation tally")
    for op_name, buffers in by_op.items():
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                raise ExperimentError("phone session has an invalid node count")
            if backend != "HTP0" and not (op_name == "GET_ROWS" and backend == "CPU"):
                raise ExperimentError(f"undeclared phone fallback {op_name}@{backend}")


class PowerSampler:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            [
                "nvidia-smi", "-i", str(GPU_INDEX),
                "--query-gpu=power.draw,power.limit,utilization.gpu,pstate",
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
                if len(parts) != 4:
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


class LowSession:
    def __init__(self, label: str, row_dir: Path, prompt: str,
                 reference: list[int], expected_exchanges: int) -> None:
        self.label = label
        self.row_dir = row_dir
        self.prompt = prompt
        self.reference = reference
        self.expected_exchanges = expected_exchanges
        self.process: MUX.MonitoredProcess | None = None
        self.host_pids: list[int] = []
        self.worker_pids: list[int] = []
        self.worker_nonces: list[int] = []
        self.route_wall_us: list[int] = []
        self.exchange_windows_us: list[list[int]] = []
        self.forwarded = False

    def _control(self) -> tuple[tuple[str, ...], dict[str, str]]:
        command = (
            str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
            "--mode", "monodriver", "-n", str(N_GEN),
            "--driver-batch", str(BATCH), "--driver-context", str(CONTEXT),
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
        return command, environment

    def _treatment(self) -> tuple[tuple[str, ...], dict[str, str] | None]:
        adb("shell", "pkill -9 llama-layersplit >/dev/null 2>&1 || true")
        subprocess.run(["adb", "-s", SERIAL, "forward", "--remove", f"tcp:{PORT}"],
                       capture_output=True, timeout=10)
        checked(["adb", "-s", SERIAL, "forward", f"tcp:{PORT}", f"tcp:{PORT}"])
        self.forwarded = True
        phone_shell = (
            f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
            "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
            f"./llama-layersplit -m {SHARD} --devices HTP0 -ngl 99 --mode stagenet "
            f"--port {PORT} -n {N_GEN} --driver-batch {BATCH} "
            f"--driver-context {CONTEXT} --driver-max-prefill {MAX_PREFILL}"
        )
        host_command = [
            str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99",
            "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
            "-n", str(N_GEN), "--driver-batch", str(BATCH),
            "--driver-context", str(CONTEXT), "--driver-max-prefill", str(MAX_PREFILL),
            "--persistent-jsonl",
        ]
        mux_artifacts = self.row_dir / "mux-artifacts"
        config = {
            "schema": "s15-physical-mux-config-v1",
            "host_command": host_command,
            "phone_command": ["adb", "-s", SERIAL, "shell", phone_shell],
            "host_env": {
                "CUDA_VISIBLE_DEVICES": GPU_UUID,
                "LD_LIBRARY_PATH": str(HOST_LIB),
                "LLAMA_LAYER_START": "8",
                "LAYERSPLIT_PLACEMENT_CERT": "1",
            },
            "artifact_root": str(mux_artifacts.resolve()),
            "host_layer_start": 8,
            "host_layer_end": 48,
            "host_backend": "CUDA0",
        }
        config_path = self.row_dir / "mux.config.json"
        config_path.write_bytes(canonical(config))
        return (sys.executable, str(TYPED / "physical_mux.py"), "--config", str(config_path)), None

    def start(self) -> None:
        command, environment = self._control() if self.label == "P0" else self._treatment()
        self.process = MUX.MonitoredProcess(command, environment)
        line = self.process.wait_line("stderr", MUX.HOST_READY_PREFIX, 360)
        ready = strict_object(line[len(MUX.HOST_READY_PREFIX):], "low readiness")
        if ready.get("schema") != "layersplit-persistent-driver-v1" \
                or ready.get("batch_size") != BATCH or ready.get("max_n_gen") != N_GEN:
            raise ExperimentError("low persistent readiness failed")

    def exchange(self, launch_id: int) -> None:
        if self.process is None:
            raise ExperimentError("low exchange before readiness")
        session_end = "STOP" if launch_id == self.expected_exchanges else "DETACH"
        command = canonical({
            "schema": "layersplit-persistent-command-v1",
            "launch_id": launch_id,
            "prompt": self.prompt,
            "n_gen": N_GEN,
            "request_count": BATCH,
            "session_end": session_end,
        })
        start = self.process.snapshot()
        start_us = time.time_ns() // 1000
        self.process.send(command)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            stdout = [line for line in self.process.lines_since("stdout", start[0]) if line.strip()]
            stderr = self.process.lines_since("stderr", start[1])
            markers = [line for line in stderr if line.startswith(MUX.HOST_END_PREFIX)]
            placements = [line for line in stderr if line.startswith(b"PLACEMENTCERT ")]
            sessions = [line for line in stderr if line.startswith(MUX.SESSION_PREFIX)]
            expected_sessions = 1 if self.label == "P2" else 0
            if len(stdout) == len(markers) == len(placements) == 1 \
                    and len(sessions) == expected_sessions:
                end_us = time.time_ns() // 1000
                result = strict_object(stdout[0], "low persistent result")
                validate_result_record(result, self.reference, self.label, launch_id, session_end)
                placement = parse_json_after(placements[0], b"PLACEMENTCERT ", "host placement")
                validate_host_placement(placement, self.label, result["host_pid"])
                if sessions:
                    session = parse_json_after(sessions[0], MUX.SESSION_PREFIX, "phone session")
                    validate_phone_session(session, launch_id, session_end)
                    self.worker_pids.append(session["worker_pid"])
                    self.worker_nonces.append(session["worker_boot_nonce"])
                self.host_pids.append(result["host_pid"])
                self.route_wall_us.append(result["route_wall_us"])
                self.exchange_windows_us.append([start_us, end_us])
                if session_end == "STOP" and self.process.wait(20) != 0:
                    raise ExperimentError("low persistent process did not stop cleanly")
                return
            if len(stdout) > 1 or len(markers) > 1 or len(placements) > 1 \
                    or len(sessions) > expected_sessions:
                raise ExperimentError("duplicate low exchange evidence")
            if self.process.error is not None:
                raise ExperimentError(self.process.error)
            if self.process.process.poll() is not None and not stdout:
                raise ExperimentError("low process exited before result")
            time.sleep(0.01)
        raise ExperimentError("low exchange timed out")

    def validate_persistence(self) -> None:
        if len(self.route_wall_us) != self.expected_exchanges or len(set(self.host_pids)) != 1:
            raise ExperimentError("host persistence or exchange count failed")
        if self.label == "P2" and (
                len(set(self.worker_pids)) != 1 or len(set(self.worker_nonces)) != 1):
            raise ExperimentError("phone persistence identity changed")

    def close(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.persist(self.row_dir, "low")
        if self.forwarded:
            subprocess.run(["adb", "-s", SERIAL, "forward", "--remove", f"tcp:{PORT}"],
                           capture_output=True, timeout=10)
            subprocess.run(["adb", "-s", SERIAL, "shell", "pkill -9 llama-layersplit"],
                           capture_output=True, timeout=10)


def preflight() -> dict[str, Any]:
    for path in (FULL_MODEL, HOST_BIN, ANDROID_BIN, COHORT, INPUT_MANIFEST, REFERENCE_REPORT):
        if not path.is_file():
            raise ExperimentError(f"missing required artifact: {path}")
    devices = checked(["adb", "devices"]).decode("ascii", errors="replace")
    if f"{SERIAL}\tdevice" not in devices:
        raise ExperimentError("OP15 is not connected")
    remote_binary = adb("shell", f"sha256sum {REMOTE}/llama-layersplit").decode().split()[0]
    if "sha256:" + remote_binary != digest_file(ANDROID_BIN):
        raise ExperimentError("remote OP15 worker differs from the current certified Android binary")
    remote_shard = adb("shell", f"sha256sum {SHARD}", timeout=180).decode().split()[0]
    if remote_shard != SHARD_SHA256:
        raise ExperimentError("remote OP15 shard digest mismatch")
    gpu_rows = checked([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ]).decode().splitlines()
    if not any(row.replace(" ", "") == f"{GPU_INDEX},{GPU_UUID}" for row in gpu_rows):
        raise ExperimentError("selected GPU index/UUID mapping changed")
    if not CPD.second_gpu_idle():
        raise ExperimentError("second A6000 has an active compute process")
    boot_id = adb("shell", "cat /proc/sys/kernel/random/boot_id").decode().strip()
    return {
        "host_binary": digest_file(HOST_BIN),
        "android_binary": digest_file(ANDROID_BIN),
        "full_model": digest_file(FULL_MODEL),
        "physical_mux": digest_file(TYPED / "physical_mux.py"),
        "device_boot_id": boot_id,
        "shard": "sha256:" + remote_shard,
    }


def write_power(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("wb") as output:
        for row in rows:
            output.write(canonical(row))


def run_row(label: str, pair: int, row_dir: Path, prompt: str, reference: list[int],
            bge_batch: int, bge_words: int, bge_exact: int, bge_reps: int,
            low_cohorts: int, require_quality: bool) -> dict[str, Any]:
    row_dir.mkdir(parents=True)
    low = LowSession(label, row_dir, prompt, reference, low_cohorts)
    high = None
    sampler = PowerSampler()
    thermal_start = thermal_end = None
    try:
        if label == "P2":
            thermal_start = CPE.wait_for_phone_thermal(SERIAL, 300)
        low.start()
        high = CPE.bge_command(bge_batch, bge_words, bge_exact, bge_reps)
        high.start()
        high.wait_ready(360)
        if not CPD.second_gpu_idle():
            raise ExperimentError("second A6000 became active before GO")
        sampler.start()
        time.sleep(1.2)
        high.go()
        time.sleep(0.2)
        for launch_id in range(1, low_cohorts + 1):
            low.exchange(launch_id)
        low.validate_persistence()
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
        reopened = parse_power_jsonl(power_path.read_bytes())
        power = integrate_power(reopened, paid_start_us, paid_end_us, require_quality)
        if label == "P2":
            thermal_end = CPE.CPB.thermal_snapshot(SERIAL)
            CPE.validate_end_thermal(thermal_end, SERIAL)
        all_inside = all(start >= paid_start_us and end <= paid_end_us
                         for start, end in low.exchange_windows_us)
        return {
            "label": label,
            "pair": pair,
            "bge": bge,
            "gemma_batches": low_cohorts,
            "gemma_requests": low_cohorts * BATCH,
            "gemma_tokens": low_cohorts * BATCH * N_GEN,
            "low_route_wall_us": low.route_wall_us,
            "low_exchange_windows_us": low.exchange_windows_us,
            "host_pid": low.host_pids[0],
            "worker_pid": low.worker_pids[0] if low.worker_pids else None,
            "all_tokens_match": True,
            "all_low_inside_bge": all_inside,
            "power": power,
            "power_artifact": str(power_path.relative_to(row_dir.parent.parent)),
            "power_sha256": digest_file(power_path),
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
        }
    finally:
        try:
            sampler.stop()
        except ExperimentError:
            pass
        if high is not None:
            high.terminate()
            (row_dir / "high.stdout").write_text(high.stdout_text(), encoding="ascii", errors="replace")
            (row_dir / "high.stderr").write_text(high.stderr_text(), encoding="ascii", errors="replace")
        low.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--screen", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    output_root = HERE / ("screen-results" if args.screen else "results")
    if output_root.exists():
        raise ExperimentError(f"output directory already exists: {output_root}")
    output_root.mkdir()
    prompt, reference, input_digests = load_frozen_input()
    artifacts = preflight()
    bge_points, knee, exact, bge_profile = CPD.load_bge_profile(CPD.BGE_PROFILE, 32)
    if knee != 16:
        raise ExperimentError("measured BGE knee is no longer B16")
    p50_us = next(point.duration_us for point in bge_points if point.batch_size == knee)
    window_s = 15 if args.screen else 80
    bge_reps = math.ceil(window_s * 1_000_000 / p50_us)
    schedule = (("P0", 0), ("P2", 0)) if args.screen else FULL_SCHEDULE
    low_cohorts = 2 if args.screen else LOW_COHORTS
    rows = []
    report_path = output_root / "report.json"
    try:
        for index, (label, pair) in enumerate(schedule):
            print(f"S16 row {index + 1}/{len(schedule)} {label} pair={pair}", flush=True)
            rows.append(run_row(
                label, pair, output_root / f"row-{index:02d}-{label.lower()}",
                prompt, reference, knee, 25, exact, bge_reps, low_cohorts,
                require_quality=not args.screen,
            ))
        summary = summarize(rows, require_power_quality=not args.screen)
        status = "S16_SCREEN_PASS" if args.screen and summary["screen_pass"] else \
            "S16_MIXED_GPU_BOARD_DIAGNOSTIC_PASS" if not args.screen and summary["overall_pass"] else \
            "S16_FAIL_GATE"
        report = {
            "schema": SCHEMA,
            "status": status,
            "mode": "screen" if args.screen else "full",
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
                "gemma_batch": BATCH,
                "gemma_n_gen": N_GEN,
                "gemma_cohorts": low_cohorts,
            },
            "reference_tokens": reference,
            "input_digests": input_digests,
            "bge_profile_sha256": bge_profile,
            "artifacts": artifacts,
            "summary": summary,
            "rows": rows,
            "formal_total_energy_claim": "NONE",
            "phone_energy": "UNKNOWN",
            "usb_energy": "UNKNOWN",
            "total_system_energy": "UNKNOWN",
        }
        report_path.write_bytes(canonical(report))
        print(json.dumps({"status": status, "summary": summary}, indent=2, sort_keys=True))
        return 0 if status.endswith("PASS") else 2
    except Exception as exc:
        report_path.write_bytes(canonical({
            "schema": SCHEMA,
            "status": "S16_FAIL_MEASUREMENT",
            "mode": "screen" if args.screen else "full",
            "error": str(exc),
            "completed_rows": rows,
            "formal_total_energy_claim": "NONE",
        }))
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ExperimentError, CPE.LiveGateError, CPD.CPDError, OSError) as exc:
        print(f"S16_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
