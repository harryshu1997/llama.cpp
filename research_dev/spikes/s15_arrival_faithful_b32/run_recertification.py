#!/usr/bin/env python3
"""Recertify OP15 B32 with the prompt revealed only after host readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVE = HERE.parent / "s15_live_launcher"
sys.path.insert(0, str(LIVE))

import physical_launcher as base  # noqa: E402


HOST_DIR = ROOT / "build-cuda/bin"
HOST_BIN = HOST_DIR / "llama-layersplit"
SOURCE = ROOT / "examples/layersplit/layersplit.cpp"
RESULTS = HERE / "results"
PROCESSES = 7
MAX_COV = 0.05
PORT = 5951
LOCAL_LIBS = (
    "libllama-common.so.0",
    "libllama.so.0",
    "libggml.so.0",
    "libggml-base.so.0",
    "libggml-cpu.so.0",
    "libggml-cuda.so.0",
)


class RecertificationError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def local_artifacts() -> dict[str, str]:
    paths = {"llama-layersplit": HOST_BIN, "layersplit.cpp": SOURCE}
    for name in LOCAL_LIBS:
        paths[name] = (HOST_DIR / name).resolve()
    values = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RecertificationError(f"missing current host artifact: {path}")
        values[name] = digest(path)
    return values


def host_environment(layer_start: int | None) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = base.GPU_UUID
    env["LD_LIBRARY_PATH"] = str(HOST_DIR)
    if layer_start is None:
        env.pop("LLAMA_LAYER_START", None)
    else:
        env["LLAMA_LAYER_START"] = str(layer_start)
    env.pop("LLAMA_LAYER_END", None)
    return env


def reference_tokens(prompt: str, output_dir: Path) -> list[int]:
    command = [
        str(HOST_BIN), "-m", str(base.FULL_MODEL), "-ngl", "99",
        "--mode", "monodriver", "-p", prompt, "-n", str(base.N_GEN),
        "--driver-requests", str(base.BATCH), "--driver-warmup", "0",
        "--driver-batch", str(base.BATCH), "--driver-context", str(base.CONTEXT),
        "--driver-max-prefill", str(base.MAX_PREFILL),
    ]
    process = subprocess.run(
        command, capture_output=True, env=host_environment(None),
        timeout=base.PREPARE_TIMEOUT_S, check=False,
    )
    (output_dir / "reference.stdout.bin").write_bytes(process.stdout)
    (output_dir / "reference.stderr.bin").write_bytes(process.stderr)
    (output_dir / "reference.command.json").write_bytes(canonical(command))
    rows = base.prefixed_objects(process.stderr, b"ROUTEJSON ")
    if process.returncode != 0 or len(rows) != base.BATCH \
            or sorted(row.get("stream_index") for row in rows) != list(range(base.BATCH)):
        raise RecertificationError("current-host CUDA reference failed")
    tokens = rows[0].get("token_ids")
    if type(tokens) is not list or len(tokens) != base.N_GEN \
            or any(type(token) is not int for token in tokens) \
            or any(row.get("token_ids") != tokens for row in rows):
        raise RecertificationError("current-host CUDA reference streams disagree")
    return tokens


def prepare_route(raw_dir: Path) -> tuple[base.ReadyProcess, base.ReadyProcess, list[str], list[str]]:
    base.adb("forward", "--remove", f"tcp:{PORT}")
    if base.adb("forward", f"tcp:{PORT}", f"tcp:{PORT}").returncode != 0:
        raise RecertificationError("failed to install OP15 port forward")
    phone_shell = (
        f"cd {base.REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {base.SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {base.N_GEN} --driver-batch {base.BATCH} "
        f"--driver-context {base.CONTEXT} --driver-max-prefill {base.MAX_PREFILL}"
    )
    phone_command = ["adb", "-s", base.SERIAL, "shell", phone_shell]
    phone = base.ReadyProcess(
        phone_command, None, raw_dir / "phone.stdout.bin", raw_dir / "phone.stderr.bin",
        b"[stagenet] listening", ("stdout", "stderr"),
    )
    phone.wait_ready(base.PREPARE_TIMEOUT_S)

    host_command = [
        str(HOST_BIN), "-m", str(base.FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
        "--prompt-after-load", "-n", str(base.N_GEN),
        "--driver-requests", str(base.BATCH), "--driver-warmup", "0",
        "--driver-batch", str(base.BATCH), "--driver-context", str(base.CONTEXT),
        "--driver-max-prefill", str(base.MAX_PREFILL),
    ]
    host = base.ReadyProcess(
        host_command, host_environment(8), raw_dir / "host.stdout.bin",
        raw_dir / "host.stderr.bin", b"DRIVER_INPUT_READY ",
    )
    try:
        host.wait_ready(base.PREPARE_TIMEOUT_S)
    except Exception:
        host.terminate()
        phone.terminate()
        raise
    return host, phone, host_command, phone_command


def clean_route(host: base.ReadyProcess | None, phone: base.ReadyProcess | None) -> None:
    for process in (host, phone):
        if process is not None:
            process.terminate()
    base.adb("forward", "--remove", f"tcp:{PORT}")
    base.adb("shell", "pkill -9 -f llama-layersplit")


class ExitWatch:
    def __init__(self, process: base.ReadyProcess) -> None:
        self.process = process
        self.returncode = None
        self.observed_ns = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._wait, daemon=True)
        self.thread.start()

    def _wait(self) -> None:
        self.returncode = self.process.process.wait()
        self.observed_ns = time.monotonic_ns()
        self.done.set()

    def finish(self, timeout: float) -> tuple[int, int]:
        if not self.done.wait(timeout):
            self.process.terminate()
            raise RecertificationError("prepared process completion timeout")
        self.thread.join(timeout=1)
        for reader in self.process.threads:
            reader.join(timeout=5)
        if self.thread.is_alive() or any(reader.is_alive() for reader in self.process.threads) \
                or type(self.returncode) is not int or type(self.observed_ns) is not int:
            raise RecertificationError("prepared process watcher did not terminate")
        self.process.close()
        return self.returncode, self.observed_ns


def run_process(index: int, prompt: str, reference: list[int],
                thermal: base.ThermalMonitor) -> dict:
    raw_dir = RESULTS / f"process-{index}"
    raw_dir.mkdir(parents=True)
    host = None
    phone = None
    try:
        start_thermal = thermal.snapshot()
        if start_thermal.get("sample_age_us", 1_000_001) > 1_000_000 \
                or not base.thermal_ok(start_thermal, base.THERMAL_START_MAX_MILLIC):
            raise RecertificationError("start thermal gate failed")
        host, phone, host_command, phone_command = prepare_route(raw_dir)
        pre_prompt = (raw_dir / "host.stderr.bin").read_bytes()
        if pre_prompt.count(b"DRIVER_INPUT_READY ") != 1 \
                or b"DRIVER_INPUT_ACCEPTED " in pre_prompt \
                or prompt.encode("utf-8") in pre_prompt \
                or "-p" in host_command or prompt in host_command:
            raise RecertificationError("pre-prompt host evidence is not input-blind")
        (raw_dir / "host.pre_prompt.stderr.bin").write_bytes(pre_prompt)
        (raw_dir / "host.command.json").write_bytes(canonical(host_command))
        (raw_dir / "phone.command.json").write_bytes(canonical(phone_command))

        prompt_payload = (prompt + "\n").encode("utf-8")
        prompt_sha256 = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        host_exit = ExitWatch(host)
        phone_exit = ExitWatch(phone)
        ready_observed_ns = time.monotonic_ns()
        paid_start_ns = time.monotonic_ns()
        if host.process.stdin is None:
            raise RecertificationError("prepared host stdin is unavailable")
        host.process.stdin.write(prompt_payload)
        host.process.stdin.flush()
        host.process.stdin.close()
        prompt_submitted_ns = time.monotonic_ns()
        host_rc, host_exit_ns = host_exit.finish(30)
        phone_rc, phone_exit_ns = phone_exit.finish(30)
        paid_end_ns = max(host_exit_ns, phone_exit_ns)
        end_thermal = thermal.snapshot()
        if end_thermal.get("sample_age_us", 1_000_001) > 1_000_000 \
                or not base.thermal_ok(end_thermal, base.THERMAL_END_MAX_MILLIC):
            raise RecertificationError("end thermal gate failed")
        if host_rc != 0 or phone_rc != 0:
            raise RecertificationError("host or phone process failed")

        host_stderr = (raw_dir / "host.stderr.bin").read_bytes()
        if host_stderr.count(b"DRIVER_INPUT_READY ") != 1 \
                or host_stderr.count(b"DRIVER_INPUT_ACCEPTED ") != 1:
            raise RecertificationError("post-load input marker count failed")
        placement, rows = base.validate_completion(
            host_stderr,
            (raw_dir / "phone.stdout.bin").read_bytes()
            + (raw_dir / "phone.stderr.bin").read_bytes(),
            reference,
        )
        elapsed_us = (paid_end_ns - paid_start_ns) // 1000
        record = {
            "schema": "s15-post-load-b32-process-v1",
            "process_index": index,
            "eligible": True,
            "host_command": host_command,
            "phone_command": phone_command,
            "prompt_sha256": prompt_sha256,
            "prompt_in_argv": False,
            "ready_observed_ns": ready_observed_ns,
            "paid_start_ns": paid_start_ns,
            "prompt_submitted_ns": prompt_submitted_ns,
            "host_exit_observed_ns": host_exit_ns,
            "phone_exit_observed_ns": phone_exit_ns,
            "paid_end_ns": paid_end_ns,
            "completion_elapsed_us": elapsed_us,
            "route_wall_us_max": max(row["request_wall_us"] for row in rows),
            "thermal_start": start_thermal,
            "thermal_end": end_thermal,
            "placement": placement,
            "token_ids": reference,
            "stream_count": len(rows),
        }
        (raw_dir / "process.json").write_bytes(canonical(record))
        return record
    except Exception as exc:
        (raw_dir / "error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="ascii", errors="backslashreplace",
        )
        return {
            "schema": "s15-post-load-b32-process-v1",
            "process_index": index,
            "eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        clean_route(host, phone)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise RecertificationError("physical acquisition requires --run")
    if RESULTS.exists():
        raise RecertificationError("results directory already exists")
    RESULTS.mkdir(parents=True)

    cohort, inputs, request_ids, prompt = base.load_workload()
    if len(request_ids) != base.BATCH:
        raise RecertificationError("frozen cohort is not B32")
    remaining_budget_us = cohort["admission_schedule"]["earliest_deadline_us"] \
        - cohort["admission_schedule"]["planned_launch_us"]
    if remaining_budget_us != 4_000_000:
        raise RecertificationError("unexpected frozen remaining deadline budget")
    artifacts = local_artifacts()
    frozen = base.verify_artifacts()
    boot_id, deployed = base.deploy(frozen)
    thermal_paths, initial_thermal = base.discover_thermal_paths()
    thermal = base.ThermalMonitor(
        RESULTS / "thermal_stream.log", RESULTS / "thermal_stream.stderr.bin", thermal_paths,
    )
    thermal.wait_ready()
    try:
        reference = reference_tokens(prompt, RESULTS)
        records = []
        for index in range(PROCESSES):
            records.append(run_process(index, prompt, reference, thermal))
            time.sleep(1)
    finally:
        thermal.stop()
        clean_route(None, None)

    elapsed = [record["completion_elapsed_us"] for record in records if record.get("eligible") is True]
    problems = []
    if len(elapsed) != PROCESSES:
        problems.append("not all seven physical processes passed")
    cov = statistics.pstdev(elapsed) / statistics.mean(elapsed) if len(elapsed) > 1 else None
    if type(cov) is not float or cov > MAX_COV:
        problems.append("completion-time CoV exceeds five percent")
    conservative_us = max(elapsed) if elapsed else None
    if type(conservative_us) is not int or conservative_us > remaining_budget_us:
        problems.append("post-admission route exceeds the remaining cohort deadline budget")
    verdict = "POST_LOAD_B32_ROUTE_CERTIFIED" if not problems else "POST_LOAD_B32_ROUTE_NOT_CERTIFIED"
    profile_identity = {
        "schema": "s15-post-load-b32-profile-identity-v1",
        "host_artifacts": artifacts,
        "phone_worker_sha256": base.EXPECTED_WORKER,
        "device_id": base.EXPECTED_DEVICE,
        "cohort_sha256": base.EXPECTED_COHORT,
        "input_manifest_sha256": base.EXPECTED_INPUT,
        "layer_range": [0, 8],
        "batch": 32,
        "n_gen": 8,
        "completion_elapsed_us_by_process": elapsed,
    }
    profile_id = "sha256:" + hashlib.sha256(canonical(profile_identity)).hexdigest()
    report = {
        "schema": "s15-post-load-b32-recertification-v1",
        "verdict": verdict,
        "scope": "REAL_OP15_A6000_POST_LOAD_PROMPT_LATENCY_CORRECTNESS_PLACEMENT_ENERGY_UNKNOWN",
        "energy_scope": "UNKNOWN",
        "problems": problems,
        "processes_required": PROCESSES,
        "records": records,
        "completion_elapsed_us_by_process": elapsed,
        "completion_p50_us": statistics.median(elapsed) if elapsed else None,
        "completion_conservative_us": conservative_us,
        "completion_cov": cov,
        "remaining_deadline_budget_us": remaining_budget_us,
        "profile_id": profile_id,
        "profile_identity": profile_identity,
        "reference_tokens": reference,
        "host_artifacts": artifacts,
        "frozen_phone_manifest_sha256": digest(base.ARTIFACTS / "SHA256SUMS.txt"),
        "deployed": deployed,
        "device_boot_id": boot_id,
        "thermal_paths": thermal_paths,
        "initial_thermal": initial_thermal,
        "cohort_sha256": base.EXPECTED_COHORT,
        "input_manifest_sha256": base.EXPECTED_INPUT,
        "prompt_visible_during_preflight": False,
    }
    report["artifact_hashes_before_report"] = {
        str(path.relative_to(RESULTS)): digest(path)
        for path in sorted(RESULTS.rglob("*")) if path.is_file()
    }
    (RESULTS / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii",
    )
    print(json.dumps({"verdict": verdict, "problems": problems}, indent=2))
    return 0 if not problems else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecertificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
