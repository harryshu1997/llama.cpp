#!/usr/bin/env python3
"""Run two B32 exchanges with one resident OP15 worker and CUDA tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVE = HERE.parent / "s15_live_launcher"
ARRIVAL = HERE.parent / "s15_arrival_faithful_b32"
PERSISTENT = HERE.parent / "s15_persistent_b32"
for path in (LIVE, ARRIVAL, PERSISTENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_launcher as base  # noqa: E402
import run_recertification as recert  # noqa: E402
import run_gate as prior  # noqa: E402


ANDROID_BIN = prior.ANDROID_BIN
HOST_BIN = prior.HOST_BIN
SOURCE = ROOT / "examples/layersplit/layersplit.cpp"
REMOTE = "/data/local/tmp/ls-s15-persistent-host"
PORT = 5963
SESSIONS = 2
BATCH = 32
N_GEN = 8
RESULTS = HERE / "results"
REPORT = RESULTS / "report.json"
HOST_CERT_KEYS = {
    "schema", "role", "mode", "layer_start", "layer_end", "n_layer", "pid",
    "run_rc", "compute_nodes", "copy_nodes", "metadata_nodes",
    "missing_buffer_compute_nodes", "compute_by_buffer_type", "compute_by_op",
    "compute_by_op_and_buffer", "copy_by_buffer_type", "status",
}


class GateError(RuntimeError):
    pass


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def wait_objects(path: Path, prefix: bytes, count: int, timeout_s: float) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            values = base.prefixed_objects(path.read_bytes(), prefix)
            if len(values) >= count:
                return values
        time.sleep(0.02)
    raise GateError(f"timed out waiting for {prefix.decode('ascii')} record {count}")


def wait_results(path: Path, count: int, timeout_s: float) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            payload = path.read_bytes()
            lines = payload.splitlines(keepends=True)
            if len(lines) >= count:
                if any(not line.endswith(b"\n") for line in lines):
                    raise GateError("host stdout contains a partial result")
                return [base.strict_object(line, "persistent result") for line in lines]
        time.sleep(0.02)
    raise GateError(f"timed out waiting for persistent result {count}")


def validate_host_cert(cert: dict, host_pid: int) -> None:
    if type(cert) is not dict or set(cert) != HOST_CERT_KEYS:
        raise GateError("host placement certificate has an invalid key set")
    expected = {
        "schema": "layersplit-scheduled-placement-v2",
        "role": "host_tail",
        "mode": "pipedriver",
        "layer_start": 8,
        "layer_end": 48,
        "n_layer": 48,
        "pid": host_pid,
        "run_rc": 0,
        "missing_buffer_compute_nodes": 0,
        "status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected.items():
        if type(cert.get(key)) is not type(value) or cert.get(key) != value:
            raise GateError(f"host placement certificate mismatch: {key}")
    if type(cert["compute_nodes"]) is not int or cert["compute_nodes"] <= 0:
        raise GateError("host placement certificate observed no compute")
    expected_nodes = 0
    mapping = cert["compute_by_op_and_buffer"]
    if type(mapping) is not dict or not mapping:
        raise GateError("host placement mapping is empty")
    for op, buffers in mapping.items():
        if type(op) is not str or not op or type(buffers) is not dict or not buffers:
            raise GateError("host placement entry is invalid")
        for backend, nodes in buffers.items():
            if type(backend) is not str or type(nodes) is not int or nodes <= 0:
                raise GateError("host placement count is invalid")
            if backend == "CUDA0":
                expected_nodes += nodes
            elif backend != "CPU" or op != "GET_ROWS":
                raise GateError("host placement contains undeclared fallback")
    if expected_nodes == 0:
        raise GateError("host placement observed no CUDA0 compute")


def validate_result(result: dict, launch_id: int, end: str,
                    host_pid: int, reference: list[int]) -> None:
    expected_keys = {
        "schema", "launch_id", "outcome", "host_pid", "request_count",
        "batch_size", "n_gen", "session_end", "elapsed_us", "route_wall_us",
        "token_ids",
    }
    if type(result) is not dict or set(result) != expected_keys:
        raise GateError("persistent result has an invalid key set")
    expected = {
        "schema": "layersplit-persistent-result-v1",
        "launch_id": launch_id,
        "outcome": "completed",
        "host_pid": host_pid,
        "request_count": BATCH,
        "batch_size": BATCH,
        "n_gen": N_GEN,
        "session_end": end,
    }
    for key, value in expected.items():
        if type(result.get(key)) is not type(value) or result.get(key) != value:
            raise GateError(f"persistent result mismatch: {key}")
    for key in ("elapsed_us", "route_wall_us"):
        if type(result[key]) is not int or result[key] <= 0:
            raise GateError(f"persistent result has invalid {key}")
    if result["elapsed_us"] > 4_000_000:
        raise GateError("persistent exchange exceeded its post-launch budget")
    token_ids = result["token_ids"]
    if type(token_ids) is not list or len(token_ids) != BATCH \
            or any(type(tokens) is not list or tokens != reference for tokens in token_ids):
        raise GateError("persistent exchange is not token-exact")


def deploy() -> tuple[str, dict[str, str]]:
    if not ANDROID_BIN.is_file() or not HOST_BIN.is_file() or not SOURCE.is_file():
        raise GateError("current build artifacts are missing")
    base.adb_checked("shell", f"rm -rf {REMOTE} && cp -a {base.REMOTE} {REMOTE}")
    process = base.adb("push", str(ANDROID_BIN), f"{REMOTE}/llama-layersplit", timeout=240)
    if process.returncode != 0:
        raise GateError("failed to deploy the current Android worker")
    base.adb_checked("shell", f"chmod 755 {REMOTE}/llama-layersplit")
    remote = base.adb_checked(
        "shell", f"sha256sum {REMOTE}/llama-layersplit",
    ).decode("ascii").split()[0]
    if "sha256:" + remote != digest(ANDROID_BIN):
        raise GateError("deployed worker digest mismatch")
    shard = base.adb_checked(
        "shell", f"sha256sum {base.SHARD}",
    ).decode("ascii").split()[0]
    if shard != base.SHARD_SHA256:
        raise GateError("phone shard digest mismatch")
    boot_id = base.adb_checked(
        "shell", "cat /proc/sys/kernel/random/boot_id",
    ).decode("ascii").strip()
    return boot_id, {
        "android_worker": digest(ANDROID_BIN),
        "host_worker": digest(HOST_BIN),
        "source": digest(SOURCE),
        "phone_shard": "sha256:" + shard,
    }


def prepare_phone(output_dir: Path) -> tuple[base.ReadyProcess, list[str]]:
    base.adb("forward", "--remove", f"tcp:{PORT}")
    if base.adb("forward", f"tcp:{PORT}", f"tcp:{PORT}").returncode != 0:
        raise GateError("failed to install the OP15 port forward")
    shell = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {base.SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {N_GEN} --driver-batch {BATCH} "
        f"--driver-context {base.CONTEXT} --driver-max-prefill {base.MAX_PREFILL}"
    )
    command = ["adb", "-s", base.SERIAL, "shell", shell]
    phone = base.ReadyProcess(
        command, None, output_dir / "phone.stdout.bin", output_dir / "phone.stderr.bin",
        b"[stagenet] listening", ("stdout", "stderr"),
    )
    phone.wait_ready(base.PREPARE_TIMEOUT_S)
    return phone, command


def prepare_host(output_dir: Path) -> tuple[base.ReadyProcess, list[str]]:
    command = [
        str(HOST_BIN), "-m", str(base.FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
        "-n", str(N_GEN), "--driver-batch", str(BATCH),
        "--driver-context", str(base.CONTEXT),
        "--driver-max-prefill", str(base.MAX_PREFILL), "--persistent-jsonl",
    ]
    environment = recert.host_environment(8)
    environment["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    host = base.ReadyProcess(
        command, environment, output_dir / "host.stdout.bin", output_dir / "host.stderr.bin",
        b"PERSISTENT_DRIVER_READY ", ("stderr",),
    )
    host.wait_ready(base.PREPARE_TIMEOUT_S)
    return host, command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        raise GateError("physical execution requires --run")
    if RESULTS.exists():
        raise GateError("result directory already exists")
    RESULTS.mkdir(parents=True)
    phone = None
    host = None
    thermal = None
    try:
        base.adb("shell", "pkill -9 llama-layersplit")
        boot_id, artifacts = deploy()
        prompt = "Explain batching."
        reference = recert.reference_tokens(prompt, RESULTS)
        thermal_paths, thermal_start = base.discover_thermal_paths()
        thermal = base.ThermalMonitor(
            RESULTS / "thermal.log", RESULTS / "thermal.stderr.bin", thermal_paths,
        )
        thermal.wait_ready()
        phone, phone_command = prepare_phone(RESULTS)
        host, host_command = prepare_host(RESULTS)
        host_pid = host.process.pid
        sessions = []
        worker_identity = None
        previous_steps = 0
        for launch_id in range(1, SESSIONS + 1):
            end = "STOP" if launch_id == SESSIONS else "DETACH"
            command = {
                "schema": "layersplit-persistent-command-v1",
                "launch_id": launch_id,
                "prompt": prompt,
                "n_gen": N_GEN,
                "request_count": BATCH,
                "session_end": end,
            }
            if host.process.stdin is None:
                raise GateError("persistent host stdin is unavailable")
            start_ns = time.monotonic_ns()
            host.process.stdin.write(canonical(command))
            host.process.stdin.flush()
            results = wait_results(
                RESULTS / "host.stdout.bin", launch_id, 8.0,
            )
            result = results[launch_id - 1]
            validate_result(result, launch_id, end, host_pid, reference)
            markers = wait_objects(
                RESULTS / "host.stderr.bin", b"PERSISTENT_DRIVER_EXCHANGE_END ",
                launch_id, 2.0,
            )
            if markers[launch_id - 1] != {"launch_id": launch_id}:
                raise GateError("persistent host exchange marker mismatch")
            host_certs = wait_objects(
                RESULTS / "host.stderr.bin", b"PLACEMENTCERT ", launch_id, 2.0,
            )
            validate_host_cert(host_certs[launch_id - 1], host_pid)
            phone_certs = prior.wait_for_certs(
                RESULTS / "phone.stderr.bin", launch_id,
            )
            cert = phone_certs[launch_id - 1]
            worker_identity = prior.validate_session_cert(
                cert, launch_id, end, boot_id, worker_identity,
            )
            if cert["steps_total"] <= previous_steps:
                raise GateError("worker step counter did not advance")
            previous_steps = cert["steps_total"]
            sessions.append({
                "launch_id": launch_id,
                "session_end": end,
                "wall_us": (time.monotonic_ns() - start_ns) // 1000,
                "result": result,
                "host_placement": host_certs[launch_id - 1],
                "phone_session": cert,
            })
        if host.process.stdin is not None:
            host.process.stdin.close()
        host_rc = host.wait(20)
        host = None
        phone_rc = phone.wait(20)
        phone = None
        thermal_end = thermal.snapshot()
        if host_rc != 0 or phone_rc != 0:
            raise GateError("persistent host or phone did not stop cleanly")
        if not base.thermal_ok(thermal_end, base.THERMAL_END_MAX_MILLIC):
            raise GateError("phone thermal end gate failed")
        report = {
            "schema": "s15-persistent-host-tail-gate-v1",
            "verdict": "PERSISTENT_OP15_B32_HOST_AND_PHONE_PASS_ENERGY_UNKNOWN",
            "scope": "REAL_OP15_A6000_TWO_EXCHANGES_SYNTHETIC_PAYLOAD_ENERGY_UNKNOWN",
            "artifacts": artifacts,
            "boot_id": boot_id,
            "host_pid": host_pid,
            "resident_worker_pid": worker_identity[0],
            "resident_worker_nonce": worker_identity[1],
            "phone_command": phone_command,
            "host_command": host_command,
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            "reference_tokens": reference,
            "sessions": sessions,
            "problems": [],
        }
        REPORT.write_bytes(canonical(report))
        print(json.dumps({
            "verdict": report["verdict"],
            "host_pid": host_pid,
            "worker_pid": worker_identity[0],
            "max_elapsed_us": max(value["result"]["elapsed_us"] for value in sessions),
        }, sort_keys=True))
        return 0
    finally:
        if host is not None:
            host.terminate()
        if phone is not None:
            phone.terminate()
        if thermal is not None:
            thermal.stop()
        base.adb("forward", "--remove", f"tcp:{PORT}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, base.LauncherError, prior.GateError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
