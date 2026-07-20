#!/usr/bin/env python3
"""Run seven exact B32 sessions against one resident OP15 worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LIVE = HERE.parent / "s15_live_launcher"
ARRIVAL = HERE.parent / "s15_arrival_faithful_b32"
for path in (LIVE, ARRIVAL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_launcher as base  # noqa: E402
import run_recertification as recert  # noqa: E402


ANDROID_DIR = ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin"
ANDROID_BIN = ANDROID_DIR / "llama-layersplit"
HOST_BIN = ROOT / "build-cuda/bin/llama-layersplit"
SOURCE = ROOT / "examples/layersplit/layersplit.cpp"
REMOTE = "/data/local/tmp/ls-s15-persistent-b32"
PORT = 5961
SESSIONS = 7
RESULTS = HERE / "results"
REPORT = RESULTS / "report.json"
SESSION_KEYS = {
    "schema", "proto_version", "session_id", "session_end", "expected_backend",
    "worker_pid", "worker_boot_nonce", "device_boot_id", "layer_start", "layer_end",
    "n_layer", "steps_session", "steps_total", "reset_applied",
    "missing_buffer_compute_nodes", "compute_by_op_and_buffer", "placement_status",
}


class GateError(RuntimeError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def validate_session_cert(cert: dict, session_id: int, end: str,
                          boot_id: str, identity: tuple[int, str] | None) -> tuple[int, str]:
    if type(cert) is not dict or set(cert) != SESSION_KEYS:
        raise GateError("session certificate has an invalid key set")
    expected = {
        "schema": "ls-stagenet-session-v2",
        "proto_version": 2,
        "session_id": session_id,
        "session_end": end,
        "expected_backend": "HTP0",
        "device_boot_id": boot_id,
        "layer_start": 0,
        "layer_end": 8,
        "n_layer": 48,
        "reset_applied": end == "DETACH",
        "missing_buffer_compute_nodes": 0,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected.items():
        if type(cert.get(key)) is not type(value) or cert.get(key) != value:
            raise GateError(f"session certificate mismatch: {key}")
    for key in ("worker_pid", "steps_session", "steps_total"):
        if type(cert[key]) is not int or cert[key] <= 0:
            raise GateError(f"session certificate has invalid {key}")
    nonce = cert["worker_boot_nonce"]
    if type(nonce) is not str or not nonce:
        raise GateError("session certificate has invalid worker nonce")
    current_identity = (cert["worker_pid"], nonce)
    if identity is not None and current_identity != identity:
        raise GateError("resident worker identity changed")
    placement = cert["compute_by_op_and_buffer"]
    if type(placement) is not dict or not placement:
        raise GateError("session placement is empty")
    for op, buffers in placement.items():
        if type(op) is not str or not op or type(buffers) is not dict or not buffers:
            raise GateError("session placement entry is invalid")
        for buffer_name, count in buffers.items():
            if type(buffer_name) is not str or type(count) is not int or count <= 0:
                raise GateError("session placement count is invalid")
            if buffer_name != "HTP0" and not (op == "GET_ROWS" and buffer_name == "CPU"):
                raise GateError("session placement contains undeclared fallback")
    return current_identity


def session_certs(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return base.prefixed_objects(path.read_bytes(), b"SESSIONCERT ")


def wait_for_certs(path: Path, count: int, timeout_s: float = 10) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        values = session_certs(path)
        if len(values) >= count:
            return values
        time.sleep(0.05)
    raise GateError(f"timed out waiting for session certificate {count}")


def deploy() -> tuple[str, dict[str, str]]:
    if not ANDROID_BIN.is_file() or not HOST_BIN.is_file() or not SOURCE.is_file():
        raise GateError("current build artifacts are missing")
    base.adb_checked("shell", f"rm -rf {REMOTE} && cp -a {base.REMOTE} {REMOTE}")
    process = base.adb("push", str(ANDROID_BIN), f"{REMOTE}/llama-layersplit", timeout=240)
    if process.returncode != 0:
        raise GateError("failed to deploy current Android worker")
    base.adb_checked("shell", f"chmod 755 {REMOTE}/llama-layersplit")
    remote_digest = base.adb_checked(
        "shell", f"sha256sum {REMOTE}/llama-layersplit"
    ).decode("ascii").split()[0]
    if "sha256:" + remote_digest != digest(ANDROID_BIN):
        raise GateError("deployed worker digest mismatch")
    shard_digest = base.adb_checked("shell", f"sha256sum {base.SHARD}").decode("ascii").split()[0]
    if shard_digest != base.SHARD_SHA256:
        raise GateError("phone shard digest mismatch")
    boot_id = base.adb_checked("shell", "cat /proc/sys/kernel/random/boot_id").decode("ascii").strip()
    return boot_id, {
        "android_worker": digest(ANDROID_BIN),
        "host_worker": digest(HOST_BIN),
        "source": digest(SOURCE),
        "phone_shard": "sha256:" + shard_digest,
    }


def prepare_phone(output_dir: Path) -> tuple[base.ReadyProcess, list[str]]:
    base.adb("forward", "--remove", f"tcp:{PORT}")
    if base.adb("forward", f"tcp:{PORT}", f"tcp:{PORT}").returncode != 0:
        raise GateError("failed to install OP15 port forward")
    shell = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=4192 LLAMA_LAYER_END=8 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {base.SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {base.N_GEN} --driver-batch {base.BATCH} "
        f"--driver-context {base.CONTEXT} --driver-max-prefill {base.MAX_PREFILL}"
    )
    command = ["adb", "-s", base.SERIAL, "shell", shell]
    phone = base.ReadyProcess(
        command, None, output_dir / "phone.stdout.bin", output_dir / "phone.stderr.bin",
        b"[stagenet] listening", ("stdout", "stderr"),
    )
    phone.wait_ready(base.PREPARE_TIMEOUT_S)
    return phone, command


def run_host_session(index: int, end: str, prompt: str, reference: list[int]) -> dict:
    output_dir = RESULTS / f"session-{index}"
    output_dir.mkdir()
    command = [
        str(HOST_BIN), "-m", str(base.FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(PORT),
        "--prompt-after-load", "-n", str(base.N_GEN),
        "--driver-requests", str(base.BATCH), "--driver-warmup", "0",
        "--driver-batch", str(base.BATCH), "--driver-context", str(base.CONTEXT),
        "--driver-max-prefill", str(base.MAX_PREFILL), "--session-end", end.lower(),
    ]
    (output_dir / "command.json").write_bytes(canonical(command))
    host = base.ReadyProcess(
        command, recert.host_environment(8), output_dir / "host.stdout.bin",
        output_dir / "host.stderr.bin", b"DRIVER_INPUT_READY ",
    )
    try:
        host.wait_ready(base.PREPARE_TIMEOUT_S)
        watcher = recert.ExitWatch(host)
        if host.process.stdin is None:
            raise GateError("prepared host stdin is unavailable")
        start_ns = time.monotonic_ns()
        host.process.stdin.write((prompt + "\n").encode("utf-8"))
        host.process.stdin.flush()
        host.process.stdin.close()
        returncode, exit_ns = watcher.finish(4.5)
        host.wait(5)
        elapsed_us = (exit_ns - start_ns) // 1000
        stderr = (output_dir / "host.stderr.bin").read_bytes()
        rows = base.prefixed_objects(stderr, b"ROUTEJSON ")
        if returncode != 0 or len(rows) != base.BATCH \
                or sorted(row.get("stream_index") for row in rows) != list(range(base.BATCH)) \
                or any(row.get("status") != "ok" or row.get("batch_size") != base.BATCH
                       or row.get("token_ids") != reference for row in rows):
            raise GateError(f"host session {index} failed exact B32 output")
        if elapsed_us > 4_000_000:
            raise GateError(f"host session {index} exceeded the post-launch budget")
        return {
            "session_id": index,
            "session_end": end,
            "elapsed_us": elapsed_us,
            "route_wall_us_max": max(row["request_wall_us"] for row in rows),
            "token_ids": reference,
        }
    except Exception:
        host.terminate()
        raise


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
        sessions = []
        identity = None
        previous_steps = 0
        for index in range(1, SESSIONS + 1):
            end = "STOP" if index == SESSIONS else "DETACH"
            sessions.append(run_host_session(index, end, prompt, reference))
            certs = wait_for_certs(RESULTS / "phone.stderr.bin", index)
            cert = certs[index - 1]
            identity = validate_session_cert(cert, index, end, boot_id, identity)
            if cert["steps_total"] <= previous_steps:
                raise GateError("worker step counter did not advance")
            previous_steps = cert["steps_total"]
            sessions[-1]["certificate"] = cert
        phone_rc = phone.wait(20)
        phone = None
        thermal_end = thermal.snapshot()
        if phone_rc != 0 or not base.thermal_ok(thermal_end, base.THERMAL_END_MAX_MILLIC):
            raise GateError("phone exit or thermal gate failed")
        report = {
            "schema": "s15-persistent-b32-gate-v1",
            "verdict": "PERSISTENT_OP15_B32_MECHANICS_PASS_ENERGY_UNKNOWN",
            "scope": "REAL_OP15_A6000_SEVEN_SESSIONS_SYNTHETIC_PAYLOAD_ENERGY_UNKNOWN",
            "artifacts": artifacts,
            "boot_id": boot_id,
            "phone_command": phone_command,
            "resident_worker_pid": identity[0],
            "resident_worker_nonce": identity[1],
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            "sessions": sessions,
            "problems": [],
        }
        REPORT.write_bytes(canonical(report))
        print(json.dumps({
            "verdict": report["verdict"],
            "max_elapsed_us": max(value["elapsed_us"] for value in sessions),
            "worker_pid": identity[0],
        }, sort_keys=True))
        return 0
    finally:
        if phone is not None:
            phone.terminate()
        if thermal is not None:
            thermal.stop()
        base.adb("forward", "--remove", f"tcp:{PORT}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
