#!/usr/bin/env python3
"""Run two real persistent B32 exchanges through OP12 and the CUDA tail."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MUX_DIR = HERE.parent / "s15_persistent_typed_gate"
ENERGY = HERE.parent / "s14_mixed_streaming_scheduler/energy"
S16 = HERE.parent / "s16_mixed_persistent_energy"
for path in (MUX_DIR, ENERGY, S16):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import physical_mux as MUX  # noqa: E402
import stageb_headcert as SB  # noqa: E402
from experiment_contract import (  # noqa: E402
    BATCH, N_GEN, canonical, digest_file, strict_object, validate_result_record,
)


SERIAL = "5ae7a43d"
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
HOST_BIN = ROOT / "build-cuda/bin/llama-layersplit"
ANDROID = ROOT / "npu-harness/build/llamacpp/android-arm64-hexagon-release-eafdc75e/bin"
ANDROID_BIN = ANDROID / "llama-layersplit"
REMOTE_BASE = "/data/local/tmp/ls-s15-b32"
REMOTE = "/data/local/tmp/ls-s16-op12"
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf"
SHARD_SHA256 = "d507b7bb453242dff12ba1ce0add53189755b8a9a2b960743ead1578d8f1a6b5"
PORT = 5965
SLO_US = 12_000_000
RESULTS = HERE / "results"
INPUT = HERE.parent / "s15_burst_cohort/input_manifest.json"
REFERENCE = HERE.parent / "s15_persistent_typed_gate/results/report.json"
RUNTIME_LIBS = (
    "libggml-base.so", "libggml-cpu.so", "libggml-hexagon.so",
    "libggml-opencl.so", "libggml.so", "libllama-common.so", "libllama.so",
)


class GateError(RuntimeError):
    pass


def adb(*args: str, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["adb", "-s", SERIAL, *args], capture_output=True, timeout=timeout,
    )
    if check and result.returncode != 0:
        raise GateError(result.stderr.decode("utf-8", errors="replace")[-1000:])
    return result


def deploy() -> dict[str, str]:
    adb("shell", f"rm -rf {REMOTE} && cp -a {REMOTE_BASE} {REMOTE}")
    files = ("llama-layersplit",) + RUNTIME_LIBS
    for name in files:
        local = ANDROID_BIN if name == "llama-layersplit" else ANDROID / name
        if not local.is_file():
            raise GateError(f"missing Android runtime {local}")
        adb("push", str(local), f"{REMOTE}/{name}", timeout=180)
        observed = adb("shell", f"sha256sum {REMOTE}/{name}").stdout.decode().split()[0]
        if "sha256:" + observed != digest_file(local):
            raise GateError(f"deployed digest mismatch for {name}")
    adb("shell", f"chmod 755 {REMOTE}/llama-layersplit")
    shard = adb("shell", f"sha256sum {SHARD}", timeout=180).stdout.decode().split()[0]
    if shard != SHARD_SHA256:
        raise GateError("OP12 shard digest mismatch")
    return {
        "android_binary": digest_file(ANDROID_BIN),
        "host_binary": digest_file(HOST_BIN),
        "full_model": digest_file(FULL_MODEL),
        "shard": "sha256:" + shard,
        "device_boot_id": adb("shell", "cat /proc/sys/kernel/random/boot_id").stdout.decode().strip(),
    }


def frozen_input() -> tuple[str, list[int]]:
    prompt = strict_object(INPUT.read_bytes(), "input manifest").get("prompt_text")
    reference = strict_object(REFERENCE.read_bytes(), "reference report", False).get("reference_tokens")
    if type(prompt) is not str or not prompt or type(reference) is not list \
            or len(reference) != N_GEN:
        raise GateError("frozen input is invalid")
    return prompt, reference


def phone_session(lines: list[bytes], launch_id: int, end: str) -> dict:
    matches = [line for line in lines if line.startswith(MUX.SESSION_PREFIX)]
    if len(matches) != 1:
        raise GateError("phone session certificate count failed")
    value = strict_object(matches[0][len(MUX.SESSION_PREFIX):], "phone session", False)
    if value.get("schema") != "ls-stagenet-session-v2" \
            or value.get("session_id") != launch_id or value.get("session_end") != end \
            or value.get("expected_backend") != "HTP0" \
            or value.get("layer_start") != 0 or value.get("layer_end") != 6 \
            or value.get("n_layer") != 48 \
            or value.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
            or value.get("missing_buffer_compute_nodes") != 0 \
            or value.get("reset_applied") is not (end == "DETACH"):
        raise GateError("phone session identity failed")
    by_op = value.get("compute_by_op_and_buffer")
    if type(by_op) is not dict or not by_op:
        raise GateError("phone session has no operation tally")
    for op_name, buffers in by_op.items():
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0 \
                    or (backend != "HTP0" and not (op_name == "GET_ROWS" and backend == "CPU")):
                raise GateError(f"undeclared phone placement {op_name}@{backend}")
    return value


def main() -> int:
    if RESULTS.exists():
        raise GateError("results already exist")
    RESULTS.mkdir()
    prompt, reference = frozen_input()
    artifacts = deploy()
    thermal_start = SB.thermal_snapshot(SERIAL)
    if not SB.thermal_ok(thermal_start, SB.THERMAL_START_MAX_MILLIC):
        raise GateError("OP12 start thermal gate failed")
    adb("shell", "pkill -9 llama-layersplit >/dev/null 2>&1 || true", check=False)
    adb("forward", "--remove", f"tcp:{PORT}", check=False)
    adb("forward", f"tcp:{PORT}", f"tcp:{PORT}")

    phone_shell = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=3336 LLAMA_LAYER_END=6 LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {PORT} -n {N_GEN} --driver-batch {BATCH} "
        "--driver-context 512 --driver-max-prefill 64"
    )
    phone_command = ("adb", "-s", SERIAL, "shell", phone_shell)
    host_command = (
        str(HOST_BIN), "-m", str(FULL_MODEL), "-ngl", "99", "--mode", "pipedriver",
        "--host", "127.0.0.1", "--port", str(PORT), "-n", str(N_GEN),
        "--driver-batch", str(BATCH), "--driver-context", "512",
        "--driver-max-prefill", "64", "--persistent-jsonl",
    )
    host_env = dict(os.environ)
    host_env.update({
        "CUDA_VISIBLE_DEVICES": GPU_UUID,
        "LD_LIBRARY_PATH": str(HOST_BIN.parent),
        "LLAMA_LAYER_START": "6",
        "LAYERSPLIT_PLACEMENT_CERT": "1",
    })
    phone = host = None
    try:
        phone = MUX.MonitoredProcess(phone_command, None)
        phone.wait_line("stderr", MUX.PHONE_READY_PREFIX, 300)
        host = MUX.MonitoredProcess(host_command, host_env)
        ready_line = host.wait_line("stderr", MUX.HOST_READY_PREFIX, 300)
        ready = strict_object(ready_line[len(MUX.HOST_READY_PREFIX):], "host readiness")
        if ready.get("batch_size") != BATCH or ready.get("max_n_gen") != N_GEN:
            raise GateError("host readiness failed")
        results = []
        sessions = []
        for launch_id in (1, 2):
            end = "DETACH" if launch_id == 1 else "STOP"
            payload = canonical({
                "schema": "layersplit-persistent-command-v1",
                "launch_id": launch_id,
                "prompt": prompt,
                "n_gen": N_GEN,
                "request_count": BATCH,
                "session_end": end,
            })
            host_start = host.snapshot()
            phone_start = phone.snapshot()
            host.send(payload)
            result_line, _, phone_stderr, _ = MUX.wait_exchange(
                host, phone, host_start, phone_start, launch_id, 20_000_000,
                ready["host_pid"], 6, 48, "CUDA0",
            )
            result = strict_object(result_line, "host result")
            validate_result_record(result, reference, "OP12", launch_id, end)
            if result["route_wall_us"] > SLO_US:
                raise GateError("OP12 lower-priority SLO failed")
            results.append(result)
            sessions.append(phone_session(phone_stderr, launch_id, end))
        if host.wait(20) != 0 or phone.wait(20) != 0:
            raise GateError("STOP did not terminate both processes")
        if len({item["host_pid"] for item in results}) != 1 \
                or len({item["worker_pid"] for item in sessions}) != 1 \
                or len({item["worker_boot_nonce"] for item in sessions}) != 1:
            raise GateError("persistent process identity changed")
        thermal_end = SB.thermal_snapshot(SERIAL)
        if not SB.thermal_ok(thermal_end, SB.THERMAL_END_MAX_MILLIC):
            raise GateError("OP12 end thermal gate failed")
        report = {
            "schema": "s16-op12-persistent-lane-v1",
            "status": "OP12_PERSISTENT_B32_LANE_PASS",
            "scope": "REAL_OP12_A6000_MECHANICS_ENERGY_UNKNOWN",
            "synthetic_slo_us": SLO_US,
            "prompt_sha256": "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reference_tokens": reference,
            "artifacts": artifacts,
            "results": results,
            "sessions": sessions,
            "thermal_start": thermal_start,
            "thermal_end": thermal_end,
            "energy": "UNKNOWN",
        }
        (RESULTS / "report.json").write_bytes(canonical(report))
        print(json.dumps({
            "status": report["status"],
            "route_wall_us": [item["route_wall_us"] for item in results],
            "host_pid": results[0]["host_pid"],
            "worker_pid": sessions[0]["worker_pid"],
        }, sort_keys=True))
        return 0
    finally:
        if host is not None:
            host.terminate()
            host.persist(RESULTS, "host")
        if phone is not None:
            phone.terminate()
            phone.persist(RESULTS, "phone")
        adb("forward", "--remove", f"tcp:{PORT}", check=False)
        adb("shell", "pkill -9 llama-layersplit >/dev/null 2>&1 || true", check=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateError, MUX.MuxError, OSError, ValueError) as exc:
        print(f"OP12_GATE_ERROR {exc}", file=sys.stderr)
        raise SystemExit(2)
