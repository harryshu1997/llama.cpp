#!/usr/bin/env python3
"""Corrected real-device persistence gate for two phone heads and one CUDA tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
HOST = ART / "host"
ANDROID = ART / "android"
RESULTS = HERE / "results"
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf"
REMOTE = "/data/local/tmp/ls-s14-persistent-v2"
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
PROMPT = "Explain in one sentence why the sky is blue."
PHONES = (
    {"name": "op15", "serial": "3C15AU002CL00000", "port": 5911, "skel": "libggml-htp-v81.so"},
    {"name": "op12", "serial": "5ae7a43d", "port": 5912, "skel": "libggml-htp-v75.so"},
)
K = 6
N_LAYER = 48
N_SESSIONS = 7
N_GEN = 16
MBUF = 3336
START_THERMAL_LIMIT = 60_000
END_THERMAL_LIMIT = 85_000
MAX_COV = 0.05
DECLARED_CPU_OPS = {"GET_ROWS"}


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(raw: str) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=no_duplicates)
    if type(value) is not dict:
        raise GateError("JSON record is not an object")
    return value


def records(text: str, prefix: str) -> list[dict]:
    result = []
    marker = prefix + " "
    for line in text.splitlines():
        if line.startswith(marker):
            result.append(strict_json(line[len(marker):]))
    return result


def adb(serial: str, *args: str, check: bool = True, timeout: int = 180):
    return subprocess.run(
        ["adb", "-s", serial, *args], capture_output=True, text=True,
        check=check, timeout=timeout,
    )


def verify_manifest() -> dict[str, str]:
    manifest_path = ART / "SHA256SUMS.txt"
    expected = {}
    for line in manifest_path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GateError("invalid artifact manifest")
        expected[relative] = digest
    if not expected:
        raise GateError("empty artifact manifest")
    for relative, digest in expected.items():
        path = ART / relative
        if not path.is_file() or sha256(path) != digest:
            raise GateError(f"artifact digest mismatch: {relative}")
    return expected


def thermal_snapshot(serial: str) -> dict:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); t=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty=$t;; esac; done"
    )
    process = adb(serial, "shell", command, check=False)
    sensors = {}
    for line in process.stdout.splitlines():
        if "=" not in line:
            continue
        name, raw = line.split("=", 1)
        try:
            value = int(raw)
        except ValueError:
            continue
        if 10_000 <= value <= 120_000:
            sensors[name] = value
    return {
        "valid": process.returncode == 0 and bool(sensors),
        "sensors_millic": dict(sorted(sensors.items())),
        "max_millic": max(sensors.values()) if sensors else None,
    }


def thermal_ok(value: dict, limit: int) -> bool:
    sensors = value.get("sensors_millic")
    maximum = value.get("max_millic")
    return value.get("valid") is True and type(sensors) is dict and bool(sensors) \
        and type(maximum) is int and maximum == max(sensors.values()) and maximum <= limit


def deploy(phone: dict) -> dict:
    serial = phone["serial"]
    adb(serial, "shell", f"rm -rf {shlex.quote(REMOTE)} && mkdir -p {shlex.quote(REMOTE)}")
    names = [
        "llama-layersplit", "libggml-base.so", "libggml-cpu.so",
        "libggml-hexagon.so", "libggml-opencl.so", "libggml.so",
        "libllama-common.so", "libllama.so", "libc++_shared.so", phone["skel"],
    ]
    hashes = {}
    for name in names:
        local = ANDROID / name
        adb(serial, "push", str(local), f"{REMOTE}/{name}")
        remote = adb(serial, "shell", f"sha256sum {shlex.quote(REMOTE + '/' + name)}").stdout.split()[0]
        expected = sha256(local)
        if remote != expected:
            raise GateError(f"{phone['name']}: deployed digest mismatch for {name}")
        hashes[name] = remote
    adb(serial, "shell", f"chmod 755 {REMOTE}/llama-layersplit")
    shard_hash = adb(serial, "shell", f"sha256sum {shlex.quote(SHARD)}").stdout.split()[0]
    boot_id = adb(serial, "shell", "cat /proc/sys/kernel/random/boot_id").stdout.strip()
    return {"files": hashes, "shard_sha256": shard_hash, "device_boot_id": boot_id}


def host_env(layer_start: int | None) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = str(HOST)
    if layer_start is None:
        env.pop("LLAMA_LAYER_START", None)
    else:
        env["LLAMA_LAYER_START"] = str(layer_start)
    env.pop("LLAMA_LAYER_END", None)
    return env


def mono_reference(log_path: Path) -> list[int]:
    command = [
        str(HOST / "llama-layersplit"), "-m", str(FULL_MODEL), "-ngl", "99",
        "--mode", "monodriver", "-p", PROMPT, "-n", str(N_GEN),
        "--driver-requests", "1", "--driver-warmup", "0", "--driver-batch", "1",
        "--driver-context", "4096", "--driver-max-prefill", "512",
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, env=host_env(None), timeout=180,
    )
    log_path.write_text(process.stderr, encoding="utf-8", errors="replace")
    rows = records(process.stderr, "ROUTEJSON")
    if process.returncode != 0 or len(rows) != 1 or rows[0].get("status") != "ok":
        raise GateError("mono reference failed")
    tokens = rows[0].get("token_ids")
    if type(tokens) is not list or len(tokens) != N_GEN or any(type(token) is not int for token in tokens):
        raise GateError("mono reference tokens invalid")
    return tokens


def start_worker(phone: dict, log_path: Path):
    serial = phone["serial"]
    adb(serial, "forward", "--remove", f"tcp:{phone['port']}", check=False)
    adb(serial, "forward", f"tcp:{phone['port']}", f"tcp:{phone['port']}")
    adb(serial, "shell", "pkill -9 -f llama-layersplit", check=False)
    command = (
        f"cd {REMOTE} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        f"GGML_HEXAGON_MBUF={MBUF} LLAMA_LAYER_END={K} LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {SHARD} --devices HTP0 -ngl 99 --mode stagenet "
        f"--port {phone['port']} --driver-batch 1 --driver-context 4096 "
        f"--driver-max-prefill 512 -n {N_GEN}"
    )
    handle = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
        ["adb", "-s", serial, "shell", command],
        stdout=handle, stderr=subprocess.STDOUT, text=True,
    )
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        handle.flush()
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "[stagenet] listening" in text:
            return process, handle
        if process.poll() is not None:
            raise GateError(f"{phone['name']}: worker exited before listen")
        time.sleep(0.5)
    raise GateError(f"{phone['name']}: worker listen timeout")


def run_session(index: int, end: str, log_path: Path) -> dict:
    command = [
        str(HOST / "llama-layersplit"), "-m", str(FULL_MODEL), "-ngl", "99",
        "--mode", "pipedriver", "--parallel-heads", "--parallel-tail-batch", "1",
        "--driver-batch", "2", "--host", "127.0.0.1", "--port", str(PHONES[0]["port"]),
        "--port2", str(PHONES[1]["port"]), "-p", PROMPT, "-n", str(N_GEN),
        "--driver-requests", "2", "--driver-warmup", "0", "--driver-context", "4096",
        "--driver-max-prefill", "512", "--session-end", end,
    ]
    start_ns = time.monotonic_ns()
    process = subprocess.run(
        command, capture_output=True, text=True, env=host_env(K), timeout=400,
    )
    elapsed_us = (time.monotonic_ns() - start_ns) // 1000
    log_path.write_text(
        "=== CMD ===\n" + " ".join(shlex.quote(part) for part in command)
        + "\n=== STDOUT ===\n" + process.stdout + "\n=== STDERR ===\n" + process.stderr,
        encoding="utf-8", errors="replace",
    )
    return {
        "index": index,
        "session_end": end,
        "returncode": process.returncode,
        "elapsed_us": elapsed_us,
        "rows": records(process.stderr, "ROUTEJSON"),
        "log_sha256": sha256(log_path),
    }


def verify_certificate_set(phone: dict, certs: list[dict], boot_id: str) -> list[str]:
    problems = []
    if len(certs) != N_SESSIONS:
        return [f"{phone['name']}: expected 7 session certificates, got {len(certs)}"]
    expected_ends = ["DETACH"] * 6 + ["STOP"]
    expected_resets = [True] * 6 + [False]
    pids = set()
    nonces = set()
    placement_maps = []
    for index, cert in enumerate(certs, 1):
        if cert.get("schema") != "ls-stagenet-session-v2" or cert.get("proto_version") != 2:
            problems.append(f"{phone['name']} session {index}: protocol mismatch")
        if cert.get("session_id") != index or cert.get("session_end") != expected_ends[index - 1]:
            problems.append(f"{phone['name']} session {index}: sequence mismatch")
        if cert.get("reset_applied") is not expected_resets[index - 1]:
            problems.append(f"{phone['name']} session {index}: reset flag mismatch")
        if cert.get("device_boot_id") != boot_id or cert.get("layer_start") != 0 \
                or cert.get("layer_end") != K or cert.get("n_layer") != N_LAYER \
                or cert.get("expected_backend") != "HTP0":
            problems.append(f"{phone['name']} session {index}: device/range mismatch")
        if cert.get("steps_session") != 26 or cert.get("steps_total") != index * 26:
            problems.append(f"{phone['name']} session {index}: step accounting mismatch")
        missing = cert.get("missing_buffer_compute_nodes")
        if cert.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
                or type(missing) is not int or missing != 0:
            problems.append(f"{phone['name']} session {index}: placement status failed")
        mapping = cert.get("compute_by_op_and_buffer")
        htp_nodes = 0
        if type(mapping) is not dict or not mapping:
            problems.append(f"{phone['name']} session {index}: placement map missing")
            continue
        for op, buffers in mapping.items():
            if type(buffers) is not dict or not buffers:
                problems.append(f"{phone['name']} session {index}: invalid placement map")
                continue
            for backend, count in buffers.items():
                if type(count) is not int or count <= 0:
                    problems.append(f"{phone['name']} session {index}: invalid node count")
                elif backend == "HTP0":
                    htp_nodes += count
                elif backend != "CPU" or op not in DECLARED_CPU_OPS:
                    problems.append(f"{phone['name']} session {index}: undeclared {op}@{backend}")
        if htp_nodes == 0:
            problems.append(f"{phone['name']} session {index}: zero HTP compute")
        pids.add(cert.get("worker_pid"))
        nonces.add(cert.get("worker_boot_nonce"))
        placement_maps.append(mapping)
    if len(pids) != 1 or len(nonces) != 1:
        problems.append(f"{phone['name']}: resident worker identity changed")
    if len({json.dumps(value, sort_keys=True) for value in placement_maps}) != 1:
        problems.append(f"{phone['name']}: placement is not session-scoped/repeatable")
    return problems


def coefficient_of_variation(values: list[int]) -> float:
    return statistics.pstdev(values) / statistics.mean(values) if len(values) > 1 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="authorize the real-device acquisition")
    args = parser.parse_args()
    if not args.run:
        raise GateError("real-device acquisition requires --run")
    if RESULTS.exists() and any(RESULTS.iterdir()):
        raise GateError("results directory is not empty")
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest = verify_manifest()
    harness_before = sha256(Path(__file__))
    model_sha256 = sha256(FULL_MODEL)
    host_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    gpu_inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.splitlines()
    workers = []
    report = None
    try:
        deployments = {phone["name"]: deploy(phone) for phone in PHONES}
        thermals = {
            phone["name"]: {"start": thermal_snapshot(phone["serial"]), "end": None}
            for phone in PHONES
        }
        if not all(thermal_ok(value["start"], START_THERMAL_LIMIT) for value in thermals.values()):
            raise GateError("start thermal gate failed")
        reference = mono_reference(RESULTS / "mono_reference.log")
        for phone in PHONES:
            process, handle = start_worker(phone, RESULTS / f"worker_{phone['name']}.log")
            workers.append((phone, process, handle))
        sessions = []
        for index in range(1, N_SESSIONS + 1):
            sessions.append(run_session(
                index, "detach" if index < N_SESSIONS else "stop",
                RESULTS / f"host_session_{index}.log",
            ))
            time.sleep(1)
        time.sleep(2)
        problems = []
        route_walls = []
        for session in sessions:
            rows = session["rows"]
            if session["returncode"] != 0 or len(rows) != 2:
                problems.append(f"session {session['index']}: incomplete host result")
                continue
            if sorted(row.get("stream_index") for row in rows) != [0, 1]:
                problems.append(f"session {session['index']}: stream ownership mismatch")
            for row in rows:
                if row.get("status") != "ok" or row.get("token_ids") != reference:
                    problems.append(f"session {session['index']}: token mismatch")
            walls = [row.get("request_wall_us") for row in rows]
            if any(type(value) is not int or value <= 0 for value in walls):
                problems.append(f"session {session['index']}: invalid route timing")
            else:
                route_walls.append(max(walls))
        worker_records = {}
        for phone, process, handle in workers:
            handle.flush()
            text = (RESULTS / f"worker_{phone['name']}.log").read_text(
                encoding="utf-8", errors="replace",
            )
            certs = records(text, "SESSIONCERT")
            worker_records[phone["name"]] = certs
            problems.extend(verify_certificate_set(
                phone, certs, deployments[phone["name"]]["device_boot_id"],
            ))
            if process.poll() is None:
                problems.append(f"{phone['name']}: worker alive after STOP")
        for phone in PHONES:
            thermals[phone["name"]]["end"] = thermal_snapshot(phone["serial"])
            if not thermal_ok(thermals[phone["name"]]["end"], END_THERMAL_LIMIT):
                problems.append(f"{phone['name']}: end thermal gate failed")
        cov = coefficient_of_variation(route_walls) if len(route_walls) == N_SESSIONS else None
        if type(cov) is not float or not math.isfinite(cov) or cov > MAX_COV:
            rendered = "missing" if cov is None else f"{cov:.6f}"
            problems.append(f"route wall CoV {rendered} exceeds {MAX_COV:.2f}")
        if verify_manifest() != manifest or sha256(Path(__file__)) != harness_before:
            problems.append("artifact or harness bytes changed during acquisition")
        report = {
            "schema": "s15-persistence-gate-v2",
            "verdict": "PERSISTENT_SESSION_REAL_GATE_PASS" if not problems else "PERSISTENT_SESSION_REAL_GATE_FAIL",
            "certified": not problems,
            "scope": "REAL_DEVICE_CORRECTNESS_PLACEMENT_PERSISTENCE_LATENCY_ONLY_ENERGY_UNKNOWN",
            "problems": problems,
            "reference_tokens": reference,
            "sessions": sessions,
            "route_wall_us": route_walls,
            "route_wall_cov": cov,
            "thermal": thermals,
            "deployments": deployments,
            "model": {"path": str(FULL_MODEL), "sha256": model_sha256},
            "host_boot_id": host_boot_id,
            "gpu_inventory": gpu_inventory,
            "worker_certificates": worker_records,
            "artifact_manifest_sha256": sha256(ART / "SHA256SUMS.txt"),
            "harness_sha256": harness_before,
            "energy_scope": "UNKNOWN",
        }
        (RESULTS / "gate_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii",
        )
        print(json.dumps({"verdict": report["verdict"], "problems": problems}, indent=2))
        return 0 if not problems else 2
    finally:
        for phone, process, handle in workers:
            if process.poll() is None:
                process.kill()
            handle.close()
        for phone in PHONES:
            adb(phone["serial"], "shell", "pkill -9 -f llama-layersplit", check=False)
            adb(phone["serial"], "forward", "--remove", f"tcp:{phone['port']}", check=False)


if __name__ == "__main__":
    raise SystemExit(main())
