#!/usr/bin/env python3
"""Independent real-device B32 screen for the two S15 phone lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S14_ENERGY = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/energy"
PERSIST = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/persistence_v2"
ART = PERSIST / "artifacts"
HOST = ART / "host"
ANDROID = ART / "android"
FULL_MODEL = Path("/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf")
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
REMOTE = "/data/local/tmp/ls-s15-b32"
PROMPT = "Explain batching."
BATCH = 32
N_GEN = 8
REQUESTS = 32
WARMUPS = 0
CONTEXT = 512
MAX_PREFILL = 64
MAX_COV = 0.05

sys.path.insert(0, str(S14_ENERGY))
import stageb_headcert as sb  # noqa: E402


DEVICES = (
    {
        "name": "op15",
        "serial": "3C15AU002CL00000",
        "k": 8,
        "port": 5931,
        "shard": "/data/local/tmp/ls-npu/12b-f16-head-0-8.gguf",
        "skel": "libggml-htp-v81.so",
    },
    {
        "name": "op12",
        "serial": "5ae7a43d",
        "k": 6,
        "port": 5932,
        "shard": "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf",
        "skel": "libggml-htp-v75.so",
    },
)


class GateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts() -> dict[str, str]:
    expected = {}
    manifest = ART / "SHA256SUMS.txt"
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in expected or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GateError("invalid frozen artifact manifest")
        expected[relative] = digest
    if not expected:
        raise GateError("empty frozen artifact manifest")
    for relative, digest in expected.items():
        path = ART / relative
        if not path.is_file() or sha256(path) != digest:
            raise GateError(f"frozen artifact mismatch: {relative}")
    return expected


def adb(serial: str, *args: str, check: bool = True, timeout: int = 240):
    return subprocess.run(
        ["adb", "-s", serial, *args], capture_output=True, text=True,
        check=check, timeout=timeout,
    )


def deploy(device: dict) -> dict:
    serial = device["serial"]
    adb(serial, "shell", f"rm -rf {shlex.quote(REMOTE)} && mkdir -p {shlex.quote(REMOTE)}")
    names = [
        "llama-layersplit", "libggml-base.so", "libggml-cpu.so",
        "libggml-hexagon.so", "libggml-opencl.so", "libggml.so",
        "libllama-common.so", "libllama.so", "libc++_shared.so", device["skel"],
    ]
    files = {}
    for name in names:
        local = ANDROID / name
        adb(serial, "push", str(local), f"{REMOTE}/{name}")
        remote = adb(serial, "shell", f"sha256sum {shlex.quote(REMOTE + '/' + name)}").stdout.split()[0]
        local_digest = sha256(local)
        if remote != local_digest:
            raise GateError(f"{device['name']}: deployed digest mismatch for {name}")
        files[name] = remote
    adb(serial, "shell", f"chmod 755 {REMOTE}/llama-layersplit")
    shard = adb(serial, "shell", f"sha256sum {shlex.quote(device['shard'])}").stdout.split()[0]
    boot_id = adb(serial, "shell", "cat /proc/sys/kernel/random/boot_id").stdout.strip()
    return {"files": files, "shard_sha256": shard, "device_boot_id": boot_id}


def same_batch_reference(timeout: int, results: Path) -> tuple[list[int], str]:
    sb.HOST_BIN = str(HOST / "llama-layersplit")
    sb.FULL_MODEL = str(FULL_MODEL)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = str(HOST)
    env.pop("LLAMA_LAYER_START", None)
    env.pop("LLAMA_LAYER_END", None)
    command = [
        sb.HOST_BIN, "-m", str(FULL_MODEL), "-ngl", "99", "--mode", "monodriver",
        "-p", PROMPT, "-n", str(N_GEN), "--driver-requests", str(BATCH),
        "--driver-warmup", "0", "--driver-batch", str(BATCH),
        "--driver-context", str(CONTEXT), "--driver-max-prefill", str(MAX_PREFILL),
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=timeout,
    )
    log_path = results / "cuda_b32_reference.log"
    log_path.write_text(
        "=== CMD ===\n" + " ".join(shlex.quote(value) for value in command)
        + "\n=== STDOUT ===\n" + process.stdout
        + "\n=== STDERR ===\n" + process.stderr,
        encoding="utf-8", errors="replace",
    )
    rows = sb.parse_routejson(process.stderr)
    if process.returncode != 0 or len(rows) != BATCH \
            or sorted(row.get("stream_index") for row in rows) != list(range(BATCH)):
        raise GateError("same-batch CUDA reference did not complete all streams")
    tokens = rows[0].get("token_ids")
    if any(row.get("token_ids") != tokens for row in rows):
        raise GateError("same-input CUDA reference streams disagree")
    return tokens, sha256(log_path)


def placement_ok(row: dict, reference: list[int]) -> bool:
    row["token_match_vs_mono"] = row.get("token_ids") == reference
    row["all_tokens_match_vs_mono"] = all(
        tokens == reference for tokens in row.get("token_ids_by_request", [])
    )
    return sb.row_is_eligible(row, REQUESTS)


def run_device(device: dict, process_index: int, reference: list[int], timeout: int,
               results: Path) -> dict:
    log_dir = results / f"{device['name']}_p{process_index}"
    log_dir.mkdir(parents=True)
    thermal_start = sb.thermal_snapshot(device["serial"])
    start_ns = time.monotonic_ns()
    try:
        row = sb.run_one_k(
            device["serial"], GPU_UUID, device["k"], device["port"],
            BATCH, N_GEN, REQUESTS, WARMUPS, CONTEXT, MAX_PREFILL, PROMPT,
            REMOTE, "llama-layersplit", log_dir, timeout,
        )
        error = None
    except Exception as exc:
        row = None
        error = f"{type(exc).__name__}: {exc}"
    elapsed_us = (time.monotonic_ns() - start_ns) // 1000
    thermal_end = sb.thermal_snapshot(device["serial"])
    eligible = type(row) is dict and placement_ok(row, reference)
    log_artifacts = {
        str(path.relative_to(results)): sha256(path)
        for path in sorted(log_dir.rglob("*"))
        if path.is_file()
    }
    return {
        "device": device["name"],
        "process_index": process_index,
        "eligible": eligible,
        "error": error,
        "elapsed_us": elapsed_us,
        "thermal_start": thermal_start,
        "thermal_end": thermal_end,
        "log_artifacts": log_artifacts,
        "row": row,
    }


def row_problem(record: dict) -> str | None:
    name = record["device"]
    index = record["process_index"]
    if record.get("error") is not None:
        return f"{name} p{index}: {record['error']}"
    if record.get("eligible") is not True:
        return f"{name} p{index}: correctness or placement gate failed"
    if not sb.thermal_ok(record["thermal_start"], sb.THERMAL_START_MAX_MILLIC):
        return f"{name} p{index}: start thermal gate failed"
    if not sb.thermal_ok(record["thermal_end"], sb.THERMAL_END_MAX_MILLIC):
        return f"{name} p{index}: end thermal gate failed"
    return None


def cleanup_devices() -> None:
    for device in DEVICES:
        for command in (
            ("shell", "pkill -9 -f llama-layersplit"),
            ("forward", "--remove", f"tcp:{device['port']}"),
        ):
            try:
                adb(device["serial"], *command, check=False, timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--processes", type=int, default=7)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    args = parser.parse_args()
    if not args.run:
        raise GateError("physical acquisition requires --run")
    if args.processes < 1 or args.processes > 7 or args.timeout < 1:
        raise GateError("processes must be 1..7 and timeout must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise GateError("output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = verify_artifacts()
    harness_before = sha256(Path(__file__))
    helper_before = sha256(Path(sb.__file__))
    model_digest = sha256(FULL_MODEL)
    deployments = {device["name"]: deploy(device) for device in DEVICES}
    reference, reference_log_sha256 = same_batch_reference(args.timeout, args.output_dir)
    if len(reference) != N_GEN or any(type(token) is not int for token in reference):
        raise GateError("same-batch CUDA reference is invalid")

    records = []
    problems = []
    for process_index in range(args.processes):
        for device in DEVICES:
            record = run_device(device, process_index, reference, args.timeout, args.output_dir)
            records.append(record)
            problem = row_problem(record)
            if problem is not None:
                problems.append(problem)
        time.sleep(1)

    profiles = {}
    for device in DEVICES:
        rows = [record for record in records if record["device"] == device["name"]]
        walls = [record["row"]["request_wall_us_p50"] for record in rows
                 if record.get("eligible") is True]
        cov = statistics.pstdev(walls) / statistics.mean(walls) if len(walls) > 1 else None
        if len(walls) != args.processes:
            problems.append(f"{device['name']}: incomplete eligible process set")
        if args.processes == 7 and (type(cov) is not float or cov > MAX_COV):
            problems.append(f"{device['name']}: process CoV failed")
        profiles[device["name"]] = {
            "n_processes": len(walls),
            "request_wall_us_p50_by_process": walls,
            "process_cov": cov,
        }

    if verify_artifacts() != manifest or sha256(Path(__file__)) != harness_before \
            or sha256(Path(sb.__file__)) != helper_before:
        problems.append("source or frozen artifacts changed during acquisition")
    full_gate = args.processes == 7
    certified = full_gate and not problems
    verdict = "B32_INDEPENDENT_PHONE_GATE_PASS" if certified else \
        "B32_SCREEN_PASS_NEEDS_7_PROCESSES" if not problems else "B32_INDEPENDENT_PHONE_GATE_FAIL"
    report = {
        "schema": "s15-independent-b32-gate-v1",
        "verdict": verdict,
        "certified": certified,
        "scope": "REAL_DEVICE_B32_CORRECTNESS_PLACEMENT_LATENCY_ONLY_ENERGY_UNKNOWN",
        "energy_scope": "UNKNOWN",
        "batch": BATCH,
        "n_gen": N_GEN,
        "requests_per_process": REQUESTS,
        "processes_requested": args.processes,
        "prompt": PROMPT,
        "reference_tokens": reference,
        "reference_log_sha256": reference_log_sha256,
        "problems": problems,
        "profiles": profiles,
        "records": records,
        "deployments": deployments,
        "model": {"path": str(FULL_MODEL), "sha256": model_digest},
        "gpu_uuid": GPU_UUID,
        "artifact_manifest_sha256": sha256(ART / "SHA256SUMS.txt"),
        "harness_sha256": harness_before,
        "stageb_helper_sha256": helper_before,
    }
    (args.output_dir / "gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii",
    )
    print(json.dumps({"verdict": verdict, "problems": problems}, indent=2))
    return 0 if not problems else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        cleanup_devices()
