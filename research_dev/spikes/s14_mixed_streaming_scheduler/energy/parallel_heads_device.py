#!/usr/bin/env python3
"""Real two-phone parallel-head to one shared CUDA-tail correctness screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import cp_e_live_priority as CPE
import pipe3_device as P3
import stageb_headcert as SB


HERE = Path(__file__).resolve().parent
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
SECOND_GPU_UUID = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"
REMOTE_DIR = "/data/local/tmp/ls-s14-cpe"
REMOTE_BIN = "llama-layersplit"
K = 6
SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_result(host_rc: int, rows: list[dict[str, Any]], reference: list[int], tail_batch: int,
                    cert_op15: dict | None, cert_op12: dict | None,
                    host_cert: dict | None, thermals: dict[str, Any]) -> dict[str, bool]:
    return {
        "host_returncode_zero": host_rc == 0,
        "two_rows_complete": len(rows) == 2 and
            sorted(row.get("stream_index") for row in rows) == [0, 1],
        "tail_batch_matches": len(rows) == 2 and
            all(row.get("batch_size") == tail_batch for row in rows),
        "all_tokens_match_reference": len(rows) == 2 and
            all(row.get("token_ids") == reference for row in rows),
        "op15_cert_ok": not P3.validate_stage_cert(cert_op15, 0, K),
        "op12_cert_ok": not P3.validate_stage_cert(cert_op12, 0, K),
        "host_tail_cert_ok": _host_cert_ok(host_cert),
        "thermal_start_ok": all(
            SB.thermal_ok(thermals[device]["start"], SB.THERMAL_START_MAX_MILLIC)
            for device in ("op15", "op12")
        ),
        "thermal_end_ok": all(
            SB.thermal_ok(thermals[device]["end"], SB.THERMAL_END_MAX_MILLIC)
            for device in ("op15", "op12")
        ),
    }


def _host_cert_ok(cert: dict | None) -> bool:
    if type(cert) is not dict or cert.get("status") != "SCHEDULED_PLACEMENT_OK" \
            or cert.get("layer_start") != K or cert.get("layer_end") != SB.N_LAYER \
            or cert.get("missing_buffer_compute_nodes") != 0:
        return False
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict:
        return False
    cuda_nodes = 0
    for op, buffers in mapping.items():
        if type(buffers) is not dict:
            return False
        for buffer, count in buffers.items():
            if type(count) is not int or count <= 0:
                return False
            if buffer == "CUDA0":
                cuda_nodes += count
            elif op != "GET_ROWS":
                return False
    return cuda_nodes > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-op15", type=int, default=16715)
    parser.add_argument("--port-op12", type=int, default=16712)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--driver-context", type=int, default=512)
    parser.add_argument("--driver-max-prefill", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--tail-batch", type=int, choices=(1, 2), default=1)
    parser.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    parser.add_argument("--output", type=Path, default=HERE / "parallel_heads_result.json")
    args = parser.parse_args()
    if not args.run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                              for char in args.run_id):
        parser.error("--run-id must contain only ASCII letters, digits, '-' or '_'")
    if args.output.exists():
        parser.error("--output already exists")
    if args.requests <= 0 or args.requests % 2 != 0 or args.warmups < 0 \
            or args.n_gen <= 0 or args.driver_context <= 0 \
            or args.driver_max_prefill <= 0 or args.timeout <= 0:
        parser.error("invalid parallel-head shape")

    log_dir = HERE / "logs_parallel_heads" / args.run_id
    if log_dir.exists():
        parser.error("run log directory already exists")
    log_dir.mkdir(parents=True)
    paths = {
        "op15": log_dir / "op15.log",
        "op12": log_dir / "op12.log",
        "host": log_dir / "host.stderr",
        "host_stdout": log_dir / "host.stdout",
    }
    started_utc_us = time.time_ns() // 1000
    thermals = {
        "op15": {"start": SB.thermal_snapshot(P3.OP15_SERIAL), "end": None},
        "op12": {"start": SB.thermal_snapshot(P3.OP12_SERIAL), "end": None},
    }
    if not all(SB.thermal_ok(thermals[device]["start"], SB.THERMAL_START_MAX_MILLIC)
               for device in thermals):
        raise RuntimeError(f"invalid start thermals: {thermals}")
    reference = SB.run_mono_reference(
        GPU_UUID, args.prompt, args.n_gen, args.timeout, batch=args.tail_batch,
        ctx=args.driver_context, max_prefill=args.driver_max_prefill,
    )

    proc15 = handle15 = proc12 = handle12 = None
    host: subprocess.CompletedProcess | None = None
    host_stderr = ""
    host_stdout = ""
    failure: str | None = None
    command: list[str] = []
    try:
        proc15, handle15 = P3.start_stage(
            P3.OP15_SERIAL, SHARD, 0, K, args.port_op15, 1, args.n_gen,
            args.driver_context, args.driver_max_prefill, SB.mbuf_for_k(K),
            paths["op15"], REMOTE_DIR, REMOTE_BIN, args.timeout,
        )
        proc12, handle12 = P3.start_stage(
            P3.OP12_SERIAL, SHARD, 0, K, args.port_op12, 1, args.n_gen,
            args.driver_context, args.driver_max_prefill, SB.mbuf_for_k(K),
            paths["op12"], REMOTE_DIR, REMOTE_BIN, args.timeout,
        )
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
        env["LD_LIBRARY_PATH"] = str(Path(SB.HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["LLAMA_LAYER_START"] = str(K)
        env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
        env.pop("LLAMA_LAYER_END", None)
        command = [
            SB.HOST_BIN, "-m", SB.FULL_MODEL, "-ngl", "99", "--mode", "pipedriver",
            "--host", "127.0.0.1", "--port", str(args.port_op15),
            "--port2", str(args.port_op12), "--parallel-heads",
            "--parallel-tail-batch", str(args.tail_batch),
            "-p", args.prompt, "-n", str(args.n_gen),
            "--driver-requests", str(args.requests), "--driver-warmup", str(args.warmups),
            "--driver-batch", "2", "--driver-context", str(args.driver_context),
            "--driver-max-prefill", str(args.driver_max_prefill),
        ]
        try:
            host = subprocess.run(command, capture_output=True, text=True, env=env, timeout=args.timeout)
            host_stderr = host.stderr
            host_stdout = host.stdout
        except subprocess.TimeoutExpired as exc:
            raw_stderr = exc.stderr or ""
            raw_stdout = exc.stdout or ""
            host_stderr = raw_stderr.decode(errors="replace") if isinstance(raw_stderr, bytes) else raw_stderr
            host_stdout = raw_stdout.decode(errors="replace") if isinstance(raw_stdout, bytes) else raw_stdout
            failure = f"host driver timed out after {args.timeout} seconds"
        paths["host"].write_text(host_stderr, encoding="utf-8", errors="replace")
        paths["host_stdout"].write_text(host_stdout, encoding="utf-8", errors="replace")
    finally:
        if proc12 is not None:
            P3.stop_stage(P3.OP12_SERIAL, proc12, handle12, args.port_op12, REMOTE_BIN)
        if proc15 is not None:
            P3.stop_stage(P3.OP15_SERIAL, proc15, handle15, args.port_op15, REMOTE_BIN)
        thermals["op15"]["end"] = SB.thermal_snapshot(P3.OP15_SERIAL)
        thermals["op12"]["end"] = SB.thermal_snapshot(P3.OP12_SERIAL)

    phone_text = {
        device: paths[device].read_text(encoding="utf-8", errors="replace")
        for device in ("op15", "op12")
    }
    rows = SB.parse_routejson(host_stderr)
    certs = {
        "op15": SB.parse_placement_cert(phone_text["op15"]),
        "op12": SB.parse_placement_cert(phone_text["op12"]),
        "host": SB.parse_placement_cert(host_stderr),
    }
    host_returncode = host.returncode if host is not None else -1
    checks = validate_result(host_returncode, rows, reference, args.tail_batch, certs["op15"],
                             certs["op12"], certs["host"], thermals)
    passed = failure is None and all(checks.values())
    artifacts = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    artifacts.update({
        "host_binary": {"path": SB.HOST_BIN, "sha256": sha256_file(Path(SB.HOST_BIN))},
        "phone_binary_op15": {"path": f"{REMOTE_DIR}/{REMOTE_BIN}",
                              "sha256": SB.sha256_remote(P3.OP15_SERIAL, f"{REMOTE_DIR}/{REMOTE_BIN}")},
        "phone_binary_op12": {"path": f"{REMOTE_DIR}/{REMOTE_BIN}",
                              "sha256": SB.sha256_remote(P3.OP12_SERIAL, f"{REMOTE_DIR}/{REMOTE_BIN}")},
        "shard_op15": {"path": SHARD, "sha256": SB.sha256_remote(P3.OP15_SERIAL, SHARD)},
        "shard_op12": {"path": SHARD, "sha256": SB.sha256_remote(P3.OP12_SERIAL, SHARD)},
        "harness": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
    })
    result = {
        "schema": "s14-parallel-heads-device-v1",
        "status": "PARALLEL_HEADS_TOKEN_CORRECT_PASS" if passed else
                  ("PARALLEL_HEADS_FAIL_MEASUREMENT" if failure else "PARALLEL_HEADS_FAIL"),
        "formal_claim": "PARALLEL_HEADS_SHARED_TAIL_MECHANICS" if passed else "NONE",
        "scope": "OP15_AND_OP12_PARALLEL_HEADS_ONE_SHARED_A6000_TAIL_MECHANICS_ONLY_ENERGY_UNKNOWN",
        "selected_gpu_uuid": GPU_UUID,
        "second_gpu_uuid": SECOND_GPU_UUID,
        "devices": {
            "op15": {"serial": P3.OP15_SERIAL, "hexagon": "v81", "backend": "HTP0"},
            "op12": {"serial": P3.OP12_SERIAL, "hexagon": "v75", "backend": "HTP0"},
        },
        "model": {
            "id": "gemma-4-12b-it-f16",
            "version": "sha256:bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a",
        },
        "route": {"op15": [0, K], "op12": [0, K], "shared_server_tail": [K, SB.N_LAYER]},
        "head_batch_per_phone": 1,
        "shared_tail_batch": args.tail_batch,
        "workload": {
            "prompt": args.prompt,
            "requests": args.requests,
            "warmup_groups": args.warmups,
            "n_gen": args.n_gen,
            "driver_context": args.driver_context,
            "driver_max_prefill": args.driver_max_prefill,
        },
        "host_command": command,
        "reference_tokens_full_model": reference,
        "checks": checks,
        "rows": rows,
        "certificates": certs,
        "thermals": thermals,
        "artifacts": artifacts,
        "host_returncode": host_returncode,
        "error": failure,
        "energy_scope": "UNKNOWN",
        "run_id": args.run_id,
        "process_identity": {
            "host_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip(),
            "pid": os.getpid(),
            "started_utc_us": started_utc_us,
            "ended_utc_us": time.time_ns() // 1000,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
