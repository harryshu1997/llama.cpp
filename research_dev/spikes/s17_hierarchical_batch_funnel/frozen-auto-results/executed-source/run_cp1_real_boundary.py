#!/usr/bin/env python3
"""Feed real OP12/OP15 [0,6) boundaries into the S17 OP15 B64 middle gate."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import socket
import struct
import subprocess
import time
from array import array
from pathlib import Path
from typing import Any

import run_cp1_middle as base


HEAD_SHARD = "/data/local/tmp/ls-npu/12b-f16-head-0-6.gguf"
PHONE_CONFIG = {
    "op15": {
        "serial": "3C15AU002CL00000",
        "directory": "/data/local/tmp/ls-s15-persistent-typed",
        "port": 17515,
    },
    "op12": {
        "serial": "5ae7a43d",
        "directory": "/data/local/tmp/ls-s16-op12",
        "port": 17512,
    },
}


def adb(serial: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-s", serial, *args],
        check=check,
        text=True,
        capture_output=True,
    )


def remote_sha256(serial: str, path: str) -> str:
    words = adb(serial, "shell", "sha256sum", path).stdout.strip().split()
    if len(words) != 2 or len(words[0]) != 64:
        raise base.GateError(f"cannot hash {serial}:{path}")
    return words[0]


def start_head(name: str, log_path: Path, timeout: int) -> tuple[subprocess.Popen[Any], Any]:
    config = PHONE_CONFIG[name]
    serial = config["serial"]
    port = config["port"]
    directory = config["directory"]
    adb(serial, "forward", "--remove", f"tcp:{port}", check=False)
    adb(serial, "forward", f"tcp:{port}", f"tcp:{port}")
    command = (
        f"cd {shlex.quote(directory)} && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. "
        "GGML_HEXAGON_MBUF=3072 LLAMA_LAYER_START=0 LLAMA_LAYER_END=6 "
        "LAYERSPLIT_PLACEMENT_CERT=1 "
        f"./llama-layersplit -m {shlex.quote(HEAD_SHARD)} --devices HTP0 -ngl 99 "
        f"--mode stagenet --port {port} -n 1 --driver-batch 32 "
        "--driver-context 64 --driver-max-prefill 1"
    )
    handle = log_path.open("wb")
    process = subprocess.Popen(
        ["adb", "-s", serial, "shell", command],
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    try:
        base.wait_for_log(log_path, "[stagenet] listening", process, timeout)
    except Exception:
        handle.close()
        adb(serial, "forward", "--remove", f"tcp:{port}", check=False)
        raise
    return process, handle


def head_decode(sock: socket.socket, tokens: list[int]) -> tuple[array, int]:
    rows = len(tokens)
    if rows != 32:
        raise base.GateError("head capture requires B32")
    seq_ids = list(range(rows))
    positions = [0] * rows
    started = time.monotonic_ns()
    base.send_i32(sock, base.STAGE_BATCH_DECODE, rows, 0)
    sock.sendall(struct.pack("<" + "i" * rows, *seq_ids))
    sock.sendall(struct.pack("<" + "i" * rows, *positions))
    sock.sendall(struct.pack("<" + "i" * rows, *tokens))
    output_rows, output_embd = base.recv_i32(sock, 2)
    if output_rows != rows or output_embd != base.N_EMBD:
        raise base.GateError(f"invalid head output {output_rows}x{output_embd}")
    output = array("f")
    output.frombytes(base.recv_exact(sock, rows * base.N_EMBD * 4))
    return output, (time.monotonic_ns() - started) // 1000


def capture_head(name: str, output_dir: Path, timeout: int) -> dict[str, Any]:
    config = PHONE_CONFIG[name]
    log_path = output_dir / f"{name}-head.log"
    process, handle = start_head(name, log_path, timeout)
    tokens = [2 + (index * 131) % 4096 for index in range(32)]
    output = array("f")
    elapsed_us = 0
    error = None
    hello = None
    try:
        with socket.create_connection(("127.0.0.1", config["port"]), timeout=timeout) as sock:
            sock.settimeout(timeout)
            hello = base.stage_hello(sock, 32, 0, 6)
            base.stage_reset(sock)
            head_decode(sock, tokens)
            base.stage_reset(sock)
            output, elapsed_us = head_decode(sock, tokens)
            base.send_i32(sock, base.STAGE_STOP)
    except Exception as exc:
        error = str(exc)
        process.kill()
    try:
        returncode = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=5)
        error = error or "head worker did not terminate"
    handle.flush()
    handle.close()
    adb(config["serial"], "forward", "--remove", f"tcp:{config['port']}", check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    cert = base.parse_json_record(text, "PLACEMENTCERT ")
    output_path = output_dir / f"{name}-head.f32"
    if output:
        output_path.write_bytes(output.tobytes())
    return {
        "name": name,
        "tokens": tokens,
        "output": output,
        "elapsed_us": elapsed_us,
        "finite": base.finite(output),
        "returncode": returncode,
        "error": error,
        "hello": hello,
        "placement": cert,
        "placement_problems": base.placement_problems(cert, "HTP0", 0, 6),
        "binary_sha256": remote_sha256(
            config["serial"], f"{config['directory']}/llama-layersplit"
        ),
        "shard_sha256": remote_sha256(config["serial"], HEAD_SHARD),
    }


def public_head(head: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in head.items() if key != "output"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=base.HERE / "real-boundary-results")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--decode-no-fa", action="store_true")
    args = parser.parse_args()
    if args.reps < 3 or args.timeout <= 0:
        parser.error("invalid repetition count or timeout")
    if args.output_dir.exists():
        parser.error("--output-dir already exists")
    if not base.HOST_BIN.is_file() or not base.LOCAL_SHARD.is_file():
        parser.error("middle gate artifacts are missing")
    if base.remote_sha256(base.PHONE_SHARD) != base.sha256_file(base.LOCAL_SHARD):
        parser.error("middle shard digest mismatch")

    args.output_dir.mkdir(parents=True)
    source_dir = args.output_dir / "executed-source"
    source_dir.mkdir()
    harness_copy = source_dir / Path(__file__).name
    middle_harness_copy = source_dir / Path(base.__file__).name
    shutil.copyfile(Path(__file__).resolve(), harness_copy)
    shutil.copyfile(Path(base.__file__).resolve(), middle_harness_copy)
    heads: dict[str, dict[str, Any]] = {}
    middles: dict[str, dict[str, Any]] = {}
    failure = None
    try:
        heads["op15"] = capture_head("op15", args.output_dir, args.timeout)
        heads["op12"] = capture_head("op12", args.output_dir, args.timeout)
        tokens = heads["op15"]["tokens"] + heads["op12"]["tokens"]
        residual = array("f", heads["op15"]["output"])
        residual.extend(heads["op12"]["output"])
        (args.output_dir / "combined-real-boundary.f32").write_bytes(residual.tobytes())
        (args.output_dir / "combined-tokens.json").write_text(
            json.dumps(tokens, separators=(",", ":")) + "\n", encoding="ascii"
        )
        middles["cuda"] = base.run_worker(
            "cuda", 17618, args.output_dir, args.reps, args.timeout, tokens, residual,
            args.decode_no_fa,
        )
        middles["htp"] = base.run_worker(
            "htp", 17617, args.output_dir, args.reps, args.timeout, tokens, residual,
            args.decode_no_fa,
        )
    except Exception as exc:
        failure = str(exc)

    htp = middles.get("htp")
    cuda = middles.get("cuda")
    cross_l2 = base.rel_l2(htp["b64_output"], cuda["b64_output"]) if htp and cuda else float("inf")
    cross_argmax = (
        base.row_argmax_mismatches(htp["b64_output"], cuda["b64_output"], 64)
        if htp and cuda else 64
    )
    checks = {
        "head_workers_complete": failure is None and len(heads) == 2 and all(
            head["returncode"] == 0 and head["error"] is None for head in heads.values()
        ),
        "head_placement_valid": len(heads) == 2 and all(
            not head["placement_problems"] for head in heads.values()
        ),
        "head_outputs_finite": len(heads) == 2 and all(head["finite"] for head in heads.values()),
        "middle_workers_complete": failure is None and len(middles) == 2 and all(
            middle["returncode"] == 0 and middle["error"] is None for middle in middles.values()
        ),
        "middle_placement_valid": len(middles) == 2 and all(
            not middle["placement_problems"] for middle in middles.values()
        ),
        "middle_outputs_finite": len(middles) == 2 and all(
            middle["finite"] for middle in middles.values()
        ),
        "middle_repeat_stable": len(middles) == 2 and all(
            middle["repeat_rel_l2_max"] <= 5e-3 for middle in middles.values()
        ),
        "htp_b64_vs_2xb32": htp is not None and
            htp["b64_vs_split_rel_l2"] <= 5e-3 and
            htp["b64_vs_split_argmax_mismatches"] == 0,
        "htp_vs_cuda_b64": cross_l2 <= 5e-3 and cross_argmax == 0,
        "htp_coalescing_faster": htp is not None and htp["coalesce_speedup"] >= 1.0,
        "head_shards_identical": len(heads) == 2 and
            len({head["shard_sha256"] for head in heads.values()}) == 1,
    }
    passed = all(checks.values())
    paths = [path for path in args.output_dir.iterdir() if path.is_file()]
    result = {
        "schema": "s17-real-boundary-middle-screen-v1",
        "status": "CP1_REAL_BOUNDARY_PASS" if passed else "CP1_REAL_BOUNDARY_FAIL",
        "scope": "REAL_PHONE_BOUNDARY_AND_OP15_MIDDLE_MECHANICS_NO_PIPELINE_OR_ENERGY_CLAIM",
        "attention_route": "explicit" if args.decode_no_fa else "auto",
        "checks": checks,
        "heads": {name: public_head(value) for name, value in heads.items()},
        "middles": {name: base.public_worker(value) for name, value in middles.items()},
        "cross_backend": {
            "b64_rel_l2": base.json_number(cross_l2),
            "b64_argmax_mismatches": cross_argmax,
        },
        "identity": {
            "harness": base.artifact(harness_copy),
            "middle_harness": base.artifact(middle_harness_copy),
            "host_binary": base.artifact(base.HOST_BIN),
            "middle_shard": base.artifact(base.LOCAL_SHARD),
        },
        "artifacts": {path.name: base.artifact(path) for path in paths},
        "failure": failure,
        "phone_energy": "UNKNOWN",
        "total_system_energy": "UNKNOWN",
    }
    result = base.json_safe(result)
    report = args.output_dir / "report.json"
    report.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
