#!/usr/bin/env python3
"""Real OP15 [6,12) B64 middle-island screen with a matched CUDA control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import statistics
import struct
import subprocess
import time
from array import array
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HOST_BIN = REPO / "build-cuda/bin/llama-layersplit"
LOCAL_SHARD = Path("/home/myid/zs89458/Documents/models/12b-f16-mid-6-12.gguf")
GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
PHONE_SERIAL = "3C15AU002CL00000"
PHONE_DIR = "/data/local/tmp/ls-s15-persistent-typed"
PHONE_BIN = "llama-layersplit"
PHONE_SHARD = "/data/local/tmp/ls-npu/12b-f16-mid-6-12.gguf"
LAYER_START = 6
LAYER_END = 12
N_LAYER = 48
N_EMBD = 3840
STAGE_STOP = -1
STAGE_RESET = -2
STAGE_BATCH_DECODE = -5
STAGE_HELLO = -6
HELLO_MAGIC = 0x4C535432
HELLO_VERSION = 1
ALLOWED_CPU_OPS = {"GET_ROWS"}


class GateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adb(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["adb", "-s", PHONE_SERIAL, *args],
        check=check,
        text=True,
        capture_output=True,
    )


def remote_sha256(path: str) -> str:
    output = adb("shell", "sha256sum", path).stdout.strip().split()
    if len(output) != 2 or len(output[0]) != 64:
        raise GateError(f"cannot hash remote artifact {path}")
    return output[0]


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise GateError(f"short socket read: {len(data)} of {size}")
        data.extend(chunk)
    return bytes(data)


def send_i32(sock: socket.socket, *values: int) -> None:
    sock.sendall(struct.pack("<" + "i" * len(values), *values))


def recv_i32(sock: socket.socket, count: int = 1) -> tuple[int, ...]:
    return struct.unpack("<" + "i" * count, recv_exact(sock, 4 * count))


def make_inputs(batch: int) -> tuple[list[int], array]:
    tokens = [2 + (stream * 131) % 4096 for stream in range(batch)]
    residual = array("f")
    residual.extend(
        0.001 * (((column + stream) % 17) - 8)
        for stream in range(batch)
        for column in range(N_EMBD)
    )
    if residual.itemsize != 4:
        raise GateError("float array is not 32-bit")
    return tokens, residual


def stage_hello(
    sock: socket.socket,
    batch: int,
    layer_start: int = LAYER_START,
    layer_end: int = LAYER_END,
) -> dict[str, int]:
    send_i32(sock, STAGE_HELLO)
    words = recv_i32(sock, 10)
    keys = (
        "magic", "version", "layer_start", "layer_end", "n_layer", "n_embd",
        "n_seq_max", "n_ctx_seq", "n_batch", "n_ubatch",
    )
    hello = dict(zip(keys, words))
    expected = {
        "magic": HELLO_MAGIC,
        "version": HELLO_VERSION,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "n_layer": N_LAYER,
        "n_embd": N_EMBD,
        "n_seq_max": batch,
    }
    for key, value in expected.items():
        if hello[key] != value:
            raise GateError(f"HELLO {key}={hello[key]} expected {value}")
    if hello["n_batch"] < batch or hello["n_ubatch"] < batch:
        raise GateError("worker cannot accept the declared batch")
    return hello


def stage_reset(sock: socket.socket) -> None:
    send_i32(sock, STAGE_RESET)
    if recv_i32(sock)[0] != 0:
        raise GateError("stage reset failed")


def stage_decode(
    sock: socket.socket,
    seq_ids: list[int],
    tokens: list[int],
    residual: array,
) -> tuple[array, int]:
    rows = len(seq_ids)
    if rows <= 0 or len(tokens) != rows or len(residual) != rows * N_EMBD:
        raise GateError("invalid decode input shape")
    positions = [0] * rows
    started = time.monotonic_ns()
    send_i32(sock, STAGE_BATCH_DECODE, rows, N_EMBD)
    sock.sendall(struct.pack("<" + "i" * rows, *seq_ids))
    sock.sendall(struct.pack("<" + "i" * rows, *positions))
    sock.sendall(struct.pack("<" + "i" * rows, *tokens))
    sock.sendall(residual.tobytes())
    output_rows, output_embd = recv_i32(sock, 2)
    if output_rows != rows or output_embd != N_EMBD:
        raise GateError(f"invalid output shape {output_rows}x{output_embd}")
    output = array("f")
    output.frombytes(recv_exact(sock, rows * N_EMBD * 4))
    elapsed_us = (time.monotonic_ns() - started) // 1000
    return output, elapsed_us


def finite(values: array) -> bool:
    return all(math.isfinite(value) for value in values)


def rel_l2(values: array, reference: array) -> float:
    if len(values) != len(reference) or not values:
        return math.inf
    diff2 = 0.0
    ref2 = 0.0
    for value, ref in zip(values, reference):
        if not math.isfinite(value) or not math.isfinite(ref):
            return math.inf
        delta = float(value) - float(ref)
        diff2 += delta * delta
        ref2 += float(ref) * float(ref)
    if ref2 == 0.0:
        return 0.0 if diff2 == 0.0 else math.inf
    return math.sqrt(diff2 / ref2)


def row_argmax_mismatches(values: array, reference: array, rows: int) -> int:
    if len(values) != rows * N_EMBD or len(reference) != len(values):
        return rows
    mismatches = 0
    for row in range(rows):
        begin = row * N_EMBD
        end = begin + N_EMBD
        lhs = max(range(begin, end), key=values.__getitem__) - begin
        rhs = max(range(begin, end), key=reference.__getitem__) - begin
        mismatches += lhs != rhs
    return mismatches


def parse_json_record(text: str, prefix: str) -> dict[str, Any] | None:
    records = []
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                value = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                continue
            if type(value) is dict:
                records.append(value)
    return records[-1] if records else None


def placement_problems(
    cert: dict[str, Any] | None,
    backend: str,
    layer_start: int = LAYER_START,
    layer_end: int = LAYER_END,
) -> list[str]:
    if type(cert) is not dict:
        return ["missing placement certificate"]
    problems = []
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK":
        problems.append(f"status={cert.get('status')}")
    if cert.get("layer_start") != layer_start or cert.get("layer_end") != layer_end:
        problems.append("wrong layer range")
    if cert.get("missing_buffer_compute_nodes") != 0:
        problems.append("missing compute buffer")
    mapping = cert.get("compute_by_op_and_buffer")
    if type(mapping) is not dict:
        return problems + ["missing placement map"]
    backend_nodes = 0
    for op, buffers in mapping.items():
        if type(op) is not str or type(buffers) is not dict:
            problems.append("invalid placement map")
            continue
        for buffer, count in buffers.items():
            if type(buffer) is not str or type(count) is not int or count <= 0:
                problems.append("invalid placement count")
            elif backend in buffer:
                backend_nodes += count
            elif op not in ALLOWED_CPU_OPS:
                problems.append(f"undeclared placement {op}@{buffer}")
    if backend_nodes == 0:
        problems.append(f"no {backend} compute")
    return sorted(set(problems))


def wait_for_log(path: Path, marker: str, process: subprocess.Popen[Any], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and marker in path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
            raise GateError(f"worker exited before readiness: {text[-2000:]}")
        time.sleep(0.2)
    process.kill()
    raise GateError(f"worker readiness timeout after {timeout}s")


def thermal_snapshot() -> dict[str, Any]:
    command = (
        "for z in /sys/class/thermal/thermal_zone*; do "
        "ty=$(cat $z/type 2>/dev/null); v=$(cat $z/temp 2>/dev/null); "
        "case $ty in nsphmx-*) echo $ty=$v;; esac; done"
    )
    result = adb("shell", command, check=False)
    sensors = {}
    for line in result.stdout.splitlines():
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
        "sensor_class": "nsphmx-*",
        "sensors_millic": dict(sorted(sensors.items())),
        "maximum_millic": max(sensors.values()) if sensors else None,
        "valid": result.returncode == 0 and bool(sensors),
    }


def start_worker(
    kind: str,
    port: int,
    log_path: Path,
    timeout: int,
    decode_no_fa: bool = False,
) -> tuple[subprocess.Popen[Any], Any]:
    handle = log_path.open("wb")
    common = [
        "-m", str(LOCAL_SHARD if kind == "cuda" else PHONE_SHARD),
        "-ngl", "99", "--mode", "stagenet", "--port", str(port),
        "-n", "1", "--driver-batch", "64", "--driver-context", "64",
        "--driver-max-prefill", "1",
    ]
    if kind == "cuda":
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": GPU_UUID,
            "LD_LIBRARY_PATH": str(HOST_BIN.parent) + ":" + env.get("LD_LIBRARY_PATH", ""),
            "LLAMA_LAYER_START": str(LAYER_START),
            "LLAMA_LAYER_END": str(LAYER_END),
            "LAYERSPLIT_PLACEMENT_CERT": "1",
        })
        if decode_no_fa:
            env["GGML_DECODE_NO_FA"] = "1"
        else:
            env.pop("GGML_DECODE_NO_FA", None)
        process = subprocess.Popen(
            [str(HOST_BIN), *common, "--devices", "CUDA0"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )
    elif kind == "htp":
        adb("forward", "--remove", f"tcp:{port}", check=False)
        adb("forward", f"tcp:{port}", f"tcp:{port}")
        remote = [f"./{PHONE_BIN}", *common, "--devices", "HTP0"]
        exports = (
            "LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3584 "
            f"LLAMA_LAYER_START={LAYER_START} LLAMA_LAYER_END={LAYER_END} "
            "LAYERSPLIT_PLACEMENT_CERT=1"
        )
        if decode_no_fa:
            exports += " GGML_DECODE_NO_FA=1"
        command = f"cd {shlex.quote(PHONE_DIR)} && env {exports} " + " ".join(
            shlex.quote(part) for part in remote
        )
        process = subprocess.Popen(
            ["adb", "-s", PHONE_SERIAL, "shell", command],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    else:
        handle.close()
        raise GateError(f"unknown worker kind {kind}")
    try:
        wait_for_log(log_path, "[stagenet] listening", process, timeout)
    except Exception:
        handle.close()
        raise
    return process, handle


def run_worker(
    kind: str,
    port: int,
    output_dir: Path,
    reps: int,
    timeout: int,
    tokens_input: list[int] | None = None,
    residual_input: array | None = None,
    decode_no_fa: bool = False,
) -> dict[str, Any]:
    log_path = output_dir / f"{kind}.log"
    process, handle = start_worker(kind, port, log_path, timeout, decode_no_fa)
    if tokens_input is None or residual_input is None:
        tokens, residual = make_inputs(64)
    else:
        tokens = list(tokens_input)
        residual = array("f", residual_input)
    if len(tokens) != 64 or len(residual) != 64 * N_EMBD:
        raise GateError("middle worker requires exactly 64 input rows")
    b64_outputs: list[array] = []
    split_outputs: list[array] = []
    b64_us: list[int] = []
    split_us: list[int] = []
    hello: dict[str, int] | None = None
    error: str | None = None
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            hello = stage_hello(sock, 64)

            stage_reset(sock)
            stage_decode(sock, list(range(64)), tokens, residual)
            for _ in range(reps):
                stage_reset(sock)
                output, elapsed = stage_decode(sock, list(range(64)), tokens, residual)
                b64_outputs.append(output)
                b64_us.append(elapsed)

            stage_reset(sock)
            stage_decode(sock, list(range(32)), tokens[:32], residual[:32 * N_EMBD])
            stage_decode(sock, list(range(32, 64)), tokens[32:], residual[32 * N_EMBD:])
            for _ in range(reps):
                stage_reset(sock)
                started = time.monotonic_ns()
                first, _ = stage_decode(
                    sock, list(range(32)), tokens[:32], residual[:32 * N_EMBD]
                )
                second, _ = stage_decode(
                    sock, list(range(32, 64)), tokens[32:], residual[32 * N_EMBD:]
                )
                split_us.append((time.monotonic_ns() - started) // 1000)
                first.extend(second)
                split_outputs.append(first)
            send_i32(sock, STAGE_STOP)
    except Exception as exc:
        error = str(exc)
        process.kill()
    try:
        returncode = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        returncode = process.wait(timeout=5)
        error = error or "worker did not terminate after STOP"
    handle.flush()
    handle.close()
    if kind == "htp":
        adb("forward", "--remove", f"tcp:{port}", check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    cert = parse_json_record(text, "PLACEMENTCERT ")
    if not b64_outputs or not split_outputs:
        error = error or "worker produced no complete output"
        b64_last = array("f")
        split_last = array("f")
    else:
        b64_last = b64_outputs[-1]
        split_last = split_outputs[-1]
        (output_dir / f"{kind}.b64.f32").write_bytes(b64_last.tobytes())
        (output_dir / f"{kind}.split32.f32").write_bytes(split_last.tobytes())
    repeat_rel_l2 = max(
        [rel_l2(output, b64_last) for output in b64_outputs] +
        [rel_l2(output, split_last) for output in split_outputs],
        default=math.inf,
    )
    return {
        "kind": kind,
        "attention_route": "explicit" if decode_no_fa else "auto",
        "returncode": returncode,
        "error": error,
        "hello": hello,
        "placement": cert,
        "placement_problems": placement_problems(cert, "HTP0" if kind == "htp" else "CUDA0"),
        "b64_us": b64_us,
        "split32_us": split_us,
        "b64_p50_us": int(statistics.median(b64_us)) if b64_us else None,
        "split32_p50_us": int(statistics.median(split_us)) if split_us else None,
        "coalesce_speedup": (
            statistics.median(split_us) / statistics.median(b64_us)
            if b64_us and split_us and statistics.median(b64_us) > 0 else 0.0
        ),
        "b64_vs_split_rel_l2": rel_l2(b64_last, split_last),
        "b64_vs_split_argmax_mismatches": row_argmax_mismatches(b64_last, split_last, 64),
        "repeat_rel_l2_max": repeat_rel_l2,
        "finite": finite(b64_last) and finite(split_last),
        "b64_output": b64_last,
        "split_output": split_last,
    }


def public_worker(worker: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in worker.items() if key not in {"b64_output", "split_output"}}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def json_number(value: float) -> float | None:
    return value if math.isfinite(value) else None


def json_safe(value: Any) -> Any:
    if type(value) is float:
        return json_number(value)
    if type(value) is dict:
        return {key: json_safe(item) for key, item in value.items()}
    if type(value) is list:
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=HERE / "screen-results")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--decode-no-fa", action="store_true")
    args = parser.parse_args()
    if args.reps < 3 or args.timeout <= 0:
        parser.error("--reps must be at least 3 and --timeout must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists")
    if not HOST_BIN.is_file() or not LOCAL_SHARD.is_file():
        parser.error("host binary or local [6,12) shard is missing")
    if remote_sha256(PHONE_SHARD) != sha256_file(LOCAL_SHARD):
        parser.error("phone and host shard digests differ")

    args.output_dir.mkdir(parents=True)
    source_dir = args.output_dir / "executed-source"
    source_dir.mkdir()
    harness_copy = source_dir / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), harness_copy)
    thermal_start = thermal_snapshot()
    workers: dict[str, dict[str, Any]] = {}
    failure: str | None = None
    try:
        workers["cuda"] = run_worker(
            "cuda", 17418, args.output_dir, args.reps, args.timeout,
            decode_no_fa=args.decode_no_fa,
        )
        workers["htp"] = run_worker(
            "htp", 17417, args.output_dir, args.reps, args.timeout,
            decode_no_fa=args.decode_no_fa,
        )
    except Exception as exc:
        failure = str(exc)
    thermal_end = thermal_snapshot()

    htp = workers.get("htp")
    cuda = workers.get("cuda")
    cross_rel_l2 = math.inf
    cross_argmax = 64
    if htp and cuda:
        cross_rel_l2 = rel_l2(htp["b64_output"], cuda["b64_output"])
        cross_argmax = row_argmax_mismatches(htp["b64_output"], cuda["b64_output"], 64)
    checks = {
        "workers_complete": failure is None and htp is not None and cuda is not None and
            all(worker["returncode"] == 0 and worker["error"] is None for worker in workers.values()),
        "placement_valid": htp is not None and cuda is not None and
            all(not worker["placement_problems"] for worker in workers.values()),
        "outputs_finite": htp is not None and cuda is not None and
            all(worker["finite"] for worker in workers.values()),
        "repeat_stable": htp is not None and cuda is not None and
            all(worker["repeat_rel_l2_max"] <= 5e-3 for worker in workers.values()),
        "htp_b64_vs_2xb32": htp is not None and
            htp["b64_vs_split_rel_l2"] <= 5e-3 and
            htp["b64_vs_split_argmax_mismatches"] == 0,
        "htp_vs_cuda_b64": cross_rel_l2 <= 5e-3 and cross_argmax == 0,
        "htp_coalescing_faster": htp is not None and htp["coalesce_speedup"] >= 1.0,
    }
    passed = all(checks.values())

    artifact_paths = [path for path in args.output_dir.iterdir() if path.is_file()]
    result = {
        "schema": "s17-middle-island-screen-v1",
        "status": "CP1_MIDDLE_SCREEN_PASS" if passed else "CP1_MIDDLE_SCREEN_FAIL",
        "scope": "REAL_OP15_MIDDLE_ISLAND_MECHANICS_ONLY_NO_PIPELINE_OR_ENERGY_CLAIM",
        "route_fragment": {"device": "op15", "layer_range": [LAYER_START, LAYER_END]},
        "batch_shapes": {"coalesced": 64, "control": [32, 32], "decode_position": 0},
        "attention_route": "explicit" if args.decode_no_fa else "auto",
        "checks": checks,
        "cross_backend": {
            "b64_rel_l2": json_number(cross_rel_l2),
            "b64_argmax_mismatches": cross_argmax,
        },
        "workers": {key: public_worker(value) for key, value in workers.items()},
        "thermal": {"start": thermal_start, "end": thermal_end},
        "identity": {
            "harness": artifact(harness_copy),
            "host_binary": artifact(HOST_BIN),
            "host_shard": artifact(LOCAL_SHARD),
            "phone_binary_sha256": remote_sha256(f"{PHONE_DIR}/{PHONE_BIN}"),
            "phone_shard_sha256": remote_sha256(PHONE_SHARD),
            "selected_gpu_uuid": GPU_UUID,
            "phone_serial": PHONE_SERIAL,
        },
        "artifacts": {path.name: artifact(path) for path in artifact_paths},
        "error": failure,
        "phone_energy": "UNKNOWN",
        "total_system_energy": "UNKNOWN",
    }
    result = json_safe(result)
    result_path = args.output_dir / "report.json"
    result_path.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "checks": checks}, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
