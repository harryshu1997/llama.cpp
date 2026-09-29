#!/usr/bin/env python3
"""Bracket one fully offloaded CUDA model replacement."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request


def reader(stream, rows: list[str]) -> None:
    for line in stream:
        rows.append(line)


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1.0
        ) as response:
            value = json.loads(response.read())
        return value.get("status") == "ok"
    except (
        ConnectionError,
        json.JSONDecodeError,
        OSError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ):
        return False


def launch(
    server: Path,
    model: Path,
    cuda_lib_dir: Path,
    port: int,
) -> tuple[subprocess.Popen[str], threading.Thread, list[str]]:
    command = [
        str(server),
        "--model", str(model),
        "--n-gpu-layers", "99",
        "--ctx-size", "2048",
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-warmup",
        "--log-verbosity", "4",
    ]
    output: list[str] = []
    process = subprocess.Popen(
        command,
        env={**os.environ, "LD_LIBRARY_PATH": str(cuda_lib_dir)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("server output pipe")
    thread = threading.Thread(target=reader, args=(process.stdout, output))
    thread.start()
    return process, thread, output


def wait_ready(process: subprocess.Popen[str], port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if healthy(port):
            return
        time.sleep(0.05)
    raise RuntimeError("server did not become ready")


def stop(process: subprocess.Popen[str], thread: threading.Thread) -> None:
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("server output reader did not stop")


def offloaded_layers(output: str) -> str:
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", output)
    if not matches or matches[-1][0] != matches[-1][1]:
        raise RuntimeError("full CUDA offload was not proven")
    return f"{matches[-1][0]}/{matches[-1][1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    for path in (args.server, args.source_model, args.target_model):
        if not path.is_file():
            raise RuntimeError(f"missing input: {path}")
    if not os.access(args.server, os.X_OK) or not args.cuda_lib_dir.is_dir():
        raise RuntimeError("server or CUDA runtime is unavailable")
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("invalid port")

    source_process, source_thread, source_output = launch(
        args.server, args.source_model, args.cuda_lib_dir, args.port
    )
    source_stopped = False
    target_process: subprocess.Popen[str] | None = None
    target_thread: threading.Thread | None = None
    target_output: list[str] = []
    try:
        wait_ready(source_process, args.port, args.timeout_s)

        started = time.monotonic_ns()
        print(
            f"ENERGY_WINDOW_START unix_ns={time.time_ns()} "
            "mode=model-switch n=1",
            flush=True,
        )
        stop(source_process, source_thread)
        source_stopped = True
        target_process, target_thread, target_output = launch(
            args.server, args.target_model, args.cuda_lib_dir, args.port
        )
        wait_ready(target_process, args.port, args.timeout_s)
        completed = time.monotonic_ns()
        print(
            f"ENERGY_WINDOW_END unix_ns={time.time_ns()} "
            "mode=model-switch n=1",
            flush=True,
        )
        stop(target_process, target_thread)
        target_process = None
    finally:
        if not source_stopped and source_process.poll() is None:
            stop(source_process, source_thread)
        if target_process is not None and target_process.poll() is None:
            assert target_thread is not None
            stop(target_process, target_thread)

    source_text = "".join(source_output)
    target_text = "".join(target_output)
    source_layers = offloaded_layers(source_text)
    target_layers = offloaded_layers(target_text)
    print(source_text, end="")
    print(target_text, end="")
    print(
        "MODEL_SWITCH_RESULT status=PASS "
        f"source={args.source_model.name} target={args.target_model.name} "
        f"target_bytes={args.target_model.stat().st_size} "
        f"elapsed_ms={(completed - started) / 1e6:.6f} "
        f"source_layers={source_layers} target_layers={target_layers}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"S42_MODEL_SWITCH_ERROR: {error}")
        raise SystemExit(2)
