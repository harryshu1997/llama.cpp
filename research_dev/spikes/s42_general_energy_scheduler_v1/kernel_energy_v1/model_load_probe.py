#!/usr/bin/env python3
"""Launch one CUDA llama-server and bracket model-ready time only."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if not args.server.is_file() or not os.access(args.server, os.X_OK):
        raise RuntimeError("server is unavailable")
    if not args.model.is_file() or not args.cuda_lib_dir.is_dir():
        raise RuntimeError("model or CUDA runtime is unavailable")
    if not 1024 <= args.port <= 65535:
        raise RuntimeError("invalid port")

    command = [
        str(args.server),
        "--model", str(args.model),
        "--n-gpu-layers", "99",
        "--ctx-size", "2048",
        "--parallel", "1",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--no-warmup",
        "--log-verbosity", "4",
    ]
    environment = {
        **os.environ,
        "LD_LIBRARY_PATH": str(args.cuda_lib_dir),
    }
    output: list[str] = []
    started = time.monotonic_ns()
    print(
        f"ENERGY_WINDOW_START unix_ns={time.time_ns()} mode=model-load n=1",
        flush=True,
    )
    process = subprocess.Popen(
        command,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError("server output pipe")
    thread = threading.Thread(target=reader, args=(process.stdout, output))
    thread.start()
    deadline = time.monotonic() + args.timeout_s
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        if healthy(args.port):
            ready = True
            break
        time.sleep(0.05)
    completed = time.monotonic_ns()
    print(
        f"ENERGY_WINDOW_END unix_ns={time.time_ns()} mode=model-load n=1",
        flush=True,
    )
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("server output reader did not stop")
    combined = "".join(output)
    print(combined, end="")
    if not ready:
        raise RuntimeError("server did not become ready")
    matches = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", combined)
    if not matches or matches[-1][0] != matches[-1][1]:
        raise RuntimeError("full CUDA offload was not proven")
    print(
        "MODEL_LOAD_RESULT status=PASS "
        f"model={args.model.name} bytes={args.model.stat().st_size} "
        f"elapsed_ms={(completed - started) / 1e6:.6f} "
        f"layers={matches[-1][0]}/{matches[-1][1]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"S42_MODEL_LOAD_ERROR: {error}")
        raise SystemExit(2)
