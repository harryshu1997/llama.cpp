#!/usr/bin/env python3
"""One-command process bridge to a resident warm-tier gateway."""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

from executor_bundle import BundleError, validate_runtime_environment


try:
    validate_runtime_environment(Path(__file__).resolve())
except (BundleError, OSError) as error:
    print(f"executor bundle failed: {error}", file=sys.stderr)
    raise SystemExit(2)

MAX_BYTES = 4 * 1024 * 1024


def read_one(stream) -> bytes:
    data = stream.buffer.readline(MAX_BYTES + 1)
    if not data.endswith(b"\n") or len(data) > MAX_BYTES:
        raise RuntimeError("invalid bridge input frame")
    return data


def recv_one(connection: socket.socket) -> bytes:
    data = bytearray()
    while len(data) <= MAX_BYTES:
        chunk = connection.recv(min(65536, MAX_BYTES + 1 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        if data.endswith(b"\n"):
            break
    if not data.endswith(b"\n") or len(data) > MAX_BYTES:
        raise RuntimeError("invalid bridge result frame")
    return bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if not args.socket.is_absolute() or args.timeout <= 0:
        parser.error("socket and timeout are invalid")

    raw = read_one(sys.stdin)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(args.timeout)
    try:
        connection.connect(str(args.socket))
        connection.sendall(raw)
        connection.shutdown(socket.SHUT_WR)
        result = recv_one(connection)
    finally:
        connection.close()
    sys.stdout.buffer.write(result)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"gateway bridge failed: {error}", file=sys.stderr)
        raise SystemExit(2)
