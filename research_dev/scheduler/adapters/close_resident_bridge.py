#!/usr/bin/env python3
"""Arm a resident composite worker and close it without executing work."""

from __future__ import annotations

import argparse
import socket
import struct


PROTOCOL_MAGIC = 0x46534631
PROTOCOL_VERSION = 5
HELLO_REQUEST = 1
F16_IO_FLAG = 1
SWIGLU_FLAG = 2


def receive_exact(stream: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise RuntimeError("bridge closed during HELLO")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--layer-mask", type=int, required=True)
    parser.add_argument("--n-embd", type=int, required=True)
    parser.add_argument("--columns", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--activation", choices=("geglu", "swiglu"), default="swiglu"
    )
    parser.add_argument("--terminate-session", action="store_true")
    args = parser.parse_args()
    if (
        args.port <= 0 or args.port > 65535 or args.layer_mask <= 0
        or args.n_embd <= 0 or args.columns <= 0
        or args.max_tokens <= 0 or args.max_tokens > 65535
    ):
        parser.error("invalid resident worker identity")

    request = struct.pack(
        "<IHHQIIHH4x",
        PROTOCOL_MAGIC,
        PROTOCOL_VERSION,
        HELLO_REQUEST,
        args.layer_mask,
        args.n_embd,
        args.columns,
        F16_IO_FLAG | (SWIGLU_FLAG if args.activation == "swiglu" else 0),
        args.max_tokens,
    )
    with socket.create_connection((args.host, args.port), timeout=30) as stream:
        stream.sendall(request)
        response = receive_exact(stream, 64)
        magic, version, message, status = struct.unpack_from("<IHHH", response)
        if (
            magic != PROTOCOL_MAGIC or version != PROTOCOL_VERSION
            or message != 2 or status != 0
        ):
            raise RuntimeError("resident worker rejected HELLO")
        if args.terminate_session:
            stream.sendall(struct.pack(
                "<IHHIiIIIII",
                PROTOCOL_MAGIC,
                PROTOCOL_VERSION,
                3,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
            ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
