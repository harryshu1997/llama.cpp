#!/usr/bin/env python3
"""Exercise the prefetch fence protocol without starting llama.cpp or OP15."""

from __future__ import annotations

import argparse
import socket
import struct
import time
from pathlib import Path


MAGIC = 0x53343250
REQUEST = struct.Struct("=IHHQIIIIq")
RESPONSE = struct.Struct("=IHHQIIQIIqq")


def exchange(
    connection: socket.socket,
    sequence: int,
    request_id: int,
) -> tuple[int, int]:
    connection.sendall(REQUEST.pack(
        MAGIC,
        1,
        1,
        sequence,
        request_id,
        sequence % 12,
        1,
        0,
        time.monotonic_ns(),
    ))
    payload = connection.recv(RESPONSE.size)
    if len(payload) != RESPONSE.size:
        raise RuntimeError("short fence response")
    row = RESPONSE.unpack(payload)
    if (
        row[0] != MAGIC
        or row[1] != 1
        or row[2] != 2
        or row[3] != sequence
        or row[4] != request_id
        or row[5] != 0
        or row[10] < row[9]
    ):
        raise RuntimeError("invalid fence response")
    return row[6], row[7]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--calls", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    args = parser.parse_args()
    if (
        not args.socket.is_absolute()
        or not args.arm_file.is_absolute()
        or args.arm_file.exists()
        or args.calls <= 0
        or args.expected_bytes < 0
    ):
        parser.error("invalid smoke-test arguments")

    copied_bytes = 0
    copied_chunks = 0
    with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as connection:
        connection.connect(str(args.socket))
        for sequence in (1, 2):
            copied, chunks = exchange(connection, sequence, sequence)
            if copied != 0 or chunks != 0:
                raise RuntimeError("unarmed fence copied data")
        args.arm_file.write_text(f"{time.monotonic_ns()}\n", encoding="ascii")
        for sequence in range(3, args.calls + 3):
            copied, chunks = exchange(connection, sequence, sequence)
            copied_bytes += copied
            copied_chunks += chunks
    if copied_bytes != args.expected_bytes:
        raise RuntimeError(
            f"copied {copied_bytes} bytes, expected {args.expected_bytes}"
        )
    print(
        f"PREFETCH_SMOKE status=PASS calls={args.calls} "
        f"copied_bytes={copied_bytes} copied_chunks={copied_chunks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
