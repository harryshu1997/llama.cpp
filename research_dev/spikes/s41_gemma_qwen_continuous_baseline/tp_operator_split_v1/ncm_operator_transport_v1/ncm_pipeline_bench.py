#!/usr/bin/env python3
"""Measure queued persistent NCM TCP exchanges at fixed payload sizes."""

import argparse
import json
import math
import socket
import statistics
import struct
import time
from pathlib import Path


STOP_LENGTH = 0xFFFFFFFF


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))
    return ordered[index]


def receive_exact(connection, storage, size):
    view = memoryview(storage)[:size]
    received = 0
    while received < size:
        count = connection.recv_into(view[received:])
        if count <= 0:
            raise RuntimeError("connection closed during response")
        received += count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--port", type=int, default=5202)
    parser.add_argument("--request-bytes", type=int, required=True)
    parser.add_argument("--response-bytes", type=int, required=True)
    parser.add_argument("--depth", action="append", type=int, required=True)
    parser.add_argument("--send-mode", choices=("frames", "batch"),
                        default="frames")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.request_bytes < 0 or args.response_bytes < 0:
        raise RuntimeError("payload sizes must be non-negative")
    if any(depth <= 0 for depth in args.depth):
        raise RuntimeError("depth must be positive")

    scope_id = socket.if_nametoindex(args.interface)
    connection = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    connection.settimeout(10.0)
    connection.connect((args.host, args.port, 0, scope_id))

    payload = bytes((index * 17 + 3) & 0xFF
                    for index in range(args.request_bytes))
    frame = struct.pack("!II", args.request_bytes, args.response_bytes) + payload
    response_size = 4 + args.response_bytes
    maximum_depth = max(args.depth)
    response = bytearray(response_size * maximum_depth)
    rows = []
    try:
        for depth in args.depth:
            batch = frame * depth
            values = []
            for index in range(args.warmup + args.iterations):
                started = time.perf_counter_ns()
                if args.send_mode == "batch":
                    connection.sendall(batch)
                else:
                    for _ in range(depth):
                        connection.sendall(frame)
                receive_exact(connection, response, response_size * depth)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                for response_index in range(depth):
                    offset = response_index * response_size
                    returned_bytes = struct.unpack_from("!I", response, offset)[0]
                    if returned_bytes != args.response_bytes:
                        raise RuntimeError("response length mismatch")
                if index >= args.warmup:
                    values.append(elapsed_ms)
            median_batch_ms = statistics.median(values)
            rows.append({
                "depth": depth,
                "iterations": args.iterations,
                "median_batch_ms": median_batch_ms,
                "median_amortized_ms": median_batch_ms / depth,
                "p90_batch_ms": percentile(values, 0.90),
                "p99_batch_ms": percentile(values, 0.99),
                "min_batch_ms": min(values),
                "aggregate_payload_MBps_at_median": (
                    depth * (args.request_bytes + args.response_bytes) /
                    (median_batch_ms / 1000.0) / 1e6
                    if args.request_bytes + args.response_bytes else 0.0
                ),
                "samples_ms": values,
            })
    finally:
        try:
            connection.sendall(struct.pack("!II", STOP_LENGTH, STOP_LENGTH))
        finally:
            connection.close()

    result = {
        "schema": "s41_ncm_pipeline_v1",
        "host": args.host,
        "interface": args.interface,
        "port": args.port,
        "request_bytes": args.request_bytes,
        "response_bytes": args.response_bytes,
        "send_mode": args.send_mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tcp_nodelay": True,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
