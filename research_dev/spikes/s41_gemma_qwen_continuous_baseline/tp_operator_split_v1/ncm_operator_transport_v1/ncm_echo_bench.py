#!/usr/bin/env python3
"""Measure persistent NCM request/response latency at operator payload sizes."""

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


def parse_case(value):
    fields = value.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("case must be name:request:response")
    name = fields[0]
    request_bytes = int(fields[1])
    response_bytes = int(fields[2])
    if not name or request_bytes < 0 or response_bytes < 0:
        raise argparse.ArgumentTypeError("invalid case")
    return name, request_bytes, response_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--port", type=int, default=5202)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scope_id = socket.if_nametoindex(args.interface)
    connection = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    connection.settimeout(10.0)
    connection.connect((args.host, args.port, 0, scope_id))

    maximum_response = max(item[2] for item in args.case)
    response = bytearray(4 + maximum_response)
    rows = []
    try:
        for name, request_bytes, response_bytes in args.case:
            payload = bytes((index * 17 + 3) & 0xFF
                            for index in range(request_bytes))
            frame = struct.pack("!II", request_bytes, response_bytes) + payload
            values = []
            for index in range(args.warmup + args.iterations):
                started = time.perf_counter_ns()
                connection.sendall(frame)
                receive_exact(connection, response, 4 + response_bytes)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                returned_bytes = struct.unpack_from("!I", response)[0]
                if returned_bytes != response_bytes:
                    raise RuntimeError("response length mismatch")
                if index >= args.warmup:
                    values.append(elapsed_ms)
            total_bytes = request_bytes + response_bytes
            median_ms = statistics.median(values)
            rows.append({
                "name": name,
                "request_bytes": request_bytes,
                "response_bytes": response_bytes,
                "iterations": args.iterations,
                "median_ms": median_ms,
                "p90_ms": percentile(values, 0.90),
                "p99_ms": percentile(values, 0.99),
                "min_ms": min(values),
                "effective_payload_MBps_at_median": (
                    total_bytes / (median_ms / 1000.0) / 1e6
                    if total_bytes else 0.0
                ),
                "samples_ms": values,
            })
    finally:
        try:
            connection.sendall(struct.pack("!II", STOP_LENGTH, STOP_LENGTH))
        finally:
            connection.close()

    result = {
        "schema": "s41_ncm_operator_transport_v1",
        "host": args.host,
        "interface": args.interface,
        "port": args.port,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "tcp_nodelay": True,
        "persistent_connection": True,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
