#!/usr/bin/env python3
"""Measure synchronous NCM UDP request/response latency."""

import argparse
import json
import math
import socket
import statistics
import struct
import time
from pathlib import Path


STOP_LENGTH = 0xFFFFFFFF
MAX_PAYLOAD_BYTES = 60000


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))
    return ordered[index]


def parse_case(value):
    fields = value.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("case must be name:request:response")
    name = fields[0]
    request_bytes = int(fields[1])
    response_bytes = int(fields[2])
    if (not name or request_bytes < 0 or response_bytes < 0 or
            request_bytes > MAX_PAYLOAD_BYTES or
            response_bytes > MAX_PAYLOAD_BYTES):
        raise argparse.ArgumentTypeError("invalid case")
    return name, request_bytes, response_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--port", type=int, default=5203)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--case", action="append", type=parse_case,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scope_id = socket.if_nametoindex(args.interface)
    connection = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    connection.settimeout(2.0)
    connection.connect((args.host, args.port, 0, scope_id))

    maximum_response = max(item[2] for item in args.case)
    response = bytearray(2 * sizeof_u32() + maximum_response)
    rows = []
    sequence = 0
    try:
        for name, request_bytes, response_bytes in args.case:
            payload = bytes((index * 17 + 3) & 0xFF
                            for index in range(request_bytes))
            values = []
            timeouts = 0
            for index in range(args.warmup + args.iterations):
                sequence += 1
                frame = struct.pack("!III", request_bytes, response_bytes,
                                    sequence) + payload
                started = time.perf_counter_ns()
                connection.send(frame)
                try:
                    count = connection.recv_into(response)
                except TimeoutError:
                    timeouts += 1
                    continue
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                if count != 2 * sizeof_u32() + response_bytes:
                    raise RuntimeError("response size mismatch")
                returned_bytes, returned_sequence = struct.unpack_from(
                    "!II", response)
                if (returned_bytes != response_bytes or
                        returned_sequence != sequence):
                    raise RuntimeError("response header mismatch")
                if index >= args.warmup:
                    values.append(elapsed_ms)
            if len(values) != args.iterations:
                raise RuntimeError(
                    f"{name}: received {len(values)} paid responses, "
                    f"expected {args.iterations}; timeouts={timeouts}"
                )
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
                    (request_bytes + response_bytes) /
                    (median_ms / 1000.0) / 1e6
                    if request_bytes + response_bytes else 0.0
                ),
                "samples_ms": values,
            })
    finally:
        connection.send(struct.pack("!III", STOP_LENGTH, STOP_LENGTH, 0))
        connection.close()

    result = {
        "schema": "s41_ncm_udp_transport_v1",
        "host": args.host,
        "interface": args.interface,
        "port": args.port,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def sizeof_u32():
    return 4


if __name__ == "__main__":
    main()
