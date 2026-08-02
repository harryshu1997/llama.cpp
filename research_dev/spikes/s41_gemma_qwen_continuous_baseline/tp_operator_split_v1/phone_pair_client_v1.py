#!/usr/bin/env python3
"""Measure one or two phone operator slices over persistent endpoints."""

import argparse
import concurrent.futures
import ctypes
import json
import math
import pathlib
import socket
import statistics
import struct
import threading
import time

import numpy as np

import aoa_bench


MASK64 = (1 << 64) - 1
TYPE_Q8_0 = 8

FFN_REQUEST_MAGIC = 0x53343151
FFN_RESPONSE_MAGIC = 0x53343152
FFN_VERSION = 2
FFN_FAST_HASH = 1
FFN_F16_INPUT = 2
FFN_F16_OUTPUT = 8
FFN_RESIDUAL_OUTPUT = 16
FFN_REQUEST = struct.Struct("<IHHIIIIIIIIQ")
FFN_RESPONSE = struct.Struct("<IHHIIIIQ")

ATTN_REQUEST_MAGIC = 0x53344151
ATTN_RESPONSE_MAGIC = 0x53344152
ATTN_VERSION = 3
ATTN_FAST_HASH = 1
ATTN_F16_INPUT = 2
ATTN_F16_OUTPUT = 4
ATTN_LAST_SLOT_UPDATE = 8
ATTN_SINGLE_GROUP = 16
ATTN_RESIDUAL_OUTPUT = 64
ATTN_REQUEST = struct.Struct("<IHHIIIIIIIIIIQ")
ATTN_RESPONSE = struct.Struct("<IHHIIIIQIIII")


def mix64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def activation_value(index):
    bits = (mix64(0x41A7C9E3 ^ index) >> 40) & 0xFFFFFFFF
    centered = int(bits) - 0x800000
    return np.float32(centered) * np.float32(1.0 / 8388608.0)


def activation_bytes(elements):
    values = np.fromiter(
        (activation_value(index) for index in range(elements)),
        dtype=np.float32,
        count=elements,
    )
    return values.astype("<f2").tobytes()


def fast_hash(data):
    size = len(data)
    value = mix64(14695981039346656037 ^ size)
    offset = 0
    while offset + 8 <= size:
        lane = int.from_bytes(data[offset:offset + 8], "little")
        value ^= mix64((lane + offset) & MASK64)
        value = (
            (((value << 27) & MASK64) | (value >> 37))
            * 0x3C79AC492BA7B653
            + 0x1C69B3F74AC4AE35
        ) & MASK64
        offset += 8
    tail = int.from_bytes(data[offset:], "little")
    value ^= mix64(tail ^ (size - offset))
    value = mix64(value)
    return (value ^ (value >> 32)) & 0xFFFFFFFF


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))
    return ordered[index]


def latency_summary(values):
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
    }


class TcpEndpoint:
    def __init__(self, port):
        self.socket = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.socket.settimeout(30)

    def exchange(self, request, response_bytes):
        self.socket.sendall(request)
        chunks = []
        remaining = response_bytes
        while remaining:
            chunk = self.socket.recv(remaining)
            if not chunk:
                raise RuntimeError("TCP endpoint closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def close(self):
        self.socket.close()


class AoaEndpoint:
    def __init__(self):
        self.libusb, self.context = aoa_bench.load()
        handle, pid = aoa_bench.open_accessory(self.libusb, self.context)
        if not handle:
            raise RuntimeError("no AOA accessory found")
        self.handle = ctypes.c_void_p(handle)
        self.pid = pid
        self.libusb.libusb_detach_kernel_driver(self.handle, 0)
        status = self.libusb.libusb_claim_interface(self.handle, 0)
        if status != 0:
            raise RuntimeError(f"libusb claim failed: {status}")

    def _transfer(self, endpoint, data):
        offset = 0
        while offset < len(data):
            chunk = (ctypes.c_ubyte * (len(data) - offset)).from_buffer_copy(
                data[offset:]
            )
            transferred = ctypes.c_int()
            status = self.libusb.libusb_bulk_transfer(
                self.handle,
                endpoint,
                chunk,
                len(chunk),
                ctypes.byref(transferred),
                30000,
            )
            if status != 0 or transferred.value <= 0:
                raise RuntimeError(
                    f"libusb OUT failed: status={status} bytes={transferred.value}"
                )
            offset += transferred.value

    def _receive(self, endpoint, size):
        output = bytearray(size)
        offset = 0
        while offset < size:
            chunk = (ctypes.c_ubyte * (size - offset))()
            transferred = ctypes.c_int()
            status = self.libusb.libusb_bulk_transfer(
                self.handle,
                endpoint,
                chunk,
                len(chunk),
                ctypes.byref(transferred),
                30000,
            )
            if status != 0 or transferred.value <= 0:
                raise RuntimeError(
                    f"libusb IN failed: status={status} bytes={transferred.value}"
                )
            output[offset:offset + transferred.value] = bytes(
                chunk[:transferred.value]
            )
            offset += transferred.value
        return bytes(output)

    def exchange(self, request, response_bytes):
        self._transfer(0x01, request)
        return self._receive(0x81, response_bytes)

    def close(self):
        self.libusb.libusb_release_interface(self.handle, 0)
        self.libusb.libusb_close(self.handle)
        self.libusb.libusb_exit(self.context)


def open_endpoint(specification):
    if specification == "aoa":
        return AoaEndpoint()
    if specification.startswith("tcp:"):
        return TcpEndpoint(int(specification.split(":", 1)[1]))
    raise ValueError(f"unsupported endpoint: {specification}")


class FfnLeg:
    def __init__(self, endpoint, args, offset, count, weight_hash):
        self.endpoint = endpoint
        self.k = args.k
        self.n_ff = args.n_ff
        self.offset = offset
        self.count = count
        self.batch = args.batch
        self.weight_hash = weight_hash
        self.output_elements = self.k * self.batch
        base_flags = (
            FFN_FAST_HASH
            | FFN_F16_INPUT
            | FFN_F16_OUTPUT
            | FFN_RESIDUAL_OUTPUT
        )
        self.flags = base_flags | (self.batch << 8 if self.batch > 1 else 0)

    def exchange(self, request_id, input_data):
        header = FFN_REQUEST.pack(
            FFN_REQUEST_MAGIC,
            FFN_VERSION,
            self.flags,
            request_id,
            TYPE_Q8_0,
            self.k,
            self.n_ff,
            self.offset,
            self.count,
            len(input_data),
            fast_hash(input_data),
            self.weight_hash,
        )
        size = FFN_RESPONSE.size + 2 * self.output_elements
        response = self.endpoint.exchange(header + input_data, size)
        fields = FFN_RESPONSE.unpack_from(response)
        expected = (
            FFN_RESPONSE_MAGIC,
            FFN_VERSION,
            0,
            request_id,
            self.output_elements,
            2 * self.output_elements,
        )
        if fields[:6] != expected or fields[7] != self.weight_hash:
            raise RuntimeError(f"invalid FFN response: {fields}")
        output_data = response[FFN_RESPONSE.size:]
        if fields[6] != fast_hash(output_data):
            raise RuntimeError("FFN response hash mismatch")
        return np.frombuffer(output_data, dtype="<f2").astype(np.float32), {}


class AttentionLeg:
    def __init__(self, endpoint, args, offset, count, weight_hash):
        self.endpoint = endpoint
        self.k = args.k
        self.n_kv = args.n_kv
        self.offset = offset
        self.count = count
        self.weight_hash = weight_hash
        self.output_elements = self.k
        self.flags = (
            ATTN_FAST_HASH
            | ATTN_F16_INPUT
            | ATTN_F16_OUTPUT
            | ATTN_LAST_SLOT_UPDATE
            | (ATTN_SINGLE_GROUP if count == 1 else 0)
            | ATTN_RESIDUAL_OUTPUT
        )

    def exchange(self, request_id, input_data):
        header = ATTN_REQUEST.pack(
            ATTN_REQUEST_MAGIC,
            ATTN_VERSION,
            self.flags,
            request_id,
            TYPE_Q8_0,
            self.k,
            self.n_kv,
            40,
            8,
            self.offset,
            self.count,
            len(input_data),
            fast_hash(input_data),
            self.weight_hash,
        )
        size = ATTN_RESPONSE.size + 2 * self.output_elements
        response = self.endpoint.exchange(header + input_data, size)
        fields = ATTN_RESPONSE.unpack_from(response)
        expected = (
            ATTN_RESPONSE_MAGIC,
            ATTN_VERSION,
            0,
            request_id,
            self.output_elements,
            2 * self.output_elements,
        )
        if fields[:6] != expected or fields[7] != self.weight_hash:
            raise RuntimeError(f"invalid attention response: {fields}")
        output_data = response[ATTN_RESPONSE.size:]
        if fields[6] != fast_hash(output_data):
            raise RuntimeError("attention response hash mismatch")
        timers = {
            "set_us": fields[8],
            "compute_us": fields[9],
            "get_us": fields[10],
        }
        return np.frombuffer(output_data, dtype="<f2").astype(np.float32), timers


def timed_exchange(leg, barrier, request_id, input_data):
    barrier.wait()
    started = time.perf_counter_ns()
    output, timers = leg.exchange(request_id, input_data)
    completed = time.perf_counter_ns()
    return output, timers, started, completed


def run_iteration(executor, legs, request_id, input_data):
    barrier = threading.Barrier(len(legs) + 1)
    futures = [
        executor.submit(timed_exchange, leg, barrier, request_id, input_data)
        for leg in legs
    ]
    barrier.wait()
    results = [future.result() for future in futures]
    started = min(result[2] for result in results)
    completed = max(result[3] for result in results)
    leg_ms = [(result[3] - result[2]) / 1e6 for result in results]
    return results, leg_ms, (completed - started) / 1e6


def compare(reference, candidate, width):
    if reference.shape != candidate.shape:
        raise ValueError("reference shape mismatch")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    denominator = np.dot(reference.astype(np.float64), reference.astype(np.float64))
    relative_l2 = math.sqrt(np.dot(difference, difference) / denominator)
    rows = reference.size // width
    reference_rows = reference.reshape(rows, width)
    candidate_rows = candidate.reshape(rows, width)
    argmax_matches = int(np.sum(
        np.argmax(reference_rows, axis=1) == np.argmax(candidate_rows, axis=1)
    ))
    return {
        "relative_l2": relative_l2,
        "max_absolute": float(np.max(np.abs(difference))),
        "non_finite": int(np.sum(~np.isfinite(candidate))),
        "argmax_matches": argmax_matches,
        "argmax_rows": rows,
    }


def parse_hash(value):
    return int(value, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", choices=("ffn", "attention"))
    parser.add_argument("--endpoint", action="append", required=True)
    parser.add_argument("--offset", action="append", type=int, required=True)
    parser.add_argument("--count", action="append", type=int, required=True)
    parser.add_argument("--weight-hash", action="append", type=parse_hash, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n-ff", type=int)
    parser.add_argument("--n-kv", type=int)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--reference", type=pathlib.Path)
    args = parser.parse_args()

    leg_count = len(args.endpoint)
    if not (
        leg_count == len(args.offset)
        and leg_count == len(args.count)
        and leg_count == len(args.weight_hash)
    ):
        parser.error("endpoint, offset, count, and weight-hash counts must match")
    if leg_count not in (1, 2):
        parser.error("exactly one or two endpoints are supported")
    if args.operator == "ffn" and (not args.n_ff or args.n_kv is not None):
        parser.error("FFN requires --n-ff and forbids --n-kv")
    if args.operator == "attention" and (not args.n_kv or args.n_ff is not None):
        parser.error("attention requires --n-kv and forbids --n-ff")

    input_data = activation_bytes(args.k * args.batch)
    endpoints = [open_endpoint(value) for value in args.endpoint]
    leg_type = FfnLeg if args.operator == "ffn" else AttentionLeg
    legs = [
        leg_type(endpoint, args, offset, count, weight_hash)
        for endpoint, offset, count, weight_hash in zip(
            endpoints, args.offset, args.count, args.weight_hash
        )
    ]

    leg_latencies = [[] for _ in legs]
    pair_latencies = []
    backend_timers = [[] for _ in legs]
    final_outputs = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=leg_count) as executor:
            for index in range(args.warmup + args.iterations):
                results, leg_ms, pair_ms = run_iteration(
                    executor, legs, index + 1, input_data
                )
                if index >= args.warmup:
                    for leg_index, value in enumerate(leg_ms):
                        leg_latencies[leg_index].append(value)
                        backend_timers[leg_index].append(results[leg_index][1])
                    pair_latencies.append(pair_ms)
                final_outputs = [result[0] for result in results]
    finally:
        for endpoint in endpoints:
            endpoint.close()

    aggregate = np.sum(np.stack(final_outputs), axis=0, dtype=np.float32)
    report = {
        "schema": "s41-phone-pair-client-v1",
        "label": args.label,
        "operator": args.operator,
        "geometry": {
            "k": args.k,
            "n_ff": args.n_ff,
            "n_kv": args.n_kv,
            "batch": args.batch,
            "offsets": args.offset,
            "counts": args.count,
        },
        "transports": args.endpoint,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "pair_makespan": latency_summary(pair_latencies),
        "legs": [],
        "output_fast_hash": f"{fast_hash(aggregate.astype('<f4').tobytes()):08x}",
    }
    for index, values in enumerate(leg_latencies):
        leg_report = {"index": index, "latency": latency_summary(values)}
        if args.operator == "attention":
            for timer in ("set_us", "compute_us", "get_us"):
                leg_report[timer] = latency_summary([
                    value[timer] / 1000.0 for value in backend_timers[index]
                ])
        report["legs"].append(leg_report)

    if args.reference:
        with np.load(args.reference, allow_pickle=False) as reference_file:
            reference = reference_file["aggregate"]
        report["correctness"] = compare(reference, aggregate, args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        aggregate=aggregate,
        report=np.array(json.dumps(report, sort_keys=True)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
