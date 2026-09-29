#!/usr/bin/env python3
"""Measure synchronous FunctionFS bulk request/response latency."""

import argparse
import ctypes
import ctypes.util
import json
import math
import statistics
import struct
import time
from pathlib import Path


STOP_LENGTH = 0xFFFFFFFF


class Context(ctypes.Structure):
    pass


CONTEXT_POINTER = ctypes.POINTER(Context)


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
    if not name or request_bytes < 0 or response_bytes < 0:
        raise argparse.ArgumentTypeError("invalid case")
    return name, request_bytes, response_bytes


def load_libusb():
    library = ctypes.CDLL(ctypes.util.find_library("usb-1.0"))
    library.libusb_init.argtypes = [ctypes.POINTER(CONTEXT_POINTER)]
    library.libusb_open_device_with_vid_pid.restype = ctypes.c_void_p
    library.libusb_open_device_with_vid_pid.argtypes = [
        CONTEXT_POINTER, ctypes.c_uint16, ctypes.c_uint16]
    library.libusb_claim_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.libusb_release_interface.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.libusb_bulk_transfer.argtypes = [
        ctypes.c_void_p, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_uint]
    library.libusb_close.argtypes = [ctypes.c_void_p]
    library.libusb_exit.argtypes = [CONTEXT_POINTER]
    context = CONTEXT_POINTER()
    if library.libusb_init(ctypes.byref(context)) != 0:
        raise RuntimeError("libusb_init failed")
    return library, context


def bulk_out(library, handle, endpoint, data, timeout_ms):
    transferred = ctypes.c_int()
    status = library.libusb_bulk_transfer(
        ctypes.c_void_p(handle), endpoint, data, len(data),
        ctypes.byref(transferred), timeout_ms)
    if status != 0 or transferred.value != len(data):
        raise RuntimeError(
            f"bulk OUT status={status} transferred={transferred.value}"
        )


def bulk_in_exact(library, handle, endpoint, storage, size, timeout_ms):
    received = 0
    while received < size:
        transferred = ctypes.c_int()
        pointer = ctypes.cast(ctypes.byref(storage, received),
                              ctypes.POINTER(ctypes.c_ubyte))
        status = library.libusb_bulk_transfer(
            ctypes.c_void_p(handle), endpoint, pointer, size - received,
            ctypes.byref(transferred), timeout_ms)
        if status != 0 or transferred.value <= 0:
            raise RuntimeError(
                f"bulk IN status={status} transferred={transferred.value}"
            )
        received += transferred.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vid", type=lambda value: int(value, 16),
                        default=0x18D1)
    parser.add_argument("--pid", type=lambda value: int(value, 16),
                        default=0x2D00)
    parser.add_argument("--out-endpoint", type=lambda value: int(value, 16),
                        default=0x01)
    parser.add_argument("--in-endpoint", type=lambda value: int(value, 16),
                        default=0x82)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--case", action="append", type=parse_case,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    maximum_response = max(item[2] for item in args.case)
    response = (ctypes.c_ubyte * (4 + maximum_response))()
    library, context = load_libusb()
    handle = library.libusb_open_device_with_vid_pid(
        context, args.vid, args.pid)
    if not handle:
        raise RuntimeError("FunctionFS device not found")
    if library.libusb_claim_interface(ctypes.c_void_p(handle), 0) != 0:
        raise RuntimeError("claim interface failed")

    rows = []
    try:
        for name, request_bytes, response_bytes in args.case:
            payload = bytes((index * 17 + 3) & 0xFF
                            for index in range(request_bytes))
            header_bytes = struct.pack("!II", request_bytes, response_bytes)
            header = (ctypes.c_ubyte * len(header_bytes)).from_buffer_copy(
                header_bytes)
            request = ((ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
                       if payload else None)
            values = []
            for index in range(args.warmup + args.iterations):
                started = time.perf_counter_ns()
                bulk_out(library, handle, args.out_endpoint, header,
                         args.timeout_ms)
                if request is not None:
                    bulk_out(library, handle, args.out_endpoint, request,
                             args.timeout_ms)
                bulk_in_exact(library, handle, args.in_endpoint, response,
                              4 + response_bytes, args.timeout_ms)
                elapsed_ms = (time.perf_counter_ns() - started) / 1e6
                returned_bytes = struct.unpack_from("!I", response)[0]
                if returned_bytes != response_bytes:
                    raise RuntimeError("response length mismatch")
                if index >= args.warmup:
                    values.append(elapsed_ms)
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
        stop_bytes = struct.pack("!II", STOP_LENGTH, STOP_LENGTH)
        stop = (ctypes.c_ubyte * len(stop_bytes)).from_buffer_copy(stop_bytes)
        bulk_out(library, handle, args.out_endpoint, stop, args.timeout_ms)
    finally:
        library.libusb_release_interface(ctypes.c_void_p(handle), 0)
        library.libusb_close(ctypes.c_void_p(handle))
        library.libusb_exit(context)

    result = {
        "schema": "s41_functionfs_transport_v1",
        "vid": f"0x{args.vid:04x}",
        "pid": f"0x{args.pid:04x}",
        "out_endpoint": f"0x{args.out_endpoint:02x}",
        "in_endpoint": f"0x{args.in_endpoint:02x}",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
