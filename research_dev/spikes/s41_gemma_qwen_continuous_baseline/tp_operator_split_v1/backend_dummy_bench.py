#!/usr/bin/env python3
"""Measure matched AOA and phone-backend stages for a dummy GGML graph."""

import argparse
import ctypes
import json
import math
import statistics
import struct
import time
import zlib

from aoa_bench import load, open_accessory


REQUEST_MAGIC = 0x53344451
RESPONSE_MAGIC = 0x53344452
PROTOCOL_VERSION = 1
REQUEST = struct.Struct("<IHHIIII")
RESPONSE = struct.Struct("<IHHIIIIQQQQQQQQQ")
OUT_ENDPOINT = 0x01
IN_ENDPOINT = 0x81

STAGE_NAMES = (
    "worker_validate",
    "worker_decode",
    "backend_set",
    "backend_submit",
    "backend_sync",
    "backend_get",
    "worker_encode_hash",
    "worker_prewrite",
    "worker_previous_write",
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(fraction * (len(ordered) - 1)))
    return ordered[index]


def summary(values):
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
    }


def transfer_exact(libusb, handle, endpoint, data, timeout_ms):
    completed = 0
    transferred = ctypes.c_int()
    while completed < len(data):
        pointer = ctypes.cast(
            ctypes.byref(data, completed), ctypes.POINTER(ctypes.c_ubyte)
        )
        status = libusb.libusb_bulk_transfer(
            ctypes.c_void_p(handle), endpoint, pointer,
            len(data) - completed, ctypes.byref(transferred), timeout_ms
        )
        if status != 0:
            raise RuntimeError(f"bulk endpoint 0x{endpoint:02x} failed: {status}")
        if transferred.value <= 0:
            raise RuntimeError(f"bulk endpoint 0x{endpoint:02x} made no progress")
        completed += transferred.value


def make_input(elements):
    values = [((index % 257) - 128) / 256.0 for index in range(elements)]
    return struct.pack(f"<{elements}e", *values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=2816)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--op", choices=("noop", "sqr"), required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=4000)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.elements <= 0 or args.warmup < 0 or args.iters <= 0:
        parser.error("invalid dimensions or iteration count")

    payload = make_input(args.elements)
    payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
    response_size = RESPONSE.size + len(payload)
    libusb, context = load()
    handle, pid = open_accessory(libusb, context)
    if not handle:
        raise SystemExit("no accessory-mode device found")
    libusb.libusb_detach_kernel_driver(ctypes.c_void_p(handle), 0)
    status = libusb.libusb_claim_interface(ctypes.c_void_p(handle), 0)
    if status != 0:
        raise SystemExit(f"claim_interface failed: {status}")

    samples = []
    request_id = 1
    total_requests = args.warmup + args.iters
    try:
        for index in range(total_requests):
            total_started = time.perf_counter_ns()
            prepare_started = time.perf_counter_ns()
            header = REQUEST.pack(
                REQUEST_MAGIC, PROTOCOL_VERSION, 0, request_id,
                args.elements, len(payload), payload_crc
            )
            request_data = (ctypes.c_ubyte * (len(header) + len(payload))).from_buffer_copy(
                header + payload
            )
            prepare_ms = (time.perf_counter_ns() - prepare_started) / 1e6

            started = time.perf_counter_ns()
            transfer_exact(
                libusb, handle, OUT_ENDPOINT, request_data, args.timeout_ms
            )
            usb_out_ms = (time.perf_counter_ns() - started) / 1e6

            response_data = (ctypes.c_ubyte * response_size)()
            started = time.perf_counter_ns()
            transfer_exact(
                libusb, handle, IN_ENDPOINT, response_data, args.timeout_ms
            )
            usb_in_ms = (time.perf_counter_ns() - started) / 1e6

            validate_started = time.perf_counter_ns()
            response_bytes = bytes(response_data)
            fields = RESPONSE.unpack_from(response_bytes)
            output = response_bytes[RESPONSE.size:]
            magic, version, response_status, response_id = fields[:4]
            response_elements, output_bytes, output_crc = fields[4:7]
            if (
                magic != RESPONSE_MAGIC
                or version != PROTOCOL_VERSION
                or response_status != 0
                or response_id != request_id
                or response_elements != args.elements
                or output_bytes != len(output)
                or output_crc != (zlib.crc32(output) & 0xFFFFFFFF)
            ):
                raise RuntimeError(f"invalid response for request {request_id}")
            validate_ms = (time.perf_counter_ns() - validate_started) / 1e6
            total_ms = (time.perf_counter_ns() - total_started) / 1e6

            if index >= args.warmup:
                sample = {
                    "host_prepare_ms": prepare_ms,
                    "usb_out_ms": usb_out_ms,
                    "usb_in_wait_ms": usb_in_ms,
                    "host_validate_ms": validate_ms,
                    "e2e_ms": total_ms,
                }
                for name, value_ns in zip(STAGE_NAMES, fields[7:]):
                    sample[f"{name}_ms"] = value_ns / 1e6
                sample["response_path_residual_ms"] = max(
                    0.0, usb_in_ms - sample["worker_prewrite_ms"]
                )
                sample["worker_unattributed_ms"] = max(
                    0.0,
                    sample["worker_prewrite_ms"]
                    - sum(
                        sample[name]
                        for name in (
                            "worker_validate_ms",
                            "worker_decode_ms",
                            "backend_set_ms",
                            "backend_submit_ms",
                            "backend_sync_ms",
                            "backend_get_ms",
                            "worker_encode_hash_ms",
                        )
                    ),
                )
                samples.append(sample)
            request_id += 1
    finally:
        libusb.libusb_release_interface(ctypes.c_void_p(handle), 0)
        libusb.libusb_close(ctypes.c_void_p(handle))
        libusb.libusb_exit(context)

    stages = {
        name: summary([sample[name] for sample in samples])
        for name in samples[0]
    }
    result = {
        "schema": "s41_backend_dummy_latency_v1",
        "backend": args.backend,
        "op": args.op,
        "repeats": args.repeats,
        "elements": args.elements,
        "input_bytes": len(payload),
        "output_bytes": len(payload),
        "warmup": args.warmup,
        "iterations": args.iters,
        "accessory_pid": f"0x{pid:04x}",
        "stages": stages,
        "samples": samples,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="ascii") as output_file:
            output_file.write(encoded)
            output_file.write("\n")
    print(
        f"BACKEND_DUMMY backend={args.backend} op={args.op} "
        f"repeats={args.repeats} elements={args.elements} n={args.iters}"
    )
    for name in (
        "e2e_ms",
        "usb_out_ms",
        "worker_validate_ms",
        "worker_decode_ms",
        "backend_set_ms",
        "backend_submit_ms",
        "backend_sync_ms",
        "backend_get_ms",
        "worker_encode_hash_ms",
        "worker_unattributed_ms",
        "response_path_residual_ms",
        "host_validate_ms",
    ):
        values = stages[name]
        print(
            f"  {name}: median={values['median_ms']:.6f} ms "
            f"p90={values['p90_ms']:.6f} ms"
        )


if __name__ == "__main__":
    main()
