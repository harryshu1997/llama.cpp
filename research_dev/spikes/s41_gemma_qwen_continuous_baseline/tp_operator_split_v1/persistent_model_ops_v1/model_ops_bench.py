#!/usr/bin/env python3
"""Acquire changing-input Qwen3-14B-shaped phone operator traces."""

import argparse
import ctypes
import json
import math
import statistics
import struct
import sys
import time
import zlib
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aoa_bench import load, open_accessory


REQUEST_MAGIC = 0x534D4F51
RESPONSE_MAGIC = 0x534D4F52
PROTOCOL_VERSION = 1
REQUEST = struct.Struct("<IHHIIIII")
RESPONSE = struct.Struct("<IHHIIIII4xQQQQQQQQQ")
OUT_ENDPOINT = 0x01
IN_ENDPOINT = 0x81

HIDDEN = 5120
INTERMEDIATE = 17408
HEAD_DIM = 128
GQA = 5
OPCODES = {"rmsnorm": 1, "swiglu": 2, "attention": 3}
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


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def pack_half(values):
    return struct.pack(f"<{len(values)}e", *values)


def unpack_half(data):
    return struct.unpack(f"<{len(data) // 2}e", data)


def input_values(op, variant):
    if op == "rmsnorm":
        return [
            (((index * 29 + variant * 31 + 7) % 257) - 128) / 128.0
            for index in range(HIDDEN)
        ]
    if op == "swiglu":
        gate = [
            (((index * 23 + variant * 37 + 5) % 257) - 128) / 64.0
            for index in range(INTERMEDIATE)
        ]
        up = [
            (((index * 19 + variant * 41 + 17) % 257) - 128) / 128.0
            for index in range(INTERMEDIATE)
        ]
        return gate + up
    return [
        (((index * 31 + variant * 43 + 13) % 257) - 128) / 256.0
        for index in range(GQA * HEAD_DIM)
    ]


def rmsnorm_expected(inputs):
    mean_square = sum(value * value for value in inputs) / HIDDEN
    inverse = 1.0 / math.sqrt(mean_square + 1.0e-6)
    result = []
    for index, value in enumerate(inputs):
        centered = ((index * 17 + 3) % 33) - 16
        weight = half(1.0 + centered / 2048.0)
        result.append(value * inverse * weight)
    return result


def swiglu_expected(inputs):
    result = []
    for index in range(INTERMEDIATE):
        gate = inputs[index]
        up = inputs[INTERMEDIATE + index]
        result.append((gate / (1.0 + math.exp(-gate))) * up)
    return result


def make_attention_cache(n_kv):
    key = array("f")
    value = array("f")
    for token in range(n_kv):
        for dimension in range(HEAD_DIM):
            key.append(
                (((token * 13 + dimension * 7 + 11) % 257) - 128) / 256.0
            )
            value.append(
                (((token * 5 + dimension * 11 + 19) % 257) - 128) / 512.0
            )
    return key, value


def attention_expected(inputs, n_kv, key, value):
    result = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for head in range(GQA):
        query = inputs[head * HEAD_DIM:(head + 1) * HEAD_DIM]
        scores = [0.0] * n_kv
        maximum = -math.inf
        for token in range(n_kv):
            base = token * HEAD_DIM
            dot = 0.0
            for dimension in range(HEAD_DIM):
                dot += query[dimension] * key[base + dimension]
            dot *= scale
            scores[token] = dot
            maximum = max(maximum, dot)
        total = 0.0
        for token, score in enumerate(scores):
            probability = math.exp(score - maximum)
            scores[token] = probability
            total += probability
        inverse = 1.0 / total
        for dimension in range(HEAD_DIM):
            item = 0.0
            for token, probability in enumerate(scores):
                item += probability * value[token * HEAD_DIM + dimension]
            result.append(item * inverse)
    return result


def make_reference(op, n_kv, variants):
    key = value = None
    if op == "attention":
        key, value = make_attention_cache(n_kv)
    cases = []
    for variant in range(variants):
        payload = pack_half(input_values(op, variant))
        rounded_input = unpack_half(payload)
        if op == "rmsnorm":
            expected = rmsnorm_expected(rounded_input)
        elif op == "swiglu":
            expected = swiglu_expected(rounded_input)
        else:
            expected = attention_expected(rounded_input, n_kv, key, value)
        expected_bytes = pack_half(expected)
        cases.append({
            "variant": variant,
            "payload_hex": payload.hex(),
            "payload_crc32": zlib.crc32(payload) & 0xFFFFFFFF,
            "expected_hex": expected_bytes.hex(),
            "expected_crc32": zlib.crc32(expected_bytes) & 0xFFFFFFFF,
        })
    return {
        "schema": "s41_persistent_model_ops_reference_v1",
        "model": "Qwen3-14B",
        "op": op,
        "opcode": OPCODES[op],
        "n_kv": n_kv,
        "variants": variants,
        "input_elements": len(bytes.fromhex(cases[0]["payload_hex"])) // 2,
        "output_elements": len(bytes.fromhex(cases[0]["expected_hex"])) // 2,
        "cases": cases,
    }


def relative_error(actual, expected):
    squared_error = 0.0
    squared_reference = 0.0
    maximum = 0.0
    for observed, wanted in zip(actual, expected):
        difference = observed - wanted
        squared_error += difference * difference
        squared_reference += wanted * wanted
        maximum = max(maximum, abs(difference))
    return math.sqrt(squared_error / max(squared_reference, 1.0e-30)), maximum


def acquire(args, reference):
    if (
        reference["op"] != args.op
        or reference["n_kv"] != args.n_kv
        or reference["variants"] < 2
    ):
        raise ValueError("reference does not match acquisition")
    cases = []
    for case in reference["cases"]:
        cases.append({
            **case,
            "payload": bytes.fromhex(case["payload_hex"]),
            "expected": bytes.fromhex(case["expected_hex"]),
        })
    input_elements = reference["input_elements"]
    output_elements = reference["output_elements"]
    output_bytes = output_elements * 2
    response_size = RESPONSE.size + output_bytes

    libusb, context = load()
    handle, pid = open_accessory(libusb, context)
    if not handle:
        raise SystemExit("no accessory-mode device found")
    libusb.libusb_detach_kernel_driver(ctypes.c_void_p(handle), 0)
    status = libusb.libusb_claim_interface(ctypes.c_void_p(handle), 0)
    if status != 0:
        raise SystemExit(f"claim_interface failed: {status}")

    samples = []
    variant_crcs = {}
    request_id = 1
    total_requests = args.warmup + args.iters
    tolerance = {"rmsnorm": 0.01, "swiglu": 0.01, "attention": 0.03}[args.op]
    max_abs_tolerance = {"rmsnorm": 0.03, "swiglu": 0.03, "attention": 0.03}[args.op]
    try:
        for index in range(total_requests):
            case = cases[index % len(cases)]
            payload = case["payload"]
            total_started = time.perf_counter_ns()
            prepare_started = time.perf_counter_ns()
            header = REQUEST.pack(
                REQUEST_MAGIC, PROTOCOL_VERSION, OPCODES[args.op], request_id,
                input_elements, output_elements, len(payload),
                case["payload_crc32"]
            )
            request_data = (
                ctypes.c_ubyte * (len(header) + len(payload))
            ).from_buffer_copy(header + payload)
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
            (
                magic, version, response_status, response_id, response_op,
                response_elements, returned_bytes, output_crc,
            ) = fields[:8]
            if (
                magic != RESPONSE_MAGIC
                or version != PROTOCOL_VERSION
                or response_status != 0
                or response_id != request_id
                or response_op != OPCODES[args.op]
                or response_elements != output_elements
                or returned_bytes != output_bytes
                or output_crc != (zlib.crc32(output) & 0xFFFFFFFF)
            ):
                raise RuntimeError(f"invalid response for request {request_id}")
            actual_values = unpack_half(output)
            expected_values = unpack_half(case["expected"])
            rel_l2, max_abs = relative_error(actual_values, expected_values)
            if rel_l2 > tolerance or max_abs > max_abs_tolerance:
                raise RuntimeError(
                    f"semantic mismatch request={request_id} "
                    f"rel_l2={rel_l2:.6g} max_abs={max_abs:.6g}"
                )
            previous_crc = variant_crcs.setdefault(case["variant"], output_crc)
            if previous_crc != output_crc:
                raise RuntimeError(
                    f"non-repeatable output for variant {case['variant']}"
                )
            validate_ms = (time.perf_counter_ns() - validate_started) / 1e6
            total_ms = (time.perf_counter_ns() - total_started) / 1e6

            if index >= args.warmup:
                sample = {
                    "variant": case["variant"],
                    "output_crc32": output_crc,
                    "relative_l2": rel_l2,
                    "max_abs": max_abs,
                    "host_prepare_ms": prepare_ms,
                    "usb_out_ms": usb_out_ms,
                    "usb_in_wait_ms": usb_in_ms,
                    "host_validate_ms": validate_ms,
                    "e2e_ms": total_ms,
                }
                for name, value_ns in zip(STAGE_NAMES, fields[8:]):
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
        if name not in {"variant", "output_crc32"}
    }
    return {
        "schema": "s41_persistent_model_ops_acquisition_v1",
        "model": "Qwen3-14B",
        "op": args.op,
        "backend": args.backend,
        "n_kv": args.n_kv,
        "input_elements": input_elements,
        "output_elements": output_elements,
        "input_bytes": input_elements * 2,
        "output_bytes": output_bytes,
        "warmup": args.warmup,
        "iterations": args.iters,
        "variants": reference["variants"],
        "semantic_matches": args.iters,
        "max_relative_l2": max(sample["relative_l2"] for sample in samples),
        "max_abs_error": max(sample["max_abs"] for sample in samples),
        "variant_output_crc32": {
            str(key): value for key, value in sorted(variant_crcs.items())
        },
        "stages": stages,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=tuple(OPCODES), required=True)
    parser.add_argument("--backend")
    parser.add_argument("--n-kv", type=int, default=8192)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.n_kv <= 0 or args.variants < 2 or args.warmup < 0 or args.iters <= 0:
        parser.error("invalid acquisition dimensions")

    reference_path = Path(args.reference)
    if args.reference_only:
        reference = make_reference(args.op, args.n_kv, args.variants)
        reference_path.write_text(
            json.dumps(reference, indent=2, sort_keys=True) + "\n"
        )
        return
    if not args.backend or not args.output:
        parser.error("acquisition requires --backend and --output")
    reference = json.loads(reference_path.read_text())
    result = acquire(args, reference)
    Path(args.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "backend": args.backend,
        "op": args.op,
        "e2e_median_ms": result["stages"]["e2e_ms"]["median_ms"],
        "e2e_p90_ms": result["stages"]["e2e_ms"]["p90_ms"],
        "backend_median_ms": (
            result["stages"]["backend_submit_ms"]["median_ms"]
            + result["stages"]["backend_sync_ms"]["median_ms"]
        ),
        "max_relative_l2": result["max_relative_l2"],
        "semantic_matches": result["semantic_matches"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
