#!/usr/bin/env python3

import argparse
import json
import math
import socket
import struct
import time


MAGIC = 0x4C484431
VERSION = 1
N_EMBD = 3840
ROWS = 46080
TOP_K = 32


def hash_bytes(data):
    value = 2166136261
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def receive_exact(sock, size):
    result = bytearray()
    while len(result) < size:
        chunk = sock.recv(size - len(result))
        if not chunk:
            raise RuntimeError("short socket read")
        result.extend(chunk)
    return bytes(result)


def connect_worker(host, port):
    sock = socket.create_connection((host, port), timeout=10)
    sock.sendall(struct.pack(
        "<IHHIIIHH", MAGIC, VERSION, 1, N_EMBD, ROWS, TOP_K, 1, 0))
    hello = struct.unpack("<IHHHHIIIIII4xQ", receive_exact(sock, 48))
    if hello[0] != MAGIC or hello[2] != 2 or hello[3] != 0:
        sock.close()
        raise RuntimeError("hello failed")
    return sock, hello


def execute(sock, request_id, payload):
    header = struct.pack(
        "<IHHIIII", MAGIC, VERSION, 3, request_id, N_EMBD,
        len(payload), hash_bytes(payload))
    start_ns = time.perf_counter_ns()
    sock.sendall(header + payload)
    response = struct.unpack("<IHHHHIIII4xQQ", receive_exact(sock, 48))
    body = receive_exact(sock, response[7])
    wall_us = (time.perf_counter_ns() - start_ns) / 1000.0
    candidates = [
        struct.unpack_from("<If", body, index * 8)
        for index in range(response[6])
    ]
    return response, body, candidates, wall_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--pattern", choices=("same", "aba"), default="same")
    parser.add_argument("--reference-port", type=int)
    args = parser.parse_args()

    payload_a = b"".join(
        struct.pack("<e", math.sin(index * 0.017) * 0.125)
        for index in range(N_EMBD)
    )
    payload_b = b"".join(
        struct.pack("<e", math.cos(index * 0.013) * 0.09375)
        for index in range(N_EMBD)
    )
    if args.pattern == "aba" and args.requests != 3:
        parser.error("--pattern aba requires --requests 3")
    payloads = (payload_a, payload_b, payload_a) if args.pattern == "aba" else None
    sock, hello = connect_worker(args.host, args.port)
    reference_sock = None
    try:
        if args.reference_port is not None:
            reference_sock, reference_hello = connect_worker(
                args.host, args.reference_port)
            if hello[5:11] != reference_hello[5:11] or hello[11] != reference_hello[11]:
                raise RuntimeError("worker metadata mismatch")
        for request_id in range(1, args.requests + 1):
            payload = payloads[request_id - 1] if payloads is not None else payload_a
            response, body, candidates, wall_us = execute(
                sock, request_id, payload)
            result = {
                "request_id": request_id,
                "input": "ABA"[request_id - 1] if payloads is not None else "A",
                "input_hash": f"{hash_bytes(payload):08x}",
                "status": response[3],
                "compute_us": response[9],
                "reduce_us": response[10],
                "wall_us": wall_us,
                "payload_hash_ok": hash_bytes(body) == response[8],
                "weight_type": hello[9],
                "weight_hash": f"{hello[11]:016x}",
                "top_ids": [candidate[0] for candidate in candidates[:8]],
                "top_scores": [candidate[1] for candidate in candidates[:8]],
            }
            if reference_sock is not None:
                ref_response, ref_body, ref_candidates, ref_wall_us = execute(
                    reference_sock, request_id, payload)
                reference_scores = dict(ref_candidates)
                common_ids = set(dict(candidates)) & set(reference_scores)
                diff2 = sum(
                    (score - reference_scores[token_id]) ** 2
                    for token_id, score in candidates if token_id in common_ids)
                ref2 = sum(reference_scores[token_id] ** 2 for token_id in common_ids)
                result.update({
                    "reference_status": ref_response[3],
                    "reference_compute_us": ref_response[9],
                    "reference_reduce_us": ref_response[10],
                    "reference_wall_us": ref_wall_us,
                    "reference_payload_hash_ok":
                        hash_bytes(ref_body) == ref_response[8],
                    "top32_overlap": len(common_ids),
                    "top8_order_equal":
                        [item[0] for item in candidates[:8]] ==
                        [item[0] for item in ref_candidates[:8]],
                    "top8_set_equal":
                        {item[0] for item in candidates[:8]} ==
                        {item[0] for item in ref_candidates[:8]},
                    "common_score_nmse": diff2 / ref2 if ref2 else diff2,
                    "common_score_max_abs": max(
                        (abs(score - reference_scores[token_id])
                         for token_id, score in candidates if token_id in common_ids),
                        default=0.0),
                })
            print(json.dumps(result, separators=(",", ":")))
    finally:
        sock.close()
        if reference_sock is not None:
            reference_sock.close()


if __name__ == "__main__":
    main()
