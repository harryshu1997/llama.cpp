#!/usr/bin/env python3
"""Exact-contract persistent child fixture for the S15 bridge tests."""

from __future__ import annotations

import argparse
import json
import os
import sys


COMMAND_KEYS = {"schema", "launch_id", "prompt", "n_gen", "request_count", "session_end"}


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def wire(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal")
    parser.add_argument("--boot", required=True)
    parser.add_argument("--layer-start", type=int, required=True)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--n-layer", type=int, default=48)
    args = parser.parse_args()
    sys.stderr.write("PERSISTENT_DRIVER_READY " + canonical({
        "schema": "layersplit-persistent-driver-v1",
        "host_pid": os.getpid(),
        "batch_size": 2,
        "max_n_gen": 8,
    }) + "\n")
    sys.stderr.flush()
    if args.mode == "unsolicited":
        sys.stdout.write(canonical({"unsolicited": True}) + "\n")
        sys.stdout.flush()
    session_id = 1
    steps_total = 0
    for raw in sys.stdin:
        command = json.loads(raw)
        if set(command) != COMMAND_KEYS \
                or command["schema"] != "layersplit-persistent-command-v1":
            return 3
        request_count = command["request_count"]
        n_gen = command["n_gen"]
        steps = request_count * n_gen
        steps_total += steps
        token_ids = [
            [1000 + command["launch_id"] * 100 + stream * 10 + index
             for index in range(n_gen)]
            for stream in range(request_count)
        ]
        cert = {
            "schema": "ls-stagenet-session-v2",
            "proto_version": 2,
            "session_id": session_id + (1 if args.mode == "cert_gap" else 0),
            "session_end": command["session_end"],
            "expected_backend": "HTP0",
            "worker_pid": 555 + (1 if args.mode == "changed_pid" and session_id > 1 else 0),
            "worker_boot_nonce": "0123456789abcdef",
            "device_boot_id": args.boot,
            "layer_start": args.layer_start,
            "layer_end": args.layer_end,
            "n_layer": args.n_layer,
            "steps_session": steps,
            "steps_total": steps_total,
            "reset_applied": command["session_end"] == "DETACH" and args.mode != "reset_false",
            "missing_buffer_compute_nodes": 0,
            "compute_by_op_and_buffer": {
                "GET_ROWS": {"CPU": request_count},
                "MUL_MAT": {"HTP0": 100},
            },
            "placement_status": "SCHEDULED_PLACEMENT_OK",
        }
        host_backend = "CUDA0"
        if args.mode == "host_cpu":
            host_backend = "CPU"
        elif args.mode == "host_wrong_backend":
            host_backend = "CUDA1"
        host_placement = {
            "schema": "layersplit-scheduled-placement-v2",
            "role": "host_tail",
            "mode": "pipedriver",
            "layer_start": 9 if args.mode == "host_range" else 8,
            "layer_end": 48,
            "n_layer": 48,
            "pid": os.getpid(),
            "run_rc": 0,
            "compute_nodes": 100,
            "copy_nodes": 0,
            "metadata_nodes": 12,
            "missing_buffer_compute_nodes": 0,
            "compute_by_buffer_type": {host_backend: 100},
            "compute_by_op": {"MUL_MAT": 100},
            "compute_by_op_and_buffer": {"MUL_MAT": {host_backend: 100}},
            "copy_by_buffer_type": {},
            "status": "SCHEDULED_PLACEMENT_OK",
        }
        result = {
            "schema": "layersplit-persistent-result-v1",
            "launch_id": command["launch_id"],
            "outcome": "completed",
            "host_pid": os.getpid() + (
                1 if args.mode == "changed_host_pid" and session_id > 1 else 0
            ),
            "request_count": request_count,
            "batch_size": request_count,
            "n_gen": n_gen,
            "session_end": command["session_end"],
            "elapsed_us": 1000 + command["launch_id"],
            "route_wall_us": 800 + command["launch_id"],
            "token_ids": token_ids,
        }
        if args.mode == "malformed_result":
            result["unknown"] = 1
        if args.mode == "wrong_token_count":
            result["token_ids"][0].pop()
        sys.stderr.write("host diagnostic\n")
        if args.mode != "missing_host_placement":
            placement_line = "PLACEMENTCERT " + wire(host_placement) + "\n"
            sys.stderr.write(placement_line)
            if args.mode == "duplicate_host_placement":
                sys.stderr.write(placement_line)
        cert_payload = canonical(cert)
        if args.mode == "noncanonical_session":
            cert_payload = json.dumps(cert, separators=(", ", ": "))
        elif args.mode == "duplicate_session_key":
            cert_payload = '{"schema":"duplicate",' + cert_payload[1:]
        sys.stderr.write("SESSIONCERT " + cert_payload + "\n")
        if args.mode != "missing_marker":
            marker_id = command["launch_id"] + (1 if args.mode == "wrong_marker" else 0)
            marker = canonical({"launch_id": marker_id})
            sys.stderr.write("PERSISTENT_DRIVER_EXCHANGE_END " + marker + "\n")
            if args.mode == "duplicate_marker":
                sys.stderr.write("PERSISTENT_DRIVER_EXCHANGE_END " + marker + "\n")
        if args.mode == "trailing_stderr":
            sys.stderr.write("late child diagnostic\n")
        sys.stderr.flush()
        payload = canonical(result) + "\n"
        if args.mode == "partial_result":
            sys.stdout.write(payload[:-1])
            sys.stdout.flush()
            return 0
        sys.stdout.write(payload)
        if args.mode in ("duplicate_result", "trailing_stdout"):
            sys.stdout.write(payload)
        sys.stdout.flush()
        session_id += 1
        if command["session_end"] == "STOP":
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
