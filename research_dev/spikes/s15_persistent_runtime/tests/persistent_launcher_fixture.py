#!/usr/bin/env python3
"""Subprocess fixture for persistent transport tests."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def emit(value: dict) -> None:
    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def mark(launch_id: int) -> None:
    value = json.dumps({"launch_id": launch_id}, sort_keys=True, separators=(",", ":"))
    sys.stderr.write("LAUNCHER_EXCHANGE_END " + value + "\n")
    sys.stderr.flush()


def main() -> int:
    mode = sys.argv[1]
    if mode == "crosstalk":
        emit({"unsolicited": True})
    ready = json.dumps({"mode": mode}, sort_keys=True, separators=(",", ":"))
    sys.stderr.write("LAUNCHER_READY " + ready + "\n")
    sys.stderr.flush()
    sequence = 0
    adapter = None
    for line in sys.stdin.buffer:
        request = json.loads(line)
        sequence += 1
        if mode == "physical_adapter":
            if adapter is None:
                here = Path(__file__).resolve().parent
                spike = here.parent
                s15 = spike.parent / "s15_runtime_dispatch"
                s14 = spike.parent / "s14_mixed_streaming_scheduler"
                sys.path[:0] = [str(spike), str(s15), str(s14)]
                from executor_contract import ExecutionRequest
                from physical_executor import PhysicalRouteBinding
                from session_adapter import (
                    PersistentWorkerCapability,
                    StageNetSessionAdapter,
                    StageNetSessionBinding,
                )
                session_binding = StageNetSessionBinding(
                    request["worker_binary_sha256"], 1, request["device_boot_id"],
                    tuple(request["layer_range"]), 48,
                )
                adapter = StageNetSessionAdapter(
                    session_binding,
                    (PersistentWorkerCapability(request["worker_binary_sha256"], 2, True),),
                )
            typed = ExecutionRequest(
                launch_id=request["launch_id"],
                route_id=request["route_id"],
                profile_id=request["profile_id"],
                device_id=request["device_id"],
                route_epoch=request["route_epoch"],
                residency_epoch=request["residency_epoch"],
                lease_epoch=request["lease_epoch"],
                device_boot_epoch=request["device_boot_epoch"],
                registry_generation=request["registry_generation"],
                compatibility_key=request["compatibility_key"],
                request_ids=tuple(request["request_ids"]),
                cohort_sha256=request["cohort_sha256"],
                input_manifest_sha256=request["input_manifest_sha256"],
                timeout_us=request["timeout_us"],
                expected_boundary_schema=request["expected_boundary_schema"],
            )
            binding = PhysicalRouteBinding(
                route_id=request["route_id"],
                profile_id=request["profile_id"],
                device_id=request["device_id"],
                protocol_version=request["protocol_version"],
                worker_binary_sha256=request["worker_binary_sha256"],
                worker_generation=request["worker_generation"],
                first_session_id=1,
                device_boot_id=request["device_boot_id"],
                cohort_sha256=request["cohort_sha256"],
                input_manifest_sha256=request["input_manifest_sha256"],
                layer_range=tuple(request["layer_range"]),
                expected_backend="HTP0",
            )
            worker_cert = {
                "schema": "ls-stagenet-session-v2",
                "proto_version": 2,
                "session_id": sequence,
                "session_end": "DETACH",
                "expected_backend": "HTP0",
                "worker_pid": 1234,
                "worker_boot_nonce": "0123456789abcdef",
                "device_boot_id": request["device_boot_id"],
                "layer_start": request["layer_range"][0],
                "layer_end": request["layer_range"][1],
                "n_layer": 48,
                "steps_session": len(request["request_ids"]),
                "steps_total": sequence * len(request["request_ids"]),
                "reset_applied": True,
                "missing_buffer_compute_nodes": 0,
                "compute_by_op_and_buffer": {"MUL_MAT": {"HTP0": 100}},
                "placement_status": "SCHEDULED_PLACEMENT_OK",
            }
            cert_payload = (
                "SESSIONCERT " + json.dumps(worker_cert, separators=(",", ":")) + "\n"
            ).encode("ascii")
            completion = [
                {
                    "request_id": request_id,
                    "identity_ok": True,
                    "epoch_ok": True,
                    "correctness_ok": True,
                    "d2h_complete": True,
                }
                for request_id in request["request_ids"]
            ]
            adapter.begin("DETACH")
            physical = adapter.accept(cert_payload, 0, False, typed, binding, completion)
            adapter.release_after_detach()
            mark(request["launch_id"])
            sys.stdout.buffer.write(physical)
            sys.stdout.buffer.flush()
            continue
        reply = {"launch_id": request["launch_id"], "sequence": sequence}
        if mode == "diagnostic":
            sys.stderr.write(f"diagnostic-{request['launch_id']}\n")
            sys.stderr.flush()
        if mode != "missing_marker":
            mark(request["launch_id"] + (1 if mode == "wrong_marker" else 0))
        if mode == "duplicate_marker":
            mark(request["launch_id"])
        if mode == "trailing_stderr":
            sys.stderr.write("late diagnostic\n")
            sys.stderr.flush()
        if mode == "late_duplicate" and sequence == 1:
            emit(reply)
            time.sleep(0.1)
            emit(reply)
        elif mode == "duplicate":
            payload = json.dumps(reply, sort_keys=True, separators=(",", ":")) + "\n"
            sys.stdout.write(payload + payload)
            sys.stdout.flush()
        elif mode == "partial":
            sys.stdout.write('{"partial":')
            sys.stdout.flush()
            time.sleep(2)
        elif mode == "missing":
            time.sleep(2)
        elif mode == "slow":
            time.sleep(0.4)
            emit(reply)
        else:
            emit(reply)
        if mode in ("duplicate", "partial", "missing", "stop"):
            return 0
    return 3 if mode == "nonzero" else 0


if __name__ == "__main__":
    raise SystemExit(main())
