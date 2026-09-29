#!/usr/bin/env python3
"""Validate the complete raw AOA asynchronous transport campaign."""

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA = "s41_aoa_async_transport_v1"
WORKLOADS = {
    "attention": (1308, 1384, 50, 300),
    "hidden_m1": (10268, 10344, 50, 300),
    "swiglu": (69660, 34920, 50, 300),
    "hidden_m8": (81948, 82024, 50, 300),
    "host_to_phone_1m": (1048604, 104, 20, 100),
    "phone_to_host_1m": (64, 1048680, 20, 100),
}
CONFIGURATIONS = {
    "serial_sync": ("serial", "sync", 1),
    "buffered_sync": ("buffered", "sync", 1),
    "buffered_async_q1": ("buffered", "async", 1),
    "buffered_async_q2": ("buffered", "async", 2),
    "buffered_async_q4": ("buffered", "async", 4),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    expected = {
        f"{workload}_r{repetition}_{configuration}"
        for workload in WORKLOADS
        for repetition in (1, 2, 3)
        for configuration in CONFIGURATIONS
    }
    raw_json = sorted(args.root.glob("*_r[123]_*.json"))
    seen = set()
    paid_samples = 0
    file_hashes = {}
    for path in raw_json:
        data = json.loads(path.read_text())
        if data.get("schema") != SCHEMA:
            raise RuntimeError(f"unexpected schema in {path}")
        case_name = data["case_name"]
        if path.stem != case_name or case_name not in expected:
            raise RuntimeError(f"unexpected case name in {path}")
        if case_name in seen:
            raise RuntimeError(f"duplicate case: {case_name}")
        seen.add(case_name)

        workload = data["workload"]
        request_bytes, response_bytes, warmup, iterations = WORKLOADS[workload]
        prefix = f"{workload}_r{data['repetition']}_"
        configuration = case_name.removeprefix(prefix)
        phone_mode, host_mode, queue_depth = CONFIGURATIONS[configuration]
        expected_fields = {
            "request_bytes": request_bytes,
            "response_bytes": response_bytes,
            "warmup": warmup,
            "iterations": iterations,
            "phone_mode": phone_mode,
            "mode": host_mode,
            "queue_depth": queue_depth,
            "preposted_in": host_mode == "async",
        }
        for field, value in expected_fields.items():
            if data.get(field) != value:
                raise RuntimeError(
                    f"unexpected {field} in {path}: {data.get(field)}"
                )
        for field in (
                "response_ready_samples_ms", "out_completion_samples_ms",
                "post_out_tail_samples_ms"):
            values = data[field]
            if len(values) != iterations:
                raise RuntimeError(f"sample count mismatch in {path}: {field}")
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise RuntimeError(f"invalid timing in {path}: {field}")

        log_path = path.with_suffix(".worker.log")
        log = log_path.read_text()
        configured = (
            f"[aoa-buffer] configured mode={phone_mode} "
            f"request={request_bytes} response={response_bytes} "
            f"requests={warmup + iterations} warmup={warmup} "
            f"depth={queue_depth}"
        )
        complete = (
            f"[aoa-buffer] complete status=0 "
            f"requests={warmup + iterations}"
        )
        if configured not in log or complete not in log:
            raise RuntimeError(f"worker lifecycle mismatch in {log_path}")
        if log.count("[aoa-buffer] read_us ") != 1 or \
                log.count("[aoa-buffer] write_us ") != 1:
            raise RuntimeError(f"worker timing mismatch in {log_path}")
        lowered = log.lower()
        if any(word in lowered for word in ("failed", "invalid", "killed")):
            raise RuntimeError(f"worker error text in {log_path}")

        paid_samples += iterations
        file_hashes[path.name] = sha256(path)
        file_hashes[log_path.name] = sha256(log_path)

    missing = expected - seen
    extra = seen - expected
    if missing or extra or len(raw_json) != len(expected):
        raise RuntimeError(
            f"campaign matrix mismatch: missing={sorted(missing)} "
            f"extra={sorted(extra)}"
        )

    environment = (args.root / "ENVIRONMENT.txt").read_text()
    required_environment = (
        "gpu=NVIDIA GeForce RTX 4060 Ti, "
        "GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08",
        "phone_serial=3C15AU002CL00000",
        "phone_boot_id=629dfd78-8bb3-4436-bdd4-17b410d100fb",
        "phone_usb_config=ptp,adb",
        "phone_usb_state=ptp,adb",
        "phone_usb_speed=super-speed",
        "phone_worker_pid=\nphone_binary_sha256=",
    )
    for value in required_environment:
        if value not in environment:
            raise RuntimeError(f"missing environment evidence: {value}")

    result = {
        "schema": "s41_aoa_async_transport_validation_v1",
        "verdict": "PASS",
        "raw_runs": len(raw_json),
        "worker_logs": len(raw_json),
        "paid_samples": paid_samples,
        "expected_cases": len(expected),
        "exact_response_validation": "enforced online by host",
        "phone_terminal_state": "ptp,adb; no aoa_buffered_daemon pid",
        "raw_file_sha256": dict(sorted(file_hashes.items())),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
