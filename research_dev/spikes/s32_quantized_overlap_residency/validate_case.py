#!/usr/bin/env python3
"""Bind an S32 probe to its model hash and realized phone placement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SCHEMA = "s32-quantized-residency-case-v1"
HASH_RE = re.compile(rb"^([0-9a-f]{64})  ([^\r\n]+)\r?\n$")


class CaseError(ValueError):
    pass


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CaseError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw, object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaseError(f"invalid {label} JSON") from exc
    if type(value) is not dict:
        raise CaseError(f"{label} must be an object")
    return value


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def one_record(log: bytes, prefix: bytes, label: str) -> dict[str, object]:
    rows = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise CaseError(f"expected exactly one {label}")
    return decode_json(rows[0], label)


def require_int(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaseError(f"{label} must be an integer >= {minimum}")
    return value


def validate_compute_map(value: object) -> None:
    if type(value) is not dict or not value:
        raise CaseError("compute_by_op_and_buffer must be a nonempty object")
    for op, buffers in value.items():
        if type(op) is not str or not op or type(buffers) is not dict or not buffers:
            raise CaseError("compute placement map is malformed")
        for backend, count in buffers.items():
            if backend not in {"HTP0", "CPU"}:
                raise CaseError(f"unexpected compute backend: {backend}")
            require_int(count, f"{op}/{backend} count", 1)
            if backend == "CPU" and op != "GET_ROWS":
                raise CaseError(f"CPU fallback is not allowed for {op}")


def validate_case(
    probe_path: Path,
    log_path: Path,
    remote_hash_path: Path,
    expected_start: int,
    expected_end: int,
    expected_quantization: str,
    expected_model_sha256: str,
    expected_remote_model: str,
) -> dict[str, object]:
    probe_raw = probe_path.read_bytes()
    probe = decode_json(probe_raw, "probe")
    if probe_raw != canonical(probe):
        raise CaseError("probe is not canonical JSONL")
    if probe.get("schema") != "s32-quantized-residency-probe-v1":
        raise CaseError("probe schema mismatch")
    if probe.get("status") not in {"PROBE_PASS", "PROBE_PERF_ONLY"}:
        raise CaseError("probe did not complete its resource gates")
    if probe.get("scheduler_eligible") is not False:
        raise CaseError("raw probe must not self-authorize scheduler eligibility")
    if probe.get("resource_failures") != []:
        raise CaseError("probe has a resource failure")
    if probe.get("layer_range") != [expected_start, expected_end]:
        raise CaseError("probe layer range mismatch")
    if probe.get("quantization") != expected_quantization:
        raise CaseError("probe quantization mismatch")
    if probe.get("model_sha256") != expected_model_sha256:
        raise CaseError("probe model digest mismatch")
    batch = require_int(probe.get("batch"), "batch", 32)
    warmups = require_int(probe.get("warmups_discarded"), "warmups", 1)
    reps = require_int(probe.get("reps"), "reps", 7)
    steps = probe.get("steps")
    cohorts = probe.get("measured_cohorts")
    if type(steps) is not list or not steps or type(cohorts) is not list or len(cohorts) != reps:
        raise CaseError("probe cohort dimensions are invalid")
    timing = probe.get("b32_step_us")
    if type(timing) is not dict or require_int(timing.get("count"), "timing count", 1) != reps * len(steps):
        raise CaseError("probe timing count mismatch")
    memory_after = probe.get("memory_after")
    if type(memory_after) is not dict or type(memory_after.get("process_kib")) is not dict:
        raise CaseError("probe memory evidence is missing")
    if memory_after["process_kib"].get("VmSwap") != 0:
        raise CaseError("probe used swap")

    digest_columns: list[list[str]] = []
    for index, cohort in enumerate(cohorts):
        if type(cohort) is not dict or cohort.get("rep") != index:
            raise CaseError("probe cohort order mismatch")
        digests = cohort.get("activation_sha256")
        latencies = cohort.get("step_us")
        if type(digests) is not list or len(digests) != len(steps):
            raise CaseError("probe activation digest dimensions are invalid")
        if type(latencies) is not list or len(latencies) != len(steps):
            raise CaseError("probe latency dimensions are invalid")
        if not all(type(item) is str and re.fullmatch(r"[0-9a-f]{64}", item) for item in digests):
            raise CaseError("probe activation digest is invalid")
        if not all(type(item) is int and item > 0 for item in latencies):
            raise CaseError("probe latency is invalid")
        digest_columns.append(digests)
    for position in range(len(steps)):
        if len({row[position] for row in digest_columns}) != 1:
            raise CaseError("phone output changed across measured repetitions")

    hash_raw = remote_hash_path.read_bytes()
    match = HASH_RE.fullmatch(hash_raw)
    if match is None:
        raise CaseError("remote model hash record is malformed")
    if match.group(1).decode("ascii") != expected_model_sha256:
        raise CaseError("remote model hash mismatch")
    if match.group(2).decode("ascii") != expected_remote_model:
        raise CaseError("remote model path mismatch")

    log_raw = log_path.read_bytes()
    session = one_record(log_raw, b"SESSIONCERT ", "SESSIONCERT")
    placement = one_record(log_raw, b"PLACEMENTCERT ", "PLACEMENTCERT")
    for record, label in ((session, "session"), (placement, "placement")):
        if record.get("layer_start") != expected_start or record.get("layer_end") != expected_end:
            raise CaseError(f"{label} layer range mismatch")
        if record.get("missing_buffer_compute_nodes") != 0:
            raise CaseError(f"{label} has missing-buffer compute")
        validate_compute_map(record.get("compute_by_op_and_buffer"))
    if session.get("session_end") != "STOP" or session.get("placement_status") != "SCHEDULED_PLACEMENT_OK":
        raise CaseError("session did not stop with valid placement")
    if placement.get("status") != "SCHEDULED_PLACEMENT_OK" or placement.get("run_rc") != 0:
        raise CaseError("placement certificate did not pass")
    if session.get("compute_by_op_and_buffer") != placement.get("compute_by_op_and_buffer"):
        raise CaseError("session and placement compute maps differ")
    expected_rows = batch * (warmups + reps) * len(steps)
    if session.get("steps_session") != expected_rows:
        raise CaseError("session row count mismatch")
    compute_nodes = require_int(placement.get("compute_nodes"), "compute nodes", 1)
    by_buffer = placement.get("compute_by_buffer_type")
    if type(by_buffer) is not dict or sum(
        require_int(count, f"{backend} compute count") for backend, count in by_buffer.items()
    ) != compute_nodes:
        raise CaseError("placement compute totals do not balance")

    numeric = probe.get("status") == "PROBE_PASS" and probe.get("correctness_status") == "PASS"
    return {
        "schema": SCHEMA,
        "status": "CASE_PASS" if numeric else "CAPACITY_PERF_PASS_NUMERIC_BLOCKED",
        "scheduler_eligible": numeric,
        "layer_range": [expected_start, expected_end],
        "quantization": expected_quantization,
        "model_sha256": expected_model_sha256,
        "batch": batch,
        "b32_step_us": timing,
        "b32_per_layer_median_us": probe.get("b32_per_layer_median_us"),
        "phone_vmhwm_kib": memory_after["process_kib"].get("VmHWM"),
        "phone_vmswap_kib": 0,
        "compute_nodes": compute_nodes,
        "compute_by_buffer_type": by_buffer,
        "artifacts": {
            "probe_sha256": sha256(probe_raw),
            "phone_log_sha256": sha256(log_raw),
            "remote_hash_record_sha256": sha256(hash_raw),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--phone-log", type=Path, required=True)
    parser.add_argument("--remote-hash-record", type=Path, required=True)
    parser.add_argument("--expected-start", type=int, required=True)
    parser.add_argument("--expected-end", type=int, required=True)
    parser.add_argument("--quantization", choices=("Q8_0", "Q4_0"), required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--remote-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    try:
        result = validate_case(
            args.probe,
            args.phone_log,
            args.remote_hash_record,
            args.expected_start,
            args.expected_end,
            args.quantization,
            args.model_sha256,
            args.remote_model,
        )
    except (OSError, CaseError) as exc:
        print(f"S32_CASE_FAIL: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result))
    print(canonical(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
