#!/usr/bin/env python3
"""Independent validator for the real persistent OP15 B32 gate."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT = RESULTS / "report.json"
ARTIFACTS = HERE / "artifacts"
EXPECTED_SCOPE = "REAL_OP15_A6000_SEVEN_SESSIONS_SYNTHETIC_PAYLOAD_ENERGY_UNKNOWN"
EXPECTED_VERDICT = "PERSISTENT_OP15_B32_MECHANICS_PASS_ENERGY_UNKNOWN"
EXPECTED_SHARD = "sha256:a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8"
SESSION_KEYS = {
    "schema", "proto_version", "session_id", "session_end", "expected_backend",
    "worker_pid", "worker_boot_nonce", "device_boot_id", "layer_start", "layer_end",
    "n_layer", "steps_session", "steps_total", "reset_applied",
    "missing_buffer_compute_nodes", "compute_by_op_and_buffer", "placement_status",
}


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def strict_bytes(payload: bytes, label: str) -> object:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def constant(value):
        fail(f"invalid JSON constant in {label}: {value}")

    try:
        return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON in {label}: {exc}") from exc


def strict_object(path: Path) -> dict:
    value = strict_bytes(path.read_bytes(), str(path))
    if type(value) is not dict:
        fail(f"{path} is not an object")
    return value


def prefixed(path: Path, prefix: bytes) -> list[dict]:
    values = []
    for line in path.read_bytes().splitlines():
        if line.startswith(prefix):
            value = strict_bytes(line[len(prefix):], str(path))
            if type(value) is not dict:
                fail(f"non-object prefixed record in {path}")
            values.append(value)
    return values


def validate_artifacts(report: dict) -> None:
    manifest = ARTIFACTS / "SHA256SUMS.txt"
    declared = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        value, relative = line.split("  ", 1)
        if not relative.startswith("./") or relative in declared \
                or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            fail("invalid artifact manifest")
        declared[relative] = value
    actual = {
        "./" + str(path.relative_to(ARTIFACTS)): digest(path)[7:]
        for path in sorted(ARTIFACTS.rglob("*"))
        if path.is_file() and path != manifest
    }
    if declared != actual:
        fail("artifact manifest does not match frozen files")
    expected = {
        "android_worker": digest(ARTIFACTS / "android/llama-layersplit"),
        "host_worker": digest(ARTIFACTS / "host/llama-layersplit"),
        "source": digest(ARTIFACTS / "source/layersplit.cpp"),
        "phone_shard": EXPECTED_SHARD,
    }
    if report.get("artifacts") != expected:
        fail("report artifact identity failed")

    remote = {}
    for line in (RESULTS / "remote_sha256.txt").read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9+_.-]+)", line)
        if match is None or match.group(2) in remote:
            fail("invalid remote digest evidence")
        remote[match.group(2)] = match.group(1)
    local_android = {
        path.name: digest(path)[7:]
        for path in (ARTIFACTS / "android").iterdir() if path.is_file()
    }
    if remote != local_android:
        fail("remote Android runtime differs from the frozen bundle")


def validate_certificate(cert: dict, session_id: int, boot_id: str,
                         worker_pid: int, nonce: str) -> None:
    if set(cert) != SESSION_KEYS:
        fail("session certificate key set failed")
    end = "STOP" if session_id == 7 else "DETACH"
    expected = {
        "schema": "ls-stagenet-session-v2",
        "proto_version": 2,
        "session_id": session_id,
        "session_end": end,
        "expected_backend": "HTP0",
        "worker_pid": worker_pid,
        "worker_boot_nonce": nonce,
        "device_boot_id": boot_id,
        "layer_start": 0,
        "layer_end": 8,
        "n_layer": 48,
        "steps_session": 384,
        "steps_total": 384 * session_id,
        "reset_applied": session_id != 7,
        "missing_buffer_compute_nodes": 0,
        "placement_status": "SCHEDULED_PLACEMENT_OK",
    }
    for key, value in expected.items():
        if type(cert.get(key)) is not type(value) or cert.get(key) != value:
            fail(f"session {session_id} certificate mismatch: {key}")
    placement = cert.get("compute_by_op_and_buffer")
    if type(placement) is not dict or not placement:
        fail(f"session {session_id} placement is empty")
    for op, buffers in placement.items():
        if type(op) is not str or type(buffers) is not dict or not buffers:
            fail(f"session {session_id} placement entry failed")
        for buffer_name, count in buffers.items():
            if type(count) is not int or count <= 0 \
                    or (buffer_name != "HTP0" and not (op == "GET_ROWS" and buffer_name == "CPU")):
                fail(f"session {session_id} has undeclared fallback")


def validate() -> dict:
    report = strict_object(REPORT)
    if report.get("schema") != "s15-persistent-b32-gate-v1" \
            or report.get("verdict") != EXPECTED_VERDICT \
            or report.get("scope") != EXPECTED_SCOPE \
            or report.get("problems") != []:
        fail("report scope or verdict failed")
    boot_id = report.get("boot_id")
    worker_pid = report.get("resident_worker_pid")
    nonce = report.get("resident_worker_nonce")
    if type(boot_id) is not str or not boot_id or type(worker_pid) is not int \
            or worker_pid <= 0 or type(nonce) is not str or not nonce:
        fail("resident worker identity failed")
    validate_artifacts(report)

    reference_rows = prefixed(RESULTS / "reference.stderr.bin", b"ROUTEJSON ")
    if len(reference_rows) != 32:
        fail("same-batch CUDA reference is incomplete")
    reference = reference_rows[0].get("token_ids")
    if type(reference) is not list or len(reference) != 8 \
            or any(type(token) is not int for token in reference) \
            or any(row.get("token_ids") != reference for row in reference_rows):
        fail("same-batch CUDA reference disagrees")

    raw_certs = prefixed(RESULTS / "phone.stderr.bin", b"SESSIONCERT ")
    sessions = report.get("sessions")
    if type(sessions) is not list or len(sessions) != 7 or len(raw_certs) != 7:
        fail("session count failed")
    for index, (session, raw_cert) in enumerate(zip(sessions, raw_certs), 1):
        if type(session) is not dict or session.get("certificate") != raw_cert:
            fail(f"session {index} report differs from raw certificate")
        validate_certificate(raw_cert, index, boot_id, worker_pid, nonce)
        end = "STOP" if index == 7 else "DETACH"
        if session.get("session_id") != index or session.get("session_end") != end \
                or session.get("token_ids") != reference \
                or type(session.get("elapsed_us")) is not int \
                or not 0 < session["elapsed_us"] <= 4_000_000:
            fail(f"session {index} summary failed")
        directory = RESULTS / f"session-{index}"
        command = strict_bytes((directory / "command.json").read_bytes(), "host command")
        expected_end = end.lower()
        if type(command) is not list or "--prompt-after-load" not in command \
                or "-p" in command or "--session-end" not in command \
                or command[command.index("--session-end") + 1] != expected_end:
            fail(f"session {index} host command failed")
        stderr = directory / "host.stderr.bin"
        payload = stderr.read_bytes()
        ready = payload.find(b"DRIVER_INPUT_READY ")
        accepted = payload.find(b"DRIVER_INPUT_ACCEPTED ")
        done = payload.find(b"DRIVER_DONE ")
        rows = prefixed(stderr, b"ROUTEJSON ")
        if not 0 <= ready < accepted < done or len(rows) != 32 \
                or sorted(row.get("stream_index") for row in rows) != list(range(32)) \
                or any(row.get("status") != "ok" or row.get("batch_size") != 32
                       or row.get("token_ids") != reference for row in rows) \
                or session.get("route_wall_us_max") != max(row["request_wall_us"] for row in rows):
            fail(f"session {index} physical output failed")
    return report


def main() -> int:
    report = validate()
    print(
        "VALID_PERSISTENT_OP15_B32 "
        f"sessions={len(report['sessions'])} "
        f"max_elapsed_us={max(value['elapsed_us'] for value in report['sessions'])} "
        "energy=UNKNOWN"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
