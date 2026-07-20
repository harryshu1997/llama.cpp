#!/usr/bin/env python3
"""Independent validator for the persistent host-tail physical gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPORT = HERE / "results/report.json"
EXPECTED_VERDICT = "PERSISTENT_OP15_B32_HOST_AND_PHONE_PASS_ENERGY_UNKNOWN"


class ValidationError(ValueError):
    pass


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def strict_object(payload: bytes, label: str) -> dict:
    def no_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValidationError(f"duplicate key in {label}: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(payload, object_pairs_hook=no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label}: {exc}") from exc
    if type(value) is not dict:
        raise ValidationError(f"{label} must be an object")
    return value


def prefixed(path: Path, prefix: bytes) -> list[dict]:
    return [
        strict_object(line[len(prefix):], prefix.decode("ascii").strip())
        for line in path.read_bytes().splitlines()
        if line.startswith(prefix)
    ]


def integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def validate(data: dict) -> None:
    expected_keys = {
        "schema", "verdict", "scope", "artifacts", "boot_id", "host_pid",
        "resident_worker_pid", "resident_worker_nonce", "phone_command",
        "host_command", "thermal_start", "thermal_end", "reference_tokens",
        "sessions", "problems",
    }
    if type(data) is not dict or set(data) != expected_keys:
        raise ValidationError("report has an invalid key set")
    if data["schema"] != "s15-persistent-host-tail-gate-v1" \
            or data["verdict"] != EXPECTED_VERDICT \
            or data["scope"] != "REAL_OP15_A6000_TWO_EXCHANGES_SYNTHETIC_PAYLOAD_ENERGY_UNKNOWN" \
            or data["problems"] != []:
        raise ValidationError("report status fields are invalid")
    artifacts = data["artifacts"]
    expected_artifacts = {
        "android_worker": digest(HERE / "artifacts/android/llama-layersplit"),
        "host_worker": digest(HERE / "artifacts/host/llama-layersplit"),
        "source": digest(HERE / "artifacts/source/layersplit.cpp"),
    }
    if type(artifacts) is not dict or set(artifacts) != {
        "android_worker", "host_worker", "source", "phone_shard",
    }:
        raise ValidationError("artifact binding has an invalid key set")
    for key, value in expected_artifacts.items():
        if artifacts[key] != value:
            raise ValidationError(f"artifact digest mismatch: {key}")
    if artifacts["phone_shard"] != \
            "sha256:a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8":
        raise ValidationError("phone shard digest mismatch")
    host_pid = integer("host_pid", data["host_pid"], 1)
    worker_pid = integer("resident_worker_pid", data["resident_worker_pid"], 1)
    if type(data["boot_id"]) is not str or not data["boot_id"] \
            or type(data["resident_worker_nonce"]) is not str \
            or not data["resident_worker_nonce"]:
        raise ValidationError("resident process identity is incomplete")
    reference = data["reference_tokens"]
    if type(reference) is not list or len(reference) != 8 \
            or any(type(token) is not int or token < 0 for token in reference):
        raise ValidationError("reference token list is invalid")
    sessions = data["sessions"]
    if type(sessions) is not list or len(sessions) != 2:
        raise ValidationError("report must contain two sessions")
    previous_steps = 0
    expected_ends = ("DETACH", "STOP")
    for index, (session, end) in enumerate(zip(sessions, expected_ends), 1):
        if type(session) is not dict or set(session) != {
            "launch_id", "session_end", "wall_us", "result",
            "host_placement", "phone_session",
        }:
            raise ValidationError("session has an invalid key set")
        if session["launch_id"] != index or session["session_end"] != end:
            raise ValidationError("session order or ending is invalid")
        integer("session wall_us", session["wall_us"], 1)
        result = session["result"]
        if result.get("schema") != "layersplit-persistent-result-v1" \
                or result.get("launch_id") != index \
                or result.get("outcome") != "completed" \
                or result.get("host_pid") != host_pid \
                or result.get("request_count") != 32 \
                or result.get("batch_size") != 32 \
                or result.get("n_gen") != 8 \
                or result.get("session_end") != end:
            raise ValidationError("persistent result identity mismatch")
        if integer("result elapsed_us", result.get("elapsed_us"), 1) > 4_000_000:
            raise ValidationError("persistent result exceeded the latency gate")
        integer("route_wall_us", result.get("route_wall_us"), 1)
        tokens = result.get("token_ids")
        if type(tokens) is not list or len(tokens) != 32 \
                or any(value != reference for value in tokens):
            raise ValidationError("persistent result is not token-exact")
        host = session["host_placement"]
        if host.get("pid") != host_pid or host.get("run_rc") != 0 \
                or host.get("layer_start") != 8 or host.get("layer_end") != 48 \
                or host.get("status") != "SCHEDULED_PLACEMENT_OK" \
                or host.get("missing_buffer_compute_nodes") != 0 \
                or set(host.get("compute_by_buffer_type", {})) != {"CUDA0"}:
            raise ValidationError("host placement gate failed")
        phone = session["phone_session"]
        if phone.get("session_id") != index or phone.get("session_end") != end \
                or phone.get("worker_pid") != worker_pid \
                or phone.get("worker_boot_nonce") != data["resident_worker_nonce"] \
                or phone.get("device_boot_id") != data["boot_id"] \
                or phone.get("layer_start") != 0 or phone.get("layer_end") != 8 \
                or phone.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
                or phone.get("missing_buffer_compute_nodes") != 0 \
                or phone.get("reset_applied") is not (end == "DETACH"):
            raise ValidationError("phone session gate failed")
        steps_session = integer("steps_session", phone.get("steps_session"), 1)
        steps_total = integer("steps_total", phone.get("steps_total"), 1)
        if steps_total != previous_steps + steps_session:
            raise ValidationError("phone step totals are not contiguous")
        previous_steps = steps_total
    raw_results = [
        strict_object(line, "raw persistent result")
        for line in (HERE / "results/host.stdout.bin").read_bytes().splitlines()
    ]
    if raw_results != [session["result"] for session in sessions]:
        raise ValidationError("raw host results differ from the report")
    host_certs = prefixed(HERE / "results/host.stderr.bin", b"PLACEMENTCERT ")
    markers = prefixed(
        HERE / "results/host.stderr.bin", b"PERSISTENT_DRIVER_EXCHANGE_END ",
    )
    phone_certs = prefixed(HERE / "results/phone.stderr.bin", b"SESSIONCERT ")
    if host_certs != [session["host_placement"] for session in sessions] \
            or markers != [{"launch_id": 1}, {"launch_id": 2}] \
            or phone_certs != [session["phone_session"] for session in sessions]:
        raise ValidationError("raw certificate records differ from the report")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else REPORT
    if len(sys.argv) > 2:
        raise ValidationError("usage: validate_report.py [report.json]")
    data = strict_object(path.read_bytes(), "report")
    validate(data)
    print(
        "VALID_PERSISTENT_HOST_TAIL sessions=2 "
        f"host_pid={data['host_pid']} worker_pid={data['resident_worker_pid']} "
        "energy=UNKNOWN"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(2)
