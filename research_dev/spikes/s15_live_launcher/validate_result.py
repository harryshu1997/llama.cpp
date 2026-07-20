#!/usr/bin/env python3
"""Independent validation of the coordinator-triggered physical B32 result."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = HERE / "results"
REPORT = RESULTS / "result.json"
COHORT_DIR = HERE.parent / "s15_burst_cohort"
sys.path.insert(0, str(COHORT_DIR))

import validate_cohort  # noqa: E402


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
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value):
        fail(f"invalid JSON constant {value!r} in {label}")

    try:
        return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid {label}: {exc}") from exc


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


def thermal_ok(value: object, limit: int) -> bool:
    if type(value) is not dict or value.get("valid") is not True:
        return False
    sensors = value.get("sensors_millic")
    maximum = value.get("max_millic")
    return type(sensors) is dict and bool(sensors) \
        and all(type(item) is int for item in sensors.values()) \
        and type(maximum) is int and maximum == max(sensors.values()) and maximum <= limit


def validate_replay_timing(cohort: dict, decisions: object, transport_elapsed_us: object) -> int:
    requests = cohort.get("requests")
    schedule = cohort.get("admission_schedule")
    sidecar = cohort.get("synthetic_sidecar")
    if type(requests) is not list or len(requests) != 32 \
            or type(schedule) is not dict or type(sidecar) is not dict:
        fail("cohort timing inputs failed")
    if type(transport_elapsed_us) is not int or transport_elapsed_us <= 0:
        fail("transport elapsed time failed")
    relative_deadline_us = sidecar.get("relative_deadline_us")
    latest_safe_launch_us = schedule.get("latest_safe_launch_us")
    if type(relative_deadline_us) is not int or relative_deadline_us <= 0 \
            or type(latest_safe_launch_us) is not int:
        fail("cohort SLO inputs failed")

    request_ids = [value.get("event_id") for value in requests]
    arrival_times = [value.get("observed_t_us") for value in requests]
    if any(type(value) is not str or not value for value in request_ids) \
            or len(set(request_ids)) != len(request_ids) \
            or any(type(value) is not int or value < 0 for value in arrival_times):
        fail("cohort request timing identity failed")
    expected = [
        {
            "after_request_id": request_id,
            "at_us": arrival_us,
            "action": "WAIT",
            "reason": "bounded_batch_wait",
            "next_wake_us": latest_safe_launch_us,
        }
        for request_id, arrival_us in zip(request_ids[:-1], arrival_times[:-1])
    ]
    expected.append({
        "after_request_id": request_ids[-1],
        "at_us": arrival_times[-1],
        "action": "LAUNCH",
        "reason": "target_batch_ready",
        "batch_size": 32,
        "request_ids": request_ids,
    })
    if decisions != expected:
        fail("coordinator decisions do not match the frozen arrival replay")

    launch_us = arrival_times[-1]
    if schedule.get("planned_launch_us") != launch_us:
        fail("cohort planned launch does not match the replay launch")
    earliest_deadline_us = min(
        arrival_us + relative_deadline_us for arrival_us in arrival_times
    )
    if schedule.get("earliest_deadline_us") != earliest_deadline_us:
        fail("cohort earliest deadline is not derived from the requests")
    completion_us = launch_us + transport_elapsed_us
    if completion_us > earliest_deadline_us:
        fail("full transport reply misses a cohort request deadline")
    return earliest_deadline_us - completion_us


def validate() -> tuple[dict, int, int]:
    report = strict_object(REPORT)
    cohort = validate_cohort.validate()
    input_path = COHORT_DIR / "input_manifest.json"
    cohort_path = COHORT_DIR / "cohort.json"
    if digest(cohort_path) != "sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4" \
            or digest(input_path) != "sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858":
        fail("frozen cohort/input file digest mismatch")
    request_ids = [value["event_id"] for value in cohort["requests"]]
    if report.get("schema") != "s15-coordinator-physical-b32-v1" \
            or report.get("verdict") != "COORDINATOR_TRIGGERED_PHYSICAL_MECHANICS_PASS_ARRIVAL_FAITHFUL_SLO_BLOCKED" \
            or report.get("energy_scope") != "UNKNOWN" \
            or report.get("arrival_faithful_slo_claim") is not False \
            or report.get("prompt_visible_during_preflight") is not True \
            or report.get("problems") != [] \
            or report.get("request_ids") != request_ids:
        fail("result scope or workload identity failed")

    declared_hashes = report.get("artifact_hashes_before_report")
    actual_paths = [path for path in sorted(RESULTS.rglob("*")) if path.is_file() and path != REPORT]
    actual_hashes = {str(path.relative_to(RESULTS)): digest(path) for path in actual_paths}
    if declared_hashes != actual_hashes:
        fail("result artifact hash set mismatch")

    decisions = report.get("decisions")
    if set(report.get("terminal_states", {}).keys()) != set(request_ids) \
            or set(report["terminal_states"].values()) != {"completed_phone"}:
        fail("terminal ownership failed")

    raw = RESULTS / "raw/launch-1"
    preflight = strict_object(raw / "preflight.json")
    paid = strict_object(raw / "paid_window.json")
    transport = strict_object(RESULTS / "transport/transport.json")
    request = strict_object(RESULTS / "transport/request.json")
    session_payload = (RESULTS / "transport/stdout.bin").read_bytes()
    session = strict_bytes(session_payload, "session stdout")
    if type(session) is not dict or session_payload != (
            json.dumps(session, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"):
        fail("launcher stdout is not exactly one canonical session record")
    if (raw / "request.json").read_bytes() != (RESULTS / "transport/request.json").read_bytes():
        fail("launcher and transport request bytes differ")
    if report.get("paid_window") != paid:
        fail("reported paid window differs from the hashed raw record")

    expected_cohort = "sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4"
    expected_input = "sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858"
    for value in (request, session, preflight):
        if value.get("cohort_sha256") != expected_cohort \
                or value.get("input_manifest_sha256") != expected_input:
            fail("cohort/input digest is not carried end-to-end")
    if request.get("request_ids") != request_ids or session.get("request_ids") != request_ids \
            or request.get("route_epoch") != 12 or request.get("residency_epoch") != 1 \
            or request.get("lease_epoch") != 1 or request.get("device_boot_epoch") != 1 \
            or request.get("registry_generation") != 1:
        fail("request ownership or epoch binding failed")
    if session.get("outcome") != "completed" or session.get("session_id") != 1 \
            or session.get("device_boot_id") != preflight.get("device_boot_id") \
            or session.get("worker_binary_sha256") != preflight.get("worker_binary_sha256"):
        fail("session identity failed")
    boundaries = session.get("boundaries")
    if type(boundaries) is not list or [value.get("request_id") for value in boundaries] != request_ids \
            or any(set(value) != {"request_id", "identity_ok", "epoch_ok", "correctness_ok", "d2h_complete"}
                   or any(value[key] is not True for key in
                          ("identity_ok", "epoch_ok", "correctness_ok", "d2h_complete"))
                   for value in boundaries):
        fail("completion boundaries failed")

    reference = strict_object(raw / "reference_tokens.json").get("token_ids")
    reference_rows = prefixed(raw / "reference.stderr.bin", b"ROUTEJSON ")
    route_rows = prefixed(raw / "host.stderr.bin", b"ROUTEJSON ")
    done = prefixed(raw / "host.stderr.bin", b"DRIVER_DONE ")
    certs = prefixed(raw / "phone.stdout.bin", b"PLACEMENTCERT ") \
        + prefixed(raw / "phone.stderr.bin", b"PLACEMENTCERT ")
    if type(reference) is not list or len(reference) != 8 \
            or len(reference_rows) != 32 or len(route_rows) != 32 \
            or len(done) != 1 or len(certs) != 1:
        fail("raw completion evidence count failed")
    if sorted(row.get("stream_index") for row in reference_rows) != list(range(32)) \
            or sorted(row.get("stream_index") for row in route_rows) != list(range(32)) \
            or any(row.get("token_ids") != reference for row in reference_rows + route_rows):
        fail("raw token correctness failed")
    cert = certs[0]
    mapping = cert.get("compute_by_op_and_buffer")
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" or cert.get("layer_start") != 0 \
            or cert.get("layer_end") != 8 or cert.get("missing_buffer_compute_nodes") != 0 \
            or type(mapping) is not dict:
        fail("raw placement certificate failed")
    htp = 0
    for op, buffers in mapping.items():
        if type(buffers) is not dict:
            fail("raw placement map failed")
        for backend, count in buffers.items():
            if type(count) is not int or count <= 0:
                fail("raw placement count failed")
            if backend == "HTP0":
                htp += count
            elif backend != "CPU" or op != "GET_ROWS":
                fail("raw placement contains undeclared fallback")
    if htp == 0 or session.get("placement", {}).get("compute_by_op_and_buffer") != mapping:
        fail("session placement is not derived from raw placement")

    if preflight.get("scope") != "SETUP_OUTSIDE_PAID_WINDOW_PROMPT_VISIBLE_BEFORE_ARRIVAL" \
            or not thermal_ok(preflight.get("thermal_start"), 60_000) \
            or not thermal_ok(paid.get("thermal_end"), 85_000):
        fail("preflight scope or thermal gate failed")
    thermal_paths = preflight.get("thermal_paths")
    if type(thermal_paths) is not dict or set(thermal_paths) != set(
            preflight["thermal_start"]["sensors_millic"]) \
            or any(type(path) is not str or not path.startswith("/sys/class/thermal/thermal_zone")
                   for path in thermal_paths.values()) \
            or paid["thermal_end"].get("sample_age_us", 1_000_001) > 1_000_000:
        fail("continuous thermal path/sample binding failed")
    if type(paid.get("elapsed_us")) is not int or paid["elapsed_us"] <= 0 \
            or paid["elapsed_us"] > 5_000_000 \
            or type(paid.get("route_wall_us_max")) is not int \
            or paid["route_wall_us_max"] > paid["elapsed_us"]:
        fail("paid window failed")
    if transport.get("state") != "reply" or type(transport.get("elapsed_us")) is not int \
            or transport["elapsed_us"] < paid["elapsed_us"] \
            or transport["elapsed_us"] > 5_000_000:
        fail("transport paid-window binding failed")
    replay_deadline_margin_us = validate_replay_timing(
        cohort, decisions, transport["elapsed_us"],
    )

    artifact_manifest = ROOT / "research_dev/spikes/s14_mixed_streaming_scheduler/persistence_v2/artifacts/SHA256SUMS.txt"
    if preflight.get("artifact_manifest_sha256") != digest(artifact_manifest):
        fail("frozen artifact manifest binding failed")
    deployed = preflight.get("deployed_sha256")
    if type(deployed) is not dict or deployed.get("llama-layersplit") \
            != "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f":
        fail("deployed worker binding failed")
    return report, replay_deadline_margin_us, transport["elapsed_us"]


def main() -> int:
    report, replay_deadline_margin_us, transport_elapsed_us = validate()
    print(
        "VALID_S15_PHYSICAL_B32 "
        f"paid_elapsed_us={report['paid_window']['elapsed_us']} "
        f"transport_elapsed_us={transport_elapsed_us} "
        f"deadline_margin_us={replay_deadline_margin_us} "
        "energy=UNKNOWN arrival_faithful_slo=false"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
