#!/usr/bin/env python3
"""Independent validator for the arrival-faithful physical B32 result."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "physical_results"
REPORT = RESULTS / "result.json"
COHORT_DIR = HERE.parent / "s15_burst_cohort"
RUNTIME = HERE.parent / "s15_runtime_dispatch"
for path in (HERE, COHORT_DIR, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import validate_cohort  # noqa: E402
import validate_recertification as recert  # noqa: E402
from route_fixtures import op15_post_load_b32_snapshot  # noqa: E402


EXPECTED_COHORT = "sha256:85e0614d31a6f35ad7d09c3b35b2ab4fa9ef2916bc4a1dcfc32d6f4c70a51ec4"
EXPECTED_INPUT = "sha256:ea1f2d47b8c6dec6096c8b95b6073181ecba6838d97741ca92ec2ceca6937858"
EXPECTED_WORKER = "sha256:2494f191515ca576cacd8c411de79fa8985f719cba2a11cd6371aa6e1024be9f"
EXPECTED_PROFILE = "sha256:6fea9922ca8a47779d07d235d0512b25661b176c643e82eb612811519b88d297"


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
                fail(f"duplicate key {key!r} in {label}")
            result[key] = value
        return result

    def reject_constant(value):
        fail(f"invalid constant {value!r} in {label}")

    try:
        return json.loads(payload, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid JSON in {label}: {exc}") from exc


def strict_object(path: Path) -> dict:
    value = strict_bytes(path.read_bytes(), str(path))
    if type(value) is not dict:
        fail(f"{path} is not an object")
    return value


def prefixed(payload: bytes, prefix: bytes, label: str) -> list[dict]:
    values = []
    for line in payload.splitlines():
        if line.startswith(prefix):
            value = strict_bytes(line[len(prefix):], label)
            if type(value) is not dict:
                fail(f"non-object {label} record")
            values.append(value)
    return values


def expected_decisions(cohort: dict, request_ids: list[str], duration_us: int) -> list[dict]:
    requests = cohort["requests"]
    earliest_deadline = min(value["observed_t_us"] + 5_000_000 for value in requests)
    next_wake = earliest_deadline - duration_us
    values = [
        {
            "after_request_id": request["event_id"],
            "at_us": request["observed_t_us"],
            "action": "WAIT",
            "reason": "bounded_batch_wait",
            "next_wake_us": next_wake,
        }
        for request in requests[:-1]
    ]
    values.append({
        "after_request_id": requests[-1]["event_id"],
        "at_us": requests[-1]["observed_t_us"],
        "action": "LAUNCH",
        "reason": "target_batch_ready",
        "batch_size": 32,
        "predicted_duration_us": duration_us,
        "request_ids": request_ids,
    })
    return values


def validate_transport_timing(report: dict, cohort: dict, paid: dict, transport: dict) -> None:
    if transport.get("state") != "reply" or type(transport.get("elapsed_us")) is not int \
            or transport["elapsed_us"] < paid["elapsed_us"] \
            or transport["elapsed_us"] > 4_000_000 \
            or report.get("transport_elapsed_us") != transport["elapsed_us"]:
        fail("full response timing failed")
    launch_us = cohort["admission_schedule"]["planned_launch_us"]
    deadline_us = cohort["admission_schedule"]["earliest_deadline_us"]
    finish_us = launch_us + transport["elapsed_us"]
    if report.get("logical_launch_us") != launch_us \
            or report.get("earliest_deadline_us") != deadline_us \
            or report.get("logical_finish_us") != finish_us \
            or report.get("deadline_margin_us") != deadline_us - finish_us \
            or finish_us > deadline_us:
        fail("logical replay deadline failed")


def validate() -> dict:
    report = strict_object(REPORT)
    cohort = validate_cohort.validate()
    inputs = strict_object(COHORT_DIR / "input_manifest.json")
    prompt = inputs.get("prompt_text")
    request_ids = [value["event_id"] for value in cohort["requests"]]
    snapshot = op15_post_load_b32_snapshot()
    b32 = next(value for value in snapshot.config.points if value.batch_size == 32)
    profile_evidence = recert.validate()
    if type(prompt) is not str or not prompt \
            or digest(COHORT_DIR / "cohort.json") != EXPECTED_COHORT \
            or digest(COHORT_DIR / "input_manifest.json") != EXPECTED_INPUT \
            or snapshot.profile_id != EXPECTED_PROFILE or b32.duration_us != 3_968_367:
        fail("frozen workload or route profile failed")
    if report.get("schema") != "s15-arrival-faithful-physical-b32-v1" \
            or report.get("verdict") != "ARRIVAL_FAITHFUL_PHYSICAL_B32_PASS" \
            or report.get("scope") != "REAL_OP15_A6000_OBSERVED_ARRIVAL_REPLAY_SYNTHETIC_PAYLOAD_PRIORITY_SLO_ENERGY_UNKNOWN" \
            or report.get("energy_scope") != "UNKNOWN" \
            or report.get("cohort_sha256") != EXPECTED_COHORT \
            or report.get("input_manifest_sha256") != EXPECTED_INPUT \
            or report.get("profile_id") != EXPECTED_PROFILE \
            or report.get("profile_duration_us") != b32.duration_us \
            or report.get("prompt_visible_during_preflight") is not False \
            or report.get("arrival_replay_slo_claim") is not True \
            or report.get("problems") != [] \
            or report.get("request_ids") != request_ids:
        fail("result scope or identity failed")

    actual_hashes = {
        str(path.relative_to(RESULTS)): digest(path)
        for path in sorted(RESULTS.rglob("*")) if path.is_file() and path != REPORT
    }
    if report.get("artifact_hashes_before_report") != actual_hashes:
        fail("physical artifact hash set failed")
    if report.get("decisions") != expected_decisions(cohort, request_ids, b32.duration_us):
        fail("scheduler decisions do not match the frozen replay")
    if report.get("terminal_states") != {value: "completed_phone" for value in request_ids}:
        fail("terminal ownership failed")

    raw = RESULTS / "raw/launch-1"
    preflight = strict_object(raw / "preflight.json")
    paid = strict_object(raw / "paid_window.json")
    request = strict_object(raw / "request.json")
    transport_request = strict_object(RESULTS / "transport/request.json")
    transport = strict_object(RESULTS / "transport/transport.json")
    session_payload = (RESULTS / "transport/stdout.bin").read_bytes()
    session = strict_bytes(session_payload, "session")
    if type(session) is not dict or session_payload != (
            json.dumps(session, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii"):
        fail("session is not one canonical object")
    if request != transport_request or (raw / "request.json").read_bytes() \
            != (RESULTS / "transport/request.json").read_bytes():
        fail("typed request bytes differ across the boundary")
    if report.get("paid_window") != paid:
        fail("reported paid window differs from raw evidence")
    expected_request_fields = {
        "profile_id": EXPECTED_PROFILE,
        "route_epoch": 13,
        "timeout_us": 4_000_000,
        "cohort_sha256": EXPECTED_COHORT,
        "input_manifest_sha256": EXPECTED_INPUT,
        "request_ids": request_ids,
        "worker_binary_sha256": EXPECTED_WORKER,
        "layer_range": [0, 8],
    }
    if any(request.get(key) != value for key, value in expected_request_fields.items()):
        fail("typed request identity failed")
    if session.get("outcome") != "completed" or session.get("profile_id") != EXPECTED_PROFILE \
            or session.get("route_epoch") != 13 or session.get("request_ids") != request_ids \
            or session.get("device_boot_id") != request.get("device_boot_id"):
        fail("session identity failed")
    boundaries = session.get("boundaries")
    if type(boundaries) is not list or [value.get("request_id") for value in boundaries] != request_ids \
            or any(set(value) != {
                "request_id", "identity_ok", "epoch_ok", "correctness_ok", "d2h_complete"}
                or any(value[key] is not True for key in
                       ("identity_ok", "epoch_ok", "correctness_ok", "d2h_complete"))
                for value in boundaries):
        fail("completion boundary failed")

    if preflight.get("schema") != "s15-arrival-preflight-v1" \
            or preflight.get("scope") != "MODELS_RESIDENT_HOST_PROMPT_UNSUBMITTED" \
            or preflight.get("prompt_in_host_argv") is not False \
            or preflight.get("profile_id") != EXPECTED_PROFILE \
            or preflight.get("profile_evidence_id") != profile_evidence["profile_id"] \
            or preflight.get("host_artifacts") != profile_evidence["host_artifacts"] \
            or preflight.get("worker_binary_sha256") != EXPECTED_WORKER:
        fail("preflight identity failed")
    host_command = preflight.get("host_command")
    if type(host_command) is not list or "--prompt-after-load" not in host_command \
            or "-p" in host_command or prompt in host_command:
        fail("host command exposed the prompt before launch")
    pre_prompt = (raw / "host.pre_prompt.stderr.bin").read_bytes()
    host_stderr = (raw / "host.stderr.bin").read_bytes()
    if not host_stderr.startswith(pre_prompt) \
            or pre_prompt.count(b"DRIVER_INPUT_READY ") != 1 \
            or b"DRIVER_INPUT_ACCEPTED " in pre_prompt \
            or prompt.encode("utf-8") in pre_prompt:
        fail("pre-prompt host bytes failed")
    ready_offset = host_stderr.find(b"DRIVER_INPUT_READY ")
    accepted_offset = host_stderr.find(b"DRIVER_INPUT_ACCEPTED ")
    driver_offset = host_stderr.find(b"DRIVER_READY ")
    done_offset = host_stderr.find(b"DRIVER_DONE ")
    if not (0 <= ready_offset < accepted_offset < driver_offset < done_offset) \
            or len(prefixed(host_stderr, b"DRIVER_INPUT_READY ", "input ready")) != 1 \
            or prefixed(host_stderr, b"DRIVER_INPUT_ACCEPTED ", "input accepted") != [{
                "schema": "layersplit-driver-input-v1",
                "prompt_bytes": len(prompt.encode("utf-8")),
            }]:
        fail("prompt acceptance ordering failed")

    if paid.get("schema") != "s15-arrival-paid-window-v1" \
            or paid.get("prompt_visible_during_preflight") is not False:
        fail("paid-window scope failed")
    times = [
        paid.get("start_monotonic_ns"), paid.get("prompt_submitted_ns"),
        paid.get("end_monotonic_ns"),
    ]
    if any(type(value) is not int for value in times) or times != sorted(times) \
            or paid.get("end_monotonic_ns") != max(
                paid.get("host_exit_observed_ns", -1), paid.get("phone_exit_observed_ns", -1)) \
            or paid.get("elapsed_us") != (times[-1] - times[0]) // 1000 \
            or paid.get("elapsed_us") > 4_000_000:
        fail("paid prompt-to-completion timeline failed")
    observed_thermal = recert.thermal_samples((raw / "thermal_stream.log").read_bytes())
    if not recert.thermal_ok(paid.get("thermal_end"), 85_000) \
            or tuple(sorted(paid["thermal_end"]["sensors_millic"].items())) not in observed_thermal:
        fail("paid thermal evidence failed")
    validate_transport_timing(report, cohort, paid, transport)

    reference = profile_evidence["reference_tokens"]
    rows = prefixed(host_stderr, b"ROUTEJSON ", "route")
    certs = prefixed(
        (raw / "phone.stdout.bin").read_bytes() + (raw / "phone.stderr.bin").read_bytes(),
        b"PLACEMENTCERT ", "placement",
    )
    if len(rows) != 32 or sorted(value.get("stream_index") for value in rows) != list(range(32)) \
            or any(value.get("status") != "ok" or value.get("batch_size") != 32
                   or value.get("generated_tokens") != 8 or value.get("token_ids") != reference
                   for value in rows) or len(certs) != 1:
        fail("physical token evidence failed")
    placement = recert.validate_placement(certs[0])
    if session.get("placement") != placement \
            or paid.get("route_wall_us_max") != max(value["request_wall_us"] for value in rows):
        fail("physical placement or timing projection failed")
    return report


def main() -> int:
    report = validate()
    print(
        "VALID_ARRIVAL_FAITHFUL_B32 "
        f"transport_elapsed_us={report['transport_elapsed_us']} "
        f"deadline_margin_us={report['deadline_margin_us']} energy=UNKNOWN"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
