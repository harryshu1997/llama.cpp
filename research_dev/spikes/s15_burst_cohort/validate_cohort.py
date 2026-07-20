#!/usr/bin/env python3
"""Independently validate the frozen BurstGPT B32 cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COHORT = HERE / "cohort.json"
INPUT_MANIFEST = HERE / "input_manifest.json"
TRACE = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.jsonl"
TRACE_MANIFEST = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.manifest.json"
GATE_REPORT = ROOT / "research_dev/spikes/s15_batch32_gate/results/gate_report.json"
BATCH = 32
N_GEN = 8
DEADLINE_BUDGET_US = 5_000_000


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def strict_value(raw: str) -> object:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        fail(f"invalid JSON constant {value}")

    try:
        return json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValidationError(str(exc)) from exc


def strict_object(path: Path) -> dict:
    try:
        value = strict_value(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(str(exc)) from exc
    if type(value) is not dict:
        fail("JSON artifact is not an object")
    return value


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def content_address(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def source_rows(path: Path) -> tuple[dict[str, tuple[dict, str]], list[tuple[dict, str]]]:
    rows = {}
    ordered = []
    previous = None
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            fail("empty source trace line")
        row = strict_value(line)
        if type(row) is not dict:
            fail("source trace row is not an object")
        event_id = row.get("event_id")
        t_us = row.get("t_us")
        if type(event_id) is not str or not event_id or type(t_us) is not int or t_us < 0:
            fail("invalid source row identity")
        key = (t_us, event_id)
        if previous is not None and key < previous:
            fail("source trace order changed")
        if event_id in rows:
            fail("duplicate source event")
        previous = key
        item = (row, hashlib.sha256((line + "\n").encode("ascii")).hexdigest())
        rows[event_id] = item
        ordered.append(item)
    if not ordered:
        fail("empty source trace")
    return rows, ordered


def eligible(row: dict) -> bool:
    source_fields = row.get("source_fields")
    return row.get("source") == "burstgpt-v2" \
        and row.get("provenance") == "real" \
        and row.get("service") == "api_generation" \
        and row.get("deadline_us") is None \
        and row.get("deadline_provenance") == "none" \
        and row.get("priority_class") is None \
        and row.get("priority_provenance") == "none" \
        and type(source_fields) is dict \
        and source_fields.get("burstgpt_failed") is False \
        and type(row.get("input_tokens")) is int and row["input_tokens"] > 0 \
        and type(row.get("output_tokens")) is int and row["output_tokens"] >= N_GEN


def select(ordered: list[tuple[dict, str]], width_us: int) -> tuple[list[tuple[dict, str]], dict]:
    if type(width_us) is not int or width_us <= 0:
        fail("invalid selection window")
    candidates = [item for item in ordered if eligible(item[0])]
    if len(candidates) < BATCH:
        fail("fewer than 32 eligible source requests")
    left = 0
    best = None
    for right, (row, _) in enumerate(candidates):
        while row["t_us"] - candidates[left][0]["t_us"] > width_us:
            left += 1
        first = candidates[left][0]
        key = (-(right - left + 1), first["t_us"], first["event_id"], row["event_id"])
        if best is None or key < best[0]:
            best = (key, left, right)
    if best is None:
        fail("no source selection window")
    _, left, right = best
    window = candidates[left:right + 1]
    if len(window) < BATCH:
        fail("densest source window cannot form B32")
    return window[:BATCH], {
        "eligible_request_count": len(candidates),
        "densest_window_count": len(window),
        "densest_window_start_us": window[0][0]["t_us"],
        "densest_window_end_us": window[-1][0]["t_us"],
        "selection_rule": "first-32-in-earliest-max-count-window-v1",
    }


def validate(
    cohort_path: Path = COHORT,
    input_path: Path = INPUT_MANIFEST,
    trace_path: Path = TRACE,
    trace_manifest_path: Path = TRACE_MANIFEST,
    gate_path: Path = GATE_REPORT,
) -> dict:
    cohort = strict_object(cohort_path)
    inputs = strict_object(input_path)
    if cohort.get("schema") != "s15-burst-b32-cohort-v1" \
            or cohort.get("scope") != "OBSERVED_ARRIVALS_SYNTHETIC_PAYLOAD_PRIORITY_AND_SLO_NO_EXECUTION_CLAIM" \
            or cohort.get("verdict") != "OBSERVED_BURST_CAN_FORM_OP15_B32_WITHIN_SYNTHETIC_5S_SLO" \
            or cohort.get("cohort_hash") != content_address(cohort, "cohort_hash"):
        fail("cohort identity failed")
    if inputs.get("schema") != "s15-burst-b32-input-manifest-v1" \
            or inputs.get("payload_provenance") != "synthetic-fixed-prompt" \
            or inputs.get("payload_scope") != "arrival-policy-test-only-not-source-request-replay" \
            or inputs.get("input_manifest_hash") != content_address(inputs, "input_manifest_hash"):
        fail("input manifest identity failed")

    source = cohort.get("source")
    if type(source) is not dict \
            or source.get("trace_path") != str(TRACE.relative_to(ROOT)) \
            or source.get("manifest_path") != str(TRACE_MANIFEST.relative_to(ROOT)) \
            or source.get("trace_sha256") != "sha256:" + digest(trace_path) \
            or source.get("manifest_sha256") != "sha256:" + digest(trace_manifest_path):
        fail("source binding failed")
    trace_manifest = strict_object(trace_manifest_path)
    if trace_manifest.get("schema_version") != 2 \
            or trace_manifest.get("provenance") != "real" \
            or trace_manifest.get("source") != "burstgpt-v2" \
            or trace_manifest.get("output_sha256") != "sha256:" + digest(trace_path) \
            or trace_manifest.get("time_scale_num") != 1 \
            or trace_manifest.get("time_scale_den") != 1 \
            or type(trace_manifest.get("window")) is not dict \
            or trace_manifest["window"].get("scenario") != "burst":
        fail("normalized trace manifest failed")
    target = cohort.get("execution_target")
    if type(target) is not dict \
            or target.get("input_manifest_hash") != inputs["input_manifest_hash"] \
            or target.get("profile_gate_path") != str(GATE_REPORT.relative_to(ROOT)) \
            or target.get("profile_gate_sha256") != "sha256:" + digest(gate_path) \
            or target.get("device") != "op15" or target.get("backend") != "HTP0" \
            or target.get("batch") != 32 or target.get("n_gen") != 8 \
            or target.get("layer_range") != [0, 8]:
        fail("execution target failed")
    gate = strict_object(gate_path)
    gate_model = gate.get("model")
    if gate.get("schema") != "s15-independent-b32-gate-v1" \
            or gate.get("verdict") != "B32_INDEPENDENT_PHONE_GATE_PASS" \
            or gate.get("certified") is not True \
            or gate.get("problems") != [] \
            or gate.get("batch") != BATCH or gate.get("n_gen") != N_GEN \
            or gate.get("energy_scope") != "UNKNOWN" \
            or gate.get("prompt") != inputs.get("prompt_text") \
            or type(gate_model) is not dict \
            or target.get("model_sha256") != "sha256:" + str(gate_model.get("sha256")):
        fail("physical gate semantics failed")
    profile = gate.get("profiles", {}).get("op15")
    samples = profile.get("request_wall_us_p50_by_process") if type(profile) is dict else None
    if type(samples) is not list or len(samples) != 7 \
            or any(type(value) is not int or value <= 0 for value in samples) \
            or profile.get("n_processes") != 7 \
            or target.get("profile_process_samples_us") != samples \
            or target.get("conservative_duration_us") != max(samples):
        fail("measured duration binding failed")

    prompt = inputs.get("prompt_text")
    prompt_digest = "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest() \
        if type(prompt) is str else None
    payloads = inputs.get("request_payloads")
    requests = cohort.get("requests")
    if inputs.get("prompt_encoding") != "utf-8" \
            or prompt_digest != inputs.get("prompt_sha256") \
            or type(requests) is not list or len(requests) != 32 \
            or type(payloads) is not list or len(payloads) != 32:
        fail("payload or request count failed")
    ids = [request.get("event_id") for request in requests if type(request) is dict]
    if len(ids) != 32 or len(set(ids)) != 32 \
            or payloads != [{"event_id": event_id, "payload_sha256": prompt_digest} for event_id in ids] \
            or any(request.get("payload_sha256") != prompt_digest for request in requests):
        fail("request payload binding failed")

    rows, ordered = source_rows(trace_path)
    if trace_manifest.get("output_row_count") != len(ordered):
        fail("normalized trace row count failed")
    for request in requests:
        event_id = request["event_id"]
        if event_id not in rows:
            fail("request is absent from source trace")
        row, row_digest = rows[event_id]
        expected = {
            "event_id": event_id,
            "observed_t_us": row["t_us"],
            "observed_input_tokens": row["input_tokens"],
            "observed_output_tokens": row["output_tokens"],
            "source_row_id": row["source_row_id"],
            "source_row_sha256": "sha256:" + row_digest,
            "payload_sha256": prompt_digest,
        }
        if request != expected or not eligible(row):
            fail("request source replay failed")

    p95_us = target["conservative_duration_us"]
    budget_us = DEADLINE_BUDGET_US
    selected, selection = select(ordered, budget_us - p95_us)
    expected_ids = [row["event_id"] for row, _ in selected]
    declared_selection = cohort.get("selection")
    if ids != expected_ids or type(declared_selection) is not dict \
            or declared_selection.get("eligible_request_count") != selection["eligible_request_count"] \
            or declared_selection.get("densest_window_count") != selection["densest_window_count"] \
            or declared_selection.get("densest_window_start_us") != selection["densest_window_start_us"] \
            or declared_selection.get("densest_window_end_us") != selection["densest_window_end_us"] \
            or declared_selection.get("selection_rule") != selection["selection_rule"] \
            or declared_selection.get("eligibility") != "real api_generation; nonfailure; input_tokens>0; output_tokens>=8; null source priority/deadline" \
            or declared_selection.get("window_width_us") != budget_us - p95_us \
            or declared_selection.get("selected_count") != 32:
        fail("deterministic selection replay failed")
    sidecar = cohort.get("synthetic_sidecar")
    if sidecar != {
        "provenance": "s15-synthetic-sidecar",
        "priority_class": 1,
        "relative_deadline_us": budget_us,
    }:
        fail("synthetic sidecar failed")

    first_us = requests[0]["observed_t_us"]
    last_us = requests[-1]["observed_t_us"]
    finish_us = last_us + p95_us
    deadline_us = first_us + budget_us
    expected_schedule = {
        "first_arrival_us": first_us,
        "last_arrival_us": last_us,
        "formation_delay_us": last_us - first_us,
        "latest_safe_launch_us": deadline_us - p95_us,
        "planned_launch_us": last_us,
        "predicted_finish_us": finish_us,
        "earliest_deadline_us": deadline_us,
        "predicted_slack_us": deadline_us - finish_us,
    }
    if cohort.get("admission_schedule") != expected_schedule or finish_us > deadline_us:
        fail("SLO schedule failed")
    return cohort


def main() -> int:
    cohort = validate()
    schedule = cohort["admission_schedule"]
    print(
        "VALID_BURST_B32_COHORT "
        f"formation_us={schedule['formation_delay_us']} "
        f"predicted_slack_us={schedule['predicted_slack_us']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
