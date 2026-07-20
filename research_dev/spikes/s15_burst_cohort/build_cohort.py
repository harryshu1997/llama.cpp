#!/usr/bin/env python3
"""Freeze an observed-arrival BurstGPT cohort for the physical B32 gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRACE = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.jsonl"
TRACE_MANIFEST = ROOT / "scratchpad/s8_gate_a/run-a/burstgpt-burst.manifest.json"
GATE_REPORT = ROOT / "research_dev/spikes/s15_batch32_gate/results/gate_report.json"
INPUT_MANIFEST = HERE / "input_manifest.json"
COHORT = HERE / "cohort.json"

BATCH = 32
N_GEN = 8
DEADLINE_BUDGET_US = 5_000_000
PRIORITY_CLASS = 1
PROMPT = "Explain batching."
DEVICE = "op15"
LAYER_RANGE = [0, 8]


class BuildError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(raw: str) -> dict:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BuildError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise BuildError(f"invalid JSON constant {value}")

    value = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=reject_constant)
    if type(value) is not dict:
        raise BuildError("JSON value is not an object")
    return value


def canonical(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def content_address(value: dict, field: str) -> str:
    payload = dict(value)
    payload.pop(field, None)
    return "sha256:" + sha256_bytes(canonical(payload))


def load_rows(path: Path = TRACE) -> list[tuple[dict, str]]:
    rows = []
    seen = set()
    previous = None
    for line in path.read_text(encoding="ascii").splitlines():
        if not line:
            raise BuildError("empty trace line")
        row = strict_json(line)
        event_id = row.get("event_id")
        t_us = row.get("t_us")
        if type(event_id) is not str or not event_id or type(t_us) is not int or t_us < 0:
            raise BuildError("invalid trace identity")
        key = (t_us, event_id)
        if previous is not None and key < previous:
            raise BuildError("trace is not canonically ordered")
        if event_id in seen:
            raise BuildError(f"duplicate event_id {event_id!r}")
        previous = key
        seen.add(event_id)
        rows.append((row, sha256_bytes((line + "\n").encode("ascii"))))
    if not rows:
        raise BuildError("empty trace")
    return rows


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


def select(rows: list[tuple[dict, str]], width_us: int) -> tuple[list[tuple[dict, str]], dict]:
    candidates = [item for item in rows if eligible(item[0])]
    if len(candidates) < BATCH:
        raise BuildError("fewer than 32 eligible requests")
    left = 0
    best = None
    for right, (row, _) in enumerate(candidates):
        while row["t_us"] - candidates[left][0]["t_us"] > width_us:
            left += 1
        first = candidates[left][0]
        key = (-(right - left + 1), first["t_us"], first["event_id"], row["event_id"])
        if best is None or key < best[0]:
            best = (key, left, right)
    assert best is not None
    _, left, right = best
    window = candidates[left:right + 1]
    if len(window) < BATCH:
        raise BuildError("densest eligible window cannot form B32")
    return window[:BATCH], {
        "eligible_request_count": len(candidates),
        "densest_window_count": len(window),
        "densest_window_start_us": window[0][0]["t_us"],
        "densest_window_end_us": window[-1][0]["t_us"],
        "selection_rule": "first-32-in-earliest-max-count-window-v1",
    }


def load_profile() -> tuple[int, list[int], str, str]:
    report = strict_json(GATE_REPORT.read_text(encoding="ascii"))
    if report.get("certified") is not True or report.get("batch") != BATCH \
            or report.get("n_gen") != N_GEN or report.get("energy_scope") != "UNKNOWN":
        raise BuildError("physical gate is not an eligible B32 latency source")
    profiles = report.get("profiles")
    profile = profiles.get(DEVICE) if type(profiles) is dict else None
    samples = profile.get("request_wall_us_p50_by_process") if type(profile) is dict else None
    if type(samples) is not list or len(samples) != 7 \
            or any(type(value) is not int or value <= 0 for value in samples):
        raise BuildError("physical gate lacks seven integer process samples")
    model = report.get("model")
    if type(model) is not dict or type(model.get("sha256")) is not str:
        raise BuildError("physical gate lacks a model digest")
    return max(samples), samples, sha256_file(GATE_REPORT), model["sha256"]


def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def build() -> tuple[dict, dict]:
    p95_us, samples, gate_digest, model_digest = load_profile()
    queue_budget_us = DEADLINE_BUDGET_US - p95_us
    if queue_budget_us <= 0:
        raise BuildError("B32 profile cannot meet the declared SLO")
    selected, selection = select(load_rows(), queue_budget_us)
    first_us = selected[0][0]["t_us"]
    last_us = selected[-1][0]["t_us"]
    launch_us = last_us
    finish_us = launch_us + p95_us
    deadline_us = first_us + DEADLINE_BUDGET_US
    if finish_us > deadline_us:
        raise BuildError("selected B32 cohort misses the declared SLO")

    prompt_bytes = PROMPT.encode("utf-8")
    prompt_digest = "sha256:" + sha256_bytes(prompt_bytes)
    requests = []
    for row, row_digest in selected:
        requests.append({
            "event_id": row["event_id"],
            "observed_t_us": row["t_us"],
            "observed_input_tokens": row["input_tokens"],
            "observed_output_tokens": row["output_tokens"],
            "source_row_id": row["source_row_id"],
            "source_row_sha256": "sha256:" + row_digest,
            "payload_sha256": prompt_digest,
        })
    input_manifest = {
        "schema": "s15-burst-b32-input-manifest-v1",
        "payload_provenance": "synthetic-fixed-prompt",
        "payload_scope": "arrival-policy-test-only-not-source-request-replay",
        "prompt_encoding": "utf-8",
        "prompt_text": PROMPT,
        "prompt_sha256": prompt_digest,
        "request_payloads": [
            {"event_id": request["event_id"], "payload_sha256": prompt_digest}
            for request in requests
        ],
    }
    input_manifest["input_manifest_hash"] = content_address(input_manifest, "input_manifest_hash")

    cohort = {
        "schema": "s15-burst-b32-cohort-v1",
        "scope": "OBSERVED_ARRIVALS_SYNTHETIC_PAYLOAD_PRIORITY_AND_SLO_NO_EXECUTION_CLAIM",
        "source": {
            "trace_path": str(TRACE.relative_to(ROOT)),
            "trace_sha256": "sha256:" + sha256_file(TRACE),
            "manifest_path": str(TRACE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": "sha256:" + sha256_file(TRACE_MANIFEST),
        },
        "selection": {
            **selection,
            "eligibility": "real api_generation; nonfailure; input_tokens>0; output_tokens>=8; null source priority/deadline",
            "window_width_us": queue_budget_us,
            "selected_count": BATCH,
        },
        "synthetic_sidecar": {
            "provenance": "s15-synthetic-sidecar",
            "priority_class": PRIORITY_CLASS,
            "relative_deadline_us": DEADLINE_BUDGET_US,
        },
        "execution_target": {
            "device": DEVICE,
            "backend": "HTP0",
            "batch": BATCH,
            "n_gen": N_GEN,
            "layer_range": LAYER_RANGE,
            "model_sha256": "sha256:" + model_digest,
            "input_manifest_hash": input_manifest["input_manifest_hash"],
            "profile_gate_path": str(GATE_REPORT.relative_to(ROOT)),
            "profile_gate_sha256": "sha256:" + gate_digest,
            "profile_process_samples_us": samples,
            "conservative_duration_us": p95_us,
        },
        "admission_schedule": {
            "first_arrival_us": first_us,
            "last_arrival_us": last_us,
            "formation_delay_us": last_us - first_us,
            "latest_safe_launch_us": deadline_us - p95_us,
            "planned_launch_us": launch_us,
            "predicted_finish_us": finish_us,
            "earliest_deadline_us": deadline_us,
            "predicted_slack_us": deadline_us - finish_us,
        },
        "requests": requests,
        "verdict": "OBSERVED_BURST_CAN_FORM_OP15_B32_WITHIN_SYNTHETIC_5S_SLO",
    }
    cohort["cohort_hash"] = content_address(cohort, "cohort_hash")
    return cohort, input_manifest


def main() -> int:
    cohort, input_manifest = build()
    write_atomic(INPUT_MANIFEST, canonical(input_manifest))
    write_atomic(COHORT, canonical(cohort))
    print(cohort["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
