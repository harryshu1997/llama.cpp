#!/usr/bin/env python3
"""Quantify phone-assist coverage and waiting from one campaign RESULT.json.

Execution windows are physical: ``execution_receipt.started_us`` to
``finished_us``. Coverage is a ladder per request, each rung checked from
run evidence: model weights resident in a READY layout during the window;
an executable helper (the command's helper envelope names a layout that was
READY at start with the same artifact, geometry, and session set); an
attachment (only provable from exported ``request_helper_events``; without
them a request with no envelope is ``unattached``, never "attached");
a probed non-zero fraction; and physical phone calls. Requests below
``--short-output-tokens`` are classed as too short to amortize a probe, and
requests whose model was resident for less than ``--minimum-opportunity``
of their window are ``insufficient_opportunity``. The idle metrics describe
pending or executing work whose model had no resident shard; physical phone
activity itself is not measured here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def _sec(value_us: int | None) -> float | None:
    return None if value_us is None else round(value_us / 1e6, 1)


def _overlap(start: int, end: int, intervals) -> int:
    covered = 0
    for lo, hi in intervals:
        a = max(start, lo)
        b = min(end, hi if hi is not None else end)
        if b > a:
            covered += b - a
    return covered


def _ready_layouts(
    events: list[dict], role_by_artifact: dict
) -> tuple[list[list], dict[int, dict]]:
    # READY layouts as (start, end, generation, {session: role}).
    ready: list[list] = []
    proposals: dict[int, dict] = {}
    for event in events:
        kind = event.get("kind")
        t = int(event["observed_at_us"])
        generation = event.get("generation") or event.get("failed_generation")
        if kind == "PROPOSED":
            proposals[generation] = {"proposed_at_us": t}
        elif kind == "PREPARING":
            proposals.setdefault(generation, {})["preparing_at_us"] = t
        elif kind == "READY":
            proposals.setdefault(generation, {})["ready_at_us"] = t
            sessions = {
                shard["session_id"]: role_by_artifact.get(
                    shard["artifact_sha256"], "?"
                )
                for shard in event["layout"]["shards"]
            }
            if ready:
                ready[-1][1] = t
            ready.append([t, None, generation, sessions, event["layout"]])
        elif kind == "TRANSITION_FAILED":
            row = proposals.setdefault(generation, {})
            row["failed_at_us"] = t
            row["failure_reason"] = (event.get("reason") or "")[:120]
    return ready, proposals


def _resident_intervals(ready: list[list], role: str):
    return [
        (s, e) for s, e, _g, sessions, _layout in ready
        if role in sessions.values()
    ]


def _ready_layout_at(ready: list[list], t: int):
    for s, e, g, _sessions, layout in ready:
        if s <= t and (e is None or t < e):
            return g, layout
    return None, None


def _envelope_executable(
    ready: list[list], envelope, artifact: str, t: int
) -> bool:
    """Executable means the envelope names the READY layout exactly."""
    generation, layout = _ready_layout_at(ready, t)
    if not isinstance(envelope, dict) or layout is None:
        return False
    if envelope.get("phone_layout_generation") != generation:
        return False
    if envelope.get("phone_layout_geometry_sha256") != layout.get(
        "geometry_sha256"
    ):
        return False
    expected_sessions = sorted(
        shard["session_id"] for shard in layout["shards"]
        if shard["artifact_sha256"] == artifact
    )
    contract = (
        (envelope.get("helper_plan") or {}).get("execution_contract")
        or {}
    )
    envelope_sessions = sorted(
        shard["session_id"] for shard in contract.get("phone_shards", [])
    )
    if envelope_sessions and envelope_sessions != expected_sessions:
        return False
    plans = {
        shard["operator_plan_sha256"] for shard in layout["shards"]
        if shard["artifact_sha256"] == artifact
    }
    envelope_plan = envelope.get("operator_plan_sha256")
    if envelope_plan is not None and plans and envelope_plan not in plans and (
        contract.get("phone_shards")
        and any(
            shard.get("operator_plan_sha256") not in plans
            for shard in contract["phone_shards"]
        )
    ):
        return False
    return bool(expected_sessions)


def _request_outcome(
    *,
    calls: int,
    resident_us: int,
    resident_ppm: int,
    minimum_opportunity_ppm: int,
    short: bool,
    command_generation,
    command_executable: bool,
    helper_events: list[dict],
    attached: bool,
    probed: list,
    envelope,
) -> str:
    if calls > 0:
        return "assisted"
    if resident_us == 0:
        return "no_residency"
    if resident_ppm < minimum_opportunity_ppm:
        return "insufficient_opportunity"
    if short:
        return "short_request"
    if command_generation is not None and not command_executable:
        return "stale_helper_generation"
    if helper_events:
        if not attached:
            return "not_attached"
        if not probed:
            return "attached_not_probed"
        return "probed_returned_to_zero"
    if envelope is None:
        # No exported helper events and no envelope: attachment cannot
        # be asserted; this is a probable missed attachment.
        return "unattached_no_envelope"
    if not probed:
        # Executable envelope, baseline never left: either a deliberate
        # cached desktop decision or a control failure; needs events.
        return "executable_not_probed_needs_policy_evidence"
    return "probed_returned_to_zero"


def _request_row(
    row: dict,
    *,
    ready: list[list],
    role_by_model: dict,
    helper_events_by_request: dict[str, list[dict]],
    short_output_tokens: int,
    minimum_opportunity_ppm: int,
) -> dict:
    receipt = row["completion"]["execution_receipt"]
    start_us = int(receipt["started_us"])
    end_us = int(receipt["finished_us"])
    arrival_us = int(row["replay_arrival_us"])
    role = role_by_model.get(row["model_id"], row["model_id"])
    window_us = max(1, end_us - start_us)
    resident_us = _overlap(start_us, end_us, _resident_intervals(ready, role))
    proof = row.get("physical_execution_proof") or {}
    calls = int(proof.get("phone_call_count") or 0)
    command = row.get("execution_command") or {}
    envelope = command.get("helper_envelope")
    command_generation = (
        None if not isinstance(envelope, dict)
        else envelope.get("phone_layout_generation")
    )
    command_executable = _envelope_executable(
        ready,
        envelope,
        row["terminal_ticket"]["model"]["artifact_sha256"],
        start_us,
    )
    history = row.get("fraction_history") or {}
    explored = list(history.get("explored_split_fractions_ppm") or [])
    probed = list(history.get("phone_executed_split_fractions_ppm") or [])
    selected = history.get("selected_split_fraction_ppm")
    short = int(row["output_tokens"]) < short_output_tokens
    helper_events = helper_events_by_request.get(row["request_id"], [])
    event_kinds = [event.get("kind") for event in helper_events]
    attached = "ATTACHED" in event_kinds
    attach_evidence = (
        "events" if helper_events else "none"
    )
    resident_ppm = int(resident_us * 1_000_000 / window_us)
    outcome = _request_outcome(
        calls=calls,
        resident_us=resident_us,
        resident_ppm=resident_ppm,
        minimum_opportunity_ppm=minimum_opportunity_ppm,
        short=short,
        command_generation=command_generation,
        command_executable=command_executable,
        helper_events=helper_events,
        attached=attached,
        probed=probed,
        envelope=envelope,
    )
    return {
        "index": row["combined_request_index"],
        "role": role,
        "output_tokens": int(row["output_tokens"]),
        "arrival_s": _sec(arrival_us),
        "physical_start_s": _sec(start_us),
        "physical_end_s": _sec(end_us),
        "delay_before_inference_s": _sec(start_us - arrival_us),
        "inference_s": _sec(end_us - start_us),
        "resident_ppm": int(resident_us * 1_000_000 / window_us),
        "command_helper_generation": command_generation,
        "command_helper_executable_at_start": command_executable,
        "explored_fractions_ppm": explored,
        "probed_fractions_ppm": probed,
        "selected_fraction_ppm": selected,
        "phone_calls": calls,
        "attach_evidence": attach_evidence,
        "helper_event_kinds": event_kinds,
        "outcome": outcome,
    }


def _layout_rows(proposals: dict[int, dict]) -> list[dict]:
    layouts = []
    for generation in sorted(proposals):
        row = proposals[generation]
        proposed = row.get("proposed_at_us")
        preparing = row.get("preparing_at_us")
        finished = row.get("ready_at_us") or row.get("failed_at_us")
        layouts.append({
            "generation": generation,
            "proposed_s": _sec(proposed),
            "proposed_to_preparing_s": (
                None if preparing is None or proposed is None
                else _sec(preparing - proposed)
            ),
            "load_s": (
                None if preparing is None or finished is None
                else _sec(finished - preparing)
            ),
            "outcome": (
                "READY" if "ready_at_us" in row
                else "FAILED" if "failed_at_us" in row
                else "PREPARING" if preparing is not None
                else "PROPOSED_ONLY"
            ),
            "failure_reason": row.get("failure_reason"),
        })
    return layouts


def _uncovered_work_us(
    rows: list[dict],
    ready: list[list],
    role_by_model: dict,
    arrivals: list[int],
    starts: list[int],
    ends: list[int],
) -> tuple[int, int, int]:
    # Work with no resident shard for its model: pending (arrival to end)
    # and executing (physical window) variants, both by layout evidence only.
    points = sorted({*arrivals, *starts, *ends, *(s for s, _e, _g, _l, _y in ready)})
    pending_uncovered_us = executing_uncovered_us = no_layout_us = 0
    for lo, hi in zip(points, points[1:]):
        mid = (lo + hi) // 2
        layout = next((
            sessions for s, e, _g, sessions, _y in ready
            if s <= mid and (e is None or mid < e)
        ), None)
        if layout is None:
            no_layout_us += hi - lo
        resident_roles = set() if layout is None else set(layout.values())
        pending = {
            role_by_model.get(r["model_id"], r["model_id"])
            for r, a, e in zip(rows, arrivals, ends) if a <= mid < e
        }
        executing = {
            role_by_model.get(r["model_id"], r["model_id"])
            for r, s, e in zip(rows, starts, ends) if s <= mid < e
        }
        if pending - resident_roles:
            pending_uncovered_us += hi - lo
        if executing - resident_roles:
            executing_uncovered_us += hi - lo
    return pending_uncovered_us, executing_uncovered_us, no_layout_us


def _first_resident_after_first_arrival(
    rows: list[dict],
    arrivals: list[int],
    ready: list[list],
    role_by_model: dict,
) -> dict:
    first_arrival_by_role: dict[str, int] = {}
    for r, a in zip(rows, arrivals):
        role = role_by_model.get(r["model_id"], r["model_id"])
        first_arrival_by_role[role] = min(first_arrival_by_role.get(role, a), a)
    first_ready_by_role: dict[str, int] = {}
    for s, _e, _g, sessions, _y in ready:
        for role in set(sessions.values()):
            first_ready_by_role.setdefault(role, s)
    return {
        role: _sec(first_ready_by_role[role] - arrival)
        if role in first_ready_by_role else None
        for role, arrival in sorted(first_arrival_by_role.items())
    }


def analyze(
    result: dict,
    *,
    short_output_tokens: int = 32,
    minimum_opportunity_ppm: int = 300_000,
) -> dict:
    roles = result["model_roles"]
    artifacts = result.get("model_artifacts") or {}
    role_by_artifact = {
        (artifacts.get(model_id) or {}).get("artifact_sha256"): role
        for role, model_id in roles.items()
    }
    role_by_model = {model_id: role for role, model_id in roles.items()}

    events = sorted(
        result.get("phone_residency_events") or [],
        key=lambda row: row["observed_at_us"],
    )
    ready, proposals = _ready_layouts(events, role_by_artifact)

    # Exported helper events (attach, fraction, detach) make the ladder exact.
    helper_events_by_request: dict[str, list[dict]] = {}
    for event in result.get("request_helper_events") or []:
        helper_events_by_request.setdefault(event["request_id"], []).append(event)

    requests = [
        _request_row(
            row,
            ready=ready,
            role_by_model=role_by_model,
            helper_events_by_request=helper_events_by_request,
            short_output_tokens=short_output_tokens,
            minimum_opportunity_ppm=minimum_opportunity_ppm,
        )
        for row in result["request_results"]
    ]

    layouts = _layout_rows(proposals)

    rows = result["request_results"]
    starts = [int(r["completion"]["execution_receipt"]["started_us"]) for r in rows]
    ends = [int(r["completion"]["execution_receipt"]["finished_us"]) for r in rows]
    arrivals = [int(r["replay_arrival_us"]) for r in rows]
    trace_start, trace_end = min(arrivals), max(ends)
    pending_uncovered_us, executing_uncovered_us, no_layout_us = (
        _uncovered_work_us(rows, ready, role_by_model, arrivals, starts, ends)
    )

    outcomes: dict[str, int] = {}
    for r in requests:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    summary = {
        "trace_span_s": _sec(trace_end - trace_start),
        "layouts_proposed": len(proposals),
        "layouts_ready": sum(1 for r in layouts if r["outcome"] == "READY"),
        "layouts_failed": sum(1 for r in layouts if r["outcome"] == "FAILED"),
        "layouts_never_prepared": sum(
            1 for r in layouts if r["outcome"] == "PROPOSED_ONLY"
        ),
        "phone_no_layout_s": _sec(no_layout_us),
        "pending_work_without_model_residency_s": _sec(pending_uncovered_us),
        "executing_work_without_model_residency_s": _sec(executing_uncovered_us),
        "requests": len(requests),
        "request_outcomes": dict(sorted(outcomes.items())),
        "delay_before_inference_s_max": max(
            r["delay_before_inference_s"] for r in requests
        ),
        "first_resident_after_first_arrival_s": (
            _first_resident_after_first_arrival(
                rows, arrivals, ready, role_by_model
            )
        ),
        "short_output_tokens_threshold": short_output_tokens,
        "minimum_opportunity_ppm": minimum_opportunity_ppm,
        "request_helper_events_exported": bool(
            result.get("request_helper_events")
        ),
    }
    return {
        "layouts": layouts,
        "requests": requests,
        "schema": "s42-phone-wait-timeline-v2",
        "summary": summary,
    }


def render(report: dict) -> str:
    lines = ["== summary"]
    for key, value in report["summary"].items():
        lines.append(f"  {key}: {value}")
    lines.append("== layouts (generation, proposed, wait->preparing, load, outcome)")
    for row in report["layouts"]:
        lines.append(
            f"  gen {row['generation']:>2}  t={row['proposed_s']!s:>7}s"
            f"  wait={row['proposed_to_preparing_s']!s:>6}s"
            f"  load={row['load_s']!s:>6}s  {row['outcome']}"
            + (f"  {row['failure_reason']}" if row["failure_reason"] else "")
        )
    lines.append(
        "== requests (index, role, tokens, arrival, delay, inference,"
        " resident, cmd-gen/exec, explored, probed, calls, outcome)"
    )
    for row in report["requests"]:
        lines.append(
            f"  {row['index']:>3} {row['role']:<6} tok={row['output_tokens']:>4}"
            f" arr={row['arrival_s']!s:>7}s delay={row['delay_before_inference_s']!s:>6}s"
            f" inf={row['inference_s']!s:>6}s res={row['resident_ppm'] / 10000:5.1f}%"
            f" gen={row['command_helper_generation']!s:>4}/"
            f"{'y' if row['command_helper_executable_at_start'] else 'n'}"
            f" expl={row['explored_fractions_ppm']!s:<24}"
            f" probed={'y' if row['probed_fractions_ppm'] else 'n'}"
            f" calls={row['phone_calls']:>6} {row['outcome']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--short-output-tokens", type=int, default=32)
    parser.add_argument("--minimum-opportunity", type=float, default=0.3)
    args = parser.parse_args()
    report = analyze(
        load(args.result),
        short_output_tokens=args.short_output_tokens,
        minimum_opportunity_ppm=int(args.minimum_opportunity * 1_000_000),
    )
    if args.json is not None:
        args.json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
    sys.stdout.write(render(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
