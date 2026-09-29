#!/usr/bin/env python3
"""Closed-loop replay of one recorded request through the adaptive controller of a source tree.

The controller is driven through its public API (start, boundary per token, record_window,
acknowledge, helper_disturbance when the tree has it). While its decisions match the recording,
the recorded windows, token times and acknowledgements are fed verbatim ("replay"). From the
first decision that differs, every window is synthesized ("synthetic") from the recorded windows
of the same policy in the same request: a per-policy cursor walks that policy's recorded regular
windows in order, then cycles its eligible ones; phone windows that overlapped a helper-phone
session load are never used as synthetic sources; a policy the request never measured undisturbed
takes windows of the same model and split fraction from the run's other requests. Transitions take the request's median recorded transition length. Phone session loads on
the helper's phone (SESSION_LOADING..SESSION_VERIFIED publication times) are reported at window
ends and acknowledgements, as _unified does.

Usage: replay_decisions.py --source TREE/research_dev RUN_DIR REQUEST_ID [--json OUT] [--disable F1a,F4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys

SCRIPTS = ("/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/"
           "reports/20260924-phone-rejection-diagnosis/scripts")
TRANSITION = "physical:control-transition-ack"


def label(policy):
    return "B" if policy.baseline else "P%d" % (policy.split_fraction_ppm // 10000)


def load_intervals(result, phone_prefix):
    """(start_us, end_us) of session loads on one phone, by publication time."""
    starts, intervals = {}, []
    for event in result["phone_residency_events"]:
        session = event.get("session")
        if not isinstance(session, dict) or not str(session.get("endpoint", "")).startswith(phone_prefix):
            continue
        at = event.get("published_at_us", event["observed_at_us"])
        key = (session["session_id"], event.get("layout_generation"))
        if event["kind"] == "SESSION_LOADING":
            starts[key] = at
        elif event["kind"] in {"SESSION_VERIFIED", "SESSION_FAILED", "SESSION_UNAVAILABLE"} and key in starts:
            intervals.append((starts.pop(key), at))
    intervals.extend((start, 2**62) for start in starts.values())
    return sorted(intervals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("run_dir")
    parser.add_argument("request_id")
    parser.add_argument("--json")
    parser.add_argument("--phone-prefix", default="session://op15-phone/")
    parser.add_argument("--disable", default="",
                        help="comma-separated fixes to switch off in-process: F1a,F1b,F2,F3,F4")
    args = parser.parse_args()
    disabled = {x.strip() for x in args.disable.split(",") if x.strip()}
    sys.path.insert(0, os.path.abspath(args.source))
    sys.path.insert(0, SCRIPTS)
    from scheduler._internal.adaptive_decode import AdaptiveDecodeController
    from scheduler._internal.adaptive_decode_contracts import (
        AdaptiveDecodeConfig, AdaptiveDecodeGroupedObservation, AdaptiveDecodePolicyAck,
        AdaptiveDecodeRawWindowObservation, AdaptiveDecodeWindowReceipt)
    from timeline import load, run_groups
    import scheduler._internal.adaptive_decode_ops.promotion as promotion_ops
    import scheduler._internal.adaptive_decode_ops.sequencing as sequencing_ops
    import scheduler._internal.adaptive_decode_ops.windows as windows_ops
    if "F1a" in disabled:
        sequencing_ops._incumbent_inconclusive = lambda *a, **k: False
    if "F1b" in disabled:
        sequencing_ops._candidate_requalified = lambda *a, **k: False
    if "F2" in disabled:
        sequencing_ops._single_reference_rejection = lambda *a, **k: None
        promotion_ops._single_reference_probe = lambda *a, **k: False
    if "F3" in disabled:
        windows_ops.token_stream_caught_up = lambda *a, **k: False

    obs, result = load(args.run_dir)
    group = next(g for g in run_groups(obs, result) if g["request_id"] == args.request_id)
    raw_windows = group["windows"]
    windows = [AdaptiveDecodeWindowReceipt.from_json(w) for w in raw_windows]
    decisions = [e for e in result["request_helper_events"]
                 if e.get("request_id") == args.request_id and e["kind"] == "ASSISTANCE_DECISION"]
    first = decisions[0]
    ticket = next(r for r in result["request_results"] if r["request_id"] == args.request_id)
    output_tokens = ticket["output_tokens"]
    times = {}
    with open(os.path.join(args.run_dir, "adaptive-timing-events.json")) as f:
        for event in json.load(f):
            if event["kind"] == "DECODE_BOUNDARY_OBSERVED" and event["request_id"] == args.request_id:
                times.setdefault(event["token_index"], event["token_observed_at_us"])
    loads = load_intervals(result, args.phone_prefix)

    def loading(at_us):
        return any(start <= at_us <= end for start, end in loads)

    base_config = dict(result["adaptive_controller_configuration"])
    base_config["coarse_probe_fractions_ppm"] = tuple(base_config["coarse_probe_fractions_ppm"])
    base_config["refinement_steps_ppm"] = tuple(base_config["refinement_steps_ppm"])
    config = dataclasses.replace(
        AdaptiveDecodeConfig(**base_config),
        minimum_energy_saving_ppm=first["minimum_energy_saving_ppm"],
        maximum_latency_ppm=first["maximum_latency_ppm"],
        allow_assumed_phone_power_for_operational_selection=True)
    baseline = next(w.policy for w in windows if w.policy.baseline)
    phones = []
    for w in windows:
        if not w.policy.baseline and w.policy not in phones:
            phones.append(w.policy)
    last = decisions[-1]
    generation = last["helper_layout_generation"]
    geometry = last["helper_layout_geometry_sha256"]

    # Per-policy sources for synthetic windows (regular windows only, recorded order).
    def gaps_of(w):
        previous, result_gaps = w.started_at_us, []
        for token in range(w.token_start + 1, w.token_end + 1):
            at = times.get(token)
            if at is None or at < previous:
                return None
            result_gaps.append(at - previous)
            previous = at
        return result_gaps

    regular = [w for w in windows if TRANSITION not in w.evidence_ids and w.execution_context_available
               and w.failure_reason is None]
    pools, cycles, borrowed = {}, {}, {}
    for w in regular:
        disturbed = not w.policy.baseline and any(
            start <= w.finished_at_us and w.started_at_us <= end for start, end in loads)
        if disturbed:
            continue
        pools.setdefault(w.policy.policy_hash, []).append(w)
        if w.measurement_eligible:
            cycles.setdefault(w.policy.policy_hash, []).append(w)
    # Same model and split fraction from the run's other requests, for a policy this request
    # never measured undisturbed.
    fallback = {}
    for other in run_groups(obs, result):
        if other["request_id"] == args.request_id or other["model_artifact_sha256"] != group["model_artifact_sha256"]:
            continue
        for raw in other["windows"]:
            w = AdaptiveDecodeWindowReceipt.from_json(raw)
            if (TRANSITION in w.evidence_ids or not w.execution_context_available or w.failure_reason
                    or w.active_batch != windows[0].active_batch):
                continue
            if not w.policy.baseline and any(
                    start <= w.finished_at_us and w.started_at_us <= end for start, end in loads):
                continue
            fallback.setdefault((w.policy.baseline, w.policy.split_fraction_ppm), []).append(w)
    for policy in (baseline, *phones):
        key = policy.policy_hash
        if key not in pools and (policy.baseline, policy.split_fraction_ppm) in fallback:
            pools[key] = fallback[(policy.baseline, policy.split_fraction_ppm)]
            cycles[key] = [w for w in pools[key] if w.measurement_eligible]
            borrowed[label(policy)] = sorted({w.request_id[-3:] for w in pools[key]})
    cursor = {key: 0 for key in pools}
    cycle_cursor = {key: 0 for key in cycles}
    transition_lengths = [w.token_count for w in windows if TRANSITION in w.evidence_ids]
    transition_tokens = int(statistics.median(transition_lengths)) if transition_lengths else 3

    def source_for(policy):
        key = policy.policy_hash
        if cursor.get(key, 0) < len(pools.get(key, ())):
            row = pools[key][cursor[key]]
            cursor[key] += 1
            return row
        rows = cycles.get(key) or pools.get(key)
        if not rows:
            raise SystemExit("no undisturbed source windows for " + label(policy))
        row = rows[cycle_cursor.get(key, 0) % len(rows)]
        cycle_cursor[key] = cycle_cursor.get(key, 0) + 1
        return row

    def observation(source, tokens, *, transition=False, context=True):
        scale = lambda value: value * tokens // max(1, source.token_count)
        evidence = tuple(sorted(set(source.evidence_ids) - {TRANSITION, "physical:token-boundary-ack"}
                                | {TRANSITION if transition else "physical:token-boundary-ack"}))
        calls = source.completed_phone_calls
        return AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain={k: scale(v) for k, v in source.fleet_energy_uj_by_domain.items()},
            phone_compute_us=scale(source.phone_compute_us), usb_transfer_us=scale(source.usb_transfer_us),
            rpc_us=scale(source.rpc_us), exposed_tail_us=scale(source.exposed_tail_us),
            output_valid=True, evidence_ids=evidence, energy_boundary_id=source.energy_boundary_id,
            energy_attribution_kind=source.energy_attribution_kind,
            usb_upload_bytes=scale(source.usb_upload_bytes), usb_download_bytes=scale(source.usb_download_bytes),
            desktop_compute_us=scale(source.desktop_compute_us),
            active_batch=source.active_batch, execution_context_available=context,
            completed_phone_calls=None if calls is None else (max(1, scale(calls)) if calls else 0),
            completed_phone_input_rows=(None if calls is None else (max(
                max(1, scale(calls)), scale(source.completed_phone_input_rows)) if calls else 0)),
            external_activity_sha256=source.external_activity_sha256,
        )

    def recorded_observation(w):
        return AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain=dict(w.fleet_energy_uj_by_domain), phone_compute_us=w.phone_compute_us,
            usb_transfer_us=w.usb_transfer_us, rpc_us=w.rpc_us, exposed_tail_us=w.exposed_tail_us,
            output_valid=w.output_valid, evidence_ids=w.evidence_ids, energy_boundary_id=w.energy_boundary_id,
            energy_attribution_kind=w.energy_attribution_kind, failure_reason=w.failure_reason,
            usb_upload_bytes=w.usb_upload_bytes, usb_download_bytes=w.usb_download_bytes,
            desktop_compute_us=w.desktop_compute_us, useful_overlap_us=w.useful_overlap_us,
            request_queue_delay_us=w.request_queue_delay_us, active_batch=w.active_batch,
            next_active_batch=w.next_active_batch, membership_changed=w.membership_changed,
            execution_context_available=w.execution_context_available, usb_h2d_us=w.usb_h2d_us,
            usb_d2h_us=w.usb_d2h_us, accounting_token_count=w.accounting_token_count,
            configured_queue_depth=w.configured_queue_depth, maximum_active_slots=w.maximum_active_slots,
            maximum_outstanding_transfers=w.maximum_outstanding_transfers,
            completed_phone_calls=w.completed_phone_calls,
            completed_phone_input_rows=w.completed_phone_input_rows,
            external_activity_sha256=w.external_activity_sha256,
        )

    controller = AdaptiveDecodeController()
    this_run = {g["ticket_id"] for g in run_groups(obs, result)}
    history = []
    for raw in obs["groups"]:
        if raw["request_id"] == args.request_id:
            continue
        if raw["ticket_id"] in this_run and max(w["finished_at_us"] for w in raw["windows"]) > windows[0].started_at_us:
            continue
        grouped = AdaptiveDecodeGroupedObservation.from_json(raw)
        controller._history[grouped.grouped_observation_sha256] = grouped
        history.append(raw["request_id"])
    has_disturbance = hasattr(controller, "helper_disturbance") and "F4" not in disabled
    rid = args.request_id
    slot = windows[0].slot_id

    def report_disturbance(at_us):
        if has_disturbance:
            controller.helper_disturbance(rid, reason="HELPER_PHONE_SESSION_LOAD" if loading(at_us) else None)

    start_us = windows[0].started_at_us
    directive = controller.start(
        request_id=rid, ticket_id=group["ticket_id"], model_artifact_sha256=group["model_artifact_sha256"],
        planning_profile_sha256=group["planning_profile_sha256"],
        component_capability_sha256=group["planning_profile_sha256"],
        baseline=baseline, candidates=tuple(phones), output_tokens=output_tokens,
        context_length=windows[0].context_length - windows[0].token_start, active_batch=windows[0].active_batch,
        deadline_us=ticket["terminal_ticket"]["request"]["deadline_us"], slot_id=slot,
        first_token_index=windows[0].token_start, first_token_at_us=start_us, config=config,
        helper_available=True, helper_layout_generation=generation, helper_layout_geometry_sha256=geometry,
        helper_evidence_state=first["helper_evidence_state"])
    report_disturbance(start_us)
    log = [(windows[0].token_start, directive.reason, label(controller.active_policy(rid)
                                                                  or baseline), {}, controller._sessions[rid].stage)]
    synchronized, next_recorded, diverged_at = True, 0, None
    tail = output_tokens - min(2, output_tokens - 1)
    segments = []

    def session():
        return controller._sessions.get(rid)

    def names(reasons):
        by_hash = {p.policy_hash: label(p) for p in (baseline, *phones)}
        return {by_hash.get(k, k[7:15]): v for k, v in reasons.items()}

    def note(token, directive):
        s = session() or controller._sealed_sessions.get(rid)
        log.append((token, directive.reason,
                    label(directive.control.policy) if directive.control is not None
                    else label(s.current_policy) if s.current_policy is not None else "-",
                    names(s.eliminated_policy_reasons), s.stage))

    while True:
        s = session()
        if s is None:
            break
        if s.awaiting_control is not None:
            control = s.awaiting_control
            start_token, started_at = s.transition_start_token, s.transition_started_at_us
            recorded = windows[next_recorded] if synchronized and next_recorded < len(windows) else None
            if recorded is not None and TRANSITION in recorded.evidence_ids \
                    and recorded.token_start == start_token and recorded.policy == s.transition_policy:
                ack_token, ack_us = recorded.token_end, recorded.finished_at_us
                transition_obs = recorded_observation(recorded)
                next_recorded += 1
            elif recorded is not None and recorded.applied_ack is not None \
                    and recorded.token_start == start_token and recorded.policy == control.policy:
                ack_token, ack_us, transition_obs = start_token, recorded.applied_ack.applied_at_us, None
            else:
                if synchronized:
                    synchronized, diverged_at = False, start_token
                ack_token = min(output_tokens, start_token + transition_tokens)
                source = source_for(s.transition_policy)
                ack_us = started_at + max(1, source.latency_per_token_us * (ack_token - start_token)) + 1_000
                transition_obs = (None if ack_token == start_token else
                                  observation(source, ack_token - start_token, transition=True))
            segments.append((s.transition_policy, start_token, ack_token, transition_obs))
            report_disturbance(ack_us)
            directive = controller.acknowledge(rid, AdaptiveDecodePolicyAck(
                request_id=rid, slot_id=slot, plan_generation=control.plan_generation,
                applied_token_index=ack_token, applied_at_us=ack_us, policy_hash=control.policy.policy_hash),
                transition_observation=transition_obs)
            note(ack_token, directive)
            continue
        if s.window_start_token is None or s.target_token is None:
            break
        policy, start_token, target = s.current_policy, s.window_start_token, s.target_token
        if start_token >= tail:
            segments.append((policy, start_token, output_tokens, None))
            break
        recorded = windows[next_recorded] if synchronized and next_recorded < len(windows) else None
        if recorded is not None and recorded.failure_reason is not None:
            # The run discarded this released-slot tail instead of recording it.
            segments.append((policy, start_token, output_tokens, None))
            break
        if (recorded is not None and recorded.policy == policy and recorded.token_start == start_token
                and recorded.token_end == target and TRANSITION not in recorded.evidence_ids):
            if recorded in pools.get(policy.policy_hash, ()):
                cursor[policy.policy_hash] = pools[policy.policy_hash].index(recorded) + 1
            token_times = [(t, times[t]) for t in range(start_token + 1, target) if t in times]
            token_times.append((target, recorded.finished_at_us))
            obs_value = recorded_observation(recorded)
            next_recorded += 1
            source = recorded
        else:
            if synchronized:
                synchronized, diverged_at = False, start_token
            source = source_for(policy)
            tokens = target - start_token
            gaps = gaps_of(source)
            if gaps is None or len(gaps) != tokens:
                gaps = [source.latency_per_token_us] * tokens
            at, token_times = s.window_start_us, []
            for index, gap in enumerate(gaps):
                at += max(1, gap)
                token_times.append((start_token + index + 1, at))
            obs_value = observation(source, tokens)
        boundary_directive = None
        token_times = [(t, at) for t, at in token_times if at > s.window_start_us or t == token_times[-1][0]]
        for token, at in token_times:
            if token >= tail:
                controller.seal_tail(rid, slot_id=slot, token_index=token, reason="server_release_guard")
                boundary_directive = controller.boundary(rid, slot_id=slot, token_index=token, at_us=at,
                                                         terminal=True)
                break
            boundary_directive = controller.boundary(rid, slot_id=slot, token_index=token, at_us=at)
            if boundary_directive is not None:
                break
        if boundary_directive is None or boundary_directive.boundary is None:
            break
        boundary = boundary_directive.boundary
        report_disturbance(boundary.finished_at_us)
        segments.append((policy, boundary.token_start, boundary.token_end, obs_value))
        directive = controller.record_window(rid, boundary, obs_value)
        note(boundary.token_end, directive)
        if directive.reason in {"TAIL_SEALED", "COHORT_MEMBERSHIP_CHANGED"}:
            s = session() or controller._sealed_sessions.get(rid)
            if s is not None and s.current_policy is not None:
                segments.append((s.current_policy, boundary.token_end, output_tokens, None))
            break

    final = session() or controller._sealed_sessions.get(rid)
    tokens_by = {}
    energy_uj = 0
    for policy, start_token, end_token, obs_value in segments:
        key = "phone" if not policy.baseline else "host"
        tokens_by[key] = tokens_by.get(key, 0) + end_token - start_token
        if obs_value is not None:
            energy_uj += sum(obs_value.fleet_energy_uj_by_domain.values())
    recorded_tokens = {}
    for w in windows:
        key = "phone" if not w.policy.baseline else "host"
        recorded_tokens[key] = recorded_tokens.get(key, 0) + w.token_count
    records = final.records
    summary = {
        "request_id": rid, "source": args.source, "disabled": sorted(disabled),
        "history_groups_this_run": [r for r in history if r.startswith(rid.rsplit(":", 1)[0])],
        "borrowed_sources": borrowed,
        "diverged_at_token": diverged_at,
        "final_policy": label(final.current_policy) if final.current_policy else None,
        "incumbent": None if final.incumbent_policy is None else label(final.incumbent_policy),
        "eliminated": names(final.eliminated_policy_reasons),
        "state_history": final.state_history,
        "tokens": tokens_by, "recorded_tokens": recorded_tokens,
        "window_energy_j": round(energy_uj / 1e6, 1),
        "recorded_window_energy_j": round(sum(w.whole_fleet_energy_uj for w in windows) / 1e6, 1),
        "guarded_windows": {str(r.window_index): getattr(r, "measurement_ineligible_reason", None)
                            for r in records if getattr(r, "measurement_ineligible_reason", None)},
        "decisions": [{"token": t, "reason": r, "policy": p, "eliminated": e, "stage": st}
                      for t, r, p, e, st in log],
        "windows": [{"w": r.window_index, "policy": label(r.policy), "tok": [r.token_start, r.token_end],
                     "eligible": r.measurement_eligible,
                     "guard": getattr(r, "measurement_ineligible_reason", None),
                     "j_tok": round(r.energy_per_token_uj / 1e6, 2), "ms_tok": round(r.latency_per_token_us / 1e3, 1)}
                    for r in records],
    }
    print("%s  source=%s disabled=%s" % (rid, args.source, sorted(disabled) or "-"))
    print("  diverged from the recording at token:", diverged_at)
    print("  final policy: %s | incumbent: %s | eliminated: %s" % (
        summary["final_policy"], summary["incumbent"], summary["eliminated"]))
    print("  tokens simulated:", tokens_by, "| recorded:", recorded_tokens)
    print("  window energy (ASSUMED_4P5W, diagnostic): simulated %.1f J, recorded %.1f J" % (
        summary["window_energy_j"], summary["recorded_window_energy_j"]))
    print("  guard-removed windows:", summary["guarded_windows"])
    print("  synthetic sources borrowed from other requests (same model, same fraction):", borrowed)
    previous = None
    for d in summary["decisions"]:
        text = "%s %s" % (d["reason"], d["eliminated"] or "")
        if text != previous or d["reason"] not in {"WINDOW_OPENED"}:
            print("   tok %4d  %-32s -> %-5s stage=%-22s %s" % (
                d["token"], d["reason"], d["policy"], d["stage"], d["eliminated"] or ""))
        previous = text
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
