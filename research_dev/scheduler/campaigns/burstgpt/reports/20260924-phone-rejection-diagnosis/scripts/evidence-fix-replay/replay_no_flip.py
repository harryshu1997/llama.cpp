#!/usr/bin/env python3
"""Can the evidence fixes turn a recorded host decision into a phone decision?

Every promotion or keep path of the fixed controller ends in _qualifies (F1b re-qualification,
F1a resolution via _finish_verification, F2-deferred candidates reaching _select_probe_winner or
_best_valid_policy). _qualifies itself is unchanged. So a sufficient no-flip condition is: at no
recorded window of a context does any measured phone candidate pass _qualifies, evaluated with
every elimination cleared (the most permissive state F2 could leave) and with the F3/F4
eligibility rules applied to the recorded windows.

The session is rebuilt window by window (context resets at batch/membership/external-activity
changes, like record_window). F3 uses the source tree's token_stream_caught_up on the recorded
token observation times; F4 removes phone windows that overlap a session load on the helper's
phone (SESSION_LOADING..SESSION_VERIFIED publication times).

Usage: replay_no_flip.py --source TREE/research_dev RUN_DIR [REQUEST_ID ...] [--json OUT]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

SCRIPTS = ("/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/"
           "reports/20260924-phone-rejection-diagnosis/scripts")
TRANSITION = "physical:control-transition-ack"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("run_dir")
    parser.add_argument("request_ids", nargs="*")
    parser.add_argument("--json")
    parser.add_argument("--phone-prefix", default="session://op15-phone/")
    parser.add_argument("--no-guards", action="store_true", help="recorded eligibility only (no F3/F4)")
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.source))
    sys.path.insert(0, SCRIPTS)
    from scheduler._internal.adaptive_decode import AdaptiveDecodeController
    from scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeConfig, AdaptiveDecodeWindowReceipt
    from scheduler._internal.adaptive_decode_state import _AdaptiveSession
    from scheduler._internal.adaptive_decode_ops.windows import token_stream_caught_up
    from timeline import load, run_groups
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from replay_decisions import load_intervals

    obs, result = load(args.run_dir)
    loads = load_intervals(result, args.phone_prefix)
    times = {}
    with open(os.path.join(args.run_dir, "adaptive-timing-events.json")) as f:
        for event in json.load(f):
            if event["kind"] == "DECODE_BOUNDARY_OBSERVED":
                times.setdefault(event["request_id"], {}).setdefault(
                    event["token_index"], event["token_observed_at_us"])
    base_config = dict(result["adaptive_controller_configuration"])
    base_config["coarse_probe_fractions_ppm"] = tuple(base_config["coarse_probe_fractions_ppm"])
    base_config["refinement_steps_ppm"] = tuple(base_config["refinement_steps_ppm"])
    controller = AdaptiveDecodeController()
    report = []
    for group in sorted(run_groups(obs, result), key=lambda g: g["request_id"]):
        rid = group["request_id"]
        if args.request_ids and rid not in args.request_ids and rid[-3:] not in args.request_ids:
            continue
        decisions = [e for e in result["request_helper_events"]
                     if e.get("request_id") == rid and e["kind"] == "ASSISTANCE_DECISION"]
        if not decisions:
            continue
        first = decisions[0]
        config = dataclasses.replace(
            AdaptiveDecodeConfig(**base_config),
            minimum_energy_saving_ppm=first["minimum_energy_saving_ppm"],
            maximum_latency_ppm=first["maximum_latency_ppm"],
            allow_assumed_phone_power_for_operational_selection=True)
        receipts = [AdaptiveDecodeWindowReceipt.from_json(w) for w in group["windows"]]
        baseline = next((r.policy for r in receipts if r.policy.baseline), None)
        phones = []
        for r in receipts:
            if not r.policy.baseline and r.policy not in phones:
                phones.append(r.policy)
        if baseline is None or not phones:
            continue
        label = {baseline.policy_hash: "B", **{p.policy_hash: "P%d" % (p.split_fraction_ppm // 10000)
                                              for p in phones}}
        ticket = next(r for r in result["request_results"] if r["request_id"] == rid)
        session = _AdaptiveSession(
            request_id=rid, ticket_id=group["ticket_id"], model_artifact_sha256=group["model_artifact_sha256"],
            planning_profile_sha256=group["planning_profile_sha256"],
            component_capability_sha256=group["planning_profile_sha256"],
            baseline=baseline, candidates=tuple(phones), output_tokens=ticket["output_tokens"],
            context_length=receipts[0].context_length - receipts[0].token_start,
            active_batch=receipts[0].active_batch, deadline_us=ticket["terminal_ticket"]["request"]["deadline_us"],
            config=config, slot_id=receipts[0].slot_id, helper_evidence_state=first["helper_evidence_state"])
        session.probe_candidates = list(phones)
        token_times = times.get(rid, {})
        previous_caught_up = False
        guarded = {}
        qualifying = []
        by_token = {}
        for e in decisions:
            by_token.setdefault(e["token_index"], e)
        for index, receipt in enumerate(receipts):
            observed = [(t, token_times[t]) for t in range(receipt.token_start + 1, receipt.token_end + 1)
                        if t in token_times]
            transition = TRANSITION in receipt.evidence_ids
            caught_up = (not transition) and token_stream_caught_up(
                receipt.token_start, receipt.started_at_us, observed)
            late_start = previous_caught_up and receipt.applied_ack is None and not transition
            previous_caught_up = caught_up
            disturbed = not receipt.policy.baseline and any(
                start <= receipt.finished_at_us and receipt.started_at_us <= end for start, end in loads)
            if not args.no_guards and receipt.measurement_eligible and (caught_up or late_start or disturbed):
                guarded[index] = "TOKEN_STREAM_CATCH_UP" if (caught_up or late_start) else "HELPER_PHONE_SESSION_LOAD"
                receipt = dataclasses.replace(receipt, measurement_eligible=False)
            session.records.append(receipt)
            session.acknowledged_policy = receipt.policy
            session.eliminated_policy_reasons = {}
            for policy in phones:
                if not controller._current_valid_records(session, policy, operational=True):
                    continue
                if controller._qualifies(session, policy, receipt.token_end, receipt.finished_at_us):
                    event = by_token.get(receipt.token_end)
                    qualifying.append({
                        "window": index, "token": receipt.token_end, "batch": session.active_batch,
                        "candidate": label[policy.policy_hash], "window_policy": label[receipt.policy.policy_hash],
                        "recorded_reason": None if event is None else event["reason"],
                        "recorded_selected_fraction": None if event is None else event["selected_fraction_ppm"],
                    })
            next_batch = receipt.next_active_batch or receipt.active_batch
            if receipt.membership_changed or next_batch != session.active_batch or receipt.external_activity_changed:
                session.active_batch = next_batch
                session.context_record_start = len(session.records)
        batches = sorted({r.active_batch for r in receipts})
        phone_at = {b: sorted({label[r.policy.policy_hash] for r in receipts
                               if r.active_batch == b and not r.policy.baseline}) for b in batches}
        recorded_elims = {}
        for e in decisions:
            for key, value in e["eliminated_policy_reasons"].items():
                recorded_elims.setdefault(label.get(key, key[7:15]), set()).add(value)
        row = {
            "request_id": rid, "final": ("B" if group["final_policy"]["baseline"] else
                                         "P%d" % (group["final_policy"]["split_fraction_ppm"] // 10000)),
            "batches": batches, "phone_measured_at": phone_at,
            "recorded_eliminations": {k: sorted(v) for k, v in recorded_elims.items()},
            "guarded_windows": guarded,
            "qualifying_points": qualifying,
            "qualifying_at_batch_2": [q for q in qualifying if q["batch"] == 2],
        }
        report.append(row)
        print("%s final=%s batches=%s guarded=%d qualifying(any batch)=%d qualifying@b2=%d" % (
            rid[-3:], row["final"], batches, len(guarded), len(qualifying), len(row["qualifying_at_batch_2"])))
        if row["qualifying_at_batch_2"]:
            for q in row["qualifying_at_batch_2"][:10]:
                print("     b2 qualifies:", q)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
