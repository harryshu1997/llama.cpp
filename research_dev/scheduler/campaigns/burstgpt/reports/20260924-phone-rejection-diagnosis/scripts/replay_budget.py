#!/usr/bin/env python3
"""Why could a request not re-probe the phone after its batch composition changed?

Rebuilds the session of one request at a recorded boundary (records up to that boundary, the recorded
active batch / context start, probe tokens, exploration high-water mark and probe attempts from the
ASSISTANCE_DECISION event at that token) and evaluates _measurement_pair_budget's inputs for each phone
candidate with the run-time source.

Usage: PYTHONDONTWRITEBYTECODE=1 replay_budget.py --source RUN_SOURCE/research_dev RUN_DIR REQUEST_ID TOKEN
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("run_dir")
    parser.add_argument("request_id")
    parser.add_argument("token", type=int)
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.source))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scheduler._internal.adaptive_decode import AdaptiveDecodeController
    from scheduler._internal.adaptive_decode_contracts import AdaptiveDecodeConfig, AdaptiveDecodeWindowReceipt
    from scheduler._internal.adaptive_decode_state import _AdaptiveSession
    from timeline import load, run_groups

    obs, result = load(args.run_dir)
    group = next(g for g in run_groups(obs, result) if g["request_id"] == args.request_id)
    decisions = [e for e in result["request_helper_events"]
                 if e.get("request_id") == args.request_id and e["kind"] == "ASSISTANCE_DECISION"]
    event = next(e for e in decisions if e["token_index"] == args.token)
    first = decisions[0]
    base_config = dict(result["adaptive_controller_configuration"])
    base_config["coarse_probe_fractions_ppm"] = tuple(base_config["coarse_probe_fractions_ppm"])
    base_config["refinement_steps_ppm"] = tuple(base_config["refinement_steps_ppm"])
    config = dataclasses.replace(
        AdaptiveDecodeConfig(**base_config),
        minimum_energy_saving_ppm=first["minimum_energy_saving_ppm"],
        maximum_latency_ppm=first["maximum_latency_ppm"],
        allow_assumed_phone_power_for_operational_selection=True)
    receipts = [AdaptiveDecodeWindowReceipt.from_json(w) for w in group["windows"]
                if w["token_end"] <= args.token]
    all_receipts = [AdaptiveDecodeWindowReceipt.from_json(w) for w in group["windows"]]
    baseline = next(r.policy for r in all_receipts if r.policy.baseline)
    phone = []
    for r in all_receipts:
        if not r.policy.baseline and r.policy not in phone:
            phone.append(r.policy)
    ticket = next(r for r in result["request_results"] if r["request_id"] == args.request_id)
    batch = receipts[-1].next_active_batch or receipts[-1].active_batch
    start = len(receipts)
    while start > 0 and receipts[start - 1].active_batch == batch:
        start -= 1
    session = _AdaptiveSession(
        request_id=group["request_id"], ticket_id=group["ticket_id"],
        model_artifact_sha256=group["model_artifact_sha256"],
        planning_profile_sha256=group["planning_profile_sha256"],
        component_capability_sha256=group["planning_profile_sha256"],
        baseline=baseline, candidates=tuple(phone), output_tokens=ticket["output_tokens"],
        context_length=all_receipts[0].context_length - all_receipts[0].token_start,
        active_batch=batch, deadline_us=ticket["terminal_ticket"]["request"]["deadline_us"], config=config,
        slot_id=receipts[0].slot_id, helper_evidence_state=event["helper_evidence_state"],
        state="EXPLOITING")
    session.records = receipts
    session.context_record_start = start
    session.probe_candidates = list(phone)
    session.probe_tokens = config.maximum_probe_tokens - event["remaining_probe_tokens"]
    session.exploration_overhead_high_water_uj = event["estimated_exploration_overhead_uj"]
    controller = AdaptiveDecodeController()
    at_us = receipts[-1].finished_at_us
    remaining = controller._remaining_tokens(session, args.token)
    baseline_energy = controller._estimated_token_energy(session, baseline)
    spent = controller._spent_exploration_energy(session)
    print("request %s token %d batch %d context_record_start %d remaining %d probe_tokens %d" % (
        args.request_id, args.token, batch, start, remaining, session.probe_tokens))
    print("  baseline energy estimate (host lower bound) %.2f J/tok; spent exploration %.1f J; "
          "energy allowance = max(0, %.2f * %d * %.2f - spent) = %.1f J" % (
              (baseline_energy or 0) / 1e6, spent / 1e6, (baseline_energy or 0) / 1e6, remaining,
              config.exploration_energy_budget_ppm / 1e6,
              max(0, (baseline_energy or 0) * remaining * config.exploration_energy_budget_ppm // 1_000_000
                  - spent) / 1e6))
    for policy in phone:
        budget = controller._measurement_pair_budget(session, policy, args.token, at_us)
        print("  P%d: _measurement_pair_budget -> %s" % (
            policy.split_fraction_ppm // 10000, "None (no probe)" if budget is None else budget))


if __name__ == "__main__":
    main()
