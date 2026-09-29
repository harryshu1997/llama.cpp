#!/usr/bin/env python3
"""Re-evaluate the controller's qualification functions on the recorded windows, with the exact source
that ran (SOURCE_MANIFEST-verified tree), after every recorded window of one request.

Usage:
  PYTHONDONTWRITEBYTECODE=1 replay_qualification.py --source RUN_SOURCE/research_dev RUN_DIR REQUEST_ID
      [--eliminated-from-events] [--json OUT]

For each boundary it prints, per phone policy P and for the baseline B:
  bounds = (energy mean, lower, upper [J/tok], latency mean, upper [ms/tok]) from _bounds(operational=True)
  learn  = _learning_probe_improves(P)       (LEARNING-state paired gate)
  qual   = _qualifies(P)                     (promotion / incumbent gate)
  more   = _qualification_needs_more_evidence(P, check_budget=False)
  target = comparable_measurement_targets(P) (window counts that would resolve the bound)
  update_elim = what _update_elimination(P) (run after every phone window) would record, evaluated on a
                copy with P not yet eliminated and not the incumbent
Eliminations are taken from the ASSISTANCE_DECISION event emitted at that boundary when
--eliminated-from-events is given (what the controller actually had); otherwise none (counterfactual).
Nothing here mutates controller state other than a private session object.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="research_dev directory of the run-time source")
    parser.add_argument("run_dir")
    parser.add_argument("request_id")
    parser.add_argument("--eliminated-from-events", action="store_true")
    parser.add_argument("--json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exclude-windows", default="",
                        help="comma-separated window indexes treated as measurement-ineligible (counterfactual)")
    parser.add_argument("--sqrt-bands", action="store_true",
                        help="counterfactual: continuous sqrt instead of math.isqrt in the uncertainty bands")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.source))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scheduler._internal.adaptive_decode import AdaptiveDecodeController
    from scheduler._internal.adaptive_decode_contracts import (
        AdaptiveDecodeConfig, AdaptiveDecodePolicy, AdaptiveDecodeWindowReceipt)
    from scheduler._internal.adaptive_decode_state import _AdaptiveSession
    from scheduler._internal.adaptive_decode_ops.budgeting import comparable_measurement_targets
    from timeline import load, run_groups
    if args.sqrt_bands:
        import math
        import scheduler._internal.adaptive_decode_ops.bounds as bounds_module
        bounds_module.isqrt = math.sqrt  # _bounds/_latency_bounds/_current_latency_bounds only
    excluded = {int(x) for x in args.exclude_windows.split(",") if x.strip()}

    obs, result = load(args.run_dir)
    group = next(g for g in run_groups(obs, result) if g["request_id"] == args.request_id)
    decisions = [e for e in result["request_helper_events"]
                 if e.get("request_id") == args.request_id and e["kind"] == "ASSISTANCE_DECISION"]
    first = decisions[0]
    ticket = next(r for r in result["request_results"] if r["request_id"] == args.request_id)
    deadline_us = ticket["terminal_ticket"]["request"]["deadline_us"]

    base_config = dict(result["adaptive_controller_configuration"])
    base_config["coarse_probe_fractions_ppm"] = tuple(base_config["coarse_probe_fractions_ppm"])
    base_config["refinement_steps_ppm"] = tuple(base_config["refinement_steps_ppm"])
    # _adaptive_start_config(): campaign ratios + phone power profile (allow_assumed_for_scheduling=true).
    config = dataclasses.replace(
        AdaptiveDecodeConfig(**base_config),
        minimum_energy_saving_ppm=first["minimum_energy_saving_ppm"],
        maximum_latency_ppm=first["maximum_latency_ppm"],
        allow_assumed_phone_power_for_operational_selection=True,
    )

    receipts = [AdaptiveDecodeWindowReceipt.from_json(w) for w in group["windows"]]
    for raw, receipt in zip(group["windows"], receipts):
        assert receipt.record_sha256 == raw["record_sha256"], "window hash differs"
    receipts = [dataclasses.replace(r, measurement_eligible=False) if i in excluded else r
                for i, r in enumerate(receipts)]
    baseline = next(r.policy for r in receipts if r.policy.baseline)
    phone = []
    for r in receipts:
        if not r.policy.baseline and r.policy not in phone:
            phone.append(r.policy)
    labels = {baseline.policy_hash: "B"}
    labels.update({p.policy_hash: "P%d" % (p.split_fraction_ppm // 10000) for p in phone})

    controller = AdaptiveDecodeController()
    session = _AdaptiveSession(
        request_id=group["request_id"], ticket_id=group["ticket_id"],
        model_artifact_sha256=group["model_artifact_sha256"],
        planning_profile_sha256=group["planning_profile_sha256"],
        component_capability_sha256=group["planning_profile_sha256"],
        baseline=baseline, candidates=tuple(phone), output_tokens=ticket["output_tokens"],
        context_length=receipts[0].context_length - receipts[0].token_start,
        active_batch=receipts[0].active_batch, deadline_us=deadline_us, config=config,
        slot_id=receipts[0].slot_id, helper_evidence_state=first["helper_evidence_state"],
    )
    session.probe_candidates = list(phone)
    by_token = {}
    for e in decisions:
        by_token.setdefault(e["token_index"], e)

    def fmt_bounds(b):
        if b is None:
            return None
        return (round(b[0] / 1e6, 2), round(b[1] / 1e6, 2), round(b[2] / 1e6, 2),
                round(b[3] / 1e3, 1), round(b[4] / 1e3, 1))

    rows = []
    for index, receipt in enumerate(receipts):
        if receipt.active_batch != session.active_batch:
            session.active_batch = receipt.active_batch
            session.context_record_start = index
        session.records.append(receipt)
        token, at_us = receipt.token_end, receipt.finished_at_us
        event = by_token.get(token)
        session.eliminated_policy_reasons = (
            dict(event["eliminated_policy_reasons"]) if args.eliminated_from_events and event else {})
        session.acknowledged_policy = receipt.policy
        row = {
            "window": index, "policy": labels[receipt.policy.policy_hash], "tok": [receipt.token_start, token],
            "eligible": receipt.measurement_eligible, "j_tok": round(receipt.energy_per_token_uj / 1e6, 2),
            "ms_tok": round(receipt.latency_per_token_us / 1e3, 1),
            "recorded_reason": event["reason"] if event else None,
            "B": fmt_bounds(controller._bounds(session, baseline, operational=True)),
            "phone": {},
        }
        for policy in phone:
            label = labels[policy.policy_hash]
            bounds = controller._bounds(session, policy, operational=True)
            if bounds is None:
                continue
            probe = copy.copy(session)
            probe.eliminated_policy_reasons = dict(session.eliminated_policy_reasons)
            probe.eliminated_policy_reasons.pop(policy.policy_hash, None)
            probe.incumbent_policy = None
            controller._update_elimination(probe, policy)
            row["phone"][label] = {
                "update_elimination": probe.eliminated_policy_reasons.get(policy.policy_hash),
                "bounds": fmt_bounds(bounds),
                "learn": controller._learning_probe_improves(session, policy),
                "qual": controller._qualifies(session, policy, token, at_us),
                "more": controller._qualification_needs_more_evidence(
                    session, policy, token, at_us, check_budget=False),
                "target": comparable_measurement_targets(controller, session, policy),
                "eliminated": session.eliminated_policy_reasons.get(policy.policy_hash),
            }
        rows.append(row)
        if not args.quiet:
            print("W%-3d %-4s tok=%-10s elig=%-5s %6.2f J %6.1f ms  recorded=%-30s B=%s" % (
                index, row["policy"], row["tok"], row["eligible"], row["j_tok"], row["ms_tok"],
                row["recorded_reason"], row["B"]))
            for label, value in row["phone"].items():
                print("       %-4s bounds=%s learn=%s qual=%s more=%s target=%s elim=%s update_elim=%s" % (
                    label, value["bounds"], value["learn"], value["qual"], value["more"], value["target"],
                    value["eliminated"], value["update_elimination"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"request_id": args.request_id, "config": dataclasses.asdict(config),
                       "eliminated_from_events": args.eliminated_from_events, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
