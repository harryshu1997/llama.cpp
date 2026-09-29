#!/usr/bin/env python3
"""Which completed groups seeded a request's operational verification winner?

Loads the run's final observation store minus the request's own group (and optionally minus other
groups) into a fresh controller, rebuilds the request's session identity at start, and calls
_cached_verification_policy / _operational_verification_policy with the run-time source.

Usage: PYTHONDONTWRITEBYTECODE=1 replay_seeding.py --source RUN_SOURCE/research_dev RUN_DIR REQUEST_ID
           [--drop REQUEST_ID ...]
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
    parser.add_argument("--drop", nargs="*", default=[])
    args = parser.parse_args()
    sys.path.insert(0, os.path.abspath(args.source))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from scheduler._internal.adaptive_decode import AdaptiveDecodeController
    from scheduler._internal.adaptive_decode_contracts import (
        AdaptiveDecodeConfig, AdaptiveDecodeGroupedObservation, AdaptiveDecodeWindowReceipt)
    from scheduler._internal.adaptive_decode_state import _AdaptiveSession
    from timeline import load, run_groups

    obs, result = load(args.run_dir)
    target = next(g for g in run_groups(obs, result) if g["request_id"] == args.request_id)
    dropped = set(args.drop) | {args.request_id}
    # Only what the controller could have held at the request's start: the input store plus this run's
    # groups whose last window finished before the request's first window started.
    this_run = {g["ticket_id"] for g in run_groups(obs, result)}
    start_us = target["windows"][0]["started_at_us"]
    controller = AdaptiveDecodeController()
    kept = []
    for raw in obs["groups"]:
        if raw["request_id"] in dropped:
            continue
        if raw["ticket_id"] in this_run and max(w["finished_at_us"] for w in raw["windows"]) > start_us:
            continue
        grouped = AdaptiveDecodeGroupedObservation.from_json(raw)
        controller._history[grouped.grouped_observation_sha256] = grouped
        kept.append(raw["request_id"])
    decisions = [e for e in result["request_helper_events"]
                 if e.get("request_id") == args.request_id and e["kind"] == "ASSISTANCE_DECISION"]
    first = decisions[0]
    base_config = dict(result["adaptive_controller_configuration"])
    base_config["coarse_probe_fractions_ppm"] = tuple(base_config["coarse_probe_fractions_ppm"])
    base_config["refinement_steps_ppm"] = tuple(base_config["refinement_steps_ppm"])
    config = dataclasses.replace(
        AdaptiveDecodeConfig(**base_config),
        minimum_energy_saving_ppm=first["minimum_energy_saving_ppm"],
        maximum_latency_ppm=first["maximum_latency_ppm"],
        allow_assumed_phone_power_for_operational_selection=True)
    receipts = [AdaptiveDecodeWindowReceipt.from_json(w) for w in target["windows"]]
    baseline = next(r.policy for r in receipts if r.policy.baseline)
    phone = []
    for r in receipts:
        if not r.policy.baseline and r.policy not in phone:
            phone.append(r.policy)
    ticket = next(r for r in result["request_results"] if r["request_id"] == args.request_id)
    session = _AdaptiveSession(
        request_id=target["request_id"], ticket_id=target["ticket_id"],
        model_artifact_sha256=target["model_artifact_sha256"],
        planning_profile_sha256=target["planning_profile_sha256"],
        component_capability_sha256=target["planning_profile_sha256"],
        baseline=baseline, candidates=tuple(phone), output_tokens=ticket["output_tokens"],
        context_length=receipts[0].context_length - receipts[0].token_start,
        active_batch=receipts[0].active_batch,
        deadline_us=ticket["terminal_ticket"]["request"]["deadline_us"], config=config,
        slot_id=receipts[0].slot_id, helper_evidence_state=first["helper_evidence_state"],
        helper_layout_generation=first["helper_layout_generation"],
        helper_layout_geometry_sha256=first["helper_layout_geometry_sha256"])
    matching = [g.request_id for g in controller._history.values() if controller._group_matches_session(g, session)]
    print("history groups available at start:", len(kept), "| matching this session:", matching)
    for policy in (baseline, *phone):
        rows, groups = controller._historical_policy_records(session, policy, compatible_context=True)
        print("  historical records for %s: windows=%d energy-eligible groups=%d" % (
            "B" if policy.baseline else "P%d" % (policy.split_fraction_ppm // 10000), len(rows), groups))
    cached = controller._cached_verification_policy(session)
    operational = controller._operational_verification_policy(session)
    name = lambda p: None if p is None else ("B" if p.baseline else "P%d" % (p.split_fraction_ppm // 10000))
    print("cached (energy-certified) winner:", name(cached), "| operational (ASSUMED power) winner:", name(operational))


if __name__ == "__main__":
    main()
