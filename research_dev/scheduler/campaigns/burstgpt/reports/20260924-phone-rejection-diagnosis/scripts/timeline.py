#!/usr/bin/env python3
"""Per-request adaptive decode timeline: windows + ASSISTANCE_DECISION events.

Usage: timeline.py RUN_DIR [REQUEST_ID ...] [--json OUT]
RUN_DIR holds ADAPTIVE_DECODE_OBSERVATIONS.json and RESULT.json of one run.
"""

from __future__ import annotations

import argparse
import json
import os


def load(run_dir):
    with open(os.path.join(run_dir, "ADAPTIVE_DECODE_OBSERVATIONS.json")) as f:
        obs = json.load(f)
    with open(os.path.join(run_dir, "RESULT.json")) as f:
        result = json.load(f)
    result["request_helper_events"] = helper_events(run_dir, result)
    return obs, result


def helper_events(run_dir, result):
    """RESULT keeps only the last 4096 helper events; COMPLETED decision-log records keep each request's."""
    events = {e["event_sha256"]: e for e in result["request_helper_events"]}
    path = os.path.join(run_dir, "SCHEDULER_DECISION_LOG.json")
    if os.path.exists(path):
        with open(path) as f:
            log = json.load(f)
        for record in log["records"]:
            for e in (record.get("selected") or {}).get("request_helper_events") or ():
                events.setdefault(e["event_sha256"], e)
    return sorted(events.values(), key=lambda e: (e.get("event_index", 0), e.get("observed_at_us", 0)))


def run_groups(obs, result):
    """Groups written by this run (ticket ids of this run's request results)."""
    tickets = set()
    for row in result["request_results"]:
        tickets.update(row.get("attempt_ticket_ids") or ())
        term = row.get("terminal_ticket") or {}
        if isinstance(term, dict) and term.get("ticket_id"):
            tickets.add(term["ticket_id"])
    return [g for g in obs["groups"] if g["ticket_id"] in tickets]


def policy_label(policy, hashes):
    if policy["baseline"]:
        return "B"
    label = "P%d" % (policy["split_fraction_ppm"] // 10000)
    hashes.setdefault(policy["policy_hash"], label)
    return label


def window_rows(group, hashes):
    rows = []
    for w in group["windows"]:
        tokens = w["token_end"] - w["token_start"]
        rows.append({
            "window": w["window_index"],
            "tok": [w["token_start"], w["token_end"]],
            "policy": policy_label(w["policy"], hashes),
            "policy_hash": w["policy"]["policy_hash"],
            "layers": len(w["policy"]["layer_indices"]),
            "eligible": w.get("measurement_eligible", True),
            "role": w.get("window_role"),
            "batch": w["active_batch"],
            "next_batch": w.get("next_active_batch"),
            "lat_ms_tok": round(w["latency_per_token_us"] / 1000, 1),
            "j_tok": round(w["energy_per_token_uj"] / 1e6, 2),
            "start_s": round(w["started_at_us"] / 1e6, 2),
            "end_s": round(w["finished_at_us"] / 1e6, 2),
            "transition": "physical:control-transition-ack" in w["evidence_ids"],
            "phone_calls": w.get("completed_phone_calls"),
            "failure": w.get("failure_reason"),
            "external": (w.get("external_activity_sha256") or "")[7:15] or None,
            "external_changed": w.get("external_activity_changed", False),
            "context_available": w.get("execution_context_available", True),
            "tokens": tokens,
        })
    return rows


def decision_rows(result, request_id, hashes):
    rows = []
    for e in result["request_helper_events"]:
        if e.get("request_id") != request_id:
            continue
        kind = e["kind"]
        if kind == "ASSISTANCE_DECISION":
            ver = e.get("verification") or {}
            ev = e.get("evidence") or {}
            evidence = {
                hashes.get(k, k[7:15]): (v["current_valid_windows"], v["historical_groups"])
                for k, v in ev.items()
                if v["current_valid_windows"] or v["historical_groups"]
            }
            rows.append({
                "t_s": round(e["observed_at_us"] / 1e6, 2),
                "kind": kind,
                "tok": e["token_index"],
                "phase": e["phase"],
                "reason": e["reason"],
                "selected": e["selected_fraction_ppm"] // 10000,
                "role": e["window_role"],
                "incumbent": hashes.get(e["incumbent_policy_hash"], e["incumbent_policy_hash"]),
                "challenger": hashes.get(e["challenger_policy_hash"], e["challenger_policy_hash"]),
                "eliminated": {hashes.get(k, k[7:15]): v for k, v in e["eliminated_policy_reasons"].items()},
                "evidence(cur,hist)": evidence,
                "verification": {k: ver.get(k) for k in ("attempts", "outcome", "reason")}
                if ver.get("outcome") or ver.get("attempts") else None,
                "state": e["helper_evidence_state"],
                "min_saving_ppm": e["minimum_energy_saving_ppm"],
                "max_latency_ppm": e["maximum_latency_ppm"],
                "remaining": e["remaining_output_tokens"],
                "remaining_probe_tokens": e["remaining_probe_tokens"],
                "probe_budget": e.get("probe_budget"),
                "prior": e.get("context_monitor_prior"),
            })
        elif kind in {
            "PROBE_CANDIDATE_REJECTED", "PROBE_INCOMPLETE", "INSUFFICIENT_OPPORTUNITY",
            "VERIFICATION_RESERVED", "VERIFICATION_VERIFIED", "VERIFICATION_REJECTED",
            "VERIFICATION_INCOMPLETE", "HELPER_RETAINED_FOR_READY_LAYOUT", "HELPER_EXPANDED",
            "ATTACHED", "DETACHED", "ELIGIBLE", "FRACTION_APPLIED",
        }:
            rows.append({
                "t_s": round(e["observed_at_us"] / 1e6, 2),
                "kind": kind,
                "tok": e.get("token_index"),
                "reason": e.get("reason"),
                "eliminated": {hashes.get(k, k[7:15]): v for k, v in (e.get("eliminated_policy_reasons") or {}).items()},
                "fraction": e.get("selected_fraction_ppm"),
                "state": e.get("evidence_state"),
                "layout_generation": e.get("phone_layout_generation"),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("request_ids", nargs="*")
    parser.add_argument("--json")
    args = parser.parse_args()
    obs, result = load(args.run_dir)
    groups = run_groups(obs, result)
    out = {}
    for g in sorted(groups, key=lambda g: g["request_id"]):
        if args.request_ids and g["request_id"] not in args.request_ids:
            continue
        hashes = {}
        windows = window_rows(g, hashes)
        decisions = decision_rows(result, g["request_id"], hashes)
        out[g["request_id"]] = {
            "ticket_id": g["ticket_id"],
            "state_history": g["state_history"],
            "final_policy": policy_label(g["final_policy"], hashes) if g["final_policy"] else None,
            "policies": hashes,
            "windows": windows,
            "decisions": decisions,
        }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)
    for rid, data in out.items():
        print("=" * 100)
        print(rid, data["ticket_id"], data["state_history"], "final", data["final_policy"])
        print("policies", data["policies"])
        for w in data["windows"]:
            print("  W%-3d %-4s tok=%-10s elig=%-5s role=%-12s b=%s lat=%7.1f J=%6.2f t=%.2f-%.2f%s%s%s" % (
                w["window"], w["policy"], w["tok"], w["eligible"], w["role"], w["batch"],
                w["lat_ms_tok"], w["j_tok"], w["start_s"], w["end_s"],
                " TRANSITION" if w["transition"] else "",
                " FAIL=" + str(w["failure"]) if w["failure"] else "",
                " EXT_CHANGED" if w["external_changed"] else ""))
        for d in data["decisions"]:
            print("  D", json.dumps({k: v for k, v in d.items() if v not in (None, {}, [])}))


if __name__ == "__main__":
    main()
