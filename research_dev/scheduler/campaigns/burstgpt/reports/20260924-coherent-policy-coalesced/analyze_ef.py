#!/usr/bin/env python3
"""Accounting for the post-evidence-fix dev_v2 A/B (plainEF vs coherentEF, against dev2base and the earlier arms).

    python3 analyze_ef.py --arm base=<run> --arm plainEF=<run> --arm coherentEF=<run> [...] --out X.json --md X.md

Extends analyze_coherent_arm.py (energy, server counters, window tables, decisions) with:
- window J/token and latency by batch composition from the receipts' own energy_per_token_uj (whole fleet,
  assumed phone power included) and latency_per_token_us, token-weighted, all valid windows and eligible ones;
- overlap: per model, decode tokens at active_batch >= 2 and pairwise wall-clock overlap of the requests'
  window spans (RESULT active_slots_peak misses adaptive requests);
- evidence-fix markers: CANDIDATE_REQUALIFIED decisions, windows' measurement_ineligible_reason
  (TOKEN_STREAM_CATCH_UP, HELPER_PHONE_SESSION_LOAD, ...), raw occurrences in the run's JSON artifacts;
- lock-out check: helper PREPARATION_FAILED and phone SESSION_FAILED/TRANSITION_FAILED counts by model.
Reads artifacts only.
"""
import argparse
import collections
import itertools
import json
import pathlib

import analyze_coherent_arm as base

MARKERS = ("CANDIDATE_REQUALIFIED", "TOKEN_STREAM_CATCH_UP", "HELPER_PHONE_SESSION_LOAD",
           "measurement_ineligible_reason")
RAW_FILES = ("RESULT.json", "ADAPTIVE_DECODE_OBSERVATIONS.json", "adaptive-timing-events.json",
             "SCHEDULER_DECISION_LOG.json")


def valid(w):
    return w.get("output_valid") and w.get("failure_reason") is None


def eligible(w):
    return w.get("measurement_eligible", True) is not False


def window_detail(run, result):
    store = run / "ADAPTIVE_DECODE_OBSERVATIONS.json"
    if not store.exists() or result is None:
        return {}
    model_of = {row["request_id"]: base.model_key(row["model_id"]) for row in result["request_results"]}
    acc = collections.defaultdict(lambda: collections.Counter())
    spans = collections.defaultdict(list)
    ineligible = collections.Counter()
    batch_tokens = collections.Counter()
    for group in json.loads(store.read_text())["groups"]:
        model = model_of.get(group["request_id"])
        if model is None:
            continue
        for w in group["windows"]:
            tokens = w["token_end"] - w["token_start"]
            policy = "host" if w["policy"]["baseline"] else "phone"
            batch = w.get("active_batch") or 1
            spans[(model, group["request_id"])].append((w["started_at_us"], w["finished_at_us"]))
            batch_tokens[(model, policy, batch)] += tokens
            reason = w.get("measurement_ineligible_reason")
            if reason:
                ineligible[(model, policy, reason)] += 1
            if not valid(w):
                continue
            for scope in ("all", "eligible"):
                if scope == "eligible" and not eligible(w):
                    continue
                a = acc[(model, policy, batch, scope)]
                a["windows"] += 1
                a["tokens"] += tokens
                a["energy_uj_x_tokens"] += w["energy_per_token_uj"] * tokens
                a["latency_us_x_tokens"] += w["latency_per_token_us"] * tokens
                dom = w.get("fleet_energy_uj_by_domain") or {}
                a["host_uj"] += dom.get("cpu-package", 0) + dom.get("gpu-board", 0)
                a["phone_calls"] += w.get("completed_phone_calls") or 0
                a["phone_rows"] += w.get("completed_phone_input_rows") or 0
    table = []
    for (model, policy, batch, scope), a in sorted(acc.items()):
        t = a["tokens"]
        table.append({"model": model, "policy": policy, "active_batch": batch, "scope": scope,
                      "windows": a["windows"], "tokens": t,
                      "fleet_j_per_slot_token": round(a["energy_uj_x_tokens"] / t / 1e6, 2),
                      "fleet_j_per_produced_token": round(a["energy_uj_x_tokens"] / t / 1e6 / batch, 2),
                      "host_j_per_slot_token": round(a["host_uj"] / t / 1e6, 2),
                      "host_j_per_produced_token": round(a["host_uj"] / t / 1e6 / batch, 2),
                      "ms_per_slot_token": round(a["latency_us_x_tokens"] / t / 1e3, 1),
                      "phone_calls": a["phone_calls"], "phone_rows": a["phone_rows"],
                      "rows_per_call": round(a["phone_rows"] / a["phone_calls"], 2) if a["phone_calls"] else None})
    # Pairwise overlap of same-model requests' window spans (first window start .. last window end).
    overlap = []
    by_model = collections.defaultdict(list)
    for (model, rid), rows in spans.items():
        by_model[model].append((rid, min(s for s, _ in rows), max(e for _, e in rows)))
    for model, rows in sorted(by_model.items()):
        for (a, sa, ea), (b, sb, eb) in itertools.combinations(sorted(rows, key=lambda r: r[1]), 2):
            seconds = (min(ea, eb) - max(sa, sb)) / 1e6
            if seconds > 0:
                overlap.append({"model": model, "a": a.split(":")[-1], "b": b.split(":")[-1],
                                "overlap_s": round(seconds, 1)})
    tokens_by_batch = {}
    for (model, policy, batch), t in sorted(batch_tokens.items()):
        tokens_by_batch.setdefault(model, {}).setdefault(policy, {})[str(batch)] = t
    return {"by_model_policy_batch_scope": table, "window_tokens_by_model_policy_batch": tokens_by_batch,
            "decode_span_overlaps": overlap,
            "measurement_ineligible_reasons": {"|".join(k): v for k, v in sorted(ineligible.items())}}


def markers(run, result):
    raw = {}
    for name in RAW_FILES:
        path = run / name
        if path.exists():
            text = path.read_text(errors="replace")
            raw[name] = {marker: text.count(marker) for marker in MARKERS}
    requalified = collections.Counter()
    if result is not None:
        model_of = {row["request_id"]: base.model_key(row["model_id"]) for row in result["request_results"]}
        for event in result.get("request_helper_events") or []:
            if event.get("kind") == "ASSISTANCE_DECISION":
                for candidate in base._dicts(event):
                    if candidate.get("reason") == "CANDIDATE_REQUALIFIED":
                        requalified[model_of.get(event.get("request_id"), "?")] += 1
                        break
    return {"raw_occurrences": raw, "candidate_requalified_decisions_by_model": dict(requalified)}


def lockout(result):
    if result is None:
        return {}
    model_of = {row["request_id"]: base.model_key(row["model_id"]) for row in result["request_results"]}
    helper = collections.Counter()
    for event in result.get("request_helper_events") or []:
        if event.get("kind") in ("PREPARATION_FAILED", "PREPARATION_READY", "ATTACHED", "DETACHED"):
            helper[(model_of.get(event.get("request_id"), "?"), event["kind"])] += 1
    residency = collections.Counter(e.get("kind") for e in result.get("phone_residency_events") or []
                                    if e.get("kind") in ("SESSION_FAILED", "TRANSITION_FAILED", "SESSION_READY",
                                                         "READY", "PROPOSED"))
    return {"helper_events_by_model": {"|".join(k): v for k, v in sorted(helper.items())},
            "phone_residency_events": dict(sorted(residency.items()))}


def request_intervals(result):
    if result is None:
        return []
    rows = []
    for row in result["request_results"]:
        acquired = [r.get("observed_at_us") for r in row.get("dispatch_receipts") or [] if r.get("status") == "ACQUIRED"]
        rows.append({"request": row["request_id"].split(":")[-1], "model": base.model_key(row["model_id"]),
                     "output_tokens": row["output_tokens"], "arrival_s": round(row["replay_arrival_us"] / 1e6, 1),
                     "acquired_s": round(min(acquired) / 1e6, 1) if acquired else None,
                     "end_s": round((row.get("completion") or {}).get("actual_end_us", 0) / 1e6, 1),
                     "phone_calls": (row.get("physical_execution_proof") or {}).get("phone_call_count", 0)})
    return sorted(rows, key=lambda r: r["request"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, help="label=run dir")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    report = {}
    for spec in args.arm:
        label, _, path = spec.partition("=")
        run = pathlib.Path(path)
        result = base.load_result(run)
        report[label] = {
            "run": str(run), "failure": (json.loads((run / "FAILURE.json").read_text())
                                         if (run / "FAILURE.json").exists() else None),
            "energy": base.energy(result) if result else None,
            "server_logs": base.server_logs(run),
            "windows": base.windows(run, result),
            "window_detail": window_detail(run, result),
            "decisions": base.decisions(run, result),
            "markers": markers(run, result),
            "lockout": lockout(result),
            "request_intervals": request_intervals(result),
            "requests": base.per_request(run, result),
        }
    labels = list(report)
    for label in labels[1:]:
        for ref in labels[:labels.index(label)]:
            e, b = report[label]["energy"], report[ref]["energy"]
            if e and b:
                e.setdefault("vs", {})[ref] = {
                    "host_kj_percent": round(100 * (e["host_kj"] / b["host_kj"] - 1), 2),
                    "duration_percent": round(100 * (e["duration_s"] / b["duration_s"] - 1), 2)}
    # arm order is the command-line order (deltas are against earlier arms)
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
