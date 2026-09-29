#!/usr/bin/env python3
"""Per-request phone-vs-host summary of one run and the "measured better, ended on the host" pattern.

Usage: summarize_requests.py RUN_DIR [--json OUT] [--csv OUT]

Measured: window token counts, per-window J/token and ms/token (whole-fleet diagnostic energy including the
ASSUMED_4P5W phone term, i.e. exactly what the controller consumed), controller reasons from
ASSISTANCE_DECISION events. Everything is compared within one active batch (the controller only compares
windows of the session's current batch). Inferred: "left on the table" = host tokens executed after the
last phone window times the same request's measured (host - best phone) J/token at that batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics

from timeline import load, run_groups


def eligible(w):
    return (w.get("measurement_eligible", True) and w["output_valid"]
            and w.get("failure_reason") is None)


def label(policy):
    return "B" if policy["baseline"] else "P%d" % (policy["split_fraction_ppm"] // 10000)


def weighted(rows):
    tokens = sum(w["token_end"] - w["token_start"] for w in rows)
    if not tokens:
        return None
    energy = sum(w["whole_fleet_energy_uj"] for w in rows)
    duration = sum(w["finished_at_us"] - w["started_at_us"] for w in rows)
    return {"windows": len(rows), "tokens": tokens,
            "j_tok": round(energy / tokens / 1e6, 2), "ms_tok": round(duration / tokens / 1e3, 1),
            "median_j_tok": round(statistics.median(w["energy_per_token_uj"] for w in rows) / 1e6, 2)}


def summarize(run_dir):
    obs, result = load(run_dir)
    model_by_artifact = {v["artifact_sha256"]: k for k, v in result["model_artifacts"].items()}
    out_tokens = {r["request_id"]: r["output_tokens"] for r in result["request_results"]}
    rows = []
    for g in sorted(run_groups(obs, result), key=lambda g: g["request_id"]):
        rid = g["request_id"]
        windows = g["windows"]
        tokens_by_label = {}
        by_key = {}
        for w in windows:
            key = (label(w["policy"]), w["active_batch"])
            tokens_by_label[key[0]] = tokens_by_label.get(key[0], 0) + w["token_end"] - w["token_start"]
            if eligible(w):
                by_key.setdefault(key, []).append(w)
        stats = {"%s@b%d" % k: weighted(v) for k, v in sorted(by_key.items())}
        phone_tokens = sum(v for k, v in tokens_by_label.items() if k != "B")
        # Compare at the batch where the phone has the most eligible windows.
        phone_batches = {}
        for (lab, batch), v in by_key.items():
            if lab != "B":
                phone_batches[batch] = phone_batches.get(batch, 0) + len(v)
        batch = max(phone_batches, key=lambda b: (phone_batches[b], -b)) if phone_batches else None
        base = weighted(by_key.get(("B", batch), [])) if batch is not None else None
        phone = {lab: weighted(v) for (lab, b), v in by_key.items() if lab != "B" and b == batch}
        best = min(phone.items(), key=lambda kv: kv[1]["j_tok"]) if phone else None
        final_label = label(g["final_policy"]) if g["final_policy"] else "B"
        last_phone = max((i for i, w in enumerate(windows) if not w["policy"]["baseline"]), default=None)
        host_after = (sum(w["token_end"] - w["token_start"] for w in windows[last_phone + 1:])
                      if last_phone is not None else tokens_by_label.get("B", 0))
        decisions = [e for e in result["request_helper_events"]
                     if e.get("request_id") == rid and e["kind"] == "ASSISTANCE_DECISION"]
        reasons = []
        for e in decisions:
            if e["reason"] and (not reasons or reasons[-1] != e["reason"]):
                reasons.append(e["reason"])
        label_by_hash = {w["policy"]["policy_hash"]: label(w["policy"]) for w in windows}
        eliminated = {}
        for e in decisions:
            for k, v in e["eliminated_policy_reasons"].items():
                eliminated.setdefault(label_by_hash.get(k, k[7:15]), v)
        states = sorted({e["helper_evidence_state"] for e in decisions})
        verification = sorted({(e["verification"] or {}).get("reason") for e in decisions} - {None})
        phone_better = bool(base and best and best[1]["j_tok"] < base["j_tok"] * 0.99
                            and best[1]["ms_tok"] <= base["ms_tok"] * 1.25)
        pattern = phone_better and final_label == "B"
        lost_j = round(host_after * (base["j_tok"] - best[1]["j_tok"]), 1) if pattern else 0.0
        rows.append({
            "request_id": rid,
            "model": model_by_artifact.get(g["model_artifact_sha256"], g["model_artifact_sha256"][7:19]),
            "output_tokens": out_tokens.get(rid),
            "window_tokens": sum(tokens_by_label.values()),
            "phone_tokens": phone_tokens,
            "final": final_label,
            "state_history": g["state_history"],
            "evidence_state": states,
            "compare_batch": batch,
            "stats_by_policy_batch": stats,
            "baseline": base,
            "phone": phone,
            "best_phone": best[0] if best else None,
            "phone_better_measured": phone_better,
            "pattern_better_but_host": pattern,
            "host_tokens_after_last_phone_window": host_after,
            "left_on_table_j_inferred": lost_j,
            "reasons": reasons,
            "eliminated_ever": eliminated,
            "verification_reasons": verification,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json")
    parser.add_argument("--csv")
    args = parser.parse_args()
    rows = summarize(args.run_dir)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["request_id", "model", "output_tokens", "phone_tokens", "final", "compare_batch",
                             "best_phone", "base_j_tok", "base_ms_tok", "base_windows", "phone_j_tok",
                             "phone_ms_tok", "phone_windows", "phone_better_measured",
                             "pattern_better_but_host", "host_tokens_after_last_phone_window",
                             "left_on_table_j_inferred", "reasons", "eliminated_ever"])
            for r in rows:
                b = r["baseline"] or {}
                p = r["phone"].get(r["best_phone"], {}) if r["best_phone"] else {}
                writer.writerow([r["request_id"], r["model"], r["output_tokens"], r["phone_tokens"], r["final"],
                                 r["compare_batch"], r["best_phone"], b.get("j_tok"), b.get("ms_tok"),
                                 b.get("windows"), p.get("j_tok"), p.get("ms_tok"), p.get("windows"),
                                 r["phone_better_measured"], r["pattern_better_but_host"],
                                 r["host_tokens_after_last_phone_window"], r["left_on_table_j_inferred"],
                                 " > ".join(r["reasons"]), json.dumps(r["eliminated_ever"], sort_keys=True)])
    total = sum(r["window_tokens"] for r in rows)
    phone = sum(r["phone_tokens"] for r in rows)
    print("requests %d window tokens %d phone tokens %d (%.1f%%)" % (len(rows), total, phone,
                                                                      100.0 * phone / max(1, total)))
    for r in rows:
        b = r["baseline"] or {}
        p = r["phone"].get(r["best_phone"], {}) if r["best_phone"] else {}
        print("%-26s %-8s out=%-5s phone_tok=%-5s final=%-4s b=%s host=%s/%s(n=%s) best=%s %s/%s(n=%s) "
              "better=%s PATTERN=%s host_after=%s lost=%sJ" % (
                  r["request_id"][-24:], r["model"][:8], r["output_tokens"], r["phone_tokens"], r["final"],
                  r["compare_batch"], b.get("j_tok"), b.get("ms_tok"), b.get("windows"), r["best_phone"],
                  p.get("j_tok"), p.get("ms_tok"), p.get("windows"), r["phone_better_measured"],
                  r["pattern_better_but_host"], r["host_tokens_after_last_phone_window"],
                  r["left_on_table_j_inferred"]))
        print("      stats:", json.dumps({k: (v["j_tok"], v["ms_tok"], v["windows"])
                                          for k, v in r["stats_by_policy_batch"].items() if v}))
        print("      reasons:", " > ".join(r["reasons"]), "| eliminated:", r["eliminated_ever"],
              "| verification:", r["verification_reasons"], "| state:", r["evidence_state"])


if __name__ == "__main__":
    main()
