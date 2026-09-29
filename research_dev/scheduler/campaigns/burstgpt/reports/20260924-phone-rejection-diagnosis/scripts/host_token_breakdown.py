#!/usr/bin/env python3
"""Where did the host tokens of phone-capable requests go, and what would the phone have saved there?

Per request: host/phone tokens per active batch, host tokens after the request's last phone window
("tail"), and whether the phone was measured at that batch in this request. Run-wide medians of eligible
windows per (model, policy, batch) give the inferred per-token saving used for the cost column
(host tail tokens x (host median - P100 median) at that batch; positive only where the phone is cheaper).

Usage: host_token_breakdown.py RUN_DIR [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import statistics

from timeline import load, run_groups


def label(policy):
    return "B" if policy["baseline"] else "P%d" % (policy["split_fraction_ppm"] // 10000)


def eligible(w):
    return w.get("measurement_eligible", True) and w["output_valid"] and not w.get("failure_reason")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json")
    args = parser.parse_args()
    obs, result = load(args.run_dir)
    model_by_artifact = {v["artifact_sha256"]: k for k, v in result["model_artifacts"].items()}
    groups = run_groups(obs, result)
    samples = {}
    for g in groups:
        model = model_by_artifact[g["model_artifact_sha256"]]
        for w in g["windows"]:
            if eligible(w):
                key = (model, label(w["policy"]), w["active_batch"])
                samples.setdefault(key, []).append((w["energy_per_token_uj"] / 1e6, w["latency_per_token_us"] / 1e3))
    medians = {k: (round(statistics.median(x[0] for x in v), 2), round(statistics.median(x[1] for x in v), 1), len(v))
               for k, v in samples.items()}
    rows = []
    for g in sorted(groups, key=lambda g: g["request_id"]):
        model = model_by_artifact[g["model_artifact_sha256"]]
        windows = g["windows"]
        tokens = {}
        for w in windows:
            k = (label(w["policy"]) == "B", w["active_batch"])
            tokens[k] = tokens.get(k, 0) + w["token_end"] - w["token_start"]
        last_phone = max((i for i, w in enumerate(windows) if not w["policy"]["baseline"]), default=None)
        tail = {}
        for w in (windows[last_phone + 1:] if last_phone is not None else windows):
            tail[w["active_batch"]] = tail.get(w["active_batch"], 0) + w["token_end"] - w["token_start"]
        measured_phone_batches = sorted({w["active_batch"] for w in windows
                                         if not w["policy"]["baseline"] and eligible(w)})
        cost = 0.0
        for batch, n in tail.items():
            host = medians.get((model, "B", batch))
            phone = medians.get((model, "P100", batch))
            if host and phone and host[0] > phone[0]:
                cost += n * (host[0] - phone[0])
        rows.append({
            "request_id": g["request_id"], "model": model,
            "final": label(g["final_policy"]) if g["final_policy"] else "B",
            "host_tokens_by_batch": {str(b): n for (host, b), n in sorted(tokens.items()) if host},
            "phone_tokens_by_batch": {str(b): n for (host, b), n in sorted(tokens.items()) if not host},
            "host_tail_tokens_by_batch": {str(b): n for b, n in sorted(tail.items())},
            "phone_measured_at_batches": measured_phone_batches,
            "tail_cost_j_inferred_vs_p100_median": round(cost, 1),
        })
    print("run-wide medians of eligible windows: (model, policy, batch) -> (J/tok, ms/tok, n)")
    for k in sorted(medians):
        print("  ", k, medians[k])
    total_host = sum(sum(r["host_tokens_by_batch"].values()) for r in rows)
    total_phone = sum(sum(r["phone_tokens_by_batch"].values()) for r in rows)
    print("host tokens %d phone tokens %d" % (total_host, total_phone))
    for r in rows:
        print("%-28s %-28s final=%-4s host=%s phone=%s tail=%s measured_at=%s tail_cost=%.0f J" % (
            r["request_id"][-28:], r["model"], r["final"], r["host_tokens_by_batch"], r["phone_tokens_by_batch"],
            r["host_tail_tokens_by_batch"], r["phone_measured_at_batches"], r["tail_cost_j_inferred_vs_p100_median"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"medians": {"%s|%s|b%d" % k: v for k, v in medians.items()}, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
