#!/usr/bin/env python3
"""How noisy is one 4-token window? Deviation of each eligible window from the median of the same
(request, policy, active batch), plus the first eligible host window of each request (the one-window
reference the LEARNING pair gate and the first qualification use) against that request's later host
windows at the same batch.

Usage: window_noise.py RUN_DIR [RUN_DIR ...] [--json OUT]
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


def analyze(run_dir):
    obs, result = load(run_dir)
    deviations = []
    first_host = []
    for g in run_groups(obs, result):
        by_key = {}
        for w in g["windows"]:
            if eligible(w):
                by_key.setdefault((label(w["policy"]), w["active_batch"]), []).append(w)
        for key, rows in by_key.items():
            if len(rows) < 3:
                continue
            med_j = statistics.median(w["energy_per_token_uj"] for w in rows)
            med_ms = statistics.median(w["latency_per_token_us"] for w in rows)
            for w in rows:
                deviations.append({
                    "request_id": g["request_id"], "key": "%s@b%d" % key, "window": w["window_index"],
                    "j_dev": w["energy_per_token_uj"] / med_j - 1, "ms_dev": w["latency_per_token_us"] / med_ms - 1})
        host = [w for w in g["windows"] if eligible(w) and w["policy"]["baseline"]]
        if host:
            first = host[0]
            later = [w for w in host[1:] if w["active_batch"] == first["active_batch"]]
            if len(later) >= 3:
                med_j = statistics.median(w["energy_per_token_uj"] for w in later)
                med_ms = statistics.median(w["latency_per_token_us"] for w in later)
                first_host.append({
                    "request_id": g["request_id"], "window": first["window_index"], "batch": first["active_batch"],
                    "first_j_tok": round(first["energy_per_token_uj"] / 1e6, 2),
                    "later_median_j_tok": round(med_j / 1e6, 2),
                    "j_dev": round(first["energy_per_token_uj"] / med_j - 1, 3),
                    "first_ms_tok": round(first["latency_per_token_us"] / 1e3, 1),
                    "later_median_ms_tok": round(med_ms / 1e3, 1),
                    "ms_dev": round(first["latency_per_token_us"] / med_ms - 1, 3),
                    "later_windows": len(later)})
    n = len(deviations)
    summary = {
        "run_dir": run_dir,
        "eligible_windows_in_groups_with_3plus": n,
        "energy_dev_gt_10pct": sum(abs(d["j_dev"]) > 0.10 for d in deviations),
        "energy_dev_gt_20pct": sum(abs(d["j_dev"]) > 0.20 for d in deviations),
        "latency_dev_gt_10pct": sum(abs(d["ms_dev"]) > 0.10 for d in deviations),
        "latency_dev_gt_20pct": sum(abs(d["ms_dev"]) > 0.20 for d in deviations),
        "energy_dev_p50_abs": round(statistics.median(abs(d["j_dev"]) for d in deviations), 4) if n else None,
        "first_host_windows": first_host,
        "first_host_abs_dev_gt_10pct": sum(abs(r["j_dev"]) > 0.10 for r in first_host),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--json")
    args = parser.parse_args()
    out = [analyze(d) for d in args.run_dirs]
    for s in out:
        print(s["run_dir"])
        print("  eligible windows (groups with >=3):", s["eligible_windows_in_groups_with_3plus"],
              "| |dJ|>10%:", s["energy_dev_gt_10pct"], "|dJ|>20%:", s["energy_dev_gt_20pct"],
              "| |dt|>10%:", s["latency_dev_gt_10pct"], "|dt|>20%:", s["latency_dev_gt_20pct"],
              "| median |dJ|:", s["energy_dev_p50_abs"])
        print("  first eligible host window vs later host median (same batch):",
              s["first_host_abs_dev_gt_10pct"], "of", len(s["first_host_windows"]), "deviate >10%")
        for r in s["first_host_windows"]:
            print("    %-28s W%-3d b%d first %6.2f J %6.1f ms | later median %6.2f J %6.1f ms (n=%d) | dJ %+.1f%% dt %+.1f%%" % (
                r["request_id"][-28:], r["window"], r["batch"], r["first_j_tok"], r["first_ms_tok"],
                r["later_median_j_tok"], r["later_median_ms_tok"], r["later_windows"],
                100 * r["j_dev"], 100 * r["ms_dev"]))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
