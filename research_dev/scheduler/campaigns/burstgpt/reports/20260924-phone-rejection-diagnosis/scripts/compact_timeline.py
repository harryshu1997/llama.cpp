#!/usr/bin/env python3
"""Compact per-request run-length view: policy@batch segments with token counts and eligible J/token,
interleaved with the controller reasons that changed the policy.

Usage: compact_timeline.py RUN_DIR [REQUEST_ID ...]
"""

from __future__ import annotations

import argparse

from timeline import load, run_groups


def label(policy):
    return "B" if policy["baseline"] else "P%d" % (policy["split_fraction_ppm"] // 10000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("request_ids", nargs="*")
    args = parser.parse_args()
    obs, result = load(args.run_dir)
    for g in sorted(run_groups(obs, result), key=lambda g: g["request_id"]):
        rid = g["request_id"]
        if args.request_ids and rid not in args.request_ids:
            continue
        hashes = {w["policy"]["policy_hash"]: label(w["policy"]) for w in g["windows"]}
        decisions = [e for e in result["request_helper_events"]
                     if e.get("request_id") == rid and e["kind"] == "ASSISTANCE_DECISION"]
        by_token = {}
        for e in decisions:
            by_token.setdefault(e["token_index"], []).append(e)
        print("=" * 110)
        final = label(g["final_policy"]) if g["final_policy"] else "B"
        print(rid, "final", final, g["state_history"])
        segments = []
        for w in g["windows"]:
            key = (label(w["policy"]), w["active_batch"])
            elig = w.get("measurement_eligible", True) and w["output_valid"] and not w.get("failure_reason")
            tokens = w["token_end"] - w["token_start"]
            if segments and segments[-1]["key"] == key:
                s = segments[-1]
            else:
                s = {"key": key, "tok0": w["token_start"], "t0": w["started_at_us"] / 1e6, "tokens": 0,
                     "elig": [], "reasons": []}
                segments.append(s)
            s["tokens"] += tokens
            s["tok1"] = w["token_end"]
            s["t1"] = w["finished_at_us"] / 1e6
            if elig:
                s["elig"].append((round(w["energy_per_token_uj"] / 1e6, 1), round(w["latency_per_token_us"] / 1e3)))
            for e in by_token.get(w["token_end"], ()):
                reason = e["reason"] or "-"
                if reason in {"WINDOW_OPENED", "INITIAL_BASELINE"}:
                    continue
                elim = {hashes.get(k, k[7:15]): v for k, v in e["eliminated_policy_reasons"].items()}
                text = reason + ("" if not elim else " elim=" + ",".join("%s:%s" % kv for kv in sorted(elim.items())))
                if not s["reasons"] or s["reasons"][-1] != text:
                    s["reasons"].append(text)
        for s in segments:
            lab, batch = s["key"]
            elig = s["elig"]
            print("  %-5s b%d tok[%4d,%4d) %4d tok  t=%7.1f-%7.1f  eligible J/tok,ms: %s" % (
                lab, batch, s["tok0"], s["tok1"], s["tokens"], s["t0"], s["t1"],
                " ".join("%s/%s" % x for x in elig[:6]) + (" ... (%d)" % len(elig) if len(elig) > 6 else "")))
            for r in s["reasons"]:
                print("        -> " + r)


if __name__ == "__main__":
    main()
