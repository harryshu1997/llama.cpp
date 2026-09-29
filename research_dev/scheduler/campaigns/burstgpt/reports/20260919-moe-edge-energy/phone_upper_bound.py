#!/usr/bin/env python3
"""Upper bound of the host-energy saving a phone-resident expert tier could deliver.

Inputs are measured: the reference arm (all experts in RAM) and a paging arm (experts do not fit the
host memory scope) from ``moe_energy_gate.py``, and the coverage curve from ``analyze_routing.py``.
The phone side is NOT measured here; it enters as parameters (active power, per-token added latency)
and the bound is deliberately favorable to the phone:

* every selection the tier serves is one the host would otherwise have missed (capped by the
  measured miss fraction), so the paging penalty shrinks by min(coverage / miss_fraction, 1);
* the host does no work for served selections;
* the only costs charged are phone active energy and the host's idle floor during the phone time.

Anything the bound says is "not worth it" is a real negative; anything it says is "worth it" still
needs the phone arm to be built and measured.
"""
import argparse
import json
import pathlib


def decode_summary(result):
    reqs = [r for r in result["requests"] if r["decode"].get("host_j_per_token") is not None]
    if not reqs:
        raise SystemExit(f"no decode energy in {result['arm']}")
    r = reqs[-1]  # last request: warm within the arm's own regime
    n = r["decode"]["tokens"] or 1
    return {
        "arm": result["arm"],
        "host_j_per_token": r["decode"]["host_j_per_token"],
        "cpu_j_per_token": r["decode"]["cpu_j_per_token"],
        "gpu_j_per_token": r["decode"]["gpu_j_per_token"],
        "ms_per_token": r["decode"].get("ms_per_token"),
        "read_bytes_per_token": r.get("decode_read_bytes", r["read_bytes_delta"]) / n,
        "majflt_per_token": r.get("decode_majflt", r["majflt_delta"]) / n,
        "request_read_bytes": r["read_bytes_delta"],
        "prefill_s": r["prefill"]["seconds"],
        "prefill_host_j": (r["prefill"]["cpu_j"] or 0) + (r["prefill"]["gpu_j"] or 0),
        "idle_floor_w": (result["idle"]["cpu_w"] or 0) + (result["idle"]["gpu_w"] or 0),
    }


def curve_coverage(curve, nbytes):
    pts = curve
    if nbytes <= 0:
        return 0.0
    for a, b in zip(pts, pts[1:]):
        if a["bytes"] <= nbytes <= b["bytes"]:
            f = (nbytes - a["bytes"]) / max(b["bytes"] - a["bytes"], 1)
            return a["coverage"] + f * (b["coverage"] - a["coverage"])
    return pts[-1]["coverage"]


def curve_inverse(curve, coverage):
    """Resident bytes at which the frequency-ordered tier first reaches `coverage`."""
    for a, b in zip(curve, curve[1:]):
        if a["coverage"] <= coverage <= b["coverage"]:
            f = (coverage - a["coverage"]) / max(b["coverage"] - a["coverage"], 1e-12)
            return a["bytes"] + f * (b["bytes"] - a["bytes"])
    return curve[-1]["bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--paging", required=True, nargs="+")
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--phone-power-w", type=float, default=4.5)
    ap.add_argument("--phone-ms-per-token", default="50,100")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = decode_summary(json.load(open(args.reference)))
    cov = json.load(open(args.coverage))
    per_token_expert_bytes = cov["per_token_expert_bytes"]
    rows = []
    pagings = []
    curve = cov["curve"]
    for path in args.paging:
        pag = decode_summary(json.load(open(path)))
        pagings.append(pag)
        penalty = pag["host_j_per_token"] - ref["host_j_per_token"]
        miss = min(1.0, pag["read_bytes_per_token"] / per_token_expert_bytes) if per_token_expert_bytes else None
        # The host's own page cache behaves like a frequency-ordered tier: find the resident size on the
        # coverage curve that reproduces the measured miss fraction, then add the phone tier to it.
        host_eff_bytes = curve_inverse(curve, 1.0 - miss) if miss is not None else None
        # Power the host draws while it waits (measured in the paging arm's decode window), charged for the
        # phone's added latency.
        wait_floor_w = pag["host_j_per_token"] / (pag["ms_per_token"] / 1000.0)
        pag["host_effective_cache_bytes"] = host_eff_bytes
        pag["wait_floor_w"] = wait_floor_w
        for tier in cov["tiers"]:
            c = tier["coverage"]
            removed_favorable = min(c / miss, 1.0) if miss and miss > 0 else 0.0
            combined_cov = curve_coverage(curve, (host_eff_bytes or 0) + tier["budget_bytes"])
            miss_combined = 1.0 - combined_cov
            removed_realistic = (miss - miss_combined) / miss if miss and miss > 0 else 0.0
            for ms in [float(x) for x in args.phone_ms_per_token.split(",")]:
                phone_j = args.phone_power_w * ms / 1000.0
                floor_j = wait_floor_w * ms / 1000.0
                row = {
                    "paging_arm": pag["arm"], "tier_gib": tier["tier_gib"], "coverage": c,
                    "miss_fraction": miss, "penalty_j_per_token": penalty,
                    "host_effective_cache_gib": (host_eff_bytes or 0) / 2**30,
                    "combined_coverage": combined_cov, "miss_with_phone_tier": miss_combined,
                    "penalty_removed_favorable": removed_favorable, "penalty_removed_realistic": removed_realistic,
                    "phone_ms_per_token": ms, "phone_j_per_token": phone_j, "host_floor_j_per_token": floor_j,
                    "wait_floor_w": wait_floor_w,
                    "net_saving_favorable_j_per_token": penalty * removed_favorable - phone_j - floor_j,
                    "net_saving_realistic_j_per_token": penalty * removed_realistic - phone_j - floor_j,
                    "paging_host_j_per_token": pag["host_j_per_token"], "reference_host_j_per_token": ref["host_j_per_token"],
                }
                row["net_saving_realistic_fraction"] = row["net_saving_realistic_j_per_token"] / pag["host_j_per_token"]
                row["net_saving_favorable_fraction"] = row["net_saving_favorable_j_per_token"] / pag["host_j_per_token"]
                rows.append(row)
    out = {"reference": ref, "paging": pagings,
           "per_token_expert_bytes": per_token_expert_bytes, "rows": rows,
           "assumptions": {"phone_power_w": args.phone_power_w, "phone_ms_per_token": args.phone_ms_per_token,
                           "favorable": "served selections are exactly the host misses; host does no work for them",
                           "realistic": "host page cache = frequency tier of the size that reproduces the measured miss; phone tier adds to it; host wait floor = paging-arm decode power"}}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"reference {ref['host_j_per_token']:.2f} J/tok @ {ref['ms_per_token']:.0f} ms/tok, idle floor {ref['idle_floor_w']:.1f} W")
    for p in out["paging"]:
        print(f"paging {p['arm']}: {p['host_j_per_token']:.2f} J/tok @ {p['ms_per_token']:.0f} ms/tok, "
              f"{p['read_bytes_per_token']/2**20:.0f} MiB read/tok, {p['majflt_per_token']:.0f} majflt/tok")
    for r in rows:
        print(f"  {r['paging_arm']:>14} tier {r['tier_gib']:>5} GiB: host cache~{r['host_effective_cache_gib']:4.1f} GiB miss {r['miss_fraction']*100:5.1f}% -> "
              f"{r['miss_with_phone_tier']*100:5.1f}% with tier; phone {r['phone_ms_per_token']:>4.0f} ms @ floor {r['wait_floor_w']:4.1f} W: "
              f"realistic {r['net_saving_realistic_j_per_token']:+6.2f} J/tok ({r['net_saving_realistic_fraction']*100:+5.1f}%), "
              f"favorable {r['net_saving_favorable_j_per_token']:+6.2f} J/tok")


if __name__ == "__main__":
    main()
