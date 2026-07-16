#!/usr/bin/env python3
"""S10-V0 CP4: run C0..C5 over the frozen primary + sweep instances, validate EVERY
certificate with the INDEPENDENT checker, compute C4/C5 vs C1 relief, and evaluate
the opportunity / causal / mechanism gates. Deterministic. Writes artifacts.

Gates (from PLAN.md / NEXT_PLAN.md):
  opportunity: C4 >= 15% total-wall-energy relief vs C1 in TWO ADJACENT declared
               load bins, no worse SLO.
  causal:      C5 >= 10% vs C1 in the same bins AND C5 retains >= 2/3 of C4's abs gain.
  mechanism:   C5 timely phone islands > C2 AND a measured larger server batch / lower
               cap / break-even low-power interval explains the benefit.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SP, "oracle"))
sys.path.insert(0, os.path.join(SP, "policies"))
sys.path.insert(0, os.path.join(SP, "checker"))
import policies as POL          # noqa: E402
import checker as CHK           # noqa: E402
import model_data as M          # noqa: E402

ART = os.path.join(SP, "artifacts")
FROZEN = os.path.join(SP, "fixtures", "frozen")
SWEEP = os.path.join(FROZEN, "sweep")


def eval_instance(path):
    inst = M.load_instance(path)
    rows = {}
    for nm in ("C0", "C1", "C2", "C3", "C4", "C5"):
        cert, sim = POL.run_policy(inst, nm)
        fails = CHK.check(inst, cert)   # independent validation of the policy's own cert
        if fails:
            raise AssertionError(f"{inst['instance_id']}/{nm} cert FAILED checker: {fails[:2]}")
        rows[nm] = {"energy_nj": cert["energy_nj"], "met": cert["met"], "tardy": cert["tardy"],
                    "timeout": cert["timeout"],
                    "timely_phone": POL.timely_phone_islands(cert),
                    "server_batches": len(cert["server_batches"]),
                    "max_batch_tokens": max([b["tokens"] for b in cert["server_batches"]], default=0),
                    "offload": sum(len(v) for v in cert["phone_islands"].values())}
    c1 = rows["C1"]["energy_nj"]
    for nm in rows:
        rows[nm]["rel_vs_C1_pct"] = 100.0 * (c1 - rows[nm]["energy_nj"]) / c1 if c1 else 0.0
    gainC4 = c1 - rows["C4"]["energy_nj"]
    gainC5 = c1 - rows["C5"]["energy_nj"]
    return {
        "instance": inst["instance_id"],
        "params": inst["cp1_params"],
        "power_model": inst["power_model"]["name"],
        "rows": rows,
        "C4_rel_pct": rows["C4"]["rel_vs_C1_pct"],
        "C5_rel_pct": rows["C5"]["rel_vs_C1_pct"],
        "C5_retains_frac_of_C4": (gainC5 / gainC4) if gainC4 > 0 else (1.0 if gainC5 >= 0 else 0.0),
        "C5_timely_gt_C2": rows["C5"]["timely_phone"] > rows["C2"]["timely_phone"],
        "SLO_no_worse": rows["C4"]["met"] >= rows["C1"]["met"] and rows["C5"]["met"] >= rows["C1"]["met"],
        # did any C4 offload create a LARGER server batch than C1? (mechanism premise)
        "C4_max_batch": rows["C4"]["max_batch_tokens"], "C1_max_batch": rows["C1"]["max_batch_tokens"],
        "C4_makes_larger_batch": rows["C4"]["max_batch_tokens"] > rows["C1"]["max_batch_tokens"],
    }


def main():
    results = []
    # primary
    for tag in ("favorable", "conservative"):
        results.append(eval_instance(os.path.join(FROZEN, f"primary_{tag}.json")))
    # sweep
    for fn in sorted(os.listdir(SWEEP)):
        if fn.endswith(".json"):
            results.append(eval_instance(os.path.join(SWEEP, fn)))

    with open(os.path.join(ART, "cp4_results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    # ---- gate evaluation ----
    # group by (power_model, mix) and look at adjacent load bins
    def key(r):
        p = r["params"]
        return (r["power_model"], "".join(sorted({w for w, _ in p["weight_mix"]})) if False else None)

    # organize sweep rows by (power_model, mixname, slack) -> {load: C4_rel, C5_rel, ...}
    grid = {}
    for r in results:
        if not r["instance"].startswith("sweep_"):
            continue
        parts = r["instance"].split("_")
        # sweep_<pm>_<mix>_load<L>_<slack>
        pm = parts[1]; mix = parts[2]; load = int(parts[3].replace("load", "")); slack = parts[4]
        grid.setdefault((pm, mix, slack), {})[load] = r

    opportunity_hits = []   # (pm,mix,slack, load_pair) where BOTH adjacent bins >=15% and SLO no worse
    for (pm, mix, slack), bins in grid.items():
        loads = sorted(bins)
        for i in range(len(loads) - 1):
            a, b = loads[i], loads[i + 1]
            ra, rb = bins[a], bins[b]
            if (ra["C4_rel_pct"] >= 15.0 and rb["C4_rel_pct"] >= 15.0
                    and ra["SLO_no_worse"] and rb["SLO_no_worse"]):
                opportunity_hits.append({
                    "power_model": pm, "mix": mix, "slack": slack, "load_pair": [a, b],
                    "C4_rel": [round(ra["C4_rel_pct"], 1), round(rb["C4_rel_pct"], 1)],
                    "C5_rel": [round(ra["C5_rel_pct"], 1), round(rb["C5_rel_pct"], 1)],
                    "C5_retain": [round(ra["C5_retains_frac_of_C4"], 2), round(rb["C5_retains_frac_of_C4"], 2)],
                    "C4_makes_larger_batch": [ra["C4_makes_larger_batch"], rb["C4_makes_larger_batch"]],
                    "C5_timely_gt_C2": [ra["C5_timely_gt_C2"], rb["C5_timely_gt_C2"]],
                })

    # measured-hardware mechanism availability (from CP2): no lower power state, caps unsettable.
    lower_state_exists = False
    power_cap_settable = False
    # a valid mechanism pass needs a causal batch/cap/low-power change explaining the benefit
    mechanism_available = lower_state_exists or power_cap_settable  # "larger batch" tested per-hit below

    summary = {
        "opportunity_gate": {
            "hits": opportunity_hits,
            "any_conservative_hit": any(h["power_model"] == "conservative" for h in opportunity_hits),
            "any_hit": len(opportunity_hits) > 0,
            "all_hits_favorable_only_and_offload_not_batch": all(
                (h["power_model"] == "favorable" and not any(h["C4_makes_larger_batch"]))
                for h in opportunity_hits) if opportunity_hits else None,
        },
        "measured_mechanism_availability": {
            "lower_power_state_below_idle": lower_state_exists,
            "power_cap_settable": power_cap_settable,
            "any_C4_creates_larger_server_batch": any(r["C4_makes_larger_batch"] for r in results),
            "note": "CP2: A6000 idles at auto-P8 (25W, no deeper state); caps need root (unsettable). "
                    "Offload only SHRINKS server batches, never enlarges them.",
        },
        "verdict_inputs": {
            "opportunity_robust": any(h["power_model"] == "conservative" for h in opportunity_hits),
            "mechanism_gate_can_pass": mechanism_available or any(r["C4_makes_larger_batch"] for r in results),
            "physical_gate": "ENERGY_BLOCKED",
        },
    }
    with open(os.path.join(ART, "cp4_gate_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # console
    print("=== CP4 per-instance C4/C5 vs C1 (%) ===")
    for r in results:
        print(f"  {r['instance']:<44} pm={r['power_model']:<12} "
              f"C4={r['C4_rel_pct']:+6.1f}%  C5={r['C5_rel_pct']:+6.1f}%  "
              f"C5/C4={r['C5_retains_frac_of_C4']:.2f}  larger_batch={r['C4_makes_larger_batch']}  "
              f"SLO_ok={r['SLO_no_worse']}")
    print("\n=== OPPORTUNITY GATE (C4>=15% in two adjacent load bins, SLO no worse) ===")
    if not opportunity_hits:
        print("  NO adjacent-bin hits anywhere -> opportunity gate NOT met.")
    for h in opportunity_hits:
        print(f"  HIT pm={h['power_model']} mix={h['mix']} slack={h['slack']} loads={h['load_pair']} "
              f"C4={h['C4_rel']}% larger_batch={h['C4_makes_larger_batch']} C5_timely>C2={h['C5_timely_gt_C2']}")
    print("\n=== MEASURED MECHANISM AVAILABILITY ===")
    print(f"  lower power state below idle: {lower_state_exists}   power cap settable: {power_cap_settable}")
    print(f"  any C4 offload creates a LARGER server batch: {summary['measured_mechanism_availability']['any_C4_creates_larger_server_batch']}")
    print(f"  opportunity robust (holds under conservative power): {summary['verdict_inputs']['opportunity_robust']}")
    print(f"  physical gate: {summary['verdict_inputs']['physical_gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
