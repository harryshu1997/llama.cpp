#!/usr/bin/env python3
"""CP2 gate analysis: median provisioning goodput per window vs window=1, both phones."""
import json, sys, statistics as st, collections, glob, os

A = os.path.dirname(os.path.abspath(__file__)) + "/artifacts"

def load(path):
    rows = collections.defaultdict(list)
    errs = []
    correctness = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception as e:
            errs.append({"parse_error": str(e), "line": line[:120]})
            continue
        if "error" in o:
            errs.append(o); continue
        w = o.get("stage_window")
        rows[w].append(o)
        # correctness invariants per run
        correctness.append({
            "window": w, "rep": o.get("rep"),
            "verdict": o.get("verdict"), "model_source": o.get("model_source"),
            "chunks": o.get("stage_chunks_sent"), "bytes": o.get("stage_bytes_sent"),
            "sha_ok": o.get("model_sha256") == "5cfba18d2a47acc190f317d650895bcc53e914e9a0bc61631860be9591ed360d",
            "rel_l2": o.get("rel_l2_max"),
        })
    return rows, errs, correctness

def med(v): return st.median(v) if v else 0.0
def cov(v): return (st.pstdev(v)/st.mean(v)*100) if len(v) > 1 and st.mean(v) else 0.0

summary = {}
for dev, tag in [("OP12","op12"),("OP15","op15")]:
    path = f"{A}/{tag}_sweep.jsonl"
    if not os.path.exists(path):
        print(f"{dev}: no sweep file yet"); continue
    rows, errs, corr = load(path)
    print(f"\n===== {dev} ({path}) =====")
    bad = [c for c in corr if not (c["verdict"]=="DYNAMIC_FFN_PASS" and c["model_source"]=="published_store"
             and c["chunks"]==111 and c["bytes"]==464114176 and c["sha_ok"] and (c["rel_l2"] or 1) < 5e-3)]
    print(f"runs={sum(len(v) for v in rows.values())}  correctness_failures={len(bad)}  errors={len(errs)}")
    if bad: print("  BAD:", bad[:3])
    if errs: print("  ERR:", errs[:3])
    w1 = [o["stage_useful_goodput_mib_s"] for o in rows.get(1, [])]
    m1 = med(w1)
    print(f"{'win':>4} {'n':>3} {'med MiB/s':>10} {'min':>7} {'max':>7} {'CoV%':>6} {'ratio_vs_w1':>12}")
    dev_sum = {}
    for w in sorted(rows):
        g = [o["stage_useful_goodput_mib_s"] for o in rows[w]]
        e = [o["stage_e2e_ms"] for o in rows[w]]
        ratio = (med(g)/m1) if m1 else 0.0
        print(f"{w:>4} {len(g):>3} {med(g):>10.2f} {min(g):>7.2f} {max(g):>7.2f} {cov(g):>6.1f} {ratio:>11.2f}x")
        dev_sum[w] = {"n":len(g), "median_goodput_mib_s":round(med(g),2), "median_e2e_ms":round(med(e),1),
                      "min":round(min(g),2),"max":round(max(g),2),"cov_pct":round(cov(g),1),
                      "ratio_vs_w1":round(ratio,3)}
    best = max((dev_sum[w]["ratio_vs_w1"], w) for w in dev_sum) if dev_sum else (0,None)
    gate = best[0] >= 1.20
    print(f"  BEST window={best[1]} ratio={best[0]:.2f}x  GATE(>=1.20x): {'PASS' if gate else 'FAIL'}")
    summary[dev] = {"per_window":dev_sum, "best_window":best[1], "best_ratio":round(best[0],3),
                    "gate_pass": gate, "correctness_failures":len(bad), "errors":len(errs)}

if summary:
    both = all(s["gate_pass"] for s in summary.values()) and len(summary)==2
    print(f"\n==== OVERALL GATE (both phones >=1.20x): {'PASS' if both else 'FAIL/INCOMPLETE'} ====")
    summary["overall_gate_pass"] = both
    json.dump(summary, open(f"{A}/cp2_sweep_summary.json","w"), indent=2)
    print(f"wrote {A}/cp2_sweep_summary.json")
