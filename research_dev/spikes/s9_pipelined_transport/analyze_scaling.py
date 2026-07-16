#!/usr/bin/env python3
# Summarize provisioning goodput for the 64/256 MiB provision-only phases and the
# simultaneous phase. Reports median + min/max and the min(best)/max(w1) conservative
# ratio alongside the median ratio. Goodput key differs by record type.
import sys, json, statistics as st, collections

def goodput(o):
    return o.get("stage_useful_goodput_mib_s", o.get("useful_goodput_mib_s"))

def summarize(path):
    by = collections.defaultdict(lambda: collections.defaultdict(list))  # device -> window -> [goodput]
    errs = 0
    for line in open(path):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if "error" in o: errs += 1; continue
        g = goodput(o)
        w = o.get("matrix_window", o.get("stage_window"))
        dev = o.get("device", "?")
        if g is not None and w is not None:
            by[dev][w].append(g)
    print(f"\n===== {path} (errors={errs}) =====")
    for dev in sorted(by):
        ws = by[dev]
        base = st.median(ws[1]) if 1 in ws else None
        print(f"  {dev}:")
        for w in sorted(ws):
            g = ws[w]
            r = f"{st.median(g)/base:.2f}x" if base else "-"
            print(f"    w{w}: n={len(g)} med={st.median(g):.2f} min={min(g):.2f} max={max(g):.2f} ratio_vs_w1={r}")
        if base:
            best = max((st.median(ws[w]), w) for w in ws)
            cons = min(ws[best[1]]) / max(ws[1])
            print(f"    best=w{best[1]} median_ratio={best[0]/base:.2f}x conservative(min_best/max_w1)={cons:.2f}x")

for p in sys.argv[1:]:
    summarize(p)
