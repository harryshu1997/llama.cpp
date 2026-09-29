#!/usr/bin/env python3
"""Summarize DUALBENCH_RESULT lines from logs/*.log into per-(M, f) tables."""
import glob, os, re, statistics as st, sys
from collections import defaultdict

root = os.path.dirname(os.path.abspath(__file__))
logs = sorted(glob.glob(os.path.join(root, sys.argv[1] if len(sys.argv) > 1 else "logs", "*_f*.log")))
rows = defaultdict(list)   # (M, f, mode) -> list of dicts (one per rep)
checks = []
for p in logs:
    tag = os.path.basename(p)[:-4]
    for line in open(p, errors="replace"):
        if line.startswith("DUALBENCH_RESULT"):
            d = dict(kv.split("=", 1) for kv in line.split()[1:])
            key = (int(d["M"]), float(d["frac"]), d["mode"])
            rows[key].append({k: (float(v) if re.match(r"^-?[0-9.]+$", v) else v) for k, v in d.items()} | {"tag": tag})
        elif line.startswith("DUALBENCH_CHECK"):
            checks.append((tag, line.strip()))

LAYER_BYTES = 534773760
def fmt_spread(vals):
    return f"{st.mean(vals):.3f} [{min(vals):.3f}-{max(vals):.3f}]"

base = {}
for (M, f, mode), rs in rows.items():
    if f == 1.0 and mode == "npu_solo":
        base[M] = st.mean(r["wall_p50"] for r in rs)

print("| M | f (actual) | NPU cols | GPU cols | reps | wall p50 ms mean [min-max over reps] | wall mean | wall p90 | NPU leg p50 (dual) | NPU solo p50 | GPU leg p50 (dual) | GPU solo p50 | sync p50 | merge | agg GB/s | speedup vs NPU-only |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for M in sorted({k[0] for k in rows}):
    fs = sorted({k[1] for k in rows if k[0] == M}, reverse=True)
    for f in fs:
        main = "dual" if (M, f, "dual") in rows else ("npu_solo" if f == 1.0 else "gpu_solo")
        rs = rows[(M, f, main)]
        npu_solo = rows.get((M, f, "npu_solo"), [])
        gpu_solo = rows.get((M, f, "gpu_solo"), [])
        p50 = [r["wall_p50"] for r in rs]
        wmean = st.mean(r["wall_mean"] for r in rs)
        p90 = st.mean(r["wall_p90"] for r in rs)
        npu_leg = st.mean(r["npu_p50"] for r in rs) if main != "gpu_solo" else 0
        gpu_leg = st.mean(r["gpu_p50"] for r in rs) if main != "npu_solo" else 0
        ns = st.mean(r["npu_p50"] for r in npu_solo) if npu_solo else 0
        gs = st.mean(r["gpu_p50"] for r in gpu_solo) if gpu_solo else 0
        sync = st.mean(r["sync_p50"] for r in rs)
        merge = st.mean(r["merge_mean"] for r in rs)
        agg = LAYER_BYTES / (st.mean(p50) * 1e6)
        sp = base.get(M, float("nan")) / st.mean(p50)
        r0 = rs[0]
        print(f"| {M} | {f:.4f} | {int(r0['npu_cols'])} | {int(r0['gpu_cols'])} | {len(rs)} | {fmt_spread(p50)} | {wmean:.3f} | {p90:.3f} | "
              f"{npu_leg:.3f} | {ns:.3f} | {gpu_leg:.3f} | {gs:.3f} | {sync:.4f} | {merge:.4f} | {agg:.1f} | {sp:.3f}x |")
print()
print("Correctness (layer 10, fixed input, vs fp64 CPU reference):")
for tag, c in checks:
    print(f"- {tag}: {c}")
