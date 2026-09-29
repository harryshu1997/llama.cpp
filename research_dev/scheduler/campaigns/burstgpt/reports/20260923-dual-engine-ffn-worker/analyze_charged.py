#!/usr/bin/env python3
"""Aggregate the CHARGED reps (raw/c_*): per-config latency, engine split,
bandwidth, clocks in the driver window, battery trajectory, numerics."""
import csv
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "raw"
N_EMBD = 5120
BYTES_PER_COL = 3 * N_EMBD * 2  # gate+up+down F16


def load_run(d, tag):
    calls = list(csv.DictReader(open(os.path.join(d, f"calls_{tag}.csv"))))
    compute = [float(c["compute_us"]) for c in calls]
    n = len(compute)
    dual = {k: [] for k in ("primary_us", "secondary_us", "wait_us", "merge_us")}
    pcols = scols = None
    for line in open(os.path.join(d, f"worker_{tag}.log"), errors="replace"):
        if line.startswith("S43DUALFFN"):
            f = dict(kv.split("=") for kv in line.split()[1:])
            for k in dual:
                dual[k].append(float(f[k]))
            pcols, scols = int(f["primary_columns"]), int(f["secondary_columns"])
    for k in dual:
        dual[k] = dual[k][-n:]
    cols = int(calls[0]["columns"])
    meta = open(os.path.join(d, f"battery_{tag}.txt")).read()
    t0 = float(re.search(r"driver_start (\S+)", meta).group(1))
    t1 = float(re.search(r"driver_end (\S+)", meta).group(1))
    batt = {}
    for which in ("before", "after"):
        m = re.search(which + r" .*?level:(\d+).*?voltage:(\d+).*?temperature:(\d+).*?notify=(\S*)", meta)
        if m:
            batt[which] = dict(level=int(m.group(1)), mv=int(m.group(2)),
                               temp_dC=int(m.group(3)), notify=m.group(4))
    clocks = []
    cpath = os.path.join(d, f"clocks_{tag}.txt")
    if os.path.exists(cpath):
        for line in open(cpath):
            p = line.split()
            if len(p) >= 4 and t0 <= float(p[0]) <= t1:
                clocks.append([float(x) for x in p[1:4]] + [float(x) for x in p[4:8]] + [0.0] * (8 - len(p)))
    row = dict(run=os.path.basename(d.rstrip("/")), tag=tag, tokens=int(calls[0]["tokens"]),
               columns=cols, calls=n,
               p50_ms=statistics.median(compute) / 1e3,
               mean_ms=statistics.fmean(compute) / 1e3,
               p10_ms=float(np.percentile(compute, 10)) / 1e3,
               p90_ms=float(np.percentile(compute, 90)) / 1e3,
               gbps=BYTES_PER_COL * cols / statistics.median(compute) / 1e3,
               battery=batt)
    if dual["primary_us"]:
        row.update(primary_ms=statistics.median(dual["primary_us"]) / 1e3,
                   secondary_ms=statistics.median(dual["secondary_us"]) / 1e3,
                   wait_ms=statistics.median(dual["wait_us"]) / 1e3,
                   merge_ms=statistics.median(dual["merge_us"]) / 1e3,
                   primary_columns=pcols, secondary_columns=scols)
        row["npu_gbps"] = BYTES_PER_COL * pcols / (row["primary_ms"] * 1e6)
        if scols:
            row["gpu_gbps"] = BYTES_PER_COL * scols / (row["secondary_ms"] * 1e6)
    if clocks:
        c = np.array(clocks)
        row.update(gpuclk_mhz_med=float(np.median(c[:, 0])) / 1e6,
                   gpuclk_mhz_max=float(c[:, 0].max()) / 1e6,
                   ddr_med=float(np.median(c[:, 1])), ddr_min=float(c[:, 1].min()),
                   llcc_med=float(np.median(c[:, 2])), clock_samples=len(c),
                   ddr_mean=float(c[:, 1].mean()), ddr_frac_high=float((c[:, 1] > 547000).mean()),
                   bwmon_mean=float(c[:, 3].mean()), memlat_gold_mean=float(c[:, 4].mean()),
                   memlat_prime_mean=float(c[:, 5].mean()), memlat_gold_compute_mean=float(c[:, 6].mean()))
    base = os.path.join(d, f"out_T{row['tokens']}_f0.bin")
    out = os.path.join(d, f"out_{tag}.bin")
    if os.path.exists(base) and os.path.exists(out):
        a, b = np.fromfile(base, np.float32), np.fromfile(out, np.float32)
        if a.size == b.size and a.size:
            row["rel_l2"] = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            row["max_abs"] = float(np.abs(a - b).max())
    return row


rows = []
for d in sorted(glob.glob(os.path.join(root, "c_*"))):
    for p in sorted(glob.glob(os.path.join(d, "calls_*.csv"))):
        tag = re.match(r"calls_(.*)\.csv", os.path.basename(p)).group(1)
        rows.append(load_run(d, tag))
json.dump(rows, open(os.path.join(root, "charged_runs.json"), "w"), indent=1)

print("## per run")
print("| run | tag | p50 ms | mean | p10-p90 | GB/s | NPU ms | GPU ms | wait | merge | NPU GB/s | GPU GB/s | gpuclk MHz | DDR med | DDR>547 share | DDR mean | LLCC | batt lvl/mV | rel L2 |")
print("|" + "---|" * 19)
for r in rows:
    b = r["battery"]
    bs = f"{b.get('before', {}).get('level')}->{b.get('after', {}).get('level')} / {b.get('before', {}).get('mv')}->{b.get('after', {}).get('mv')}"
    f = lambda k, fmt="{:.2f}": fmt.format(r[k]) if k in r else ""
    print(f"| {r['run']} | {r['tag']} | {r['p50_ms']:.2f} | {r['mean_ms']:.2f} | {r['p10_ms']:.2f}-{r['p90_ms']:.2f} | {r['gbps']:.1f} | "
          f"{f('primary_ms')} | {f('secondary_ms')} | {f('wait_ms')} | {f('merge_ms', '{:.3f}')} | {f('npu_gbps', '{:.1f}')} | {f('gpu_gbps', '{:.1f}')} | "
          f"{f('gpuclk_mhz_med', '{:.0f}')} | {f('ddr_med', '{:.0f}')} | {f('ddr_frac_high', '{:.2f}')} | {f('ddr_mean', '{:.0f}')} | {f('llcc_med', '{:.0f}')} | {bs} | {f('rel_l2', '{:.2e}')} |")

print("\n## per config across reps")
print("| tag | reps | p50 ms per rep | mean of p50 | min-max | speedup vs off (per-rep paired) |")
print("|---|---|---|---|---|---|")
by = {}
for r in rows:
    by.setdefault(r["tag"], []).append(r)
for tag in sorted(by, key=lambda t: (t.split("_")[0], -1 if t.endswith("none") else float(t.split("_f")[1]))):
    rs = by[tag]
    p = [r["p50_ms"] for r in rs]
    sp = []
    for r in rs:
        off = next((o for o in rows if o["run"] == r["run"] and o["tag"] == f"T{r['tokens']}_f0"), None)
        if off:
            sp.append(off["p50_ms"] / r["p50_ms"])
    print(f"| {tag} | {len(rs)} | {', '.join(f'{x:.2f}' for x in p)} | {statistics.fmean(p):.2f} | {min(p):.2f}-{max(p):.2f} | "
          f"{', '.join(f'{x:.2f}x' for x in sp)} |")
