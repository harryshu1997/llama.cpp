#!/usr/bin/env python3
"""Summarise a phone_sweep.sh output dir: latency per config, speedup and
numerics vs the NPU-only (f=0) run with the same inputs."""
import csv
import glob
import json
import os
import re
import statistics
import sys

import numpy as np

root = sys.argv[1]
rows = []
tags = sorted({re.match(r"calls_(T\d+_f[0-9a-z.]+)\.csv", os.path.basename(p)).group(1)
               for p in glob.glob(os.path.join(root, "calls_*.csv"))})


def key(tag):
    t, f = tag.split("_")
    return int(t[1:]), (-1.0 if f[1:] == "none" else float(f[1:]))


for tag in sorted(tags, key=key):
    tokens, fraction = key(tag)
    with open(os.path.join(root, f"calls_{tag}.csv")) as fh:
        calls = list(csv.DictReader(fh))
    compute = [float(c["compute_us"]) for c in calls]
    dual = {"primary_us": [], "secondary_us": [], "wait_us": [], "merge_us": []}
    secondary_columns = 0
    log = open(os.path.join(root, f"worker_{tag}.log"), errors="replace").read()
    for line in log.splitlines():
        if line.startswith("S43DUALFFN"):
            fields = dict(kv.split("=") for kv in line.split()[1:])
            for k in dual:
                dual[k].append(float(fields[k]))
            secondary_columns = int(fields["secondary_columns"])
    # drop the warmup calls logged by the worker (driver excludes them)
    n = len(compute)
    for k in dual:
        dual[k] = dual[k][-n:] if dual[k] else []
    row = {
        "tag": tag, "tokens": tokens, "fraction_requested": fraction,
        "calls": n,
        "secondary_columns": secondary_columns,
        "compute_p50_us": statistics.median(compute),
        "compute_mean_us": statistics.fmean(compute),
    }
    for k, v in dual.items():
        if v:
            row[k.replace("_us", "_p50_us")] = statistics.median(v)
    base = os.path.join(root, f"out_T{tokens}_f0.bin")
    out = os.path.join(root, f"out_{tag}.bin")
    if os.path.exists(base) and os.path.exists(out):
        a = np.fromfile(base, np.float32)
        b = np.fromfile(out, np.float32)
        if a.size == b.size and a.size:
            row["max_abs_vs_off"] = float(np.abs(a - b).max())
            row["rel_l2_vs_off"] = float(np.linalg.norm(a - b) / np.linalg.norm(a))
            row["ref_max_abs"] = float(np.abs(a).max())
    rows.append(row)

for row in rows:
    base = next((r for r in rows if r["tokens"] == row["tokens"] and r["fraction_requested"] == 0), None)
    if base:
        row["speedup_vs_off"] = base["compute_p50_us"] / row["compute_p50_us"]

json.dump(rows, open(os.path.join(root, "summary.json"), "w"), indent=1)
hdr = ["tag", "secondary_columns", "compute_p50_us", "speedup_vs_off", "primary_p50_us",
       "secondary_p50_us", "wait_p50_us", "merge_p50_us", "max_abs_vs_off", "rel_l2_vs_off"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for r in rows:
    vals = []
    for h in hdr:
        v = r.get(h, "")
        vals.append(f"{v:.4g}" if isinstance(v, float) else str(v))
    print("| " + " | ".join(vals) + " |")
