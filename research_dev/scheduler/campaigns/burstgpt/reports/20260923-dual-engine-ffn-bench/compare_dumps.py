#!/usr/bin/env python3
"""Compare layer-10 outputs of each dual/GPU run against the NPU-only (f=1.0) run of the same rep and M."""
import glob, os, re
import numpy as np

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dumps")
files = sorted(glob.glob(os.path.join(root, "*_M*.f32")))
index = {}
for p in files:
    m = re.match(r"(\w+?)(\d)_f([0-9.]+)_M(\d+)\.f32", os.path.basename(p))
    if m:
        index[(m.group(1), m.group(2), m.group(3), int(m.group(4)))] = p
print("| run | M | f | max_abs vs NPU-only | rel_L2 vs NPU-only |")
print("|---|---|---|---|---|")
for (kind, rep, f, M), p in sorted(index.items()):
    ref = index.get((kind, rep, "1.0", M))
    if ref is None or f == "1.0":
        continue
    a, b = np.fromfile(p, np.float32), np.fromfile(ref, np.float32)
    d = a.astype(np.float64) - b
    print(f"| {kind}{rep} | {M} | {f} | {np.abs(d).max():.3g} | {np.linalg.norm(d) / np.linalg.norm(b):.3g} |")
