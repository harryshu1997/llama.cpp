#!/usr/bin/env python3
"""Phone compute per call vs idle gap before the call and vs position in the per-token burst.

Uses S41SERVERFFNUSB per-call lines (host clock started/completed ns, phone compute_us) of one or more runs.
    python3 gap_analysis.py LABEL=RUN_DIR[:COLUMNS] ...
"""
import pathlib, re, statistics, sys
LINE = re.compile(r"S41SERVERFFNUSB request=(\d+) layer=(\d+) tokens=(\d+) columns=(\d+) slot=\d+ h2d_bytes=\d+ d2h_bytes=\d+ started_ns=(\d+) h2d_completed_ns=(\d+) d2h_completed_ns=(\d+) compute_us=(\d+)")
BUCKETS = ((0, 1), (1, 5), (5, 20), (20, 100), (100, 1000), (1000, 10 ** 9))
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
for spec in sys.argv[1:]:
    label, _, rest = spec.partition("=")
    run_dir, _, cols = rest.partition(":")
    want = set(int(c) for c in cols.split(",")) if cols else None
    by_gap = {b: [] for b in BUCKETS}
    by_layer = {}
    for path in sorted(pathlib.Path(run_dir).glob("large-model-*.stderr")):
        prev_end = None
        for line in path.read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if not m:
                continue
            _, layer, tokens, columns, start, h2d, d2h, compute = map(int, m.groups())
            gap_ms = None if prev_end is None else (start - prev_end) / 1e6
            prev_end = d2h
            if tokens != 1 or (want and columns not in want) or gap_ms is None:
                continue
            for lo, hi in BUCKETS:
                if lo <= gap_ms < hi:
                    by_gap[(lo, hi)].append(compute / 1000)
                    break
            by_layer.setdefault(layer, []).append(compute / 1000)
    print(f"### {label} ({run_dir}) columns={sorted(want) if want else all}")
    print("| idle gap before call (ms) | calls | compute mean ms | p50 | p90 |")
    print("|---|---|---|---|---|")
    for (lo, hi), v in by_gap.items():
        if v:
            print(f"| [{lo}, {hi if hi < 10 ** 9 else "inf"}) | {len(v)} | {statistics.fmean(v):.3f} | {pct(v, .5):.3f} | {pct(v, .9):.3f} |")
    print("| layer | calls | compute mean ms | p50 |")
    print("|---|---|---|---|")
    for layer in sorted(by_layer):
        v = by_layer[layer]
        print(f"| {layer} | {len(v)} | {statistics.fmean(v):.3f} | {pct(v, .5):.3f} |")
    print()
