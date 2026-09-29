#!/usr/bin/env python3
"""Per-shape statistics of S41SERVERFFNUSB per-call lines (phone-reported compute_us, USB round trip)."""
import pathlib, re, statistics, sys
LINE = re.compile(r"S41SERVERFFNUSB request=(\d+) layer=(\d+) tokens=(\d+) columns=(\d+) slot=\d+ h2d_bytes=\d+ d2h_bytes=\d+ started_ns=(\d+) h2d_completed_ns=(\d+) d2h_completed_ns=(\d+) compute_us=(\d+)")
def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]
groups = {}
for arg in sys.argv[1:]:
    for path in sorted(pathlib.Path(arg).glob("large-model-*.stderr")):
        for line in path.read_text(errors="replace").splitlines():
            m = LINE.search(line)
            if not m: continue
            _, layer, tokens, columns, start, h2d, d2h, compute = map(int, m.groups())
            key = (int(tokens), int(columns))
            groups.setdefault(key, {"compute": [], "rtt": [], "processes": set()})
            groups[key]["compute"].append(compute / 1000.0)
            groups[key]["rtt"].append((d2h - start) / 1e6)
            groups[key]["processes"].add(path.name)
print("| tokens | columns | calls | processes | compute mean ms | p10 | p50 | p90 | USB round trip mean ms |")
print("|---|---|---|---|---|---|---|---|---|")
rows = []
for (tokens, columns), g in sorted(groups.items()):
    c = g["compute"]
    row = dict(tokens=tokens, columns=columns, calls=len(c), processes=len(g["processes"]), compute_mean_ms=statistics.fmean(c),
               p10=pct(c, .1), p50=pct(c, .5), p90=pct(c, .9), rtt_mean_ms=statistics.fmean(g["rtt"]))
    rows.append(row)
    print("| %d | %d | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f |" % (tokens, columns, len(c), row["processes"], row["compute_mean_ms"], row["p10"], row["p50"], row["p90"], row["rtt_mean_ms"]))
