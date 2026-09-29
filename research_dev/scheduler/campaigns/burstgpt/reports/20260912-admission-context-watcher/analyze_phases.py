"""Attribute package and GPU energy of a run to model load/prefill, decode and idle phases.

usage: analyze_phases.py <run dir> [<run dir> ...]
Overlapping phases share each sample interval's energy equally.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def attribute(run: Path) -> None:
    r = json.load(open(run / "RESULT.json")); base = r["paid_start_ns"]
    roles = {v: k for k, v in r["model_roles"].items()}
    phases = []
    for x in r["request_results"]:
        role = roles.get(x["model_id"], x["model_id"])
        disp = min((d["observed_at_us"] for d in x["dispatch_receipts"] if d["status"] == "ACQUIRED"), default=None)
        ft = None if not x.get("first_token_ns") else (x["first_token_ns"] - base) / 1e9
        end = (x["completion"] or {}).get("actual_end_us")
        if disp is None or ft is None or end is None:
            continue
        cold = any(tr.get("evicted_artifact_sha256s") or tr.get("target_state") in ("hot", "warm")
                   for tr in (x["terminal_ticket"].get("transition_receipts") or []))
        phases.append((disp / 1e6, ft, f"{role}-{'load+prefill' if cold else 'prefill'}"))
        phases.append((ft, end / 1e6, f"{role}-decode"))
    rows = [json.loads(l) for l in open(run / "resource-samples.jsonl")]
    rows = [x for x in rows if x.get("rapl_package") and x.get("gpu")]
    E = collections.defaultdict(float); G = collections.defaultdict(float); T = collections.defaultdict(float)
    for a, b in zip(rows, rows[1:]):
        t0 = (a["t_ns"] - base) / 1e9; t1 = (b["t_ns"] - base) / 1e9
        if t1 <= t0:
            continue
        de = b["rapl_package"]["energy_uj"] - a["rapl_package"]["energy_uj"]
        if de < 0:
            de += a["rapl_package"]["max_energy_range_uj"]
        if de > 500_000 * (t1 - t0) * 1000:  # > 500 W: counter reset, skip
            continue
        dg = (a["gpu"]["power_mw"] + b["gpu"]["power_mw"]) / 2 * (t1 - t0) * 1000
        active = [c for (s, e, c) in phases if s < t1 and e > t0] or ["idle/gaps"]
        for c in active:
            E[c] += de / len(active); G[c] += dg / len(active); T[c] += (t1 - t0) / len(active)
    tok = collections.Counter()
    for x in r["request_results"]:
        tok[roles.get(x["model_id"])] += x["output_tokens"]
    print(f"== {run.parent.name}  package {sum(E.values())/1e9:.2f} kJ  gpu {sum(G.values())/1e9:.2f} kJ  duration {r['duration_us']/1e6:.1f} s")
    print(f"   {'category':22} {'time s':>8} {'pkg kJ':>8} {'gpu kJ':>8} {'pkg W':>7} {'gpu W':>7}")
    for c in sorted(E, key=lambda k: -(E[k] + G[k])):
        print(f"   {c:22} {T[c]:8.1f} {E[c]/1e9:8.2f} {G[c]/1e9:8.2f} {E[c]/1e6/max(T[c],1e-9):7.1f} {G[c]/1e6/max(T[c],1e-9):7.1f}")
    for role in ("gemma", "qwen"):
        c = f"{role}-decode"
        print(f"   {role} decode J/token (pkg+gpu): {(E[c]+G[c])/1e6/max(1,tok[role]):.1f}  tokens {tok[role]}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        attribute(Path(path))
