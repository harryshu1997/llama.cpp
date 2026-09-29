#!/usr/bin/env python3
"""Utilization report for one campaign run directory (read-only).

Sections: request timeline with concurrent service (joins), dispatch-policy notes, phone calls by rows per
call, per-token split from the USB call lines (phone round trip / CPU attention between phone layers / rest),
and host energy per activity state from resource-samples.jsonl. Usage: analyze_run.py <run-eval/run dir>.
"""
import glob, json, os, re, statistics as st, sys

RUN = sys.argv[1].rstrip("/")
USB = re.compile(r"layer=(\d+) tokens=(\d+) .*?started_ns=(\d+) h2d_completed_ns=(\d+) d2h_completed_ns=(\d+) compute_us=(\d+)")


def load_result():
    return json.load(open(os.path.join(RUN, "RESULT.json")))


def timeline(d):
    p0 = d["paid_start_ns"]; T = d["duration_us"] / 1e6
    rows = []
    for r in sorted(d["request_results"], key=lambda r: r["replay_arrival_us"]):
        ft = (r["first_token_ns"] - p0) / 1e9 if r.get("first_token_ns") else None
        end = r["completion"]["actual_end_us"] / 1e6
        notes = []
        for rc in r.get("dispatch_receipts") or []:
            for k, v in rc.items():
                if "CONTINUOUS_JOIN" in json.dumps(v):
                    notes.append(k)
        rows.append((r["request_id"].split(":")[-1], r["model_id"][:5], r["replay_arrival_us"] / 1e6, ft, end,
                     r["output_tokens"], r["actual_executor_id"].split(":")[-1], notes))
    print("== timeline (s): id model arrival first_token end out executor  [concurrent-with]")
    for x in rows:
        conc = [y[0] for y in rows if y is not x and y[3] and x[3] and y[3] < x[4] and x[3] < y[4] and y[1] == x[1]]
        print("%s %-5s %7.1f %7.1f %7.1f %4d %-8s %s %s" % (x[0], x[1], x[2], x[3] or 0, x[4], x[5], x[6],
              ("co-decoded with " + ",".join(conc)) if conc else "", " ".join(x[7])))
    iv = sorted((x[3] or x[2], x[4]) for x in rows)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    print("serving %.0f s of %.0f (%.0f%%); waited total %.0f s; active_slots_peak %s; last arrival %.0f s" % (
        busy, T, 100 * busy / T, sum((x[3] or x[2]) - x[2] for x in rows), d.get("active_slots_peak"), max(x[2] for x in rows)))
    dp = json.dumps(d.get("dispatch_policy"))
    print("dispatch_policy:", dp[:300])
    txt = open(os.path.join(RUN, "RESULT.json")).read()
    print("markers:", {k: txt.count(k) for k in ("CONTINUOUS_JOIN_DESKTOP_PARENT", "CONTINUOUS_JOIN_BARRIER_BYPASS",
                                                  "RECOVERY_RETAINED_QUEUE_PLACE", "DEVICE_POWER_STATE", "THERMAL_DEFERRAL")})


def usb_calls(path):
    out = []
    for line in open(path, errors="replace"):
        if "S41SERVERFFNUSB" in line:
            m = USB.search(line)
            if m:
                out.append(tuple(int(x) for x in m.groups()))
    return out


def rows_histogram():
    print("== phone calls by rows per call (USB helper): server: rows=n calls rt/compute p50 ms")
    for f in sorted(glob.glob(os.path.join(RUN, "large-model-*-desktop.stderr"))):
        by = {}
        for layer, ntok, s, h, d2, c in usb_calls(f):
            by.setdefault(ntok, []).append(((d2 - s) / 1e6, c / 1e3))
        if by:
            name = os.path.basename(f).replace("large-model-", "").replace("-desktop.stderr", "")
            print(name, " | ".join("rows=%d: %5d calls rt %.1f comp %.1f" % (k, len(v), st.median(x[0] for x in v),
                  st.median(x[1] for x in v)) for k, v in sorted(by.items())))


def per_token():
    print("== per-token split (median): period = OP15 round trip + CPU attention between phone layers + rest")
    for f in sorted(glob.glob(os.path.join(RUN, "large-model-*-desktop.stderr"))):
        calls = usb_calls(f)
        if not calls:
            continue
        toks, cur = [], None
        for layer, ntok, s, h, d2, c in calls:
            if layer == 0:
                if cur:
                    toks.append(cur)
                cur = {"start": s, "rt": 0.0, "comp": 0.0, "gaps": 0.0, "n": 0, "rows": ntok, "last": None}
            if cur is None:
                continue
            if cur["last"] is not None:
                cur["gaps"] += (s - cur["last"]) / 1e6
            cur["rt"] += (d2 - s) / 1e6; cur["comp"] += c / 1e3; cur["n"] += 1; cur["last"] = d2
        if cur:
            toks.append(cur)
        byrows = {}
        for a, b in zip(toks, toks[1:]):
            p = (b["start"] - a["start"]) / 1e6
            if 0 < p < 5000:
                byrows.setdefault(a["rows"], []).append((p, a["rt"], a["comp"], a["gaps"], (b["start"] - a["last"]) / 1e6, a["n"]))
        name = os.path.basename(f).replace("large-model-", "").replace("-desktop.stderr", "")
        for rows, v in sorted(byrows.items()):
            if len(v) < 10:
                continue
            P, RT, C, G, TL, n = (st.median(x[i] for x in v) for i in range(6))
            print("%-16s rows=%d tokens=%4d layers=%2d | period %4.0f = rt %3.0f (comp %3.0f + usb %2.0f, %2.0f%%) + attn %3.0f (%.1f/layer) + rest %3.0f" % (
                name, rows, len(v), n, P, RT, C, RT - C, 100 * RT / P, G, G / max(1, n - 1), TL))


def energy_states():
    S, prev = [], None
    path = os.path.join(RUN, "resource-samples.jsonl")
    if not os.path.exists(path):
        return
    for line in open(path):
        try:
            s = json.loads(line)
        except Exception:
            continue
        g, h, r = s["gpu"], s["host_activity"], s.get("rapl_package") or {}
        j = h.get("cpu_jiffies")
        if prev is not None and j and r.get("energy_uj") is not None:
            dt = (s["t_ns"] - prev[0]) / 1e9
            de = r["energy_uj"] - prev[2]
            if de < 0:
                de += r.get("max_energy_range_uj", 0)
            tot = sum(j.values()) - sum(prev[1].values())
            busy = (sum(v for k, v in j.items() if k not in ("idle", "iowait")) - sum(v for k, v in prev[1].items() if k not in ("idle", "iowait"))) / max(1, tot)
            if 0 < dt < 5:
                S.append((dt, de / 1e6 / dt, g["power_mw"] / 1000, g["utilization_pct"], g["memory_used_bytes"] / 2 ** 30, busy, g.get("clocks_sm")))
        prev = (s["t_ns"], j, r.get("energy_uj"))

    def cls(x):
        if x[4] < 2:
            return "unloaded-idle"
        if x[5] < 0.05:
            return "loaded-idle" if x[3] == 0 else "loaded-gpu-only"
        return "active"
    tt = sum(x[0] for x in S); te = sum(x[0] * (x[1] + x[2]) for x in S)
    print("== host energy by state (cpu-package + gpu-board): total %.0f s %.1f kJ mean %.1f W" % (tt, te / 1e3, te / tt))
    for c in ("unloaded-idle", "loaded-idle", "loaded-gpu-only", "active"):
        xs = [x for x in S if cls(x) == c]
        if xs:
            t = sum(x[0] for x in xs); e = sum(x[0] * (x[1] + x[2]) for x in xs)
            sm = [x[6] for x in xs if x[6] is not None]
            print("  %-16s %6.0f s %3.0f%%  %6.1f kJ %3.0f%%  cpu %5.1f W  gpu %5.1f W%s" % (c, t, 100 * t / tt, e / 1e3, 100 * e / te,
                  sum(x[0] * x[1] for x in xs) / t, sum(x[0] * x[2] for x in xs) / t, ("  sm p50 %.0f MHz" % st.median(sm)) if sm else ""))


if __name__ == "__main__":
    d = load_result()
    timeline(d); rows_histogram(); per_token(); energy_states()
