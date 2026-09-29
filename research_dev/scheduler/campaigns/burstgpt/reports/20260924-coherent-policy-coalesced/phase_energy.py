#!/usr/bin/env python3
"""Host energy (RAPL package + NVML board) split into adaptive-decode window time vs the rest of the paid interval.

Window times are microseconds from RESULT paid_start_ns; resource samples carry absolute t_ns on the same clock.
Energy in an interval = RAPL counter delta (linear interpolation) + trapezoid of GPU power."""
import bisect, json, sys, pathlib, collections

def load(run):
    run = pathlib.Path(run)
    r = json.load(open(run / "RESULT.json"))
    samples = [json.loads(l) for l in open(run / "resource-samples.jsonl")]
    rapl = sorted((s["rapl_package"]["sample_t_ns"], s["rapl_package"]["energy_uj"], s["rapl_package"]["max_energy_range_uj"]) for s in samples if s.get("rapl_package"))
    # unwrap
    t_r, e_r, off, prev = [], [], 0, None
    for t, e, m in rapl:
        if prev is not None and e < prev:
            off += m
        prev = e
        t_r.append(t); e_r.append(e + off)
    gpu = sorted((s["gpu"]["sample_t_ns"], s["gpu"]["power_mw"]) for s in samples if s.get("gpu"))
    return r, (t_r, e_r), gpu

def rapl_at(rapl, t):
    ts, es = rapl
    i = bisect.bisect_left(ts, t)
    if i <= 0: return es[0]
    if i >= len(ts): return es[-1]
    t0, t1, e0, e1 = ts[i-1], ts[i], es[i-1], es[i]
    return e0 + (e1 - e0) * (t - t0) / (t1 - t0)

def gpu_energy(gpu, a, b):
    ts = [t for t, _ in gpu]; ps = [p for _, p in gpu]
    def p_at(t):
        i = bisect.bisect_left(ts, t)
        if i <= 0: return ps[0]
        if i >= len(ts): return ps[-1]
        return ps[i-1] + (ps[i] - ps[i-1]) * (t - ts[i-1]) / (ts[i] - ts[i-1])
    pts = [a] + [t for t in ts if a < t < b] + [b]
    return sum((p_at(x) + p_at(y)) / 2 * (y - x) for x, y in zip(pts, pts[1:])) / 1e12  # mW*ns -> J

def energy(rapl, gpu, a, b):
    return (rapl_at(rapl, b) - rapl_at(rapl, a)) / 1e6, gpu_energy(gpu, a, b)

def union(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def main(label, run):
    r, rapl, gpu = load(run)
    t0 = r["paid_start_ns"]; t1 = r["paid_end_ns"]
    obs = json.load(open(pathlib.Path(run) / "ADAPTIVE_DECODE_OBSERVATIONS.json"))
    model = {x["request_id"]: x["model_id"].split("-")[0] for x in r["request_results"]}
    by = collections.defaultdict(list)
    for g in obs["groups"]:
        for w in g["windows"]:
            if g["request_id"] not in model:
                continue
            by[model[g["request_id"]]].append((t0 + w["started_at_us"] * 1000, t0 + w["finished_at_us"] * 1000))
    total = energy(rapl, gpu, t0, t1)
    res = {"label": label, "paid_s": round((t1 - t0) / 1e9, 1), "total_cpu_kj": round(total[0] / 1e3, 2), "total_gpu_kj": round(total[1] / 1e3, 2)}
    all_iv = union([iv for ivs in by.values() for iv in ivs])
    dec = [energy(rapl, gpu, a, b) for a, b in all_iv]
    res["decode_s"] = round(sum(b - a for a, b in all_iv) / 1e9, 1)
    res["decode_host_kj"] = round(sum(c + g for c, g in dec) / 1e3, 2)
    res["other_s"] = round(res["paid_s"] - res["decode_s"], 1)
    res["other_host_kj"] = round(sum(total) / 1e3 - res["decode_host_kj"], 2)
    for m, ivs in sorted(by.items()):
        u = union(ivs)
        e = [energy(rapl, gpu, a, b) for a, b in u]
        res[m + "_decode_s"] = round(sum(b - a for a, b in u) / 1e9, 1)
        res[m + "_decode_host_kj"] = round(sum(c + g for c, g in e) / 1e3, 2)
    te = r["trace_energy"]["fleet_energy_uj_by_domain"]
    res["result_host_kj"] = round((te["cpu-package"] + te["gpu-board"]) / 1e9, 2)
    print(json.dumps(res))

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        label, _, run = spec.partition("=")
        main(label, run)
