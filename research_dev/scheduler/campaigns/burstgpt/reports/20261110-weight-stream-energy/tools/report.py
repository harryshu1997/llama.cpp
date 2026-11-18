#!/usr/bin/env python3
"""Fit and final tables of the weight-stream energy benchmark.

    report.py --bytes MODEL_BYTES.json --device "OnePlus 15 NPU=op15/RESULTS.json" --device "..." [--md OUT.md] [--json OUT.json]

Per model and device: J/GB = least-squares slope through the origin of J per call against GB per call over the
unflagged runs of that model (2, 4, 8 layers). Per device: the same fit over every unflagged run, and a fit with
an intercept (fixed J per call). J/token = model J/GB x GB the full model streams per decoded token.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

ORDER = ["qwen25-05b", "qwen3-06b", "llama32-1b", "qwen3-17b", "gemma4-e2b", "qwen3-4b", "qwen3-8b", "gemma4-12b",
         "qwen3-14b", "qwen3-32b"]
LABEL = {"qwen25-05b": "Qwen2.5-0.5B", "qwen3-06b": "Qwen3-0.6B *", "llama32-1b": "Llama-3.2-1B",
         "qwen3-17b": "Qwen3-1.7B *", "gemma4-e2b": "Gemma-4-E2B", "qwen3-4b": "Qwen3-4B *", "qwen3-8b": "Qwen3-8B",
         "gemma4-12b": "Gemma-4-12B", "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B"}


def slope0(points):
    sxx = sum(x * x for x, _ in points)
    return sum(x * y for x, y in points) / sxx if sxx else None


def linfit(points):
    n = len(points)
    if n < 3:
        return None
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if not sxx:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    ss_res = sum((y - a - b * x) ** 2 for x, y in points)
    ss_tot = sum((y - my) ** 2 for _, y in points)
    return {"intercept_j": a, "slope_j_per_gb": b, "r2": 1 - ss_res / ss_tot if ss_tot else None}


def fmt(value, digits=3):
    if value is None:
        return "-"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.{digits}f}" if value < 1 else f"{value:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=Path, required=True)
    ap.add_argument("--device", action="append", required=True, metavar="NAME=RESULTS.json")
    ap.add_argument("--key", default="j_per_call", choices=["j_per_call", "net"])
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    model_bytes = json.loads(args.bytes.read_text())
    devices = OrderedDict()
    for item in args.device:
        name, _, path = item.partition("=")
        devices[name] = json.loads(Path(path).read_text())

    def energy(run):
        if run.get("watts") is None:
            return None
        if args.key == "net":
            if run.get("idle_w") is None:
                return None
            return (run["watts"] - run["idle_w"]) * run["compute_ms"] / 1e3
        return run["j_per_call"]

    summary = {"devices": {}, "models": {}}
    lines = []
    for name, runs in devices.items():
        # DRAM-bound asymptote: median GB/s of the three largest working sets; a run faster than 1.15x that is
        # served partly from the CPU cache (it does not represent a full model, which never fits)
        is_host = any("threads" in r for r in runs)   # the cache test is for CPU legs only (NPU runs never sit in cache)
        largest = sorted(runs, key=lambda r: r["mb"])[-3:]
        asymptote = sorted(r["gb_per_s"] for r in largest)[len(largest) // 2] if largest else None
        for r in runs:
            r["cache_assisted"] = bool(is_host and asymptote and r["gb_per_s"] > 1.15 * asymptote)
        usable = [r for r in runs if not r.get("throttled") and not r.get("unsteady") and not r["cache_assisted"] and energy(r) is not None]
        points = [(r["mb"] / 1e3, energy(r)) for r in usable]
        fit = {"runs": len(runs), "used": len(usable), "flagged": [r["tag"] for r in runs if r.get("throttled") or r.get("unsteady")],
               "dram_bound_gb_per_s": asymptote, "cache_assisted": [r["tag"] for r in runs if r["cache_assisted"]],
               "j_per_gb_origin": slope0(points) if points else None, "with_intercept": linfit(points)}
        per_model = {}
        for model in ORDER:
            mine = [r for r in runs if r["model"] == model]
            good = [r for r in mine if not r.get("throttled") and not r.get("unsteady") and not r["cache_assisted"] and energy(r) is not None]
            timing = [r for r in mine if not r.get("throttled") and not r.get("unsteady") and not r["cache_assisted"]]
            from_device_fit = not timing
            timing = timing or mine
            if not mine:
                continue
            gbs = sum(r["mb"] / 1e3 for r in timing) / sum(r["compute_ms"] / 1e3 for r in timing)
            if from_device_fit and asymptote:
                gbs = asymptote  # every run of this model sat in the cache: use the device's DRAM-bound numbers
            per_model[model] = {
                "j_per_gb": slope0([(r["mb"] / 1e3, energy(r)) for r in good]) if good else None,
                "gb_per_s": gbs, "runs": len(mine), "used": len(good),
                "watts": sum(r["watts"] for r in good) / len(good) if good else None,
                "from_device_fit": from_device_fit,
            }
            if from_device_fit and fit["j_per_gb_origin"] is not None:
                per_model[model]["j_per_gb"] = fit["j_per_gb_origin"]
            if per_model[model]["j_per_gb"] is not None and model in model_bytes:
                per_model[model]["j_per_token"] = per_model[model]["j_per_gb"] * model_bytes[model]["streamed_bytes"] / 1e9
                per_model[model]["s_per_token"] = model_bytes[model]["streamed_bytes"] / 1e9 / gbs
        fit["per_model"] = per_model
        summary["devices"][name] = fit
    for model in ORDER:
        if model in model_bytes:
            summary["models"][model] = {"streamed_gb_per_token": model_bytes[model]["streamed_bytes"] / 1e9,
                                        "ffn_share": model_bytes[model]["ffn_share"]}

    names = list(devices)
    lines.append("| Model | GB/token | " + " | ".join(f"{n} J/GB | {n} J/token" for n in names) + " |")
    lines.append("|---|---:|" + "---:|---:|" * len(names))
    for model in ORDER:
        if not any(model in summary["devices"][n]["per_model"] for n in names):
            continue
        row = [LABEL[model], f"{summary['models'][model]['streamed_gb_per_token']:.2f}"]
        for n in names:
            pm = summary["devices"][n]["per_model"].get(model, {})
            mark = " †" if pm.get("from_device_fit") and pm.get("j_per_gb") is not None else ""
            row += [fmt(pm.get("j_per_gb")) + mark, fmt(pm.get("j_per_token"), 2) + mark]
        lines.append("| " + " | ".join(row) + " |")
    row = ["**Device fit (all runs)**", ""]
    for n in names:
        row += [f"**{fmt(summary['devices'][n]['j_per_gb_origin'])}**", ""]
    lines.append("| " + " | ".join(row) + " |")
    text = "\n".join(lines)
    print(text)
    for n in names:
        d = summary["devices"][n]
        wi = d["with_intercept"]
        print(f"\n{n}: {d['used']}/{d['runs']} runs used, J/GB through origin {fmt(d['j_per_gb_origin'])}"
              + (f", with intercept {fmt(wi['slope_j_per_gb'])} J/GB + {wi['intercept_j']*1e3:.2f} mJ/call (R2 {wi['r2']:.4f})" if wi else "")
              + (f", flagged {d['flagged']}" if d["flagged"] else "")
              + (f", DRAM-bound {d['dram_bound_gb_per_s']:.1f} GB/s, cache-assisted runs excluded: {len(d['cache_assisted'])}" if d.get("cache_assisted") else ""))
    if args.md:
        args.md.write_text(text + "\n")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
