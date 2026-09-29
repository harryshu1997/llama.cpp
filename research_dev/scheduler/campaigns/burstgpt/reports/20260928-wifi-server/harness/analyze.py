#!/usr/bin/env python3
"""Per-arm table + per-token time-model check for server_bench.py results.

    python3 analyze.py RESULTS_DIR [--config wifi_config.json] [--rtt op15=W1_op15.json --rtt pixel=W1_pixel.json]
    python3 analyze.py --predict [--config ...] [--rtt-ms 1.5,3,5]     # expectations, no results needed

Table columns (per arm and concurrency c)
    step ms        median decode step period (all c rows advance once per step)
    ms/tok         step / c;  tok/s = c * 1000 / step
    CPU-res ms     step minus the GPU part (16 GPU layers; from RESULT-gpu.json scaled by bytes, else
                   bytes / --gpu-bw-gbs); the part the phones can shorten
    rtt p50/p99    phone round trip per call (tap proxy if the arm ran with --tap; else the server's
                   shutdown summary, which has p50/p90 only; FunctionFS runs also have per-call lines)
    comp p50       phone compute per call (worker compute_us)
    host p50       host partial FFN time per call (server summary host_p50_ms: CPU columns of that layer)
    GPU W, GPU J/tok   NVML board power / energy over the steady decode window
    CPU J/tok      RAPL package energy (null when energy_uj is not readable)
    ident          token sequences identical to the cpu arm (same c, wave, slot)
    vs cpu         step time and GPU energy relative to the cpu arm (negative = saving)

Model check: predicted CPU-resident time per step for every arm, with the CPU bandwidth fitted on the
cpu arm, two models (benchlib.predict_cpu_resident_ms):
    overlap    per layer attn + max(CPU (1-f) FFN, owner phone f FFN + RTT)    (what the server does)
    aggregate  bytes / (BW_cpu + BW_phones) + calls x RTT                      (PLAN.md physics)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchlib as bl  # noqa: E402


def load_results(directory):
    results = {}
    for path in sorted(Path(directory).glob("RESULT-*.json")):
        with open(path) as handle:
            result = json.load(handle)
        results[result["arm"]["name"]] = result
    return results


def fmt(value, spec="%.1f"):
    return "-" if value is None else spec % value


def owners_from_config(config):
    owner_of = {}
    for helper in config["helpers"]:
        for layer in bl.mask_layers(helper["layer_mask"]):
            owner_of[layer] = helper["phone"]
    return owner_of


def gpu_part_ms(results, table, n_layer, gpu_layers, concurrency, gpu_bw_gbs):
    """GPU-resident share of a step: from the gpu arm (all layers on GPU) scaled by bytes, else bytes/BW."""
    gpu_layers_list = list(range(n_layer - gpu_layers, n_layer))
    gpu_bytes = sum(sum(table["layers"][layer].values()) for layer in gpu_layers_list)
    all_bytes = sum(sum(row.values()) for row in table["layers"].values()) + table["output"]
    reference = results.get("gpu")
    if reference:
        for level in reference["levels"]:
            if level["concurrency"] == concurrency and level["metrics"]["step_period_ms"]:
                return level["metrics"]["step_period_ms"] * gpu_bytes / all_bytes, "gpu-arm"
    return gpu_bytes / (gpu_bw_gbs * 1e9) * 1e3, "bytes/%.0fGBs" % gpu_bw_gbs


def helper_stats(result):
    """{label: {rtt_p50, rtt_p99, compute_p50, host_p50, wait_p50, transport_p50, source}}."""
    out = {}
    tap = result.get("tap")
    summaries = {row.get("helper", "helper0"): row for row in result.get("ffn", {}).get("summaries", [])}
    labels = set(summaries) | set((tap or {}).get("by_helper", {}))
    for label in sorted(labels):
        row = {"source": None}
        summary = summaries.get(label, {})
        if tap and label in tap.get("by_helper", {}):
            calls = tap["by_helper"][label]
            row.update(rtt_p50=calls["rpc_ms"]["p50"], rtt_p99=calls["rpc_ms"]["p99"],
                       compute_p50=calls["compute_ms"]["p50"], transport_p50=calls["transport_ms"]["p50"],
                       transport_p99=calls["transport_ms"]["p99"], source="tap")
        elif summary.get("calls"):
            row.update(rtt_p50=summary.get("rpc_p50_ms"), rtt_p99=None, rtt_p90=summary.get("rpc_p90_ms"),
                       compute_p50=summary.get("compute_p50_ms"),
                       transport_p50=(summary.get("rpc_p50_ms") or 0) - (summary.get("compute_p50_ms") or 0),
                       transport_p99=None, source="server-summary")
        row["host_p50"] = summary.get("host_p50_ms")
        row["wait_p50"] = summary.get("wait_p50_ms")
        row["calls"] = summary.get("calls")
        row["layer_mask"] = summary.get("layer_mask")
        out[label] = row
    return out


def phone_physics(config, results, table, rtt_override=None):
    """Per phone {bw_gbs, rtt_ms, source}: measured phone arm > config model_physics defaults."""
    physics = config.get("model_physics", {}).get("phones", {})
    ffn_layer = table["layers"][0]["ffn"]
    phones = {}
    for helper in config["helpers"]:
        name = helper["phone"]
        default = physics.get(name, {"compute_ms_full_layer": 8.9, "rtt_ms": 1.5})
        phones.setdefault(name, {"bw_gbs": ffn_layer / (default["compute_ms_full_layer"] * 1e6),
                                 "rtt_ms": default["rtt_ms"], "source": "config"})
    measured = results.get("phone")
    if measured:
        stats = helper_stats(measured)
        by_phone = {}
        for helper in config["helpers"]:
            row = stats.get(helper["label"])
            if row and row.get("compute_p50"):
                by_phone.setdefault(helper["phone"], []).append(row)
        for name, rows in by_phone.items():
            compute = sorted(row["compute_p50"] for row in rows)[len(rows) // 2]
            transport = sorted(row["transport_p50"] for row in rows if row.get("transport_p50") is not None)
            phones[name] = {"bw_gbs": ffn_layer / (compute * 1e6),
                            "rtt_ms": transport[len(transport) // 2] if transport else phones[name]["rtt_ms"],
                            "source": "phone-arm"}
    for name, value in (rtt_override or {}).items():
        if name in phones:
            phones[name]["rtt_ms"], phones[name]["source"] = value, phones[name]["source"] + "+rtt-override"
    return phones


def rtt_from_wifi_json(path):
    with open(path) as handle:
        data = json.load(handle)
    return data["rtt_ms"]["p50"]


def analyze(directory, config, gpu_bw_gbs, rtt_override):
    results = load_results(directory)
    if not results:
        raise SystemExit("no RESULT-*.json in " + str(directory))
    table = bl.model_bytes(config["server"]["model"])
    n_layer = table["n_layer"]
    gpu_layers = config["server"]["gpu_layers"]
    cpu_layers = bl.cpu_resident_layers(n_layer, gpu_layers)
    cpu_bytes = sum(sum(table["layers"][layer].values()) for layer in cpu_layers) + table["output"]
    owner_of = owners_from_config(config)
    phones = phone_physics(config, results, table, rtt_override)
    lines = []
    lines.append("model %s: %d layers, CPU-resident layers %d-%d + output head = %.2f GB/step; FFN %.1f MB/layer"
                 % (table["source"], n_layer, cpu_layers[0], cpu_layers[-1], cpu_bytes / 1e9,
                    table["layers"][0]["ffn"] / 1e6))
    header = ("| arm | c | step ms | ms/tok | tok/s | CPU-res ms | rtt p50 | rtt p99 | comp p50 | host p50 | "
              "GPU W | GPU J/tok | CPU J/tok | ident | step vs cpu | GPU J vs cpu |")
    lines += ["", header, "|" + "|".join(["---"] * 16) + "|"]
    base = {}
    if "cpu" in results:
        for level in results["cpu"]["levels"]:
            base[level["concurrency"]] = level["metrics"]
    fits = {}
    order = ["cpu", "cpu-helpers", "phone"] + sorted(name for name in results if name.startswith("split-")) + ["gpu"]
    for name in [n for n in order if n in results]:
        result = results[name]
        stats = helper_stats(result)
        rtt50 = [row["rtt_p50"] for row in stats.values() if row.get("rtt_p50")]
        rtt99 = [row["rtt_p99"] for row in stats.values() if row.get("rtt_p99")]
        comp = [row["compute_p50"] for row in stats.values() if row.get("compute_p50")]
        host = [row["host_p50"] for row in stats.values() if row.get("host_p50")]
        med = lambda values: sorted(values)[len(values) // 2] if values else None
        for level in result["levels"]:
            m = level["metrics"]
            c = m["concurrency"]
            gpu_ms, gpu_src = gpu_part_ms(results, table, n_layer, gpu_layers, c, gpu_bw_gbs)
            cpu_res = m["step_period_ms"] - gpu_ms if m["step_period_ms"] and name != "gpu" else None
            if name == "cpu" and cpu_res:
                fits[c] = cpu_bytes / (cpu_res * 1e6)
            identity = m.get("identity_vs_cpu")
            ref = base.get(c)
            rel = lambda a, b: None if a is None or b in (None, 0) else 100.0 * (a / b - 1.0)
            lines.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name, c, fmt(m["step_period_ms"]), fmt(m["ms_per_token"]), fmt(m["decode_tok_s"], "%.2f"),
                fmt(cpu_res), fmt(med(rtt50), "%.2f"), fmt(max(rtt99) if rtt99 else None, "%.2f"),
                fmt(med(comp), "%.2f"), fmt(med(host), "%.2f"), fmt(m["gpu_decode_mean_w"]),
                fmt(m["gpu_decode_j_per_token"], "%.2f"), fmt(m.get("cpu_pkg_decode_j_per_token"), "%.2f"),
                "-" if not identity else "%d/%d" % (identity["identical"], identity["compared"]),
                fmt(rel(m["step_period_ms"], ref and ref["step_period_ms"]), "%+.1f%%"),
                fmt(rel(m["gpu_decode_j_per_token"], ref and ref["gpu_decode_j_per_token"]), "%+.1f%%")))
    lines.append("")
    lines.append("GPU part of a step: %s (%s); RAPL: %s" % (
        fmt(gpu_part_ms(results, table, n_layer, gpu_layers, 1, gpu_bw_gbs)[0]),
        gpu_part_ms(results, table, n_layer, gpu_layers, 1, gpu_bw_gbs)[1],
        next(iter(results.values())).get("power_sources", {}).get("rapl", {}).get("reason", "readable")))
    helper_rows = []
    for name in [n for n in order if n in results]:
        for label, row in helper_stats(results[name]).items():
            if row.get("calls"):
                helper_rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    name, label, row["calls"], fmt(row.get("rtt_p50"), "%.2f"), fmt(row.get("rtt_p99"), "%.2f"),
                    fmt(row.get("compute_p50"), "%.2f"), fmt(row.get("transport_p50"), "%.2f"),
                    fmt(row.get("host_p50"), "%.2f"), fmt(row.get("wait_p50"), "%.2f")))
    if helper_rows:
        lines += ["", "| arm | helper | calls | rpc p50 | rpc p99 | compute p50 | net+overhead p50 | host p50 | wait p50 |",
                  "|" + "|".join(["---"] * 9) + "|"] + helper_rows
    # model check
    lines.append("")
    bw_fit = fits.get(1) or next(iter(fits.values()), None)
    bw_ref = config.get("model_physics", {}).get("cpu_bw_gbs", 69.0)
    lines.append("time model (CPU-resident ms per step): BW_cpu fitted on the cpu arm = %s GB/s (reference %.0f); phones: %s"
                 % (fmt(bw_fit), bw_ref, ", ".join("%s %.1f GB/s rtt %.2f ms (%s)" % (k, v["bw_gbs"], v["rtt_ms"], v["source"])
                                                  for k, v in sorted(phones.items()))))
    lines += ["", "| arm | c | measured | overlap (fit) | err | aggregate (fit) | err | overlap @%.0f GB/s |" % bw_ref,
              "|" + "|".join(["---"] * 8) + "|"]
    for name in [n for n in order if n in results and n != "gpu"]:
        arm = results[name]["arm"]
        for level in results[name]["levels"]:
            m = level["metrics"]
            c = m["concurrency"]
            gpu_ms, _ = gpu_part_ms(results, table, n_layer, gpu_layers, c, gpu_bw_gbs)
            measured = m["step_period_ms"] - gpu_ms if m["step_period_ms"] else None
            bw = fits.get(c, bw_fit)
            if not bw or measured is None:
                continue
            share = arm["share"] if arm["helpers"] else 0.0
            overlap = bl.predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw, phones, "overlap")["ms"]
            aggregate = bl.predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw, phones, "aggregate")["ms"]
            ref = bl.predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw_ref, phones, "overlap")["ms"]
            lines.append("| %s | %d | %.1f | %.1f | %+.0f%% | %.1f | %+.0f%% | %.1f |" % (
                name, c, measured, overlap, 100 * (overlap / measured - 1), aggregate,
                100 * (aggregate / measured - 1), ref))
    return "\n".join(lines), results


def predict(config, rtts, bw_cpu, gpu_ms):
    table = bl.model_bytes(config["server"]["model"])
    n_layer = table["n_layer"]
    cpu_layers = bl.cpu_resident_layers(n_layer, config["server"]["gpu_layers"])
    owner_of = owners_from_config(config)
    physics = config.get("model_physics", {}).get("phones", {})
    ffn = table["layers"][0]["ffn"]
    lines = ["model %s, CPU layers %d-%d + output, BW_cpu %.0f GB/s, GPU part %.1f ms/step" % (
        table["source"], cpu_layers[0], cpu_layers[-1], bw_cpu, gpu_ms)]
    header = "| rtt ms | arm | overlap step ms | vs cpu | aggregate step ms | vs cpu |"
    lines += ["", header, "|" + "|".join(["---"] * 6) + "|"]
    base = bl.predict_cpu_resident_ms(table, cpu_layers, 0.0, owner_of, bw_cpu, {}, "overlap")["ms"] + gpu_ms
    for rtt in rtts:
        phones = {name: {"bw_gbs": ffn / (spec["compute_ms_full_layer"] * 1e6), "rtt_ms": rtt}
                  for name, spec in physics.items()}
        for share, name in ((0.0, "cpu"), (0.25, "split-25"), (0.5, "split-50"), (0.75, "split-75"), (1.0, "phone")):
            overlap = bl.predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw_cpu, phones, "overlap")["ms"] + gpu_ms
            aggregate = bl.predict_cpu_resident_ms(table, cpu_layers, share, owner_of, bw_cpu, phones, "aggregate")["ms"] + gpu_ms
            lines.append("| %.1f | %s | %.1f | %+.1f%% | %.1f | %+.1f%% |" % (
                rtt, name, overlap, 100 * (overlap / base - 1), aggregate, 100 * (aggregate / base - 1)))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="?")
    parser.add_argument("--config", default=str(HERE / "wifi_config.json"))
    parser.add_argument("--gpu-bw-gbs", type=float, default=600.0)
    parser.add_argument("--rtt", action="append", default=[], help="PHONE=wifi_rtt.json (W1 output) overrides the RTT")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--rtt-ms", default="0.5,1.5,3,5")
    parser.add_argument("--bw-cpu", type=float, default=69.0)
    parser.add_argument("--gpu-ms", type=float, default=17.0)
    parser.add_argument("--out", help="write the report here (default RESULTS/ANALYSIS.md)")
    args = parser.parse_args(argv)
    config_path = args.config
    if not os.path.isfile(config_path):
        config_path = str(HERE / "wifi_config.example.json")
    config = bl.load_config(config_path)
    if args.predict:
        print(predict(config, [float(v) for v in args.rtt_ms.split(",")], args.bw_cpu, args.gpu_ms))
        return 0
    if not args.results:
        parser.error("RESULTS directory required (or --predict)")
    run = Path(args.results) / "RUN.json"
    if run.is_file():
        # the configuration the run actually used (model path, helpers, physics)
        raw = json.load(open(run))["config"]
        config = bl.load_config({key: raw[key] for key in ("server", "phones", "model_physics", "workload")
                                 if key in raw})
    overrides = {}
    for item in args.rtt:
        phone, _, path = item.partition("=")
        overrides[phone] = rtt_from_wifi_json(path)
    report, _ = analyze(args.results, config, args.gpu_bw_gbs, overrides)
    print(report)
    out = args.out or str(Path(args.results) / "ANALYSIS.md")
    with open(out, "w") as handle:
        handle.write("# analyze.py %s\n\n%s\n" % (args.results, report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
