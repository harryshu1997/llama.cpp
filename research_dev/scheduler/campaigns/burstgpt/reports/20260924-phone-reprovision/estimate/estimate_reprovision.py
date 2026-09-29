#!/usr/bin/env python3
"""Change #4 estimates with the implemented constraints, using the report's calibrated replay model.

Imports ../20260923-offload-accounting/hypotheticals.py read-only; only the phone reload rate and the
layer counts differ from the report's variant D.
"""
import importlib.util
import json
import pathlib
import sys

REPORT = pathlib.Path("/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/burstgpt/reports/20260923-offload-accounting")
spec = importlib.util.spec_from_file_location("hypotheticals", REPORT / "hypotheticals.py")
hyp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyp)

accounting = json.loads((REPORT / "data/accounting.json").read_text())
cal = hyp.calibrate(accounting)
requests = accounting["treatment"]["requests"]
for row in requests:
    if row["server_prompt_ms"] is None:
        row["server_prompt_ms"] = 40.0
loads = {}
for model, role in (("qwen", "hot"), ("gemma", "cold")):
    values = [m["load_s"] for m in accounting["treatment"]["model_loads"] if m["role"] == role]
    loads[model] = sum(values) / len(values)
switch_extra_s = accounting["treatment"].get("switch_extra_s") or 9.0
order = [r["request_id"] for r in sorted(requests, key=lambda r: (r["scheduled_start_s"], r["arrival_s"]))]
partners = {r["request_id"]: tuple(p for p in r["decode_partners"].split(",") if p) for r in requests}


def run(name, rate, **overrides):
    hyp.PHONE_RELOAD_BYTES_PER_S = rate
    cfg = hyp.base_config()
    cfg["admission"] = "replay"
    cfg["replay_order"], cfg["replay_partners"] = order, partners
    for key, value in overrides.items():
        cfg[key] = {**cfg[key], **value} if isinstance(value, dict) and isinstance(cfg.get(key), dict) else value
    result = hyp.Sim(requests, cal, loads, switch_extra_s, cfg).run()
    result["name"], result["reload_bytes_per_s"] = name, rate
    return result


reference = run("replay (12 Qwen + 8 Gemma, today)", 270e6)
rows = [reference]
for phone in ("current", "coherent_coalesced"):
    for label, layers, rate in (
        ("report D: 18/26 @270 MB/s", {"qwen": 18, "gemma": 26}, 270e6),
        ("impl, existing shards: 18/24 @205 MB/s", {"qwen": 18, "gemma": 24}, 205e6),
        ("impl, live limit < 9.63 GB: 17/24 @205 MB/s", {"qwen": 17, "gemma": 24}, 205e6),
        ("impl + Gemma 24-25 shards: 18/26 @205 MB/s", {"qwen": 18, "gemma": 26}, 205e6),
    ):
        if phone == "current" and label.startswith("replay"):
            continue
        rows.append(run(f"{label} [{phone}]", rate, layers=layers, phone=phone))
print(f"loads_s={ {k: round(v, 1) for k, v in loads.items()} } switch_extra_s={switch_extra_s}")
for model, rate in (("qwen", 205e6), ("gemma", 205e6)):
    for layers in ((17, 18) if model == "qwen" else (24, 26)):
        print(f"full {model} {layers} layers reload @205 MB/s: {layers * hyp.LAYER_BYTES[model] / rate:5.1f} s "
              f"(mean desktop load {loads[model]:.1f} s)")
print(f"{'variant':<62} {'dur s':>7} {'host kJ':>8} {'dE%':>6} {'dT%':>6} {'phone%':>7}")
for row in rows:
    print(f"{row['name']:<62} {row['duration_s']:7.0f} {row['host_kj']:8.1f} "
          f"{100 * (row['host_kj'] / reference['host_kj'] - 1):+6.1f} {100 * (row['duration_s'] / reference['duration_s'] - 1):+6.1f} "
          f"{100 * row['phone_share']:6.0f}%")
out = pathlib.Path(__file__).with_name("estimate_reprovision.json")
out.write_text(json.dumps(rows, indent=1) + "\n")
