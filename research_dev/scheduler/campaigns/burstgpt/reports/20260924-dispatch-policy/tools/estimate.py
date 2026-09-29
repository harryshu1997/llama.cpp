#!/usr/bin/env python3
"""Time/energy estimate of simulated dispatch orders with the accounting's calibrated pass model.

The dispatcher simulation (simulate_dispatch.py, real scheduler code, recorded durations) gives the
service order and which requests overlapped on one server. hypotheticals.py (offload accounting,
2026-09-23) replays such an order at pass level with its calibrated power/latency model. Only the
long-tail v1 trace has a calibration (accounting.json), so this covers run-5 / the baseline arm.

Usage: estimate.py SIM_JSON [SIM_JSON ...]
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

REPORT = Path("/home/myid/zs89458/Documents/llama.cpp-release/research_dev/scheduler/campaigns/"
              "burstgpt/reports/20260923-offload-accounting")
spec = importlib.util.spec_from_file_location("hypotheticals", REPORT / "hypotheticals.py")
hyp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyp)


def order_and_partners(sim: dict) -> tuple[list[str], dict[str, tuple[str, ...]]]:
    rows = {key.rsplit(":", 1)[-1]: value for key, value in sim["requests"].items()}
    order = [key for key, _ in sorted(rows.items(), key=lambda item: (item[1]["acquired_us"], item[0]))]
    model = {}
    for line in sim["log"]:
        parts = line.split()
        if parts and parts[0] == "ACQUIRE":
            route = parts[3]
            model[parts[2]] = ("llama" if "4b90" in route or "control" in route
                               else "gemma" if "cold:desktop" in route else "qwen")
    partners = {}
    for key, row in rows.items():
        partners[key] = tuple(sorted(
            other for other, peer in rows.items()
            if other != key and model.get(other) == model.get(key)
            and peer["acquired_us"] < row["done_us"] and row["acquired_us"] < peer["done_us"]
        ))
    return order, partners


PROMPT_MS_PER_TOKEN = {"qwen": 11.77, "gemma": 17.83, "llama": 1.31}  # long-tail v1 server logs


def dev2_requests() -> list[dict]:
    """dev_v2 request shapes from the dev2base RESULT; prompt time from the long-tail per-token rate."""
    result = json.loads((Path(__file__).resolve().parents[1] / "runs/dev2base/RESULT.json").read_text())
    rows = []
    for row in result["request_results"]:
        model = ("gemma" if "gemma" in row["model_id"] else
                 "qwen" if "qwen" in row["model_id"] else "llama")
        rows.append({
            "request_id": row["request_id"].rsplit(":", 1)[-1], "model": model,
            "arrival_s": row["replay_arrival_us"] / 1e6,
            "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"],
            "server_prompt_ms": row["input_tokens"] * PROMPT_MS_PER_TOKEN[model],
        })
    return rows


def main() -> int:
    accounting = json.loads((REPORT / "data/accounting.json").read_text())
    cal = hyp.calibrate(accounting)
    dev2 = "--dev2" in sys.argv
    if dev2:
        sys.argv.remove("--dev2")
        requests = dev2_requests()
        loads = {"gemma": (97.1 + 72.6 + 76.5) / 3, "qwen": (90.7 + 84.6) / 2}  # M: dev2base loads
    else:
        requests = accounting["treatment"]["requests"]
        loads = {}
        for model, role in (("qwen", "hot"), ("gemma", "cold")):
            values = [m["load_s"] for m in accounting["treatment"]["model_loads"] if m["role"] == role]
            loads[model] = sum(values) / len(values)
    for row in requests:
        if row["server_prompt_ms"] is None:
            row["server_prompt_ms"] = 40.0
    switch_extra_s = accounting["treatment"].get("switch_extra_s") or 9.0
    print(f"{'order source':<36} {'phone':<20} {'dur s':>7} {'host kJ':>8} {'switches':>8} {'phone%':>7} {'mean wait s':>11}")
    results = {}
    for path in sys.argv[1:]:
        sim = json.loads(Path(path).read_text())
        order, partners = order_and_partners(sim)
        for phone, overrides in (
            ("none", {"phone": "none"}),
            ("current", {}),
            ("coherent_coalesced", {"phone": "coherent_coalesced"}),
        ):
            cfg = hyp.base_config()
            cfg["admission"] = "replay"
            cfg["replay_order"], cfg["replay_partners"] = order, partners
            cfg["max_batch"] = {"qwen": 4, "gemma": 2}
            cfg.update(overrides)
            out = hyp.Sim(copy.deepcopy(requests), cal, loads, switch_extra_s, cfg).run()
            results[(Path(path).stem, phone)] = out
            print(f"{Path(path).stem:<36} {phone:<20} {out['duration_s']:7.0f} {out['host_kj']:8.1f} "
                  f"{out['switches']:>8} {100 * out['phone_share']:6.0f}% {out['mean_queue_wait_s']:>11}")
    Path(sys.argv[1]).parent.joinpath("estimates_dev2.json" if dev2 else "estimates.json").write_text(json.dumps(
        {f"{k[0]}|{k[1]}": v for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
