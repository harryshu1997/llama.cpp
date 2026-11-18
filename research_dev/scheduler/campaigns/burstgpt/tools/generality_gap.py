"""Decompose a treatment arm's energy saving against its all-desktop baseline.

For a baseline run (phones idle) and a treatment run (phone-assisted) on the same rig this reports

* host energy split into an *idle floor* (the lowest ``--idle-window-s`` mean host power seen in the
  run, times the paid duration) and *dynamic* energy above it;
* decode tokens per model, split by whether the request's physical proof shows phone calls;
* the planner's predicted per-token saving (``cpu_only_energy_j_per_token`` - ``energy_j_per_token``
  from a layer-placement REPORT) applied to (a) every eligible token (full phone use) and (b) only the
  tokens that were actually phone-assisted; the difference is the saving lost to helper deferral;
* the measured host / phone / fleet deltas, so the gap between prediction and measurement is explicit.

Usage::

    generality_gap.py --baseline nph=<run dir> --treatment new=<run dir> --plan REPORT.json \
        [--meter nph=METER.json --meter new=METER.json] [--idle-window-s 60] [--json OUT] [--md OUT]

Energy numbers come from ``fleet_energy.account_run`` (RAPL package + NVML board, phones from the meter).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_energy  # noqa: E402

SCHEMA = "generality-gap-v1"


def host_power_series(samples: list[dict[str, Any]], start_s: float, end_s: float) -> list[tuple[float, float]]:
    """(t_s, host_w) at the sample cadence inside the paid window: GPU board + RAPL package."""
    gpu = sorted((int(r["gpu"]["sample_t_ns"]) / 1e9, float(r["gpu"]["power_mw"]) / 1000) for r in samples)
    rapl = sorted((int(r["rapl_package"]["sample_t_ns"]) / 1e9, int(r["rapl_package"]["energy_uj"]),
                   int(r["rapl_package"]["max_energy_range_uj"])) for r in samples)
    out = []
    for (t0, e0, m0), (t1, e1, _m1) in zip(rapl, rapl[1:]):
        if t1 <= t0 or t1 < start_s or t0 > end_s:
            continue
        delta = e1 - e0
        if delta < 0:
            delta += m0
        cpu_w = delta / 1e6 / (t1 - t0)
        gpu_w = fleet_energy.interpolate(gpu, (t0 + t1) / 2)
        out.append(((t0 + t1) / 2, cpu_w + gpu_w))
    return out


def idle_floor(series: list[tuple[float, float]], window_s: float) -> dict[str, float | None]:
    """Lowest mean host power over any ``window_s`` stretch, plus the 5th percentile, as floor estimates."""
    if len(series) < 2:
        return {"window_min_w": None, "p05_w": None}
    best = None
    j = 0
    acc = 0.0
    for i in range(len(series)):
        while j < len(series) and series[j][0] - series[i][0] <= window_s:
            acc += series[j][1]
            j += 1
        count = j - i
        if series[j - 1][0] - series[i][0] >= window_s * 0.9 and count > 0:
            mean = acc / count
            best = mean if best is None else min(best, mean)
        acc -= series[i][1]
    powers = sorted(w for _t, w in series)
    return {"window_min_w": best, "p05_w": powers[max(0, int(0.05 * len(powers)) - 1)]}


def request_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    proofs = result.get("physical_execution_proofs") or {}
    rows = []
    for r in result.get("request_results") or []:
        proof = r.get("physical_execution_proof")
        calls = None
        if isinstance(proof, dict):
            calls = proof.get("phone_call_count")
        if calls is None and isinstance(proofs, dict):
            for key, value in proofs.items():
                if key.startswith(str(r.get("request_id"))) and isinstance(value, dict):
                    calls = value.get("phone_call_count")
        rows.append({"request_id": r.get("request_id"), "model_id": r.get("model_id"),
                     "input_tokens": r.get("input_tokens") or 0, "output_tokens": r.get("output_tokens") or 0,
                     "latency_s": (r.get("actual_latency_us") or 0) / 1e6, "phone_calls": calls or 0})
    return rows


def plan_models(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    models = (report.get("executable") or report).get("models") or {}
    out = {}
    for model_id, row in models.items():
        out[model_id] = {"cpu_only_j_per_token": float(row["cpu_only_energy_j_per_token"]),
                         "plan_j_per_token": float(row["energy_j_per_token"]),
                         "saving_j_per_token": float(row["cpu_only_energy_j_per_token"]) - float(row["energy_j_per_token"])}
    return out


def account(label: str, directory: Path, meter: Path | None, window_s: float,
            idle_w: float | None = None) -> dict[str, Any]:
    base = fleet_energy.account_run(label, directory, meter_path=meter)
    result = json.loads(fleet_energy.find_result(directory).read_text())
    start_s, end_s = result["paid_start_ns"] / 1e9, result["paid_end_ns"] / 1e9
    samples = fleet_energy.load_samples(directory) or []
    series = host_power_series(samples, start_s, end_s)
    floor = idle_floor(series, window_s)
    if idle_w is not None:
        # a dense trace never idles, so its lowest window is not a floor; use the rig's measured idle power instead
        floor = {"window_min_w": idle_w, "p05_w": floor["p05_w"]}
    duration = base["duration_s"]
    host_j = base["desktop"]["host_j"]
    idle_j = floor["window_min_w"] * duration if floor["window_min_w"] is not None and host_j is not None else None
    rows = request_rows(result)
    tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"phone": 0, "no_phone": 0, "requests_phone": 0, "requests": 0})
    for row in rows:
        bucket = "phone" if row["phone_calls"] > 0 else "no_phone"
        tokens[row["model_id"]][bucket] += row["output_tokens"]
        tokens[row["model_id"]]["requests"] += 1
        if row["phone_calls"] > 0:
            tokens[row["model_id"]]["requests_phone"] += 1
    return {"label": label, "directory": str(directory), "duration_s": duration, "host_j": host_j,
            "cpu_package_j": base["desktop"]["cpu_package_j"], "gpu_board_j": base["desktop"]["gpu_board_j"],
            "phones_j": base["phones_j"], "fleet_j": base["fleet_j"],
            "idle_floor_w": floor["window_min_w"], "idle_floor_p05_w": floor["p05_w"], "idle_j": idle_j,
            "dynamic_j": None if idle_j is None else host_j - idle_j,
            "mean_host_w": None if host_j is None else host_j / duration,
            "tokens_by_model": dict(tokens), "requests": rows,
            "phones": {k: {"system_j": v.get("system_j"), "mean_w": v.get("mean_system_w")} for k, v in base["phones"].items()}}


def decompose(baseline: dict[str, Any], treatment: dict[str, Any], plan: dict[str, dict[str, float]]) -> dict[str, Any]:
    full, actual = 0.0, 0.0
    per_model = {}
    for model_id, row in treatment["tokens_by_model"].items():
        if model_id not in plan:
            continue
        s = plan[model_id]["saving_j_per_token"]
        eligible = row["phone"] + row["no_phone"]
        per_model[model_id] = {"tokens_phone": row["phone"], "tokens_no_phone": row["no_phone"],
                               "requests_phone": f"{row['requests_phone']}/{row['requests']}",
                               "predicted_saving_j_per_token": s,
                               "predicted_saving_full_j": s * eligible, "predicted_saving_actual_j": s * row["phone"]}
        full += s * eligible
        actual += s * row["phone"]
    measured_fleet = None if None in (baseline["fleet_j"], treatment["fleet_j"]) else baseline["fleet_j"] - treatment["fleet_j"]
    measured_host = None if None in (baseline["host_j"], treatment["host_j"]) else baseline["host_j"] - treatment["host_j"]
    measured_dyn = None if None in (baseline["dynamic_j"], treatment["dynamic_j"]) else baseline["dynamic_j"] - treatment["dynamic_j"]
    phone_cost = None if None in (baseline["phones_j"], treatment["phones_j"]) else treatment["phones_j"] - baseline["phones_j"]
    return {"per_model": per_model, "predicted_saving_full_use_j": full, "predicted_saving_actual_use_j": actual,
            "lost_to_deferral_j": full - actual,
            "measured_host_saving_j": measured_host, "measured_dynamic_saving_j": measured_dyn,
            "measured_phone_cost_j": phone_cost, "measured_fleet_saving_j": measured_fleet,
            "residual_j": None if measured_fleet is None else actual - measured_fleet,
            "fleet_saving_fraction": None if measured_fleet is None else measured_fleet / baseline["fleet_j"],
            "dynamic_saving_fraction": None if measured_dyn is None or not baseline["dynamic_j"] else measured_dyn / baseline["dynamic_j"],
            "predicted_full_use_fleet_fraction": None if baseline["fleet_j"] is None else full / baseline["fleet_j"],
            "predicted_full_use_dynamic_fraction": None if not baseline["dynamic_j"] else full / baseline["dynamic_j"]}


def kj(v: float | None) -> str:
    return "-" if v is None else f"{v / 1000:.1f}"


def pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:+.1f} %"


def markdown(runs: list[dict[str, Any]], decomposition: dict[str, Any], window_s: float,
             idle_w: float | None = None) -> str:
    lines = [f"Idle floor = lowest {window_s:.0f} s mean host power (RAPL package + GPU board) in the paid window, x duration"
             + (" (overridden by --idle-w)" if idle_w is not None else "") + ".", "",
             "| run | dur s | host kJ | idle floor W | idle kJ | dynamic kJ | phones kJ | fleet kJ | mean host W |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {r['label']} | {r['duration_s']:.0f} | {kj(r['host_j'])} | "
                     f"{'-' if r['idle_floor_w'] is None else f'{r['idle_floor_w']:.1f}'} | {kj(r['idle_j'])} | {kj(r['dynamic_j'])} | "
                     f"{kj(r['phones_j'])} | {kj(r['fleet_j'])} | {'-' if r['mean_host_w'] is None else f'{r['mean_host_w']:.1f}'} |")
    d = decomposition
    lines += ["", "| model | requests w/ phone | decode tokens w/ phone | w/o phone | planner saving J/token | predicted saving full use kJ | at actual use kJ |",
              "|---|---|---|---|---|---|---|"]
    for m, row in d["per_model"].items():
        lines.append(f"| {m} | {row['requests_phone']} | {row['tokens_phone']} | {row['tokens_no_phone']} | "
                     f"{row['predicted_saving_j_per_token']:.2f} | {kj(row['predicted_saving_full_j'])} | {kj(row['predicted_saving_actual_j'])} |")
    lines += ["", "| quantity | kJ | vs baseline fleet | vs baseline dynamic |", "|---|---|---|---|",
              f"| predicted saving at full phone use | {kj(d['predicted_saving_full_use_j'])} | {pct(d['predicted_full_use_fleet_fraction'])} | {pct(d['predicted_full_use_dynamic_fraction'])} |",
              f"| predicted saving at actual phone use | {kj(d['predicted_saving_actual_use_j'])} | | |",
              f"| lost to helper deferral (prediction) | {kj(d['lost_to_deferral_j'])} | | |",
              f"| measured host saving | {kj(d['measured_host_saving_j'])} | | |",
              f"| measured dynamic saving | {kj(d['measured_dynamic_saving_j'])} | | {pct(d['dynamic_saving_fraction'])} |",
              f"| measured extra phone energy | {kj(d['measured_phone_cost_j'])} | | |",
              f"| measured fleet saving | {kj(d['measured_fleet_saving_j'])} | {pct(d['fleet_saving_fraction'])} | |",
              f"| residual (predicted at actual use - measured fleet) | {kj(d['residual_j'])} | | |"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="label=run directory of the all-desktop arm")
    parser.add_argument("--treatment", required=True, help="label=run directory of the phone-assisted arm")
    parser.add_argument("--plan", required=True, type=Path, help="layer-placement REPORT.json used to derive the treatment")
    parser.add_argument("--meter", action="append", default=[], help="label=METER.json (repeatable)")
    parser.add_argument("--idle-window-s", type=float, default=60.0)
    parser.add_argument("--idle-w", type=float, default=None,
                        help="rig idle host power (W) to use as the floor instead of the lowest in-run window")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)
    meters = {k: Path(v) for k, v in fleet_energy.parse_specs(args.meter).items()}
    runs = []
    for spec in (args.baseline, args.treatment):
        label, directory = spec.split("=", 1)
        runs.append(account(label, Path(directory), meters.get(label), args.idle_window_s, args.idle_w))
    plan = plan_models(json.loads(args.plan.read_text()))
    decomposition = decompose(runs[0], runs[1], plan)
    report = {"schema": SCHEMA, "idle_window_s": args.idle_window_s, "idle_w_override": args.idle_w, "plan": str(args.plan),
              "runs": [{k: v for k, v in r.items() if k != "requests"} for r in runs],
              "requests": {r["label"]: r["requests"] for r in runs}, "decomposition": decomposition}
    text = markdown(runs, decomposition, args.idle_window_s, args.idle_w)
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=fleet_energy._json_default))
    if args.md:
        args.md.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
