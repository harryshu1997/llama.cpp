#!/usr/bin/env python3
"""Offline evaluation of the joint planner against the sequential rule cascade on longtail_eval_v2.

Sections (all offline, no device access):
1. calibration: cost-model parameters re-derived from the recorded run summaries next to the frozen
   values; replay accounting error (the power model integrated over each recorded timeline) and policy
   simulation error (the emulated policy on the trace) against measured host kJ, duration and latency;
2. comparison on the trace with the WS1 fix on both sides: legacy, desktop-only dispatcher, sequential
   cascade, joint planner (budget / depth grid);
3. robustness: perturbed instances (desktop load durations resampled from the measured ones, arrival
   jitter) under the sequential cascade and the planner;
4. small cases and recorded timing forks: sequential, planner, clairvoyant exhaustive optimum, gap g.

Usage: joint_planner_eval.py --out DIR [--instances N] [--seed S]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_dev.scheduler._internal.joint_planner import (  # noqa: E402
    JointPlanner,
    JointPlannerConfig,
    clairvoyant_leaf,
    exhaustive_optimum,
    schedule_space_optimum,
)
from research_dev.scheduler._internal.joint_planner_model import (  # noqa: E402
    CostModel,
    measured_eval_v2_cost_model,
)
from research_dev.scheduler._internal.joint_planner_sim import (  # noqa: E402
    LegacyPolicy,
    SequentialPolicy,
    SimOptions,
    SimRequest,
    SimResult,
    Simulator,
    percentile,
)

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "joint_planner"
RUNS_PATH = DATA / "eval_v2_runs.json"
TOTALS_PATH = DATA / "eval_v2_recorded_totals.json"
MODELS = ("gemma", "qwen", "llama")


def load_runs(path: Path = RUNS_PATH) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="ascii"))
    return {run["label"]: run for run in value["runs"]}


def load_totals(path: Path = TOTALS_PATH) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="ascii"))
    return {run["label"]: run for run in value["runs"]}


def trace_requests(run: Mapping[str, Any]) -> list[SimRequest]:
    return [
        SimRequest(q["id"], q["model"], q["arrival_s"], q["input_tokens"], q["output_tokens"])
        for q in run["requests"]
    ]


# 1. calibration ---------------------------------------------------------------------------------


def _assisted_share(request: Mapping[str, Any]) -> float:
    if not request.get("phone_layers"):
        return 0.0
    return min(1.0, request["phone_calls_per_layer"] / max(1, request["output_tokens"]))


def _decode_interval(request: Mapping[str, Any]) -> tuple[float, float] | None:
    first = request.get("first_token_s")
    if first is None or request["end_s"] <= first:
        return None
    return first, request["end_s"]


def co_decode_batch(run: Mapping[str, Any], request: Mapping[str, Any]) -> int | None:
    """Batch size when every same-model neighbour covers >= 80 % or <= 5 % of this decode interval."""
    mine = _decode_interval(request)
    if mine is None:
        return None
    length = mine[1] - mine[0]
    batch = 1
    for other in run["requests"]:
        if other is request or other["model"] != request["model"]:
            continue
        theirs = _decode_interval(other)
        if theirs is None:
            continue
        overlap = max(0.0, min(mine[1], theirs[1]) - max(mine[0], theirs[0])) / length
        if overlap >= 0.8:
            batch += 1
        elif overlap > 0.05:
            return None
    return batch


def derive_parameters(runs: Mapping[str, Mapping[str, Any]],
                      prefill_power_w: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Re-derive step periods, decode power, loads, prefill and phone provisioning.

    Decode power = (execution-window energy - prefill time x ``prefill_power_w``) / decode time; the
    prefill power defaults to the frozen model's host-only batch-1 power.
    """
    if prefill_power_w is None:
        frozen = measured_eval_v2_cost_model()
        prefill_power_w = {m: frozen.model(m).power_w(1, False) for m in frozen.models}
    periods: dict[str, list[float]] = {}
    powers: dict[str, list[float]] = {}
    for run in runs.values():
        for q in run["requests"]:
            if q["model"] == "llama":
                continue
            batch = co_decode_batch(run, q)
            share = _assisted_share(q)
            if batch is None or 0.05 < share < 0.95:
                continue
            key = "%s-%s-b%d" % (q["model"], "assisted" if share >= 0.95 else "desktop", batch)
            if q.get("decode_period_ms") and q["first_token_s"] - q["exec_start_s"] < 15:
                periods.setdefault(key, []).append(q["decode_period_ms"] / 1e3)
            window = q["end_s"] - q["exec_start_s"]
            prefill_s = q["first_token_s"] - q["exec_start_s"]
            if window > 20 and prefill_s < 15:
                energy = q["exec_cpu_j"] + q["exec_gpu_j"] - prefill_power_w[q["model"]] * prefill_s
                powers.setdefault(key, []).append(energy / (q["end_s"] - q["first_token_s"]))
    loads: dict[str, list[tuple[float, float]]] = {}
    for run in runs.values():
        for load in run["loads"]:
            loads.setdefault(load["model"], []).append(
                (load["end_s"] - load["start_s"], load["cpu_j"] + load["gpu_j"])
            )
    prefill: dict[str, tuple[float, float, int]] = {}
    for model in ("gemma", "qwen"):
        points = [
            (q["input_tokens"], q["first_token_s"] - q["exec_start_s"])
            for run in runs.values() for q in run["requests"]
            if q["model"] == model and q["first_token_s"] is not None
            and q["first_token_s"] - q["exec_start_s"] < 15
        ]
        mean_x = statistics.fmean(x for x, _ in points)
        mean_y = statistics.fmean(y for _, y in points)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points)
        prefill[model] = (mean_y - slope * mean_x, slope, len(points))
    sessions = [
        s["ready_s"] - s["loading_s"] for run in runs.values() for s in run["phone_sessions"]
    ]
    return {
        "periods": {k: (statistics.median(v), len(v)) for k, v in sorted(periods.items())},
        "powers": {k: (statistics.median(v), len(v)) for k, v in sorted(powers.items())},
        "loads": {m: _load_fit(v) for m, v in sorted(loads.items())},
        "prefill": prefill,
        "session_load_s": (statistics.fmean(sessions), len(sessions)),
    }


def _load_fit(points: Sequence[tuple[float, float]]) -> tuple[float, float, float, int]:
    """Mean duration and the least-squares energy line E = fixed + slope x duration."""
    mean_x = statistics.fmean(x for x, _ in points)
    mean_y = statistics.fmean(y for _, y in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sum((x - mean_x) ** 2 for x, _ in points)
    return mean_x, mean_y - slope * mean_x, slope, len(points)


def calibration_rows(runs: Mapping[str, Mapping[str, Any]], cost: CostModel) -> list[list[str]]:
    derived = derive_parameters(runs)
    rows = []
    for key, (value, n) in derived["periods"].items():
        model, placement, batch = key.split("-")
        b = int(batch[1:])
        frozen = cost.model(model).step_s(b, placement == "assisted")
        rows.append(["step " + key, "%.3f s" % value, "%.3f s" % frozen, str(n)])
    for key, (value, n) in derived["powers"].items():
        model, placement, batch = key.split("-")
        b = int(batch[1:])
        frozen = cost.model(model).power_w(b, placement == "assisted")
        rows.append(["decode power " + key, "%.1f W" % value, "%.1f W" % frozen, str(n)])
    for model, (seconds, fixed, slope, n) in derived["loads"].items():
        costs = cost.model(model)
        rows.append(["load " + model, "%.1f s, %.0f J + %.1f W" % (seconds, fixed, slope),
                     "%.1f s, %.0f J + %.1f W" % (costs.load_s, costs.load_fixed_j, costs.load_power_w), str(n)])
    for model, (fixed, slope, n) in derived["prefill"].items():
        costs = cost.model(model)
        rows.append(["prefill " + model, "%.2f s + %.2f ms/tok" % (fixed, slope * 1e3),
                     "%.2f s + %.2f ms/tok" % (costs.prefill_fixed_s, costs.prefill_s_per_token * 1e3), str(n)])
    value, n = derived["session_load_s"]
    rows.append(["OP15 session load", "%.1f s" % value,
                 "%.1f s" % cost.phones[cost.primary_phone].session_load_s, str(n)])
    return rows


def replay_energy_kj(run: Mapping[str, Any], cost: CostModel, *, legacy: bool = False) -> float:
    """Host energy of a recorded timeline under the power model (decisions and durations as recorded)."""
    duration = run["duration_s"]
    loads = [(l["start_s"], l["end_s"], l["model"]) for l in run["loads"]]
    decode = []
    prefill = []
    for q in run["requests"]:
        start = q["exec_start_s"]
        if legacy:
            pf = cost.model(q["model"]).prefill_s(q["input_tokens"])
            first = start + pf
        else:
            first = q.get("first_token_s")
            if first is None:
                continue
            pf = min(first - start, cost.model(q["model"]).prefill_s(q["input_tokens"]))
        prefill.append((start, start + pf, q["model"]))
        if q["end_s"] > first:
            decode.append((first, q["end_s"], q["model"], 0.0 if legacy else _assisted_share(q)))
    points = sorted({0.0, duration, *[t for iv in loads + prefill for t in iv[:2]],
                     *[t for iv in decode for t in iv[:2]]})
    first_load = min((s for s, _, _ in loads), default=0.0)
    energy = sum(cost.model(m).load_fixed_j for _, _, m in loads)
    for a, b in zip(points, points[1:]):
        if b <= a or a >= duration:
            continue
        b = min(b, duration)
        mid = (a + b) / 2
        load = next((m for s, e, m in loads if s <= mid < e), None)
        if load is not None:
            power = cost.model(load).load_power_w
        else:
            active = [(m, share) for s, e, m, share in decode if s <= mid < e]
            if active:
                model = max(set(m for m, _ in active), key=lambda m: sum(1 for x, _ in active if x == m))
                members = [share for m, share in active if m == model]
                costs = cost.model(model)
                batch = len(members)
                share = statistics.fmean(members)
                power = costs.power_w(batch, False)
                if costs.assistable and share > 0:
                    power = share * costs.power_w(batch, True) + (1 - share) * power
            elif any(s <= mid < e for s, e, _ in prefill):
                model = next(m for s, e, m in prefill if s <= mid < e)
                power = cost.model(model).power_w(1, False)
            else:
                power = cost.idle_loaded_w if mid >= first_load else cost.idle_unloaded_w
        energy += power * (b - a)
    return energy / 1e3


def measured_latency(run: Mapping[str, Any]) -> tuple[float, float]:
    values = [q["end_s"] - q["arrival_s"] for q in run["requests"]]
    return percentile(values, 50), percentile(values, 90)


def recorded_loads(run: Mapping[str, Any]) -> tuple[tuple[str, tuple[float, ...]], ...]:
    by_model: dict[str, list[float]] = {}
    for load in run["loads"]:
        by_model.setdefault(load["model"], []).append(load["end_s"] - load["start_s"])
    return tuple((m, tuple(v)) for m, v in sorted(by_model.items()))


# s2a: every OP15 route was THERMAL_LIMIT at the 1272.0 s compile, none at 1610.0 s, and 008 was
# assisted from 1334.6 s (WS1 analysis of the s2a artifacts); the onset is unobserved but precedes
# the Gemma->Qwen handover at 1052.9 s, where the release re-evaluation proposed a Qwen layout that
# was never prepared. Assumed window [1000, 1300) s: it covers the recorded handover and the
# simulated one (1029-1044 s) and ends inside the bracket 1272-1334.6 s.
S2A_THERMAL_WINDOW = (("op15", 1000.0, 1300.0),)


def s2a_options(**extra) -> SimOptions:
    """The s2a conditions: OP15 thermally excluded over the Gemma->Qwen handover."""
    return SimOptions(thermal_exclusions=S2A_THERMAL_WINDOW, **extra)


def run_policy(requests, policy, cost, options=SimOptions()) -> SimResult:
    return Simulator(requests, cost, options).run(policy)


def metrics(result: SimResult) -> dict[str, float]:
    latencies = [c.latency_s for c in result.completions.values()]
    tokens = sum(c.output_tokens for c in result.completions.values())
    return {
        "host_kj": result.host_j / 1e3,
        "fleet_kj": result.fleet_j / 1e3,
        "end_s": result.end_s,
        "p50_s": percentile(latencies, 50),
        "p90_s": percentile(latencies, 90),
        "loads": float(result.loads),
        "provisions": float(result.provisions),
        "assisted_share": sum(c.assisted_tokens for c in result.completions.values()) / max(1, tokens),
        "unfinished": float(len(result.unfinished)),
    }


def policy_error_rows(runs, totals, cost) -> list[list[str]]:
    requests = trace_requests(runs["s2a"])
    rows = []
    cases = [
        ("legacy", LegacyPolicy(), SimOptions(phones_enabled=False), totals["legacy"], "same code"),
        ("desktop-dispatcher", SequentialPolicy(helpers=False), SimOptions(phones_enabled=False),
         totals["desktop-dispatcher"], "09-25 code (no join/hysteresis; host-only so no lane serialization)"),
    ]
    for label in ("s1a", "s1b", "s1c", "s1d"):
        cases.append((label, SequentialPolicy(), SimOptions(), runs[label],
                      "older code (no batch-growth inheritance)"))
    cases.append(("s2a", SequentialPolicy(), s2a_options(), runs["s2a"],
                  "frozen config, OP15 thermal window 1000-1300 s"))
    for label, policy, options, measured, note in cases:
        for variant, opts in (("mean loads", options), ("recorded loads", replace(options, load_s_by_model=recorded_loads(measured)))):
            if variant == "recorded loads" and not measured.get("loads"):
                continue
            result = metrics(run_policy(requests, policy, cost, opts))
            host = measured["host_cpu_kj"] + measured["host_gpu_kj"]
            p50, p90 = measured_latency(measured) if measured.get("requests") else (math.nan, math.nan)
            rows.append([
                label, variant, "%.1f" % host, "%.1f" % result["host_kj"], _pct(result["host_kj"], host),
                "%.0f" % measured["duration_s"], "%.0f" % result["end_s"], _pct(result["end_s"], measured["duration_s"]),
                _num(p50) + " / " + _num(p90), "%.0f / %.0f" % (result["p50_s"], result["p90_s"]), note,
            ])
    return rows


def replay_error_rows(runs, totals, cost) -> list[list[str]]:
    rows = []
    for label in ("s1a", "s1b", "s1c", "s1d", "s2a"):
        run = runs[label]
        measured = run["host_cpu_kj"] + run["host_gpu_kj"]
        predicted = replay_energy_kj(run, cost)
        rows.append([label, "%.1f" % measured, "%.1f" % predicted, _pct(predicted, measured)])
    legacy = totals["legacy"]
    measured = legacy["host_cpu_kj"] + legacy["host_gpu_kj"]
    predicted = replay_energy_kj(legacy, cost, legacy=True)
    rows.append(["legacy", "%.1f" % measured, "%.1f" % predicted, _pct(predicted, measured)])
    return rows


def _pct(value: float, reference: float) -> str:
    return "%+.1f %%" % (100.0 * (value - reference) / reference)


def _num(value: float) -> str:
    return "-" if math.isnan(value) else "%.0f" % value


# 2. comparison ----------------------------------------------------------------------------------


PLANNER_GRID = (
    ("planner d1 25 ms", JointPlannerConfig(depth=1, budget_ms=25.0)),
    ("planner d2 250 ms", JointPlannerConfig(depth=2, budget_ms=250.0)),
    ("planner d3 2 s", JointPlannerConfig(depth=3, budget_ms=2000.0)),
)


def comparison_rows(requests, cost) -> tuple[list[list[str]], dict[str, Any]]:
    arms = [
        ("legacy (all-desktop)", LegacyPolicy(), SimOptions(phones_enabled=False)),
        ("desktop dispatcher", SequentialPolicy(helpers=False), SimOptions(phones_enabled=False)),
        ("sequential, s2a thermal window", SequentialPolicy(), s2a_options()),
        ("planner, s2a thermal window", JointPlanner(PLANNER_GRID[1][1]), s2a_options()),
        ("sequential, no release retry (hypothetical)", SequentialPolicy(retry_reprovision_on_release=False),
         SimOptions()),
        ("planner on the no-retry base", JointPlanner(PLANNER_GRID[1][1], SequentialPolicy(
            retry_reprovision_on_release=False)), SimOptions()),
        ("sequential (cool OP15)", SequentialPolicy(), SimOptions()),
    ]
    arms += [(label, JointPlanner(config), SimOptions()) for label, config in PLANNER_GRID]
    rows = []
    detail = {}
    for label, policy, options in arms:
        started = time.perf_counter()
        result = run_policy(requests, policy, cost, options)
        wall = time.perf_counter() - started
        m = metrics(result)
        plans = getattr(policy, "plans", [])
        deviations = [(round(t, 1), [repr(a) for a in p.actions], round(p.predicted_gain_j)) for t, p in plans if p.deviates]
        max_ms = max((p.elapsed_ms for _, p in plans), default=0.0)
        rows.append([
            label, "%.1f" % m["host_kj"], "%.1f" % m["fleet_kj"], "%.0f" % m["end_s"],
            "%.0f" % m["p50_s"], "%.0f" % m["p90_s"], "%d" % m["loads"], "%.2f" % m["assisted_share"],
            str(len(deviations)) if plans else "-", ("%.0f" % max_ms) if plans else "-", "%.1f" % wall,
        ])
        detail[label] = {
            "metrics": m, "deviations": deviations,
            "completions": {
                k: {"admitted_s": round(v.admitted_s, 1), "decode_start_s": round(v.decode_start_s, 1),
                    "end_s": round(v.end_s, 1), "assisted": round(v.assisted_tokens / v.output_tokens, 2)}
                for k, v in sorted(result.completions.items())
            },
            "decisions": result.decisions,
        }
    return rows, detail


# 3. robustness ----------------------------------------------------------------------------------


def perturbed_instances(requests, runs, totals, count: int, seed: int, jitter_s: float):
    pool: dict[str, list[float]] = {}
    for run in list(runs.values()):
        for load in run["loads"]:
            pool.setdefault(load["model"], []).append(load["end_s"] - load["start_s"])
    rng = random.Random(seed)
    for _ in range(count):
        loads = tuple(
            (model, tuple(rng.choice(values) for _ in range(8))) for model, values in sorted(pool.items())
        )
        moved = [
            replace(r, arrival_s=max(0.0, r.arrival_s + rng.uniform(-jitter_s, jitter_s))) for r in requests
        ]
        yield moved, SimOptions(load_s_by_model=loads)


def robustness_rows(requests, runs, totals, cost, count: int, seed: int) -> list[list[str]]:
    rows = []
    for jitter in (0.0, 10.0):
        samples: dict[str, list[dict[str, float]]] = {
            "sequential (cool OP15)": [], "planner d2 250 ms": [],
            "sequential, s2a thermal window": [], "planner, s2a thermal window": [],
        }
        for moved, options in perturbed_instances(requests, runs, totals, count, seed, jitter):
            hot = replace(options, thermal_exclusions=S2A_THERMAL_WINDOW)
            samples["sequential (cool OP15)"].append(metrics(run_policy(moved, SequentialPolicy(), cost, options)))
            samples["planner d2 250 ms"].append(metrics(run_policy(moved, JointPlanner(PLANNER_GRID[1][1]), cost, options)))
            samples["sequential, s2a thermal window"].append(metrics(run_policy(moved, SequentialPolicy(), cost, hot)))
            samples["planner, s2a thermal window"].append(
                metrics(run_policy(moved, JointPlanner(PLANNER_GRID[1][1]), cost, hot)))
        reference = {
            "planner d2 250 ms": "sequential (cool OP15)",
            "sequential, s2a thermal window": "sequential (cool OP15)",
            "planner, s2a thermal window": "sequential, s2a thermal window",
        }
        for label, values in samples.items():
            host = [v["host_kj"] for v in values]
            p90 = [v["p90_s"] for v in values]
            base_label = reference.get(label)
            if base_label is None:
                wins = "-"
            else:
                base = samples[base_label]
                cheaper = sum(1 for v, b in zip(values, base) if v["host_kj"] < b["host_kj"] - 0.05)
                costlier = sum(1 for v, b in zip(values, base) if v["host_kj"] > b["host_kj"] + 0.05)
                wins = "%d cheaper / %d costlier of %d vs %s" % (cheaper, costlier, len(values), base_label)
            rows.append([
                "%.0f s" % jitter, label, "%.1f" % statistics.fmean(host),
                "%.1f / %.1f" % (percentile(host, 10), percentile(host, 90)),
                "%.0f" % statistics.fmean(v["p50_s"] for v in values),
                "%.0f" % statistics.fmean(p90), "%.1f" % max(host), wins,
            ])
    return rows


# 4. small cases and timing forks ----------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    requests: tuple[SimRequest, ...]
    options: SimOptions = SimOptions()
    sequential: SequentialPolicy = SequentialPolicy()


def _window(requests: Sequence[SimRequest], ids: Sequence[str], shift_s: float) -> tuple[SimRequest, ...]:
    chosen = [r for r in requests if r.request_id in ids]
    return tuple(replace(r, arrival_s=round(r.arrival_s - shift_s, 3)) for r in chosen)


def scenarios(requests: Sequence[SimRequest], cost: CostModel) -> list[Scenario]:
    qwen_load = cost.model("qwen").load_s
    qwen_prefill = cost.model("qwen").prefill_s(160)
    cohort_close = qwen_load + qwen_prefill * 0 + 2.5
    handover = (
        SimRequest("002", "gemma", 0.0, 420, 250),
        SimRequest("006", "gemma", 1.0, 24, 240),
        SimRequest("005", "qwen", 20.0, 24, 265),
        SimRequest("007", "qwen", 60.0, 201, 271),
    )
    gemma = cost.model("gemma")
    handover_end = gemma.load_s + gemma.prefill_s(420) + gemma.prefill_s(24) + 250 * gemma.step_s(2, True)
    return [
        Scenario(
            "F1 join after cohort window",
            "Qwen 001 starts assisted; Qwen 003 arrives 2 s after the 2.5 s cohort window closed, a Gemma "
            "request is queued (s2a 003/004, cj3/cj5 fork)",
            (
                SimRequest("001", "qwen", 0.0, 160, 124),
                SimRequest("002", "gemma", 30.0, 420, 300),
                SimRequest("003", "qwen", round(cohort_close + 2.0, 3), 246, 262),
            ),
        ),
        Scenario(
            "F1b same, pre-09-28 affinity (cj3)",
            "as F1 but the arrival that would only be serialized stays behind the queued Gemma switch "
            "(affinity refused: displacement does not help) -> extra reload",
            (
                SimRequest("001", "qwen", 0.0, 160, 124),
                SimRequest("002", "gemma", 30.0, 420, 300),
                SimRequest("003", "qwen", round(cohort_close + 2.0, 3), 246, 262),
            ),
            sequential=SequentialPolicy(affinity_refuses_serialized_join=True),
        ),
        Scenario(
            "F2 thermal window over the handover (s2a)",
            "a Gemma pair ends while the OP15 is thermally excluded (s2a 1052.9 s): the Qwen re-provision "
            "cannot be prepared until the exclusion ends after the Qwen pair",
            handover,
            SimOptions(thermal_exclusions=(("op15", 100.0, 500.0),)),
        ),
        Scenario(
            "F2b exclusion clears 40 s after the handover",
            "as F2 but the phone becomes admissible again during the Qwen load; the re-evaluation on "
            "clearing (WS1 event_replanning) re-provisions and the pair adopts the helper late",
            handover,
            SimOptions(thermal_exclusions=(("op15", 100.0, handover_end + 40.0),)),
        ),
        Scenario(
            "F2c cohort lease race without a release retry (hypothetical)",
            "the switch is committed while the pair's cohort lease is still held and nothing retries",
            handover,
            sequential=SequentialPolicy(retry_reprovision_on_release=False),
        ),
        Scenario(
            "F3 arrival 2 s after the leader ended (cj5)",
            "Qwen 001 runs, Gemma 002 queued; Qwen 003 arrives 2 s after 001 ends, after the switch "
            "started (only a clairvoyant schedule can hold the switch)",
            (
                SimRequest("001", "qwen", 0.0, 160, 124),
                SimRequest("002", "gemma", 30.0, 420, 300),
                SimRequest("003", "qwen", round(qwen_load + cost.model("qwen").prefill_s(160)
                                                + 124 * cost.model("qwen").step_s(1, True) + 2.0, 3), 246, 262),
            ),
        ),
        Scenario("W1 trace 000-004", "first five eval_v2 requests", _window(requests, ("000", "001", "002", "003", "004"), 0.0)),
        Scenario("W2 trace 005-007", "eval_v2 005-007 (shifted to 0)", _window(requests, ("005", "006", "007"), 735.0)),
        Scenario("W3 trace 008-llama", "eval_v2 008, 009, 010, llama00 (shifted)", _window(requests, ("008", "009", "010", "llama00"), 1272.0)),
        Scenario("W4 trace 011-012", "eval_v2 011, 012 (shifted)", _window(requests, ("011", "012"), 1610.0)),
    ]


def small_case_rows(requests, cost, *, node_limit: int = 60_000) -> tuple[list[list[str]], dict[str, Any]]:
    """Sequential, planner and J* = min(exact phase-schedule enumeration, node-limited event-level
    branch-and-bound, the two runs), all scored over a common window with the same feasibility rules."""
    rows = []
    detail = {}
    config = PLANNER_GRID[1][1]
    end_power = cost.idle_loaded_w + sum(p.idle_power_w for p in cost.phones.values())
    for scenario in scenarios(requests, cost):
        reqs = list(scenario.requests)
        runs = {}
        for label, policy in (("sequential", scenario.sequential), ("planner", JointPlanner(config, scenario.sequential))):
            sim = Simulator(reqs, cost, scenario.options)
            sim.run(policy)
            runs[label] = (sim, clairvoyant_leaf(reqs, cost, scenario.options, sim, config=config,
                                                 sequential=scenario.sequential))
        started = time.perf_counter()
        enumerated = schedule_space_optimum(reqs, cost, scenario.options, config=config,
                                            sequential=scenario.sequential)
        bnb = exhaustive_optimum(reqs, cost, scenario.options, config=config,
                                 sequential=scenario.sequential, node_limit=node_limit)
        search_s = time.perf_counter() - started
        candidates = [("schedule enumeration" if enumerated.source == "schedule" else enumerated.source,
                       enumerated.leaf), ("event B&B", bnb.leaf)]
        feasible = [c for c in candidates if c[1].feasible] or candidates
        star_label, star = min(feasible, key=lambda c: c[1].score)
        window_end = max(runs["sequential"][0].now, runs["planner"][0].now, star.end_s)

        def windowed(energy: float, end_s: float) -> float:
            return energy + end_power * (window_end - end_s)

        j_seq = windowed(runs["sequential"][0].host_j + runs["sequential"][0].phone_j, runs["sequential"][0].now)
        j_plan = windowed(runs["planner"][0].host_j + runs["planner"][0].phone_j, runs["planner"][0].now)
        j_opt = windowed(star.energy_j, star.end_s)
        lat = {k: [c.latency_s for c in v[0].completions.values()] for k, v in runs.items()}
        rows.append([
            scenario.name, "%.2f" % (j_seq / 1e3), "%.2f" % (j_plan / 1e3), "%.2f" % (j_opt / 1e3),
            "%+.1f %%" % (100 * (j_plan - j_opt) / j_opt), "%+.1f %%" % (100 * (j_seq - j_opt) / j_opt),
            "%.0f / %.0f" % (percentile(lat["sequential"], 50), percentile(lat["sequential"], 90)),
            "%.0f / %.0f" % (percentile(lat["planner"], 50), percentile(lat["planner"], 90)),
            "yes" if runs["planner"][1].feasible else "no: " + ",".join(runs["planner"][1].violations),
            star_label, "%d / %s %d" % (enumerated.evaluated, "complete" if bnb.complete else "cut", bnb.nodes),
            "%.1f" % search_s,
        ])
        detail[scenario.name] = {
            "description": scenario.description,
            "requests": [r.__dict__ for r in reqs],
            "sequential_decisions": runs["sequential"][0].decisions,
            "planner_decisions": runs["planner"][0].decisions,
            "optimum_source": star_label,
            "enumerated_schedule": None if enumerated.schedule is None else [p.__dict__ for p in enumerated.schedule],
            "enumerated_decisions": enumerated.decisions,
            "bnb_decisions": bnb.decisions,
            "bnb_complete": bnb.complete,
        }
    return rows, detail


# 5. shadow replay of a recorded run --------------------------------------------------------------


MODEL_IDS = {"gemma": "gemma-4-12b-q40-dequant-f16", "qwen": "qwen3-14b-q4km-dequant-f16",
             "llama": "llama-3.2-1b-instruct-q4_0"}


def _stub_ticket(q: Mapping[str, Any], load: bool):
    from types import SimpleNamespace
    model_id = MODEL_IDS[q["model"]]
    transitions = (SimpleNamespace(transition_id="load:" + model_id + ":x", device_id="desktop-cuda"),) if load else ()
    return SimpleNamespace(
        request=SimpleNamespace(request_id=q["id"], arrival_us=int(q["arrival_s"] * 1e6),
                                input_tokens=q["input_tokens"], output_tokens=q["output_tokens"]),
        model=SimpleNamespace(model_id=model_id, artifact_sha256="sha256:" + model_id),
        decision=SimpleNamespace(start_us=int(q["exec_start_s"] * 1e6), finish_upper_us=0),
        execution_plan=SimpleNamespace(route_id="recorded", transitions=transitions),
        dispatch_state="QUEUED",
    )


def _stub_scheduler(run: Mapping[str, Any], at_s: float):
    """Decode progress and OP15 session residency of the recorded run at ``at_s`` (lock-free fields)."""
    from types import SimpleNamespace
    progress = {}
    helpers = False
    for q in run["requests"]:
        first = q.get("first_token_s")
        if first is not None and first <= at_s < q["end_s"] and q.get("decode_period_ms"):
            done = min(q["output_tokens"] - 1, int((at_s - first) / (q["decode_period_ms"] / 1e3)))
            progress[q["id"]] = (q["output_tokens"], done)
            helpers = helpers or _assisted_share(q) > 0.5
    sessions = {}
    for row in sorted(run["phone_sessions"], key=lambda r: r["loading_s"]):
        if row["loading_s"] > at_s:
            continue
        state = "READY" if row["ready_s"] <= at_s else "LOADING"
        model_id = MODEL_IDS[row["model"]]
        sessions[row["session"]] = SimpleNamespace(
            endpoint="session://op15-phone/" + row["session"], state=state,
            resident_artifact_sha256="sha256:" + model_id,
            active_helper_references=("recorded",) if helpers and state == "READY" else (),
        )
    return SimpleNamespace(_model_placement_controller=SimpleNamespace(
        _request_decode_progress=progress, _phone_session_states=sessions))


def shadow_replay_rows(run: Mapping[str, Any], cost: CostModel) -> tuple[list[list[str]], dict[str, Any]]:
    """Feed the recorded arrival / start / end events through the shadow hook (as the scheduler would)."""
    from research_dev.scheduler._internal.joint_planner_shadow import (
        JointPlannerShadow,
        JointPlannerShadowConfig,
    )
    shadow = JointPlannerShadow(JointPlannerShadowConfig(mode="shadow", budget_ms=250.0), cost=cost,
                                synchronous=True)
    load_start = {load["request"]: load["start_s"] for load in run["loads"]}
    loads = set(load_start)
    events = []
    for q in run["requests"]:
        events.append((q["arrival_s"], 0, "DECISION", "QUEUED", q))
        events.append((min(q["exec_start_s"], load_start.get(q["id"], math.inf)), 1, "ACQUIRED", "ACQUIRED", q))
        events.append((q["end_s"], 2, "COMPLETED", "COMPLETED", q))
    for at_s, _, kind, state, q in sorted(events, key=lambda e: (e[0], e[1], e[4]["id"])):
        ticket = _stub_ticket(q, load=kind == "ACQUIRED" and q["id"] in loads)
        shadow.observe_ticket(_stub_scheduler(run, at_s), kind, ticket, int(at_s * 1e6), state)
    records = shadow.records()
    rows = [
        ["%.1f" % r["event_time_s"], r["event_kind"], r["request_id"], "; ".join(r["sequential_emulated"]) or "-",
         "; ".join(r["planner"]) or "-", "%.0f" % r["predicted_gain_j"]]
        for r in records if "error" not in r and r["deviates"]
    ]
    return rows, shadow.summary()


# report -----------------------------------------------------------------------------------------


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--node-limit", type=int, default=60_000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cost = measured_eval_v2_cost_model()
    runs = load_runs()
    totals = load_totals()
    requests = trace_requests(runs["s2a"])
    sections = []
    calibration = calibration_rows(runs, cost)
    sections.append("## 1a. Cost model: re-derived from the run summaries vs frozen\n\n" + table(
        ["parameter", "derived (median / fit)", "frozen", "n"], calibration))
    replay = replay_error_rows(runs, totals, cost)
    sections.append("## 1b. Replay accounting error (recorded timeline, modelled power)\n\n" + table(
        ["run", "measured host kJ", "predicted kJ", "error"], replay))
    policy = policy_error_rows(runs, totals, cost)
    sections.append("## 1c. Policy simulation error (emulated policy on the trace)\n\n" + table(
        ["run", "load durations", "measured kJ", "sim kJ", "err", "measured s", "sim s", "err",
         "measured p50/p90 s", "sim p50/p90 s", "note"], policy))
    comparison, comparison_detail = comparison_rows(requests, cost)
    sections.append("## 2. longtail_eval_v2, mean load durations\n\n" + table(
        ["arm", "host kJ", "fleet kJ", "end s", "p50 s", "p90 s", "loads", "assisted share",
         "planner deviations", "max plan ms", "wall s"], comparison))
    robustness = robustness_rows(requests, runs, totals, cost, args.instances, args.seed)
    sections.append("## 3. Robustness: %d perturbed instances (loads resampled from the measured ones)\n\n" % args.instances + table(
        ["arrival jitter", "arm", "mean host kJ", "p10 / p90 kJ", "mean p50 s", "mean p90 s", "max kJ",
         "host kJ per instance"], robustness))
    small, small_detail = small_case_rows(requests, cost, node_limit=args.node_limit)
    sections.append("## 4. Small cases and timing forks (fleet J over a common window)\n\n" + table(
        ["case", "sequential kJ", "planner kJ", "J* kJ", "planner gap g", "sequential gap",
         "seq p50/p90 s", "planner p50/p90 s", "planner feasible", "J* from",
         "schedules / B&B nodes", "search s"], small))
    shadow_rows, shadow_summary = shadow_replay_rows(runs["s2a"], cost)
    sections.append(
        "## 5. Shadow replay of s2a (recorded arrival / start / end events through the shadow hook)\n\n"
        "epochs %d, deviations %d, errors %d, max plan %.1f ms\n\n" % (
            shadow_summary["epochs"], shadow_summary["deviations"], shadow_summary["errors"],
            shadow_summary["plan_ms_max"])
        + table(["t s", "event", "request", "sequential (emulated)", "planner", "predicted gain J"], shadow_rows))
    text = "# Joint planner offline evaluation\n\n" + "\n\n".join(sections) + "\n"
    (args.out / "RESULTS.md").write_text(text, encoding="ascii")
    (args.out / "RESULTS.json").write_text(json.dumps({
        "cost_model": cost.to_json(),
        "calibration": calibration, "replay_error": replay, "policy_error": policy,
        "comparison": comparison, "comparison_detail": comparison_detail,
        "robustness": robustness, "small_cases": small, "small_case_detail": small_detail,
        "shadow_replay_s2a": {"deviations": shadow_rows, "summary": shadow_summary},
    }, indent=1, sort_keys=True, default=str, ensure_ascii=True) + "\n", encoding="ascii")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
