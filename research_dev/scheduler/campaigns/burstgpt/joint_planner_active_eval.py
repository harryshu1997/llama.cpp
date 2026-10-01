#!/usr/bin/env python3
"""Offline evaluation of the ACTIVE joint planner (``dispatch_policy.joint_planner`` mode active).

The active mode executes one planner action, the joint join (a same-model request joins the running
phone-assisted batch instead of waiting behind it; the co-tenants run the host policy while it
prefills). Every other planner action stays advisory: the sequential decision is executed. This
script compares, on the measured cost model and simulator of ``joint_planner_eval.py``:

1. sequential cascade vs active mode (joins only) vs the full planner (every action, the WS2 upper
   bound), on longtail_eval_v2 with a cool OP15 and with the s2a thermal window;
2. the join lag: the joiner waits ``lag`` seconds after the decision before the holders have
   yielded (one decode step on the rig; 0 / 1 / 3 s here);
3. robustness on 40 load-resampled instances (arrival jitter 0 / 10 s);
4. the recorded s2a event stream through the real active hook (decisions, fallbacks, plan times).

Usage: joint_planner_active_eval.py --out DIR [--instances N] [--seed S]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_planner_eval as base  # noqa: E402
from research_dev.scheduler._internal.joint_planner import JointPlanner, JointPlannerConfig  # noqa: E402
from research_dev.scheduler._internal.joint_planner_active import (  # noqa: E402
    JointPlannerActive, JointPlannerActiveConfig, classify_plan,
)
from research_dev.scheduler._internal.joint_planner_model import measured_eval_v2_cost_model  # noqa: E402
from research_dev.scheduler._internal.joint_planner_sim import (  # noqa: E402
    Admit, Park, SequentialPolicy, SimOptions, Simulator, Wake, percentile,
)


@dataclass
class ActiveExecutedPolicy:
    """What the active mode executes in the simulator: the planner's plan at every epoch, of which
    only the joint join is applied (the sequential incumbent otherwise). ``lag_s``: the joiner stays
    queued that long after the decision (the holders' window close and host control); if the batch
    is no longer phone-assisted by then, the sequential decision takes over."""

    config: JointPlannerConfig = JointPlannerConfig(budget_ms=250.0)
    lag_s: float = 0.0
    sequential: SequentialPolicy = SequentialPolicy()
    planner: JointPlanner = field(init=False)
    joins: list[tuple[float, str]] = field(default_factory=list)
    fallbacks: list[tuple[float, str]] = field(default_factory=list)
    plans_ms: list[float] = field(default_factory=list)
    _pending: dict[str, float] = field(default_factory=dict)
    _decided: tuple[float, int] | None = None

    def __post_init__(self) -> None:
        self.planner = JointPlanner(self.config, self.sequential)

    def decide(self, sim: Simulator) -> list:
        marker = (sim.now, sim.events)
        if self._decided == marker:
            return []
        self._decided = marker
        plan = self.planner.plan(sim)
        self.plans_ms.append(plan.elapsed_ms)
        actions = list(plan.incumbent)
        server = sim.server
        assisted = bool(server.loading is None and server.decoding() and server.assisted)
        joins = [] if plan.elapsed_ms > self.config.budget_ms else classify_plan(plan, sim)["joins"]
        if plan.deviates and plan.elapsed_ms > self.config.budget_ms:
            self.fallbacks.append((round(sim.now, 1), "BUDGET_EXPIRED"))
        queued = {r.request_id for r in sim.queue}
        for rid in list(self._pending):
            if rid not in queued or not assisted:
                del self._pending[rid]  # admitted meanwhile, or the batch ended: the cascade decides
        due = sorted(rid for rid, at in self._pending.items() if at <= sim.now + 1e-9)
        for rid in due:
            del self._pending[rid]
        new = [rid for rid in joins if rid not in self._pending and rid not in due]
        if self.lag_s > 0:
            for rid in new:
                self._pending[rid] = sim.now + self.lag_s
            now_join = due
        else:
            now_join = sorted(set(new) | set(due))
        waiting = set(self._pending)
        if not (now_join or waiting):
            return actions
        out: list = []
        joined = set(now_join)
        for action in actions:
            if isinstance(action, (Park, Admit)):
                keep = tuple(rid for rid in action.request_ids if rid not in joined and rid not in waiting)
                if keep:
                    out.append(replace(action, request_ids=keep))
                continue
            out.append(action)
        if now_join and assisted:
            free = sim.cost.model(server.resident).slots - len(server.rows)
            ids = tuple(rid for rid in now_join if any(r.request_id == rid and r.model == server.resident
                                                       for r in sim.queue))[:max(free, 0)]
            if ids:
                out.insert(0, Admit(ids))
                self.joins.extend((round(sim.now, 1), rid) for rid in ids)
        for rid, at in self._pending.items():
            out.append(Wake(at))
        return out


def arms(lags=(0.0,)):
    rows = [
        ("sequential", lambda: SequentialPolicy()),
        ("active (joins executed), lag 0 s", lambda: ActiveExecutedPolicy()),
    ]
    rows += [("active, lag %g s" % lag, (lambda lag=lag: ActiveExecutedPolicy(lag_s=lag))) for lag in lags if lag]
    rows.append(("full planner (all actions, WS2 upper bound)",
                 lambda: JointPlanner(JointPlannerConfig(budget_ms=250.0))))
    return rows


def trace_rows(requests, cost):
    rows, detail = [], {}
    for condition, options in (("cool OP15", SimOptions()), ("s2a thermal window", base.s2a_options())):
        for label, make in arms(lags=(1.0, 3.0)):
            policy = make()
            started = time.perf_counter()
            result = base.run_policy(requests, policy, cost, options)
            m = base.metrics(result)
            joins = getattr(policy, "joins", None)
            rows.append([condition, label, "%.1f" % m["host_kj"], "%.1f" % m["fleet_kj"], "%.0f" % m["end_s"],
                         "%.0f" % m["p50_s"], "%.0f" % m["p90_s"], "%.2f" % m["assisted_share"],
                         "-" if joins is None else ", ".join("%s@%.0f" % (rid, t) for t, rid in joins) or "none",
                         "%.1f" % (time.perf_counter() - started)])
            detail[condition + " | " + label] = {
                "metrics": m,
                "joins": joins,
                "completions": {k: {"admitted_s": round(v.admitted_s, 1), "decode_start_s": round(v.decode_start_s, 1),
                                    "end_s": round(v.end_s, 1)} for k, v in sorted(result.completions.items())},
                "plan_ms_max": max(getattr(policy, "plans_ms", []) or [0.0]),
            }
    return rows, detail


def robustness_rows(requests, runs, totals, cost, count, seed):
    rows = []
    for jitter in (0.0, 10.0):
        for condition, thermal in (("cool OP15", ()), ("s2a thermal window", base.S2A_THERMAL_WINDOW)):
            samples = {"sequential": [], "active (joins)": [], "full planner": []}
            for moved, options in base.perturbed_instances(requests, runs, totals, count, seed, jitter):
                options = replace(options, thermal_exclusions=thermal)
                samples["sequential"].append(base.metrics(base.run_policy(moved, SequentialPolicy(), cost, options)))
                samples["active (joins)"].append(base.metrics(base.run_policy(moved, ActiveExecutedPolicy(), cost, options)))
                samples["full planner"].append(base.metrics(base.run_policy(
                    moved, JointPlanner(JointPlannerConfig(budget_ms=250.0)), cost, options)))
            for label, values in samples.items():
                host = [v["host_kj"] for v in values]
                if label == "sequential":
                    wins = "-"
                else:
                    ref = samples["sequential"]
                    cheaper = sum(1 for v, b in zip(values, ref) if v["host_kj"] < b["host_kj"] - 0.05)
                    costlier = sum(1 for v, b in zip(values, ref) if v["host_kj"] > b["host_kj"] + 0.05)
                    wins = "%d cheaper / %d costlier of %d" % (cheaper, costlier, len(values))
                rows.append(["%.0f s" % jitter, condition, label, "%.1f" % statistics.fmean(host),
                             "%.1f / %.1f" % (percentile(host, 10), percentile(host, 90)),
                             "%.0f" % statistics.fmean(v["p50_s"] for v in values),
                             "%.0f" % statistics.fmean(v["p90_s"] for v in values), wins])
    return rows


def active_hook_replay(run, cost):
    """The recorded s2a arrival / acquisition / completion stream through the real active hook
    (stub scheduler: recorded decode progress and OP15 residency; the registration accepts)."""
    active = JointPlannerActive(JointPlannerActiveConfig(), cost=cost)
    registered = []

    class Stub:
        def __init__(self, at_s):
            self._model_placement_controller = base._stub_scheduler(run, at_s)._model_placement_controller

        def register_joint_join_prefill_yield(self, ticket, *, at_us, expires_at_us):
            registered.append((ticket.request.request_id, at_us / 1e6, expires_at_us / 1e6))
            return {"outcome": "REGISTERED", "co_tenant_request_ids": []}

        def clear_joint_join_prefill_yield(self, request_id, *, at_us, reason):
            return False

        def joint_join_prefill_yield_events(self):
            return ()

    load_start = {load["request"]: load["start_s"] for load in run["loads"]}
    events = []
    for q in run["requests"]:
        events.append((q["arrival_s"], 0, "DECISION", "QUEUED", q))
        events.append((min(q["exec_start_s"], load_start.get(q["id"], math.inf)), 1, "ACQUIRED", "ACQUIRED", q))
        events.append((q["end_s"], 2, "COMPLETED", "COMPLETED", q))
    for at_s, _, kind, state, q in sorted(events, key=lambda e: (e[0], e[1], e[4]["id"])):
        ticket = base._stub_ticket(q, load=kind == "ACQUIRED" and q["id"] in load_start)
        active.observe_ticket(Stub(at_s), kind, ticket, int(at_s * 1e6), state)
    rows = []
    for r in active.records():
        if "error" in r or not (r["deviates"] or r["fallback"]):
            continue
        rows.append(["%.1f" % r["event_time_s"], r["event_kind"], r["request_id"], r["view"],
                     "; ".join(r["sequential"]) or "-", "; ".join(r["planner"]) or "-",
                     "%.0f" % r["predicted"]["gain_j"], "; ".join(r["executed"]) or
                     ("deferred to acquisition" if r["deferred_to_acquisition"] else "-"),
                     "-" if not r["fallback"] else r["fallback"]["reason"]])
    return rows, active.summary(), registered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cost = measured_eval_v2_cost_model()
    runs, totals = base.load_runs(), base.load_totals()
    requests = base.trace_requests(runs["s2a"])
    sections = []
    trace, trace_detail = trace_rows(requests, cost)
    sections.append("## 1-2. longtail_eval_v2 (mean loads): sequential vs active vs full planner, join lag\n\n"
                    + base.table(["condition", "arm", "host kJ", "fleet kJ", "end s", "p50 s", "p90 s",
                                  "assisted share", "joins executed (request@s)", "wall s"], trace))
    robust = robustness_rows(requests, runs, totals, cost, args.instances, args.seed)
    sections.append("## 3. Robustness: %d instances, loads resampled from the measured ones\n\n" % args.instances
                    + base.table(["arrival jitter", "condition", "arm", "mean host kJ", "p10 / p90 kJ",
                                  "mean p50 s", "mean p90 s", "vs sequential"], robust))
    replay, summary, registered = active_hook_replay(runs["s2a"], cost)
    sections.append(
        "## 4. Recorded s2a stream through the active hook\n\n"
        "epochs %d, errors %d, deviations %d, executed joins %s, fallbacks %s, max plan %.1f ms, "
        "predicted gain of the executed joins %.0f J\n\n" % (
            summary["epochs"], summary["errors"], summary["deviations"], summary["executed_joins"],
            summary["fallbacks_by_reason"], summary["plan_ms_max"], summary["predicted_gain_executed_j"])
        + base.table(["t s", "event", "request", "view", "sequential", "planner", "gain J", "executed",
                      "fallback"], replay))
    text = "# Active joint planner: offline evaluation\n\n" + "\n\n".join(sections) + "\n"
    (args.out / "ACTIVE_RESULTS.md").write_text(text, encoding="ascii")
    (args.out / "ACTIVE_RESULTS.json").write_text(json.dumps({
        "trace": trace, "trace_detail": trace_detail, "robustness": robust,
        "s2a_active_hook": {"rows": replay, "summary": summary, "registered": registered},
    }, indent=1, sort_keys=True, default=str, ensure_ascii=True) + "\n", encoding="ascii")
    print(text)
    return 0


__all__: Sequence[str] = ("ActiveExecutedPolicy", "active_hook_replay", "main")

if __name__ == "__main__":
    raise SystemExit(main())
