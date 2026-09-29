#!/usr/bin/env python3
"""Discrete-event replay of a recorded trace through the real scheduler (no hardware).

Arrivals are submitted at their recorded times; dispatch, replans, lease renewals,
transition completion and request completion are driven by the scheduler's own API.
Physical durations come from the recorded run: per-model load time (first load of a
model and later loads separately) and per-request execution time (acquire -> complete,
as measured, including prefill). Snapshots are synthesized from recorded snapshots of
the same residency state (resident large model), retimed to the simulated clock.

Usage: simulate_dispatch.py TREE POLICY [--json OUT]
  POLICY: off | wc | aff | aff:<max_bypasses>:<max_wait_s>
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import sys
import time
from pathlib import Path

from replay_common import WouldBlock, probe_ready, setup

GUARD_US = 250_000
VERBOSE = bool(__import__("os").environ.get("SIM_VERBOSE"))
QUANTUM_US = 2_000_000


def derive_run(run_dir: Path) -> dict:
    """Arrivals, measured durations and residency templates of one recorded run."""
    records = json.loads((run_dir / "SCHEDULER_DECISION_LOG.json").read_text())["records"]
    schedule = json.loads((run_dir / "REPLAY_SCHEDULE.json").read_text())["schedule"]
    index = {row["request_id"]: row["combined_request_index"] for row in schedule}
    arrivals, last_plan, acquired, completed, loads = {}, {}, {}, {}, []
    for row in records:
        rid = row["request_ids"][0]
        kind = row["event_kind"]
        if kind == "DECISION" and rid not in arrivals:
            arrivals[rid] = row["event_time_us"]
        if kind in {"DECISION", "REPLAN"}:
            last_plan[rid] = row["event_time_us"]
        if kind == "ACQUIRED":
            acquired[rid] = row["event_time_us"]
            if row["selected"].get("transition_receipts"):
                model = row["selected"]["executor"]["executor_id"]
                loads.append((rid, model, row["event_time_us"] - last_plan[rid]))
        if kind == "COMPLETED":
            completed[rid] = row["event_time_us"]
    exec_us = {rid: completed[rid] - acquired[rid] for rid in completed}
    by_model: dict[str, list[int]] = {}
    for rid, executor, duration in loads:
        by_model.setdefault(model_key_for_executor(executor), []).append(duration)
    load_us = {
        key: (values[0], int(sum(values[1:]) / len(values[1:])) if values[1:] else values[0])
        for key, values in by_model.items()
    }
    snaps = sorted(
        (path for path in (run_dir / "allsnaps").glob("runtime-*.json")
         if not path.name.endswith(".scheduler.json")),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    templates = {"none": run_dir / "allsnaps" / "request-000.json"}
    for path in snaps:
        rows = json.loads(path.read_text())["residency"]
        keys = {
            model_key_for_executor(row["executor_id"])
            for row in rows if row["device_id"] == "desktop-cuda" and row["state"] == "hot"
        }
        if len(keys) == 1:
            templates.setdefault(keys.pop(), path)
    return {
        "arrivals": sorted(
            ((rid, t, index[rid]) for rid, t in arrivals.items()), key=lambda row: row[1]
        ),
        "exec_us": exec_us, "load_us": load_us, "templates": templates,
    }


def model_key_for_executor(executor_id: str) -> str:
    if "desktop-control" in executor_id:
        return "llama"
    return "gemma" if ":cold:" in executor_id else "qwen" if ":hot:" in executor_id else "llama"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree")
    parser.add_argument("policy")
    parser.add_argument("--json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--run", default="dev2base")
    args = parser.parse_args()
    ctx = setup(args.tree, args.run)
    run = derive_run(ctx["run_dir"])
    ARRIVALS, EXEC_US, LOAD_US = run["arrivals"], run["exec_us"], run["load_us"]
    sch = ctx["scheduler"]
    from research_dev.scheduler import (
        HeterogeneousRuntimeSnapshot, RuntimeExecutionReceipt, RuntimeTransitionReceipt,
    )
    if args.policy != "off":
        from research_dev.scheduler._internal.runtime_dispatch_policy import RuntimeDispatchPolicy
        fields = {"work_conserving_admission": True}
        if args.policy.startswith("aff"):
            fields["model_affinity"] = True
            parts = args.policy.split(":")
            if len(parts) == 3:
                fields["affinity_maximum_bypasses"] = int(parts[1])
                fields["affinity_maximum_wait_us"] = int(float(parts[2]) * 1_000_000)
        sch.configure_runtime_dispatch_policy(RuntimeDispatchPolicy(**fields))
    templates = {key: json.loads(path.read_text()) for key, path in run["templates"].items()}
    state = {"resident": "none", "loaded": set()}
    log = []

    def say(*parts):
        line = " ".join(str(p) for p in parts)
        log.append(line)
        if not args.quiet:
            print(line, flush=True)

    def snapshot(t):
        raw = copy.deepcopy(templates[state["resident"]])
        span = raw["valid_until_us"] - raw["captured_at_us"]
        raw["captured_at_us"] = t
        raw["valid_until_us"] = t + span
        raw["memory"]["captured_at_us"] = t
        raw["memory"]["valid_until_us"] = t + span
        raw["snapshot_id"] = "sim-%d" % t
        raw["memory"]["snapshot_id"] = "sim-memory-%d" % t
        return HeterogeneousRuntimeSnapshot.from_json(raw)

    def epoch_at(t):
        return time.monotonic_ns() - t * 1000

    events = []
    seq = [0]

    def push(t, kind, rid):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, rid))

    for rid, t, idx in ARRIVALS:
        push(t, "arrival", rid)
    submitted, done, active = set(), {}, {}
    retries: dict[str, int] = {}
    starts, loads = {}, []

    def observe(t):
        try:
            changed = sch.observe_automated_runtime_snapshot(snapshot(t), observed_at_us=t)
        except Exception as exc:  # noqa: BLE001
            say("  OBSERVE-FAIL", t, type(exc).__name__, str(exc)[:200])
            return
        if changed:
            say("  observe", t, [c.split(":")[-1] for c in changed])

    def renew(t):
        for rid in sorted(active):
            ticket = sch.runtime_ticket(rid)
            if ticket.dispatch_state != "ACQUIRED":
                continue
            live = [row.token for row in ticket.live_leases]
            horizon = min(ticket.final_reserved_until_us[x] for x in live)
            if horizon - GUARD_US >= t:
                continue
            at = max(horizon - GUARD_US, sch.runtime_ticket(rid).dispatch_receipt.observed_at_us)
            until = max(horizon + QUANTUM_US, t + QUANTUM_US)
            try:
                receipt = sch.extend_runtime_request(rid, at_us=at, reserved_until_us=until)
                replans = getattr(receipt, "cancelled_followers", None)
                say("  renew", rid.split(":")[-1], "->", until, replans or "")
            except Exception as exc:  # noqa: BLE001
                say("  RENEW-FAIL", rid.split(":")[-1], type(exc).__name__, str(exc)[:200])

    def dispatch(t):
        for _ in range(200):
            progressed = False
            for rid in sorted(submitted - set(done) - set(active)):
                queue = sch._runtime_controller.queue
                result = probe_ready(queue, rid, t)
                if isinstance(result, WouldBlock):
                    if result.timeout is not None:
                        push(t + max(1, int(result.timeout * 1_000_000)), "wake", rid)
                    continue
                ticket = sch.wait_runtime_request(rid, epoch_at(t))
                if ticket.dispatch_state == "REPLAN_REQUIRED":
                    receipt = ticket.dispatch_receipt
                    try:
                        new = sch.replan_automated_request(
                            rid, observed_at_us=t, reason=receipt.wake_reason,
                            snapshot=snapshot(t), expected_ticket_id=ticket.ticket_id,
                            expected_queue_generation=receipt.queue_generation,
                        )
                        say("  REPLAN", rid.split(":")[-1], receipt.wake_reason, "->", new.ticket_id.split(":")[-1],
                            "start", new.decision.start_us, "transition", new.transition_status)
                    except Exception as exc:  # noqa: BLE001
                        chain, cause = [], exc
                        while cause is not None:
                            chain.append(type(cause).__name__ + ": " + str(cause)[:400])
                            cause = cause.__cause__ or cause.__context__
                        say("  REPLAN-FAIL", rid.split(":")[-1], " <- ".join(chain))
                        entry = sch._runtime_controller.queue._entries.get(rid)
                        say("     fail-state", None if entry is None else (entry.state, entry.generation, entry.wake_reason),
                            "receipt", receipt.status, receipt.queue_generation, receipt.wake_reason,
                            "ticket", sch.runtime_ticket(rid).dispatch_state)
                        if type(exc).__name__ != "RuntimeReplanRetryRequired":
                            raise
                        retries[rid] = retries.get(rid, 0) + 1
                        if retries[rid] > 4:
                            raise
                        push(t + 1_000_000, "wake", rid)
                        continue
                    progressed = True
                    continue
                if ticket.dispatch_state != "ACQUIRED":
                    continue
                active[rid] = ticket
                starts.setdefault(rid, []).append(t)
                key = model_key_for_executor(ticket.binding.executor_id)
                say("ACQUIRE", t, rid.split(":")[-1], ticket.decision.route_id[-32:],
                    "transition", ticket.transition_status, "wake", ticket.dispatch_receipt.wake_reason)
                if ticket.transition_status == "PENDING":
                    first, later = LOAD_US[key]
                    duration = later if key in state["loaded"] else first
                    push(t + duration, "loaded", rid)
                else:
                    push(t + EXEC_US[rid], "finished", rid)
                progressed = True
            if not progressed:
                return
        raise RuntimeError("dispatch did not converge")

    trace_ids = [x for x in __import__("os").environ.get("SIM_TRACE", "").split(",") if x]

    def trace(t, label):
        if not trace_ids:
            return
        view = sch._runtime_controller.queue.dispatch_order_view()
        for short in trace_ids:
            full = next((k for k in view if k.endswith(":" + short)), None)
            if full is None:
                continue
            row = view[full]
            ticket = sch.runtime_ticket(full)
            say("     trace", label, t, short, row["state"], "barrier", row["residency_transition_barrier"],
                "preds", [p.split(":")[-1] for p in row["predecessor_request_ids"]],
                "start", ticket.decision.start_us, "transitions", len(ticket.execution_plan.transitions))

    while events:
        t, _, kind, rid = heapq.heappop(events)
        trace(t, "before-" + kind + ":" + rid.split(":")[-1])
        renew(t)
        if kind == "arrival":
            request, model_id, _ = ctx["req"](rid)
            observe(t)
            ticket = sch.submit_automated_request(
                request, model_id, snapshot(t), observed_at_us=t, selection_mode="desktop-baseline",
            )
            submitted.add(rid)
            say("SUBMIT", t, rid.split(":")[-1], ticket.decision.route_id[-32:], "start",
                ticket.decision.start_us, "transition", ticket.transition_status)
            if VERBOSE:
                for lease in ticket.decision.leases:
                    say("     lease", lease.resource_id, lease.lanes, lease.start_us, lease.reserved_until_us)
                say("     preds", sch._runtime_controller.queue.dispatch_order_view()[rid]["predecessor_request_ids"]
                    if hasattr(sch._runtime_controller.queue, "dispatch_order_view") else "")
        elif kind == "loaded":
            ticket = sch.runtime_ticket(rid)
            started = ticket.dispatch_receipt.observed_at_us
            participants = {row.device_id: row for row in ticket.binding.participants}
            receipts = tuple(
                RuntimeTransitionReceipt(
                    ticket_id=ticket.ticket_id, request_id=rid,
                    artifact_sha256=ticket.model.artifact_sha256,
                    operator_plan_sha256=ticket.execution_plan.plan_sha256,
                    transition_id=tr.transition_id,
                    executor_id=(ticket.binding.executor_id if tr.executor_id == ticket.binding.executor_id
                                 else participants[tr.device_id].executor_id),
                    endpoint=(ticket.binding.endpoint if tr.executor_id == ticket.binding.executor_id
                              else participants[tr.device_id].endpoint),
                    device_id=tr.device_id, source_state=tr.source_state, target_state=tr.target_state,
                    resource_ids=tr.resource_ids, resource_slots=tr.resource_slots,
                    started_us=started, finished_us=t, status="COMPLETED",
                )
                for tr in ticket.execution_plan.transitions
            )
            sch.record_automated_transition_receipts(rid, receipts)
            key = model_key_for_executor(ticket.binding.executor_id)
            previous = state["resident"]
            state["resident"] = key
            state["loaded"].add(key)
            loads.append((started, t, key, rid))
            say("LOADED", t, rid.split(":")[-1], key, "(was", previous + ")")
            if VERBOSE:
                try:
                    from research_dev.scheduler._unified.automated_requests_ops.observations import (
                        _transitions_are_realized,
                    )
                except ImportError:
                    _transitions_are_realized = None
                preds = sch._runtime_controller.projection_causal_predecessors()
                view = sch._runtime_controller.queue.snapshot()["entry_states"]
                for other in sch._runtime_controller.current_tickets():
                    oid = other.request.request_id
                    if other.dispatch_state in {"COMPLETED", "FAILED", "CANCELLED"}:
                        continue
                    say("     dbg", oid.split(":")[-1], other.dispatch_state, view.get(oid, {}).get("state"), view.get(oid, {}).get("wake_reason"), sch._runtime_controller.queue.dispatch_order_view().get(oid, {}).get("predecessor_request_ids"),
                        "preds", [x.split(":")[-1] for x in preds.get(oid, ())],
                        "transitions", [(tr.source_state, tr.target_state, len(tr.evictions), tr.executor_id) for tr in other.execution_plan.transitions],
                        "realized", None if _transitions_are_realized is None else _transitions_are_realized(snapshot(t), other))
            observe(t)
            push(t + EXEC_US[rid], "finished", rid)
        elif kind == "finished":
            ticket = sch.runtime_execution_ticket(rid)
            sch.release_automated_runtime_capacity(rid, t, expected_ticket_id=ticket.ticket_id)
            receipt = RuntimeExecutionReceipt(
                ticket_id=ticket.ticket_id, request_id=rid,
                artifact_sha256=ticket.model.artifact_sha256,
                operator_plan_sha256=ticket.execution_plan.plan_sha256,
                executor_id=ticket.binding.executor_id, endpoint=ticket.binding.endpoint,
                operator_plan_protocol=ticket.binding.operator_plan_protocol,
                participant_executor_ids=tuple(row.executor_id for row in ticket.binding.participants),
                started_us=starts[rid][-1], finished_us=t,
                output_sha256="sha256:" + "a" * 64, status="COMPLETED",
            )
            sch.complete_automated_request(rid, receipt, snapshot_provider=lambda _t, at: snapshot(at))
            del active[rid]
            done[rid] = t
            say("DONE", t, rid.split(":")[-1])
            observe(t)
        dispatch(t)
        trace(t, "after")
    summary = {
        "policy": args.policy,
        "tree": str(Path(args.tree).resolve()),
        "makespan_us": max(done.values()),
        "loads": [{"start_us": a, "end_us": b, "model": k, "request": r.split(":", 1)[1]} for a, b, k, r in loads],
        "derived": {"load_us": LOAD_US, "exec_us": {k.split(":", 1)[1]: v for k, v in EXEC_US.items()}},
        "requests": {
            rid.split(":", 1)[1]: {
                "arrival_us": dict((e, t) for e, t, _ in ARRIVALS)[rid],
                "acquired_us": starts[rid][-1],
                "done_us": done[rid],
            }
            for rid in sorted(done)
        },
        "dispatch_policy_state": (
            dict(sch.runtime_dispatch_policy_state())
            if hasattr(sch, "runtime_dispatch_policy_state") else None
        ),
        "log": log,
    }
    say("MAKESPAN", summary["makespan_us"], "loads", [(x["model"], x["request"]) for x in summary["loads"]])
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
