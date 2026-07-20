#!/usr/bin/env python3
"""S19 CP2 fail-closed variable-cohort dispatcher.

Consumes real READY requests and the CP1 measured atlas. Selects a batch size
online from certified measurements only, reserves downstream CUDA-tail and phone
credits before launching, splits oversized queues into measured microbatches,
waits within SLO slack, and forces partial release at the earliest latest-start.

It never selects an unmeasured batch, never launches phone work without
downstream capacity, and never drops or duplicates a request. Every decision is
one s19-dispatch-decision-v1 record. Output is deterministic across
PYTHONHASHSEED (all ordering is explicit sort; serialization uses sort_keys).

This is VARIABLE_COHORT_BATCHING: one static cohort per persistent exchange. It
is NOT token-boundary continuous batching.

The Executor is abstract: MockExecutor for unit/adversarial tests,
DeviceExecutor (device_executor.py) for the real-device live run.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

PHONE_ROUTES = ("OP15_R1", "OP12_R1")
ALL_ROUTES = ("CUDA_R0", "OP15_R1", "OP12_R1", "urgent")

REASON_CODES = {
    "unmeasured_batch", "insufficient_memory", "no_downstream_credit",
    "no_phone_credit", "urgent_small_batch", "r0_fallback",
    "form_largest_useful", "split_microbatch", "wait_for_batch",
    "deadline_release", "stale_epoch", "worker_failure", "conserve_shutdown",
}


# ---------------------------------------------------------------------------
# atlas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AtlasRow:
    route: str
    batch: int
    device: str
    p50_us: int
    p95_us: int
    verdict: str            # ELIGIBLE | ELIGIBLE_SCREEN | INELIGIBLE
    urgent_control: bool = False

    @property
    def selectable(self) -> bool:
        return self.verdict in ("ELIGIBLE", "ELIGIBLE_SCREEN")


class Atlas:
    """Only selectable (ELIGIBLE / ELIGIBLE_SCREEN) rows are exposed."""

    def __init__(self, rows: list[AtlasRow]) -> None:
        self._by_route: dict[str, dict[int, AtlasRow]] = {}
        for r in rows:
            if not r.selectable:
                continue
            self._by_route.setdefault(r.route, {})[r.batch] = r

    def batches(self, route: str, urgent: bool = False) -> list[int]:
        rows = self._by_route.get(route, {})
        out = [b for b, r in rows.items() if r.urgent_control == urgent]
        return sorted(out)

    def has(self, route: str, batch: int) -> bool:
        return batch in self._by_route.get(route, {})

    def row(self, route: str, batch: int) -> Optional[AtlasRow]:
        return self._by_route.get(route, {}).get(batch)

    def conservative_finish_us(self, route: str, batch: int) -> Optional[int]:
        r = self.row(route, batch)
        if r is None:
            return None
        # conservative: p95 plus a 20 percent guard band
        return int(r.p95_us * 1.2)


# ---------------------------------------------------------------------------
# requests, credits, epochs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Request:
    rid: int
    arrival_us: int
    deadline_us: int          # latest finish (absolute, synthetic)
    priority: str             # "high" | "low"
    compat: str               # compatibility key
    urgent: bool = False


@dataclass
class Credits:
    free_kv: dict[str, int]           # per device: free KV slots
    phone_lane: dict[str, int]        # per phone route: remaining exchanges
    usb: dict[str, int]               # per phone route: activation/result credits
    cuda_tail: dict[str, int]         # per route: reserved downstream tail slots

    def snapshot(self) -> dict[str, Any]:
        return {
            "free_kv": dict(sorted(self.free_kv.items())),
            "phone_lane": dict(sorted(self.phone_lane.items())),
            "usb": dict(sorted(self.usb.items())),
            "cuda_tail": dict(sorted(self.cuda_tail.items())),
        }


@dataclass(frozen=True)
class Epochs:
    route: int
    residency: int
    session: int

    def as_dict(self) -> dict[str, int]:
        return {"route": self.route, "residency": self.residency, "session": self.session}


# ---------------------------------------------------------------------------
# executor interface
# ---------------------------------------------------------------------------
@dataclass
class ExecResult:
    ok: bool
    device: str
    worker_pid: Optional[str]
    worker_boot_nonce: Optional[str]
    route_wall_us: int
    token_ids: list[int]
    session_end: str
    placement_ok: bool
    error: Optional[str] = None

    def executed_record(self) -> dict[str, Any]:
        digest = hashlib.sha256(
            json.dumps(self.token_ids, sort_keys=True).encode()).hexdigest()
        return {
            "device": self.device,
            "worker_pid": self.worker_pid,
            "worker_boot_nonce": self.worker_boot_nonce,
            "route_wall_us": self.route_wall_us,
            "token_ids_sha256": digest,
            "session_end": self.session_end,
            "placement_ok": self.placement_ok,
        }


class Executor:
    def execute(self, route: str, batch: int, request_ids: list[int],
                epochs: Epochs, session_end: str) -> ExecResult:
        raise NotImplementedError


class MockExecutor(Executor):
    """Deterministic executor for tests. `fail_on` forces a worker failure."""

    def __init__(self, fail_on: Optional[Callable[[str, int, list[int]], bool]] = None,
                 tokens: Optional[list[int]] = None) -> None:
        self.fail_on = fail_on
        self.tokens = tokens if tokens is not None else [45518, 107, 100, 45518, 107, 101, 1509, 7412]
        self.calls: list[dict[str, Any]] = []

    def execute(self, route, batch, request_ids, epochs, session_end):
        self.calls.append({"route": route, "batch": batch,
                           "request_ids": list(request_ids), "session_end": session_end})
        if self.fail_on and self.fail_on(route, batch, request_ids):
            return ExecResult(False, route, None, None, 0, [], session_end, False,
                              error="worker_failure")
        return ExecResult(True, route, "mock-pid", "mock-nonce", 1000 * batch,
                          list(self.tokens), session_end, True)


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
@dataclass
class DispatchConfig:
    urgent_slack_us: int = 1_500_000       # <= this remaining slack => urgent
    high_priority_route: str = "CUDA_R0"
    low_priority_route_order: tuple = PHONE_ROUTES
    r0_fallback: str = "CUDA_R0"
    guard_us: int = 0                       # extra scheduling guard


class Dispatcher:
    def __init__(self, atlas: Atlas, credits: Credits, executor: Executor,
                 epochs: dict[str, Epochs], config: Optional[DispatchConfig] = None,
                 route_device: Optional[dict[str, str]] = None,
                 live_epochs: Optional[dict[str, Epochs]] = None) -> None:
        self.atlas = atlas
        self.credits = credits
        self.executor = executor
        self.epochs = dict(epochs)          # per-route committed epochs
        # the live worker generation per route; a committed epoch that does not
        # match the live generation is stale and rejected fail-closed.
        self.live_epochs = dict(live_epochs) if live_epochs is not None else dict(epochs)
        self.config = config or DispatchConfig()
        self.route_device = route_device or {
            "CUDA_R0": "CUDA0", "OP15_R1": "3C15AU002CL00000",
            "OP12_R1": "5ae7a43d", "urgent": "CUDA0"}
        self.pending: dict[int, Request] = {}
        self.terminals: dict[int, str] = {}   # rid -> outcome
        self.decisions: list[dict[str, Any]] = []
        self.epoch_ix = 0
        self.now_us = 0

    # -- decision log helper ------------------------------------------------
    def _emit(self, event, ready_before, compat, priority, earliest_slo,
              route, batch, cohort, microbatch_plan, reason, epochs, outcome,
              executed) -> None:
        rec = {
            "schema": "s19-dispatch-decision-v1",
            "epoch": self.epoch_ix,
            "now_us": self.now_us,
            "event": event,
            "ready_before": sorted(ready_before),
            "compatibility_key": compat,
            "priority": priority,
            "earliest_slo_us": earliest_slo,
            "selected_route": route,
            "selected_batch": batch,
            "cohort_request_ids": sorted(cohort),
            "microbatch_plan": list(microbatch_plan),
            "reason_code": reason,
            "credits_after": self.credits.snapshot(),
            "epochs": epochs.as_dict() if epochs else None,
            "outcome": outcome,
            "executed": executed,
        }
        assert reason in REASON_CODES, f"unknown reason {reason}"
        self.decisions.append(rec)
        self.epoch_ix += 1

    # -- route/credit feasibility ------------------------------------------
    def _phone_feasible(self, route: str, batch: int) -> Optional[str]:
        """Return None if feasible, else a fail-closed reason code."""
        if not self.atlas.has(route, batch):
            return "unmeasured_batch"
        device = self.route_device[route]
        if self.credits.free_kv.get(device, 0) < batch:
            return "insufficient_memory"
        if self.credits.cuda_tail.get(route, 0) <= 0:
            return "no_downstream_credit"
        if self.credits.phone_lane.get(route, 0) <= 0:
            return "no_phone_credit"
        if self.credits.usb.get(route, 0) <= 0:
            return "no_phone_credit"
        return None

    def _cuda_feasible(self, route: str, batch: int) -> Optional[str]:
        if not self.atlas.has(route, batch):
            return "unmeasured_batch"
        device = self.route_device[route]
        if self.credits.free_kv.get(device, 0) < batch:
            return "insufficient_memory"
        return None

    def _largest_useful(self, route: str, budget_us: int, group_size: int,
                        urgent: bool = False) -> Optional[int]:
        """Largest certified batch that fits the deadline budget and group."""
        candidates = [b for b in self.atlas.batches(route, urgent=urgent)
                      if b <= group_size]
        candidates.sort(reverse=True)
        for b in candidates:
            fin = self.atlas.conservative_finish_us(route, b)
            if fin is not None and fin <= budget_us:
                return b
        return None

    def _smallest_feasible(self, route: str, budget_us: int,
                           urgent: bool = False) -> Optional[int]:
        for b in self.atlas.batches(route, urgent=urgent):
            fin = self.atlas.conservative_finish_us(route, b)
            if fin is not None and fin <= budget_us:
                return b
        return None

    # -- committing a cohort ------------------------------------------------
    def _reserve_and_execute(self, route, batch, cohort, priority, compat,
                             earliest_slo, ready_before, reason, event,
                             microbatch_plan, session_end="DETACH"):
        device = self.route_device[route]
        epochs = self.epochs.get(route)
        # stale-epoch guard: the committed route epoch must exist and must match
        # the live worker generation. Absence or mismatch is a fail-closed reject.
        if epochs is None or epochs != self.live_epochs.get(route):
            self._emit(event, ready_before, compat, priority, earliest_slo, route,
                       batch, [], microbatch_plan, "stale_epoch",
                       epochs if epochs is not None else None, "rejected", None)
            for rid in cohort:
                self.terminals[rid] = "rejected_terminal"
                self.pending.pop(rid, None)
            return
        # consume credits
        self.credits.free_kv[device] = self.credits.free_kv.get(device, 0) - batch
        if route in PHONE_ROUTES:
            self.credits.phone_lane[route] -= 1
            self.credits.usb[route] -= 1
            self.credits.cuda_tail[route] -= 1
        res = self.executor.execute(route, batch, sorted(cohort), epochs, session_end)
        # release the transient KV slots after the exchange completes
        self.credits.free_kv[device] = self.credits.free_kv.get(device, 0) + batch
        if not res.ok:
            # worker failure -> fall back to R0 if possible, else terminal failure
            fb_reason = "worker_failure"
            fb_route = self.config.r0_fallback
            fb = self._cuda_feasible(fb_route, batch) if self.atlas.has(fb_route, batch) else "unmeasured_batch"
            if fb is None:
                self._emit(event, ready_before, compat, priority, earliest_slo,
                           route, batch, sorted(cohort), microbatch_plan,
                           "worker_failure", epochs, "fell_back", None)
                self._reserve_and_execute(fb_route, batch, cohort, priority, compat,
                                          earliest_slo, ready_before, "r0_fallback",
                                          event, microbatch_plan, session_end)
                return
            self._emit(event, ready_before, compat, priority, earliest_slo, route,
                       batch, sorted(cohort), microbatch_plan, "worker_failure",
                       epochs, "terminal_failure", None)
            for rid in cohort:
                self.terminals[rid] = "terminal_failure"
                self.pending.pop(rid, None)
            return
        for rid in cohort:
            self.terminals[rid] = "completed"
            self.pending.pop(rid, None)
        self._emit(event, ready_before, compat, priority, earliest_slo, route,
                   batch, sorted(cohort), microbatch_plan, reason, epochs,
                   "dispatched", res.executed_record())

    # -- public API ---------------------------------------------------------
    def on_arrival(self, req: Request, now_us: int) -> None:
        self.now_us = now_us
        self.pending[req.rid] = req
        self._emit("arrival", sorted(self.pending), req.compat, req.priority,
                   req.deadline_us, None, None, [], [], "wait_for_batch", None,
                   "waited", None)

    def _select_route(self, priority: str, compat: str) -> tuple[str, bool]:
        """Return (route, is_phone). Deterministic phone spread by compat.

        Route selection does NOT pre-filter by credit: an under-credited phone is
        still attempted so the feasibility check emits an explicit refusal before
        the R0 fallback. Fail-closed to R0 when no phone route has an atlas.
        """
        if priority == "high":
            return self.config.high_priority_route, False
        avail = [r for r in self.config.low_priority_route_order if self.atlas.batches(r)]
        if not avail:
            return self.config.r0_fallback, False
        idx = int(hashlib.sha256(compat.encode()).hexdigest(), 16) % len(avail)
        return avail[idx], True

    def form_cohorts(self, now_us: int, drain: bool = False) -> None:
        """Group ready requests by compatibility and try to release."""
        self.now_us = now_us
        # group by compatibility key, deterministic order
        groups: dict[str, list[Request]] = {}
        for rid in sorted(self.pending):
            req = self.pending[rid]
            groups.setdefault(req.compat, []).append(req)
        for compat in sorted(groups):
            self._release_group(compat, groups[compat], now_us, drain)

    def _release_group(self, compat: str, reqs: list[Request], now_us: int,
                       drain: bool) -> None:
        reqs = [self.pending[r.rid] for r in reqs if r.rid in self.pending]
        if not reqs:
            return
        priority = reqs[0].priority
        earliest_slo = min(r.deadline_us for r in reqs)
        budget = max(0, earliest_slo - now_us - self.config.guard_us)
        ready_before = [r.rid for r in reqs]
        group_size = len(reqs)
        is_urgent = any(r.urgent for r in reqs) or budget <= self.config.urgent_slack_us

        route, is_phone = self._select_route(priority, compat)

        # URGENT: smallest certified feasible batch, or urgent control, or R0
        if is_urgent:
            self._release_urgent(compat, reqs, route, is_phone, priority,
                                 earliest_slo, budget, ready_before, now_us)
            return

        # MEMORY-BOUND decode: largest useful measured batch
        preferred = self._largest_useful(route, budget, group_size)
        if preferred is None:
            # nothing fits deadline on this route -> check downstream/credit or R0
            self._fallback_or_wait(compat, reqs, route, is_phone, priority,
                                   earliest_slo, budget, ready_before, drain, now_us)
            return

        # if the group is smaller than the smallest certified batch, wait within slack
        min_batch = min(self.atlas.batches(route)) if self.atlas.batches(route) else None
        if min_batch is not None and group_size < min_batch and not drain:
            fin = self.atlas.conservative_finish_us(route, min_batch)
            if fin is not None and fin <= budget:
                self._emit("form_cohort", ready_before, compat, priority,
                           earliest_slo, route, None, [], [], "wait_for_batch",
                           self.epochs.get(route), "waited", None)
                return

        # feasibility (credit/memory) for the preferred batch
        reason = (self._phone_feasible(route, preferred) if is_phone
                  else self._cuda_feasible(route, preferred))
        if reason is not None:
            self._fallback_or_wait(compat, reqs, route, is_phone, priority,
                                   earliest_slo, budget, ready_before, drain, now_us,
                                   deny_reason=reason)
            return

        # form largest useful cohort; split the remainder into measured microbatches
        cohort_ids = sorted(r.rid for r in reqs)[:preferred]
        remainder = group_size - preferred
        plan = self._microbatch_plan(route, remainder, budget)
        reason_code = "split_microbatch" if remainder > 0 else "form_largest_useful"
        self._reserve_and_execute(route, preferred, cohort_ids, priority, compat,
                                  earliest_slo, ready_before, reason_code,
                                  "form_cohort", [preferred] + plan)
        # recurse on the remainder within the same epoch
        if self.pending and any(r.rid in self.pending for r in reqs):
            self._release_group(compat, [self.pending[r.rid] for r in reqs
                                         if r.rid in self.pending], now_us, drain)

    def _microbatch_plan(self, route: str, remainder: int, budget: int) -> list[int]:
        plan: list[int] = []
        left = remainder
        avail = sorted(self.atlas.batches(route), reverse=True)
        while left > 0 and avail:
            chosen = None
            for b in avail:
                if b <= left:
                    chosen = b
                    break
            if chosen is None:
                chosen = avail[-1]     # smallest certified batch covers a small tail
            plan.append(chosen)
            left -= chosen
        return plan

    def _release_urgent(self, compat, reqs, route, is_phone, priority,
                        earliest_slo, budget, ready_before, now_us) -> None:
        # try urgent control batches (B1/B2) first if certified for a route,
        # else smallest certified normal batch, else R0 fallback
        urgent_route = "urgent"
        if self.atlas.batches(urgent_route, urgent=True):
            b = self._smallest_feasible(urgent_route, budget, urgent=True)
            if b is not None:
                cohort = sorted(r.rid for r in reqs)[:b]
                # urgent control runs on the selected A6000 (route metadata)
                if "urgent" not in self.epochs:
                    self.epochs["urgent"] = self.epochs.get("CUDA_R0")
                self._reserve_and_execute(urgent_route, b, cohort, priority, compat,
                                          earliest_slo, ready_before,
                                          "urgent_small_batch", "form_cohort", [b])
                if self.pending and any(r.rid in self.pending for r in reqs):
                    self._release_group(compat, [self.pending[r.rid] for r in reqs
                                                 if r.rid in self.pending], now_us, False)
                return
        # smallest certified normal batch on R0 (never a phone under tight SLO)
        r0 = self.config.r0_fallback
        b = self._smallest_feasible(r0, budget)
        if b is not None and self._cuda_feasible(r0, b) is None and len(reqs) >= b:
            cohort = sorted(r.rid for r in reqs)[:b]
            self._reserve_and_execute(r0, b, cohort, priority, compat, earliest_slo,
                                      ready_before, "r0_fallback", "form_cohort", [b])
            if self.pending and any(r.rid in self.pending for r in reqs):
                self._release_group(compat, [self.pending[r.rid] for r in reqs
                                             if r.rid in self.pending], now_us, False)
            return
        # deadline release: smallest batch that the group can fill right now
        self._deadline_release(compat, reqs, r0, priority, earliest_slo,
                               ready_before, now_us)

    def _fallback_or_wait(self, compat, reqs, route, is_phone, priority,
                          earliest_slo, budget, ready_before, drain, now_us,
                          deny_reason=None) -> None:
        # record the phone refusal reason, then try R0 fallback
        if is_phone and deny_reason is not None:
            self._emit("form_cohort", ready_before, compat, priority, earliest_slo,
                       route, None, [], [], deny_reason, self.epochs.get(route),
                       "rejected", None)
        r0 = self.config.r0_fallback
        group_size = len(reqs)
        preferred = self._largest_useful(r0, budget, group_size)
        if preferred is not None and self._cuda_feasible(r0, preferred) is None:
            cohort = sorted(r.rid for r in reqs)[:preferred]
            self._reserve_and_execute(r0, preferred, cohort, priority, compat,
                                      earliest_slo, ready_before, "r0_fallback",
                                      "form_cohort", [preferred])
            if self.pending and any(r.rid in self.pending for r in reqs):
                self._release_group(compat, [self.pending[r.rid] for r in reqs
                                             if r.rid in self.pending], now_us, drain)
            return
        if drain:
            self._deadline_release(compat, reqs, r0, priority, earliest_slo,
                                   ready_before, now_us)
        else:
            self._emit("form_cohort", ready_before, compat, priority, earliest_slo,
                       route, None, [], [], "wait_for_batch", self.epochs.get(route),
                       "waited", None)

    def _deadline_release(self, compat, reqs, route, priority, earliest_slo,
                          ready_before, now_us) -> None:
        group_size = len(reqs)
        # largest certified batch that the group can fill NOW (ignore deadline budget:
        # the deadline already forces release)
        avail = [b for b in self.atlas.batches(route) if b <= group_size]
        if not avail:
            # cannot even fill the smallest certified batch -> terminal (fail closed)
            self._emit("release", ready_before, compat, priority, earliest_slo,
                       route, None, sorted(r.rid for r in reqs), [], "deadline_release",
                       self.epochs.get(route), "terminal_failure", None)
            for r in reqs:
                self.terminals[r.rid] = "terminal_failure"
                self.pending.pop(r.rid, None)
            return
        b = max(avail)
        if self._cuda_feasible(route, b) is not None:
            self._emit("release", ready_before, compat, priority, earliest_slo,
                       route, b, [], [], "insufficient_memory", self.epochs.get(route),
                       "rejected", None)
            for r in reqs:
                self.terminals[r.rid] = "rejected_terminal"
                self.pending.pop(r.rid, None)
            return
        cohort = sorted(r.rid for r in reqs)[:b]
        self._reserve_and_execute(route, b, cohort, priority, compat, earliest_slo,
                                  ready_before, "deadline_release", "release", [b])
        if self.pending and any(r.rid in self.pending for r in reqs):
            self._deadline_release(compat, [self.pending[r.rid] for r in reqs
                                            if r.rid in self.pending], route,
                                   priority, earliest_slo, ready_before, now_us)

    def shutdown(self, now_us: int) -> None:
        """Drain: force-release everything, then verify conservation."""
        self.now_us = now_us
        self.form_cohorts(now_us, drain=True)
        # anything still pending after a drain is a terminal (fail closed)
        leftover = sorted(self.pending)
        if leftover:
            for rid in leftover:
                req = self.pending[rid]
                self._emit("shutdown", leftover, req.compat, req.priority,
                           req.deadline_us, None, None, [rid], [], "conserve_shutdown",
                           None, "terminal_failure", None)
                self.terminals[rid] = "terminal_failure"
            self.pending.clear()
        else:
            self._emit("shutdown", [], "*", "low", now_us, None, None, [], [],
                       "conserve_shutdown", None, "completed", None)


# ---------------------------------------------------------------------------
# schedule driver
# ---------------------------------------------------------------------------
def run_schedule(events: list[dict], atlas: Atlas, credits: Credits,
                 executor: Executor, epochs: dict[str, Epochs],
                 config: Optional[DispatchConfig] = None,
                 route_device: Optional[dict[str, str]] = None) -> Dispatcher:
    """events: sorted list of {kind: 'arrival'|'tick'|'shutdown', now_us, request?}."""
    d = Dispatcher(atlas, credits, executor, epochs, config, route_device)
    for ev in sorted(events, key=lambda e: (e["now_us"], 0 if e["kind"] == "arrival" else 1)):
        if ev["kind"] == "arrival":
            d.on_arrival(ev["request"], ev["now_us"])
        elif ev["kind"] == "tick":
            d.form_cohorts(ev["now_us"], drain=ev.get("drain", False))
        elif ev["kind"] == "shutdown":
            d.shutdown(ev["now_us"])
    return d


def conservation_report(d: Dispatcher, all_rids: list[int]) -> dict[str, Any]:
    terminal = set(d.terminals)
    all_set = set(all_rids)
    missing = sorted(all_set - terminal)
    extra = sorted(terminal - all_set)
    # count double completions from decisions: each rid should appear in exactly one
    # dispatched/fell_back/terminal cohort
    committed: dict[int, int] = {}
    for dec in d.decisions:
        if dec["outcome"] in ("dispatched", "terminal_failure", "rejected") and dec["cohort_request_ids"]:
            for rid in dec["cohort_request_ids"]:
                committed[rid] = committed.get(rid, 0) + 1
    duplicates = sorted(rid for rid, c in committed.items() if c > 1)
    return {
        "total": len(all_set),
        "terminal": len(terminal),
        "missing": missing,
        "extra": extra,
        "duplicates": duplicates,
        "conserved": not missing and not extra and not duplicates,
        "by_outcome": _count_outcomes(d),
    }


def _count_outcomes(d: Dispatcher) -> dict[str, int]:
    out: dict[str, int] = {}
    for rid, outcome in d.terminals.items():
        out[outcome] = out.get(outcome, 0) + 1
    return dict(sorted(out.items()))
