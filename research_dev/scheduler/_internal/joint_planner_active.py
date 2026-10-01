"""Active mode of the joint planner: plan at every decision epoch and execute the joint join.

Opt-in through ``dispatch_policy.joint_planner = {"mode": "active", ...}``. The hook is the shadow's
(called after every runtime journal record, under the scheduler's runtime lock), but planning runs
synchronously within a per-epoch time budget, and one planner decision is executed:

* joint join (executed): a same-model request that the cascade admits into a server whose batch
  decodes under a phone policy physically waits there until the holder finishes (llama-server cannot
  batch its prompt with a phone-policy slot; ``adaptive_decode_ops.prefill_yield``). When the planner
  joins it instead of parking it (the sequential decision), the co-tenants run the host policy while it
  prefills; at its first token server policy coherence moves it onto the group's phone policy, the
  co-tenants return to it, both rows share one forward, one coalesced phone call and the holder's
  helper lease tokens (``SERVER_HELPER_LEASES_SHARED``). The decision is taken on the state just
  before the acquisition (the joiner still queued), so it is fresh when it is executed.
* advisory (logged, not executed in this phase): admitting a request the cascade did not admit,
  model switch timing and holds, phone re-provisioning and early helper release, helper use of a new
  batch, late adoption. Executing them would bypass admission barriers, the residency hysteresis or
  the re-provisioning gates; the cascade's decision stands and the record says so.

Every epoch records the planner decision, the sequential alternative, both predicted energies and
scores, the executed choice and, when a planner decision is not executed, a ``JOINT_PLANNER_FALLBACK``
with its reason. A plan that exceeded the budget, a planner failure, a capture failure or a join the
scheduler refuses (coherence off, joiner not adaptive, no co-tenant session, ...) falls back to the
sequential decision. The planner never calls admission, selection, lease or residency code: the
only effect of an executed join is the prefill yield, after which every phone control is an ordinary
one (helper readiness, identity, lease and thermal checks unchanged).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import time
from typing import Any, Callable, Mapping, Sequence

from .joint_planner import JointPlannerConfig, JointPlannerError, PlanResult
from .joint_planner_model import CostModel
from .joint_planner_shadow import (
    TERMINAL_EVENTS,
    JointPlannerShadow,
    JointPlannerShadowConfig,
    JointPlannerShadowError,
    ShadowEpoch,
    default_model_key,
    simulator_from_epoch,
)
from .joint_planner_sim import Admit, DeferProvision, Park, Simulator, Wake, _describe


JOINT_PLANNER_ACTIVE_SCHEMA = "research-scheduler-joint-planner-active-v1"
FALLBACK = "JOINT_PLANNER_FALLBACK"
_ACTIVE_FIELDS = ("mode", "horizon_s", "budget_ms", "depth", "objective", "max_prefill_yield_s")
# A DECISION whose live start is within this of the event is an admission (ACQUIRED follows).
ADMISSION_TOLERANCE_S = 5.0
# Prefill yield bound: margin + factor x the modelled prefill, capped by max_prefill_yield_s.
PREFILL_YIELD_MARGIN_S = 10.0
PREFILL_YIELD_FACTOR = 3.0


class JointPlannerActiveError(JointPlannerShadowError):
    pass


@dataclass(frozen=True)
class JointPlannerActiveConfig:
    mode: str = "active"
    horizon_s: float = 3600.0
    budget_ms: float = 50.0
    depth: int = 2
    objective: str = "fleet"
    max_prefill_yield_s: float = 60.0

    def __post_init__(self) -> None:
        if self.mode != "active":
            raise JointPlannerActiveError("joint planner active configuration needs mode 'active'")
        if type(self.max_prefill_yield_s) not in (int, float) or not self.max_prefill_yield_s > 0:
            raise JointPlannerActiveError("joint planner max_prefill_yield_s must be a positive number")
        self.planner_config()

    def planner_config(self) -> JointPlannerConfig:
        try:
            return JointPlannerConfig(
                horizon_s=float(self.horizon_s), budget_ms=float(self.budget_ms),
                depth=self.depth, objective=self.objective,
            )
        except JointPlannerError as exc:
            raise JointPlannerActiveError(str(exc)) from exc

    @classmethod
    def from_json(cls, value: object) -> "JointPlannerActiveConfig":
        if not isinstance(value, Mapping):
            raise JointPlannerActiveError("joint planner configuration must be an object")
        unknown = sorted(set(value) - set(_ACTIVE_FIELDS))
        if unknown:
            raise JointPlannerActiveError("joint planner configuration has unknown fields: " + ", ".join(unknown))
        for name in ("horizon_s", "budget_ms", "max_prefill_yield_s"):
            if name in value and (type(value[name]) not in (int, float) or not value[name] > 0):
                raise JointPlannerActiveError("joint planner " + name + " must be a positive number")
        if "depth" in value and (type(value["depth"]) is not int or value["depth"] < 0):
            raise JointPlannerActiveError("joint planner depth must be a nonnegative integer")
        if "objective" in value and type(value["objective"]) is not str:
            raise JointPlannerActiveError("joint planner objective must be a string")
        return cls(**dict(value))

    def to_json(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _ACTIVE_FIELDS}


def joint_planner_config_from_json(value: object) -> JointPlannerShadowConfig | JointPlannerActiveConfig:
    """``dispatch_policy.joint_planner``: mode "shadow" or "active" (errors are JointPlannerShadowError)."""
    if isinstance(value, Mapping) and value.get("mode") == "active":
        return JointPlannerActiveConfig.from_json(value)
    return JointPlannerShadowConfig.from_json(value)


def joint_planner_from_policy_json(
    value: Mapping[str, object],
) -> tuple[dict[str, object], JointPlannerShadowConfig | JointPlannerActiveConfig | None]:
    """Split ``joint_planner`` off a dispatch policy object (the rest is the RuntimeDispatchPolicy)."""
    rest = {k: v for k, v in value.items() if k != "joint_planner"}
    if "joint_planner" not in value:
        return rest, None
    return rest, joint_planner_config_from_json(value["joint_planner"])


# Plan classification ------------------------------------------------------------------------------


def _atoms(actions: Sequence[object]) -> set[str]:
    """Per-request admission atoms plus the description of every other action (waits excluded)."""
    atoms: set[str] = set()
    for action in actions:
        if isinstance(action, (Wake, DeferProvision)):
            continue
        if isinstance(action, Admit):
            flag = "" if action.assisted is None else (":assisted" if action.assisted else ":desktop")
            atoms.update("admit:" + rid + flag for rid in action.request_ids)
        elif isinstance(action, Park):
            atoms.update("park:" + rid for rid in action.request_ids)
        else:
            atoms.add(_describe(action))
    return atoms


def classify_plan(plan: PlanResult, sim: Simulator) -> dict[str, Any]:
    """Joins (planner admits into the running phone-assisted batch what the cascade parks) and the
    advisory remainder of the deviation (symmetric difference of the other atoms)."""
    server = sim.server
    assisted_batch = bool(server.loading is None and server.decoding() and server.assisted)
    planner, incumbent = _atoms(plan.actions), _atoms(plan.incumbent)
    joins: list[str] = []
    if assisted_batch:
        joins = sorted(atom[len("park:"):] for atom in incumbent
                       if atom.startswith("park:") and "admit:" + atom[len("park:"):] in planner)
    covered = {"park:" + rid for rid in joins} | {"admit:" + rid for rid in joins}
    advisory = sorted((planner ^ incumbent) - covered)
    return {"assisted_batch": assisted_batch, "joins": joins, "advisory": advisory}


def _fallback(reason: str, detail: object = None) -> dict[str, object]:
    return {"kind": FALLBACK, "reason": reason, **({} if detail is None else {"detail": detail})}


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


# Active planner --------------------------------------------------------------------------------------


class JointPlannerActive(JointPlannerShadow):
    """Plans synchronously at every journal epoch and executes the joint join (see module docstring)."""

    def __init__(
        self,
        config: JointPlannerActiveConfig,
        *,
        cost: CostModel | None = None,
        model_key: Callable[[str], str | None] = default_model_key,
    ) -> None:
        super().__init__(config, cost=cost, model_key=model_key, synchronous=True)  # type: ignore[arg-type]
        self._joined: dict[str, float] = {}
        self._scheduler: object | None = None

    # hook (under the scheduler's runtime lock; never raises)

    def observe_ticket(self, scheduler: object, event_kind: str, ticket: object,
                       event_time_us: int, lifecycle_state: str) -> None:
        if self._closed:
            return
        self._scheduler = scheduler
        try:
            with self._capture_lock:
                epoch = self._capture(scheduler, event_kind, ticket, event_time_us, lifecycle_state)
                if epoch is None:
                    return
                request_id = ticket.request.request_id
                if event_kind in TERMINAL_EVENTS or lifecycle_state in TERMINAL_EVENTS:
                    self._end_join(scheduler, request_id, event_time_us, event_kind)
                epoch, view = self._active_view(epoch, event_kind, request_id)
        except Exception as exc:  # noqa: BLE001 - a planner failure must never disturb dispatch
            self._record_error(event_kind, event_time_us, "CAPTURE_FAILED", exc)
            return
        try:
            self._plan_and_execute(scheduler, epoch, view, ticket, event_kind)
        except Exception as exc:  # noqa: BLE001
            self._record_error(event_kind, event_time_us, "PLAN_FAILED", exc)

    def _active_view(self, epoch: ShadowEpoch, event_kind: str, request_id: str) -> tuple[ShadowEpoch, str]:
        """At the acquisition of a same-model request into a running batch, plan on the state just
        before it (the request still queued). Rows that have not decoded a token next to a
        phone-assisted batch, and were not joined by the planner, wait physically: parked."""
        rows, queue, view = list(epoch.rows), list(epoch.queue), "live"
        if event_kind == "ACQUIRED" and epoch.loading is None:
            mine = [row for row in rows if row[0].request_id == request_id]
            others = [row for row in rows if row[0].request_id != request_id]
            if mine and others:
                rows, view = others, "pre_acquisition"
                queue.append(mine[0][0])
        parked: tuple[str, ...] = ()
        if epoch.assisted:
            waiting = tuple(sorted(r.request_id for r, left in rows
                                   if left >= r.output_tokens and r.request_id not in self._joined))
            if waiting and len(waiting) < len(rows):
                parked = waiting
        return replace(epoch, rows=tuple(rows), queue=tuple(queue), parked=parked), view

    def _plan_and_execute(self, scheduler, epoch: ShadowEpoch, view: str, ticket, event_kind: str) -> None:
        started = time.perf_counter()
        sim = simulator_from_epoch(epoch, self.cost)
        sim.events = epoch.sequence
        plan = self.planner.plan(sim)
        decision = classify_plan(plan, sim)
        executed, deferred, fallback, detail = self._execute(scheduler, plan, decision, epoch, view, ticket,
                                                             event_kind)
        scores, energies = plan.scores, plan.energies
        chosen = plan.chosen_index
        record = {
            "sequence": epoch.sequence,
            "event_kind": epoch.event_kind,
            "lifecycle_state": epoch.lifecycle_state,
            "event_time_s": round(epoch.now_s, 6),
            "request_id": epoch.request_id,
            "view": view,
            "live": dict(epoch.live),
            "state": {
                "queued": [r.request_id for r in epoch.queue],
                "running": [r.request_id for r, _ in epoch.rows],
                "parked": list(epoch.parked),
                "resident": epoch.resident, "loading": epoch.loading,
                "phone_model": epoch.phone_model, "phone_loading": epoch.phone_loading,
                "assisted": epoch.assisted, "unmodelled": list(epoch.unmodelled),
            },
            "sequential": [_describe(a) for a in plan.incumbent],
            "planner": [_describe(a) for a in plan.actions],
            "deviates": plan.deviates,
            "predicted": {
                "sequential_score_j": _round(scores[0] if scores else None),
                "planner_score_j": _round(scores[chosen] if len(scores) > chosen else None),
                "sequential_energy_j": _round(energies[0] if energies else None),
                "planner_energy_j": _round(energies[chosen] if len(energies) > chosen else None),
                "gain_j": round(plan.predicted_gain_j, 1),
            },
            "joins": decision["joins"],
            "advisory": decision["advisory"],
            "executed": executed,
            "deferred_to_acquisition": deferred,
            "execution_detail": detail,
            "fallback": fallback,
            "candidates": len(plan.candidates),
            "rollouts": plan.rollouts,
            "plan_ms": round(plan.elapsed_ms, 3),
            "budget_exhausted": plan.budget_exhausted,
            "budget_expired": plan.elapsed_ms > self.config.budget_ms,
            "fallback_reason": plan.fallback_reason,
            "wall_ms": round((time.perf_counter() - started) * 1e3, 3),
        }
        with self._lock:
            self._records.append(record)

    def _execute(self, scheduler, plan: PlanResult, decision, epoch: ShadowEpoch, view: str, ticket,
                 event_kind: str) -> tuple[list[str], list[str], dict[str, object] | None, dict[str, object]]:
        """(executed planner actions, joins deferred to the acquisition, fallback, detail). Nothing
        executed means the sequential decision stands."""
        if plan.fallback_reason is not None:
            return [], [], _fallback("PLANNER_FAILED", plan.fallback_reason), {}
        if plan.elapsed_ms > self.config.budget_ms:
            return [], [], _fallback(
                "BUDGET_EXPIRED", "%.1f ms > %.1f ms" % (plan.elapsed_ms, self.config.budget_ms)), {}
        if not plan.deviates:
            return [], [], None, {}
        executed: list[str] = []
        deferred: list[str] = []
        reasons: list[str] = []
        details: dict[str, object] = {}
        request_id = epoch.request_id
        for joiner in decision["joins"]:
            if joiner == request_id and event_kind == "ACQUIRED" and view == "pre_acquisition":
                outcome = self._register_join(scheduler, ticket, epoch)
                if outcome["outcome"] == "REGISTERED":
                    executed.append("join " + joiner + " (prefill yield)")
                    details["prefill_yield"] = outcome
                else:
                    reasons.append(str(outcome["outcome"]))
                    details["prefill_yield"] = outcome
            elif joiner == request_id and event_kind in ("DECISION", "REPLAN"):
                start = epoch.live.get("planned_start_s")
                if type(start) in (int, float) and start <= epoch.now_s + ADMISSION_TOLERANCE_S:
                    deferred.append(joiner)
                else:
                    reasons.append("CASCADE_DID_NOT_ADMIT")
                    details["planned_start_s"] = start
            else:
                reasons.append("ADMISSION_NOT_OVERRIDDEN")
                details.setdefault("not_overridden", []).append(joiner)
        if decision["advisory"]:
            reasons.append("ADVISORY_ONLY")
            details["advisory"] = decision["advisory"]
        if not reasons:
            return executed, deferred, None, details
        return executed, deferred, _fallback("+".join(sorted(set(reasons))), details), details

    def _register_join(self, scheduler, ticket, epoch: ShadowEpoch) -> Mapping[str, object]:
        register = getattr(scheduler, "register_joint_join_prefill_yield", None)
        if register is None:
            return {"outcome": "SCHEDULER_WITHOUT_PREFILL_YIELD"}
        key = self.model_key(ticket.model.model_id)
        prefill_s = (self.cost.model(key).prefill_s(int(ticket.request.input_tokens))
                     if key in self.cost.models else 0.0)
        bound_s = min(float(self.config.max_prefill_yield_s),
                      PREFILL_YIELD_MARGIN_S + PREFILL_YIELD_FACTOR * prefill_s)
        at_us = int(round(epoch.now_s * 1e6))
        outcome = dict(register(ticket, at_us=at_us, expires_at_us=at_us + max(1, int(bound_s * 1e6))))
        outcome["bound_s"] = round(bound_s, 3)
        if outcome.get("outcome") == "REGISTERED":
            self._joined[ticket.request.request_id] = epoch.now_s
        return outcome

    def _end_join(self, scheduler, request_id: str, event_time_us: int, event_kind: str) -> None:
        if self._joined.pop(request_id, None) is None:
            return
        clear = getattr(scheduler, "clear_joint_join_prefill_yield", None)
        if clear is not None:
            clear(request_id, at_us=int(event_time_us), reason="JOINER_" + event_kind)

    def _record_error(self, event_kind: str, event_time_us: int, reason: str, exc: BaseException) -> None:
        with self._lock:
            self._errors += 1
            self._records.append({
                "event_kind": event_kind, "event_time_s": round(event_time_us / 1e6, 6),
                "error": reason, "detail": type(exc).__name__ + ": " + str(exc)[:200],
                "fallback": _fallback(reason),
            })

    # results

    def close(self, timeout_s: float = 120.0) -> None:
        self._closed = True

    def yield_events(self) -> list[dict[str, Any]]:
        events = getattr(self._scheduler, "joint_join_prefill_yield_events", None)
        return [] if events is None else [dict(row) for row in events()]

    def summary(self) -> dict[str, Any]:
        records = self.records()
        planned = [r for r in records if "error" not in r]
        times = [r["plan_ms"] for r in planned]
        fallbacks = Counter(r["fallback"]["reason"] for r in records if r.get("fallback"))
        executed = [r for r in planned if r["executed"]]
        yields = Counter(row["kind"] for row in self.yield_events())
        ended = Counter(row.get("reason") for row in self.yield_events() if row["kind"] == "ENDED")
        return {
            "schema": JOINT_PLANNER_ACTIVE_SCHEMA,
            "configuration": self.config.to_json(),
            "cost_model_schema": self.cost.to_json()["schema"],
            "epochs": len(planned),
            "errors": self._errors,
            "deviations": sum(1 for r in planned if r["deviates"]),
            "executed_epochs": len(executed),
            "executed_joins": sorted({a.split()[1] for r in executed for a in r["executed"]}),
            "predicted_gain_executed_j": round(sum(r["predicted"]["gain_j"] for r in executed), 1),
            "fallbacks": sum(fallbacks.values()),
            "fallbacks_by_reason": dict(sorted(fallbacks.items())),
            "budget_exhausted": sum(1 for r in planned if r["budget_exhausted"]),
            "budget_expired": sum(1 for r in planned if r["budget_expired"]),
            "plan_ms_max": max(times, default=0.0),
            "plan_ms_mean": round(sum(times) / len(times), 3) if times else 0.0,
            "prefill_yields": {
                "registered": yields.get("REGISTERED", 0), "applied": yields.get("APPLIED", 0),
                "ended_by_reason": dict(sorted((str(k), v) for k, v in ended.items())),
            },
            "artifact": "JOINT_PLANNER_ACTIVE.json",
        }

    def result(self) -> dict[str, Any]:
        """RESULT ``joint_planner_active``: the summary and one compact row per epoch (the full records,
        with state and candidates, are in JOINT_PLANNER_ACTIVE.json)."""
        rows = []
        for r in self.records():
            if "error" in r:
                rows.append({"t_s": r["event_time_s"], "event": r["event_kind"], "error": r["error"],
                             "fallback": r["fallback"]["reason"]})
                continue
            rows.append({
                "t_s": r["event_time_s"], "event": r["event_kind"], "request_id": r["request_id"],
                "view": r["view"], "sequential": r["sequential"], "planner": r["planner"],
                "sequential_score_j": r["predicted"]["sequential_score_j"],
                "planner_score_j": r["predicted"]["planner_score_j"], "gain_j": r["predicted"]["gain_j"],
                "executed": r["executed"], "deferred_to_acquisition": r["deferred_to_acquisition"],
                "fallback": None if r["fallback"] is None else r["fallback"]["reason"], "plan_ms": r["plan_ms"],
            })
        return {**self.summary(), "epoch_decisions": rows}

    def artifact(self) -> dict[str, Any]:
        return {**self.summary(), "records": self.records(), "prefill_yield_events": self.yield_events()}


__all__: Sequence[str] = (
    "ADMISSION_TOLERANCE_S",
    "FALLBACK",
    "JOINT_PLANNER_ACTIVE_SCHEMA",
    "JointPlannerActive",
    "JointPlannerActiveConfig",
    "JointPlannerActiveError",
    "classify_plan",
    "joint_planner_config_from_json",
    "joint_planner_from_policy_json",
)
