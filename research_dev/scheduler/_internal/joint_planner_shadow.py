"""Shadow mode for the joint planner: log what it would decide next to what the sequential cascade did.

Opt-in through ``dispatch_policy.joint_planner = {"mode": "shadow", ...}``. The scheduler calls
``observe_ticket`` after every runtime journal record (DECISION, REPLAN, FALLBACK, ACQUIRED, COMPLETED,
FAILED, CANCELLED). The hook only reads: it keeps its own mirror of the queue and the server from the
journal stream, reads decode progress and phone session residency without locks or side effects,
freezes a ``ShadowEpoch`` and hands it to a worker thread, which plans off the scheduler's locks.
Nothing it computes flows back into dispatch; any error is recorded and swallowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from .joint_planner import JointPlanner, JointPlannerConfig, JointPlannerError
from .joint_planner_model import CostModel, measured_eval_v2_cost_model
from .joint_planner_sim import (
    Row,
    SequentialPolicy,
    SimOptions,
    SimRequest,
    Simulator,
    _describe,
)


JOINT_PLANNER_SHADOW_SCHEMA = "research-scheduler-joint-planner-shadow-v1"
JOINT_PLANNER_MODES = ("shadow",)
TERMINAL_EVENTS = frozenset({"COMPLETED", "FAILED", "CANCELLED"})
_CONFIG_FIELDS = ("mode", "horizon_s", "budget_ms", "depth", "objective")


class JointPlannerShadowError(ValueError):
    pass


@dataclass(frozen=True)
class JointPlannerShadowConfig:
    mode: str = "shadow"
    horizon_s: float = 3600.0
    budget_ms: float = 50.0
    depth: int = 2
    objective: str = "fleet"

    def __post_init__(self) -> None:
        if self.mode not in JOINT_PLANNER_MODES:
            raise JointPlannerShadowError(
                "joint planner mode must be 'shadow' or 'active' (joint_planner_active)"
            )
        self.planner_config()

    def planner_config(self) -> JointPlannerConfig:
        try:
            return JointPlannerConfig(
                horizon_s=float(self.horizon_s), budget_ms=float(self.budget_ms),
                depth=self.depth, objective=self.objective,
            )
        except JointPlannerError as exc:
            raise JointPlannerShadowError(str(exc)) from exc

    @classmethod
    def from_json(cls, value: object) -> "JointPlannerShadowConfig":
        if not isinstance(value, Mapping):
            raise JointPlannerShadowError("joint planner configuration must be an object")
        unknown = sorted(set(value) - set(_CONFIG_FIELDS))
        if unknown:
            raise JointPlannerShadowError("joint planner configuration has unknown fields: " + ", ".join(unknown))
        if "mode" not in value:
            raise JointPlannerShadowError("joint planner configuration needs a mode")
        for name in ("horizon_s", "budget_ms"):
            if name in value and (type(value[name]) not in (int, float) or not value[name] > 0):
                raise JointPlannerShadowError("joint planner " + name + " must be a positive number")
        if "depth" in value and (type(value["depth"]) is not int or value["depth"] < 0):
            raise JointPlannerShadowError("joint planner depth must be a nonnegative integer")
        if "objective" in value and type(value["objective"]) is not str:
            raise JointPlannerShadowError("joint planner objective must be a string")
        return cls(**dict(value))

    def to_json(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _CONFIG_FIELDS}


def default_model_key(model_id: str) -> str | None:
    """Cost-model key for a scheduler model id (None: not modelled)."""
    lowered = model_id.lower()
    for prefix, key in (("qwen", "qwen"), ("gemma", "gemma"), ("llama", "llama")):
        if lowered.startswith(prefix):
            return key
    return None


@dataclass(frozen=True)
class ShadowEpoch:
    sequence: int
    event_kind: str
    lifecycle_state: str
    now_s: float
    request_id: str
    live: Mapping[str, object]
    queue: tuple[SimRequest, ...]
    rows: tuple[tuple[SimRequest, float], ...]
    resident: str | None
    loading: str | None
    load_end_s: float
    assisted: bool
    phone_model: str | None
    phone_loading: str | None
    phone_ready_s: float
    unmodelled: tuple[str, ...] = ()
    # Active mode only: running rows that physically wait behind the batch (their prompt cannot be
    # batched with a phone-policy co-tenant); the shadow never sets it.
    parked: tuple[str, ...] = ()


def simulator_from_epoch(epoch: ShadowEpoch, cost: CostModel) -> Simulator:
    """A simulator positioned at the epoch (known work only; no future arrivals)."""
    sim = Simulator((), cost, SimOptions(), start_s=epoch.now_s)
    sim.record_decisions = False
    sim.queue = sorted(epoch.queue, key=lambda r: (r.arrival_s, r.request_id))
    for request in sim.queue:
        sim.last_arrival.setdefault(request.model, request.arrival_s)
    server = sim.server
    server.resident = epoch.resident
    if epoch.loading is not None:
        server.resident = None
        server.loading = epoch.loading
        server.load_end_s = max(epoch.now_s + 1e-3, epoch.load_end_s)
    parked = set(epoch.parked) if any(r.request_id not in epoch.parked for r, _ in epoch.rows) else set()
    for request, tokens_left in epoch.rows:
        waiting = request.request_id in parked
        server.rows.append(Row(request=request, admitted_s=epoch.now_s, tokens_left=max(1.0, tokens_left),
                               parked=waiting, decode_start_s=None if waiting else epoch.now_s))
    if server.rows and server.loading is None:
        server.resident = server.rows[0].request.model
        server.phase_start_s = epoch.now_s - 60.0
        server.phase_rows = len(server.rows)
        server.assisted = bool(epoch.assisted and sim.cost.model(server.resident).assistable)
    phone = sim.primary
    phone.model = epoch.phone_model
    if epoch.phone_loading is not None:
        phone.model = None
        phone.loading = epoch.phone_loading
        phone.ready_s = max(epoch.now_s + 1e-3, epoch.phone_ready_s)
    if server.assisted and not sim.helper_available(server.resident):
        server.assisted = False
    return sim


@dataclass
class _Mirror:
    queued: dict[str, SimRequest] = field(default_factory=dict)
    running: dict[str, SimRequest] = field(default_factory=dict)
    acquired_s: dict[str, float] = field(default_factory=dict)
    resident: str | None = None
    loading: str | None = None
    load_end_s: float = 0.0
    artifact_models: dict[str, str] = field(default_factory=dict)


class JointPlannerShadow:
    """Observes journal events, plans in a worker thread, keeps the records for the run artifacts."""

    def __init__(
        self,
        config: JointPlannerShadowConfig,
        *,
        cost: CostModel | None = None,
        model_key: Callable[[str], str | None] = default_model_key,
        synchronous: bool = False,
    ) -> None:
        self.config = config
        self.cost = measured_eval_v2_cost_model() if cost is None else cost
        self.model_key = model_key
        self.planner = JointPlanner(config.planner_config(), SequentialPolicy())
        self.synchronous = synchronous
        self._mirror = _Mirror()
        self._sequence = 0
        self._records: list[dict[str, Any]] = []
        self._errors = 0
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._closed = False

    # capture (called under the scheduler's locks: read-only and cheap)

    def observe_ticket(self, scheduler: object, event_kind: str, ticket: object,
                       event_time_us: int, lifecycle_state: str) -> None:
        try:
            with self._capture_lock:
                epoch = self._capture(scheduler, event_kind, ticket, event_time_us, lifecycle_state)
        except Exception as exc:  # noqa: BLE001 - the shadow must never disturb dispatch
            self._record_error(event_kind, event_time_us, "CAPTURE_FAILED", exc)
            return
        if epoch is None:
            return
        if self.synchronous:
            self._plan(epoch)
            return
        self._ensure_worker()
        self._queue.put(epoch)

    def _capture(self, scheduler, event_kind, ticket, event_time_us, lifecycle_state) -> ShadowEpoch | None:
        if self._closed:
            return None
        mirror = self._mirror
        now_s = event_time_us / 1e6
        request = ticket.request
        model_id = ticket.model.model_id
        mirror.artifact_models[ticket.model.artifact_sha256] = model_id
        key = self.model_key(model_id)
        shadow_request = None if key is None else SimRequest(
            request.request_id, key, request.arrival_us / 1e6, int(request.input_tokens), int(request.output_tokens),
        )
        request_id = request.request_id
        if mirror.loading is not None and now_s >= mirror.load_end_s:
            mirror.resident, mirror.loading = mirror.loading, None
        load_model = _desktop_load_model(ticket)
        if event_kind in TERMINAL_EVENTS or lifecycle_state in TERMINAL_EVENTS:
            mirror.queued.pop(request_id, None)
            if mirror.running.pop(request_id, None) is not None and event_kind == "COMPLETED" and key is not None:
                if mirror.loading is None:
                    mirror.resident = key
        elif event_kind == "ACQUIRED" or lifecycle_state == "ACQUIRED":
            mirror.queued.pop(request_id, None)
            if shadow_request is not None:
                mirror.running[request_id] = shadow_request
                mirror.acquired_s[request_id] = now_s
            if load_model is not None:
                load_key = self.model_key(load_model)
                if load_key is not None and load_key in self.cost.models:
                    mirror.loading = load_key
                    mirror.load_end_s = now_s + self.cost.model(load_key).load_s
                    mirror.resident = None
        elif shadow_request is not None:
            mirror.queued[request_id] = shadow_request
        self._sequence += 1
        unmodelled = tuple(sorted(
            r.request_id for r in (*mirror.queued.values(), *mirror.running.values())
            if r.model not in self.cost.models
        ))
        rows = []
        progress = _decode_progress(scheduler)
        for rid, row_request in sorted(mirror.running.items()):
            if row_request.model not in self.cost.models or mirror.loading is not None:
                continue
            record = progress.get(rid)
            done = record[1] if isinstance(record, tuple) and len(record) == 2 else 0
            rows.append((row_request, float(max(1, row_request.output_tokens - int(done)))))
        phone_model, phone_loading, assisted = _phone_state(scheduler, mirror.artifact_models, self.model_key,
                                                           self.cost.primary_phone)
        phone_ready_s = now_s + self.cost.phones[self.cost.primary_phone].provision_s if phone_loading else 0.0
        decision = getattr(ticket, "decision", None)
        plan = getattr(ticket, "execution_plan", None)
        live = {
            "dispatch_state": getattr(ticket, "dispatch_state", None),
            "planned_start_s": None if decision is None else getattr(decision, "start_us", 0) / 1e6,
            "planned_finish_upper_s": None if decision is None else getattr(decision, "finish_upper_us", 0) / 1e6,
            "route_id": None if plan is None else getattr(plan, "route_id", None),
            "desktop_load_model": None if load_model is None else self.model_key(load_model),
        }
        queued = tuple(r for r in mirror.queued.values() if r.model in self.cost.models)
        return ShadowEpoch(
            sequence=self._sequence, event_kind=event_kind, lifecycle_state=lifecycle_state, now_s=now_s,
            request_id=request_id, live=live, queue=queued, rows=tuple(rows), resident=mirror.resident,
            loading=mirror.loading, load_end_s=mirror.load_end_s, assisted=assisted,
            phone_model=phone_model, phone_loading=phone_loading, phone_ready_s=phone_ready_s,
            unmodelled=unmodelled,
        )

    # planning (worker thread)

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._run_worker, name="joint-planner-shadow", daemon=True)
            self._worker.start()

    def _run_worker(self) -> None:
        while True:
            epoch = self._queue.get()
            if epoch is None:
                return
            self._plan(epoch)

    def _plan(self, epoch: ShadowEpoch) -> None:
        started = time.perf_counter()
        try:
            sim = simulator_from_epoch(epoch, self.cost)
            sim.events = epoch.sequence
            plan = self.planner.plan(sim)
            emulated = [_describe(a) for a in plan.incumbent]
            record = {
                "sequence": epoch.sequence,
                "event_kind": epoch.event_kind,
                "lifecycle_state": epoch.lifecycle_state,
                "event_time_s": round(epoch.now_s, 6),
                "request_id": epoch.request_id,
                "live": dict(epoch.live),
                "state": {
                    "queued": [r.request_id for r in epoch.queue],
                    "running": [r.request_id for r, _ in epoch.rows],
                    "resident": epoch.resident, "loading": epoch.loading,
                    "phone_model": epoch.phone_model, "phone_loading": epoch.phone_loading,
                    "assisted": epoch.assisted, "unmodelled": list(epoch.unmodelled),
                },
                "sequential_emulated": emulated,
                "planner": [_describe(a) for a in plan.actions],
                "deviates": plan.deviates,
                "predicted_gain_j": round(plan.predicted_gain_j, 1),
                "candidates": len(plan.candidates),
                "rollouts": plan.rollouts,
                "plan_ms": round(plan.elapsed_ms, 3),
                "budget_exhausted": plan.budget_exhausted,
                "fallback_reason": plan.fallback_reason,
            }
        except Exception as exc:  # noqa: BLE001
            self._record_error(epoch.event_kind, int(epoch.now_s * 1e6), "PLAN_FAILED", exc)
            return
        record["wall_ms"] = round((time.perf_counter() - started) * 1e3, 3)
        with self._lock:
            self._records.append(record)

    def _record_error(self, event_kind: str, event_time_us: int, reason: str, exc: BaseException) -> None:
        with self._lock:
            self._errors += 1
            self._records.append({
                "event_kind": event_kind, "event_time_s": round(event_time_us / 1e6, 6),
                "error": reason, "detail": type(exc).__name__ + ": " + str(exc)[:200],
            })

    # results

    def close(self, timeout_s: float = 120.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout_s)

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted((dict(r) for r in self._records), key=lambda r: (r.get("sequence", 0), r["event_time_s"]))

    def summary(self) -> dict[str, Any]:
        records = self.records()
        planned = [r for r in records if "error" not in r]
        times = [r["plan_ms"] for r in planned]
        return {
            "schema": JOINT_PLANNER_SHADOW_SCHEMA,
            "configuration": self.config.to_json(),
            "cost_model_schema": self.cost.to_json()["schema"],
            "epochs": len(planned),
            "errors": self._errors,
            "deviations": sum(1 for r in planned if r["deviates"]),
            "predicted_gain_j": round(sum(r["predicted_gain_j"] for r in planned if r["deviates"]), 1),
            "budget_exhausted": sum(1 for r in planned if r["budget_exhausted"]),
            "plan_ms_max": max(times, default=0.0),
            "plan_ms_mean": round(sum(times) / len(times), 3) if times else 0.0,
            "artifact": "JOINT_PLANNER_SHADOW.json",
        }

    def artifact(self) -> dict[str, Any]:
        return {**self.summary(), "records": self.records()}


def _desktop_load_model(ticket: object) -> str | None:
    plan = getattr(ticket, "execution_plan", None)
    for transition in getattr(plan, "transitions", ()) or ():
        transition_id = getattr(transition, "transition_id", "")
        device_id = getattr(transition, "device_id", "")
        if transition_id.startswith("load:") and "desktop" in device_id:
            parts = transition_id.split(":")
            if len(parts) > 1:
                return parts[1]
    return None


def _decode_progress(scheduler: object) -> Mapping[str, object]:
    controller = getattr(scheduler, "_model_placement_controller", None)
    progress = getattr(controller, "_request_decode_progress", None)
    return dict(progress) if isinstance(progress, Mapping) else {}


def _phone_state(
    scheduler: object,
    artifact_models: Mapping[str, str],
    model_key: Callable[[str], str | None],
    primary_phone: str,
) -> tuple[str | None, str | None, bool]:
    """(resident model, loading model, helper in use) of the primary phone's HTP sessions."""
    controller = getattr(scheduler, "_model_placement_controller", None)
    sessions = getattr(controller, "_phone_session_states", None)
    if not isinstance(sessions, Mapping):
        return None, None, False
    resident, loading, assisted = set(), set(), False
    for state in sessions.values():
        if primary_phone not in str(getattr(state, "endpoint", "")):
            continue
        model_id = artifact_models.get(getattr(state, "resident_artifact_sha256", None) or "")
        key = None if model_id is None else model_key(model_id)
        status = getattr(state, "state", "")
        if status in ("READY", "VERIFIED") and key is not None:
            resident.add(key)
        elif status == "LOADING" and key is not None:
            loading.add(key)
        elif status in ("EMPTY", "DRAINING", "FAILED"):
            resident.add(None)
        if getattr(state, "active_helper_references", ()):
            assisted = True
    if loading:
        return None, sorted(loading)[0], assisted
    if len(resident) == 1:
        return next(iter(resident)), None, assisted
    return None, None, assisted


def shadow_config_from_policy_json(value: Mapping[str, object]) -> tuple[dict[str, object], JointPlannerShadowConfig | None]:
    """Split ``joint_planner`` off a dispatch policy object (the rest is the RuntimeDispatchPolicy)."""
    rest = {k: v for k, v in value.items() if k != "joint_planner"}
    if "joint_planner" not in value:
        return rest, None
    return rest, JointPlannerShadowConfig.from_json(value["joint_planner"])


__all__: Sequence[str] = (
    "JOINT_PLANNER_SHADOW_SCHEMA",
    "JointPlannerShadow",
    "JointPlannerShadowConfig",
    "JointPlannerShadowError",
    "ShadowEpoch",
    "default_model_key",
    "shadow_config_from_policy_json",
    "simulator_from_epoch",
)
