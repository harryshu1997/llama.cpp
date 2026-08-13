"""Event-driven dispatch and background runtime snapshots."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .policy import Decision


RUNTIME_DISPATCH_QUEUE_SCHEMA = "s42-runtime-dispatch-queue-v2"


class RuntimeQueueError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeQueueError(f"{name} must be non-empty ASCII text")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeQueueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RuntimeDispatchReceipt:
    request_id: str
    route_id: str
    status: str
    scheduled_start_us: int
    observed_at_us: int
    queue_wait_us: int
    wake_reason: str

    def to_json(self) -> dict[str, int | str]:
        return {
            "observed_at_us": self.observed_at_us,
            "queue_wait_us": self.queue_wait_us,
            "request_id": self.request_id,
            "route_id": self.route_id,
            "scheduled_start_us": self.scheduled_start_us,
            "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
            "status": self.status,
            "wake_reason": self.wake_reason,
        }


@dataclass
class _DispatchEntry:
    decision: Decision
    admitted_at_us: int
    sequence: int
    state: str = "QUEUED"
    wake_reason: str = "calendar"


class RuntimeDispatchQueue:
    """Preserve scheduler decisions until their physical executor can run."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._entries: dict[str, _DispatchEntry] = {}
        self._active: dict[str, _DispatchEntry] = {}
        self._sequence = 0
        self._history: list[dict[str, int | str]] = []

    @staticmethod
    def _dispatch_key(entry: _DispatchEntry) -> tuple[int, int, str]:
        return (
            entry.decision.start_us,
            entry.sequence,
            entry.decision.request_id,
        )

    @staticmethod
    def _lanes(decision: Decision) -> frozenset[tuple[str, int]]:
        lanes = frozenset(
            (lease.resource_id, lane)
            for lease in decision.leases
            for lane in lease.lanes
        )
        if lanes:
            return lanes
        return frozenset({(f"route:{decision.route_id}", 0)})

    def _active_conflict(self, entry: _DispatchEntry) -> bool:
        lanes = self._lanes(entry.decision)
        return any(
            lanes & self._lanes(active.decision)
            for active in self._active.values()
        )

    def _earlier_queued_conflict(self, entry: _DispatchEntry) -> bool:
        lanes = self._lanes(entry.decision)
        key = self._dispatch_key(entry)
        return any(
            other.state == "QUEUED"
            and self._dispatch_key(other) < key
            and bool(lanes & self._lanes(other.decision))
            for other in self._entries.values()
        )

    def admit(self, decision: Decision, admitted_at_us: int) -> None:
        if not isinstance(decision, Decision):
            raise RuntimeQueueError("dispatch admission requires a Decision")
        admitted_at_us = _nonnegative_int(
            "dispatch admitted_at_us", admitted_at_us
        )
        with self._condition:
            if decision.request_id in self._entries:
                raise RuntimeQueueError("request already exists in dispatch queue")
            self._sequence += 1
            entry = _DispatchEntry(decision, admitted_at_us, self._sequence)
            self._entries[decision.request_id] = entry
            self._condition.notify_all()

    def wait(
        self,
        request_id: str,
        epoch_ns: int,
    ) -> RuntimeDispatchReceipt:
        request_id = _text("dispatch request_id", request_id)
        epoch_ns = _nonnegative_int("dispatch epoch_ns", epoch_ns)
        with self._condition:
            while True:
                entry = self._entries.get(request_id)
                if entry is None:
                    raise RuntimeQueueError("request is absent from dispatch queue")
                decision = entry.decision
                now_us = max(0, (time.monotonic_ns() - epoch_ns) // 1000)
                if entry.state == "REPLAN_REQUIRED":
                    return RuntimeDispatchReceipt(
                        request_id=request_id,
                        route_id=decision.route_id,
                        status="REPLAN_REQUIRED",
                        scheduled_start_us=decision.start_us,
                        observed_at_us=now_us,
                        queue_wait_us=max(0, now_us - entry.admitted_at_us),
                        wake_reason=entry.wake_reason,
                    )
                if entry.state != "QUEUED":
                    raise RuntimeQueueError("request has invalid queue state")
                resource_ready = (
                    not self._active_conflict(entry)
                    and not self._earlier_queued_conflict(entry)
                )
                if resource_ready and now_us >= decision.start_us:
                    entry.state = "ACTIVE"
                    self._active[request_id] = entry
                    wake_reason = entry.wake_reason
                    if wake_reason == "calendar":
                        wake_reason = (
                            "calendar_deadline"
                            if now_us <= decision.start_us + 1000
                            else "calendar_elapsed"
                        )
                    receipt = RuntimeDispatchReceipt(
                        request_id=request_id,
                        route_id=decision.route_id,
                        status="ACQUIRED",
                        scheduled_start_us=decision.start_us,
                        observed_at_us=now_us,
                        queue_wait_us=max(0, now_us - entry.admitted_at_us),
                        wake_reason=wake_reason,
                    )
                    self._history.append(receipt.to_json())
                    self._condition.notify_all()
                    return receipt
                timeout_s = None
                if resource_ready and now_us < decision.start_us:
                    timeout_s = (decision.start_us - now_us) / 1_000_000
                self._condition.wait(timeout=timeout_s)

    def complete(self, request_id: str, completed_at_us: int) -> None:
        request_id = _text("completed dispatch request_id", request_id)
        completed_at_us = _nonnegative_int(
            "dispatch completed_at_us", completed_at_us
        )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "ACTIVE":
                raise RuntimeQueueError("completed request is not active")
            route_id = entry.decision.route_id
            if self._active.get(request_id) is not entry:
                raise RuntimeQueueError("active route ownership mismatch")
            completed_lanes = self._lanes(entry.decision)
            del self._active[request_id]
            del self._entries[request_id]
            for queued in self._entries.values():
                if (
                    queued.state == "QUEUED"
                    and completed_lanes & self._lanes(queued.decision)
                ):
                    queued.wake_reason = "predecessor_completion"
            self._history.append({
                "completed_at_us": completed_at_us,
                "request_id": request_id,
                "route_id": route_id,
                "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
                "status": "COMPLETED",
            })
            self._condition.notify_all()

    def require_replan(self, request_id: str, reason: str) -> bool:
        request_id = _text("replanned dispatch request_id", request_id)
        reason = _text("dispatch replan reason", reason)
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                return False
            if entry.state == "ACTIVE":
                raise RuntimeQueueError("cannot replan an active request")
            if entry.state == "REPLAN_REQUIRED":
                return False
            entry.state = "REPLAN_REQUIRED"
            entry.wake_reason = reason
            self._condition.notify_all()
            return True

    def retire_replan(self, request_id: str) -> None:
        request_id = _text("retired dispatch request_id", request_id)
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "REPLAN_REQUIRED":
                raise RuntimeQueueError("request does not await replanning")
            del self._entries[request_id]

    def conflicting_queued_requests(
        self,
        active_decision: Decision,
        extended_until_us: int,
    ) -> tuple[str, ...]:
        if not isinstance(active_decision, Decision):
            raise RuntimeQueueError("active conflict query requires a Decision")
        extended_until_us = _nonnegative_int(
            "dispatch extension end", extended_until_us
        )
        active_lanes = self._lanes(active_decision)
        with self._condition:
            conflicts = []
            for request_id, entry in self._entries.items():
                if (
                    request_id == active_decision.request_id
                    or entry.state != "QUEUED"
                ):
                    continue
                if any(
                    lease.start_us < extended_until_us
                    and any(
                        (lease.resource_id, lane) in active_lanes
                        for lane in lease.lanes
                    )
                    for lease in entry.decision.leases
                ):
                    conflicts.append(entry)
            return tuple(
                entry.decision.request_id
                for entry in sorted(conflicts, key=self._dispatch_key)
            )

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "active": {
                    request_id: {
                        "lanes": [
                            [resource_id, lane]
                            for resource_id, lane in sorted(
                                self._lanes(entry.decision)
                            )
                        ],
                        "route_id": entry.decision.route_id,
                    }
                    for request_id, entry in sorted(self._active.items())
                },
                "active_by_route": {
                    route_id: sorted(
                        request_id
                        for request_id, entry in self._active.items()
                        if entry.decision.route_id == route_id
                    )
                    for route_id in sorted({
                        entry.decision.route_id
                        for entry in self._active.values()
                    })
                },
                "history": copy.deepcopy(self._history),
                "queued": {
                    route_id: [
                        entry.decision.request_id
                        for entry in sorted(
                            self._entries.values(), key=self._dispatch_key
                        )
                        if entry.state == "QUEUED"
                        and entry.decision.route_id == route_id
                    ]
                    for route_id in sorted({
                        entry.decision.route_id
                        for entry in self._entries.values()
                        if entry.state == "QUEUED"
                    })
                },
                "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
            }


@dataclass(frozen=True)
class BackgroundRuntimeSnapshot:
    name: str
    captured_at_ns: int
    value: object | None
    error: str | None
    stale: bool

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_ns": self.captured_at_ns,
            "error": self.error,
            "name": self.name,
            "stale": self.stale,
            "value": copy.deepcopy(self.value),
        }


class BackgroundRuntimeMonitor:
    """Refresh slow health probes away from request scheduling threads."""

    def __init__(
        self,
        probes: Mapping[str, Callable[[], object]],
        *,
        refresh_interval_s: float = 1.0,
        stale_after_s: float = 3.0,
    ) -> None:
        if refresh_interval_s <= 0 or stale_after_s <= 0:
            raise RuntimeQueueError("runtime monitor intervals must be positive")
        if stale_after_s < refresh_interval_s:
            raise RuntimeQueueError("runtime monitor stale bound is too short")
        self._condition = threading.Condition()
        self._probes = {
            _text("runtime probe name", name): probe
            for name, probe in probes.items()
        }
        if not self._probes or any(not callable(probe) for probe in self._probes.values()):
            raise RuntimeQueueError("runtime monitor requires callable probes")
        self._refresh_interval_s = refresh_interval_s
        self._stale_after_ns = round(stale_after_s * 1e9)
        self._snapshots: dict[str, BackgroundRuntimeSnapshot] = {}
        self._refresh = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeQueueError("runtime monitor is already started")
            self._thread = threading.Thread(
                target=self._run,
                name="unified-runtime-monitor",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            # Consume the wake that started this pass before probing. A refresh
            # requested during a slow probe must remain set for the next pass.
            self._refresh.clear()
            for name, probe in self._probes.items():
                try:
                    value = probe()
                    error = None
                except BaseException as exc:
                    value = None
                    error = f"{type(exc).__name__}: {exc}"
                snapshot = BackgroundRuntimeSnapshot(
                    name=name,
                    captured_at_ns=time.monotonic_ns(),
                    value=copy.deepcopy(value),
                    error=error,
                    stale=False,
                )
                with self._condition:
                    self._snapshots[name] = snapshot
                    self._condition.notify_all()
            self._refresh.wait(self._refresh_interval_s)

    def request_refresh(self) -> None:
        self._refresh.set()

    def wait_until_populated(
        self,
        names: tuple[str, ...],
        timeout_s: float,
    ) -> bool:
        names = tuple(_text("runtime snapshot name", name) for name in names)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while not set(names).issubset(self._snapshots):
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return False
                self._condition.wait(remaining_s)
            return True

    def snapshot(self, name: str) -> BackgroundRuntimeSnapshot:
        name = _text("runtime snapshot name", name)
        now_ns = time.monotonic_ns()
        with self._condition:
            snapshot = self._snapshots.get(name)
            if snapshot is None:
                return BackgroundRuntimeSnapshot(
                    name=name,
                    captured_at_ns=0,
                    value=None,
                    error="snapshot is not populated",
                    stale=True,
                )
            return BackgroundRuntimeSnapshot(
                name=snapshot.name,
                captured_at_ns=snapshot.captured_at_ns,
                value=copy.deepcopy(snapshot.value),
                error=snapshot.error,
                stale=(now_ns - snapshot.captured_at_ns > self._stale_after_ns),
            )

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._refresh.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                raise RuntimeQueueError("runtime monitor did not stop")

    @property
    def probe_names(self) -> Mapping[str, Callable[[], object]]:
        return MappingProxyType(dict(self._probes))
