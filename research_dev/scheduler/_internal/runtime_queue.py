"""Event-driven dispatch and background runtime snapshots."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import threading
import time
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence, TypeVar

from .policy import Decision, LeaseRecord
from .runtime_decode_cohort import RuntimeDecodeCohortBinding
from .runtime_dispatch_policy import (
    DEFAULT_RUNTIME_DISPATCH_POLICY,
    RuntimeDispatchPolicy,
)
from .runtime_residency_hysteresis import (
    DECISION_LOG_LIMIT,
    REASON_SAME_MODEL_QUEUED,
    ResidencyArrivalHistory,
    ResidencyHysteresisDecision,
    ResidencyHysteresisError,
    decide_residency_hysteresis,
    validated_arrivals,
    validated_current_decisions,
    validated_decisions,
    validated_release,
)


RUNTIME_DISPATCH_QUEUE_SCHEMA = "s42-runtime-dispatch-queue-v2"
_DEFERRED_REPLAN = "DEFERRED_REPLAN"
_CommitResult = TypeVar("_CommitResult")


class RuntimeQueueError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeQueueError(f"{name} must be non-empty ASCII text")
    return value


def _optional_key(value: object) -> bool:
    """Whether ``value`` is a residency hysteresis key (None or a non-empty str)."""
    return value is None or (type(value) is str and bool(value))


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
    queue_generation: int = 0

    def to_json(self) -> dict[str, int | str]:
        return {
            "observed_at_us": self.observed_at_us,
            "queue_wait_us": self.queue_wait_us,
            "queue_generation": self.queue_generation,
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
    residency_transition_barrier: bool = False
    decode_cohort: RuntimeDecodeCohortBinding | None = None
    state: str = "QUEUED"
    wake_reason: str = "calendar"
    predecessor_request_ids: set[str] = field(default_factory=set)
    generation: int = 1
    capacity_released_at_us: int | None = None
    capacity_replan_count: int = 0
    capacity_frontier_wake_count: int = 0
    # Model of an entry holding the exclusive residency resource; set only
    # under dispatch_policy.residency_hysteresis_s.
    residency_hysteresis_key: str | None = None
    # First admission of the request (kept across replans); hysteresis fairness.
    queued_since_us: int | None = None


@dataclass(frozen=True)
class _DispatchEntrySnapshot:
    decision: Decision
    admitted_at_us: int
    sequence: int
    residency_transition_barrier: bool
    decode_cohort: RuntimeDecodeCohortBinding | None
    state: str
    wake_reason: str
    predecessor_request_ids: tuple[str, ...]
    generation: int
    capacity_released_at_us: int | None
    capacity_replan_count: int
    capacity_frontier_wake_count: int
    residency_hysteresis_key: str | None = None
    queued_since_us: int | None = None


@dataclass(frozen=True)
class _DispatchQueueCheckpoint:
    entries: tuple[tuple[str, _DispatchEntrySnapshot], ...]
    active_request_ids: tuple[str, ...]
    sequence: int
    history: tuple[tuple[tuple[str, int | str], ...], ...]
    policy_events: tuple[tuple[str, int], ...] = ()
    residency_release: tuple[int, str, tuple[str, ...]] | None = None
    residency_hysteresis_holds: tuple[tuple[str, int, int], ...] = ()
    residency_arrivals: tuple[tuple[str, ResidencyArrivalHistory], ...] = ()
    residency_hysteresis_decisions: tuple[ResidencyHysteresisDecision, ...] = ()
    residency_hysteresis_log: tuple[ResidencyHysteresisDecision, ...] = ()


RESIDENCY_HYSTERESIS_WAKE_REASON = "residency_hysteresis_released"
RESIDENCY_HYSTERESIS_STAT_NAMES = (
    "residency_hysteresis_admissions",
    "residency_hysteresis_holds",
    "residency_hysteresis_skips",
)
_NOT_DISPATCHED_STATES = frozenset({
    _DEFERRED_REPLAN, "QUEUED", "REPLANNING", "REPLAN_REQUIRED",
})


class RuntimeDispatchQueue:
    """Preserve scheduler decisions until their physical executor can run."""

    def __init__(
        self, policy: RuntimeDispatchPolicy = DEFAULT_RUNTIME_DISPATCH_POLICY
    ) -> None:
        self._condition = threading.Condition()
        self._entries: dict[str, _DispatchEntry] = {}
        self._policy = DEFAULT_RUNTIME_DISPATCH_POLICY
        self.set_policy(policy)
        self._policy_events: dict[str, int] = {}
        self._active: dict[str, _DispatchEntry] = {}
        self._sequence = 0
        self._history: list[dict[str, int | str]] = []
        self._wake_holds: set[int] = set()
        self._next_wake_hold = 0
        # (completed_at_us, key, lane resource ids) of the last capacity release
        # on the exclusive residency resource, and request_id ->
        # (held_until_us, released_at_us) of the holds a waiter observed.
        self._residency_release: tuple[int, str, tuple[str, ...]] | None = None
        self._residency_hysteresis_holds: dict[str, tuple[int, int]] = {}
        # Per-model admission history, the current decision per queued change
        # and the (bounded) log of every decision, all under hysteresis only.
        self._residency_arrivals: dict[str, ResidencyArrivalHistory] = {}
        self._residency_hysteresis_decisions: dict[
            str, ResidencyHysteresisDecision
        ] = {}
        self._residency_hysteresis_log: list[ResidencyHysteresisDecision] = []

    @property
    def policy(self) -> RuntimeDispatchPolicy:
        return self._policy

    def set_policy(self, policy: RuntimeDispatchPolicy) -> None:
        """Select the ordering policy; only an empty queue may change it."""
        if not isinstance(policy, RuntimeDispatchPolicy):
            raise RuntimeQueueError("dispatch policy is invalid")
        with self._condition:
            if self._entries:
                raise RuntimeQueueError(
                    "dispatch policy cannot change with queued work"
                )
            self._policy = policy

    def hold_wake(self) -> int:
        """Keep queued dispatch waiters asleep until publication completes."""
        with self._condition:
            self._next_wake_hold += 1
            token = self._next_wake_hold
            self._wake_holds.add(token)
            return token

    def release_wake(self, token: int) -> None:
        if type(token) is not int or token < 1:
            raise RuntimeQueueError("dispatch wake hold token is invalid")
        with self._condition:
            if token not in self._wake_holds:
                raise RuntimeQueueError("dispatch wake hold is absent")
            self._wake_holds.remove(token)
            if not self._wake_holds:
                self._condition.notify_all()

    @contextmanager
    def defer_wake(self) -> Iterator[None]:
        """Keep queue state private until one scheduler update is complete."""
        token = self.hold_wake()
        try:
            yield
        finally:
            self.release_wake(token)

    def checkpoint(self) -> object:
        with self._condition:
            if any(
                self._entries.get(request_id) is not entry
                for request_id, entry in self._active.items()
            ):
                raise RuntimeQueueError("active route ownership mismatch")
            return _DispatchQueueCheckpoint(
                entries=tuple(
                    (
                        request_id,
                        _DispatchEntrySnapshot(
                            decision=entry.decision,
                            admitted_at_us=entry.admitted_at_us,
                            sequence=entry.sequence,
                            residency_transition_barrier=(
                                entry.residency_transition_barrier
                            ),
                            decode_cohort=entry.decode_cohort,
                            state=entry.state,
                            wake_reason=entry.wake_reason,
                            predecessor_request_ids=tuple(sorted(
                                entry.predecessor_request_ids
                            )),
                            generation=entry.generation,
                            capacity_released_at_us=(
                                entry.capacity_released_at_us
                            ),
                            capacity_replan_count=(
                                entry.capacity_replan_count
                            ),
                            capacity_frontier_wake_count=(
                                entry.capacity_frontier_wake_count
                            ),
                            residency_hysteresis_key=(
                                entry.residency_hysteresis_key
                            ),
                            queued_since_us=entry.queued_since_us,
                        ),
                    )
                    for request_id, entry in sorted(self._entries.items())
                ),
                active_request_ids=tuple(sorted(self._active)),
                sequence=self._sequence,
                history=tuple(
                    tuple(sorted(row.items())) for row in self._history
                ),
                policy_events=tuple(sorted(self._policy_events.items())),
                residency_release=self._residency_release,
                residency_hysteresis_holds=tuple(
                    (request_id, held_until_us, released_at_us)
                    for request_id, (held_until_us, released_at_us)
                    in sorted(self._residency_hysteresis_holds.items())
                ),
                residency_arrivals=tuple(sorted(
                    self._residency_arrivals.items()
                )),
                residency_hysteresis_decisions=tuple(
                    row for _, row in sorted(
                        self._residency_hysteresis_decisions.items()
                    )
                ),
                residency_hysteresis_log=tuple(self._residency_hysteresis_log),
            )

    def restore(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _DispatchQueueCheckpoint):
            raise RuntimeQueueError("dispatch checkpoint is invalid")
        entries: dict[str, _DispatchEntry] = {}
        for request_id, snapshot in checkpoint.entries:
            if (
                type(request_id) is not str
                or not request_id
                or not isinstance(snapshot, _DispatchEntrySnapshot)
                or request_id in entries
                or snapshot.decision.request_id != request_id
                or type(snapshot.residency_transition_barrier) is not bool
                or not _optional_key(snapshot.residency_hysteresis_key)
                or snapshot.queued_since_us is not None and (
                    type(snapshot.queued_since_us) is not int
                    or snapshot.queued_since_us < 0
                )
                or snapshot.state not in {
                    "ACTIVE",
                    _DEFERRED_REPLAN,
                    "FINISHING",
                    "QUEUED",
                    "REPLANNING",
                    "REPLAN_REQUIRED",
                }
            ):
                raise RuntimeQueueError("dispatch checkpoint is invalid")
            entries[request_id] = _DispatchEntry(
                decision=snapshot.decision,
                admitted_at_us=snapshot.admitted_at_us,
                sequence=snapshot.sequence,
                residency_transition_barrier=(
                    snapshot.residency_transition_barrier
                ),
                decode_cohort=snapshot.decode_cohort,
                state=snapshot.state,
                wake_reason=snapshot.wake_reason,
                predecessor_request_ids=set(
                    snapshot.predecessor_request_ids
                ),
                generation=snapshot.generation,
                capacity_released_at_us=(
                    snapshot.capacity_released_at_us
                ),
                capacity_replan_count=snapshot.capacity_replan_count,
                capacity_frontier_wake_count=(
                    snapshot.capacity_frontier_wake_count
                ),
                residency_hysteresis_key=snapshot.residency_hysteresis_key,
                queued_since_us=snapshot.queued_since_us,
            )
            self._validate_cohort(snapshot.decision, snapshot.decode_cohort)
        active_request_ids = checkpoint.active_request_ids
        if (
            type(active_request_ids) is not tuple
            or len(set(active_request_ids)) != len(active_request_ids)
            or any(request_id not in entries for request_id in active_request_ids)
            or any(
                entry.state == "ACTIVE"
                and request_id not in active_request_ids
                for request_id, entry in entries.items()
            )
            or any(entries[request_id].state != "ACTIVE"
                   for request_id in active_request_ids)
            or any(
                request_id in entry.predecessor_request_ids
                or not entry.predecessor_request_ids.issubset(entries)
                for request_id, entry in entries.items()
            )
            or not self._causal_graph_is_acyclic(entries)
            or type(checkpoint.sequence) is not int
            or checkpoint.sequence < 0
            or any(entry.sequence > checkpoint.sequence
                   for entry in entries.values())
            or any(entry.generation < 1 for entry in entries.values())
            or any(
                (
                    entry.state == "FINISHING"
                    and entry.capacity_released_at_us is None
                )
                or (
                    entry.state != "FINISHING"
                    and entry.capacity_released_at_us is not None
                )
                or entry.capacity_replan_count < 0
                or entry.capacity_frontier_wake_count < 0
                for entry in entries.values()
            )
        ):
            raise RuntimeQueueError("dispatch checkpoint is invalid")
        try:
            history = [dict(row) for row in checkpoint.history]
        except (TypeError, ValueError) as exc:
            raise RuntimeQueueError("dispatch checkpoint is invalid") from exc
        if any(
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or any(type(value) not in {int, str} for value in row.values())
            for row in history
        ):
            raise RuntimeQueueError("dispatch checkpoint is invalid")
        active = {
            request_id: entries[request_id]
            for request_id in active_request_ids
        }
        if any(
            type(row) is not tuple
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not int
            for row in checkpoint.policy_events
        ):
            raise RuntimeQueueError("dispatch checkpoint is invalid")
        release, arrivals, decisions, log = (
            self._restored_residency_hysteresis(checkpoint)
        )
        holds = checkpoint.residency_hysteresis_holds
        if type(holds) is not tuple or any(
            type(row) is not tuple or len(row) != 3
            or type(row[0]) is not str or not row[0]
            or type(row[1]) is not int or type(row[2]) is not int
            or row[1] < row[2] or row[2] < 0
            for row in holds
        ):
            raise RuntimeQueueError("dispatch checkpoint is invalid")
        with self._condition:
            self._entries = entries
            self._active = active
            self._sequence = checkpoint.sequence
            self._history = history
            self._policy_events = dict(checkpoint.policy_events)
            self._residency_release = release
            self._residency_hysteresis_holds = {
                row[0]: (row[1], row[2]) for row in holds
            }
            self._residency_arrivals = arrivals
            self._residency_hysteresis_decisions = decisions
            self._residency_hysteresis_log = log
            self._condition.notify_all()

    @staticmethod
    def _restored_residency_hysteresis(
        checkpoint: _DispatchQueueCheckpoint,
    ) -> tuple[
        tuple[int, str, tuple[str, ...]] | None,
        dict[str, ResidencyArrivalHistory],
        dict[str, ResidencyHysteresisDecision],
        list[ResidencyHysteresisDecision],
    ]:
        """Validate the checkpointed hysteresis state (fail-closed)."""
        try:
            release = validated_release(checkpoint.residency_release)
            arrivals = validated_arrivals(checkpoint.residency_arrivals)
            decisions = validated_current_decisions(
                checkpoint.residency_hysteresis_decisions
            )
            log = list(validated_decisions(checkpoint.residency_hysteresis_log))
        except ResidencyHysteresisError as exc:
            raise RuntimeQueueError("dispatch checkpoint is invalid") from exc
        return release, arrivals, decisions, log

    @staticmethod
    def _dispatch_key(entry: _DispatchEntry) -> tuple[int, int, str]:
        return (
            entry.decision.start_us,
            entry.sequence,
            entry.decision.request_id,
        )

    @staticmethod
    def _causal_graph_is_acyclic(
        entries: Mapping[str, _DispatchEntry],
    ) -> bool:
        visited: set[str] = set()
        active: set[str] = set()

        def visit(request_id: str) -> bool:
            if request_id in active:
                return False
            if request_id in visited:
                return True
            active.add(request_id)
            if any(
                not visit(predecessor_id)
                for predecessor_id in entries[
                    request_id
                ].predecessor_request_ids
            ):
                return False
            active.remove(request_id)
            visited.add(request_id)
            return True

        return all(visit(request_id) for request_id in entries)

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

    @staticmethod
    def _lane_tokens(decision: Decision) -> Mapping[tuple[str, int], frozenset[str]]:
        result: dict[tuple[str, int], set[str]] = {}
        for lease in decision.leases:
            for lane in lease.lanes:
                result.setdefault((lease.resource_id, lane), set()).add(lease.token)
        if result:
            return {lane: frozenset(tokens) for lane, tokens in result.items()}
        return {(f"route:{decision.route_id}", 0): frozenset({decision.request_id})}

    @classmethod
    def _resource_slots(cls, decision: Decision) -> dict[str, int]:
        result: dict[str, int] = {}
        for resource_id, _ in cls._lanes(decision):
            result[resource_id] = result.get(resource_id, 0) + 1
        return result

    @classmethod
    def _decisions_conflict(
        cls, first: Decision, second: Decision
    ) -> bool:
        first_lanes = cls._lane_tokens(first)
        second_lanes = cls._lane_tokens(second)
        return any(
            first_lanes[lane] != second_lanes[lane]
            for lane in set(first_lanes) & set(second_lanes)
        )

    @classmethod
    def _entries_conflict(
        cls, first: _DispatchEntry, second: _DispatchEntry
    ) -> bool:
        left, right = first.decode_cohort, second.decode_cohort
        if left is not None and right is not None and (
            left.cohort_id == right.cohort_id
            and left.key_sha256 == right.key_sha256
            and left.shared_lease_tokens == right.shared_lease_tokens
        ):
            return False
        return cls._decisions_conflict(first.decision, second.decision)

    @staticmethod
    def _validate_cohort(
        decision: Decision, binding: RuntimeDecodeCohortBinding | None
    ) -> None:
        if binding is not None and (
            not isinstance(binding, RuntimeDecodeCohortBinding)
            or decision.request_id not in binding.member_request_ids
            or not {row.token for row in decision.leases}.issubset(
                binding.shared_lease_tokens
            )
            or not decision.leases
        ):
            raise RuntimeQueueError(
                "dispatch decode cohort differs from its reservation"
            )

    def bind_decode_cohort(
        self, request_id: str, binding: RuntimeDecodeCohortBinding | None
    ) -> None:
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                raise RuntimeQueueError("dispatch decode cohort member is absent")
            self._validate_cohort(entry.decision, binding)
            if entry.decode_cohort == binding:
                return
            entry.decode_cohort = binding
            self._rebind_causal_predecessors(request_id)
            entry.generation += 1
            self._condition.notify_all()

    def _active_conflict(
        self,
        entry: _DispatchEntry,
        excluding_request_id: str | None = None,
    ) -> bool:
        return any(
            self._entries_conflict(entry, active)
            for request_id, active in self._active.items()
            if request_id != excluding_request_id
        )

    def _earlier_queued_conflict(
        self,
        entry: _DispatchEntry,
        excluding_request_id: str | None = None,
    ) -> bool:
        key = self._dispatch_key(entry)
        return any(
            other.state in {
                _DEFERRED_REPLAN,
                "QUEUED",
                "REPLANNING",
                "REPLAN_REQUIRED",
            }
            and request_id != excluding_request_id
            and self._dispatch_key(other) < key
            and not self._causally_depends_on(
                request_id, entry.decision.request_id
            )
            and self._entries_conflict(entry, other)
            for request_id, other in self._entries.items()
        )

    @staticmethod
    def _lane_windows(
        decision: Decision,
    ) -> Mapping[tuple[str, int], tuple[int, int]]:
        result: dict[tuple[str, int], tuple[int, int]] = {}
        for lease in decision.leases:
            for lane in lease.lanes:
                start_us, end_us = result.get(
                    (lease.resource_id, lane),
                    (lease.start_us, lease.reserved_until_us),
                )
                result[(lease.resource_id, lane)] = (
                    min(start_us, lease.start_us),
                    max(end_us, lease.reserved_until_us),
                )
        if result:
            return result
        return {
            (f"route:{decision.route_id}", 0): (
                decision.start_us, decision.finish_upper_us
            )
        }

    @classmethod
    def _leases_precede(
        cls,
        first: _DispatchEntry,
        second: _DispatchEntry,
        second_not_before_us: int | None = None,
    ) -> bool:
        """Whether every lane shared with ``second`` is free before it starts.

        ``second_not_before_us`` (a residency hysteresis hold) raises the
        start of ``second``'s windows.
        """
        first_windows = cls._lane_windows(first.decision)
        second_windows = cls._lane_windows(second.decision)
        shared = set(first_windows) & set(second_windows)
        return bool(shared) and all(
            first_windows[lane][1] <= max(
                second_windows[lane][0], second_not_before_us or 0
            )
            for lane in shared
        )

    def _residency_hysteresis_until(
        self, entry: _DispatchEntry
    ) -> int | None:
        """When a held residency change may start, or None when it is not held.

        A queued change of another model than the one whose request last
        released the exclusive residency resource may wait
        ``residency_hysteresis_s`` after that release when its decision for
        that release window holds it; a hold justified by a queued
        same-model request ends once no such request can still run first.
        """
        decision = self._residency_hysteresis_decision(entry)
        if decision is None or decision.held_until_us is None:
            return None
        if (
            decision.reason == REASON_SAME_MODEL_QUEUED
            and not self._same_model_pending(decision.released_key, entry)
        ):
            return None
        return decision.held_until_us

    def _residency_hysteresis_decision(
        self, entry: _DispatchEntry
    ) -> ResidencyHysteresisDecision | None:
        """The decision for a queued change in the current release window.

        Made once per change and window at its first evaluation while it is
        QUEUED (a change being replanned keeps the decision it had), logged
        and, for a skip, counted; None when hysteresis does not apply (off,
        no release, not a keyed change of another model).
        """
        release = self._residency_release
        key = entry.residency_hysteresis_key
        if (
            self._policy.residency_hysteresis_us == 0
            or release is None
            or not entry.residency_transition_barrier
            or key is None
            or key == release[1]
        ):
            return None
        request_id = entry.decision.request_id
        current = self._residency_hysteresis_decisions.get(request_id)
        if (
            current is not None
            and current.released_at_us == release[0]
            and current.released_key == release[1]
            and current.key == key
        ):
            return current
        if entry.state != "QUEUED":
            return None
        decision = decide_residency_hysteresis(
            request_id=request_id,
            key=key,
            release=release,
            change_resources=self._lane_resource_ids(entry.decision),
            own_model_waited_us=self._own_model_waited_us(key, release[0]),
            same_model_pending=self._same_model_pending(release[1], entry),
            arrivals=self._residency_arrivals.get(release[1]),
            window_us=self._policy.residency_hysteresis_us,
            minimum_probability_ppm=(
                self._policy.residency_hysteresis_min_probability_ppm
            ),
        )
        self._residency_hysteresis_decisions[request_id] = decision
        if len(self._residency_hysteresis_log) < DECISION_LOG_LIMIT:
            self._residency_hysteresis_log.append(decision)
        if not decision.held:
            self._count_policy_event("residency_hysteresis_skips")
        return decision

    @classmethod
    def _lane_resource_ids(cls, decision: Decision) -> frozenset[str]:
        return frozenset(resource_id for resource_id, _ in cls._lanes(decision))

    def _own_model_waited_us(self, key: str, at_us: int) -> int:
        """How long the oldest not yet dispatched work of ``key`` had waited at ``at_us``."""
        return max((
            at_us - (
                entry.admitted_at_us if entry.queued_since_us is None
                else entry.queued_since_us
            )
            for entry in self._entries.values()
            if entry.residency_hysteresis_key == key
            and entry.state in _NOT_DISPATCHED_STATES
        ), default=0)

    def _same_model_pending(
        self, key: str, change: _DispatchEntry
    ) -> bool:
        """Whether queued work of ``key`` could still dispatch before ``change``."""
        change_id = change.decision.request_id
        return any(
            other is not change
            and other.residency_hysteresis_key == key
            and other.state in _NOT_DISPATCHED_STATES
            and not self._causally_depends_on(request_id, change_id)
            for request_id, other in self._entries.items()
        )

    def _record_residency_arrival(
        self, key: str | None, admitted_at_us: int
    ) -> None:
        """Learn the inter-arrival gaps of a model from its first admissions."""
        if key is None or self._policy.residency_hysteresis_us == 0:
            return
        previous = self._residency_arrivals.get(key)
        self._residency_arrivals[key] = (
            ResidencyArrivalHistory(admitted_at_us) if previous is None
            else previous.admitted(admitted_at_us)
        )

    def _record_residency_hysteresis_hold(
        self, entry: _DispatchEntry, held_until_us: int
    ) -> None:
        """Count one hold of a residency change (once per release window)."""
        request_id = entry.decision.request_id
        released_at_us = held_until_us - self._policy.residency_hysteresis_us
        record = (held_until_us, released_at_us)
        if self._residency_hysteresis_holds.get(request_id) == record:
            return
        self._residency_hysteresis_holds[request_id] = record
        self._count_policy_event("residency_hysteresis_holds")

    def _count_residency_hysteresis_admission(
        self, entry: _DispatchEntry, admitted_at_us: int
    ) -> None:
        """Count a same-model admission while a residency change is held."""
        release = self._residency_release
        if (
            release is None
            or entry.residency_transition_barrier
            or entry.residency_hysteresis_key is None
            or entry.residency_hysteresis_key != release[1]
            or not any(
                other is not entry
                and other.state not in {"ACTIVE", "FINISHING"}
                and (self._residency_hysteresis_until(other) or 0)
                    > admitted_at_us
                for other in self._entries.values()
            )
        ):
            return
        self._count_policy_event("residency_hysteresis_admissions")

    def residency_hysteresis_until(self, request_id: str) -> int | None:
        """Public view of ``_residency_hysteresis_until`` for one queued entry."""
        with self._condition:
            entry = self._entries.get(_text("dispatch request_id", request_id))
            if entry is None or entry.state != "QUEUED":
                return None
            return self._residency_hysteresis_until(entry)

    def residency_hysteresis_decision(
        self, request_id: str
    ) -> Mapping[str, object] | None:
        """The last hold decision taken for one residency change, or None."""
        with self._condition:
            decision = self._residency_hysteresis_decisions.get(
                _text("dispatch request_id", request_id)
            )
            return None if decision is None else MappingProxyType(
                decision.to_json()
            )

    def residency_hysteresis_decisions(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        """Every hold decision in order (the first ``DECISION_LOG_LIMIT``)."""
        with self._condition:
            return tuple(
                MappingProxyType(row.to_json())
                for row in self._residency_hysteresis_log
            )

    def residency_hysteresis_hold(
        self, request_id: str
    ) -> Mapping[str, int] | None:
        """The last observed hold of one residency change, or None."""
        with self._condition:
            record = self._residency_hysteresis_holds.get(
                _text("dispatch request_id", request_id)
            )
            if record is None:
                return None
            return MappingProxyType({
                "held_until_us": record[0], "released_at_us": record[1],
            })

    def _frees_lanes_before(
        self, keeper: _DispatchEntry, barrier: _DispatchEntry
    ) -> bool:
        """Whether ``keeper`` releases its lanes before ``barrier`` can start.

        A queued residency change starts at its reservation. A cancelled one
        cannot start before its running predecessors end, so their reserved
        ends bound it; without a running predecessor it may start at once.
        """
        if keeper.state != "QUEUED":
            return False
        if barrier.state == "QUEUED":
            return self._leases_precede(
                keeper, barrier, self._residency_hysteresis_until(barrier)
            )
        if barrier.state not in {
            _DEFERRED_REPLAN, "REPLANNING", "REPLAN_REQUIRED"
        }:
            return False
        running_ends = [
            lease.reserved_until_us
            for request_id in barrier.predecessor_request_ids
            if request_id in self._entries
            and self._entries[request_id].state == "ACTIVE"
            for lease in self._entries[request_id].decision.leases
        ]
        keeper_windows = self._lane_windows(keeper.decision)
        shared = set(keeper_windows) & set(
            self._lane_windows(barrier.decision)
        )
        return bool(running_ends) and bool(shared) and all(
            keeper_windows[lane][1] <= max(running_ends) for lane in shared
        )

    def _other_dispatches_first(
        self,
        entry: _DispatchEntry,
        other: _DispatchEntry,
        *,
        preserve_arrival_order: bool,
    ) -> bool:
        if other.state == "ACTIVE":
            return True
        if preserve_arrival_order:
            return other.sequence < entry.sequence
        if entry.state == "ACTIVE":
            return False
        if (
            entry.residency_transition_barrier
            != other.residency_transition_barrier
            and self._policy.work_conserving_admission
        ):
            # Work that keeps the current residency runs first when its
            # lanes are free again before the residency change can start.
            barrier, keeper = (
                (entry, other)
                if entry.residency_transition_barrier
                else (other, entry)
            )
            if self._frees_lanes_before(keeper, barrier):
                return other is keeper
        if (
            entry.residency_transition_barrier
            or other.residency_transition_barrier
        ):
            return other.sequence < entry.sequence
        return self._dispatch_key(other) < self._dispatch_key(entry)

    def _bind_causal_predecessors(
        self,
        request_id: str,
        *,
        preserve_arrival_order: bool,
        precede_request_ids: frozenset[str] = frozenset(),
    ) -> None:
        entry = self._entries[request_id]
        for other_request_id in sorted(precede_request_ids):
            if not self._causally_depends_on(other_request_id, request_id):
                self._add_causal_predecessor(other_request_id, request_id)
        for other_request_id, other in self._entries.items():
            if (
                other_request_id == request_id
                or other.state == "FINISHING"
                or not self._entries_conflict(entry, other)
                or self._causally_depends_on(
                    request_id, other_request_id
                )
                or self._causally_depends_on(
                    other_request_id, request_id
                )
            ):
                continue
            if self._other_dispatches_first(
                entry, other, preserve_arrival_order=preserve_arrival_order
            ):
                self._add_causal_predecessor(
                    request_id, other_request_id
                )
            else:
                self._add_causal_predecessor(
                    other_request_id, request_id
                )

    def _rebind_causal_predecessors(self, request_id: str) -> None:
        entry = self._entries[request_id]
        entry_changed = False
        for other_request_id, other in self._entries.items():
            if other_request_id == request_id:
                continue
            if other.state == "FINISHING":
                if other_request_id in entry.predecessor_request_ids:
                    entry.predecessor_request_ids.remove(other_request_id)
                    entry_changed = True
                if request_id in other.predecessor_request_ids:
                    other.predecessor_request_ids.remove(request_id)
                    other.generation += 1
                continue
            if self._entries_conflict(entry, other):
                continue
            if other_request_id in entry.predecessor_request_ids:
                entry.predecessor_request_ids.remove(other_request_id)
                entry_changed = True
            if request_id in other.predecessor_request_ids:
                other.predecessor_request_ids.remove(request_id)
                other.generation += 1
        if entry_changed:
            entry.generation += 1
        for other_request_id, other in self._entries.items():
            if (
                other_request_id == request_id
                or other.state == "FINISHING"
                or not self._entries_conflict(entry, other)
                or self._causally_depends_on(
                    request_id, other_request_id
                )
                or self._causally_depends_on(
                    other_request_id, request_id
                )
            ):
                continue
            if self._other_dispatches_first(
                entry,
                other,
                preserve_arrival_order=(
                    not self._policy.work_conserving_admission
                ),
            ):
                self._add_causal_predecessor(
                    request_id, other_request_id
                )
            else:
                self._add_causal_predecessor(
                    other_request_id, request_id
                )
        if not self._causal_graph_is_acyclic(self._entries):
            raise RuntimeQueueError("dispatch causal dependency is cyclic")

    def _causally_depends_on(
        self, request_id: str, predecessor_request_id: str
    ) -> bool:
        pending = list(
            self._entries[request_id].predecessor_request_ids
        )
        seen = set()
        while pending:
            current = pending.pop()
            if current == predecessor_request_id:
                return True
            if current in seen or current not in self._entries:
                continue
            seen.add(current)
            pending.extend(
                self._entries[current].predecessor_request_ids
            )
        return False

    def _add_causal_predecessor(
        self, request_id: str, predecessor_request_id: str
    ) -> None:
        if (
            request_id == predecessor_request_id
            or self._causally_depends_on(
                predecessor_request_id, request_id
            )
        ):
            raise RuntimeQueueError("dispatch causal dependency is cyclic")
        self._entries[request_id].predecessor_request_ids.add(
            predecessor_request_id
        )
        self._entries[request_id].generation += 1

    def _causal_ready(self, entry: _DispatchEntry) -> bool:
        return not any(
            request_id in self._entries
            for request_id in entry.predecessor_request_ids
        )

    def _promote_deferred_frontier(self, reason: str) -> None:
        occupied_lanes: set[tuple[str, int]] = set()
        for entry in sorted(
            self._entries.values(), key=lambda row: row.sequence
        ):
            if entry.state != _DEFERRED_REPLAN:
                continue
            if (
                not self._causal_ready(entry)
                or self._active_conflict(entry)
                or self._earlier_queued_conflict(entry)
            ):
                continue
            lanes = set(self._lane_tokens(entry.decision))
            if lanes & occupied_lanes:
                continue
            entry.state = "REPLAN_REQUIRED"
            if entry.wake_reason == "calendar":
                entry.wake_reason = reason
            entry.generation += 1
            occupied_lanes.update(lanes)

    def _release_causal_frontier(
        self, predecessor_request_id: str
    ) -> tuple[_DispatchEntry, ...]:
        affected = []
        for entry in sorted(
            self._entries.values(), key=lambda row: row.sequence
        ):
            if predecessor_request_id not in entry.predecessor_request_ids:
                continue
            entry.predecessor_request_ids.remove(predecessor_request_id)
            if (
                entry.state in {"QUEUED", _DEFERRED_REPLAN}
                and self._causal_ready(entry)
                and not self._active_conflict(entry)
                and not self._earlier_queued_conflict(entry)
            ):
                affected.append(entry)
        return tuple(affected)

    def _follower_frontier(
        self,
        entry: _DispatchEntry,
        candidate_request_ids: frozenset[str] | None = None,
    ) -> tuple[_DispatchEntry, ...]:
        """Return the first later owner for each conflicting lane."""

        frontier_by_lane: dict[tuple[str, int], _DispatchEntry] = {}
        entry_lanes = set(self._lane_tokens(entry.decision))
        for request_id, other in self._entries.items():
            if (
                request_id == entry.decision.request_id
                or other.state != "QUEUED"
                or (
                    candidate_request_ids is not None
                    and request_id not in candidate_request_ids
                )
            ):
                continue
            shared_lanes = entry_lanes & set(
                self._lane_tokens(other.decision)
            )
            if not shared_lanes:
                continue
            if (
                entry.decision.request_id
                    not in other.predecessor_request_ids
                and other.sequence <= entry.sequence
            ):
                continue
            for lane in shared_lanes:
                previous = frontier_by_lane.get(lane)
                if previous is None or other.sequence < previous.sequence:
                    frontier_by_lane[lane] = other
        unique = {
            row.decision.request_id: row
            for row in frontier_by_lane.values()
        }
        return tuple(sorted(unique.values(), key=lambda row: row.sequence))

    def causal_follower_frontier(
        self,
        request_id: str,
        candidate_request_ids: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Return a bounded causal frontier for one queued or active owner."""

        request_id = _text("causal frontier request_id", request_id)
        candidates = tuple(
            _text("causal frontier candidate", value)
            for value in candidate_request_ids
        )
        if len(candidates) != len(set(candidates)):
            raise RuntimeQueueError(
                "causal frontier candidates are duplicated"
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                raise RuntimeQueueError(
                    "causal frontier owner is absent"
                )
            frontier = self._follower_frontier(
                entry,
                None if not candidates else frozenset(candidates),
            )
            return tuple(
                row.decision.request_id
                for row in frontier
                if not any(
                    predecessor_id != request_id
                    and predecessor_id in self._entries
                    for predecessor_id in row.predecessor_request_ids
                )
                and not self._active_conflict(
                    row, excluding_request_id=request_id
                )
                and not self._earlier_queued_conflict(row)
            )

    def downstream_projection_requests(
        self, request_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Return queued projections causally behind a runnable frontier."""

        roots = tuple(
            _text("projection frontier request_id", value)
            for value in request_ids
        )
        if not roots or len(roots) != len(set(roots)):
            raise RuntimeQueueError(
                "projection frontier request ids are invalid"
            )
        with self._condition:
            if any(request_id not in self._entries for request_id in roots):
                raise RuntimeQueueError(
                    "projection frontier request is absent"
                )
            root_set = frozenset(roots)
            return tuple(
                entry.decision.request_id
                for entry in sorted(
                    self._entries.values(), key=lambda row: row.sequence
                )
                if entry.state == "QUEUED"
                and entry.decision.request_id not in root_set
                and any(
                    self._causally_depends_on(
                        entry.decision.request_id, root_id
                    )
                    for root_id in roots
                )
            )

    def priority_compaction_followers(
        self,
        request_id: str,
        expected_generation: int | None = None,
    ) -> tuple[str, ...]:
        """Return later queued reservations that block priority repair."""

        request_id = _text(
            "priority compaction request_id", request_id
        )
        if expected_generation is not None:
            expected_generation = _nonnegative_int(
                "priority compaction generation", expected_generation
            )
        with self._condition:
            root = self._entries.get(request_id)
            if (
                root is None
                or root.state != "REPLAN_REQUIRED"
                or (
                    expected_generation is not None
                    and root.generation != expected_generation
                )
            ):
                raise RuntimeQueueError(
                    "priority compaction root generation changed"
                )
            affected_lanes = set(self._lane_tokens(root.decision))
            selected = []
            for entry in sorted(
                self._entries.values(), key=lambda row: row.sequence
            ):
                if (
                    entry.sequence <= root.sequence
                    or entry.state != "QUEUED"
                ):
                    continue
                entry_lanes = set(self._lane_tokens(entry.decision))
                if not affected_lanes.intersection(entry_lanes):
                    continue
                selected.append(entry.decision.request_id)
                affected_lanes.update(entry_lanes)
            return tuple(selected)

    def promote_priority_compaction_follower(
        self,
        request_id: str,
        observed_at_us: int,
    ) -> RuntimeDispatchReceipt:
        """Expose one detached follower for ordered replacement."""

        request_id = _text(
            "priority compaction follower request_id", request_id
        )
        observed_at_us = _nonnegative_int(
            "priority compaction observation time", observed_at_us
        )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state not in {
                _DEFERRED_REPLAN, "REPLAN_REQUIRED"
            }:
                raise RuntimeQueueError(
                    "priority compaction follower state changed"
                )
            if entry.state == _DEFERRED_REPLAN:
                entry.state = "REPLAN_REQUIRED"
                entry.wake_reason = "priority_compaction_follower"
                entry.generation += 1
            receipt = RuntimeDispatchReceipt(
                request_id=request_id,
                route_id=entry.decision.route_id,
                status="REPLAN_REQUIRED",
                scheduled_start_us=entry.decision.start_us,
                observed_at_us=observed_at_us,
                queue_wait_us=max(
                    0, observed_at_us - entry.admitted_at_us
                ),
                wake_reason=entry.wake_reason,
                queue_generation=entry.generation,
            )
            self._condition.notify_all()
            return receipt

    def priority_compaction_order(
        self, request_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Return selected requests in their immutable queue order."""

        rows = tuple(
            _text("priority compaction request_id", value)
            for value in request_ids
        )
        if len(rows) != len(set(rows)):
            raise RuntimeQueueError(
                "priority compaction requests are duplicated"
            )
        with self._condition:
            if any(value not in self._entries for value in rows):
                raise RuntimeQueueError(
                    "priority compaction request is absent"
                )
            return tuple(
                entry.decision.request_id
                for entry in sorted(
                    (self._entries[value] for value in rows),
                    key=lambda entry: entry.sequence,
                )
            )

    def replanned_capacity_frontier(
        self,
        request_id: str,
        previous_decision: Decision,
        resource_capacities: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Return the first causal successors that fit after a replan."""

        request_id = _text(
            "replanned capacity request_id", request_id
        )
        if (
            not isinstance(previous_decision, Decision)
            or previous_decision.request_id != request_id
        ):
            raise RuntimeQueueError(
                "replanned capacity decision identity differs"
            )
        if resource_capacities is not None and (
            not isinstance(resource_capacities, Mapping)
            or any(
                type(resource_id) is not str
                or not resource_id
                or not resource_id.isascii()
                or type(capacity) is not int
                or capacity <= 0
                for resource_id, capacity in resource_capacities.items()
            )
        ):
            raise RuntimeQueueError(
                "replanned resource capacities are invalid"
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "QUEUED":
                raise RuntimeQueueError(
                    "replanned capacity owner is not queued"
                )
            predecessor_resources = {
                resource_id
                for resource_id, _lane in self._lane_tokens(
                    previous_decision
                )
            }
            lane_universe = set(self._lane_tokens(previous_decision))
            for candidate in self._entries.values():
                lane_universe.update(
                    self._lane_tokens(candidate.decision)
                )
            for resource_id, capacity in (
                {} if resource_capacities is None else resource_capacities
            ).items():
                lane_universe.update(
                    (resource_id, lane) for lane in range(capacity)
                )
            occupied_lanes = set(self._lane_tokens(entry.decision))
            for candidate in self._entries.values():
                if candidate is entry or candidate.state == "FINISHING":
                    continue
                if candidate.state == "ACTIVE" or (
                    candidate.state == "QUEUED"
                    and candidate.sequence < entry.sequence
                    and candidate.decision.start_us
                        <= entry.decision.start_us
                ):
                    occupied_lanes.update(
                        self._lane_tokens(candidate.decision)
                    )
            available_by_resource: dict[str, set[tuple[str, int]]] = {}
            for lane in lane_universe - occupied_lanes:
                available_by_resource.setdefault(lane[0], set()).add(lane)
            representative_slots = self._resource_slots(entry.decision)
            selected = []
            for other in sorted(
                self._entries.values(), key=lambda row: row.sequence
            ):
                if (
                    other is entry
                    or other.state not in {"QUEUED", _DEFERRED_REPLAN}
                    or request_id not in other.predecessor_request_ids
                    or any(
                        predecessor_id != request_id
                        and predecessor_id in self._entries
                        for predecessor_id
                        in other.predecessor_request_ids
                    )
                ):
                    continue
                other_lanes = set(self._lane_tokens(other.decision))
                if not predecessor_resources.intersection(
                    resource_id for resource_id, _lane in other_lanes
                ):
                    continue
                needed_by_resource = {
                    resource_id: min(slots, representative_slots.get(resource_id, slots))
                    for resource_id, slots in self._resource_slots(other.decision).items()
                }
                if any(
                    len(available_by_resource.get(resource_id, ()))
                        < needed
                    for resource_id, needed in needed_by_resource.items()
                ):
                    continue
                selected.append(other.decision.request_id)
                for resource_id, needed in needed_by_resource.items():
                    free = available_by_resource[resource_id]
                    for lane in sorted(free)[:needed]:
                        free.remove(lane)
            return tuple(
                selected
            )

    def early_completion_frontier(
        self,
        request_id: str,
        completed_at_us: int,
        final_reserved_until_us: Mapping[str, int],
        resource_capacities: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Return one runnable successor per lane released ahead of plan."""

        request_id = _text(
            "early completion request_id", request_id
        )
        completed_at_us = _nonnegative_int(
            "early completion at_us", completed_at_us
        )
        if not isinstance(final_reserved_until_us, Mapping):
            raise RuntimeQueueError(
                "early completion lease coverage is invalid"
            )
        if resource_capacities is not None and (
            not isinstance(resource_capacities, Mapping)
            or any(
                type(resource_id) is not str
                or not resource_id
                or not resource_id.isascii()
                or type(capacity) is not int
                or capacity <= 0
                for resource_id, capacity in resource_capacities.items()
            )
        ):
            raise RuntimeQueueError(
                "early completion resource capacities are invalid"
            )
        with self._condition:
            owner = self._entries.get(request_id)
            if (
                owner is None
                or owner.state != "ACTIVE"
                or self._active.get(request_id) is not owner
            ):
                raise RuntimeQueueError(
                    "early completion request is not active"
                )
            lease_by_token = {
                lease.token: lease for lease in owner.decision.leases
            }
            if set(final_reserved_until_us) != set(lease_by_token) or any(
                type(value) is not int
                or value < lease_by_token[token].reserved_until_us
                for token, value in final_reserved_until_us.items()
            ):
                raise RuntimeQueueError(
                    "early completion lease coverage differs from decision"
                )
            early_lanes = {
                (lease.resource_id, lane)
                for lease in owner.decision.leases
                if completed_at_us < final_reserved_until_us[lease.token]
                for lane in lease.lanes
            }
            if not early_lanes:
                return ()
            capacity_by_resource = dict(resource_capacities or {})
            for entry in self._entries.values():
                for resource_id, lane in self._lane_tokens(entry.decision):
                    capacity_by_resource[resource_id] = max(
                        capacity_by_resource.get(resource_id, 0), lane + 1
                    )
            occupied_by_resource: dict[str, set[int]] = {}
            for active in self._active.values():
                if active is owner:
                    continue
                for resource_id, lane in self._lane_tokens(
                    active.decision
                ):
                    occupied_by_resource.setdefault(
                        resource_id, set()
                    ).add(lane)
            available_by_resource = {
                resource_id: max(
                    0,
                    capacity - len(occupied_by_resource.get(
                        resource_id, ()
                    )),
                )
                for resource_id, capacity in capacity_by_resource.items()
            }
            owner_slots = self._resource_slots(owner.decision)
            selected = []
            selected_ids: set[str] = set()
            for other in sorted(
                self._entries.values(), key=lambda row: row.sequence
            ):
                if (
                    other is owner
                    or other.state not in {"QUEUED", _DEFERRED_REPLAN}
                    or (
                        request_id not in other.predecessor_request_ids
                        and not self._backfills_early_capacity(
                            other, completed_at_us
                        )
                    )
                    or any(
                        predecessor_id != request_id
                        and predecessor_id in self._entries
                        and predecessor_id not in selected_ids
                        and self._entries[predecessor_id].state not in {
                            "ACTIVE", "FINISHING"
                        }
                        for predecessor_id
                        in other.predecessor_request_ids
                    )
                ):
                    continue
                needed_by_resource = {
                    resource_id: min(slots, owner_slots.get(resource_id, slots))
                    for resource_id, slots in self._resource_slots(other.decision).items()
                }
                shared_resources = (
                    set(needed_by_resource)
                    & {resource_id for resource_id, _ in early_lanes}
                )
                if not shared_resources:
                    continue
                if any(
                    available_by_resource.get(resource_id, 0) < needed
                    for resource_id, needed in needed_by_resource.items()
                ):
                    continue
                selected.append(other.decision.request_id)
                selected_ids.add(other.decision.request_id)
                for resource_id, needed in needed_by_resource.items():
                    available_by_resource[resource_id] -= needed
            return tuple(selected)

    def _backfills_early_capacity(
        self, entry: _DispatchEntry, completed_at_us: int
    ) -> bool:
        """Queued work of the current residency may take capacity freed early.

        It waits for a later reservation although it does not depend on the
        completed owner; replanning it lets it start in the freed lanes.
        """
        return (
            self._policy.work_conserving_admission
            and entry.state == "QUEUED"
            and not entry.residency_transition_barrier
            and entry.decision.start_us > completed_at_us
        )

    def idle_completion_frontier(
        self,
        request_id: str,
        completed_at_us: int,
        resource_capacities: Mapping[str, int] | None = None,
    ) -> tuple[str, ...]:
        """Return a capacity-bounded runnable frontier after an idle release."""

        request_id = _text(
            "idle completion request_id", request_id
        )
        completed_at_us = _nonnegative_int(
            "idle completion at_us", completed_at_us
        )
        if resource_capacities is not None and (
            not isinstance(resource_capacities, Mapping)
            or any(
                type(resource_id) is not str
                or not resource_id
                or not resource_id.isascii()
                or type(capacity) is not int
                or capacity <= 0
                for resource_id, capacity in resource_capacities.items()
            )
        ):
            raise RuntimeQueueError(
                "idle completion resource capacities are invalid"
            )
        with self._condition:
            owner = self._entries.get(request_id)
            if (
                owner is None
                or owner.state != "ACTIVE"
                or self._active.get(request_id) is not owner
            ):
                raise RuntimeQueueError(
                    "idle completion request is not active"
                )
            if len(self._active) != 1:
                return ()
            capacity_by_resource = dict(resource_capacities or {})
            for entry in self._entries.values():
                for resource_id, lane in self._lane_tokens(entry.decision):
                    capacity_by_resource[resource_id] = max(
                        capacity_by_resource.get(resource_id, 0), lane + 1
                    )
            occupied_by_resource: dict[str, set[int]] = {}
            for active in self._active.values():
                if active is owner:
                    continue
                for resource_id, lane in self._lane_tokens(
                    active.decision
                ):
                    occupied_by_resource.setdefault(
                        resource_id, set()
                    ).add(lane)
            available_by_resource = {
                resource_id: max(
                    0,
                    capacity - len(occupied_by_resource.get(
                        resource_id, ()
                    )),
                )
                for resource_id, capacity in capacity_by_resource.items()
            }
            selected = []
            runnable_ids: set[str] = set()
            for entry in sorted(
                self._entries.values(), key=lambda row: row.sequence
            ):
                if (
                    entry is owner
                    or entry.state not in {
                        "QUEUED", _DEFERRED_REPLAN
                    }
                    or (
                        entry.state == _DEFERRED_REPLAN
                        and entry.wake_reason
                            != "residency_observation_changed"
                    )
                    or any(
                        predecessor_id != request_id
                        and predecessor_id not in runnable_ids
                        and predecessor_id in self._entries
                        and self._entries[predecessor_id].state not in {
                            "ACTIVE", "FINISHING"
                        }
                        for predecessor_id
                        in entry.predecessor_request_ids
                    )
                ):
                    continue
                needed_by_resource = self._resource_slots(entry.decision)
                if any(
                    available_by_resource.get(resource_id, 0) < needed
                    for resource_id, needed in needed_by_resource.items()
                ):
                    continue
                runnable_ids.add(entry.decision.request_id)
                if (
                    entry.state == _DEFERRED_REPLAN
                    or entry.decision.start_us > completed_at_us
                ):
                    selected.append(entry.decision.request_id)
                for resource_id, needed in needed_by_resource.items():
                    available_by_resource[resource_id] -= needed
            return tuple(selected)

    def admit(
        self,
        decision: Decision,
        admitted_at_us: int,
        *,
        residency_transition_barrier: bool = False,
        decode_cohort: RuntimeDecodeCohortBinding | None = None,
        precede_request_ids: Sequence[str] = (),
        retained_order: tuple[int, Sequence[str]] | None = None,
        residency_hysteresis_key: str | None = None,
    ) -> None:
        """Queue one decision; ``retained_order`` keeps a failed attempt's place.

        ``retained_order`` is ``(sequence, follower_request_ids)`` of the
        request's failed attempt (elastic-phone recovery): the new attempt
        takes that earlier sequence, and the followers that waited on the
        failed attempt wait on the new one. ``residency_hysteresis_key``
        names the model of a decision holding the exclusive residency
        resource (residency hysteresis only).
        """
        if not isinstance(decision, Decision):
            raise RuntimeQueueError("dispatch admission requires a Decision")
        if not _optional_key(residency_hysteresis_key):
            raise RuntimeQueueError(
                "dispatch residency hysteresis key is invalid"
            )
        admitted_at_us = _nonnegative_int(
            "dispatch admitted_at_us", admitted_at_us
        )
        if type(residency_transition_barrier) is not bool:
            raise RuntimeQueueError(
                "dispatch residency transition barrier is invalid"
            )
        precede = tuple(
            _text("dispatch precedence request_id", value)
            for value in precede_request_ids
        )
        if (
            len(precede) != len(set(precede))
            or decision.request_id in precede
            or (precede and not self._policy.precedence_enabled)
        ):
            raise RuntimeQueueError("dispatch precedence is invalid")
        retained_sequence, retained_followers = (
            (None, ()) if retained_order is None
            else self._retained_order(decision.request_id, retained_order)
        )
        self._validate_cohort(decision, decode_cohort)
        with self._condition:
            if any(
                request_id not in self._entries
                or self._entries[request_id].state in {"ACTIVE", "FINISHING"}
                for request_id in precede
            ):
                raise RuntimeQueueError(
                    "dispatch precedence target is not queued"
                )
            if retained_sequence is not None and (
                decision.request_id in self._entries
                or retained_sequence > self._sequence
                or any(
                    entry.sequence == retained_sequence
                    for entry in self._entries.values()
                )
            ):
                raise RuntimeQueueError("dispatch retained order is invalid")
            previous = self._entries.get(decision.request_id)
            if (
                precede
                and previous is not None
                and previous.state != "REPLANNING"
            ):
                raise RuntimeQueueError(
                    "dispatch precedence requires a new admission or a replan"
                )
            if previous is not None and previous.state == "REPLANNING":
                replan_reason = previous.wake_reason
                previous.decision = decision
                previous.admitted_at_us = admitted_at_us
                previous.residency_transition_barrier = (
                    residency_transition_barrier
                )
                previous.residency_hysteresis_key = residency_hysteresis_key
                previous.decode_cohort = decode_cohort
                previous.state = "QUEUED"
                previous.wake_reason = "replanned"
                previous.generation += 1
                self._count_residency_hysteresis_admission(
                    previous, admitted_at_us
                )
                # The displaced work now waits on this attempt instead.
                previous.predecessor_request_ids.difference_update(precede)
                for follower_id in sorted(precede):
                    if not self._causally_depends_on(
                        follower_id, decision.request_id
                    ):
                        self._add_causal_predecessor(
                            follower_id, decision.request_id
                        )
                self._rebind_causal_predecessors(decision.request_id)
                if (
                    replan_reason in {
                        "external_phase_lease_overrun",
                        "lease_upper_bound_overrun",
                    }
                    or not previous.predecessor_request_ids
                ):
                    self._promote_deferred_frontier(
                        "predecessor_replanned"
                    )
                if not self._active:
                    self._promote_deferred_frontier(
                        "capacity_released_early"
                    )
                self._condition.notify_all()
                return
            if previous is not None:
                raise RuntimeQueueError("request already exists in dispatch queue")
            if retained_sequence is None:
                self._sequence += 1
            entry = _DispatchEntry(
                decision,
                admitted_at_us,
                (
                    self._sequence
                    if retained_sequence is None
                    else retained_sequence
                ),
                residency_transition_barrier,
                decode_cohort,
                residency_hysteresis_key=residency_hysteresis_key,
                queued_since_us=admitted_at_us,
            )
            self._entries[decision.request_id] = entry
            if retained_sequence is None:
                self._record_residency_arrival(
                    residency_hysteresis_key, admitted_at_us
                )
            self._count_residency_hysteresis_admission(entry, admitted_at_us)
            self._bind_causal_predecessors(
                decision.request_id,
                preserve_arrival_order=False,
                precede_request_ids=frozenset(precede).union(
                    request_id for request_id in retained_followers
                    if request_id in self._entries
                    and self._entries[request_id].state
                    not in {"ACTIVE", "FINISHING"}
                    and self._entries_conflict(
                        entry, self._entries[request_id]
                    )
                ),
            )
            self._condition.notify_all()

    @staticmethod
    def _retained_order(
        request_id: str, retained_order: object
    ) -> tuple[int, tuple[str, ...]]:
        try:
            sequence, followers = retained_order
            followers = tuple(
                _text("dispatch retained follower", value)
                for value in followers
            )
        except (TypeError, ValueError, RuntimeQueueError) as exc:
            raise RuntimeQueueError(
                "dispatch retained order is invalid"
            ) from exc
        if (
            type(sequence) is not int
            or sequence < 1
            or len(followers) != len(set(followers))
            or request_id in followers
        ):
            raise RuntimeQueueError("dispatch retained order is invalid")
        return sequence, followers

    def release_prepare_leases(self, request_id: str, tokens: Sequence[str], *,
                               preparation_complete: bool = True) -> None:
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "ACTIVE":
                raise RuntimeQueueError("prepare phase owner is not active")
            released = set(tokens)
            if not released or not released.issubset({
                row.token for row in entry.decision.leases
            }):
                raise RuntimeQueueError("prepare phase tokens differ from owner")
            leases = tuple(
                row for row in entry.decision.leases if row.token not in released
            )
            if not leases:
                raise RuntimeQueueError("prepare phase cannot release execution leases")
            entry.decision = replace(entry.decision, leases=leases)
            entry.residency_transition_barrier = not preparation_complete
            self._rebind_causal_predecessors(request_id)
            entry.generation += 1
            if preparation_complete and self._policy.work_conserving_admission:
                self._promote_deferred_behind_published_work()
                deferred = {
                    other_id for other_id, other in self._entries.items()
                    if other.state == _DEFERRED_REPLAN
                }
                self._promote_deferred_frontier("preparation_phase_completed")
                for other_id in sorted(deferred):
                    if self._entries[other_id].state == "REPLAN_REQUIRED":
                        self._count_policy_event("published_work_promotions")
            self._condition.notify_all()

    def _promote_deferred_behind_published_work(self) -> None:
        """Replan deferred work once everything it waits on is running.

        Every predecessor is active past its residency change (or is promoted
        in this pass), so a replan observes the residency those predecessors
        leave. Deferrals caused by an invalid residency projection keep
        waiting for completion.
        """
        promoted: set[str] = set()
        for entry in sorted(
            self._entries.values(), key=lambda row: row.sequence
        ):
            if (
                entry.state != _DEFERRED_REPLAN
                or entry.wake_reason == "residency_projection_invalid"
            ):
                continue
            predecessors = tuple(
                request_id
                for request_id in entry.predecessor_request_ids
                if request_id in self._entries
            )
            if not predecessors or any(
                request_id not in promoted
                and (
                    self._entries[request_id].state != "ACTIVE"
                    or self._entries[request_id].residency_transition_barrier
                )
                for request_id in predecessors
            ):
                continue
            entry.state = "REPLAN_REQUIRED"
            entry.generation += 1
            promoted.add(entry.decision.request_id)
            self._count_policy_event("published_work_promotions")

    def _count_policy_event(self, name: str) -> None:
        self._policy_events[name] = self._policy_events.get(name, 0) + 1

    def policy_events(self) -> Mapping[str, int]:
        with self._condition:
            return MappingProxyType(dict(sorted(self._policy_events.items())))

    def replace_decision(
        self,
        request_id: str,
        expected: Decision,
        replacement: Decision,
    ) -> None:
        """Replace queue metadata without changing physical reservations."""
        request_id = _text("dispatch request_id", request_id)
        if not isinstance(expected, Decision) or not isinstance(
            replacement, Decision
        ):
            raise RuntimeQueueError(
                "dispatch decision replacement is invalid"
            )
        if (
            expected.request_id != request_id
            or replacement.request_id != request_id
            or expected.route_id != replacement.route_id
            or tuple(row.token for row in expected.leases)
                != tuple(row.token for row in replacement.leases)
            or tuple(
                (row.resource_id, row.lanes, row.start_us)
                for row in expected.leases
            ) != tuple(
                (row.resource_id, row.lanes, row.start_us)
                for row in replacement.leases
            )
        ):
            raise RuntimeQueueError(
                "dispatch decision replacement changes placement"
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.decision != expected:
                raise RuntimeQueueError(
                    "dispatch decision replacement state differs"
                )
            entry.decision = replacement
            entry.generation += 1
            self._condition.notify_all()

    def wait_ready(
        self,
        request_id: str,
        epoch_ns: int,
    ) -> RuntimeDispatchReceipt:
        """Wait until dispatch can be committed without mutating queue state."""
        request_id = _text("dispatch request_id", request_id)
        epoch_ns = _nonnegative_int("dispatch epoch_ns", epoch_ns)
        with self._condition:
            while True:
                entry = self._entries.get(request_id)
                if entry is None:
                    raise RuntimeQueueError("request is absent from dispatch queue")
                if self._wake_holds:
                    self._condition.wait()
                    continue
                decision = entry.decision
                now_us = max(0, (time.monotonic_ns() - epoch_ns) // 1000)
                if entry.state == "REPLAN_REQUIRED":
                    earlier_replan = any(
                        request_id != other_request_id
                        and other.state in {
                            "REPLANNING", "REPLAN_REQUIRED"
                        }
                        and other.sequence < entry.sequence
                        and not self._causally_depends_on(
                            other_request_id, request_id
                        )
                        and self._entries_conflict(entry, other)
                        for other_request_id, other in self._entries.items()
                    )
                    if earlier_replan:
                        self._condition.wait()
                        continue
                    return RuntimeDispatchReceipt(
                        request_id=request_id,
                        route_id=decision.route_id,
                        status="REPLAN_REQUIRED",
                        scheduled_start_us=decision.start_us,
                        observed_at_us=now_us,
                        queue_wait_us=max(0, now_us - entry.admitted_at_us),
                        wake_reason=entry.wake_reason,
                        queue_generation=entry.generation,
                    )
                if entry.state == _DEFERRED_REPLAN:
                    self._condition.wait()
                    continue
                if entry.state == "REPLANNING":
                    self._condition.wait()
                    continue
                if entry.state != "QUEUED":
                    raise RuntimeQueueError("request has invalid queue state")
                resource_ready = (
                    self._causal_ready(entry)
                    and not self._active_conflict(entry)
                    and not self._earlier_queued_conflict(entry)
                )
                held_until_us = self._residency_hysteresis_until(entry)
                start_us = decision.start_us
                if held_until_us is not None and held_until_us > start_us:
                    start_us = held_until_us
                    if resource_ready and now_us < held_until_us:
                        self._record_residency_hysteresis_hold(
                            entry, held_until_us
                        )
                if resource_ready and now_us >= start_us:
                    wake_reason = entry.wake_reason
                    if wake_reason == "calendar":
                        wake_reason = (
                            "calendar_deadline"
                            if now_us <= decision.start_us + 1000
                            else "calendar_elapsed"
                        )
                    if (
                        start_us > decision.start_us
                        and request_id in self._residency_hysteresis_holds
                    ):
                        wake_reason = RESIDENCY_HYSTERESIS_WAKE_REASON
                    receipt = RuntimeDispatchReceipt(
                        request_id=request_id,
                        route_id=decision.route_id,
                        status="ACQUIRED",
                        scheduled_start_us=decision.start_us,
                        observed_at_us=now_us,
                        queue_wait_us=max(0, now_us - entry.admitted_at_us),
                        wake_reason=wake_reason,
                        queue_generation=entry.generation,
                    )
                    return receipt
                timeout_s = None
                if resource_ready and now_us < start_us:
                    timeout_s = (start_us - now_us) / 1_000_000
                self._condition.wait(timeout=timeout_s)

    def commit_ready(
        self, receipt: RuntimeDispatchReceipt
    ) -> RuntimeDispatchReceipt | None:
        """Commit one readiness receipt after its owner enters a transaction."""
        if not isinstance(receipt, RuntimeDispatchReceipt):
            raise RuntimeQueueError("dispatch readiness receipt is invalid")
        if receipt.status != "ACQUIRED":
            raise RuntimeQueueError("dispatch readiness is not an acquisition")
        with self._condition:
            entry = self._entries.get(receipt.request_id)
            if (
                entry is None
                or entry.state != "QUEUED"
                or entry.decision.route_id != receipt.route_id
                or entry.decision.start_us != receipt.scheduled_start_us
                or entry.generation != receipt.queue_generation
                or self._wake_holds
                or not self._causal_ready(entry)
                or self._active_conflict(entry)
                or self._earlier_queued_conflict(entry)
                or (self._residency_hysteresis_until(entry) or 0)
                    > receipt.observed_at_us
            ):
                return None
            if (
                entry.wake_reason in {
                    "predecessor_cancelled",
                    "predecessor_completion",
                }
                and any(
                    lease.reserved_until_us <= receipt.observed_at_us
                    for lease in entry.decision.leases
                )
            ):
                entry.state = "REPLAN_REQUIRED"
                entry.wake_reason = "lease_coverage_expired_before_dispatch"
                entry.generation += 1
                self._condition.notify_all()
                return RuntimeDispatchReceipt(
                    request_id=receipt.request_id,
                    route_id=receipt.route_id,
                    status="REPLAN_REQUIRED",
                    scheduled_start_us=receipt.scheduled_start_us,
                    observed_at_us=receipt.observed_at_us,
                    queue_wait_us=receipt.queue_wait_us,
                    wake_reason=entry.wake_reason,
                    queue_generation=entry.generation,
                )
            entry.state = "ACTIVE"
            entry.generation += 1
            self._active[receipt.request_id] = entry
            committed = RuntimeDispatchReceipt(
                request_id=receipt.request_id,
                route_id=receipt.route_id,
                status=receipt.status,
                scheduled_start_us=receipt.scheduled_start_us,
                observed_at_us=receipt.observed_at_us,
                queue_wait_us=receipt.queue_wait_us,
                wake_reason=receipt.wake_reason,
                queue_generation=entry.generation,
            )
            self._history.append(committed.to_json())
            self._condition.notify_all()
            return committed

    def wait(
        self,
        request_id: str,
        epoch_ns: int,
    ) -> RuntimeDispatchReceipt:
        """Wait for and directly commit dispatch for queue-only callers."""
        while True:
            receipt = self.wait_ready(request_id, epoch_ns)
            if receipt.status != "ACQUIRED":
                return receipt
            committed = self.commit_ready(receipt)
            if committed is not None:
                return committed

    def _release_capacity_locked(
        self,
        request_id: str,
        completed_at_us: int,
        early_replans: tuple[str, ...],
        enforce_work_conserving: bool,
    ) -> None:
        early_replan_set = frozenset(early_replans)
        entry = self._entries.get(request_id)
        if entry is None or entry.state != "ACTIVE":
            raise RuntimeQueueError("released request is not active")
        route_id = entry.decision.route_id
        if self._active.get(request_id) is not entry:
            raise RuntimeQueueError("active route ownership mismatch")
        del self._active[request_id]
        entry.state = "FINISHING"
        entry.capacity_released_at_us = completed_at_us
        if (
            entry.residency_hysteresis_key is not None
            and self._policy.residency_hysteresis_us > 0
        ):
            self._residency_release = (
                completed_at_us,
                entry.residency_hysteresis_key,
                tuple(sorted(self._lane_resource_ids(entry.decision))),
            )
        frontier = self._release_causal_frontier(request_id)
        for queued_request_id in early_replans:
            queued = self._entries.get(queued_request_id)
            if (
                queued is None
                or queued.state not in {
                    _DEFERRED_REPLAN, "REPLAN_REQUIRED"
                }
                or queued.wake_reason != "capacity_released_early"
            ):
                raise RuntimeQueueError(
                    "early completion replan state differs"
                )
        for queued in frontier:
            queued_request_id = queued.decision.request_id
            if queued_request_id in early_replan_set:
                if (
                    queued.state != _DEFERRED_REPLAN
                    or queued.wake_reason != "capacity_released_early"
                ):
                    raise RuntimeQueueError(
                        "early completion replan state differs"
                    )
                queued.state = "REPLAN_REQUIRED"
                queued.generation += 1
                continue
            if queued.state == "QUEUED":
                queued.wake_reason = "predecessor_completion"
                queued.generation += 1
            else:
                queued.state = "REPLAN_REQUIRED"
                if queued.wake_reason != (
                    "residency_transition_completed"
                ):
                    queued.wake_reason = "predecessor_completion"
                queued.generation += 1
        if self._policy.work_conserving_admission:
            # Early replans that only wait on running work take the freed
            # capacity now instead of sleeping until that work completes.
            for queued_request_id in early_replans:
                queued = self._entries[queued_request_id]
                if queued.state == _DEFERRED_REPLAN and all(
                    predecessor_id not in self._entries
                    or self._entries[predecessor_id].state
                        in {"ACTIVE", "FINISHING"}
                    for predecessor_id in queued.predecessor_request_ids
                ):
                    queued.state = "REPLAN_REQUIRED"
                    queued.generation += 1
                    self._count_policy_event("early_capacity_promotions")
        entry.capacity_replan_count = len(early_replans)
        entry.capacity_frontier_wake_count = len(frontier)
        self._history.append({
            "completed_at_us": completed_at_us,
            "early_replan_count": len(early_replans),
            "frontier_wake_count": len(frontier),
            "request_id": request_id,
            "route_id": route_id,
            "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
            "status": "CAPACITY_RELEASED",
        })
        self._promote_deferred_frontier(
            "predecessor_completion"
        )
        if enforce_work_conserving and not self._active:
            repair_is_runnable = any(
                other.state in {"REPLAN_REQUIRED", "REPLANNING"}
                for other in self._entries.values()
                if other is not entry
            )
            sleeping = tuple(
                other.decision.request_id
                for other in self._entries.values()
                if other.state == "QUEUED"
                and other.decision.start_us > completed_at_us
                and self._causal_ready(other)
                and not self._active_conflict(other)
                and not self._earlier_queued_conflict(other)
            )
            if sleeping and not repair_is_runnable:
                raise RuntimeQueueError(
                    "runnable queued work sleeps after capacity release:"
                    + request_id
                    + ":"
                    + ",".join(sleeping)
                )
        self._condition.notify_all()

    def release_capacity(
        self,
        request_id: str,
        completed_at_us: int,
        *,
        early_replan_request_ids: Sequence[str] | None = None,
    ) -> None:
        """Release physical lanes before terminal measurement completes."""

        request_id = _text("released dispatch request_id", request_id)
        completed_at_us = _nonnegative_int(
            "dispatch capacity release at_us", completed_at_us
        )
        early_replans = tuple(
            _text("early completion replan request_id", value)
            for value in (early_replan_request_ids or ())
        )
        if len(early_replans) != len(set(early_replans)):
            raise RuntimeQueueError(
                "early completion replans are duplicated"
            )
        with self._condition:
            self._release_capacity_locked(
                request_id,
                completed_at_us,
                early_replans,
                early_replan_request_ids is not None,
            )

    def complete(
        self,
        request_id: str,
        completed_at_us: int,
        *,
        early_replan_request_ids: Sequence[str] | None = None,
    ) -> None:
        request_id = _text("completed dispatch request_id", request_id)
        completed_at_us = _nonnegative_int(
            "dispatch completed_at_us", completed_at_us
        )
        early_replans = tuple(
            _text("early completion replan request_id", value)
            for value in (early_replan_request_ids or ())
        )
        if len(early_replans) != len(set(early_replans)):
            raise RuntimeQueueError(
                "early completion replans are duplicated"
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                raise RuntimeQueueError("completed request is absent")
            if entry.state == "ACTIVE":
                self._release_capacity_locked(
                    request_id,
                    completed_at_us,
                    early_replans,
                    early_replan_request_ids is not None,
                )
                entry = self._entries[request_id]
            elif early_replans:
                raise RuntimeQueueError(
                    "terminal completion repeats capacity replans"
                )
            if (
                entry.state != "FINISHING"
                or entry.capacity_released_at_us is None
                or completed_at_us < entry.capacity_released_at_us
            ):
                raise RuntimeQueueError(
                    "completed request has not released capacity"
                )
            del self._entries[request_id]
            self._residency_hysteresis_decisions.pop(request_id, None)
            self._history.append({
                "completed_at_us": completed_at_us,
                "early_replan_count": entry.capacity_replan_count,
                "frontier_wake_count": (
                    entry.capacity_frontier_wake_count
                ),
                "request_id": request_id,
                "route_id": entry.decision.route_id,
                "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
                "status": "COMPLETED",
            })
            self._condition.notify_all()

    def cancel_queued(
        self,
        request_id: str,
        cancelled_at_us: int,
        reason: str,
    ) -> None:
        request_id = _text("cancelled dispatch request_id", request_id)
        cancelled_at_us = _nonnegative_int(
            "dispatch cancelled_at_us", cancelled_at_us
        )
        reason = _text("dispatch cancellation reason", reason)
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state == "ACTIVE":
                raise RuntimeQueueError("cancelled request is not queued")
            del self._entries[request_id]
            self._residency_hysteresis_decisions.pop(request_id, None)
            for dependent in self._release_causal_frontier(request_id):
                dependent.state = "REPLAN_REQUIRED"
                dependent.wake_reason = "predecessor_cancelled"
                dependent.generation += 1
            self._history.append({
                "cancelled_at_us": cancelled_at_us,
                "reason": reason,
                "request_id": request_id,
                "route_id": entry.decision.route_id,
                "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
                "status": "CANCELLED",
            })
            self._promote_deferred_frontier(
                "predecessor_cancelled"
            )
            self._condition.notify_all()

    def rollback_acquire(
        self,
        request_id: str,
        receipt: RuntimeDispatchReceipt,
    ) -> None:
        """Undo an acquisition whose external transaction did not commit."""
        request_id = _text("rolled back dispatch request_id", request_id)
        if not isinstance(receipt, RuntimeDispatchReceipt):
            raise RuntimeQueueError("dispatch rollback receipt is invalid")
        with self._condition:
            entry = self._entries.get(request_id)
            if (
                entry is None
                or entry.state != "ACTIVE"
                or self._active.get(request_id) is not entry
                or receipt.request_id != request_id
                or receipt.route_id != entry.decision.route_id
                or receipt.status != "ACQUIRED"
                or not self._history
                or self._history[-1] != receipt.to_json()
            ):
                raise RuntimeQueueError(
                    "dispatch acquisition rollback state differs"
                )
            self._history.pop()
            del self._active[request_id]
            entry.state = "QUEUED"
            entry.wake_reason = (
                "calendar"
                if receipt.wake_reason.startswith("calendar_")
                else receipt.wake_reason
            )
            entry.generation += 1
            self._condition.notify_all()

    def require_replan(
        self,
        request_id: str,
        reason: str,
        *,
        defer_behind_predecessors: bool = False,
        allow_active_predecessors: bool = False,
    ) -> bool:
        request_id = _text("replanned dispatch request_id", request_id)
        reason = _text("dispatch replan reason", reason)
        if type(defer_behind_predecessors) is not bool:
            raise RuntimeQueueError(
                "dispatch replan deferral is invalid"
            )
        if type(allow_active_predecessors) is not bool:
            raise RuntimeQueueError(
                "dispatch active predecessor policy is invalid"
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None:
                return False
            if entry.state == "ACTIVE":
                raise RuntimeQueueError("cannot replan an active request")
            if entry.state == "REPLAN_REQUIRED":
                return False
            if (
                entry.state == _DEFERRED_REPLAN
                and not defer_behind_predecessors
            ):
                entry.state = "REPLAN_REQUIRED"
                entry.wake_reason = reason
                entry.generation += 1
                self._condition.notify_all()
                return True
            if entry.state == _DEFERRED_REPLAN:
                if (
                    reason == "capacity_released_early"
                    and entry.wake_reason != reason
                    and entry.wake_reason
                        not in {
                            "residency_projection_invalid",
                            "residency_transition_completed",
                        }
                ):
                    entry.wake_reason = reason
                    entry.generation += 1
                    self._condition.notify_all()
                    return True
                return False
            has_blocking_predecessor = any(
                predecessor_id in self._entries
                and not (
                    allow_active_predecessors
                    and self._entries[predecessor_id].state == "ACTIVE"
                )
                for predecessor_id in entry.predecessor_request_ids
            )
            entry.state = (
                _DEFERRED_REPLAN
                if defer_behind_predecessors
                and has_blocking_predecessor
                else "REPLAN_REQUIRED"
            )
            entry.wake_reason = reason
            entry.generation += 1
            self._condition.notify_all()
            return True

    def replan_required_requests(self) -> tuple[str, ...]:
        """Return replan-ready requests in their original queue order."""
        with self._condition:
            return tuple(
                entry.decision.request_id
                for entry in sorted(
                    (
                        row for row in self._entries.values()
                        if row.state == "REPLAN_REQUIRED"
                    ),
                    key=lambda row: row.sequence,
                )
            )

    def active_request_count(self) -> int:
        with self._condition:
            return len(self._active)

    def projection_request_ids(self) -> tuple[str, ...]:
        """Return attempts whose scheduled placement is still authoritative."""
        with self._condition:
            return tuple(
                entry.decision.request_id
                for entry in sorted(
                    (
                        row for row in self._entries.values()
                        if row.state in {"ACTIVE", "QUEUED"}
                    ),
                    key=self._dispatch_key,
                )
            )

    def dispatch_order_view(self) -> Mapping[str, Mapping[str, object]]:
        """Return each entry's queue state, barrier and direct predecessors."""
        with self._condition:
            return MappingProxyType({
                request_id: MappingProxyType({
                    "predecessor_request_ids": tuple(sorted(
                        entry.predecessor_request_ids
                    )),
                    "residency_transition_barrier": (
                        entry.residency_transition_barrier
                    ),
                    "sequence": entry.sequence,
                    "state": entry.state,
                })
                for request_id, entry in sorted(self._entries.items())
            })

    def projection_causal_predecessors(
        self,
    ) -> Mapping[str, tuple[str, ...]]:
        """Return dispatch-gating predecessors of authoritative attempts."""
        with self._condition:
            return MappingProxyType({
                request_id: tuple(sorted(entry.predecessor_request_ids))
                for request_id, entry in sorted(self._entries.items())
                if entry.state in {"ACTIVE", "QUEUED"}
                and entry.predecessor_request_ids
            })

    def defer_replan_behind(
        self,
        request_id: str,
        predecessor_request_id: str,
        reason: str,
    ) -> None:
        """Make an earlier conflicting queued request replan first."""
        request_id = _text("deferred replan request_id", request_id)
        predecessor_request_id = _text(
            "deferred replan predecessor request_id",
            predecessor_request_id,
        )
        reason = _text("deferred replan reason", reason)
        if request_id == predecessor_request_id:
            raise RuntimeQueueError("deferred replan predecessor is the owner")
        with self._condition:
            entry = self._entries.get(request_id)
            predecessor = self._entries.get(predecessor_request_id)
            if entry is None or entry.state != "REPLAN_REQUIRED":
                raise RuntimeQueueError("request does not await replanning")
            if predecessor is None or predecessor.state != "QUEUED":
                raise RuntimeQueueError(
                    "deferred replan predecessor is not queued"
                )
            if predecessor.sequence >= entry.sequence:
                raise RuntimeQueueError(
                    "deferred replan predecessor is not earlier"
                )
            if not self._entries_conflict(predecessor, entry):
                raise RuntimeQueueError(
                    "deferred replan predecessor does not conflict"
                )
            entry.state = _DEFERRED_REPLAN
            entry.wake_reason = reason
            entry.generation += 1
            predecessor.state = "REPLAN_REQUIRED"
            predecessor.wake_reason = reason
            predecessor.generation += 1
            if not self._causally_depends_on(
                request_id, predecessor_request_id
            ):
                self._add_causal_predecessor(
                    request_id, predecessor_request_id
                )
            self._condition.notify_all()

    def defer_replan_until_active_completion(
        self,
        request_id: str,
        predecessor_request_id: str,
        reason: str,
    ) -> None:
        """Keep a replan asleep until its active predecessor completes."""
        request_id = _text("deferred replan request_id", request_id)
        predecessor_request_id = _text(
            "active replan predecessor request_id",
            predecessor_request_id,
        )
        reason = _text("dispatch replan reason", reason)
        if request_id == predecessor_request_id:
            raise RuntimeQueueError("deferred replan predecessor is the owner")
        with self._condition:
            entry = self._entries.get(request_id)
            predecessor = self._entries.get(predecessor_request_id)
            if entry is None or entry.state != "REPLAN_REQUIRED":
                raise RuntimeQueueError("request does not await replanning")
            if (
                predecessor is None
                or predecessor.state != "ACTIVE"
                or self._active.get(predecessor_request_id) is not predecessor
            ):
                raise RuntimeQueueError(
                    "deferred replan predecessor is not active"
                )
            if not self._entries_conflict(predecessor, entry):
                raise RuntimeQueueError(
                    "deferred replan predecessor does not conflict"
                )
            entry.state = _DEFERRED_REPLAN
            entry.wake_reason = reason
            entry.generation += 1
            if not self._causally_depends_on(
                request_id, predecessor_request_id
            ):
                self._add_causal_predecessor(
                    request_id, predecessor_request_id
                )
            self._condition.notify_all()

    def dependent_queued_requests(
        self, request_id: str
    ) -> tuple[str, ...]:
        """Return the next causal owner on each replanned resource lane."""
        request_id = _text("dependent dispatch request_id", request_id)
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "REPLAN_REQUIRED":
                raise RuntimeQueueError(
                    "dependent dispatch owner does not await replanning"
                )
            return tuple(
                other.decision.request_id
                for other in self._follower_frontier(entry)
            )

    def queued_causal_dependents(
        self, request_id: str, before_us: int
    ) -> tuple[str, ...]:
        """Return queued attempts gated on one owner but reserved before it."""
        request_id = _text("causal dependent request_id", request_id)
        before_us = _nonnegative_int("causal dependent before_us", before_us)
        with self._condition:
            if request_id not in self._entries:
                raise RuntimeQueueError("causal dependent owner is absent")
            return tuple(
                other.decision.request_id
                for other in sorted(
                    self._entries.values(), key=self._dispatch_key
                )
                if other.state == "QUEUED"
                and other.decision.start_us < before_us
                and self._causally_depends_on(
                    other.decision.request_id, request_id
                )
            )

    def preceding_scheduled_requests(
        self,
        request_id: str,
        expected_generation: int | None = None,
    ) -> tuple[str, ...]:
        """Return stable work scheduled before one queued replan."""
        request_id = _text("preceding dispatch request_id", request_id)
        if expected_generation is not None:
            expected_generation = _nonnegative_int(
                "expected dispatch generation", expected_generation
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if (
                entry is None
                or entry.state != "REPLAN_REQUIRED"
                or (
                    expected_generation is not None
                    and entry.generation != expected_generation
                )
            ):
                raise RuntimeQueueError(
                    "runtime replan queue generation changed"
                )
            preceding = tuple(
                other
                for other_request_id, other in self._entries.items()
                if other.state in {"ACTIVE", "QUEUED"}
                and other_request_id in entry.predecessor_request_ids
            )
            return tuple(
                other.decision.request_id
                for other in sorted(
                    preceding, key=lambda row: row.sequence
                )
            )

    def retire_replan(
        self,
        request_id: str,
        expected_generation: int | None = None,
    ) -> None:
        request_id = _text("retired dispatch request_id", request_id)
        if expected_generation is not None:
            expected_generation = _nonnegative_int(
                "expected dispatch generation", expected_generation
            )
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state != "REPLAN_REQUIRED":
                raise RuntimeQueueError("request does not await replanning")
            if (
                expected_generation is not None
                and entry.generation != expected_generation
            ):
                raise RuntimeQueueError(
                    "runtime replan queue generation changed"
                )
            entry.state = "REPLANNING"
            entry.generation += 1
            self._condition.notify_all()

    def validate_replan_generation(
        self, request_id: str, expected_generation: int
    ) -> None:
        request_id = _text("validated dispatch request_id", request_id)
        expected_generation = _nonnegative_int(
            "expected dispatch generation", expected_generation
        )
        with self._condition:
            entry = self._entries.get(request_id)
            if (
                entry is None
                or entry.state != "REPLAN_REQUIRED"
                or entry.generation != expected_generation
            ):
                raise RuntimeQueueError(
                    "runtime replan queue generation changed"
                )

    def fail_replan(
        self,
        request_id: str,
        failed_at_us: int,
        reason: str,
    ) -> tuple[str, ...]:
        """Retire one failed replan and expose its ordered followers."""
        request_id = _text("failed replan request_id", request_id)
        failed_at_us = _nonnegative_int(
            "failed replan at_us", failed_at_us
        )
        reason = _text("failed replan reason", reason)
        with self._condition:
            entry = self._entries.get(request_id)
            if entry is None or entry.state not in {
                "REPLAN_REQUIRED", "REPLANNING"
            }:
                raise RuntimeQueueError(
                    "runtime request does not have a failed replan"
                )
            del self._entries[request_id]
            self._residency_hysteresis_decisions.pop(request_id, None)
            promoted = []
            for follower in self._release_causal_frontier(request_id):
                follower.state = "REPLAN_REQUIRED"
                follower.wake_reason = "predecessor_replan_failed"
                follower.generation += 1
                promoted.append(follower.decision.request_id)
            self._history.append({
                "failed_at_us": failed_at_us,
                "reason": reason,
                "request_id": request_id,
                "route_id": entry.decision.route_id,
                "schema": RUNTIME_DISPATCH_QUEUE_SCHEMA,
                "status": "FAILED",
            })
            self._promote_deferred_frontier(
                "predecessor_replan_failed"
            )
            self._condition.notify_all()
            return tuple(promoted)

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
        with self._condition:
            conflicts = []
            for request_id, entry in self._entries.items():
                if (
                    request_id == active_decision.request_id
                    or entry.state != "QUEUED"
                ):
                    continue
                conflicts_with_completed_owner = (
                    active_decision.request_id
                        in entry.predecessor_request_ids
                    or (
                        self._decisions_conflict(
                            active_decision, entry.decision
                        ) and any(
                            lease.start_us < extended_until_us
                            for lease in entry.decision.leases
                        )
                    )
                )
                remains_blocked = (
                    self._active_conflict(
                        entry,
                        excluding_request_id=active_decision.request_id,
                    )
                    or self._earlier_queued_conflict(
                        entry,
                        excluding_request_id=active_decision.request_id,
                    )
                )
                if conflicts_with_completed_owner and not remains_blocked:
                    conflicts.append(entry)
            return tuple(
                entry.decision.request_id
                for entry in sorted(conflicts, key=self._dispatch_key)
            )

    def commit_conflicting_replans(
        self,
        active_owner_id: str,
        active_leases: Sequence[LeaseRecord],
        extended_until_us: int,
        reason: str,
        commit: Callable[[tuple[str, ...]], _CommitResult],
        *,
        phase_windows: Mapping[str, tuple[int, int]] | None = None,
    ) -> tuple[tuple[str, ...], _CommitResult]:
        """Commit a lease transaction before exposing queue replans."""
        active_owner_id = _text(
            "active dispatch owner_id", active_owner_id
        )
        leases = tuple(active_leases)
        if not leases or any(not isinstance(row, LeaseRecord) for row in leases):
            raise RuntimeQueueError("active dispatch leases are invalid")
        extended_until_us = _nonnegative_int(
            "dispatch extension end", extended_until_us
        )
        reason = _text("dispatch replan reason", reason)
        if not callable(commit):
            raise RuntimeQueueError("dispatch lease commit is not callable")
        active_lanes = frozenset(
            (lease.resource_id, lane)
            for lease in leases
            for lane in lease.lanes
        )
        with self._condition:
            active_entry = self._entries.get(active_owner_id)
            conflicts = []
            for request_id, entry in self._entries.items():
                if (
                    request_id == active_owner_id
                    or entry.state in {"ACTIVE", "FINISHING"}
                ):
                    continue
                if (
                    active_entry is not None
                    and active_owner_id
                        not in entry.predecessor_request_ids
                    and not self._entries_conflict(active_entry, entry)
                ):
                    continue
                if phase_windows is None:
                    overlaps = any(
                        lease.start_us < extended_until_us
                        and any((lease.resource_id, lane) in active_lanes for lane in lease.lanes)
                        for lease in entry.decision.leases)
                else:
                    overlaps = any(
                        phase_windows[active.token][0] < phase_windows[active.token][1]
                        and lease.start_us < phase_windows[active.token][1]
                        and phase_windows[active.token][0] < lease.reserved_until_us
                        and lease.resource_id == active.resource_id
                        and set(lease.lanes).intersection(active.lanes)
                        for lease in entry.decision.leases for active in leases
                        if active.token in phase_windows)
                if overlaps:
                    conflicts.append(entry)
            request_ids = tuple(
                entry.decision.request_id
                for entry in sorted(conflicts, key=self._dispatch_key)
            )
            result = commit(request_ids)
            for request_id in request_ids:
                entry = self._entries[request_id]
                entry.state = _DEFERRED_REPLAN
                entry.wake_reason = reason
                entry.generation += 1
            return request_ids, result

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            policy = (
                {"dispatch_policy": self._policy.to_json()}
                if self._policy.enabled else {}
            )
            return {
                **policy,
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
                "entry_states": {
                    request_id: {
                        "generation": entry.generation,
                        "residency_transition_barrier": (
                            entry.residency_transition_barrier
                        ),
                        "state": entry.state,
                        "wake_reason": entry.wake_reason,
                    }
                    for request_id, entry in sorted(self._entries.items())
                },
                "wake_hold_count": len(self._wake_holds),
                "causal_predecessors": {
                    request_id: sorted(entry.predecessor_request_ids)
                    for request_id, entry in sorted(self._entries.items())
                    if entry.predecessor_request_ids
                },
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
    age_ns: int | None = None

    def to_json(self) -> dict[str, object]:
        validity = (
            "MISSING" if self.captured_at_ns == 0 else
            "TIMED_OUT" if self.error and self.error.startswith(("TimeoutError:", "TimeoutExpired:")) else
            "UNAVAILABLE" if self.error or self.value is None else
            "STALE" if self.stale else "VALID"
        )
        serialize = getattr(self.value, "to_json", None)
        return {
            "captured_at_ns": self.captured_at_ns,
            "error": self.error,
            "name": self.name,
            "stale": self.stale,
            "value": serialize() if callable(serialize) else copy.deepcopy(self.value),
            "source": self.name,
            "age_ns": self.age_ns,
            "validity": validity,
            "failure_reason": self.error or (
                "runtime monitor sample expired" if self.stale else
                "probe returned no observation" if self.value is None else None
            ),
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
        self._refresh = {name: threading.Event() for name in self._probes}
        self._stop = threading.Event()
        self._threads: dict[str, threading.Thread] = {}

    def start(self) -> None:
        with self._condition:
            if self._threads:
                raise RuntimeQueueError("runtime monitor is already started")
            for name, probe in self._probes.items():
                thread = threading.Thread(
                    target=self._run,
                    args=(name, probe),
                    name="unified-runtime-monitor-" + name,
                    daemon=True,
                )
                self._threads[name] = thread
                thread.start()

    def _run(self, name: str, probe: Callable[[], object]) -> None:
        refresh = self._refresh[name]
        while not self._stop.is_set():
            # Consume the wake that started this pass before probing. A refresh
            # requested during a slow probe must remain set for the next pass.
            refresh.clear()
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
            refresh.wait(self._refresh_interval_s)

    def request_refresh(self, name: str | None = None) -> None:
        names = tuple(self._refresh) if name is None else (name,)
        for probe_name in names:
            if probe_name not in self._refresh:
                raise RuntimeQueueError("runtime probe is absent")
            self._refresh[probe_name].set()

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
                age_ns=max(0, now_ns - snapshot.captured_at_ns),
            )

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self.request_refresh()
        deadline = time.monotonic() + timeout_s
        for thread in self._threads.values():
            thread.join(max(0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self._threads.values()):
            raise RuntimeQueueError("runtime monitor did not stop")

    @property
    def probe_names(self) -> Mapping[str, Callable[[], object]]:
        return MappingProxyType(dict(self._probes))
