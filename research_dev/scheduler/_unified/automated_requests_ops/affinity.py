"""Model-affinity displacement for arrivals of the resident model."""

from __future__ import annotations

from typing import Mapping

from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from .common import _AutomatedSubmitResolution


AFFINITY_DISPLACEMENT_REASON = "model_affinity_displaced"
_NOT_STARTED = frozenset({"QUEUED", "DEFERRED_REPLAN", "REPLAN_REQUIRED"})


class _NoAffinityGain(Exception):
    """Roll a displacement back when it would not start the arrival earlier."""


def no_gain_reason(
    resolution: _AutomatedSubmitResolution,
    original: _AutomatedSubmitResolution,
    exclusive_by_device: Mapping[object, str],
    prefix: str,
    bound: str | None = None,
) -> str | None:
    """Why a re-resolved attempt gains nothing on ``original``; None when it does.

    The attempt gains when it is reserved earlier and its plan prepares no
    exclusive residency device; the reason names the failing half with both
    starts or the residency transitions. ``bound`` (what holds the re-resolved
    start, see ``continuous_join.bound_detail``) is appended to the start half.
    """
    return start_refusal(
        prefix, resolution.preview.start_us, original.preview.start_us, bound
    ) or residency_refusal(
        prefix, resolution.selected.plan.transitions, exclusive_by_device
    )


def start_refusal(
    prefix: str, start_us: int, original_start_us: int, bound: str | None = None
) -> str | None:
    """The start half of ``no_gain_reason``: None when ``start_us`` is earlier."""
    if start_us < original_start_us:
        return None
    reason = f"{prefix}: re-resolved start {start_us} >= original {original_start_us}"
    return reason if bound is None else f"{reason} ({bound})"


def residency_refusal(
    prefix: str, transitions, exclusive_by_device: Mapping[object, str]
) -> str | None:
    """The residency half of ``no_gain_reason``: the transitions on exclusive devices."""
    if not plan_changes_residency(transitions, exclusive_by_device):
        return None
    return f"{prefix}: re-resolved plan changes residency: " + ", ".join(
        f"{row.transition_id} ({row.source_state}->{row.target_state})"
        for row in transitions
        if any(
            device_id in exclusive_by_device
            for device_id in row.prepares_device_ids
        )
    )


def _model_is_resident(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot,
    artifact_sha256: str,
    exclusive_by_device: Mapping[object, str],
) -> bool:
    executor_by_device = controller._runtime_capabilities.executor_by_device
    for row in snapshot.residency:
        executor_id = row.executor_id
        if executor_id is None and row.device_id in executor_by_device:
            executor_id = executor_by_device[row.device_id].executor_id
        executor = (
            None if executor_id is None else snapshot.executors.get(executor_id)
        )
        if (
            row.artifact_sha256 == artifact_sha256
            and row.state == "hot"
            and row.resident_bytes > 0
            and (
                row.device_id in exclusive_by_device
                or (executor_id, row.device_id) in exclusive_by_device
            )
            and executor is not None
            and executor.ready
        ):
            return True
    return False


def model_affinity_displacement(
    controller,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    artifact_sha256: str,
    resolution: _AutomatedSubmitResolution,
    observed_at_us: int,
) -> tuple[tuple[str, ...], dict[str, object]] | None:
    """Return the queued work an arrival of the resident model may precede.

    The arrival qualifies when its model is resident with a live executor
    and a residency change of another model would run first: a queued change
    reserved at or before the arrival's start, or a cancelled change (awaiting
    its replan) that its running predecessors do not hold back past the
    arrival's reservation. The displaced set is those changes plus every
    attempt queued behind them. It is refused when any displaced attempt of
    another model is protected by the fairness bound, holds a decode cohort
    or a phone layout transition, or when another model's change is running
    or being replanned.
    """
    runtime = controller._runtime_controller
    policy = runtime.dispatch_policy
    if not policy.model_affinity:
        return None
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    if not _model_is_resident(
        controller, snapshot, artifact_sha256, exclusive_by_device
    ):
        return None
    preview = resolution.preview
    exclusive_resources = frozenset(exclusive_by_device.values())
    arrival_resources = exclusive_resources.intersection(
        row.resource_id for row in preview.plans
    ) or exclusive_resources.intersection(resolution.selected.plan.resource_ids)
    if not arrival_resources:
        return None
    arrival_end_us = max(
        (row.reserved_until_us for row in preview.plans),
        default=preview.finish_us,
    )
    view = runtime.dispatch_order_view()
    tickets = {
        row.request.request_id: row for row in runtime.current_tickets()
    }

    def other_model_change(request_id: str) -> bool:
        ticket = tickets.get(request_id)
        return bool(
            ticket is not None
            and ticket.execution_plan is not None
            and ticket.model.artifact_sha256 != artifact_sha256
            and plan_changes_residency(
                ticket.execution_plan.transitions, exclusive_by_device
            )
        )

    def runs_first(request_id: str) -> bool:
        row = view[request_id]
        if row["state"] == "QUEUED":
            return any(
                lease.resource_id in arrival_resources
                and lease.start_us <= preview.start_us
                for lease in tickets[request_id].decision.leases
            )
        running_ends = [
            lease.reserved_until_us
            for predecessor_id in row["predecessor_request_ids"]
            if view.get(predecessor_id, {}).get("state") == "ACTIVE"
            and predecessor_id in tickets
            for lease in tickets[predecessor_id].decision.leases
        ]
        return not running_ends or arrival_end_us > max(running_ends)

    changes = tuple(sorted(
        request_id for request_id in view if other_model_change(request_id)
    ))
    if any(
        view[request_id]["state"] not in _NOT_STARTED for request_id in changes
    ):
        return None
    blocking = tuple(
        request_id for request_id in changes if runs_first(request_id)
    )
    if not blocking:
        return None
    closed = _close_displacement(
        runtime, view, tickets, artifact_sha256, blocking, observed_at_us
    )
    if closed is None:
        return None
    displaced, bypassed = closed
    return displaced, {
        "blocking_request_ids": list(blocking),
        "bypassed_request_ids": list(bypassed),
        "observed_at_us": observed_at_us,
        "reserved_start_without_displacement_us": preview.start_us,
    }


def model_affinity_replan_displacement(
    controller,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    current,
    observed_at_us: int,
    reason: str,
    record_refusal: bool = True,
) -> tuple[tuple[str, ...], dict[str, object]] | None:
    """Return the queued work a replan of the resident model may precede.

    A replan while the request's model is resident with a live executor is
    treated like an arrival: a request that arrived during its model's load
    was ordered behind the other models' residency changes by arrival, and
    keeps that order after the load is published. The not-started residency
    changes of other models it still waits on, and every attempt queued
    behind them, are displaced under the same bounds and refusals. The
    caller keeps the displacement only when the replanned attempt needs no
    residency transition.
    """
    runtime = controller._runtime_controller
    policy = runtime.dispatch_policy
    if not policy.model_affinity:
        return None
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    artifact_sha256 = current.model.artifact_sha256
    if not _model_is_resident(
        controller, snapshot, artifact_sha256, exclusive_by_device
    ):
        return None
    request_id = current.request.request_id
    view = runtime.dispatch_order_view()
    if request_id not in view:
        return None
    tickets = {
        row.request.request_id: row for row in runtime.current_tickets()
    }
    waits_on = waited_on(view, request_id)
    changes = tuple(sorted(
        other_id for other_id in view
        if other_id in tickets
        and tickets[other_id].execution_plan is not None
        and tickets[other_id].model.artifact_sha256 != artifact_sha256
        and plan_changes_residency(
            tickets[other_id].execution_plan.transitions, exclusive_by_device
        )
    ))
    if any(
        view[other_id]["state"] not in _NOT_STARTED for other_id in changes
    ):
        return None
    blocking = tuple(other_id for other_id in changes if other_id in waits_on)
    if not blocking:
        return None
    closed = _close_displacement(
        runtime, view, tickets, artifact_sha256, blocking, observed_at_us,
        owner_request_id=request_id, record_refusal=record_refusal,
    )
    if closed is None:
        return None
    displaced, bypassed = closed
    return displaced, {
        "blocking_request_ids": list(blocking),
        "bypassed_request_ids": list(bypassed),
        "observed_at_us": observed_at_us,
        "replan_reason": reason,
    }


def waited_on(view, request_id: str) -> frozenset[str]:
    """Every queue entry ``request_id`` waits on, through its predecessors."""
    waits_on: set[str] = set()
    pending = list(view[request_id]["predecessor_request_ids"])
    while pending:
        predecessor_id = pending.pop()
        if predecessor_id in waits_on or predecessor_id not in view:
            continue
        waits_on.add(predecessor_id)
        pending.extend(view[predecessor_id]["predecessor_request_ids"])
    return frozenset(waits_on)


def _close_displacement(
    runtime,
    view,
    tickets,
    artifact_sha256: str,
    blocking: tuple[str, ...],
    observed_at_us: int,
    *,
    owner_request_id: str | None = None,
    record_refusal: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Add the work queued behind ``blocking``; apply the fairness bounds."""
    policy = runtime.dispatch_policy
    displaced = set(blocking)
    changed = True
    while changed:
        changed = False
        for request_id, row in view.items():
            if (
                request_id not in displaced
                and request_id != owner_request_id
                and row["state"] not in {"ACTIVE", "FINISHING"}
                and displaced.intersection(row["predecessor_request_ids"])
            ):
                displaced.add(request_id)
                changed = True
    bypassed = tuple(sorted(
        request_id for request_id in displaced
        if request_id in tickets
        and tickets[request_id].model.artifact_sha256 != artifact_sha256
    ))
    protected = tuple(
        request_id for request_id in bypassed
        if runtime.dispatch_bypass_count(request_id)
            >= policy.affinity_maximum_bypasses
        or observed_at_us - tickets[request_id].request.arrival_us
            >= policy.affinity_maximum_wait_us
    )
    unsupported = any(
        request_id not in tickets
        or view[request_id]["state"] == "REPLANNING"
        or tickets[request_id].decode_cohort is not None
        or tickets[request_id].residency_projection_token is not None
        for request_id in displaced
    )
    if protected or unsupported:
        if record_refusal:
            runtime.record_dispatch_policy_event("affinity_refusals")
        return None
    return tuple(sorted(displaced)), bypassed
