"""Continuous-join barrier bypass: a same-model joiner precedes a queued residency change.

Under work-conserving admission a request of the resident model runs ahead of a
queued residency change of another model only when its reserved lanes are free
again before that change can start. A joiner whose decode outlasts the running
requests of its model is reserved behind the change instead, although the server
keeps decoding its model either way. With ``dispatch_policy.continuous_join`` the
joiner may precede the change when its predicted finish extends the server's
committed busy window (the latest lease horizon of the ACQUIRED requests of its
model on the exclusive resources) by at most ``max_barrier_extension_s``; 0 never
extends the window. The change and the work queued behind it replan after the
joiner, as under model affinity, and the decision carries the
``CONTINUOUS_JOIN_BARRIER_BYPASS`` note.

The bypass re-resolution is judged from the live tickets only: the caller's
causal not-before barriers are kept only where a ticket outside the displaced
set still justifies them (``live_not_before_after_displacement``), and the
epoch pre-projection previews the epoch's desktop-parent template rather than
its assisted one (``selection._epoch_preprojection_template``). A refusal on
the start names what holds the re-resolved start (``bound_detail``).

A replan of a not-started attempt of the resident model gets the same bounded
bypass (``continuous_join_replan_bypass``): the attempt waits on queued
residency changes of other models through its queue predecessors (it arrived
while its server was loading, see ``join_publication``) and is kept ahead of
them only when its re-resolved start precedes theirs, its plan changes no
residency and it extends the committed busy window by at most the bound.
"""

from __future__ import annotations

from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.runtime_controller_ops.dispatch import CONTINUOUS_JOIN_BYPASS_KIND
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from .affinity import (
    _model_is_resident,
    residency_refusal,
    start_refusal,
    waited_on,
)
from .common import _AutomatedSubmitContext, _AutomatedSubmitResolution

BARRIER_DISPLACEMENT_REASON = "continuous_join_displaced"
_NOT_STARTED = frozenset({"QUEUED", "DEFERRED_REPLAN", "REPLAN_REQUIRED"})
_TERMINAL = frozenset({"CANCELLED", "COMPLETED", "FAILED"})


class _NoJoinGain(Exception):
    """Roll a bypass back: the joiner would not start earlier or would exceed the bound."""


def committed_busy_end_us(ticket, resource_ids: frozenset[str]) -> int:
    """The busy window end one ACQUIRED ticket commits the server to.

    The later of its admission prediction and its live lease horizon on the
    given resources, which renewals move while the request decodes.
    """
    horizon = max(
        (
            ticket.final_reserved_until_us.get(lease.token, lease.reserved_until_us)
            for lease in ticket.live_leases
            if lease.resource_id in resource_ids
        ),
        default=ticket.decision.finish_upper_us,
    )
    return max(ticket.decision.finish_upper_us, horizon)


def live_not_before_after_displacement(
    controller,
    causal_not_before_by_resource: Mapping[str, int],
    displaced: tuple[str, ...],
) -> dict[str, int]:
    """The caller's causal barriers a live ticket still justifies once ``displaced`` is cancelled.

    A causal barrier is the lease horizon of a ticket on an exclusive resource
    (a stale-projection repair). The cancel truncates the displaced change's
    leases, so a barrier only they justified would still reserve the joiner
    behind the change; an entry is kept while a ticket outside the displaced
    set, neither cancelled nor terminal, holds a lease on its resource that
    reaches it.
    """
    live_tickets = tuple(
        ticket
        for ticket in controller._runtime_controller.current_tickets()
        if ticket.request.request_id not in displaced
        and ticket.lease_status != "CANCELLED"
        and ticket.dispatch_state not in _TERMINAL
    )
    kept: dict[str, int] = {}
    for resource_id, not_before_us in causal_not_before_by_resource.items():
        if type(resource_id) is not str or type(not_before_us) is not int:
            raise UnifiedScheduleError("causal not-before barrier is invalid")
        if any(
            lease.resource_id == resource_id
            and ticket.final_reserved_until_us.get(
                lease.token, lease.reserved_until_us
            ) >= not_before_us
            for ticket in live_tickets
            for lease in ticket.decision.leases
        ):
            kept[resource_id] = not_before_us
    return kept


def bound_detail(
    context: _AutomatedSubmitContext, resolution: _AutomatedSubmitResolution
) -> str:
    """Name what holds a re-resolved start: barriers, transitions, then the calendar.

    The not-before barriers of the selected plan's resources at or past the
    start, the plan's transitions (a load waits for every lease on the device),
    and the calendar resources that pushed the preview.
    """
    start_us = resolution.preview.start_us
    plan = resolution.selected.plan
    resource_ids = frozenset(plan.resource_ids)
    parts = [
        f"not_before[{resource_id}]={not_before_us}"
        for resource_id, not_before_us in sorted(
            context.live_not_before_by_resource.items()
        )
        if resource_id in resource_ids and not_before_us >= start_us
    ]
    parts.extend(f"transition[{row.transition_id}]" for row in plan.transitions)
    parts.extend(
        f"calendar[{resource_id}]"
        for resource_id in resolution.preview.blocking_resources
    )
    if not parts:
        return "no barrier recorded"
    return "bounded by " + ", ".join(parts)


def _displaced_closure(
    view, blocking: tuple[str, ...], owner_request_id: str | None = None
) -> tuple[str, ...]:
    """``blocking`` plus every not-started attempt queued behind it but the owner."""
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
    return tuple(sorted(displaced))


def continuous_join_barrier_bypass(
    controller,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    artifact_sha256: str,
    resolution: _AutomatedSubmitResolution,
    observed_at_us: int,
) -> tuple[tuple[str, ...], dict[str, object]] | None:
    """Return the queued work a same-model joiner may precede, with its note.

    The arrival qualifies when its model is resident with a live executor,
    requests of its model are ACQUIRED on the exclusive residency resources it
    uses, and a not-started residency change of another model would run first:
    a queued change reserved at or before the arrival's start, or a cancelled
    change (awaiting its replan) that its running predecessors do not hold
    back past the arrival's reservation. Reserved behind the change, the
    arrival's own plan may reload its model; the caller keeps the bypass only
    when the plan reserved ahead of the change needs no residency transition
    and stays within the extension bound. The displaced set is those changes
    plus every attempt queued behind them. Refused (counted) when another
    model's change is running or being replanned, or a displaced attempt holds
    a decode cohort or a phone layout transition.
    """
    runtime = controller._runtime_controller
    if not runtime.dispatch_policy.continuous_join:
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
    co_tenants = tuple(
        ticket for ticket in runtime.current_tickets(("ACQUIRED",))
        if ticket.model.artifact_sha256 == artifact_sha256
        and any(
            lease.resource_id in arrival_resources for lease in ticket.live_leases
        )
    )
    if not co_tenants:
        return None
    committed_end_us = max(
        committed_busy_end_us(ticket, arrival_resources) for ticket in co_tenants
    )
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
    blocking = tuple(sorted(
        (request_id for request_id in changes if runs_first(request_id)),
        key=lambda request_id: view[request_id]["sequence"],
    ))
    if not blocking:
        return None
    displaced = _displaced_closure(view, blocking)
    if any(
        request_id not in tickets
        or view[request_id]["state"] == "REPLANNING"
        or tickets[request_id].decode_cohort is not None
        or tickets[request_id].residency_projection_token is not None
        for request_id in displaced
    ):
        runtime.record_dispatch_policy_event("continuous_join_refusals")
        return None
    return displaced, {
        "kind": CONTINUOUS_JOIN_BYPASS_KIND,
        "barrier_request_id": blocking[0],
        "barrier_request_ids": list(blocking),
        "bypassed_request_ids": [
            request_id for request_id in displaced
            if tickets[request_id].model.artifact_sha256 != artifact_sha256
        ],
        "committed_end_us": committed_end_us,
        "observed_at_us": observed_at_us,
        "reserved_start_without_bypass_us": preview.start_us,
    }


REPLAN_BYPASS_PREFIX = "bypass does not start the joiner before the change"


def _co_tenant_window(runtime, artifact_sha256: str, resources: frozenset[str]):
    """The ACQUIRED requests of the model on ``resources`` and their committed window end."""
    co_tenants = tuple(
        ticket for ticket in runtime.current_tickets(("ACQUIRED",))
        if ticket.model.artifact_sha256 == artifact_sha256
        and any(lease.resource_id in resources for lease in ticket.live_leases)
    )
    if not co_tenants:
        return (), None
    return co_tenants, max(
        committed_busy_end_us(ticket, resources) for ticket in co_tenants
    )


def _waited_changes(current, exclusive_by_device, view, tickets):
    """The other models' residency changes ``current`` waits on, in queue order.

    None when another model's change already runs, or when one it waits on is
    not reserved (a cancelled change is judged once its replan reserved it).
    """
    artifact_sha256 = current.model.artifact_sha256
    changes = tuple(
        request_id for request_id, ticket in tickets.items()
        if request_id in view
        and ticket.execution_plan is not None
        and ticket.model.artifact_sha256 != artifact_sha256
        and plan_changes_residency(
            ticket.execution_plan.transitions, exclusive_by_device
        )
    )
    if any(view[request_id]["state"] not in _NOT_STARTED for request_id in changes):
        return None
    waits_on = waited_on(view, current.request.request_id)
    blocking = tuple(sorted(
        (request_id for request_id in changes if request_id in waits_on),
        key=lambda request_id: view[request_id]["sequence"],
    ))
    if any(view[request_id]["state"] != "QUEUED" for request_id in blocking):
        return None
    return blocking


def _lease_span(
    tickets, request_ids, resources: frozenset[str]
) -> tuple[int, int] | None:
    """(earliest start, latest end) of the attempts' leases on ``resources``.

    None when one of them holds no lease there (not a change of this server).
    """
    spans = []
    for request_id in request_ids:
        leases = tuple(
            lease for lease in tickets[request_id].decision.leases
            if lease.resource_id in resources
        )
        if not leases:
            return None
        spans.extend(leases)
    return (
        min(lease.start_us for lease in spans),
        max(lease.reserved_until_us for lease in spans),
    )


def continuous_join_replan_bypass(
    controller,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    current,
    observed_at_us: int,
    reason: str,
    record_refusal: bool = True,
) -> tuple[tuple[str, ...], dict[str, object]] | None:
    """Return the queued residency changes a replan of the resident model may precede.

    The replan twin of ``continuous_join_barrier_bypass``: the attempt's model
    is resident with a live executor, requests of its model are ACQUIRED on the
    exclusive residency resources its plan uses, and it waits, through its
    queue predecessors, on queued residency changes of other models (none of
    which runs). The displaced set is those changes plus every not-started
    attempt queued behind them, except the attempt itself. Refused (counted
    when ``record_refusal``) when a displaced attempt is being replanned or
    holds a decode cohort or a phone layout transition. The note's
    ``barrier_start_us`` is the changes' earliest reserved start on the server
    (the re-resolved start must precede it) and
    ``reserved_start_without_bypass_us`` their latest reserved end there.
    """
    runtime = controller._runtime_controller
    if not runtime.dispatch_policy.continuous_join or current.execution_plan is None:
        return None
    if type(observed_at_us) is not int or type(reason) is not str:
        raise UnifiedScheduleError("continuous join replan bypass input is invalid")
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    artifact_sha256 = current.model.artifact_sha256
    if not _model_is_resident(controller, snapshot, artifact_sha256, exclusive_by_device):
        return None
    resources = frozenset(exclusive_by_device.values()).intersection(
        current.execution_plan.resource_ids
    )
    co_tenants, committed_end_us = _co_tenant_window(runtime, artifact_sha256, resources)
    view = runtime.dispatch_order_view()
    request_id = current.request.request_id
    if not co_tenants or request_id not in view:
        return None
    tickets = {row.request.request_id: row for row in runtime.current_tickets()}
    blocking = _waited_changes(current, exclusive_by_device, view, tickets)
    span = None if not blocking else _lease_span(tickets, blocking, resources)
    if span is None:
        return None
    displaced = _displaced_closure(view, blocking, request_id)
    if any(
        other_id not in tickets
        or view[other_id]["state"] == "REPLANNING"
        or tickets[other_id].decode_cohort is not None
        or tickets[other_id].residency_projection_token is not None
        for other_id in displaced
    ):
        if record_refusal:
            runtime.record_dispatch_policy_event("continuous_join_refusals")
        return None
    barrier_start_us, barrier_end_us = span
    return displaced, {
        "kind": CONTINUOUS_JOIN_BYPASS_KIND,
        "barrier_request_id": blocking[0],
        "barrier_request_ids": list(blocking),
        "barrier_start_us": barrier_start_us,
        "bypassed_request_ids": [
            other_id for other_id in displaced
            if tickets[other_id].model.artifact_sha256 != artifact_sha256
        ],
        "committed_end_us": committed_end_us,
        "observed_at_us": observed_at_us,
        "replan_reason": reason,
        "reserved_start_without_bypass_us": barrier_end_us,
    }


def replan_bypass_refusal(
    selected, preview, note: Mapping[str, object], exclusive_by_device, bound_us: int
) -> tuple[str | None, int]:
    """(Why a replanned joiner must not precede the change, or None; its extension).

    The re-resolved start must precede the change's reserved start, the plan
    must change no residency, and the predicted finish may extend the
    committed busy window by at most ``bound_us``.
    """
    extension_us = max(0, preview.finish_upper_us - note["committed_end_us"])
    refusal = start_refusal(
        REPLAN_BYPASS_PREFIX, preview.start_us, note["barrier_start_us"]
    ) or residency_refusal(
        REPLAN_BYPASS_PREFIX, selected.plan.transitions, exclusive_by_device
    )
    if refusal is None and extension_us > bound_us:
        refusal = "joiner would extend the committed busy window"
    return refusal, extension_us
