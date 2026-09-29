"""Continuous join: a same-model arrival joins its running server as a desktop parent.

Phone-assisted routes hold capacity-one resources (the phone's HTP, its transport,
the USB root). While a running request of the same model holds them on a server
with a free slot, an arrival of that model selecting an assisted route is reserved
behind the holder although the server could decode both at once (a decode step
costs the same at batch 4 as at batch 1; a phone FFN call with 2-3 rows costs
3-13 % more). Under ``dispatch_policy.continuous_join`` such rows are rejected with
``PHONE_LANES_HELD_BY_RUNNING_REQUEST`` and the desktop parent is selected with
``CONTINUOUS_JOIN_DESKTOP_PARENT``; phone assistance reaches it later through the
shared helper window under server policy coherence. The ticket's physical batch
contract stays authoritative.

The held lanes are matched by lease owner: a co-tenant admitted into a decode
cohort holds its lanes under the cohort id, not its request id. While the
runtime controller marks a barrier-bypass re-resolution for the arrival, rows
whose plan prepares an exclusive residency device (a load on a free phone, a
reload of the server) are also rejected with
``CONTINUOUS_JOIN_RESIDENCY_TRANSITION``: the bypass never admits them, so the
joiner is planned as the desktop parent of its running server instead.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.runtime_decode_cohort import RuntimeDecodeCohortManager
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from ..._internal.runtime_plan import AutomatedRouteCandidate
from .common import _AutomatedSelectionContext

JOIN_REASON = "CONTINUOUS_JOIN_DESKTOP_PARENT"
PHONE_LANES_HELD = "PHONE_LANES_HELD_BY_RUNNING_REQUEST"
RESIDENCY_TRANSITION = "CONTINUOUS_JOIN_RESIDENCY_TRANSITION"


def running_co_tenants(
    controller, context: _AutomatedSelectionContext
) -> tuple[RuntimeRequestTicket, ...]:
    """ACQUIRED requests of the baseline's model on the baseline's server endpoint."""
    baseline = context.baseline
    endpoint = baseline.binding.endpoint
    if endpoint is None:
        return ()
    return tuple(
        ticket
        for ticket in controller._runtime_controller.current_tickets(("ACQUIRED",))
        if ticket.request.request_id != context.request.request_id
        and ticket.model.artifact_sha256 == baseline.binding.artifact_sha256
        and ticket.binding.endpoint == endpoint
    )


def join_slots(
    baseline: AutomatedRouteCandidate, row: AutomatedRouteCandidate
) -> int:
    """Requests the server (``parallel``) and the row's decode cohort can co-decode."""
    parallel = baseline.plan.adapter_parameters.get("parallel")
    if type(parallel) is not int or parallel < 2:
        return 1
    return min(parallel, RuntimeDecodeCohortManager.capacity(row.plan))


def lease_holders(co_tenants: Iterable[RuntimeRequestTicket]) -> frozenset[str]:
    """The lease owner ids of the co-tenants: their request ids and cohort ids."""
    return frozenset(
        owner_id
        for ticket in co_tenants
        for owner_id in (
            ticket.request.request_id,
            *(lease.owner_id for lease in ticket.live_leases),
        )
    )


def _holds_capacity_one_resource(
    resources: Mapping[str, Mapping[str, object]],
    baseline: AutomatedRouteCandidate,
    row: AutomatedRouteCandidate,
    holders: frozenset[str],
) -> bool:
    return any(
        state["capacity"] == 1
        and state["free_slots"] == 0
        and holders.intersection(state["active_owners"])
        for resource_id in row.plan.resource_ids
        if resource_id not in baseline.plan.resource_ids
        for state in (resources.get(resource_id),)
        if state is not None
    )


def _bypass_transition_rejections(
    controller,
    context: _AutomatedSelectionContext,
    candidates: tuple[AutomatedRouteCandidate, ...],
) -> dict[str, str]:
    """Rows a barrier bypass never admits: their plan prepares an exclusive device."""
    runtime = controller._runtime_controller
    if (
        runtime.continuous_join_resolution_request_id()
        != context.request.request_id
    ):
        return {}
    exclusive_by_device = controller._runtime_exclusive_memory_resources()
    return {
        row.candidate_id: RESIDENCY_TRANSITION
        for row in candidates
        if row.candidate_id != context.baseline.candidate_id
        and row.candidate_id not in context.excluded
        and plan_changes_residency(row.plan.transitions, exclusive_by_device)
    }


def _held_lane_rejections(
    controller,
    context: _AutomatedSelectionContext,
    candidates: tuple[AutomatedRouteCandidate, ...],
    rejections: dict[str, str],
) -> None:
    """Add the rows whose capacity-one lanes a running co-tenant holds."""
    baseline = context.baseline
    if (
        not context.baseline_available
        or baseline.plan.execution_contract.execution_mode != "desktop"
    ):
        return
    co_tenants = running_co_tenants(controller, context)
    if not co_tenants:
        return
    executor = context.snapshot.executors.get(baseline.binding.executor_id)
    if executor is None or executor.free_slots < 1:
        return
    resources = controller.timeline.resource_snapshot(context.observed_at_us)
    holders = lease_holders(co_tenants)
    for row in candidates:
        if (
            row.candidate_id == baseline.candidate_id
            or row.candidate_id in context.excluded
            or row.candidate_id in rejections
            or len(co_tenants) >= join_slots(baseline, row)
        ):
            continue
        if _holds_capacity_one_resource(resources, baseline, row, holders):
            rejections[row.candidate_id] = PHONE_LANES_HELD


def continuous_join_rejections(
    controller,
    context: _AutomatedSelectionContext,
    rows: Iterable[AutomatedRouteCandidate] | None = None,
) -> Mapping[str, str]:
    """Rows a joiner must not take: a running co-tenant holds their capacity-one lanes.

    Empty unless ``continuous_join`` is on and the selection knows its snapshot.
    Held lanes are rejected when the desktop baseline is available, a co-tenant
    of the same model is ACQUIRED on the baseline's endpoint, the live executor
    reports a free slot, and the co-tenants are fewer than ``join_slots`` for
    the row. During the arrival's barrier-bypass re-resolution, rows whose plan
    prepares an exclusive residency device are rejected as well.
    """
    policy = controller._runtime_controller.dispatch_policy
    if (
        not policy.continuous_join
        or context.snapshot is None
        or context.observed_at_us is None
    ):
        return {}
    candidates = tuple(
        context.candidate_set.candidates if rows is None else rows
    )
    rejections = _bypass_transition_rejections(controller, context, candidates)
    _held_lane_rejections(controller, context, candidates, rejections)
    return rejections


def continuous_join_reason(
    rejections: Mapping[str, str],
    selected: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
    reason: str,
) -> str:
    """Name the join when held lanes left the desktop parent as the selection."""
    if rejections and selected.candidate_id == baseline.candidate_id:
        return JOIN_REASON
    return reason
