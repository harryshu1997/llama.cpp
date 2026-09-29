"""Continuous join at publication: replan the resident model's queued work once its server is live.

A request that arrives while its model's server is loading cannot join it: the
join rules need the model resident with a live executor and an ACQUIRED
same-model co-tenant. It is queued, and when another model's residency change
was queued earlier it is ordered behind that change. Nothing replans it when
the server goes live, so it waits for the switch, a reload and the other
model's work (hardware s1c: Qwen 003/004 arrived during 001's load, queued
behind the Gemma switch 002, and waited 737 s although 001 decoded with free
slots).

Under ``dispatch_policy.continuous_join`` such a request is replanned (wake
reason ``continuous_join_server_live``, statistic
``continuous_join_publication_replans``) at the first point where a
displacement is possible: when the server's publication is observed
(``observe_automated_runtime_snapshot``) and whenever another model's residency
change is committed afterwards (the change may still be awaiting its replan
when the server is published, with a stale plan that changes nothing). Its
replan then goes through model affinity and the continuous-join bypass, with
their fairness bounds. A request is eligible only while its attempt was
planned before its server was published, so each attempt is replanned at most
once by this rule.
"""

from __future__ import annotations

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.runtime_dispatch_policy import plan_changes_residency
from .affinity import model_affinity_replan_displacement
from .continuous_join import continuous_join_replan_bypass

SERVER_LIVE_REASON = "continuous_join_server_live"
PUBLICATION_REPLAN_STAT = "continuous_join_publication_replans"
_WAITING = frozenset({"QUEUED", "DEFERRED_REPLAN"})


def server_published_at_us(joiner, tickets) -> int | None:
    """When the joiner's server went live: the latest finished preparation of a co-tenant.

    Co-tenants are ACQUIRED attempts of the joiner's model on its executor
    whose transitions completed; None without one (nothing was published for
    the joiner to join).
    """
    finished = [
        max(row.finished_us for row in ticket.transition_receipts)
        for ticket in tickets
        if ticket.dispatch_state == "ACQUIRED"
        and ticket.request.request_id != joiner.request.request_id
        and ticket.model.artifact_sha256 == joiner.model.artifact_sha256
        and ticket.binding.executor_id == joiner.binding.executor_id
        and ticket.transition_status == "COMPLETED"
        and ticket.transition_receipts
    ]
    return max(finished, default=None)


def planned_before_publication(joiner, tickets) -> bool:
    """Whether the joiner's attempt was decided before its server was published.

    The attempt's observation time is the scheduling time of its decision.
    """
    published_us = server_published_at_us(joiner, tickets)
    return (
        published_us is not None
        and joiner.runtime_observation.captured_at_us < published_us
    )


def _displacement_possible(controller, snapshot, joiner, observed_at_us: int) -> bool:
    """Whether model affinity or the continuous-join bypass may reorder the joiner."""
    if model_affinity_replan_displacement(
        controller,
        snapshot=snapshot,
        current=joiner,
        observed_at_us=observed_at_us,
        reason=SERVER_LIVE_REASON,
        record_refusal=False,
    ) is not None:
        return True
    return continuous_join_replan_bypass(
        controller,
        snapshot=snapshot,
        current=joiner,
        observed_at_us=observed_at_us,
        reason=SERVER_LIVE_REASON,
        record_refusal=False,
    ) is not None


def server_live_joiners(
    controller, snapshot: HeterogeneousRuntimeSnapshot, observed_at_us: int
) -> tuple[str, ...]:
    """Not-started attempts to replan now because their server is live, in queue order.

    Each is QUEUED or deferred, was planned before its server was published
    (``planned_before_publication``), and a displacement of the other models'
    queued residency changes it waits on is possible now.
    """
    runtime = controller._runtime_controller
    if not runtime.dispatch_policy.continuous_join:
        return ()
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError("continuous join publication snapshot is invalid")
    if type(observed_at_us) is not int or observed_at_us < 0:
        raise UnifiedScheduleError("continuous join publication time is invalid")
    view = runtime.dispatch_order_view()
    tickets = runtime.current_tickets()
    by_request = {row.request.request_id: row for row in tickets}
    return tuple(
        request_id
        for request_id in sorted(view, key=lambda key: view[key]["sequence"])
        if view[request_id]["state"] in _WAITING
        and request_id in by_request
        and by_request[request_id].execution_plan is not None
        and planned_before_publication(by_request[request_id], tickets)
        and _displacement_possible(
            controller, snapshot, by_request[request_id], observed_at_us
        )
    )


def wake_server_live_joiners(
    controller, snapshot: HeterogeneousRuntimeSnapshot, observed_at_us: int
) -> tuple[str, ...]:
    """Mark the ``server_live_joiners`` for replan now; count and return them.

    The caller holds a scheduler transaction.
    """
    joiners = server_live_joiners(controller, snapshot, observed_at_us)
    if not joiners:
        return ()
    runtime = controller._runtime_controller
    woken = runtime.replan_queued_now(
        joiners,
        SERVER_LIVE_REASON,
        observed_at_us,
        cancel_owner=controller.cancel,
        release_memory=controller._runtime_memory.release_owner,
    )
    if woken:
        runtime.record_dispatch_policy_event(PUBLICATION_REPLAN_STAT, len(woken))
    return woken


def commits_residency_change(controller, ticket) -> bool:
    """Whether a replan just reserved a residency change that joiners may now displace.

    Only under ``continuous_join``: the ticket is QUEUED and its plan prepares
    an exclusive residency device.
    """
    return bool(
        controller._runtime_controller.dispatch_policy.continuous_join
        and ticket.dispatch_state == "QUEUED"
        and ticket.execution_plan is not None
        and plan_changes_residency(
            ticket.execution_plan.transitions,
            controller._runtime_exclusive_memory_resources(),
        )
    )
