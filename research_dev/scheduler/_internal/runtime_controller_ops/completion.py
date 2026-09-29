"""RuntimeController completion operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from ..runtime_queue import RuntimeQueueError
from ..runtime_plan import RuntimeExecutionReceipt
from ..request_contracts.common import (
    RuntimeControllerError,
    _text,
    _integer,
    RUNTIME_TERMINAL_STATES,
)
from ..request_contracts.ticket import _validate_execution_receipt, RuntimeRequestTicket
from ..request_contracts.receipts import RuntimeCompletionReceipt


# Failure reasons written by adapters.runtime for the elastic-phone failure
# kinds (``_internal.runtime_execution.ELASTIC_FAILURE_PHASES``; not imported
# here because runtime_execution imports the runtime controller). The route is
# healthy: the lost helper DEVICE is quarantined by the scheduler and an
# exited server is reloaded by the next plan, so the route stays admissible.
_PHYSICAL_BACKEND_REASON_PREFIX = "physical_backend_failed:"
_ROUTE_RETAINED_ACTIONS = {
    "helper_lost": "route_retained_helper_device_lost",
    "server_exited": "route_retained_executor_reload",
}


def _retained_route_kind(reason: str) -> str:
    if not reason.startswith(_PHYSICAL_BACKEND_REASON_PREFIX):
        return "physical"
    phase = reason[len(_PHYSICAL_BACKEND_REASON_PREFIX):].split(":", 1)[0]
    return phase if phase in _ROUTE_RETAINED_ACTIONS else "physical"


def _route_violation(
    controller,
    ticket: RuntimeRequestTicket,
    actual_end_us: int,
    kind: str,
) -> str:
    selected = next(
        row for row in ticket.cost_estimates.estimates
        if row.route_id == ticket.decision.route_id
    )
    state = controller._route_uncertainty.setdefault(
        ticket.decision.route_id,
        {"physical_failures": 0, "upper_bound_violations": 0},
    )
    key = (
        "upper_bound_violations"
        if kind == "prediction"
        else "physical_failures"
    )
    state[key] = int(state[key]) + 1
    state["last_actual_end_us"] = actual_end_us
    state["last_finish_upper_us"] = ticket.decision.finish_upper_us
    if kind in _ROUTE_RETAINED_ACTIONS:
        return _ROUTE_RETAINED_ACTIONS[kind]
    if selected.baseline:
        return "baseline_requires_recalibration"
    controller._quarantined_routes.add(ticket.decision.route_id)
    return "disabled_for_remaining_run"


def complete(
    controller,
    request_id: str,
    actual_end_us: int,
    *,
    resource_capacities: Mapping[str, int] | None = None,
    release_lease: Callable[[str, int], None],
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    execution_receipt: RuntimeExecutionReceipt | None = None,
) -> RuntimeCompletionReceipt:
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError("runtime request is not active")
        if ticket.transition_status not in {
            "COMPLETED", "NOT_REQUIRED"
        }:
            raise RuntimeControllerError(
                "runtime transitions are not complete"
            )
        if ticket.execution_plan is not None:
            if execution_receipt is None:
                raise RuntimeControllerError(
                    "automated runtime completion requires physical proof"
                )
            _validate_execution_receipt(
                ticket, execution_receipt, actual_end_us
            )
        elif execution_receipt is not None:
            raise RuntimeControllerError(
                "legacy runtime completion has unexpected physical proof"
            )
        staged = controller._capacity_releases.get(request_id)
        if staged is None:
            staged = controller.release_capacity(
                request_id,
                actual_end_us,
                resource_capacities=resource_capacities,
                release_lease=release_lease,
                cancel_owner=cancel_owner,
                release_memory=release_memory,
            )
            ticket = controller.ticket(request_id)
        elif staged.actual_end_us != actual_end_us:
            raise RuntimeControllerError(
                "runtime terminal finish differs from capacity release"
            )
        try:
            controller.queue.complete(request_id, actual_end_us)
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        quarantine_action = None
        if not staged.latency_upper_bound.met:
            quarantine_action = controller._route_violation(
                ticket, actual_end_us, "prediction"
            )
        updated = replace(
            ticket,
            dispatch_state="COMPLETED",
            dispatch_receipt=None,
            prediction_status=staged.latency_upper_bound.status,
            lease_status=staged.lease_coverage.status,
            actual_end_us=actual_end_us,
            execution_receipt=execution_receipt,
        )
        controller._tickets[request_id] = updated
        controller._terminal_tickets[updated.ticket_id] = updated
        del controller._capacity_releases[request_id]
        return replace(
            staged,
            quarantine_action=quarantine_action,
            execution_receipt=execution_receipt,
        )


def fail(
    controller,
    request_id: str,
    failed_at_us: int,
    reason: str,
    *,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    failed_resource_ids: Sequence[str] = (),
) -> tuple[RuntimeRequestTicket, tuple[str, ...], str | None]:
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError("runtime request is not active")
        failed_at_us = _integer("runtime failure at_us", failed_at_us)
        reason = _text("runtime failure reason", reason)
        failed_resources = tuple(sorted(
            _text("runtime failed resource", value)
            for value in failed_resource_ids
        ))
        if (
            len(failed_resources) != len(set(failed_resources))
            or not set(failed_resources).issubset(
                ticket.binding.resource_ids
            )
        ):
            raise RuntimeControllerError(
                "runtime failed resources differ from the ticket"
            )
        staged = controller._capacity_releases.get(request_id)
        cancelled = (
            () if staged is not None
            else cancel_owner(request_id, failed_at_us)
        )
        if release_memory is not None and staged is None:
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "runtime memory cancellation differs from ticket"
                )
        if staged is None:
            try:
                followers = controller.queue.causal_follower_frontier(
                    request_id
                )
            except RuntimeQueueError as exc:
                raise RuntimeControllerError(str(exc)) from exc
            controller._mark_selected_followers(
                followers,
                "predecessor_execution_failed",
                cancel_owner,
                release_memory,
            )
        try:
            controller.queue.complete(
                request_id,
                max(
                    failed_at_us,
                    0 if staged is None else staged.actual_end_us,
                ),
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        action = controller._route_violation(
            ticket, failed_at_us, _retained_route_kind(reason)
        )
        if failed_resources:
            controller._quarantined_resources.update(failed_resources)
            action = "disabled_failed_resources_for_remaining_run"
        failed = replace(
            ticket,
            dispatch_state="FAILED",
            dispatch_receipt=None,
            lease_status="CANCELLED",
            memory_reservation_status=(
                "CANCELLED"
                if ticket.memory_reservations
                else ticket.memory_reservation_status
            ),
            failure_reason=reason,
            actual_end_us=failed_at_us,
        )
        controller._tickets[request_id] = failed
        controller._terminal_tickets[failed.ticket_id] = failed
        controller._capacity_releases.pop(request_id, None)
        return failed, cancelled, action


def cancel(
    controller,
    request_id: str,
    at_us: int,
    reason: str,
    *,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    terminal_is_noop: bool = False,
) -> tuple[str, ...]:
    with controller._lock:
        ticket = controller.ticket(request_id)
        at_us = _integer("runtime cancellation at_us", at_us)
        reason = _text("runtime cancellation reason", reason)
        if ticket.dispatch_state in RUNTIME_TERMINAL_STATES:
            if not terminal_is_noop:
                raise RuntimeControllerError(
                    "runtime terminal ticket cannot be rewritten"
                )
            controller._cancellation_audit.append({
                "at_us": at_us,
                "reason": reason,
                "request_id": request_id,
                "result": "ALREADY_TERMINAL",
                "terminal_state": ticket.dispatch_state,
                "ticket_id": ticket.ticket_id,
            })
            return ()
        staged = controller._capacity_releases.get(request_id)
        cancelled = (
            ()
            if ticket.lease_status in {
                "CANCELLED", "RELEASED_PENDING_RECEIPT"
            }
            else cancel_owner(request_id, at_us)
        )
        memory_status = ticket.memory_reservation_status
        if (
            release_memory is not None
            and memory_status == "RESERVED_ATOMIC"
        ):
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "runtime memory cancellation differs from ticket"
                )
            memory_status = "CANCELLED"
        try:
            if ticket.dispatch_state == "ACQUIRED":
                controller.queue.complete(
                    request_id,
                    max(
                        at_us,
                        0 if staged is None else staged.actual_end_us,
                    ),
                )
            elif ticket.dispatch_state in {"QUEUED", "REPLAN_REQUIRED"}:
                controller.queue.cancel_queued(request_id, at_us, reason)
            else:
                raise RuntimeControllerError(
                    "runtime request has invalid cancellation state"
                )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        updated = replace(
            ticket,
            dispatch_state="CANCELLED",
            dispatch_receipt=None,
            lease_status="CANCELLED",
            memory_reservation_status=(
                "CANCELLED"
                if staged is not None and ticket.memory_reservations
                else memory_status
            ),
            failure_reason=reason,
            actual_end_us=at_us,
        )
        controller._tickets[request_id] = updated
        controller._terminal_tickets[updated.ticket_id] = updated
        controller._capacity_releases.pop(request_id, None)
        return cancelled
