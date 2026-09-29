"""RuntimeController replan operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from ..runtime_queue import RuntimeQueueError
from ..request_contracts.common import (
    RuntimeControllerError,
    RuntimeReplanRetryRequired,
    _text,
    _integer,
)
from ..request_contracts.ticket import RuntimeRequestTicket


def prepare_replan(
    controller,
    request_id: str,
    at_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    *,
    invalidate_dependents: bool | None = None,
    expected_ticket_id: str | None = None,
    expected_queue_generation: int | None = None,
) -> RuntimeRequestTicket:
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "REPLAN_REQUIRED":
            raise RuntimeControllerError(
                "runtime request does not need replan"
            )
        if (
            expected_ticket_id is not None
            and ticket.ticket_id != _text(
                "runtime expected ticket id", expected_ticket_id
            )
        ):
            raise RuntimeReplanRetryRequired(
                request_id,
                ticket.ticket_id,
                "EXPECTED_TICKET_CHANGED",
            )
        if expected_queue_generation is not None:
            expected_queue_generation = _integer(
                "runtime expected queue generation",
                expected_queue_generation,
            )
            try:
                controller.queue.validate_replan_generation(
                    request_id, expected_queue_generation
                )
            except RuntimeQueueError as exc:
                raise RuntimeReplanRetryRequired(
                    request_id,
                    ticket.ticket_id,
                    "QUEUE_GENERATION_CHANGED",
                ) from exc
        if invalidate_dependents is not None and type(
            invalidate_dependents
        ) is not bool:
            raise RuntimeControllerError(
                "runtime dependent invalidation is invalid"
            )
        dependent_ids = ()
        invalidate = (
            ticket.dispatch_receipt is not None
            and ticket.dispatch_receipt.wake_reason
                != "predecessor_completion"
            and (
                ticket.execution_plan is None
                or ticket.execution_plan.transitions
            )
            if invalidate_dependents is None
            else invalidate_dependents
        )
        if invalidate:
            try:
                dependent_ids = controller.queue.dependent_queued_requests(
                    request_id
                )
            except RuntimeQueueError as exc:
                raise RuntimeControllerError(str(exc)) from exc
        _defer_dependents(
            controller, dependent_ids, cancel_owner, release_memory
        )
        if ticket.lease_status != "CANCELLED":
            cancel_owner(request_id, at_us)
        if (
            release_memory is not None
            and ticket.memory_reservation_status == "RESERVED_ATOMIC"
        ):
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "runtime replan memory release differs from ticket"
                )
        try:
            controller.queue.retire_replan(
                request_id, expected_queue_generation
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        return ticket


def _defer_dependents(
    controller,
    dependent_ids: tuple[str, ...],
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None,
) -> None:
    for dependent_id in dependent_ids:
        dependent = controller.ticket(dependent_id)
        try:
            changed = controller.queue.require_replan(
                dependent_id,
                "predecessor_replan",
                defer_behind_predecessors=True,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        if not changed:
            raise RuntimeControllerError(
                "dependent runtime request was not queued"
            )
        if dependent.lease_status != "CANCELLED":
            cancel_owner(dependent_id, 0)
        memory_status = dependent.memory_reservation_status
        if (
            release_memory is not None
            and memory_status == "RESERVED_ATOMIC"
        ):
            released_memory = release_memory(dependent_id)
            if set(released_memory) != {
                row.token for row in dependent.memory_reservations
            }:
                raise RuntimeControllerError(
                    "dependent runtime replan memory release differs"
                )
            memory_status = "CANCELLED"
        controller._tickets[dependent_id] = replace(
            dependent,
            lease_status="CANCELLED",
            memory_reservation_status=memory_status,
        )


def defer_causal_dependents_before(
    controller,
    request_id: str,
    before_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Defer queued dependents whose reservations precede their predecessor."""
    with controller._lock:
        before_us = _integer("runtime dependent deferral time", before_us)
        try:
            dependent_ids = controller.queue.queued_causal_dependents(
                request_id, before_us
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        _defer_dependents(
            controller, dependent_ids, cancel_owner, release_memory
        )
        return dependent_ids


def queued_causal_dependents(
    controller,
    request_id: str,
    before_us: int,
) -> tuple[str, ...]:
    with controller._lock:
        before_us = _integer("runtime dependent deferral time", before_us)
        try:
            return controller.queue.queued_causal_dependents(
                request_id, before_us
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc


def replan_projection_request_ids(
    controller,
    request_id: str,
    expected_queue_generation: int | None = None,
) -> tuple[str, ...]:
    with controller._lock:
        ticket = controller.ticket(request_id)
        try:
            return controller.queue.preceding_scheduled_requests(
                request_id, expected_queue_generation
            )
        except RuntimeQueueError as exc:
            if str(exc) == "runtime replan queue generation changed":
                raise RuntimeReplanRetryRequired(
                    request_id,
                    ticket.ticket_id,
                    "REPLAN_WAKE_ROLLED_BACK",
                ) from exc
            raise RuntimeControllerError(str(exc)) from exc


def fail_replan(
    controller,
    request_id: str,
    failed_at_us: int,
    reason: str,
    *,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[RuntimeRequestTicket, tuple[str, ...]]:
    """Terminalize a failed queued replan without stranding followers."""
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "REPLAN_REQUIRED":
            raise RuntimeControllerError(
                "runtime request does not have a failed replan"
            )
        reason = _text("runtime replan failure reason", reason)
        if ticket.lease_status != "CANCELLED":
            cancel_owner(request_id, failed_at_us)
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
                    "runtime failed replan memory release differs"
                )
            memory_status = "CANCELLED"
        try:
            promoted = controller.queue.fail_replan(
                request_id, failed_at_us, reason
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        failed = replace(
            ticket,
            dispatch_state="FAILED",
            dispatch_receipt=None,
            lease_status="CANCELLED",
            memory_reservation_status=memory_status,
            failure_reason=reason,
            actual_end_us=failed_at_us,
        )
        controller._tickets[request_id] = failed
        controller._terminal_tickets[failed.ticket_id] = failed
        return failed, promoted
