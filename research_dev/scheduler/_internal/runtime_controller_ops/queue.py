"""RuntimeController queue operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from ..policy import Decision
from ..runtime_queue import RuntimeDispatchReceipt, RuntimeQueueError
from ..runtime_plan import RuntimeTransitionReceipt
from ..request_contracts.common import (
    RUNTIME_MEMORY_RESERVED_ATOMIC,
    RUNTIME_MEMORY_CANCELLED,
    RuntimeControllerError,
    _integer,
    RUNTIME_TERMINAL_STATES,
)
from ..request_contracts.ticket import RuntimeRequestTicket
from .dispatch import record_residency_hysteresis_hold


def defer_dispatch_wake(controller):
    """Delay physical queue wakeups through one scheduler transaction."""
    return controller.queue.defer_wake()


def hold_dispatch_wake(controller) -> int:
    """Hold physical queue wakeups through asynchronous publication."""
    try:
        return controller.queue.hold_wake()
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc


def release_dispatch_wake(controller, token: int) -> None:
    """Release one asynchronous publication wake hold."""
    try:
        controller.queue.release_wake(token)
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc


def require_queued_replan(
    controller,
    request_id: str,
    reason: str,
    *,
    defer_behind_predecessors: bool = False,
) -> bool:
    """Wake one queued owner without changing its immutable attempt."""
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "QUEUED":
            return False
        try:
            return controller.queue.require_replan(
                request_id,
                reason,
                defer_behind_predecessors=(
                    defer_behind_predecessors
                ),
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc


def invalidate_queued_attempts(
    controller,
    request_ids: Sequence[str],
    reason: str,
    at_us: int,
    *,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Cancel stale reservations while exposing only a causal frontier."""

    with controller._lock:
        replanned, _ = controller._mark_selected_followers(
            request_ids,
            reason,
            cancel_owner,
            release_memory,
            cancelled_at_us=at_us,
            defer_behind_predecessors=True,
            allow_active_predecessors=True,
        )
        return replanned


def replan_required_requests(controller) -> tuple[str, ...]:
    """Return replan-ready request IDs in stable queue order."""
    with controller._lock:
        return controller.queue.replan_required_requests()


def projection_request_ids(controller) -> tuple[str, ...]:
    """Return attempts whose queue placements remain authoritative."""
    with controller._lock:
        return controller.queue.projection_request_ids()


def projection_causal_predecessors(
    controller,
) -> Mapping[str, tuple[str, ...]]:
    """Return the queue edges that order projected dispatch."""
    with controller._lock:
        return controller.queue.projection_causal_predecessors()


def defer_replan_behind(
    controller,
    request_id: str,
    predecessor_request_id: str,
    predecessor_ticket_id: str,
    reason: str,
    at_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> None:
    """Order a stale queued predecessor before a conflicting replan."""
    with controller._lock:
        ticket = controller.ticket(request_id)
        predecessor = controller.ticket(predecessor_request_id)
        if ticket.dispatch_state != "REPLAN_REQUIRED":
            raise RuntimeControllerError(
                "runtime request does not need replan"
            )
        if predecessor.ticket_id != predecessor_ticket_id:
            raise RuntimeControllerError(
                "runtime replan predecessor attempt changed"
            )
        if predecessor.dispatch_state != "QUEUED":
            raise RuntimeControllerError(
                "runtime replan predecessor is not queued"
            )
        memory_status = ticket.memory_reservation_status
        if ticket.lease_status != "CANCELLED":
            cancel_owner(request_id, at_us)
        if (
            release_memory is not None
            and memory_status == "RESERVED_ATOMIC"
        ):
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "deferred replan memory release differs"
                )
            memory_status = "CANCELLED"
        try:
            controller.queue.defer_replan_behind(
                request_id,
                predecessor_request_id,
                reason,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        controller._tickets[request_id] = replace(
            ticket,
            lease_status="CANCELLED",
            memory_reservation_status=memory_status,
        )


def defer_replan_until_active_completion(
    controller,
    request_id: str,
    predecessor_request_id: str,
    predecessor_ticket_id: str,
    reason: str,
    at_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> None:
    """Keep a stale successor behind an acquired physical attempt."""
    with controller._lock:
        ticket = controller.ticket(request_id)
        predecessor = controller.ticket(predecessor_request_id)
        if ticket.dispatch_state != "REPLAN_REQUIRED":
            raise RuntimeControllerError(
                "runtime request does not need replan"
            )
        if predecessor.ticket_id != predecessor_ticket_id:
            raise RuntimeControllerError(
                "runtime replan predecessor attempt changed"
            )
        if predecessor.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError(
                "runtime replan predecessor is not acquired"
            )
        memory_status = ticket.memory_reservation_status
        if ticket.lease_status != "CANCELLED":
            cancel_owner(request_id, at_us)
        if (
            release_memory is not None
            and memory_status == "RESERVED_ATOMIC"
        ):
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "deferred replan memory release differs"
                )
            memory_status = "CANCELLED"
        try:
            controller.queue.defer_replan_until_active_completion(
                request_id,
                predecessor_request_id,
                reason,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        controller._tickets[request_id] = replace(
            ticket,
            lease_status="CANCELLED",
            memory_reservation_status=memory_status,
        )


def wait_ready(
    controller,
    request_id: str,
    epoch_ns: int,
) -> RuntimeDispatchReceipt:
    """Wait for a queue event without committing scheduler-owned state."""
    try:
        receipt = controller.queue.wait_ready(request_id, epoch_ns)
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc
    if controller.dispatch_policy.residency_hysteresis_s:
        record_residency_hysteresis_hold(controller, request_id)
    return receipt


def commit_wait(
    controller,
    receipt: RuntimeDispatchReceipt,
    commit_acquired: Callable[[RuntimeRequestTicket], None] | None = None,
) -> RuntimeRequestTicket | None:
    """Atomically bind one still-current queue event to its ticket."""
    if not isinstance(receipt, RuntimeDispatchReceipt):
        raise RuntimeControllerError("runtime dispatch receipt is invalid")
    if commit_acquired is not None and not callable(commit_acquired):
        raise RuntimeControllerError(
            "runtime acquisition commit callback is invalid"
        )
    with controller._lock:
        request_id = receipt.request_id
        committed = False
        queue_checkpoint = None
        try:
            ticket = controller.ticket(request_id)
            if ticket.dispatch_state in RUNTIME_TERMINAL_STATES:
                return ticket
            if receipt.status == "ACQUIRED":
                queue_checkpoint = controller.queue.checkpoint()
                current_receipt = controller.queue.commit_ready(receipt)
                if current_receipt is None:
                    return None
                receipt = current_receipt
                committed = receipt.status == "ACQUIRED"
            elif receipt.status == "REPLAN_REQUIRED":
                try:
                    controller.queue.validate_replan_generation(
                        request_id, receipt.queue_generation
                    )
                except RuntimeQueueError:
                    return None
            else:
                raise RuntimeControllerError(
                    "runtime dispatch receipt status is invalid"
                )
            updated = replace(
                ticket,
                dispatch_state=receipt.status,
                dispatch_receipt=receipt,
            )
            if (
                commit_acquired is not None
                and receipt.status == "ACQUIRED"
            ):
                commit_acquired(updated)
            controller._tickets[request_id] = updated
            if receipt.status == "ACQUIRED":
                controller._acquired_tickets[updated.ticket_id] = updated
            return updated
        except BaseException:
            if committed:
                try:
                    controller.queue.restore(queue_checkpoint)
                except RuntimeQueueError as rollback_error:
                    raise RuntimeControllerError(
                        "runtime acquisition rollback failed"
                    ) from rollback_error
            raise


def wait(
    controller,
    request_id: str,
    epoch_ns: int,
    commit_acquired: Callable[[RuntimeRequestTicket], None] | None = None,
) -> RuntimeRequestTicket:
    """Wait for dispatch when no outer scheduler transaction is required."""
    while True:
        try:
            receipt = controller.wait_ready(request_id, epoch_ns)
        except RuntimeControllerError:
            with controller._lock:
                terminal = controller._tickets.get(request_id)
                if (
                    terminal is not None
                    and terminal.dispatch_state in RUNTIME_TERMINAL_STATES
                ):
                    return terminal
            raise
        result = controller.commit_wait(receipt, commit_acquired)
        if result is not None:
            return result


def record_transition_receipts(
    controller,
    request_id: str,
    receipts: Sequence[RuntimeTransitionReceipt],
) -> RuntimeRequestTicket:
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError(
                "runtime transition resources are not acquired"
            )
        if ticket.transition_status != "PENDING":
            raise RuntimeControllerError(
                "runtime transition state cannot be rewritten"
            )
        rows = tuple(receipts)
        previous = {row.transition_id: row for row in ticket.transition_receipts}
        if any(row not in rows for row in previous.values()):
            raise RuntimeControllerError("runtime transition receipt prefix changed")
        status = (
            "FAILED"
            if any(row.status == "FAILED" for row in rows)
            else ("COMPLETED" if len(rows) == len(ticket.execution_plan.transitions) else "PENDING")
        )
        updated = replace(
            ticket,
            transition_status=status,
            transition_receipts=rows,
        )
        controller._tickets[request_id] = updated
        return updated


def _mark_followers(
    controller,
    decision: Decision,
    extended_until_us: int,
    reason: str,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
    try:
        request_ids = controller.queue.conflicting_queued_requests(
            decision, extended_until_us
        )
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc
    return controller._mark_selected_followers(
        request_ids, reason, cancel_owner, release_memory
    )


def _mark_selected_followers(
    controller,
    request_ids: Sequence[str],
    reason: str,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    memory_owner_tokens: Callable[
        [str], tuple[str, ...]
    ] | None = None,
    *,
    cancelled_at_us: int = 0,
    defer_behind_predecessors: bool = True,
    allow_active_predecessors: bool = False,
) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
    cancelled_at_us = _integer(
        "runtime follower cancellation at_us", cancelled_at_us
    )
    if type(defer_behind_predecessors) is not bool:
        raise RuntimeControllerError(
            "runtime follower replan deferral is invalid"
        )
    if type(allow_active_predecessors) is not bool:
        raise RuntimeControllerError(
            "runtime active predecessor policy is invalid"
        )
    cancelled: dict[str, tuple[str, ...]] = {}
    replanned = []
    for request_id in request_ids:
        ticket = controller.ticket(request_id)
        if (
            reason == "capacity_released_early"
            and ticket.failure_reason == reason
            and cancelled_at_us < ticket.decision.start_us
            and controller.queue.active_request_count() > 1
        ):
            continue
        try:
            changed = controller.queue.require_replan(
                request_id,
                reason,
                defer_behind_predecessors=defer_behind_predecessors,
                allow_active_predecessors=allow_active_predecessors,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        if not changed:
            continue
        cancelled[request_id] = (
            ()
            if ticket.lease_status == "CANCELLED"
            else cancel_owner(request_id, cancelled_at_us)
        )
        if memory_owner_tokens is None:
            memory_status = ticket.memory_reservation_status
            if (
                release_memory is not None
                and memory_status == RUNTIME_MEMORY_RESERVED_ATOMIC
            ):
                released_memory = release_memory(request_id)
                if set(released_memory) != {
                    row.token for row in ticket.memory_reservations
                }:
                    raise RuntimeControllerError(
                        "runtime follower memory release differs"
                    )
                memory_status = RUNTIME_MEMORY_CANCELLED
        else:
            memory_status = controller._detach_compaction_follower_memory(
                ticket,
                release_memory,
                memory_owner_tokens,
            )
        controller._tickets[request_id] = replace(
            ticket,
            lease_status="CANCELLED",
            memory_reservation_status=memory_status,
        )
        replanned.append(request_id)
    return tuple(replanned), MappingProxyType(cancelled)
