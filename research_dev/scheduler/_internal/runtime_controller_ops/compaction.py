"""RuntimeController compaction operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from ..policy import Decision
from ..runtime_queue import RuntimeQueueError
from ..request_contracts.common import (
    RUNTIME_MEMORY_RESERVED_ATOMIC,
    RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT,
    RUNTIME_MEMORY_CANCELLED,
    RuntimeControllerError,
    _integer,
)
from ..request_contracts.ticket import RuntimeRequestTicket


def _detach_compaction_follower_memory(
    ticket: RuntimeRequestTicket,
    release_memory: Callable[[str], tuple[str, ...]] | None,
    memory_owner_tokens: Callable[[str], tuple[str, ...]],
) -> str:
    """Detach or validate one follower's memory ownership exactly once."""

    request_id = ticket.request.request_id
    recorded_tokens = tuple(sorted(
        row.token for row in ticket.memory_reservations
    ))
    live_tokens = tuple(sorted(memory_owner_tokens(request_id)))
    status = ticket.memory_reservation_status
    if status == RUNTIME_MEMORY_RESERVED_ATOMIC:
        if not recorded_tokens or live_tokens != recorded_tokens:
            raise RuntimeControllerError(
                "priority compaction follower memory ownership differs"
            )
        if release_memory is None:
            raise RuntimeControllerError(
                "priority compaction follower memory is live"
            )
        released_tokens = tuple(sorted(release_memory(request_id)))
        if released_tokens != recorded_tokens:
            raise RuntimeControllerError(
                "priority compaction follower memory release differs"
            )
        if memory_owner_tokens(request_id):
            raise RuntimeControllerError(
                "priority compaction follower memory remains owned"
            )
        return RUNTIME_MEMORY_CANCELLED
    if status == RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT:
        if recorded_tokens or live_tokens:
            raise RuntimeControllerError(
                "priority compaction resident memory ownership differs"
            )
        return status
    if status == RUNTIME_MEMORY_CANCELLED:
        if live_tokens:
            raise RuntimeControllerError(
                "priority compaction cancelled memory remains owned"
            )
        return status
    raise RuntimeControllerError(
        "priority compaction follower memory state differs"
    )


def release_replanned_capacity_frontier(
    controller,
    request_id: str,
    previous_decision: Decision,
    at_us: int,
    *,
    resource_capacities: Mapping[str, int] | None = None,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    memory_owner_tokens: Callable[
        [str], tuple[str, ...]
    ] | None = None,
) -> tuple[str, ...]:
    """Cancel and expose only the next owners of newly freed lanes."""

    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "QUEUED":
            raise RuntimeControllerError(
                "replanned capacity owner is not queued"
            )
        at_us = _integer("replanned capacity at_us", at_us)
        try:
            request_ids = controller.queue.replanned_capacity_frontier(
                request_id,
                previous_decision,
                resource_capacities,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        replanned, _ = controller._mark_selected_followers(
            request_ids,
            "capacity_released_early",
            cancel_owner,
            release_memory,
            memory_owner_tokens,
            cancelled_at_us=at_us,
            defer_behind_predecessors=False,
        )
        return replanned


def _mark_residency_transition_frontier(
    controller,
    ticket: RuntimeRequestTicket,
    completed_at_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    plan = ticket.execution_plan
    if plan is None or not plan.transitions:
        return ()
    prepared_device_ids = {
        device_id
        for transition in plan.transitions
        for device_id in transition.prepares_device_ids
    }
    if not prepared_device_ids:
        return ()
    candidates = tuple(
        request_id
        for request_id, candidate in controller._tickets.items()
        if candidate.dispatch_state == "QUEUED"
        and candidate.execution_plan is not None
        and (
            any(
                prepared_device_ids.intersection(
                    transition.prepares_device_ids
                )
                for transition in candidate.execution_plan.transitions
            )
            or (
                candidate.model.artifact_sha256
                    != ticket.model.artifact_sha256
                and any(
                    demand.kind == "model_weights"
                    and demand.resident_bytes > 0
                    and demand.demand_id == "weights:" + device_id
                    for device_id in prepared_device_ids
                    for demand in candidate.execution_plan.memory_demands
                )
            )
        )
    )
    if not candidates:
        return ()
    try:
        affected = controller.queue.causal_follower_frontier(
            ticket.request.request_id,
            candidates,
        )
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc
    invalidated, _ = controller._mark_selected_followers(
        affected,
        "residency_transition_completed",
        cancel_owner,
        release_memory,
        cancelled_at_us=completed_at_us,
    )
    return invalidated


def prepare_priority_compaction_followers(
    controller,
    request_id: str,
    at_us: int,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    memory_owner_tokens: Callable[
        [str], tuple[str, ...]
    ] | None = None,
    *,
    expected_queue_generation: int | None = None,
) -> tuple[tuple[str, str], ...]:
    """Detach later conflicting attempts without exposing waiters."""

    with controller._lock:
        if memory_owner_tokens is None:
            raise RuntimeControllerError(
                "priority compaction memory ownership is unavailable"
            )
        try:
            request_ids = controller.queue.priority_compaction_followers(
                request_id, expected_queue_generation
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        identities = tuple(
            (follower_id, controller.ticket(follower_id).ticket_id)
            for follower_id in request_ids
        )
        replanned, _ = controller._mark_selected_followers(
            request_ids,
            "priority_compaction_follower",
            cancel_owner,
            release_memory,
            memory_owner_tokens,
            cancelled_at_us=at_us,
            defer_behind_predecessors=True,
        )
        replanned_ids = frozenset(replanned)
        for follower_id in request_ids:
            if follower_id in replanned_ids:
                continue
            follower = controller.ticket(follower_id)
            memory_status = controller._detach_compaction_follower_memory(
                follower,
                release_memory,
                memory_owner_tokens,
            )
            if memory_status != follower.memory_reservation_status:
                follower = replace(
                    follower,
                    memory_reservation_status=memory_status,
                )
                controller._tickets[follower_id] = follower
            if (
                follower.lease_status != "CANCELLED"
            ):
                raise RuntimeControllerError(
                    "priority compaction follower set changed"
                )
        return identities


def priority_compaction_followers(
    controller,
    request_id: str,
    expected_queue_generation: int | None = None,
) -> tuple[str, ...]:
    """Inspect later live reservations without changing them."""

    try:
        return controller.queue.priority_compaction_followers(
            request_id, expected_queue_generation
        )
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc


def promote_priority_compaction_follower(
    controller,
    request_id: str,
    expected_ticket_id: str,
    observed_at_us: int,
    cancel_owner: Callable[
        [str, int], tuple[str, ...]
    ] | None = None,
    release_memory: Callable[
        [str], tuple[str, ...]
    ] | None = None,
    memory_owner_tokens: Callable[
        [str], tuple[str, ...]
    ] | None = None,
) -> RuntimeRequestTicket:
    """Publish one detached follower's ordered replan receipt."""

    with controller._lock:
        if memory_owner_tokens is None:
            raise RuntimeControllerError(
                "priority compaction memory ownership is unavailable"
            )
        ticket = controller.ticket(request_id)
        if (
            ticket.ticket_id != expected_ticket_id
            or ticket.dispatch_state != "QUEUED"
        ):
            raise RuntimeControllerError(
                "priority compaction follower identity changed"
            )
        try:
            receipt = controller.queue.promote_priority_compaction_follower(
                request_id, observed_at_us
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        lease_status = ticket.lease_status
        if lease_status != "CANCELLED":
            if cancel_owner is None:
                raise RuntimeControllerError(
                    "priority compaction follower leases are live"
                )
            cancel_owner(request_id, observed_at_us)
            lease_status = "CANCELLED"
        memory_status = controller._detach_compaction_follower_memory(
            ticket,
            release_memory,
            memory_owner_tokens,
        )
        updated = replace(
            ticket,
            dispatch_state="REPLAN_REQUIRED",
            dispatch_receipt=receipt,
            lease_status=lease_status,
            memory_reservation_status=memory_status,
        )
        controller._tickets[request_id] = updated
        return updated


def priority_compaction_order(
    controller, request_ids: Sequence[str]
) -> tuple[str, ...]:
    """Return detached followers in immutable dispatch order."""

    try:
        return controller.queue.priority_compaction_order(request_ids)
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc
