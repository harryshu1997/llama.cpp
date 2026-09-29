"""RuntimeController leases operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from ..policy import LeaseRecord
from ..runtime_queue import RuntimeQueueError
from ..runtime_resources import runtime_preparation_windows
from ..request_contracts.common import RuntimeControllerError, _text, _integer
from ..request_contracts.receipts import (
    assess_runtime_completion,
    RuntimeLeaseExtensionReceipt,
    RuntimeCompletionReceipt,
)


def extend(
    controller,
    request_id: str,
    *,
    at_us: int,
    reserved_until_us: int,
    extend_leases: Callable[
        [Sequence[str], int, Sequence[str], int],
        tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]],
    ],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
    retime_leases: Callable | None = None,
) -> RuntimeLeaseExtensionReceipt:
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError("runtime request is not active")
        if ticket.lease_status == "RELEASED_PENDING_RECEIPT":
            raise RuntimeControllerError(
                "runtime request capacity is already released"
            )
        at_us = _integer("runtime lease extension at_us", at_us)
        reserved_until_us = _integer(
            "runtime lease extension end", reserved_until_us
        )
        if ticket.transition_status == "PENDING" and (
            ticket.prepare_lease_tokens - ticket.completed_prepare_lease_tokens
        ):
            if retime_leases is None:
                raise RuntimeControllerError("phase lease renewal callback is absent")
            return _extend_preparation(
                controller, ticket, at_us, reserved_until_us, retime_leases, release_memory)
        changed = tuple(
            lease for lease in ticket.live_leases
            if ticket.final_reserved_until_us[lease.token]
                < reserved_until_us
        )
        if not changed:
            return RuntimeLeaseExtensionReceipt(
                request_id=request_id,
                at_us=at_us,
                cancelled_queued_tokens={},
                extended_leases=(),
            )

        def commit(
            follower_ids: tuple[str, ...],
        ) -> tuple[
            Mapping[str, int], Mapping[str, tuple[str, ...]]
        ]:
            return extend_leases(
                tuple(lease.token for lease in changed),
                reserved_until_us,
                follower_ids,
                0,
            )

        try:
            follower_ids, result = (
                controller.queue.commit_conflicting_replans(
                    request_id,
                    changed,
                    reserved_until_us,
                    "lease_upper_bound_overrun",
                    commit,
                )
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        previous, cancelled = result
        final = dict(ticket.final_reserved_until_us)
        for lease in changed:
            final[lease.token] = reserved_until_us
        for follower_id in follower_ids:
            follower = controller.ticket(follower_id)
            memory_status = follower.memory_reservation_status
            if (
                release_memory is not None
                and memory_status == "RESERVED_ATOMIC"
            ):
                released_memory = release_memory(follower_id)
                if set(released_memory) != {
                    row.token for row in follower.memory_reservations
                }:
                    raise RuntimeControllerError(
                        "runtime overrun follower memory release differs"
                    )
                memory_status = "CANCELLED"
            controller._tickets[follower_id] = replace(
                follower,
                lease_status="CANCELLED",
                memory_reservation_status=memory_status,
            )
        controller._tickets[request_id] = replace(
            ticket, final_reserved_until_us=final
        )
        return RuntimeLeaseExtensionReceipt(
            request_id=request_id,
            at_us=at_us,
            cancelled_queued_tokens=cancelled,
            extended_leases=tuple(
                MappingProxyType({
                    "previous_reserved_until_us": previous[lease.token],
                    "reserved_until_us": reserved_until_us,
                    "resource_id": lease.resource_id,
                    "token": lease.token,
                })
                for lease in changed
            ),
        )


def _commit_phase_windows(controller, ticket, windows, retime_leases, release_memory, reason):
    request_id = ticket.request.request_id
    try:
        follower_ids, result = controller.queue.commit_conflicting_replans(
            request_id, ticket.decision.leases, max(end for _, end in windows.values()),
            reason, lambda followers: retime_leases(
                windows, cancelled_owner_ids=followers, cancellation_at_us=0),
            phase_windows=windows,
        )
    except RuntimeQueueError as exc:
        raise RuntimeControllerError(str(exc)) from exc
    for identity in follower_ids:
        follower = controller.ticket(identity)
        status = follower.memory_reservation_status
        if release_memory is not None and status == "RESERVED_ATOMIC":
            if set(release_memory(identity)) != {row.token for row in follower.memory_reservations}:
                raise RuntimeControllerError("runtime phase follower memory release differs")
            status = "CANCELLED"
        controller._tickets[identity] = replace(follower, lease_status="CANCELLED",
                                               memory_reservation_status=status)
    return result


def _extend_preparation(controller, ticket, at_us, until_us, retime_leases, release_memory):
    pending = ticket.prepare_lease_tokens - ticket.completed_prepare_lease_tokens
    prepare_end = min(ticket.final_reserved_until_us[token] for token in pending)
    if prepare_end >= until_us:
        return RuntimeLeaseExtensionReceipt(ticket.request.request_id, at_us, {}, ())
    windows = runtime_preparation_windows(ticket, active_until_us=until_us)
    previous, cancelled = _commit_phase_windows(
        controller, ticket, windows, retime_leases, release_memory, "preparation_lease_overrun")
    final = {token: max(ticket.final_reserved_until_us[token], end)
             for token, (_, end) in windows.items()}
    controller._tickets[ticket.request.request_id] = replace(ticket, final_reserved_until_us=final)
    return RuntimeLeaseExtensionReceipt(
        ticket.request.request_id, at_us, cancelled,
        tuple({"previous_reserved_until_us": previous[row.token],
               "reserved_until_us": final[row.token], "resource_id": row.resource_id,
               "token": row.token} for row in ticket.decision.leases
              if row.token not in ticket.completed_prepare_lease_tokens
              and windows[row.token][1] > previous[row.token]),
    )


def finish_prepare_phase(controller, request_id, *, retime_leases, release_memory,
                         previously_completed_tokens=()):
    with controller._lock:
        ticket = controller.ticket(request_id)
        if not ticket.prepare_lease_tokens:
            return ticket
        if not ticket.transition_receipts or ticket.transition_status not in {"PENDING", "COMPLETED"}:
            raise RuntimeControllerError("phase release requires completed transitions")
        windows = runtime_preparation_windows(ticket)
        final = {token: max(ticket.final_reserved_until_us[token], end)
                 for token, (_, end) in windows.items()}
        _commit_phase_windows(controller, ticket, windows, retime_leases,
                              release_memory, "preparation_phase_completed")
        released = ticket.completed_prepare_lease_tokens - set(previously_completed_tokens)
        if released:
            controller.queue.release_prepare_leases(
                request_id, released, preparation_complete=ticket.transition_status == "COMPLETED")
        updated = replace(ticket, final_reserved_until_us=final)
        controller._tickets[request_id] = updated
        return updated


def extend_external(
    controller,
    owner_id: str,
    leases: Sequence[LeaseRecord],
    *,
    at_us: int,
    reserved_until_us: int,
    extend_leases: Callable[
        [Sequence[str], int, Sequence[str], int],
        tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]],
    ],
) -> tuple[str, ...]:
    """Atomically extend an observed phase and replan its followers."""
    with controller._lock:
        owner_id = _text("runtime external owner_id", owner_id)
        lease_rows = tuple(leases)
        if not lease_rows or any(
            not isinstance(lease, LeaseRecord) for lease in lease_rows
        ):
            raise RuntimeControllerError(
                "runtime external leases are invalid"
            )
        at_us = _integer("runtime external extension at_us", at_us)
        reserved_until_us = _integer(
            "runtime external extension end", reserved_until_us
        )

        def commit(follower_ids: tuple[str, ...]) -> object:
            return extend_leases(
                tuple(lease.token for lease in lease_rows),
                reserved_until_us,
                follower_ids,
                0,
            )

        try:
            follower_ids, _ = controller.queue.commit_conflicting_replans(
                owner_id,
                lease_rows,
                reserved_until_us,
                "external_phase_lease_overrun",
                commit,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        for follower_id in follower_ids:
            follower = controller.ticket(follower_id)
            controller._tickets[follower_id] = replace(
                follower,
                lease_status="CANCELLED",
            )
        return follower_ids


def release_capacity(
    controller,
    request_id: str,
    actual_end_us: int,
    *,
    resource_capacities: Mapping[str, int] | None = None,
    release_lease: Callable[[str, int], None],
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> RuntimeCompletionReceipt:
    """Release physical resources while terminal evidence is collected."""

    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state != "ACQUIRED":
            raise RuntimeControllerError("runtime request is not active")
        if request_id in controller._capacity_releases:
            raise RuntimeControllerError(
                "runtime request capacity is already released"
            )
        if ticket.transition_status not in {
            "COMPLETED", "NOT_REQUIRED"
        }:
            raise RuntimeControllerError(
                "runtime transitions are not complete"
            )
        latency, coverage = assess_runtime_completion(
            replace(ticket.decision, leases=ticket.live_leases),
            {row.token: ticket.final_reserved_until_us[row.token] for row in ticket.live_leases},
            actual_end_us,
        )
        try:
            early_frontier = controller.queue.early_completion_frontier(
                request_id,
                actual_end_us,
                coverage.final_reserved_until_us,
                resource_capacities,
            )
            idle_frontier = controller.queue.idle_completion_frontier(
                request_id,
                actual_end_us,
                resource_capacities,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        compatible_frontier = tuple(
            follower_id
            for follower_id in dict.fromkeys(
                early_frontier + idle_frontier
            )
            if (
                controller.ticket(follower_id).model.artifact_sha256
                    == ticket.model.artifact_sha256
                and controller.ticket(follower_id).binding.executor_id
                    == ticket.binding.executor_id
            )
        )
        capacity_frontier = tuple(dict.fromkeys(
            early_frontier + idle_frontier
        ))
        try:
            downstream_projections = (
                ()
                if not capacity_frontier
                else controller.queue.downstream_projection_requests(
                    capacity_frontier
                )
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        early_replans, _ = controller._mark_selected_followers(
            compatible_frontier,
            "capacity_released_early",
            cancel_owner,
            release_memory,
            cancelled_at_us=actual_end_us,
        )
        replanned = controller._mark_residency_transition_frontier(
            ticket, actual_end_us, cancel_owner, release_memory
        )
        if any(
            actual_end_us
                > ticket.final_reserved_until_us[lease.token]
            for lease in ticket.live_leases
        ):
            overrun_replans, _ = controller._mark_followers(
                replace(ticket.decision, leases=ticket.live_leases),
                actual_end_us,
                "lease_coverage_overrun",
                cancel_owner,
                release_memory,
            )
            replanned = tuple(dict.fromkeys(
                replanned + overrun_replans
            ))
        remaining_early_replans, _ = controller._mark_selected_followers(
            capacity_frontier,
            "capacity_released_early",
            cancel_owner,
            release_memory,
            cancelled_at_us=actual_end_us,
        )
        controller._mark_selected_followers(
            downstream_projections,
            "predecessor_replan",
            cancel_owner,
            release_memory,
            cancelled_at_us=actual_end_us,
            defer_behind_predecessors=True,
        )
        capacity_replans = tuple(dict.fromkeys(
            early_replans + remaining_early_replans
        ))
        replanned = tuple(dict.fromkeys(
            capacity_replans + replanned
        ))
        released = []
        expired_phases = []
        late = []
        for lease in ticket.live_leases:
            release_end_us = max(
                (max(row.finished_us for row in ticket.transition_receipts)
                 if ticket.prepare_lease_tokens else lease.start_us),
                min(
                    actual_end_us,
                    ticket.final_reserved_until_us[lease.token],
                ),
            )
            release_lease(lease.token, release_end_us)
            released.append(lease.token)
            if actual_end_us > ticket.final_reserved_until_us[lease.token]:
                if lease.predicted_end_us < ticket.decision.finish_us:
                    expired_phases.append(lease.token)
                else:
                    late.append(lease.token)
        if release_memory is not None:
            released_memory = release_memory(request_id)
            if set(released_memory) != {
                row.token for row in ticket.memory_reservations
            }:
                raise RuntimeControllerError(
                    "runtime memory release differs from ticket"
                )
        try:
            controller.queue.release_capacity(
                request_id,
                actual_end_us,
                early_replan_request_ids=capacity_replans,
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        updated = replace(
            ticket,
            lease_status="RELEASED_PENDING_RECEIPT",
            memory_reservation_status=(
                "RELEASED"
                if ticket.memory_reservations
                else ticket.memory_reservation_status
            ),
            actual_end_us=actual_end_us,
        )
        controller._tickets[request_id] = updated
        receipt = RuntimeCompletionReceipt(
            request_id=request_id,
            route_id=ticket.decision.route_id,
            actual_end_us=actual_end_us,
            released_tokens=tuple(released),
            completion_event_replans=replanned,
            expired_phase_tokens=tuple(expired_phases),
            late_tokens=tuple(late),
            latency_upper_bound=latency,
            lease_coverage=coverage,
            quarantine_action=None,
        )
        controller._capacity_releases[request_id] = receipt
        return receipt
