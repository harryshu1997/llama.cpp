"""RuntimeController cohorts operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..policy import LeaseRecord
from ..runtime_queue import RuntimeQueueError
from ..runtime_decode_cohort import RuntimeDecodeCohortBinding
from ..request_contracts.common import RuntimeControllerError, _text, RUNTIME_TERMINAL_STATES
from ..request_contracts.ticket import RuntimeRequestTicket


def detach_decode_cohort_for_replan(
    controller,
    request_id: str,
    previous: RuntimeDecodeCohortBinding,
    transferred_leases: Sequence[LeaseRecord] = (),
) -> RuntimeRequestTicket:
    """Remove prospective cohort state from a replacement attempt."""
    if not isinstance(previous, RuntimeDecodeCohortBinding):
        raise RuntimeControllerError(
            "runtime decode cohort binding is invalid"
        )
    rows = tuple(transferred_leases)
    with controller._lock:
        ticket = controller.ticket(request_id)
        if (
            ticket.dispatch_state != "REPLAN_REQUIRED"
            or ticket.decode_cohort is None
            or ticket.decode_cohort.cohort_id != previous.cohort_id
            or request_id not in previous.member_request_ids
            or tuple(
                row.token for row in ticket.decision.leases
            ) != previous.shared_lease_tokens
        ):
            raise RuntimeControllerError(
                "runtime decode cohort replan state differs"
            )
        if rows:
            by_token = {row.token: row for row in rows}
            existing = {
                row.token: row for row in ticket.decision.leases
            }
            if (
                len(by_token) != len(rows)
                or tuple(by_token) != previous.shared_lease_tokens
                or set(existing) != set(by_token)
                or any(
                    row.owner_id != request_id
                    or row.resource_id
                        != existing[row.token].resource_id
                    or row.lanes != existing[row.token].lanes
                    or row.start_us != existing[row.token].start_us
                    or row.predicted_end_us
                        != existing[row.token].predicted_end_us
                    or row.reserved_until_us
                        != existing[row.token].reserved_until_us
                    for row in rows
                )
            ):
                raise RuntimeControllerError(
                    "runtime decode cohort replan leases differ"
                )
        updated = replace(
            ticket,
            decode_cohort=None,
        )
        controller._tickets[request_id] = updated
        return updated


def dissolve_decode_cohort_singleton(
    controller,
    request_id: str,
    binding: RuntimeDecodeCohortBinding,
    leases: Sequence[LeaseRecord],
) -> RuntimeRequestTicket:
    """Return a one-member prospective cohort to request ownership."""
    if not isinstance(binding, RuntimeDecodeCohortBinding):
        raise RuntimeControllerError(
            "runtime decode cohort binding is invalid"
        )
    rows = tuple(leases)
    with controller._lock:
        ticket = controller.ticket(request_id)
        if (
            binding.member_request_ids != (request_id,)
            or ticket.dispatch_state != "QUEUED"
            or ticket.decode_cohort is None
            or not binding.sealed
            or replace(ticket.decode_cohort, sealed=True) != binding
            or tuple(row.token for row in rows)
                != binding.shared_lease_tokens
            or any(row.owner_id != request_id for row in rows)
        ):
            raise RuntimeControllerError(
                "runtime decode cohort singleton state differs"
            )
        previous = {row.token: row for row in ticket.decision.leases}
        if (
            set(previous) != set(binding.shared_lease_tokens)
            or any(
                row.resource_id != previous[row.token].resource_id
                or row.lanes != previous[row.token].lanes
                or row.start_us != previous[row.token].start_us
                or row.predicted_end_us
                    != previous[row.token].predicted_end_us
                or row.reserved_until_us
                    != previous[row.token].reserved_until_us
                for row in rows
            )
        ):
            raise RuntimeControllerError(
                "runtime decode cohort singleton leases differ"
            )
        decision = replace(ticket.decision, leases=rows)
        try:
            controller.queue.replace_decision(
                request_id, ticket.decision, decision
            )
            controller.queue.bind_decode_cohort(request_id, None)
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        updated = replace(
            ticket,
            decision=decision,
            final_reserved_until_us={
                row.token: row.reserved_until_us for row in rows
            },
            decode_cohort=None,
        )
        controller._tickets[request_id] = updated
        return updated


def transfer_decode_cohort_singleton(
    controller,
    request_id: str,
    binding: RuntimeDecodeCohortBinding,
    leases: Sequence[LeaseRecord],
) -> RuntimeRequestTicket:
    """Transfer an active cohort lease to its sole survivor."""
    if not isinstance(binding, RuntimeDecodeCohortBinding):
        raise RuntimeControllerError(
            "runtime decode cohort binding is invalid"
        )
    rows = tuple(leases)
    with controller._lock:
        ticket = controller.ticket(request_id)
        if (
            ticket.dispatch_state in RUNTIME_TERMINAL_STATES
            or ticket.decode_cohort is None
            or ticket.decode_cohort.cohort_id != binding.cohort_id
            or request_id not in binding.member_request_ids
            or tuple(row.token for row in rows)
                != binding.shared_lease_tokens
            or any(row.owner_id != request_id for row in rows)
        ):
            raise RuntimeControllerError(
                "runtime decode cohort handoff state differs"
            )
        previous = {row.token: row for row in ticket.decision.leases}
        if (
            set(previous) != set(binding.shared_lease_tokens)
            or any(
                row.resource_id != previous[row.token].resource_id
                or row.lanes != previous[row.token].lanes
                or row.start_us != previous[row.token].start_us
                or row.predicted_end_us
                    != previous[row.token].predicted_end_us
                or row.reserved_until_us
                    < previous[row.token].reserved_until_us
                or row.reserved_until_us
                    != ticket.final_reserved_until_us.get(row.token)
                for row in rows
            )
        ):
            raise RuntimeControllerError(
                "runtime decode cohort handoff leases differ"
            )
        decision = replace(ticket.decision, leases=rows)
        live_rows = tuple(
            row for row in rows
            if row.token not in ticket.completed_prepare_lease_tokens
        )
        try:
            controller.queue.replace_decision(
                request_id,
                replace(ticket.decision, leases=ticket.live_leases),
                replace(decision, leases=live_rows),
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        updated = replace(
            ticket,
            decision=decision,
            final_reserved_until_us={
                row.token: row.reserved_until_us for row in rows
            },
        )
        controller._tickets[request_id] = updated
        return updated


def extend_decode_cohort(
    controller,
    binding: RuntimeDecodeCohortBinding,
    leases: Sequence[LeaseRecord],
    *,
    pending_member_request_id: str | None = None,
) -> None:
    """Apply one shared lease update before publishing a new member."""
    if not isinstance(binding, RuntimeDecodeCohortBinding):
        raise RuntimeControllerError(
            "runtime decode cohort binding is invalid"
        )
    if pending_member_request_id is not None:
        pending_member_request_id = _text(
            "runtime pending decode cohort member",
            pending_member_request_id,
        )
        if pending_member_request_id not in binding.member_request_ids:
            raise RuntimeControllerError(
                "runtime pending decode cohort member differs"
            )
    rows = tuple(leases)
    by_token = {row.token: row for row in rows}
    if (
        not rows
        or len(by_token) != len(rows)
        or tuple(by_token) != binding.shared_lease_tokens
        or any(row.owner_id != binding.cohort_id for row in rows)
    ):
        raise RuntimeControllerError(
            "runtime decode cohort lease update is invalid"
        )
    with controller._lock:
        for request_id in binding.member_request_ids:
            if request_id == pending_member_request_id:
                continue
            ticket = controller._tickets.get(request_id)
            if (
                ticket is None
                or ticket.dispatch_state in RUNTIME_TERMINAL_STATES
            ):
                continue
            if ticket.decode_cohort is not None and (
                ticket.decode_cohort.cohort_id != binding.cohort_id
            ):
                raise RuntimeControllerError(
                    "runtime decode cohort member differs"
                )
            final = dict(ticket.final_reserved_until_us)
            for token in binding.shared_lease_tokens:
                if token not in final:
                    raise RuntimeControllerError(
                        "runtime decode cohort lease differs"
                    )
                if (
                    by_token[token].reserved_until_us < final[token]
                ):
                    raise RuntimeControllerError(
                        "runtime decode cohort lease was shortened"
                    )
                final[token] = by_token[token].reserved_until_us
            try:
                controller.queue.bind_decode_cohort(request_id, binding)
            except RuntimeQueueError as exc:
                raise RuntimeControllerError(str(exc)) from exc
            controller._tickets[request_id] = replace(
                ticket,
                final_reserved_until_us=final,
                decode_cohort=binding,
            )
