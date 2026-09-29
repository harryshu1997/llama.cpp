"""RuntimeController dispatch-policy operations on its existing owner."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence

from ..runtime_dispatch_policy import RuntimeDispatchPolicy
from ..runtime_queue import RESIDENCY_HYSTERESIS_STAT_NAMES, RuntimeQueueError
from ..request_contracts.common import RuntimeControllerError, _integer, _text


DISPATCH_POLICY_STAT_NAMES = (
    "affinity_displacements",
    "affinity_displaced_attempts",
    "affinity_refusals",
    "continuous_join_bypasses",
    "continuous_join_publication_replans",
    "continuous_join_refusals",
    "publication_replans",
)
# Reported only under dispatch_policy.continuous_join, so other policies keep
# their statistics unchanged.
CONTINUOUS_JOIN_STAT_NAMES = frozenset({
    "continuous_join_bypasses",
    "continuous_join_publication_replans",
    "continuous_join_refusals",
})
CONTINUOUS_JOIN_BYPASS_KIND = "CONTINUOUS_JOIN_BARRIER_BYPASS"
RESIDENCY_HYSTERESIS_HELD_KIND = "RESIDENCY_HYSTERESIS_HELD"
RESIDENCY_HYSTERESIS_SKIPPED_KIND = "RESIDENCY_HYSTERESIS_SKIPPED"
_RESIDENCY_HYSTERESIS_NOTE_KINDS = frozenset({
    RESIDENCY_HYSTERESIS_HELD_KIND, RESIDENCY_HYSTERESIS_SKIPPED_KIND,
})
QUEUE_POLICY_EVENT_NAMES = (
    "early_capacity_promotions",
    "published_work_promotions",
)


def empty_dispatch_state() -> tuple[dict, dict, dict]:
    return {}, {}, {name: 0 for name in DISPATCH_POLICY_STAT_NAMES}


def configure_dispatch_policy(
    controller, policy: RuntimeDispatchPolicy
) -> None:
    if not isinstance(policy, RuntimeDispatchPolicy):
        raise RuntimeControllerError("runtime dispatch policy is invalid")
    with controller._lock:
        if controller._ticket_history:
            raise RuntimeControllerError(
                "runtime dispatch policy cannot change after admission"
            )
        try:
            controller.queue.set_policy(policy)
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        controller.dispatch_policy = policy


def dispatch_order_view(controller) -> Mapping[str, Mapping[str, object]]:
    with controller._lock:
        return controller.queue.dispatch_order_view()


def replan_queued_now(
    controller,
    request_ids: Sequence[str],
    reason: str,
    at_us: int,
    *,
    cancel_owner: Callable[[str, int], tuple[str, ...]],
    release_memory: Callable[[str], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Cancel queued reservations so their owners replan right away."""
    with controller._lock:
        replanned, _ = controller._mark_selected_followers(
            request_ids,
            reason,
            cancel_owner,
            release_memory,
            cancelled_at_us=_integer("runtime displacement at_us", at_us),
            defer_behind_predecessors=False,
        )
        return replanned


def refresh_replan_receipt(
    controller, request_id: str, expected_ticket_id: str, observed_at_us: int
):
    """Bind a waiting replan to its current queue generation.

    A replan wake committed earlier keeps its receipt while the queue entry
    is deferred and promoted again; replanning it for another owner must
    use the generation the queue holds now.
    """
    with controller._lock:
        ticket = controller.ticket(request_id)
        if (
            ticket.ticket_id != expected_ticket_id
            or ticket.dispatch_state != "REPLAN_REQUIRED"
        ):
            raise RuntimeControllerError(
                "runtime replan receipt owner changed"
            )
        try:
            receipt = controller.queue.promote_priority_compaction_follower(
                request_id, _integer("runtime replan receipt at_us", observed_at_us)
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        updated = replace(ticket, dispatch_receipt=receipt)
        controller._tickets[request_id] = updated
        return updated


def dispatch_bypass_count(controller, request_id: str) -> int:
    with controller._lock:
        return controller._dispatch_bypass_counts.get(request_id, 0)


@contextmanager
def dispatch_precedence(
    controller,
    request_id: str,
    follower_request_ids: Sequence[str],
    note: Mapping[str, object],
) -> Iterator[None]:
    """Let the next admission of ``request_id`` run before its followers."""
    request_id = _text("dispatch precedence request_id", request_id)
    followers = tuple(sorted(
        _text("dispatch precedence follower", value)
        for value in follower_request_ids
    ))
    if (
        not followers
        or len(followers) != len(set(followers))
        or request_id in followers
        or not controller.dispatch_policy.precedence_enabled
    ):
        raise RuntimeControllerError("runtime dispatch precedence is invalid")
    with controller._lock:
        if controller._pending_dispatch_precedence is not None:
            raise RuntimeControllerError(
                "runtime dispatch precedence is already pending"
            )
        controller._pending_dispatch_precedence = (
            request_id, followers, copy.deepcopy(dict(note)),
        )
    try:
        yield
        with controller._lock:
            if controller._pending_dispatch_precedence is not None:
                raise RuntimeControllerError(
                    "runtime dispatch precedence was not admitted"
                )
    finally:
        with controller._lock:
            controller._pending_dispatch_precedence = None


@contextmanager
def continuous_join_resolution(controller, request_id: str) -> Iterator[None]:
    """Mark the re-resolution of one joiner ahead of a displaced residency change.

    While it is pending the join selection also rejects the rows that the
    barrier bypass could never admit (plans preparing an exclusive residency
    device), so the joiner is planned as the desktop parent of its running
    server whenever that parent exists.
    """
    request_id = _text("continuous join resolution request_id", request_id)
    with controller._lock:
        if controller._pending_continuous_join_resolution is not None:
            raise RuntimeControllerError(
                "runtime continuous join resolution is already pending"
            )
        controller._pending_continuous_join_resolution = request_id
    try:
        yield
    finally:
        with controller._lock:
            controller._pending_continuous_join_resolution = None


def continuous_join_resolution_request_id(controller) -> str | None:
    """The joiner whose barrier-bypass re-resolution is in progress, if any."""
    with controller._lock:
        return controller._pending_continuous_join_resolution


def pending_dispatch_precedence(
    controller, request_id: str
) -> tuple[str, ...]:
    """Return the followers pending for the admission of ``request_id``."""
    pending = controller._pending_dispatch_precedence
    if pending is None or pending[0] != request_id:
        return ()
    return pending[1]


def account_dispatch_precedence(
    controller, request_id: str, ticket_id: str
) -> None:
    """Record one admitted displacement; called after the queue accepted it."""
    pending = controller._pending_dispatch_precedence
    if pending is None or pending[0] != request_id:
        return
    controller._pending_dispatch_precedence = None
    _, followers, note = pending
    if note.get("kind") == CONTINUOUS_JOIN_BYPASS_KIND:
        # The extension bound, not the affinity bypass counts, bounds a join.
        controller._dispatch_policy_stats["continuous_join_bypasses"] += 1
        controller._dispatch_policy_notes[ticket_id] = {
            **note,
            "displaced_request_ids": list(followers),
        }
        return
    bypassed = tuple(note.get("bypassed_request_ids", ()))
    for follower_id in bypassed:
        controller._dispatch_bypass_counts[follower_id] = (
            controller._dispatch_bypass_counts.get(follower_id, 0) + 1
        )
    stats = controller._dispatch_policy_stats
    stats["affinity_displacements"] += 1
    stats["affinity_displaced_attempts"] += len(followers)
    controller._dispatch_policy_notes[ticket_id] = {
        **note,
        "bypass_counts": {
            follower_id: controller._dispatch_bypass_counts[follower_id]
            for follower_id in sorted(bypassed)
        },
        "displaced_request_ids": list(followers),
        "kind": "MODEL_AFFINITY_DISPLACEMENT",
    }


def recovery_dispatch_order(
    controller, request_id: str
) -> tuple[int, tuple[str, ...]] | None:
    """Return the queue place of an active attempt that is about to fail.

    That is its sequence and the not-started attempts that wait on it, but
    only when one of them belongs to another model: its recovery then keeps
    that place (``retained_dispatch_order``), so the failure cannot hand the
    executor to a residency change the request was ahead of. Without such a
    waiter the order among same-model work is left as it is (None).
    """
    request_id = _text("runtime recovery order request_id", request_id)
    with controller._lock:
        view = controller.queue.dispatch_order_view()
        row = view.get(request_id)
        if row is None or row["state"] not in {"ACTIVE", "FINISHING"}:
            raise RuntimeControllerError(
                "runtime recovery order owner is not active"
            )
        waiting = tuple(
            other_id for other_id, other in view.items()
            if request_id in other["predecessor_request_ids"]
            and other["state"] not in {"ACTIVE", "FINISHING"}
        )
        artifact_sha256 = controller.ticket(request_id).model.artifact_sha256
        if not any(
            controller.ticket(other_id).model.artifact_sha256
            != artifact_sha256
            for other_id in waiting
        ):
            return None
        return row["sequence"], waiting


@contextmanager
def retained_dispatch_order(
    controller, request_id: str, order: tuple[int, Sequence[str]]
) -> Iterator[None]:
    """Let the next admission of ``request_id`` keep a failed attempt's place."""
    request_id = _text("runtime retained order request_id", request_id)
    with controller._lock:
        if controller._pending_retained_order is not None:
            raise RuntimeControllerError(
                "runtime retained order is already pending"
            )
        controller._pending_retained_order = (request_id, order)
    try:
        yield
    finally:
        with controller._lock:
            controller._pending_retained_order = None


def pending_retained_order(
    controller, request_id: str
) -> tuple[int, Sequence[str]] | None:
    pending = controller._pending_retained_order
    if pending is None or pending[0] != request_id:
        return None
    return pending[1]


def account_retained_order(
    controller, request_id: str, ticket_id: str
) -> None:
    """Consume the retained order and note the place the recovery kept."""
    pending = controller._pending_retained_order
    if pending is None or pending[0] != request_id:
        return
    controller._pending_retained_order = None
    view = controller.queue.dispatch_order_view()
    controller._dispatch_policy_notes[ticket_id] = {
        "kind": "RECOVERY_RETAINED_QUEUE_PLACE",
        "sequence": view[request_id]["sequence"],
        "waiting_request_ids": sorted(
            other_id for other_id, other in view.items()
            if request_id in other["predecessor_request_ids"]
        ),
    }


def _residency_hysteresis_note(
    queue, request_id: str
) -> tuple[str, dict[str, object]] | None:
    """(kind, body) of the note for a change's current hysteresis decision.

    A held decision is noted only once a waiter observed its hold; a skipped
    one as soon as it was taken.
    """
    decision = queue.residency_hysteresis_decision(request_id)
    if decision is None:
        return None
    body = {
        "barrier_request_id": request_id,
        "reason": decision["reason"],
        "released_at_us": decision["released_at_us"],
        **({} if "arrival_probability_ppm" not in decision else {
            "arrival_probability_ppm": decision["arrival_probability_ppm"],
        }),
    }
    if not decision["held"]:
        return RESIDENCY_HYSTERESIS_SKIPPED_KIND, body
    hold = queue.residency_hysteresis_hold(request_id)
    if hold is None or hold["released_at_us"] != decision["released_at_us"]:
        return None
    return RESIDENCY_HYSTERESIS_HELD_KIND, {
        **body, "held_until_us": hold["held_until_us"],
    }


def record_residency_hysteresis_hold(controller, request_id: str) -> None:
    """Note on the residency change's ticket whether hysteresis held it, and why.

    Called after a queue wait returns: RESIDENCY_HYSTERESIS_HELD (with the
    hold reason) once a waiter observed the hold, RESIDENCY_HYSTERESIS_SKIPPED
    (with the skip reason) for a change that is not held. The latest decision
    replaces an earlier hysteresis note and nests under a note of another kind.
    """
    request_id = _text("residency hysteresis request_id", request_id)
    with controller._lock:
        ticket = controller._tickets.get(request_id)
        found = _residency_hysteresis_note(controller.queue, request_id)
        if found is None or ticket is None:
            return
        kind, body = found
        note = controller._dispatch_policy_notes.get(ticket.ticket_id)
        if note is None or note.get("kind") in _RESIDENCY_HYSTERESIS_NOTE_KINDS:
            note = {"kind": kind, **body}
        else:
            note = {**note, "residency_hysteresis": {"kind": kind, **body}}
        controller._dispatch_policy_notes[ticket.ticket_id] = note


def record_dispatch_policy_event(
    controller, name: str, count: int = 1
) -> None:
    if name not in DISPATCH_POLICY_STAT_NAMES:
        raise RuntimeControllerError("runtime dispatch statistic is unknown")
    count = _integer("runtime dispatch statistic count", count)
    with controller._lock:
        controller._dispatch_policy_stats[name] += count


DISPATCH_POLICY_REFUSAL_KINDS = frozenset({"AFFINITY_REFUSED", "CONTINUOUS_JOIN_REFUSED"})
DISPATCH_POLICY_REFUSAL_LIMIT = 512


def record_dispatch_policy_refusal(
    controller, kind: str, request_id: str, reason: str, observed_at_us: int
) -> None:
    """Append one refused bypass with its reason (diagnostic; never rolled back).

    The first ``DISPATCH_POLICY_REFUSAL_LIMIT`` refusals are kept; the list is
    reported by ``dispatch_policy_state`` only when it is non-empty.
    """
    if kind not in DISPATCH_POLICY_REFUSAL_KINDS:
        raise RuntimeControllerError("runtime dispatch refusal kind is unknown")
    if type(request_id) is not str or type(reason) is not str:
        raise RuntimeControllerError("runtime dispatch refusal fields must be strings")
    observed_at_us = _integer("runtime dispatch refusal time", observed_at_us)
    with controller._lock:
        if len(controller._dispatch_policy_refusals) < DISPATCH_POLICY_REFUSAL_LIMIT:
            controller._dispatch_policy_refusals.append({
                "kind": kind,
                "observed_at_us": observed_at_us,
                "reason": reason,
                "request_id": request_id,
            })


def dispatch_policy_note(
    controller, ticket_id: str
) -> Mapping[str, object] | None:
    with controller._lock:
        note = controller._dispatch_policy_notes.get(ticket_id)
        return None if note is None else copy.deepcopy(note)


def dispatch_policy_state(controller) -> Mapping[str, object]:
    with controller._lock:
        return MappingProxyType({
            "bypass_counts": dict(sorted(
                controller._dispatch_bypass_counts.items()
            )),
            "policy": controller.dispatch_policy.to_json(),
            **({"refusals": copy.deepcopy(controller._dispatch_policy_refusals)}
               if controller._dispatch_policy_refusals else {}),
            **_residency_hysteresis_decisions(controller),
            "statistics": dict(sorted({
                **{name: 0 for name in QUEUE_POLICY_EVENT_NAMES},
                **({name: 0 for name in RESIDENCY_HYSTERESIS_STAT_NAMES}
                   if controller.dispatch_policy.residency_hysteresis_s else {}),
                **{
                    name: value
                    for name, value in controller._dispatch_policy_stats.items()
                    if name not in CONTINUOUS_JOIN_STAT_NAMES
                    or controller.dispatch_policy.continuous_join
                },
                **controller.queue.policy_events(),
            }.items())),
        })


def _residency_hysteresis_decisions(controller) -> dict[str, object]:
    """Every hold decision (held or skipped, with its reason) under hysteresis."""
    if not controller.dispatch_policy.residency_hysteresis_s:
        return {}
    rows = [dict(row) for row in controller.queue.residency_hysteresis_decisions()]
    return {"residency_hysteresis_decisions": rows} if rows else {}


def checkpoint_dispatch_state(controller) -> tuple[dict, dict, dict]:
    return (
        dict(controller._dispatch_bypass_counts),
        copy.deepcopy(controller._dispatch_policy_notes),
        dict(controller._dispatch_policy_stats),
    )


def parse_dispatch_state(state: object) -> tuple[dict, dict, dict]:
    try:
        counts, notes, stats = state
    except (TypeError, ValueError) as exc:
        raise RuntimeControllerError(
            "runtime controller checkpoint is invalid"
        ) from exc
    if (
        type(counts) is not dict
        or type(notes) is not dict
        or type(stats) is not dict
        or set(stats) != set(DISPATCH_POLICY_STAT_NAMES)
    ):
        raise RuntimeControllerError("runtime controller checkpoint is invalid")
    return dict(counts), copy.deepcopy(notes), dict(stats)
