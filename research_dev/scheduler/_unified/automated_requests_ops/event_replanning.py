"""Event re-planning: revisit queued and deferred decisions when their cause changes.

Opt-in with ``dispatch_policy.event_replanning``; without it nothing here runs and no
event is recorded (every entry point checks the flag first).

Without the flag a decision that waited for a runtime event is revisited only at the next
arrival, dispatch, request completion or decode boundary:

- a phone re-provisioning deferred because its sessions were in use
  (``PHONE_RESIDENCY_REPROVISION_DEFERRED_IN_USE``) is re-evaluated at a request completion
  (``runtime_requests.py`` ``complete_automated_request``) only when the last recorded
  decision still names blocked sessions or another model's desktop load is decided; a
  cancellation, a failure, a decode completion (the helper detaches at its last window)
  and a lease release re-evaluate nothing;
- a phone that becomes admissible again (thermal gate cleared, telemetry recovered,
  readmitted after quarantine) re-evaluates nothing, and attempts decided while it was out
  keep their plans;
- a proposed phone layout that no live request can prepare (e.g. every phone route is
  ``THERMAL_LIMIT`` in the fresh compile of the preparation envelope) is retried silently by
  the physical adapter's helper watchers, so the gap is invisible in the results (s2a:
  Gemma->Qwen proposal at 1,052.9 s never prepared, 005/007 decoded host-only ~170 s).

Under the flag:

- **release**: every helper-session release (``_release_request_helper_leases``,
  ``_close_request_helper_runtime``: decode completion, lease release, request completion,
  cancellation, failure) is noted; when the outermost scheduler call returns, the phone
  re-provisioning is re-evaluated if sessions that the last recorded decision counted as in
  use are free now (statistic ``event_replanning_release_reevaluations``);
- **device recovery**: a thermal gate clearing (``THERMAL_DEFERRAL_CLEARED`` of the route
  compilers' shared thermal log), telemetry recovery and a readmission re-evaluate the phone
  layout (``event_replanning_recovery_reevaluations``) and replan the not-started attempts
  decided while the device was out (wake reason ``event_device_admissible``,
  ``event_replanning_recovery_replans``);
- **blocked preparation**: a live request whose watcher cannot materialize the envelope of
  the proposed layout that changes its model records ``PREPARATION_BLOCKED`` (once per
  generation and reason, with the rejection reasons of the layout's routes) and
  ``PREPARATION_UNBLOCKED`` with the blocked duration when it can
  (``event_replanning_preparation_blocks``).

Batch membership changes and model availability already have handlers (queue wakes,
publication re-plans, READY-layout re-materialization); see the WS1 audit.
"""

from __future__ import annotations

from typing import Mapping

from ..common import _RECOVERABLE_ERRORS, _event_replanning_active

RELEASE_REEVALUATION_STAT = "event_replanning_release_reevaluations"
RECOVERY_REEVALUATION_STAT = "event_replanning_recovery_reevaluations"
RECOVERY_REPLAN_STAT = "event_replanning_recovery_replans"
PREPARATION_BLOCK_STAT = "event_replanning_preparation_blocks"
DEVICE_ADMISSIBLE_REASON = "event_device_admissible"
PREPARATION_BLOCKED = "PREPARATION_BLOCKED"
PREPARATION_UNBLOCKED = "PREPARATION_UNBLOCKED"
RELEASE_REEVALUATION_FAILED = "EVENT_RELEASE_LAYOUT_REEVALUATION_FAILED"
RECOVERY_REEVALUATION_FAILED = "EVENT_RECOVERY_LAYOUT_REEVALUATION_FAILED"
RECOVERY_REPLAN_FAILED = "EVENT_RECOVERY_REPLAN_FAILED"
_TERMINAL_STATES = frozenset({"CANCELLED", "COMPLETED", "FAILED"})
# materialization outcomes that are not a block: the layout does not concern the request
_NOT_BLOCKED = frozenset({"MODEL_NOT_CHANGED_BY_LAYOUT", "PREPARATION_CONTEXT_UNAVAILABLE"})


def event_replanning_enabled(controller: object) -> bool:
    return _event_replanning_active(controller)


def _pending(controller) -> list[dict[str, object]]:
    rows = controller.__dict__.get("_event_replanning_pending")
    if rows is None:
        rows = []
        controller._event_replanning_pending = rows
    return rows


def note_resource_release(controller, request_id: str, at_us: int, source: str) -> None:
    """A request released phone helper sessions or leases (no-op without the flag)."""
    if not event_replanning_enabled(controller) or type(at_us) is not int:
        return
    _pending(controller).append({
        "at_us": at_us, "kind": "RELEASE", "request_id": request_id, "source": source,
    })


def note_device_admissible(
    controller, device_id: str, at_us: int, source: str, since_us: int | None,
) -> None:
    """A device became admissible again after ``since_us`` (no-op without the flag)."""
    if not event_replanning_enabled(controller) or type(at_us) is not int:
        return
    _pending(controller).append({
        "at_us": at_us, "device_id": device_id, "kind": "RECOVERY",
        "since_us": since_us if type(since_us) is int else None, "source": source,
    })


def _poll_thermal_clears(controller) -> None:
    """Turn new THERMAL_DEFERRAL_CLEARED rows of the shared thermal gate log into recoveries."""
    log = getattr(controller, "_thermal_gate_log", None)
    if log is None:
        return
    rows = log.events()
    seen = controller.__dict__.get("_event_replanning_thermal_rows", 0)
    controller._event_replanning_thermal_rows = len(rows)
    for row in rows[seen:]:
        if row.get("kind") == "THERMAL_DEFERRAL_CLEARED":
            note_device_admissible(
                controller, str(row["device_id"]), int(row["at_us"]),
                "THERMAL_DEFERRAL_CLEARED", row.get("onset_at_us"),
            )


def process_pending_events(controller) -> None:
    """Handle the noted events once; called when the outermost serialized call returns."""
    if not event_replanning_enabled(controller) or controller.__dict__.get(
        "_event_replanning_processing"
    ):
        return
    controller._event_replanning_processing = True
    try:
        _poll_thermal_clears(controller)
        pending = controller.__dict__.get("_event_replanning_pending")
        if not pending:
            return
        rows = tuple(pending)
        pending.clear()
        releases = tuple(row for row in rows if row["kind"] == "RELEASE")
        recoveries = tuple(row for row in rows if row["kind"] == "RECOVERY")
        if releases:
            _reevaluate_after_release(controller, releases)
        if recoveries:
            _recover_devices(controller, recoveries)
    finally:
        controller._event_replanning_processing = False


def _stat(controller, name: str, count: int = 1) -> None:
    controller._runtime_controller.record_dispatch_policy_event(name, count)


def _live_tickets(controller):
    return tuple(
        row for row in controller._runtime_controller.current_tickets()
        if row.dispatch_state not in _TERMINAL_STATES
    )


def _evaluation_ticket(controller):
    """The ticket a layout re-evaluation is made for: a decided desktop load first (its
    load window times the swap), else the first live ticket; None without one."""
    from ..phone_residency_ops import reprovision

    decided = reprovision._decided_desktop_load(controller)
    if decided is not None:
        return decided
    return min(_live_tickets(controller), key=lambda row: row.request.request_id, default=None)


def _layout_reevaluable(controller) -> bool:
    placement = controller._model_placement_controller
    return (
        controller._runtime_capabilities is not None
        and getattr(controller, "_fixed_phone_residency", None) is None
        and placement.preparing_phone_layout() is None
        and any(row.phone_sessions for row in controller._runtime_capabilities.executors)
    )


def _reevaluate_after_release(controller, releases) -> None:
    """Re-evaluate a phone re-provisioning whose in-use sessions a release freed."""
    from ..phone_residency_ops import reprovision

    if getattr(controller, "_phone_reprovisioning", None) is None or not _layout_reevaluable(
        controller
    ):
        return
    recorded = reprovision._last_recorded_reprovision(controller)
    if recorded is None:
        return
    previous = recorded.get("in_use_session_ids") or ()
    freed = set(previous if isinstance(previous, (list, tuple)) else ()) - set(
        reprovision._in_use_session_ids(controller)
    )
    if not freed:
        return
    ticket = _evaluation_ticket(controller)
    if ticket is None:
        return
    released = sorted(str(row["request_id"]) for row in releases)
    reprovision._reevaluate_phone_layout(
        controller, ticket, max(int(row["at_us"]) for row in releases),
        RELEASE_REEVALUATION_FAILED, event_request_id=released[0],
    )
    _stat(controller, RELEASE_REEVALUATION_STAT)


def _device_ids_of_executor(catalog, executor_id: str | None) -> frozenset[str]:
    if executor_id is None:
        return frozenset()
    capability = catalog.executor_by_id.get(executor_id)
    if capability is not None:
        return frozenset({capability.device_id})
    coordinator = catalog.composite_executor_by_id.get(executor_id)
    if coordinator is not None:
        return frozenset(coordinator.participant_device_ids)
    return frozenset()


def _decided_without_device(controller, ticket, device_id: str, since_us, at_us: int) -> bool:
    """A not-started attempt decided while the device was out whose plan does not use it
    although one of its routes did."""
    plan = ticket.execution_plan
    decided_at_us = ticket.runtime_observation.captured_at_us
    if (
        plan is None
        or device_id in plan.device_ids
        or decided_at_us > at_us
        or (since_us is not None and decided_at_us < since_us)
    ):
        return False
    catalog = controller._runtime_capabilities
    return any(
        device_id in _device_ids_of_executor(catalog, row.executor_id)
        for row in ticket.cost_estimates.estimates
    )


def _recover_devices(controller, recoveries) -> None:
    """Re-evaluate the phone layout and replan attempts decided while a device was out."""
    catalog = controller._runtime_capabilities
    if catalog is None:
        return
    at_us = max(int(row["at_us"]) for row in recoveries)
    phone_devices = {row.device_id for row in catalog.executors if row.phone_sessions}
    if phone_devices.intersection(row["device_id"] for row in recoveries) and _layout_reevaluable(
        controller
    ):
        ticket = _evaluation_ticket(controller)
        if ticket is not None:
            from ..phone_residency_ops import reprovision

            reprovision._reevaluate_phone_layout(
                controller, ticket, at_us, RECOVERY_REEVALUATION_FAILED,
            )
            _stat(controller, RECOVERY_REEVALUATION_STAT)
    runtime = controller._runtime_controller
    view = runtime.dispatch_order_view()
    waiting = [
        row for row in runtime.current_tickets(("QUEUED",))
        if any(
            _decided_without_device(
                controller, row, str(recovery["device_id"]), recovery["since_us"],
                int(recovery["at_us"]),
            )
            for recovery in recoveries
        )
    ]
    if not waiting:
        return
    order = sorted(
        (row.request.request_id for row in waiting),
        key=lambda request_id: (view.get(request_id, {}).get("sequence", 0), request_id),
    )
    try:
        with controller._transaction(convert=False):
            woken = runtime.replan_queued_now(
                tuple(order), DEVICE_ADMISSIBLE_REASON, at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            order[0], RECOVERY_REPLAN_FAILED, at_us,
            {"reason": str(exc), "request_ids": list(order)},
        )
        return
    if woken:
        _stat(controller, RECOVERY_REPLAN_STAT, len(woken))


def _last_block(controller, request_id: str, generation: int) -> Mapping[str, object] | None:
    """The latest PREPARATION_BLOCKED/UNBLOCKED row of one request and layout generation."""
    for row in reversed(controller._model_placement_controller.request_helper_events(request_id)):
        if (
            row.get("kind") in {PREPARATION_BLOCKED, PREPARATION_UNBLOCKED}
            and row.get("phone_layout_generation") == generation
        ):
            return row
    return None


def record_preparation_outcome(
    controller, ticket, layout, helper, diagnostics: Mapping[str, object], observed_at_us: int,
) -> None:
    """Make a blocked preparation of a proposed layout visible (event re-planning only)."""
    request_id = ticket.request.request_id
    previous = _last_block(controller, request_id, layout.generation)
    placement = controller._model_placement_controller
    if helper is not None:
        if previous is not None and previous.get("kind") == PREPARATION_BLOCKED:
            since = int(previous.get("blocked_since_us", previous["observed_at_us"]))
            placement.record_request_helper_event(request_id, PREPARATION_UNBLOCKED, observed_at_us, {
                "blocked_reason": previous.get("reason"),
                "blocked_since_us": since,
                "blocked_us": max(0, observed_at_us - since),
                "phone_layout_generation": layout.generation,
                "phone_layout_geometry_sha256": layout.layout.geometry_sha256,
                "request_ticket_id": ticket.ticket_id,
            })
        return
    reason = diagnostics.get("reason")
    if type(reason) is not str or reason in _NOT_BLOCKED:
        return
    payload = {
        **{key: value for key, value in diagnostics.items() if key != "reason"},
        "phone_layout_generation": layout.generation,
        "phone_layout_geometry_sha256": layout.layout.geometry_sha256,
        "phone_layout_state": layout.state,
        "reason": reason,
        "request_ticket_id": ticket.ticket_id,
    }
    if previous is not None and previous.get("kind") == PREPARATION_BLOCKED and all(
        previous.get(key) == value for key, value in payload.items()
    ):
        return
    since = (
        int(previous.get("blocked_since_us", previous["observed_at_us"]))
        if previous is not None and previous.get("kind") == PREPARATION_BLOCKED
        else observed_at_us
    )
    placement.record_request_helper_event(
        request_id, PREPARATION_BLOCKED, observed_at_us, {**payload, "blocked_since_us": since},
    )
    _stat(controller, PREPARATION_BLOCK_STAT)
