"""AdaptiveDecodeController completion operations on its existing owner."""

from __future__ import annotations

import time

from . import coherence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeControl,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodeRawWindowObservation,
)
from ..adaptive_decode_state import _AdaptiveSession


def _seal_session(controller, session: _AdaptiveSession) -> None:
    if (
        session.tail_seal_token is None
        or session.awaiting_boundary is not None
        or session.awaiting_control is not None
        or session.transition_policy is not None
        or session.transition_ack is not None
        or controller._sessions.get(session.request_id) is not session
        or session.request_id in controller._sealed_sessions
    ):
        raise AdaptiveDecodeError("adaptive sealed state is invalid")
    del controller._sessions[session.request_id]
    controller._sealed_sessions[session.request_id] = session


def _seal_released_phone_tail(
    controller, session, boundary, observation, terminal_token_index, terminal_at_us,
) -> AdaptiveDecodeDirective:
    previous = session.records[-1] if session.records else None
    acknowledged = next(
        (row for row in reversed(session.records) if row.applied_ack is not None), None,
    )
    if (
        session.awaiting_boundary != boundary
        or not isinstance(observation, AdaptiveDecodeRawWindowObservation)
        or observation.failure_reason != "released_slot_phone_tail"
        or not observation.output_valid
        or "physical:terminal-release-confirmed" not in observation.evidence_ids
        or type(terminal_token_index) is not int
        or terminal_token_index != session.output_tokens
        or type(terminal_at_us) is not int
        or terminal_at_us < boundary.finished_at_us
        or session.tail_seal_token != boundary.token_end
        or session.tail_seal_reason != "server_release_guard"
        or session.state != "EXPLOITING"
        or session.current_policy != boundary.policy
        or boundary.policy.baseline
        or session.current_ack != boundary.applied_ack
        or acknowledged is None
        or acknowledged.policy != boundary.policy
        or session.awaiting_control is not None
        or session.transition_policy is not None
        or session.transition_ack is not None
        or session.pending_session_drain_policy is not None
        or session.pending_helper_refresh_policy is not None
        or previous is None
        or previous.token_end != boundary.token_start
        or previous.policy != boundary.policy
        or not previous.output_valid
        or previous.failure_reason is not None
    ):
        raise AdaptiveDecodeError("adaptive released phone tail transaction differs")
    # No counters were observed. Terminal native proofs must account for this tail.
    session.awaiting_boundary = None
    session.tail_seal_token = boundary.token_start
    session.window_start_token = None
    session.window_start_us = None
    session.target_token = None
    controller._seal_session(session)
    return AdaptiveDecodeDirective(
        state=session.state, reason="TAIL_SEALED", target_token_index=None,
    )


def seal_tail(
    controller,
    request_id: str,
    *,
    slot_id: int,
    token_index: int,
    reason: str,
) -> None:
    if type(reason) is not str or not reason or not reason.isascii():
        raise AdaptiveDecodeError("adaptive tail seal reason is invalid")
    with controller._lock:
        session = controller._session(request_id)
        if slot_id != session.slot_id:
            raise AdaptiveDecodeError("adaptive slot identity differs")
        acknowledged_boundary = (
            session.window_start_token == token_index
            and session.current_ack is not None
            and session.current_ack.applied_token_index == token_index
            and bool(session.records)
            and session.records[-1].token_end == token_index
        )
        if (
            session.tail_seal_token is not None
            or session.awaiting_boundary is not None
            or session.awaiting_control is not None
            or session.window_start_token is None
            or not (
                session.window_start_token < token_index <= session.output_tokens
                or acknowledged_boundary
            )
        ):
            raise AdaptiveDecodeError("adaptive tail seal is invalid")
        session.tail_seal_token = token_index
        session.tail_seal_reason = (
            reason if token_index < session.output_tokens else None
        )
        if acknowledged_boundary:
            session.window_start_token = None
            session.window_start_us = None
            session.target_token = None
            controller._state(session, "EXPLOITING")
            controller._seal_session(session)


def control_failed(
    controller,
    request_id: str,
    control: AdaptiveDecodeControl,
    reason: str,
    *,
    at_us: int,
) -> AdaptiveDecodeDirective:
    started_ns = time.perf_counter_ns()
    if type(reason) is not str or not reason or not reason.isascii():
        raise AdaptiveDecodeError("adaptive control failure is invalid")
    with controller._lock:
        session = controller._session(request_id)
        if session.awaiting_control != control:
            raise AdaptiveDecodeError("adaptive failed control differs")
        session.awaiting_control = None
        session.deferred_policy = None
        session.deferred_control_reason = None
        if control.policy.baseline:
            raise AdaptiveDecodeError("adaptive baseline recovery failed")
        coherence.server_policy_failed(controller, session, at_us, control.policy)
        controller._state(session, "RECOVERING")
        session.zero_assistance_reason = reason
        directive = controller._control(session, session.baseline)
        if session.operational_verification:
            session.operational_verification = False
            session.verification_budget = None
            directive = controller._verification_result(session, directive, "INCOMPLETE", reason)
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return directive


def defer_control(
    controller,
    request_id: str,
    control: AdaptiveDecodeControl,
    reason: str,
    *,
    at_us: int,
) -> AdaptiveDecodeDirective:
    """Continue the current window and retry a transient helper control."""

    started_ns = time.perf_counter_ns()
    if (
        type(reason) is not str
        or not reason
        or not reason.isascii()
        or type(at_us) is not int
        or at_us < 0
    ):
        raise AdaptiveDecodeError(
            "adaptive deferred control reason is invalid"
        )
    with controller._lock:
        session = controller._session(request_id)
        previous_policy = session.transition_policy
        previous_ack = session.transition_ack
        token_index = session.transition_start_token
        transition_started_at_us = session.transition_started_at_us
        if (
            session.awaiting_control != control
            or control.policy.baseline
            or previous_policy is None
            or token_index is None
            or transition_started_at_us is None
            or at_us < transition_started_at_us
        ):
            raise AdaptiveDecodeError(
                "adaptive deferred control differs"
            )
        session.awaiting_control = None
        session.transition_policy = None
        session.transition_ack = None
        session.transition_start_token = None
        session.transition_started_at_us = None
        if not session.operational_verification:
            session.probe_budget = None
            session.probe_policy_hash = None
        session.deferred_policy = control.policy
        session.deferred_control_reason = reason
        session.deferred_control_count += 1
        directive = controller._open_window(
            session,
            previous_policy,
            token_index,
            at_us,
            previous_ack,
        )
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return AdaptiveDecodeDirective(
            state=directive.state,
            reason="HELPER_CONTROL_DEFERRED",
            target_token_index=directive.target_token_index,
        )


def discard_stale_control(
    controller,
    request_id: str,
    control: AdaptiveDecodeControl,
    reason: str,
    *,
    at_us: int,
) -> AdaptiveDecodeDirective:
    """Discard a control after its request has released the bound slot."""
    started_ns = time.perf_counter_ns()
    if (
        type(reason) is not str
        or not reason
        or not reason.isascii()
        or type(at_us) is not int
        or at_us < 0
    ):
        raise AdaptiveDecodeError(
            "adaptive stale control reason is invalid"
        )
    with controller._lock:
        session = controller._session(request_id)
        previous_policy = session.transition_policy
        transition_token = session.transition_start_token
        if (
            session.awaiting_control != control
            or previous_policy is None
            or transition_token is None
            or not session.records
            or session.records[-1].token_end != transition_token
        ):
            raise AdaptiveDecodeError(
                "adaptive stale control transaction differs"
            )
        session.awaiting_control = None
        session.transition_policy = None
        session.transition_ack = None
        session.transition_start_token = None
        session.transition_started_at_us = None
        session.current_policy = previous_policy
        session.current_ack = None
        session.window_start_token = None
        session.window_start_us = None
        session.target_token = None
        session.tail_seal_token = transition_token
        session.tail_seal_reason = (
            reason if transition_token < session.output_tokens else None
        )
        session.tail_seal_pending_control = False
        controller._state(session, "EXPLOITING")
        directive = AdaptiveDecodeDirective(
            state=session.state,
            reason="STALE_SLOT_DIRECTIVE_DISCARDED",
            target_token_index=None,
        )
        controller._seal_session(session)
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return directive


def _validate_terminal(
    session: _AdaptiveSession,
    terminal_status: str,
    terminal_reason: str | None,
) -> None:
    if terminal_status not in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise AdaptiveDecodeError("adaptive terminal status is invalid")
    if terminal_reason is not None and (
        type(terminal_reason) is not str
        or not terminal_reason
        or not terminal_reason.isascii()
    ):
        raise AdaptiveDecodeError("adaptive terminal reason is invalid")
    if (
        session.awaiting_boundary is not None
        or session.awaiting_control is not None
        or session.transition_policy is not None
    ):
        raise AdaptiveDecodeError(
            "adaptive terminal transaction is pending"
        )
    if not session.records or session.current_policy is None:
        raise AdaptiveDecodeError(
            "adaptive terminal windows are incomplete"
        )


def _terminal_group(
    session: _AdaptiveSession,
    terminal_status: str,
    terminal_reason: str | None,
    state_history: tuple[str, ...],
) -> AdaptiveDecodeGroupedObservation:
    return AdaptiveDecodeGroupedObservation(
        request_id=session.request_id,
        ticket_id=session.ticket_id,
        model_artifact_sha256=session.model_artifact_sha256,
        planning_profile_sha256=session.planning_profile_sha256,
        desktop_placement_sha256=(
            session.baseline.desktop_placement_sha256
        ),
        windows=tuple(session.records),
        final_policy=session.current_policy,
        terminal_status=terminal_status,
        state_history=state_history,
        terminal_reason=terminal_reason,
        helper_layout_geometry_sha256=(
            session.history_helper_layout_geometry_sha256
        ),
        unmeasured_tail_tokens=(
            0
            if session.tail_seal_token is None
            else session.output_tokens - session.tail_seal_token
        ),
        unmeasured_tail_reason=session.tail_seal_reason,
        final_policy_ack=(
            session.current_ack
            if session.tail_seal_token is not None
            and session.tail_seal_token < session.output_tokens
            and not session.current_policy.baseline
            and session.current_ack is not None
            and not any(row.applied_ack == session.current_ack
                        for row in session.records)
            else None
        ),
    )


def preview_completion(
    controller,
    request_id: str,
    terminal_status: str = "COMPLETED",
    terminal_reason: str | None = None,
) -> AdaptiveDecodeGroupedObservation:
    """Build the immutable terminal proof without mutating lifecycle state."""
    with controller._lock:
        session = controller._terminal_session(request_id)
        controller._validate_terminal(
            session, terminal_status, terminal_reason
        )
        history = tuple(session.state_history)
        if history[-1] != "COMPLETED":
            history = (*history, "COMPLETED")
        return controller._terminal_group(
            session,
            terminal_status,
            terminal_reason,
            history,
        )


def complete(
    controller,
    request_id: str,
    terminal_status: str,
    terminal_reason: str | None = None,
) -> AdaptiveDecodeGroupedObservation:
    with controller._lock:
        session = controller._terminal_session(request_id)
        controller._validate_terminal(
            session, terminal_status, terminal_reason
        )
        controller._state(session, "COMPLETED")
        grouped = controller._terminal_group(
            session,
            terminal_status,
            terminal_reason,
            tuple(session.state_history),
        )
        session.grouped = grouped
        controller._completed[request_id] = grouped
        controller._history[grouped.grouped_observation_sha256] = grouped
        controller._history_component_bindings[
            grouped.grouped_observation_sha256
        ] = (
            session.component_capability_sha256,
            session.component_capability_sha256,
        )
        controller._sessions.pop(request_id, None)
        controller._sealed_sessions.pop(request_id, None)
        return grouped


def recover_attempt_for_restart(
    controller, request_id: str, ticket_id: str, reason: str
) -> AdaptiveDecodeGroupedObservation | None:
    """Close the adaptive session of one failed attempt (elastic phones), whatever its mode.

    ``recover_for_restart`` serves adaptive-split tickets; a desktop parent with a dormant FFN
    runtime or a helper envelope also opens a session, which a failed attempt must not leave
    open. Only the session of ``ticket_id`` is closed.
    """
    if type(ticket_id) is not str or not ticket_id:
        raise AdaptiveDecodeError("adaptive restart ticket is invalid")
    with controller._lock:
        session = controller._sessions.get(request_id) or controller._sealed_sessions.get(
            request_id
        )
        if session is None or session.ticket_id != ticket_id:
            return None
        return recover_for_restart(controller, request_id, reason)


def release_restarted_attempt(
    controller,
    request_id: str,
    stale_ticket_ids: tuple[str, ...],
    reason: str,
) -> dict[str, object] | None:
    """Free the registration an earlier, failed attempt of the same request left behind.

    Elastic phones: a request recovered after a helper_lost / server_exited failure starts a new
    adaptive session under a new ticket. A still-open session of a stale attempt is closed first
    (as ``recover_for_restart``); the attempt's terminal observation leaves ``completed`` (it stays
    in the history). A registration of any other ticket (the current attempt, or one not known to
    have failed) is kept, so a genuine duplicate start still fails. Returns the released row.
    """
    stale = frozenset(stale_ticket_ids)
    if not stale or any(type(row) is not str or not row for row in stale):
        raise AdaptiveDecodeError("adaptive restarted attempts are invalid")
    with controller._lock:
        session = controller._sessions.get(request_id) or controller._sealed_sessions.get(
            request_id
        )
        registration = None
        if session is not None:
            if session.ticket_id not in stale:
                return None
            registration = (
                "SEALED" if request_id in controller._sealed_sessions else "ACTIVE"
            )
            ticket_id = session.ticket_id
            recover_for_restart(controller, request_id, reason)
        completed = controller._completed.get(request_id)
        if completed is not None and completed.ticket_id in stale:
            del controller._completed[request_id]
            return {
                "grouped_observation_sha256": completed.grouped_observation_sha256,
                "reason": reason,
                "registration": registration or "COMPLETED",
                "request_id": request_id,
                "terminal_status": completed.terminal_status,
                "ticket_id": completed.ticket_id,
            }
        if registration is None:
            return None
        return {
            "grouped_observation_sha256": None,
            "reason": reason,
            "registration": registration,
            "request_id": request_id,
            "terminal_status": None,
            "ticket_id": ticket_id,
        }


def recover_for_restart(
    controller, request_id: str, reason: str
) -> AdaptiveDecodeGroupedObservation | None:
    """Close an interrupted adaptive attempt before desktop restart."""
    if type(reason) is not str or not reason or not reason.isascii():
        raise AdaptiveDecodeError("adaptive restart reason is invalid")
    with controller._lock:
        session = controller._sessions.get(request_id)
        if session is None:
            session = controller._sealed_sessions.get(request_id)
        if session is None:
            return None
        controller._state(session, "RECOVERING")
        controller._state(session, "BASELINE")
        final_policy = (
            session.current_policy
            or session.transition_policy
            or session.baseline
        )
        session.awaiting_boundary = None
        session.awaiting_control = None
        session.transition_policy = None
        session.transition_ack = None
        session.transition_start_token = None
        session.transition_started_at_us = None
        session.window_start_token = None
        session.window_start_us = None
        session.target_token = None
        if not session.records:
            controller._sessions.pop(request_id, None)
            controller._sealed_sessions.pop(request_id, None)
            return None
        controller._state(session, "COMPLETED")
        grouped = AdaptiveDecodeGroupedObservation(
            request_id=request_id,
            ticket_id=session.ticket_id,
            model_artifact_sha256=session.model_artifact_sha256,
            planning_profile_sha256=session.planning_profile_sha256,
            desktop_placement_sha256=(
                session.baseline.desktop_placement_sha256
            ),
            windows=tuple(session.records),
            final_policy=final_policy,
            terminal_status="FAILED",
            state_history=tuple(session.state_history),
            terminal_reason=reason,
            helper_layout_geometry_sha256=(
                session.history_helper_layout_geometry_sha256
            ),
        )
        controller._completed[request_id] = grouped
        controller._history[grouped.grouped_observation_sha256] = grouped
        controller._history_component_bindings[
            grouped.grouped_observation_sha256
        ] = (
            session.component_capability_sha256,
            session.component_capability_sha256,
        )
        controller._sessions.pop(request_id, None)
        controller._sealed_sessions.pop(request_id, None)
        return grouped
