"""AdaptiveDecodeController sequencing operations on its existing owner."""

from __future__ import annotations

from . import coherence

from dataclasses import replace

from ..adaptive_decode_contracts import (
    AdaptiveDecodeControl,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
)
from ..adaptive_decode_state import _AdaptiveSession
from .promotion import advance_comparable_measurements


def _state(session: _AdaptiveSession, state: str) -> None:
    if session.state != state:
        session.state = state
        session.state_history.append(state)


def _incomplete_probe(controller, session, token_index, at_us):
    policy = session.challenger_policy
    session.probe_retry_policy = (
        policy if policy is not None and session.probe_attempts.get(
            controller._probe_attempt_key(session, policy), 0
        ) < session.config.maximum_probe_attempts_per_context else None
    )
    session.probe_budget = None
    session.probe_policy_hash = None
    session.stage = "probe_retry"
    return controller._continue_best(session, token_index, at_us, "PROBE_INCOMPLETE")


def hold_unknown_context(controller, session, token_index, at_us):
    """Suspend optimization without inventing a new context or losing its evidence."""
    reason = "EXECUTION_CONTEXT_UNAVAILABLE"
    if session.operational_verification and session.verification_budget is not None:
        directive = controller._defer_verification(session, token_index, at_us, reason)
    elif session.state == "PROBING" and session.probe_budget is not None:
        directive = controller._incomplete_probe(session, token_index, at_us)
    else:
        directive = controller._continue_best(session, token_index, at_us, reason)
    session.zero_assistance_reason = reason
    return replace(directive, reason=reason)


def _reserve_verification_pair(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
) -> bool:
    policy = session.verification_policy
    if policy is None or session.verification_attempts >= session.config.maximum_probe_attempts_per_context:
        return False
    budget = controller._measurement_pair_budget(session, policy, token_index, at_us)
    if budget is None:
        return False
    session.verification_attempts += 1
    session.verification_record_start = budget["record_start"]
    session.verification_budget = budget
    session.challenger_policy = policy
    fresh_baseline = budget["fresh_baseline"]
    session.stage = "verification_candidate" if fresh_baseline else "cached_candidate"
    return True


def _verification_result(
    session: _AdaptiveSession, directive: AdaptiveDecodeDirective,
    outcome: str, reason: str,
) -> AdaptiveDecodeDirective:
    changed = (session.verification_outcome, session.verification_reason) != (outcome, reason)
    session.verification_outcome, session.verification_reason = outcome, reason
    return replace(directive, reason="VERIFICATION_" + outcome) if changed else directive


def _defer_verification(
    controller, session: _AdaptiveSession, token_index: int, at_us: int, reason: str,
) -> AdaptiveDecodeDirective:
    session.verification_budget = None
    session.stage = "verification_pending"
    directive = controller._continue_best(session, token_index, at_us, "VERIFICATION_INCOMPLETE")
    return controller._verification_result(session, directive, "INCOMPLETE", reason)


def _prior_monitor_eligible(
    controller, session: _AdaptiveSession, token_index: int,
) -> bool:
    """A seeded operational winner may run under monitoring when the request
    cannot afford its complete paired verification but can still measure
    the winner itself (one warm-up window plus at least one measured window).
    """
    policy = session.verification_policy
    if (
        policy is None
        or policy.baseline
        or not session.operational_verification
        or not session.helper_available
        or not session.execution_context_available
        or policy not in session.candidates
        or policy.policy_hash in session.eliminated_policy_reasons
        or session.context_monitor_prior is not None
    ):
        return False
    warmups = max(
        0,
        session.config.warmup_windows_per_policy
        - session.warmup_windows_seen_by_policy.get(policy.policy_hash, 0),
    )
    needed = (warmups + 1) * session.config.minimum_window_tokens
    return controller._remaining_tokens(session, token_index) >= needed


def _begin_prior_monitor(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
) -> AdaptiveDecodeDirective:
    """Exploit the compatible prior winner as a starting point under
    per-window monitoring; the request pays no verification windows at 0%
    and the prior is never promoted to incumbent by this path."""
    policy = session.verification_policy
    session.operational_verification = False
    session.verification_budget = None
    session.prior_monitor_policy = policy
    session.prior_monitor_windows = 0
    session.stage = "prior_monitor"
    controller._state(session, "EXPLOITING")
    directive = controller._control(session, policy, token_index, at_us)
    return controller._verification_result(
        session, directive, "MONITORING", "PRIOR_UNDER_MONITOR"
    )


def _prior_monitor_holds(controller, session: _AdaptiveSession) -> bool:
    """The prior keeps running only while its own measured windows in this
    request beat the historical baseline by the configured margin and stay
    inside the latency limit; unmeasurable evidence never counts as a pass."""
    policy = session.prior_monitor_policy
    baseline = controller._bounds(session, session.baseline, operational=True)
    rows = controller._current_valid_records(session, policy, operational=True)
    if baseline is None or not rows:
        return False
    energy_tokens = sum(row.energy_token_count for row in rows)
    if energy_tokens <= 0:
        return False
    energy_mean = max(1, sum(row.whole_fleet_energy_uj for row in rows) // energy_tokens)
    from math import isqrt
    uncertainty = max(
        20_000, session.config.uncertainty_ppm // max(1, isqrt(len(rows)))
    )
    energy_upper = (energy_mean * (1_000_000 + uncertainty) + 999_999) // 1_000_000
    required_upper = (
        baseline[1] * (1_000_000 - session.config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    latency = controller._current_latency_bounds(session, policy, minimum_count=1)
    latency_limit = (
        baseline[4] * session.config.maximum_latency_ppm + 999_999
    ) // 1_000_000
    return (
        energy_upper <= required_upper
        and latency is not None
        and latency[1] <= latency_limit
    )


def _end_prior_monitor(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
    reason: str, *, rejected: bool,
) -> AdaptiveDecodeDirective:
    policy = session.prior_monitor_policy
    session.prior_monitor_policy = None
    session.stage = "initial_baseline"
    if rejected and policy is not None:
        session.eliminated_policy_reasons[policy.policy_hash] = "PRIOR_MONITOR_NOT_IMPROVED"
    directive = controller._continue_best(session, token_index, at_us, reason)
    return controller._verification_result(
        session, directive, "REJECTED" if rejected else "INCOMPLETE", reason
    )


def _advance_prior_monitor(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
) -> AdaptiveDecodeDirective:
    policy = session.prior_monitor_policy
    if (
        policy is None
        or policy not in session.candidates
        or policy.policy_hash in session.eliminated_policy_reasons
    ):
        return controller._end_prior_monitor(
            session, token_index, at_us, "PRIOR_MONITOR_ENDED", rejected=False
        )
    latest = session.records[-1]
    if latest.policy != policy:
        # The prior is not yet the running policy (control pending or a
        # baseline window closed first): re-issue it once.
        if session.current_policy == policy:
            return controller._open_window(
                session, policy, token_index, at_us, session.current_ack
            )
        return controller._control(session, policy, token_index, at_us)
    if not latest.measurement_eligible:
        return controller._open_window(
            session, policy, token_index, at_us, session.current_ack
        )
    if controller._prior_monitor_holds(session):
        session.prior_monitor_windows += 1
        return controller._open_window(
            session, policy, token_index, at_us, session.current_ack
        )
    return controller._end_prior_monitor(
        session, token_index, at_us, "PRIOR_MONITOR_REJECTED", rejected=True
    )


def _advance_operational_verification(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
) -> AdaptiveDecodeDirective:
    if session.verification_budget is None:
        if not controller._reserve_verification_pair(session, token_index, at_us):
            retry_limit = (
                session.verification_attempts
                >= session.config.maximum_probe_attempts_per_context
            )
            if not retry_limit and controller._prior_monitor_eligible(
                session, token_index
            ):
                return controller._begin_prior_monitor(session, token_index, at_us)
            return controller._defer_verification(
                session, token_index, at_us,
                "RETRY_LIMIT" if retry_limit else "COMPLETE_PAIR_BUDGET")
        controller._state(session, "PREPARING")
        controller._state(session, "PROBING")
        return controller._verification_result(
            session, controller._control(session, session.verification_policy, token_index, at_us),
            "RESERVED", "COMPLETE_PAIR")
    budget = session.verification_budget
    if at_us > budget["deadline_us"] or token_index > budget["token_limit"]:
        return controller._defer_verification(session, token_index, at_us, "RESERVATION_EXHAUSTED")
    latest = session.records[-1]
    if not latest.measurement_eligible:
        return controller._open_window(session, latest.policy, token_index, at_us, session.current_ack)
    if session.stage == "cached_candidate":
        session.stage = "cached_baseline"
        return controller._control(session, session.baseline, token_index, at_us)
    return controller._finish_verification(session, session.verification_policy, token_index, at_us)


def _open_window(
    _controller_class,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int,
    at_us: int,
    ack: AdaptiveDecodePolicyAck | None,
) -> AdaptiveDecodeDirective:
    session.current_policy = policy
    session.current_ack = ack
    session.window_start_token = token_index
    session.window_start_us = at_us
    session.window_role = ("exploitation" if session.state == "EXPLOITING"
                           and session.stage != "server_probe" else "exploration")
    window_tokens = _controller_class._window_tokens(
        session, policy, token_index
    )
    session.target_token = min(
        session.output_tokens, token_index + window_tokens
    )
    return AdaptiveDecodeDirective(
        state=session.state,
        reason="WINDOW_OPENED",
        target_token_index=session.target_token,
    )


def _control(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int | None = None,
    at_us: int | None = None,
) -> AdaptiveDecodeDirective:
    if not policy.baseline and not session.helper_available:
        policy = session.baseline
        session.zero_assistance_reason = "PHONE_HELPER_UNAVAILABLE"
        controller._state(session, "RECOVERING")
    if (not policy.baseline and not session.execution_context_available
            and policy.policy_hash not in session.maintenance_policy_hashes):
        policy = session.baseline
        session.zero_assistance_reason = "EXECUTION_CONTEXT_UNAVAILABLE"
        controller._state(session, "RECOVERING")
    if (not policy.baseline and session.state == "PROBING"
            and not session.operational_verification
            and token_index is not None and at_us is not None
            and not controller._probe_admitted(session, policy, token_index, at_us)):
        budget = controller._measurement_pair_budget(session, policy, token_index, at_us)
        if budget is None:
            session.probe_budget = None
            session.probe_policy_hash = None
            session.deferred_policy = None
            session.deferred_control_reason = None
            return controller._continue_best(session, token_index, at_us, "INSUFFICIENT_OPPORTUNITY")
        session.probe_budget = budget
        session.probe_policy_hash = policy.policy_hash
        session.challenger_policy = policy
    if session.transition_policy is None:
        if (
            session.current_policy is None
            or type(token_index) is not int
            or type(at_us) is not int
        ):
            raise AdaptiveDecodeError(
                "adaptive control transition is incomplete"
            )
        session.transition_policy = session.current_policy
        session.transition_window_role = session.window_role
        session.transition_ack = (
            session.current_ack
            if session.current_ack is not None
            and session.current_ack.applied_token_index == token_index
            else None
        )
        session.transition_start_token = token_index
        session.transition_started_at_us = at_us
    elif token_index is not None or at_us is not None:
        raise AdaptiveDecodeError(
            "adaptive control transition is already pending"
        )
    session.plan_generation += 1
    control = AdaptiveDecodeControl(
        request_id=session.request_id,
        slot_id=session.slot_id,
        plan_generation=session.plan_generation,
        policy=policy,
    )
    session.awaiting_control = control
    coherence.publish_server_policy(controller, session, policy, at_us or session.transition_started_at_us or 0)
    session.current_policy = None
    session.current_ack = None
    session.window_start_token = None
    session.window_start_us = None
    session.target_token = None
    return AdaptiveDecodeDirective(
        state=session.state,
        reason="POLICY_CHANGE_REQUIRED",
        target_token_index=None,
        control=control,
    )


def _initial_start_directive(
    controller,
    session: _AdaptiveSession,
    baseline: AdaptiveDecodePolicy,
    first_token_index: int,
    first_token_at_us: int,
) -> AdaptiveDecodeDirective:
    directive = coherence.server_directive(controller, session, first_token_index, first_token_at_us)
    if directive is not None:
        return directive
    if not session.execution_context_available:
        session.zero_assistance_reason = "EXECUTION_CONTEXT_UNAVAILABLE"
        controller._state(session, "EXPLOITING")
        return controller._open_window(session, baseline, first_token_index, first_token_at_us, None)
    if session.helper_available and session.cached_winner is None:
        session.cached_winner = controller._coherent_phone_policy(session)
    if session.helper_available and session.cached_winner is not None:
        session.incumbent_policy = session.cached_winner
        session.incumbent_context_sha256 = controller._context_identity(session)
        controller._state(session, "EXPLOITING")
        session.current_policy = baseline
        return controller._control(
            session,
            session.cached_winner,
            first_token_index,
            first_token_at_us,
        )
    if session.helper_available and session.operational_verification:
        session.current_policy = baseline
        return controller._advance_operational_verification(
            session, first_token_index, first_token_at_us)
    if (
        session.helper_available
        and session.probe_candidates
        and controller._can_probe(
            session, first_token_index, first_token_at_us
        )
    ):
        controller._state(session, "PREPARING")
        controller._state(session, "PROBING")
        if (
            session.helper_evidence_state == "LEARNING"
            and controller._valid_records(session, baseline, operational=True)
        ):
            policy = session.probe_candidates[0]
            session.next_probe_index = 1
            session.stage = "candidate"
            session.current_policy = baseline
            return controller._control(
                session,
                policy,
                first_token_index,
                first_token_at_us,
            )
    elif session.probe_candidates and not session.helper_available:
        controller._state(session, "PREPARING")
    else:
        controller._state(session, "EXPLOITING")
        if (
            controller._remaining_tokens(session, first_token_index)
            <= session.config.minimum_window_tokens
        ):
            return controller._open_window(
                session,
                baseline,
                first_token_index,
                first_token_at_us,
                None,
            )
        preferred = (
            session.cached_winner
            or controller._ticket_fallback(session)
        )
        if preferred is not None:
            session.current_policy = baseline
            return controller._control(
                session,
                preferred,
                first_token_index,
                first_token_at_us,
            )
    directive = controller._open_window(
        session,
        baseline,
        first_token_index,
        first_token_at_us,
        None,
    )
    if (session.state == "EXPLOITING" and session.helper_available
            and session.probe_candidates):
        session.zero_assistance_reason = "INSUFFICIENT_OPPORTUNITY"
        return replace(directive, reason="INSUFFICIENT_OPPORTUNITY")
    return directive


def _advance_probe_candidate(
    controller, session: _AdaptiveSession, token_index: int, at_us: int,
) -> AdaptiveDecodeDirective:
    while session.next_probe_index < len(session.probe_candidates):
        policy = session.probe_candidates[session.next_probe_index]
        session.next_probe_index += 1
        if (policy.policy_hash not in session.eliminated_policy_reasons
                and controller._measurement_pair_budget(session, policy, token_index, at_us) is not None):
            return controller._control(session, policy, token_index, at_us)
    return controller._select_probe_winner(session, token_index, at_us)


def take_pending_policy(session):
    """Maintenance always precedes optimization; issued controls are immutable."""
    if session.pending_session_drain_policy is not None:
        policy = session.pending_session_drain_policy
        session.pending_session_drain_policy = None
        return policy, True
    policy = session.pending_helper_refresh_policy
    session.pending_helper_refresh_policy = None
    return policy, False


def _next_after_window(
    controller, session: _AdaptiveSession, token_index: int, at_us: int
) -> AdaptiveDecodeDirective:
    if session.tail_seal_token is not None:
        if token_index != session.tail_seal_token:
            raise AdaptiveDecodeError(
                "adaptive tail seal boundary differs"
            )
        if session.tail_seal_reason == "cohort_membership_changed":
            if (
                session.current_policy is not None
                and not session.current_policy.baseline
            ):
                controller._state(session, "RECOVERING")
                session.tail_seal_pending_control = True
                return controller._control(
                    session, session.baseline, token_index, at_us
                )
            controller._state(session, "EXPLOITING")
            return AdaptiveDecodeDirective(
                state=session.state,
                reason="COHORT_MEMBERSHIP_CHANGED",
                target_token_index=None,
            )
        if session.state != "EXPLOITING" and (
            session.current_policy is not None
            and not session.current_policy.baseline
        ):
            controller._state(session, "RECOVERING")
            session.tail_seal_pending_control = True
            return controller._control(
                session, session.baseline, token_index, at_us
            )
        controller._state(session, "EXPLOITING")
        return AdaptiveDecodeDirective(
            state=session.state,
            reason="TAIL_SEALED",
            target_token_index=None,
        )
    if token_index >= session.output_tokens:
        return AdaptiveDecodeDirective(
            state=session.state,
            reason="TERMINAL_WINDOW_RECORDED",
            target_token_index=None,
        )
    if not session.helper_available:
        controller._state(session, "PREPARING")
        session.zero_assistance_reason = "PHONE_HELPER_UNAVAILABLE"
        if (
            session.current_policy is not None
            and not session.current_policy.baseline
        ):
            controller._state(session, "RECOVERING")
            return controller._control(
                session, session.baseline, token_index, at_us
            )
        return controller._open_window(
            session,
            session.baseline,
            token_index,
            at_us,
            session.current_ack,
        )
    pending_drain, maintenance = take_pending_policy(session)
    if pending_drain is not None:
        if maintenance:
            controller._state(session, "EXPLOITING")
        if session.current_policy == pending_drain:
            return controller._open_window(
                session,
                pending_drain,
                token_index,
                at_us,
                session.current_ack,
            )
        return controller._control(
            session, pending_drain, token_index, at_us
        )
    if not session.execution_context_available:
        return hold_unknown_context(controller, session, token_index, at_us)
    directive = coherence.server_directive(controller, session, token_index, at_us)
    if directive is not None:
        return directive
    if session.deferred_policy is not None:
        if session.deferred_policy not in session.candidates:
            session.deferred_policy = None
            session.deferred_control_reason = None
            return controller._continue_best(
                session, token_index, at_us, "DEFERRED_POLICY_NOT_CURRENT"
            )
        budget = session.verification_budget
        if session.operational_verification and budget is not None and (
            at_us > budget["deadline_us"] or token_index > budget["token_limit"]
        ):
            session.deferred_policy = None
            session.deferred_control_reason = None
            return controller._defer_verification(session, token_index, at_us, "RESERVATION_EXHAUSTED")
        if (
            session.deferred_policy.policy_hash
                in session.eliminated_policy_reasons
            or controller._remaining_tokens(session, token_index)
                < session.config.minimum_window_tokens
        ):
            session.deferred_policy = None
            session.deferred_control_reason = None
        else:
            if (not session.operational_verification
                    and session.deferred_policy == session.challenger_policy
                    and session.deferred_policy != session.incumbent_policy):
                controller._state(session, "PROBING")
                session.stage = "candidate"
            return controller._control(
                session,
                session.deferred_policy,
                token_index,
                at_us,
            )
    if session.stage == "prior_monitor":
        return controller._advance_prior_monitor(session, token_index, at_us)
    if session.stage == "resolving_evidence" and session.state == "PROBING":
        return advance_comparable_measurements(controller, session, token_index, at_us)
    if session.operational_verification:
        return controller._advance_operational_verification(session, token_index, at_us)
    if session.stage == "probe_retry":
        retry = session.probe_retry_policy
        if (retry is not None and retry in session.candidates
                and retry.policy_hash not in session.eliminated_policy_reasons
                and controller._measurement_pair_budget(session, retry, token_index, at_us) is not None):
            session.probe_retry_policy = None
            session.stage = "candidate"
            controller._state(session, "PROBING")
            return controller._control(session, retry, token_index, at_us)
        return controller._continue_best(session, token_index, at_us, "PROBE_INCOMPLETE")
    latest = session.records[-1]
    if (session.state == "PROBING" and not latest.policy.baseline
            and not latest.measurement_eligible
            and latest.policy.policy_hash in session.eliminated_policy_reasons):
        if session.verification_policy == latest.policy:
            session.verification_policy = None
        return replace(controller._advance_probe_candidate(session, token_index, at_us),
                       reason="PROBE_CANDIDATE_REJECTED")
    if (
        session.state == "PROBING"
        and latest.policy == session.current_policy
        and latest.failure_reason is None
        and not latest.measurement_eligible
    ):
        if (not latest.policy.baseline
                and not controller._probe_admitted(session, latest.policy, token_index, at_us)):
            return controller._incomplete_probe(session, token_index, at_us)
        return controller._open_window(
            session,
            latest.policy,
            token_index,
            at_us,
            session.current_ack,
        )
    if (session.state == "EXPLOITING" and session.stage == "initial_baseline"
            and latest.policy.baseline and latest.measurement_eligible
            and controller._can_probe(session, token_index, at_us)):
        controller._state(session, "PROBING")
    if session.state in {"BASELINE", "EXPLOITING"}:
        if (session.incumbent_policy is not None and session.current_policy == session.incumbent_policy
                and not controller._qualifies(session, session.incumbent_policy, token_index, at_us)):
            return controller._continue_best(session, token_index, at_us, "INCUMBENT_NO_LONGER_BENEFICIAL")
        return controller._open_window(
            session,
            session.current_policy or session.baseline,
            token_index,
            at_us,
            session.current_ack,
        )
    if (not controller._can_probe(session, token_index, at_us)
            and session.stage == "initial_baseline"):
        return controller._continue_best(session, token_index, at_us, "INSUFFICIENT_OPPORTUNITY")

    if session.stage == "initial_baseline":
        policy = session.probe_candidates[0]
        session.next_probe_index = 1
        session.stage = "candidate"
        return controller._control(session, policy, token_index, at_us)
    if session.stage == "cached_candidate":
        session.stage = "cached_baseline"
        return controller._control(session, session.baseline, token_index, at_us)
    if session.stage == "cached_baseline":
        return controller._finish_verification(
            session, session.verification_policy, token_index, at_us
        )
    if session.stage == "candidate":
        policy = session.current_policy
        if policy is not None and not policy.baseline:
            if not controller._current_valid_records(session, policy, operational=True):
                return controller._incomplete_probe(session, token_index, at_us)
            if not controller._current_valid_records(session, session.baseline, operational=True):
                session.verification_policy = policy
                session.stage = "verification_baseline"
                session.zero_assistance_reason = "VERIFICATION"
                return controller._control(session, session.baseline, token_index, at_us)
        if (
            session.helper_evidence_state == "LEARNING"
            and session.current_policy is not None
            and not session.current_policy.baseline
            and not controller._learning_probe_improves(
                session, session.current_policy
            )
        ):
            session.eliminated_policy_reasons[
                session.current_policy.policy_hash
            ] = "LEARNING_NO_PAIRED_IMPROVEMENT"
            return replace(controller._advance_probe_candidate(session, token_index, at_us),
                           reason="PROBE_CANDIDATE_REJECTED")
        return controller._advance_probe_candidate(session, token_index, at_us)
    if session.stage == "verification":
        leader = session.current_policy
        if leader is None or leader.baseline:
            raise AdaptiveDecodeError("adaptive verification policy differs")
        return controller._finish_verification(
            session, leader, token_index, at_us
        )
    if session.stage == "verification_baseline":
        policy = session.verification_policy
        if (
            policy is None
            or session.current_policy is None
            or not session.current_policy.baseline
        ):
            raise AdaptiveDecodeError(
                "adaptive verification baseline differs"
            )
        if controller._qualifies(session, policy, token_index, at_us):
            return controller._finish_verification(session, policy, token_index, at_us)
        session.stage = "verification_candidate"
        return controller._control(session, policy, token_index, at_us)
    if session.stage == "verification_candidate":
        policy = session.verification_policy
        if policy is None or session.current_policy != policy:
            raise AdaptiveDecodeError(
                "adaptive verification candidate differs"
            )
        return controller._finish_verification(
            session, policy, token_index, at_us
        )
    raise AdaptiveDecodeError("adaptive probe stage is invalid")
