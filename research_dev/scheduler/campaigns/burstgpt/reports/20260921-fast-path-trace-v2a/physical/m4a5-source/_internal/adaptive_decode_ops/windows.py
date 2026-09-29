"""AdaptiveDecodeController windows operations on its existing owner."""

from __future__ import annotations

from . import coherence

from dataclasses import replace
import time

from ..adaptive_decode_contracts import (
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowBoundary,
    AdaptiveDecodeWindowReceipt,
)
from ..adaptive_decode_state import _AdaptiveSession
from .budgeting import policy_evidence_key, pending_probe_measurement_us
from .sequencing import hold_unknown_context, take_pending_policy


def boundary(
    controller,
    request_id: str,
    *,
    slot_id: int,
    token_index: int,
    at_us: int,
    terminal: bool = False,
) -> AdaptiveDecodeDirective | None:
    started_ns = time.perf_counter_ns()
    with controller._lock:
        session = controller._session(request_id)
        if slot_id != session.slot_id:
            raise AdaptiveDecodeError("adaptive slot identity differs")
        if session.awaiting_control is not None:
            return None
        if session.awaiting_boundary is not None:
            return AdaptiveDecodeDirective(
                state=session.state,
                reason="WINDOW_MEASUREMENT_PENDING",
                target_token_index=None,
                boundary=session.awaiting_boundary,
            )
        if (
            session.current_policy is None
            or session.window_start_token is None
            or session.window_start_us is None
            or session.target_token is None
        ):
            raise AdaptiveDecodeError("adaptive window is not open")
        if (
            session.current_ack is not None
            and token_index <= session.current_ack.applied_token_index
        ):
            return None
        if token_index < session.window_start_token or at_us <= session.window_start_us:
            raise AdaptiveDecodeError("adaptive progress moved backward")
        if (
            not terminal
            and token_index < session.target_token
            and session.pending_session_drain_policy is None
            and session.pending_helper_refresh_policy is None
            and not coherence.server_policy_pending(controller, session)
        ):
            return None
        if token_index <= session.window_start_token:
            return None
        boundary = AdaptiveDecodeWindowBoundary(
            request_id=request_id,
            slot_id=slot_id,
            window_index=len(session.records),
            token_start=session.window_start_token,
            token_end=min(token_index, session.output_tokens),
            started_at_us=session.window_start_us,
            finished_at_us=at_us,
            policy=session.current_policy,
            applied_ack=session.current_ack,
            window_role=session.window_role,
        )
        session.awaiting_boundary = boundary
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return AdaptiveDecodeDirective(
            state=session.state,
            reason="WINDOW_MEASUREMENT_REQUIRED",
            target_token_index=None,
            boundary=boundary,
        )


def acknowledge(
    controller,
    request_id: str,
    acknowledgement: AdaptiveDecodePolicyAck,
    transition_observation: AdaptiveDecodeRawWindowObservation | None = None,
) -> AdaptiveDecodeDirective:
    started_ns = time.perf_counter_ns()
    with controller._lock:
        session = controller._session(request_id)
        control = session.awaiting_control
        if control is None:
            raise AdaptiveDecodeError("adaptive control is not pending")
        if (
            not isinstance(acknowledgement, AdaptiveDecodePolicyAck)
            or acknowledgement.request_id != request_id
            or acknowledgement.slot_id != session.slot_id
            or acknowledgement.plan_generation != control.plan_generation
            or acknowledgement.policy_hash != control.policy.policy_hash
        ):
            raise AdaptiveDecodeError("adaptive acknowledgement differs")
        transition_policy = session.transition_policy
        transition_start_token = session.transition_start_token
        transition_started_at_us = session.transition_started_at_us
        if (
            transition_policy is None
            or transition_start_token is None
            or transition_started_at_us is None
            or acknowledgement.applied_token_index < transition_start_token
            or acknowledgement.applied_token_index > session.output_tokens
        ):
            raise AdaptiveDecodeError(
                "adaptive control transition differs"
            )
        transition_tokens = (
            acknowledgement.applied_token_index - transition_start_token
        )
        if transition_tokens and transition_observation is None:
            raise AdaptiveDecodeError(
                "adaptive transition observation is absent"
            )
        if not transition_tokens and transition_observation is not None:
            raise AdaptiveDecodeError(
                "adaptive transition observation is unexpected"
            )
        if transition_tokens:
            if (
                acknowledgement.applied_at_us <= transition_started_at_us
                or not isinstance(
                    transition_observation,
                    AdaptiveDecodeRawWindowObservation,
                )
            ):
                raise AdaptiveDecodeError(
                    "adaptive transition observation differs"
                )
            boundary = AdaptiveDecodeWindowBoundary(
                request_id=request_id,
                slot_id=session.slot_id,
                window_index=len(session.records),
                token_start=transition_start_token,
                token_end=acknowledgement.applied_token_index,
                started_at_us=transition_started_at_us,
                finished_at_us=acknowledgement.applied_at_us,
                policy=transition_policy,
                applied_ack=session.transition_ack,
                window_role=session.transition_window_role,
            )
            controller._append_window(
                session,
                boundary,
                transition_observation,
                stable_window=False,
            )
            if boundary.window_role == "exploration" and not transition_policy.baseline:
                session.probe_tokens += transition_tokens
        session.acknowledged_policy = control.policy
        session.observed_control_cost_us = max(
            session.observed_control_cost_us,
            acknowledgement.applied_at_us - transition_started_at_us,
        )
        session.observed_control_tokens = max(session.observed_control_tokens, transition_tokens)
        budget = session.verification_budget if session.operational_verification else session.probe_budget
        if (budget is not None and not control.policy.baseline and not budget.get("acknowledged")):
            budget["acknowledged"] = 1
            key = controller._probe_attempt_key(session, control.policy)
            session.probe_attempts[key] = session.probe_attempts.get(key, 0) + 1
        session.awaiting_control = None
        session.transition_policy = None
        session.transition_ack = None
        session.transition_start_token = None
        session.transition_started_at_us = None
        if (
            session.deferred_policy is not None
            and session.deferred_policy.policy_hash
                == control.policy.policy_hash
        ):
            session.deferred_policy = None
            session.deferred_control_reason = None
        if session.state == "RECOVERING":
            controller._state(session, "BASELINE")
        if session.tail_seal_pending_control:
            session.tail_seal_pending_control = False
            session.tail_seal_token = acknowledgement.applied_token_index
            if session.tail_seal_token == session.output_tokens:
                session.tail_seal_reason = None
            session.current_policy = control.policy
            session.current_ack = acknowledgement
            session.window_start_token = None
            session.window_start_us = None
            session.target_token = None
            controller._state(session, "EXPLOITING")
            directive = AdaptiveDecodeDirective(
                state=session.state,
                reason="TAIL_SEALED",
                target_token_index=None,
            )
            controller._seal_session(session)
            session.decision_time_us.append(
                (time.perf_counter_ns() - started_ns) // 1000
            )
            return directive
        if not session.helper_available:
            session.pending_session_drain_policy = None
            session.pending_helper_refresh_policy = None
            session.deferred_policy = None
            session.deferred_control_reason = None
            session.zero_assistance_reason = "PHONE_HELPER_UNAVAILABLE"
            if not control.policy.baseline:
                session.current_policy = control.policy
                session.current_ack = acknowledgement
                controller._state(session, "RECOVERING")
                directive = controller._control(
                    session, session.baseline,
                    acknowledgement.applied_token_index,
                    acknowledgement.applied_at_us,
                )
                session.decision_time_us.append(
                    (time.perf_counter_ns() - started_ns) // 1000
                )
                return directive
            controller._state(session, "PREPARING")
        pending_drain, maintenance = take_pending_policy(session)
        if pending_drain is not None:
            session.current_policy = control.policy
            session.current_ack = acknowledgement
            if maintenance:
                controller._state(session, "EXPLOITING")
            directive = controller._control(
                session,
                pending_drain,
                acknowledgement.applied_token_index,
                acknowledgement.applied_at_us,
            )
            session.decision_time_us.append(
                (time.perf_counter_ns() - started_ns) // 1000
            )
            return directive
        if not session.execution_context_available:
            session.current_policy = control.policy
            session.current_ack = acknowledgement
            return hold_unknown_context(
                controller, session, acknowledgement.applied_token_index, acknowledgement.applied_at_us
            )
        if session.config.server_policy_coherence:
            session.current_policy = control.policy
            session.current_ack = acknowledgement
            coherence.server_policy_acknowledged(controller, session, acknowledgement.applied_at_us)
            shared = coherence.server_directive(
                controller, session, acknowledgement.applied_token_index, acknowledgement.applied_at_us)
            if shared is not None:
                session.decision_time_us.append((time.perf_counter_ns() - started_ns) // 1000)
                return shared
        if (session.state == "PROBING" and budget is not None and not control.policy.baseline
                and acknowledgement.applied_token_index + session.config.minimum_window_tokens < session.output_tokens
                and (acknowledgement.applied_token_index >= budget["token_limit"]
                     or acknowledgement.applied_at_us + pending_probe_measurement_us(
                         controller, session, control.policy, acknowledgement.applied_token_index, budget
                     ) > budget["deadline_us"])):
            session.current_policy = control.policy
            session.current_ack = acknowledgement
            return (controller._defer_verification(
                session, acknowledgement.applied_token_index, acknowledgement.applied_at_us,
                "ACK_LEFT_INSUFFICIENT_MEASUREMENT_TIME",
            ) if session.operational_verification else controller._incomplete_probe(
                session, acknowledgement.applied_token_index, acknowledgement.applied_at_us))
        directive = controller._open_window(
            session,
            control.policy,
            acknowledgement.applied_token_index,
            acknowledgement.applied_at_us,
            acknowledgement,
        )
        coherence.server_policy_acknowledged(controller, session, acknowledgement.applied_at_us)
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return directive


def _append_window(
    controller,
    session: _AdaptiveSession,
    boundary: AdaptiveDecodeWindowBoundary,
    observation: AdaptiveDecodeRawWindowObservation,
    *,
    stable_window: bool = True,
    external_activity_changed: bool = False,
) -> AdaptiveDecodeWindowReceipt:
    stable_window = stable_window and observation.execution_context_available
    policy_hash = boundary.policy.policy_hash
    warmup_seen = session.warmup_windows_seen_by_policy.get(
        policy_hash, 0
    )
    measurement_eligible = (
        stable_window
        and policy_hash not in session.maintenance_policy_hashes
        and warmup_seen >= session.config.warmup_windows_per_policy
    )
    if (
        stable_window
        and policy_hash not in session.maintenance_policy_hashes
    ):
        if warmup_seen < session.config.warmup_windows_per_policy:
            session.warmup_latency_us_by_policy[policy_hash] = max(
                session.warmup_latency_us_by_policy.get(policy_hash, 0),
                (boundary.finished_at_us - boundary.started_at_us + boundary.token_count - 1)
                // boundary.token_count,
            )
        session.warmup_windows_seen_by_policy[policy_hash] = (
            warmup_seen + 1
        )
    previous = (
        "0" * 64
        if not session.records
        else session.records[-1].record_sha256.removeprefix("sha256:")
    )
    receipt = AdaptiveDecodeWindowReceipt(
        request_id=session.request_id,
        slot_id=session.slot_id,
        window_index=len(session.records),
        token_start=boundary.token_start,
        token_end=boundary.token_end,
        context_length=session.context_length + boundary.token_start,
        active_batch=(
            session.active_batch
            if observation.active_batch is None
            else observation.active_batch
        ),
        started_at_us=boundary.started_at_us,
        finished_at_us=boundary.finished_at_us,
        policy=boundary.policy,
        applied_ack=boundary.applied_ack,
        fleet_energy_uj_by_domain=observation.fleet_energy_uj_by_domain,
        latency_per_token_us=max(
            1,
            (boundary.finished_at_us - boundary.started_at_us)
            // boundary.token_count,
        ),
        phone_compute_us=observation.phone_compute_us,
        usb_transfer_us=observation.usb_transfer_us,
        rpc_us=observation.rpc_us,
        exposed_tail_us=observation.exposed_tail_us,
        output_valid=observation.output_valid,
        evidence_ids=observation.evidence_ids,
        energy_boundary_id=observation.energy_boundary_id,
        energy_attribution_kind=observation.energy_attribution_kind,
        failure_reason=observation.failure_reason,
        previous_record_sha256=previous,
        measurement_eligible=measurement_eligible,
        usb_upload_bytes=observation.usb_upload_bytes,
        usb_download_bytes=observation.usb_download_bytes,
        desktop_compute_us=observation.desktop_compute_us,
        useful_overlap_us=observation.useful_overlap_us,
        request_queue_delay_us=observation.request_queue_delay_us,
        protected_interference_us=(
            observation.protected_interference_us
        ),
        usb_h2d_us=observation.usb_h2d_us,
        usb_d2h_us=observation.usb_d2h_us,
        accounting_token_count=observation.accounting_token_count,
        cohort_id=observation.cohort_id,
        cohort_member_request_ids=(
            observation.cohort_member_request_ids
        ),
        energy_owner_request_id=observation.energy_owner_request_id,
        configured_queue_depth=observation.configured_queue_depth,
        maximum_active_slots=observation.maximum_active_slots,
        maximum_outstanding_transfers=(
            observation.maximum_outstanding_transfers
        ),
        batched_calls=observation.batched_calls,
        transfer_subrequests=observation.transfer_subrequests,
        maximum_tokens=observation.maximum_tokens,
        completed_phone_calls=observation.completed_phone_calls,
        completed_phone_input_rows=(
            observation.completed_phone_input_rows
        ),
        next_active_batch=observation.next_active_batch,
        membership_changed=observation.membership_changed,
        execution_context_available=observation.execution_context_available,
        external_activity_sha256=observation.external_activity_sha256,
        external_activity_changed=external_activity_changed,
        window_role=(session.window_role if boundary.window_role == "legacy_unspecified"
                     else boundary.window_role),
    )
    session.records.append(receipt)
    coherence.record_server_window(controller, session, receipt)
    session.exploration_overhead_high_water_uj = controller._spent_exploration_energy(session)
    return receipt


def _context_monitor_prior(controller, session, boundary):
    policy = session.incumbent_policy
    prior = None
    if (policy is not None
            and session.incumbent_context_sha256 == controller._context_identity(session)
            and controller._qualifies(session, policy, boundary.token_end, boundary.finished_at_us)):
        prior = {
            "source_context_sha256": controller._context_identity(session),
            "source_active_batch": session.active_batch,
            "source_policy_hash": policy.policy_hash,
            "baseline_cost_bounds": controller._bounds(session, session.baseline, operational=True),
            "assisted_cost_bounds": controller._bounds(session, policy, operational=True),
            "evidence_state": "PRIOR_ONLY",
        }
    elif session.context_monitor_prior is not None and session.operational_verification:
        policy = session.verification_policy
        prior = dict(session.context_monitor_prior)
    if (prior is None or policy is None or policy not in session.candidates
            or not session.helper_available or boundary.policy.baseline
            or policy_evidence_key(session, policy) != policy_evidence_key(session, boundary.policy)):
        return None, None
    return policy, prior


def _begin_context_monitor(controller, session, policy, prior, boundary):
    session.context_monitor_prior = prior
    prior["target_context_sha256"] = controller._context_identity(session)
    session.probe_candidates = [policy]
    session.next_probe_index = 1
    session.refinement_added = True
    session.ticket_policy = None
    session.cached_winner = None
    session.verification_policy = policy
    session.operational_verification = True
    if not controller._reserve_verification_pair(session, boundary.token_end, boundary.finished_at_us):
        session.operational_verification = False
        session.verification_policy = None
        session.stage = "context_monitor_unaffordable"
        return controller._continue_best(
            session, boundary.token_end, boundary.finished_at_us, "CONTEXT_MONITOR_UNAFFORDABLE"
        )
    controller._state(session, "PROBING")
    if session.current_policy != policy:
        directive = controller._control(session, policy, boundary.token_end, boundary.finished_at_us)
    else:
        # The retained policy is already acknowledged; count this new measurement once.
        session.verification_budget["acknowledged"] = 1
        key = controller._probe_attempt_key(session, policy)
        session.probe_attempts[key] = session.probe_attempts.get(key, 0) + 1
        directive = controller._open_window(
            session, policy, boundary.token_end, boundary.finished_at_us, None
        )
    return controller._verification_result(session, directive, "RESERVED", "CONTEXT_MONITORING")


def record_window(
    controller,
    request_id: str,
    boundary: AdaptiveDecodeWindowBoundary,
    observation: AdaptiveDecodeRawWindowObservation,
    *,
    compatible_batch_change: bool = False,
) -> AdaptiveDecodeDirective:
    started_ns = time.perf_counter_ns()
    with controller._lock:
        session = controller._session(request_id)
        if session.awaiting_boundary != boundary:
            raise AdaptiveDecodeError("adaptive window transaction differs")
        if not isinstance(observation, AdaptiveDecodeRawWindowObservation):
            raise AdaptiveDecodeError("adaptive raw observation is invalid")
        if type(compatible_batch_change) is not bool:
            raise AdaptiveDecodeError("adaptive batch compatibility is invalid")
        drain_shortened_window = (
            (session.pending_session_drain_policy is not None
             or session.pending_helper_refresh_policy is not None)
            and session.target_token is not None
            and boundary.token_end < session.target_token
        )
        next_active_batch = (observation.next_active_batch
                             or observation.active_batch or session.active_batch)
        batch_changed = (observation.membership_changed
                         or next_active_batch != session.active_batch)
        # Other desktop work changes the comparability of every window; the
        # first observed identity only names the context already measured.
        external_changed = (
            observation.external_activity_sha256 is not None
            and session.external_activity_sha256 is not None
            and observation.external_activity_sha256 != session.external_activity_sha256
        )
        if observation.external_activity_sha256 is not None:
            session.external_activity_sha256 = observation.external_activity_sha256
        context_changed = batch_changed or external_changed
        compatible_context_change = (
            compatible_batch_change if batch_changed else external_changed
        )
        context_recovered = not session.execution_context_available and observation.execution_context_available
        session.execution_context_available = observation.execution_context_available
        if boundary.window_role == "exploration" and not boundary.policy.baseline:
            session.probe_tokens += boundary.token_count
        # Early maintenance boundaries retain accounting, not qualification.
        receipt = controller._append_window(
            session, boundary, observation,
            stable_window=not (drain_shortened_window or context_changed or context_recovered)
                and coherence.server_window_is_comparable(controller, session, boundary),
            external_activity_changed=external_changed,
        )
        monitor_policy, monitor_prior = (
            _context_monitor_prior(controller, session, boundary)
            if context_changed and compatible_context_change else (None, None)
        )
        if context_changed:
            previous_winner = controller._leading_candidate(session)
            session.incumbent_policy = None
            session.incumbent_context_sha256 = None
            session.challenger_policy = None
            session.probe_retry_policy = None
            session.pending_helper_refresh_policy = None
            session.deferred_policy = None
            session.deferred_control_reason = None
            session.zero_assistance_reason = (
                "CONTEXT_CHANGED" if batch_changed else "EXTERNAL_ACTIVITY_CHANGED"
            )
            session.active_batch = next_active_batch
            session.context_record_start = len(session.records)
            session.probe_budget = None
            session.probe_policy_hash = None
            session.verification_budget = None
            session.historical_records.clear()
            session.historical_group_counts.clear()
            for policy in (session.baseline, *session.candidates):
                identity = controller._policy_identity(policy)
                historical, group_count = (
                    controller._historical_policy_records(session, policy)
                )
                if historical:
                    session.historical_records[identity] = historical
                    session.historical_group_counts[identity] = (
                        group_count
                    )
            session.eliminated_policy_reasons.clear()
            session.warmup_windows_seen_by_policy.clear()
            session.warmup_latency_us_by_policy.clear()
            session.observed_control_cost_us = 0
            session.observed_control_tokens = 0
            session.probe_candidates = controller._sample_candidates(
                session.candidates,
                session.config,
                session.helper_evidence_state,
            )
            session.next_probe_index = 0
            session.stage = "initial_baseline"
            session.refinement_added = (
                session.helper_evidence_state == "LEARNING"
            )
            session.verification_policy = None
            session.operational_verification = False
            session.prior_monitor_policy = None
            session.cached_winner = (
                None
                if session.helper_evidence_state == "LEARNING" else
                controller._cached_verification_policy(session)
            )
            preferred = (
                session.cached_winner
                or previous_winner
                or controller._ticket_fallback(session)
            )
            if (
                preferred is not None
                and session.helper_evidence_state != "LEARNING"
            ):
                session.probe_candidates = [min(
                    session.candidates,
                    key=lambda row: (
                        abs(
                            row.split_fraction_ppm
                            - preferred.split_fraction_ppm
                        ),
                        row.policy_hash,
                    ),
                )]
                session.refinement_added = True
            if session.probe_candidates and controller._can_probe(
                session,
                boundary.token_end,
                boundary.finished_at_us,
            ):
                controller._state(session, "PROBING")
            else:
                controller._state(session, "EXPLOITING")
        session.awaiting_boundary = None
        session.current_policy = boundary.policy
        session.current_ack = None
        session.window_start_token = None
        session.window_start_us = None
        session.target_token = None
        server_directive = None
        if (session.config.server_policy_coherence and observation.failure_reason is None
                and session.pending_session_drain_policy is None
                and session.pending_helper_refresh_policy is None):
            server_directive = coherence.server_directive(
                controller, session, boundary.token_end, boundary.finished_at_us)
        if observation.failure_reason is not None:
            session.zero_assistance_reason = observation.failure_reason
            if boundary.policy.baseline:
                raise AdaptiveDecodeError(
                    "adaptive desktop baseline window failed"
                )
            controller._state(session, "RECOVERING")
            directive = controller._control(
                session,
                session.baseline,
                boundary.token_end,
                boundary.finished_at_us,
            )
            if session.operational_verification:
                session.operational_verification = False
                session.verification_budget = None
                directive = controller._verification_result(
                    session, directive, "INCOMPLETE", observation.failure_reason)
        elif server_directive is not None:
            directive = server_directive
        elif not session.execution_context_available:
            directive = controller._next_after_window(
                session, boundary.token_end, boundary.finished_at_us
            )
        elif context_changed and session.pending_session_drain_policy is not None:
            directive = controller._next_after_window(
                session, boundary.token_end, boundary.finished_at_us
            )
        elif context_changed and monitor_policy is not None:
            directive = _begin_context_monitor(
                controller, session, monitor_policy, monitor_prior, boundary
            )
        elif (context_changed and compatible_context_change and session.helper_available
              and boundary.policy.policy_hash in session.maintenance_policy_hashes):
            controller._state(session, "EXPLOITING")
            directive = controller._open_window(
                session, boundary.policy, boundary.token_end, boundary.finished_at_us, None
            )
        elif context_changed and not boundary.policy.baseline:
            directive = controller._control(
                session,
                session.baseline,
                boundary.token_end,
                boundary.finished_at_us,
            )
        elif (context_recovered and not context_changed and session.incumbent_policy is not None
              and next((row.policy for row in reversed(session.records) if not row.policy.baseline), None)
                  == session.incumbent_policy):
            directive = controller._continue_best(
                session, boundary.token_end, boundary.finished_at_us, "EXECUTION_CONTEXT_RECOVERED"
            )
        else:
            if not boundary.policy.baseline:
                controller._update_elimination(session, boundary.policy)
                if receipt.measurement_eligible:
                    controller._consider_incumbent(session, boundary.policy,
                                             boundary.token_end, boundary.finished_at_us)
                    coherence.qualify_server_policy(controller, session, boundary.policy,
                                                     boundary.token_end, boundary.finished_at_us)
            directive = controller._follow_coherent_policy(
                session, boundary.token_end, boundary.finished_at_us
            ) if boundary.policy.baseline else None
            if directive is None:
                directive = controller._next_after_window(
                    session, boundary.token_end, boundary.finished_at_us
                )
        if directive.reason in {
            "COHORT_MEMBERSHIP_CHANGED",
            "TAIL_SEALED",
        } and directive.control is None:
            controller._seal_session(session)
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return directive


def discard_stale_window(
    controller,
    request_id: str,
    boundary: AdaptiveDecodeWindowBoundary,
    observation: AdaptiveDecodeRawWindowObservation,
    reason: str,
    *,
    terminal_token_index: int | None = None,
    terminal_at_us: int | None = None,
) -> AdaptiveDecodeDirective:
    """Seal a released-slot tail without treating it as cost evidence."""
    started_ns = time.perf_counter_ns()
    if type(reason) is not str or not reason or not reason.isascii():
        raise AdaptiveDecodeError("adaptive stale window reason is invalid")
    with controller._lock:
        session = controller._session(request_id)
        if reason == "released_slot_phone_tail":
            return controller._seal_released_phone_tail(
                session, boundary, observation,
                terminal_token_index, terminal_at_us,
            )
        released = terminal_token_index is not None or terminal_at_us is not None
        acknowledgements = [
            (row.policy, row.applied_ack)
            for row in (*session.records, boundary)
            if row.applied_ack is not None
        ] if released else []
        last_policy, last_ack = (
            acknowledgements[-1] if acknowledgements else (None, None)
        )
        if (
            session.awaiting_boundary != boundary
            or not isinstance(
                observation, AdaptiveDecodeRawWindowObservation
            )
            or observation.failure_reason != reason
            or (not released and session.tail_seal_token != boundary.token_end)
            or (released and (
                type(terminal_token_index) is not int
                or terminal_token_index != session.output_tokens
                or type(terminal_at_us) is not int
                or terminal_at_us < boundary.finished_at_us
                or reason != "released_slot_baseline_tail"
                or "physical:terminal-release-confirmed" not in observation.evidence_ids
                or not boundary.policy.baseline
                or session.current_policy != boundary.policy
                or observation.completed_phone_calls != 0
                or observation.completed_phone_input_rows != 0
                or session.awaiting_control is not None
                or session.tail_seal_token not in (None, boundary.token_end)
                or (last_ack is not None and last_policy != boundary.policy)
                or (last_ack is None and any(
                    not row.policy.baseline for row in session.records))
            ))
        ):
            raise AdaptiveDecodeError(
                "adaptive stale window transaction differs"
            )
        if released:
            boundary = replace(
                boundary, token_end=terminal_token_index,
                finished_at_us=terminal_at_us,
            )
            session.tail_seal_token = terminal_token_index
        controller._append_window(
            session, boundary, observation, stable_window=False
        )
        session.awaiting_boundary = None
        session.current_policy = boundary.policy
        session.current_ack = boundary.applied_ack
        session.window_start_token = None
        session.window_start_us = None
        session.target_token = None
        session.tail_seal_reason = (
            reason if session.tail_seal_token < session.output_tokens else None
        )
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
