"""AdaptiveDecodeController reporting operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodeGroupedObservation


def grouped_observation(
    controller, request_id: str
) -> AdaptiveDecodeGroupedObservation:
    with controller._lock:
        result = controller._completed.get(request_id)
        if result is None:
            raise AdaptiveDecodeError("adaptive grouped observation is absent")
        return result


def timing(controller, request_id: str) -> Mapping[str, int]:
    with controller._lock:
        session = controller._sessions.get(request_id)
        if session is None:
            session = controller._sealed_sessions.get(request_id)
        if session is None:
            raise AdaptiveDecodeError("adaptive timing session is absent")
        rows = sorted(session.decision_time_us)
        if not rows:
            return MappingProxyType({})
        p95 = rows[min(len(rows) - 1, (95 * len(rows) - 1) // 100)]
        return MappingProxyType({
            "count": len(rows),
            "maximum_us": rows[-1],
            "p50_us": rows[(len(rows) - 1) // 2],
            "p95_us": p95,
            "total_us": sum(rows),
        })


def snapshot(controller, request_id: str) -> Mapping[str, object]:
    with controller._lock:
        session = controller._terminal_session(request_id)
        return MappingProxyType({
            "incumbent_policy_hash": (None if session.incumbent_policy is None
                                      else session.incumbent_policy.policy_hash),
            "incumbent_fraction_ppm": (None if session.incumbent_policy is None
                                       else session.incumbent_policy.split_fraction_ppm),
            "incumbent_context_sha256": session.incumbent_context_sha256,
            "challenger_policy_hash": (None if session.challenger_policy is None
                                       else session.challenger_policy.policy_hash),
            "challenger_fraction_ppm": (None if session.challenger_policy is None
                                        else session.challenger_policy.split_fraction_ppm),
            "acknowledged_policy_hash": (None if session.acknowledged_policy is None
                                          else session.acknowledged_policy.policy_hash),
            "context_identity_sha256": controller._context_identity(session),
            "probe_attempts": dict(sorted(session.probe_attempts.items())),
            "remaining_probe_tokens": max(0, session.config.maximum_probe_tokens - session.probe_tokens),
            "estimated_exploration_overhead_uj": controller._spent_exploration_energy(session),
            "zero_assistance_reason": session.zero_assistance_reason,
            "maximum_latency_ppm": session.config.maximum_latency_ppm,
            "remaining_output_tokens": controller._remaining_tokens(session, max(
                session.window_start_token or 0, session.transition_start_token or 0,
                session.records[-1].token_end if session.records else 0)),
            "stage": session.stage,
            "qualification_measurement_plan": (None if session.qualification_measurement_plan is None
                                               else dict(session.qualification_measurement_plan)),
            "evidence": {
                row.policy_hash: {
                    "current_valid_windows": len(controller._current_valid_records(session, row, operational=True)),
                    "historical_groups": session.historical_group_counts.get(controller._policy_identity(row), 0),
                } for row in (session.baseline, *session.candidates)
            },
            "awaiting_boundary": (
                None
                if session.awaiting_boundary is None
                else {
                    "token_end": session.awaiting_boundary.token_end,
                    "window_index": session.awaiting_boundary.window_index,
                }
            ),
            "awaiting_control": (
                None
                if session.awaiting_control is None
                else session.awaiting_control.to_json()
            ),
            "candidate_policy_hashes": tuple(
                row.policy_hash for row in session.candidates
            ),
            "active_batch": session.active_batch,
            "external_activity_sha256": session.external_activity_sha256,
            "window_role": session.window_role,
            "context_monitor_prior": (None if session.context_monitor_prior is None
                                      else dict(session.context_monitor_prior)),
            "pending_helper_refresh_policy_hash": (
                None if session.pending_helper_refresh_policy is None
                else session.pending_helper_refresh_policy.policy_hash
            ),
            "eliminated_policy_reasons": dict(
                sorted(session.eliminated_policy_reasons.items())
            ),
            "current_policy_hash": (
                None
                if session.current_policy is None
                else session.current_policy.policy_hash
            ),
            "deferred_control_count": session.deferred_control_count,
            "deferred_control_reason": (
                session.deferred_control_reason
            ),
            "deferred_policy_hash": (
                None
                if session.deferred_policy is None
                else session.deferred_policy.policy_hash
            ),
            "helper_available": session.helper_available,
            "helper_layout_generation": (
                session.helper_layout_generation
            ),
            "helper_layout_geometry_sha256": (
                session.helper_layout_geometry_sha256
            ),
            "helper_evidence_state": session.helper_evidence_state,
            "allow_assumed_phone_power_for_operational_selection": (
                session.config.allow_assumed_phone_power_for_operational_selection
            ),
            "maintenance_policy_hashes": tuple(sorted(
                session.maintenance_policy_hashes
            )),
            "pending_session_drain_layer_mask": (
                None
                if session.pending_session_drain_policy is None
                else session.pending_session_drain_policy.layer_mask
            ),
            "pending_session_drain_policy_hash": (
                None
                if session.pending_session_drain_policy is None
                else session.pending_session_drain_policy.policy_hash
            ),
            "probe_tokens": session.probe_tokens,
            "execution_context_available": session.execution_context_available,
            "observed_control_cost_us": session.observed_control_cost_us,
            "observed_control_tokens": session.observed_control_tokens,
            "warmup_latency_us_by_policy": dict(session.warmup_latency_us_by_policy),
            "context_record_start": session.context_record_start,
            "probe_budget": (None if session.probe_budget is None else dict(session.probe_budget)),
            "minimum_energy_saving_ppm": session.config.minimum_energy_saving_ppm,
            "verification": {
                "outcome": session.verification_outcome,
                "reason": session.verification_reason,
                "attempts": session.verification_attempts,
                "budget": (None if session.verification_budget is None
                           else dict(session.verification_budget)),
                "policy_hash": (None if session.verification_policy is None
                                else session.verification_policy.policy_hash),
            },
            "probe_fractions_ppm": tuple(
                row.split_fraction_ppm
                for row in session.probe_candidates
            ),
            "record_count": len(session.records),
            "tail_state": (
                "SEALED"
                if request_id in controller._sealed_sessions else "ACTIVE"
            ),
            "measurement_eligible_record_count": sum(
                row.measurement_eligible for row in session.records
            ),
            "transition_start_token": session.transition_start_token,
            "state": session.state,
            "state_history": tuple(session.state_history),
            "target_token": session.target_token,
        })


def assistance_summaries(controller) -> tuple[Mapping[str, object], ...]:
    """Per active request: is phone assistance in use, and what could be lost.

    ``loss_upper_per_token_uj`` is the baseline energy upper bound minus the
    assisted lower bound per token from operational evidence; None when either
    bound is missing, which callers must treat as unknown.
    """
    with controller._lock:
        rows = []
        for request_id, session in controller._sessions.items():
            assisted_policy = None
            if session.helper_available:
                if session.incumbent_policy is not None and not session.incumbent_policy.baseline:
                    assisted_policy = session.incumbent_policy
                elif session.current_policy is not None and not session.current_policy.baseline:
                    assisted_policy = session.current_policy
            baseline = controller._bounds(session, session.baseline, operational=True)
            assisted = (None if assisted_policy is None
                        else controller._bounds(session, assisted_policy, operational=True))
            token_index = max(
                session.window_start_token or 0, session.transition_start_token or 0,
                session.records[-1].token_end if session.records else 0)
            rows.append(MappingProxyType({
                "request_id": request_id,
                "ticket_id": session.ticket_id,
                "assisted": assisted_policy is not None,
                "assisted_fraction_ppm": (None if assisted_policy is None
                                          else assisted_policy.split_fraction_ppm),
                "remaining_tokens": controller._remaining_tokens(session, token_index),
                "loss_upper_per_token_uj": (
                    None if baseline is None or assisted is None
                    else max(0, baseline[2] - assisted[1])),
                "assisted_latency_per_token_us": (None if assisted is None else assisted[3]),
                "external_activity_sha256": session.external_activity_sha256,
            }))
        return tuple(rows)
