"""AdaptiveDecodeController bounds operations on its existing owner."""

from __future__ import annotations

from math import isqrt

from ..adaptive_decode_contracts import AdaptiveDecodePolicy, AdaptiveDecodeWindowReceipt
from ..adaptive_decode_state import _AdaptiveSession
from .budgeting import policy_evidence_key


def _operational_energy_eligible(
    session: _AdaptiveSession,
    row: AdaptiveDecodeWindowReceipt,
) -> bool:
    return row.energy_measurement_eligible or (
        session.config.allow_assumed_phone_power_for_operational_selection
        and row.measurement_eligible
        and "ASSUMED_4P5W" in row.evidence_ids
    )


def _valid_records(
    _controller_class,
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    *,
    operational: bool = False,
) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
    current = tuple(
        row for row in session.records[session.context_record_start:]
        if policy_evidence_key(session, row.policy) == policy_evidence_key(session, policy)
        and row.active_batch == session.active_batch
        and row.output_valid
        and row.failure_reason is None
        and (
            controller_type._operational_energy_eligible(session, row)
            if operational else row.energy_measurement_eligible
        )
    )
    historical = session.historical_records.get(
        _controller_class._policy_identity(policy), ()
    )
    return current + tuple(
        row for row in historical if (
            row.active_batch == session.active_batch
            and (not session.config.server_policy_coherence
                 or row.external_activity_sha256 == session.external_activity_sha256)
        ) and (
            controller_type._operational_energy_eligible(session, row)
            if operational else row.energy_measurement_eligible
        )
    )


def _current_valid_records(
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    *,
    operational: bool = False,
) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
    return tuple(
        row for row in session.records[session.context_record_start:]
        if policy_evidence_key(session, row.policy) == policy_evidence_key(session, policy)
        and row.active_batch == session.active_batch
        and row.output_valid
        and row.failure_reason is None
        and (
            controller_type._operational_energy_eligible(session, row)
            if operational else row.energy_measurement_eligible
        )
    )


def _current_valid_latency_records(
    session: _AdaptiveSession, policy: AdaptiveDecodePolicy
) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
    return tuple(
        row for row in session.records[session.context_record_start:]
        if policy_evidence_key(session, row.policy) == policy_evidence_key(session, policy)
        and row.active_batch == session.active_batch
        and row.output_valid
        and row.failure_reason is None
        and row.measurement_eligible
    )


def _current_latency_bounds(
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    *,
    minimum_count: int = 2,
) -> tuple[int, int] | None:
    rows = controller_type._current_valid_latency_records(session, policy)
    if len(rows) < minimum_count:
        return None
    tokens = sum(row.token_count for row in rows)
    duration = sum(
        row.finished_at_us - row.started_at_us for row in rows
    )
    mean = max(1, duration // tokens)
    uncertainty = max(
        20_000,
        session.config.uncertainty_ppm // max(1, isqrt(len(rows))),
    )
    upper = (
        mean * (1_000_000 + uncertainty) + 999_999
    ) // 1_000_000
    return mean, upper


def _valid_latency_records(
    _controller_class,
    session: _AdaptiveSession, policy: AdaptiveDecodePolicy
) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
    current = tuple(
        row for row in session.records[session.context_record_start:]
        if policy_evidence_key(session, row.policy) == policy_evidence_key(session, policy)
        and row.active_batch == session.active_batch
        and row.output_valid
        and row.failure_reason is None
        and row.measurement_eligible
    )
    historical = session.historical_records.get(
        _controller_class._policy_identity(policy), ()
    )
    return current + tuple(row for row in historical
                           if row.active_batch == session.active_batch
                           and (not session.config.server_policy_coherence
                                or row.external_activity_sha256 == session.external_activity_sha256))


def _latency_bounds(
    controller_type, session: _AdaptiveSession, policy: AdaptiveDecodePolicy
) -> tuple[int, int] | None:
    rows = controller_type._valid_latency_records(session, policy)
    if not rows:
        return None
    tokens = sum(row.token_count for row in rows)
    duration = sum(
        row.finished_at_us - row.started_at_us for row in rows
    )
    mean = max(1, duration // tokens)
    current_count = sum(
        policy_evidence_key(session, row.policy) == policy_evidence_key(session, policy)
        and row.active_batch == session.active_batch
        and row.output_valid
        and row.failure_reason is None
        and row.measurement_eligible
        for row in session.records[session.context_record_start:]
    )
    historical_groups = session.historical_group_counts.get(
        controller_type._policy_identity(policy), 0
    )
    uncertainty = max(
        20_000,
        session.config.uncertainty_ppm // max(
            1, isqrt(current_count + historical_groups)
        ),
    )
    upper = (
        mean * (1_000_000 + uncertainty) + 999_999
    ) // 1_000_000
    return mean, upper


def _bounds(
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    *,
    operational: bool = False,
) -> tuple[int, int, int, int, int] | None:
    rows = controller_type._valid_records(
        session, policy, operational=operational
    )
    if not rows:
        return None
    energy_tokens = sum(row.energy_token_count for row in rows)
    energy = sum(row.whole_fleet_energy_uj for row in rows)
    energy_mean = max(1, energy // energy_tokens)
    current_count = len(controller_type._current_valid_records(
        session, policy, operational=operational
    ))
    historical_groups = session.historical_group_counts.get(
        controller_type._policy_identity(policy), 0
    )
    uncertainty = max(
        20_000,
        session.config.uncertainty_ppm // max(
            1, isqrt(current_count + historical_groups)
        ),
    )
    energy_lower = max(
        1, energy_mean * (1_000_000 - uncertainty) // 1_000_000
    )
    energy_upper = (
        energy_mean * (1_000_000 + uncertainty) + 999_999
    ) // 1_000_000
    latency = controller_type._latency_bounds(session, policy)
    if latency is None:
        return None
    latency_mean, latency_upper = latency
    return energy_mean, energy_lower, energy_upper, latency_mean, latency_upper


def _token_latency_us(
    controller_type, session: _AdaptiveSession, policy: AdaptiveDecodePolicy
) -> int:
    bounds = controller_type._latency_bounds(session, policy)
    if bounds is not None:
        return bounds[0]
    if policy.predicted_latency_per_token_us is not None:
        return policy.predicted_latency_per_token_us
    return max(1, session.config.measurement_resolution_us)
