"""AdaptiveDecodeController estimates operations on its existing owner."""

from __future__ import annotations

from math import isqrt
from typing import Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
    validate_policy_set,
)
from ..adaptive_decode_state import _AdaptiveSession, AdaptiveDecodeHistoricalEstimate


def historical_route_estimate(
    controller,
    *,
    model_artifact_sha256: str,
    planning_profile_sha256: str,
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
    context_length: int,
    active_batch: int,
    config: AdaptiveDecodeConfig,
    minimum_group_count: int = 2,
    component_capability_sha256: str | None = None,
) -> AdaptiveDecodeHistoricalEstimate | None:
    """Return a qualified fraction estimate from grouped observations."""
    rows = validate_policy_set(baseline, candidates)
    if (
        type(context_length) is not int
        or context_length <= 0
        or type(active_batch) is not int
        or active_batch <= 0
        or not isinstance(config, AdaptiveDecodeConfig)
        or type(minimum_group_count) is not int
        or minimum_group_count < 1
    ):
        raise AdaptiveDecodeError(
            "adaptive historical estimate input is invalid"
        )
    requested_context_bucket = context_length.bit_length()
    component_capability_sha256 = (
        planning_profile_sha256
        if component_capability_sha256 is None
        else component_capability_sha256
    )
    if not controller._valid_sha256(component_capability_sha256):
        raise AdaptiveDecodeError(
            "adaptive historical component identity is invalid"
        )
    with controller._lock:
        if not controller._history:
            return None
        probe_session = _AdaptiveSession(
            request_id="historical-estimate-probe",
            ticket_id="historical-estimate-probe:attempt:0",
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            component_capability_sha256=(
                component_capability_sha256
            ),
            baseline=baseline,
            candidates=rows,
            output_tokens=1,
            context_length=context_length,
            active_batch=active_batch,
            deadline_us=1,
            config=config,
            slot_id=0,
        )
        context_lengths = controller._compatible_context_lengths(
            probe_session
        )
        for evidence_context_length in context_lengths:
            session = _AdaptiveSession(
                request_id="historical-estimate",
                ticket_id="historical-estimate:attempt:0",
                model_artifact_sha256=model_artifact_sha256,
                planning_profile_sha256=planning_profile_sha256,
                component_capability_sha256=(
                    component_capability_sha256
                ),
                baseline=baseline,
                candidates=rows,
                output_tokens=1,
                context_length=evidence_context_length,
                active_batch=active_batch,
                deadline_us=1,
                config=config,
                slot_id=0,
            )
            for policy in (baseline, *rows):
                identity = controller._policy_identity(policy)
                historical, group_count = (
                    controller._historical_policy_records(session, policy)
                )
                if historical:
                    session.historical_records[identity] = historical
                    session.historical_group_counts[identity] = (
                        group_count
                    )
            baseline_identity = controller._policy_identity(baseline)
            baseline_groups = session.historical_group_counts.get(
                baseline_identity, 0
            )
            candidate_groups = max(
                (
                    session.historical_group_counts.get(
                        controller._policy_identity(policy), 0
                    )
                    for policy in rows
                ),
                default=0,
            )
            evidence_is_conclusive = min(
                baseline_groups, candidate_groups
            ) >= minimum_group_count
            selected = controller._cached_verification_policy(session)
            if selected is None:
                if evidence_is_conclusive:
                    return None
                continue
            selected_identity = controller._policy_identity(selected)
            selected_groups = session.historical_group_counts.get(
                selected_identity, 0
            )
            if min(
                baseline_groups, selected_groups
            ) < minimum_group_count:
                if evidence_is_conclusive:
                    return None
                continue
            baseline_bounds = controller._bounds(session, baseline)
            selected_bounds = controller._bounds(session, selected)
            if baseline_bounds is None or selected_bounds is None:
                if evidence_is_conclusive:
                    return None
                continue
            energy_records = tuple(
                row
                for policy in (baseline, selected)
                for row in session.historical_records.get(
                    controller._policy_identity(policy), ()
                )
                if row.energy_measurement_eligible
            )
            boundaries = {
                row.energy_boundary_id for row in energy_records
            }
            if len(boundaries) != 1:
                return None
            return AdaptiveDecodeHistoricalEstimate(
                baseline_policy=baseline,
                selected_policy=selected,
                baseline_energy_per_token_uj=baseline_bounds[0],
                baseline_energy_lower_per_token_uj=baseline_bounds[1],
                baseline_energy_upper_per_token_uj=baseline_bounds[2],
                selected_energy_per_token_uj=selected_bounds[0],
                selected_energy_lower_per_token_uj=selected_bounds[1],
                selected_energy_upper_per_token_uj=selected_bounds[2],
                baseline_latency_per_token_us=baseline_bounds[3],
                baseline_latency_upper_per_token_us=(
                    baseline_bounds[4]
                ),
                selected_latency_per_token_us=selected_bounds[3],
                selected_latency_upper_per_token_us=selected_bounds[4],
                baseline_group_count=baseline_groups,
                selected_group_count=selected_groups,
                energy_boundary_id=next(iter(boundaries)),
                requested_context_bucket=requested_context_bucket,
                evidence_context_bucket=(
                    evidence_context_length.bit_length()
                ),
            )
        return None


def historical_component_latency(
    controller,
    *,
    model_artifact_sha256: str,
    planning_profile_sha256: str,
    component_capability_sha256: str,
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
    selected_policy: AdaptiveDecodePolicy,
    context_length: int,
    active_batch: int,
    config: AdaptiveDecodeConfig,
    minimum_group_count: int = 1,
) -> tuple[int, int, int] | None:
    """Return physical latency evidence without reusing route energy."""
    rows = validate_policy_set(baseline, candidates)
    identities = {
        controller._policy_identity(policy) for policy in (baseline, *rows)
    }
    selected_identity = controller._policy_identity(selected_policy)
    if (
        not controller._valid_sha256(model_artifact_sha256)
        or not controller._valid_sha256(planning_profile_sha256)
        or not controller._valid_sha256(component_capability_sha256)
        or selected_identity not in identities
        or type(context_length) is not int
        or context_length <= 0
        or type(active_batch) is not int
        or active_batch <= 0
        or not isinstance(config, AdaptiveDecodeConfig)
        or type(minimum_group_count) is not int
        or minimum_group_count < 1
    ):
        raise AdaptiveDecodeError(
            "adaptive component latency input is invalid"
        )
    with controller._lock:
        probe_session = _AdaptiveSession(
            request_id="historical-component-latency-probe",
            ticket_id=(
                "historical-component-latency-probe:attempt:0"
            ),
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            component_capability_sha256=(
                component_capability_sha256
            ),
            baseline=baseline,
            candidates=rows,
            output_tokens=1,
            context_length=context_length,
            active_batch=active_batch,
            deadline_us=1,
            config=config,
            slot_id=0,
        )
        context_lengths = controller._compatible_context_lengths(
            probe_session
        )
        for evidence_context_length in context_lengths:
            session = _AdaptiveSession(
                request_id="historical-component-latency",
                ticket_id="historical-component-latency:attempt:0",
                model_artifact_sha256=model_artifact_sha256,
                planning_profile_sha256=planning_profile_sha256,
                component_capability_sha256=(
                    component_capability_sha256
                ),
                baseline=baseline,
                candidates=rows,
                output_tokens=1,
                context_length=evidence_context_length,
                active_batch=active_batch,
                deadline_us=1,
                config=config,
                slot_id=0,
            )
            records = []
            group_count = 0
            context_bucket = evidence_context_length.bit_length()
            for grouped in controller._history.values():
                if not controller._group_matches_session(grouped, session):
                    continue
                selected = tuple(
                    row for row in grouped.windows
                    if row.output_valid
                    and row.failure_reason is None
                    and row.measurement_eligible
                    and row.active_batch == active_batch
                    and row.context_length.bit_length()
                        == context_bucket
                    and controller._policy_identity(row.policy)
                        == selected_identity
                    and (
                        selected_policy.baseline
                        or (
                            row.completed_phone_calls is not None
                            and row.completed_phone_calls > 0
                        )
                        or (
                            row.phone_compute_us > 0
                            and (
                                row.usb_upload_bytes > 0
                                or row.usb_download_bytes > 0
                            )
                        )
                    )
                )
                if selected:
                    records.extend(selected)
                    group_count += 1
            if group_count < minimum_group_count or not records:
                continue
            token_count = sum(row.token_count for row in records)
            duration_us = sum(
                row.finished_at_us - row.started_at_us
                for row in records
            )
            mean_us = max(1, duration_us // token_count)
            uncertainty_ppm = max(
                20_000,
                config.uncertainty_ppm
                    // max(1, isqrt(group_count)),
            )
            upper_us = (
                mean_us * (1_000_000 + uncertainty_ppm) + 999_999
            ) // 1_000_000
            return mean_us, upper_us, group_count
    return None
