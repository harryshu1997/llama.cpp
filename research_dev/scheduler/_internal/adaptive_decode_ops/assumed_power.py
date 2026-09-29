"""AdaptiveDecodeController assumed power operations on its existing owner."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowReceipt,
    validate_policy_set,
)
from ..adaptive_decode_state import (
    _AdaptiveSession,
    _AssumedPhonePowerQuery,
    AdaptiveDecodeHistoricalEstimate,
)
from .common import (
    _assumed_phone_power_bounds,
    _assumed_phone_power_session,
    _operator_subset_bounds,
    _window_has_phone_work,
)


def historical_route_estimate_with_assumed_phone_power(
    controller,
    *,
    model_artifact_sha256: str,
    planning_profile_sha256: str,
    component_capability_sha256: str,
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
    context_length: int,
    active_batch: int,
    config: AdaptiveDecodeConfig,
    phone_power_by_domain: Mapping[str, tuple[int, int]],
    minimum_group_count: int = 2,
    route_geometry_prior: bool = False,
    operator_subset_prior: bool = False,
    operator_subset_source_layer_mask: int | None = None,
) -> AdaptiveDecodeHistoricalEstimate | None:
    """Rebuild fleet energy from separable domains and phone power."""
    rows = validate_policy_set(baseline, candidates)
    domains = {
        str(domain_id): tuple(powers)
        for domain_id, powers in phone_power_by_domain.items()
    }
    query = _AssumedPhonePowerQuery(
        model_artifact_sha256=model_artifact_sha256,
        planning_profile_sha256=planning_profile_sha256,
        component_capability_sha256=component_capability_sha256,
        baseline=baseline,
        rows=rows,
        active_batch=active_batch,
        config=config,
        domains=domains,
        minimum_group_count=minimum_group_count,
        route_geometry_prior=route_geometry_prior,
        operator_subset_prior=operator_subset_prior,
        operator_subset_source_layer_mask=(
            operator_subset_source_layer_mask
        ),
    )
    controller._validate_assumed_phone_power_query(query, context_length)
    requested_context_bucket = context_length.bit_length()
    with controller._lock:
        probe_session = _assumed_phone_power_session(
            query,
            "historical-assumed-phone-power-probe",
            context_length,
        )
        if route_geometry_prior or operator_subset_prior:
            group_matches, context_lengths = (
                controller._assumed_phone_power_prior_context_lengths(
                    query, context_length
                )
            )
        else:
            group_matches = lambda grouped: (
                controller._group_matches_session(grouped, probe_session)
            )
            context_lengths = controller._compatible_context_lengths(
                probe_session
            )
        for evidence_context_length in context_lengths:
            estimate = controller._assumed_phone_power_estimate_for_context(
                query,
                group_matches,
                evidence_context_length,
                requested_context_bucket,
            )
            if estimate is not None:
                return estimate
    return None


def _validate_assumed_phone_power_query(
    controller,
    query: _AssumedPhonePowerQuery,
    context_length: int,
) -> None:
    domains = query.domains
    if (
        not controller._valid_sha256(query.model_artifact_sha256)
        or not controller._valid_sha256(query.planning_profile_sha256)
        or not controller._valid_sha256(query.component_capability_sha256)
        or type(context_length) is not int
        or context_length <= 0
        or type(query.active_batch) is not int
        or query.active_batch <= 0
        or not isinstance(query.config, AdaptiveDecodeConfig)
        or type(query.minimum_group_count) is not int
        or query.minimum_group_count < 1
        or type(query.route_geometry_prior) is not bool
        or type(query.operator_subset_prior) is not bool
        or (query.route_geometry_prior and query.operator_subset_prior)
        or (
            query.operator_subset_prior
            and (
                type(query.operator_subset_source_layer_mask) is not int
                or query.operator_subset_source_layer_mask <= 0
            )
        )
        or (
            not query.operator_subset_prior
            and query.operator_subset_source_layer_mask is not None
        )
        or not domains
        or any(
            not domain_id
            or not domain_id.isascii()
            or len(powers) != 2
            or any(type(value) is not int or value <= 0 for value in powers)
            for domain_id, powers in domains.items()
        )
    ):
        raise AdaptiveDecodeError(
            "adaptive assumed phone power input is invalid"
        )


def _assumed_phone_power_prior_context_lengths(
    controller,
    query: _AssumedPhonePowerQuery,
    context_length: int,
) -> tuple[
    Callable[[AdaptiveDecodeGroupedObservation], bool],
    tuple[int, ...],
]:
    baseline = query.baseline
    route_geometry_prior = query.route_geometry_prior
    operator_subset_source_layer_mask = (
        query.operator_subset_source_layer_mask
    )
    policy_geometries = {
        controller._policy_geometry_identity(policy)
        for policy in (baseline, *query.rows)
    }

    def group_matches(
        grouped: AdaptiveDecodeGroupedObservation,
    ) -> bool:
        return (
            grouped.terminal_status == "COMPLETED"
            and grouped.model_artifact_sha256
                == query.model_artifact_sha256
            and grouped.desktop_placement_sha256
                == baseline.desktop_placement_sha256
            and any(
                (
                    controller._policy_geometry_identity(window.policy)
                        in policy_geometries
                    if route_geometry_prior else
                    (
                        not window.policy.baseline
                        and window.policy.layer_mask
                            == operator_subset_source_layer_mask
                    )
                )
                for window in grouped.windows
            )
        )

    context_buckets = {
        window.context_length.bit_length()
        for grouped in controller._history.values()
        if group_matches(grouped)
        for window in grouped.windows
        if window.output_valid
        and window.failure_reason is None
        and window.measurement_eligible
        and window.active_batch == query.active_batch
    }
    requested_bucket = context_length.bit_length()
    context_lengths = (context_length,) + tuple(
        1 << (bucket - 1)
        for bucket in sorted(
            context_buckets - {requested_bucket},
            key=lambda bucket: (
                abs(bucket - requested_bucket),
                bucket < requested_bucket,
                bucket,
            ),
        )
    )
    return group_matches, context_lengths


def _assumed_phone_power_policy_matches(
    controller,
    query: _AssumedPhonePowerQuery,
    observed: AdaptiveDecodePolicy,
    policy: AdaptiveDecodePolicy,
) -> bool:
    if query.route_geometry_prior:
        return (
            controller._policy_geometry_identity(observed)
            == controller._policy_geometry_identity(policy)
        )
    if not query.operator_subset_prior:
        return (
            controller._policy_identity(observed)
            == controller._policy_identity(policy)
        )
    if policy.baseline:
        return (
            observed.baseline
            and observed.desktop_placement_sha256
                == policy.desktop_placement_sha256
        )
    return (
        not observed.baseline
        and observed.layer_mask
            == query.operator_subset_source_layer_mask
        and observed.desktop_placement_sha256
            == policy.desktop_placement_sha256
        and observed.columns == policy.columns
        and observed.split_fraction_ppm
            == policy.split_fraction_ppm
        and observed.layer_mask & policy.layer_mask
            == policy.layer_mask
    )


def _assumed_phone_power_records_for(
    controller,
    query: _AssumedPhonePowerQuery,
    policy: AdaptiveDecodePolicy,
    session: _AdaptiveSession,
    group_matches: Callable[[AdaptiveDecodeGroupedObservation], bool],
    context_bucket: int,
) -> tuple[
    tuple[AdaptiveDecodeWindowReceipt, ...],
    int,
    AdaptiveDecodePolicy | None,
]:
    prior = query.route_geometry_prior or query.operator_subset_prior
    domains = query.domains
    records_by_geometry: dict[
        str, list[AdaptiveDecodeWindowReceipt]
    ] = {}
    groups_by_geometry: dict[str, set[str]] = {}
    policies_by_geometry: dict[
        str, AdaptiveDecodePolicy
    ] = {}
    for grouped in controller._history.values():
        if not (
            group_matches(grouped)
            if prior else
            controller._group_matches_session(grouped, session)
        ):
            continue
        selected = tuple(
            window for window in grouped.windows
            if window.output_valid
            and window.failure_reason is None
            and window.energy_measurement_eligible
            and window.active_batch == query.active_batch
            and window.context_length.bit_length()
                == context_bucket
            and controller._assumed_phone_power_policy_matches(
                query, window.policy, policy
            )
            and all(
                domain_id
                    in window.fleet_energy_uj_by_domain
                for domain_id in domains
            )
            and (policy.baseline or _window_has_phone_work(window))
        )
        if selected:
            source_policy = selected[0].policy
            source_identity = (
                controller._policy_geometry_identity(source_policy)
                if prior else
                controller._policy_identity(source_policy)
            )
            if any(
                controller._policy_geometry_identity(window.policy)
                    != controller._policy_geometry_identity(
                        source_policy
                    )
                for window in selected
            ):
                raise AdaptiveDecodeError(
                    "adaptive history group mixes policies"
                )
            records_by_geometry.setdefault(
                source_identity, []
            ).extend(selected)
            groups_by_geometry.setdefault(
                source_identity, set()
            ).add(grouped.grouped_observation_sha256)
            policies_by_geometry[source_identity] = (
                source_policy
            )
    if not records_by_geometry:
        return (), 0, None
    eligible_geometries = tuple(
        identity for identity in records_by_geometry
        if len(groups_by_geometry[identity])
            >= query.minimum_group_count
    )
    pool = (
        eligible_geometries
        if eligible_geometries else
        tuple(records_by_geometry)
    )
    selected_identity = min(
        pool,
        key=lambda identity: (
            policies_by_geometry[identity]
                .layer_mask.bit_count(),
            -len(groups_by_geometry[identity]),
            identity,
        ),
    )
    return (
        tuple(records_by_geometry[selected_identity]),
        len(groups_by_geometry[selected_identity]),
        policies_by_geometry[selected_identity],
    )


def _assumed_phone_power_estimate_for_context(
    controller,
    query: _AssumedPhonePowerQuery,
    group_matches: Callable[[AdaptiveDecodeGroupedObservation], bool],
    evidence_context_length: int,
    requested_context_bucket: int,
) -> AdaptiveDecodeHistoricalEstimate | None:
    baseline = query.baseline
    config = query.config
    session = _assumed_phone_power_session(
        query,
        "historical-assumed-phone-power",
        evidence_context_length,
    )
    context_bucket = evidence_context_length.bit_length()
    (
        baseline_records,
        baseline_groups,
        _baseline_source_policy,
    ) = controller._assumed_phone_power_records_for(
        query, baseline, session, group_matches, context_bucket
    )
    baseline_bounds = _assumed_phone_power_bounds(
        query, baseline_records, baseline_groups
    )
    if baseline_bounds is None:
        return None
    required_energy_upper = (
        baseline_bounds[1]
        * (1_000_000 - config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    latency_limit = (
        baseline_bounds[4] * config.maximum_latency_ppm
        + 999_999
    ) // 1_000_000
    choices = []
    for policy in query.rows:
        (
            candidate_records,
            candidate_groups,
            source_policy,
        ) = controller._assumed_phone_power_records_for(
            query, policy, session, group_matches, context_bucket
        )
        candidate_bounds = _assumed_phone_power_bounds(
            query, candidate_records, candidate_groups
        )
        if (
            candidate_bounds is not None
            and query.operator_subset_prior
            and source_policy is not None
        ):
            candidate_bounds = _operator_subset_bounds(
                baseline_bounds,
                candidate_bounds,
                policy,
                source_policy,
            )
        if (
            candidate_bounds is None
            or candidate_bounds[2] > required_energy_upper
            or candidate_bounds[4] > latency_limit
        ):
            continue
        choices.append((
            candidate_bounds[2],
            candidate_bounds[4],
            policy.policy_hash,
            policy,
            candidate_bounds,
            candidate_groups,
            candidate_records,
            source_policy,
        ))
    if not choices:
        return None
    (
        _energy_upper,
        _latency_upper,
        _policy_hash,
        selected_policy,
        selected_bounds,
        selected_groups,
        selected_records,
        source_policy,
    ) = min(choices)
    boundaries = {
        window.energy_boundary_id
        for window in (*baseline_records, *selected_records)
    }
    if len(boundaries) != 1:
        return None
    return AdaptiveDecodeHistoricalEstimate(
        baseline_policy=baseline,
        selected_policy=selected_policy,
        baseline_energy_per_token_uj=baseline_bounds[0],
        baseline_energy_lower_per_token_uj=baseline_bounds[1],
        baseline_energy_upper_per_token_uj=baseline_bounds[2],
        selected_energy_per_token_uj=selected_bounds[0],
        selected_energy_lower_per_token_uj=selected_bounds[1],
        selected_energy_upper_per_token_uj=selected_bounds[2],
        baseline_latency_per_token_us=baseline_bounds[3],
        baseline_latency_upper_per_token_us=baseline_bounds[4],
        selected_latency_per_token_us=selected_bounds[3],
        selected_latency_upper_per_token_us=selected_bounds[4],
        baseline_group_count=baseline_groups,
        selected_group_count=selected_groups,
        energy_boundary_id=next(iter(boundaries)),
        requested_context_bucket=requested_context_bucket,
        evidence_context_bucket=context_bucket,
        evidence_match=(
            "route_geometry_prior"
            if query.route_geometry_prior else
            "route_operator_subset_prior"
            if query.operator_subset_prior else "component"
        ),
        source_layer_mask=(
            None
            if source_policy is None
            else source_policy.layer_mask
        ),
        target_layer_mask=selected_policy.layer_mask,
        evidence_scale_ppm=(
            1_000_000
            if source_policy is None
            or source_policy.layer_mask == 0 else
            selected_policy.layer_mask.bit_count()
                * 1_000_000
                // source_policy.layer_mask.bit_count()
        ),
    )
