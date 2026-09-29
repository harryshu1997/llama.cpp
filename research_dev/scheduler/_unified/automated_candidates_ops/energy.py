"""AutomatedCandidateMixin energy operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.model_manifest import ModelManifest
from ..._internal.route_generation import AutomatedRouteCompiler
from ..._internal.runtime_plan import AutomatedRouteCandidate
from ..common import _apportion_measured_route_energy
from .common import (
    _AdaptiveEnergyDeltas,
    _AdaptiveHistoryContext,
    _AdaptiveWarmEstimate,
    _ExactRouteDecompositions,
    _TransitionEnergyEstimate,
)


def _adaptive_warm_estimate(
    parent_cost,
    history,
    output_tokens: int,
) -> _AdaptiveWarmEstimate:
    parent_warm_lower = parent_cost.warm_execution_energy_lower_uj
    parent_warm = parent_cost.warm_execution_energy_uj
    parent_warm_upper = parent_cost.warm_execution_energy_upper_uj
    assert parent_warm_lower is not None
    assert parent_warm is not None
    assert parent_warm_upper is not None
    shared_lower = max(
        1,
        parent_warm_lower
        - history.baseline_energy_upper_per_token_uj * output_tokens,
    )
    shared = max(
        1,
        parent_warm
        - history.baseline_energy_per_token_uj * output_tokens,
    )
    shared_upper = max(
        shared,
        parent_warm_upper
        - history.baseline_energy_lower_per_token_uj * output_tokens,
    )
    warm_lower = (
        shared_lower
        + history.selected_energy_lower_per_token_uj * output_tokens
    )
    warm = shared + history.selected_energy_per_token_uj * output_tokens
    warm_upper = (
        shared_upper
        + history.selected_energy_upper_per_token_uj * output_tokens
    )
    shared_service_us = max(
        1,
        parent_cost.component_service_us
        - history.baseline_latency_per_token_us * output_tokens,
    )
    shared_service_upper_us = max(
        shared_service_us,
        parent_cost.component_service_us
        - history.baseline_latency_upper_per_token_us * output_tokens,
    )
    return _AdaptiveWarmEstimate(
        warm_lower_uj=min(warm_lower, warm),
        warm_energy_uj=warm,
        warm_upper_uj=max(warm_upper, warm),
        parent_component_service_us=(
            shared_service_us
            + history.baseline_latency_per_token_us * output_tokens
        ),
        parent_component_service_upper_us=(
            shared_service_upper_us
            + history.baseline_latency_upper_per_token_us * output_tokens
        ),
        selected_component_service_us=(
            shared_service_us
            + history.selected_latency_per_token_us * output_tokens
        ),
        selected_component_service_upper_us=(
            shared_service_upper_us
            + history.selected_latency_upper_per_token_us * output_tokens
        ),
        parent_warm_lower_uj=(
            shared_lower
            + history.baseline_energy_lower_per_token_uj * output_tokens
        ),
        parent_warm_energy_uj=(
            shared
            + history.baseline_energy_per_token_uj * output_tokens
        ),
        parent_warm_upper_uj=(
            shared_upper
            + history.baseline_energy_upper_per_token_uj * output_tokens
        ),
    )


def _adaptive_transition_energy(
    compiler: AutomatedRouteCompiler,
    manifest: ModelManifest,
    route: AutomatedRouteCandidate,
    default: tuple[int, int, int],
) -> _TransitionEnergyEstimate:
    qualified = all(
        row.energy_maturity == "QUALIFIED"
        for row in route.plan.transitions
    )
    estimates = compiler.transition_estimates_for_plan(
        manifest, route.plan, route.binding.executor_id
    )
    lower, point, upper = default
    if qualified and route.plan.transitions:
        if any(
            estimate is None
            or estimate.energy_lower_uj is None
            or estimate.energy_uj is None
            or estimate.energy_upper_uj is None
            for estimate in estimates
        ):
            qualified = False
        else:
            lower = sum(
                estimate.energy_lower_uj
                for estimate in estimates
                if estimate is not None
                and estimate.energy_lower_uj is not None
            )
            point = sum(
                estimate.energy_uj
                for estimate in estimates
                if estimate is not None and estimate.energy_uj is not None
            )
            upper = sum(
                estimate.energy_upper_uj
                for estimate in estimates
                if estimate is not None
                and estimate.energy_upper_uj is not None
            )
    return _TransitionEnergyEstimate(lower, point, upper, qualified)


def _adaptive_energy_deltas(
    history,
    output_tokens: int,
    candidate_transition: _TransitionEnergyEstimate,
    parent_transition: _TransitionEnergyEstimate,
) -> _AdaptiveEnergyDeltas:
    warm_lower = (
        history.selected_energy_lower_per_token_uj
        - history.baseline_energy_upper_per_token_uj
    ) * output_tokens
    warm = (
        history.selected_energy_per_token_uj
        - history.baseline_energy_per_token_uj
    ) * output_tokens
    warm_upper = (
        history.selected_energy_upper_per_token_uj
        - history.baseline_energy_lower_per_token_uj
    ) * output_tokens
    transition_lower = (
        candidate_transition.lower_uj - parent_transition.upper_uj
    )
    transition = (
        candidate_transition.energy_uj - parent_transition.energy_uj
    )
    transition_upper = (
        candidate_transition.upper_uj - parent_transition.lower_uj
    )
    return _AdaptiveEnergyDeltas(
        paired_warm_lower_uj=warm_lower,
        paired_warm_energy_uj=warm,
        paired_warm_upper_uj=warm_upper,
        paired_transition_lower_uj=transition_lower,
        paired_transition_energy_uj=transition,
        paired_transition_upper_uj=transition_upper,
        paired_lower_uj=warm_lower + transition_lower,
        paired_energy_uj=warm + transition,
        paired_upper_uj=warm_upper + transition_upper,
    )


def _adaptive_exact_decompositions(
    *,
    exact_route_qualified: bool,
    exact_parent,
    exact_candidate,
    baseline: AutomatedRouteCandidate,
    candidate: AutomatedRouteCandidate,
    warm: _AdaptiveWarmEstimate,
) -> _ExactRouteDecompositions:
    if not exact_route_qualified:
        return _ExactRouteDecompositions(False, None, None)
    assert exact_parent is not None
    assert exact_candidate is not None
    assert exact_parent.energy_lower_uj is not None
    assert exact_parent.energy_uj is not None
    assert exact_parent.energy_upper_uj is not None
    assert exact_candidate.energy_lower_uj is not None
    assert exact_candidate.energy_uj is not None
    assert exact_candidate.energy_upper_uj is not None
    parent = _apportion_measured_route_energy(
        exact_parent.energy_lower_uj,
        exact_parent.energy_uj,
        exact_parent.energy_upper_uj,
        warm.parent_warm_energy_uj,
        has_transition=bool(baseline.plan.transitions),
    )
    selected = _apportion_measured_route_energy(
        exact_candidate.energy_lower_uj,
        exact_candidate.energy_uj,
        exact_candidate.energy_upper_uj,
        warm.warm_energy_uj,
        has_transition=bool(candidate.plan.transitions),
    )
    return _ExactRouteDecompositions(
        parent is not None and selected is not None,
        parent,
        selected,
    )


def _adaptive_parent_history_candidate(
    *,
    baseline: AutomatedRouteCandidate,
    history,
    history_context: _AdaptiveHistoryContext,
    warm: _AdaptiveWarmEstimate,
    parent_transition: _TransitionEnergyEstimate,
    exact_decompositions: _ExactRouteDecompositions,
    exact_parent,
) -> AutomatedRouteCandidate:
    parent_cost = baseline.cost
    if exact_decompositions.qualified:
        assert exact_parent is not None
        assert exact_decompositions.parent is not None
        (
            warm_lower,
            warm_energy,
            warm_upper,
            transition_lower,
            transition_energy,
            transition_upper,
        ) = exact_decompositions.parent
        total_lower = exact_parent.energy_lower_uj
        total_energy = exact_parent.energy_uj
        total_upper = exact_parent.energy_upper_uj
        measured_service_us = exact_parent.service_us
        measured_service_upper_us = exact_parent.service_upper_us
    else:
        warm_lower = warm.parent_warm_lower_uj
        warm_energy = warm.parent_warm_energy_uj
        warm_upper = warm.parent_warm_upper_uj
        transition_lower = parent_transition.lower_uj
        transition_energy = parent_transition.energy_uj
        transition_upper = parent_transition.upper_uj
        total_lower = warm_lower + transition_lower
        total_energy = warm_energy + transition_energy
        total_upper = warm_upper + transition_upper
        measured_service_us = None
        measured_service_upper_us = None
    fixed_service_us = max(
        0, parent_cost.service_us - parent_cost.component_service_us
    )
    fixed_service_upper_us = max(
        fixed_service_us,
        parent_cost.service_upper_us - parent_cost.component_service_us,
    )
    service_us = fixed_service_us + warm.parent_component_service_us
    service_upper_us = max(
        service_us,
        fixed_service_upper_us + warm.parent_component_service_upper_us,
    )
    if measured_service_us is not None:
        service_us = measured_service_us
    if measured_service_upper_us is not None:
        service_upper_us = measured_service_upper_us
    parent_route_qualified = (
        parent_transition.qualified
        or exact_decompositions.qualified
        or history_context.history_uses_assumed_phone_power
    )
    learned_cost = replace(
        parent_cost,
        finish_us=parent_cost.start_us + service_us,
        finish_upper_us=parent_cost.start_us + service_upper_us,
        service_us=service_us,
        service_upper_us=service_upper_us,
        fleet_energy_lower_uj=total_lower,
        fleet_energy_uj=total_energy,
        fleet_energy_upper_uj=total_upper,
        component_service_us=warm.parent_component_service_us,
        warm_execution_energy_lower_uj=warm_lower,
        warm_execution_energy_uj=warm_energy,
        warm_execution_energy_upper_uj=warm_upper,
        transition_energy_lower_uj=transition_lower,
        transition_energy_uj=transition_energy,
        transition_energy_upper_uj=transition_upper,
        latency_evidence=history_context.history_latency_evidence,
        energy_evidence=(
            "ASSUMED"
            if history_context.history_uses_assumed_phone_power
            else "MEASURED"
            if parent_transition.qualified
            or exact_decompositions.qualified
            else "ASSUMED"
        ),
    )
    break_even = {
        **dict(baseline.residency_break_even or {}),
        "adaptive_history_evidence_match": history.evidence_match,
        "adaptive_history_source_layer_mask": history.source_layer_mask,
        "adaptive_history_target_layer_mask": history.target_layer_mask,
        "adaptive_history_evidence_scale_ppm": history.evidence_scale_ppm,
        **({
            "adaptive_history_operator_subset_proof": dict(
                history_context.subset_evidence
            ),
        } if history.evidence_match == "route_operator_subset_prior" else {}),
        **({
            "phone_energy_evidence": "ASSUMED_4P5W",
            "phone_power_profile_sha256s": list(
                history_context.assumed_phone_profile_sha256s
            ),
        } if history_context.history_uses_assumed_phone_power else {}),
    }
    return replace(
        baseline,
        cost=learned_cost,
        maturity="QUALIFIED" if parent_route_qualified else baseline.maturity,
        residency_break_even=break_even or None,
    )


def _adaptive_candidate_history_cost(
    *,
    candidate: AutomatedRouteCandidate,
    warm: _AdaptiveWarmEstimate,
    candidate_transition: _TransitionEnergyEstimate,
    exact_decompositions: _ExactRouteDecompositions,
    exact_candidate,
    history_context: _AdaptiveHistoryContext,
):
    cost = candidate.cost
    fixed_service_us = max(0, cost.service_us - cost.component_service_us)
    fixed_service_upper_us = max(
        fixed_service_us,
        cost.service_upper_us - cost.component_service_us,
    )
    service_us = fixed_service_us + warm.selected_component_service_us
    service_upper_us = max(
        service_us,
        fixed_service_upper_us + warm.selected_component_service_upper_us,
    )
    if exact_decompositions.qualified:
        assert exact_candidate is not None
        assert exact_decompositions.candidate is not None
        (
            warm_lower,
            warm_energy,
            warm_upper,
            transition_lower,
            transition_energy,
            transition_upper,
        ) = exact_decompositions.candidate
        total_lower = exact_candidate.energy_lower_uj
        total_energy = exact_candidate.energy_uj
        total_upper = exact_candidate.energy_upper_uj
        service_us = exact_candidate.service_us
        service_upper_us = exact_candidate.service_upper_us
    else:
        warm_lower = warm.warm_lower_uj
        warm_energy = warm.warm_energy_uj
        warm_upper = warm.warm_upper_uj
        transition_lower = candidate_transition.lower_uj
        transition_energy = candidate_transition.energy_uj
        transition_upper = candidate_transition.upper_uj
        total_lower = warm_lower + transition_lower
        total_energy = warm_energy + transition_energy
        total_upper = warm_upper + transition_upper
    return replace(
        cost,
        finish_us=cost.start_us + service_us,
        finish_upper_us=cost.start_us + service_upper_us,
        service_us=service_us,
        service_upper_us=service_upper_us,
        fleet_energy_lower_uj=total_lower,
        fleet_energy_uj=total_energy,
        fleet_energy_upper_uj=total_upper,
        warm_execution_energy_lower_uj=warm_lower,
        warm_execution_energy_uj=warm_energy,
        warm_execution_energy_upper_uj=warm_upper,
        transition_energy_lower_uj=transition_lower,
        transition_energy_uj=transition_energy,
        transition_energy_upper_uj=transition_upper,
        component_service_us=warm.selected_component_service_us,
        energy_evidence=(
            "ASSUMED"
            if history_context.history_uses_assumed_phone_power
            else "MEASURED"
            if candidate_transition.qualified
            or exact_decompositions.qualified
            else "ASSUMED"
        ),
        latency_evidence=history_context.history_latency_evidence,
    )
