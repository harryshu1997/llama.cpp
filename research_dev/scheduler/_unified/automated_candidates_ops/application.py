"""AutomatedCandidateMixin application operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.route_generation import AutomatedRouteCompiler
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from ..._internal.adaptive_decode_planning import adaptive_probe_contracts
from .common import (
    _AdaptiveCandidateApplication,
    _AdaptiveCohortBounds,
    _AdaptiveEnergyDeltas,
    _AdaptiveHistoryContext,
    _AdaptiveWarmEstimate,
    _ExactRouteDecompositions,
    _TransitionEnergyEstimate,
)


def _adaptive_cohort_bounds(
    *,
    candidate: AutomatedRouteCandidate,
    history_context: _AdaptiveHistoryContext,
    warm: _AdaptiveWarmEstimate,
    candidate_transition: _TransitionEnergyEstimate,
    parent_history_candidate: AutomatedRouteCandidate,
    energy_deltas: _AdaptiveEnergyDeltas,
    exact_decompositions: _ExactRouteDecompositions,
    exact_parent,
    exact_candidate,
) -> _AdaptiveCohortBounds:
    expected_uses = int(
        (candidate.residency_break_even or {}).get("expected_use_count", 1)
    )
    restore_energy = (
        candidate_transition.energy_uj
        if any(row.evictions for row in candidate.plan.transitions)
        else 0
    )
    if exact_decompositions.qualified:
        assert exact_candidate is not None
        assert exact_parent is not None
        assert exact_candidate.energy_upper_uj is not None
        assert exact_parent.energy_lower_uj is not None
        candidate_upper = (
            exact_candidate.energy_upper_uj
            + warm.warm_upper_uj * max(0, expected_uses - 1)
            + restore_energy
        )
        parent_lower = (
            exact_parent.energy_lower_uj
            + warm.parent_warm_lower_uj * max(0, expected_uses - 1)
        )
    else:
        candidate_upper = (
            warm.warm_upper_uj * expected_uses
            + candidate_transition.upper_uj
            + restore_energy
        )
        parent_transition_lower = (
            parent_history_candidate.cost.transition_energy_lower_uj
        )
        assert parent_transition_lower is not None
        parent_lower = (
            warm.parent_warm_lower_uj * expected_uses
            + parent_transition_lower
        )
    parent_total_lower = (
        parent_history_candidate.cost.fleet_energy_lower_uj
    )
    assert parent_total_lower is not None
    paired_candidate_upper = parent_total_lower + energy_deltas.paired_upper_uj
    paired_warm_upper = (
        warm.parent_warm_lower_uj + energy_deltas.paired_warm_upper_uj
    )
    paired_cohort_upper = (
        parent_lower
        + energy_deltas.paired_upper_uj
        + energy_deltas.paired_warm_upper_uj
        * max(0, expected_uses - 1)
        + restore_energy
    )
    return _AdaptiveCohortBounds(
        candidate_upper_uj=candidate_upper,
        parent_lower_uj=parent_lower,
        paired_candidate_upper_uj=paired_candidate_upper,
        paired_warm_candidate_upper_uj=paired_warm_upper,
        paired_cohort_upper_uj=paired_cohort_upper,
        route_energy_qualified=(
            candidate_transition.qualified
            or exact_decompositions.qualified
            or history_context.history_uses_assumed_phone_power
        ),
    )


def _adaptive_break_even(
    _controller_class,
    *,
    baseline: AutomatedRouteCandidate,
    candidate: AutomatedRouteCandidate,
    history,
    history_context: _AdaptiveHistoryContext,
    warm: _AdaptiveWarmEstimate,
    candidate_transition: _TransitionEnergyEstimate,
    parent_history_candidate: AutomatedRouteCandidate,
    energy_deltas: _AdaptiveEnergyDeltas,
    exact_decompositions: _ExactRouteDecompositions,
    exact_parent,
    exact_candidate,
) -> tuple[dict[str, object], bool]:
    bounds = _controller_class._adaptive_cohort_bounds(
        candidate=candidate,
        history_context=history_context,
        warm=warm,
        candidate_transition=candidate_transition,
        parent_history_candidate=parent_history_candidate,
        energy_deltas=energy_deltas,
        exact_decompositions=exact_decompositions,
        exact_parent=exact_parent,
        exact_candidate=exact_candidate,
    )
    break_even = {
        **dict(candidate.residency_break_even or {}),
        "adaptive_history_applied": True,
        "adaptive_history_baseline_group_count": history.baseline_group_count,
        "adaptive_history_energy_boundary_id": history.energy_boundary_id,
        "adaptive_history_policy_hash": history.selected_policy.policy_hash,
        "adaptive_history_selected_fraction_ppm": (
            history.selected_policy.split_fraction_ppm
        ),
        "adaptive_history_selected_group_count": history.selected_group_count,
        "adaptive_history_requested_context_bucket": (
            history.requested_context_bucket
        ),
        "adaptive_history_evidence_context_bucket": (
            history.evidence_context_bucket
        ),
        "adaptive_history_evidence_match": history.evidence_match,
        "adaptive_history_source_layer_mask": history.source_layer_mask,
        "adaptive_history_target_layer_mask": history.target_layer_mask,
        "adaptive_history_evidence_scale_ppm": history.evidence_scale_ppm,
        "exact_route_history_applied": exact_decompositions.qualified,
        "exact_route_history_candidate_profile_id": (
            None if exact_candidate is None else exact_candidate.profile_id
        ) if exact_decompositions.qualified else None,
        "exact_route_history_parent_profile_id": (
            None if exact_parent is None else exact_parent.profile_id
        ) if exact_decompositions.qualified else None,
        "exact_route_history_energy_attribution": (
            "route_total_proportional_decomposition"
            if exact_decompositions.qualified else None
        ),
        "absolute_candidate_cohort_upper_uj": bounds.candidate_upper_uj,
        "candidate_cohort_upper_uj": bounds.paired_cohort_upper_uj,
        "desktop_cohort_lower_uj": bounds.parent_lower_uj,
        "paired_energy_boundary_id": history.energy_boundary_id,
        "paired_energy_delta_lower_uj": energy_deltas.paired_lower_uj,
        "paired_energy_delta_uj": energy_deltas.paired_energy_uj,
        "paired_energy_delta_upper_uj": energy_deltas.paired_upper_uj,
        "paired_energy_evidence": (
            "ASSUMED_4P5W"
            if history_context.history_uses_assumed_phone_power
            else "MEASURED"
            if bounds.route_energy_qualified else "ABSENT"
        ),
        "paired_energy_parent_placement_sha256": (
            baseline.plan.desktop_placement_sha256
        ),
        "paired_energy_parent_route_id": baseline.candidate_id,
        "paired_route_candidate_upper_uj": (
            bounds.paired_candidate_upper_uj
        ),
        "paired_transition_energy_delta_lower_uj": (
            energy_deltas.paired_transition_lower_uj
        ),
        "paired_transition_energy_delta_uj": (
            energy_deltas.paired_transition_energy_uj
        ),
        "paired_transition_energy_delta_upper_uj": (
            energy_deltas.paired_transition_upper_uj
        ),
        "paired_warm_candidate_upper_uj": (
            bounds.paired_warm_candidate_upper_uj
        ),
        "paired_warm_energy_delta_lower_uj": (
            energy_deltas.paired_warm_lower_uj
        ),
        "paired_warm_energy_delta_uj": energy_deltas.paired_warm_energy_uj,
        "paired_warm_energy_delta_upper_uj": (
            energy_deltas.paired_warm_upper_uj
        ),
        "paired_warm_energy_delta_lower_per_decode_token_uj": (
            history.selected_energy_lower_per_token_uj
            - history.baseline_energy_upper_per_token_uj
        ),
        "paired_warm_energy_delta_per_decode_token_uj": (
            history.selected_energy_per_token_uj
            - history.baseline_energy_per_token_uj
        ),
        "paired_warm_energy_delta_upper_per_decode_token_uj": (
            history.selected_energy_upper_per_token_uj
            - history.baseline_energy_lower_per_token_uj
        ),
        "paired_warm_energy_normalization": "decode_token",
        "passed": bounds.route_energy_qualified
        and bounds.paired_cohort_upper_uj <= bounds.parent_lower_uj,
        **({
            "phone_energy_evidence": "ASSUMED_4P5W",
            "phone_power_profile_sha256s": list(
                history_context.assumed_phone_profile_sha256s
            ),
        } if history_context.history_uses_assumed_phone_power else {}),
        "warm_route_upper_uj": warm.warm_upper_uj,
    }
    return break_even, bounds.route_energy_qualified


def _candidate_with_adaptive_history(
    candidate: AutomatedRouteCandidate,
    learned_cost,
    break_even: Mapping[str, object],
    route_energy_qualified: bool,
) -> AutomatedRouteCandidate:
    reasons = set(candidate.rejection_reasons)
    reasons.discard("ENERGY_UNKNOWN")
    reasons.discard("SLO_UPPER_BOUND")
    if route_energy_qualified:
        reasons.discard("ROUTE_NOT_QUALIFIED")
        if break_even["passed"]:
            reasons.discard("COLD_RESIDENCY_BREAK_EVEN")
        else:
            reasons.add("COLD_RESIDENCY_BREAK_EVEN")
    else:
        reasons.add("ROUTE_NOT_QUALIFIED")
    rejection_reasons = tuple(sorted(reasons))
    admitted = route_energy_qualified and not rejection_reasons
    return replace(
        candidate,
        binding=replace(
            candidate.binding,
            ready=admitted,
            eligibility_reasons=rejection_reasons,
        ),
        cost=learned_cost,
        maturity="QUALIFIED" if route_energy_qualified else "SHADOW",
        admitted=admitted,
        rejection_reasons=rejection_reasons,
        residency_break_even=break_even,
    )


def _apply_adaptive_history_candidate(
    controller,
    *,
    candidate: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
    baseline_policy: AdaptiveDecodePolicy,
    contract: Mapping[str, object],
    request: Request,
    manifest: ModelManifest,
    cost_features: Mapping[str, int],
    compiler: AutomatedRouteCompiler,
) -> _AdaptiveCandidateApplication:
    assert controller._runtime_capabilities is not None
    history_context = controller._adaptive_history_context(
        candidate=candidate,
        baseline_policy=baseline_policy,
        contract=contract,
        request=request,
        manifest=manifest,
        compiler=compiler,
    )
    physical_candidate = controller._apply_physical_latency_history(
        candidate, request, history_context
    )
    if physical_candidate is not None:
        return _AdaptiveCandidateApplication(
            baseline, physical_candidate, None, True
        )
    history = history_context.history
    if (
        history is None
        or history.energy_boundary_id
        != controller._runtime_capabilities.placement_profile.energy_boundary_id
    ):
        return _AdaptiveCandidateApplication(
            baseline, candidate, None, False
        )
    exact_parent, exact_candidate, exact_qualified = (
        controller._exact_adaptive_route_history(
            compiler,
            manifest,
            request,
            cost_features,
            baseline,
            candidate,
            history_context.history_uses_assumed_phone_power,
        )
    )
    baseline = controller._with_measured_transitions(
        compiler, manifest, baseline
    )
    candidate = controller._with_measured_transitions(
        compiler, manifest, candidate
    )
    parent_cost = baseline.cost
    cost = candidate.cost
    decomposition = (
        parent_cost.warm_execution_energy_lower_uj,
        parent_cost.warm_execution_energy_uj,
        parent_cost.warm_execution_energy_upper_uj,
        cost.transition_energy_lower_uj,
        cost.transition_energy_uj,
        cost.transition_energy_upper_uj,
    )
    if any(value is None for value in decomposition):
        return _AdaptiveCandidateApplication(
            baseline, candidate, None, False
        )
    warm = controller._adaptive_warm_estimate(
        parent_cost, history, request.output_tokens
    )
    assert cost.transition_energy_lower_uj is not None
    assert cost.transition_energy_uj is not None
    assert cost.transition_energy_upper_uj is not None
    candidate_transition = controller._adaptive_transition_energy(
        compiler,
        manifest,
        candidate,
        (
            cost.transition_energy_lower_uj,
            cost.transition_energy_uj,
            cost.transition_energy_upper_uj,
        ),
    )
    parent_transition = controller._adaptive_transition_energy(
        compiler,
        manifest,
        baseline,
        (
            parent_cost.transition_energy_lower_uj or 0,
            parent_cost.transition_energy_uj or 0,
            parent_cost.transition_energy_upper_uj or 0,
        ),
    )
    energy_deltas = controller._adaptive_energy_deltas(
        history,
        request.output_tokens,
        candidate_transition,
        parent_transition,
    )
    exact = controller._adaptive_exact_decompositions(
        exact_route_qualified=exact_qualified,
        exact_parent=exact_parent,
        exact_candidate=exact_candidate,
        baseline=baseline,
        candidate=candidate,
        warm=warm,
    )
    learned_baseline = controller._adaptive_parent_history_candidate(
        baseline=baseline,
        history=history,
        history_context=history_context,
        warm=warm,
        parent_transition=parent_transition,
        exact_decompositions=exact,
        exact_parent=exact_parent,
    )
    learned_cost = controller._adaptive_candidate_history_cost(
        candidate=candidate,
        warm=warm,
        candidate_transition=candidate_transition,
        exact_decompositions=exact,
        exact_candidate=exact_candidate,
        history_context=history_context,
    )
    break_even, energy_qualified = controller._adaptive_break_even(
        baseline=baseline,
        candidate=candidate,
        history=history,
        history_context=history_context,
        warm=warm,
        candidate_transition=candidate_transition,
        parent_history_candidate=learned_baseline,
        energy_deltas=energy_deltas,
        exact_decompositions=exact,
        exact_parent=exact_parent,
        exact_candidate=exact_candidate,
    )
    return _AdaptiveCandidateApplication(
        baseline,
        controller._candidate_with_adaptive_history(
            candidate, learned_cost, break_even, energy_qualified
        ),
        learned_baseline,
        True,
    )


def _apply_adaptive_history_costs(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest: ModelManifest,
    cost_features: Mapping[str, int] | None = None,
) -> AutomatedCandidateSet:
    if (
        controller._runtime_capabilities is None
        or controller._runtime_capability_generation_sha256 is None
    ):
        return candidate_set
    cost_features = {} if cost_features is None else cost_features
    compiler = controller._automated_compiler()
    try:
        contracts = adaptive_probe_contracts(
            candidate_set,
            manifest,
            controller._runtime_capabilities,
            request.output_tokens,
        )
        baseline_contract = contracts.get(
            candidate_set.baseline_route_id
        )
        if baseline_contract is None:
            return candidate_set
        baseline_policy = AdaptiveDecodePolicy.from_json(
            baseline_contract
        )
    except AdaptiveDecodeError:
        return candidate_set
    baseline = candidate_set.baseline
    rows = []
    changed = False
    learned_baseline = baseline
    learned_baseline_applied = False
    live_route_values = candidate_set.search_metadata.get(
        "route_template_live_route_ids"
    )
    live_route_ids = (
        None
        if type(live_route_values) not in {list, tuple}
        else frozenset(live_route_values)
    )
    for candidate in candidate_set.candidates:
        if (
            live_route_ids is not None
            and candidate.candidate_id not in live_route_ids
        ):
            rows.append(candidate)
            continue
        contract = contracts.get(candidate.candidate_id)
        if not controller._adaptive_candidate_accepts_history(
            candidate, baseline, contract
        ):
            rows.append(candidate)
            continue
        assert contract is not None
        application = controller._apply_adaptive_history_candidate(
            candidate=candidate,
            baseline=baseline,
            baseline_policy=baseline_policy,
            contract=contract,
            request=request,
            manifest=manifest,
            cost_features=cost_features,
            compiler=compiler,
        )
        baseline = application.baseline
        rows.append(application.candidate)
        parent = application.learned_baseline
        if parent is not None and (
            not learned_baseline_applied
            or parent.cost.fleet_energy_lower_uj
            < learned_baseline.cost.fleet_energy_lower_uj
        ):
            learned_baseline = parent
            learned_baseline_applied = True
        changed = changed or application.changed
    if not changed:
        return candidate_set
    return replace(candidate_set, candidates=tuple(
        learned_baseline
        if row.candidate_id == baseline.candidate_id else row
        for row in rows
    ))
