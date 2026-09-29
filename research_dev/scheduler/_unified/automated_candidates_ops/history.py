"""AutomatedCandidateMixin history operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.route_generation import AutomatedRouteCompiler, RouteGenerationError
from ..._internal.runtime_learning import RuntimeLearningError
from ..._internal.runtime_plan import AutomatedRouteCandidate
from ..._internal.types import canonical_sha256
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from .common import _ADAPTIVE_HISTORY_ALLOWED_REJECTIONS, _AdaptiveHistoryContext


def _adaptive_candidate_accepts_history(
    controller,
    candidate: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
    contract: Mapping[str, object] | None,
) -> bool:
    assert controller._runtime_capabilities is not None
    coordinator = (
        controller._runtime_capabilities.composite_executor_by_id.get(
            candidate.binding.executor_id
        )
    )
    return not (
        contract is None
        or not contract.get("adaptive_envelope", False)
        or candidate.paired_baseline_route_id != baseline.candidate_id
        or candidate.plan.desktop_placement_sha256
        != baseline.plan.desktop_placement_sha256
        or set(candidate.rejection_reasons)
        - _ADAPTIVE_HISTORY_ALLOWED_REJECTIONS
        or candidate.binding.endpoint is None
        or candidate.binding.operator_plan_protocol is None
        or coordinator is None
        or coordinator.maturity not in {"SHADOW", "QUALIFIED"}
        or not coordinator.evidence_ids
        or any(
            controller._runtime_capabilities.executor_by_device[
                device_id
            ].maturity != "QUALIFIED"
            for device_id in candidate.device_ids
        )
        or any(
            transition.maturity != "QUALIFIED"
            for transition in candidate.plan.transitions
        )
    )


def _assumed_phone_history_query(
    controller,
    *,
    manifest: ModelManifest,
    request: Request,
    component_capability_sha256: str,
    baseline_policy: AdaptiveDecodePolicy,
    policies: tuple[AdaptiveDecodePolicy, ...],
    adaptive_config,
    phone_power_by_domain: Mapping[str, tuple[int, int]],
    route_geometry_prior: bool = False,
    subset_source_layer_mask: int | None = None,
):
    assert controller._runtime_capability_generation_sha256 is not None
    return controller._adaptive_decode.historical_route_estimate_with_assumed_phone_power(
        model_artifact_sha256=manifest.artifact_sha256,
        planning_profile_sha256=controller._runtime_capability_generation_sha256,
        component_capability_sha256=component_capability_sha256,
        baseline=baseline_policy,
        candidates=policies,
        context_length=request.input_tokens,
        active_batch=1,
        config=adaptive_config,
        phone_power_by_domain=phone_power_by_domain,
        route_geometry_prior=route_geometry_prior,
        operator_subset_prior=subset_source_layer_mask is not None,
        operator_subset_source_layer_mask=subset_source_layer_mask,
    )


def _adaptive_physical_latency(
    controller,
    *,
    manifest: ModelManifest,
    request: Request,
    component_capability_sha256: str,
    baseline_policy: AdaptiveDecodePolicy,
    policies: tuple[AdaptiveDecodePolicy, ...],
    selected_policy: AdaptiveDecodePolicy,
    adaptive_config,
) -> tuple[tuple[int, int, int] | None, AdaptiveDecodePolicy | None]:
    assert controller._runtime_capability_generation_sha256 is not None
    ordered_policies = tuple(dict.fromkeys((
        selected_policy,
        *(row for row in policies if not row.baseline),
    )))
    for policy in ordered_policies:
        physical_latency = controller._adaptive_decode.historical_component_latency(
            model_artifact_sha256=manifest.artifact_sha256,
            planning_profile_sha256=(
                controller._runtime_capability_generation_sha256
            ),
            component_capability_sha256=component_capability_sha256,
            baseline=baseline_policy,
            candidates=policies,
            selected_policy=policy,
            context_length=request.input_tokens,
            active_batch=1,
            config=adaptive_config,
        )
        if physical_latency is not None:
            return physical_latency, policy
    return None, None


def _adaptive_history_context(
    controller,
    *,
    candidate: AutomatedRouteCandidate,
    baseline_policy: AdaptiveDecodePolicy,
    contract: Mapping[str, object],
    request: Request,
    manifest: ModelManifest,
    compiler: AutomatedRouteCompiler,
) -> _AdaptiveHistoryContext:
    assert controller._runtime_capabilities is not None
    assert controller._runtime_capability_generation_sha256 is not None
    empty = _AdaptiveHistoryContext(
        None, None, None, False, False, "MEASURED", None, ()
    )
    try:
        selected_policy = AdaptiveDecodePolicy.from_json(contract)
        probe_values = contract.get("adaptive_decode_probe_contracts", [])
        if type(probe_values) is not list:
            raise AdaptiveDecodeError("adaptive probe contracts are invalid")
        policies = tuple(
            AdaptiveDecodePolicy.from_json(value) for value in probe_values
        )
        if all(
            policy.policy_hash != selected_policy.policy_hash
            for policy in policies
        ):
            policies += (selected_policy,)
        component_sha256 = compiler.component_capability_identity(
            candidate.plan, candidate.binding.executor_id
        )
        phone_profiles = tuple(
            controller._runtime_capabilities.phone_power_profile_by_device[device_id]
            for device_id in candidate.device_ids
            if device_id
            in controller._runtime_capabilities.phone_power_profile_by_device
        )
        assumed_sha256s = tuple(sorted(
            canonical_sha256(profile) for profile in phone_profiles
        ))
        use_assumed = (
            bool(phone_profiles)
            and candidate.cost.energy_evidence == "ASSUMED"
            and all(row.allow_assumed_for_scheduling for row in phone_profiles)
        )
        adaptive_config = replace(
            controller._adaptive_decode_config,
            minimum_energy_saving_ppm=(
                controller._runtime_capabilities.minimum_energy_saving_ppm
            ),
            maximum_latency_ppm=(
                controller._runtime_capabilities.maximum_latency_ppm
            ),
        )
        if not use_assumed:
            history = controller._adaptive_decode.historical_route_estimate(
                model_artifact_sha256=manifest.artifact_sha256,
                planning_profile_sha256=(
                    controller._runtime_capability_generation_sha256
                ),
                component_capability_sha256=component_sha256,
                baseline=baseline_policy,
                candidates=policies,
                context_length=request.input_tokens,
                active_batch=1,
                config=adaptive_config,
            )
            return replace(empty, history=history)
        power_by_domain = {
            row.domain_id: (row.active_power_mw, row.idle_power_mw)
            for row in phone_profiles
        }
        history = controller._assumed_phone_history_query(
            manifest=manifest,
            request=request,
            component_capability_sha256=component_sha256,
            baseline_policy=baseline_policy,
            policies=policies,
            adaptive_config=adaptive_config,
            phone_power_by_domain=power_by_domain,
        )
        if history is None:
            history = controller._assumed_phone_history_query(
                manifest=manifest,
                request=request,
                component_capability_sha256=component_sha256,
                baseline_policy=baseline_policy,
                policies=policies,
                adaptive_config=adaptive_config,
                phone_power_by_domain=power_by_domain,
                route_geometry_prior=True,
            )
        subset_evidence = None
        if history is None:
            subset_evidence = compiler.phone_residency_subset_evidence(
                manifest, candidate
            )
            if subset_evidence is not None:
                history = controller._assumed_phone_history_query(
                    manifest=manifest,
                    request=request,
                    component_capability_sha256=component_sha256,
                    baseline_policy=baseline_policy,
                    policies=policies,
                    adaptive_config=adaptive_config,
                    phone_power_by_domain=power_by_domain,
                    subset_source_layer_mask=int(
                        subset_evidence["source_layer_mask"]
                    ),
                )
        physical_latency = None
        physical_policy = None
        if history is None:
            physical_latency, physical_policy = (
                controller._adaptive_physical_latency(
                    manifest=manifest,
                    request=request,
                    component_capability_sha256=component_sha256,
                    baseline_policy=baseline_policy,
                    policies=policies,
                    selected_policy=selected_policy,
                    adaptive_config=adaptive_config,
                )
            )
        evidence = (
            "CALIBRATED"
            if history is not None
            and history.evidence_match in {
                "route_geometry_prior",
                "component_operator_subset_prior",
                "route_operator_subset_prior",
            }
            else "MEASURED"
        )
        return _AdaptiveHistoryContext(
            history=history,
            physical_latency=physical_latency,
            physical_latency_policy=physical_policy,
            use_assumed_phone_power=True,
            history_uses_assumed_phone_power=history is not None,
            history_latency_evidence=evidence,
            subset_evidence=subset_evidence,
            assumed_phone_profile_sha256s=assumed_sha256s,
        )
    except AdaptiveDecodeError:
        return empty


def _apply_physical_latency_history(
    candidate: AutomatedRouteCandidate,
    request: Request,
    context: _AdaptiveHistoryContext,
) -> AutomatedRouteCandidate | None:
    if (
        not context.use_assumed_phone_power
        or context.physical_latency is None
    ):
        return None
    latency_per_token_us, latency_upper_per_token_us, groups = (
        context.physical_latency
    )
    cost = candidate.cost
    fixed_service_us = max(0, cost.service_us - cost.component_service_us)
    fixed_service_upper_us = max(
        fixed_service_us,
        cost.service_upper_us - cost.component_service_us,
    )
    observed_component_us = max(
        cost.component_service_us,
        latency_per_token_us * request.output_tokens,
    )
    observed_component_upper_us = max(
        observed_component_us,
        latency_upper_per_token_us * request.output_tokens,
    )
    service_us = fixed_service_us + observed_component_us
    service_upper_us = max(
        service_us, fixed_service_upper_us + observed_component_upper_us
    )
    learned_cost = replace(
        cost,
        component_service_us=observed_component_us,
        finish_us=cost.start_us + service_us,
        finish_upper_us=cost.start_us + service_upper_us,
        latency_evidence="MEASURED",
        service_us=service_us,
        service_upper_us=service_upper_us,
    )
    reasons = set(candidate.rejection_reasons)
    reasons.discard("ROUTE_NOT_QUALIFIED")
    reasons.discard("SLO_UPPER_BOUND")
    if learned_cost.finish_upper_us > request.deadline_us:
        reasons.add("SLO_UPPER_BOUND")
    rejection_reasons = tuple(sorted(reasons))
    break_even = {
        **dict(candidate.residency_break_even or {}),
        "physical_component_group_count": groups,
        "physical_component_latency_per_token_us": latency_per_token_us,
        "physical_component_latency_upper_per_token_us": (
            latency_upper_per_token_us
        ),
        "physical_component_policy_hash": (
            None
            if context.physical_latency_policy is None
            else context.physical_latency_policy.policy_hash
        ),
        "phone_energy_evidence": "ASSUMED_4P5W",
    }
    return replace(
        candidate,
        admitted=not rejection_reasons,
        binding=replace(
            candidate.binding,
            ready=not rejection_reasons,
            eligibility_reasons=rejection_reasons,
        ),
        cost=learned_cost,
        maturity="QUALIFIED",
        rejection_reasons=rejection_reasons,
        residency_break_even=break_even,
    )


def _with_measured_transitions(
    compiler: AutomatedRouteCompiler,
    manifest: ModelManifest,
    route: AutomatedRouteCandidate,
) -> AutomatedRouteCandidate:
    estimates = compiler.transition_estimates_for_plan(
        manifest, route.plan, route.binding.executor_id
    )
    if not estimates or not any(
        estimate is not None for estimate in estimates
    ):
        return route
    transitions = tuple(
        transition
        if estimate is None
        else replace(
            transition,
            latency_us=(
                estimate.latency_us
                if estimate.latency_maturity == "QUALIFIED"
                else transition.latency_us
            ),
            energy_uj=(
                estimate.energy_uj
                if estimate.energy_maturity == "QUALIFIED"
                and estimate.energy_uj is not None
                else transition.energy_uj
            ),
            maturity=(
                estimate.maturity
                if estimate.maturity in {"QUALIFIED", "QUARANTINED"}
                else transition.maturity
            ),
            energy_maturity=(
                estimate.energy_maturity
                if estimate.energy_maturity in {
                    "QUALIFIED", "QUARANTINED"
                }
                else transition.energy_maturity
            ),
        )
        for transition, estimate in zip(route.plan.transitions, estimates)
    )
    if transitions == route.plan.transitions:
        return route
    plan = replace(route.plan, transitions=transitions)
    return replace(
        route,
        plan=plan,
        binding=replace(
            route.binding, operator_plan_sha256=plan.plan_sha256
        ),
    )


def _exact_adaptive_route_history(
    compiler: AutomatedRouteCompiler,
    manifest: ModelManifest,
    request: Request,
    cost_features: Mapping[str, int],
    baseline: AutomatedRouteCandidate,
    candidate: AutomatedRouteCandidate,
    history_uses_assumed_phone_power: bool,
) -> tuple[object | None, object | None, bool]:
    try:
        exact_parent = compiler.exact_route_estimate(
            manifest,
            request,
            baseline.plan,
            cost_features,
            baseline.binding.executor_id,
        )
        exact_candidate = compiler.exact_route_estimate(
            manifest,
            request,
            candidate.plan,
            cost_features,
            candidate.binding.executor_id,
        )
    except (RouteGenerationError, RuntimeLearningError):
        exact_parent = None
        exact_candidate = None
    qualified = all(
        estimate is not None
        and estimate.energy_scope == "route_total"
        and estimate.latency_maturity == "QUALIFIED"
        and estimate.energy_maturity == "QUALIFIED"
        and estimate.energy_lower_uj is not None
        and estimate.energy_uj is not None
        and estimate.energy_upper_uj is not None
        for estimate in (exact_parent, exact_candidate)
    )
    qualified = (
        qualified
        and not any(
            transition.evictions
            for route in (baseline, candidate)
            for transition in route.plan.transitions
        )
        and not history_uses_assumed_phone_power
    )
    return exact_parent, exact_candidate, qualified
