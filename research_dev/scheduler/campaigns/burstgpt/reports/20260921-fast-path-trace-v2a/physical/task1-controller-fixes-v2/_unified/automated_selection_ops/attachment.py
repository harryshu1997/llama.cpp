"""AutomatedSelectionMixin attachment operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import json

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    HelperOpportunity,
    RuntimeExecutionPlan,
)
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from ..._internal.adaptive_decode_planning import (
    adaptive_desktop_control_is_qualified,
    adaptive_decode_policies,
)
from ..common import _DORMANT_PHONE_FFN_RUNTIME_PARAMETER


def _async_helper_layout_matches(
    envelope: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
    layout,
    manifest: ModelManifest,
) -> bool:
    geometry = envelope.plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    return not (
        geometry != layout.layout.geometry_sha256
        or layout.state not in {"PROPOSED", "PREPARING", "READY"}
        or not layout.covers_artifact(manifest.artifact_sha256)
        or envelope.paired_baseline_route_id
            != baseline.candidate_id
        or envelope.plan.desktop_placement_sha256
            != baseline.plan.desktop_placement_sha256
    )


def _async_helper_preparation_authorized(
    controller,
    baseline: AutomatedRouteCandidate,
    envelope: AutomatedRouteCandidate,
    selected: AutomatedRouteCandidate,
    layout,
    manifest: ModelManifest,
) -> bool:
    parent_lower = (
        baseline.cost.warm_execution_energy_lower_uj
        if baseline.cost.warm_execution_energy_lower_uj is not None
        else baseline.cost.fleet_energy_lower_uj
    )
    helper_upper = envelope.cost.warm_execution_energy_upper_uj
    latency_upper = max(
        1,
        envelope.cost.service_upper_us
            - envelope.cost.switching_us,
    )
    latency_limit = (
        baseline.cost.service_upper_us
        * controller._runtime_capabilities.maximum_latency_ppm
        // 1_000_000
    )
    break_even = envelope.residency_break_even or {}
    strict_selected = selected.candidate_id == envelope.candidate_id
    energy_positive = bool(
        parent_lower is not None
        and helper_upper is not None
        and helper_upper
            * 1_000_000
            <= parent_lower
                * (
                    1_000_000
                    - controller._runtime_capabilities
                        .minimum_energy_saving_ppm
                )
    )
    return bool(
        strict_selected
        or (
            energy_positive
            and latency_upper <= latency_limit
            and (
                layout.state in {"PREPARING", "READY"}
                or bool(break_even.get("passed", False))
                or bool(
                    break_even.get(
                        "phone_residency_portfolio_authorization"
                    )
                )
                or controller._phone_residency_portfolio_authorization(
                    envelope, manifest
                ) is not None
            )
        )
    )


def _bind_async_phone_helper(
    controller,
    candidate_set: AutomatedCandidateSet,
    baseline: AutomatedRouteCandidate,
    selected: AutomatedRouteCandidate,
    rejected: tuple[tuple[str, str], ...],
    envelope: AutomatedRouteCandidate,
    layout,
    manifest: ModelManifest,
) -> tuple[
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
]:
    helper_plan, helper_binding = controller._authorize_phone_helper_plan(
        envelope.plan,
        envelope.binding,
        layout,
        model_id=manifest.model_id,
        artifact_sha256=manifest.artifact_sha256,
    )
    helper = controller._build_helper_envelope(
        artifact_sha256=manifest.artifact_sha256,
        desktop_parent_route_id=baseline.candidate_id,
        desktop_placement_sha256=(
            baseline.plan.desktop_placement_sha256
        ),
        helper_plan=helper_plan,
        helper_binding=helper_binding,
        layout=layout,
    )
    base_plan = replace(baseline.plan, helper_envelope=helper)
    base_binding = replace(
        baseline.binding,
        operator_plan_sha256=base_plan.plan_sha256,
    )
    desktop = replace(
        baseline,
        plan=base_plan,
        binding=base_binding,
    )
    candidate_set = replace(
        candidate_set,
        candidates=tuple(
            desktop if row.candidate_id == desktop.candidate_id else row
            for row in candidate_set.candidates
        ),
    )
    rejected_by_route = dict(rejected)
    rejected_by_route.pop(desktop.candidate_id, None)
    if selected.candidate_id != desktop.candidate_id:
        rejected_by_route[selected.candidate_id] = (
            "ASYNC_HELPER_PREPARATION"
        )
    return (
        candidate_set,
        desktop,
        tuple(sorted(rejected_by_route.items())),
        "READY_DESKTOP_WITH_ASYNC_PHONE_HELPER",
    )


def _has_dormant_phone_ffn_runtime(
    plan: RuntimeExecutionPlan,
) -> bool:
    raw = plan.adapter_parameters.get(
        _DORMANT_PHONE_FFN_RUNTIME_PARAMETER
    )
    return type(raw) is str and bool(raw)


def _dormant_phone_ffn_runtime_supports(
    resident_plan: RuntimeExecutionPlan,
    requested_contract: str,
) -> bool:
    """Check that a live server's dormant FFN runtime is a superset."""

    resident_contract = resident_plan.adapter_parameters.get(
        _DORMANT_PHONE_FFN_RUNTIME_PARAMETER
    )
    if (
        type(resident_contract) is not str
        or type(requested_contract) is not str
    ):
        return False
    try:
        resident = json.loads(resident_contract)
        requested = json.loads(requested_contract)
    except (TypeError, ValueError):
        return False
    if type(resident) is not dict or type(requested) is not dict:
        return False
    resident_mask = resident.pop("ffn_resident_layer_mask", None)
    requested_mask = requested.pop("ffn_resident_layer_mask", None)
    return bool(
        type(resident_mask) is int
        and type(requested_mask) is int
        and requested_mask > 0
        and requested_mask & ~resident_mask == 0
        and resident == requested
    )


def _compact_helper_opportunities(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
) -> tuple[HelperOpportunity, ...]:
    """Retain only the executable helper permission needed at runtime."""

    if controller._runtime_capabilities is None:
        return ()
    try:
        _baseline_policy, policies, envelope = adaptive_decode_policies(
            candidate_set,
            controller.runtime_model_manifest(candidate_set.model_id),
            controller._runtime_capabilities,
            request.output_tokens,
            controller._maximum_phone_sessions,
        )
    except AdaptiveDecodeError:
        return ()
    if envelope is None or not policies:
        return ()
    baseline = candidate_set.baseline
    plan = envelope.plan
    geometry = plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    required_resources = tuple(sorted(
        set(plan.resource_ids) - set(baseline.plan.resource_ids)
    ))
    if (
        type(geometry) is not str
        or not required_resources
        or envelope.paired_baseline_route_id
            != baseline.candidate_id
        or plan.desktop_placement_sha256
            != baseline.plan.desktop_placement_sha256
    ):
        return ()
    layout = controller._model_placement_controller.planning_phone_layout()
    layout_generation = (
        layout.generation
        if layout is not None
        and layout.layout.geometry_sha256 == geometry
        else None
    )
    evidence_state = (
        "TRUSTED"
        if adaptive_desktop_control_is_qualified(
            candidate_set, controller._runtime_capabilities
        )
        and envelope.maturity == "QUALIFIED"
        and envelope.cost.energy_evidence in {"CALIBRATED", "MEASURED"}
        and not {"ROUTE_NOT_QUALIFIED", "MARGINAL_SYSTEM_COST_UNKNOWN"}.intersection(
            envelope.rejection_reasons
        )
        else "LEARNING"
    )
    try:
        opportunity = HelperOpportunity(
            route_family_identity=plan.route_family,
            desktop_parent_route_id=baseline.candidate_id,
            desktop_parent_placement_sha256=str(
                baseline.plan.desktop_placement_sha256
            ),
            helper_operator_plan=plan,
            helper_binding=envelope.binding,
            phone_layout_generation=layout_generation,
            phone_layout_geometry_sha256=geometry,
            required_resource_ids=required_resources,
            evidence_state=evidence_state,
            maximum_allowed_fraction_ppm=max(
                plan.execution_contract
                    .allowed_adaptive_fractions_ppm
            ),
        )
    except ValueError:
        return ()
    return (opportunity,)
