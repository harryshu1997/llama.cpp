"""AutomatedSelectionMixin materialization operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from ..._internal.model_placement_controller import ModelDemandSnapshot, ModelPlacementAction
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_resources import RuntimeResidencyProjectionToken
from ..._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from .common import _ModelPlacementMaterializeContext, _ModelPlacementPass


def _materialize_model_placement_candidate(
    controller,
    *,
    candidate_set: AutomatedCandidateSet,
    prospective: AutomatedRouteCandidate,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    selection_mode: str,
    invalidation_reason: str,
    runtime_rejections: Mapping[str, str],
    current_epoch: RuntimeModelPlacementEpoch | None,
    demand_snapshot: ModelDemandSnapshot,
    placement_action: ModelPlacementAction,
    residency_holds: Mapping[str, object],
    route_compiler: AutomatedRouteCompiler | None = None,
    memory_source_snapshot: HeterogeneousRuntimeSnapshot | None = None,
    memory_not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
    memory_projection_token: (
        RuntimeResidencyProjectionToken | None
    ) = None,
) -> tuple[
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
    RuntimeModelPlacementEpoch | None,
    RuntimeRouteTemplateSet | None,
]:
    compiler = (
        controller._automated_compiler()
        if route_compiler is None else route_compiler
    )
    proposed_epoch, proposed_templates = (
        controller._propose_model_placement_epoch(
        request=request,
        manifest=manifest,
        candidate_set=candidate_set,
        selected=prospective,
        observed_at_us=observed_at_us,
        selection_mode=selection_mode,
        invalidation_reason=invalidation_reason,
        snapshot=snapshot,
        route_compiler=compiler,
        demand_snapshot=demand_snapshot,
        placement_action=placement_action,
        current_epoch=current_epoch,
    ))
    epoch, templates, publish_epoch, publish_templates = (
        controller._retained_model_placement_epoch(
            current_epoch,
            prospective,
            proposed_epoch,
            proposed_templates,
        )
    )
    context = _ModelPlacementMaterializeContext(
        compiler=compiler,
        request=request,
        manifest=manifest,
        snapshot=snapshot,
        observed_at_us=observed_at_us,
        selection_mode=selection_mode,
        invalidation_reason=invalidation_reason,
        residency_holds=residency_holds,
        memory_source_snapshot=memory_source_snapshot,
        memory_not_before_by_resource=memory_not_before_by_resource,
        memory_projection_token=memory_projection_token,
    )
    resolution_passes: list = []
    first = controller._model_placement_pass(
        context, candidate_set, epoch, templates, resolution_passes
    )
    first_candidates = first.candidates
    first_selected = first.selected
    if first.compatible:
        invalidated_epoch_sha256 = (
            current_epoch.epoch_sha256
            if current_epoch is not None
            and publish_epoch is not None
            and current_epoch.epoch_sha256
                != publish_epoch.epoch_sha256
            else None
        )
        return controller._resolved_model_placement_result(
            first_candidates,
            first_selected,
            first.rejected,
            first.reason,
            resolution_passes,
            outcome=(
                "COMPATIBLE_CURRENT_EPOCH"
                if publish_epoch is None
                else "PUBLISHED_COMPATIBLE_ROUTE"
            ),
            invalidated_epoch_sha256=invalidated_epoch_sha256,
            publish_epoch=publish_epoch,
            publish_templates=publish_templates,
        )

    replacement_epoch, replacement_templates = (
        controller._propose_model_placement_epoch(
        request=request,
        manifest=manifest,
        candidate_set=first_candidates,
        selected=first_selected,
        observed_at_us=observed_at_us,
        selection_mode=selection_mode,
        invalidation_reason="LIVE_COMPONENT_CHANGED",
        snapshot=snapshot,
        route_compiler=compiler,
        demand_snapshot=demand_snapshot,
        placement_action=placement_action,
        current_epoch=None,
    ))
    second = controller._model_placement_pass(
        context,
        first_candidates,
        replacement_epoch,
        replacement_templates,
        resolution_passes,
    )
    if second.compatible:
        return controller._resolved_model_placement_result(
            second.candidates,
            second.selected,
            second.rejected,
            second.reason,
            resolution_passes,
            outcome="REPUBLISHED_LIVE_COMPONENT",
            invalidated_epoch_sha256=(
                None
                if current_epoch is None
                else current_epoch.epoch_sha256
            ),
            publish_epoch=replacement_epoch,
            publish_templates=replacement_templates,
        )
    return controller._desktop_fallback_after_nonconvergence(
        context,
        second.candidates,
        second.memory_rejections,
        resolution_passes,
        current_epoch,
    )


def _model_placement_pass(
    controller,
    context: _ModelPlacementMaterializeContext,
    source: AutomatedCandidateSet,
    epoch: RuntimeModelPlacementEpoch,
    templates: RuntimeRouteTemplateSet,
    resolution_passes: list,
) -> _ModelPlacementPass:
    """Materialize one epoch, record its resolution pass, check compatibility."""
    (
        candidates,
        selected,
        rejected,
        reason,
        memory_rejections,
    ) = controller._materialize_epoch_candidates(
        context, source, epoch, templates
    )
    resolution_passes.append(
        controller._model_placement_resolution_pass(
            epoch,
            epoch.selected_route_id,
            selected,
            context.manifest,
            candidates,
        )
    )
    compatibility = controller._model_placement_compatibility(
        epoch, selected, context.manifest, candidates
    )
    return _ModelPlacementPass(
        candidates,
        selected,
        rejected,
        reason,
        memory_rejections,
        compatibility.compatible,
    )


def _retained_model_placement_epoch(
    controller,
    current_epoch: RuntimeModelPlacementEpoch | None,
    prospective: AutomatedRouteCandidate,
    proposed_epoch: RuntimeModelPlacementEpoch,
    proposed_templates: RuntimeRouteTemplateSet,
) -> tuple[
    RuntimeModelPlacementEpoch,
    RuntimeRouteTemplateSet,
    RuntimeModelPlacementEpoch | None,
    RuntimeRouteTemplateSet | None,
]:
    """Keep the current epoch when the proposal re-selects its route."""
    retained_current_epoch = (
        current_epoch is not None
        and proposed_epoch.selected_route_id
            == current_epoch.selected_route_id
        and prospective.candidate_id
            != current_epoch.selected_route_id
    )
    if retained_current_epoch:
        retained_templates = controller._runtime_route_template_sets.get(
            current_epoch.epoch_sha256
        )
        if retained_templates is not None:
            return current_epoch, retained_templates, None, None
    return (
        proposed_epoch,
        proposed_templates,
        proposed_epoch,
        proposed_templates,
    )


def _rematerialize_epoch_candidates(
    controller,
    context: _ModelPlacementMaterializeContext,
    selected_epoch: RuntimeModelPlacementEpoch,
    selected_templates: RuntimeRouteTemplateSet,
) -> AutomatedCandidateSet:
    rematerialized = context.compiler.materialize_route_template_set(
        selected_templates,
        context.request,
        context.manifest,
        context.snapshot,
        observed_at_us=context.observed_at_us,
        residency_holds=context.residency_holds,
        search_metadata={
            "model_placement_epoch_generation": (
                selected_epoch.generation
            ),
            "model_placement_epoch_sha256": (
                selected_epoch.epoch_sha256
            ),
            "virtual_queue_request_count": (
                selected_epoch.virtual_queue_request_count
            ),
        },
    )
    controller._prepare_legacy_evidence_migrations(
        rematerialized, context.request, context.manifest
    )
    rematerialized = controller._apply_adaptive_history_costs(
        rematerialized,
        context.request,
        context.manifest,
        context.snapshot.cost_features,
    )
    return controller._apply_phone_residency_portfolio_authorization(
        rematerialized, context.manifest, context.request,
        snapshot=context.snapshot, observed_at_us=context.observed_at_us,
    )


def _materialize_epoch_candidates(
    controller,
    context: _ModelPlacementMaterializeContext,
    source: AutomatedCandidateSet,
    selected_epoch: RuntimeModelPlacementEpoch,
    selected_templates: RuntimeRouteTemplateSet,
) -> tuple[
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
    Mapping[str, str],
]:
    authorized = next((
        row for row in source.candidates
        if row.candidate_id == selected_epoch.selected_route_id
    ), None)
    requires_rematerialization = (
        authorized is None
        or "MODEL_EPOCH_AUDIT_ONLY"
            in authorized.rejection_reasons
        or "MODEL_EPOCH_AUDIT_ONLY"
            in authorized.binding.eligibility_reasons
    )
    if requires_rematerialization:
        rematerialized = controller._rematerialize_epoch_candidates(
            context, selected_epoch, selected_templates
        )
    else:
        rematerialized = source
    rematerialized = controller._candidate_set_with_epoch(
        rematerialized,
        selected_epoch,
        fast_path=False,
        invalidation_reason=context.invalidation_reason,
    )
    rematerialized = controller._authorize_epoch_compatible_routes(
        rematerialized, selected_epoch, context.manifest
    )
    live_memory_rejections = controller._runtime_memory_rejections(
        rematerialized,
        context.snapshot,
        request=(
            None
            if context.memory_source_snapshot is None
            else context.request
        ),
        source_snapshot=context.memory_source_snapshot,
        observed_at_us=(
            None
            if context.memory_source_snapshot is None
            else context.observed_at_us
        ),
        not_before_by_resource=context.memory_not_before_by_resource,
        projection_token=context.memory_projection_token,
    )
    selected, rejected, reason = (
        controller._select_automated_candidate(
            rematerialized,
            context.request,
            context.observed_at_us,
            selection_mode=context.selection_mode,
            runtime_rejections=live_memory_rejections,
            route_compiler=context.compiler,
            snapshot=context.snapshot,
        )
    )
    return (
        rematerialized,
        selected,
        rejected,
        reason,
        live_memory_rejections,
    )


def _desktop_fallback_after_nonconvergence(
    controller,
    context: _ModelPlacementMaterializeContext,
    second_candidates: AutomatedCandidateSet,
    second_memory_rejections: Mapping[str, str],
    resolution_passes: list,
    current_epoch: RuntimeModelPlacementEpoch | None,
) -> tuple[
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
    RuntimeModelPlacementEpoch | None,
    RuntimeRouteTemplateSet | None,
]:
    fallback_candidates = controller._candidate_set_without_published_epoch(
        second_candidates
    )
    fallback, rejected, reason = controller._select_automated_candidate(
        fallback_candidates,
        context.request,
        context.observed_at_us,
        selection_mode="desktop-baseline",
        runtime_rejections=second_memory_rejections,
        route_compiler=context.compiler,
    )
    return controller._resolved_model_placement_result(
        fallback_candidates,
        fallback,
        rejected,
        reason,
        resolution_passes,
        outcome="DESKTOP_FALLBACK_AFTER_NONCONVERGENCE",
        invalidated_epoch_sha256=(
            None
            if current_epoch is None
            else current_epoch.epoch_sha256
        ),
        publish_epoch=None,
        publish_templates=None,
    )


def _resolved_model_placement_result(
    controller,
    candidates: AutomatedCandidateSet,
    selected: AutomatedRouteCandidate,
    rejected: tuple[tuple[str, str], ...],
    reason: str,
    resolution_passes: list,
    *,
    outcome: str,
    invalidated_epoch_sha256: str | None,
    publish_epoch: RuntimeModelPlacementEpoch | None,
    publish_templates: RuntimeRouteTemplateSet | None,
) -> tuple[
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
    RuntimeModelPlacementEpoch | None,
    RuntimeRouteTemplateSet | None,
]:
    candidates = controller._candidate_set_with_placement_resolution(
        candidates,
        resolution_passes,
        outcome=outcome,
        invalidated_epoch_sha256=invalidated_epoch_sha256,
    )
    return (
        candidates,
        selected,
        rejected,
        reason,
        publish_epoch,
        publish_templates,
    )


def _calibration_model_placement_publication(
    controller,
    *,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    invalidation_reason: str,
    runtime_rejections: Mapping[str, str],
    current_epoch: RuntimeModelPlacementEpoch | None,
    demand_snapshot: ModelDemandSnapshot,
    placement_action: ModelPlacementAction,
    route_compiler: AutomatedRouteCompiler | None = None,
) -> tuple[
    AutomatedCandidateSet,
    RuntimeModelPlacementEpoch,
    RuntimeRouteTemplateSet,
]:
    """Publish a safe epoch without constraining calibration dispatch."""
    compiler = (
        controller._automated_compiler()
        if route_compiler is None else route_compiler
    )
    prospective, _, _ = controller._select_model_placement_candidate(
        candidate_set,
        request,
        observed_at_us,
        selection_mode="energy-aware",
        runtime_rejections=runtime_rejections,
        route_compiler=compiler,
    )
    epoch, templates = controller._propose_model_placement_epoch(
        request=request,
        manifest=manifest,
        candidate_set=candidate_set,
        selected=prospective,
        observed_at_us=observed_at_us,
        selection_mode="calibration",
        invalidation_reason=invalidation_reason,
        snapshot=snapshot,
        route_compiler=compiler,
        demand_snapshot=demand_snapshot,
        placement_action=placement_action,
        current_epoch=current_epoch,
    )
    return (
        controller._candidate_set_with_epoch(
            candidate_set,
            epoch,
            fast_path=False,
            invalidation_reason=invalidation_reason,
        ),
        epoch,
        templates,
    )
