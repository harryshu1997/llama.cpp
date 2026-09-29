"""AutomatedRequestMixin replan selection operations on its existing owner."""

from __future__ import annotations

import time
from typing import Mapping

from ..._internal.route_generation import RouteGenerationError
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from ..._internal.runtime_controller import RuntimeRequestTicket
from .common import _AutomatedReplanPreparation, _AutomatedSubmitContext


def _replan_candidate_set(
    controller,
    *,
    preparation: _AutomatedReplanPreparation,
    current: RuntimeRequestTicket,
    manifest,
    compiler,
    observed_at_us: int,
) -> AutomatedCandidateSet:
    context = preparation.context
    started_ns = time.perf_counter_ns()
    if (
        current.selection_mode == "calibration"
        or context.placement_epoch is None
        or context.route_templates is None
    ):
        result = controller._generate_automated_candidate_set(
            current.request,
            manifest,
            context.snapshot,
            observed_at_us,
            use_residency_holds=preparation.prioritize_resident_component,
        )
        context.timings_ns["generation"] += (
            time.perf_counter_ns() - started_ns
        )
        return result
    validation_started_ns = time.perf_counter_ns()
    epoch = context.placement_epoch
    templates = context.route_templates
    epoch_metadata = {
        "model_placement_epoch_sha256": epoch.epoch_sha256,
        "model_placement_epoch_generation": epoch.generation,
        "virtual_queue_request_count": epoch.virtual_queue_request_count,
    }
    try:
        result = compiler.materialize_route_template_set(
            templates,
            current.request,
            manifest,
            context.snapshot,
            observed_at_us=observed_at_us,
            residency_holds=(
                preparation.residency_holds
                if preparation.prioritize_resident_component else {}
            ),
            search_metadata=epoch_metadata,
            additional_live_route_ids=(
                controller._route_template_phone_helper_route_ids(
                    templates, manifest
                )
            ),
        )
    except RouteGenerationError:
        context.epoch_invalidation_reason = (
            "ROUTE_TEMPLATE_LIVE_REVALIDATION_FAILED"
        )
        controller._record_epoch_invalidation(
            context.epoch_invalidation_reason
        )
        context.placement_epoch = None
        context.route_templates = None
        result = controller._generate_automated_candidate_set(
            current.request,
            manifest,
            context.snapshot,
            observed_at_us,
            use_residency_holds=preparation.prioritize_resident_component,
        )
    else:
        controller._prepare_legacy_evidence_migrations(
            result, current.request, manifest
        )
        result = controller._apply_adaptive_history_costs(
            result,
            current.request,
            manifest,
            context.snapshot.cost_features,
        )
        compiler.capture_phone_residency_route_evidence(
            manifest, result, current.request.output_tokens
        )
        if controller._update_phone_residency_portfolio(
            current.request, manifest, observed_at_us, context.snapshot
        ):
            context.epoch_invalidation_reason = (
                "PHONE_RESIDENCY_PORTFOLIO_CHANGED"
            )
            controller._record_epoch_invalidation(
                context.epoch_invalidation_reason
            )
            context.placement_epoch = None
            context.route_templates = None
            result = controller._generate_automated_candidate_set(
                current.request,
                manifest,
                context.snapshot,
                observed_at_us,
                use_residency_holds=(
                    preparation.prioritize_resident_component
                ),
            )
        else:
            result = controller._apply_phone_residency_portfolio_authorization(
                result, manifest, current.request,
                snapshot=context.snapshot, observed_at_us=observed_at_us,
            )
    context.timings_ns["epoch_validation"] += (
        time.perf_counter_ns() - validation_started_ns
    )
    if context.placement_epoch is not None:
        result = controller._candidate_set_with_epoch(
            result,
            context.placement_epoch,
            fast_path=True,
            invalidation_reason=context.epoch_invalidation_reason,
        )
        result = controller._authorize_epoch_compatible_routes(
            result, context.placement_epoch, manifest
        )
    context.timings_ns["generation"] += (
        time.perf_counter_ns() - started_ns
    )
    return result


def _materialize_replan_candidate(
    controller,
    *,
    context: _AutomatedSubmitContext,
    candidate_set: AutomatedCandidateSet,
    prospective: AutomatedRouteCandidate,
    current: RuntimeRequestTicket,
    manifest,
    observed_at_us: int,
    memory_rejections: Mapping[str, tuple[str, ...]],
    residency_holds: Mapping[str, object],
    invalidation_reason: str,
    current_epoch: RuntimeModelPlacementEpoch | None,
    compiler=None,
) -> tuple:
    return controller._materialize_model_placement_candidate(
        candidate_set=candidate_set,
        prospective=prospective,
        request=current.request,
        manifest=manifest,
        snapshot=context.snapshot,
        observed_at_us=observed_at_us,
        selection_mode=current.selection_mode,
        invalidation_reason=invalidation_reason,
        runtime_rejections=memory_rejections,
        current_epoch=current_epoch,
        demand_snapshot=context.demand_snapshot,
        placement_action=context.placement_action,
        residency_holds=residency_holds,
        route_compiler=compiler,
        memory_projection_token=controller._current_phone_projection_token(),
    )


def _replan_candidate_selection(
    controller,
    *,
    preparation: _AutomatedReplanPreparation,
    candidate_set: AutomatedCandidateSet,
    current: RuntimeRequestTicket,
    manifest,
    compiler,
    observed_at_us: int,
    memory_rejections: Mapping[str, tuple[str, ...]],
) -> tuple:
    context = preparation.context
    refresh_required = (
        context.placement_epoch is None
        or context.epoch_invalidation_reason not in {
            "NONE", "MODEL_PLACEMENT_EPOCH_COMPATIBLE_SHAPE"
        }
        or context.placement_action.kind != "KEEP_EPOCH"
    )
    evaluate_prospective = (
        current.selection_mode != "calibration" and refresh_required
    )
    prospective = None
    if evaluate_prospective:
        prospective, _, _ = controller._select_model_placement_candidate(
            candidate_set,
            current.request,
            observed_at_us,
            selection_mode=current.selection_mode,
            runtime_rejections=memory_rejections,
            snapshot=context.snapshot,
        )
    compatible = (
        context.placement_epoch is not None
        and prospective is not None
        and controller._model_placement_compatibility(
            context.placement_epoch,
            prospective,
            manifest,
            candidate_set,
        ).compatible
    )
    if current.selection_mode == "calibration" and refresh_required:
        candidate_set, publish_epoch, templates = (
            controller._calibration_model_placement_publication(
                candidate_set=candidate_set,
                request=current.request,
                manifest=manifest,
                snapshot=context.snapshot,
                observed_at_us=observed_at_us,
                invalidation_reason=context.epoch_invalidation_reason,
                runtime_rejections=memory_rejections,
                current_epoch=context.current_epoch_for_publication,
                demand_snapshot=context.demand_snapshot,
                placement_action=context.placement_action,
            )
        )
        selected, rejected, reason = controller._select_automated_candidate(
            candidate_set,
            current.request,
            observed_at_us,
            selection_mode=current.selection_mode,
            runtime_rejections=memory_rejections,
        )
        result = candidate_set, selected, rejected, reason, publish_epoch, templates
    elif evaluate_prospective and (
        context.placement_epoch is None or not compatible
    ):
        assert prospective is not None
        result = controller._materialize_replan_candidate(
            context=context,
            candidate_set=candidate_set,
            prospective=prospective,
            current=current,
            manifest=manifest,
            observed_at_us=observed_at_us,
            memory_rejections=memory_rejections,
            residency_holds=preparation.residency_holds,
            invalidation_reason=context.epoch_invalidation_reason,
            current_epoch=context.current_epoch_for_publication,
        )
    else:
        selected, rejected, reason = controller._select_automated_candidate(
            candidate_set,
            current.request,
            observed_at_us,
            selection_mode=current.selection_mode,
            runtime_rejections=memory_rejections,
            snapshot=context.snapshot,
        )
        result = candidate_set, selected, rejected, reason, None, None
        if context.placement_epoch is not None:
            compatibility = controller._model_placement_compatibility(
                context.placement_epoch,
                selected,
                manifest,
                candidate_set,
            )
            if compatibility.compatible:
                candidate_set = controller._candidate_set_with_placement_resolution(
                    candidate_set,
                    (controller._model_placement_resolution_pass(
                        context.placement_epoch,
                        context.placement_epoch.selected_route_id,
                        selected,
                        manifest,
                        candidate_set,
                    ),),
                    outcome="COMPATIBLE_LIVE_ROUTE",
                )
                result = candidate_set, selected, rejected, reason, None, None
            else:
                result = controller._materialize_replan_candidate(
                    context=context,
                    candidate_set=candidate_set,
                    prospective=selected,
                    current=current,
                    manifest=manifest,
                    observed_at_us=observed_at_us,
                    memory_rejections=memory_rejections,
                    residency_holds=preparation.residency_holds,
                    invalidation_reason="LIVE_COMPONENT_CHANGED",
                    current_epoch=context.placement_epoch,
                    compiler=compiler,
                )
    candidate_set, selected, rejected, reason, epoch, templates = result
    candidate_set, selected, rejected, reason = (
        controller._desktop_with_async_phone_helper(
            candidate_set,
            selected,
            rejected,
            reason,
            current.request,
            manifest,
            current.selection_mode,
        )
    )
    return candidate_set, selected, rejected, reason, epoch, templates
