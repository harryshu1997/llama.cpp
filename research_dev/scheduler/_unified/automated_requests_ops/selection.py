"""AutomatedRequestMixin selection operations on its existing owner."""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import LeasePreview, Request, SchedulerError
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeCostEstimateSet
from ..._internal.model_manifest import ModelManifestError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.route_generation import RouteGenerationError, candidate_set_to_runtime_costs
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from ..._internal.runtime_controller import RuntimeControllerError, RuntimeRequestTicket
from .affinity import (
    AFFINITY_DISPLACEMENT_REASON,
    _NoAffinityGain,
    model_affinity_displacement,
    no_gain_reason,
)
from .common import _AutomatedSubmitContext, _AutomatedSubmitResolution
from .continuous_join import (
    BARRIER_DISPLACEMENT_REASON,
    _NoJoinGain,
    bound_detail,
    continuous_join_barrier_bypass,
    live_not_before_after_displacement,
)


def _prepare_automated_submit_context(
    controller,
    *,
    request: Request,
    manifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    selection_mode: str,
    causal_not_before_by_resource: Mapping[str, int],
) -> _AutomatedSubmitContext:
    timings = {
        key: 0
        for key in (
            "projection",
            "generation",
            "conversion",
            "admission",
            "selection",
            "preview",
            "epoch_lookup",
            "epoch_validation",
            "epoch_preprojection",
        )
    }
    request.validate()
    source_snapshot = snapshot
    stage_started_ns = time.perf_counter_ns()
    snapshot = controller._automated_snapshot_for_request(
        request, source_snapshot, project_before_us=observed_at_us
    )
    timings["projection"] += time.perf_counter_ns() - stage_started_ns
    stage_started_ns = time.perf_counter_ns()
    placement_epoch, route_templates, invalidation_reason = (
        controller._published_epoch_for_request(
            request,
            manifest,
            observed_at_us,
            selection_mode,
        )
    )
    current_epoch = placement_epoch
    timings["epoch_lookup"] += time.perf_counter_ns() - stage_started_ns
    controller._model_placement_controller.notify(
        manifest.artifact_sha256, "REQUEST_ARRIVAL", observed_at_us
    )
    demand_snapshot, placement_action = controller._evaluate_model_placement(
        request,
        manifest,
        snapshot,
        observed_at_us,
        placement_epoch,
        selection_mode,
    )
    if placement_action.kind in {"RECOMPUTE_NOW", "FALLBACK"}:
        if placement_epoch is not None:
            invalidation_reason = "+".join(
                placement_action.trigger_reasons
            )
            controller._record_epoch_invalidation(invalidation_reason)
        placement_epoch = None
        route_templates = None
    elif placement_action.kind == "REFRESH_IN_BACKGROUND":
        controller._prepared_placement_frontier(
            request,
            manifest,
            snapshot,
            observed_at_us,
            demand_snapshot=demand_snapshot,
        )
    barriers = dict(causal_not_before_by_resource)
    if placement_epoch is not None and route_templates is not None:
        stage_started_ns = time.perf_counter_ns()
        epoch_selected = _epoch_preprojection_template(
            controller, request, route_templates
        )
        epoch_preview = controller._preview_automated_resources(
            epoch_selected,
            observed_at_us=controller._causal_candidate_observed_at(
                epoch_selected, observed_at_us, barriers
            ),
        )
        projection_started_ns = time.perf_counter_ns()
        projected = controller._automated_snapshot_for_request(
            request,
            source_snapshot,
            project_before_us=epoch_preview.start_us,
        )
        if (
            projected.residency != snapshot.residency
            or projected.memory != snapshot.memory
        ):
            for resource_id in epoch_selected.plan.resource_ids:
                barriers[resource_id] = max(
                    barriers.get(resource_id, 0), epoch_preview.start_us
                )
        snapshot = projected
        finished_ns = time.perf_counter_ns()
        timings["epoch_preprojection"] += finished_ns - stage_started_ns
        timings["projection"] += finished_ns - projection_started_ns
    return _AutomatedSubmitContext(
        source_snapshot=source_snapshot,
        snapshot=snapshot,
        placement_epoch=placement_epoch,
        current_epoch_for_publication=current_epoch,
        route_templates=route_templates,
        demand_snapshot=demand_snapshot,
        placement_action=placement_action,
        epoch_invalidation_reason=invalidation_reason,
        live_not_before_by_resource=barriers,
        timings_ns=timings,
    )


def _epoch_preprojection_template(
    controller, request: Request, route_templates
) -> AutomatedRouteCandidate:
    """The epoch template whose earliest start the residency pre-projection uses.

    The epoch's selected route, except during the request's barrier-bypass
    re-resolution: the bypass admits only a plan free of residency transitions,
    normally the desktop parent of the running server, while the selected
    (assisted) template is previewed behind the co-tenant holding its phone
    lanes and would bound every resource of the parent to that horizon. The
    epoch's baseline template is used then; the selected one when it is absent.
    """
    candidates = route_templates.candidate_set.candidates
    selected = next(
        row for row in candidates
        if row.candidate_id == route_templates.selected_route_id
    )
    runtime = controller._runtime_controller
    if runtime.continuous_join_resolution_request_id() != request.request_id:
        return selected
    baseline_route_id = route_templates.candidate_set.baseline_route_id
    return next(
        (row for row in candidates if row.candidate_id == baseline_route_id),
        selected,
    )


def _displacement_not_before(
    controller,
    causal_not_before_by_resource: Mapping[str, int],
    displaced: tuple[str, ...],
) -> Mapping[str, int]:
    """The causal barriers a displacement re-resolution starts from.

    Under ``continuous_join`` the barriers the cancelled work alone justified
    are dropped (``live_not_before_after_displacement``); otherwise the
    caller's map is used unchanged.
    """
    if not controller._runtime_controller.dispatch_policy.continuous_join:
        return causal_not_before_by_resource
    return live_not_before_after_displacement(
        controller, causal_not_before_by_resource, displaced
    )


def _displacement_bound(
    controller,
    context: _AutomatedSubmitContext,
    resolution: _AutomatedSubmitResolution,
) -> str | None:
    """What holds a displacement re-resolution's start; None unless ``continuous_join``."""
    if not controller._runtime_controller.dispatch_policy.continuous_join:
        return None
    return bound_detail(context, resolution)


def _submit_candidate_set(
    controller,
    *,
    context: _AutomatedSubmitContext,
    request: Request,
    manifest,
    compiler,
    observed_at_us: int,
    selection_mode: str,
) -> AutomatedCandidateSet:
    if (
        selection_mode == "calibration"
        or context.placement_epoch is None
        or context.route_templates is None
    ):
        return controller._generate_automated_candidate_set(
            request, manifest, context.snapshot, observed_at_us
        )
    validation_started_ns = time.perf_counter_ns()
    residency_holds = controller._runtime_residency_cohorts.holds(
        controller._runtime_capabilities,
        context.snapshot,
        controller._runtime_controller.current_tickets(),
        observed_at_us,
    )
    context.timings_ns["epoch_validation"] += (
        time.perf_counter_ns() - validation_started_ns
    )
    epoch = context.placement_epoch
    templates = context.route_templates
    epoch_metadata = {
        "model_placement_epoch_sha256": epoch.epoch_sha256,
        "model_placement_epoch_generation": epoch.generation,
        "virtual_queue_request_count": epoch.virtual_queue_request_count,
    }
    try:
        candidate_set = compiler.reuse_exact_route_template_set(
            templates,
            request,
            manifest,
            context.snapshot,
            observed_at_us=observed_at_us,
            search_metadata=epoch_metadata,
        )
        if candidate_set is None:
            candidate_set = compiler.materialize_route_template_set(
                templates,
                request,
                manifest,
                context.snapshot,
                observed_at_us=observed_at_us,
                residency_holds=residency_holds,
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
        return controller._generate_automated_candidate_set(
            request, manifest, context.snapshot, observed_at_us
        )
    controller._prepare_legacy_evidence_migrations(
        candidate_set, request, manifest
    )
    candidate_set = controller._apply_adaptive_history_costs(
        candidate_set,
        request,
        manifest,
        context.snapshot.cost_features,
    )
    compiler.capture_phone_residency_route_evidence(
        manifest, candidate_set, request.output_tokens
    )
    portfolio_changed = controller._update_phone_residency_portfolio(
        request, manifest, observed_at_us, context.snapshot
    )
    if portfolio_changed:
        context.epoch_invalidation_reason = (
            "PHONE_RESIDENCY_PORTFOLIO_CHANGED"
        )
        controller._record_epoch_invalidation(
            context.epoch_invalidation_reason
        )
        context.placement_epoch = None
        context.route_templates = None
        return controller._generate_automated_candidate_set(
            request, manifest, context.snapshot, observed_at_us
        )
    candidate_set = controller._apply_phone_residency_portfolio_authorization(
        candidate_set, manifest, request,
        snapshot=context.snapshot, observed_at_us=observed_at_us,
    )
    candidate_set = controller._candidate_set_with_epoch(
        candidate_set,
        epoch,
        fast_path=True,
        invalidation_reason=context.epoch_invalidation_reason,
    )
    return controller._authorize_epoch_compatible_routes(
        candidate_set, epoch, manifest
    )


def _select_existing_epoch_candidate(
    controller,
    *,
    context: _AutomatedSubmitContext,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest,
    compiler,
    observed_at_us: int,
    selection_mode: str,
    memory_rejections: Mapping[str, tuple[str, ...]],
) -> tuple:
    selected, rejected, reason = controller._select_automated_candidate(
        candidate_set,
        request,
        observed_at_us,
        selection_mode=selection_mode,
        runtime_rejections=memory_rejections,
        snapshot=context.snapshot,
    )
    if context.placement_epoch is None:
        return candidate_set, selected, rejected, reason, None, None
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
        return candidate_set, selected, rejected, reason, None, None
    residency_holds = controller._runtime_residency_cohorts.holds(
        controller._runtime_capabilities,
        context.snapshot,
        controller._runtime_controller.current_tickets(),
        observed_at_us,
    )
    return controller._materialize_model_placement_candidate(
        candidate_set=candidate_set,
        prospective=selected,
        request=request,
        manifest=manifest,
        snapshot=context.snapshot,
        observed_at_us=observed_at_us,
        selection_mode=selection_mode,
        invalidation_reason="LIVE_COMPONENT_CHANGED",
        runtime_rejections=memory_rejections,
        current_epoch=context.placement_epoch,
        demand_snapshot=context.demand_snapshot,
        placement_action=context.placement_action,
        residency_holds=residency_holds,
        route_compiler=compiler,
        memory_source_snapshot=context.source_snapshot,
        memory_not_before_by_resource=(
            context.live_not_before_by_resource
        ),
        memory_projection_token=controller._current_phone_projection_token(),
    )


def _submit_candidate_selection(
    controller,
    *,
    context: _AutomatedSubmitContext,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest,
    compiler,
    observed_at_us: int,
    selection_mode: str,
    memory_rejections: Mapping[str, tuple[str, ...]],
) -> tuple:
    placement_refresh_required = (
        context.placement_epoch is None
        or context.epoch_invalidation_reason not in {
            "NONE",
            "MODEL_PLACEMENT_EPOCH_COMPATIBLE_SHAPE",
        }
        or context.placement_action.kind != "KEEP_EPOCH"
    )
    evaluate_prospective = (
        selection_mode != "calibration" and placement_refresh_required
    )
    prospective = None
    if evaluate_prospective:
        prospective, _, _ = controller._select_model_placement_candidate(
            candidate_set,
            request,
            observed_at_us,
            selection_mode=selection_mode,
            runtime_rejections=memory_rejections,
            snapshot=context.snapshot,
        )
    prospective_compatible = (
        context.placement_epoch is not None
        and prospective is not None
        and controller._model_placement_compatibility(
            context.placement_epoch,
            prospective,
            manifest,
            candidate_set,
        ).compatible
    )
    if selection_mode == "calibration" and placement_refresh_required:
        candidate_set, publish_epoch, publish_templates = (
            controller._calibration_model_placement_publication(
                candidate_set=candidate_set,
                request=request,
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
            request,
            observed_at_us,
            selection_mode=selection_mode,
            runtime_rejections=memory_rejections,
        )
        result = (
            candidate_set,
            selected,
            rejected,
            reason,
            publish_epoch,
            publish_templates,
        )
    elif evaluate_prospective and (
        context.placement_epoch is None or not prospective_compatible
    ):
        residency_holds = controller._runtime_residency_cohorts.holds(
            controller._runtime_capabilities,
            context.snapshot,
            controller._runtime_controller.current_tickets(),
            observed_at_us,
        )
        result = controller._materialize_model_placement_candidate(
            candidate_set=candidate_set,
            prospective=prospective,
            request=request,
            manifest=manifest,
            snapshot=context.snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            invalidation_reason=context.epoch_invalidation_reason,
            runtime_rejections=memory_rejections,
            current_epoch=context.current_epoch_for_publication,
            demand_snapshot=context.demand_snapshot,
            placement_action=context.placement_action,
            residency_holds=residency_holds,
            memory_source_snapshot=context.source_snapshot,
            memory_not_before_by_resource=(
                context.live_not_before_by_resource
            ),
            memory_projection_token=controller._current_phone_projection_token(),
        )
    else:
        result = controller._select_existing_epoch_candidate(
            context=context,
            candidate_set=candidate_set,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            memory_rejections=memory_rejections,
        )
    candidate_set, selected, rejected, reason, publish_epoch, templates = result
    candidate_set, selected, rejected, reason = (
        controller._desktop_with_async_phone_helper(
            candidate_set,
            selected,
            rejected,
            reason,
            request,
            manifest,
            selection_mode,
        )
    )
    return (
        candidate_set,
        selected,
        rejected,
        reason,
        publish_epoch,
        templates,
    )


def _project_submit_selection(
    controller,
    *,
    context: _AutomatedSubmitContext,
    request: Request,
    selected: AutomatedRouteCandidate,
    observed_at_us: int,
) -> tuple[LeasePreview, bool]:
    for resource_id, barrier_us in (
        controller._runtime_plan_not_before_by_resource(selected.plan).items()
    ):
        context.live_not_before_by_resource[resource_id] = max(
            context.live_not_before_by_resource.get(resource_id, 0),
            barrier_us,
        )
    stage_started_ns = time.perf_counter_ns()
    preview = controller._preview_automated_resources(
        selected,
        observed_at_us=controller._causal_candidate_observed_at(
            selected,
            observed_at_us,
            context.live_not_before_by_resource,
        ),
    )
    projected = controller._automated_snapshot_for_request(
        request,
        context.source_snapshot,
        project_before_us=preview.start_us,
    )
    elapsed_ns = time.perf_counter_ns() - stage_started_ns
    if (
        projected.residency != context.snapshot.residency
        or projected.memory != context.snapshot.memory
    ):
        context.timings_ns["preview"] += elapsed_ns
        context.timings_ns["projection"] += elapsed_ns
        for resource_id in selected.plan.resource_ids:
            context.live_not_before_by_resource[resource_id] = max(
                context.live_not_before_by_resource.get(resource_id, 0),
                preview.start_us,
            )
        context.snapshot = projected
        return preview, False
    context.timings_ns["projection"] += elapsed_ns
    controller._preview_automated_memory(
        selected.plan,
        context.snapshot,
        start_us=preview.start_us,
        reserved_until_us=preview.finish_upper_us,
    )
    context.timings_ns["preview"] += (
        time.perf_counter_ns() - stage_started_ns
    )
    return preview, True


def _reproject_rejected_baseline(
    controller,
    *,
    context: _AutomatedSubmitContext,
    request: Request,
    candidate_set: AutomatedCandidateSet,
    memory_rejections: Mapping[str, str],
    observed_at_us: int,
) -> bool:
    """Re-plan against residency at a memory-rejected baseline's start.

    The earliest calendar slot can fall behind a queued exclusive
    replacement; the candidate set then describes stale residency.
    """
    baseline = candidate_set.baseline
    if memory_rejections.get(baseline.candidate_id) in {
        None, "RESOURCE_CALENDAR_CURRENT",
    }:
        return False
    not_before = dict(context.live_not_before_by_resource)
    for resource_id, barrier_us in (
        controller._runtime_plan_not_before_by_resource(baseline.plan).items()
    ):
        not_before[resource_id] = max(
            not_before.get(resource_id, 0), barrier_us
        )
    started_ns = time.perf_counter_ns()
    try:
        preview = controller._preview_automated_resources(
            baseline,
            observed_at_us=controller._causal_candidate_observed_at(
                baseline, observed_at_us, not_before
            ),
        )
    except SchedulerError:
        return False
    projected = controller._automated_snapshot_for_request(
        request,
        context.source_snapshot,
        project_before_us=preview.start_us,
    )
    context.timings_ns["projection"] += time.perf_counter_ns() - started_ns
    if (
        projected.residency == context.snapshot.residency
        and projected.memory == context.snapshot.memory
    ):
        return False
    for resource_id in baseline.plan.resource_ids:
        context.live_not_before_by_resource[resource_id] = max(
            context.live_not_before_by_resource.get(resource_id, 0),
            preview.start_us,
        )
    context.snapshot = projected
    return True


def _resolve_automated_submit(
    controller,
    *,
    context: _AutomatedSubmitContext,
    request: Request,
    manifest,
    compiler,
    observed_at_us: int,
    selection_mode: str,
) -> _AutomatedSubmitResolution:
    for iteration in range(1, 4):
        started_ns = time.perf_counter_ns()
        candidate_set = controller._submit_candidate_set(
            context=context,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
        )
        context.timings_ns["generation"] += (
            time.perf_counter_ns() - started_ns
        )
        started_ns = time.perf_counter_ns()
        memory_rejections = controller._runtime_memory_rejections(
            candidate_set,
            context.snapshot,
            request=request,
            source_snapshot=context.source_snapshot,
            observed_at_us=observed_at_us,
            not_before_by_resource=context.live_not_before_by_resource,
            projection_token=controller._current_phone_projection_token(),
        )
        context.timings_ns["admission"] += (
            time.perf_counter_ns() - started_ns
        )
        if _reproject_rejected_baseline(
            controller,
            context=context,
            request=request,
            candidate_set=candidate_set,
            memory_rejections=memory_rejections,
            observed_at_us=observed_at_us,
        ):
            continue
        started_ns = time.perf_counter_ns()
        (
            candidate_set,
            selected,
            rejected,
            reason,
            publish_epoch,
            publish_templates,
        ) = controller._submit_candidate_selection(
            context=context,
            candidate_set=candidate_set,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            memory_rejections=memory_rejections,
        )
        context.timings_ns["selection"] += (
            time.perf_counter_ns() - started_ns
        )
        preview, converged = controller._project_submit_selection(
            context=context,
            request=request,
            selected=selected,
            observed_at_us=observed_at_us,
        )
        if converged:
            return _AutomatedSubmitResolution(
                candidate_set,
                selected,
                preview,
                rejected,
                reason,
                publish_epoch,
                publish_templates,
                iteration,
            )
    raise UnifiedScheduleError(
        "runtime residency planning did not converge"
    )


def _convert_submit_candidate_set(
    controller,
    context: _AutomatedSubmitContext,
    resolution: _AutomatedSubmitResolution,
    request: Request,
    manifest,
) -> RuntimeCostEstimateSet:
    started_ns = time.perf_counter_ns()
    estimates = candidate_set_to_runtime_costs(
        resolution.candidate_set,
        request,
        manifest,
        context.snapshot,
        controller._runtime_capabilities,
        planning_profile_sha256=(
            controller._runtime_capability_generation_sha256
        ),
        model_manifest_sha256=(
            controller._runtime_manifest_generation_sha256.get(
                manifest.model_id
            )
        ),
    )
    context.timings_ns["conversion"] += time.perf_counter_ns() - started_ns
    return estimates


def _commit_submitted_automated_request(
    controller,
    *,
    submit_started_ns: int,
    context: _AutomatedSubmitContext,
    resolution: _AutomatedSubmitResolution,
    estimates: RuntimeCostEstimateSet,
    request: Request,
    selection_mode: str,
    compiler,
    observed_at_us: int,
) -> RuntimeRequestTicket:
    checkpoint_started_ns = time.perf_counter_ns()
    try:
        with controller._transaction(errors=Exception, convert=False):
            checkpoint_us = (
                time.perf_counter_ns() - checkpoint_started_ns
            ) // 1000
            ticket = controller._commit_automated_attempt(
                request=request,
                snapshot=context.snapshot,
                candidate_set=resolution.candidate_set,
                estimates=estimates,
                selected=resolution.selected,
                preview=resolution.preview,
                rejected=resolution.rejected,
                reason=resolution.reason,
                observed_at_us=observed_at_us,
                event_kind="DECISION",
                selection_mode=selection_mode,
                placement_epoch=resolution.publish_epoch,
                route_templates=resolution.publish_templates,
                bound_placement_epoch=(
                    resolution.publish_epoch or context.placement_epoch
                ),
            )
            finished_ns = time.perf_counter_ns()
            timings = context.timings_ns
            controller._runtime_decision_timings.append(MappingProxyType({
                "attempt_index": ticket.attempt_index,
                "candidate_generation": dict(
                    compiler.last_generation_timing()
                ),
                "candidate_generation_us": timings["generation"] // 1000,
                "candidate_serialization_us": timings["conversion"] // 1000,
                "checkpoint_us": checkpoint_us,
                "commit": dict(controller._last_runtime_commit_timing),
                "eligibility_admission_us": timings["admission"] // 1000,
                "epoch_lookup_us": timings["epoch_lookup"] // 1000,
                "epoch_preprojection_us": (
                    timings["epoch_preprojection"] // 1000
                ),
                "epoch_validation_us": (
                    timings["epoch_validation"] // 1000
                ),
                "event_kind": "DECISION",
                "model_placement_epoch_fast_path": (
                    context.placement_epoch is not None
                ),
                "model_placement_epoch_invalidation_reason": (
                    context.epoch_invalidation_reason
                ),
                "request_id": request.request_id,
                "reservation_preview_us": timings["preview"] // 1000,
                "residency_fixed_point_iterations": (
                    resolution.residency_iterations
                ),
                "selection_us": timings["selection"] // 1000,
                "snapshot_projection_us": timings["projection"] // 1000,
                "ticket_id": ticket.ticket_id,
                "total_us": (finished_ns - submit_started_ns) // 1000,
            }))
            return ticket
    except Exception as exc:
        if isinstance(exc, UnifiedScheduleError):
            raise
        raise UnifiedScheduleError(str(exc)) from exc


def _submit_automated_request_once(
    controller,
    request: Request,
    model_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    observed_at_us: int | None = None,
    selection_mode: str = "energy-aware",
    causal_not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
) -> RuntimeRequestTicket:
    """Discover, cost, reserve, bind, journal, and queue one request."""
    submit_started_ns = time.perf_counter_ns()
    # Permanently unsupported shapes are rejected here with an exact reason
    # (RequestShapeUnsupportedError) before any planning; contention still
    # queues through the resource calendar below.
    controller._require_supported_request_shape(request, model_id)
    manifest = controller.runtime_model_manifest(model_id)
    compiler = controller._automated_compiler()
    observed_at_us = (
        request.arrival_us if observed_at_us is None else observed_at_us
    )
    if (
        type(observed_at_us) is not int
        or observed_at_us < request.arrival_us
    ):
        raise UnifiedScheduleError(
            "automated observation time is invalid"
        )
    try:
        context = controller._prepare_automated_submit_context(
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            causal_not_before_by_resource=(
                causal_not_before_by_resource
            ),
        )
        resolution = controller._resolve_automated_submit(
            context=context,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
        )
        estimates = controller._convert_submit_candidate_set(
            context, resolution, request, manifest
        )
    except (
        ModelManifestError,
        RouteGenerationError,
        RuntimeCapabilityError,
        RuntimeResourceError,
        RuntimeResidencyCohortError,
        RuntimeResidencyProjectionError,
        SchedulerError,
    ) as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    displacement = model_affinity_displacement(
        controller,
        snapshot=snapshot,
        artifact_sha256=manifest.artifact_sha256,
        resolution=resolution,
        observed_at_us=observed_at_us,
    )
    if displacement is not None:
        ticket = _submit_with_model_affinity(
            controller,
            displacement=displacement,
            submit_started_ns=submit_started_ns,
            original=resolution,
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            causal_not_before_by_resource=causal_not_before_by_resource,
        )
        if ticket is not None:
            return ticket
    # A refused affinity displacement rolled back; the bounded barrier bypass
    # judges the same arrival by its own rule (the committed-window extension).
    bypass = continuous_join_barrier_bypass(
        controller,
        snapshot=snapshot,
        artifact_sha256=manifest.artifact_sha256,
        resolution=resolution,
        observed_at_us=observed_at_us,
    )
    if bypass is not None:
        ticket = _submit_with_barrier_bypass(
            controller,
            bypass=bypass,
            submit_started_ns=submit_started_ns,
            original=resolution,
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            causal_not_before_by_resource=causal_not_before_by_resource,
        )
        if ticket is not None:
            return ticket
    return controller._commit_submitted_automated_request(
        submit_started_ns=submit_started_ns,
        context=context,
        resolution=resolution,
        estimates=estimates,
        request=request,
        selection_mode=selection_mode,
        compiler=compiler,
        observed_at_us=observed_at_us,
    )


def _submit_with_barrier_bypass(
    controller,
    *,
    bypass: tuple[tuple[str, ...], dict[str, object]],
    submit_started_ns: int,
    original: _AutomatedSubmitResolution,
    request: Request,
    manifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    compiler,
    observed_at_us: int,
    selection_mode: str,
    causal_not_before_by_resource: Mapping[str, int],
) -> RuntimeRequestTicket | None:
    """Reserve the joiner ahead of the displaced change within the bound, or roll back to None.

    The change is cancelled, the joiner re-planned as a joiner from the live
    tickets only (the caller's causal barriers that only the cancelled work
    justified are dropped, the epoch pre-projection previews the desktop-parent
    template, and the join selection rejects plans preparing an exclusive
    residency device, so the desktop parent of its running server is selected
    when it exists), and the attempt is kept only when it starts earlier,
    changes no residency, and extends the committed busy window by at most
    ``max_barrier_extension_s``; otherwise the transaction rolls back and a
    refusal is counted with the failing half of that rule and what holds the
    re-resolved start.
    """
    displaced, note = bypass
    runtime = controller._runtime_controller
    bound_us = runtime.dispatch_policy.max_barrier_extension_s * 1_000_000
    try:
        with runtime.defer_dispatch_wake(), controller._transaction(
            errors=Exception, convert=False
        ):
            runtime.replan_queued_now(
                displaced,
                BARRIER_DISPLACEMENT_REASON,
                observed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
            try:
                with runtime.continuous_join_resolution(request.request_id):
                    context = controller._prepare_automated_submit_context(
                        request=request,
                        manifest=manifest,
                        snapshot=snapshot,
                        observed_at_us=observed_at_us,
                        selection_mode=selection_mode,
                        causal_not_before_by_resource=(
                            live_not_before_after_displacement(
                                controller,
                                causal_not_before_by_resource,
                                displaced,
                            )
                        ),
                    )
                    resolution = controller._resolve_automated_submit(
                        context=context,
                        request=request,
                        manifest=manifest,
                        compiler=compiler,
                        observed_at_us=observed_at_us,
                        selection_mode=selection_mode,
                    )
                estimates = controller._convert_submit_candidate_set(
                    context, resolution, request, manifest
                )
            except (
                ModelManifestError,
                RouteGenerationError,
                RuntimeCapabilityError,
                RuntimeResourceError,
                RuntimeResidencyCohortError,
                RuntimeResidencyProjectionError,
                SchedulerError,
            ) as exc:
                raise _NoJoinGain(str(exc)) from exc
            refusal = no_gain_reason(
                resolution,
                original,
                controller._runtime_exclusive_memory_resources(),
                "bypass does not start the joiner earlier",
                bound=bound_detail(context, resolution),
            )
            if refusal is not None:
                raise _NoJoinGain(refusal)
            extension_us = max(
                0, resolution.preview.finish_upper_us - note["committed_end_us"]
            )
            if extension_us > bound_us:
                raise _NoJoinGain("joiner would extend the committed busy window")
            try:
                with runtime.dispatch_precedence(
                    request.request_id,
                    displaced,
                    {
                        **note,
                        "extension_us": extension_us,
                        "max_barrier_extension_us": bound_us,
                        "finish_upper_us": resolution.preview.finish_upper_us,
                        "reserved_start_us": resolution.preview.start_us,
                    },
                ):
                    ticket = controller._commit_submitted_automated_request(
                        submit_started_ns=submit_started_ns,
                        context=context,
                        resolution=resolution,
                        estimates=estimates,
                        request=request,
                        selection_mode=selection_mode,
                        compiler=compiler,
                        observed_at_us=observed_at_us,
                    )
            except (RuntimeControllerError, UnifiedScheduleError) as exc:
                raise _NoJoinGain(str(exc)) from exc
            return ticket
    except _NoJoinGain as exc:
        runtime.record_dispatch_policy_event("continuous_join_refusals")
        runtime.record_dispatch_policy_refusal(
            "CONTINUOUS_JOIN_REFUSED", request.request_id, str(exc), observed_at_us
        )
        return None


def _submit_with_model_affinity(
    controller,
    *,
    displacement: tuple[tuple[str, ...], dict[str, object]],
    submit_started_ns: int,
    original: _AutomatedSubmitResolution,
    request: Request,
    manifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    compiler,
    observed_at_us: int,
    selection_mode: str,
    causal_not_before_by_resource: Mapping[str, int],
) -> RuntimeRequestTicket | None:
    """Plan the arrival ahead of the displaced work, or roll back to None."""
    displaced, note = displacement
    runtime = controller._runtime_controller
    try:
        with runtime.defer_dispatch_wake(), controller._transaction(
            errors=Exception, convert=False
        ):
            runtime.replan_queued_now(
                displaced,
                AFFINITY_DISPLACEMENT_REASON,
                observed_at_us,
                cancel_owner=controller.cancel,
                release_memory=controller._runtime_memory.release_owner,
            )
            try:
                context = controller._prepare_automated_submit_context(
                    request=request,
                    manifest=manifest,
                    snapshot=snapshot,
                    observed_at_us=observed_at_us,
                    selection_mode=selection_mode,
                    causal_not_before_by_resource=(
                        _displacement_not_before(
                            controller, causal_not_before_by_resource, displaced
                        )
                    ),
                )
                resolution = controller._resolve_automated_submit(
                    context=context,
                    request=request,
                    manifest=manifest,
                    compiler=compiler,
                    observed_at_us=observed_at_us,
                    selection_mode=selection_mode,
                )
                estimates = controller._convert_submit_candidate_set(
                    context, resolution, request, manifest
                )
            except (
                ModelManifestError,
                RouteGenerationError,
                RuntimeCapabilityError,
                RuntimeResourceError,
                RuntimeResidencyCohortError,
                RuntimeResidencyProjectionError,
                SchedulerError,
            ) as exc:
                raise _NoAffinityGain(str(exc)) from exc
            refusal = no_gain_reason(
                resolution,
                original,
                controller._runtime_exclusive_memory_resources(),
                "displacement does not help",
                bound=_displacement_bound(controller, context, resolution),
            )
            if refusal is not None:
                raise _NoAffinityGain(refusal)
            try:
                with runtime.dispatch_precedence(
                    request.request_id,
                    displaced,
                    {**note, "reserved_start_us": resolution.preview.start_us},
                ):
                    ticket = controller._commit_submitted_automated_request(
                        submit_started_ns=submit_started_ns,
                        context=context,
                        resolution=resolution,
                        estimates=estimates,
                        request=request,
                        selection_mode=selection_mode,
                        compiler=compiler,
                        observed_at_us=observed_at_us,
                    )
            except (RuntimeControllerError, UnifiedScheduleError) as exc:
                raise _NoAffinityGain(str(exc)) from exc
            return ticket
    except _NoAffinityGain as exc:
        runtime.record_dispatch_policy_event("affinity_refusals")
        runtime.record_dispatch_policy_refusal(
            "AFFINITY_REFUSED", request.request_id, str(exc), observed_at_us
        )
        return None
