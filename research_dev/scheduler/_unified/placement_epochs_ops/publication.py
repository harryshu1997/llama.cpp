"""PlacementEpochMixin publication operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from ..._internal.model_placement_controller import ModelDemandSnapshot, ModelPlacementAction
from ..._internal.runtime_search import request_shape_bucket
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)
from .common import _PlacementChoice


def _placement_contract_identity(
    controller,
    choice: _PlacementChoice,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
):
    validity_us = snapshot.cost_features.get(
        "model_placement_epoch_validity_us", 600_000_000
    )
    if type(validity_us) is not int or validity_us < 1:
        raise UnifiedScheduleError(
            "model placement epoch validity is invalid"
        )
    contract = choice.selected.plan.execution_contract
    fractions = contract.allowed_adaptive_fractions_ppm
    if not fractions:
        fractions = (
            (0,) if contract.execution_mode == "desktop"
            else (contract.initial_split_fraction_ppm,)
        )
    component = runtime_residency_component_identity(
        manifest.artifact_sha256,
        choice.selected.plan,
        choice.selected.binding,
    )
    if (
        component.identity_sha256
            != choice.templates.selected_component_identity_sha256
        or choice.paired.plan.desktop_placement_sha256 is None
    ):
        raise UnifiedScheduleError(
            "model placement resident component differs"
        )
    phone_generation = None
    if choice.selected.plan.execution_contract.phone_shards:
        phone_layout = (
            controller._model_placement_controller.planning_phone_layout()
        )
        selected_geometry = (
            choice.selected.plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            )
        )
        if (
            phone_layout is None
            or selected_geometry
                != phone_layout.layout.geometry_sha256
        ):
            raise UnifiedScheduleError(
                "model placement phone layout is not authoritative: "
                + str(selected_geometry)
                + " != "
                + (
                    "absent"
                    if phone_layout is None
                    else phone_layout.layout.geometry_sha256
                )
            )
        phone_generation = phone_layout.generation
    return validity_us, fractions, component, phone_generation


def _build_model_placement_epoch(
    controller,
    *,
    request: Request,
    manifest: ModelManifest,
    observed_at_us: int,
    selection_mode: str,
    invalidation_reason: str,
    demand: ModelDemandSnapshot,
    statistics,
    choice: _PlacementChoice,
    validity_us: int,
    fractions: tuple[int, ...],
    component,
    phone_layout_generation: int | None,
    input_bucket: int,
    output_bucket: int,
    placement_learning_generation_sha256: str | None,
) -> RuntimeModelPlacementEpoch:
    return RuntimeModelPlacementEpoch(
        artifact_sha256=manifest.artifact_sha256,
        generation=(
            controller._runtime_residency_cohorts.next_published_generation(
                manifest.artifact_sha256,
                input_bucket,
                output_bucket,
                request.quality_requirement,
                selection_mode,
                controller._runtime_capabilities.maximum_latency_ppm,
            )
        ),
        observed_at_us=observed_at_us,
        active_request_count=statistics.active_request_count,
        virtual_queue_request_count=statistics.virtual_queue_request_count,
        component_opportunity_counts=statistics.component_opportunity_counts,
        reuse_projections=statistics.reuse_projections,
        selected_component_identity_sha256=(
            choice.templates.selected_component_identity_sha256
        ),
        route_template_identity_sha256=(
            choice.templates.route_template_identity_sha256
        ),
        selected_route_id=choice.selected.candidate_id,
        selected_executor_id=choice.selected.binding.executor_id,
        selected_operator_plan_sha256=choice.selected.plan.plan_sha256,
        capability_generation_sha256=(
            controller._runtime_capability_generation_sha256
        ),
        profile_generation_sha256=controller._runtime_profile_generation_sha256,
        transport_generation_sha256=(
            controller._runtime_transport_generation_sha256
        ),
        learning_generation_sha256=(
            controller._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            )
            if placement_learning_generation_sha256 is None
            else placement_learning_generation_sha256
        ),
        input_token_bucket=input_bucket,
        output_token_bucket=output_bucket,
        quality_requirement=request.quality_requirement,
        objective=selection_mode,
        maximum_latency_ppm=controller._runtime_capabilities.maximum_latency_ppm,
        expected_reuse_count=choice.expected_reuse_count,
        valid_from_us=observed_at_us,
        valid_until_us=observed_at_us + validity_us,
        demand_generation_sha256=demand.demand_generation_sha256,
        pressure_bucket=demand.pressure_bucket,
        trigger_reasons=choice.trigger_reasons,
        break_even_use_count=choice.break_even_use_count,
        selected_transition_latency_us=choice.transition_latency_us,
        selected_transition_energy_uj=choice.transition_energy_uj,
        old_component_identity_sha256=(
            demand.current_resident_component_identity_sha256
        ),
        selected_desktop_parent_route_id=choice.paired.candidate_id,
        selected_desktop_parent_plan_sha256=choice.paired.plan.plan_sha256,
        selected_desktop_parent_placement_sha256=(
            choice.paired.plan.desktop_placement_sha256
        ),
        selected_resident_artifact_sha256s=(
            component.resident_artifact_sha256s
        ),
        selected_resident_shard_geometry_sha256s=(
            component.resident_shard_geometry_sha256
        ),
        selected_session_resource_ids=component.session_resource_ids,
        phone_layout_generation=phone_layout_generation,
        allowed_adaptive_fractions_ppm=fractions,
        invalidation_reason=(
            None if invalidation_reason == "NONE"
            else invalidation_reason
        ),
    )


def _propose_model_placement_epoch(
    controller,
    *,
    request: Request,
    manifest: ModelManifest,
    candidate_set: AutomatedCandidateSet,
    selected: AutomatedRouteCandidate,
    observed_at_us: int,
    selection_mode: str,
    invalidation_reason: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    route_compiler: AutomatedRouteCompiler | None = None,
    demand_snapshot: ModelDemandSnapshot | None = None,
    placement_action: ModelPlacementAction | None = None,
    current_epoch: RuntimeModelPlacementEpoch | None = None,
    placement_learning_generation_sha256: str | None = None,
) -> tuple[RuntimeModelPlacementEpoch, RuntimeRouteTemplateSet]:
    if (
        controller._runtime_capabilities is None
        or controller._runtime_capability_generation_sha256 is None
        or controller._runtime_profile_generation_sha256 is None
        or controller._runtime_transport_generation_sha256 is None
    ):
        raise UnifiedScheduleError(
            "runtime placement generation is absent"
        )
    input_bucket, output_bucket = request_shape_bucket(
        request.input_tokens, request.output_tokens
    )
    compiler = (
        controller._automated_compiler()
        if route_compiler is None else route_compiler
    )
    templates = controller._compile_placement_templates(
        compiler,
        candidate_set,
        selected,
        manifest,
        request,
        snapshot,
        input_bucket,
        output_bucket,
    )
    statistics = controller._runtime_residency_cohorts.model_placement_epoch(
        manifest.artifact_sha256,
        observed_at_us,
        request_id=request.request_id,
        queued_request_ids=tuple(
            ticket.request.request_id
            for ticket in controller._runtime_controller.current_tickets()
            if ticket.dispatch_state in {
                "QUEUED", "REPLAN_REQUIRED"
            }
        ),
    )
    demand = (
        controller._model_demand_snapshot(
            request, manifest, snapshot, observed_at_us
        )
        if demand_snapshot is None else demand_snapshot
    )
    choice, basis = controller._initial_placement_choice(
        selected=selected,
        templates=templates,
        candidate_set=candidate_set,
        demand=demand,
        placement_action=placement_action,
        invalidation_reason=invalidation_reason,
    )
    choice = controller._authorize_placement_choice(
        choice=choice,
        basis=basis,
        compiler=compiler,
        candidate_set=candidate_set,
        manifest=manifest,
        request=request,
        snapshot=snapshot,
        demand=demand,
        current_epoch=current_epoch,
        selection_mode=selection_mode,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
    )
    choice = controller._authoritative_phone_placement_choice(
        choice=choice,
        compiler=compiler,
        candidate_set=candidate_set,
        manifest=manifest,
        request=request,
        snapshot=snapshot,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
    )
    validity, fractions, component, phone_generation = (
        controller._placement_contract_identity(choice, manifest, snapshot)
    )
    epoch = controller._build_model_placement_epoch(
        request=request,
        manifest=manifest,
        observed_at_us=observed_at_us,
        selection_mode=selection_mode,
        invalidation_reason=invalidation_reason,
        demand=demand,
        statistics=statistics,
        choice=choice,
        validity_us=validity,
        fractions=fractions,
        component=component,
        phone_layout_generation=phone_generation,
        input_bucket=input_bucket,
        output_bucket=output_bucket,
        placement_learning_generation_sha256=(
            placement_learning_generation_sha256
        ),
    )
    return epoch, choice.templates


def _candidate_set_with_epoch(
    candidate_set: AutomatedCandidateSet,
    epoch: RuntimeModelPlacementEpoch,
    *,
    fast_path: bool,
    invalidation_reason: str,
) -> AutomatedCandidateSet:
    return replace(
        candidate_set,
        search_metadata={
            **dict(candidate_set.search_metadata),
            "model_placement_epoch": epoch.to_json(),
            "model_placement_epoch_fast_path": fast_path,
            "model_placement_epoch_generation": epoch.generation,
            "model_placement_epoch_invalidation_reason": (
                invalidation_reason
            ),
            "model_placement_epoch_sha256": epoch.epoch_sha256,
            "route_template_identity_sha256": (
                epoch.route_template_identity_sha256
            ),
            "selected_residency_component_identity_sha256": (
                epoch.selected_component_identity_sha256
            ),
            "virtual_queue_request_count": (
                epoch.virtual_queue_request_count
            ),
        },
    )


def _publish_model_placement_epoch(
    controller,
    epoch: RuntimeModelPlacementEpoch,
    templates: RuntimeRouteTemplateSet,
) -> None:
    manifest = next((
        row for row in controller._runtime_manifests.values()
        if row.artifact_sha256 == epoch.artifact_sha256
    ), None)
    selected = next((
        row for row in templates.candidate_set.candidates
        if row.candidate_id == templates.selected_route_id
    ), None)
    if (
        not epoch.published
        or manifest is None
        or selected is None
        or templates.artifact_sha256 != epoch.artifact_sha256
        or templates.route_template_identity_sha256
            != epoch.route_template_identity_sha256
        or templates.selected_component_identity_sha256
            != epoch.selected_component_identity_sha256
        or templates.selected_route_id != epoch.selected_route_id
    ):
        raise UnifiedScheduleError(
            "model placement epoch template differs"
        )
    compatibility = controller._model_placement_compatibility(
        epoch, selected, manifest, templates.candidate_set
    )
    if not compatibility.compatible:
        raise UnifiedScheduleError(
            "model placement epoch component differs: "
            + ",".join(compatibility.rejection_reasons)
        )
    previous_epoch = (
        controller._runtime_residency_cohorts
        .published_model_placement_epoch(
            epoch.artifact_sha256,
            epoch.input_token_bucket,
            epoch.output_token_bucket,
            epoch.quality_requirement,
            epoch.objective,
            epoch.maximum_latency_ppm,
        )
    )
    pinned_before = (
        controller._runtime_residency_cohorts
        .published_model_placement_epoch_hashes()
    )
    replacing = previous_epoch is not None
    if (
        not replacing
        and len(pinned_before)
            >= controller._runtime_route_template_cache_maximum
    ):
        raise UnifiedScheduleError(
            "published model placement template capacity is exhausted"
        )
    previous_template = controller._runtime_route_template_sets.get(
        epoch.epoch_sha256
    )
    placement_key = (
        epoch.artifact_sha256,
        epoch.input_token_bucket,
        epoch.output_token_bucket,
        epoch.quality_requirement,
        epoch.objective,
    )
    previous_learning_generation = (
        controller._runtime_placement_learning_generation_by_artifact.get(
            epoch.artifact_sha256
        )
    )
    previous_learning_signature = (
        controller._runtime_placement_learning_signature_by_key.get(
            placement_key
        )
    )
    learning_signature = controller._runtime_placement_learning_signature(
        templates.candidate_set
    )
    controller._runtime_route_template_sets[epoch.epoch_sha256] = templates
    try:
        controller._runtime_residency_cohorts.publish_model_placement_epoch(
            epoch
        )
        controller._runtime_placement_learning_generation_by_artifact[
            epoch.artifact_sha256
        ] = epoch.learning_generation_sha256
        controller._runtime_placement_learning_signature_by_key[
            placement_key
        ] = learning_signature
    except BaseException:
        if previous_template is None:
            controller._runtime_route_template_sets.pop(
                epoch.epoch_sha256, None
            )
        else:
            controller._runtime_route_template_sets[
                epoch.epoch_sha256
            ] = previous_template
        if previous_learning_generation is None:
            controller._runtime_placement_learning_generation_by_artifact.pop(
                epoch.artifact_sha256, None
            )
        else:
            controller._runtime_placement_learning_generation_by_artifact[
                epoch.artifact_sha256
            ] = previous_learning_generation
        if previous_learning_signature is None:
            controller._runtime_placement_learning_signature_by_key.pop(
                placement_key, None
            )
        else:
            controller._runtime_placement_learning_signature_by_key[
                placement_key
            ] = previous_learning_signature
        raise
    pinned_after = (
        controller._runtime_residency_cohorts
        .published_model_placement_epoch_hashes()
    )
    removable = [
        template_sha256
        for template_sha256 in controller._runtime_route_template_sets
        if template_sha256 not in pinned_after
    ]
    while (
        len(controller._runtime_route_template_sets)
            > controller._runtime_route_template_cache_maximum
        or (
            previous_epoch is not None
            and previous_epoch.epoch_sha256
                in controller._runtime_route_template_sets
            and previous_epoch.epoch_sha256 not in pinned_after
        )
    ):
        if not removable:
            raise UnifiedScheduleError(
                "published model placement template cannot be evicted"
            )
        removed = removable.pop(0)
        controller._runtime_route_template_sets.pop(removed)
        controller._runtime_route_template_cache_evictions += 1
