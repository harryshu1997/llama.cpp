"""PlacementEpochMixin lookup operations on its existing owner."""

from __future__ import annotations

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from ..._internal.runtime_search import request_shape_bucket
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)


def _record_epoch_refresh_failure(controller, reason: object) -> None:
    value = (
        reason
        if type(reason) is str
        else type(reason).__name__ + ": " + str(reason)
    )
    if not value.isascii():
        value = value.encode("ascii", "backslashreplace").decode("ascii")
    controller._runtime_epoch_background_refresh_failure_reasons[value] = (
        controller._runtime_epoch_background_refresh_failure_reasons.get(
            value, 0
        ) + 1
    )


def _record_epoch_invalidation(controller, reason: str) -> None:
    controller._runtime_epoch_invalidations[reason] = (
        controller._runtime_epoch_invalidations.get(reason, 0) + 1
    )


def _published_epoch_for_request(
    controller,
    request: Request,
    manifest: ModelManifest,
    observed_at_us: int,
    selection_mode: str,
) -> tuple[
    RuntimeModelPlacementEpoch | None,
    RuntimeRouteTemplateSet | None,
    str,
]:
    if controller._runtime_capabilities is None:
        return None, None, "RUNTIME_CAPABILITIES_ABSENT"
    input_bucket, output_bucket = request_shape_bucket(
        request.input_tokens, request.output_tokens
    )
    epoch = (
        controller._runtime_residency_cohorts
        .published_model_placement_epoch(
            manifest.artifact_sha256,
            input_bucket,
            output_bucket,
            request.quality_requirement,
            selection_mode,
            controller._runtime_capabilities.maximum_latency_ppm,
        )
    )
    compatible_shape = False
    if epoch is None:
        compatible = (
            controller._runtime_residency_cohorts
            .compatible_published_model_placement_epochs(
                manifest.artifact_sha256,
                request.quality_requirement,
                selection_mode,
                controller._runtime_capabilities.maximum_latency_ppm,
            )
        )
        epoch = next((
            row for row in compatible
            if row.capability_generation_sha256
                == controller._runtime_capability_generation_sha256
            and row.profile_generation_sha256
                == controller._runtime_profile_generation_sha256
            and row.transport_generation_sha256
                == controller._runtime_transport_generation_sha256
            and row.valid_from_us <= observed_at_us
                < row.valid_until_us
            and row.epoch_sha256
                in controller._runtime_route_template_sets
        ), None)
        if epoch is None:
            controller._runtime_route_template_cache_misses += 1
            return None, None, "MODEL_PLACEMENT_EPOCH_ABSENT"
        compatible_shape = True
    checks = (
        (
            observed_at_us < epoch.valid_from_us,
            "MODEL_PLACEMENT_EPOCH_NOT_YET_VALID",
        ),
        (
            observed_at_us >= epoch.valid_until_us,
            "MODEL_PLACEMENT_EPOCH_EXPIRED",
        ),
        (
            epoch.capability_generation_sha256
                != controller._runtime_capability_generation_sha256,
            "CAPABILITY_GENERATION_CHANGED",
        ),
        (
            epoch.profile_generation_sha256
                != controller._runtime_profile_generation_sha256,
            "PROFILE_GENERATION_CHANGED",
        ),
        (
            epoch.transport_generation_sha256
                != controller._runtime_transport_generation_sha256,
            "TRANSPORT_GENERATION_CHANGED",
        ),
    )
    for failed, reason in checks:
        if failed:
            controller._runtime_route_template_cache_misses += 1
            controller._record_epoch_invalidation(reason)
            return None, None, reason
    learning_changed = selection_mode != "desktop-baseline" and (
        epoch.learning_generation_sha256
            != controller._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            )
    )
    templates = controller._runtime_route_template_sets.get(
        epoch.epoch_sha256
    )
    if templates is None:
        controller._runtime_route_template_cache_misses += 1
        reason = "ROUTE_TEMPLATE_EVICTED"
        controller._record_epoch_invalidation(reason)
        return None, None, reason
    if (
        templates.route_template_identity_sha256
            != epoch.route_template_identity_sha256
        or templates.selected_component_identity_sha256
            != epoch.selected_component_identity_sha256
        or templates.artifact_sha256 != manifest.artifact_sha256
        or templates.quality_requirement
            != request.quality_requirement
    ):
        controller._runtime_route_template_cache_misses += 1
        reason = "ROUTE_TEMPLATE_IDENTITY_CHANGED"
        controller._record_epoch_invalidation(reason)
        return None, None, reason
    controller._runtime_route_template_cache_hits += 1
    return (
        epoch,
        templates,
        (
            "LEARNING_GENERATION_CHANGED"
            if learning_changed
            else "MODEL_PLACEMENT_EPOCH_COMPATIBLE_SHAPE"
            if compatible_shape
            else "NONE"
        ),
    )


def _route_template_phone_helper_route_ids(
    controller,
    templates: RuntimeRouteTemplateSet,
    manifest: ModelManifest,
) -> tuple[str, ...]:
    layout = controller._model_placement_controller.planning_phone_layout()
    layout_geometry = (
        None
        if layout is None
        or layout.state not in {"PROPOSED", "PREPARING", "READY"}
        or not layout.covers_artifact(manifest.artifact_sha256)
        else layout.layout.geometry_sha256
    )
    active_components = {
        runtime_residency_component_identity(
            ticket.model.artifact_sha256,
            ticket.execution_plan,
            ticket.binding,
        ).identity_sha256
        for ticket in controller._runtime_controller.current_tickets()
        if ticket.model.artifact_sha256 == manifest.artifact_sha256
        and ticket.execution_plan is not None
        and ticket.decode_cohort is not None
        and ticket.dispatch_state in {
            "ACQUIRED", "QUEUED", "REPLAN_REQUIRED"
        }
    }
    return tuple(sorted(
        candidate.candidate_id
        for candidate in templates.candidate_set.candidates
        if (
            layout_geometry is not None
            and candidate.plan.execution_contract.phone_shards
            and candidate.plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            ) == layout_geometry
        )
        or runtime_residency_component_identity(
            manifest.artifact_sha256,
            candidate.plan,
            candidate.binding,
        ).identity_sha256 in active_components
    ))


def _compile_placement_templates(
    controller,
    compiler: AutomatedRouteCompiler,
    candidate_set: AutomatedCandidateSet,
    selected: AutomatedRouteCandidate,
    manifest: ModelManifest,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    input_bucket: int,
    output_bucket: int,
) -> RuntimeRouteTemplateSet:
    return compiler.compile_route_template_set(
        candidate_set,
        selected,
        manifest,
        input_token_bucket=input_bucket,
        output_token_bucket=output_bucket,
        quality_requirement=request.quality_requirement,
        snapshot=snapshot,
    )


def _placement_transition_cost(
    selected: AutomatedRouteCandidate,
) -> tuple[int, int]:
    return (
        sum(row.latency_us for row in selected.plan.transitions),
        (
            selected.cost.transition_energy_upper_uj
            if selected.cost.transition_energy_upper_uj is not None
            else sum(
                row.energy_uj for row in selected.plan.transitions
            )
        ),
    )
