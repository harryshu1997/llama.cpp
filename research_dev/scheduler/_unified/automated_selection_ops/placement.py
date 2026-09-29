"""AutomatedSelectionMixin placement operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.policy import Request, SchedulerError
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.route_generation import AutomatedRouteCompiler, RouteGenerationError
from ..._internal.model_placement_controller import ModelPlacementControllerError
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_projection import (
    RuntimeResidencyProjectionError,
    project_scheduler_residency,
)
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)
from ..common import _ModelPlacementCompatibility, _text


def generate_automated_candidates(
    controller,
    request: Request,
    model_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    observed_at_us: int | None = None,
) -> AutomatedCandidateSet:
    """Generate and cost the bounded capability-backed search frontier."""
    if not isinstance(request, Request):
        raise UnifiedScheduleError("automated request is invalid")
    try:
        request.validate()
        snapshot = controller._automated_snapshot_for_request(
            request, snapshot
        )
        observed_at_us = (
            request.arrival_us
            if observed_at_us is None else observed_at_us
        )
        return controller._generate_automated_candidate_set(
            request,
            controller.runtime_model_manifest(model_id),
            snapshot,
            observed_at_us,
        )
    except (
        SchedulerError,
        RouteGenerationError,
        RuntimeCapabilityError,
        RuntimeResidencyProjectionError,
    ) as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def _automated_snapshot_for_request(
    controller,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    exclude_request_id: str | None = None,
    project_before_us: int | None = None,
    project_request_ids: Sequence[str] | None = None,
) -> HeterogeneousRuntimeSnapshot:
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError("runtime system snapshot is invalid")
    features = dict(snapshot.cost_features)
    changed = False
    for name, value in request.features.items():
        if name in features and features[name] != value:
            raise UnifiedScheduleError(
                "request and runtime cost feature differ: " + name
            )
        if name not in features:
            changed = True
        features[name] = value
    if changed:
        snapshot = replace(snapshot, cost_features=features)
    if controller._runtime_capabilities is None:
        return snapshot
    current_tickets = controller._runtime_controller.current_tickets()
    projection_request_ids = tuple(
        controller._runtime_controller.projection_request_ids()
        if project_request_ids is None else project_request_ids
    )
    if len(projection_request_ids) != len(set(projection_request_ids)):
        raise UnifiedScheduleError(
            "runtime residency projection requests are duplicated"
        )
    ticket_by_request_id = {
        ticket.request.request_id: ticket
        for ticket in current_tickets
    }
    tickets = tuple(
        ticket_by_request_id[request_id]
        for request_id in projection_request_ids
        if request_id in ticket_by_request_id
    )
    preserve_ticket_order = project_request_ids is not None
    return project_scheduler_residency(
        snapshot,
        controller._runtime_capabilities,
        tickets,
        controller._runtime_manifests,
        exclude_request_id=exclude_request_id,
        stop_before_us=project_before_us,
        preserve_ticket_order=preserve_ticket_order,
        causal_predecessors=(
            None
            if preserve_ticket_order
            else controller._runtime_controller.projection_causal_predecessors()
        ),
    )


def _model_placement_candidate_set(
    candidate_set: AutomatedCandidateSet,
) -> AutomatedCandidateSet:
    """Build a non-dispatching view for model epoch comparison."""
    marker = "MODEL_EPOCH_AUDIT_ONLY"
    rows = []
    for candidate in candidate_set.candidates:
        if marker not in candidate.rejection_reasons:
            rows.append(candidate)
            continue
        reasons = tuple(
            reason for reason in candidate.rejection_reasons
            if reason != marker
        )
        binding_reasons = tuple(
            reason for reason in candidate.binding.eligibility_reasons
            if reason != marker
        )
        rows.append(replace(
            candidate,
            binding=replace(
                candidate.binding,
                ready=not binding_reasons,
                eligibility_reasons=binding_reasons,
            ),
            admitted=not reasons and not binding_reasons,
            rejection_reasons=reasons,
        ))
    return replace(candidate_set, candidates=tuple(rows))


def _model_placement_compatibility(
    controller,
    epoch: RuntimeModelPlacementEpoch,
    candidate: AutomatedRouteCandidate,
    manifest: ModelManifest,
    candidate_set: AutomatedCandidateSet | None = None,
) -> _ModelPlacementCompatibility:
    component = runtime_residency_component_identity(
        manifest.artifact_sha256,
        candidate.plan,
        candidate.binding,
    )
    reasons = []
    if epoch.artifact_sha256 != manifest.artifact_sha256:
        reasons.append("EPOCH_ARTIFACT_MISMATCH")
    desktop_placement_sha256 = (
        candidate.plan.desktop_placement_sha256
    )
    if candidate_set is not None:
        parent_route_id = (
            candidate.paired_baseline_route_id
            or epoch.selected_desktop_parent_route_id
        )
        parent = next((
            row for row in candidate_set.candidates
            if row.candidate_id
                == parent_route_id
        ), None)
        if parent is None:
            parent = candidate_set.baseline
        if parent is not None:
            desktop_placement_sha256 = (
                parent.plan.desktop_placement_sha256
            )
    if desktop_placement_sha256 != (
        epoch.selected_desktop_parent_placement_sha256
    ):
        reasons.append("EPOCH_DESKTOP_PLACEMENT_MISMATCH")
    if (
        component.resident_artifact_sha256s
            != epoch.selected_resident_artifact_sha256s
    ):
        reasons.append("EPOCH_RESIDENT_ARTIFACT_MISMATCH")
    if (
        component.resident_shard_geometry_sha256
            != epoch.selected_resident_shard_geometry_sha256s
    ):
        reasons.append("EPOCH_SHARD_GEOMETRY_MISMATCH")
    if (
        component.session_resource_ids
            != epoch.selected_session_resource_ids
    ):
        reasons.append("EPOCH_SESSION_RESOURCE_MISMATCH")
    candidate_uses_phone_layout = bool(
        candidate.plan.execution_contract.phone_shards
    )
    epoch_uses_phone_layout = (
        epoch.phone_layout_generation is not None
    )
    if candidate_uses_phone_layout != epoch_uses_phone_layout:
        reasons.append("EPOCH_PHONE_LAYOUT_GENERATION_MISMATCH")
    if epoch.phone_layout_generation is not None:
        try:
            phone_layout = controller._model_placement_controller.phone_layout(
                epoch.phone_layout_generation
            )
        except ModelPlacementControllerError:
            reasons.append("EPOCH_PHONE_LAYOUT_GENERATION_STALE")
        else:
            candidate_geometry = candidate.plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            )
            if (
                candidate_geometry
                    != phone_layout.layout.geometry_sha256
                or not phone_layout.covers_artifact(
                    manifest.artifact_sha256
                )
            ):
                reasons.append("EPOCH_PHONE_LAYOUT_MISMATCH")
    if (
        component.identity_sha256
            != epoch.selected_component_identity_sha256
    ):
        reasons.append("EPOCH_COMPONENT_MISMATCH")
    fraction = (
        candidate.plan.execution_contract
            .initial_split_fraction_ppm
    )
    if fraction not in epoch.allowed_adaptive_fractions_ppm:
        reasons.append("EPOCH_ADAPTIVE_FRACTION_NOT_ALLOWED")
    reasons = tuple(sorted(set(reasons)))
    return _ModelPlacementCompatibility(
        compatible=not reasons,
        component=component,
        rejection_reasons=reasons,
    )


def _authorize_epoch_compatible_routes(
    controller,
    candidate_set: AutomatedCandidateSet,
    epoch: RuntimeModelPlacementEpoch,
    manifest: ModelManifest,
) -> AutomatedCandidateSet:
    marker = "MODEL_EPOCH_AUDIT_ONLY"
    live_route_values = candidate_set.search_metadata.get(
        "route_template_live_route_ids"
    )
    live_route_ids = (
        None
        if type(live_route_values) not in {list, tuple}
        else frozenset(live_route_values)
    )
    rows = []
    for candidate in candidate_set.candidates:
        if (
            live_route_ids is not None
            and candidate.candidate_id not in live_route_ids
        ):
            rows.append(candidate)
            continue
        compatibility = controller._model_placement_compatibility(
            epoch, candidate, manifest, candidate_set
        )
        if not compatibility.compatible or (
            marker not in candidate.rejection_reasons
            and marker not in candidate.binding.eligibility_reasons
        ):
            rows.append(candidate)
            continue
        reasons = tuple(sorted(set(
            reason for reason in (
                candidate.rejection_reasons
                + candidate.binding.eligibility_reasons
            )
            if reason != marker
        )))
        rows.append(replace(
            candidate,
            binding=replace(
                candidate.binding,
                ready=not reasons,
                eligibility_reasons=reasons,
            ),
            admitted=not reasons,
            rejection_reasons=reasons,
        ))
    return replace(candidate_set, candidates=tuple(rows))


def _candidate_set_with_placement_resolution(
    candidate_set: AutomatedCandidateSet,
    passes: Sequence[Mapping[str, object]],
    *,
    outcome: str,
    invalidated_epoch_sha256: str | None = None,
) -> AutomatedCandidateSet:
    if not passes or len(passes) > 2:
        raise UnifiedScheduleError(
            "model placement resolution pass count is invalid"
        )
    metadata = dict(candidate_set.search_metadata)
    metadata["model_placement_resolution"] = {
        "invalidated_epoch_sha256": invalidated_epoch_sha256,
        "outcome": _text(
            "model placement resolution outcome", outcome
        ),
        "pass_count": len(passes),
        "passes": [dict(row) for row in passes],
        "schema": "runtime-model-placement-resolution-v1",
    }
    return replace(candidate_set, search_metadata=metadata)


def _candidate_set_without_published_epoch(
    candidate_set: AutomatedCandidateSet,
) -> AutomatedCandidateSet:
    metadata = dict(candidate_set.search_metadata)
    for name in (
        "model_placement_epoch",
        "model_placement_epoch_fast_path",
        "model_placement_epoch_invalidation_reason",
        "route_template_identity_sha256",
        "selected_residency_component_identity_sha256",
    ):
        metadata.pop(name, None)
    return replace(candidate_set, search_metadata=metadata)


def _model_placement_resolution_pass(
    controller,
    epoch: RuntimeModelPlacementEpoch,
    proposed_route_id: str,
    selected: AutomatedRouteCandidate,
    manifest: ModelManifest,
    candidate_set: AutomatedCandidateSet,
) -> Mapping[str, object]:
    compatibility = controller._model_placement_compatibility(
        epoch, selected, manifest, candidate_set
    )
    return MappingProxyType({
        "epoch_generation": epoch.generation,
        "live_component_identity_sha256": (
            compatibility.component.identity_sha256
        ),
        "live_selected_route_id": selected.candidate_id,
        "proposed_component_identity_sha256": (
            epoch.selected_component_identity_sha256
        ),
        "proposed_route_id": _text(
            "model placement proposed route", proposed_route_id
        ),
        "rejection_reasons": list(
            compatibility.rejection_reasons
        ),
    })


def _select_model_placement_candidate(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    observed_at_us: int | None = None,
    *,
    selection_mode: str = "energy-aware",
    runtime_rejections: Mapping[str, str] | None = None,
    route_compiler: AutomatedRouteCompiler | None = None,
    snapshot: HeterogeneousRuntimeSnapshot | None = None,
) -> tuple[
    AutomatedRouteCandidate,
    tuple[tuple[str, str], ...],
    str,
]:
    """Compare prospective epochs without making them dispatchable."""
    prospective = controller._model_placement_candidate_set(candidate_set)
    return controller._select_automated_candidate(
        prospective,
        request,
        observed_at_us,
        selection_mode=selection_mode,
        runtime_rejections=runtime_rejections,
        route_compiler=route_compiler,
        snapshot=snapshot,
    )
