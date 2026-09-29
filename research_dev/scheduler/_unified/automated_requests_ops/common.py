"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..._internal.policy import LeasePreview
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import RuntimeRouteTemplateSet
from ..._internal.model_placement_controller import ModelDemandSnapshot, ModelPlacementAction
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from ..._internal.runtime_controller import RuntimeRequestTicket


@dataclass
class _AutomatedSubmitContext:
    source_snapshot: HeterogeneousRuntimeSnapshot
    snapshot: HeterogeneousRuntimeSnapshot
    placement_epoch: RuntimeModelPlacementEpoch | None
    current_epoch_for_publication: RuntimeModelPlacementEpoch | None
    route_templates: RuntimeRouteTemplateSet | None
    demand_snapshot: ModelDemandSnapshot
    placement_action: ModelPlacementAction
    epoch_invalidation_reason: str
    live_not_before_by_resource: dict[str, int]
    timings_ns: dict[str, int]


@dataclass(frozen=True)
class _AutomatedSubmitResolution:
    candidate_set: AutomatedCandidateSet
    selected: AutomatedRouteCandidate
    preview: LeasePreview
    rejected: tuple[tuple[str, str], ...]
    reason: str
    publish_epoch: RuntimeModelPlacementEpoch | None
    publish_templates: RuntimeRouteTemplateSet | None
    residency_iterations: int


@dataclass(frozen=True)
class _AutomatedReplanPreparation:
    context: _AutomatedSubmitContext
    previous: RuntimeRequestTicket
    residency_holds: Mapping[str, object]
    prioritize_resident_component: bool
    hot_compatible_components: frozenset[str]
    # Re-projection starts from the replan's input snapshot, never from the
    # already projected context.snapshot.
    observed_snapshot: HeterogeneousRuntimeSnapshot
    projection_request_ids: tuple[str, ...]
