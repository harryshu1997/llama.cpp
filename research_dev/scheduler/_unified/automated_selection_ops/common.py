"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

from ..._internal.policy import Request
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_resources import RuntimeResidencyProjectionToken


@dataclass(frozen=True)
class _ModelPlacementMaterializeContext:
    """Fixed inputs shared by every epoch materialization pass."""

    compiler: AutomatedRouteCompiler
    request: Request
    manifest: ModelManifest
    snapshot: HeterogeneousRuntimeSnapshot
    observed_at_us: int
    selection_mode: str
    invalidation_reason: str
    residency_holds: Mapping[str, object]
    memory_source_snapshot: HeterogeneousRuntimeSnapshot | None
    memory_not_before_by_resource: Mapping[str, int]
    memory_projection_token: RuntimeResidencyProjectionToken | None


class _ModelPlacementPass(NamedTuple):
    candidates: AutomatedCandidateSet
    selected: AutomatedRouteCandidate
    rejected: tuple[tuple[str, str], ...]
    reason: str
    memory_rejections: Mapping[str, str]
    compatible: bool


@dataclass(frozen=True)
class _AutomatedSelectionContext:
    """Validated inputs of one ``_select_automated_candidate`` call."""

    candidate_set: AutomatedCandidateSet
    request: Request
    baseline: AutomatedRouteCandidate
    excluded: frozenset[str]
    runtime_rejection: Mapping[str, str]
    candidate_by_id: Mapping[str, AutomatedRouteCandidate]
    baseline_available: bool
    selection_mode: str
    route_compiler: AutomatedRouteCompiler | None
    # The live view behind the candidates; absent for callers that select
    # without one, where continuous join then stays off.
    observed_at_us: int | None = None
    snapshot: HeterogeneousRuntimeSnapshot | None = None
