"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass

from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import RuntimeRouteTemplateSet
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from ..._internal.runtime_controller import RuntimeRequestTicket


@dataclass(frozen=True)
class _PlacementRefreshDemand:
    ticket: RuntimeRequestTicket
    manifest: ModelManifest
    snapshot: HeterogeneousRuntimeSnapshot
    stale_epoch: RuntimeModelPlacementEpoch
    templates: RuntimeRouteTemplateSet
    reuse_projections: object
    expected_reuse_count: int
    observed_at_us: int


@dataclass(frozen=True)
class _PlacementRefreshResult:
    demand: _PlacementRefreshDemand
    candidate_set: AutomatedCandidateSet | None
    error: BaseException | None


@dataclass(frozen=True)
class _PlacementTicketDemand:
    all_tickets: tuple[RuntimeRequestTicket, ...]
    active: tuple[RuntimeRequestTicket, ...]
    queued: tuple[RuntimeRequestTicket, ...]
    queued_request_count: int
    queued_input_tokens: int
    queued_output_tokens: int
    oldest_wait_us: int
    predicted_drain_us: int


@dataclass(frozen=True)
class _PlacementChoice:
    selected: AutomatedRouteCandidate
    templates: RuntimeRouteTemplateSet
    paired: AutomatedRouteCandidate
    expected_reuse_count: int
    observed_reuse_count: int
    trigger_reasons: tuple[str, ...]
    transition_latency_us: int
    transition_energy_uj: int
    break_even_use_count: int | None


@dataclass(frozen=True)
class _PlacementPublicationBasis:
    paired_warm_upper_uj: int | None
    transition_energy_uj: int
    restore_energy_uj: int
    predicted_reuse_count: int
