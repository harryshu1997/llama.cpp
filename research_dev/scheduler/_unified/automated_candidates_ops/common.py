"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..._internal.route_generation import PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS
from ..._internal.runtime_plan import AutomatedRouteCandidate
from ..._internal.adaptive_decode_contracts import AdaptiveDecodePolicy


_ADAPTIVE_HISTORY_ALLOWED_REJECTIONS = frozenset({
    "COLD_RESIDENCY_BREAK_EVEN",
    "ENERGY_UNKNOWN",
    "MODEL_EPOCH_AUDIT_ONLY",
    "PHONE_RESIDENCY_LAYOUT_NOT_SELECTED",
    "ROUTE_NOT_QUALIFIED",
    "SLO_UPPER_BOUND",
    *PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS,
})


@dataclass(frozen=True)
class _AdaptiveHistoryContext:
    history: object | None
    physical_latency: tuple[int, int, int] | None
    physical_latency_policy: AdaptiveDecodePolicy | None
    use_assumed_phone_power: bool
    history_uses_assumed_phone_power: bool
    history_latency_evidence: str
    subset_evidence: Mapping[str, object] | None
    assumed_phone_profile_sha256s: tuple[str, ...]


@dataclass(frozen=True)
class _AdaptiveWarmEstimate:
    warm_lower_uj: int
    warm_energy_uj: int
    warm_upper_uj: int
    parent_component_service_us: int
    parent_component_service_upper_us: int
    selected_component_service_us: int
    selected_component_service_upper_us: int
    parent_warm_lower_uj: int
    parent_warm_energy_uj: int
    parent_warm_upper_uj: int


@dataclass(frozen=True)
class _TransitionEnergyEstimate:
    lower_uj: int
    energy_uj: int
    upper_uj: int
    qualified: bool


@dataclass(frozen=True)
class _AdaptiveEnergyDeltas:
    paired_warm_lower_uj: int
    paired_warm_energy_uj: int
    paired_warm_upper_uj: int
    paired_transition_lower_uj: int
    paired_transition_energy_uj: int
    paired_transition_upper_uj: int
    paired_lower_uj: int
    paired_energy_uj: int
    paired_upper_uj: int


@dataclass(frozen=True)
class _ExactRouteDecompositions:
    qualified: bool
    parent: tuple[int, int, int, int, int, int] | None
    candidate: tuple[int, int, int, int, int, int] | None


@dataclass(frozen=True)
class _AdaptiveCandidateApplication:
    baseline: AutomatedRouteCandidate
    candidate: AutomatedRouteCandidate
    learned_baseline: AutomatedRouteCandidate | None
    changed: bool


@dataclass(frozen=True)
class _AdaptiveCohortBounds:
    candidate_upper_uj: int
    parent_lower_uj: int
    paired_candidate_upper_uj: int
    paired_warm_candidate_upper_uj: int
    paired_cohort_upper_uj: int
    route_energy_qualified: bool
