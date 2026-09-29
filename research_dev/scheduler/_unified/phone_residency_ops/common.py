"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..._internal.runtime_capabilities import RuntimePhoneSessionCapability
from ..._internal.model_placement_controller import (
    ModelPhoneResidencyLayout,
    PhoneSessionMarginalGain,
    PhoneLayoutRequestImpact,
)
from ..._internal.phone_shards import PhoneFfnResidencyDemand, PhoneFfnResidencyLayout


@dataclass(frozen=True)
class _PhoneQueueDemand:
    active_count_by_artifact: dict[str, int]
    queued_count_by_artifact: dict[str, int]
    active_remaining_tokens_by_artifact: dict[str, int]
    queued_output_tokens_by_artifact: dict[str, int]
    queued_work_by_artifact: dict[str, int]
    # set by phone_residency_ops.reprovision when the layout demand follows the desktop
    reprovision: object | None = None


@dataclass(frozen=True)
class _PhoneDemandDiscovery:
    demand_rows: tuple[PhoneFfnResidencyDemand, ...]
    sessions: tuple[RuntimePhoneSessionCapability, ...] | None
    helper_id: str | None
    route_evidence_by_artifact: dict[str, Mapping[str, object]]


@dataclass(frozen=True)
class _OfflineLearningDemand:
    demand: PhoneFfnResidencyDemand
    sessions: tuple[RuntimePhoneSessionCapability, ...]
    helper_id: str
    status: Mapping[str, object]
    # a model that releases its host FFN share claims a contested session domain first
    release_priority: bool = False


@dataclass(frozen=True)
class _PhoneMemoryBudget:
    current: PhoneFfnResidencyLayout | None
    planning: PhoneFfnResidencyLayout | None
    accepted_resident_bytes: int
    live_capacity: object | None
    persistent_service_reserve_by_artifact: Mapping[str, int]
    persistent_service_reserve_bytes: int
    live_phone_wide_limit: int | None
    phone_wide_limit: int
    configured_htp_cap_bytes: int | None = None
    htp_workspace_bytes: int = 0
    persistent_service_peak_by_artifact: Mapping[str, int] = field(default_factory=dict)
    reservation_error: str | None = None

    def cap_evidence(self) -> dict[str, object]:
        if self.configured_htp_cap_bytes is None:
            return {}
        return {
            "phone_memory_htp_cap_bytes": self.configured_htp_cap_bytes,
            "phone_memory_htp_workspace_bytes": self.htp_workspace_bytes,
            "phone_memory_service_peak_by_artifact": dict(self.persistent_service_peak_by_artifact),
            "phone_memory_reservation_error": self.reservation_error,
        }


@dataclass(frozen=True)
class _PhoneLayoutDecision:
    layouts: tuple[PhoneFfnResidencyLayout, ...]
    current: PhoneFfnResidencyLayout | None
    planning: PhoneFfnResidencyLayout | None
    evaluated_selected: PhoneFfnResidencyLayout | None
    selected: PhoneFfnResidencyLayout | None
    selected_state: ModelPhoneResidencyLayout | None
    reason: str
    session_marginal_gains: tuple[PhoneSessionMarginalGain, ...]
    transition_latencies: Mapping[str, int]
    switching_margin_uj: int
    minimum_residency_us: int
    selection_confirmed: bool
    selection_snapshot_count: int
    selection_snapshot_sha256: str
    request_impacts_by_geometry: Mapping[str, tuple[PhoneLayoutRequestImpact, ...]] = field(default_factory=dict)
    reprovision: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _PhoneCandidateChoice:
    layouts: tuple[PhoneFfnResidencyLayout, ...]
    selected: PhoneFfnResidencyLayout | None
    reason: str
    marginal_gains: tuple[PhoneSessionMarginalGain, ...]
    transition_latencies: Mapping[str, int]
    force: bool
    switching_margin_uj: int
    minimum_residency_us: int
    request_impacts_by_geometry: Mapping[str, tuple[PhoneLayoutRequestImpact, ...]] = field(default_factory=dict)
    # a live desktop commitment confirms without the snapshot debounce
    confirmed: bool = False
    reprovision: Mapping[str, object] | None = None
