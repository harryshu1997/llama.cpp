"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..phone_shards import PhoneFfnResidencyLayout
from ..model_placement_contracts.demand import ModelDemandSnapshot
from ..model_placement_contracts.layout import PhoneSessionResidencyState, ModelPhoneResidencyLayout
from ..model_placement_contracts.requests import RequestHelperRebind, RequestPlacementBinding


_INFORMATIONAL_NOTIFICATIONS = frozenset({"REQUEST_ARRIVAL"})


@dataclass(frozen=True)
class _ControllerCheckpoint:
    snapshots: tuple[tuple[str, ModelDemandSnapshot], ...]
    pending_reasons: tuple[tuple[str, tuple[str, ...]], ...]
    background_inflight: tuple[str, ...]
    last_refresh_us: tuple[tuple[str, int], ...]
    request_bindings: tuple[
        tuple[str, RequestPlacementBinding], ...
    ]
    acquired_request_ids: tuple[str, ...]
    request_decode_progress: tuple[
        tuple[str, tuple[int, int]], ...
    ]
    request_helper_events: tuple[Mapping[str, object], ...]
    request_helper_rebinds: tuple[
        tuple[str, RequestHelperRebind], ...
    ]
    phone_layout_generation: int
    phone_layouts: tuple[
        tuple[int, ModelPhoneResidencyLayout], ...
    ]
    phone_session_states: tuple[
        tuple[str, PhoneSessionResidencyState], ...
    ]
    phone_session_generation_by_id: tuple[tuple[str, int], ...]
    phone_session_replacement_sources: tuple[
        tuple[int, tuple[tuple[str, PhoneSessionResidencyState], ...]], ...
    ]
    ready_phone_layout_generation: int | None
    target_phone_layout_generation: int | None
    phone_preload_layouts: tuple[PhoneFfnResidencyLayout, ...]
    pending_phone_layout_geometry_sha256: str | None
    pending_phone_layout_snapshot_sha256: str | None
    pending_phone_layout_snapshot_count: int
    pending_phone_layout_sampled_at_us: int | None
    phone_layout_events: tuple[Mapping[str, object], ...]
    events: tuple[Mapping[str, object], ...]
    counters: tuple[tuple[str, int], ...]
