"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from dataclasses import dataclass

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_plan import HelperOpportunity, RuntimeHelperExecutionEnvelope
from ..._internal.runtime_residency_cohorts import RuntimeResidencyComponentIdentity
from ..._internal.adaptive_decode_contracts import AdaptiveDecodePolicy


@dataclass(frozen=True)
class _ReadyHelperMaterialization:
    helper: RuntimeHelperExecutionEnvelope
    opportunity: HelperOpportunity
    generated_parent_route_id: str
    baseline: AdaptiveDecodePolicy
    policies: tuple[AdaptiveDecodePolicy, ...]
    ticket_policy: AdaptiveDecodePolicy
    component: RuntimeResidencyComponentIdentity


class _ReadyHelperParentUnavailable(UnifiedScheduleError):
    pass
