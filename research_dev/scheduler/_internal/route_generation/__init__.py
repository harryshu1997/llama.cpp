"""Capability-driven route generation and GGUF work cost materialization."""

from __future__ import annotations

from .common import (
    RouteGenerationError,
    DesktopControlUnavailableError,
    PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS,
    CO_HELPER_UNAVAILABLE,
    unavailable_co_helpers,
    _DORMANT_PHONE_FFN_RUNTIME_PARAMETER,
    _static_executor_identity,
    _static_coordinator_identity,
    _MATURITY_RANK,
    _ceil_div,
    _minimum_maturity,
    _FfnResidentEnvelope,
    _PhoneResidencyRouteEvidence,
    RuntimeRouteTemplateSet,
    _placement_rejection_reasons,
    _Pattern,
)
from .compiler import AutomatedRouteCompiler
from .feasibility import ThermalGateLog
from .conversion import (
    candidate_set_to_runtime_costs,
)

__all__ = [
    "RouteGenerationError",
    "DesktopControlUnavailableError",
    "PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS",
    "CO_HELPER_UNAVAILABLE",
    "unavailable_co_helpers",
    "_DORMANT_PHONE_FFN_RUNTIME_PARAMETER",
    "_static_executor_identity",
    "_static_coordinator_identity",
    "_MATURITY_RANK",
    "_ceil_div",
    "_minimum_maturity",
    "_FfnResidentEnvelope",
    "_PhoneResidencyRouteEvidence",
    "RuntimeRouteTemplateSet",
    "_placement_rejection_reasons",
    "_Pattern",
    "AutomatedRouteCompiler",
    "ThermalGateLog",
    "candidate_set_to_runtime_costs",
]
