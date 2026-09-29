"""Execution-plan contracts grouped by responsibility: common."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


RUNTIME_EXECUTION_PLAN_SCHEMA = "research-scheduler-execution-plan-v1"


AUTOMATED_CANDIDATE_SET_SCHEMA = "research-scheduler-candidate-set-v1"


RUNTIME_TRANSITION_RECEIPT_STATUSES = frozenset({"COMPLETED", "FAILED"})


RUNTIME_EXECUTION_RECEIPT_STATUSES = frozenset({"COMPLETED"})


RUNTIME_EXECUTION_MODES = frozenset({
    "adaptive-split",
    "desktop",
    "static-split",
})


RUNTIME_BATCH_PLANS = frozenset({
    "coalesced-batch",
    "none",
    "single",
    "split-row",
})


HELPER_OPPORTUNITY_EVIDENCE_STATES = frozenset({
    "UNSUPPORTED",
    "EXECUTABLE",
    "LEARNING",
    "TRUSTED",
})


class RuntimePlanError(ValueError):
    pass


_REJECTION_STAGE = {
    "COMPOSITE_COORDINATOR_ABSENT": 0,
    "EXECUTION_PLAN_ABSENT": 0,
    "KERNEL_SUPPORT_ABSENT": 0,
    "PLACEMENT_INFEASIBLE": 0,
    "QUANTIZATION_BLOCK_MISALIGNED": 0,
    "RUNTIME_PARTITION_CAPACITY": 0,
    "EXECUTOR_OBSERVATION_ABSENT": 1,
    "EXECUTOR_UNHEALTHY": 1,
    "EXECUTOR_NOT_READY": 1,
    "PHONE_SESSION_NOT_READY": 1,
    "REMOTE_RESIDENT_OWNER_NOT_READY": 1,
    "EXECUTOR_CAPACITY_UNAVAILABLE": 1,
    "HELPER_RESIDENCY_CAPACITY": 1,
    "RESIDENCY_EXECUTOR_MISMATCH": 2,
    "RESIDENCY_TRANSITION_ABSENT": 2,
    "RESIDENCY_VARIANT_NOT_CURRENT": 2,
    "MEMORY_RESOURCE_ABSENT": 3,
    "MEMORY_CAPACITY": 3,
    "LINK_OBSERVATION_ABSENT": 4,
    "LINK_NOT_READY": 4,
    "TRANSFER_PLAN_ABSENT": 4,
    "TRANSPORT_PROFILE_INCOMPLETE": 4,
    "RESOURCE_CALENDAR_INFEASIBLE": 5,
    "REQUEST_EXCEEDS_CONTEXT_CAPACITY": 0,
    "REQUEST_EXCEEDS_DESKTOP_CONTROL_CONTEXT": 0,
    "THERMAL_LIMIT": 5,
    "BATTERY_LIMIT": 5,
    "PROTECTED_WORK_DELAY": 5,
    "MARGINAL_SYSTEM_COST_UNKNOWN": 6,
    "COLD_WARM_ENERGY_DECOMPOSITION_INVALID": 6,
    "COLD_WARM_ENERGY_DECOMPOSITION_UNKNOWN": 6,
    "ROUTE_NOT_QUALIFIED": 6,
    "ENERGY_UNKNOWN": 7,
    "SLO_UPPER_BOUND": 8,
    "PARETO_DOMINATED": 9,
    "SEARCH_COVERAGE_ONLY": 9,
}


def primary_rejection_reason(reasons: tuple[str, ...]) -> str | None:
    """Return the earliest physical cause while preserving every reason."""
    if not reasons:
        return None
    return min(reasons, key=lambda value: (
        _REJECTION_STAGE.get(value.split(":", 1)[0], 10), value
    ))


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimePlanError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimePlanError(f"{name} must be an integer >= {minimum}")
    return value


def _estimation_metadata(
    values: Mapping[str, int | str | bool],
) -> Mapping[str, int | str | bool]:
    result: dict[str, int | str | bool] = {}
    for raw_name, raw_value in values.items():
        name = _text("receipt energy estimation metadata", raw_name)
        if type(raw_value) is bool:
            result[name] = raw_value
        elif type(raw_value) is int and raw_value >= 0:
            result[name] = raw_value
        elif type(raw_value) is str:
            result[name] = _text(
                "receipt energy estimation metadata value", raw_value
            )
        else:
            raise RuntimePlanError(
                "receipt energy estimation metadata is invalid"
            )
    return MappingProxyType(dict(sorted(result.items())))


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value)
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(character not in "0123456789abcdef" for character in digest[7:])
    ):
        raise RuntimePlanError(name + " must be SHA-256")
    return digest
