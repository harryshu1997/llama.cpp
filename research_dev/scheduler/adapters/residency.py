"""Physical validation for scheduler-issued residency transitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

from .._internal.plan_contracts.co_helpers import (
    PHONE_HELPERS_PARAMETER,
    phone_helpers_support,
)
from .._internal.runtime_plan import RuntimeResidencyEviction
from .contracts import (
    DORMANT_PHONE_FFN_RUNTIME_PARAMETER,
    PhysicalAdapterError,
    dormant_phone_ffn_parameters,
)


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(name + " is invalid")
    return value


@dataclass(frozen=True)
class PhysicalResidentEndpoint:
    executor_id: str
    endpoint: str
    artifact_sha256: str
    generation: int
    participant_device_ids: tuple[str, ...]
    replacement_resource_ids: tuple[str, ...]
    session_resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("executor_id", "endpoint", "artifact_sha256"):
            _text("physical residency " + name, getattr(self, name))
        if (
            not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.artifact_sha256[7:]
            )
            or type(self.generation) is not int
            or self.generation <= 0
        ):
            raise PhysicalAdapterError(
                "physical residency identity is invalid"
            )
        for name in (
            "participant_device_ids",
            "replacement_resource_ids",
            "session_resource_ids",
        ):
            values = tuple(getattr(self, name))
            if len(values) != len(set(values)) or any(
                type(value) is not str
                or not value
                or not value.isascii()
                for value in values
            ):
                raise PhysicalAdapterError(
                    "physical residency resources are invalid"
                )
            object.__setattr__(self, name, tuple(sorted(values)))


@dataclass(frozen=True)
class PhysicalPhoneSessionEndpoint:
    session_id: str
    executor_id: str
    endpoint: str
    artifact_sha256: str
    resident_geometry_sha256: str
    operator_plan_sha256: str
    session_generation: int
    device_id: str
    resident_bytes: int

    def __post_init__(self) -> None:
        for name in ("session_id", "executor_id", "endpoint", "device_id"):
            _text("physical phone session " + name, getattr(self, name))
        for name in (
            "artifact_sha256",
            "resident_geometry_sha256",
            "operator_plan_sha256",
        ):
            value = getattr(self, name)
            if (
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in value[7:]
                )
            ):
                raise PhysicalAdapterError(
                    "physical phone session identity is invalid"
                )
        if (
            type(self.session_generation) is not int
            or self.session_generation < 1
            or type(self.resident_bytes) is not int
            or self.resident_bytes <= 0
        ):
            raise PhysicalAdapterError(
                "physical phone session residency is invalid"
            )


def _plan_text(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    return value


def _plan_texts(value: object) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    result = tuple(_plan_text(row) for row in value)
    if len(result) != len(set(result)):
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    return tuple(sorted(result))


def _execution_plan_residency_identity(
    plan: Mapping[str, object],
) -> tuple[object, ...]:
    if (
        not isinstance(plan, Mapping)
        or plan.get("schema") != "research-scheduler-execution-plan-v1"
    ):
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    split_fraction = plan.get("split_fraction_ppm")
    if type(split_fraction) is not int or not 0 <= split_fraction < 1_000_000:
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    desktop_placement = _plan_text(
        plan.get("desktop_placement_sha256"), optional=True
    )
    if desktop_placement is not None and (
        not desktop_placement.startswith("sha256:")
        or len(desktop_placement) != 71
        or any(
            value not in "0123456789abcdef"
            for value in desktop_placement[7:]
        )
    ):
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    raw_operators = plan.get("operators")
    raw_memory = plan.get("memory_demands")
    if type(raw_operators) is not list or type(raw_memory) is not list:
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    operators = []
    observed_operator_ids = set()
    for raw in raw_operators:
        if not isinstance(raw, Mapping):
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        operator_id = _plan_text(raw.get("operator_id"))
        if operator_id in observed_operator_ids:
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        observed_operator_ids.add(operator_id)
        operator_fraction = raw.get("split_fraction_ppm")
        if (
            type(operator_fraction) is not int
            or not 0 <= operator_fraction < 1_000_000
        ):
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        raw_profiles = raw.get("kernel_profile_ids", [])
        if type(raw_profiles) is not list:
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        profiles = tuple(_plan_text(value) for value in raw_profiles)
        if len(profiles) != len(set(profiles)):
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        operators.append((
            operator_id,
            _plan_text(raw.get("operator_kind")),
            _plan_texts(raw.get("device_ids")),
            _plan_text(raw.get("split_axis")),
            operator_fraction,
            tuple(sorted(profiles)),
        ))
    if not operators:
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    weights = []
    observed_demand_ids = set()
    for raw in raw_memory:
        if not isinstance(raw, Mapping) or raw.get("kind") != "model_weights":
            continue
        demand_id = _plan_text(raw.get("demand_id"))
        required_bytes = raw.get("required_bytes")
        if (
            demand_id in observed_demand_ids
            or type(required_bytes) is not int
            or required_bytes <= 0
        ):
            raise PhysicalAdapterError(
                "physical residency execution plan is invalid"
            )
        observed_demand_ids.add(demand_id)
        weights.append((
            demand_id,
            _plan_text(raw.get("device_id")),
            _plan_text(raw.get("resource_id")),
            required_bytes,
            _plan_text(raw.get("lifetime")),
            _plan_text(raw.get("share_key"), optional=True),
            _plan_text(raw.get("replacement_group"), optional=True),
        ))
    if not weights:
        raise PhysicalAdapterError(
            "physical residency execution plan is invalid"
        )
    return (
        _plan_text(plan.get("route_family")),
        _plan_texts(plan.get("device_ids")),
        _plan_text(plan.get("assisted_operator_kind"), optional=True),
        _plan_text(plan.get("split_axis")),
        split_fraction,
        _plan_text(plan.get("overlap_kind")),
        _plan_text(plan.get("baseline_executor_id"), optional=True),
        desktop_placement,
        tuple(sorted(operators)),
        tuple(sorted(weights)),
    )


def _resident_model_identity(
    plan: Mapping[str, object],
) -> str | None:
    parameters = plan.get("adapter_parameters", {})
    if not isinstance(parameters, Mapping):
        raise PhysicalAdapterError(
            "physical residency adapter parameters are invalid"
        )
    value = parameters.get("resident_model_identity_sha256")
    if value is None:
        return None
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PhysicalAdapterError(
            "physical resident model identity is invalid"
        )
    return value


def physical_residency_parameters_match(
    resident_parameters: Mapping[str, int | str],
    requested_parameters: Mapping[str, int | str],
) -> bool:
    """Compare immutable launch identity while ignoring runtime policy."""
    resident = resident_parameters.get("resident_model_identity_sha256")
    requested = requested_parameters.get("resident_model_identity_sha256")
    if resident is None and requested is None:
        resident_values = dict(resident_parameters)
        requested_values = dict(requested_parameters)
        resident_dormant = resident_values.pop(
            DORMANT_PHONE_FFN_RUNTIME_PARAMETER, None
        )
        requested_dormant = requested_values.pop(
            DORMANT_PHONE_FFN_RUNTIME_PARAMETER, None
        )
        if resident_values != requested_values:
            return False
        if requested_dormant is None:
            return True
        if resident_dormant is None:
            return False
        try:
            resident_contract = dict(dormant_phone_ffn_parameters({
                DORMANT_PHONE_FFN_RUNTIME_PARAMETER: resident_dormant,
            }) or {})
            requested_contract = dict(dormant_phone_ffn_parameters({
                DORMANT_PHONE_FFN_RUNTIME_PARAMETER: requested_dormant,
            }) or {})
        except PhysicalAdapterError:
            return False
        if (
            json.dumps(
                resident_contract,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ) != resident_dormant
            or json.dumps(
                requested_contract,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ) != requested_dormant
        ):
            return False
        resident_mask = resident_contract.pop(
            "ffn_resident_layer_mask", None
        )
        requested_mask = requested_contract.pop(
            "ffn_resident_layer_mask", None
        )
        # several helper phones: each owner may serve a subset of its live layers
        resident_helpers = resident_contract.pop(PHONE_HELPERS_PARAMETER, None)
        requested_helpers = requested_contract.pop(PHONE_HELPERS_PARAMETER, None)
        return (
            type(resident_mask) is int
            and type(requested_mask) is int
            and requested_mask != 0
            and requested_mask & ~resident_mask == 0
            and resident_contract == requested_contract
            and (
                resident_helpers == requested_helpers
                or phone_helpers_support(resident_helpers, requested_helpers)
            )
        )
    return (
        type(resident) is str
        and type(requested) is str
        and resident == requested
    )


def physical_residency_supports_execution_plan(
    resident_plan: Mapping[str, object],
    requested_plan: Mapping[str, object],
) -> bool:
    """Check immutable loaded weights without comparing request memory."""
    try:
        resident_model = _resident_model_identity(resident_plan)
        requested_model = _resident_model_identity(requested_plan)
        if resident_model is not None or requested_model is not None:
            return (
                resident_model is not None
                and resident_model == requested_model
            )
        resident = _execution_plan_residency_identity(resident_plan)
        requested = _execution_plan_residency_identity(requested_plan)
    except PhysicalAdapterError:
        return False
    return resident == requested


def physical_transition_stop_set(
    live: Mapping[str, PhysicalResidentEndpoint],
    *,
    target_artifact_sha256: str,
    target_executor_id: str,
    target_endpoint: str,
    target_replacement_resource_ids: Sequence[str],
    target_session_resource_ids: Sequence[str],
    evictions: Sequence[RuntimeResidencyEviction],
    phone_sessions: Mapping[
        str, PhysicalPhoneSessionEndpoint
    ] | None = None,
) -> tuple[str, ...]:
    """Return only endpoints the exact transition ticket may replace."""
    _text("physical transition target artifact", target_artifact_sha256)
    if (
        not target_artifact_sha256.startswith("sha256:")
        or len(target_artifact_sha256) != 71
        or any(
            value not in "0123456789abcdef"
            for value in target_artifact_sha256[7:]
        )
    ):
        raise PhysicalAdapterError(
            "physical transition target artifact is invalid"
        )
    _text("physical transition target executor", target_executor_id)
    _text("physical transition target endpoint", target_endpoint)
    rows = dict(live)
    if any(
        key != row.executor_id
        or not isinstance(row, PhysicalResidentEndpoint)
        for key, row in rows.items()
    ):
        raise PhysicalAdapterError(
            "physical residency registry is invalid"
        )
    target_replacement = set(target_replacement_resource_ids)
    target_session = set(target_session_resource_ids)
    session_rows = {} if phone_sessions is None else dict(phone_sessions)
    if any(
        session_id != state.session_id
        or not isinstance(state, PhysicalPhoneSessionEndpoint)
        for session_id, state in session_rows.items()
    ):
        raise PhysicalAdapterError(
            "physical phone session registry is invalid"
        )
    evicted_executor_ids = set()
    for eviction in evictions:
        if not isinstance(eviction, RuntimeResidencyEviction):
            raise PhysicalAdapterError(
                "physical transition eviction is invalid"
            )
        if eviction.session_id is not None:
            state = session_rows.get(eviction.session_id)
            if (
                state is None
                or state.artifact_sha256 != eviction.artifact_sha256
                or state.session_generation != eviction.generation
                or state.device_id != eviction.device_id
                or state.resident_bytes != eviction.resident_bytes
                or state.resident_geometry_sha256
                    != eviction.resident_geometry_sha256
                or state.operator_plan_sha256
                    != eviction.operator_plan_sha256
            ):
                raise PhysicalAdapterError(
                    "transition eviction source differs from physical phone session"
                )
            evicted_executor_ids.add(state.executor_id)
            continue
        matches = tuple(
            state for state in rows.values()
            if state.artifact_sha256 == eviction.artifact_sha256
            and state.generation == eviction.generation
            and eviction.device_id in state.participant_device_ids
            and (
                eviction.executor_id is None
                or eviction.executor_id == state.executor_id
            )
            and (
                eviction.replacement_group is None
                or eviction.replacement_group
                    in state.replacement_resource_ids
            )
        )
        if len(matches) != 1:
            raise PhysicalAdapterError(
                "transition eviction source differs from physical endpoint"
            )
        evicted_executor_ids.add(matches[0].executor_id)
    conflicts = set(evicted_executor_ids)
    for state in rows.values():
        replacement_conflict = bool(
            set(state.replacement_resource_ids) & target_replacement
        )
        session_conflict = bool(
            set(state.session_resource_ids) & target_session
        )
        if (
            state.executor_id == target_executor_id
            or state.endpoint == target_endpoint
            or replacement_conflict
            or session_conflict
        ):
            conflicts.add(state.executor_id)
        if (
            replacement_conflict
            and (
                state.executor_id != target_executor_id
                or state.artifact_sha256 != target_artifact_sha256
            )
            and state.executor_id not in evicted_executor_ids
        ):
            raise PhysicalAdapterError(
                "exclusive residency replacement lacks an exact eviction"
            )
    return tuple(sorted(conflicts))


__all__ = [
    "PhysicalPhoneSessionEndpoint",
    "PhysicalResidentEndpoint",
    "physical_residency_parameters_match",
    "physical_residency_supports_execution_plan",
    "physical_transition_stop_set",
]
