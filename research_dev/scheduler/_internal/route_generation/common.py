"""Shared errors, identities, and records for automated route generation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Sequence
from ..phone_shards import PhoneFfnSessionIdentity
from ..plan_contracts.co_helpers import co_helper_declaration
from ..runtime_capabilities import (
    ROUTE_MATURITY_STATES,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
)
from ..runtime_plan import AutomatedCandidateSet, RuntimePhoneShard
from ..types import canonical_dataclass_fields


class RouteGenerationError(ValueError):
    pass


class DesktopControlUnavailableError(RouteGenerationError):
    pass


# A static co-helper that left the fleet (elastic phones): the route keeps its device sets and the
# adaptive controller runs the ones without it until it rejoins.
CO_HELPER_UNAVAILABLE = "CO_HELPER_UNAVAILABLE"
CO_HELPER_MEMBERSHIP_STATES = frozenset({"ABSENT", "QUARANTINED"})

PHONE_RESIDENCY_EVIDENCE_NEUTRAL_REJECTIONS = frozenset({
    "BATTERY_LIMIT",
    CO_HELPER_UNAVAILABLE,
    "THERMAL_LIMIT",
})


def unavailable_co_helpers(coordinator: object, snapshot: object) -> frozenset[str]:
    """Declared static co-helpers of a composite that the rig reports lost or absent.

    Only an elastic-phones rig publishes a ``membership`` telemetry row, so a static rig sees none."""
    if not isinstance(coordinator, RuntimeCompositeExecutorCapability):
        return frozenset()
    declaration = co_helper_declaration(coordinator.adapter_parameters)
    if declaration is None:
        return frozenset()
    return frozenset(
        device_id for device_id in declaration.device_ids
        if (row := snapshot.telemetry_observations.get(device_id)) is not None
        and row.get("membership") in CO_HELPER_MEMBERSHIP_STATES
    )


_DORMANT_PHONE_FFN_RUNTIME_PARAMETER = "dormant_phone_ffn_runtime_v1"


def _static_executor_identity(
    executor: RuntimeExecutorCapability,
    *,
    include_residency_ownership: bool = True,
    exclude_phone_power_accounting: bool = False,
) -> Mapping[str, object]:
    """Keep live phone diagnostics out of reusable capability evidence."""
    result = {
        item.name: getattr(executor, item.name)
        for item in canonical_dataclass_fields(executor)
        if (
            include_residency_ownership
            or item.name != "exclusive_residency_resource_id"
        )
        and not (
            exclude_phone_power_accounting
            and item.name == "minimum_battery_ppm"
        )
    }
    result["phone_sessions"] = tuple(
        {
            item.name: getattr(session, item.name)
            for item in fields(session)
            if item.name != "unavailable_reason"
        }
        for session in executor.phone_sessions
    )
    return MappingProxyType(result)


def _static_coordinator_identity(
    coordinator: (
        RuntimeExecutorCapability | RuntimeCompositeExecutorCapability
    ),
    *,
    include_residency_ownership: bool = True,
    exclude_phone_power_accounting: bool = False,
) -> Mapping[str, object]:
    if isinstance(coordinator, RuntimeExecutorCapability):
        return _static_executor_identity(
            coordinator,
            include_residency_ownership=include_residency_ownership,
            exclude_phone_power_accounting=(
                exclude_phone_power_accounting
            ),
        )
    return MappingProxyType({
        item.name: getattr(coordinator, item.name)
        for item in fields(coordinator)
        if (
            include_residency_ownership
            or item.name != "replacement_group_by_device"
        )
    })


_MATURITY_RANK = {
    "QUARANTINED": 0,
    "PRIOR_ONLY": 1,
    "CALIBRATION_PENDING": 2,
    "SHADOW": 3,
    "QUALIFIED": 4,
}


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise RouteGenerationError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _minimum_maturity(values: Sequence[str]) -> str:
    rows = tuple(values)
    if not rows or any(value not in ROUTE_MATURITY_STATES for value in rows):
        raise RouteGenerationError("route maturity input is invalid")
    return min(rows, key=lambda value: (_MATURITY_RANK[value], value))


@dataclass(frozen=True)
class _FfnResidentEnvelope:
    operator_ids: tuple[str, ...]
    layer_mask: int
    columns: int
    weight_bytes: int
    column_quantum: int
    partition_count: int
    geometry_sha256: str
    shards: tuple[RuntimePhoneShard, ...] = ()
    residency_shards: tuple[RuntimePhoneShard, ...] = ()
    changed_session_ids: tuple[str, ...] = ()
    replacement_source_identities: tuple[
        PhoneFfnSessionIdentity, ...
    ] = ()
    replacement_source_resident_bytes_by_session: Mapping[str, int] = (
        MappingProxyType({})
    )
    packing_value: int = 0
    packing_value_kind: str = "rough_compute_ops"
    unavailable_session_ids: tuple[str, ...] = ()

    @property
    def session_count(self) -> int:
        return max(1, len(self.shards))

    @property
    def transition_shards(self) -> tuple[RuntimePhoneShard, ...]:
        return self.residency_shards or self.shards


@dataclass(frozen=True)
class _PhoneResidencyRouteEvidence:
    artifact_sha256: str
    source_route_id: str
    paired_desktop_route_id: str
    assisted_operator_ids: tuple[str, ...]
    benefit_by_operator_per_decode_token_uj: Mapping[str, int]
    route_benefit_uj: int
    decode_tokens: int
    normalized_benefit_uj: int
    normalization_remainder_uj: int
    benefit_value_kind: str
    energy_evidence: str
    transition_energy_upper_uj: int
    transition_energy_upper_uj_by_session: Mapping[str, int]
    transition_energy_aggregation: str
    source_component_capability_sha256: str
    source_desktop_placement_sha256: str
    source_executor_id: str
    source_endpoint: str
    source_operator_plan_protocol: str
    source_layer_mask: int
    source_maximum_columns: int
    source_session_ids: tuple[str, ...]
    source_batch_plan: str
    source_maximum_batch_size: int
    source_queue_depth: int


@dataclass(frozen=True)
class RuntimeRouteTemplateSet:
    """Immutable structural and audit input for one published placement."""

    artifact_sha256: str
    input_token_bucket: int
    output_token_bucket: int
    quality_requirement: str
    selected_route_id: str
    selected_route_key: str
    selected_component_identity_sha256: str
    route_template_identity_sha256: str
    candidate_set: AutomatedCandidateSet
    audit_sha256: str
    live_state_sha256: str | None


def _placement_rejection_reasons(message: str) -> tuple[str, ...]:
    """Map a placement failure to its first actionable physical causes."""
    if "no placement meets the final transfer" in message:
        return ("TRANSFER_PLAN_ABSENT",)
    marker = ": candidate_gate="
    if marker not in message:
        return ("PLACEMENT_INFEASIBLE",)
    raw = message.split(marker, 1)[1]
    values = {}
    try:
        for item in raw.split(", "):
            name, count = item.split("=", 1)
            values[name] = int(count)
    except (TypeError, ValueError):
        return ("PLACEMENT_INFEASIBLE",)
    reasons = []
    if values.get("candidate_gate", 0):
        reasons.append("KERNEL_SUPPORT_ABSENT")
    if values.get("memory", 0):
        reasons.append("MEMORY_CAPACITY")
    if values.get("transfer", 0):
        reasons.append("TRANSFER_PLAN_ABSENT")
    if values.get("deadline", 0):
        reasons.append("SLO_UPPER_BOUND")
    return tuple(reasons or ("PLACEMENT_INFEASIBLE",))


class _Pattern:
    def __init__(
        self,
        route_key: str,
        route_family: str,
        assignments: Mapping[str, tuple[str, str | None, int]],
        assisted_operator_kind: str | None,
        split_axis: str,
        split_fraction_ppm: int,
        overlap_kind: str,
        coordinator_device_id: str,
        coordinator_executor_id: str | None = None,
        baseline_executor_id: str | None = None,
        desktop_assignments: Mapping[str, str] | None = None,
        assistance_phase: str = "all",
        resident_envelope: bool = False,
        phone_session_count: int = 0,
        phone_resident_envelope: _FfnResidentEnvelope | None = None,
        resident_owner_device_ids: tuple[str, ...] = (),
    ) -> None:
        if assistance_phase not in {"all", "decode"}:
            raise RouteGenerationError(
                "generated assistance phase is unsupported"
            )
        self.route_key = route_key
        self.route_family = route_family
        self.assignments = MappingProxyType(dict(assignments))
        self.assisted_operator_kind = assisted_operator_kind
        self.split_axis = split_axis
        self.split_fraction_ppm = split_fraction_ppm
        self.overlap_kind = overlap_kind
        self.coordinator_device_id = coordinator_device_id
        self.coordinator_executor_id = coordinator_executor_id
        self.baseline_executor_id = baseline_executor_id
        self.assistance_phase = assistance_phase
        self.resident_envelope = resident_envelope
        self.phone_session_count = phone_session_count
        self.phone_resident_envelope = phone_resident_envelope
        if (
            phone_resident_envelope is not None
            and phone_resident_envelope.session_count != phone_session_count
        ):
            raise RouteGenerationError(
                "phone resident envelope session count differs"
            )
        self.desktop_assignments = MappingProxyType(dict(
            {
                operator_id: primary
                for operator_id, (primary, _helper, _fraction)
                in assignments.items()
            }
            if desktop_assignments is None else desktop_assignments
        ))
        devices = {
            device_id
            for primary, helper, _ in self.assignments.values()
            for device_id in (primary, helper)
            if device_id is not None
        }
        if self.assistance_phase == "decode":
            devices.update(self.desktop_assignments.values())
        devices.update(resident_owner_device_ids)
        self._device_ids = tuple(sorted(devices))

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self._device_ids
