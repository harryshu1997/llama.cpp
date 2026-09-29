"""Execution-plan contracts grouped by responsibility: execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from ..runtime_capabilities import SPLIT_AXES
from ..runtime_cost import RuntimeExecutorBinding, RuntimeMemoryDemand
from .common import (
    HELPER_OPPORTUNITY_EVIDENCE_STATES,
    RUNTIME_EXECUTION_PLAN_SCHEMA,
    RUNTIME_EXECUTION_RECEIPT_STATUSES,
    RuntimePlanError,
    _estimation_metadata,
    _integer,
    _sha256,
    _text,
)
from .phone import PhoneSessionReplacementAuthorization
from .operators import (
    RuntimeExecutionContract,
    RuntimeOperatorAssignment,
    _OPERATOR_JSON_PLACEHOLDER,
    _canonical_operator_json,
)
from .transitions import RuntimeTransitionPlan


@dataclass(frozen=True)
class RuntimeExecutionReceipt:
    """Proof that the selected endpoint executed the hash-bound plan."""

    ticket_id: str
    request_id: str
    artifact_sha256: str
    operator_plan_sha256: str
    executor_id: str
    endpoint: str
    operator_plan_protocol: str
    participant_executor_ids: tuple[str, ...]
    started_us: int
    finished_us: int
    output_sha256: str
    status: str
    energy_boundary_id: str | None = None
    fleet_energy_uj_by_domain: Mapping[str, int] = field(
        default_factory=dict
    )
    transfer_energy_uj_by_link: Mapping[str, int] = field(
        default_factory=dict
    )
    measurement_evidence_ids: tuple[str, ...] = ()
    energy_attribution_kind: str | None = None
    energy_scope: str = "route_total"
    energy_estimation_metadata: Mapping[
        str, int | str | bool
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "ticket_id",
            "request_id",
            "artifact_sha256",
            "operator_plan_sha256",
            "executor_id",
            "endpoint",
            "operator_plan_protocol",
            "output_sha256",
        ):
            _text(f"execution receipt {name}", getattr(self, name))
        for name in (
            "artifact_sha256",
            "operator_plan_sha256",
            "output_sha256",
        ):
            value = getattr(self, name)
            if (
                not value.startswith("sha256:")
                or len(value) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in value[7:]
                )
            ):
                raise RuntimePlanError(
                    f"execution receipt {name} must be SHA-256"
                )
        participants = tuple(
            _text("execution receipt participant", value)
            for value in self.participant_executor_ids
        )
        if not participants or len(participants) != len(set(participants)):
            raise RuntimePlanError(
                "execution receipt participants are invalid"
            )
        _integer("execution receipt started", self.started_us)
        _integer("execution receipt finished", self.finished_us)
        if self.finished_us < self.started_us:
            raise RuntimePlanError(
                "execution receipt finishes before it starts"
            )
        if self.status not in RUNTIME_EXECUTION_RECEIPT_STATUSES:
            raise RuntimePlanError("execution receipt status is invalid")
        fleet_energy = {
            _text("execution receipt energy domain", key): _integer(
                "execution receipt domain energy", value
            )
            for key, value in self.fleet_energy_uj_by_domain.items()
        }
        transfer_energy = {
            _text("execution receipt transfer link", key): _integer(
                "execution receipt transfer energy", value
            )
            for key, value in self.transfer_energy_uj_by_link.items()
        }
        evidence = tuple(
            _text("execution receipt measurement evidence", value)
            for value in self.measurement_evidence_ids
        )
        if self.energy_boundary_id is None:
            if (
                fleet_energy or transfer_energy or evidence
                or self.energy_attribution_kind is not None
                or self.energy_estimation_metadata
            ):
                raise RuntimePlanError(
                    "execution receipt measurement lacks an energy boundary"
                )
        else:
            _text(
                "execution receipt energy boundary",
                self.energy_boundary_id,
            )
            if not fleet_energy or not evidence:
                raise RuntimePlanError(
                    "execution receipt energy measurement is incomplete"
                )
            if self.energy_attribution_kind is None:
                object.__setattr__(
                    self, "energy_attribution_kind", "diagnostic"
                )
            if self.energy_attribution_kind not in {
                "diagnostic", "isolated", "matched_abba", "device_domain"
            }:
                raise RuntimePlanError(
                    "execution receipt energy attribution is invalid"
                )
        object.__setattr__(
            self,
            "participant_executor_ids",
            tuple(sorted(participants)),
        )
        object.__setattr__(
            self,
            "fleet_energy_uj_by_domain",
            MappingProxyType(dict(sorted(fleet_energy.items()))),
        )
        object.__setattr__(
            self,
            "transfer_energy_uj_by_link",
            MappingProxyType(dict(sorted(transfer_energy.items()))),
        )
        object.__setattr__(
            self,
            "measurement_evidence_ids",
            tuple(sorted(evidence)),
        )
        object.__setattr__(
            self,
            "energy_estimation_metadata",
            _estimation_metadata(self.energy_estimation_metadata),
        )
        if self.energy_scope not in {"route_total", "warm_execution"}:
            raise RuntimePlanError(
                "execution receipt energy scope is invalid"
            )

    @property
    def actual_latency_us(self) -> int:
        return self.finished_us - self.started_us

    @property
    def whole_fleet_energy_uj(self) -> int | None:
        if self.energy_boundary_id is None:
            return None
        return sum(self.fleet_energy_uj_by_domain.values())

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "endpoint": self.endpoint,
            "executor_id": self.executor_id,
            "energy_boundary_id": self.energy_boundary_id,
            "energy_attribution_kind": self.energy_attribution_kind,
            "energy_estimation_metadata": dict(
                self.energy_estimation_metadata
            ),
            "fleet_energy_uj_by_domain": dict(
                self.fleet_energy_uj_by_domain
            ),
            "finished_us": self.finished_us,
            "measurement_evidence_ids": list(
                self.measurement_evidence_ids
            ),
            "operator_plan_protocol": self.operator_plan_protocol,
            "operator_plan_sha256": self.operator_plan_sha256,
            "output_sha256": self.output_sha256,
            "participant_executor_ids": list(
                self.participant_executor_ids
            ),
            "request_id": self.request_id,
            "started_us": self.started_us,
            "status": self.status,
            "ticket_id": self.ticket_id,
            "transfer_energy_uj_by_link": dict(
                self.transfer_energy_uj_by_link
            ),
            "whole_fleet_energy_uj": self.whole_fleet_energy_uj,
        }
        if self.energy_scope != "route_total":
            result["energy_scope"] = self.energy_scope
        return result


@dataclass(frozen=True)
class RuntimeExecutionPlan:
    route_id: str
    route_family: str
    device_ids: tuple[str, ...]
    assisted_operator_kind: str | None
    split_axis: str
    split_fraction_ppm: int
    residency_variant: str
    overlap_kind: str
    operators: tuple[RuntimeOperatorAssignment, ...]
    transitions: tuple[RuntimeTransitionPlan, ...]
    resource_ids: tuple[str, ...]
    memory_demands: tuple[RuntimeMemoryDemand, ...]
    execution_contract: RuntimeExecutionContract
    route_profile_id: str | None = None
    resource_slots: Mapping[str, int] = field(default_factory=dict)
    adapter_parameters: Mapping[str, int | str] = field(default_factory=dict)
    baseline_executor_id: str | None = None
    desktop_placement_sha256: str | None = None
    helper_envelope: RuntimeHelperExecutionEnvelope | None = None
    _plan_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text("execution plan route id", self.route_id)
        _text("execution plan route family", self.route_family)
        devices = tuple(self.device_ids)
        resources = tuple(self.resource_ids)
        if not devices or len(devices) != len(set(devices)):
            raise RuntimePlanError("execution plan devices are invalid")
        if not resources or len(resources) != len(set(resources)):
            raise RuntimePlanError("execution plan resources are invalid")
        for value in devices + resources:
            _text("execution plan reference", value)
        slots = {
            _text("execution plan slot resource", resource_id): _integer(
                "execution plan resource slots", value, 1
            )
            for resource_id, value in self.resource_slots.items()
        }
        if set(slots) - set(resources):
            raise RuntimePlanError(
                "execution plan slots reference an unknown resource"
            )
        slots = {
            resource_id: slots.get(resource_id, 1)
            for resource_id in resources
        }
        if self.assisted_operator_kind is not None:
            _text(
                "execution plan assisted operator", self.assisted_operator_kind
            )
        if self.split_axis != "none" and self.split_axis not in SPLIT_AXES:
            raise RuntimePlanError("execution plan split axis is invalid")
        _integer("execution plan split fraction", self.split_fraction_ppm)
        if self.split_axis == "none" and self.split_fraction_ppm:
            raise RuntimePlanError("unsplit execution plan carries a fraction")
        if self.split_axis != "none" and not 0 < self.split_fraction_ppm < 1_000_000:
            raise RuntimePlanError("execution plan split fraction is invalid")
        _text("execution plan residency variant", self.residency_variant)
        _text("execution plan overlap kind", self.overlap_kind)
        if self.route_profile_id is not None:
            _text("execution plan route profile", self.route_profile_id)
        if self.baseline_executor_id is not None:
            _text(
                "execution plan baseline executor",
                self.baseline_executor_id,
            )
        if self.desktop_placement_sha256 is not None:
            value = _text(
                "execution plan desktop placement hash",
                self.desktop_placement_sha256,
            )
            if (
                not value.startswith("sha256:")
                or len(value) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in value[7:]
                )
            ):
                raise RuntimePlanError(
                    "execution plan desktop placement hash must be SHA-256"
                )
        operators = tuple(self.operators)
        transitions = tuple(self.transitions)
        memory = tuple(self.memory_demands)
        adapter_parameters: dict[str, int | str] = {}
        for raw_name, raw_value in self.adapter_parameters.items():
            name = _text("execution plan adapter parameter", raw_name)
            if type(raw_value) is int:
                if raw_value < 0:
                    raise RuntimePlanError(
                        "execution plan adapter integer parameter is negative"
                    )
                adapter_parameters[name] = raw_value
            elif type(raw_value) is str:
                adapter_parameters[name] = _text(
                    "execution plan adapter parameter value", raw_value
                )
            else:
                raise RuntimePlanError(
                    "execution plan adapter parameter value is invalid"
                )
        if not operators or any(
            not isinstance(row, RuntimeOperatorAssignment) for row in operators
        ):
            raise RuntimePlanError("execution plan operators are invalid")
        if len({row.operator_id for row in operators}) != len(operators):
            raise RuntimePlanError("execution plan operator ids are duplicated")
        if any(not isinstance(row, RuntimeTransitionPlan) for row in transitions):
            raise RuntimePlanError("execution plan transitions are invalid")
        if any(not isinstance(row, RuntimeMemoryDemand) for row in memory):
            raise RuntimePlanError("execution plan memory demands are invalid")
        if not isinstance(self.execution_contract, RuntimeExecutionContract):
            raise RuntimePlanError(
                "execution plan data-plane contract is invalid"
            )
        if self.helper_envelope is not None:
            if not isinstance(
                self.helper_envelope, RuntimeHelperExecutionEnvelope
            ):
                raise RuntimePlanError(
                    "execution plan helper envelope is invalid"
                )
            if self.execution_contract.execution_mode != "desktop":
                raise RuntimePlanError(
                    "helper envelope requires a desktop base plan"
                )
            if (
                self.desktop_placement_sha256 is None
                or self.helper_envelope.desktop_parent_route_id
                    != self.route_id
                or self.helper_envelope.desktop_placement_sha256
                    != self.desktop_placement_sha256
            ):
                raise RuntimePlanError(
                    "helper envelope differs from the desktop base"
                )
        if len({row.demand_id for row in memory}) != len(memory):
            raise RuntimePlanError("execution plan memory demands are duplicated")
        object.__setattr__(self, "device_ids", tuple(sorted(devices)))
        object.__setattr__(self, "resource_ids", tuple(sorted(resources)))
        object.__setattr__(
            self,
            "resource_slots",
            MappingProxyType(dict(sorted(slots.items()))),
        )
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(
            self, "memory_demands", tuple(sorted(memory, key=lambda row: row.demand_id))
        )
        object.__setattr__(
            self,
            "adapter_parameters",
            MappingProxyType(dict(sorted(adapter_parameters.items()))),
        )
        object.__setattr__(
            self,
            "_plan_sha256",
            "sha256:" + hashlib.sha256(
                self._canonical_bytes_without_hash()
            ).hexdigest(),
        )

    @property
    def plan_sha256(self) -> str:
        return self._plan_sha256

    def _json_without_hash(
        self, *, operator_placeholder: bool = False
    ) -> dict[str, object]:
        result = {
            "assisted_operator_kind": self.assisted_operator_kind,
            "device_ids": list(self.device_ids),
            "execution_contract": self.execution_contract.to_json(),
            "memory_demands": [row.to_json() for row in self.memory_demands],
            "operators": (
                _OPERATOR_JSON_PLACEHOLDER
                if operator_placeholder
                else [row.to_json() for row in self.operators]
            ),
            "overlap_kind": self.overlap_kind,
            "residency_variant": self.residency_variant,
            "resource_ids": list(self.resource_ids),
            "route_family": self.route_family,
            "route_id": self.route_id,
            "schema": RUNTIME_EXECUTION_PLAN_SCHEMA,
            "split_axis": self.split_axis,
            "split_fraction_ppm": self.split_fraction_ppm,
            "transitions": [row.to_json() for row in self.transitions],
        }
        if self.route_profile_id is not None:
            result["route_profile_id"] = self.route_profile_id
        if self.baseline_executor_id is not None:
            result["baseline_executor_id"] = self.baseline_executor_id
        if self.desktop_placement_sha256 is not None:
            result["desktop_placement_sha256"] = (
                self.desktop_placement_sha256
            )
        if self.helper_envelope is not None:
            result["helper_envelope"] = self.helper_envelope.to_json()
        if self.adapter_parameters:
            result["adapter_parameters"] = dict(self.adapter_parameters)
        if any(value != 1 for value in self.resource_slots.values()):
            result["resource_slots"] = dict(self.resource_slots)
        return result

    def _canonical_bytes_without_hash(self) -> bytes:
        encoded = json.dumps(
            self._json_without_hash(operator_placeholder=True),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        placeholder = json.dumps(
            _OPERATOR_JSON_PLACEHOLDER,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        if encoded.count(placeholder) != 1:
            raise RuntimePlanError(
                "execution plan operator placeholder is ambiguous"
            )
        return encoded.replace(
            placeholder,
            _canonical_operator_json(self.operators),
            1,
        )

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["plan_sha256"] = self.plan_sha256
        return result


@dataclass(frozen=True)
class HelperOpportunity:
    """Compact permission to attach one helper after it becomes ready."""

    route_family_identity: str
    desktop_parent_route_id: str
    desktop_parent_placement_sha256: str
    helper_operator_plan: RuntimeExecutionPlan
    helper_binding: RuntimeExecutorBinding
    phone_layout_generation: int | None
    phone_layout_geometry_sha256: str
    required_resource_ids: tuple[str, ...]
    evidence_state: str
    maximum_allowed_fraction_ppm: int

    def __post_init__(self) -> None:
        family = _text(
            "helper opportunity route family",
            self.route_family_identity,
        )
        _text(
            "helper opportunity desktop parent",
            self.desktop_parent_route_id,
        )
        _sha256(
            "helper opportunity desktop placement",
            self.desktop_parent_placement_sha256,
        )
        plan = self.helper_operator_plan
        binding = self.helper_binding
        if (
            not isinstance(plan, RuntimeExecutionPlan)
            or not isinstance(binding, RuntimeExecutorBinding)
            or plan.helper_envelope is not None
            or plan.route_family != family
            or plan.assisted_operator_kind != "ffn"
            or plan.execution_contract.execution_mode != "adaptive-split"
            or plan.execution_contract.operator_kind != "ffn"
            or not plan.execution_contract.phone_shards
            or plan.desktop_placement_sha256
                != self.desktop_parent_placement_sha256
            or binding.route_id != plan.route_id
            or binding.operator_plan_sha256 != plan.plan_sha256
            or binding.endpoint is None
            or binding.operator_plan_protocol is None
        ):
            raise RuntimePlanError(
                "helper opportunity operator plan is invalid"
            )
        if self.phone_layout_generation is not None:
            _integer(
                "helper opportunity phone layout generation",
                self.phone_layout_generation,
                1,
            )
        geometry = _sha256(
            "helper opportunity phone layout geometry",
            self.phone_layout_geometry_sha256,
        )
        if plan.adapter_parameters.get(
            "phone_shard_set_geometry_sha256"
        ) != geometry:
            raise RuntimePlanError(
                "helper opportunity phone layout geometry differs"
            )
        resources = tuple(sorted(
            _text("helper opportunity resource", value)
            for value in self.required_resource_ids
        ))
        if (
            not resources
            or len(resources) != len(set(resources))
            or not set(resources).issubset(plan.resource_ids)
        ):
            raise RuntimePlanError(
                "helper opportunity resources are invalid"
            )
        if self.evidence_state not in HELPER_OPPORTUNITY_EVIDENCE_STATES:
            raise RuntimePlanError(
                "helper opportunity evidence state is invalid"
            )
        maximum = _integer(
            "helper opportunity maximum fraction",
            self.maximum_allowed_fraction_ppm,
            1,
        )
        allowed = plan.execution_contract.allowed_adaptive_fractions_ppm
        if maximum > 1_000_000 or maximum not in allowed:
            raise RuntimePlanError(
                "helper opportunity maximum fraction is invalid"
            )
        object.__setattr__(self, "required_resource_ids", resources)

    @property
    def route_id(self) -> str:
        return self.helper_operator_plan.route_id

    @property
    def operator_plan_sha256(self) -> str:
        return self.helper_operator_plan.plan_sha256

    def to_json(self) -> dict[str, object]:
        return {
            "desktop_parent_placement_sha256": (
                self.desktop_parent_placement_sha256
            ),
            "desktop_parent_route_id": self.desktop_parent_route_id,
            "evidence_state": self.evidence_state,
            "helper_binding": self.helper_binding.to_json(),
            "helper_operator_plan": self.helper_operator_plan.to_json(),
            "maximum_allowed_fraction_ppm": (
                self.maximum_allowed_fraction_ppm
            ),
            "operator_plan_sha256": self.operator_plan_sha256,
            "phone_layout_generation": self.phone_layout_generation,
            "phone_layout_geometry_sha256": (
                self.phone_layout_geometry_sha256
            ),
            "required_resource_ids": list(self.required_resource_ids),
            "route_family_identity": self.route_family_identity,
            "route_id": self.route_id,
            "schema": "research-scheduler-helper-opportunity-v1",
        }


def helper_preparation_changed_session_ids(
    layout_changed_session_ids: Sequence[str],
    contract_session_ids: Sequence[str],
) -> tuple[str, ...]:
    """Scope layout-wide changed sessions to one helper's own shards.

    A layout change lists every session it replaces. A helper envelope may
    only name the changed sessions that its execution contract actually
    covers: a retained helper whose shards are untouched gets an empty
    tuple, and a new helper gets exactly its own replaced sessions.
    """

    changed = tuple(sorted(set(
        _text("helper changed session", value)
        for value in layout_changed_session_ids
    )))
    contract = {
        _text("helper contract session", value)
        for value in contract_session_ids
    }
    return tuple(value for value in changed if value in contract)


@dataclass(frozen=True)
class RuntimeHelperExecutionEnvelope:
    """A phone helper that may attach without changing desktop placement."""

    artifact_sha256: str
    desktop_parent_route_id: str
    desktop_placement_sha256: str
    helper_plan: RuntimeExecutionPlan
    helper_binding: RuntimeExecutorBinding
    phone_layout_generation: int
    phone_layout_geometry_sha256: str
    activation_dtype: str
    preparation_changed_session_ids: tuple[str, ...] = ()
    replacement_authorization: PhoneSessionReplacementAuthorization | None = (
        None
    )

    def __post_init__(self) -> None:
        _sha256("helper envelope artifact", self.artifact_sha256)
        _text(
            "helper envelope desktop parent",
            self.desktop_parent_route_id,
        )
        _sha256(
            "helper envelope desktop placement",
            self.desktop_placement_sha256,
        )
        if (
            not isinstance(self.helper_plan, RuntimeExecutionPlan)
            or self.helper_plan.helper_envelope is not None
            or not isinstance(
                self.helper_binding, RuntimeExecutorBinding
            )
        ):
            raise RuntimePlanError("helper envelope plan is invalid")
        plan = self.helper_plan
        binding = self.helper_binding
        contract = plan.execution_contract
        if (
            contract.execution_mode != "adaptive-split"
            or contract.operator_kind != "ffn"
            or not contract.phone_shards
            or plan.assisted_operator_kind != "ffn"
            or plan.desktop_placement_sha256
                != self.desktop_placement_sha256
            or binding.route_id != plan.route_id
            or binding.operator_plan_sha256 != plan.plan_sha256
            or binding.artifact_sha256 != self.artifact_sha256
            or binding.endpoint is None
            or binding.operator_plan_protocol is None
        ):
            raise RuntimePlanError(
                "helper envelope is not an executable adaptive FFN plan"
            )
        geometry = _sha256(
            "helper envelope phone layout geometry",
            self.phone_layout_geometry_sha256,
        )
        _integer(
            "helper envelope phone layout generation",
            self.phone_layout_generation,
            1,
        )
        if plan.adapter_parameters.get(
            "phone_shard_set_geometry_sha256"
        ) != geometry:
            raise RuntimePlanError(
                "helper envelope phone layout geometry differs"
            )
        dtype = _text(
            "helper envelope activation dtype", self.activation_dtype
        )
        wire_bytes = plan.adapter_parameters.get(
            "ffn_wire_element_bytes"
        )
        if dtype != "f16" or wire_bytes not in {None, 2}:
            raise RuntimePlanError(
                "helper envelope activation dtype is unsupported"
            )
        changed_sessions = tuple(sorted(
            _text("helper envelope changed session", value)
            for value in self.preparation_changed_session_ids
        ))
        if (
            len(changed_sessions) != len(set(changed_sessions))
            or set(changed_sessions) - {
                row.session_id for row in contract.phone_shards
            }
        ):
            raise RuntimePlanError(
                "helper envelope changed sessions are invalid"
            )
        authorization = self.replacement_authorization
        if authorization is not None and (
            not isinstance(
                authorization, PhoneSessionReplacementAuthorization
            )
            or changed_sessions != (authorization.selected_session_id,)
        ):
            raise RuntimePlanError(
                "helper envelope replacement authorization differs"
            )
        object.__setattr__(
            self,
            "preparation_changed_session_ids",
            changed_sessions,
        )

    @property
    def operator_plan_sha256(self) -> str:
        return self.helper_plan.plan_sha256

    @property
    def route_id(self) -> str:
        return self.helper_plan.route_id

    @property
    def resident_layer_mask(self) -> int:
        return int(self.helper_plan.adapter_parameters[
            "ffn_resident_layer_mask"
        ])

    @property
    def resident_columns(self) -> int:
        return int(self.helper_plan.adapter_parameters[
            "ffn_resident_columns"
        ])

    @property
    def preparation_transitions(self) -> tuple[RuntimeTransitionPlan, ...]:
        """Return the phone-only transition nested under a desktop request."""

        phone_device_id = self.helper_plan.execution_contract.phone_device_id
        participants = tuple(
            row for row in self.helper_binding.participants
            if row.device_id == phone_device_id
        )
        if phone_device_id is None or len(participants) != 1:
            raise RuntimePlanError(
                "helper envelope phone participant is invalid"
            )
        phone_resources = set(participants[0].resource_ids)
        result = []
        for transition in self.helper_plan.transitions:
            if (
                phone_device_id not in transition.prepares_device_ids
                and not transition.phone_shards
            ):
                continue
            resource_ids = tuple(
                resource_id for resource_id in transition.resource_ids
                if resource_id in phone_resources
            )
            if not resource_ids:
                raise RuntimePlanError(
                    "helper preparation has no phone resources"
                )
            evictions = tuple(
                row for row in transition.evictions
                if row.device_id == phone_device_id
            )
            if (
                self.preparation_changed_session_ids
                and self.replacement_authorization is not None
            ):
                retained_bytes = {}
                for shard in transition.phone_shards:
                    if shard.session_id not in self.preparation_changed_session_ids:
                        retained_bytes[shard.artifact_sha256] = (
                            retained_bytes.get(shard.artifact_sha256, 0)
                            + shard.resident_bytes
                        )
                scoped_evictions = []
                for eviction in evictions:
                    if eviction.session_id in self.preparation_changed_session_ids:
                        scoped_evictions.append(eviction)
                    elif (
                        eviction.session_id is not None
                        or retained_bytes.get(eviction.artifact_sha256)
                            != eviction.resident_bytes
                    ):
                        raise RuntimePlanError(
                            "retained phone eviction differs from the session map"
                        )
                # READY templates may retain historical planning metadata;
                # only an authorized load consumes session-scoped evictions.
                evictions = tuple(scoped_evictions)
            result.append(replace(
                transition,
                device_id=phone_device_id,
                resource_ids=resource_ids,
                resource_slots={
                    resource_id: transition.resource_slots[resource_id]
                    for resource_id in resource_ids
                },
                evictions=evictions,
                prepares_device_ids=(phone_device_id,),
                changed_phone_session_ids=(
                    self.preparation_changed_session_ids
                ),
            ))
        return tuple(result)

    def to_json(self) -> dict[str, object]:
        result = {
            "activation_dtype": self.activation_dtype,
            "artifact_sha256": self.artifact_sha256,
            "desktop_parent_route_id": self.desktop_parent_route_id,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "helper_binding": self.helper_binding.to_json(),
            "helper_plan": self.helper_plan.to_json(),
            "operator_plan_sha256": self.operator_plan_sha256,
            "preparation_transitions": [
                row.to_json() for row in self.preparation_transitions
            ],
            "phone_layout_generation": self.phone_layout_generation,
            "phone_layout_geometry_sha256": (
                self.phone_layout_geometry_sha256
            ),
            "route_id": self.route_id,
            "schema": "research-scheduler-runtime-helper-envelope-v1",
        }
        if self.preparation_changed_session_ids:
            result["preparation_changed_session_ids"] = list(
                self.preparation_changed_session_ids
            )
        if self.replacement_authorization is not None:
            result["replacement_authorization"] = (
                self.replacement_authorization.to_json()
            )
        return result

    def preparation_ticket_id(self, request_ticket_id: str) -> str:
        _text("helper preparation request ticket", request_ticket_id)
        return (
            "phone-helper-layout-"
            + str(self.phone_layout_generation)
            + "-"
            + self.phone_layout_geometry_sha256[7:23]
        )
