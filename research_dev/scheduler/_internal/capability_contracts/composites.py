"""Device, executor, catalog and snapshot capability contracts: composites."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from ..runtime_system_cost import ROUTE_MATURITY_STATES
from .common import (
    COMPOSITE_ROUTE_FAMILIES,
    RESIDENCY_STATES,
    RuntimeCapabilityError,
    SPLIT_AXES,
    _fractions,
    _list,
    _object,
    _text,
    _texts,
)
from .desktop import RuntimeCompositeOperatorPlacement


def _composite_basic_values(
    capability: RuntimeCompositeExecutorCapability,
) -> tuple[
    tuple[str, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[RuntimeCompositeOperatorPlacement, ...],
]:
    for name in (
        "executor_id",
        "endpoint",
        "backend",
        "coordinator_device_id",
        "operator_plan_protocol",
    ):
        _text("composite executor " + name, getattr(capability, name))
    participants = _texts(
        "composite participant device", capability.participant_device_ids
    )
    if (
        not participants
        or len(participants) != len(set(participants))
        or participants[0] != capability.coordinator_device_id
    ):
        raise RuntimeCapabilityError(
            "composite participants must start with the coordinator"
        )
    if capability.route_family not in COMPOSITE_ROUTE_FAMILIES:
        raise RuntimeCapabilityError(
            "composite executor route family is unsupported"
        )
    if len(participants) == 1 and (
        capability.route_family != "layer_placement"
        or capability.assisted_operator_kind is not None
        or capability.helper_device_id is not None
    ):
        raise RuntimeCapabilityError(
            "single-device coordinator must be a local placement"
        )
    if capability.assisted_operator_kind is not None:
        _text(
            "composite assisted operator", capability.assisted_operator_kind
        )
    if capability.split_axis != "none" and capability.split_axis not in SPLIT_AXES:
        raise RuntimeCapabilityError("composite split axis is unsupported")
    split_fractions = _fractions(
        "composite split fractions", capability.split_fractions_ppm
    )
    layer_fractions = _fractions(
        "composite layer fractions", capability.layer_fractions_ppm
    )
    placements = tuple(capability.operator_placements)
    if capability.baseline_executor_id is not None:
        _text(
            "composite baseline executor id", capability.baseline_executor_id
        )
    if capability.helper_device_id is not None:
        helper_device_id = _text(
            "composite helper device id", capability.helper_device_id
        )
        if (
            helper_device_id not in participants
            or helper_device_id == capability.coordinator_device_id
        ):
            raise RuntimeCapabilityError(
                "composite helper device is invalid"
            )
    if (
        any(not isinstance(row, RuntimeCompositeOperatorPlacement)
            for row in placements)
        or len({row.operator_id for row in placements}) != len(placements)
    ):
        raise RuntimeCapabilityError(
            "composite operator placements are invalid"
        )
    return participants, split_fractions, layer_fractions, placements


def _validate_composite_route_contract(
    capability: RuntimeCompositeExecutorCapability,
    participants: tuple[str, ...],
    split_fractions: tuple[int, ...],
    layer_fractions: tuple[int, ...],
    placements: tuple[RuntimeCompositeOperatorPlacement, ...],
) -> None:
    if placements:
        used_devices = {
            device_id
            for row in placements
            for device_id in (row.primary_device_id, row.helper_device_id)
            if device_id is not None
        }
        parameters = capability.adapter_parameters
        if parameters.get("remote_resident_ffn_v1") is not None:
            phone = parameters.get("phone_device_id")
            if type(phone) is not str or not phone:
                raise RuntimeCapabilityError("remote-resident coordinator lacks its phone device")
            used_devices.add(phone)
        if used_devices != set(participants):
            raise RuntimeCapabilityError(
                "composite operator placements differ from participants"
            )
        split_rows = tuple(
            row for row in placements if row.helper_device_id is not None
        )
        assisted_rows = tuple(row for row in placements if row.assisted)
        if capability.route_family == "operator_split":
            invalid = (
                capability.assisted_operator_kind is None
                or capability.split_axis == "none"
                or not assisted_rows
                or set(split_rows) != set(assisted_rows)
                or any(
                    row.split_axis != capability.split_axis
                    or row.split_fraction_ppm not in split_fractions
                    for row in split_rows
                )
            )
            message = "explicit operator split contract is incomplete"
        elif capability.route_family == "operator_offload":
            invalid = (
                capability.assisted_operator_kind is None
                or capability.split_axis != "none"
                or bool(split_fractions)
                or bool(split_rows)
                or not assisted_rows
            )
            message = "explicit operator offload contract is invalid"
        elif capability.route_family == "layer_placement":
            invalid = (
                capability.assisted_operator_kind is not None
                or capability.split_axis != "none"
                or bool(split_fractions)
                or bool(split_rows)
                or bool(assisted_rows)
            )
            message = "explicit layer placement contract is invalid"
        else:
            raise RuntimeCapabilityError(
                "explicit placements require a supported composite route"
            )
        if invalid:
            raise RuntimeCapabilityError(message)
        if layer_fractions or capability.operator_ids:
            raise RuntimeCapabilityError(
                "explicit operator placements cannot mix legacy selectors"
            )
        return
    multi_requires_identity = (
        len(participants) > 2
        and (
            capability.baseline_executor_id is None
            or capability.helper_device_id is None
        )
    )
    if capability.route_family == "operator_split":
        if (
            capability.assisted_operator_kind is None
            or capability.split_axis == "none"
            or not split_fractions
            or multi_requires_identity
        ):
            raise RuntimeCapabilityError(
                "operator split coordinator contract is incomplete"
            )
    elif capability.route_family == "operator_offload":
        if (
            capability.assisted_operator_kind is None
            or capability.split_axis != "none"
            or split_fractions
            or multi_requires_identity
        ):
            raise RuntimeCapabilityError(
                "operator offload coordinator contract is invalid"
            )
    elif (
        capability.assisted_operator_kind is not None
        or capability.split_axis != "none"
        or split_fractions
        or capability.baseline_executor_id is not None
        or capability.helper_device_id is not None
        or len(layer_fractions) != len(participants) - 1
    ):
        raise RuntimeCapabilityError("layer coordinator contract is invalid")


def _composite_metadata_values(
    capability: RuntimeCompositeExecutorCapability,
    participants: tuple[str, ...],
) -> dict[str, object]:
    residency = _texts(
        "composite residency state", capability.residency_states
    )
    if set(residency) - RESIDENCY_STATES:
        raise RuntimeCapabilityError("composite residency state is invalid")
    resources = _texts("composite resource", capability.resource_ids)
    participant_resources = {
        _text("composite resource device", device_id): _texts(
            "composite participant resource", values
        )
        for device_id, values in capability.participant_resource_ids.items()
    }
    if set(participant_resources) != set(participants) or any(
        set(values) - set(resources)
        for values in participant_resources.values()
    ):
        raise RuntimeCapabilityError(
            "composite participant resources differ from its resources"
        )
    if capability.maturity not in ROUTE_MATURITY_STATES:
        raise RuntimeCapabilityError(
            "composite executor maturity is invalid"
        )
    evidence = _texts("composite evidence id", capability.evidence_ids)
    operators = _texts(
        "composite operator id", capability.operator_ids, allow_empty=True
    )
    adapter_parameters: dict[str, int | str] = {}
    for raw_name, raw_value in capability.adapter_parameters.items():
        name = _text("composite adapter parameter", raw_name)
        if type(raw_value) is int:
            if raw_value < 0:
                raise RuntimeCapabilityError(
                    "composite adapter integer parameter is negative"
                )
            adapter_parameters[name] = raw_value
        elif type(raw_value) is str:
            adapter_parameters[name] = _text(
                "composite adapter parameter value", raw_value
            )
        else:
            raise RuntimeCapabilityError(
                "composite adapter parameter value is invalid"
            )
    if capability.artifact_sha256 is not None:
        artifact = _text(
            "composite artifact hash", capability.artifact_sha256
        )
        if (
            not artifact.startswith("sha256:")
            or len(artifact) != 71
            or any(value not in "0123456789abcdef" for value in artifact[7:])
        ):
            raise RuntimeCapabilityError(
                "composite artifact hash must be SHA-256"
            )
    replacement_groups = {
        _text("composite replacement device", device_id): _text(
            "composite replacement group", resource_id
        )
        for device_id, resource_id in capability.replacement_group_by_device.items()
    }
    if (
        set(replacement_groups) - set(participants)
        or set(replacement_groups.values()) - set(resources)
    ):
        raise RuntimeCapabilityError(
            "composite replacement ownership is invalid"
        )
    return {
        "residency": residency,
        "resources": resources,
        "participant_resources": participant_resources,
        "evidence": evidence,
        "operators": operators,
        "adapter_parameters": adapter_parameters,
        "replacement_groups": replacement_groups,
    }


def _initialize_composite_executor_capability(
    capability: RuntimeCompositeExecutorCapability,
) -> None:
    participants, split_fractions, layer_fractions, placements = (
        _composite_basic_values(capability)
    )
    _validate_composite_route_contract(
        capability,
        participants,
        split_fractions,
        layer_fractions,
        placements,
    )
    values = _composite_metadata_values(capability, participants)
    normalized = {
        "participant_device_ids": participants,
        "participant_resource_ids": MappingProxyType(dict(sorted(
            values["participant_resources"].items()
        ))),
        "split_fractions_ppm": split_fractions,
        "layer_fractions_ppm": layer_fractions,
        "residency_states": tuple(sorted(values["residency"])),
        "resource_ids": values["resources"],
        "evidence_ids": values["evidence"],
        "operator_ids": tuple(sorted(values["operators"])),
        "operator_placements": tuple(sorted(
            placements, key=lambda row: row.operator_id
        )),
        "adapter_parameters": MappingProxyType(dict(sorted(
            values["adapter_parameters"].items()
        ))),
        "replacement_group_by_device": MappingProxyType(dict(sorted(
            values["replacement_groups"].items()
        ))),
    }
    for name, value in normalized.items():
        object.__setattr__(capability, name, value)


@dataclass(frozen=True)
class RuntimeCompositeExecutorCapability:
    """A physical coordinator for one capability-backed composite family."""

    executor_id: str
    endpoint: str
    backend: str
    coordinator_device_id: str
    participant_device_ids: tuple[str, ...]
    participant_resource_ids: Mapping[str, tuple[str, ...]]
    route_family: str
    assisted_operator_kind: str | None
    split_axis: str
    split_fractions_ppm: tuple[int, ...]
    layer_fractions_ppm: tuple[int, ...]
    residency_states: tuple[str, ...]
    resource_ids: tuple[str, ...]
    operator_plan_protocol: str
    maturity: str
    evidence_ids: tuple[str, ...]
    artifact_sha256: str | None = None
    operator_ids: tuple[str, ...] = ()
    operator_placements: tuple[RuntimeCompositeOperatorPlacement, ...] = ()
    adapter_parameters: Mapping[str, int | str] = field(default_factory=dict)
    replacement_group_by_device: Mapping[str, str] = field(
        default_factory=dict
    )
    baseline_executor_id: str | None = None
    helper_device_id: str | None = None

    def __post_init__(self) -> None:
        _initialize_composite_executor_capability(self)

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "baseline_executor_id": self.baseline_executor_id,
            "assisted_operator_kind": self.assisted_operator_kind,
            "backend": self.backend,
            "coordinator_device_id": self.coordinator_device_id,
            "endpoint": self.endpoint,
            "evidence_ids": list(self.evidence_ids),
            "executor_id": self.executor_id,
            "layer_fractions_ppm": list(self.layer_fractions_ppm),
            "helper_device_id": self.helper_device_id,
            "maturity": self.maturity,
            "operator_ids": list(self.operator_ids),
            "operator_plan_protocol": self.operator_plan_protocol,
            "participant_device_ids": list(self.participant_device_ids),
            "participant_resource_ids": {
                key: list(value)
                for key, value in self.participant_resource_ids.items()
            },
            "residency_states": list(self.residency_states),
            "resource_ids": list(self.resource_ids),
            "route_family": self.route_family,
            "split_axis": self.split_axis,
            "split_fractions_ppm": list(self.split_fractions_ppm),
        }
        if self.operator_placements:
            result["operator_placements"] = [
                row.to_json() for row in self.operator_placements
            ]
        if self.adapter_parameters:
            result["adapter_parameters"] = dict(self.adapter_parameters)
        if self.replacement_group_by_device:
            result["replacement_group_by_device"] = dict(
                self.replacement_group_by_device
            )
        return result

    @classmethod
    def from_json(
        cls, value: object
    ) -> "RuntimeCompositeExecutorCapability":
        row = _object("runtime composite executor", value)
        participant_resources = _object(
            "runtime composite participant resources",
            row.get("participant_resource_ids"),
        )
        return cls(
            executor_id=row.get("executor_id"),
            endpoint=row.get("endpoint"),
            backend=row.get("backend"),
            coordinator_device_id=row.get("coordinator_device_id"),
            participant_device_ids=tuple(_list(
                "runtime composite participants",
                row.get("participant_device_ids"),
            )),
            participant_resource_ids={
                key: tuple(_list(
                    "runtime composite participant resource rows", rows
                ))
                for key, rows in participant_resources.items()
            },
            route_family=row.get("route_family"),
            assisted_operator_kind=row.get("assisted_operator_kind"),
            split_axis=row.get("split_axis"),
            split_fractions_ppm=tuple(_list(
                "runtime composite split fractions",
                row.get("split_fractions_ppm", []),
            )),
            layer_fractions_ppm=tuple(_list(
                "runtime composite layer fractions",
                row.get("layer_fractions_ppm", []),
            )),
            residency_states=tuple(_list(
                "runtime composite residency states",
                row.get("residency_states"),
            )),
            resource_ids=tuple(_list(
                "runtime composite resources", row.get("resource_ids")
            )),
            operator_plan_protocol=row.get("operator_plan_protocol"),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime composite evidence", row.get("evidence_ids")
            )),
            artifact_sha256=row.get("artifact_sha256"),
            baseline_executor_id=row.get("baseline_executor_id"),
            helper_device_id=row.get("helper_device_id"),
            operator_ids=tuple(_list(
                "runtime composite operators", row.get("operator_ids", [])
            )),
            operator_placements=tuple(
                RuntimeCompositeOperatorPlacement.from_json(value)
                for value in _list(
                    "runtime composite operator placements",
                    row.get("operator_placements", []),
                )
            ),
            adapter_parameters={
                key: item
                for key, item in _object(
                    "runtime composite adapter parameters",
                    row.get("adapter_parameters", {}),
                ).items()
            },
            replacement_group_by_device={
                key: item
                for key, item in _object(
                    "runtime composite replacement ownership",
                    row.get("replacement_group_by_device", {}),
                ).items()
            },
        )
