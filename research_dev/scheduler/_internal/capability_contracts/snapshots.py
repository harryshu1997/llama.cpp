"""Device, executor, catalog and snapshot capability contracts: snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

from ..placement import PlacementHardwareProfile
from ..runtime_placement import RuntimePlacementSnapshot
from ..runtime_system_cost import RuntimeProtectedWorkObservation
from .common import (
    RESIDENCY_STATES,
    RUNTIME_SYSTEM_SNAPSHOT_SCHEMA,
    RuntimeCapabilityError,
    _boolean,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)
from .catalog import RuntimeCapabilityCatalog
from ..types import CANONICAL_OMIT_DEFAULT


@dataclass(frozen=True)
class RuntimeExecutorState:
    executor_id: str
    healthy: bool
    ready: bool
    temperature_millic: int
    battery_ppm: int
    free_slots: int
    busy_until_us: int
    thermal_qualified: bool | None = None
    charging: bool | None = None
    # raw Android thermal status (0..6); carried only under an opt-in device policy
    thermal_status: int | None = field(
        default=None, metadata={CANONICAL_OMIT_DEFAULT: True}
    )

    def __post_init__(self) -> None:
        _text("runtime executor state id", self.executor_id)
        _boolean("runtime executor healthy", self.healthy)
        _boolean("runtime executor ready", self.ready)
        _integer("runtime executor temperature", self.temperature_millic)
        battery = _integer("runtime executor battery", self.battery_ppm)
        if battery > 1_000_000:
            raise RuntimeCapabilityError("runtime battery exceeds one")
        _integer("runtime executor free slots", self.free_slots)
        _integer("runtime executor busy until", self.busy_until_us)
        if self.thermal_qualified is not None:
            _boolean(
                "runtime executor thermal qualification",
                self.thermal_qualified,
            )
        if self.charging is not None:
            _boolean("runtime executor charging", self.charging)
        if self.thermal_status is not None:
            _integer("runtime executor thermal status", self.thermal_status)

    def thermal_qualified_under(
        self, maximum_thermal_status: int
    ) -> bool | None:
        """Thermal qualification under a device's Android thermal status limit.

        An observed raw status qualifies while it does not exceed the limit;
        without one the probe's own verdict stands (None when unknown).
        """
        if self.thermal_status is not None:
            return self.thermal_status <= maximum_thermal_status
        return self.thermal_qualified

    def to_json(self) -> dict[str, int | bool | str]:
        result = {
            "battery_ppm": self.battery_ppm,
            "busy_until_us": self.busy_until_us,
            "executor_id": self.executor_id,
            "free_slots": self.free_slots,
            "healthy": self.healthy,
            "ready": self.ready,
            "temperature_millic": self.temperature_millic,
        }
        if self.thermal_qualified is not None:
            result["thermal_qualified"] = self.thermal_qualified
        if self.charging is not None:
            result["charging"] = self.charging
        if self.thermal_status is not None:
            result["thermal_status"] = self.thermal_status
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeExecutorState":
        row = _object("runtime executor state", value)
        return cls(
            executor_id=row.get("executor_id"),
            healthy=row.get("healthy"),
            ready=row.get("ready"),
            temperature_millic=row.get("temperature_millic"),
            battery_ppm=row.get("battery_ppm"),
            free_slots=row.get("free_slots"),
            busy_until_us=row.get("busy_until_us"),
            thermal_qualified=row.get("thermal_qualified"),
            charging=row.get("charging"),
            thermal_status=row.get("thermal_status"),
        )


@dataclass(frozen=True)
class RuntimeLinkState:
    link_id: str
    ready: bool
    measured_bandwidth_bytes_per_s: int
    busy_until_us: int

    def __post_init__(self) -> None:
        _text("runtime link state id", self.link_id)
        _boolean("runtime link ready", self.ready)
        _integer(
            "runtime link bandwidth", self.measured_bandwidth_bytes_per_s, 1
        )
        _integer("runtime link busy until", self.busy_until_us)

    def to_json(self) -> dict[str, int | bool | str]:
        return {
            "busy_until_us": self.busy_until_us,
            "link_id": self.link_id,
            "measured_bandwidth_bytes_per_s": (
                self.measured_bandwidth_bytes_per_s
            ),
            "ready": self.ready,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeLinkState":
        row = _object("runtime link state", value)
        return cls(
            link_id=row.get("link_id"),
            ready=row.get("ready"),
            measured_bandwidth_bytes_per_s=row.get(
                "measured_bandwidth_bytes_per_s"
            ),
            busy_until_us=row.get("busy_until_us"),
        )


@dataclass(frozen=True)
class ModelResidencyObservation:
    model_id: str
    artifact_sha256: str
    device_id: str
    state: str
    resident_tensor_ids: tuple[str, ...]
    resident_bytes: int
    generation: int
    executor_id: str | None = None
    reclaimable_bytes: int | None = None
    resident_geometry_sha256: str | None = None
    resident_adapter_parameters: Mapping[str, int | str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in ("model_id", "artifact_sha256", "device_id"):
            _text(f"model residency {name}", getattr(self, name))
        if self.state not in RESIDENCY_STATES:
            raise RuntimeCapabilityError("model residency state is invalid")
        tensors = _texts(
            "resident tensor id", self.resident_tensor_ids, allow_empty=True
        )
        _integer("model resident bytes", self.resident_bytes)
        _integer("model residency generation", self.generation)
        if self.executor_id is not None:
            _text("model residency executor id", self.executor_id)
        if self.reclaimable_bytes is not None:
            _integer(
                "model reclaimable bytes",
                self.reclaimable_bytes,
                self.resident_bytes,
            )
            if (
                self.reclaimable_bytes > self.resident_bytes
                and self.executor_id is None
            ):
                raise RuntimeCapabilityError(
                    "additional reclaimable memory requires an executor"
                )
        if self.resident_geometry_sha256 is not None:
            geometry = self.resident_geometry_sha256
            if (
                self.executor_id is None
                or not geometry.startswith("sha256:")
                or len(geometry) != 71
                or any(
                    value not in "0123456789abcdef"
                    for value in geometry[7:]
                )
            ):
                raise RuntimeCapabilityError(
                    "model resident geometry identity is invalid"
                )
        parameters = dict(self.resident_adapter_parameters)
        if any(
            type(name) is not str
            or not name
            or not name.isascii()
            or type(value) not in {int, str}
            or (type(value) is int and value < 0)
            or (type(value) is str and (not value or not value.isascii()))
            for name, value in parameters.items()
        ):
            raise RuntimeCapabilityError(
                "model resident adapter parameters are invalid"
            )
        if self.state == "cold" and (
            tensors
            or self.resident_bytes
            or self.executor_id is not None
            or self.reclaimable_bytes is not None
            or self.resident_geometry_sha256 is not None
            or parameters
        ):
            raise RuntimeCapabilityError("cold residency cannot own tensors")
        object.__setattr__(self, "resident_tensor_ids", tuple(sorted(tensors)))
        object.__setattr__(
            self,
            "resident_adapter_parameters",
            MappingProxyType(dict(sorted(parameters.items()))),
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "device_id": self.device_id,
            "generation": self.generation,
            "model_id": self.model_id,
            "resident_bytes": self.resident_bytes,
            "resident_tensor_ids": list(self.resident_tensor_ids),
            "state": self.state,
        }
        if self.executor_id is not None:
            result["executor_id"] = self.executor_id
        if self.reclaimable_bytes is not None:
            result["reclaimable_bytes"] = self.reclaimable_bytes
        if self.resident_geometry_sha256 is not None:
            result["resident_geometry_sha256"] = (
                self.resident_geometry_sha256
            )
        if self.resident_adapter_parameters:
            result["resident_adapter_parameters"] = dict(
                self.resident_adapter_parameters
            )
        return result

    @classmethod
    def from_json(cls, value: object) -> "ModelResidencyObservation":
        row = _object("model residency observation", value)
        return cls(
            model_id=row.get("model_id"),
            artifact_sha256=row.get("artifact_sha256"),
            device_id=row.get("device_id"),
            state=row.get("state"),
            resident_tensor_ids=tuple(_list(
                "model resident tensor ids", row.get("resident_tensor_ids")
            )),
            resident_bytes=row.get("resident_bytes"),
            generation=row.get("generation"),
            executor_id=row.get("executor_id"),
            reclaimable_bytes=row.get("reclaimable_bytes"),
            resident_geometry_sha256=row.get(
                "resident_geometry_sha256"
            ),
            resident_adapter_parameters=row.get(
                "resident_adapter_parameters", {}
            ),
        )


@dataclass(frozen=True)
class PhoneSessionResidencyObservation:
    session_id: str
    device_id: str
    executor_id: str
    endpoint: str
    artifact_sha256: str
    resident_geometry_sha256: str
    operator_plan_sha256: str
    session_generation: int
    resident_bytes: int
    state: str = "READY"

    def __post_init__(self) -> None:
        for name in ("session_id", "device_id", "executor_id", "endpoint"):
            _text("phone session residency " + name, getattr(self, name))
        for name in (
            "artifact_sha256",
            "resident_geometry_sha256",
            "operator_plan_sha256",
        ):
            value = _text(
                "phone session residency " + name, getattr(self, name)
            )
            if (
                not value.startswith("sha256:")
                or len(value) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in value[7:]
                )
            ):
                raise RuntimeCapabilityError(
                    "phone session residency identity is invalid"
                )
        _integer(
            "phone session residency generation",
            self.session_generation,
            1,
        )
        _integer(
            "phone session resident bytes", self.resident_bytes, 1
        )
        if self.state not in {
            "READY", "DRAINING", "LOADING", "FAILED"
        }:
            raise RuntimeCapabilityError(
                "phone session residency state is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "device_id": self.device_id,
            "endpoint": self.endpoint,
            "executor_id": self.executor_id,
            "operator_plan_sha256": self.operator_plan_sha256,
            "resident_bytes": self.resident_bytes,
            "resident_geometry_sha256": (
                self.resident_geometry_sha256
            ),
            "session_generation": self.session_generation,
            "session_id": self.session_id,
            "state": self.state,
        }

    @classmethod
    def from_json(
        cls, value: object
    ) -> "PhoneSessionResidencyObservation":
        row = _object("phone session residency observation", value)
        return cls(
            session_id=row.get("session_id"),
            device_id=row.get("device_id"),
            executor_id=row.get("executor_id"),
            endpoint=row.get("endpoint"),
            artifact_sha256=row.get("artifact_sha256"),
            resident_geometry_sha256=row.get(
                "resident_geometry_sha256"
            ),
            operator_plan_sha256=row.get("operator_plan_sha256"),
            session_generation=row.get("session_generation"),
            resident_bytes=row.get("resident_bytes"),
            state=row.get("state", "READY"),
        )


@dataclass(frozen=True)
class HeterogeneousRuntimeSnapshot:
    snapshot_id: str
    captured_at_us: int
    valid_until_us: int
    memory: RuntimePlacementSnapshot
    executors: Mapping[str, RuntimeExecutorState]
    links: Mapping[str, RuntimeLinkState]
    residency: tuple[ModelResidencyObservation, ...]
    cost_features: Mapping[str, int] = field(default_factory=dict)
    protected_work: RuntimeProtectedWorkObservation | None = None
    phone_session_residency: tuple[
        PhoneSessionResidencyObservation, ...
    ] = ()
    telemetry_observations: Mapping[str, Mapping[str, object]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _text("system snapshot id", self.snapshot_id)
        _integer("system snapshot captured at", self.captured_at_us)
        _integer("system snapshot valid until", self.valid_until_us, 1)
        if self.valid_until_us <= self.captured_at_us:
            raise RuntimeCapabilityError("system snapshot validity is empty")
        if not isinstance(self.memory, RuntimePlacementSnapshot):
            raise RuntimeCapabilityError("system memory snapshot is invalid")
        if (
            self.memory.captured_at_us > self.captured_at_us
            or self.memory.valid_until_us < self.valid_until_us
        ):
            raise RuntimeCapabilityError("memory snapshot does not cover system snapshot")
        executors = dict(self.executors)
        if any(
            not isinstance(row, RuntimeExecutorState) or key != row.executor_id
            for key, row in executors.items()
        ):
            raise RuntimeCapabilityError("system executor states are invalid")
        links = dict(self.links)
        if any(
            not isinstance(row, RuntimeLinkState) or key != row.link_id
            for key, row in links.items()
        ):
            raise RuntimeCapabilityError("system link states are invalid")
        residency = tuple(self.residency)
        if any(not isinstance(row, ModelResidencyObservation) for row in residency):
            raise RuntimeCapabilityError("system residency rows are invalid")
        keys = [(row.model_id, row.artifact_sha256, row.device_id) for row in residency]
        if len(keys) != len(set(keys)):
            raise RuntimeCapabilityError("system residency rows are duplicated")
        phone_session_residency = tuple(self.phone_session_residency)
        if (
            any(
                not isinstance(row, PhoneSessionResidencyObservation)
                for row in phone_session_residency
            )
            or len({
                (row.device_id, row.session_id)
                for row in phone_session_residency
            }) != len(phone_session_residency)
        ):
            raise RuntimeCapabilityError(
                "system phone session residency rows are invalid"
            )
        cost_features = {
            _text("system cost feature", key): _integer(
                "system cost feature value", value
            )
            for key, value in self.cost_features.items()
        }
        if (
            self.protected_work is not None
            and not isinstance(
                self.protected_work, RuntimeProtectedWorkObservation
            )
        ):
            raise RuntimeCapabilityError(
                "system protected work observation is invalid"
            )
        object.__setattr__(
            self, "executors", MappingProxyType(dict(sorted(executors.items())))
        )
        object.__setattr__(self, "links", MappingProxyType(dict(sorted(links.items()))))
        object.__setattr__(
            self,
            "residency",
            tuple(sorted(residency, key=lambda row: (
                row.model_id, row.artifact_sha256, row.device_id
            ))),
        )
        object.__setattr__(
            self,
            "cost_features",
            MappingProxyType(dict(sorted(cost_features.items()))),
        )
        object.__setattr__(
            self,
            "phone_session_residency",
            tuple(sorted(
                phone_session_residency,
                key=lambda row: (row.device_id, row.session_id),
            )),
        )
        observations = {}
        for device_id, observation in self.telemetry_observations.items():
            _text("telemetry device", device_id)
            if not isinstance(observation, Mapping):
                raise RuntimeCapabilityError("telemetry observation must be an object")
            row = dict(observation)
            if row.get("validity") not in {
                "VALID", "MISSING", "STALE", "TIMED_OUT", "MALFORMED",
                "UNAVAILABLE",
            } or row.get("valid") is not (row.get("validity") == "VALID"):
                raise RuntimeCapabilityError("telemetry validity is invalid")
            _text("telemetry source", row.get("source"))
            if row["valid"]:
                _integer("telemetry age", row.get("age_us"))
                _integer("telemetry maximum age", row.get("maximum_age_us"), 1)
                _integer("telemetry sample timestamp", row.get("sample_timestamp_ns"))
            else:
                _text("telemetry failure reason", row.get("failure_reason"))
            observations[device_id] = MappingProxyType(row)
        object.__setattr__(self, "telemetry_observations", MappingProxyType(observations))

    def telemetry_unavailable_reason(
        self, device_id: str, observed_at_us: int,
    ) -> str | None:
        row = self.telemetry_observations.get(device_id)
        if row is None:
            return None
        if not row["valid"]:
            return str(row["validity"]) + ": " + str(row["failure_reason"])
        age_us = int(row["age_us"]) + max(0, observed_at_us - self.captured_at_us)
        if age_us > int(row["maximum_age_us"]):
            return "STALE: phone sample age_us=" + str(age_us)
        return None

    def validate_at(self, observed_at_us: int) -> None:
        observed_at_us = _integer("system observation time", observed_at_us)
        if not self.captured_at_us <= observed_at_us < self.valid_until_us:
            raise RuntimeCapabilityError("system snapshot is stale")

    def residency_for(
        self, model_id: str, artifact_sha256: str, device_id: str
    ) -> ModelResidencyObservation | None:
        return next((
            row for row in self.residency
            if row.model_id == model_id
            and row.artifact_sha256 == artifact_sha256
            and row.device_id == device_id
        ), None)

    def effective_placement_profile(
        self, catalog: RuntimeCapabilityCatalog
    ) -> PlacementHardwareProfile:
        devices = {}
        by_device = catalog.executor_by_device
        for device_id, device in catalog.placement_profile.devices.items():
            capability = by_device.get(device_id)
            if capability is None:
                devices[device_id] = device
                continue
            state = self.executors.get(capability.executor_id)
            devices[device_id] = replace(
                device,
                ready=(
                    state is not None
                    and state.healthy
                    and state.ready
                    and state.free_slots > 0
                ),
            )
        links = tuple(
            replace(
                link,
                ready=(state is not None and state.ready),
                bandwidth_bytes_per_s=(
                    link.bandwidth_bytes_per_s
                    if state is None
                    else state.measured_bandwidth_bytes_per_s
                ),
            )
            for link in catalog.placement_profile.links
            for state in (self.links.get(link.link_id),)
        )
        return replace(catalog.placement_profile, devices=devices, links=links)

    def busy_until_by_resource(
        self, catalog: RuntimeCapabilityCatalog
    ) -> Mapping[str, int]:
        values: dict[str, int] = {}
        for capability in catalog.executors:
            state = self.executors.get(capability.executor_id)
            if state is None:
                continue
            for resource_id in capability.execution_resource_ids:
                values[resource_id] = max(
                    values.get(resource_id, 0), state.busy_until_us
                )
        for link_id, state in self.links.items():
            resource_id = "link:" + link_id
            if resource_id in catalog.resources:
                values[resource_id] = max(
                    values.get(resource_id, 0), state.busy_until_us
                )
        return MappingProxyType(dict(sorted(values.items())))

    def to_json(self) -> dict[str, object]:
        result = {
            "captured_at_us": self.captured_at_us,
            "cost_features": dict(self.cost_features),
            "executors": [row.to_json() for row in self.executors.values()],
            "links": [row.to_json() for row in self.links.values()],
            "memory": self.memory.to_json(),
            "residency": [row.to_json() for row in self.residency],
            "phone_session_residency": [
                row.to_json() for row in self.phone_session_residency
            ],
            "schema": RUNTIME_SYSTEM_SNAPSHOT_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "valid_until_us": self.valid_until_us,
        }
        if self.protected_work is not None:
            result["protected_work"] = self.protected_work.to_json()
        if self.telemetry_observations:
            result["telemetry_observations"] = {
                key: dict(value) for key, value in self.telemetry_observations.items()
            }
        return result

    @classmethod
    def from_json(cls, value: object) -> "HeterogeneousRuntimeSnapshot":
        row = _object("heterogeneous runtime snapshot", value)
        if row.get("schema") != RUNTIME_SYSTEM_SNAPSHOT_SCHEMA:
            raise RuntimeCapabilityError(
                "runtime system snapshot schema differs"
            )
        executors = tuple(
            RuntimeExecutorState.from_json(value)
            for value in _list(
                "runtime system executors", row.get("executors")
            )
        )
        links = tuple(
            RuntimeLinkState.from_json(value)
            for value in _list("runtime system links", row.get("links"))
        )
        return cls(
            snapshot_id=row.get("snapshot_id"),
            captured_at_us=row.get("captured_at_us"),
            valid_until_us=row.get("valid_until_us"),
            memory=RuntimePlacementSnapshot.from_json(row.get("memory")),
            executors={item.executor_id: item for item in executors},
            links={item.link_id: item for item in links},
            residency=tuple(
                ModelResidencyObservation.from_json(value)
                for value in _list(
                    "runtime system residency", row.get("residency")
                )
            ),
            cost_features=dict(_object(
                "runtime system cost features",
                row.get("cost_features", {}),
            )),
            protected_work=(
                None
                if row.get("protected_work") is None
                else RuntimeProtectedWorkObservation.from_json(
                    row.get("protected_work")
                )
            ),
            phone_session_residency=tuple(
                PhoneSessionResidencyObservation.from_json(value)
                for value in _list(
                    "runtime system phone session residency",
                    row.get("phone_session_residency", []),
                )
            ),
            telemetry_observations=row.get("telemetry_observations", {}),
        )
