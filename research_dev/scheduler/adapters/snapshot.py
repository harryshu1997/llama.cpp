"""Build scheduler runtime snapshots from raw endpoint and device probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    ModelResidencyObservation,
    PhoneSessionResidencyObservation,
    RuntimeCapabilityCatalog,
    RuntimeExecutorState,
    RuntimeLinkState,
)
from .._internal.runtime_placement import RuntimePlacementSnapshot
from .._internal.runtime_system_cost import RuntimeProtectedWorkObservation
from .contracts import PhysicalAdapterError


@dataclass(frozen=True)
class EndpointRuntimeSample:
    health: str
    slots_probe: str
    free_slots: int
    busy_until_us: int = 0
    transition_available: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.health) is not str
            or not self.health
            or not self.health.isascii()
            or type(self.slots_probe) is not str
            or not self.slots_probe
            or not self.slots_probe.isascii()
            or type(self.free_slots) is not int
            or self.free_slots < 0
            or type(self.busy_until_us) is not int
            or self.busy_until_us < 0
            or type(self.transition_available) is not bool
        ):
            raise PhysicalAdapterError("endpoint runtime sample is invalid")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> "EndpointRuntimeSample":
        return cls(
            health=str(value.get("health", "unavailable")),
            slots_probe=str(value.get("slots_probe", "unavailable")),
            free_slots=int(value.get("free_slots", 0)),
            busy_until_us=int(value.get("busy_until_us", 0)),
            transition_available=(
                value.get("transition_available", False) is True
            ),
        )

    @property
    def healthy(self) -> bool:
        return self.health == "healthy" or self.transition_available

    @property
    def ready(self) -> bool:
        return self.healthy and self.slots_probe == "live"


def materialize_preflight_executor_samples(
    catalog: RuntimeCapabilityCatalog,
    endpoint_samples: Mapping[str, EndpointRuntimeSample],
    *,
    bootstrap_executor_ids: Sequence[str] = (),
) -> Mapping[str, EndpointRuntimeSample]:
    """Combine raw endpoint probes with declared bootstrap transitions."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("preflight executor catalog is invalid")
    samples = dict(endpoint_samples)
    if any(
        type(endpoint) is not str
        or not endpoint
        or not endpoint.isascii()
        or not isinstance(sample, EndpointRuntimeSample)
        for endpoint, sample in samples.items()
    ):
        raise PhysicalAdapterError("preflight endpoint samples are invalid")
    bootstrap = tuple(bootstrap_executor_ids)
    if (
        len(bootstrap) != len(set(bootstrap))
        or any(
            type(executor_id) is not str
            or not executor_id
            or not executor_id.isascii()
            for executor_id in bootstrap
        )
    ):
        raise PhysicalAdapterError("preflight bootstrap executors are invalid")
    capabilities = (*catalog.executors, *catalog.composite_executors)
    known_ids = {row.executor_id for row in capabilities}
    if set(bootstrap) - known_ids:
        raise PhysicalAdapterError("preflight bootstrap executor is unknown")
    transition_ids = {
        row.executor_id
        for row in catalog.transitions
        if row.executor_id is not None and row.maturity == "QUALIFIED"
    }
    unavailable = EndpointRuntimeSample(
        "unavailable", "unavailable", 0
    )
    result = {}
    for capability in capabilities:
        observed = samples.get(capability.endpoint, unavailable)
        if observed.ready:
            result[capability.executor_id] = observed
            continue
        if capability.executor_id in bootstrap:
            resource_ids = (
                capability.execution_resource_ids
                if hasattr(capability, "execution_resource_ids")
                else capability.resource_ids
            )
            free_slots = min(
                catalog.resources[resource_id].capacity
                for resource_id in resource_ids
            )
            result[capability.executor_id] = EndpointRuntimeSample(
                "healthy", "live", free_slots
            )
            continue
        if capability.executor_id in transition_ids:
            result[capability.executor_id] = EndpointRuntimeSample(
                observed.health,
                observed.slots_probe,
                observed.free_slots,
                observed.busy_until_us,
                transition_available=True,
            )
            continue
        result[capability.executor_id] = observed
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class DeviceRuntimeTelemetry:
    temperature_millic: int = 0
    battery_ppm: int = 1_000_000
    thermal_qualified: bool | None = None
    charging: bool | None = None
    thermal_status: int | None = None

    def __post_init__(self) -> None:
        if (
            type(self.temperature_millic) is not int
            or self.temperature_millic < 0
            or type(self.battery_ppm) is not int
            or not 0 <= self.battery_ppm <= 1_000_000
            or (
                self.thermal_qualified is not None
                and type(self.thermal_qualified) is not bool
            )
            or (
                self.charging is not None
                and type(self.charging) is not bool
            )
            or (
                self.thermal_status is not None
                and (type(self.thermal_status) is not int or self.thermal_status < 0)
            )
        ):
            raise PhysicalAdapterError("device runtime telemetry is invalid")


def _policy_thermal_status(
    capability: object, live: DeviceRuntimeTelemetry
) -> int | None:
    """Raw Android thermal status for the executor state.

    Carried only for a device with an opt-in ``maximum_thermal_status`` policy,
    so a default catalog builds byte-identical snapshots.
    """
    if capability.maximum_thermal_status == 0:
        return None
    return live.thermal_status


@dataclass(frozen=True)
class ExecutorResidencySample:
    """A raw observation that one executor currently holds one model."""

    manifest: ModelManifest
    executor_id: str
    generation: int
    state: str = "hot"
    resident_device_ids: tuple[str, ...] = ()
    reclaimable_bytes_by_device: Mapping[str, int] = field(
        default_factory=dict
    )
    operator_plan: Mapping[str, object] | None = None
    resident_tensor_ids_by_device: Mapping[
        str, tuple[str, ...]
    ] = field(default_factory=dict)
    resident_bytes_by_device: Mapping[str, int] = field(
        default_factory=dict
    )
    resident_geometry_sha256: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, ModelManifest)
            or type(self.executor_id) is not str
            or not self.executor_id
            or not self.executor_id.isascii()
            or type(self.generation) is not int
            or self.generation < 0
            or self.state not in {"hot", "warm"}
            or type(self.resident_device_ids) is not tuple
            or len(self.resident_device_ids)
                != len(set(self.resident_device_ids))
            or any(
                type(device_id) is not str
                or not device_id
                or not device_id.isascii()
                for device_id in self.resident_device_ids
            )
        ):
            raise PhysicalAdapterError(
                "executor residency sample is invalid"
            )
        reclaimable = dict(self.reclaimable_bytes_by_device)
        if any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            or type(amount) is not int
            or amount <= 0
            for device_id, amount in reclaimable.items()
        ):
            raise PhysicalAdapterError(
                "executor reclaimable memory sample is invalid"
            )
        object.__setattr__(
            self,
            "reclaimable_bytes_by_device",
            MappingProxyType(dict(sorted(reclaimable.items()))),
        )
        if self.operator_plan is not None:
            if not isinstance(self.operator_plan, Mapping):
                raise PhysicalAdapterError(
                    "executor residency operator plan is invalid"
                )
            object.__setattr__(
                self,
                "operator_plan",
                MappingProxyType(dict(self.operator_plan)),
            )
        tensor_ids = {
            device_id: tuple(values)
            for device_id, values in self.resident_tensor_ids_by_device.items()
        }
        resident_bytes = dict(self.resident_bytes_by_device)
        if bool(tensor_ids) != bool(resident_bytes) or any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            or not values
            or len(values) != len(set(values))
            or any(
                type(value) is not str
                or not value
                or not value.isascii()
                for value in values
            )
            or type(resident_bytes.get(device_id)) is not int
            or resident_bytes[device_id] <= 0
            for device_id, values in tensor_ids.items()
        ) or set(tensor_ids) != set(resident_bytes):
            raise PhysicalAdapterError(
                "executor exact residency sample is invalid"
            )
        geometry = self.resident_geometry_sha256
        if geometry is not None and (
            not tensor_ids
            or not geometry.startswith("sha256:")
            or len(geometry) != 71
            or any(
                value not in "0123456789abcdef"
                for value in geometry[7:]
            )
        ):
            raise PhysicalAdapterError(
                "executor exact residency geometry is invalid"
            )
        object.__setattr__(
            self,
            "resident_tensor_ids_by_device",
            MappingProxyType(dict(sorted(tensor_ids.items()))),
        )
        object.__setattr__(
            self,
            "resident_bytes_by_device",
            MappingProxyType(dict(sorted(resident_bytes.items()))),
        )


def _preloaded_device_ids(capability: object) -> tuple[str, ...]:
    parameters = getattr(capability, "adapter_parameters", {})
    raw = parameters.get("preloaded_resident_device_ids")
    if raw is None:
        return ()
    if type(raw) is not str or not raw:
        raise PhysicalAdapterError(
            "preloaded residency device list is invalid"
        )
    device_ids = tuple(raw.split(","))
    participants = tuple(capability.participant_device_ids)
    if (
        len(device_ids) != len(set(device_ids))
        or any(device_id not in participants for device_id in device_ids)
    ):
        raise PhysicalAdapterError(
            "preloaded residency device list is invalid"
        )
    return device_ids


def catalog_preloaded_residency_samples(
    catalog: RuntimeCapabilityCatalog,
    manifests: Mapping[str, ModelManifest],
) -> tuple[ExecutorResidencySample, ...]:
    """Return exact participant allocations declared preloaded by a catalog."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("preloaded residency catalog is invalid")
    manifest_rows = dict(manifests)
    if any(
        type(model_id) is not str
        or not isinstance(manifest, ModelManifest)
        or manifest.model_id != model_id
        for model_id, manifest in manifest_rows.items()
    ):
        raise PhysicalAdapterError("preloaded residency manifests are invalid")
    by_artifact = {
        manifest.artifact_sha256: manifest
        for manifest in manifest_rows.values()
    }
    result = []
    for capability in catalog.composite_executors:
        device_ids = _preloaded_device_ids(capability)
        if not device_ids:
            continue
        manifest = by_artifact.get(capability.artifact_sha256)
        if manifest is None:
            raise PhysicalAdapterError(
                "preloaded residency artifact is not registered"
            )
        result.append(ExecutorResidencySample(
            manifest,
            capability.executor_id,
            1,
            resident_device_ids=device_ids,
        ))
    return tuple(result)


def live_executor_residency_sample(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    executor_id: str,
    *,
    generation: int,
    state: str = "hot",
    reclaimable_bytes_by_device: Mapping[str, int] | None = None,
    operator_plan: Mapping[str, object] | None = None,
) -> ExecutorResidencySample | None:
    """Describe only allocations created by the currently live endpoint."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("live residency catalog is invalid")
    base = catalog.executor_by_id.get(executor_id)
    composite = catalog.composite_executor_by_id.get(executor_id)
    if (base is None) == (composite is None):
        raise PhysicalAdapterError("live residency executor is unknown")
    reclaimable = (
        {}
        if reclaimable_bytes_by_device is None
        else dict(reclaimable_bytes_by_device)
    )
    if composite is None:
        return ExecutorResidencySample(
            manifest,
            executor_id,
            generation,
            state=state,
            reclaimable_bytes_by_device=reclaimable,
            operator_plan=operator_plan,
        )
    preloaded = set(_preloaded_device_ids(composite))
    if not preloaded:
        return ExecutorResidencySample(
            manifest,
            executor_id,
            generation,
            state=state,
            reclaimable_bytes_by_device=reclaimable,
            operator_plan=operator_plan,
        )
    dynamic_device_ids = tuple(
        device_id
        for device_id in composite.participant_device_ids
        if device_id not in preloaded
    )
    if any(device_id not in dynamic_device_ids for device_id in reclaimable):
        raise PhysicalAdapterError(
            "live reclaimable memory belongs to a preloaded allocation"
        )
    if not dynamic_device_ids:
        return None
    return ExecutorResidencySample(
        manifest,
        executor_id,
        generation,
        state=state,
        resident_device_ids=dynamic_device_ids,
        reclaimable_bytes_by_device=reclaimable,
        operator_plan=operator_plan,
    )


class RuntimeSnapshotBuilder:
    """Convert raw probes into generic device/link/residency contracts."""

    def __init__(
        self,
        catalog: RuntimeCapabilityCatalog,
        manifest: ModelManifest,
    ) -> None:
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise PhysicalAdapterError("runtime capability catalog is invalid")
        if not isinstance(manifest, ModelManifest):
            raise PhysicalAdapterError("runtime model manifest is invalid")
        self._catalog = catalog
        self._manifest = manifest

    def build(
        self,
        *,
        snapshot_id: str,
        captured_at_us: int,
        valid_until_us: int,
        memory: RuntimePlacementSnapshot,
        endpoint_samples: Mapping[str, EndpointRuntimeSample],
        device_telemetry: Mapping[str, DeviceRuntimeTelemetry] | None = None,
        resident_endpoints: Sequence[str] = (),
        cost_features: Mapping[str, int] | None = None,
        protected_work: RuntimeProtectedWorkObservation | None = None,
    ) -> HeterogeneousRuntimeSnapshot:
        if (
            type(snapshot_id) is not str
            or not snapshot_id
            or not snapshot_id.isascii()
            or type(captured_at_us) is not int
            or captured_at_us < 0
            or type(valid_until_us) is not int
            or valid_until_us < captured_at_us
            or not isinstance(memory, RuntimePlacementSnapshot)
        ):
            raise PhysicalAdapterError("runtime snapshot envelope is invalid")
        samples = dict(endpoint_samples)
        if any(
            type(endpoint) is not str
            or not endpoint
            or not isinstance(sample, EndpointRuntimeSample)
            for endpoint, sample in samples.items()
        ):
            raise PhysicalAdapterError("runtime endpoint samples are invalid")
        telemetry = {} if device_telemetry is None else dict(device_telemetry)
        if any(
            not isinstance(value, DeviceRuntimeTelemetry)
            for value in telemetry.values()
        ):
            raise PhysicalAdapterError("runtime device telemetry is invalid")
        resident = frozenset(resident_endpoints)
        state_by_device = {}
        executor_states = {}
        device_health: dict[str, bool] = {}
        for capability in self._catalog.executors:
            sample = samples.get(capability.endpoint)
            if sample is None:
                continue
            live = telemetry.get(
                capability.device_id, DeviceRuntimeTelemetry()
            )
            state = RuntimeExecutorState(
                executor_id=capability.executor_id,
                healthy=sample.healthy,
                ready=sample.ready,
                temperature_millic=live.temperature_millic,
                battery_ppm=live.battery_ppm,
                free_slots=(sample.free_slots if sample.ready else 0),
                busy_until_us=sample.busy_until_us,
                thermal_qualified=live.thermal_qualified,
                charging=live.charging,
                thermal_status=_policy_thermal_status(capability, live),
            )
            state_by_device[capability.device_id] = state
            executor_states[state.executor_id] = state
        for capability in self._catalog.composite_executors:
            sample = samples.get(capability.endpoint)
            if sample is None:
                continue
            executor_states[capability.executor_id] = RuntimeExecutorState(
                executor_id=capability.executor_id,
                healthy=sample.healthy,
                ready=sample.ready,
                temperature_millic=0,
                battery_ppm=1_000_000,
                free_slots=(sample.free_slots if sample.ready else 0),
                busy_until_us=sample.busy_until_us,
                thermal_qualified=None,
            )
            for device_id in capability.participant_device_ids:
                device_health[device_id] = (
                    device_health.get(device_id, False) or sample.healthy
                )
        links = {}
        for link in self._catalog.placement_profile.links:
            source = state_by_device.get(link.source_device)
            target = state_by_device.get(link.target_device)
            links[link.link_id] = RuntimeLinkState(
                link_id=link.link_id,
                ready=(
                    source is not None
                    and target is not None
                    and source.healthy
                    and target.healthy
                ),
                measured_bandwidth_bytes_per_s=link.bandwidth_bytes_per_s,
                busy_until_us=0,
            )
        resident_devices = {
            capability.device_id
            for capability in self._catalog.executors
            if capability.endpoint in resident
        }
        return HeterogeneousRuntimeSnapshot(
            snapshot_id=snapshot_id,
            captured_at_us=captured_at_us,
            valid_until_us=valid_until_us,
            memory=memory,
            executors=executor_states,
            links=links,
            residency=tuple(
                ModelResidencyObservation(
                    model_id=self._manifest.model_id,
                    artifact_sha256=self._manifest.artifact_sha256,
                    device_id=device_id,
                    state="hot",
                    resident_tensor_ids=tuple(
                        row.tensor_id for row in self._manifest.tensors
                    ),
                    resident_bytes=self._manifest.tensor_bytes,
                    generation=1,
                )
                for device_id in sorted(resident_devices)
            ),
            cost_features=(
                {} if cost_features is None else cost_features
            ),
            protected_work=protected_work,
        )


class UnifiedRuntimeSnapshotBuilder:
    """Build one snapshot for every registered executor and resident model."""

    def __init__(self, catalog: RuntimeCapabilityCatalog) -> None:
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise PhysicalAdapterError("runtime capability catalog is invalid")
        self._catalog = catalog

    def build(
        self,
        *,
        snapshot_id: str,
        captured_at_us: int,
        valid_until_us: int,
        memory: RuntimePlacementSnapshot,
        executor_samples: Mapping[str, EndpointRuntimeSample],
        residencies: Sequence[ExecutorResidencySample] = (),
        phone_session_residencies: Sequence[
            PhoneSessionResidencyObservation
        ] = (),
        device_telemetry: Mapping[str, DeviceRuntimeTelemetry] | None = None,
        link_bandwidth_samples: Mapping[str, int] | None = None,
        cost_features: Mapping[str, int] | None = None,
        protected_work: RuntimeProtectedWorkObservation | None = None,
    ) -> HeterogeneousRuntimeSnapshot:
        if (
            type(snapshot_id) is not str
            or not snapshot_id
            or not snapshot_id.isascii()
            or type(captured_at_us) is not int
            or captured_at_us < 0
            or type(valid_until_us) is not int
            or valid_until_us < captured_at_us
            or not isinstance(memory, RuntimePlacementSnapshot)
        ):
            raise PhysicalAdapterError("runtime snapshot envelope is invalid")
        inputs = self._validated_build_inputs(
            executor_samples,
            device_telemetry,
            link_bandwidth_samples,
            phone_session_residencies,
        )
        executor_states, device_health = self._executor_states(
            inputs.samples, inputs.telemetry
        )
        link_states = self._link_states(device_health, inputs.link_samples)
        residency_by_key = self._merged_residency(residencies)
        return HeterogeneousRuntimeSnapshot(
            snapshot_id=snapshot_id,
            captured_at_us=captured_at_us,
            valid_until_us=valid_until_us,
            memory=memory,
            executors=executor_states,
            links=link_states,
            residency=tuple(
                residency_by_key[key]
                for key in sorted(residency_by_key)
            ),
            phone_session_residency=inputs.phone_session_rows,
            cost_features=(
                {} if cost_features is None else cost_features
            ),
            protected_work=protected_work,
        )

    def _validated_build_inputs(
        self,
        executor_samples: Mapping[str, EndpointRuntimeSample],
        device_telemetry: Mapping[str, DeviceRuntimeTelemetry] | None,
        link_bandwidth_samples: Mapping[str, int] | None,
        phone_session_residencies: Sequence[PhoneSessionResidencyObservation],
    ) -> "_UnifiedBuildInputs":
        samples = dict(executor_samples)
        known_executor_ids = {
            *(row.executor_id for row in self._catalog.executors),
            *(
                row.executor_id
                for row in self._catalog.composite_executors
            ),
        }
        if any(
            executor_id not in known_executor_ids
            or not isinstance(sample, EndpointRuntimeSample)
            for executor_id, sample in samples.items()
        ):
            raise PhysicalAdapterError(
                "runtime executor samples are invalid"
            )
        telemetry = {} if device_telemetry is None else dict(device_telemetry)
        if any(
            device_id not in self._catalog.placement_profile.devices
            or not isinstance(value, DeviceRuntimeTelemetry)
            for device_id, value in telemetry.items()
        ):
            raise PhysicalAdapterError("runtime device telemetry is invalid")
        link_samples = (
            {} if link_bandwidth_samples is None
            else dict(link_bandwidth_samples)
        )
        known_link_ids = {
            row.link_id for row in self._catalog.placement_profile.links
        }
        if any(
            link_id not in known_link_ids
            or type(value) is not int
            or value <= 0
            for link_id, value in link_samples.items()
        ):
            raise PhysicalAdapterError(
                "runtime link bandwidth samples are invalid"
            )
        phone_session_rows = tuple(phone_session_residencies)
        if any(
            not isinstance(row, PhoneSessionResidencyObservation)
            for row in phone_session_rows
        ):
            raise PhysicalAdapterError(
                "runtime phone session observations are invalid"
            )
        return _UnifiedBuildInputs(
            samples=samples,
            telemetry=telemetry,
            link_samples=link_samples,
            phone_session_rows=phone_session_rows,
        )

    def _executor_states(
        self,
        samples: Mapping[str, EndpointRuntimeSample],
        telemetry: Mapping[str, DeviceRuntimeTelemetry],
    ) -> tuple[dict[str, RuntimeExecutorState], dict[str, bool]]:
        executor_states = {}
        device_health: dict[str, bool] = {}
        for capability in self._catalog.executors:
            sample = samples.get(capability.executor_id)
            if sample is None:
                continue
            live = telemetry.get(
                capability.device_id, DeviceRuntimeTelemetry()
            )
            executor_states[capability.executor_id] = RuntimeExecutorState(
                executor_id=capability.executor_id,
                healthy=sample.healthy,
                ready=sample.ready,
                temperature_millic=live.temperature_millic,
                battery_ppm=live.battery_ppm,
                free_slots=(sample.free_slots if sample.ready else 0),
                busy_until_us=sample.busy_until_us,
                thermal_qualified=live.thermal_qualified,
                charging=live.charging,
                thermal_status=_policy_thermal_status(capability, live),
            )
            device_health[capability.device_id] = sample.healthy
        for capability in self._catalog.composite_executors:
            sample = samples.get(capability.executor_id)
            if sample is None:
                continue
            executor_states[capability.executor_id] = (
                _composite_executor_state(capability, sample, telemetry)
            )
            for device_id in capability.participant_device_ids:
                device_health[device_id] = (
                    device_health.get(device_id, False) or sample.healthy
                )
        return executor_states, device_health

    def _link_states(
        self,
        device_health: Mapping[str, bool],
        link_samples: Mapping[str, int],
    ) -> dict[str, RuntimeLinkState]:
        return {
            link.link_id: RuntimeLinkState(
                link_id=link.link_id,
                ready=(
                    link.ready
                    and device_health.get(link.source_device, False)
                    and device_health.get(link.target_device, False)
                ),
                measured_bandwidth_bytes_per_s=link_samples.get(
                    link.link_id, link.bandwidth_bytes_per_s
                ),
                busy_until_us=0,
            )
            for link in self._catalog.placement_profile.links
        }

    def _merged_residency(
        self,
        residencies: Sequence[ExecutorResidencySample],
    ) -> dict[tuple[str, str, str], ModelResidencyObservation]:
        residency_by_key: dict[
            tuple[str, str, str], ModelResidencyObservation
        ] = {}
        for sample in tuple(residencies):
            if not isinstance(sample, ExecutorResidencySample):
                raise PhysicalAdapterError(
                    "runtime residency observations are invalid"
                )
            for row in model_residency_observations(
                self._catalog,
                sample.manifest,
                sample.executor_id,
                generation=sample.generation,
                state=sample.state,
                resident_device_ids=sample.resident_device_ids,
                reclaimable_bytes_by_device=(
                    sample.reclaimable_bytes_by_device
                ),
                operator_plan=sample.operator_plan,
                resident_tensor_ids_by_device=(
                    sample.resident_tensor_ids_by_device
                ),
                resident_bytes_by_device=(
                    sample.resident_bytes_by_device
                ),
                resident_geometry_sha256=(
                    sample.resident_geometry_sha256
                ),
            ):
                key = (row.model_id, row.artifact_sha256, row.device_id)
                previous = residency_by_key.get(key)
                if previous is None:
                    residency_by_key[key] = row
                    continue
                residency_by_key[key] = _merge_residency_rows(previous, row)
        return residency_by_key


@dataclass(frozen=True)
class _UnifiedBuildInputs:
    samples: dict[str, EndpointRuntimeSample]
    telemetry: dict[str, DeviceRuntimeTelemetry]
    link_samples: dict[str, int]
    phone_session_rows: tuple[PhoneSessionResidencyObservation, ...]


def _composite_executor_state(
    capability: object,
    sample: EndpointRuntimeSample,
    telemetry: Mapping[str, DeviceRuntimeTelemetry],
) -> RuntimeExecutorState:
    participant_telemetry = tuple(
        telemetry.get(device_id, DeviceRuntimeTelemetry())
        for device_id in capability.participant_device_ids
    )
    thermal_qualifications = tuple(
        row.thermal_qualified
        for row in participant_telemetry
        if row.thermal_qualified is not None
    )
    return RuntimeExecutorState(
        executor_id=capability.executor_id,
        healthy=sample.healthy,
        ready=sample.ready,
        temperature_millic=max(
            row.temperature_millic
            for row in participant_telemetry
        ),
        battery_ppm=min(row.battery_ppm for row in participant_telemetry),
        free_slots=(sample.free_slots if sample.ready else 0),
        busy_until_us=sample.busy_until_us,
        thermal_qualified=(
            None
            if not thermal_qualifications
            else all(thermal_qualifications)
        ),
        charging=(
            True
            if any(row.charging is True for row in participant_telemetry)
            else False
            if participant_telemetry
            and all(
                row.charging is False
                for row in participant_telemetry
            )
            else None
        ),
    )


def _merge_residency_rows(
    previous: ModelResidencyObservation,
    row: ModelResidencyObservation,
) -> ModelResidencyObservation:
    """Merge two observations of one model on one device."""
    tensor_ids = tuple(sorted(set(
        previous.resident_tensor_ids
    ) | set(row.resident_tensor_ids)))
    exact_rows = tuple(
        value for value in (previous, row)
        if value.executor_id is not None
    )
    if (
        len(exact_rows) == 2
        and exact_rows[0].generation
            != exact_rows[1].generation
    ):
        return max(exact_rows, key=lambda value: value.generation)
    if len(exact_rows) == 2 and (
        exact_rows[0].executor_id != exact_rows[1].executor_id
        or exact_rows[0].resident_adapter_parameters
            != exact_rows[1].resident_adapter_parameters
        or (
            exact_rows[0].reclaimable_bytes is not None
            and exact_rows[1].reclaimable_bytes is not None
            and exact_rows[0].reclaimable_bytes
                != exact_rows[1].reclaimable_bytes
        )
        or (
            exact_rows[0].resident_geometry_sha256 is not None
            and exact_rows[1].resident_geometry_sha256 is not None
            and exact_rows[0].resident_geometry_sha256
                != exact_rows[1].resident_geometry_sha256
        )
    ):
        raise PhysicalAdapterError(
            "runtime residency endpoint allocation conflicts"
        )
    exact = exact_rows[0] if exact_rows else None
    return ModelResidencyObservation(
        model_id=row.model_id,
        artifact_sha256=row.artifact_sha256,
        device_id=row.device_id,
        state=(
            "hot"
            if "hot" in {previous.state, row.state}
            else "warm"
        ),
        resident_tensor_ids=tensor_ids,
        resident_bytes=max(
            previous.resident_bytes, row.resident_bytes
        ),
        generation=max(previous.generation, row.generation),
        executor_id=(
            None if exact is None else exact.executor_id
        ),
        reclaimable_bytes=(
            None
            if exact is None
            else next(
                (
                    value.reclaimable_bytes
                    for value in exact_rows
                    if value.reclaimable_bytes is not None
                ),
                None,
            )
        ),
        resident_geometry_sha256=(
            None
            if exact is None
            else next(
                (
                    value.resident_geometry_sha256
                    for value in exact_rows
                    if value.resident_geometry_sha256
                        is not None
                ),
                None,
            )
        ),
        resident_adapter_parameters=(
            {}
            if exact is None
            else exact.resident_adapter_parameters
        ),
    )


def _operator_plan_residency(
    manifest: ModelManifest,
    participant_device_ids: Sequence[str],
    operator_plan: Mapping[str, object],
) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    if operator_plan.get("schema") != "research-scheduler-execution-plan-v1":
        raise PhysicalAdapterError(
            "runtime residency operator plan schema is invalid"
        )
    raw_operators = operator_plan.get("operators")
    raw_memory = operator_plan.get("memory_demands")
    if type(raw_operators) is not list or type(raw_memory) is not list:
        raise PhysicalAdapterError(
            "runtime residency operator plan is incomplete"
        )
    participants = set(participant_device_ids)
    operator_by_id = {
        row.operator_id: row for row in manifest.operators
    }
    device_tensors: dict[str, dict[str, int]] = {
        device_id: {} for device_id in participant_device_ids
    }
    observed_operator_ids = set()
    for raw in raw_operators:
        if type(raw) is not dict:
            raise PhysicalAdapterError(
                "runtime residency operator assignment is invalid"
            )
        operator_id = raw.get("operator_id")
        device_ids = raw.get("device_ids")
        if (
            type(operator_id) is not str
            or operator_id not in operator_by_id
            or operator_id in observed_operator_ids
            or type(device_ids) is not list
            or not device_ids
            or len(device_ids) != len(set(device_ids))
            or not set(device_ids).issubset(participants)
        ):
            raise PhysicalAdapterError(
                "runtime residency operator assignment is invalid"
            )
        observed_operator_ids.add(operator_id)
        operator = operator_by_id[operator_id]
        for device_id in device_ids:
            tensors = device_tensors[device_id]
            for tensor_id in operator.tensor_ids:
                tensors[tensor_id] = manifest.tensor_by_id[tensor_id].nbytes
    if observed_operator_ids != set(operator_by_id):
        raise PhysicalAdapterError(
            "runtime residency operator coverage is incomplete"
        )
    resident_bytes_by_device = {}
    for raw in raw_memory:
        if type(raw) is not dict or raw.get("kind") != "model_weights":
            continue
        device_id = raw.get("device_id")
        required_bytes = raw.get("required_bytes")
        if (
            type(device_id) is not str
            or device_id not in participants
            or device_id in resident_bytes_by_device
            or type(required_bytes) is not int
            or required_bytes <= 0
        ):
            raise PhysicalAdapterError(
                "runtime residency weight demand is invalid"
            )
        resident_bytes_by_device[device_id] = required_bytes
    populated = {
        device_id for device_id, tensors in device_tensors.items()
        if tensors
    }
    if set(resident_bytes_by_device) != populated:
        raise PhysicalAdapterError(
            "runtime residency weight coverage is incomplete"
        )
    return device_tensors, resident_bytes_by_device


def _exact_residency_device_tensors(
    manifest: ModelManifest,
    selected_device_ids: tuple[str, ...],
    exact_tensor_ids: Mapping[str, tuple[str, ...]],
    exact_resident_bytes: Mapping[str, int],
    resident_geometry_sha256: str | None,
) -> dict[str, dict[str, int]]:
    if (
        set(exact_tensor_ids) != set(exact_resident_bytes)
        or resident_geometry_sha256 is None
        or any(
            device_id not in selected_device_ids
            or type(amount) is not int
            or amount <= 0
            or not tensor_ids
            or len(tensor_ids) != len(set(tensor_ids))
            or set(tensor_ids) - set(manifest.tensor_by_id)
            for device_id, tensor_ids in exact_tensor_ids.items()
            for amount in (exact_resident_bytes.get(device_id),)
        )
    ):
        raise PhysicalAdapterError(
            "runtime exact residency observation is invalid"
        )
    return {
        device_id: {
            tensor_id: manifest.tensor_by_id[tensor_id].nbytes
            for tensor_id in tensor_ids
        }
        for device_id, tensor_ids in exact_tensor_ids.items()
    }


def _operator_plan_geometry_sha256(
    parameters: Mapping[str, object],
) -> object:
    resident_geometry_sha256 = parameters.get(
        "phone_shard_set_geometry_sha256",
        parameters.get("ffn_resident_geometry_sha256"),
    )
    if resident_geometry_sha256 is not None and (
        type(resident_geometry_sha256) is not str
        or not resident_geometry_sha256.startswith("sha256:")
        or len(resident_geometry_sha256) != 71
        or any(
            value not in "0123456789abcdef"
            for value in resident_geometry_sha256[7:]
        )
    ):
        raise PhysicalAdapterError(
            "runtime resident geometry identity is invalid"
        )
    return resident_geometry_sha256


def _composite_placement_residency(
    manifest: ModelManifest,
    composite: object,
) -> dict[str, dict[str, int]]:
    if (
        composite.artifact_sha256 not in {
            None, manifest.artifact_sha256
        }
        or not composite.operator_placements
    ):
        raise PhysicalAdapterError(
            "runtime composite residency contract is incomplete"
        )
    operator_by_id = {
        row.operator_id: row for row in manifest.operators
    }
    device_tensors: dict[str, dict[str, int]] = {
        device_id: {}
        for device_id in composite.participant_device_ids
    }
    for placement in composite.operator_placements:
        operator = operator_by_id.get(placement.operator_id)
        if operator is None:
            raise PhysicalAdapterError(
                "runtime composite operator is absent from the model"
            )
        for tensor_id in operator.tensor_ids:
            tensor_bytes = manifest.tensor_by_id[tensor_id].nbytes
            helper_bytes = (
                0
                if placement.helper_device_id is None
                else (
                    tensor_bytes * placement.split_fraction_ppm
                    + 999_999
                ) // 1_000_000
            )
            primary_bytes = tensor_bytes - helper_bytes
            primary = device_tensors[placement.primary_device_id]
            primary[tensor_id] = max(
                primary.get(tensor_id, 0), primary_bytes
            )
            if placement.helper_device_id is not None:
                helper = device_tensors[placement.helper_device_id]
                helper[tensor_id] = max(
                    helper.get(tensor_id, 0), helper_bytes
                )
    return device_tensors


def _effective_reclaimable_bytes(
    reclaimable: dict[str, int],
    device_tensors: Mapping[str, Mapping[str, int]],
    resident_bytes_by_device: Mapping[str, int],
) -> dict[str, int]:
    if any(
        device_id not in device_tensors
        or type(amount) is not int
        or amount <= 0
        for device_id, amount in reclaimable.items()
    ):
        raise PhysicalAdapterError(
            "runtime reclaimable memory observation is invalid"
        )
    for device_id, amount in tuple(reclaimable.items()):
        resident_bytes = resident_bytes_by_device.get(
            device_id, sum(device_tensors[device_id].values())
        )
        if amount < resident_bytes:
            del reclaimable[device_id]
    return reclaimable


def model_residency_observations(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    executor_id: str,
    *,
    generation: int,
    state: str = "hot",
    resident_device_ids: Sequence[str] = (),
    reclaimable_bytes_by_device: Mapping[str, int] | None = None,
    operator_plan: Mapping[str, object] | None = None,
    resident_tensor_ids_by_device: Mapping[
        str, tuple[str, ...]
    ] | None = None,
    resident_bytes_by_device: Mapping[str, int] | None = None,
    resident_geometry_sha256: str | None = None,
) -> tuple[ModelResidencyObservation, ...]:
    """Describe weights placed by one live physical executor binding."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise PhysicalAdapterError("runtime residency catalog is invalid")
    if not isinstance(manifest, ModelManifest):
        raise PhysicalAdapterError("runtime residency manifest is invalid")
    if type(generation) is not int or generation < 0:
        raise PhysicalAdapterError("runtime residency generation is invalid")
    if state not in {"hot", "warm"}:
        raise PhysicalAdapterError("runtime residency state is invalid")
    selected_device_ids = tuple(resident_device_ids)
    reclaimable = (
        {} if reclaimable_bytes_by_device is None
        else dict(reclaimable_bytes_by_device)
    )
    if (
        len(selected_device_ids) != len(set(selected_device_ids))
        or any(
            type(device_id) is not str
            or not device_id
            or not device_id.isascii()
            for device_id in selected_device_ids
        )
    ):
        raise PhysicalAdapterError("runtime residency devices are invalid")
    base = catalog.executor_by_id.get(executor_id)
    composite = catalog.composite_executor_by_id.get(executor_id)
    if (base is None) == (composite is None):
        raise PhysicalAdapterError("runtime residency executor is unknown")
    exact_tensor_ids = (
        {} if resident_tensor_ids_by_device is None
        else dict(resident_tensor_ids_by_device)
    )
    exact_resident_bytes = (
        {} if resident_bytes_by_device is None
        else dict(resident_bytes_by_device)
    )
    resident_adapter_parameters = {}
    if operator_plan is not None:
        parameters = operator_plan.get("adapter_parameters", {})
        if not isinstance(parameters, Mapping):
            raise PhysicalAdapterError(
                "runtime residency adapter parameters are invalid"
            )
        resident_adapter_parameters = dict(parameters)
    resident_bytes_by_device = {}
    if bool(exact_tensor_ids) != bool(exact_resident_bytes):
        raise PhysicalAdapterError(
            "runtime exact residency observation is incomplete"
        )
    if exact_tensor_ids:
        device_tensors = _exact_residency_device_tensors(
            manifest,
            selected_device_ids,
            exact_tensor_ids,
            exact_resident_bytes,
            resident_geometry_sha256,
        )
        resident_bytes_by_device = exact_resident_bytes
    elif operator_plan is not None:
        resident_geometry_sha256 = _operator_plan_geometry_sha256(
            resident_adapter_parameters
        )
    if exact_tensor_ids:
        pass
    elif base is not None:
        device_tensors = {
            base.device_id: {
                row.tensor_id: row.nbytes for row in manifest.tensors
            }
        }
    elif operator_plan is not None:
        device_tensors, resident_bytes_by_device = _operator_plan_residency(
            manifest,
            composite.participant_device_ids,
            operator_plan,
        )
    else:
        device_tensors = _composite_placement_residency(manifest, composite)
    if selected_device_ids:
        if not set(selected_device_ids).issubset(device_tensors):
            raise PhysicalAdapterError(
                "runtime residency device is not an executor participant"
            )
        device_tensors = {
            device_id: device_tensors[device_id]
            for device_id in selected_device_ids
        }
        resident_bytes_by_device = {
            device_id: resident_bytes_by_device[device_id]
            for device_id in selected_device_ids
            if device_id in resident_bytes_by_device
        }
    reclaimable = _effective_reclaimable_bytes(
        reclaimable, device_tensors, resident_bytes_by_device
    )
    return tuple(
        ModelResidencyObservation(
            model_id=manifest.model_id,
            artifact_sha256=manifest.artifact_sha256,
            device_id=device_id,
            state=state,
            resident_tensor_ids=tuple(sorted(tensor_ids)),
            resident_bytes=resident_bytes_by_device.get(
                device_id, sum(tensor_bytes.values())
            ),
            generation=generation,
            executor_id=executor_id,
            reclaimable_bytes=reclaimable.get(device_id),
            resident_geometry_sha256=resident_geometry_sha256,
            resident_adapter_parameters=resident_adapter_parameters,
        )
        for device_id, tensor_bytes in sorted(device_tensors.items())
        for tensor_ids in (tuple(sorted(tensor_bytes)),)
        if tensor_bytes
    )
