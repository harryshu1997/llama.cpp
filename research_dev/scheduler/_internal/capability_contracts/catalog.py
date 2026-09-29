"""Device, executor, catalog and snapshot capability contracts: catalog."""

from __future__ import annotations

from dataclasses import dataclass
import json
from functools import cached_property
from types import MappingProxyType
from typing import Mapping

from ..placement import PlacementHardwareProfile
from ..policy import ResourceProfile
from ..runtime_system_cost import RuntimeSystemCostProfile
from .common import (
    RUNTIME_CAPABILITY_SCHEMA,
    RuntimeCapabilityError,
    _integer,
    _list,
    _object,
    _placement_profile_json,
    _text,
)
from .composites import RuntimeCompositeExecutorCapability
from .desktop import RuntimeDesktopControlProfile
from ..plan_contracts.co_helpers import co_helper_declaration
from ..plan_contracts.remote_resident import RuntimePlanError, RuntimeRemoteResidentFfn
from .executors import RuntimeExecutorCapability
from .phone import RuntimePhonePowerProfile
from .profiles import RuntimeRouteShapeProfile
from .transitions import RuntimeTransitionCapability


def _validate_catalog_executor(
    catalog: RuntimeCapabilityCatalog,
    executor: RuntimeExecutorCapability,
    resources: Mapping[str, ResourceProfile],
) -> None:
    profile = catalog.placement_profile
    if executor.device_id not in profile.devices:
        raise RuntimeCapabilityError("executor device is absent")
    if set(executor.execution_resource_ids) - set(resources):
        raise RuntimeCapabilityError("executor resource is absent")
    if (
        executor.exclusive_residency_resource_id is not None
        and executor.exclusive_residency_resource_id not in resources
    ):
        raise RuntimeCapabilityError(
            "executor exclusive residency resource is absent"
        )
    if executor.memory_resource_id not in profile.memory_pools:
        raise RuntimeCapabilityError("executor memory pool is absent")
    if set(executor.kernel_profiles.values()) - set(profile.kernels):
        raise RuntimeCapabilityError("executor kernel profile is absent")
    for session in executor.phone_sessions:
        if session.memory_resource_id not in profile.memory_pools:
            raise RuntimeCapabilityError("phone session memory pool is absent")
        session_resources = {
            session.shared_compute_resource_id,
            *session.shared_transport_resource_ids,
        }
        if session_resources - set(resources):
            raise RuntimeCapabilityError(
                "phone session shared resource is absent"
            )
        if session_resources - set(executor.execution_resource_ids):
            raise RuntimeCapabilityError(
                "phone session resource differs from its executor"
            )
        pool = profile.memory_pools[session.memory_resource_id]
        if session.resident_memory_limit_bytes > (
            pool.capacity_bytes - pool.reserved_bytes
        ):
            raise RuntimeCapabilityError(
                "phone session resident limit exceeds its pool"
            )
    for selector in executor.kernel_shape_profiles:
        kernel = profile.kernels.get(selector.profile_id)
        if kernel is None:
            raise RuntimeCapabilityError("kernel shape profile is absent")
        if kernel.device_id != executor.device_id:
            raise RuntimeCapabilityError(
                "kernel shape profile device differs"
            )


def _catalog_base_values(
    catalog: RuntimeCapabilityCatalog,
) -> tuple[dict[str, ResourceProfile], tuple[RuntimeExecutorCapability, ...]]:
    _text("runtime catalog id", catalog.catalog_id)
    if not isinstance(catalog.placement_profile, PlacementHardwareProfile):
        raise RuntimeCapabilityError("placement profile is invalid")
    resources = dict(catalog.resources)
    if not resources or any(
        not isinstance(value, ResourceProfile) or key != value.resource_id
        for key, value in resources.items()
    ):
        raise RuntimeCapabilityError("runtime resources are invalid")
    executors = tuple(catalog.executors)
    if not executors or any(
        not isinstance(row, RuntimeExecutorCapability) for row in executors
    ):
        raise RuntimeCapabilityError("runtime executors are invalid")
    if len({row.executor_id for row in executors}) != len(executors):
        raise RuntimeCapabilityError("runtime executor ids are duplicated")
    if len({row.device_id for row in executors}) != len(executors):
        raise RuntimeCapabilityError("runtime device ids are duplicated")
    if len([row for row in executors if row.qualified_fallback]) != 1:
        raise RuntimeCapabilityError("catalog requires one qualified fallback")
    for executor in executors:
        _validate_catalog_executor(catalog, executor, resources)
    return resources, executors


def _catalog_composite_executors(
    catalog: RuntimeCapabilityCatalog,
    executors: tuple[RuntimeExecutorCapability, ...],
    resources: Mapping[str, ResourceProfile],
) -> tuple[RuntimeCompositeExecutorCapability, ...]:
    composites = tuple(catalog.composite_executors)
    if (
        any(not isinstance(row, RuntimeCompositeExecutorCapability)
            for row in composites)
        or len({row.executor_id for row in composites}) != len(composites)
    ):
        raise RuntimeCapabilityError(
            "runtime composite executors are invalid"
        )
    base_ids = {row.executor_id for row in executors}
    if any(row.executor_id in base_ids for row in composites):
        raise RuntimeCapabilityError(
            "runtime executor ids are duplicated across capability kinds"
        )
    device_ids = {row.device_id for row in executors}
    executor_by_device = {row.device_id: row for row in executors}
    for coordinator in composites:
        if set(coordinator.participant_device_ids) - device_ids:
            raise RuntimeCapabilityError(
                "runtime composite participant is absent"
            )
        if set(coordinator.resource_ids) - set(resources):
            raise RuntimeCapabilityError("runtime composite resource is absent")
        _validate_composite_co_helpers(catalog, coordinator, executor_by_device)
        if coordinator.baseline_executor_id is not None:
            baseline = next((
                row for row in composites
                if row.executor_id == coordinator.baseline_executor_id
            ), None)
            if (
                baseline is None
                or baseline.route_family != "layer_placement"
                or not baseline.operator_placements
                or baseline.artifact_sha256 != coordinator.artifact_sha256
                or set(baseline.participant_device_ids)
                    - set(coordinator.participant_device_ids)
                or coordinator.helper_device_id
                    in baseline.participant_device_ids
            ):
                raise RuntimeCapabilityError(
                    "composite baseline executor is incompatible"
                )
        for replacement_group in set(
            coordinator.replacement_group_by_device.values()
        ):
            anchors = tuple(
                device_id for device_id in coordinator.participant_device_ids
                if executor_by_device[device_id].exclusive_residency_resource_id
                    == replacement_group
                or any(session.shared_compute_resource_id == replacement_group
                       for session in executor_by_device[device_id].phone_sessions)
            )
            if not anchors or any(
                coordinator.replacement_group_by_device.get(device_id)
                    != replacement_group
                for device_id in anchors
            ):
                raise RuntimeCapabilityError(
                    "composite replacement group lacks an exclusive anchor"
                )
    return composites


def _validate_composite_co_helpers(
    catalog: RuntimeCapabilityCatalog,
    coordinator: RuntimeCompositeExecutorCapability,
    executor_by_device: Mapping[str, RuntimeExecutorCapability],
) -> None:
    try:
        declaration = co_helper_declaration(coordinator.adapter_parameters)
    except RuntimePlanError as error:
        raise RuntimeCapabilityError(
            "composite co-helper declaration is invalid"
        ) from error
    if declaration is None:
        return
    if (
        coordinator.assisted_operator_kind != "ffn"
        or coordinator.helper_device_id is None
        or coordinator.baseline_executor_id is None
        or any(
            device_id not in coordinator.participant_device_ids
            or device_id in {
                coordinator.coordinator_device_id,
                coordinator.helper_device_id,
            }
            or "ffn" not in executor_by_device[device_id].operator_kinds
            or not catalog.placement_profile.devices[device_id].kind.startswith(
                "phone"
            )
            for device_id in declaration.device_ids
        )
    ):
        raise RuntimeCapabilityError(
            "composite co-helper phones are not FFN phone participants"
        )


def _catalog_desktop_controls(
    catalog: RuntimeCapabilityCatalog,
    executors: tuple[RuntimeExecutorCapability, ...],
    composites: tuple[RuntimeCompositeExecutorCapability, ...],
) -> tuple[RuntimeDesktopControlProfile, ...]:
    controls = tuple(catalog.desktop_control_profiles)
    if (
        any(not isinstance(row, RuntimeDesktopControlProfile) for row in controls)
        or len({row.profile_id for row in controls}) != len(controls)
        or len({row.artifact_sha256 for row in controls}) != len(controls)
    ):
        raise RuntimeCapabilityError(
            "runtime desktop control profiles are invalid"
        )
    all_executors = {
        **{row.executor_id: row for row in executors},
        **{row.executor_id: row for row in composites},
    }
    for control in controls:
        executor = all_executors.get(control.executor_id)
        if executor is None:
            raise RuntimeCapabilityError("desktop control executor is absent")
        if control.cuda_graph_mode != executor.adapter_parameters.get(
            "cuda_graph_mode", "default"
        ):
            raise RuntimeCapabilityError("desktop control CUDA graph mode differs")
        declared = executor.adapter_parameters.get("remote_resident_ffn_v1")
        if declared is not None:
            try:
                declared = RuntimeRemoteResidentFfn.from_json(json.loads(declared))
            except (TypeError, ValueError, RuntimePlanError) as error:
                raise RuntimeCapabilityError(
                    "desktop executor remote-resident declaration is invalid"
                ) from error
        if declared != control.remote_resident_ffn:
            raise RuntimeCapabilityError(
                "desktop control remote-resident group differs from its executor"
            )
        placement_devices = {
            row.primary_device_id for row in control.operator_placements
        }
        if any(
            device_id not in catalog.placement_profile.devices
            or catalog.placement_profile.devices[device_id].kind
                not in {"cpu", "gpu"}
            for device_id in placement_devices
        ):
            raise RuntimeCapabilityError(
                "desktop control contains a non-desktop device"
            )
        if isinstance(executor, RuntimeCompositeExecutorCapability):
            if (
                executor.route_family != "layer_placement"
                or executor.artifact_sha256 != control.artifact_sha256
                or executor.operator_placements != control.operator_placements
            ):
                raise RuntimeCapabilityError(
                    "desktop control differs from its composite executor"
                )
        elif placement_devices != {executor.device_id}:
            raise RuntimeCapabilityError(
                "desktop control differs from its device executor"
            )
    return controls


def _catalog_transitions(
    catalog: RuntimeCapabilityCatalog,
    executors: tuple[RuntimeExecutorCapability, ...],
    composites: tuple[RuntimeCompositeExecutorCapability, ...],
    resources: Mapping[str, ResourceProfile],
) -> tuple[RuntimeTransitionCapability, ...]:
    transitions = tuple(catalog.transitions)
    if any(not isinstance(row, RuntimeTransitionCapability) for row in transitions):
        raise RuntimeCapabilityError("runtime transitions are invalid")
    if len({row.transition_id for row in transitions}) != len(transitions):
        raise RuntimeCapabilityError("runtime transition ids are duplicated")
    device_ids = {row.device_id for row in executors}
    executor_ids = {
        *(row.executor_id for row in executors),
        *(row.executor_id for row in composites),
    }
    composite_by_id = {row.executor_id: row for row in composites}
    for transition in transitions:
        if transition.device_id not in device_ids:
            raise RuntimeCapabilityError("transition device is absent")
        if set(transition.prepares_device_ids) - device_ids:
            raise RuntimeCapabilityError("transition prepared device is absent")
        if set(transition.resource_ids) - set(resources):
            raise RuntimeCapabilityError("transition resource is absent")
        if (
            transition.executor_id is not None
            and transition.executor_id not in executor_ids
        ):
            raise RuntimeCapabilityError("transition executor is absent")
        coordinator = (
            None if transition.executor_id is None
            else composite_by_id.get(transition.executor_id)
        )
        if coordinator is not None and not set(
            transition.prepares_device_ids
        ).issubset(coordinator.participant_device_ids):
            raise RuntimeCapabilityError(
                "transition prepared devices differ from executor"
            )
        if any(
            slots > resources[resource_id].capacity
            for resource_id, slots in transition.resource_slots.items()
        ):
            raise RuntimeCapabilityError(
                "transition resource slots exceed capacity"
            )
    return transitions


def _catalog_cost_profiles(
    catalog: RuntimeCapabilityCatalog,
    executors: tuple[RuntimeExecutorCapability, ...],
    composites: tuple[RuntimeCompositeExecutorCapability, ...],
    resources: Mapping[str, ResourceProfile],
) -> tuple[
    tuple[RuntimeRouteShapeProfile, ...],
    tuple[RuntimeSystemCostProfile, ...],
]:
    route_profiles = tuple(catalog.route_shape_profiles)
    if any(not isinstance(row, RuntimeRouteShapeProfile) for row in route_profiles):
        raise RuntimeCapabilityError("runtime route profile is invalid")
    if len({row.selector_id for row in route_profiles}) != len(route_profiles):
        raise RuntimeCapabilityError("runtime route profile ids are duplicated")
    device_ids = {row.device_id for row in executors}
    executor_ids = {
        *(row.executor_id for row in executors),
        *(row.executor_id for row in composites),
    }
    for route_profile in route_profiles:
        if set(route_profile.device_ids) - device_ids:
            raise RuntimeCapabilityError(
                "runtime route profile device is absent"
            )
        if (
            route_profile.executor_id is not None
            and route_profile.executor_id not in executor_ids
        ):
            raise RuntimeCapabilityError(
                "runtime route profile executor is absent"
            )
    system_profiles = tuple(catalog.system_cost_profiles)
    if any(
        not isinstance(row, RuntimeSystemCostProfile) for row in system_profiles
    ):
        raise RuntimeCapabilityError("runtime system cost profile is invalid")
    if len({row.selector_id for row in system_profiles}) != len(system_profiles):
        raise RuntimeCapabilityError(
            "runtime system cost profile ids are duplicated"
        )
    for profile in system_profiles:
        if set(profile.interference_ppm_by_resource) - set(resources):
            raise RuntimeCapabilityError(
                "runtime system cost resource is absent"
            )
    return route_profiles, system_profiles


def _catalog_phone_power_profiles(
    catalog: RuntimeCapabilityCatalog,
    executors: tuple[RuntimeExecutorCapability, ...],
) -> tuple[RuntimePhonePowerProfile, ...]:
    profiles = tuple(catalog.phone_power_profiles)
    if (
        any(not isinstance(row, RuntimePhonePowerProfile) for row in profiles)
        or len({row.device_id for row in profiles}) != len(profiles)
    ):
        raise RuntimeCapabilityError(
            "runtime phone power profiles are invalid"
        )
    executor_by_device = {row.device_id: row for row in executors}
    for phone_power in profiles:
        device = catalog.placement_profile.devices.get(phone_power.device_id)
        domain = catalog.placement_profile.domains.get(phone_power.domain_id)
        if (
            device is None
            or not device.kind.startswith("phone")
            or domain is None
            or phone_power.device_id not in executor_by_device
        ):
            raise RuntimeCapabilityError(
                "runtime phone power profile differs from its device"
            )
        phone_kernels = tuple(
            row for row in catalog.placement_profile.kernels.values()
            if row.device_id == phone_power.device_id
        )
        if not phone_kernels or any(
            row.kernel.domain_id != phone_power.domain_id
            for row in phone_kernels
        ):
            raise RuntimeCapabilityError(
                "runtime phone power profile differs from its kernels"
            )
    return profiles


def _initialize_runtime_capability_catalog(
    catalog: RuntimeCapabilityCatalog,
) -> None:
    resources, executors = _catalog_base_values(catalog)
    composites = _catalog_composite_executors(catalog, executors, resources)
    controls = _catalog_desktop_controls(catalog, executors, composites)
    transitions = _catalog_transitions(
        catalog, executors, composites, resources
    )
    route_profiles, system_profiles = _catalog_cost_profiles(
        catalog, executors, composites, resources
    )
    phone_power_profiles = _catalog_phone_power_profiles(catalog, executors)
    saving = _integer(
        "catalog minimum energy saving", catalog.minimum_energy_saving_ppm
    )
    latency = _integer(
        "catalog maximum latency", catalog.maximum_latency_ppm, 1
    )
    if saving >= 1_000_000 or latency > 10_000_000:
        raise RuntimeCapabilityError("catalog policy bound is invalid")
    values = {
        "resources": MappingProxyType(dict(sorted(resources.items()))),
        "executors": tuple(sorted(executors, key=lambda row: row.executor_id)),
        "composite_executors": tuple(sorted(
            composites, key=lambda row: row.executor_id
        )),
        "desktop_control_profiles": tuple(sorted(
            controls, key=lambda row: row.profile_id
        )),
        "transitions": tuple(sorted(
            transitions, key=lambda row: row.transition_id
        )),
        "route_shape_profiles": tuple(sorted(
            route_profiles, key=lambda row: row.selector_id
        )),
        "system_cost_profiles": tuple(sorted(
            system_profiles, key=lambda row: row.selector_id
        )),
        "phone_power_profiles": tuple(sorted(
            phone_power_profiles, key=lambda row: row.device_id
        )),
    }
    for name, value in values.items():
        object.__setattr__(catalog, name, value)


@dataclass(frozen=True)
class RuntimeCapabilityCatalog:
    catalog_id: str
    placement_profile: PlacementHardwareProfile
    resources: Mapping[str, ResourceProfile]
    executors: tuple[RuntimeExecutorCapability, ...]
    transitions: tuple[RuntimeTransitionCapability, ...]
    minimum_energy_saving_ppm: int
    maximum_latency_ppm: int
    composite_executors: tuple[RuntimeCompositeExecutorCapability, ...] = ()
    desktop_control_profiles: tuple[RuntimeDesktopControlProfile, ...] = ()
    route_shape_profiles: tuple[RuntimeRouteShapeProfile, ...] = ()
    system_cost_profiles: tuple[RuntimeSystemCostProfile, ...] = ()
    phone_power_profiles: tuple[RuntimePhonePowerProfile, ...] = ()

    def __post_init__(self) -> None:
        _initialize_runtime_capability_catalog(self)

    @cached_property
    def executor_by_id(self) -> Mapping[str, RuntimeExecutorCapability]:
        return MappingProxyType({
            row.executor_id: row for row in self.executors
        })

    @cached_property
    def executor_by_device(self) -> Mapping[str, RuntimeExecutorCapability]:
        return MappingProxyType({
            row.device_id: row for row in self.executors
        })

    def residency_group(self, executor_id: str | None, device_id: str) -> str | None:
        """Resolve physical residency ownership, independently of shared compute."""
        coordinator = self.composite_executor_by_id.get(executor_id)
        if coordinator is not None:
            return coordinator.replacement_group_by_device.get(device_id)
        executor = self.executor_by_device.get(device_id)
        return None if executor is None else executor.exclusive_residency_resource_id

    @cached_property
    def exclusive_residency_resources(self) -> Mapping[str | tuple[str, str], str]:
        """Include exact session owners without treating associated devices as anchors."""
        resources = {
            row.device_id: row.exclusive_residency_resource_id
            for row in self.executors
            if row.exclusive_residency_resource_id is not None
        }
        for coordinator in self.composite_executors:
            for device_id, group in coordinator.replacement_group_by_device.items():
                if any(session.shared_compute_resource_id == group
                       for session in self.executor_by_device[device_id].phone_sessions):
                    resources[(device_id, coordinator.executor_id)] = group
        return MappingProxyType(resources)

    @cached_property
    def fallback(self) -> RuntimeExecutorCapability:
        return next(row for row in self.executors if row.qualified_fallback)

    @cached_property
    def phone_power_profile_by_device(
        self,
    ) -> Mapping[str, RuntimePhonePowerProfile]:
        return MappingProxyType({
            row.device_id: row for row in self.phone_power_profiles
        })

    @cached_property
    def composite_executor_by_id(
        self,
    ) -> Mapping[str, RuntimeCompositeExecutorCapability]:
        return MappingProxyType({
            row.executor_id: row for row in self.composite_executors
        })

    @cached_property
    def desktop_control_by_artifact(
        self,
    ) -> Mapping[str, RuntimeDesktopControlProfile]:
        return MappingProxyType({
            row.artifact_sha256: row
            for row in self.desktop_control_profiles
        })

    def route_profile_for(
        self,
        *,
        artifact_sha256: str,
        route_family: str,
        device_ids: tuple[str, ...],
        assisted_operator_kind: str | None,
        split_axis: str,
        split_fraction_ppm: int,
        residency_variant: str,
        input_tokens: int,
        output_tokens: int,
        cost_features: Mapping[str, int],
        executor_id: str | None = None,
    ) -> RuntimeRouteShapeProfile | None:
        matches = [
            row for row in self.route_shape_profiles
            if row.matches(
                artifact_sha256=artifact_sha256,
                route_family=route_family,
                device_ids=device_ids,
                assisted_operator_kind=assisted_operator_kind,
                split_axis=split_axis,
                split_fraction_ppm=split_fraction_ppm,
                residency_variant=residency_variant,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_features=cost_features,
                executor_id=executor_id,
            )
        ]
        return None if not matches else min(
            matches, key=lambda row: row.specificity
        )

    def system_cost_profile_for(
        self, cost_features: Mapping[str, int]
    ) -> RuntimeSystemCostProfile | None:
        matches = [
            row for row in self.system_cost_profiles
            if row.matches(cost_features)
        ]
        return None if not matches else min(
            matches, key=lambda row: row.specificity
        )

    def to_json(self) -> dict[str, object]:
        return {
            "catalog_id": self.catalog_id,
            "composite_executors": [
                row.to_json() for row in self.composite_executors
            ],
            "desktop_control_profiles": [
                row.to_json() for row in self.desktop_control_profiles
            ],
            "executors": [row.to_json() for row in self.executors],
            "maximum_latency_ppm": self.maximum_latency_ppm,
            "minimum_energy_saving_ppm": self.minimum_energy_saving_ppm,
            "placement_profile": _placement_profile_json(
                self.placement_profile
            ),
            "placement_profile_id": self.placement_profile.profile_id,
            "phone_power_profiles": [
                row.to_json() for row in self.phone_power_profiles
            ],
            "resources": [
                {
                    "capacity": row.capacity,
                    "identity": row.identity,
                    "kind": row.kind,
                    "ready": row.ready,
                    "resource_id": row.resource_id,
                }
                for row in self.resources.values()
            ],
            "route_shape_profiles": [
                row.to_json() for row in self.route_shape_profiles
            ],
            "schema": RUNTIME_CAPABILITY_SCHEMA,
            "system_cost_profiles": [
                row.to_json() for row in self.system_cost_profiles
            ],
            "transitions": [row.to_json() for row in self.transitions],
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeCapabilityCatalog":
        row = _object("runtime capability catalog", value)
        if row.get("schema") != RUNTIME_CAPABILITY_SCHEMA:
            raise RuntimeCapabilityError(
                "runtime capability catalog schema differs"
            )
        profile = PlacementHardwareProfile.from_json(
            row.get("placement_profile")
        )
        if row.get("placement_profile_id") != profile.profile_id:
            raise RuntimeCapabilityError(
                "runtime placement profile identity differs"
            )
        resources = tuple(
            ResourceProfile.from_json(value)
            for value in _list(
                "runtime capability resources", row.get("resources")
            )
        )
        return cls(
            catalog_id=row.get("catalog_id"),
            placement_profile=profile,
            resources={item.resource_id: item for item in resources},
            executors=tuple(
                RuntimeExecutorCapability.from_json(value)
                for value in _list(
                    "runtime capability executors", row.get("executors")
                )
            ),
            composite_executors=tuple(
                RuntimeCompositeExecutorCapability.from_json(value)
                for value in _list(
                    "runtime composite executors",
                    row.get("composite_executors", []),
                )
            ),
            desktop_control_profiles=tuple(
                RuntimeDesktopControlProfile.from_json(value)
                for value in _list(
                    "runtime desktop control profiles",
                    row.get("desktop_control_profiles", []),
                )
            ),
            transitions=tuple(
                RuntimeTransitionCapability.from_json(value)
                for value in _list(
                    "runtime capability transitions", row.get("transitions")
                )
            ),
            minimum_energy_saving_ppm=row.get("minimum_energy_saving_ppm"),
            maximum_latency_ppm=row.get("maximum_latency_ppm"),
            route_shape_profiles=tuple(
                RuntimeRouteShapeProfile.from_json(value)
                for value in _list(
                    "runtime route shape profiles",
                    row.get("route_shape_profiles", []),
                )
            ),
            system_cost_profiles=tuple(
                RuntimeSystemCostProfile.from_json(value)
                for value in _list(
                    "runtime system cost profiles",
                    row.get("system_cost_profiles", []),
                )
            ),
            phone_power_profiles=tuple(
                RuntimePhonePowerProfile.from_json(value)
                for value in _list(
                    "runtime phone power profiles",
                    row.get("phone_power_profiles", []),
                )
            ),
        )
