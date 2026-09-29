"""Device, executor, catalog and snapshot capability contracts: executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from ..runtime_system_cost import ROUTE_MATURITY_STATES
from .common import (
    COMPOSITE_ROUTE_FAMILIES,
    RESIDENCY_STATES,
    RuntimeCapabilityError,
    SPLIT_AXES,
    _MATURITY_RANK,
    _boolean,
    _fractions,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)
from .profiles import RuntimeKernelShapeProfile
from .phone import RuntimePhoneSessionCapability
from ..types import CANONICAL_OMIT_DEFAULT


_GENERATED_REQUEST_TRANSPORT_PARAMETERS = frozenset({
    "request_transport", "request_transport_allocator",
    "request_transport_capacity_d2h_profile_id", "request_transport_capacity_h2d_profile_id",
    "request_transport_concurrent_streams", "request_transport_d2h_profile_id",
    "request_transport_full_duplex", "request_transport_h2d_profile_id",
    "request_transport_identity_sha256", "request_transport_latency_accounting",
    "request_transport_max_payload_bytes", "request_transport_profile_id",
    "request_transport_queue_depth",
})
# Android PowerManager thermal status scale: NONE (0) .. SHUTDOWN (6).
ANDROID_THERMAL_STATUS_SHUTDOWN = 6

_GENERATED_PHONE_POWER_PARAMETERS = frozenset({
    "phone_power_active_mw", "phone_power_allow_assumed", "phone_power_device_ids",
    "phone_power_evidence_kind", "phone_power_estimation_version", "phone_power_idle_mw",
})


def whole_phone_launch_parameters(parameters: Mapping[str, int | str]) -> dict[str, int | str]:
    """Exclude generated transport/energy accounting, retaining launch configuration."""
    return {key: value for key, value in parameters.items()
            if key not in _GENERATED_REQUEST_TRANSPORT_PARAMETERS
            and key not in _GENERATED_PHONE_POWER_PARAMETERS}


def _executor_capability_values(
    capability: RuntimeExecutorCapability,
) -> dict[str, object]:
    for name in (
        "executor_id", "device_id", "endpoint", "backend", "memory_resource_id"
    ):
        _text(f"executor capability {name}", getattr(capability, name))
    resources = _texts(
        "executor execution resource", capability.execution_resource_ids
    )
    kernels = {
        _text("executor operator kind", kind): _text(
            "executor kernel profile", profile_id
        )
        for kind, profile_id in capability.kernel_profiles.items()
    }
    if not kernels:
        raise RuntimeCapabilityError("executor requires kernel profiles")
    quantizations = _texts(
        "executor quantization", capability.supported_quantizations
    )
    for name in (
        "supports_whole_model",
        "supports_layer_placement",
        "supports_operator_placement",
        "supports_kv_cache",
        "supports_split_coordinator",
        "supports_split_helper",
        "qualified_fallback",
    ):
        _boolean(f"executor {name}", getattr(capability, name))
    axes = _texts("executor split axis", capability.split_axes, allow_empty=True)
    if set(axes) - SPLIT_AXES:
        raise RuntimeCapabilityError("executor split axis is unsupported")
    split_fractions = _fractions(
        "executor split fractions", capability.split_fractions_ppm
    )
    layer_fractions = _fractions(
        "executor layer fractions", capability.layer_fractions_ppm
    )
    if axes and not split_fractions:
        raise RuntimeCapabilityError("split axes require split fractions")
    residency = _texts("executor residency state", capability.residency_states)
    if set(residency) - RESIDENCY_STATES:
        raise RuntimeCapabilityError("executor residency state is invalid")
    if capability.maturity not in ROUTE_MATURITY_STATES:
        raise RuntimeCapabilityError("executor maturity is invalid")
    evidence = _texts("executor evidence id", capability.evidence_ids)
    _integer(
        "executor maximum temperature",
        capability.maximum_temperature_millic,
        1,
    )
    if _integer(
        "executor maximum thermal status", capability.maximum_thermal_status
    ) > ANDROID_THERMAL_STATUS_SHUTDOWN:
        raise RuntimeCapabilityError(
            "executor maximum thermal status exceeds Android SHUTDOWN (6)"
        )
    battery = _integer(
        "executor minimum battery", capability.minimum_battery_ppm
    )
    if battery > 1_000_000:
        raise RuntimeCapabilityError("executor battery threshold exceeds one")
    _integer(
        "executor workspace bytes per token",
        capability.workspace_bytes_per_token,
    )
    coordinated = _texts(
        "executor coordinated route family",
        capability.coordinated_route_families,
        allow_empty=True,
    )
    if set(coordinated) - COMPOSITE_ROUTE_FAMILIES:
        raise RuntimeCapabilityError(
            "executor coordinated route family is unsupported"
        )
    if capability.operator_plan_protocol is not None:
        _text(
            "executor operator plan protocol",
            capability.operator_plan_protocol,
        )
    if bool(coordinated) != (capability.operator_plan_protocol is not None):
        raise RuntimeCapabilityError(
            "coordinated routes require one operator plan protocol"
        )
    if capability.exclusive_residency_resource_id is not None:
        _text(
            "executor exclusive residency resource",
            capability.exclusive_residency_resource_id,
        )
    return {
        "resources": resources,
        "kernels": kernels,
        "quantizations": quantizations,
        "axes": axes,
        "split_fractions": split_fractions,
        "layer_fractions": layer_fractions,
        "residency": residency,
        "evidence": evidence,
        "coordinated": coordinated,
    }


def _executor_adapter_and_shapes(
    capability: RuntimeExecutorCapability,
    kernels: Mapping[str, str],
) -> tuple[dict[str, int | str], tuple[RuntimeKernelShapeProfile, ...]]:
    adapter_parameters: dict[str, int | str] = {}
    for raw_name, raw_value in capability.adapter_parameters.items():
        name = _text("executor adapter parameter", raw_name)
        if type(raw_value) is int:
            if raw_value < 0:
                raise RuntimeCapabilityError(
                    "executor adapter integer parameter is negative"
                )
            adapter_parameters[name] = raw_value
        elif type(raw_value) is str:
            adapter_parameters[name] = _text(
                "executor adapter parameter value", raw_value
            )
        else:
            raise RuntimeCapabilityError(
                "executor adapter parameter value is invalid"
            )
    shape_profiles = tuple(capability.kernel_shape_profiles)
    if any(not isinstance(row, RuntimeKernelShapeProfile) for row in shape_profiles):
        raise RuntimeCapabilityError(
            "executor kernel shape profile is invalid"
        )
    if len({row.selector_id for row in shape_profiles}) != len(shape_profiles):
        raise RuntimeCapabilityError(
            "executor kernel shape selector ids are duplicated"
        )
    if any(row.operator_kind not in kernels for row in shape_profiles):
        raise RuntimeCapabilityError(
            "kernel shape profile references an unsupported operator"
        )
    return adapter_parameters, shape_profiles


def _executor_phone_sessions(
    capability: RuntimeExecutorCapability,
    resources: Sequence[str],
) -> tuple[RuntimePhoneSessionCapability, ...]:
    sessions = tuple(capability.phone_sessions)
    if any(
        not isinstance(row, RuntimePhoneSessionCapability)
        or row.device_id != capability.device_id
        for row in sessions
    ):
        raise RuntimeCapabilityError(
            "executor phone session capability is invalid"
        )
    if (
        len({row.session_id for row in sessions}) != len(sessions)
        or len({row.endpoint for row in sessions}) != len(sessions)
        or len({row.memory_resource_id for row in sessions}) != len(sessions)
    ):
        raise RuntimeCapabilityError(
            "executor phone session identities are duplicated"
        )
    if sessions and not capability.supports_split_helper:
        raise RuntimeCapabilityError(
            "phone sessions require a split-helper executor"
        )
    if sessions and (
        len({row.shared_compute_resource_id for row in sessions}) != 1
        or len({row.shared_transport_resource_ids for row in sessions}) != 1
    ):
        raise RuntimeCapabilityError(
            "phone residency sessions must share execution resources"
        )
    if sessions:
        session_resources = {
            sessions[0].shared_compute_resource_id,
            *sessions[0].shared_transport_resource_ids,
        }
        if not session_resources.issubset(resources):
            raise RuntimeCapabilityError(
                "phone session resources are absent from the executor"
            )
    return sessions


def _initialize_executor_capability(
    capability: RuntimeExecutorCapability,
) -> None:
    values = _executor_capability_values(capability)
    adapter_parameters, shape_profiles = _executor_adapter_and_shapes(
        capability, values["kernels"]
    )
    phone_sessions = _executor_phone_sessions(
        capability, values["resources"]
    )
    normalized = {
        "execution_resource_ids": values["resources"],
        "kernel_profiles": MappingProxyType(dict(sorted(
            values["kernels"].items()
        ))),
        "supported_quantizations": values["quantizations"],
        "split_axes": tuple(sorted(values["axes"])),
        "split_fractions_ppm": values["split_fractions"],
        "layer_fractions_ppm": values["layer_fractions"],
        "residency_states": tuple(sorted(values["residency"])),
        "evidence_ids": values["evidence"],
        "coordinated_route_families": tuple(sorted(values["coordinated"])),
        "kernel_shape_profiles": tuple(sorted(
            shape_profiles, key=lambda row: row.selector_id
        )),
        "adapter_parameters": MappingProxyType(dict(sorted(
            adapter_parameters.items()
        ))),
        "phone_sessions": tuple(sorted(
            phone_sessions, key=lambda row: row.session_id
        )),
    }
    for name, value in normalized.items():
        object.__setattr__(capability, name, value)


def _executor_overlay_structures(
    base: RuntimeExecutorCapability,
    overlay: RuntimeExecutorCapability,
) -> tuple[
    str | None,
    str | None,
    tuple[RuntimeKernelShapeProfile, ...],
    tuple[RuntimePhoneSessionCapability, ...],
    str,
]:
    if not isinstance(overlay, RuntimeExecutorCapability):
        raise RuntimeCapabilityError("executor overlay is invalid")
    for name in ("executor_id", "device_id", "memory_resource_id"):
        if getattr(base, name) != getattr(overlay, name):
            raise RuntimeCapabilityError(
                "executor overlay identity differs: " + name
            )
    exclusive = base.exclusive_residency_resource_id
    if exclusive is None:
        exclusive = overlay.exclusive_residency_resource_id
    elif (
        overlay.exclusive_residency_resource_id is not None
        and overlay.exclusive_residency_resource_id != exclusive
    ):
        raise RuntimeCapabilityError(
            "executor overlay exclusive residency resource differs"
        )
    protocol = base.operator_plan_protocol
    if protocol is None:
        protocol = overlay.operator_plan_protocol
    elif (
        overlay.operator_plan_protocol is not None
        and overlay.operator_plan_protocol != protocol
    ):
        raise RuntimeCapabilityError(
            "executor overlay operator protocol differs"
        )
    shape_profiles = {
        row.selector_id: row for row in base.kernel_shape_profiles
    }
    for row in overlay.kernel_shape_profiles:
        current = shape_profiles.get(row.selector_id)
        if current is not None and current != row:
            raise RuntimeCapabilityError(
                "executor overlay kernel shape profile differs: "
                + row.selector_id
            )
        shape_profiles[row.selector_id] = row
    phone_sessions = {row.session_id: row for row in base.phone_sessions}
    for row in overlay.phone_sessions:
        current = phone_sessions.get(row.session_id)
        if current is not None and (
            current.device_id != row.device_id
            or current.endpoint != row.endpoint
            or current.worker_identity_sha256 != row.worker_identity_sha256
            or current.memory_resource_id != row.memory_resource_id
            or current.resident_memory_limit_bytes
                != row.resident_memory_limit_bytes
            or current.shared_compute_resource_id
                != row.shared_compute_resource_id
            or current.shared_transport_resource_ids
                != row.shared_transport_resource_ids
            or current.supported_layer_mask != row.supported_layer_mask
            or current.maximum_columns != row.maximum_columns
            or current.column_quantum != row.column_quantum
            or current.supported_data_types != row.supported_data_types
            or current.batch_plans != row.batch_plans
        ):
            raise RuntimeCapabilityError(
                "executor overlay phone session structure differs: "
                + row.session_id
            )
        phone_sessions[row.session_id] = row
    maturity = min(
        (base.maturity, overlay.maturity),
        key=lambda value: _MATURITY_RANK[value],
    )
    return (
        exclusive,
        protocol,
        tuple(shape_profiles.values()),
        tuple(phone_sessions.values()),
        maturity,
    )


@dataclass(frozen=True)
class RuntimeExecutorCapability:
    executor_id: str
    device_id: str
    endpoint: str
    backend: str
    execution_resource_ids: tuple[str, ...]
    memory_resource_id: str
    kernel_profiles: Mapping[str, str]
    supported_quantizations: tuple[str, ...]
    supports_whole_model: bool
    supports_layer_placement: bool
    supports_operator_placement: bool
    supports_kv_cache: bool
    supports_split_coordinator: bool
    supports_split_helper: bool
    split_axes: tuple[str, ...]
    split_fractions_ppm: tuple[int, ...]
    layer_fractions_ppm: tuple[int, ...]
    residency_states: tuple[str, ...]
    maturity: str
    evidence_ids: tuple[str, ...]
    qualified_fallback: bool
    maximum_temperature_millic: int
    minimum_battery_ppm: int
    workspace_bytes_per_token: int
    coordinated_route_families: tuple[str, ...] = ()
    operator_plan_protocol: str | None = None
    kernel_shape_profiles: tuple[RuntimeKernelShapeProfile, ...] = ()
    exclusive_residency_resource_id: str | None = None
    adapter_parameters: Mapping[str, int | str] = field(default_factory=dict)
    phone_sessions: tuple[RuntimePhoneSessionCapability, ...] = ()
    # opt-in per-device policy: the device stays thermally qualified while its Android
    # thermal status is at most this value; 0 keeps the platform rule (status == 0)
    maximum_thermal_status: int = field(
        default=0, metadata={CANONICAL_OMIT_DEFAULT: True}
    )

    def __post_init__(self) -> None:
        _initialize_executor_capability(self)

    @property
    def operator_kinds(self) -> frozenset[str]:
        return frozenset(self.kernel_profiles)

    def supports_quantization(self, quantization: str) -> bool:
        return "*" in self.supported_quantizations or quantization in (
            self.supported_quantizations
        )

    def kernel_profile_for(
        self,
        operator_kind: str,
        *,
        input_tokens: int,
        output_tokens: int,
        compute_ops: int,
        memory_bytes: int,
    ) -> tuple[str, RuntimeKernelShapeProfile | None]:
        matches = [
            row for row in self.kernel_shape_profiles
            if row.operator_kind == operator_kind
            and row.matches(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                compute_ops=compute_ops,
                memory_bytes=memory_bytes,
            )
        ]
        if not matches:
            return self.kernel_profiles[operator_kind], None
        selected = min(matches, key=lambda row: row.specificity)
        return selected.profile_id, selected

    def with_runtime_overlay(
        self, overlay: "RuntimeExecutorCapability"
    ) -> "RuntimeExecutorCapability":
        """Merge runtime transport facts without dropping structure."""

        (
            exclusive,
            protocol,
            shape_profiles,
            phone_sessions,
            maturity,
        ) = _executor_overlay_structures(self, overlay)
        return RuntimeExecutorCapability(
            executor_id=self.executor_id,
            device_id=self.device_id,
            endpoint=overlay.endpoint,
            backend=overlay.backend,
            execution_resource_ids=tuple(sorted(set(
                self.execution_resource_ids
                + overlay.execution_resource_ids
            ))),
            memory_resource_id=self.memory_resource_id,
            kernel_profiles={
                **self.kernel_profiles,
                **overlay.kernel_profiles,
            },
            supported_quantizations=tuple(sorted(set(
                self.supported_quantizations
                + overlay.supported_quantizations
            ))),
            supports_whole_model=(
                self.supports_whole_model
                or overlay.supports_whole_model
            ),
            supports_layer_placement=(
                self.supports_layer_placement
                or overlay.supports_layer_placement
            ),
            supports_operator_placement=(
                self.supports_operator_placement
                or overlay.supports_operator_placement
            ),
            supports_kv_cache=(
                self.supports_kv_cache or overlay.supports_kv_cache
            ),
            supports_split_coordinator=(
                self.supports_split_coordinator
                or overlay.supports_split_coordinator
            ),
            supports_split_helper=(
                self.supports_split_helper
                or overlay.supports_split_helper
            ),
            split_axes=tuple(sorted(set(
                self.split_axes + overlay.split_axes
            ))),
            split_fractions_ppm=tuple(sorted(set(
                self.split_fractions_ppm
                + overlay.split_fractions_ppm
            ))),
            layer_fractions_ppm=tuple(sorted(set(
                self.layer_fractions_ppm
                + overlay.layer_fractions_ppm
            ))),
            residency_states=tuple(sorted(set(
                self.residency_states + overlay.residency_states
            ))),
            maturity=maturity,
            evidence_ids=tuple(sorted(set(
                self.evidence_ids + overlay.evidence_ids
            ))),
            qualified_fallback=(
                self.qualified_fallback or overlay.qualified_fallback
            ),
            maximum_temperature_millic=min(
                self.maximum_temperature_millic,
                overlay.maximum_temperature_millic,
            ),
            minimum_battery_ppm=max(
                self.minimum_battery_ppm,
                overlay.minimum_battery_ppm,
            ),
            workspace_bytes_per_token=max(
                self.workspace_bytes_per_token,
                overlay.workspace_bytes_per_token,
            ),
            coordinated_route_families=tuple(sorted(set(
                self.coordinated_route_families
                + overlay.coordinated_route_families
            ))),
            operator_plan_protocol=protocol,
            kernel_shape_profiles=shape_profiles,
            exclusive_residency_resource_id=exclusive,
            adapter_parameters={
                **self.adapter_parameters,
                **overlay.adapter_parameters,
            },
            phone_sessions=phone_sessions,
            maximum_thermal_status=min(
                self.maximum_thermal_status,
                overlay.maximum_thermal_status,
            ),
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "backend": self.backend,
            "device_id": self.device_id,
            "endpoint": self.endpoint,
            "evidence_ids": list(self.evidence_ids),
            "execution_resource_ids": list(self.execution_resource_ids),
            "executor_id": self.executor_id,
            "kernel_profiles": dict(self.kernel_profiles),
            "kernel_shape_profiles": [
                row.to_json() for row in self.kernel_shape_profiles
            ],
            "layer_fractions_ppm": list(self.layer_fractions_ppm),
            "maturity": self.maturity,
            "maximum_temperature_millic": self.maximum_temperature_millic,
            "memory_resource_id": self.memory_resource_id,
            "minimum_battery_ppm": self.minimum_battery_ppm,
            "coordinated_route_families": list(
                self.coordinated_route_families
            ),
            "operator_plan_protocol": self.operator_plan_protocol,
            "qualified_fallback": self.qualified_fallback,
            "residency_states": list(self.residency_states),
            "split_axes": list(self.split_axes),
            "split_fractions_ppm": list(self.split_fractions_ppm),
            "supported_quantizations": list(self.supported_quantizations),
            "supports_kv_cache": self.supports_kv_cache,
            "supports_split_coordinator": self.supports_split_coordinator,
            "supports_split_helper": self.supports_split_helper,
            "supports_layer_placement": self.supports_layer_placement,
            "supports_operator_placement": self.supports_operator_placement,
            "supports_whole_model": self.supports_whole_model,
            "workspace_bytes_per_token": self.workspace_bytes_per_token,
        }
        if self.exclusive_residency_resource_id is not None:
            result["exclusive_residency_resource_id"] = (
                self.exclusive_residency_resource_id
            )
        if self.adapter_parameters:
            result["adapter_parameters"] = dict(self.adapter_parameters)
        if self.phone_sessions:
            result["phone_sessions"] = [
                row.to_json() for row in self.phone_sessions
            ]
        if self.maximum_thermal_status:
            result["maximum_thermal_status"] = self.maximum_thermal_status
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeExecutorCapability":
        row = _object("runtime executor capability", value)
        kernels = _object(
            "runtime executor kernel profiles", row.get("kernel_profiles")
        )
        return cls(
            executor_id=row.get("executor_id"),
            device_id=row.get("device_id"),
            endpoint=row.get("endpoint"),
            backend=row.get("backend"),
            execution_resource_ids=tuple(_list(
                "runtime executor resources",
                row.get("execution_resource_ids"),
            )),
            memory_resource_id=row.get("memory_resource_id"),
            kernel_profiles=dict(kernels),
            supported_quantizations=tuple(_list(
                "runtime executor quantizations",
                row.get("supported_quantizations"),
            )),
            supports_whole_model=row.get("supports_whole_model"),
            supports_layer_placement=row.get("supports_layer_placement"),
            supports_operator_placement=row.get("supports_operator_placement"),
            supports_kv_cache=row.get("supports_kv_cache"),
            supports_split_coordinator=row.get("supports_split_coordinator"),
            supports_split_helper=row.get("supports_split_helper"),
            split_axes=tuple(_list(
                "runtime executor split axes", row.get("split_axes")
            )),
            split_fractions_ppm=tuple(_list(
                "runtime executor split fractions",
                row.get("split_fractions_ppm"),
            )),
            layer_fractions_ppm=tuple(_list(
                "runtime executor layer fractions",
                row.get("layer_fractions_ppm"),
            )),
            residency_states=tuple(_list(
                "runtime executor residency states",
                row.get("residency_states"),
            )),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime executor evidence", row.get("evidence_ids")
            )),
            qualified_fallback=row.get("qualified_fallback"),
            maximum_temperature_millic=row.get(
                "maximum_temperature_millic"
            ),
            minimum_battery_ppm=row.get("minimum_battery_ppm"),
            workspace_bytes_per_token=row.get("workspace_bytes_per_token"),
            coordinated_route_families=tuple(_list(
                "runtime executor coordinated route families",
                row.get("coordinated_route_families", []),
            )),
            operator_plan_protocol=row.get("operator_plan_protocol"),
            kernel_shape_profiles=tuple(
                RuntimeKernelShapeProfile.from_json(value)
                for value in _list(
                    "runtime executor kernel shape profiles",
                    row.get("kernel_shape_profiles", []),
                )
            ),
            exclusive_residency_resource_id=row.get(
                "exclusive_residency_resource_id"
            ),
            adapter_parameters={
                key: value
                for key, value in _object(
                    "runtime executor adapter parameters",
                    row.get("adapter_parameters", {}),
                ).items()
            },
            phone_sessions=tuple(
                RuntimePhoneSessionCapability.from_json(item)
                for item in _list(
                    "runtime executor phone sessions",
                    row.get("phone_sessions", []),
                )
            ),
            maximum_thermal_status=row.get("maximum_thermal_status", 0),
        )
