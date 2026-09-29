"""Capability-driven construction of executable runtime catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.model_manifest import ModelManifest
from .._internal.placement import PlacementHardwareProfile, TransferLink
from .._internal.plan_contracts.co_helpers import (
    PHONE_CO_HELPERS_PARAMETER,
    RuntimeCoHelperDeclaration,
    co_helper_declaration,
)
from .._internal.policy import ResourceProfile
from .._internal.runtime_capabilities import (
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeCompositeOperatorPlacement,
    RuntimeDesktopControlProfile,
    RuntimeExecutorCapability,
    RuntimePhoneSessionCapability,
    RuntimePhonePowerProfile,
    RuntimeRouteShapeProfile,
    RuntimeSystemCostProfile,
    RuntimeTransitionCapability,
)


class CatalogMaterializationError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise CatalogMaterializationError(name + " is invalid")
    return value


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise CatalogMaterializationError(name + " is invalid")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value)
    if (
        not result.startswith("sha256:")
        or len(result) != 71
        or any(character not in "0123456789abcdef" for character in result[7:])
    ):
        raise CatalogMaterializationError(name + " must be SHA-256")
    return result


def _evidence(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted({_sha256(name, value) for value in values}))
    if not result:
        raise CatalogMaterializationError(name + " is absent")
    return result


@dataclass(frozen=True)
class MeasuredOperatorAssistProfile:
    route_family: str
    operator_kind: str
    split_axis: str
    split_fraction_ppm: int

    def __post_init__(self) -> None:
        if self.route_family not in {"operator_offload", "operator_split"}:
            raise CatalogMaterializationError(
                "measured operator route family is invalid"
            )
        if self.operator_kind != "ffn":
            raise CatalogMaterializationError(
                "measured operator kind is unsupported"
            )
        if self.route_family == "operator_offload":
            valid = self.split_axis == "none" and (
                self.split_fraction_ppm == 0
            )
        else:
            valid = self.split_axis == "column" and (
                type(self.split_fraction_ppm) is int
                and 0 < self.split_fraction_ppm < 1_000_000
            )
        if not valid:
            raise CatalogMaterializationError(
                "measured operator split is invalid"
            )


def measured_ffn_assist_profile(
    manifest: ModelManifest,
    observations: Sequence[Mapping[str, object]],
) -> MeasuredOperatorAssistProfile:
    """Normalize exact physical FFN phase observations for route fitting."""
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("measured FFN manifest is invalid")
    profiles = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise CatalogMaterializationError(
                "measured FFN observation is invalid"
            )
        if observation.get("n_embd") != manifest.embedding_length:
            raise CatalogMaterializationError(
                "measured FFN model geometry differs"
            )
        columns = observation.get("phone_columns")
        if (
            type(columns) is not int
            or not 0 < columns <= manifest.feed_forward_length
        ):
            raise CatalogMaterializationError(
                "measured FFN columns are invalid"
            )
        mode = observation.get("execution_mode")
        if mode == "full_replacement":
            if columns != manifest.feed_forward_length:
                raise CatalogMaterializationError(
                    "measured FFN replacement is incomplete"
                )
            profile = MeasuredOperatorAssistProfile(
                route_family="operator_offload",
                operator_kind="ffn",
                split_axis="none",
                split_fraction_ppm=0,
            )
        elif mode == "parallel_split":
            numerator = columns * 1_000_000
            if (
                columns >= manifest.feed_forward_length
                or numerator % manifest.feed_forward_length
            ):
                raise CatalogMaterializationError(
                    "measured FFN split fraction is not exact"
                )
            profile = MeasuredOperatorAssistProfile(
                route_family="operator_split",
                operator_kind="ffn",
                split_axis="column",
                split_fraction_ppm=(
                    numerator // manifest.feed_forward_length
                ),
            )
        else:
            raise CatalogMaterializationError(
                "measured FFN execution mode is unsupported"
            )
        profiles.append(profile)
    if not profiles or len(set(profiles)) != 1:
        raise CatalogMaterializationError(
            "measured FFN placement differs across observations"
        )
    return profiles[0]


@dataclass(frozen=True)
class RuntimeHelperPhoneTopology:
    """Resources of one static FFN co-helper phone (rig ``topology.helper_phones``)."""

    device_id: str
    memory_resource_id: str
    transport_resource_ids: tuple[str, ...]
    compute_resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("runtime helper phone device", self.device_id)
        _text("runtime helper phone memory", self.memory_resource_id)
        transport = tuple(dict.fromkeys(
            _text("runtime helper phone transport", value)
            for value in self.transport_resource_ids
        ))
        compute = tuple(dict.fromkeys(
            _text("runtime helper phone compute", value)
            for value in self.compute_resource_ids
        ))
        if not transport or not compute or set(transport) & set(compute):
            raise CatalogMaterializationError(
                "runtime helper phone resources are invalid"
            )
        object.__setattr__(self, "transport_resource_ids", transport)
        object.__setattr__(self, "compute_resource_ids", compute)


@dataclass(frozen=True)
class RuntimePhysicalTopology:
    cpu_device_id: str
    gpu_device_id: str
    phone_device_id: str
    cpu_resource_id: str
    gpu_resource_id: str
    host_memory_resource_id: str
    gpu_memory_resource_id: str
    phone_memory_resource_id: str
    functionfs_resource_id: str
    phone_transport_resource_ids: tuple[str, ...]
    phone_compute_resource_ids: tuple[str, ...]
    gpu_exclusive_residency_resource_id: str
    resource_capacities: Mapping[str, int]
    resource_identities: Mapping[str, str]
    phone_sessions: tuple[RuntimePhoneSessionCapability, ...] = ()
    phone_exclusive_residency_resource_id: str | None = None
    helper_phones: tuple[RuntimeHelperPhoneTopology, ...] = ()

    def __post_init__(self) -> None:
        names = (
            self.cpu_device_id,
            self.gpu_device_id,
            self.phone_device_id,
            self.cpu_resource_id,
            self.gpu_resource_id,
            self.host_memory_resource_id,
            self.gpu_memory_resource_id,
            self.phone_memory_resource_id,
            self.functionfs_resource_id,
            self.gpu_exclusive_residency_resource_id,
        )
        if any(not _text("runtime topology identity", value) for value in names):
            raise CatalogMaterializationError("runtime topology is invalid")
        if len({self.cpu_device_id, self.gpu_device_id,
                self.phone_device_id}) != 3:
            raise CatalogMaterializationError(
                "runtime topology devices must differ"
            )
        transport = tuple(sorted({
            _text("runtime phone transport resource", value)
            for value in self.phone_transport_resource_ids
        }))
        compute = tuple(sorted({
            _text("runtime phone compute resource", value)
            for value in self.phone_compute_resource_ids
        }))
        if not transport or not compute or set(transport) & set(compute):
            raise CatalogMaterializationError(
                "runtime phone resources are invalid"
            )
        if self.functionfs_resource_id not in transport:
            raise CatalogMaterializationError(
                "runtime FunctionFS resource is not a phone transport"
            )
        phone_residency = self.phone_exclusive_residency_resource_id
        if phone_residency is not None and (
            not _text("runtime phone residency resource", phone_residency)
            or phone_residency not in compute
        ):
            raise CatalogMaterializationError(
                "runtime phone residency resource is not shared compute"
            )
        capacities = {
            _text("runtime resource capacity id", key): _positive_integer(
                "runtime resource capacity", value
            )
            for key, value in self.resource_capacities.items()
        }
        required = {
            self.cpu_resource_id,
            self.gpu_resource_id,
            self.gpu_exclusive_residency_resource_id,
            *transport,
            *compute,
        }
        if not required.issubset(capacities):
            raise CatalogMaterializationError(
                "runtime resource capacities are incomplete"
            )
        identities = {
            _text("runtime resource identity id", key): _text(
                "runtime resource identity", value
            )
            for key, value in self.resource_identities.items()
        }
        if set(identities) - set(capacities):
            raise CatalogMaterializationError(
                "runtime resource identity is unknown"
            )
        sessions = tuple(self.phone_sessions)
        if any(
            not isinstance(row, RuntimePhoneSessionCapability)
            or row.device_id != self.phone_device_id
            or row.shared_compute_resource_id not in compute
            or set(row.shared_transport_resource_ids) - set(transport)
            for row in sessions
        ) or len({row.session_id for row in sessions}) != len(sessions):
            raise CatalogMaterializationError(
                "runtime phone sessions are invalid"
            )
        object.__setattr__(self, "phone_transport_resource_ids", transport)
        object.__setattr__(self, "phone_compute_resource_ids", compute)
        object.__setattr__(
            self, "resource_capacities",
            MappingProxyType(dict(sorted(capacities.items()))),
        )
        object.__setattr__(
            self, "resource_identities",
            MappingProxyType(dict(sorted(identities.items()))),
        )
        object.__setattr__(
            self,
            "phone_sessions",
            tuple(sorted(sessions, key=lambda row: row.session_id)),
        )
        helpers = tuple(self.helper_phones)
        seen_compute = set(compute)
        for row in helpers:
            if (
                not isinstance(row, RuntimeHelperPhoneTopology)
                or row.device_id in {
                    self.cpu_device_id, self.gpu_device_id, self.phone_device_id
                }
                or row.memory_resource_id == self.phone_memory_resource_id
                or set(row.compute_resource_ids) & seen_compute
                or not {
                    *row.transport_resource_ids, *row.compute_resource_ids
                }.issubset(capacities)
            ):
                raise CatalogMaterializationError(
                    "runtime helper phone topology is invalid"
                )
            seen_compute.update(row.compute_resource_ids)
        if len({row.device_id for row in helpers}) != len(helpers):
            raise CatalogMaterializationError(
                "runtime helper phones are duplicated"
            )
        object.__setattr__(self, "helper_phones", helpers)

    def helper_phone(self, device_id: str) -> RuntimeHelperPhoneTopology:
        for row in self.helper_phones:
            if row.device_id == device_id:
                return row
        raise CatalogMaterializationError(
            "runtime helper phone is absent: " + device_id
        )

    @property
    def desktop_device_ids(self) -> tuple[str, str]:
        return (self.cpu_device_id, self.gpu_device_id)

    @property
    def all_device_ids(self) -> tuple[str, str, str]:
        return self.desktop_device_ids + (self.phone_device_id,)


@dataclass(frozen=True)
class RuntimeModelEndpointCapability:
    manifest: ModelManifest
    desktop_executor_id: str
    desktop_endpoint: str
    desktop_backend: str
    phone_executor_prefix: str
    phone_endpoint: str
    phone_backend: str
    desktop_gpu_first_layer: int
    adapter_parameters: Mapping[str, int | str]
    desktop_evidence_ids: tuple[str, ...]
    phone_evidence_ids: tuple[str, ...]
    phone_preloaded: bool
    phone_resident_limit_bytes: int
    ffn_column_quantum: int = 512
    ffn_max_runtime_partitions: int = 12
    operator_plan_protocol: str = "s41-static-ffn-plan-v1"
    phone_adapter_parameters: Mapping[str, int | str] = field(
        default_factory=dict
    )
    phone_runtime_control_protocol: str | None = None
    cpu_executor_id: str | None = None
    cpu_endpoint: str | None = None
    cpu_backend: str | None = None
    cpu_phone_executor_prefix: str | None = None
    cpu_phone_endpoint: str | None = None
    cpu_phone_backend: str | None = None
    phone_batch_plans: tuple[str, ...] = ("split-row",)
    qualified_phone_batch_plans: tuple[str, ...] = ("split-row",)
    cpu_adapter_parameters: Mapping[str, int | str] | None = None
    cpu_evidence_ids: tuple[str, ...] | None = None
    # static FFN co-helper phones of every phone-assisted family (Stage A two-phone dispatch)
    co_helpers: RuntimeCoHelperDeclaration | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ModelManifest):
            raise CatalogMaterializationError(
                "runtime endpoint manifest is invalid"
            )
        for name, value in (
            ("desktop executor", self.desktop_executor_id),
            ("desktop endpoint", self.desktop_endpoint),
            ("desktop backend", self.desktop_backend),
            ("phone executor prefix", self.phone_executor_prefix),
            ("phone endpoint", self.phone_endpoint),
            ("phone backend", self.phone_backend),
            ("operator plan protocol", self.operator_plan_protocol),
        ):
            _text("runtime " + name, value)
        if (
            type(self.desktop_gpu_first_layer) is not int
            or not 0 <= self.desktop_gpu_first_layer <= self.manifest.block_count
        ):
            raise CatalogMaterializationError(
                "runtime desktop layer boundary is invalid"
            )
        if type(self.phone_preloaded) is not bool:
            raise CatalogMaterializationError(
                "runtime phone residency state is invalid"
            )
        batch_plans = tuple(self.phone_batch_plans)
        qualified_batch_plans = tuple(self.qualified_phone_batch_plans)
        allowed_batch_plans = {"coalesced-batch", "split-row"}
        if (
            not batch_plans
            or len(batch_plans) != len(set(batch_plans))
            or set(batch_plans) - allowed_batch_plans
            or set(qualified_batch_plans) - set(batch_plans)
        ):
            raise CatalogMaterializationError(
                "runtime phone batch plans are invalid"
            )
        _positive_integer(
            "runtime phone resident limit", self.phone_resident_limit_bytes
        )
        _positive_integer("runtime FFN column quantum", self.ffn_column_quantum)
        _positive_integer(
            "runtime FFN maximum partitions",
            self.ffn_max_runtime_partitions,
        )
        parameters = self._validated_adapter_parameters(
            self.adapter_parameters
        )
        phone_parameters = self._validated_adapter_parameters(
            self.phone_adapter_parameters
        )
        if (
            self.phone_runtime_control_protocol is not None
            and self.phone_runtime_control_protocol != "decode-boundary-v1"
        ):
            raise CatalogMaterializationError(
                "runtime phone control protocol is unsupported"
            )
        if self.co_helpers is not None and (
            not isinstance(self.co_helpers, RuntimeCoHelperDeclaration)
            or self.phone_runtime_control_protocol != "decode-boundary-v1"
            # the server splits the FFN of CPU-resident layers only
            or self.co_helpers.layer_mask >> self.desktop_gpu_first_layer
            or self.co_helpers.layer_mask >> self.manifest.block_count
        ):
            raise CatalogMaterializationError(
                "runtime co-helper phones need decode control on CPU-parent layers"
            )
        cpu_values = (
            self.cpu_executor_id,
            self.cpu_endpoint,
            self.cpu_backend,
            self.cpu_phone_executor_prefix,
            self.cpu_phone_endpoint,
            self.cpu_phone_backend,
        )
        if any(value is not None for value in cpu_values):
            if any(value is None for value in cpu_values):
                raise CatalogMaterializationError(
                    "runtime CPU phone endpoint contract is incomplete"
                )
            for value in cpu_values:
                _text("runtime CPU phone endpoint", value)
        if self.cpu_adapter_parameters is not None or self.cpu_evidence_ids is not None:
            if self.cpu_executor_id is None:
                raise CatalogMaterializationError("runtime CPU parent is absent")
        if self.cpu_adapter_parameters is not None:
            object.__setattr__(self, "cpu_adapter_parameters",
                               self._validated_adapter_parameters(self.cpu_adapter_parameters))
        if self.cpu_evidence_ids is not None:
            object.__setattr__(self, "cpu_evidence_ids",
                               _evidence("runtime CPU evidence", self.cpu_evidence_ids))
        object.__setattr__(self, "adapter_parameters", parameters)
        object.__setattr__(
            self, "phone_adapter_parameters", phone_parameters
        )
        object.__setattr__(self, "phone_batch_plans", batch_plans)
        object.__setattr__(
            self,
            "qualified_phone_batch_plans",
            qualified_batch_plans,
        )
        object.__setattr__(
            self, "desktop_evidence_ids",
            _evidence("runtime desktop evidence", self.desktop_evidence_ids),
        )
        object.__setattr__(
            self, "phone_evidence_ids",
            _evidence("runtime phone evidence", self.phone_evidence_ids),
        )

    @staticmethod
    def _validated_adapter_parameters(
        values: Mapping[str, int | str],
    ) -> Mapping[str, int | str]:
        if not isinstance(values, Mapping):
            raise CatalogMaterializationError(
                "runtime adapter parameters are invalid"
            )
        parameters = {}
        for key, value in values.items():
            key = _text("runtime adapter parameter", key)
            if type(value) is int:
                if value < 0:
                    raise CatalogMaterializationError(
                        "runtime adapter integer is negative"
                    )
            elif type(value) is str:
                _text("runtime adapter value", value)
            else:
                raise CatalogMaterializationError(
                    "runtime adapter value is invalid"
                )
            parameters[key] = value
        return MappingProxyType(dict(sorted(parameters.items())))


def _layer_index(layer_id: str) -> int | None:
    if not layer_id.startswith("layer:"):
        return None
    try:
        return int(layer_id.removeprefix("layer:"))
    except ValueError as exc:
        raise CatalogMaterializationError(
            "runtime operator layer id is invalid"
        ) from exc


def _co_helper_layer(co_helper_mask: int, layer_id: str) -> bool:
    if not co_helper_mask:
        return False
    index = _layer_index(layer_id)
    return index is not None and bool(co_helper_mask >> index & 1)


def desktop_operator_placements(
    manifest: ModelManifest,
    gpu_first_layer: int,
    topology: RuntimePhysicalTopology,
) -> tuple[RuntimeCompositeOperatorPlacement, ...]:
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    if type(gpu_first_layer) is not int or not (
        0 <= gpu_first_layer <= manifest.block_count
    ):
        raise CatalogMaterializationError(
            "runtime desktop layer boundary is invalid"
        )
    rows = []
    for operator in manifest.operators:
        index = _layer_index(operator.layer_id)
        device_id = (
            topology.gpu_device_id
            if index is not None and index >= gpu_first_layer
            else topology.cpu_device_id
        )
        rows.append(RuntimeCompositeOperatorPlacement(
            operator_id=operator.operator_id,
            primary_device_id=device_id,
            helper_device_id=None,
            split_axis="none",
            split_fraction_ppm=0,
        ))
    return tuple(rows)


def measured_desktop_control_profile(
    manifest: ModelManifest,
    *,
    executor_id: str,
    operator_placements: Sequence[RuntimeCompositeOperatorPlacement],
    evidence_ids: Sequence[str],
    cuda_graph_mode: str = "default",
) -> RuntimeDesktopControlProfile:
    """Freeze one physically measured desktop placement for an artifact."""
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    executor_id = _text("runtime desktop control executor", executor_id)
    placements = tuple(operator_placements)
    if {row.operator_id for row in placements} != {
        row.operator_id for row in manifest.operators
    }:
        raise CatalogMaterializationError(
            "runtime desktop control does not cover the model graph"
        )
    return RuntimeDesktopControlProfile(
        profile_id=(
            "desktop-control:"
            + manifest.artifact_sha256.removeprefix("sha256:")[:16]
        ),
        artifact_sha256=manifest.artifact_sha256,
        executor_id=executor_id,
        operator_placements=placements,
        maturity="QUALIFIED",
        cuda_graph_mode=cuda_graph_mode,
        evidence_ids=_evidence(
            "runtime desktop control evidence", evidence_ids
        ),
    )


def derived_desktop_gpu_first_layer(
    manifest: ModelManifest,
    *,
    gpu_capacity_bytes: int,
    gpu_reserve_bytes: int,
) -> int:
    """Return the largest contiguous layer suffix that fits GPU memory."""
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    gpu_capacity_bytes = _positive_integer(
        "runtime GPU capacity", gpu_capacity_bytes
    )
    if type(gpu_reserve_bytes) is not int or gpu_reserve_bytes < 0:
        raise CatalogMaterializationError(
            "runtime GPU reserve is invalid"
        )
    if gpu_reserve_bytes >= gpu_capacity_bytes:
        raise CatalogMaterializationError(
            "runtime GPU reserve exhausts capacity"
        )
    budget = gpu_capacity_bytes - gpu_reserve_bytes
    tensors = manifest.tensor_by_id
    for first_layer in range(manifest.block_count + 1):
        tensor_ids = {
            tensor_id
            for operator in manifest.operators
            for layer_index in (_layer_index(operator.layer_id),)
            if layer_index is not None and layer_index >= first_layer
            for tensor_id in operator.tensor_ids
        }
        required = sum(tensors[tensor_id].nbytes for tensor_id in tensor_ids)
        if required <= budget:
            return first_layer
    return manifest.block_count


def desktop_model_composite_capability(
    manifest: ModelManifest,
    topology: RuntimePhysicalTopology,
    *,
    executor_id: str,
    endpoint: str,
    backend: str,
    gpu_first_layer: int,
    adapter_parameters: Mapping[str, int | str],
    evidence_ids: Sequence[str],
    operator_plan_protocol: str = "s41-static-ffn-plan-v1",
) -> RuntimeCompositeExecutorCapability:
    """Bind one measured desktop placement to an executable endpoint."""
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    if not isinstance(topology, RuntimePhysicalTopology):
        raise CatalogMaterializationError("runtime topology is invalid")
    executor_id = _text("runtime desktop executor", executor_id)
    endpoint = _text("runtime desktop endpoint", endpoint)
    backend = _text("runtime desktop backend", backend)
    operator_plan_protocol = _text(
        "runtime desktop operator plan protocol", operator_plan_protocol
    )
    if type(gpu_first_layer) is not int or not (
        0 <= gpu_first_layer <= manifest.block_count
    ):
        raise CatalogMaterializationError(
            "runtime desktop layer boundary is invalid"
        )
    parameters = RuntimeModelEndpointCapability._validated_adapter_parameters(
        adapter_parameters
    )
    coordinator_resource = _coordinator_resource(executor_id)
    has_context_contract = all(
        name in parameters for name in ("context_size", "parallel")
    )
    if has_context_contract:
        _positive_integer(
            "runtime desktop context size", parameters.get("context_size")
        )
        _positive_integer(
            "runtime desktop parallelism", parameters.get("parallel")
        )
    context_resource = (
        _context_resource(executor_id) if has_context_contract else None
    )
    context_token_quantum = (
        _context_token_quantum(parameters)
        if has_context_contract else None
    )
    gpu_resources = tuple(dict.fromkeys((
        topology.gpu_resource_id,
        topology.gpu_exclusive_residency_resource_id,
    )))
    return RuntimeCompositeExecutorCapability(
        executor_id=executor_id,
        endpoint=endpoint,
        backend=backend,
        coordinator_device_id=topology.cpu_device_id,
        participant_device_ids=topology.desktop_device_ids,
        participant_resource_ids={
            topology.cpu_device_id: (
                topology.cpu_resource_id,
                coordinator_resource,
                *((context_resource,) if context_resource is not None else ()),
            ),
            topology.gpu_device_id: gpu_resources,
        },
        route_family="layer_placement",
        assisted_operator_kind=None,
        split_axis="none",
        split_fractions_ppm=(),
        layer_fractions_ppm=(),
        residency_states=("cold", "hot", "warm"),
        resource_ids=(
            topology.cpu_resource_id,
            *gpu_resources,
            coordinator_resource,
            *((context_resource,) if context_resource is not None else ()),
        ),
        operator_plan_protocol=operator_plan_protocol,
        maturity="QUALIFIED",
        evidence_ids=_evidence(
            "runtime desktop evidence", evidence_ids
        ),
        artifact_sha256=manifest.artifact_sha256,
        operator_placements=desktop_operator_placements(
            manifest, gpu_first_layer, topology
        ),
        adapter_parameters={
            **parameters,
            "cpu_device_id": topology.cpu_device_id,
            "gpu_device_id": topology.gpu_device_id,
            "gpu_layers": manifest.block_count - gpu_first_layer,
            **(
                {
                    "context_resource_id": context_resource,
                    "context_token_quantum": context_token_quantum,
                    "request_memory_mode": "preallocated",
                }
                if context_resource is not None else {}
            ),
        },
        replacement_group_by_device={
            topology.cpu_device_id: (
                topology.gpu_exclusive_residency_resource_id
            ),
            topology.gpu_device_id: (
                topology.gpu_exclusive_residency_resource_id
            ),
        },
    )


def base_executor_capabilities(
    topology: RuntimePhysicalTopology,
    *,
    evidence_id: str,
    phone_minimum_battery_ppm: int = 50_000,
    qualified_helper_phone_ids: Sequence[str] = (),
) -> tuple[RuntimeExecutorCapability, ...]:
    evidence_id = _sha256("runtime executor evidence", evidence_id)
    operator_kinds = (
        "attention",
        "attention_projection",
        "embedding",
        "ffn",
        "kv_cache",
        "lm_head",
    )
    rows = []
    for device_id, backend, resource_id, memory_id in (
        (
            topology.cpu_device_id,
            "cpu",
            topology.cpu_resource_id,
            topology.host_memory_resource_id,
        ),
        (
            topology.gpu_device_id,
            "gpu",
            topology.gpu_resource_id,
            topology.gpu_memory_resource_id,
        ),
        (
            topology.phone_device_id,
            "phone",
            topology.phone_compute_resource_ids[0],
            topology.phone_memory_resource_id,
        ),
    ):
        kinds = operator_kinds if backend != "phone" else ("ffn",)
        rows.append(RuntimeExecutorCapability(
            executor_id="physical:" + device_id,
            device_id=device_id,
            endpoint="physical://" + device_id,
            backend=backend,
            execution_resource_ids=(
                (resource_id,)
                if backend != "phone"
                else tuple(dict.fromkeys((
                    *topology.phone_compute_resource_ids,
                    *topology.phone_transport_resource_ids,
                )))
            ),
            memory_resource_id=memory_id,
            kernel_profiles={
                kind: "prior:" + device_id + ":" + kind for kind in kinds
            },
            supported_quantizations=("*",),
            supports_whole_model=backend == "cpu",
            supports_layer_placement=backend != "phone",
            supports_operator_placement=True,
            supports_kv_cache=backend != "phone",
            supports_split_coordinator=backend != "phone",
            supports_split_helper=backend == "phone",
            split_axes=("column",) if backend == "phone" else (),
            split_fractions_ppm=(
                (250_000, 500_000, 750_000)
                if backend == "phone" else ()
            ),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            maturity="QUALIFIED",
            evidence_ids=(evidence_id,),
            qualified_fallback=backend == "cpu",
            maximum_temperature_millic=90_000,
            minimum_battery_ppm=(
                phone_minimum_battery_ppm if backend == "phone" else 0
            ),
            workspace_bytes_per_token=4_096,
            coordinated_route_families=(),
            operator_plan_protocol=None,
            exclusive_residency_resource_id=(
                topology.gpu_exclusive_residency_resource_id
                if backend == "gpu"
                else (
                    topology.phone_exclusive_residency_resource_id
                    if backend == "phone" else None
                )
            ),
            phone_sessions=(
                topology.phone_sessions if backend == "phone" else ()
            ),
        ))
    qualified = set(qualified_helper_phone_ids)
    if qualified - {row.device_id for row in topology.helper_phones}:
        raise CatalogMaterializationError(
            "qualified helper phone is not in the topology"
        )
    for helper in topology.helper_phones:
        # an FFN-only co-helper: never a generic placement target, only a composite participant
        rows.append(RuntimeExecutorCapability(
            executor_id="physical:" + helper.device_id,
            device_id=helper.device_id,
            endpoint="physical://" + helper.device_id,
            backend="phone",
            execution_resource_ids=tuple(dict.fromkeys((
                *helper.compute_resource_ids,
                *helper.transport_resource_ids,
            ))),
            memory_resource_id=helper.memory_resource_id,
            kernel_profiles={"ffn": "prior:" + helper.device_id + ":ffn"},
            supported_quantizations=("*",),
            supports_whole_model=False,
            supports_layer_placement=False,
            supports_operator_placement=False,
            supports_kv_cache=False,
            supports_split_coordinator=False,
            supports_split_helper=True,
            split_axes=("column",),
            split_fractions_ppm=(250_000, 500_000, 750_000),
            layer_fractions_ppm=(),
            residency_states=("cold", "hot", "warm"),
            # never qualified from the primary phone's receipts
            maturity="QUALIFIED" if helper.device_id in qualified else "SHADOW",
            evidence_ids=(evidence_id,),
            qualified_fallback=False,
            maximum_temperature_millic=90_000,
            minimum_battery_ppm=phone_minimum_battery_ppm,
            workspace_bytes_per_token=4_096,
            coordinated_route_families=(),
            operator_plan_protocol=None,
            exclusive_residency_resource_id=None,
        ))
    return tuple(rows)


def apply_assumed_phone_power_profile(
    profile: PlacementHardwareProfile,
    phone_power: RuntimePhonePowerProfile,
) -> PlacementHardwareProfile:
    """Validate an orthogonal charging-safe whole-phone power model."""
    if not isinstance(profile, PlacementHardwareProfile) or not isinstance(
        phone_power, RuntimePhonePowerProfile
    ):
        raise CatalogMaterializationError(
            "assumed phone power materialization is invalid"
        )
    device = profile.devices.get(phone_power.device_id)
    domain = profile.domains.get(phone_power.domain_id)
    phone_kernels = tuple(
        row for row in profile.kernels.values()
        if row.device_id == phone_power.device_id
    )
    if (
        device is None
        or not device.kind.startswith("phone")
        or domain is None
        or not phone_kernels
        or any(
            row.kernel.domain_id != phone_power.domain_id
            for row in phone_kernels
        )
    ):
        raise CatalogMaterializationError(
            "assumed phone power profile has no matching device domain"
        )
    return profile


def apply_assumed_phone_power_catalog(
    catalog: RuntimeCapabilityCatalog,
    phone_power: RuntimePhonePowerProfile,
) -> RuntimeCapabilityCatalog:
    """Bind an assumed phone power model to an existing catalog."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise CatalogMaterializationError(
            "assumed phone power catalog is invalid"
        )
    executors = tuple(catalog.executors)
    if not any(
        row.device_id == phone_power.device_id for row in executors
    ):
        raise CatalogMaterializationError(
            "assumed phone power catalog has no matching executor"
        )
    profiles = tuple(sorted(
        (*(
            row for row in catalog.phone_power_profiles
            if row.device_id != phone_power.device_id
        ), phone_power),
        key=lambda row: row.device_id,
    ))
    return replace(
        catalog,
        placement_profile=apply_assumed_phone_power_profile(
            catalog.placement_profile, phone_power
        ),
        executors=executors,
        phone_power_profiles=profiles,
    )


def _coordinator_resource(executor_id: str) -> str:
    return "coordinator:" + executor_id.removeprefix("physical:")


def _context_resource(executor_id: str) -> str:
    return "context:" + executor_id.removeprefix("physical:")


def _context_token_quantum(
    parameters: Mapping[str, int | str],
) -> int:
    context_size = _positive_integer(
        "runtime context size", parameters.get("context_size")
    )
    quantum = _positive_integer(
        "runtime context token quantum",
        parameters.get(
            "context_token_quantum",
            parameters.get("ubatch_size", 1),
        ),
    )
    if quantum > context_size:
        raise CatalogMaterializationError(
            "runtime context token quantum exceeds context size"
        )
    return quantum


def _cpu_composite_capability(
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
) -> RuntimeCompositeExecutorCapability:
    manifest = model.manifest
    assert model.cpu_executor_id is not None
    assert model.cpu_endpoint is not None
    assert model.cpu_backend is not None
    parameters = (
        model.adapter_parameters
        if model.cpu_adapter_parameters is None else model.cpu_adapter_parameters
    )
    cpu_coordinator_resource = _coordinator_resource(
        model.cpu_executor_id
    )
    cpu_context_resource = (
        _context_resource(model.cpu_executor_id)
        if "context_size" in parameters
        and "parallel" in parameters
        else None
    )
    cpu_context_token_quantum = (
        _context_token_quantum(parameters)
        if cpu_context_resource is not None else None
    )
    cpu_parameters = {
        **parameters,
        "cpu_device_id": topology.cpu_device_id,
        "gpu_device_id": topology.gpu_device_id,
        "gpu_layers": 0,
        "requires_measured_route_profile": 1,
        **(
            {
                "context_resource_id": cpu_context_resource,
                "context_token_quantum": cpu_context_token_quantum,
                "request_memory_mode": "preallocated",
            }
            if cpu_context_resource is not None else {}
        ),
    }
    return RuntimeCompositeExecutorCapability(
        executor_id=model.cpu_executor_id,
        endpoint=model.cpu_endpoint,
        backend=model.cpu_backend,
        coordinator_device_id=topology.cpu_device_id,
        participant_device_ids=(topology.cpu_device_id,),
        participant_resource_ids={
            topology.cpu_device_id: (
                topology.cpu_resource_id,
                cpu_coordinator_resource,
                *(
                    (cpu_context_resource,)
                    if cpu_context_resource is not None else ()
                ),
            ),
        },
        route_family="layer_placement",
        assisted_operator_kind=None,
        split_axis="none",
        split_fractions_ppm=(),
        layer_fractions_ppm=(),
        residency_states=("cold", "hot", "warm"),
        resource_ids=(
            topology.cpu_resource_id,
            cpu_coordinator_resource,
            *(
                (cpu_context_resource,)
                if cpu_context_resource is not None else ()
            ),
        ),
        operator_plan_protocol=model.operator_plan_protocol,
        maturity="QUALIFIED",
        evidence_ids=(model.desktop_evidence_ids
                      if model.cpu_evidence_ids is None else model.cpu_evidence_ids),
        artifact_sha256=manifest.artifact_sha256,
        operator_placements=desktop_operator_placements(
            manifest, manifest.block_count, topology
        ),
        adapter_parameters=cpu_parameters,
    )


def _phone_family_adapter_parameters(
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
    baseline: RuntimeCompositeExecutorCapability,
) -> dict[str, object]:
    manifest = model.manifest
    phone_parameters = {
        **dict(baseline.adapter_parameters),
        **model.phone_adapter_parameters,
        "ffn_column_quantum": model.ffn_column_quantum,
        "ffn_max_runtime_partitions": (
            model.ffn_max_runtime_partitions
        ),
        "ffn_n_embd": manifest.embedding_length,
        "ffn_wire_element_bytes": 2,
        "functionfs_resource_id": topology.functionfs_resource_id,
        "maximum_helper_resident_weight_bytes": (
            model.phone_resident_limit_bytes
        ),
        "phone_device_id": topology.phone_device_id,
    }
    phone_parameters.setdefault("usb_batch_plan", "split-row")
    if model.co_helpers is not None:
        phone_parameters[PHONE_CO_HELPERS_PARAMETER] = model.co_helpers.encode()
        phone_parameters["ffn_column_quantum"] = max(
            model.ffn_column_quantum, model.co_helpers.column_quantum(1)
        )
    if model.phone_runtime_control_protocol is not None:
        phone_parameters["ffn_runtime_control_protocol"] = (
            model.phone_runtime_control_protocol
        )
    if topology.gpu_device_id not in baseline.participant_device_ids:
        phone_parameters["requires_measured_route_profile"] = 1
    if model.phone_preloaded:
        phone_parameters["preloaded_resident_device_ids"] = (
            topology.phone_device_id
        )
    return phone_parameters


def _phone_family_participant_resources(
    topology: RuntimePhysicalTopology,
    baseline: RuntimeCompositeExecutorCapability,
    coordinator_resource: str,
    co_helper_device_ids: Sequence[str] = (),
) -> dict[str, tuple[str, ...]]:
    participant_resources = {
        device_id: tuple(resource_ids)
        for device_id, resource_ids in (
            baseline.participant_resource_ids.items()
        )
    }
    participant_resources[topology.cpu_device_id] = tuple(
        dict.fromkeys((
            *participant_resources[topology.cpu_device_id],
            coordinator_resource,
        ))
    )
    participant_resources[topology.phone_device_id] = (
        *topology.phone_transport_resource_ids,
        *topology.phone_compute_resource_ids,
    )
    for device_id in co_helper_device_ids:
        helper = topology.helper_phone(device_id)
        participant_resources[device_id] = (
            *helper.transport_resource_ids,
            *helper.compute_resource_ids,
        )
    return participant_resources


def _phone_family_capabilities(
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
    baseline: RuntimeCompositeExecutorCapability,
    prefix: str,
    endpoint: str,
    backend: str,
    family: str,
) -> list[RuntimeCompositeExecutorCapability]:
    """Materialize one phone-assisted route family plus batch-plan variants."""
    manifest = model.manifest
    executor_id = prefix + ":" + family
    coordinator_resource = _coordinator_resource(executor_id)
    co_helper_ids = (
        () if model.co_helpers is None else model.co_helpers.device_ids
    )
    co_helper_mask = 0 if model.co_helpers is None else model.co_helpers.layer_mask
    resources = tuple(dict.fromkeys((
        *tuple(baseline.resource_ids),
        coordinator_resource,
        *topology.phone_transport_resource_ids,
        *topology.phone_compute_resource_ids,
        *(
            resource_id
            for device_id in co_helper_ids
            for helper in (topology.helper_phone(device_id),)
            for resource_id in (
                *helper.transport_resource_ids, *helper.compute_resource_ids
            )
        ),
    )))
    phone_parameters = _phone_family_adapter_parameters(
        model, topology, baseline
    )
    fractions = tuple(
        fraction
        for fraction in (250_000, 500_000, 750_000)
        if manifest.feed_forward_length * fraction % 1_000_000 == 0
    )
    participant_devices = tuple(dict.fromkeys((
        *baseline.participant_device_ids, topology.phone_device_id,
        *co_helper_ids,
    )))
    participant_resources = _phone_family_participant_resources(
        topology, baseline, coordinator_resource, co_helper_ids
    )
    base_phone_capability = RuntimeCompositeExecutorCapability(
        executor_id=executor_id,
        endpoint=endpoint,
        backend=backend,
        coordinator_device_id=topology.cpu_device_id,
        participant_device_ids=participant_devices,
        participant_resource_ids=participant_resources,
        route_family=family,
        assisted_operator_kind="ffn",
        split_axis="column" if family == "operator_split" else "none",
        split_fractions_ppm=(
            fractions if family == "operator_split" else ()
        ),
        layer_fractions_ppm=(),
        residency_states=("cold", "hot", "warm"),
        resource_ids=resources,
        operator_plan_protocol=model.operator_plan_protocol,
        maturity="QUALIFIED",
        evidence_ids=model.phone_evidence_ids,
        artifact_sha256=manifest.artifact_sha256,
        # the primary phone's candidates; co-helper layers belong to their owner
        operator_ids=tuple(
            operator.operator_id for operator in manifest.operators
            if operator.kind == "ffn"
            and not _co_helper_layer(co_helper_mask, operator.layer_id)
        ),
        adapter_parameters=phone_parameters,
        replacement_group_by_device={
            **dict(baseline.replacement_group_by_device),
            **(
                {}
                if topology.phone_exclusive_residency_resource_id
                    is None
                else {
                    topology.phone_device_id: (
                        topology.phone_exclusive_residency_resource_id
                    )
                }
            ),
        },
        baseline_executor_id=baseline.executor_id,
        helper_device_id=topology.phone_device_id,
    )
    rows = []
    for batch_plan in model.phone_batch_plans:
        rows.append(replace(
            base_phone_capability,
            executor_id=(
                executor_id if batch_plan == "split-row"
                else executor_id + ":" + batch_plan
            ),
            maturity=(
                "QUALIFIED"
                if batch_plan in model.qualified_phone_batch_plans
                else "SHADOW"
            ),
            adapter_parameters={
                **dict(base_phone_capability.adapter_parameters),
                "usb_batch_plan": batch_plan,
            },
        ))
    return rows


def model_composite_capabilities(
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
) -> tuple[RuntimeCompositeExecutorCapability, ...]:
    manifest = model.manifest
    desktop = desktop_model_composite_capability(
        manifest,
        topology,
        executor_id=model.desktop_executor_id,
        endpoint=model.desktop_endpoint,
        backend=model.desktop_backend,
        gpu_first_layer=model.desktop_gpu_first_layer,
        adapter_parameters=model.adapter_parameters,
        evidence_ids=model.desktop_evidence_ids,
        operator_plan_protocol=model.operator_plan_protocol,
    )
    rows = [desktop]
    baselines = [(
        desktop,
        model.phone_executor_prefix,
        model.phone_endpoint,
        model.phone_backend,
    )]
    if model.cpu_executor_id is not None:
        assert model.cpu_phone_executor_prefix is not None
        assert model.cpu_phone_endpoint is not None
        assert model.cpu_phone_backend is not None
        cpu = _cpu_composite_capability(model, topology)
        rows.append(cpu)
        baselines.append((
            cpu,
            model.cpu_phone_executor_prefix,
            model.cpu_phone_endpoint,
            model.cpu_phone_backend,
        ))
    for baseline, prefix, endpoint, backend in baselines:
        for family in ("operator_offload", "operator_split"):
            rows.extend(_phone_family_capabilities(
                model, topology, baseline, prefix, endpoint, backend, family
            ))
    return tuple(rows)


def resource_profiles_for_catalog(
    topology: RuntimePhysicalTopology,
    composites: Sequence[RuntimeCompositeExecutorCapability],
    link_ids: Sequence[str],
) -> Mapping[str, ResourceProfile]:
    coordinator_capacities = {
        _coordinator_resource(row.executor_id): _positive_integer(
            "runtime coordinator capacity",
            row.adapter_parameters.get("parallel"),
        )
        for row in composites
    }
    for row in composites:
        for resource_id in row.resource_ids:
            if resource_id.startswith("coordinator:"):
                coordinator_capacities.setdefault(
                    resource_id,
                    coordinator_capacities[_coordinator_resource(row.executor_id)],
                )
    context_capacities = {
        _text(
            "runtime context resource",
            row.adapter_parameters.get("context_resource_id"),
        ): _positive_integer(
            "runtime context capacity",
            row.adapter_parameters.get("context_size"),
        ) // _positive_integer(
            "runtime context token quantum",
            row.adapter_parameters.get("context_token_quantum"),
        )
        for row in composites
        if row.adapter_parameters.get("request_memory_mode")
            == "preallocated"
    }
    resource_ids = {
        topology.cpu_resource_id,
        topology.gpu_resource_id,
        topology.gpu_exclusive_residency_resource_id,
        *topology.phone_transport_resource_ids,
        *topology.phone_compute_resource_ids,
        *(
            resource_id for row in topology.helper_phones
            for resource_id in (
                *row.transport_resource_ids, *row.compute_resource_ids
            )
        ),
        *coordinator_capacities,
        *context_capacities,
        *("link:" + _text("runtime link id", link_id) for link_id in link_ids),
    }
    rows = {}
    for resource_id in resource_ids:
        if resource_id.startswith("coordinator:"):
            kind = "coordinator"
        elif resource_id.startswith("context:"):
            kind = "memory-pool"
        elif resource_id in {
            topology.cpu_resource_id,
            topology.gpu_resource_id,
            topology.gpu_exclusive_residency_resource_id,
            *topology.phone_compute_resource_ids,
            *(
                value for row in topology.helper_phones
                for value in row.compute_resource_ids
            ),
        }:
            kind = "compute"
        else:
            kind = "transport"
        rows[resource_id] = ResourceProfile(
            resource_id=resource_id,
            kind=kind,
            capacity=coordinator_capacities.get(
                resource_id,
                context_capacities.get(
                    resource_id,
                    topology.resource_capacities.get(resource_id, 1),
                ),
            ),
            ready=True,
            identity=(
                resource_id.removeprefix("link:")
                if resource_id.startswith("link:")
                else topology.resource_identities.get(resource_id, resource_id)
            ),
        )
    return MappingProxyType(dict(sorted(rows.items())))


def model_transition_capabilities(
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
    composites: Sequence[RuntimeCompositeExecutorCapability],
    *,
    latency_us: int,
    energy_uj: int,
) -> tuple[RuntimeTransitionCapability, ...]:
    return model_manifest_transition_capabilities(
        model.manifest,
        topology,
        composites,
        latency_us=latency_us,
        energy_uj=energy_uj,
    )


def model_manifest_transition_capabilities(
    manifest: ModelManifest,
    topology: RuntimePhysicalTopology,
    composites: Sequence[RuntimeCompositeExecutorCapability],
    *,
    latency_us: int,
    energy_uj: int,
) -> tuple[RuntimeTransitionCapability, ...]:
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    if not isinstance(topology, RuntimePhysicalTopology):
        raise CatalogMaterializationError("runtime topology is invalid")
    latency_us = _positive_integer("runtime transition latency", latency_us)
    energy_uj = _positive_integer("runtime transition energy", energy_uj)
    rows = []
    for composite in composites:
        if composite.artifact_sha256 != manifest.artifact_sha256:
            continue
        resources = list(composite.resource_ids)
        exclusive_devices = (
            (topology.gpu_device_id,)
            if topology.gpu_device_id in composite.participant_device_ids
            and topology.gpu_exclusive_residency_resource_id in resources
            else ()
        )
        transition_device_id = (
            exclusive_devices[0]
            if exclusive_devices else composite.coordinator_device_id
        )
        co_helpers = co_helper_declaration(composite.adapter_parameters)
        # a static co-helper is started by the rig for the trace, never by a model transition
        prepared_device_ids = tuple(
            device_id for device_id in composite.participant_device_ids
            if co_helpers is None or device_id not in co_helpers.device_ids
        )
        for source_state in ("cold", "hot", "warm"):
            rows.append(RuntimeTransitionCapability(
                transition_id=(
                    "load:" + manifest.model_id + ":"
                    + composite.executor_id + ":" + source_state
                ),
                device_id=transition_device_id,
                source_state=source_state,
                target_state="hot",
                fixed_latency_us=latency_us,
                bandwidth_bytes_per_s=10**15,
                fixed_energy_uj=energy_uj,
                dynamic_pj_per_byte=0,
                resource_ids=tuple(resources),
                maturity="QUALIFIED",
                evidence_ids=composite.evidence_ids,
                resource_slots={
                    resource_id: topology.resource_capacities.get(
                        resource_id, 1
                    )
                    for resource_id in resources
                },
                artifact_sha256=manifest.artifact_sha256,
                executor_id=composite.executor_id,
                prepares_device_ids=prepared_device_ids,
                energy_maturity="SHADOW",
            ))
    return tuple(rows)


def materialize_cpu_phone_endpoints(
    catalog: RuntimeCapabilityCatalog,
    model: RuntimeModelEndpointCapability,
    topology: RuntimePhysicalTopology,
    *,
    transition_latency_us: int,
    transition_energy_uj: int,
    cpu_transition_energy_maturity: str = "SHADOW",
    route_shape_profiles: Sequence[RuntimeRouteShapeProfile] = (),
) -> RuntimeCapabilityCatalog:
    """Add an independently measured CPU parent without replacing its GPU control."""
    if model.cpu_executor_id is None or model.cpu_evidence_ids is None:
        raise CatalogMaterializationError("runtime CPU parent evidence is absent")
    if model.cpu_adapter_parameters is None:
        raise CatalogMaterializationError("runtime CPU parent parameters are absent")
    if model.manifest.artifact_sha256 not in catalog.desktop_control_by_artifact:
        raise CatalogMaterializationError("runtime desktop control is absent")
    composites = tuple(
        row for row in model_composite_capabilities(model, topology)
        if row.executor_id == model.cpu_executor_id
        or row.baseline_executor_id == model.cpu_executor_id
    )
    by_id = {row.executor_id: row for row in composites}
    if set(by_id).intersection(catalog.composite_executor_by_id):
        raise CatalogMaterializationError("runtime CPU executor already exists")
    profiles = tuple(route_shape_profiles)
    if any(
        row.executor_id not in by_id
        or row.artifact_sha256 != model.manifest.artifact_sha256
        or row.route_family != by_id[row.executor_id].route_family
        or row.device_ids != tuple(sorted(by_id[row.executor_id].participant_device_ids))
        for row in profiles
    ):
        raise CatalogMaterializationError("runtime CPU profile is not bound to its parent")
    resources = dict(catalog.resources)
    for resource_id, resource in resource_profiles_for_catalog(topology, composites, ()).items():
        if resource_id not in resources:
            resources[resource_id] = resource
    transitions = model_manifest_transition_capabilities(
        model.manifest, topology, composites,
        latency_us=transition_latency_us, energy_uj=transition_energy_uj,
    )
    transitions = tuple(
        replace(row, energy_maturity=cpu_transition_energy_maturity)
        if row.executor_id == model.cpu_executor_id else row
        for row in transitions
    )
    return replace(
        catalog,
        resources=resources,
        composite_executors=catalog.composite_executors + composites,
        transitions=catalog.transitions + transitions,
        route_shape_profiles=catalog.route_shape_profiles + profiles,
    )


def materialize_desktop_control(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    topology: RuntimePhysicalTopology,
    *,
    executor_id: str,
    endpoint: str,
    backend: str,
    gpu_first_layer: int,
    adapter_parameters: Mapping[str, int | str],
    evidence_ids: Sequence[str],
    transition_latency_us: int,
    transition_energy_uj: int,
    route_shape_profiles: Sequence[RuntimeRouteShapeProfile] = (),
    operator_plan_protocol: str = "s41-static-ffn-plan-v1",
) -> RuntimeCapabilityCatalog:
    """Add one frozen desktop control and its executable transition path."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise CatalogMaterializationError("runtime catalog is invalid")
    if manifest.artifact_sha256 in catalog.desktop_control_by_artifact:
        raise CatalogMaterializationError(
            "runtime desktop control already exists"
        )
    composite = desktop_model_composite_capability(
        manifest,
        topology,
        executor_id=executor_id,
        endpoint=endpoint,
        backend=backend,
        gpu_first_layer=gpu_first_layer,
        adapter_parameters=adapter_parameters,
        evidence_ids=evidence_ids,
        operator_plan_protocol=operator_plan_protocol,
    )
    if composite.executor_id in catalog.composite_executor_by_id:
        raise CatalogMaterializationError(
            "runtime desktop executor already exists"
        )
    profiles = tuple(route_shape_profiles)
    if any(
        not isinstance(row, RuntimeRouteShapeProfile)
        or row.artifact_sha256 != manifest.artifact_sha256
        or row.executor_id != composite.executor_id
        or row.route_family != "layer_placement"
        or row.device_ids != tuple(sorted(topology.desktop_device_ids))
        for row in profiles
    ):
        raise CatalogMaterializationError(
            "runtime desktop route profile is not bound to the control"
        )
    resources = dict(catalog.resources)
    coordinator_resource = _coordinator_resource(composite.executor_id)
    if coordinator_resource in resources:
        raise CatalogMaterializationError(
            "runtime desktop coordinator already exists"
        )
    parallel = adapter_parameters.get("parallel", 1)
    resources[coordinator_resource] = ResourceProfile(
        resource_id=coordinator_resource,
        kind="coordinator",
        capacity=_positive_integer(
            "runtime desktop parallel capacity", parallel
        ),
        ready=True,
        identity=composite.executor_id,
    )
    context_resource = composite.adapter_parameters.get(
        "context_resource_id"
    )
    if context_resource is not None:
        if context_resource in resources:
            raise CatalogMaterializationError(
                "runtime desktop context resource already exists"
            )
        resources[context_resource] = ResourceProfile(
            resource_id=context_resource,
            kind="memory-pool",
            capacity=_positive_integer(
                "runtime desktop context capacity",
                composite.adapter_parameters.get("context_size"),
            ) // _positive_integer(
                "runtime desktop context token quantum",
                composite.adapter_parameters.get(
                    "context_token_quantum"
                ),
            ),
            ready=True,
            identity=composite.executor_id,
        )
    control = measured_desktop_control_profile(
        manifest,
        executor_id=composite.executor_id,
        operator_placements=composite.operator_placements,
        evidence_ids=evidence_ids,
    )
    transitions = model_manifest_transition_capabilities(
        manifest,
        topology,
        (composite,),
        latency_us=transition_latency_us,
        energy_uj=transition_energy_uj,
    )
    return RuntimeCapabilityCatalog(
        catalog_id=catalog.catalog_id,
        placement_profile=catalog.placement_profile,
        resources=resources,
        executors=catalog.executors,
        composite_executors=(
            catalog.composite_executors + (composite,)
        ),
        desktop_control_profiles=(
            catalog.desktop_control_profiles + (control,)
        ),
        transitions=catalog.transitions + transitions,
        minimum_energy_saving_ppm=(
            catalog.minimum_energy_saving_ppm
        ),
        maximum_latency_ppm=catalog.maximum_latency_ppm,
        route_shape_profiles=(
            catalog.route_shape_profiles + profiles
        ),
        system_cost_profiles=catalog.system_cost_profiles,
        phone_power_profiles=catalog.phone_power_profiles,
    )


_WHOLE_MODEL_REQUIRED_PARAMETERS = frozenset({
    "batch_size",
    "context_size",
    "cpu_device_id",
    "executable_device",
    "execution_adapter",
    "forward_port",
    "gpu_device_id",
    "model_alias",
    "parallel",
    "remote_library_directory",
    "remote_model_path",
    "remote_port",
    "remote_server_path",
    "remote_server_sha256",
    "request_io_protocol",
    "ubatch_size",
})


def _whole_model_parameters(
    adapter_parameters: Mapping[str, int | str],
    capability: RuntimeExecutorCapability,
) -> dict[str, int | str]:
    parameters = RuntimeModelEndpointCapability._validated_adapter_parameters(
        adapter_parameters
    )
    parameters = {
        **parameters,
        "request_transport_generation": (
            "adb-ncm-token-http-v1" if parameters.get("android_control_transport") == "adb-ncm"
            else "adb-token-http-v1"
        ),
    }
    if (
        set(parameters) < _WHOLE_MODEL_REQUIRED_PARAMETERS
        or parameters["execution_adapter"] != "android-llama-server-v1"
        or parameters["request_io_protocol"] != "token-ids-v1"
        or parameters["gpu_device_id"] != capability.device_id
        or type(parameters["remote_server_sha256"]) is not str
        or not parameters["remote_server_sha256"].startswith("sha256:")
        or len(parameters["remote_server_sha256"]) != 71
    ):
        raise CatalogMaterializationError(
            "whole-model execution contract is incomplete"
        )
    mode = parameters.get("android_control_transport", "adb-usb")
    if mode not in {"adb-usb", "adb-ncm"}:
        raise CatalogMaterializationError("whole-model control transport is invalid")
    if mode == "adb-ncm":
        _text("whole-model NCM control endpoint", parameters.get("android_control_endpoint"))
        _sha256("whole-model NCM control script", parameters.get("android_control_script_sha256"))
    return parameters


def _whole_model_request_link(
    *,
    link_id: str,
    source_device: str,
    target_device: str,
    fixed_latency_us: int,
    bandwidth_bytes_per_s: int,
    maximum_payload_bytes: int,
    evidence: tuple[str, ...],
    transport_profile_id: str,
    qualification_identity_sha256: str,
    transport_generation: str = "adb-token-http-v1",
) -> TransferLink:
    return TransferLink(
        link_id=link_id,
        source_device=source_device,
        target_device=target_device,
        fixed_latency_us=fixed_latency_us,
        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        fixed_dynamic_uj=0,
        dynamic_pj_per_byte=0,
        domain_active_power_mw={},
        status="estimated" if transport_generation == "adb-ncm-token-http-v1" else "measured",
        ready=True,
        evidence_ids=evidence,
        minimum_payload_bytes=1,
        maximum_payload_bytes=maximum_payload_bytes,
        queue_depth=1,
        concurrent_streams=1,
        allocator="adb-forward",
        full_duplex=False,
        transport_generation=transport_generation,
        transport_profile_id=transport_profile_id,
        qualification_identity_sha256=(None if transport_generation == "adb-ncm-token-http-v1"
                                       else qualification_identity_sha256),
    )


def _whole_model_request_links(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    capability: RuntimeExecutorCapability,
    cpu_device_id: str,
    *,
    fixed_latency_us: int,
    bandwidth_bytes_per_s: int,
    maximum_payload_bytes: int,
    evidence: tuple[str, ...],
    transport_identity: str,
    transport_generation: str = "adb-token-http-v1",
) -> tuple[TransferLink, TransferLink]:
    if (
        cpu_device_id not in catalog.placement_profile.devices
        or capability.device_id not in catalog.placement_profile.devices
    ):
        raise CatalogMaterializationError(
            "whole-model request transport device is absent"
        )
    transport_prefix = (
        "token-rpc-" + manifest.artifact_sha256[7:23]
    )
    request_links = (
        _whole_model_request_link(
            link_id=transport_prefix + "-h2d",
            source_device=cpu_device_id,
            target_device=capability.device_id,
            fixed_latency_us=fixed_latency_us,
            bandwidth_bytes_per_s=bandwidth_bytes_per_s,
            maximum_payload_bytes=maximum_payload_bytes,
            evidence=evidence,
            transport_profile_id=transport_prefix + ":h2d",
            qualification_identity_sha256=transport_identity,
            transport_generation=transport_generation,
        ),
        _whole_model_request_link(
            link_id=transport_prefix + "-d2h",
            source_device=capability.device_id,
            target_device=cpu_device_id,
            fixed_latency_us=fixed_latency_us,
            bandwidth_bytes_per_s=bandwidth_bytes_per_s,
            maximum_payload_bytes=maximum_payload_bytes,
            evidence=evidence,
            transport_profile_id=transport_prefix + ":d2h",
            qualification_identity_sha256=transport_identity,
            transport_generation=transport_generation,
        ),
    )
    existing_link_ids = {
        row.link_id for row in catalog.placement_profile.links
    }
    if existing_link_ids & {row.link_id for row in request_links}:
        raise CatalogMaterializationError(
            "whole-model request transport already exists"
        )
    return request_links


def _whole_model_placement_profile(
    catalog: RuntimeCapabilityCatalog,
    capability: RuntimeExecutorCapability,
    power_prior_mw: int,
    request_links: tuple[TransferLink, ...],
) -> PlacementHardwareProfile:
    profiled_kernels = {
        profile_id: (
            replace(
                row,
                kernel=replace(
                    row.kernel,
                    active_power_mw=max(
                        power_prior_mw,
                        row.kernel.active_power_mw,
                    ),
                ),
            )
            if row.device_id == capability.device_id
            and row.status == "estimated"
            else row
        )
        for profile_id, row in catalog.placement_profile.kernels.items()
    }
    return replace(
        catalog.placement_profile,
        kernels=profiled_kernels,
        links=tuple(sorted(
            catalog.placement_profile.links + request_links,
            key=lambda row: row.link_id,
        )),
    )


def _whole_model_transitions(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    updated: RuntimeExecutorCapability,
    *,
    executor_id: str,
    transition_latency_us: int,
    transition_energy_uj: int,
    transition_maturity: str,
    transition_energy_maturity: str,
    evidence: tuple[str, ...],
) -> tuple[RuntimeTransitionCapability, ...]:
    resources = updated.execution_resource_ids
    transition_ids = {
        "load:" + manifest.model_id + ":" + executor_id + ":" + state
        for state in ("cold", "hot", "warm")
    }
    if transition_ids & {row.transition_id for row in catalog.transitions}:
        raise CatalogMaterializationError(
            "whole-model transition already exists"
        )
    return tuple(
        RuntimeTransitionCapability(
            transition_id=(
                "load:" + manifest.model_id + ":" + executor_id + ":" + state
            ),
            device_id=updated.device_id,
            source_state=state,
            target_state="hot",
            fixed_latency_us=transition_latency_us,
            bandwidth_bytes_per_s=10**15,
            fixed_energy_uj=transition_energy_uj,
            dynamic_pj_per_byte=0,
            resource_ids=resources,
            maturity=transition_maturity,
            evidence_ids=evidence,
            resource_slots={
                resource_id: 1 for resource_id in resources
            },
            artifact_sha256=manifest.artifact_sha256,
            executor_id=executor_id,
            prepares_device_ids=(updated.device_id,),
            energy_maturity=transition_energy_maturity,
        )
        for state in ("cold", "hot", "warm")
    )


def materialize_whole_model_endpoint(
    catalog: RuntimeCapabilityCatalog,
    manifest: ModelManifest,
    *,
    executor_id: str,
    endpoint: str,
    backend: str,
    adapter_parameters: Mapping[str, int | str],
    evidence_ids: Sequence[str],
    transition_latency_us: int,
    transition_energy_uj: int,
    transition_maturity: str = "QUALIFIED",
    transition_energy_maturity: str = "SHADOW",
    request_transport_fixed_latency_us: int = 1_000,
    request_transport_bandwidth_bytes_per_s: int = 10_000_000,
    request_transport_maximum_payload_bytes: int = 65_536,
    request_transport_identity_sha256: str | None = None,
) -> RuntimeCapabilityCatalog:
    """Publish one executable token-in/token-out whole-model endpoint."""
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise CatalogMaterializationError("runtime catalog is invalid")
    if not isinstance(manifest, ModelManifest):
        raise CatalogMaterializationError("runtime manifest is invalid")
    executor_id = _text("whole-model executor", executor_id)
    endpoint = _text("whole-model endpoint", endpoint)
    backend = _text("whole-model backend", backend)
    capability = catalog.executor_by_id.get(executor_id)
    if capability is None:
        raise CatalogMaterializationError(
            "whole-model base executor is absent"
        )
    parameters = _whole_model_parameters(adapter_parameters, capability)
    peak = parameters.get("whole_model_peak_memory_bytes")
    if peak is not None and (
        type(peak) is not int or peak < manifest.tensor_bytes
    ):
        raise CatalogMaterializationError("whole-model peak memory is smaller than its weights")
    operator_kinds = {row.kind for row in manifest.operators}
    if operator_kinds - capability.operator_kinds:
        raise CatalogMaterializationError(
            "whole-model executor lacks required kernels"
        )
    evidence = tuple(sorted({
        *capability.evidence_ids,
        *_evidence("whole-model evidence", evidence_ids),
    }))
    transport_identity = _sha256(
        "whole-model request transport identity",
        (
            evidence[0]
            if request_transport_identity_sha256 is None
            else request_transport_identity_sha256
        ),
    )
    request_transport_fixed_latency_us = _positive_integer(
        "whole-model request transport latency",
        request_transport_fixed_latency_us,
    )
    request_transport_bandwidth_bytes_per_s = _positive_integer(
        "whole-model request transport bandwidth",
        request_transport_bandwidth_bytes_per_s,
    )
    request_transport_maximum_payload_bytes = _positive_integer(
        "whole-model request transport maximum payload",
        request_transport_maximum_payload_bytes,
    )
    cpu_device_id = _text(
        "whole-model request source device",
        parameters["cpu_device_id"],
    )
    request_links = _whole_model_request_links(
        catalog,
        manifest,
        capability,
        cpu_device_id,
        fixed_latency_us=request_transport_fixed_latency_us,
        bandwidth_bytes_per_s=request_transport_bandwidth_bytes_per_s,
        maximum_payload_bytes=request_transport_maximum_payload_bytes,
        evidence=evidence,
        transport_identity=transport_identity,
        transport_generation=parameters["request_transport_generation"],
    )
    power_prior_mw = parameters.get("whole_model_power_prior_mw", 5_000)
    power_prior_mw = _positive_integer(
        "whole-model power prior", power_prior_mw
    )
    placement_profile = _whole_model_placement_profile(
        catalog, capability, power_prior_mw, request_links
    )
    resources_by_id = dict(catalog.resources)
    residency_resource = capability.exclusive_residency_resource_id
    execution_resources = capability.execution_resource_ids
    if capability.phone_sessions:
        residency_resource = "residency:whole:" + executor_id
        resources_by_id[residency_resource] = ResourceProfile(
            resource_id=residency_resource, kind="residency", capacity=1,
            ready=True, identity=executor_id,
        )
        execution_resources = tuple(sorted({*execution_resources, residency_resource}))
    for link in request_links:
        resource_id = "link:" + link.link_id
        resources_by_id[resource_id] = ResourceProfile(
            resource_id=resource_id,
            kind="transport",
            capacity=1,
            ready=True,
            identity=link.transport_profile_id,
        )
    updated = replace(
        capability,
        endpoint=endpoint,
        backend=backend,
        exclusive_residency_resource_id=residency_resource,
        execution_resource_ids=execution_resources,
        supports_whole_model=True,
        supports_kv_cache=True,
        evidence_ids=evidence,
        adapter_parameters={
            **capability.adapter_parameters,
            **parameters,
        },
    )
    transition_latency_us = _positive_integer(
        "whole-model transition latency", transition_latency_us
    )
    transition_energy_uj = _positive_integer(
        "whole-model transition energy", transition_energy_uj
    )
    transitions = _whole_model_transitions(
        catalog,
        manifest,
        updated,
        executor_id=executor_id,
        transition_latency_us=transition_latency_us,
        transition_energy_uj=transition_energy_uj,
        transition_maturity=transition_maturity,
        transition_energy_maturity=transition_energy_maturity,
        evidence=evidence,
    )
    return RuntimeCapabilityCatalog(
        catalog_id=catalog.catalog_id,
        placement_profile=placement_profile,
        resources=resources_by_id,
        executors=tuple(
            updated if row.executor_id == executor_id else row
            for row in catalog.executors
        ),
        composite_executors=catalog.composite_executors,
        desktop_control_profiles=catalog.desktop_control_profiles,
        transitions=catalog.transitions + transitions,
        minimum_energy_saving_ppm=catalog.minimum_energy_saving_ppm,
        maximum_latency_ppm=catalog.maximum_latency_ppm,
        route_shape_profiles=catalog.route_shape_profiles,
        system_cost_profiles=catalog.system_cost_profiles,
        phone_power_profiles=catalog.phone_power_profiles,
    )


def merge_runtime_capability_catalogs(
    base: RuntimeCapabilityCatalog,
    overlay: RuntimeCapabilityCatalog,
) -> RuntimeCapabilityCatalog:
    """Merge physical catalogs without replacing complete executor contracts."""
    if not isinstance(base, RuntimeCapabilityCatalog) or not isinstance(
        overlay, RuntimeCapabilityCatalog
    ):
        raise CatalogMaterializationError("runtime catalog merge is invalid")
    base_pool_ids = set(base.placement_profile.memory_pools)
    overlay_pool_ids = set(overlay.placement_profile.memory_pools)
    session_pool_ids = {
        session.memory_resource_id
        for executor in base.executors
        for session in executor.phone_sessions
    }
    if (
        set(base.placement_profile.devices)
            != set(overlay.placement_profile.devices)
        or not overlay_pool_ids.issubset(base_pool_ids)
        or base_pool_ids - overlay_pool_ids - session_pool_ids
        or set(base.placement_profile.kernels)
            != set(overlay.placement_profile.kernels)
    ):
        raise CatalogMaterializationError("overlay hardware topology differs")
    available_links = {row.link_id for row in base.placement_profile.links}
    resources = dict(base.resources)
    for resource_id, resource in overlay.resources.items():
        if (
            resource_id.startswith("link:")
            and resource_id.removeprefix("link:") not in available_links
        ):
            continue
        current = resources.get(resource_id)
        if current is not None and current != resource:
            if resource_id.startswith("link:"):
                continue
            raise CatalogMaterializationError(
                "shared resource differs: " + resource_id
            )
        resources[resource_id] = resource
    base_by_device = {row.device_id: row for row in base.executors}
    overlay_by_device = {row.device_id: row for row in overlay.executors}
    if set(base_by_device) != set(overlay_by_device):
        raise CatalogMaterializationError("overlay executor devices differ")
    phone_power_profiles = tuple({
        row.device_id: row
        for row in (
            *base.phone_power_profiles,
            *overlay.phone_power_profiles,
        )
    }.values())
    phone_power_by_device = {
        row.device_id: row for row in phone_power_profiles
    }

    def merge_executor(device_id: str) -> RuntimeExecutorCapability:
        result = base_by_device[device_id].with_runtime_overlay(
            overlay_by_device[device_id]
        )
        phone_power = phone_power_by_device.get(device_id)
        return (
            result
            if phone_power is None
            else replace(
                result,
                minimum_battery_ppm=phone_power.minimum_battery_ppm,
            )
        )

    executors = tuple(
        merge_executor(device_id)
        for device_id in sorted(base_by_device)
    )
    composites = base.composite_executors + tuple(
        replace(
            row,
            resource_ids=tuple(
                resource_id for resource_id in row.resource_ids
                if not resource_id.startswith("link:")
                or resource_id.removeprefix("link:") in available_links
            ),
            participant_resource_ids={
                device_id: tuple(
                    resource_id for resource_id in resource_ids
                    if not resource_id.startswith("link:")
                    or resource_id.removeprefix("link:") in available_links
                )
                for device_id, resource_ids in (
                    row.participant_resource_ids.items()
                )
            },
        )
        for row in overlay.composite_executors
    )
    system_cost_profiles: tuple[RuntimeSystemCostProfile, ...] = tuple(
        replace(
            row,
            interference_ppm_by_resource={
                resource_id: value
                for resource_id, value in (
                    row.interference_ppm_by_resource.items()
                )
                if resource_id in resources
            },
        )
        for row in overlay.system_cost_profiles
    )
    return RuntimeCapabilityCatalog(
        catalog_id=base.catalog_id,
        placement_profile=base.placement_profile,
        resources=resources,
        executors=executors,
        composite_executors=composites,
        desktop_control_profiles=(
            base.desktop_control_profiles
            + overlay.desktop_control_profiles
        ),
        transitions=base.transitions + overlay.transitions,
        minimum_energy_saving_ppm=max(
            base.minimum_energy_saving_ppm,
            overlay.minimum_energy_saving_ppm,
        ),
        maximum_latency_ppm=min(
            base.maximum_latency_ppm,
            overlay.maximum_latency_ppm,
        ),
        route_shape_profiles=(
            base.route_shape_profiles + overlay.route_shape_profiles
        ),
        system_cost_profiles=system_cost_profiles,
        phone_power_profiles=phone_power_profiles,
    )


__all__ = [
    "CatalogMaterializationError",
    "MeasuredOperatorAssistProfile",
    "RuntimeModelEndpointCapability",
    "RuntimePhysicalTopology",
    "apply_assumed_phone_power_catalog",
    "base_executor_capabilities",
    "apply_assumed_phone_power_profile",
    "derived_desktop_gpu_first_layer",
    "desktop_model_composite_capability",
    "desktop_operator_placements",
    "materialize_desktop_control",
    "materialize_whole_model_endpoint",
    "merge_runtime_capability_catalogs",
    "measured_desktop_control_profile",
    "measured_ffn_assist_profile",
    "model_composite_capabilities",
    "model_manifest_transition_capabilities",
    "model_transition_capabilities",
    "resource_profiles_for_catalog",
]
