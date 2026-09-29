"""Portable heterogeneous execution lifecycle for scheduler tickets."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping
from urllib.parse import urlsplit

from .._internal.adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
)
from .._internal.capacity import DeviceMemoryCapacity
from .._internal.capability_contracts.executors import whole_phone_launch_parameters
from .._internal.types import canonical_sha256
from .._internal.model_manifest import ModelManifest
from .._internal.policy import Request
from .._internal.runtime_capabilities import (
    PhoneSessionResidencyObservation,
    RuntimeCapabilityCatalog,
)
from .._internal.runtime_placement import RuntimePlacementSnapshot
from .._internal.runtime_plan import (
    RuntimeHelperExecutionEnvelope,
    RuntimePhoneShard,
)
from .._internal.runtime_queue import BackgroundRuntimeMonitor
from .._internal.runtime_system_cost import RuntimeProtectedWorkObservation
from .activity import RuntimeActivityTracker
from .android_llama_server import (
    ANDROID_LLAMA_SERVER_ADAPTER,
    AndroidLlamaServerProcessConfiguration,
    AndroidLlamaServerProcessLauncher,
    ManagedAndroidLlamaServer,
)
from .bridge import (
    close_functionfs_bridge,
    parse_functionfs_bridge_qualification,
    probe_functionfs_usb_device,
    qualify_functionfs_bridge,
    verify_android_usb_restored,
)
from .contracts import (
    PhysicalAdapterError,
    PhysicalBackendFailure,
    RawEnergyMeasurement,
)
from .energy import (
    PhoneActivityIntervalTracker,
    PolledPhonePowerSampler,
    RaplNvmlPhoneEnergyMeter,
)
from .http_backend import (
    CanonicalHttpExecutionBackend,
    LlamaCppCompletionPayload,
    LlamaCppHttpClient,
)
from .llama_server import (
    LlamaServerExecutionMarker,
    LlamaServerProcessConfiguration,
    LlamaServerProcessLauncher,
    ManagedLlamaServer,
    PhoneFfnExecutionContract,
    phone_ffn_resident_contract,
)
from .phone_session import (
    DirectPhoneFfnReconfigurationReceipt,
    DirectPhoneFfnSession,
    DirectPhoneFfnSessionConfiguration,
)
from .phone_transport import phone_transport_contract
from .residency import (
    PhysicalPhoneSessionEndpoint,
    PhysicalResidentEndpoint,
    physical_residency_parameters_match,
    physical_residency_supports_execution_plan,
    physical_transition_stop_set,
)
from .probes import (
    LinuxHostRuntimeProbe,
    PhoneRuntimeObservation,
    probe_llama_endpoint,
    probe_nvidia_process_memory_bytes,
    probe_phone_power_with_adb_fallback,
    probe_phone_power_history,
    probe_phone_runtime_with_adb_fallback,
)
from .snapshot import (
    DeviceRuntimeTelemetry,
    EndpointRuntimeSample,
    ExecutorResidencySample,
    UnifiedRuntimeSnapshotBuilder,
    catalog_preloaded_residency_samples,
    live_executor_residency_sample,
)
from .ticket import (
    PhysicalExecutionCommand,
    PhysicalTransitionCommand,
    validate_phone_session_replacement_command,
)
from .transitions import CanonicalTransitionRegistry
from .host_runtime import (
    CapturedProcess,
    HostEnergySampler,
    HostMetricCallbacks,
    server_energy_summary,
)


@dataclass(frozen=True)
class HeterogeneousRigConfiguration:
    catalog: RuntimeCapabilityCatalog
    manifests: Mapping[str, ModelManifest]
    server_path: Path
    resident_server_path: Path
    model_paths_by_artifact: Mapping[str, Path]
    cuda_library_directory: Path
    resident_library_directory: Path
    bridge_path: Path
    close_helper_path: Path
    direct_phone_session: DirectPhoneFfnSessionConfiguration
    host_metrics: HostMetricCallbacks
    phone_diagnostic_endpoint: str
    phone_usb_serial: str
    phone_device_id: str
    phone_memory_resource_id: str
    gpu_device_id: str
    gpu_memory_resource_id: str
    host_memory_resource_id: str
    adb_port: int
    minimum_usb_speed_mbps: int
    output_directory: Path
    large_phase_id_by_model: Mapping[str, int]
    transition_phase_id: int
    preloaded_model_by_executor: Mapping[str, str]
    active_device_cost_features: Mapping[str, str]
    resident_model_id: str
    resident_executor_id: str
    android_phone_server: AndroidLlamaServerProcessConfiguration | None = None
    energy_attribution_kind: str = "diagnostic"

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, RuntimeCapabilityCatalog):
            raise PhysicalAdapterError("physical rig catalog is invalid")
        manifests = dict(self.manifests)
        if not manifests or any(
            model_id != manifest.model_id
            or not isinstance(manifest, ModelManifest)
            for model_id, manifest in manifests.items()
        ):
            raise PhysicalAdapterError("physical rig manifests are invalid")
        phase_ids = dict(self.large_phase_id_by_model)
        if any(
            model_id not in manifests
            or type(phase_id) is not int
            or phase_id <= 0
            for model_id, phase_id in phase_ids.items()
        ):
            raise PhysicalAdapterError("physical rig phase map is invalid")
        if (
            not self.resident_server_path.is_file()
            or not os.access(self.resident_server_path, os.X_OK)
            or not self.resident_library_directory.is_dir()
            or not self.bridge_path.is_file()
            or not os.access(self.bridge_path, os.X_OK)
            or not self.close_helper_path.is_file()
            or not isinstance(
                self.direct_phone_session,
                DirectPhoneFfnSessionConfiguration,
            )
            or not isinstance(self.host_metrics, HostMetricCallbacks)
        ):
            raise PhysicalAdapterError("physical rig executable is invalid")
        if (
            self.resident_model_id not in manifests
            or self.resident_executor_id not in self.catalog.executor_by_id
        ):
            raise PhysicalAdapterError(
                "physical rig resident endpoint is invalid"
            )
        requires_android_server = any(
            row.adapter_parameters.get("execution_adapter")
                == ANDROID_LLAMA_SERVER_ADAPTER
            for row in self.catalog.executors
        )
        if requires_android_server and not isinstance(
            self.android_phone_server,
            AndroidLlamaServerProcessConfiguration,
        ):
            raise PhysicalAdapterError(
                "physical rig Android llama-server adapter is absent"
            )
        if self.android_phone_server is not None and self.android_phone_server.control_transport == "adb-ncm":
            if (self.android_phone_server.serial != self.phone_usb_serial
                or self.android_phone_server.ncm_adb_endpoint.rsplit(":", 1)[0]
                    != urlsplit(self.phone_diagnostic_endpoint).hostname):
                raise PhysicalAdapterError("whole-phone NCM control differs from the FunctionFS device")
        if self.energy_attribution_kind not in {
            "diagnostic", "isolated", "matched_abba", "device_domain"
        }:
            raise PhysicalAdapterError(
                "physical rig energy attribution is invalid"
            )
        if (
            type(self.phone_usb_serial) is not str
            or not self.phone_usb_serial
            or not self.phone_usb_serial.isascii()
            or type(self.phone_device_id) is not str
            or self.phone_device_id not in self.catalog.placement_profile.devices
            or type(self.phone_memory_resource_id) is not str
            or self.phone_memory_resource_id
                not in self.catalog.placement_profile.memory_pools
            or self.gpu_device_id
                not in self.catalog.placement_profile.devices
            or self.gpu_memory_resource_id
                not in self.catalog.placement_profile.memory_pools
            or self.host_memory_resource_id
                not in self.catalog.placement_profile.memory_pools
            or type(self.adb_port) is not int
            or not 0 < self.adb_port <= 65535
            or type(self.minimum_usb_speed_mbps) is not int
            or self.minimum_usb_speed_mbps <= 0
            or type(self.transition_phase_id) is not int
            or self.transition_phase_id <= 0
        ):
            raise PhysicalAdapterError("physical rig USB identity is invalid")
        object.__setattr__(self, "manifests", dict(manifests))
        object.__setattr__(
            self, "large_phase_id_by_model", dict(phase_ids)
        )
        preloaded = dict(self.preloaded_model_by_executor)
        if any(
            executor_id not in self.catalog.executor_by_id
            or model_id not in manifests
            for executor_id, model_id in preloaded.items()
        ):
            raise PhysicalAdapterError("preloaded endpoint map is invalid")
        cost_features = dict(self.active_device_cost_features)
        if any(
            type(feature) is not str
            or not feature
            or not feature.isascii()
            or device_id not in self.catalog.placement_profile.devices
            for feature, device_id in cost_features.items()
        ):
            raise PhysicalAdapterError("active device feature map is invalid")
        object.__setattr__(self, "preloaded_model_by_executor", preloaded)
        object.__setattr__(self, "active_device_cost_features", cost_features)


@dataclass(frozen=True)
class _LiveExecutorResidency:
    executor_id: str
    endpoint: str
    server: ManagedLlamaServer | ManagedAndroidLlamaServer
    manifest: ModelManifest
    parameters: Mapping[str, int | str]
    operator_plan: Mapping[str, object]
    generation: int
    participant_device_ids: tuple[str, ...]
    replacement_resource_ids: tuple[str, ...]
    session_resource_ids: tuple[str, ...]
    owns_phone_session: bool = False

    def physical_identity(self) -> PhysicalResidentEndpoint:
        return PhysicalResidentEndpoint(
            executor_id=self.executor_id,
            endpoint=self.endpoint,
            artifact_sha256=self.manifest.artifact_sha256,
            generation=self.generation,
            participant_device_ids=self.participant_device_ids,
            replacement_resource_ids=self.replacement_resource_ids,
            session_resource_ids=self.session_resource_ids,
        )


@dataclass(frozen=True)
class _PersistentPhoneResidency:
    executor_id: str
    endpoint: str
    phone_shards: tuple[RuntimePhoneShard, ...]
    layout_geometry_sha256: str
    manifests_by_artifact: Mapping[str, ModelManifest]
    parameters_by_artifact: Mapping[str, Mapping[str, int | str]]
    operator_plans_by_artifact: Mapping[str, Mapping[str, object]]
    executions_by_artifact: Mapping[str, PhoneFfnExecutionContract]
    load_count_by_session: Mapping[str, int]
    column_quantum_by_session: Mapping[str, int]
    max_tokens_by_session: Mapping[str, int]
    generation: int
    participant_device_ids: tuple[str, ...]
    replacement_resource_ids: tuple[str, ...]
    session_resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        shards = tuple(sorted(
            self.phone_shards, key=lambda row: row.session_id
        ))
        sessions = {row.session_id for row in shards}
        artifacts = {
            row.artifact_sha256 for row in shards
            if row.artifact_sha256 is not None
        }
        manifests = dict(self.manifests_by_artifact)
        parameters = {
            artifact: MappingProxyType(dict(values))
            for artifact, values in self.parameters_by_artifact.items()
        }
        plans = {
            artifact: MappingProxyType(dict(values))
            for artifact, values in self.operator_plans_by_artifact.items()
        }
        executions = dict(self.executions_by_artifact)
        load_counts = dict(self.load_count_by_session)
        column_quanta = dict(self.column_quantum_by_session)
        max_tokens = dict(self.max_tokens_by_session)
        if (
            not shards
            or len(sessions) != len(shards)
            or None in {row.artifact_sha256 for row in shards}
            or set(manifests) != artifacts
            or set(parameters) != artifacts
            or set(plans) != artifacts
            or set(executions) != artifacts
            or set(load_counts) != sessions
            or set(column_quanta) != sessions
            or set(max_tokens) != sessions
            or any(
                manifest.artifact_sha256 != artifact
                for artifact, manifest in manifests.items()
            )
            or any(
                type(value) is not int or value <= 0
                for value in (
                    *load_counts.values(),
                    *column_quanta.values(),
                    *max_tokens.values(),
                )
            )
            or any(
                row.maximum_columns
                    % column_quanta.get(row.session_id, 0)
                for row in shards
            )
            or any(row.session_generation < 1 for row in shards)
            or type(self.layout_geometry_sha256) is not str
            or not self.layout_geometry_sha256.startswith("sha256:")
            or len(self.layout_geometry_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.layout_geometry_sha256[7:]
            )
        ):
            raise PhysicalAdapterError(
                "persistent phone residency layout is invalid"
            )
        object.__setattr__(self, "phone_shards", shards)
        object.__setattr__(
            self,
            "manifests_by_artifact",
            MappingProxyType(dict(sorted(manifests.items()))),
        )
        object.__setattr__(
            self,
            "parameters_by_artifact",
            MappingProxyType(dict(sorted(parameters.items()))),
        )
        object.__setattr__(
            self,
            "operator_plans_by_artifact",
            MappingProxyType(dict(sorted(plans.items()))),
        )
        object.__setattr__(
            self,
            "executions_by_artifact",
            MappingProxyType(dict(sorted(executions.items()))),
        )
        object.__setattr__(
            self,
            "load_count_by_session",
            MappingProxyType(dict(sorted(load_counts.items()))),
        )
        object.__setattr__(
            self,
            "column_quantum_by_session",
            MappingProxyType(dict(sorted(column_quanta.items()))),
        )
        object.__setattr__(
            self,
            "max_tokens_by_session",
            MappingProxyType(dict(sorted(max_tokens.items()))),
        )

    @property
    def covered_artifacts(self) -> tuple[str, ...]:
        return tuple(self.manifests_by_artifact)

    def parameters_for(
        self, artifact_sha256: str
    ) -> Mapping[str, int | str]:
        return self.parameters_by_artifact[artifact_sha256]

    def physical_identity(
        self, artifact_sha256: str | None = None
    ) -> PhysicalResidentEndpoint:
        explicit_artifact = artifact_sha256 is not None
        if artifact_sha256 is None:
            artifact_sha256 = (
                self.covered_artifacts[0]
                if len(self.covered_artifacts) == 1
                else self.layout_geometry_sha256
            )
        if (
            explicit_artifact
            and artifact_sha256 not in self.manifests_by_artifact
        ):
            raise PhysicalAdapterError(
                "persistent phone artifact is not resident"
            )
        return PhysicalResidentEndpoint(
            executor_id=self.executor_id,
            endpoint=self.endpoint,
            artifact_sha256=artifact_sha256,
            generation=self.generation,
            participant_device_ids=self.participant_device_ids,
            replacement_resource_ids=self.replacement_resource_ids,
            session_resource_ids=self.session_resource_ids,
        )

    def physical_session_identities(
        self,
    ) -> Mapping[str, PhysicalPhoneSessionEndpoint]:
        if len(self.participant_device_ids) != 1:
            raise PhysicalAdapterError(
                "persistent phone session device is ambiguous"
            )
        device_id = self.participant_device_ids[0]
        return MappingProxyType({
            row.session_id: PhysicalPhoneSessionEndpoint(
                session_id=row.session_id,
                executor_id=self.executor_id,
                endpoint=row.endpoint,
                artifact_sha256=str(row.artifact_sha256),
                resident_geometry_sha256=(
                    row.resident_geometry_sha256
                ),
                operator_plan_sha256=row.operator_plan_sha256,
                session_generation=row.session_generation,
                device_id=device_id,
                resident_bytes=row.resident_bytes,
            )
            for row in self.phone_shards
        })

    def session_residency_observations(
        self,
    ) -> tuple[PhoneSessionResidencyObservation, ...]:
        return tuple(
            PhoneSessionResidencyObservation(
                session_id=row.session_id,
                device_id=state.device_id,
                executor_id=state.executor_id,
                endpoint=state.endpoint,
                artifact_sha256=state.artifact_sha256,
                resident_geometry_sha256=(
                    state.resident_geometry_sha256
                ),
                operator_plan_sha256=state.operator_plan_sha256,
                session_generation=state.session_generation,
                resident_bytes=state.resident_bytes,
                state="READY",
            )
            for row in self.phone_shards
            for state in (
                self.physical_session_identities()[row.session_id],
            )
        )


@dataclass(frozen=True)
class _HelperReconfiguration:
    """Rollback authority for one completed helper-only phone transition."""

    transition_id: str
    changed_session_ids: tuple[str, ...]
    receipt: DirectPhoneFfnReconfigurationReceipt | None
    previous_phone_residency: _PersistentPhoneResidency | None
    fresh_start: bool = False


@dataclass
class _TransitionExecutionState:
    manifest: ModelManifest
    target_executor_id: str
    helper_only: bool
    target_devices: tuple[str, ...]
    target_replacement: tuple[str, ...]
    target_session: tuple[str, ...]
    live: dict[str, _LiveExecutorResidency]
    previous_phone_residency: _PersistentPhoneResidency | None
    phone: object
    direct_transport: object | None
    direct_phone_reused: bool
    direct_phone_reconfigurable: bool
    direct_phone_partial_requested: bool
    transition_started_ns: int
    phone_activity: object | None
    phone_activity_id: str | None
    phone_activity_started: bool = False
    mutation_started: bool = False
    direct_failure_resources: tuple[str, ...] = ()
    direct_phone_used: bool = False
    direct_phone_reconfigured: bool = False
    direct_phone_reconfiguration_receipt: object | None = None


class HeterogeneousPhysicalRig:
    """Execute exact scheduler tickets using registered capabilities."""

    def __init__(
        self,
        configuration: HeterogeneousRigConfiguration,
        *,
        epoch_ns: int,
    ) -> None:
        if not isinstance(configuration, HeterogeneousRigConfiguration):
            raise PhysicalAdapterError("physical rig configuration is invalid")
        self.configuration = configuration
        self.epoch_ns = epoch_ns
        self._snapshot_builder = UnifiedRuntimeSnapshotBuilder(
            configuration.catalog
        )
        self._host_probe = LinuxHostRuntimeProbe()
        self._activity = RuntimeActivityTracker(configuration.catalog)
        self._client = LlamaCppHttpClient()
        self._launcher = LlamaServerProcessLauncher(
            LlamaServerProcessConfiguration(
                server_path=configuration.server_path,
                model_paths_by_artifact=configuration.model_paths_by_artifact,
                library_paths_by_device={
                    "desktop-cuda": (
                        configuration.cuda_library_directory,
                    ),
                },
                executable_device_names={"desktop-cuda": "CUDA0"},
                output_directory=configuration.output_directory,
                common_library_paths=(
                    configuration.server_path.parent,
                    configuration.cuda_library_directory,
                ),
            )
        )
        self._resident_launcher = LlamaServerProcessLauncher(
            LlamaServerProcessConfiguration(
                server_path=configuration.resident_server_path,
                model_paths_by_artifact=configuration.model_paths_by_artifact,
                library_paths_by_device={
                    "desktop-cuda": (
                        configuration.resident_library_directory,
                    ),
                },
                executable_device_names={"desktop-cuda": "CUDA0"},
                output_directory=configuration.output_directory,
                common_library_paths=(
                    configuration.resident_server_path.parent,
                    configuration.resident_library_directory,
                    configuration.cuda_library_directory,
                ),
            )
        )
        self._android_phone_launcher = (
            None
            if configuration.android_phone_server is None
            else AndroidLlamaServerProcessLauncher(
                configuration.android_phone_server
            )
        )
        self._sampler = HostEnergySampler(configuration.host_metrics)
        self._phone_sampler = PolledPhonePowerSampler(
            lambda: probe_phone_power_with_adb_fallback(
                configuration.phone_diagnostic_endpoint,
                configuration.phone_usb_serial,
                configuration.adb_port,
                fallback_serial_provider=self._phone_fallback_serial,
            ),
            history_probe=lambda: probe_phone_power_history(
                configuration.phone_diagnostic_endpoint
            ),
            coverage_timeout_s=180,
        )
        self._phone_activity = PhoneActivityIntervalTracker()
        self._phone_execution_activity_ids: set[str] = set()
        self._phone_power_profile = (
            configuration.catalog.phone_power_profile_by_device.get(
                configuration.phone_device_id
            )
        )
        runtime_probes = {
            "endpoint:" + row.executor_id: (
                lambda endpoint=row.endpoint: probe_llama_endpoint(endpoint)
            )
            for row in configuration.catalog.executors
        }
        runtime_probes["phone-runtime"] = (
            lambda: probe_phone_runtime_with_adb_fallback(
                configuration.phone_diagnostic_endpoint,
                configuration.phone_usb_serial,
                configuration.adb_port,
                diagnostic=True,
                fallback_serial_provider=self._phone_fallback_serial,
            )
        )
        runtime_probes.update({
            "phone-allocation:" + row.executor_id:
                (lambda executor_id=row.executor_id: self._probe_phone_allocation(executor_id))
            for row in configuration.catalog.executors
            if row.adapter_parameters.get("execution_adapter") == ANDROID_LLAMA_SERVER_ADAPTER
        })
        self._runtime_monitor = BackgroundRuntimeMonitor(
            runtime_probes,
            refresh_interval_s=0.5,
            stale_after_s=5,
        )
        self._lock = threading.RLock()
        self._transition_lock = threading.Lock()
        self._desktop_transition_lock = threading.Lock()
        self._live_executors: dict[str, _LiveExecutorResidency] = {}
        self._resident_server: ManagedLlamaServer | None = None
        self._current_bridge: CapturedProcess | None = None
        self._current_direct_phone: DirectPhoneFfnSession | None = None
        self._direct_phone_session = DirectPhoneFfnSession(
            configuration.direct_phone_session
        )
        self._direct_phone_receipts: list[dict[str, object]] = []
        self._bridge_terminal_receipts: list[dict[str, object]] = []
        self._transport_qualifications: list[dict[str, object]] = []
        self._usb_restore_receipts: list[dict[str, object]] = []
        self._link_bandwidth_samples: dict[str, int] = {}
        self._bridge_sha256 = self._digest(configuration.bridge_path)
        self._phone_executor_id: str | None = None
        self._phone_parameters: dict[str, int | str] | None = None
        self._last_phone_parameters: dict[str, int | str] | None = None
        self._phone_residency: _PersistentPhoneResidency | None = None
        self._phone_session_active = False
        self._phone_terminal_sent = False
        self._terminal_launch_attempt = 0
        preloaded_generation = max(
            (
                sample.generation
                for sample in catalog_preloaded_residency_samples(
                    configuration.catalog,
                    configuration.manifests,
                )
            ),
            default=0,
        )
        if configuration.preloaded_model_by_executor:
            preloaded_generation = max(preloaded_generation, 1)
        self._generation = preloaded_generation
        self._launch_attempt = 0
        self._transition_active = False
        self._active_transition_count = 0
        self._server_sampler_started = False
        self._phone_sampler_started = False
        self._runtime_monitor_started = False
        self._active_large: dict[str, PhysicalExecutionCommand] = {}
        self._request_shapes: dict[str, tuple[int, int]] = {}
        self._execution_markers: dict[
            str,
            tuple[
                ManagedLlamaServer | ManagedAndroidLlamaServer,
                ModelManifest,
                LlamaServerExecutionMarker,
                tuple[int, int],
            ],
        ] = {}
        self._execution_commands: dict[str, PhysicalExecutionCommand] = {}
        self._cohort_stderr_indices: dict[str, tuple[
            ManagedLlamaServer | ManagedAndroidLlamaServer,
            str,
            tuple[str, ...],
            int,
            set[str],
        ]] = {}
        self._execution_proofs: dict[str, dict[str, object]] = {}
        self._helper_reconfigurations: dict[str, _HelperReconfiguration] = {}
        self._transition_registry = CanonicalTransitionRegistry({
            row.transition_id: self._execute_transition
            for row in configuration.catalog.transitions
        })
        self._execution_backend: CanonicalHttpExecutionBackend | None = None

    @staticmethod
    def _digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                value.update(block)
        return "sha256:" + value.hexdigest()

    def _capability(self, executor_id: str):
        capability = (
            self.configuration.catalog.executor_by_id.get(executor_id)
            or self.configuration.catalog.composite_executor_by_id.get(
                executor_id
            )
        )
        if capability is None:
            raise PhysicalAdapterError(
                "physical executor capability is absent"
            )
        return capability

    def _residency_resources(
        self, executor_id: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        capability = self._capability(executor_id)
        if hasattr(capability, "participant_device_ids"):
            device_ids = tuple(capability.participant_device_ids)
            replacement = set(
                capability.replacement_group_by_device.values()
            )
            resource_ids = tuple(capability.resource_ids)
        else:
            device_ids = (capability.device_id,)
            replacement = set()
            if capability.exclusive_residency_resource_id is not None:
                replacement.add(
                    capability.exclusive_residency_resource_id
                )
            resource_ids = tuple(capability.execution_resource_ids)
        session = set()
        if self.configuration.phone_device_id in device_ids:
            phone_resources = set(
                self.configuration.catalog.executor_by_device[
                    self.configuration.phone_device_id
                ].execution_resource_ids
            )
            session.update(
                resource_id for resource_id in resource_ids
                if resource_id in phone_resources
                or self.configuration.catalog.resources[
                    resource_id
                ].kind == "transport"
            )
        return (
            device_ids,
            tuple(sorted(replacement)),
            tuple(sorted(session)),
        )

    def _current_transition_resources(self) -> tuple[str, ...]:
        with self._lock:
            executor_id = self._phone_executor_id
        if executor_id is None:
            return ()
        capability = self._capability(executor_id)
        return (
            capability.resource_ids
            if hasattr(capability, "resource_ids")
            else capability.execution_resource_ids
        )

    def _phone_residency_resources(
        self, executor_id: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        capability = self._capability(executor_id)
        _, _, session = self._residency_resources(executor_id)
        if hasattr(capability, "replacement_group_by_device"):
            replacement = capability.replacement_group_by_device.get(
                self.configuration.phone_device_id
            )
        else:
            replacement = (
                capability.exclusive_residency_resource_id
                if capability.device_id
                    == self.configuration.phone_device_id
                else None
            )
        return (
            (self.configuration.phone_device_id,),
            (() if replacement is None else (replacement,)),
            session,
        )

    @staticmethod
    def _phone_shard_tensor_ids(
        manifest: ModelManifest,
        shards: tuple[RuntimePhoneShard, ...],
    ) -> tuple[str, ...]:
        layer_indices = {
            index
            for shard in shards
            for index in shard.layer_indices
        }
        tensor_ids = set()
        for operator in manifest.operators:
            prefix, separator, raw_index = operator.layer_id.partition(":")
            if (
                operator.kind != "ffn"
                or prefix != "layer"
                or separator != ":"
                or not raw_index.isdigit()
                or int(raw_index) not in layer_indices
            ):
                continue
            tensor_ids.update(operator.tensor_ids)
        if not tensor_ids:
            raise PhysicalAdapterError(
                "phone shard residency has no manifest tensors"
            )
        return tuple(sorted(tensor_ids))

    def _persistent_phone_residency_state(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        direct_phone: DirectPhoneFfnSession,
        *,
        fallback_generation: int,
        previous: _PersistentPhoneResidency | None,
    ) -> _PersistentPhoneResidency:
        target_shards = direct_phone.phone_shards
        if not target_shards:
            raise PhysicalAdapterError(
                "persistent phone session lacks its shard layout"
            )
        previous_by_session = (
            {} if previous is None else {
                row.session_id: row for row in previous.phone_shards
            }
        )
        shards = tuple(
            previous_by_session[row.session_id]
            if previous_by_session.get(row.session_id) == row
            else row
            for row in target_shards
        )
        artifacts = {
            row.artifact_sha256 for row in shards
            if row.artifact_sha256 is not None
        }
        manifests = (
            {} if previous is None
            else dict(previous.manifests_by_artifact)
        )
        parameters = (
            {} if previous is None
            else {
                artifact: dict(values)
                for artifact, values
                in previous.parameters_by_artifact.items()
            }
        )
        plans = (
            {} if previous is None
            else {
                artifact: dict(values)
                for artifact, values
                in previous.operator_plans_by_artifact.items()
            }
        )
        executions = (
            {} if previous is None
            else dict(previous.executions_by_artifact)
        )
        manifests[manifest.artifact_sha256] = manifest
        parameters[manifest.artifact_sha256] = dict(
            command.adapter_parameters
        )
        plans[manifest.artifact_sha256] = dict(command.operator_plan)
        executions[manifest.artifact_sha256] = phone_ffn_resident_contract(
            command, manifest
        )
        for artifact in tuple(manifests):
            if artifact not in artifacts:
                manifests.pop(artifact, None)
                parameters.pop(artifact, None)
                plans.pop(artifact, None)
                executions.pop(artifact, None)
        if set(manifests) != artifacts:
            raise PhysicalAdapterError(
                "persistent phone shard artifact evidence is absent"
            )
        geometry = command.adapter_parameters.get(
            "phone_shard_set_geometry_sha256",
            command.adapter_parameters.get(
                "ffn_resident_geometry_sha256"
            ),
        )
        generation = command.phone_layout_generation
        if generation is None:
            generation = fallback_generation
        phone_devices, phone_replacement, phone_session = (
            self._phone_residency_resources(
                command.participant.executor_id
            )
        )
        return _PersistentPhoneResidency(
            executor_id=command.participant.executor_id,
            endpoint=command.participant.endpoint,
            phone_shards=shards,
            layout_geometry_sha256=geometry,
            manifests_by_artifact=manifests,
            parameters_by_artifact=parameters,
            operator_plans_by_artifact=plans,
            executions_by_artifact=executions,
            load_count_by_session=direct_phone.load_count_by_session,
            column_quantum_by_session=(
                direct_phone.column_quantum_by_session
            ),
            max_tokens_by_session=direct_phone.max_tokens_by_session,
            generation=generation,
            participant_device_ids=phone_devices,
            replacement_resource_ids=phone_replacement,
            session_resource_ids=phone_session,
        )

    def _transport_identity(
        self,
        resource_ids: set[str] | tuple[str, ...],
        parameters: Mapping[str, int | str] | None = None,
    ) -> str:
        values = (
            self._phone_parameters
            if parameters is None else parameters
        )
        resource_id = (
            None if values is None else values.get("functionfs_resource_id")
        )
        resource = (
            None
            if type(resource_id) is not str
            else self.configuration.catalog.resources.get(resource_id)
        )
        if (
            resource is None
            or resource_id not in set(resource_ids)
            or resource.kind != "transport"
        ):
            raise PhysicalAdapterError(
                "FunctionFS transport resource differs from the ticket"
            )
        return resource.identity

    @property
    def bridge_terminal_receipts(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._bridge_terminal_receipts)

    def direct_phone_preflight(self):
        return self._direct_phone_session.preflight()

    def _phone_fallback_serial(self) -> str:
        launcher = self._android_phone_launcher
        if (launcher is not None
            and self.configuration.android_phone_server.control_transport == "adb-ncm"
            and self._direct_phone_session.active):
            launcher.connect_ncm_control()
            return launcher.control_serial
        return self.configuration.phone_usb_serial

    @property
    def android_control_events(self) -> tuple[dict[str, object], ...]:
        launcher = getattr(self, "_android_phone_launcher", None)
        return () if launcher is None else tuple(dict(row) for row in launcher.control_events)

    @property
    def transport_qualifications(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._transport_qualifications)

    @property
    def usb_restore_receipts(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._usb_restore_receipts)

    @property
    def direct_phone_receipts(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in self._direct_phone_receipts)

    @property
    def phone_residency_phase_events(
        self,
    ) -> tuple[dict[str, object], ...]:
        events = self._direct_phone_session.residency_phase_events()
        return tuple(row.to_json() for row in events)

    @property
    def phone_residency_call_events(
        self,
    ) -> tuple[dict[str, object], ...]:
        events = self._direct_phone_session.residency_call_events()
        return tuple(row.to_json() for row in events)

    @property
    def desktop_ffn_call_events(
        self,
    ) -> tuple[dict[str, object], ...]:
        """Return per-call desktop observations without changing proof data."""

        with self._lock:
            live = tuple(self._live_executors.values())
        result = []
        seen_servers = set()
        for state in live:
            server = state.server
            identity = id(server)
            if identity in seen_servers or not isinstance(
                server, ManagedLlamaServer
            ):
                continue
            seen_servers.add(identity)
            for row in server.ffn_call_events():
                result.append({
                    **row,
                    "artifact_sha256": state.manifest.artifact_sha256,
                    "executor_id": state.executor_id,
                    "model_id": state.manifest.model_id,
                    "server_label": server.label,
                })
        return tuple(sorted(
            result,
            key=lambda row: (
                int(row["observed_epoch_us"]),
                str(row["server_label"]),
                int(row["line_index"]),
            ),
        ))

    @property
    def direct_phone_residency_state(self) -> Mapping[str, object]:
        direct = self._direct_phone_session
        return MappingProxyType({
            "active": direct.active,
            "load_count_by_session": dict(
                direct.load_count_by_session
            ),
            "column_quantum_by_session": dict(
                direct.column_quantum_by_session
            ),
            "phone_shards": [
                row.to_json() for row in direct.phone_shards
            ],
            "residency_generation": direct.residency_generation,
            "weight_sources": [
                row.to_json() for row in direct.weight_sources
            ],
        })

    @property
    def execution_proofs(self) -> Mapping[str, dict[str, object]]:
        with self._lock:
            return {
                ticket_id: dict(proof)
                for ticket_id, proof in self._execution_proofs.items()
            }

    @property
    def host_samples(self) -> tuple[dict[str, object], ...]:
        return self._sampler.rows()

    @property
    def host_power_diagnostics(self) -> Mapping[str, object]:
        return self._sampler.diagnostics()

    @property
    def phone_power_diagnostics(self) -> Mapping[str, object]:
        return self._phone_sampler.diagnostics()

    @property
    def adaptive_timing_events(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            backend = self._execution_backend
            return () if backend is None else backend.adaptive_timing_events

    def start(self, warm_payload: LlamaCppCompletionPayload) -> None:
        if not isinstance(warm_payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("physical warm payload is invalid")
        self._host_probe.sample()
        self._sampler.start()
        self._server_sampler_started = True
        self._phone_sampler.start()
        self._phone_sampler_started = True
        deadline = time.monotonic() + 5
        while not self._sampler.rows():
            if time.monotonic() >= deadline:
                raise PhysicalAdapterError("physical sampler did not start")
            time.sleep(0.02)
        capability = self.configuration.catalog.executor_by_id[
            self.configuration.resident_executor_id
        ]
        manifest = self.configuration.manifests[
            self.configuration.resident_model_id
        ]
        self._resident_server = self._resident_launcher.launch_capability(
            capability,
            manifest,
            label="resident-desktop-cpu",
            control_check=lambda: None,
        )
        self._client.complete(
            capability.endpoint,
            replace(
                warm_payload,
                input_tokens=1,
                output_tokens=2,
                prompt_tokens=(warm_payload.prompt_tokens[0],),
                stream_path=warm_payload.stream_path.with_name(
                    "warm-resident-desktop-cpu.raw"
                ),
                on_first_token=lambda _value: None,
            ),
            lambda: None,
        )
        for executor_id, model_id in sorted(
            self.configuration.preloaded_model_by_executor.items()
        ):
            if executor_id == self.configuration.resident_executor_id:
                continue
            phone = self.configuration.catalog.executor_by_id[executor_id]
            if not probe_llama_endpoint(phone.endpoint).ready:
                continue
            self._client.complete(
                phone.endpoint,
                replace(
                    warm_payload,
                    input_tokens=1,
                    output_tokens=2,
                    prompt_tokens=(warm_payload.prompt_tokens[0],),
                    stream_path=warm_payload.stream_path.with_name(
                        "warm-" + executor_id.replace(":", "-") + ".raw"
                    ),
                    on_first_token=lambda _value: None,
                ),
                lambda: None,
            )
        self._runtime_monitor.start()
        self._runtime_monitor_started = True
        probe_names = tuple(self._runtime_monitor.probe_names)
        if not self._runtime_monitor.wait_until_populated(probe_names, 15):
            raise PhysicalAdapterError(
                "physical runtime monitor did not start"
            )

    def _begin_measurement_epoch(self, epoch_ns: int) -> None:
        if type(epoch_ns) is not int or epoch_ns <= 0:
            raise PhysicalAdapterError(
                "physical measurement epoch is invalid"
            )
        with self._lock:
            if self._transition_active:
                raise PhysicalAdapterError(
                    "physical measurement epoch changed during transition"
                )
            self._execution_backend = None
        self.epoch_ns = epoch_ns

    def begin_offline_preload(self, epoch_ns: int) -> None:
        self._begin_measurement_epoch(epoch_ns)

    def begin_trace(self, epoch_ns: int) -> None:
        self._begin_measurement_epoch(epoch_ns)

    def restart_desktop_campaign(self) -> Mapping[str, object]:
        """Stop dynamic desktop endpoints while preserving phone residency."""

        before = dict(self.direct_phone_residency_state)
        if not before["active"]:
            raise PhysicalAdapterError(
                "desktop campaign restart requires active phone residency"
            )
        with self._lock:
            if self._transition_active or self._execution_markers:
                raise PhysicalAdapterError(
                    "desktop campaign restart requires an idle rig"
                )
        started_ns = time.monotonic_ns()
        self._stop_dynamic_executors(terminate_phone_session=False)
        with self._lock:
            self._execution_backend = None
        finished_ns = time.monotonic_ns()
        after = dict(self.direct_phone_residency_state)
        if (
            not after["active"]
            or after["phone_shards"] != before["phone_shards"]
            or after["load_count_by_session"]
                != before["load_count_by_session"]
            or after["column_quantum_by_session"]
                != before["column_quantum_by_session"]
            or after["residency_generation"]
                != before["residency_generation"]
        ):
            raise PhysicalAdapterError(
                "desktop campaign restart changed phone residency"
            )
        return MappingProxyType({
            "finished_epoch_ns": finished_ns,
            "phone_residency_after": after,
            "phone_residency_before": before,
            "started_epoch_ns": started_ns,
        })

    def trace_energy(
        self, start_ns: int, end_ns: int
    ) -> RawEnergyMeasurement:
        return RaplNvmlPhoneEnergyMeter(
            self._sampler.latest_rows,
            server_energy_summary,
            self._phone_sampler,
            energy_boundary_id=(
                self.configuration.catalog.placement_profile.energy_boundary_id
            ),
            attribution_kind=self.configuration.energy_attribution_kind,
            server_window_rows=self._sampler.rows_between,
            phone_power_profile=self._phone_power_profile,
            phone_activity=(
                self._phone_activity
                if self._phone_power_profile is not None else None
            ),
        ).measure(start_ns, end_ns)

    def end_trace(self, *, require_phone_execution: bool = True) -> None:
        self._stop_dynamic_executors(
            terminate_phone_session=True,
            require_phone_execution=require_phone_execution,
        )

    def _finish_direct_phone(
        self,
        direct_phone: DirectPhoneFfnSession,
        *,
        require_execution: bool = True,
    ) -> None:
        receipt = direct_phone.finish(require_execution=require_execution)
        with self._lock:
            row = receipt.to_json()
            self._direct_phone_receipts.append(row)
            self._usb_restore_receipts.append(dict(row["restoration"]))
            self._phone_session_active = False
            self._last_phone_parameters = None
            self._phone_terminal_sent = True
            self._phone_executor_id = None
            self._phone_parameters = None
            self._phone_residency = None

    def _abort_direct_phone(
        self, direct_phone: DirectPhoneFfnSession
    ) -> None:
        restoration = direct_phone.abort()
        with self._lock:
            restored = restoration.to_json()
            if restored not in self._usb_restore_receipts:
                self._usb_restore_receipts.append(restored)
            self._phone_session_active = False
            self._last_phone_parameters = None
            self._phone_terminal_sent = True
            self._phone_executor_id = None
            self._phone_parameters = None
            self._phone_residency = None

    def _close_bridge(self, *, terminate_phone_session: bool) -> None:
        with self._lock:
            bridge = self._current_bridge
            parameters = getattr(
                self,
                "_phone_parameters",
                getattr(self, "_current_parameters", None),
            )
            phone_session_active = self._phone_session_active
            last_phone_parameters = self._last_phone_parameters
            phone_terminal_sent = self._phone_terminal_sent
        direct_session = getattr(self, "_direct_phone_session", None)
        if (
            bridge is None
            and direct_session is not None
            and direct_session.active
        ):
            if terminate_phone_session:
                self._finish_direct_phone(direct_session)
            return
        if (
            bridge is None
            and terminate_phone_session
            and parameters is not None
            and parameters.get("ffn_transport") == "functionfs-usb"
        ):
            restoration = verify_android_usb_restored(
                serial=self.configuration.phone_usb_serial,
                adb_port=self.configuration.adb_port,
                minimum_speed_mbps=(
                    self.configuration.minimum_usb_speed_mbps
                ),
            )
            with self._lock:
                restored = restoration.to_json()
                if restored not in self._usb_restore_receipts:
                    self._usb_restore_receipts.append(restored)
                self._phone_session_active = False
                self._last_phone_parameters = None
                self._phone_terminal_sent = True
                self._phone_executor_id = None
                self._phone_parameters = None
                self._phone_residency = None
            return
        if terminate_phone_session and phone_terminal_sent:
            restoration = verify_android_usb_restored(
                serial=self.configuration.phone_usb_serial,
                adb_port=self.configuration.adb_port,
                minimum_speed_mbps=(
                    self.configuration.minimum_usb_speed_mbps
                ),
            )
            with self._lock:
                restored = restoration.to_json()
                if restored not in self._usb_restore_receipts:
                    self._usb_restore_receipts.append(restored)
                self._phone_session_active = False
                self._last_phone_parameters = None
                self._phone_executor_id = None
                self._phone_parameters = None
                self._phone_residency = None
            return
        if bridge is None:
            if not terminate_phone_session or not phone_session_active:
                return
            parameters = last_phone_parameters
            if parameters is None:
                raise PhysicalAdapterError(
                    "physical phone session contract is absent"
                )
            with self._lock:
                self._terminal_launch_attempt += 1
                terminal_attempt = self._terminal_launch_attempt
            label = (
                "bridge-terminal-"
                + str(self._launch_attempt)
                + "-"
                + str(terminal_attempt)
            )
            bridge = CapturedProcess(
                [
                    str(self.configuration.bridge_path),
                    str(parameters["ffn_bridge_host"]),
                    str(parameters["ffn_bridge_port"]),
                    str(parameters["bridge_allocator"]),
                ],
                os.environ.copy(),
                self.configuration.output_directory,
                label,
            )
            bridge.start()
            bridge.wait_stderr("[ffn-dmabuf-bridge] ready", 30)
        if parameters is None:
            raise PhysicalAdapterError("physical bridge contract is absent")

        def request_shutdown() -> int:
            command = [
                sys.executable,
                str(self.configuration.close_helper_path),
                "--port", str(parameters["ffn_bridge_port"]),
                "--layer-mask", str(parameters["ffn_layer_mask"]),
                "--n-embd", str(parameters["ffn_n_embd"]),
                "--columns", str(parameters["ffn_columns"]),
                "--activation", str(parameters["ffn_activation"]),
            ]
            if terminate_phone_session:
                command.append("--terminate-session")
            result = subprocess.run(
                command,
                check=False,
                timeout=60,
            )
            return result.returncode

        if bridge.process is None:
            raise PhysicalAdapterError("physical bridge process is absent")
        try:
            receipt = close_functionfs_bridge(
                poll=bridge.process.poll,
                stderr_lines=lambda: tuple(bridge.stderr_lines),
                request_shutdown=request_shutdown,
                wait_for_exit=lambda: bridge.process.wait(timeout=30),
                finalize=bridge.terminate,
            )
            restoration = None
            continued_usb = None
            if terminate_phone_session:
                identity = self._transport_identity(
                    self._current_transition_resources()
                )
                row = {
                    "allocator": receipt.allocator,
                    "bridge_binary_sha256": self._bridge_sha256,
                    "functionfs_identity": identity,
                    "queue_depth": int(parameters.get(
                        "bridge_queue_depth", 1
                    )),
                    "phone_session_terminated": True,
                    "post_close_functionfs": None,
                    "terminal": receipt.to_json(),
                }
                with self._lock:
                    self._bridge_terminal_receipts.append(row)
                    self._phone_terminal_sent = True
                restoration = verify_android_usb_restored(
                    serial=self.configuration.phone_usb_serial,
                    adb_port=self.configuration.adb_port,
                    minimum_speed_mbps=(
                        self.configuration.minimum_usb_speed_mbps
                    ),
                )
            else:
                deadline = time.monotonic() + 60
                last_error = "FunctionFS did not re-enable"
                while time.monotonic() < deadline:
                    try:
                        observation = probe_functionfs_usb_device(
                            vendor_id=str(
                                parameters["functionfs_vendor_id"]
                            ),
                            product_id=str(
                                parameters["functionfs_product_id"]
                            ),
                        )
                    except PhysicalAdapterError as error:
                        last_error = str(error)
                        time.sleep(0.25)
                        continue
                    if observation.negotiated_speed_mbps < (
                            self.configuration.minimum_usb_speed_mbps):
                        last_error = "FunctionFS USB speed is below minimum"
                        time.sleep(0.25)
                        continue
                    continued_usb = observation
                    break
                if continued_usb is None:
                    raise PhysicalAdapterError(last_error)
                identity = self._transport_identity(
                    self._current_transition_resources()
                )
                row = {
                    "allocator": receipt.allocator,
                    "bridge_binary_sha256": self._bridge_sha256,
                    "functionfs_identity": identity,
                    "queue_depth": int(parameters.get(
                        "bridge_queue_depth", 1
                    )),
                    "phone_session_terminated": False,
                    "post_close_functionfs": continued_usb.to_json(),
                    "terminal": receipt.to_json(),
                }
            with self._lock:
                if not terminate_phone_session:
                    self._bridge_terminal_receipts.append(row)
                if restoration is not None:
                    self._usb_restore_receipts.append(
                        restoration.to_json()
                    )
                    self._phone_session_active = False
                    self._last_phone_parameters = None
                    self._phone_executor_id = None
                    self._phone_parameters = None
                    self._phone_residency = None
                else:
                    self._phone_session_active = True
                    self._last_phone_parameters = dict(parameters)
        finally:
            with self._lock:
                if self._current_bridge is bridge:
                    self._current_bridge = None

    def _stop_phone_session(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            direct_phone = self._current_direct_phone
            if terminate_phone_session and any(
                row.parameters.get("android_control_transport") == "adb-ncm"
                for row in getattr(self, "_live_executors", {}).values()
            ):
                raise PhysicalAdapterError("FunctionFS transport is still leased by a whole-phone endpoint")
        if direct_phone is not None:
            if not terminate_phone_session:
                return
            try:
                if allow_incomplete_direct_phone:
                    self._abort_direct_phone(direct_phone)
                else:
                    self._finish_direct_phone(
                        direct_phone,
                        require_execution=require_phone_execution,
                    )
            finally:
                with self._lock:
                    if self._current_direct_phone is direct_phone:
                        self._current_direct_phone = None
        else:
            self._close_bridge(
                terminate_phone_session=terminate_phone_session
            )

    def _stop_executor(
        self,
        executor_id: str,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            state = self._live_executors.pop(executor_id, None)
            owns_phone = (
                state.owns_phone_session
                if state is not None
                else self._phone_executor_id == executor_id
            )
        if state is None and not owns_phone:
            return
        if state is not None:
            try:
                state.server.stop()
            except BaseException:
                if state.parameters.get("android_control_transport") == "adb-ncm":
                    with self._lock:
                        self._live_executors.setdefault(executor_id, state)
                raise
        if owns_phone:
            self._stop_phone_session(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )

    def _stop_dynamic_executors(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            executor_ids = tuple(sorted(self._live_executors, key=lambda key:
                self._live_executors[key].parameters.get("android_control_transport") != "adb-ncm"
            )) if terminate_phone_session else tuple(self._live_executors)
            phone_executor_id = self._phone_executor_id
        for executor_id in executor_ids:
            self._stop_executor(
                executor_id,
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )
        with self._lock:
            phone_still_active = self._phone_executor_id is not None
        if phone_executor_id is not None and phone_still_active:
            self._stop_phone_session(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )

    def _stop_current(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        """Compatibility entry point for callers predating the residency map."""
        if hasattr(self, "_live_executors"):
            self._stop_dynamic_executors(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )
            return
        with self._lock:
            server = getattr(self, "_current_server", None)
            direct_phone = getattr(self, "_current_direct_phone", None)
        if server is not None:
            server.stop()
        if direct_phone is not None:
            if allow_incomplete_direct_phone:
                self._abort_direct_phone(direct_phone)
            else:
                self._finish_direct_phone(
                    direct_phone,
                    require_execution=require_phone_execution,
                )
            with self._lock:
                self._current_direct_phone = None
        else:
            self._close_bridge(
                terminate_phone_session=terminate_phone_session
            )
        with self._lock:
            self._current_executor_id = None
            self._current_manifest = None
            self._current_parameters = None
            self._current_operator_plan = None

    def _start_bridge(
        self,
        command: PhysicalTransitionCommand,
    ) -> CapturedProcess:
        parameters = command.adapter_parameters
        output = self.configuration.output_directory
        label = "bridge-" + str(self._launch_attempt)
        environment = os.environ.copy()
        environment["S42_FFN_BRIDGE_QUALIFICATION_CLIENTS"] = "1"
        environment["S42_FFN_BRIDGE_SHUTDOWN_CLIENTS"] = "1"
        process = CapturedProcess(
            [
                str(self.configuration.bridge_path),
                str(parameters["ffn_bridge_host"]),
                str(parameters["ffn_bridge_port"]),
                str(parameters["bridge_allocator"]),
            ],
            environment,
            output,
            label,
        )
        process.start()
        try:
            self._qualify_started_bridge(process, command)
        except BaseException:
            process.terminate()
            raise
        return process

    def _qualify_started_bridge(
        self,
        process: CapturedProcess,
        command: PhysicalTransitionCommand,
    ) -> None:
        parameters = command.adapter_parameters
        process.wait_stderr("[ffn-dmabuf-bridge] ready", 30)
        usb = probe_functionfs_usb_device(
            vendor_id=str(parameters["functionfs_vendor_id"]),
            product_id=str(parameters["functionfs_product_id"]),
        )
        client_receipt = qualify_functionfs_bridge(
            host=str(parameters["ffn_bridge_host"]),
            port=int(parameters["ffn_bridge_port"]),
            layer_mask=int(parameters["ffn_layer_mask"]),
            n_embd=int(parameters["ffn_n_embd"]),
            columns=int(parameters["ffn_columns"]),
            max_tokens=int(parameters["bridge_max_tokens"]),
            activation=str(parameters["ffn_activation"]),
            token_shapes=(1, min(32, int(parameters["bridge_max_tokens"]))),
            repeats=4,
            timeout_s=120,
        )
        process.wait_stderr("FFNDMABUFQUAL ", 120)
        bridge_receipt = parse_functionfs_bridge_qualification(
            tuple(process.stderr_lines)
        )
        client_values = client_receipt.to_json()
        if (
            bridge_receipt["calls"] != client_values["calls"]
            or bridge_receipt["upload_bytes"]
                != client_values["request_payload_bytes"]
            or bridge_receipt["download_bytes"]
                != client_values["response_payload_bytes"]
            or bridge_receipt["allocator"]
                != parameters["bridge_allocator"]
        ):
            raise PhysicalAdapterError(
                "FunctionFS qualification endpoints disagree"
            )
        h2d_bandwidth = int(
            float(bridge_receipt["h2d_payload_bytes_per_s"])
        )
        d2h_bandwidth = int(float(
            bridge_receipt[
                "d2h_conservative_payload_bytes_per_s"
            ]
        ))
        if h2d_bandwidth <= 0 or d2h_bandwidth <= 0:
            raise PhysicalAdapterError(
                "FunctionFS qualification bandwidth is invalid"
            )
        composite = self.configuration.catalog.composite_executor_by_id.get(
            command.participant.executor_id
        )
        used_resources = set(
            command.participant.resource_ids
            if composite is None else composite.resource_ids
        )
        identity = self._transport_identity(
            used_resources, command.adapter_parameters
        )
        link_bandwidth = {}
        for link in self.configuration.catalog.placement_profile.links:
            if "link:" + link.link_id not in used_resources:
                continue
            if link.target_device == self.configuration.phone_device_id:
                observed_bandwidth = h2d_bandwidth
            elif link.source_device == self.configuration.phone_device_id:
                observed_bandwidth = d2h_bandwidth
            else:
                continue
            link_bandwidth[link.link_id] = min(
                link.bandwidth_bytes_per_s, observed_bandwidth
            )
        if not link_bandwidth:
            raise PhysicalAdapterError(
                "FunctionFS qualification has no catalog transport link"
            )
        qualification = {
            "allocator": bridge_receipt["allocator"],
            "bridge": dict(bridge_receipt),
            "bridge_binary_sha256": self._bridge_sha256,
            "client": client_values,
            "cost_estimator_bandwidth_bytes_per_s": dict(sorted(
                link_bandwidth.items()
            )),
            "functionfs_identity": identity,
            "queue_depth": int(parameters["bridge_queue_depth"]),
            "usb": usb.to_json(),
        }
        with self._lock:
            self._transport_qualifications.append(qualification)
            self._link_bandwidth_samples.update(link_bandwidth)

    def _transition_manifest(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
    ) -> ModelManifest:
        validate_phone_session_replacement_command(command)
        if not isinstance(payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("transition payload is invalid")
        manifest = next((
            row
            for row in self.configuration.manifests.values()
            if row.artifact_sha256 == command.artifact_sha256
        ), None)
        if manifest is None:
            raise PhysicalAdapterError(
                "transition model artifact is not registered"
            )
        return manifest

    def _reject_legacy_transition(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        control_check: Callable[[], None],
    ) -> None:
        if hasattr(self, "_live_executors"):
            return
        with self._lock:
            executor_id = getattr(self, "_current_executor_id", None)
            legacy_manifest = getattr(self, "_current_manifest", None)
            generation = getattr(self, "_generation", -1)
        artifact = getattr(legacy_manifest, "artifact_sha256", None)
        if any(
            eviction.artifact_sha256 != artifact
            or eviction.generation != generation
            or (
                eviction.executor_id is not None
                and eviction.executor_id != executor_id
            )
            for eviction in command.transition.evictions
        ):
            raise PhysicalAdapterError(
                "transition eviction source differs from physical endpoint"
            )
        control_check()
        raise PhysicalAdapterError(
            "legacy physical transition lacks a capability registry"
        )

    def _begin_transition_execution(
        self,
        command: PhysicalTransitionCommand,
        manifest: ModelManifest,
        transition_started_ns: int,
    ) -> _TransitionExecutionState | None:
        target_executor_id = command.participant.executor_id
        helper_only = bool(getattr(command, "helper_only", False))
        resources = (
            self._phone_residency_resources(target_executor_id)
            if helper_only
            else self._residency_resources(target_executor_id)
        )
        target_devices, target_replacement, target_session = resources
        with self._lock:
            live = dict(self._live_executors)
            previous_phone = getattr(self, "_phone_residency", None)
            current = live.get(target_executor_id)
            if current is not None and (
                current.manifest == manifest
                and physical_residency_parameters_match(
                    current.parameters, command.adapter_parameters
                )
                and physical_residency_supports_execution_plan(
                    current.operator_plan, command.operator_plan
                )
                and current.server.process is not None
                and current.server.process.poll() is None
            ):
                return None
        phone = command.adapter_parameters.get("phone_device_id")
        direct_transport = (
            None
            if phone is None
            else phone_transport_contract(command.adapter_parameters)
        )
        direct_phone_reused = False
        direct_phone_reconfigurable = False
        direct_phone_partial_requested = False
        if (
            direct_transport is not None
            and direct_transport.transport == "functionfs-usb"
            and self._direct_phone_session.active
        ):
            direct_phone_reused = self._direct_phone_session.supports(
                command, manifest, direct_transport
            )
            direct_phone_reconfigurable = (
                helper_only
                and not direct_phone_reused
                and self._direct_phone_session
                    .supports_partial_reconfiguration(
                        command, manifest, direct_transport
                    )
            )
            direct_phone_partial_requested = bool(
                helper_only
                and not direct_phone_reused
                and command.transition.changed_phone_session_ids
            )
        ticket_id = getattr(command, "ticket_id", None)
        state = _TransitionExecutionState(
            manifest=manifest,
            target_executor_id=target_executor_id,
            helper_only=helper_only,
            target_devices=target_devices,
            target_replacement=target_replacement,
            target_session=target_session,
            live=live,
            previous_phone_residency=previous_phone,
            phone=phone,
            direct_transport=direct_transport,
            direct_phone_reused=direct_phone_reused,
            direct_phone_reconfigurable=direct_phone_reconfigurable,
            direct_phone_partial_requested=direct_phone_partial_requested,
            transition_started_ns=transition_started_ns,
            phone_activity=getattr(self, "_phone_activity", None),
            phone_activity_id=(
                None if ticket_id is None
                else "transition:" + ticket_id
            ),
        )
        self._begin_transition_phone_activity(command, state)
        return state

    def _whole_phone_device(self, command) -> str | None:
        parameters = command.adapter_parameters
        if parameters.get("execution_adapter") != ANDROID_LLAMA_SERVER_ADAPTER:
            return None
        participant = getattr(command, "participant", None)
        executor_id = command.executor_id if participant is None else participant.executor_id
        endpoint = command.endpoint if participant is None else participant.endpoint
        capability = self.configuration.catalog.executor_by_id.get(executor_id)
        phone = parameters.get("gpu_device_id")
        if (
            capability is None
            or phone != self.configuration.phone_device_id
            or capability.device_id != phone
            or capability.endpoint != endpoint
            or capability.adapter_parameters.get("execution_adapter") != ANDROID_LLAMA_SERVER_ADAPTER
            or capability.adapter_parameters.get("gpu_device_id") != phone
            or (participant is not None and participant.device_id != phone)
        ):
            raise PhysicalAdapterError("whole-phone physical device differs from the ticket")
        return phone

    def _begin_transition_phone_activity(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
    ) -> None:
        contract = getattr(command, "execution_contract", None)
        phone = None if contract is None else contract.phone_device_id or self._whole_phone_device(command)
        if (
            state.phone_activity is None
            or state.phone_activity_id is None
            or contract is None
            or phone is None
            or phone not in command.transition.prepares_device_ids
            or state.direct_phone_reused
        ):
            return
        state.phone_activity.begin(
            state.phone_activity_id,
            "endpoint_preparation",
            state.transition_started_ns,
        )
        state.phone_activity_started = True

    def _transition_conflicting_executors(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
    ) -> tuple[str, ...]:
        physical_live = {
            executor_id: row.physical_identity()
            for executor_id, row in state.live.items()
        }
        previous_phone = state.previous_phone_residency
        if (
            previous_phone is not None
            and previous_phone.executor_id not in physical_live
        ):
            eviction_artifacts = {
                row.artifact_sha256
                for row in command.transition.evictions
                if row.device_id == self.configuration.phone_device_id
            }
            artifact = (
                next(iter(eviction_artifacts))
                if len(eviction_artifacts) == 1 else None
            )
            physical_live[previous_phone.executor_id] = (
                previous_phone.physical_identity(artifact)
            )
        if (
            state.direct_phone_partial_requested
            and not state.direct_phone_reconfigurable
        ):
            raise PhysicalAdapterError(
                "exact partial phone residency transition is unavailable"
            )
        phone_sessions = (
            {}
            if previous_phone is None
            else dict(previous_phone.physical_session_identities())
        )
        if state.direct_phone_partial_requested:
            eviction_sessions = tuple(sorted(
                row.session_id
                for row in command.transition.evictions
                if row.session_id is not None
            ))
            expected_eviction_sessions = tuple(
                session_id
                for session_id in (
                    command.transition.changed_phone_session_ids
                )
                if session_id in phone_sessions
            )
            if eviction_sessions != expected_eviction_sessions:
                raise PhysicalAdapterError(
                    "partial phone transition lacks exact session eviction"
                )
        if state.direct_phone_reconfigurable and previous_phone is not None:
            # Session identities authorize this diff; the retained router is not evicted.
            physical_live.pop(previous_phone.executor_id, None)
        conflicts = physical_transition_stop_set(
            physical_live,
            target_artifact_sha256=command.artifact_sha256,
            target_executor_id=state.target_executor_id,
            target_endpoint=command.participant.endpoint,
            target_replacement_resource_ids=state.target_replacement,
            target_session_resource_ids=state.target_session,
            evictions=command.transition.evictions,
            phone_sessions=phone_sessions,
        )
        if state.direct_phone_reconfigurable:
            conflicts = tuple(
                executor_id for executor_id in conflicts
                if executor_id != self._phone_executor_id
            )
        return conflicts

    def _start_transition_mutation(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        conflicts = self._transition_conflicting_executors(
            command, state
        )
        self._validate_phone_transition_observation(command, state, control_check)
        control_check()
        with self._lock:
            self._launch_attempt += 1
        state.mutation_started = True
        for executor_id in conflicts:
            self._stop_executor(
                executor_id, terminate_phone_session=False
            )
        control_check()

    def _validate_phone_transition_observation(self, command, state, control_check):
        phone_device_id = state.phone or self._whole_phone_device(command)
        if phone_device_id is None or state.direct_phone_reused:
            return
        deadline = time.monotonic() + 15
        while True:
            phone, observation = self.phone_runtime_observation()
            if phone is not None:
                break
            control_check()
            if time.monotonic() >= deadline:
                raise PhysicalAdapterError(
                    "PHONE_TELEMETRY_UNAVAILABLE: " + str(observation["failure_reason"])
                )
            time.sleep(0.05)
        capability = self.configuration.catalog.executor_by_device[phone_device_id]
        if (
            not phone.thermal_qualified
            or phone.temperature_millic > capability.maximum_temperature_millic
            or phone.battery_ppm < capability.minimum_battery_ppm
        ):
            raise PhysicalAdapterError("phone transition live safety check failed")
        changed = set(command.transition.changed_phone_session_ids)
        required = sum(row.resident_bytes for row in command.transition.phone_shards
                       if not changed or row.session_id in changed)
        reclaimed = sum(row.resident_bytes for row in command.transition.evictions
                        if row.device_id == phone_device_id)
        workspace = sum(
            int(row["required_bytes"])
            for row in command.operator_plan.get("memory_demands", ())
            if row.get("kind") == "workspace"
            and row.get("resource_id") == capability.memory_resource_id
        )
        if not command.transition.phone_shards:
            required = sum(
                int(row["required_bytes"])
                for row in command.operator_plan.get("memory_demands", ())
                if row.get("resource_id") == capability.memory_resource_id
            )
            peak = command.adapter_parameters.get("whole_model_peak_memory_bytes")
            if peak is not None:
                if type(peak) is not int or peak <= 0:
                    raise PhysicalAdapterError("whole-model peak memory is invalid")
                required = max(peak, required)
            workspace = 0
        available = max(0, phone.available_bytes - min(768 << 20, phone.available_bytes))
        if required + workspace > available + reclaimed:
            raise PhysicalAdapterError("phone transition live memory capacity is insufficient")
        with self._lock:
            self._direct_phone_receipts.append({
                "kind": "phone_transition_admission",
                "ticket_id": command.ticket_id,
                "observation": observation,
                "required_bytes": required + workspace,
                "reclaimed_bytes": reclaimed,
                "available_bytes_after_reserve": available,
            })

    def _direct_phone_failure_resources(
        self, command: PhysicalTransitionCommand, phone: str
    ) -> tuple[str, ...]:
        capability = (
            self.configuration.catalog.executor_by_device[phone]
        )
        transition_resources = set(command.transition.resource_ids)
        resources = tuple(sorted(
            (
                set(capability.execution_resource_ids)
                | {
                    resource_id
                    for resource_id in transition_resources
                    if self.configuration.catalog.resources[
                        resource_id
                    ].kind == "transport"
                }
            )
            & transition_resources
        ))
        if not resources:
            raise PhysicalAdapterError(
                "direct phone failure domain is absent"
            )
        return resources

    def _prepare_functionfs_phone(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        direct_phone = self._direct_phone_session
        state.direct_failure_resources = (
            self._direct_phone_failure_resources(
                command, str(state.phone)
            )
        )
        if state.direct_phone_reused:
            direct_phone.bind(
                command, state.manifest, state.direct_transport
            )
        elif state.direct_phone_reconfigurable:
            receipt = direct_phone.reconfigure(
                command, state.manifest, state.direct_transport
            )
            state.direct_phone_reconfiguration_receipt = receipt
            state.direct_phone_reconfigured = True
            with self._lock:
                self._direct_phone_receipts.append({
                    **receipt.to_json(),
                    "command_changed_session_ids": list(
                        command.transition.changed_phone_session_ids
                    ),
                    "phone_layout_generation": (
                        command.phone_layout_generation
                    ),
                    "transition_id": command.transition.transition_id,
                })
        else:
            if direct_phone.active:
                self._stop_phone_session(
                    terminate_phone_session=True
                )
            android_launcher = getattr(self, "_android_phone_launcher", None)
            if android_launcher is not None:
                android_launcher.prepare_ncm_control(direct_phone.configuration.functionfs_gadget_path)
            direct_phone.start(
                command,
                state.manifest,
                state.direct_transport,
                control_check=control_check,
            )
        state.direct_phone_used = True
        with self._lock:
            self._current_direct_phone = direct_phone
            self._phone_executor_id = state.target_executor_id
            self._phone_parameters = dict(command.adapter_parameters)
            self._last_phone_parameters = dict(
                command.adapter_parameters
            )
            self._phone_session_active = True
            self._phone_terminal_sent = False

    def _prepare_transition_phone(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ) -> None:
        if state.phone is None:
            return
        assert state.direct_transport is not None
        if state.direct_transport.transport == "functionfs-usb":
            self._prepare_functionfs_phone(
                command, state, control_check
            )
            return
        bridge = self._start_bridge(command)
        with self._lock:
            self._current_bridge = bridge
            self._phone_executor_id = state.target_executor_id
            self._phone_parameters = dict(command.adapter_parameters)
            self._last_phone_parameters = dict(
                command.adapter_parameters
            )
            self._phone_session_active = True
            self._phone_terminal_sent = False

    def _publish_helper_transition(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
    ) -> None:
        if not state.direct_phone_used:
            raise PhysicalAdapterError(
                "request helper transition lacks direct phone"
            )
        direct_phone = self._direct_phone_session
        with self._lock:
            self._generation += 1
            generation = self._generation
            previous_phone = self._phone_residency
            if state.direct_phone_reused and previous_phone is None:
                raise PhysicalAdapterError(
                    "persistent phone residency state is absent"
                )
            self._phone_residency = (
                previous_phone
                if state.direct_phone_reused
                else self._persistent_phone_residency_state(
                    command,
                    state.manifest,
                    direct_phone,
                    fallback_generation=generation,
                    previous=previous_phone,
                )
            )
            self._helper_reconfigurations[command.ticket_id] = (
                _HelperReconfiguration(
                    transition_id=command.transition.transition_id,
                    changed_session_ids=tuple(
                        command.transition.changed_phone_session_ids
                    ),
                    receipt=(
                        state.direct_phone_reconfiguration_receipt
                        if state.direct_phone_reconfigured else None
                    ),
                    previous_phone_residency=(
                        state.previous_phone_residency
                    ),
                    fresh_start=(
                        not state.direct_phone_reused
                        and not state.direct_phone_reconfigured
                    ),
                )
            )

    def _launch_transition_server(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        control_check: Callable[[], None],
    ):
        label = (
            "large-model-"
            + str(self._launch_attempt)
            + "-"
            + command.participant.executor_id.replace(":", "-")
        )
        if (
            command.adapter_parameters.get("execution_adapter")
            == ANDROID_LLAMA_SERVER_ADAPTER
        ):
            if self._android_phone_launcher is None:
                raise PhysicalAdapterError(
                    "Android llama-server launcher is absent"
                )
            return self._android_phone_launcher.launch(
                command,
                state.manifest,
                label=label,
                control_check=control_check,
            )
        return self._launcher.launch(
            command,
            state.manifest,
            label=label,
            control_check=control_check,
        )

    def _publish_transition_server(
        self,
        command: PhysicalTransitionCommand,
        state: _TransitionExecutionState,
        server,
    ) -> int:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._live_executors[state.target_executor_id] = (
                _LiveExecutorResidency(
                    executor_id=state.target_executor_id,
                    endpoint=command.participant.endpoint,
                    server=server,
                    manifest=state.manifest,
                    parameters=dict(command.adapter_parameters),
                    operator_plan=dict(command.operator_plan),
                    generation=generation,
                    participant_device_ids=state.target_devices,
                    replacement_resource_ids=state.target_replacement,
                    session_resource_ids=state.target_session,
                    owns_phone_session=state.direct_phone_used,
                )
            )
            if state.direct_phone_used:
                previous_phone = self._phone_residency
                if state.direct_phone_reused and previous_phone is None:
                    raise PhysicalAdapterError(
                        "persistent phone residency state is absent"
                    )
                self._phone_residency = (
                    previous_phone
                    if state.direct_phone_reused
                    else self._persistent_phone_residency_state(
                        command,
                        state.manifest,
                        self._direct_phone_session,
                        fallback_generation=generation,
                        previous=previous_phone,
                    )
                )
        return generation

    def _warm_transition_server(
        self,
        command: PhysicalTransitionCommand,
        payload: LlamaCppCompletionPayload,
        generation: int,
        control_check: Callable[[], None],
    ) -> None:
        warm_path = payload.stream_path.with_name(
            payload.stream_path.stem
            + "-transition-warm-"
            + str(generation)
            + payload.stream_path.suffix
        )
        self._client.complete(
            command.participant.endpoint,
            replace(
                payload,
                input_tokens=1,
                output_tokens=2,
                prompt_tokens=(payload.prompt_tokens[0],),
                quality_mode="accounting-only",
                stream_path=warm_path,
                on_first_token=lambda _value: None,
            ),
            control_check,
        )

    def _cleanup_transition_failure(
        self,
        state: _TransitionExecutionState,
        primary_error: BaseException,
    ) -> None:
        if not state.mutation_started:
            return
        try:
            if state.helper_only:
                if (
                    state.direct_phone_used
                    and not state.direct_phone_reused
                    and not state.direct_phone_reconfigurable
                ):
                    self._stop_phone_session(
                        terminate_phone_session=True,
                        allow_incomplete_direct_phone=True,
                    )
            else:
                self._stop_executor(
                    state.target_executor_id,
                    terminate_phone_session=(
                        not state.direct_phone_reused
                        and not state.direct_phone_reconfigurable
                    ),
                    allow_incomplete_direct_phone=True,
                )
            with self._lock:
                orphan_phone = (
                    not state.helper_only
                    and state.direct_phone_used
                    and self._phone_executor_id
                        == state.target_executor_id
                )
            if (
                orphan_phone
                and not state.direct_phone_reused
                and not state.direct_phone_reconfigurable
            ):
                self._stop_phone_session(
                    terminate_phone_session=True,
                    allow_incomplete_direct_phone=True,
                )
            elif (
                state.direct_phone_reused
                or (
                    state.direct_phone_reconfigurable
                    and not state.direct_phone_reconfigured
                )
            ):
                self._restore_previous_phone_residency(
                    state.previous_phone_residency
                )
            elif state.direct_phone_reconfigured:
                receipt = state.direct_phone_reconfiguration_receipt
                if receipt is None:
                    raise PhysicalAdapterError(
                        "partial phone transition receipt is absent"
                    )
                try:
                    self._direct_phone_session.rollback_reconfiguration(
                        receipt
                    )
                except BaseException:
                    with self._lock:
                        self._phone_residency = None
                    raise
                with self._lock:
                    self._phone_residency = replace(
                        state.previous_phone_residency,
                        load_count_by_session=(
                            self._direct_phone_session
                                .load_count_by_session
                        ),
                        column_quantum_by_session=(
                            self._direct_phone_session
                                .column_quantum_by_session
                        ),
                        max_tokens_by_session=(
                            self._direct_phone_session
                                .max_tokens_by_session
                        ),
                    )
        except BaseException as cleanup_error:
            primary_error.add_note(
                "transition cleanup failed: " + str(cleanup_error)
            )

    def _restore_previous_phone_residency(
        self,
        previous: _PersistentPhoneResidency | None,
    ) -> None:
        with self._lock:
            self._phone_residency = previous
            self._phone_executor_id = (
                None if previous is None else previous.executor_id
            )
            self._phone_parameters = (
                None
                if previous is None
                else dict(next(iter(
                    previous.parameters_by_artifact.values()
                )))
            )

    def _direct_transition_failure(
        self,
        state: _TransitionExecutionState,
        primary_error: BaseException,
    ) -> PhysicalBackendFailure | None:
        if (
            not state.direct_failure_resources
            or isinstance(primary_error, PhysicalBackendFailure)
        ):
            return None
        finished_ns = time.monotonic_ns()
        return PhysicalBackendFailure(
            str(primary_error),
            phase="direct_phone_transition",
            retry_safe=True,
            execution_started=False,
            started_us=max(
                0,
                (state.transition_started_ns - self.epoch_ns) // 1000,
            ),
            finished_us=max(
                0, (finished_ns - self.epoch_ns) // 1000
            ),
            failed_resource_ids=state.direct_failure_resources,
        )

    def _finish_transition_execution(
        self, state: _TransitionExecutionState
    ) -> None:
        if state.phone_activity_started:
            state.phone_activity.finish(
                state.phone_activity_id, time.monotonic_ns()
            )

    @contextmanager
    def _transition_scope(self, command: PhysicalTransitionCommand):
        phone_device_id = self.configuration.phone_device_id
        devices = set(command.transition.prepares_device_ids) | {
            row.device_id for row in command.transition.evictions
        }
        helper_only = bool(getattr(command, "helper_only", False))
        if helper_only and devices != {phone_device_id}:
            raise PhysicalAdapterError(
                "helper transition includes non-phone residency changes"
            )
        uses_phone = (
            phone_device_id in devices
            or command.adapter_parameters.get("phone_device_id") is not None
        )
        uses_desktop = not helper_only and (
            bool(devices - {phone_device_id}) or not uses_phone
        )
        with ExitStack() as stack:
            if uses_phone:
                stack.enter_context(self._transition_lock)
            if uses_desktop:
                stack.enter_context(self._desktop_transition_lock)
            with self._lock:
                self._active_transition_count += 1
                self._transition_active = True
            try:
                yield
            finally:
                with self._lock:
                    self._active_transition_count -= 1
                    self._transition_active = self._active_transition_count > 0

    def _execute_transition(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
        control_check: Callable[[], None],
    ) -> None:
        manifest = self._transition_manifest(command, payload)
        with self._transition_scope(command):
            transition_started_ns = time.monotonic_ns()
            self._reject_legacy_transition(
                command, manifest, control_check
            )
            state = self._begin_transition_execution(
                command, manifest, transition_started_ns
            )
            if state is None:
                return
            try:
                self._start_transition_mutation(
                    command, state, control_check
                )
                self._prepare_transition_phone(
                    command, state, control_check
                )
                if state.helper_only:
                    self._publish_helper_transition(command, state)
                    return
                server = self._launch_transition_server(
                    command, state, control_check
                )
                generation = self._publish_transition_server(
                    command, state, server
                )
                self._warm_transition_server(
                    command, payload, generation, control_check
                )
            except BaseException as primary_error:
                self._cleanup_transition_failure(state, primary_error)
                failure = self._direct_transition_failure(
                    state, primary_error
                )
                if failure is not None:
                    raise failure from primary_error
                raise
            finally:
                self._finish_transition_execution(state)

    def rollback_helper_transition(
        self, command: PhysicalTransitionCommand
    ) -> Mapping[str, object]:
        """Physically restore the source shard of one completed helper load.

        The scheduler calls this when its commit fails after the phone was
        already reconfigured. Success is verified against the captured
        source layout; any failure leaves the phone residency unknown so the
        replaced session is never published as resident again.
        """

        with self._transition_lock:
            with self._lock:
                record = self._helper_reconfigurations.pop(
                    command.ticket_id, None
                )
                direct_phone = self._current_direct_phone
            transition_id = command.transition.transition_id
            if record is None or record.transition_id != transition_id:
                raise PhysicalAdapterError(
                    "helper transition rollback has no reconfiguration receipt"
                )
            base = {
                "command_changed_session_ids": list(
                    command.transition.changed_phone_session_ids
                ),
                "kind": "helper_transition_rollback",
                "phone_layout_generation": command.phone_layout_generation,
                "schema": "research-scheduler-phone-session-rollback-v1",
                "ticket_id": command.ticket_id,
                "transition_id": transition_id,
            }
            if record.fresh_start:
                # The load created the whole phone residency; undoing it
                # means stopping that session so nothing stays resident.
                try:
                    self._stop_phone_session(
                        terminate_phone_session=True,
                        allow_incomplete_direct_phone=True,
                    )
                finally:
                    with self._lock:
                        self._phone_residency = None
                        self._phone_executor_id = None
                        self._phone_parameters = None
                receipt = {
                    **base,
                    "physical_change": True,
                    "restored_shards": [],
                }
                with self._lock:
                    self._direct_phone_receipts.append(receipt)
                return MappingProxyType(receipt)
            if record.receipt is None:
                receipt = {**base, "physical_change": False}
                with self._lock:
                    self._direct_phone_receipts.append(receipt)
                return MappingProxyType(receipt)
            previous = record.previous_phone_residency
            if (
                direct_phone is None
                or not direct_phone.active
                or previous is None
            ):
                with self._lock:
                    self._phone_residency = None
                raise PhysicalAdapterError(
                    "helper transition rollback has no active phone session"
                )
            try:
                direct_phone.rollback_reconfiguration(record.receipt)
            except BaseException:
                with self._lock:
                    self._phone_residency = None
                raise
            restored = tuple(sorted(
                direct_phone.phone_shards, key=lambda row: row.session_id
            ))
            changed_session_id = record.receipt.changed_session_id
            expected_restored = (
                previous.phone_shards
                if record.receipt.previous_shard is None else
                tuple(
                    replace(
                        row,
                        session_generation=(
                            record.receipt.target_shard.session_generation + 1
                        ),
                    )
                    if row.session_id == changed_session_id else row
                    for row in previous.phone_shards
                )
            )
            if restored != expected_restored:
                with self._lock:
                    self._phone_residency = None
                raise PhysicalAdapterError(
                    "helper transition rollback did not restore the source"
                )
            restored_shard = next((
                row for row in restored
                if row.session_id == changed_session_id
            ), None)
            with self._lock:
                self._phone_residency = replace(
                    previous,
                    phone_shards=restored,
                    load_count_by_session=(
                        direct_phone.load_count_by_session
                    ),
                    column_quantum_by_session=(
                        direct_phone.column_quantum_by_session
                    ),
                    max_tokens_by_session=(
                        direct_phone.max_tokens_by_session
                    ),
                )
                receipt = {
                    **base,
                    "changed_session_id": changed_session_id,
                    "load_count": direct_phone.load_count_by_session.get(
                        changed_session_id, 0
                    ),
                    "physical_change": True,
                    "residency_generation": direct_phone.residency_generation,
                    "restored_session_generations": {
                        row.session_id: row.session_generation
                        for row in restored
                    },
                    "restored_empty_session_ids": (
                        [changed_session_id]
                        if restored_shard is None else []
                    ),
                    "restored_shard": (
                        None if restored_shard is None else
                        restored_shard.to_json()
                    ),
                    "restored_shards": [
                        row.to_json() for row in restored
                    ],
                    "reverted_shard": record.receipt.target_shard.to_json(),
                }
                self._direct_phone_receipts.append(receipt)
            return MappingProxyType(receipt)

    def _execution_start(self, command: PhysicalExecutionCommand) -> None:
        with self._lock:
            shape = self._request_shapes.get(command.request_id)
            if command.executor_id == self.configuration.resident_executor_id:
                server = self._resident_server
                state = None
            else:
                state = self._live_executors.get(command.executor_id)
                server = None if state is None else state.server
        if shape is None:
            raise PhysicalAdapterError(
                "physical request shape observation is absent"
            )
        manifest = next((
            row for row in self.configuration.manifests.values()
            if row.artifact_sha256 == command.artifact_sha256
        ), None)
        if server is None or manifest is None:
            raise PhysicalAdapterError(
                "physical execution endpoint differs from the ticket"
            )
        if command.executor_id == self.configuration.resident_executor_id:
            resident_manifest = self.configuration.manifests[
                self.configuration.resident_model_id
            ]
            if manifest != resident_manifest:
                raise PhysicalAdapterError(
                    "resident execution artifact differs from the ticket"
                )
        elif (
            state is None
            or state.manifest != manifest
            or state.endpoint != command.endpoint
            or not physical_residency_parameters_match(
                state.parameters,
                command.adapter_parameters,
            )
            or not physical_residency_supports_execution_plan(
                state.operator_plan, command.operator_plan
            )
        ):
            raise PhysicalAdapterError(
                "physical execution residency differs from the ticket"
            )
        with self._lock:
            if (
                command.ticket_id in self._execution_markers
                or command.ticket_id in self._execution_proofs
            ):
                raise PhysicalAdapterError(
                    "physical execution proof is duplicated"
                )
        whole_phone_device = self._whole_phone_device(command)
        marker = server.begin_execution(command, manifest)
        execution_contract = getattr(command, "execution_contract", None)
        if (
            execution_contract is not None
            and execution_contract.phone_device_id is not None
            and self._direct_phone_session.active
        ):
            self._direct_phone_session.bind_ticket_generation(
                command.ticket_id
            )
        if (
            execution_contract is not None
            and (execution_contract.phone_device_id is not None or whole_phone_device is not None)
            and execution_contract.execution_mode != "adaptive-split"
            and hasattr(self, "_phone_activity")
            and hasattr(self, "_phone_execution_activity_ids")
        ):
            activity_id = "execution:" + command.ticket_id
            self._phone_activity.begin(
                activity_id,
                "phone_execution",
                time.monotonic_ns(),
            )
            with self._lock:
                self._phone_execution_activity_ids.add(activity_id)
        self._activity.start(
            command, input_tokens=shape[0], output_tokens=shape[1]
        )
        with self._lock:
            cohort = getattr(command, "decode_cohort", None)
            if cohort is not None:
                cohort_id = str(cohort["cohort_id"])
                members = tuple(cohort["member_request_ids"])
                shared = self._cohort_stderr_indices.get(cohort_id)
                if shared is None:
                    shared = (
                        server,
                        command.artifact_sha256,
                        members,
                        marker.stderr_index,
                        set(),
                    )
                    self._cohort_stderr_indices[cohort_id] = shared
                elif (
                    shared[0] is not server
                    or shared[1] != command.artifact_sha256
                    or shared[2] != members
                ):
                    raise PhysicalAdapterError(
                        "physical cohort execution marker differs"
                    )
                marker = replace(marker, stderr_index=shared[3])
            self._execution_markers[command.ticket_id] = (
                server, manifest, marker, shape
            )
            if cohort is not None:
                self._execution_commands[command.ticket_id] = command
            if command.model_id in self.configuration.large_phase_id_by_model:
                self._active_large[command.ticket_id] = command

    def _execution_success(
        self,
        command: PhysicalExecutionCommand,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        with self._lock:
            execution = self._execution_markers.get(command.ticket_id)
        if execution is None:
            raise PhysicalAdapterError(
                "physical execution proof marker is absent"
            )
        server, manifest, marker, shape = execution
        adaptive = None
        raw_adaptive = payload.get("adaptive_decode_observation")
        if raw_adaptive is not None:
            try:
                adaptive = AdaptiveDecodeGroupedObservation.from_json(
                    raw_adaptive
                )
            except (TypeError, AdaptiveDecodeError) as error:
                raise PhysicalAdapterError(
                    "adaptive execution observation is invalid"
                ) from error
        if command.helper_envelope is not None and marker.phone_contract is None:
            marker = server.bind_ready_helper(marker, command, manifest)
        helper_envelopes = payload.get("_runtime_helper_envelopes", ())
        if (
            not isinstance(helper_envelopes, tuple)
            or any(
                not isinstance(row, RuntimeHelperExecutionEnvelope)
                for row in helper_envelopes
            )
        ):
            raise PhysicalAdapterError(
                "physical helper history is invalid"
            )
        execution_proof = server.finish_execution(
            marker,
            command,
            manifest,
            output_tokens=shape[1],
            adaptive_observation=adaptive,
            static_control_ack=payload.get("static_ffn_control_ack"),
            helper_envelopes=helper_envelopes,
        )
        if execution_proof.phone_calls_by_session:
            direct_phone = self._direct_phone_session
            if not direct_phone.active:
                raise PhysicalAdapterError(
                    "phone execution proof has no active phone session"
                )
            direct_phone.record_execution_proof(
                command.ticket_id,
                command.artifact_sha256,
                execution_proof.phone_calls_by_session,
            )
        proof = execution_proof.to_json()
        with self._lock:
            if command.ticket_id in self._execution_proofs:
                raise PhysicalAdapterError(
                    "physical execution proof is duplicated"
                )
            self._execution_proofs[command.ticket_id] = proof
        return proof

    def _execution_finish(self, command: PhysicalExecutionCommand) -> None:
        with self._lock:
            marker = self._execution_markers.pop(command.ticket_id, None)
            proof = self._execution_proofs.get(command.ticket_id)
        if marker is None:
            raise PhysicalAdapterError(
                "physical execution proof marker is absent"
            )
        cohort = getattr(command, "decode_cohort", None)
        if cohort is not None:
            cohort_id = str(cohort["cohort_id"])
            with self._lock:
                recorded_command = self._execution_commands.pop(
                    command.ticket_id, None
                )
                if recorded_command != command:
                    raise PhysicalAdapterError(
                        "physical execution command changed before completion"
                    )
                shared = self._cohort_stderr_indices.get(cohort_id)
                if shared is None:
                    raise PhysicalAdapterError(
                        "physical cohort execution marker is absent"
                    )
                shared[4].add(command.request_id)
                if len(shared[4]) == len(shared[2]):
                    del self._cohort_stderr_indices[cohort_id]
        if command.model_id in self.configuration.large_phase_id_by_model:
            with self._lock:
                current = self._active_large.pop(command.ticket_id, None)
                if current != command:
                    raise PhysicalAdapterError(
                        "large-model activity completion differs"
                    )
        activity_id = "execution:" + command.ticket_id
        with self._lock:
            tracked_phone_execution = (
                activity_id in self._phone_execution_activity_ids
            )
            if tracked_phone_execution:
                self._phone_execution_activity_ids.remove(activity_id)
        if tracked_phone_execution:
            phone_calls = (
                0 if proof is None else int(proof.get("phone_call_count", 0))
            )
            whole_phone = (
                command.execution_contract.operator_kind == "whole_model"
                or self._whole_phone_device(command) is not None
            )
            self._phone_activity.finish(
                activity_id,
                time.monotonic_ns(),
                record=(proof is None or whole_phone or phone_calls > 0),
            )
        self._activity.finish(command)

    def backend(self) -> CanonicalHttpExecutionBackend:
        with self._lock:
            if self._execution_backend is None:
                energy_meter = RaplNvmlPhoneEnergyMeter(
                    self._sampler.latest_rows,
                    server_energy_summary,
                    self._phone_sampler,
                    energy_boundary_id=(
                        self.configuration.catalog.placement_profile
                            .energy_boundary_id
                    ),
                    attribution_kind=(
                        self.configuration.energy_attribution_kind
                    ),
                    server_window_rows=self._sampler.rows_between,
                    phone_power_profile=self._phone_power_profile,
                    phone_activity=(
                        self._phone_activity
                        if self._phone_power_profile is not None else None
                    ),
                )
                self._execution_backend = CanonicalHttpExecutionBackend(
                    self._client,
                    energy_meter,
                    epoch_ns=self.epoch_ns,
                    prepare_transition=self._transition_registry.execute,
                    rollback_transition=self.rollback_helper_transition,
                    on_execution_start=self._execution_start,
                    on_execution_success=self._execution_success,
                    on_execution_finish=self._execution_finish,
                )
            return self._execution_backend

    def _executor_samples(self) -> dict[str, EndpointRuntimeSample]:
        catalog = self.configuration.catalog
        unavailable = EndpointRuntimeSample(
            "unavailable", "unavailable", 0
        )
        result = {}
        transition_executors = {
            row.executor_id for row in catalog.transitions
            if row.executor_id is not None
        }
        for row in catalog.executors:
            observed = self._runtime_monitor.snapshot(
                "endpoint:" + row.executor_id
            )
            sample = (
                observed.value
                if not observed.stale
                and observed.error is None
                and isinstance(observed.value, EndpointRuntimeSample)
                else unavailable
            )
            result[row.executor_id] = (
                sample
                if sample.ready or row.executor_id not in transition_executors
                else replace(sample, transition_available=True)
            )
        with self._lock:
            live = dict(self._live_executors)
            active_large = tuple(self._active_large.values())
        for row in catalog.composite_executors:
            state = live.get(row.executor_id)
            if state is not None:
                process_alive = (
                    state.server.process is not None
                    and state.server.process.poll() is None
                )
                parallel = int(state.parameters["parallel"])
                active = sum(
                    command.executor_id == row.executor_id
                    for command in active_large
                )
                result[row.executor_id] = EndpointRuntimeSample(
                    "healthy" if process_alive else "unavailable",
                    "live" if process_alive else "unavailable",
                    max(0, parallel - active),
                    transition_available=not process_alive,
                )
            elif row.executor_id in transition_executors:
                result[row.executor_id] = EndpointRuntimeSample(
                    "unavailable",
                    "unavailable",
                    0,
                    transition_available=True,
                )
            else:
                result[row.executor_id] = EndpointRuntimeSample(
                    "unavailable", "unavailable", 0
                )
        return result

    @staticmethod
    def _phone_allocation_identity(state: _LiveExecutorResidency) -> dict[str, object]:
        server = state.server
        parameters = whole_phone_launch_parameters(state.parameters)
        if (not isinstance(server, ManagedAndroidLlamaServer) or state.generation <= 0
                or server.artifact_sha256 != state.manifest.artifact_sha256
                or server.endpoint != state.endpoint or server.launch_parameters != parameters):
            raise PhysicalAdapterError("Android allocation launch identity differs")
        return {
            "executor_id": state.executor_id, "endpoint": state.endpoint,
            "artifact_sha256": state.manifest.artifact_sha256, "generation": state.generation,
            "launch_parameters_sha256": canonical_sha256(parameters),
            "process_identity": server.process_identity.to_json(),
        }

    def _probe_phone_allocation(self, executor_id: str) -> dict[str, object]:
        with self._lock:
            state = self._live_executors.get(executor_id)
        if state is None:
            raise PhysicalAdapterError("Android allocation endpoint is not resident")
        identity = self._phone_allocation_identity(state)
        observed = state.server.observe_allocation()
        with self._lock:
            if self._live_executors.get(executor_id) is not state:
                raise PhysicalAdapterError("Android allocation residency generation changed during probe")
        return {**observed, **identity}

    def _phone_allocation_observation(self, state, occupied_bytes):
        name = "phone-allocation:" + state.executor_id
        sample = self._runtime_monitor.snapshot(name)
        raw = sample.value if isinstance(sample.value, dict) else {}
        now_ns = time.monotonic_ns()
        started = raw.get("captured_at_ns")
        finished = raw.get("finished_at_ns")
        amount = raw.get("allocated_bytes")
        reason = sample.error
        validity = "MISSING" if not raw else "UNAVAILABLE"
        if reason is not None and reason.startswith(("TimeoutExpired:", "TimeoutError:")):
            validity = "TIMED_OUT"
        if reason is None:
            try:
                identity = self._phone_allocation_identity(state)
                if any(raw.get(key) != value for key, value in identity.items()):
                    reason = "Android allocation does not match current residency"
                elif (type(started) is not int or type(finished) is not int
                      or not 0 < started <= finished <= now_ns or type(amount) is not int or amount <= 0
                      or raw.get("source") != "android-dumpsys-meminfo-pss-v1"
                      or type(raw.get("raw_sha256")) is not str or len(raw["raw_sha256"]) != 71
                      or not raw["raw_sha256"].startswith("sha256:")
                      or set(raw["raw_sha256"][7:]) - set("0123456789abcdef")):
                    reason, validity = "Android allocation measurement is malformed", "MALFORMED"
                elif sample.stale or now_ns - started >= 5_000_000_000:
                    reason, validity = "Android allocation sample expired", "STALE"
                elif amount > occupied_bytes:
                    reason = "Android allocation exceeds observed phone occupied memory"
                elif state.server.process is None or state.server.process.poll() is not None:
                    reason = "Android allocation endpoint is no longer live"
                else:
                    with self._lock:
                        if self._live_executors.get(state.executor_id) is not state:
                            reason = "Android allocation residency changed before publication"
            except PhysicalAdapterError as error:
                reason = str(error)
        if reason is not None:
            self._runtime_monitor.request_refresh(name)
        detail = {"source": "android-dumpsys-meminfo-pss-v1", **raw, "probe": name,
                  "expected_generation": state.generation,
                  "validity": "VALID" if reason is None else validity,
                  "valid": reason is None,
                  "failure_reason": reason, "checked_at_ns": now_ns,
                  "age_us": None if type(started) is not int else max(0, (now_ns - started) // 1000),
                  "maximum_age_us": 5_000_000}
        return (amount if reason is None else None), detail

    def snapshot(
        self,
        request: Request,
        model_id: str,
        observed_at_us: int,
    ):
        wake_observed_at_us = observed_at_us
        manifest = self.configuration.manifests[model_id]
        shape = (request.input_tokens, request.output_tokens)
        with self._lock:
            previous_shape = self._request_shapes.setdefault(
                request.request_id, shape
            )
        if previous_shape != shape:
            raise PhysicalAdapterError(
                "physical request shape identity changed"
            )
        host = self._host_probe.sample()
        gpu = self._sampler.latest_gpu()
        phone, phone_observation = self.phone_runtime_observation()
        captured_at_us = max(
            0, (time.monotonic_ns() - self.epoch_ns) // 1000
        )
        if captured_at_us < wake_observed_at_us:
            raise PhysicalAdapterError(
                "physical snapshot precedes its queue wake"
            )
        phone_capacity = (
            self.configuration.catalog.placement_profile.memory_pools[
                self.configuration.phone_memory_resource_id
            ].capacity_bytes
            if phone is None else phone.capacity_bytes
        )
        phone_available = 0 if phone is None else phone.available_bytes
        phone_reserve = min(768 * 1024**2, phone_available)
        memory = RuntimePlacementSnapshot(
            snapshot_id=(
                "physical-memory-"
                + request.request_id.replace(":", "-")
                + "-wake-"
                + str(wake_observed_at_us)
                + "-captured-"
                + str(captured_at_us)
            ),
            captured_at_us=captured_at_us,
            valid_until_us=captured_at_us + 2_500_000,
            capacities={
                self.configuration.host_memory_resource_id:
                    DeviceMemoryCapacity(
                    self.configuration.host_memory_resource_id,
                    host.memory_total_bytes,
                    host.memory_total_bytes - host.memory_available_bytes,
                    min(1024**3, host.memory_available_bytes),
                ),
                self.configuration.gpu_memory_resource_id:
                    DeviceMemoryCapacity(
                    self.configuration.gpu_memory_resource_id,
                    int(gpu["memory_total_bytes"]),
                    int(gpu["memory_total_bytes"])
                        - int(gpu["memory_free_bytes"]),
                    min(512 * 1024**2, int(gpu["memory_free_bytes"])),
                ),
                self.configuration.phone_memory_resource_id:
                    DeviceMemoryCapacity(
                    self.configuration.phone_memory_resource_id,
                    phone_capacity,
                    phone_capacity - phone_available,
                    phone_reserve,
                ),
                **{
                    session.memory_resource_id: DeviceMemoryCapacity(
                        session.memory_resource_id,
                        session.resident_memory_limit_bytes,
                        0,
                        0,
                    )
                    for executor in (
                        self.configuration.catalog.executors
                    )
                    for session in executor.phone_sessions
                },
            },
        )
        activity = self._activity.snapshot()
        with self._lock:
            active_large = tuple(self._active_large.values())
            switching = self._transition_active
            live_executors = tuple(self._live_executors.values())
            phone_residency = self._phone_residency
            link_bandwidth_samples = dict(self._link_bandwidth_samples)
        live_residencies = []
        phone_allocations = []
        for state in live_executors:
            reclaimable_by_device = {}
            gpu_devices = tuple(
                device_id for device_id in state.participant_device_ids
                if self.configuration.catalog.placement_profile.devices[
                    device_id
                ].kind == "gpu"
            )
            process = state.server.process
            if (
                len(gpu_devices) == 1
                and process is not None
                and process.poll() is None
            ):
                process_bytes = probe_nvidia_process_memory_bytes(process.pid)
                if (
                    process_bytes is not None
                    and process_bytes <= int(gpu["memory_used_bytes"])
                ):
                    reclaimable_by_device[gpu_devices[0]] = process_bytes
            if state.parameters.get("execution_adapter") == ANDROID_LLAMA_SERVER_ADAPTER:
                amount, allocation = self._phone_allocation_observation(
                    state, 0 if phone is None else phone_capacity - phone_available,
                )
                phone_allocations.append(allocation)
                if amount is not None:
                    reclaimable_by_device[self.configuration.phone_device_id] = amount
                    memory = replace(memory, valid_until_us=min(
                        memory.valid_until_us,
                        (allocation["captured_at_ns"] + 5_000_000_000 - self.epoch_ns) // 1000,
                    ))
            split_phone_residency = (
                phone_residency is not None
                and state.executor_id == phone_residency.executor_id
                and state.manifest.artifact_sha256
                    in phone_residency.manifests_by_artifact
            )
            if split_phone_residency:
                desktop_device_ids = tuple(
                    device_id for device_id in state.participant_device_ids
                    if device_id != self.configuration.phone_device_id
                )
                live_residency = (
                    None
                    if not desktop_device_ids
                    else ExecutorResidencySample(
                        state.manifest,
                        state.executor_id,
                        state.generation,
                        resident_device_ids=desktop_device_ids,
                        reclaimable_bytes_by_device=(
                            reclaimable_by_device
                        ),
                        operator_plan=state.operator_plan,
                    )
                )
            else:
                live_residency = live_executor_residency_sample(
                    self.configuration.catalog,
                    state.manifest,
                    state.executor_id,
                    generation=state.generation,
                    reclaimable_bytes_by_device=reclaimable_by_device,
                    operator_plan=state.operator_plan,
                )
            if live_residency is not None:
                live_residencies.append(live_residency)
        if phone_residency is not None:
            for artifact_sha256 in phone_residency.covered_artifacts:
                artifact_shards = tuple(
                    row for row in phone_residency.phone_shards
                    if row.artifact_sha256 == artifact_sha256
                )
                manifest = phone_residency.manifests_by_artifact[
                    artifact_sha256
                ]
                live_residencies.append(ExecutorResidencySample(
                    manifest,
                    phone_residency.executor_id,
                    phone_residency.generation,
                    resident_device_ids=(
                        self.configuration.phone_device_id,
                    ),
                    operator_plan=(
                        phone_residency.operator_plans_by_artifact[
                            artifact_sha256
                        ]
                    ),
                    resident_tensor_ids_by_device={
                        self.configuration.phone_device_id:
                            self._phone_shard_tensor_ids(
                                manifest, artifact_shards
                            ),
                    },
                    resident_bytes_by_device={
                        self.configuration.phone_device_id: sum(
                            row.resident_bytes for row in artifact_shards
                        ),
                    },
                    resident_geometry_sha256=(
                        phone_residency.layout_geometry_sha256
                    ),
                ))
        if switching:
            large_phase_id = self.configuration.transition_phase_id
            active_device_features = {
                feature: 0
                for feature in self.configuration.active_device_cost_features
            }
        elif active_large:
            phase_ids = {
                self.configuration.large_phase_id_by_model[row.model_id]
                for row in active_large
            }
            large_phase_id = (
                next(iter(phase_ids))
                if len(phase_ids) == 1
                else max((
                    self.configuration.transition_phase_id,
                    *self.configuration.large_phase_id_by_model.values(),
                )) + 1
            )
            active_ids = {
                participant.device_id
                for row in active_large
                for participant in row.participants
            }
            active_device_features = {
                feature: int(device_id in active_ids)
                for feature, device_id in (
                    self.configuration.active_device_cost_features.items()
                )
            }
        else:
            large_phase_id = 0
            active_device_features = {
                feature: 0
                for feature in self.configuration.active_device_cost_features
            }
        residencies = list(catalog_preloaded_residency_samples(
            self.configuration.catalog,
            self.configuration.manifests,
        ))
        by_id = self.configuration.catalog.executor_by_id
        endpoint_samples = self._executor_samples()
        for executor_id, model_id in sorted(
            self.configuration.preloaded_model_by_executor.items()
        ):
            capability = by_id.get(executor_id)
            sample = endpoint_samples.get(executor_id)
            if capability is not None and sample is not None and sample.ready:
                residencies.append(ExecutorResidencySample(
                    self.configuration.manifests[model_id], executor_id, 1
                ))
        residencies.extend(live_residencies)
        cost_features = {
            "active_model_input_tokens": sum(
                activity.active_input_tokens_by_model.get(model_id, 0)
                for model_id in self.configuration.large_phase_id_by_model
            ),
            "active_model_output_tokens": sum(
                activity.active_output_tokens_by_model.get(model_id, 0)
                for model_id in self.configuration.large_phase_id_by_model
            ),
            "active_model_requests": sum(
                activity.active_requests_by_model.get(model_id, 0)
                for model_id in self.configuration.large_phase_id_by_model
            ),
            "active_cpu_requests": activity.active_by_device_kind.get(
                "cpu", 0
            ),
            "active_cpu_slots": activity.active_by_device_kind.get("cpu", 0),
            "active_large_phase_count": len({
                self.configuration.large_phase_id_by_model[row.model_id]
                for row in active_large
            }),
            "actual_batch_size": min(request.input_tokens, 512),
            "active_request_batch_size": max(
                1,
                activity.active_requests_by_model.get(model_id, 0) + 1,
            ),
            "cpu_utilization_pct": host.cpu_utilization_pct,
            "large_phase_id": large_phase_id,
            "memory_bandwidth_pressure_basis_points": (
                host.memory_stall_avg10_basis_points
            ),
            "memory_stall_avg10_basis_points": (
                host.memory_stall_avg10_basis_points
            ),
            "prompt_ubatch_count": (request.input_tokens + 511) // 512,
        }
        cost_features.update(active_device_features)
        telemetry = {
            self.configuration.phone_device_id: DeviceRuntimeTelemetry(
                temperature_millic=(
                    0 if phone is None else phone.temperature_millic
                ),
                battery_ppm=(0 if phone is None else phone.battery_ppm),
                thermal_qualified=(
                    None if phone is None else phone.thermal_qualified
                ),
                charging=(None if phone is None else phone.charging),
            )
        }
        protected_work = None
        if switching or active_large:
            protected_work = RuntimeProtectedWorkObservation(
                observation_id="physical-current-power",
                critical_path_end_us=captured_at_us,
                phase_power_mw=int(gpu["power_mw"]),
                stranded_idle_power_mw=0,
                causal_tail_power_mw=0,
                sample_count=1,
                measured=True,
            )
        result = self._snapshot_builder.build(
            snapshot_id=(
                "physical-runtime-"
                + request.request_id.replace(":", "-")
                + "-wake-"
                + str(wake_observed_at_us)
                + "-captured-"
                + str(captured_at_us)
            ),
            captured_at_us=captured_at_us,
            valid_until_us=memory.valid_until_us,
            memory=memory,
            executor_samples=endpoint_samples,
            residencies=tuple(residencies),
            phone_session_residencies=(
                ()
                if phone_residency is None
                else phone_residency.session_residency_observations()
            ),
            device_telemetry=telemetry,
            link_bandwidth_samples=link_bandwidth_samples,
            cost_features=cost_features,
            protected_work=protected_work,
        )
        return replace(result, telemetry_observations={
            self.configuration.phone_device_id: {
                **phone_observation,
                **({"resident_allocations": phone_allocations} if phone_allocations else {}),
            },
        })

    def request_runtime_observation_refresh(self) -> None:
        self._runtime_monitor.request_refresh("phone-runtime")

    def phone_runtime_observation(self):
        observed = self._runtime_monitor.snapshot("phone-runtime")
        raw = observed.value
        if isinstance(raw, PhoneRuntimeObservation):
            detail = raw.to_json()
        else:
            detail = PhoneRuntimeObservation(
                source="phone-runtime-monitor",
                captured_at_ns=None,
                checked_at_ns=time.monotonic_ns(),
                validity="MISSING" if observed.captured_at_ns == 0 or observed.error is None else "UNAVAILABLE",
                failure_reason=observed.error or "phone observation is missing",
            ).to_json()
        if observed.stale and detail["valid"]:
            detail.update(valid=False, validity="STALE",
                          failure_reason="runtime monitor sample expired")
        detail["monitor_captured_at_ns"] = observed.captured_at_ns
        detail["monitor_error"] = observed.error
        if not detail["valid"]:
            self.request_runtime_observation_refresh()
        return (raw.value if detail["valid"] else None), detail

    def snapshot_for_ticket(self, ticket, observed_at_us: int):
        return self.snapshot(
            ticket.request, ticket.model.model_id, observed_at_us
        )

    def close(self, *, require_phone_execution: bool = True) -> None:
        cleanup_errors = []
        try:
            if require_phone_execution:
                self._stop_current()
            else:
                self._stop_current(require_phone_execution=False)
        except BaseException as error:
            cleanup_errors.append(error)
        android_launcher = getattr(self, "_android_phone_launcher", None)
        if android_launcher is not None:
            try:
                android_launcher.close_control()
            except BaseException as error:
                cleanup_errors.append(error)
        if self._resident_server is not None:
            try:
                self._resident_server.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._resident_server = None
        if self._runtime_monitor_started:
            try:
                self._runtime_monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._runtime_monitor_started = False
        if self._phone_sampler_started:
            try:
                self._phone_sampler.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._phone_sampler_started = False
        if self._server_sampler_started:
            try:
                self._sampler.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._server_sampler_started = False
        if cleanup_errors:
            raise PhysicalAdapterError(
                "physical cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
