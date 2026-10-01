"""Portable heterogeneous execution lifecycle for scheduler tickets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
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
from .._internal.decode_split_selection import ShareBinding
from .._internal.runtime_resources import host_share_release_lower_bound_bytes
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
from .co_helper_lifecycle import CoHelperLifecycle
from ..configuration.campaign import (
    DevicePowerConfiguration,
    SpeculativeRowsModelConfiguration,
    elastic_phones_configuration,
    speculative_rows_configuration,
)
from .speculative_rows import (
    DRAFT_MODEL_PATH_PARAMETER,
    DRAFT_SHA256_PARAMETER,
    PATCHED_SERVER_PARAMETER,
    speculative_adapter_parameters,
)
from .device_power import DevicePowerController
from .energy import (
    PhoneActivityIntervalTracker,
    PolledPhonePowerSampler,
    RaplNvmlPhoneEnergyMeter,
)
from .dormant_share_coordinator import DormantShareCoordinator
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
    probe_android_phone_runtime,
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

from .heterogeneous_rig_ops.helper_masks import RigHelperMaskMixin
from .heterogeneous_rig_ops.lifecycle import RigLifecycleMixin
from .heterogeneous_rig_ops.observations import RigObservationMixin
from .heterogeneous_rig_ops.residency import RigResidencyMixin
from .heterogeneous_rig_ops.transitions import RigTransitionMixin
from .heterogeneous_rig_ops.common import (
    _LiveExecutorResidency as _LiveExecutorResidency,
    _PersistentPhoneResidency as _PersistentPhoneResidency,
    _HelperReconfiguration as _HelperReconfiguration,
    _TransitionExecutionState as _TransitionExecutionState,
)

__all__ = [
    'ANDROID_LLAMA_SERVER_ADAPTER',
    'AdaptiveDecodeError',
    'AdaptiveDecodeGroupedObservation',
    'AndroidLlamaServerProcessConfiguration',
    'AndroidLlamaServerProcessLauncher',
    'BackgroundRuntimeMonitor',
    'CanonicalHttpExecutionBackend',
    'CanonicalTransitionRegistry',
    'CapturedProcess',
    'DeviceMemoryCapacity',
    'DeviceRuntimeTelemetry',
    'DirectPhoneFfnReconfigurationReceipt',
    'DirectPhoneFfnSession',
    'DirectPhoneFfnSessionConfiguration',
    'EndpointRuntimeSample',
    'ExecutorResidencySample',
    'HeterogeneousPhysicalRig',
    'HeterogeneousRigConfiguration',
    'HostEnergySampler',
    'HostMetricCallbacks',
    'LinuxHostRuntimeProbe',
    'LlamaCppCompletionPayload',
    'LlamaCppHttpClient',
    'LlamaServerExecutionMarker',
    'LlamaServerProcessConfiguration',
    'LlamaServerProcessLauncher',
    'ManagedAndroidLlamaServer',
    'ManagedLlamaServer',
    'ModelManifest',
    'PhoneActivityIntervalTracker',
    'PhoneFfnExecutionContract',
    'PhoneRuntimeObservation',
    'PhoneSessionResidencyObservation',
    'PhysicalAdapterError',
    'PhysicalBackendFailure',
    'PhysicalExecutionCommand',
    'PhysicalPhoneSessionEndpoint',
    'PhysicalResidentEndpoint',
    'PhysicalTransitionCommand',
    'PolledPhonePowerSampler',
    'RaplNvmlPhoneEnergyMeter',
    'RawEnergyMeasurement',
    'Request',
    'RigHelperMaskMixin',
    'RigLifecycleMixin',
    'RigObservationMixin',
    'RigResidencyMixin',
    'RigTransitionMixin',
    'RuntimeActivityTracker',
    'RuntimeCapabilityCatalog',
    'RuntimeHelperExecutionEnvelope',
    'RuntimePhoneShard',
    'RuntimePlacementSnapshot',
    'RuntimeProtectedWorkObservation',
    'UnifiedRuntimeSnapshotBuilder',
    '_HelperReconfiguration',
    '_LiveExecutorResidency',
    '_PersistentPhoneResidency',
    '_TransitionExecutionState',
    'canonical_sha256',
    'catalog_preloaded_residency_samples',
    'close_functionfs_bridge',
    'live_executor_residency_sample',
    'parse_functionfs_bridge_qualification',
    'phone_ffn_resident_contract',
    'phone_transport_contract',
    'physical_residency_parameters_match',
    'physical_residency_supports_execution_plan',
    'physical_transition_stop_set',
    'probe_functionfs_usb_device',
    'probe_llama_endpoint',
    'probe_nvidia_process_memory_bytes',
    'probe_phone_power_history',
    'probe_phone_power_with_adb_fallback',
    'probe_phone_runtime_with_adb_fallback',
    'qualify_functionfs_bridge',
    'server_energy_summary',
    'validate_phone_session_replacement_command',
    'verify_android_usb_restored',
    'whole_phone_launch_parameters',
]


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
    # decode-only relocation admission: when set, every launched server is booked against this host
    # budget and a server's released FFN share must be re-reserved before it receives a prompt
    host_memory_budget_bytes: int | None = None
    dormant_share_safety_bytes: int = 256 * 1024**2
    dormant_share_hold_timeout_s: float = 900.0
    dormant_share_workspace_bytes: int = 1024**3
    co_helper_lifecycles: Mapping[str, CoHelperLifecycle] = MappingProxyType({})
    # campaign ``elastic_phones``; None keeps the static phone set
    elastic_phones: Mapping[str, object] | None = None
    # campaign ``device_power`` (the CLI mapping or the typed object); None keeps the clocks alone
    device_power: DevicePowerConfiguration | Mapping[str, object] | None = None
    # campaign ``speculative_rows`` (the CLI mapping, keyed by model id); None keeps every launch draft-free
    speculative_rows: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.catalog, RuntimeCapabilityCatalog):
            raise PhysicalAdapterError("physical rig catalog is invalid")
        try:
            elastic = elastic_phones_configuration(
                None if self.elastic_phones is None else dict(self.elastic_phones))
        except ValueError as error:
            raise PhysicalAdapterError("physical rig elastic phones: " + str(error)) from error
        object.__setattr__(self, "elastic_phones", elastic)
        object.__setattr__(self, "device_power", rig_device_power_policy(self.device_power, self.gpu_device_id))
        object.__setattr__(self, "speculative_rows", rig_speculative_rows(
            self.speculative_rows, self.catalog, dict(self.manifests)))
        if self.host_memory_budget_bytes is not None and (
            type(self.host_memory_budget_bytes) is not int or self.host_memory_budget_bytes <= 0
        ):
            raise PhysicalAdapterError("physical rig host memory budget is invalid")
        manifests = dict(self.manifests)
        known_artifacts = {row.artifact_sha256 for row in manifests.values()}
        if any(artifact not in known_artifacts or not isinstance(lifecycle, CoHelperLifecycle)
               for artifact, lifecycle in self.co_helper_lifecycles.items()):
            raise PhysicalAdapterError("physical static co-helper configuration is invalid")
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






# Elastic phones: a started co-helper is checked every interval and lost after consecutive failures
# (one adb hiccup never quarantines a phone). The check shares the phone's adb link with its FFN
# forward, so it stays sparse; a loss during a call is reported by the execution path at once.
# A worker whose own adb client has exited is lost at the first failed check: that cannot recover,
# so a second check would only delay the loss by one interval.
HELPER_LIVENESS_INTERVAL_S = 5.0
HELPER_LIVENESS_FAILURES = 2


@dataclass
class _PhoneMembership:
    """Membership of one phone under elastic phones: MEMBER, ABSENT (at start) or QUARANTINED."""

    device_id: str
    state: str
    reason: str
    since_ns: int
    readmissions: int = 0
    liveness_failures: int = 0
    last_liveness_ns: int = 0
    last_join_ns: int = 0
    join_attempts: int = 0
    last_rejection: str | None = None
    identity_sha256: str | None = None
    exhausted: bool = False
    # monotonic ns of the last readmission (JOINED); a loss reported by an older attempt is stale
    readmitted_ns: int = 0

    def to_json(self) -> dict[str, object]:
        return asdict(self)


def _catalog_speculative_drafts(catalog: RuntimeCapabilityCatalog) -> dict[str, tuple[str, str, object]]:
    """Draft bindings the catalog's executors carry, keyed by model alias: (path, digest, pin)."""
    drafts: dict[str, tuple[str, str, object]] = {}
    for row in (*catalog.executors, *catalog.composite_executors):
        parameters = row.adapter_parameters
        if DRAFT_SHA256_PARAMETER not in parameters:
            continue
        alias = parameters.get("model_alias")
        binding = (parameters.get(DRAFT_MODEL_PATH_PARAMETER), parameters[DRAFT_SHA256_PARAMETER],
                   parameters.get(PATCHED_SERVER_PARAMETER))
        if type(alias) is not str or drafts.setdefault(alias, binding) != binding:
            raise PhysicalAdapterError("physical rig catalog carries conflicting speculative drafts")
    return drafts


def rig_speculative_rows(
    value: Mapping[str, object] | None,
    catalog: RuntimeCapabilityCatalog,
    manifests: Mapping[str, ModelManifest],
) -> Mapping[str, SpeculativeRowsModelConfiguration] | None:
    """Re-validate the campaign ``speculative_rows`` mapping for this rig.

    Every named model must be loaded by the run with a draft that exists and shares its
    vocabulary, and the catalog (materialized from the same campaign) must bind exactly these
    drafts by digest: a catalog with a draft the key does not name, or the reverse, fails the
    run before any server launches. None keeps today's behaviour."""
    try:
        configuration = speculative_rows_configuration(None if value is None else dict(value), Path("/"))
    except ValueError as error:
        raise PhysicalAdapterError("physical rig speculative rows: " + str(error)) from error
    catalog_drafts = _catalog_speculative_drafts(catalog)
    if configuration is None:
        if catalog_drafts:
            raise PhysicalAdapterError("physical rig catalog carries speculative drafts without the campaign key")
        return None
    aliases = {}
    for model_id, row in configuration.items():
        manifest = manifests.get(model_id)
        if manifest is None:
            raise PhysicalAdapterError("physical rig speculative rows name a model the run does not load")
        parameters = speculative_adapter_parameters(row, manifest)
        alias = next((
            executor.adapter_parameters.get("model_alias") for executor in catalog.composite_executors
            if executor.adapter_parameters.get(DRAFT_SHA256_PARAMETER) == parameters[DRAFT_SHA256_PARAMETER]
        ), None)
        if alias is None or catalog_drafts.get(alias) != (
            parameters[DRAFT_MODEL_PATH_PARAMETER], parameters[DRAFT_SHA256_PARAMETER],
            parameters.get(PATCHED_SERVER_PARAMETER),
        ):
            raise PhysicalAdapterError(
                "physical rig catalog does not bind the speculative draft of " + model_id)
        aliases[alias] = model_id
    if set(catalog_drafts) != set(aliases):
        raise PhysicalAdapterError("physical rig catalog carries speculative drafts the campaign key does not name")
    return configuration


def rig_device_power_policy(
    value: DevicePowerConfiguration | Mapping[str, object] | None, gpu_device_id: str
) -> DevicePowerConfiguration | None:
    """Re-validate the campaign ``device_power`` mapping (or typed object) for this rig: it must
    parse exactly like the manifest field and name the rig's GPU device. None stays None."""
    if value is None:
        return None
    try:
        row = value.to_json() if isinstance(value, DevicePowerConfiguration) else dict(value)
        policy = DevicePowerConfiguration.from_json(row)
    except (TypeError, ValueError) as error:
        raise PhysicalAdapterError("physical rig device power: " + str(error)) from error
    if policy.device != gpu_device_id:
        raise PhysicalAdapterError("physical rig device power device is not the rig GPU device")
    return policy


class HeterogeneousPhysicalRig(
    RigResidencyMixin,
    RigTransitionMixin,
    RigLifecycleMixin,
    RigObservationMixin,
    RigHelperMaskMixin,
):
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
        self._scheduler = None
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
        self._device_power: DevicePowerController | None = (
            None if configuration.device_power is None else DevicePowerController(
                configuration.device_power,
                epoch_ns_provider=lambda: self.epoch_ns,
                output_directory=configuration.output_directory,
            )
        )
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
        self._co_helper_lifecycles = dict(configuration.co_helper_lifecycles)
        self._co_helper_sessions = {
            device: session for lifecycle in self._co_helper_lifecycles.values()
            for device, session in lifecycle.sessions.items()
        }
        self._co_helper_activity = {device: PhoneActivityIntervalTracker() for device in self._co_helper_sessions}
        self._co_helper_receipts = []
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
        for device, session in self._co_helper_sessions.items():
            worker = session.configuration
            runtime_probes.pop("endpoint:physical:" + device, None)
            runtime_probes["helper-runtime:" + device] = (
                lambda worker=worker: probe_android_phone_runtime(worker.serial, worker.adb_port, diagnostic=True))
        runtime_probes.update({
            "phone-allocation:" + row.executor_id:
                (lambda executor_id=row.executor_id: self._probe_phone_allocation(executor_id))
            for row in configuration.catalog.executors
            if row.adapter_parameters.get("execution_adapter") == ANDROID_LLAMA_SERVER_ADAPTER
        })
        self._init_phone_membership(configuration.elastic_phones)
        if self._elastic_phones is not None:
            for device in self._co_helper_sessions:
                runtime_probes["helper-membership:" + device] = (
                    lambda device=device: self._probe_helper_membership(device))
            if self._elastic_phones["join"]:
                runtime_probes["phone-membership:" + configuration.phone_device_id] = (
                    self._probe_primary_membership)
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
        self._active_large_since_ns: dict[str, int] = {}
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
        self._dormant_shares: DormantShareCoordinator | None = (
            None if configuration.host_memory_budget_bytes is None else DormantShareCoordinator(
                budget_bytes=configuration.host_memory_budget_bytes,
                safety_bytes=configuration.dormant_share_safety_bytes,
                pool=configuration.host_memory_resource_id,
                hold_timeout_s=configuration.dormant_share_hold_timeout_s,
            )
        )

    # ---- decode-only relocation admission (rig-level) ---------------------------------------------
    def dormant_share_coordinator(self) -> DormantShareCoordinator | None:
        # Tests build rigs without running __init__; treat a missing ledger as "no coordinator".
        return getattr(self, "_dormant_shares", None)

    def _dormant_control_ack(self, endpoint: str, request_id: str, runtime_stats) -> None:
        shares = self.dormant_share_coordinator()
        if shares is not None:
            shares.on_control_ack(endpoint, request_id, runtime_stats)

    def _dormant_before_prompt(self, endpoint: str, request_id: str) -> None:
        shares = self.dormant_share_coordinator()
        if shares is not None:
            shares.before_prompt(endpoint, request_id)

    def _dormant_book_server(self, endpoint: str, server, manifest: ModelManifest, parameters) -> None:
        """A server is READY: book its resident footprint and, when launched with the dormant host share,
        the largest FFN share it may release (the whole masked FFN, host_columns 0)."""
        shares = self.dormant_share_coordinator()
        if shares is None:
            return
        rss = 0
        try:
            for line in Path(f"/proc/{server.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                    break
        except (OSError, ValueError, AttributeError):
            rss = 0
        environment = dict(getattr(server, "environment", {}) or {})
        binding = None
        expected = 0
        if environment.get("S41_SERVER_FFN_DORMANT_HOST_SHARE") == "1":
            layer_mask = int(environment.get("S41_SERVER_FFN_LAYER_MASK", "0"))
            quantum = int((parameters or {}).get("ffn_column_quantum", 1) or 1)
            expected = host_share_release_lower_bound_bytes(manifest, layer_mask, 0)
            binding = ShareBinding(endpoint, manifest.artifact_sha256, layer_mask, 0, expected, column_quantum=quantum)
        base = max(1, rss - expected)
        shares.book_server(
            endpoint, base_bytes=base, binding=binding,
            workspace_bytes=self.configuration.dormant_share_workspace_bytes if binding is not None else 0,
            strict=False)

    def _dormant_forget_server(self, endpoint: str) -> None:
        shares = self.dormant_share_coordinator()
        if shares is not None:
            shares.forget_server(endpoint)

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
    def execution_proofs(self) -> Mapping[str, dict[str, object]]:
        with self._lock:
            return {
                ticket_id: dict(proof)
                for ticket_id, proof in self._execution_proofs.items()
            }

    @property
    def host_samples(self) -> tuple[dict[str, object], ...]:
        return self._sampler.rows()

    # ---- device power control (opt-in) ------------------------------------------------------
    def _device_power_controller(self) -> DevicePowerController | None:
        # Tests build rigs without running __init__; no controller means no power control.
        return getattr(self, "_device_power", None)

    @property
    def device_power_events(self) -> tuple[dict[str, object], ...]:
        controller = self._device_power_controller()
        return () if controller is None else controller.events

    def note_next_arrival(self, arrival_us: int | None) -> None:
        """The runner's next trace arrival (RESULT clock); None once every arrival was submitted."""
        controller = self._device_power_controller()
        if controller is not None:
            controller.note_next_arrival_us(arrival_us)

    def note_arrival_observed(self, request_id: str, model_id: str, observed_at_us: int) -> None:
        """Online device power: a trace request already arrived at ``observed_at_us``."""
        controller = self._device_power_controller()
        if controller is not None:
            controller.note_arrival_observed(request_id, model_id, observed_at_us)

    def note_first_token(self, request_id: str) -> None:
        """Device power ``protect_prefill``: the request streamed its first token."""
        controller = self._device_power_controller()
        if controller is not None:
            controller.note_first_token(request_id)

    @property
    def device_power_telemetry(self) -> dict[str, object] | None:
        controller = self._device_power_controller()
        return None if controller is None else controller.telemetry

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

    def start(
        self, warm_payload: LlamaCppCompletionPayload, *, preload_resident: bool = True
    ) -> None:
        if not isinstance(warm_payload, LlamaCppCompletionPayload):
            raise PhysicalAdapterError("physical warm payload is invalid")
        if type(preload_resident) is not bool:
            raise PhysicalAdapterError("physical resident preload mode is invalid")
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
        if self._device_power is not None:
            # probed inside the sampled window so the restore to a known state is on record
            self._device_power.capability_probe()
            self._device_power.start()
        if not preload_resident:
            self._start_runtime_monitor()
            return
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
        self._dormant_book_server(capability.endpoint, self._resident_server, manifest, {})
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
        self._start_runtime_monitor()

    def _start_runtime_monitor(self) -> None:
        self._runtime_monitor.start()
        self._runtime_monitor_started = True
        probe_names = tuple(self._runtime_monitor.probe_names)
        if not self._runtime_monitor.wait_until_populated(probe_names, 15):
            raise PhysicalAdapterError(
                "physical runtime monitor did not start"
            )

    def begin_offline_preload(self, epoch_ns: int) -> None:
        self._begin_measurement_epoch(epoch_ns)

    def begin_trace(self, epoch_ns: int) -> None:
        self._begin_measurement_epoch(epoch_ns)
        elastic = getattr(self, "_elastic_phones", None)
        # an absent co-helper is tolerated only when it may join later
        tolerate_absent = bool(elastic and elastic["join"])
        for artifact, lifecycle in getattr(self, "_co_helper_lifecycles", {}).items():
            directory = self.configuration.output_directory / ("co-helper-" + artifact[7:19])
            directory.mkdir(parents=True, exist_ok=True)
            for device in getattr(lifecycle, "sessions", {}):
                getattr(self, "_membership_join_directories", {})[device] = directory
            trackers = [getattr(self, "_co_helper_activity", {})[device]
                        for device in getattr(lifecycle, "sessions", {})]
            for tracker in trackers:
                tracker.begin("helper-startup:" + artifact, "helper_startup", time.monotonic_ns())
            try:
                self._co_helper_receipts.extend(
                    lifecycle.start_trace(directory, tolerate_absent=tolerate_absent, elastic=True)
                    if elastic is not None else lifecycle.start_trace(directory))
            finally:
                for tracker in trackers:
                    tracker.finish("helper-startup:" + artifact, time.monotonic_ns())
                self._save_co_helper_receipts()
        self._save_co_helper_receipts()
        if elastic is not None:
            self._begin_phone_membership()

    def _save_co_helper_receipts(self):
        if getattr(self, "_co_helper_lifecycles", {}):
            (self.configuration.output_directory / "CO_HELPER_LIFECYCLE.json").write_text(
                json.dumps(self._co_helper_receipts, indent=2, sort_keys=True) + "\n")

    def _stop_co_helpers(self):
        if getattr(self, "_elastic_phones", None) is not None:
            self._close_phone_membership()
            errors = []
            for lifecycle in getattr(self, "_co_helper_lifecycles", {}).values():
                try:
                    self._co_helper_receipts.extend(lifecycle.end_trace({}, continue_past_failures=True))
                except PhysicalAdapterError as error:
                    self._co_helper_receipts.extend(getattr(error, "receipts", ()))
                    errors.append(str(error))
                finally:
                    self._save_co_helper_receipts()
            if errors:
                raise PhysicalAdapterError("; ".join(errors))
            return
        for lifecycle in getattr(self, "_co_helper_lifecycles", {}).values():
            try:
                self._co_helper_receipts.extend(lifecycle.end_trace({}))
            finally:
                self._save_co_helper_receipts()

    # ---- elastic phones: membership of co-helpers and the primary phone ----------------------------
    def _init_phone_membership(self, elastic_phones: Mapping[str, object] | None) -> None:
        self._elastic_phones = elastic_phones
        self._membership_lock = threading.RLock()
        self._membership: dict[str, _PhoneMembership] = {}
        self._membership_busy = {device: threading.Lock() for device in self._co_helper_sessions}
        self._primary_membership_busy = threading.Lock()
        self._membership_log: list[dict[str, object]] = []
        # Membership calls reach the scheduler in order: queued until the scheduler is bound AND
        # every call queued before it has been delivered (``_flush_scheduler_membership``).
        self._membership_pending: list[tuple[str, str, dict[str, object]]] = []
        self._membership_scheduler = None
        self._membership_flush_lock = threading.Lock()
        self._membership_closed = False
        self._membership_join_directories: dict[str, Path] = {}

    @property
    def helper_membership_events(self) -> tuple[dict[str, object], ...]:
        with self._membership_lock:
            return tuple(dict(row) for row in self._membership_log)

    def _lifecycle_for(self, device_id: str) -> CoHelperLifecycle:
        return next(lifecycle for lifecycle in self._co_helper_lifecycles.values()
                    if device_id in lifecycle.sessions)

    def _membership_at_us(self) -> int:
        return max(0, (time.monotonic_ns() - self.epoch_ns) // 1000)

    def _membership_event(self, kind: str, device_id: str, **details: object) -> None:
        row = {"at_us": self._membership_at_us(), "device_id": device_id, "kind": kind, **details}
        with self._membership_lock:
            self._membership_log.append(row)
            (self.configuration.output_directory / "HELPER_MEMBERSHIP.json").write_text(
                json.dumps(self._membership_log, indent=2, sort_keys=True) + "\n")

    def _scheduler_membership(self, action: str, device_id: str, **arguments: object) -> None:
        """Forward a membership change to the scheduler; queued until the scheduler is bound and
        the calls queued before the binding are delivered, so none overtakes an earlier one."""
        with self._membership_lock:
            scheduler = self._membership_scheduler
            if scheduler is None:
                self._membership_pending.append((action, device_id, dict(arguments)))
                return
        self._call_scheduler_membership(scheduler, action, device_id, arguments)

    def _call_scheduler_membership(self, scheduler, action, device_id, arguments) -> None:
        try:
            getattr(scheduler, action)(device_id, **arguments)
        except Exception as error:
            self._membership_event("SCHEDULER_MEMBERSHIP_FAILED", device_id, action=action,
                                   reason=type(error).__name__ + ": " + str(error))
            raise

    def _flush_scheduler_membership(self, scheduler) -> None:
        """Deliver the queued membership calls in order, then publish the scheduler to membership.

        Calls made during the flush queue behind the delivered ones (a probe's readmission never
        overtakes the quarantine queued before the binding). A call that raises stays queued with
        every later one and the error propagates; the next binding resumes the flush. One binder
        flushes at a time; the scheduler is called without the membership lock (snapshots take it).
        """
        if not self._membership_flush_lock.acquire(blocking=False):
            return
        try:
            while True:
                with self._membership_lock:
                    if self._membership_scheduler is not None:
                        return
                    if not self._membership_pending:
                        self._membership_scheduler = scheduler
                        return
                    action, device_id, arguments = self._membership_pending[0]
                self._call_scheduler_membership(scheduler, action, device_id, arguments)
                with self._membership_lock:
                    self._membership_pending.pop(0)
        finally:
            self._membership_flush_lock.release()

    def _begin_phone_membership(self) -> None:
        now = time.monotonic_ns()
        for lifecycle in self._co_helper_lifecycles.values():
            for device_id in lifecycle.declaration.device_ids:
                member = device_id in lifecycle.started
                with self._membership_lock:
                    self._membership[device_id] = _PhoneMembership(
                        device_id, "MEMBER" if member else "ABSENT",
                        "STARTED" if member else "DEVICE_ABSENT_AT_START", now)
                if not member:
                    self._membership_event("DEVICE_ABSENT_AT_START", device_id)
                    self._scheduler_membership("quarantine_device", device_id,
                                               reason="DEVICE_ABSENT_AT_START",
                                               at_us=self._membership_at_us())

    def _close_phone_membership(self) -> None:
        """No membership change after the trace: wait for a join or liveness check in flight."""
        with self._membership_lock:
            self._membership_closed = True
        for lock in (*self._membership_busy.values(), self._primary_membership_busy):
            with lock:
                pass

    def _helper_membership_observations(self) -> dict[str, dict[str, object]]:
        """UNAVAILABLE telemetry rows for co-helpers out of the fleet: route generation keeps
        new requests off them while their device sets stay in the envelope.

        A co-helper the scheduler holds quarantined (a helper_lost failure of the execution path)
        is out at once, before its membership probe has reconciled the loss."""
        if getattr(self, "_elastic_phones", None) is None:
            return {}
        now = time.monotonic_ns()
        quarantined = getattr(self._scheduler, "quarantined_devices", None)
        scheduler_quarantined = dict(quarantined()) if callable(quarantined) else {}
        with self._membership_lock:
            states = {row.device_id: (row.state, row.reason) for row in self._membership.values()
                      if row.state != "MEMBER"}
        for device_id, reason in scheduler_quarantined.items():
            states.setdefault(device_id, ("QUARANTINED", reason))
        return {
            device_id: {
                "checked_at_ns": now,
                "failure_reason": "co-helper " + device_id + " is " + state + ": " + reason,
                "membership": state,
                "membership_reason": reason,
                "source": "helper-membership",
                "valid": False,
                "validity": "UNAVAILABLE",
            }
            for device_id, (state, reason) in sorted(states.items())
            if device_id in self._co_helper_sessions
        }

    def _probe_helper_membership(self, device_id: str) -> dict[str, object]:
        busy = self._membership_busy[device_id]
        if not busy.acquire(blocking=False):
            return {"device_id": device_id, "state": "BUSY"}
        try:
            with self._membership_lock:
                state = self._membership.get(device_id)
                closed = self._membership_closed
            if closed or state is None:
                return {"device_id": device_id, "state": "CLOSED" if closed else "PENDING"}
            self._reconcile_scheduler_quarantine(state)
            if state.state == "MEMBER":
                self._check_helper_liveness(state)
            elif self._elastic_phones["join"]:
                self._maybe_join_helper(state)
            with self._membership_lock:
                return state.to_json()
        finally:
            busy.release()

    def _reconcile_scheduler_quarantine(self, state: _PhoneMembership) -> None:
        """A quarantine the execution path reported (a helper_lost failure) releases the worker too,
        so the same join path readmits it."""
        scheduler = self._membership_scheduler
        quarantined = getattr(scheduler, "quarantined_devices", None)
        if state.state != "MEMBER" or not callable(quarantined):
            return
        reason = quarantined().get(state.device_id)
        if reason is not None:
            self._helper_lost(state, reason, notify_scheduler=False)

    def _check_helper_liveness(self, state: _PhoneMembership) -> None:
        now = time.monotonic_ns()
        if now - state.last_liveness_ns < HELPER_LIVENESS_INTERVAL_S * 1e9:
            return
        state.last_liveness_ns = now
        # A probe that raises counts as "not alive": the worker must prove it serves. The decision is
        # journaled (HELPER_MEMBERSHIP_PROBE.jsonl) so a late loss detection can be explained afterwards.
        error = None
        session = self._co_helper_sessions[state.device_id]
        try:
            alive = bool(session.alive())
        except BaseException as exc:  # noqa: BLE001 - fail closed, record the reason
            alive, error = False, f"{type(exc).__name__}: {exc}"[:200]
        exited = False
        if alive:
            state.liveness_failures = 0
        else:
            state.liveness_failures += 1
            client_exited = getattr(session, "client_exited", None)
            try:
                exited = callable(client_exited) and client_exited() is True
            except Exception:  # noqa: BLE001 - unknown: the consecutive-failure rule decides
                exited = False
        self._membership_probe_row(state, alive=alive, error=error, client_exited=exited,
                                   elapsed_us=(time.monotonic_ns() - now) // 1000)
        if not alive and (exited or state.liveness_failures >= HELPER_LIVENESS_FAILURES):
            self._helper_lost(state, "HELPER_LOST", notify_scheduler=True)

    def _membership_probe_row(self, state: _PhoneMembership, **details: object) -> None:
        """Append one liveness decision to ``HELPER_MEMBERSHIP_PROBE.jsonl`` (best effort, never raises)."""
        row = {"at_us": self._membership_at_us(), "device_id": state.device_id, "state": state.state,
               "liveness_failures": state.liveness_failures, **details}
        try:
            with self._membership_lock:
                with (self.configuration.output_directory / "HELPER_MEMBERSHIP_PROBE.jsonl").open("a") as out:
                    out.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            pass

    def _helper_lost(self, state: _PhoneMembership, reason: str, *, notify_scheduler: bool) -> None:
        with self._membership_lock:
            if state.state != "MEMBER":
                return
            state.state, state.reason, state.since_ns = "QUARANTINED", reason, time.monotonic_ns()
            state.liveness_failures = 0
        self._membership_event("HELPER_LOST", state.device_id, reason=reason)
        self._lifecycle_for(state.device_id).lose(state.device_id)
        if notify_scheduler:
            self._scheduler_membership("quarantine_device", state.device_id, reason=reason,
                                       at_us=self._membership_at_us())

    def _join_due(self, state: _PhoneMembership) -> bool:
        elastic = self._elastic_phones
        now = time.monotonic_ns()
        if state.readmissions >= elastic["max_readmissions_per_device"]:
            if not state.exhausted:
                state.exhausted = True
                self._membership_event("READMISSION_LIMIT", state.device_id,
                                       readmissions=state.readmissions)
            return False
        if (now - state.since_ns < elastic["readmission_cooldown_s"] * 1e9
                or now - state.last_join_ns < elastic["join_probe_interval_s"] * 1e9):
            return False
        state.last_join_ns = now
        state.join_attempts += 1
        return True

    def _join_rejected(self, state: _PhoneMembership, error: BaseException) -> None:
        reason = type(error).__name__ + ": " + str(error)
        if reason != state.last_rejection:
            state.last_rejection = reason
            self._membership_event("JOIN_REJECTED", state.device_id, reason=reason)

    def _maybe_join_helper(self, state: _PhoneMembership) -> None:
        lifecycle = self._lifecycle_for(state.device_id)
        if state.device_id not in lifecycle.absent or not self._join_due(state):
            return
        session_id = lifecycle._helper(state.device_id).session_id
        directory = self._membership_join_directories[state.device_id]
        log_path = directory / f"{session_id}-worker-join{state.join_attempts}.log"
        try:
            receipts = lifecycle.join(state.device_id, log_path)
        except Exception as error:  # fail closed: any failed join is a recorded rejection
            self._join_rejected(state, error)
            return
        identity = next(row["identity_sha256"] for row in receipts if row.get("kind") == "JOINED")
        with self._membership_lock:
            state.state, state.reason = "MEMBER", "READMITTED"
            state.readmissions += 1
            state.identity_sha256, state.last_rejection = identity, None
            state.liveness_failures, state.last_liveness_ns = 0, time.monotonic_ns()
            state.readmitted_ns = state.last_liveness_ns
            self._co_helper_receipts.extend(receipts)
        self._save_co_helper_receipts()
        self._membership_event("JOINED", state.device_id, identity_sha256=identity,
                               readmissions=state.readmissions)
        # a live server that masked the helper out takes it back before any policy may own it
        self._reattach_masked_helpers(state.device_id)
        self._scheduler_membership("readmit_device", state.device_id,
                                   at_us=self._membership_at_us(), identity_sha256=identity)

    def _probe_primary_membership(self) -> dict[str, object]:
        """Readmit a primary phone the execution path quarantined, once ADB answers at USB speed
        with its pinned serial and USB identity and it runs the pinned kernel; its session is
        relaunched by the normal transition path of the next phone route."""
        device_id = self.configuration.phone_device_id
        busy = self._primary_membership_busy
        if not busy.acquire(blocking=False):
            return {"device_id": device_id, "state": "BUSY"}
        try:
            return self._probe_primary_membership_once(device_id)
        finally:
            busy.release()

    def _probe_primary_membership_once(self, device_id: str) -> dict[str, object]:
        scheduler = self._membership_scheduler
        quarantined = getattr(scheduler, "quarantined_devices", None)
        with self._membership_lock:
            closed = self._membership_closed
        if closed or not callable(quarantined):
            return {"device_id": device_id, "state": "CLOSED" if closed else "PENDING"}
        reason = quarantined().get(device_id)
        with self._membership_lock:
            state = self._membership.get(device_id)
            if reason is None:
                if state is not None:
                    state.state = "MEMBER"
                return {"device_id": device_id, "state": "MEMBER"}
            if state is None or state.state == "MEMBER":
                readmissions = 0 if state is None else state.readmissions
                state = self._membership[device_id] = _PhoneMembership(
                    device_id, "QUARANTINED", reason, time.monotonic_ns(), readmissions=readmissions)
        if self._join_due(state):
            try:
                restored = verify_android_usb_restored(
                    serial=self.configuration.phone_usb_serial,
                    adb_port=self.configuration.adb_port,
                    minimum_speed_mbps=self.configuration.minimum_usb_speed_mbps,
                    timeout_s=0,
                )
            except Exception as error:  # fail closed: adb timeouts included, a recorded rejection
                self._join_rejected(state, error)
                restored = None
            kernel = None if restored is None else self._primary_kernel_identity(state)
            if kernel is not None:
                identity = canonical_sha256({
                    "device_id": device_id, "product_id": restored.product_id,
                    "serial": restored.serial, "vendor_id": restored.vendor_id,
                    **({"kernel_release": kernel} if kernel else {}),
                })
                with self._membership_lock:
                    if self._membership_closed:
                        # the trace ended during the USB check: no readmission after it
                        return {"device_id": device_id, "state": "CLOSED"}
                    state.state, state.reason = "MEMBER", "READMITTED"
                    state.readmissions += 1
                    state.identity_sha256, state.last_rejection = identity, None
                    state.readmitted_ns = time.monotonic_ns()
                self._membership_event("JOINED", device_id, identity_sha256=identity,
                                       readmissions=state.readmissions)
                self._reattach_masked_helpers(device_id)
                self._scheduler_membership("readmit_device", device_id, at_us=self._membership_at_us(),
                                           identity_sha256=identity)
        with self._membership_lock:
            return state.to_json()

    def _primary_kernel_identity(self, state: _PhoneMembership) -> str | None:
        """The pinned kernel release the primary phone runs ("" when nothing pins it), or None after
        a recorded ``JOIN_REJECTED`` (the phone stays quarantined; the next due join retries).

        The FunctionFS transport identity pins ``rig.phone.kernel_release`` and a session launch
        refuses any other kernel (``phone_session_ops/transport.py::_phone_kernel_release``, the
        same pin and the same ``uname -r``, here with stdin closed). A phone that rebooted into its
        stock kernel therefore stays quarantined instead of being readmitted into a failing
        relaunch. The boot image pin (``rig.phone.boot_image_sha256``) is not measured on the phone
        (a fastboot-booted image is not in the boot partition): the running kernel is the check.
        """
        session = getattr(getattr(self, "configuration", None), "direct_phone_session", None)
        pinned = getattr(session, "required_kernel_release", None)
        if type(pinned) is not str or not pinned:
            return ""
        try:
            completed = subprocess.run(
                [str(session.adb_path), "-P", str(session.adb_port), "-s", session.serial,
                 "shell", "uname -r"],
                stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="ascii",
                errors="backslashreplace", timeout=10, check=False,
            )
            observed = completed.stdout.strip() if completed.returncode == 0 else ""
            detail = observed or (completed.stderr.strip() or "no kernel release")[:200]
        except (OSError, subprocess.SubprocessError) as error:
            observed, detail = "", (type(error).__name__ + ": " + str(error))[:200]
        if observed == pinned:
            return pinned
        reason = "KERNEL_RELEASE_MISMATCH" if observed else "KERNEL_RELEASE_UNVERIFIED"
        if reason + ":" + detail != state.last_rejection:
            state.last_rejection = reason + ":" + detail
            self._membership_event("JOIN_REJECTED", state.device_id, reason=reason,
                                   observed=detail, pinned=pinned)
        return None

    def _stale_helper_losses(
        self, device_ids: tuple[str, ...], attempt_started_ns: int | None, ticket_id: str,
    ) -> tuple[str, ...]:
        """Lost helpers readmitted after the failed attempt started (elastic phones): the attempt
        lost the worker the device ran before its readmission, so the loss quarantines nothing; it
        is recorded as ``HELPER_LOST_STALE`` (the request is still recovered)."""
        if attempt_started_ns is None or getattr(self, "_elastic_phones", None) is None:
            return ()
        with self._membership_lock:
            stale = tuple(
                device_id for device_id in device_ids
                if (row := self._membership.get(device_id)) is not None
                and row.readmitted_ns > attempt_started_ns
            )
        for device_id in stale:
            self._membership_event("HELPER_LOST_STALE", device_id, ticket_id=ticket_id,
                                   attempt_started_us=max(0, (attempt_started_ns - self.epoch_ns) // 1000))
        return stale

    def verify_startup_parent(self, command: PhysicalExecutionCommand) -> Mapping[str, object]:
        """Report only a live endpoint whose exact warmup contract was executed."""
        with self._lock:
            resident = self._live_executors.get(command.executor_id)
            proof = self._execution_proofs.get(command.ticket_id)
            if (resident is None or resident.server.process is None
                or resident.server.process.poll() is not None or resident.generation < 1
                or resident.endpoint != command.endpoint
                or resident.manifest.artifact_sha256 != command.artifact_sha256
                or not physical_residency_parameters_match(resident.parameters, command.adapter_parameters)
                or not physical_residency_supports_execution_plan(resident.operator_plan, command.operator_plan)
                or proof is None
                or any(proof.get(key) != getattr(command, key) for key in (
                    "ticket_id", "artifact_sha256", "operator_plan_sha256", "executor_id"))):
                raise PhysicalAdapterError("startup exact parent lacks live residency and execution proof")
            return {
                "artifact_sha256": resident.manifest.artifact_sha256,
                "desktop_placement_sha256": resident.operator_plan["desktop_placement_sha256"],
                "executor_id": resident.executor_id, "endpoint": resident.endpoint,
                "generation": resident.generation, "state": "READY",
                "operator_plan_sha256": command.operator_plan_sha256,
                "adapter_parameters": dict(resident.parameters),
                "physical_execution_proof": dict(proof),
                "verified_at_us": (time.monotonic_ns() - self.epoch_ns) // 1000,
            }

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

    def end_trace(self, *, require_phone_execution: bool = True) -> None:
        self._stop_dynamic_executors(
            terminate_phone_session=True,
            require_phone_execution=require_phone_execution,
        )
        self._stop_co_helpers()
        controller = self._device_power_controller()
        if controller is not None:
            # the restore lands inside the paid window and in RESULT.device_power_events
            controller.end_trace()

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
            not phone.thermal_qualified_under(
                getattr(capability, "maximum_thermal_status", 0)
            )
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
            power = self._device_power_controller()
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
                if power is not None:
                    power.on_load_begin()
                server = self._launch_transition_server(
                    command, state, control_check
                )
                if power is not None:
                    power.on_load_end()
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
                if power is not None:
                    power.on_load_end()
                self._finish_transition_execution(state)

    def _execution_start(self, command: PhysicalExecutionCommand) -> None:
        with self._lock:
            shape = self._request_shapes.get(command.request_id)
            if command.executor_id == self.configuration.resident_executor_id:
                server = self._resident_server
                state = None
            else:
                state = self._live_executors.get(command.executor_id)
                server = None if state is None else state.server
        if command.executor_id != self.configuration.resident_executor_id:
            # drop recovery: a ticket dispatched onto a server this rig retired or reaped since
            self._require_unretired_endpoint(command.executor_id, state)
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
        remote_owner = (
            execution_contract is not None
            and execution_contract.remote_resident_ffn is not None
        )
        if (
            execution_contract is not None
            and (execution_contract.phone_device_id is not None or remote_owner)
            and self._direct_phone_session.active
        ):
            self._direct_phone_session.bind_ticket_generation(
                command.ticket_id
            )
        if (
            execution_contract is not None
            and (execution_contract.phone_device_id is not None or remote_owner
                 or whole_phone_device is not None)
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
            lifecycle = getattr(self, "_co_helper_lifecycles", {}).get(command.artifact_sha256)
            if lifecycle is not None:
                for device in lifecycle.declaration.device_ids:
                    self._co_helper_activity[device].begin(activity_id, "static_phone_execution", time.monotonic_ns())
            with self._lock:
                self._phone_execution_activity_ids.add(activity_id)
        power = self._device_power_controller()
        if power is not None:
            # outside every rig lock: a late restore runs one sudo nvidia-smi synchronously
            power.on_execution_start(command)
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
                self._active_large_since_ns[command.ticket_id] = time.monotonic_ns()

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
                self._active_large_since_ns.pop(command.ticket_id, None)
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
            lifecycle = getattr(self, "_co_helper_lifecycles", {}).get(command.artifact_sha256)
            if lifecycle is not None:
                for device in lifecycle.declaration.device_ids:
                    self._co_helper_activity[device].finish(activity_id, time.monotonic_ns(),
                                                          record=proof is None or phone_calls > 0)
        self._activity.finish(command)
        power = self._device_power_controller()
        if power is not None:
            power.on_execution_finish(command)

    def _bind_scheduler(self, scheduler):
        if not callable(getattr(scheduler, "runtime_protected_work_end_us", None)):
            raise PhysicalAdapterError("physical protected-work ledger is absent")
        power = self._device_power_controller()
        queued_start = getattr(scheduler, "runtime_queued_start_us", None)
        if power is not None and callable(queued_start):
            power.bind_queued_start_provider(queued_start)
        if getattr(self, "_elastic_phones", None) is None:
            self._scheduler = scheduler
            return
        # The protected-work ledger reads ``_scheduler`` at once; membership calls are published to
        # the scheduler only after the calls queued before this binding are delivered in order.
        self._scheduler = scheduler
        self._flush_scheduler_membership(scheduler)

    def _transition_control_available(self, capability) -> bool:
        if capability.adapter_parameters.get("android_control_transport") != "adb-ncm":
            return True
        launcher = self._android_phone_launcher
        return bool(
            launcher is not None
            and launcher.ncm_control_prepared
            and self._direct_phone_session.active
        )

    def _control_link_states(self, links):
        result = dict(links)
        for capability in self.configuration.catalog.executors:
            if self._transition_control_available(capability):
                continue
            generation = capability.adapter_parameters.get("request_transport_generation")
            for link in self.configuration.catalog.placement_profile.links:
                if link.transport_generation == generation and link.link_id in result:
                    result[link.link_id] = replace(result[link.link_id], ready=False)
        return result

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
        expired = []
        for _ in range(3):
            result = self._snapshot_once(request, model_id, observed_at_us)
            checked_at_us = max(
                observed_at_us, (time.monotonic_ns() - self.epoch_ns) // 1000)
            if checked_at_us < result.captured_at_us:
                raise PhysicalAdapterError("physical snapshot capture is in the future")
            if checked_at_us < result.valid_until_us:
                if expired:
                    observations = {key: dict(value)
                                    for key, value in result.telemetry_observations.items()}
                    observations[self.configuration.phone_device_id]["snapshot_expired_captures"] = expired
                    result = replace(result, telemetry_observations=observations)
                return result
            expired.append({"captured_at_us": result.captured_at_us,
                            "valid_until_us": result.valid_until_us,
                            "checked_at_us": checked_at_us})
            observed_at_us = checked_at_us
        raise PhysicalAdapterError("physical snapshot capture retries exhausted: " + str(expired))

    def _snapshot_once(
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
        drop_recovery = self._elastic_drop_recovery()
        if drop_recovery:
            # an exited server leaves the residency map before this snapshot reads it
            self._reap_exited_live_executors()
        host = self._host_probe.sample()
        gpu = self._gpu_memory_sample() if drop_recovery else self._sampler.latest_gpu()
        phone, phone_observation = self.phone_runtime_observation()
        helpers = {}
        for device, session in getattr(self, "_co_helper_sessions", {}).items():
            observed = self._runtime_monitor.snapshot("helper-runtime:" + device)
            raw = observed.value
            helpers[device] = (raw.value if not observed.stale and observed.error is None
                               and isinstance(raw, PhoneRuntimeObservation) and raw.to_json()["valid"] else None)
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
        if helpers:
            capacities = dict(memory.capacities)
            for device, sample in helpers.items():
                pool = self.configuration.catalog.placement_profile.devices[device].memory_pool_id
                capacity = (sample.capacity_bytes if sample else
                            self.configuration.catalog.placement_profile.memory_pools[pool].capacity_bytes)
                available = sample.available_bytes if sample else 0
                capacities[pool] = DeviceMemoryCapacity(pool, capacity, capacity - available,
                                                       min(768 * 1024**2, available))
            memory = replace(memory, capacities=capacities)
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
                if drop_recovery and process_bytes is not None:
                    # the VRAM a reap credits until the GPU samples catch up with the exit
                    with self._lock:
                        if not hasattr(self, "_executor_gpu_bytes"):
                            self._executor_gpu_bytes = {}
                        self._executor_gpu_bytes[state.executor_id] = (
                            state.server, process_bytes, time.monotonic_ns()
                        )
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
                    if device_id != self.configuration.phone_device_id and device_id not in helpers
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
                thermal_status=(None if phone is None else phone.thermal_status),
            )
        }
        for device, sample in helpers.items():
            telemetry[device] = DeviceRuntimeTelemetry(
                temperature_millic=sample.temperature_millic if sample else 0,
                battery_ppm=sample.battery_ppm if sample else 0,
                thermal_qualified=sample.thermal_qualified if sample else None,
                charging=sample.charging if sample else None,
                thermal_status=sample.thermal_status if sample else None)
            session = self._co_helper_sessions[device]
            ready = sample is not None and session.active and session._process.poll() is None
            endpoint_samples["physical:" + device] = EndpointRuntimeSample(
                "healthy" if ready else "unavailable", "live" if ready else "unavailable", 1 if ready else 0)
        protected_work = None
        protected_scope = {}
        if switching or active_large:
            protected_work, power_features = self._protected_work_observation(
                active_large, switching, captured_at_us, request_id=request.request_id)
            protected_scope = {key: value for key, value in power_features.items()
                               if key.startswith("protected_work_")}
            cost_features.update({key: value for key, value in power_features.items()
                                  if key not in protected_scope})
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
        if helpers:
            helper_residency = tuple(
                row for artifact, lifecycle in self._co_helper_lifecycles.items()
                for model in self.configuration.manifests.values() if model.artifact_sha256 == artifact
                for row in lifecycle.residency_observations(model)
                if helpers.get(row.device_id) is not None)
            result = replace(result, residency=tuple(row for row in result.residency
                                                    if row.device_id not in helpers) + helper_residency)
        return replace(result, links=self._control_link_states(result.links), telemetry_observations={
            self.configuration.phone_device_id: {
                **phone_observation,
                **({"resident_allocations": phone_allocations} if phone_allocations else {}),
                **({"protected_work_scope": protected_scope} if protected_scope else {}),
            },
            **self._helper_membership_observations(),
        })

    def request_runtime_observation_refresh(self) -> None:
        self._runtime_monitor.request_refresh("phone-runtime")
        for device in getattr(self, "_co_helper_sessions", {}):
            self._runtime_monitor.request_refresh("helper-runtime:" + device)

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
