"""Records shared by the physical rig and its operation mixins."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import PhoneSessionResidencyObservation
from ..._internal.runtime_plan import RuntimePhoneShard
from ..android_llama_server import ManagedAndroidLlamaServer
from ..contracts import PhysicalAdapterError
from ..llama_server import ManagedLlamaServer, PhoneFfnExecutionContract
from ..phone_session import DirectPhoneFfnReconfigurationReceipt
from ..residency import PhysicalPhoneSessionEndpoint, PhysicalResidentEndpoint


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
