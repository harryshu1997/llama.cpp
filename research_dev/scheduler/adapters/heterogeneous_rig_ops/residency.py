"""Physical rig: residency resources, persistent phone residency state and residency views."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import RuntimePhoneShard
from ..contracts import PhysicalAdapterError
from ..llama_server import (
    ManagedLlamaServer,
    phone_ffn_resident_contract,
    primary_phone_ffn_contract,
)
from ..phone_session import DirectPhoneFfnSession
from ..ticket import PhysicalTransitionCommand
from .common import _PersistentPhoneResidency as _PersistentPhoneResidency


class RigResidencyMixin:
    """Physical rig: residency resources, persistent phone residency state and residency views."""

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
        executions[manifest.artifact_sha256] = primary_phone_ffn_contract(
            command, phone_ffn_resident_contract(command, manifest)
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
