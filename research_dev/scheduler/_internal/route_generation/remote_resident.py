"""Route generation for desktop parents whose FFN weights live only on phone sessions.

A desktop coordinator declares its remote-resident group in ``adapter_parameters`` under
``remote_resident_ffn_v1`` (the JSON of ``RuntimeRemoteResidentFfn`` with unbound sessions).
Route generation then

* folds the group into the desktop placement identity and the execution contract,
* binds the owning sessions' physical generations from the READY phone layout,
* removes the omitted tensor bytes from the desktop weight demand and adds the owning
  sessions as residency constraints,
* rejects the candidate while an owner is not READY with the declared shard geometry.

Everything fails closed: a declaration that disagrees with the model manifest, the phone
session capabilities or the shard geometry raises ``RouteGenerationError``.
"""

from __future__ import annotations

import json

from ..model_manifest import ModelManifest
from ..runtime_cost import RuntimeMemoryDemand
from ..runtime_plan import (
    RuntimePlanError,
    RuntimeRemoteResidentFfn,
    remote_resident_tensor_ids,
)
from .common import RouteGenerationError, _Pattern


REMOTE_RESIDENT_FFN_PARAMETER = "remote_resident_ffn_v1"
REMOTE_RESIDENT_OWNER_NOT_READY = "REMOTE_RESIDENT_OWNER_NOT_READY"
_DTYPE_BY_QUANTIZATION = {"F16": "f16"}


def remote_resident_link_ids(profile, parameters) -> tuple[str, ...]:
    """Remote owners need RPC links even when the parent operator rows are local."""
    if REMOTE_RESIDENT_FFN_PARAMETER not in parameters:
        return ()
    cpu, phone = parameters.get("cpu_device_id"), parameters.get("phone_device_id")
    directions = {(cpu, phone), (phone, cpu)}
    return tuple(sorted(
        row.link_id for row in profile.links
        if (row.source_device, row.target_device) in directions
        and row.ready and row.status == "measured"
        and row.transport_generation.startswith("functionfs")
    ))


def parse_remote_resident_declaration(value: object) -> RuntimeRemoteResidentFfn:
    """Decode the coordinator's declaration; unbound sessions only."""
    if type(value) is not str or not value:
        raise RouteGenerationError("remote-resident declaration must be JSON text")
    try:
        group = RuntimeRemoteResidentFfn.from_json(json.loads(value))
    except (ValueError, RuntimePlanError) as error:
        raise RouteGenerationError(
            "remote-resident declaration is invalid: " + str(error)
        ) from error
    if any(row.session_generation != 0 for row in group.sessions):
        raise RouteGenerationError(
            "remote-resident declaration must not carry session generations"
        )
    return group


class RouteRemoteResidentMixin:
    """Remote-resident FFN handling shared by candidate generation stages."""

    def _phone_session_capability(self, session_id: str):
        cache = self.__dict__.setdefault("_phone_session_capability_cache", {})
        if session_id in cache:
            return cache[session_id]
        found = None
        for device_id, executor in self.catalog.executor_by_device.items():
            for row in executor.phone_sessions:
                if row.session_id == session_id:
                    if found is not None:
                        raise RouteGenerationError(
                            "phone session id is duplicated across devices: " + session_id
                        )
                    found = (device_id, row)
        cache[session_id] = found
        return found

    def _remote_resident_group(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        coordinator=None,
    ) -> RuntimeRemoteResidentFfn | None:
        """The coordinator's declared group, or None for every other pattern.

        Only composite coordinators can own a declaration in this milestone, so patterns
        without ``coordinator_executor_id`` never consult the catalog.
        """
        if coordinator is None:
            executor_id = getattr(pattern, "coordinator_executor_id", None)
            if executor_id is None:
                return None
            coordinator = self.catalog.composite_executor_by_id.get(executor_id)
            if coordinator is None:
                return None
        parameters = getattr(coordinator, "adapter_parameters", None)
        declaration = None if parameters is None else parameters.get(REMOTE_RESIDENT_FFN_PARAMETER)
        if declaration is None:
            return None
        cache = self.__dict__.setdefault("_remote_resident_group_cache", {})
        key = (coordinator.executor_id, manifest.artifact_sha256)
        if key in cache:
            return cache[key]
        group = parse_remote_resident_declaration(declaration)
        RouteRemoteResidentMixin._validate_remote_resident_group(
            self, manifest, pattern, coordinator, group
        )
        cache[key] = group
        return group

    def _validate_remote_resident_group(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        coordinator,
        group: RuntimeRemoteResidentFfn,
    ) -> None:
        if group.parent_artifact_sha256 != manifest.artifact_sha256:
            raise RouteGenerationError(
                "remote-resident group belongs to another artifact"
            )
        if group.layer_mask >> manifest.block_count:
            raise RouteGenerationError(
                "remote-resident layers exceed the model's block count"
            )
        if pattern.assisted_operator_kind is not None or any(
            helper is not None for _primary, helper, _fraction in pattern.assignments.values()
        ):
            raise RouteGenerationError(
                "remote-resident group requires a plain desktop parent pattern"
            )
        tensor_by_id = manifest.tensor_by_id
        omitted = 0
        for tensor_id in remote_resident_tensor_ids(group.layer_mask):
            tensor = tensor_by_id.get(tensor_id)
            if tensor is None:
                raise RouteGenerationError(
                    "remote-resident tensor is absent from the manifest: " + tensor_id
                )
            dtype = _DTYPE_BY_QUANTIZATION.get(tensor.quantization)
            if dtype is None or dtype != group.dtype:
                raise RouteGenerationError(
                    "remote-resident tensor dtype is unsupported: " + tensor_id
                    + " " + str(tensor.quantization)
                )
            omitted += tensor.nbytes
        if omitted != group.omitted_bytes:
            raise RouteGenerationError(
                "remote-resident omitted bytes differ from the manifest: "
                f"{group.omitted_bytes} != {omitted}"
            )
        devices = set()
        for session in group.sessions:
            located = self._phone_session_capability(session.session_id)
            if located is None:
                raise RouteGenerationError(
                    "remote-resident owner session is unknown: " + session.session_id
                )
            device_id, capability = located
            devices.add(device_id)
            if (
                capability.endpoint != session.endpoint
                or session.layer_mask & ~capability.supported_layer_mask
                or capability.maximum_columns < manifest.feed_forward_length
                or "F16" not in capability.supported_data_types
            ):
                raise RouteGenerationError(
                    "remote-resident owner session cannot hold complete shards: "
                    + session.session_id
                )
        if len(devices) != 1:
            raise RouteGenerationError(
                "remote-resident owners must live on one phone for this milestone"
            )
        phone_device_id = next(iter(devices))
        if coordinator.adapter_parameters.get("phone_device_id") != phone_device_id:
            raise RouteGenerationError(
                "remote-resident coordinator lacks its phone runtime parameters"
            )
        resident_mask = coordinator.adapter_parameters.get("ffn_resident_layer_mask")
        if type(resident_mask) is not int or group.layer_mask & ~resident_mask:
            raise RouteGenerationError(
                "remote-resident layers exceed the coordinator's resident layer mask"
            )
        if coordinator.adapter_parameters.get("ffn_resident_columns") != (
            manifest.feed_forward_length
        ):
            raise RouteGenerationError(
                "remote-resident coordinator must expose complete FFN columns"
            )
        max_tokens = coordinator.adapter_parameters.get("ffn_max_tokens")
        ubatch = coordinator.adapter_parameters.get("ubatch_size")
        if type(max_tokens) is not int or type(ubatch) is not int or max_tokens != ubatch:
            raise RouteGenerationError(
                "remote-resident prefill requires ffn_max_tokens equal to the ubatch"
            )

    def _remote_resident_owner_status(
        self,
        manifest: ModelManifest,
        group: RuntimeRemoteResidentFfn,
        snapshot=None,
    ) -> tuple[RuntimeRemoteResidentFfn | None, tuple[str, ...]]:
        """Bind physical generations from the READY layout or explain each blocker."""
        layout = self._phone_residency_layout
        shard_by_session = {} if layout is None else {
            row.session_id: row for row in layout.shards
            if row.artifact_sha256 == manifest.artifact_sha256
        }
        generation_by_session = (
            {} if layout is None else dict(layout.session_generation_by_id)
        )
        reasons = []
        generations = {}
        operator_plans = {}
        observed = {} if snapshot is None else {
            row.session_id: row for row in snapshot.phone_session_residency
        }
        for session in group.sessions:
            located = self._phone_session_capability(session.session_id)
            capability = None if located is None else located[1]
            shard = shard_by_session.get(session.session_id)
            generation = generation_by_session.get(session.session_id, 0)
            physical = observed.get(session.session_id)
            if snapshot is None:
                resident_ready = capability is not None and (
                    capability.residency_state in {"hot", "warm"}
                    and capability.resident_artifact_sha256 == manifest.artifact_sha256
                    and capability.resident_geometry_sha256 == session.resident_geometry_sha256
                )
            else:
                resident_ready = (
                    located is not None and physical is not None and shard is not None
                    and physical.device_id == located[0] and physical.state == "READY"
                    and physical.endpoint == session.endpoint
                    and physical.artifact_sha256 == manifest.artifact_sha256
                    and physical.resident_geometry_sha256 == session.resident_geometry_sha256
                    and physical.operator_plan_sha256 == shard.operator_plan_sha256
                    and physical.session_generation == generation
                    and physical.resident_bytes == session.resident_bytes
                )
            ready = (
                capability is not None
                and capability.ready
                and resident_ready
                and shard is not None
                and shard.endpoint == session.endpoint
                and shard.layer_mask == session.layer_mask
                and shard.resident_geometry_sha256 == session.resident_geometry_sha256
                and shard.maximum_columns == manifest.feed_forward_length
                and shard.resident_bytes == session.resident_bytes
                and generation >= 1
            )
            if not ready:
                suffix = (
                    ""
                    if capability is None or capability.unavailable_reason is None
                    else ":" + capability.unavailable_reason.upper()
                )
                reasons.append(
                    REMOTE_RESIDENT_OWNER_NOT_READY + ":" + session.session_id + suffix
                )
                continue
            generations[session.session_id] = generation
            operator_plans[session.session_id] = shard.operator_plan_sha256
        if reasons:
            return None, tuple(reasons)
        return group.with_session_generations(
            generations, operator_plan_by_id=operator_plans
        ), ()

    def _remote_resident_memory_demands(
        self,
        manifest: ModelManifest,
        group: RuntimeRemoteResidentFfn,
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        snapshot=None,
    ) -> tuple[RuntimeMemoryDemand, ...]:
        """Pin the owning sessions; credit only sessions that already hold the shard."""
        bound, _reasons = self._remote_resident_owner_status(manifest, group, snapshot)
        rows = list(memory_demands)
        existing = {row.demand_id for row in rows}
        for session in group.sessions:
            demand_id = "phone-session:" + session.session_id + ":weights"
            if demand_id in existing:
                raise RouteGenerationError(
                    "remote-resident owner already appears as a helper shard: "
                    + session.session_id
                )
            device_id, capability = self._phone_session_capability(session.session_id)
            rows.append(RuntimeMemoryDemand(
                demand_id=demand_id,
                resource_id=capability.memory_resource_id,
                kind="session_residency_constraint",
                required_bytes=session.resident_bytes,
                resident_bytes=session.resident_bytes if bound is not None else 0,
                lifetime="resident",
                share_key=(
                    manifest.artifact_sha256 + ":" + session.resident_geometry_sha256
                ),
                device_id=device_id,
            ))
        return tuple(rows)

    @staticmethod
    def _remote_resident_tensor_ids(
        group: RuntimeRemoteResidentFfn | None,
    ) -> frozenset[str]:
        return frozenset() if group is None else frozenset(group.tensor_ids)
