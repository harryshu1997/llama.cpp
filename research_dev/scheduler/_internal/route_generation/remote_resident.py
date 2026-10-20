"""Route generation for desktop parents whose FFN weights live only on phone sessions.

A desktop coordinator declares its remote-resident group in ``adapter_parameters`` under
``remote_resident_ffn_v1`` (the JSON of ``RuntimeRemoteResidentFfn`` with unbound sessions).
Route generation then

* folds the group into the desktop placement identity and the execution contract,
* binds the owning sessions' physical generations from the READY phone layout,
* removes the omitted tensor bytes from the desktop weight demand and adds the owning
  sessions as residency constraints,
* rejects the candidate while an owner is not READY with the declared shard geometry,
* binds static co-helper phones (``phone_co_helpers_v1``) of disjoint layers next to the owner.

Everything fails closed: a declaration that disagrees with the model manifest, the phone
session capabilities or the shard geometry raises ``RouteGenerationError``.
"""

from __future__ import annotations

import json

from ..model_manifest import ModelManifest
from ..plan_contracts.co_helpers import PHONE_HELPERS_PARAMETER, co_helper_declaration
from ..plan_contracts.remote_resident import REMOTE_RESIDENT_FFN_QUANTIZATION_BY_DTYPE
from ..runtime_cost import RuntimeMemoryDemand
from ..runtime_plan import (
    RuntimePlanError,
    RuntimeRemoteResidentFfn,
    remote_resident_tensor_ids,
)
from .common import RouteGenerationError, _Pattern


REMOTE_RESIDENT_FFN_PARAMETER = "remote_resident_ffn_v1"
# opt-in (campaign remote_resident_owner.static_co_helpers): static co-helpers serve decode only, each
# under its own row cap (S41_SERVER_FFN_HELPER<k>_MAX_TOKENS), driven by a static decode policy
REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER = "remote_resident_co_helper_decode"
REMOTE_RESIDENT_OWNER_NOT_READY = "REMOTE_RESIDENT_OWNER_NOT_READY"
_DTYPE_BY_QUANTIZATION = {
    quantization: dtype
    for dtype, quantization in REMOTE_RESIDENT_FFN_QUANTIZATION_BY_DTYPE.items()
}


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


REMOTE_RESIDENT_SHARE_KEY_PREFIX = "remote-resident-server:"


def remote_resident_share_keys(
    catalog,
    *,
    executor_id: str,
    endpoint: str | None,
    parameters,
    resource_ids,
):
    """Co-tenancy keys of a remote-resident parent's phone lanes.

    The parent's server batches the rows of all its running requests into one call per
    layer on each phone it drives (the owner sessions and any static decode co-helpers),
    so its requests share those phones' lanes (compute, transport and the links touching
    the phones) under one key per server, and any other executor's lease stays exclusive
    with them. Desktop resources keep per-request leases, which bound the tenants by the
    server's slots. Empty for every other route.
    """
    phone = parameters.get("phone_device_id")
    if REMOTE_RESIDENT_FFN_PARAMETER not in parameters or type(phone) is not str:
        return {}
    phones = {phone}
    declaration = co_helper_declaration(parameters)
    if declaration is not None:
        phones.update(declaration.device_ids)
    lanes = set()
    for device_id in phones:
        executor = catalog.executor_by_device.get(device_id)
        if executor is None:
            continue
        lanes.update(executor.execution_resource_ids)
        for session in executor.phone_sessions:
            lanes.update(session.shared_transport_resource_ids)
    lanes.update(
        "link:" + row.link_id
        for row in catalog.placement_profile.links
        if row.source_device in phones or row.target_device in phones
    )
    key = REMOTE_RESIDENT_SHARE_KEY_PREFIX + executor_id + "@" + str(endpoint)
    return {
        resource_id: key for resource_id in sorted(resource_ids) if resource_id in lanes
    }


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


def validate_remote_resident_co_helpers(
    parameters,
    group: RuntimeRemoteResidentFfn,
    block_count: int,
) -> None:
    """Static co-helpers of a remote-resident parent never serve a remote-resident layer.

    The omitted layers have no desktop copy, so only their owner may run them. Every client of
    the server shares one maximum batch, which here is the whole ubatch (prefill on the owner),
    unless the parent opts in to decode co-helpers: those may cap their rows lower.
    """
    try:
        declaration = co_helper_declaration(parameters)
    except RuntimePlanError as error:
        raise RouteGenerationError(
            "remote-resident co-helper declaration is invalid: " + str(error)
        ) from error
    if declaration is None:
        return
    if parameters.get("phone_device_id") in declaration.device_ids:
        raise RouteGenerationError(
            "remote-resident owner phone cannot also be a static co-helper"
        )
    if declaration.layer_mask & group.layer_mask:
        raise RouteGenerationError(
            "static co-helper overlaps remote-resident layers"
        )
    resident_mask = parameters.get("ffn_resident_layer_mask")
    if type(resident_mask) is not int or declaration.layer_mask & resident_mask:
        raise RouteGenerationError(
            "static co-helper overlaps the owner phone's resident layers"
        )
    if declaration.layer_mask >> block_count:
        raise RouteGenerationError(
            "static co-helper layers exceed the model's block count"
        )
    maximum_tokens = parameters.get("ffn_max_tokens")
    decode_only = parameters.get(REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER) == 1
    if any(
        row.max_tokens != maximum_tokens
        and not (decode_only and type(maximum_tokens) is int and row.max_tokens < maximum_tokens)
        for row in declaration.helpers
    ):
        raise RouteGenerationError(
            "static co-helper batch capacity differs from the remote-resident ubatch"
        )


def remote_resident_co_helper_parameters(parameters: dict[str, object]) -> None:
    """Bind a remote-resident parent's static co-helpers into its launch parameters.

    The owner phone stays the first helper with its resident layers; the server's layer mask
    becomes the union, as for a phone-assisted route with co-helpers.
    """
    try:
        declaration = co_helper_declaration(parameters)
        if declaration is None:
            return
        owner_mask = parameters["ffn_resident_layer_mask"]
        binding = declaration.phone_helpers(
            parameters["phone_device_id"], owner_mask,
            **({"maximum_tokens": parameters["ffn_max_tokens"]}
               if parameters.get(REMOTE_RESIDENT_CO_HELPER_DECODE_PARAMETER) == 1 else {}),
        )
    except (KeyError, RuntimePlanError) as error:
        raise RouteGenerationError(
            "remote-resident co-helper binding is invalid: " + str(error)
        ) from error
    parameters["ffn_resident_layer_mask"] = owner_mask | declaration.layer_mask
    parameters[PHONE_HELPERS_PARAMETER] = binding


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
        quantization = REMOTE_RESIDENT_FFN_QUANTIZATION_BY_DTYPE[group.dtype]
        tensor_by_id = manifest.tensor_by_id
        omitted = 0
        for tensor_id in remote_resident_tensor_ids(group.layer_mask):
            tensor = tensor_by_id.get(tensor_id)
            if tensor is None:
                raise RouteGenerationError(
                    "remote-resident tensor is absent from the manifest: " + tensor_id
                )
            if tensor.quantization not in _DTYPE_BY_QUANTIZATION:
                raise RouteGenerationError(
                    "remote-resident tensor dtype is unsupported: " + tensor_id
                    + " " + str(tensor.quantization)
                )
            if tensor.quantization != quantization:
                raise RouteGenerationError(
                    "remote-resident tensor dtype differs from the group dtype "
                    + group.dtype + ": " + tensor_id + " " + tensor.quantization
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
            ):
                raise RouteGenerationError(
                    "remote-resident owner session cannot hold complete shards: "
                    + session.session_id
                )
            if quantization not in capability.supported_data_types:
                raise RouteGenerationError(
                    "remote-resident owner session does not serve " + quantization
                    + ": " + session.session_id
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
        validate_remote_resident_co_helpers(
            coordinator.adapter_parameters, group, manifest.block_count
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
