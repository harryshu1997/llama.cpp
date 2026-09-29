"""PlacementEpochMixin frontier operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.background_placement import (
    BackgroundPlacementError,
    material_count_bucket,
    material_horizon_bucket_us,
    placement_frontier_key,
)
from ..._internal.model_placement_controller import ModelDemandSnapshot
from ..._internal.runtime_search import request_shape_bucket
from ..._internal.types import canonical_sha256


def _frontier_resource_generation(
    controller,
    demand: ModelDemandSnapshot,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> str:
    phone_layout = (
        controller._model_placement_controller.planning_phone_layout()
    )
    return canonical_sha256({
        **({"phone_htp_memory_caps": dict(sorted(controller._phone_htp_memory_caps.items()))}
           if controller._phone_htp_memory_caps else {}),
        "available_device_ids": demand.available_device_ids,
        "available_session_ids": demand.available_session_ids,
        "executor_state": [
            {
                "executor_id": row.executor_id,
                "healthy": row.healthy,
            }
            for row in snapshot.executors.values()
        ],
        "link_state": [
            {"link_id": row.link_id}
            for row in snapshot.links.values()
        ],
        "memory_state": [
            {
                "available_64mib_bucket": material_count_bucket(
                    row.available_bytes // (64 * 1024**2)
                ),
                "capacity_bytes": row.capacity_bytes,
                "reserve_bytes": row.reserve_bytes,
                "resource_id": row.resource_id,
            }
            for row in snapshot.memory.capacities.values()
        ],
        "pressure_bucket": demand.pressure_bucket,
        "phone_residency_layout_sha256": (
            None
            if phone_layout is None
            else phone_layout.layout.geometry_sha256
        ),
        "resource_calendar_horizon_us": (
            material_horizon_bucket_us(
                demand.predicted_queue_drain_us
            )
        ),
        "residency_generation_sha256": (
            demand.residency_generation_sha256
        ),
        "schema": "placement-frontier-demand-resource-v1",
    })


def _lookup_prepared_frontier(
    controller,
    planner,
    key,
    artifact_sha256: str,
    observed_at_us: int,
):
    envelope = planner.lookup(key, observed_at_us)
    if envelope is not None:
        controller._model_placement_controller.mark_background_complete(
            artifact_sha256, observed_at_us
        )
    return envelope


def _placement_frontier_reasons(
    controller,
    demand_id: tuple[object, ...],
    previous,
    key,
    resource_state: tuple[object, ...],
) -> tuple[str, ...]:
    if previous is None:
        return ("model_or_shape_arrival",)
    reasons = []
    if (
        controller._background_resource_states.get(demand_id)
        != resource_state
    ):
        reasons.append("virtual_queue_changed")
    for name, reason in (
        (
            "capability_generation_sha256",
            "device_capability_changed",
        ),
        (
            "residency_generation_sha256",
            "model_residency_changed",
        ),
        (
            "capacity_generation_sha256",
            "capacity_or_queue_changed",
        ),
        (
            "cost_profile_generation_sha256",
            "cost_or_link_profile_changed",
        ),
        (
            "demand_resource_generation_sha256",
            "demand_or_resource_generation_changed",
        ),
    ):
        if getattr(previous, name) != getattr(key, name):
            reasons.append(reason)
    if not reasons:
        reasons.append("periodic_refresh")
    return tuple(reasons)


def _prepared_placement_frontier(
    controller,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    *,
    demand_snapshot: ModelDemandSnapshot | None = None,
):
    planner = controller._background_placement_planner
    if planner is None or controller._runtime_capabilities is None:
        return None
    try:
        input_bucket, output_bucket = request_shape_bucket(
            request.input_tokens, request.output_tokens
        )
        demand_id = (
            manifest.artifact_sha256,
            input_bucket,
            output_bucket,
            request.quality_requirement,
        )
        previous = controller._background_frontier_keys.get(demand_id)
        observation_state = ((
            "placement_learning_generation_sha256",
            controller._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            ),
        ),)
        demand = (
            controller._model_demand_snapshot(
                request, manifest, snapshot, observed_at_us
            )
            if demand_snapshot is None else demand_snapshot
        )
        resource_generation = controller._frontier_resource_generation(
            demand, snapshot
        )
        resource_state: tuple[object, ...] = (resource_generation,)
        if (
            previous is not None
            and controller._background_snapshot_objects.get(demand_id)
                is snapshot
            and controller._background_observation_states.get(demand_id)
                == observation_state
            and controller._background_resource_states.get(demand_id)
                == resource_state
        ):
            envelope = controller._lookup_prepared_frontier(
                planner,
                previous,
                manifest.artifact_sha256,
                observed_at_us,
            )
            if envelope is not None:
                return envelope.frontier
        capability_generation = (
            controller._runtime_capability_generation_sha256
        )
        if capability_generation is None:
            raise UnifiedScheduleError(
                "runtime capability identity is absent"
            )
        key = placement_frontier_key(
            manifest=manifest,
            capability_generation_sha256=capability_generation,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            quality_requirement=request.quality_requirement,
            snapshot=snapshot,
            online_cost_generation_sha256=canonical_sha256({
                "observations": dict(observation_state),
            }),
            demand_resource_generation_sha256=resource_generation,
        )
        if previous == key:
            envelope = controller._lookup_prepared_frontier(
                planner,
                key,
                manifest.artifact_sha256,
                observed_at_us,
            )
            if envelope is not None:
                return envelope.frontier
        planner.request(
            key=key,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            trigger_reasons=controller._placement_frontier_reasons(
                demand_id, previous, key, resource_state
            ),
            defer_start=True,
        )
        controller._background_frontier_keys[demand_id] = key
        controller._background_snapshot_objects[demand_id] = snapshot
        controller._background_observation_states[demand_id] = (
            observation_state
        )
        controller._background_resource_states[demand_id] = resource_state
        envelope = controller._lookup_prepared_frontier(
            planner,
            key,
            manifest.artifact_sha256,
            observed_at_us,
        )
        return None if envelope is None else envelope.frontier
    except BackgroundPlacementError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def _arrived_decode_work_by_artifact(
    active_remaining_tokens_by_artifact: Mapping[str, int],
    queued_output_tokens_by_artifact: Mapping[str, int],
) -> Mapping[str, int]:
    if any(
        type(value) is not int or value < 0
        for value in (
            *active_remaining_tokens_by_artifact.values(),
            *queued_output_tokens_by_artifact.values(),
        )
    ):
        raise UnifiedScheduleError(
            "arrived decode work is invalid"
        )
    result = {}
    for artifact_sha256 in (
        set(active_remaining_tokens_by_artifact)
        | set(queued_output_tokens_by_artifact)
    ):
        remaining_tokens = (
            active_remaining_tokens_by_artifact.get(
                artifact_sha256, 0
            )
            + queued_output_tokens_by_artifact.get(
                artifact_sha256, 0
            )
        )
        if remaining_tokens > 0:
            result[artifact_sha256] = remaining_tokens
    return MappingProxyType(dict(sorted(result.items())))


def _persistent_phone_service_reserve_by_artifact(
    controller,
    phone_device_id: str,
    snapshot: HeterogeneousRuntimeSnapshot | None,
    *, include_observed: bool = False,
) -> Mapping[str, int]:
    """Reserve peak bytes once, crediting only observed physical residency."""

    catalog = controller._runtime_capabilities
    if catalog is None:
        return MappingProxyType({})
    executor = catalog.executor_by_device.get(phone_device_id)
    if (
        executor is None
        or executor.adapter_parameters.get("persistent_residency") != 1
    ):
        return MappingProxyType({})
    persistent_artifacts = {
        row.artifact_sha256
        for row in catalog.transitions
        if row.executor_id == executor.executor_id
        and row.artifact_sha256 is not None
        and row.target_state in {"hot", "warm"}
        and row.energy_maturity == "QUALIFIED"
    }
    persistent_artifacts.update(
        row.artifact_sha256 for row in (() if snapshot is None else snapshot.residency)
        if row.device_id == phone_device_id and row.executor_id == executor.executor_id
        and row.state in {"hot", "warm"} and row.resident_bytes > 0
    )
    manifest_by_artifact = {
        row.artifact_sha256: row
        for row in controller._runtime_manifests.values()
    }
    peak = executor.adapter_parameters.get("whole_model_peak_memory_bytes")
    if peak is not None and (type(peak) is not int or peak <= 0):
        raise UnifiedScheduleError("persistent phone service peak memory is invalid")
    observed_by_artifact: dict[str, int] = {}
    for row in (
        () if snapshot is None else snapshot.residency
    ):
        if (
            row.device_id == phone_device_id
            and row.artifact_sha256 in persistent_artifacts
            and row.state in {"hot", "warm"}
            and (peak is None or row.executor_id == executor.executor_id)
        ):
            observed_by_artifact[row.artifact_sha256] = max(
                observed_by_artifact.get(row.artifact_sha256, 0),
                row.resident_bytes if peak is None else (
                    row.reclaimable_bytes or row.resident_bytes
                ),
            )
    result = {}
    for artifact_sha256 in sorted(persistent_artifacts):
        manifest = manifest_by_artifact.get(artifact_sha256)
        if manifest is None:
            continue
        required = manifest.tensor_bytes if peak is None else peak
        if required < manifest.tensor_bytes:
            raise UnifiedScheduleError("persistent phone service peak is smaller than its weights")
        missing = max(
            0,
            required - (0 if include_observed else
                        observed_by_artifact.get(artifact_sha256, 0)),
        )
        if missing:
            result[artifact_sha256] = missing
    return MappingProxyType(result)
