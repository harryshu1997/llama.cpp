"""Project scheduler-issued residency transitions across queued work."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from .model_manifest import ModelManifest
from .runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    ModelResidencyObservation,
    RuntimeCapabilityCatalog,
)
from .runtime_controller import RuntimeRequestTicket
from .runtime_plan import RuntimeResidencyEviction, RuntimeTransitionPlan
from .runtime_resources import RuntimeResidencyProjectionToken, runtime_preparation_windows
from .types import canonical_sha256


class RuntimeResidencyProjectionError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        request_id: str | None = None,
        ticket_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.ticket_id = ticket_id


def _phone_shards_sha256(shards: Sequence[object]) -> str:
    return canonical_sha256([
        {
            "artifact_sha256": shard.artifact_sha256,
            "endpoint": shard.endpoint,
            "layer_mask": shard.layer_mask,
            "maximum_columns": shard.maximum_columns,
            "operator_plan_sha256": shard.operator_plan_sha256,
            "resident_bytes": shard.resident_bytes,
            "resident_geometry_sha256": (
                shard.resident_geometry_sha256
            ),
            "session_id": shard.session_id,
        }
        for shard in sorted(shards, key=lambda row: row.session_id)
    ])


def runtime_residency_projection_token(
    ticket: RuntimeRequestTicket,
    layout_generation: int,
) -> RuntimeResidencyProjectionToken:
    """Create the exact token for one queued phone layout transition."""

    if not isinstance(ticket, RuntimeRequestTicket):
        raise RuntimeResidencyProjectionError(
            "residency projection ticket is invalid"
        )
    if type(layout_generation) is not int or layout_generation < 1:
        raise RuntimeResidencyProjectionError(
            "residency projection layout generation is invalid"
        )
    plan = ticket.execution_plan
    if plan is None:
        raise RuntimeResidencyProjectionError(
            "residency projection ticket lacks an execution plan"
        )
    geometry = plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    transitions = tuple(
        transition for transition in plan.transitions
        if transition.phone_shards
    )
    if (
        type(geometry) is not str
        or not geometry.startswith("sha256:")
        or len(geometry) != 71
        or len(transitions) != 1
    ):
        raise RuntimeResidencyProjectionError(
            "residency projection phone transition is ambiguous",
            request_id=ticket.request.request_id,
            ticket_id=ticket.ticket_id,
        )
    transition = transitions[0]
    lease_by_resource = ticket.transition_lease_by_resource(transition)
    if set(transition.resource_ids) - set(lease_by_resource):
        raise RuntimeResidencyProjectionError(
            "residency projection transition lacks a resource lease",
            request_id=ticket.request.request_id,
            ticket_id=ticket.ticket_id,
        )
    ready_at_us = max(
        lease_by_resource[resource_id].start_us
        for resource_id in transition.resource_ids
    ) + transition.latency_us
    phone_demand_ids = {
        demand.demand_id
        for demand in plan.memory_demands
        if demand.device_id == transition.device_id
        and demand.kind in {
            "model_weights",
            "session_residency_constraint",
            "workspace",
        }
    }
    memory_delta: dict[str, int] = {}
    for reservation in ticket.memory_reservations:
        if reservation.demand_id not in phone_demand_ids:
            continue
        memory_delta[reservation.resource_id] = (
            memory_delta.get(reservation.resource_id, 0)
            + reservation.reserved_bytes
            - reservation.replaced_bytes
        )
    if not memory_delta:
        raise RuntimeResidencyProjectionError(
            "residency projection transition lacks a memory delta",
            request_id=ticket.request.request_id,
            ticket_id=ticket.ticket_id,
        )
    return RuntimeResidencyProjectionToken(
        predecessor_ticket_id=ticket.ticket_id,
        predecessor_request_id=ticket.request.request_id,
        transition_id=transition.transition_id,
        target_geometry_sha256=geometry,
        target_shards_sha256=_phone_shards_sha256(
            transition.phone_shards
        ),
        layout_generation=layout_generation,
        ready_at_us=ready_at_us,
        resource_ids=transition.resource_ids,
        memory_delta_bytes_by_resource=memory_delta,
    )


def projection_token_matches_plan(
    token: RuntimeResidencyProjectionToken,
    plan: object,
) -> bool:
    """Return whether a route uses exactly the token's resident shards."""

    if not isinstance(token, RuntimeResidencyProjectionToken):
        return False
    geometry = getattr(plan, "adapter_parameters", {}).get(
        "phone_shard_set_geometry_sha256"
    )
    contract = getattr(plan, "execution_contract", None)
    shards = () if contract is None else contract.phone_shards
    return (
        geometry == token.target_geometry_sha256
        and bool(shards)
        and _phone_shards_sha256(shards) == token.target_shards_sha256
    )


@dataclass(frozen=True)
class _ProjectionEvent:
    start_us: int
    end_us: int
    attempt_index: int
    ticket_id: str
    resource_id: str
    anchor_device_id: str
    executor_id: str
    ticket: RuntimeRequestTicket
    manifest: ModelManifest
    allocations: tuple[tuple[str, int, int], ...]
    evictions: tuple[RuntimeResidencyEviction, ...]
    transition: RuntimeTransitionPlan


def transition_target_is_observed(
    snapshot: HeterogeneousRuntimeSnapshot,
    ticket: RuntimeRequestTicket,
    transition: RuntimeTransitionPlan,
) -> bool:
    """Return whether a state-changing transition is already complete."""

    if transition.source_state == transition.target_state:
        return False
    expected_executor_id = transition.executor_id
    if expected_executor_id is None:
        participant = next(
            (
                row for row in ticket.binding.participants
                if row.device_id == transition.device_id
            ),
            None,
        )
        expected_executor_id = (
            ticket.binding.executor_id
            if participant is None
            else participant.executor_id
        )
    return all(
        (
            residency := snapshot.residency_for(
                ticket.model.model_id,
                ticket.model.artifact_sha256,
                device_id,
            )
        ) is not None
        and residency.state == transition.target_state
        and residency.executor_id == expected_executor_id
        for device_id in transition.prepares_device_ids
    )


def _target_tensor_ids(
    ticket: RuntimeRequestTicket,
    manifest: ModelManifest,
    device_id: str,
) -> tuple[str, ...]:
    plan = ticket.execution_plan
    if plan is None:
        raise RuntimeResidencyProjectionError(
            "projected residency ticket has no execution plan"
        )
    operator_by_id = {
        operator.operator_id: operator for operator in manifest.operators
    }
    values = set()
    for assignment in plan.operators:
        if device_id not in assignment.device_ids:
            continue
        operator = operator_by_id.get(assignment.operator_id)
        if operator is None:
            raise RuntimeResidencyProjectionError(
                "projected residency operator is absent from its manifest"
            )
        values.update(operator.tensor_ids)
    return tuple(sorted(values))


def _resident_weight_share_key(
    ticket: RuntimeRequestTicket,
    device_id: str,
) -> str:
    geometry = _resident_geometry_sha256(ticket, device_id)
    if geometry is None:
        return ticket.model.artifact_sha256 + ":" + device_id
    return "phone-residency:" + geometry


def _resident_geometry_sha256(
    ticket: RuntimeRequestTicket,
    device_id: str,
) -> str | None:
    plan = ticket.execution_plan
    if plan is None:
        raise RuntimeResidencyProjectionError(
            "projected residency ticket has no execution plan"
        )
    contract = plan.execution_contract
    if contract is None or not contract.phone_shards:
        return None
    phone_device_id = plan.adapter_parameters.get("phone_device_id")
    geometry = plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    if phone_device_id != device_id:
        return None
    if (
        type(geometry) is not str
        or not geometry.startswith("sha256:")
        or len(geometry) != 71
        or any(value not in "0123456789abcdef" for value in geometry[7:])
    ):
        raise RuntimeResidencyProjectionError(
            "projected phone residency geometry is invalid"
        )
    return geometry


def _pending_exclusive_allocation_bytes(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    device_id: str,
    resource_id: str,
    start_us: int,
    end_us: int,
    planned_bytes: int,
) -> int:
    if (
        ticket.transition_status != "PENDING"
        or not start_us <= snapshot.captured_at_us < end_us
        or ticket.runtime_observation.captured_at_us > start_us
    ):
        return planned_bytes
    lease = next((
        row for row in ticket.decision.leases
        if row.resource_id == resource_id
    ), None)
    resource = catalog.resources.get(resource_id)
    capability = catalog.executor_by_device.get(device_id)
    if (
        lease is None
        or resource is None
        or capability is None
        or len(lease.lanes) != resource.capacity
    ):
        return planned_bytes
    memory_resource_id = capability.memory_resource_id
    capacity = snapshot.memory.capacities.get(memory_resource_id)
    baseline_occupied = (
        ticket.runtime_observation.memory_occupied_bytes.get(
            memory_resource_id
        )
    )
    if capacity is None or baseline_occupied is None:
        return planned_bytes
    measured_bytes = max(0, capacity.occupied_bytes - baseline_occupied)
    return max(planned_bytes, measured_bytes)


def _authoritative_transition_covers_snapshot(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    resource_id: str,
    start_us: int,
    end_us: int,
) -> bool:
    lease = next((
        row for row in ticket.decision.leases
        if row.resource_id == resource_id
    ), None)
    resource = catalog.resources.get(resource_id)
    physical_memory_changed = False
    if ticket.dispatch_state == "QUEUED" and ticket.execution_plan is not None:
        prepared_device_ids = {
            device_id
            for transition in ticket.execution_plan.transitions
            if resource_id in transition.resource_ids
            for device_id in transition.prepares_device_ids
        }
        physical_memory_changed = any(
            (
                capacity := snapshot.memory.capacities.get(
                    catalog.executor_by_device[
                        device_id
                    ].memory_resource_id
                )
            ) is not None
            and capacity.occupied_bytes
                != ticket.runtime_observation.memory_occupied_bytes.get(
                    capacity.resource_id
                )
            for device_id in prepared_device_ids
            if device_id in catalog.executor_by_device
        )
    return (
        (
            ticket.dispatch_state == "ACQUIRED"
            or physical_memory_changed
        )
        and ticket.transition_status in {"PENDING", "COMPLETED"}
        and start_us <= snapshot.captured_at_us < end_us
        and ticket.runtime_observation.captured_at_us <= start_us
        and lease is not None
        and resource is not None
        and len(lease.lanes) == resource.capacity
    )


def _authoritative_replacement_occupancy(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    resource_id: str,
    start_us: int,
    end_us: int,
    allocations: Sequence[tuple[str, int, int]],
    evictions: Sequence[RuntimeResidencyEviction],
) -> Mapping[str, int]:
    if (
        not _authoritative_transition_covers_snapshot(
            snapshot,
            catalog,
            ticket,
            resource_id,
            start_us,
            end_us,
        )
        or not evictions
    ):
        return {}
    allocated = {}
    for device_id, _resident_bytes, reclaimable_bytes in allocations:
        resource_id = catalog.executor_by_device[device_id].memory_resource_id
        allocated[resource_id] = (
            allocated.get(resource_id, 0) + reclaimable_bytes
        )
    reclaimed = {}
    for eviction in evictions:
        resource_id = catalog.executor_by_device[
            eviction.device_id
        ].memory_resource_id
        reclaimable_bytes = (
            eviction.resident_bytes
            if eviction.reclaimable_bytes is None
            else eviction.reclaimable_bytes
        )
        reclaimed[resource_id] = (
            reclaimed.get(resource_id, 0) + reclaimable_bytes
        )
    result = {}
    for resource_id in sorted(set(allocated) | set(reclaimed)):
        capacity = snapshot.memory.capacities.get(resource_id)
        baseline = ticket.runtime_observation.memory_occupied_bytes.get(
            resource_id
        )
        if capacity is None or baseline is None:
            continue
        released_bytes = reclaimed.get(resource_id, 0)
        allocated_bytes = allocated.get(resource_id, 0)
        if (
            ticket.transition_status == "PENDING"
            and allocated_bytes <= capacity.occupied_bytes
            and capacity.occupied_bytes + capacity.reserve_bytes
                == capacity.capacity_bytes
        ):
            result[resource_id] = capacity.occupied_bytes
            continue
        reflected_release_bytes = (
            released_bytes if released_bytes <= baseline else 0
        )
        expected_end = (
            baseline - reflected_release_bytes + allocated_bytes
        )
        expected_peak = baseline + allocated_bytes
        if ticket.transition_status == "COMPLETED":
            result[resource_id] = max(
                expected_end, capacity.occupied_bytes
            )
        else:
            unrelated_growth = max(
                0, capacity.occupied_bytes - expected_peak
            )
            result[resource_id] = expected_end + unrelated_growth
    return result


def _project_atomic_replacement_occupancy(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    replacement_group_id: str,
    allocations: Sequence[tuple[str, int, int]],
    evictions: Sequence[RuntimeResidencyEviction],
    projected_occupied: Mapping[str, int],
    authoritative: Mapping[str, int],
    replacement_ceiling: Mapping[tuple[str, str], int],
) -> tuple[Mapping[str, int], Mapping[tuple[str, str], int]]:
    """Project one exact replacement without reclaiming unobserved pages."""

    allocated = {}
    for device_id, _resident_bytes, reclaimable_bytes in allocations:
        resource_id = catalog.executor_by_device[device_id].memory_resource_id
        allocated[resource_id] = (
            allocated.get(resource_id, 0) + reclaimable_bytes
        )
    reclaimed = {}
    for eviction in evictions:
        resource_id = catalog.executor_by_device[
            eviction.device_id
        ].memory_resource_id
        reclaimable_bytes = (
            eviction.resident_bytes
            if eviction.reclaimable_bytes is None
            else eviction.reclaimable_bytes
        )
        reclaimed[resource_id] = (
            reclaimed.get(resource_id, 0) + reclaimable_bytes
        )

    replacement = dict(projected_occupied)
    ceilings = dict(replacement_ceiling)
    for resource_id in sorted(set(allocated) | set(reclaimed)):
        capacity = snapshot.memory.capacities.get(resource_id)
        if capacity is None or resource_id not in replacement:
            raise RuntimeResidencyProjectionError(
                "projected replacement memory resource is absent: "
                + resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )
        ceiling_key = (resource_id, replacement_group_id)
        if resource_id in authoritative:
            occupied_bytes = authoritative[resource_id]
            ceilings[ceiling_key] = allocated.get(resource_id, 0)
        else:
            released_bytes = reclaimed.get(resource_id, 0)
            if released_bytes > replacement[resource_id]:
                released_bytes = 0
            occupied_bytes = (
                replacement[resource_id]
                - released_bytes
                + allocated.get(resource_id, 0)
            )
            ceilings[ceiling_key] = allocated.get(resource_id, 0)
        if occupied_bytes + capacity.reserve_bytes > capacity.capacity_bytes:
            raise RuntimeResidencyProjectionError(
                "projected allocation exceeds memory capacity: "
                + resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )
        replacement[resource_id] = occupied_bytes
    return replacement, ceilings


def _validate_projected_evictions(
    residency_by_key: Mapping[
        tuple[str, str, str], ModelResidencyObservation
    ],
    transition: RuntimeTransitionPlan,
    ticket: RuntimeRequestTicket,
    owned_device_ids: Sequence[str],
    *,
    allow_missing: bool = False,
    allow_in_progress: bool = False,
) -> None:
    owned = frozenset(owned_device_ids)
    for eviction in transition.evictions:
        if eviction.device_id not in owned:
            continue
        observed = residency_by_key.get((
            eviction.model_id,
            eviction.artifact_sha256,
            eviction.device_id,
        ))
        observed_reclaimable = (
            None
            if observed is None
            else (
                observed.resident_bytes
                if observed.reclaimable_bytes is None
                else observed.reclaimable_bytes
            )
        )
        expected_reclaimable = (
            eviction.resident_bytes
            if eviction.reclaimable_bytes is None
            else eviction.reclaimable_bytes
        )
        if observed is None and allow_missing:
            continue
        if allow_in_progress and observed is not None and (
            observed.model_id == eviction.model_id
            and observed.artifact_sha256 == eviction.artifact_sha256
            and observed.device_id == eviction.device_id
        ):
            continue
        if (
            observed is None
            or observed.state not in {"hot", "warm"}
            or observed.resident_bytes != eviction.resident_bytes
            or observed.generation != eviction.generation
            or observed.executor_id != eviction.executor_id
            or observed_reclaimable != expected_reclaimable
        ):
            raise RuntimeResidencyProjectionError(
                "projected transition eviction is stale: "
                + eviction.device_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )


def _validate_projected_current_eviction(
    current: ModelResidencyObservation,
    transition: RuntimeTransitionPlan,
    ticket: RuntimeRequestTicket,
    *,
    allow_in_progress: bool = False,
) -> None:
    matching_evictions = tuple(
        eviction
        for eviction in transition.evictions
        if eviction.model_id == current.model_id
        and eviction.artifact_sha256 == current.artifact_sha256
        and eviction.device_id == current.device_id
    )
    if allow_in_progress and matching_evictions:
        return
    if any(
        _eviction_matches_observation(eviction, current)
        for eviction in matching_evictions
    ):
        return
    raise RuntimeResidencyProjectionError(
        "projected transition lacks current exclusive eviction: "
        + current.device_id,
        request_id=ticket.request.request_id,
        ticket_id=ticket.ticket_id,
    )


def _eviction_matches_observation(
    eviction: RuntimeResidencyEviction,
    current: ModelResidencyObservation,
) -> bool:
    observed_reclaimable = (
        current.resident_bytes
        if current.reclaimable_bytes is None
        else current.reclaimable_bytes
    )
    expected_reclaimable = (
        eviction.resident_bytes
        if eviction.reclaimable_bytes is None
        else eviction.reclaimable_bytes
    )
    return (
        eviction.model_id == current.model_id
        and eviction.artifact_sha256 == current.artifact_sha256
        and eviction.device_id == current.device_id
        and eviction.resident_bytes == current.resident_bytes
        and eviction.generation == current.generation
        and eviction.executor_id == current.executor_id
        and expected_reclaimable == observed_reclaimable
    )


@dataclass
class _ProjectionState:
    residency_by_key: dict[tuple[str, str, str], ModelResidencyObservation]
    projected_occupied: dict[str, int]
    current_by_device: dict[str, ModelResidencyObservation]
    current_rows_by_device: dict[str, list[ModelResidencyObservation]]
    maximum_generation: dict[str, int]
    projected_executor_by_device: dict[str, str | None]
    replacement_ceiling: dict[tuple[str, str], int]


def _validate_projection_arguments(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    rows: tuple[RuntimeRequestTicket, ...],
    *,
    exclude_request_id: str | None,
    stop_before_ticket_id: str | None,
    stop_before_us: int | None,
    preserve_ticket_order: bool,
) -> None:
    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise RuntimeResidencyProjectionError(
            "residency projection snapshot is invalid"
        )
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise RuntimeResidencyProjectionError(
            "residency projection catalog is invalid"
        )
    if any(not isinstance(row, RuntimeRequestTicket) for row in rows):
        raise RuntimeResidencyProjectionError(
            "residency projection ticket is invalid"
        )
    if (
        stop_before_ticket_id is not None
        and (
            type(stop_before_ticket_id) is not str
            or not stop_before_ticket_id
            or not stop_before_ticket_id.isascii()
            or stop_before_ticket_id == exclude_request_id
        )
    ):
        raise RuntimeResidencyProjectionError(
            "residency projection cutoff is invalid"
        )
    if (
        stop_before_us is not None
        and (type(stop_before_us) is not int or stop_before_us < 0)
    ):
        raise RuntimeResidencyProjectionError(
            "residency projection time cutoff is invalid"
        )
    if stop_before_ticket_id is not None and stop_before_us is not None:
        raise RuntimeResidencyProjectionError(
            "residency projection cutoffs conflict"
        )
    if type(preserve_ticket_order) is not bool:
        raise RuntimeResidencyProjectionError(
            "residency projection order is invalid"
        )


def _initial_projection_state(
    snapshot: HeterogeneousRuntimeSnapshot,
    exclusive_by_device: Mapping[str, str],
    catalog: RuntimeCapabilityCatalog,
) -> _ProjectionState:
    residency_by_key = {
        (row.model_id, row.artifact_sha256, row.device_id): row
        for row in snapshot.residency
    }
    projected_occupied = {
        resource_id: capacity.occupied_bytes
        for resource_id, capacity in snapshot.memory.capacities.items()
    }
    current_by_device: dict[str, ModelResidencyObservation] = {}
    current_rows_by_device: dict[
        str, list[ModelResidencyObservation]
    ] = {}
    maximum_generation: dict[str, int] = {}
    for residency in snapshot.residency:
        device_id = residency.device_id
        maximum_generation[device_id] = max(
            maximum_generation.get(device_id, 0), residency.generation
        )
        if (
            device_id not in exclusive_by_device
            or residency.state not in {"hot", "warm"}
            or residency.resident_bytes == 0
        ):
            continue
        current_rows = current_rows_by_device.setdefault(device_id, [])
        if current_rows and any(
            catalog.residency_group(previous.executor_id, device_id)
            == catalog.residency_group(residency.executor_id, device_id)
            and (previous.executor_id != residency.executor_id
            or previous.generation != residency.generation
            or previous.resident_geometry_sha256 is None
            or previous.resident_geometry_sha256
                != residency.resident_geometry_sha256)
            for previous in current_rows
        ):
            raise RuntimeResidencyProjectionError(
                "exclusive device reports incompatible resident artifacts: "
                + device_id
            )
        current_rows.append(residency)
        current_by_device.setdefault(device_id, residency)
    return _ProjectionState(
        residency_by_key=residency_by_key,
        projected_occupied=projected_occupied,
        current_by_device=current_by_device,
        current_rows_by_device=current_rows_by_device,
        maximum_generation=maximum_generation,
        projected_executor_by_device={},
        replacement_ceiling={},
    )


def _group_weight_allocations(
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    owned_devices: tuple[str, ...],
    anchor_device_id: str,
    resource_id: str,
) -> tuple[tuple[str, int, int], ...]:
    allocations = []
    for owned_device_id in owned_devices:
        expected_share_key = _resident_weight_share_key(
            ticket, owned_device_id
        )
        demands = tuple(
            demand
            for demand in ticket.execution_plan.memory_demands
            if demand.demand_id
                == "weights:" + owned_device_id
            and demand.device_id == owned_device_id
            and demand.kind == "model_weights"
            and demand.lifetime == "resident"
            and demand.replacement_group == resource_id
            and demand.share_key == expected_share_key
        )
        if len(demands) != 1 or demands[0].required_bytes <= 0:
            raise RuntimeResidencyProjectionError(
                "exclusive transition lacks exact weight demand: "
                + owned_device_id
            )
        memory_resource_id = (
            catalog.executor_by_device[
                owned_device_id
            ].memory_resource_id
        )
        reclaimable_bytes = sum(
            demand.required_bytes
            for demand in ticket.execution_plan.memory_demands
            if demand.device_id == owned_device_id
            and demand.resource_id == memory_resource_id
        )
        if reclaimable_bytes < demands[0].required_bytes:
            raise RuntimeResidencyProjectionError(
                "exclusive transition allocation is incomplete: "
                + owned_device_id
            )
        allocations.append((
            owned_device_id,
            demands[0].required_bytes,
            reclaimable_bytes,
        ))
    if not allocations:
        raise RuntimeResidencyProjectionError(
            "exclusive transition lacks exact weight demand: "
            + anchor_device_id
        )
    return tuple(allocations)


def _append_transition_events(
    events: list[_ProjectionEvent],
    event_keys: set[tuple[str, str, str]],
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    manifest: ModelManifest,
    transition: RuntimeTransitionPlan,
    lease_by_resource: Mapping[str, object],
    exclusive_by_device: Mapping[str, str],
) -> None:
    coordinator = catalog.composite_executor_by_id.get(
        transition.executor_id or ticket.binding.executor_id
    )
    replacement_group_by_device = (
        {}
        if coordinator is None
        else dict(coordinator.replacement_group_by_device)
    )
    devices_by_resource: dict[str, list[str]] = {}
    for device_id in transition.prepares_device_ids:
        resource_id = replacement_group_by_device.get(
            device_id,
            exclusive_by_device.get(device_id),
        )
        if resource_id is None:
            continue
        devices_by_resource.setdefault(resource_id, []).append(
            device_id
        )
    for resource_id, group_devices in sorted(
        devices_by_resource.items()
    ):
        lease = lease_by_resource.get(resource_id)
        if (
            resource_id not in transition.resource_ids
            or lease is None
        ):
            continue
        event_key = (
            ticket.ticket_id,
            transition.transition_id,
            resource_id,
        )
        if event_key in event_keys:
            continue
        event_keys.add(event_key)
        owned_devices = tuple(sorted(group_devices))
        anchor_device_id = (
            transition.device_id
            if transition.device_id in owned_devices
            else owned_devices[0]
        )
        allocations = _group_weight_allocations(
            catalog, ticket, owned_devices, anchor_device_id, resource_id
        )
        group_evictions = tuple(
            eviction for eviction in transition.evictions
            if eviction.device_id in owned_devices
            and (
                eviction.replacement_group == resource_id
                or (
                    eviction.replacement_group is None
                    and exclusive_by_device.get(eviction.device_id)
                        == resource_id
                )
            )
        )
        windows = runtime_preparation_windows(ticket) if ticket.prepare_lease_tokens else {}
        events.append(_ProjectionEvent(
            start_us=windows.get(lease.token, (lease.start_us, lease.predicted_end_us))[0],
            end_us=max(
                ticket.final_reserved_until_us.get(row.token, row.predicted_end_us)
                for row in ticket.decision.leases
                if row.resource_id == lease.resource_id
            ),
            attempt_index=ticket.attempt_index,
            ticket_id=ticket.ticket_id,
            resource_id=resource_id,
            anchor_device_id=anchor_device_id,
            executor_id=(
                transition.executor_id
                or ticket.binding.executor_id
            ),
            ticket=ticket,
            manifest=manifest,
            allocations=allocations,
            evictions=group_evictions,
            transition=transition,
        ))


def _collect_projection_events(
    catalog: RuntimeCapabilityCatalog,
    rows: tuple[RuntimeRequestTicket, ...],
    manifests: Mapping[str, ModelManifest],
    *,
    exclude_request_id: str | None,
    exclusive_by_device: Mapping[str, str],
) -> list[_ProjectionEvent]:
    events: list[_ProjectionEvent] = []
    event_keys: set[tuple[str, str, str]] = set()
    for ticket in rows:
        if (
            ticket.request.request_id == exclude_request_id
            or ticket.dispatch_state not in {"QUEUED", "ACQUIRED"}
            or ticket.transition_status not in {"PENDING", "COMPLETED"}
            or ticket.execution_plan is None
        ):
            continue
        manifest = manifests.get(ticket.model.model_id)
        if (
            manifest is None
            or manifest.artifact_sha256 != ticket.model.artifact_sha256
            or manifest.artifact_bytes != ticket.model.artifact_bytes
        ):
            raise RuntimeResidencyProjectionError(
                "projected residency model identity differs"
            )
        for transition in ticket.execution_plan.transitions:
            if transition.target_state != "hot":
                continue
            _append_transition_events(
                events,
                event_keys,
                catalog,
                ticket,
                manifest,
                transition,
                ticket.transition_lease_by_resource(transition),
                exclusive_by_device,
            )
    return events


def _causal_projection_floors(
    events: Sequence[_ProjectionEvent],
    rows: tuple[RuntimeRequestTicket, ...],
    causal_predecessors: Mapping[str, Sequence[str]],
) -> dict[str, tuple[int, int]]:
    """Order each attempt after the queued attempts that gate its dispatch."""
    ticket_by_request = {row.request.request_id: row for row in rows}
    first_start: dict[str, int] = {}
    for event in events:
        request_id = event.ticket.request.request_id
        first_start[request_id] = min(
            first_start.get(request_id, event.start_us), event.start_us
        )
    resolved: dict[str, tuple[int, int]] = {}
    visiting: set[str] = set()

    def resolve(request_id: str) -> tuple[int, int]:
        if request_id in resolved:
            return resolved[request_id]
        if request_id in visiting:
            raise RuntimeResidencyProjectionError(
                "residency projection causal order is cyclic"
            )
        visiting.add(request_id)
        floor_us = 0
        depth = 0
        for predecessor_id in causal_predecessors.get(request_id, ()):
            predecessor = ticket_by_request.get(predecessor_id)
            if predecessor is None:
                continue
            predecessor_floor, predecessor_depth = resolve(predecessor_id)
            floor_us = max(
                floor_us,
                predecessor_floor,
                first_start.get(
                    predecessor_id, predecessor.decision.start_us
                ),
            )
            depth = max(depth, predecessor_depth + 1)
        visiting.remove(request_id)
        resolved[request_id] = (floor_us, depth)
        return resolved[request_id]

    for request_id in ticket_by_request:
        resolve(request_id)
    return resolved


def _order_projection_events(
    events: Sequence[_ProjectionEvent],
    rows: tuple[RuntimeRequestTicket, ...],
    preserve_ticket_order: bool,
    causal_predecessors: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[int, _ProjectionEvent]]:
    event_order_by_ticket_id = {
        ticket.ticket_id: index for index, ticket in enumerate(rows)
    }
    if preserve_ticket_order:
        return [
            (row.start_us, row)
            for row in sorted(
                events,
                key=lambda row: (
                    event_order_by_ticket_id[row.ticket_id],
                    row.start_us,
                    row.end_us,
                    row.resource_id,
                ),
            )
        ]
    floors = (
        {}
        if not causal_predecessors
        else _causal_projection_floors(events, rows, causal_predecessors)
    )

    def effective(row: _ProjectionEvent) -> tuple[int, int]:
        floor_us, depth = floors.get(row.ticket.request.request_id, (0, 0))
        return max(row.start_us, floor_us), depth

    return [
        (effective(row)[0], row)
        for row in sorted(
            events,
            key=lambda row: (
                effective(row),
                row.start_us,
                row.end_us,
                row.attempt_index,
                row.ticket_id,
                row.resource_id,
            ),
        )
    ]


def _current_for_owned_device(
    state: _ProjectionState,
    ticket: RuntimeRequestTicket,
    executor_id: str,
    owned_device_id: str,
    evictions: Sequence[RuntimeResidencyEviction],
    allocation: tuple[str, int, int],
    target_tensor_ids: Sequence[str],
    catalog: RuntimeCapabilityCatalog,
    resource_id: str,
) -> ModelResidencyObservation | None:
    residency_by_key = state.residency_by_key
    current_options = tuple(row for row in state.current_rows_by_device.get(owned_device_id, ())
                            if catalog.residency_group(row.executor_id, owned_device_id) == resource_id)
    matching_evictions = tuple(
        eviction for eviction in evictions
        if eviction.device_id == owned_device_id
    )
    current = residency_by_key.get((
        ticket.model.model_id,
        ticket.model.artifact_sha256,
        owned_device_id,
    ))
    if current is not None and not (
        current.resident_bytes == allocation[1]
        and (
            current.executor_id is None
            or current.executor_id == executor_id
        )
        and set(target_tensor_ids).issubset(current.resident_tensor_ids)
    ):
        current = None
    if current is None:
        for eviction in matching_evictions:
            observed = residency_by_key.get((
                eviction.model_id,
                eviction.artifact_sha256,
                eviction.device_id,
            ))
            if (
                observed is not None
                and _eviction_matches_observation(
                    eviction, observed
                )
            ):
                current = observed
                break
    if current is None:
        matching_artifacts = {
            eviction.artifact_sha256
            for eviction in matching_evictions
        }
        matching_current = tuple(
            row for row in current_options
            if row.artifact_sha256 in matching_artifacts
            and any(
                _eviction_matches_observation(eviction, row)
                for eviction in matching_evictions
            )
        )
        if len(matching_current) == 1:
            current = matching_current[0]
    if current is None:
        current = next((
            row for row in current_options
            if row.model_id == ticket.model.model_id
            and row.artifact_sha256
                == ticket.model.artifact_sha256
        ), None)
    if current is None:
        current = next(iter(current_options), None)
    if current is None:
        current = residency_by_key.get((
            ticket.model.model_id,
            ticket.model.artifact_sha256,
            owned_device_id,
        ))
    if current is None:
        if len(matching_evictions) == 1:
            eviction = matching_evictions[0]
            current = residency_by_key.get((
                eviction.model_id,
                eviction.artifact_sha256,
                eviction.device_id,
            ))
    if current is not None and catalog.residency_group(current.executor_id, owned_device_id) != resource_id:
        return None
    return current


def _measured_anchor_allocations(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    event: _ProjectionEvent,
    allocations: tuple[tuple[str, int, int], ...],
) -> tuple[tuple[str, int, int], ...]:
    device_id = event.anchor_device_id
    anchor = next(
        row for row in allocations if row[0] == device_id
    )
    measured_bytes = _pending_exclusive_allocation_bytes(
        snapshot,
        catalog,
        event.ticket,
        device_id,
        event.resource_id,
        event.start_us,
        event.end_us,
        anchor[2],
    )
    if measured_bytes != anchor[2]:
        allocations = tuple(
            (row[0], row[1], measured_bytes)
            if row[0] == device_id else row
            for row in allocations
        )
    return allocations


def _allocations_already_observed(
    event: _ProjectionEvent,
    owned_device_ids: tuple[str, ...],
    current_by_owned_device: Mapping[str, ModelResidencyObservation | None],
    allocation_by_device: Mapping[str, tuple[str, int, int]],
    target_tensor_ids_by_device: Mapping[str, Sequence[str]],
) -> bool:
    ticket = event.ticket
    return all(
        (current := current_by_owned_device[owned_device_id])
            is not None
        and (
            any(row.transition_id == event.transition.transition_id and row.status == "COMPLETED"
                for row in ticket.transition_receipts)
            or (
                event.transition.source_state
                    == event.transition.target_state == "hot"
                and not event.evictions
                and current.state == "hot"
                and current.executor_id == event.executor_id
                and current.resident_geometry_sha256
                    == _resident_geometry_sha256(ticket, owned_device_id)
            )
        )
        and current.model_id == ticket.model.model_id
        and current.artifact_sha256 == ticket.model.artifact_sha256
        and current.resident_bytes
            == allocation_by_device[owned_device_id][1]
        and set(target_tensor_ids_by_device[owned_device_id]).issubset(
            current.resident_tensor_ids
        )
        for owned_device_id in owned_device_ids
    )


def _allocations_same_projection(
    state: _ProjectionState,
    ticket: RuntimeRequestTicket,
    executor_id: str,
    owned_device_ids: tuple[str, ...],
    current_by_owned_device: Mapping[str, ModelResidencyObservation | None],
) -> bool:
    return all(
        (current := current_by_owned_device[owned_device_id])
            is not None
        and current.model_id == ticket.model.model_id
        and current.artifact_sha256 == ticket.model.artifact_sha256
        and state.projected_executor_by_device.get(
            owned_device_id, current.executor_id
        ) == executor_id
        for owned_device_id in owned_device_ids
    )


def _validate_replacement_evictions(
    state: _ProjectionState,
    event: _ProjectionEvent,
    current_by_owned_device: Mapping[str, ModelResidencyObservation | None],
    owned_device_ids: tuple[str, ...],
    *,
    authoritative_transition: bool,
    transition_in_progress: bool,
) -> None:
    ticket = event.ticket
    executor_id = event.executor_id
    for current in current_by_owned_device.values():
        if current is not None and (
            current.model_id != ticket.model.model_id
            or current.artifact_sha256
                != ticket.model.artifact_sha256
            or (
                current.executor_id is not None
                and current.executor_id != executor_id
            )
        ):
            _validate_projected_current_eviction(
                current,
                event.transition,
                ticket,
                allow_in_progress=transition_in_progress,
            )
    _validate_projected_evictions(
        state.residency_by_key,
        event.transition,
        ticket,
        owned_device_ids,
        allow_missing=authoritative_transition,
        allow_in_progress=transition_in_progress,
    )


def _reflected_occupancy_by_resource(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    allocations: tuple[tuple[str, int, int], ...],
    owned_device_set: frozenset[str],
    start_us: int,
    *,
    authoritative_transition: bool,
) -> dict[str, int]:
    reflected_by_resource: dict[str, int] = {}
    allocation_by_resource: dict[str, int] = {}
    for (
        owned_device_id,
        _owned_resident_bytes,
        owned_reclaimable_bytes,
    ) in allocations:
        memory_resource_id = catalog.executor_by_device[
            owned_device_id
        ].memory_resource_id
        allocation_by_resource[memory_resource_id] = (
            allocation_by_resource.get(memory_resource_id, 0)
            + owned_reclaimable_bytes
        )
    weight_bytes_by_resource: dict[str, int] = {}
    for demand in ticket.execution_plan.memory_demands:
        if (
            demand.device_id not in owned_device_set
            or demand.kind != "model_weights"
        ):
            continue
        weight_bytes_by_resource[demand.resource_id] = (
            weight_bytes_by_resource.get(demand.resource_id, 0)
            + demand.required_bytes
        )
    for memory_resource_id, allocation_bytes in (
        allocation_by_resource.items()
    ):
        baseline_occupied = (
            ticket.runtime_observation.memory_occupied_bytes.get(
                memory_resource_id
            )
        )
        capacity = snapshot.memory.capacities.get(
            memory_resource_id
        )
        if baseline_occupied is None or capacity is None:
            continue
        if (
            authoritative_transition
            and allocation_bytes <= capacity.occupied_bytes
            and capacity.occupied_bytes + capacity.reserve_bytes
                == capacity.capacity_bytes
        ):
            reflected_by_resource[memory_resource_id] = (
                allocation_bytes
            )
            continue
        observed_delta = max(
            0, capacity.occupied_bytes - baseline_occupied
        )
        exact_pending_bytes = 0
        if (
            snapshot.captured_at_us >= start_us
            and capacity.occupied_bytes in {
                allocation_bytes,
                weight_bytes_by_resource.get(
                    memory_resource_id, -1
                ),
            }
        ):
            exact_pending_bytes = capacity.occupied_bytes
        reflected_by_resource[memory_resource_id] = min(
            allocation_bytes,
            max(observed_delta, exact_pending_bytes),
        )
    return reflected_by_resource


def _apply_projected_allocation_occupancy(
    state: _ProjectionState,
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    ticket: RuntimeRequestTicket,
    allocations: tuple[tuple[str, int, int], ...],
    pending_replacement: Mapping[str, int],
    reflected_by_resource: dict[str, int],
) -> None:
    projected_occupied = state.projected_occupied
    for (
        owned_device_id,
        _owned_resident_bytes,
        owned_reclaimable_bytes,
    ) in allocations:
        memory_resource_id = catalog.executor_by_device[
            owned_device_id
        ].memory_resource_id
        if memory_resource_id in pending_replacement:
            continue
        capacity = snapshot.memory.capacities.get(
            memory_resource_id
        )
        if capacity is None:
            raise RuntimeResidencyProjectionError(
                "projected allocation memory resource is absent: "
                + memory_resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )
        reflected_bytes = min(
            owned_reclaimable_bytes,
            reflected_by_resource.get(memory_resource_id, 0),
        )
        reflected_by_resource[memory_resource_id] = (
            reflected_by_resource.get(memory_resource_id, 0)
            - reflected_bytes
        )
        projected_occupied[memory_resource_id] += (
            owned_reclaimable_bytes - reflected_bytes
        )
        if (
            projected_occupied[memory_resource_id]
            + capacity.reserve_bytes
            > capacity.capacity_bytes
        ):
            raise RuntimeResidencyProjectionError(
                "projected allocation exceeds memory capacity: "
                + memory_resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )
    for memory_resource_id, occupied_bytes in (
        pending_replacement.items()
    ):
        capacity = snapshot.memory.capacities.get(
            memory_resource_id
        )
        if capacity is None:
            raise RuntimeResidencyProjectionError(
                "projected allocation memory resource is absent: "
                + memory_resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )
        projected_occupied[memory_resource_id] = occupied_bytes
        if occupied_bytes + capacity.reserve_bytes > (
            capacity.capacity_bytes
        ):
            raise RuntimeResidencyProjectionError(
                "projected allocation exceeds memory capacity: "
                + memory_resource_id,
                request_id=ticket.request.request_id,
                ticket_id=ticket.ticket_id,
            )


def _project_replacement_occupancy(
    state: _ProjectionState,
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    event: _ProjectionEvent,
    allocations: tuple[tuple[str, int, int], ...],
    owned_device_set: frozenset[str],
    *,
    authoritative_transition: bool,
) -> None:
    ticket = event.ticket
    evictions = event.evictions
    pending_replacement = _authoritative_replacement_occupancy(
        snapshot,
        catalog,
        ticket,
        event.resource_id,
        event.start_us,
        event.end_us,
        allocations,
        evictions,
    )
    reflected_by_resource: dict[str, int] = {}
    if not evictions:
        reflected_by_resource = _reflected_occupancy_by_resource(
            snapshot,
            catalog,
            ticket,
            allocations,
            owned_device_set,
            event.start_us,
            authoritative_transition=authoritative_transition,
        )
    if evictions:
        projected, ceilings = _project_atomic_replacement_occupancy(
            snapshot,
            catalog,
            ticket,
            event.resource_id,
            allocations,
            evictions,
            state.projected_occupied,
            pending_replacement,
            state.replacement_ceiling,
        )
        state.projected_occupied = dict(projected)
        state.replacement_ceiling = dict(ceilings)
    else:
        _apply_projected_allocation_occupancy(
            state,
            snapshot,
            catalog,
            ticket,
            allocations,
            pending_replacement,
            reflected_by_resource,
        )


def _advance_projection_generation(
    state: _ProjectionState,
    event: _ProjectionEvent,
    current_by_owned_device: Mapping[str, ModelResidencyObservation | None],
    owned_device_ids: tuple[str, ...],
    owned_device_set: frozenset[str],
    *,
    same_projection: bool,
) -> int:
    if same_projection:
        return max(
            current.generation
            for current in current_by_owned_device.values()
            if current is not None
        )
    generation = max(
        state.maximum_generation.get(owned_device_id, 0)
        for owned_device_id in owned_device_ids
    ) + 1
    for owned_device_id in owned_device_ids:
        state.maximum_generation[owned_device_id] = generation
    for current in current_by_owned_device.values():
        if current is None:
            continue
        state.residency_by_key.pop((
            current.model_id,
            current.artifact_sha256,
            current.device_id,
        ), None)
    for eviction in event.evictions:
        if eviction.device_id in owned_device_set:
            state.residency_by_key.pop((
                eviction.model_id,
                eviction.artifact_sha256,
                eviction.device_id,
            ), None)
    return generation


def _record_projected_residency(
    state: _ProjectionState,
    event: _ProjectionEvent,
    allocations: tuple[tuple[str, int, int], ...],
    generation: int,
) -> None:
    ticket = event.ticket
    for (
        owned_device_id,
        owned_resident_bytes,
        owned_reclaimable_bytes,
    ) in allocations:
        projected = ModelResidencyObservation(
            model_id=ticket.model.model_id,
            artifact_sha256=ticket.model.artifact_sha256,
            device_id=owned_device_id,
            state="hot",
            resident_tensor_ids=_target_tensor_ids(
                ticket, event.manifest, owned_device_id
            ),
            resident_bytes=owned_resident_bytes,
            generation=generation,
            executor_id=event.executor_id,
            reclaimable_bytes=owned_reclaimable_bytes,
            resident_geometry_sha256=(
                _resident_geometry_sha256(ticket, owned_device_id)
            ),
        )
        state.residency_by_key[
            (
                projected.model_id,
                projected.artifact_sha256,
                projected.device_id,
            )
        ] = projected
        state.current_by_device[owned_device_id] = projected
        state.current_rows_by_device[owned_device_id] = [
            row for row in state.residency_by_key.values()
            if row.device_id == owned_device_id and row.state in {"hot", "warm"}
            and row.resident_bytes > 0
        ]
        state.projected_executor_by_device[owned_device_id] = (
            event.executor_id
        )


def _apply_projection_event(
    state: _ProjectionState,
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    event: _ProjectionEvent,
) -> None:
    ticket = event.ticket
    executor_id = event.executor_id
    evictions = event.evictions
    allocations = event.allocations
    owned_device_ids = tuple(row[0] for row in allocations)
    owned_device_set = frozenset(owned_device_ids)
    allocation_by_device = {
        row[0]: row for row in allocations
    }
    target_tensor_ids_by_device = {
        owned_device_id: _target_tensor_ids(
            ticket, event.manifest, owned_device_id
        )
        for owned_device_id in owned_device_ids
    }
    current_by_owned_device = {
        owned_device_id: _current_for_owned_device(
            state,
            ticket,
            executor_id,
            owned_device_id,
            evictions,
            allocation_by_device[owned_device_id],
            target_tensor_ids_by_device[owned_device_id],
            catalog,
            event.resource_id,
        )
        for owned_device_id in owned_device_ids
    }
    if (
        current_by_owned_device.get(event.anchor_device_id) is None
        and not evictions
    ):
        allocations = _measured_anchor_allocations(
            snapshot, catalog, event, allocations
        )
        allocation_by_device = {
            row[0]: row for row in allocations
        }
    already_observed = _allocations_already_observed(
        event,
        owned_device_ids,
        current_by_owned_device,
        allocation_by_device,
        target_tensor_ids_by_device,
    )
    if already_observed:
        for owned_device_id in owned_device_ids:
            state.projected_executor_by_device[
                owned_device_id
            ] = executor_id
        return
    same_projection = _allocations_same_projection(
        state,
        ticket,
        executor_id,
        owned_device_ids,
        current_by_owned_device,
    )
    authoritative_transition = _authoritative_transition_covers_snapshot(
        snapshot,
        catalog,
        ticket,
        event.resource_id,
        event.start_us,
        event.end_us,
    )
    transition_in_progress = (
        authoritative_transition
        and ticket.dispatch_state == "ACQUIRED"
        and ticket.transition_status == "PENDING"
        and not any(row.transition_id == event.transition.transition_id
                    for row in ticket.transition_receipts)
    )
    if not already_observed and not same_projection:
        _validate_replacement_evictions(
            state,
            event,
            current_by_owned_device,
            owned_device_ids,
            authoritative_transition=authoritative_transition,
            transition_in_progress=transition_in_progress,
        )
        _project_replacement_occupancy(
            state,
            snapshot,
            catalog,
            event,
            allocations,
            owned_device_set,
            authoritative_transition=authoritative_transition,
        )
    generation = _advance_projection_generation(
        state,
        event,
        current_by_owned_device,
        owned_device_ids,
        owned_device_set,
        same_projection=same_projection,
    )
    _record_projected_residency(state, event, allocations, generation)


def _projected_snapshot(
    snapshot: HeterogeneousRuntimeSnapshot,
    state: _ProjectionState,
) -> HeterogeneousRuntimeSnapshot:
    return replace(
        snapshot,
        memory=replace(
            snapshot.memory,
            capacities={
                resource_id: replace(
                    capacity,
                    occupied_bytes=state.projected_occupied[resource_id],
                )
                for resource_id, capacity in (
                    snapshot.memory.capacities.items()
                )
            },
        ),
        residency=tuple(sorted(
            state.residency_by_key.values(),
            key=lambda row: (
                row.model_id, row.artifact_sha256, row.device_id
            ),
        )),
    )


def project_scheduler_residency(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    tickets: Sequence[RuntimeRequestTicket],
    manifests: Mapping[str, ModelManifest],
    *,
    exclude_request_id: str | None = None,
    stop_before_ticket_id: str | None = None,
    stop_before_us: int | None = None,
    preserve_ticket_order: bool = False,
    causal_predecessors: Mapping[str, Sequence[str]] | None = None,
) -> HeterogeneousRuntimeSnapshot:
    """Return residency at the end of all scheduler-owned transition leases.

    Without an explicit ticket order, events follow lease start but never
    precede the events of queued attempts that gate their dispatch.
    """

    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise RuntimeResidencyProjectionError(
            "residency projection snapshot is invalid"
        )
    if not isinstance(catalog, RuntimeCapabilityCatalog):
        raise RuntimeResidencyProjectionError(
            "residency projection catalog is invalid"
        )
    rows = tuple(tickets)
    _validate_projection_arguments(
        snapshot,
        catalog,
        rows,
        exclude_request_id=exclude_request_id,
        stop_before_ticket_id=stop_before_ticket_id,
        stop_before_us=stop_before_us,
        preserve_ticket_order=preserve_ticket_order,
    )
    exclusive_by_device = {
        device_id: capability.exclusive_residency_resource_id
        for device_id, capability in catalog.executor_by_device.items()
        if capability.exclusive_residency_resource_id is not None
    }
    if not exclusive_by_device:
        return snapshot

    state = _initial_projection_state(snapshot, exclusive_by_device, catalog)
    events = _collect_projection_events(
        catalog,
        rows,
        manifests,
        exclude_request_id=exclude_request_id,
        exclusive_by_device=exclusive_by_device,
    )
    if (
        stop_before_ticket_id is not None
        and stop_before_ticket_id not in {
            row.ticket_id for row in events
        }
    ):
        raise RuntimeResidencyProjectionError(
            "residency projection cutoff transition is absent"
        )

    for effective_start_us, event in _order_projection_events(
        events, rows, preserve_ticket_order, causal_predecessors
    ):
        if event.ticket_id == stop_before_ticket_id:
            break
        if (
            stop_before_us is not None
            and not preserve_ticket_order
            and effective_start_us >= stop_before_us
        ):
            break
        _apply_projection_event(state, snapshot, catalog, event)

    if not events:
        return snapshot
    return _projected_snapshot(snapshot, state)


def project_scheduler_residency_with_token(
    snapshot: HeterogeneousRuntimeSnapshot,
    catalog: RuntimeCapabilityCatalog,
    tickets: Sequence[RuntimeRequestTicket],
    manifests: Mapping[str, ModelManifest],
    token: RuntimeResidencyProjectionToken,
    *,
    candidate_start_us: int,
) -> HeterogeneousRuntimeSnapshot:
    """Project only the predecessor transition named by an exact token."""

    if not isinstance(token, RuntimeResidencyProjectionToken):
        raise RuntimeResidencyProjectionError(
            "residency projection token is invalid"
        )
    if type(candidate_start_us) is not int or candidate_start_us < 0:
        raise RuntimeResidencyProjectionError(
            "residency projection candidate start is invalid"
        )
    ticket = next((
        row for row in tickets
        if row.ticket_id == token.predecessor_ticket_id
        and row.request.request_id == token.predecessor_request_id
    ), None)
    if ticket is None or ticket.dispatch_state not in {"QUEUED", "ACQUIRED"}:
        raise RuntimeResidencyProjectionError(
            "residency projection predecessor is stale",
            request_id=token.predecessor_request_id,
            ticket_id=token.predecessor_ticket_id,
        )
    if candidate_start_us < token.ready_at_us:
        raise RuntimeResidencyProjectionError(
            "residency projection candidate precedes transition readiness",
            request_id=token.predecessor_request_id,
            ticket_id=token.predecessor_ticket_id,
        )
    rebuilt = runtime_residency_projection_token(
        ticket, token.layout_generation
    )
    if rebuilt != token:
        raise RuntimeResidencyProjectionError(
            "residency projection token is stale",
            request_id=token.predecessor_request_id,
            ticket_id=token.predecessor_ticket_id,
        )
    capacities = dict(snapshot.memory.capacities)
    for resource_id, delta_bytes in (
        token.memory_delta_bytes_by_resource.items()
    ):
        capacity = capacities.get(resource_id)
        baseline_occupied = (
            ticket.runtime_observation.memory_occupied_bytes.get(
                resource_id
            )
        )
        if capacity is None or baseline_occupied is None:
            raise RuntimeResidencyProjectionError(
                "residency projection memory resource is absent: "
                + resource_id,
                request_id=token.predecessor_request_id,
                ticket_id=token.predecessor_ticket_id,
            )
        projected_occupied = baseline_occupied + delta_bytes
        low = min(baseline_occupied, projected_occupied)
        high = max(baseline_occupied, projected_occupied)
        if (
            capacity.occupied_bytes < low
            or capacity.occupied_bytes > high
            or projected_occupied < 0
            or projected_occupied + capacity.reserve_bytes
                > capacity.capacity_bytes
        ):
            raise RuntimeResidencyProjectionError(
                "residency projection memory delta is stale: "
                + resource_id,
                request_id=token.predecessor_request_id,
                ticket_id=token.predecessor_ticket_id,
            )
        capacities[resource_id] = replace(
            capacity, occupied_bytes=projected_occupied
        )
    return replace(
        snapshot,
        memory=replace(snapshot.memory, capacities=capacities),
    )
