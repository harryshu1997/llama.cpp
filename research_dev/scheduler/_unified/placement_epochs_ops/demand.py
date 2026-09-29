"""PlacementEpochMixin demand operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    ModelDemandSnapshot,
    ModelPlacementAction,
    ModelPlacementControllerError,
    ModelPlacementTrigger,
)
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.types import canonical_sha256
from .common import _PlacementTicketDemand


def _placement_ticket_demand(
    controller,
    request: Request,
    manifest: ModelManifest,
    observed_at_us: int,
) -> _PlacementTicketDemand:
    all_tickets = controller._runtime_controller.current_tickets()
    tickets = tuple(
        row for row in all_tickets
        if row.model.artifact_sha256 == manifest.artifact_sha256
    )
    active = tuple(
        row for row in tickets if row.dispatch_state == "ACQUIRED"
    )
    queued = tuple(
        row for row in tickets
        if row.dispatch_state in {"QUEUED", "REPLAN_REQUIRED"}
    )
    incoming = not any(
        row.request.request_id == request.request_id
        for row in tickets
    )
    arrivals = tuple(
        row.request.arrival_us for row in queued
    ) + ((request.arrival_us,) if incoming else ())
    finishes = tuple(
        row.decision.finish_upper_us for row in active + queued
    )
    return _PlacementTicketDemand(
        all_tickets=all_tickets,
        active=active,
        queued=queued,
        queued_request_count=len(queued) + int(incoming),
        queued_input_tokens=(
            sum(row.request.input_tokens for row in queued)
            + (request.input_tokens if incoming else 0)
        ),
        queued_output_tokens=(
            sum(row.request.output_tokens for row in queued)
            + (request.output_tokens if incoming else 0)
        ),
        oldest_wait_us=(
            0 if not arrivals
            else max(0, observed_at_us - min(arrivals))
        ),
        predicted_drain_us=(
            0 if not finishes
            else max(0, max(finishes) - observed_at_us)
        ),
    )


def _current_placement_component(
    controller,
    manifest: ModelManifest,
    active: tuple[RuntimeRequestTicket, ...],
    confirmed_components: Mapping[str, object],
    epoch: RuntimeModelPlacementEpoch | None,
) -> str | None:
    current = None
    for ticket in active:
        if ticket.execution_plan is None:
            continue
        component = runtime_residency_component_identity(
            manifest.artifact_sha256,
            ticket.execution_plan,
            ticket.binding,
        )
        if component.session_resource_ids:
            current = component.identity_sha256
            break
    if current is None:
        phone_resources = {
            executor.exclusive_residency_resource_id
            for executor in controller._runtime_capabilities.executors
            if executor.phone_sessions
            and executor.exclusive_residency_resource_id is not None
        }
        phone_components = {
            component.identity_sha256
            for resource_id, component in confirmed_components.items()
            if resource_id in phone_resources
        }
        if len(phone_components) == 1:
            current = next(iter(phone_components))
    if current is None and epoch is not None:
        current = epoch.selected_component_identity_sha256
    return current


def _placement_available_resources(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    executor_states = snapshot.executors
    participant_health = {
        device_id
        for coordinator in controller._runtime_capabilities.composite_executors
        for state in (executor_states.get(coordinator.executor_id),)
        if state is not None and state.healthy
        for device_id in coordinator.participant_device_ids
    }
    devices = tuple(sorted({
        capability.device_id
        for capability in controller._runtime_capabilities.executors
        for state in (executor_states.get(capability.executor_id),)
        if (
            state is not None and state.healthy
            or capability.device_id in participant_health
        )
    }))
    preparable = {
        device_id
        for transition in controller._runtime_capabilities.transitions
        if transition.maturity == "QUALIFIED"
        for state in (
            None
            if transition.executor_id is None
            else executor_states.get(transition.executor_id),
        )
        if transition.executor_id is None
            or state is not None and state.healthy
        for device_id in transition.prepares_device_ids
    }
    sessions = tuple(sorted({
        session.session_id
        for capability in controller._runtime_capabilities.executors
        for session in capability.phone_sessions
        if session.ready
        and capability.device_id in participant_health
        and capability.device_id in preparable
    }))
    return devices, sessions


def _placement_demand_generations(
    controller,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    all_tickets: tuple[RuntimeRequestTicket, ...],
    confirmed_components: Mapping[str, object],
) -> tuple[str, str, str]:
    memory_generation = canonical_sha256({
        "ledger": controller._runtime_memory.snapshot(),
        "snapshot": snapshot.memory.to_json(),
    })
    resource_generation = canonical_sha256({
        "executors": [
            row.to_json() for row in snapshot.executors.values()
        ],
        "leases": [
            {
                "request_id": ticket.request.request_id,
                "route_id": ticket.decision.route_id,
                "rows": [
                    {
                        "resource_id": lease.resource_id,
                        "reserved_until_us": lease.reserved_until_us,
                        "slots": len(lease.lanes),
                        "start_us": lease.start_us,
                    }
                    for lease in ticket.decision.leases
                ],
            }
            for ticket in all_tickets
        ],
        "links": [row.to_json() for row in snapshot.links.values()],
        "schema": "runtime-resource-calendar-generation-v1",
    })
    phone_device_ids = {
        executor.device_id
        for executor in controller._runtime_capabilities.executors
        if executor.phone_sessions
    }
    residency_generation = canonical_sha256({
        "artifact_sha256": manifest.artifact_sha256,
        "confirmed_components": {
            resource_id: component.identity_sha256
            for resource_id, component in confirmed_components.items()
        },
        "rows": [
            row.to_json() for row in snapshot.residency
            if row.artifact_sha256 == manifest.artifact_sha256
            or row.device_id in phone_device_ids
        ],
        "sessions": [
            session.to_json()
            for executor in controller._runtime_capabilities.executors
            for session in executor.phone_sessions
        ],
    })
    return memory_generation, resource_generation, residency_generation


def _model_demand_snapshot(
    controller,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    epoch: RuntimeModelPlacementEpoch | None = None,
) -> ModelDemandSnapshot:
    if (
        controller._runtime_capabilities is None
        or controller._runtime_capability_generation_sha256 is None
        or controller._runtime_profile_generation_sha256 is None
        or controller._runtime_transport_generation_sha256 is None
    ):
        raise UnifiedScheduleError(
            "runtime placement generation is absent"
        )
    demand = controller._placement_ticket_demand(
        request, manifest, observed_at_us
    )
    confirmed = (
        controller._runtime_residency_cohorts.confirmed_resident_components(
            controller._runtime_capabilities, snapshot
        )
    )
    current_component = controller._current_placement_component(
        manifest, demand.active, confirmed, epoch
    )
    available_devices, available_sessions = (
        controller._placement_available_resources(snapshot)
    )
    memory_generation, resource_generation, residency_generation = (
        controller._placement_demand_generations(
            manifest, snapshot, demand.all_tickets, confirmed
        )
    )
    return ModelDemandSnapshot(
        artifact_sha256=manifest.artifact_sha256,
        observed_at_us=observed_at_us,
        active_request_count=len(demand.active),
        queued_request_count=demand.queued_request_count,
        queued_input_tokens=demand.queued_input_tokens,
        queued_output_tokens=demand.queued_output_tokens,
        oldest_queued_wait_us=demand.oldest_wait_us,
        predicted_queue_drain_us=demand.predicted_drain_us,
        current_resident_component_identity_sha256=current_component,
        available_device_ids=available_devices,
        available_session_ids=available_sessions,
        memory_generation_sha256=memory_generation,
        resource_calendar_generation_sha256=resource_generation,
        capability_generation_sha256=(
            controller._runtime_capability_generation_sha256
        ),
        profile_generation_sha256=(
            controller._runtime_profile_generation_sha256
        ),
        transport_generation_sha256=(
            controller._runtime_transport_generation_sha256
        ),
        residency_generation_sha256=residency_generation,
        learning_generation_sha256=(
            controller._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            )
        ),
    )


def _evaluate_model_placement(
    controller,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    epoch: RuntimeModelPlacementEpoch | None,
    selection_mode: str,
) -> tuple[ModelDemandSnapshot, ModelPlacementAction]:
    demand = controller._model_demand_snapshot(
        request, manifest, snapshot, observed_at_us, epoch
    )
    if selection_mode == "desktop-baseline" and epoch is not None:
        demand = replace(
            demand,
            learning_generation_sha256=(
                epoch.learning_generation_sha256
            ),
        )
    feasible = True
    if epoch is not None:
        state = snapshot.executors.get(epoch.selected_executor_id)
        feasible = state is not None and state.healthy
    try:
        action = controller._model_placement_controller.evaluate(
            ModelPlacementTrigger(
                snapshot=demand,
                epoch_sha256=(
                    None if epoch is None else epoch.epoch_sha256
                ),
                epoch_demand_generation_sha256=(
                    None
                    if epoch is None
                    else epoch.demand_generation_sha256
                ),
                epoch_pressure_bucket=(
                    None if epoch is None else epoch.pressure_bucket
                ),
                epoch_valid_until_us=(
                    None if epoch is None else epoch.valid_until_us
                ),
                selected_route_feasible=feasible,
                old_component_identity_sha256=(
                    demand.current_resident_component_identity_sha256
                ),
                proposed_component_identity_sha256=(
                    None
                    if epoch is None
                    else epoch.selected_component_identity_sha256
                ),
                maximum_latency_ppm=(
                    controller._runtime_capabilities.maximum_latency_ppm
                ),
                predicted_reuse_count=0,
            )
        )
    except ModelPlacementControllerError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    return demand, action
