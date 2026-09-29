"""PhoneResidencyMixin offline inputs operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.capacity import DeviceMemoryCapacity
from ..._internal.lifecycle import PhoneTelemetryUnavailable, UnifiedScheduleError
from ..._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeExecutorState,
    RuntimePhoneSessionCapability,
)
from ..._internal.model_placement_controller import ModelPhoneResidencyLayout
from ..._internal.phone_shards import PhoneFfnResidencyLayout, generate_mixed_ffn_residency_layouts
from ..._internal.offline_phone_residency import (
    OfflinePhoneResidencyPlan,
    select_offline_resident_superset,
)
from ..._internal.runtime_placement import RuntimePlacementSnapshot
from ..common import _phone_shard_structure
from .common import _PhoneDemandDiscovery, _PhoneMemoryBudget


def _offline_phone_target(
    controller,
    discovery: _PhoneDemandDiscovery,
    queued_work_by_artifact: Mapping[str, int],
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[
    PhoneFfnResidencyLayout,
    tuple[RuntimePhoneSessionCapability, ...],
    _PhoneMemoryBudget,
]:
    if (
        not discovery.demand_rows
        or discovery.sessions is None
        or discovery.helper_id is None
    ):
        raise UnifiedScheduleError(
            "offline phone residency demand is unavailable"
        )
    sessions = tuple(sorted(
        discovery.sessions, key=lambda row: row.session_id
    ))
    if controller._maximum_phone_sessions is not None:
        sessions = sessions[:controller._maximum_phone_sessions]
    shared_domains = {
        (
            row.shared_compute_resource_id,
            row.shared_transport_resource_ids,
        )
        for row in sessions
    }
    if len(shared_domains) != 1:
        raise UnifiedScheduleError(
            "offline phone sessions do not share one resource domain"
        )
    memory = controller._phone_memory_budget(
        discovery.helper_id, sessions, snapshot
    )
    transition_costs, _latencies = controller._phone_transition_estimates(
        discovery.helper_id, sessions, queued_work_by_artifact
    )
    demands, packing_sessions = controller._fixed_phone_inputs(discovery, sessions)
    layouts = generate_mixed_ffn_residency_layouts(
        demands,
        packing_sessions,
        shard_storage=controller._phone_ffn_shard_storage,
        phone_wide_limit_bytes=memory.phone_wide_limit,
        current_shards=(
            () if memory.current is None else memory.current.shards
        ),
        transition_energy_uj_by_session=transition_costs,
    )
    if controller._fixed_phone_residency is not None:
        layouts = tuple(row for row in layouts if controller._fixed_phone_layout_matches(row))
        if len(layouts) != 1:
            raise UnifiedScheduleError(
                "fixed residency assignment is infeasible; capacity or shard coverage differs"
            )
    try:
        target = select_offline_resident_superset(layouts)
    except ValueError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    if target.resident_bytes > memory.phone_wide_limit:
        raise UnifiedScheduleError(
            "offline phone target exceeds its memory ledger"
        )
    return target, sessions, memory


def _materialize_offline_phone_helper(
    controller,
    request: Request,
    model_id: str,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
):
    manifest = controller.runtime_model_manifest(model_id)
    candidate_set = controller._generate_automated_candidate_set(
        request,
        manifest,
        snapshot,
        observed_at_us,
        use_residency_holds=False,
        update_phone_residency_portfolio=False,
    )
    candidate_set = controller._candidate_set_for_ready_phone_layout(
        candidate_set, manifest.artifact_sha256, layout
    )
    candidate_set = controller._candidate_set_with_verified_partial_phone_memory(
        candidate_set,
        manifest,
        layout,
        snapshot,
        observed_at_us,
        raise_capacity_error=True,
    )
    expected_shards = tuple(
        _phone_shard_structure(row)
        for row in layout.layout.shards
        if row.artifact_sha256 == manifest.artifact_sha256
    )
    opportunities = tuple(
        row for row in controller._compact_helper_opportunities(
            candidate_set, request
        )
        if row.phone_layout_geometry_sha256
            == layout.layout.geometry_sha256
        and row.desktop_parent_route_id
            == candidate_set.baseline_route_id
        and tuple(
            _phone_shard_structure(shard)
            for shard in row.helper_operator_plan
                .execution_contract.phone_shards
        ) == expected_shards
    )
    if len(opportunities) != 1:
        candidate_diagnostics = tuple(
            (
                row.candidate_id,
                row.assisted_operator_kind,
                row.rejection_reasons,
                row.plan.adapter_parameters.get(
                    "phone_shard_set_geometry_sha256"
                ),
                tuple(
                    shard.session_id
                    for shard in row.plan.execution_contract.phone_shards
                ),
            )
            for row in candidate_set.candidates
        )
        raise UnifiedScheduleError(
            "offline phone helper opportunity is not exact; candidates="
            + repr(candidate_diagnostics)
            + "; opportunity_count="
            + str(len(opportunities))
        )
    opportunity = opportunities[0]
    authorized_plan, authorized_binding = (
        controller._authorize_phone_helper_plan(
            opportunity.helper_operator_plan,
            opportunity.helper_binding,
            layout,
            model_id=model_id,
            artifact_sha256=manifest.artifact_sha256,
        )
    )
    helper = controller._build_helper_envelope(
        artifact_sha256=manifest.artifact_sha256,
        desktop_parent_route_id=candidate_set.baseline_route_id,
        desktop_placement_sha256=str(
            candidate_set.baseline.plan.desktop_placement_sha256
        ),
        helper_plan=authorized_plan,
        helper_binding=authorized_binding,
        layout=layout,
    )
    if (
        helper.preparation_changed_session_ids
            != layout.layout.changed_session_ids
        or len(helper.preparation_transitions) != 1
    ):
        raise UnifiedScheduleError(
            "offline phone helper is not one session-specific load"
        )
    return helper


def _offline_phone_safety_snapshot(
    controller,
    plan: OfflinePhoneResidencyPlan,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[HeterogeneousRuntimeSnapshot, RuntimeExecutorState]:
    """Keep legacy evidence readable; new load admission uses live telemetry."""

    if controller._runtime_capabilities is None:
        raise UnifiedScheduleError(
            "offline phone capabilities are absent"
        )
    endpoints = dict(plan.session_endpoints)
    capabilities = tuple(
        capability
        for capability in controller._runtime_capabilities.executors
        if capability.phone_sessions
        and endpoints.items() <= {
            session.session_id: session.endpoint
            for session in capability.phone_sessions
        }.items()
    )
    if len(capabilities) != 1:
        raise UnifiedScheduleError(
            "offline phone session capability is not exact"
        )
    capability = capabilities[0]
    current = snapshot.executors.get(capability.executor_id)
    if current is None:
        raise UnifiedScheduleError(
            "offline phone safety executor is absent"
        )
    if capability.device_id in snapshot.telemetry_observations:
        reason = snapshot.telemetry_unavailable_reason(
            capability.device_id, snapshot.captured_at_us
        )
        if reason is not None:
            raise PhoneTelemetryUnavailable(reason)
        return snapshot, current
    admitted_safety_state = plan.phone_safety_state
    if admitted_safety_state is None:
        admitted_safety_state = plan.materialization_snapshot\
            .executors.get(capability.executor_id)
    effective = controller._effective_verified_phone_state(
        capability,
        current,
        capability.device_id,
        admitted_safety_state,
    )
    if effective == current:
        return snapshot, effective
    capacities = dict(snapshot.memory.capacities)
    memory_resource_id = capability.memory_resource_id
    live_capacity = capacities.get(memory_resource_id)
    if live_capacity is None:
        raise UnifiedScheduleError(
            "offline phone memory capacity is absent"
        )
    if live_capacity.available_bytes == 0:
        endpoints = dict(plan.session_endpoints)
        observed_sessions = tuple(
            row for row in snapshot.phone_session_residency
            if row.device_id == capability.device_id
        )
        if len(observed_sessions) != len(
            snapshot.phone_session_residency
        ) or any(
            row.state != "READY"
            or endpoints.get(row.session_id) != row.endpoint
            for row in observed_sessions
        ):
            raise UnifiedScheduleError(
                "offline phone memory session map is not exact"
            )
        pool = controller._runtime_capabilities.placement_profile\
            .memory_pools[memory_resource_id]
        occupied_bytes = sum(
            row.resident_bytes for row in observed_sessions
        )
        reserve_bytes = (
            pool.reserved_bytes
            + plan.persistent_service_reserve_bytes
        )
        if occupied_bytes + reserve_bytes > pool.capacity_bytes:
            raise UnifiedScheduleError(
                "offline phone memory ledger exceeds capacity"
            )
        capacities[memory_resource_id] = DeviceMemoryCapacity(
            resource_id=memory_resource_id,
            capacity_bytes=pool.capacity_bytes,
            occupied_bytes=occupied_bytes,
            reserve_bytes=reserve_bytes,
        )
    effective_snapshot = replace(
        snapshot,
        executors={
            **snapshot.executors,
            capability.executor_id: effective,
        },
        memory=RuntimePlacementSnapshot(
            snapshot_id=snapshot.memory.snapshot_id,
            captured_at_us=snapshot.memory.captured_at_us,
            valid_until_us=snapshot.memory.valid_until_us,
            capacities=capacities,
        ),
    )
    return effective_snapshot, effective


def _offline_phone_materialization_snapshot(
    controller,
    plan: OfflinePhoneResidencyPlan,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[HeterogeneousRuntimeSnapshot, RuntimeExecutorState]:
    """Materialize phone loads from the admitted desktop parent view."""

    current, phone_safety_state = controller._offline_phone_safety_snapshot(
        plan, snapshot
    )
    base = plan.materialization_snapshot
    endpoints = dict(plan.session_endpoints)
    capabilities = tuple(
        capability
        for capability in controller._runtime_capabilities.executors
        if capability.phone_sessions
        and endpoints.items() <= {
            session.session_id: session.endpoint
            for session in capability.phone_sessions
        }.items()
    )
    if len(capabilities) != 1:
        raise UnifiedScheduleError(
            "offline phone session capability is not exact"
        )
    capability = capabilities[0]
    phone_memory_resource_ids = {
        capability.memory_resource_id,
        *(
            session.memory_resource_id
            for session in capability.phone_sessions
        ),
    }
    capacities = dict(base.memory.capacities)
    for resource_id in phone_memory_resource_ids:
        capacity = current.memory.capacities.get(resource_id)
        if capacity is None:
            raise UnifiedScheduleError(
                "offline phone memory capacity is absent"
            )
        capacities[resource_id] = capacity
    executors = dict(base.executors)
    executors[capability.executor_id] = current.executors[
        capability.executor_id
    ]
    return HeterogeneousRuntimeSnapshot(
        snapshot_id=snapshot.snapshot_id,
        captured_at_us=snapshot.captured_at_us,
        valid_until_us=snapshot.valid_until_us,
        memory=RuntimePlacementSnapshot(
            snapshot_id=snapshot.memory.snapshot_id,
            captured_at_us=snapshot.memory.captured_at_us,
            valid_until_us=snapshot.memory.valid_until_us,
            capacities=capacities,
        ),
        executors=executors,
        links=snapshot.links,
        residency=snapshot.residency,
        cost_features=snapshot.cost_features,
        protected_work=snapshot.protected_work,
        phone_session_residency=snapshot.phone_session_residency,
        telemetry_observations=snapshot.telemetry_observations,
    ), phone_safety_state


def _offline_phone_discovery_snapshot(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> HeterogeneousRuntimeSnapshot:
    """Expose replaceable HTP bytes only while discovering shard shapes."""

    plan_id = controller._active_offline_phone_residency_plan_id
    plan = (
        None if plan_id is None else
        controller._offline_phone_residency_plans.get(plan_id)
    )
    current = controller._model_placement_controller.ready_phone_layout()
    if plan is None or plan.state != "READY" or current is None:
        return snapshot
    effective, _safety = controller._offline_phone_safety_snapshot(
        plan, snapshot
    )
    phone_device_id = next(iter({
        row.device_id
        for row in effective.phone_session_residency
        if row.session_id in plan.session_endpoints
    }), None)
    helper = (
        None if phone_device_id is None else
        controller._runtime_capabilities.executor_by_device.get(
            phone_device_id
        )
    )
    if helper is None or not helper.phone_sessions:
        raise UnifiedScheduleError(
            "offline phone discovery executor is absent"
        )
    resource_id = helper.memory_resource_id
    capacity = effective.memory.capacities.get(resource_id)
    pool = controller._runtime_capabilities.placement_profile.memory_pools[
        resource_id
    ]
    reserve_bytes = (
        pool.reserved_bytes + plan.persistent_service_reserve_bytes
    )
    if (
        capacity is None
        or reserve_bytes > pool.capacity_bytes
    ):
        raise UnifiedScheduleError(
            "offline phone discovery memory identity differs"
        )
    capacities = dict(effective.memory.capacities)
    capacities[resource_id] = DeviceMemoryCapacity(
        resource_id=resource_id,
        capacity_bytes=pool.capacity_bytes,
        occupied_bytes=0,
        reserve_bytes=reserve_bytes,
    )
    return replace(
        effective,
        memory=RuntimePlacementSnapshot(
            snapshot_id=effective.memory.snapshot_id,
            captured_at_us=effective.memory.captured_at_us,
            valid_until_us=effective.memory.valid_until_us,
            capacities=capacities,
        ),
    )
