"""HelperEnvelopeMixin ready plan operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterator

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeExecutorBinding
from ..._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
    RuntimeExecutorState,
)
from ..._internal.model_placement_controller import ModelPhoneResidencyLayout
from ..._internal.runtime_plan import (
    AutomatedRouteCandidate,
    HelperOpportunity,
    RuntimeExecutionPlan,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _ReadyHelperSafetyDeferred, _phone_shard_structure
from .common import _ReadyHelperParentUnavailable


def _snapshot_with_owned_base_executor(
    ticket: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> HeterogeneousRuntimeSnapshot:
    """Expose the acquired request's own base slot for rematerialization."""

    if ticket.dispatch_state != "ACQUIRED":
        raise UnifiedScheduleError(
            "helper rematerialization request is not acquired"
        )
    state = snapshot.executors.get(ticket.binding.executor_id)
    if state is None or not state.healthy:
        raise UnifiedScheduleError(
            "helper rematerialization base executor is unavailable"
        )
    if state.ready and state.free_slots > 0:
        return snapshot
    return replace(
        snapshot,
        executors={
            **snapshot.executors,
            ticket.binding.executor_id: replace(
                state,
                ready=True,
                free_slots=max(1, state.free_slots),
            ),
        },
    )


def _resolve_verified_ready_helper_plan(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
) -> tuple[RuntimeExecutionPlan, RuntimeExecutorBinding] | None:
    ready = controller._model_placement_controller.ready_phone_layout()
    plan = ticket.execution_plan
    helper = None if plan is None else plan.helper_envelope
    if (
        ready is None
        or ready.generation != layout.generation
        or ready.layout.geometry_sha256 != layout.layout.geometry_sha256
        or layout.state != "READY"
        or not layout.covers_artifact(ticket.model.artifact_sha256)
    ):
        raise UnifiedScheduleError(
            "helper rematerialization layout is not authoritative"
        )
    if plan is None or plan.desktop_placement_sha256 is None:
        raise UnifiedScheduleError(
            "helper rematerialization base plan is absent"
        )
    if helper is not None:
        return helper.helper_plan, helper.helper_binding
    template = controller._authoritative_ready_helper_template(
        artifact_sha256=ticket.model.artifact_sha256,
        desktop_parent_route_id=ticket.decision.route_id,
        desktop_placement_sha256=plan.desktop_placement_sha256,
        baseline_executor_id=ticket.binding.executor_id,
        allow_parent_route_rebind=True,
    )
    if template is not None:
        return template.helper_plan, template.helper_binding
    expected_shards = [
        _phone_shard_structure(row)
        for row in layout.layout.shards
        if row.artifact_sha256 == ticket.model.artifact_sha256
    ]
    opportunities = tuple(
        row for row in controller._request_helper_opportunities.get(
            ticket.request.request_id, ()
        )
        if (
            row.desktop_parent_route_id == ticket.decision.route_id
            and row.desktop_parent_placement_sha256
                == plan.desktop_placement_sha256
            and row.phone_layout_generation in {None, layout.generation}
            and row.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and [
                _phone_shard_structure(shard)
                for shard in row.helper_operator_plan
                    .execution_contract.phone_shards
            ] == expected_shards
        )
    )
    if len(opportunities) == 1:
        return (
            opportunities[0].helper_operator_plan,
            opportunities[0].helper_binding,
        )
    if opportunities:
        raise UnifiedScheduleError(
            "helper rematerialization opportunity is ambiguous"
        )
    historical = tuple(
        envelope
        for envelope in controller._request_helper_envelope_history.get(
            ticket.request.request_id, {}
        ).values()
        if (
            envelope.artifact_sha256
                == ticket.model.artifact_sha256
            and envelope.desktop_parent_route_id
                == ticket.decision.route_id
            and envelope.desktop_placement_sha256
                == plan.desktop_placement_sha256
            and envelope.helper_plan.baseline_executor_id
                == ticket.binding.executor_id
        )
    )
    if historical:
        endpoint_identities = {
            (
                envelope.helper_binding.executor_id,
                envelope.helper_binding.endpoint,
                envelope.helper_binding.backend,
                envelope.helper_binding.operator_plan_protocol,
                tuple(
                    (
                        participant.executor_id,
                        participant.device_id,
                        participant.endpoint,
                        participant.backend,
                        participant.resource_ids,
                    )
                    for participant in (
                        envelope.helper_binding.participants
                    )
                ),
            )
            for envelope in historical
        }
        if len(endpoint_identities) != 1:
            raise UnifiedScheduleError(
                "historical helper endpoint identity is ambiguous"
            )
        bootstrap = max(
            historical,
            key=lambda row: (
                row.phone_layout_generation,
                row.operator_plan_sha256,
            ),
        )
        return bootstrap.helper_plan, bootstrap.helper_binding
    system_templates = tuple(
        envelope
        for envelope in getattr(
            controller, "_phone_helper_endpoint_templates", {}
        ).values()
        if (
            envelope.artifact_sha256
                == ticket.model.artifact_sha256
            and envelope.desktop_placement_sha256
                == plan.desktop_placement_sha256
            and envelope.helper_plan.baseline_executor_id
                == ticket.binding.executor_id
            and envelope.helper_binding.artifact_sha256
                == ticket.model.artifact_sha256
            and envelope.helper_binding.model_id
                == ticket.model.model_id
        )
    )
    if system_templates:
        endpoint_identities = {
            controller._phone_helper_endpoint_identity(envelope)
            for envelope in system_templates
        }
        if len(endpoint_identities) != 1:
            raise UnifiedScheduleError(
                "system helper endpoint identity is ambiguous"
            )
        bootstrap = max(
            system_templates,
            key=lambda row: (
                row.phone_layout_generation,
                row.operator_plan_sha256,
            ),
        )
        return bootstrap.helper_plan, bootstrap.helper_binding
    offline_by_plan = {}
    for offline_plan in controller._offline_phone_residency_plans.values():
        for stage in offline_plan.stages:
            envelope = stage.helper_envelope
            envelope_shards = tuple(
                envelope.helper_plan.execution_contract.phone_shards
            )
            if (
                stage.state != "READY"
                or stage.layout.layout.geometry_sha256
                    != layout.layout.geometry_sha256
                or envelope.artifact_sha256
                    != ticket.model.artifact_sha256
                or envelope.phone_layout_geometry_sha256
                    != layout.layout.geometry_sha256
                or envelope.desktop_placement_sha256
                    != plan.desktop_placement_sha256
                or envelope.helper_plan.baseline_executor_id
                    != ticket.binding.executor_id
                or [
                    _phone_shard_structure(shard)
                    for shard in envelope_shards
                ] != expected_shards
                or any(
                    shard.session_generation
                        != layout.layout.session_generation_by_id.get(
                            shard.session_id
                        )
                    for shard in envelope_shards
                )
            ):
                continue
            previous = offline_by_plan.get(
                envelope.operator_plan_sha256
            )
            if previous is not None and previous != envelope:
                raise UnifiedScheduleError(
                    "offline helper bootstrap identity differs"
                )
            offline_by_plan[envelope.operator_plan_sha256] = envelope
    if len(offline_by_plan) == 1:
        offline = next(iter(offline_by_plan.values()))
        return offline.helper_plan, offline.helper_binding
    if offline_by_plan:
        raise UnifiedScheduleError(
            "offline helper bootstrap opportunity is ambiguous"
        )
    prepared_by_plan = {}
    for key, envelope in controller._request_helper_preparation_envelopes.items():
        if (
            key[2] != layout.generation
            or envelope.artifact_sha256 != ticket.model.artifact_sha256
            or envelope.desktop_parent_route_id != ticket.decision.route_id
            or envelope.desktop_placement_sha256
                != plan.desktop_placement_sha256
            or envelope.phone_layout_generation != layout.generation
            or envelope.phone_layout_geometry_sha256
                != layout.layout.geometry_sha256
            or envelope.helper_plan.baseline_executor_id
                != ticket.binding.executor_id
            or [
                _phone_shard_structure(shard)
                for shard in envelope.helper_plan
                    .execution_contract.phone_shards
            ] != expected_shards
        ):
            continue
        previous = prepared_by_plan.get(envelope.operator_plan_sha256)
        if previous is not None and previous != envelope:
            raise UnifiedScheduleError(
                "helper preparation envelope identity differs"
            )
        prepared_by_plan[envelope.operator_plan_sha256] = envelope
    if not prepared_by_plan:
        return None
    if len(prepared_by_plan) != 1:
        raise UnifiedScheduleError(
            "helper rematerialization opportunity is not exact"
        )
    prepared = next(iter(prepared_by_plan.values()))
    return prepared.helper_plan, prepared.helper_binding


def _ready_parent_helper_coordinator(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> RuntimeCompositeExecutorCapability:
    parents = tuple(
        row for row in controller._runtime_capabilities.composite_executors
        if row.baseline_executor_id == ticket.binding.executor_id
        and row.artifact_sha256 == ticket.model.artifact_sha256
        and row.route_family == "operator_split"
        and row.assisted_operator_kind == "ffn"
        and row.maturity == "QUALIFIED"
    )
    if len(parents) != 1:
        raise _ReadyHelperParentUnavailable(
            "ready layout lacks one parent-compatible helper executor: "
            + ticket.binding.executor_id
        )
    coordinator = parents[0]
    for shard in layout.layout.shards:
        if shard.artifact_sha256 != ticket.model.artifact_sha256:
            continue
        observed = tuple(
            row for row in snapshot.phone_session_residency
            if row.device_id == coordinator.helper_device_id
            and row.session_id == shard.session_id
        )
        if not observed or any(
            row.state != "READY"
            or row.endpoint != shard.endpoint
            or row.artifact_sha256 != shard.artifact_sha256
            or row.resident_geometry_sha256 != shard.resident_geometry_sha256
            or row.operator_plan_sha256 != shard.operator_plan_sha256
            or row.resident_bytes != shard.resident_bytes
            or row.session_generation != layout.layout.session_generation_by_id[
                shard.session_id
            ]
            for row in observed
        ):
            raise UnifiedScheduleError(
                "ready parent helper physical session identity differs: "
                + shard.session_id
            )
    return coordinator


def _ready_layout_compiler_view(
    controller,
    layout: ModelPhoneResidencyLayout,
) -> Iterator[None]:
    """Generate a late helper from the authoritative READY layout."""

    compilers = []
    for compiler in (
        controller._automated_compiler(),
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None and all(
            compiler is not row for row in compilers
        ):
            compilers.append(compiler)
    previous = tuple(
        (compiler, compiler.phone_residency_layout)
        for compiler in compilers
    )
    try:
        for compiler in compilers:
            compiler.set_phone_residency_layout(layout.layout)
        yield
    finally:
        for compiler, previous_layout in previous:
            compiler.set_phone_residency_layout(previous_layout)


def _verified_helper_endpoint_states(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    helper_plan: RuntimeExecutionPlan,
    helper_binding: RuntimeExecutorBinding,
) -> tuple[object, RuntimeExecutorState, object, RuntimeExecutorState, str]:
    coordinator = controller._runtime_capabilities.composite_executor_by_id.get(
        helper_binding.executor_id
    )
    contract = helper_plan.execution_contract
    phone_device_id = contract.phone_device_id
    if (
        coordinator is None
        or coordinator.route_family != "operator_split"
        or coordinator.assisted_operator_kind != "ffn"
        or coordinator.baseline_executor_id != ticket.binding.executor_id
        or coordinator.helper_device_id != phone_device_id
        or coordinator.endpoint != helper_binding.endpoint
        or coordinator.backend != helper_binding.backend
        or coordinator.operator_plan_protocol
            != helper_binding.operator_plan_protocol
        or phone_device_id is None
    ):
        raise UnifiedScheduleError(
            "helper rematerialization executor identity differs"
        )
    return controller._verified_helper_coordinator_states(
        ticket, layout, snapshot, coordinator
    )


def _verified_helper_coordinator_states(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    coordinator: RuntimeCompositeExecutorCapability,
) -> tuple[
    RuntimeCompositeExecutorCapability, RuntimeExecutorState,
    RuntimeExecutorCapability, RuntimeExecutorState, str,
]:
    phone_device_id = coordinator.helper_device_id
    helper_state = snapshot.executors.get(coordinator.executor_id)
    phone_capability = (
        controller._runtime_capabilities.executor_by_device.get(phone_device_id)
    )
    phone_state = (
        None if phone_capability is None else
        snapshot.executors.get(phone_capability.executor_id)
    )
    if (
        helper_state is None
        or not helper_state.healthy
        or phone_capability is None
        or phone_state is None
    ):
        raise UnifiedScheduleError("verified helper endpoint is unavailable")
    session_by_id = {
        row.session_id: row for row in phone_capability.phone_sessions
    }
    layout_shards = tuple(
        row for row in layout.layout.shards
        if row.artifact_sha256 == ticket.model.artifact_sha256
    )
    if not layout_shards or any(
        shard.session_id not in session_by_id
        or session_by_id[shard.session_id].endpoint != shard.endpoint
        for shard in layout_shards
    ):
        raise UnifiedScheduleError(
            "verified helper sessions differ from capabilities"
        )
    return (
        coordinator,
        helper_state,
        phone_capability,
        phone_state,
        phone_device_id,
    )


def _effective_verified_phone_state(
    controller,
    phone_capability: object,
    phone_state: RuntimeExecutorState,
    phone_device_id: str,
    phone_safety_state: RuntimeExecutorState | None,
) -> RuntimeExecutorState:
    current_safety_valid = (
        phone_state.battery_ppm > 0
        and phone_state.temperature_millic > 0
        and phone_state.temperature_millic
            <= phone_capability.maximum_temperature_millic
        and phone_state.thermal_qualified_under(
            phone_capability.maximum_thermal_status
        ) is not False
    )
    if current_safety_valid:
        return phone_state
    minimum_battery_ppm = phone_capability.minimum_battery_ppm
    phone_power = (
        controller._runtime_capabilities.phone_power_profile_by_device.get(
            phone_device_id
        )
    )
    if phone_power is not None:
        minimum_battery_ppm = phone_power.minimum_battery_ppm
    if (
        phone_safety_state is None
        or phone_safety_state.executor_id != phone_capability.executor_id
        or phone_safety_state.battery_ppm < minimum_battery_ppm
        or phone_safety_state.temperature_millic <= 0
        or phone_safety_state.temperature_millic
            > phone_capability.maximum_temperature_millic
        or phone_safety_state.thermal_qualified_under(
            phone_capability.maximum_thermal_status
        ) is False
    ):
        raise _ReadyHelperSafetyDeferred(
            "verified helper lacks an admitted phone safety sample"
        )
    return replace(
        phone_state,
        temperature_millic=phone_safety_state.temperature_millic,
        battery_ppm=phone_safety_state.battery_ppm,
        thermal_qualified=phone_safety_state.thermal_qualified,
        charging=phone_safety_state.charging,
        thermal_status=phone_safety_state.thermal_status,
    )


def _snapshot_with_verified_ready_helper(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    phone_safety_state: RuntimeExecutorState | None,
) -> HeterogeneousRuntimeSnapshot:
    """Expose only the exact helper proven by a READY layout."""

    if controller._runtime_capabilities is None:
        raise UnifiedScheduleError(
            "helper rematerialization capabilities are absent"
        )
    if ticket.dispatch_state != "ACQUIRED":
        raise UnifiedScheduleError(
            "helper rematerialization request is not acquired"
        )
    resolved = controller._resolve_verified_ready_helper_plan(
        ticket, layout
    )
    if resolved is None:
        coordinator = controller._ready_parent_helper_coordinator(
            ticket, layout, snapshot
        )
        states = controller._verified_helper_coordinator_states(
            ticket, layout, snapshot, coordinator
        )
    else:
        states = controller._verified_helper_endpoint_states(
            ticket, layout, snapshot, *resolved
        )
    (
        coordinator,
        helper_state,
        phone_capability,
        phone_state,
        phone_device_id,
    ) = states
    effective_phone_state = controller._effective_verified_phone_state(
        phone_capability,
        phone_state,
        phone_device_id,
        phone_safety_state,
    )
    return replace(
        snapshot,
        executors={
            **snapshot.executors,
            coordinator.executor_id: replace(
                helper_state,
                ready=True,
                free_slots=max(1, helper_state.free_slots),
            ),
            phone_capability.executor_id: effective_phone_state,
        },
    )


def _ready_helper_parent_rejection_reasons(
    ticket: RuntimeRequestTicket,
    generated_baseline: AutomatedRouteCandidate,
    opportunity: HelperOpportunity,
) -> tuple[str, ...]:
    """Validate a regenerated helper against the acquired desktop base."""

    plan = ticket.execution_plan
    generated_plan = generated_baseline.plan
    generated_binding = generated_baseline.binding
    ticket_binding = ticket.binding
    reasons = []
    if plan is None:
        reasons.append("BASE_PLAN_ABSENT")
        return tuple(reasons)
    if (
        not generated_baseline.baseline
        or generated_baseline.candidate_id
            != opportunity.desktop_parent_route_id
    ):
        reasons.append("GENERATED_PARENT_IDENTITY_MISMATCH")
    if (
        plan.desktop_placement_sha256 is None
        or generated_plan.desktop_placement_sha256
            != plan.desktop_placement_sha256
        or opportunity.desktop_parent_placement_sha256
            != plan.desktop_placement_sha256
    ):
        reasons.append("DESKTOP_PLACEMENT_MISMATCH")
    if (
        generated_binding.artifact_sha256
            != ticket.model.artifact_sha256
        or ticket_binding.artifact_sha256
            != ticket.model.artifact_sha256
        or generated_binding.model_id != ticket.model.model_id
        or ticket_binding.model_id != ticket.model.model_id
    ):
        reasons.append("DESKTOP_ARTIFACT_MISMATCH")
    if (
        generated_binding.executor_id != ticket_binding.executor_id
        or generated_binding.endpoint != ticket_binding.endpoint
        or generated_binding.backend != ticket_binding.backend
        or generated_binding.participants != ticket_binding.participants
    ):
        reasons.append("DESKTOP_EXECUTOR_MISMATCH")
    if (
        generated_plan.route_family != plan.route_family
        or generated_plan.device_ids != plan.device_ids
        or generated_plan.operators != plan.operators
        or generated_plan.execution_contract != plan.execution_contract
        or generated_plan.resource_ids != plan.resource_ids
    ):
        reasons.append("DESKTOP_PLAN_MISMATCH")
    if any(
        generated_plan.resource_slots[resource_id]
            > plan.resource_slots.get(resource_id, 0)
        for resource_id in generated_plan.resource_ids
    ):
        reasons.append("DESKTOP_RESOURCE_SLOT_INSUFFICIENT")
    if (
        opportunity.helper_operator_plan.baseline_executor_id
            != ticket_binding.executor_id
    ):
        reasons.append("HELPER_BASE_EXECUTOR_MISMATCH")
    return tuple(reasons)
