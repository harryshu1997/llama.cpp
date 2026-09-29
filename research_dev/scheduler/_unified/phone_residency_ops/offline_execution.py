"""PhoneResidencyMixin offline execution operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.offline_phone_residency import (
    OfflinePhoneResidencyPlan,
    check_offline_transition_receipts,
    verify_offline_phone_layout,
)
from ..._internal.runtime_plan import RuntimeTransitionReceipt
from ..helper_preparation import (
    _helper_preparation_lease_demands,
    _helper_preparation_projection_sha256,
)


def begin_offline_phone_residency_stage(
    controller,
    plan_id: str,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> Mapping[str, object]:
    plan = controller._offline_phone_residency_plans.get(plan_id)
    stage = None if plan is None else plan.current_stage
    if plan is None or stage is None:
        raise UnifiedScheduleError(
            "offline phone residency stage is absent"
        )
    if stage.state == "LOADING":
        return MappingProxyType({
            **stage.to_json(), "status": "FOLLOWER",
        })
    if stage.state != "PROPOSED":
        return MappingProxyType({
            **stage.to_json(), "status": "NOT_REQUIRED",
        })
    snapshot.validate_at(observed_at_us)
    deferred = controller._phone_telemetry_deferral(
        snapshot, observed_at_us, stage.request_id
    )
    if deferred is not None:
        return deferred
    planning_snapshot, phone_safety_state = (
        controller._offline_phone_safety_snapshot(plan, snapshot)
    )
    helper = stage.helper_envelope
    transitions = helper.preparation_transitions
    duration_us = max(1, sum(row.latency_us for row in transitions))
    ready_at_us = observed_at_us + duration_us
    target = stage.layout
    blockers = (
        controller._model_placement_controller
        .phone_layout_transition_blockers(target.generation)
    )
    if blockers:
        return controller._defer_preparation_for_blockers(
            stage.request_id,
            target,
            blockers,
            observed_at_us,
        )
    yielding_resource_ids = (
        controller._copy_on_write_preparation_yielding_resources(
            controller._model_placement_controller.ready_phone_layout(),
            target,
        )
    )
    demands = _helper_preparation_lease_demands(
        transitions,
        preparation_ticket_id=stage.preparation_ticket_id,
        yielding_resource_ids=yielding_resource_ids,
        duration_us=duration_us,
    )
    memory_owner_id = (
        "offline-phone-preparation:" + stage.preparation_ticket_id
    )
    with controller._transaction(errors=Exception, convert=False):
        if demands:
            preview = controller.timeline.preview_leases(
                demands, observed_at_us, duration_us, duration_us
            )
            if preview.start_us != observed_at_us:
                return MappingProxyType({
                    "next_start_us": preview.start_us,
                    "status": "DEFERRED",
                })
            leases = controller.timeline.commit_leases(
                preview, memory_owner_id
            )
        else:
            leases = ()
        phone_device_id = (
            helper.helper_plan.execution_contract.phone_device_id
        )
        if phone_device_id is None:
            raise UnifiedScheduleError(
                "offline phone helper device is absent"
            )
        memory_demands = controller._reserve_helper_preparation_memory(
            helper,
            target,
            transitions,
            planning_snapshot,
            phone_device_id=phone_device_id,
            memory_owner_id=memory_owner_id,
            observed_at_us=observed_at_us,
        )
        projection_sha256 = _helper_preparation_projection_sha256(
            target,
            helper,
            memory_demands,
            preparation_ticket_id=stage.preparation_ticket_id,
            ready_at_us=ready_at_us,
            transition_ids=stage.transition_ids,
            yielding_resource_ids=yielding_resource_ids,
        )
        controller._prevalidate_target_layout_helper_envelopes(
            target,
            request_id=stage.request_id,
            observed_at_us=observed_at_us,
        )
        controller._model_placement_controller.begin_phone_layout_transition(
            target.generation,
            ticket_id=stage.preparation_ticket_id,
            transition_ids=stage.transition_ids,
            ready_at_us=ready_at_us,
            projection_token_sha256=projection_sha256,
            workspace_bytes=sum(
                row.required_bytes for row in memory_demands
                if row.kind == "workspace"
            ),
            observed_at_us=observed_at_us,
        )
        loading = replace(
            stage,
            state="LOADING",
            resource_lease_tokens=tuple(
                row.token for row in leases
            ),
            yielding_resource_ids=yielding_resource_ids,
            memory_owner_id=memory_owner_id,
            projection_token_sha256=projection_sha256,
            started_at_us=observed_at_us,
            ready_at_us=ready_at_us,
            phone_safety_state=(
                phone_safety_state
            ),
        )
        updated = replace(
            plan,
            stages=(*plan.stages[:-1], loading),
            state="LOADING",
        )
        controller._offline_phone_residency_plans[plan_id] = updated
        return MappingProxyType({
            **loading.to_json(), "status": "OWNER",
        })


def check_offline_phone_residency_stage(
    controller,
    plan_id: str,
    *,
    observed_at_us: int,
    guard_us: int = 250_000,
    quantum_us: int = 2_000_000,
) -> None:
    stage = controller.offline_phone_residency_stage(plan_id)
    if stage is None or stage.state != "LOADING":
        raise UnifiedScheduleError(
            "offline phone residency stage is not loading"
        )
    assert stage.ready_at_us is not None
    if observed_at_us + guard_us < stage.ready_at_us:
        return
    with controller._transaction():
        extended = max(
            stage.ready_at_us + quantum_us,
            observed_at_us + quantum_us,
        )
        for token in stage.resource_lease_tokens:
            controller.extend_lease(token, extended)
        plan = controller._offline_phone_residency_plans[plan_id]
        controller._offline_phone_residency_plans[plan_id] = replace(
            plan,
            stages=(
                *plan.stages[:-1],
                replace(stage, ready_at_us=extended),
            ),
        )


def complete_offline_phone_residency_stage(
    controller,
    plan_id: str,
    receipts: Sequence[RuntimeTransitionReceipt],
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> OfflinePhoneResidencyPlan:
    plan = controller._offline_phone_residency_plans.get(plan_id)
    stage = None if plan is None else plan.current_stage
    rows = tuple(receipts)
    if plan is None or stage is None:
        raise UnifiedScheduleError(
            "offline phone residency stage is absent"
        )
    try:
        rows = check_offline_transition_receipts(stage, rows)
    except ValueError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    finished_at_us = max(row.finished_us for row in rows)
    verified_at_us = max(finished_at_us, snapshot.captured_at_us)
    snapshot.validate_at(verified_at_us)
    phone_device_id = (
        stage.helper_envelope.helper_plan.execution_contract
            .phone_device_id
    )
    if phone_device_id is None:
        raise UnifiedScheduleError(
            "offline phone helper device is absent"
        )
    try:
        verification_sha256 = verify_offline_phone_layout(
            stage.layout,
            snapshot,
            phone_device_id=phone_device_id,
            executor_id=(
                stage.helper_envelope.helper_binding.executor_id
            ),
        )
    except ValueError as exc:
        raise UnifiedScheduleError(str(exc)) from exc
    if snapshot.telemetry_unavailable_reason(phone_device_id, verified_at_us):
        # Physical proof commits the completed load, not a new admission.
        verified_safety_state = stage.phone_safety_state
    else:
        verified_safety_state = controller._offline_phone_safety_snapshot(
            plan, snapshot
        )[1]
    assert stage.projection_token_sha256 is not None
    with controller._transaction(errors=Exception, convert=False):
        if stage.ready_at_us is not None and (
            finished_at_us > stage.ready_at_us
        ):
            for token in stage.resource_lease_tokens:
                controller.extend_lease(token, finished_at_us)
        ready = (
            controller._model_placement_controller
            .complete_phone_layout_transition(
                generation=stage.layout.generation,
                ticket_id=stage.preparation_ticket_id,
                transition_ids=stage.transition_ids,
                geometry_sha256=(
                    stage.layout.layout.geometry_sha256
                ),
                projection_token_sha256=(
                    stage.projection_token_sha256
                ),
                finished_at_us=finished_at_us,
            )
        )
        if ready is None:
            raise UnifiedScheduleError(
                "offline phone residency completion is stale"
            )
        controller._release_request_helper_preparation(stage, finished_at_us)
        completed = replace(
            stage,
            layout=ready,
            state="READY",
            ready_at_us=finished_at_us,
            verified_at_us=verified_at_us,
            transition_receipts=rows,
            verification_sha256=verification_sha256,
            phone_safety_state=verified_safety_state,
        )
        state = "READY" if not plan.pending_layouts else "PARTIAL"
        updated = replace(
            plan,
            target_layout=(
                ready.layout if state == "READY" else plan.target_layout
            ),
            stages=(*plan.stages[:-1], completed),
            state=state,
            finished_at_us=(
                verified_at_us if state == "READY" else None
            ),
        )
        controller._offline_phone_residency_plans[plan_id] = updated
        for compiler in (
            controller._automated_route_compiler,
            controller._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(ready.layout)
    controller._model_placement_controller.record_request_helper_event(
        stage.request_id,
        "OFFLINE_RESIDENCY_STAGE_READY",
        verified_at_us,
        {
            "phone_layout_generation": ready.generation,
            "phone_layout_geometry_sha256": (
                ready.layout.geometry_sha256
            ),
            "selected_session_id": stage.selected_session_id,
        },
    )
    rematerialized = controller._rematerialize_committed_layout_helpers(
        stage.request_id,
        ready,
        snapshot,
        verified_at_us,
        controller._preparation_phone_safety_state(
            phone_device_id, snapshot
        ),
    )
    controller._model_placement_controller.record_request_helper_event(
        stage.request_id,
        "OFFLINE_RESIDENCY_HELPERS_REFRESHED",
        verified_at_us,
        {
            "phone_layout_generation": ready.generation,
            "rematerialized_request_ids": list(rematerialized),
            "selected_session_id": stage.selected_session_id,
        },
    )
    return updated


def fail_offline_phone_residency_stage(
    controller,
    plan_id: str,
    *,
    failed_at_us: int,
    reason: str,
    unavailable_session_ids: Sequence[str] = (),
    restored_session_generations: Mapping[str, int] | None = None,
) -> OfflinePhoneResidencyPlan:
    plan = controller._offline_phone_residency_plans.get(plan_id)
    stage = None if plan is None else plan.current_stage
    if plan is None or stage is None or stage.state != "LOADING":
        raise UnifiedScheduleError(
            "offline phone residency stage is not loading"
        )
    assert stage.projection_token_sha256 is not None
    with controller._transaction(errors=Exception, convert=False):
        controller._cancel_request_helper_preparation(stage, failed_at_us)
        controller._model_placement_controller.fail_phone_layout_transition(
            stage.preparation_ticket_id,
            generation=stage.layout.generation,
            projection_token_sha256=stage.projection_token_sha256,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable_session_ids=unavailable_session_ids,
            restored_session_generations=(
                restored_session_generations
            ),
        )
        failed = replace(
            stage,
            state="FAILED",
            verified_at_us=failed_at_us,
            failure_reason=reason,
        )
        unavailable = tuple(sorted(set(
            (*plan.unavailable_session_ids, *unavailable_session_ids)
        )))
        has_ready = any(
            row.state == "READY" for row in plan.stages[:-1]
        )
        updated = replace(
            plan,
            stages=(*plan.stages[:-1], failed),
            pending_layouts=(),
            state="PARTIAL" if has_ready else "FAILED",
            unavailable_session_ids=unavailable,
            finished_at_us=failed_at_us,
        )
        controller._offline_phone_residency_plans[plan_id] = updated
        ready = controller._model_placement_controller.ready_phone_layout()
        for compiler in (
            controller._automated_route_compiler,
            controller._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(
                    None if ready is None else ready.layout
                )
        return updated


def adopt_offline_phone_residency(
    controller,
    plan: OfflinePhoneResidencyPlan,
    requests_by_model: Mapping[str, Sequence[Request]],
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> OfflinePhoneResidencyPlan:
    """Adopt an exact persisted superset after a desktop-only restart."""

    if (
        not isinstance(plan, OfflinePhoneResidencyPlan)
        or plan.state != "READY"
    ):
        raise UnifiedScheduleError(
            "offline phone adoption plan is not ready"
        )
    workload_sha256, _requests, _work = controller._offline_phone_requests(
        requests_by_model
    )
    if workload_sha256 != plan.workload_sha256:
        raise UnifiedScheduleError(
            "offline phone adoption workload differs"
        )
    snapshot.validate_at(observed_at_us)
    verified_at_us = max(observed_at_us, snapshot.captured_at_us)
    last_stage = plan.current_stage
    if last_stage is None or last_stage.state != "READY":
        raise UnifiedScheduleError(
            "offline phone adoption proof stage is absent"
        )
    with controller._transaction(convert=False):
        observed_rows = tuple(snapshot.phone_session_residency)
        executor_ids = {
            row.executor_id for row in observed_rows
            if row.session_id in plan.session_endpoints
        }
        device_ids = {
            row.device_id for row in observed_rows
            if row.session_id in plan.session_endpoints
        }
        if len(executor_ids) != 1 or len(device_ids) != 1:
            raise UnifiedScheduleError(
                "offline phone adoption endpoint is ambiguous"
            )
        try:
            verification_sha256 = verify_offline_phone_layout(
                last_stage.layout,
                snapshot,
                phone_device_id=next(iter(device_ids)),
                executor_id=next(iter(executor_ids)),
            )
        except ValueError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        ready = controller._model_placement_controller.adopt_verified_phone_layout(
            plan.target_layout,
            workspace_bytes=plan.workspace_bytes,
            shared_compute_resource_id=(
                plan.shared_compute_resource_id
            ),
            shared_transport_resource_ids=(
                plan.shared_transport_resource_ids
            ),
            verified_at_us=verified_at_us,
            verification_sha256=verification_sha256,
            selection_reason="OFFLINE_RESIDENCY_REUSE",
        )
        for compiler in (
            controller._automated_route_compiler,
            controller._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(ready.layout)
        adopted = replace(
            plan,
            target_layout=ready.layout,
            state="READY",
            adoption_started_at_us=observed_at_us,
            adoption_verified_at_us=verified_at_us,
            metadata={
                **dict(plan.metadata),
                "adoption_verification_sha256": verification_sha256,
            },
        )
        controller._offline_phone_residency_plans[plan.plan_id] = adopted
        controller._active_offline_phone_residency_plan_id = plan.plan_id
        return adopted


def offline_phone_residency_snapshot(
    controller, plan_id: str | None = None
) -> Mapping[str, object] | None:
    selected = (
        controller._active_offline_phone_residency_plan_id
        if plan_id is None else plan_id
    )
    plan = (
        None if selected is None else
        controller._offline_phone_residency_plans.get(selected)
    )
    if plan is None:
        return None
    return MappingProxyType({
        **plan.to_json(),
        "session_states": [
            row.to_json()
            for row in controller._model_placement_controller
                .phone_session_states()
        ],
    })
