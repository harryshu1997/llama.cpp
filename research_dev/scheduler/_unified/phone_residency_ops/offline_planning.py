"""PhoneResidencyMixin offline planning operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..._internal.policy import Request
from ..._internal.lifecycle import PhoneTelemetryUnavailable, UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.phone_shards import (
    generate_mixed_ffn_residency_layouts,
    progressive_ffn_residency_layouts,
)
from ..._internal.offline_phone_residency import (
    OfflinePhoneResidencyPlan,
    OfflinePhoneResidencyStage,
)
from ..._internal.types import canonical_sha256


def _propose_offline_phone_stage(
    controller,
    plan: OfflinePhoneResidencyPlan,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> OfflinePhoneResidencyPlan:
    if not plan.pending_layouts:
        return replace(
            plan,
            state="READY",
            finished_at_us=observed_at_us,
        )
    structural = plan.pending_layouts[0]
    proposed = controller._model_placement_controller.propose_phone_layout(
        structural,
        workspace_bytes=plan.workspace_bytes,
        shared_compute_resource_id=plan.shared_compute_resource_id,
        shared_transport_resource_ids=(
            plan.shared_transport_resource_ids
        ),
        observed_at_us=observed_at_us,
        selection_reason="OFFLINE_RESIDENT_SUPERSET",
        queue_work_by_artifact=(
            structural.queued_work_by_artifact
        ),
        queue_benefit_uj=structural.queue_benefit,
        transition_cost_uj=structural.transition_cost,
        switching_margin_uj=0,
        minimum_residency_us=0,
        force=True,
    )
    if proposed.state != "PROPOSED":
        raise UnifiedScheduleError(
            "offline phone stage was not proposed"
        )
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(proposed.layout)
    selected_session_id = proposed.layout.changed_session_ids[0]
    selected_shard = next((
        row for row in proposed.layout.shards
        if row.session_id == selected_session_id
    ), None)
    if selected_shard is None:
        raise UnifiedScheduleError(
            "offline phone stage cannot remove a resident session"
        )
    try:
        model_id, representative = plan.request_by_artifact[
            selected_shard.artifact_sha256
        ]
    except KeyError as exc:
        raise UnifiedScheduleError(
            "offline phone stage artifact has no demand owner"
        ) from exc
    stage_index = len(plan.stages)
    stage_id = (
        "offline-phone-"
        + plan.plan_id[7:19]
        + "-"
        + str(stage_index)
        + "-"
        + selected_session_id
    )
    request = controller._offline_planning_request(
        representative,
        request_id=stage_id,
        observed_at_us=observed_at_us,
    )
    planning_snapshot, phone_safety_state = (
        controller._offline_phone_materialization_snapshot(plan, snapshot)
    )
    helper = controller._materialize_offline_phone_helper(
        request,
        model_id,
        proposed,
        planning_snapshot,
        observed_at_us,
    )
    preparation_ticket_id = helper.preparation_ticket_id(stage_id)
    stage = OfflinePhoneResidencyStage(
        plan_id=plan.plan_id,
        stage_id=stage_id,
        stage_index=stage_index,
        model_id=model_id,
        request=request,
        layout=proposed,
        helper_envelope=helper,
        preparation_ticket_id=preparation_ticket_id,
        transition_ids=tuple(
            row.transition_id
            for row in helper.preparation_transitions
        ),
        phone_safety_state=phone_safety_state,
    )
    ready = controller._model_placement_controller.ready_phone_layout()
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(
                None if ready is None else ready.layout
            )
    updated = replace(
        plan,
        pending_layouts=plan.pending_layouts[1:],
        stages=(*plan.stages, stage),
        state=(
            "PARTIAL"
            if any(row.state == "READY" for row in plan.stages)
            else "PLANNED"
        ),
    )
    controller._offline_phone_residency_plans[plan.plan_id] = updated
    return updated


def plan_offline_phone_residency(
    controller,
    requests_by_model: Mapping[str, Sequence[Request]],
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> OfflinePhoneResidencyPlan:
    """Choose and queue a progressive resident superset before a trace."""

    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError(
            "offline phone planning snapshot is invalid"
        )
    snapshot.validate_at(observed_at_us)
    deferred = controller._phone_telemetry_deferral(
        snapshot, observed_at_us, "offline-phone-residency"
    )
    if deferred is not None:
        raise PhoneTelemetryUnavailable(deferred["detail"])
    if controller._fixed_phone_residency is None and any(
        row.dispatch_state not in {"CANCELLED", "COMPLETED", "FAILED"}
        for row in controller._runtime_controller.current_tickets()
    ):
        raise UnifiedScheduleError(
            "offline phone planning requires an idle request queue"
        )
    if controller._active_offline_phone_residency_plan_id is not None:
        current = controller._offline_phone_residency_plans[
            controller._active_offline_phone_residency_plan_id
        ]
        if current.state not in {"READY", "FAILED"}:
            raise UnifiedScheduleError(
                "offline phone residency plan is already active"
            )
    (
        workload_sha256,
        request_by_artifact,
        queued_work_by_artifact,
    ) = controller._offline_phone_requests(requests_by_model)
    with controller._transaction(convert=False):
        discovery_snapshot = controller._offline_phone_discovery_snapshot(
            snapshot
        )
        discovery = controller._offline_phone_discovery(
            request_by_artifact,
            queued_work_by_artifact,
            discovery_snapshot,
            observed_at_us,
        )
        target, sessions, memory = controller._offline_phone_target(
            discovery, queued_work_by_artifact, snapshot
        )
        current = controller._model_placement_controller.ready_phone_layout()
        stages = progressive_ffn_residency_layouts(
            target,
            current_shards=(
                () if current is None else current.layout.shards
            ),
        )
        if not stages:
            raise UnifiedScheduleError(
                "offline phone target is already resident"
            )
        helper = controller._runtime_capabilities.executor_by_device[
            discovery.helper_id
        ]
        plan_id = canonical_sha256({
            "phone_wide_limit_bytes": memory.phone_wide_limit,
            "source_layout_geometry_sha256": (
                None if current is None else
                current.layout.geometry_sha256
            ),
            "target_layout": target.to_json(),
            "workload_sha256": workload_sha256,
        })
        plan = OfflinePhoneResidencyPlan(
            plan_id=plan_id,
            workload_sha256=workload_sha256,
            target_layout=target,
            pending_layouts=stages,
            request_by_artifact=request_by_artifact,
            session_endpoints={
                row.session_id: row.endpoint for row in sessions
            },
            shared_compute_resource_id=(
                sessions[0].shared_compute_resource_id
            ),
            shared_transport_resource_ids=(
                sessions[0].shared_transport_resource_ids
            ),
            workspace_bytes=helper.workspace_bytes_per_token,
            phone_wide_limit_bytes=memory.phone_wide_limit,
            persistent_service_reserve_bytes=(
                memory.persistent_service_reserve_bytes
            ),
            created_at_us=observed_at_us,
            source_snapshot_id=snapshot.snapshot_id,
            materialization_snapshot=discovery_snapshot,
            metadata={
                **({} if controller._fixed_phone_residency is None else {
                    "fixed_residency": controller.fixed_phone_residency_configuration(),
                }),
                "candidate_count": 1 if controller._fixed_phone_residency is not None else len(
                    generate_mixed_ffn_residency_layouts(
                        discovery.demand_rows,
                        sessions,
                        shard_storage=controller._phone_ffn_shard_storage,
                        phone_wide_limit_bytes=memory.phone_wide_limit,
                        current_shards=(
                            () if current is None else
                            current.layout.shards
                        ),
                        transition_energy_uj_by_session=(
                            controller._phone_transition_estimates(
                                discovery.helper_id,
                                sessions,
                                queued_work_by_artifact,
                            )[0]
                        ),
                    )
                ),
                "selection_policy": (
                    "largest-amortized-resident-superset"
                    if controller._fixed_phone_residency is None else
                    "fixed-experimental-assignment"
                ),
                "demand_selection_rule": (
                    "measured-queue-energy-first;"
                        "learning-rough-compute-only-without-measured-demand"
                    if controller._fixed_phone_residency is None else
                    "frozen-artifact-session-layer-column-assignment;no-future-arrivals"
                ),
                "route_evidence_state_by_artifact": {
                    artifact: str(
                        status.get("evidence_state", "TRUSTED")
                    )
                    for artifact, status in sorted(
                        discovery.route_evidence_by_artifact.items()
                    )
                },
            },
        )
        controller._model_placement_controller.register_empty_phone_sessions(
            plan.session_endpoints,
            observed_at_us=observed_at_us,
        )
        controller._offline_phone_residency_plans[plan_id] = plan
        controller._active_offline_phone_residency_plan_id = plan_id
        return controller._propose_offline_phone_stage(
            plan, snapshot, observed_at_us
        )


def next_offline_phone_residency_stage(
    controller,
    plan_id: str,
    *,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> OfflinePhoneResidencyPlan:
    plan = controller._offline_phone_residency_plans.get(plan_id)
    if plan is None:
        raise UnifiedScheduleError(
            "offline phone residency plan is absent"
        )
    current = plan.current_stage
    if current is not None and current.state in {"PROPOSED", "LOADING"}:
        return plan
    if plan.state in {"READY", "FAILED"}:
        return plan
    snapshot.validate_at(observed_at_us)
    deferred = controller._phone_telemetry_deferral(
        snapshot, observed_at_us, "offline-phone-residency"
    )
    if deferred is not None:
        raise PhoneTelemetryUnavailable(deferred["detail"])
    with controller._transaction(convert=False):
        updated = controller._propose_offline_phone_stage(
            plan, snapshot, observed_at_us
        )
        controller._offline_phone_residency_plans[plan_id] = updated
        return updated


def offline_phone_residency_stage(
    controller, plan_id: str
) -> OfflinePhoneResidencyStage | None:
    plan = controller._offline_phone_residency_plans.get(plan_id)
    return None if plan is None else plan.current_stage
