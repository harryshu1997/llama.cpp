"""PhoneResidencyMixin portfolio operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.phone_shards import PhoneFfnResidencyLayout
from ..._internal.background_placement import material_count_bucket
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _RECOVERABLE_ERRORS
from . import reprovision as _reprovision
from .common import (
    _OfflineLearningDemand,
    _PhoneDemandDiscovery,
    _PhoneLayoutDecision,
    _PhoneMemoryBudget,
    _PhoneQueueDemand,
)


def _record_preparing_phone_layout_evaluation(
    controller,
    request_id: str,
    observed_at_us: int,
    demand: _PhoneQueueDemand,
    preparing_layout: ModelPhoneResidencyLayout,
) -> None:
    controller._model_placement_controller.record_phone_layout_evaluation(
        observed_at_us,
        {
            "active_request_count_by_artifact": dict(sorted(
                demand.active_count_by_artifact.items()
            )),
            "active_remaining_decode_tokens_by_artifact": dict(sorted(
                demand.active_remaining_tokens_by_artifact.items()
            )),
            "candidates": (),
            "current_geometry_sha256": (
                preparing_layout.layout.geometry_sha256
            ),
            "observed_at_us": observed_at_us,
            "phone_layout_generation": preparing_layout.generation,
            "pressure_kind": "arrived_remaining_decode_tokens",
            "queue_work_by_artifact": dict(sorted(
                demand.queued_work_by_artifact.items()
            )),
            "queue_work_unit": "decode_token",
            "queued_output_tokens_by_artifact": dict(sorted(
                demand.queued_output_tokens_by_artifact.items()
            )),
            "queued_request_count_by_artifact": dict(sorted(
                demand.queued_count_by_artifact.items()
            )),
            "reason": "PHONE_RESIDENCY_TRANSITION_IN_PROGRESS",
            "request_id": request_id,
            "selected_geometry_sha256": (
                preparing_layout.layout.geometry_sha256
            ),
            "switching_margin_uj": 0,
            "transition_ticket_id": (
                preparing_layout.transition_ticket_id
            ),
        },
    )


def _record_phone_residency_demand_unavailable(
    controller,
    request_id: str,
    observed_at_us: int,
    demand: _PhoneQueueDemand,
    route_evidence: Mapping[str, Mapping[str, object]],
) -> None:
    current_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    current_geometry = (
        None
        if current_state is None
        else current_state.layout.geometry_sha256
    )
    event = MappingProxyType({
        "candidates": (),
        "current_geometry_sha256": current_geometry,
        "observed_at_us": observed_at_us,
        "active_request_count_by_artifact": dict(sorted(
            demand.active_count_by_artifact.items()
        )),
        "active_remaining_decode_tokens_by_artifact": dict(sorted(
            demand.active_remaining_tokens_by_artifact.items()
        )),
        "queued_request_count_by_artifact": dict(sorted(
            demand.queued_count_by_artifact.items()
        )),
        "queued_output_tokens_by_artifact": dict(sorted(
            demand.queued_output_tokens_by_artifact.items()
        )),
        "pressure_kind": "arrived_remaining_decode_tokens",
        "queue_work_by_artifact": dict(sorted(
            demand.queued_work_by_artifact.items()
        )),
        "queue_work_unit": "decode_token",
        "reason": "PHONE_RESIDENCY_DEMAND_UNAVAILABLE",
        "request_id": request_id,
        "route_evidence_by_artifact": {
            key: dict(value)
            for key, value in sorted(route_evidence.items())
        },
        "selected_geometry_sha256": current_geometry,
        "switching_margin_uj": 0,
    })
    controller._model_placement_controller.record_phone_layout_evaluation(
        observed_at_us, event
    )


def _record_phone_shared_resource_mismatch(
    controller,
    request_id: str,
    observed_at_us: int,
    demand: _PhoneQueueDemand,
    shared_domains: set[tuple[str, tuple[str, ...]]],
) -> None:
    current_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    controller._model_placement_controller.record_phone_layout_evaluation(
        observed_at_us,
        {
            "candidates": (),
            "current_geometry_sha256": (
                None
                if current_state is None
                else current_state.layout.geometry_sha256
            ),
            "observed_at_us": observed_at_us,
            "active_remaining_decode_tokens_by_artifact": dict(sorted(
                demand.active_remaining_tokens_by_artifact.items()
            )),
            "queued_output_tokens_by_artifact": dict(sorted(
                demand.queued_output_tokens_by_artifact.items()
            )),
            "queue_work_by_artifact": dict(sorted(
                demand.queued_work_by_artifact.items()
            )),
            "queue_work_unit": "decode_token",
            "reason": "PHONE_RESIDENCY_SHARED_RESOURCE_MISMATCH",
            "request_id": request_id,
            "selected_geometry_sha256": None,
            "shared_resource_domains": [
                {
                    "compute_resource_id": compute,
                    "transport_resource_ids": list(transports),
                }
                for compute, transports in sorted(shared_domains)
            ],
            "switching_margin_uj": 0,
        },
    )


def _publish_phone_layout_view(
    controller,
    selected: PhoneFfnResidencyLayout | None,
    changed: bool,
    queued_work_by_artifact: Mapping[str, int],
) -> None:
    for compiler in (
        controller._automated_route_compiler,
        controller._runtime_epoch_route_compiler,
    ):
        if compiler is not None:
            compiler.set_phone_residency_layout(selected)
    if changed:
        controller._background_frontier_keys = {
            key: value
            for key, value in controller._background_frontier_keys.items()
            if key[0] not in queued_work_by_artifact
        }


def _record_phone_layout_selection(
    controller,
    request_id: str,
    observed_at_us: int,
    demand: _PhoneQueueDemand,
    discovery: _PhoneDemandDiscovery,
    memory: _PhoneMemoryBudget,
    decision: _PhoneLayoutDecision,
) -> None:
    selected = decision.selected
    selected_state = decision.selected_state
    evaluated = decision.evaluated_selected
    live_capacity = memory.live_capacity
    event = MappingProxyType({
        **memory.cap_evidence(),
        **({"request_impacts_by_geometry": {
            geometry: [row.to_json() for row in rows]
            for geometry, rows in sorted(decision.request_impacts_by_geometry.items())
        }} if decision.request_impacts_by_geometry else {}),
        **({"desktop_reprovision": dict(decision.reprovision)}
           if decision.reprovision else {}),
        "candidates": tuple(row.to_json() for row in decision.layouts),
        "current_geometry_sha256": (
            None
            if decision.current is None
            else decision.current.geometry_sha256
        ),
        "planning_geometry_sha256": (
            None
            if decision.planning is None
            else decision.planning.geometry_sha256
        ),
        "observed_at_us": observed_at_us,
        "active_request_count_by_artifact": dict(sorted(
            demand.active_count_by_artifact.items()
        )),
        "active_remaining_decode_tokens_by_artifact": dict(sorted(
            demand.active_remaining_tokens_by_artifact.items()
        )),
        "queued_request_count_by_artifact": dict(sorted(
            demand.queued_count_by_artifact.items()
        )),
        "queued_output_tokens_by_artifact": dict(sorted(
            demand.queued_output_tokens_by_artifact.items()
        )),
        "pressure_kind": "arrived_remaining_decode_tokens",
        "phone_memory_accepted_resident_bytes": (
            memory.accepted_resident_bytes
        ),
        "phone_memory_live_available_bytes": (
            None if live_capacity is None else live_capacity.available_bytes
        ),
        "phone_memory_live_limit_bytes": memory.live_phone_wide_limit,
        "phone_memory_persistent_service_reserve_bytes": (
            memory.persistent_service_reserve_bytes
        ),
        "phone_memory_persistent_service_reserve_by_artifact": dict(
            memory.persistent_service_reserve_by_artifact
        ),
        "phone_memory_selected_limit_bytes": memory.phone_wide_limit,
        "queue_work_by_artifact": dict(sorted(
            demand.queued_work_by_artifact.items()
        )),
        "queue_work_unit": "decode_token",
        "reason": decision.reason,
        "request_id": request_id,
        "route_evidence_by_artifact": {
            key: dict(value) for key, value in sorted(
                discovery.route_evidence_by_artifact.items()
            )
        },
        "selected_geometry_sha256": (
            None if selected is None else selected.geometry_sha256
        ),
        "selected_objective_uj": (
            None if selected is None else selected.objective
        ),
        "selected_queue_benefit_uj": (
            None if selected is None else selected.queue_benefit
        ),
        "selected_transition_cost_uj": (
            None if selected is None else selected.transition_cost
        ),
        "selected_layout_generation": (
            None if selected_state is None else selected_state.generation
        ),
        "selected_layout_state": (
            None if selected_state is None else selected_state.state
        ),
        "evaluated_geometry_sha256": (
            None if evaluated is None else evaluated.geometry_sha256
        ),
        "evaluated_objective_uj": (
            None if evaluated is None else evaluated.objective
        ),
        "evaluated_queue_benefit_uj": (
            None if evaluated is None else evaluated.queue_benefit
        ),
        "evaluated_transition_cost_uj": (
            None if evaluated is None else evaluated.transition_cost
        ),
        "selection_confirmed": decision.selection_confirmed,
        "selection_snapshot_count": decision.selection_snapshot_count,
        "selection_snapshot_sha256": decision.selection_snapshot_sha256,
        "session_marginal_gains": tuple(
            row.to_json() for row in decision.session_marginal_gains
        ),
        "minimum_residency_us": decision.minimum_residency_us,
        "transition_latency_us_by_session": dict(sorted(
            decision.transition_latencies.items()
        )),
        "switching_margin_uj": decision.switching_margin_uj,
    })
    if decision.reprovision:
        event = _reprovision._coalesce_unchanged_decision(controller, event)
        if event is None:
            return
    controller._model_placement_controller.record_phone_layout_evaluation(
        observed_at_us, event
    )


def _update_phone_residency_portfolio(
    controller,
    request: Request,
    manifest: ModelManifest,
    observed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot | None = None,
) -> bool:
    """Publish the best arrived-work phone portfolio to route compilers."""

    catalog = controller._runtime_capabilities
    if catalog is None:
        return False
    if controller._fixed_phone_residency is not None:
        ready = controller._model_placement_controller.ready_phone_layout()
        if ready is not None and not controller._fixed_phone_layout_matches(ready.layout, partial=True):
            raise UnifiedScheduleError("fixed residency READY assignment differs")
        controller._publish_phone_layout_view(
            None if ready is None else ready.layout, False, {},
        )
        return False
    if controller._phone_telemetry_deferral(
        snapshot, observed_at_us, request.request_id
    ) is not None:
        return False
    if controller._model_placement_controller.phone_preload_inflight():
        return False
    offline_plan_id = getattr(
        controller, "_active_offline_phone_residency_plan_id", None
    )
    offline_plan = (
        None if offline_plan_id is None else
        getattr(controller, "_offline_phone_residency_plans", {}).get(
            offline_plan_id
        )
    )
    if (
        offline_plan is not None
        and offline_plan.state not in {"READY", "FAILED"}
    ):
        ready = controller._model_placement_controller.ready_phone_layout()
        controller._publish_phone_layout_view(
            None if ready is None else ready.layout,
            False,
            {},
        )
        return False
    demand = controller._phone_queue_demand(request, manifest)
    preparing = (
        controller._model_placement_controller.preparing_phone_layout()
    )
    if preparing is None:
        target = controller._model_placement_controller.target_phone_layout()
        if target is not None and any(
            rebind is not None and rebind["target_generation"] == target.generation
            for ticket in controller._runtime_controller.current_tickets()
            for rebind in (controller._model_placement_controller.request_helper_rebind_state(
                ticket.request.request_id),)
        ):
            preparing = target
    if preparing is not None:
        controller._record_preparing_phone_layout_evaluation(
            request.request_id, observed_at_us, demand, preparing
        )
        return False
    discovery = controller._discover_phone_residency_demand(
        controller._automated_compiler(), demand.queued_work_by_artifact
    )
    if not discovery.demand_rows and snapshot is not None:
        discovery = controller._online_learning_phone_discovery(
            request,
            manifest,
            demand,
            snapshot,
            observed_at_us,
            discovery,
        )
    if not discovery.demand_rows:
        discovery = controller._cached_online_learning_phone_discovery(
            demand, discovery
        )
    cap_only = False
    if not discovery.demand_rows and controller._phone_htp_memory_caps:
        ready = controller._model_placement_controller.ready_phone_layout()
        helpers = tuple(row for row in catalog.executors
                        if row.device_id in controller._phone_htp_memory_caps
                        and row.phone_sessions and ready is not None
                        and {shard.session_id for shard in ready.layout.shards}.issubset(
                            session.session_id for session in row.phone_sessions))
        if len(helpers) == 1:
            helper = helpers[0]
            discovery = _PhoneDemandDiscovery(
                demand_rows=(), sessions=helper.phone_sessions,
                helper_id=helper.device_id,
                route_evidence_by_artifact=discovery.route_evidence_by_artifact,
            )
            cap_only = True
    if (
        not discovery.demand_rows and not cap_only
        or discovery.sessions is None
        or discovery.helper_id is None
    ):
        controller._record_phone_residency_demand_unavailable(
            request.request_id,
            observed_at_us,
            demand,
            discovery.route_evidence_by_artifact,
        )
        return False
    sessions = tuple(sorted(
        discovery.sessions, key=lambda row: row.session_id
    ))
    if controller._maximum_phone_sessions is not None:
        sessions = sessions[:controller._maximum_phone_sessions]
    shared_domains = {
        (row.shared_compute_resource_id,
         row.shared_transport_resource_ids)
        for row in sessions
    }
    if len(shared_domains) != 1:
        controller._record_phone_shared_resource_mismatch(
            request.request_id, observed_at_us, demand, shared_domains
        )
        return False
    helper = catalog.executor_by_device[discovery.helper_id]
    memory = controller._phone_memory_budget(
        discovery.helper_id, sessions, snapshot
    )
    layout_demand, discovery = controller._phone_reprovision_demand(
        demand, discovery, snapshot, observed_at_us
    )
    choice = controller._phone_candidate_choice(
        discovery, layout_demand, sessions, memory, observed_at_us
    )
    if choice.reason == "PHONE_RESIDENCY_MEMORY_CAP_REBALANCE":
        changed_artifacts = {row.artifact_sha256 for row in choice.selected.shards
                             if row.session_id in choice.selected.changed_session_ids}
        tickets = controller._runtime_controller.current_tickets()
        arriving_context = manifest.artifact_sha256 in changed_artifacts and not any(
            row.request.request_id == request.request_id for row in tickets
        )
        has_context = arriving_context or any(
            row.model.artifact_sha256 in changed_artifacts
            and row.dispatch_state not in {"CANCELLED", "COMPLETED", "FAILED"}
            and row.execution_plan is not None
            and row.execution_plan.desktop_placement_sha256 is not None
            for row in tickets
        )
        if not has_context:
            choice = replace(choice, selected=memory.current, force=False,
                             reason="PHONE_RESIDENCY_MEMORY_CAP_CONTEXT_UNAVAILABLE")
    selected = choice.selected
    evaluated_selected = selected
    snapshot_sha256 = controller._phone_selection_snapshot(
        demand, memory, sessions, selected, choice.marginal_gains
    )
    confirmed, snapshot_count = (
        controller._confirm_phone_layout_selection(
            selected,
            controller._model_placement_controller.target_phone_layout(),
            snapshot_sha256,
            observed_at_us,
            choice.force or choice.confirmed,
            snapshot,
        )
    )
    previous_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    controller._propose_confirmed_phone_layout(
        selected,
        confirmed,
        sessions,
        helper,
        demand,
        observed_at_us,
        choice.reason,
        choice.switching_margin_uj,
        choice.minimum_residency_us,
        choice.force,
    )
    selected_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    selected = (
        None if selected_state is None else selected_state.layout
    )
    changed = (
        (previous_state is None) != (selected_state is None)
        or (
            previous_state is not None
            and selected_state is not None
            and previous_state.generation != selected_state.generation
        )
    )
    controller._publish_phone_layout_view(
        selected, changed, demand.queued_work_by_artifact
    )
    controller._record_phone_layout_selection(
        request.request_id,
        observed_at_us,
        demand,
        discovery,
        memory,
        _PhoneLayoutDecision(
            layouts=choice.layouts,
            current=memory.current,
            planning=memory.planning,
            evaluated_selected=evaluated_selected,
            selected=selected,
            selected_state=selected_state,
            reason=choice.reason,
            session_marginal_gains=choice.marginal_gains,
            transition_latencies=choice.transition_latencies,
            switching_margin_uj=choice.switching_margin_uj,
            minimum_residency_us=choice.minimum_residency_us,
            selection_confirmed=confirmed,
            selection_snapshot_count=snapshot_count,
            selection_snapshot_sha256=snapshot_sha256,
            request_impacts_by_geometry=choice.request_impacts_by_geometry,
            reprovision=choice.reprovision,
        ),
    )
    return changed


def _reevaluate_pending_phone_layout_at_boundary(
    controller,
    ticket: RuntimeRequestTicket,
    observed_at_us: int,
) -> None:
    """Supply changed arrived work to phone-layout hysteresis."""

    try:
        pending = (
            controller._model_placement_controller
            .pending_phone_layout_candidate()
        )
        ready = (
            controller._model_placement_controller.ready_phone_layout()
        )
        target = (
            controller._model_placement_controller.target_phone_layout()
        )
    except ModelPlacementControllerError:
        return
    compiler = controller._automated_compiler()
    uncovered_artifacts = set()
    current_work_by_artifact: dict[str, int] = {}
    if ready is not None and target is None:
        for current in controller._runtime_controller.current_tickets():
            if current.dispatch_state not in {
                "ACQUIRED", "QUEUED", "REPLAN_REQUIRED", "REPLANNING",
            }:
                continue
            try:
                remaining = (
                    controller._model_placement_controller
                    .remaining_request_decode_tokens(
                        current.request.request_id,
                        current.request.output_tokens,
                    )
                    if current.dispatch_state == "ACQUIRED" else
                    current.request.output_tokens
                )
            except ModelPlacementControllerError:
                continue
            if remaining > 0:
                artifact_sha256 = current.model.artifact_sha256
                current_work_by_artifact[artifact_sha256] = (
                    current_work_by_artifact.get(
                        artifact_sha256, 0
                    ) + remaining
                )
            if (
                remaining > 0
                and not ready.covers_artifact(
                    current.model.artifact_sha256
                )
            ):
                evidence = compiler.phone_residency_evidence_status(
                    artifact_sha256
                )
                cached_learning = (
                    controller._online_learning_phone_demand_cache.get(
                        artifact_sha256
                    )
                )
                if (
                    (
                        evidence.get("reason")
                            == "PHONE_RESIDENCY_ROUTE_EVIDENCE_READY"
                        and type(evidence.get(
                            "normalized_benefit_uj"
                        )) is int
                        and evidence["normalized_benefit_uj"] > 0
                    )
                    or isinstance(
                        cached_learning, _OfflineLearningDemand
                    )
                ):
                    uncovered_artifacts.add(artifact_sha256)
    if pending is None and not uncovered_artifacts:
        return
    if pending is None:
        if not _reprovision._boundary_reevaluation_due(
            controller, compiler, current_work_by_artifact, uncovered_artifacts, observed_at_us
        ):
            return
        events = controller._model_placement_controller.phone_layout_events()
        previous = next((
            row for row in reversed(events)
            if row.get("kind") == "EVALUATED"
        ), None)
        if previous is not None:
            previous_work = previous.get("queue_work_by_artifact", {})
            if isinstance(previous_work, Mapping):
                current_buckets = {
                    artifact_sha256: material_count_bucket(work)
                    for artifact_sha256, work in sorted(
                        current_work_by_artifact.items()
                    )
                }
                previous_buckets = {
                    artifact_sha256: material_count_bucket(work)
                    for artifact_sha256, work in sorted(
                        previous_work.items()
                    )
                    if (
                        artifact_sha256 in current_work_by_artifact
                        and type(work) is int
                        and work >= 0
                    )
                }
                previous_evidence = previous.get(
                    "route_evidence_by_artifact", {}
                )
                if (
                    current_buckets == previous_buckets
                    and isinstance(previous_evidence, Mapping)
                    and all(
                        isinstance(
                            previous_evidence.get(artifact_sha256),
                            Mapping,
                        )
                        and previous_evidence[artifact_sha256].get(
                            "source_route_id"
                        ) == compiler.phone_residency_evidence_status(
                            artifact_sha256
                        ).get("source_route_id")
                        and previous_evidence[artifact_sha256].get(
                            "normalized_benefit_uj"
                        ) == compiler.phone_residency_evidence_status(
                            artifact_sha256
                        ).get("normalized_benefit_uj")
                        for artifact_sha256 in uncovered_artifacts
                    )
                ):
                    return
    try:
        with controller._transaction(convert=False):
            controller._update_phone_residency_portfolio(
                ticket.request,
                controller.runtime_model_manifest(ticket.model.model_id),
                observed_at_us,
                None,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            ticket.request.request_id,
            "BOUNDARY_LAYOUT_REEVALUATION_FAILED",
            observed_at_us,
            {
                "candidate_geometry_sha256": (
                    None if pending is None else
                    pending["geometry_sha256"]
                ),
                "ready_geometry_sha256": (
                    None if ready is None else
                    ready.layout.geometry_sha256
                ),
                "reason": str(exc),
                "uncovered_artifact_sha256s": sorted(
                    uncovered_artifacts
                ),
            },
        )
