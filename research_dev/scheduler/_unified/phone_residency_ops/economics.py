"""PhoneResidencyMixin economics operations on its existing owner."""

from __future__ import annotations

from typing import Mapping
from types import MappingProxyType

from ..._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimePhoneSessionCapability,
)
from ..._internal.model_placement_controller import (
    ModelPhoneResidencyLayout,
    PhoneSessionMarginalGain,
)
from ..._internal.phone_shards import (
    PhoneFfnResidencyLayout, generate_mixed_ffn_residency_layouts,
    progressive_ffn_residency_layouts,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.types import canonical_sha256
from ..common import _RECOVERABLE_ERRORS, _phone_shard_structure
from .common import (
    _PhoneCandidateChoice,
    _PhoneDemandDiscovery,
    _PhoneMemoryBudget,
    _PhoneQueueDemand,
)
from . import reprovision as _reprovision


def set_phone_htp_memory_cap(controller, phone_device_id, cap_bytes, *, workspace_bytes, observed_at_us):
    catalog = controller._runtime_capabilities
    helper = None if catalog is None else catalog.executor_by_device.get(phone_device_id)
    if helper is None or not helper.phone_sessions:
        raise UnifiedScheduleError("phone HTP memory cap device is absent")
    if (cap_bytes is not None and (type(cap_bytes) is not int or cap_bytes < 0)
            or type(workspace_bytes) is not int or workspace_bytes < 0
            or type(observed_at_us) is not int or observed_at_us < 0
            or cap_bytes is not None and workspace_bytes > cap_bytes):
        raise UnifiedScheduleError("phone HTP memory cap is invalid")
    if controller._fixed_phone_residency is not None:
        raise UnifiedScheduleError("fixed phone residency cannot change its memory cap")
    value = None if cap_bytes is None else (cap_bytes, workspace_bytes)
    previous = controller._phone_htp_memory_caps.get(phone_device_id)
    if previous == value:
        return
    with controller._transaction():
        if value is None:
            controller._phone_htp_memory_caps.pop(phone_device_id, None)
        else:
            controller._phone_htp_memory_caps[phone_device_id] = value
        controller._model_placement_controller.record_phone_layout_evaluation(observed_at_us, {
            "reason": "PHONE_HTP_MEMORY_CAP_CHANGED",
            "phone_device_id": phone_device_id,
            "previous_cap_bytes": None if previous is None else previous[0],
            "phone_memory_htp_cap_bytes": cap_bytes,
            "phone_memory_htp_workspace_bytes": workspace_bytes,
            "physical_memory_released_bytes": 0,
        })


def _phone_memory_budget(
    controller,
    helper_id: str,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
    snapshot: HeterogeneousRuntimeSnapshot | None,
) -> _PhoneMemoryBudget:
    catalog = controller._runtime_capabilities
    helper = catalog.executor_by_device[helper_id]
    device = catalog.placement_profile.devices[helper_id]
    pool = catalog.placement_profile.memory_pools[
        helper.memory_resource_id
    ]
    current_state = (
        controller._model_placement_controller.ready_phone_layout()
    )
    current = None if current_state is None else current_state.layout
    planning_state = (
        controller._model_placement_controller.planning_phone_layout()
    )
    planning = (
        None if planning_state is None else planning_state.layout
    )
    accepted = (
        0
        if current is None
        else sum(shard.resident_bytes for shard in current.shards)
    )
    live_capacity = (
        None
        if snapshot is None
        else snapshot.memory.capacities.get(helper.memory_resource_id)
    )
    pool_limit = max(0, min(device.allocation_limit_bytes, pool.capacity_bytes)
                     - max(pool.reserved_bytes, 0 if live_capacity is None else live_capacity.reserve_bytes))
    reserves = controller._persistent_phone_service_reserve_by_artifact(
        helper_id, snapshot
    )
    reserve_bytes = sum(reserves.values())
    live_limit = (
        None
        if live_capacity is None
        else max(
            0,
            live_capacity.available_bytes + accepted - reserve_bytes,
        )
    )
    phone_limit = min(
        sum(row.resident_memory_limit_bytes for row in sessions),
        device.allocation_limit_bytes,
        pool.capacity_bytes - pool.reserved_bytes,
        *((() if live_limit is None else (live_limit,))),
    )
    cap = controller._phone_htp_memory_caps.get(helper_id)
    peaks, workspace, error = {}, 0, None
    if cap is not None:
        configured_cap, workspace = cap
        resident_workspace = 0 if current_state is None else current_state.workspace_bytes
        workspace = max(workspace, resident_workspace)
        peaks = controller._persistent_phone_service_reserve_by_artifact(
            helper_id, snapshot, include_observed=True,
        )
        if peaks and "whole_model_peak_memory_bytes" not in helper.adapter_parameters:
            error = "PHONE_SERVICE_PEAK_MEMORY_UNKNOWN"
        static_limit = max(0, pool_limit - sum(peaks.values()))
        if live_limit is not None:
            # READY workspace is already included in live occupied memory.
            live_limit = max(0, live_limit - (workspace - resident_workspace))
        phone_limit = min(
            sum(row.resident_memory_limit_bytes for row in sessions),
            max(0, min(configured_cap, static_limit) - workspace),
            *((() if live_limit is None else (live_limit,))),
        )
        if error is not None:
            phone_limit = 0
    return _PhoneMemoryBudget(
        current=current,
        planning=planning,
        accepted_resident_bytes=accepted,
        live_capacity=live_capacity,
        persistent_service_reserve_by_artifact=reserves,
        persistent_service_reserve_bytes=reserve_bytes,
        live_phone_wide_limit=live_limit,
        phone_wide_limit=phone_limit,
        configured_htp_cap_bytes=None if cap is None else cap[0],
        htp_workspace_bytes=workspace,
        persistent_service_peak_by_artifact=peaks,
        reservation_error=error,
    )


def _phone_transition_estimates(
    controller,
    helper_id: str,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
    queued_work_by_artifact: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    costs = {}
    latencies = {}
    for session in sessions:
        estimates = tuple(
            transition.cost(session.resident_memory_limit_bytes)
            for transition in controller._runtime_capabilities.transitions
            if helper_id in transition.prepares_device_ids
            and transition.target_state in {"hot", "warm"}
            and transition.maturity == "QUALIFIED"
            and transition.energy_maturity == "QUALIFIED"
            and (
                transition.artifact_sha256 is None
                or transition.artifact_sha256
                    in queued_work_by_artifact
            )
        )
        costs[session.session_id] = (
            max(value[1] for value in estimates)
            if estimates else 10**18
        )
        latencies[session.session_id] = (
            max(value[0] for value in estimates)
            if estimates
            else controller._model_placement_controller.policy
                .phone_minimum_residency_us
        )
    return costs, latencies


def _phone_switching_constraints(
    controller,
    selected: PhoneFfnResidencyLayout | None,
    session_marginal_gains: tuple[PhoneSessionMarginalGain, ...],
) -> tuple[int, int]:
    policy = controller._model_placement_controller.policy
    margin = (
        policy.phone_session_hysteresis_uj
        + policy.phone_session_interference_energy_uj
        + policy.phone_session_safety_margin_uj
        + (
            0
            if selected is None
            else (
                selected.transition_cost
                * controller._runtime_capabilities.minimum_energy_saving_ppm
                + 999_999
            ) // 1_000_000
        )
    )
    minimum_residency = max(
        (row.minimum_residency_us for row in session_marginal_gains),
        default=policy.phone_minimum_residency_us,
    )
    return margin, minimum_residency


def _phone_selection_snapshot(
    controller,
    demand: _PhoneQueueDemand,
    memory: _PhoneMemoryBudget,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
    selected: PhoneFfnResidencyLayout | None,
    session_marginal_gains: tuple[PhoneSessionMarginalGain, ...],
) -> str:
    return canonical_sha256({
        **memory.cap_evidence(),
        "active_request_count_by_artifact": dict(sorted(
            demand.active_count_by_artifact.items()
        )),
        "active_remaining_decode_tokens_by_artifact": dict(sorted(
            demand.active_remaining_tokens_by_artifact.items()
        )),
        "current_geometry_sha256": (
            None
            if memory.current is None
            else memory.current.geometry_sha256
        ),
        "planning_geometry_sha256": (
            None
            if memory.planning is None
            else memory.planning.geometry_sha256
        ),
        "live_phone_available_bytes": (
            None
            if memory.live_capacity is None
            else memory.live_capacity.available_bytes
        ),
        "persistent_phone_service_reserve_by_artifact": dict(
            memory.persistent_service_reserve_by_artifact
        ),
        "queued_request_count_by_artifact": dict(sorted(
            demand.queued_count_by_artifact.items()
        )),
        "queued_output_tokens_by_artifact": dict(sorted(
            demand.queued_output_tokens_by_artifact.items()
        )),
        "ready_session_ids": sorted(
            row.session_id for row in sessions if row.ready
        ),
        "selected_geometry_sha256": (
            None if selected is None else selected.geometry_sha256
        ),
        "selected_objective": (
            None if selected is None else selected.objective
        ),
        "selected_transition_cost": (
            None if selected is None else selected.transition_cost
        ),
        "session_marginal_gains": [
            row.to_json() for row in session_marginal_gains
        ],
    })


def _phone_candidate_choice(
    controller,
    discovery: _PhoneDemandDiscovery,
    demand: _PhoneQueueDemand,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
    memory: _PhoneMemoryBudget,
    observed_at_us: int = 0,
) -> _PhoneCandidateChoice:
    if memory.phone_wide_limit <= 0:
        return _PhoneCandidateChoice(
            layouts=(),
            selected=memory.current,
            reason=memory.reservation_error or "NO_FEASIBLE_PHONE_RESIDENCY",
            marginal_gains=(),
            transition_latencies={},
            force=False,
            switching_margin_uj=0,
            minimum_residency_us=(
                controller._model_placement_controller.policy
                    .phone_minimum_residency_us
            ),
        )
    transition_costs, transition_latencies = (
        controller._phone_transition_estimates(
            discovery.helper_id,
            sessions,
            demand.queued_work_by_artifact,
        )
    )
    layouts = generate_mixed_ffn_residency_layouts(
        discovery.demand_rows,
        sessions,
        shard_storage=controller._phone_ffn_shard_storage,
        phone_wide_limit_bytes=memory.phone_wide_limit,
        current_shards=(
            () if memory.current is None else memory.current.shards
        ),
        transition_energy_uj_by_session=transition_costs,
        resident_manifests=tuple(controller._runtime_manifests.values()),
        allow_resident_shrink=memory.configured_htp_cap_bytes is not None,
    )
    if (memory.configured_htp_cap_bytes is not None and memory.current is not None
            and memory.current.resident_bytes > memory.phone_wide_limit):
        return _memory_cap_candidate_choice(
            controller, layouts, memory, transition_latencies, observed_at_us,
        )
    if demand.reprovision is not None:
        chosen = _reprovision._reprovision_candidate_choice(
            controller, demand.reprovision, layouts, sessions, memory,
            transition_latencies, observed_at_us,
        )
        if chosen is not None:
            return chosen
    force = bool(
        memory.current is not None
        and set(row.session_id for row in memory.current.shards)
            - set(row.session_id for row in sessions if row.ready)
    )
    request_impacts = {
        row.geometry_sha256: impacts for row in layouts
        if row.changed_session_ids
        if (impacts := controller._phone_layout_request_impacts(
            row, observed_at_us, sum(transition_latencies.get(s, 0) for s in row.changed_session_ids)
        ))
    }
    selected, reason, marginal_gains = (
        controller._model_placement_controller.select_phone_layout_candidate(
            layouts,
            current_layout=memory.current,
            minimum_energy_saving_ppm=(
                controller._runtime_capabilities.minimum_energy_saving_ppm
            ),
            transition_latency_us_by_session=transition_latencies,
            request_impacts_by_geometry=request_impacts,
            force=force,
        )
    )
    switching_margin, minimum_residency = (
        controller._phone_switching_constraints(selected, marginal_gains)
    )
    if selected is None and memory.current is not None:
        selected = memory.current
    return _PhoneCandidateChoice(
        layouts=layouts,
        selected=selected,
        reason=reason,
        marginal_gains=marginal_gains,
        transition_latencies=transition_latencies,
        request_impacts_by_geometry=request_impacts,
        force=force,
        switching_margin_uj=switching_margin,
        minimum_residency_us=minimum_residency,
    )


def _memory_cap_candidate_choice(controller, layouts, memory, transition_latencies, observed_at_us):
    source = {row.session_id: row for row in memory.current.shards}
    choices = []
    impacts_by_geometry = {}
    for target in layouts:
        if (target.resident_bytes > memory.phone_wide_limit
                or {row.session_id for row in target.shards} != set(source)
                or any(row.artifact_sha256 != source[row.session_id].artifact_sha256
                       or row.layer_mask & ~source[row.session_id].layer_mask
                       or row.maximum_columns != source[row.session_id].maximum_columns
                       or row.resident_bytes > source[row.session_id].resident_bytes
                       for row in target.shards)):
            continue
        stages = progressive_ffn_residency_layouts(target, current_shards=memory.current.shards)
        if not stages:
            continue
        stage = stages[0]
        impacts = controller._phone_layout_request_impacts(
            stage, observed_at_us,
            sum(transition_latencies.get(key, 0) for key in stage.changed_session_ids),
        )
        impacts_by_geometry[stage.geometry_sha256] = impacts
        cost = sum(row.incremental_cost_uj for row in impacts)
        score = target.objective + (cost if target.objective_kind == "queue_energy_delta_uj" else 0)
        choices.append(((score, len(target.changed_session_ids),
                         -target.resident_bytes, target.geometry_sha256), stage))
    selected = min(choices, key=lambda row: row[0])[1] if choices else memory.current
    return _PhoneCandidateChoice(
        layouts=layouts, selected=selected,
        reason="PHONE_RESIDENCY_MEMORY_CAP_REBALANCE" if choices else "PHONE_RESIDENCY_MEMORY_CAP_DEFERRED",
        marginal_gains=(), transition_latencies=transition_latencies,
        force=bool(choices), switching_margin_uj=0,
        minimum_residency_us=controller._model_placement_controller.policy.phone_minimum_residency_us,
        request_impacts_by_geometry=impacts_by_geometry,
    )


def _phone_layout_request_impacts(controller, layout, observed_at_us, transition_latency_us):
    current = controller._model_placement_controller.ready_phone_layout()
    if current is None or layout.geometry_sha256 == current.layout.geometry_sha256:
        return ()
    source = {row.session_id: row for row in current.layout.shards}
    target = {row.session_id: row for row in layout.shards}
    impacts = []
    for ticket in controller._runtime_controller.current_tickets():
        if ticket.dispatch_state != "ACQUIRED":
            continue
        context = controller._late_request_helper_contexts.get(ticket.request.request_id)
        helper = (context.helper if context is not None else
                  None if ticket.execution_plan is None else ticket.execution_plan.helper_envelope)
        if helper is None or controller._ready_request_helper(ticket, helper) is None:
            continue
        mask = 0
        for shard in helper.helper_plan.execution_contract.phone_shards:
            before, after = source.get(shard.session_id), target.get(shard.session_id)
            if (before is not None and after is not None
                    and _phone_shard_structure(shard) == _phone_shard_structure(before)
                    and _phone_shard_structure(before) == _phone_shard_structure(after)
                    and shard.session_generation == current.layout.session_generation_by_id.get(shard.session_id)
                    and shard.session_generation > 0
                    and shard.session_id not in layout.changed_session_ids):
                mask |= shard.layer_mask
        request_id = ticket.request.request_id
        remaining = controller._model_placement_controller.remaining_request_decode_tokens(
            request_id, ticket.request.output_tokens)
        impact = controller._adaptive_decode.preview_helper_replacement(
            request_id, retained_layer_mask=mask, token_index=ticket.request.output_tokens - remaining,
            at_us=observed_at_us, transition_latency_us=transition_latency_us)
        if impact is not None:
            impacts.append(impact)
    return tuple(sorted(impacts, key=lambda row: row.request_id))


def _defer_phone_layout_revalidation(controller, request_id, state, observed_at_us, transition_latency_us):
    """Reserve economics before draining; never cancel an issued maintenance ACK."""
    current = controller._model_placement_controller.ready_phone_layout()
    if current is None or state.selection_reason in {
        "PHONE_RESIDENCY_SESSION_DEGRADATION", "PHONE_RESIDENCY_MEMORY_CAP_REBALANCE",
    }:
        return None
    for ticket in controller._runtime_controller.current_tickets():
        rebind = controller._model_placement_controller.request_helper_rebind_state(ticket.request.request_id)
        if rebind is not None and rebind["target_generation"] == state.generation:
            return None
    impacts = controller._phone_layout_request_impacts(state.layout, observed_at_us, transition_latency_us)
    if not impacts:
        return None
    selected, reason, _ = controller._model_placement_controller.select_phone_layout_candidate(
        (current.layout, state.layout), current_layout=current.layout,
        minimum_energy_saving_ppm=controller._runtime_capabilities.minimum_energy_saving_ppm,
        request_impacts_by_geometry={state.layout.geometry_sha256: impacts})
    if selected is not None and selected.geometry_sha256 == state.layout.geometry_sha256:
        return None
    payload = {"phone_layout_generation": state.generation, "reason": reason,
               "affected_requests": [row.to_json() for row in impacts]}
    prior = controller._model_placement_controller.request_helper_events(request_id)
    if not any(row.get("kind") == "PREPARATION_DEFERRED" and row.get("reason") == reason
               and row.get("phone_layout_generation") == state.generation for row in prior):
        controller._model_placement_controller.record_request_helper_event(
            request_id, "PREPARATION_DEFERRED", observed_at_us, payload)
    return MappingProxyType({"status": "DEFERRED", **payload})


def _confirm_phone_layout_selection(
    controller,
    selected: PhoneFfnResidencyLayout | None,
    target_state: ModelPhoneResidencyLayout | None,
    selection_snapshot_sha256: str,
    observed_at_us: int,
    force: bool,
    snapshot: HeterogeneousRuntimeSnapshot | None = None,
) -> tuple[bool, int]:
    if selected is None:
        return True, 0
    if (
        target_state is not None
        and target_state.layout.geometry_sha256
            == selected.geometry_sha256
    ):
        return True, 0
    observation = controller._phone_layout_confirmation_observation(
        snapshot, observed_at_us
    )
    return (
        controller._model_placement_controller
        .confirm_phone_layout_candidate(
            selected.geometry_sha256,
            selection_snapshot_sha256,
            observed_at_us=observed_at_us,
            force=force,
            **({} if observation is None else {
                "observation_sha256": observation[0],
                "sampled_at_us": observation[1],
            }),
        )
    )


def _phone_layout_confirmation_observation(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot | None,
    observed_at_us: int,
) -> tuple[str, int] | None:
    if snapshot is None or not snapshot.telemetry_observations:
        return None
    samples = {}
    for capability in controller._runtime_capabilities.executors:
        if not capability.phone_sessions:
            continue
        row = snapshot.telemetry_observations.get(capability.device_id)
        if row is None or snapshot.telemetry_unavailable_reason(
            capability.device_id, observed_at_us
        ) is not None:
            return None
        samples[capability.device_id] = {
            "source": row["source"],
            "sample_timestamp_ns": row["sample_timestamp_ns"],
        }
    if not samples:
        return None
    return canonical_sha256(samples), min(
        row["sample_timestamp_ns"] // 1000 for row in samples.values()
    )


def _phone_cap_needs_reevaluation(controller, snapshot):
    owner = controller._model_placement_controller
    ready = owner.ready_phone_layout()
    if ready is None:
        return False
    budgets = tuple(
        controller._phone_memory_budget(row.device_id, row.phone_sessions, snapshot)
        for row in controller._runtime_capabilities.executors
        if row.device_id in controller._phone_htp_memory_caps and row.phone_sessions
    )
    if any(ready.layout.resident_bytes > row.phone_wide_limit for row in budgets):
        return True
    growth = tuple(row for row in budgets if ready.layout.resident_bytes < row.phone_wide_limit)
    if not growth:
        return False
    previous = next((row for row in reversed(owner.phone_layout_events())
                     if row.get("kind") == "EVALUATED"
                     and "phone_memory_selected_limit_bytes" in row), None)
    return (previous is None
            or previous.get("current_geometry_sha256") != ready.layout.geometry_sha256
            or any(row.phone_wide_limit > previous["phone_memory_selected_limit_bytes"] for row in growth))


def _reevaluate_pending_phone_layout_observation(
    _controller,
    ticket: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> None:
    controller = _controller._model_placement_controller
    pending = controller.pending_phone_layout_candidate()
    if (controller.target_phone_layout() is not None
            or pending is None and not _controller._phone_htp_memory_caps):
        return
    if _controller._phone_telemetry_deferral(
        snapshot, observed_at_us, ticket.request.request_id
    ) is not None:
        return
    if pending is None and not _phone_cap_needs_reevaluation(_controller, snapshot):
        return
    observation = _controller._phone_layout_confirmation_observation(
        snapshot, observed_at_us
    )
    if observation is None:
        return
    previous_sample = None if pending is None else pending.get("sampled_at_us")
    if previous_sample is not None and (
        observation[1] <= previous_sample
        or observation[1] - previous_sample < controller.policy.debounce_us
    ):
        return
    try:
        with _controller._transaction(convert=False):
            _controller._update_phone_residency_portfolio(
                ticket.request,
                _controller.runtime_model_manifest(ticket.model.model_id),
                observed_at_us,
                snapshot,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller.record_request_helper_event(
            ticket.request.request_id,
            "OBSERVATION_LAYOUT_REEVALUATION_FAILED",
            observed_at_us,
            {"reason": str(exc), "observation_sha256": observation[0]},
        )


def _propose_confirmed_phone_layout(
    controller,
    selected: PhoneFfnResidencyLayout | None,
    selection_confirmed: bool,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
    helper,
    demand: _PhoneQueueDemand,
    observed_at_us: int,
    reason: str,
    switching_margin_uj: int,
    minimum_residency_us: int,
    force: bool,
) -> None:
    if selected is None or not selection_confirmed:
        return
    controller._model_placement_controller.propose_phone_layout(
        selected,
        workspace_bytes=helper.workspace_bytes_per_token,
        shared_compute_resource_id=(
            sessions[0].shared_compute_resource_id
        ),
        shared_transport_resource_ids=(
            sessions[0].shared_transport_resource_ids
        ),
        observed_at_us=observed_at_us,
        selection_reason=reason,
        queue_work_by_artifact=demand.queued_work_by_artifact,
        queue_benefit_uj=selected.queue_benefit,
        transition_cost_uj=selected.transition_cost,
        switching_margin_uj=switching_margin_uj,
        minimum_residency_us=minimum_residency_us,
        force=force,
        progressive=bool(controller._phone_ffn_shard_storage),
    )
