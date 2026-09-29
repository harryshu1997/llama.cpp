"""PhoneResidencyMixin verification operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeExecutorBinding
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import RuntimeExecutionPlan
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.types import canonical_sha256
from ..common import _phone_shard_structure


def _observed_phone_layout_verification(
    controller,
    state: ModelPhoneResidencyLayout,
    ticket: RuntimeRequestTicket,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[int, str] | None:
    """Verify that a proposed layout is already physically resident."""

    plan = ticket.execution_plan
    if plan is None:
        return None
    return controller._phone_layout_snapshot_verification(
        state,
        model_id=ticket.model.model_id,
        artifact_sha256=ticket.model.artifact_sha256,
        plan=plan,
        binding=ticket.binding,
        base_executor_id=ticket.binding.executor_id,
        snapshot=snapshot,
    )


def _phone_layout_contract_identity(
    controller,
    state: ModelPhoneResidencyLayout,
    artifact_sha256: str,
    plan: RuntimeExecutionPlan,
    snapshot: HeterogeneousRuntimeSnapshot,
):
    contract = plan.execution_contract
    phone_device_id = contract.phone_device_id
    if phone_device_id is None or not contract.phone_shards:
        return None
    expected_shards = [
        _phone_shard_structure(row)
        for row in state.layout.shards
        if row.artifact_sha256 == artifact_sha256
    ]
    actual_shards = [
        _phone_shard_structure(row) for row in contract.phone_shards
    ]
    actual_generations = {
        row.session_id: row.session_generation
        for row in contract.phone_shards
    }
    layout_generations = getattr(
        state.layout,
        "session_generation_by_id",
        {
            row.session_id: getattr(row, "session_generation", 0)
            for row in state.layout.shards
        },
    )
    expected_generations = {
        row.session_id: layout_generations[row.session_id]
        for row in state.layout.shards
        if row.artifact_sha256 == artifact_sha256
    }
    generations_match = (
        actual_generations == expected_generations
        or (
            not getattr(snapshot, "phone_session_residency", ())
            and set(actual_generations.values()) == {0}
        )
    )
    if (
        not expected_shards
        or actual_shards != expected_shards
        or not generations_match
    ):
        return None
    return phone_device_id, actual_shards, layout_generations


def _phone_layout_execution_path_ready(
    plan: RuntimeExecutionPlan,
    base_executor_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    require_base_executor_ready: bool,
) -> bool:
    executor_state = snapshot.executors.get(base_executor_id)
    if (
        executor_state is None
        or not executor_state.healthy
        or (
            require_base_executor_ready
            and not executor_state.ready
        )
    ):
        return False
    used_link_ids = {
        resource_id.removeprefix("link:")
        for resource_id in plan.resource_ids
        if resource_id.startswith("link:")
    }
    return not any(
        link_id not in snapshot.links
        or not snapshot.links[link_id].ready
        for link_id in used_link_ids
    )


def _phone_layout_observed_residency(
    controller,
    state: ModelPhoneResidencyLayout,
    phone_device_id: str,
    binding: RuntimeExecutorBinding,
    snapshot: HeterogeneousRuntimeSnapshot,
):
    exact_rows = tuple(
        row for row in snapshot.residency
        if row.artifact_sha256 in state.covered_artifact_sha256s
        if row.device_id == phone_device_id
        and row.state in {"hot", "warm"}
        and row.executor_id == binding.executor_id
        and row.resident_geometry_sha256
            == state.layout.geometry_sha256
    )
    capability = controller._runtime_capabilities.executor_by_device.get(
        phone_device_id
    )
    session_by_id = (
        {} if capability is None else {
            row.session_id: row for row in capability.phone_sessions
        }
    )
    observed_sessions = tuple(
        row for row in getattr(
            snapshot, "phone_session_residency", ()
        )
        if row.device_id == phone_device_id
        and row.executor_id == binding.executor_id
    )
    return exact_rows, session_by_id, observed_sessions


def _phone_layout_residency_matches(
    state: ModelPhoneResidencyLayout,
    exact_rows,
    session_by_id,
    observed_sessions,
    layout_generations,
) -> bool:
    sessions_match = bool(state.layout.shards) and all(
        shard.session_id in session_by_id
        and session_by_id[shard.session_id].ready
        and session_by_id[shard.session_id].residency_state
            in {"hot", "warm"}
        and session_by_id[shard.session_id].endpoint
            == shard.endpoint
        and session_by_id[shard.session_id]
            .resident_artifact_sha256 == shard.artifact_sha256
        and session_by_id[shard.session_id]
            .resident_geometry_sha256
                == shard.resident_geometry_sha256
        for shard in state.layout.shards
    )
    observed_by_id = {
        row.session_id: row for row in observed_sessions
    }
    exact_sessions_match = (
        len(observed_by_id) == len(state.layout.shards)
        and all(
            (observed := observed_by_id.get(shard.session_id))
                is not None
            and observed.state == "READY"
            and observed.endpoint == shard.endpoint
            and observed.artifact_sha256 == shard.artifact_sha256
            and observed.resident_geometry_sha256
                == shard.resident_geometry_sha256
            and observed.operator_plan_sha256
                == shard.operator_plan_sha256
            and observed.session_generation
                == layout_generations[shard.session_id]
            and observed.resident_bytes == shard.resident_bytes
            for shard in state.layout.shards
        )
    )
    aggregate_matches = (
        bool(exact_rows)
        and sum(row.resident_bytes for row in exact_rows)
            >= state.layout.resident_bytes
        and {row.artifact_sha256 for row in exact_rows}
            == set(state.covered_artifact_sha256s)
    )
    return (
        exact_sessions_match
        if observed_sessions
        else aggregate_matches or sessions_match
    )


def _phone_layout_verification_proof(
    state: ModelPhoneResidencyLayout,
    *,
    model_id: str,
    artifact_sha256: str,
    plan: RuntimeExecutionPlan,
    binding: RuntimeExecutorBinding,
    base_executor_id: str,
    phone_device_id: str,
    actual_shards,
    exact_rows,
    session_by_id,
    observed_sessions,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[int, str]:
    workspace_bytes = sum(
        demand.required_bytes
        for demand in plan.memory_demands
        if demand.kind == "workspace"
        and demand.device_id == phone_device_id
    )
    verification_sha256 = canonical_sha256({
        "artifact_sha256": artifact_sha256,
        "base_executor_id": base_executor_id,
        "executor_id": binding.executor_id,
        "geometry_sha256": state.layout.geometry_sha256,
        "layout_generation": state.generation,
        "model_id": model_id,
        "phone_device_id": phone_device_id,
        "residency": [row.to_json() for row in exact_rows],
        "phone_session_residency": [
            row.to_json() for row in observed_sessions
        ],
        "schema": "runtime-phone-layout-verification-v1",
        "sessions": [
            session_by_id[row.session_id].to_json()
            for row in state.layout.shards
            if row.session_id in session_by_id
        ],
        "shards": actual_shards,
        "snapshot_captured_at_us": snapshot.captured_at_us,
        "snapshot_id": snapshot.snapshot_id,
        "workspace_bytes": workspace_bytes,
    })
    return workspace_bytes, verification_sha256


def _phone_layout_snapshot_verification(
    controller,
    state: ModelPhoneResidencyLayout,
    *,
    model_id: str,
    artifact_sha256: str,
    plan: RuntimeExecutionPlan,
    binding: RuntimeExecutorBinding,
    base_executor_id: str,
    snapshot: HeterogeneousRuntimeSnapshot,
    require_base_executor_ready: bool = True,
) -> tuple[int, str] | None:
    """Bind phone readiness to one exact physical snapshot."""

    if controller._runtime_capabilities is None:
        return None
    identity = controller._phone_layout_contract_identity(
        state, artifact_sha256, plan, snapshot
    )
    if identity is None or not controller._phone_layout_execution_path_ready(
        plan,
        base_executor_id,
        snapshot,
        require_base_executor_ready,
    ):
        return None
    phone_device_id, actual_shards, layout_generations = identity
    exact_rows, session_by_id, observed_sessions = (
        controller._phone_layout_observed_residency(
            state, phone_device_id, binding, snapshot
        )
    )
    if not controller._phone_layout_residency_matches(
        state,
        exact_rows,
        session_by_id,
        observed_sessions,
        layout_generations,
    ):
        return None
    return controller._phone_layout_verification_proof(
        state,
        model_id=model_id,
        artifact_sha256=artifact_sha256,
        plan=plan,
        binding=binding,
        base_executor_id=base_executor_id,
        phone_device_id=phone_device_id,
        actual_shards=actual_shards,
        exact_rows=exact_rows,
        session_by_id=session_by_id,
        observed_sessions=observed_sessions,
        snapshot=snapshot,
    )


def replay_observed_phone_layout(
    controller,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> Mapping[str, object]:
    """Publish a proposed layout proven by one replay snapshot."""

    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError(
            "replay phone layout snapshot is invalid"
        )
    snapshot.validate_at(snapshot.captured_at_us)
    target = controller._model_placement_controller.target_phone_layout()

    def result(
        status: str,
        layout: ModelPhoneResidencyLayout | None = None,
    ) -> Mapping[str, object]:
        values = {
            "phone_layout_events": (
                controller._model_placement_controller.phone_layout_events()
            ),
            "request_helper_events": (
                controller._model_placement_controller
                .request_helper_events()
            ),
            "status": status,
        }
        if layout is not None:
            values["layout"] = layout.to_json()
        return MappingProxyType(values)

    if target is None or target.state != "PROPOSED":
        return result("NOT_REQUIRED")

    helpers = []
    seen = set()
    for ticket in controller._runtime_controller.current_tickets():
        plan = ticket.execution_plan
        helper = None if plan is None else plan.helper_envelope
        candidates = (
            () if helper is None else (helper,)
        ) + tuple(
            value
            for key, value in sorted(
                controller._request_helper_preparation_envelopes.items()
            )
            if key[0] == ticket.request.request_id
            and key[1] == ticket.ticket_id
            and key[2] == target.generation
        )
        for candidate in candidates:
            identity = (
                ticket.ticket_id,
                candidate.operator_plan_sha256,
                candidate.phone_layout_generation,
            )
            if identity in seen:
                continue
            seen.add(identity)
            helpers.append((ticket, candidate))

    for ticket, helper in reversed(helpers):
        if helper.phone_layout_generation != target.generation:
            continue
        verification = controller._phone_layout_snapshot_verification(
            target,
            model_id=ticket.model.model_id,
            artifact_sha256=ticket.model.artifact_sha256,
            plan=helper.helper_plan,
            binding=helper.helper_binding,
            base_executor_id=ticket.binding.executor_id,
            snapshot=snapshot,
            require_base_executor_ready=False,
        )
        if verification is None:
            continue
        try:
            ready = (
                controller._model_placement_controller
                .verify_observed_phone_layout(
                    target.generation,
                    workspace_bytes=verification[0],
                    verified_at_us=snapshot.captured_at_us,
                    verification_sha256=verification[1],
                )
            )
        except ModelPlacementControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if ready is None:
            break
        for compiler in (
            controller._automated_route_compiler,
            controller._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(ready.layout)
        return result("READY", ready)
    return result("NOT_OBSERVED", target)
