"""Validation and memory calculations for helper preparation: checks."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from .._internal.policy import LeaseDemand
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_cost import RuntimeMemoryDemand
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.model_placement_controller import ModelPhoneResidencyLayout
from .._internal.runtime_plan import (
    RuntimeHelperExecutionEnvelope,
    RuntimeTransitionPlan,
    RuntimeTransitionReceipt,
)
from .._internal.types import canonical_sha256
from .common import _RequestHelperPreparation


def _partial_phone_replacement_session_diff(
    source: ModelPhoneResidencyLayout,
    target: ModelPhoneResidencyLayout,
) -> tuple[dict, tuple[str, ...]] | None:
    """Source shards by session and the changed ids of one exact replacement."""

    source_by_session = {
        row.session_id: row for row in source.layout.shards
    }
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    all_session_ids = set(source_by_session) | set(target_by_session)
    changed_session_ids = tuple(sorted(
        session_id for session_id in all_session_ids
        if source_by_session.get(session_id)
            != target_by_session.get(session_id)
    ))
    retained_session_ids = tuple(sorted(
        set(source_by_session) & set(target_by_session)
        - set(changed_session_ids)
    ))
    if (
        not changed_session_ids
        or not retained_session_ids
        or changed_session_ids
            != tuple(sorted(target.layout.changed_session_ids))
    ):
        return None
    return source_by_session, changed_session_ids


def _check_partial_phone_replacement_transitions(
    target: ModelPhoneResidencyLayout,
    transitions: Sequence[RuntimeTransitionPlan],
    *,
    phone_device_id: str,
    changed_session_ids: tuple[str, ...],
    source_by_session: Mapping[str, object],
) -> None:
    expected_shards = tuple(
        (
            row.session_id,
            row.artifact_sha256,
            row.endpoint,
            row.layer_mask,
            row.maximum_columns,
            row.resident_bytes,
            row.resident_geometry_sha256,
            row.operator_plan_sha256,
            target.layout.session_generation_by_id[row.session_id],
        )
        for row in target.layout.shards
    )
    phone_transitions = tuple(
        row for row in transitions
        if phone_device_id in row.prepares_device_ids
    )
    if not phone_transitions or any(
        tuple(sorted(row.changed_phone_session_ids))
            != changed_session_ids
        or tuple(
            (
                shard.session_id,
                shard.artifact_sha256,
                shard.endpoint,
                shard.layer_mask,
                shard.maximum_columns,
                shard.resident_bytes,
                shard.resident_geometry_sha256,
                shard.operator_plan_sha256,
                shard.session_generation,
            )
            for shard in row.phone_shards
        ) != expected_shards
        or tuple(sorted(
            eviction.session_id
            for eviction in row.evictions
            if eviction.session_id is not None
        )) != tuple(sorted(
            session_id for session_id in changed_session_ids
            if session_id in source_by_session
        ))
        for row in phone_transitions
    ):
        raise UnifiedScheduleError(
            "partial phone replacement transition differs from layout"
        )


def _check_partial_phone_replacement_source_exact(
    source: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    phone_device_id: str,
) -> None:
    session_observations = tuple(getattr(
        snapshot, "phone_session_residency", ()
    ))
    if session_observations:
        observed_by_session = {
            row.session_id: row
            for row in session_observations
            if row.device_id == phone_device_id
        }
        if len(observed_by_session) != len(session_observations):
            raise UnifiedScheduleError(
                "partial phone replacement session domain differs"
            )
        for shard in source.layout.shards:
            identity = source.layout.session_identity(
                shard.session_id
            )
            observed = observed_by_session.get(shard.session_id)
            if (
                observed is None
                or observed.state != "READY"
                or observed.artifact_sha256
                    != identity.artifact_sha256
                or observed.resident_geometry_sha256
                    != identity.resident_geometry_sha256
                or observed.operator_plan_sha256
                    != identity.operator_plan_sha256
                or observed.session_generation
                    != identity.session_generation
                or observed.resident_bytes != shard.resident_bytes
            ):
                raise UnifiedScheduleError(
                    "partial phone replacement source session is not physically exact"
                )
        return
    source_artifacts = {
        row.artifact_sha256 for row in source.layout.shards
    }
    observed_rows = tuple(
        row for row in snapshot.residency
        if row.device_id == phone_device_id
        and row.state in {"hot", "warm"}
        and row.generation == source.generation
        and row.resident_geometry_sha256
            == source.layout.geometry_sha256
    )
    if (
        {row.artifact_sha256 for row in observed_rows}
            != source_artifacts
        or sum(row.resident_bytes for row in observed_rows)
            != source.layout.resident_bytes
    ):
        raise UnifiedScheduleError(
            "partial phone replacement source is not physically exact"
        )


def _rewrite_partial_phone_replacement_demands(
    rows: tuple[RuntimeMemoryDemand, ...],
    *,
    phone_device_id: str,
    source_by_session: Mapping[str, object],
    target: ModelPhoneResidencyLayout,
    changed_session_ids: tuple[str, ...],
) -> tuple[RuntimeMemoryDemand, ...]:
    target_session_by_memory = {
        row.memory_resource_id: row
        for row in target.layout.shards
    }
    if len(target_session_by_memory) != len(target.layout.shards):
        raise UnifiedScheduleError(
            "partial phone replacement session memory is ambiguous"
        )
    result = []
    weight_demands = tuple(
        row for row in rows
        if row.device_id == phone_device_id
        and row.kind == "model_weights"
    )
    if (
        len(weight_demands) != 1
        or weight_demands[0].required_bytes
            != target.layout.resident_bytes
    ):
        raise UnifiedScheduleError(
            "partial phone replacement global memory differs"
        )
    for demand in rows:
        if demand.device_id != phone_device_id:
            result.append(demand)
            continue
        if demand.kind == "model_weights":
            resident_bytes = sum(
                min(
                    target_shard.resident_bytes,
                    (
                        0
                        if source_by_session.get(
                            target_shard.session_id
                        ) is None
                        else source_by_session[
                            target_shard.session_id
                        ].resident_bytes
                    ),
                )
                for target_shard in target.layout.shards
            )
            result.append(replace(
                demand,
                resident_bytes=resident_bytes,
                replacement_group=None,
                replaceable_bytes=0,
            ))
            continue
        if demand.kind == "session_residency_constraint":
            target_shard = target_session_by_memory.get(
                demand.resource_id
            )
            if target_shard is None:
                raise UnifiedScheduleError(
                    "partial phone replacement session memory differs"
                )
            source_shard = source_by_session.get(
                target_shard.session_id
            )
            result.append(replace(
                demand,
                resident_bytes=(
                    demand.required_bytes
                    if target_shard.session_id
                        not in changed_session_ids
                    else min(
                        demand.required_bytes,
                        0
                        if source_shard is None
                        else source_shard.resident_bytes,
                    )
                ),
                replacement_group=None,
                replaceable_bytes=0,
            ))
            continue
        if (
            demand.kind == "workspace"
            and demand.lifetime == "resident"
        ):
            result.append(replace(
                demand,
                resident_bytes=demand.required_bytes,
                replacement_group=None,
                replaceable_bytes=0,
            ))
            continue
        result.append(demand)
    return tuple(result)


def _helper_preparation_lease_demands(
    transitions: Sequence[RuntimeTransitionPlan],
    *,
    preparation_ticket_id: str,
    yielding_resource_ids: Sequence[str],
    duration_us: int,
) -> tuple[LeaseDemand, ...]:
    """Peak slots per non-yielding resource across the transitions."""

    yielding_resources = frozenset(yielding_resource_ids)
    slots_by_resource: dict[str, int] = {}
    for transition in transitions:
        for resource_id in transition.resource_ids:
            if resource_id in yielding_resources:
                continue
            slots_by_resource[resource_id] = max(
                slots_by_resource.get(resource_id, 0),
                transition.resource_slots[resource_id],
            )
    return tuple(
        LeaseDemand(
            lease_id=(
                "helper-preparation:"
                + preparation_ticket_id
                + ":"
                + resource_id
            ),
            resource_id=resource_id,
            slots=slots,
            start_offset_us=0,
            duration_us=duration_us,
            duration_upper_us=duration_us,
        )
        for resource_id, slots in sorted(
            slots_by_resource.items()
        )
    )


def _helper_preparation_projection_sha256(
    state: ModelPhoneResidencyLayout,
    helper: RuntimeHelperExecutionEnvelope,
    memory_demands: Sequence[RuntimeMemoryDemand],
    *,
    preparation_ticket_id: str,
    ready_at_us: int,
    transition_ids: Sequence[str],
    yielding_resource_ids: Sequence[str],
) -> str:
    return canonical_sha256({
        "memory_demands": [
            row.to_json() for row in memory_demands
        ],
        "phone_layout_generation": state.generation,
        "phone_layout_geometry_sha256": (
            state.layout.geometry_sha256
        ),
        "replacement_assignment_hash": (
            None
            if helper.replacement_authorization is None
            else helper.replacement_authorization.assignment_hash
        ),
        "preparation_ticket_id": preparation_ticket_id,
        "ready_at_us": ready_at_us,
        "schema": "request-helper-preparation-projection-v1",
        "transition_ids": list(transition_ids),
        "yielding_resource_ids": list(yielding_resource_ids),
    })


def _check_preparation_completion_receipts(
    preparation: _RequestHelperPreparation | None,
    rows: tuple[RuntimeTransitionReceipt, ...],
    *,
    request_id: str,
    preparation_ticket_id: str,
) -> None:
    if (
        preparation is None
        or preparation.request_id != request_id
        or preparation.state != "TRANSITIONING"
        or tuple(row.transition_id for row in rows)
            != preparation.transition_ids
        or any(
            row.ticket_id != preparation_ticket_id
            or row.request_id != request_id
            or row.operator_plan_sha256
                != preparation.operator_plan_sha256
            or row.status != "COMPLETED"
            for row in rows
        )
    ):
        raise UnifiedScheduleError(
            "request helper preparation receipt differs"
        )


def _rebind_blocker_quiescence(
    affected_binding: Mapping[str, object] | None,
    affected_rebind: Mapping[str, object],
) -> tuple[bool, bool, tuple[str, ...]]:
    """(active_masked, fully_quiesced, retained_session_ids) of one blocker."""

    affected_attachment = (
        None if affected_binding is None else
        affected_binding.get("helper_attachment")
    )
    retained_session_ids = tuple(
        affected_rebind.get("target_allowed_session_ids", ())
    )
    removed_session_ids = set(
        affected_rebind.get("removed_session_ids", ())
    )
    previous_fraction_ppm = int(
        affected_rebind.get("previous_fraction_ppm", -1)
    )
    allowed_session_ids = (
        () if not isinstance(affected_attachment, Mapping) else
        tuple(affected_attachment.get(
            "allowed_session_ids", ()
        ))
    )
    lease_tokens = (
        () if not isinstance(affected_attachment, Mapping) else
        tuple(affected_attachment.get("lease_tokens", ()))
    )
    active_masked = bool(
        isinstance(affected_attachment, Mapping)
        and retained_session_ids
        and allowed_session_ids == retained_session_ids
        and not set(allowed_session_ids) & removed_session_ids
        and int(affected_binding.get("fraction_ppm", -1))
            == previous_fraction_ppm
        and int(affected_attachment.get("fraction_ppm", -1))
            == previous_fraction_ppm
        and (
            previous_fraction_ppm == 0
            or bool(lease_tokens)
            and affected_attachment.get(
                "lease_reserved_until_us"
            ) is not None
        )
    )
    fully_quiesced = bool(
        isinstance(affected_attachment, Mapping)
        and int(affected_binding.get("fraction_ppm", -1)) == 0
        and int(affected_attachment.get("fraction_ppm", -1)) == 0
        and not lease_tokens
    )
    return active_masked, fully_quiesced, retained_session_ids
