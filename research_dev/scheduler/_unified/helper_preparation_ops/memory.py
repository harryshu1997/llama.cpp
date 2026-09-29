"""HelperPreparationMixin memory operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeMemoryDemand
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import RouteGenerationError
from ..._internal.model_placement_controller import ModelPhoneResidencyLayout
from ..._internal.runtime_plan import AutomatedCandidateSet, RuntimeTransitionPlan
from ..._internal.runtime_resources import RuntimeResourceError
from ..helper_preparation_checks import (
    _partial_phone_replacement_session_diff,
    _check_partial_phone_replacement_transitions,
    _check_partial_phone_replacement_source_exact,
    _rewrite_partial_phone_replacement_demands,
)


def _partial_phone_replacement_memory_demands(
    demands: Sequence[RuntimeMemoryDemand],
    *,
    phone_device_id: str,
    source: ModelPhoneResidencyLayout | None,
    target: ModelPhoneResidencyLayout,
    transitions: Sequence[RuntimeTransitionPlan],
    snapshot: HeterogeneousRuntimeSnapshot,
) -> tuple[RuntimeMemoryDemand, ...]:
    """Account one exact session replacement without double allocation."""

    rows = tuple(demands)
    if source is None or source.state != "READY":
        return rows
    diff = _partial_phone_replacement_session_diff(source, target)
    if diff is None:
        return rows
    source_by_session, changed_session_ids = diff
    _check_partial_phone_replacement_transitions(
        target,
        transitions,
        phone_device_id=phone_device_id,
        changed_session_ids=changed_session_ids,
        source_by_session=source_by_session,
    )
    _check_partial_phone_replacement_source_exact(
        source, snapshot, phone_device_id
    )
    return _rewrite_partial_phone_replacement_demands(
        rows,
        phone_device_id=phone_device_id,
        source_by_session=source_by_session,
        target=target,
        changed_session_ids=changed_session_ids,
    )


def _candidate_set_with_verified_partial_phone_memory(
    controller,
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    target: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    *,
    raise_capacity_error: bool = False,
) -> AutomatedCandidateSet:
    """Clear a coarse memory rejection after exact COW validation."""

    source = controller._model_placement_controller.ready_phone_layout()
    if (
        target.state != "PROPOSED"
        or len(target.layout.changed_session_ids) != 1
        or source is None
        or source.state != "READY"
        or source.generation == target.generation
    ):
        return candidate_set
    assert controller._runtime_capabilities is not None
    exclusive = controller._runtime_capabilities.exclusive_residency_resources
    changed = False
    rows = []
    for candidate in candidate_set.candidates:
        plan = candidate.plan
        if (
            "MEMORY_CAPACITY" not in candidate.rejection_reasons
            or candidate.assisted_operator_kind != "ffn"
            or plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            ) != target.layout.geometry_sha256
            or plan.execution_contract.phone_device_id is None
        ):
            rows.append(candidate)
            continue
        try:
            authorized_plan, authorized_binding = (
                controller._authorize_phone_helper_plan(
                    plan,
                    candidate.binding,
                    target,
                    model_id=manifest.model_id,
                    artifact_sha256=manifest.artifact_sha256,
                )
            )
            helper = controller._build_helper_envelope(
                artifact_sha256=manifest.artifact_sha256,
                desktop_parent_route_id=(
                    candidate.paired_baseline_route_id
                    or candidate_set.baseline_route_id
                ),
                desktop_placement_sha256=str(
                    plan.desktop_placement_sha256
                ),
                helper_plan=authorized_plan,
                helper_binding=authorized_binding,
                layout=target,
            )
            phone_device_id = (
                authorized_plan.execution_contract.phone_device_id
            )
            assert phone_device_id is not None
            transitions = helper.preparation_transitions
            demands = tuple(
                row for row in authorized_plan.memory_demands
                if row.device_id == phone_device_id
            )
            if not demands or not transitions:
                raise UnifiedScheduleError(
                    "partial phone helper preparation is incomplete"
                )
            adjusted = controller._partial_phone_replacement_memory_demands(
                demands,
                phone_device_id=phone_device_id,
                source=source,
                target=target,
                transitions=transitions,
                snapshot=snapshot,
            )
            controller._runtime_memory.preview(
                adjusted,
                snapshot.memory,
                start_us=observed_at_us,
                transitions=transitions,
                residency=snapshot.residency,
                exclusive_resource_by_device=exclusive,
            )
        except RuntimeResourceError as exc:
            if str(exc).startswith(
                "memory capacity is insufficient: "
            ):
                if raise_capacity_error:
                    checkpoint = controller._runtime_memory.checkpoint()
                    raise UnifiedScheduleError(
                        str(exc)
                        + "; capacities="
                        + repr({
                            row.resource_id: row.to_json()
                            for row in snapshot.memory.capacities.values()
                            if row.resource_id in {
                                demand.resource_id
                                for demand in adjusted
                            }
                        })
                        + "; demands="
                        + repr(tuple(
                            demand.to_json() for demand in adjusted
                        ))
                        + "; reservations="
                        + repr(tuple(
                            reservation.to_json()
                            for reservation in checkpoint
                                .reservations.values()
                        ))
                    ) from exc
                rows.append(candidate)
                continue
            raise UnifiedScheduleError(str(exc)) from exc
        except (RouteGenerationError, ValueError) as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        reasons = tuple(
            reason for reason in candidate.rejection_reasons
            if reason != "MEMORY_CAPACITY"
        )
        rows.append(replace(
            candidate,
            binding=replace(
                candidate.binding,
                eligibility_reasons=tuple(
                    reason
                    for reason in candidate.binding.eligibility_reasons
                    if reason != "MEMORY_CAPACITY"
                ),
            ),
            rejection_reasons=reasons,
        ))
        changed = True
    return (
        replace(candidate_set, candidates=tuple(rows))
        if changed else candidate_set
    )


def _copy_on_write_preparation_yielding_resources(
    source: ModelPhoneResidencyLayout | None,
    target: ModelPhoneResidencyLayout,
) -> tuple[str, ...]:
    """Keep long replacement preparation off shared inference leases."""

    if source is None or source.state not in {"READY", "DRAINING"}:
        return ()
    source_by_session = {
        row.session_id: row for row in source.layout.shards
    }
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    changed_session_ids = tuple(sorted(
        session_id
        for session_id in set(source_by_session) | set(target_by_session)
        if source_by_session.get(session_id)
            != target_by_session.get(session_id)
    ))
    retained_session_ids = (
        set(source_by_session) & set(target_by_session)
    ) - set(changed_session_ids)
    if (
        len(changed_session_ids) != 1
        or changed_session_ids
            != tuple(target.layout.changed_session_ids)
        or not retained_session_ids
        or any(
            source_by_session[session_id]
                != target_by_session[session_id]
            for session_id in retained_session_ids
        )
    ):
        return ()
    return tuple(sorted({
        target.shared_compute_resource_id,
        *target.shared_transport_resource_ids,
    }))
