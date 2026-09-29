"""ModelPlacementController economics operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ..phone_shards import PhoneFfnResidencyLayout
from ..model_placement_contracts.common import ModelPlacementControllerError, _integer, _sha256
from ..model_placement_contracts.layout import PhoneLayoutRequestImpact, PhoneSessionMarginalGain
from ..model_placement_contracts.requests import RequestPlacementBinding


def select_phone_layout_candidate(
    controller,
    layouts: Sequence[PhoneFfnResidencyLayout],
    *,
    current_layout: PhoneFfnResidencyLayout | None,
    minimum_energy_saving_ppm: int,
    transition_latency_us_by_session: Mapping[str, int] | None = None,
    request_impacts_by_geometry: Mapping[str, tuple[PhoneLayoutRequestImpact, ...]] | None = None,
    force: bool = False,
) -> tuple[
    PhoneFfnResidencyLayout | None,
    str,
    tuple[PhoneSessionMarginalGain, ...],
]:
    """Select a cold layout or one profitable session replacement."""

    rows = tuple(layouts)
    impacts = dict(request_impacts_by_geometry or {})
    for geometry, requests in impacts.items():
        _sha256("phone layout impact geometry", geometry)
        if (any(not isinstance(row, PhoneLayoutRequestImpact) for row in requests)
                or len({row.request_id for row in requests}) != len(requests)):
            raise ModelPlacementControllerError("phone layout request impacts are invalid")

    def impact_reason(layout):
        if force:
            return None
        affected = impacts.get(layout.geometry_sha256, ())
        if any(not row.verification_feasible for row in affected):
            return "PHONE_RESIDENCY_REVALIDATION_UNAFFORDABLE"
        if (layout.objective_kind == "queue_rough_compute_ops"
                and any(row.incremental_cost_uj for row in affected)):
            return "PHONE_RESIDENCY_REVALIDATION_ENERGY_UNKNOWN"
        return None
    if (
        not rows
        or any(
            not isinstance(row, PhoneFfnResidencyLayout)
            for row in rows
        )
        or type(force) is not bool
    ):
        if rows:
            raise ModelPlacementControllerError(
                "phone layout candidates are invalid"
            )
        return None, "NO_FEASIBLE_PHONE_RESIDENCY", ()
    saving_ppm = _integer(
        "phone layout minimum energy saving",
        minimum_energy_saving_ppm,
    )
    if saving_ppm > 1_000_000:
        raise ModelPlacementControllerError(
            "phone layout minimum energy saving exceeds one"
        )
    transition_latency_by_session = dict(
        transition_latency_us_by_session or {}
    )
    if any(
        type(session_id) is not str
        or not session_id
        or not session_id.isascii()
        or type(value) is not int
        or value < 0
        for session_id, value in (
            transition_latency_by_session.items()
        )
    ):
        raise ModelPlacementControllerError(
            "phone session transition latency is invalid"
        )
    energy_rows = tuple(
        row for row in rows
        if row.objective_kind == "queue_energy_delta_uj"
    )
    if not energy_rows:
        learning_rows = tuple(
            row for row in rows
            if row.objective_kind == "queue_rough_compute_ops"
        )
        if not learning_rows:
            return None, "PHONE_RESIDENCY_ENERGY_UNKNOWN", ()
        current_by_session = (
            {}
            if current_layout is None
            else {
                row.session_id: row for row in current_layout.shards
            }
        )

        def same_learning_shard(left: object, right: object) -> bool:
            return (
                left is not None
                and right is not None
                and left.artifact_sha256 == right.artifact_sha256
                and left.resident_geometry_sha256
                    == right.resident_geometry_sha256
                and left.operator_plan_sha256
                    == right.operator_plan_sha256
            )

        if current_layout is None:
            selected = min(learning_rows, key=lambda row: (
                row.objective,
                -len(row.shards),
                -row.resident_bytes,
                row.geometry_sha256,
            ))
            return (
                selected,
                "PHONE_RESIDENCY_LEARNING_EXPLORATION",
                (),
            )
        demanded_artifacts = {
            artifact_sha256
            for row in learning_rows
            for artifact_sha256, work in (
                row.queued_work_by_artifact.items()
            )
            if work > 0
        }
        current_artifacts = {
            row.artifact_sha256 for row in current_layout.shards
        }

        def preserves_current_shards(row) -> bool:
            # Additive coverage: every resident shard stays as it is and only
            # empty sessions are filled. Anything that evicts or reshapes a
            # resident shard is a replacement.
            proposed = {shard.session_id: shard for shard in row.shards}
            return bool(
                row.changed_session_ids
                and not set(row.changed_session_ids) & set(current_by_session)
                and len(row.shards) > len(current_by_session)
                and all(
                    same_learning_shard(proposed.get(session_id), shard)
                    for session_id, shard in current_by_session.items()
                )
            )

        expansion_only = bool(
            demanded_artifacts
            and demanded_artifacts.issubset(current_artifacts)
            and not force
        )
        if expansion_only:
            learning_rows = tuple(
                row for row in learning_rows if preserves_current_shards(row)
            )
            if not learning_rows:
                return (
                    current_layout,
                    "PHONE_RESIDENCY_LEARNING_RETAINED",
                    (),
                )
        current_benefit_by_session = {
            session_id: max(
                (
                    row.queue_benefit_by_session.get(session_id, 0)
                    for row in learning_rows
                    for candidate_shard in row.shards
                    if candidate_shard.session_id == session_id
                    and same_learning_shard(
                        candidate_shard, current_shard
                    )
                ),
                default=0,
            )
            for session_id, current_shard in current_by_session.items()
        }
        choices = []
        deferred_reasons = []
        for row in learning_rows:
            if (
                row.geometry_sha256 == current_layout.geometry_sha256
                or len(row.changed_session_ids) > 1 and not force
                or not row.changed_session_ids
            ):
                continue
            deferred = impact_reason(row)
            if deferred is not None:
                deferred_reasons.append(deferred)
                continue
            gain = sum(
                row.queue_benefit_by_session.get(session_id, 0)
                - current_benefit_by_session.get(session_id, 0)
                for session_id in row.changed_session_ids
            )
            choices.append((gain, row))
        if not choices:
            return (
                current_layout,
                min(deferred_reasons) if deferred_reasons else "PHONE_RESIDENCY_LEARNING_RETAINED",
                (),
            )
        gain, selected = max(choices, key=lambda item: (
            item[0],
            -item[1].transition_cost,
            item[1].geometry_sha256,
        ))
        if gain <= 0 and not force:
            return (
                current_layout,
                "PHONE_RESIDENCY_LEARNING_RETAINED",
                (),
            )
        return (
            selected,
            (
                "PHONE_RESIDENCY_LEARNING_EXPANSION"
                if expansion_only else "PHONE_RESIDENCY_LEARNING_EXPLORATION"
            ),
            (),
        )

    current_by_session = (
        {}
        if current_layout is None
        else {
            row.session_id: row for row in current_layout.shards
        }
    )

    def same_shard(left: object, right: object) -> bool:
        return (
            left is not None
            and right is not None
            and left.artifact_sha256 == right.artifact_sha256
            and left.resident_geometry_sha256
                == right.resident_geometry_sha256
            and left.operator_plan_sha256
                == right.operator_plan_sha256
        )

    current_benefit_by_session: dict[str, int] = {}
    for session_id, current_shard in current_by_session.items():
        current_benefit_by_session[session_id] = max(
            (
                row.queue_benefit_by_session.get(session_id, 0)
                for row in energy_rows
                for candidate_shard in row.shards
                if candidate_shard.session_id == session_id
                and same_shard(candidate_shard, current_shard)
            ),
            default=0,
        )

    def evidence_for(
        layout: PhoneFfnResidencyLayout,
    ) -> tuple[PhoneSessionMarginalGain, ...]:
        proposed_by_session = {
            row.session_id: row for row in layout.shards
        }
        result = []
        for session_id in layout.changed_session_ids:
            current_shard = current_by_session.get(session_id)
            proposed_shard = proposed_by_session.get(session_id)
            current_benefit = current_benefit_by_session.get(
                session_id, 0
            )
            proposed_benefit = layout.queue_benefit_by_session.get(
                session_id, 0
            )
            transition = layout.transition_cost_by_session.get(
                session_id, 0
            )
            load_energy = transition
            eviction_energy = 0
            replacement_energy = load_energy + eviction_energy
            latency_penalty = (
                controller.policy.phone_session_latency_penalty_uj
            )
            interference_energy = (
                controller.policy.phone_session_interference_energy_uj
            )
            safety_margin = (
                controller.policy.phone_session_safety_margin_uj
                + controller.policy.phone_session_hysteresis_uj
                + (transition * saving_ppm + 999_999) // 1_000_000
            )
            incremental_saving = max(
                0, proposed_benefit - current_benefit
            )
            remaining_work = (
                0
                if proposed_shard is None
                else layout.queued_work_by_artifact.get(
                    proposed_shard.artifact_sha256, 0
                )
            )
            amortization_cost = (
                load_energy
                + eviction_energy
                + latency_penalty
                + interference_energy
                + safety_margin
            )
            revalidation_cost = (sum(
                row.incremental_cost_uj for row in impacts.get(layout.geometry_sha256, ())
            ) if session_id == min(layout.changed_session_ids) else 0)
            amortization_cost += revalidation_cost
            break_even_work = (
                0
                if amortization_cost == 0
                else remaining_work + 1
                if incremental_saving == 0
                else (
                    amortization_cost * remaining_work
                    + incremental_saving - 1
                ) // incremental_saving
            )
            transition_latency_us = (
                transition_latency_by_session.get(session_id, 0)
            )
            break_even_interval_us = (
                transition_latency_us
                + (
                    transition_latency_us * break_even_work
                    + max(remaining_work, 1) - 1
                ) // max(remaining_work, 1)
            )
            minimum_residency_us = max(
                controller.policy.phone_minimum_residency_us,
                break_even_interval_us,
            )
            projected = (
                proposed_benefit
                - load_energy
                - eviction_energy
                - latency_penalty
                - interference_energy
                - safety_margin
                - revalidation_cost
            )
            result.append(PhoneSessionMarginalGain(
                session_id=session_id,
                current_artifact_sha256=(
                    None
                    if current_shard is None
                    else current_shard.artifact_sha256
                ),
                proposed_artifact_sha256=(
                    None
                    if proposed_shard is None
                    else proposed_shard.artifact_sha256
                ),
                current_geometry_sha256=(
                    None
                    if current_shard is None
                    else current_shard.resident_geometry_sha256
                ),
                proposed_geometry_sha256=(
                    None
                    if proposed_shard is None
                    else proposed_shard.resident_geometry_sha256
                ),
                remaining_work=remaining_work,
                current_warm_energy_saved_uj=current_benefit,
                proposed_warm_energy_saved_uj=proposed_benefit,
                session_load_energy_uj=load_energy,
                session_eviction_energy_uj=eviction_energy,
                session_replacement_energy_uj=replacement_energy,
                latency_penalty_uj=latency_penalty,
                interference_energy_uj=interference_energy,
                safety_margin_uj=safety_margin,
                transition_latency_us=transition_latency_us,
                break_even_work=break_even_work,
                break_even_interval_us=break_even_interval_us,
                minimum_residency_us=minimum_residency_us,
                projected_gain_uj=projected,
                gain_over_current_uj=projected - current_benefit,
                retained_revalidation_cost_uj=revalidation_cost,
            ))
        return tuple(result)

    if current_layout is None:
        ranked = tuple(sorted(energy_rows, key=lambda row: (
            row.objective
            + controller.policy.phone_session_latency_penalty_uj
                * len(row.changed_session_ids),
            -row.queue_benefit,
            row.transition_cost,
            row.geometry_sha256,
        )))
        selected = ranked[0]
        evidence = evidence_for(selected)
        gain = sum(row.projected_gain_uj for row in evidence)
        if gain <= 0:
            return None, "PHONE_RESIDENCY_SWITCH_MARGIN", evidence
        return (
            selected,
            "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN",
            evidence,
        )

    feasible = tuple(
        row for row in energy_rows
        if row.geometry_sha256 == current_layout.geometry_sha256
        or len(row.changed_session_ids) <= 1
        or force
    )
    choices = []
    deferred_reasons = []
    for row in feasible:
        if row.geometry_sha256 == current_layout.geometry_sha256:
            continue
        deferred = impact_reason(row)
        if deferred is not None:
            deferred_reasons.append(deferred)
            continue
        evidence = evidence_for(row)
        if not evidence:
            continue
        gain = sum(item.gain_over_current_uj for item in evidence)
        choices.append((gain, row, evidence))
    if not choices:
        return current_layout, min(deferred_reasons) if deferred_reasons else "PHONE_RESIDENCY_RETAINED", ()
    gain, selected, evidence = max(
        choices,
        key=lambda item: (
            item[0], -item[1].transition_cost,
            item[1].geometry_sha256,
        ),
    )
    if gain <= 0 and not force:
        return current_layout, "PHONE_RESIDENCY_SESSION_HYSTERESIS", evidence
    return (
        selected,
        (
            "PHONE_RESIDENCY_SESSION_DEGRADATION"
            if force else "PHONE_RESIDENCY_SESSION_MARGINAL_GAIN"
        ),
        evidence,
    )


def confirm_phone_layout_candidate(
    controller,
    geometry_sha256: str,
    snapshot_sha256: str,
    *,
    observed_at_us: int,
    force: bool = False,
    observation_sha256: str | None = None,
    sampled_at_us: int | None = None,
) -> tuple[bool, int]:
    """Require stable live pressure before replacing a ready layout."""

    geometry = _sha256(
        "phone layout candidate geometry", geometry_sha256
    )
    snapshot = _sha256(
        "phone layout candidate snapshot", snapshot_sha256
    )
    _integer("phone layout candidate time", observed_at_us)
    if observation_sha256 is not None:
        _sha256("phone layout observation", observation_sha256)
        _integer("phone layout observation time", sampled_at_us)
    elif sampled_at_us is not None:
        raise ModelPlacementControllerError(
            "phone layout observation identity is absent"
        )
    if type(force) is not bool:
        raise ModelPlacementControllerError(
            "phone layout candidate force flag is invalid"
        )
    ready = controller.ready_phone_layout()
    current_geometry = (
        None if ready is None else ready.layout.geometry_sha256
    )
    if force or current_geometry in {None, geometry}:
        controller._pending_phone_layout_geometry_sha256 = None
        controller._pending_phone_layout_snapshot_sha256 = None
        controller._pending_phone_layout_snapshot_count = 0
        controller._pending_phone_layout_sampled_at_us = None
        controller._note_helper_state_change()
        return True, controller.policy.phone_layout_confirmation_snapshots
    previous_sample = controller._pending_phone_layout_sampled_at_us
    if previous_sample is not None and (
        sampled_at_us is None
        or sampled_at_us < previous_sample
        or (
            controller._pending_phone_layout_geometry_sha256 == geometry
            and (sampled_at_us == previous_sample
                 or sampled_at_us - previous_sample < controller.policy.debounce_us)
        )
    ):
        return False, controller._pending_phone_layout_snapshot_count
    confirmation_identity = observation_sha256 or snapshot
    if controller._pending_phone_layout_geometry_sha256 != geometry:
        controller._pending_phone_layout_geometry_sha256 = geometry
        controller._pending_phone_layout_snapshot_sha256 = confirmation_identity
        controller._pending_phone_layout_snapshot_count = 1
    elif controller._pending_phone_layout_snapshot_sha256 != confirmation_identity:
        controller._pending_phone_layout_snapshot_sha256 = confirmation_identity
        controller._pending_phone_layout_snapshot_count += 1
    if observation_sha256 is not None:
        controller._pending_phone_layout_snapshot_count = min(
            controller._pending_phone_layout_snapshot_count,
            controller.policy.phone_layout_confirmation_snapshots,
        )
    controller._pending_phone_layout_sampled_at_us = sampled_at_us
    controller._note_helper_state_change()
    count = controller._pending_phone_layout_snapshot_count
    confirmed = count >= controller.policy.phone_layout_confirmation_snapshots
    controller._record_phone_layout_event(
        "SELECTION_CONFIRMED" if confirmed else "SELECTION_OBSERVED",
        observed_at_us,
        {
            "candidate_geometry_sha256": geometry,
            "consecutive_snapshot_count": count,
            "current_geometry_sha256": current_geometry,
            "required_snapshot_count": (
                controller.policy.phone_layout_confirmation_snapshots
            ),
            "snapshot_sha256": snapshot,
            **({} if observation_sha256 is None else {
                "observation_sha256": observation_sha256,
                "sampled_at_us": sampled_at_us,
            }),
        },
    )
    if confirmed and observation_sha256 is None:
        controller._pending_phone_layout_geometry_sha256 = None
        controller._pending_phone_layout_snapshot_sha256 = None
        controller._pending_phone_layout_snapshot_count = 0
        controller._pending_phone_layout_sampled_at_us = None
        controller._note_helper_state_change()
    return confirmed, count


def pending_phone_layout_candidate(
    controller,
) -> Mapping[str, object] | None:
    """Return the candidate waiting for another causal observation."""

    geometry = controller._pending_phone_layout_geometry_sha256
    snapshot = controller._pending_phone_layout_snapshot_sha256
    count = controller._pending_phone_layout_snapshot_count
    if geometry is None:
        if snapshot is not None or count != 0:
            raise ModelPlacementControllerError(
                "phone layout confirmation state is inconsistent"
            )
        return None
    if snapshot is None or count < 1:
        raise ModelPlacementControllerError(
            "phone layout confirmation state is incomplete"
        )
    return MappingProxyType({
        "geometry_sha256": geometry,
        "snapshot_count": count,
        "snapshot_sha256": snapshot,
        **({} if controller._pending_phone_layout_sampled_at_us is None else {
            "sampled_at_us": controller._pending_phone_layout_sampled_at_us,
        }),
    })


def phone_layout_transition_blockers(
    controller, target_generation: int
) -> tuple[str, ...]:
    """Return admitted requests bound to the layout being replaced."""

    target = controller.phone_layout(target_generation)
    if target.state not in {"PROPOSED", "PREPARING"}:
        return ()
    ready = controller.ready_phone_layout()
    if ready is None or ready.generation == target_generation:
        return ()

    def blocks(
        request_id: str, binding: RequestPlacementBinding
    ) -> bool:
        attachment = binding.helper_attachment
        if (
            request_id not in controller._acquired_request_ids
            or attachment is None
            or not (
                set(attachment.allowed_session_ids)
                & set(target.layout.changed_session_ids)
            )
            or (
                attachment.completed_phone_calls <= 0
                and attachment.fraction_ppm <= 0
                and not attachment.lease_tokens
            )
            or (
                attachment.fallback_outcome is not None
                and attachment.fraction_ppm == 0
                and binding.fraction_ppm == 0
                and not attachment.lease_tokens
                and attachment.lease_reserved_until_us is None
            )
        ):
            return False
        rebind = controller._request_helper_rebinds.get(request_id)
        return not (
            rebind is not None
            and rebind.state == "QUIESCED"
            and rebind.source_generation == ready.generation
            and rebind.target_generation == target_generation
            and (
                attachment.fraction_ppm == 0
                and binding.fraction_ppm == 0
                and not attachment.lease_tokens
                and attachment.lease_reserved_until_us is None
                or attachment.allowed_session_ids
                    == rebind.target_allowed_session_ids
                and not (
                    set(attachment.allowed_session_ids)
                    & set(rebind.removed_session_ids)
                )
            )
        )

    return tuple(sorted(
        request_id
        for request_id, binding in controller._request_bindings.items()
        if blocks(request_id, binding)
    ))
