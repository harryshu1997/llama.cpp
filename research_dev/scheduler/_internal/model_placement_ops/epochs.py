"""ModelPlacementController epochs operations on its existing owner."""

from __future__ import annotations

from ..model_placement_contracts.common import (
    ModelPlacementControllerError,
    _text,
    _integer,
    _sha256,
)
from ..model_placement_contracts.demand import (
    ModelDemandSnapshot,
    ModelPlacementTrigger,
    ModelPlacementAction,
)
from .common import _INFORMATIONAL_NOTIFICATIONS


def notify(
    controller, artifact_sha256: str, reason: str, observed_at_us: int
) -> None:
    artifact_sha256 = _sha256(
        "model placement notification artifact", artifact_sha256
    )
    reason = _text("model placement notification reason", reason)
    _integer("model placement notification time", observed_at_us)
    controller._pending_reasons.setdefault(artifact_sha256, set()).add(reason)


def _snapshot_reasons(
    previous: ModelDemandSnapshot | None,
    current: ModelDemandSnapshot,
) -> set[str]:
    if previous is None:
        return {"FIRST_MODEL_ARRIVAL"}
    reasons: set[str] = set()
    if previous.pressure_bucket != current.pressure_bucket:
        reasons.add("QUEUE_PRESSURE_BUCKET_CHANGED")
    if previous.available_device_ids != current.available_device_ids:
        reasons.add("DEVICE_AVAILABILITY_CHANGED")
    if previous.available_session_ids != current.available_session_ids:
        reasons.add("SESSION_AVAILABILITY_CHANGED")
    for name, reason in (
        ("memory_generation_sha256", "MEMORY_CAPACITY_CHANGED"),
        (
            "resource_calendar_generation_sha256",
            "RESOURCE_CALENDAR_CHANGED",
        ),
        (
            "current_resident_component_identity_sha256",
            "RESIDENT_COMPONENT_CHANGED",
        ),
        ("capability_generation_sha256", "CAPABILITY_CHANGED"),
        ("profile_generation_sha256", "PROFILE_CHANGED"),
        ("transport_generation_sha256", "TRANSPORT_CHANGED"),
        ("residency_generation_sha256", "RESIDENCY_CHANGED"),
        ("learning_generation_sha256", "LEARNING_GENERATION_CHANGED"),
    ):
        if getattr(previous, name) != getattr(current, name):
            reasons.add(reason)
    return reasons


def evaluate(
    controller, trigger: ModelPlacementTrigger
) -> ModelPlacementAction:
    if not isinstance(trigger, ModelPlacementTrigger):
        raise ModelPlacementControllerError(
            "model placement trigger is invalid"
        )
    snapshot = trigger.snapshot
    artifact = snapshot.artifact_sha256
    previous = controller._snapshots.get(artifact)
    reasons = controller._snapshot_reasons(previous, snapshot)
    reasons.update(trigger.notification_reasons)
    notifications = controller._pending_reasons.pop(artifact, set())
    reasons.update(notifications - _INFORMATIONAL_NOTIFICATIONS)
    if trigger.epoch_sha256 is None:
        reasons.add("MODEL_PLACEMENT_EPOCH_ABSENT")
    if (
        trigger.epoch_demand_generation_sha256 is not None
        and trigger.epoch_demand_generation_sha256
            != snapshot.demand_generation_sha256
    ):
        reasons.add("DEMAND_GENERATION_CHANGED")
    if (
        trigger.epoch_pressure_bucket is not None
        and trigger.epoch_pressure_bucket != snapshot.pressure_bucket
    ):
        reasons.add("QUEUE_PRESSURE_BUCKET_CHANGED")
    if (
        trigger.epoch_valid_until_us is not None
        and snapshot.observed_at_us >= trigger.epoch_valid_until_us
    ):
        reasons.add("MODEL_PLACEMENT_EPOCH_EXPIRED")
    if not trigger.selected_route_feasible:
        reasons.add("SELECTED_ROUTE_INFEASIBLE")
    if not reasons:
        reasons.add("NO_MATERIAL_CHANGE")

    old_component = (
        trigger.old_component_identity_sha256
        or snapshot.current_resident_component_identity_sha256
    )
    new_component = trigger.proposed_component_identity_sha256
    expected_reuse = trigger.expected_reuse_count
    switching = (
        new_component is not None
        and old_component is not None
        and new_component != old_component
    )

    urgent = bool({
        "CAPABILITY_CHANGED",
        "DEVICE_AVAILABILITY_CHANGED",
        "MODEL_PLACEMENT_EPOCH_ABSENT",
        "MODEL_PLACEMENT_EPOCH_EXPIRED",
        "PROFILE_CHANGED",
        "SELECTED_ROUTE_INFEASIBLE",
        "SESSION_AVAILABILITY_CHANGED",
        "TRANSPORT_CHANGED",
    }.intersection(reasons))
    material = reasons != {"NO_MATERIAL_CHANGE"}
    last_refresh = controller._last_refresh_us.get(artifact)
    within_debounce = (
        last_refresh is not None
        and snapshot.observed_at_us - last_refresh
            < controller.policy.debounce_us
    )
    within_hysteresis = (
        switching
        and last_refresh is not None
        and snapshot.observed_at_us - last_refresh
            < controller.policy.switch_hysteresis_us
    )
    coalesced = False
    if not trigger.selected_route_feasible and not snapshot.available_device_ids:
        kind = "FALLBACK"
    elif within_hysteresis:
        kind = "KEEP_EPOCH"
    elif urgent:
        kind = "RECOMPUTE_NOW"
    elif not material:
        kind = "KEEP_EPOCH"
    elif within_debounce or artifact in controller._background_inflight:
        controller._pending_reasons.setdefault(artifact, set()).update(reasons)
        kind = "KEEP_EPOCH"
        coalesced = True
    else:
        kind = "REFRESH_IN_BACKGROUND"
        controller._background_inflight.add(artifact)

    if kind in {"RECOMPUTE_NOW", "REFRESH_IN_BACKGROUND", "FALLBACK"}:
        controller._last_refresh_us[artifact] = snapshot.observed_at_us
    controller._snapshots[artifact] = snapshot
    action = ModelPlacementAction(
        kind=kind,
        trigger_reasons=tuple(sorted(reasons)),
        demand_generation_sha256=snapshot.demand_generation_sha256,
        pressure_bucket=snapshot.pressure_bucket,
        expected_reuse_count=expected_reuse,
        break_even_use_count=None,
        old_component_identity_sha256=old_component,
        new_component_identity_sha256=new_component,
        selected_transition_energy_uj=None,
        coalesced=coalesced,
    )
    controller._record_event(snapshot, action)
    return action


def authorize_publication(
    controller, trigger: ModelPlacementTrigger
) -> ModelPlacementAction:
    """Apply the sole current-versus-proposed epoch publication gate."""
    if not isinstance(trigger, ModelPlacementTrigger):
        raise ModelPlacementControllerError(
            "model placement publication trigger is invalid"
        )
    snapshot = trigger.snapshot
    old_component = (
        trigger.old_component_identity_sha256
        or snapshot.current_resident_component_identity_sha256
    )
    new_component = trigger.proposed_component_identity_sha256
    switching = (
        old_component is not None
        and new_component is not None
        and old_component != new_component
    )
    reasons = set(trigger.notification_reasons)
    latency_ok = True
    if trigger.proposed_latency_upper_us is not None:
        assert trigger.desktop_latency_upper_us is not None
        latency_ok = (
            trigger.proposed_latency_upper_us * 1_000_000
            <= trigger.desktop_latency_upper_us
                * trigger.maximum_latency_ppm
        )
        if not latency_ok:
            reasons.add("LATENCY_RATIO_EXCEEDED")
    energy_positive = trigger.warm_energy_positive
    break_even = trigger.break_even_use_count
    expected_reuse = trigger.expected_reuse_count
    if switching and trigger.selected_route_feasible:
        if energy_positive is None:
            reasons.add("PLACEMENT_WARM_ENERGY_UNKNOWN")
        elif not energy_positive:
            reasons.add("PROPOSED_WARM_ENERGY_NOT_LOWER")
        elif break_even is None:
            reasons.add("PLACEMENT_BREAK_EVEN_UNKNOWN")
        elif expected_reuse < break_even:
            reasons.add("TRANSITION_BREAK_EVEN_NOT_MET")
    rejected = bool({
        "LATENCY_RATIO_EXCEEDED",
        "PLACEMENT_WARM_ENERGY_UNKNOWN",
        "PROPOSED_WARM_ENERGY_NOT_LOWER",
        "PLACEMENT_BREAK_EVEN_UNKNOWN",
        "TRANSITION_BREAK_EVEN_NOT_MET",
    }.intersection(reasons))
    if not trigger.selected_route_feasible:
        reasons.add("INFEASIBLE_EPOCH_REPLACEMENT")
        rejected = not latency_ok
    if rejected:
        kind = "KEEP_EPOCH"
    else:
        kind = "RECOMPUTE_NOW"
        reasons.add("PLACEMENT_PUBLICATION_APPROVED")
    action = ModelPlacementAction(
        kind=kind,
        trigger_reasons=tuple(sorted(reasons)),
        demand_generation_sha256=snapshot.demand_generation_sha256,
        pressure_bucket=snapshot.pressure_bucket,
        expected_reuse_count=expected_reuse,
        break_even_use_count=break_even,
        old_component_identity_sha256=old_component,
        new_component_identity_sha256=new_component,
        selected_transition_energy_uj=(
            trigger.new_transition_energy_upper_uj
        ),
    )
    controller._record_event(snapshot, action)
    return action


def mark_background_complete(
    controller, artifact_sha256: str, observed_at_us: int
) -> None:
    artifact_sha256 = _sha256(
        "model placement background artifact", artifact_sha256
    )
    _integer("model placement background time", observed_at_us)
    controller._background_inflight.discard(artifact_sha256)
