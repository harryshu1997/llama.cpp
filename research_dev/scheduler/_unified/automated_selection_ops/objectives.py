"""AutomatedSelectionMixin objectives operations on its existing owner."""

from __future__ import annotations

from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..common import _paired_energy_upper_uj
from .common import _AutomatedSelectionContext
from .continuous_join import continuous_join_reason, continuous_join_rejections


def protected_work_budget_allows(
    policy: str,
    candidate: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
    minimum_saving_ppm: int,
    *,
    assistance_loss_upper_uj: int | None = 0,
) -> bool:
    """Admit a start beside protected work only on incremental fleet energy.

    Start now pays the alternative's own execution and preparation, the
    protected work's measured slowdown and the assistance that protected work
    may lose. Waiting pays the baseline's own execution and preparation plus
    only the idle energy accrued after the protected work ends; idle energy
    while it still runs is common to both and cancels. Unknown loss stays
    unknown and cannot be admitted.
    """
    marginal = candidate.marginal_system_cost
    if (
        policy != "energy-budgeted"
        or marginal is None
        or marginal.get("measured") is not True
        or assistance_loss_upper_uj is None
    ):
        return False
    baseline_marginal = baseline.marginal_system_cost or {}
    baseline_lower = baseline_marginal.get(
        "route_energy_incremental_lower_uj",
        baseline_marginal.get(
            "route_energy_lower_uj", baseline.cost.fleet_energy_lower_uj
        ),
    )
    alternative_upper = marginal.get(
        "route_energy_incremental_upper_uj",
        marginal.get("route_energy_upper_uj"),
    )
    extension_upper = marginal.get("upper_uj")
    if any(
        type(value) is not int or value < 0
        for value in (baseline_lower, alternative_upper, extension_upper,
                      assistance_loss_upper_uj)
    ):
        return False
    budget = (
        baseline_lower * (1_000_000 - minimum_saving_ppm) // 1_000_000
        - alternative_upper
    )
    return extension_upper + assistance_loss_upper_uj < budget


def _objective_row_rejection(
    controller,
    context: _AutomatedSelectionContext,
    row: AutomatedRouteCandidate,
) -> str | None:
    """Reject a non-baseline row before objective ranking; None keeps it."""
    candidate_set = context.candidate_set
    baseline = context.baseline
    selection_mode = context.selection_mode
    if (
        row.candidate_id
            == candidate_set.recovery_fallback_route_id
    ):
        return "RECOVERY_FALLBACK_ONLY"
    if row.candidate_id in context.excluded:
        return "FAILED_ROUTE_EXCLUDED"
    if controller._candidate_quarantine_reason(row) is not None:
        return controller._candidate_quarantine_reason(row)
    if row.candidate_id in context.runtime_rejection:
        return context.runtime_rejection[row.candidate_id]
    if not row.admitted:
        return (
            row.primary_rejection_reason
            or "PLACEMENT_INFEASIBLE"
        )
    if (
        row.marginal_system_cost is not None
        and int(row.marginal_system_cost["interference_upper_us"])
        > 0
    ):
        if (
            selection_mode not in {"energy-aware", "energy-first"}
            or not protected_work_budget_allows(
                controller._protected_work_policy, row, baseline,
                controller._runtime_capabilities.minimum_energy_saving_ppm,
                assistance_loss_upper_uj=(
                    controller._protected_assistance_loss_upper_uj(
                        context.request, row,
                        controller._objective_assistance_summaries(context),
                    )
                ),
            )
        ):
            return "PROTECTED_WORK_DELAY"
    if (
        selection_mode in {"energy-aware", "energy-first"}
        and row.cost.energy_evidence
            not in {"CALIBRATED", "MEASURED"}
        and not controller._assumed_phone_energy_allowed(row)
    ):
        return "ENERGY_EVIDENCE_NOT_QUALIFIED"
    if (
        selection_mode in {"energy-aware", "energy-first"}
        and baseline.cost.energy_evidence
            not in {"CALIBRATED", "MEASURED"}
        and not controller._assumed_phone_energy_allowed(baseline)
    ):
        return "BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED"
    if (
        row.cost.fleet_energy_lower_uj is None
        or row.cost.fleet_energy_upper_uj is None
        or baseline.cost.fleet_energy_lower_uj is None
    ):
        return "ENERGY_UNKNOWN"
    if row.paired_baseline_route_id is not None:
        return controller._paired_baseline_rejection(context, row)
    return None


def _paired_baseline_rejection(
    controller,
    context: _AutomatedSelectionContext,
    row: AutomatedRouteCandidate,
) -> str | None:
    selection_mode = context.selection_mode
    parent = context.candidate_by_id[row.paired_baseline_route_id]
    if (
        row.plan.desktop_placement_sha256 is None
        or row.plan.desktop_placement_sha256
            != parent.plan.desktop_placement_sha256
    ):
        return "PAIRED_BASELINE_PLACEMENT_MISMATCH"
    if parent.maturity != "QUALIFIED":
        return "PAIRED_BASELINE_NOT_QUALIFIED"
    if parent.cost.fleet_energy_lower_uj is None:
        return "PAIRED_BASELINE_ENERGY_UNKNOWN"
    if (
        selection_mode in {"energy-aware", "energy-first"}
        and parent.cost.energy_evidence
            not in {"CALIBRATED", "MEASURED"}
        and not controller._assumed_phone_energy_allowed(parent)
    ):
        return "PAIRED_BASELINE_ENERGY_NOT_QUALIFIED"
    paired_required_upper = (
        parent.cost.fleet_energy_lower_uj
        * (
            1_000_000
            - controller._runtime_capabilities
                .minimum_energy_saving_ppm
        )
        // 1_000_000
    )
    paired_energy_upper = _paired_energy_upper_uj(
        row,
        parent,
        controller._runtime_capabilities
            .placement_profile.energy_boundary_id,
        warm=False,
    )
    if (
        (
            row.cost.fleet_energy_upper_uj
            if paired_energy_upper is None
            else paired_energy_upper
        )
        > paired_required_upper
    ):
        return "PAIRED_BASELINE_ENERGY_NOT_POSITIVE"
    return None


def _objective_latency_rejection(
    controller,
    context: _AutomatedSelectionContext,
    row: AutomatedRouteCandidate,
    baseline_tardy: bool,
) -> str | None:
    request = context.request
    baseline = context.baseline
    baseline_duration_upper_us = max(
        1,
        baseline.cost.finish_upper_us - request.arrival_us,
    )
    relative_finish_upper_us = (
        request.arrival_us
        + (
            baseline_duration_upper_us
            * controller._runtime_capabilities.maximum_latency_ppm
            + 999_999
        ) // 1_000_000
    )
    finish_upper_limit_us = (
        relative_finish_upper_us
        if baseline_tardy
        else min(request.deadline_us, relative_finish_upper_us)
    )
    if row.cost.finish_upper_us > finish_upper_limit_us:
        return (
            "BASELINE_LATENCY_REGRESSION"
            if baseline_tardy
            else "SLO_UPPER_BOUND"
        )
    return None


def _pick_objective_candidate(
    controller,
    context: _AutomatedSelectionContext,
    alternatives: list[AutomatedRouteCandidate],
    effective_energy_upper_by_route: Mapping[str, int],
) -> tuple[AutomatedRouteCandidate, str]:
    request = context.request
    baseline = context.baseline
    selection_mode = context.selection_mode
    if selection_mode == "deadline-first":
        choices = (
            ([baseline] if context.baseline_available else [])
            + alternatives
        )
        if not choices:
            raise UnifiedScheduleError(
                "no qualified deadline-first route is available"
            )
        selected = min(choices, key=lambda row: (
            max(0, row.cost.finish_upper_us - request.deadline_us),
            row.cost.fleet_energy_upper_uj
            if row.cost.fleet_energy_upper_uj is not None else 2**63 - 1,
            row.cost.finish_upper_us,
            row.candidate_id,
        ))
        return selected, "MINIMUM_CONSERVATIVE_TARDINESS"
    if alternatives:
        selected = min(alternatives, key=lambda row: (
            effective_energy_upper_by_route.get(
                row.candidate_id,
                row.cost.fleet_energy_upper_uj,
            ),
            row.cost.finish_upper_us,
            row.candidate_id,
        ))
        if selection_mode == "energy-first":
            reason = "CONSERVATIVE_FLEET_ENERGY_FIRST"
        elif (
            effective_energy_upper_by_route.get(
                selected.candidate_id
            ) != selected.cost.fleet_energy_upper_uj
        ):
            reason = "CONSERVATIVE_PAIRED_FLEET_ENERGY_SAVING"
        else:
            reason = "CONSERVATIVE_FLEET_ENERGY_SAVING"
        if (
            selected.marginal_system_cost is not None
            and int(selected.marginal_system_cost["interference_upper_us"]) > 0
        ):
            reason = "BUDGETED_PROTECTED_WORK_ENERGY_SAVING"
        return selected, reason
    if context.baseline_available:
        return baseline, "QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE"
    raise UnifiedScheduleError(
        "qualified desktop baseline is not available: "
        + (context.runtime_rejection.get(baseline.candidate_id)
           or controller._candidate_quarantine_reason(baseline)
           or "FAILED_ROUTE_EXCLUDED")
    )


def _select_objective_candidate(
    controller,
    context: _AutomatedSelectionContext,
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    candidate_set = context.candidate_set
    request = context.request
    baseline = context.baseline
    selection_mode = context.selection_mode
    rejection: dict[str, str] = {}
    alternatives = []
    effective_energy_upper_by_route: dict[str, int] = {}
    baseline_tardy = baseline.cost.finish_upper_us > request.deadline_us
    join_rejection = continuous_join_rejections(controller, context)
    for row in candidate_set.candidates:
        if row.candidate_id == baseline.candidate_id:
            continue
        if row.candidate_id in join_rejection:
            rejection[row.candidate_id] = join_rejection[row.candidate_id]
            continue
        row_rejection = controller._objective_row_rejection(context, row)
        if row_rejection is not None:
            rejection[row.candidate_id] = row_rejection
            continue
        if selection_mode == "deadline-first":
            alternatives.append(row)
            continue
        if selection_mode != "energy-first":
            latency_rejection = controller._objective_latency_rejection(
                context, row, baseline_tardy
            )
            if latency_rejection is not None:
                rejection[row.candidate_id] = latency_rejection
                continue
        required_upper = (
            baseline.cost.fleet_energy_lower_uj
            * (
                1_000_000
                - controller._runtime_capabilities.minimum_energy_saving_ppm
            )
            // 1_000_000
        )
        effective_energy_upper = row.cost.fleet_energy_upper_uj
        if row.paired_baseline_route_id == baseline.candidate_id:
            paired_energy_upper = _paired_energy_upper_uj(
                row,
                baseline,
                controller._runtime_capabilities
                    .placement_profile.energy_boundary_id,
                warm=False,
            )
            if paired_energy_upper is not None:
                effective_energy_upper = paired_energy_upper
        if effective_energy_upper > required_upper:
            rejection[row.candidate_id] = "ENERGY_NOT_POSITIVE"
            continue
        effective_energy_upper_by_route[row.candidate_id] = (
            effective_energy_upper
        )
        alternatives.append(row)
    selected, reason = controller._pick_objective_candidate(
        context, alternatives, effective_energy_upper_by_route
    )
    for row in candidate_set.candidates:
        if row.candidate_id not in {
            selected.candidate_id, baseline.candidate_id
        } and row.candidate_id not in rejection:
            rejection[row.candidate_id] = "HIGHER_OBJECTIVE_COST"
    if selected.candidate_id != baseline.candidate_id:
        rejection.setdefault(
            baseline.candidate_id, "HIGHER_OBJECTIVE_COST"
        )
    return (
        selected,
        tuple(sorted(rejection.items())),
        continuous_join_reason(join_rejection, selected, baseline, reason),
    )


def quarantined_device_ids(controller) -> frozenset[str]:
    """Devices the elastic-phones quarantine (slice 2) currently excludes.

    Read from ``quarantined_devices`` on the scheduler or its runtime
    controller (a method or a collection); empty while that API is absent.
    """
    for owner in (controller, getattr(controller, "_runtime_controller", None)):
        value = getattr(owner, "quarantined_devices", None)
        if value is None:
            continue
        if callable(value):
            value = value()
        if isinstance(value, Mapping):
            value = tuple(value)
        if isinstance(value, (str, bytes)) or not isinstance(
            value, (set, frozenset, tuple, list)
        ) or any(type(row) is not str or not row for row in value):
            raise UnifiedScheduleError("quarantined devices are invalid")
        return frozenset(value)
    return frozenset()


def _require_recovery_devices_available(
    controller, candidate: AutomatedRouteCandidate, label: str
) -> None:
    quarantined = quarantined_device_ids(controller)
    if quarantined and set(candidate.plan.device_ids) & quarantined:
        raise UnifiedScheduleError(label + " uses a quarantined device")


def _select_automated_recovery_fallback(
    controller,
    candidate_set: AutomatedCandidateSet,
    *,
    failed_route_id: str,
    runtime_rejections: Mapping[str, str],
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    fallback = candidate_set.recovery_fallback
    if fallback.candidate_id == failed_route_id:
        raise UnifiedScheduleError(
            "failed route is also the recovery fallback"
        )
    if controller._runtime_controller.route_is_quarantined(
        fallback.candidate_id
    ) or controller._runtime_controller.resources_are_quarantined(
        fallback.plan.resource_ids
    ):
        raise UnifiedScheduleError(
            "qualified recovery fallback is quarantined"
        )
    _require_recovery_devices_available(
        controller, fallback, "qualified recovery fallback"
    )
    if fallback.candidate_id in runtime_rejections:
        raise UnifiedScheduleError(
            "qualified recovery fallback is not memory safe"
        )
    if not fallback.admitted:
        raise UnifiedScheduleError(
            "qualified recovery fallback is not admitted"
        )
    rejected = tuple(sorted(
        (
            row.candidate_id,
            "FAILED_ROUTE_EXCLUDED"
            if row.candidate_id == failed_route_id
            else "RECOVERY_FALLBACK_ONLY",
        )
        for row in candidate_set.candidates
        if row.candidate_id != fallback.candidate_id
    ))
    return fallback, rejected, "QUALIFIED_RECOVERY_FALLBACK"


def _select_adaptive_desktop_recovery(
    controller,
    candidate_set: AutomatedCandidateSet,
    *,
    failed_route_id: str,
    runtime_rejections: Mapping[str, str],
    exited_executor_id: str | None = None,
    masked_executor_id: str | None = None,
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    baseline = candidate_set.baseline
    if baseline.candidate_id == failed_route_id and (
        baseline.binding.executor_id not in {exited_executor_id, masked_executor_id} - {None}
    ):
        # (elastic phones: the failed attempt's server exited, so a hot residency of the same
        # route is a relaunched server, not the one that failed; or the live server masked the
        # lost helper out (S2a) and serves the route without it)
        raise UnifiedScheduleError(
            "failed adaptive route is the paired desktop baseline"
        )
    if controller._runtime_controller.route_is_quarantined(
        baseline.candidate_id
    ) or controller._runtime_controller.resources_are_quarantined(
        baseline.plan.resource_ids
    ):
        raise UnifiedScheduleError(
            "paired desktop recovery is quarantined"
        )
    # A dead paired desktop executor was reaped from the residency map, so
    # this baseline may be a cold route whose plan reloads it; that load is
    # the recovery's penalty and is admitted like any other transition.
    _require_recovery_devices_available(
        controller, baseline, "paired desktop recovery"
    )
    if baseline.candidate_id in runtime_rejections:
        raise UnifiedScheduleError(
            "paired desktop recovery is not memory safe"
        )
    if not baseline.admitted:
        raise UnifiedScheduleError(
            "paired desktop recovery is not admitted"
        )
    rejected = tuple(sorted(
        (
            row.candidate_id,
            "FAILED_ROUTE_EXCLUDED"
            if row.candidate_id == failed_route_id
            else "PAIRED_DESKTOP_RECOVERY_ONLY",
        )
        for row in candidate_set.candidates
        if row.candidate_id != baseline.candidate_id
    ))
    return baseline, rejected, "PAIRED_DESKTOP_RECOVERY"
