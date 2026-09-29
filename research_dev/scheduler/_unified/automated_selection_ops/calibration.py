"""AutomatedSelectionMixin calibration operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_plan import AutomatedRouteCandidate, primary_rejection_reason
from .common import _AutomatedSelectionContext


def _calibration_physically_qualified(
    controller,
    row: AutomatedRouteCandidate,
) -> bool:
    capabilities = tuple(
        controller._runtime_capabilities.executor_by_device[device_id]
        for device_id in row.device_ids
    )
    coordinator = (
        controller._runtime_capabilities.composite_executor_by_id.get(
            row.binding.executor_id
        )
    )
    direct = (
        controller._runtime_capabilities.executor_by_id.get(
            row.binding.executor_id
        )
    )
    return (
        all(
            capability.maturity == "QUALIFIED"
            for capability in capabilities
        )
        and row.binding.endpoint is not None
        and row.binding.operator_plan_protocol is not None
        and (
            (
                coordinator is not None
                and coordinator.maturity in {
                    "CALIBRATION_PENDING",
                    "SHADOW",
                    "QUALIFIED",
                }
                and bool(coordinator.evidence_ids)
            )
            or (
                direct is not None
                and direct.maturity == "QUALIFIED"
                and direct.supports_whole_model
                and direct.adapter_parameters.get(
                    "execution_adapter"
                ) is not None
                and direct.adapter_parameters.get(
                    "request_io_protocol"
                ) is not None
            )
        )
        and all(
            transition.maturity == "QUALIFIED"
            for transition in row.plan.transitions
        )
    )


def _calibration_alternatives(
    controller,
    context: _AutomatedSelectionContext,
    rejection: dict[str, str],
) -> list[AutomatedRouteCandidate]:
    candidate_set = context.candidate_set
    baseline = context.baseline
    allowed_rejections = {
        "COLD_RESIDENCY_BREAK_EVEN",
        "ENERGY_UNKNOWN",
        "ROUTE_NOT_QUALIFIED",
        "SLO_UPPER_BOUND",
    }
    alternatives = []
    for row in candidate_set.candidates:
        if row.candidate_id == baseline.candidate_id:
            continue
        if (
            row.candidate_id
                == candidate_set.recovery_fallback_route_id
        ):
            rejection[row.candidate_id] = (
                "RECOVERY_FALLBACK_ONLY"
            )
            continue
        if row.candidate_id in context.excluded:
            rejection[row.candidate_id] = "FAILED_ROUTE_EXCLUDED"
            continue
        if controller._candidate_quarantine_reason(row) is not None:
            rejection[row.candidate_id] = (
                controller._candidate_quarantine_reason(row)
            )
            continue
        if row.candidate_id in context.runtime_rejection:
            rejection[row.candidate_id] = context.runtime_rejection[
                row.candidate_id
            ]
            continue
        remaining = set(row.rejection_reasons) - allowed_rejections
        physically_qualified = controller._calibration_physically_qualified(
            row
        )
        if remaining:
            rejection[row.candidate_id] = (
                primary_rejection_reason(tuple(remaining))
                or "PLACEMENT_INFEASIBLE"
            )
            continue
        if not physically_qualified:
            rejection[row.candidate_id] = (
                "CALIBRATION_PHYSICAL_EVIDENCE_INCOMPLETE"
            )
            continue
        if (
            row.maturity == "QUALIFIED"
            and "ENERGY_UNKNOWN" not in row.rejection_reasons
            and all(
                transition.energy_maturity == "QUALIFIED"
                for transition in row.plan.transitions
            )
        ):
            rejection[row.candidate_id] = (
                "CALIBRATION_EVIDENCE_COMPLETE"
            )
            continue
        if (
            row.marginal_system_cost is not None
            and int(row.marginal_system_cost[
                "interference_upper_us"
            ]) > 0
        ):
            rejection[row.candidate_id] = "PROTECTED_WORK_DELAY"
            continue
        alternatives.append(replace(
            row,
            binding=replace(
                row.binding,
                ready=True,
                eligibility_reasons=(),
            ),
            admitted=True,
            rejection_reasons=(),
        ))
    return alternatives


def _calibration_coverage_bytes(
    controller,
    row: AutomatedRouteCandidate,
    baseline_devices: frozenset[str],
) -> int:
    comparison_devices = baseline_devices
    coordinator = (
        controller._runtime_capabilities
        .composite_executor_by_id.get(
            row.binding.executor_id
        )
    )
    if (
        coordinator is not None
        and coordinator.baseline_executor_id is not None
    ):
        comparison_devices = frozenset(
            controller._runtime_capabilities
            .composite_executor_by_id[
                coordinator.baseline_executor_id
            ].participant_device_ids
        )
    return sum(
        demand.required_bytes
        for demand in row.plan.memory_demands
        if demand.kind == "model_weights"
        and demand.device_id not in comparison_devices
    )


def _calibration_mechanism_rank(
    controller,
    row: AutomatedRouteCandidate,
) -> int:
    coordinator = (
        controller._runtime_capabilities
        .composite_executor_by_id.get(
            row.binding.executor_id
        )
    )
    direct = controller._runtime_capabilities.executor_by_id.get(
        row.binding.executor_id
    )
    maturity = (
        coordinator.maturity
        if coordinator is not None
        else direct.maturity
        if direct is not None
        else "QUALIFIED"
    )
    return {
        "CALIBRATION_PENDING": 0,
        "SHADOW": 1,
        "QUALIFIED": 2,
    }.get(maturity, 3)


def _rank_calibration_alternatives(
    controller,
    context: _AutomatedSelectionContext,
    alternatives: list[AutomatedRouteCandidate],
) -> tuple[AutomatedRouteCandidate, str]:
    request = context.request
    route_compiler = context.route_compiler
    baseline_devices = frozenset(context.baseline.device_ids)
    manifest = controller.runtime_model_manifest(
        context.candidate_set.model_id
    )

    def calibration_information(row) -> Mapping[str, int]:
        compiler = (
            controller._automated_compiler()
            if route_compiler is None else route_compiler
        )
        return compiler.calibration_information(
            manifest, request, row.plan
        )

    cohort_alternatives = tuple(
        row for row in alternatives
        if controller._runtime_decode_cohorts.can_join(
            row.plan,
            row.binding,
            quality_requirement=request.quality_requirement,
        )
    )
    selection_pool = (
        cohort_alternatives
        if cohort_alternatives else alternatives
    )
    selected = min(
        selection_pool,
        key=lambda row: (
            controller._calibration_mechanism_rank(row),
            -controller._calibration_coverage_bytes(row, baseline_devices),
            calibration_information(row)["priority_class"],
            calibration_information(row)["samples_needed"],
            row.cost.finish_upper_us,
            row.cost.finish_us,
            row.candidate_id,
        ),
    )
    reason = (
        "PHYSICAL_CALIBRATION_COHORT_FOLLOWER"
        if cohort_alternatives
        else "PHYSICAL_CALIBRATION_SAMPLE"
    )
    return selected, reason


def _select_calibration_candidate(
    controller,
    context: _AutomatedSelectionContext,
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    candidate_set = context.candidate_set
    baseline = context.baseline
    rejection: dict[str, str] = {}
    alternatives = controller._calibration_alternatives(context, rejection)
    if alternatives:
        selected, reason = controller._rank_calibration_alternatives(
            context, alternatives
        )
    elif context.baseline_available:
        selected = baseline
        reason = "CALIBRATION_FALLBACK_NO_SAFE_ALTERNATIVE"
    else:
        raise UnifiedScheduleError(
            "no safe calibration route or desktop baseline is available"
        )
    for row in candidate_set.candidates:
        if row.candidate_id not in {
            selected.candidate_id, baseline.candidate_id
        } and row.candidate_id not in rejection:
            rejection[row.candidate_id] = "HIGHER_CALIBRATION_PRIORITY"
    if selected.candidate_id != baseline.candidate_id:
        rejection.setdefault(
            baseline.candidate_id, "CALIBRATION_ROUTE_SELECTED"
        )
    return selected, tuple(sorted(rejection.items())), reason
