"""AutomatedSelectionMixin adaptive operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.route_generation import AutomatedRouteCompiler
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.types import canonical_sha256
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from ..._internal.adaptive_decode_planning import adaptive_decode_policies, adaptive_probe_contracts
from .common import _AutomatedSelectionContext
from .continuous_join import continuous_join_reason, continuous_join_rejections


def _select_automated_candidate(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    observed_at_us: int | None = None,
    excluded_route_ids: Sequence[str] = (),
    selection_mode: str = "energy-aware",
    runtime_rejections: Mapping[str, str] | None = None,
    route_compiler: AutomatedRouteCompiler | None = None,
    snapshot: HeterogeneousRuntimeSnapshot | None = None,
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    if controller._runtime_capabilities is None:
        raise UnifiedScheduleError("runtime capabilities are not registered")
    baseline = candidate_set.baseline
    excluded = frozenset(excluded_route_ids)
    runtime_rejection = dict(runtime_rejections or {})
    candidate_ids = {
        row.candidate_id for row in candidate_set.candidates
    }
    candidate_by_id = {
        row.candidate_id: row for row in candidate_set.candidates
    }
    controller._reject_adaptive_routes_without_contract(
        candidate_set, request, runtime_rejection
    )
    if set(runtime_rejection) - candidate_ids:
        raise UnifiedScheduleError(
            "runtime rejection references an unknown candidate"
        )
    baseline_available = (
        baseline.candidate_id not in excluded
        and baseline.candidate_id not in runtime_rejection
        and controller._candidate_quarantine_reason(baseline) is None
    )
    observed_at_us = (
        request.arrival_us if observed_at_us is None else observed_at_us
    )
    if selection_mode == "desktop-baseline":
        if baseline_available:
            return (
                baseline,
                tuple(sorted(
                    (
                        row.candidate_id,
                        "BASELINE_CONTROL_NOT_SELECTED",
                    )
                    for row in candidate_set.candidates
                    if row.candidate_id != baseline.candidate_id
                )),
                "DESKTOP_BASELINE_CONTROL",
            )
        raise UnifiedScheduleError(
            "frozen desktop control is not currently feasible"
        )
    if selection_mode not in {
        "adaptive-decode",
        "calibration",
        "deadline-first",
        "desktop-baseline",
        "energy-aware",
        "energy-first",
    }:
        raise UnifiedScheduleError(
            "automated selection mode is invalid"
        )
    context = _AutomatedSelectionContext(
        candidate_set=candidate_set,
        request=request,
        baseline=baseline,
        excluded=excluded,
        runtime_rejection=runtime_rejection,
        candidate_by_id=candidate_by_id,
        baseline_available=baseline_available,
        selection_mode=selection_mode,
        route_compiler=route_compiler,
        observed_at_us=observed_at_us,
        snapshot=snapshot,
    )
    if selection_mode == "adaptive-decode":
        return controller._select_adaptive_decode_candidate(context)
    if selection_mode == "calibration":
        return controller._select_calibration_candidate(context)
    return controller._select_objective_candidate(context)


def _reject_adaptive_routes_without_contract(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    runtime_rejection: dict[str, str],
) -> None:
    if not any(
        row.plan.execution_contract.execution_mode == "adaptive-split"
        for row in candidate_set.candidates
    ):
        return
    try:
        adaptive_contract_routes = frozenset(
            adaptive_probe_contracts(
                candidate_set,
                controller.runtime_model_manifest(candidate_set.model_id),
                controller._runtime_capabilities,
                request.output_tokens,
            )
        )
    except AdaptiveDecodeError:
        adaptive_contract_routes = frozenset()
    for row in candidate_set.candidates:
        if (
            row.plan.execution_contract.execution_mode
                == "adaptive-split"
            and row.candidate_id not in adaptive_contract_routes
        ):
            runtime_rejection.setdefault(
                row.candidate_id,
                "ADAPTIVE_EXECUTION_CONTRACT_ABSENT",
            )


def _candidate_quarantine_reason(
    controller,
    row: AutomatedRouteCandidate,
) -> str | None:
    if (
        controller._maximum_phone_sessions is not None
        and len(row.plan.execution_contract.phone_shards)
            > controller._maximum_phone_sessions
    ):
        return "PHONE_SESSION_LIMIT"
    if controller._runtime_controller.route_is_quarantined(
        row.candidate_id
    ):
        return "ROUTE_QUARANTINED"
    if controller._runtime_controller.resources_are_quarantined(
        row.plan.resource_ids
    ):
        return "RESOURCE_QUARANTINED"
    return None


def _assumed_phone_energy_allowed(
    controller,
    row: AutomatedRouteCandidate,
) -> bool:
    catalog_profiles = tuple(
        controller._runtime_capabilities.phone_power_profiles
    )
    expected_profile_sha256s = tuple(sorted(
        canonical_sha256(profile)
        for profile in catalog_profiles
    ))
    learned = row.residency_break_even or {}
    if (
        row.cost.energy_evidence == "ASSUMED"
        and bool(catalog_profiles)
        and all(
            profile.allow_assumed_for_scheduling
            for profile in catalog_profiles
        )
        and learned.get("phone_energy_evidence")
            == "ASSUMED_4P5W"
        and tuple(learned.get(
            "phone_power_profile_sha256s", ()
        )) == expected_profile_sha256s
    ):
        return True
    parameters = row.plan.adapter_parameters
    raw_profile_devices = parameters.get(
        "phone_power_device_ids"
    )
    profile_devices = (
        ()
        if type(raw_profile_devices) is not str
        else tuple(raw_profile_devices.split(","))
    )
    profiles = tuple(
        controller._runtime_capabilities
            .phone_power_profile_by_device[device_id]
        for device_id in profile_devices
    )
    return (
        row.cost.energy_evidence == "ASSUMED"
        and bool(profiles)
        and all(
            profile.allow_assumed_for_scheduling
            for profile in profiles
        )
        and parameters.get("phone_power_allow_assumed") == 1
        and parameters.get("phone_power_device_ids")
            == ",".join(sorted(profile_devices))
        and parameters.get("phone_power_evidence_kind")
            == "ASSUMED_4P5W"
        and all(
            parameters.get("phone_power_active_mw")
                == profile.active_power_mw
            and parameters.get("phone_power_idle_mw")
                == profile.idle_power_mw
            and parameters.get("phone_power_estimation_version")
                == profile.estimation_version
            for profile in profiles
        )
    )


def _adaptive_exploration_budget_exceeded(
    controller,
    row: AutomatedRouteCandidate,
    baseline: AutomatedRouteCandidate,
) -> bool:
    break_even = row.residency_break_even
    if break_even is None or bool(break_even.get("passed", False)):
        return False
    config = controller._adaptive_decode_config
    transition_energy_uj = int(
        break_even.get(
            "incremental_transition_energy_uj",
            break_even.get("transition_energy_uj", 0),
        )
    )
    transition_latency_us = int(
        break_even.get(
            "incremental_transition_latency_us",
            sum(
                transition.latency_us
                for transition in row.plan.transitions
            ),
        )
    )
    baseline_energy_uj = baseline.cost.fleet_energy_uj
    return (
        baseline_energy_uj is None
        or transition_energy_uj
            > baseline_energy_uj
                * config.exploration_energy_budget_ppm
                // 1_000_000
        or transition_latency_us
            > baseline.cost.service_us
                * config.exploration_latency_budget_ppm
                // 1_000_000
    )


def _adaptive_eligible_policies(
    controller,
    context: _AutomatedSelectionContext,
    policies,
    envelope: AutomatedRouteCandidate | None,
    rejection: dict[str, str],
) -> list:
    eligible = []
    for policy in policies:
        row = context.candidate_by_id.get(policy.route_id)
        if row is None and envelope is not None and (
            policy.executor_id == envelope.binding.executor_id
            and policy.operator_plan_sha256
                == envelope.plan.plan_sha256
        ):
            row = envelope
        if row is None:
            continue
        if row.candidate_id in context.excluded:
            rejection[row.candidate_id] = "FAILED_ROUTE_EXCLUDED"
            continue
        quarantine = controller._candidate_quarantine_reason(row)
        if quarantine is not None:
            rejection[row.candidate_id] = quarantine
            continue
        if row.candidate_id in context.runtime_rejection:
            rejection[row.candidate_id] = context.runtime_rejection[
                row.candidate_id
            ]
            continue
        if (
            row.marginal_system_cost is not None
            and int(row.marginal_system_cost[
                "interference_upper_us"
            ]) > 0
        ):
            rejection[row.candidate_id] = "PROTECTED_WORK_DELAY"
            continue
        if controller._adaptive_exploration_budget_exceeded(
            row, context.baseline
        ):
            rejection[row.candidate_id] = (
                "COLD_RESIDENCY_EXPLORATION_BUDGET"
            )
            continue
        eligible.append((policy, row))
    return eligible


def _select_adaptive_decode_candidate(
    controller,
    context: _AutomatedSelectionContext,
) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
    candidate_set = context.candidate_set
    request = context.request
    baseline = context.baseline
    try:
        _, policies, envelope = adaptive_decode_policies(
            candidate_set,
            controller.runtime_model_manifest(candidate_set.model_id),
            controller._runtime_capabilities,
            request.output_tokens,
            controller._maximum_phone_sessions,
        )
    except AdaptiveDecodeError:
        policies = ()
        envelope = None
    rejection: dict[str, str] = {}
    eligible = controller._adaptive_eligible_policies(
        context, policies, envelope, rejection
    )
    join_rejection = continuous_join_rejections(
        controller, context, (row for _policy_row, row in eligible)
    )
    if join_rejection:
        rejection.update(join_rejection)
        eligible = [
            item for item in eligible
            if item[1].candidate_id not in join_rejection
        ]
    hot_envelope_available = any(
        row.plan.residency_variant == "hot"
        for _policy_row, row in eligible
    )
    if eligible and (
        request.output_tokens
            >= controller._adaptive_envelope_minimum_remaining_tokens
        or hot_envelope_available
    ):
        policy, original = max(
            eligible,
            key=lambda item: (
                item[1].plan.residency_variant == "hot",
                len(item[0].layer_indices),
                item[0].columns,
                item[0].policy_hash,
            ),
        )
        selected = replace(
            original,
            binding=replace(
                original.binding,
                ready=True,
                eligibility_reasons=(),
            ),
            admitted=True,
            rejection_reasons=(),
        )
        reason = (
            "INTRA_REQUEST_CALIBRATION_ENVELOPE"
            if request.output_tokens
                >= controller._adaptive_envelope_minimum_remaining_tokens
            else "HOT_ADAPTIVE_ENVELOPE_REUSED"
        )
    elif context.baseline_available:
        selected = baseline
        reason = "ADAPTIVE_BASELINE_NO_SAFE_ENVELOPE"
    else:
        raise UnifiedScheduleError(
            "adaptive decode has no safe desktop baseline"
        )
    for row in candidate_set.candidates:
        if row.candidate_id not in {
            selected.candidate_id, baseline.candidate_id
        } and row.candidate_id not in rejection:
            rejection[row.candidate_id] = (
                controller._candidate_quarantine_reason(row)
                or "NOT_SELECTED_FOR_ADAPTIVE_ENVELOPE"
            )
    if selected.candidate_id != baseline.candidate_id:
        rejection.setdefault(
            baseline.candidate_id,
            "ADAPTIVE_ENVELOPE_PREPARED",
        )
    return (
        selected,
        tuple(sorted(rejection.items())),
        continuous_join_reason(join_rejection, selected, baseline, reason),
    )
