"""PlacementEpochMixin selection operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from ..._internal.model_placement_controller import (
    ModelDemandSnapshot,
    ModelPlacementAction,
    ModelPlacementTrigger,
)
from ..._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from ..._internal.runtime_residency_cohorts import (
    RuntimeModelPlacementEpoch,
    runtime_residency_component_identity,
)
from ..common import _paired_energy_upper_uj, _residency_break_even_warm_upper_uj
from .common import _PlacementChoice, _PlacementPublicationBasis


def _initial_placement_choice(
    controller,
    *,
    selected: AutomatedRouteCandidate,
    templates: RuntimeRouteTemplateSet,
    candidate_set: AutomatedCandidateSet,
    demand: ModelDemandSnapshot,
    placement_action: ModelPlacementAction | None,
    invalidation_reason: str,
) -> tuple[_PlacementChoice, _PlacementPublicationBasis]:
    expected_reuse = max(
        demand.active_request_count + demand.queued_request_count, 1
    )
    observed_reuse = expected_reuse
    trigger_reasons = (
        placement_action.trigger_reasons
        if placement_action is not None
        else (
            "ROUTE_SELECTED"
            if invalidation_reason == "NONE"
            else invalidation_reason,
        )
    )
    paired_id = (
        selected.paired_baseline_route_id
        or candidate_set.baseline_route_id
    )
    paired = next(
        row for row in candidate_set.candidates
        if row.candidate_id == paired_id
    )
    paired_warm_upper = _paired_energy_upper_uj(
        selected,
        paired,
        controller._runtime_capabilities.placement_profile.energy_boundary_id,
        warm=True,
    )
    transition_latency, transition_energy = (
        controller._placement_transition_cost(selected)
    )
    break_even = selected.residency_break_even
    break_even_count = None
    publication_transition_energy = transition_energy
    restore_energy = 0
    candidate_warm = _residency_break_even_warm_upper_uj(
        break_even, paired_upper_uj=paired_warm_upper
    )
    if break_even is not None and candidate_warm is not None:
        desktop_warm = paired.cost.warm_execution_energy_lower_uj
        if desktop_warm is None:
            raise UnifiedScheduleError(
                "paired desktop warm energy is unknown"
            )
        transition_total = int(break_even.get(
            "incremental_transition_energy_uj",
            break_even.get("transition_energy_uj", 0),
        ))
        publication_transition_energy = transition_total
        restore_energy = int(break_even.get("restore_energy_uj", 0))
        transition_total += restore_energy
        warm_saving = desktop_warm - candidate_warm
        if warm_saving > 0:
            break_even_count = (
                transition_total + warm_saving - 1
            ) // warm_saving
        expected_reuse = max(
            expected_reuse,
            int(break_even.get("expected_use_count", 1)),
        )
    return (
        _PlacementChoice(
            selected=selected,
            templates=templates,
            paired=paired,
            expected_reuse_count=expected_reuse,
            observed_reuse_count=observed_reuse,
            trigger_reasons=trigger_reasons,
            transition_latency_us=transition_latency,
            transition_energy_uj=transition_energy,
            break_even_use_count=break_even_count,
        ),
        _PlacementPublicationBasis(
            paired_warm_upper_uj=paired_warm_upper,
            transition_energy_uj=publication_transition_energy,
            restore_energy_uj=restore_energy,
            predicted_reuse_count=max(
                0, expected_reuse - observed_reuse
            ),
        ),
    )


def _placement_publication_action(
    controller,
    *,
    choice: _PlacementChoice,
    basis: _PlacementPublicationBasis,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    demand: ModelDemandSnapshot,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    current_epoch: RuntimeModelPlacementEpoch | None,
    selection_mode: str,
) -> tuple[ModelPlacementAction | None, AutomatedRouteCandidate | None]:
    old_candidate = (
        next((
            row for row in candidate_set.candidates
            if row.candidate_id == current_epoch.selected_route_id
        ), None)
        if current_epoch is not None else choice.paired
    )
    if old_candidate is None:
        return None, None
    old_warm_lower = (
        old_candidate.cost.warm_execution_energy_lower_uj
    )
    new_warm_upper = (
        choice.selected.cost.warm_execution_energy_upper_uj
    )
    if (
        old_candidate.candidate_id == choice.paired.candidate_id
        and basis.paired_warm_upper_uj is not None
    ):
        new_warm_upper = basis.paired_warm_upper_uj
    current_state = snapshot.executors.get(
        old_candidate.binding.executor_id
    )
    current_feasible = (
        old_candidate.admitted
        and current_state is not None
        and current_state.healthy
    )
    old_component = (
        demand.current_resident_component_identity_sha256
        or runtime_residency_component_identity(
            manifest.artifact_sha256,
            old_candidate.plan,
            old_candidate.binding,
        ).identity_sha256
    )
    uses_phone = bool(
        choice.selected.plan.execution_contract.phone_shards
        or any(
            row.kind == "session_residency_constraint"
            for row in choice.selected.plan.memory_demands
        )
    )
    proposed_component = (
        choice.templates.selected_component_identity_sha256
        if uses_phone else old_component
    )
    energy_known = (
        old_warm_lower is not None and new_warm_upper is not None
    )
    desktop_latency = choice.paired.cost.service_upper_us
    proposed_latency = choice.selected.cost.service_upper_us
    if (
        choice.selected.plan.execution_contract.execution_mode == "desktop"
        and not choice.selected.plan.transitions
    ):
        # Dispatch to existing residency uses the request's queue-aware bound.
        desktop_latency = max(1, choice.paired.cost.finish_upper_us - request.arrival_us)
        proposed_latency = max(1, choice.selected.cost.finish_upper_us - request.arrival_us)
    action = controller._model_placement_controller.authorize_publication(
        ModelPlacementTrigger(
            snapshot=demand,
            epoch_sha256=(
                None if current_epoch is None
                else current_epoch.epoch_sha256
            ),
            epoch_demand_generation_sha256=(
                None if current_epoch is None
                else current_epoch.demand_generation_sha256
            ),
            epoch_pressure_bucket=(
                None if current_epoch is None
                else current_epoch.pressure_bucket
            ),
            epoch_valid_until_us=(
                None if current_epoch is None
                else current_epoch.valid_until_us
            ),
            selected_route_feasible=current_feasible,
            notification_reasons=("PROSPECTIVE_PLACEMENT_READY",),
            old_component_identity_sha256=old_component,
            proposed_component_identity_sha256=proposed_component,
            old_warm_energy_lower_uj=(
                old_warm_lower if energy_known else None
            ),
            old_warm_energy_upper_uj=(
                old_candidate.cost.warm_execution_energy_upper_uj
                if energy_known else None
            ),
            new_warm_energy_upper_uj=(
                new_warm_upper if energy_known else None
            ),
            new_transition_energy_upper_uj=(
                basis.transition_energy_uj if energy_known else None
            ),
            old_restore_energy_upper_uj=(
                basis.restore_energy_uj if energy_known else None
            ),
            desktop_latency_upper_us=desktop_latency,
            proposed_latency_upper_us=proposed_latency,
            maximum_latency_ppm=(
                2**63 - 1
                if selection_mode == "energy-first"
                else controller._runtime_capabilities.maximum_latency_ppm
            ),
            predicted_reuse_count=basis.predicted_reuse_count,
        )
    )
    return action, old_candidate


def _retained_placement_choice(
    controller,
    *,
    choice: _PlacementChoice,
    selected: AutomatedRouteCandidate,
    compiler: AutomatedRouteCompiler,
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    input_bucket: int,
    output_bucket: int,
) -> _PlacementChoice:
    templates = controller._compile_placement_templates(
        compiler,
        candidate_set,
        selected,
        manifest,
        request,
        snapshot,
        input_bucket,
        output_bucket,
    )
    paired_id = (
        selected.paired_baseline_route_id
        or candidate_set.baseline_route_id
    )
    paired = next(
        row for row in candidate_set.candidates
        if row.candidate_id == paired_id
    )
    latency, energy = controller._placement_transition_cost(selected)
    expected_reuse = choice.observed_reuse_count
    break_even = selected.residency_break_even
    break_even_count = None
    candidate_warm = _residency_break_even_warm_upper_uj(break_even)
    if break_even is not None and candidate_warm is not None:
        expected_reuse = max(
            expected_reuse,
            int(break_even.get("expected_use_count", 1)),
        )
        desktop_warm = paired.cost.warm_execution_energy_lower_uj
        if desktop_warm is None:
            raise UnifiedScheduleError(
                "paired desktop warm energy is unknown"
            )
        warm_saving = desktop_warm - candidate_warm
        if warm_saving > 0:
            transition_total = int(break_even.get(
                "incremental_transition_energy_uj",
                break_even.get("transition_energy_uj", 0),
            )) + int(break_even.get("restore_energy_uj", 0))
            break_even_count = (
                transition_total + warm_saving - 1
            ) // warm_saving
    return replace(
        choice,
        selected=selected,
        templates=templates,
        paired=paired,
        expected_reuse_count=expected_reuse,
        transition_latency_us=latency,
        transition_energy_uj=energy,
        break_even_use_count=break_even_count,
    )


def _authorize_placement_choice(
    controller,
    *,
    choice: _PlacementChoice,
    basis: _PlacementPublicationBasis,
    compiler: AutomatedRouteCompiler,
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    demand: ModelDemandSnapshot,
    current_epoch: RuntimeModelPlacementEpoch | None,
    selection_mode: str,
    input_bucket: int,
    output_bucket: int,
) -> _PlacementChoice:
    action, old_candidate = controller._placement_publication_action(
        choice=choice,
        basis=basis,
        candidate_set=candidate_set,
        request=request,
        demand=demand,
        manifest=manifest,
        snapshot=snapshot,
        current_epoch=current_epoch,
        selection_mode=selection_mode,
    )
    if action is None:
        return choice
    if action.kind == "KEEP_EPOCH":
        if old_candidate is None:
            raise UnifiedScheduleError(
                "model placement publication rejected without "
                    "the current route template"
            )
        choice = controller._retained_placement_choice(
            choice=choice,
            selected=old_candidate,
            compiler=compiler,
            candidate_set=candidate_set,
            manifest=manifest,
            request=request,
            snapshot=snapshot,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
        )
    else:
        choice = replace(
            choice,
            break_even_use_count=action.break_even_use_count,
        )
    return replace(
        choice,
        trigger_reasons=tuple(sorted(set(
            choice.trigger_reasons + action.trigger_reasons
        ))),
    )


def _authoritative_phone_placement_choice(
    controller,
    *,
    choice: _PlacementChoice,
    compiler: AutomatedRouteCompiler,
    candidate_set: AutomatedCandidateSet,
    manifest: ModelManifest,
    request: Request,
    snapshot: HeterogeneousRuntimeSnapshot,
    input_bucket: int,
    output_bucket: int,
) -> _PlacementChoice:
    selected_geometry = choice.selected.plan.adapter_parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    phone_layout = (
        controller._model_placement_controller.planning_phone_layout()
    )
    if (
        not choice.selected.plan.execution_contract.phone_shards
        or (
            phone_layout is not None
            and selected_geometry
                == phone_layout.layout.geometry_sha256
        )
    ):
        return choice
    selected = choice.paired
    templates = controller._compile_placement_templates(
        compiler,
        candidate_set,
        selected,
        manifest,
        request,
        snapshot,
        input_bucket,
        output_bucket,
    )
    latency, energy = controller._placement_transition_cost(selected)
    return replace(
        choice,
        selected=selected,
        templates=templates,
        transition_latency_us=latency,
        transition_energy_uj=energy,
        break_even_use_count=None,
        expected_reuse_count=choice.observed_reuse_count,
        trigger_reasons=tuple(sorted(set((
            *choice.trigger_reasons,
            "PHONE_LAYOUT_NOT_AUTHORITATIVE",
        )))),
    )
