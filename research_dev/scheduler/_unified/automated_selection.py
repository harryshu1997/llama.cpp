"""Automated candidate selection: placement compatibility, materialization, memory admission.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

import weakref
from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.policy import LeasePreview, Request
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from .._internal.model_placement_controller import ModelDemandSnapshot, ModelPlacementAction
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    HelperOpportunity,
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionPlan,
)
from .._internal.runtime_resources import RuntimeResidencyProjectionToken, RuntimeResourceError
from .._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from .common import _ModelPlacementCompatibility, _runtime_serialized

from .automated_selection_ops.common import (
    _AutomatedSelectionContext as _AutomatedSelectionContext,
    _ModelPlacementMaterializeContext as _ModelPlacementMaterializeContext,
    _ModelPlacementPass as _ModelPlacementPass,
)
from .automated_selection_ops import adaptive as _adaptive
from .automated_selection_ops import attachment as _attachment
from .automated_selection_ops import calibration as _calibration
from .automated_selection_ops import dormant as _dormant
from .automated_selection_ops import helpers as _helpers
from .automated_selection_ops import materialization as _materialization
from .automated_selection_ops import objectives as _objectives
from .automated_selection_ops import placement as _placement
from .automated_selection_ops import resources as _resources


class AutomatedSelectionMixin:
    """Automated candidate selection: placement compatibility, materialization, memory admission."""

    @_runtime_serialized
    def generate_automated_candidates(
        self,
        request: Request,
        model_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int | None = None,
    ) -> AutomatedCandidateSet:
        """Generate and cost the bounded capability-backed search frontier."""
        return _placement.generate_automated_candidates(
            self,
            request,
            model_id,
            snapshot,
            observed_at_us=observed_at_us,
        )

    def _automated_snapshot_for_request(
        self,
        request: Request,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        exclude_request_id: str | None = None,
        project_before_us: int | None = None,
        project_request_ids: Sequence[str] | None = None,
    ) -> HeterogeneousRuntimeSnapshot:
        return _placement._automated_snapshot_for_request(
            self,
            request,
            snapshot,
            exclude_request_id=exclude_request_id,
            project_before_us=project_before_us,
            project_request_ids=project_request_ids,
        )

    @staticmethod
    def _model_placement_candidate_set(
        candidate_set: AutomatedCandidateSet,
    ) -> AutomatedCandidateSet:
        """Build a non-dispatching view for model epoch comparison."""
        return _placement._model_placement_candidate_set(candidate_set)

    def _model_placement_compatibility(
        self,
        epoch: RuntimeModelPlacementEpoch,
        candidate: AutomatedRouteCandidate,
        manifest: ModelManifest,
        candidate_set: AutomatedCandidateSet | None = None,
    ) -> _ModelPlacementCompatibility:
        return _placement._model_placement_compatibility(
            self,
            epoch,
            candidate,
            manifest,
            candidate_set,
        )

    def _authorize_epoch_compatible_routes(
        self,
        candidate_set: AutomatedCandidateSet,
        epoch: RuntimeModelPlacementEpoch,
        manifest: ModelManifest,
    ) -> AutomatedCandidateSet:
        return _placement._authorize_epoch_compatible_routes(self, candidate_set, epoch, manifest)

    @staticmethod
    def _candidate_set_with_placement_resolution(
        candidate_set: AutomatedCandidateSet,
        passes: Sequence[Mapping[str, object]],
        *,
        outcome: str,
        invalidated_epoch_sha256: str | None = None,
    ) -> AutomatedCandidateSet:
        return _placement._candidate_set_with_placement_resolution(
            candidate_set,
            passes,
            outcome=outcome,
            invalidated_epoch_sha256=invalidated_epoch_sha256,
        )

    @staticmethod
    def _candidate_set_without_published_epoch(
        candidate_set: AutomatedCandidateSet,
    ) -> AutomatedCandidateSet:
        return _placement._candidate_set_without_published_epoch(candidate_set)

    def _model_placement_resolution_pass(
        self,
        epoch: RuntimeModelPlacementEpoch,
        proposed_route_id: str,
        selected: AutomatedRouteCandidate,
        manifest: ModelManifest,
        candidate_set: AutomatedCandidateSet,
    ) -> Mapping[str, object]:
        return _placement._model_placement_resolution_pass(
            self,
            epoch,
            proposed_route_id,
            selected,
            manifest,
            candidate_set,
        )

    def _select_model_placement_candidate(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        observed_at_us: int | None = None,
        *,
        selection_mode: str = "energy-aware",
        runtime_rejections: Mapping[str, str] | None = None,
        route_compiler: AutomatedRouteCompiler | None = None,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> tuple[
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
    ]:
        """Compare prospective epochs without making them dispatchable."""
        return _placement._select_model_placement_candidate(
            self,
            candidate_set,
            request,
            observed_at_us,
            selection_mode=selection_mode,
            runtime_rejections=runtime_rejections,
            route_compiler=route_compiler,
            snapshot=snapshot,
        )

    def _materialize_model_placement_candidate(
        self,
        *,
        candidate_set: AutomatedCandidateSet,
        prospective: AutomatedRouteCandidate,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        selection_mode: str,
        invalidation_reason: str,
        runtime_rejections: Mapping[str, str],
        current_epoch: RuntimeModelPlacementEpoch | None,
        demand_snapshot: ModelDemandSnapshot,
        placement_action: ModelPlacementAction,
        residency_holds: Mapping[str, object],
        route_compiler: AutomatedRouteCompiler | None = None,
        memory_source_snapshot: HeterogeneousRuntimeSnapshot | None = None,
        memory_not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
        memory_projection_token: (
            RuntimeResidencyProjectionToken | None
        ) = None,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
        RuntimeModelPlacementEpoch | None,
        RuntimeRouteTemplateSet | None,
    ]:
        return _materialization._materialize_model_placement_candidate(
            self,
            candidate_set=candidate_set,
            prospective=prospective,
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            invalidation_reason=invalidation_reason,
            runtime_rejections=runtime_rejections,
            current_epoch=current_epoch,
            demand_snapshot=demand_snapshot,
            placement_action=placement_action,
            residency_holds=residency_holds,
            route_compiler=route_compiler,
            memory_source_snapshot=memory_source_snapshot,
            memory_not_before_by_resource=memory_not_before_by_resource,
            memory_projection_token=memory_projection_token,
        )

    def _model_placement_pass(
        self,
        context: _ModelPlacementMaterializeContext,
        source: AutomatedCandidateSet,
        epoch: RuntimeModelPlacementEpoch,
        templates: RuntimeRouteTemplateSet,
        resolution_passes: list,
    ) -> _ModelPlacementPass:
        """Materialize one epoch, record its resolution pass, check compatibility."""
        return _materialization._model_placement_pass(
            self,
            context,
            source,
            epoch,
            templates,
            resolution_passes,
        )

    def _retained_model_placement_epoch(
        self,
        current_epoch: RuntimeModelPlacementEpoch | None,
        prospective: AutomatedRouteCandidate,
        proposed_epoch: RuntimeModelPlacementEpoch,
        proposed_templates: RuntimeRouteTemplateSet,
    ) -> tuple[
        RuntimeModelPlacementEpoch,
        RuntimeRouteTemplateSet,
        RuntimeModelPlacementEpoch | None,
        RuntimeRouteTemplateSet | None,
    ]:
        """Keep the current epoch when the proposal re-selects its route."""
        return _materialization._retained_model_placement_epoch(
            self,
            current_epoch,
            prospective,
            proposed_epoch,
            proposed_templates,
        )

    def _rematerialize_epoch_candidates(
        self,
        context: _ModelPlacementMaterializeContext,
        selected_epoch: RuntimeModelPlacementEpoch,
        selected_templates: RuntimeRouteTemplateSet,
    ) -> AutomatedCandidateSet:
        return _materialization._rematerialize_epoch_candidates(
            self,
            context,
            selected_epoch,
            selected_templates,
        )

    def _materialize_epoch_candidates(
        self,
        context: _ModelPlacementMaterializeContext,
        source: AutomatedCandidateSet,
        selected_epoch: RuntimeModelPlacementEpoch,
        selected_templates: RuntimeRouteTemplateSet,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
        Mapping[str, str],
    ]:
        return _materialization._materialize_epoch_candidates(
            self,
            context,
            source,
            selected_epoch,
            selected_templates,
        )

    def _desktop_fallback_after_nonconvergence(
        self,
        context: _ModelPlacementMaterializeContext,
        second_candidates: AutomatedCandidateSet,
        second_memory_rejections: Mapping[str, str],
        resolution_passes: list,
        current_epoch: RuntimeModelPlacementEpoch | None,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
        RuntimeModelPlacementEpoch | None,
        RuntimeRouteTemplateSet | None,
    ]:
        return _materialization._desktop_fallback_after_nonconvergence(
            self,
            context,
            second_candidates,
            second_memory_rejections,
            resolution_passes,
            current_epoch,
        )

    def _resolved_model_placement_result(
        self,
        candidates: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        rejected: tuple[tuple[str, str], ...],
        reason: str,
        resolution_passes: list,
        *,
        outcome: str,
        invalidated_epoch_sha256: str | None,
        publish_epoch: RuntimeModelPlacementEpoch | None,
        publish_templates: RuntimeRouteTemplateSet | None,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
        RuntimeModelPlacementEpoch | None,
        RuntimeRouteTemplateSet | None,
    ]:
        return _materialization._resolved_model_placement_result(
            self,
            candidates,
            selected,
            rejected,
            reason,
            resolution_passes,
            outcome=outcome,
            invalidated_epoch_sha256=invalidated_epoch_sha256,
            publish_epoch=publish_epoch,
            publish_templates=publish_templates,
        )

    def _calibration_model_placement_publication(
        self,
        *,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        invalidation_reason: str,
        runtime_rejections: Mapping[str, str],
        current_epoch: RuntimeModelPlacementEpoch | None,
        demand_snapshot: ModelDemandSnapshot,
        placement_action: ModelPlacementAction,
        route_compiler: AutomatedRouteCompiler | None = None,
    ) -> tuple[
        AutomatedCandidateSet,
        RuntimeModelPlacementEpoch,
        RuntimeRouteTemplateSet,
    ]:
        """Publish a safe epoch without constraining calibration dispatch."""
        return _materialization._calibration_model_placement_publication(
            self,
            candidate_set=candidate_set,
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            invalidation_reason=invalidation_reason,
            runtime_rejections=runtime_rejections,
            current_epoch=current_epoch,
            demand_snapshot=demand_snapshot,
            placement_action=placement_action,
            route_compiler=route_compiler,
        )

    def _desktop_with_async_phone_helper(
        self,
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        rejected: tuple[tuple[str, str], ...],
        reason: str,
        request: Request,
        manifest: ModelManifest,
        selection_mode: str,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
    ]:
        """Bind a profitable phone envelope without delaying its desktop parent."""
        return _helpers._desktop_with_async_phone_helper(
            self,
            candidate_set,
            selected,
            rejected,
            reason,
            request,
            manifest,
            selection_mode,
        )

    @staticmethod
    def _defer_unready_phone_selection(
        candidate_set: AutomatedCandidateSet,
        baseline: AutomatedRouteCandidate,
        selected: AutomatedRouteCandidate,
        rejected: tuple[tuple[str, str], ...],
        reason: str,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
    ]:
        return _helpers._defer_unready_phone_selection(
            candidate_set,
            baseline,
            selected,
            rejected,
            reason,
        )

    def _retained_request_helper_for_baseline(
        self,
        request: Request,
        manifest: ModelManifest,
        baseline: AutomatedRouteCandidate,
    ) -> RuntimeHelperExecutionEnvelope | None:
        """Find the one prepared helper still bound to this desktop parent."""
        return _helpers._retained_request_helper_for_baseline(self, request, manifest, baseline)

    @staticmethod
    def _required_dormant_phone_ffn_keys(transport: object) -> set[str]:
        return _dormant._required_dormant_phone_ffn_keys(transport)

    def _dormant_phone_ffn_storage_superset(
        self,
        parameters: Mapping[str, object],
        desktop: RuntimeExecutionPlan,
        manifest: ModelManifest,
    ) -> dict[str, object]:
        """Allow later READY slices without changing the desktop launch."""
        return _dormant._dormant_phone_ffn_storage_superset(self, parameters, desktop, manifest)

    def _complete_dormant_phone_ffn_parameters(
        self,
        *parameter_sets: Mapping[str, object],
    ) -> dict[str, object] | None:
        return _dormant._complete_dormant_phone_ffn_parameters(self, *parameter_sets)

    @staticmethod
    def _dormant_phone_ffn_parent_parameters(
        candidate_set: AutomatedCandidateSet,
        baseline: AutomatedRouteCandidate,
        manifest: ModelManifest,
        qualified_executor_ids: frozenset[str] | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """Enable zero-assistance control without authorizing phone execution."""
        return _dormant._dormant_phone_ffn_parent_parameters(
            candidate_set, baseline, manifest, qualified_executor_ids
        )

    @staticmethod
    def _dormant_helper_runtime_unavailable(
        candidate_set: AutomatedCandidateSet,
        baseline: AutomatedRouteCandidate,
        rejected: tuple[tuple[str, str], ...],
        envelope: AutomatedRouteCandidate | None,
        retained_helper: RuntimeHelperExecutionEnvelope | None,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
    ]:
        return _dormant._dormant_helper_runtime_unavailable(
            candidate_set,
            baseline,
            rejected,
            envelope,
            retained_helper,
        )

    @staticmethod
    def _baseline_with_dormant_phone_ffn(
        candidate_set: AutomatedCandidateSet,
        baseline: AutomatedRouteCandidate,
        selected: AutomatedRouteCandidate,
        encoded_dormant: str,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        AutomatedRouteCandidate,
    ]:
        return _dormant._baseline_with_dormant_phone_ffn(
            candidate_set,
            baseline,
            selected,
            encoded_dormant,
        )

    @staticmethod
    def _async_helper_layout_matches(
        envelope: AutomatedRouteCandidate,
        baseline: AutomatedRouteCandidate,
        layout,
        manifest: ModelManifest,
    ) -> bool:
        return _attachment._async_helper_layout_matches(envelope, baseline, layout, manifest)

    def _async_helper_preparation_authorized(
        self,
        baseline: AutomatedRouteCandidate,
        envelope: AutomatedRouteCandidate,
        selected: AutomatedRouteCandidate,
        layout,
        manifest: ModelManifest,
    ) -> bool:
        return _attachment._async_helper_preparation_authorized(
            self,
            baseline,
            envelope,
            selected,
            layout,
            manifest,
        )

    def _bind_async_phone_helper(
        self,
        candidate_set: AutomatedCandidateSet,
        baseline: AutomatedRouteCandidate,
        selected: AutomatedRouteCandidate,
        rejected: tuple[tuple[str, str], ...],
        envelope: AutomatedRouteCandidate,
        layout,
        manifest: ModelManifest,
    ) -> tuple[
        AutomatedCandidateSet,
        AutomatedRouteCandidate,
        tuple[tuple[str, str], ...],
        str,
    ]:
        return _attachment._bind_async_phone_helper(
            self,
            candidate_set,
            baseline,
            selected,
            rejected,
            envelope,
            layout,
            manifest,
        )

    @staticmethod
    def _has_dormant_phone_ffn_runtime(
        plan: RuntimeExecutionPlan,
    ) -> bool:
        return _attachment._has_dormant_phone_ffn_runtime(plan)

    @staticmethod
    def _dormant_phone_ffn_runtime_supports(
        resident_plan: RuntimeExecutionPlan,
        requested_contract: str,
    ) -> bool:
        """Check that a live server's dormant FFN runtime is a superset."""
        return _attachment._dormant_phone_ffn_runtime_supports(resident_plan, requested_contract)

    def _compact_helper_opportunities(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
    ) -> tuple[HelperOpportunity, ...]:
        """Retain only the executable helper permission needed at runtime."""
        return _attachment._compact_helper_opportunities(self, candidate_set, request)

    def _select_automated_candidate(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        observed_at_us: int | None = None,
        excluded_route_ids: Sequence[str] = (),
        selection_mode: str = "energy-aware",
        runtime_rejections: Mapping[str, str] | None = None,
        route_compiler: AutomatedRouteCompiler | None = None,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _adaptive._select_automated_candidate(
            self,
            candidate_set,
            request,
            observed_at_us,
            excluded_route_ids,
            selection_mode,
            runtime_rejections,
            route_compiler,
            snapshot=snapshot,
        )

    def _reject_adaptive_routes_without_contract(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        runtime_rejection: dict[str, str],
    ) -> None:
        return _adaptive._reject_adaptive_routes_without_contract(
            self,
            candidate_set,
            request,
            runtime_rejection,
        )

    def _candidate_quarantine_reason(
        self,
        row: AutomatedRouteCandidate,
    ) -> str | None:
        return _adaptive._candidate_quarantine_reason(self, row)

    def _assumed_phone_energy_allowed(
        self,
        row: AutomatedRouteCandidate,
    ) -> bool:
        return _adaptive._assumed_phone_energy_allowed(self, row)

    def _adaptive_exploration_budget_exceeded(
        self,
        row: AutomatedRouteCandidate,
        baseline: AutomatedRouteCandidate,
    ) -> bool:
        return _adaptive._adaptive_exploration_budget_exceeded(self, row, baseline)

    def _adaptive_eligible_policies(
        self,
        context: _AutomatedSelectionContext,
        policies,
        envelope: AutomatedRouteCandidate | None,
        rejection: dict[str, str],
    ) -> list:
        return _adaptive._adaptive_eligible_policies(self, context, policies, envelope, rejection)

    def _select_adaptive_decode_candidate(
        self,
        context: _AutomatedSelectionContext,
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _adaptive._select_adaptive_decode_candidate(self, context)

    def _calibration_physically_qualified(
        self,
        row: AutomatedRouteCandidate,
    ) -> bool:
        return _calibration._calibration_physically_qualified(self, row)

    def _calibration_alternatives(
        self,
        context: _AutomatedSelectionContext,
        rejection: dict[str, str],
    ) -> list[AutomatedRouteCandidate]:
        return _calibration._calibration_alternatives(self, context, rejection)

    def _calibration_coverage_bytes(
        self,
        row: AutomatedRouteCandidate,
        baseline_devices: frozenset[str],
    ) -> int:
        return _calibration._calibration_coverage_bytes(self, row, baseline_devices)

    def _calibration_mechanism_rank(
        self,
        row: AutomatedRouteCandidate,
    ) -> int:
        return _calibration._calibration_mechanism_rank(self, row)

    def _rank_calibration_alternatives(
        self,
        context: _AutomatedSelectionContext,
        alternatives: list[AutomatedRouteCandidate],
    ) -> tuple[AutomatedRouteCandidate, str]:
        return _calibration._rank_calibration_alternatives(self, context, alternatives)

    def _select_calibration_candidate(
        self,
        context: _AutomatedSelectionContext,
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _calibration._select_calibration_candidate(self, context)

    def _objective_assistance_summaries(self, context):
        """Assistance summaries computed once per selection context."""
        adaptive = getattr(self, "_adaptive_decode", None)
        if adaptive is None:
            return ()
        cached = getattr(self, "_objective_assistance_cache", None)
        if cached is not None and cached[0]() is context:
            return cached[1]
        summaries = adaptive.assistance_summaries()
        self._objective_assistance_cache = (weakref.ref(context), summaries)
        return summaries

    def _protected_assistance_loss_upper_uj(self, request, row, summaries=None) -> int | None:
        """Upper bound on phone assistance other active requests could lose.

        Zero when no other request is currently assisted; None when an
        assisted request lacks operational baseline/assisted bounds, so the
        loss is unknown rather than assumed small.
        """
        if summaries is None:
            adaptive = getattr(self, "_adaptive_decode", None)
            if adaptive is None:
                return 0
            summaries = adaptive.assistance_summaries()
        total = 0
        service_upper_us = int(row.cost.service_upper_us)
        for summary in summaries:
            if summary["request_id"] == request.request_id or not summary["assisted"]:
                continue
            loss = summary["loss_upper_per_token_uj"]
            remaining = summary["remaining_tokens"]
            latency = summary["assisted_latency_per_token_us"]
            if loss is None or remaining is None:
                return None
            overlap_tokens = remaining
            if latency:
                overlap_tokens = min(remaining, -(-service_upper_us // latency))
            total += loss * overlap_tokens
        return total

    def _objective_row_rejection(
        self,
        context: _AutomatedSelectionContext,
        row: AutomatedRouteCandidate,
    ) -> str | None:
        """Reject a non-baseline row before objective ranking; None keeps it."""
        return _objectives._objective_row_rejection(self, context, row)

    def _paired_baseline_rejection(
        self,
        context: _AutomatedSelectionContext,
        row: AutomatedRouteCandidate,
    ) -> str | None:
        return _objectives._paired_baseline_rejection(self, context, row)

    def _objective_latency_rejection(
        self,
        context: _AutomatedSelectionContext,
        row: AutomatedRouteCandidate,
        baseline_tardy: bool,
    ) -> str | None:
        return _objectives._objective_latency_rejection(self, context, row, baseline_tardy)

    def _pick_objective_candidate(
        self,
        context: _AutomatedSelectionContext,
        alternatives: list[AutomatedRouteCandidate],
        effective_energy_upper_by_route: Mapping[str, int],
    ) -> tuple[AutomatedRouteCandidate, str]:
        return _objectives._pick_objective_candidate(
            self,
            context,
            alternatives,
            effective_energy_upper_by_route,
        )

    def _select_objective_candidate(
        self,
        context: _AutomatedSelectionContext,
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _objectives._select_objective_candidate(self, context)

    def _select_automated_recovery_fallback(
        self,
        candidate_set: AutomatedCandidateSet,
        *,
        failed_route_id: str,
        runtime_rejections: Mapping[str, str],
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _objectives._select_automated_recovery_fallback(
            self,
            candidate_set,
            failed_route_id=failed_route_id,
            runtime_rejections=runtime_rejections,
        )

    def _select_adaptive_desktop_recovery(
        self,
        candidate_set: AutomatedCandidateSet,
        *,
        failed_route_id: str,
        runtime_rejections: Mapping[str, str],
        exited_executor_id: str | None = None,
        masked_executor_id: str | None = None,
    ) -> tuple[AutomatedRouteCandidate, tuple[tuple[str, str], ...], str]:
        return _objectives._select_adaptive_desktop_recovery(
            self,
            candidate_set,
            failed_route_id=failed_route_id,
            runtime_rejections=runtime_rejections,
            **({} if exited_executor_id is None else {"exited_executor_id": exited_executor_id}),
            **({} if masked_executor_id is None else {"masked_executor_id": masked_executor_id}),
        )

    @staticmethod
    def _runtime_memory_rejection(error: RuntimeResourceError) -> str:
        return _resources._runtime_memory_rejection(error)

    def _runtime_memory_rejections(
        self,
        candidate_set: AutomatedCandidateSet,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        exclude_owner_id: str | None = None,
        request: Request | None = None,
        source_snapshot: HeterogeneousRuntimeSnapshot | None = None,
        observed_at_us: int | None = None,
        not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
        projection_token: RuntimeResidencyProjectionToken | None = None,
    ) -> Mapping[str, str]:
        return _resources._runtime_memory_rejections(
            self,
            candidate_set,
            snapshot,
            exclude_owner_id=exclude_owner_id,
            request=request,
            source_snapshot=source_snapshot,
            observed_at_us=observed_at_us,
            not_before_by_resource=not_before_by_resource,
            projection_token=projection_token,
        )

    def _preview_automated_resources(
        self,
        candidate: AutomatedRouteCandidate,
        *,
        observed_at_us: int | None = None,
    ) -> LeasePreview:
        return _resources._preview_automated_resources(
            self,
            candidate,
            observed_at_us=observed_at_us,
        )

    @staticmethod
    def _automated_prediction_finish_upper_us(
        candidate: AutomatedRouteCandidate,
        preview: LeasePreview,
    ) -> int:
        return _resources._automated_prediction_finish_upper_us(candidate, preview)

    @staticmethod
    def _causal_candidate_observed_at(
        candidate: AutomatedRouteCandidate,
        observed_at_us: int,
        not_before_by_resource: Mapping[str, int],
    ) -> int:
        return _resources._causal_candidate_observed_at(
            candidate,
            observed_at_us,
            not_before_by_resource,
        )

    def _exclusive_transition_barriers(
        self, plan: RuntimeExecutionPlan
    ) -> Mapping[str, int]:
        return _resources._exclusive_transition_barriers(self, plan)

    def _runtime_plan_not_before_by_resource(
        self, plan: RuntimeExecutionPlan
    ) -> Mapping[str, int]:
        return _resources._runtime_plan_not_before_by_resource(self, plan)

    def _runtime_exclusive_memory_resources(self) -> Mapping[str | tuple[str, str], str]:
        return _resources._runtime_exclusive_memory_resources(self)

    def _runtime_residency_order_barrier(
        self, plan: RuntimeExecutionPlan
    ) -> bool:
        return _resources._runtime_residency_order_barrier(self, plan)

    def _runtime_residency_hysteresis_key(
        self, plan: RuntimeExecutionPlan, artifact_sha256: str
    ) -> str | None:
        return _resources._runtime_residency_hysteresis_key(
            self, plan, artifact_sha256
        )

    def _preview_automated_memory(
        self,
        plan: RuntimeExecutionPlan,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        start_us: int,
        reserved_until_us: int,
        exclude_owner_id: str | None = None,
        enforce_live_capacity: bool = True,
    ) -> Mapping[str, int]:
        return _resources._preview_automated_memory(
            self,
            plan,
            snapshot,
            start_us=start_us,
            reserved_until_us=reserved_until_us,
            exclude_owner_id=exclude_owner_id,
            enforce_live_capacity=enforce_live_capacity,
        )
