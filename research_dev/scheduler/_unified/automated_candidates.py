"""Automated candidate generation, legacy evidence migration, and adaptive history costs.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from typing import Mapping

from .._internal.policy import Request
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityCatalog
from .._internal.route_generation import AutomatedRouteCompiler
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeExecutionPlan,
)
from .._internal.adaptive_decode_contracts import AdaptiveDecodePolicy
from .common import _runtime_serialized

from .automated_candidates_ops.common import (
    _AdaptiveCandidateApplication as _AdaptiveCandidateApplication,
    _AdaptiveCohortBounds as _AdaptiveCohortBounds,
    _AdaptiveEnergyDeltas as _AdaptiveEnergyDeltas,
    _AdaptiveHistoryContext as _AdaptiveHistoryContext,
    _AdaptiveWarmEstimate as _AdaptiveWarmEstimate,
    _ExactRouteDecompositions as _ExactRouteDecompositions,
    _TransitionEnergyEstimate as _TransitionEnergyEstimate,
)
from .automated_candidates_ops import application as _application
from .automated_candidates_ops import energy as _energy
from .automated_candidates_ops import generation as _generation
from .automated_candidates_ops import history as _history
from .automated_candidates_ops import observations as _observations


class AutomatedCandidateMixin:
    """Automated candidate generation, legacy evidence migration, and adaptive history costs."""

    def _generate_automated_candidate_set(
        self,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        *,
        use_residency_holds: bool = True,
        update_phone_residency_portfolio: bool = True,
        desktop_parent: tuple[str, str] | None = None,
    ) -> AutomatedCandidateSet:
        return _generation._generate_automated_candidate_set(
            self,
            request,
            manifest,
            snapshot,
            observed_at_us,
            use_residency_holds=use_residency_holds,
            update_phone_residency_portfolio=update_phone_residency_portfolio,
            desktop_parent=desktop_parent,
        )

    def _prepare_legacy_evidence_migrations(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest: ModelManifest,
    ) -> None:
        """Migrate compatible legacy evidence once per source identity."""
        return _generation._prepare_legacy_evidence_migrations(
            self,
            candidate_set,
            request,
            manifest,
        )

    def _adaptive_candidate_accepts_history(
        self,
        candidate: AutomatedRouteCandidate,
        baseline: AutomatedRouteCandidate,
        contract: Mapping[str, object] | None,
    ) -> bool:
        return _history._adaptive_candidate_accepts_history(self, candidate, baseline, contract)

    def _assumed_phone_history_query(
        self,
        *,
        manifest: ModelManifest,
        request: Request,
        component_capability_sha256: str,
        baseline_policy: AdaptiveDecodePolicy,
        policies: tuple[AdaptiveDecodePolicy, ...],
        adaptive_config,
        phone_power_by_domain: Mapping[str, tuple[int, int]],
        route_geometry_prior: bool = False,
        subset_source_layer_mask: int | None = None,
    ):
        return _history._assumed_phone_history_query(
            self,
            manifest=manifest,
            request=request,
            component_capability_sha256=component_capability_sha256,
            baseline_policy=baseline_policy,
            policies=policies,
            adaptive_config=adaptive_config,
            phone_power_by_domain=phone_power_by_domain,
            route_geometry_prior=route_geometry_prior,
            subset_source_layer_mask=subset_source_layer_mask,
        )

    def _adaptive_physical_latency(
        self,
        *,
        manifest: ModelManifest,
        request: Request,
        component_capability_sha256: str,
        baseline_policy: AdaptiveDecodePolicy,
        policies: tuple[AdaptiveDecodePolicy, ...],
        selected_policy: AdaptiveDecodePolicy,
        adaptive_config,
    ) -> tuple[tuple[int, int, int] | None, AdaptiveDecodePolicy | None]:
        return _history._adaptive_physical_latency(
            self,
            manifest=manifest,
            request=request,
            component_capability_sha256=component_capability_sha256,
            baseline_policy=baseline_policy,
            policies=policies,
            selected_policy=selected_policy,
            adaptive_config=adaptive_config,
        )

    def _adaptive_history_context(
        self,
        *,
        candidate: AutomatedRouteCandidate,
        baseline_policy: AdaptiveDecodePolicy,
        contract: Mapping[str, object],
        request: Request,
        manifest: ModelManifest,
        compiler: AutomatedRouteCompiler,
    ) -> _AdaptiveHistoryContext:
        return _history._adaptive_history_context(
            self,
            candidate=candidate,
            baseline_policy=baseline_policy,
            contract=contract,
            request=request,
            manifest=manifest,
            compiler=compiler,
        )

    @staticmethod
    def _apply_physical_latency_history(
        candidate: AutomatedRouteCandidate,
        request: Request,
        context: _AdaptiveHistoryContext,
    ) -> AutomatedRouteCandidate | None:
        return _history._apply_physical_latency_history(candidate, request, context)

    @staticmethod
    def _with_measured_transitions(
        compiler: AutomatedRouteCompiler,
        manifest: ModelManifest,
        route: AutomatedRouteCandidate,
    ) -> AutomatedRouteCandidate:
        return _history._with_measured_transitions(compiler, manifest, route)

    @staticmethod
    def _exact_adaptive_route_history(
        compiler: AutomatedRouteCompiler,
        manifest: ModelManifest,
        request: Request,
        cost_features: Mapping[str, int],
        baseline: AutomatedRouteCandidate,
        candidate: AutomatedRouteCandidate,
        history_uses_assumed_phone_power: bool,
    ) -> tuple[object | None, object | None, bool]:
        return _history._exact_adaptive_route_history(
            compiler,
            manifest,
            request,
            cost_features,
            baseline,
            candidate,
            history_uses_assumed_phone_power,
        )

    @staticmethod
    def _adaptive_warm_estimate(
        parent_cost,
        history,
        output_tokens: int,
    ) -> _AdaptiveWarmEstimate:
        return _energy._adaptive_warm_estimate(parent_cost, history, output_tokens)

    @staticmethod
    def _adaptive_transition_energy(
        compiler: AutomatedRouteCompiler,
        manifest: ModelManifest,
        route: AutomatedRouteCandidate,
        default: tuple[int, int, int],
    ) -> _TransitionEnergyEstimate:
        return _energy._adaptive_transition_energy(compiler, manifest, route, default)

    @staticmethod
    def _adaptive_energy_deltas(
        history,
        output_tokens: int,
        candidate_transition: _TransitionEnergyEstimate,
        parent_transition: _TransitionEnergyEstimate,
    ) -> _AdaptiveEnergyDeltas:
        return _energy._adaptive_energy_deltas(
            history,
            output_tokens,
            candidate_transition,
            parent_transition,
        )

    @staticmethod
    def _adaptive_exact_decompositions(
        *,
        exact_route_qualified: bool,
        exact_parent,
        exact_candidate,
        baseline: AutomatedRouteCandidate,
        candidate: AutomatedRouteCandidate,
        warm: _AdaptiveWarmEstimate,
    ) -> _ExactRouteDecompositions:
        return _energy._adaptive_exact_decompositions(
            exact_route_qualified=exact_route_qualified,
            exact_parent=exact_parent,
            exact_candidate=exact_candidate,
            baseline=baseline,
            candidate=candidate,
            warm=warm,
        )

    @staticmethod
    def _adaptive_parent_history_candidate(
        *,
        baseline: AutomatedRouteCandidate,
        history,
        history_context: _AdaptiveHistoryContext,
        warm: _AdaptiveWarmEstimate,
        parent_transition: _TransitionEnergyEstimate,
        exact_decompositions: _ExactRouteDecompositions,
        exact_parent,
    ) -> AutomatedRouteCandidate:
        return _energy._adaptive_parent_history_candidate(
            baseline=baseline,
            history=history,
            history_context=history_context,
            warm=warm,
            parent_transition=parent_transition,
            exact_decompositions=exact_decompositions,
            exact_parent=exact_parent,
        )

    @staticmethod
    def _adaptive_candidate_history_cost(
        *,
        candidate: AutomatedRouteCandidate,
        warm: _AdaptiveWarmEstimate,
        candidate_transition: _TransitionEnergyEstimate,
        exact_decompositions: _ExactRouteDecompositions,
        exact_candidate,
        history_context: _AdaptiveHistoryContext,
    ):
        return _energy._adaptive_candidate_history_cost(
            candidate=candidate,
            warm=warm,
            candidate_transition=candidate_transition,
            exact_decompositions=exact_decompositions,
            exact_candidate=exact_candidate,
            history_context=history_context,
        )

    @staticmethod
    def _adaptive_cohort_bounds(
        *,
        candidate: AutomatedRouteCandidate,
        history_context: _AdaptiveHistoryContext,
        warm: _AdaptiveWarmEstimate,
        candidate_transition: _TransitionEnergyEstimate,
        parent_history_candidate: AutomatedRouteCandidate,
        energy_deltas: _AdaptiveEnergyDeltas,
        exact_decompositions: _ExactRouteDecompositions,
        exact_parent,
        exact_candidate,
    ) -> _AdaptiveCohortBounds:
        return _application._adaptive_cohort_bounds(
            candidate=candidate,
            history_context=history_context,
            warm=warm,
            candidate_transition=candidate_transition,
            parent_history_candidate=parent_history_candidate,
            energy_deltas=energy_deltas,
            exact_decompositions=exact_decompositions,
            exact_parent=exact_parent,
            exact_candidate=exact_candidate,
        )

    @staticmethod
    def _adaptive_break_even(
        *,
        baseline: AutomatedRouteCandidate,
        candidate: AutomatedRouteCandidate,
        history,
        history_context: _AdaptiveHistoryContext,
        warm: _AdaptiveWarmEstimate,
        candidate_transition: _TransitionEnergyEstimate,
        parent_history_candidate: AutomatedRouteCandidate,
        energy_deltas: _AdaptiveEnergyDeltas,
        exact_decompositions: _ExactRouteDecompositions,
        exact_parent,
        exact_candidate,
    ) -> tuple[dict[str, object], bool]:
        return _application._adaptive_break_even(
            AutomatedCandidateMixin,
            baseline=baseline,
            candidate=candidate,
            history=history,
            history_context=history_context,
            warm=warm,
            candidate_transition=candidate_transition,
            parent_history_candidate=parent_history_candidate,
            energy_deltas=energy_deltas,
            exact_decompositions=exact_decompositions,
            exact_parent=exact_parent,
            exact_candidate=exact_candidate,
        )

    @staticmethod
    def _candidate_with_adaptive_history(
        candidate: AutomatedRouteCandidate,
        learned_cost,
        break_even: Mapping[str, object],
        route_energy_qualified: bool,
    ) -> AutomatedRouteCandidate:
        return _application._candidate_with_adaptive_history(
            candidate,
            learned_cost,
            break_even,
            route_energy_qualified,
        )

    def _apply_adaptive_history_candidate(
        self,
        *,
        candidate: AutomatedRouteCandidate,
        baseline: AutomatedRouteCandidate,
        baseline_policy: AdaptiveDecodePolicy,
        contract: Mapping[str, object],
        request: Request,
        manifest: ModelManifest,
        cost_features: Mapping[str, int],
        compiler: AutomatedRouteCompiler,
    ) -> _AdaptiveCandidateApplication:
        return _application._apply_adaptive_history_candidate(
            self,
            candidate=candidate,
            baseline=baseline,
            baseline_policy=baseline_policy,
            contract=contract,
            request=request,
            manifest=manifest,
            cost_features=cost_features,
            compiler=compiler,
        )

    def _apply_adaptive_history_costs(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest: ModelManifest,
        cost_features: Mapping[str, int] | None = None,
    ) -> AutomatedCandidateSet:
        return _application._apply_adaptive_history_costs(
            self,
            candidate_set,
            request,
            manifest,
            cost_features,
        )

    def runtime_residency_cohort_state(self) -> Mapping[str, object]:
        return _observations.runtime_residency_cohort_state(self)

    def automated_observation_state(self) -> Mapping[str, int]:
        return _observations.automated_observation_state(self)

    def automated_observation_snapshot(self) -> Mapping[str, object]:
        return _observations.automated_observation_snapshot(self)

    @_runtime_serialized
    def load_automated_observations(
        self,
        value: object,
        *,
        source_catalog: RuntimeCapabilityCatalog | None = None,
    ) -> None:
        return _observations.load_automated_observations(self, value, source_catalog=source_catalog)

    @_runtime_serialized
    def merge_automated_observations(self, value: object) -> None:
        return _observations.merge_automated_observations(self, value)

    @_runtime_serialized
    def rebind_legacy_automated_component_observations(
        self,
        source_catalog: RuntimeCapabilityCatalog,
        model_id: str,
        source_plan: RuntimeExecutionPlan,
        executor_id: str,
        target_plan: RuntimeExecutionPlan | None = None,
    ) -> int:
        """Migrate explicitly verified execution-component observations."""
        return _observations.rebind_legacy_automated_component_observations(
            self,
            source_catalog,
            model_id,
            source_plan,
            executor_id,
            target_plan,
        )

    def runtime_decision_timings(self) -> tuple[Mapping[str, object], ...]:
        return _observations.runtime_decision_timings(self)
