"""Model placement epochs: demand snapshots, proposals, publication, refresh after learning.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .._internal.policy import Request
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.route_generation import AutomatedRouteCompiler, RuntimeRouteTemplateSet
from .._internal.model_placement_controller import ModelDemandSnapshot, ModelPlacementAction
from .._internal.runtime_plan import AutomatedCandidateSet, AutomatedRouteCandidate
from .._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from .._internal.runtime_controller import RuntimeRequestTicket

from .placement_epochs_ops.common import (
    _PlacementChoice as _PlacementChoice,
    _PlacementPublicationBasis as _PlacementPublicationBasis,
    _PlacementRefreshDemand as _PlacementRefreshDemand,
    _PlacementRefreshResult as _PlacementRefreshResult,
    _PlacementTicketDemand as _PlacementTicketDemand,
)
from .placement_epochs_ops import demand as _demand
from .placement_epochs_ops import frontier as _frontier
from .placement_epochs_ops import lookup as _lookup
from .placement_epochs_ops import publication as _publication
from .placement_epochs_ops import refresh as _refresh
from .placement_epochs_ops import selection as _selection


class PlacementEpochMixin:
    """Model placement epochs: demand snapshots, proposals, publication, refresh after learning."""

    def model_placement_epoch_stats(self) -> Mapping[str, object]:
        result = {
            "background_refresh_candidates": (
                self._runtime_epoch_background_refresh_candidates
            ),
            "background_refresh_failures": (
                self._runtime_epoch_background_refresh_failures
            ),
            "background_refresh_total_us": (
                self._runtime_epoch_background_refresh_us
            ),
            "background_refreshes": (
                self._runtime_epoch_background_refreshes
            ),
            "entries": len(self._runtime_route_template_sets),
            "evictions": self._runtime_route_template_cache_evictions,
            "hits": self._runtime_route_template_cache_hits,
            "invalidations": sum(
                self._runtime_epoch_invalidations.values()
            ),
            "invalidation_reasons": dict(sorted(
                self._runtime_epoch_invalidations.items()
            )),
            "misses": self._runtime_route_template_cache_misses,
        }
        if self._runtime_epoch_background_refresh_failure_reasons:
            result["background_refresh_failure_reasons"] = dict(sorted(
                self._runtime_epoch_background_refresh_failure_reasons.items()
            ))
        return MappingProxyType(result)

    def model_placement_controller_stats(self) -> Mapping[str, object]:
        return self._model_placement_controller.stats()

    def model_placement_events(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        return self._model_placement_controller.events()

    def _placement_ticket_demand(
        self,
        request: Request,
        manifest: ModelManifest,
        observed_at_us: int,
    ) -> _PlacementTicketDemand:
        return _demand._placement_ticket_demand(self, request, manifest, observed_at_us)

    def _current_placement_component(
        self,
        manifest: ModelManifest,
        active: tuple[RuntimeRequestTicket, ...],
        confirmed_components: Mapping[str, object],
        epoch: RuntimeModelPlacementEpoch | None,
    ) -> str | None:
        return _demand._current_placement_component(
            self,
            manifest,
            active,
            confirmed_components,
            epoch,
        )

    def _placement_available_resources(
        self,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return _demand._placement_available_resources(self, snapshot)

    def _placement_demand_generations(
        self,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        all_tickets: tuple[RuntimeRequestTicket, ...],
        confirmed_components: Mapping[str, object],
    ) -> tuple[str, str, str]:
        return _demand._placement_demand_generations(
            self,
            manifest,
            snapshot,
            all_tickets,
            confirmed_components,
        )

    def _model_demand_snapshot(
        self,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        epoch: RuntimeModelPlacementEpoch | None = None,
    ) -> ModelDemandSnapshot:
        return _demand._model_demand_snapshot(
            self,
            request,
            manifest,
            snapshot,
            observed_at_us,
            epoch,
        )

    def _evaluate_model_placement(
        self,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        epoch: RuntimeModelPlacementEpoch | None,
        selection_mode: str,
    ) -> tuple[ModelDemandSnapshot, ModelPlacementAction]:
        return _demand._evaluate_model_placement(
            self,
            request,
            manifest,
            snapshot,
            observed_at_us,
            epoch,
            selection_mode,
        )

    def _record_epoch_refresh_failure(self, reason: object) -> None:
        return _lookup._record_epoch_refresh_failure(self, reason)

    def _record_epoch_invalidation(self, reason: str) -> None:
        return _lookup._record_epoch_invalidation(self, reason)

    def _published_epoch_for_request(
        self,
        request: Request,
        manifest: ModelManifest,
        observed_at_us: int,
        selection_mode: str,
    ) -> tuple[
        RuntimeModelPlacementEpoch | None,
        RuntimeRouteTemplateSet | None,
        str,
    ]:
        return _lookup._published_epoch_for_request(
            self,
            request,
            manifest,
            observed_at_us,
            selection_mode,
        )

    def _route_template_phone_helper_route_ids(
        self,
        templates: RuntimeRouteTemplateSet,
        manifest: ModelManifest,
    ) -> tuple[str, ...]:
        return _lookup._route_template_phone_helper_route_ids(self, templates, manifest)

    def _compile_placement_templates(
        self,
        compiler: AutomatedRouteCompiler,
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        manifest: ModelManifest,
        request: Request,
        snapshot: HeterogeneousRuntimeSnapshot,
        input_bucket: int,
        output_bucket: int,
    ) -> RuntimeRouteTemplateSet:
        return _lookup._compile_placement_templates(
            self,
            compiler,
            candidate_set,
            selected,
            manifest,
            request,
            snapshot,
            input_bucket,
            output_bucket,
        )

    @staticmethod
    def _placement_transition_cost(
        selected: AutomatedRouteCandidate,
    ) -> tuple[int, int]:
        return _lookup._placement_transition_cost(selected)

    def _initial_placement_choice(
        self,
        *,
        selected: AutomatedRouteCandidate,
        templates: RuntimeRouteTemplateSet,
        candidate_set: AutomatedCandidateSet,
        demand: ModelDemandSnapshot,
        placement_action: ModelPlacementAction | None,
        invalidation_reason: str,
    ) -> tuple[_PlacementChoice, _PlacementPublicationBasis]:
        return _selection._initial_placement_choice(
            self,
            selected=selected,
            templates=templates,
            candidate_set=candidate_set,
            demand=demand,
            placement_action=placement_action,
            invalidation_reason=invalidation_reason,
        )

    def _placement_publication_action(
        self,
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
        return _selection._placement_publication_action(
            self,
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

    def _retained_placement_choice(
        self,
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
        return _selection._retained_placement_choice(
            self,
            choice=choice,
            selected=selected,
            compiler=compiler,
            candidate_set=candidate_set,
            manifest=manifest,
            request=request,
            snapshot=snapshot,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
        )

    def _authorize_placement_choice(
        self,
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
        return _selection._authorize_placement_choice(
            self,
            choice=choice,
            basis=basis,
            compiler=compiler,
            candidate_set=candidate_set,
            manifest=manifest,
            request=request,
            snapshot=snapshot,
            demand=demand,
            current_epoch=current_epoch,
            selection_mode=selection_mode,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
        )

    def _authoritative_phone_placement_choice(
        self,
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
        return _selection._authoritative_phone_placement_choice(
            self,
            choice=choice,
            compiler=compiler,
            candidate_set=candidate_set,
            manifest=manifest,
            request=request,
            snapshot=snapshot,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
        )

    def _placement_contract_identity(
        self,
        choice: _PlacementChoice,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
    ):
        return _publication._placement_contract_identity(self, choice, manifest, snapshot)

    def _build_model_placement_epoch(
        self,
        *,
        request: Request,
        manifest: ModelManifest,
        observed_at_us: int,
        selection_mode: str,
        invalidation_reason: str,
        demand: ModelDemandSnapshot,
        statistics,
        choice: _PlacementChoice,
        validity_us: int,
        fractions: tuple[int, ...],
        component,
        phone_layout_generation: int | None,
        input_bucket: int,
        output_bucket: int,
        placement_learning_generation_sha256: str | None,
    ) -> RuntimeModelPlacementEpoch:
        return _publication._build_model_placement_epoch(
            self,
            request=request,
            manifest=manifest,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            invalidation_reason=invalidation_reason,
            demand=demand,
            statistics=statistics,
            choice=choice,
            validity_us=validity_us,
            fractions=fractions,
            component=component,
            phone_layout_generation=phone_layout_generation,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            placement_learning_generation_sha256=placement_learning_generation_sha256,
        )

    def _propose_model_placement_epoch(
        self,
        *,
        request: Request,
        manifest: ModelManifest,
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        observed_at_us: int,
        selection_mode: str,
        invalidation_reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        route_compiler: AutomatedRouteCompiler | None = None,
        demand_snapshot: ModelDemandSnapshot | None = None,
        placement_action: ModelPlacementAction | None = None,
        current_epoch: RuntimeModelPlacementEpoch | None = None,
        placement_learning_generation_sha256: str | None = None,
    ) -> tuple[RuntimeModelPlacementEpoch, RuntimeRouteTemplateSet]:
        return _publication._propose_model_placement_epoch(
            self,
            request=request,
            manifest=manifest,
            candidate_set=candidate_set,
            selected=selected,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            invalidation_reason=invalidation_reason,
            snapshot=snapshot,
            route_compiler=route_compiler,
            demand_snapshot=demand_snapshot,
            placement_action=placement_action,
            current_epoch=current_epoch,
            placement_learning_generation_sha256=placement_learning_generation_sha256,
        )

    @staticmethod
    def _candidate_set_with_epoch(
        candidate_set: AutomatedCandidateSet,
        epoch: RuntimeModelPlacementEpoch,
        *,
        fast_path: bool,
        invalidation_reason: str,
    ) -> AutomatedCandidateSet:
        return _publication._candidate_set_with_epoch(
            candidate_set,
            epoch,
            fast_path=fast_path,
            invalidation_reason=invalidation_reason,
        )

    def _publish_model_placement_epoch(
        self,
        epoch: RuntimeModelPlacementEpoch,
        templates: RuntimeRouteTemplateSet,
    ) -> None:
        return _publication._publish_model_placement_epoch(self, epoch, templates)

    def _refresh_model_placement_epochs_after_learning(
        self,
        request_ids: Sequence[str],
        observed_at_us: int,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ] | None,
    ) -> None:
        """Coalesce model placement checks away from request completion."""
        return _refresh._refresh_model_placement_epochs_after_learning(
            self,
            request_ids,
            observed_at_us,
            snapshot_provider,
        )

    def _placement_refresh_request_ids(
        self,
        request_ids: Sequence[str],
        artifact_sha256: str | None,
    ) -> tuple[str, ...]:
        return _refresh._placement_refresh_request_ids(self, request_ids, artifact_sha256)

    def _placement_refresh_demand(
        self,
        request_id: str,
        request_ids: tuple[str, ...],
        observed_at_us: int,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ],
        seen_epochs: set[str],
    ) -> _PlacementRefreshDemand | None:
        return _refresh._placement_refresh_demand(
            self,
            request_id,
            request_ids,
            observed_at_us,
            snapshot_provider,
            seen_epochs,
        )

    def _collect_placement_refresh_demands(
        self,
        request_ids: tuple[str, ...],
        observed_at_us: int,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ],
    ) -> tuple[_PlacementRefreshDemand, ...]:
        return _refresh._collect_placement_refresh_demands(
            self,
            request_ids,
            observed_at_us,
            snapshot_provider,
        )

    @staticmethod
    def _placement_refresh_generations(scheduler) -> tuple[object, ...]:
        return _refresh._placement_refresh_generations(scheduler)

    @staticmethod
    def _rerank_placement_refresh_demands(
        compiler: AutomatedRouteCompiler,
        observation_snapshot,
        demands: tuple[_PlacementRefreshDemand, ...],
    ) -> tuple[_PlacementRefreshResult, ...]:
        return _refresh._rerank_placement_refresh_demands(compiler, observation_snapshot, demands)

    def _publish_placement_refresh(
        self,
        demand: _PlacementRefreshDemand,
        candidate_set: AutomatedCandidateSet,
        compiler: AutomatedRouteCompiler,
        observation_generation_sha256: str,
    ) -> int | None:
        return _refresh._publish_placement_refresh(
            self,
            demand,
            candidate_set,
            compiler,
            observation_generation_sha256,
        )

    def _publish_placement_refresh_results(
        self,
        results: tuple[_PlacementRefreshResult, ...],
        compiler: AutomatedRouteCompiler,
        target_generations: tuple[object, ...],
    ) -> None:
        return _refresh._publish_placement_refresh_results(
            self,
            results,
            compiler,
            target_generations,
        )

    def _run_model_placement_epoch_refresh_after_learning(
        self,
        request_ids: Sequence[str],
        observed_at_us: int,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ],
        *,
        artifact_sha256: str | None = None,
    ) -> None:
        return _refresh._run_model_placement_epoch_refresh_after_learning(
            self,
            request_ids,
            observed_at_us,
            snapshot_provider,
            artifact_sha256=artifact_sha256,
        )

    def _frontier_resource_generation(
        self,
        demand: ModelDemandSnapshot,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> str:
        return _frontier._frontier_resource_generation(self, demand, snapshot)

    def _lookup_prepared_frontier(
        self,
        planner,
        key,
        artifact_sha256: str,
        observed_at_us: int,
    ):
        return _frontier._lookup_prepared_frontier(
            self,
            planner,
            key,
            artifact_sha256,
            observed_at_us,
        )

    def _placement_frontier_reasons(
        self,
        demand_id: tuple[object, ...],
        previous,
        key,
        resource_state: tuple[object, ...],
    ) -> tuple[str, ...]:
        return _frontier._placement_frontier_reasons(self, demand_id, previous, key, resource_state)

    def _prepared_placement_frontier(
        self,
        request: Request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        *,
        demand_snapshot: ModelDemandSnapshot | None = None,
    ):
        return _frontier._prepared_placement_frontier(
            self,
            request,
            manifest,
            snapshot,
            observed_at_us,
            demand_snapshot=demand_snapshot,
        )

    @staticmethod
    def _arrived_decode_work_by_artifact(
        active_remaining_tokens_by_artifact: Mapping[str, int],
        queued_output_tokens_by_artifact: Mapping[str, int],
    ) -> Mapping[str, int]:
        return _frontier._arrived_decode_work_by_artifact(
            active_remaining_tokens_by_artifact,
            queued_output_tokens_by_artifact,
        )

    def _persistent_phone_service_reserve_by_artifact(
        self,
        phone_device_id: str,
        snapshot: HeterogeneousRuntimeSnapshot | None,
        *, include_observed: bool = False,
    ) -> Mapping[str, int]:
        """Reserve the declared peak footprint for admitted persistent services."""
        return _frontier._persistent_phone_service_reserve_by_artifact(
            self,
            phone_device_id,
            snapshot,
            include_observed=include_observed,
        )
