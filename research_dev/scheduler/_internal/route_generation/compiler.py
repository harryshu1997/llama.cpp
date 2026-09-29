"""``AutomatedRouteCompiler``: construction, legacy observation rebinding, and ``generate``."""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
import time
from types import MappingProxyType
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..placement import (
    HierarchicalPlacementPlanner,
    OperatorNode,
    PlacementError,
    PlacementHardwareProfile,
)
from ..policy import ResourceTimeline
from ..phone_shards import PhoneFfnResidencyLayout
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCapabilityCatalog,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
)
from ..runtime_cost import RuntimeMemoryDemand, RuntimeParticipantBinding
from ..runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeExecutionPlan,
    RuntimeOperatorAssignment,
)
from ..runtime_residency_cohorts import (
    RuntimeResidencyCohortHold,
    RuntimeResidencyReuseProjection,
    runtime_residency_component_identity,
)
from ..runtime_search import (
    BoundedPlacementCompiler,
    RuntimeSearchError,
    request_shape_bucket,
)
from ..runtime_learning import RuntimeLearningError, RuntimeRouteObservationStore
from ..types import canonical_sha256
from .common import (
    RouteGenerationError,
    _FfnResidentEnvelope,
    _PhoneResidencyRouteEvidence,
    _Pattern,
)
from .residency_evidence import PhoneResidencyEvidenceMixin
from .identity import RouteIdentityMixin
from .templates import RouteTemplateMixin
from .patterns import RoutePatternMixin
from .candidates import RouteCandidateMixin
from .feasibility import RouteFeasibilityMixin, ThermalGateLog
from .envelopes import RouteEnvelopeMixin
from .costing import RouteCostingMixin
from .costing_demands import RouteDemandMixin
from .costing_estimates import RouteEstimateMixin
from .costing_parameters import RouteParameterMixin
from .costing_rough import RouteRoughCostMixin
from .remote_resident import RouteRemoteResidentMixin


class AutomatedRouteCompiler(
    PhoneResidencyEvidenceMixin,
    RouteIdentityMixin,
    RouteTemplateMixin,
    RoutePatternMixin,
    RouteCandidateMixin,
    RouteFeasibilityMixin,
    RouteEnvelopeMixin,
    RouteCostingMixin,
    RouteDemandMixin,
    RouteParameterMixin,
    RouteEstimateMixin,
    RouteRoughCostMixin,
    RouteRemoteResidentMixin,
):
    """Generate supported families, cost all rows, then mark domination."""

    def __init__(
        self,
        catalog: RuntimeCapabilityCatalog,
        timeline: ResourceTimeline,
        observation_store: RuntimeRouteObservationStore | None = None,
    ) -> None:
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise RouteGenerationError("runtime capability catalog is invalid")
        if not isinstance(timeline, ResourceTimeline):
            raise RouteGenerationError("resource timeline is invalid")
        self.catalog = catalog
        self.timeline = timeline
        self._capability_generation_sha256 = canonical_sha256(catalog)
        self._pattern_cache: dict[
            tuple[str, str], tuple[_Pattern, ...]
        ] = {}
        self._pattern_by_key_cache: dict[
            tuple[str, str], Mapping[str, _Pattern]
        ] = {}
        self._request_work_cache: dict[
            tuple[str, int, int], ModelRequestWork
        ] = {}
        self._kernel_signature_cache: dict[
            tuple[str, int, int, str], tuple[object, ...]
        ] = {}
        self._operator_assignment_cache: dict[
            tuple[object, ...], tuple[RuntimeOperatorAssignment, ...]
        ] = {}
        self._memory_demand_cache: dict[
            tuple[str, int, int, str], tuple[RuntimeMemoryDemand, ...]
        ] = {}
        self._component_time_cache: dict[
            tuple[str, int, int, str], tuple[int, int]
        ] = {}
        self._rough_memory_cache: dict[tuple[str, str], bool] = {}
        self._rough_cost_cache: dict[
            tuple[object, ...], tuple[int, int]
        ] = {}
        self._rough_group_cache: dict[
            tuple[str, str],
            tuple[tuple[str, str, str | None, int, str | None, int], ...],
        ] = {}
        self._route_capability_identity_cache: dict[
            tuple[object, ...], str
        ] = {}
        self._coordinator_cache: dict[
            str,
            RuntimeExecutorCapability
            | RuntimeCompositeExecutorCapability
            | None,
        ] = {}
        self._participant_binding_cache: dict[
            str, tuple[RuntimeParticipantBinding, ...]
        ] = {}
        self._desktop_placement_hash_cache: dict[
            tuple[str, tuple[tuple[str, str], ...]], str
        ] = {}
        self._ffn_resident_envelope_cache: dict[
            tuple[str, str, str], tuple[_FfnResidentEnvelope, ...]
        ] = {}
        # The scheduler shares one log between its compilers.
        self.thermal_gate_log = ThermalGateLog()
        self._phone_ffn_shard_storage_by_artifact = MappingProxyType({})
        self._phone_ffn_shard_storage_sha256: str | None = None
        self._phone_residency_layout: PhoneFfnResidencyLayout | None = None
        self._phone_residency_route_evidence: dict[
            str, _PhoneResidencyRouteEvidence
        ] = {}
        self._phone_residency_evidence_status: dict[
            str, Mapping[str, object]
        ] = {}
        self._model_id_by_artifact: dict[str, str] = {}
        self._execution_plan_cache: dict[
            tuple[object, ...], RuntimeExecutionPlan
        ] = {}
        self._cost_profile_cache: dict[
            tuple[object, ...], tuple[PlacementHardwareProfile, str]
        ] = {}
        self._executor_id_by_plan_sha256: dict[str, str] = {}
        self._placement_cache: dict[
            tuple[object, ...],
            tuple[
                object,
                str | None,
                tuple[OperatorNode, ...],
                tuple[RuntimeOperatorAssignment, ...],
            ],
        ] = {}
        self._placement_cache_hits = 0
        self._placement_cache_misses = 0
        self._placement_cache_evictions = 0
        self._placement_cache_generations = 0
        self._placement_singleflight_waits = 0
        self._placement_inflight: dict[
            tuple[object, ...], threading.Event
        ] = {}
        self._frontier_singleflight_waits = 0
        self._frontier_generations = 0
        self._frontier_inflight: dict[
            tuple[object, ...], threading.Event
        ] = {}
        self._cache_lock = threading.RLock()
        self._generation_lock = threading.Lock()
        self._static_cache_evictions = 0
        self._rough_compiler = BoundedPlacementCompiler(
            search_budget=32,
            refinement_budget=24,
        )
        self._observation_store = (
            RuntimeRouteObservationStore()
            if observation_store is None
            else observation_store
        )
        self._last_generation_timing: Mapping[str, int] = MappingProxyType({})

    def rebind_legacy_component_observations(
        self,
        source_catalog: RuntimeCapabilityCatalog,
        manifest: ModelManifest,
        source_plan: RuntimeExecutionPlan,
        executor_id: str,
        target_plan: RuntimeExecutionPlan | None = None,
        *,
        latency_only: bool = False,
    ) -> int:
        """Re-key explicitly verified execution-component evidence."""
        if not isinstance(source_catalog, RuntimeCapabilityCatalog):
            raise RouteGenerationError(
                "runtime source capability catalog is invalid"
            )
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError(
                "runtime component manifest is invalid"
            )
        if not isinstance(source_plan, RuntimeExecutionPlan):
            raise RouteGenerationError(
                "runtime source component plan is invalid"
            )
        if target_plan is None:
            target_plan = source_plan
        if not isinstance(target_plan, RuntimeExecutionPlan):
            raise RouteGenerationError(
                "runtime target component plan is invalid"
            )
        source = AutomatedRouteCompiler(source_catalog, self.timeline)
        source_component = source.component_capability_identity(
            source_plan, executor_id
        )
        component = self.component_capability_identity(
            target_plan, executor_id
        )
        if source_component != component:
            source_execution = source._capability_identity(
                source_plan,
                executor_id,
                include_transitions=False,
                include_residency_ownership=False,
            )
            target_execution = self._capability_identity(
                target_plan,
                executor_id,
                include_transitions=False,
                include_residency_ownership=False,
            )
            source_neutral = source.phone_power_accounting_neutral_identity(
                source_plan,
                executor_id,
                include_transitions=False,
                include_residency_ownership=False,
            )
            target_neutral = self.phone_power_accounting_neutral_identity(
                target_plan,
                executor_id,
                include_transitions=False,
                include_residency_ownership=False,
            )
            if (
                source_execution != target_execution
                and source_neutral != target_neutral
            ):
                raise RouteGenerationError(
                    "runtime component capability changed"
                )
        target_template = (
            self._observation_store._component_template_sha256(target_plan)
        )
        try:
            return self._observation_store.rebind_legacy_component_rows(
                artifact_sha256=manifest.artifact_sha256,
                source_capability_sha256=source_component,
                source_template_sha256=(
                    source.observation_store._component_template_sha256(
                        source_plan
                    )
                ),
                component_capability_sha256=component,
                component_template_sha256=target_template,
                latency_only=latency_only,
            )
        except RuntimeLearningError as exc:
            if str(exc) != "legacy component observation rows are absent":
                raise
        return self._observation_store.rebind_legacy_component_rows(
            artifact_sha256=manifest.artifact_sha256,
            source_capability_sha256=source.route_capability_identity(
                source_plan, executor_id
            ),
            source_template_sha256=(
                source.observation_store._template_sha256(source_plan)
            ),
            component_capability_sha256=component,
            component_template_sha256=target_template,
            latency_only=latency_only,
        )

    def rebind_legacy_exact_route_observations(
        self,
        source_catalog: RuntimeCapabilityCatalog,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        executor_id: str,
        *,
        latency_only: bool = False,
    ) -> int:
        """Re-key route-total evidence after exact capability validation."""
        if not isinstance(source_catalog, RuntimeCapabilityCatalog):
            raise RouteGenerationError(
                "runtime source capability catalog is invalid"
            )
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError(
                "runtime exact-route manifest is invalid"
            )
        if not isinstance(plan, RuntimeExecutionPlan):
            raise RouteGenerationError(
                "runtime exact-route plan is invalid"
            )
        source = AutomatedRouteCompiler(source_catalog, self.timeline)
        source_capability = source.route_capability_identity(
            plan, executor_id
        )
        target_capability = self.route_capability_identity(
            plan, executor_id
        )
        if source_capability != target_capability and (
            source.phone_power_accounting_neutral_identity(
                plan,
                executor_id,
                include_transitions=True,
            )
            != self.phone_power_accounting_neutral_identity(
                plan,
                executor_id,
                include_transitions=True,
            )
        ):
            raise RouteGenerationError(
                "runtime exact-route capability changed"
            )
        return self._observation_store.rebind_legacy_exact_route_rows(
            artifact_sha256=manifest.artifact_sha256,
            source_capability_sha256=source_capability,
            target_capability_sha256=target_capability,
            plan=plan,
            latency_only=latency_only,
        )

    def rebind_legacy_transition_observations(
        self,
        source_catalog: RuntimeCapabilityCatalog,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        executor_id: str,
        *,
        latency_only: bool = False,
    ) -> int:
        """Re-key an identical cold-load path after ownership metadata."""
        if not plan.transitions:
            return 0
        if any(transition.evictions for transition in plan.transitions):
            raise RouteGenerationError(
                "runtime transition rebind cannot include eviction"
            )
        source = AutomatedRouteCompiler(source_catalog, self.timeline)
        source_component = source.component_capability_identity(
            plan, executor_id
        )
        target_component = self.component_capability_identity(
            plan, executor_id
        )
        source_execution = source._capability_identity(
            plan,
            executor_id,
            include_transitions=False,
            include_residency_ownership=False,
        )
        target_execution = self._capability_identity(
            plan,
            executor_id,
            include_transitions=False,
            include_residency_ownership=False,
        )
        source_neutral = source.phone_power_accounting_neutral_identity(
            plan,
            executor_id,
            include_transitions=False,
            include_residency_ownership=False,
        )
        target_neutral = self.phone_power_accounting_neutral_identity(
            plan,
            executor_id,
            include_transitions=False,
            include_residency_ownership=False,
        )
        if (
            source_execution != target_execution
            and source_neutral != target_neutral
        ):
            raise RouteGenerationError(
                "runtime transition execution capability changed"
            )
        source_transitions = {
            transition.transition_id: transition
            for transition in source_catalog.transitions
        }
        target_transitions = {
            transition.transition_id: transition
            for transition in self.catalog.transitions
        }
        if any(
            source_transitions.get(transition.transition_id)
                != target_transitions.get(transition.transition_id)
            for transition in plan.transitions
        ):
            raise RouteGenerationError(
                "runtime transition capability changed"
            )
        component_identity = self._transition_component_identity(
            manifest, plan, executor_id
        )
        return self._observation_store.rebind_legacy_transition_rows(
            artifact_sha256=manifest.artifact_sha256,
            source_capability_sha256=source_component,
            target_capability_sha256=target_component,
            component_identity_sha256=component_identity,
            transitions=plan.transitions,
            latency_only=latency_only,
        )

    def generate(
        self,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int | None = None,
        prepared_frontier=None,
        residency_holds: Mapping[
            str, RuntimeResidencyCohortHold
        ] | None = None,
        reuse_projections: Mapping[
            str, RuntimeResidencyReuseProjection
        ] | None = None,
        expected_reuse_count: int = 1,
        desktop_parent: tuple[str, str] | None = None,
    ) -> AutomatedCandidateSet:
        started_ns = time.perf_counter_ns()
        observed_at_us, residency_holds, reuse_projections = (
            self._validate_generate_inputs(
                request,
                manifest,
                snapshot,
                observed_at_us,
                residency_holds,
                reuse_projections,
                expected_reuse_count,
            )
        )
        work = self._generate_request_work(request, manifest)
        profile, profile_sha256 = self._generate_cost_profile(snapshot)
        capability_generation_sha256 = self._capability_generation_sha256
        rows = []
        work_profile_finished_ns = time.perf_counter_ns()
        input_bucket, output_bucket = request_shape_bucket(
            work.input_tokens, work.output_tokens
        )
        patterns = self._patterns(manifest)
        template_finished_ns = time.perf_counter_ns()
        pattern_by_key = {row.route_key: row for row in patterns}
        frontier, frontier_cache_hit = self._generate_frontier(
            request,
            manifest,
            work,
            snapshot,
            observed_at_us,
            prepared_frontier,
            patterns,
            profile,
            profile_sha256,
            capability_generation_sha256,
            input_bucket,
            output_bucket,
        )
        frontier_finished_ns = time.perf_counter_ns()
        context = _VisitContext(
            request=request,
            manifest=manifest,
            work=work,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            residency_holds=residency_holds,
            frontier=frontier,
            pattern_by_key=pattern_by_key,
            fallback_key=f"whole:{self.catalog.fallback.device_id}",
            profile=profile,
            profile_sha256=profile_sha256,
            planner=HierarchicalPlacementPlanner(profile),
            pattern_profiles={},
            cache_keys_seen=set(),
        )
        for visit in frontier.visits:
            rows.append(self._generate_visit_candidate(context, visit))
        live_cost_finished_ns = time.perf_counter_ns()
        rows, fallback_id = self._generate_fallback_rows(
            rows,
            request,
            manifest,
            snapshot,
            pattern_by_key,
            context.fallback_key,
        )
        rows = self._generate_provisional_rows(rows, fallback_id)
        rows = self._generate_paired_rows(rows)
        rows = self._generate_break_even_rows(
            rows, manifest, reuse_projections, expected_reuse_count
        )
        if desktop_parent is not None and not any(
            (row.binding.executor_id, row.plan.desktop_placement_sha256) == desktop_parent
            for row in rows
        ):
            for pattern in patterns:
                if (pattern.coordinator_executor_id == desktop_parent[0]
                    and pattern.split_fraction_ppm == 0
                    and pattern.assisted_operator_kind is None):
                    rows.append(self._materialize_route_key(
                        request=request, manifest=manifest, work=work, snapshot=snapshot,
                        profile=profile, profile_sha256=profile_sha256, pattern=pattern,
                        input_token_bucket=input_bucket, output_token_bucket=output_bucket,
                        observed_at_us=observed_at_us, residency_holds=residency_holds,
                    ))
        rows, baseline_id = self._generate_baseline_rows(
            rows, request, manifest, desktop_parent
        )
        final = self._generate_pareto_rows(rows)
        result = self._generate_candidate_set(
            request,
            manifest,
            snapshot,
            final,
            baseline_id,
            fallback_id,
            frontier,
            frontier_cache_hit,
        )
        finished_ns = time.perf_counter_ns()
        self._record_generation_timing(
            started_ns,
            work_profile_finished_ns,
            template_finished_ns,
            frontier_finished_ns,
            live_cost_finished_ns,
            finished_ns,
            len(result.candidates),
        )
        return result

    def _validate_generate_inputs(
        self,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int | None,
        residency_holds: Mapping[str, RuntimeResidencyCohortHold] | None,
        reuse_projections: Mapping[
            str, RuntimeResidencyReuseProjection
        ] | None,
        expected_reuse_count: int,
    ) -> tuple[int, dict, dict]:
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError("model manifest is invalid")
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise RouteGenerationError("runtime system snapshot is invalid")
        observed_at_us = (
            request.arrival_us if observed_at_us is None else observed_at_us
        )
        if type(observed_at_us) is not int or observed_at_us < request.arrival_us:
            raise RouteGenerationError("automated observation time is invalid")
        snapshot.validate_at(observed_at_us)
        residency_holds = dict(residency_holds or {})
        if any(
            not isinstance(hold, RuntimeResidencyCohortHold)
            or resource_id != hold.resource_id
            for resource_id, hold in residency_holds.items()
        ):
            raise RouteGenerationError(
                "runtime residency cohort hold is invalid"
            )
        reuse_projections = dict(reuse_projections or {})
        if type(expected_reuse_count) is not int or expected_reuse_count < 1:
            raise RouteGenerationError(
                "runtime expected reuse count is invalid"
            )
        if any(
            not isinstance(projection, RuntimeResidencyReuseProjection)
            or component_id != projection.component_identity_sha256
            or projection.artifact_sha256 != manifest.artifact_sha256
            for component_id, projection in reuse_projections.items()
        ):
            raise RouteGenerationError(
                "runtime residency reuse projections are invalid"
            )
        return observed_at_us, residency_holds, reuse_projections

    def _generate_request_work(
        self, request, manifest: ModelManifest
    ) -> ModelRequestWork:
        work_key = (
            manifest.artifact_sha256,
            request.input_tokens,
            request.output_tokens,
        )
        work = self._request_work_cache.get(work_key)
        if work is None:
            work = manifest.request_work(
                request.input_tokens, request.output_tokens
            )
            if len(self._request_work_cache) >= 512:
                self._request_work_cache.pop(next(iter(
                    self._request_work_cache
                )))
            self._request_work_cache[work_key] = work
        return work

    def _generate_cost_profile(
        self, snapshot: HeterogeneousRuntimeSnapshot
    ) -> tuple[PlacementHardwareProfile, str]:
        profile_key = tuple(
            (
                link.link_id,
                link.bandwidth_bytes_per_s
                if snapshot.links.get(link.link_id) is None
                else snapshot.links[
                    link.link_id
                ].measured_bandwidth_bytes_per_s,
            )
            for link in self.catalog.placement_profile.links
        ) + tuple(
            (
                "phone-power",
                row.device_id,
                row.domain_id,
                row.active_power_mw,
                row.idle_power_mw,
                row.estimation_version,
            )
            for row in self.catalog.phone_power_profiles
        )
        cached_profile = self._cost_profile_cache.get(profile_key)
        if cached_profile is None:
            profile = self._effective_cost_profile(snapshot)
            # Readiness remains an admission reason instead of erasing rows.
            profile = replace(
                profile,
                devices={
                    device_id: replace(device, ready=True)
                    for device_id, device in profile.devices.items()
                },
                links=tuple(
                    replace(link, ready=True) for link in profile.links
                ),
            )
            cached_profile = (profile, canonical_sha256(profile))
            if len(self._cost_profile_cache) >= 64:
                self._cost_profile_cache.pop(next(iter(
                    self._cost_profile_cache
                )))
            self._cost_profile_cache[profile_key] = cached_profile
        return cached_profile

    def _generate_frontier(
        self,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        prepared_frontier,
        patterns: tuple[_Pattern, ...],
        profile: PlacementHardwareProfile,
        profile_sha256: str,
        capability_generation_sha256: str,
        input_bucket: int,
        output_bucket: int,
    ) -> tuple[object, bool]:
        frontier = prepared_frontier
        if frontier is not None and (
            frontier.artifact_sha256 != manifest.artifact_sha256
            or frontier.capability_generation_sha256
                != capability_generation_sha256
            or frontier.input_token_bucket != input_bucket
            or frontier.output_token_bucket != output_bucket
        ):
            raise RouteGenerationError(
                "prepared placement frontier identity differs"
            )
        if frontier is None:
            frontier = self._rough_compiler.lookup(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=capability_generation_sha256,
                input_tokens=work.input_tokens,
                output_tokens=work.output_tokens,
                quality_requirement=request.quality_requirement,
            )
        frontier_cache_hit = frontier is not None
        if frontier is None:
            rough_work = manifest.request_work(input_bucket, output_bucket)
            try:
                frontier, _ = self._rough_compiler.compile(
                    artifact_sha256=manifest.artifact_sha256,
                    capability_generation_sha256=(
                        capability_generation_sha256
                    ),
                    input_tokens=work.input_tokens,
                    output_tokens=work.output_tokens,
                    quality_requirement=request.quality_requirement,
                    visits=self._rough_visits(
                        manifest,
                        patterns,
                        rough_work,
                        profile,
                        profile_sha256,
                        snapshot,
                        observed_at_us,
                    ),
                )
            except RuntimeSearchError as exc:
                raise RouteGenerationError(str(exc)) from exc
        return frontier, frontier_cache_hit

    def _generate_visit_candidate(
        self, context: _VisitContext, visit
    ) -> AutomatedRouteCandidate:
        pattern = context.pattern_by_key.get(visit.route_key)
        if pattern is None:
            raise RouteGenerationError(
                "cached rough plan references an absent route"
            )
        residency_states, residency_variant = self._visit_residency(
            context.manifest,
            pattern,
            context.snapshot,
            visit,
            context.fallback_key,
        )
        kernel_signature = self._cached_kernel_signature(
            context.manifest, context.work, pattern
        )
        pattern_profile, pattern_profile_sha256, pattern_planner = (
            self._pattern_profile(
                pattern,
                context.profile,
                context.profile_sha256,
                context.planner,
                context.pattern_profiles,
            )
        )
        cache_key = (
            context.manifest.artifact_sha256,
            context.frontier.input_token_bucket,
            context.frontier.output_token_bucket,
            kernel_signature,
            pattern.route_key,
            pattern_profile_sha256,
            context.request.quality_requirement,
        )
        placement, placement_error, nodes, assignments = (
            self._placement_for_visit(
                context.request,
                context.manifest,
                context.work,
                pattern,
                pattern_planner,
                cache_key,
                context.cache_keys_seen,
            )
        )
        candidate = self._one(
            context.request,
            context.manifest,
            context.work,
            context.snapshot,
            pattern_profile,
            pattern,
            residency_variant,
            residency_states,
            context.observed_at_us,
            nodes,
            assignments,
            placement,
            placement_error,
            context.residency_holds,
        )
        if visit.coverage_only:
            candidate = _reject_row(candidate, "SEARCH_COVERAGE_ONLY")
        return candidate

    def _visit_residency(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        visit,
        fallback_key: str,
    ) -> tuple[dict[str, str], str]:
        residency_states = {
            device_id: (
                "cold" if residency is None else residency.state
            )
            for device_id in pattern.device_ids
            for residency in (
                self._matching_residency(
                    manifest, pattern, snapshot, device_id
                ),
            )
        }
        remote = self._remote_resident_group(manifest, pattern)
        if remote is not None:
            bound, _ = self._remote_resident_owner_status(manifest, remote, snapshot)
            if bound is not None:
                for owner in bound.sessions:
                    residency_states[self._phone_session_capability(owner.session_id)[0]] = "hot"
        if pattern.route_key == fallback_key:
            residency_states = {
                device_id: visit.residency_variant
                for device_id in pattern.device_ids
            }
            residency_variant = visit.residency_variant
        else:
            residency_variant = (
                "cold"
                if "cold" in residency_states.values()
                else "warm"
                if "warm" in residency_states.values()
                else "hot"
            )
        return residency_states, residency_variant

    def _cached_kernel_signature(
        self,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
    ):
        signature_key = (
            manifest.artifact_sha256,
            work.input_tokens,
            work.output_tokens,
            pattern.route_key,
        )
        kernel_signature = self._kernel_signature_cache.get(
            signature_key
        )
        if kernel_signature is None:
            kernel_signature = self._kernel_signature(pattern, work)
            if len(self._kernel_signature_cache) >= 4_096:
                self._kernel_signature_cache.pop(next(iter(
                    self._kernel_signature_cache
                )))
            self._kernel_signature_cache[
                signature_key
            ] = kernel_signature
        return kernel_signature

    def _pattern_profile(
        self,
        pattern: _Pattern,
        profile: PlacementHardwareProfile,
        profile_sha256: str,
        planner: HierarchicalPlacementPlanner,
        pattern_profiles: dict[
            tuple[str, str], tuple[object, str, HierarchicalPlacementPlanner]
        ],
    ) -> tuple[PlacementHardwareProfile, str, HierarchicalPlacementPlanner]:
        pattern_profile = profile
        pattern_profile_sha256 = profile_sha256
        pattern_planner = planner
        coordinator = self._coordinator(pattern)
        transport_generation = (
            coordinator.adapter_parameters.get(
                "request_transport_generation"
            )
            if coordinator is not None else None
        )
        if transport_generation is not None:
            profile_key_for_pattern = (
                pattern.route_key, transport_generation
            )
            profiled = pattern_profiles.get(profile_key_for_pattern)
            if profiled is None:
                participating = frozenset(pattern.device_ids)
                pattern_profile = replace(
                    profile,
                    links=tuple(
                        link for link in profile.links
                        if (
                            link.source_device not in participating
                            and link.target_device not in participating
                        )
                        or link.transport_generation
                            == transport_generation
                    ),
                )
                pattern_profile_sha256 = canonical_sha256(
                    pattern_profile
                )
                pattern_planner = HierarchicalPlacementPlanner(
                    pattern_profile
                )
                profiled = (
                    pattern_profile,
                    pattern_profile_sha256,
                    pattern_planner,
                )
                pattern_profiles[profile_key_for_pattern] = profiled
            (
                pattern_profile,
                pattern_profile_sha256,
                pattern_planner,
            ) = profiled
        return pattern_profile, pattern_profile_sha256, pattern_planner

    def _placement_for_visit(
        self,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
        pattern_planner: HierarchicalPlacementPlanner,
        cache_key: tuple[object, ...],
        cache_keys_seen: set[tuple[object, ...]],
    ) -> tuple[object, str | None, object, object]:
        def build_placement():
            nodes = self._nodes(manifest, work, pattern)
            assignments = self._operator_assignments(
                manifest, nodes, pattern
            )
            placement = None
            placement_error = None
            try:
                placement = pattern_planner.plan_sequence(
                    problem_id="auto:" + pattern.route_key,
                    nodes=nodes,
                    initial_device=self.catalog.fallback.device_id,
                    final_device=self.catalog.fallback.device_id,
                    deadline_us=2**63 - 1,
                    required_quality=request.quality_requirement,
                    require_measured=False,
                    defer_memory_validation=True,
                )
            except PlacementError as exc:
                placement_error = str(exc)
            return (
                placement, placement_error, nodes, assignments
            )
        cached, cache_hit = self._cached_placement(
            cache_key, build_placement
        )
        if cache_hit and cache_key not in cache_keys_seen:
            self._placement_cache_hits += 1
        cache_keys_seen.add(cache_key)
        return cached

    def _generate_fallback_rows(
        self,
        rows: list[AutomatedRouteCandidate],
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        pattern_by_key: Mapping[str, _Pattern],
        fallback_key: str,
    ) -> tuple[list[AutomatedRouteCandidate], str]:
        fallback_pattern = pattern_by_key[fallback_key]
        fallback_state = self._matching_residency(
            manifest,
            fallback_pattern,
            snapshot,
            self.catalog.fallback.device_id,
        )
        fallback_variant = "cold" if fallback_state is None else fallback_state.state
        fallback_id = f"auto:{fallback_key}:residency:{fallback_variant}"
        if not any(row.candidate_id == fallback_id for row in rows):
            raise RouteGenerationError("qualified fallback route was not generated")
        fallback = next(row for row in rows if row.candidate_id == fallback_id)
        if fallback.cost.finish_upper_us > request.deadline_us:
            rows = [
                self._without_rejection(row, "SLO_UPPER_BOUND")
                for row in rows
            ]
        return rows, fallback_id

    def _generate_provisional_rows(
        self, rows: list[AutomatedRouteCandidate], fallback_id: str
    ) -> list[AutomatedRouteCandidate]:
        provisional = []
        for row in rows:
            is_fallback = row.candidate_id == fallback_id
            reasons = tuple(
                reason for reason in row.rejection_reasons
                if not (
                    is_fallback
                    and reason in self._desktop_cost_only_rejections()
                )
            )
            provisional.append(replace(
                row,
                binding=replace(
                    row.binding,
                    ready=not reasons,
                    eligibility_reasons=reasons,
                ),
                admitted=not reasons,
                rejection_reasons=reasons,
                baseline=False,
            ))
        return provisional

    @staticmethod
    def _generate_paired_rows(
        rows: list[AutomatedRouteCandidate],
    ) -> list[AutomatedRouteCandidate]:
        paired_rows = []
        for row in rows:
            baseline_executor_id = row.plan.baseline_executor_id
            if baseline_executor_id is None:
                paired_rows.append(row)
                continue
            parents = tuple(
                parent for parent in rows
                if parent.binding.executor_id == baseline_executor_id
                and parent.plan.baseline_executor_id is None
                and parent.plan.desktop_placement_sha256
                    == row.plan.desktop_placement_sha256
            )
            if len(parents) == 1:
                paired_rows.append(replace(
                    row,
                    paired_baseline_route_id=parents[0].candidate_id,
                ))
                continue
            paired_rows.append(_reject_row(row, "PAIRED_BASELINE_ABSENT"))
        return paired_rows

    def _generate_break_even_rows(
        self,
        rows: list[AutomatedRouteCandidate],
        manifest: ModelManifest,
        reuse_projections: Mapping[str, RuntimeResidencyReuseProjection],
        expected_reuse_count: int,
    ) -> list[AutomatedRouteCandidate]:
        paired_by_id = {row.candidate_id: row for row in rows}
        break_even_rows = []
        for row in rows:
            if (
                row.paired_baseline_route_id is None
                or row.residency_variant != "cold"
                or not row.plan.transitions
            ):
                break_even_rows.append(row)
                continue
            parent = paired_by_id[row.paired_baseline_route_id]
            if (
                row.cost.fleet_energy_upper_uj is None
                or parent.cost.fleet_energy_lower_uj is None
            ):
                break_even_rows.append(row)
                continue
            break_even_rows.append(self._break_even_row(
                row, parent, manifest, reuse_projections, expected_reuse_count
            ))
        return break_even_rows

    def _break_even_row(
        self,
        row: AutomatedRouteCandidate,
        parent: AutomatedRouteCandidate,
        manifest: ModelManifest,
        reuse_projections: Mapping[str, RuntimeResidencyReuseProjection],
        expected_reuse_count: int,
    ) -> AutomatedRouteCandidate:
        component_identity = runtime_residency_component_identity(
            manifest.artifact_sha256, row.plan, row.binding
        )
        planned_identity = row.plan.adapter_parameters.get(
            "resident_model_identity_sha256"
        )
        if (
            planned_identity is not None
            and planned_identity != component_identity.identity_sha256
        ):
            raise RouteGenerationError(
                "resident model component identity differs"
            )
        projection = reuse_projections.get(
            component_identity.identity_sha256
        )
        decomposition = (
            row.cost.warm_execution_energy_upper_uj,
            row.cost.transition_energy_uj,
            row.cost.transition_energy_upper_uj,
            parent.cost.warm_execution_energy_lower_uj,
            parent.cost.transition_energy_uj,
            parent.cost.transition_energy_lower_uj,
        )
        if any(value is None for value in decomposition):
            return _reject_row(
                row, "COLD_WARM_ENERGY_DECOMPOSITION_UNKNOWN"
            )
        passed, break_even = _break_even_evidence(
            row,
            parent,
            projection,
            component_identity.identity_sha256,
            expected_reuse_count,
        )
        if passed:
            return replace(row, residency_break_even=break_even)
        return replace(
            _reject_row(row, "COLD_RESIDENCY_BREAK_EVEN"),
            residency_break_even=break_even,
        )

    def _generate_baseline_rows(
        self,
        rows: list[AutomatedRouteCandidate],
        request,
        manifest: ModelManifest,
        desktop_parent: tuple[str, str] | None = None,
    ) -> tuple[list[AutomatedRouteCandidate], str]:
        if desktop_parent is None:
            desktop_baseline = self._desktop_control(rows, manifest)
        else:
            # A running parent is an execution constraint, not qualification.
            parents = [
                row for row in rows
                if (row.binding.executor_id, row.plan.desktop_placement_sha256)
                    == desktop_parent
                and row.plan.execution_contract.execution_mode == "desktop"
                and self._physically_qualified_desktop(row)
                and set(row.rejection_reasons).issubset(
                    self._desktop_cost_only_rejections()
                )
            ]
            if len(parents) != 1:
                raise RouteGenerationError(
                    "execution desktop parent was not generated: "
                    + repr(desktop_parent)
                )
            desktop_baseline = self._without_rejections(
                parents[0], self._desktop_cost_only_rejections()
            )
        if desktop_baseline.cost.finish_upper_us > request.deadline_us:
            rows = [
                self._without_rejection(row, "SLO_UPPER_BOUND")
                for row in rows
            ]
            desktop_baseline = self._without_rejection(
                desktop_baseline, "SLO_UPPER_BOUND"
            )
        rows = [
            desktop_baseline
            if row.candidate_id == desktop_baseline.candidate_id
            else row
            for row in rows
        ]
        rows = [
            replace(row, baseline=(
                row.candidate_id == desktop_baseline.candidate_id
            ))
            for row in rows
        ]
        return rows, desktop_baseline.candidate_id

    def _generate_pareto_rows(
        self, rows: list[AutomatedRouteCandidate]
    ) -> list[AutomatedRouteCandidate]:
        grouped: dict[tuple[object, ...], list[AutomatedRouteCandidate]] = {}
        for row in rows:
            key = (
                row.route_family,
                row.device_ids,
                row.assisted_operator_kind,
                row.split_axis,
                row.residency_variant,
            )
            grouped.setdefault(key, []).append(row)
        dominated_ids = {
            right.candidate_id
            for group in grouped.values()
            for left in group
            for right in group
            if left.candidate_id != right.candidate_id
            and self._dominated(left, right)
            and not right.baseline
        }
        final = []
        for row in rows:
            if row.candidate_id in dominated_ids:
                row = replace(
                    _reject_row(row, "PARETO_DOMINATED"),
                    pareto_dominated=True,
                )
            final.append(row)
        return final

    @staticmethod
    def _generate_candidate_set(
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        final: list[AutomatedRouteCandidate],
        baseline_id: str,
        fallback_id: str,
        frontier,
        frontier_cache_hit: bool,
    ) -> AutomatedCandidateSet:
        search_metadata = dict(frontier.metadata(
            cache_hit=frontier_cache_hit
        ))
        search_metadata["evaluated_plan_count"] = len(final)
        search_metadata["visited_plan_ids"] = tuple(
            row.candidate_id for row in final
        )
        try:
            return AutomatedCandidateSet(
                request_id=request.request_id,
                model_id=manifest.model_id,
                snapshot_id=snapshot.snapshot_id,
                candidates=tuple(final),
                baseline_route_id=baseline_id,
                recovery_fallback_route_id=fallback_id,
                search_metadata=search_metadata,
            )
        except ValueError as exc:
            raise RouteGenerationError(str(exc)) from exc

    def _record_generation_timing(
        self,
        started_ns: int,
        work_profile_finished_ns: int,
        template_finished_ns: int,
        frontier_finished_ns: int,
        live_cost_finished_ns: int,
        finished_ns: int,
        visited_candidates: int,
    ) -> None:
        self._last_generation_timing = MappingProxyType({
            "candidate_template_lookup_us": (
                template_finished_ns - work_profile_finished_ns
            ) // 1000,
            "live_cost_update_us": (
                live_cost_finished_ns - frontier_finished_ns
            ) // 1000,
            "pareto_and_contract_us": (
                finished_ns - live_cost_finished_ns
            ) // 1000,
            "request_work_and_profile_us": (
                work_profile_finished_ns - started_ns
            ) // 1000,
            "rough_frontier_us": (
                frontier_finished_ns - template_finished_ns
            ) // 1000,
            "total_us": (finished_ns - started_ns) // 1000,
            "visited_candidates": visited_candidates,
        })


@dataclass(frozen=True)
class _VisitContext:
    """Per-generation constants shared by every frontier visit."""

    request: object
    manifest: ModelManifest
    work: ModelRequestWork
    snapshot: HeterogeneousRuntimeSnapshot
    observed_at_us: int
    residency_holds: Mapping[str, RuntimeResidencyCohortHold]
    frontier: object
    pattern_by_key: Mapping[str, _Pattern]
    fallback_key: str
    profile: PlacementHardwareProfile
    profile_sha256: str
    planner: HierarchicalPlacementPlanner
    pattern_profiles: dict[
        tuple[str, str], tuple[object, str, HierarchicalPlacementPlanner]
    ]
    cache_keys_seen: set[tuple[object, ...]]


def _reject_row(
    row: AutomatedRouteCandidate, reason: str
) -> AutomatedRouteCandidate:
    reasons = tuple(sorted(set(row.rejection_reasons + (reason,))))
    return replace(
        row,
        binding=replace(
            row.binding,
            ready=False,
            eligibility_reasons=reasons,
        ),
        admitted=False,
        rejection_reasons=reasons,
    )


def _break_even_evidence(
    row: AutomatedRouteCandidate,
    parent: AutomatedRouteCandidate,
    projection: RuntimeResidencyReuseProjection | None,
    component_identity_sha256: str,
    expected_reuse_count: int,
) -> tuple[bool, dict[str, object]]:
    transition_latency_us = sum(
        transition.latency_us
        for transition in row.plan.transitions
    )
    transition_energy_uj = row.cost.transition_energy_uj
    assert transition_energy_uj is not None
    parent_transition_latency_us = sum(
        transition.latency_us
        for transition in parent.plan.transitions
    )
    parent_transition_energy_uj = (
        parent.cost.transition_energy_uj
    )
    assert parent_transition_energy_uj is not None
    incremental_transition_latency_us = max(
        0,
        transition_latency_us - parent_transition_latency_us,
    )
    incremental_transition_energy_uj = max(
        0,
        transition_energy_uj - parent_transition_energy_uj,
    )
    restore_energy_uj = (
        transition_energy_uj
        if any(
            transition.evictions
            for transition in row.plan.transitions
        )
        else 0
    )
    expected_uses = (
        expected_reuse_count
        if projection is None
        else max(
            expected_reuse_count,
            projection.expected_use_count,
        )
    )
    warm_upper_uj = row.cost.warm_execution_energy_upper_uj
    parent_warm_lower_uj = (
        parent.cost.warm_execution_energy_lower_uj
    )
    assert warm_upper_uj is not None
    assert parent_warm_lower_uj is not None
    assert row.cost.transition_energy_upper_uj is not None
    assert parent.cost.transition_energy_lower_uj is not None
    candidate_cohort_upper_uj = (
        warm_upper_uj
        + row.cost.transition_energy_upper_uj
        + (expected_uses - 1) * warm_upper_uj
        + restore_energy_uj
    )
    desktop_cohort_lower_uj = (
        parent_warm_lower_uj
        + parent.cost.transition_energy_lower_uj
        + (expected_uses - 1) * parent_warm_lower_uj
    )
    passed = (
        candidate_cohort_upper_uj
        <= desktop_cohort_lower_uj
    )
    break_even = {
        "candidate_cohort_upper_uj": candidate_cohort_upper_uj,
        "desktop_cohort_lower_uj": desktop_cohort_lower_uj,
        "expected_use_count": expected_uses,
        "horizon_us": (
            0
            if projection is None
            else projection.horizon_us
        ),
        "residency_component_identity_sha256": (
            component_identity_sha256
        ),
        "passed": passed,
        "incremental_transition_energy_uj": (
            incremental_transition_energy_uj
        ),
        "incremental_transition_latency_us": (
            incremental_transition_latency_us
        ),
        "paired_transition_energy_uj": (
            parent_transition_energy_uj
        ),
        "paired_transition_latency_us": (
            parent_transition_latency_us
        ),
        "restore_energy_uj": restore_energy_uj,
        "transition_energy_evidence": (
            "MEASURED"
            if all(
                transition.energy_maturity == "QUALIFIED"
                for transition in row.plan.transitions
            )
            else "ASSUMED"
        ),
        "transition_energy_uj": transition_energy_uj,
        "warm_route_upper_uj": warm_upper_uj,
    }
    return passed, break_even
