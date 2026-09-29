"""AutomatedCandidateMixin generation operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.route_generation import AutomatedRouteCompiler, RouteGenerationError
from ..._internal.runtime_learning import RuntimeLearningError
from ..._internal.runtime_plan import AutomatedCandidateSet
from ..._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from ..._internal.adaptive_decode_planning import adaptive_probe_contracts


def _prepare_route_evidence_migrations(controller, candidate_set, manifest) -> int:
    """Recover exact route evidence independently of FFN helper eligibility."""
    if not controller._automated_observation_sources:
        return 0
    compiler = controller._automated_compiler()
    live = candidate_set.search_metadata.get("route_template_live_route_ids")
    recorded = 0
    for route in candidate_set.candidates:
        if type(live) in {list, tuple} and route.candidate_id not in live:
            continue
        try:
            component = compiler.component_capability_identity(route.plan, route.binding.executor_id)
        except RouteGenerationError:
            continue
        for source_sha256, source_catalog in controller._automated_observation_sources.items():
            key = ("route:" + source_sha256, component, route.plan.plan_sha256)
            if key in controller._legacy_evidence_migration_cache:
                continue
            latency_only = bool(set(route.device_ids) & set(
                controller._runtime_capabilities.phone_power_profile_by_device
            ))
            for rebind in (compiler.rebind_legacy_transition_observations,
                           compiler.rebind_legacy_exact_route_observations):
                try:
                    recorded += rebind(source_catalog, manifest, route.plan,
                                       route.binding.executor_id, latency_only=latency_only)
                except (RouteGenerationError, RuntimeLearningError):
                    pass
            controller._legacy_evidence_migration_cache.add(key)
    return recorded


def _generate_automated_candidate_set(
    controller,
    request: Request,
    manifest: ModelManifest,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    *,
    use_residency_holds: bool = True,
    update_phone_residency_portfolio: bool = True,
    desktop_parent: tuple[str, str] | None = None,
) -> AutomatedCandidateSet:
    if controller._runtime_capabilities is None:
        raise UnifiedScheduleError(
            "runtime capabilities are not registered"
        )
    if update_phone_residency_portfolio:
        controller._update_phone_residency_portfolio(
            request, manifest, observed_at_us, snapshot
        )
    prepared_frontier = (
        controller._prepared_placement_frontier(
            request,
            manifest,
            snapshot,
            observed_at_us,
        )
        if update_phone_residency_portfolio else None
    )
    try:
        residency_holds = (
            controller._runtime_residency_cohorts.holds(
                controller._runtime_capabilities,
                snapshot,
                controller._runtime_controller.current_tickets(),
                observed_at_us,
            )
            if use_residency_holds else {}
        )
        placement_epoch = (
            controller._runtime_residency_cohorts.model_placement_epoch(
                manifest.artifact_sha256,
                observed_at_us,
                request_id=request.request_id,
                queued_request_ids=tuple(
                    ticket.request.request_id
                    for ticket in (
                        controller._runtime_controller.current_tickets()
                    )
                    if ticket.dispatch_state in {
                        "QUEUED", "REPLAN_REQUIRED"
                    }
                ),
            )
        )
        try:
            compiler = controller._automated_compiler()
            generation_arguments = dict(
                observed_at_us=observed_at_us,
                prepared_frontier=prepared_frontier,
                residency_holds=residency_holds,
                reuse_projections=placement_epoch.reuse_projections,
                expected_reuse_count=max(
                    1,
                    placement_epoch.active_request_count
                        + placement_epoch.virtual_queue_request_count,
                ),
                **({"desktop_parent": desktop_parent}
                   if desktop_parent is not None else {}),
            )
            candidate_set = compiler.generate(request, manifest, snapshot, **generation_arguments)
            if _prepare_route_evidence_migrations(controller, candidate_set, manifest):
                candidate_set = compiler.generate(request, manifest, snapshot, **generation_arguments)
            controller._prepare_legacy_evidence_migrations(
                candidate_set, request, manifest
            )
            candidate_set = controller._apply_adaptive_history_costs(
                candidate_set,
                request,
                manifest,
                snapshot.cost_features,
            )
            compiler.capture_phone_residency_route_evidence(
                manifest, candidate_set, request.output_tokens
            )
            portfolio_changed = (
                controller._update_phone_residency_portfolio(
                    request, manifest, observed_at_us, snapshot
                )
                if update_phone_residency_portfolio else False
            )
            if portfolio_changed:
                candidate_set = compiler.generate(
                    request,
                    manifest,
                    snapshot,
                    observed_at_us=observed_at_us,
                    prepared_frontier=None,
                    residency_holds=residency_holds,
                    reuse_projections=(
                        placement_epoch.reuse_projections
                    ),
                    expected_reuse_count=max(
                        1,
                        placement_epoch.active_request_count
                            + placement_epoch
                                .virtual_queue_request_count,
                    ),
                    **({"desktop_parent": desktop_parent}
                       if desktop_parent is not None else {}),
                )
                controller._prepare_legacy_evidence_migrations(
                    candidate_set, request, manifest
                )
                candidate_set = controller._apply_adaptive_history_costs(
                    candidate_set,
                    request,
                    manifest,
                    snapshot.cost_features,
                )
                compiler.capture_phone_residency_route_evidence(
                    manifest, candidate_set, request.output_tokens
                )
            candidate_set = (
                controller._apply_phone_residency_portfolio_authorization(
                    candidate_set, manifest, request, snapshot=snapshot,
                    observed_at_us=observed_at_us,
                )
            )
            search_metadata = dict(candidate_set.search_metadata)
            phone_events = (
                controller._model_placement_controller.phone_layout_events()
            )
            if phone_events:
                search_metadata["phone_residency"] = dict(
                    phone_events[-1]
                )
            return replace(
                candidate_set,
                search_metadata={
                    **search_metadata,
                    "model_placement_epoch_sha256": (
                        placement_epoch.epoch_sha256
                    ),
                    "model_placement_epoch_generation": (
                        placement_epoch.generation
                    ),
                    "virtual_queue_request_count": (
                        placement_epoch.virtual_queue_request_count
                    ),
                },
            )
        finally:
            planner = controller._background_placement_planner
            if planner is not None:
                planner.start()
    except RuntimeResidencyCohortError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def _prepare_legacy_evidence_migrations(
    controller,
    candidate_set: AutomatedCandidateSet,
    request: Request,
    manifest: ModelManifest,
) -> None:
    """Migrate compatible legacy evidence once per source identity."""
    if (
        controller._runtime_capabilities is None
        or controller._runtime_capability_generation_sha256 is None
        or not (
            controller._adaptive_observation_sources
            or controller._automated_observation_sources
        )
    ):
        return
    compiler = controller._automated_compiler()
    try:
        contracts = adaptive_probe_contracts(
            candidate_set,
            manifest,
            controller._runtime_capabilities,
            request.output_tokens,
        )
        baseline_contract = contracts.get(
            candidate_set.baseline_route_id
        )
        if baseline_contract is None:
            return
        baseline_policy = AdaptiveDecodePolicy.from_json(
            baseline_contract
        )
    except AdaptiveDecodeError:
        return
    completed: list[tuple[str, str, str]] = []
    live_route_values = candidate_set.search_metadata.get(
        "route_template_live_route_ids"
    )
    live_route_ids = (
        None
        if type(live_route_values) not in {list, tuple}
        else frozenset(live_route_values)
    )
    for candidate in candidate_set.candidates:
        if (
            live_route_ids is not None
            and candidate.candidate_id not in live_route_ids
        ):
            continue
        contract = contracts.get(candidate.candidate_id)
        if contract is None or not contract.get(
            "adaptive_envelope", False
        ):
            continue
        try:
            selected_policy = AdaptiveDecodePolicy.from_json(contract)
            probe_values = contract.get(
                "adaptive_decode_probe_contracts", []
            )
            if type(probe_values) is not list:
                continue
            policies = tuple(
                AdaptiveDecodePolicy.from_json(value)
                for value in probe_values
            )
            if all(
                row.policy_hash != selected_policy.policy_hash
                for row in policies
            ):
                policies += (selected_policy,)
            target_component = (
                compiler.component_capability_identity(
                    candidate.plan,
                    candidate.binding.executor_id,
                )
            )
        except (AdaptiveDecodeError, RouteGenerationError):
            continue
        for source_sha256, (
            source_catalog,
            source_profiles,
        ) in controller._adaptive_observation_sources.items():
            cache_key = (
                "adaptive:" + source_sha256,
                target_component,
                candidate.plan.plan_sha256,
            )
            if cache_key in controller._legacy_evidence_migration_cache:
                continue
            source_compiler = (
                controller._adaptive_observation_source_compilers.get(
                    source_sha256
                )
            )
            if source_compiler is None:
                source_compiler = AutomatedRouteCompiler(
                    source_catalog, controller.timeline
                )
                controller._adaptive_observation_source_compilers[
                    source_sha256
                ] = source_compiler
            try:
                source_component = (
                    source_compiler.component_capability_identity(
                        candidate.plan,
                        candidate.binding.executor_id,
                    )
                )
                if source_component != target_component:
                    source_execution = (
                        source_compiler._capability_identity(
                            candidate.plan,
                            candidate.binding.executor_id,
                            include_transitions=False,
                            include_residency_ownership=False,
                        )
                    )
                    target_execution = compiler._capability_identity(
                        candidate.plan,
                        candidate.binding.executor_id,
                        include_transitions=False,
                        include_residency_ownership=False,
                    )
                    source_neutral = (
                        source_compiler
                        .phone_power_accounting_neutral_identity(
                            candidate.plan,
                            candidate.binding.executor_id,
                            include_transitions=False,
                            include_residency_ownership=False,
                        )
                    )
                    target_neutral = (
                        compiler
                        .phone_power_accounting_neutral_identity(
                            candidate.plan,
                            candidate.binding.executor_id,
                            include_transitions=False,
                            include_residency_ownership=False,
                        )
                    )
                    if (
                        source_execution != target_execution
                        and source_neutral != target_neutral
                    ):
                        completed.append(cache_key)
                        continue
                controller._adaptive_decode \
                    .rebind_legacy_component_observations(
                        model_artifact_sha256=(
                            manifest.artifact_sha256
                        ),
                        source_component_capability_sha256=(
                            source_component
                        ),
                        component_capability_sha256=(
                            target_component
                        ),
                        source_planning_profile_sha256s=(
                            source_profiles
                        ),
                        baseline=baseline_policy,
                        candidates=policies,
                    )
            except AdaptiveDecodeError as exc:
                if str(exc) == (
                    "adaptive evidence cannot rebind while active"
                ):
                    controller._legacy_evidence_migration_cache.update(
                        completed
                    )
                    return
                if str(exc) != (
                    "legacy adaptive component observations are absent"
                ):
                    raise
            except RouteGenerationError:
                pass
            completed.append(cache_key)
    controller._legacy_evidence_migration_cache.update(completed)
