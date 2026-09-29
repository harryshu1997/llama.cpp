"""PlacementEpochMixin refresh operations on its existing owner."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, Sequence

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeCapabilityError
from ..._internal.route_generation import AutomatedRouteCompiler, RouteGenerationError
from ..._internal.runtime_search import request_shape_bucket
from ..._internal.runtime_plan import AutomatedCandidateSet
from ..._internal.runtime_resources import RuntimeResourceError
from ..._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from ..._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from ..._internal.runtime_controller import RuntimeControllerError, RuntimeRequestTicket
from ..._internal.types import canonical_sha256
from .common import _PlacementRefreshDemand, _PlacementRefreshResult


def _refresh_model_placement_epochs_after_learning(
    controller,
    request_ids: Sequence[str],
    observed_at_us: int,
    snapshot_provider: Callable[
        [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
    ] | None,
) -> None:
    """Coalesce model placement checks away from request completion."""
    if snapshot_provider is None or controller._runtime_capabilities is None:
        return
    observation_generation = controller._runtime_learning_generation_sha256()
    queued = tuple(
        ticket for ticket in controller._runtime_controller.current_tickets()
        if ticket.dispatch_state in {"QUEUED", "REPLAN_REQUIRED"}
        and ticket.selection_mode != "desktop-baseline"
    )
    requested = set(request_ids)
    eligible = tuple(
        ticket for ticket in queued
        if not requested or ticket.request.request_id in requested
    )
    if not eligible:
        eligible = queued
    artifacts = tuple(sorted({
        ticket.model.artifact_sha256 for ticket in eligible
    }))
    if not artifacts:
        return
    if controller._runtime_epoch_refresh_executor is None:
        controller._runtime_epoch_refresh_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="unified-epoch-refresh",
        )
    for artifact_sha256 in artifacts:
        controller._runtime_epoch_refresh_requested[artifact_sha256] = (
            observation_generation,
            observed_at_us,
            snapshot_provider,
        )
        if artifact_sha256 in controller._runtime_epoch_refresh_inflight:
            continue
        controller._runtime_epoch_refresh_inflight.add(artifact_sha256)

        def refresh(artifact: str = artifact_sha256) -> None:
            try:
                while True:
                    with controller._runtime_lock:
                        work = controller._runtime_epoch_refresh_requested.get(
                            artifact
                        )
                    if work is None:
                        return
                    generation, refresh_at_us, provider = work
                    request_ids_for_artifact = tuple(
                        ticket.request.request_id
                        for ticket in (
                            controller._runtime_controller.current_tickets()
                        )
                        if ticket.model.artifact_sha256 == artifact
                        and ticket.dispatch_state in {
                            "QUEUED", "REPLAN_REQUIRED"
                        }
                        and ticket.selection_mode
                            != "desktop-baseline"
                    )
                    controller._run_model_placement_epoch_refresh_after_learning(
                        request_ids_for_artifact,
                        refresh_at_us,
                        provider,
                        artifact_sha256=artifact,
                    )
                    with controller._runtime_lock:
                        current = (
                            controller._runtime_epoch_refresh_requested.get(
                                artifact
                            )
                        )
                        if current is not None and current[0] == generation:
                            controller._runtime_epoch_refresh_requested.pop(
                                artifact, None
                            )
                            return
            finally:
                with controller._runtime_lock:
                    controller._runtime_epoch_refresh_inflight.discard(
                        artifact
                    )

        try:
            controller._runtime_epoch_refresh_executor.submit(refresh)
        except RuntimeError:
            controller._runtime_epoch_refresh_inflight.discard(
                artifact_sha256
            )
            controller._runtime_epoch_refresh_requested.pop(
                artifact_sha256, None
            )
            controller._runtime_epoch_background_refresh_failures += 1
            controller._record_epoch_refresh_failure(
                "BACKGROUND_REFRESH_EXECUTOR_UNAVAILABLE"
            )


def _placement_refresh_request_ids(
    controller,
    request_ids: Sequence[str],
    artifact_sha256: str | None,
) -> tuple[str, ...]:
    return tuple(sorted(
        set(request_ids) | {
            ticket.request.request_id
            for ticket in controller._runtime_controller.current_tickets()
            if ticket.dispatch_state in {
                "QUEUED", "REPLAN_REQUIRED"
            }
            and ticket.selection_mode != "desktop-baseline"
            and (
                artifact_sha256 is None
                or ticket.model.artifact_sha256 == artifact_sha256
            )
        }
    ))


def _placement_refresh_demand(
    controller,
    request_id: str,
    request_ids: tuple[str, ...],
    observed_at_us: int,
    snapshot_provider: Callable[
        [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
    ],
    seen_epochs: set[str],
) -> _PlacementRefreshDemand | None:
    ticket = controller.runtime_ticket(request_id)
    if ticket.dispatch_state not in {"QUEUED", "REPLAN_REQUIRED"}:
        return None
    manifest = controller.runtime_model_manifest(ticket.model.model_id)
    input_bucket, output_bucket = request_shape_bucket(
        ticket.request.input_tokens, ticket.request.output_tokens
    )
    stale_epoch = (
        controller._runtime_residency_cohorts
        .published_model_placement_epoch(
            manifest.artifact_sha256,
            input_bucket,
            output_bucket,
            ticket.request.quality_requirement,
            ticket.selection_mode,
            controller._runtime_capabilities.maximum_latency_ppm,
        )
    )
    if stale_epoch is None:
        stale_epoch = next(iter(
            controller._runtime_residency_cohorts
            .compatible_published_model_placement_epochs(
                manifest.artifact_sha256,
                ticket.request.quality_requirement,
                ticket.selection_mode,
                controller._runtime_capabilities.maximum_latency_ppm,
            )
        ), None)
    if (
        stale_epoch is None
        or stale_epoch.epoch_sha256 in seen_epochs
    ):
        return None
    templates = controller._runtime_route_template_sets.get(
        stale_epoch.epoch_sha256
    )
    if templates is None:
        controller._runtime_epoch_background_refresh_failures += 1
        return None
    projection_request_ids = (
        controller._runtime_controller.replan_projection_request_ids(
            ticket.request.request_id
        )
        if ticket.dispatch_state == "REPLAN_REQUIRED"
        else tuple(
            row
            for row in controller._runtime_controller.projection_request_ids()
            if row != ticket.request.request_id
        )
    )
    snapshot = snapshot_provider(ticket, observed_at_us)
    scheduling_at = max(
        ticket.request.arrival_us, snapshot.captured_at_us
    )
    snapshot.validate_at(scheduling_at)
    snapshot = controller._automated_snapshot_for_request(
        ticket.request,
        snapshot,
        exclude_request_id=ticket.request.request_id,
        project_before_us=ticket.decision.start_us,
        project_request_ids=projection_request_ids,
    )
    statistics = controller._runtime_residency_cohorts.model_placement_epoch(
        manifest.artifact_sha256,
        scheduling_at,
        request_id=ticket.request.request_id,
        queued_request_ids=request_ids,
    )
    seen_epochs.add(stale_epoch.epoch_sha256)
    return _PlacementRefreshDemand(
        ticket=ticket,
        manifest=manifest,
        snapshot=snapshot,
        stale_epoch=stale_epoch,
        templates=templates,
        reuse_projections=statistics.reuse_projections,
        expected_reuse_count=max(
            1,
            statistics.active_request_count
                + statistics.virtual_queue_request_count,
        ),
        observed_at_us=scheduling_at,
    )


def _collect_placement_refresh_demands(
    controller,
    request_ids: tuple[str, ...],
    observed_at_us: int,
    snapshot_provider: Callable[
        [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
    ],
) -> tuple[_PlacementRefreshDemand, ...]:
    demands = []
    seen_epochs = set()
    for request_id in request_ids:
        try:
            demand = controller._placement_refresh_demand(
                request_id,
                request_ids,
                observed_at_us,
                snapshot_provider,
                seen_epochs,
            )
        except (
            RuntimeCapabilityError,
            RuntimeControllerError,
            RuntimeResidencyCohortError,
            RuntimeResidencyProjectionError,
            UnifiedScheduleError,
        ):
            controller._runtime_epoch_background_refresh_failures += 1
            continue
        if demand is not None:
            demands.append(demand)
    return tuple(demands)


def _placement_refresh_generations(scheduler) -> tuple[object, ...]:
    return (
        scheduler._runtime_capability_generation_sha256,
        scheduler._runtime_profile_generation_sha256,
        scheduler._runtime_transport_generation_sha256,
        scheduler._runtime_learning_generation_sha256(),
    )


def _rerank_placement_refresh_demands(
    compiler: AutomatedRouteCompiler,
    observation_snapshot,
    demands: tuple[_PlacementRefreshDemand, ...],
) -> tuple[_PlacementRefreshResult, ...]:
    results = []
    try:
        compiler.restore_observations(observation_snapshot)
        for demand in demands:
            try:
                candidate_set, _ = compiler.rerank_route_template_set(
                    demand.templates,
                    demand.ticket.request,
                    demand.manifest,
                    demand.snapshot,
                    observed_at_us=demand.observed_at_us,
                    reuse_projections=demand.reuse_projections,
                    expected_reuse_count=demand.expected_reuse_count,
                )
                results.append(_PlacementRefreshResult(
                    demand, candidate_set, None
                ))
            except BaseException as exc:
                results.append(_PlacementRefreshResult(
                    demand, None, exc
                ))
    except BaseException as exc:
        results = [
            _PlacementRefreshResult(demand, None, exc)
            for demand in demands
        ]
    return tuple(results)


def _publish_placement_refresh(
    controller,
    demand: _PlacementRefreshDemand,
    candidate_set: AutomatedCandidateSet,
    compiler: AutomatedRouteCompiler,
    observation_generation_sha256: str,
) -> int | None:
    ticket = demand.ticket
    manifest = demand.manifest
    current = controller.runtime_ticket(ticket.request.request_id)
    if (
        current.model.artifact_sha256 != manifest.artifact_sha256
        or current.selection_mode != ticket.selection_mode
    ):
        return None
    controller._prepare_legacy_evidence_migrations(
        candidate_set, ticket.request, manifest
    )
    candidate_set = controller._apply_adaptive_history_costs(
        candidate_set,
        ticket.request,
        manifest,
        demand.snapshot.cost_features,
    )
    placement_key = controller._runtime_placement_learning_key(
        ticket.request, manifest, ticket.selection_mode
    )
    signature = controller._runtime_placement_learning_signature(
        candidate_set
    )
    if (
        controller._runtime_placement_learning_signature_by_key.get(
            placement_key
        ) == signature
    ):
        controller._model_placement_controller.mark_background_complete(
            manifest.artifact_sha256, demand.observed_at_us
        )
        return None
    learning_generation = canonical_sha256({
        "artifact_sha256": manifest.artifact_sha256,
        "observation_generation_sha256": (
            observation_generation_sha256
        ),
        "previous_generation_sha256": (
            controller._runtime_placement_learning_generation_sha256(
                manifest.artifact_sha256
            )
        ),
        "signature_sha256": signature,
        "schema": "runtime-placement-learning-generation-v2",
    })
    memory_rejections = controller._runtime_memory_rejections(
        candidate_set,
        demand.snapshot,
        exclude_owner_id=ticket.request.request_id,
    )
    selected, _, _ = controller._select_model_placement_candidate(
        candidate_set,
        ticket.request,
        demand.observed_at_us,
        selection_mode=ticket.selection_mode,
        runtime_rejections=memory_rejections,
        route_compiler=compiler,
    )
    epoch, refreshed_templates = controller._propose_model_placement_epoch(
        request=ticket.request,
        manifest=manifest,
        candidate_set=candidate_set,
        selected=selected,
        observed_at_us=demand.observed_at_us,
        selection_mode=ticket.selection_mode,
        invalidation_reason="LEARNING_GENERATION_CHANGED",
        snapshot=demand.snapshot,
        route_compiler=compiler,
        current_epoch=demand.stale_epoch,
        placement_learning_generation_sha256=learning_generation,
    )
    controller._publish_model_placement_epoch(epoch, refreshed_templates)
    controller._model_placement_controller.mark_background_complete(
        manifest.artifact_sha256, demand.observed_at_us
    )
    return len(candidate_set.candidates)


def _publish_placement_refresh_results(
    controller,
    results: tuple[_PlacementRefreshResult, ...],
    compiler: AutomatedRouteCompiler,
    target_generations: tuple[object, ...],
) -> None:
    if controller._placement_refresh_generations(controller) != target_generations:
        controller._runtime_epoch_background_refresh_failures += len(results)
        controller._record_epoch_refresh_failure(
            "BACKGROUND_REFRESH_SUPERSEDED"
        )
        controller._record_epoch_invalidation(
            "BACKGROUND_REFRESH_SUPERSEDED"
        )
        return
    for result in results:
        if result.error is not None or result.candidate_set is None:
            controller._runtime_epoch_background_refresh_failures += 1
            controller._record_epoch_refresh_failure(
                result.error or "CANDIDATE_SET_ABSENT"
            )
            continue
        try:
            count = controller._publish_placement_refresh(
                result.demand,
                result.candidate_set,
                compiler,
                target_generations[3],
            )
        except (
            RouteGenerationError,
            RuntimeCapabilityError,
            RuntimeControllerError,
            RuntimeResourceError,
            RuntimeResidencyCohortError,
            UnifiedScheduleError,
        ) as exc:
            controller._runtime_epoch_background_refresh_failures += 1
            controller._record_epoch_refresh_failure(exc)
            continue
        if count is None:
            continue
        controller._record_epoch_invalidation(
            "LEARNING_GENERATION_CHANGED"
        )
        controller._runtime_epoch_background_refreshes += 1
        controller._runtime_epoch_background_refresh_candidates += count


def _run_model_placement_epoch_refresh_after_learning(
    controller,
    request_ids: Sequence[str],
    observed_at_us: int,
    snapshot_provider: Callable[
        [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
    ],
    *,
    artifact_sha256: str | None = None,
) -> None:
    request_ids = controller._placement_refresh_request_ids(
        request_ids, artifact_sha256
    )
    if not request_ids or controller._runtime_capabilities is None:
        return
    started_ns = time.perf_counter_ns()
    demands = controller._collect_placement_refresh_demands(
        request_ids, observed_at_us, snapshot_provider
    )
    if not demands:
        controller._runtime_epoch_background_refresh_us += (
            time.perf_counter_ns() - started_ns
        ) // 1000
        return
    compiler = controller._runtime_epoch_route_compiler
    if compiler is None:
        controller._runtime_epoch_background_refresh_failures += len(demands)
        controller._runtime_epoch_background_refresh_us += (
            time.perf_counter_ns() - started_ns
        ) // 1000
        return
    observation_snapshot = (
        controller._automated_compiler().observation_checkpoint()
    )
    target_generations = controller._placement_refresh_generations(controller)
    results = controller._rerank_placement_refresh_demands(
        compiler, observation_snapshot, demands
    )
    with controller._runtime_lock:
        try:
            controller._publish_placement_refresh_results(
                results, compiler, target_generations
            )
        finally:
            controller._runtime_epoch_background_refresh_us += (
                time.perf_counter_ns() - started_ns
            ) // 1000
