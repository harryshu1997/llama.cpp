"""AutomatedCandidateMixin observations operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import RuntimeCapabilityCatalog
from ..._internal.route_generation import RouteGenerationError
from ..._internal.runtime_learning import RuntimeLearningError
from ..._internal.runtime_plan import RuntimeExecutionPlan
from ..._internal.types import canonical_sha256


def runtime_residency_cohort_state(controller) -> Mapping[str, object]:
    return controller._runtime_residency_cohorts.snapshot()


def automated_observation_state(controller) -> Mapping[str, int]:
    return controller._automated_compiler().observation_state()


def automated_observation_snapshot(controller) -> Mapping[str, object]:
    return controller._automated_compiler().observation_export()


def load_automated_observations(
    controller,
    value: object,
    *,
    source_catalog: RuntimeCapabilityCatalog | None = None,
) -> None:
    try:
        controller._automated_compiler().import_observations(value)
        if source_catalog is not None:
            if not isinstance(source_catalog, RuntimeCapabilityCatalog):
                raise RuntimeLearningError(
                    "runtime observation source catalog is invalid"
                )
            controller._automated_observation_sources[
                canonical_sha256(source_catalog)
            ] = source_catalog
        for manifest in controller._runtime_manifests.values():
            controller._model_placement_controller.notify(
                manifest.artifact_sha256,
                "LEARNING_GENERATION_CHANGED",
                0,
            )
    except RuntimeLearningError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def merge_automated_observations(controller, value: object) -> None:
    try:
        controller._automated_compiler().merge_observations(value)
        for manifest in controller._runtime_manifests.values():
            controller._model_placement_controller.notify(
                manifest.artifact_sha256,
                "LEARNING_GENERATION_CHANGED",
                0,
            )
    except RuntimeLearningError as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def rebind_legacy_automated_component_observations(
    controller,
    source_catalog: RuntimeCapabilityCatalog,
    model_id: str,
    source_plan: RuntimeExecutionPlan,
    executor_id: str,
    target_plan: RuntimeExecutionPlan | None = None,
) -> int:
    """Migrate explicitly verified execution-component observations."""
    manifest = controller.runtime_model_manifest(model_id)
    try:
        return controller._automated_compiler(
        ).rebind_legacy_component_observations(
            source_catalog,
            manifest,
            source_plan,
            executor_id,
            target_plan,
        )
    except (RouteGenerationError, RuntimeLearningError) as exc:
        raise UnifiedScheduleError(str(exc)) from exc


def runtime_decision_timings(controller) -> tuple[Mapping[str, object], ...]:
    return tuple(
        MappingProxyType({
            key: (
                dict(value) if isinstance(value, Mapping) else value
            )
            for key, value in row.items()
        })
        for row in controller._runtime_decision_timings
    )
