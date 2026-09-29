"""AutomatedSelectionMixin helpers operations on its existing owner."""

from __future__ import annotations

import json

from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeHelperExecutionEnvelope,
)
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError
from ..._internal.adaptive_decode_planning import (
    adaptive_candidate_set_for_parent,
    adaptive_decode_policies,
)


def _qualified_helper_executor_ids(controller) -> frozenset[str] | None:
    """Composite helper executors a dormant desktop runtime may serve later."""

    catalog = getattr(controller, "_runtime_capabilities", None)
    if catalog is None:
        return None
    return frozenset(
        row.executor_id
        for row in catalog.composite_executors
        if row.maturity == "QUALIFIED"
    )


def _desktop_with_async_phone_helper(
    controller,
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

    if selection_mode in {"desktop-baseline", "deadline-first"}:
        return candidate_set, selected, rejected, reason
    baseline = (
        selected
        if selected.plan.execution_contract.execution_mode == "desktop"
        else candidate_set.baseline
    )
    if baseline.plan.execution_contract.execution_mode != "desktop":
        return candidate_set, selected, rejected, reason
    try:
        _baseline_policy, policies, envelope = adaptive_decode_policies(
            adaptive_candidate_set_for_parent(candidate_set, baseline),
            manifest,
            controller._runtime_capabilities,
            request.output_tokens,
            controller._maximum_phone_sessions,
        )
    except AdaptiveDecodeError:
        return candidate_set, selected, rejected, reason
    retained_helper = None
    if envelope is None:
        retained_helper = controller._retained_request_helper_for_baseline(
            request, manifest, baseline
        )
    ready_helper = controller._authoritative_ready_helper_template(
        artifact_sha256=manifest.artifact_sha256,
        desktop_parent_route_id=baseline.candidate_id,
        desktop_placement_sha256=(
            baseline.plan.desktop_placement_sha256 or ""
        ),
        baseline_executor_id=baseline.binding.executor_id,
        allow_parent_route_rebind=True,
    )
    dormant_parameters = controller._complete_dormant_phone_ffn_parameters(
        *(
            (ready_helper.helper_plan.adapter_parameters,)
            if ready_helper is not None else ()
        ),
        *(
            (envelope.plan.adapter_parameters,)
            if envelope is not None else ()
        ),
        *(
            (retained_helper.helper_plan.adapter_parameters,)
            if retained_helper is not None else ()
        ),
        *controller._dormant_phone_ffn_parent_parameters(
            candidate_set,
            baseline,
            manifest,
            _qualified_helper_executor_ids(controller),
        ),
    )
    if dormant_parameters is not None:
        dormant_parameters = controller._dormant_phone_ffn_storage_superset(
            dormant_parameters, baseline.plan, manifest
        )
        encoded_dormant = json.dumps(
            dormant_parameters,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if (
            baseline.plan.residency_variant in {"hot", "warm"}
            and not controller._dormant_phone_ffn_runtime_supports(
                baseline.plan, encoded_dormant
            )
        ):
            return controller._dormant_helper_runtime_unavailable(
                candidate_set,
                baseline,
                rejected,
                envelope,
                retained_helper,
            )
        candidate_set, baseline, selected = (
            controller._baseline_with_dormant_phone_ffn(
                candidate_set, baseline, selected, encoded_dormant
            )
        )
    layout = controller._model_placement_controller.planning_phone_layout()
    if envelope is None or not policies or layout is None:
        return controller._defer_unready_phone_selection(
            candidate_set, baseline, selected, rejected, reason
        )
    if not controller._async_helper_layout_matches(
        envelope, baseline, layout, manifest
    ):
        return controller._defer_unready_phone_selection(
            candidate_set, baseline, selected, rejected, reason
        )
    if not controller._async_helper_preparation_authorized(
        baseline, envelope, selected, layout, manifest
    ):
        return controller._defer_unready_phone_selection(
            candidate_set, baseline, selected, rejected, reason
        )
    if baseline.plan.desktop_placement_sha256 is None:
        raise UnifiedScheduleError(
            "desktop helper parent lacks its placement identity"
        )
    return controller._bind_async_phone_helper(
        candidate_set,
        baseline,
        selected,
        rejected,
        envelope,
        layout,
        manifest,
    )


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
    if (
        selected.candidate_id == baseline.candidate_id
        or not selected.plan.execution_contract.phone_shards
    ):
        return candidate_set, selected, rejected, reason
    rejected_by_route = dict(rejected)
    rejected_by_route[selected.candidate_id] = (
        "ASYNC_HELPER_PREPARATION"
    )
    rejected_by_route.pop(baseline.candidate_id, None)
    return (
        candidate_set,
        baseline,
        tuple(sorted(rejected_by_route.items())),
        "READY_DESKTOP_WITH_ASYNC_PHONE_HELPER",
    )


def _retained_request_helper_for_baseline(
    controller,
    request: Request,
    manifest: ModelManifest,
    baseline: AutomatedRouteCandidate,
) -> RuntimeHelperExecutionEnvelope | None:
    """Find the one prepared helper still bound to this desktop parent."""
    layout = (
        controller._model_placement_controller.planning_phone_layout()
    )
    if layout is None or layout.state not in {
        "PROPOSED", "PREPARING", "READY"
    }:
        return None
    expected_shards = tuple(
        row.to_json() for row in layout.layout.shards
        if row.artifact_sha256 == manifest.artifact_sha256
    )
    matches = {
        helper.operator_plan_sha256: helper
        for key, helper in (
            controller._request_helper_preparation_envelopes.items()
        )
        if (
            key[0] == request.request_id
            and helper.artifact_sha256
                == manifest.artifact_sha256
            and helper.desktop_parent_route_id
                == baseline.candidate_id
            and helper.desktop_placement_sha256
                == baseline.plan.desktop_placement_sha256
            and helper.helper_plan.baseline_executor_id
                == baseline.binding.executor_id
            and helper.helper_binding.artifact_sha256
                == manifest.artifact_sha256
            and helper.phone_layout_generation
                == layout.generation
            and helper.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and tuple(
                row.to_json() for row in
                helper.helper_plan.execution_contract.phone_shards
            ) == expected_shards
        )
    }
    if len(matches) == 1:
        return next(iter(matches.values()))
    return None
