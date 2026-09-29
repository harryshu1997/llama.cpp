"""AutomatedSelectionMixin dormant operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.model_manifest import ModelManifest
from ..._internal.plan_contracts.co_helpers import (
    PHONE_HELPERS_PARAMETER,
    phone_helper_layer_masks,
    phone_helpers_with_primary_layers,
)
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionPlan,
)
from ..common import _DORMANT_PHONE_FFN_RUNTIME_PARAMETER, _DORMANT_PHONE_FFN_RUNTIME_KEYS


def _required_dormant_phone_ffn_keys(transport: object) -> set[str]:
    required_dormant = {
        "ffn_activation",
        "ffn_assistance_phase",
        "ffn_max_tokens",
        "ffn_n_embd",
        "ffn_resident_columns",
        "ffn_resident_layer_mask",
        "ffn_runtime_control_protocol",
        "ffn_timeout_ms",
        "ffn_transport",
        "phone_device_id",
    }
    if transport == "functionfs-usb":
        required_dormant.update({
            "usb_allocator",
            "usb_batch_plan",
            "usb_full_duplex",
            "usb_max_payload_bytes",
            "usb_product_id",
            "usb_queue_depth",
            "usb_slot_safety_bytes",
            "usb_split_h2d",
            "usb_transport_generation",
            "usb_transport_profile_id",
            "usb_vendor_id",
            "usbfs_available_bytes",
        })
    elif transport == "tcp":
        required_dormant.update({
            "bridge_allocator",
            "bridge_queue_depth",
            "ffn_bridge_host",
            "ffn_bridge_port",
        })
    return required_dormant


def _dormant_phone_ffn_storage_superset(
    controller,
    parameters: Mapping[str, object],
    desktop: RuntimeExecutionPlan,
    manifest: ModelManifest,
) -> dict[str, object]:
    """Allow later READY slices without changing the desktop launch."""

    result = dict(parameters)
    cpu_device_id = desktop.adapter_parameters.get("cpu_device_id")
    cpu_operator_ids = {
        row.operator_id for row in desktop.operators
        if row.operator_kind == "ffn"
        and row.device_ids == (cpu_device_id,)
        and row.split_axis == "none"
    }
    cpu_mask = 0
    for operator in manifest.operators:
        if operator.operator_id in cpu_operator_ids:
            prefix, separator, index = operator.layer_id.partition(":")
            if prefix == "layer" and separator and index.isdecimal():
                cpu_mask |= 1 << int(index)
    stored_mask = 0
    for row in controller._phone_ffn_shard_storage:
        if (
            row.parent_artifact_sha256 == manifest.artifact_sha256
            and row.maximum_columns >= result["ffn_resident_columns"]
        ):
            stored_mask |= row.layer_mask
    helpers = result.get(PHONE_HELPERS_PARAMETER)
    if helpers is None:
        result["ffn_resident_layer_mask"] |= stored_mask & cpu_mask
        return result
    # the stored slices belong to the ticket's phone; a co-helper keeps its own layers
    masks = tuple(phone_helper_layer_masks(helpers).values())
    primary = masks[0] | stored_mask & cpu_mask & ~sum(masks[1:])
    result[PHONE_HELPERS_PARAMETER] = phone_helpers_with_primary_layers(
        helpers, primary
    )
    result["ffn_resident_layer_mask"] = primary | sum(masks[1:])
    return result


def _complete_dormant_phone_ffn_parameters(
    controller,
    *parameter_sets: Mapping[str, object],
) -> dict[str, object] | None:
    for parameters in parameter_sets:
        dormant = {
            name: value
            for name, value in parameters.items()
            if name in _DORMANT_PHONE_FFN_RUNTIME_KEYS
        }
        required = controller._required_dormant_phone_ffn_keys(
            dormant.get("ffn_transport")
        )
        if required <= set(dormant):
            return dormant
    return None


def _dormant_phone_ffn_parent_parameters(
    candidate_set: AutomatedCandidateSet,
    baseline: AutomatedRouteCandidate,
    manifest: ModelManifest,
    qualified_executor_ids: frozenset[str] | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Enable zero-assistance control without authorizing phone execution.

    The dormant runtime is fixed for the life of the desktop server it
    launches, and a later READY helper must match its batch plan (server
    policy coherence).  When every helper route is transiently rejected (e.g.
    THERMAL_LIMIT) this fallback alone picks the runtime, so a helper whose
    coordinator (or route) is QUALIFIED ranks before one that merely sorts
    first: ``operator_split:750000`` (split-row) < ``operator_split:
    coalesced-batch:750000`` would otherwise lock a coalesced-only campaign
    into an unqualified split-row server for the whole model phase.  Without
    ``qualified_executor_ids`` (or when every row qualifies) the order is
    unchanged.
    """

    def unqualified(row: AutomatedRouteCandidate) -> bool:
        return (
            qualified_executor_ids is not None
            and row.binding.executor_id not in qualified_executor_ids
            and row.maturity != "QUALIFIED"
        )

    candidates = (
        row for row in candidate_set.candidates
        if row.assisted_operator_kind == "ffn"
        and row.route_family == "operator_split"
        and row.paired_baseline_route_id == baseline.candidate_id
        and row.binding.artifact_sha256 == manifest.artifact_sha256
        and row.plan.baseline_executor_id == baseline.binding.executor_id
        and row.plan.desktop_placement_sha256
            == baseline.plan.desktop_placement_sha256
        and row.plan.execution_contract.execution_mode == "adaptive-split"
        and row.plan.adapter_parameters.get("ffn_runtime_control_protocol")
            == "decode-boundary-v1"
        and "TRANSPORT_PROFILE_INCOMPLETE" not in row.rejection_reasons
    )
    return tuple(row.plan.adapter_parameters for row in sorted(
        candidates,
        key=lambda row: (
            unqualified(row),
            -int(row.plan.adapter_parameters.get("ffn_resident_layer_mask", 0)).bit_count(),
            -int(row.plan.adapter_parameters.get("ffn_resident_columns", 0)),
            row.candidate_id,
        ),
    ))


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
    rejected_by_route = dict(rejected)
    helper_route_id = (
        envelope.candidate_id
        if envelope is not None else
        retained_helper.route_id
        if retained_helper is not None else None
    )
    if helper_route_id is not None:
        rejected_by_route[helper_route_id] = (
            "RESIDENT_DESKTOP_HELPER_RUNTIME_UNAVAILABLE"
        )
    return (
        candidate_set,
        baseline,
        tuple(sorted(rejected_by_route.items())),
        "READY_DESKTOP_HELPER_RUNTIME_UNAVAILABLE",
    )


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
    base_parameters = dict(baseline.plan.adapter_parameters)
    base_parameters[_DORMANT_PHONE_FFN_RUNTIME_PARAMETER] = (
        encoded_dormant
    )
    base_plan = replace(
        baseline.plan,
        adapter_parameters=base_parameters,
    )
    baseline = replace(
        baseline,
        plan=base_plan,
        binding=replace(
            baseline.binding,
            operator_plan_sha256=base_plan.plan_sha256,
        ),
    )
    candidate_set = replace(
        candidate_set,
        candidates=tuple(
            baseline
            if row.candidate_id == baseline.candidate_id else row
            for row in candidate_set.candidates
        ),
    )
    if selected.candidate_id == baseline.candidate_id:
        selected = baseline
    return candidate_set, baseline, selected
