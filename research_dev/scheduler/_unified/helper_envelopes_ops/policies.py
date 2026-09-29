"""HelperEnvelopeMixin policies operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_cost import RuntimeExecutorBinding
from ..._internal.model_placement_controller import (
    ModelPhoneResidencyLayout,
    RequestHelperEnvelopeBinding,
)
from ..._internal.runtime_plan import (
    AutomatedCandidateSet,
    HelperOpportunity,
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionPlan,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from ..common import _phone_shard_structure
from .common import _ReadyHelperParentUnavailable


def _build_helper_envelope(
    controller,
    *,
    artifact_sha256: str,
    desktop_parent_route_id: str,
    desktop_placement_sha256: str,
    helper_plan: RuntimeExecutionPlan,
    helper_binding: RuntimeExecutorBinding,
    layout: ModelPhoneResidencyLayout,
) -> RuntimeHelperExecutionEnvelope:
    changed_session_ids = (
        () if layout.state == "READY"
        else controller._helper_changed_session_ids(layout, helper_plan)
    )
    return RuntimeHelperExecutionEnvelope(
        artifact_sha256=artifact_sha256,
        desktop_parent_route_id=desktop_parent_route_id,
        desktop_placement_sha256=desktop_placement_sha256,
        helper_plan=helper_plan,
        helper_binding=helper_binding,
        phone_layout_generation=layout.generation,
        phone_layout_geometry_sha256=layout.layout.geometry_sha256,
        activation_dtype="f16",
        preparation_changed_session_ids=changed_session_ids,
        replacement_authorization=(
            controller._phone_session_replacement_authorization(layout)
            if len(changed_session_ids) == 1 else None
        ),
    )


def _request_helper_envelope_binding(
    helper: RuntimeHelperExecutionEnvelope,
    layout: ModelPhoneResidencyLayout,
) -> RequestHelperEnvelopeBinding:
    contract = helper.helper_plan.execution_contract
    return RequestHelperEnvelopeBinding(
        route_id=helper.route_id,
        operator_plan_sha256=helper.operator_plan_sha256,
        desktop_parent_route_id=helper.desktop_parent_route_id,
        desktop_placement_sha256=helper.desktop_placement_sha256,
        phone_layout_generation=layout.generation,
        phone_layout_geometry_sha256=layout.layout.geometry_sha256,
        activation_dtype=helper.activation_dtype,
        assisted_layer_mask=helper.resident_layer_mask,
        maximum_columns=helper.resident_columns,
        allowed_fractions_ppm=(
            contract.allowed_adaptive_fractions_ppm
        ),
        phone_session_ids=tuple(
            row.session_id for row in contract.phone_shards
        ),
        resource_ids=helper.helper_plan.resource_ids,
    )


def _ticket_adaptive_policy_records(
    ticket: RuntimeRequestTicket,
) -> tuple[AdaptiveDecodePolicy, tuple[AdaptiveDecodePolicy, ...]]:
    policies = []
    for estimate in ticket.cost_estimates.estimates:
        value = estimate.details.get("adaptive_decode_contract")
        if value is None:
            continue
        try:
            contract = dict(value)
            policies.append(AdaptiveDecodePolicy.from_json(contract))
            probes = contract.get(
                "adaptive_decode_probe_contracts", []
            )
            if type(probes) is not list:
                raise AdaptiveDecodeError(
                    "adaptive probe contracts are invalid"
                )
            policies.extend(
                AdaptiveDecodePolicy.from_json(row)
                for row in probes
            )
        except (TypeError, AdaptiveDecodeError) as exc:
            raise UnifiedScheduleError(str(exc)) from exc
    by_hash: dict[str, AdaptiveDecodePolicy] = {}
    for policy in policies:
        current = by_hash.get(policy.policy_hash)
        if current is not None and current != policy:
            raise UnifiedScheduleError(
                "adaptive policy identity collision"
            )
        by_hash[policy.policy_hash] = policy
    rows = tuple(by_hash.values())
    plan = ticket.execution_plan
    parent_executor = (
        plan.baseline_executor_id or ticket.binding.executor_id
    )
    baseline = tuple(
        row for row in rows if row.baseline
        and row.executor_id == parent_executor
        and row.desktop_placement_sha256 == plan.desktop_placement_sha256
    )
    if len(baseline) != 1:
        raise _ReadyHelperParentUnavailable(
            "adaptive ticket lacks one desktop baseline policy"
        )
    return baseline[0], tuple(row for row in rows if not row.baseline)


def _helper_opportunity_policies(
    controller,
    ticket: RuntimeRequestTicket,
    opportunity: HelperOpportunity,
) -> tuple[
    AdaptiveDecodePolicy,
    tuple[AdaptiveDecodePolicy, ...],
    AdaptiveDecodePolicy,
] | None:
    try:
        baseline, candidates = controller._ticket_adaptive_policy_records(
            ticket
        )
    except UnifiedScheduleError:
        return None
    plan = opportunity.helper_operator_plan
    parameters = plan.adapter_parameters
    resident_layer_mask = parameters.get("ffn_resident_layer_mask")
    resident_columns = parameters.get("ffn_resident_columns")
    binding = opportunity.helper_binding
    if (
        binding is None
        or binding.operator_plan_sha256 != plan.plan_sha256
        or binding.endpoint is None
        or binding.operator_plan_protocol is None
        or baseline.route_id != opportunity.desktop_parent_route_id
        or baseline.desktop_placement_sha256
            != opportunity.desktop_parent_placement_sha256
        or type(resident_layer_mask) is not int
        or resident_layer_mask <= 0
        or type(resident_columns) is not int
        or resident_columns <= 0
    ):
        return None
    selected = tuple(
        row for row in candidates
        if row.route_id == opportunity.route_id
        and row.executor_id == binding.executor_id
        and row.operator_plan_sha256 == plan.plan_sha256
    )
    if len(selected) != 1:
        return None
    allowed = set(
        plan.execution_contract.allowed_adaptive_fractions_ppm
    )
    policies = tuple(sorted(
        (
            row for row in candidates
            if row.executor_id == binding.executor_id
            and row.operator_plan_sha256 == plan.plan_sha256
            and row.desktop_parent_route_id
                == opportunity.desktop_parent_route_id
            and row.desktop_placement_sha256
                == opportunity.desktop_parent_placement_sha256
            and row.layer_mask & ~resident_layer_mask == 0
            and row.columns <= resident_columns
            and row.split_fraction_ppm in allowed
            and row.split_fraction_ppm
                <= opportunity.maximum_allowed_fraction_ppm
        ),
        key=lambda row: (
            row.split_fraction_ppm,
            row.layer_mask,
            row.policy_hash,
        ),
    ))
    if not policies:
        return None
    return baseline, policies, selected[0]


def _candidate_set_for_ready_phone_layout(
    candidate_set: AutomatedCandidateSet,
    artifact_sha256: str,
    layout: ModelPhoneResidencyLayout,
    *, usb_batch_plan: str | None = None,
) -> AutomatedCandidateSet:
    expected_shards = tuple(
        _phone_shard_structure(row) for row in layout.layout.shards
        if row.artifact_sha256 == artifact_sha256
    )
    required_route_ids = {
        candidate_set.baseline_route_id,
        candidate_set.recovery_fallback_route_id
            or candidate_set.baseline_route_id,
    }
    candidates = tuple(
        row for row in candidate_set.candidates
        if row.candidate_id in required_route_ids
        or (
            row.assisted_operator_kind == "ffn"
            and (usb_batch_plan is None
                 or row.plan.adapter_parameters.get("usb_batch_plan") == usb_batch_plan)
            and row.plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            ) == layout.layout.geometry_sha256
            and tuple(
                _phone_shard_structure(shard) for shard in
                row.plan.execution_contract.phone_shards
            ) == expected_shards
        )
    )
    search_metadata = dict(candidate_set.search_metadata)
    if search_metadata:
        search_metadata["evaluated_plan_count"] = len(candidates)
        search_metadata["visited_plan_ids"] = tuple(
            row.candidate_id for row in candidates
        )
    return replace(
        candidate_set,
        candidates=candidates,
        search_metadata=search_metadata,
    )


def _authorize_phone_helper_plan(
    controller,
    plan: RuntimeExecutionPlan,
    binding: RuntimeExecutorBinding,
    layout: ModelPhoneResidencyLayout,
    *,
    model_id: str,
    artifact_sha256: str,
) -> tuple[RuntimeExecutionPlan, RuntimeExecutorBinding]:
    authorized = (
        controller._automated_compiler().authorize_phone_residency_plan(
            plan,
            controller._phone_helper_authorization_layout(layout),
            model_id=model_id,
            artifact_sha256=artifact_sha256,
        )
    )
    return authorized, replace(
        binding, operator_plan_sha256=authorized.plan_sha256
    )
