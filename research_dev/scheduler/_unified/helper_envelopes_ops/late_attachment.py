"""HelperEnvelopeMixin late attachment operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.route_generation import RouteGenerationError
from ..._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
    RequestHelperEnvelopeBinding,
)
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from ..._internal.runtime_residency_cohorts import (
    RuntimeResidencyComponentIdentity,
    RuntimeResidencyCohortError,
    runtime_residency_component_identity,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from ..common import _LateRequestHelperContext, _phone_shard_structure


def _validated_cached_late_request_helper(
    controller,
    request_id: str,
    cached: _LateRequestHelperContext,
) -> _LateRequestHelperContext | None:
    try:
        state = controller._model_placement_controller.phone_layout(
            cached.helper.phone_layout_generation
        )
    except ModelPlacementControllerError:
        return None
    envelope = controller._request_helper_envelope_binding(
        cached.helper, state
    )
    if (
        state.layout.geometry_sha256
            == cached.helper.phone_layout_geometry_sha256
        and (
            controller._model_placement_controller
                .request_helper_layout_is_usable(
                    request_id,
                    cached.helper.phone_layout_generation,
                    cached.helper.phone_layout_geometry_sha256,
                )
            or controller._model_placement_controller
                .request_helper_envelope_is_additive(
                    request_id, envelope
                )
        )
    ):
        return cached
    return None


def _materialize_late_request_helper(
    controller,
    ticket: RuntimeRequestTicket,
) -> _LateRequestHelperContext | None:
    request_id = ticket.request.request_id
    cached = controller._late_request_helper_contexts.get(request_id)
    if cached is not None:
        validated = controller._validated_cached_late_request_helper(
            request_id, cached
        )
        if validated is not None:
            return validated
        controller._late_request_helper_contexts.pop(request_id, None)
    plan = ticket.execution_plan
    opportunities = controller._request_helper_opportunities.get(request_id)
    if (
        plan is None
        or not opportunities
        or not controller._has_dormant_phone_ffn_runtime(plan)
    ):
        return None
    layout = controller._model_placement_controller.ready_phone_layout()
    if (
        layout is None
        or not layout.covers_artifact(ticket.model.artifact_sha256)
    ):
        return None
    expected_shards = [
        _phone_shard_structure(row)
        for row in layout.layout.shards
        if row.artifact_sha256 == ticket.model.artifact_sha256
    ]
    exact = tuple(
        opportunity
        for opportunity in opportunities
        if (
            opportunity.desktop_parent_route_id
                == ticket.decision.route_id
            and opportunity.desktop_parent_placement_sha256
                == plan.desktop_placement_sha256
            and opportunity.phone_layout_generation
                in {None, layout.generation}
            and opportunity.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and [
                _phone_shard_structure(shard)
                for shard in opportunity.helper_operator_plan
                    .execution_contract.phone_shards
            ] == expected_shards
        )
    )
    if len(exact) != 1:
        return None
    opportunity = exact[0]
    policy_rows = controller._helper_opportunity_policies(ticket, opportunity)
    if policy_rows is None:
        return None
    baseline_policy, policies, ticket_policy = policy_rows
    if plan.desktop_placement_sha256 is None:
        return None
    helper_plan = opportunity.helper_operator_plan
    helper_binding = opportunity.helper_binding
    try:
        helper_plan, helper_binding = (
            controller._authorize_phone_helper_plan(
                helper_plan,
                helper_binding,
                layout,
                model_id=ticket.model.model_id,
                artifact_sha256=ticket.model.artifact_sha256,
            )
        )
        policies = tuple(
            replace(
                row,
                operator_plan_sha256=helper_plan.plan_sha256,
            )
            for row in policies
        )
        matching_ticket_policies = tuple(
            row for row in policies
            if row.route_id == ticket_policy.route_id
        )
        if len(matching_ticket_policies) != 1:
            return None
        ticket_policy = matching_ticket_policies[0]
        helper = controller._build_helper_envelope(
            artifact_sha256=ticket.model.artifact_sha256,
            desktop_parent_route_id=(
                opportunity.desktop_parent_route_id
            ),
            desktop_placement_sha256=(
                plan.desktop_placement_sha256
            ),
            helper_plan=helper_plan,
            helper_binding=helper_binding,
            layout=layout,
        )
        component = runtime_residency_component_identity(
            ticket.model.artifact_sha256,
            helper_plan,
            helper_binding,
        )
        contract = helper.helper_plan.execution_contract
        controller._model_placement_controller.bind_request_helper_envelope(
            request_id,
            RequestHelperEnvelopeBinding(
                route_id=helper.route_id,
                operator_plan_sha256=helper.operator_plan_sha256,
                desktop_parent_route_id=(
                    helper.desktop_parent_route_id
                ),
                desktop_placement_sha256=(
                    helper.desktop_placement_sha256
                ),
                phone_layout_generation=(
                    helper.phone_layout_generation
                ),
                phone_layout_geometry_sha256=(
                    helper.phone_layout_geometry_sha256
                ),
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
            ),
            observed_at_us=max(
                ticket.dispatch_receipt.observed_at_us,
                layout.ready_at_us or 0,
            ),
        )
    except (
        AdaptiveDecodeError,
        ModelPlacementControllerError,
        RouteGenerationError,
        RuntimeResidencyCohortError,
    ):
        return None
    return controller._store_late_request_helper(
        request_id,
        helper,
        baseline_policy,
        policies,
        ticket_policy,
        component,
        opportunity.evidence_state,
    )


def _store_late_request_helper(
    controller,
    request_id: str,
    helper: RuntimeHelperExecutionEnvelope,
    baseline_policy: AdaptiveDecodePolicy,
    policies: tuple[AdaptiveDecodePolicy, ...],
    ticket_policy: AdaptiveDecodePolicy,
    component: RuntimeResidencyComponentIdentity,
    evidence_state: str,
) -> _LateRequestHelperContext:
    result = _LateRequestHelperContext(
        helper=helper,
        baseline=baseline_policy,
        candidates=policies,
        ticket_policy=ticket_policy,
        component=component,
        evidence_state=evidence_state,
    )
    controller._late_request_helper_contexts[request_id] = result
    controller._remember_request_helper_envelope(request_id, helper)
    return result


def _remember_request_helper_envelope(
    controller,
    request_id: str,
    helper: RuntimeHelperExecutionEnvelope,
) -> None:
    by_plan = controller._request_helper_envelope_history.setdefault(
        request_id, {}
    )
    previous = by_plan.get(helper.operator_plan_sha256)
    if previous is not None and previous != helper:
        raise UnifiedScheduleError(
            "runtime helper plan history changed identity"
        )
    by_plan[helper.operator_plan_sha256] = helper
    controller._remember_phone_helper_endpoint_template(helper)


def _phone_helper_endpoint_identity(
    helper: RuntimeHelperExecutionEnvelope,
) -> tuple[object, ...]:
    binding = helper.helper_binding
    return (
        helper.artifact_sha256,
        helper.desktop_placement_sha256,
        helper.helper_plan.baseline_executor_id,
        binding.executor_id,
        binding.endpoint,
        binding.backend,
        binding.operator_plan_protocol,
        tuple(
            (
                participant.executor_id,
                participant.device_id,
                participant.endpoint,
                participant.backend,
                participant.resource_ids,
            )
            for participant in binding.participants
        ),
    )


def _phone_helper_endpoint_template_key(
    _controller_class,
    helper: RuntimeHelperExecutionEnvelope,
) -> tuple[object, ...]:
    shards = helper.helper_plan.execution_contract.phone_shards
    return (
        *_controller_class._phone_helper_endpoint_identity(helper),
        helper.phone_layout_geometry_sha256,
        tuple(
            (
                *_phone_shard_structure(shard),
                shard.session_generation,
            )
            for shard in shards
        ),
    )


def _remember_phone_helper_endpoint_template(
    controller,
    helper: RuntimeHelperExecutionEnvelope,
) -> None:
    helper = controller._reusable_phone_helper_template(helper)
    key = controller._phone_helper_endpoint_template_key(helper)
    templates = getattr(controller, "_phone_helper_endpoint_templates", None)
    if templates is None:
        templates = {}
        controller._phone_helper_endpoint_templates = templates
    previous = templates.get(key)
    if previous is None or (
        helper.phone_layout_generation,
        helper.operator_plan_sha256,
    ) >= (
        previous.phone_layout_generation,
        previous.operator_plan_sha256,
    ):
        templates[key] = helper


def _request_helper_envelope(
    controller,
    ticket: RuntimeRequestTicket,
) -> RuntimeHelperExecutionEnvelope | None:
    plan = ticket.execution_plan
    if plan is None:
        return None
    context = controller._materialize_late_request_helper(ticket)
    if context is not None:
        return context.helper
    if plan.helper_envelope is not None:
        controller._remember_request_helper_envelope(
            ticket.request.request_id, plan.helper_envelope
        )
        return plan.helper_envelope
    return None


def _ready_request_helper(
    controller,
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope | None = None,
) -> tuple[
    RuntimeHelperExecutionEnvelope,
    ModelPhoneResidencyLayout,
    RuntimeResidencyComponentIdentity,
] | None:
    plan = ticket.execution_plan
    if plan is None:
        return None
    if helper is None:
        helper = controller._request_helper_envelope(ticket)
    if helper is None:
        return None
    request_binding = (
        controller._model_placement_controller.request_binding(
            ticket.request.request_id
        )
    )
    request_attachment = (
        None if request_binding is None else
        request_binding.get("helper_attachment")
    )
    if (
        isinstance(request_attachment, Mapping)
        and request_attachment.get("fallback_outcome")
            == "PHONE_LAYOUT_TRANSITION_FAILED"
    ):
        return None
    try:
        layout = controller._model_placement_controller.phone_layout(
            helper.phone_layout_generation
        )
    except ModelPlacementControllerError:
        return None
    envelope_binding = controller._request_helper_envelope_binding(
        helper, layout
    )
    layout_is_usable = (
        controller._model_placement_controller
            .request_helper_layout_is_usable(
                ticket.request.request_id,
                helper.phone_layout_generation,
                helper.phone_layout_geometry_sha256,
            )
        or controller._model_placement_controller
            .request_helper_envelope_is_additive(
                ticket.request.request_id, envelope_binding
            )
    )
    if (
        not layout_is_usable
        or layout.layout.geometry_sha256
            != helper.phone_layout_geometry_sha256
        or not layout.covers_artifact(ticket.model.artifact_sha256)
        or helper.artifact_sha256 != ticket.model.artifact_sha256
        or helper.desktop_parent_route_id != plan.route_id
        or helper.desktop_placement_sha256
            != plan.desktop_placement_sha256
        or [_phone_shard_structure(row) for row in (
            helper.helper_plan.execution_contract.phone_shards
        )] != [
            _phone_shard_structure(row)
            for row in layout.layout.shards
            if row.artifact_sha256
                == ticket.model.artifact_sha256
        ]
    ):
        return None
    try:
        component = runtime_residency_component_identity(
            ticket.model.artifact_sha256,
            helper.helper_plan,
            helper.helper_binding,
        )
    except RuntimeResidencyCohortError:
        return None
    return helper, layout, component
