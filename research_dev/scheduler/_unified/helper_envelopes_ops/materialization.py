"""HelperEnvelopeMixin materialization operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import json

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot, RuntimeExecutorState
from ..._internal.model_placement_controller import ModelPhoneResidencyLayout
from ..._internal.runtime_plan import AutomatedCandidateSet, HelperOpportunity
from ..._internal.runtime_residency_cohorts import runtime_residency_component_identity
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.adaptive_decode_contracts import AdaptiveDecodeError, AdaptiveDecodePolicy
from ..._internal.adaptive_decode_planning import adaptive_decode_policies
from ..common import _ReadyHelperSafetyDeferred, _LateRequestHelperContext, _phone_shard_structure
from .common import _ReadyHelperMaterialization, _ReadyHelperParentUnavailable
from .refresh import retained_execution_layer_mask


def _dormant_batch_plan_note(
    controller,
    generated: AutomatedCandidateSet,
    artifact_sha256: str,
    layout: ModelPhoneResidencyLayout,
    batch_plan: str | None,
) -> str:
    """Name the coherence filter when it, not the layout, removed the helper.

    The listed rejection reasons are all ones adaptive_decode_policies admits,
    so without this note a READY layout whose geometry matches looks like an
    unexplained miss.  The usual cause is a desktop server launched with a
    dormant FFN runtime of an unqualified batch plan: its requests can only
    use that plan until the server is relaunched.
    """

    if batch_plan is None:
        return ""
    catalog = controller._runtime_capabilities
    unfiltered = controller._candidate_set_for_ready_phone_layout(
        generated, artifact_sha256, layout
    )
    qualified_by_plan: dict[str, bool] = {}
    for row in unfiltered.candidates:
        if row.assisted_operator_kind != "ffn":
            continue
        plan = str(row.plan.adapter_parameters.get("usb_batch_plan"))
        coordinator = (
            None if catalog is None else
            catalog.composite_executor_by_id.get(row.binding.executor_id)
        )
        qualified = bool(
            row.maturity == "QUALIFIED"
            or (
                coordinator is not None
                and coordinator.maturity == "QUALIFIED"
            )
        )
        qualified_by_plan[plan] = qualified_by_plan.get(plan, False) or qualified
    return (
        "; desktop dormant usb_batch_plan="
        + repr(batch_plan)
        + " (helper executor qualified: "
        + repr(qualified_by_plan.get(batch_plan, False))
        + "); READY-geometry batch plans: "
        + repr(tuple(sorted(qualified_by_plan.items())))
    )


def _ready_helper_candidate_records(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    phone_safety_state: RuntimeExecutorState | None,
) -> tuple[
    object,
    AutomatedCandidateSet,
    HelperOpportunity,
    object,
    str,
    AdaptiveDecodePolicy,
]:
    request_id = ticket.request.request_id
    manifest = controller.runtime_model_manifest(ticket.model.model_id)
    request_snapshot = controller._automated_snapshot_for_request(
        ticket.request,
        snapshot,
        exclude_request_id=request_id,
        project_request_ids=(),
    )
    request_snapshot = controller._snapshot_with_owned_base_executor(
        ticket, request_snapshot
    )
    request_snapshot = controller._snapshot_with_verified_ready_helper(
        ticket, layout, request_snapshot, phone_safety_state
    )
    batch_plan = None
    if controller._adaptive_decode_config.server_policy_coherence:
        dormant = ticket.execution_plan.adapter_parameters.get("dormant_phone_ffn_runtime_v1")
        if dormant:
            batch_plan = json.loads(dormant).get("usb_batch_plan")
    with controller._ready_layout_compiler_view(layout):
        generated = controller._generate_automated_candidate_set(
            ticket.request,
            manifest,
            request_snapshot,
            observed_at_us,
            use_residency_holds=False,
            update_phone_residency_portfolio=False,
            desktop_parent=(
                ticket.binding.executor_id,
                ticket.execution_plan.desktop_placement_sha256,
            ),
        )
        candidate_set = controller._candidate_set_for_ready_phone_layout(
            generated, manifest.artifact_sha256, layout, usb_batch_plan=batch_plan
        )
        opportunities = controller._compact_helper_opportunities(
            candidate_set, ticket.request
        )
    if not opportunities:
        diagnostics = tuple(
            (row.candidate_id, row.rejection_reasons)
            for row in candidate_set.candidates
            if row.assisted_operator_kind == "ffn"
        )
        raise UnifiedScheduleError(
            "ready layout produced no helper opportunity: "
            + repr(diagnostics)
            + _dormant_batch_plan_note(
                controller, generated, manifest.artifact_sha256, layout,
                batch_plan,
            )
        )
    generated_baseline = candidate_set.baseline
    exact_geometry = tuple(
        row for row in opportunities
        if (
            row.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and row.desktop_parent_placement_sha256
                == ticket.execution_plan.desktop_placement_sha256
        )
    )
    compatible = tuple(
        row for row in exact_geometry
        if not controller._ready_helper_parent_rejection_reasons(
            ticket, generated_baseline, row
        )
    )
    if len(compatible) != 1:
        diagnostics = tuple(
            (
                row.desktop_parent_route_id,
                controller._ready_helper_parent_rejection_reasons(
                    ticket, generated_baseline, row
                ),
            )
            for row in exact_geometry
        )
        raise _ReadyHelperParentUnavailable(
            "ready layout has no compatible helper opportunity: "
            + repr(diagnostics)
        )
    generated_opportunity = compatible[0]
    opportunity = replace(
        generated_opportunity,
        desktop_parent_route_id=ticket.decision.route_id,
    )
    assisted = tuple(
        row for row in candidate_set.candidates
        if row.candidate_id == opportunity.route_id
        and row.plan.plan_sha256 == opportunity.operator_plan_sha256
    )
    if len(assisted) != 1:
        raise UnifiedScheduleError(
            "ready layout helper candidate is ambiguous"
        )
    current_baseline, _ = controller._ticket_adaptive_policy_records(ticket)
    if (
        current_baseline.route_id != ticket.decision.route_id
        or current_baseline.desktop_placement_sha256
            != opportunity.desktop_parent_placement_sha256
    ):
        raise UnifiedScheduleError(
            "acquired desktop policy differs from its ticket"
        )
    controller._record_ready_helper_event_once(
        ticket,
        layout,
        "TEMPLATE_FOUND",
        observed_at_us,
        accepted=True,
    )
    return (
        manifest,
        candidate_set,
        opportunity,
        assisted[0],
        generated_opportunity.desktop_parent_route_id,
        current_baseline,
    )


def _build_ready_helper_materialization(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    manifest: object,
    candidate_set: AutomatedCandidateSet,
    opportunity: HelperOpportunity,
    assisted_candidate: object,
    generated_parent_route_id: str,
    current_baseline: AdaptiveDecodePolicy,
) -> _ReadyHelperMaterialization:
    _, policies, _ = adaptive_decode_policies(
        candidate_set,
        manifest,
        controller._runtime_capabilities,
        ticket.request.output_tokens,
        controller._maximum_phone_sessions,
    )
    policies = tuple(
        replace(row, desktop_parent_route_id=ticket.decision.route_id)
        for row in policies
        if row.operator_plan_sha256 == opportunity.operator_plan_sha256
    )
    ticket_policies = tuple(
        row for row in policies if row.route_id == opportunity.route_id
    )
    if not policies or len(ticket_policies) != 1:
        raise UnifiedScheduleError(
            "ready layout adaptive policies are incomplete"
        )
    helper_plan, helper_binding = controller._authorize_phone_helper_plan(
        opportunity.helper_operator_plan,
        assisted_candidate.binding,
        layout,
        model_id=manifest.model_id,
        artifact_sha256=manifest.artifact_sha256,
    )
    policies = tuple(
        replace(row, operator_plan_sha256=helper_plan.plan_sha256)
        for row in policies
    )
    ticket_policies = tuple(
        row for row in policies if row.route_id == opportunity.route_id
    )
    helper = controller._build_helper_envelope(
        artifact_sha256=manifest.artifact_sha256,
        desktop_parent_route_id=opportunity.desktop_parent_route_id,
        desktop_placement_sha256=(
            opportunity.desktop_parent_placement_sha256
        ),
        helper_plan=helper_plan,
        helper_binding=helper_binding,
        layout=layout,
    )
    component = runtime_residency_component_identity(
        manifest.artifact_sha256,
        helper.helper_plan,
        helper.helper_binding,
    )
    return _ReadyHelperMaterialization(
        helper=helper,
        opportunity=opportunity,
        generated_parent_route_id=generated_parent_route_id,
        baseline=current_baseline,
        policies=policies,
        ticket_policy=ticket_policies[0],
        component=component,
    )


def _bind_ready_helper_materialization(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    materialized: _ReadyHelperMaterialization,
    observed_at_us: int,
) -> bool:
    request_id = ticket.request.request_id
    helper = materialized.helper
    envelope = controller._request_helper_envelope_binding(
        helper, layout
    )
    token_index = controller._model_placement_controller.request_decode_token_index(
        request_id
    )
    rebind = controller._model_placement_controller.request_helper_rebind_state(
        request_id
    )
    masked_rebind = bool(
        rebind is not None
        and tuple(rebind.get("target_allowed_session_ids", ()))
    )
    additive_rebind = (
        rebind is None
        and controller._model_placement_controller
            .request_helper_envelope_is_additive(
                request_id, envelope
            )
    )
    if rebind is not None:
        controller._model_placement_controller.commit_request_helper_rebind(
            request_id,
            envelope,
            resident_component_identity_sha256=(
                materialized.component.identity_sha256
            ),
            start_token_index=token_index,
            observed_at_us=observed_at_us,
        )
    elif not additive_rebind:
        controller._model_placement_controller.bind_request_helper_envelope(
            request_id, envelope, observed_at_us=observed_at_us
        )
        binding = controller._model_placement_controller.request_binding(request_id)
        if binding is not None and binding.get("helper_attachment") is None:
            controller._model_placement_controller.attach_request_helper(
                request_id,
                phone_layout_generation=layout.generation,
                phone_layout_geometry_sha256=layout.layout.geometry_sha256,
                resident_component_identity_sha256=(
                    materialized.component.identity_sha256
                ),
                operator_plan_sha256=helper.operator_plan_sha256,
                start_token_index=token_index,
                fraction_ppm=0,
                lease_tokens=(),
                lease_reserved_until_us=None,
                observed_at_us=observed_at_us,
            )
    return masked_rebind or additive_rebind


def _publish_ready_helper_materialization(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    materialized: _ReadyHelperMaterialization,
    masked_rebind: bool,
    observed_at_us: int,
) -> None:
    request_id = ticket.request.request_id
    helper = materialized.helper
    compatible_layers_by_plan = {
        plan_hash: retained_execution_layer_mask(previous, helper, layout)
        for plan_hash, previous in getattr(controller, "_request_helper_envelope_history", {}).get(
            request_id, {}).items()
    }
    controller._request_helper_opportunities[request_id] = (
        materialized.opportunity,
    )
    controller._late_request_helper_contexts[request_id] = _LateRequestHelperContext(
        helper=helper,
        baseline=materialized.baseline,
        candidates=materialized.policies,
        ticket_policy=materialized.ticket_policy,
        component=materialized.component,
        evidence_state=materialized.opportunity.evidence_state,
    )
    controller._remember_request_helper_envelope(request_id, helper)
    active_rebind = False
    try:
        adaptive_state = controller._adaptive_decode.snapshot(request_id)
    except AdaptiveDecodeError:
        pass
    else:
        active_rebind = adaptive_state["helper_available"]
        if not active_rebind and _late_helper_adoption(controller):
            controller._record_late_helper_adoption(
                request_id, helper,
                token_index=controller._model_placement_controller
                    .request_decode_token_index(request_id),
                at_us=observed_at_us,
                source="READY_LAYOUT_PUBLISHED",
            )
        method = (
            controller._adaptive_decode.helper_rebound
            if active_rebind
            else controller._adaptive_decode.helper_ready
        )
        power_policy = {"compatible_layers_by_plan": compatible_layers_by_plan} if active_rebind else {
            "allow_assumed_phone_power_for_operational_selection": (
                controller._adaptive_start_allow_assumed_phone_power(
                    ticket.execution_plan, helper, None
                )
            ),
        }
        identity = controller._adaptive_layout_identity_sha256(ticket, layout.generation)
        if identity is not None:
            power_policy["phone_layout_identity_sha256"] = identity
        method(
            request_id,
            phone_layout_generation=layout.generation,
            phone_layout_geometry_sha256=layout.layout.geometry_sha256,
            candidates=materialized.policies,
            component_capability_sha256=(
                materialized.component.identity_sha256
            ),
            ticket_policy=materialized.ticket_policy,
            helper_evidence_state=materialized.opportunity.evidence_state,
            **power_policy,
        )
        controller._record_adaptive_helper_energy_policy_binding(
            ticket, helper, adaptive_state, observed_at_us,
        )
    contract = helper.helper_plan.execution_contract
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "HELPER_REMATERIALIZED",
        observed_at_us,
        {
            "operator_plan_sha256": helper.operator_plan_sha256,
            "generated_desktop_parent_route_id": (
                materialized.generated_parent_route_id
            ),
            "immutable_desktop_parent_route_id": ticket.decision.route_id,
            "phone_layout_generation": layout.generation,
            "phone_layout_geometry_sha256": layout.layout.geometry_sha256,
            "phone_session_ids": [
                row.session_id for row in contract.phone_shards
            ],
            "evidence_state": materialized.opportunity.evidence_state,
            **({"retained_evidence_layers_by_plan": compatible_layers_by_plan}
               if active_rebind and any(compatible_layers_by_plan.values()) else {}),
            "qualification_state": (
                "QUALIFIED"
                if materialized.opportunity.evidence_state == "TRUSTED"
                else "DIAGNOSTIC"
            ),
        },
    )
    controller._record_ready_helper_event_once(
        ticket,
        layout,
        "MATERIALIZED",
        observed_at_us,
        helper=helper,
        accepted=True,
    )


def _rematerialize_ready_layout_helper(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
    phone_safety_state: RuntimeExecutorState | None,
) -> None:
    records = controller._ready_helper_candidate_records(
        ticket, layout, snapshot, observed_at_us, phone_safety_state
    )
    materialized = controller._build_ready_helper_materialization(
        ticket, layout, *records
    )
    masked_rebind = controller._bind_ready_helper_materialization(
        ticket, layout, materialized, observed_at_us
    )
    controller._publish_ready_helper_materialization(
        ticket, layout, materialized, masked_rebind, observed_at_us
    )


def _retain_usable_ready_layout_helper(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    observed_at_us: int,
) -> bool:
    """Keep an exact session helper when another session is published."""

    request_id = ticket.request.request_id
    context = controller._late_request_helper_contexts.get(request_id)
    if context is None:
        return False
    helper = context.helper
    if (
        helper.artifact_sha256 != ticket.model.artifact_sha256
        or not controller._model_placement_controller
            .request_helper_layout_is_usable(
                request_id,
                helper.phone_layout_generation,
                helper.phone_layout_geometry_sha256,
            )
    ):
        return False
    target_by_session = {
        row.session_id: row for row in layout.layout.shards
    }
    target_session_ids = {
        row.session_id for row in layout.layout.shards
        if row.artifact_sha256 == ticket.model.artifact_sha256
    }
    helper_shards = tuple(
        helper.helper_plan.execution_contract.phone_shards
    )
    if (
        not helper_shards
        or {row.session_id for row in helper_shards}
            != target_session_ids
        or any(
        shard.session_id not in target_by_session
        or _phone_shard_structure(shard)
            != _phone_shard_structure(
                target_by_session[shard.session_id]
            )
        for shard in helper_shards
        )
    ):
        return False
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "HELPER_RETAINED_FOR_READY_LAYOUT",
        observed_at_us,
        {
            "helper_phone_layout_generation": (
                helper.phone_layout_generation
            ),
            "phone_layout_generation": layout.generation,
            "phone_layout_geometry_sha256": (
                layout.layout.geometry_sha256
            ),
            "phone_session_ids": [
                row.session_id for row in helper_shards
            ],
        },
    )
    return True


def _late_helper_adoption(controller) -> bool:
    """Whether ``adaptive_decode_overrides.late_helper_adoption`` is on."""
    config = getattr(controller, "_adaptive_decode_config", None)
    return getattr(config, "late_helper_adoption", False) is True


def _retain_materialized_ready_layout_helper(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    observed_at_us: int,
) -> bool:
    """Keep a helper already materialized for exactly this READY layout.

    Opt-in with ``late_helper_adoption``: the helper waits for its attachment
    to be expanded at the request's next boundary. Materializing it again
    from a later snapshot either reproduces it or yields an envelope whose
    operator plan is recorded with another identity, and that failure drops
    the request's helper (s1a 001: the READY refresh 1 s after the gen-5
    publication failed with ``runtime helper plan history changed identity``
    and the session decoded helper-less until gen 7).
    """
    if not _late_helper_adoption(controller):
        return False
    context = controller._late_request_helper_contexts.get(
        ticket.request.request_id
    )
    if (
        context is None
        or context.helper.phone_layout_generation != layout.generation
        or context.helper.phone_layout_geometry_sha256
            != layout.layout.geometry_sha256
        or any(
            shard.session_generation
                != layout.layout.session_generation_by_id.get(shard.session_id)
            for shard in context.helper.helper_plan.execution_contract.phone_shards
        )
        or controller._ready_request_helper(ticket, context.helper) is None
    ):
        return False
    controller._record_ready_helper_event_once(
        ticket,
        layout,
        "HELPER_MATERIALIZATION_RETAINED",
        observed_at_us,
        helper=context.helper,
        accepted=True,
    )
    return True


def _handle_ready_helper_materialization_error(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    error: Exception,
    observed_at_us: int,
) -> None:
    request_id = ticket.request.request_id
    if isinstance(error, _ReadyHelperSafetyDeferred):
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "HELPER_REMATERIALIZATION_DEFERRED",
            observed_at_us,
            {"reason": str(error)},
        )
        return
    controller._request_helper_opportunities.pop(request_id, None)
    controller._late_request_helper_contexts.pop(request_id, None)
    controller._model_placement_controller.cancel_request_helper_rebind(
        request_id,
        observed_at_us=observed_at_us,
        reason="HELPER_REMATERIALIZATION_FAILED",
    )
    failure = {"reason": str(error)}
    if isinstance(error, _ReadyHelperParentUnavailable):
        failure["configuration_sha256"] = controller._ready_helper_configuration(
            ticket, layout
        )
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "HELPER_REMATERIALIZATION_FAILED",
        observed_at_us,
        failure,
    )
    controller._record_ready_helper_event_once(
        ticket,
        layout,
        "REJECTED",
        observed_at_us,
        accepted=False,
        reason=str(error),
    )
