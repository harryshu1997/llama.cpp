"""HelperEnvelopeMixin replacement operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from ..._internal.model_placement_controller import (
    REQUEST_HELPER_REBIND_STATES,
    ModelPhoneResidencyLayout,
)
from ..._internal.runtime_plan import (
    HelperOpportunity,
    PhoneSessionReplacementAuthorization,
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionPlan,
    helper_preparation_changed_session_ids,
    phone_session_map_sha256,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..common import _RECOVERABLE_ERRORS, _StalePhoneSessionAssignment


def _helper_changed_session_ids(
    layout: ModelPhoneResidencyLayout,
    helper_plan: RuntimeExecutionPlan,
) -> tuple[str, ...]:
    """Scope a layout's changed sessions to one helper's own shards."""

    return helper_preparation_changed_session_ids(
        layout.layout.changed_session_ids,
        tuple(
            row.session_id
            for row in helper_plan.execution_contract.phone_shards
        ),
    )


def _phone_session_replacement_authorization(
    controller,
    target: ModelPhoneResidencyLayout,
) -> PhoneSessionReplacementAuthorization | None:
    """Bind a partial target to the exact selected source session."""

    changed = tuple(target.layout.changed_session_ids)
    source = controller._model_placement_controller.ready_phone_layout()
    if len(changed) != 1:
        return None
    if source is not None and source.generation == target.generation:
        if not target.layout.replacement_source_identities:
            return None
        exact = []
        for envelope in (
            *controller._request_helper_preparation_envelopes.values(),
            *(
                row.helper_envelope
                for row in controller._request_helper_preparations.values()
            ),
        ):
            authorization = envelope.replacement_authorization
            if (
                envelope.phone_layout_generation == target.generation
                and envelope.phone_layout_geometry_sha256
                    == target.layout.geometry_sha256
                and authorization is not None
                and authorization.selected_session_id == changed[0]
                and authorization.target_layout_hash
                    == phone_session_map_sha256(
                        target.layout.shards,
                        target.layout.session_generation_by_id,
                    )
            ):
                if authorization not in exact:
                    exact.append(authorization)
        if len(exact) != 1:
            raise UnifiedScheduleError(
                "ready one-session replacement authority is not exact"
            )
        return exact[0]
    if source is None:
        return None
    selected = changed[0]
    source_by_session = {
        row.session_id: row for row in source.layout.shards
    }
    target_by_session = {
        row.session_id: row for row in target.layout.shards
    }
    if (
        selected not in target_by_session
        or set(source_by_session) - set(target_by_session)
        or set(target_by_session) - set(source_by_session) - {selected}
        or target.layout.session_generation_by_id.get(selected)
            != source.layout.session_generation_by_id.get(selected, 0) + 1
    ):
        raise UnifiedScheduleError(
            "one-session replacement authority differs from layout"
        )
    return PhoneSessionReplacementAuthorization.create(
        selected_session_id=selected,
        source_shards=source.layout.shards,
        target_shards=target.layout.shards,
        source_generation_by_id=(
            source.layout.session_generation_by_id
        ),
        target_generation_by_id=(
            target.layout.session_generation_by_id
        ),
    )


def _validate_phone_session_replacement_authorization(
    controller,
    target: ModelPhoneResidencyLayout,
    authorization: PhoneSessionReplacementAuthorization | None,
) -> None:
    """Validate the selected session without changing its assignment."""

    changed = tuple(target.layout.changed_session_ids)
    if len(changed) != 1:
        if authorization is not None:
            raise _StalePhoneSessionAssignment(
                "non-partial layout carries replacement authorization"
            )
        return
    source = controller._model_placement_controller.ready_phone_layout()
    if source is None or source.generation == target.generation:
        if authorization is None:
            return
        raise _StalePhoneSessionAssignment(
            "cold layout carries replacement authorization"
        )
    if (
        authorization is None
        or changed != (authorization.selected_session_id,)
        or phone_session_map_sha256(
            source.layout.shards,
        source.layout.session_generation_by_id,
        ) != authorization.source_layout_hash
        or phone_session_map_sha256(
            target.layout.shards,
            target.layout.session_generation_by_id,
        ) != authorization.target_layout_hash
        or source.layout.session_generation_by_id.get(changed[0], 0)
            != authorization.source_generation
        or target.layout.session_generation_by_id.get(changed[0])
            != authorization.target_generation
    ):
        raise _StalePhoneSessionAssignment(
            "phone session replacement assignment is stale"
        )
    expected = controller._phone_session_replacement_authorization(target)
    if expected != authorization:
        raise _StalePhoneSessionAssignment(
            "phone session replacement assignment differs"
        )


def _prevalidate_target_layout_helper_envelopes(
    controller,
    target: ModelPhoneResidencyLayout,
    *,
    request_id: str,
    observed_at_us: int,
) -> None:
    """Check every retained and new helper's session scope before a load.

        Each acquired request covered by the target layout will receive a
        helper envelope whose changed sessions are scoped to its own shards.
        A retained helper must not see its attached sessions as changed, and
        a helper whose sessions are replaced must already be quiescing.
        """

    changed = set(target.layout.changed_session_ids)
    sessions_by_artifact: dict[str, set[str]] = {}
    artifact_by_session: dict[str, str] = {}
    for shard in target.layout.shards:
        sessions_by_artifact.setdefault(
            shard.artifact_sha256, set()
        ).add(shard.session_id)
        artifact_by_session[shard.session_id] = shard.artifact_sha256
    rows = []
    for ticket in controller._runtime_controller.current_tickets(("ACQUIRED",)):
        artifact = ticket.model.artifact_sha256
        sessions = sessions_by_artifact.get(artifact)
        if sessions is None:
            continue
        owner_id = ticket.request.request_id
        scoped = helper_preparation_changed_session_ids(
            target.layout.changed_session_ids, tuple(sorted(sessions))
        )
        expected = tuple(sorted(
            session_id for session_id in changed
            if artifact_by_session.get(session_id) == artifact
        ))
        binding = controller._model_placement_controller.request_binding(
            owner_id
        )
        attachment = (
            None if binding is None else binding.get("helper_attachment")
        )
        attached = (
            ()
            if not isinstance(attachment, Mapping)
            else tuple(attachment.get("phone_session_ids", ()))
        )
        detached = (
            isinstance(attachment, Mapping)
            and attachment.get("fallback_outcome") is not None
        )
        rebind = (
            controller._model_placement_controller
            .request_helper_rebind_state(owner_id)
        )
        replaced = tuple(sorted(set(attached) & changed))
        quiescing = bool(
            rebind is not None
            and rebind["target_generation"] == target.generation
            and rebind["state"] in REQUEST_HELPER_REBIND_STATES
        )
        row = {
            "artifact_sha256": artifact,
            "attached_session_ids": list(attached),
            "changed_session_ids": list(scoped),
            "detached": detached,
            "quiescing": quiescing,
            "rebind_state": None if rebind is None else rebind["state"],
            "replaced_session_ids": list(replaced),
            "request_id": owner_id,
            "target_session_ids": sorted(sessions),
        }
        rows.append(row)
        if scoped != expected:
            raise UnifiedScheduleError(
                "helper envelope changed sessions are not artifact-local: "
                + repr(row)
            )
        if not replaced and attached and set(attached) & set(scoped):
            raise UnifiedScheduleError(
                "retained helper envelope names its own sessions as "
                    "changed: " + repr(row)
            )
    controller._model_placement_controller.record_request_helper_event(
        request_id,
        "PREPARATION_ENVELOPES_PREVALIDATED",
        observed_at_us,
        {
            "helpers": rows,
            "phone_layout_generation": target.generation,
            "phone_layout_geometry_sha256": (
                target.layout.geometry_sha256
            ),
        },
    )


def _materialize_phone_layout_preparation_envelope(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    observed_at_us: int,
) -> RuntimeHelperExecutionEnvelope | None:
    """Materialize one exact phone-only transition for a target layout."""

    if (
        layout.state not in {"PROPOSED", "PREPARING"}
        or ticket.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}
        or ticket.execution_plan is None
        or ticket.execution_plan.desktop_placement_sha256 is None
    ):
        return None
    changed_artifacts = {
        shard.artifact_sha256
        for shard in layout.layout.shards
        if shard.session_id in layout.layout.changed_session_ids
    }
    if ticket.model.artifact_sha256 not in changed_artifacts:
        return None
    key = (
        ticket.request.request_id,
        ticket.ticket_id,
        layout.generation,
    )
    cached = controller._request_helper_preparation_envelopes.get(key)
    if cached is not None:
        if (
            cached.phone_layout_generation != layout.generation
            or cached.phone_layout_geometry_sha256
                != layout.layout.geometry_sha256
            or cached.artifact_sha256
                != ticket.model.artifact_sha256
            or cached.desktop_parent_route_id
                != ticket.decision.route_id
            or cached.desktop_placement_sha256
                != ticket.execution_plan.desktop_placement_sha256
        ):
            raise UnifiedScheduleError(
                "cached helper preparation identity differs"
            )
        authorized_plan, authorized_binding = (
            controller._authorize_phone_helper_plan(
                cached.helper_plan,
                cached.helper_binding,
                layout,
                model_id=ticket.model.model_id,
                artifact_sha256=ticket.model.artifact_sha256,
            )
        )
        refreshed = controller._build_helper_envelope(
            artifact_sha256=ticket.model.artifact_sha256,
            desktop_parent_route_id=ticket.decision.route_id,
            desktop_placement_sha256=(
                ticket.execution_plan.desktop_placement_sha256
            ),
            helper_plan=authorized_plan,
            helper_binding=authorized_binding,
            layout=layout,
        )
        controller._request_helper_preparation_envelopes[key] = refreshed
        controller._remember_phone_helper_endpoint_template(refreshed)
        return refreshed

    manifest = controller.runtime_model_manifest(ticket.model.model_id)
    request_snapshot = controller._automated_snapshot_for_request(
        ticket.request,
        snapshot,
        exclude_request_id=ticket.request.request_id,
        project_request_ids=(),
    )
    base_state = request_snapshot.executors.get(
        ticket.binding.executor_id
    )
    if base_state is None or not base_state.healthy:
        return None
    if not base_state.ready or base_state.free_slots < 1:
        request_snapshot = replace(
            request_snapshot,
            executors={
                **request_snapshot.executors,
                ticket.binding.executor_id: replace(
                    base_state, ready=True, free_slots=1
                ),
            },
        )
    candidate_set = controller._generate_automated_candidate_set(
        ticket.request,
        manifest,
        request_snapshot,
        observed_at_us,
        use_residency_holds=False,
        update_phone_residency_portfolio=False,
    )
    candidate_set = controller._candidate_set_for_ready_phone_layout(
        candidate_set,
        manifest.artifact_sha256,
        layout,
    )
    candidate_set = (
        controller._candidate_set_with_verified_partial_phone_memory(
            candidate_set,
            manifest,
            layout,
            request_snapshot,
            observed_at_us,
        )
    )
    opportunities = tuple(
        opportunity
        for opportunity in controller._compact_helper_opportunities(
            candidate_set, ticket.request
        )
        if (
            opportunity.phone_layout_geometry_sha256
                == layout.layout.geometry_sha256
            and opportunity.desktop_parent_placement_sha256
                == ticket.execution_plan.desktop_placement_sha256
            and opportunity.helper_operator_plan.baseline_executor_id
                == ticket.binding.executor_id
        )
    )
    if len(opportunities) != 1:
        return None
    opportunity = opportunities[0]
    assisted = tuple(
        candidate
        for candidate in candidate_set.candidates
        if (
            candidate.candidate_id == opportunity.route_id
            and candidate.plan.plan_sha256
                == opportunity.operator_plan_sha256
        )
    )
    if len(assisted) != 1:
        return None
    authorized_plan, authorized_binding = (
        controller._authorize_phone_helper_plan(
            opportunity.helper_operator_plan,
            assisted[0].binding,
            layout,
            model_id=manifest.model_id,
            artifact_sha256=manifest.artifact_sha256,
        )
    )
    helper = controller._build_helper_envelope(
        artifact_sha256=manifest.artifact_sha256,
        desktop_parent_route_id=ticket.decision.route_id,
        desktop_placement_sha256=(
            ticket.execution_plan.desktop_placement_sha256
        ),
        helper_plan=authorized_plan,
        helper_binding=authorized_binding,
        layout=layout,
    )
    controller._request_helper_opportunities[ticket.request.request_id] = (
        replace(
            opportunity,
            desktop_parent_route_id=ticket.decision.route_id,
            phone_layout_generation=layout.generation,
        ),
    )
    controller._request_helper_preparation_envelopes[key] = helper
    controller._remember_phone_helper_endpoint_template(helper)
    controller._record_preparation_envelope_materialized(
        ticket, layout, helper, opportunity, observed_at_us
    )
    return helper


def _record_preparation_envelope_materialized(
    controller,
    ticket: RuntimeRequestTicket,
    layout: ModelPhoneResidencyLayout,
    helper: RuntimeHelperExecutionEnvelope,
    opportunity: HelperOpportunity,
    observed_at_us: int,
) -> None:
    controller._model_placement_controller.record_request_helper_event(
        ticket.request.request_id,
        "PREPARATION_ENVELOPE_MATERIALIZED",
        observed_at_us,
        {
            "operator_plan_sha256": helper.operator_plan_sha256,
            "phone_layout_generation": layout.generation,
            "phone_layout_geometry_sha256": (
                layout.layout.geometry_sha256
            ),
            "phone_session_ids": [
                shard.session_id
                for shard in helper.helper_plan.execution_contract
                    .phone_shards
            ],
            "evidence_state": opportunity.evidence_state,
            "qualification_state": (
                "QUALIFIED"
                if opportunity.evidence_state == "TRUSTED" else
                "DIAGNOSTIC"
            ),
            "request_ticket_id": ticket.ticket_id,
        },
    )


def runtime_request_helper_preparation_envelope(
    controller,
    request_id: str,
    *,
    expected_ticket_id: str,
    observed_at_us: int,
    snapshot: HeterogeneousRuntimeSnapshot,
) -> RuntimeHelperExecutionEnvelope | None:
    """Return an exact scheduler-issued helper transition envelope."""

    if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
        raise UnifiedScheduleError(
            "helper preparation snapshot is invalid"
        )
    snapshot.validate_at(observed_at_us)
    ticket = controller.runtime_ticket(request_id)
    if (
        ticket.ticket_id != expected_ticket_id
        or ticket.selection_mode == "desktop-baseline"
        or ticket.dispatch_state in {"CANCELLED", "COMPLETED", "FAILED"}
    ):
        return None
    offline_plan_id = getattr(
        controller, "_active_offline_phone_residency_plan_id", None
    )
    offline_plan = (
        None
        if offline_plan_id is None else
        getattr(controller, "_offline_phone_residency_plans", {}).get(
            offline_plan_id
        )
    )
    if (
        offline_plan is not None
        and offline_plan.state not in {"READY", "FAILED"}
    ):
        return None
    controller._reevaluate_pending_phone_layout_observation(
        ticket, snapshot, observed_at_us
    )
    target = controller._model_placement_controller.target_phone_layout()
    if target is None:
        return None
    try:
        with controller._transaction(convert=False):
            return controller._materialize_phone_layout_preparation_envelope(
                ticket, target, snapshot, observed_at_us
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            request_id,
            "PREPARATION_ENVELOPE_REJECTED",
            observed_at_us,
            {
                "phone_layout_generation": target.generation,
                "phone_layout_geometry_sha256": (
                    target.layout.geometry_sha256
                ),
                "reason": str(exc),
                "request_ticket_id": ticket.ticket_id,
            },
        )
        return None


def runtime_background_helper_preparation_allowed(
    controller,
    request_id: str,
    *,
    expected_ticket_id: str,
) -> bool:
    """Permit demanded phone preparation for helper-eligible requests."""

    ticket = controller.runtime_ticket(request_id)
    if (
        ticket.ticket_id != expected_ticket_id
        or ticket.selection_mode == "desktop-baseline"
        or ticket.dispatch_state in {
            "CANCELLED", "COMPLETED", "FAILED"
        }
    ):
        return False
    if ticket.dispatch_state not in {
        "ACQUIRED",
        "QUEUED", "REPLAN_REQUIRED", "REPLANNING"
    }:
        return False
    target = controller._model_placement_controller.target_phone_layout()
    if target is None and (
        controller._model_placement_controller.pending_phone_layout_candidate()
        is not None
    ):
        return True
    if (
        target is None
        or target.state not in {"PROPOSED", "PREPARING"}
    ):
        return (
            ticket.dispatch_state == "ACQUIRED"
            and ticket.transition_status != "PENDING"
        )
    changed_artifacts = {
        shard.artifact_sha256
        for shard in target.layout.shards
        if shard.session_id in target.layout.changed_session_ids
    }
    return ticket.model.artifact_sha256 in changed_artifacts
