"""AdaptiveDecodeController admission operations on its existing owner."""

from __future__ import annotations

import time
from typing import Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
    validate_policy_set,
)
from ..adaptive_decode_state import _AdaptiveSession
from .common import HELPER_EVIDENCE_STATES


def _validate_start_arguments(
    controller,
    *,
    model_artifact_sha256: str,
    planning_profile_sha256: str,
    component_capability_sha256: str,
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
    output_tokens: int,
    context_length: int,
    active_batch: int,
    deadline_us: int,
    slot_id: int,
    first_token_index: int,
    first_token_at_us: int,
    config: AdaptiveDecodeConfig,
    ticket_policy: AdaptiveDecodePolicy | None,
    helper_available: bool,
    helper_layout_generation: int | None,
    helper_layout_geometry_sha256: str | None,
    helper_evidence_state: str,
) -> tuple[tuple[AdaptiveDecodePolicy, ...], AdaptiveDecodePolicy | None]:
    rows = validate_policy_set(baseline, candidates)
    if helper_evidence_state not in HELPER_EVIDENCE_STATES:
        raise AdaptiveDecodeError(
            "adaptive helper evidence state is invalid"
        )
    if helper_evidence_state == "LEARNING":
        ticket_policy = None
    elif ticket_policy is not None:
        matches = tuple(
            row for row in rows
            if row.policy_hash == ticket_policy.policy_hash
        )
        if len(matches) != 1 or matches[0] != ticket_policy:
            raise AdaptiveDecodeError(
                "adaptive ticket policy is not a candidate"
            )
        ticket_policy = matches[0]
    for name, value in (
        ("model artifact", model_artifact_sha256),
        ("planning profile", planning_profile_sha256),
        ("component capability", component_capability_sha256),
    ):
        if (
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
        ):
            raise AdaptiveDecodeError(
                "adaptive " + name + " is invalid"
            )
    for name, value, minimum in (
        ("output tokens", output_tokens, 1),
        ("context length", context_length, 1),
        ("active batch", active_batch, 1),
        ("deadline", deadline_us, 1),
        ("slot", slot_id, 0),
        ("first token", first_token_index, 0),
        ("first token time", first_token_at_us, 0),
    ):
        if type(value) is not int or value < minimum:
            raise AdaptiveDecodeError(f"adaptive {name} is invalid")
    if first_token_index >= output_tokens:
        raise AdaptiveDecodeError("adaptive request has no decode remainder")
    if not isinstance(config, AdaptiveDecodeConfig):
        raise AdaptiveDecodeError("adaptive configuration is invalid")
    if type(helper_available) is not bool:
        raise AdaptiveDecodeError(
            "adaptive helper availability is invalid"
        )
    if helper_available:
        identity_present = (
            helper_layout_generation is not None
            or helper_layout_geometry_sha256 is not None
        )
        if identity_present and (
            type(helper_layout_generation) is not int
            or helper_layout_generation < 1
            or not controller._valid_sha256(helper_layout_geometry_sha256)
        ):
            raise AdaptiveDecodeError(
                "adaptive ready helper identity is invalid"
            )
    elif (
        helper_layout_generation is not None
        or helper_layout_geometry_sha256 is not None
    ):
        raise AdaptiveDecodeError(
            "adaptive unavailable helper carries an identity"
        )
    return rows, ticket_policy


def _seed_session_history(
    controller,
    session: _AdaptiveSession,
    baseline: AdaptiveDecodePolicy,
    rows: tuple[AdaptiveDecodePolicy, ...],
    config: AdaptiveDecodeConfig,
) -> None:
    for policy in (baseline, *rows):
        identity = controller._policy_identity(policy)
        historical, group_count = controller._historical_policy_records(
            session, policy, compatible_context=True
        )
        if historical:
            session.historical_records[identity] = historical
            session.historical_group_counts[identity] = group_count
    session.probe_candidates = controller._sample_candidates(
        rows, config, session.helper_evidence_state
    )
    controller._seed_verification_candidate(session)
    if session.helper_evidence_state == "LEARNING":
        session.refinement_added = True


def start(
    controller,
    *,
    request_id: str,
    ticket_id: str,
    model_artifact_sha256: str,
    planning_profile_sha256: str,
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
    output_tokens: int,
    context_length: int,
    active_batch: int,
    deadline_us: int,
    slot_id: int,
    first_token_index: int,
    first_token_at_us: int,
    config: AdaptiveDecodeConfig,
    ticket_policy: AdaptiveDecodePolicy | None = None,
    component_capability_sha256: str | None = None,
    helper_available: bool = True,
    helper_layout_generation: int | None = None,
    helper_layout_geometry_sha256: str | None = None,
    helper_evidence_state: str = "TRUSTED",
    phone_power_policy_from_capability: bool = False,
    execution_context_available: bool = True,
    helper_layout_identity_sha256: str | None = None,
) -> AdaptiveDecodeDirective:
    started_ns = time.perf_counter_ns()
    if type(phone_power_policy_from_capability) is not bool:
        raise AdaptiveDecodeError("adaptive phone power policy source is invalid")
    if helper_layout_identity_sha256 is not None and (
        helper_layout_generation is None
        or not controller._valid_sha256(helper_layout_identity_sha256)
    ):
        raise AdaptiveDecodeError("adaptive helper layout identity is invalid")
    if type(execution_context_available) is not bool:
        raise AdaptiveDecodeError("adaptive execution context availability is invalid")
    component_capability_sha256 = (
        planning_profile_sha256
        if component_capability_sha256 is None
        else component_capability_sha256
    )
    rows, ticket_policy = controller._validate_start_arguments(
        model_artifact_sha256=model_artifact_sha256,
        planning_profile_sha256=planning_profile_sha256,
        component_capability_sha256=component_capability_sha256,
        baseline=baseline,
        candidates=candidates,
        output_tokens=output_tokens,
        context_length=context_length,
        active_batch=active_batch,
        deadline_us=deadline_us,
        slot_id=slot_id,
        first_token_index=first_token_index,
        first_token_at_us=first_token_at_us,
        config=config,
        ticket_policy=ticket_policy,
        helper_available=helper_available,
        helper_layout_generation=helper_layout_generation,
        helper_layout_geometry_sha256=helper_layout_geometry_sha256,
        helper_evidence_state=helper_evidence_state,
    )
    with controller._lock:
        if (
            request_id in controller._sessions
            or request_id in controller._sealed_sessions
            or request_id in controller._completed
        ):
            raise AdaptiveDecodeError("adaptive request already exists")
        session = _AdaptiveSession(
            request_id=request_id,
            ticket_id=ticket_id,
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            component_capability_sha256=(
                component_capability_sha256
            ),
            baseline=baseline,
            candidates=rows,
            output_tokens=output_tokens,
            context_length=context_length,
            active_batch=active_batch,
            deadline_us=deadline_us,
            config=config,
            slot_id=slot_id,
            ticket_policy=ticket_policy,
            helper_available=helper_available,
            helper_layout_generation=helper_layout_generation,
            helper_layout_geometry_sha256=(
                helper_layout_geometry_sha256
            ),
            helper_layout_identity_sha256=helper_layout_identity_sha256,
            helper_evidence_state=helper_evidence_state,
            history_helper_layout_geometry_sha256=(
                helper_layout_geometry_sha256
            ),
            phone_power_policy_from_capability=(
                phone_power_policy_from_capability
            ),
            acknowledged_policy=baseline,
            execution_context_available=execution_context_available,
        )
        controller._seed_session_history(session, baseline, rows, config)
        controller._sessions[request_id] = session
        directive = controller._initial_start_directive(
            session, baseline, first_token_index, first_token_at_us
        )
        session.decision_time_us.append(
            (time.perf_counter_ns() - started_ns) // 1000
        )
        return directive
