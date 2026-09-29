"""Per-server policy coherence.

Slots whose FFN split policies differ cannot share a forward pass (`server_slot::can_batch_with`), so a
phone-assisted request next to a host-decoding co-tenant is served in alternating batches: it waits for
the co-tenant's host step, then runs its own phone step, and its measured window carries the co-tenant's
energy. Measured 2026-09-21 on the 24-request trace: 39 J/token for the phone policy alone, 128 J/token
and twice the latency next to a host-policy slot, 77 J/token for the host at either concurrency. Every
probe that ran under that mix was rejected although the phone path wins by 45 % when the slots batch
together.

With server_policy_coherence enabled, one owner selects the policy for a model, desktop parent, and
phone layout. Followers share that decision, and unfinished probes survive an owner's completion.
Only windows with matching batch composition can qualify or eliminate a policy. The default path
retains the earlier request-local coherence rule.
"""
from __future__ import annotations

from dataclasses import replace

from ..adaptive_decode_contracts import AdaptiveDecodeDirective, AdaptiveDecodePolicy
from ..adaptive_decode_state import _AdaptiveSession
from ..adaptive_decode_state import _AdaptiveServerPolicy

COHERENCE_REASON = "SERVER_POLICY_COHERENCE"


def server_policy_key(session: _AdaptiveSession) -> tuple | None:
    if (not session.config.server_policy_coherence or not session.helper_available
            or session.helper_layout_generation is None
            or session.helper_layout_geometry_sha256 is None):
        return None
    return (session.model_artifact_sha256, session.baseline.desktop_placement_sha256,
            session.helper_layout_generation, session.helper_layout_geometry_sha256)


def _server_group(controller, session, at_us):
    key = server_policy_key(session)
    if key is None:
        return None
    group = controller._server_policies.get(key)
    if group is None:
        group = _AdaptiveServerPolicy(session.request_id, session.baseline, at_us)
        group.records = tuple(row for row in session.records
                              if row.policy.baseline and row.measurement_eligible)
        running = _running_policy(session)
        if running is not None and not running.baseline:
            current = _matching_policy(controller, session, running)
            if current is not None:
                group.policy = group.proposal = current
        controller._server_policies[key] = group
    owner = controller._sessions.get(group.owner_request_id)
    if owner is None or server_policy_key(owner) != key:
        group.owner_request_id = session.request_id
        if not group.qualified and group.proposal is None:
            session.stage = "initial_baseline"
    if (session.stage == "server_policy" and group.owner_request_id == session.request_id
            and not group.qualified and group.proposal is None):
        session.stage = "initial_baseline"
    return group


def _matching_policy(controller, session, policy):
    return next((row for row in (session.baseline, *session.candidates)
                 if controller._policy_identity(row) == controller._policy_identity(policy)), None)


def _seed_server_evidence(controller, session, group):
    for policy in (session.baseline, *session.candidates):
        identity = controller._policy_identity(policy)
        records = tuple(row for row in group.records
                        if row.request_id != session.request_id
                        and row.active_batch == session.active_batch
                        and row.external_activity_sha256 == session.external_activity_sha256
                        and controller._policy_identity(row.policy) == identity)
        if records:
            session.historical_records[identity] = records
            # Concurrent windows are correlated measurements of the same server.
            session.historical_group_counts[identity] = 1


def record_server_window(controller, session, receipt):
    group = controller._server_policies.get(server_policy_key(session))
    if group is None:
        return
    if group.proposal is not None and not group.qualified:
        group.probe_tokens += receipt.token_count
    if receipt.measurement_eligible and receipt.output_valid and receipt.failure_reason is None:
        group.records = (*group.records, receipt)


def _server_probe_policy(controller, session, group, token_index, at_us):
    proposal = _matching_policy(controller, session, group.proposal)
    if proposal is None:
        group.proposal = None
        group.policy = session.baseline
        group.qualified = True
        return session.baseline
    baseline = controller._valid_records(session, session.baseline, operational=True)
    candidate = controller._valid_records(session, proposal, operational=True)
    if baseline and candidate:
        group.qualified = True
        group.proposal = None
        baseline_bounds = controller._bounds(session, session.baseline, operational=True)
        candidate_bounds = controller._bounds(session, proposal, operational=True)
        if (baseline_bounds is not None and candidate_bounds is not None
                and controller._learning_probe_improves(session, proposal)
                and candidate_bounds[2] * 1_000_000 <= baseline_bounds[1]
                    * (1_000_000 - session.config.minimum_energy_saving_ppm)
                and candidate_bounds[4] * 1_000_000 <= baseline_bounds[4]
                    * session.config.maximum_latency_ppm):
            return proposal
        session.zero_assistance_reason = "SERVER_PAIR_NOT_IMPROVED"
        return session.baseline
    if group.probe_tokens >= session.config.maximum_probe_tokens:
        group.qualified = True
        group.proposal = None
        session.zero_assistance_reason = "SERVER_PROBE_BUDGET_EXHAUSTED"
        return session.baseline
    return session.baseline if candidate and not baseline else proposal


def server_directive(controller, session, token_index, at_us):
    """Only the elected server owner probes; the other slots execute its decision."""
    group = _server_group(controller, session, at_us)
    if (group is None or session.tail_seal_token is not None
            or not session.execution_context_available):
        return None
    owner = group.owner_request_id == session.request_id
    _seed_server_evidence(controller, session, group)
    if owner and not group.qualified and group.proposal is None:
        return None
    if owner and group.proposal is not None:
        selected = _server_probe_policy(controller, session, group, token_index, at_us)
        if controller._policy_identity(selected) != controller._policy_identity(group.policy):
            group.changed_at_us = at_us
        group.policy = selected
    policy = _matching_policy(controller, session, group.policy)
    if policy is None:
        return None
    if owner and group.qualified and not policy.baseline:
        controller._update_elimination(session, policy)
    if owner and group.qualified and policy.policy_hash in session.eliminated_policy_reasons:
        group.policy = session.baseline
        group.proposal = None
        group.changed_at_us = at_us
        policy = session.baseline
    session.stage = "server_policy" if group.qualified else "server_probe"
    controller._state(session, "EXPLOITING")
    if session.current_policy is None:
        session.current_policy = session.baseline
    if not policy.baseline:
        session.zero_assistance_reason = None
    directive = (
        controller._open_window(session, policy, token_index, at_us, session.current_ack)
        if session.current_policy == policy else
        controller._control(session, policy, token_index, at_us)
    )
    return replace(directive, reason=COHERENCE_REASON)


def server_helper_window_eligible(controller, session, policy):
    group = controller._server_policies.get(server_policy_key(session))
    return bool(
        group is not None and not group.policy.baseline
        and controller._policy_identity(policy) == controller._policy_identity(group.policy)
        and (group.qualified or group.proposal is not None
             and group.probe_tokens < session.config.maximum_probe_tokens)
    )


def publish_server_policy(controller, session, policy, at_us):
    key = server_policy_key(session)
    group = controller._server_policies.get(key)
    if (group is None or group.owner_request_id != session.request_id
            or session.tail_seal_token is not None):
        return
    if controller._policy_identity(group.policy) != controller._policy_identity(policy):
        group.policy = policy
        group.qualified = policy.baseline
        group.proposal = None if policy.baseline else policy
        group.probe_tokens = 0
        group.changed_at_us = at_us


def qualify_server_policy(controller, session, policy, token_index, at_us):
    group = controller._server_policies.get(server_policy_key(session))
    if (group is not None and group.owner_request_id == session.request_id
            and controller._policy_identity(group.policy) == controller._policy_identity(policy)
            and controller._qualifies(session, policy, token_index, at_us)):
        group.qualified = True
        group.proposal = None


def server_policy_pending(controller, session):
    group = controller._server_policies.get(server_policy_key(session))
    if group is None or group.owner_request_id == session.request_id:
        return False
    policy = _matching_policy(controller, session, group.policy)
    return policy is not None and policy != session.current_policy


def server_window_is_comparable(controller, session, boundary):
    key = server_policy_key(session)
    if key is None:
        return True
    group = controller._server_policies.get(key)
    peers = [row for row in (*controller._sessions.values(), *controller._sealed_sessions.values())
             if server_policy_key(row) == key]
    return bool(
        group is not None and group.changed_at_us <= boundary.started_at_us
        and len(peers) == session.active_batch
        and all(row.awaiting_control is None and row.current_policy is not None
                and controller._policy_identity(row.current_policy)
                    == controller._policy_identity(boundary.policy) for row in peers)
    )


def server_policy_acknowledged(controller, session, at_us):
    group = controller._server_policies.get(server_policy_key(session))
    if group is not None:
        group.changed_at_us = max(group.changed_at_us, at_us)


def _coherence_key(session: _AdaptiveSession) -> tuple[str, str]:
    return session.model_artifact_sha256, session.baseline.desktop_placement_sha256


def _running_policy(session: _AdaptiveSession) -> AdaptiveDecodePolicy | None:
    if session.awaiting_control is not None:
        return session.awaiting_control.policy
    return session.current_policy


def _coherent_phone_policy(
    controller, session: _AdaptiveSession
) -> AdaptiveDecodePolicy | None:
    """The candidate of `session` that matches a co-tenant's running phone policy, if any."""
    if session.config.server_policy_coherence:
        group = controller._server_policies.get(server_policy_key(session))
        return (None if group is None or group.policy.baseline else
                _matching_policy(controller, session, group.policy))
    key = _coherence_key(session)
    by_identity = {
        controller._policy_identity(candidate): candidate
        for candidate in session.candidates
        if not candidate.baseline
        and candidate.policy_hash not in session.eliminated_policy_reasons
    }
    if not by_identity:
        return None
    for other in controller._sessions.values():
        if other.request_id == session.request_id or _coherence_key(other) != key:
            continue
        running = _running_policy(other)
        if running is None or running.baseline:
            continue
        match = by_identity.get(controller._policy_identity(running))
        if match is not None:
            return match
    return None


def _follow_coherent_policy(
    controller, session: _AdaptiveSession, token_index: int, at_us: int
) -> AdaptiveDecodeDirective | None:
    """Issue the co-tenants' phone policy to a session exploiting the baseline.

    Called once a baseline window has been recorded, so the control lands on a window boundary and the
    grouped observation keeps contiguous coverage."""
    if session.config.server_policy_coherence:
        return server_directive(controller, session, token_index, at_us)
    if (
        session.state != "EXPLOITING"
        or session.current_policy is None
        or not session.current_policy.baseline
        or session.awaiting_control is not None
        or session.tail_seal_token is not None
        or not session.helper_available
        or not session.execution_context_available
        or controller._remaining_tokens(session, token_index)
            < 2 * session.config.minimum_window_tokens
    ):
        return None
    policy = _coherent_phone_policy(controller, session)
    if policy is None:
        return None
    session.incumbent_policy = policy
    session.incumbent_context_sha256 = controller._context_identity(session)
    session.zero_assistance_reason = None
    directive = controller._control(session, policy, token_index, at_us)
    return replace(directive, reason=COHERENCE_REASON)
