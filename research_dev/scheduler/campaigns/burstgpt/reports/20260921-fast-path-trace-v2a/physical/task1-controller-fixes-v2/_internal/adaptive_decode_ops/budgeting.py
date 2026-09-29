"""AdaptiveDecodeController budgeting operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from math import isqrt

from ..adaptive_decode_contracts import AdaptiveDecodePolicy
from ..model_placement_contracts.layout import PhoneLayoutRequestImpact
from ..types import canonical_sha256
from ..adaptive_decode_state import _AdaptiveSession
from . import coherence


def _window_tokens(
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int,
) -> int:
    latency_us = controller_type._token_latency_us(session, policy)
    measurement_tokens = (
        2 * session.config.measurement_resolution_us + latency_us - 1
    ) // latency_us
    transition_tokens = (
        2 * session.config.transition_cost_us + latency_us - 1
    ) // latency_us
    target = max(
        session.config.minimum_window_tokens,
        measurement_tokens,
        transition_tokens,
    )
    target = min(target, session.config.maximum_window_tokens)
    remaining = max(1, session.output_tokens - token_index)
    return min(target, remaining)


def _remaining_tokens(session: _AdaptiveSession, token_index: int) -> int:
    return max(0, session.output_tokens - token_index)


def _can_probe(
    controller, session: _AdaptiveSession, token_index: int, at_us: int
) -> bool:
    active = (session.awaiting_control.policy
              if session.awaiting_control is not None else session.current_policy)
    if active is not None and controller._probe_admitted(session, active, token_index, at_us):
        return True
    return any(
        row.policy_hash not in session.eliminated_policy_reasons
        and controller._measurement_pair_budget(session, row, token_index, at_us) is not None
        for row in session.probe_candidates
    )


def _probe_admitted(session, policy, token_index, at_us) -> bool:
    budget = (session.verification_budget if session.operational_verification
              and session.verification_policy == policy else session.probe_budget
              if session.probe_policy_hash == policy.policy_hash else None)
    return bool(
        not policy.baseline and budget is not None
        and session.state == "PROBING"
        and session.helper_available
        and session.execution_context_available
        and policy.policy_hash not in session.eliminated_policy_reasons
        and token_index < budget["token_limit"] and at_us <= budget["deadline_us"]
        and session.probe_tokens < session.config.maximum_probe_tokens
    )


def helper_attachment_opportunity(controller, request_id: str, *, token_index: int, at_us: int) -> str:
    """Use the measurement admission budget before acquiring helper leases."""
    with controller._lock:
        session = controller._sessions.get(request_id)
        if session is None:
            return "ADAPTIVE_CONTEXT_PENDING"
        if not session.execution_context_available:
            return "EXECUTION_CONTEXT_UNAVAILABLE"
        if session.config.server_policy_coherence and controller._coherent_phone_policy(session) is not None:
            return "ELIGIBLE"
        if session.cached_winner is not None or controller._ticket_fallback(session) is not None:
            return "ELIGIBLE"
        if session.stage == "prior_monitor" and session.prior_monitor_policy is not None:
            return "ELIGIBLE"
        if any(controller._qualifies(session, row, token_index, at_us) for row in session.candidates):
            return "ELIGIBLE"
        if controller._can_probe(session, token_index, at_us):
            return "ELIGIBLE"
        if controller._prior_monitor_eligible(session, token_index):
            return "ELIGIBLE"
        return ("MEASURED_REJECTION" if session.candidates and all(
            row.policy_hash in session.eliminated_policy_reasons for row in session.candidates
        ) else "INSUFFICIENT_OPPORTUNITY")


def preview_helper_replacement(
    controller, request_id: str, *, retained_layer_mask: int,
    token_index: int, at_us: int, transition_latency_us: int,
) -> PhoneLayoutRequestImpact | None:
    """Price lost retained assistance without mutating policy, leases or evidence."""
    with controller._lock:
        session = controller._sessions.get(request_id)
        if session is None or token_index >= session.output_tokens or not session.helper_available:
            return None
        policy = session.incumbent_policy
        if policy is None or not controller._qualifies(session, policy, token_index, at_us):
            policy = session.current_policy
        if policy is None or policy.baseline:
            return None
        retained = retained_layer_mask & policy.layer_mask
        fields = dict(request_id=request_id,
                      remaining_tokens=controller._remaining_tokens(session, token_index),
                      current_layer_mask=policy.layer_mask, retained_layer_mask=retained)
        if retained == policy.layer_mask:
            return PhoneLayoutRequestImpact(**fields, evidence_reused=True, verification_feasible=True)
        if not retained:
            return PhoneLayoutRequestImpact(**fields, evidence_reused=False, verification_feasible=True,
                                           reason="REMOVED_BENEFIT_IN_SESSION_OBJECTIVE")
        prior = next((row for row in (*session.candidates, *(r.policy for r in session.records)) if (
            row.layer_mask == retained and row.columns == policy.columns
            and row.split_fraction_ppm == policy.split_fraction_ppm
            and row.desktop_placement_sha256 == policy.desktop_placement_sha256
            and row.desktop_parent_route_id == policy.desktop_parent_route_id
            and row.executor_id == policy.executor_id
            and row.operator_plan_sha256 == policy.operator_plan_sha256
            and controller._qualifies(session, row, token_index, at_us)
        )), None)
        if prior is not None:
            return PhoneLayoutRequestImpact(**fields, evidence_reused=True, verification_feasible=True,
                                           reason="EXACT_RETAINED_EVIDENCE_REUSED")
        preview = controller._clone_session(session)
        candidate = replace(policy, layer_mask=retained,
                            layer_indices=tuple(layer for layer in policy.layer_indices if retained & (1 << layer)),
                            predicted_energy_per_token_uj=None,
                            predicted_latency_per_token_us=None)
        # The changed mask has no qualification. Existing measurements are cost priors only.
        preview.incumbent_policy = None
        preview.operational_verification = True
        preview.verification_policy = candidate
        latency = controller._token_latency_us(session, policy)
        load_tokens = (transition_latency_us + latency - 1) // latency
        budget = controller._measurement_pair_budget(
            preview, candidate, token_index + load_tokens, at_us + transition_latency_us)
        if budget is None:
            return PhoneLayoutRequestImpact(**fields, evidence_reused=False, verification_feasible=False,
                                           reason="RETAINED_COMPLETE_PAIR_UNAFFORDABLE")
        baseline_energy = controller._estimated_token_energy(session, session.baseline) or 0
        incumbent_energy = controller._estimated_token_energy(session, policy) or baseline_energy
        lost = (max(0, baseline_energy - incumbent_energy) * budget["required_tokens"]
                * retained.bit_count() + policy.layer_mask.bit_count() - 1) // policy.layer_mask.bit_count()
        return PhoneLayoutRequestImpact(
            **fields, evidence_reused=False, verification_feasible=True,
            verification_tokens=budget["required_tokens"], verification_us=budget["required_us"],
            retained_assistance_loss_uj=lost, verification_overhead_uj=budget["required_energy_uj"],
            reason="RETAINED_COMPLETE_PAIR_RESERVED_ESTIMATE")


def _verification_control_cost(controller, session: _AdaptiveSession) -> tuple[int, int]:
    latency = controller._token_latency_us(session, session.baseline)
    cost = max(session.config.transition_cost_us, session.observed_control_cost_us)
    tokens = max(1, session.observed_control_tokens,
                 (session.config.transition_cost_us + latency - 1) // latency)
    groups = [session.records[session.context_record_start:]]
    groups.extend(group.windows for group in controller._history.values()
                  if controller._group_matches_session(group, session))
    for rows in groups:
        for before, after in zip(rows, rows[1:]):
            if (after.applied_ack is not None
                    and before.policy != after.policy
                    and not before.measurement_eligible
                    and before.token_end == after.applied_ack.applied_token_index
                    and before.active_batch == session.active_batch
                    and after.active_batch == session.active_batch
                    and not before.membership_changed and not after.membership_changed
                    and before.next_active_batch in (None, session.active_batch)
                    and after.next_active_batch in (None, session.active_batch)
                    and before.context_length.bit_length() == session.context_length.bit_length()):
                cost = max(cost, before.finished_at_us - before.started_at_us)
                tokens = max(tokens, before.token_count)
    return cost, tokens


def _measurement_pair_budget(
    controller, session: _AdaptiveSession, policy: AdaptiveDecodePolicy,
    token_index: int, at_us: int,
    *, baseline_windows: int | None = None, candidate_windows: int = 1,
) -> dict[str, int] | None:
    attempt_key = controller._probe_attempt_key(session, policy)
    if session.probe_attempts.get(attempt_key, 0) >= session.config.maximum_probe_attempts_per_context:
        return None
    remaining = controller._remaining_tokens(session, token_index)
    probe_tokens = session.probe_tokens
    server_key = coherence.server_policy_key(session)
    if server_key is not None:
        # The next owner can finish the pair; concurrent work is not additive.
        for other in controller._sessions.values():
            if (other is session or coherence.server_policy_key(other) != server_key
                    or not other.execution_context_available or other.tail_seal_token is not None):
                continue
            position = max(other.target_token or 0, other.transition_start_token or 0,
                           other.records[-1].token_end if other.records else 0)
            remaining = max(remaining, controller._remaining_tokens(other, position))
        group = controller._server_policies.get(server_key)
        if group is not None:
            probe_tokens = max(probe_tokens, group.probe_tokens)
    baseline_latency = controller._token_latency_us(session, session.baseline)
    baseline_work = baseline_latency * remaining
    relative_budget = baseline_work * session.config.maximum_latency_ppm // 1_000_000
    deadline_budget = max(0, session.deadline_us - at_us)
    # Use the same already-unattainable-SLO treatment as _qualifies().
    work_budget = (min(deadline_budget, relative_budget)
                   if baseline_work <= deadline_budget else relative_budget)
    if not session.operational_verification and deadline_budget > baseline_work:
        work_budget = deadline_budget
    allowance = work_budget * session.config.exploration_latency_budget_ppm // 1_000_000
    control_us, control_tokens = controller._verification_control_cost(session)
    required_us, required_tokens = 3 * control_us, 3 * control_tokens
    fresh_baseline = bool(
        controller._current_valid_records(session, session.baseline, operational=True)
        and (not session.operational_verification or (
            session.records and session.current_policy == session.baseline
            and session.records[-1] in controller._current_valid_records(
                session, session.baseline, operational=True))))
    if baseline_windows is not None:
        fresh_baseline = baseline_windows == 0
    baseline_energy = controller._estimated_token_energy(session, session.baseline)
    candidate_energy = controller._estimated_token_energy(session, policy)
    if baseline_energy is None:
        return None
    candidate_energy_prior = candidate_energy is None
    if candidate_energy_prior:
        candidate_energy = (baseline_energy * (1_000_000 + session.config.uncertainty_ppm)
                            + 999_999) // 1_000_000
    reference_energy = baseline_energy
    if (session.incumbent_policy is not None
            and session.incumbent_context_sha256 == controller._context_identity(session)):
        incumbent_energy = controller._estimated_token_energy(session, session.incumbent_policy)
        if incumbent_energy is not None:
            reference_energy = min(reference_energy, incumbent_energy)
    required_energy = 3 * session.config.transition_energy_uj
    for row in (session.baseline, policy):
        measurements = ((0 if fresh_baseline else 1) if baseline_windows is None
                        else baseline_windows) if row.baseline else candidate_windows
        if measurements == 0:
            continue
        warmups = max(0, session.config.warmup_windows_per_policy
                      - session.warmup_windows_seen_by_policy.get(row.policy_hash, 0))
        tokens = (warmups + measurements) * controller._window_tokens(session, row, token_index)
        required_tokens += tokens
        measured_latency = controller._token_latency_us(session, row)
        warmup_latency = max(measured_latency, session.warmup_latency_us_by_policy.get(row.policy_hash, 0))
        window_tokens = controller._window_tokens(session, row, token_index)
        required_us += window_tokens * (warmups * warmup_latency + measurements * measured_latency)
        required_energy += tokens * max(
            0, (baseline_energy if row.baseline else candidate_energy) - reference_energy)
    required_us = (required_us * (1_000_000 + session.config.uncertainty_ppm)
                   + 999_999) // 1_000_000
    required_energy = (required_energy * (1_000_000 + session.config.uncertainty_ppm)
                       + 999_999) // 1_000_000
    energy_allowance = max(0, baseline_energy * remaining
                           * session.config.exploration_energy_budget_ppm // 1_000_000
                           - controller._spent_exploration_energy(session))
    if (remaining < max(session.config.minimum_remaining_tokens,
                        required_tokens + session.config.minimum_window_tokens)
            or probe_tokens + required_tokens > session.config.maximum_probe_tokens
            or required_us > allowance or required_energy > energy_allowance):
        return None
    return {
        "started_at_us": at_us,
        "deadline_us": at_us + allowance,
        "required_us": required_us,
        "allowance_us": allowance,
        "required_tokens": required_tokens,
        "token_limit": token_index + required_tokens,
        "record_start": len(session.records) - int(fresh_baseline),
        "fresh_baseline": int(fresh_baseline),
        "required_energy_uj": required_energy,
        "energy_allowance_uj": energy_allowance,
        "candidate_energy_prior": int(candidate_energy_prior),
        "acknowledged": 0,
        "baseline_window_target": len(controller._current_valid_records(
            session, session.baseline, operational=True)) + (
                (0 if fresh_baseline else 1) if baseline_windows is None else baseline_windows),
        "candidate_window_target": len(controller._current_valid_records(
            session, policy, operational=True)) + candidate_windows,
    }


def pending_probe_measurement_us(controller, session, policy, token_index, budget):
    """Use observed warmup cost without turning warmup into qualifying evidence."""
    required = 0
    for row, key in ((session.baseline, "baseline_window_target"), (policy, "candidate_window_target")):
        missing = max(0, budget.get(key, 1) - len(controller._current_valid_records(
            session, row, operational=True)))
        if not missing:
            continue
        warmups = max(0, session.config.warmup_windows_per_policy
                      - session.warmup_windows_seen_by_policy.get(row.policy_hash, 0))
        latency = controller._token_latency_us(session, row)
        warmup_latency = max(latency, session.warmup_latency_us_by_policy.get(row.policy_hash, 0))
        required += controller._window_tokens(session, row, token_index) * (
            warmups * warmup_latency + missing * latency)
    return (required * (1_000_000 + session.config.uncertainty_ppm) + 999_999) // 1_000_000


def comparable_measurement_targets(controller, session, policy):
    """Forecast the next useful heuristic bound; never split existing windows."""
    baseline = controller._bounds(session, session.baseline, operational=True)
    candidate = controller._bounds(session, policy, operational=True)
    counts = [len(controller._current_valid_records(session, row, operational=True))
              + session.historical_group_counts.get(controller._policy_identity(row), 0)
              for row in (session.baseline, policy)]
    if baseline is None or candidate is None:
        return (max(1, counts[0]), max(1, counts[1]))
    maximum_count = max(counts) + session.config.maximum_probe_tokens // session.config.minimum_window_tokens
    for root in range(1, isqrt(maximum_count) + 1):
        targets = tuple(max(count, root * root) for count in counts)
        uncertainty = [max(20_000, session.config.uncertainty_ppm // max(1, isqrt(count)))
                       for count in targets]
        baseline_lower = baseline[0] * (1_000_000 - uncertainty[0]) // 1_000_000
        candidate_upper = (candidate[0] * (1_000_000 + uncertainty[1]) + 999_999) // 1_000_000
        required = baseline_lower * (1_000_000 - session.config.minimum_energy_saving_ppm) // 1_000_000
        if candidate_upper <= required:
            return targets
    return None


def _context_identity(session: _AdaptiveSession) -> str:
    return canonical_sha256({
        "artifact": session.model_artifact_sha256,
        "parent": session.baseline.desktop_placement_sha256,
        "component": (session.evidence_context_component_sha256
                      or session.component_capability_sha256),
        "context_bucket": session.context_length.bit_length(),
        "active_batch": session.active_batch,
        "membership_record_start": session.context_record_start,
    })


def _probe_attempt_key(controller, session, policy) -> str:
    return controller._context_identity(session) + ":" + policy_evidence_key(session, policy)


def policy_evidence_key(session: _AdaptiveSession, policy: AdaptiveDecodePolicy) -> str:
    """Alias only contracts checked against unchanged physical session identities."""
    return session.policy_evidence_aliases.get(policy.policy_hash, policy.policy_hash)


def _estimated_token_energy(controller, session, policy) -> int | None:
    bounds = controller._bounds(session, policy, operational=True)
    if bounds is not None:
        return bounds[1] if policy.baseline else bounds[2]
    return policy.predicted_energy_per_token_uj


def _spent_exploration_energy(controller, session) -> int:
    baseline = controller._estimated_token_energy(session, session.baseline) or 0
    by_batch = {}
    spent = 0
    for row in session.records:
        if row.policy.baseline and row.output_valid and row.failure_reason is None:
            by_batch[row.active_batch] = row.energy_per_token_uj
        if row.window_role == "exploration":
            spent += max(0, row.energy_per_token_uj - by_batch.get(row.active_batch, baseline)) * row.energy_token_count
            if row.applied_ack is not None:
                spent += session.config.transition_energy_uj
    return max(spent, session.exploration_overhead_high_water_uj)
