"""AdaptiveDecodeController promotion operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeDirective,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowReceipt,
)
from ..adaptive_decode_state import _AdaptiveSession
from .budgeting import comparable_measurement_targets


def _consider_incumbent(controller, session, policy, token_index, at_us) -> None:
    if (not session.helper_available or policy.baseline or policy not in session.candidates
            or session.operational_verification and session.verification_policy == policy
            or policy.policy_hash in session.eliminated_policy_reasons
            or not controller._qualifies(session, policy, token_index, at_us)):
        return
    incumbent = session.incumbent_policy
    if (incumbent is None or incumbent not in session.candidates
            or session.incumbent_context_sha256 != controller._context_identity(session)
            or controller._estimated_token_energy(session, policy) < controller._estimated_token_energy(session, incumbent)):
        session.incumbent_policy = policy
        session.incumbent_context_sha256 = controller._context_identity(session)


def _best_valid_policy(controller, session, token_index, at_us):
    if not session.helper_available or not session.execution_context_available:
        return session.baseline
    incumbent = session.incumbent_policy
    if (incumbent is not None and incumbent in session.candidates
            and session.incumbent_context_sha256 == controller._context_identity(session)
            and incumbent.policy_hash not in session.eliminated_policy_reasons
            and controller._qualifies(session, incumbent, token_index, at_us)):
        return incumbent
    session.incumbent_policy = None
    session.incumbent_context_sha256 = None
    for policy in session.probe_candidates:
        controller._consider_incumbent(session, policy, token_index, at_us)
    return session.incumbent_policy or controller._ticket_fallback(session) or session.baseline


def _continue_best(controller, session, token_index, at_us, reason):
    controller._state(session, "EXPLOITING")
    winner = controller._best_valid_policy(session, token_index, at_us)
    if winner.baseline:
        session.zero_assistance_reason = reason
    directive = (controller._open_window(session, winner, token_index, at_us, session.current_ack)
                 if session.current_policy == winner else controller._control(session, winner, token_index, at_us))
    return replace(directive, reason=reason)


def _host_evidence_count(controller, session, *, latency: bool = False) -> int:
    rows = (controller._current_valid_latency_records(session, session.baseline) if latency
            else controller._current_valid_records(session, session.baseline, operational=True))
    return len(rows) + session.historical_group_counts.get(
        controller._policy_identity(session.baseline), 0)


def _single_reference_probe(controller, session, policy) -> bool:
    """A probe candidate measured against one host window is re-tested with a second
    reference window (sequencing) before its latency bound can eliminate it."""
    return (
        session.state == "PROBING" and session.stage == "candidate"
        and policy == session.current_policy and policy != session.incumbent_policy
        and _host_evidence_count(controller, session, latency=True) < 2
    )


def _update_elimination(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
) -> None:
    if policy.baseline or policy.policy_hash in (
        session.eliminated_policy_reasons
    ):
        return
    candidate = controller._bounds(session, policy, operational=True)
    baseline = controller._bounds(
        session, session.baseline, operational=True
    )
    if candidate is None or baseline is None:
        return
    latency_limit = (
        baseline[4] * session.config.maximum_latency_ppm + 999_999
    ) // 1_000_000
    budget = session.verification_budget if session.operational_verification else session.probe_budget
    resolving = (session.stage == "resolving_evidence" and session.verification_policy == policy
                 and budget is not None and "resolution_candidate_target" in budget
                 and _evidence_count(controller, _verification_evidence(session), policy)
                     < budget["resolution_candidate_target"])
    mean_latency_passes = candidate[3] * 1_000_000 <= baseline[3] * session.config.maximum_latency_ppm
    if (candidate[4] > latency_limit and not (resolving and mean_latency_passes)
            and not _single_reference_probe(controller, session, policy)):
        session.eliminated_policy_reasons[policy.policy_hash] = (
            "LATENCY_BOUND_EXCEEDED"
        )
        return
    alternatives = [session.baseline]
    alternatives.extend(
        row for row in session.probe_candidates
        if row.policy_hash != policy.policy_hash
        and row.policy_hash not in session.eliminated_policy_reasons
    )
    for alternative in alternatives:
        bounds = controller._bounds(
            session, alternative, operational=True
        )
        if (policy != session.incumbent_policy
                and bounds is not None and candidate[1] >= bounds[2]):
            session.eliminated_policy_reasons[policy.policy_hash] = (
                "ENERGY_DOMINATED"
            )
            return


def _qualifies(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int,
    at_us: int,
) -> bool:
    if (
        policy.policy_hash in session.eliminated_policy_reasons
        or session.helper_evidence_state == "LEARNING"
        and not controller._learning_probe_improves(session, policy)
    ):
        return False
    comparable_records = (controller._valid_records if session.config.server_policy_coherence
                          else controller._current_valid_records)
    baseline_rows = comparable_records(
        session, session.baseline, operational=True
    )
    candidate_rows = comparable_records(
        session, policy, operational=True
    )
    current_energy_ready = (
        bool(baseline_rows) and bool(candidate_rows)
    )
    cached_latency_replay = False
    if (
        not current_energy_ready
        and not baseline_rows
        and not candidate_rows
        and session.cached_winner == policy
    ):
        baseline_latency = controller._current_latency_bounds(
            session, session.baseline, minimum_count=1
        )
        candidate_latency = controller._current_latency_bounds(
            session, policy, minimum_count=1
        )
        if baseline_latency is not None and candidate_latency is not None:
            latency_limit = (
                baseline_latency[1]
                * session.config.maximum_latency_ppm
                + 999_999
            ) // 1_000_000
            cached_latency_replay = candidate_latency[1] <= latency_limit
    if not current_energy_ready and not cached_latency_replay:
        return False
    baseline = controller._bounds(
        session, session.baseline, operational=True
    )
    candidate = controller._bounds(session, policy, operational=True)
    assert baseline is not None and candidate is not None
    required_upper = (
        baseline[1]
        * (1_000_000 - session.config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    if candidate[2] > required_upper:
        return False
    remaining = controller._remaining_tokens(session, token_index)
    deadline_budget = max(0, session.deadline_us - at_us)
    baseline_remaining = baseline[4] * remaining
    relative_budget = (
        baseline_remaining * session.config.maximum_latency_ppm
        + 999_999
    ) // 1_000_000
    remaining_latency_budget = (
        min(deadline_budget, relative_budget)
        if baseline_remaining <= deadline_budget
        else relative_budget
    )
    if candidate[4] * remaining > remaining_latency_budget:
        return False
    per_token_saving = max(0, baseline[1] - candidate[2])
    # Spent exploration remains in receipts; only new costs affect continuation.
    switching_cost = (0 if session.acknowledged_policy == policy
                      else session.config.transition_energy_uj)
    margin = (
        baseline[1]
        * remaining
        * session.config.minimum_energy_saving_ppm
        // 1_000_000
    )
    return per_token_saving * remaining > (
        switching_cost + margin
    )


def _learning_probe_improves(
    controller_type,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
) -> bool:
    if policy.baseline:
        return False
    baseline_energy = controller_type._valid_records(
        session, session.baseline, operational=True
    )
    candidate_energy = controller_type._valid_records(
        session, policy, operational=True
    )
    boundary_ids = {
        row.energy_boundary_id for row in baseline_energy
    } & {
        row.energy_boundary_id for row in candidate_energy
    }
    if not boundary_ids:
        return False
    baseline_energy = tuple(
        row for row in baseline_energy
        if row.energy_boundary_id in boundary_ids
    )
    candidate_energy = tuple(
        row for row in candidate_energy
        if row.energy_boundary_id in boundary_ids
    )
    baseline_latency = tuple(
        row for row in baseline_energy if row.measurement_eligible
    )
    candidate_latency = tuple(
        row for row in candidate_energy if row.measurement_eligible
    )
    if (
        not baseline_energy
        or not candidate_energy
        or not baseline_latency
        or not candidate_latency
        or not any(
            (row.completed_phone_calls or 0) > 0
            for row in candidate_energy
        )
    ):
        return False

    def energy_per_token(
        rows: Sequence[AdaptiveDecodeWindowReceipt],
    ) -> int:
        return max(
            1,
            sum(row.whole_fleet_energy_uj for row in rows)
            // sum(row.energy_token_count for row in rows),
        )

    def latency_per_token(
        rows: Sequence[AdaptiveDecodeWindowReceipt],
    ) -> int:
        return max(
            1,
            sum(row.finished_at_us - row.started_at_us for row in rows)
            // sum(row.token_count for row in rows),
        )

    return (
        energy_per_token(candidate_energy)
            < energy_per_token(baseline_energy)
        and latency_per_token(candidate_latency) * 1_000_000
            <= latency_per_token(baseline_latency) * session.config.maximum_latency_ppm
    )


def _qualification_needs_more_evidence(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int,
    at_us: int,
    *,
    check_budget: bool = True,
) -> bool:
    if (
        policy.policy_hash in session.eliminated_policy_reasons
        or check_budget and not controller._can_probe(session, token_index, at_us)
    ):
        return False
    if (not controller._current_valid_records(session, session.baseline, operational=True)
            or not controller._current_valid_records(session, policy, operational=True)):
        return True
    baseline = controller._bounds(
        session, session.baseline, operational=True
    )
    candidate = controller._bounds(session, policy, operational=True)
    if baseline is None or candidate is None:
        return True
    required_mean = (
        baseline[0]
        * (1_000_000 - session.config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    latency_limit = (
        baseline[3] * session.config.maximum_latency_ppm + 999_999
    ) // 1_000_000
    required_upper = (
        baseline[1]
        * (1_000_000 - session.config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    return (
        candidate[0] <= required_mean
        and candidate[3] <= latency_limit
        and candidate[2] > required_upper
    )


def _verification_evidence(session):
    return (replace(session, records=session.records[session.verification_record_start:],
                    context_record_start=0, historical_records={}, historical_group_counts={})
            if session.operational_verification else session)


def _evidence_count(controller, evidence, policy):
    return (len(controller._current_valid_records(evidence, policy, operational=True))
            + evidence.historical_group_counts.get(controller._policy_identity(policy), 0))


def _reserve_comparable_measurements(controller, session, policy, token_index, at_us):
    evidence = _verification_evidence(session)
    targets = comparable_measurement_targets(controller, evidence, policy)
    counts = tuple(_evidence_count(controller, evidence, row) for row in (session.baseline, policy))
    budget = None if targets is None else controller._measurement_pair_budget(
        session, policy, token_index, at_us,
        baseline_windows=max(0, targets[0] - counts[0]),
        candidate_windows=max(0, targets[1] - counts[1]))
    session.qualification_measurement_plan = {
        "policy_hash": policy.policy_hash, "context_sha256": controller._context_identity(session),
        "bound_kind": "heuristic_integer_sqrt", "counts_before": counts,
        "target_counts": targets, "normal_window_tokens": (
            controller._window_tokens(session, session.baseline, token_index),
            controller._window_tokens(session, policy, token_index)),
        "status": "RESERVED" if budget is not None else "INCONCLUSIVE",
        "reason": "COMPARABLE_MEASUREMENT_BLOCK" if budget is not None else "BOUND_RESOLUTION_UNAFFORDABLE",
    }
    if budget is None:
        session.verification_budget = None
        session.probe_budget = None
        session.probe_policy_hash = None
        session.verification_policy = None
        session.operational_verification = False
        session.stage = "evidence_inconclusive"
        directive = controller._continue_best(session, token_index, at_us, "INCONCLUSIVE")
        return controller._verification_result(session, directive, "INCONCLUSIVE", "BOUND_RESOLUTION_UNAFFORDABLE")
    budget.update(resolution_baseline_target=targets[0], resolution_candidate_target=targets[1])
    if session.operational_verification:
        session.verification_budget = budget
        session.verification_attempts += 1
    else:
        session.probe_budget = budget
        session.probe_policy_hash = policy.policy_hash
    session.verification_policy = policy
    session.challenger_policy = policy
    session.stage = "resolving_evidence"
    controller._state(session, "PROBING")
    return advance_comparable_measurements(controller, session, token_index, at_us)


def advance_comparable_measurements(controller, session, token_index, at_us):
    policy = session.verification_policy
    budget = session.verification_budget if session.operational_verification else session.probe_budget
    if policy is None or policy not in session.candidates or budget is None:
        return controller._incomplete_probe(session, token_index, at_us)
    if policy.policy_hash in session.eliminated_policy_reasons:
        return controller._finish_verification(session, policy, token_index, at_us)
    if at_us > budget["deadline_us"] or token_index >= budget["token_limit"]:
        session.qualification_measurement_plan["status"] = "INCOMPLETE"
        return (controller._defer_verification(session, token_index, at_us, "RESERVATION_EXHAUSTED")
                if session.operational_verification else controller._incomplete_probe(session, token_index, at_us))
    evidence = _verification_evidence(session)
    for row, key in ((session.baseline, "resolution_baseline_target"), (policy, "resolution_candidate_target")):
        if _evidence_count(controller, evidence, row) < budget[key]:
            return (controller._open_window(session, row, token_index, at_us, session.current_ack)
                    if session.current_policy == row else controller._control(session, row, token_index, at_us))
    session.qualification_measurement_plan["status"] = "MEASURED"
    return controller._finish_verification(session, policy, token_index, at_us)


def _finish_verification(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    token_index: int,
    at_us: int,
) -> AdaptiveDecodeDirective:
    operational = session.operational_verification
    evidence = _verification_evidence(session)
    if controller._qualifies(evidence, policy, token_index, at_us):
        session.verification_policy = None
        session.operational_verification = False
        session.incumbent_policy = policy
        session.incumbent_context_sha256 = controller._context_identity(session)
        session.challenger_policy = None
        controller._state(session, "EXPLOITING")
        directive = (controller._control(session, policy, token_index, at_us)
                     if session.current_policy != policy else controller._open_window(
                         session, policy, token_index, at_us, session.current_ack))
        return (controller._verification_result(session, directive, "VERIFIED", "CURRENT_PAIR_IMPROVES")
                if operational else directive)
    if controller._qualification_needs_more_evidence(
        evidence, policy, token_index, at_us, check_budget=False,
    ):
        if operational and (not controller._current_valid_records(evidence, evidence.baseline, operational=True)
                            or not controller._current_valid_records(evidence, policy, operational=True)):
            return controller._defer_verification(session, token_index, at_us, "PAIRED_EVIDENCE_INCOMPLETE")
        return _reserve_comparable_measurements(controller, session, policy, token_index, at_us)
    session.eliminated_policy_reasons[policy.policy_hash] = "CURRENT_PAIR_NOT_IMPROVED"
    session.verification_policy = None
    session.operational_verification = False
    directive = controller._continue_best(session, token_index, at_us, "MEASURED_REJECTION")
    return (controller._verification_result(session, directive, "REJECTED", "CURRENT_PAIR_NOT_IMPROVED")
            if operational else directive)


def _select_probe_winner(
    controller,
    session: _AdaptiveSession,
    token_index: int,
    at_us: int,
) -> AdaptiveDecodeDirective:
    leader = controller._leading_candidate(session)
    if leader is None:
        return controller._continue_best(session, token_index, at_us, "MEASURED_REJECTION")
    if controller._qualifies(session, leader, token_index, at_us):
        return controller._finish_verification(session, leader, token_index, at_us)
    if controller._qualification_needs_more_evidence(session, leader, token_index, at_us, check_budget=False):
        return _reserve_comparable_measurements(controller, session, leader, token_index, at_us)
    if not session.refinement_added:
        session.refinement_added = True
        refinements = controller._refinement_candidates(session, leader)
        if refinements:
            session.probe_candidates.extend(refinements)
            policy = session.probe_candidates[
                session.next_probe_index
            ]
            session.next_probe_index += 1
            session.stage = "candidate"
            return controller._control(
                session, policy, token_index, at_us
            )
    return controller._finish_verification(session, leader, token_index, at_us)
