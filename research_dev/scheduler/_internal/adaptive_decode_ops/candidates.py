"""AdaptiveDecodeController candidates operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeConfig, AdaptiveDecodePolicy, device_set_order, policy_device_set,
)
from ..adaptive_decode_state import _AdaptiveSession


def _cached_verification_policy(
    controller, session: _AdaptiveSession, *, operational: bool = False
) -> AdaptiveDecodePolicy | None:
    baseline_identity = controller._policy_identity(session.baseline)
    if session.historical_group_counts.get(baseline_identity, 0) < 1:
        return None
    baseline = controller._bounds(session, session.baseline, operational=operational)
    if baseline is None:
        return None
    required_upper = (
        baseline[1]
        * (1_000_000 - session.config.minimum_energy_saving_ppm)
        // 1_000_000
    )
    latency_limit = (
        baseline[4] * session.config.maximum_latency_ppm + 999_999
    ) // 1_000_000
    final_policy_counts: dict[str, int] = {}
    context_bucket = session.context_length.bit_length()
    for grouped in controller._history.values():
        if not controller._group_matches_session(grouped, session):
            continue
        if operational and grouped.helper_layout_geometry_sha256 != session.helper_layout_geometry_sha256:
            continue
        identity = controller._policy_identity(grouped.final_policy)
        if any(
            row.output_valid
            and row.failure_reason is None
            and (controller._operational_energy_eligible(session, row)
                 if operational else row.energy_measurement_eligible)
            and row.active_batch == session.active_batch
            and row.context_length.bit_length() == context_bucket
            and controller._policy_identity(row.policy) == identity
            for row in grouped.windows
        ):
            final_policy_counts[identity] = (
                final_policy_counts.get(identity, 0) + 1
            )
    choices = []
    measured_choices = []
    for policy in session.candidates:
        identity = controller._policy_identity(policy)
        final_count = final_policy_counts.get(identity, 0)
        if session.historical_group_counts.get(identity, 0) < 1:
            continue
        bounds = controller._bounds(session, policy, operational=operational)
        if (
            bounds is not None
            and bounds[2] <= required_upper
            and bounds[4] <= latency_limit
        ):
            choice = (
                bounds[2],
                bounds[4],
                policy.policy_hash,
                policy,
            )
            measured_choices.append(choice)
            if final_count:
                choices.append((-final_count, *choice))
    if choices:
        return min(choices)[-1]
    return None if not measured_choices else min(measured_choices)[-1]


def _operational_verification_policy(
    controller, session: _AdaptiveSession
) -> AdaptiveDecodePolicy | None:
    if (not session.config.allow_assumed_phone_power_for_operational_selection
            or session.helper_layout_geometry_sha256 is None):
        return None
    # The exact prompt-length bucket first, then the nearest buckets with
    # evidence: a neighbouring bucket's measured winner is a starting point
    # for a fresh paired probe or monitoring, never qualified history.
    for context_length in controller._compatible_context_lengths(session):
        context_bucket = context_length.bit_length()
        records, counts = {}, {}
        for policy in (session.baseline, *session.candidates):
            identity = controller._policy_identity(policy)
            rows, groups = [], 0
            for group in controller._history.values():
                if (not controller._group_matches_session(group, session)
                        or group.helper_layout_geometry_sha256 != session.helper_layout_geometry_sha256):
                    continue
                selected = [row for row in group.windows
                            if row.output_valid and row.failure_reason is None
                            and row.measurement_eligible
                            and row.context_length.bit_length() == context_bucket
                            and row.active_batch == session.active_batch
                            and controller._policy_identity(row.policy) == identity
                            and controller._operational_energy_eligible(session, row)]
                if selected:
                    rows.extend(selected)
                    groups += 1
            records[identity], counts[identity] = tuple(rows), groups
        prior = replace(session, records=[], historical_records=records,
                        historical_group_counts=counts)
        winner = controller._cached_verification_policy(prior, operational=True)
        if winner is not None:
            return winner
    return None


def _seed_verification_candidate(controller, session: _AdaptiveSession) -> None:
    if session.operational_verification:
        session.verification_policy = None
        session.stage = "initial_baseline"
    session.operational_verification = False
    session.verification_budget = None
    session.prior_monitor_policy = None
    session.cached_winner = controller._cached_verification_policy(session)
    winner = session.cached_winner or controller._operational_verification_policy(session)
    if winner is not None:
        session.probe_candidates = [winner]
        session.refinement_added = True
        if session.cached_winner is None:
            session.operational_verification = True
            session.verification_policy = winner
            session.stage = "verification_baseline"


def _representative_candidates(
    candidates: Sequence[AdaptiveDecodePolicy],
) -> list[AdaptiveDecodePolicy]:
    # One representative per (device set, fraction); several helper phones keep one per set.
    by_fraction = {}
    for policy in candidates:
        key = (policy.split_fraction_ppm, policy_device_set(policy))
        current = by_fraction.get(key)
        if current is None or (
            len(policy.layer_indices), policy.columns, policy.policy_hash
        ) > (
            len(current.layer_indices),
            current.columns,
            current.policy_hash,
        ):
            by_fraction[key] = policy
    rank = {devices: index for index, devices in enumerate(device_set_order(candidates))}
    return [by_fraction[key] for key in sorted(
        by_fraction, key=lambda key: (key[0], rank.get(key[1], len(rank))))]


def _sample_candidates(
    controller_type,
    candidates: Sequence[AdaptiveDecodePolicy],
    config: AdaptiveDecodeConfig,
    helper_evidence_state: str = "TRUSTED",
) -> list[AdaptiveDecodePolicy]:
    rows = controller_type._representative_candidates(candidates)
    if not rows:
        return []
    order = device_set_order(rows)
    if len(order) > 1:
        return _sample_device_set_candidates(rows, order, config)
    return _sample_fractions(rows, config)


def _sample_device_set_candidates(
    rows: Sequence[AdaptiveDecodePolicy],
    order: Sequence[tuple[str, ...]],
    config: AdaptiveDecodeConfig,
) -> list[AdaptiveDecodePolicy]:
    """Cold start with several helper phones: the leading fraction of every set that contains the
    primary phone (primary alone first, then adding co-helpers), then the primary set's other
    fractions. Sets without the primary are failure fallbacks, never cold-start probes."""
    primary = order[0][0]
    sweeps = {
        devices: _sample_fractions([row for row in rows if policy_device_set(row) == devices], config)
        for devices in order if primary in devices
    }
    selected = [sweep[0] for sweep in sweeps.values() if sweep]
    selected.extend(sweeps[order[0]][1:])
    return selected[:min(4, config.maximum_probe_candidates)]


def _sample_fractions(
    rows: Sequence[AdaptiveDecodePolicy],
    config: AdaptiveDecodeConfig,
) -> list[AdaptiveDecodePolicy]:
    if not rows:
        return []
    by_fraction = {row.split_fraction_ppm: row for row in rows}
    selected = []
    selected_hashes = set()
    limit = min(4, config.maximum_probe_candidates)
    maximum_fraction = rows[-1].split_fraction_ppm
    for target in config.coarse_probe_fractions_ppm:
        policy = by_fraction.get(target)
        if (
            policy is None
            and target == 500_000
            and maximum_fraction < target
        ):
            policy = rows[-1]
        if (
            policy is None
            or policy.policy_hash in selected_hashes
        ):
            continue
        selected.append(policy)
        selected_hashes.add(policy.policy_hash)
        if len(selected) == limit:
            return selected
    for policy in reversed(rows):
        if policy.policy_hash in selected_hashes:
            continue
        selected.append(policy)
        selected_hashes.add(policy.policy_hash)
        if len(selected) == limit:
            break
    return selected


def _leading_candidate(
    controller, session: _AdaptiveSession
) -> AdaptiveDecodePolicy | None:
    measured = []
    for policy in session.probe_candidates:
        if policy.policy_hash in session.eliminated_policy_reasons:
            continue
        bounds = controller._bounds(session, policy, operational=True)
        if bounds is not None:
            measured.append((
                bounds[2],
                bounds[4],
                policy.columns,
                policy.policy_hash,
                policy,
            ))
    return None if not measured else min(measured)[-1]


def _ticket_fallback(
    session: _AdaptiveSession,
) -> AdaptiveDecodePolicy | None:
    if session.helper_evidence_state == "LEARNING":
        return None
    policy = session.ticket_policy
    if (
        policy is not None
        and policy.policy_hash not in session.eliminated_policy_reasons
    ):
        return policy
    return None


def _refinement_candidates(
    controller,
    session: _AdaptiveSession,
    leader: AdaptiveDecodePolicy,
) -> list[AdaptiveDecodePolicy]:
    rows = [row for row in controller._representative_candidates(session.candidates)
            if policy_device_set(row) == policy_device_set(leader)]
    selected_hashes = {
        row.policy_hash for row in session.probe_candidates
    }
    result = []
    for step in session.config.refinement_steps_ppm:
        for target in (
            leader.split_fraction_ppm - step,
            leader.split_fraction_ppm + step,
        ):
            if not 0 < target <= 1_000_000:
                continue
            available = [
                row for row in rows
                if row.policy_hash not in selected_hashes
                and row.policy_hash not in
                    session.eliminated_policy_reasons
            ]
            if not available:
                return result
            policy = min(available, key=lambda row: (
                abs(row.split_fraction_ppm - target),
                row.split_fraction_ppm,
                row.policy_hash,
            ))
            if (
                abs(policy.split_fraction_ppm - target)
                > max(1, step // 2)
            ):
                continue
            result.append(policy)
            selected_hashes.add(policy.policy_hash)
            if (
                len(session.probe_candidates) + len(result)
                >= session.config.maximum_probe_candidates
            ):
                return result
    return result
