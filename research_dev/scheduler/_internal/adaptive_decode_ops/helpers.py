"""AdaptiveDecodeController helpers operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodePolicy,
    device_set_order,
    policy_device_set,
    validate_policy_set,
)
from .common import HELPER_EVIDENCE_STATES
from .budgeting import adopts_late_helper as _adopts_late_helper, policy_evidence_key
from .coherence import server_helper_window_eligible


def helper_ready(
    controller,
    request_id: str,
    *,
    phone_layout_generation: int,
    phone_layout_geometry_sha256: str,
    candidates: Sequence[AdaptiveDecodePolicy] | None = None,
    component_capability_sha256: str | None = None,
    ticket_policy: AdaptiveDecodePolicy | None = None,
    helper_evidence_state: str | None = None,
    ready_at_token_index: int | None = None,
    allow_assumed_phone_power_for_operational_selection: bool | None = None,
    phone_layout_identity_sha256: str | None = None,
) -> None:
    """Make one existing session eligible to probe at its next boundary."""

    with controller._lock:
        session = controller._session(request_id)
        if (
            allow_assumed_phone_power_for_operational_selection is not None
            and type(allow_assumed_phone_power_for_operational_selection) is not bool
        ):
            raise AdaptiveDecodeError("adaptive helper power policy is invalid")
        if ready_at_token_index is not None and (
            type(ready_at_token_index) is not int
            or ready_at_token_index < 0
        ):
            raise AdaptiveDecodeError(
                "adaptive helper ready token is invalid"
            )
        if (
            type(phone_layout_generation) is not int
            or phone_layout_generation < 1
            or not controller._valid_sha256(phone_layout_geometry_sha256)
            or phone_layout_identity_sha256 is not None
            and not controller._valid_sha256(phone_layout_identity_sha256)
        ):
            raise AdaptiveDecodeError(
                "adaptive helper readiness identity is invalid"
            )
        if (
            helper_evidence_state is not None
            and helper_evidence_state not in HELPER_EVIDENCE_STATES
        ):
            raise AdaptiveDecodeError(
                "adaptive helper evidence state is invalid"
            )
        if candidates is not None:
            rows = validate_policy_set(session.baseline, candidates)
            evidence_state = (
                session.helper_evidence_state
                if helper_evidence_state is None else
                helper_evidence_state
            )
            component = (
                session.component_capability_sha256
                if component_capability_sha256 is None
                else component_capability_sha256
            )
            if not controller._valid_sha256(component):
                raise AdaptiveDecodeError(
                    "adaptive helper component identity is invalid"
                )
            if evidence_state == "LEARNING":
                ticket_policy = None
            elif ticket_policy is not None and ticket_policy not in rows:
                raise AdaptiveDecodeError(
                    "adaptive helper ticket policy is not a candidate"
                )
            safe_baseline_rebind = (
                not session.helper_available
                and session.state in {
                    "PREPARING", "BASELINE", "RECOVERING", "EXPLOITING"
                }
                and session.awaiting_control is None
                and session.current_policy is not None
                and session.current_policy.baseline
            )
            if not safe_baseline_rebind:
                raise AdaptiveDecodeError(
                    "adaptive helper candidates changed after use"
                )
            session.helper_evidence_state = evidence_state
            session.candidates = rows
            session.component_capability_sha256 = component
            session.ticket_policy = ticket_policy
            session.probe_candidates = controller._sample_candidates(
                rows,
                session.config,
                session.helper_evidence_state,
            )
            session.next_probe_index = 0
            session.stage = "initial_baseline"
            session.refinement_added = (
                session.helper_evidence_state == "LEARNING"
            )
            session.cached_winner = None
            session.verification_policy = None
            session.operational_verification = False
            session.prior_monitor_policy = None
            session.eliminated_policy_reasons.clear()
        elif (
            helper_evidence_state is not None
            and helper_evidence_state
                != session.helper_evidence_state
        ):
            raise AdaptiveDecodeError(
                "adaptive helper evidence state changed after use"
            )
        if session.helper_available and (
            session.helper_layout_generation
                != phone_layout_generation
            or session.helper_layout_geometry_sha256
                != phone_layout_geometry_sha256
        ):
            raise AdaptiveDecodeError(
                "adaptive helper generation changed after attachment"
            )
        became_ready = not session.helper_available
        if (
            became_ready
            and session.phone_power_policy_from_capability
            and allow_assumed_phone_power_for_operational_selection is not None
        ):
            session.config = replace(
                session.config,
                allow_assumed_phone_power_for_operational_selection=(
                    allow_assumed_phone_power_for_operational_selection
                ),
            )
        if became_ready:
            session.helper_layout_identity_sha256 = phone_layout_identity_sha256
        session.helper_available = True
        session.helper_layout_generation = phone_layout_generation
        session.helper_layout_geometry_sha256 = (
            phone_layout_geometry_sha256
        )
        session.history_helper_layout_geometry_sha256 = (
            phone_layout_geometry_sha256
        )
        if became_ready and session.state in {
            "PREPARING", "BASELINE", "RECOVERING", "EXPLOITING"
        } and session.current_policy is not None \
                and session.current_policy.baseline:
            session.historical_records.clear()
            session.historical_group_counts.clear()
            for policy in (session.baseline, *session.candidates):
                identity = controller._policy_identity(policy)
                historical, group_count = (
                    controller._historical_policy_records(
                        session, policy, compatible_context=True
                    )
                )
                if historical:
                    session.historical_records[identity] = historical
                    session.historical_group_counts[identity] = (
                        group_count
                    )
            controller._seed_verification_candidate(session)
            controller._state(
                session,
                "PROBING" if session.probe_candidates else "EXPLOITING",
            )
        if became_ready and (
            session.current_policy is not None
            and session.current_policy.baseline
            and session.window_start_token is not None
            and session.target_token is not None
        ):
            session.helper_ready_boundary_pending = True
        if (
            session.helper_ready_boundary_pending
            and ready_at_token_index is not None
            and session.current_policy is not None
            and session.current_policy.baseline
            and session.window_start_token is not None
            and session.target_token is not None
            and ready_at_token_index >= session.window_start_token
        ):
            session.target_token = min(
                session.target_token,
                max(
                    session.window_start_token + 1,
                    ready_at_token_index,
                ),
            )
            session.helper_ready_boundary_pending = False


def adopts_late_helper(controller, request_id: str, *, token_index: int) -> bool:
    """Whether a ready helper attached now is a late adoption for this session."""
    if type(token_index) is not int or token_index < 0:
        raise AdaptiveDecodeError("adaptive late adoption token index is invalid")
    with controller._lock:
        session = controller._sessions.get(request_id)
        return session is not None and _adopts_late_helper(controller, session, token_index)


def helper_unavailable(controller, request_id: str) -> None:
    """Prevent new phone controls while preserving the request."""

    with controller._lock:
        session = controller._session(request_id)
        session.helper_available = False
        session.helper_layout_generation = None
        session.helper_layout_geometry_sha256 = None
        session.helper_layout_identity_sha256 = None
        session.helper_ready_boundary_pending = False
        session.verification_budget = None
        session.verification_policy = None
        session.reference_policy = None
        session.prior_monitor_policy = None
        session.operational_verification = False
        session.probe_budget = None
        session.probe_policy_hash = None
        session.probe_retry_policy = None
        session.challenger_policy = None
        session.cached_winner = None
        session.pending_session_drain_policy = None
        session.pending_helper_refresh_policy = None
        session.context_monitor_prior = None
        session.deferred_policy = None
        session.deferred_control_reason = None
        session.incumbent_policy = None
        session.incumbent_context_sha256 = None
        session.policy_evidence_aliases.clear()
        session.evidence_context_component_sha256 = None
        session.zero_assistance_reason = "PHONE_HELPER_UNAVAILABLE"
        if session.current_policy is not None and (
            session.current_policy.baseline
        ):
            controller._state(session, "PREPARING")


def helper_disturbance(controller, request_id: str, *, reason: str | None) -> None:
    """Record work on the helper's phone (such as another session's load).

    Phone windows open while it runs are not evidence, and no probe or
    verification starts until it ends.
    """

    if reason is not None and (
        type(reason) is not str or not reason or not reason.isascii()
    ):
        raise AdaptiveDecodeError("adaptive helper disturbance is invalid")
    with controller._lock:
        session = controller._sessions.get(request_id)
        if session is None:
            return
        session.helper_disturbance = reason
        if (reason is not None and session.window_start_token is not None
                and session.window_disturbance is None):
            session.window_disturbance = reason


def request_helper_session_drain(
    controller,
    request_id: str,
    *,
    retained_layer_mask: int,
) -> Mapping[str, object]:
    """Request a retained-session mask at the next decode boundary."""

    if (
        type(retained_layer_mask) is not int
        or retained_layer_mask <= 0
        or retained_layer_mask >= 1 << 64
    ):
        raise AdaptiveDecodeError(
            "adaptive retained session layer mask is invalid"
        )
    with controller._lock:
        session = controller._session(request_id)
        source = (
            session.awaiting_control.policy
            if session.awaiting_control is not None else
            session.current_policy
        )
        if source is None:
            raise AdaptiveDecodeError(
                "adaptive session drain lacks a current policy"
            )
        order = device_set_order(session.candidates)
        if source.baseline or (
            len(order) > 1 and order[0][0] not in policy_device_set(source)
        ):
            # the retained mask narrows the primary phone's sessions; a device set
            # without the primary phone uses none of them
            return MappingProxyType({
                "already_applied": session.awaiting_control is None,
                "policy_hash": source.policy_hash,
                "retained_layer_mask": retained_layer_mask,
            })
        layer_mask = source.layer_mask & retained_layer_mask
        if layer_mask <= 0:
            raise AdaptiveDecodeError(
                "adaptive session drain removes all active layers"
            )
        if layer_mask == source.layer_mask:
            return MappingProxyType({
                "already_applied": session.awaiting_control is None,
                "policy_hash": source.policy_hash,
                "retained_layer_mask": layer_mask,
            })
        policy = replace(
            source,
            layer_indices=tuple(
                index for index in range(64)
                if layer_mask & (1 << index)
            ),
            layer_mask=layer_mask,
        )
        pending = session.pending_session_drain_policy
        if pending is not None and pending != policy:
            raise AdaptiveDecodeError(
                "adaptive session drain policy changed"
            )
        session.pending_session_drain_policy = policy
        session.pending_helper_refresh_policy = None
        session.maintenance_policy_hashes.add(policy.policy_hash)
        session.verification_budget = None
        session.operational_verification = False
        return MappingProxyType({
            "already_applied": False,
            "policy_hash": policy.policy_hash,
            "retained_layer_mask": layer_mask,
        })


def helper_rebound(
    controller,
    request_id: str,
    *,
    phone_layout_generation: int,
    phone_layout_geometry_sha256: str,
    candidates: Sequence[AdaptiveDecodePolicy],
    component_capability_sha256: str,
    ticket_policy: AdaptiveDecodePolicy | None,
    helper_evidence_state: str,
    compatible_layers_by_plan: Mapping[str, int] | None = None,
    phone_layout_identity_sha256: str | None = None,
) -> None:
    """Rebind an active retained-session policy after COW commit."""

    with controller._lock:
        session = controller._session(request_id)
        rows = validate_policy_set(session.baseline, candidates)
        compatible_masks = dict(compatible_layers_by_plan or {})
        if any(not controller._valid_sha256(key) or type(mask) is not int
               or not 0 <= mask < 1 << 64 for key, mask in compatible_masks.items()):
            raise AdaptiveDecodeError("adaptive retained execution identity is invalid")
        current = (
            session.current_policy
            if session.current_policy is not None else
            session.awaiting_control.policy
            if session.awaiting_control is not None else None
        )
        if (
            type(phone_layout_generation) is not int
            or phone_layout_generation < 1
            or not controller._valid_sha256(
                phone_layout_geometry_sha256
            )
            or not controller._valid_sha256(component_capability_sha256)
            or phone_layout_identity_sha256 is not None
            and not controller._valid_sha256(phone_layout_identity_sha256)
            or helper_evidence_state not in HELPER_EVIDENCE_STATES
            or not session.helper_available
            or current is None
        ):
            raise AdaptiveDecodeError(
                "adaptive active helper rebind is invalid"
            )
        exact_matching = () if current.baseline else tuple(
            row for row in rows
            if row.layer_mask == current.layer_mask
            and row.columns == current.columns
            and row.split_fraction_ppm
                == current.split_fraction_ppm
            and row.desktop_placement_sha256
                == current.desktop_placement_sha256
        )
        expanded_matching = (
            ()
            if current.baseline or exact_matching else
            tuple(
                row for row in rows
                if row.layer_mask & current.layer_mask
                    == current.layer_mask
                and row.layer_mask != current.layer_mask
                and row.columns == current.columns
                and row.split_fraction_ppm
                    == current.split_fraction_ppm
                and row.desktop_placement_sha256
                    == current.desktop_placement_sha256
            )
        )
        matching = exact_matching or expanded_matching
        if not current.baseline and len(matching) != 1:
            raise AdaptiveDecodeError(
                "adaptive active helper rebind geometry differs"
            )
        if helper_evidence_state == "LEARNING":
            ticket_policy = None
        elif ticket_policy is not None and ticket_policy not in rows:
            raise AdaptiveDecodeError(
                "adaptive rebound ticket policy is not a candidate"
            )
        if (
            session.candidates == rows
            and session.component_capability_sha256 == component_capability_sha256
            and session.ticket_policy == ticket_policy
            and session.helper_layout_generation == phone_layout_generation
            and session.helper_layout_geometry_sha256
                == phone_layout_geometry_sha256
            and session.helper_layout_identity_sha256
                == phone_layout_identity_sha256
            and session.helper_evidence_state == helper_evidence_state
        ):
            return
        # Publish only after every compatibility and pending-control check passes.
        session = controller._clone_session(session)
        if session.pending_session_drain_policy is not None:
            raise AdaptiveDecodeError("adaptive helper rebound policy changed")
        retained_policies = _retain_compatible_policy_evidence(
            controller, session, rows, compatible_masks
        )
        pending = session.pending_helper_refresh_policy
        if (pending is not None and pending.policy_hash in retained_policies
                and pending.policy_hash not in session.maintenance_policy_hashes):
            session.pending_helper_refresh_policy = retained_policies[pending.policy_hash]
        if (session.pending_helper_refresh_policy is not None
                and session.pending_helper_refresh_policy not in rows):
            raise AdaptiveDecodeError("adaptive helper rebound policy changed")
        if not retained_policies:
            session.evidence_context_component_sha256 = None
            session.context_monitor_prior = None
            session.deferred_policy = None
            session.deferred_control_reason = None
        continuity = bool(retained_policies) and (
            current.baseline or current.policy_hash in retained_policies
        )
        learning_reprobe = (
            helper_evidence_state == "LEARNING" and session.candidates != rows
            and not continuity
        )
        session.candidates = rows
        session.component_capability_sha256 = (
            component_capability_sha256
        )
        session.ticket_policy = ticket_policy
        session.helper_layout_generation = phone_layout_generation
        session.helper_layout_geometry_sha256 = (
            phone_layout_geometry_sha256
        )
        session.helper_layout_identity_sha256 = phone_layout_identity_sha256
        session.history_helper_layout_geometry_sha256 = (
            phone_layout_geometry_sha256
        )
        session.helper_evidence_state = helper_evidence_state
        session.helper_available = True
        if (continuity and session.context_monitor_prior is not None
                and session.operational_verification and session.verification_policy in rows):
            policy = session.verification_policy
            session.probe_candidates = [policy]
            if current != policy and not current.baseline:
                session.pending_helper_refresh_policy = policy
            controller._state(session, "PROBING")
            controller._sessions[request_id] = session
            return
        if continuity:
            token_index = session.window_start_token or 0
            at_us = session.window_start_us or 0
            winner = controller._best_valid_policy(session, token_index, at_us)
            if not winner.baseline:
                # The new envelope still needs its own physical acknowledgement.
                pending = session.pending_helper_refresh_policy
                if pending is not None and pending != winner:
                    raise AdaptiveDecodeError("adaptive helper rebound policy changed")
                session.pending_helper_refresh_policy = winner
                controller._state(session, "EXPLOITING")
                controller._sessions[request_id] = session
                return
            learning_reprobe = helper_evidence_state == "LEARNING"
        if current.baseline:
            session.probe_candidates = controller._sample_candidates(
                rows,
                session.config,
                helper_evidence_state,
            )
            session.next_probe_index = 0
            session.stage = "initial_baseline"
            session.refinement_added = helper_evidence_state == "LEARNING"
            session.cached_winner = None
            session.verification_policy = None
            session.operational_verification = False
            session.prior_monitor_policy = None
            if not retained_policies:
                session.eliminated_policy_reasons.clear()
        elif learning_reprobe:
            # A changed execution contract keeps the useful fraction as a
            # prior, not as inherited qualification: measure the running
            # fraction's expanded form and, if different, the measured
            # leader's, against the same baseline. No sweep restart and no
            # refilled exploration or verification budget.
            session.probe_candidates = _expanded_prior_candidates(
                controller, session, rows, matching[0]
            )
            session.next_probe_index = 1
            session.stage = "candidate"
            session.refinement_added = True
            session.cached_winner = None
            session.verification_policy = None
            session.operational_verification = False
            session.prior_monitor_policy = None
        if (
            matching
            and matching[0].policy_hash != current.policy_hash
        ):
            pending = session.pending_helper_refresh_policy
            if pending is not None and pending != matching[0]:
                raise AdaptiveDecodeError(
                    "adaptive helper rebound policy changed"
                )
            session.pending_helper_refresh_policy = matching[0]
        controller._state(
            session,
            "PROBING"
            if rows and (current.baseline or learning_reprobe) else "EXPLOITING",
        )
        controller._sessions[request_id] = session


def _expanded_prior_candidates(controller, session, rows, continuation):
    """The continuation of the running fraction plus the measured leader's."""
    leader = controller._leading_candidate(session)
    candidates = [continuation]
    if (leader is not None
            and leader.split_fraction_ppm != continuation.split_fraction_ppm):
        expanded = tuple(
            row for row in rows
            if row.layer_mask == continuation.layer_mask
            and row.split_fraction_ppm == leader.split_fraction_ppm
            and row.columns == leader.columns
            and row.desktop_placement_sha256 == leader.desktop_placement_sha256
        )
        if len(expanded) == 1 and expanded[0] != continuation:
            candidates.append(expanded[0])
    return candidates


def _retain_compatible_policy_evidence(controller, session, rows, compatible_masks):
    """Keep receipts immutable while rebinding their unchanged execution contract."""
    old_policies = (
        *session.candidates, *(row.policy for row in session.records),
        session.current_policy, session.incumbent_policy,
        session.deferred_policy,
    )
    replacements = {}
    for old in old_policies:
        layer_mask = 0 if old is None else compatible_masks.get(old.operator_plan_sha256, 0)
        if (old is None or old.baseline or old.layer_mask & layer_mask != old.layer_mask
                or not layer_mask):
            continue
        matching = tuple(row for row in rows if (
            row.layer_mask == old.layer_mask and row.columns == old.columns
            and row.split_fraction_ppm == old.split_fraction_ppm
            and row.desktop_placement_sha256 == old.desktop_placement_sha256
            and row.desktop_parent_route_id == old.desktop_parent_route_id
            and row.executor_id == old.executor_id
        ))
        if len(matching) != 1:
            continue
        new = matching[0]
        previous_key = policy_evidence_key(session, old)
        key = session.policy_evidence_aliases.get(new.policy_hash, previous_key)
        for alias, value in tuple(session.policy_evidence_aliases.items()):
            if value == previous_key:
                session.policy_evidence_aliases[alias] = key
        session.policy_evidence_aliases[old.policy_hash] = key
        session.policy_evidence_aliases[new.policy_hash] = key
        replacements[old.policy_hash] = new
        session.warmup_windows_seen_by_policy[new.policy_hash] = max(
            session.warmup_windows_seen_by_policy.get(new.policy_hash, 0),
            session.warmup_windows_seen_by_policy.get(old.policy_hash, 0),
        )
        session.warmup_latency_us_by_policy[new.policy_hash] = max(
            session.warmup_latency_us_by_policy.get(new.policy_hash, 0),
            session.warmup_latency_us_by_policy.get(old.policy_hash, 0),
        )
        if old.policy_hash in session.eliminated_policy_reasons:
            session.eliminated_policy_reasons[new.policy_hash] = (
                session.eliminated_policy_reasons[old.policy_hash]
            )
        old_identity = controller._policy_identity(old)
        new_identity = controller._policy_identity(new)
        if old_identity in session.historical_records:
            session.historical_records[new_identity] = session.historical_records[old_identity]
            session.historical_group_counts[new_identity] = (
                session.historical_group_counts.get(old_identity, 0)
            )
    if not replacements:
        return replacements
    session.evidence_context_component_sha256 = (
        session.evidence_context_component_sha256 or session.component_capability_sha256
    )
    for field in ("incumbent_policy", "cached_winner", "challenger_policy",
                  "verification_policy", "probe_retry_policy", "deferred_policy"):
        old = getattr(session, field)
        if old is not None:
            setattr(session, field, replacements.get(old.policy_hash))
    if session.deferred_policy is None:
        session.deferred_control_reason = None
    if session.probe_policy_hash in replacements:
        session.probe_policy_hash = replacements[session.probe_policy_hash].policy_hash
    session.probe_candidates = [
        replacements[row.policy_hash] for row in session.probe_candidates
        if row.policy_hash in replacements
    ]
    # A reduced mask may have valid windows collected while the load was running.
    session.probe_candidates.extend(row for row in rows if row not in session.probe_candidates)
    return replacements


def helper_window_bid(
    controller,
    request_id: str,
    *,
    requested_fraction_ppm: int | None = None,
) -> Mapping[str, object]:
    """Return one conservative bid for the shared helper window."""

    with controller._lock:
        session = controller._session(request_id)
        if requested_fraction_ppm is not None and (
            type(requested_fraction_ppm) is not int
            or not 0 < requested_fraction_ppm <= 1_000_000
        ):
            raise AdaptiveDecodeError(
                "adaptive helper bid fraction is invalid"
            )
        policy = None
        if requested_fraction_ppm is not None:
            active = (
                session.awaiting_control.policy
                if session.awaiting_control is not None else
                session.current_policy
            )
            policy = next(
                (
                    row for row in (active, *session.candidates)
                    if row is not None and row.split_fraction_ppm
                        == requested_fraction_ppm
                ),
                None,
            )
            if policy is None:
                raise AdaptiveDecodeError(
                    "adaptive helper bid fraction is unavailable"
                )
        elif (
            session.awaiting_control is not None
            and not session.awaiting_control.policy.baseline
        ):
            policy = session.awaiting_control.policy
        elif (
            session.current_policy is not None
            and not session.current_policy.baseline
        ):
            policy = session.current_policy
        elif (
            session.cached_winner is not None
            and not session.cached_winner.baseline
        ):
            policy = session.cached_winner
        else:
            policy = controller._leading_candidate(session)
        if policy is None:
            policy = min(
                session.candidates,
                key=lambda row: (
                    row.predicted_energy_per_token_uj
                    if row.predicted_energy_per_token_uj is not None
                    else 2**63 - 1,
                    row.predicted_latency_per_token_us
                    if row.predicted_latency_per_token_us is not None
                    else 2**63 - 1,
                    row.split_fraction_ppm,
                    row.policy_hash,
                ),
            )
        current_token = max(
            0,
            session.window_start_token
            if session.window_start_token is not None else
            session.transition_start_token
            if session.transition_start_token is not None else 0,
        )
        window_tokens = controller._window_tokens(
            session, policy, current_token
        )
        baseline_bounds = controller._bounds(
            session, session.baseline, operational=True
        )
        policy_bounds = controller._bounds(
            session, policy, operational=True
        )
        if baseline_bounds is not None and policy_bounds is not None:
            gain_per_token_uj = (
                baseline_bounds[1] - policy_bounds[2]
            )
            evidence = "CONSERVATIVE_MEASURED"
            baseline_latency_us = baseline_bounds[4]
            policy_latency_us = policy_bounds[4]
        else:
            baseline_energy = (
                session.baseline.predicted_energy_per_token_uj
            )
            policy_energy = policy.predicted_energy_per_token_uj
            gain_per_token_uj = (
                0
                if baseline_energy is None or policy_energy is None
                else baseline_energy - policy_energy
            )
            evidence = "PREDICTED"
            baseline_latency_us = controller._token_latency_us(
                session, session.baseline
            )
            policy_latency_us = controller._token_latency_us(
                session, policy
            )
        latency_eligible = (
            policy_latency_us * 1_000_000
            <= baseline_latency_us
                * session.config.maximum_latency_ppm
        )
        policy_records = tuple(
            row for row in session.records[session.context_record_start:]
            if row.policy.policy_hash == policy.policy_hash
        )
        learning_exploration_eligible = bool(
            session.helper_evidence_state == "LEARNING"
            and session.helper_available
            and (session.awaiting_control.policy if session.awaiting_control is not None
                 else session.current_policy) == policy
            and controller._probe_admitted(
                session,
                policy,
                current_token,
                session.transition_started_at_us
                if session.transition_started_at_us is not None
                else session.window_start_us or 0,
            )
        )
        return MappingProxyType({
            "evidence": evidence,
            "gain_per_token_uj": gain_per_token_uj,
            "gain_uj": (
                gain_per_token_uj * window_tokens
                if latency_eligible else -(2**63 - 1)
            ),
            "latency_eligible": latency_eligible,
            "latency_per_token_us": policy_latency_us,
            "learning_exploration_eligible": learning_exploration_eligible,
            "server_policy_eligible": server_helper_window_eligible(controller, session, policy),
            "policy_hash": policy.policy_hash,
            "policy_phone_window_count": len(policy_records),
            "phone_window_count": sum(
                not row.policy.baseline for row in session.records
            ),
            "requested_fraction_ppm": policy.split_fraction_ppm,
            "window_tokens": window_tokens,
        })


def active_policy(
    controller, request_id: str
) -> AdaptiveDecodePolicy | None:
    with controller._lock:
        return controller._session(request_id).current_policy


def yield_helper_window(
    controller,
    request_id: str,
    *,
    token_index: int,
    at_us: int,
) -> AdaptiveDecodeDirective:
    """Return a just-opened phone window to its desktop parent."""

    with controller._lock:
        session = controller._session(request_id)
        if (
            session.awaiting_control is not None
            or session.awaiting_boundary is not None
            or session.current_policy is None
            or session.current_policy.baseline
            or session.window_start_token != token_index
            or session.window_start_us != at_us
        ):
            raise AdaptiveDecodeError(
                "adaptive helper window cannot be yielded"
            )
        session.helper_available = False
        session.helper_layout_generation = None
        session.helper_layout_geometry_sha256 = None
        session.helper_layout_identity_sha256 = None
        controller._state(session, "RECOVERING")
        return controller._control(
            session, session.baseline, token_index, at_us
        )
