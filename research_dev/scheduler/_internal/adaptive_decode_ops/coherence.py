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
With a content identity of the model's own phone shards, the decisions survive layout generations that
leave those shards unchanged (a re-provisioning of other sessions, a recurring layout); a different
layer set starts fresh. Without one, a group lives for one layout generation.
Only windows with matching batch composition can qualify or eliminate a policy, and every decision is
stored per batch composition: a phone verdict measured alone becomes the proposal when a co-tenant
joins, the owner then measures one like-for-like host window at the new batch size, and a rejection at
that size does not switch a later single-tenant stretch back to the host. The host policy is never a
verdict by itself (comparison windows, paused probes and recoveries keep the pending proposal); only a
measured pair, a like-for-like elimination, a failed phone control or window, or the last exhausted
probe attempt decides a batch composition. A pair whose means favor the phone but whose bounds overlap
is not a rejection: the owner keeps measuring within the shared probe budget, as does a rejection that
rests on one host window. The default path retains the earlier request-local coherence rule.

With several helper phones (policies carrying `device_layer_masks`) every batch composition also
chooses the device set, from measured evidence. The cold start probes the primary phone alone, the
cheapest probe; once a set holds the verdict, each set that adds co-helpers challenges it and
replaces it only when it beats the incumbent by the unchanged bounds (energy upper below the
incumbent's lower minus the minimum saving, latency within the host bound). A set that measures
worse is dropped for that composition; a failed set is dropped with its supersets for every
composition, and the server falls back to the remaining sets (primary first, then sets without the
primary) before the host becomes the verdict. One phone keeps every rule above unchanged.

A desktop-parent joiner (continuous join) starts without a ready helper: its helper is attached at
fraction 0 at a decode boundary, which makes the session helper-available, and it follows the group's
phone policy at its next boundary. That attachment is eligible whenever a group of its model and
desktop parent runs a phone policy (`joiner_group_runs_phone_policy`), without a probe budget of its
own; the shared helper window then reuses the group's leases (`SERVER_HELPER_LEASES_SHARED`).

A quarantined device (lost at runtime or absent at start, reported by the rig) is dropped with every
set that contains it for every composition of every group, including groups created later; its
verdicts and proposals go, and a server running it returns to the host. Readmission removes every
drop and exhausted attempt of its sets (and the host verdicts that failures or the quarantine
decided), so the normal cold-start order probes it again on fresh evidence.

With batch_growth_verdict_inheritance (opt-in), a composition without a measured verdict inherits
the phone verdict of the nearest smaller composition as its provisional decision
(BATCH_GROWTH_INHERITED). Decode is bandwidth-bound on every stage (measured on this rig: the
desktop step costs 611 ms at batch 1 and 615-632 ms at batch 4; an OnePlus 15 FFN call costs 9.6 ms
for one row and 12.7 ms for four), so the host streaming a phone saves per step is the same at b+1
while the phone's own cost grows about 8 % per row: a phone win at b is a win at b+1. The
like-for-like probe at the larger composition still runs within the shared budget, and a measured
rejection there (or a monitored elimination of the inherited verdict) still returns the server to
the host; an exhausted budget keeps the inherited policy instead of the host. Hardware runs s1b and
s1d (2026-09-28) spent the whole budget while the joiner was still prefilling, so no window was
comparable and the pair lost the phone for its ~200 s co-decode. A smaller composition never
inherits from a larger one, a host verdict at the nearest smaller composition blocks inheritance,
and verdicts never leave their group (`server_policy_key`: model artifact, desktop placement, layout
identity or generation and geometry; a helper-unavailable session has no group).
"""
from __future__ import annotations

from dataclasses import replace
from math import isqrt

from ..adaptive_decode_contracts import (
    AdaptiveDecodeDirective, AdaptiveDecodePolicy, device_set_order, policy_device_set,
)
from ..adaptive_decode_state import _AdaptiveSession
from ..adaptive_decode_state import _AdaptiveServerPolicy
from . import budgeting as _budgeting

COHERENCE_REASON = "SERVER_POLICY_COHERENCE"
# The server runs the phone verdict its composition inherited from a smaller one.
INHERITED_REASON = "BATCH_GROWTH_INHERITED"
DEVICE_SET_FAILED = "SERVER_PHONE_POLICY_FAILED"
DEVICE_QUARANTINED = "SERVER_DEVICE_QUARANTINED"
# Drops that say a phone cannot run (not that it measured worse): they open the sets without it.
_UNAVAILABLE_DROPS = frozenset({DEVICE_SET_FAILED, DEVICE_QUARANTINED})
# The request-local rule (server_policy_coherence off): a request follows a co-tenant's phone policy.
CO_TENANT_REASON = "CO_TENANT_POLICY_FOLLOW"


def server_policy_key(session: _AdaptiveSession) -> tuple | None:
    if (not session.config.server_policy_coherence or not session.helper_available
            or session.helper_layout_generation is None
            or session.helper_layout_geometry_sha256 is None):
        return None
    if session.helper_layout_identity_sha256 is not None:
        return (session.model_artifact_sha256, session.baseline.desktop_placement_sha256,
                None, session.helper_layout_identity_sha256)
    return (session.model_artifact_sha256, session.baseline.desktop_placement_sha256,
            session.helper_layout_generation, session.helper_layout_geometry_sha256)


def server_verdict(group: _AdaptiveServerPolicy, active_batch: int) -> AdaptiveDecodePolicy | None:
    """The measured decision for one batch composition, if any."""
    return next((policy for batch, policy in group.verdicts if batch == active_batch), None)


def server_reason(group: _AdaptiveServerPolicy, active_batch: int) -> str | None:
    """Why the server last ran (or decided) the host policy for one batch composition."""
    return next((reason for batch, reason in group.reasons if batch == active_batch), None)


def _set_reason(group, active_batch, reason):
    group.reasons = (*(row for row in group.reasons if row[0] != active_batch),
                     *(() if reason is None else ((active_batch, reason),)))


def _set_verdict(group, active_batch, policy, reason=None):
    group.verdicts = (*(row for row in group.verdicts if row[0] != active_batch), (active_batch, policy))
    group.proposal = None
    _set_reason(group, active_batch, None if not policy.baseline else reason)
    _clear_inherited(group, active_batch)


def _nearest_smaller_verdict(group, active_batch):
    """(batch, policy) of the largest composition below `active_batch` with a verdict, or None."""
    return max((row for row in group.verdicts if row[0] < active_batch),
               key=lambda row: row[0], default=None)


def _growth_inheritance(controller, session, group):
    """(inherited_from_batch, policy) when the session's composition inherits a smaller one's phone
    verdict (batch_growth_verdict_inheritance), else None.

    The source is the nearest smaller composition with a verdict; a host verdict there blocks
    inheritance. The policy must be a candidate of this session that is not eliminated at this
    composition, and with several phones its device set must not be dropped here (failures and
    quarantines drop it for every composition)."""
    if not session.config.batch_growth_verdict_inheritance or not session.helper_available:
        return None
    source = _nearest_smaller_verdict(group, session.active_batch)
    if source is None or source[1].baseline:
        return None
    policy = _matching_policy(controller, session, source[1])
    if (policy is None or policy.policy_hash in session.eliminated_policy_reasons
            or (group.device_sets and _drop_reason(
                group, session.active_batch, policy_device_set(policy)) is not None)):
        return None
    return source[0], policy


def inherited_from_batch(group: _AdaptiveServerPolicy, active_batch: int) -> int | None:
    """The smaller composition whose phone verdict `active_batch` runs by inheritance, while that
    verdict stands."""
    source = next((source for batch, source in group.inherited if batch == active_batch), None)
    verdict = None if source is None else server_verdict(group, source)
    return None if verdict is None or verdict.baseline else source


def _mark_inherited(group, active_batch, source):
    group.inherited = (*(row for row in group.inherited if row[0] != active_batch),
                       (active_batch, source))


def _clear_inherited(group, active_batch):
    group.inherited = tuple(row for row in group.inherited if row[0] != active_batch)


def _keep_inherited_policy(controller, session, group):
    """An exhausted probe at an inheriting composition keeps the inherited phone policy as its
    verdict (BATCH_GROWTH_INHERITED) instead of the host; None when nothing is inherited.

    The budget bounds unqualified phone execution, and a policy that won at a smaller composition
    is not unqualified here (module docstring: decode is bandwidth-bound on every stage)."""
    inherited = _growth_inheritance(controller, session, group)
    if inherited is None:
        return None
    source, policy = inherited
    _set_verdict(group, session.active_batch, policy)
    _mark_inherited(group, session.active_batch, source)
    group.probe_tokens = 0
    return policy


def _decision_reason(controller, session, group, policy):
    """BATCH_GROWTH_INHERITED while the server runs the phone policy its composition inherited."""
    source = inherited_from_batch(group, session.active_batch)
    if (source is None or policy.baseline or controller._policy_identity(policy)
            != controller._policy_identity(server_verdict(group, source))):
        return COHERENCE_REASON
    return INHERITED_REASON


def _count_probe_attempt(group, active_batch):
    count = next((value for batch, value in group.attempts if batch == active_batch), 0) + 1
    group.attempts = (*(row for row in group.attempts if row[0] != active_batch), (active_batch, count))
    return count


def _server_group(controller, session, at_us):
    key = server_policy_key(session)
    if key is None:
        return None
    group = controller._server_policies.get(key)
    if group is None:
        order = device_set_order(session.candidates)
        group = _AdaptiveServerPolicy(session.request_id, session.baseline, at_us,
                                      layout_identity_sha256=session.helper_layout_identity_sha256,
                                      device_sets=order if len(order) > 1 else ())
        group.records = tuple(row for row in session.records
                              if row.policy.baseline and row.measurement_eligible)
        running = _running_policy(session)
        if running is not None and not running.baseline:
            current = _matching_policy(controller, session, running)
            if current is not None:
                group.policy = group.proposal = current
        controller._server_policies[key] = group
    for device_id in _quarantined_devices(controller):
        _quarantine_group(group, device_id, session.baseline, at_us)
    owner = controller._sessions.get(group.owner_request_id)
    if owner is None or server_policy_key(owner) != key:
        group.owner_request_id = session.request_id
    if group.owner_request_id != session.request_id:
        return group
    verdict = server_verdict(group, session.active_batch)
    if verdict is not None:
        # A pending proposal belongs to another batch composition; it must not consume the
        # probe budget while this one runs its decided policy. With several phones the only
        # proposal next to a verdict is the set that challenges it by adding co-helpers.
        challenger = (None if not group.device_sets
                      else _challenger(controller, session, group, verdict))
        if challenger is None:
            group.proposal = None
        elif (group.proposal is None or controller._policy_identity(group.proposal)
                != controller._policy_identity(challenger)):
            group.proposal = challenger
            group.probe_tokens = 0
    elif (group.proposal is None
          and (inherited := _growth_inheritance(controller, session, group)) is not None):
        # batch_growth_verdict_inheritance: the nearest smaller composition's phone verdict keeps
        # running as the provisional decision while the like-for-like probe collects evidence.
        group.proposal, group.probe_tokens = inherited[1], 0
        _mark_inherited(group, session.active_batch, inherited[0])
    elif group.proposal is None and not group.policy.baseline:
        # A phone verdict from another batch composition is this one's proposal: it keeps
        # running while the like-for-like comparison is collected, bounded by the probe budget.
        # Several phones start the new composition from the cheapest eligible device set.
        group.proposal = (group.policy if not group.device_sets else _first_eligible_policy(
            controller, session, group, session.active_batch, group.policy.split_fraction_ppm)
            or group.policy)
        group.probe_tokens = 0
    elif (group.proposal is None and group.device_sets and _failed_here(group, session.active_batch)
            and (fallback := _fallback_proposal(controller, session, group)) is not None):
        # After a device failure the server drives the fallback: a new owner's own probe order
        # starts with the primary phone, which may be the phone that failed.
        group.proposal, group.probe_tokens = fallback, 0
    elif group.proposal is None and session.stage in {"server_policy", "server_probe"}:
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
    # The shared budget bounds unqualified phone execution; host windows (a follower still
    # attaching, the owner's comparison window) execute nothing unqualified.
    if group.proposal is not None and not receipt.policy.baseline and (
            not group.device_sets or controller._policy_identity(receipt.policy)
            == controller._policy_identity(group.proposal)):
        group.probe_tokens += receipt.token_count
    if receipt.measurement_eligible and receipt.output_valid and receipt.failure_reason is None:
        group.records = (*group.records, receipt)


def _evidence_count(controller, session, policy):
    return (len(controller._current_valid_records(session, policy, operational=True))
            + session.historical_group_counts.get(controller._policy_identity(policy), 0))


def _server_pair_next_measurement(controller, session, proposal, baseline_bounds, candidate_bounds):
    """(policy to measure next, reason) for a pair that failed the qualification test, or None if
    the failure is decisive.

    Means that favor the phone with overlapping bounds are inconclusive: the side below the next
    resolving target is measured, host first (`advance_comparable_measurements`); without such a
    target the side with the wider band (fewer windows by isqrt, host on ties) is measured, so the
    sides take turns one band step at a time. The shared probe budget bounds the phone windows.
    Means against the phone that rest on one host window get a second host reference unless the
    energy bounds already separate the pair (sequencing's single-reference rejection)."""
    if baseline_bounds is None or candidate_bounds is None:
        return None
    config = session.config
    counts = tuple(_evidence_count(controller, session, row) for row in (session.baseline, proposal))
    # The qualification test on means instead of bounds.
    if (controller._learning_probe_improves(session, proposal)
            and candidate_bounds[0] * 1_000_000
                <= baseline_bounds[0] * (1_000_000 - config.minimum_energy_saving_ppm)
            and candidate_bounds[3] * 1_000_000 <= baseline_bounds[3] * config.maximum_latency_ppm):
        targets = _budgeting.comparable_measurement_targets(controller, session, proposal)
        if targets is not None and any(count < target for count, target in zip(counts, targets)):
            host = counts[0] < targets[0]
        else:
            host = isqrt(counts[0]) <= isqrt(counts[1])
        return (session.baseline if host else proposal), "SERVER_PAIR_INCONCLUSIVE"
    if counts[0] < 2 and candidate_bounds[1] < baseline_bounds[2]:
        return session.baseline, "SERVER_REFERENCE_BASELINE"
    return None


def _server_probe_policy(controller, session, group, token_index, at_us):
    batch = session.active_batch
    proposal = _matching_policy(controller, session, group.proposal)
    if proposal is None:
        _set_verdict(group, batch, session.baseline, "SERVER_PROPOSAL_NOT_A_CANDIDATE")
        return session.baseline
    if group.device_sets and _drop_reason(group, batch, policy_device_set(proposal)) is not None:
        # A proposal of a dropped device set moves to the next eligible set.
        replacement = _first_eligible_policy(controller, session, group, batch,
                                             proposal.split_fraction_ppm)
        if replacement is None:
            _set_verdict(group, batch, session.baseline,
                         _drop_reason(group, batch, policy_device_set(proposal)))
            return session.baseline
        group.proposal, group.probe_tokens = replacement, 0
        proposal = replacement
    if proposal.policy_hash in session.eliminated_policy_reasons:
        # The owner's eliminations use evidence at this batch composition only.
        return _reject_proposal(controller, session, group, batch, proposal,
                                "SERVER_" + session.eliminated_policy_reasons[proposal.policy_hash])
    baseline = controller._valid_records(session, session.baseline, operational=True)
    candidate = controller._valid_records(session, proposal, operational=True)
    measure, reason = proposal, None
    if baseline and candidate:
        baseline_bounds = controller._bounds(session, session.baseline, operational=True)
        candidate_bounds = controller._bounds(session, proposal, operational=True)
        if (baseline_bounds is not None and candidate_bounds is not None
                and controller._learning_probe_improves(session, proposal)
                and candidate_bounds[2] * 1_000_000 <= baseline_bounds[1]
                    * (1_000_000 - session.config.minimum_energy_saving_ppm)
                and candidate_bounds[4] * 1_000_000 <= baseline_bounds[4]
                    * session.config.maximum_latency_ppm):
            _set_verdict(group, batch, proposal)
            return proposal
        pending = _server_pair_next_measurement(
            controller, session, proposal, baseline_bounds, candidate_bounds)
        if pending is None:
            return _reject_proposal(controller, session, group, batch, proposal,
                                    "SERVER_PAIR_NOT_IMPROVED")
        measure, reason = pending
    elif candidate:
        measure, reason = session.baseline, "SERVER_COMPARISON_HOST_WINDOW"
    if group.probe_tokens >= session.config.maximum_probe_tokens:
        kept = _keep_inherited_policy(controller, session, group)
        if kept is not None:
            return kept
        # An exhausted attempt is not a measurement: the next owner may probe again until the
        # attempt cap turns the exhaustion into the host verdict for this batch composition.
        group.proposal = None
        group.probe_tokens = 0
        _set_reason(group, batch, "SERVER_PROBE_BUDGET_EXHAUSTED")
        if (_count_probe_attempt(group, batch)
                >= session.config.maximum_probe_attempts_per_context):
            return _reject_proposal(controller, session, group, batch, proposal,
                                    "SERVER_PROBE_BUDGET_EXHAUSTED")
        return session.baseline
    if reason is not None:
        _set_reason(group, batch, reason)
    return measure


def _drop_reason(group, active_batch, devices):
    """Why a device set is dropped at one batch composition (batch 0 rows drop every composition)."""
    return next((reason for batch, row, reason in group.device_drops
                 if row == devices and batch in (0, active_batch)), None)


def _drop_device_set(group, active_batch, devices, reason):
    group.device_drops = (*(row for row in group.device_drops
                            if row[:2] != (active_batch, devices)), (active_batch, devices, reason))


def _eligible_device_sets(group, active_batch):
    """Sets not dropped at this composition in exploration order; the sets without the primary
    phone only once the primary alone failed."""
    primary = group.device_sets[0][0]
    primary_failed = (_drop_reason(group, active_batch, (primary,)) in _UNAVAILABLE_DROPS
                      or (0, (primary,), DEVICE_QUARANTINED) in group.device_drops)
    return [devices for devices in group.device_sets
            if _drop_reason(group, active_batch, devices) is None
            and (primary in devices or primary_failed)]


def _device_set_policy(controller, session, devices, fraction):
    rows = [row for row in session.candidates
            if policy_device_set(row) == devices and row.split_fraction_ppm == fraction
            and row.policy_hash not in session.eliminated_policy_reasons]
    return max(rows, key=lambda row: (len(row.layer_indices), row.columns, row.policy_hash),
               default=None)


def _first_eligible_policy(controller, session, group, active_batch, fraction):
    return next((policy for policy in (
        _device_set_policy(controller, session, devices, fraction)
        for devices in _eligible_device_sets(group, active_batch)) if policy is not None), None)


def _failed_here(group, active_batch):
    return any(reason in _UNAVAILABLE_DROPS and batch in (0, active_batch)
               for batch, _devices, reason in group.device_drops)


def _fallback_proposal(controller, session, group):
    """The first eligible set at its widest fraction."""
    for devices in _eligible_device_sets(group, session.active_batch):
        fractions = sorted({row.split_fraction_ppm for row in session.candidates
                            if policy_device_set(row) == devices}, reverse=True)
        for fraction in fractions:
            policy = _device_set_policy(controller, session, devices, fraction)
            if policy is not None:
                return policy
    return None


def _challenger(controller, session, group, verdict):
    """The next eligible set that adds co-helpers to the verdict's set, at its fraction."""
    if verdict.baseline:
        return None
    decided = set(policy_device_set(verdict))
    return next((policy for policy in (
        _device_set_policy(controller, session, devices, verdict.split_fraction_ppm)
        for devices in _eligible_device_sets(group, session.active_batch)
        if set(devices) > decided) if policy is not None), None)


def _mirror_device_set_drops(controller, session, group):
    """Dropped sets are eliminated for the session's current composition, so no per-request rule
    (qualification, dominance, the probe sweep) runs or compares against them."""
    if not group.device_sets:
        return
    for policy in session.candidates:
        reason = _drop_reason(group, session.active_batch, policy_device_set(policy))
        if reason is not None:
            session.eliminated_policy_reasons.setdefault(policy.policy_hash, reason)


def _reject_proposal(controller, session, group, active_batch, proposal, reason):
    """A decisive rejection of the probed proposal. One phone: the host verdict for this batch
    composition. Several phones: the proposal's set is dropped for it and the next eligible set is
    probed with a fresh budget; the host becomes the verdict once no set remains."""
    if not group.device_sets:
        _set_verdict(group, active_batch, session.baseline, reason)
        return session.baseline
    _drop_device_set(group, active_batch, policy_device_set(proposal), reason)
    _mirror_device_set_drops(controller, session, group)
    _clear_inherited(group, active_batch)
    following = _first_eligible_policy(controller, session, group, active_batch,
                                       proposal.split_fraction_ppm)
    if following is None:
        _set_verdict(group, active_batch, session.baseline, reason)
        return session.baseline
    group.proposal, group.probe_tokens = following, 0
    group.attempts = tuple(row for row in group.attempts if row[0] != active_batch)
    _set_reason(group, active_batch, reason)
    return following


def _count_device_attempt(group, active_batch, devices):
    count = next((value for batch, row, value in group.device_attempts
                  if (batch, row) == (active_batch, devices)), 0) + 1
    group.device_attempts = (*(row for row in group.device_attempts
                               if row[:2] != (active_batch, devices)), (active_batch, devices, count))
    return count


def _server_challenge_policy(controller, session, group, incumbent, challenger):
    """A set that adds co-helpers against the incumbent verdict of this batch composition.

    It replaces the incumbent only when its energy upper bound is below the incumbent's lower bound
    minus the minimum saving (and the host's, when measured) and its latency upper bound stays within
    the host latency bound (the incumbent's without host evidence). Means that favor it with
    overlapping bounds keep measuring alternately within the shared probe budget; means against it
    that rest on one incumbent window take a second one; otherwise, or after the last exhausted
    attempt, it is dropped for this composition and the incumbent keeps running."""
    batch, config = session.active_batch, session.config
    devices = policy_device_set(challenger)
    measure = challenger
    incumbent_rows = controller._valid_records(session, incumbent, operational=True)
    challenger_rows = controller._valid_records(session, challenger, operational=True)
    if challenger_rows and not incumbent_rows:
        measure = incumbent
    elif challenger_rows:
        current = controller._bounds(session, incumbent, operational=True)
        candidate = controller._bounds(session, challenger, operational=True)
        host = controller._bounds(session, session.baseline, operational=True)
        if current is not None and candidate is not None:
            latency_reference = host or current
            saving = 1_000_000 - config.minimum_energy_saving_ppm
            if (candidate[2] * 1_000_000 <= current[1] * saving
                    and (host is None or candidate[2] * 1_000_000 <= host[1] * saving)
                    and candidate[4] * 1_000_000 <= latency_reference[4] * config.maximum_latency_ppm):
                _set_verdict(group, batch, challenger)
                return challenger
            counts = tuple(_evidence_count(controller, session, row) for row in (incumbent, challenger))
            latency_mean_passes = (candidate[3] * 1_000_000
                                   <= latency_reference[3] * config.maximum_latency_ppm)
            if latency_mean_passes and candidate[0] * 1_000_000 <= current[0] * saving:
                # inconclusive: the side with the wider band next, the incumbent on ties
                measure = incumbent if isqrt(counts[0]) <= isqrt(counts[1]) else challenger
            elif latency_mean_passes and counts[0] < 2 and candidate[1] < current[2]:
                measure = incumbent
            else:
                _drop_device_set(group, batch, devices, "SERVER_DEVICE_SET_LATENCY_BOUND_EXCEEDED"
                                 if not latency_mean_passes else "SERVER_DEVICE_SET_NOT_IMPROVED")
                _mirror_device_set_drops(controller, session, group)
                group.proposal, group.probe_tokens = None, 0
                return incumbent
    if group.probe_tokens >= config.maximum_probe_tokens:
        group.proposal, group.probe_tokens = None, 0
        if _count_device_attempt(group, batch, devices) >= config.maximum_probe_attempts_per_context:
            _drop_device_set(group, batch, devices, "SERVER_DEVICE_SET_PROBE_BUDGET_EXHAUSTED")
            _mirror_device_set_drops(controller, session, group)
        return incumbent
    return measure


def _reject_verdict(controller, session, group, token_index, at_us, decided, reason):
    """The monitored phone verdict of this composition no longer qualifies. Several phones: its set
    is dropped here and the next eligible set is probed against the host."""
    batch = session.active_batch
    _drop_device_set(group, batch, policy_device_set(decided), reason)
    _mirror_device_set_drops(controller, session, group)
    group.verdicts = tuple(row for row in group.verdicts if row[0] != batch)
    _clear_inherited(group, batch)
    following = _first_eligible_policy(controller, session, group, batch, decided.split_fraction_ppm)
    if following is None:
        _set_verdict(group, batch, session.baseline, reason)
        return session.baseline
    group.proposal, group.probe_tokens = following, 0
    group.attempts = tuple(row for row in group.attempts if row[0] != batch)
    _set_reason(group, batch, reason)
    return _server_probe_policy(controller, session, group, token_index, at_us)


def server_directive(controller, session, token_index, at_us):
    """Only the elected server owner probes; the other slots execute its decision."""
    group = _server_group(controller, session, at_us)
    if (group is None or session.tail_seal_token is not None
            or not session.execution_context_available):
        return None
    owner = group.owner_request_id == session.request_id
    _seed_server_evidence(controller, session, group)
    _mirror_device_set_drops(controller, session, group)
    verdict = server_verdict(group, session.active_batch)
    if owner and verdict is None and group.proposal is None:
        return None
    if owner and verdict is None:
        selected = _server_probe_policy(controller, session, group, token_index, at_us)
        if controller._policy_identity(selected) != controller._policy_identity(group.policy):
            group.changed_at_us = at_us
        group.policy = selected
        verdict = server_verdict(group, session.active_batch)
    elif owner:
        decided = _matching_policy(controller, session, verdict)
        challenger = (None if group.proposal is None or decided is None or decided.baseline
                      else _matching_policy(controller, session, group.proposal))
        if challenger is not None:
            verdict = _server_challenge_policy(controller, session, group, decided, challenger)
        elif decided is not None and not decided.baseline:
            controller._update_elimination(session, decided)
            if decided.policy_hash in session.eliminated_policy_reasons:
                reason = "SERVER_" + session.eliminated_policy_reasons[decided.policy_hash]
                if group.device_sets:
                    verdict = _reject_verdict(controller, session, group, token_index, at_us,
                                              decided, reason)
                else:
                    verdict = session.baseline
                    _set_verdict(group, session.active_batch, verdict, reason)
        if controller._policy_identity(verdict) != controller._policy_identity(group.policy):
            group.policy = verdict
            group.changed_at_us = at_us
        verdict = server_verdict(group, session.active_batch)
    policy = _matching_policy(controller, session, group.policy)
    if policy is None:
        return None
    probing = verdict is None or bool(
        group.device_sets and group.proposal is not None
        and controller._policy_identity(group.proposal) == controller._policy_identity(policy))
    session.stage = "server_probe" if probing else "server_policy"
    controller._state(session, "EXPLOITING")
    if session.current_policy is None:
        session.current_policy = session.baseline
    reason = server_reason(group, session.active_batch)
    if not policy.baseline:
        session.zero_assistance_reason = None
    elif reason is not None:
        session.zero_assistance_reason = reason
    directive = (
        controller._open_window(session, policy, token_index, at_us, session.current_ack)
        if session.current_policy == policy else
        controller._control(session, policy, token_index, at_us)
    )
    return replace(directive, reason=_decision_reason(controller, session, group, policy))


def server_helper_window_eligible(controller, session, policy):
    """The server runs `policy` by a measured verdict or by a bounded shared probe."""
    group = controller._server_policies.get(server_policy_key(session))
    if group is None or group.policy.baseline:
        return False
    identity = controller._policy_identity(policy)
    if identity != controller._policy_identity(group.policy):
        return False
    return bool(
        any(controller._policy_identity(row) == identity for _, row in group.verdicts)
        or group.proposal is not None
        and group.probe_tokens < session.config.maximum_probe_tokens
    )


def publish_server_policy(controller, session, policy, at_us):
    key = server_policy_key(session)
    group = controller._server_policies.get(key)
    if (group is None or group.owner_request_id != session.request_id
            or session.tail_seal_token is not None):
        return
    if controller._policy_identity(group.policy) != controller._policy_identity(policy):
        group.policy = policy
        group.changed_at_us = at_us
    # The host policy is never a verdict here: a comparison window, a paused probe or a recovery
    # leaves the pending proposal and the batch verdicts as they are.
    if (not policy.baseline and server_verdict(group, session.active_batch) is None
            and (group.proposal is None or controller._policy_identity(group.proposal)
                 != controller._policy_identity(policy))):
        group.proposal = policy
        group.probe_tokens = 0


def server_policy_failed(controller, session, at_us, policy=None):
    """A failed phone control or window ends the phone policy for this batch composition.

    With several helper phones the failed policy's device set and its supersets are dropped for
    every composition (which phone failed is not known), for the request's own controller as well;
    the server moves to the host and probes the next eligible set before the host is the verdict."""
    devices = () if policy is None else policy_device_set(policy)
    multiple = bool(devices) and len(device_set_order(session.candidates)) > 1
    if multiple:
        for row in session.candidates:
            if set(policy_device_set(row)) >= set(devices):
                session.eliminated_policy_reasons.setdefault(row.policy_hash, DEVICE_SET_FAILED)
    group = controller._server_policies.get(server_policy_key(session))
    if group is None:
        return
    if multiple and group.device_sets:
        for row in group.device_sets:
            if set(row) >= set(devices):
                _drop_device_set(group, 0, row, DEVICE_SET_FAILED)
        group.verdicts = tuple(verdict for verdict in group.verdicts
                               if verdict[1].baseline
                               or not set(policy_device_set(verdict[1])) >= set(devices))
        _mirror_device_set_drops(controller, session, group)
        following = _first_eligible_policy(controller, session, group, session.active_batch,
                                           policy.split_fraction_ppm)
        if following is not None:
            group.proposal, group.probe_tokens = following, 0
            group.attempts = tuple(row for row in group.attempts if row[0] != session.active_batch)
            _set_reason(group, session.active_batch, DEVICE_SET_FAILED)
            _clear_inherited(group, session.active_batch)
            if not group.policy.baseline:
                group.policy = session.baseline
                group.changed_at_us = at_us
            return
    _set_verdict(group, session.active_batch, session.baseline, "SERVER_PHONE_POLICY_FAILED")
    if not group.policy.baseline:
        group.policy = session.baseline
        group.changed_at_us = at_us


def _quarantined_devices(controller) -> tuple[str, ...]:
    return tuple(sorted(getattr(controller, "_quarantined_devices", {})))


def _uses_device(policy, device_id) -> bool:
    return device_id in policy_device_set(policy)


def _quarantine_group(group, device_id, baseline, at_us) -> bool:
    """Drop every set with the device for every composition; idempotent. Without a baseline (no
    session of the group is live) a group still running the device keeps its policy until the next
    owner resets it."""
    if not group.device_sets:
        return False
    changed = False
    for devices in group.device_sets:
        if device_id in devices and _drop_reason(group, 0, devices) is None:
            _drop_device_set(group, 0, devices, DEVICE_QUARANTINED)
            changed = True
    verdicts = tuple(row for row in group.verdicts
                     if row[1].baseline or not _uses_device(row[1], device_id))
    if verdicts != group.verdicts:
        group.verdicts, changed = verdicts, True
    if group.proposal is not None and _uses_device(group.proposal, device_id):
        group.proposal, group.probe_tokens, changed = None, 0, True
    if baseline is not None and _uses_device(group.policy, device_id):
        group.policy = baseline
        group.changed_at_us = max(group.changed_at_us, at_us)
        changed = True
    return changed


def eliminate_quarantined_policies(controller, session) -> None:
    """Keep a session's own (request-local) probing off quarantined devices: at start and whenever
    its candidates are replaced."""
    devices = set(_quarantined_devices(controller))
    quarantined = {row.policy_hash for row in session.candidates
                   if devices & set(policy_device_set(row))}
    if not quarantined:
        return
    for policy_hash in sorted(quarantined):
        session.eliminated_policy_reasons.setdefault(policy_hash, DEVICE_QUARANTINED)
    if session.cached_winner is not None and session.cached_winner.policy_hash in quarantined:
        session.cached_winner = None
    if (session.verification_policy is not None
            and session.verification_policy.policy_hash in quarantined):
        session.verification_policy = None
        session.operational_verification = False
        session.verification_budget = None
        session.stage = "initial_baseline"
    if session.prior_monitor_policy is not None and session.prior_monitor_policy.policy_hash in quarantined:
        session.prior_monitor_policy = None
    probes = [row for row in session.probe_candidates if row.policy_hash not in quarantined]
    if len(probes) != len(session.probe_candidates):
        session.probe_candidates = probes or controller._sample_candidates(
            [row for row in session.candidates if row.policy_hash not in quarantined],
            session.config, session.helper_evidence_state)
        session.next_probe_index = min(session.next_probe_index, len(session.probe_candidates))


def quarantine_device(controller, device_id: str, reason: str, at_us: int) -> bool:
    """A device left the fleet: every group drops its sets, running groups return to the host and
    live sessions eliminate its policies. Returns False when it was already quarantined."""
    if device_id in controller._quarantined_devices:
        return False
    controller._quarantined_devices[device_id] = reason
    for key, group in controller._server_policies.items():
        live = next((row for row in controller._sessions.values()
                     if server_policy_key(row) == key), None)
        _quarantine_group(group, device_id, None if live is None else live.baseline, at_us)
    for session in controller._sessions.values():
        eliminate_quarantined_policies(controller, session)
        group = controller._server_policies.get(server_policy_key(session))
        if group is not None:
            _mirror_device_set_drops(controller, session, group)
    return True


def readmit_device(controller, device_id: str) -> bool:
    """A device rejoined: its drops, exhausted attempts, the host verdicts decided by failures or the
    quarantine and the matching session eliminations are cleared, so the normal probe (the primary
    alone first, then supersets) decides on fresh evidence. False when it was not quarantined."""
    if controller._quarantined_devices.pop(device_id, None) is None:
        return False
    cleared = {DEVICE_QUARANTINED}
    for group in controller._server_policies.values():
        removed = tuple(row for row in group.device_drops if device_id in row[1])
        attempts = tuple(row for row in group.device_attempts if device_id not in row[1])
        if not removed and attempts == group.device_attempts:
            continue
        cleared.update(row[2] for row in removed)
        group.device_drops = tuple(row for row in group.device_drops if device_id not in row[1])
        group.device_attempts = attempts
        # a set that failed without the device still drops its supersets
        for batch, failed, reason in tuple(group.device_drops):
            if reason == DEVICE_SET_FAILED and batch == 0:
                for devices in group.device_sets:
                    if set(devices) > set(failed) and _drop_reason(group, 0, devices) is None:
                        _drop_device_set(group, 0, devices, DEVICE_SET_FAILED)
        undecided = {batch for batch, reason in group.reasons if reason in _UNAVAILABLE_DROPS}
        group.verdicts = tuple(row for row in group.verdicts
                               if not (row[1].baseline and row[0] in undecided))
    for session in controller._sessions.values():
        # request-local sessions keep their own failure eliminations; a group re-mirrors its drops
        reasons = cleared if server_policy_key(session) in controller._server_policies else {DEVICE_QUARANTINED}
        for policy in session.candidates:
            if (_uses_device(policy, device_id)
                    and session.eliminated_policy_reasons.get(policy.policy_hash) in reasons):
                del session.eliminated_policy_reasons[policy.policy_hash]
    return True


def qualify_server_policy(controller, session, policy, token_index, at_us):
    group = controller._server_policies.get(server_policy_key(session))
    if (group is not None and group.owner_request_id == session.request_id
            and controller._policy_identity(group.policy) == controller._policy_identity(policy)
            and controller._qualifies(session, policy, token_index, at_us)):
        _set_verdict(group, session.active_batch, policy)


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


def server_policy_snapshot(controller, session):
    """The shared server decision as seen by one session, for snapshots and decision records."""
    group = controller._server_policies.get(server_policy_key(session))
    if group is None:
        return None
    return {
        "owner_request_id": group.owner_request_id,
        "policy_hash": group.policy.policy_hash,
        "policy_fraction_ppm": group.policy.split_fraction_ppm,
        "proposal_hash": None if group.proposal is None else group.proposal.policy_hash,
        "probe_tokens": group.probe_tokens,
        "verdicts": {str(batch): policy.policy_hash for batch, policy in group.verdicts},
        "verdict_fractions_ppm": {str(batch): policy.split_fraction_ppm for batch, policy in group.verdicts},
        "attempts": {str(batch): count for batch, count in group.attempts},
        "reason": server_reason(group, session.active_batch),
        "reasons": {str(batch): reason for batch, reason in group.reasons},
        "records": len(group.records),
        "layout_identity_sha256": group.layout_identity_sha256,
        **({} if not group.inherited else {
            "inherited_from_batch": inherited_from_batch(group, session.active_batch),
            "inherited_batches": {str(batch): source for batch, source in group.inherited
                                  if inherited_from_batch(group, batch) is not None},
        }),
        **({} if not group.device_sets else {
            "device_sets": ["+".join(row) for row in group.device_sets],
            "policy_device_set": "+".join(policy_device_set(group.policy)),
            "proposal_device_set": (None if group.proposal is None
                                    else "+".join(policy_device_set(group.proposal))),
            "verdict_device_sets": {str(batch): "+".join(policy_device_set(policy))
                                    for batch, policy in group.verdicts},
            "device_set_drops": [{"active_batch": batch, "device_set": "+".join(devices),
                                  "reason": reason} for batch, devices, reason in group.device_drops],
            "device_set_attempts": [{"active_batch": batch, "device_set": "+".join(devices),
                                     "count": count} for batch, devices, count in group.device_attempts],
        }),
    }


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


def joiner_group_runs_phone_policy(controller, session: _AdaptiveSession) -> bool:
    """Whether a server group of the session's model and desktop parent runs a phone policy
    while the session carries no helper layout yet.

    A desktop-parent joiner starts without a ready helper, so its ``server_policy_key`` is None
    until the helper is attached at fraction 0 and ``helper_ready`` binds the layout; the group
    it will then follow is found by model and desktop placement alone. False for a session that
    already carries its layout: its exact group decides through ``_coherent_phone_policy``.
    """
    if not session.config.server_policy_coherence or server_policy_key(session) is not None:
        return False
    prefix = _coherence_key(session)
    return any(
        key[:2] == prefix and not group.policy.baseline
        for key, group in controller._server_policies.items()
    )


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
    return replace(directive, reason=CO_TENANT_REASON)
