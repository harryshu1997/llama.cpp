"""Opt-in phone FFN re-provisioning that follows the desktop's model.

The arrived-work portfolio sums the queued decode work of every model, so with two
large models queued the learning selector keeps a static split of the HTP sessions
while the desktop serves one model at a time. With
``PhoneResidentModelReprovisioningConfiguration`` the portfolio instead:

- follows the phone-capable models the desktop is loading or executing (else the ones
  it holds hot, else the last followed ones): one followed model gets every session
  RAM allows, several share the sessions in proportion to their remaining decode work,
  and without desktop knowledge the sessions are split over all arrived work;
- holds the layout while the followed model has no remaining work, so a queued model is
  loaded when the desktop switches to it, not when it arrives;
- never replaces a session an acquired helper uses or that is mid-transition; that
  change waits for the release while free sessions change now;
- publishes one session per proposal through the existing copy-on-write transaction,
  confirmed at once while the desktop commitment is live, so the phone load overlaps
  the desktop load; and
- estimates the swap from observed SESSION_LOADING -> SESSION_VERIFIED windows; and
- re-evaluates at a decode boundary only when the state it reacts to changed, else at
  most once per ``boundary_reevaluation_interval_us``, and records a layout-keeping
  decision only when it differs from the last record (the others are counted in the
  next record). Dispatch, release and session-ready re-evaluations are not gated;
- with ``count_queued_demand`` also follows the phone-capable model whose queued
  requests are next in dispatch order (source ``queued``; the arrived work of those
  requests is the demand), so the swap can start before that model is dispatched. A
  queued model is phone-capable through its learning demand, which its phone routes
  would lose to ``MARGINAL_SYSTEM_COST_UNKNOWN`` while another model's work is
  protected on the desktop (exactly while its switch waits); under this knob the
  learning demand tolerates that rejection; and
- with ``early_on_transition`` re-evaluates when a plan carrying a desktop load is
  committed (the residency change is decided) and at every release while such a load
  is decided or pending, overlapping the swap with the wait and the desktop load.

Shard selection, hashes, certification and the physical transaction are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ..._internal.background_placement import material_count_bucket
from ..._internal.lifecycle import UnifiedScheduleError
from ..._internal.phone_shards import PhoneFfnResidencyLayout
from ..._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimePhoneSessionCapability,
)
from ..._internal.runtime_controller import RuntimeRequestTicket
from ..._internal.types import canonical_sha256
from ...configuration.campaign import PhoneResidentModelReprovisioningConfiguration
from ..common import _RECOVERABLE_ERRORS
from .common import (
    _PhoneCandidateChoice,
    _PhoneDemandDiscovery,
    _PhoneMemoryBudget,
    _PhoneQueueDemand,
)

_TERMINAL_STATES = frozenset({"CANCELLED", "COMPLETED", "FAILED"})
_IDLE_SESSION_STATES = frozenset({"EMPTY", "READY"})
_LIVE_SOURCES = frozenset({"loading", "executing", "resident", "queued"})
_NOT_STARTED_STATES = frozenset({"QUEUED", "REPLAN_REQUIRED", "REPLANNING", "DEFERRED_REPLAN"})

MODE_FOLLOW = "FOLLOW"
MODE_PROPORTIONAL = "PROPORTIONAL"
MODE_HOLD = "HOLD"

REASON_REPROVISION = "PHONE_RESIDENCY_DESKTOP_REPROVISION"
REASON_PROPORTIONAL = "PHONE_RESIDENCY_PROPORTIONAL_SPLIT"
REASON_RETAINED = "PHONE_RESIDENCY_REPROVISION_RETAINED"
REASON_DEFERRED_IN_USE = "PHONE_RESIDENCY_REPROVISION_DEFERRED_IN_USE"
REASON_DEFERRED_REVALIDATION = "PHONE_RESIDENCY_REPROVISION_DEFERRED_REVALIDATION"
REASON_HOLD = "PHONE_RESIDENCY_REPROVISION_HOLD_IDLE_MODEL"
_CHANGE_REASONS = frozenset({REASON_REPROVISION, REASON_PROPORTIONAL})
# layout-keeping decisions that are recorded only when they differ from the last record
_UNCHANGED_REASONS = frozenset({REASON_RETAINED, REASON_HOLD})
_DEFAULT_BOUNDARY_INTERVAL_US = 10_000_000
_COUNTER_FIELDS = frozenset({
    "boundary_evaluations_skipped", "boundary_evaluations_skipped_total",
    "unchanged_decisions_coalesced", "unchanged_decisions_coalesced_total",
})


@dataclass(frozen=True)
class _PhoneReprovisionContext:
    configuration: PhoneResidentModelReprovisioningConfiguration
    mode: str
    followed_artifacts: tuple[str, ...]
    split_artifacts: tuple[str, ...]
    commitment_source: str | None
    leader_request_id: str | None
    load_window_start_us: int | None
    load_window_end_us: int | None
    in_use_session_ids: tuple[str, ...]
    work_by_artifact: Mapping[str, int]
    load_bytes_per_second: int
    learned_load_samples: int

    @property
    def confirmed(self) -> bool:
        return self.mode != MODE_HOLD and self.commitment_source in _LIVE_SOURCES

    def to_json(self) -> dict[str, object]:
        return {
            "arrived_work_by_artifact": dict(sorted(self.work_by_artifact.items())),
            "desktop_commitment_source": self.commitment_source,
            "desktop_load_window_end_us": self.load_window_end_us,
            "desktop_load_window_start_us": self.load_window_start_us,
            "followed_artifact_sha256s": list(self.followed_artifacts),
            "in_use_session_ids": list(self.in_use_session_ids),
            "leader_request_id": self.leader_request_id,
            "learned_load_samples": self.learned_load_samples,
            "load_bytes_per_second": self.load_bytes_per_second,
            "mode": self.mode,
            "split_artifact_sha256s": list(self.split_artifacts),
        }


def configure_phone_resident_model_reprovisioning(controller, configuration) -> None:
    if configuration is not None and not isinstance(
        configuration, PhoneResidentModelReprovisioningConfiguration
    ):
        raise UnifiedScheduleError("phone resident-model reprovisioning configuration is invalid")
    if configuration is not None and controller._fixed_phone_residency is not None:
        raise UnifiedScheduleError("fixed phone residency cannot be re-provisioned")
    if controller._phone_reprovisioning == configuration:
        return
    controller._phone_reprovisioning = configuration
    controller._phone_reprovision_boundary_gate = None
    controller._model_placement_controller.carry_replacement_sources_on_update = (
        configuration is not None
    )
    controller._model_placement_controller.record_phone_layout_evaluation(0, {
        "reason": "PHONE_RESIDENT_MODEL_REPROVISIONING_CONFIGURED",
        "configuration": None if configuration is None else configuration.to_json(),
    })


def _desktop_device_ids(catalog) -> frozenset[str]:
    phones = {row.device_id for row in catalog.executors if row.phone_sessions}
    phones.update(getattr(catalog, "phone_power_profile_by_device", None) or ())
    profile = getattr(catalog, "placement_profile", None)
    for device_id, device in (getattr(profile, "devices", None) or {}).items():
        if getattr(device, "kind", None) == "phone":
            phones.add(device_id)
    return frozenset(
        row.device_id for row in catalog.executors if row.device_id not in phones
    )


def _desktop_load_transitions(plan, desktop_devices: frozenset[str]) -> tuple:
    return tuple(
        row for row in plan.transitions
        if row.source_state != row.target_state
        and (
            row.device_id in desktop_devices
            or desktop_devices.intersection(row.prepares_device_ids or ())
        )
    )


def _dispatched_at_us(ticket) -> int:
    receipt = getattr(ticket, "dispatch_receipt", None)
    return ticket.decision.start_us if receipt is None else receipt.observed_at_us


def _last_recorded_reprovision(controller) -> Mapping[str, object] | None:
    for event in reversed(controller._model_placement_controller.phone_layout_events()):
        recorded = event.get("desktop_reprovision")
        if event.get("kind") == "EVALUATED" and isinstance(recorded, Mapping):
            return recorded
    return None


def _opt_in(configuration, name: str) -> bool:
    """Whether an opt-in reprovisioning flag is set (absent or non-boolean counts as off)."""
    return getattr(configuration, name, False) is True


_QUEUED_DEMAND_LEARNING_REJECTIONS = frozenset({"MARGINAL_SYSTEM_COST_UNKNOWN"})


def queued_demand_learning_rejections(controller) -> frozenset[str]:
    """Route rejections a learning demand also tolerates under ``count_queued_demand``.

    The marginal system cost of a phone route is unknown while other work is
    protected on the desktop, i.e. while another model runs and this model's
    switch waits; without the knob nothing more is tolerated."""
    configuration = getattr(controller, "_phone_reprovisioning", None)
    if _opt_in(configuration, "count_queued_demand"):
        return _QUEUED_DEMAND_LEARNING_REJECTIONS
    return frozenset()


def _next_in_dispatch_order(controller, request_id: str) -> bool:
    """Whether a not-started request waits on no live predecessor (fail-closed without a view)."""

    view_of = getattr(controller._runtime_controller, "dispatch_order_view", None)
    if view_of is None:
        return False
    view = view_of()
    row = view.get(request_id)
    if row is None or row.get("state") not in _NOT_STARTED_STATES:
        return False
    return not any(
        view.get(predecessor, {}).get("state") not in {None, "FINISHING"}
        for predecessor in row.get("predecessor_request_ids", ())
    )


def _queued_next_commitments(controller, phone_capable: frozenset[str], desktop: frozenset[str]) -> list:
    """(start, request, artifact, end) of queued phone-capable desktop work next in dispatch order.

    Only under ``count_queued_demand``; ``end`` is None for a plan without a desktop load."""

    if not _opt_in(getattr(controller, "_phone_reprovisioning", None), "count_queued_demand"):
        return []
    rows = []
    for ticket in controller._runtime_controller.current_tickets():
        plan = ticket.execution_plan
        artifact = ticket.model.artifact_sha256
        if (
            ticket.dispatch_state not in _NOT_STARTED_STATES
            or plan is None
            or artifact not in phone_capable
            or not desktop.intersection(plan.device_ids)
            or not _next_in_dispatch_order(controller, ticket.request.request_id)
        ):
            continue
        loads = _desktop_load_transitions(plan, desktop)
        start = _dispatched_at_us(ticket)
        end = start + sum(row.latency_us for row in loads) if loads else None
        rows.append((start, ticket.request.request_id, artifact, end))
    return rows


def _desktop_commitment(
    controller,
    phone_capable: frozenset[str],
    snapshot: HeterogeneousRuntimeSnapshot | None,
) -> tuple[tuple[str, ...], str | None, str | None, int | None, int | None]:
    """Return (followed artifacts, source, leader request, load start, load end)."""

    desktop = _desktop_device_ids(controller._runtime_capabilities)
    loading = []
    executing = set()
    for ticket in controller._runtime_controller.current_tickets():
        plan = ticket.execution_plan
        artifact = ticket.model.artifact_sha256
        if ticket.dispatch_state != "ACQUIRED" or plan is None or artifact not in phone_capable:
            continue
        loads = _desktop_load_transitions(plan, desktop)
        if ticket.transition_status == "PENDING" and loads:
            start = _dispatched_at_us(ticket)
            end = start + sum(row.latency_us for row in loads)
            loading.append((start, ticket.request.request_id, artifact, end))
        elif desktop.intersection(plan.device_ids):
            executing.add(artifact)
    queued = _queued_next_commitments(controller, phone_capable, desktop)
    if loading:
        start, leader, _artifact, end = max(loading)
        followed = executing | {row[2] for row in loading} | {row[2] for row in queued}
        return tuple(sorted(followed)), "loading", leader, start, end
    if executing:
        followed = executing | {row[2] for row in queued}
        return tuple(sorted(followed)), "executing", None, None, None
    if queued:
        start, leader, _artifact, end = max(queued)
        return tuple(sorted({row[2] for row in queued})), "queued", leader, start, end
    if snapshot is not None:
        rows = tuple(
            row for row in snapshot.residency
            if row.device_id in desktop
            and row.artifact_sha256 in phone_capable
            and row.resident_bytes > 0
        )
        for state in ("hot", "warm"):
            resident = sorted({row.artifact_sha256 for row in rows if row.state == state})
            if resident:
                return tuple(resident), "resident", None, None, None
    recorded = _last_recorded_reprovision(controller)
    previous = () if recorded is None else recorded.get("followed_artifact_sha256s", ())
    retained = tuple(sorted(
        value for value in (previous if isinstance(previous, (list, tuple)) else ())
        if value in phone_capable
    ))
    if retained:
        return retained, "retained", None, None, None
    return (), None, None, None, None


def _attachment_in_use(binding: Mapping[str, object]) -> bool:
    """Mirror of the transition-blocker test on one acquired binding."""

    attachment = binding.get("helper_attachment")
    if not isinstance(attachment, Mapping):
        return False
    fraction = attachment.get("fraction_ppm") or 0
    leases = attachment.get("lease_tokens") or ()
    if (attachment.get("completed_phone_calls") or 0) <= 0 and fraction <= 0 and not leases:
        return False
    return not (
        attachment.get("fallback_outcome") is not None
        and fraction == 0
        and (binding.get("fraction_ppm") or 0) == 0
        and not leases
        and attachment.get("lease_reserved_until_us") is None
    )


def _in_use_session_ids(controller) -> tuple[str, ...]:
    """Sessions an acquired helper still uses, or that are mid-transition."""

    placement = controller._model_placement_controller
    in_use = {
        state.session_id for state in placement.phone_session_states()
        if state.active_helper_references or state.state not in _IDLE_SESSION_STATES
    }
    for ticket in controller._runtime_controller.current_tickets():
        if ticket.dispatch_state != "ACQUIRED":
            continue
        binding = placement.request_binding(ticket.request.request_id)
        if isinstance(binding, Mapping) and _attachment_in_use(binding):
            in_use.update(
                value for value in binding["helper_attachment"].get("allowed_session_ids", ())
                if type(value) is str
            )
    return tuple(sorted(in_use))


def _learned_phone_load_rate(
    controller, configuration: PhoneResidentModelReprovisioningConfiguration,
) -> tuple[int, int]:
    """Bytes per second over complete SESSION_LOADING -> SESSION_VERIFIED windows.

    Sessions of one layout generation load in one wall window; a retried load restarts
    its session's window."""

    loading: dict[int, dict[str, tuple[int, int]]] = {}
    verified: dict[int, dict[str, int]] = {}
    for event in controller._model_placement_controller.phone_layout_events():
        kind = event.get("kind")
        generation = event.get("layout_generation")
        session = event.get("session")
        at_us = event.get("observed_at_us")
        if (
            kind not in {"SESSION_LOADING", "SESSION_VERIFIED"}
            or type(generation) is not int
            or type(at_us) is not int
            or not isinstance(session, Mapping)
            or type(session.get("session_id")) is not str
        ):
            continue
        session_id = session["session_id"]
        if kind == "SESSION_VERIFIED":
            verified.setdefault(generation, {})[session_id] = at_us
            continue
        size = session.get("resident_bytes")
        if type(size) is int and size > 0:
            loading.setdefault(generation, {})[session_id] = (at_us, size)
            verified.get(generation, {}).pop(session_id, None)
    total_bytes = total_us = samples = 0
    for generation, sessions in sorted(loading.items()):
        done = verified.get(generation, {})
        if set(done) != set(sessions) or any(done[key] <= at for key, (at, _) in sessions.items()):
            continue
        total_bytes += sum(size for _, size in sessions.values())
        total_us += max(done.values()) - min(at for at, _ in sessions.values())
        samples += 1
    if samples < configuration.minimum_learned_samples:
        return configuration.load_bytes_per_second, 0
    return max(1, total_bytes * 1_000_000 // total_us), samples


def _load_latency_us(resident_bytes: int, bytes_per_second: int) -> int:
    return (resident_bytes * 1_000_000 + bytes_per_second - 1) // bytes_per_second


def _phone_reprovision_demand(
    controller,
    demand: _PhoneQueueDemand,
    discovery: _PhoneDemandDiscovery,
    snapshot: HeterogeneousRuntimeSnapshot | None,
    observed_at_us: int,
) -> tuple[_PhoneQueueDemand, _PhoneDemandDiscovery]:
    """Restrict the layout demand to the followed models; unchanged when the knob is off."""

    configuration = getattr(controller, "_phone_reprovisioning", None)
    if configuration is None or not discovery.demand_rows:
        return demand, discovery
    placement = controller._model_placement_controller
    demanded = frozenset(row.manifest.artifact_sha256 for row in discovery.demand_rows)
    phone_capable = frozenset(
        demanded
        | {
            shard.artifact_sha256
            for state in (placement.ready_phone_layout(), placement.planning_phone_layout())
            if state is not None
            for shard in state.layout.shards
        }
    )
    followed, source, leader, start, end = _desktop_commitment(controller, phone_capable, snapshot)
    work = {
        key: value for key, value in sorted(demand.queued_work_by_artifact.items())
        if key in demanded and value > 0
    }
    live = tuple(key for key in followed if key in work) if followed else tuple(work)
    mode = MODE_HOLD if not live else MODE_FOLLOW if followed and len(live) == 1 else MODE_PROPORTIONAL
    rate, samples = _learned_phone_load_rate(controller, configuration)
    context = _PhoneReprovisionContext(
        configuration=configuration,
        mode=mode,
        followed_artifacts=followed,
        split_artifacts=live,
        commitment_source=source,
        leader_request_id=leader,
        load_window_start_us=start,
        load_window_end_us=end,
        in_use_session_ids=_in_use_session_ids(controller),
        work_by_artifact=MappingProxyType(work),
        load_bytes_per_second=rate,
        learned_load_samples=samples,
    )
    if mode != MODE_HOLD:
        discovery = replace(discovery, demand_rows=tuple(
            row for row in discovery.demand_rows if row.manifest.artifact_sha256 in live
        ))
        demand = replace(demand, queued_work_by_artifact={key: work[key] for key in live})
    return replace(demand, reprovision=context), discovery


def _proportional_session_counts(
    work_by_artifact: Mapping[str, int], session_count: int,
) -> dict[str, int]:
    rows = tuple(
        (artifact, work) for artifact, work in sorted(work_by_artifact.items()) if work > 0
    )
    total = sum(work for _, work in rows)
    if not rows or session_count <= 0:
        return {}
    counts = {artifact: session_count * work // total for artifact, work in rows}
    by_remainder = sorted(rows, key=lambda row: (-(session_count * row[1] % total), row[0]))
    for artifact, _ in by_remainder[:session_count - sum(counts.values())]:
        counts[artifact] += 1
    return {key: value for key, value in counts.items() if value > 0}


def _session_counts(layout: PhoneFfnResidencyLayout) -> dict[str, int]:
    counts: dict[str, int] = {}
    for shard in layout.shards:
        counts[shard.artifact_sha256] = counts.get(shard.artifact_sha256, 0) + 1
    return counts


def _layers_by_artifact(layout: PhoneFfnResidencyLayout | None) -> dict[str, int]:
    layers: dict[str, int] = {}
    for shard in () if layout is None else layout.shards:
        layers[shard.artifact_sha256] = layers.get(shard.artifact_sha256, 0) + shard.layer_mask.bit_count()
    return dict(sorted(layers.items()))


def _swap_latency_us(layout: PhoneFfnResidencyLayout | None, bytes_per_second: int) -> dict[str, int]:
    if layout is None:
        return {}
    by_session = {row.session_id: row for row in layout.shards}
    return {
        session_id: _load_latency_us(by_session[session_id].resident_bytes, bytes_per_second)
        for session_id in layout.changed_session_ids
        if session_id in by_session
    }


def _reprovision_candidate_choice(
    controller,
    context: _PhoneReprovisionContext,
    layouts: Sequence[PhoneFfnResidencyLayout],
    sessions: Sequence[RuntimePhoneSessionCapability],
    memory: _PhoneMemoryBudget,
    transition_latencies: Mapping[str, int],
    observed_at_us: int,
) -> _PhoneCandidateChoice | None:
    """Follow the desktop one session at a time; None defers to the default selector."""

    current = memory.current
    ready_ids = {row.session_id for row in sessions if row.ready}
    current_ids = frozenset(() if current is None else (row.session_id for row in current.shards))
    if current_ids - ready_ids:
        return None  # session degradation stays with the default (forced) selector
    rows = tuple(layouts)
    in_use = frozenset(context.in_use_session_ids)
    split = frozenset(context.split_artifacts)
    targets = (
        {context.split_artifacts[0]: len(ready_ids)} if context.mode == MODE_FOLLOW
        else _proportional_session_counts(
            {key: context.work_by_artifact[key] for key in context.split_artifacts},
            len(ready_ids),
        )
    )

    def benefit(layout) -> int:
        return sum(
            value for key, value in layout.queue_benefit_by_artifact.items() if key in split
        )

    def rank(layout, *, resident=False) -> tuple:
        counts = {key: value for key, value in _session_counts(layout).items() if key in split}
        distance = sum(abs(counts.get(key, 0) - targets.get(key, 0)) for key in set(counts) | set(targets))
        head = (-benefit(layout),) if context.mode == MODE_FOLLOW else (distance, -benefit(layout))
        if resident:
            return (*head, 0, 0, layout.geometry_sha256)
        return (*head, len(layout.changed_session_ids), layout.transition_cost, layout.geometry_sha256)

    def keeps_current(layout) -> bool:
        return bool(layout.changed_session_ids) and current_ids <= {row.session_id for row in layout.shards}

    def admissible(layout) -> bool:
        return keeps_current(layout) and not in_use.intersection(layout.changed_session_ids)

    def choice(selected, reason, *, target=None, confirmed=False, impacts=None, **extra):
        target_latency = _swap_latency_us(target, context.load_bytes_per_second)
        stage_latency = (
            {} if selected is None or selected is current
            else _swap_latency_us(selected, context.load_bytes_per_second)
        )
        window_end = context.load_window_end_us
        payload = {
            **context.to_json(),
            **extra,
            "fits_load_window": (
                None if window_end is None or target is None
                else sum(target_latency.values()) <= max(0, window_end - observed_at_us)
            ),
            "reason": reason,
            "selected_layers_by_artifact": _layers_by_artifact(selected),
            "stage_swap_latency_us": sum(stage_latency.values()),
            "target_geometry_sha256": None if target is None else target.geometry_sha256,
            "target_layers_by_artifact": _layers_by_artifact(target),
            "target_session_counts_by_artifact": dict(sorted(targets.items())),
            "target_swap_latency_us": sum(target_latency.values()),
        }
        return _PhoneCandidateChoice(
            layouts=rows,
            selected=selected,
            reason=reason,
            marginal_gains=(),
            transition_latencies={**dict(transition_latencies), **target_latency, **stage_latency},
            force=False,
            switching_margin_uj=0,
            minimum_residency_us=controller._model_placement_controller.policy.phone_minimum_residency_us,
            request_impacts_by_geometry={} if impacts is None else impacts,
            confirmed=confirmed,
            reprovision=MappingProxyType(payload),
        )

    if context.mode == MODE_HOLD:
        return choice(current, REASON_HOLD)
    if not targets:
        return None
    ranked = sorted((row for row in rows if keeps_current(row)), key=rank)
    best_any = ranked[0] if ranked else None
    target = next((row for row in ranked if admissible(row)), None)
    blocked = (
        () if best_any is None or best_any is target
        else tuple(sorted(in_use.intersection(best_any.changed_session_ids)))
    )
    if current is None:
        current_rank, stage = None, target
    else:
        current_row = next((row for row in rows if row.geometry_sha256 == current.geometry_sha256), None)
        current_rank = rank(current if current_row is None else current_row, resident=True)
        stage = next((
            row for row in ranked
            if len(row.changed_session_ids) == 1 and admissible(row) and rank(row) < current_rank
        ), None)
    if stage is None or current_rank is not None and rank(target) >= current_rank:
        if blocked and (current_rank is None or rank(best_any) < current_rank):
            return choice(current, REASON_DEFERRED_IN_USE, target=best_any, blocked_session_ids=list(blocked))
        return choice(current, REASON_RETAINED)
    extra = {"blocked_session_ids": list(blocked)} if blocked else {}
    stage_latency = sum(_swap_latency_us(stage, context.load_bytes_per_second).values())
    found = controller._phone_layout_request_impacts(stage, observed_at_us, stage_latency)
    impacts = {stage.geometry_sha256: found} if found else {}
    if any(not row.verification_feasible for row in found or ()):
        return choice(current, REASON_DEFERRED_REVALIDATION, target=target, impacts=impacts, **extra)
    return choice(
        stage,
        REASON_REPROVISION if context.mode == MODE_FOLLOW else REASON_PROPORTIONAL,
        target=target, confirmed=context.confirmed, impacts=impacts, **extra,
    )


def _defer_preparation_until_release(controller, request_id, state, blockers, observed_at_us):
    """Wait for helpers on a re-provisioned session instead of draining them; None otherwise."""

    if controller._phone_reprovisioning is None or state.selection_reason not in _CHANGE_REASONS:
        return None
    payload = {
        "blocking_request_ids": sorted(blockers),
        "phone_layout_generation": state.generation,
        "reason": "WAITING_FOR_HELPER_RELEASE",
    }
    placement = controller._model_placement_controller
    if not any(
        row.get("kind") == "PREPARATION_DEFERRED"
        and row.get("reason") == payload["reason"]
        and row.get("phone_layout_generation") == state.generation
        for row in placement.request_helper_events(request_id)
    ):
        placement.record_request_helper_event(request_id, "PREPARATION_DEFERRED", observed_at_us, payload)
    return MappingProxyType({**payload, "blocking_request_ids": tuple(payload["blocking_request_ids"]),
                             "status": "DEFERRED"})


@dataclass
class _BoundaryGate:
    """Decode-boundary re-evaluation state; counters since the last record and in total."""

    state_sha256: str | None = None
    evaluated_at_us: int | None = None
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(sorted(_COUNTER_FIELDS), 0))

    def count(self, name: str) -> None:
        self.counts[name] += 1
        self.counts[name + "_total"] += 1


def _boundary_gate(controller) -> _BoundaryGate:
    gate = getattr(controller, "_phone_reprovision_boundary_gate", None)
    if gate is None:
        gate = _BoundaryGate()
        controller._phone_reprovision_boundary_gate = gate
    return gate


def _boundary_state_sha256(controller, compiler, work_by_artifact: Mapping[str, int],
                           uncovered_artifacts) -> str:
    """What a boundary re-evaluation reacts to: requests with their dispatch and desktop-load
    state, layout generations, session states and use, arrived-work buckets, route evidence."""

    placement = controller._model_placement_controller
    ready = placement.ready_phone_layout()
    target = placement.target_phone_layout()
    evidence = {}
    for artifact_sha256 in sorted(work_by_artifact):
        status = compiler.phone_residency_evidence_status(artifact_sha256)
        evidence[artifact_sha256] = [status.get(key) for key in (
            "reason", "source_route_id", "normalized_benefit_uj")]
    return canonical_sha256({
        "evidence": evidence,
        "in_use_session_ids": list(_in_use_session_ids(controller)),
        "layout_generations": [None if row is None else row.generation for row in (ready, target)],
        "schema": "phone-reprovision-boundary-state-v1",
        "sessions": [
            [row.session_id, row.state, row.session_generation, row.resident_artifact_sha256,
             len(row.active_helper_references)]
            for row in placement.phone_session_states()
        ],
        "tickets": sorted(
            [row.request.request_id, row.dispatch_state, row.transition_status, row.model.artifact_sha256]
            for row in controller._runtime_controller.current_tickets()
            if row.dispatch_state not in _TERMINAL_STATES
        ),
        "uncovered_artifact_sha256s": sorted(uncovered_artifacts),
        "work_buckets": {key: material_count_bucket(value) for key, value in sorted(work_by_artifact.items())},
    })


def _boundary_reevaluation_due(controller, compiler, work_by_artifact: Mapping[str, int],
                               uncovered_artifacts, observed_at_us: int) -> bool:
    """Whether a decode boundary re-evaluates the layout; always with the knob off.

    The first boundary after a state change evaluates; unchanged boundaries are counted
    until ``boundary_reevaluation_interval_us`` has passed since the last evaluation."""

    configuration = getattr(controller, "_phone_reprovisioning", None)
    if configuration is None:
        return True
    interval_us = getattr(configuration, "boundary_reevaluation_interval_us", _DEFAULT_BOUNDARY_INTERVAL_US)
    gate = _boundary_gate(controller)
    state = _boundary_state_sha256(controller, compiler, work_by_artifact, uncovered_artifacts)
    if (
        state == gate.state_sha256
        and gate.evaluated_at_us is not None
        and observed_at_us - gate.evaluated_at_us < interval_us
    ):
        gate.count("boundary_evaluations_skipped")
        return False
    gate.state_sha256, gate.evaluated_at_us = state, observed_at_us
    return True


def _unchanged_decision_key(event: Mapping[str, object]) -> str | None:
    recorded = event.get("desktop_reprovision")
    if not isinstance(recorded, Mapping) or recorded.get("reason") not in _UNCHANGED_REASONS:
        return None
    work = recorded.get("arrived_work_by_artifact")
    return canonical_sha256({
        "decision": {
            key: value for key, value in sorted(recorded.items())
            if key != "arrived_work_by_artifact" and key not in _COUNTER_FIELDS
        },
        "layout": [event.get(key) for key in (
            "reason", "current_geometry_sha256", "planning_geometry_sha256", "selected_geometry_sha256",
            "selected_layout_generation", "selected_layout_state", "selection_confirmed",
            "phone_memory_selected_limit_bytes",
        )],
        "work_buckets": {
            key: material_count_bucket(value) for key, value in sorted(
                (work if isinstance(work, Mapping) else {}).items())
        },
    })


def _coalesce_unchanged_decision(
    controller, event: Mapping[str, object],
) -> Mapping[str, object] | None:
    """The EVALUATED event to record, or None for a RETAINED/HOLD decision equal to the last
    record; the skipped boundaries and coalesced decisions are counted into the next record."""

    gate = _boundary_gate(controller)
    key = _unchanged_decision_key(event)
    if key is not None:
        previous = next((
            row for row in reversed(controller._model_placement_controller.phone_layout_events())
            if row.get("kind") == "EVALUATED"
        ), None)
        if previous is not None and _unchanged_decision_key(previous) == key:
            gate.count("unchanged_decisions_coalesced")
            return None
    recorded = {**dict(event["desktop_reprovision"]), **gate.counts}
    for name in _COUNTER_FIELDS:
        if not name.endswith("_total"):
            gate.counts[name] = 0
    return MappingProxyType({**event, "desktop_reprovision": recorded})


def _reevaluate_phone_layout(controller, ticket: RuntimeRequestTicket, at_us: int, failure: str,
                             event_request_id: str | None = None) -> None:
    try:
        with controller._transaction(convert=False):
            controller._update_phone_residency_portfolio(
                ticket.request,
                controller.runtime_model_manifest(ticket.model.model_id),
                at_us,
                None,
            )
    except _RECOVERABLE_ERRORS as exc:
        controller._model_placement_controller.record_request_helper_event(
            event_request_id or ticket.request.request_id, failure, at_us, {"reason": str(exc)},
        )


def _reevaluate_phone_layout_for_desktop_load(controller, ticket: RuntimeRequestTicket) -> None:
    """Start the phone swap when a dispatched request begins a desktop model load."""

    catalog = controller._runtime_capabilities
    plan = ticket.execution_plan
    if (
        catalog is None
        or plan is None
        or ticket.dispatch_state != "ACQUIRED"
        or ticket.transition_status != "PENDING"
        or not _desktop_load_transitions(plan, _desktop_device_ids(catalog))
    ):
        return
    _reevaluate_phone_layout(
        controller, ticket, _dispatched_at_us(ticket), "DESKTOP_LOAD_LAYOUT_REEVALUATION_FAILED",
    )


def _decided_desktop_load(controller, *, exclude_request_id: str | None = None):
    """The first (by request id) non-terminal ticket whose plan carries a not yet completed
    desktop load of a phone-capable model, or None."""

    catalog = controller._runtime_capabilities
    if catalog is None:
        return None
    desktop = _desktop_device_ids(catalog)
    rows = (
        row for row in controller._runtime_controller.current_tickets()
        if row.dispatch_state not in _TERMINAL_STATES
        and row.request.request_id != exclude_request_id
        and row.execution_plan is not None
        and row.transition_status == "PENDING"
        and _desktop_load_transitions(row.execution_plan, desktop)
    )
    return min(rows, key=lambda row: row.request.request_id, default=None)


def _reevaluate_phone_layout_for_decided_transition(
    controller, ticket: RuntimeRequestTicket, observed_at_us: int,
) -> None:
    """Start the phone swap when a plan carrying a desktop model load is committed.

    Opt-in (``early_on_transition``): the residency change is decided at the commit,
    before the loading request is dispatched, so the swap overlaps the wait for the
    server and the desktop load itself."""

    catalog = controller._runtime_capabilities
    plan = ticket.execution_plan
    if (
        not _opt_in(getattr(controller, "_phone_reprovisioning", None), "early_on_transition")
        or catalog is None
        or plan is None
        or ticket.dispatch_state != "QUEUED"
        or ticket.transition_status != "PENDING"
        or not _desktop_load_transitions(plan, _desktop_device_ids(catalog))
    ):
        return
    _reevaluate_phone_layout(
        controller, ticket, observed_at_us, "DECIDED_TRANSITION_LAYOUT_REEVALUATION_FAILED",
    )


def _reevaluate_phone_layout_after_release(
    controller, ticket: RuntimeRequestTicket, observed_at_us: int,
) -> None:
    """Continue a re-provisioning that waited for this request's sessions.

    Under ``early_on_transition`` a release also re-evaluates while another model's
    desktop load is decided or pending, so the swap follows the released sessions at
    once instead of waiting for the next dispatch or decode boundary."""

    decided = (
        _decided_desktop_load(controller, exclude_request_id=ticket.request.request_id)
        if _opt_in(getattr(controller, "_phone_reprovisioning", None), "early_on_transition")
        else None
    )
    recorded = _last_recorded_reprovision(controller)
    if decided is None and (recorded is None or not recorded.get("blocked_session_ids")):
        return
    other = decided if decided is not None else min(
        (
            row for row in controller._runtime_controller.current_tickets()
            if row.dispatch_state not in _TERMINAL_STATES
            and row.request.request_id != ticket.request.request_id
        ),
        key=lambda row: row.request.request_id,
        default=None,
    )
    if other is not None:
        _reevaluate_phone_layout(
            controller, other, observed_at_us, "RELEASE_LAYOUT_REEVALUATION_FAILED",
            event_request_id=ticket.request.request_id,
        )
