"""Causal model-level placement refresh control."""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .phone_shards import (
    PhoneFfnResidencyLayout,
    PhoneFfnSessionIdentity,
    PhoneFfnShardPlacement,
    progressive_ffn_residency_layouts,
)
from .types import canonical_sha256


from .model_placement_contracts.common import (
    MODEL_PLACEMENT_ACTIONS as MODEL_PLACEMENT_ACTIONS,
    PHONE_RESIDENCY_LAYOUT_STATES as PHONE_RESIDENCY_LAYOUT_STATES,
    PHONE_SESSION_RESIDENCY_STATES as PHONE_SESSION_RESIDENCY_STATES,
    REQUEST_HELPER_REBIND_STATES as REQUEST_HELPER_REBIND_STATES,
    ModelPlacementControllerError as ModelPlacementControllerError,
    _text as _text,
    _integer as _integer,
    _sha256 as _sha256,
    _optional_sha256 as _optional_sha256,
    _identities as _identities,
    _material_bucket as _material_bucket,
)
from .model_placement_contracts.demand import (
    ModelPlacementPolicy as ModelPlacementPolicy,
    ModelDemandSnapshot as ModelDemandSnapshot,
    ModelPlacementTrigger as ModelPlacementTrigger,
    ModelPlacementAction as ModelPlacementAction,
)
from .model_placement_contracts.layout import (
    PhoneSessionMarginalGain as PhoneSessionMarginalGain,
    PhoneLayoutRequestImpact as PhoneLayoutRequestImpact,
    PhoneSessionResidencyState as PhoneSessionResidencyState,
    ModelPhoneResidencyLayout as ModelPhoneResidencyLayout,
)
from .model_placement_contracts.requests import (
    RequestBasePlacementBinding as RequestBasePlacementBinding,
    RequestHelperEnvelopeBinding as RequestHelperEnvelopeBinding,
    RequestHelperAttachment as RequestHelperAttachment,
    RequestHelperRebind as RequestHelperRebind,
    RequestPlacementBinding as RequestPlacementBinding,
)

from .model_placement_ops.common import (
    _ControllerCheckpoint as _ControllerCheckpoint,
    _INFORMATIONAL_NOTIFICATIONS as _INFORMATIONAL_NOTIFICATIONS,
)
from .model_placement_ops import attachment as _attachment
from .model_placement_ops import economics as _economics
from .model_placement_ops import epochs as _epochs
from .model_placement_ops import planning as _planning
from .model_placement_ops import rebind as _rebind
from .model_placement_ops import requests as _requests
from .model_placement_ops import sessions as _sessions
from .model_placement_ops import transitions as _transitions

__all__ = [
    'MODEL_PLACEMENT_ACTIONS',
    'ModelDemandSnapshot',
    'ModelPhoneResidencyLayout',
    'ModelPlacementAction',
    'ModelPlacementController',
    'ModelPlacementControllerError',
    'ModelPlacementPolicy',
    'ModelPlacementTrigger',
    'PHONE_RESIDENCY_LAYOUT_STATES',
    'PHONE_SESSION_RESIDENCY_STATES',
    'PhoneFfnResidencyLayout',
    'PhoneFfnSessionIdentity',
    'PhoneFfnShardPlacement',
    'PhoneLayoutRequestImpact',
    'PhoneSessionMarginalGain',
    'PhoneSessionResidencyState',
    'REQUEST_HELPER_REBIND_STATES',
    'RequestBasePlacementBinding',
    'RequestHelperAttachment',
    'RequestHelperEnvelopeBinding',
    'RequestHelperRebind',
    'RequestPlacementBinding',
    '_ControllerCheckpoint',
    '_INFORMATIONAL_NOTIFICATIONS',
    '_attachment',
    '_economics',
    '_epochs',
    '_identities',
    '_integer',
    '_material_bucket',
    '_optional_sha256',
    '_planning',
    '_rebind',
    '_requests',
    '_sessions',
    '_sha256',
    '_text',
    '_transitions',
    'canonical_sha256',
    'progressive_ffn_residency_layouts',
]


class ModelPlacementController:
    """Decide when existing placement epochs need fresh route costing."""

    def __init__(
        self, policy: ModelPlacementPolicy | None = None
    ) -> None:
        self.policy = policy or ModelPlacementPolicy()
        self._snapshots: dict[str, ModelDemandSnapshot] = {}
        self._pending_reasons: dict[str, set[str]] = {}
        self._background_inflight: set[str] = set()
        self._last_refresh_us: dict[str, int] = {}
        self._request_bindings: dict[str, RequestPlacementBinding] = {}
        self._acquired_request_ids: set[str] = set()
        self._request_decode_progress: dict[str, tuple[int, int]] = {}
        self._request_helper_events: list[Mapping[str, object]] = []
        self._request_helper_rebinds: dict[str, RequestHelperRebind] = {}
        self._phone_layout_generation = 0
        self._phone_layouts: dict[int, ModelPhoneResidencyLayout] = {}
        self._phone_session_states: dict[
            str, PhoneSessionResidencyState
        ] = {}
        self._phone_session_generation_by_id: dict[str, int] = {}
        self._phone_session_replacement_sources: dict[
            int, dict[str, PhoneSessionResidencyState]
        ] = {}
        self._ready_phone_layout_generation: int | None = None
        self._target_phone_layout_generation: int | None = None
        self._phone_preload_layouts: tuple[PhoneFfnResidencyLayout, ...] = ()
        self._pending_phone_layout_geometry_sha256: str | None = None
        self._pending_phone_layout_snapshot_sha256: str | None = None
        self._pending_phone_layout_snapshot_count = 0
        self._pending_phone_layout_sampled_at_us: int | None = None
        self._phone_layout_events: list[Mapping[str, object]] = []
        self._phone_layout_event_clock: Callable[[], int] | None = None
        # Opt-in (phone_resident_model_reprovisioning): a same-geometry
        # PROPOSAL_UPDATED keeps the stamped replacement sources.  Off, the
        # recorded proposal JSON of static-layout runs is unchanged.
        self.carry_replacement_sources_on_update = False
        self._events: list[Mapping[str, object]] = []
        self._counters: dict[str, int] = {}

    def checkpoint(self) -> object:
        return _ControllerCheckpoint(
            snapshots=tuple(sorted(self._snapshots.items())),
            pending_reasons=tuple(sorted(
                (artifact, tuple(sorted(reasons)))
                for artifact, reasons in self._pending_reasons.items()
            )),
            background_inflight=tuple(sorted(self._background_inflight)),
            last_refresh_us=tuple(sorted(self._last_refresh_us.items())),
            request_bindings=tuple(sorted(self._request_bindings.items())),
            acquired_request_ids=tuple(sorted(self._acquired_request_ids)),
            request_decode_progress=tuple(sorted(
                self._request_decode_progress.items()
            )),
            request_helper_events=tuple(self._request_helper_events),
            request_helper_rebinds=tuple(sorted(
                self._request_helper_rebinds.items()
            )),
            phone_layout_generation=self._phone_layout_generation,
            phone_layouts=tuple(sorted(self._phone_layouts.items())),
            phone_session_states=tuple(sorted(
                self._phone_session_states.items()
            )),
            phone_session_generation_by_id=tuple(sorted(
                self._phone_session_generation_by_id.items()
            )),
            phone_session_replacement_sources=tuple(sorted(
                (
                    generation,
                    tuple(sorted(states.items())),
                )
                for generation, states in (
                    self._phone_session_replacement_sources.items()
                )
            )),
            ready_phone_layout_generation=(
                self._ready_phone_layout_generation
            ),
            target_phone_layout_generation=(
                self._target_phone_layout_generation
            ),
            phone_preload_layouts=self._phone_preload_layouts,
            pending_phone_layout_geometry_sha256=(
                self._pending_phone_layout_geometry_sha256
            ),
            pending_phone_layout_snapshot_sha256=(
                self._pending_phone_layout_snapshot_sha256
            ),
            pending_phone_layout_snapshot_count=(
                self._pending_phone_layout_snapshot_count
            ),
            pending_phone_layout_sampled_at_us=(
                self._pending_phone_layout_sampled_at_us
            ),
            phone_layout_events=tuple(self._phone_layout_events),
            events=tuple(self._events),
            counters=tuple(sorted(self._counters.items())),
        )

    def restore(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _ControllerCheckpoint):
            raise ModelPlacementControllerError(
                "model placement checkpoint is invalid"
            )
        self._snapshots = dict(checkpoint.snapshots)
        self._pending_reasons = {
            artifact: set(reasons)
            for artifact, reasons in checkpoint.pending_reasons
        }
        self._background_inflight = set(checkpoint.background_inflight)
        self._last_refresh_us = dict(checkpoint.last_refresh_us)
        self._request_bindings = dict(checkpoint.request_bindings)
        self._acquired_request_ids = set(
            checkpoint.acquired_request_ids
        )
        self._request_decode_progress = dict(
            checkpoint.request_decode_progress
        )
        self._request_helper_events = list(
            checkpoint.request_helper_events
        )
        self._request_helper_rebinds = dict(
            checkpoint.request_helper_rebinds
        )
        self._phone_layout_generation = checkpoint.phone_layout_generation
        self._phone_layouts = dict(checkpoint.phone_layouts)
        self._phone_session_states = dict(
            checkpoint.phone_session_states
        )
        self._phone_session_generation_by_id = dict(
            checkpoint.phone_session_generation_by_id
        )
        self._phone_session_replacement_sources = {
            generation: dict(states)
            for generation, states in (
                checkpoint.phone_session_replacement_sources
            )
        }
        self._ready_phone_layout_generation = (
            checkpoint.ready_phone_layout_generation
        )
        self._target_phone_layout_generation = (
            checkpoint.target_phone_layout_generation
        )
        self._phone_preload_layouts = checkpoint.phone_preload_layouts
        self._pending_phone_layout_geometry_sha256 = (
            checkpoint.pending_phone_layout_geometry_sha256
        )
        self._pending_phone_layout_snapshot_sha256 = (
            checkpoint.pending_phone_layout_snapshot_sha256
        )
        self._pending_phone_layout_snapshot_count = (
            checkpoint.pending_phone_layout_snapshot_count
        )
        self._pending_phone_layout_sampled_at_us = (
            checkpoint.pending_phone_layout_sampled_at_us
        )
        self._phone_layout_events = list(checkpoint.phone_layout_events)
        self._events = list(checkpoint.events)
        self._counters = dict(checkpoint.counters)

    # Monotonic count of helper-relevant state changes (layout events,
    # request helper events, acquisitions, pending-candidate updates). The
    # physical adapter's per-request watcher sleeps until this moves or a
    # bounded fallback interval elapses, instead of re-planning every 50 ms.
    _helper_state_generation: int = 0

    def helper_state_generation(self) -> int:
        return self._helper_state_generation

    def _note_helper_state_change(self) -> None:
        self._helper_state_generation += 1

    def set_phone_layout_event_clock(
        self, clock: Callable[[], int] | None
    ) -> None:
        if clock is not None and not callable(clock):
            raise ModelPlacementControllerError(
                "phone layout event clock is invalid"
            )
        self._phone_layout_event_clock = clock

    def _record_phone_layout_event(
        self,
        kind: str,
        observed_at_us: int,
        values: Mapping[str, object],
    ) -> None:
        kind = _text("phone layout event kind", kind)
        _integer("phone layout event time", observed_at_us)
        body = {
            "event_index": len(self._phone_layout_events),
            "kind": kind,
            "observed_at_us": observed_at_us,
            "schema": "research-scheduler-phone-layout-event-v1",
            **dict(values),
        }
        if self._phone_layout_event_clock is not None:
            published_at_us = self._phone_layout_event_clock()
            _integer("phone layout publication time", published_at_us)
            body["published_at_us"] = published_at_us
        event = MappingProxyType({
            **body,
            "event_sha256": canonical_sha256(body),
        })
        self._phone_layout_events.append(event)
        self._note_helper_state_change()
        if len(self._phone_layout_events) > self.policy.maximum_events:
            del self._phone_layout_events[:(
                len(self._phone_layout_events)
                - self.policy.maximum_events
            )]

    def record_phone_layout_evaluation(
        self,
        observed_at_us: int,
        values: Mapping[str, object],
    ) -> None:
        if not isinstance(values, Mapping):
            raise ModelPlacementControllerError(
                "phone layout evaluation is invalid"
            )
        self._record_phone_layout_event(
            "EVALUATED", observed_at_us, values
        )

    @staticmethod
    def _session_shard_matches(
        state: PhoneSessionResidencyState,
        shard: PhoneFfnShardPlacement,
    ) -> bool:
        return _sessions._session_shard_matches(state, shard)

    def _stamp_phone_layout(
        self, layout: PhoneFfnResidencyLayout
    ) -> PhoneFfnResidencyLayout:
        return _sessions._stamp_phone_layout(self, layout)

    @staticmethod
    def _session_state_from_shard(
        shard: PhoneFfnShardPlacement,
        identity: PhoneFfnSessionIdentity,
        *,
        state: str,
        minimum_resident_until_us: int,
        replacement_cost_uj: int,
        active_helper_references: Sequence[str] = (),
    ) -> PhoneSessionResidencyState:
        return _sessions._session_state_from_shard(
            shard,
            identity,
            state=state,
            minimum_resident_until_us=minimum_resident_until_us,
            replacement_cost_uj=replacement_cost_uj,
            active_helper_references=active_helper_references,
        )

    def _sync_phone_session_references(self) -> None:
        return _sessions._sync_phone_session_references(self)

    def phone_session_state(
        self, session_id: str
    ) -> PhoneSessionResidencyState | None:
        return _sessions.phone_session_state(self, session_id)

    def phone_session_states(
        self,
    ) -> tuple[PhoneSessionResidencyState, ...]:
        return _sessions.phone_session_states(self)

    def register_empty_phone_sessions(
        self,
        session_endpoints: Mapping[str, str],
        *,
        observed_at_us: int,
    ) -> tuple[PhoneSessionResidencyState, ...]:
        """Register discovered residency arenas before their first load."""
        return _sessions.register_empty_phone_sessions(
            self,
            session_endpoints,
            observed_at_us=observed_at_us,
        )

    def _normalize_helper_envelope(
        self, envelope: RequestHelperEnvelopeBinding
    ) -> RequestHelperEnvelopeBinding:
        return _sessions._normalize_helper_envelope(self, envelope)

    def _session_identities_are_ready(
        self, identities: Sequence[PhoneFfnSessionIdentity]
    ) -> bool:
        return _sessions._session_identities_are_ready(self, identities)

    def phone_layout_sessions_are_usable(
        self,
        generation: int,
        geometry_sha256: str,
        session_ids: Sequence[str],
    ) -> bool:
        """Return whether one audited layout subset is still exactly ready."""
        return _sessions.phone_layout_sessions_are_usable(
            self,
            generation,
            geometry_sha256,
            session_ids,
        )

    def select_phone_layout_candidate(
        self,
        layouts: Sequence[PhoneFfnResidencyLayout],
        *,
        current_layout: PhoneFfnResidencyLayout | None,
        minimum_energy_saving_ppm: int,
        transition_latency_us_by_session: Mapping[str, int] | None = None,
        request_impacts_by_geometry: Mapping[str, tuple[PhoneLayoutRequestImpact, ...]] | None = None,
        force: bool = False,
    ) -> tuple[
        PhoneFfnResidencyLayout | None,
        str,
        tuple[PhoneSessionMarginalGain, ...],
    ]:
        """Select a cold layout or one profitable session replacement."""
        return _economics.select_phone_layout_candidate(
            self,
            layouts,
            current_layout=current_layout,
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            transition_latency_us_by_session=transition_latency_us_by_session,
            request_impacts_by_geometry=request_impacts_by_geometry,
            force=force,
        )

    def confirm_phone_layout_candidate(
        self,
        geometry_sha256: str,
        snapshot_sha256: str,
        *,
        observed_at_us: int,
        force: bool = False,
        observation_sha256: str | None = None,
        sampled_at_us: int | None = None,
    ) -> tuple[bool, int]:
        """Require stable live pressure before replacing a ready layout."""
        return _economics.confirm_phone_layout_candidate(
            self,
            geometry_sha256,
            snapshot_sha256,
            observed_at_us=observed_at_us,
            force=force,
            observation_sha256=observation_sha256,
            sampled_at_us=sampled_at_us,
        )

    def pending_phone_layout_candidate(
        self,
    ) -> Mapping[str, object] | None:
        """Return the candidate waiting for another causal observation."""
        return _economics.pending_phone_layout_candidate(self)

    def phone_layout_transition_blockers(
        self, target_generation: int
    ) -> tuple[str, ...]:
        """Return admitted requests bound to the layout being replaced."""
        return _economics.phone_layout_transition_blockers(self, target_generation)

    def request_helper_rebind(
        self,
        request_id: str,
        target_generation: int,
        *,
        observed_at_us: int,
    ) -> RequestHelperRebind:
        """Request an acknowledged helper quiescence for one layout change."""
        return _rebind.request_helper_rebind(
            self,
            request_id,
            target_generation,
            observed_at_us=observed_at_us,
        )

    def bind_request_helper_rebind_drain_policy(
        self,
        request_id: str,
        target_generation: int,
        policy_sha256: str,
        *,
        observed_at_us: int,
    ) -> RequestHelperRebind:
        """Bind one runtime mask control to an active session rebind."""
        return _rebind.bind_request_helper_rebind_drain_policy(
            self,
            request_id,
            target_generation,
            policy_sha256,
            observed_at_us=observed_at_us,
        )

    def mark_request_helper_rebind_quiesced(
        self,
        request_id: str,
        target_generation: int,
        *,
        observed_at_us: int,
        allowed_session_ids: Sequence[str] | None = None,
        drain_policy_sha256: str | None = None,
    ) -> RequestHelperRebind:
        """Record an acknowledged session mask at a decode boundary."""
        return _rebind.mark_request_helper_rebind_quiesced(
            self,
            request_id,
            target_generation,
            observed_at_us=observed_at_us,
            allowed_session_ids=allowed_session_ids,
            drain_policy_sha256=drain_policy_sha256,
        )

    def cancel_request_helper_rebind(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        reason: str,
    ) -> None:
        return _rebind.cancel_request_helper_rebind(
            self,
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
        )

    def request_helper_rebind_state(
        self, request_id: str
    ) -> Mapping[str, object] | None:
        return _rebind.request_helper_rebind_state(self, request_id)

    @staticmethod
    def _sessions_unchanged(
        source: ModelPhoneResidencyLayout,
        target: ModelPhoneResidencyLayout,
        session_ids: Sequence[str],
    ) -> bool:
        return _planning._sessions_unchanged(source, target, session_ids)

    def request_helper_layout_is_usable(
        self,
        request_id: str,
        generation: int,
        geometry_sha256: str,
    ) -> bool:
        """Return whether an acquired helper's exact sessions remain ready."""
        return _planning.request_helper_layout_is_usable(
            self,
            request_id,
            generation,
            geometry_sha256,
        )

    def _collect_drained_phone_layouts(self) -> None:
        return _planning._collect_drained_phone_layouts(self)

    def _restore_ready_phone_layout(self) -> ModelPhoneResidencyLayout | None:
        return _planning._restore_ready_phone_layout(self)

    def reject_phone_layout_proposal(
        self,
        generation: int,
        *,
        observed_at_us: int,
        reason: str,
    ) -> ModelPhoneResidencyLayout | None:
        """Discard one stale proposal without changing physical residency."""
        return _planning.reject_phone_layout_proposal(
            self,
            generation,
            observed_at_us=observed_at_us,
            reason=reason,
        )

    def propose_phone_layout(
        self,
        layout: PhoneFfnResidencyLayout,
        *,
        workspace_bytes: int,
        shared_compute_resource_id: str,
        shared_transport_resource_ids: Sequence[str],
        observed_at_us: int,
        selection_reason: str | None = None,
        queue_work_by_artifact: Mapping[str, int] | None = None,
        queue_benefit_uj: int | None = None,
        transition_cost_uj: int | None = None,
        switching_margin_uj: int | None = None,
        minimum_residency_us: int | None = None,
        force: bool = False,
        progressive: bool = False,
    ) -> ModelPhoneResidencyLayout:
        """Publish one structural proposal without claiming residency."""
        return _planning.propose_phone_layout(
            self,
            layout,
            workspace_bytes=workspace_bytes,
            shared_compute_resource_id=shared_compute_resource_id,
            shared_transport_resource_ids=shared_transport_resource_ids,
            observed_at_us=observed_at_us,
            selection_reason=selection_reason,
            queue_work_by_artifact=queue_work_by_artifact,
            queue_benefit_uj=queue_benefit_uj,
            transition_cost_uj=transition_cost_uj,
            switching_margin_uj=switching_margin_uj,
            minimum_residency_us=minimum_residency_us,
            force=force,
            progressive=progressive,
        )

    def phone_preload_inflight(self) -> bool:
        """An already selected cold superset is still being published."""
        return _planning.phone_preload_inflight(self)

    def _advance_phone_preload(
        self, ready: ModelPhoneResidencyLayout, observed_at_us: int
    ) -> None:
        return _planning._advance_phone_preload(self, ready, observed_at_us)

    def _restore_phone_session_replacement(
        self,
        generation: int,
        *,
        observed_at_us: int,
        reason: str,
        restored_session_generations: Mapping[str, int] | None = None,
    ) -> None:
        return _planning._restore_phone_session_replacement(
            self,
            generation,
            observed_at_us=observed_at_us,
            reason=reason,
            restored_session_generations=restored_session_generations,
        )

    def begin_phone_layout_transition(
        self,
        generation: int,
        *,
        ticket_id: str,
        transition_ids: Sequence[str],
        ready_at_us: int,
        projection_token_sha256: str,
        workspace_bytes: int,
        observed_at_us: int,
    ) -> ModelPhoneResidencyLayout:
        return _transitions.begin_phone_layout_transition(
            self,
            generation,
            ticket_id=ticket_id,
            transition_ids=transition_ids,
            ready_at_us=ready_at_us,
            projection_token_sha256=projection_token_sha256,
            workspace_bytes=workspace_bytes,
            observed_at_us=observed_at_us,
        )

    def complete_phone_layout_transition(
        self,
        *,
        generation: int,
        ticket_id: str,
        transition_ids: Sequence[str],
        geometry_sha256: str,
        projection_token_sha256: str,
        finished_at_us: int,
    ) -> ModelPhoneResidencyLayout | None:
        return _transitions.complete_phone_layout_transition(
            self,
            generation=generation,
            ticket_id=ticket_id,
            transition_ids=transition_ids,
            geometry_sha256=geometry_sha256,
            projection_token_sha256=projection_token_sha256,
            finished_at_us=finished_at_us,
        )

    def verify_observed_phone_layout(
        self,
        generation: int,
        *,
        workspace_bytes: int,
        verified_at_us: int,
        verification_sha256: str,
    ) -> ModelPhoneResidencyLayout | None:
        """Adopt an already-hot layout proven by a runtime snapshot."""
        return _transitions.verify_observed_phone_layout(
            self,
            generation,
            workspace_bytes=workspace_bytes,
            verified_at_us=verified_at_us,
            verification_sha256=verification_sha256,
        )

    def adopt_verified_phone_layout(
        self,
        layout: PhoneFfnResidencyLayout,
        *,
        workspace_bytes: int,
        shared_compute_resource_id: str,
        shared_transport_resource_ids: Sequence[str],
        verified_at_us: int,
        verification_sha256: str,
        selection_reason: str | None = None,
    ) -> ModelPhoneResidencyLayout:
        """Hydrate exact physical session epochs after a scheduler restart."""
        return _transitions.adopt_verified_phone_layout(
            self,
            layout,
            workspace_bytes=workspace_bytes,
            shared_compute_resource_id=shared_compute_resource_id,
            shared_transport_resource_ids=shared_transport_resource_ids,
            verified_at_us=verified_at_us,
            verification_sha256=verification_sha256,
            selection_reason=selection_reason,
        )

    def fail_phone_layout_transition(
        self,
        ticket_id: str,
        *,
        generation: int,
        projection_token_sha256: str,
        failed_at_us: int,
        reason: str,
        unavailable_session_ids: Sequence[str] = (),
        restored_session_generations: Mapping[str, int] | None = None,
    ) -> bool:
        return _transitions.fail_phone_layout_transition(
            self,
            ticket_id,
            generation=generation,
            projection_token_sha256=projection_token_sha256,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable_session_ids=unavailable_session_ids,
            restored_session_generations=restored_session_generations,
        )

    def reset_phone_layout_transition(
        self,
        ticket_id: str,
        *,
        generation: int,
        projection_token_sha256: str,
        observed_at_us: int,
        reason: str,
    ) -> ModelPhoneResidencyLayout | None:
        """Return an obsolete queued transition to its proposal state."""
        return _transitions.reset_phone_layout_transition(
            self,
            ticket_id,
            generation=generation,
            projection_token_sha256=projection_token_sha256,
            observed_at_us=observed_at_us,
            reason=reason,
        )

    def phone_layout(self, generation: int) -> ModelPhoneResidencyLayout:
        generation = _integer("phone layout generation", generation, 1)
        try:
            return self._phone_layouts[generation]
        except KeyError as exc:
            raise ModelPlacementControllerError(
                "phone layout generation is absent"
            ) from exc

    def ready_phone_layout(self) -> ModelPhoneResidencyLayout | None:
        generation = self._ready_phone_layout_generation
        return None if generation is None else self._phone_layouts[generation]

    def target_phone_layout(self) -> ModelPhoneResidencyLayout | None:
        generation = self._target_phone_layout_generation
        return None if generation is None else self._phone_layouts[generation]

    def planning_phone_layout(self) -> ModelPhoneResidencyLayout | None:
        return self.target_phone_layout() or self.ready_phone_layout()

    def preparing_phone_layout(
        self,
    ) -> ModelPhoneResidencyLayout | None:
        target = self.target_phone_layout()
        if target is None or target.state != "PREPARING":
            return None
        return target

    def phone_layout_events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            MappingProxyType(dict(event))
            for event in self._phone_layout_events
        )

    def notify(
        self, artifact_sha256: str, reason: str, observed_at_us: int
    ) -> None:
        return _epochs.notify(self, artifact_sha256, reason, observed_at_us)

    @staticmethod
    def _snapshot_reasons(
        previous: ModelDemandSnapshot | None,
        current: ModelDemandSnapshot,
    ) -> set[str]:
        return _epochs._snapshot_reasons(previous, current)

    def evaluate(
        self, trigger: ModelPlacementTrigger
    ) -> ModelPlacementAction:
        return _epochs.evaluate(self, trigger)

    def authorize_publication(
        self, trigger: ModelPlacementTrigger
    ) -> ModelPlacementAction:
        """Apply the sole current-versus-proposed epoch publication gate."""
        return _epochs.authorize_publication(self, trigger)

    def mark_background_complete(
        self, artifact_sha256: str, observed_at_us: int
    ) -> None:
        return _epochs.mark_background_complete(self, artifact_sha256, observed_at_us)

    def bind_dispatched_request(
        self,
        request_id: str,
        artifact_sha256: str,
        route_id: str,
        resident_component_identity_sha256: str,
        fraction_ppm: int,
        *,
        desktop_placement_sha256: str | None = None,
        kv_cache_owner_id: str | None = None,
        sequence_identity: str | None = None,
        server_slot_id: int | None = None,
        helper_envelope: RequestHelperEnvelopeBinding | None = None,
        output_tokens: int | None = None,
    ) -> None:
        return _requests.bind_dispatched_request(
            self,
            request_id,
            artifact_sha256,
            route_id,
            resident_component_identity_sha256,
            fraction_ppm,
            desktop_placement_sha256=desktop_placement_sha256,
            kv_cache_owner_id=kv_cache_owner_id,
            sequence_identity=sequence_identity,
            server_slot_id=server_slot_id,
            helper_envelope=helper_envelope,
            output_tokens=output_tokens,
        )

    def mark_request_acquired(self, request_id: str) -> None:
        return _requests.mark_request_acquired(self, request_id)

    def request_is_acquired(self, request_id: str) -> bool:
        return _requests.request_is_acquired(self, request_id)

    def record_request_decode_progress(
        self, request_id: str, token_index: int
    ) -> int:
        return _requests.record_request_decode_progress(self, request_id, token_index)

    def remaining_request_decode_tokens(
        self, request_id: str, output_tokens: int
    ) -> int:
        return _requests.remaining_request_decode_tokens(self, request_id, output_tokens)

    def request_decode_token_index(self, request_id: str) -> int:
        return _requests.request_decode_token_index(self, request_id)

    def bind_request_server_slot(
        self,
        request_id: str,
        *,
        sequence_identity: str,
        server_slot_id: int,
    ) -> None:
        return _requests.bind_request_server_slot(
            self,
            request_id,
            sequence_identity=sequence_identity,
            server_slot_id=server_slot_id,
        )

    def bind_request_helper_envelope(
        self,
        request_id: str,
        envelope: RequestHelperEnvelopeBinding,
        *,
        observed_at_us: int,
    ) -> RequestPlacementBinding:
        """Bind one exact helper generation without changing the base route."""
        return _requests.bind_request_helper_envelope(
            self,
            request_id,
            envelope,
            observed_at_us=observed_at_us,
        )

    def request_helper_envelope_is_additive(
        self,
        request_id: str,
        envelope: RequestHelperEnvelopeBinding,
    ) -> bool:
        """Return whether a READY envelope only adds verified sessions."""
        return _requests.request_helper_envelope_is_additive(self, request_id, envelope)

    def expand_request_helper_envelope(
        self,
        request_id: str,
        envelope: RequestHelperEnvelopeBinding,
        *,
        resident_component_identity_sha256: str,
        start_token_index: int,
        fraction_ppm: int,
        lease_tokens: Sequence[str],
        lease_reserved_until_us: int | None,
        observed_at_us: int,
    ) -> RequestPlacementBinding:
        """Attach newly READY sessions at one decode boundary."""
        return _requests.expand_request_helper_envelope(
            self,
            request_id,
            envelope,
            resident_component_identity_sha256=resident_component_identity_sha256,
            start_token_index=start_token_index,
            fraction_ppm=fraction_ppm,
            lease_tokens=lease_tokens,
            lease_reserved_until_us=lease_reserved_until_us,
            observed_at_us=observed_at_us,
        )

    def commit_request_helper_rebind(
        self,
        request_id: str,
        envelope: RequestHelperEnvelopeBinding,
        *,
        resident_component_identity_sha256: str,
        start_token_index: int,
        observed_at_us: int,
    ) -> RequestPlacementBinding:
        """Replace only a quiesced helper identity after its target is ready."""
        return _rebind.commit_request_helper_rebind(
            self,
            request_id,
            envelope,
            resident_component_identity_sha256=resident_component_identity_sha256,
            start_token_index=start_token_index,
            observed_at_us=observed_at_us,
        )

    def _record_request_helper_event(
        self,
        request_id: str,
        kind: str,
        observed_at_us: int,
        values: Mapping[str, object],
    ) -> None:
        return _attachment._record_request_helper_event(
            self,
            request_id,
            kind,
            observed_at_us,
            values,
        )

    def attach_request_helper(
        self,
        request_id: str,
        *,
        phone_layout_generation: int,
        phone_layout_geometry_sha256: str,
        resident_component_identity_sha256: str,
        operator_plan_sha256: str,
        start_token_index: int,
        fraction_ppm: int,
        lease_tokens: Sequence[str],
        lease_reserved_until_us: int | None,
        observed_at_us: int,
    ) -> RequestPlacementBinding:
        return _attachment.attach_request_helper(
            self,
            request_id,
            phone_layout_generation=phone_layout_generation,
            phone_layout_geometry_sha256=phone_layout_geometry_sha256,
            resident_component_identity_sha256=resident_component_identity_sha256,
            operator_plan_sha256=operator_plan_sha256,
            start_token_index=start_token_index,
            fraction_ppm=fraction_ppm,
            lease_tokens=lease_tokens,
            lease_reserved_until_us=lease_reserved_until_us,
            observed_at_us=observed_at_us,
        )

    def record_request_helper_work(
        self,
        request_id: str,
        completed_phone_calls: int,
        *,
        observed_at_us: int,
    ) -> None:
        """Record physical work that makes an acquired layout a blocker."""
        return _attachment.record_request_helper_work(
            self,
            request_id,
            completed_phone_calls,
            observed_at_us=observed_at_us,
        )

    def update_request_fraction(
        self,
        request_id: str,
        fraction_ppm: int,
        *,
        resident_component_identity_sha256: str | None = None,
        observed_at_us: int = 0,
    ) -> None:
        return _attachment.update_request_fraction(
            self,
            request_id,
            fraction_ppm,
            resident_component_identity_sha256=resident_component_identity_sha256,
            observed_at_us=observed_at_us,
        )

    def renew_request_helper_leases(
        self,
        request_id: str,
        *,
        lease_tokens: Sequence[str],
        reserved_until_us: int,
        observed_at_us: int,
    ) -> None:
        return _attachment.renew_request_helper_leases(
            self,
            request_id,
            lease_tokens=lease_tokens,
            reserved_until_us=reserved_until_us,
            observed_at_us=observed_at_us,
        )

    def detach_request_helper(
        self,
        request_id: str,
        *,
        fallback_outcome: str,
        observed_at_us: int,
    ) -> RequestPlacementBinding:
        return _attachment.detach_request_helper(
            self,
            request_id,
            fallback_outcome=fallback_outcome,
            observed_at_us=observed_at_us,
        )

    def release_request(self, request_id: str) -> None:
        return _attachment.release_request(self, request_id)

    def request_binding(
        self, request_id: str
    ) -> Mapping[str, object] | None:
        return _attachment.request_binding(self, request_id)

    def request_helper_events(
        self, request_id: str | None = None
    ) -> tuple[Mapping[str, object], ...]:
        return _attachment.request_helper_events(self, request_id)

    def record_request_helper_event(
        self,
        request_id: str,
        kind: str,
        observed_at_us: int,
        values: Mapping[str, object],
    ) -> None:
        return _attachment.record_request_helper_event(
            self,
            request_id,
            kind,
            observed_at_us,
            values,
        )

    def _record_event(
        self,
        snapshot: ModelDemandSnapshot,
        action: ModelPlacementAction,
    ) -> None:
        body = {
            "action": action.to_json(),
            "artifact_sha256": snapshot.artifact_sha256,
            "event_index": len(self._events),
            "observed_at_us": snapshot.observed_at_us,
            "snapshot": snapshot.to_json(),
        }
        event = MappingProxyType({
            **body,
            "event_sha256": canonical_sha256(body),
            "schema": "research-scheduler-model-placement-event-v1",
        })
        self._events.append(event)
        if len(self._events) > self.policy.maximum_events:
            del self._events[:len(self._events) - self.policy.maximum_events]
        self._counters[action.kind] = self._counters.get(action.kind, 0) + 1

    def events(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            MappingProxyType(dict(event)) for event in self._events
        )

    def stats(self) -> Mapping[str, object]:
        ready = self.ready_phone_layout()
        target = self.target_phone_layout()
        return MappingProxyType({
            "background_inflight": len(self._background_inflight),
            "acquired_requests": len(self._acquired_request_ids),
            "bound_requests": len(self._request_bindings),
            "events": len(self._events),
            "pending_artifacts": len(self._pending_reasons),
            "phone_layout_events": len(self._phone_layout_events),
            "phone_layout_generation": self._phone_layout_generation,
            "phone_sessions": [
                state.to_json()
                for _, state in sorted(self._phone_session_states.items())
            ],
            "ready_phone_layout_generation": (
                None if ready is None else ready.generation
            ),
            "snapshots": len(self._snapshots),
            "target_phone_layout_generation": (
                None if target is None else target.generation
            ),
            "target_phone_layout_state": (
                None if target is None else target.state
            ),
            "pending_phone_layout_geometry_sha256": (
                self._pending_phone_layout_geometry_sha256
            ),
            "pending_phone_layout_snapshot_count": (
                self._pending_phone_layout_snapshot_count
            ),
            **dict(sorted(self._counters.items())),
        })
