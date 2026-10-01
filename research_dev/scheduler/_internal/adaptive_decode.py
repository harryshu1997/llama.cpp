"""Scheduler-owned online adaptation across decode-token windows."""

from __future__ import annotations

import copy
import threading
from typing import Callable, Mapping, Sequence

from .adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeControl,
    AdaptiveDecodeDirective,
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeRawWindowObservation,
    AdaptiveDecodeWindowBoundary,
    AdaptiveDecodeWindowReceipt,
    validate_policy_set,
)
from .types import canonical_sha256


from .adaptive_decode_state import (
    _AdaptiveServerPolicy,
    _AdaptiveSession as _AdaptiveSession,
    _AssumedPhonePowerQuery as _AssumedPhonePowerQuery,
    AdaptiveDecodeHistoricalEstimate as AdaptiveDecodeHistoricalEstimate,
)

from .adaptive_decode_ops.common import (
    ADAPTIVE_OBSERVATION_STORE_SCHEMA as ADAPTIVE_OBSERVATION_STORE_SCHEMA,
    HELPER_EVIDENCE_STATES as HELPER_EVIDENCE_STATES,
    _assumed_phone_power_bounds as _assumed_phone_power_bounds,
    _assumed_phone_power_session as _assumed_phone_power_session,
    _operator_subset_bounds as _operator_subset_bounds,
    _window_has_phone_work as _window_has_phone_work,
)
from .adaptive_decode_ops import admission as _admission
from .adaptive_decode_ops import assumed_power as _assumed_power
from .adaptive_decode_ops import bounds as _bounds
from .adaptive_decode_ops import budgeting as _budgeting
from .adaptive_decode_ops import candidates as _candidates
from .adaptive_decode_ops import completion as _completion
from .adaptive_decode_ops import estimates as _estimates
from .adaptive_decode_ops import helpers as _helpers
from .adaptive_decode_ops import history as _history
from .adaptive_decode_ops import promotion as _promotion
from .adaptive_decode_ops import reporting as _reporting
from .adaptive_decode_ops import sequencing as _sequencing
from .adaptive_decode_ops import coherence as _coherence
from .adaptive_decode_ops import prefill_yield as _prefill_yield
from .adaptive_decode_ops import windows as _windows

__all__ = [
    'ADAPTIVE_OBSERVATION_STORE_SCHEMA',
    'AdaptiveDecodeConfig',
    'AdaptiveDecodeControl',
    'AdaptiveDecodeController',
    'AdaptiveDecodeDirective',
    'AdaptiveDecodeError',
    'AdaptiveDecodeGroupedObservation',
    'AdaptiveDecodeHistoricalEstimate',
    'AdaptiveDecodePolicy',
    'AdaptiveDecodePolicyAck',
    'AdaptiveDecodeRawWindowObservation',
    'AdaptiveDecodeWindowBoundary',
    'AdaptiveDecodeWindowReceipt',
    'HELPER_EVIDENCE_STATES',
    '_AdaptiveSession',
    '_AssumedPhonePowerQuery',
    '_admission',
    '_assumed_phone_power_bounds',
    '_assumed_phone_power_session',
    '_assumed_power',
    '_bounds',
    '_budgeting',
    '_candidates',
    '_completion',
    '_estimates',
    '_helpers',
    '_history',
    '_operator_subset_bounds',
    '_promotion',
    '_reporting',
    '_sequencing',
    '_window_has_phone_work',
    '_windows',
    'canonical_sha256',
    'validate_policy_set',
]


class AdaptiveDecodeController:
    """Own probe sequencing, conservative promotion, and window journaling."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, _AdaptiveSession] = {}
        self._sealed_sessions: dict[str, _AdaptiveSession] = {}
        self._completed: dict[str, AdaptiveDecodeGroupedObservation] = {}
        self._history: dict[str, AdaptiveDecodeGroupedObservation] = {}
        self._history_component_bindings: dict[
            str, tuple[str, str]
        ] = {}
        self._server_policies: dict[tuple, _AdaptiveServerPolicy] = {}
        # Device membership (phones lost or absent), a physical fact kept outside checkpoints:
        # quarantine drops are re-applied to every group that is fetched, restored or created.
        self._quarantined_devices: dict[str, str] = {}
        # Joiners in their prompt phase next to co-tenants (dispatch_policy.joint_planner active,
        # adaptive_decode_ops.prefill_yield), also a physical fact kept outside checkpoints.
        self._prefill_yields: dict[str, dict[str, object]] = {}
        self._prefill_yield_events: list[dict[str, object]] = []

    @staticmethod
    def _clone_session(session: _AdaptiveSession) -> _AdaptiveSession:
        result = copy.copy(session)
        result.state_history = list(session.state_history)
        result.records = list(session.records)
        result.probe_candidates = list(session.probe_candidates)
        result.decision_time_us = list(session.decision_time_us)
        result.historical_records = dict(session.historical_records)
        result.historical_group_counts = dict(
            session.historical_group_counts
        )
        result.warmup_windows_seen_by_policy = dict(
            session.warmup_windows_seen_by_policy
        )
        result.warmup_latency_us_by_policy = dict(session.warmup_latency_us_by_policy)
        result.eliminated_policy_reasons = dict(
            session.eliminated_policy_reasons
        )
        result.maintenance_policy_hashes = set(
            session.maintenance_policy_hashes
        )
        result.verification_budget = (
            None if session.verification_budget is None
            else dict(session.verification_budget)
        )
        result.probe_budget = (
            None if session.probe_budget is None else dict(session.probe_budget)
        )
        result.probe_attempts = dict(session.probe_attempts)
        result.qualification_measurement_plan = (
            None if session.qualification_measurement_plan is None
            else dict(session.qualification_measurement_plan)
        )
        result.policy_evidence_aliases = dict(session.policy_evidence_aliases)
        result.context_monitor_prior = (
            None if session.context_monitor_prior is None else dict(session.context_monitor_prior)
        )
        result.window_token_observations = list(session.window_token_observations)
        return result

    def checkpoint(self) -> object:
        with self._lock:
            return (
                "adaptive-decode-checkpoint-v4",
                {
                    request_id: self._clone_session(session)
                    for request_id, session in self._sessions.items()
                },
                {
                    request_id: self._clone_session(session)
                    for request_id, session in self._sealed_sessions.items()
                },
                dict(self._completed),
                dict(self._history),
                dict(self._history_component_bindings),
                {key: copy.copy(value) for key, value in self._server_policies.items()},
            )

    def restore(self, checkpoint: object) -> None:
        if (
            type(checkpoint) is not tuple
            or len(checkpoint) != 7
            or checkpoint[0] != "adaptive-decode-checkpoint-v4"
            or type(checkpoint[1]) is not dict
            or type(checkpoint[2]) is not dict
            or type(checkpoint[3]) is not dict
            or type(checkpoint[4]) is not dict
            or type(checkpoint[5]) is not dict
            or type(checkpoint[6]) is not dict
        ):
            raise AdaptiveDecodeError("adaptive checkpoint is invalid")
        sessions = checkpoint[1]
        sealed_sessions = checkpoint[2]
        completed = checkpoint[3]
        history = checkpoint[4]
        component_bindings = checkpoint[5]
        if any(type(key) is not tuple or len(key) != 4
               or not isinstance(value, _AdaptiveServerPolicy)
               for key, value in checkpoint[6].items()):
            raise AdaptiveDecodeError("adaptive checkpoint server policies are invalid")
        if any(
            type(request_id) is not str
            or not isinstance(session, _AdaptiveSession)
            for request_id, session in sessions.items()
        ) or any(
            type(request_id) is not str
            or not isinstance(session, _AdaptiveSession)
            for request_id, session in sealed_sessions.items()
        ) or any(
            type(request_id) is not str
            or not isinstance(grouped, AdaptiveDecodeGroupedObservation)
            for request_id, grouped in completed.items()
        ):
            raise AdaptiveDecodeError("adaptive checkpoint state is invalid")
        if set(sessions) & set(sealed_sessions):
            raise AdaptiveDecodeError("adaptive checkpoint owner is duplicated")
        if any(
            type(group_sha256) is not str
            or not isinstance(grouped, AdaptiveDecodeGroupedObservation)
            or group_sha256 != grouped.grouped_observation_sha256
            for group_sha256, grouped in history.items()
        ):
            raise AdaptiveDecodeError("adaptive checkpoint history is invalid")
        if any(
            group_sha256 not in history
            or type(binding) is not tuple
            or len(binding) != 2
            or any(not self._valid_sha256(value) for value in binding)
            for group_sha256, binding in component_bindings.items()
        ):
            raise AdaptiveDecodeError(
                "adaptive checkpoint component binding is invalid"
            )
        with self._lock:
            self._sessions = {
                request_id: self._clone_session(session)
                for request_id, session in sessions.items()
            }
            self._sealed_sessions = {
                request_id: self._clone_session(session)
                for request_id, session in sealed_sessions.items()
            }
            self._completed = dict(completed)
            self._history = dict(history)
            self._history_component_bindings = dict(component_bindings)
            self._server_policies = {
                key: copy.copy(value) for key, value in checkpoint[6].items()
            }

    def observation_snapshot(self) -> Mapping[str, object]:
        return _history.observation_snapshot(self)

    def load_observations(
        self, value: object, *, merge: bool = False
    ) -> None:
        return _history.load_observations(self, value, merge=merge)

    def observation_state(self) -> Mapping[str, int]:
        return _history.observation_state(self)

    @staticmethod
    def _valid_sha256(value: object) -> bool:
        return _history._valid_sha256(value)

    def rebind_legacy_component_observations(
        self,
        *,
        model_artifact_sha256: str,
        source_component_capability_sha256: str,
        component_capability_sha256: str,
        source_planning_profile_sha256s: Sequence[str],
        baseline: AdaptiveDecodePolicy,
        candidates: Sequence[AdaptiveDecodePolicy],
    ) -> int:
        """Bind verified legacy groups to one stable execution component."""
        return _history.rebind_legacy_component_observations(
            self,
            model_artifact_sha256=model_artifact_sha256,
            source_component_capability_sha256=source_component_capability_sha256,
            component_capability_sha256=component_capability_sha256,
            source_planning_profile_sha256s=source_planning_profile_sha256s,
            baseline=baseline,
            candidates=candidates,
        )

    def historical_route_estimate(
        self,
        *,
        model_artifact_sha256: str,
        planning_profile_sha256: str,
        baseline: AdaptiveDecodePolicy,
        candidates: Sequence[AdaptiveDecodePolicy],
        context_length: int,
        active_batch: int,
        config: AdaptiveDecodeConfig,
        minimum_group_count: int = 2,
        component_capability_sha256: str | None = None,
    ) -> AdaptiveDecodeHistoricalEstimate | None:
        """Return a qualified fraction estimate from grouped observations."""
        return _estimates.historical_route_estimate(
            self,
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            baseline=baseline,
            candidates=candidates,
            context_length=context_length,
            active_batch=active_batch,
            config=config,
            minimum_group_count=minimum_group_count,
            component_capability_sha256=component_capability_sha256,
        )

    def historical_component_latency(
        self,
        *,
        model_artifact_sha256: str,
        planning_profile_sha256: str,
        component_capability_sha256: str,
        baseline: AdaptiveDecodePolicy,
        candidates: Sequence[AdaptiveDecodePolicy],
        selected_policy: AdaptiveDecodePolicy,
        context_length: int,
        active_batch: int,
        config: AdaptiveDecodeConfig,
        minimum_group_count: int = 1,
    ) -> tuple[int, int, int] | None:
        """Return physical latency evidence without reusing route energy."""
        return _estimates.historical_component_latency(
            self,
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            component_capability_sha256=component_capability_sha256,
            baseline=baseline,
            candidates=candidates,
            selected_policy=selected_policy,
            context_length=context_length,
            active_batch=active_batch,
            config=config,
            minimum_group_count=minimum_group_count,
        )

    def historical_route_estimate_with_assumed_phone_power(
        self,
        *,
        model_artifact_sha256: str,
        planning_profile_sha256: str,
        component_capability_sha256: str,
        baseline: AdaptiveDecodePolicy,
        candidates: Sequence[AdaptiveDecodePolicy],
        context_length: int,
        active_batch: int,
        config: AdaptiveDecodeConfig,
        phone_power_by_domain: Mapping[str, tuple[int, int]],
        minimum_group_count: int = 2,
        route_geometry_prior: bool = False,
        operator_subset_prior: bool = False,
        operator_subset_source_layer_mask: int | None = None,
    ) -> AdaptiveDecodeHistoricalEstimate | None:
        """Rebuild fleet energy from separable domains and phone power."""
        return _assumed_power.historical_route_estimate_with_assumed_phone_power(
            self,
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
            component_capability_sha256=component_capability_sha256,
            baseline=baseline,
            candidates=candidates,
            context_length=context_length,
            active_batch=active_batch,
            config=config,
            phone_power_by_domain=phone_power_by_domain,
            minimum_group_count=minimum_group_count,
            route_geometry_prior=route_geometry_prior,
            operator_subset_prior=operator_subset_prior,
            operator_subset_source_layer_mask=operator_subset_source_layer_mask,
        )

    def _validate_assumed_phone_power_query(
        self,
        query: _AssumedPhonePowerQuery,
        context_length: int,
    ) -> None:
        return _assumed_power._validate_assumed_phone_power_query(self, query, context_length)

    def _assumed_phone_power_prior_context_lengths(
        self,
        query: _AssumedPhonePowerQuery,
        context_length: int,
    ) -> tuple[
        Callable[[AdaptiveDecodeGroupedObservation], bool],
        tuple[int, ...],
    ]:
        return _assumed_power._assumed_phone_power_prior_context_lengths(
            self,
            query,
            context_length,
        )

    def _assumed_phone_power_policy_matches(
        self,
        query: _AssumedPhonePowerQuery,
        observed: AdaptiveDecodePolicy,
        policy: AdaptiveDecodePolicy,
    ) -> bool:
        return _assumed_power._assumed_phone_power_policy_matches(self, query, observed, policy)

    def _assumed_phone_power_records_for(
        self,
        query: _AssumedPhonePowerQuery,
        policy: AdaptiveDecodePolicy,
        session: _AdaptiveSession,
        group_matches: Callable[[AdaptiveDecodeGroupedObservation], bool],
        context_bucket: int,
    ) -> tuple[
        tuple[AdaptiveDecodeWindowReceipt, ...],
        int,
        AdaptiveDecodePolicy | None,
    ]:
        return _assumed_power._assumed_phone_power_records_for(
            self,
            query,
            policy,
            session,
            group_matches,
            context_bucket,
        )

    def _assumed_phone_power_estimate_for_context(
        self,
        query: _AssumedPhonePowerQuery,
        group_matches: Callable[[AdaptiveDecodeGroupedObservation], bool],
        evidence_context_length: int,
        requested_context_bucket: int,
    ) -> AdaptiveDecodeHistoricalEstimate | None:
        return _assumed_power._assumed_phone_power_estimate_for_context(
            self,
            query,
            group_matches,
            evidence_context_length,
            requested_context_bucket,
        )

    @staticmethod
    def _policy_identity(policy: AdaptiveDecodePolicy) -> str:
        return _history._policy_identity(policy)

    @staticmethod
    def _policy_geometry_identity(policy: AdaptiveDecodePolicy) -> str:
        return _history._policy_geometry_identity(policy)

    def _group_matches_session(
        self,
        grouped: AdaptiveDecodeGroupedObservation,
        session: _AdaptiveSession,
    ) -> bool:
        return _history._group_matches_session(self, grouped, session)

    def _compatible_context_lengths(
        self, session: _AdaptiveSession
    ) -> tuple[int, ...]:
        return _history._compatible_context_lengths(self, session)

    def _historical_policy_records(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        *,
        compatible_context: bool = False,
    ) -> tuple[tuple[AdaptiveDecodeWindowReceipt, ...], int]:
        return _history._historical_policy_records(
            self,
            session,
            policy,
            compatible_context=compatible_context,
        )

    def _cached_verification_policy(
        self, session: _AdaptiveSession, *, operational: bool = False
    ) -> AdaptiveDecodePolicy | None:
        return _candidates._cached_verification_policy(self, session, operational=operational)

    def _operational_verification_policy(
        self, session: _AdaptiveSession
    ) -> AdaptiveDecodePolicy | None:
        return _candidates._operational_verification_policy(self, session)

    def _seed_verification_candidate(self, session: _AdaptiveSession) -> None:
        return _candidates._seed_verification_candidate(self, session)

    @staticmethod
    def _representative_candidates(
        candidates: Sequence[AdaptiveDecodePolicy],
    ) -> list[AdaptiveDecodePolicy]:
        return _candidates._representative_candidates(candidates)

    @classmethod
    def _sample_candidates(
        cls,
        candidates: Sequence[AdaptiveDecodePolicy],
        config: AdaptiveDecodeConfig,
        helper_evidence_state: str = "TRUSTED",
    ) -> list[AdaptiveDecodePolicy]:
        return _candidates._sample_candidates(cls, candidates, config, helper_evidence_state)

    @staticmethod
    def _state(session: _AdaptiveSession, state: str) -> None:
        return _sequencing._state(session, state)

    @staticmethod
    def _operational_energy_eligible(
        session: _AdaptiveSession,
        row: AdaptiveDecodeWindowReceipt,
    ) -> bool:
        return _bounds._operational_energy_eligible(session, row)

    @classmethod
    def _valid_records(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        *,
        operational: bool = False,
    ) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
        return _bounds._valid_records(
            AdaptiveDecodeController,
            cls,
            session,
            policy,
            operational=operational,
        )

    @classmethod
    def _current_valid_records(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        *,
        operational: bool = False,
    ) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
        return _bounds._current_valid_records(cls, session, policy, operational=operational)

    @staticmethod
    def _current_valid_latency_records(
        session: _AdaptiveSession, policy: AdaptiveDecodePolicy
    ) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
        return _bounds._current_valid_latency_records(session, policy)

    @classmethod
    def _current_latency_bounds(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        *,
        minimum_count: int = 2,
    ) -> tuple[int, int] | None:
        return _bounds._current_latency_bounds(cls, session, policy, minimum_count=minimum_count)

    @staticmethod
    def _valid_latency_records(
        session: _AdaptiveSession, policy: AdaptiveDecodePolicy
    ) -> tuple[AdaptiveDecodeWindowReceipt, ...]:
        return _bounds._valid_latency_records(AdaptiveDecodeController, session, policy)

    @classmethod
    def _latency_bounds(
        cls, session: _AdaptiveSession, policy: AdaptiveDecodePolicy
    ) -> tuple[int, int] | None:
        return _bounds._latency_bounds(cls, session, policy)

    @classmethod
    def _bounds(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        *,
        operational: bool = False,
    ) -> tuple[int, int, int, int, int] | None:
        return _bounds._bounds(cls, session, policy, operational=operational)

    @classmethod
    def _token_latency_us(
        cls, session: _AdaptiveSession, policy: AdaptiveDecodePolicy
    ) -> int:
        return _bounds._token_latency_us(cls, session, policy)

    @classmethod
    def _window_tokens(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int,
    ) -> int:
        return _budgeting._window_tokens(cls, session, policy, token_index)

    @staticmethod
    def _remaining_tokens(session: _AdaptiveSession, token_index: int) -> int:
        return _budgeting._remaining_tokens(session, token_index)

    def _can_probe(
        self, session: _AdaptiveSession, token_index: int, at_us: int
    ) -> bool:
        return _budgeting._can_probe(self, session, token_index, at_us)

    @staticmethod
    def _probe_admitted(session, policy, token_index, at_us) -> bool:
        return _budgeting._probe_admitted(session, policy, token_index, at_us)

    def helper_attachment_opportunity(self, request_id: str, *, token_index: int, at_us: int) -> str:
        """Use the measurement admission budget before acquiring helper leases."""
        return _budgeting.helper_attachment_opportunity(
            self,
            request_id,
            token_index=token_index,
            at_us=at_us,
        )

    def _verification_control_cost(self, session: _AdaptiveSession) -> tuple[int, int]:
        return _budgeting._verification_control_cost(self, session)

    def preview_helper_replacement(self, request_id: str, *, retained_layer_mask: int,
                                   token_index: int, at_us: int, transition_latency_us: int):
        return _budgeting.preview_helper_replacement(
            self, request_id, retained_layer_mask=retained_layer_mask,
            token_index=token_index, at_us=at_us, transition_latency_us=transition_latency_us)

    def _measurement_pair_budget(
        self, session: _AdaptiveSession, policy: AdaptiveDecodePolicy,
        token_index: int, at_us: int,
        *, baseline_windows: int | None = None, candidate_windows: int = 1,
    ) -> dict[str, int] | None:
        return _budgeting._measurement_pair_budget(
            self, session, policy, token_index, at_us,
            baseline_windows=baseline_windows, candidate_windows=candidate_windows)

    @staticmethod
    def _context_identity(session: _AdaptiveSession) -> str:
        return _budgeting._context_identity(session)

    def _probe_attempt_key(self, session, policy) -> str:
        return _budgeting._probe_attempt_key(self, session, policy)

    def _estimated_token_energy(self, session, policy) -> int | None:
        return _budgeting._estimated_token_energy(self, session, policy)

    def _spent_exploration_energy(self, session) -> int:
        return _budgeting._spent_exploration_energy(self, session)

    def _consider_incumbent(self, session, policy, token_index, at_us) -> None:
        return _promotion._consider_incumbent(self, session, policy, token_index, at_us)

    def _best_valid_policy(self, session, token_index, at_us):
        return _promotion._best_valid_policy(self, session, token_index, at_us)

    def _continue_best(self, session, token_index, at_us, reason):
        return _promotion._continue_best(self, session, token_index, at_us, reason)

    def _incomplete_probe(self, session, token_index, at_us):
        return _sequencing._incomplete_probe(self, session, token_index, at_us)

    def _reserve_verification_pair(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
    ) -> bool:
        return _sequencing._reserve_verification_pair(self, session, token_index, at_us)

    @staticmethod
    def _verification_result(
        session: _AdaptiveSession, directive: AdaptiveDecodeDirective,
        outcome: str, reason: str,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._verification_result(session, directive, outcome, reason)

    def _defer_verification(
        self, session: _AdaptiveSession, token_index: int, at_us: int, reason: str,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._defer_verification(self, session, token_index, at_us, reason)

    def _advance_operational_verification(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._advance_operational_verification(self, session, token_index, at_us)

    def _prior_monitor_eligible(
        self, session: _AdaptiveSession, token_index: int,
    ) -> bool:
        return _sequencing._prior_monitor_eligible(self, session, token_index)

    def _begin_prior_monitor(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._begin_prior_monitor(self, session, token_index, at_us)

    def _prior_monitor_holds(self, session: _AdaptiveSession) -> bool:
        return _sequencing._prior_monitor_holds(self, session)

    def _end_prior_monitor(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
        reason: str, *, rejected: bool,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._end_prior_monitor(
            self, session, token_index, at_us, reason, rejected=rejected
        )

    def _advance_prior_monitor(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._advance_prior_monitor(self, session, token_index, at_us)

    @staticmethod
    def _open_window(
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int,
        at_us: int,
        ack: AdaptiveDecodePolicyAck | None,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._open_window(
            AdaptiveDecodeController,
            session,
            policy,
            token_index,
            at_us,
            ack,
        )

    def _control(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int | None = None,
        at_us: int | None = None,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._control(self, session, policy, token_index, at_us)

    def _coherent_phone_policy(
        self, session: _AdaptiveSession
    ) -> AdaptiveDecodePolicy | None:
        return _coherence._coherent_phone_policy(self, session)

    def _follow_coherent_policy(
        self, session: _AdaptiveSession, token_index: int, at_us: int
    ) -> AdaptiveDecodeDirective | None:
        return _coherence._follow_coherent_policy(self, session, token_index, at_us)

    def shared_server_policy_key(self, request_id: str) -> tuple | None:
        with self._lock:
            return _coherence.server_policy_key(self._terminal_session(request_id))

    def _validate_start_arguments(
        self,
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
        return _admission._validate_start_arguments(
            self,
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

    def _seed_session_history(
        self,
        session: _AdaptiveSession,
        baseline: AdaptiveDecodePolicy,
        rows: tuple[AdaptiveDecodePolicy, ...],
        config: AdaptiveDecodeConfig,
    ) -> None:
        _admission._seed_session_history(self, session, baseline, rows, config)
        _coherence.eliminate_quarantined_policies(self, session)

    @staticmethod
    def _device_id(device_id: str) -> str:
        if type(device_id) is not str or not device_id or not device_id.isascii():
            raise AdaptiveDecodeError("adaptive device id is invalid")
        return device_id

    def quarantine_device(self, device_id: str, *, reason: str, at_us: int) -> bool:
        """Remove one phone from every device set until it is readmitted (idempotent)."""
        device_id = self._device_id(device_id)
        if type(reason) is not str or not reason or not reason.isascii():
            raise AdaptiveDecodeError("adaptive quarantine reason is invalid")
        if type(at_us) is not int or at_us < 0:
            raise AdaptiveDecodeError("adaptive quarantine time is invalid")
        with self._lock:
            return _coherence.quarantine_device(self, device_id, reason, at_us)

    def readmit_device(self, device_id: str) -> bool:
        """Offer a rejoined phone's device sets again; the normal probe decides (idempotent)."""
        device_id = self._device_id(device_id)
        with self._lock:
            return _coherence.readmit_device(self, device_id)

    @property
    def quarantined_devices(self) -> Mapping[str, str]:
        with self._lock:
            return dict(self._quarantined_devices)

    def register_prefill_yield(
        self, joiner_request_id: str, *, model_artifact_sha256: str, desktop_placement_sha256: str,
        at_us: int, expires_at_us: int,
    ) -> tuple[str, tuple[str, ...]]:
        """Co-tenants of the joiner's model and desktop parent run the host policy until its decode
        starts (``adaptive_decode_ops.prefill_yield``); returns (outcome, co-tenant request ids)."""
        if type(joiner_request_id) is not str or not joiner_request_id:
            raise AdaptiveDecodeError("prefill yield joiner is invalid")
        if not (self._valid_sha256(model_artifact_sha256) and self._valid_sha256(desktop_placement_sha256)):
            raise AdaptiveDecodeError("prefill yield identity is invalid")
        if type(at_us) is not int or type(expires_at_us) is not int or not 0 <= at_us < expires_at_us:
            raise AdaptiveDecodeError("prefill yield time is invalid")
        with self._lock:
            return _prefill_yield.register(
                self, joiner_request_id, model_artifact_sha256=model_artifact_sha256,
                desktop_placement_sha256=desktop_placement_sha256, at_us=at_us,
                expires_at_us=expires_at_us)

    def clear_prefill_yield(self, joiner_request_id: str, *, at_us: int, reason: str) -> bool:
        with self._lock:
            return _prefill_yield.end(self, joiner_request_id, at_us, reason)

    def prefill_yield_events(self) -> tuple[Mapping[str, object], ...]:
        with self._lock:
            return _prefill_yield.events(self)

    def _initial_start_directive(
        self,
        session: _AdaptiveSession,
        baseline: AdaptiveDecodePolicy,
        first_token_index: int,
        first_token_at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._initial_start_directive(
            self,
            session,
            baseline,
            first_token_index,
            first_token_at_us,
        )

    def start(
        self,
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
        directive = _admission.start(
            self,
            request_id=request_id,
            ticket_id=ticket_id,
            model_artifact_sha256=model_artifact_sha256,
            planning_profile_sha256=planning_profile_sha256,
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
            component_capability_sha256=component_capability_sha256,
            helper_available=helper_available,
            helper_layout_generation=helper_layout_generation,
            helper_layout_geometry_sha256=helper_layout_geometry_sha256,
            helper_evidence_state=helper_evidence_state,
            phone_power_policy_from_capability=phone_power_policy_from_capability,
            execution_context_available=execution_context_available,
            helper_layout_identity_sha256=helper_layout_identity_sha256,
        )
        if self._prefill_yields:
            with self._lock:
                _prefill_yield.end(self, request_id, first_token_at_us, "JOINER_STARTED")
        return directive

    def helper_ready(
        self,
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
        result = _helpers.helper_ready(
            self,
            request_id,
            phone_layout_generation=phone_layout_generation,
            phone_layout_geometry_sha256=phone_layout_geometry_sha256,
            candidates=candidates,
            component_capability_sha256=component_capability_sha256,
            ticket_policy=ticket_policy,
            helper_evidence_state=helper_evidence_state,
            ready_at_token_index=ready_at_token_index,
            allow_assumed_phone_power_for_operational_selection=allow_assumed_phone_power_for_operational_selection,
            phone_layout_identity_sha256=phone_layout_identity_sha256,
        )
        self._eliminate_quarantined(request_id)
        return result

    def _eliminate_quarantined(self, request_id: str) -> None:
        with self._lock:
            session = self._sessions.get(request_id)
            if session is not None and self._quarantined_devices:
                _coherence.eliminate_quarantined_policies(self, session)

    def helper_unavailable(self, request_id: str) -> None:
        """Prevent new phone controls while preserving the request."""
        return _helpers.helper_unavailable(self, request_id)

    def adopts_late_helper(self, request_id: str, *, token_index: int) -> bool:
        """Whether a helper attached at this boundary is a late adoption (opt-in)."""
        return _helpers.adopts_late_helper(self, request_id, token_index=token_index)

    def helper_disturbance(self, request_id: str, *, reason: str | None) -> None:
        """Report work on the helper's phone that disturbs its measurements."""
        return _helpers.helper_disturbance(self, request_id, reason=reason)

    def request_helper_session_drain(
        self,
        request_id: str,
        *,
        retained_layer_mask: int,
    ) -> Mapping[str, object]:
        """Request a retained-session mask at the next decode boundary."""
        return _helpers.request_helper_session_drain(
            self,
            request_id,
            retained_layer_mask=retained_layer_mask,
        )

    def helper_rebound(
        self,
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
        result = _helpers.helper_rebound(
            self,
            request_id,
            phone_layout_generation=phone_layout_generation,
            phone_layout_geometry_sha256=phone_layout_geometry_sha256,
            candidates=candidates,
            component_capability_sha256=component_capability_sha256,
            ticket_policy=ticket_policy,
            helper_evidence_state=helper_evidence_state,
            compatible_layers_by_plan=compatible_layers_by_plan,
            phone_layout_identity_sha256=phone_layout_identity_sha256,
        )
        self._eliminate_quarantined(request_id)
        return result

    def helper_window_bid(
        self,
        request_id: str,
        *,
        requested_fraction_ppm: int | None = None,
    ) -> Mapping[str, object]:
        """Return one conservative bid for the shared helper window."""
        return _helpers.helper_window_bid(
            self,
            request_id,
            requested_fraction_ppm=requested_fraction_ppm,
        )

    def active_policy(
        self, request_id: str
    ) -> AdaptiveDecodePolicy | None:
        return _helpers.active_policy(self, request_id)

    def yield_helper_window(
        self,
        request_id: str,
        *,
        token_index: int,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        """Return a just-opened phone window to its desktop parent."""
        return _helpers.yield_helper_window(self, request_id, token_index=token_index, at_us=at_us)

    def boundary(
        self,
        request_id: str,
        *,
        slot_id: int,
        token_index: int,
        at_us: int,
        terminal: bool = False,
    ) -> AdaptiveDecodeDirective | None:
        return _windows.boundary(
            self,
            request_id,
            slot_id=slot_id,
            token_index=token_index,
            at_us=at_us,
            terminal=terminal,
        )

    def _session(self, request_id: str) -> _AdaptiveSession:
        if type(request_id) is not str or not request_id:
            raise AdaptiveDecodeError("adaptive request id is invalid")
        session = self._sessions.get(request_id)
        if session is None:
            raise AdaptiveDecodeError("adaptive request is not active")
        return session

    def _terminal_session(self, request_id: str) -> _AdaptiveSession:
        if type(request_id) is not str or not request_id:
            raise AdaptiveDecodeError("adaptive request id is invalid")
        session = self._sealed_sessions.get(request_id)
        if session is None:
            session = self._sessions.get(request_id)
        if session is None:
            raise AdaptiveDecodeError("adaptive request is not active")
        return session

    def _seal_session(self, session: _AdaptiveSession) -> None:
        return _completion._seal_session(self, session)

    def acknowledge(
        self,
        request_id: str,
        acknowledgement: AdaptiveDecodePolicyAck,
        transition_observation: AdaptiveDecodeRawWindowObservation | None = None,
    ) -> AdaptiveDecodeDirective:
        return _windows.acknowledge(self, request_id, acknowledgement, transition_observation)

    def _leading_candidate(
        self, session: _AdaptiveSession
    ) -> AdaptiveDecodePolicy | None:
        return _candidates._leading_candidate(self, session)

    @staticmethod
    def _ticket_fallback(
        session: _AdaptiveSession,
    ) -> AdaptiveDecodePolicy | None:
        return _candidates._ticket_fallback(session)

    def _refinement_candidates(
        self,
        session: _AdaptiveSession,
        leader: AdaptiveDecodePolicy,
    ) -> list[AdaptiveDecodePolicy]:
        return _candidates._refinement_candidates(self, session, leader)

    def _update_elimination(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
    ) -> None:
        return _promotion._update_elimination(self, session, policy)

    def _qualifies(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int,
        at_us: int,
    ) -> bool:
        return _promotion._qualifies(self, session, policy, token_index, at_us)

    @classmethod
    def _learning_probe_improves(
        cls,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
    ) -> bool:
        return _promotion._learning_probe_improves(cls, session, policy)

    def _qualification_needs_more_evidence(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int,
        at_us: int,
        *,
        check_budget: bool = True,
    ) -> bool:
        return _promotion._qualification_needs_more_evidence(
            self,
            session,
            policy,
            token_index,
            at_us,
            check_budget=check_budget,
        )

    def _finish_verification(
        self,
        session: _AdaptiveSession,
        policy: AdaptiveDecodePolicy,
        token_index: int,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _promotion._finish_verification(self, session, policy, token_index, at_us)

    def _select_probe_winner(
        self,
        session: _AdaptiveSession,
        token_index: int,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _promotion._select_probe_winner(self, session, token_index, at_us)

    def _advance_probe_candidate(
        self, session: _AdaptiveSession, token_index: int, at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _sequencing._advance_probe_candidate(self, session, token_index, at_us)

    def _next_after_window(
        self, session: _AdaptiveSession, token_index: int, at_us: int
    ) -> AdaptiveDecodeDirective:
        return _sequencing._next_after_window(self, session, token_index, at_us)

    def _append_window(
        self,
        session: _AdaptiveSession,
        boundary: AdaptiveDecodeWindowBoundary,
        observation: AdaptiveDecodeRawWindowObservation,
        *,
        stable_window: bool = True,
        external_activity_changed: bool = False,
        ineligible_reason: str | None = None,
    ) -> AdaptiveDecodeWindowReceipt:
        return _windows._append_window(
            self,
            session,
            boundary,
            observation,
            stable_window=stable_window,
            external_activity_changed=external_activity_changed,
            ineligible_reason=ineligible_reason,
        )

    def record_window(
        self,
        request_id: str,
        boundary: AdaptiveDecodeWindowBoundary,
        observation: AdaptiveDecodeRawWindowObservation,
        *,
        compatible_batch_change: bool = False,
    ) -> AdaptiveDecodeDirective:
        return _windows.record_window(
            self, request_id, boundary, observation,
            compatible_batch_change=compatible_batch_change,
        )

    def discard_stale_window(
        self,
        request_id: str,
        boundary: AdaptiveDecodeWindowBoundary,
        observation: AdaptiveDecodeRawWindowObservation,
        reason: str,
        *,
        terminal_token_index: int | None = None,
        terminal_at_us: int | None = None,
    ) -> AdaptiveDecodeDirective:
        """Seal a released-slot tail without treating it as cost evidence."""
        return _windows.discard_stale_window(
            self,
            request_id,
            boundary,
            observation,
            reason,
            terminal_token_index=terminal_token_index,
            terminal_at_us=terminal_at_us,
        )

    def _seal_released_phone_tail(
        self, session, boundary, observation, terminal_token_index, terminal_at_us,
    ) -> AdaptiveDecodeDirective:
        return _completion._seal_released_phone_tail(
            self,
            session,
            boundary,
            observation,
            terminal_token_index,
            terminal_at_us,
        )

    def seal_tail(
        self,
        request_id: str,
        *,
        slot_id: int,
        token_index: int,
        reason: str,
    ) -> None:
        return _completion.seal_tail(
            self,
            request_id,
            slot_id=slot_id,
            token_index=token_index,
            reason=reason,
        )

    def control_failed(
        self,
        request_id: str,
        control: AdaptiveDecodeControl,
        reason: str,
        *,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        return _completion.control_failed(self, request_id, control, reason, at_us=at_us)

    def defer_control(
        self,
        request_id: str,
        control: AdaptiveDecodeControl,
        reason: str,
        *,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        """Continue the current window and retry a transient helper control."""
        return _completion.defer_control(self, request_id, control, reason, at_us=at_us)

    def discard_stale_control(
        self,
        request_id: str,
        control: AdaptiveDecodeControl,
        reason: str,
        *,
        at_us: int,
    ) -> AdaptiveDecodeDirective:
        """Discard a control after its request has released the bound slot."""
        return _completion.discard_stale_control(self, request_id, control, reason, at_us=at_us)

    @staticmethod
    def _validate_terminal(
        session: _AdaptiveSession,
        terminal_status: str,
        terminal_reason: str | None,
    ) -> None:
        return _completion._validate_terminal(session, terminal_status, terminal_reason)

    @staticmethod
    def _terminal_group(
        session: _AdaptiveSession,
        terminal_status: str,
        terminal_reason: str | None,
        state_history: tuple[str, ...],
    ) -> AdaptiveDecodeGroupedObservation:
        return _completion._terminal_group(session, terminal_status, terminal_reason, state_history)

    def preview_completion(
        self,
        request_id: str,
        terminal_status: str = "COMPLETED",
        terminal_reason: str | None = None,
    ) -> AdaptiveDecodeGroupedObservation:
        """Build the immutable terminal proof without mutating lifecycle state."""
        return _completion.preview_completion(self, request_id, terminal_status, terminal_reason)

    def complete(
        self,
        request_id: str,
        terminal_status: str,
        terminal_reason: str | None = None,
    ) -> AdaptiveDecodeGroupedObservation:
        return _completion.complete(self, request_id, terminal_status, terminal_reason)

    def recover_for_restart(
        self, request_id: str, reason: str
    ) -> AdaptiveDecodeGroupedObservation | None:
        """Close an interrupted adaptive attempt before desktop restart."""
        return _completion.recover_for_restart(self, request_id, reason)

    def recover_attempt_for_restart(
        self, request_id: str, ticket_id: str, reason: str
    ) -> AdaptiveDecodeGroupedObservation | None:
        """Close the adaptive session of one failed attempt, whatever its mode (elastic phones)."""
        return _completion.recover_attempt_for_restart(self, request_id, ticket_id, reason)

    def release_restarted_attempt(
        self, request_id: str, stale_ticket_ids: tuple[str, ...], reason: str
    ) -> dict[str, object] | None:
        """Free a failed earlier attempt's registration before a restart (elastic phones)."""
        return _completion.release_restarted_attempt(
            self, request_id, stale_ticket_ids, reason
        )

    def grouped_observation(
        self, request_id: str
    ) -> AdaptiveDecodeGroupedObservation:
        return _reporting.grouped_observation(self, request_id)

    def timing(self, request_id: str) -> Mapping[str, int]:
        return _reporting.timing(self, request_id)

    def snapshot(self, request_id: str) -> Mapping[str, object]:
        return _reporting.snapshot(self, request_id)

    def assistance_summaries(self) -> tuple[Mapping[str, object], ...]:
        return _reporting.assistance_summaries(self)
