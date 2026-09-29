"""Decode request state and reusable evidence records: state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .adaptive_decode_contracts import (
    AdaptiveDecodeConfig,
    AdaptiveDecodeControl,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodePolicyAck,
    AdaptiveDecodeWindowBoundary,
    AdaptiveDecodeWindowReceipt,
)


@dataclass
class _AdaptiveServerPolicy:
    owner_request_id: str
    policy: AdaptiveDecodePolicy
    changed_at_us: int
    qualified: bool = False
    proposal: AdaptiveDecodePolicy | None = None
    probe_tokens: int = 0
    records: tuple[AdaptiveDecodeWindowReceipt, ...] = ()


@dataclass
class _AdaptiveSession:
    request_id: str
    ticket_id: str
    model_artifact_sha256: str
    planning_profile_sha256: str
    component_capability_sha256: str
    baseline: AdaptiveDecodePolicy
    candidates: tuple[AdaptiveDecodePolicy, ...]
    output_tokens: int
    context_length: int
    active_batch: int
    deadline_us: int
    config: AdaptiveDecodeConfig
    slot_id: int
    ticket_policy: AdaptiveDecodePolicy | None = None
    state: str = "BASELINE"
    state_history: list[str] = field(default_factory=lambda: ["BASELINE"])
    current_policy: AdaptiveDecodePolicy | None = None
    current_ack: AdaptiveDecodePolicyAck | None = None
    window_start_token: int | None = None
    window_start_us: int | None = None
    window_role: str = "exploration"
    target_token: int | None = None
    awaiting_boundary: AdaptiveDecodeWindowBoundary | None = None
    awaiting_control: AdaptiveDecodeControl | None = None
    transition_policy: AdaptiveDecodePolicy | None = None
    transition_ack: AdaptiveDecodePolicyAck | None = None
    transition_start_token: int | None = None
    transition_started_at_us: int | None = None
    transition_window_role: str = "exploration"
    records: list[AdaptiveDecodeWindowReceipt] = field(default_factory=list)
    probe_candidates: list[AdaptiveDecodePolicy] = field(default_factory=list)
    next_probe_index: int = 0
    stage: str = "initial_baseline"
    plan_generation: int = 0
    probe_tokens: int = 0
    exploration_overhead_high_water_uj: int = 0
    context_record_start: int = 0
    probe_budget: dict[str, int] | None = None
    probe_policy_hash: str | None = None
    observed_control_cost_us: int = 0
    observed_control_tokens: int = 0
    warmup_latency_us_by_policy: dict[str, int] = field(default_factory=dict)
    decision_time_us: list[int] = field(default_factory=list)
    grouped: AdaptiveDecodeGroupedObservation | None = None
    tail_seal_token: int | None = None
    tail_seal_reason: str | None = None
    tail_seal_pending_control: bool = False
    historical_records: dict[
        str, tuple[AdaptiveDecodeWindowReceipt, ...]
    ] = field(default_factory=dict)
    historical_group_counts: dict[str, int] = field(default_factory=dict)
    warmup_windows_seen_by_policy: dict[str, int] = field(
        default_factory=dict
    )
    eliminated_policy_reasons: dict[str, str] = field(
        default_factory=dict
    )
    refinement_added: bool = False
    cached_winner: AdaptiveDecodePolicy | None = None
    incumbent_policy: AdaptiveDecodePolicy | None = None
    incumbent_context_sha256: str | None = None
    challenger_policy: AdaptiveDecodePolicy | None = None
    acknowledged_policy: AdaptiveDecodePolicy | None = None
    probe_attempts: dict[str, int] = field(default_factory=dict)
    policy_evidence_aliases: dict[str, str] = field(default_factory=dict)
    evidence_context_component_sha256: str | None = None
    probe_retry_policy: AdaptiveDecodePolicy | None = None
    zero_assistance_reason: str = "INITIAL_BASELINE"
    verification_policy: AdaptiveDecodePolicy | None = None
    operational_verification: bool = False
    verification_budget: dict[str, int] | None = None
    qualification_measurement_plan: dict[str, object] | None = None
    verification_attempts: int = 0
    verification_record_start: int = 0
    verification_outcome: str | None = None
    verification_reason: str | None = None
    context_monitor_prior: dict[str, object] | None = None
    # A compatible prior winner exploited under per-window monitoring when
    # the request cannot afford a complete paired verification (starting
    # point, never inherited qualification).
    prior_monitor_policy: AdaptiveDecodePolicy | None = None
    prior_monitor_windows: int = 0
    helper_available: bool = True
    execution_context_available: bool = True
    helper_layout_generation: int | None = None
    helper_layout_geometry_sha256: str | None = None
    helper_ready_boundary_pending: bool = False
    helper_evidence_state: str = "TRUSTED"
    history_helper_layout_geometry_sha256: str | None = None
    deferred_policy: AdaptiveDecodePolicy | None = None
    deferred_control_reason: str | None = None
    deferred_control_count: int = 0
    pending_session_drain_policy: AdaptiveDecodePolicy | None = None
    pending_helper_refresh_policy: AdaptiveDecodePolicy | None = None
    maintenance_policy_hashes: set[str] = field(default_factory=set)
    phone_power_policy_from_capability: bool = False
    external_activity_sha256: str | None = None


@dataclass(frozen=True)
class _AssumedPhonePowerQuery:
    model_artifact_sha256: str
    planning_profile_sha256: str
    component_capability_sha256: str
    baseline: AdaptiveDecodePolicy
    rows: tuple[AdaptiveDecodePolicy, ...]
    active_batch: int
    config: AdaptiveDecodeConfig
    domains: Mapping[str, tuple[int, int]]
    minimum_group_count: int
    route_geometry_prior: bool
    operator_subset_prior: bool
    operator_subset_source_layer_mask: int | None


@dataclass(frozen=True)
class AdaptiveDecodeHistoricalEstimate:
    baseline_policy: AdaptiveDecodePolicy
    selected_policy: AdaptiveDecodePolicy
    baseline_energy_per_token_uj: int
    baseline_energy_lower_per_token_uj: int
    baseline_energy_upper_per_token_uj: int
    selected_energy_per_token_uj: int
    selected_energy_lower_per_token_uj: int
    selected_energy_upper_per_token_uj: int
    baseline_latency_per_token_us: int
    baseline_latency_upper_per_token_us: int
    selected_latency_per_token_us: int
    selected_latency_upper_per_token_us: int
    baseline_group_count: int
    selected_group_count: int
    energy_boundary_id: str
    requested_context_bucket: int
    evidence_context_bucket: int
    evidence_match: str = "component"
    source_layer_mask: int | None = None
    target_layer_mask: int | None = None
    evidence_scale_ppm: int = 1_000_000
