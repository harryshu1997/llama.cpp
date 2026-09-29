"""Shared records and helpers; no independent controller state."""

from __future__ import annotations

from math import isqrt
from typing import Sequence

from ..adaptive_decode_contracts import AdaptiveDecodePolicy, AdaptiveDecodeWindowReceipt
from ..adaptive_decode_state import _AdaptiveSession, _AssumedPhonePowerQuery


ADAPTIVE_OBSERVATION_STORE_SCHEMA = "adaptive-decode-observations-v1"


HELPER_EVIDENCE_STATES = frozenset({"LEARNING", "TRUSTED"})


def _assumed_phone_power_session(
    query: _AssumedPhonePowerQuery,
    request_id: str,
    context_length: int,
) -> _AdaptiveSession:
    return _AdaptiveSession(
        request_id=request_id,
        ticket_id=request_id + ":attempt:0",
        model_artifact_sha256=query.model_artifact_sha256,
        planning_profile_sha256=query.planning_profile_sha256,
        component_capability_sha256=query.component_capability_sha256,
        baseline=query.baseline,
        candidates=query.rows,
        output_tokens=1,
        context_length=context_length,
        active_batch=query.active_batch,
        deadline_us=1,
        config=query.config,
        slot_id=0,
    )


def _window_has_phone_work(window: AdaptiveDecodeWindowReceipt) -> bool:
    return (
        (window.completed_phone_calls or 0) > 0
        or window.phone_compute_us > 0
        or window.usb_transfer_us > 0
        or window.usb_upload_bytes > 0
        or window.usb_download_bytes > 0
    )


def _assumed_phone_power_bounds(
    query: _AssumedPhonePowerQuery,
    records: Sequence[AdaptiveDecodeWindowReceipt],
    group_count: int,
) -> tuple[int, int, int, int, int] | None:
    domains = query.domains
    if not records or group_count < query.minimum_group_count:
        return None
    energy_uj = 0
    energy_tokens = 0
    latency_tokens = 0
    latency_us = 0
    for window in records:
        if any(
            domain_id not in window.fleet_energy_uj_by_domain
            for domain_id in domains
        ):
            return None
        duration_us = window.finished_at_us - window.started_at_us
        work_us = 0
        if _window_has_phone_work(window):
            work_us = min(
                duration_us,
                window.phone_compute_us
                + window.usb_transfer_us
                + window.rpc_us,
            )
            if work_us <= 0:
                return None
        nonphone_uj = sum(
            value
            for domain_id, value in (
                window.fleet_energy_uj_by_domain.items()
            )
            if domain_id not in domains
        )
        assumed_phone_uj = sum(
            (
                active_mw * work_us
                + idle_mw * (duration_us - work_us)
                + 999
            ) // 1_000
            for active_mw, idle_mw in domains.values()
        )
        energy_uj += nonphone_uj + assumed_phone_uj
        energy_tokens += window.energy_token_count
        latency_tokens += window.token_count
        latency_us += duration_us
    if energy_tokens <= 0 or latency_tokens <= 0:
        return None
    energy_mean = max(1, energy_uj // energy_tokens)
    latency_mean = max(1, latency_us // latency_tokens)
    uncertainty_ppm = max(
        20_000,
        query.config.uncertainty_ppm
            // max(1, isqrt(group_count)),
    )
    energy_lower = max(
        1,
        energy_mean * (1_000_000 - uncertainty_ppm) // 1_000_000,
    )
    energy_upper = (
        energy_mean * (1_000_000 + uncertainty_ppm) + 999_999
    ) // 1_000_000
    latency_upper = (
        latency_mean * (1_000_000 + uncertainty_ppm) + 999_999
    ) // 1_000_000
    return (
        energy_mean,
        energy_lower,
        energy_upper,
        latency_mean,
        latency_upper,
    )


def _operator_subset_bounds(
    baseline_values: tuple[int, int, int, int, int],
    source_values: tuple[int, int, int, int, int],
    target: AdaptiveDecodePolicy,
    source: AdaptiveDecodePolicy,
) -> tuple[int, int, int, int, int] | None:
    target_layers = target.layer_mask.bit_count()
    source_layers = source.layer_mask.bit_count()
    if (
        target.baseline
        or source.baseline
        or target_layers <= 0
        or target_layers >= source_layers
        or source.layer_mask & target.layer_mask
            != target.layer_mask
    ):
        return source_values
    saving_lower = (
        baseline_values[1] - source_values[2]
    )
    if saving_lower <= 0:
        return None
    saving_mean = max(
        saving_lower,
        baseline_values[0] - source_values[0],
    )
    saving_upper = max(
        saving_mean,
        baseline_values[2] - source_values[1],
    )
    scaled_lower = saving_lower * target_layers // source_layers
    scaled_mean = saving_mean * target_layers // source_layers
    scaled_upper = saving_upper * target_layers // source_layers
    if scaled_lower <= 0:
        return None
    energy_upper = max(1, baseline_values[1] - scaled_lower)
    energy_mean = max(1, baseline_values[0] - scaled_mean)
    energy_lower = max(1, baseline_values[1] - scaled_upper)
    energy_mean = min(energy_upper, max(energy_lower, energy_mean))
    energy_lower = min(energy_lower, energy_mean)
    latency_mean = max(
        baseline_values[3], source_values[3]
    )
    latency_upper = max(
        latency_mean,
        baseline_values[4],
        source_values[4],
    )
    return (
        energy_mean,
        energy_lower,
        energy_upper,
        latency_mean,
        latency_upper,
    )
