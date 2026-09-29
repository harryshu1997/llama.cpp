"""AdaptiveDecodeController history operations on its existing owner."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from ..adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
    AdaptiveDecodePolicy,
    AdaptiveDecodeWindowReceipt,
    validate_policy_set,
)
from ..types import canonical_sha256
from ..adaptive_decode_state import _AdaptiveSession
from .common import ADAPTIVE_OBSERVATION_STORE_SCHEMA


def observation_snapshot(controller) -> Mapping[str, object]:
    with controller._lock:
        body = {
            "component_bindings": [
                {
                    "component_capability_sha256": target,
                    "grouped_observation_sha256": group_sha256,
                    "source_component_capability_sha256": source,
                }
                for group_sha256, (source, target) in sorted(
                    controller._history_component_bindings.items()
                )
            ],
            "groups": [
                controller._history[group_sha256].to_json()
                for group_sha256 in sorted(controller._history)
            ],
            "schema": ADAPTIVE_OBSERVATION_STORE_SCHEMA,
        }
        return MappingProxyType({
            **body,
            "store_sha256": canonical_sha256(body),
        })


def load_observations(
    controller, value: object, *, merge: bool = False
) -> None:
    if type(value) is not dict:
        raise AdaptiveDecodeError(
            "adaptive observation store is invalid"
        )
    body = dict(value)
    observed_hash = body.pop("store_sha256", None)
    groups = body.get("groups")
    binding_values = body.get("component_bindings", [])
    if (
        body.get("schema") != ADAPTIVE_OBSERVATION_STORE_SCHEMA
        or type(groups) is not list
        or type(binding_values) is not list
        or observed_hash != canonical_sha256(body)
    ):
        raise AdaptiveDecodeError(
            "adaptive observation store identity differs"
        )
    parsed = tuple(
        AdaptiveDecodeGroupedObservation.from_json(row)
        for row in groups
    )
    by_hash = {
        row.grouped_observation_sha256: row for row in parsed
    }
    if len(by_hash) != len(parsed):
        raise AdaptiveDecodeError(
            "adaptive observation groups are duplicated"
        )
    bindings: dict[str, tuple[str, str]] = {}
    for value in binding_values:
        if type(value) is not dict:
            raise AdaptiveDecodeError(
                "adaptive component binding is invalid"
            )
        group_sha256 = value.get("grouped_observation_sha256")
        source = value.get("source_component_capability_sha256")
        target = value.get("component_capability_sha256")
        if (
            group_sha256 not in by_hash
            or group_sha256 in bindings
            or not controller._valid_sha256(source)
            or not controller._valid_sha256(target)
        ):
            raise AdaptiveDecodeError(
                "adaptive component binding identity differs"
            )
        bindings[group_sha256] = (source, target)
    with controller._lock:
        if controller._sessions or controller._sealed_sessions:
            raise AdaptiveDecodeError(
                "adaptive observations cannot change while active"
            )
        if not merge:
            controller._history = by_hash
            controller._history_component_bindings = bindings
            return
        for group_sha256, grouped in by_hash.items():
            current = controller._history.get(group_sha256)
            if current is not None and current != grouped:
                raise AdaptiveDecodeError(
                    "adaptive observation hash collision"
                )
            controller._history[group_sha256] = grouped
        for group_sha256, binding in bindings.items():
            current = controller._history_component_bindings.get(group_sha256)
            if current is not None and current != binding:
                raise AdaptiveDecodeError(
                    "adaptive component binding conflicts"
                )
            controller._history_component_bindings[group_sha256] = binding


def observation_state(controller) -> Mapping[str, int]:
    with controller._lock:
        return MappingProxyType({
            "completed_requests": len(controller._completed),
            "grouped_observations": len(controller._history),
            "valid_windows": sum(
                row.output_valid and row.failure_reason is None
                and row.measurement_eligible
                for grouped in controller._history.values()
                for row in grouped.windows
            ),
            "energy_valid_windows": sum(
                row.output_valid and row.failure_reason is None
                and row.energy_measurement_eligible
                for grouped in controller._history.values()
                for row in grouped.windows
            ),
            "component_bound_groups": len(
                controller._history_component_bindings
            ),
        })


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and value.startswith("sha256:")
        and len(value) == 71
        and all(
            character in "0123456789abcdef"
            for character in value[7:]
        )
    )


def rebind_legacy_component_observations(
    controller,
    *,
    model_artifact_sha256: str,
    source_component_capability_sha256: str,
    component_capability_sha256: str,
    source_planning_profile_sha256s: Sequence[str],
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
) -> int:
    """Bind verified legacy groups to one stable execution component."""
    rows = validate_policy_set(baseline, candidates)
    profiles = frozenset(source_planning_profile_sha256s)
    if (
        not controller._valid_sha256(model_artifact_sha256)
        or not controller._valid_sha256(
            source_component_capability_sha256
        )
        or not controller._valid_sha256(component_capability_sha256)
        or not profiles
        or any(not controller._valid_sha256(value) for value in profiles)
    ):
        raise AdaptiveDecodeError(
            "adaptive component rebind identity is invalid"
        )
    identities = {
        controller._policy_identity(policy)
        for policy in (baseline, *rows)
    }
    baseline_identity = controller._policy_identity(baseline)
    rebound = 0
    with controller._lock:
        if controller._sessions or controller._sealed_sessions:
            raise AdaptiveDecodeError(
                "adaptive evidence cannot rebind while active"
            )
        for group_sha256, grouped in controller._history.items():
            if group_sha256 in controller._history_component_bindings:
                continue
            group_identities = {
                controller._policy_identity(window.policy)
                for window in grouped.windows
            }
            if (
                grouped.terminal_status != "COMPLETED"
                or grouped.model_artifact_sha256
                    != model_artifact_sha256
                or grouped.planning_profile_sha256 not in profiles
                or grouped.desktop_placement_sha256
                    != baseline.desktop_placement_sha256
                or controller._policy_identity(grouped.final_policy)
                    not in identities
                or baseline_identity not in group_identities
                or not group_identities.issubset(identities)
            ):
                continue
            controller._history_component_bindings[group_sha256] = (
                source_component_capability_sha256,
                component_capability_sha256,
            )
            rebound += 1
    if not rebound:
        raise AdaptiveDecodeError(
            "legacy adaptive component observations are absent"
        )
    return rebound


def _policy_identity(policy: AdaptiveDecodePolicy) -> str:
    return canonical_sha256({
        "columns": policy.columns,
        "desktop_placement_sha256": (
            policy.desktop_placement_sha256
        ),
        "executor_id": policy.executor_id,
        "layer_mask": policy.layer_mask,
        "resource_ids": list(policy.resource_ids),
        "split_fraction_ppm": policy.split_fraction_ppm,
    })


def _policy_geometry_identity(policy: AdaptiveDecodePolicy) -> str:
    return canonical_sha256({
        "baseline": policy.baseline,
        "columns": policy.columns,
        "desktop_placement_sha256": (
            policy.desktop_placement_sha256
        ),
        "layer_mask": policy.layer_mask,
        "split_fraction_ppm": policy.split_fraction_ppm,
    })


def _group_matches_session(
    controller,
    grouped: AdaptiveDecodeGroupedObservation,
    session: _AdaptiveSession,
) -> bool:
    binding = controller._history_component_bindings.get(
        grouped.grouped_observation_sha256
    )
    component_matches = (
        binding[1] == session.component_capability_sha256
        if binding is not None
        else grouped.planning_profile_sha256
            == session.planning_profile_sha256
    )
    layout_matches = (
        session.helper_layout_geometry_sha256 is None
        or grouped.helper_layout_geometry_sha256 is None
        or grouped.helper_layout_geometry_sha256
            == session.helper_layout_geometry_sha256
    )
    return (
        grouped.terminal_status == "COMPLETED"
        and grouped.model_artifact_sha256
            == session.model_artifact_sha256
        and component_matches
        and layout_matches
        and grouped.desktop_placement_sha256
            == session.baseline.desktop_placement_sha256
    )


def _compatible_context_lengths(
    controller, session: _AdaptiveSession
) -> tuple[int, ...]:
    requested_bucket = session.context_length.bit_length()
    buckets = {
        window.context_length.bit_length()
        for grouped in controller._history.values()
        if controller._group_matches_session(grouped, session)
        for window in grouped.windows
        if window.output_valid
        and window.failure_reason is None
        and window.measurement_eligible
        and window.active_batch == session.active_batch
    }
    ordered = sorted(
        buckets - {requested_bucket},
        key=lambda bucket: (
            abs(bucket - requested_bucket),
            bucket < requested_bucket,
            bucket,
        ),
    )
    return (session.context_length,) + tuple(
        1 << (bucket - 1) for bucket in ordered
    )


def _historical_policy_records(
    controller,
    session: _AdaptiveSession,
    policy: AdaptiveDecodePolicy,
    *,
    compatible_context: bool = False,
) -> tuple[tuple[AdaptiveDecodeWindowReceipt, ...], int]:
    identity = controller._policy_identity(policy)
    context_lengths = (
        controller._compatible_context_lengths(session)
        if compatible_context else (session.context_length,)
    )
    for context_length in context_lengths:
        records = []
        groups = 0
        context_bucket = context_length.bit_length()
        for grouped in controller._history.values():
            if not controller._group_matches_session(grouped, session):
                continue
            selected = tuple(
                row for row in grouped.windows
                if row.output_valid
                and row.failure_reason is None
                and row.measurement_eligible
                and row.active_batch == session.active_batch
                and row.context_length.bit_length() == context_bucket
                and controller._policy_identity(row.policy) == identity
            )
            if selected:
                records.extend(selected)
                if any(
                    row.energy_measurement_eligible
                    for row in selected
                ):
                    groups += 1
        if records:
            return tuple(records), groups
    return (), 0
