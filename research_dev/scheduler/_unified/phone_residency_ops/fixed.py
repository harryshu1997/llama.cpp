"""PhoneResidencyMixin fixed operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

from ...config import FixedPhoneResidencyConfiguration
from ..._internal.policy import Request
from ..._internal.lifecycle import UnifiedScheduleError
from ..automated_requests_ops.event_replanning import (
    event_replanning_enabled,
    note_device_admissible,
)


def configure_fixed_phone_residency(
    controller, configuration: FixedPhoneResidencyConfiguration,
) -> None:
    """Freeze an evaluation assignment before any layout or request exists."""
    if not isinstance(configuration, FixedPhoneResidencyConfiguration):
        raise UnifiedScheduleError("fixed residency configuration is invalid")
    if (
        controller._runtime_controller.current_tickets()
        or controller._model_placement_controller.planning_phone_layout() is not None
        or controller._active_offline_phone_residency_plan_id is not None
        or controller._fixed_phone_residency not in (None, configuration)
    ):
        raise UnifiedScheduleError("fixed residency configuration is already frozen")
    controller._fixed_phone_residency = configuration


def fixed_phone_residency_configuration(controller) -> Mapping[str, object] | None:
    fixed = controller._fixed_phone_residency
    return None if fixed is None else {
        **fixed.to_json(), "assignment_sha256": fixed.assignment_sha256,
    }


def fixed_phone_residency_requests(controller) -> Mapping[str, tuple[Request, ...]]:
    """Use preparation-only shapes, never future trace requests or arrivals."""
    fixed = controller._fixed_phone_residency
    if fixed is None:
        raise UnifiedScheduleError("fixed residency is not configured")
    requests = {}
    for artifact in sorted({row[1] for row in fixed.assignments}):
        models = [row for row in controller._runtime_manifests.values()
                  if row.artifact_sha256 == artifact]
        if len(models) != 1:
            raise UnifiedScheduleError("fixed residency artifact is not registered exactly once")
        model = models[0]
        requests[model.model_id] = (Request(
            request_id="fixed-preparation:" + artifact[7:23],
            workload_id="fixed-residency-preparation", arrival_us=0,
            deadline_us=86_400_000_000, input_tokens=1,
            output_tokens=max(64, controller._adaptive_decode_config.minimum_remaining_tokens * 4),
            quality_requirement="semantic",
        ),)
    return MappingProxyType(requests)


def _fixed_phone_inputs(controller, discovery, sessions):
    fixed = controller._fixed_phone_residency
    if fixed is None:
        return discovery.demand_rows, sessions
    assignment = {row[0]: row for row in fixed.assignments}
    if not set(assignment).issubset(row.session_id for row in sessions):
        raise UnifiedScheduleError("fixed residency session is unavailable")
    demands = []
    for artifact in sorted({row[1] for row in fixed.assignments}):
        source = next((row for row in discovery.demand_rows
                       if row.manifest.artifact_sha256 == artifact), None)
        control = controller._runtime_capabilities.desktop_control_by_artifact.get(artifact)
        if source is None or control is None:
            raise UnifiedScheduleError("fixed residency artifact has no compatible desktop parent")
        mask = sum(row[2] for row in fixed.assignments if row[1] == artifact)
        columns = {row[3] for row in fixed.assignments if row[1] == artifact}
        if len(columns) != 1:
            raise UnifiedScheduleError("fixed residency artifact widths differ")
        placements = {row.operator_id: row for row in control.operator_placements}
        operators = tuple(
            row.operator_id for row in source.manifest.operators
            if row.kind == "ffn" and mask & (1 << int(row.layer_id.split(":")[1]))
        )
        covered = sum(1 << int(source.manifest.operator_by_id[key].layer_id.split(":")[1])
                      for key in operators)
        if covered != mask or any(
            controller._runtime_capabilities.placement_profile.devices[
                placements[key].primary_device_id
            ].kind != "cpu" for key in operators
        ):
            raise UnifiedScheduleError("fixed residency layers are not desktop CPU FFNs")
        demands.append(replace(
            source, queued_work=1, allowed_operator_ids=operators,
            maximum_columns=next(iter(columns)), benefit_by_operator={},
            benefit_value_kind="rough_compute_ops",
        ))
    restricted = tuple(
        replace(row, supported_layer_mask=row.supported_layer_mask & assignment[row.session_id][2])
        for row in sessions if row.session_id in assignment
    )
    return tuple(demands), restricted


def _fixed_phone_layout_matches(controller, layout, *, partial=False):
    actual = tuple(sorted(
        (row.session_id, row.artifact_sha256, row.layer_mask, row.maximum_columns)
        for row in layout.shards
    ))
    expected = controller._fixed_phone_residency.assignments
    return set(actual).issubset(expected) if partial else actual == expected


def _phone_telemetry_deferral(controller, snapshot, observed_at_us, request_id):
    if (
        snapshot is None or controller._runtime_capabilities is None
        or not any(row.phone_sessions for row in controller._runtime_capabilities.executors)
        or not snapshot.telemetry_observations
    ):
        return None
    pending = controller._phone_telemetry_deferrals
    for capability in controller._runtime_capabilities.executors:
        if not capability.phone_sessions:
            continue
        reason = snapshot.telemetry_unavailable_reason(
            capability.device_id, observed_at_us
        )
        if reason is None:
            continue
        if request_id not in pending:
            pending[request_id] = observed_at_us
            controller._model_placement_controller.record_request_helper_event(
                request_id, "PHONE_TELEMETRY_DEFERRED", observed_at_us,
                {"reason": reason, "snapshot_id": snapshot.snapshot_id,
                 "refresh_required": True,
                 "observation": dict(snapshot.telemetry_observations[capability.device_id])},
            )
        return MappingProxyType({
            "status": "DEFERRED", "reason": "PHONE_TELEMETRY_UNAVAILABLE",
            "detail": reason, "refresh_required": True,
        })
    if request_id in pending:
        started_at_us = pending.pop(request_id)
        controller._model_placement_controller.record_request_helper_event(
            request_id, "PHONE_TELEMETRY_RECOVERED", observed_at_us,
            {"deferred_at_us": started_at_us,
             "outage_duration_us": observed_at_us - started_at_us,
             "snapshot_id": snapshot.snapshot_id},
        )
        for capability in (
            controller._runtime_capabilities.executors
            if event_replanning_enabled(controller) else ()
        ):
            if capability.phone_sessions:
                note_device_admissible(
                    controller, capability.device_id, observed_at_us,
                    "PHONE_TELEMETRY_RECOVERED", started_at_us,
                )
    return None
