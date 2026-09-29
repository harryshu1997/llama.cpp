"""Transitions, memory demands, eligibility, transport parameters, resources, participants.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from ..runtime_resources import runtime_phase_lease_demands

from dataclasses import replace
import threading
from types import MappingProxyType
from typing import Mapping, Sequence
from ..model_manifest import ModelManifest, ModelRequestWork
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
)
from ..runtime_cost import RuntimeMemoryDemand, RuntimeParticipantBinding
from ..runtime_plan import (
    AutomatedRouteCandidate,
    RuntimeResidencyEviction,
    RuntimeTransitionPlan,
)
from ..runtime_resources import RuntimeResourceError, preview_runtime_memory
from .remote_resident import RouteRemoteResidentMixin
from .common import (
    CO_HELPER_UNAVAILABLE,
    RouteGenerationError,
    _ceil_div,
    _Pattern,
    unavailable_co_helpers,
)


def resident_phone_workspace(manifest, capability, parameters, device_id, current):
    """Account for the provisioned worker batch even for a decode-only request."""
    if parameters.get("phone_device_id") != device_id:
        return current
    tokens = parameters.get("ffn_max_tokens", 1)
    if type(tokens) is not int or tokens < 1:
        raise RouteGenerationError("resident FFN batch capacity is invalid")
    if tokens == 1:
        return current
    work = manifest.request_work(tokens, 1)
    return max(current, max(row.workspace_bytes for row in work.operators
                            if manifest.operator_by_id[row.operator_id].kind == "ffn")
               + capability.workspace_bytes_per_token * tokens)


class ThermalGateLog:
    """One THERMAL_DEFERRAL row per onset of a device's thermal gate.

    Route feasibility excludes every route through a device (THERMAL_LIMIT)
    while the gate holds; the first exclusion opens an onset and a later
    observation that passes the gate closes it (THERMAL_DEFERRAL_CLEARED).
    Observations older than the device's last one are ignored, so a replan
    at an earlier time cannot reopen or close it. Under an opt-in
    ``maximum_thermal_status`` policy the row also carries that limit and the
    observed raw Android thermal status.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[Mapping[str, object]] = []
        self._limited_since_us: dict[str, int] = {}
        self._observed_at_us: dict[str, int] = {}

    def observe(self, capability, state, at_us: int, limited: bool) -> None:
        device_id = capability.device_id
        with self._lock:
            if at_us < self._observed_at_us.get(device_id, 0):
                return
            self._observed_at_us[device_id] = at_us
            onset_us = self._limited_since_us.get(device_id)
            if limited == (onset_us is not None):
                return
            row = {
                "at_us": at_us,
                "device_id": device_id,
                "executor_id": capability.executor_id,
                "kind": (
                    "THERMAL_DEFERRAL" if limited
                    else "THERMAL_DEFERRAL_CLEARED"
                ),
                "maximum_temperature_millic": (
                    capability.maximum_temperature_millic
                ),
                "observed_temperature_millic": state.temperature_millic,
                "thermal_qualified": state.thermal_qualified,
            }
            if (
                capability.maximum_thermal_status
                or state.thermal_status is not None
            ):
                row["maximum_thermal_status"] = capability.maximum_thermal_status
                row["observed_thermal_status"] = state.thermal_status
            if limited:
                self._limited_since_us[device_id] = at_us
            else:
                del self._limited_since_us[device_id]
                row["onset_at_us"] = onset_us
            self._events.append(MappingProxyType(row))

    def events(self) -> tuple[Mapping[str, object], ...]:
        with self._lock:
            return tuple(self._events)


class RouteFeasibilityMixin:
    """Transitions, memory demands, eligibility, transport parameters, resources, participants."""

    def _transitions(
        self,
        device_ids: tuple[str, ...],
        residency_states: Mapping[str, str],
        weights_by_device: Mapping[str, int],
        artifact_sha256: str,
        executor_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[RuntimeTransitionPlan, ...]:
        matches = tuple(
            row for row in self.catalog.transitions
            if row.device_id in device_ids
            and row.source_state == residency_states[row.device_id]
            and row.target_state == "hot"
            and row.artifact_sha256 in {None, artifact_sha256}
            and row.executor_id in {None, executor_id}
            and set(row.prepares_device_ids).issubset(device_ids)
            and (
                any(
                    residency_states[device_id] != "hot"
                    for device_id in row.prepares_device_ids
                )
                or (
                    row.executor_id is not None
                    and (
                        snapshot.executors.get(row.executor_id) is None
                        or not snapshot.executors[row.executor_id].ready
                    )
                )
            )
        )
        rows = []
        covered: set[str] = set()
        for capability in sorted(
            matches,
            key=lambda row: (
                row.artifact_sha256 is None,
                row.executor_id is None,
                -len(row.prepares_device_ids),
                row.transition_id,
            ),
        ):
            needed = {
                device_id for device_id in capability.prepares_device_ids
                if residency_states[device_id] != "hot"
            }
            publication_needed = (
                capability.executor_id is not None
                and (
                    snapshot.executors.get(capability.executor_id) is None
                    or not snapshot.executors[
                        capability.executor_id
                    ].ready
                )
            )
            coverage_key = set(needed)
            if publication_needed:
                coverage_key.add("executor:" + capability.executor_id)
            if not coverage_key or coverage_key.issubset(covered):
                continue
            amount_bytes = sum(
                weights_by_device[device_id]
                for device_id in capability.prepares_device_ids
            )
            latency, energy = capability.cost(amount_bytes)
            replacement_group_by_device = {
                device_id: executor.exclusive_residency_resource_id
                for device_id, executor in (
                    self.catalog.executor_by_device.items()
                )
                if executor.exclusive_residency_resource_id is not None
            }
            coordinator = self.catalog.composite_executor_by_id.get(
                executor_id
            )
            if coordinator is not None:
                replacement_group_by_device.update(
                    coordinator.replacement_group_by_device
                )
            exclusive_by_device = {
                device_id: executor.exclusive_residency_resource_id
                for device_id, executor in (
                    self.catalog.executor_by_device.items()
                )
            }
            exclusive_anchor_identities = {
                (
                    row.model_id,
                    row.artifact_sha256,
                    row.generation,
                    row.executor_id,
                    exclusive_by_device[row.device_id],
                )
                for row in snapshot.residency
                if row.device_id in capability.prepares_device_ids
                and exclusive_by_device[row.device_id] is not None
                and self.catalog.residency_group(row.executor_id, row.device_id)
                    == exclusive_by_device[row.device_id]
                and row.state in {"hot", "warm"}
                and row.resident_bytes > 0
            }
            evictions = tuple(
                RuntimeResidencyEviction(
                    model_id=row.model_id,
                    artifact_sha256=row.artifact_sha256,
                    device_id=row.device_id,
                    resident_bytes=row.resident_bytes,
                    generation=row.generation,
                    executor_id=row.executor_id,
                    reclaimable_bytes=row.reclaimable_bytes,
                    replacement_group=(
                        replacement_group_by_device.get(row.device_id)
                    ),
                )
                for row in snapshot.residency
                if row.device_id in capability.prepares_device_ids
                and (
                    row.artifact_sha256 != artifact_sha256
                    or (
                        row.executor_id is not None
                        and row.executor_id != executor_id
                    )
                )
                and row.state in {"hot", "warm"}
                and row.resident_bytes > 0
                and replacement_group_by_device.get(row.device_id)
                    is not None
                and (
                    self.catalog.residency_group(row.executor_id, row.device_id)
                        == replacement_group_by_device[row.device_id]
                    or (
                        row.executor_id is not None
                        and (
                            row.model_id,
                            row.artifact_sha256,
                            row.generation,
                            row.executor_id,
                            replacement_group_by_device[row.device_id],
                        ) in exclusive_anchor_identities
                    )
                )
            )
            rows.append(RuntimeTransitionPlan(
                transition_id=capability.transition_id,
                device_id=capability.device_id,
                source_state=capability.source_state,
                target_state="hot",
                latency_us=latency,
                energy_uj=energy,
                resource_ids=capability.resource_ids,
                maturity=capability.maturity,
                resource_slots=capability.resource_slots,
                evictions=evictions,
                executor_id=capability.executor_id,
                prepares_device_ids=capability.prepares_device_ids,
                energy_maturity=capability.energy_maturity,
            ))
            covered.update(coverage_key)
        return tuple(rows)

    def _memory_demands(
        self,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        residency_states: Mapping[str, str],
    ) -> tuple[RuntimeMemoryDemand, ...]:
        cache_key = (
            manifest.artifact_sha256,
            work.input_tokens,
            work.output_tokens,
            pattern.route_key,
        )
        cached = self._memory_demand_cache.get(cache_key)
        if cached is None:
            tensor_by_id = manifest.tensor_by_id
            operator_by_id = {
                row.operator_id: row for row in manifest.operators
            }
            # remote-resident FFN weights are never allocated on the desktop
            omitted_tensor_ids = RouteRemoteResidentMixin._remote_resident_tensor_ids(
                RouteRemoteResidentMixin._remote_resident_group(self, manifest, pattern)
            )
            weights = {device: 0 for device in pattern.device_ids}
            kv = {device: 0 for device in pattern.device_ids}
            workspace = {device: 0 for device in pattern.device_ids}
            decode_by_id = (
                self._request_phase(work, "decode").by_operator_id
                if pattern.assistance_phase == "decode" else {}
            )
            for operator_work in work.operators:
                primary, helper, fraction = pattern.assignments[
                    operator_work.operator_id
                ]
                tensor_bytes = sum(
                    tensor_by_id[value].nbytes
                    for value in operator_by_id[
                        operator_work.operator_id
                    ].tensor_ids
                    if value not in omitted_tensor_ids
                )
                if pattern.assistance_phase == "decode":
                    desktop = pattern.desktop_assignments[
                        operator_work.operator_id
                    ]
                    weights[desktop] += tensor_bytes
                    kv[desktop] += operator_work.kv_cache_bytes
                    workspace[desktop] = max(
                        workspace[desktop], operator_work.workspace_bytes
                    )
                    if helper is not None:
                        weights[helper] += self._fraction(
                            tensor_bytes, fraction
                        )[1]
                        helper_device = helper
                    elif primary != desktop:
                        weights[primary] += tensor_bytes
                        helper_device = primary
                    else:
                        helper_device = None
                    if helper_device is not None:
                        decode_work = decode_by_id[
                            operator_work.operator_id
                        ]
                        workspace[helper_device] = max(
                            workspace[helper_device],
                            max(
                                1,
                                decode_work.workspace_bytes
                                // work.output_tokens,
                            ),
                        )
                elif helper is None:
                    weights[primary] += tensor_bytes
                    kv[primary] += operator_work.kv_cache_bytes
                    workspace[primary] = max(
                        workspace[primary], operator_work.workspace_bytes
                    )
                else:
                    base_bytes, helper_bytes = self._fraction(
                        tensor_bytes, fraction
                    )
                    weights[primary] += base_bytes
                    weights[helper] += helper_bytes
                    base_kv, helper_kv = self._fraction(
                        operator_work.kv_cache_bytes, fraction
                    )
                    kv[primary] += base_kv
                    kv[helper] += helper_kv
                    workspace[primary] = max(
                        workspace[primary], operator_work.workspace_bytes
                    )
                    workspace[helper] = max(
                        workspace[helper], operator_work.workspace_bytes
                    )
            replacement_group_by_device = {}
            if pattern.coordinator_executor_id is not None:
                coordinator = self.catalog.composite_executor_by_id.get(
                    pattern.coordinator_executor_id
                )
                if coordinator is not None:
                    replacement_group_by_device.update(
                        coordinator.replacement_group_by_device
                    )
            demands = []
            for device_id in pattern.device_ids:
                capability = self.catalog.executor_by_device[device_id]
                if weights[device_id]:
                    demands.append(RuntimeMemoryDemand(
                        demand_id=f"weights:{device_id}",
                        resource_id=capability.memory_resource_id,
                        kind="model_weights",
                        required_bytes=weights[device_id],
                        resident_bytes=0,
                        lifetime="resident",
                        share_key=(
                            manifest.artifact_sha256 + ":" + device_id
                        ),
                        replacement_group=(
                            replacement_group_by_device.get(
                                device_id,
                                capability.exclusive_residency_resource_id,
                            )
                        ),
                        device_id=device_id,
                    ))
                if kv[device_id]:
                    demands.append(RuntimeMemoryDemand(
                        demand_id=f"kv:{device_id}",
                        resource_id=capability.memory_resource_id,
                        kind="kv_cache",
                        required_bytes=kv[device_id],
                        resident_bytes=0,
                        lifetime="request",
                        device_id=device_id,
                    ))
                request_workspace = max(
                    1,
                    workspace[device_id]
                    + capability.workspace_bytes_per_token
                    * (work.input_tokens + work.output_tokens),
                )
                coordinator = self.catalog.composite_executor_by_id.get(
                    pattern.coordinator_executor_id
                )
                parameters = {} if coordinator is None else coordinator.adapter_parameters
                request_workspace = resident_phone_workspace(
                    manifest, capability, parameters, device_id, request_workspace,
                )
                demands.append(RuntimeMemoryDemand(
                    demand_id=f"workspace:{device_id}",
                    resource_id=capability.memory_resource_id,
                    kind="workspace",
                    required_bytes=request_workspace,
                    resident_bytes=0,
                    lifetime="request",
                    device_id=device_id,
                ))
            cached = tuple(demands)
            if len(self._memory_demand_cache) >= 4_096:
                self._memory_demand_cache.pop(next(iter(
                    self._memory_demand_cache
                )))
            self._memory_demand_cache[cache_key] = cached
        resident_devices = {
            device_id
            for device_id in pattern.device_ids
            if residency_states[device_id] == "hot"
            and (
                residency := self._matching_residency(
                    manifest, pattern, snapshot, device_id
                )
            ) is not None
            and residency.state == "hot"
        }
        return tuple(
            replace(
                demand,
                resident_bytes=demand.required_bytes,
            )
            if demand.kind == "model_weights"
            and demand.device_id in resident_devices
            else demand
            for demand in cached
        )

    def _preallocated_kv_by_device(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        *,
        context_size: int,
        parallel: int,
        sliding_window_padding_tokens: int,
    ) -> dict[str, int]:
        result = {device_id: 0 for device_id in pattern.device_ids}
        for operator in manifest.operators:
            if operator.kind != "kv_cache":
                continue
            required_bytes = manifest.preallocated_kv_cache_bytes(
                operator.operator_id,
                context_size=context_size,
                parallel=parallel,
                sliding_window_padding_tokens=(
                    sliding_window_padding_tokens
                ),
            )
            primary, helper, fraction = pattern.assignments[
                operator.operator_id
            ]
            if pattern.assistance_phase == "decode":
                result[pattern.desktop_assignments[operator.operator_id]] += (
                    required_bytes
                )
            elif helper is None:
                result[primary] += required_bytes
            else:
                primary_bytes, helper_bytes = self._fraction(
                    required_bytes, fraction
                )
                result[primary] += primary_bytes
                result[helper] += helper_bytes
        return result

    @staticmethod
    def _apply_runtime_memory_profile(
        coordinator: (
            RuntimeExecutorCapability
            | RuntimeCompositeExecutorCapability
            | None
        ),
        memory_demands: Sequence[RuntimeMemoryDemand],
    ) -> tuple[RuntimeMemoryDemand, ...]:
        if coordinator is None:
            return tuple(memory_demands)
        parameters = coordinator.adapter_parameters
        result = []
        for demand in memory_demands:
            required_bytes = demand.required_bytes
            if demand.kind == "model_weights":
                key = (
                    "memory_model_weight_allocation_ppm:"
                    + str(demand.device_id)
                )
                multiplier_ppm = parameters.get(key, 1_000_000)
                if (
                    type(multiplier_ppm) is not int
                    or multiplier_ppm < 1_000_000
                ):
                    raise RouteGenerationError(
                        "model weight allocation profile is invalid"
                    )
                required_bytes = _ceil_div(
                    required_bytes * multiplier_ppm,
                    1_000_000,
                )
            elif demand.kind == "workspace":
                key = (
                    "memory_workspace_minimum_bytes:"
                    + str(demand.device_id)
                )
                minimum_bytes = parameters.get(key, 0)
                if type(minimum_bytes) is not int or minimum_bytes < 0:
                    raise RouteGenerationError(
                        "workspace allocation profile is invalid"
                    )
                required_bytes = max(required_bytes, minimum_bytes)
            resident_bytes = (
                required_bytes if demand.resident_bytes > 0 else 0
            )
            result.append(
                demand
                if (
                    required_bytes == demand.required_bytes
                    and resident_bytes == demand.resident_bytes
                )
                else replace(
                    demand,
                    required_bytes=required_bytes,
                    resident_bytes=resident_bytes,
                )
            )
        peak = parameters.get("whole_model_peak_memory_bytes")
        if peak is not None and parameters.get("execution_adapter") == "android-llama-server-v1":
            if type(peak) is not int or peak <= 0:
                raise RouteGenerationError("whole-model peak memory is invalid")
            device_id = parameters.get("gpu_device_id")
            resident = sum(row.required_bytes for row in result if row.device_id == device_id)
            extra = max(0, peak - resident)
            workspace = next((row for row in result if row.device_id == device_id
                              and row.kind == "workspace"), None)
            if extra and workspace is None:
                raise RouteGenerationError("whole-model peak memory has no workspace demand")
            result = [replace(row, required_bytes=row.required_bytes + extra)
                      if row is workspace and extra else row for row in result]
        return tuple(result)

    def _eligibility(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        residency_variant: str,
        residency_states: Mapping[str, str],
        transitions: Sequence[RuntimeTransitionPlan],
        used_links: Sequence[str],
        maturity: str,
        memory_demands: Sequence[RuntimeMemoryDemand],
        coordinator: (
            RuntimeExecutorCapability
            | RuntimeCompositeExecutorCapability
            | None
        ),
    ) -> tuple[str, ...]:
        reasons = []
        absent_co_helpers = unavailable_co_helpers(coordinator, snapshot) & set(pattern.device_ids)
        unknown_telemetry = {
            device_id for device_id in pattern.device_ids
            if device_id not in absent_co_helpers
            and snapshot.telemetry_unavailable_reason(device_id, observed_at_us)
                is not None
        }
        if unknown_telemetry:
            reasons.append("PHONE_TELEMETRY_UNAVAILABLE")
        if absent_co_helpers:
            reasons.append(CO_HELPER_UNAVAILABLE)
        memory_resource_ids = set()
        if isinstance(coordinator, RuntimeCompositeExecutorCapability):
            memory_resource_ids.update(coordinator.resource_ids)
        for device_id in pattern.device_ids:
            memory_resource_ids.update(
                self.catalog.executor_by_device[
                    device_id
                ].execution_resource_ids
            )
        for transition in transitions:
            memory_resource_ids.update(transition.resource_ids)
        scheduler_owned_memory_wait = (
            bool(memory_resource_ids)
            and self.timeline.owned_busy_until_us(
                tuple(sorted(memory_resource_ids)), observed_at_us
            ) is not None
        )
        if pattern.phone_resident_envelope is not None:
            selected_layout = self._phone_residency_layout
            if (
                selected_layout is not None
                and pattern.phone_resident_envelope.geometry_sha256
                    != selected_layout.geometry_sha256
            ):
                reasons.append("PHONE_RESIDENCY_LAYOUT_NOT_SELECTED")
            helper_device_id = (
                coordinator.helper_device_id
                if isinstance(
                    coordinator, RuntimeCompositeExecutorCapability
                )
                else None
            )
            session_by_id = (
                {}
                if helper_device_id is None
                else {
                    row.session_id: row
                    for row in self.catalog.executor_by_device[
                        helper_device_id
                    ].phone_sessions
                }
            )
            for session_id in (
                pattern.phone_resident_envelope.unavailable_session_ids
            ):
                session = session_by_id.get(session_id)
                suffix = (
                    ""
                    if session is None
                    or session.unavailable_reason is None
                    else ":" + session.unavailable_reason.upper()
                )
                reasons.append(
                    "PHONE_SESSION_NOT_READY:"
                    + session_id
                    + suffix
                )
        remote_resident = RouteRemoteResidentMixin._remote_resident_group(
            self, manifest, pattern, coordinator
        )
        remote_owner_devices = set()
        if remote_resident is not None:
            bound, owner_reasons = self._remote_resident_owner_status(
                manifest, remote_resident, snapshot
            )
            if bound is not None:
                remote_owner_devices.update(self._phone_session_capability(row.session_id)[0]
                                            for row in bound.sessions)
            reasons.extend(owner_reasons)
        if len(pattern.device_ids) > 1 and coordinator is None:
            reasons.append("COMPOSITE_COORDINATOR_ABSENT")
        if isinstance(coordinator, RuntimeCompositeExecutorCapability):
            coordinator_state = snapshot.executors.get(
                coordinator.executor_id
            )
            transition_prepares_coordinator = any(
                transition.executor_id == coordinator.executor_id
                for transition in transitions
            )
            owned_release_bounds = tuple(
                self.timeline.owned_busy_until_us(
                    (resource_id,), observed_at_us
                )
                for resource_id in coordinator.resource_ids
            )
            scheduler_owned_wait = (
                not transition_prepares_coordinator
                and bool(owned_release_bounds)
                and all(
                    release_us is not None
                    for release_us in owned_release_bounds
                )
                and all(
                    residency_states[device_id] == "hot"
                    and (
                        residency := self._matching_residency(
                            manifest, pattern, snapshot, device_id
                        )
                    ) is not None
                    and residency.state == "hot"
                    and residency.executor_id == coordinator.executor_id
                    for device_id in coordinator.participant_device_ids
                )
            )
            if coordinator_state is None:
                reasons.append("EXECUTOR_OBSERVATION_ABSENT")
            else:
                if not coordinator_state.healthy:
                    reasons.append("EXECUTOR_UNHEALTHY")
                if (
                    not coordinator_state.ready
                    and not transition_prepares_coordinator
                    and not scheduler_owned_wait
                ):
                    reasons.append("EXECUTOR_NOT_READY")
                elif (
                    coordinator_state.free_slots == 0
                    and coordinator_state.busy_until_us <= observed_at_us
                    and not transition_prepares_coordinator
                    and not scheduler_owned_wait
                ):
                    reasons.append("EXECUTOR_CAPACITY_UNAVAILABLE")
            helper_limit = coordinator.adapter_parameters.get(
                "maximum_helper_resident_weight_bytes"
            )
            if type(helper_limit) is int and coordinator.helper_device_id:
                session_residency = any(
                    demand.kind == "session_residency_constraint"
                    and demand.device_id == coordinator.helper_device_id
                    for demand in memory_demands
                )
                helper_weights = sum(
                    demand.required_bytes
                    for demand in memory_demands
                    if demand.kind == "model_weights"
                    and demand.device_id == coordinator.helper_device_id
                )
                if helper_weights > helper_limit and not session_residency:
                    reasons.append("HELPER_RESIDENCY_CAPACITY")
        if pattern.split_axis != "none":
            operator_by_id = {
                row.operator_id: row for row in manifest.operators
            }
            tensor_by_id = manifest.tensor_by_id
            for operator_id, (_, helper, fraction) in (
                pattern.assignments.items()
            ):
                if helper is None:
                    continue
                for tensor_id in operator_by_id[operator_id].tensor_ids:
                    tensor = tensor_by_id[tensor_id]
                    if (
                        len(tensor.shape) < 2
                        or tensor.quantization_block_size == 1
                    ):
                        continue
                    dimensions = []
                    if pattern.split_axis in {"row", "tensor"}:
                        dimensions.append(tensor.shape[0])
                    if (
                        pattern.split_axis in {"column", "tensor"}
                        and operator_by_id[operator_id].kind == "ffn"
                    ):
                        dimensions.append(
                            tensor.shape[0]
                            if ".ffn_down." in tensor.tensor_id
                            else tensor.shape[-1]
                        )
                    for dimension in dimensions:
                        helper_width = _ceil_div(
                            dimension * fraction, 1_000_000
                        )
                        base_width = dimension - helper_width
                        if (
                            helper_width % tensor.quantization_block_size
                            or base_width % tensor.quantization_block_size
                        ):
                            reasons.append(
                                "QUANTIZATION_BLOCK_MISALIGNED"
                            )
                            break
                    if "QUANTIZATION_BLOCK_MISALIGNED" in reasons:
                        break
                if "QUANTIZATION_BLOCK_MISALIGNED" in reasons:
                    break
        for device_id in pattern.device_ids:
            capability = self.catalog.executor_by_device[device_id]
            state = snapshot.executors.get(capability.executor_id)
            observed = snapshot.residency_for(
                manifest.model_id, manifest.artifact_sha256, device_id
            )
            actual = self._matching_residency(
                manifest, pattern, snapshot, device_id
            )
            if device_id in absent_co_helpers:
                # Out of the fleet: nothing prepares a static co-helper; only the variant must be
                # the observed one.
                if residency_states[device_id] != (
                    "cold" if actual is None else actual.state
                ):
                    reasons.append("RESIDENCY_VARIANT_NOT_CURRENT")
                continue
            transition_prepares_executor = any(
                device_id in row.prepares_device_ids
                for row in transitions
            )
            transition_rebinds_residency = any(
                device_id in row.prepares_device_ids
                and row.executor_id
                    == self._residency_executor_id(pattern, device_id)
                for row in transitions
            )
            if (
                observed is not None
                and actual is None
                and device_id not in remote_owner_devices
                and not transition_rebinds_residency
            ):
                reasons.append("RESIDENCY_EXECUTOR_MISMATCH")
            actual_state = ("hot" if device_id in remote_owner_devices
                            else "cold" if actual is None else actual.state)
            if state is None:
                reasons.append("EXECUTOR_OBSERVATION_ABSENT")
                continue
            owned_release_bounds = tuple(
                self.timeline.owned_busy_until_us(
                    (resource_id,), observed_at_us
                )
                for resource_id in capability.execution_resource_ids
            )
            scheduler_owned_wait = (
                not transition_prepares_executor
                and bool(owned_release_bounds)
                and all(
                    release_us is not None
                    for release_us in owned_release_bounds
                )
            )
            if not isinstance(
                coordinator, RuntimeCompositeExecutorCapability
            ):
                if not state.healthy and not transition_prepares_executor:
                    reasons.append("EXECUTOR_UNHEALTHY")
                if (
                    not state.ready
                    and not transition_prepares_executor
                    and not scheduler_owned_wait
                ):
                    reasons.append("EXECUTOR_NOT_READY")
                elif (
                    state.free_slots == 0
                    and state.busy_until_us <= observed_at_us
                    and not transition_prepares_executor
                    and not scheduler_owned_wait
                ):
                    reasons.append("EXECUTOR_CAPACITY_UNAVAILABLE")
            if device_id not in unknown_telemetry:
                thermal_qualified = state.thermal_qualified_under(
                    capability.maximum_thermal_status
                )
                thermal_limited = (
                    thermal_qualified is False
                    or (
                        thermal_qualified is None
                        and state.temperature_millic
                            > capability.maximum_temperature_millic
                    )
                )
                self.thermal_gate_log.observe(
                    capability, state, observed_at_us, thermal_limited
                )
                if thermal_limited:
                    reasons.append("THERMAL_LIMIT")
            if (device_id not in unknown_telemetry
                    and state.battery_ppm < self._minimum_battery_ppm(device_id)):
                reasons.append("BATTERY_LIMIT")
            if residency_states[device_id] != actual_state:
                reasons.append("RESIDENCY_VARIANT_NOT_CURRENT")
            if (
                actual_state != "hot"
                and residency_states[device_id] == actual_state
            ):
                if not any(
                    device_id in row.prepares_device_ids
                    for row in transitions
                ):
                    reasons.append("RESIDENCY_TRANSITION_ABSENT")
        for link_id in used_links:
            state = snapshot.links.get(link_id)
            if state is None:
                reasons.append("LINK_OBSERVATION_ABSENT")
            elif not state.ready:
                reasons.append("LINK_NOT_READY")
        if maturity != "QUALIFIED":
            reasons.append("ROUTE_NOT_QUALIFIED")
        try:
            preview_runtime_memory(
                memory_demands,
                snapshot.memory,
                transitions=transitions,
                residency=snapshot.residency,
                exclusive_resource_by_device=self.catalog.exclusive_residency_resources,
                enforce_live_capacity=not scheduler_owned_memory_wait,
            )
        except RuntimeResourceError as error:
            message = str(error)
            if message.startswith("memory resource is absent: "):
                reasons.append("MEMORY_RESOURCE_ABSENT")
            elif message.startswith("memory capacity is insufficient: "):
                resource_id = message.removeprefix("memory capacity is insufficient: ")
                unknown_resources = {
                    self.catalog.executor_by_device[device_id].memory_resource_id
                    for device_id in unknown_telemetry | absent_co_helpers
                }
                if resource_id not in unknown_resources:
                    reasons.append("MEMORY_CAPACITY")
                else:
                    try:
                        preview_runtime_memory(
                            memory_demands, snapshot.memory,
                            transitions=transitions, residency=snapshot.residency,
                            exclusive_resource_by_device=self.catalog.exclusive_residency_resources,
                            enforce_live_capacity=False,
                        )
                    except RuntimeResourceError as static_error:
                        if str(static_error).startswith("memory capacity is insufficient: "):
                            reasons.append("MEMORY_CAPACITY")
                        else:
                            raise RouteGenerationError(str(static_error)) from static_error
            else:
                raise RouteGenerationError(message) from error
        return tuple(sorted(set(reasons)))

    @staticmethod
    def _used_links(placement) -> tuple[str, ...]:
        values = set()
        for operator in placement.operator_decisions:
            transfers = list(operator.internal_transfers)
            if operator.transition is not None:
                transfers.append(operator.transition)
            for transfer in transfers:
                values.update(transfer.link_ids)
        if placement.final_transfer is not None:
            values.update(placement.final_transfer.link_ids)
        return tuple(sorted(values))

    @staticmethod
    def _transport_adapter_parameters(
        profile,
        used_links: Sequence[str],
        required_maximum_payload_bytes: int = 0,
        slot_payload_multiplier: int = 1,
        request_io_protocol: str | None = None,
        batch_plan: str | None = None,
        maximum_tokens: int = 1,
    ) -> Mapping[str, int | str]:
        if (
            type(slot_payload_multiplier) is not int
            or slot_payload_multiplier <= 0
            or type(maximum_tokens) is not int
            or maximum_tokens <= 0
        ):
            raise RouteGenerationError(
                "phone transport slot geometry is invalid"
            )
        if batch_plan is not None and (
            type(batch_plan) is not str or not batch_plan.isascii()
        ):
            raise RouteGenerationError(
                "phone transport batch plan is invalid"
            )
        by_id = {row.link_id: row for row in profile.links}
        phone_links = []
        for link_id in used_links:
            link = by_id[link_id]
            source_kind = profile.devices[link.source_device].kind
            target_kind = profile.devices[link.target_device].kind
            if source_kind.startswith("phone") != target_kind.startswith(
                "phone"
            ):
                phone_links.append(link)
        if not phone_links:
            return {}
        if all(
            row.transport_generation == "legacy"
            and row.allocator == "unspecified"
            for row in phone_links
        ):
            return {}
        if any(
            row.transport_generation == "legacy"
            or row.allocator == "unspecified"
            for row in phone_links
        ):
            raise RouteGenerationError(
                "phone transport profile is only partially specified"
            )
        directions = {
            name: tuple(
                row for row in phone_links
                if (
                    profile.devices[row.source_device].kind.startswith(
                        "phone"
                    )
                ) == (name == "d2h")
            )
            for name in ("h2d", "d2h")
        }
        if not all(directions.values()):
            raise RouteGenerationError(
                "phone route requires measured links in both directions"
            )
        h2d = directions["h2d"][0]
        d2h = directions["d2h"][0]
        shared = (
            h2d.allocator,
            h2d.full_duplex,
            h2d.transport_generation,
            h2d.usbfs_available_bytes,
            h2d.slot_safety_bytes,
            h2d.qualification_identity_sha256,
        )
        if any(
            shared != (
                row.allocator,
                row.full_duplex,
                row.transport_generation,
                row.usbfs_available_bytes,
                row.slot_safety_bytes,
                row.qualification_identity_sha256,
            )
            for row in phone_links
        ):
            raise RouteGenerationError(
                "phone transport direction profiles are incompatible"
            )
        qualification_links = list(phone_links)
        capacity_h2d = h2d
        capacity_d2h = d2h
        qualified_transfer_payload_bytes = required_maximum_payload_bytes
        if required_maximum_payload_bytes and batch_plan == "split-row":
            qualified_transfer_payload_bytes = _ceil_div(
                required_maximum_payload_bytes, maximum_tokens
            )
        if required_maximum_payload_bytes:
            capacity_rows: dict[
                tuple[object, ...], dict[str, object]
            ] = {}
            for row in profile.links:
                source_phone = profile.devices[
                    row.source_device
                ].kind.startswith("phone")
                target_phone = profile.devices[
                    row.target_device
                ].kind.startswith("phone")
                if (
                    not row.ready
                    or row.status != "measured"
                    or source_phone == target_phone
                    or row.maximum_payload_bytes
                        < qualified_transfer_payload_bytes
                    or shared != (
                        row.allocator,
                        row.full_duplex,
                        row.transport_generation,
                        row.usbfs_available_bytes,
                        row.slot_safety_bytes,
                        row.qualification_identity_sha256,
                    )
                ):
                    continue
                direction = "d2h" if source_phone else "h2d"
                reference = directions[direction][0]
                if (
                    row.source_device != reference.source_device
                    or row.target_device != reference.target_device
                ):
                    continue
                key = (
                    row.allocator,
                    row.queue_depth,
                    row.concurrent_streams,
                    row.full_duplex,
                    row.transport_generation,
                    row.usbfs_available_bytes,
                    row.slot_safety_bytes,
                    row.qualification_identity_sha256,
                    row.maximum_payload_bytes,
                )
                current = capacity_rows.setdefault(key, {})
                selected = current.get(direction)
                if selected is None or (
                    row.fixed_latency_us + _ceil_div(
                        qualified_transfer_payload_bytes * 1_000_000,
                        row.bandwidth_bytes_per_s,
                    ),
                    row.link_id,
                ) < (
                    selected.fixed_latency_us + _ceil_div(
                        qualified_transfer_payload_bytes * 1_000_000,
                        selected.bandwidth_bytes_per_s,
                    ),
                    selected.link_id,
                ):
                    current[direction] = row
            complete = tuple(
                (key, values["h2d"], values["d2h"])
                for key, values in capacity_rows.items()
                if set(values) == {"h2d", "d2h"}
            )
            if not complete:
                raise RouteGenerationError(
                    "phone transport maximum payload is not qualified"
                )
            _, capacity_h2d, capacity_d2h = min(
                complete,
                key=lambda item: (
                    -item[0][1],
                    item[0][8],
                    -item[0][2],
                    item[1].link_id,
                    item[2].link_id,
                ),
            )
            qualification_links.extend((capacity_h2d, capacity_d2h))
        maximum_payloads = tuple(
            row.maximum_payload_bytes for row in qualification_links
        )
        if any(value <= 0 for value in maximum_payloads):
            raise RouteGenerationError(
                "phone transport maximum payload is unbounded"
            )
        configured_maximum = (
            required_maximum_payload_bytes
            if required_maximum_payload_bytes
            else max(maximum_payloads)
        )
        slot_bytes = (
            configured_maximum * slot_payload_multiplier
            + capacity_h2d.slot_safety_bytes
        )
        if (
            capacity_h2d.usbfs_available_bytes
            and slot_bytes * capacity_h2d.queue_depth
                > capacity_h2d.usbfs_available_bytes
        ):
            raise RouteGenerationError(
                "phone transport slot buffers exceed usbfs capacity"
            )
        profile_ids = "+".join(sorted({
            row.transport_profile_id for row in qualification_links
        }))
        if request_io_protocol == "token-ids-v1":
            return {
                "request_transport": capacity_h2d.transport_generation,
                "request_transport_allocator": capacity_h2d.allocator,
                "request_transport_capacity_d2h_profile_id": (
                    capacity_d2h.transport_profile_id
                ),
                "request_transport_capacity_h2d_profile_id": (
                    capacity_h2d.transport_profile_id
                ),
                "request_transport_concurrent_streams": (
                    capacity_h2d.concurrent_streams
                ),
                "request_transport_d2h_profile_id": (
                    d2h.transport_profile_id
                ),
                "request_transport_full_duplex": int(
                    capacity_h2d.full_duplex
                ),
                "request_transport_h2d_profile_id": (
                    h2d.transport_profile_id
                ),
                "request_transport_max_payload_bytes": configured_maximum,
                "request_transport_profile_id": profile_ids,
                "request_transport_queue_depth": capacity_h2d.queue_depth,
                "request_transport_latency_accounting": (
                    "per-message-wave-v1"
                ),
                **(
                    {}
                    if capacity_h2d.qualification_identity_sha256 is None
                    else {
                        "request_transport_identity_sha256": (
                            capacity_h2d.qualification_identity_sha256
                        )
                    }
                ),
            }
        return {
            "ffn_transport": "functionfs-usb",
            "usb_allocator": capacity_h2d.allocator,
            "usb_capacity_d2h_transport_profile_id": (
                capacity_d2h.transport_profile_id
            ),
            "usb_capacity_h2d_transport_profile_id": (
                capacity_h2d.transport_profile_id
            ),
            "usb_concurrent_streams": capacity_h2d.concurrent_streams,
            "usb_cost_d2h_concurrent_streams": d2h.concurrent_streams,
            "usb_cost_d2h_queue_depth": d2h.queue_depth,
            "usb_cost_h2d_concurrent_streams": h2d.concurrent_streams,
            "usb_cost_h2d_queue_depth": h2d.queue_depth,
            "usb_d2h_transport_profile_id": d2h.transport_profile_id,
            "usb_full_duplex": int(capacity_h2d.full_duplex),
            "usb_h2d_transport_profile_id": h2d.transport_profile_id,
            "usb_max_payload_bytes": configured_maximum,
            "usb_qualified_transfer_payload_bytes": (
                qualified_transfer_payload_bytes
            ),
            "usb_queue_depth": capacity_h2d.queue_depth,
            "usb_latency_accounting": "per-message-wave-v1",
            "usb_slot_safety_bytes": capacity_h2d.slot_safety_bytes,
            "usb_transport_generation": capacity_h2d.transport_generation,
            **(
                {}
                if capacity_h2d.qualification_identity_sha256 is None
                else {
                    "usb_transport_qualification_identity_sha256": (
                        capacity_h2d.qualification_identity_sha256
                    )
                }
            ),
            "usb_transport_profile_id": profile_ids,
            "usbfs_available_bytes": capacity_h2d.usbfs_available_bytes,
        }

    def _resources(
        self,
        pattern: _Pattern,
        used_links: Sequence[str],
        transitions: Sequence[RuntimeTransitionPlan],
        coordinator: (
            RuntimeExecutorCapability
            | RuntimeCompositeExecutorCapability
            | None
        ),
    ) -> tuple[str, ...]:
        if isinstance(coordinator, RuntimeCompositeExecutorCapability):
            values = set(coordinator.resource_ids)
        else:
            values = {
                resource_id
                for device_id in pattern.device_ids
                for resource_id in self.catalog.executor_by_device[
                    device_id
                ].execution_resource_ids
            }
        values.update("link:" + link_id for link_id in used_links)
        values.update(
            resource_id
            for transition in transitions
            for resource_id in transition.resource_ids
        )
        unknown = values - set(self.catalog.resources)
        if unknown:
            raise RouteGenerationError(
                "generated route references unknown resources: "
                + ", ".join(sorted(unknown))
            )
        return tuple(sorted(values))

    def _lease_preview(
        self,
        route_id: str,
        resource_slots: Mapping[str, int],
        arrival_us: int,
        service_us: int,
        service_upper_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
        *, transitions=(),
    ):
        busy = snapshot.busy_until_by_resource(self.catalog)
        not_before = max(
            (
                arrival_us,
                *(busy.get(resource_id, 0) for resource_id in resource_slots),
            )
        )
        demands = runtime_phase_lease_demands(
            route_id, resource_slots, transitions, service_us, service_upper_us,
        )
        return self.timeline.preview_leases(
            demands, not_before, service_us, service_upper_us
        )

    def preview_candidate(
        self,
        candidate: AutomatedRouteCandidate,
        request,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int | None = None,
    ):
        if not isinstance(candidate, AutomatedRouteCandidate):
            raise RouteGenerationError("automated candidate is invalid")
        if not candidate.admitted:
            raise RouteGenerationError("automated candidate is not admitted")
        arrival_us = (
            request.arrival_us if observed_at_us is None else observed_at_us
        )
        arrival_us = max(arrival_us, candidate.cost.start_us)
        return self._lease_preview(
            candidate.candidate_id,
            candidate.plan.resource_slots,
            arrival_us,
            candidate.cost.service_us,
            candidate.cost.service_upper_us,
            snapshot,
            transitions=candidate.plan.transitions,
        )

    def _participants(
        self,
        pattern: _Pattern,
        coordinator: (
            RuntimeExecutorCapability
            | RuntimeCompositeExecutorCapability
            | None
        ),
    ) -> tuple[RuntimeParticipantBinding, ...]:
        cached = self._participant_binding_cache.get(pattern.route_key)
        if cached is not None:
            return cached
        result = tuple(
            RuntimeParticipantBinding(
                executor_id=capability.executor_id,
                device_id=capability.device_id,
                endpoint=capability.endpoint,
                backend=capability.backend,
                resource_ids=(
                    capability.execution_resource_ids
                    if not isinstance(
                        coordinator, RuntimeCompositeExecutorCapability
                    )
                    else coordinator.participant_resource_ids[
                        capability.device_id
                    ]
                ),
            )
            for capability in (
                self.catalog.executor_by_device[device_id]
                for device_id in pattern.device_ids
            )
        )
        self._bounded_cache_store(
            self._participant_binding_cache,
            pattern.route_key,
            result,
            4_096,
        )
        return result

    def _coordinator(
        self, pattern: _Pattern
    ) -> (
        RuntimeExecutorCapability
        | RuntimeCompositeExecutorCapability
        | None
    ):
        if pattern.route_key in self._coordinator_cache:
            return self._coordinator_cache[pattern.route_key]
        if pattern.coordinator_executor_id is not None:
            result = self.catalog.composite_executor_by_id[
                pattern.coordinator_executor_id
            ]
        else:
            capability = self.catalog.executor_by_device[
                pattern.coordinator_device_id
            ]
            if len(pattern.device_ids) == 1 or (
                pattern.route_family
                    in capability.coordinated_route_families
                and capability.operator_plan_protocol is not None
            ):
                result = capability
            else:
                result = None
        self._bounded_cache_store(
            self._coordinator_cache,
            pattern.route_key,
            result,
            4_096,
        )
        return result
