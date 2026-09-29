"""Route costing: memory demands, residency transitions and their measured bounds."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
    RuntimeExecutorCapability,
)
from ..capability_contracts.executors import whole_phone_launch_parameters
from ..runtime_cost import RuntimeMemoryDemand
from ..runtime_plan import RuntimeOperatorAssignment
from ..runtime_resources import transition_adjusted_memory_demands
from ..types import canonical_sha256
from .common import (
    RouteGenerationError,
    _ceil_div,
    _minimum_maturity,
    _FfnResidentEnvelope,
    _Pattern,
)
from .costing import (
    _CandidateServiceEstimate,
)


class RouteDemandMixin:
    """Route costing: memory demands, residency transitions and their measured bounds."""

    def _resident_phone_memory_demands(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        coordinator: RuntimeCompositeExecutorCapability | None,
        memory_demands: tuple[RuntimeMemoryDemand, ...],
    ) -> tuple[
        tuple[RuntimeMemoryDemand, ...],
        _FfnResidentEnvelope | None,
        str | None,
        bool,
    ]:
        resident_envelope = (
            pattern.phone_resident_envelope
            if isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and pattern.assistance_phase == "decode"
            else None
        )
        if resident_envelope is None:
            return memory_demands, None, None, False
        residency_shards = resident_envelope.transition_shards
        assert coordinator is not None
        helper_id = coordinator.helper_device_id
        assisted_ids = {
            operator_id
            for operator_id, (primary, helper, _fraction)
            in pattern.assignments.items()
            if helper_id in {primary, helper}
            and pattern.desktop_assignments.get(operator_id) != helper_id
        }
        if not assisted_ids.issubset(resident_envelope.operator_ids):
            return (
                memory_demands,
                resident_envelope,
                "resident FFN superset does not cover the candidate",
                False,
            )
        session_by_id = {
            row.session_id: row
            for row in self.catalog.executor_by_device[helper_id].phone_sessions
        }
        live_helper_residency = self._matching_residency(
            manifest, pattern, snapshot, helper_id
        )
        if live_helper_residency is None and any(
            shard.artifact_sha256 != manifest.artifact_sha256
            for shard in residency_shards
        ):
            live_helper_residency = next((
                row for row in snapshot.residency
                if row.device_id == helper_id
                and row.state in {"hot", "warm"}
                and row.executor_id == coordinator.executor_id
                and row.resident_geometry_sha256
                    == resident_envelope.geometry_sha256
            ), None)
        live_shard_set = (
            live_helper_residency is not None
            and live_helper_residency.state in {"hot", "warm"}
            and live_helper_residency.executor_id == coordinator.executor_id
            and live_helper_residency.resident_geometry_sha256
                == resident_envelope.geometry_sha256
        )
        observed_session_by_id = {
            row.session_id: row
            for row in snapshot.phone_session_residency
            if row.device_id == helper_id
        }
        exact_session_set = bool(residency_shards) and all(
            (observed := observed_session_by_id.get(shard.session_id))
                is not None
            and observed.state == "READY"
            and observed.endpoint == shard.endpoint
            and observed.artifact_sha256 == shard.artifact_sha256
            and observed.resident_geometry_sha256
                == shard.resident_geometry_sha256
            and observed.operator_plan_sha256
                == shard.operator_plan_sha256
            and observed.session_generation == shard.session_generation
            and observed.resident_bytes == shard.resident_bytes
            for shard in residency_shards
        )
        live_shard_set = live_shard_set or exact_session_set
        phone_shards_require_transition = bool(
            residency_shards
            and not live_shard_set
            and any(
                session_by_id[shard.session_id].residency_state
                    not in {"hot", "warm"}
                or session_by_id[
                    shard.session_id
                ].resident_artifact_sha256 != shard.artifact_sha256
                or session_by_id[
                    shard.session_id
                ].resident_geometry_sha256
                    != shard.resident_geometry_sha256
                for shard in residency_shards
            )
        )
        memory_demands = tuple(
            replace(
                demand,
                required_bytes=resident_envelope.weight_bytes,
                resident_bytes=(
                    resident_envelope.weight_bytes
                    if not phone_shards_require_transition
                    and (demand.resident_bytes > 0 or exact_session_set)
                    else 0
                ),
                share_key="phone-residency:" + resident_envelope.geometry_sha256,
            )
            if demand.kind == "model_weights"
            and demand.device_id == helper_id
            else demand
            for demand in memory_demands
        )
        session_demands = []
        for shard in residency_shards:
            session = session_by_id[shard.session_id]
            resident = (
                shard.resident_bytes
                if live_shard_set or (
                    session.residency_state in {"hot", "warm"}
                    and session.resident_artifact_sha256
                        == shard.artifact_sha256
                    and session.resident_geometry_sha256
                        == shard.resident_geometry_sha256
                )
                else 0
            )
            session_demands.append(RuntimeMemoryDemand(
                demand_id="phone-session:" + shard.session_id + ":weights",
                resource_id=session.memory_resource_id,
                kind="session_residency_constraint",
                required_bytes=shard.resident_bytes,
                resident_bytes=resident,
                lifetime="resident",
                share_key=(
                    str(shard.artifact_sha256)
                    + ":"
                    + shard.resident_geometry_sha256
                ),
                device_id=helper_id,
            ))
        return (
            tuple(memory_demands) + tuple(session_demands),
            resident_envelope,
            None,
            phone_shards_require_transition,
        )

    def _preallocated_request_memory_demands(
        self,
        manifest: ModelManifest,
        work: ModelRequestWork,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        coordinator: RuntimeCompositeExecutorCapability | None,
        memory_demands: tuple[RuntimeMemoryDemand, ...],
    ) -> tuple[RuntimeMemoryDemand, ...]:
        if (
            not isinstance(coordinator, RuntimeCompositeExecutorCapability)
            or coordinator.adapter_parameters.get("request_memory_mode")
                != "preallocated"
        ):
            return memory_demands
        context_size = coordinator.adapter_parameters.get("context_size")
        if type(context_size) is not int or context_size <= 0:
            raise RouteGenerationError(
                "preallocated request memory lacks a context size"
            )
        parallel = coordinator.adapter_parameters.get("parallel", 1)
        if type(parallel) is not int or parallel <= 0:
            raise RouteGenerationError(
                "preallocated request memory lacks parallelism"
            )
        sliding_window_padding_tokens = coordinator.adapter_parameters.get(
            "kv_cache_swa_padding_tokens"
        )
        request_tokens = work.input_tokens + work.output_tokens
        if sliding_window_padding_tokens is None:
            memory_demands = tuple(
                replace(
                    demand,
                    required_bytes=_ceil_div(
                        demand.required_bytes * context_size,
                        request_tokens,
                    ),
                )
                if demand.kind == "kv_cache" else demand
                for demand in memory_demands
            )
        else:
            if (
                type(sliding_window_padding_tokens) is not int
                or sliding_window_padding_tokens < 0
            ):
                raise RouteGenerationError(
                    "preallocated SWA KV padding is invalid"
                )
            preallocated_kv = self._preallocated_kv_by_device(
                manifest,
                pattern,
                context_size=context_size,
                parallel=parallel,
                sliding_window_padding_tokens=sliding_window_padding_tokens,
            )
            memory_demands = tuple(
                replace(
                    demand,
                    required_bytes=preallocated_kv[demand.device_id],
                )
                if demand.kind == "kv_cache" else demand
                for demand in memory_demands
            )
        preallocated_memory_identity = canonical_sha256({
            "artifact_sha256": manifest.artifact_sha256,
            "context_resource_id": coordinator.adapter_parameters.get(
                "context_resource_id"
            ),
            "endpoint": coordinator.endpoint,
            "executor_id": coordinator.executor_id,
            "request_memory_mode": "preallocated",
        })
        preallocated_devices = {
            device_id
            for device_id in pattern.device_ids
            for residency in (
                self._matching_residency(
                    manifest, pattern, snapshot, device_id
                ),
            )
            if residency is not None
            and residency.state == "hot"
            and residency.executor_id == coordinator.executor_id
        }
        return tuple(
            replace(
                demand,
                resident_bytes=(
                    demand.required_bytes
                    if demand.device_id in preallocated_devices else 0
                ),
                lifetime="resident",
                share_key=(
                    preallocated_memory_identity
                    + ":"
                    + str(demand.device_id)
                    + ":"
                    + demand.resource_id
                    + ":"
                    + demand.kind
                ),
            )
            if demand.kind in {"kv_cache", "workspace"}
            else demand
            for demand in memory_demands
        )

    def _persistent_whole_phone_memory_demands(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        coordinator: RuntimeExecutorCapability | RuntimeCompositeExecutorCapability | None,
        residency_states: Mapping[str, str],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
    ) -> tuple[RuntimeMemoryDemand, ...]:
        parameters = {} if coordinator is None else coordinator.adapter_parameters
        peak = parameters.get("whole_model_peak_memory_bytes")
        if (parameters.get("execution_adapter") != "android-llama-server-v1"
                or parameters.get("persistent_residency") != 1 or peak is None):
            return memory_demands
        device_id = parameters["gpu_device_id"]
        weights = sum(row.required_bytes for row in memory_demands
                      if row.device_id == device_id and row.kind == "model_weights")
        runtime_bytes = max(0, peak - weights)
        if not runtime_bytes:
            return memory_demands
        observed = self._matching_residency(manifest, pattern, snapshot, device_id)
        live = snapshot.executors.get(coordinator.executor_id)
        resident_bytes = 0
        if (observed is not None and observed.state == "hot"
                and residency_states.get(device_id) == "hot" and observed.generation > 0
                and observed.executor_id == coordinator.executor_id
                and observed.resident_bytes >= manifest.tensor_bytes
                and set(manifest.tensor_by_id).issubset(observed.resident_tensor_ids)
                and live is not None and live.ready and live.healthy
                and whole_phone_launch_parameters(parameters)
                    == whole_phone_launch_parameters(observed.resident_adapter_parameters)):
            resident_bytes = min(runtime_bytes, max(
                0, (observed.reclaimable_bytes or observed.resident_bytes) - weights,
            ))
        identity = canonical_sha256({
            "artifact_sha256": manifest.artifact_sha256,
            "device_id": device_id,
            "endpoint": coordinator.endpoint,
            "executor_id": coordinator.executor_id,
            "adapter_parameters": whole_phone_launch_parameters(parameters),
        })
        # The endpoint peak is shared; request growth beyond it is not.
        remaining = runtime_bytes
        result = []
        runtime_resource_id = None
        for row in memory_demands:
            if row.device_id == device_id and row.kind in {"kv_cache", "workspace"}:
                if runtime_resource_id not in {None, row.resource_id}:
                    raise RouteGenerationError("whole-model runtime memory uses multiple resources")
                runtime_resource_id = row.resource_id
                consumed = min(remaining, row.required_bytes)
                remaining -= consumed
                if row.required_bytes > consumed:
                    result.append(replace(row, required_bytes=row.required_bytes - consumed,
                                          resident_bytes=0))
            else:
                result.append(row)
        if remaining or runtime_resource_id is None:
            raise RouteGenerationError("whole-model peak memory lacks runtime demands")
        result.append(RuntimeMemoryDemand(
            demand_id="whole-service-runtime:" + device_id,
            resource_id=runtime_resource_id,
            kind="workspace",
            required_bytes=runtime_bytes,
            resident_bytes=resident_bytes,
            lifetime="resident",
            share_key=identity + ":whole-service-runtime",
            device_id=device_id,
        ))
        return tuple(result)

    def _candidate_transitions(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        coordinator: RuntimeExecutorCapability | RuntimeCompositeExecutorCapability | None,
        residency_states: Mapping[str, str],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        resident_envelope: _FfnResidentEnvelope | None,
        phone_shards_require_transition: bool,
    ) -> tuple[tuple[RuntimeMemoryDemand, ...], tuple, str]:
        memory_demands = self._apply_runtime_memory_profile(
            coordinator, memory_demands
        )
        memory_demands = self._persistent_whole_phone_memory_demands(
            manifest, pattern, snapshot, coordinator, residency_states, memory_demands,
        )
        weights_by_device = {
            device_id: sum(
                demand.required_bytes
                for demand in memory_demands
                if demand.demand_id == f"weights:{device_id}"
            )
            for device_id in pattern.device_ids
        }
        transition_executor_id = (
            self.catalog.executor_by_device[
                pattern.coordinator_device_id
            ].executor_id
            if coordinator is None else coordinator.executor_id
        )
        transition_residency_states = residency_states
        if phone_shards_require_transition:
            assert coordinator is not None
            transition_residency_states = {
                **residency_states,
                coordinator.helper_device_id: "cold",
            }
        transitions = self._transitions(
            pattern.device_ids,
            transition_residency_states,
            weights_by_device,
            manifest.artifact_sha256,
            transition_executor_id,
            snapshot,
        )
        if resident_envelope is not None and resident_envelope.transition_shards:
            assert coordinator is not None
            helper_id = coordinator.helper_device_id
            transitions = tuple(
                self._bind_phone_shards_to_transition(
                    transition,
                    resident_envelope.transition_shards,
                    resident_envelope.changed_session_ids,
                    resident_envelope.replacement_source_identities,
                    resident_envelope
                        .replacement_source_resident_bytes_by_session,
                    manifest.model_id,
                    manifest.artifact_sha256,
                    helper_id,
                    self._model_id_by_artifact,
                )
                if helper_id in transition.prepares_device_ids
                else transition
                for transition in transitions
            )
        memory_demands = transition_adjusted_memory_demands(
            memory_demands,
            transitions=transitions,
            residency=snapshot.residency,
            exclusive_resource_by_device=self.catalog.exclusive_residency_resources,
        )
        return memory_demands, transitions, transition_executor_id

    def _candidate_measured_transitions(
        self,
        *,
        manifest: ModelManifest,
        plan,
        transitions: tuple,
        transition_executor_id: str,
        route_id: str,
        pattern: _Pattern,
        residency_variant: str,
        overlap_kind: str,
        assignments: tuple[RuntimeOperatorAssignment, ...],
        resources: tuple[str, ...],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        execution_contract,
        route_profile,
        resource_slots: Mapping[str, int],
        adapter_parameters: Mapping[str, object],
        desktop_placement_sha256: str,
    ) -> tuple:
        transition_component_identity = (
            self._transition_component_identity(manifest, plan)
        )
        component_capability_sha256 = self.component_capability_identity(
            plan, transition_executor_id
        )
        transition_estimates = tuple(
            self._observation_store.transition_estimate_for_plan(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=component_capability_sha256,
                component_identity_sha256=transition_component_identity,
                transition=transition,
                executor_id=(
                    transition.executor_id
                    or self.catalog.executor_by_device[
                        transition.device_id
                    ].executor_id
                ),
            )
            for transition in transitions
        )
        measured_transitions = tuple(
            replace(
                transition,
                latency_us=(
                    estimate.latency_us
                    if estimate is not None
                    and estimate.latency_upper_maturity == "QUALIFIED"
                    else transition.latency_us
                ),
                energy_uj=(
                    estimate.energy_uj
                    if estimate is not None
                    and estimate.energy_maturity == "QUALIFIED"
                    and estimate.energy_uj is not None
                    else transition.energy_uj
                ),
                maturity=(
                    _minimum_maturity((
                        transition.maturity,
                        estimate.latency_upper_maturity,
                    ))
                    if estimate is not None
                    and estimate.latency_upper_maturity == "QUALIFIED"
                    else transition.maturity
                ),
                energy_maturity=(
                    "QUARANTINED"
                    if transition.energy_maturity == "QUARANTINED"
                    else estimate.energy_maturity
                    if estimate is not None
                    and estimate.energy_maturity == "QUALIFIED"
                    else transition.energy_maturity
                ),
            )
            for transition, estimate in zip(
                transitions, transition_estimates
            )
        )
        if measured_transitions != transitions:
            transitions = measured_transitions
            plan = self._cached_execution_plan(
                route_id=route_id,
                pattern=pattern,
                residency_variant=residency_variant,
                overlap_kind=overlap_kind,
                assignments=assignments,
                transitions=transitions,
                resources=resources,
                memory_demands=memory_demands,
                execution_contract=execution_contract,
                route_profile_id=(
                    None if route_profile is None else route_profile.selector_id
                ),
                resource_slots=resource_slots,
                adapter_parameters=adapter_parameters,
                desktop_placement_sha256=desktop_placement_sha256,
            )
            self._bounded_cache_store(
                self._executor_id_by_plan_sha256,
                plan.plan_sha256,
                transition_executor_id,
                4_096,
            )
        return transitions, plan, transition_estimates, component_capability_sha256

    @staticmethod
    def _candidate_transition_energy_bounds(
        profile,
        transitions: tuple,
        transition_estimates: tuple,
        service: _CandidateServiceEstimate,
    ) -> tuple[int, int, int]:
        idle_transition_energy = sum(
            _ceil_div(
                domain.domain.idle_power_mw * service.transition_us,
                1000,
            )
            for domain in profile.domains.values()
        )
        measured_bounds = bool(transitions) and all(
            estimate is not None
            and estimate.energy_maturity == "QUALIFIED"
            and estimate.energy_uj is not None
            and estimate.energy_lower_uj is not None
            and estimate.energy_upper_uj is not None
            for estimate in transition_estimates
        )
        if not transitions:
            return 0, 0, 0
        if measured_bounds:
            return (
                sum(
                    estimate.energy_uj
                    for estimate in transition_estimates
                    if estimate is not None and estimate.energy_uj is not None
                ),
                sum(
                    estimate.energy_lower_uj
                    for estimate in transition_estimates
                    if estimate is not None
                    and estimate.energy_lower_uj is not None
                ),
                sum(
                    estimate.energy_upper_uj
                    for estimate in transition_estimates
                    if estimate is not None
                    and estimate.energy_upper_uj is not None
                ),
            )
        point = service.transition_energy_uj + idle_transition_energy
        if service.transition_energy_known:
            return point, point, point
        return (
            point,
            point * 500_000 // 1_000_000,
            _ceil_div(point * 1_500_000, 1_000_000),
        )
