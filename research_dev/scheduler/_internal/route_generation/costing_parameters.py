"""Route costing: route profiles, resource context, adapter parameters, FFN geometry and execution plans."""

from __future__ import annotations

from math import gcd
from typing import Mapping
from ..model_manifest import ModelManifest, ModelRequestWork
from ..plan_contracts.co_helpers import (
    PHONE_HELPERS_PARAMETER,
    co_helper_declaration,
)
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
)
from ..runtime_cost import RuntimeExecutorBinding, RuntimeMemoryDemand
from ..runtime_plan import RuntimeOperatorAssignment
from ..runtime_residency_cohorts import runtime_residency_component_identity_from_parts
from ..runtime_search import request_shape_bucket
from .common import (
    RouteGenerationError,
    _DORMANT_PHONE_FFN_RUNTIME_PARAMETER,
    _MATURITY_RANK,
    _ceil_div,
    _minimum_maturity,
    _FfnResidentEnvelope,
    _Pattern,
)
from .costing import (
    _physical_interference_resources,
)
from .remote_resident import REMOTE_RESIDENT_FFN_PARAMETER, remote_resident_link_ids


class RouteParameterMixin:
    """Route costing: route profiles, resource context, adapter parameters, FFN geometry and execution plans."""

    def _candidate_route_profile(
        self,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        pattern: _Pattern,
        residency_variant: str,
        coordinator: RuntimeCompositeExecutorCapability | None,
        transitions: tuple,
        resident_envelope: _FfnResidentEnvelope | None,
        placement,
    ) -> tuple[object | None, tuple, bool, str]:
        route_profile = self.catalog.route_profile_for(
            artifact_sha256=manifest.artifact_sha256,
            route_family=pattern.route_family,
            device_ids=pattern.device_ids,
            assisted_operator_kind=pattern.assisted_operator_kind,
            split_axis=pattern.split_axis,
            split_fraction_ppm=pattern.split_fraction_ppm,
            residency_variant=residency_variant,
            input_tokens=work.input_tokens,
            output_tokens=work.output_tokens,
            cost_features=snapshot.cost_features,
            executor_id=(
                None if coordinator is None else coordinator.executor_id
            ),
        )
        if transitions and (
            route_profile is None or route_profile.maturity != "QUALIFIED"
        ):
            warm_profile = self.catalog.route_profile_for(
                artifact_sha256=manifest.artifact_sha256,
                route_family=pattern.route_family,
                device_ids=pattern.device_ids,
                assisted_operator_kind=pattern.assisted_operator_kind,
                split_axis=pattern.split_axis,
                split_fraction_ppm=pattern.split_fraction_ppm,
                residency_variant="hot",
                input_tokens=work.input_tokens,
                output_tokens=work.output_tokens,
                cost_features=snapshot.cost_features,
                executor_id=(
                    None if coordinator is None else coordinator.executor_id
                ),
            )
            if warm_profile is not None and (
                route_profile is None
                or _MATURITY_RANK[warm_profile.maturity]
                    > _MATURITY_RANK[route_profile.maturity]
            ):
                route_profile = warm_profile
        if resident_envelope is not None and resident_envelope.shards:
            route_profile = None
        phone_power_profiles = tuple(
            row for row in self.catalog.phone_power_profiles
            if row.device_id in pattern.device_ids
        )
        use_assumed_phone_energy = bool(phone_power_profiles)
        maturity = _minimum_maturity(
            [
                self.catalog.executor_by_device[device_id].maturity
                for device_id in pattern.device_ids
            ]
            + (
                [coordinator.maturity]
                if isinstance(
                    coordinator, RuntimeCompositeExecutorCapability
                )
                else []
            )
            + [row.maturity for row in transitions]
            + (
                [route_profile.maturity]
                if route_profile is not None
                else ["SHADOW"]
                if (
                    placement is not None
                    and not placement.measured
                    and not use_assumed_phone_energy
                )
                else []
            )
        )
        if resident_envelope is not None and resident_envelope.shards:
            maturity = _minimum_maturity((maturity, "SHADOW"))
        return (
            route_profile,
            phone_power_profiles,
            use_assumed_phone_energy,
            maturity,
        )

    def _candidate_resource_context(
        self,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        pattern: _Pattern,
        coordinator: RuntimeCompositeExecutorCapability | None,
        transitions: tuple,
        placement,
        observed_at_us: int,
        profile=None,
    ) -> tuple:
        coordinator_parameters = (
            coordinator.adapter_parameters
            if isinstance(coordinator, RuntimeCompositeExecutorCapability)
            else {}
        )
        used_links = () if placement is None else self._used_links(placement)
        used_links = tuple(sorted(set(used_links) | set(remote_resident_link_ids(
            self.catalog.placement_profile if profile is None else profile,
            coordinator_parameters,
        ))))
        resources = self._resources(
            pattern, used_links, transitions, coordinator
        )
        resource_slots = {resource_id: 1 for resource_id in resources}
        context_resource_id = coordinator_parameters.get(
            "context_resource_id"
        )
        if coordinator_parameters.get("request_memory_mode") == "preallocated":
            if (
                type(context_resource_id) is not str
                or context_resource_id not in resource_slots
            ):
                raise RouteGenerationError(
                    "preallocated request memory lacks a context resource"
                )
            context_token_quantum = coordinator_parameters.get(
                "context_token_quantum"
            )
            if (
                type(context_token_quantum) is not int
                or context_token_quantum <= 0
            ):
                raise RouteGenerationError(
                    "preallocated request memory lacks a context quantum"
                )
            resource_slots[context_resource_id] = _ceil_div(
                work.input_tokens + work.output_tokens,
                context_token_quantum,
            )
        protected_work = snapshot.protected_work
        if (
            protected_work is not None
            and protected_work.measured
            and protected_work.critical_path_end_us <= observed_at_us
        ):
            protected_work = None
        system_profile = self.catalog.system_cost_profile_for(
            snapshot.cost_features
        )
        interference_resources = _physical_interference_resources(
            self.catalog, resources, used_links, coordinator, system_profile,
        )
        control_cost_known = (
            system_profile is not None
            and system_profile.maturity == "QUALIFIED"
            and not system_profile.missing_resources(interference_resources)
        )
        system_cost_known = (
            protected_work is not None
            and protected_work.measured
            and control_cost_known
        )
        control_delay_us = (
            system_profile.control_delay_us
            if control_cost_known and system_profile is not None
            else 0
        )
        control_delay_upper_us = (
            system_profile.control_delay_upper_us
            if control_cost_known and system_profile is not None
            else 0
        )
        return (
            used_links,
            resources,
            interference_resources,
            resource_slots,
            protected_work,
            system_profile,
            system_cost_known,
            control_delay_us,
            control_delay_upper_us,
        )

    def _base_candidate_adapter_parameters(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        snapshot: HeterogeneousRuntimeSnapshot,
        residency_variant: str,
        coordinator: RuntimeCompositeExecutorCapability | None,
        phone_power_profiles: tuple,
    ) -> dict[str, object]:
        adapter_parameters = dict(
            coordinator.adapter_parameters
            if isinstance(coordinator, RuntimeCompositeExecutorCapability)
            else self.catalog.executor_by_device[
                pattern.coordinator_device_id
            ].adapter_parameters
        )
        if (
            residency_variant in {"hot", "warm"}
            and pattern.assisted_operator_kind is None
        ):
            resident_dormant_contracts = {
                value
                for device_id in pattern.device_ids
                for residency in (
                    self._matching_residency(
                        manifest, pattern, snapshot, device_id
                    ),
                )
                if residency is not None
                for value in (
                    residency.resident_adapter_parameters.get(
                        _DORMANT_PHONE_FFN_RUNTIME_PARAMETER
                    ),
                )
                if value is not None
            }
            if len(resident_dormant_contracts) > 1 or any(
                type(value) is not str or not value
                for value in resident_dormant_contracts
            ):
                raise RouteGenerationError(
                    "resident dormant FFN runtime contracts differ"
                )
            if resident_dormant_contracts:
                adapter_parameters[
                    _DORMANT_PHONE_FFN_RUNTIME_PARAMETER
                ] = next(iter(resident_dormant_contracts))
        if phone_power_profiles:
            power_models = {
                (
                    row.active_power_mw,
                    row.idle_power_mw,
                    row.evidence_kind,
                    row.estimation_version,
                    row.allow_assumed_for_scheduling,
                )
                for row in phone_power_profiles
            }
            if len(power_models) != 1:
                raise RouteGenerationError(
                    "phone power models differ within one route"
                )
            phone_power = phone_power_profiles[0]
            adapter_parameters.update({
                "phone_power_active_mw": phone_power.active_power_mw,
                "phone_power_allow_assumed": int(
                    phone_power.allow_assumed_for_scheduling
                ),
                "phone_power_device_ids": ",".join(sorted(
                    row.device_id for row in phone_power_profiles
                )),
                "phone_power_evidence_kind": phone_power.evidence_kind,
                "phone_power_estimation_version": (
                    phone_power.estimation_version
                ),
                "phone_power_idle_mw": phone_power.idle_power_mw,
            })
        if pattern.assistance_phase == "decode":
            adapter_parameters["ffn_assistance_phase"] = "decode"
        return adapter_parameters

    def _selected_ffn_geometry(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        adapter_parameters: dict[str, object],
        resident_envelope: _FfnResidentEnvelope | None,
    ) -> tuple[int, int] | None:
        if (
            adapter_parameters.get("ffn_weight_buffer_layout")
                != "selected-width"
            or "phone_device_id" not in adapter_parameters
            or pattern.assisted_operator_kind != "ffn"
        ):
            return None
        if pattern.route_family == "operator_offload":
            selected_columns = manifest.feed_forward_length
        elif pattern.route_family == "operator_split":
            scaled_columns = (
                manifest.feed_forward_length * pattern.split_fraction_ppm
            )
            if scaled_columns % 1_000_000:
                raise RouteGenerationError(
                    "phone FFN split width is not column exact"
                )
            selected_columns = scaled_columns // 1_000_000
        else:
            selected_columns = 0
        helper_device_id = adapter_parameters["phone_device_id"]
        selected_operator_ids = {
            operator_id
            for operator_id, (primary, helper, _fraction)
            in pattern.assignments.items()
            if helper_device_id in {primary, helper}
            and pattern.desktop_assignments.get(operator_id)
                != helper_device_id
        }
        selected_layer_mask = 0
        for operator in manifest.operators:
            if operator.operator_id not in selected_operator_ids:
                continue
            prefix, separator, raw_index = operator.layer_id.partition(":")
            try:
                layer_index = int(raw_index)
            except ValueError as exc:
                raise RouteGenerationError(
                    "phone FFN layer identity is invalid"
                ) from exc
            if (
                prefix != "layer"
                or separator != ":"
                or not 0 <= layer_index < 64
            ):
                raise RouteGenerationError(
                    "phone FFN layer identity is invalid"
                )
            selected_layer_mask |= 1 << layer_index
        minimum_quantum = adapter_parameters.get("ffn_column_quantum")
        selected_quantum = (
            resident_envelope.column_quantum
            if pattern.assistance_phase == "decode"
            and resident_envelope is not None
            else minimum_quantum
        )
        if (
            selected_columns <= 0
            or type(minimum_quantum) is not int
            or minimum_quantum <= 0
            or type(selected_quantum) is not int
            or selected_quantum < minimum_quantum
            or selected_columns % selected_quantum
        ):
            raise RouteGenerationError(
                "phone FFN selected width is not supported"
            )
        if selected_layer_mask == 0:
            raise RouteGenerationError(
                "phone FFN selected layer set is empty"
            )
        adapter_parameters["ffn_selected_columns"] = selected_columns
        adapter_parameters["ffn_selected_layer_mask"] = selected_layer_mask
        if pattern.assistance_phase != "decode":
            adapter_parameters["ffn_column_quantum"] = selected_columns
        return selected_columns, minimum_quantum

    @staticmethod
    def _apply_ffn_partition_parameters(
        pattern: _Pattern,
        coordinator: RuntimeCompositeExecutorCapability | None,
        adapter_parameters: dict[str, object],
        resident_envelope: _FfnResidentEnvelope | None,
        feed_forward_length: int,
        selected_columns: int,
        minimum_quantum: int,
        runtime_partition_error: str | None,
    ) -> str | None:
        if pattern.assistance_phase == "decode" and resident_envelope is not None:
            adapter_parameters.update({
                "ffn_weight_buffer_layout": "resident-superset",
                "ffn_resident_columns": resident_envelope.columns,
                "ffn_resident_layer_mask": resident_envelope.layer_mask,
                "ffn_resident_weight_bytes": resident_envelope.weight_bytes,
                "ffn_resident_geometry_sha256": (
                    resident_envelope.geometry_sha256
                ),
                "ffn_minimum_column_quantum": minimum_quantum,
                "ffn_column_quantum": resident_envelope.column_quantum,
                "ffn_runtime_partition_count": resident_envelope.partition_count,
            })
            if resident_envelope.shards:
                adapter_parameters.update({
                    "phone_session_count": resident_envelope.session_count,
                    "phone_shard_set_geometry_sha256": (
                        resident_envelope.geometry_sha256
                    ),
                    "phone_shard_packing_value": resident_envelope.packing_value,
                    "phone_shard_packing_value_kind": (
                        resident_envelope.packing_value_kind
                    ),
                })
                if resident_envelope.unavailable_session_ids:
                    adapter_parameters["phone_unavailable_session_ids"] = (
                        ",".join(resident_envelope.unavailable_session_ids)
                    )
            co_helpers = co_helper_declaration(adapter_parameters)
            if co_helpers is not None:
                # the server's union mask; the ticket's phone keeps its envelope layers
                adapter_parameters["ffn_resident_layer_mask"] = (
                    resident_envelope.layer_mask | co_helpers.layer_mask
                )
                adapter_parameters[PHONE_HELPERS_PARAMETER] = co_helpers.phone_helpers(
                    adapter_parameters["phone_device_id"],
                    resident_envelope.layer_mask,
                )
            return runtime_partition_error
        runtime_columns = [selected_columns]
        if isinstance(coordinator, RuntimeCompositeExecutorCapability):
            for fraction in coordinator.split_fractions_ppm:
                scaled_columns = feed_forward_length * fraction
                if scaled_columns % 1_000_000:
                    continue
                columns = scaled_columns // 1_000_000
                if (
                    0 < columns <= selected_columns
                    and columns % minimum_quantum == 0
                ):
                    runtime_columns.append(columns)
        runtime_quantum = runtime_columns[0]
        for columns in runtime_columns[1:]:
            runtime_quantum = gcd(runtime_quantum, columns)
        runtime_partitions = selected_columns // runtime_quantum
        maximum_partitions = adapter_parameters.get("ffn_max_runtime_partitions")
        if (
            runtime_quantum < minimum_quantum
            or runtime_quantum % minimum_quantum
            or maximum_partitions is not None
            and (
                type(maximum_partitions) is not int
                or maximum_partitions <= 0
                or runtime_partitions > maximum_partitions
            )
        ):
            runtime_partition_error = (
                "runtime FFN partition capacity is insufficient"
            )
        adapter_parameters.update({
            "ffn_minimum_column_quantum": minimum_quantum,
            "ffn_column_quantum": runtime_quantum,
            "ffn_runtime_partition_count": runtime_partitions,
        })
        return runtime_partition_error

    def _candidate_transport_parameters(
        self,
        profile,
        used_links: tuple[str, ...],
        work: ModelRequestWork,
        pattern: _Pattern,
        adapter_parameters: dict[str, object],
    ) -> str | None:
        required_maximum_payload_bytes = 0
        co_helpers = co_helper_declaration(adapter_parameters)
        if co_helpers is not None:
            # a co-helper's adb forward is not the primary phone's qualified transport
            by_id = {row.link_id: row for row in profile.links}
            used_links = tuple(
                link_id for link_id in used_links
                if by_id[link_id].source_device not in co_helpers.device_ids
                and by_id[link_id].target_device not in co_helpers.device_ids
            )
        if "phone_device_id" in adapter_parameters:
            n_embd = adapter_parameters.get("ffn_n_embd")
            ubatch_size = adapter_parameters.get("ubatch_size")
            wire_element_bytes = adapter_parameters.get(
                "ffn_wire_element_bytes", 2
            )
            if type(n_embd) is int and type(ubatch_size) is int:
                if (
                    type(wire_element_bytes) is not int
                    or wire_element_bytes <= 0
                ):
                    raise RouteGenerationError(
                        "FFN wire element size is invalid"
                    )
                if pattern.assistance_phase == "decode":
                    parallel = adapter_parameters.get("parallel", 1)
                    if type(parallel) is not int or parallel <= 0:
                        raise RouteGenerationError(
                            "decode parallel capacity is invalid"
                        )
                    maximum_tokens = min(ubatch_size, parallel)
                else:
                    input_bucket, _ = request_shape_bucket(
                        work.input_tokens, work.output_tokens
                    )
                    maximum_tokens = min(ubatch_size, input_bucket)
                declared_tokens = adapter_parameters.get("ffn_max_tokens", maximum_tokens)
                if (type(declared_tokens) is not int or declared_tokens < maximum_tokens
                        or declared_tokens > ubatch_size):
                    raise RouteGenerationError("resident FFN batch capacity is invalid")
                maximum_tokens = declared_tokens
                adapter_parameters["ffn_max_tokens"] = maximum_tokens
                if co_helpers is not None and any(
                    row.max_tokens != maximum_tokens for row in co_helpers.helpers
                ):
                    # every client's HELLO must carry the server's maximum tokens
                    return "co-helper phone batch capacity differs"
                if pattern.assistance_phase != "decode":
                    adapter_parameters["ubatch_size"] = maximum_tokens
                required_maximum_payload_bytes = (
                    n_embd * maximum_tokens * wire_element_bytes
                )
        try:
            adapter_parameters.update(self._transport_adapter_parameters(
                profile,
                used_links,
                required_maximum_payload_bytes,
                adapter_parameters.get(
                    "ffn_transport_slot_payload_multiplier", 1
                ),
                adapter_parameters.get("request_io_protocol"),
                adapter_parameters.get("usb_batch_plan"),
                adapter_parameters.get("ffn_max_tokens", 1),
            ))
        except RouteGenerationError as exc:
            return str(exc)
        if REMOTE_RESIDENT_FFN_PARAMETER in adapter_parameters and (
            adapter_parameters.get("ffn_transport") != "functionfs-usb"
            or not adapter_parameters.get("usb_transport_qualification_identity_sha256")
        ):
            return "remote-resident phone transport is not qualified"
        return None

    def _candidate_execution_plan(
        self,
        *,
        route_id: str,
        manifest: ModelManifest,
        pattern: _Pattern,
        coordinator: RuntimeCompositeExecutorCapability | None,
        resident_envelope: _FfnResidentEnvelope | None,
        adapter_parameters: dict[str, object],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        transition_executor_id: str,
        residency_variant: str,
        assignments: tuple[RuntimeOperatorAssignment, ...],
        transitions: tuple,
        resources: tuple[str, ...],
        route_profile,
        resource_slots: Mapping[str, int],
        snapshot=None,
    ) -> tuple:
        desktop_placement_sha256 = self._desktop_placement_sha256(
            manifest, pattern
        )
        overlap_kind = (
            "decode_window_" + pattern.overlap_kind
            if pattern.assistance_phase == "decode"
            else pattern.overlap_kind
        )
        execution_contract = self._execution_contract(
            manifest,
            pattern,
            adapter_parameters,
            resident_envelope,
            snapshot=snapshot,
        )
        if (
            resident_envelope is not None
            and isinstance(coordinator, RuntimeCompositeExecutorCapability)
        ):
            transport_generation = adapter_parameters.get(
                "usb_transport_generation",
                adapter_parameters.get("request_transport"),
            )
            operator_protocol = (
                coordinator.operator_plan_protocol
                if len(pattern.device_ids) > 1
                else "direct-executor-v1"
            )
            phone_shards = execution_contract.phone_shards
            resident_identity = runtime_residency_component_identity_from_parts(
                artifact_sha256=manifest.artifact_sha256,
                resident_shard_geometry_sha256=tuple(sorted(set(
                    tuple(
                        shard.resident_geometry_sha256
                        for shard in phone_shards
                    )
                    + (resident_envelope.geometry_sha256,)
                ))),
                desktop_placement_sha256=desktop_placement_sha256,
                transport_generation=transport_generation,
                operator_protocol=operator_protocol,
                session_resource_ids=tuple(
                    demand.resource_id for demand in memory_demands
                    if demand.kind == "session_residency_constraint"
                ),
                executor_id=transition_executor_id,
                resident_endpoint=coordinator.endpoint,
                phone_shards=phone_shards,
            )
            adapter_parameters["resident_model_identity_sha256"] = (
                resident_identity.identity_sha256
            )
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
        return plan, execution_contract, desktop_placement_sha256, overlap_kind

    def _candidate_binding(
        self,
        *,
        route_id: str,
        manifest: ModelManifest,
        pattern: _Pattern,
        coordinator: RuntimeCompositeExecutorCapability | None,
        resources: tuple[str, ...],
        memory_demands: tuple[RuntimeMemoryDemand, ...],
        transitions: tuple,
        reasons: tuple[str, ...],
        plan,
    ) -> RuntimeExecutorBinding:
        binding_coordinator = (
            self.catalog.executor_by_device[pattern.coordinator_device_id]
            if coordinator is None else coordinator
        )
        return RuntimeExecutorBinding(
            executor_id=binding_coordinator.executor_id,
            route_id=route_id,
            model_id=manifest.model_id,
            artifact_sha256=manifest.artifact_sha256,
            artifact_bytes=manifest.artifact_bytes,
            backend=binding_coordinator.backend,
            resource_ids=resources,
            memory_resource_id=(
                self.catalog.executor_by_device[
                    pattern.device_ids[0]
                ].memory_resource_id
                if len(pattern.device_ids) == 1 else None
            ),
            resident=all(row.residency_satisfied for row in memory_demands),
            ready=not reasons,
            memory_demands=memory_demands,
            queueable=False,
            residency_candidate_id=(
                None if not transitions else transitions[0].transition_id
            ),
            route_family=pattern.route_family,
            eligibility_reasons=reasons,
            participants=self._participants(pattern, coordinator),
            operator_plan_sha256=plan.plan_sha256,
            endpoint=(
                binding_coordinator.endpoint
                if coordinator is not None
                or (
                    len(pattern.device_ids) == 1
                    and pattern.route_family == "whole_model"
                )
                else None
            ),
            operator_plan_protocol=(
                "direct-executor-v1"
                if len(pattern.device_ids) == 1
                else (
                    None
                    if coordinator is None
                    else coordinator.operator_plan_protocol
                )
            ),
        )
