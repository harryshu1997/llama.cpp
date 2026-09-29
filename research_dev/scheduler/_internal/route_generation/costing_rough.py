"""Route costing: rough (pre-refinement) feasibility, cost and visit generation."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence
from ..model_manifest import ModelManifest, ModelRequestWork
from ..placement import TransferLink
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
)
from ..runtime_plan import AutomatedRouteCandidate
from ..runtime_search import RoughPlacementVisit
from .common import _ceil_div, _Pattern


class RouteRoughCostMixin:
    """Route costing: rough (pre-refinement) feasibility, cost and visit generation."""

    @staticmethod
    def _dominated(left: AutomatedRouteCandidate, right: AutomatedRouteCandidate) -> bool:
        if not left.admitted or not right.admitted:
            return False
        left_energy = left.cost.fleet_energy_upper_uj
        right_energy = right.cost.fleet_energy_upper_uj
        if left_energy is None or right_energy is None:
            return False
        memory_resources = set(left.cost.memory_by_resource_bytes) | set(
            right.cost.memory_by_resource_bytes
        )
        left_memory_dominates = all(
            left.cost.memory_by_resource_bytes.get(resource_id, 0)
                <= right.cost.memory_by_resource_bytes.get(resource_id, 0)
            for resource_id in memory_resources
        )
        left_memory_is_lower = any(
            left.cost.memory_by_resource_bytes.get(resource_id, 0)
                < right.cost.memory_by_resource_bytes.get(resource_id, 0)
            for resource_id in memory_resources
        )
        return (
            left.cost.finish_upper_us <= right.cost.finish_upper_us
            and left_energy <= right_energy
            and left_memory_dominates
            and (
                left.cost.finish_upper_us < right.cost.finish_upper_us
                or left_energy < right_energy
                or left_memory_is_lower
            )
        )

    def _rough_phase_rows(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        work: ModelRequestWork,
    ) -> tuple[bool, tuple]:
        coordinator = self._coordinator(pattern)
        full_width_output = (
            isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and coordinator.adapter_parameters.get("split_output_width") == "full"
        )
        phase_rows = (
            tuple((row.tokens, row.invocations, row.operators) for row in work.phases)
            if work.phases
            else ((work.input_tokens + work.output_tokens, 1, work.operators),)
        )
        static_groups = self._prepare_rough_groups(manifest, pattern)
        result = []
        for phase_tokens, invocations, operators in phase_rows:
            by_operator_id = {
                operator.operator_id: operator for operator in operators
            }
            grouped = tuple(
                (
                    by_operator_id[operator_id],
                    primary,
                    helper,
                    fraction,
                    offload_base,
                    repeat,
                )
                for (
                    operator_id,
                    primary,
                    helper,
                    fraction,
                    offload_base,
                    repeat,
                ) in static_groups
            )
            signature = tuple(sorted(
                (
                    operator.kind,
                    operator.compute_ops,
                    operator.memory_bytes,
                    operator.activation_bytes,
                    operator.output_bytes,
                    primary,
                    "" if helper is None else helper,
                    fraction,
                    "" if offload_base is None else offload_base,
                    repeat,
                )
                for operator, primary, helper, fraction, offload_base, repeat
                in grouped
            ))
            result.append((phase_tokens, invocations, grouped, signature))
        return full_width_output, tuple(result)

    def _rough_branch_cost(
        self,
        *,
        device_id: str,
        operator,
        share: int,
        invocations: int,
        work: ModelRequestWork,
        profile,
        kernel_cache: dict[tuple[object, ...], tuple[int, int]],
    ) -> tuple[int, int]:
        ops = _ceil_div(operator.compute_ops * share, 1_000_000)
        memory = _ceil_div(operator.memory_bytes * share, 1_000_000)
        kernel_key = (device_id, operator.kind, invocations, memory, ops)
        cached = kernel_cache.get(kernel_key)
        if cached is not None:
            return cached
        capability = self.catalog.executor_by_device[device_id]
        profile_id, _ = capability.kernel_profile_for(
            operator.kind,
            input_tokens=work.input_tokens,
            output_tokens=work.output_tokens,
            compute_ops=ops,
            memory_bytes=memory,
        )
        kernel = profile.kernels[profile_id].kernel
        compute_us = _ceil_div(ops * 1_000_000, kernel.effective_ops_per_s)
        memory_us = _ceil_div(memory * 1_000_000, kernel.effective_bytes_per_s)
        branch_us = kernel.launch_us * invocations + max(compute_us, memory_us)
        domain = profile.domains[kernel.domain_id].domain
        branch_energy_uj = _ceil_div(
            (kernel.active_power_mw - domain.idle_power_mw) * branch_us,
            1000,
        )
        result = branch_us, branch_energy_uj
        kernel_cache[kernel_key] = result
        return result

    @staticmethod
    def _rough_link_cost(
        *,
        source: str,
        target: str,
        payload: int,
        transfer_invocations: int,
        invocations: int,
        profile,
        links: Mapping[tuple[str, str], tuple[TransferLink, ...]],
    ) -> tuple[int, int]:
        eligible = tuple(
            link for link in links.get((source, target), ())
            if link.ready and link.supports_payload(payload)
        )
        if not eligible:
            return 2**40, 0
        link = min(
            eligible,
            key=lambda row: (
                row.fixed_latency_us
                + _ceil_div(payload * 1_000_000, row.bandwidth_bytes_per_s),
                row.link_id,
            ),
        )
        part = (
            link.fixed_latency_us * transfer_invocations
            + _ceil_div(
                payload * transfer_invocations * 1_000_000,
                link.bandwidth_bytes_per_s,
            )
        )
        energy = (
            link.fixed_dynamic_uj * transfer_invocations
            + _ceil_div(
                payload * invocations * link.dynamic_pj_per_byte,
                1_000_000,
            )
        )
        energy += sum(
            _ceil_div(
                (
                    active_power_mw
                    - profile.domains[domain_id].domain.idle_power_mw
                ) * part,
                1000,
            )
            for domain_id, active_power_mw
            in link.domain_active_power_mw.items()
        )
        return part, energy

    def _rough_transfer_cost(
        self,
        *,
        phase_tokens: int,
        invocations: int,
        operator,
        primary: str,
        helper: str | None,
        fraction: int,
        offload_base: str | None,
        full_width_output: bool,
        coordinator,
        work: ModelRequestWork,
        profile,
        links: Mapping[tuple[str, str], tuple[TransferLink, ...]],
    ) -> tuple[int, int]:
        transfer_base = primary if offload_base is None else offload_base
        transfer_helper = (
            helper
            if helper is not None
            else primary if offload_base is not None else None
        )
        if transfer_helper is None:
            return 0, 0
        transfer_invocations = invocations
        activation_payload = max(1, operator.activation_bytes // invocations)
        output_payload = max(1, operator.output_bytes // invocations)
        maximum_transfer_tokens = (
            coordinator.adapter_parameters.get("ubatch_size")
            if isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and operator.kind == coordinator.assisted_operator_kind
            else None
        )
        if (
            maximum_transfer_tokens is not None
            and type(maximum_transfer_tokens) is int
            and work.phases
            and invocations == 1
        ):
            transfer_invocations = _ceil_div(
                phase_tokens, maximum_transfer_tokens
            )
            transfer_tokens = min(phase_tokens, maximum_transfer_tokens)
            activation_payload = (
                operator.activation_bytes // phase_tokens * transfer_tokens
            )
            output_payload = operator.output_bytes // phase_tokens * transfer_tokens
        if helper is not None and not full_width_output:
            output_payload = max(1, self._fraction(output_payload, fraction)[1])
        latency_us = 0
        energy_uj = 0
        for source, target, payload in (
            (transfer_base, transfer_helper, activation_payload),
            (transfer_helper, transfer_base, output_payload),
        ):
            part_us, part_energy_uj = self._rough_link_cost(
                source=source,
                target=target,
                payload=payload,
                transfer_invocations=transfer_invocations,
                invocations=invocations,
                profile=profile,
                links=links,
            )
            latency_us += part_us
            energy_uj += part_energy_uj
        return latency_us, energy_uj

    def _rough_operator_cost(
        self,
        *,
        phase_tokens: int,
        invocations: int,
        row: tuple,
        full_width_output: bool,
        coordinator,
        work: ModelRequestWork,
        profile,
        kernel_cache: dict[tuple[object, ...], tuple[int, int]],
        links: Mapping[tuple[str, str], tuple[TransferLink, ...]],
    ) -> tuple[int, int, int]:
        operator, primary, helper, fraction, offload_base, repeat = row
        branches = (
            ((primary, 1_000_000),)
            if helper is None
            else ((primary, 1_000_000 - fraction), (helper, fraction))
        )
        branch_costs = tuple(
            self._rough_branch_cost(
                device_id=device_id,
                operator=operator,
                share=share,
                invocations=invocations,
                work=work,
                profile=profile,
                kernel_cache=kernel_cache,
            )
            for device_id, share in branches
        )
        transfer_us, transfer_energy_uj = self._rough_transfer_cost(
            phase_tokens=phase_tokens,
            invocations=invocations,
            operator=operator,
            primary=primary,
            helper=helper,
            fraction=fraction,
            offload_base=offload_base,
            full_width_output=full_width_output,
            coordinator=coordinator,
            work=work,
            profile=profile,
            links=links,
        )
        operator_us = max(row[0] for row in branch_costs) + transfer_us
        operator_energy_uj = sum(row[1] for row in branch_costs)
        operator_energy_uj += transfer_energy_uj
        return max(1, operator_us) * repeat, operator_energy_uj * repeat, repeat

    def _rough_cost(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        work: ModelRequestWork,
        profile,
        profile_identity: str,
        kernel_cache: dict[tuple[object, ...], tuple[int, int]],
        links: Mapping[tuple[str, str], tuple[TransferLink, ...]],
    ) -> tuple[int, int]:
        coordinator = self._coordinator(pattern)
        full_width_output, grouped_phase_rows = self._rough_phase_rows(
            manifest, pattern, work
        )
        cache_key = (
            profile_identity,
            work.input_tokens,
            work.output_tokens,
            getattr(coordinator, "executor_id", None),
            full_width_output,
            tuple(
                (phase_tokens, invocations, signature)
                for phase_tokens, invocations, _, signature
                in grouped_phase_rows
            ),
        )
        cached_cost = self._rough_cost_cache.get(cache_key)
        if cached_cost is not None:
            return cached_cost
        latency_us = 0
        energy_uj = 0
        for phase_tokens, invocations, grouped, _ in grouped_phase_rows:
            for row in grouped:
                row_latency_us, row_energy_uj, _ = self._rough_operator_cost(
                    phase_tokens=phase_tokens,
                    invocations=invocations,
                    row=row,
                    full_width_output=full_width_output,
                    coordinator=coordinator,
                    work=work,
                    profile=profile,
                    kernel_cache=kernel_cache,
                    links=links,
                )
                latency_us += row_latency_us
                energy_uj += row_energy_uj
        energy_uj += sum(
            _ceil_div(
                profile.domains[domain_id].domain.idle_power_mw
                * latency_us,
                1000,
            )
            for domain_id in profile.idle_charge_domains
        )
        result = (max(1, latency_us), max(1, energy_uj))
        if len(self._rough_cost_cache) >= 4_096:
            self._rough_cost_cache.pop(next(iter(self._rough_cost_cache)))
        self._rough_cost_cache[cache_key] = result
        return result

    def _rough_memory_feasible(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
        profile,
    ) -> bool:
        cache_key = (manifest.artifact_sha256, pattern.route_key)
        cached = self._rough_memory_cache.get(cache_key)
        if cached is not None:
            return cached
        tensors = manifest.tensor_by_id
        allocations: dict[str, dict[str, int]] = {}
        coordinator = (
            self.catalog.composite_executor_by_id[
                pattern.coordinator_executor_id
            ]
            if pattern.coordinator_executor_id is not None else None
        )
        resident_envelope = (
            pattern.phone_resident_envelope
            if coordinator is not None
            and pattern.assistance_phase == "decode"
            else None
        )
        for operator in manifest.operators:
            primary, helper, fraction = pattern.assignments[
                operator.operator_id
            ]
            for tensor_id in operator.tensor_ids:
                nbytes = tensors[tensor_id].nbytes
                if pattern.assistance_phase == "decode":
                    desktop = pattern.desktop_assignments[
                        operator.operator_id
                    ]
                    allocations.setdefault(desktop, {})[tensor_id] = max(
                        allocations.setdefault(desktop, {}).get(tensor_id, 0),
                        nbytes,
                    )
                    continue
                primary_bytes = nbytes
                if helper is not None:
                    helper_bytes = _ceil_div(
                        nbytes * fraction, 1_000_000
                    )
                    primary_bytes = nbytes - helper_bytes
                    allocations.setdefault(helper, {})[tensor_id] = max(
                        allocations.setdefault(helper, {}).get(tensor_id, 0),
                        helper_bytes,
                    )
                allocations.setdefault(primary, {})[tensor_id] = max(
                    allocations.setdefault(primary, {}).get(tensor_id, 0),
                    primary_bytes,
                )
        if resident_envelope is not None:
            assert coordinator is not None
            helper_id = coordinator.helper_device_id
            assert helper_id is not None
            allocations.setdefault(helper_id, {})[
                "resident-envelope:" + resident_envelope.geometry_sha256
            ] = resident_envelope.weight_bytes
        for device_id, rows in allocations.items():
            device = profile.devices[device_id]
            pool = profile.memory_pools[device.memory_pool_id]
            limit = min(
                device.allocation_limit_bytes,
                pool.capacity_bytes - pool.reserved_bytes,
            )
            if sum(rows.values()) > limit:
                self._rough_memory_cache[cache_key] = False
                return False
        if coordinator is not None:
            helper_limit = coordinator.adapter_parameters.get(
                "maximum_helper_resident_weight_bytes"
            )
            helper_id = coordinator.helper_device_id
            if (
                type(helper_limit) is int
                and helper_id is not None
                and not (
                    resident_envelope is not None
                    and resident_envelope.shards
                )
                and sum(allocations.get(helper_id, {}).values())
                    > helper_limit
            ):
                self._rough_memory_cache[cache_key] = False
                return False
        if resident_envelope is not None and resident_envelope.shards:
            helper = self.catalog.executor_by_device[
                coordinator.helper_device_id
            ]
            session_by_id = {
                row.session_id: row for row in helper.phone_sessions
            }
            if any(
                shard.resident_bytes
                    > session_by_id[
                        shard.session_id
                    ].resident_memory_limit_bytes
                for shard in resident_envelope.shards
            ):
                self._rough_memory_cache[cache_key] = False
                return False
        if len(self._rough_memory_cache) >= 4_096:
            self._rough_memory_cache.pop(next(iter(self._rough_memory_cache)))
        self._rough_memory_cache[cache_key] = True
        return True

    def _kernel_signature(
        self,
        pattern: _Pattern,
        work: ModelRequestWork,
    ) -> tuple[tuple[str, ...], ...]:
        rows = []
        for operator in work.operators:
            primary, helper, fraction = pattern.assignments[
                operator.operator_id
            ]
            if helper is None:
                branches = ((primary, 1_000_000),)
            else:
                branches = (
                    (primary, 1_000_000 - fraction),
                    (helper, fraction),
                )
            profiles = []
            for device_id, share in branches:
                capability = self.catalog.executor_by_device[device_id]
                profile_id, _ = capability.kernel_profile_for(
                    operator.kind,
                    input_tokens=work.input_tokens,
                    output_tokens=work.output_tokens,
                    compute_ops=_ceil_div(
                        operator.compute_ops * share, 1_000_000
                    ),
                    memory_bytes=_ceil_div(
                        operator.memory_bytes * share, 1_000_000
                    ),
                )
                profiles.append(profile_id)
            rows.append(tuple(profiles))
        return tuple(rows)

    def _required_group(self, pattern: _Pattern) -> str | None:
        if pattern.coordinator_executor_id is not None:
            group = "physical-coordinator:" + pattern.coordinator_executor_id
            if pattern.resident_envelope:
                coordinator = self.catalog.composite_executor_by_id[
                    pattern.coordinator_executor_id
                ]
                return (
                    "resident-envelope:parent:"
                    + str(coordinator.baseline_executor_id)
                    + ":sessions:"
                    + str(pattern.phone_session_count)
                    + ":batch:"
                    + str(coordinator.adapter_parameters.get(
                        "usb_batch_plan", "none"
                    ))
                    + ":transport:"
                    + str(coordinator.adapter_parameters.get(
                        "request_transport_generation", "default"
                    ))
                )
            coordinator = self.catalog.composite_executor_by_id[
                pattern.coordinator_executor_id
            ]
            helper_id = coordinator.helper_device_id
            if helper_id is None:
                return group
            assisted_count = sum(
                primary == helper_id or helper == helper_id
                for primary, helper, _ in pattern.assignments.values()
            )
            if assisted_count <= 0:
                return group
            scale = 1 << (assisted_count - 1).bit_length()
            return group + ":scale:" + str(scale)
        kinds = {
            device_id: self.catalog.placement_profile.devices[device_id].kind
            for device_id in pattern.device_ids
        }
        phone_ids = {
            device_id for device_id, kind in kinds.items()
            if kind == "phone"
        }
        if pattern.route_family == "whole_model":
            return "whole:" + pattern.device_ids[0]
        if pattern.route_family == "layer_placement":
            return "layers:" + "+".join(pattern.device_ids)
        if not phone_ids:
            return None
        if self.catalog.placement_profile.devices[
            pattern.coordinator_device_id
        ].kind == "phone":
            return None
        if pattern.route_family == "operator_offload":
            if pattern.assisted_operator_kind == "attention_projection":
                return None
            group = (
                "phone-offload:" + str(pattern.assisted_operator_kind)
            )
        elif (
            pattern.route_family == "operator_split"
            and pattern.assisted_operator_kind == "ffn"
        ):
            if pattern.split_axis == "column":
                group = (
                    "phone-ffn-column:"
                    + pattern.coordinator_device_id
                    + ":"
                    + str(pattern.split_fraction_ppm)
                )
            elif pattern.split_axis == "row":
                group = "phone-ffn-axis:row"
            else:
                return None
        elif (
            pattern.route_family == "operator_split"
            and pattern.assisted_operator_kind == "lm_head"
            and pattern.split_axis in {"row", "tensor"}
        ):
            group = "phone-non-ffn-axis:" + pattern.split_axis
        else:
            return None
        return group

    def _rough_visit(
        self,
        *,
        manifest: ModelManifest,
        pattern: _Pattern,
        work: ModelRequestWork,
        profile,
        profile_identity: str,
        kernel_cache: dict[tuple[object, ...], tuple[int, int]],
        links: Mapping[tuple[str, str], tuple[TransferLink, ...]],
        busy_until_by_resource: Mapping[str, int],
        observed_at_us: int,
        idle_power_mw: int,
        desktop_control,
        fallback_key: str,
    ) -> tuple[RoughPlacementVisit, bool, str | None]:
        latency_us, energy_uj = self._rough_cost(
            manifest,
            pattern,
            work,
            profile,
            profile_identity,
            kernel_cache,
            links,
        )
        coordinator = self._coordinator(pattern)
        resource_ids = {
            resource_id
            for device_id in pattern.device_ids
            for resource_id in self.catalog.executor_by_device[
                device_id
            ].execution_resource_ids
        }
        if isinstance(coordinator, RuntimeCompositeExecutorCapability):
            resource_ids.update(coordinator.resource_ids)
        queue_delay_us = max(
            (
                busy_until_by_resource.get(resource_id, observed_at_us)
                - observed_at_us
                for resource_id in resource_ids
            ),
            default=0,
        )
        queue_delay_us = max(0, queue_delay_us)
        latency_us += queue_delay_us
        energy_uj += _ceil_div(idle_power_mw * queue_delay_us, 1000)
        coverage_group = "devices:" + "+".join(sorted(
            self.catalog.placement_profile.devices[device_id].kind
            for device_id in pattern.device_ids
        ))
        required_group = self._required_group(pattern)
        is_desktop_control = (
            desktop_control is not None
            and coordinator is not None
            and coordinator.executor_id == desktop_control.executor_id
            and all(
                self.catalog.placement_profile.devices[device_id].kind
                in {"cpu", "gpu"}
                for device_id in pattern.device_ids
            )
            and self._desktop_placement_sha256(manifest, pattern)
            == desktop_control.placement_sha256
        )
        return (
            RoughPlacementVisit(
                route_key=pattern.route_key,
                residency_variant="hot",
                rough_latency_us=latency_us,
                rough_energy_uj=energy_uj,
                required_group=required_group,
                coverage_group=coverage_group,
                rough_memory_feasible=self._rough_memory_feasible(
                    manifest, pattern, profile
                ),
                mandatory=pattern.route_key == fallback_key,
            ),
            is_desktop_control,
            required_group,
        )

    def _rough_visits(
        self,
        manifest: ModelManifest,
        patterns: Sequence[_Pattern],
        work: ModelRequestWork,
        profile,
        profile_identity: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> tuple[RoughPlacementVisit, ...]:
        rows = []
        fallback_key = f"whole:{self.catalog.fallback.device_id}"
        desktop_control = self.catalog.desktop_control_by_artifact.get(
            manifest.artifact_sha256
        )
        desktop_control_visits: list[RoughPlacementVisit] = []
        required_session_groups: set[str] = set()
        selected_portfolio_visit_ids: set[str] = set()
        adaptive_split_visit_ids: set[str] = set()
        kernel_cache: dict[tuple[object, ...], tuple[int, int]] = {}
        links: dict[tuple[str, str], list[TransferLink]] = {}
        for row in profile.links:
            links.setdefault(
                (row.source_device, row.target_device), []
            ).append(row)
        immutable_links = {
            key: tuple(value) for key, value in links.items()
        }
        busy_until_by_resource = dict(snapshot.busy_until_by_resource(
            self.catalog
        ))
        for resource_id, state in self.timeline.resource_snapshot(
            observed_at_us
        ).items():
            busy_until_by_resource[resource_id] = max(
                busy_until_by_resource.get(resource_id, observed_at_us),
                # Rough ranking is a lower bound; exact phase leases follow.
                int(state["reserved_until_us"] if state["next_free_us"] is None
                    else state["next_free_us"]),
            )
        idle_power_mw = sum(
            profile.domains[domain_id].domain.idle_power_mw
            for domain_id in profile.idle_charge_domains
        )
        for pattern in patterns:
            visit, is_desktop_control, required_group = self._rough_visit(
                manifest=manifest,
                pattern=pattern,
                work=work,
                profile=profile,
                profile_identity=profile_identity,
                kernel_cache=kernel_cache,
                links=immutable_links,
                busy_until_by_resource=busy_until_by_resource,
                observed_at_us=observed_at_us,
                idle_power_mw=idle_power_mw,
                desktop_control=desktop_control,
                fallback_key=fallback_key,
            )
            rows.append(visit)
            coordinator = self._coordinator(pattern)
            if (
                pattern.resident_envelope
                and desktop_control is not None
                and isinstance(coordinator, RuntimeCompositeExecutorCapability)
                and coordinator.baseline_executor_id == desktop_control.executor_id
                and required_group is not None
            ):
                required_session_groups.add(required_group)
                if (
                    getattr(pattern, "route_family", None) == "operator_split"
                    and getattr(pattern, "assisted_operator_kind", None) == "ffn"
                ):
                    adaptive_split_visit_ids.add(visit.visit_id)
                if (
                    self._phone_residency_layout is not None
                    and pattern.phone_resident_envelope is not None
                    and pattern.phone_resident_envelope.geometry_sha256
                        == self._phone_residency_layout.geometry_sha256
                ):
                    selected_portfolio_visit_ids.add(visit.visit_id)
            if is_desktop_control:
                desktop_control_visits.append(visit)
            if pattern.route_key == fallback_key:
                for state in ("warm", "cold"):
                    rows.append(RoughPlacementVisit(
                        route_key=pattern.route_key,
                        residency_variant=state,
                        rough_latency_us=visit.rough_latency_us,
                        rough_energy_uj=visit.rough_energy_uj,
                        required_group="fallback-residency:" + state,
                        coverage_group=visit.coverage_group,
                        rough_memory_feasible=self._rough_memory_feasible(
                            manifest, pattern, profile
                        ),
                        mandatory=True,
                    ))
        mandatory_ids = {
            min(
                (
                    row for row in rows
                    if row.required_group == group
                ),
                # The adaptive controller probes only the split envelope of a group; an
                # offload visit of the same envelope stays a regular group candidate.
                key=lambda row: (
                    row.visit_id not in selected_portfolio_visit_ids,
                    row.visit_id not in adaptive_split_visit_ids,
                    not row.rough_memory_feasible,
                    row.rough_energy_uj,
                    row.rough_latency_us,
                    row.visit_id,
                ),
            ).visit_id
            for group in required_session_groups
        }
        if desktop_control_visits:
            mandatory_ids.add(min(
                desktop_control_visits,
                key=lambda row: (
                    not row.rough_memory_feasible,
                    row.rough_energy_uj,
                    row.rough_latency_us,
                    row.visit_id,
                ),
            ).visit_id)
        return tuple(
            replace(row, mandatory=True)
            if row.visit_id in mandatory_ids
            else row
            for row in rows
        )
