"""Route template sets: compile, cache, frontier preparation, desktop control, materialize, rerank.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import replace
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from ..model_manifest import ModelManifest, ModelRequestWork
from ..capability_contracts.executors import whole_phone_launch_parameters
from ..placement import (
    HierarchicalPlacementPlanner,
    OperatorNode,
    PlacementError,
    PlacementHardwareProfile,
)
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCompositeExecutorCapability,
)
from ..runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeOperatorAssignment,
)
from ..runtime_residency_cohorts import (
    RuntimeResidencyCohortHold,
    RuntimeResidencyReuseProjection,
    runtime_residency_component_identity,
)
from ..runtime_search import (
    RoughPlacementFrontier,
    RoughPlacementVisit,
    RuntimeSearchError,
    request_shape_bucket,
)
from ..types import canonical_sha256
from .common import (
    RouteGenerationError,
    DesktopControlUnavailableError,
    RuntimeRouteTemplateSet,
    _Pattern,
)


class RouteTemplateMixin:
    """Route template sets: compile, cache, frontier preparation, desktop control, materialize, rerank."""

    @staticmethod
    def _route_key(candidate_id: str) -> str:
        prefix = "auto:"
        marker = ":residency:"
        if not candidate_id.startswith(prefix) or marker not in candidate_id:
            raise RouteGenerationError(
                "automated candidate route identity is invalid"
            )
        route_key, residency = candidate_id[len(prefix):].rsplit(
            marker, 1
        )
        if not route_key or residency not in {"cold", "hot", "warm"}:
            raise RouteGenerationError(
                "automated candidate residency identity is invalid"
            )
        return route_key

    def compile_route_template_set(
        self,
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        manifest: ModelManifest,
        *,
        input_token_bucket: int,
        output_token_bucket: int,
        quality_requirement: str,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> RuntimeRouteTemplateSet:
        if (
            not isinstance(candidate_set, AutomatedCandidateSet)
            or not any(
                row.candidate_id == selected.candidate_id
                for row in candidate_set.candidates
            )
            or not isinstance(manifest, ModelManifest)
            or manifest.model_id != candidate_set.model_id
            or type(input_token_bucket) is not int
            or input_token_bucket < 1
            or type(output_token_bucket) is not int
            or output_token_bucket < 1
            or type(quality_requirement) is not str
            or not quality_requirement
        ):
            raise RouteGenerationError(
                "runtime route template input is invalid"
            )
        component = runtime_residency_component_identity(
            manifest.artifact_sha256, selected.plan, selected.binding
        )
        route_key = self._route_key(selected.candidate_id)
        stored_candidate_set = replace(
            candidate_set,
            candidates=tuple(
                selected
                if row.candidate_id == selected.candidate_id else row
                for row in candidate_set.candidates
            ),
        )
        static_plan = {
            "assisted_operator_kind": selected.plan.assisted_operator_kind,
            "baseline_executor_id": selected.plan.baseline_executor_id,
            "desktop_placement_sha256": (
                selected.plan.desktop_placement_sha256
            ),
            "device_ids": selected.plan.device_ids,
            "execution_contract": (
                selected.plan.execution_contract.to_json()
            ),
            "executor_id": selected.binding.executor_id,
            "operator_assignments": tuple(
                (
                    row.operator_id,
                    row.operator_kind,
                    row.device_ids,
                    row.split_axis,
                    row.split_fraction_ppm,
                    row.kernel_profile_ids,
                )
                for row in selected.plan.operators
            ),
            "operator_plan_protocol": (
                selected.binding.operator_plan_protocol
            ),
            "quality_requirement": quality_requirement,
            "resource_ids": selected.plan.resource_ids,
            "route_family": selected.plan.route_family,
            "route_key": route_key,
            "route_profile_id": selected.plan.route_profile_id,
            "schema": "runtime-route-template-v1",
            "selected_component_identity_sha256": (
                component.identity_sha256
            ),
            "split_axis": selected.plan.split_axis,
            "split_fraction_ppm": selected.plan.split_fraction_ppm,
        }
        route_template_identity = canonical_sha256(static_plan)
        audit_sha256 = canonical_sha256({
            "candidate_generation_sha256": (
                stored_candidate_set.generation_sha256
            ),
            "candidate_ids": tuple(
                row.candidate_id
                for row in stored_candidate_set.candidates
            ),
            "operator_plan_sha256s": tuple(
                row.plan.plan_sha256
                for row in stored_candidate_set.candidates
            ),
            "schema": "runtime-route-template-audit-v1",
        })
        return RuntimeRouteTemplateSet(
            artifact_sha256=manifest.artifact_sha256,
            input_token_bucket=input_token_bucket,
            output_token_bucket=output_token_bucket,
            quality_requirement=quality_requirement,
            selected_route_id=selected.candidate_id,
            selected_route_key=route_key,
            selected_component_identity_sha256=(
                component.identity_sha256
            ),
            route_template_identity_sha256=(
                route_template_identity
            ),
            candidate_set=stored_candidate_set,
            audit_sha256=audit_sha256,
            live_state_sha256=(
                None
                if snapshot is None
                else self.route_template_live_state_sha256(snapshot)
            ),
        )

    def route_template_live_state_sha256(
        self, snapshot: HeterogeneousRuntimeSnapshot
    ) -> str:
        """Hash live feasibility inputs while excluding queue and memory state."""
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise RouteGenerationError(
                "runtime route template snapshot is invalid"
            )
        executor_rows = []
        for executor_id, state in snapshot.executors.items():
            capability = self.catalog.executor_by_id.get(executor_id)
            if capability is None:
                coordinator = self.catalog.composite_executor_by_id.get(
                    executor_id
                )
                maximum_temperature_millic = None
                maximum_thermal_status = None
                minimum_battery_ppm = None
                if coordinator is not None:
                    participants = tuple(
                        self.catalog.executor_by_device[device_id]
                        for device_id in coordinator.participant_device_ids
                    )
                    maximum_temperature_millic = min(
                        row.maximum_temperature_millic
                        for row in participants
                    )
                    maximum_thermal_status = min(
                        row.maximum_thermal_status for row in participants
                    )
                    minimum_battery_ppm = max(
                        self._minimum_battery_ppm(row.device_id)
                        for row in participants
                    )
            else:
                maximum_temperature_millic = (
                    capability.maximum_temperature_millic
                )
                maximum_thermal_status = capability.maximum_thermal_status
                minimum_battery_ppm = self._minimum_battery_ppm(
                    capability.device_id
                )
            thermal_qualified = (
                state.thermal_qualified
                if maximum_thermal_status is None
                else state.thermal_qualified_under(maximum_thermal_status)
            )
            executor_rows.append((
                executor_id,
                state.healthy,
                state.ready,
                state.free_slots > 0,
                (
                    thermal_qualified
                    if thermal_qualified is not None
                    else maximum_temperature_millic is not None
                    and state.temperature_millic
                        <= maximum_temperature_millic
                ),
                (
                    minimum_battery_ppm is None
                    or state.battery_ppm >= minimum_battery_ppm
                ),
            ))
        peak_executors = {
            row.executor_id for row in self.catalog.executors
            if row.adapter_parameters.get("execution_adapter") == "android-llama-server-v1"
            and row.adapter_parameters.get("persistent_residency") == 1
            and "whole_model_peak_memory_bytes" in row.adapter_parameters
        }
        return canonical_sha256({
            "cost_features": dict(snapshot.cost_features),
            "executors": tuple(executor_rows),
            "links": tuple(
                (
                    link_id,
                    state.ready,
                    state.measured_bandwidth_bytes_per_s,
                )
                for link_id, state in snapshot.links.items()
            ),
            "protected_work": (
                None
                if snapshot.protected_work is None
                else snapshot.protected_work.to_json()
            ),
            "residency": tuple(
                (
                    row.model_id,
                    row.artifact_sha256,
                    row.device_id,
                    row.state,
                    row.resident_bytes,
                    row.generation,
                    row.executor_id,
                    row.reclaimable_bytes,
                    row.resident_geometry_sha256,
                ) + ((tuple(sorted(whole_phone_launch_parameters(row.resident_adapter_parameters).items())),
                      row.resident_tensor_ids)
                     if row.executor_id in peak_executors else ())
                for row in snapshot.residency
            ),
            "schema": "runtime-route-template-live-state-v1",
        })

    def _bounded_cache_store(
        self,
        cache: dict,
        key: object,
        value: object,
        maximum_entries: int,
    ) -> object:
        with self._cache_lock:
            if key not in cache and len(cache) >= maximum_entries:
                cache.pop(next(iter(cache)))
                self._static_cache_evictions += 1
            cache[key] = value
        return value

    def _cached_placement(
        self,
        key: tuple[object, ...],
        builder: Callable[[], tuple[
            object,
            str | None,
            tuple[OperatorNode, ...],
            tuple[RuntimeOperatorAssignment, ...],
        ]],
    ) -> tuple[
        tuple[
            object,
            str | None,
            tuple[OperatorNode, ...],
            tuple[RuntimeOperatorAssignment, ...],
        ],
        bool,
    ]:
        while True:
            with self._cache_lock:
                cached = self._placement_cache.get(key)
                if cached is not None:
                    return cached, True
                inflight = self._placement_inflight.get(key)
                if inflight is None:
                    inflight = threading.Event()
                    self._placement_inflight[key] = inflight
                    break
                self._placement_singleflight_waits += 1
            inflight.wait()
        try:
            value = builder()
        except BaseException:
            with self._cache_lock:
                completed = self._placement_inflight.pop(key, None)
                if completed is not None:
                    completed.set()
            raise
        with self._cache_lock:
            if len(self._placement_cache) >= 4_096:
                self._placement_cache.pop(next(iter(
                    self._placement_cache
                )))
                self._placement_cache_evictions += 1
            self._placement_cache[key] = value
            self._placement_cache_misses += 1
            self._placement_cache_generations += 1
            completed = self._placement_inflight.pop(key, None)
            if completed is not None:
                completed.set()
        return value, False

    def prepare_manifest(self, manifest: ModelManifest) -> int:
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError("model manifest is invalid")
        existing_model_id = self._model_id_by_artifact.get(
            manifest.artifact_sha256
        )
        if (
            existing_model_id is not None
            and existing_model_id != manifest.model_id
        ):
            raise RouteGenerationError(
                "model artifact has multiple model identities"
            )
        patterns = self._patterns(manifest)
        for pattern in patterns:
            self._prepare_rough_groups(manifest, pattern)
            self._rough_memory_feasible(
                manifest, pattern, self.catalog.placement_profile
            )
        self._model_id_by_artifact[
            manifest.artifact_sha256
        ] = manifest.model_id
        return len(patterns)

    def prepare_frontier(
        self,
        manifest: ModelManifest,
        input_tokens: int,
        output_tokens: int,
        quality_requirement: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        *,
        force_refresh: bool = False,
    ):
        """Compile one bucket frontier without entering request admission."""
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError("model manifest is invalid")
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise RouteGenerationError("runtime system snapshot is invalid")
        if (
            type(input_tokens) is not int
            or input_tokens <= 0
            or type(output_tokens) is not int
            or output_tokens <= 0
            or type(quality_requirement) is not str
            or not quality_requirement
            or not quality_requirement.isascii()
            or type(observed_at_us) is not int
            or observed_at_us < 0
            or type(force_refresh) is not bool
        ):
            raise RouteGenerationError("frontier planning input is invalid")
        snapshot.validate_at(observed_at_us)
        input_bucket, output_bucket = request_shape_bucket(
            input_tokens, output_tokens
        )
        frontier_key = (
            manifest.artifact_sha256,
            self._capability_generation_sha256,
            input_bucket,
            output_bucket,
            quality_requirement,
        )
        while True:
            if not force_refresh:
                cached = self._rough_compiler.lookup(
                    artifact_sha256=manifest.artifact_sha256,
                    capability_generation_sha256=(
                        self._capability_generation_sha256
                    ),
                    input_tokens=input_bucket,
                    output_tokens=output_bucket,
                    quality_requirement=quality_requirement,
                )
                if cached is not None:
                    return cached
            with self._cache_lock:
                inflight = self._frontier_inflight.get(frontier_key)
                if inflight is None:
                    inflight = threading.Event()
                    self._frontier_inflight[frontier_key] = inflight
                    self._frontier_generations += 1
                    break
                self._frontier_singleflight_waits += 1
            inflight.wait()
            force_refresh = False
        try:
            work = manifest.request_work(input_bucket, output_bucket)
            profile = self._effective_cost_profile(snapshot)
            profile = replace(
                profile,
                devices={
                    device_id: replace(device, ready=True)
                    for device_id, device in profile.devices.items()
                },
                links=tuple(
                    replace(link, ready=True) for link in profile.links
                ),
            )
            profile_sha256 = canonical_sha256(profile)
            patterns = self._patterns(manifest)
            try:
                frontier, _ = self._rough_compiler.compile(
                    artifact_sha256=manifest.artifact_sha256,
                    capability_generation_sha256=(
                        self._capability_generation_sha256
                    ),
                    input_tokens=input_bucket,
                    output_tokens=output_bucket,
                    quality_requirement=quality_requirement,
                    visits=self._rough_visits(
                        manifest,
                        patterns,
                        work,
                        profile,
                        profile_sha256,
                        snapshot,
                        observed_at_us,
                    ),
                    replace_existing=force_refresh,
                )
            except RuntimeSearchError as exc:
                raise RouteGenerationError(str(exc)) from exc
            return frontier
        finally:
            with self._cache_lock:
                completed = self._frontier_inflight.pop(
                    frontier_key, None
                )
                if completed is not None:
                    completed.set()

    def _prepare_rough_groups(
        self,
        manifest: ModelManifest,
        pattern: _Pattern,
    ) -> tuple[tuple[str, str, str | None, int, str | None, int], ...]:
        cache_key = (manifest.artifact_sha256, pattern.route_key)
        cached = self._rough_group_cache.get(cache_key)
        if cached is not None:
            return cached
        coordinator = self._coordinator(pattern)
        offload_baseline_by_id = {}
        if (
            isinstance(coordinator, RuntimeCompositeExecutorCapability)
            and coordinator.route_family == "operator_offload"
            and coordinator.baseline_executor_id is not None
        ):
            baseline = self.catalog.composite_executor_by_id[
                coordinator.baseline_executor_id
            ]
            offload_baseline_by_id = {
                row.operator_id: row.primary_device_id
                for row in baseline.operator_placements
            }
        tensor_by_id = manifest.tensor_by_id
        grouped: dict[
            tuple[object, ...],
            tuple[str, str, str | None, int, str | None, int],
        ] = {}
        for operator in manifest.operators:
            primary, helper, fraction = pattern.assignments[
                operator.operator_id
            ]
            offload_base = (
                offload_baseline_by_id.get(operator.operator_id)
                if helper is None
                and primary == getattr(coordinator, "helper_device_id", None)
                else None
            )
            tensors = tuple(
                tensor_by_id[tensor_id] for tensor_id in operator.tensor_ids
            )
            group_key = (
                operator.kind,
                sum(
                    tensor.elements for tensor in tensors
                    if len(tensor.shape) >= 2
                ),
                sum(tensor.nbytes for tensor in tensors),
                primary,
                helper,
                fraction,
                offload_base,
            )
            previous = grouped.get(group_key)
            if previous is None:
                grouped[group_key] = (
                    operator.operator_id,
                    primary,
                    helper,
                    fraction,
                    offload_base,
                    1,
                )
            else:
                grouped[group_key] = previous[:-1] + (previous[-1] + 1,)
        result = tuple(grouped.values())
        if len(self._rough_group_cache) >= 4_096:
            self._rough_group_cache.pop(next(iter(self._rough_group_cache)))
        self._rough_group_cache[cache_key] = result
        return result

    def last_generation_timing(self) -> Mapping[str, int]:
        return self._last_generation_timing

    @staticmethod
    def _desktop_cost_only_rejections() -> frozenset[str]:
        return frozenset({
            "ENERGY_UNKNOWN",
            "MARGINAL_SYSTEM_COST_UNKNOWN",
            "ROUTE_NOT_QUALIFIED",
            "SLO_UPPER_BOUND",
            "TRANSITION_PROFILE_QUARANTINED",
        })

    def _desktop_control(
        self,
        rows: Sequence[AutomatedRouteCandidate],
        manifest: ModelManifest,
    ) -> AutomatedRouteCandidate:
        physically_qualified = [
            row for row in rows
            if all(
                self.catalog.placement_profile.devices[device_id].kind
                in {"cpu", "gpu"}
                for device_id in row.device_ids
            )
            and self._physically_qualified_desktop(row)
            and set(row.rejection_reasons).issubset(
                self._desktop_cost_only_rejections()
            )
        ]
        profile = self.catalog.desktop_control_by_artifact.get(
            manifest.artifact_sha256
        )
        if profile is not None:
            if profile.maturity != "QUALIFIED":
                raise RouteGenerationError(
                    "desktop control profile is not qualified"
                )
            matching = [
                row for row in physically_qualified
                if row.binding.executor_id == profile.executor_id
                and row.plan.desktop_placement_sha256
                    == profile.placement_sha256
            ]
            if len(matching) != 1:
                details = ";".join(
                    row.candidate_id
                    + ":executor=" + row.binding.executor_id
                    + ":placement="
                    + str(row.plan.desktop_placement_sha256)
                    + ":maturity=" + row.maturity
                    + ":physical="
                    + str(self._physically_qualified_desktop(row))
                    + ":rejections="
                    + ",".join(row.rejection_reasons or ("none",))
                    for row in rows
                    if all(
                        self.catalog.placement_profile.devices[
                            device_id
                        ].kind in {"cpu", "gpu"}
                        for device_id in row.device_ids
                    )
                )
                raise DesktopControlUnavailableError(
                    "qualified desktop control was not generated"
                    + "; expected_executor=" + profile.executor_id
                    + "; expected_placement=" + profile.placement_sha256
                    + "; physically_qualified=" + details
                )
            return self._without_rejections(
                matching[0], self._desktop_cost_only_rejections()
            )
        gpu_rows = [
            row for row in physically_qualified
            if any(
                self.catalog.placement_profile.devices[device_id].kind
                    == "gpu"
                for device_id in row.device_ids
            )
        ]
        desktop = gpu_rows or physically_qualified
        if not desktop:
            details = "; ".join(
                row.candidate_id + "=["
                + ",".join(row.rejection_reasons)
                + "]"
                for row in rows
                if all(
                    self.catalog.placement_profile.devices[
                        device_id
                    ].kind in {"cpu", "gpu"}
                    for device_id in row.device_ids
                )
            )
            raise DesktopControlUnavailableError(
                "qualified desktop control was not generated"
                + ("; " + details if details else "")
            )
        selected = min(desktop, key=lambda row: (
            row.cost.finish_upper_us,
            row.cost.fleet_energy_upper_uj
            if row.cost.fleet_energy_upper_uj is not None else 2**63 - 1,
            row.candidate_id,
        ))
        return self._without_rejections(
            selected, self._desktop_cost_only_rejections()
        )

    def _physically_qualified_desktop(
        self,
        row: AutomatedRouteCandidate,
    ) -> bool:
        if not all(
            self.catalog.placement_profile.devices[device_id].kind
            in {"cpu", "gpu"}
            and self.catalog.executor_by_device[device_id].maturity
            == "QUALIFIED"
            for device_id in row.device_ids
        ):
            return False
        coordinator = self.catalog.composite_executor_by_id.get(
            row.binding.executor_id
        )
        if coordinator is not None and coordinator.maturity != "QUALIFIED":
            return False
        return all(
            transition.maturity == "QUALIFIED"
            for transition in row.plan.transitions
        )

    @staticmethod
    def _without_rejection(
        row: AutomatedRouteCandidate,
        reason: str,
    ) -> AutomatedRouteCandidate:
        reasons = tuple(
            value for value in row.rejection_reasons
            if value != reason
        )
        return replace(
            row,
            binding=replace(
                row.binding,
                ready=not reasons,
                eligibility_reasons=reasons,
            ),
            admitted=not reasons,
            rejection_reasons=reasons,
        )

    @staticmethod
    def _without_rejections(
        row: AutomatedRouteCandidate,
        removed: frozenset[str],
    ) -> AutomatedRouteCandidate:
        reasons = tuple(
            value for value in row.rejection_reasons
            if value not in removed
        )
        return replace(
            row,
            binding=replace(
                row.binding,
                ready=not reasons,
                eligibility_reasons=reasons,
            ),
            admitted=not reasons,
            rejection_reasons=reasons,
        )

    def _materialize_route_key(
        self,
        *,
        request,
        manifest: ModelManifest,
        work: ModelRequestWork,
        snapshot: HeterogeneousRuntimeSnapshot,
        profile: PlacementHardwareProfile,
        profile_sha256: str,
        pattern: _Pattern,
        input_token_bucket: int,
        output_token_bucket: int,
        observed_at_us: int,
        residency_holds: Mapping[str, RuntimeResidencyCohortHold],
    ) -> AutomatedRouteCandidate:
        residency_states = {
            device_id: (
                "cold" if residency is None else residency.state
            )
            for device_id in pattern.device_ids
            for residency in (
                self._matching_residency(
                    manifest, pattern, snapshot, device_id
                ),
            )
        }
        residency_variant = (
            "cold"
            if "cold" in residency_states.values()
            else "warm"
            if "warm" in residency_states.values()
            else "hot"
        )
        coordinator = self._coordinator(pattern)
        pattern_profile = profile
        pattern_profile_sha256 = profile_sha256
        if coordinator is not None:
            transport_generation = coordinator.adapter_parameters.get(
                "request_transport_generation"
            )
            if transport_generation is not None:
                participating = frozenset(pattern.device_ids)
                pattern_profile = replace(
                    profile,
                    links=tuple(
                        link for link in profile.links
                        if (
                            link.source_device not in participating
                            and link.target_device not in participating
                        )
                        or link.transport_generation
                            == transport_generation
                    ),
                )
                pattern_profile_sha256 = canonical_sha256(
                    pattern_profile
                )
        signature_key = (
            manifest.artifact_sha256,
            work.input_tokens,
            work.output_tokens,
            pattern.route_key,
        )
        kernel_signature = self._kernel_signature_cache.get(
            signature_key
        )
        if kernel_signature is None:
            kernel_signature = self._kernel_signature(pattern, work)
            self._bounded_cache_store(
                self._kernel_signature_cache,
                signature_key,
                kernel_signature,
                4_096,
            )
        cache_key = (
            manifest.artifact_sha256,
            input_token_bucket,
            output_token_bucket,
            kernel_signature,
            pattern.route_key,
            pattern_profile_sha256,
            request.quality_requirement,
        )

        def build_placement():
            nodes = self._nodes(manifest, work, pattern)
            assignments = self._operator_assignments(
                manifest, nodes, pattern
            )
            placement = None
            placement_error = None
            try:
                placement = HierarchicalPlacementPlanner(
                    pattern_profile
                ).plan_sequence(
                    problem_id="auto:" + pattern.route_key,
                    nodes=nodes,
                    initial_device=self.catalog.fallback.device_id,
                    final_device=self.catalog.fallback.device_id,
                    deadline_us=2**63 - 1,
                    required_quality=request.quality_requirement,
                    require_measured=False,
                    defer_memory_validation=True,
                )
            except PlacementError as exc:
                placement_error = str(exc)
            return placement, placement_error, nodes, assignments

        cached, cache_hit = self._cached_placement(
            cache_key, build_placement
        )
        if cache_hit:
            self._placement_cache_hits += 1
        placement, placement_error, nodes, assignments = cached
        return self._one(
            request,
            manifest,
            work,
            snapshot,
            pattern_profile,
            pattern,
            residency_variant,
            residency_states,
            observed_at_us,
            nodes,
            assignments,
            placement,
            placement_error,
            residency_holds,
        )

    def reuse_exact_route_template_set(
        self,
        templates: RuntimeRouteTemplateSet,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int,
        search_metadata: Mapping[str, object],
    ) -> AutomatedCandidateSet | None:
        """Reuse a request-specific published result after live revalidation."""
        started_ns = time.perf_counter_ns()
        if (
            not isinstance(templates, RuntimeRouteTemplateSet)
            or templates.artifact_sha256 != manifest.artifact_sha256
            or templates.quality_requirement != request.quality_requirement
        ):
            raise RouteGenerationError(
                "runtime route template identity differs"
            )
        snapshot.validate_at(observed_at_us)
        if (
            templates.candidate_set.request_id != request.request_id
            or templates.live_state_sha256 is None
            or templates.live_state_sha256
                != self.route_template_live_state_sha256(snapshot)
        ):
            return None
        source_rows = templates.candidate_set.candidates
        source_by_id = {
            row.candidate_id: row for row in source_rows
        }
        selected = source_by_id.get(templates.selected_route_id)
        if selected is None:
            raise RouteGenerationError(
                "published route template is absent"
            )
        component = runtime_residency_component_identity(
            manifest.artifact_sha256,
            selected.plan,
            selected.binding,
        )
        if (
            component.identity_sha256
                != templates.selected_component_identity_sha256
        ):
            raise RouteGenerationError(
                "published route component differs"
            )
        live_ids = {
            templates.selected_route_id,
            templates.candidate_set.baseline_route_id,
        }
        if selected.paired_baseline_route_id is not None:
            live_ids.add(selected.paired_baseline_route_id)
        if templates.candidate_set.recovery_fallback_route_id is not None:
            live_ids.add(
                templates.candidate_set.recovery_fallback_route_id
            )
        final = []
        for source in source_rows:
            if source.candidate_id in live_ids:
                final.append(replace(
                    source,
                    baseline=(
                        source.candidate_id
                            == templates.candidate_set.baseline_route_id
                    ),
                ))
                continue
            reasons = tuple(sorted(set(
                source.rejection_reasons
                + ("MODEL_EPOCH_AUDIT_ONLY",)
            )))
            final.append(replace(
                source,
                binding=replace(
                    source.binding,
                    ready=False,
                    eligibility_reasons=reasons,
                ),
                admitted=False,
                baseline=False,
                rejection_reasons=reasons,
            ))
        metadata = {
            **dict(templates.candidate_set.search_metadata),
            **dict(search_metadata),
            "evaluated_plan_count": len(final),
            "route_template_audit_sha256": templates.audit_sha256,
            "route_template_cross_shape_reuse": False,
            "route_template_exact_request_reuse": True,
            "route_template_request_shape_bucket": (
                request_shape_bucket(
                    request.input_tokens, request.output_tokens
                )
            ),
            "route_template_source_shape_bucket": (
                templates.input_token_bucket,
                templates.output_token_bucket,
            ),
            "visited_plan_ids": tuple(
                row.candidate_id for row in final
            ),
        }
        result = AutomatedCandidateSet(
            request_id=request.request_id,
            model_id=manifest.model_id,
            snapshot_id=snapshot.snapshot_id,
            candidates=tuple(final),
            baseline_route_id=(
                templates.candidate_set.baseline_route_id
            ),
            recovery_fallback_route_id=(
                templates.candidate_set.recovery_fallback_route_id
            ),
            search_metadata=metadata,
        )
        finished_ns = time.perf_counter_ns()
        total_us = (finished_ns - started_ns) // 1000
        self._last_generation_timing = MappingProxyType({
            "candidate_template_lookup_us": total_us,
            "epoch_validation_us": total_us,
            "live_cost_update_us": 0,
            "pareto_and_contract_us": 0,
            "request_work_and_profile_us": 0,
            "rough_frontier_us": 0,
            "total_us": total_us,
            "visited_candidates": len(result.candidates),
        })
        return result

    def _template_work_and_profile(
        self,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
    ):
        work_key = (
            manifest.artifact_sha256,
            request.input_tokens,
            request.output_tokens,
        )
        work = self._request_work_cache.get(work_key)
        if work is None:
            work = manifest.request_work(
                request.input_tokens, request.output_tokens
            )
            self._bounded_cache_store(
                self._request_work_cache, work_key, work, 512
            )
        profile_key = tuple(
            (
                link.link_id,
                link.bandwidth_bytes_per_s
                if snapshot.links.get(link.link_id) is None
                else snapshot.links[
                    link.link_id
                ].measured_bandwidth_bytes_per_s,
            )
            for link in self.catalog.placement_profile.links
        ) + tuple(
            (
                "phone-power",
                row.device_id,
                row.domain_id,
                row.active_power_mw,
                row.idle_power_mw,
                row.estimation_version,
            )
            for row in self.catalog.phone_power_profiles
        )
        cached_profile = self._cost_profile_cache.get(profile_key)
        if cached_profile is None:
            profile = self._effective_cost_profile(snapshot)
            profile = replace(
                profile,
                devices={
                    device_id: replace(device, ready=True)
                    for device_id, device in profile.devices.items()
                },
                links=tuple(
                    replace(link, ready=True) for link in profile.links
                ),
            )
            cached_profile = (profile, canonical_sha256(profile))
            self._bounded_cache_store(
                self._cost_profile_cache,
                profile_key,
                cached_profile,
                64,
            )
        return work, *cached_profile

    def _template_live_keys(
        self,
        templates: RuntimeRouteTemplateSet,
        manifest: ModelManifest,
        additional_live_route_ids: Sequence[str] | None,
    ):
        patterns = self._patterns_by_key(manifest)
        source_rows = templates.candidate_set.candidates
        source_by_id = {
            row.candidate_id: row for row in source_rows
        }
        selected_source = source_by_id.get(templates.selected_route_id)
        if selected_source is None:
            raise RouteGenerationError(
                "published route template is absent"
            )
        dispatch_live_keys = {
            templates.selected_route_key,
            self._route_key(
                templates.candidate_set.baseline_route_id
            ),
        }
        if selected_source.paired_baseline_route_id is not None:
            dispatch_live_keys.add(self._route_key(
                selected_source.paired_baseline_route_id
            ))
        audit_source_by_key = {
            route_key: row
            for row in source_rows
            for route_key in (self._route_key(row.candidate_id),)
            if route_key not in dispatch_live_keys
        }
        additional = (
            tuple(
                row.candidate_id
                for row in audit_source_by_key.values()
            )
            if additional_live_route_ids is None
            else tuple(additional_live_route_ids)
        )
        if (
            any(
                type(route_id) is not str or not route_id
                for route_id in additional
            )
            or any(
                route_id not in source_by_id
                for route_id in additional
            )
        ):
            raise RouteGenerationError(
                "additional live route identity differs"
            )
        live_keys = dispatch_live_keys | {
            self._route_key(route_id) for route_id in additional
        }
        live_keys.update(
            self._route_key(parent_id)
            for route_id in additional
            for parent_id in (
                source_by_id[route_id].paired_baseline_route_id,
            )
            if parent_id is not None
        )
        if set(live_keys) - set(patterns):
            raise RouteGenerationError(
                "published route references an absent capability"
            )
        return (
            patterns,
            source_rows,
            source_by_id,
            selected_source,
            audit_source_by_key,
            live_keys,
        )

    def _materialized_template_routes(
        self,
        *,
        request,
        manifest,
        work,
        snapshot,
        profile,
        profile_sha256,
        patterns,
        live_keys,
        input_bucket,
        output_bucket,
        observed_at_us,
        residency_holds,
    ):
        return {
            route_key: self._materialize_route_key(
                request=request,
                manifest=manifest,
                work=work,
                snapshot=snapshot,
                profile=profile,
                profile_sha256=profile_sha256,
                pattern=patterns[route_key],
                input_token_bucket=input_bucket,
                output_token_bucket=output_bucket,
                observed_at_us=observed_at_us,
                residency_holds=residency_holds,
            )
            for route_key in sorted(live_keys)
        }

    def _revalidate_selected_template(
        self,
        templates: RuntimeRouteTemplateSet,
        manifest: ModelManifest,
        selected_source: AutomatedRouteCandidate,
        live_by_key,
        baseline_key,
    ):
        selected = live_by_key[templates.selected_route_key]
        parent_key = (
            None
            if selected_source.paired_baseline_route_id is None
            else self._route_key(
                selected_source.paired_baseline_route_id
            )
        )
        if parent_key is not None:
            selected = replace(
                selected,
                paired_baseline_route_id=(
                    live_by_key[parent_key].candidate_id
                ),
            )
        component = runtime_residency_component_identity(
            manifest.artifact_sha256,
            selected.plan,
            selected.binding,
        )
        if (
            component.identity_sha256
            != templates.selected_component_identity_sha256
        ):
            raise RouteGenerationError(
                "published route component differs"
            )
        if (
            selected.residency_variant == "cold"
            and templates.selected_route_key != baseline_key
        ):
            if (
                selected_source.residency_variant != "cold"
                or not selected_source.admitted
            ):
                reasons = tuple(sorted(set(
                    selected.rejection_reasons
                    + ("MODEL_EPOCH_COLD_REVALIDATION_REQUIRED",)
                )))
                selected = replace(
                    selected,
                    binding=replace(
                        selected.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    admitted=False,
                    rejection_reasons=reasons,
                )
            else:
                selected = replace(
                    selected,
                    residency_break_even=(
                        selected_source.residency_break_even
                    ),
                )
        live_by_key[templates.selected_route_key] = selected
        return selected, parent_key

    def _revalidate_template_controls(
        self,
        templates: RuntimeRouteTemplateSet,
        request,
        live_by_key,
        baseline_key,
        fallback_key,
    ) -> None:
        baseline = self._without_rejections(
            live_by_key[baseline_key],
            self._desktop_cost_only_rejections(),
        )
        live_by_key[baseline_key] = baseline
        fallback = (
            None if fallback_key is None
            else live_by_key.get(fallback_key)
        )
        if fallback is not None:
            fallback = self._without_rejections(
                fallback,
                frozenset(
                    reason for reason in fallback.rejection_reasons
                    if reason in self._desktop_cost_only_rejections()
                ),
            )
            live_by_key[fallback_key] = fallback
        if baseline.cost.finish_upper_us > request.deadline_us:
            live_by_key.update({
                route_key: self._without_rejection(
                    row, "SLO_UPPER_BOUND"
                )
                for route_key, row in live_by_key.items()
            })

    def _materialized_template_rows(
        self,
        templates: RuntimeRouteTemplateSet,
        source_rows,
        source_by_id,
        selected_source,
        audit_source_by_key,
        live_by_key,
        baseline_key,
        fallback_key,
        parent_key,
    ):
        source_id_by_live_key = {
            templates.selected_route_key: selected_source.candidate_id,
            baseline_key: templates.candidate_set.baseline_route_id,
        }
        source_id_by_live_key.update({
            route_key: source.candidate_id
            for route_key, source in audit_source_by_key.items()
            if route_key not in source_id_by_live_key
        })
        if fallback_key is not None and fallback_key in live_by_key:
            source_id_by_live_key[fallback_key] = (
                templates.candidate_set.recovery_fallback_route_id
            )
        if parent_key is not None:
            source_id_by_live_key[parent_key] = (
                selected_source.paired_baseline_route_id
            )
        source_ids = frozenset(source_by_id)
        source_id_by_live_key = {
            route_key: (
                live_by_key[route_key].candidate_id
                if (
                    route_key in live_by_key
                    and live_by_key[route_key].candidate_id
                        in source_ids
                )
                else source_id
            )
            for route_key, source_id in source_id_by_live_key.items()
        }
        final = []
        source_parent_ids = []
        replacement_id_by_source_id = {}
        for source in source_rows:
            route_key = self._route_key(source.candidate_id)
            row = (
                live_by_key.get(route_key)
                if source_id_by_live_key.get(route_key)
                    == source.candidate_id
                else None
            )
            if row is None:
                reasons = tuple(sorted(set(
                    source.rejection_reasons
                    + ("MODEL_EPOCH_AUDIT_ONLY",)
                )))
                row = replace(
                    source,
                    binding=replace(
                        source.binding,
                        ready=False,
                        eligibility_reasons=reasons,
                    ),
                    admitted=False,
                    baseline=False,
                    rejection_reasons=reasons,
                )
            else:
                preserved = {
                    reason for reason in source.rejection_reasons
                    if reason in {
                        "MODEL_EPOCH_AUDIT_ONLY",
                        "SEARCH_COVERAGE_ONLY",
                    }
                    and not (
                        reason == "MODEL_EPOCH_AUDIT_ONLY"
                        and source.candidate_id
                            == templates.selected_route_id
                    )
                }
                if route_key in audit_source_by_key:
                    preserved.add("MODEL_EPOCH_AUDIT_ONLY")
                reasons = tuple(sorted(
                    set(row.rejection_reasons) | preserved
                ))
                row = replace(
                    row,
                    binding=replace(
                        row.binding,
                        ready=not reasons,
                        eligibility_reasons=reasons,
                    ),
                    admitted=not reasons,
                    baseline=(route_key == baseline_key),
                    rejection_reasons=reasons,
                )
            source_parent_ids.append(
                source.paired_baseline_route_id
                if source.paired_baseline_route_id is not None
                else row.paired_baseline_route_id
            )
            replacement_id_by_source_id[source.candidate_id] = (
                row.candidate_id
            )
            final.append(row)
        return final, source_parent_ids, replacement_id_by_source_id

    @staticmethod
    def _repair_template_parent_routes(
        final,
        source_parent_ids,
        replacement_id_by_source_id,
    ):
        final_by_id = {row.candidate_id: row for row in final}
        repaired = []
        for row, parent_id in zip(final, source_parent_ids):
            if parent_id is None:
                repaired.append(row)
                continue
            parent_id = replacement_id_by_source_id.get(
                parent_id, parent_id
            )
            parent = final_by_id.get(parent_id)
            if parent is not None and (
                row.plan.baseline_executor_id
                    == parent.binding.executor_id
                and parent.plan.baseline_executor_id is None
                and row.plan.desktop_placement_sha256
                    == parent.plan.desktop_placement_sha256
            ):
                repaired.append(replace(
                    row, paired_baseline_route_id=parent_id
                ))
                continue
            reasons = tuple(sorted(set(
                row.rejection_reasons + ("PAIRED_BASELINE_ABSENT",)
            )))
            repaired.append(replace(
                row,
                binding=replace(
                    row.binding,
                    ready=False,
                    eligibility_reasons=reasons,
                ),
                admitted=False,
                rejection_reasons=reasons,
                paired_baseline_route_id=None,
            ))
        return repaired

    @staticmethod
    def _validate_materialized_baseline(
        final,
        baseline_id: str,
    ) -> None:
        baseline = next(
            (row for row in final if row.candidate_id == baseline_id),
            None,
        )
        if baseline is None:
            raise RouteGenerationError(
                "published desktop baseline is absent after live "
                "revalidation"
            )
        if not baseline.admitted:
            reasons = (
                baseline.rejection_reasons
                or baseline.binding.eligibility_reasons
                or ("UNKNOWN",)
            )
            raise RouteGenerationError(
                "published desktop baseline live revalidation failed: "
                + ",".join(reasons)
            )

    @staticmethod
    def _materialized_template_metadata(
        templates,
        search_metadata,
        final,
        input_bucket,
        output_bucket,
        live_by_key,
        live_keys,
    ):
        return {
            **dict(templates.candidate_set.search_metadata),
            **dict(search_metadata),
            "evaluated_plan_count": len(final),
            "route_template_audit_sha256": templates.audit_sha256,
            "route_template_cross_shape_reuse": (
                templates.input_token_bucket != input_bucket
                or templates.output_token_bucket != output_bucket
            ),
            "route_template_request_shape_bucket": (
                input_bucket, output_bucket
            ),
            "route_template_live_route_ids": tuple(sorted(
                live_by_key[route_key].candidate_id
                for route_key in live_keys
            )),
            "route_template_source_shape_bucket": (
                templates.input_token_bucket,
                templates.output_token_bucket,
            ),
            "visited_plan_ids": tuple(
                row.candidate_id for row in final
            ),
        }

    def materialize_route_template_set(
        self,
        templates: RuntimeRouteTemplateSet,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int,
        residency_holds: Mapping[str, RuntimeResidencyCohortHold],
        search_metadata: Mapping[str, object],
        additional_live_route_ids: Sequence[str] | None = None,
    ) -> AutomatedCandidateSet:
        """Revalidate one published route while retaining cached audit rows."""

        started_ns = time.perf_counter_ns()
        input_bucket, output_bucket = request_shape_bucket(
            request.input_tokens, request.output_tokens
        )
        if (
            not isinstance(templates, RuntimeRouteTemplateSet)
            or templates.artifact_sha256 != manifest.artifact_sha256
            or templates.quality_requirement != request.quality_requirement
        ):
            raise RouteGenerationError(
                "runtime route template identity differs"
            )
        snapshot.validate_at(observed_at_us)
        work, profile, profile_sha256 = (
            self._template_work_and_profile(
                request, manifest, snapshot
            )
        )
        setup_finished_ns = time.perf_counter_ns()
        (
            patterns,
            source_rows,
            source_by_id,
            selected_source,
            audit_source_by_key,
            live_keys,
        ) = self._template_live_keys(
            templates, manifest, additional_live_route_ids
        )
        live_by_key = self._materialized_template_routes(
            request=request,
            manifest=manifest,
            work=work,
            snapshot=snapshot,
            profile=profile,
            profile_sha256=profile_sha256,
            patterns=patterns,
            live_keys=live_keys,
            input_bucket=input_bucket,
            output_bucket=output_bucket,
            observed_at_us=observed_at_us,
            residency_holds=residency_holds,
        )
        materialized_finished_ns = time.perf_counter_ns()
        baseline_key = self._route_key(
            templates.candidate_set.baseline_route_id
        )
        fallback_key = (
            None
            if templates.candidate_set.recovery_fallback_route_id is None
            else self._route_key(
                templates.candidate_set.recovery_fallback_route_id
            )
        )
        _, parent_key = self._revalidate_selected_template(
            templates,
            manifest,
            selected_source,
            live_by_key,
            baseline_key,
        )
        self._revalidate_template_controls(
            templates,
            request,
            live_by_key,
            baseline_key,
            fallback_key,
        )
        final, parent_ids, replacement_ids = (
            self._materialized_template_rows(
                templates,
                source_rows,
                source_by_id,
                selected_source,
                audit_source_by_key,
                live_by_key,
                baseline_key,
                fallback_key,
                parent_key,
            )
        )
        final = self._repair_template_parent_routes(
            final, parent_ids, replacement_ids
        )
        baseline_id = live_by_key[baseline_key].candidate_id
        self._validate_materialized_baseline(final, baseline_id)
        fallback_id = (
            None
            if fallback_key is None
            else templates.candidate_set.recovery_fallback_route_id
            if fallback_key not in live_by_key
            else live_by_key[fallback_key].candidate_id
        )
        result = AutomatedCandidateSet(
            request_id=request.request_id,
            model_id=manifest.model_id,
            snapshot_id=snapshot.snapshot_id,
            candidates=tuple(final),
            baseline_route_id=baseline_id,
            recovery_fallback_route_id=fallback_id,
            search_metadata=self._materialized_template_metadata(
                templates,
                search_metadata,
                final,
                input_bucket,
                output_bucket,
                live_by_key,
                live_keys,
            ),
        )
        finished_ns = time.perf_counter_ns()
        self._last_generation_timing = MappingProxyType({
            "candidate_template_lookup_us": (
                setup_finished_ns - started_ns
            ) // 1000,
            "epoch_validation_us": 0,
            "live_cost_update_us": (
                materialized_finished_ns - setup_finished_ns
            ) // 1000,
            "pareto_and_contract_us": (
                finished_ns - materialized_finished_ns
            ) // 1000,
            "request_work_and_profile_us": (
                setup_finished_ns - started_ns
            ) // 1000,
            "rough_frontier_us": 0,
            "total_us": (finished_ns - started_ns) // 1000,
            "visited_candidates": len(result.candidates),
        })
        return result

    def rerank_route_template_set(
        self,
        templates: RuntimeRouteTemplateSet,
        request,
        manifest: ModelManifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int,
        reuse_projections: Mapping[
            str, RuntimeResidencyReuseProjection
        ],
        expected_reuse_count: int = 1,
    ) -> tuple[AutomatedCandidateSet, Mapping[str, int]]:
        """Re-cost a cached structural search without rebuilding its DAG."""
        if not isinstance(templates, RuntimeRouteTemplateSet):
            raise RouteGenerationError(
                "runtime route template refresh is invalid"
            )
        input_bucket, output_bucket = request_shape_bucket(
            request.input_tokens, request.output_tokens
        )
        source_rows = templates.candidate_set.candidates
        for row in source_rows:
            known_executor_id = self._executor_id_by_plan_sha256.get(
                row.plan.plan_sha256
            )
            if (
                known_executor_id is not None
                and known_executor_id != row.binding.executor_id
            ):
                raise RouteGenerationError(
                    "runtime route template executor identity differs"
                )
            self._bounded_cache_store(
                self._executor_id_by_plan_sha256,
                row.plan.plan_sha256,
                row.binding.executor_id,
                4_096,
            )
        visits = tuple(
            RoughPlacementVisit(
                route_key=self._route_key(row.candidate_id),
                residency_variant=row.residency_variant,
                rough_latency_us=max(1, row.cost.service_upper_us),
                rough_energy_uj=max(
                    1, row.cost.fleet_energy_upper_uj or 1
                ),
                coverage_group=(
                    "audit:" + row.route_family + ":"
                    + "+".join(row.device_ids)
                ),
                coverage_only=(
                    "SEARCH_COVERAGE_ONLY" in row.rejection_reasons
                ),
                rough_memory_feasible=True,
                mandatory=True,
            )
            for row in source_rows
        )
        if len(visits) > 32:
            raise RouteGenerationError(
                "runtime route template exceeds the bounded search budget"
            )
        frontier = RoughPlacementFrontier(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=(
                self._capability_generation_sha256
            ),
            input_token_bucket=input_bucket,
            output_token_bucket=output_bucket,
            search_budget=32,
            visits=visits,
            rough_plan_count=len(visits),
        )
        with self._generation_lock:
            refreshed = self.generate(
                request,
                manifest,
                snapshot,
                observed_at_us=observed_at_us,
                prepared_frontier=frontier,
                residency_holds={},
                reuse_projections=reuse_projections,
                expected_reuse_count=expected_reuse_count,
            )
            timing = MappingProxyType(dict(self._last_generation_timing))
        source_keys = tuple(sorted(
            self._route_key(row.candidate_id)
            for row in templates.candidate_set.candidates
        ))
        refreshed_keys = tuple(sorted(
            self._route_key(row.candidate_id)
            for row in refreshed.candidates
        ))
        if source_keys != refreshed_keys:
            raise RouteGenerationError(
                "runtime route template coverage changed during refresh"
            )
        return refreshed, timing
