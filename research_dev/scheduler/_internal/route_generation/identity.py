"""Route capability identities, observation store access, estimates, and cache statistics.

Mixin of ``AutomatedRouteCompiler``; methods moved here verbatim.
"""

from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType
from typing import Mapping, Sequence
from ..model_manifest import ModelManifest
from ..placement import PlacementHardwareProfile, TransferLink
from ..runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeExecutorCapability,
)
from ..runtime_plan import (
    RuntimeExecutionPlan,
    RuntimeExecutionReceipt,
    RuntimeTransitionReceipt,
)
from ..runtime_learning import (
    RuntimeLearnedRouteEstimate,
    RuntimeLearnedTransitionEstimate,
    RuntimeRouteObservationStore,
)
from ..types import canonical_sha256
from .common import (
    RouteGenerationError,
    _static_executor_identity,
    _static_coordinator_identity,
)


class RouteIdentityMixin:
    """Route capability identities, observation store access, estimates, and cache statistics."""

    def _effective_cost_profile(
        self, snapshot: HeterogeneousRuntimeSnapshot
    ) -> PlacementHardwareProfile:
        """Apply operational power without changing physical identity."""
        profile = snapshot.effective_placement_profile(self.catalog)
        if not self.catalog.phone_power_profiles:
            return profile
        domains = dict(profile.domains)
        kernels = dict(profile.kernels)
        links = list(profile.links)
        for phone_power in self.catalog.phone_power_profiles:
            domain = domains[phone_power.domain_id]
            domains[phone_power.domain_id] = replace(
                domain,
                domain=replace(
                    domain.domain,
                    idle_power_mw=phone_power.idle_power_mw,
                ),
            )
            kernels.update({
                profile_id: replace(
                    row,
                    kernel=replace(
                        row.kernel,
                        active_power_mw=phone_power.active_power_mw,
                    ),
                )
                for profile_id, row in kernels.items()
                if row.device_id == phone_power.device_id
            })
            links = [
                replace(
                    row,
                    domain_active_power_mw={
                        **row.domain_active_power_mw,
                        phone_power.domain_id: phone_power.active_power_mw,
                    },
                )
                if phone_power.device_id in {
                    row.source_device, row.target_device
                } else row
                for row in links
            ]
        return replace(
            profile,
            domains=domains,
            kernels=kernels,
            links=tuple(links),
        )

    def route_capability_identity(
        self,
        plan: RuntimeExecutionPlan,
        executor_id: str | None = None,
    ) -> str:
        """Bind learned evidence to the physical subgraph used by one plan."""
        return self._capability_identity(
            plan,
            executor_id,
            include_transitions=True,
        )

    def _capability_identity(
        self,
        plan: RuntimeExecutionPlan,
        executor_id: str | None,
        *,
        include_transitions: bool,
        include_residency_ownership: bool = True,
        exclude_phone_power_accounting: bool = False,
    ) -> str:
        if not isinstance(plan, RuntimeExecutionPlan):
            raise RouteGenerationError(
                "runtime route capability plan is invalid"
            )
        executor_id = (
            self._executor_id_by_plan_sha256.get(plan.plan_sha256)
            if executor_id is None else executor_id
        )
        if type(executor_id) is not str:
            raise RouteGenerationError(
                "runtime route capability executor is absent"
            )
        coordinator = (
            self.catalog.executor_by_id.get(executor_id)
            or self.catalog.composite_executor_by_id.get(executor_id)
        )
        if coordinator is None:
            raise RouteGenerationError(
                "runtime route capability executor is unknown"
            )
        if set(plan.resource_ids) - set(self.catalog.resources):
            raise RouteGenerationError("runtime route capability resource is unknown")
        profile = self.catalog.placement_profile
        kernel_ids = tuple(sorted({
            profile_id
            for assignment in plan.operators
            for profile_id in assignment.kernel_profile_ids
        }))
        if set(kernel_ids) - set(profile.kernels):
            raise RouteGenerationError(
                "runtime route capability kernel is unknown"
            )
        link_ids = {
            resource_id.removeprefix("link:")
            for resource_id in plan.resource_ids
            if resource_id.startswith("link:")
        }
        links = tuple(
            link for link in profile.links if link.link_id in link_ids
        )
        if {link.link_id for link in links} != link_ids:
            raise RouteGenerationError(
                "runtime route capability link is unknown"
            )
        memory_pool_ids = {
            demand.resource_id for demand in plan.memory_demands
        }
        if memory_pool_ids - set(profile.memory_pools):
            raise RouteGenerationError(
                "runtime route capability memory pool is unknown"
            )
        transition_by_id = {
            transition.transition_id: transition
            for transition in self.catalog.transitions
        }
        transition_ids = (
            tuple(row.transition_id for row in plan.transitions)
            if include_transitions else ()
        )
        try:
            transitions = tuple(
                transition_by_id[transition_id]
                for transition_id in transition_ids
            )
        except KeyError as exc:
            raise RouteGenerationError(
                "runtime route capability transition is unknown"
            ) from exc
        cache_key = (
            executor_id,
            include_residency_ownership,
            exclude_phone_power_accounting,
            plan.device_ids,
            kernel_ids,
            tuple(sorted(link_ids)),
            tuple(sorted(memory_pool_ids)),
            plan.resource_ids,
            transition_ids,
        )
        identity = self._route_capability_identity_cache.get(cache_key)
        if identity is None:
            phone_device_ids = frozenset(
                device_id for device_id in plan.device_ids
                if profile.devices[device_id].kind.startswith("phone")
            )
            phone_domain_ids = frozenset(
                profile.kernels[profile_id].kernel.domain_id
                for profile_id in kernel_ids
                if profile.kernels[profile_id].device_id
                    in phone_device_ids
            )
            power_evidence_ids = frozenset({
                "ASSUMED_4P5W",
                "assumed-phone-power-v1",
            })

            def evidence_without_power(
                values: Sequence[str],
            ) -> tuple[str, ...]:
                return tuple(
                    value for value in values
                    if value not in power_evidence_ids
                )

            def domain_identity(domain_id: str) -> object:
                row = profile.domains[domain_id]
                if (
                    not exclude_phone_power_accounting
                    or domain_id not in phone_domain_ids
                ):
                    return row
                return {
                    "domain": {
                        item.name: getattr(row.domain, item.name)
                        for item in fields(row.domain)
                        if item.name != "idle_power_mw"
                    },
                    "evidence_ids": evidence_without_power(
                        row.evidence_ids
                    ),
                    "status": "phone-power-accounting-neutral",
                }

            def kernel_identity(profile_id: str) -> object:
                row = profile.kernels[profile_id]
                if (
                    not exclude_phone_power_accounting
                    or row.device_id not in phone_device_ids
                ):
                    return row
                return {
                    "device_id": row.device_id,
                    "evidence_ids": evidence_without_power(
                        row.evidence_ids
                    ),
                    "kernel": {
                        item.name: getattr(row.kernel, item.name)
                        for item in fields(row.kernel)
                        if item.name != "active_power_mw"
                    },
                    "profile_id": row.profile_id,
                    "status": "phone-power-accounting-neutral",
                }

            def link_identity(link: TransferLink) -> object:
                if (
                    not exclude_phone_power_accounting
                    or not phone_device_ids.intersection({
                        link.source_device, link.target_device
                    })
                ):
                    return link
                return {
                    item.name: (
                        {
                            domain_id: value
                            for domain_id, value in (
                                link.domain_active_power_mw.items()
                            )
                            if domain_id not in phone_domain_ids
                        }
                        if item.name == "domain_active_power_mw" else
                        evidence_without_power(link.evidence_ids)
                        if item.name == "evidence_ids" else
                        getattr(link, item.name)
                    )
                    for item in fields(link)
                }

            identity = canonical_sha256({
                "coordinator": _static_coordinator_identity(
                    coordinator,
                    include_residency_ownership=(
                        include_residency_ownership
                    ),
                    exclude_phone_power_accounting=(
                        exclude_phone_power_accounting
                        and isinstance(
                            coordinator, RuntimeExecutorCapability
                        )
                        and coordinator.device_id in phone_device_ids
                    ),
                ),
                "devices": {
                    device_id: profile.devices[device_id]
                    for device_id in plan.device_ids
                },
                "energy_boundary_id": profile.energy_boundary_id,
                "energy_domains": {
                    domain_id: domain_identity(domain_id)
                    for domain_id in profile.idle_charge_domains
                },
                "executors": {
                    device_id: _static_executor_identity(
                        self.catalog.executor_by_device[device_id],
                        include_residency_ownership=(
                            include_residency_ownership
                        ),
                        exclude_phone_power_accounting=(
                            exclude_phone_power_accounting
                            and device_id in phone_device_ids
                        ),
                    )
                    for device_id in plan.device_ids
                },
                "kernels": {
                    profile_id: kernel_identity(profile_id)
                    for profile_id in kernel_ids
                },
                "links": tuple(link_identity(link) for link in links),
                "memory_pools": {
                    pool_id: profile.memory_pools[pool_id]
                    for pool_id in memory_pool_ids
                },
                "resources": {
                    resource_id: self.catalog.resources[resource_id]
                    for resource_id in plan.resource_ids
                },
                "transitions": transitions,
            })
            self._bounded_cache_store(
                self._route_capability_identity_cache,
                cache_key,
                identity,
                4_096,
            )
        return identity

    def phone_power_accounting_neutral_identity(
        self,
        plan: RuntimeExecutionPlan,
        executor_id: str | None = None,
        *,
        include_transitions: bool,
        include_residency_ownership: bool = True,
    ) -> str:
        """Bind physical execution while excluding phone wattage policy."""
        return self._capability_identity(
            plan,
            executor_id,
            include_transitions=include_transitions,
            include_residency_ownership=include_residency_ownership,
            exclude_phone_power_accounting=True,
        )

    def component_capability_identity(
        self,
        plan: RuntimeExecutionPlan,
        executor_id: str | None = None,
    ) -> str:
        """Bind reusable execution evidence without residency transitions."""
        executor_id = (
            self._executor_id_by_plan_sha256.get(plan.plan_sha256)
            if executor_id is None else executor_id
        )
        if type(executor_id) is not str:
            raise RouteGenerationError(
                "runtime component capability executor is absent"
            )
        return self._capability_identity(
            plan,
            executor_id,
            include_transitions=False,
        )

    @property
    def observation_store(self) -> RuntimeRouteObservationStore:
        return self._observation_store

    def observation_checkpoint(self) -> tuple[object, ...]:
        return self._observation_store.checkpoint()

    def restore_observations(self, checkpoint: tuple[object, ...]) -> None:
        self._observation_store.restore(checkpoint)

    def observation_state(self) -> Mapping[str, int]:
        return self._observation_store.state()

    def observation_export(self) -> Mapping[str, object]:
        return MappingProxyType(self._observation_store.to_json())

    def import_observations(self, value: object) -> None:
        self._observation_store.import_json(value)

    def replace_observations(self, value: object) -> None:
        """Atomically replace the private store used by a planning worker."""
        replacement = RuntimeRouteObservationStore(
            self._observation_store.qualification_samples
        )
        replacement.import_json(value)
        self._observation_store = replacement

    def merge_observations(self, value: object) -> None:
        self._observation_store.merge_json(value)

    def transition_estimates_for_plan(
        self,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        executor_id: str | None = None,
    ) -> tuple[RuntimeLearnedTransitionEstimate | None, ...]:
        component_capability = self.component_capability_identity(
            plan, executor_id
        )
        component_identity = self._transition_component_identity(
            manifest, plan, executor_id
        )
        return tuple(
            self._observation_store.transition_estimate_for_plan(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=component_capability,
                component_identity_sha256=component_identity,
                transition=transition,
                executor_id=executor_id,
            )
            for transition in plan.transitions
        )

    def exact_route_estimate(
        self,
        manifest: ModelManifest,
        request,
        plan: RuntimeExecutionPlan,
        cost_features: Mapping[str, int],
        executor_id: str | None = None,
    ) -> RuntimeLearnedRouteEstimate | None:
        """Return qualified evidence for this complete physical route."""
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError(
                "runtime exact-route manifest is invalid"
            )
        return self._observation_store.exact_estimate(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=(
                self.route_capability_identity(plan, executor_id)
            ),
            plan=plan,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            quality_requirement=request.quality_requirement,
            cost_features=cost_features,
        )

    def calibration_information(
        self,
        manifest: ModelManifest,
        request,
        plan: RuntimeExecutionPlan,
    ) -> Mapping[str, int]:
        if not isinstance(manifest, ModelManifest):
            raise RouteGenerationError(
                "runtime calibration manifest is invalid"
            )
        return self._observation_store.calibration_information(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=(
                self.route_capability_identity(plan)
            ),
            component_capability_generation_sha256=(
                self.component_capability_identity(plan)
            ),
            plan=plan,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            quality_requirement=request.quality_requirement,
        )

    def record_execution_observation(
        self,
        request,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        receipt: RuntimeExecutionReceipt,
        cost_features: Mapping[str, int],
        component_service_us: int,
        component_energy_uj: int | None,
        executor_id: str | None = None,
    ) -> bool:
        return self._observation_store.record(
            artifact_sha256=manifest.artifact_sha256,
            capability_generation_sha256=(
                self.route_capability_identity(plan, executor_id)
            ),
            plan=plan,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            quality_requirement=request.quality_requirement,
            cost_features=cost_features,
            receipt=receipt,
            energy_boundary_id=(
                self.catalog.placement_profile.energy_boundary_id
            ),
            required_domain_ids=tuple(
                self.catalog.placement_profile.idle_charge_domains
            ),
            component_service_us=component_service_us,
            component_energy_uj=component_energy_uj,
            component_capability_generation_sha256=(
                self.component_capability_identity(plan, executor_id)
            ),
        )

    def record_transition_observations(
        self,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        receipts: Sequence[RuntimeTransitionReceipt],
        executor_id: str | None = None,
    ) -> int:
        if not isinstance(manifest, ModelManifest) or not isinstance(
            plan, RuntimeExecutionPlan
        ):
            raise RouteGenerationError(
                "runtime transition observation input is invalid"
            )
        component_identity = self._transition_component_identity(
            manifest, plan, executor_id
        )
        recorded = 0
        for receipt in receipts:
            recorded += self._observation_store.record_transition(
                artifact_sha256=manifest.artifact_sha256,
                capability_generation_sha256=(
                    self.component_capability_identity(plan, executor_id)
                ),
                component_identity_sha256=component_identity,
                plan=plan,
                receipt=receipt,
                energy_boundary_id=(
                    self.catalog.placement_profile.energy_boundary_id
                ),
                required_domain_ids=tuple(
                    self.catalog.placement_profile.idle_charge_domains
                ),
            )
        return recorded

    def _transition_component_identity(
        self,
        manifest: ModelManifest,
        plan: RuntimeExecutionPlan,
        executor_id: str | None = None,
    ) -> str:
        component_identity = plan.adapter_parameters.get(
            "resident_model_identity_sha256"
        )
        if component_identity is None:
            component_identity = canonical_sha256({
                "artifact_sha256": manifest.artifact_sha256,
                "component_capability_sha256": (
                    self.component_capability_identity(plan, executor_id)
                ),
                "component_template_sha256": (
                    self._observation_store._component_template_sha256(plan)
                ),
                "schema": "runtime-transition-component-v1",
            })
        if (
            type(component_identity) is not str
            or not component_identity.startswith("sha256:")
            or len(component_identity) != 71
        ):
            raise RouteGenerationError(
                "runtime transition component identity is invalid"
            )
        return component_identity

    def cache_stats(self) -> Mapping[str, int]:
        rough = self._rough_compiler.stats()
        return MappingProxyType({
            "entries": len(self._placement_cache),
            "evictions": self._placement_cache_evictions,
            "generations": self._placement_cache_generations,
            "hits": self._placement_cache_hits,
            "misses": self._placement_cache_misses,
            "singleflight_waits": self._placement_singleflight_waits,
            "frontier_generations": self._frontier_generations,
            "frontier_singleflight_waits": (
                self._frontier_singleflight_waits
            ),
            "static_cache_evictions": self._static_cache_evictions,
            "rough_evictions": rough["evictions"],
            "rough_generations": rough["generations"],
            "rough_hits": rough["hits"],
            "rough_misses": rough["misses"],
            "shape_bucket_entries": rough["shape_bucket_entries"],
            "template_entries": len(self._pattern_cache),
            "template_index_entries": len(self._pattern_by_key_cache),
            "work_entries": len(self._request_work_cache),
            "kernel_signature_entries": len(
                self._kernel_signature_cache
            ),
            "operator_assignment_entries": len(
                self._operator_assignment_cache
            ),
            "memory_demand_entries": len(self._memory_demand_cache),
            "rough_cost_entries": len(self._rough_cost_cache),
            "rough_group_entries": len(self._rough_group_cache),
            "desktop_placement_entries": len(
                self._desktop_placement_hash_cache
            ),
            "ffn_resident_envelope_entries": len(
                self._ffn_resident_envelope_cache
            ),
            "execution_plan_entries": len(self._execution_plan_cache),
        })
