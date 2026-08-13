"""One runtime owner for every scheduling granularity and trace."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from ._internal.policy import (
    Decision,
    RoutePolicy,
    LeaseDemand,
    LeaseRecord,
    MarginalSystemCostContext,
    POLICY_MODES,
    ProfileBundle,
    Request,
    ResourceProfile,
    ResourceTimeline,
    SchedulerError,
)
from ._internal.lifecycle import (
    LifecycleProfileSet,
    LifecycleReceipt,
    UnifiedScheduleError,
)
from ._internal.matmul import (
    MatmulScheduleError,
    MatmulSystemProfile,
    MatmulPlanner,
    ModelProgram,
)
from ._internal.runtime_gates import RuntimeSnapshot
from ._internal.runtime_placement import (
    RuntimePlacementCandidate,
    RuntimePlacementDecision,
    RuntimePlacementError,
    RuntimePlacementPlanner,
    RuntimePlacementSnapshot,
)
from ._internal.runtime_cost import (
    RuntimeCostError,
    RuntimeCostEstimateSet,
    RuntimeCostEstimator,
    RuntimeExecutorBinding,
    RuntimeModelArtifact,
)
from ._internal.online_placement import (
    OnlinePlacementError,
    OnlinePlacementReceipt,
    OnlinePlacementTracker,
)
from ._internal.types import canonical_sha256
from ._internal.dynamic_residency import (
    DynamicResidencyCandidate,
    DynamicResidencyDecision,
    DynamicResidencyError,
    DynamicResidencyReceipt,
    DynamicResidencySnapshot,
    apply_dynamic_residency_receipt,
    select_dynamic_residency_transition,
)
from ._internal.gpu_backfill import (
    GpuBackfillCandidate,
    GpuBackfillDecision,
    GpuBackfillError,
    GpuBubbleWindow,
    GpuWavefrontDecision,
    GpuWavefrontSnapshot,
    select_gpu_backfill,
    select_gpu_wavefront_backfill,
)
from ._internal.phone_residency import (
    PhoneArmGroup,
    PhoneArmSignal,
    PhoneOffloadCandidate,
    PhoneOffloadDecision,
    PhoneResidencyError,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    select_energy_positive_offload,
)
from ._internal.phone_arbiter import (
    PhoneArbiterDecision,
    PhoneArbiterError,
    PhoneArbiterQueue,
    PhoneArbiterWindow,
    select_phone_arbiter_work,
)


__all__ = [
    "DynamicPlacementLease",
    "DynamicResidencySchedule",
    "GpuBackfillSchedule",
    "GpuWavefrontSchedule",
    "LifecycleProfileSet",
    "LifecycleReceipt",
    "PhoneArbiterSchedule",
    "PhoneOffloadSchedule",
    "UnifiedScheduleError",
    "UnifiedScheduler",
]


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise UnifiedScheduleError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise UnifiedScheduleError(f"{name} must be ASCII") from exc
    return value


@dataclass(frozen=True)
class DynamicPlacementLease:
    lease_id: str
    owner_id: str
    placement_ids: tuple[str, ...]
    source_snapshot_id: str
    source_generation: int
    source_epoch_key: str
    acquired_at_us: int


@dataclass(frozen=True)
class PhoneOffloadSchedule:
    decision: PhoneOffloadDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]
    placement_ids: tuple[str, ...]
    source_snapshot_id: str | None
    source_generation: int | None
    source_epoch_key: str | None


@dataclass(frozen=True)
class PhoneArbiterSchedule:
    decision: PhoneArbiterDecision
    phone_schedule: PhoneOffloadSchedule | None

    @property
    def owner_id(self) -> str | None:
        return (
            None
            if self.phone_schedule is None
            else self.phone_schedule.owner_id
        )

    @property
    def leases(self) -> tuple[LeaseRecord, ...]:
        return () if self.phone_schedule is None else self.phone_schedule.leases

    @property
    def queue_by_resource_us(self) -> Mapping[str, int]:
        return (
            MappingProxyType({})
            if self.phone_schedule is None
            else self.phone_schedule.queue_by_resource_us
        )

    @property
    def blocking_resources(self) -> tuple[str, ...]:
        return (
            ()
            if self.phone_schedule is None
            else self.phone_schedule.blocking_resources
        )


@dataclass(frozen=True)
class DynamicResidencySchedule:
    decision: DynamicResidencyDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]


@dataclass(frozen=True)
class GpuBackfillSchedule:
    decision: GpuBackfillDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]


@dataclass(frozen=True)
class GpuWavefrontSchedule:
    decision: GpuWavefrontDecision
    backfill_schedule: GpuBackfillSchedule

    @property
    def owner_id(self) -> str | None:
        return self.backfill_schedule.owner_id

    @property
    def leases(self) -> tuple[LeaseRecord, ...]:
        return self.backfill_schedule.leases

    @property
    def queue_by_resource_us(self) -> Mapping[str, int]:
        return self.backfill_schedule.queue_by_resource_us

    @property
    def blocking_resources(self) -> tuple[str, ...]:
        return self.backfill_schedule.blocking_resources


class UnifiedScheduler:
    """Own one resource timeline for every registered scheduling level."""

    def __init__(
        self,
        profiles: Sequence[ProfileBundle],
        mode: str,
        *,
        lifecycle_profiles: Sequence[LifecycleProfileSet] = (),
        matmul_profile: MatmulSystemProfile | None = None,
        runtime_snapshot: RuntimeSnapshot | None = None,
        phone_residency_plan: PhoneResidencyPlan | None = None,
        phone_residency_snapshot: PhoneResidencySnapshot | None = None,
        dynamic_residency_snapshot: DynamicResidencySnapshot | None = None,
    ) -> None:
        if mode not in POLICY_MODES:
            raise UnifiedScheduleError("unknown unified policy mode")
        if not profiles and not lifecycle_profiles and matmul_profile is None:
            raise UnifiedScheduleError("unified scheduler has no profile")
        if any(not isinstance(profile, ProfileBundle) for profile in profiles):
            raise UnifiedScheduleError("route profile is invalid")
        if any(
            not isinstance(profile_set, LifecycleProfileSet)
            for profile_set in lifecycle_profiles
        ):
            raise UnifiedScheduleError("lifecycle profile set is invalid")

        bundles = list(profiles)
        for profile_set in lifecycle_profiles:
            bundles.extend(profile_set.profiles.values())
        resources = self._merge_resources(
            bundles,
            None if matmul_profile is None else matmul_profile.resources,
        )
        if phone_residency_snapshot is not None and phone_residency_plan is None:
            raise UnifiedScheduleError(
                "phone residency snapshot has no plan"
            )
        if phone_residency_plan is not None:
            if not isinstance(phone_residency_plan, PhoneResidencyPlan):
                raise UnifiedScheduleError("phone residency plan is invalid")
            for resource_id in phone_residency_plan.execution_resource_ids:
                resource = resources.get(resource_id)
                if resource is None:
                    raise UnifiedScheduleError(
                        f"phone residency resource is absent: {resource_id}"
                    )
                if resource.capacity != 1:
                    raise UnifiedScheduleError(
                        f"phone residency resource must have capacity one: {resource_id}"
                    )
        self.mode = mode
        self.timeline = ResourceTimeline(resources)
        self._resource_ids = frozenset(resources)
        self.runtime_snapshot = runtime_snapshot
        self.phone_residency_plan = phone_residency_plan
        self.phone_residency_snapshot: PhoneResidencySnapshot | None = None
        if phone_residency_snapshot is not None:
            self.update_phone_residency(phone_residency_snapshot)
        self.dynamic_residency_snapshot: DynamicResidencySnapshot | None = None
        self._pending_dynamic_transition_id: str | None = None
        self._pending_dynamic_schedule: DynamicResidencySchedule | None = None
        self._active_dynamic_placement_leases: dict[
            str, DynamicPlacementLease
        ] = {}
        self._next_dynamic_placement_lease = 1
        self._active_gpu_backfills: dict[str, GpuBackfillSchedule] = {}
        self._next_gpu_backfill_owner = 1
        self._active_gpu_wavefronts: dict[str, GpuWavefrontSchedule] = {}
        self._gpu_wavefront_next_by_pipeline: dict[str, int] = {}
        self._gpu_wavefront_last_receipt_by_pipeline: dict[str, str] = {}
        self._completed_gpu_wavefront_chunks: dict[str, str] = {}
        self._active_phone_placement_schedules: dict[
            str, PhoneOffloadSchedule
        ] = {}
        self._next_phone_owner = 1
        self._active_phone_arbiters: dict[str, PhoneArbiterSchedule] = {}
        self._phone_arbiter_next_by_pipeline: dict[str, int] = {}
        self._phone_arbiter_last_receipt_by_pipeline: dict[str, str] = {}
        self._completed_phone_arbiter_work: dict[str, str] = {}
        self._online_placement = OnlinePlacementTracker()
        if dynamic_residency_snapshot is not None:
            self.update_dynamic_residency(dynamic_residency_snapshot)
        self._route_policies: dict[str, RoutePolicy] = {}
        profile_by_id: dict[str, ProfileBundle] = {}
        for profile in bundles:
            previous = profile_by_id.get(profile.profile_id)
            if previous is not None:
                if previous != profile:
                    raise UnifiedScheduleError(
                        f"profile id has different contents: {profile.profile_id}"
                    )
                continue
            profile_by_id[profile.profile_id] = profile
            self._route_policies[profile.profile_id] = RoutePolicy(
                profile,
                mode,
                runtime_snapshot=runtime_snapshot,
                timeline=self.timeline,
            )

        self._direct_workloads: dict[str, str] = {}
        self._lifecycle_workloads: dict[str, str] = {}
        for profile in profiles:
            for workload_id in {
                route.workload_id for route in profile.routes
            }:
                self._claim_workload(workload_id)
                self._direct_workloads[workload_id] = profile.profile_id

        self._profile_sets: dict[str, LifecycleProfileSet] = {}
        self._lifecycle_receipts: dict[str, LifecycleReceipt | None] = {}
        for profile_set in lifecycle_profiles:
            if profile_set.state_key in self._profile_sets:
                raise UnifiedScheduleError("duplicate lifecycle state key")
            self._profile_sets[profile_set.state_key] = profile_set
            self._lifecycle_receipts[profile_set.state_key] = None
            for workload_id in profile_set.workloads:
                self._claim_workload(workload_id)
                self._lifecycle_workloads[workload_id] = profile_set.state_key

        self.matmul = (
            None
            if matmul_profile is None
            else MatmulPlanner(
                matmul_profile,
                timeline=self.timeline,
            )
        )

    @staticmethod
    def _merge_resources(
        profiles: Sequence[ProfileBundle],
        matmul_resources: Mapping[str, ResourceProfile] | None,
    ) -> Mapping[str, ResourceProfile]:
        resources: dict[str, ResourceProfile] = {}
        collections = [profile.resources for profile in profiles]
        if matmul_resources is not None:
            collections.append(matmul_resources)
        for collection in collections:
            for resource_id, resource in collection.items():
                current = resources.get(resource_id)
                if current is not None and current != resource:
                    raise UnifiedScheduleError(
                        f"resource profile differs: {resource_id}"
                    )
                resources[resource_id] = resource
        return resources

    def _claim_workload(self, workload_id: str) -> None:
        if (
            workload_id in self._direct_workloads
            or workload_id in self._lifecycle_workloads
        ):
            raise UnifiedScheduleError(
                f"workload has multiple profile owners: {workload_id}"
            )

    def update_runtime_snapshot(self, snapshot: RuntimeSnapshot) -> None:
        for policy in self._route_policies.values():
            policy.update_runtime_snapshot(snapshot)
        self.runtime_snapshot = snapshot

    def select_runtime_placement(
        self,
        candidates: Sequence[RuntimePlacementCandidate],
        *,
        baseline_candidate_id: str,
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
        minimum_energy_saving_ppm: int,
        maximum_latency_ppm: int,
        minimum_samples: int = 2,
    ) -> RuntimePlacementDecision:
        """Select one measured whole-workload placement from live capacity."""
        try:
            return RuntimePlacementPlanner(minimum_samples).plan(
                candidates=candidates,
                baseline_candidate_id=baseline_candidate_id,
                snapshot=snapshot,
                now_us=now_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                maximum_latency_ppm=maximum_latency_ppm,
            )
        except RuntimePlacementError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def estimate_runtime_costs(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[RuntimeExecutorBinding],
        *,
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
    ) -> RuntimeCostEstimateSet:
        """Estimate current route costs from shape, residency, and capacity."""
        try:
            return RuntimeCostEstimator().estimate(
                profile=self.profile_for(request.workload_id),
                request=request,
                model=model,
                bindings=bindings,
                snapshot=snapshot,
                now_us=now_us,
            )
        except RuntimeCostError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def schedule_runtime_costs(
        self,
        request: Request,
        estimates: RuntimeCostEstimateSet,
        *,
        runtime_now_us: int,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> Decision:
        """Commit a route using the admissions from one runtime estimate."""
        if not isinstance(estimates, RuntimeCostEstimateSet):
            raise UnifiedScheduleError("runtime cost estimate is invalid")
        if (
            estimates.request_id != request.request_id
            or estimates.workload_id != request.workload_id
        ):
            raise UnifiedScheduleError(
                "runtime cost estimate does not match the request"
            )
        admissions = {
            estimate.route_id: (
                "ADMITTED" if estimate.admitted else estimate.reason
            )
            for estimate in estimates.estimates
        }
        profile = self.profile_for(request.workload_id)
        try:
            return self._route_policies[profile.profile_id].schedule(
                request,
                runtime_now_us,
                runtime_route_admissions=admissions,
                marginal_system_context=marginal_system_context,
            )
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def schedule_runtime(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[RuntimeExecutorBinding],
        *,
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> tuple[RuntimeCostEstimateSet, Decision]:
        """Estimate live routes and atomically gate the committed decision."""
        estimates = self.estimate_runtime_costs(
            request,
            model,
            bindings,
            snapshot=snapshot,
            now_us=now_us,
        )
        decision = self.schedule_runtime_costs(
            request,
            estimates,
            runtime_now_us=now_us,
            marginal_system_context=marginal_system_context,
        )
        return estimates, decision

    def schedule_online_request(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[RuntimeExecutorBinding],
        *,
        snapshot: RuntimePlacementSnapshot,
        observed_at_us: int,
        family_routes: Mapping[str, str | None],
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> tuple[
        RuntimeCostEstimateSet,
        Decision,
        OnlinePlacementReceipt,
    ]:
        """Place one observed request without accepting future-work inputs."""
        try:
            profile = self.profile_for(request.workload_id)
            scheduler_state = {
                "marginal_system_context_sha256": (
                    None
                    if marginal_system_context is None
                    else canonical_sha256(marginal_system_context)
                ),
                "mode": self.mode,
                "profile_id": profile.profile_id,
                "profile_sha256": canonical_sha256(profile),
                "resource_timeline": self.timeline.causal_state(),
                "runtime_snapshot_sha256": (
                    None
                    if self.runtime_snapshot is None
                    else canonical_sha256(self.runtime_snapshot)
                ),
                "schema": "research-scheduler-online-state-v1",
            }
            routes, causal_hash, causal_input = (
                self._online_placement.prepare(
                    request=request,
                    model=model,
                    bindings=bindings,
                    snapshot=snapshot,
                    observed_at_us=observed_at_us,
                    family_routes=family_routes,
                    scheduler_state=scheduler_state,
                )
            )
            estimates = self.estimate_runtime_costs(
                request,
                model,
                bindings,
                snapshot=snapshot,
                now_us=observed_at_us,
            )
            mapped_route_ids = {
                route_id for route_id in routes.values()
                if route_id is not None
            }
            unmapped_route_ids = {
                estimate.route_id for estimate in estimates.estimates
            } - mapped_route_ids
            if unmapped_route_ids:
                raise OnlinePlacementError(
                    "runtime routes lack online families: "
                    + ", ".join(sorted(unmapped_route_ids))
                )
            decision = self.schedule_runtime_costs(
                request,
                estimates,
                runtime_now_us=observed_at_us,
                marginal_system_context=marginal_system_context,
            )
            receipt = self._online_placement.commit(
                request=request,
                snapshot=snapshot,
                observed_at_us=observed_at_us,
                family_routes=routes,
                causal_input=causal_input,
                causal_input_sha256=causal_hash,
                estimates=estimates,
                decision=decision,
            )
            return estimates, decision, receipt
        except OnlinePlacementError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def update_phone_residency(
        self, snapshot: PhoneResidencySnapshot
    ) -> None:
        if self.phone_residency_plan is None:
            raise UnifiedScheduleError("phone residency is not configured")
        if not isinstance(snapshot, PhoneResidencySnapshot):
            raise UnifiedScheduleError("phone residency snapshot is invalid")
        try:
            snapshot.validate_against(self.phone_residency_plan)
        except PhoneResidencyError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self.phone_residency_snapshot = snapshot

    def update_dynamic_residency(
        self, snapshot: DynamicResidencySnapshot
    ) -> None:
        if not isinstance(snapshot, DynamicResidencySnapshot):
            raise UnifiedScheduleError(
                "dynamic residency snapshot is invalid"
            )
        unknown_execution_resources = {
            resource_id
            for placement in snapshot.placements.values()
            for resource_id in placement.spec.execution_resource_ids
            if resource_id not in self._resource_ids
        }
        if unknown_execution_resources:
            raise UnifiedScheduleError(
                "dynamic placement execution resource is absent: "
                + ", ".join(sorted(unknown_execution_resources))
            )
        phone_plan = self.phone_residency_plan
        if (
            phone_plan is not None
            and phone_plan.memory_resource_id in snapshot.memory
        ):
            phone_memory = snapshot.memory[phone_plan.memory_resource_id]
            if (
                phone_memory.capacity_bytes
                    != phone_plan.memory_capacity_bytes
                or phone_memory.reserve_bytes
                    < phone_plan.minimum_available_bytes
            ):
                raise UnifiedScheduleError(
                    "dynamic phone memory contract differs from its plan"
                )
        current = self.dynamic_residency_snapshot
        if current is not None:
            if self._pending_dynamic_transition_id is not None:
                raise UnifiedScheduleError(
                    "dynamic residency refresh is blocked by a transition"
                )
            if any(
                placement.active_leases
                for placement in current.placements.values()
            ):
                raise UnifiedScheduleError(
                    "dynamic residency refresh is blocked by placement leases"
                )
            if (
                snapshot.generation != current.generation
                or snapshot.epoch_key != current.epoch_key
            ):
                raise UnifiedScheduleError(
                    "dynamic residency epoch change requires a receipt"
                )
            if snapshot.captured_at_us < current.captured_at_us:
                raise UnifiedScheduleError(
                    "dynamic residency snapshot moves backward"
                )
            if set(snapshot.placements) != set(current.placements):
                raise UnifiedScheduleError(
                    "dynamic residency refresh changes placements"
                )
            for placement_id, placement in snapshot.placements.items():
                previous = current.placements[placement_id]
                if (
                    placement.spec != previous.spec
                    or placement.generation != previous.generation
                    or placement.resident_since_us
                        != previous.resident_since_us
                    or placement.minimum_resident_until_us
                        != previous.minimum_resident_until_us
                    or placement.active_leases != previous.active_leases
                ):
                    raise UnifiedScheduleError(
                        "dynamic residency refresh changes placement identity"
                    )
            if set(snapshot.memory) != set(current.memory):
                raise UnifiedScheduleError(
                    "dynamic residency refresh changes memory resources"
                )
            for resource_id, capacity in snapshot.memory.items():
                previous = current.memory[resource_id]
                if (
                    capacity.capacity_bytes != previous.capacity_bytes
                    or capacity.reserve_bytes != previous.reserve_bytes
                ):
                    raise UnifiedScheduleError(
                        "dynamic residency refresh changes memory capacity"
                    )
        self.dynamic_residency_snapshot = snapshot

    def _adjust_dynamic_placement_leases(
        self,
        placement_ids: Sequence[str],
        delta: int,
        at_us: int,
    ) -> None:
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        identifiers = tuple(placement_ids)
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise UnifiedScheduleError(
                "dynamic placement lease ids are invalid"
            )
        placements = dict(snapshot.placements)
        for placement_id in identifiers:
            placement = placements.get(placement_id)
            if placement is None:
                raise UnifiedScheduleError(
                    f"dynamic placement is absent: {placement_id}"
                )
            active_leases = placement.active_leases + delta
            if active_leases < 0:
                raise UnifiedScheduleError(
                    "dynamic placement lease count underflow"
                )
            placements[placement_id] = replace(
                placement, active_leases=active_leases
            )
        self.dynamic_residency_snapshot = DynamicResidencySnapshot(
            snapshot_id=snapshot.snapshot_id,
            epoch_key=snapshot.epoch_key,
            generation=snapshot.generation,
            captured_at_us=min(
                max(snapshot.captured_at_us, at_us),
                snapshot.valid_until_us - 1,
            ),
            valid_until_us=snapshot.valid_until_us,
            memory=snapshot.memory,
            placements=placements,
        )

    def acquire_dynamic_placements(
        self,
        placement_ids: Sequence[str],
        *,
        owner_id: str,
        now_us: int,
    ) -> DynamicPlacementLease:
        owner_id = _text("dynamic placement owner_id", owner_id)
        if type(now_us) is not int or now_us < 0:
            raise UnifiedScheduleError(
                "dynamic placement acquisition time must be non-negative"
            )
        identifiers = tuple(
            _text("dynamic placement lease id", placement_id)
            for placement_id in placement_ids
        )
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise UnifiedScheduleError(
                "dynamic placement lease ids must be non-empty and unique"
            )
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        if self._pending_dynamic_transition_id is not None:
            raise UnifiedScheduleError(
                "dynamic placements are blocked by a residency transition"
            )
        if not (
            snapshot.captured_at_us <= now_us < snapshot.valid_until_us
        ):
            raise UnifiedScheduleError(
                "dynamic residency snapshot is expired"
            )
        missing = sorted(set(identifiers) - set(snapshot.placements))
        if missing:
            raise UnifiedScheduleError(
                "dynamic placement is absent: " + ", ".join(missing)
            )
        lease_id = (
            f"placement:{self._next_dynamic_placement_lease}:{owner_id}"
        )
        self._next_dynamic_placement_lease += 1
        lease = DynamicPlacementLease(
            lease_id=lease_id,
            owner_id=owner_id,
            placement_ids=identifiers,
            source_snapshot_id=snapshot.snapshot_id,
            source_generation=snapshot.generation,
            source_epoch_key=snapshot.epoch_key,
            acquired_at_us=now_us,
        )
        self._adjust_dynamic_placement_leases(identifiers, 1, now_us)
        self._active_dynamic_placement_leases[lease_id] = lease
        return lease

    def release_dynamic_placements(
        self,
        lease: DynamicPlacementLease,
        actual_end_us: int,
    ) -> None:
        if not isinstance(lease, DynamicPlacementLease):
            raise UnifiedScheduleError(
                "dynamic placement lease is invalid"
            )
        if self._active_dynamic_placement_leases.get(lease.lease_id) != lease:
            raise UnifiedScheduleError(
                "dynamic placement lease is not active"
            )
        if (
            type(actual_end_us) is not int
            or actual_end_us < lease.acquired_at_us
        ):
            raise UnifiedScheduleError(
                "dynamic placement release precedes acquisition"
            )
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None or (
            snapshot.snapshot_id != lease.source_snapshot_id
            or snapshot.generation != lease.source_generation
            or snapshot.epoch_key != lease.source_epoch_key
        ):
            raise UnifiedScheduleError(
                "dynamic placement epoch changed before release"
            )
        self._adjust_dynamic_placement_leases(
            lease.placement_ids, -1, actual_end_us
        )
        del self._active_dynamic_placement_leases[lease.lease_id]

    def _phone_dynamic_placement_ids(
        self,
        plan: PhoneResidencyPlan,
        arm: PhoneArmSignal | PhoneArmGroup | None,
    ) -> tuple[str, ...]:
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            return ()
        managed = tuple(
            placement
            for placement in snapshot.placements.values()
            if placement.spec.resource_id == plan.memory_resource_id
        )
        if not managed:
            return ()
        if self._pending_dynamic_transition_id is not None:
            raise UnifiedScheduleError(
                "phone offload is blocked by a residency transition"
            )
        if isinstance(arm, PhoneArmSignal):
            signals = (arm,)
        elif isinstance(arm, PhoneArmGroup):
            signals = arm.signals
        else:
            raise UnifiedScheduleError(
                "selected phone offload has no arm signal"
            )
        placement_ids: list[str] = []
        for signal in signals:
            matches = [
                placement
                for placement in managed
                if placement.spec.slice_id == signal.slice_id
            ]
            if len(matches) != 1:
                raise UnifiedScheduleError(
                    "dynamic phone placement is missing or ambiguous: "
                    + signal.slice_id
                )
            placement = matches[0]
            spec = placement.spec
            if (
                spec.model_hash != signal.model_hash
                or spec.weight_hash != signal.weight_hash
                or signal.shared_compute_resource_id
                    not in spec.execution_resource_ids
                or signal.session_id not in spec.runtime_binding_ids
                or signal.compute_backend not in spec.runtime_binding_ids
            ):
                raise UnifiedScheduleError(
                    "dynamic phone placement identity mismatch: "
                    + signal.slice_id
                )
            placement_ids.append(placement.placement_id)
        if len(placement_ids) != len(set(placement_ids)):
            raise UnifiedScheduleError(
                "phone arm repeats a dynamic placement"
            )
        return tuple(placement_ids)

    def update_lifecycle(
        self,
        state_key: str,
        receipt: LifecycleReceipt | None,
    ) -> str:
        try:
            profile_set = self._profile_sets[state_key]
        except KeyError as exc:
            raise UnifiedScheduleError(
                f"unknown lifecycle state key: {state_key}"
            ) from exc
        state = profile_set.select_state(receipt)
        self._lifecycle_receipts[state_key] = receipt
        return state

    def profile_for(self, workload_id: str) -> ProfileBundle:
        direct = self._direct_workloads.get(workload_id)
        if direct is not None:
            return self._route_policies[direct].profile
        state_key = self._lifecycle_workloads.get(workload_id)
        if state_key is None:
            raise UnifiedScheduleError(
                f"request has no profile owner: {workload_id}"
            )
        profile_set = self._profile_sets[state_key]
        state = profile_set.select_state(
            self._lifecycle_receipts[state_key]
        )
        return profile_set.profiles[state]

    def schedule(
        self,
        request: Request,
        runtime_now_us: int | None = None,
    ) -> Decision:
        profile = self.profile_for(request.workload_id)
        try:
            return self._route_policies[profile.profile_id].schedule(
                request,
                runtime_now_us,
            )
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def schedule_dynamic_residency(
        self,
        candidates: Sequence[DynamicResidencyCandidate],
        *,
        now_us: int,
        minimum_energy_saving_ppm: int = 50_000,
        latency_limit_ppm: int = 1_000_000,
        require_measured: bool = True,
    ) -> DynamicResidencySchedule:
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        if self._pending_dynamic_transition_id is not None:
            raise UnifiedScheduleError(
                "a dynamic residency transition is already pending"
            )
        if any(
            placement.active_leases
            for placement in snapshot.placements.values()
        ):
            raise UnifiedScheduleError(
                "dynamic residency waits for active placement leases"
            )
        candidate_rows = tuple(candidates)
        if any(
            not isinstance(candidate, DynamicResidencyCandidate)
            for candidate in candidate_rows
        ):
            raise UnifiedScheduleError(
                "dynamic residency candidate is invalid"
            )
        candidate_ids = [candidate.candidate_id for candidate in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise UnifiedScheduleError(
                "duplicate dynamic residency candidate id"
            )
        unknown_execution_resources = {
            resource_id
            for candidate in candidate_rows
            for resource_id in candidate.target.execution_resource_ids
            if resource_id not in self._resource_ids
        }
        if unknown_execution_resources:
            raise UnifiedScheduleError(
                "dynamic target execution resource is absent: "
                + ", ".join(sorted(unknown_execution_resources))
            )

        previews = {}
        ready_us: dict[str, int | None] = {}
        for candidate in candidate_rows:
            preview_at_us = max(
                now_us,
                (
                    now_us
                    if candidate.fallback_contract is None
                    else candidate.fallback_contract.ready_at_us
                ),
            )
            demands = tuple(
                LeaseDemand(
                    lease_id=(
                        f"residency:{candidate.candidate_id}:{index}"
                    ),
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=(
                        candidate.protected_transition_latency_mean_us
                    ),
                    duration_upper_us=(
                        candidate.protected_transition_latency_upper_us
                    ),
                )
                for index, resource_id in enumerate(
                    candidate.transition_resource_ids
                )
            )
            try:
                preview = self.timeline.preview_leases(
                    demands,
                    preview_at_us,
                    candidate.protected_transition_latency_mean_us,
                    candidate.protected_transition_latency_upper_us,
                )
            except SchedulerError as exc:
                if "resource is not ready" not in str(exc):
                    raise UnifiedScheduleError(str(exc)) from exc
                ready_us[candidate.candidate_id] = None
                continue
            previews[candidate.candidate_id] = preview
            ready_us[candidate.candidate_id] = preview.start_us

        try:
            decision = select_dynamic_residency_transition(
                snapshot,
                candidate_rows,
                now_us=now_us,
                transition_resource_ready_us=ready_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                latency_limit_ppm=latency_limit_ppm,
                require_measured=require_measured,
            )
        except DynamicResidencyError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if decision.candidate_id is None:
            return DynamicResidencySchedule(
                decision=decision,
                owner_id=None,
                leases=(),
                queue_by_resource_us=MappingProxyType({}),
                blocking_resources=(),
            )
        preview = previews[decision.candidate_id]
        assert decision.transition_id is not None
        owner_id = "dynamic:" + decision.transition_id
        try:
            leases = self.timeline.commit_leases(preview, owner_id)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self._pending_dynamic_transition_id = decision.transition_id
        schedule = DynamicResidencySchedule(
            decision=decision,
            owner_id=owner_id,
            leases=leases,
            queue_by_resource_us=preview.queue_by_resource_us,
            blocking_resources=preview.blocking_resources,
        )
        self._pending_dynamic_schedule = schedule
        return schedule

    def complete_dynamic_residency(
        self,
        schedule: DynamicResidencySchedule,
        receipt: DynamicResidencyReceipt,
    ) -> DynamicResidencySnapshot:
        if not isinstance(schedule, DynamicResidencySchedule):
            raise UnifiedScheduleError(
                "dynamic residency schedule is invalid"
            )
        if not isinstance(receipt, DynamicResidencyReceipt):
            raise UnifiedScheduleError(
                "dynamic residency receipt is invalid"
            )
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        if (
            schedule.decision.transition_id is None
            or schedule.decision.transition_id
                != self._pending_dynamic_transition_id
            or schedule != self._pending_dynamic_schedule
        ):
            raise UnifiedScheduleError(
                "dynamic residency transition is not pending"
            )
        try:
            result = apply_dynamic_residency_receipt(
                snapshot, schedule.decision, receipt
            )
        except DynamicResidencyError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        for lease in schedule.leases:
            end_us = max(
                lease.start_us,
                min(receipt.completed_at_us, lease.reserved_until_us),
            )
            self.release(lease.token, end_us)
        self.dynamic_residency_snapshot = result
        self._pending_dynamic_transition_id = None
        self._pending_dynamic_schedule = None
        return result

    def schedule_gpu_backfill(
        self,
        bubble: GpuBubbleWindow,
        candidates: Sequence[GpuBackfillCandidate],
        *,
        now_us: int,
        minimum_energy_saving_ppm: int = 50_000,
        require_measured: bool = True,
    ) -> GpuBackfillSchedule:
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        if self._pending_dynamic_transition_id is not None:
            raise UnifiedScheduleError(
                "GPU backfill is blocked by a residency transition"
            )
        candidate_rows = tuple(candidates)
        if any(
            not isinstance(candidate, GpuBackfillCandidate)
            for candidate in candidate_rows
        ):
            raise UnifiedScheduleError("GPU backfill candidate is invalid")
        candidate_ids = [candidate.candidate_id for candidate in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise UnifiedScheduleError("duplicate GPU backfill candidate id")

        previews = {}
        ready_us: dict[str, int | None] = {}
        for candidate in candidate_rows:
            demands = tuple(
                LeaseDemand(
                    lease_id=f"gpu-backfill:{candidate.candidate_id}:{index}",
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=candidate.lease_duration_mean_us,
                    duration_upper_us=candidate.lease_duration_upper_us,
                )
                for index, resource_id in enumerate(
                    candidate.execution_resource_ids
                )
            )
            try:
                preview = self.timeline.preview_leases(
                    demands,
                    now_us,
                    candidate.lease_duration_mean_us,
                    candidate.lease_duration_upper_us,
                )
            except SchedulerError as exc:
                if "resource is not ready" not in str(exc):
                    raise UnifiedScheduleError(str(exc)) from exc
                ready_us[candidate.candidate_id] = None
                continue
            previews[candidate.candidate_id] = preview
            ready_us[candidate.candidate_id] = (
                preview.start_us
                if preview.finish_upper_us <= candidate.deadline_us
                else None
            )

        try:
            decision = select_gpu_backfill(
                snapshot,
                bubble,
                candidate_rows,
                now_us=now_us,
                resource_ready_us=ready_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                require_measured=require_measured,
            )
        except GpuBackfillError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if decision.candidate_id is None:
            return GpuBackfillSchedule(
                decision=decision,
                owner_id=None,
                leases=(),
                queue_by_resource_us=MappingProxyType({}),
                blocking_resources=(),
            )
        preview = previews[decision.candidate_id]
        owner_id = (
            f"gpu-backfill:{self._next_gpu_backfill_owner}:"
            f"{bubble.bubble_id}:{decision.candidate_id}"
        )
        self._next_gpu_backfill_owner += 1
        try:
            leases = self.timeline.commit_leases(preview, owner_id)
            self._adjust_dynamic_placement_leases(
                decision.required_placement_ids, 1, now_us
            )
        except (SchedulerError, UnifiedScheduleError) as exc:
            try:
                self.timeline.cancel_owner(owner_id, now_us)
            except SchedulerError:
                pass
            raise UnifiedScheduleError(str(exc)) from exc
        schedule = GpuBackfillSchedule(
            decision=decision,
            owner_id=owner_id,
            leases=leases,
            queue_by_resource_us=preview.queue_by_resource_us,
            blocking_resources=preview.blocking_resources,
        )
        self._active_gpu_backfills[owner_id] = schedule
        return schedule

    def release_gpu_backfill(
        self,
        schedule: GpuBackfillSchedule,
        actual_end_us: int,
    ) -> None:
        if not isinstance(schedule, GpuBackfillSchedule):
            raise UnifiedScheduleError("GPU backfill schedule is invalid")
        if schedule.decision.candidate_id is None:
            if schedule.owner_id is not None or schedule.leases:
                raise UnifiedScheduleError(
                    "rejected GPU backfill carries resource leases"
                )
            return
        owner_id = schedule.owner_id
        if (
            owner_id is None
            or self._active_gpu_backfills.get(owner_id) != schedule
        ):
            raise UnifiedScheduleError("GPU backfill schedule is not active")
        if type(actual_end_us) is not int or actual_end_us < 0:
            raise UnifiedScheduleError(
                "GPU backfill completion time must be non-negative"
            )
        if (
            schedule.decision.start_us is None
            or schedule.decision.restore_finish_upper_us is None
            or actual_end_us < schedule.decision.start_us
            or actual_end_us
                > schedule.decision.restore_finish_upper_us
            or any(
                actual_end_us < lease.start_us
                or actual_end_us > lease.reserved_until_us
                for lease in schedule.leases
            )
        ):
            raise UnifiedScheduleError(
                "GPU backfill completion is outside its decision envelope"
            )
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None or (
            snapshot.generation != schedule.decision.source_generation
            or snapshot.epoch_key != schedule.decision.source_epoch_key
        ):
            raise UnifiedScheduleError(
                "GPU backfill residency epoch changed before release"
            )
        for lease in schedule.leases:
            self.release(lease.token, actual_end_us)
        self._adjust_dynamic_placement_leases(
            schedule.decision.required_placement_ids, -1, actual_end_us
        )
        del self._active_gpu_backfills[owner_id]

    def abort_gpu_backfill(
        self,
        schedule: GpuBackfillSchedule,
        at_us: int,
    ) -> None:
        if not isinstance(schedule, GpuBackfillSchedule):
            raise UnifiedScheduleError("GPU backfill schedule is invalid")
        owner_id = schedule.owner_id
        if (
            schedule.decision.candidate_id is None
            or owner_id is None
            or self._active_gpu_backfills.get(owner_id) != schedule
        ):
            raise UnifiedScheduleError("GPU backfill schedule is not active")
        if type(at_us) is not int or at_us < 0:
            raise UnifiedScheduleError(
                "GPU backfill abort time must be non-negative"
            )
        try:
            self.timeline.cancel_owner(owner_id, at_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self._adjust_dynamic_placement_leases(
            schedule.decision.required_placement_ids, -1, at_us
        )
        del self._active_gpu_backfills[owner_id]

    def _gpu_backfill_ready_times(
        self,
        candidates: Sequence[GpuBackfillCandidate],
        now_us: int,
    ) -> dict[str, int | None]:
        ready_us: dict[str, int | None] = {}
        for candidate in candidates:
            demands = tuple(
                LeaseDemand(
                    lease_id=(
                        f"gpu-wavefront-preview:{candidate.candidate_id}:"
                        f"{index}"
                    ),
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=candidate.lease_duration_mean_us,
                    duration_upper_us=candidate.lease_duration_upper_us,
                )
                for index, resource_id in enumerate(
                    candidate.execution_resource_ids
                )
            )
            try:
                preview = self.timeline.preview_leases(
                    demands,
                    now_us,
                    candidate.lease_duration_mean_us,
                    candidate.lease_duration_upper_us,
                )
            except SchedulerError as exc:
                if "resource is not ready" not in str(exc):
                    raise UnifiedScheduleError(str(exc)) from exc
                ready_us[candidate.candidate_id] = None
                continue
            ready_us[candidate.candidate_id] = (
                preview.start_us
                if preview.finish_upper_us <= candidate.deadline_us
                else None
            )
        return ready_us

    def schedule_gpu_wavefront_backfill(
        self,
        bubble: GpuBubbleWindow,
        wavefront: GpuWavefrontSnapshot,
        *,
        now_us: int,
        minimum_energy_saving_ppm: int = 50_000,
        require_measured: bool = True,
        objective: str = "coverage_then_energy",
    ) -> GpuWavefrontSchedule:
        snapshot = self.dynamic_residency_snapshot
        if snapshot is None:
            raise UnifiedScheduleError(
                "dynamic residency snapshot is required"
            )
        if self._pending_dynamic_transition_id is not None:
            raise UnifiedScheduleError(
                "GPU wavefront is blocked by a residency transition"
            )
        if not isinstance(wavefront, GpuWavefrontSnapshot):
            raise UnifiedScheduleError("GPU wavefront snapshot is invalid")

        active_pipelines = {
            schedule.decision.pipeline_id
            for schedule in self._active_gpu_wavefronts.values()
        }
        for pipeline_id, next_index in (
            wavefront.next_sequence_by_pipeline.items()
        ):
            current = self._gpu_wavefront_next_by_pipeline.get(pipeline_id)
            if current is not None and current != next_index:
                raise UnifiedScheduleError(
                    "GPU wavefront pipeline sequence is stale"
                )
            receipt = self._gpu_wavefront_last_receipt_by_pipeline.get(
                pipeline_id
            )
            if (
                receipt is not None
                and receipt not in wavefront.completed_output_receipt_ids
            ):
                raise UnifiedScheduleError(
                    "GPU wavefront omits the last output receipt"
                )
        for chunk in wavefront.ready_chunks:
            if chunk.chunk_id in self._completed_gpu_wavefront_chunks:
                raise UnifiedScheduleError(
                    "GPU wavefront repeats a completed chunk"
                )
            if chunk.pipeline_id in active_pipelines:
                raise UnifiedScheduleError(
                    "GPU wavefront pipeline already has active GPU work"
                )

        candidates = tuple(
            chunk.candidate for chunk in wavefront.ready_chunks
        )
        ready_us = self._gpu_backfill_ready_times(candidates, now_us)
        try:
            decision = select_gpu_wavefront_backfill(
                snapshot,
                bubble,
                wavefront,
                now_us=now_us,
                resource_ready_us=ready_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                require_measured=require_measured,
                objective=objective,
            )
        except GpuBackfillError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if decision.chunk_id is None:
            backfill = GpuBackfillSchedule(
                decision=decision.backfill,
                owner_id=None,
                leases=(),
                queue_by_resource_us=MappingProxyType({}),
                blocking_resources=(),
            )
            return GpuWavefrontSchedule(
                decision=decision,
                backfill_schedule=backfill,
            )

        selected = next(
            chunk
            for chunk in wavefront.ready_chunks
            if chunk.chunk_id == decision.chunk_id
        )
        current = self._gpu_wavefront_next_by_pipeline.get(
            selected.pipeline_id,
            wavefront.next_sequence_by_pipeline[selected.pipeline_id],
        )
        if current != selected.sequence_index:
            raise UnifiedScheduleError(
                "GPU wavefront selected an out-of-order chunk"
            )
        last_receipt = self._gpu_wavefront_last_receipt_by_pipeline.get(
            selected.pipeline_id
        )
        if (
            last_receipt is not None
            and selected.predecessor_output_receipt_id != last_receipt
        ):
            raise UnifiedScheduleError(
                "GPU wavefront predecessor receipt is stale"
            )

        backfill = self.schedule_gpu_backfill(
            bubble,
            (selected.candidate,),
            now_us=now_us,
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            require_measured=require_measured,
        )
        if backfill.decision.candidate_id != decision.candidate_id:
            raise UnifiedScheduleError(
                "GPU wavefront selection changed during lease commit"
            )
        schedule = GpuWavefrontSchedule(
            decision=decision,
            backfill_schedule=backfill,
        )
        if schedule.owner_id is None:
            raise UnifiedScheduleError(
                "selected GPU wavefront has no resource owner"
            )
        self._active_gpu_wavefronts[schedule.owner_id] = schedule
        return schedule

    def release_gpu_wavefront_backfill(
        self,
        schedule: GpuWavefrontSchedule,
        actual_end_us: int,
        output_receipt_id: str,
    ) -> None:
        if not isinstance(schedule, GpuWavefrontSchedule):
            raise UnifiedScheduleError("GPU wavefront schedule is invalid")
        if schedule.decision.chunk_id is None:
            if schedule.owner_id is not None or schedule.leases:
                raise UnifiedScheduleError(
                    "rejected GPU wavefront carries resource leases"
                )
            return
        owner_id = schedule.owner_id
        if (
            owner_id is None
            or self._active_gpu_wavefronts.get(owner_id) != schedule
        ):
            raise UnifiedScheduleError(
                "GPU wavefront schedule is not active"
            )
        _text("GPU wavefront output_receipt_id", output_receipt_id)
        if output_receipt_id in self._completed_gpu_wavefront_chunks.values():
            raise UnifiedScheduleError(
                "GPU wavefront output receipt is duplicated"
            )
        pipeline_id = schedule.decision.pipeline_id
        sequence_index = schedule.decision.sequence_index
        chunk_id = schedule.decision.chunk_id
        if pipeline_id is None or sequence_index is None:
            raise UnifiedScheduleError(
                "selected GPU wavefront sequence is incomplete"
            )
        current = self._gpu_wavefront_next_by_pipeline.get(
            pipeline_id, sequence_index
        )
        if current != sequence_index:
            raise UnifiedScheduleError(
                "GPU wavefront completion is out of order"
            )

        self.release_gpu_backfill(
            schedule.backfill_schedule,
            actual_end_us,
        )
        self._gpu_wavefront_next_by_pipeline[pipeline_id] = (
            sequence_index + 1
        )
        self._gpu_wavefront_last_receipt_by_pipeline[
            pipeline_id
        ] = output_receipt_id
        self._completed_gpu_wavefront_chunks[chunk_id] = output_receipt_id
        del self._active_gpu_wavefronts[owner_id]

    def abort_gpu_wavefront_backfill(
        self,
        schedule: GpuWavefrontSchedule,
        at_us: int,
    ) -> None:
        if not isinstance(schedule, GpuWavefrontSchedule):
            raise UnifiedScheduleError("GPU wavefront schedule is invalid")
        owner_id = schedule.owner_id
        if (
            schedule.decision.chunk_id is None
            or owner_id is None
            or self._active_gpu_wavefronts.get(owner_id) != schedule
        ):
            raise UnifiedScheduleError(
                "GPU wavefront schedule is not active"
            )
        self.abort_gpu_backfill(schedule.backfill_schedule, at_us)
        del self._active_gpu_wavefronts[owner_id]

    def schedule_phone_offload(
        self,
        candidates: Sequence[PhoneOffloadCandidate],
        *,
        request_id: str,
        route_id: str,
        physical_m: int,
        now_us: int,
        deadline_us: int,
        minimum_energy_saving_ppm: int = 50_000,
        latency_limit_ppm: int = 1_000_000,
        maximum_join_wait_ppm: int = 50_000,
        require_measured: bool = True,
    ) -> PhoneOffloadSchedule:
        plan = self.phone_residency_plan
        snapshot = self.phone_residency_snapshot
        if plan is None or snapshot is None:
            raise UnifiedScheduleError(
                "phone residency plan and warm receipts are required"
            )
        candidate_rows = tuple(candidates)
        if any(
            not isinstance(candidate, PhoneOffloadCandidate)
            for candidate in candidate_rows
        ):
            raise UnifiedScheduleError("phone offload candidate is invalid")
        candidate_ids = [candidate.candidate_id for candidate in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise UnifiedScheduleError("duplicate phone candidate id")

        previews = {}
        ready_us: dict[str, int | None] = {}
        for candidate in candidate_rows:
            demands = tuple(
                LeaseDemand(
                    lease_id=(
                        f"phone-path:{candidate.candidate_id}:{index}"
                    ),
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=candidate.phone_path_us.mean,
                    duration_upper_us=candidate.phone_path_us.upper,
                )
                for index, resource_id in enumerate(
                    plan.execution_resource_ids
                )
            )
            try:
                preview = self.timeline.preview_leases(
                    demands,
                    now_us,
                    candidate.phone_path_us.mean,
                    candidate.phone_path_us.upper,
                )
            except SchedulerError as exc:
                if "resource is not ready" not in str(exc):
                    raise UnifiedScheduleError(str(exc)) from exc
                ready_us[candidate.candidate_id] = None
                continue
            previews[candidate.candidate_id] = preview
            ready_us[candidate.candidate_id] = preview.start_us

        try:
            decision = select_energy_positive_offload(
                plan,
                snapshot,
                candidate_rows,
                request_id=request_id,
                route_id=route_id,
                physical_m=physical_m,
                now_us=now_us,
                phone_resource_ready_us=ready_us,
                deadline_us=deadline_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                latency_limit_ppm=latency_limit_ppm,
                maximum_join_wait_ppm=maximum_join_wait_ppm,
                require_measured=require_measured,
            )
        except PhoneResidencyError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

        if decision.candidate_id is None:
            return PhoneOffloadSchedule(
                decision=decision,
                owner_id=None,
                leases=(),
                queue_by_resource_us=MappingProxyType({}),
                blocking_resources=(),
                placement_ids=(),
                source_snapshot_id=None,
                source_generation=None,
                source_epoch_key=None,
            )
        preview = previews[decision.candidate_id]
        placement_ids = self._phone_dynamic_placement_ids(
            plan, decision.arm_signal
        )
        owner_id = (
            f"phone:{self._next_phone_owner}:{request_id}:"
            f"{route_id}:{decision.candidate_id}"
        )
        self._next_phone_owner += 1
        try:
            leases = self.timeline.commit_leases(preview, owner_id)
            if placement_ids:
                self._adjust_dynamic_placement_leases(
                    placement_ids, 1, now_us
                )
        except (SchedulerError, UnifiedScheduleError) as exc:
            try:
                self.timeline.cancel_owner(owner_id, now_us)
            except SchedulerError:
                pass
            raise UnifiedScheduleError(str(exc)) from exc
        dynamic_snapshot = self.dynamic_residency_snapshot
        schedule = PhoneOffloadSchedule(
            decision=decision,
            owner_id=owner_id,
            leases=leases,
            queue_by_resource_us=preview.queue_by_resource_us,
            blocking_resources=preview.blocking_resources,
            placement_ids=placement_ids,
            source_snapshot_id=(
                dynamic_snapshot.snapshot_id if placement_ids else None
            ),
            source_generation=(
                dynamic_snapshot.generation if placement_ids else None
            ),
            source_epoch_key=(
                dynamic_snapshot.epoch_key if placement_ids else None
            ),
        )
        if placement_ids:
            self._active_phone_placement_schedules[owner_id] = schedule
        return schedule

    def schedule_phone_arbiter_work(
        self,
        window: PhoneArbiterWindow,
        queue: PhoneArbiterQueue,
        *,
        now_us: int,
        minimum_energy_saving_ppm: int = 50_000,
        latency_limit_ppm: int = 1_000_000,
        maximum_join_wait_ppm: int = 50_000,
        require_measured: bool = True,
    ) -> PhoneArbiterSchedule:
        plan = self.phone_residency_plan
        snapshot = self.phone_residency_snapshot
        if plan is None or snapshot is None:
            raise UnifiedScheduleError(
                "phone residency plan and warm receipts are required"
            )
        if not isinstance(window, PhoneArbiterWindow):
            raise UnifiedScheduleError("phone arbiter window is invalid")
        if not isinstance(queue, PhoneArbiterQueue):
            raise UnifiedScheduleError("phone arbiter queue is invalid")

        active_pipelines = {
            schedule.decision.pipeline_id
            for schedule in self._active_phone_arbiters.values()
        }
        for pipeline_id, next_index in queue.next_sequence_by_pipeline.items():
            current = self._phone_arbiter_next_by_pipeline.get(pipeline_id)
            if current is not None and current != next_index:
                raise UnifiedScheduleError(
                    "phone arbiter pipeline sequence is stale"
                )
            receipt = self._phone_arbiter_last_receipt_by_pipeline.get(
                pipeline_id
            )
            if (
                receipt is not None
                and receipt not in queue.completed_output_receipt_ids
            ):
                raise UnifiedScheduleError(
                    "phone arbiter omits the last output receipt"
                )
        for work in queue.ready_work:
            if work.work_id in self._completed_phone_arbiter_work:
                raise UnifiedScheduleError(
                    "phone arbiter repeats completed work"
                )
            if work.pipeline_id in active_pipelines:
                raise UnifiedScheduleError(
                    "phone arbiter pipeline already has active work"
                )

        ready_us: dict[str, int | None] = {}
        for work in queue.ready_work:
            candidate = work.candidate
            demands = tuple(
                LeaseDemand(
                    lease_id=(
                        f"phone-arbiter-preview:{work.work_id}:{index}"
                    ),
                    resource_id=resource_id,
                    slots=1,
                    start_offset_us=0,
                    duration_us=candidate.phone_path_us.mean,
                    duration_upper_us=candidate.phone_path_us.upper,
                )
                for index, resource_id in enumerate(
                    plan.execution_resource_ids
                )
            )
            try:
                preview = self.timeline.preview_leases(
                    demands,
                    now_us,
                    candidate.phone_path_us.mean,
                    candidate.phone_path_us.upper,
                )
            except SchedulerError as exc:
                if "resource is not ready" not in str(exc):
                    raise UnifiedScheduleError(str(exc)) from exc
                ready_us[work.work_id] = None
                continue
            ready_us[work.work_id] = preview.start_us

        try:
            decision = select_phone_arbiter_work(
                plan,
                snapshot,
                window,
                queue,
                now_us=now_us,
                resource_ready_us=ready_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                latency_limit_ppm=latency_limit_ppm,
                maximum_join_wait_ppm=maximum_join_wait_ppm,
                require_measured=require_measured,
            )
        except PhoneArbiterError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if decision.work_id is None:
            return PhoneArbiterSchedule(
                decision=decision,
                phone_schedule=None,
            )

        selected = next(
            work for work in queue.ready_work
            if work.work_id == decision.work_id
        )
        current = self._phone_arbiter_next_by_pipeline.get(
            selected.pipeline_id,
            queue.next_sequence_by_pipeline[selected.pipeline_id],
        )
        if current != selected.sequence_index:
            raise UnifiedScheduleError(
                "phone arbiter selected out-of-order work"
            )
        last_receipt = self._phone_arbiter_last_receipt_by_pipeline.get(
            selected.pipeline_id
        )
        if (
            last_receipt is not None
            and selected.predecessor_output_receipt_id != last_receipt
        ):
            raise UnifiedScheduleError(
                "phone arbiter predecessor receipt is stale"
            )

        safe_end_us = decision.safe_end_us
        if safe_end_us is None:
            raise UnifiedScheduleError(
                "selected phone arbiter work has no safe end"
            )
        phone_schedule = self.schedule_phone_offload(
            (selected.candidate,),
            request_id=selected.request_id,
            route_id=selected.route_id,
            physical_m=selected.physical_m,
            now_us=now_us,
            deadline_us=safe_end_us,
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            latency_limit_ppm=latency_limit_ppm,
            maximum_join_wait_ppm=maximum_join_wait_ppm,
            require_measured=require_measured,
        )
        committed_arm = phone_schedule.decision.arm_signal
        committed_signals = (
            committed_arm.signals
            if isinstance(committed_arm, PhoneArmGroup)
            else (() if committed_arm is None else (committed_arm,))
        )
        if (
            phone_schedule.decision.candidate_id != decision.work_id
            or not committed_signals
            or decision.start_us is None
            or decision.phone_finish_upper_us is None
            or committed_signals[0].execute_not_before_us
                != decision.start_us
            or any(
                signal.deadline_us != safe_end_us
                for signal in committed_signals
            )
        ):
            raise UnifiedScheduleError(
                "phone arbiter selection changed during lease commit"
            )
        owner_id = phone_schedule.owner_id
        if owner_id is None:
            raise UnifiedScheduleError(
                "selected phone arbiter work has no resource owner"
            )
        schedule = PhoneArbiterSchedule(
            decision=decision,
            phone_schedule=phone_schedule,
        )
        self._active_phone_arbiters[owner_id] = schedule
        return schedule

    def release_phone_arbiter_work(
        self,
        schedule: PhoneArbiterSchedule,
        actual_end_us: int,
        output_receipt_id: str,
    ) -> None:
        if not isinstance(schedule, PhoneArbiterSchedule):
            raise UnifiedScheduleError("phone arbiter schedule is invalid")
        if schedule.decision.work_id is None:
            if schedule.phone_schedule is not None:
                raise UnifiedScheduleError(
                    "rejected phone arbiter work carries a schedule"
                )
            return
        owner_id = schedule.owner_id
        if (
            owner_id is None
            or self._active_phone_arbiters.get(owner_id) != schedule
            or schedule.phone_schedule is None
        ):
            raise UnifiedScheduleError(
                "phone arbiter schedule is not active"
            )
        _text("phone arbiter output_receipt_id", output_receipt_id)
        if output_receipt_id in self._completed_phone_arbiter_work.values():
            raise UnifiedScheduleError(
                "phone arbiter output receipt is duplicated"
            )
        decision = schedule.decision
        if (
            decision.start_us is None
            or decision.phone_finish_upper_us is None
            or type(actual_end_us) is not int
            or actual_end_us < decision.start_us
            or actual_end_us > decision.phone_finish_upper_us
        ):
            raise UnifiedScheduleError(
                "phone arbiter completion is outside its envelope"
            )
        snapshot = self.phone_residency_snapshot
        if snapshot is None or (
            snapshot.snapshot_id != decision.phone_snapshot_id
            or canonical_sha256(snapshot.to_json())
                != decision.phone_snapshot_sha256
        ):
            raise UnifiedScheduleError(
                "phone arbiter residency epoch changed before release"
            )
        pipeline_id = decision.pipeline_id
        sequence_index = decision.sequence_index
        work_id = decision.work_id
        if pipeline_id is None or sequence_index is None:
            raise UnifiedScheduleError(
                "selected phone arbiter sequence is incomplete"
            )
        current = self._phone_arbiter_next_by_pipeline.get(
            pipeline_id, sequence_index
        )
        if current != sequence_index:
            raise UnifiedScheduleError(
                "phone arbiter completion is out of order"
            )

        self.release_phone_offload(schedule.phone_schedule, actual_end_us)
        self._phone_arbiter_next_by_pipeline[pipeline_id] = sequence_index + 1
        self._phone_arbiter_last_receipt_by_pipeline[
            pipeline_id
        ] = output_receipt_id
        self._completed_phone_arbiter_work[work_id] = output_receipt_id
        del self._active_phone_arbiters[owner_id]

    def release_phone_offload(
        self, schedule: PhoneOffloadSchedule, actual_end_us: int
    ) -> None:
        if not isinstance(schedule, PhoneOffloadSchedule):
            raise UnifiedScheduleError("phone offload schedule is invalid")
        if schedule.placement_ids:
            owner_id = schedule.owner_id
            if (
                owner_id is None
                or self._active_phone_placement_schedules.get(owner_id)
                    != schedule
            ):
                raise UnifiedScheduleError(
                    "phone placement schedule is not active"
                )
            dynamic_snapshot = self.dynamic_residency_snapshot
            if dynamic_snapshot is None or (
                dynamic_snapshot.snapshot_id != schedule.source_snapshot_id
                or dynamic_snapshot.generation
                    != schedule.source_generation
                or dynamic_snapshot.epoch_key != schedule.source_epoch_key
            ):
                raise UnifiedScheduleError(
                    "phone placement epoch changed before release"
                )
            if (
                type(actual_end_us) is not int
                or any(
                    actual_end_us < lease.start_us
                    or actual_end_us > lease.reserved_until_us
                    for lease in schedule.leases
                )
            ):
                raise UnifiedScheduleError(
                    "phone completion is outside its lease envelope"
                )
        for lease in schedule.leases:
            end_us = max(
                lease.start_us,
                min(actual_end_us, lease.reserved_until_us),
            )
            self.release(lease.token, end_us)
        if schedule.placement_ids:
            self._adjust_dynamic_placement_leases(
                schedule.placement_ids, -1, actual_end_us
            )
            assert schedule.owner_id is not None
            del self._active_phone_placement_schedules[schedule.owner_id]

    def reserve_external_resource(
        self,
        resource_id: str,
        reservation_id: str,
        start_us: int,
        finish_us: int,
        slots: int = 1,
    ) -> tuple[LeaseRecord, ...]:
        _text("external reservation id", reservation_id)
        if type(start_us) is not int or start_us < 0:
            raise UnifiedScheduleError(
                "external reservation start_us must be non-negative"
            )
        if type(finish_us) is not int or finish_us <= start_us:
            raise UnifiedScheduleError(
                "external reservation finish_us must follow start_us"
            )
        if type(slots) is not int or slots < 1:
            raise UnifiedScheduleError(
                "external reservation slots must be positive"
            )
        duration_us = finish_us - start_us
        demand = LeaseDemand(
            lease_id=f"external:{reservation_id}",
            resource_id=resource_id,
            slots=slots,
            start_offset_us=0,
            duration_us=duration_us,
            duration_upper_us=duration_us,
        )
        try:
            preview = self.timeline.preview_leases(
                (demand,), start_us, duration_us, duration_us
            )
            if preview.start_us != start_us:
                raise UnifiedScheduleError(
                    "external resource window is already busy"
                )
            return self.timeline.commit_leases(
                preview,
                f"external:{reservation_id}",
            )
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def release(self, token: str, actual_end_us: int) -> None:
        try:
            self.timeline.release(token, actual_end_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def extend_lease(self, token: str, reserved_until_us: int) -> int:
        """Extend one committed lease without overlapping later work."""
        try:
            return self.timeline.extend(token, reserved_until_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def release_decision(self, decision: Decision, actual_end_us: int) -> None:
        for lease in decision.leases:
            end_us = max(
                lease.start_us,
                min(actual_end_us, lease.reserved_until_us),
            )
            self.release(lease.token, end_us)

    def cancel(self, owner_id: str, at_us: int) -> tuple[str, ...]:
        try:
            return self.timeline.cancel_owner(owner_id, at_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def set_resource_ready(
        self, resource_id: str, ready: bool, at_us: int
    ) -> tuple[str, ...]:
        if self.matmul is not None and resource_id in self.matmul.profile.resources:
            try:
                return self.matmul.set_resource_ready(
                    resource_id, ready, at_us
                )
            except MatmulScheduleError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
        try:
            if ready:
                self.timeline.restore_resource(resource_id)
                return ()
            return self.timeline.revoke_resource(resource_id, at_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def enqueue_matmul(self, program: ModelProgram) -> None:
        if self.matmul is None:
            raise UnifiedScheduleError("matmul profile is not configured")
        try:
            self.matmul.enqueue(program)
        except MatmulScheduleError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def schedule_next_matmul(self, now_us: int) -> Mapping[str, object] | None:
        if self.matmul is None:
            raise UnifiedScheduleError("matmul profile is not configured")
        try:
            return self.matmul.schedule_next(now_us)
        except MatmulScheduleError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def resource_snapshot(
        self, at_us: int
    ) -> Mapping[str, Mapping[str, object]]:
        try:
            return self.timeline.resource_snapshot(at_us)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
