"""One runtime owner for every scheduling granularity and trace."""

from __future__ import annotations

from dataclasses import replace

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, Sequence
from ._internal.policy import (
    RoutePolicy,
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
    RequestShapeUnsupportedError,
    UnifiedScheduleError,
)
from ._internal.matmul import MatmulSystemProfile, MatmulPlanner
from ._internal.runtime_gates import RuntimeSnapshot
from ._internal.runtime_cost import RuntimeExecutorRegistry
from ._internal.model_manifest import ModelManifest, ModelManifestError
from ._internal.model_manifest_cache import load_cached_gguf_manifest
from ._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCapabilityCatalog,
)
from ._internal.route_generation import (
    AutomatedRouteCompiler,
    RouteGenerationError,
    RuntimeRouteTemplateSet,
    ThermalGateLog,
)
from ._internal.background_placement import (
    PeriodicPlacementPlanner,
    PlacementFrontierKey,
    material_count_bucket,
)
from ._internal.model_placement_controller import ModelPlacementController
from ._internal.runtime_search import request_shape_bucket
from ._internal.runtime_plan import (
    AutomatedCandidateSet,
    HelperOpportunity,
    RuntimeHelperExecutionEnvelope,
)
from ._internal.runtime_resources import RuntimeMemoryLedger
from ._internal.runtime_residency_cohorts import RuntimeResidencyCohortTracker
from ._internal.runtime_decode_cohort import RuntimeDecodeCohortManager
from ._internal.online_placement import OnlinePlacementTracker
from ._internal.runtime_controller import RuntimeController, RuntimeRequestTicket
from ._internal.runtime_execution import RuntimeLeaseRenewalCoordinator
from ._internal.runtime_phase import RuntimePhaseLeaseController
from ._internal.decision_log import RuntimeDecisionLog
from ._internal.types import canonical_sha256
from ._internal.adaptive_decode_contracts import AdaptiveDecodeConfig
from ._internal.adaptive_decode import AdaptiveDecodeController
from ._internal.dynamic_residency import DynamicResidencySnapshot
from ._internal.phone_residency import PhoneResidencyPlan, PhoneResidencySnapshot
from ._internal.offline_phone_residency import OfflinePhoneResidencyPlan
from ._internal.phone_shards import PhoneFfnShardStorageMetadata
from ._internal.desktop_parent import (
    DesktopParentCapacityError,
    DesktopParentCapacitySelection,
    select_desktop_parent_for_live_vram,
)
from ._internal.runtime_placement import RuntimePlacementSnapshot
from ._unified.common import (
    _RECOVERABLE_ERRORS,
    _RequestHelperPreparation,
    _LateRequestHelperContext,
    _runtime_serialized,
    _text,
    DynamicPlacementLease,
    PhoneOffloadSchedule,
    PhoneArbiterSchedule,
    DynamicResidencySchedule,
    GpuBackfillSchedule,
    GpuWavefrontSchedule,
)
from ._unified.placement_epochs import PlacementEpochMixin
from ._unified.phone_residency import PhoneResidencyMixin
from ._unified.automated_candidates import AutomatedCandidateMixin
from ._unified.automated_selection import AutomatedSelectionMixin
from ._unified.automated_requests import AutomatedRequestMixin
from ._unified.helper_envelopes import HelperEnvelopeMixin
from ._unified.helper_preparation import HelperPreparationMixin
from ._unified.adaptive_decode_control import AdaptiveDecodeControlMixin
from ._unified.runtime_requests import RuntimeRequestMixin
from ._unified.legacy_schedules import LegacyScheduleMixin

__all__ = [
    'AdaptiveDecodeConfig',
    'AdaptiveDecodeControlMixin',
    'AdaptiveDecodeController',
    'AutomatedCandidateMixin',
    'AutomatedCandidateSet',
    'AutomatedRequestMixin',
    'AutomatedRouteCompiler',
    'AutomatedSelectionMixin',
    'DesktopParentCapacityError',
    'DesktopParentCapacitySelection',
    'DynamicPlacementLease',
    'DynamicResidencySchedule',
    'DynamicResidencySnapshot',
    'GpuBackfillSchedule',
    'GpuWavefrontSchedule',
    'HelperEnvelopeMixin',
    'HelperOpportunity',
    'HelperPreparationMixin',
    'HeterogeneousRuntimeSnapshot',
    'LegacyScheduleMixin',
    'LifecycleProfileSet',
    'LifecycleReceipt',
    'MatmulPlanner',
    'MatmulSystemProfile',
    'ModelManifest',
    'ModelManifestError',
    'ModelPlacementController',
    'OfflinePhoneResidencyPlan',
    'OnlinePlacementTracker',
    'POLICY_MODES',
    'PeriodicPlacementPlanner',
    'PhoneArbiterSchedule',
    'PhoneFfnShardStorageMetadata',
    'PhoneOffloadSchedule',
    'PhoneResidencyMixin',
    'PhoneResidencyPlan',
    'PhoneResidencySnapshot',
    'PlacementEpochMixin',
    'PlacementFrontierKey',
    'ProfileBundle',
    'Request',
    'RequestShapeUnsupportedError',
    'ResourceProfile',
    'ResourceTimeline',
    'RouteGenerationError',
    'RoutePolicy',
    'RuntimeCapabilityCatalog',
    'RuntimeController',
    'RuntimeDecisionLog',
    'RuntimeDecodeCohortManager',
    'RuntimeExecutorRegistry',
    'RuntimeHelperExecutionEnvelope',
    'RuntimeLeaseRenewalCoordinator',
    'RuntimeMemoryLedger',
    'RuntimePhaseLeaseController',
    'RuntimePlacementSnapshot',
    'RuntimeRequestMixin',
    'RuntimeRequestTicket',
    'RuntimeResidencyCohortTracker',
    'RuntimeRouteTemplateSet',
    'RuntimeSnapshot',
    'SchedulerError',
    'UnifiedScheduleError',
    'UnifiedScheduler',
    '_LateRequestHelperContext',
    '_RECOVERABLE_ERRORS',
    '_RequestHelperPreparation',
    '_runtime_serialized',
    '_text',
    'canonical_sha256',
    'load_cached_gguf_manifest',
    'material_count_bucket',
    'request_shape_bucket',
    'select_desktop_parent_for_live_vram',
]


class UnifiedScheduler(
    PlacementEpochMixin,
    PhoneResidencyMixin,
    AutomatedCandidateMixin,
    AutomatedSelectionMixin,
    AutomatedRequestMixin,
    HelperEnvelopeMixin,
    HelperPreparationMixin,
    AdaptiveDecodeControlMixin,
    RuntimeRequestMixin,
    LegacyScheduleMixin,
):
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
        runtime_discovery: bool = False,
        adaptive_decode_config: AdaptiveDecodeConfig | None = None,
        adaptive_envelope_minimum_remaining_tokens: int = 24,
        maximum_phone_sessions: int | None = None,
        protected_work_policy: str = "strict",
        learning_demand_decision_window: int = 6,
    ) -> None:
        if mode not in POLICY_MODES:
            raise UnifiedScheduleError("unknown unified policy mode")
        if protected_work_policy not in {"strict", "energy-budgeted"}:
            raise UnifiedScheduleError("protected work policy is invalid")
        self._protected_work_policy = protected_work_policy
        if (
            not profiles
            and not lifecycle_profiles
            and matmul_profile is None
            and not runtime_discovery
        ):
            raise UnifiedScheduleError("unified scheduler has no profile")
        if any(not isinstance(profile, ProfileBundle) for profile in profiles):
            raise UnifiedScheduleError("route profile is invalid")
        if any(
            not isinstance(profile_set, LifecycleProfileSet)
            for profile_set in lifecycle_profiles
        ):
            raise UnifiedScheduleError("lifecycle profile set is invalid")
        if adaptive_decode_config is not None and not isinstance(
            adaptive_decode_config, AdaptiveDecodeConfig
        ):
            raise UnifiedScheduleError("adaptive decode config is invalid")
        if (
            type(adaptive_envelope_minimum_remaining_tokens) is not int
            or adaptive_envelope_minimum_remaining_tokens <= 0
        ):
            raise UnifiedScheduleError(
                "adaptive envelope token threshold is invalid"
            )
        if maximum_phone_sessions is not None and (
            type(maximum_phone_sessions) is not int
            or maximum_phone_sessions <= 0
        ):
            raise UnifiedScheduleError(
                "maximum phone session count is invalid"
            )
        if (
            type(learning_demand_decision_window) is not int
            or learning_demand_decision_window <= 0
        ):
            raise UnifiedScheduleError(
                "learning demand decision window is invalid"
            )

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
        self._runtime_decision_log = RuntimeDecisionLog()
        self._runtime_lock = threading.RLock()
        self._runtime_controller = RuntimeController()
        self._runtime_residency_cohorts = RuntimeResidencyCohortTracker()
        self._model_placement_controller = ModelPlacementController()
        self._request_helper_preparations: dict[
            str, _RequestHelperPreparation
        ] = {}
        self._request_helper_opportunities: dict[
            str, tuple[HelperOpportunity, ...]
        ] = {}
        self._request_helper_preparation_envelopes: dict[
            tuple[str, str, int], RuntimeHelperExecutionEnvelope
        ] = {}
        self._late_request_helper_contexts: dict[
            str, _LateRequestHelperContext
        ] = {}
        self._request_helper_envelope_history: dict[
            str, dict[str, RuntimeHelperExecutionEnvelope]
        ] = {}
        self._phone_helper_endpoint_templates: dict[
            tuple[object, ...], RuntimeHelperExecutionEnvelope
        ] = {}
        self._online_learning_phone_demand_cache: dict[str, object] = {}
        self._learning_demand_decision_window = learning_demand_decision_window
        self._phone_route_use_by_artifact: dict[str, tuple[bool, ...]] = {}
        self._phone_ffn_shard_storage: tuple[
            PhoneFfnShardStorageMetadata, ...
        ] = ()
        self._offline_phone_residency_plans: dict[
            str, OfflinePhoneResidencyPlan
        ] = {}
        self._active_offline_phone_residency_plan_id: str | None = None
        self._fixed_phone_residency = None
        self._phone_reprovisioning = None
        self._phone_reprovision_boundary_gate = None
        self._phone_htp_memory_caps: dict[str, tuple[int, int]] = {}
        self._phone_telemetry_deferrals: dict[str, int] = {}
        self._runtime_decode_cohorts = RuntimeDecodeCohortManager()
        self._adaptive_decode = AdaptiveDecodeController()
        self._adaptive_observation_sources: dict[
            str, tuple[RuntimeCapabilityCatalog, frozenset[str]]
        ] = {}
        self._adaptive_observation_source_compilers: dict[
            str, AutomatedRouteCompiler
        ] = {}
        self._automated_observation_sources: dict[
            str, RuntimeCapabilityCatalog
        ] = {}
        self._legacy_evidence_migration_cache: set[
            tuple[str, str, str]
        ] = set()
        self._adaptive_decode_config = (
            AdaptiveDecodeConfig()
            if adaptive_decode_config is None else adaptive_decode_config
        )
        self._adaptive_envelope_minimum_remaining_tokens = (
            adaptive_envelope_minimum_remaining_tokens
        )
        self._maximum_phone_sessions = maximum_phone_sessions
        self._runtime_memory = RuntimeMemoryLedger()
        self._runtime_capabilities: RuntimeCapabilityCatalog | None = None
        self._automated_route_compiler: AutomatedRouteCompiler | None = None
        self._background_route_compiler: AutomatedRouteCompiler | None = None
        self._runtime_epoch_route_compiler: AutomatedRouteCompiler | None = None
        self._thermal_gate_log = ThermalGateLog()
        self._runtime_epoch_refresh_executor: ThreadPoolExecutor | None = None
        self._runtime_epoch_refresh_requested: dict[
            str,
            tuple[
                str,
                int,
                Callable[
                    [RuntimeRequestTicket, int],
                    HeterogeneousRuntimeSnapshot,
                ],
            ],
        ] = {}
        self._runtime_epoch_refresh_inflight: set[str] = set()
        self._runtime_placement_learning_generation_by_artifact: dict[
            str, str
        ] = {}
        self._runtime_placement_learning_signature_by_key: dict[
            tuple[str, int, int, str, str], str
        ] = {}
        self._background_placement_planner: (
            PeriodicPlacementPlanner | None
        ) = None
        self._background_frontier_keys: dict[
            tuple[str, int, int, str], PlacementFrontierKey
        ] = {}
        self._background_snapshot_objects: dict[
            tuple[str, int, int, str], HeterogeneousRuntimeSnapshot
        ] = {}
        self._background_observation_states: dict[
            tuple[str, int, int, str], tuple[object, ...]
        ] = {}
        self._background_resource_states: dict[
            tuple[str, int, int, str], tuple[object, ...]
        ] = {}
        self._runtime_capability_generation_sha256: str | None = None
        self._runtime_profile_generation_sha256: str | None = None
        self._runtime_transport_generation_sha256: str | None = None
        self._runtime_route_template_sets: dict[
            str, RuntimeRouteTemplateSet
        ] = {}
        self._runtime_route_template_cache_maximum = 128
        self._runtime_route_template_cache_hits = 0
        self._runtime_route_template_cache_misses = 0
        self._runtime_route_template_cache_evictions = 0
        self._runtime_epoch_invalidations: dict[str, int] = {}
        self._runtime_epoch_background_refreshes = 0
        self._runtime_epoch_background_refresh_failures = 0
        self._runtime_epoch_background_refresh_failure_reasons: dict[
            str, int
        ] = {}
        self._runtime_epoch_background_refresh_candidates = 0
        self._runtime_epoch_background_refresh_us = 0
        self._runtime_manifests: dict[str, ModelManifest] = {}
        self._runtime_manifest_generation_sha256: dict[str, str] = {}
        self._runtime_executor_registry: RuntimeExecutorRegistry | None = None
        self._runtime_decision_timings: list[Mapping[str, object]] = []
        self._last_runtime_commit_timing: Mapping[str, int] = (
            MappingProxyType({})
        )
        self._runtime_phase_controllers: dict[
            str, RuntimePhaseLeaseController
        ] = {}
        self._runtime_renewals: dict[
            str, RuntimeLeaseRenewalCoordinator
        ] = {}
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

    @classmethod
    def for_runtime_discovery(
        cls,
        mode: str,
        *,
        adaptive_decode_config: AdaptiveDecodeConfig | None = None,
        adaptive_envelope_minimum_remaining_tokens: int = 24,
        maximum_phone_sessions: int | None = None,
        protected_work_policy: str = "strict",
        learning_demand_decision_window: int = 6,
    ) -> "UnifiedScheduler":
        """Create an empty scheduler that accepts discovered runtime inputs."""
        return cls(
            (),
            mode,
            runtime_discovery=True,
            adaptive_decode_config=adaptive_decode_config,
            adaptive_envelope_minimum_remaining_tokens=(
                adaptive_envelope_minimum_remaining_tokens
            ),
            maximum_phone_sessions=maximum_phone_sessions,
            protected_work_policy=protected_work_policy,
            learning_demand_decision_window=learning_demand_decision_window,
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

    @contextmanager
    def _transaction(
        self,
        errors: type[BaseException] | tuple[type[BaseException], ...] = (
            _RECOVERABLE_ERRORS
        ),
        *,
        convert: bool = True,
        rollback_if: Callable[[BaseException], bool] | None = None,
    ) -> Iterator[None]:
        """Roll every runtime state owner back if the block raises ``errors``.

        With ``convert`` the caught error is re-raised as ``UnifiedScheduleError``
        unless it already is one; without it the original error propagates.
        """
        checkpoint = self._runtime_transaction_checkpoint()
        try:
            yield
        except errors as exc:
            if rollback_if is None or rollback_if(exc):
                self._restore_runtime_transaction(checkpoint)
            if not convert or isinstance(exc, UnifiedScheduleError):
                raise
            raise UnifiedScheduleError(str(exc)) from exc

    def _runtime_transaction_checkpoint(self) -> tuple[object, ...]:
        return (
            self.timeline.checkpoint(),
            self._runtime_memory.checkpoint(),
            self._runtime_controller.checkpoint(),
            self._runtime_decision_log.checkpoint(),
            self._adaptive_decode.checkpoint(),
            self._runtime_residency_cohorts.checkpoint(),
            self._model_placement_controller.checkpoint(),
            self._runtime_decode_cohorts.checkpoint(),
            self._runtime_executor_registry,
            frozenset(self._legacy_evidence_migration_cache),
            dict(self._runtime_route_template_sets),
            self._runtime_route_template_cache_evictions,
            dict(
                self._runtime_placement_learning_generation_by_artifact
            ),
            dict(self._runtime_placement_learning_signature_by_key),
            dict(self._request_helper_preparations),
            dict(self._request_helper_opportunities),
            dict(self._request_helper_preparation_envelopes),
            dict(self._late_request_helper_contexts),
            {
                request_id: dict(envelopes)
                for request_id, envelopes in (
                    self._request_helper_envelope_history.items()
                )
            },
            dict(self._phone_helper_endpoint_templates),
            dict(self._online_learning_phone_demand_cache),
            dict(self._phone_route_use_by_artifact),
            dict(self._offline_phone_residency_plans),
            self._active_offline_phone_residency_plan_id,
            dict(self._phone_htp_memory_caps),
            None
            if self._automated_route_compiler is None
            else self._automated_route_compiler.observation_checkpoint(),
        )

    def _restore_runtime_transaction(
        self, checkpoint: tuple[object, ...]
    ) -> None:
        (
            timeline,
            memory,
            controller,
            decision_log,
            adaptive_decode,
            residency_cohorts,
            model_placement_controller,
            decode_cohorts,
            registry,
            legacy_evidence_migration_cache,
            route_template_sets,
            route_template_cache_evictions,
            placement_learning_generations,
            placement_learning_signatures,
            request_helper_preparations,
            request_helper_opportunities,
            request_helper_preparation_envelopes,
            late_request_helper_contexts,
            request_helper_envelope_history,
            phone_helper_endpoint_templates,
            online_learning_phone_demand_cache,
            phone_route_use_by_artifact,
            offline_phone_residency_plans,
            active_offline_phone_residency_plan_id,
            phone_htp_memory_caps,
            observations,
        ) = checkpoint
        self.timeline.restore(timeline)
        self._runtime_memory.restore(memory)
        self._runtime_controller.restore(controller)
        self._runtime_decision_log.restore(decision_log)
        self._adaptive_decode.restore(adaptive_decode)
        self._runtime_residency_cohorts.restore(residency_cohorts)
        self._model_placement_controller.restore(
            model_placement_controller
        )
        phone_layout = (
            self._model_placement_controller.planning_phone_layout()
        )
        for compiler in (
            self._automated_route_compiler,
            self._runtime_epoch_route_compiler,
        ):
            if compiler is not None:
                compiler.set_phone_residency_layout(
                    None if phone_layout is None else phone_layout.layout
                )
        self._runtime_decode_cohorts.restore(decode_cohorts)
        self._runtime_executor_registry = registry
        self._legacy_evidence_migration_cache = set(
            legacy_evidence_migration_cache
        )
        self._runtime_route_template_sets = dict(route_template_sets)
        self._runtime_route_template_cache_evictions = (
            route_template_cache_evictions
        )
        self._runtime_placement_learning_generation_by_artifact = dict(
            placement_learning_generations
        )
        self._runtime_placement_learning_signature_by_key = dict(
            placement_learning_signatures
        )
        self._request_helper_preparations = dict(
            request_helper_preparations
        )
        self._request_helper_opportunities = dict(
            request_helper_opportunities
        )
        self._request_helper_preparation_envelopes = dict(
            request_helper_preparation_envelopes
        )
        self._late_request_helper_contexts = dict(
            late_request_helper_contexts
        )
        self._request_helper_envelope_history = {
            request_id: dict(envelopes)
            for request_id, envelopes in (
                request_helper_envelope_history.items()
            )
        }
        self._phone_helper_endpoint_templates = dict(
            phone_helper_endpoint_templates
        )
        self._online_learning_phone_demand_cache = dict(
            online_learning_phone_demand_cache
        )
        self._phone_route_use_by_artifact = dict(phone_route_use_by_artifact)
        self._offline_phone_residency_plans = dict(
            offline_phone_residency_plans
        )
        self._active_offline_phone_residency_plan_id = (
            active_offline_phone_residency_plan_id
        )
        self._phone_htp_memory_caps = dict(phone_htp_memory_caps)
        if (
            observations is not None
            and self._automated_route_compiler is not None
        ):
            self._automated_route_compiler.restore_observations(observations)

    @_runtime_serialized
    def register_runtime_capabilities(
        self, catalog: RuntimeCapabilityCatalog
    ) -> None:
        """Register discovered devices, links, kernels, and transitions."""
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise UnifiedScheduleError("runtime capability catalog is invalid")
        current = self._runtime_capabilities
        active = any(
            row["dispatch_state"] not in {
                "CANCELLED", "COMPLETED", "FAILED"
            }
            for row in self._runtime_controller.snapshot()["tickets"].values()
        )
        if active and current is not None and not self._catalog_update_is_additive(
            current, catalog
        ):
            raise UnifiedScheduleError(
                "runtime capability replacement conflicts with active tickets"
            )
        try:
            self.timeline.register_resources(catalog.resources)
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self._resource_ids = frozenset(
            set(self._resource_ids) | set(catalog.resources)
        )
        self._runtime_capabilities = catalog
        self._runtime_capability_generation_sha256 = canonical_sha256(
            catalog
        )
        self._runtime_profile_generation_sha256 = canonical_sha256(
            catalog.placement_profile
        )
        self._runtime_transport_generation_sha256 = canonical_sha256({
            "links": tuple(
                (
                    link.link_id,
                    link.transport_generation,
                    link.qualification_identity_sha256,
                )
                for link in catalog.placement_profile.links
            ),
            "schema": "runtime-transport-generation-v1",
        })
        if self._background_placement_planner is not None:
            self._background_placement_planner.close()
        observation_store = (
            None
            if self._automated_route_compiler is None
            else self._automated_route_compiler.observation_store
        )
        self._automated_route_compiler = AutomatedRouteCompiler(
            catalog, self.timeline, observation_store
        )
        self._background_route_compiler = self._automated_route_compiler
        self._runtime_epoch_route_compiler = AutomatedRouteCompiler(
            catalog, self.timeline
        )
        self._automated_route_compiler.thermal_gate_log = self._thermal_gate_log
        self._runtime_epoch_route_compiler.thermal_gate_log = self._thermal_gate_log
        if self._phone_ffn_shard_storage:
            self._automated_route_compiler.set_phone_ffn_shard_storage(
                self._phone_ffn_shard_storage
            )
            self._runtime_epoch_route_compiler.set_phone_ffn_shard_storage(
                self._phone_ffn_shard_storage
            )
        phone_layout = (
            self._model_placement_controller.planning_phone_layout()
        )
        self._automated_route_compiler.set_phone_residency_layout(
            None if phone_layout is None else phone_layout.layout
        )
        self._runtime_epoch_route_compiler.set_phone_residency_layout(
            None if phone_layout is None else phone_layout.layout
        )
        self._runtime_epoch_route_compiler.import_observations(
            dict(self._automated_route_compiler.observation_export())
        )
        background_compiler = self._background_route_compiler

        def compile_frontier(
            manifest,
            input_tokens,
            output_tokens,
            quality_requirement,
            snapshot,
            observed_at_us,
            trigger_reasons,
        ):
            return background_compiler.prepare_frontier(
                manifest,
                input_tokens,
                output_tokens,
                quality_requirement,
                snapshot,
                observed_at_us,
                force_refresh=bool({
                    "cost_or_link_profile_changed",
                    "device_capability_changed",
                }.intersection(trigger_reasons)),
            )

        self._background_placement_planner = PeriodicPlacementPlanner(
            compile_frontier
        )
        self._background_frontier_keys.clear()
        self._background_snapshot_objects.clear()
        self._background_observation_states.clear()
        self._background_resource_states.clear()
        try:
            for manifest in self._runtime_manifests.values():
                self._automated_route_compiler.prepare_manifest(manifest)
                self._runtime_epoch_route_compiler.prepare_manifest(
                    manifest
                )
                if (
                    self._background_route_compiler
                        is not self._automated_route_compiler
                ):
                    self._background_route_compiler.prepare_manifest(
                        manifest
                    )
        except RouteGenerationError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        for manifest in self._runtime_manifests.values():
            self._model_placement_controller.notify(
                manifest.artifact_sha256,
                "CAPABILITY_REGISTRATION_CHANGED",
                0,
            )

    @_runtime_serialized
    def register_phone_ffn_shard_storage(
        self,
        records: Sequence[PhoneFfnShardStorageMetadata],
    ) -> None:
        """Register verified per-artifact storage coverage for HTP sessions."""

        rows = tuple(records)
        if any(
            not isinstance(row, PhoneFfnShardStorageMetadata)
            for row in rows
        ):
            raise UnifiedScheduleError(
                "phone FFN shard storage metadata is invalid"
            )
        keys = tuple(
            (row.parent_artifact_sha256, row.session_id) for row in rows
        )
        if len(keys) != len(set(keys)):
            raise UnifiedScheduleError(
                "phone FFN shard storage sessions are duplicated"
            )
        artifacts = {
            manifest.artifact_sha256
            for manifest in self._runtime_manifests.values()
        }
        if any(row.parent_artifact_sha256 not in artifacts for row in rows):
            raise UnifiedScheduleError(
                "phone FFN shard storage parent is not registered"
            )
        session_ids = {
            session.session_id
            for executor in (
                () if self._runtime_capabilities is None else
                self._runtime_capabilities.executors
            )
            for session in executor.phone_sessions
        }
        by_artifact: dict[str, set[str]] = {}
        for row in rows:
            by_artifact.setdefault(row.parent_artifact_sha256, set()).add(
                row.session_id
            )
        if any(
            not stored_ids <= session_ids
            for stored_ids in by_artifact.values()
        ):
            raise UnifiedScheduleError(
                "phone FFN shard storage references undiscovered sessions"
            )
        ordered = tuple(sorted(
            rows,
            key=lambda row: (
                row.parent_artifact_sha256, row.session_id
            ),
        ))
        if ordered == self._phone_ffn_shard_storage:
            return
        self._phone_ffn_shard_storage = ordered
        compilers = {
            id(compiler): compiler
            for compiler in (
                self._automated_route_compiler,
                self._background_route_compiler,
                self._runtime_epoch_route_compiler,
            )
            if compiler is not None
        }
        try:
            for compiler in compilers.values():
                compiler.set_phone_ffn_shard_storage(ordered)
        except RouteGenerationError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self._runtime_route_template_sets.clear()
        self._online_learning_phone_demand_cache.clear()
        for artifact in sorted(by_artifact):
            self._model_placement_controller.notify(
                artifact, "PHONE_FFN_SHARD_STORAGE_CHANGED", 0
            )

    @staticmethod
    def _catalog_update_is_additive(
        current: RuntimeCapabilityCatalog,
        updated: RuntimeCapabilityCatalog,
    ) -> bool:
        if (
            current.minimum_energy_saving_ppm
                != updated.minimum_energy_saving_ppm
            or current.maximum_latency_ppm != updated.maximum_latency_ppm
            or current.fallback.executor_id != updated.fallback.executor_id
        ):
            return False

        def mapping_is_subset(left, right) -> bool:
            return all(right.get(key) == value for key, value in left.items())

        if not mapping_is_subset(current.resources, updated.resources):
            return False
        old_executors = {
            row.executor_id: row for row in current.executors
        }
        new_executors = {
            row.executor_id: row for row in updated.executors
        }
        if not mapping_is_subset(old_executors, new_executors):
            return False
        old_composite_executors = {
            row.executor_id: row for row in current.composite_executors
        }
        new_composite_executors = {
            row.executor_id: row for row in updated.composite_executors
        }
        if not mapping_is_subset(
            old_composite_executors, new_composite_executors
        ):
            return False
        old_transitions = {
            row.transition_id: row for row in current.transitions
        }
        new_transitions = {
            row.transition_id: row for row in updated.transitions
        }
        if not mapping_is_subset(old_transitions, new_transitions):
            return False
        old_route_profiles = {
            row.selector_id: row for row in current.route_shape_profiles
        }
        new_route_profiles = {
            row.selector_id: row for row in updated.route_shape_profiles
        }
        if not mapping_is_subset(old_route_profiles, new_route_profiles):
            return False
        old_system_profiles = {
            row.selector_id: row for row in current.system_cost_profiles
        }
        new_system_profiles = {
            row.selector_id: row for row in updated.system_cost_profiles
        }
        if not mapping_is_subset(old_system_profiles, new_system_profiles):
            return False
        old_profile = current.placement_profile
        new_profile = updated.placement_profile
        if (
            old_profile.energy_boundary_id != new_profile.energy_boundary_id
            or not old_profile.idle_charge_domains.issubset(
                new_profile.idle_charge_domains
            )
            or not mapping_is_subset(
                old_profile.memory_pools, new_profile.memory_pools
            )
            or not mapping_is_subset(old_profile.devices, new_profile.devices)
            or not mapping_is_subset(old_profile.domains, new_profile.domains)
            or not mapping_is_subset(old_profile.kernels, new_profile.kernels)
        ):
            return False
        old_links = {row.link_id: row for row in old_profile.links}
        new_links = {row.link_id: row for row in new_profile.links}
        return mapping_is_subset(old_links, new_links)

    @_runtime_serialized
    def register_gguf_model(
        self,
        model_id: str,
        path: str | Path,
        *,
        cache_path: Path | None = None,
    ) -> ModelManifest:
        """Register an arbitrary GGUF artifact through its physical manifest."""
        try:
            manifest = load_cached_gguf_manifest(
                model_id, path, cache_path
            )
        except ModelManifestError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        current = self._runtime_manifests.get(manifest.model_id)
        if current is not None and current != manifest:
            raise UnifiedScheduleError(
                "registered model id has a different artifact"
            )
        if self._automated_route_compiler is not None:
            try:
                self._automated_route_compiler.prepare_manifest(manifest)
                if self._runtime_epoch_route_compiler is not None:
                    self._runtime_epoch_route_compiler.prepare_manifest(
                        manifest
                    )
                if (
                    self._background_route_compiler is not None
                    and self._background_route_compiler
                        is not self._automated_route_compiler
                ):
                    self._background_route_compiler.prepare_manifest(
                        manifest
                    )
            except RouteGenerationError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
        self._runtime_manifests[manifest.model_id] = manifest
        self._runtime_manifest_generation_sha256[manifest.model_id] = (
            canonical_sha256(manifest)
        )
        self._model_placement_controller.notify(
            manifest.artifact_sha256, "GGUF_MODEL_REGISTERED", 0
        )
        return manifest

    @_runtime_serialized
    def register_model_manifest(self, manifest: ModelManifest) -> ModelManifest:
        """Register a previously validated, hash-bound GGUF manifest."""
        if not isinstance(manifest, ModelManifest):
            raise UnifiedScheduleError("runtime model manifest is invalid")
        current = self._runtime_manifests.get(manifest.model_id)
        if current is not None and current != manifest:
            raise UnifiedScheduleError(
                "registered model id has a different artifact"
            )
        if self._automated_route_compiler is not None:
            try:
                self._automated_route_compiler.prepare_manifest(manifest)
                if self._runtime_epoch_route_compiler is not None:
                    self._runtime_epoch_route_compiler.prepare_manifest(
                        manifest
                    )
                if (
                    self._background_route_compiler is not None
                    and self._background_route_compiler
                        is not self._automated_route_compiler
                ):
                    self._background_route_compiler.prepare_manifest(
                        manifest
                    )
            except RouteGenerationError as exc:
                raise UnifiedScheduleError(str(exc)) from exc
        self._runtime_manifests[manifest.model_id] = manifest
        self._runtime_manifest_generation_sha256[manifest.model_id] = (
            canonical_sha256(manifest)
        )
        self._model_placement_controller.notify(
            manifest.artifact_sha256, "GGUF_MODEL_REGISTERED", 0
        )
        return manifest

    def runtime_model_manifest(self, model_id: str) -> ModelManifest:
        model_id = _text("runtime model id", model_id)
        manifest = self._runtime_manifests.get(model_id)
        if manifest is None:
            raise UnifiedScheduleError("runtime model is not registered")
        return manifest

    @_runtime_serialized
    def select_live_vram_desktop_parent(
        self,
        model_id: str,
        memory: RuntimePlacementSnapshot,
        *,
        cuda_graph_mode: str | None = None,
        preserve_placement: bool = False,
        maximum_gpu_layers: int | None = None,
        launch_overrides: Mapping[str, int | str] | None = None,
    ) -> DesktopParentCapacitySelection:
        """Choose the largest qualified-parent GPU suffix fitting live VRAM."""
        if not isinstance(memory, RuntimePlacementSnapshot):
            raise UnifiedScheduleError(
                "desktop parent memory snapshot is invalid"
            )
        manifest = self.runtime_model_manifest(model_id)
        catalog = self._runtime_capabilities
        if catalog is None:
            raise UnifiedScheduleError(
                "runtime capabilities are not registered"
            )
        control = catalog.desktop_control_by_artifact.get(
            manifest.artifact_sha256
        )
        if control is None:
            raise UnifiedScheduleError(
                "desktop parent control is not registered"
            )
        source = catalog.composite_executor_by_id.get(control.executor_id)
        if source is None:
            raise UnifiedScheduleError(
                "desktop parent executor is not registered"
            )
        gpu_device_id = source.adapter_parameters.get("gpu_device_id")
        if type(gpu_device_id) is not str:
            raise UnifiedScheduleError(
                "desktop parent GPU device is not registered"
            )
        device = catalog.placement_profile.devices.get(gpu_device_id)
        if device is None or device.kind != "gpu":
            raise UnifiedScheduleError(
                "desktop parent GPU capability is invalid"
            )
        capacity = memory.capacities.get(device.memory_pool_id)
        if capacity is None:
            raise UnifiedScheduleError(
                "desktop parent GPU memory snapshot is absent"
            )
        if launch_overrides is not None and set(launch_overrides) - {
            "context_size", "parallel", "batch_size", "ubatch_size", "desktop_launch_mode",
        }:
            raise UnifiedScheduleError("desktop parent launch override is invalid")
        if (cuda_graph_mode is not None or preserve_placement
                or maximum_gpu_layers is not None or launch_overrides):
            if cuda_graph_mode is not None and cuda_graph_mode not in (
                "default", "disabled"
            ):
                raise UnifiedScheduleError("desktop parent CUDA graph mode is invalid")
            # Selection proposes a new calibration, not inherited qualification.
            parameters = {
                key: value for key, value in source.adapter_parameters.items()
                if not key.startswith("capacity_parent_")
            }
            if cuda_graph_mode is not None:
                parameters["cuda_graph_mode"] = cuda_graph_mode
            parameters.update(launch_overrides or {})
            source = replace(source, adapter_parameters=parameters)
        try:
            selection = select_desktop_parent_for_live_vram(
                manifest,
                source,
                capacity,
                memory_snapshot_id=memory.snapshot_id,
                maximum_gpu_layers=maximum_gpu_layers,
            )
            if preserve_placement and selection.selected.gpu_layers != (
                source.adapter_parameters["gpu_layers"]
            ):
                raise UnifiedScheduleError("exact desktop parent does not fit live VRAM")
            return selection
        except DesktopParentCapacityError as error:
            raise UnifiedScheduleError(str(error)) from error

    def _automated_compiler(self) -> AutomatedRouteCompiler:
        if self._runtime_capabilities is None:
            raise UnifiedScheduleError("runtime capabilities are not registered")
        if self._automated_route_compiler is None:
            self._automated_route_compiler = AutomatedRouteCompiler(
                self._runtime_capabilities, self.timeline
            )
            self._automated_route_compiler.thermal_gate_log = (
                self._thermal_gate_log
            )
            if self._phone_ffn_shard_storage:
                self._automated_route_compiler.set_phone_ffn_shard_storage(
                    self._phone_ffn_shard_storage
                )
        return self._automated_route_compiler

    def thermal_deferral_events(self) -> tuple[Mapping[str, object], ...]:
        """THERMAL_DEFERRAL / THERMAL_DEFERRAL_CLEARED rows: one per onset
        (and end) of a device's thermal gate in route feasibility."""
        return self._thermal_gate_log.events()

    def automated_cost_cache_stats(self) -> Mapping[str, int]:
        return self._automated_compiler().cache_stats()

    def background_placement_stats(self) -> Mapping[str, int]:
        planner = self._background_placement_planner
        if planner is None:
            return MappingProxyType({})
        return planner.stats()

    def _runtime_learning_generation_sha256(self) -> str:
        raw = self._automated_compiler().observation_state()
        material = {
            name: (
                material_count_bucket(value)
                if name in {
                    "complete_receipts",
                    "diagnostic_energy_receipts",
                    "incomplete_receipts",
                    "unattributed_transfer_receipts",
                }
                else value
            )
            for name, value in raw.items()
        }
        return canonical_sha256({
            "observations": dict(sorted(material.items())),
            "schema": "runtime-learning-generation-v1",
        })

    def _runtime_placement_learning_generation_sha256(
        self, artifact_sha256: str
    ) -> str:
        return self._runtime_placement_learning_generation_by_artifact.get(
            artifact_sha256,
            canonical_sha256({
                "artifact_sha256": artifact_sha256,
                "schema": "runtime-placement-learning-generation-v1",
            }),
        )

    @staticmethod
    def _runtime_placement_learning_key(
        request: Request,
        manifest: ModelManifest,
        selection_mode: str,
    ) -> tuple[str, int, int, str, str]:
        input_bucket, output_bucket = request_shape_bucket(
            request.input_tokens, request.output_tokens
        )
        return (
            manifest.artifact_sha256,
            input_bucket,
            output_bucket,
            request.quality_requirement,
            selection_mode,
        )

    @staticmethod
    def _runtime_placement_learning_signature(
        candidate_set: AutomatedCandidateSet,
    ) -> str:
        evidence_rejections = {
            "ENERGY_UNKNOWN",
            "MARGINAL_SYSTEM_COST_UNKNOWN",
            "PREDICTION_QUARANTINED",
            "PROFILE_QUARANTINED",
            "ROUTE_NOT_QUALIFIED",
        }
        rows = []
        ranked = []
        for candidate in candidate_set.candidates:
            break_even = candidate.residency_break_even or {}
            warm_upper = candidate.cost.warm_execution_energy_upper_uj
            transition_upper = candidate.cost.transition_energy_upper_uj
            parent = next((
                row for row in candidate_set.candidates
                if row.candidate_id
                    == candidate.paired_baseline_route_id
            ), None)
            parent_warm_lower = (
                None
                if parent is None
                else parent.cost.warm_execution_energy_lower_uj
            )
            break_even_count = None
            if (
                warm_upper is not None
                and transition_upper is not None
                and parent_warm_lower is not None
                and parent_warm_lower > warm_upper
            ):
                transition_total = int(break_even.get(
                    "incremental_transition_energy_uj",
                    transition_upper,
                )) + int(break_even.get("restore_energy_uj", 0))
                saving = parent_warm_lower - warm_upper
                break_even_count = (
                    transition_total + saving - 1
                ) // saving
            reasons = tuple(sorted(
                set(candidate.rejection_reasons)
                & evidence_rejections
            ))
            rows.append({
                "break_even_use_count": break_even_count,
                "candidate_id": candidate.candidate_id,
                "energy_evidence": candidate.cost.energy_evidence,
                "latency_evidence": candidate.cost.latency_evidence,
                "maturity": candidate.maturity,
                "rejection_reasons": reasons,
            })
            if not reasons and warm_upper is not None:
                ranked.append((
                    warm_upper,
                    candidate.cost.service_upper_us,
                    candidate.candidate_id,
                ))
        return canonical_sha256({
            "pareto_order": [row[2] for row in sorted(ranked)],
            "rows": sorted(rows, key=lambda row: row["candidate_id"]),
            "schema": "runtime-placement-learning-signature-v1",
        })
