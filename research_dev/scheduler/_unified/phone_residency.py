"""Phone FFN residency portfolio, layout transitions, snapshot verification, and replay.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from ..config import (
    FixedPhoneResidencyConfiguration,
    PhoneResidentModelReprovisioningConfiguration,
)
from .._internal.policy import Request
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_cost import RuntimeExecutorBinding
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeExecutorState,
    RuntimePhoneSessionCapability,
)
from .._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
    PhoneSessionMarginalGain,
)
from .._internal.phone_shards import PhoneFfnResidencyLayout
from .._internal.offline_phone_residency import (
    OfflinePhoneResidencyPlan,
    OfflinePhoneResidencyStage,
)
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeExecutionPlan,
    RuntimeTransitionReceipt,
)
from .._internal.runtime_resources import RuntimeResidencyProjectionToken
from .._internal.runtime_controller import RuntimeRequestTicket
from .common import _runtime_serialized

from .phone_residency_ops.common import (
    _OfflineLearningDemand as _OfflineLearningDemand,
    _PhoneCandidateChoice as _PhoneCandidateChoice,
    _PhoneDemandDiscovery as _PhoneDemandDiscovery,
    _PhoneLayoutDecision as _PhoneLayoutDecision,
    _PhoneMemoryBudget as _PhoneMemoryBudget,
    _PhoneQueueDemand as _PhoneQueueDemand,
)
from .phone_residency_ops import demand as _demand
from .phone_residency_ops import economics as _economics
from .phone_residency_ops import fixed as _fixed
from .phone_residency_ops import offline_execution as _offline_execution
from .phone_residency_ops import offline_inputs as _offline_inputs
from .phone_residency_ops import offline_planning as _offline_planning
from .phone_residency_ops import portfolio as _portfolio
from .phone_residency_ops import publication as _publication
from .phone_residency_ops import reprovision as _reprovision
from .phone_residency_ops import verification as _verification


class PhoneResidencyMixin:
    """Phone FFN residency portfolio, layout transitions, snapshot verification, and replay."""

    @_runtime_serialized
    def set_phone_htp_memory_cap(
        self, phone_device_id: str, cap_bytes: int | None, *,
        workspace_bytes: int, observed_at_us: int,
    ) -> None:
        """Set a residency budget; memory is not released until a verified resize."""
        return _economics.set_phone_htp_memory_cap(
            self, phone_device_id, cap_bytes,
            workspace_bytes=workspace_bytes, observed_at_us=observed_at_us,
        )

    @_runtime_serialized
    def configure_fixed_phone_residency(
        self, configuration: FixedPhoneResidencyConfiguration,
    ) -> None:
        """Freeze an evaluation assignment before any layout or request exists."""
        return _fixed.configure_fixed_phone_residency(self, configuration)

    @_runtime_serialized
    def configure_phone_resident_model_reprovisioning(
        self, configuration: PhoneResidentModelReprovisioningConfiguration | None,
    ) -> None:
        """Re-provision phone FFN sessions for the model(s) the desktop serves."""
        return _reprovision.configure_phone_resident_model_reprovisioning(self, configuration)

    def _phone_reprovision_demand(self, demand, discovery, snapshot, observed_at_us):
        return _reprovision._phone_reprovision_demand(
            self, demand, discovery, snapshot, observed_at_us,
        )

    def _reevaluate_phone_layout_for_desktop_load(self, ticket: RuntimeRequestTicket) -> None:
        return _reprovision._reevaluate_phone_layout_for_desktop_load(self, ticket)

    def _reevaluate_phone_layout_for_decided_transition(
        self, ticket: RuntimeRequestTicket, observed_at_us: int,
    ) -> None:
        return _reprovision._reevaluate_phone_layout_for_decided_transition(
            self, ticket, observed_at_us
        )

    def _reevaluate_phone_layout_after_release(
        self, ticket: RuntimeRequestTicket, observed_at_us: int,
    ) -> None:
        return _reprovision._reevaluate_phone_layout_after_release(self, ticket, observed_at_us)

    def _defer_preparation_until_release(self, request_id, state, blockers, observed_at_us):
        return _reprovision._defer_preparation_until_release(
            self, request_id, state, blockers, observed_at_us,
        )

    @_runtime_serialized
    def fixed_phone_residency_configuration(self) -> Mapping[str, object] | None:
        return _fixed.fixed_phone_residency_configuration(self)

    @_runtime_serialized
    def fixed_phone_residency_requests(self) -> Mapping[str, tuple[Request, ...]]:
        """Use preparation-only shapes, never future trace requests or arrivals."""
        return _fixed.fixed_phone_residency_requests(self)

    def _fixed_phone_inputs(self, discovery, sessions):
        return _fixed._fixed_phone_inputs(self, discovery, sessions)

    def _fixed_phone_layout_matches(self, layout, *, partial=False):
        return _fixed._fixed_phone_layout_matches(self, layout, partial=partial)

    def _phone_telemetry_deferral(self, snapshot, observed_at_us, request_id):
        return _fixed._phone_telemetry_deferral(self, snapshot, observed_at_us, request_id)

    @_runtime_serialized
    def set_phone_layout_event_clock(
        self, clock: Callable[[], int] | None
    ) -> None:
        """Record physical publication time separately from snapshot time."""
        try:
            self._model_placement_controller.set_phone_layout_event_clock(clock)
        except ModelPlacementControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @staticmethod
    def _offline_planning_request(
        request: Request,
        *,
        request_id: str,
        observed_at_us: int,
    ) -> Request:
        return _demand._offline_planning_request(
            request,
            request_id=request_id,
            observed_at_us=observed_at_us,
        )

    def _offline_phone_requests(
        self,
        requests_by_model: Mapping[str, Sequence[Request]],
    ) -> tuple[
        str,
        dict[str, tuple[str, Request]],
        dict[str, int],
    ]:
        return _demand._offline_phone_requests(self, requests_by_model)

    def _offline_phone_discovery(
        self,
        request_by_artifact: Mapping[str, tuple[str, Request]],
        queued_work_by_artifact: Mapping[str, int],
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> _PhoneDemandDiscovery:
        return _demand._offline_phone_discovery(
            self,
            request_by_artifact,
            queued_work_by_artifact,
            snapshot,
            observed_at_us,
        )

    def _learning_phone_demand(
        self,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest: ModelManifest,
        queued_work: int,
    ) -> _OfflineLearningDemand | None:
        return _demand._learning_phone_demand(self, candidate_set, request, manifest, queued_work)

    def _record_phone_route_use(self, ticket: RuntimeRequestTicket) -> None:
        return _demand._record_phone_route_use(self, ticket)

    def _learning_demand_active(self, artifact_sha256: str) -> bool:
        return _demand._learning_demand_active(self, artifact_sha256)

    def _online_learning_phone_discovery(
        self,
        request: Request,
        manifest: ModelManifest,
        demand: _PhoneQueueDemand,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        discovered: _PhoneDemandDiscovery,
    ) -> _PhoneDemandDiscovery:
        return _demand._online_learning_phone_discovery(
            self,
            request,
            manifest,
            demand,
            snapshot,
            observed_at_us,
            discovered,
        )

    def _cached_online_learning_phone_discovery(
        self,
        demand: _PhoneQueueDemand,
        discovered: _PhoneDemandDiscovery,
    ) -> _PhoneDemandDiscovery:
        return _demand._cached_online_learning_phone_discovery(self, demand, discovered)

    @staticmethod
    def _merge_learning_phone_discovery(
        discovered: _PhoneDemandDiscovery,
        learning_by_artifact: Mapping[str, _OfflineLearningDemand],
    ) -> _PhoneDemandDiscovery:
        return _demand._merge_learning_phone_discovery(discovered, learning_by_artifact)

    def _offline_phone_target(
        self,
        discovery: _PhoneDemandDiscovery,
        queued_work_by_artifact: Mapping[str, int],
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[
        PhoneFfnResidencyLayout,
        tuple[RuntimePhoneSessionCapability, ...],
        _PhoneMemoryBudget,
    ]:
        return _offline_inputs._offline_phone_target(
            self,
            discovery,
            queued_work_by_artifact,
            snapshot,
        )

    def _materialize_offline_phone_helper(
        self,
        request: Request,
        model_id: str,
        layout: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ):
        return _offline_inputs._materialize_offline_phone_helper(
            self,
            request,
            model_id,
            layout,
            snapshot,
            observed_at_us,
        )

    def _offline_phone_safety_snapshot(
        self,
        plan: OfflinePhoneResidencyPlan,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[HeterogeneousRuntimeSnapshot, RuntimeExecutorState]:
        """Keep legacy evidence readable; new load admission uses live telemetry."""
        return _offline_inputs._offline_phone_safety_snapshot(self, plan, snapshot)

    def _offline_phone_materialization_snapshot(
        self,
        plan: OfflinePhoneResidencyPlan,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[HeterogeneousRuntimeSnapshot, RuntimeExecutorState]:
        """Materialize phone loads from the admitted desktop parent view."""
        return _offline_inputs._offline_phone_materialization_snapshot(self, plan, snapshot)

    def _offline_phone_discovery_snapshot(
        self,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> HeterogeneousRuntimeSnapshot:
        """Expose replaceable HTP bytes only while discovering shard shapes."""
        return _offline_inputs._offline_phone_discovery_snapshot(self, snapshot)

    def _propose_offline_phone_stage(
        self,
        plan: OfflinePhoneResidencyPlan,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyPlan:
        return _offline_planning._propose_offline_phone_stage(self, plan, snapshot, observed_at_us)

    @_runtime_serialized
    def plan_offline_phone_residency(
        self,
        requests_by_model: Mapping[str, Sequence[Request]],
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyPlan:
        """Choose and queue a progressive resident superset before a trace."""
        return _offline_planning.plan_offline_phone_residency(
            self,
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
        )

    @_runtime_serialized
    def next_offline_phone_residency_stage(
        self,
        plan_id: str,
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyPlan:
        return _offline_planning.next_offline_phone_residency_stage(
            self,
            plan_id,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
        )

    @_runtime_serialized
    def offline_phone_residency_stage(
        self, plan_id: str
    ) -> OfflinePhoneResidencyStage | None:
        return _offline_planning.offline_phone_residency_stage(self, plan_id)

    @_runtime_serialized
    def begin_offline_phone_residency_stage(
        self,
        plan_id: str,
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> Mapping[str, object]:
        return _offline_execution.begin_offline_phone_residency_stage(
            self,
            plan_id,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
        )

    @_runtime_serialized
    def check_offline_phone_residency_stage(
        self,
        plan_id: str,
        *,
        observed_at_us: int,
        guard_us: int = 250_000,
        quantum_us: int = 2_000_000,
    ) -> None:
        return _offline_execution.check_offline_phone_residency_stage(
            self,
            plan_id,
            observed_at_us=observed_at_us,
            guard_us=guard_us,
            quantum_us=quantum_us,
        )

    @_runtime_serialized
    def complete_offline_phone_residency_stage(
        self,
        plan_id: str,
        receipts: Sequence[RuntimeTransitionReceipt],
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> OfflinePhoneResidencyPlan:
        return _offline_execution.complete_offline_phone_residency_stage(
            self,
            plan_id,
            receipts,
            snapshot=snapshot,
        )

    @_runtime_serialized
    def fail_offline_phone_residency_stage(
        self,
        plan_id: str,
        *,
        failed_at_us: int,
        reason: str,
        unavailable_session_ids: Sequence[str] = (),
        restored_session_generations: Mapping[str, int] | None = None,
    ) -> OfflinePhoneResidencyPlan:
        return _offline_execution.fail_offline_phone_residency_stage(
            self,
            plan_id,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable_session_ids=unavailable_session_ids,
            restored_session_generations=restored_session_generations,
        )

    @_runtime_serialized
    def adopt_offline_phone_residency(
        self,
        plan: OfflinePhoneResidencyPlan,
        requests_by_model: Mapping[str, Sequence[Request]],
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> OfflinePhoneResidencyPlan:
        """Adopt an exact persisted superset after a desktop-only restart."""
        return _offline_execution.adopt_offline_phone_residency(
            self,
            plan,
            requests_by_model,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
        )

    @_runtime_serialized
    def offline_phone_residency_snapshot(
        self, plan_id: str | None = None
    ) -> Mapping[str, object] | None:
        return _offline_execution.offline_phone_residency_snapshot(self, plan_id)

    @_runtime_serialized
    def phone_residency_session_states(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        return tuple(
            MappingProxyType(row.to_json())
            for row in self._model_placement_controller.phone_session_states()
        )

    def _phone_queue_demand(
        self,
        request: Request,
        manifest: ModelManifest,
    ) -> _PhoneQueueDemand:
        return _demand._phone_queue_demand(self, request, manifest)

    def _record_preparing_phone_layout_evaluation(
        self,
        request_id: str,
        observed_at_us: int,
        demand: _PhoneQueueDemand,
        preparing_layout: ModelPhoneResidencyLayout,
    ) -> None:
        return _portfolio._record_preparing_phone_layout_evaluation(
            self,
            request_id,
            observed_at_us,
            demand,
            preparing_layout,
        )

    def _discover_phone_residency_demand(
        self,
        compiler,
        queued_work_by_artifact: Mapping[str, int],
    ) -> _PhoneDemandDiscovery:
        return _demand._discover_phone_residency_demand(self, compiler, queued_work_by_artifact)

    def _record_phone_residency_demand_unavailable(
        self,
        request_id: str,
        observed_at_us: int,
        demand: _PhoneQueueDemand,
        route_evidence: Mapping[str, Mapping[str, object]],
    ) -> None:
        return _portfolio._record_phone_residency_demand_unavailable(
            self,
            request_id,
            observed_at_us,
            demand,
            route_evidence,
        )

    def _record_phone_shared_resource_mismatch(
        self,
        request_id: str,
        observed_at_us: int,
        demand: _PhoneQueueDemand,
        shared_domains: set[tuple[str, tuple[str, ...]]],
    ) -> None:
        return _portfolio._record_phone_shared_resource_mismatch(
            self,
            request_id,
            observed_at_us,
            demand,
            shared_domains,
        )

    def _phone_memory_budget(
        self,
        helper_id: str,
        sessions: tuple[RuntimePhoneSessionCapability, ...],
        snapshot: HeterogeneousRuntimeSnapshot | None,
    ) -> _PhoneMemoryBudget:
        return _economics._phone_memory_budget(self, helper_id, sessions, snapshot)

    def _phone_transition_estimates(
        self,
        helper_id: str,
        sessions: tuple[RuntimePhoneSessionCapability, ...],
        queued_work_by_artifact: Mapping[str, int],
    ) -> tuple[dict[str, int], dict[str, int]]:
        return _economics._phone_transition_estimates(
            self,
            helper_id,
            sessions,
            queued_work_by_artifact,
        )

    def _phone_switching_constraints(
        self,
        selected: PhoneFfnResidencyLayout | None,
        session_marginal_gains: tuple[PhoneSessionMarginalGain, ...],
    ) -> tuple[int, int]:
        return _economics._phone_switching_constraints(self, selected, session_marginal_gains)

    def _phone_selection_snapshot(
        self,
        demand: _PhoneQueueDemand,
        memory: _PhoneMemoryBudget,
        sessions: tuple[RuntimePhoneSessionCapability, ...],
        selected: PhoneFfnResidencyLayout | None,
        session_marginal_gains: tuple[PhoneSessionMarginalGain, ...],
    ) -> str:
        return _economics._phone_selection_snapshot(
            self,
            demand,
            memory,
            sessions,
            selected,
            session_marginal_gains,
        )

    def _phone_candidate_choice(
        self,
        discovery: _PhoneDemandDiscovery,
        demand: _PhoneQueueDemand,
        sessions: tuple[RuntimePhoneSessionCapability, ...],
        memory: _PhoneMemoryBudget,
        observed_at_us: int = 0,
    ) -> _PhoneCandidateChoice:
        return _economics._phone_candidate_choice(self, discovery, demand, sessions, memory, observed_at_us)

    def _phone_layout_request_impacts(self, layout, observed_at_us, transition_latency_us):
        return _economics._phone_layout_request_impacts(self, layout, observed_at_us, transition_latency_us)

    def _defer_phone_layout_revalidation(self, request_id, state, observed_at_us, transition_latency_us):
        return _economics._defer_phone_layout_revalidation(
            self, request_id, state, observed_at_us, transition_latency_us)

    def _confirm_phone_layout_selection(
        self,
        selected: PhoneFfnResidencyLayout | None,
        target_state: ModelPhoneResidencyLayout | None,
        selection_snapshot_sha256: str,
        observed_at_us: int,
        force: bool,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> tuple[bool, int]:
        return _economics._confirm_phone_layout_selection(
            self,
            selected,
            target_state,
            selection_snapshot_sha256,
            observed_at_us,
            force,
            snapshot,
        )

    def _phone_layout_confirmation_observation(
        self,
        snapshot: HeterogeneousRuntimeSnapshot | None,
        observed_at_us: int,
    ) -> tuple[str, int] | None:
        return _economics._phone_layout_confirmation_observation(self, snapshot, observed_at_us)

    def _reevaluate_pending_phone_layout_observation(
        self,
        ticket: RuntimeRequestTicket,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> None:
        return _economics._reevaluate_pending_phone_layout_observation(
            self,
            ticket,
            snapshot,
            observed_at_us,
        )

    def _propose_confirmed_phone_layout(
        self,
        selected: PhoneFfnResidencyLayout | None,
        selection_confirmed: bool,
        sessions: tuple[RuntimePhoneSessionCapability, ...],
        helper,
        demand: _PhoneQueueDemand,
        observed_at_us: int,
        reason: str,
        switching_margin_uj: int,
        minimum_residency_us: int,
        force: bool,
    ) -> None:
        return _economics._propose_confirmed_phone_layout(
            self,
            selected,
            selection_confirmed,
            sessions,
            helper,
            demand,
            observed_at_us,
            reason,
            switching_margin_uj,
            minimum_residency_us,
            force,
        )

    def _publish_phone_layout_view(
        self,
        selected: PhoneFfnResidencyLayout | None,
        changed: bool,
        queued_work_by_artifact: Mapping[str, int],
    ) -> None:
        return _portfolio._publish_phone_layout_view(
            self,
            selected,
            changed,
            queued_work_by_artifact,
        )

    def _record_phone_layout_selection(
        self,
        request_id: str,
        observed_at_us: int,
        demand: _PhoneQueueDemand,
        discovery: _PhoneDemandDiscovery,
        memory: _PhoneMemoryBudget,
        decision: _PhoneLayoutDecision,
    ) -> None:
        return _portfolio._record_phone_layout_selection(
            self,
            request_id,
            observed_at_us,
            demand,
            discovery,
            memory,
            decision,
        )

    def _update_phone_residency_portfolio(
        self,
        request: Request,
        manifest: ModelManifest,
        observed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> bool:
        """Publish the best arrived-work phone portfolio to route compilers."""
        return _portfolio._update_phone_residency_portfolio(
            self,
            request,
            manifest,
            observed_at_us,
            snapshot,
        )

    def _current_phone_projection_token(
        self,
    ) -> RuntimeResidencyProjectionToken | None:
        return _publication._current_phone_projection_token(self)

    def _phone_projection_token_for_plan(
        self, plan: RuntimeExecutionPlan
    ) -> RuntimeResidencyProjectionToken | None:
        return _publication._phone_projection_token_for_plan(self, plan)

    def _bind_phone_residency_transition(
        self,
        ticket: RuntimeRequestTicket,
        observed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> None:
        return _publication._bind_phone_residency_transition(self, ticket, observed_at_us, snapshot)

    def _observed_phone_layout_verification(
        self,
        state: ModelPhoneResidencyLayout,
        ticket: RuntimeRequestTicket,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[int, str] | None:
        """Verify that a proposed layout is already physically resident."""
        return _verification._observed_phone_layout_verification(self, state, ticket, snapshot)

    def _phone_layout_contract_identity(
        self,
        state: ModelPhoneResidencyLayout,
        artifact_sha256: str,
        plan: RuntimeExecutionPlan,
        snapshot: HeterogeneousRuntimeSnapshot,
    ):
        return _verification._phone_layout_contract_identity(
            self,
            state,
            artifact_sha256,
            plan,
            snapshot,
        )

    @staticmethod
    def _phone_layout_execution_path_ready(
        plan: RuntimeExecutionPlan,
        base_executor_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        require_base_executor_ready: bool,
    ) -> bool:
        return _verification._phone_layout_execution_path_ready(
            plan,
            base_executor_id,
            snapshot,
            require_base_executor_ready,
        )

    def _phone_layout_observed_residency(
        self,
        state: ModelPhoneResidencyLayout,
        phone_device_id: str,
        binding: RuntimeExecutorBinding,
        snapshot: HeterogeneousRuntimeSnapshot,
    ):
        return _verification._phone_layout_observed_residency(
            self,
            state,
            phone_device_id,
            binding,
            snapshot,
        )

    @staticmethod
    def _phone_layout_residency_matches(
        state: ModelPhoneResidencyLayout,
        exact_rows,
        session_by_id,
        observed_sessions,
        layout_generations,
    ) -> bool:
        return _verification._phone_layout_residency_matches(
            state,
            exact_rows,
            session_by_id,
            observed_sessions,
            layout_generations,
        )

    @staticmethod
    def _phone_layout_verification_proof(
        state: ModelPhoneResidencyLayout,
        *,
        model_id: str,
        artifact_sha256: str,
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        base_executor_id: str,
        phone_device_id: str,
        actual_shards,
        exact_rows,
        session_by_id,
        observed_sessions,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[int, str]:
        return _verification._phone_layout_verification_proof(
            state,
            model_id=model_id,
            artifact_sha256=artifact_sha256,
            plan=plan,
            binding=binding,
            base_executor_id=base_executor_id,
            phone_device_id=phone_device_id,
            actual_shards=actual_shards,
            exact_rows=exact_rows,
            session_by_id=session_by_id,
            observed_sessions=observed_sessions,
            snapshot=snapshot,
        )

    def _phone_layout_snapshot_verification(
        self,
        state: ModelPhoneResidencyLayout,
        *,
        model_id: str,
        artifact_sha256: str,
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        base_executor_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        require_base_executor_ready: bool = True,
    ) -> tuple[int, str] | None:
        """Bind phone readiness to one exact physical snapshot."""
        return _verification._phone_layout_snapshot_verification(
            self,
            state,
            model_id=model_id,
            artifact_sha256=artifact_sha256,
            plan=plan,
            binding=binding,
            base_executor_id=base_executor_id,
            snapshot=snapshot,
            require_base_executor_ready=require_base_executor_ready,
        )

    def _fail_phone_residency_transition(
        self,
        ticket_id: str,
        failed_at_us: int,
        reason: str,
    ) -> None:
        return _publication._fail_phone_residency_transition(self, ticket_id, failed_at_us, reason)

    def _phone_residency_portfolio_authorization(
        self,
        candidate: AutomatedRouteCandidate,
        manifest: ModelManifest,
    ) -> Mapping[str, object] | None:
        """Return the queue authorization for one selected cold layout."""
        return _publication._phone_residency_portfolio_authorization(self, candidate, manifest)

    def _apply_phone_residency_portfolio_authorization(
        self,
        candidate_set: AutomatedCandidateSet,
        manifest: ModelManifest,
        request: Request,
        *, snapshot: HeterogeneousRuntimeSnapshot | None = None,
        observed_at_us: int | None = None,
    ) -> AutomatedCandidateSet:
        """Replace request-local cold rejection with queue evidence."""
        return _publication._apply_phone_residency_portfolio_authorization(
            self,
            candidate_set,
            manifest,
            request,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
        )

    def phone_residency_events(self) -> tuple[Mapping[str, object], ...]:
        return (
            self._model_placement_controller.phone_layout_events()
        )

    def request_helper_events(self) -> tuple[Mapping[str, object], ...]:
        return self._model_placement_controller.request_helper_events()

    @_runtime_serialized
    def replay_observed_phone_layout(
        self,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> Mapping[str, object]:
        """Publish a proposed layout proven by one replay snapshot."""
        return _verification.replay_observed_phone_layout(self, snapshot)

    @_runtime_serialized
    def record_runtime_decode_progress(
        self,
        request_id: str,
        *,
        token_index: int,
        at_us: int,
    ) -> int:
        """Record completed decode work for arrived-work placement scoring."""

        ticket = self.runtime_execution_ticket(request_id)
        if type(at_us) is not int or at_us < 0:
            raise UnifiedScheduleError(
                "runtime decode progress time is invalid"
            )
        try:
            self._model_placement_controller.record_request_decode_progress(
                request_id, token_index
            )
            return (
                self._model_placement_controller
                .remaining_request_decode_tokens(
                    request_id, ticket.request.output_tokens
                )
            )
        except ModelPlacementControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def _reevaluate_pending_phone_layout_at_boundary(
        self,
        ticket: RuntimeRequestTicket,
        observed_at_us: int,
    ) -> None:
        """Supply changed arrived work to phone-layout hysteresis."""
        return _portfolio._reevaluate_pending_phone_layout_at_boundary(self, ticket, observed_at_us)
