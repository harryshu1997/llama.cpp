"""Runtime request API: placement, executors, decision log, submit/wait/replan/fail/extend/complete.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence
from .._internal.policy import (
    Decision,
    LeaseRecord,
    MarginalSystemCostContext,
    Request,
    SchedulerError,
)
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_gates import RuntimeSnapshot
from .._internal.runtime_placement import (
    RuntimePlacementCandidate,
    RuntimePlacementDecision,
    RuntimePlacementError,
    RuntimePlacementPlanner,
    RuntimePlacementSnapshot,
)
from .._internal.runtime_cost import (
    RuntimeCostError,
    RuntimeCostEstimateSet,
    RuntimeCostEstimator,
    RuntimeExecutorBinding,
    RuntimeExecutorRegistry,
    RuntimeModelArtifact,
)
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.runtime_plan import (
    RuntimeHelperExecutionEnvelope,
    RuntimeExecutionReceipt,
)
from .._internal.runtime_residency_cohorts import runtime_residency_component_identity
from .._internal.runtime_decode_cohort import (
    RuntimeDecodeCohortError,
    RuntimeDecodeCohortReceipt,
)
from .._internal.online_placement import (
    ONLINE_ROUTE_FAMILIES,
    OnlinePlacementError,
    OnlinePlacementReceipt,
)
from .._internal.runtime_controller import (
    RuntimeCompletionReceipt,
    RuntimeControllerError,
    RuntimeFailureRecovery,
    RuntimeLeaseExtensionReceipt,
    RuntimeRequestTicket,
)
from .._internal.runtime_admission import (
    RuntimeAdmissionError,
    RuntimeCandidateBuilder,
    RuntimeExecutorObservation,
    RuntimeRequestObservation,
)
from .._internal.runtime_execution import (
    RuntimeExecutionCoordinatorError,
    RuntimeExecutionFailure,
    RuntimeLeaseRenewalCoordinator,
)
from .._internal.runtime_phase import (
    RuntimePhaseError,
    RuntimePhaseLeaseController,
    RuntimePhaseObservation,
)
from .._internal.decision_log import DecisionLogError, RuntimeDecisionLog
from .._internal.runtime_dispatch_policy import RuntimeDispatchPolicy
from .._internal.types import canonical_sha256
from .._internal.adaptive_decode_contracts import AdaptiveDecodeError
from .helper_preparation_ops.common import renew_shared_helper_leases
from .common import (
    _runtime_serialized,
)


class RuntimeRequestMixin:
    """Runtime request API: placement, executors, decision log, submit/wait/replan/fail/extend/complete."""

    @_runtime_serialized
    def runtime_receipt_energy_context(
        self, ticket_id: str, device_ids: Sequence[str],
        start_us: int, end_us: int,
    ) -> Mapping[str, object]:
        catalog = self._runtime_capabilities
        plan = self._runtime_controller.acquired_execution_plan(ticket_id)
        if (
            catalog is None
            or plan is None
            or not device_ids
            or not set(device_ids).issubset(plan.device_ids)
            or any(
                device_id not in catalog.placement_profile.devices
                for device_id in device_ids
            )
        ):
            return {"reason": "ENERGY_LEDGER_IDENTITY_UNAVAILABLE"}
        if any(
            catalog.placement_profile.devices[device_id].kind.startswith("phone")
            for device_id in plan.device_ids
        ):
            return {"reason": "ENERGY_ROUTE_TOUCHES_PHONE"}
        server_resources = tuple(sorted({
            resource_id
            for executor in catalog.executors
            if not catalog.placement_profile.devices[executor.device_id].kind.startswith("phone")
            for resource_id in executor.execution_resource_ids
        }))
        overlaps = self._runtime_controller.acquired_lease_overlaps(
            ticket_id, server_resources, start_us, end_us
        )
        return {
            "reason": (
                "ENERGY_LEDGER_WINDOW_UNAVAILABLE" if overlaps is None
                else "ENERGY_SERVER_LEASE_OVERLAP" if overlaps
                else "ENERGY_ISOLATED_SERVER_RECEIPT"
            ),
            "overlapping_ticket_ids": overlaps or (),
        }

    def runtime_protected_work_end_us(self, ticket_ids: Sequence[str]) -> int | None:
        return self._runtime_controller.acquired_prediction_end_us(ticket_ids)

    def runtime_external_desktop_activity(
        self, ticket_id: str, start_us: int, end_us: int,
    ) -> Mapping[str, object] | None:
        """Name other acquired tickets charging server resources across a window.

        This is a measurement-context identity, not an execution-compatibility
        identity: queued requests, lease renewals and phone-only work do not
        change it. None means the acquisition history cannot cover the window.
        """
        catalog = self._runtime_capabilities
        if catalog is None or end_us <= start_us:
            return None
        resources = tuple(sorted({
            resource_id for executor in catalog.executors
            if not catalog.placement_profile.devices[executor.device_id].kind.startswith("phone")
            for resource_id in executor.execution_resource_ids
        }))
        overlaps = self._runtime_controller.acquired_lease_overlaps(
            ticket_id, resources, start_us, end_us,
        )
        if overlaps is None:
            return None
        batch_members: tuple[str, ...] = ()
        if self._runtime_controller.dispatch_policy.continuous_join:
            # Co-tenants of the same model on the same server are batch
            # composition (tracked per active_batch), not external activity.
            batch_members, overlaps = self._same_server_batch_members(ticket_id, overlaps)
        return {
            "sha256": canonical_sha256({"external_desktop_ticket_ids": list(overlaps)}),
            "ticket_ids": tuple(overlaps),
            "server_resource_ids": resources,
            **({} if not batch_members else {"batch_member_ticket_ids": batch_members}),
        }

    def _same_server_batch_members(
        self, ticket_id: str, overlaps: Sequence[str],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Split overlapping acquired tickets into same-server co-tenants and the rest."""
        controller = self._runtime_controller
        own = controller.acquired_ticket(ticket_id)
        members: list[str] = []
        external: list[str] = []
        for other_id in overlaps:
            other = controller.acquired_ticket(other_id)
            if (
                own is not None and other is not None
                and own.binding.endpoint is not None
                and other.model.artifact_sha256 == own.model.artifact_sha256
                and other.binding.endpoint == own.binding.endpoint
            ):
                members.append(other_id)
            else:
                external.append(other_id)
        return tuple(members), tuple(external)

    def runtime_protected_work_power_context(
        self, ticket_ids: Sequence[str], start_us: int, end_us: int,
    ) -> Mapping[str, object]:
        """Reject host power already containing another route's charged work."""
        catalog = self._runtime_capabilities
        if (catalog is None or end_us <= start_us
                or self.runtime_protected_work_end_us(ticket_ids) is None):
            return {"reason": "PROTECTED_POWER_IDENTITY_UNAVAILABLE"}
        resources = tuple(sorted({
            resource_id for executor in catalog.executors
            if not catalog.placement_profile.devices[executor.device_id].kind.startswith("phone")
            for resource_id in executor.execution_resource_ids
        }))
        other = set()
        for ticket_id in ticket_ids:
            overlaps = self._runtime_controller.acquired_lease_overlaps(
                ticket_id, resources, start_us, end_us)
            if overlaps is None:
                return {"reason": "PROTECTED_POWER_WINDOW_UNAVAILABLE"}
            other.update(set(overlaps) - set(ticket_ids))
        return {
            "reason": "PROTECTED_POWER_OVERLAP" if other else "PROTECTED_POWER_ISOLATED",
            "overlapping_ticket_ids": tuple(sorted(other)),
        }

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

    @property
    def runtime_executor_registry(self) -> RuntimeExecutorRegistry:
        if self._runtime_executor_registry is None:
            raise UnifiedScheduleError("runtime executor registry is empty")
        return self._runtime_executor_registry

    def _register_runtime_executors(
        self, bindings: Sequence[RuntimeExecutorBinding]
    ) -> RuntimeExecutorRegistry:
        try:
            reported = RuntimeExecutorRegistry.from_bindings(bindings)
            effective = self._runtime_controller.effective_bindings(
                tuple(reported.bindings.values())
            )
            registry = RuntimeExecutorRegistry.from_bindings(effective)
        except (RuntimeCostError, RuntimeControllerError) as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        self._runtime_executor_registry = registry
        return registry

    def _materialize_runtime_bindings(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        rows: Sequence[
            RuntimeExecutorBinding | RuntimeExecutorObservation
        ],
    ) -> tuple[RuntimeExecutorBinding, ...]:
        values = tuple(rows)
        try:
            if values and all(
                isinstance(row, RuntimeExecutorObservation)
                for row in values
            ):
                return RuntimeCandidateBuilder.build(
                    self.profile_for(request.workload_id),
                    model,
                    values,
                )
            if values and all(
                isinstance(row, RuntimeExecutorBinding) for row in values
            ):
                return values
        except RuntimeAdmissionError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        raise UnifiedScheduleError(
            "runtime executors must be one non-empty observation set"
        )

    @staticmethod
    def _runtime_observation(
        request: Request,
        observed_at_us: int,
        observation: RuntimeRequestObservation | None,
    ) -> tuple[RuntimeRequestObservation, Request]:
        try:
            value = (
                RuntimeRequestObservation(observed_at_us, request.features)
                if observation is None
                else observation
            )
            if not isinstance(value, RuntimeRequestObservation):
                raise RuntimeAdmissionError(
                    "runtime request observation is invalid"
                )
            if value.captured_at_us > observed_at_us:
                raise RuntimeAdmissionError(
                    "runtime request observation is from the future"
                )
            return value, value.evaluation_request(request)
        except RuntimeAdmissionError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @staticmethod
    def _require_runtime_identity(
        ticket: RuntimeRequestTicket,
        request: Request,
        model: RuntimeModelArtifact,
    ) -> None:
        if request != ticket.request:
            raise UnifiedScheduleError(
                "runtime request identity differs from its ticket"
            )
        if model != ticket.model:
            raise UnifiedScheduleError(
                "runtime model identity differs from its ticket"
            )

    @staticmethod
    def _runtime_family_routes(
        registry: RuntimeExecutorRegistry,
    ) -> Mapping[str, str | None]:
        routes: dict[str, str | None] = {
            family: None for family in ONLINE_ROUTE_FAMILIES
        }
        for binding in registry.bindings.values():
            family = binding.route_family
            if family not in routes:
                raise UnifiedScheduleError(
                    "runtime binding lacks a known route family"
                )
            if routes[family] is not None:
                raise UnifiedScheduleError(
                    "runtime route family has multiple bindings"
                )
            routes[family] = binding.route_id
        return MappingProxyType(routes)

    def _estimate_registered_runtime_costs(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        registry: RuntimeExecutorRegistry,
        *,
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
    ) -> RuntimeCostEstimateSet:
        try:
            return RuntimeCostEstimator().estimate(
                profile=self.profile_for(request.workload_id),
                request=request,
                model=model,
                bindings=tuple(registry.bindings.values()),
                snapshot=snapshot,
                now_us=now_us,
            )
        except RuntimeCostError as exc:
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
        registry = self._register_runtime_executors(bindings)
        return self._estimate_registered_runtime_costs(
            request,
            model,
            registry,
            snapshot=snapshot,
            now_us=now_us,
        )

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
            registry = self._register_runtime_executors(bindings)
            binding_rows = tuple(registry.bindings.values())
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
                    bindings=binding_rows,
                    snapshot=snapshot,
                    observed_at_us=observed_at_us,
                    family_routes=family_routes,
                    scheduler_state=scheduler_state,
                )
            )
            estimates = self._estimate_registered_runtime_costs(
                request,
                model,
                registry,
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

    def _selected_runtime_binding(
        self,
        estimates: RuntimeCostEstimateSet,
        decision: Decision,
    ) -> RuntimeExecutorBinding:
        binding = self.runtime_executor_registry.binding(decision.route_id)
        estimate = next(
            (
                row for row in estimates.estimates
                if row.route_id == decision.route_id
            ),
            None,
        )
        if (
            binding is None
            or estimate is None
            or not estimate.admitted
            or estimate.executor_id != binding.executor_id
        ):
            raise UnifiedScheduleError(
                "selected runtime route has no admitted executor binding"
            )
        return binding

    def _runtime_log_body(
        self,
        event_kind: str,
        ticket: RuntimeRequestTicket,
        event_time_us: int,
        lifecycle_state: str,
    ) -> Mapping[str, object]:
        profile_sha256 = ticket.planning_profile_sha256
        if profile_sha256 is None:
            profile_sha256 = canonical_sha256(
                self.profile_for(ticket.request.workload_id)
            )
        bindings = {
            row.route_id: row for row in ticket.executor_bindings
        }
        selection_rejections = dict(ticket.decision.rejected)
        candidates = []
        cost_json_by_route = {}
        if event_kind in {"DECISION", "FALLBACK", "REPLAN"}:
            for estimate in ticket.cost_estimates.estimates:
                binding = bindings.get(estimate.route_id)
                cost_json = estimate.to_json(copy_details=False)
                cost_json_by_route[estimate.route_id] = cost_json
                row = dict(cost_json)
                row.update({
                    "executor": (
                        None if binding is None else binding.to_json()
                    ),
                    "readiness": (
                        "ABSENT"
                        if binding is None
                        else "READY" if binding.ready else "NOT_READY"
                    ),
                    "selection_status": (
                        "SELECTED"
                        if estimate.route_id == ticket.decision.route_id
                        else selection_rejections.get(
                            estimate.route_id, "NOT_SELECTED"
                        )
                    ),
                })
                candidates.append(row)
        selected_cost = next(
            row for row in ticket.cost_estimates.estimates
            if row.route_id == ticket.decision.route_id
        )
        online_receipt = ticket.online_placement_receipt
        causal_hash = (
            online_receipt.causal_input_sha256
            if online_receipt is not None
            else canonical_sha256({
                "runtime_observation": (
                    ticket.runtime_observation.to_json()
                ),
                "runtime_snapshot_sha256": (
                    ticket.cost_estimates.runtime_system_snapshot_sha256
                    or canonical_sha256(ticket.cost_estimates.snapshot)
                ),
            })
        )
        model_sha256 = (
            ticket.cost_estimates.model_manifest_sha256
            or canonical_sha256(ticket.model)
        )
        runtime_snapshot_sha256 = (
            ticket.cost_estimates.runtime_system_snapshot_sha256
            or canonical_sha256(ticket.cost_estimates.snapshot)
        )
        selected = {
            "cost": (
                cost_json_by_route[selected_cost.route_id]
                if selected_cost.route_id in cost_json_by_route
                else selected_cost.to_json(copy_details=False)
            ),
            "executor": ticket.binding.to_json(),
            "resource_leases": [
                {
                    "lanes": list(lease.lanes),
                    "lease_id": lease.lease_id,
                    "predicted_end_us": lease.predicted_end_us,
                    "reserved_until_us": lease.reserved_until_us,
                    "resource_id": lease.resource_id,
                    "start_us": lease.start_us,
                    "token": lease.token,
                }
                for lease in ticket.decision.leases
            ],
            "route_id": ticket.decision.route_id,
        }
        placement_epoch = selected_cost.details.get(
            "model_placement_epoch"
        )
        if placement_epoch is not None:
            selected["model_placement_epoch"] = placement_epoch
            selected["model_placement_epoch_invalidation_reason"] = (
                selected_cost.details.get(
                    "model_placement_epoch_invalidation_reason", "NONE"
                )
            )
        placement_resolution = selected_cost.details.get(
            "model_placement_resolution"
        )
        if placement_resolution is not None:
            selected["model_placement_resolution"] = (
                placement_resolution
            )
        if ticket.failure_reason is not None:
            selected["failure_reason"] = ticket.failure_reason
        if ticket.execution_plan is not None:
            selected["operator_plan"] = ticket.execution_plan.to_json()
            helper_events = (
                self._model_placement_controller.request_helper_events(
                    ticket.request.request_id
                )
            )
            if helper_events:
                selected["request_helper_events"] = [
                    dict(row) for row in helper_events
                ]
            helper_preparations = tuple(
                row for row in self._request_helper_preparations.values()
                if row.request_id == ticket.request.request_id
            )
            if helper_preparations:
                selected["request_helper_preparations"] = [
                    self._helper_preparation_json(row)
                    for row in sorted(
                        helper_preparations,
                        key=lambda value: value.preparation_ticket_id,
                    )
                ]
            selected["memory_reservations"] = [
                row.to_json() for row in ticket.memory_reservations
            ]
            selected["transition_status"] = ticket.transition_status
            selected["transition_receipts"] = [
                row.to_json() for row in ticket.transition_receipts
            ]
            selected["previous_transition_receipts"] = [
                row.to_json()
                for row in ticket.previous_transition_receipts
            ]
            selected["execution_receipt"] = (
                None
                if ticket.execution_receipt is None
                else ticket.execution_receipt.to_json()
            )
        if ticket.residency_projection_token is not None:
            selected["residency_projection_token"] = (
                ticket.residency_projection_token.to_json()
            )
        dispatch_note = self._runtime_controller.dispatch_policy_note(
            ticket.ticket_id
        )
        if dispatch_note is not None:
            selected["dispatch_policy"] = dict(dispatch_note)
        if ticket.decode_cohort is not None:
            selected["decode_cohort"] = ticket.decode_cohort.to_json()
            cohort_receipt = self._runtime_decode_cohorts.receipt(
                ticket.decode_cohort.cohort_id
            )
            if cohort_receipt is not None:
                if (
                    ticket.request.request_id
                    == cohort_receipt.energy_owner_request_id
                ):
                    selected["decode_cohort_receipt"] = (
                        cohort_receipt.to_json()
                    )
                else:
                    selected["decode_cohort_receipt_sha256"] = (
                        cohort_receipt.receipt_sha256
                    )
        return {
            "attempt_index": ticket.attempt_index,
            "causal_input_sha256": causal_hash,
            "candidates": candidates,
            "decision_kind": "runtime_route",
            "decision_reason": ticket.decision.reason,
            "event_kind": event_kind,
            "event_time_us": event_time_us,
            "lifecycle_state": lifecycle_state,
            "model_sha256": model_sha256,
            "previous_ticket_id": ticket.previous_ticket_id,
            "profile_sha256": profile_sha256,
            "request_ids": [ticket.request.request_id],
            "request_sha256": canonical_sha256(ticket.request),
            "runtime_snapshot_sha256": runtime_snapshot_sha256,
            "selected": selected,
            "ticket_id": ticket.ticket_id,
        }

    def _append_runtime_log(
        self,
        event_kind: str,
        ticket: RuntimeRequestTicket,
        event_time_us: int,
        lifecycle_state: str,
    ) -> None:
        try:
            self._runtime_decision_log.append(
                self._runtime_log_body(
                    event_kind,
                    ticket,
                    event_time_us,
                    lifecycle_state,
                ),
                return_record=False,
                precanonical_body=True,
            )
        except DecisionLogError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        shadow = getattr(self, "_joint_planner_shadow", None)
        if shadow is not None:
            shadow.observe_ticket(self, event_kind, ticket, event_time_us, lifecycle_state)
        active = getattr(self, "_joint_planner_active", None)
        if active is not None:
            active.observe_ticket(self, event_kind, ticket, event_time_us, lifecycle_state)

    def runtime_decision_log(self) -> dict[str, object]:
        return self._runtime_decision_log.snapshot()

    def runtime_decision_log_bytes(self) -> bytes:
        return self._runtime_decision_log.canonical_bytes()

    def validate_runtime_decision_log(self, value: object) -> None:
        try:
            RuntimeDecisionLog.validate(value)
        except DecisionLogError as exc:
            raise DecisionLogError(str(exc)) from exc

    def _schedule_runtime_fallback(
        self,
        request: Request,
        estimates: RuntimeCostEstimateSet,
        *,
        runtime_now_us: int,
        marginal_system_context: MarginalSystemCostContext | None,
    ) -> Decision:
        admissions = {
            estimate.route_id: (
                "ADMITTED"
                if estimate.baseline and estimate.admitted
                else estimate.reason
                if not estimate.admitted
                else "RECOVERY_FALLBACK_ONLY"
            )
            for estimate in estimates.estimates
        }
        baseline = next(
            (row for row in estimates.estimates if row.baseline), None
        )
        if baseline is None or not baseline.admitted:
            reason = "ABSENT" if baseline is None else baseline.reason
            raise UnifiedScheduleError(
                "qualified runtime fallback is unavailable: " + reason
            )
        profile = self.profile_for(request.workload_id)
        try:
            decision = self._route_policies[profile.profile_id].schedule(
                request,
                runtime_now_us,
                runtime_route_admissions=admissions,
                marginal_system_context=marginal_system_context,
            )
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if decision.route_id != baseline.route_id:
            self.cancel(decision.request_id, 0)
            raise UnifiedScheduleError(
                "runtime recovery did not select the qualified fallback"
            )
        return decision

    @_runtime_serialized
    def submit_runtime_request(
        self,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[
            RuntimeExecutorBinding | RuntimeExecutorObservation
        ],
        *,
        snapshot: RuntimePlacementSnapshot,
        observed_at_us: int,
        runtime_observation: RuntimeRequestObservation | None = None,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> RuntimeRequestTicket:
        """Observe, select, bind, reserve, and queue one runtime request."""
        observation, evaluation_request = self._runtime_observation(
            request, observed_at_us, runtime_observation
        )
        binding_rows = self._materialize_runtime_bindings(
            request, model, bindings
        )
        registry = self._register_runtime_executors(binding_rows)
        family_routes = self._runtime_family_routes(registry)
        estimates, decision, online_receipt = self.schedule_online_request(
            evaluation_request,
            model,
            tuple(registry.bindings.values()),
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            family_routes=family_routes,
            marginal_system_context=marginal_system_context,
        )
        binding = self._selected_runtime_binding(estimates, decision)
        try:
            ticket = self._runtime_controller.admit(
                request=request,
                model=model,
                runtime_observation=observation,
                estimates=estimates,
                decision=decision,
                binding=binding,
                executor_bindings=tuple(registry.bindings.values()),
                online_receipt=online_receipt,
                admitted_at_us=observed_at_us,
            )
        except RuntimeControllerError as exc:
            self.cancel(decision.request_id, 0)
            raise UnifiedScheduleError(str(exc)) from exc
        self._append_runtime_log(
            "DECISION", ticket, observed_at_us, ticket.dispatch_state
        )
        return ticket

    def runtime_helper_state_generation(self) -> int:
        """Cheap monotonic marker of helper-relevant scheduler state.

        Advances on phone layout events, request helper events, request
        acquisitions and pending-candidate updates; the physical adapter's
        per-request watcher re-plans only when it moves or a bounded
        fallback interval elapses.
        """
        return self._model_placement_controller.helper_state_generation()

    def runtime_ticket(self, request_id: str) -> RuntimeRequestTicket:
        try:
            return self._runtime_controller.ticket(request_id)
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def runtime_ticket_by_id(self, ticket_id: str) -> RuntimeRequestTicket:
        try:
            return self._runtime_controller.ticket_by_id(ticket_id)
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def runtime_attached_helper_envelope(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
    ) -> RuntimeHelperExecutionEnvelope | None:
        """Return the exact helper attached to an acquired base ticket."""

        ticket = self.runtime_execution_ticket(request_id)
        if ticket.ticket_id != expected_ticket_id:
            raise UnifiedScheduleError(
                "runtime helper ticket identity changed"
            )
        plan = ticket.execution_plan
        if plan is None:
            raise UnifiedScheduleError(
                "runtime helper base plan is absent"
            )
        binding = self._model_placement_controller.request_binding(
            request_id
        )
        bound_envelope = (
            None if binding is None else binding.get("helper_envelope")
        )
        attachment = (
            None if binding is None else binding.get("helper_attachment")
        )
        if attachment is None:
            return None
        if (
            not isinstance(bound_envelope, Mapping)
            or not isinstance(attachment, Mapping)
        ):
            raise UnifiedScheduleError(
                "runtime helper attachment lacks its exact envelope"
            )

        def matches_bound_envelope(
            helper: RuntimeHelperExecutionEnvelope,
        ) -> bool:
            contract = helper.helper_plan.execution_contract
            return bool(
                helper.route_id == bound_envelope.get("route_id")
                and helper.operator_plan_sha256
                    == bound_envelope.get("operator_plan_sha256")
                and helper.desktop_parent_route_id
                    == bound_envelope.get("desktop_parent_route_id")
                and helper.desktop_placement_sha256
                    == bound_envelope.get("desktop_placement_sha256")
                and helper.phone_layout_generation
                    == bound_envelope.get("phone_layout_generation")
                and helper.phone_layout_geometry_sha256
                    == bound_envelope.get(
                        "phone_layout_geometry_sha256"
                    )
                and helper.activation_dtype
                    == bound_envelope.get("activation_dtype")
                and helper.resident_layer_mask
                    == bound_envelope.get("assisted_layer_mask")
                and helper.resident_columns
                    == bound_envelope.get("maximum_columns")
                and tuple(sorted(
                    contract.allowed_adaptive_fractions_ppm
                )) == tuple(bound_envelope.get(
                    "allowed_fractions_ppm", ()
                ))
                and tuple(sorted(
                    row.session_id for row in contract.phone_shards
                )) == tuple(bound_envelope.get("phone_session_ids", ()))
                and tuple(sorted(helper.helper_plan.resource_ids))
                    == tuple(bound_envelope.get("resource_ids", ()))
            )

        candidates = []
        context = self._late_request_helper_contexts.get(request_id)
        if context is not None:
            candidates.append(context.helper)
        remembered = self._request_helper_envelope_history.get(
            request_id, {}
        ).get(str(bound_envelope.get("operator_plan_sha256", "")))
        if remembered is not None:
            candidates.append(remembered)
        if plan.helper_envelope is not None:
            candidates.append(plan.helper_envelope)
        exact = []
        for helper in candidates:
            if helper in exact or not matches_bound_envelope(helper):
                continue
            exact.append(helper)
        if len(exact) != 1:
            raise UnifiedScheduleError(
                "runtime helper envelope cannot be resolved from its "
                "bound identity"
            )
        helper = exact[0]
        if (
            attachment.get("phone_layout_generation")
                != bound_envelope.get("phone_layout_generation")
            or attachment.get("phone_layout_geometry_sha256")
                != bound_envelope.get("phone_layout_geometry_sha256")
            or attachment.get("operator_plan_sha256")
                != bound_envelope.get("operator_plan_sha256")
            or tuple(attachment.get("phone_session_ids", ()))
                != tuple(bound_envelope.get("phone_session_ids", ()))
            or bound_envelope.get("desktop_parent_route_id")
                != ticket.decision.route_id
            or bound_envelope.get("desktop_placement_sha256")
                != plan.desktop_placement_sha256
        ):
            raise UnifiedScheduleError(
                "runtime helper attachment differs from its envelope"
            )
        self._remember_request_helper_envelope(request_id, helper)
        return helper

    @_runtime_serialized
    def runtime_attached_helper_envelopes(
        self,
        request_id: str,
        *,
        expected_ticket_id: str,
    ) -> tuple[RuntimeHelperExecutionEnvelope, ...]:
        """Return every exact helper identity used by one base request."""

        current = self.runtime_attached_helper_envelope(
            request_id, expected_ticket_id=expected_ticket_id
        )
        if current is not None:
            self._remember_request_helper_envelope(request_id, current)
        return tuple(sorted(
            self._request_helper_envelope_history.get(
                request_id, {}
            ).values(),
            key=lambda row: (
                row.phone_layout_generation,
                row.operator_plan_sha256,
            ),
        ))

    def wait_runtime_request(
        self, request_id: str, epoch_ns: int
    ) -> RuntimeRequestTicket:
        try:
            cohort = self._runtime_decode_cohorts.wait_until_sealed(
                request_id
            )
            if cohort is not None:
                with self._runtime_lock:
                    with self._transaction(convert=False):
                        if cohort.active_batch == 1:
                            self.timeline.reassign_owner(
                                cohort.shared_lease_tokens,
                                expected_owner_id=cohort.cohort_id,
                                owner_id=request_id,
                            )
                            leases = (
                                self._runtime_decode_cohorts
                                .dissolve_singleton(request_id)
                            )
                            self._runtime_controller\
                                .dissolve_decode_cohort_singleton(
                                    request_id, cohort, leases
                                )
                        else:
                            for member_id in cohort.member_request_ids:
                                self._runtime_controller.bind_decode_cohort(
                                    member_id, cohort
                                )
        except (
            RuntimeControllerError,
            RuntimeDecodeCohortError,
            SchedulerError,
        ) as exc:
            raise UnifiedScheduleError(str(exc)) from exc

        def commit_acquired(ticket: RuntimeRequestTicket) -> None:
            if (
                ticket.dispatch_state == "ACQUIRED"
                and ticket.transition_status != "PENDING"
                and not self._runtime_decision_log.has_acquired(
                    ticket.ticket_id
                )
            ):
                if ticket.execution_plan is not None:
                    self._model_placement_controller.mark_request_acquired(
                        ticket.request.request_id
                    )
                self._append_runtime_log(
                    "ACQUIRED",
                    ticket,
                    ticket.dispatch_receipt.observed_at_us,
                    ticket.dispatch_state,
                )

        while True:
            try:
                receipt = self._runtime_controller.wait_ready(
                    request_id, epoch_ns
                )
            except RuntimeControllerError as exc:
                try:
                    terminal = self._runtime_controller.ticket(request_id)
                except RuntimeControllerError:
                    raise UnifiedScheduleError(str(exc)) from exc
                if terminal.dispatch_state in {
                    "CANCELLED", "COMPLETED", "FAILED"
                }:
                    return terminal
                raise UnifiedScheduleError(str(exc)) from exc
            with self._runtime_lock:
                try:
                    ticket = self._runtime_controller.commit_wait(
                        receipt,
                        commit_acquired=commit_acquired,
                    )
                except RuntimeControllerError as exc:
                    raise UnifiedScheduleError(str(exc)) from exc
                if ticket is not None and self._phone_reprovisioning is not None:
                    self._reevaluate_phone_layout_for_desktop_load(ticket)
            if ticket is not None:
                return ticket

    def runtime_decode_cohort_snapshot(self) -> Mapping[str, object]:
        return self._runtime_decode_cohorts.snapshot()

    @_runtime_serialized
    def record_decode_cohort_measurement(
        self,
        cohort_id: str,
        *,
        started_at_us: int,
        finished_at_us: int,
        fleet_energy_uj_by_domain: Mapping[str, int],
        transfer_energy_uj_by_link: Mapping[str, int],
        measurement_evidence_ids: Sequence[str],
        attribution_kind: str,
        energy_boundary_id: str,
        total_input_tokens: int,
        total_output_tokens: int,
        energy_estimation_metadata: Mapping[
            str, int | str | bool
        ] | None = None,
    ) -> RuntimeDecodeCohortReceipt:
        """Normalize and retain one non-additive cohort energy receipt."""
        with self._transaction():
            binding = self._runtime_decode_cohorts.binding_by_id(cohort_id)
            if not binding.sealed:
                raise UnifiedScheduleError(
                    "decode cohort measurement precedes sealing"
                )
            receipt = RuntimeDecodeCohortReceipt(
                cohort_id=cohort_id,
                member_request_ids=binding.member_request_ids,
                energy_owner_request_id=binding.leader_request_id,
                common_policy_sha256=binding.common_policy_sha256,
                active_batch=binding.active_batch,
                started_at_us=started_at_us,
                finished_at_us=finished_at_us,
                fleet_energy_uj_by_domain=fleet_energy_uj_by_domain,
                transfer_energy_uj_by_link=transfer_energy_uj_by_link,
                measurement_evidence_ids=tuple(
                    measurement_evidence_ids
                ),
                attribution_kind=attribution_kind,
                energy_boundary_id=energy_boundary_id,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                energy_estimation_metadata=(
                    {} if energy_estimation_metadata is None
                    else energy_estimation_metadata
                ),
            )
            self._runtime_decode_cohorts.record_receipt(receipt)
            return receipt

    @_runtime_serialized
    def replan_runtime_request(
        self,
        request_id: str,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[
            RuntimeExecutorBinding | RuntimeExecutorObservation
        ],
        *,
        snapshot: RuntimePlacementSnapshot,
        observed_at_us: int,
        reason: str,
        runtime_observation: RuntimeRequestObservation | None = None,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> RuntimeRequestTicket:
        current = self.runtime_ticket(request_id)
        self._require_runtime_identity(current, request, model)
        observation, evaluation_request = self._runtime_observation(
            request, observed_at_us, runtime_observation
        )
        binding_rows = self._materialize_runtime_bindings(
            request, model, bindings
        )
        try:
            previous = self._runtime_controller.prepare_replan(
                request_id,
                observed_at_us,
                self.cancel,
                self._runtime_memory.release_owner,
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        registry = self._register_runtime_executors(binding_rows)
        estimates = self._estimate_registered_runtime_costs(
            evaluation_request,
            model,
            registry,
            snapshot=snapshot,
            now_us=observed_at_us,
        )
        decision = self.schedule_runtime_costs(
            evaluation_request,
            estimates,
            runtime_now_us=observed_at_us,
            marginal_system_context=marginal_system_context,
        )
        binding = self._selected_runtime_binding(estimates, decision)
        try:
            ticket = self._runtime_controller.admit(
                request=request,
                model=model,
                runtime_observation=observation,
                estimates=estimates,
                decision=decision,
                binding=binding,
                executor_bindings=tuple(registry.bindings.values()),
                online_receipt=None,
                admitted_at_us=observed_at_us,
                previous_ticket_id=previous.ticket_id,
                failure_reason=reason,
            )
        except RuntimeControllerError as exc:
            self.cancel(decision.request_id, 0)
            raise UnifiedScheduleError(str(exc)) from exc
        self._append_runtime_log(
            "REPLAN", ticket, observed_at_us, ticket.dispatch_state
        )
        return ticket

    @_runtime_serialized
    def fail_runtime_request(
        self,
        request_id: str,
        *,
        failed_at_us: int,
        reason: str,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[
            RuntimeExecutorBinding | RuntimeExecutorObservation
        ],
        snapshot: RuntimePlacementSnapshot,
        runtime_observation: RuntimeRequestObservation | None = None,
        physical_failure: RuntimeExecutionFailure | None = None,
        marginal_system_context: MarginalSystemCostContext | None = None,
    ) -> RuntimeFailureRecovery:
        current = self.runtime_ticket(request_id)
        self._require_runtime_identity(current, request, model)
        observation, evaluation_request = self._runtime_observation(
            request, failed_at_us, runtime_observation
        )
        binding_rows = self._materialize_runtime_bindings(
            request, model, bindings
        )
        failure = (
            RuntimeExecutionFailure("unspecified", True, False)
            if physical_failure is None
            else physical_failure
        )
        if not isinstance(failure, RuntimeExecutionFailure):
            raise UnifiedScheduleError(
                "runtime physical failure receipt is invalid"
            )
        try:
            failed, cancelled, quarantine_action = (
                self._runtime_controller.fail(
                    request_id,
                    failed_at_us,
                    reason,
                    cancel_owner=self.cancel,
                    release_memory=self._runtime_memory.release_owner,
                    failed_resource_ids=failure.failed_resource_ids,
                )
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
        if not failure.fallback_allowed:
            self._append_runtime_log(
                "FAILED", failed, failed_at_us, failed.dispatch_state
            )
            return RuntimeFailureRecovery(
                failed_ticket_id=failed.ticket_id,
                reason=reason,
                cancelled_tokens=cancelled,
                quarantine_action=quarantine_action,
                fallback=None,
            )
        registry = self._register_runtime_executors(binding_rows)
        estimates = self._estimate_registered_runtime_costs(
            evaluation_request,
            model,
            registry,
            snapshot=snapshot,
            now_us=failed_at_us,
        )
        decision = self._schedule_runtime_fallback(
            evaluation_request,
            estimates,
            runtime_now_us=failed_at_us,
            marginal_system_context=marginal_system_context,
        )
        binding = self._selected_runtime_binding(estimates, decision)
        try:
            fallback = self._runtime_controller.admit(
                request=request,
                model=model,
                runtime_observation=observation,
                estimates=estimates,
                decision=decision,
                binding=binding,
                executor_bindings=tuple(registry.bindings.values()),
                online_receipt=None,
                admitted_at_us=failed_at_us,
                previous_ticket_id=failed.ticket_id,
                failure_reason=reason,
            )
        except RuntimeControllerError as exc:
            self.cancel(decision.request_id, 0)
            raise UnifiedScheduleError(str(exc)) from exc
        recovery = RuntimeFailureRecovery(
            failed_ticket_id=failed.ticket_id,
            reason=reason,
            cancelled_tokens=cancelled,
            quarantine_action=quarantine_action,
            fallback=fallback,
        )
        self._append_runtime_log(
            "FALLBACK", fallback, failed_at_us, fallback.dispatch_state
        )
        return recovery

    def _extend_runtime_leases(
        self,
        tokens: Sequence[str],
        reserved_until_us: int,
        cancelled_owner_ids: Sequence[str],
        cancellation_at_us: int,
    ) -> tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]]:
        try:
            return self.timeline.extend_many(
                tokens,
                reserved_until_us,
                cancelled_owner_ids=cancelled_owner_ids,
                cancellation_at_us=cancellation_at_us,
            )
        except SchedulerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def _handoff_decode_cohort_singleton(
        self, cohort_id: str
    ) -> str | None:
        """Atomically move an in-flight shared lease to its sole survivor."""
        survivor_id = (
            self._runtime_decode_cohorts.pending_singleton_handoff(
                cohort_id
            )
        )
        if survivor_id is None:
            return None
        binding = self._runtime_decode_cohorts.binding_by_id(cohort_id)
        self.timeline.reassign_owner(
            binding.shared_lease_tokens,
            expected_owner_id=cohort_id,
            owner_id=survivor_id,
        )
        leases = self._runtime_decode_cohorts\
            .transfer_singleton_ownership(cohort_id, survivor_id)
        self._runtime_controller.transfer_decode_cohort_singleton(
            survivor_id, binding, leases
        )
        return survivor_id

    @_runtime_serialized
    def extend_runtime_request(
        self,
        request_id: str,
        *,
        at_us: int,
        reserved_until_us: int,
    ) -> RuntimeLeaseExtensionReceipt:
        with self._transaction():
            cohort = self._runtime_decode_cohorts.binding(request_id)
            receipt = self._runtime_controller.extend(
                request_id,
                at_us=at_us,
                reserved_until_us=reserved_until_us,
                extend_leases=self._extend_runtime_leases,
                release_memory=self._runtime_memory.release_owner,
                retime_leases=self.timeline.retime_many,
            )
            binding = self._model_placement_controller.request_binding(
                request_id
            )
            attachment = (
                None if binding is None
                else binding.get("helper_attachment")
            )
            if (
                isinstance(attachment, Mapping)
                and int(binding.get("fraction_ppm", 0)) > 0
            ):
                helper_tokens = tuple(
                    str(value)
                    for value in attachment.get("lease_tokens", ())
                )
                helper_horizon = attachment.get(
                    "lease_reserved_until_us"
                )
                if (
                    not helper_tokens
                    or type(helper_horizon) is not int
                    or helper_horizon < 1
                ):
                    raise UnifiedScheduleError(
                        "active request helper lease state is invalid"
                    )
                helper_target_us = self._helper_window_lease_horizon(
                    request_id,
                    requested_fraction_ppm=int(
                        binding.get("fraction_ppm", 0)
                    ),
                    at_us=at_us,
                )
                if helper_horizon < helper_target_us:
                    _previous, cancelled = self._extend_runtime_leases(
                        helper_tokens,
                        helper_target_us,
                        (),
                        0,
                    )
                    if cancelled:
                        raise UnifiedScheduleError(
                            "helper renewal cancelled an unrelated owner"
                        )
                    renew_shared_helper_leases(self, helper_tokens, helper_target_us, at_us)
            if receipt.extended_leases:
                self._runtime_memory.extend_owner(
                    request_id, reserved_until_us
                )
            if cohort is not None:
                ticket = self.runtime_ticket(request_id)
                leases = tuple(
                    replace(
                        row,
                        reserved_until_us=(
                            ticket.final_reserved_until_us[row.token]
                        ),
                    )
                    for row in ticket.decision.leases
                )
                cohort = self._runtime_decode_cohorts.update_shared_leases(
                    cohort.cohort_id, leases
                )
                if (
                    self._runtime_decode_cohorts.lease_owner_id(
                        cohort.cohort_id
                    ) == cohort.cohort_id
                ):
                    self._runtime_controller.extend_decode_cohort(
                        cohort, leases
                    )
            return receipt

    @_runtime_serialized
    def _runtime_lease_renewal_horizon(
        self, ticket: RuntimeRequestTicket
    ) -> int:
        """Return the earliest live base or attached-helper lease horizon."""

        current = self.runtime_ticket(ticket.request.request_id)
        if (
            current.ticket_id != ticket.ticket_id
            or current.dispatch_state != "ACQUIRED"
        ):
            raise UnifiedScheduleError(
                "runtime renewal ticket identity changed"
            )
        base_horizon = min(current.final_reserved_until_us[row.token] for row in current.live_leases)
        binding = self._model_placement_controller.request_binding(
            ticket.request.request_id
        )
        attachment = (
            None if binding is None else binding.get("helper_attachment")
        )
        if not isinstance(attachment, Mapping):
            return base_horizon
        tokens = tuple(
            str(value) for value in attachment.get("lease_tokens", ())
        )
        raw_horizon = attachment.get("lease_reserved_until_us")
        if not tokens:
            if raw_horizon is not None:
                raise UnifiedScheduleError(
                    "request helper lease horizon lacks tokens"
                )
            return base_horizon
        if type(raw_horizon) is not int or raw_horizon < 1:
            raise UnifiedScheduleError(
                "request helper lease horizon is invalid"
            )
        return min(base_horizon, raw_horizon)

    @_runtime_serialized
    def _expire_request_helper_authorization(
        self, ticket: RuntimeRequestTicket, at_us: int
    ) -> bool:
        """Detach assistance whose helper lease horizon expired unrenewed.

        Idle helper leases (fraction 0) are released; active assistance is
        told to return to the desktop at its next boundary, after which the
        idle path releases the leases. False when there is nothing attached,
        so the coordinator surfaces the expiry as a failure instead.
        """
        request_id = ticket.request.request_id
        binding = self._model_placement_controller.request_binding(request_id)
        attachment = None if binding is None else binding.get("helper_attachment")
        if not isinstance(attachment, Mapping) or not attachment.get("lease_tokens"):
            return False
        fraction_ppm = int(binding.get("fraction_ppm", 0))
        payload = {
            "fraction_ppm": fraction_ppm,
            "lease_reserved_until_us": attachment.get("lease_reserved_until_us"),
            "lease_tokens": [str(value) for value in attachment.get("lease_tokens", ())],
        }
        if fraction_ppm == 0:
            self._release_request_helper_leases(request_id, at_us)
            reason = "IDLE_HELPER_LEASES_RELEASED"
        else:
            try:
                self._adaptive_decode.helper_unavailable(request_id)
            except AdaptiveDecodeError:
                pass
            reason = "ASSISTANCE_DETACHING_AT_NEXT_BOUNDARY"
        self._model_placement_controller.record_request_helper_event(
            request_id, "HELPER_AUTHORIZATION_EXPIRED", at_us,
            {**payload, "reason": reason},
        )
        return True

    @_runtime_serialized
    def start_runtime_lease_renewal(
        self,
        ticket: RuntimeRequestTicket,
        *,
        epoch_ns: int,
        guard_us: int,
        quantum_us: int,
        on_renewal: Callable[[RuntimeLeaseExtensionReceipt], None]
            | None = None,
    ) -> None:
        if not isinstance(ticket, RuntimeRequestTicket):
            raise UnifiedScheduleError("runtime renewal ticket is invalid")
        current = self.runtime_ticket(ticket.request.request_id)
        if current != ticket or current.dispatch_state != "ACQUIRED":
            raise UnifiedScheduleError(
                "runtime renewal requires the active ticket"
            )
        if ticket.request.request_id in self._runtime_renewals:
            raise UnifiedScheduleError("runtime renewal already exists")
        try:
            coordinator = RuntimeLeaseRenewalCoordinator(
                ticket,
                epoch_ns=epoch_ns,
                guard_us=guard_us,
                quantum_us=quantum_us,
                current_ticket=self.runtime_ticket,
                extend=self.extend_runtime_request,
                on_renewal=on_renewal,
                current_horizon=self._runtime_lease_renewal_horizon,
                on_expired=self._expire_request_helper_authorization,
            )
            self._runtime_renewals[ticket.request.request_id] = coordinator
            coordinator.start()
        except RuntimeExecutionCoordinatorError as exc:
            self._runtime_renewals.pop(ticket.request.request_id, None)
            raise UnifiedScheduleError(str(exc)) from exc

    def check_runtime_lease_renewal(self, request_id: str) -> None:
        coordinator = self._runtime_renewals.get(request_id)
        if coordinator is None:
            raise UnifiedScheduleError("runtime renewal is absent")
        coordinator.check()

    def stop_runtime_lease_renewal(self, request_id: str) -> None:
        coordinator = self._runtime_renewals.get(request_id)
        if coordinator is None:
            raise UnifiedScheduleError("runtime renewal is absent")
        try:
            coordinator.stop()
        finally:
            self._runtime_renewals.pop(request_id, None)

    def runtime_route_configured(
        self, workload_id: str, route_id: str
    ) -> bool:
        profile = self.profile_for(workload_id)
        return any(
            route.workload_id == workload_id and route.route_id == route_id
            for route in profile.routes
        )

    def _extend_external_runtime_leases(
        self,
        owner_id: str,
        leases: Sequence[LeaseRecord],
        at_us: int,
        reserved_until_us: int,
    ) -> tuple[str, ...]:
        try:
            return self._runtime_controller.extend_external(
                owner_id,
                leases,
                at_us=at_us,
                reserved_until_us=reserved_until_us,
                extend_leases=self._extend_runtime_leases,
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def observe_runtime_phase(
        self,
        observation: RuntimePhaseObservation,
        *,
        observed_at_us: int,
        renewal_us: int = 5_000_000,
    ) -> Mapping[str, object]:
        if not isinstance(observation, RuntimePhaseObservation):
            raise UnifiedScheduleError(
                "runtime phase observation is invalid"
            )
        controller = self._runtime_phase_controllers.get(
            observation.source_id
        )
        try:
            if controller is None:
                controller = RuntimePhaseLeaseController(
                    observation.source_id,
                    tuple(sorted(self._resource_ids)),
                    renewal_us,
                )
                self._runtime_phase_controllers[
                    observation.source_id
                ] = controller
            elif controller.renewal_us != renewal_us:
                raise RuntimePhaseError(
                    "runtime phase renewal interval changed"
                )
            return MappingProxyType(dict(controller.observe(
                observation,
                observed_at_us,
                reserve=self.reserve_external_resources,
                extend=self._extend_external_runtime_leases,
                release=self.release,
            )))
        except RuntimePhaseError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def finish_runtime_phase(self, source_id: str, at_us: int) -> None:
        controller = self._runtime_phase_controllers.get(source_id)
        if controller is None:
            raise UnifiedScheduleError("runtime phase source is absent")
        try:
            controller.finish(
                at_us,
                extend=self._extend_external_runtime_leases,
                release=self.release,
            )
        except RuntimePhaseError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def runtime_phase_history(
        self, source_id: str
    ) -> tuple[Mapping[str, object], ...]:
        controller = self._runtime_phase_controllers.get(source_id)
        if controller is None:
            raise UnifiedScheduleError("runtime phase source is absent")
        return tuple(
            MappingProxyType(copy.deepcopy(row))
            for row in controller.history
        )

    @_runtime_serialized
    def complete_runtime_request(
        self, request_id: str, actual_end_us: int
    ) -> RuntimeCompletionReceipt:
        return self._complete_runtime_request(
            request_id, actual_end_us, execution_receipt=None
        )

    @_runtime_serialized
    def release_automated_runtime_capacity(
        self,
        request_id: str,
        actual_end_us: int,
        *,
        expected_ticket_id: str,
    ) -> RuntimeCompletionReceipt:
        """Release execution capacity before measurement finalization."""

        with self._transaction():
            with self._runtime_controller.defer_dispatch_wake():
                ticket = self.runtime_ticket(request_id)
                if ticket.ticket_id != expected_ticket_id:
                    raise UnifiedScheduleError(
                        "runtime capacity release ticket changed"
                    )
                release_shared, cohort_id = (
                    self._runtime_decode_cohorts.mark_capacity_released(
                        request_id
                    )
                )

                def release_lease(token: str, end_us: int) -> None:
                    if cohort_id is None or release_shared:
                        self.release(token, end_us)

                return self._runtime_controller.release_capacity(
                    request_id,
                    actual_end_us,
                    resource_capacities={
                        resource_id: resource.capacity
                        for resource_id, resource in (
                            self._runtime_capabilities.resources.items()
                        )
                    },
                    release_lease=release_lease,
                    cancel_owner=self.cancel,
                    release_memory=self._runtime_memory.release_owner,
                )

    @_runtime_serialized
    def complete_automated_request(
        self,
        request_id: str,
        execution_receipt: RuntimeExecutionReceipt,
        *,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ] | None = None,
    ) -> RuntimeCompletionReceipt:
        """Complete an automated plan using exact physical execution proof."""
        if not isinstance(execution_receipt, RuntimeExecutionReceipt):
            raise UnifiedScheduleError(
                "runtime physical execution receipt is invalid"
            )
        return self._complete_runtime_request(
            request_id,
            execution_receipt.finished_us,
            execution_receipt=execution_receipt,
            snapshot_provider=snapshot_provider,
        )

    def _complete_runtime_request(
        self,
        request_id: str,
        actual_end_us: int,
        *,
        execution_receipt: RuntimeExecutionReceipt | None,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ] | None = None,
    ) -> RuntimeCompletionReceipt:
        with self._transaction():
            with self._runtime_controller.defer_dispatch_wake():
                return self._complete_runtime_request_deferred(
                    request_id,
                    actual_end_us,
                    execution_receipt=execution_receipt,
                    snapshot_provider=snapshot_provider,
                )

    def _complete_runtime_request_deferred(
        self,
        request_id: str,
        actual_end_us: int,
        *,
        execution_receipt: RuntimeExecutionReceipt | None,
        snapshot_provider: Callable[
            [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
        ] | None,
    ) -> RuntimeCompletionReceipt:
        learning_generation_before = (
            None
            if execution_receipt is None
            else self._runtime_learning_generation_sha256()
        )
        release_shared, cohort_owner_id = (
            self._runtime_decode_cohorts.mark_terminal(request_id)
        )
        if cohort_owner_id is not None and not release_shared:
            self._handoff_decode_cohort_singleton(cohort_owner_id)

        def release_lease(token: str, end_us: int) -> None:
            if cohort_owner_id is None or release_shared:
                self.release(token, end_us)

        receipt = self._runtime_controller.complete(
            request_id,
            actual_end_us,
            resource_capacities=(
                None
                if self._runtime_capabilities is None
                else {
                    resource_id: resource.capacity
                    for resource_id, resource in (
                        self._runtime_capabilities.resources.items()
                    )
                }
            ),
            release_lease=release_lease,
            cancel_owner=self.cancel,
            release_memory=self._runtime_memory.release_owner,
            execution_receipt=execution_receipt,
        )
        if self._runtime_executor_registry is not None:
            self._register_runtime_executors(tuple(
                self._runtime_executor_registry.bindings.values()
            ))
        ticket = self.runtime_ticket(request_id)
        if execution_receipt is not None:
            if ticket.execution_plan is None:
                raise UnifiedScheduleError(
                    "automated completion lacks an execution plan"
                )
            selected_estimate = next(
                row for row in ticket.cost_estimates.estimates
                if row.route_id == ticket.decision.route_id
            )
            component = selected_estimate.details.get(
                "cost_breakdown", {}
            )
            if ticket.decode_cohort is None:
                if ticket.transition_receipts:
                    self._automated_compiler().record_transition_observations(
                        self.runtime_model_manifest(ticket.model.model_id),
                        ticket.execution_plan,
                        ticket.transition_receipts,
                        ticket.binding.executor_id,
                    )
                self._automated_compiler().record_execution_observation(
                    ticket.request,
                    self.runtime_model_manifest(ticket.model.model_id),
                    ticket.execution_plan,
                    execution_receipt,
                    ticket.runtime_observation.cost_features,
                    int(component["component_service_us"]),
                    (
                        None
                        if component["component_energy_uj"] is None
                        else int(component["component_energy_uj"])
                    ),
                    ticket.binding.executor_id,
                )
            elif release_shared:
                cohort_receipt = self._runtime_decode_cohorts.receipt(
                    ticket.decode_cohort.cohort_id
                )
                if cohort_receipt is None:
                    raise UnifiedScheduleError(
                        "decode cohort completion lacks its measurement"
                    )
                self._record_decode_cohort_observation(
                    cohort_receipt
                )
            assert ticket.execution_plan is not None
            if self._runtime_capabilities is not None:
                self._runtime_residency_cohorts.confirm_resident_component(
                    runtime_residency_component_identity(
                        ticket.model.artifact_sha256,
                        ticket.execution_plan,
                        ticket.binding,
                    ),
                    self._runtime_capabilities,
                )
        self._runtime_residency_cohorts.record_terminal(request_id)
        self._record_phone_route_use(ticket)
        self._close_request_helper_runtime(
            request_id, actual_end_us, "REQUEST_COMPLETED"
        )
        self._model_placement_controller.release_request(request_id)
        self._model_placement_controller.notify(
            ticket.model.artifact_sha256,
            "REQUEST_COMPLETED",
            actual_end_us,
        )
        if self._phone_reprovisioning is not None:
            self._reevaluate_phone_layout_after_release(ticket, actual_end_us)
        if (
            learning_generation_before is not None
            and ticket.selection_mode != "desktop-baseline"
            and learning_generation_before
                != self._runtime_learning_generation_sha256()
        ):
            self._refresh_model_placement_epochs_after_learning(
                receipt.completion_event_replans,
                actual_end_us,
                snapshot_provider,
            )
        self._append_runtime_log(
            "COMPLETED", ticket, actual_end_us, ticket.dispatch_state
        )
        return receipt

    def _record_decode_cohort_observation(
        self, receipt: RuntimeDecodeCohortReceipt
    ) -> None:
        """Ingest one aggregate observation after every member completes."""
        tickets = tuple(
            self.runtime_ticket(request_id)
            for request_id in receipt.member_request_ids
        )
        if any(
            ticket.dispatch_state != "COMPLETED"
            or ticket.execution_plan is None
            or ticket.execution_receipt is None
            or ticket.decode_cohort is None
            or ticket.decode_cohort.cohort_id != receipt.cohort_id
            for ticket in tickets
        ):
            raise UnifiedScheduleError(
                "decode cohort aggregate precedes member completion"
            )
        leader = next(
            ticket for ticket in tickets
            if ticket.request.request_id == receipt.energy_owner_request_id
        )
        if any(
            ticket.model.artifact_sha256
                != leader.model.artifact_sha256
            or ticket.execution_plan.desktop_placement_sha256
                != leader.execution_plan.desktop_placement_sha256
            for ticket in tickets
        ):
            raise UnifiedScheduleError(
                "decode cohort aggregate placement differs"
            )
        components = tuple(
            next(
                row for row in ticket.cost_estimates.estimates
                if row.route_id == ticket.decision.route_id
            ).details.get("cost_breakdown", {})
            for ticket in tickets
        )
        component_service_us = sum(
            int(row["component_service_us"]) for row in components
        )
        energy_rows = tuple(
            row.get("component_energy_uj") for row in components
        )
        component_energy_uj = (
            None
            if any(value is None for value in energy_rows)
            else sum(int(value) for value in energy_rows)
        )
        request = replace(
            leader.request,
            input_tokens=receipt.total_input_tokens,
            output_tokens=receipt.total_output_tokens,
            features={
                **dict(leader.request.features),
                "actual_batch_size": receipt.active_batch,
            },
        )
        cost_features = {
            **dict(leader.runtime_observation.cost_features),
            "actual_batch_size": receipt.active_batch,
        }
        execution = RuntimeExecutionReceipt(
            ticket_id=(
                leader.ticket_id + ":cohort:" + receipt.receipt_sha256[7:23]
            ),
            request_id=leader.request.request_id,
            artifact_sha256=leader.model.artifact_sha256,
            operator_plan_sha256=leader.execution_plan.plan_sha256,
            executor_id=leader.binding.executor_id,
            endpoint=leader.binding.endpoint,
            operator_plan_protocol=(
                leader.binding.operator_plan_protocol or "cohort-v1"
            ),
            participant_executor_ids=(
                leader.execution_receipt.participant_executor_ids
            ),
            started_us=receipt.started_at_us,
            finished_us=receipt.finished_at_us,
            output_sha256=canonical_sha256({
                "cohort_id": receipt.cohort_id,
                "member_outputs": [
                    ticket.execution_receipt.output_sha256
                    for ticket in tickets
                ],
            }),
            status="COMPLETED",
            energy_boundary_id=receipt.energy_boundary_id,
            fleet_energy_uj_by_domain=(
                receipt.fleet_energy_uj_by_domain
            ),
            transfer_energy_uj_by_link=(
                receipt.transfer_energy_uj_by_link
            ),
            measurement_evidence_ids=(
                receipt.measurement_evidence_ids
            ),
            energy_attribution_kind=receipt.attribution_kind,
            energy_estimation_metadata=(
                receipt.energy_estimation_metadata
            ),
        )
        self._automated_compiler().record_execution_observation(
            request,
            self.runtime_model_manifest(leader.model.model_id),
            leader.execution_plan,
            execution,
            cost_features,
            component_service_us,
            component_energy_uj,
            leader.binding.executor_id,
        )
        self._runtime_decode_cohorts.mark_estimator_ingested(
            receipt.cohort_id
        )

    @_runtime_serialized
    def cancel_runtime_request(
        self, request_id: str, at_us: int, reason: str
    ) -> tuple[str, ...]:
        with self._transaction():
            release_shared, cohort_owner_id = (
                self._runtime_decode_cohorts.mark_terminal(request_id)
            )
            if cohort_owner_id is not None and not release_shared:
                self._handoff_decode_cohort_singleton(cohort_owner_id)
            lease_owner_id = (
                None
                if cohort_owner_id is None
                else self._runtime_decode_cohorts.lease_owner_id(
                    cohort_owner_id
                )
            )

            def cancel_owner(owner_id: str, at_us: int) -> tuple[str, ...]:
                if owner_id == request_id and cohort_owner_id is not None:
                    if not release_shared:
                        return ()
                    assert lease_owner_id is not None
                    return self.cancel(lease_owner_id, at_us)
                return self.cancel(owner_id, at_us)

            cancelled = self._runtime_controller.cancel(
                request_id,
                at_us,
                reason,
                cancel_owner=cancel_owner,
                release_memory=self._runtime_memory.release_owner,
            )
            ticket = self.runtime_ticket(request_id)
            self._fail_phone_residency_transition(
                ticket.ticket_id, at_us, reason
            )
            self._runtime_residency_cohorts.record_terminal(request_id)
            self._close_request_helper_runtime(
                request_id, at_us, "REQUEST_CANCELLED"
            )
            self._model_placement_controller.release_request(request_id)
            self._model_placement_controller.notify(
                ticket.model.artifact_sha256,
                "REQUEST_CANCELLED",
                at_us,
            )
            self._append_runtime_log(
                "CANCELLED", ticket, at_us, ticket.dispatch_state
            )
            return cancelled

    @_runtime_serialized
    def cancel_runtime_request_if_pending(
        self, request_id: str, at_us: int, reason: str
    ) -> tuple[str, ...]:
        """Cancel pending work and audit an already terminal cleanup race."""
        current = self.runtime_ticket(request_id)
        if current.dispatch_state not in {
            "CANCELLED", "COMPLETED", "FAILED"
        }:
            return self.cancel_runtime_request(request_id, at_us, reason)
        try:
            return self._runtime_controller.cancel(
                request_id,
                at_us,
                reason,
                cancel_owner=self.cancel,
                release_memory=self._runtime_memory.release_owner,
                terminal_is_noop=True,
            )
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    @_runtime_serialized
    def configure_runtime_dispatch_policy(
        self, policy: RuntimeDispatchPolicy
    ) -> None:
        """Select work-conserving admission and model affinity before use."""
        try:
            self._runtime_controller.configure_dispatch_policy(policy)
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc

    def runtime_dispatch_policy_state(self) -> Mapping[str, object]:
        return self._runtime_controller.dispatch_policy_state()

    def configure_joint_planner_shadow(self, shadow: object) -> None:
        """Opt-in observer of every journal record (``dispatch_policy.joint_planner``); read-only."""
        self._joint_planner_shadow = shadow

    def joint_planner_shadow(self) -> object | None:
        return getattr(self, "_joint_planner_shadow", None)

    def configure_joint_planner_active(self, active: object) -> None:
        """Opt-in planner that decides at every journal record and executes the joint join
        (``dispatch_policy.joint_planner`` mode active, ``_internal.joint_planner_active``)."""
        self._joint_planner_active = active

    def joint_planner_active(self) -> object | None:
        return getattr(self, "_joint_planner_active", None)

    def runtime_queued_start_us(self) -> int | None:
        """Earliest planned start of a QUEUED runtime ticket, None when nothing is queued
        (device power control: a queued ticket keeps the GPU clocks restored)."""
        return min(
            (ticket.decision.start_us for ticket in self._runtime_controller.current_tickets(("QUEUED",))),
            default=None,
        )

    def runtime_controller_snapshot(self) -> Mapping[str, object]:
        state = self._runtime_controller.snapshot()
        state["executor_registry"] = (
            None
            if self._runtime_executor_registry is None
            else self._runtime_executor_registry.to_json()
        )
        return MappingProxyType(state)

    def runtime_route_is_quarantined(self, route_id: str) -> bool:
        try:
            return self._runtime_controller.route_is_quarantined(route_id)
        except RuntimeControllerError as exc:
            raise UnifiedScheduleError(str(exc)) from exc
