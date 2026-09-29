"""Automated request lifecycle: commit, submit, replan, fail, runtime snapshots and receipts.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.policy import Decision, LeasePreview, Request
from .._internal.runtime_cost import RuntimeCostEstimateSet
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.route_generation import RuntimeRouteTemplateSet
from .._internal.model_placement_controller import RequestHelperEnvelopeBinding
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    AutomatedRouteCandidate,
    RuntimeTransitionReceipt,
)
from .._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from .._internal.runtime_residency_cohorts import RuntimeModelPlacementEpoch
from .._internal.runtime_controller import RuntimeFailureRecovery, RuntimeRequestTicket
from .._internal.runtime_execution import RuntimeExecutionFailure
from .common import _runtime_serialized

from .automated_requests_ops.common import (
    _AutomatedReplanPreparation as _AutomatedReplanPreparation,
    _AutomatedSubmitContext as _AutomatedSubmitContext,
    _AutomatedSubmitResolution as _AutomatedSubmitResolution,
)
from .automated_requests_ops import commit as _commit
from .automated_requests_ops import failure as _failure
from .automated_requests_ops import join_publication as _join_publication
from .automated_requests_ops import observations as _observations
from .automated_requests_ops import replan as _replan
from .automated_requests_ops import replan_commit as _replan_commit
from .automated_requests_ops import replan_selection as _replan_selection
from .automated_requests_ops import selection as _selection
from .automated_requests_ops import admission as _admission
from .automated_requests_ops import submission as _submission


class AutomatedRequestMixin:
    """Automated request lifecycle: commit, submit, replan, fail, runtime snapshots and receipts."""

    @_runtime_serialized
    def submit_startup_parent_preload(
        self, request: Request, model_id: str, snapshot: HeterogeneousRuntimeSnapshot,
        *, executor_id: str, desktop_placement_sha256: str, observed_at_us: int,
    ) -> RuntimeRequestTicket:
        """Admit a bounded startup verification, without publishing a placement epoch."""
        return _submission.submit_startup_parent_preload(
            self, request, model_id, snapshot, executor_id=executor_id,
            desktop_placement_sha256=desktop_placement_sha256, observed_at_us=observed_at_us,
        )

    def _commit_decode_cohort_leases(
        self,
        request: Request,
        selected: AutomatedRouteCandidate,
        preview: LeasePreview,
        observed_at_us: int,
    ) -> tuple[tuple, object | None, LeasePreview]:
        return _commit._commit_decode_cohort_leases(
            self,
            request,
            selected,
            preview,
            observed_at_us,
        )

    def _reset_previous_phone_layout_transition(
        self, previous_ticket_id: str | None, observed_at_us: int
    ) -> None:
        return _commit._reset_previous_phone_layout_transition(
            self,
            previous_ticket_id,
            observed_at_us,
        )

    def _selected_phone_layout_generation(
        self,
        selected: AutomatedRouteCandidate,
        selection_mode: str,
        placement_epoch: RuntimeModelPlacementEpoch | None,
        bound_placement_epoch: RuntimeModelPlacementEpoch | None,
    ) -> int | None:
        return _commit._selected_phone_layout_generation(
            self,
            selected,
            selection_mode,
            placement_epoch,
            bound_placement_epoch,
        )

    def _automated_decision(
        self,
        *,
        request: Request,
        selected: AutomatedRouteCandidate,
        preview: LeasePreview,
        leases: tuple,
        reason: str,
        rejected: tuple[tuple[str, str], ...],
        observed_at_us: int,
        prediction_finish_upper_us: int,
    ) -> Decision:
        return _commit._automated_decision(
            self,
            request=request,
            selected=selected,
            preview=preview,
            leases=leases,
            reason=reason,
            rejected=rejected,
            observed_at_us=observed_at_us,
            prediction_finish_upper_us=prediction_finish_upper_us,
        )

    @staticmethod
    def _automated_executor_bindings(
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        estimates: RuntimeCostEstimateSet,
        selection_mode: str,
    ) -> tuple[RuntimeCostEstimateSet, tuple]:
        return _commit._automated_executor_bindings(
            candidate_set,
            selected,
            estimates,
            selection_mode,
        )

    def _admit_automated_ticket(
        self,
        *,
        request: Request,
        snapshot: HeterogeneousRuntimeSnapshot,
        estimates: RuntimeCostEstimateSet,
        decision: Decision,
        selected: AutomatedRouteCandidate,
        bindings: tuple,
        observed_at_us: int,
        memory: tuple,
        previous_ticket_id: str | None,
        failure_reason: str | None,
        previous_transition_receipts: Sequence[RuntimeTransitionReceipt],
        selection_mode: str,
        decode_cohort,
        residency_projection_token: str | None,
        phone_layout_generation: int | None,
    ) -> RuntimeRequestTicket:
        return _commit._admit_automated_ticket(
            self,
            request=request,
            snapshot=snapshot,
            estimates=estimates,
            decision=decision,
            selected=selected,
            bindings=bindings,
            observed_at_us=observed_at_us,
            memory=memory,
            previous_ticket_id=previous_ticket_id,
            failure_reason=failure_reason,
            previous_transition_receipts=previous_transition_receipts,
            selection_mode=selection_mode,
            decode_cohort=decode_cohort,
            residency_projection_token=residency_projection_token,
            phone_layout_generation=phone_layout_generation,
        )

    def _dispatched_helper_envelope_binding(
        self, selected: AutomatedRouteCandidate
    ) -> RequestHelperEnvelopeBinding | None:
        return _commit._dispatched_helper_envelope_binding(self, selected)

    def _refresh_request_helper_opportunities(
        self,
        request: Request,
        candidate_set: AutomatedCandidateSet,
        selected: AutomatedRouteCandidate,
        estimates: RuntimeCostEstimateSet,
        helper_envelope_binding: RequestHelperEnvelopeBinding | None,
    ) -> None:
        return _commit._refresh_request_helper_opportunities(
            self,
            request,
            candidate_set,
            selected,
            estimates,
            helper_envelope_binding,
        )

    @staticmethod
    def _helper_opportunity_matches_layout(
        opportunity,
        layout,
        estimates: RuntimeCostEstimateSet,
        selected: AutomatedRouteCandidate,
    ) -> bool:
        return _commit._helper_opportunity_matches_layout(opportunity, layout, estimates, selected)

    def _record_automated_residency_arrival(
        self,
        *,
        event_kind: str,
        request: Request,
        candidate_set: AutomatedCandidateSet,
        selected_component,
        estimates: RuntimeCostEstimateSet,
    ) -> None:
        return _commit._record_automated_residency_arrival(
            self,
            event_kind=event_kind,
            request=request,
            candidate_set=candidate_set,
            selected_component=selected_component,
            estimates=estimates,
        )

    def _publish_automated_placement_epoch(
        self,
        candidate_set: AutomatedCandidateSet,
        placement_epoch: RuntimeModelPlacementEpoch | None,
        route_templates: RuntimeRouteTemplateSet | None,
    ) -> None:
        return _commit._publish_automated_placement_epoch(
            self,
            candidate_set,
            placement_epoch,
            route_templates,
        )

    def _finish_automated_commit_timing(
        self,
        *,
        ticket: RuntimeRequestTicket,
        event_kind: str,
        observed_at_us: int,
        commit_started_ns: int,
        reservation_finished_ns: int,
    ) -> None:
        return _commit._finish_automated_commit_timing(
            self,
            ticket=ticket,
            event_kind=event_kind,
            observed_at_us=observed_at_us,
            commit_started_ns=commit_started_ns,
            reservation_finished_ns=reservation_finished_ns,
        )

    def _commit_automated_attempt(
        self,
        *,
        request: Request,
        snapshot: HeterogeneousRuntimeSnapshot,
        candidate_set: AutomatedCandidateSet,
        estimates: RuntimeCostEstimateSet,
        selected: AutomatedRouteCandidate,
        preview: object,
        rejected: tuple[tuple[str, str], ...],
        reason: str,
        observed_at_us: int,
        event_kind: str,
        previous_ticket_id: str | None = None,
        failure_reason: str | None = None,
        previous_transition_receipts: Sequence[
            RuntimeTransitionReceipt
        ] = (),
        selection_mode: str = "energy-aware",
        placement_epoch: RuntimeModelPlacementEpoch | None = None,
        route_templates: RuntimeRouteTemplateSet | None = None,
        bound_placement_epoch: RuntimeModelPlacementEpoch | None = None,
    ) -> RuntimeRequestTicket:
        return _commit._commit_automated_attempt(
            self,
            request=request,
            snapshot=snapshot,
            candidate_set=candidate_set,
            estimates=estimates,
            selected=selected,
            preview=preview,
            rejected=rejected,
            reason=reason,
            observed_at_us=observed_at_us,
            event_kind=event_kind,
            previous_ticket_id=previous_ticket_id,
            failure_reason=failure_reason,
            previous_transition_receipts=previous_transition_receipts,
            selection_mode=selection_mode,
            placement_epoch=placement_epoch,
            route_templates=route_templates,
            bound_placement_epoch=bound_placement_epoch,
        )

    def _detach_decode_cohort_for_replan(
        self, request_id: str
    ) -> None:
        return _commit._detach_decode_cohort_for_replan(self, request_id)

    def request_shape_support(self, request: Request, model_id: str):
        """Judge statically whether a request shape can ever be served."""
        return _admission.request_shape_support(self, request, model_id)

    def _require_supported_request_shape(
        self, request: Request, model_id: str
    ):
        return _admission.require_supported_request_shape(
            self, request, model_id
        )

    @_runtime_serialized
    def submit_automated_request(
        self,
        request: Request,
        model_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int | None = None,
        selection_mode: str = "energy-aware",
    ) -> RuntimeRequestTicket:
        """Repair causal predecessors, then admit one observed arrival."""
        ticket = _submission.submit_automated_request(
            self,
            request,
            model_id,
            snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
        )
        if self._phone_reprovisioning is not None:
            self._reevaluate_phone_layout_for_decided_transition(
                ticket, request.arrival_us if observed_at_us is None else observed_at_us,
            )
        return ticket

    def _repair_stale_projection_chain(
        self,
        error: RuntimeResidencyProjectionError,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> Mapping[str, int]:
        """Defer stale projected work until its causal state is current."""
        return _submission._repair_stale_projection_chain(self, error, snapshot, observed_at_us)

    def _prepare_automated_submit_context(
        self,
        *,
        request: Request,
        manifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        selection_mode: str,
        causal_not_before_by_resource: Mapping[str, int],
    ) -> _AutomatedSubmitContext:
        return _selection._prepare_automated_submit_context(
            self,
            request=request,
            manifest=manifest,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            causal_not_before_by_resource=causal_not_before_by_resource,
        )

    def _submit_candidate_set(
        self,
        *,
        context: _AutomatedSubmitContext,
        request: Request,
        manifest,
        compiler,
        observed_at_us: int,
        selection_mode: str,
    ) -> AutomatedCandidateSet:
        return _selection._submit_candidate_set(
            self,
            context=context,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
        )

    def _select_existing_epoch_candidate(
        self,
        *,
        context: _AutomatedSubmitContext,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest,
        compiler,
        observed_at_us: int,
        selection_mode: str,
        memory_rejections: Mapping[str, tuple[str, ...]],
    ) -> tuple:
        return _selection._select_existing_epoch_candidate(
            self,
            context=context,
            candidate_set=candidate_set,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            memory_rejections=memory_rejections,
        )

    def _submit_candidate_selection(
        self,
        *,
        context: _AutomatedSubmitContext,
        candidate_set: AutomatedCandidateSet,
        request: Request,
        manifest,
        compiler,
        observed_at_us: int,
        selection_mode: str,
        memory_rejections: Mapping[str, tuple[str, ...]],
    ) -> tuple:
        return _selection._submit_candidate_selection(
            self,
            context=context,
            candidate_set=candidate_set,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            memory_rejections=memory_rejections,
        )

    def _project_submit_selection(
        self,
        *,
        context: _AutomatedSubmitContext,
        request: Request,
        selected: AutomatedRouteCandidate,
        observed_at_us: int,
    ) -> tuple[LeasePreview, bool]:
        return _selection._project_submit_selection(
            self,
            context=context,
            request=request,
            selected=selected,
            observed_at_us=observed_at_us,
        )

    def _resolve_automated_submit(
        self,
        *,
        context: _AutomatedSubmitContext,
        request: Request,
        manifest,
        compiler,
        observed_at_us: int,
        selection_mode: str,
    ) -> _AutomatedSubmitResolution:
        return _selection._resolve_automated_submit(
            self,
            context=context,
            request=request,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
        )

    def _convert_submit_candidate_set(
        self,
        context: _AutomatedSubmitContext,
        resolution: _AutomatedSubmitResolution,
        request: Request,
        manifest,
    ) -> RuntimeCostEstimateSet:
        return _selection._convert_submit_candidate_set(
            self,
            context,
            resolution,
            request,
            manifest,
        )

    def _commit_submitted_automated_request(
        self,
        *,
        submit_started_ns: int,
        context: _AutomatedSubmitContext,
        resolution: _AutomatedSubmitResolution,
        estimates: RuntimeCostEstimateSet,
        request: Request,
        selection_mode: str,
        compiler,
        observed_at_us: int,
    ) -> RuntimeRequestTicket:
        return _selection._commit_submitted_automated_request(
            self,
            submit_started_ns=submit_started_ns,
            context=context,
            resolution=resolution,
            estimates=estimates,
            request=request,
            selection_mode=selection_mode,
            compiler=compiler,
            observed_at_us=observed_at_us,
        )

    def _submit_automated_request_once(
        self,
        request: Request,
        model_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int | None = None,
        selection_mode: str = "energy-aware",
        causal_not_before_by_resource: Mapping[str, int] = MappingProxyType({}),
    ) -> RuntimeRequestTicket:
        """Discover, cost, reserve, bind, journal, and queue one request."""
        return _selection._submit_automated_request_once(
            self,
            request,
            model_id,
            snapshot,
            observed_at_us=observed_at_us,
            selection_mode=selection_mode,
            causal_not_before_by_resource=causal_not_before_by_resource,
        )

    def _fail_runtime_ticket(
        self,
        *,
        current: RuntimeRequestTicket,
        manifest,
        failure: RuntimeExecutionFailure,
        failed_at_us: int,
        reason: str,
        transition_receipts: Sequence[RuntimeTransitionReceipt],
    ) -> tuple:
        return _failure._fail_runtime_ticket(
            self,
            current=current,
            manifest=manifest,
            failure=failure,
            failed_at_us=failed_at_us,
            reason=reason,
            transition_receipts=transition_receipts,
        )

    def _terminal_failure_recovery(
        self,
        *,
        current: RuntimeRequestTicket,
        manifest,
        failed,
        cancelled: tuple[str, ...],
        quarantine_action,
        failed_at_us: int,
        reason: str,
        notify_reason: str,
        recover_adaptive: bool,
    ) -> RuntimeFailureRecovery:
        return _failure._terminal_failure_recovery(
            self,
            current=current,
            manifest=manifest,
            failed=failed,
            cancelled=cancelled,
            quarantine_action=quarantine_action,
            failed_at_us=failed_at_us,
            reason=reason,
            notify_reason=notify_reason,
            recover_adaptive=recover_adaptive,
        )

    def _prepare_automated_failure_fallback(
        self,
        *,
        current: RuntimeRequestTicket,
        manifest,
        snapshot: HeterogeneousRuntimeSnapshot,
        failed,
        failed_at_us: int,
        elastic: bool = False,
        exited_executor_id: str | None = None,
        masked_executor_id: str | None = None,
    ) -> tuple:
        return _failure._prepare_automated_failure_fallback(
            self,
            current=current,
            manifest=manifest,
            snapshot=snapshot,
            failed=failed,
            failed_at_us=failed_at_us,
            elastic=elastic,
            exited_executor_id=exited_executor_id,
            **({} if masked_executor_id is None else {"masked_executor_id": masked_executor_id}),
        )

    @_runtime_serialized
    def fail_automated_request(
        self,
        request_id: str,
        *,
        failed_at_us: int,
        reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        physical_failure: RuntimeExecutionFailure | None = None,
        transition_receipts: Sequence[RuntimeTransitionReceipt] = (),
    ) -> RuntimeFailureRecovery:
        """Record a physical failure and make a new scheduler decision."""
        return _failure.fail_automated_request(
            self,
            request_id,
            failed_at_us=failed_at_us,
            reason=reason,
            snapshot=snapshot,
            physical_failure=physical_failure,
            transition_receipts=transition_receipts,
        )

    @_runtime_serialized
    def automated_recovery_unavailable_reason(self, request_id: str) -> str | None:
        """Why the last helper_lost / server_exited failure of a request found no recovery."""
        return _failure.recovery_unavailable_reason(self, request_id)

    @_runtime_serialized
    def replan_automated_request(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        expected_ticket_id: str | None = None,
        expected_queue_generation: int | None = None,
    ) -> RuntimeRequestTicket:
        """Replace one attempt, compacting later conflicting reservations."""
        ticket = _replan.replan_automated_request(
            self,
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )
        if self._phone_reprovisioning is not None:
            self._reevaluate_phone_layout_for_decided_transition(ticket, observed_at_us)
        if _join_publication.commits_residency_change(self, ticket):
            # Work that arrived during a load may now displace this change.
            with self._runtime_controller.defer_dispatch_wake(), self._transaction():
                _join_publication.wake_server_live_joiners(
                    self, snapshot, observed_at_us
                )
        return ticket

    def _replan_priority_compaction_without_followers(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        expected_ticket_id: str | None,
        expected_queue_generation: int | None,
    ) -> RuntimeRequestTicket:
        return _replan._replan_priority_compaction_without_followers(
            self,
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )

    def _replan_priority_compaction_with_followers(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        expected_ticket_id: str | None,
        expected_queue_generation: int | None,
        root: RuntimeRequestTicket,
    ) -> RuntimeRequestTicket:
        return _replan._replan_priority_compaction_with_followers(
            self,
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
            root=root,
        )

    def _validated_replan_ticket(
        self,
        request_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        expected_ticket_id: str | None,
        expected_queue_generation: int | None,
    ) -> tuple[RuntimeRequestTicket, int | None, bool]:
        return _replan._validated_replan_ticket(
            self,
            request_id,
            snapshot,
            observed_at_us,
            expected_ticket_id,
            expected_queue_generation,
        )

    def _prepare_automated_replan(
        self,
        *,
        current: RuntimeRequestTicket,
        snapshot: HeterogeneousRuntimeSnapshot,
        manifest,
        observed_at_us: int,
        reason: str,
        expected_queue_generation: int | None,
    ) -> _AutomatedReplanPreparation:
        return _replan._prepare_automated_replan(
            self,
            current=current,
            snapshot=snapshot,
            manifest=manifest,
            observed_at_us=observed_at_us,
            reason=reason,
            expected_queue_generation=expected_queue_generation,
        )

    def _replan_transition_projection_changed(
        self,
        current: RuntimeRequestTicket,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> bool:
        return _replan._replan_transition_projection_changed(self, current, snapshot)

    def _replan_candidate_set(
        self,
        *,
        preparation: _AutomatedReplanPreparation,
        current: RuntimeRequestTicket,
        manifest,
        compiler,
        observed_at_us: int,
    ) -> AutomatedCandidateSet:
        return _replan_selection._replan_candidate_set(
            self,
            preparation=preparation,
            current=current,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
        )

    def _materialize_replan_candidate(
        self,
        *,
        context: _AutomatedSubmitContext,
        candidate_set: AutomatedCandidateSet,
        prospective: AutomatedRouteCandidate,
        current: RuntimeRequestTicket,
        manifest,
        observed_at_us: int,
        memory_rejections: Mapping[str, tuple[str, ...]],
        residency_holds: Mapping[str, object],
        invalidation_reason: str,
        current_epoch: RuntimeModelPlacementEpoch | None,
        compiler=None,
    ) -> tuple:
        return _replan_selection._materialize_replan_candidate(
            self,
            context=context,
            candidate_set=candidate_set,
            prospective=prospective,
            current=current,
            manifest=manifest,
            observed_at_us=observed_at_us,
            memory_rejections=memory_rejections,
            residency_holds=residency_holds,
            invalidation_reason=invalidation_reason,
            current_epoch=current_epoch,
            compiler=compiler,
        )

    def _replan_candidate_selection(
        self,
        *,
        preparation: _AutomatedReplanPreparation,
        candidate_set: AutomatedCandidateSet,
        current: RuntimeRequestTicket,
        manifest,
        compiler,
        observed_at_us: int,
        memory_rejections: Mapping[str, tuple[str, ...]],
    ) -> tuple:
        return _replan_selection._replan_candidate_selection(
            self,
            preparation=preparation,
            candidate_set=candidate_set,
            current=current,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            memory_rejections=memory_rejections,
        )

    def _commit_replanned_attempt(
        self,
        *,
        preparation: _AutomatedReplanPreparation,
        candidate_set: AutomatedCandidateSet,
        estimates: RuntimeCostEstimateSet,
        selected: AutomatedRouteCandidate,
        preview: LeasePreview,
        rejected: tuple[tuple[str, str], ...],
        selection_reason: str,
        observed_at_us: int,
        reason: str,
        publish_epoch: RuntimeModelPlacementEpoch | None,
        publish_templates: RuntimeRouteTemplateSet | None,
        priority_compaction_active: bool,
    ) -> RuntimeRequestTicket:
        return _replan_commit._commit_replanned_attempt(
            self,
            preparation=preparation,
            candidate_set=candidate_set,
            estimates=estimates,
            selected=selected,
            preview=preview,
            rejected=rejected,
            selection_reason=selection_reason,
            observed_at_us=observed_at_us,
            reason=reason,
            publish_epoch=publish_epoch,
            publish_templates=publish_templates,
            priority_compaction_active=priority_compaction_active,
        )

    def _record_replan_timing(
        self,
        *,
        ticket: RuntimeRequestTicket,
        preparation: _AutomatedReplanPreparation,
        compiler,
        checkpoint_us: int,
        replan_started_ns: int,
    ) -> None:
        return _replan_commit._record_replan_timing(
            self,
            ticket=ticket,
            preparation=preparation,
            compiler=compiler,
            checkpoint_us=checkpoint_us,
            replan_started_ns=replan_started_ns,
        )

    def _execute_automated_replan(
        self,
        *,
        preparation: _AutomatedReplanPreparation,
        current: RuntimeRequestTicket,
        manifest,
        compiler,
        observed_at_us: int,
        reason: str,
        priority_compaction_active: bool,
        checkpoint_us: int,
        replan_started_ns: int,
        require_resident_plan: bool = False,
        commit_scope=None,
    ) -> RuntimeRequestTicket:
        return _replan_commit._execute_automated_replan(
            self,
            preparation=preparation,
            current=current,
            manifest=manifest,
            compiler=compiler,
            observed_at_us=observed_at_us,
            reason=reason,
            priority_compaction_active=priority_compaction_active,
            checkpoint_us=checkpoint_us,
            replan_started_ns=replan_started_ns,
            require_resident_plan=require_resident_plan,
            commit_scope=commit_scope,
        )

    def _handle_automated_replan_failure(
        self,
        *,
        error: BaseException,
        request_id: str,
        current: RuntimeRequestTicket,
        observed_at_us: int,
    ) -> RuntimeRequestTicket:
        return _replan_commit._handle_automated_replan_failure(
            self,
            error=error,
            request_id=request_id,
            current=current,
            observed_at_us=observed_at_us,
        )

    def _replan_automated_request_once(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        reason: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        expected_ticket_id: str | None = None,
        expected_queue_generation: int | None = None,
        priority_compaction_active: bool = False,
    ) -> RuntimeRequestTicket:
        """Replace one scheduler-marked queued attempt atomically."""
        return _replan_commit._replan_automated_request_once(
            self,
            request_id,
            observed_at_us=observed_at_us,
            reason=reason,
            snapshot=snapshot,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
            priority_compaction_active=priority_compaction_active,
        )

    def _defer_automated_replan_for_projection(
        self,
        request_id: str,
        error: RuntimeResidencyProjectionError,
        observed_at_us: int,
    ) -> RuntimeRequestTicket:
        return _replan_commit._defer_automated_replan_for_projection(
            self,
            request_id,
            error,
            observed_at_us,
        )

    def _publish_automated_replan_failure(
        self,
        request_id: str,
        observed_at_us: int,
        failure_reason: str,
    ) -> None:
        return _replan_commit._publish_automated_replan_failure(
            self,
            request_id,
            observed_at_us,
            failure_reason,
        )

    @_runtime_serialized
    def observe_automated_runtime_snapshot(
        self,
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        observed_at_us: int,
    ) -> tuple[str, ...]:
        """Wake queued attempts whose selected residency transition is obsolete."""
        return _observations.observe_automated_runtime_snapshot(
            self,
            snapshot,
            observed_at_us=observed_at_us,
        )

    def runtime_memory_state(self) -> Mapping[str, object]:
        return _observations.runtime_memory_state(self)

    @_runtime_serialized
    def record_automated_transition_receipts(
        self,
        request_id: str,
        receipts: Sequence[RuntimeTransitionReceipt],
    ) -> RuntimeRequestTicket:
        """Bind physical transition receipts to the selected attempt."""
        return _observations.record_automated_transition_receipts(self, request_id, receipts)

    def runtime_execution_ticket(
        self, request_id: str
    ) -> RuntimeRequestTicket:
        """Return the exact physical contract only when it is executable."""
        return _observations.runtime_execution_ticket(self, request_id)
