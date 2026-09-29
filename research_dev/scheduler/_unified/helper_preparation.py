"""Request helper preparation transaction: begin, complete, check, fail, leases, windows, attach.

Mixin of ``UnifiedScheduler``; methods were moved here verbatim and rely on the
state initialised in ``UnifiedScheduler.__init__``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from .._internal.policy import LeaseDemand
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_cost import RuntimeMemoryDemand
from .._internal.model_manifest import ModelManifest
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.route_generation import RouteGenerationError
from .._internal.model_placement_controller import (
    ModelPlacementControllerError,
    ModelPhoneResidencyLayout,
)
from .._internal.runtime_plan import (
    AutomatedCandidateSet,
    HelperOpportunity,
    RuntimeHelperExecutionEnvelope,
    RuntimeTransitionPlan,
    RuntimeTransitionReceipt,
)
from .._internal.runtime_resources import RuntimeResourceError
from .._internal.runtime_residency_projection import RuntimeResidencyProjectionError
from .._internal.runtime_residency_cohorts import RuntimeResidencyCohortError
from .._internal.runtime_controller import RuntimeRequestTicket
from .._internal.types import canonical_sha256
from .._internal.adaptive_decode_contracts import AdaptiveDecodeError
from .common import (
    _RECOVERABLE_ERRORS,
    _StalePhoneSessionAssignment,
    _RequestHelperPreparation,
    _RequestHelperAttachAttempt,
    _runtime_serialized,
    _text,
)


from .helper_preparation_checks import (
    _partial_phone_replacement_session_diff as _partial_phone_replacement_session_diff,
    _check_partial_phone_replacement_transitions as _check_partial_phone_replacement_transitions,
    _check_partial_phone_replacement_source_exact as _check_partial_phone_replacement_source_exact,
    _rewrite_partial_phone_replacement_demands as _rewrite_partial_phone_replacement_demands,
    _helper_preparation_lease_demands as _helper_preparation_lease_demands,
    _helper_preparation_projection_sha256 as _helper_preparation_projection_sha256,
    _check_preparation_completion_receipts as _check_preparation_completion_receipts,
    _rebind_blocker_quiescence as _rebind_blocker_quiescence,
)

from .helper_preparation_ops.common import _HelperAttachLeases as _HelperAttachLeases
from .helper_preparation_ops import arbitration as _arbitration
from .helper_preparation_ops import attachment as _attachment
from .helper_preparation_ops import authorization as _authorization
from .helper_preparation_ops import cleanup as _cleanup
from .helper_preparation_ops import completion as _completion
from .helper_preparation_ops import memory as _memory
from .helper_preparation_ops import recovery as _recovery
from .helper_preparation_ops import start as _start

__all__ = [
    'AdaptiveDecodeError',
    'AutomatedCandidateSet',
    'HelperOpportunity',
    'HelperPreparationMixin',
    'HeterogeneousRuntimeSnapshot',
    'LeaseDemand',
    'ModelManifest',
    'ModelPhoneResidencyLayout',
    'ModelPlacementControllerError',
    'RouteGenerationError',
    'RuntimeHelperExecutionEnvelope',
    'RuntimeMemoryDemand',
    'RuntimeRequestTicket',
    'RuntimeResidencyCohortError',
    'RuntimeResidencyProjectionError',
    'RuntimeResourceError',
    'RuntimeTransitionPlan',
    'RuntimeTransitionReceipt',
    'UnifiedScheduleError',
    '_HelperAttachLeases',
    '_RECOVERABLE_ERRORS',
    '_RequestHelperAttachAttempt',
    '_RequestHelperPreparation',
    '_StalePhoneSessionAssignment',
    '_arbitration',
    '_attachment',
    '_authorization',
    '_check_partial_phone_replacement_source_exact',
    '_check_partial_phone_replacement_transitions',
    '_check_preparation_completion_receipts',
    '_cleanup',
    '_completion',
    '_helper_preparation_lease_demands',
    '_helper_preparation_projection_sha256',
    '_memory',
    '_partial_phone_replacement_session_diff',
    '_rebind_blocker_quiescence',
    '_recovery',
    '_rewrite_partial_phone_replacement_demands',
    '_runtime_serialized',
    '_start',
    '_text',
    'canonical_sha256',
]


class HelperPreparationMixin:
    """Request helper preparation transaction: begin, complete, check, fail, leases, windows, attach."""

    @staticmethod
    def _partial_phone_replacement_memory_demands(
        demands: Sequence[RuntimeMemoryDemand],
        *,
        phone_device_id: str,
        source: ModelPhoneResidencyLayout | None,
        target: ModelPhoneResidencyLayout,
        transitions: Sequence[RuntimeTransitionPlan],
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> tuple[RuntimeMemoryDemand, ...]:
        """Account one exact session replacement without double allocation."""
        return _memory._partial_phone_replacement_memory_demands(
            demands,
            phone_device_id=phone_device_id,
            source=source,
            target=target,
            transitions=transitions,
            snapshot=snapshot,
        )

    def _candidate_set_with_verified_partial_phone_memory(
        self,
        candidate_set: AutomatedCandidateSet,
        manifest: ModelManifest,
        target: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        *,
        raise_capacity_error: bool = False,
    ) -> AutomatedCandidateSet:
        """Clear a coarse memory rejection after exact COW validation."""
        return _memory._candidate_set_with_verified_partial_phone_memory(
            self,
            candidate_set,
            manifest,
            target,
            snapshot,
            observed_at_us,
            raise_capacity_error=raise_capacity_error,
        )

    @staticmethod
    def _copy_on_write_preparation_yielding_resources(
        source: ModelPhoneResidencyLayout | None,
        target: ModelPhoneResidencyLayout,
    ) -> tuple[str, ...]:
        """Keep long replacement preparation off shared inference leases."""
        return _memory._copy_on_write_preparation_yielding_resources(source, target)

    @_runtime_serialized
    def begin_request_helper_preparation(
        self,
        request_id: str,
        *,
        observed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
        expected_phone_layout_generation: int | None = None,
        expected_phone_layout_geometry_sha256: str | None = None,
        expected_operator_plan_sha256: str | None = None,
    ) -> Mapping[str, object]:
        """Reserve and bind one exact asynchronous helper transition."""
        return _start.begin_request_helper_preparation(
            self,
            request_id,
            observed_at_us=observed_at_us,
            snapshot=snapshot,
            expected_phone_layout_generation=expected_phone_layout_generation,
            expected_phone_layout_geometry_sha256=expected_phone_layout_geometry_sha256,
            expected_operator_plan_sha256=expected_operator_plan_sha256,
        )

    def _begin_request_helper_preparation_transaction(
        self,
        *,
        request_id: str,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope,
        state: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        preparation_ticket_id: str,
        transitions: Sequence[RuntimeTransitionPlan],
        transition_ids: tuple[str, ...],
        yielding_resource_ids: tuple[str, ...],
        demands: Sequence[LeaseDemand],
        duration_us: int,
        ready_at_us: int,
        memory_owner_id: str,
        observed_at_us: int,
    ) -> Mapping[str, object]:
        return _start._begin_request_helper_preparation_transaction(
            self,
            request_id=request_id,
            ticket=ticket,
            helper=helper,
            state=state,
            snapshot=snapshot,
            preparation_ticket_id=preparation_ticket_id,
            transitions=transitions,
            transition_ids=transition_ids,
            yielding_resource_ids=yielding_resource_ids,
            demands=demands,
            duration_us=duration_us,
            ready_at_us=ready_at_us,
            memory_owner_id=memory_owner_id,
            observed_at_us=observed_at_us,
        )

    def _preparation_helper_envelope(
        self,
        ticket: RuntimeRequestTicket,
        request_id: str,
        *,
        expected_phone_layout_generation: int | None,
        expected_phone_layout_geometry_sha256: str | None,
        expected_operator_plan_sha256: str | None,
    ) -> RuntimeHelperExecutionEnvelope | str:
        """The helper envelope to prepare, or the status when there is none."""
        return _authorization._preparation_helper_envelope(
            self,
            ticket,
            request_id,
            expected_phone_layout_generation=expected_phone_layout_generation,
            expected_phone_layout_geometry_sha256=expected_phone_layout_geometry_sha256,
            expected_operator_plan_sha256=expected_operator_plan_sha256,
        )

    def _proposed_phone_layout_for_preparation(
        self,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope,
    ) -> tuple[ModelPhoneResidencyLayout | None, Mapping[str, object] | None]:
        """The PROPOSED layout to prepare, or the early result to return."""
        return _authorization._proposed_phone_layout_for_preparation(self, ticket, helper)

    def _defer_preparation_for_blockers(
        self,
        request_id: str,
        state: ModelPhoneResidencyLayout,
        blockers: Sequence[str],
        observed_at_us: int,
    ) -> Mapping[str, object]:
        """Drain the helper attachments that block one layout transition."""
        return _authorization._defer_preparation_for_blockers(
            self,
            request_id,
            state,
            blockers,
            observed_at_us,
        )

    def _verify_proposed_layout_ready(
        self,
        request_id: str,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope,
        state: ModelPhoneResidencyLayout,
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
        preparation_ticket_id: str,
    ) -> Mapping[str, object] | None:
        """Promote a PROPOSED layout the snapshot already proves READY."""
        return _authorization._verify_proposed_layout_ready(
            self,
            request_id,
            ticket,
            helper,
            state,
            snapshot=snapshot,
            observed_at_us=observed_at_us,
            preparation_ticket_id=preparation_ticket_id,
        )

    def _reserve_helper_preparation_memory(
        self,
        helper: RuntimeHelperExecutionEnvelope,
        state: ModelPhoneResidencyLayout,
        transitions: Sequence[RuntimeTransitionPlan],
        snapshot: HeterogeneousRuntimeSnapshot,
        *,
        phone_device_id: str,
        memory_owner_id: str,
        observed_at_us: int,
    ) -> tuple[RuntimeMemoryDemand, ...]:
        """Reserve the phone memory of one preparation; returns its demands."""
        return _authorization._reserve_helper_preparation_memory(
            self,
            helper,
            state,
            transitions,
            snapshot,
            phone_device_id=phone_device_id,
            memory_owner_id=memory_owner_id,
            observed_at_us=observed_at_us,
        )

    @staticmethod
    def _new_request_helper_preparation(
        request_id: str,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope,
        state: ModelPhoneResidencyLayout,
        *,
        preparation_ticket_id: str,
        projection_sha256: str,
        transition_ids: tuple[str, ...],
        lease_tokens: tuple[str, ...],
        yielding_resource_ids: tuple[str, ...],
        memory_owner_id: str,
        observed_at_us: int,
        ready_at_us: int,
        phone_safety_state,
    ) -> _RequestHelperPreparation:
        return _authorization._new_request_helper_preparation(
            request_id,
            ticket,
            helper,
            state,
            preparation_ticket_id=preparation_ticket_id,
            projection_sha256=projection_sha256,
            transition_ids=transition_ids,
            lease_tokens=lease_tokens,
            yielding_resource_ids=yielding_resource_ids,
            memory_owner_id=memory_owner_id,
            observed_at_us=observed_at_us,
            ready_at_us=ready_at_us,
            phone_safety_state=phone_safety_state,
        )

    def _preparation_phone_safety_state(
        self,
        phone_device_id: str | None,
        snapshot: HeterogeneousRuntimeSnapshot,
    ):
        return _authorization._preparation_phone_safety_state(self, phone_device_id, snapshot)

    def _defer_preparation_for_stale_session(
        self,
        request_id: str,
        ticket: RuntimeRequestTicket,
        state: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> Mapping[str, object]:
        """Reject the stale proposal and request a fresh phone plan."""
        return _authorization._defer_preparation_for_stale_session(
            self,
            request_id,
            ticket,
            state,
            snapshot,
            observed_at_us,
        )

    def _reject_stale_phone_layout_proposal(
        self,
        request_id: str,
        ticket: RuntimeRequestTicket,
        state: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        observed_at_us: int,
    ) -> Mapping[str, object] | None:
        """Revalidate a proposal against arrived demand before loading it."""
        return _authorization._reject_stale_phone_layout_proposal(
            self,
            request_id,
            ticket,
            state,
            snapshot,
            observed_at_us,
        )

    def _defer_preparation_for_memory_capacity(
        self,
        request_id: str,
        state: ModelPhoneResidencyLayout,
        exc: RuntimeResourceError,
        observed_at_us: int,
    ) -> Mapping[str, object]:
        return _authorization._defer_preparation_for_memory_capacity(
            self,
            request_id,
            state,
            exc,
            observed_at_us,
        )

    @_runtime_serialized
    def complete_request_helper_preparation(
        self,
        request_id: str,
        preparation_ticket_id: str,
        receipts: Sequence[RuntimeTransitionReceipt],
        *,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> Mapping[str, object]:
        return _completion.complete_request_helper_preparation(
            self,
            request_id,
            preparation_ticket_id,
            receipts,
            snapshot=snapshot,
        )

    def _complete_request_helper_preparation_transaction(
        self,
        *,
        request_id: str,
        preparation_ticket_id: str,
        preparation: _RequestHelperPreparation,
        rows: tuple[RuntimeTransitionReceipt, ...],
        snapshot: HeterogeneousRuntimeSnapshot,
        finished_at_us: int,
        verification_observed_at_us: int,
        verification_sha256: str,
    ) -> Mapping[str, object]:
        return _completion._complete_request_helper_preparation_transaction(
            self,
            request_id=request_id,
            preparation_ticket_id=preparation_ticket_id,
            preparation=preparation,
            rows=rows,
            snapshot=snapshot,
            finished_at_us=finished_at_us,
            verification_observed_at_us=verification_observed_at_us,
            verification_sha256=verification_sha256,
        )

    def _mark_request_helper_preparation_stale(
        self,
        preparation: _RequestHelperPreparation,
        preparation_ticket_id: str,
        finished_at_us: int,
    ) -> _RequestHelperPreparation:
        return _completion._mark_request_helper_preparation_stale(
            self,
            preparation,
            preparation_ticket_id,
            finished_at_us,
        )

    def _ignore_stale_preparation_completion(
        self,
        preparation: _RequestHelperPreparation,
        *,
        request_id: str,
        preparation_ticket_id: str,
        finished_at_us: int,
    ) -> Mapping[str, object]:
        """Release a preparation whose layout transition is no longer live."""
        return _completion._ignore_stale_preparation_completion(
            self,
            preparation,
            request_id=request_id,
            preparation_ticket_id=preparation_ticket_id,
            finished_at_us=finished_at_us,
        )

    def _rematerialize_committed_layout_helpers(
        self,
        request_id: str,
        ready: ModelPhoneResidencyLayout,
        snapshot: HeterogeneousRuntimeSnapshot,
        verification_observed_at_us: int,
        phone_safety_state,
    ) -> tuple[str, ...]:
        # The physical load is applied and verified; the layout is now
        # committed. Re-binding other requests' helpers is a post-commit
        # step that must never roll the commit back.
        return _completion._rematerialize_committed_layout_helpers(
            self,
            request_id,
            ready,
            snapshot,
            verification_observed_at_us,
            phone_safety_state,
        )

    def _reevaluate_ready_layout_portfolio(
        self,
        request_id: str,
        snapshot: HeterogeneousRuntimeSnapshot,
        verification_observed_at_us: int,
    ) -> None:
        return _completion._reevaluate_ready_layout_portfolio(
            self,
            request_id,
            snapshot,
            verification_observed_at_us,
        )

    @_runtime_serialized
    def check_request_helper_preparation(
        self,
        preparation_ticket_id: str,
        *,
        observed_at_us: int,
        guard_us: int = 250_000,
        quantum_us: int = 2_000_000,
    ) -> None:
        return _completion.check_request_helper_preparation(
            self,
            preparation_ticket_id,
            observed_at_us=observed_at_us,
            guard_us=guard_us,
            quantum_us=quantum_us,
        )

    @_runtime_serialized
    def fail_request_helper_preparation(
        self,
        request_id: str,
        preparation_ticket_id: str,
        *,
        failed_at_us: int,
        reason: str,
        unavailable_session_ids: Sequence[str] = (),
        restored_session_generations: Mapping[str, int] | None = None,
        snapshot: HeterogeneousRuntimeSnapshot | None = None,
    ) -> None:
        return _recovery.fail_request_helper_preparation(
            self,
            request_id,
            preparation_ticket_id,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable_session_ids=unavailable_session_ids,
            restored_session_generations=restored_session_generations,
            snapshot=snapshot,
        )

    def _fail_request_helper_preparation_transaction(
        self,
        *,
        request_id: str,
        preparation_ticket_id: str,
        preparation: _RequestHelperPreparation,
        failed_at_us: int,
        reason: str,
        unavailable: tuple[str, ...],
        restored_session_generations: Mapping[str, int] | None,
    ) -> None:
        return _recovery._fail_request_helper_preparation_transaction(
            self,
            request_id=request_id,
            preparation_ticket_id=preparation_ticket_id,
            preparation=preparation,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable=unavailable,
            restored_session_generations=restored_session_generations,
        )

    def _reevaluate_failed_layout_portfolio(
        self,
        request_id: str,
        failed_at_us: int,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> None:
        """Replan from the restored physical authority after a load failure."""
        return _recovery._reevaluate_failed_layout_portfolio(
            self,
            request_id,
            failed_at_us,
            snapshot,
        )

    def _rebinds_targeting_generation(
        self,
        generation: int,
    ) -> tuple[tuple[str, Mapping[str, object]], ...]:
        return _recovery._rebinds_targeting_generation(self, generation)

    def _settle_rebind_after_transition_failure(
        self,
        affected_id: str,
        affected_rebind: Mapping[str, object],
        *,
        preparation_ticket_id: str,
        failed_at_us: int,
        reason: str,
        unavailable: tuple[str, ...],
    ) -> None:
        """Retain, cancel, or detach one rebind blocked by a failed transition."""
        return _recovery._settle_rebind_after_transition_failure(
            self,
            affected_id,
            affected_rebind,
            preparation_ticket_id=preparation_ticket_id,
            failed_at_us=failed_at_us,
            reason=reason,
            unavailable=unavailable,
        )

    def _release_request_helper_leases(
        self,
        request_id: str,
        at_us: int,
    ) -> None:
        return _cleanup._release_request_helper_leases(self, request_id, at_us)

    def _close_request_helper_runtime(
        self,
        request_id: str,
        at_us: int,
        outcome: str,
    ) -> None:
        """Release request-scoped helper state without evicting residency."""
        return _cleanup._close_request_helper_runtime(self, request_id, at_us, outcome)

    def _helper_window_bid(
        self,
        request_id: str,
        *,
        requested_fraction_ppm: int | None = None,
    ) -> Mapping[str, object] | None:
        return _arbitration._helper_window_bid(
            self,
            request_id,
            requested_fraction_ppm=requested_fraction_ppm,
        )

    def _bounded_learning_exploration_bid(
        self,
        request_id: str,
        bid: Mapping[str, object] | None,
    ) -> bool:
        return _arbitration._bounded_learning_exploration_bid(self, request_id, bid)

    def _select_helper_window_owner(
        self,
        request_id: str,
        required_resource_ids: Sequence[str],
        *,
        requested_fraction_ppm: int,
        observed_at_us: int,
    ) -> tuple[str | None, Mapping[str, object] | None]:
        """Rank one shared HTP window without reserving future work."""
        return _arbitration._select_helper_window_owner(
            self,
            request_id,
            required_resource_ids,
            requested_fraction_ppm=requested_fraction_ppm,
            observed_at_us=observed_at_us,
        )

    def _helper_window_lease_horizon(
        self,
        request_id: str,
        *,
        requested_fraction_ppm: int,
        at_us: int,
    ) -> int:
        return _arbitration._helper_window_lease_horizon(
            self,
            request_id,
            requested_fraction_ppm=requested_fraction_ppm,
            at_us=at_us,
        )

    def _try_attach_ready_request_helper(
        self,
        ticket: RuntimeRequestTicket,
        *,
        slot_id: int,
        token_index: int,
        at_us: int,
        fraction_ppm: int,
        ready_helper: tuple | None = None,
    ) -> _RequestHelperAttachAttempt:
        return _attachment._try_attach_ready_request_helper(
            self,
            ticket,
            slot_id=slot_id,
            token_index=token_index,
            at_us=at_us,
            fraction_ppm=fraction_ppm,
            ready_helper=ready_helper,
        )

    def _attach_helper_window_leases(
        self,
        ticket: RuntimeRequestTicket,
        helper: RuntimeHelperExecutionEnvelope,
        layout: ModelPhoneResidencyLayout,
        *,
        current_attachment: object,
        token_index: int,
        at_us: int,
        fraction_ppm: int,
    ) -> _HelperAttachLeases | _RequestHelperAttachAttempt:
        """Reserve, extend, or release the helper window leases."""
        return _attachment._attach_helper_window_leases(
            self,
            ticket,
            helper,
            layout,
            current_attachment=current_attachment,
            token_index=token_index,
            at_us=at_us,
            fraction_ppm=fraction_ppm,
        )

    def _attach_ready_request_helper(
        self,
        ticket: RuntimeRequestTicket,
        *,
        slot_id: int,
        token_index: int,
        at_us: int,
        fraction_ppm: int,
        ready_helper: tuple | None = None,
    ) -> bool:
        return _attachment._attach_ready_request_helper(
            self,
            ticket,
            slot_id=slot_id,
            token_index=token_index,
            at_us=at_us,
            fraction_ppm=fraction_ppm,
            ready_helper=ready_helper,
        )

    def _request_helper_opportunity_for_ticket(
        self,
        ticket: RuntimeRequestTicket,
    ) -> HelperOpportunity | None:
        return _attachment._request_helper_opportunity_for_ticket(self, ticket)
