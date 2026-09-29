"""Scheduler-owned runtime request lifecycle and dispatch state."""

from __future__ import annotations

import copy
import threading
from typing import Callable, Mapping, Sequence

from .online_placement import OnlinePlacementReceipt
from .policy import Decision, LeaseRecord, Request, decision_to_json
from .runtime_admission import RuntimeRequestObservation
from .runtime_cost import (
    RuntimeCostEstimateSet,
    RuntimeExecutorBinding,
    RuntimeMemoryDemand,
    RuntimeModelArtifact,
)
from .runtime_dispatch_policy import (
    DEFAULT_RUNTIME_DISPATCH_POLICY,
    RuntimeDispatchPolicy,
)
from .runtime_queue import RuntimeDispatchQueue, RuntimeDispatchReceipt, RuntimeQueueError
from .runtime_decode_cohort import RuntimeDecodeCohortBinding
from .runtime_plan import RuntimeExecutionPlan, RuntimeExecutionReceipt, RuntimeTransitionReceipt
from .runtime_resources import RuntimeMemoryReservation, RuntimeResidencyProjectionToken


from .request_contracts.common import (
    RUNTIME_REQUEST_TICKET_SCHEMA as RUNTIME_REQUEST_TICKET_SCHEMA,
    RUNTIME_TRANSITION_STATES as RUNTIME_TRANSITION_STATES,
    RUNTIME_SELECTION_MODES as RUNTIME_SELECTION_MODES,
    RUNTIME_MEMORY_RESERVED_ATOMIC as RUNTIME_MEMORY_RESERVED_ATOMIC,
    RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT as RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT,
    RUNTIME_MEMORY_CANCELLED as RUNTIME_MEMORY_CANCELLED,
    RuntimeControllerError as RuntimeControllerError,
    RuntimeReplanRetryRequired as RuntimeReplanRetryRequired,
    _text as _text,
    _integer as _integer,
    _text_tuple as _text_tuple,
    RUNTIME_DISPATCH_STATES as RUNTIME_DISPATCH_STATES,
    RUNTIME_LEASE_STATES as RUNTIME_LEASE_STATES,
    RUNTIME_PREDICTION_STATES as RUNTIME_PREDICTION_STATES,
    RUNTIME_TERMINAL_STATES as RUNTIME_TERMINAL_STATES,
)
from .request_contracts.ticket import (
    _request_to_json as _request_to_json,
    _validate_execution_receipt as _validate_execution_receipt,
    _validate_ticket_identity as _validate_ticket_identity,
    _validate_ticket_lifecycle as _validate_ticket_lifecycle,
    _validate_ticket_execution_plan as _validate_ticket_execution_plan,
    _validate_ticket_transition_receipts as _validate_ticket_transition_receipts,
    _validate_ticket_terminal_state as _validate_ticket_terminal_state,
    _validate_ticket_runtime_bindings as _validate_ticket_runtime_bindings,
    _normalize_runtime_ticket as _normalize_runtime_ticket,
    _initialize_runtime_request_ticket as _initialize_runtime_request_ticket,
    RuntimeRequestTicket as RuntimeRequestTicket,
)
from .request_contracts.receipts import (
    RuntimeLatencyUpperBound as RuntimeLatencyUpperBound,
    RuntimeLeaseCoverage as RuntimeLeaseCoverage,
    assess_runtime_completion as assess_runtime_completion,
    RuntimeLeaseExtensionReceipt as RuntimeLeaseExtensionReceipt,
    RuntimeCompletionReceipt as RuntimeCompletionReceipt,
    RuntimeFailureRecovery as RuntimeFailureRecovery,
)

from .runtime_controller_ops import admission as _admission
from .runtime_controller_ops import cohorts as _cohorts
from .runtime_controller_ops import compaction as _compaction
from .runtime_controller_ops import completion as _completion
from .runtime_controller_ops import dispatch as _dispatch
from .runtime_controller_ops import leases as _leases
from .runtime_controller_ops import queue as _queue
from .runtime_controller_ops import replan as _replan


__all__ = [
    'Decision',
    'LeaseRecord',
    'OnlinePlacementReceipt',
    'RUNTIME_DISPATCH_STATES',
    'RUNTIME_LEASE_STATES',
    'RUNTIME_MEMORY_CANCELLED',
    'RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT',
    'RUNTIME_MEMORY_RESERVED_ATOMIC',
    'RUNTIME_PREDICTION_STATES',
    'RUNTIME_REQUEST_TICKET_SCHEMA',
    'RUNTIME_SELECTION_MODES',
    'RUNTIME_TERMINAL_STATES',
    'RUNTIME_TRANSITION_STATES',
    'Request',
    'RuntimeCompletionReceipt',
    'RuntimeController',
    'RuntimeControllerError',
    'RuntimeCostEstimateSet',
    'RuntimeDecodeCohortBinding',
    'RuntimeDispatchQueue',
    'RuntimeDispatchReceipt',
    'RuntimeExecutionPlan',
    'RuntimeExecutionReceipt',
    'RuntimeExecutorBinding',
    'RuntimeFailureRecovery',
    'RuntimeLatencyUpperBound',
    'RuntimeLeaseCoverage',
    'RuntimeLeaseExtensionReceipt',
    'RuntimeMemoryDemand',
    'RuntimeMemoryReservation',
    'RuntimeModelArtifact',
    'RuntimeQueueError',
    'RuntimeReplanRetryRequired',
    'RuntimeRequestObservation',
    'RuntimeRequestTicket',
    'RuntimeResidencyProjectionToken',
    'RuntimeTransitionReceipt',
    '_admission',
    '_cohorts',
    '_compaction',
    '_completion',
    '_dispatch',
    '_initialize_runtime_request_ticket',
    '_integer',
    '_leases',
    '_normalize_runtime_ticket',
    '_queue',
    '_replan',
    '_request_to_json',
    '_text',
    '_text_tuple',
    '_validate_execution_receipt',
    '_validate_ticket_execution_plan',
    '_validate_ticket_identity',
    '_validate_ticket_lifecycle',
    '_validate_ticket_runtime_bindings',
    '_validate_ticket_terminal_state',
    '_validate_ticket_transition_receipts',
    'assess_runtime_completion',
    'decision_to_json',
]


class RuntimeController:
    """Own runtime tickets, dispatch waiting, and route quarantine."""

    def __init__(self) -> None:
        self.dispatch_policy: RuntimeDispatchPolicy = (
            DEFAULT_RUNTIME_DISPATCH_POLICY
        )
        self.queue = RuntimeDispatchQueue(self.dispatch_policy)
        self._lock = threading.RLock()
        (
            self._dispatch_bypass_counts,
            self._dispatch_policy_notes,
            self._dispatch_policy_stats,
        ) = _dispatch.empty_dispatch_state()
        self._dispatch_policy_refusals: list[dict[str, object]] = []
        self._pending_dispatch_precedence: (
            tuple[str, tuple[str, ...], dict[str, object]] | None
        ) = None
        self._pending_retained_order: (
            tuple[str, tuple[int, Sequence[str]]] | None
        ) = None
        self._pending_continuous_join_resolution: str | None = None
        self._tickets: dict[str, RuntimeRequestTicket] = {}
        self._ticket_history: list[RuntimeRequestTicket] = []
        self._terminal_tickets: dict[str, RuntimeRequestTicket] = {}
        self._next_attempt_by_request: dict[str, int] = {}
        self._quarantined_routes: set[str] = set()
        self._quarantined_resources: set[str] = set()
        # Lost or absent phones (action ``device_lost``): membership is physical, so it stays
        # outside checkpoints and survives a transaction rollback.
        self._device_quarantines: dict[str, dict[str, object]] = {}
        self._route_uncertainty: dict[str, dict[str, object]] = {}
        self._cancellation_audit: list[dict[str, int | str]] = []
        self._capacity_releases: dict[str, RuntimeCompletionReceipt] = {}
        self._acquired_tickets: dict[str, RuntimeRequestTicket] = {}

    def checkpoint(self) -> object:
        with self._lock:
            return (
                dict(self._tickets),
                list(self._ticket_history),
                dict(self._terminal_tickets),
                dict(self._next_attempt_by_request),
                set(self._quarantined_routes),
                set(self._quarantined_resources),
                copy.deepcopy(self._route_uncertainty),
                copy.deepcopy(self._cancellation_audit),
                dict(self._capacity_releases),
                dict(self._acquired_tickets),
                self.queue.checkpoint(),
                _dispatch.checkpoint_dispatch_state(self),
            )

    def restore(self, checkpoint: object) -> None:
        try:
            (
                tickets,
                history,
                terminals,
                attempts,
                quarantined,
                quarantined_resources,
                uncertainty,
                cancellation_audit,
                capacity_releases,
                acquired_tickets,
                queue_checkpoint,
                dispatch_state,
            ) = checkpoint
        except (TypeError, ValueError) as exc:
            raise RuntimeControllerError(
                "runtime controller checkpoint is invalid"
            ) from exc
        if (
            type(tickets) is not dict
            or type(history) is not list
            or type(terminals) is not dict
            or type(attempts) is not dict
            or type(quarantined) is not set
            or type(quarantined_resources) is not set
            or type(uncertainty) is not dict
            or type(cancellation_audit) is not list
            or type(capacity_releases) is not dict
            or type(acquired_tickets) is not dict
            or any(
                not isinstance(ticket, RuntimeRequestTicket)
                or ticket.ticket_id != identity
                or ticket.dispatch_state != "ACQUIRED"
                for identity, ticket in acquired_tickets.items()
            )
            or any(
                type(request_id) is not str
                or not isinstance(receipt, RuntimeCompletionReceipt)
                or receipt.request_id != request_id
                for request_id, receipt in capacity_releases.items()
            )
        ):
            raise RuntimeControllerError(
                "runtime controller checkpoint is invalid"
            )
        dispatch_state = _dispatch.parse_dispatch_state(dispatch_state)
        with self._lock:
            self.queue.restore(queue_checkpoint)
            self._tickets = dict(tickets)
            self._ticket_history = list(history)
            self._terminal_tickets = dict(terminals)
            self._next_attempt_by_request = dict(attempts)
            self._quarantined_routes = set(quarantined)
            self._quarantined_resources = set(quarantined_resources)
            self._route_uncertainty = copy.deepcopy(uncertainty)
            self._cancellation_audit = copy.deepcopy(cancellation_audit)
            self._capacity_releases = dict(capacity_releases)
            self._acquired_tickets = dict(acquired_tickets)
            (
                self._dispatch_bypass_counts,
                self._dispatch_policy_notes,
                self._dispatch_policy_stats,
            ) = dispatch_state

    def configure_dispatch_policy(self, policy: RuntimeDispatchPolicy) -> None:
        """Select the dispatch policy before the first admission."""
        return _dispatch.configure_dispatch_policy(self, policy)

    def dispatch_bypass_count(self, request_id: str) -> int:
        return _dispatch.dispatch_bypass_count(self, request_id)

    def dispatch_order_view(self) -> Mapping[str, Mapping[str, object]]:
        return _dispatch.dispatch_order_view(self)

    def refresh_replan_receipt(
        self, request_id: str, expected_ticket_id: str, observed_at_us: int
    ) -> RuntimeRequestTicket:
        """Rebind a committed replan wake to the current queue generation."""
        return _dispatch.refresh_replan_receipt(
            self, request_id, expected_ticket_id, observed_at_us
        )

    def replan_queued_now(
        self,
        request_ids: Sequence[str],
        reason: str,
        at_us: int,
        *,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        """Cancel queued reservations whose owners must replan now."""
        return _dispatch.replan_queued_now(
            self,
            request_ids,
            reason,
            at_us,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
        )

    def dispatch_precedence(
        self,
        request_id: str,
        follower_request_ids: Sequence[str],
        note: Mapping[str, object],
    ):
        """Order the next admission of one request before queued followers."""
        return _dispatch.dispatch_precedence(
            self, request_id, follower_request_ids, note
        )

    def recovery_dispatch_order(
        self, request_id: str
    ) -> tuple[int, tuple[str, ...]] | None:
        """Queue place of an active attempt another model's work waits on."""
        return _dispatch.recovery_dispatch_order(self, request_id)

    def retained_dispatch_order(
        self, request_id: str, order: tuple[int, Sequence[str]]
    ):
        """Admit the next attempt of one request in a failed attempt's place."""
        return _dispatch.retained_dispatch_order(self, request_id, order)

    def record_dispatch_policy_event(self, name: str, count: int = 1) -> None:
        return _dispatch.record_dispatch_policy_event(self, name, count)

    def dispatch_policy_note(
        self, ticket_id: str
    ) -> Mapping[str, object] | None:
        return _dispatch.dispatch_policy_note(self, ticket_id)

    def continuous_join_resolution(self, request_id: str):
        """Mark one joiner's re-resolution ahead of a displaced residency change."""
        return _dispatch.continuous_join_resolution(self, request_id)

    def continuous_join_resolution_request_id(self) -> str | None:
        return _dispatch.continuous_join_resolution_request_id(self)

    def record_dispatch_policy_refusal(
        self, kind: str, request_id: str, reason: str, observed_at_us: int
    ) -> None:
        return _dispatch.record_dispatch_policy_refusal(
            self, kind, request_id, reason, observed_at_us
        )

    def dispatch_policy_state(self) -> Mapping[str, object]:
        return _dispatch.dispatch_policy_state(self)

    def acquired_lease_overlaps(
        self,
        ticket_id: str,
        resource_ids: Sequence[str],
        start_us: int,
        end_us: int,
    ) -> tuple[str, ...] | None:
        """Query actual acquisition history, including already terminal peers."""
        with self._lock:
            own = self._acquired_tickets.get(ticket_id)
            if own is None or own.dispatch_receipt.observed_at_us > start_us:
                return None
            resources = set(resource_ids)
            overlaps = []
            for identity, acquired in self._acquired_tickets.items():
                if identity == ticket_id:
                    continue
                current = self._terminal_tickets.get(identity)
                if current is None:
                    current = self._tickets.get(acquired.request.request_id)
                if current is None or current.ticket_id != identity:
                    return None
                if acquired.dispatch_receipt.observed_at_us >= end_us:
                    continue
                if (
                    current.actual_end_us is not None
                    and current.actual_end_us <= start_us
                ):
                    continue
                if any(
                    lease.resource_id in resources
                    for lease in acquired.decision.leases
                ):
                    overlaps.append(identity)
            return tuple(sorted(overlaps))

    def acquired_execution_plan(self, ticket_id: str) -> RuntimeExecutionPlan | None:
        with self._lock:
            ticket = self._acquired_tickets.get(ticket_id)
            return None if ticket is None else ticket.execution_plan

    def acquired_ticket(self, ticket_id: str) -> RuntimeRequestTicket | None:
        """The ticket as acquired; its model and binding do not change afterwards."""
        with self._lock:
            return self._acquired_tickets.get(ticket_id)

    def acquired_prediction_end_us(self, ticket_ids: Sequence[str]) -> int | None:
        with self._lock:
            requested = set(ticket_ids)
            rows = tuple(
                ticket for ticket in self._tickets.values()
                if ticket.ticket_id in requested
                and ticket.dispatch_state == "ACQUIRED"
                and ticket.actual_end_us is None
            )
            if {row.ticket_id for row in rows} != requested or not rows:
                return None
            return max(
                max(
                    row.decision.finish_upper_us,
                    *(row.final_reserved_until_us[lease.token]
                      for lease in row.live_leases),
                )
                for row in rows
            )

    @property
    def quarantined_routes(self) -> frozenset[str]:
        return frozenset(self._quarantined_routes)

    def route_is_quarantined(self, route_id: str) -> bool:
        return _text("runtime route_id", route_id) in self._quarantined_routes

    def _device_quarantined_resources(self) -> set[str]:
        return {
            resource_id
            for row in self._device_quarantines.values()
            for resource_id in row["resource_ids"]
        }

    @property
    def quarantined_resources(self) -> frozenset[str]:
        return frozenset(
            self._quarantined_resources | self._device_quarantined_resources()
        )

    def resources_are_quarantined(
        self, resource_ids: Sequence[str]
    ) -> bool:
        return bool(set(resource_ids) & (
            self._quarantined_resources | self._device_quarantined_resources()
        ))

    @property
    def quarantined_devices(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._device_quarantines)

    def quarantine_device(
        self,
        device_id: str,
        *,
        resource_ids: Sequence[str],
        reason: str,
        at_us: int,
    ) -> bool:
        """Keep a lost device's own resources out of admission; False if already quarantined."""
        device_id = _text("runtime quarantined device", device_id)
        reason = _text("runtime device quarantine reason", reason)
        at_us = _integer("runtime device quarantine at_us", at_us)
        resources = tuple(sorted(
            _text("runtime quarantined device resource", value)
            for value in resource_ids
        ))
        if len(resources) != len(set(resources)):
            raise RuntimeControllerError(
                "runtime quarantined device resources are duplicated"
            )
        with self._lock:
            if device_id in self._device_quarantines:
                return False
            self._device_quarantines[device_id] = {
                "action": "device_lost",
                "at_us": at_us,
                "reason": reason,
                "resource_ids": resources,
            }
            return True

    def readmit_device(self, device_id: str) -> bool:
        """Release a device quarantine; False if the device was not quarantined."""
        device_id = _text("runtime readmitted device", device_id)
        with self._lock:
            return self._device_quarantines.pop(device_id, None) is not None

    def effective_bindings(
        self, bindings: Sequence[RuntimeExecutorBinding]
    ) -> tuple[RuntimeExecutorBinding, ...]:
        return _admission.effective_bindings(self, bindings)

    def admit(
        self,
        *,
        request: Request,
        model: RuntimeModelArtifact,
        runtime_observation: RuntimeRequestObservation,
        estimates: RuntimeCostEstimateSet,
        decision: Decision,
        binding: RuntimeExecutorBinding,
        executor_bindings: Sequence[RuntimeExecutorBinding],
        online_receipt: OnlinePlacementReceipt | None,
        admitted_at_us: int,
        memory_reservations: Sequence[RuntimeMemoryReservation] = (),
        execution_plan: RuntimeExecutionPlan | None = None,
        planning_profile_sha256: str | None = None,
        previous_ticket_id: str | None = None,
        failure_reason: str | None = None,
        previous_transition_receipts: Sequence[
            RuntimeTransitionReceipt
        ] = (),
        selection_mode: str = "energy-aware",
        decode_cohort: RuntimeDecodeCohortBinding | None = None,
        residency_order_barrier: bool | None = None,
        residency_projection_token: (
            RuntimeResidencyProjectionToken | None
        ) = None,
        phone_layout_generation: int | None = None,
        residency_hysteresis_key: str | None = None,
    ) -> RuntimeRequestTicket:
        return _admission.admit(
            self,
            request=request,
            model=model,
            runtime_observation=runtime_observation,
            estimates=estimates,
            decision=decision,
            binding=binding,
            executor_bindings=executor_bindings,
            online_receipt=online_receipt,
            admitted_at_us=admitted_at_us,
            memory_reservations=memory_reservations,
            execution_plan=execution_plan,
            planning_profile_sha256=planning_profile_sha256,
            previous_ticket_id=previous_ticket_id,
            failure_reason=failure_reason,
            previous_transition_receipts=previous_transition_receipts,
            selection_mode=selection_mode,
            decode_cohort=decode_cohort,
            residency_order_barrier=residency_order_barrier,
            residency_projection_token=residency_projection_token,
            phone_layout_generation=phone_layout_generation,
            residency_hysteresis_key=residency_hysteresis_key,
        )

    def bind_decode_cohort(
        self,
        request_id: str,
        binding: RuntimeDecodeCohortBinding,
    ) -> RuntimeRequestTicket:
        """Refresh nonterminal ticket membership after cohort formation."""
        return _admission.bind_decode_cohort(self, request_id, binding)

    def detach_decode_cohort_for_replan(
        self,
        request_id: str,
        previous: RuntimeDecodeCohortBinding,
        transferred_leases: Sequence[LeaseRecord] = (),
    ) -> RuntimeRequestTicket:
        """Remove prospective cohort state from a replacement attempt."""
        return _cohorts.detach_decode_cohort_for_replan(
            self,
            request_id,
            previous,
            transferred_leases,
        )

    def dissolve_decode_cohort_singleton(
        self,
        request_id: str,
        binding: RuntimeDecodeCohortBinding,
        leases: Sequence[LeaseRecord],
    ) -> RuntimeRequestTicket:
        """Return a one-member prospective cohort to request ownership."""
        return _cohorts.dissolve_decode_cohort_singleton(self, request_id, binding, leases)

    def transfer_decode_cohort_singleton(
        self,
        request_id: str,
        binding: RuntimeDecodeCohortBinding,
        leases: Sequence[LeaseRecord],
    ) -> RuntimeRequestTicket:
        """Transfer an active cohort lease to its sole survivor."""
        return _cohorts.transfer_decode_cohort_singleton(self, request_id, binding, leases)

    def ticket(self, request_id: str) -> RuntimeRequestTicket:
        request_id = _text("runtime request_id", request_id)
        with self._lock:
            ticket = self._tickets.get(request_id)
            if ticket is None:
                raise RuntimeControllerError("runtime request has no ticket")
            return ticket

    def ticket_by_id(self, ticket_id: str) -> RuntimeRequestTicket:
        ticket_id = _text("runtime ticket_id", ticket_id)
        with self._lock:
            current = next(
                (
                    ticket for ticket in self._tickets.values()
                    if ticket.ticket_id == ticket_id
                ),
                None,
            )
            if current is not None:
                return current
            terminal = self._terminal_tickets.get(ticket_id)
            if terminal is None:
                raise RuntimeControllerError("runtime ticket is unknown")
            return terminal

    def current_tickets(
        self, dispatch_states: Sequence[str] = ()
    ) -> tuple[RuntimeRequestTicket, ...]:
        states = frozenset(dispatch_states)
        if states - RUNTIME_DISPATCH_STATES:
            raise RuntimeControllerError(
                "runtime ticket state filter is invalid"
            )
        with self._lock:
            return tuple(
                ticket
                for _, ticket in sorted(self._tickets.items())
                if not states or ticket.dispatch_state in states
            )

    def defer_dispatch_wake(self):
        """Delay physical queue wakeups through one scheduler transaction."""
        return _queue.defer_dispatch_wake(self)

    def hold_dispatch_wake(self) -> int:
        """Hold physical queue wakeups through asynchronous publication."""
        return _queue.hold_dispatch_wake(self)

    def release_dispatch_wake(self, token: int) -> None:
        """Release one asynchronous publication wake hold."""
        return _queue.release_dispatch_wake(self, token)

    def require_queued_replan(
        self,
        request_id: str,
        reason: str,
        *,
        defer_behind_predecessors: bool = False,
    ) -> bool:
        """Wake one queued owner without changing its immutable attempt."""
        return _queue.require_queued_replan(
            self,
            request_id,
            reason,
            defer_behind_predecessors=defer_behind_predecessors,
        )

    def invalidate_queued_attempts(
        self,
        request_ids: Sequence[str],
        reason: str,
        at_us: int,
        *,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        """Cancel stale reservations while exposing only a causal frontier."""
        return _queue.invalidate_queued_attempts(
            self,
            request_ids,
            reason,
            at_us,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
        )

    def replan_required_requests(self) -> tuple[str, ...]:
        """Return replan-ready request IDs in stable queue order."""
        return _queue.replan_required_requests(self)

    def projection_request_ids(self) -> tuple[str, ...]:
        """Return attempts whose queue placements remain authoritative."""
        return _queue.projection_request_ids(self)

    def projection_causal_predecessors(
        self,
    ) -> Mapping[str, tuple[str, ...]]:
        """Return the queue edges that order projected dispatch."""
        return _queue.projection_causal_predecessors(self)

    def defer_replan_behind(
        self,
        request_id: str,
        predecessor_request_id: str,
        predecessor_ticket_id: str,
        reason: str,
        at_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        """Order a stale queued predecessor before a conflicting replan."""
        return _queue.defer_replan_behind(
            self,
            request_id,
            predecessor_request_id,
            predecessor_ticket_id,
            reason,
            at_us,
            cancel_owner,
            release_memory,
        )

    def defer_replan_until_active_completion(
        self,
        request_id: str,
        predecessor_request_id: str,
        predecessor_ticket_id: str,
        reason: str,
        at_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> None:
        """Keep a stale successor behind an acquired physical attempt."""
        return _queue.defer_replan_until_active_completion(
            self,
            request_id,
            predecessor_request_id,
            predecessor_ticket_id,
            reason,
            at_us,
            cancel_owner,
            release_memory,
        )

    def wait_ready(
        self,
        request_id: str,
        epoch_ns: int,
    ) -> RuntimeDispatchReceipt:
        """Wait for a queue event without committing scheduler-owned state."""
        return _queue.wait_ready(self, request_id, epoch_ns)

    def commit_wait(
        self,
        receipt: RuntimeDispatchReceipt,
        commit_acquired: Callable[[RuntimeRequestTicket], None] | None = None,
    ) -> RuntimeRequestTicket | None:
        """Atomically bind one still-current queue event to its ticket."""
        return _queue.commit_wait(self, receipt, commit_acquired)

    def wait(
        self,
        request_id: str,
        epoch_ns: int,
        commit_acquired: Callable[[RuntimeRequestTicket], None] | None = None,
    ) -> RuntimeRequestTicket:
        """Wait for dispatch when no outer scheduler transaction is required."""
        return _queue.wait(self, request_id, epoch_ns, commit_acquired)

    def record_transition_receipts(
        self,
        request_id: str,
        receipts: Sequence[RuntimeTransitionReceipt],
    ) -> RuntimeRequestTicket:
        return _queue.record_transition_receipts(self, request_id, receipts)

    def _mark_followers(
        self,
        decision: Decision,
        extended_until_us: int,
        reason: str,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
        return _queue._mark_followers(
            self,
            decision,
            extended_until_us,
            reason,
            cancel_owner,
            release_memory,
        )

    def _mark_selected_followers(
        self,
        request_ids: Sequence[str],
        reason: str,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        memory_owner_tokens: Callable[
            [str], tuple[str, ...]
        ] | None = None,
        *,
        cancelled_at_us: int = 0,
        defer_behind_predecessors: bool = True,
        allow_active_predecessors: bool = False,
    ) -> tuple[tuple[str, ...], Mapping[str, tuple[str, ...]]]:
        return _queue._mark_selected_followers(
            self,
            request_ids,
            reason,
            cancel_owner,
            release_memory,
            memory_owner_tokens,
            cancelled_at_us=cancelled_at_us,
            defer_behind_predecessors=defer_behind_predecessors,
            allow_active_predecessors=allow_active_predecessors,
        )

    @staticmethod
    def _detach_compaction_follower_memory(
        ticket: RuntimeRequestTicket,
        release_memory: Callable[[str], tuple[str, ...]] | None,
        memory_owner_tokens: Callable[[str], tuple[str, ...]],
    ) -> str:
        """Detach or validate one follower's memory ownership exactly once."""
        return _compaction._detach_compaction_follower_memory(
            ticket,
            release_memory,
            memory_owner_tokens,
        )

    def release_replanned_capacity_frontier(
        self,
        request_id: str,
        previous_decision: Decision,
        at_us: int,
        *,
        resource_capacities: Mapping[str, int] | None = None,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        memory_owner_tokens: Callable[
            [str], tuple[str, ...]
        ] | None = None,
    ) -> tuple[str, ...]:
        """Cancel and expose only the next owners of newly freed lanes."""
        return _compaction.release_replanned_capacity_frontier(
            self,
            request_id,
            previous_decision,
            at_us,
            resource_capacities=resource_capacities,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
            memory_owner_tokens=memory_owner_tokens,
        )

    def _mark_residency_transition_frontier(
        self,
        ticket: RuntimeRequestTicket,
        completed_at_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        return _compaction._mark_residency_transition_frontier(
            self,
            ticket,
            completed_at_us,
            cancel_owner,
            release_memory,
        )

    def extend(
        self,
        request_id: str,
        *,
        at_us: int,
        reserved_until_us: int,
        extend_leases: Callable[
            [Sequence[str], int, Sequence[str], int],
            tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]],
        ],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        retime_leases: Callable | None = None,
    ) -> RuntimeLeaseExtensionReceipt:
        return _leases.extend(
            self,
            request_id,
            at_us=at_us,
            reserved_until_us=reserved_until_us,
            extend_leases=extend_leases,
            release_memory=release_memory,
            retime_leases=retime_leases,
        )

    def finish_prepare_phase(self, request_id, *, retime_leases, release_memory,
                             previously_completed_tokens=()):
        return _leases.finish_prepare_phase(
            self, request_id, retime_leases=retime_leases, release_memory=release_memory,
            previously_completed_tokens=previously_completed_tokens)

    def extend_decode_cohort(
        self,
        binding: RuntimeDecodeCohortBinding,
        leases: Sequence[LeaseRecord],
        *,
        pending_member_request_id: str | None = None,
    ) -> None:
        """Apply one shared lease update before publishing a new member."""
        return _cohorts.extend_decode_cohort(
            self,
            binding,
            leases,
            pending_member_request_id=pending_member_request_id,
        )

    def extend_external(
        self,
        owner_id: str,
        leases: Sequence[LeaseRecord],
        *,
        at_us: int,
        reserved_until_us: int,
        extend_leases: Callable[
            [Sequence[str], int, Sequence[str], int],
            tuple[Mapping[str, int], Mapping[str, tuple[str, ...]]],
        ],
    ) -> tuple[str, ...]:
        """Atomically extend an observed phase and replan its followers."""
        return _leases.extend_external(
            self,
            owner_id,
            leases,
            at_us=at_us,
            reserved_until_us=reserved_until_us,
            extend_leases=extend_leases,
        )

    def _route_violation(
        self,
        ticket: RuntimeRequestTicket,
        actual_end_us: int,
        kind: str,
    ) -> str:
        return _completion._route_violation(self, ticket, actual_end_us, kind)

    def release_capacity(
        self,
        request_id: str,
        actual_end_us: int,
        *,
        resource_capacities: Mapping[str, int] | None = None,
        release_lease: Callable[[str, int], None],
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> RuntimeCompletionReceipt:
        """Release physical resources while terminal evidence is collected."""
        return _leases.release_capacity(
            self,
            request_id,
            actual_end_us,
            resource_capacities=resource_capacities,
            release_lease=release_lease,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
        )

    def complete(
        self,
        request_id: str,
        actual_end_us: int,
        *,
        resource_capacities: Mapping[str, int] | None = None,
        release_lease: Callable[[str, int], None],
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        execution_receipt: RuntimeExecutionReceipt | None = None,
    ) -> RuntimeCompletionReceipt:
        return _completion.complete(
            self,
            request_id,
            actual_end_us,
            resource_capacities=resource_capacities,
            release_lease=release_lease,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
            execution_receipt=execution_receipt,
        )

    def fail(
        self,
        request_id: str,
        failed_at_us: int,
        reason: str,
        *,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        failed_resource_ids: Sequence[str] = (),
    ) -> tuple[RuntimeRequestTicket, tuple[str, ...], str | None]:
        return _completion.fail(
            self,
            request_id,
            failed_at_us,
            reason,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
            failed_resource_ids=failed_resource_ids,
        )

    def prepare_replan(
        self,
        request_id: str,
        at_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        *,
        invalidate_dependents: bool | None = None,
        expected_ticket_id: str | None = None,
        expected_queue_generation: int | None = None,
    ) -> RuntimeRequestTicket:
        return _replan.prepare_replan(
            self,
            request_id,
            at_us,
            cancel_owner,
            release_memory,
            invalidate_dependents=invalidate_dependents,
            expected_ticket_id=expected_ticket_id,
            expected_queue_generation=expected_queue_generation,
        )

    def defer_causal_dependents_before(
        self,
        request_id: str,
        before_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[str, ...]:
        """Defer queued dependents reserved before their predecessor."""
        return _replan.defer_causal_dependents_before(
            self, request_id, before_us, cancel_owner, release_memory
        )

    def queued_causal_dependents(
        self, request_id: str, before_us: int
    ) -> tuple[str, ...]:
        """Return queued dependents reserved before their predecessor."""
        return _replan.queued_causal_dependents(self, request_id, before_us)

    def replan_projection_request_ids(
        self,
        request_id: str,
        expected_queue_generation: int | None = None,
    ) -> tuple[str, ...]:
        return _replan.replan_projection_request_ids(self, request_id, expected_queue_generation)

    def fail_replan(
        self,
        request_id: str,
        failed_at_us: int,
        reason: str,
        *,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
    ) -> tuple[RuntimeRequestTicket, tuple[str, ...]]:
        """Terminalize a failed queued replan without stranding followers."""
        return _replan.fail_replan(
            self,
            request_id,
            failed_at_us,
            reason,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
        )

    def prepare_priority_compaction_followers(
        self,
        request_id: str,
        at_us: int,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        memory_owner_tokens: Callable[
            [str], tuple[str, ...]
        ] | None = None,
        *,
        expected_queue_generation: int | None = None,
    ) -> tuple[tuple[str, str], ...]:
        """Detach later conflicting attempts without exposing waiters."""
        return _compaction.prepare_priority_compaction_followers(
            self,
            request_id,
            at_us,
            cancel_owner,
            release_memory,
            memory_owner_tokens,
            expected_queue_generation=expected_queue_generation,
        )

    def priority_compaction_followers(
        self,
        request_id: str,
        expected_queue_generation: int | None = None,
    ) -> tuple[str, ...]:
        """Inspect later live reservations without changing them."""
        return _compaction.priority_compaction_followers(
            self,
            request_id,
            expected_queue_generation,
        )

    def promote_priority_compaction_follower(
        self,
        request_id: str,
        expected_ticket_id: str,
        observed_at_us: int,
        cancel_owner: Callable[
            [str, int], tuple[str, ...]
        ] | None = None,
        release_memory: Callable[
            [str], tuple[str, ...]
        ] | None = None,
        memory_owner_tokens: Callable[
            [str], tuple[str, ...]
        ] | None = None,
    ) -> RuntimeRequestTicket:
        """Publish one detached follower's ordered replan receipt."""
        return _compaction.promote_priority_compaction_follower(
            self,
            request_id,
            expected_ticket_id,
            observed_at_us,
            cancel_owner,
            release_memory,
            memory_owner_tokens,
        )

    def priority_compaction_order(
        self, request_ids: Sequence[str]
    ) -> tuple[str, ...]:
        """Return detached followers in immutable dispatch order."""
        return _compaction.priority_compaction_order(self, request_ids)

    def cancel(
        self,
        request_id: str,
        at_us: int,
        reason: str,
        *,
        cancel_owner: Callable[[str, int], tuple[str, ...]],
        release_memory: Callable[[str], tuple[str, ...]] | None = None,
        terminal_is_noop: bool = False,
    ) -> tuple[str, ...]:
        return _completion.cancel(
            self,
            request_id,
            at_us,
            reason,
            cancel_owner=cancel_owner,
            release_memory=release_memory,
            terminal_is_noop=terminal_is_noop,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "dispatch_queue": self.queue.snapshot(),
            "quarantined_resources": sorted(self.quarantined_resources),
            "quarantined_routes": sorted(self._quarantined_routes),
            **({} if not self._device_quarantines else {
                "quarantined_devices": {
                    device_id: {**row, "resource_ids": list(row["resource_ids"])}
                    for device_id, row in sorted(
                        self._device_quarantines.items()
                    )
                },
            }),
            "route_uncertainty": {
                route_id: dict(self._route_uncertainty[route_id])
                for route_id in sorted(self._route_uncertainty)
            },
            "cancellation_audit": copy.deepcopy(
                self._cancellation_audit
            ),
            "capacity_releases": {
                request_id: receipt.actual_end_us
                for request_id, receipt in sorted(
                    self._capacity_releases.items()
                )
            },
            "schema": "research-scheduler-runtime-controller-v1",
            "tickets": {
                request_id: {
                    "dispatch_state": ticket.dispatch_state,
                    "lease_status": ticket.lease_status,
                    "prediction_status": ticket.prediction_status,
                    "route_id": ticket.decision.route_id,
                    "ticket_id": ticket.ticket_id,
                }
                for request_id, ticket in sorted(self._tickets.items())
            },
        }
