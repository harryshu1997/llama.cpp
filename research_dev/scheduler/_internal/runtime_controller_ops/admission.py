"""RuntimeController admission operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ..online_placement import OnlinePlacementReceipt
from ..policy import Decision, Request
from ..runtime_admission import RuntimeRequestObservation
from ..runtime_cost import RuntimeCostEstimateSet, RuntimeExecutorBinding, RuntimeModelArtifact
from ..runtime_queue import RuntimeQueueError
from ..runtime_decode_cohort import RuntimeDecodeCohortBinding
from ..runtime_plan import RuntimeExecutionPlan, RuntimeTransitionReceipt
from ..runtime_resources import RuntimeMemoryReservation, RuntimeResidencyProjectionToken
from ..request_contracts.common import RuntimeControllerError, RUNTIME_TERMINAL_STATES
from ..request_contracts.ticket import RuntimeRequestTicket
from .dispatch import (
    account_dispatch_precedence,
    account_retained_order,
    pending_dispatch_precedence,
    pending_retained_order,
)


def effective_bindings(
    controller, bindings: Sequence[RuntimeExecutorBinding]
) -> tuple[RuntimeExecutorBinding, ...]:
    result = []
    for binding in bindings:
        reasons = []
        if binding.route_id in controller._quarantined_routes:
            reasons.append("ROUTE_QUARANTINED")
        if controller.resources_are_quarantined(binding.resource_ids):
            reasons.append("RESOURCE_QUARANTINED")
        if reasons:
            binding = replace(
                binding,
                ready=False,
                queueable=False,
                eligibility_reasons=tuple(sorted(set(
                    binding.eligibility_reasons + tuple(reasons)
                ))),
            )
        result.append(binding)
    return tuple(result)


def admit(
    controller,
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
    with controller._lock:
        if (
            residency_order_barrier is not None
            and type(residency_order_barrier) is not bool
        ):
            raise RuntimeControllerError(
                "runtime residency order barrier is invalid"
            )
        if residency_hysteresis_key is not None and (
            type(residency_hysteresis_key) is not str
            or not residency_hysteresis_key
        ):
            raise RuntimeControllerError(
                "runtime residency hysteresis key is invalid"
            )
        attempt = controller._next_attempt_by_request.get(request.request_id, 0)
        ticket = RuntimeRequestTicket(
            ticket_id=f"{request.request_id}:attempt:{attempt}",
            attempt_index=attempt,
            request=request,
            model=model,
            runtime_observation=runtime_observation,
            cost_estimates=estimates,
            decision=decision,
            binding=binding,
            executor_bindings=tuple(executor_bindings),
            online_placement_receipt=online_receipt,
            dispatch_state="QUEUED",
            dispatch_receipt=None,
            memory_demands=binding.memory_demands,
            memory_reservation_status=(
                "RESERVED_ATOMIC"
                if memory_reservations
                else "NOT_REQUIRED_RESIDENT"
            ),
            final_reserved_until_us={
                lease.token: lease.reserved_until_us
                for lease in decision.leases
            },
            prediction_status="PENDING",
            lease_status="RESERVED",
            memory_reservations=tuple(memory_reservations),
            execution_plan=execution_plan,
            planning_profile_sha256=planning_profile_sha256,
            previous_ticket_id=previous_ticket_id,
            failure_reason=failure_reason,
            transition_status=(
                "PENDING"
                if execution_plan is not None
                and execution_plan.transitions
                else "NOT_REQUIRED"
            ),
            previous_transition_receipts=tuple(
                previous_transition_receipts
            ),
            selection_mode=selection_mode,
            decode_cohort=decode_cohort,
            residency_projection_token=residency_projection_token,
            phone_layout_generation=phone_layout_generation,
        )
        precede_request_ids = pending_dispatch_precedence(
            controller, request.request_id
        )
        retained_order = pending_retained_order(
            controller, request.request_id
        )
        try:
            controller.queue.admit(
                decision,
                admitted_at_us,
                decode_cohort=decode_cohort,
                residency_transition_barrier=bool(
                    execution_plan is not None
                    and execution_plan.transitions
                    if residency_order_barrier is None
                    else residency_order_barrier
                ),
                precede_request_ids=precede_request_ids,
                **(
                    {} if retained_order is None
                    else {"retained_order": retained_order}
                ),
                **(
                    {} if residency_hysteresis_key is None
                    else {"residency_hysteresis_key": residency_hysteresis_key}
                ),
            )
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        account_dispatch_precedence(
            controller, request.request_id, ticket.ticket_id
        )
        if retained_order is not None:
            account_retained_order(
                controller, request.request_id, ticket.ticket_id
            )
        controller._next_attempt_by_request[request.request_id] = attempt + 1
        controller._tickets[request.request_id] = ticket
        controller._ticket_history.append(ticket)
        return ticket


def bind_decode_cohort(
    controller,
    request_id: str,
    binding: RuntimeDecodeCohortBinding,
) -> RuntimeRequestTicket:
    """Refresh nonterminal ticket membership after cohort formation."""
    if not isinstance(binding, RuntimeDecodeCohortBinding):
        raise RuntimeControllerError(
            "runtime decode cohort binding is invalid"
        )
    with controller._lock:
        ticket = controller.ticket(request_id)
        if ticket.dispatch_state in RUNTIME_TERMINAL_STATES:
            raise RuntimeControllerError(
                "runtime terminal ticket cohort is immutable"
            )
        if (
            request_id not in binding.member_request_ids
            or (
                ticket.decode_cohort is not None
                and ticket.decode_cohort.cohort_id != binding.cohort_id
            )
        ):
            raise RuntimeControllerError(
                "runtime decode cohort identity differs"
            )
        updated = replace(ticket, decode_cohort=binding)
        try:
            controller.queue.bind_decode_cohort(request_id, binding)
        except RuntimeQueueError as exc:
            raise RuntimeControllerError(str(exc)) from exc
        controller._tickets[request_id] = updated
        return updated
