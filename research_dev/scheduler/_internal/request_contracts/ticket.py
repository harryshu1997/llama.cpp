"""Runtime request tickets, validation and terminal receipts: ticket."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..online_placement import OnlinePlacementReceipt
from ..policy import Decision, LeaseRecord, Request, decision_to_json
from ..runtime_admission import RuntimeRequestObservation
from ..runtime_cost import (
    RuntimeCostEstimateSet,
    RuntimeExecutorBinding,
    RuntimeMemoryDemand,
    RuntimeModelArtifact,
)
from ..runtime_queue import RuntimeDispatchReceipt
from ..runtime_decode_cohort import RuntimeDecodeCohortBinding
from ..runtime_plan import RuntimeExecutionPlan, RuntimeExecutionReceipt, RuntimeTransitionReceipt
from ..runtime_resources import RuntimeMemoryReservation, RuntimeResidencyProjectionToken
from .common import (
    RUNTIME_DISPATCH_STATES,
    RUNTIME_LEASE_STATES,
    RUNTIME_PREDICTION_STATES,
    RUNTIME_REQUEST_TICKET_SCHEMA,
    RUNTIME_SELECTION_MODES,
    RUNTIME_TRANSITION_STATES,
    RuntimeControllerError,
    _integer,
    _text,
)


def _request_to_json(request: Request) -> dict[str, object]:
    return {
        "arrival_us": request.arrival_us,
        "deadline_us": request.deadline_us,
        "features": dict(sorted(request.features.items())),
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "quality_requirement": request.quality_requirement,
        "request_id": request.request_id,
        "workload_id": request.workload_id,
    }


def _validate_execution_receipt(
    ticket: "RuntimeRequestTicket",
    receipt: RuntimeExecutionReceipt,
    actual_end_us: int,
) -> None:
    if not isinstance(receipt, RuntimeExecutionReceipt):
        raise RuntimeControllerError(
            "runtime physical execution receipt is invalid"
        )
    plan = ticket.execution_plan
    binding = ticket.binding
    if plan is None:
        raise RuntimeControllerError(
            "runtime physical execution receipt lacks a plan"
        )
    expected_participants = tuple(sorted(
        row.executor_id for row in binding.participants
    ))
    if (
        receipt.ticket_id != ticket.ticket_id
        or receipt.request_id != ticket.request.request_id
        or receipt.artifact_sha256 != ticket.model.artifact_sha256
        or receipt.operator_plan_sha256 != plan.plan_sha256
        or receipt.executor_id != binding.executor_id
        or receipt.endpoint != binding.endpoint
        or receipt.operator_plan_protocol != binding.operator_plan_protocol
        or receipt.participant_executor_ids != expected_participants
        or receipt.started_us < ticket.decision.start_us
        or receipt.finished_us != actual_end_us
    ):
        raise RuntimeControllerError(
            "runtime physical execution receipt differs from the decision"
        )


def _validate_ticket_identity(
    ticket: RuntimeRequestTicket,
) -> tuple[tuple[RuntimeExecutorBinding, ...], object]:
    _text("runtime ticket_id", ticket.ticket_id)
    _integer("runtime ticket attempt_index", ticket.attempt_index)
    if not isinstance(ticket.request, Request):
        raise RuntimeControllerError("runtime ticket request is invalid")
    if ticket.selection_mode not in RUNTIME_SELECTION_MODES:
        raise RuntimeControllerError(
            "runtime ticket selection mode is invalid"
        )
    if not isinstance(ticket.model, RuntimeModelArtifact):
        raise RuntimeControllerError("runtime ticket model is invalid")
    if not isinstance(ticket.runtime_observation, RuntimeRequestObservation):
        raise RuntimeControllerError("runtime ticket observation is invalid")
    if not isinstance(ticket.cost_estimates, RuntimeCostEstimateSet):
        raise RuntimeControllerError("runtime ticket estimates are invalid")
    if not isinstance(ticket.decision, Decision):
        raise RuntimeControllerError("runtime ticket decision is invalid")
    if not isinstance(ticket.binding, RuntimeExecutorBinding):
        raise RuntimeControllerError("runtime ticket binding is invalid")
    bindings = tuple(ticket.executor_bindings)
    if (
        not bindings
        or any(not isinstance(row, RuntimeExecutorBinding) for row in bindings)
        or len({row.route_id for row in bindings}) != len(bindings)
    ):
        raise RuntimeControllerError(
            "runtime ticket executor bindings are invalid"
        )
    if {row.route_id: row for row in bindings}.get(
        ticket.binding.route_id
    ) != ticket.binding:
        raise RuntimeControllerError(
            "runtime ticket selected executor is not in its candidates"
        )
    if (
        ticket.request.request_id != ticket.decision.request_id
        or ticket.request.workload_id != ticket.decision.workload_id
        or ticket.cost_estimates.request_id != ticket.request.request_id
        or ticket.cost_estimates.workload_id != ticket.request.workload_id
        or ticket.cost_estimates.model != ticket.model
        or ticket.binding.model_id != ticket.model.model_id
        or ticket.binding.artifact_sha256 != ticket.model.artifact_sha256
        or ticket.binding.artifact_bytes != ticket.model.artifact_bytes
        or ticket.binding.route_id != ticket.decision.route_id
    ):
        raise RuntimeControllerError("runtime ticket identity differs")
    selected = next((
        row for row in ticket.cost_estimates.estimates
        if row.route_id == ticket.decision.route_id
    ), None)
    if (
        selected is None
        or not selected.admitted
        or selected.executor_id != ticket.binding.executor_id
        or selected.memory_demands != tuple(ticket.memory_demands)
    ):
        raise RuntimeControllerError(
            "runtime ticket selected binding is not admitted"
        )
    return bindings, selected


def _validate_ticket_lifecycle(
    ticket: RuntimeRequestTicket,
    selected: object,
) -> tuple[RuntimeMemoryReservation, ...]:
    reservations = tuple(ticket.memory_reservations)
    if (
        any(not isinstance(row, RuntimeMemoryReservation) for row in reservations)
        or len({row.token for row in reservations}) != len(reservations)
    ):
        raise RuntimeControllerError(
            "runtime ticket memory reservations are invalid"
        )
    if sum(row.reserved_bytes for row in reservations) != (
        selected.additional_bytes
    ):
        raise RuntimeControllerError(
            "runtime ticket memory reservation differs from demand"
        )
    if ticket.online_placement_receipt is not None and (
        not isinstance(ticket.online_placement_receipt, OnlinePlacementReceipt)
        or ticket.online_placement_receipt.decision != ticket.decision
    ):
        raise RuntimeControllerError("runtime ticket online receipt differs")
    if ticket.dispatch_state not in RUNTIME_DISPATCH_STATES:
        raise RuntimeControllerError("runtime ticket dispatch state is invalid")
    if ticket.dispatch_receipt is not None:
        if (
            not isinstance(ticket.dispatch_receipt, RuntimeDispatchReceipt)
            or ticket.dispatch_receipt.request_id != ticket.request.request_id
            or ticket.dispatch_receipt.route_id != ticket.decision.route_id
            or ticket.dispatch_receipt.status != ticket.dispatch_state
        ):
            raise RuntimeControllerError(
                "runtime ticket dispatch receipt differs"
            )
    elif ticket.dispatch_state in {"ACQUIRED", "REPLAN_REQUIRED"}:
        raise RuntimeControllerError(
            "runtime ticket dispatch state lacks a receipt"
        )
    expected_memory_status = (
        "NOT_REQUIRED_RESIDENT" if not reservations else "RESERVED_ATOMIC"
    )
    allowed_memory_statuses = {expected_memory_status}
    if reservations:
        allowed_memory_statuses.update({"CANCELLED", "RELEASED"})
    if ticket.memory_reservation_status not in allowed_memory_statuses:
        raise RuntimeControllerError(
            "runtime ticket memory reservation status is invalid"
        )
    terminal_memory_status = {
        "CANCELLED": "CANCELLED",
        "COMPLETED": "RELEASED",
        "FAILED": "CANCELLED",
    }.get(ticket.dispatch_state)
    if (
        reservations
        and terminal_memory_status is not None
        and ticket.memory_reservation_status != terminal_memory_status
    ):
        raise RuntimeControllerError("runtime terminal memory status is invalid")
    return reservations


def _validate_ticket_execution_plan(ticket: RuntimeRequestTicket) -> None:
    if ticket.execution_plan is not None:
        if (
            not isinstance(ticket.execution_plan, RuntimeExecutionPlan)
            or ticket.execution_plan.route_id != ticket.decision.route_id
            or ticket.execution_plan.plan_sha256
                != ticket.binding.operator_plan_sha256
            or ticket.execution_plan.memory_demands
                != tuple(ticket.memory_demands)
        ):
            raise RuntimeControllerError(
                "runtime ticket execution plan differs"
            )
        contract = ticket.execution_plan.execution_contract
        if contract.execution_mode != "desktop":
            participants = tuple(
                row for row in ticket.binding.participants
                if row.device_id == contract.phone_device_id
            )
            if (
                len(participants) != 1
                or participants[0].endpoint != contract.phone_endpoint
                or ticket.execution_plan.adapter_parameters.get(
                    "phone_device_id"
                ) != contract.phone_device_id
            ):
                raise RuntimeControllerError(
                    "runtime ticket phone endpoint differs"
                )
    elif ticket.binding.operator_plan_sha256 is not None:
        raise RuntimeControllerError(
            "runtime ticket binding lacks its execution plan"
        )


def _validate_ticket_transition_receipts(
    ticket: RuntimeRequestTicket,
) -> tuple[RuntimeTransitionReceipt, ...]:
    receipts = tuple(ticket.transition_receipts)
    if ticket.transition_status not in RUNTIME_TRANSITION_STATES:
        raise RuntimeControllerError(
            "runtime ticket transition status is invalid"
        )
    transitions = (
        () if ticket.execution_plan is None
        else ticket.execution_plan.transitions
    )
    if not transitions:
        if ticket.transition_status != "NOT_REQUIRED" or receipts:
            raise RuntimeControllerError(
                "runtime ticket has unexpected transition state"
            )
        return receipts
    if ticket.transition_status == "NOT_REQUIRED":
        raise RuntimeControllerError(
            "runtime ticket omits required transitions"
        )
    expected = {row.transition_id: row for row in transitions}
    participants = {
        row.device_id: row for row in ticket.binding.participants
    }
    if (
        any(not isinstance(row, RuntimeTransitionReceipt) for row in receipts)
        or len({row.transition_id for row in receipts}) != len(receipts)
        or {row.transition_id for row in receipts}
            != {row.transition_id for row in transitions[:len(receipts)]}
        or (ticket.transition_status == "COMPLETED" and len(receipts) != len(transitions))
        or (ticket.transition_status == "PENDING" and len(receipts) == len(transitions))
    ):
        raise RuntimeControllerError(
            "runtime transition receipts are incomplete"
        )
    for receipt in receipts:
        transition = expected[receipt.transition_id]
        if transition.executor_id == ticket.binding.executor_id:
            expected_executor_id = ticket.binding.executor_id
            expected_endpoint = ticket.binding.endpoint
        else:
            participant = participants.get(transition.device_id)
            expected_executor_id = (
                None if participant is None else participant.executor_id
            )
            expected_endpoint = None if participant is None else participant.endpoint
            if (
                transition.executor_id is not None
                and transition.executor_id != expected_executor_id
            ):
                expected_executor_id = None
        if (
            receipt.ticket_id != ticket.ticket_id
            or receipt.request_id != ticket.request.request_id
            or receipt.artifact_sha256 != ticket.model.artifact_sha256
            or receipt.operator_plan_sha256 != ticket.execution_plan.plan_sha256
            or expected_executor_id is None
            or receipt.executor_id != expected_executor_id
            or receipt.endpoint != expected_endpoint
            or receipt.device_id != transition.device_id
            or receipt.source_state != transition.source_state
            or receipt.target_state != transition.target_state
            or receipt.resource_ids != transition.resource_ids
            or receipt.resource_slots != transition.resource_slots
        ):
            raise RuntimeControllerError(
                "runtime transition receipt differs from plan"
            )
    failed = any(row.status == "FAILED" for row in receipts)
    if failed != (ticket.transition_status == "FAILED"):
        raise RuntimeControllerError(
            "runtime transition receipt status differs"
        )
    ordered = tuple(next(row for row in receipts if row.transition_id == transition.transition_id)
                    for transition in transitions[:len(receipts)])
    if failed and any(row.status == "FAILED" for row in ordered[:-1]):
        raise RuntimeControllerError(
            "runtime transition continued after failure"
        )
    return ordered


def _validate_ticket_terminal_state(
    ticket: RuntimeRequestTicket,
) -> dict[str, int]:
    if ticket.planning_profile_sha256 is not None:
        value = _text(
            "runtime ticket planning profile hash",
            ticket.planning_profile_sha256,
        )
        if (
            not value.startswith("sha256:")
            or len(value) != 71
            or value != ticket.cost_estimates.planning_profile_sha256
        ):
            raise RuntimeControllerError(
                "runtime ticket planning profile hash differs"
            )
    final = {
        _text("runtime ticket lease token", token): _integer(
            "runtime ticket lease end", value
        )
        for token, value in ticket.final_reserved_until_us.items()
    }
    initial = {
        lease.token: lease.reserved_until_us for lease in ticket.decision.leases
    }
    if set(final) != set(initial) or any(
        final[token] < value for token, value in initial.items()
    ):
        raise RuntimeControllerError("runtime ticket lease coverage is invalid")
    if ticket.prediction_status not in RUNTIME_PREDICTION_STATES:
        raise RuntimeControllerError(
            "runtime ticket prediction status is invalid"
        )
    if ticket.lease_status not in RUNTIME_LEASE_STATES:
        raise RuntimeControllerError("runtime ticket lease status is invalid")
    if ticket.previous_ticket_id is not None:
        _text("runtime previous_ticket_id", ticket.previous_ticket_id)
    if ticket.failure_reason is not None:
        _text("runtime failure_reason", ticket.failure_reason)
    if ticket.actual_end_us is not None:
        _integer("runtime ticket actual_end_us", ticket.actual_end_us)
    if ticket.execution_receipt is not None:
        if ticket.dispatch_state != "COMPLETED" or ticket.actual_end_us is None:
            raise RuntimeControllerError(
                "runtime execution receipt is not terminal"
            )
        _validate_execution_receipt(
            ticket, ticket.execution_receipt, ticket.actual_end_us
        )
    elif ticket.dispatch_state == "COMPLETED" and ticket.execution_plan is not None:
        raise RuntimeControllerError(
            "automated runtime completion lacks physical proof"
        )
    return final


def _validate_ticket_runtime_bindings(ticket: RuntimeRequestTicket) -> None:
    if ticket.decode_cohort is not None and (
        not isinstance(ticket.decode_cohort, RuntimeDecodeCohortBinding)
        or ticket.request.request_id not in ticket.decode_cohort.member_request_ids
        or set(ticket.decode_cohort.shared_lease_tokens)
            != {row.token for row in ticket.decision.leases}
    ):
        raise RuntimeControllerError("runtime ticket decode cohort differs")
    if ticket.residency_projection_token is not None:
        token = ticket.residency_projection_token
        if (
            not isinstance(token, RuntimeResidencyProjectionToken)
            or ticket.execution_plan is None
            or ticket.execution_plan.adapter_parameters.get(
                "phone_shard_set_geometry_sha256"
            ) != token.target_geometry_sha256
            or ticket.decision.start_us < token.ready_at_us
            or ticket.ticket_id == token.predecessor_ticket_id
        ):
            raise RuntimeControllerError(
                "runtime ticket residency projection differs"
            )
    has_phone_shards = bool(
        ticket.execution_plan is not None
        and (ticket.execution_plan.execution_contract.phone_shards
             or ticket.execution_plan.execution_contract.remote_resident_ffn is not None)
    )
    if has_phone_shards:
        if (
            type(ticket.phone_layout_generation) is not int
            or ticket.phone_layout_generation < 1
        ):
            raise RuntimeControllerError(
                "runtime ticket lacks its phone layout generation"
            )
        if (
            ticket.residency_projection_token is not None
            and ticket.residency_projection_token.layout_generation
                != ticket.phone_layout_generation
        ):
            raise RuntimeControllerError(
                "runtime ticket phone layout generation differs"
            )
    elif ticket.phone_layout_generation is not None:
        raise RuntimeControllerError(
            "desktop runtime ticket carries a phone layout generation"
        )


def _normalize_runtime_ticket(
    ticket: RuntimeRequestTicket,
    bindings: tuple[RuntimeExecutorBinding, ...],
    reservations: tuple[RuntimeMemoryReservation, ...],
    transition_receipts: tuple[RuntimeTransitionReceipt, ...],
    final: Mapping[str, int],
) -> None:
    object.__setattr__(ticket, "memory_demands", tuple(ticket.memory_demands))
    object.__setattr__(
        ticket,
        "memory_reservations",
        tuple(sorted(reservations, key=lambda row: row.token)),
    )
    object.__setattr__(
        ticket,
        "executor_bindings",
        tuple(sorted(bindings, key=lambda row: row.route_id)),
    )
    object.__setattr__(
        ticket,
        "transition_receipts",
        transition_receipts,
    )
    previous = tuple(ticket.previous_transition_receipts)
    if any(not isinstance(row, RuntimeTransitionReceipt) for row in previous):
        raise RuntimeControllerError(
            "previous runtime transition receipt is invalid"
        )
    if previous and (
        ticket.previous_ticket_id is None
        or any(
            row.ticket_id != ticket.previous_ticket_id
            or row.request_id != ticket.request.request_id
            or row.artifact_sha256 != ticket.model.artifact_sha256
            for row in previous
        )
    ):
        raise RuntimeControllerError(
            "previous runtime transition receipt identity differs"
        )
    object.__setattr__(
        ticket,
        "previous_transition_receipts",
        tuple(sorted(previous, key=lambda row: row.transition_id)),
    )
    object.__setattr__(
        ticket,
        "final_reserved_until_us",
        MappingProxyType(dict(sorted(final.items()))),
    )


def _initialize_runtime_request_ticket(ticket: RuntimeRequestTicket) -> None:
    bindings, selected = _validate_ticket_identity(ticket)
    reservations = _validate_ticket_lifecycle(ticket, selected)
    _validate_ticket_execution_plan(ticket)
    transition_receipts = _validate_ticket_transition_receipts(ticket)
    final = _validate_ticket_terminal_state(ticket)
    _validate_ticket_runtime_bindings(ticket)
    _normalize_runtime_ticket(
        ticket, bindings, reservations, transition_receipts, final
    )


@dataclass(frozen=True)
class RuntimeRequestTicket:
    ticket_id: str
    attempt_index: int
    request: Request
    model: RuntimeModelArtifact
    runtime_observation: RuntimeRequestObservation
    cost_estimates: RuntimeCostEstimateSet
    decision: Decision
    binding: RuntimeExecutorBinding
    executor_bindings: tuple[RuntimeExecutorBinding, ...]
    online_placement_receipt: OnlinePlacementReceipt | None
    dispatch_state: str
    dispatch_receipt: RuntimeDispatchReceipt | None
    memory_demands: tuple[RuntimeMemoryDemand, ...]
    memory_reservation_status: str
    final_reserved_until_us: Mapping[str, int]
    prediction_status: str
    lease_status: str
    memory_reservations: tuple[RuntimeMemoryReservation, ...] = ()
    execution_plan: RuntimeExecutionPlan | None = None
    planning_profile_sha256: str | None = None
    previous_ticket_id: str | None = None
    failure_reason: str | None = None
    actual_end_us: int | None = None
    transition_status: str = "NOT_REQUIRED"
    transition_receipts: tuple[RuntimeTransitionReceipt, ...] = ()
    previous_transition_receipts: tuple[RuntimeTransitionReceipt, ...] = ()
    execution_receipt: RuntimeExecutionReceipt | None = None
    selection_mode: str = "energy-aware"
    decode_cohort: RuntimeDecodeCohortBinding | None = None
    residency_projection_token: (
        RuntimeResidencyProjectionToken | None
    ) = None
    phone_layout_generation: int | None = None

    def __post_init__(self) -> None:
        _initialize_runtime_request_ticket(self)

    def to_json(self) -> dict[str, object]:
        result = {
            "actual_end_us": self.actual_end_us,
            "attempt_index": self.attempt_index,
            "binding": self.binding.to_json(),
            "cost_estimates": self.cost_estimates.to_json(),
            "decision": decision_to_json(self.decision),
            "dispatch_receipt": (
                None
                if self.dispatch_receipt is None
                else self.dispatch_receipt.to_json()
            ),
            "dispatch_state": self.dispatch_state,
            "executor_bindings": [
                row.to_json() for row in self.executor_bindings
            ],
            "failure_reason": self.failure_reason,
            "final_reserved_until_us": dict(self.final_reserved_until_us),
            "lease_status": self.lease_status,
            "memory_demands": [row.to_json() for row in self.memory_demands],
            "memory_reservation_status": self.memory_reservation_status,
            "model": self.model.to_json(),
            "online_placement_receipt": (
                None
                if self.online_placement_receipt is None
                else self.online_placement_receipt.to_json()
            ),
            "prediction_status": self.prediction_status,
            "previous_ticket_id": self.previous_ticket_id,
            "request": _request_to_json(self.request),
            "runtime_observation": self.runtime_observation.to_json(),
            "schema": RUNTIME_REQUEST_TICKET_SCHEMA,
            "selection_mode": self.selection_mode,
            "ticket_id": self.ticket_id,
        }
        if self.memory_reservations:
            result["memory_reservations"] = [
                row.to_json() for row in self.memory_reservations
            ]
        if self.execution_plan is not None:
            result["execution_plan"] = self.execution_plan.to_json()
        if self.planning_profile_sha256 is not None:
            result["planning_profile_sha256"] = self.planning_profile_sha256
        result["transition_status"] = self.transition_status
        if self.transition_receipts:
            result["transition_receipts"] = [
                row.to_json() for row in self.transition_receipts
            ]
        if self.previous_transition_receipts:
            result["previous_transition_receipts"] = [
                row.to_json() for row in self.previous_transition_receipts
            ]
        if self.execution_receipt is not None:
            result["execution_receipt"] = self.execution_receipt.to_json()
        if self.decode_cohort is not None:
            result["decode_cohort"] = self.decode_cohort.to_json()
        if self.residency_projection_token is not None:
            result["residency_projection_token"] = (
                self.residency_projection_token.to_json()
            )
        if self.phone_layout_generation is not None:
            result["phone_layout_generation"] = (
                self.phone_layout_generation
            )
        return result

    def activation_receipts(self) -> tuple[Mapping[str, int | str], ...]:
        return tuple(
            MappingProxyType({
                "lease_id": lease.lease_id,
                "predicted_end_us": lease.predicted_end_us,
                "reserved_until_us": lease.reserved_until_us,
                "resource_id": lease.resource_id,
                "token": lease.token,
            })
            for lease in self.decision.leases
        )

    @property
    def prepare_lease_tokens(self) -> frozenset[str]:
        if self.execution_plan is None:
            return frozenset()
        ids = {
            f"{self.decision.route_id}:prepare:{row.transition_id}:{resource_id}"
            for row in self.execution_plan.transitions
            for resource_id in row.resource_slots
        }
        return frozenset(
            row.token for row in self.decision.leases if row.lease_id in ids
        )

    @property
    def completed_prepare_lease_tokens(self) -> frozenset[str]:
        ids = {
            f"{self.decision.route_id}:prepare:{row.transition_id}:{resource_id}"
            for row in self.transition_receipts if row.status == "COMPLETED"
            for resource_id in row.resource_ids
        }
        return frozenset(row.token for row in self.decision.leases if row.lease_id in ids)

    @property
    def live_leases(self) -> tuple[LeaseRecord, ...]:
        released = self.completed_prepare_lease_tokens
        return tuple(row for row in self.decision.leases if row.token not in released)

    def transition_lease_by_resource(self, transition):
        result = {}
        for row in self.decision.leases:
            if (
                row.resource_id not in result
                or row.start_us < result[row.resource_id].start_us
            ):
                result[row.resource_id] = row
        for row in self.decision.leases:
            if row.lease_id == (
                f"{self.decision.route_id}:prepare:"
                f"{transition.transition_id}:{row.resource_id}"
            ):
                result[row.resource_id] = row
        return result
