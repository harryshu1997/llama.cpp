"""Convert raw physical observations into hash-bound scheduler receipts."""

from __future__ import annotations

from .._internal.runtime_plan import (
    RuntimeExecutionReceipt,
    RuntimeTransitionReceipt,
)
from .contracts import (
    PhysicalAdapterError,
    RawExecutionObservation,
    RawTransitionObservation,
)
from .ticket import PhysicalExecutionCommand, PhysicalTransitionCommand


def transition_receipt_from_observation(
    command: PhysicalTransitionCommand,
    observation: RawTransitionObservation,
) -> RuntimeTransitionReceipt:
    if not isinstance(command, PhysicalTransitionCommand) or not isinstance(
        observation, RawTransitionObservation
    ):
        raise PhysicalAdapterError("physical transition result is invalid")
    transition = command.transition
    participant = command.participant
    expected_evictions = tuple(sorted({
        row.artifact_sha256 for row in transition.evictions
    }))
    if (
        observation.status == "COMPLETED"
        and observation.evicted_artifact_sha256s != expected_evictions
    ):
        raise PhysicalAdapterError(
            "physical transition eviction receipt differs"
        )
    energy = observation.energy
    return RuntimeTransitionReceipt(
        ticket_id=command.ticket_id,
        request_id=command.request_id,
        artifact_sha256=command.artifact_sha256,
        operator_plan_sha256=command.operator_plan_sha256,
        transition_id=transition.transition_id,
        executor_id=participant.executor_id,
        endpoint=participant.endpoint,
        device_id=transition.device_id,
        source_state=transition.source_state,
        target_state=transition.target_state,
        resource_ids=transition.resource_ids,
        resource_slots=transition.resource_slots,
        started_us=observation.started_us,
        finished_us=observation.finished_us,
        status=observation.status,
        evicted_artifact_sha256s=(
            expected_evictions
            if observation.status == "COMPLETED"
            else observation.evicted_artifact_sha256s
        ),
        energy_boundary_id=(
            None if energy is None else energy.energy_boundary_id
        ),
        energy_attribution_kind=(
            None if energy is None else energy.attribution_kind
        ),
        fleet_energy_uj_by_domain=(
            {} if energy is None else energy.fleet_energy_uj_by_domain
        ),
        transfer_energy_uj_by_link=(
            {} if energy is None else energy.transfer_energy_uj_by_link
        ),
        measurement_evidence_ids=(
            () if energy is None else energy.measurement_evidence_ids
        ),
        energy_estimation_metadata=(
            {} if energy is None else energy.estimation_metadata
        ),
    )


def execution_receipt_from_observation(
    command: PhysicalExecutionCommand,
    observation: RawExecutionObservation,
) -> RuntimeExecutionReceipt:
    if not isinstance(command, PhysicalExecutionCommand) or not isinstance(
        observation, RawExecutionObservation
    ):
        raise PhysicalAdapterError("physical execution result is invalid")
    energy = observation.energy
    return RuntimeExecutionReceipt(
        ticket_id=command.ticket_id,
        request_id=command.request_id,
        artifact_sha256=command.artifact_sha256,
        operator_plan_sha256=command.operator_plan_sha256,
        executor_id=command.executor_id,
        endpoint=command.endpoint,
        operator_plan_protocol=command.operator_plan_protocol,
        participant_executor_ids=tuple(
            row.executor_id for row in command.participants
        ),
        started_us=observation.started_us,
        finished_us=observation.finished_us,
        output_sha256=observation.output_sha256,
        status="COMPLETED",
        energy_boundary_id=(
            None if energy is None else energy.energy_boundary_id
        ),
        energy_attribution_kind=(
            None if energy is None else energy.attribution_kind
        ),
        fleet_energy_uj_by_domain=(
            {} if energy is None else energy.fleet_energy_uj_by_domain
        ),
        transfer_energy_uj_by_link=(
            {} if energy is None else energy.transfer_energy_uj_by_link
        ),
        measurement_evidence_ids=(
            () if energy is None else energy.measurement_evidence_ids
        ),
        energy_estimation_metadata=(
            {} if energy is None else energy.estimation_metadata
        ),
        energy_scope=observation.energy_scope,
    )
