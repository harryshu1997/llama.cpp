"""Translate one scheduler ticket into exact physical commands."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .._internal.adaptive_decode_contracts import AdaptiveDecodePolicy
from .._internal.offline_phone_residency import OfflinePhoneResidencyStage
from .._internal.plan_contracts.co_helpers import (
    PHONE_HELPERS_PARAMETER,
    phone_helper_layer_masks,
)
from .._internal.plan_contracts.common import RuntimePlanError
from .._internal.runtime_controller import RuntimeRequestTicket
from .._internal.runtime_plan import (
    PhoneSessionReplacementAuthorization,
    RuntimeExecutionContract,
    RuntimeHelperExecutionEnvelope,
    RuntimeTransitionPlan,
    phone_session_map_sha256,
)
from .contracts import PhysicalAdapterError


@dataclass(frozen=True)
class PhysicalParticipantCommand:
    executor_id: str
    device_id: str
    endpoint: str
    backend: str
    resource_ids: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device_id": self.device_id,
            "endpoint": self.endpoint,
            "executor_id": self.executor_id,
            "resource_ids": list(self.resource_ids),
        }


@dataclass(frozen=True)
class PhysicalTransitionCommand:
    ticket_id: str
    request_id: str
    artifact_sha256: str
    route_id: str
    operator_plan_sha256: str
    participant: PhysicalParticipantCommand
    transition: RuntimeTransitionPlan
    execution_contract: RuntimeExecutionContract
    adapter_parameters: Mapping[str, int | str]
    phone_layout_generation: int | None = None
    operator_plan_protocol: str | None = None
    operator_plan: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    decode_cohort: Mapping[str, object] | None = None
    selection_mode: str = "energy-aware"
    helper_only: bool = False
    helper_envelope: RuntimeHelperExecutionEnvelope | None = None
    replacement_authorization: (
        PhoneSessionReplacementAuthorization | None
    ) = None

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "adapter_parameters": dict(self.adapter_parameters),
            "execution_contract": self.execution_contract.to_json(),
            "operator_plan_sha256": self.operator_plan_sha256,
            "operator_plan_protocol": self.operator_plan_protocol,
            "operator_plan": dict(self.operator_plan),
            "participant": self.participant.to_json(),
            "phone_layout_generation": self.phone_layout_generation,
            "request_id": self.request_id,
            "route_id": self.route_id,
            "selection_mode": self.selection_mode,
            "helper_only": self.helper_only,
            "helper_envelope": (
                None
                if self.helper_envelope is None
                else self.helper_envelope.to_json()
            ),
            "replacement_authorization": (
                None
                if self.replacement_authorization is None
                else self.replacement_authorization.to_json()
            ),
            "ticket_id": self.ticket_id,
            "transition": self.transition.to_json(),
        }
        if self.decode_cohort is not None:
            result["decode_cohort"] = dict(self.decode_cohort)
        return result


@dataclass(frozen=True)
class PhysicalExecutionCommand:
    ticket_id: str
    request_id: str
    model_id: str
    artifact_sha256: str
    route_id: str
    executor_id: str
    endpoint: str
    operator_plan_protocol: str
    operator_plan_sha256: str
    planned_start_us: int
    planned_finish_us: int
    planned_finish_upper_us: int
    operator_plan: Mapping[str, object]
    participants: tuple[PhysicalParticipantCommand, ...]
    leases: tuple[Mapping[str, int | str], ...]
    memory_reservations: tuple[Mapping[str, int | str], ...]
    transitions: tuple[PhysicalTransitionCommand, ...]
    execution_contract: RuntimeExecutionContract
    adapter_parameters: Mapping[str, int | str]
    phone_layout_generation: int | None = None
    selection_mode: str = "energy-aware"
    decode_cohort: Mapping[str, object] | None = None
    helper_envelope: RuntimeHelperExecutionEnvelope | None = None
    helper_transitions: tuple[PhysicalTransitionCommand, ...] = ()

    def to_json(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "adapter_parameters": dict(self.adapter_parameters),
            "endpoint": self.endpoint,
            "execution_contract": self.execution_contract.to_json(),
            "executor_id": self.executor_id,
            "leases": [dict(row) for row in self.leases],
            "memory_reservations": [
                dict(row) for row in self.memory_reservations
            ],
            "model_id": self.model_id,
            "operator_plan": dict(self.operator_plan),
            "operator_plan_protocol": self.operator_plan_protocol,
            "operator_plan_sha256": self.operator_plan_sha256,
            "planned_finish_upper_us": self.planned_finish_upper_us,
            "planned_finish_us": self.planned_finish_us,
            "planned_start_us": self.planned_start_us,
            "participants": [row.to_json() for row in self.participants],
            "phone_layout_generation": self.phone_layout_generation,
            "request_id": self.request_id,
            "route_id": self.route_id,
            "selection_mode": self.selection_mode,
            "ticket_id": self.ticket_id,
            "transitions": [row.to_json() for row in self.transitions],
            "helper_envelope": (
                None
                if self.helper_envelope is None
                else self.helper_envelope.to_json()
            ),
            "helper_transitions": [
                row.to_json() for row in self.helper_transitions
            ],
        }
        if self.decode_cohort is not None:
            result["decode_cohort"] = dict(self.decode_cohort)
        return result


def validate_phone_session_replacement_command(
    command: PhysicalTransitionCommand,
) -> None:
    """Fail before I/O if a partial command differs from its assignment."""

    if any(
        row.session_generation == 0
        for row in getattr(command.transition, "phone_shards", ())
    ):
        raise PhysicalAdapterError(
            "physical phone command contains session generation zero"
        )
    changed = tuple(getattr(
        command.transition, "changed_phone_session_ids", ()
    ))
    authorization = getattr(command, "replacement_authorization", None)
    has_session_eviction = bool(
        len(changed) == 1
        and any(
            eviction.session_id == changed[0]
            for eviction in getattr(command.transition, "evictions", ())
        )
    )
    if authorization is None and not has_session_eviction:
        return
    if len(changed) != 1:
        raise PhysicalAdapterError(
            "non-partial phone command carries replacement authority"
        )
    if (
        not isinstance(
            authorization, PhoneSessionReplacementAuthorization
        )
        or changed != (authorization.selected_session_id,)
    ):
        raise PhysicalAdapterError(
            "partial phone command lacks exact replacement authority"
        )
    target = tuple(command.transition.phone_shards)
    target_by_session = {row.session_id: row for row in target}
    selected = target_by_session.get(authorization.selected_session_id)
    if (
        len(target_by_session) != len(target)
        or selected is None
        or selected.session_generation != authorization.target_generation
        or phone_session_map_sha256(target)
            != authorization.target_layout_hash
    ):
        raise PhysicalAdapterError(
            "partial phone command target differs from replacement authority"
        )


def static_decode_policy(
    command: PhysicalExecutionCommand,
) -> AdaptiveDecodePolicy | None:
    parameters = command.adapter_parameters
    plan = command.operator_plan
    if (
        command.execution_contract.execution_mode != "static-split"
        or plan.get("assisted_operator_kind") != "ffn"
        or parameters.get("ffn_assistance_phase") != "decode"
        or parameters.get("ffn_runtime_control_protocol")
            != "decode-boundary-v1"
        or type(parameters.get("phone_device_id")) is not str
    ):
        return None
    layer_mask = parameters.get("ffn_selected_layer_mask")
    columns = parameters.get("ffn_selected_columns")
    resident_columns = parameters.get("ffn_resident_columns", columns)
    baseline_executor_id = plan.get("baseline_executor_id")
    desktop_placement = plan.get("desktop_placement_sha256")
    resources = plan.get("resource_ids")
    split_fraction = plan.get("split_fraction_ppm")
    if (
        type(layer_mask) is not int
        or layer_mask <= 0
        or type(columns) is not int
        or columns <= 0
        or type(resident_columns) is not int
        or resident_columns < columns
        or type(baseline_executor_id) is not str
        or type(desktop_placement) is not str
        or type(resources) is not list
        or any(type(value) is not str for value in resources)
        or type(split_fraction) is not int
    ):
        raise PhysicalAdapterError(
            "static FFN runtime policy is absent from the ticket"
        )
    if split_fraction == 0:
        if columns != resident_columns:
            raise PhysicalAdapterError(
                "static FFN offload width differs from residency"
            )
        split_fraction = 1_000_000
    if columns * 1_000_000 != resident_columns * split_fraction:
        raise PhysicalAdapterError(
            "static FFN split fraction differs from its width"
        )
    if split_fraction != (
        command.execution_contract.initial_split_fraction_ppm
    ):
        raise PhysicalAdapterError(
            "static FFN split differs from the execution contract"
        )
    layer_indices = tuple(
        index for index in range(64)
        if layer_mask & (1 << index)
    )
    operator_plan_sha256 = command.operator_plan_sha256
    if command.decode_cohort is not None:
        common_policy = command.decode_cohort.get(
            "common_policy_sha256"
        )
        if (
            type(common_policy) is not str
            or not common_policy.startswith("sha256:")
            or len(common_policy) != 71
        ):
            raise PhysicalAdapterError(
                "static FFN cohort policy identity is invalid"
            )
        operator_plan_sha256 = common_policy
    return AdaptiveDecodePolicy(
        route_id=command.route_id,
        executor_id=command.executor_id,
        operator_plan_sha256=operator_plan_sha256,
        desktop_parent_route_id=baseline_executor_id,
        desktop_placement_sha256=desktop_placement,
        layer_indices=layer_indices,
        layer_mask=layer_mask,
        columns=columns,
        split_fraction_ppm=split_fraction,
        resource_ids=tuple(resources),
    )


def _command_phone_shards(
    command: PhysicalExecutionCommand,
) -> tuple[object, ...]:
    contract = command.execution_contract
    helper = command.helper_envelope
    phone_shards = tuple(contract.phone_shards) + tuple(
        shard
        for transition in command.transitions
        for shard in transition.transition.phone_shards
    )
    if helper is not None:
        phone_shards += tuple(
            helper.helper_plan.execution_contract.phone_shards
        ) + tuple(
            shard
            for transition in helper.helper_plan.transitions
            for shard in transition.phone_shards
        )
    phone_shards += tuple(
        shard
        for transition in command.helper_transitions
        for shard in transition.transition.phone_shards
    )
    return phone_shards


def _validate_remote_resident_execution_command(
    command: PhysicalExecutionCommand,
) -> None:
    """A reduced desktop parent executes only with bound, verified phone owners."""
    remote = command.execution_contract.remote_resident_ffn
    if remote.parent_artifact_sha256 != command.artifact_sha256:
        raise PhysicalAdapterError(
            "remote-resident FFN group belongs to another artifact"
        )
    if not remote.bound:
        raise PhysicalAdapterError(
            "remote-resident FFN sessions lack physical generations"
        )
    if any(row.operator_plan_sha256 is None for row in remote.sessions):
        raise PhysicalAdapterError(
            "remote-resident FFN sessions lack physical operator plans"
        )
    parameters = command.adapter_parameters
    for name in ("phone_device_id", "ffn_resident_layer_mask", "ffn_resident_columns"):
        if name not in parameters:
            raise PhysicalAdapterError(
                "remote-resident FFN parent lacks the phone runtime parameter " + name
            )
    resident_mask = parameters.get("ffn_resident_layer_mask")
    if type(resident_mask) is not int or remote.layer_mask & ~resident_mask:
        raise PhysicalAdapterError(
            "remote-resident FFN layers exceed the phone resident layer mask"
        )
    helper = command.helper_envelope
    if helper is not None and helper.resident_layer_mask & remote.layer_mask:
        raise PhysicalAdapterError(
            "assisted-copy helper overlaps remote-resident FFN layers"
        )
    session_ids = set(remote.session_ids)
    for transition_command in (*command.transitions, *command.helper_transitions):
        evicted = {
            row.session_id for row in transition_command.transition.evictions
            if getattr(row, "session_id", None) is not None
        }
        if evicted & session_ids:
            raise PhysicalAdapterError(
                "transition evicts a remote-resident FFN owner"
            )


def _validate_desktop_execution_command(
    command: PhysicalExecutionCommand,
) -> None:
    helper = command.helper_envelope
    if (
        type(command.adapter_parameters.get("phone_device_id")) is str
        and command.execution_contract.remote_resident_ffn is None
    ):
        raise PhysicalAdapterError(
            "desktop execution command carries a phone endpoint"
        )
    if helper is None:
        if command.helper_transitions:
            raise PhysicalAdapterError(
                "desktop execution carries unrelated helper transitions"
            )
        return
    if (
        command.operator_plan.get("helper_envelope")
            != helper.to_json()
        or helper.artifact_sha256 != command.artifact_sha256
        or helper.desktop_parent_route_id != command.route_id
        or helper.desktop_placement_sha256
            != command.operator_plan.get("desktop_placement_sha256")
        or helper.helper_plan.execution_contract.execution_mode
            != "adaptive-split"
        or helper.helper_binding.endpoint is None
        or helper.helper_binding.operator_plan_protocol is None
    ):
        raise PhysicalAdapterError(
            "desktop phone helper differs from the ticket"
        )
    _validate_phone_helpers(
        helper.helper_plan.adapter_parameters,
        helper.helper_plan.execution_contract,
        tuple(row.device_id for row in helper.helper_binding.participants),
    )
    expected_transition_ids = tuple(
        row.transition_id for row in helper.preparation_transitions
    )
    if tuple(
        row.transition.transition_id
        for row in command.helper_transitions
    ) != expected_transition_ids or any(
        not row.helper_only
        or row.phone_layout_generation
            != helper.phone_layout_generation
        or row.route_id != helper.route_id
        or row.operator_plan_sha256
            != helper.operator_plan_sha256
        or row.execution_contract
            != helper.helper_plan.execution_contract
        or row.operator_plan != helper.helper_plan.to_json()
        or row.replacement_authorization
            != helper.replacement_authorization
        for row in command.helper_transitions
    ):
        raise PhysicalAdapterError(
            "desktop phone helper transition differs from the ticket"
        )


def _validate_phone_endpoint_binding(
    command: PhysicalExecutionCommand,
) -> None:
    contract = command.execution_contract
    if command.helper_envelope is not None or command.helper_transitions:
        raise PhysicalAdapterError(
            "phone execution command carries a nested helper"
        )
    phone_device_id = contract.phone_device_id
    participant = tuple(
        row for row in command.participants
        if row.device_id == phone_device_id
    )
    if (
        len(participant) != 1
        or participant[0].endpoint != contract.phone_endpoint
        or command.adapter_parameters.get("phone_device_id")
            != phone_device_id
    ):
        raise PhysicalAdapterError(
            "phone endpoint differs from the ticket contract"
        )
    if (
        contract.operator_kind != "whole_model"
        and command.operator_plan.get("assisted_operator_kind")
            != contract.operator_kind
    ):
        raise PhysicalAdapterError(
            "phone operator plan differs from the ticket contract"
        )


def _phone_shard_memory_differs(
    command: PhysicalExecutionCommand,
    resident_shards: Sequence[object],
) -> bool:
    reservation_by_demand = {
        str(row.get("demand_id")): row
        for row in command.memory_reservations
    }
    plan_memory_by_demand = {
        str(row.get("demand_id")): row
        for row in command.operator_plan.get("memory_demands", [])
    }
    if plan_memory_by_demand:
        return any(
            (
                demand := plan_memory_by_demand.get(
                    "phone-session:" + shard.session_id + ":weights"
                )
            ) is None
            or demand.get("required_bytes") != shard.resident_bytes
            or (
                int(demand.get("resident_bytes", 0))
                + int((reservation_by_demand.get(
                    "phone-session:" + shard.session_id + ":weights",
                    {},
                )).get("reserved_bytes", 0))
                + int((reservation_by_demand.get(
                    "phone-session:" + shard.session_id + ":weights",
                    {},
                )).get("replaced_bytes", 0))
            ) != shard.resident_bytes
            for shard in resident_shards
        )
    return any(
        int((reservation_by_demand.get(
            "phone-session:" + shard.session_id + ":weights",
            {},
        )).get("reserved_bytes", 0)) != shard.resident_bytes
        for shard in resident_shards
    )


def _validate_phone_helpers(
    parameters: Mapping[str, int | str],
    contract: RuntimeExecutionContract,
    participant_device_ids: Sequence[str],
) -> int:
    """Check a several-phone binding; returns the layers owned by co-helper phones."""
    if PHONE_HELPERS_PARAMETER not in parameters:
        return 0
    try:
        masks = phone_helper_layer_masks(parameters[PHONE_HELPERS_PARAMETER])
    except RuntimePlanError as error:
        raise PhysicalAdapterError(
            "phone helper binding is invalid"
        ) from error
    devices = tuple(masks)
    shard_mask = 0
    for shard in contract.phone_shards:
        shard_mask |= shard.layer_mask
    if (
        contract.operator_kind != "ffn"
        or devices[0] != contract.phone_device_id
        or devices[0] != parameters.get("phone_device_id")
        or (contract.phone_shards and masks[devices[0]] != shard_mask)
        or sum(masks.values()) != parameters.get("ffn_resident_layer_mask")
        or any(
            participant_device_ids.count(device_id) != 1
            for device_id in devices
        )
    ):
        raise PhysicalAdapterError(
            "phone helper binding differs from the ticket plan"
        )
    return sum(masks[device_id] for device_id in devices[1:])


def _validate_phone_shard_geometry(
    command: PhysicalExecutionCommand,
) -> None:
    contract = command.execution_contract
    phone_device_id = contract.phone_device_id
    shard_mask = 0
    for shard in contract.phone_shards:
        shard_mask |= shard.layer_mask
    # a static co-helper phone owns layers outside the ticket phone's shards
    shard_mask |= _validate_phone_helpers(
        command.adapter_parameters,
        contract,
        tuple(row.device_id for row in command.participants),
    )
    resident_mask = command.adapter_parameters.get(
        "ffn_resident_layer_mask"
    )
    resident_columns = command.adapter_parameters.get(
        "ffn_resident_columns"
    )
    session_count = command.adapter_parameters.get(
        "phone_session_count"
    )
    if (
        contract.operator_kind != "ffn"
        or shard_mask != resident_mask
        or session_count != len(contract.phone_shards)
        or any(
            row.maximum_columns != resident_columns
            for row in contract.phone_shards
        )
    ):
        raise PhysicalAdapterError(
            "phone shard geometry differs from the ticket plan"
        )
    phone_transitions = tuple(
        row for row in command.transitions
        if phone_device_id in row.transition.prepares_device_ids
    )
    transition_shards = tuple(
        shard
        for row in phone_transitions
        for shard in row.transition.phone_shards
    )
    transition_by_session = {
        shard.session_id: shard for shard in transition_shards
    }
    contract_by_session = {
        shard.session_id: shard for shard in contract.phone_shards
    }
    if (
        len(transition_by_session) != len(transition_shards)
        or any(
            transition_by_session.get(session_id) != shard
            for session_id, shard in contract_by_session.items()
        )
    ):
        raise PhysicalAdapterError(
            "phone shard transition differs from the ticket plan"
        )
    resident_shards = (
        transition_shards if transition_shards else contract.phone_shards
    )
    if _phone_shard_memory_differs(command, resident_shards):
        raise PhysicalAdapterError(
            "phone shard memory differs from the ticket plan"
        )
    if any(
        row.phone_layout_generation
            != command.phone_layout_generation
        for row in phone_transitions
    ):
        raise PhysicalAdapterError(
            "phone shard transition differs from the ticket plan"
        )


def _validate_phone_batch_contract(
    command: PhysicalExecutionCommand,
) -> None:
    contract = command.execution_contract
    batch_plan = command.adapter_parameters.get(
        "usb_batch_plan",
        "split-row" if contract.operator_kind == "ffn" else "single",
    )
    queue_depth = command.adapter_parameters.get("usb_queue_depth", 1)
    maximum_batch = command.adapter_parameters.get(
        "ffn_max_tokens",
        command.adapter_parameters.get("parallel", 1),
    )
    if batch_plan == "single":
        maximum_batch = 1
    if (
        batch_plan != contract.batch_plan
        or queue_depth != contract.queue_depth
        or maximum_batch != contract.maximum_batch_size
    ):
        raise PhysicalAdapterError(
            "physical batch plan differs from the ticket contract"
        )
    active_batch = (
        command.adapter_parameters.get("active_request_batch_size", 1)
        if command.decode_cohort is None
        else command.decode_cohort.get("active_batch")
    )
    if (
        type(active_batch) is not int
        or active_batch <= 0
        or active_batch > contract.maximum_batch_size
        or (contract.batch_plan == "single" and active_batch != 1)
    ):
        raise PhysicalAdapterError(
            "physical batch size cannot execute the ticket contract"
        )
    if contract.execution_mode == "adaptive-split" and (
        contract.operator_kind != "ffn"
        or command.operator_plan.get("assisted_operator_kind") != "ffn"
        or command.adapter_parameters.get("ffn_assistance_phase")
            != "decode"
        or command.adapter_parameters.get("ffn_runtime_control_protocol")
            != "decode-boundary-v1"
    ):
            raise PhysicalAdapterError(
                "adaptive split data-plane contract is not executable"
            )


def validate_physical_execution_command(
    command: PhysicalExecutionCommand,
) -> None:
    """Fail closed when a physical command differs from its ticket plan."""
    if not isinstance(command, PhysicalExecutionCommand):
        raise PhysicalAdapterError("physical execution command is invalid")
    contract = command.execution_contract
    if not isinstance(contract, RuntimeExecutionContract):
        raise PhysicalAdapterError(
            "physical execution contract is invalid"
        )
    phone_shards = _command_phone_shards(command)
    if any(row.session_generation == 0 for row in phone_shards):
        raise PhysicalAdapterError(
            "physical phone command contains session generation zero"
        )
    for transition_command in (
        *command.transitions,
        *command.helper_transitions,
    ):
        validate_phone_session_replacement_command(transition_command)
    if command.operator_plan.get("execution_contract") != contract.to_json():
        raise PhysicalAdapterError(
            "physical execution contract differs from the ticket"
        )
    if command.operator_plan.get("plan_sha256") != (
        command.operator_plan_sha256
    ):
        raise PhysicalAdapterError(
            "physical operator plan hash differs from the ticket"
        )
    if contract.phone_shards or contract.remote_resident_ffn is not None:
        if (
            type(command.phone_layout_generation) is not int
            or command.phone_layout_generation < 1
        ):
            raise PhysicalAdapterError(
                "physical command lacks its phone layout generation"
            )
    elif command.phone_layout_generation is not None:
        raise PhysicalAdapterError(
            "physical command carries an unrelated phone layout generation"
        )
    if contract.remote_resident_ffn is not None:
        _validate_remote_resident_execution_command(command)
    if contract.execution_mode == "desktop":
        _validate_desktop_execution_command(command)
        return
    _validate_phone_endpoint_binding(command)
    if contract.phone_shards:
        _validate_phone_shard_geometry(command)
    else:
        _validate_phone_helpers(
            command.adapter_parameters,
            contract,
            tuple(row.device_id for row in command.participants),
        )
    _validate_phone_batch_contract(command)


def bind_ready_helper_to_physical_command(
    command: PhysicalExecutionCommand,
    helper: RuntimeHelperExecutionEnvelope,
) -> PhysicalExecutionCommand:
    """Bind an exact late helper without changing the base execution."""

    validate_physical_execution_command(command)
    if (
        not isinstance(helper, RuntimeHelperExecutionEnvelope)
        or command.execution_contract.execution_mode != "desktop"
        or command.helper_envelope is not None
        or command.helper_transitions
        or helper.artifact_sha256 != command.artifact_sha256
        or helper.desktop_parent_route_id != command.route_id
        or helper.desktop_placement_sha256
            != command.operator_plan.get("desktop_placement_sha256")
    ):
        raise PhysicalAdapterError(
            "late phone helper differs from the base ticket"
        )
    participants = tuple(
        PhysicalParticipantCommand(
            executor_id=row.executor_id,
            device_id=row.device_id,
            endpoint=row.endpoint,
            backend=row.backend,
            resource_ids=row.resource_ids,
        )
        for row in helper.helper_binding.participants
    )
    by_device = {row.device_id: row for row in participants}
    if (
        not participants
        or len(by_device) != len(participants)
        or set(helper.helper_plan.device_ids) != set(by_device)
    ):
        raise PhysicalAdapterError(
            "late phone helper participants differ from the plan"
        )

    def transition_participant(
        transition: RuntimeTransitionPlan,
    ) -> PhysicalParticipantCommand:
        if transition.executor_id is None:
            return by_device[transition.device_id]
        if transition.executor_id == helper.helper_binding.executor_id:
            return PhysicalParticipantCommand(
                executor_id=helper.helper_binding.executor_id,
                device_id=transition.device_id,
                endpoint=helper.helper_binding.endpoint,
                backend=helper.helper_binding.backend,
                resource_ids=transition.resource_ids,
            )
        matches = tuple(
            row for row in participants
            if row.executor_id == transition.executor_id
        )
        if len(matches) != 1:
            raise PhysicalAdapterError(
                "late phone helper transition executor differs"
            )
        return matches[0]

    transitions = tuple(
        PhysicalTransitionCommand(
            ticket_id=helper.preparation_ticket_id(command.ticket_id),
            request_id=command.request_id,
            artifact_sha256=command.artifact_sha256,
            route_id=helper.route_id,
            operator_plan_sha256=helper.operator_plan_sha256,
            participant=transition_participant(transition),
            transition=transition,
            execution_contract=helper.helper_plan.execution_contract,
            adapter_parameters=MappingProxyType(
                dict(helper.helper_plan.adapter_parameters)
            ),
            phone_layout_generation=helper.phone_layout_generation,
            operator_plan_protocol=(
                helper.helper_binding.operator_plan_protocol
            ),
            operator_plan=MappingProxyType(helper.helper_plan.to_json()),
            decode_cohort=command.decode_cohort,
            selection_mode=command.selection_mode,
            helper_only=True,
            replacement_authorization=(
                helper.replacement_authorization
            ),
        )
        for transition in helper.preparation_transitions
    )
    operator_plan = MappingProxyType({
        **dict(command.operator_plan),
        "helper_envelope": helper.to_json(),
    })
    result = replace(
        command,
        operator_plan=operator_plan,
        helper_envelope=helper,
        helper_transitions=transitions,
    )
    validate_physical_execution_command(result)
    return result


def _participant_commands(
    rows: Sequence[object],
) -> tuple[PhysicalParticipantCommand, ...]:
    return tuple(
        PhysicalParticipantCommand(
            executor_id=row.executor_id,
            device_id=row.device_id,
            endpoint=row.endpoint,
            backend=row.backend,
            resource_ids=row.resource_ids,
        )
        for row in rows
    )


def _transition_participant_resolver(
    binding: object,
    participants: tuple[PhysicalParticipantCommand, ...],
    by_device: Mapping[str, PhysicalParticipantCommand],
    mismatch_message: str,
) -> Callable[[RuntimeTransitionPlan], PhysicalParticipantCommand]:
    def transition_participant(
        transition: RuntimeTransitionPlan,
    ) -> PhysicalParticipantCommand:
        if transition.executor_id is None:
            return by_device[transition.device_id]
        if transition.executor_id == binding.executor_id:
            return PhysicalParticipantCommand(
                executor_id=binding.executor_id,
                device_id=transition.device_id,
                endpoint=binding.endpoint,
                backend=binding.backend,
                resource_ids=transition.resource_ids,
            )
        matches = tuple(
            row for row in participants
            if row.executor_id == transition.executor_id
        )
        if len(matches) != 1:
            raise PhysicalAdapterError(mismatch_message)
        return matches[0]

    return transition_participant


def _ticket_decode_cohort(
    ticket: RuntimeRequestTicket,
) -> Mapping[str, object] | None:
    return (
        None
        if ticket.decode_cohort is None
        else MappingProxyType(ticket.decode_cohort.to_json())
    )


def _ticket_transition_commands(
    ticket: RuntimeRequestTicket,
    participants: tuple[PhysicalParticipantCommand, ...],
    by_device: Mapping[str, PhysicalParticipantCommand],
) -> tuple[PhysicalTransitionCommand, ...]:
    plan = ticket.execution_plan
    binding = ticket.binding
    transition_participant = _transition_participant_resolver(
        binding,
        participants,
        by_device,
        "physical transition executor differs from the ticket",
    )
    return tuple(
        PhysicalTransitionCommand(
            ticket_id=ticket.ticket_id,
            request_id=ticket.request.request_id,
            artifact_sha256=ticket.model.artifact_sha256,
            route_id=binding.route_id,
            operator_plan_sha256=plan.plan_sha256,
            participant=transition_participant(transition),
            transition=transition,
            execution_contract=plan.execution_contract,
            adapter_parameters=MappingProxyType(
                dict(plan.adapter_parameters)
            ),
            phone_layout_generation=ticket.phone_layout_generation,
            operator_plan_protocol=binding.operator_plan_protocol,
            operator_plan=MappingProxyType(plan.to_json()),
            decode_cohort=_ticket_decode_cohort(ticket),
            selection_mode=ticket.selection_mode,
            helper_envelope=plan.helper_envelope,
        )
        for transition in plan.transitions
    )


def _ticket_helper_transition_commands(
    ticket: RuntimeRequestTicket,
    helper: RuntimeHelperExecutionEnvelope,
) -> tuple[PhysicalTransitionCommand, ...]:
    helper_participants = _participant_commands(
        helper.helper_binding.participants
    )
    helper_by_device = {
        row.device_id: row for row in helper_participants
    }
    if (
        not helper_participants
        or len(helper_by_device) != len(helper_participants)
        or set(helper.helper_plan.device_ids)
            != set(helper_by_device)
    ):
        raise PhysicalAdapterError(
            "physical helper participants differ from the plan"
        )
    helper_transition_participant = _transition_participant_resolver(
        helper.helper_binding,
        helper_participants,
        helper_by_device,
        "physical helper transition executor differs",
    )
    return tuple(
        PhysicalTransitionCommand(
            ticket_id=helper.preparation_ticket_id(ticket.ticket_id),
            request_id=ticket.request.request_id,
            artifact_sha256=ticket.model.artifact_sha256,
            route_id=helper.route_id,
            operator_plan_sha256=helper.operator_plan_sha256,
            participant=helper_transition_participant(transition),
            transition=transition,
            execution_contract=(
                helper.helper_plan.execution_contract
            ),
            adapter_parameters=MappingProxyType(
                dict(helper.helper_plan.adapter_parameters)
            ),
            phone_layout_generation=(
                helper.phone_layout_generation
            ),
            operator_plan_protocol=(
                helper.helper_binding.operator_plan_protocol
            ),
            operator_plan=MappingProxyType(
                helper.helper_plan.to_json()
            ),
            decode_cohort=_ticket_decode_cohort(ticket),
            selection_mode=ticket.selection_mode,
            helper_only=True,
            replacement_authorization=(
                helper.replacement_authorization
            ),
        )
        for transition in helper.preparation_transitions
    )


def interpret_offline_phone_residency_stage(
    stage: OfflinePhoneResidencyStage,
) -> tuple[PhysicalTransitionCommand, ...]:
    """Translate one authorized offline load without a request ticket."""

    if (
        not isinstance(stage, OfflinePhoneResidencyStage)
        or stage.state != "LOADING"
    ):
        raise PhysicalAdapterError(
            "offline phone residency stage is not physically executable"
        )
    helper = stage.helper_envelope
    participants = _participant_commands(helper.helper_binding.participants)
    by_device = {row.device_id: row for row in participants}
    if (
        not participants
        or len(by_device) != len(participants)
        or set(helper.helper_plan.device_ids) != set(by_device)
    ):
        raise PhysicalAdapterError(
            "offline phone participants differ from the helper plan"
        )
    participant_for = _transition_participant_resolver(
        helper.helper_binding,
        participants,
        by_device,
        "offline phone transition executor differs",
    )
    commands = tuple(
        PhysicalTransitionCommand(
            ticket_id=stage.preparation_ticket_id,
            request_id=stage.request_id,
            artifact_sha256=helper.artifact_sha256,
            route_id=helper.route_id,
            operator_plan_sha256=helper.operator_plan_sha256,
            participant=participant_for(transition),
            transition=transition,
            execution_contract=helper.helper_plan.execution_contract,
            adapter_parameters=MappingProxyType(
                dict(helper.helper_plan.adapter_parameters)
            ),
            phone_layout_generation=helper.phone_layout_generation,
            operator_plan_protocol=(
                helper.helper_binding.operator_plan_protocol
            ),
            operator_plan=MappingProxyType(helper.helper_plan.to_json()),
            selection_mode="offline-residency",
            helper_only=True,
            helper_envelope=helper,
            replacement_authorization=(
                helper.replacement_authorization
            ),
        )
        for transition in helper.preparation_transitions
    )
    if tuple(row.transition.transition_id for row in commands) != (
        stage.transition_ids
    ):
        raise PhysicalAdapterError(
            "offline phone transition identity differs from the stage"
        )
    for command in commands:
        validate_phone_session_replacement_command(command)
    return commands


def interpret_runtime_ticket(
    ticket: RuntimeRequestTicket,
) -> PhysicalExecutionCommand:
    """Validate and expose only the placement selected by the scheduler."""
    if not isinstance(ticket, RuntimeRequestTicket):
        raise PhysicalAdapterError("physical execution ticket is invalid")
    plan = ticket.execution_plan
    binding = ticket.binding
    if plan is None:
        raise PhysicalAdapterError(
            "physical execution requires an automated operator plan"
        )
    if binding.endpoint is None or binding.operator_plan_protocol is None:
        raise PhysicalAdapterError(
            "physical execution binding is not executable"
        )
    if (
        ticket.decision.route_id != binding.route_id
        or binding.route_id != plan.route_id
        or binding.operator_plan_sha256 != plan.plan_sha256
    ):
        raise PhysicalAdapterError(
            "physical execution ticket placement identity differs"
        )
    participants = _participant_commands(binding.participants)
    by_device = {row.device_id: row for row in participants}
    if (
        not participants
        or len(by_device) != len(participants)
        or set(plan.device_ids) != set(by_device)
    ):
        raise PhysicalAdapterError(
            "physical execution participants differ from the plan"
        )
    lease_resources = {row.resource_id for row in ticket.decision.leases}
    if not set(plan.resource_ids).issubset(lease_resources):
        raise PhysicalAdapterError(
            "physical execution plan resources are not leased"
        )
    transition_commands = _ticket_transition_commands(
        ticket, participants, by_device
    )
    helper = plan.helper_envelope
    helper_transition_commands: tuple[PhysicalTransitionCommand, ...] = ()
    if helper is not None:
        helper_transition_commands = _ticket_helper_transition_commands(
            ticket, helper
        )
    operator_plan = MappingProxyType(plan.to_json())
    leases = tuple(MappingProxyType(dict(row)) for row in (
        ticket.activation_receipts()
    ))
    memory = tuple(
        MappingProxyType(row.to_json())
        for row in ticket.memory_reservations
    )
    command = PhysicalExecutionCommand(
        ticket_id=ticket.ticket_id,
        request_id=ticket.request.request_id,
        model_id=ticket.model.model_id,
        artifact_sha256=ticket.model.artifact_sha256,
        route_id=binding.route_id,
        executor_id=binding.executor_id,
        endpoint=binding.endpoint,
        operator_plan_protocol=binding.operator_plan_protocol,
        operator_plan_sha256=plan.plan_sha256,
        planned_start_us=ticket.decision.start_us,
        planned_finish_us=ticket.decision.finish_us,
        planned_finish_upper_us=ticket.decision.finish_upper_us,
        operator_plan=operator_plan,
        participants=participants,
        leases=leases,
        memory_reservations=memory,
        transitions=transition_commands,
        execution_contract=plan.execution_contract,
        adapter_parameters=MappingProxyType(dict(plan.adapter_parameters)),
        phone_layout_generation=ticket.phone_layout_generation,
        selection_mode=ticket.selection_mode,
        decode_cohort=_ticket_decode_cohort(ticket),
        helper_envelope=helper,
        helper_transitions=helper_transition_commands,
    )
    validate_physical_execution_command(command)
    return command
