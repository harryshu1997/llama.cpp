"""Physical barriers and single-accounting for scheduler decode cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable, Mapping

from .contracts import PhysicalAdapterError, RawEnergyMeasurement
from .ticket import PhysicalExecutionCommand


@dataclass
class _ExecutionState:
    cohort_id: str
    members: tuple[str, ...]
    leader_request_id: str
    common_policy_sha256: str
    started_ns_by_request: dict[str, int]
    finished_ns_by_request: dict[str, int]
    work_by_request: dict[str, tuple[int, int]]
    measurement: RawEnergyMeasurement | None = None
    measuring: bool = False
    failure: BaseException | None = None
    returned_request_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.returned_request_ids is None:
            self.returned_request_ids = set()


@dataclass(frozen=True)
class DecodeCohortExecutionResult:
    started_ns: int
    finished_ns: int
    energy: RawEnergyMeasurement | None
    total_input_tokens: int
    total_output_tokens: int


class DecodeCohortExecutionTracker:
    """Start together and attribute one whole-fleet measurement once."""

    def __init__(self, energy_meter: object) -> None:
        if not callable(getattr(energy_meter, "measure", None)):
            raise PhysicalAdapterError(
                "decode cohort energy meter is invalid"
            )
        self._energy_meter = energy_meter
        self._condition = threading.Condition()
        self._states: dict[str, _ExecutionState] = {}

    @staticmethod
    def _identity(
        command: PhysicalExecutionCommand,
    ) -> tuple[str, tuple[str, ...], str, str] | None:
        value = command.decode_cohort
        if value is None:
            return None
        try:
            cohort_id = value["cohort_id"]
            members = tuple(value["member_request_ids"])
            leader = value["leader_request_id"]
            policy = value["common_policy_sha256"]
            sealed = value["sealed"]
        except (KeyError, TypeError) as error:
            raise PhysicalAdapterError(
                "physical decode cohort contract is invalid"
            ) from error
        if (
            type(cohort_id) is not str
            or not cohort_id
            or len(members) < 2
            or len(members) != len(set(members))
            or command.request_id not in members
            or leader != members[0]
            or type(policy) is not str
            or not policy
            or sealed is not True
        ):
            raise PhysicalAdapterError(
                "physical decode cohort identity differs"
            )
        return cohort_id, members, leader, policy

    def begin(
        self,
        command: PhysicalExecutionCommand,
        *,
        input_tokens: int,
        output_tokens: int,
        timeout_s: float = 10.0,
    ) -> int | None:
        identity = self._identity(command)
        if identity is None:
            return None
        if (
            timeout_s <= 0
            or type(input_tokens) is not int
            or input_tokens <= 0
            or type(output_tokens) is not int
            or output_tokens <= 0
        ):
            raise PhysicalAdapterError(
                "decode cohort start timeout is invalid"
            )
        cohort_id, members, leader, policy = identity
        now_ns = time.monotonic_ns()
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                prepare = getattr(self._energy_meter, "prepare", None)
                if callable(prepare):
                    prepare()
                state = _ExecutionState(
                    cohort_id=cohort_id,
                    members=members,
                    leader_request_id=leader,
                    common_policy_sha256=policy,
                    started_ns_by_request={},
                    finished_ns_by_request={},
                    work_by_request={},
                )
                self._states[cohort_id] = state
            if (
                state.members != members
                or state.leader_request_id != leader
                or state.common_policy_sha256 != policy
                or command.request_id in state.started_ns_by_request
            ):
                raise PhysicalAdapterError(
                    "physical decode cohort start differs"
                )
            state.started_ns_by_request[command.request_id] = now_ns
            state.work_by_request[command.request_id] = (
                input_tokens, output_tokens
            )
            self._condition.notify_all()
            deadline = time.monotonic() + timeout_s
            while len(state.started_ns_by_request) < len(state.members):
                if state.failure is not None:
                    raise PhysicalAdapterError(
                        "physical decode cohort start failed"
                    ) from state.failure
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = PhysicalAdapterError(
                        "physical decode cohort did not assemble"
                    )
                    state.failure = error
                    self._condition.notify_all()
                    raise error
                self._condition.wait(remaining)
            return min(state.started_ns_by_request.values())

    def fail(
        self, command: PhysicalExecutionCommand, error: BaseException
    ) -> None:
        identity = self._identity(command)
        if identity is None:
            return
        with self._condition:
            state = self._states.get(identity[0])
            if state is not None and state.failure is None:
                state.failure = error
                self._condition.notify_all()

    def finish(
        self,
        command: PhysicalExecutionCommand,
        finished_ns: int,
    ) -> DecodeCohortExecutionResult | None:
        identity = self._identity(command)
        if identity is None:
            return None
        cohort_id, members, _, _ = identity
        if type(finished_ns) is not int or finished_ns <= 0:
            raise PhysicalAdapterError(
                "physical decode cohort finish is invalid"
            )
        with self._condition:
            state = self._states.get(cohort_id)
            if (
                state is None
                or state.members != members
                or command.request_id not in state.started_ns_by_request
                or command.request_id in state.finished_ns_by_request
            ):
                raise PhysicalAdapterError(
                    "physical decode cohort finish differs"
                )
            state.finished_ns_by_request[command.request_id] = finished_ns
            last = len(state.finished_ns_by_request) == len(state.members)
            if last and state.measurement is None and not state.measuring:
                state.measuring = True
                started_ns = min(state.started_ns_by_request.values())
                cohort_finished_ns = max(
                    state.finished_ns_by_request.values()
                )
                try:
                    state.measurement = self._energy_meter.measure(
                        started_ns, cohort_finished_ns
                    )
                except BaseException as error:
                    state.failure = error
                finally:
                    state.measuring = False
                    self._condition.notify_all()
            else:
                self._condition.notify_all()
            if state.failure is not None:
                raise PhysicalAdapterError(
                    "physical decode cohort measurement failed"
                ) from state.failure
            started_ns = min(state.started_ns_by_request.values())
            cohort_finished_ns = (
                max(state.finished_ns_by_request.values())
                if last else finished_ns
            )
            measurement = state.measurement if last else None
            total_input_tokens = (
                sum(row[0] for row in state.work_by_request.values())
                if last else 0
            )
            total_output_tokens = (
                sum(row[1] for row in state.work_by_request.values())
                if last else 0
            )
            state.returned_request_ids.add(command.request_id)
            if len(state.returned_request_ids) == len(state.members):
                del self._states[cohort_id]
            return DecodeCohortExecutionResult(
                started_ns=started_ns,
                finished_ns=cohort_finished_ns,
                energy=measurement,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
            )

    def abort(self, command: PhysicalExecutionCommand) -> None:
        """Retire one failed member without waiting for cohort completion."""
        identity = self._identity(command)
        if identity is None:
            return
        with self._condition:
            state = self._states.get(identity[0])
            if state is None:
                return
            state.returned_request_ids.add(command.request_id)
            if len(state.returned_request_ids) == len(state.members):
                del self._states[identity[0]]
            self._condition.notify_all()


@dataclass
class _PolicyState:
    cohort_id: str
    members: tuple[str, ...]
    leader_request_id: str
    common_policy_sha256: str
    registrations: dict[str, tuple[PhysicalExecutionCommand, object, int]]
    slot_by_request: dict[str, int]
    token_by_request: dict[str, int]
    at_ns_by_request: dict[str, int]
    fastest_token_ns_by_request: dict[str, int]
    retiring_requests: set[str]
    terminal_requests: set[str]
    http_completed_requests: set[str]
    control_ack_by_request: dict[str, Mapping[str, object]]
    policy_members: tuple[str, ...]
    controller_progress: (
        Callable[[int, int, int, bool], bool | None] | None
    ) = None
    controller_active_batch: Callable[[int], None] | None = None
    processing: bool = False
    forwarded_token: int | None = None
    forwarded_terminal: bool = False
    membership_generation: int = 0
    failure: BaseException | None = None
    returned: int = 0


@dataclass(frozen=True)
class _PendingForward:
    leader_slot: int
    frontier: int
    frontier_ns: int
    all_terminal: bool
    membership_changed: bool


def _validate_progress_arguments(
    slot_id: int, token_index: int, at_ns: int, terminal: bool
) -> None:
    if (
        type(slot_id) is not int
        or slot_id < 0
        or type(token_index) is not int
        or token_index < 0
        or type(at_ns) is not int
        or at_ns <= 0
        or type(terminal) is not bool
    ):
        raise PhysicalAdapterError(
            "decode cohort progress is invalid"
        )


def _pending_forward(state: _PolicyState) -> _PendingForward | None:
    """Return the frontier to forward, or None when nothing is due."""
    members = state.policy_members
    if not members:
        return None
    if any(
        request_id not in state.slot_by_request
        for request_id in members
    ):
        return None
    frontier = min(
        state.token_by_request[request_id]
        for request_id in members
    )
    all_terminal = all(
        request_id in state.terminal_requests
        for request_id in members
    )
    membership_changed = (
        any(
            request_id in state.terminal_requests
            for request_id in members
        )
        and not all_terminal
    )
    should_forward = (
        state.forwarded_token is None
        or frontier > state.forwarded_token
        or (all_terminal and not state.forwarded_terminal)
        or membership_changed
    )
    if not should_forward or state.processing:
        return None
    return _PendingForward(
        leader_slot=state.slot_by_request[state.leader_request_id],
        frontier=frontier,
        frontier_ns=max(
            state.at_ns_by_request[request_id]
            for request_id in members
        ),
        all_terminal=all_terminal,
        membership_changed=membership_changed,
    )


class DecodeCohortPolicyView:
    """Read-only live cohort state used by one physical policy controller."""

    def __init__(
        self,
        coordinator: "DecodeCohortPolicyCoordinator",
        cohort_id: str,
    ) -> None:
        self._coordinator = coordinator
        self.cohort_id = cohort_id

    def member_slots(self) -> tuple[tuple[str, int], ...]:
        return self._coordinator.member_slots(self.cohort_id)

    def surviving_member_slots(self) -> tuple[tuple[str, int], ...]:
        return self._coordinator.surviving_member_slots(self.cohort_id)

    def token_positions(self) -> Mapping[str, int]:
        return self._coordinator.token_positions(self.cohort_id)

    @property
    def active_batch(self) -> int:
        return self._coordinator.active_members(self.cohort_id)

    @property
    def membership_changed(self) -> bool:
        return self._coordinator.membership_changed(self.cohort_id)

    def publish_control_ack(self, value: Mapping[str, object]) -> None:
        self._coordinator.publish_control_ack(self.cohort_id, value)


class DecodeCohortPolicyCoordinator:
    """Serialize one global FFN policy across a continuous decode batch."""

    def __init__(self, recovery_timeout_s: float = 5.0) -> None:
        if recovery_timeout_s <= 0:
            raise PhysicalAdapterError(
                "decode cohort recovery timeout is invalid"
            )
        self._condition = threading.Condition()
        self._states: dict[str, _PolicyState] = {}
        self._recovery_budget_ns = round(
            recovery_timeout_s * 1_000_000_000
        )

    @staticmethod
    def _identity(
        command: PhysicalExecutionCommand,
    ) -> tuple[str, tuple[str, ...], str, str]:
        value = DecodeCohortExecutionTracker._identity(command)
        if value is None:
            raise PhysicalAdapterError(
                "decode cohort policy requires a cohort command"
            )
        return value

    def register(
        self,
        command: PhysicalExecutionCommand,
        payload: object,
        execution_started_ns: int,
        controller_factory: Callable[
            [PhysicalExecutionCommand, object, int, DecodeCohortPolicyView],
            tuple[
                Callable[[int, int, int, bool], bool | None],
                Callable[[int], None],
            ],
        ],
        *,
        timeout_s: float = 10.0,
    ) -> tuple[
        Callable[[int, int, int, bool], None],
        Callable[[int], None],
    ]:
        if (
            type(execution_started_ns) is not int
            or execution_started_ns <= 0
            or not callable(controller_factory)
            or timeout_s <= 0
        ):
            raise PhysicalAdapterError(
                "decode cohort policy registration is invalid"
            )
        cohort_id, members, leader, policy = self._identity(command)
        create = None
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                state = _PolicyState(
                    cohort_id=cohort_id,
                    members=members,
                    leader_request_id=leader,
                    common_policy_sha256=policy,
                    registrations={},
                    slot_by_request={},
                    token_by_request={},
                    at_ns_by_request={},
                    fastest_token_ns_by_request={},
                    retiring_requests=set(),
                    terminal_requests=set(),
                    http_completed_requests=set(),
                    control_ack_by_request={},
                    policy_members=members,
                )
                self._states[cohort_id] = state
            if (
                state.members != members
                or state.leader_request_id != leader
                or state.common_policy_sha256 != policy
                or command.request_id in state.registrations
            ):
                raise PhysicalAdapterError(
                    "decode cohort policy registration differs"
                )
            state.registrations[command.request_id] = (
                command,
                payload,
                execution_started_ns,
            )
            if len(state.registrations) == len(members):
                create = state.registrations[leader]
            self._condition.notify_all()

        if create is not None:
            try:
                progress, active_batch = controller_factory(
                    create[0],
                    create[1],
                    create[2],
                    DecodeCohortPolicyView(self, cohort_id),
                )
                if not callable(progress) or not callable(active_batch):
                    raise PhysicalAdapterError(
                        "decode cohort policy controller is invalid"
                    )
            except BaseException as error:
                self.fail(command, error)
                raise
            with self._condition:
                state = self._states[cohort_id]
                state.controller_progress = progress
                state.controller_active_batch = active_batch
                self._condition.notify_all()

        with self._condition:
            deadline = time.monotonic() + timeout_s
            while state.controller_progress is None and state.failure is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    state.failure = PhysicalAdapterError(
                        "decode cohort policy did not assemble"
                    )
                    self._condition.notify_all()
                    break
                self._condition.wait(remaining)
            if state.failure is not None:
                raise PhysicalAdapterError(
                    "decode cohort policy registration failed"
                ) from state.failure

        def progress_callback(
            slot_id: int,
            token_index: int,
            at_ns: int,
            terminal: bool,
        ) -> None:
            self.progress(
                cohort_id,
                command.request_id,
                slot_id,
                token_index,
                at_ns,
                terminal,
            )

        def active_batch_callback(value: int) -> None:
            self.active_batch(cohort_id, value)

        return progress_callback, active_batch_callback

    def publish_control_ack(
        self,
        cohort_id: str,
        value: Mapping[str, object],
    ) -> None:
        if not isinstance(value, Mapping):
            raise PhysicalAdapterError(
                "decode cohort control acknowledgement is invalid"
            )
        rows = value.get("cohort_members")
        with self._condition:
            state = self._states.get(cohort_id)
            if (
                state is None
                or state.control_ack_by_request
                or type(rows) is not list
                or len(rows) != len(state.policy_members)
                or type(value.get("policy_hash")) is not str
                or type(value.get("plan_generation")) is not int
            ):
                raise PhysicalAdapterError(
                    "decode cohort control acknowledgement differs"
                )
            by_request = {
                row.get("request_id"): row
                for row in rows
                if isinstance(row, Mapping)
            }
            if set(by_request) != set(state.policy_members):
                raise PhysicalAdapterError(
                    "decode cohort control members differ"
                )
            acknowledgements = {}
            for request_id in state.policy_members:
                row = by_request[request_id]
                if (
                    row.get("slot_id")
                        != state.slot_by_request.get(request_id)
                    or type(row.get("applied_token_index")) is not int
                    or row["applied_token_index"] < 0
                    or row.get("plan_generation")
                        != value["plan_generation"]
                ):
                    raise PhysicalAdapterError(
                        "decode cohort control member differs"
                    )
                acknowledgements[request_id] = {
                    "applied_token_index": row["applied_token_index"],
                    "plan_generation": value["plan_generation"],
                    "policy_hash": value["policy_hash"],
                    "request_id": request_id,
                    "slot_id": row["slot_id"],
                }
            state.control_ack_by_request = acknowledgements
            self._condition.notify_all()

    def control_ack(
        self,
        command: PhysicalExecutionCommand,
    ) -> Mapping[str, object]:
        cohort_id, members, _, _ = self._identity(command)
        with self._condition:
            state = self._states.get(cohort_id)
            value = (
                None
                if state is None
                else state.control_ack_by_request.get(command.request_id)
            )
            if state is None or state.members != members or value is None:
                raise PhysicalAdapterError(
                    "decode cohort control acknowledgement is absent"
                )
            return dict(value)

    def active_batch(self, cohort_id: str, observed: int) -> None:
        if type(observed) is not int or observed <= 0:
            raise PhysicalAdapterError(
                "decode cohort active batch is invalid"
            )
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None or state.controller_active_batch is None:
                raise PhysicalAdapterError(
                    "decode cohort policy state is absent"
                )
            callback = state.controller_active_batch
            batch = len(state.policy_members)
        if batch:
            callback(batch)

    def progress(
        self,
        cohort_id: str,
        request_id: str,
        slot_id: int,
        token_index: int,
        at_ns: int,
        terminal: bool,
    ) -> None:
        _validate_progress_arguments(slot_id, token_index, at_ns, terminal)
        first_update = True
        while True:
            with self._condition:
                state = self._states.get(cohort_id)
                if state is None or request_id not in state.members:
                    raise PhysicalAdapterError(
                        "decode cohort progress state is absent"
                    )
                if state.failure is not None:
                    raise PhysicalAdapterError(
                        "decode cohort policy failed"
                    ) from state.failure
                if first_update:
                    self._record_progress_update(
                        state, request_id, slot_id, token_index, at_ns, terminal
                    )
                    first_update = False
                forward = _pending_forward(state)
                if forward is None:
                    return
                state.processing = True
                callback = state.controller_progress
            try:
                if callback is None:
                    raise PhysicalAdapterError(
                        "decode cohort controller is absent"
                    )
                accepted = callback(
                    forward.leader_slot,
                    forward.frontier,
                    forward.frontier_ns,
                    forward.all_terminal or forward.membership_changed,
                )
            except BaseException as error:
                self._record_forward_failure(cohort_id, error)
                raise
            if not self._commit_forward(cohort_id, forward, accepted):
                return

    def _record_progress_update(
        self,
        state: _PolicyState,
        request_id: str,
        slot_id: int,
        token_index: int,
        at_ns: int,
        terminal: bool,
    ) -> None:
        previous_slot = state.slot_by_request.get(request_id)
        previous_token = state.token_by_request.get(request_id)
        previous_at = state.at_ns_by_request.get(request_id)
        if (
            previous_slot not in {None, slot_id}
            or (
                previous_token is not None
                and token_index < previous_token
            )
            or (previous_at is not None and at_ns < previous_at)
            or request_id in state.terminal_requests
        ):
            raise PhysicalAdapterError(
                "decode cohort progress moved backward"
            )
        if (
            previous_token is not None
            and previous_at is not None
            and token_index > previous_token
            and at_ns > previous_at
        ):
            interval_ns = max(
                1,
                (at_ns - previous_at)
                // (token_index - previous_token),
            )
            prior = state.fastest_token_ns_by_request.get(
                request_id
            )
            state.fastest_token_ns_by_request[request_id] = (
                interval_ns
                if prior is None else min(prior, interval_ns)
            )
        state.slot_by_request[request_id] = slot_id
        state.token_by_request[request_id] = token_index
        state.at_ns_by_request[request_id] = at_ns
        output_tokens = getattr(
            state.registrations[request_id][1],
            "output_tokens",
            None,
        )
        if type(output_tokens) is int and output_tokens > 1:
            tail_guard_tokens = self._tail_guard_tokens(
                state, request_id, output_tokens
            )
            if token_index >= output_tokens - tail_guard_tokens:
                state.retiring_requests.add(request_id)
        if terminal:
            state.retiring_requests.add(request_id)
            state.terminal_requests.add(request_id)

    def _tail_guard_tokens(
        self, state: _PolicyState, request_id: str, output_tokens: int
    ) -> int:
        fastest_ns = state.fastest_token_ns_by_request.get(
            request_id
        )
        tail_guard_tokens = 2
        if fastest_ns is not None:
            command = state.registrations[request_id][0]
            predicted_service_us = max(
                1,
                command.planned_finish_us
                - command.planned_start_us,
            )
            predicted_token_ns = max(
                1,
                predicted_service_us
                * 1_000
                // output_tokens,
            )
            token_interval_ns = max(
                fastest_ns, predicted_token_ns
            )
            tail_guard_tokens = max(
                tail_guard_tokens,
                (
                    self._recovery_budget_ns
                    + token_interval_ns - 1
                ) // token_interval_ns,
            )
        return min(tail_guard_tokens, output_tokens - 1)

    def _record_forward_failure(
        self, cohort_id: str, error: BaseException
    ) -> None:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is not None:
                state.failure = error
                state.processing = False
                self._condition.notify_all()

    def _commit_forward(
        self,
        cohort_id: str,
        forward: "_PendingForward",
        accepted: bool | None,
    ) -> bool:
        """Publish one forwarded frontier; return whether to loop again."""
        frontier = forward.frontier
        all_terminal = forward.all_terminal
        membership_changed = forward.membership_changed
        with self._condition:
            state = self._states[cohort_id]
            state.forwarded_token = frontier
            state.forwarded_terminal = (
                all_terminal and accepted is not False
            )
            if membership_changed and accepted is not False:
                state.policy_members = tuple(
                    request_id for request_id in state.policy_members
                    if request_id not in state.terminal_requests
                )
                state.membership_generation += 1
                if state.controller_active_batch is not None:
                    state.controller_active_batch(
                        len(state.policy_members)
                    )
            elif all_terminal and accepted is not False:
                state.policy_members = ()
            state.processing = False
            self._condition.notify_all()
            if not state.policy_members:
                return False
            newer_frontier = min(
                state.token_by_request[request_id]
                for request_id in state.policy_members
            )
            newer_terminal = (
                all(
                    request_id in state.terminal_requests
                    for request_id in state.policy_members
                )
            )
            return (
                newer_frontier > frontier
                or (newer_terminal and not all_terminal)
            )

    def member_slots(
        self, cohort_id: str
    ) -> tuple[tuple[str, int], ...]:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None or any(
                request_id not in state.slot_by_request
                for request_id in state.policy_members
            ):
                raise PhysicalAdapterError(
                    "decode cohort slots are incomplete"
                )
            return tuple(
                (request_id, state.slot_by_request[request_id])
                for request_id in state.policy_members
            )

    def surviving_member_slots(
        self, cohort_id: str
    ) -> tuple[tuple[str, int], ...]:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                raise PhysicalAdapterError(
                    "decode cohort policy state is absent"
                )
            members = tuple(
                request_id for request_id in state.policy_members
                if request_id not in state.terminal_requests
            )
            if not members or any(
                request_id not in state.slot_by_request
                for request_id in members
            ):
                raise PhysicalAdapterError(
                    "decode cohort surviving slots are incomplete"
                )
            return tuple(
                (request_id, state.slot_by_request[request_id])
                for request_id in members
            )

    def token_positions(self, cohort_id: str) -> Mapping[str, int]:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None or any(
                request_id not in state.token_by_request
                for request_id in state.policy_members
            ):
                raise PhysicalAdapterError(
                    "decode cohort token positions are incomplete"
                )
            return {
                request_id: state.token_by_request[request_id]
                for request_id in state.policy_members
            }

    def active_members(self, cohort_id: str) -> int:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                raise PhysicalAdapterError(
                    "decode cohort policy state is absent"
                )
            return len(state.policy_members)

    def membership_changed(self, cohort_id: str) -> bool:
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                raise PhysicalAdapterError(
                    "decode cohort policy state is absent"
                )
            return any(
                request_id in state.terminal_requests
                for request_id in state.policy_members
            ) and not all(
                request_id in state.terminal_requests
                for request_id in state.policy_members
            )

    def complete_http(
        self,
        command: PhysicalExecutionCommand,
    ) -> str | None:
        cohort_id, members, leader, _ = self._identity(command)
        with self._condition:
            state = self._states.get(cohort_id)
            if (
                state is None
                or state.members != members
                or command.request_id in state.http_completed_requests
            ):
                raise PhysicalAdapterError(
                    "decode cohort HTTP completion differs"
                )
            state.http_completed_requests.add(command.request_id)
            self._condition.notify_all()
            if state.failure is not None:
                raise PhysicalAdapterError(
                    "decode cohort HTTP execution failed"
                ) from state.failure
            if len(state.http_completed_requests) != len(state.members):
                return None
            return leader

    def fail(
        self, command: PhysicalExecutionCommand, error: BaseException
    ) -> None:
        cohort_id, _, _, _ = self._identity(command)
        with self._condition:
            state = self._states.get(cohort_id)
            if state is not None and state.failure is None:
                state.failure = error
                self._condition.notify_all()

    def release(self, command: PhysicalExecutionCommand) -> None:
        cohort_id, members, _, _ = self._identity(command)
        with self._condition:
            state = self._states.get(cohort_id)
            if state is None:
                return
            state.returned += 1
            if state.returned == len(members):
                del self._states[cohort_id]
            self._condition.notify_all()
