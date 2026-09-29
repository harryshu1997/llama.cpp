"""Exact dispatch of scheduler-selected physical transition commands."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable, Mapping

from .._internal.types import canonical_sha256
from .contracts import PhysicalAdapterError
from .ticket import PhysicalTransitionCommand


PhysicalTransitionHandler = Callable[
    [PhysicalTransitionCommand, object, Callable[[], None]], None
]


@dataclass
class _SharedTransition:
    identity_sha256: str
    member_request_ids: tuple[str, ...]
    callers: set[str]
    returned: set[str]
    complete: bool = False
    error: BaseException | None = None


class CanonicalTransitionRegistry:
    """Bind transition IDs to rig callbacks without selecting transitions."""

    def __init__(
        self, handlers: Mapping[str, PhysicalTransitionHandler]
    ) -> None:
        rows = dict(handlers)
        if not rows or any(
            type(transition_id) is not str
            or not transition_id
            or not transition_id.isascii()
            or not callable(handler)
            for transition_id, handler in rows.items()
        ):
            raise PhysicalAdapterError(
                "physical transition handlers are invalid"
            )
        self._handlers = dict(sorted(rows.items()))
        self._condition = threading.Condition()
        self._active: set[str] = set()
        self._shared: dict[tuple[str, str], _SharedTransition] = {}

    @property
    def transition_ids(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    @staticmethod
    def _cohort_identity(
        command: PhysicalTransitionCommand,
    ) -> tuple[str, tuple[str, ...], str] | None:
        value = command.decode_cohort
        if value is None:
            return None
        try:
            cohort_id = value["cohort_id"]
            members = tuple(value["member_request_ids"])
            sealed = value["sealed"]
        except (KeyError, TypeError) as error:
            raise PhysicalAdapterError(
                "physical shared transition cohort is invalid"
            ) from error
        if (
            type(cohort_id) is not str
            or not cohort_id
            or len(members) < 2
            or len(members) != len(set(members))
            or command.request_id not in members
            or sealed is not True
        ):
            raise PhysicalAdapterError(
                "physical shared transition cohort differs"
            )
        identity = canonical_sha256({
            "adapter_parameters": dict(command.adapter_parameters),
            "artifact_sha256": command.artifact_sha256,
            "cohort_id": cohort_id,
            "participant": command.participant.to_json(),
            "schema": "physical-shared-transition-v1",
            "transition": command.transition.to_json(),
        })
        return cohort_id, members, identity

    def _execute_shared(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
        control_check: Callable[[], None],
        handler: PhysicalTransitionHandler,
        cohort: tuple[str, tuple[str, ...], str],
    ) -> bool:
        cohort_id, members, identity = cohort
        transition_id = command.transition.transition_id
        key = (cohort_id, transition_id)
        execute = False
        with self._condition:
            state = self._shared.get(key)
            if state is None:
                state = _SharedTransition(
                    identity_sha256=identity,
                    member_request_ids=members,
                    callers=set(),
                    returned=set(),
                )
                self._shared[key] = state
                execute = True
            if (
                state.identity_sha256 != identity
                or state.member_request_ids != members
                or command.request_id in state.callers
            ):
                raise PhysicalAdapterError(
                    "physical shared transition identity differs"
                )
            state.callers.add(command.request_id)

        caller_error = None
        try:
            if execute:
                try:
                    handler(command, payload, control_check)
                    control_check()
                except BaseException as error:
                    with self._condition:
                        if state.error is None:
                            state.error = error
                        state.complete = True
                        self._condition.notify_all()
                else:
                    with self._condition:
                        state.complete = True
                        self._condition.notify_all()
            else:
                while True:
                    control_check()
                    with self._condition:
                        if state.complete:
                            break
                        self._condition.wait(0.1)
        except BaseException as error:
            caller_error = error
            with self._condition:
                if state.error is None:
                    state.error = error
                state.complete = True
                self._condition.notify_all()
        finally:
            with self._condition:
                error = state.error
                state.returned.add(command.request_id)
                if len(state.returned) == len(state.member_request_ids):
                    del self._shared[key]
                self._condition.notify_all()
        if caller_error is not None:
            raise PhysicalAdapterError(
                "physical shared transition control failed: "
                + str(caller_error)
            ) from caller_error
        if error is not None:
            raise PhysicalAdapterError(
                "physical shared transition failed: " + str(error)
            ) from error
        return True

    def execute(
        self,
        command: PhysicalTransitionCommand,
        payload: object,
        control_check: Callable[[], None],
    ) -> bool:
        if not isinstance(command, PhysicalTransitionCommand):
            raise PhysicalAdapterError(
                "physical transition command is invalid"
            )
        if not callable(control_check):
            raise PhysicalAdapterError(
                "physical transition control check is invalid"
            )
        transition_id = command.transition.transition_id
        handler = self._handlers.get(transition_id)
        if handler is None:
            raise PhysicalAdapterError(
                "physical transition has no registered executor"
            )
        control_check()
        cohort = self._cohort_identity(command)
        if cohort is not None:
            return self._execute_shared(
                command, payload, control_check, handler, cohort
            )
        with self._condition:
            if transition_id in self._active:
                raise PhysicalAdapterError(
                    "physical transition is already executing"
                )
            self._active.add(transition_id)
        try:
            handler(command, payload, control_check)
            control_check()
        finally:
            with self._condition:
                self._active.remove(transition_id)
        return True
