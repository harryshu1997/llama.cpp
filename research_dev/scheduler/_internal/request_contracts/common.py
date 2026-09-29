"""Runtime request tickets, validation and terminal receipts: common."""

from __future__ import annotations

from typing import Sequence


RUNTIME_REQUEST_TICKET_SCHEMA = "research-scheduler-runtime-ticket-v1"


RUNTIME_TRANSITION_STATES = frozenset({
    "COMPLETED",
    "FAILED",
    "NOT_REQUIRED",
    "PENDING",
})


RUNTIME_SELECTION_MODES = frozenset({
    "adaptive-decode",
    "calibration",
    "deadline-first",
    "desktop-baseline",
    "energy-aware",
    "energy-first",
})


RUNTIME_MEMORY_RESERVED_ATOMIC = "RESERVED_ATOMIC"


RUNTIME_MEMORY_NOT_REQUIRED_RESIDENT = "NOT_REQUIRED_RESIDENT"


RUNTIME_MEMORY_CANCELLED = "CANCELLED"


class RuntimeControllerError(ValueError):
    pass


class RuntimeReplanRetryRequired(RuntimeControllerError):
    """A queued replan needs a newly sampled runtime snapshot."""

    def __init__(
        self,
        request_id: str,
        ticket_id: str,
        reason: str,
    ) -> None:
        self.request_id = _text("runtime retry request_id", request_id)
        self.ticket_id = _text("runtime retry ticket_id", ticket_id)
        self.reason = _text("runtime retry reason", reason)
        super().__init__(self.reason)


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeControllerError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeControllerError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _text_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    result = tuple(_text(name, value) for value in values)
    if len(result) != len(set(result)):
        raise RuntimeControllerError(f"{name} values must be unique")
    return result


RUNTIME_DISPATCH_STATES = frozenset({
    "ACQUIRED",
    "CANCELLED",
    "COMPLETED",
    "FAILED",
    "QUEUED",
    "REPLAN_REQUIRED",
})


RUNTIME_LEASE_STATES = frozenset({
    "CANCELLED",
    "COVERED",
    "RELEASED_PENDING_RECEIPT",
    "RESERVED",
    "UNCOVERED",
})


RUNTIME_PREDICTION_STATES = frozenset({
    "MET",
    "PENDING",
    "VIOLATED",
})


RUNTIME_TERMINAL_STATES = frozenset({"CANCELLED", "COMPLETED", "FAILED"})
