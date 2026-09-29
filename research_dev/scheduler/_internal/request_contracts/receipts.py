"""Runtime request tickets, validation and terminal receipts: receipts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from ..policy import Decision
from ..runtime_plan import RuntimeExecutionReceipt
from .common import RuntimeControllerError, _integer, _text, _text_tuple
from .ticket import RuntimeRequestTicket


@dataclass(frozen=True)
class RuntimeLatencyUpperBound:
    actual_end_us: int
    prediction_finish_upper_us: int
    met: bool
    overrun_us: int

    def __post_init__(self) -> None:
        _integer("runtime actual_end_us", self.actual_end_us)
        _integer(
            "runtime prediction_finish_upper_us",
            self.prediction_finish_upper_us,
        )
        if type(self.met) is not bool:
            raise RuntimeControllerError("runtime prediction met must be bool")
        _integer("runtime prediction overrun_us", self.overrun_us)
        expected = max(
            0, self.actual_end_us - self.prediction_finish_upper_us
        )
        if self.overrun_us != expected or self.met != (expected == 0):
            raise RuntimeControllerError(
                "runtime prediction assessment is inconsistent"
            )

    @property
    def status(self) -> str:
        return "MET" if self.met else "VIOLATED"

    def to_json(self) -> dict[str, int | bool | str]:
        return {
            "actual_end_us": self.actual_end_us,
            "met": self.met,
            "overrun_us": self.overrun_us,
            "prediction_finish_upper_us": self.prediction_finish_upper_us,
            "status": self.status,
        }


@dataclass(frozen=True)
class RuntimeLeaseCoverage:
    actual_end_us: int
    initial_reserved_until_us: Mapping[str, int]
    final_reserved_until_us: Mapping[str, int]
    covered: bool
    uncovered_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer("runtime lease actual_end_us", self.actual_end_us)
        initial = {
            _text("runtime lease token", token): _integer(
                "runtime initial lease end", value
            )
            for token, value in self.initial_reserved_until_us.items()
        }
        final = {
            _text("runtime lease token", token): _integer(
                "runtime final lease end", value
            )
            for token, value in self.final_reserved_until_us.items()
        }
        if not initial or set(initial) != set(final):
            raise RuntimeControllerError("runtime lease coverage tokens differ")
        if any(final[token] < value for token, value in initial.items()):
            raise RuntimeControllerError("runtime final lease end moved backward")
        uncovered = tuple(sorted(
            _text("runtime uncovered lease token", token)
            for token in self.uncovered_tokens
        ))
        expected = tuple(sorted(
            token for token, value in final.items()
            if self.actual_end_us > value
        ))
        if uncovered != expected or self.covered != (not expected):
            raise RuntimeControllerError(
                "runtime lease coverage assessment is inconsistent"
            )
        object.__setattr__(
            self,
            "initial_reserved_until_us",
            MappingProxyType(dict(sorted(initial.items()))),
        )
        object.__setattr__(
            self,
            "final_reserved_until_us",
            MappingProxyType(dict(sorted(final.items()))),
        )
        object.__setattr__(self, "uncovered_tokens", uncovered)

    @property
    def status(self) -> str:
        return "COVERED" if self.covered else "UNCOVERED"

    def to_json(self) -> dict[str, object]:
        return {
            "actual_end_us": self.actual_end_us,
            "covered": self.covered,
            "final_reserved_until_us": dict(self.final_reserved_until_us),
            "initial_reserved_until_us": dict(
                self.initial_reserved_until_us
            ),
            "status": self.status,
            "uncovered_tokens": list(self.uncovered_tokens),
        }


def assess_runtime_completion(
    decision: Decision,
    final_reserved_until_us: Mapping[str, int],
    actual_end_us: int,
) -> tuple[RuntimeLatencyUpperBound, RuntimeLeaseCoverage]:
    if not isinstance(decision, Decision):
        raise RuntimeControllerError(
            "runtime completion requires a Decision"
        )
    actual_end_us = _integer("runtime completion actual_end_us", actual_end_us)
    initial = {
        lease.token: lease.reserved_until_us for lease in decision.leases
    }
    final = dict(final_reserved_until_us)
    uncovered = tuple(sorted(
        token for token, value in final.items()
        if actual_end_us > value
    ))
    overrun_us = max(0, actual_end_us - decision.finish_upper_us)
    return (
        RuntimeLatencyUpperBound(
            actual_end_us=actual_end_us,
            prediction_finish_upper_us=decision.finish_upper_us,
            met=overrun_us == 0,
            overrun_us=overrun_us,
        ),
        RuntimeLeaseCoverage(
            actual_end_us=actual_end_us,
            initial_reserved_until_us=initial,
            final_reserved_until_us=final,
            covered=not uncovered,
            uncovered_tokens=uncovered,
        ),
    )


@dataclass(frozen=True)
class RuntimeLeaseExtensionReceipt:
    request_id: str
    at_us: int
    cancelled_queued_tokens: Mapping[str, tuple[str, ...]]
    extended_leases: tuple[Mapping[str, int | str], ...]

    def __post_init__(self) -> None:
        _text("runtime extension request_id", self.request_id)
        _integer("runtime extension at_us", self.at_us)
        cancelled = {
            _text("runtime extension queued request_id", request_id): (
                _text_tuple("runtime extension cancelled token", tokens)
            )
            for request_id, tokens in self.cancelled_queued_tokens.items()
        }
        extended = []
        for row in self.extended_leases:
            if set(row) != {
                "previous_reserved_until_us",
                "reserved_until_us",
                "resource_id",
                "token",
            }:
                raise RuntimeControllerError(
                    "runtime extension lease fields are invalid"
                )
            previous = _integer(
                "runtime extension previous end",
                row["previous_reserved_until_us"],
            )
            reserved = _integer(
                "runtime extension reserved end", row["reserved_until_us"]
            )
            if reserved < previous:
                raise RuntimeControllerError(
                    "runtime extension moved a lease backward"
                )
            extended.append(MappingProxyType({
                "previous_reserved_until_us": previous,
                "reserved_until_us": reserved,
                "resource_id": _text(
                    "runtime extension resource_id", row["resource_id"]
                ),
                "token": _text(
                    "runtime extension token", row["token"]
                ),
            }))
        tokens = [row["token"] for row in extended]
        if len(tokens) != len(set(tokens)):
            raise RuntimeControllerError(
                "runtime extension lease tokens must be unique"
            )
        object.__setattr__(
            self,
            "cancelled_queued_tokens",
            MappingProxyType(dict(sorted(cancelled.items()))),
        )
        object.__setattr__(self, "extended_leases", tuple(extended))

    def to_json(self) -> dict[str, object]:
        return {
            "at_us": self.at_us,
            "cancelled_queued_tokens": {
                request_id: list(tokens)
                for request_id, tokens in sorted(
                    self.cancelled_queued_tokens.items()
                )
            },
            "extended_leases": [dict(row) for row in self.extended_leases],
            "request_id": self.request_id,
            "status": "RENEWED",
        }


@dataclass(frozen=True)
class RuntimeCompletionReceipt:
    request_id: str
    route_id: str
    actual_end_us: int
    released_tokens: tuple[str, ...]
    completion_event_replans: tuple[str, ...]
    expired_phase_tokens: tuple[str, ...]
    late_tokens: tuple[str, ...]
    latency_upper_bound: RuntimeLatencyUpperBound
    lease_coverage: RuntimeLeaseCoverage
    quarantine_action: str | None
    execution_receipt: RuntimeExecutionReceipt | None = None

    def __post_init__(self) -> None:
        _text("runtime completion request_id", self.request_id)
        _text("runtime completion route_id", self.route_id)
        _integer("runtime completion actual_end_us", self.actual_end_us)
        released = _text_tuple(
            "runtime completion released token", self.released_tokens
        )
        replans = _text_tuple(
            "runtime completion replan request_id",
            self.completion_event_replans,
        )
        expired = _text_tuple(
            "runtime completion expired token", self.expired_phase_tokens
        )
        late = _text_tuple(
            "runtime completion late token", self.late_tokens
        )
        if (
            not isinstance(self.latency_upper_bound, RuntimeLatencyUpperBound)
            or not isinstance(self.lease_coverage, RuntimeLeaseCoverage)
            or self.latency_upper_bound.actual_end_us != self.actual_end_us
            or self.lease_coverage.actual_end_us != self.actual_end_us
            or set(released)
                != set(self.lease_coverage.final_reserved_until_us)
            or set(expired) & set(late)
            or not set(expired + late).issubset(released)
        ):
            raise RuntimeControllerError(
                "runtime completion receipt is inconsistent"
            )
        if self.quarantine_action is not None:
            _text(
                "runtime completion quarantine_action",
                self.quarantine_action,
            )
        if self.execution_receipt is not None and (
            not isinstance(self.execution_receipt, RuntimeExecutionReceipt)
            or self.execution_receipt.request_id != self.request_id
            or self.execution_receipt.finished_us != self.actual_end_us
        ):
            raise RuntimeControllerError(
                "runtime completion physical receipt differs"
            )
        object.__setattr__(self, "released_tokens", released)
        object.__setattr__(self, "completion_event_replans", replans)
        object.__setattr__(self, "expired_phase_tokens", expired)
        object.__setattr__(self, "late_tokens", late)

    @property
    def upper_bound_violation(self) -> bool:
        return not self.latency_upper_bound.met

    @property
    def status(self) -> str:
        if not self.lease_coverage.covered:
            return "lease_uncovered"
        if self.upper_bound_violation:
            return "prediction_late"
        return "released"

    def to_json(self) -> dict[str, object]:
        result = {
            "actual_end_us": self.actual_end_us,
            "completion_event_replans": list(
                self.completion_event_replans
            ),
            "expired_phase_tokens": list(self.expired_phase_tokens),
            "latency_upper_bound": self.latency_upper_bound.to_json(),
            "late_tokens": list(self.late_tokens),
            "lease_coverage": self.lease_coverage.to_json(),
            "prediction_upper_us": (
                self.latency_upper_bound.prediction_finish_upper_us
            ),
            "quarantine_action": self.quarantine_action,
            "released_tokens": list(self.released_tokens),
            "status": self.status,
            "upper_bound_violation": self.upper_bound_violation,
        }
        if self.execution_receipt is not None:
            result["execution_receipt"] = self.execution_receipt.to_json()
        return result


@dataclass(frozen=True)
class RuntimeFailureRecovery:
    failed_ticket_id: str
    reason: str
    cancelled_tokens: tuple[str, ...]
    quarantine_action: str | None
    fallback: RuntimeRequestTicket | None

    def __post_init__(self) -> None:
        _text("runtime failed_ticket_id", self.failed_ticket_id)
        _text("runtime failure reason", self.reason)
        cancelled = _text_tuple(
            "runtime failure cancelled token", self.cancelled_tokens
        )
        if self.quarantine_action is not None:
            _text(
                "runtime failure quarantine_action", self.quarantine_action
            )
        if self.fallback is not None and (
            not isinstance(self.fallback, RuntimeRequestTicket)
            or self.fallback.previous_ticket_id != self.failed_ticket_id
            or self.fallback.failure_reason != self.reason
        ):
            raise RuntimeControllerError(
                "runtime failure fallback ticket is inconsistent"
            )
        object.__setattr__(self, "cancelled_tokens", cancelled)

    def to_json(self) -> dict[str, object]:
        return {
            "cancelled_tokens": list(self.cancelled_tokens),
            "failed_ticket_id": self.failed_ticket_id,
            "fallback_ticket": (
                None if self.fallback is None else self.fallback.to_json()
            ),
            "quarantine_action": self.quarantine_action,
            "reason": self.reason,
            "status": (
                "FAILED_NO_RECOVERY"
                if self.fallback is None
                else "RECOVERED_WITH_NEW_DECISION"
            ),
        }
