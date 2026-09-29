"""Physical failure facts and scheduler-owned lease renewal."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from .runtime_controller import (
    RuntimeLeaseExtensionReceipt,
    RuntimeRequestTicket,
)


class RuntimeExecutionCoordinatorError(RuntimeError):
    pass


# Elastic-phone failure kinds (research_dev/scheduler/campaigns/burstgpt/reports/
# 20260925-elastic-phones/SPEC.md section 2). Only a rig running with
# ``elastic_phones.drop_recovery`` produces them; every other failure keeps its
# historical phase and recovery semantics.
HELPER_LOST_PHASE = "helper_lost"
SERVER_EXITED_PHASE = "server_exited"
ELASTIC_FAILURE_PHASES = frozenset({HELPER_LOST_PHASE, SERVER_EXITED_PHASE})
# A helper phone died mid-stream: the request restarts from its prompt on a
# route without that phone, so the re-execution is exact although the failed
# attempt had started.
RESTARTABLE_STARTED_PHASES = frozenset({HELPER_LOST_PHASE})


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeExecutionCoordinatorError(
            f"{name} must be non-empty ASCII text"
        )
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeExecutionCoordinatorError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


@dataclass(frozen=True)
class RuntimeExecutionFailure:
    phase: str
    retry_safe: bool
    execution_started: bool
    failed_resource_ids: tuple[str, ...] = ()
    # helper_lost only: the phone device ids the failed attempt lost.
    failed_device_ids: tuple[str, ...] = ()
    # helper_lost / server_exited only: the executor whose server process the failed attempt
    # ran on and which has exited since (reaped or retired by the rig); any later residency of
    # that executor is a new server generation.
    exited_executor_id: str | None = None
    # helper_lost only (elastic phones S2a): the live server that masked the lost helpers out and
    # keeps serving, so the failed attempt's own route is a recovery target on that server.
    masked_executor_id: str | None = None

    def __post_init__(self) -> None:
        _text("runtime execution failure phase", self.phase)
        if type(self.retry_safe) is not bool:
            raise RuntimeExecutionCoordinatorError(
                "runtime execution failure retry_safe must be bool"
            )
        if type(self.execution_started) is not bool:
            raise RuntimeExecutionCoordinatorError(
                "runtime execution failure execution_started must be bool"
            )
        resources = tuple(sorted(
            _text("runtime failed resource", value)
            for value in self.failed_resource_ids
        ))
        if len(resources) != len(set(resources)):
            raise RuntimeExecutionCoordinatorError(
                "runtime failed resources are not unique"
            )
        object.__setattr__(self, "failed_resource_ids", resources)
        devices = tuple(sorted(
            _text("runtime failed device", value)
            for value in self.failed_device_ids
        ))
        if len(devices) != len(set(devices)):
            raise RuntimeExecutionCoordinatorError(
                "runtime failed devices are not unique"
            )
        if (self.phase == HELPER_LOST_PHASE) != bool(devices):
            raise RuntimeExecutionCoordinatorError(
                "runtime failed devices require a helper_lost failure"
            )
        object.__setattr__(self, "failed_device_ids", devices)
        if self.exited_executor_id is not None:
            _text("runtime exited executor", self.exited_executor_id)
            if self.phase not in ELASTIC_FAILURE_PHASES:
                raise RuntimeExecutionCoordinatorError(
                    "runtime exited executor requires an elastic failure"
                )
        if self.masked_executor_id is not None:
            _text("runtime masked-out executor", self.masked_executor_id)
            if self.phase != HELPER_LOST_PHASE or self.exited_executor_id is not None:
                raise RuntimeExecutionCoordinatorError(
                    "runtime masked-out executor requires a helper_lost failure of a live server"
                )

    @property
    def fallback_allowed(self) -> bool:
        return self.retry_safe and (
            not self.execution_started
            or self.phase in RESTARTABLE_STARTED_PHASES
        )

    @property
    def failed_device_id(self) -> str | None:
        """The first lost device (sorted), None unless helper_lost."""
        return self.failed_device_ids[0] if self.failed_device_ids else None


class RuntimeLeaseRenewalCoordinator:
    """Renew one active ticket and expose any failure to its executor."""

    def __init__(
        self,
        ticket: RuntimeRequestTicket,
        *,
        epoch_ns: int,
        guard_us: int,
        quantum_us: int,
        current_ticket: Callable[[str], RuntimeRequestTicket],
        extend: Callable[..., RuntimeLeaseExtensionReceipt],
        on_renewal: Callable[[RuntimeLeaseExtensionReceipt], None] | None,
        current_horizon: Callable[[RuntimeRequestTicket], int] | None = None,
        on_expired: Callable[[RuntimeRequestTicket, int], bool] | None = None,
        maximum_consecutive_expiries: int = 60,
    ) -> None:
        if not isinstance(ticket, RuntimeRequestTicket):
            raise RuntimeExecutionCoordinatorError(
                "runtime renewal ticket is invalid"
            )
        self._epoch_ns = _integer("runtime renewal epoch_ns", epoch_ns)
        self._guard_us = _integer("runtime renewal guard_us", guard_us, 1)
        self._quantum_us = _integer(
            "runtime renewal quantum_us", quantum_us, 1
        )
        if not callable(current_ticket) or not callable(extend):
            raise RuntimeExecutionCoordinatorError(
                "runtime renewal callbacks are invalid"
            )
        if on_renewal is not None and not callable(on_renewal):
            raise RuntimeExecutionCoordinatorError(
                "runtime renewal observer is invalid"
            )
        if current_horizon is not None and not callable(current_horizon):
            raise RuntimeExecutionCoordinatorError(
                "runtime renewal horizon callback is invalid"
            )
        if on_expired is not None and not callable(on_expired):
            raise RuntimeExecutionCoordinatorError(
                "runtime renewal expiry callback is invalid"
            )
        self._on_expired = on_expired
        self._maximum_consecutive_expiries = _integer(
            "runtime renewal maximum consecutive expiries",
            maximum_consecutive_expiries, 1,
        )
        self.request_id = ticket.request.request_id
        self.ticket_id = ticket.ticket_id
        self._current_ticket = current_ticket
        self._extend = extend
        self._on_renewal = on_renewal
        self._current_horizon = current_horizon
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._failure_lock = threading.Lock()
        self._failure: BaseException | None = None
        self.renewals = 0
        self.stalled_renewals = 0
        self.expired_horizons = 0
        self._thread = threading.Thread(
            target=self._run,
            name="runtime-lease-renewal-" + self.request_id,
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _record_failure(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error
        self._stop.set()

    def _effective_horizon_us(self, ticket) -> int:
        """Earliest end among live base leases and the attached-helper horizon.

        Completed preparation-phase leases keep their predicted end in
        final_reserved_until_us and can never be extended, so they are
        excluded; the helper horizon comes from the scheduler callback. One
        definition serves both the wake-up and the post-renewal check.
        """
        live_tokens = tuple(
            row.token for row in ticket.live_leases
            if row.token in ticket.final_reserved_until_us
        )
        horizon_us = min(
            ticket.final_reserved_until_us[token] for token in live_tokens
        ) if live_tokens else min(ticket.final_reserved_until_us.values())
        if self._current_horizon is not None:
            horizon_us = min(
                horizon_us,
                _integer(
                    "runtime renewal current horizon",
                    self._current_horizon(ticket),
                    1,
                ),
            )
        return horizon_us

    def _now_us(self) -> int:
        return max(0, (time.monotonic_ns() - self._epoch_ns) // 1000)

    def _pause(self, pause_us: int) -> bool:
        """Sleep without touching any lease; True when woken to stop."""
        if self._wake.wait(max(0, pause_us) / 1_000_000):
            return self._stop.is_set()
        return False

    def _run(self) -> None:
        try:
            consecutive_expiries = 0
            while not self._stop.is_set():
                self._wake.clear()
                ticket = self._current_ticket(self.request_id)
                if ticket.ticket_id != self.ticket_id:
                    raise RuntimeExecutionCoordinatorError(
                        "runtime renewal ticket identity changed"
                    )
                current_end_us = self._effective_horizon_us(ticket)
                now_us = self._now_us()
                wait_us = max(0, current_end_us - self._guard_us - now_us)
                if self._wake.wait(wait_us / 1_000_000):
                    if self._stop.is_set():
                        return
                    continue
                if self._stop.is_set():
                    return
                now_us = self._now_us()
                end_us = max(
                    current_end_us + self._quantum_us,
                    now_us + self._quantum_us,
                )
                receipt = self._extend(
                    self.request_id,
                    at_us=now_us,
                    reserved_until_us=end_us,
                )
                self.renewals += 1
                if self._on_renewal is not None:
                    self._on_renewal(receipt)
                refreshed = self._current_ticket(self.request_id)
                if refreshed.ticket_id != self.ticket_id:
                    raise RuntimeExecutionCoordinatorError(
                        "runtime renewal ticket identity changed"
                    )
                refreshed_end_us = self._effective_horizon_us(refreshed)
                now_us = self._now_us()
                if refreshed_end_us > current_end_us:
                    consecutive_expiries = 0
                    if refreshed_end_us - self._guard_us <= now_us:
                        # Advanced, but by less than a guard: renew again
                        # only when that validity is about to end, never in
                        # a hot loop. Sleeping here extends nothing.
                        if self._pause(min(
                            self._guard_us, refreshed_end_us - now_us
                        )):
                            return
                    continue
                # The complete horizon did not advance: a stall. Never renew
                # again immediately; retries stay inside the remaining
                # authorization and sleeping extends nothing.
                self.stalled_renewals += 1
                if now_us < refreshed_end_us:
                    if self._pause(min(self._guard_us, refreshed_end_us - now_us)):
                        return
                    continue
                # Authorization expired and renewal could not move it.
                self.expired_horizons += 1
                consecutive_expiries += 1
                handled = (
                    self._on_expired is not None
                    and bool(self._on_expired(refreshed, now_us))
                )
                if not handled or consecutive_expiries > self._maximum_consecutive_expiries:
                    raise RuntimeExecutionCoordinatorError(
                        "runtime lease authorization expired without renewal"
                        f" (horizon {refreshed_end_us} us, now {now_us} us,"
                        f" expiries {consecutive_expiries})"
                    )
                if self._pause(self._guard_us):
                    return
        except BaseException as error:
            self._record_failure(error)

    def diagnostics(self) -> dict[str, int]:
        return {
            "expired_horizons": self.expired_horizons,
            "renewals": self.renewals,
            "stalled_renewals": self.stalled_renewals,
        }

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeExecutionCoordinatorError(
                "runtime lease renewal failed"
            ) from failure

    def wake(self) -> None:
        """Recompute the live lease horizon after a late attachment."""

        self._wake.set()

    def stop(self, timeout_s: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout_s)
        if self._thread.is_alive():
            raise RuntimeExecutionCoordinatorError(
                "runtime lease renewal thread did not stop"
            )
        self.check()
