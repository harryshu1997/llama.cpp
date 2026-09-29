"""Canonical execution coordinator for scheduler-selected physical tickets."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, Sequence, TYPE_CHECKING

from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.runtime_controller import (
    RuntimeCompletionReceipt,
    RuntimeFailureRecovery,
    RuntimeLeaseExtensionReceipt,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from .._internal.runtime_queue import RuntimeDispatchReceipt
from .._internal.runtime_execution import (
    ELASTIC_FAILURE_PHASES,
    HELPER_LOST_PHASE,
    SERVER_EXITED_PHASE,
    RuntimeExecutionFailure,
)
from .contracts import (
    dormant_phone_ffn_parameters,
    PhysicalAdapterError,
    PhysicalBackendFailure,
    PhysicalExecutionBackend,
    RawExecutionObservation,
    RawTransitionObservation,
)
from .receipts import (
    execution_receipt_from_observation,
    transition_receipt_from_observation,
)
from .ticket import (
    PhysicalExecutionCommand,
    PhysicalTransitionCommand,
    bind_ready_helper_to_physical_command,
    interpret_runtime_ticket,
)

if TYPE_CHECKING:
    from ..scheduler import UnifiedScheduler


@dataclass(frozen=True)
class PhysicalAdapterResult:
    ticket: RuntimeRequestTicket
    command: PhysicalExecutionCommand
    observation: RawExecutionObservation
    completion: RuntimeCompletionReceipt
    attempt_ticket_ids: tuple[str, ...]
    dispatch_receipts: tuple[RuntimeDispatchReceipt, ...]
    recoveries: tuple[RuntimeFailureRecovery, ...]
    # Elastic phones (drop recovery): one REQUEST_RECOVERED row per recovered
    # helper_lost / server_exited attempt; empty for every other run.
    recovery_events: tuple[Mapping[str, object], ...] = ()


@dataclass
class _HelperPreparationHandle:
    thread: threading.Thread
    stop: threading.Event


SnapshotProvider = Callable[
    [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
]


def _failure_detail(error: BaseException | None) -> str:
    if error is None:
        return "unspecified"
    value = str(error).encode(
        "ascii", errors="backslashreplace"
    ).decode("ascii")
    value = " ".join(value.split())
    return value[:512] or type(error).__name__


def _physical_failure_reason(failure: PhysicalBackendFailure) -> str:
    """Ticket failure reason; elastic kinds carry the lost device or executor."""
    reason = "physical_backend_failed:" + failure.phase
    if failure.phase == HELPER_LOST_PHASE:
        return reason + ":" + ",".join(failure.failed_device_ids)
    if failure.phase == SERVER_EXITED_PHASE:
        return reason + ":" + str(failure.executor_id)
    return reason


def _streamed_token_count(path: Path) -> int | None:
    """Tokens a partial llama.cpp SSE stream delivered; None when unreadable."""
    count = 0
    try:
        content = path.read_bytes()
    except OSError:
        return None
    for raw_line in content.splitlines():
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        if not line.startswith("data:"):
            continue
        encoded = line[5:].strip()
        if not encoded or encoded == "[DONE]":
            continue
        try:
            value = json.loads(encoded)
        except ValueError:
            # A stream cut inside a chunk: count what was complete.
            break
        if type(value) is not dict:
            return None
        tokens = value.get("tokens", [])
        if type(tokens) is not list:
            return None
        count += len(tokens)
    return count


def _mask_out_recovery(
    failure: PhysicalBackendFailure, fallback: RuntimeRequestTicket
) -> dict[str, object]:
    """How a request whose server masked its lost helper out was recovered (elastic phones S2a).

    ``same_server_mask_out``: the recovery runs on the same live server without a transition (no
    reload); ``other_route``: the scheduler placed it elsewhere. Empty unless the server masked.
    """
    masked = failure.masked_executor_id
    if masked is None:
        return {}
    plan = fallback.execution_plan
    same = (
        fallback.binding.executor_id == masked
        and (plan is None or not plan.transitions)
    )
    return {
        "masked_executor_id": masked,
        "recovery": "same_server_mask_out" if same else "other_route",
    }


def move_partial_stream_aside(payload: object) -> tuple[str | None, int | None]:
    """Rename a failed attempt's stream to ``<name>.attempt<k>`` before re-execution.

    The canonical stream path always holds the terminal attempt (the offline
    identity check reads it); ``k`` counts failed attempts from 1. Returns the
    moved file name (None when the attempt streamed nothing) and its token
    count (None when the partial stream cannot be read).
    """
    stream_path = getattr(payload, "stream_path", None)
    if not isinstance(stream_path, Path) or not stream_path.exists():
        return None, 0
    tokens = _streamed_token_count(stream_path)
    attempt = 1
    while True:
        target = stream_path.with_name(
            stream_path.name + ".attempt" + str(attempt)
        )
        if not target.exists():
            break
        attempt += 1
    stream_path.rename(target)
    return target.name, tokens


HELPER_PREPARATION_FAULT_MODES = frozenset({"post-load-once"})


class HelperPreparationFaultInjector:
    """Inject one deterministic failure into helper preparation for gates.

    ``post-load-once`` raises exactly once after the phone load completed
    and before the scheduler commits, which exercises physical rollback,
    logical rollback, and the retry on real hardware.
    """

    def __init__(self, mode: str | None) -> None:
        if mode is not None and mode not in HELPER_PREPARATION_FAULT_MODES:
            raise PhysicalAdapterError(
                "helper preparation fault injection mode is invalid"
            )
        self.mode = mode
        self._lock = threading.Lock()
        self._records: list[dict[str, object]] = []

    @property
    def records(self) -> tuple[Mapping[str, object], ...]:
        with self._lock:
            return tuple(MappingProxyType(dict(row)) for row in self._records)

    def to_json(self) -> dict[str, object]:
        return {
            "injected": [dict(row) for row in self.records],
            "mode": self.mode,
            "schema": "research-scheduler-helper-fault-injection-v1",
        }

    @staticmethod
    def is_partial_replacement(
        commands: Sequence[PhysicalTransitionCommand],
    ) -> bool:
        """Return whether the loads replace a strict subset of the sessions."""

        return any(
            command.transition.changed_phone_session_ids
            and len(command.transition.changed_phone_session_ids)
                < len(command.transition.phone_shards)
            for command in commands
        )

    def check(
        self,
        point: str,
        *,
        partial_replacement: bool = True,
        **details: object,
    ) -> None:
        if self.mode is None:
            return
        if (
            self.mode == "post-load-once"
            and point == "post-load"
            and partial_replacement
        ):
            with self._lock:
                if self._records:
                    return
                self._records.append({"point": point, **details})
            raise PhysicalAdapterError(
                "injected_helper_preparation_fault:" + point
            )


class CanonicalPhysicalAdapter:
    """Execute only exact tickets returned by one UnifiedScheduler."""

    # Bounded fallback for the per-request helper watcher when no wake
    # generation change is observed (microseconds).
    _helper_watch_fallback_us: int = 1_000_000

    def __init__(
        self,
        scheduler: "UnifiedScheduler",
        backend: PhysicalExecutionBackend,
        *,
        epoch_ns: int,
        snapshot_provider: SnapshotProvider,
        lease_guard_us: int = 250_000,
        lease_quantum_us: int = 2_000_000,
        maximum_replan_snapshot_retries: int = 4,
        on_renewal: Callable[[RuntimeLeaseExtensionReceipt], None]
            | None = None,
        fault_injector: HelperPreparationFaultInjector | None = None,
    ) -> None:
        required = (
            "wait_runtime_request",
            "replan_automated_request",
            "record_automated_transition_receipts",
            "observe_automated_runtime_snapshot",
            "runtime_execution_ticket",
            "start_runtime_lease_renewal",
            "check_runtime_lease_renewal",
            "stop_runtime_lease_renewal",
            "release_automated_runtime_capacity",
            "complete_automated_request",
            "fail_automated_request",
            "runtime_ticket",
            "cancel_runtime_request",
        )
        if any(not callable(getattr(scheduler, name, None)) for name in required):
            raise PhysicalAdapterError("unified scheduler interface is invalid")
        if (
            not callable(getattr(backend, "apply_transition", None))
            or not callable(getattr(backend, "execute", None))
        ):
            raise PhysicalAdapterError("physical backend interface is invalid")
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise PhysicalAdapterError("physical adapter epoch is invalid")
        if not callable(snapshot_provider):
            raise PhysicalAdapterError("runtime snapshot provider is invalid")
        if type(lease_guard_us) is not int or lease_guard_us <= 0:
            raise PhysicalAdapterError("lease renewal guard is invalid")
        if type(lease_quantum_us) is not int or lease_quantum_us <= 0:
            raise PhysicalAdapterError("lease renewal quantum is invalid")
        if (
            type(maximum_replan_snapshot_retries) is not int
            or maximum_replan_snapshot_retries < 1
        ):
            raise PhysicalAdapterError(
                "replan snapshot retry limit is invalid"
            )
        if on_renewal is not None and not callable(on_renewal):
            raise PhysicalAdapterError("lease renewal observer is invalid")
        if fault_injector is not None and not isinstance(
            fault_injector, HelperPreparationFaultInjector
        ):
            raise PhysicalAdapterError("fault injector is invalid")
        self._fault_injector = fault_injector
        self._scheduler = scheduler
        self._backend = backend
        self._epoch_ns = epoch_ns
        self._snapshot_provider = snapshot_provider
        self._lease_guard_us = lease_guard_us
        self._lease_quantum_us = lease_quantum_us
        self._maximum_replan_snapshot_retries = (
            maximum_replan_snapshot_retries
        )
        self._on_renewal = on_renewal
        self._helper_preparations: list[_HelperPreparationHandle] = []
        self._helper_preparation_lock = threading.Lock()
        self._helper_preparation_watchers: set[str] = set()
        self._helper_preparation_rejections: list[dict[str, object]] = []
        self._started_helper_preparations: set[
            tuple[int, str, str]
        ] = set()
        bind_scheduler = getattr(backend, "bind_scheduler", None)
        if callable(bind_scheduler):
            bind_scheduler(scheduler)
        event_clock = getattr(scheduler, "set_phone_layout_event_clock", None)
        if callable(event_clock):
            event_clock(lambda: max(
                0, (time.monotonic_ns() - epoch_ns) // 1000
            ))

    def _snapshot(
        self, ticket: RuntimeRequestTicket, observed_at_us: int
    ) -> HeterogeneousRuntimeSnapshot:
        value = self._snapshot_provider(ticket, observed_at_us)
        if not isinstance(value, HeterogeneousRuntimeSnapshot):
            raise PhysicalAdapterError(
                "runtime snapshot provider returned an invalid snapshot"
            )
        return value

    def _acquire(
        self, ticket: RuntimeRequestTicket
    ) -> tuple[
        RuntimeRequestTicket,
        tuple[RuntimeDispatchReceipt, ...],
        tuple[str, ...],
    ]:
        current = ticket
        receipts = []
        attempt_ticket_ids = [ticket.ticket_id]
        stale_snapshot_retries = 0
        while True:
            current = self._scheduler.wait_runtime_request(
                current.request.request_id, self._epoch_ns
            )
            if getattr(current, "dispatch_state", None) in {
                "CANCELLED", "COMPLETED", "FAILED"
            }:
                raise PhysicalAdapterError(
                    "scheduler request terminated before dispatch: "
                    + current.dispatch_state
                )
            receipt = current.dispatch_receipt
            if receipt is None:
                raise PhysicalAdapterError(
                    "scheduler dispatch wake lacks a receipt"
                )
            receipts.append(receipt)
            if receipt.status == "ACQUIRED":
                return current, tuple(receipts), tuple(attempt_ticket_ids)
            if receipt.status != "REPLAN_REQUIRED":
                raise PhysicalAdapterError(
                    "scheduler dispatch wake is not executable"
                )
            if ticket.decision.reason == "STARTUP_PARENT_PRELOAD":
                raise PhysicalAdapterError("startup exact parent requires a fresh preload plan")
            snapshot = self._snapshot(current, receipt.observed_at_us)
            snapshot_captured_at_us = getattr(
                snapshot, "captured_at_us", receipt.observed_at_us
            )
            request_arrival_us = getattr(
                current.request, "arrival_us", snapshot_captured_at_us
            )
            scheduling_observed_at_us = max(
                request_arrival_us,
                snapshot_captured_at_us,
            )
            try:
                current = self._scheduler.replan_automated_request(
                    current.request.request_id,
                    observed_at_us=scheduling_observed_at_us,
                    reason=(
                        receipt.wake_reason or "resource_state_changed"
                    ),
                    snapshot=snapshot,
                    expected_ticket_id=current.ticket_id,
                    expected_queue_generation=getattr(
                        receipt, "queue_generation", None
                    ),
                )
            except RuntimeReplanRetryRequired as exc:
                stale_snapshot_retries += 1
                if (
                    stale_snapshot_retries
                    >= self._maximum_replan_snapshot_retries
                ):
                    raise PhysicalAdapterError(
                        "runtime replan fresh-snapshot retries exhausted"
                    ) from exc
                latest = self._scheduler.runtime_ticket(
                    current.request.request_id
                )
                if (
                    latest.ticket_id != exc.ticket_id
                    or latest.dispatch_state not in {
                        "QUEUED", "REPLAN_REQUIRED"
                    }
                ):
                    raise PhysicalAdapterError(
                        "runtime replan retry identity changed"
                    ) from exc
                current = latest
                continue
            stale_snapshot_retries = 0
            if current.ticket_id != attempt_ticket_ids[-1]:
                attempt_ticket_ids.append(current.ticket_id)

    def _recover(
        self,
        ticket: RuntimeRequestTicket,
        *,
        failed_at_us: int,
        reason: str,
        failure: RuntimeExecutionFailure,
        transition_receipts: tuple = (),
    ) -> RuntimeFailureRecovery:
        if failure.phase in ELASTIC_FAILURE_PHASES:
            return self._recover_elastic(
                ticket,
                failed_at_us=failed_at_us,
                reason=reason,
                failure=failure,
                transition_receipts=transition_receipts,
            )
        return self._scheduler.fail_automated_request(
            ticket.request.request_id,
            failed_at_us=failed_at_us,
            reason=reason,
            snapshot=self._snapshot(ticket, failed_at_us),
            physical_failure=failure,
            transition_receipts=transition_receipts,
        )

    def _recover_elastic(
        self,
        ticket: RuntimeRequestTicket,
        *,
        failed_at_us: int,
        reason: str,
        failure: RuntimeExecutionFailure,
        transition_receipts: tuple,
    ) -> RuntimeFailureRecovery:
        """Recover a helper_lost / server_exited attempt from a snapshot sampled now.

        The rig has already retired or reaped the failed server (synchronously, while it
        classified the failure), so a snapshot sampled now reports it cold with its memory
        released. The scheduler plans the recovery at that snapshot's capture time and refuses,
        before the ticket changes, a snapshot that does not cover it (``SYSTEM_SNAPSHOT_STALE``,
        e.g. it expired while the recovery waited for the scheduler): a new snapshot is sampled,
        at most ``maximum_replan_snapshot_retries`` times, then the request fails closed.
        """
        attempts = 0
        while True:
            observed_at_us = max(
                failed_at_us, (time.monotonic_ns() - self._epoch_ns) // 1000
            )
            try:
                return self._scheduler.fail_automated_request(
                    ticket.request.request_id,
                    failed_at_us=failed_at_us,
                    reason=reason,
                    snapshot=self._snapshot(ticket, observed_at_us),
                    physical_failure=failure,
                    transition_receipts=transition_receipts,
                )
            except RuntimeReplanRetryRequired as exc:
                attempts += 1
                if (
                    exc.ticket_id != ticket.ticket_id
                    or attempts >= self._maximum_replan_snapshot_retries
                ):
                    raise PhysicalAdapterError(
                        "elastic recovery snapshot retries exhausted: " + exc.reason
                    ) from exc

    def _apply_transitions(
        self,
        ticket: RuntimeRequestTicket,
        command: PhysicalExecutionCommand,
        payload: object,
    ) -> tuple[RuntimeRequestTicket | None, RuntimeFailureRecovery | None]:
        if ticket.transition_status != "PENDING":
            return ticket, None
        receipts = list(ticket.transition_receipts)
        ready = ticket
        request_id = ticket.request.request_id
        self._scheduler.start_runtime_lease_renewal(
            ticket,
            epoch_ns=self._epoch_ns,
            guard_us=self._lease_guard_us,
            quantum_us=self._lease_quantum_us,
            on_renewal=self._on_renewal,
        )
        renewal_failure = None
        transition_failure: PhysicalBackendFailure | None = None
        try:
            for transition_command in command.transitions:
                if any(row.transition_id == transition_command.transition.transition_id for row in receipts):
                    continue
                try:
                    observation = self._backend.apply_transition(
                        transition_command,
                        payload,
                        lambda: self._scheduler.check_runtime_lease_renewal(
                            request_id
                        ),
                    )
                    if not isinstance(observation, RawTransitionObservation):
                        raise PhysicalAdapterError(
                            "physical backend returned an invalid transition"
                        )
                except PhysicalBackendFailure as error:
                    transition_failure = error
                    observation = RawTransitionObservation(
                        started_us=error.started_us,
                        finished_us=error.finished_us,
                        status="FAILED",
                    )
                receipt = transition_receipt_from_observation(
                    transition_command, observation
                )
                receipts.append(receipt)
                if receipt.status == "FAILED":
                    break
                ready = self._scheduler.record_automated_transition_receipts(
                    request_id, tuple(receipts)
                )
        finally:
            try:
                self._scheduler.stop_runtime_lease_renewal(request_id)
            except BaseException as error:
                renewal_failure = error
        if renewal_failure is not None:
            failed_at_us = (
                ticket.decision.start_us
                if not receipts
                else receipts[-1].finished_us
            )
            recovery = self._recover(
                ticket,
                failed_at_us=failed_at_us,
                reason="physical_transition_lease_renewal_failed",
                failure=RuntimeExecutionFailure(
                    phase="lease_renewal",
                    retry_safe=False,
                    execution_started=False,
                ),
            )
            return None, recovery
        if receipts and receipts[-1].status == "FAILED":
            receipt = receipts[-1]
            failure = (
                RuntimeExecutionFailure(
                    phase="residency_transition",
                    retry_safe=True,
                    execution_started=False,
                )
                if transition_failure is None
                else RuntimeExecutionFailure(
                    phase=transition_failure.phase,
                    retry_safe=transition_failure.retry_safe,
                    execution_started=(
                        transition_failure.execution_started
                    ),
                    failed_resource_ids=(
                        transition_failure.failed_resource_ids
                    ),
                )
            )
            recovery = self._recover(
                ticket,
                failed_at_us=receipt.finished_us,
                reason=(
                    "physical_transition_failed:"
                    + receipt.transition_id
                    + ":"
                    + _failure_detail(transition_failure)
                ),
                failure=failure,
                transition_receipts=tuple(receipts),
            )
            return None, recovery
        observed_at_us = max(row.finished_us for row in receipts)
        snapshot = self._snapshot(ready, observed_at_us)
        scheduling_observed_at_us = max(
            observed_at_us, snapshot.captured_at_us
        )
        self._scheduler.observe_automated_runtime_snapshot(
            snapshot,
            observed_at_us=scheduling_observed_at_us,
        )
        return ready, None

    def _quarantine_lost_devices(
        self, failure: PhysicalBackendFailure, at_us: int
    ) -> tuple[str, ...]:
        """Hand each lost helper to the scheduler's device quarantine.

        ``quarantine_device`` belongs to the elastic-phones slice 2 facade;
        until it exists the recovery still excludes the failed device
        (automated_requests_ops.failure), so an absent hook is not an error.
        A stale loss (the device was readmitted after this attempt started,
        so the attempt lost its previous worker) quarantines nothing.
        """
        quarantine = getattr(self._scheduler, "quarantine_device", None)
        if not callable(quarantine):
            return ()
        stale = set(failure.stale_device_ids)
        devices = tuple(
            device_id for device_id in failure.failed_device_ids
            if device_id not in stale
        )
        for device_id in devices:
            quarantine(device_id, reason="HELPER_LOST", at_us=at_us)
        return devices

    def _execute_once(
        self,
        ticket: RuntimeRequestTicket,
        command: PhysicalExecutionCommand,
        payload: object,
        recovery_events: list[dict[str, object]] | None = None,
    ) -> tuple[RawExecutionObservation | None, RuntimeFailureRecovery | None]:
        request_id = ticket.request.request_id
        self._scheduler.start_runtime_lease_renewal(
            ticket,
            epoch_ns=self._epoch_ns,
            guard_us=self._lease_guard_us,
            quantum_us=self._lease_quantum_us,
            on_renewal=self._on_renewal,
        )
        failure: BaseException | None = None
        observation = None
        capacity_released = False
        renewal_stopped = False
        def release_capacity(finished_us: int) -> None:
            nonlocal capacity_released, renewal_stopped
            if capacity_released:
                raise PhysicalAdapterError(
                    "physical capacity release is duplicated"
                )
            self._scheduler.stop_runtime_lease_renewal(request_id)
            renewal_stopped = True
            self._scheduler.release_automated_runtime_capacity(
                request_id,
                finished_us,
                expected_ticket_id=ticket.ticket_id,
            )
            capacity_released = True

        try:
            execute_staged = getattr(
                self._backend, "execute_with_capacity_release", None
            )
            supports_staged = getattr(
                self._backend,
                "supports_staged_capacity_release",
                callable(execute_staged),
            )
            if supports_staged and callable(execute_staged):
                observation = execute_staged(
                    command,
                    payload,
                    lambda: self._scheduler.check_runtime_lease_renewal(
                        request_id
                    ),
                    release_capacity,
                )
            else:
                observation = self._backend.execute(
                    command,
                    payload,
                    lambda: self._scheduler.check_runtime_lease_renewal(
                        request_id
                    ),
                )
            if not isinstance(observation, RawExecutionObservation):
                raise PhysicalAdapterError(
                    "physical backend returned an invalid execution"
                )
        except BaseException as error:
            failure = error
        if not renewal_stopped:
            try:
                self._scheduler.stop_runtime_lease_renewal(request_id)
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(
                        "runtime lease renewal cleanup failed: " + str(error)
                    )
        if failure is None:
            return observation, None
        if capacity_released:
            raise PhysicalAdapterError(
                "physical terminal evidence failed after capacity release"
            ) from failure
        elastic = (
            isinstance(failure, PhysicalBackendFailure)
            and failure.phase in ELASTIC_FAILURE_PHASES
        )
        quarantined: tuple[str, ...] = ()
        if isinstance(failure, PhysicalBackendFailure):
            facts = RuntimeExecutionFailure(
                phase=failure.phase,
                retry_safe=failure.retry_safe,
                execution_started=failure.execution_started,
                failed_resource_ids=failure.failed_resource_ids,
                failed_device_ids=failure.failed_device_ids,
                **({"exited_executor_id": failure.executor_id} if elastic and (
                    failure.phase == SERVER_EXITED_PHASE or failure.returncode is not None
                ) else {}),
                **({"masked_executor_id": failure.masked_executor_id}
                   if elastic and failure.masked_executor_id is not None else {}),
            )
            failed_at_us = failure.finished_us
            reason = _physical_failure_reason(failure)
            if failure.phase == HELPER_LOST_PHASE:
                try:
                    quarantined = self._quarantine_lost_devices(
                        failure, failed_at_us
                    )
                except BaseException as quarantine_error:
                    failure.add_note(
                        "helper device quarantine failed: "
                        + str(quarantine_error)
                    )
                    raise PhysicalAdapterError(reason) from failure
        else:
            facts = RuntimeExecutionFailure(
                phase="execution_control",
                retry_safe=False,
                execution_started=True,
            )
            failed_at_us = max(
                ticket.decision.start_us,
                command.planned_start_us,
                (time.monotonic_ns() - self._epoch_ns) // 1000,
            )
            reason = "physical_execution_control_failed"
        try:
            recovery = self._recover(
                ticket,
                failed_at_us=failed_at_us,
                reason=reason,
                failure=facts,
            )
        except BaseException as recovery_error:
            failure.add_note(
                "scheduler recovery failed: " + str(recovery_error)
            )
            if elastic:
                raise PhysicalAdapterError(
                    reason + "; recovery unavailable: " + _failure_detail(recovery_error)
                ) from failure
            raise PhysicalAdapterError(reason) from failure
        if recovery.fallback is None:
            unavailable = (
                getattr(self._scheduler, "automated_recovery_unavailable_reason", None)
                if elastic else None
            )
            cause = unavailable(request_id) if callable(unavailable) else None
            if cause:
                # e.g. the reload of the recovery does not fit: the request ends with the
                # scheduler's reason and the run stays fail-fast (no output, no RESULT gap)
                raise PhysicalAdapterError(
                    reason + "; recovery unavailable: " + cause
                ) from failure
            raise PhysicalAdapterError(reason) from failure
        if elastic:
            # The fallback re-executes the same payload from its prompt:
            # keep the partial stream as evidence and free the canonical path.
            attempt_stream, tokens_discarded = move_partial_stream_aside(
                payload
            )
            event = {
                "attempt_started_us": failure.started_us,
                "attempt_stream": attempt_stream,
                "device_id": failure.failed_device_id,
                "executor_id": failure.executor_id,
                "failed_at_us": failed_at_us,
                "failed_device_ids": list(failure.failed_device_ids),
                "failed_ticket": recovery.failed_ticket_id,
                "failure_kind": failure.phase,
                "kind": "REQUEST_RECOVERED",
                "new_ticket": recovery.fallback.ticket_id,
                # wall time the failed attempt occupied before the failure
                "penalty_us": max(
                    0, failure.finished_us - failure.started_us
                ),
                "quarantine_action": recovery.quarantine_action,
                "quarantined_device_ids": list(quarantined),
                "request": request_id,
                "returncode": failure.returncode,
                **({"stale_device_ids": list(failure.stale_device_ids)}
                   if failure.stale_device_ids else {}),
                **_mask_out_recovery(failure, recovery.fallback),
                "tokens_discarded": tokens_discarded,
            }
            if recovery_events is not None:
                recovery_events.append(event)
        return None, recovery

    def _start_helper_preparation(
        self,
        ticket: RuntimeRequestTicket,
        command: PhysicalExecutionCommand,
        payload: object,
    ) -> _HelperPreparationHandle | None:
        dynamic_reader = getattr(
            self._scheduler,
            "runtime_request_helper_preparation_envelope",
            None,
        )
        dormant = dormant_phone_ffn_parameters(
            command.adapter_parameters
        )
        if (
            command.helper_envelope is None
            and not callable(dynamic_reader)
            and dormant is None
        ):
            return None
        required = (
            "begin_request_helper_preparation",
            "check_request_helper_preparation",
            "complete_request_helper_preparation",
            "fail_request_helper_preparation",
        )
        if any(
            not callable(getattr(self._scheduler, name, None))
            for name in required
        ):
            raise PhysicalAdapterError(
                "request helper scheduler interface is invalid"
            )
        request_id = ticket.request.request_id
        with self._helper_preparation_lock:
            if request_id in self._helper_preparation_watchers:
                return None
            self._helper_preparation_watchers.add(request_id)
        stop = threading.Event()

        def now_us(current: RuntimeRequestTicket) -> int:
            return max(
                current.request.arrival_us,
                (time.monotonic_ns() - self._epoch_ns) // 1000,
            )

        def desktop_command(
            current_command: PhysicalExecutionCommand,
        ) -> PhysicalExecutionCommand:
            if current_command.helper_envelope is None:
                return current_command
            operator_plan = dict(current_command.operator_plan)
            operator_plan.pop("helper_envelope", None)
            return replace(
                current_command,
                operator_plan=MappingProxyType(operator_plan),
                helper_envelope=None,
                helper_transitions=(),
            )

        wake_generation = getattr(
            self._scheduler, "runtime_helper_state_generation", None
        )
        if not callable(wake_generation):
            wake_generation = None

        def worker() -> None:
            last_ready_refresh_us = -5_000_000
            begin_rejections = 0
            seen_generation = None
            last_pass_us = None
            try:
                while not stop.is_set():
                    current = self._scheduler.runtime_ticket(request_id)
                    if current.dispatch_state in {
                        "CANCELLED", "COMPLETED", "FAILED"
                    }:
                        return
                    if wake_generation is not None:
                        # Re-plan only when helper-relevant scheduler state
                        # moved (layout events, acquisition, helper events)
                        # or the bounded fallback interval elapsed; the
                        # snapshot capture and envelope projection below are
                        # far too expensive to repeat every 50 ms while a
                        # queued request waits for nothing.
                        generation = wake_generation()
                        if (
                            generation == seen_generation
                            and last_pass_us is not None
                            and now_us(current) - last_pass_us
                                < self._helper_watch_fallback_us
                        ):
                            stop.wait(0.05)
                            continue
                        seen_generation = generation
                        last_pass_us = now_us(current)
                    preparation_allowed = getattr(
                        self._scheduler,
                        "runtime_background_helper_preparation_allowed",
                        None,
                    )
                    if (
                        callable(preparation_allowed)
                        and not preparation_allowed(
                            request_id,
                            expected_ticket_id=current.ticket_id,
                        )
                    ):
                        stop.wait(0.05)
                        continue
                    current_command = interpret_runtime_ticket(current)
                    observed_at_us = now_us(current)
                    snapshot = self._snapshot(current, observed_at_us)
                    observed_at_us = max(
                        observed_at_us, snapshot.captured_at_us
                    )
                    refresh_helper = getattr(
                        self._scheduler,
                        "refresh_ready_request_helper",
                        None,
                    )
                    refresh_needed = getattr(
                        self._scheduler,
                        "runtime_ready_helper_refresh_needed",
                        None,
                    )
                    needs_refresh = bool(
                        callable(refresh_needed)
                        and refresh_needed(
                            request_id,
                            expected_ticket_id=current.ticket_id,
                        )
                    )
                    helper = None
                    if needs_refresh:
                        refreshed = False
                        if (
                            callable(refresh_helper)
                            and observed_at_us - last_ready_refresh_us
                                >= 5_000_000
                        ):
                            last_ready_refresh_us = observed_at_us
                            refreshed = refresh_helper(
                                request_id,
                                expected_ticket_id=current.ticket_id,
                                observed_at_us=observed_at_us,
                                snapshot=snapshot,
                            )
                        if refreshed:
                            placement_stats = getattr(
                                self._scheduler,
                                "model_placement_controller_stats",
                                None,
                            )
                            if (
                                not callable(placement_stats)
                                or placement_stats().get(
                                    "target_phone_layout_state"
                                ) not in {"PROPOSED", "PREPARING"}
                            ):
                                return
                        if callable(dynamic_reader):
                            helper = dynamic_reader(
                                request_id,
                                expected_ticket_id=current.ticket_id,
                                observed_at_us=observed_at_us,
                                snapshot=snapshot,
                            )
                        if helper is None:
                            if refreshed:
                                return
                            stop.wait(1.0)
                            continue
                    else:
                        helper = current_command.helper_envelope
                        if callable(dynamic_reader):
                            helper = dynamic_reader(
                                request_id,
                                expected_ticket_id=current.ticket_id,
                                observed_at_us=observed_at_us,
                                snapshot=snapshot,
                            )
                    if helper is None:
                        stop.wait(0.05)
                        continue
                    preparation_key = (
                        helper.phone_layout_generation,
                        helper.phone_layout_geometry_sha256,
                        helper.operator_plan_sha256,
                    )
                    with self._helper_preparation_lock:
                        preparation_started = preparation_key in (
                            self._started_helper_preparations
                        )
                    if preparation_started:
                        stop.wait(1.0)
                        continue
                    helper_command = current_command
                    if current_command.helper_envelope != helper:
                        helper_command = bind_ready_helper_to_physical_command(
                            desktop_command(current_command), helper
                        )
                    preparation_ticket_id = helper.preparation_ticket_id(
                        current.ticket_id
                    )
                    try:
                        decision = (
                            self._scheduler.begin_request_helper_preparation(
                                request_id,
                                observed_at_us=observed_at_us,
                                snapshot=snapshot,
                                expected_phone_layout_generation=(
                                    helper.phone_layout_generation
                                ),
                                expected_phone_layout_geometry_sha256=(
                                    helper.phone_layout_geometry_sha256
                                ),
                                expected_operator_plan_sha256=(
                                    helper.operator_plan_sha256
                                ),
                            )
                        )
                    except ValueError as error:
                        # A rejected proposal must not end the watcher: the
                        # scheduler restored its transaction, so retry after
                        # the layout state settles.
                        begin_rejections += 1
                        with self._helper_preparation_lock:
                            rejections = getattr(
                                self, "_helper_preparation_rejections", None
                            )
                            if rejections is None:
                                rejections = []
                                self._helper_preparation_rejections = (
                                    rejections
                                )
                            rejections.append({
                                "preparation_ticket_id": preparation_ticket_id,
                                "reason": _failure_detail(error),
                                "request_id": request_id,
                            })
                        if begin_rejections > 60:
                            raise
                        stop.wait(1.0)
                        continue
                    begin_rejections = 0
                    status = decision.get("status")
                    if status == "READY":
                        with self._helper_preparation_lock:
                            self._started_helper_preparations.add(
                                preparation_key
                            )
                        stop.wait(0.05)
                        continue
                    if status in {
                        "DEFERRED", "FOLLOWER", "INCOMPATIBLE",
                        "NOT_REQUIRED",
                    }:
                        stop.wait(0.05)
                        continue
                    if status != "OWNER":
                        raise PhysicalAdapterError(
                            "request helper preparation decision is invalid"
                        )
                    with self._helper_preparation_lock:
                        self._started_helper_preparations.add(
                            preparation_key
                        )
                    receipts = []
                    applied_commands = []

                    def control_check() -> None:
                        if stop.is_set():
                            raise PhysicalAdapterError(
                                "request helper preparation stopped"
                            )
                        self._scheduler.check_request_helper_preparation(
                            preparation_ticket_id,
                            observed_at_us=now_us(current),
                        )

                    try:
                        for transition_command in (
                            helper_command.helper_transitions
                        ):
                            observation = self._backend.apply_transition(
                                transition_command, payload, control_check
                            )
                            if not isinstance(
                                observation, RawTransitionObservation
                            ):
                                raise PhysicalAdapterError(
                                    "helper backend returned an invalid "
                                    "transition"
                                )
                            receipt = transition_receipt_from_observation(
                                transition_command, observation
                            )
                            if receipt.status != "COMPLETED":
                                raise PhysicalAdapterError(
                                    "request helper transition failed"
                                )
                            receipts.append(receipt)
                            applied_commands.append(transition_command)
                        if not receipts:
                            raise PhysicalAdapterError(
                                "helper preparation transition is absent"
                            )
                        finished_at_us = max(
                            row.finished_us for row in receipts
                        )
                        if self._fault_injector is not None:
                            self._fault_injector.check(
                                "post-load",
                                partial_replacement=(
                                    self._fault_injector
                                    .is_partial_replacement(applied_commands)
                                ),
                                finished_at_us=finished_at_us,
                                preparation_ticket_id=preparation_ticket_id,
                                request_id=request_id,
                                transition_ids=[
                                    row.transition_id for row in receipts
                                ],
                            )
                        completion_snapshot = self._snapshot(
                            current, finished_at_us
                        )
                        self._scheduler.complete_request_helper_preparation(
                            request_id,
                            preparation_ticket_id,
                            tuple(receipts),
                            snapshot=completion_snapshot,
                        )
                    except BaseException as error:
                        # The phone may already hold the new shard: undo it
                        # physically before the scheduler restores the old
                        # layout, and mark sessions whose restoration is not
                        # proven so they are never republished as resident.
                        (
                            unavailable,
                            restored_generations,
                        ) = self._rollback_helper_transitions(
                            applied_commands, error
                        )
                        failure_options: dict[str, object] = {}
                        if unavailable:
                            failure_options["unavailable_session_ids"] = (
                                unavailable
                            )
                        if restored_generations:
                            failure_options[
                                "restored_session_generations"
                            ] = restored_generations
                        try:
                            failed_at_us = now_us(current)
                            failure_snapshot = self._snapshot(
                                current, failed_at_us
                            )
                            failed_at_us = max(
                                failed_at_us,
                                failure_snapshot.captured_at_us,
                            )
                            self._scheduler.fail_request_helper_preparation(
                                request_id,
                                preparation_ticket_id,
                                failed_at_us=failed_at_us,
                                reason=(
                                    "physical_helper_preparation_failed:"
                                    + _failure_detail(error)
                                ),
                                snapshot=failure_snapshot,
                                **failure_options,
                            )
                        except BaseException:
                            pass
                        with self._helper_preparation_lock:
                            self._started_helper_preparations.discard(
                                preparation_key
                            )
                    stop.wait(0.05)
            finally:
                with self._helper_preparation_lock:
                    self._helper_preparation_watchers.discard(request_id)

        thread = threading.Thread(
            target=worker,
            name="request-helper-" + request_id,
            daemon=True,
        )
        handle = _HelperPreparationHandle(thread=thread, stop=stop)
        with self._helper_preparation_lock:
            self._helper_preparations.append(handle)
        thread.start()
        return handle

    def _rollback_helper_transitions(
        self,
        commands: list[PhysicalTransitionCommand],
        error: BaseException,
    ) -> tuple[tuple[str, ...], Mapping[str, int]]:
        """Undo applied helper loads and return their physical epochs."""

        rollback = getattr(self._backend, "rollback_transition", None)
        unavailable: set[str] = set()
        restored_generations: dict[str, int] = {}
        for command in reversed(commands):
            changed = tuple(command.transition.changed_phone_session_ids)
            if not callable(rollback):
                unavailable.update(changed)
                error.add_note(
                    "helper transition rollback is unavailable: "
                    + command.transition.transition_id
                )
                continue
            try:
                receipt = rollback(command)
            except BaseException as rollback_error:
                unavailable.update(changed)
                error.add_note(
                    "helper transition rollback failed: "
                    + _failure_detail(rollback_error)
                )
                continue
            physical_change = bool(
                isinstance(receipt, Mapping)
                and receipt.get("physical_change")
            )
            if not physical_change:
                continue
            restored_empty = set(receipt.get(
                "restored_empty_session_ids", ()
            ))
            if restored_empty == set(changed):
                continue
            raw_generations = receipt.get(
                "restored_session_generations", {}
            )
            if not isinstance(raw_generations, Mapping) or any(
                type(raw_generations.get(session_id)) is not int
                or int(raw_generations[session_id]) < 1
                for session_id in changed
            ):
                unavailable.update(changed)
                error.add_note(
                    "helper transition rollback lacks physical epochs: "
                    + command.transition.transition_id
                )
                continue
            restored_generations.update({
                session_id: int(raw_generations[session_id])
                for session_id in changed
            })
        for session_id in unavailable:
            restored_generations.pop(session_id, None)
        return (
            tuple(sorted(unavailable)),
            MappingProxyType(dict(sorted(restored_generations.items()))),
        )

    def _execute(
        self, ticket: RuntimeRequestTicket, payload: object
    ) -> PhysicalAdapterResult:
        """Run scheduler-owned replans and execute the returned exact binding."""
        if not isinstance(ticket, RuntimeRequestTicket):
            raise PhysicalAdapterError("physical adapter ticket is invalid")
        attempts = []
        dispatch_receipts = []
        recoveries = []
        recovery_events: list[dict[str, object]] = []
        current = ticket
        self._start_helper_preparation(
            current, interpret_runtime_ticket(current), payload
        )
        while True:
            current, wakes, acquired_attempts = self._acquire(current)
            for ticket_id in acquired_attempts:
                if not attempts or attempts[-1] != ticket_id:
                    attempts.append(ticket_id)
            dispatch_receipts.extend(wakes)
            command = interpret_runtime_ticket(current)
            current, transition_recovery = self._apply_transitions(
                current, command, payload
            )
            if transition_recovery is not None:
                recoveries.append(transition_recovery)
                if ticket.decision.reason == "STARTUP_PARENT_PRELOAD":
                    raise PhysicalAdapterError("startup exact parent transition failed")
                if transition_recovery.fallback is None:
                    raise PhysicalAdapterError(
                        "physical transition failed without a fallback"
                    )
                current = transition_recovery.fallback
                continue
            if current is None:
                raise PhysicalAdapterError("physical transition state is invalid")
            current = self._scheduler.runtime_execution_ticket(
                current.request.request_id
            )
            refresh_helper = getattr(
                self._scheduler, "refresh_ready_request_helper", None
            )
            refresh_needed = getattr(
                self._scheduler,
                "runtime_ready_helper_refresh_needed",
                None,
            )
            if (
                callable(refresh_helper)
                and callable(refresh_needed)
                and refresh_needed(
                    current.request.request_id,
                    expected_ticket_id=current.ticket_id,
                )
            ):
                observed_at_us = max(
                    current.request.arrival_us,
                    (time.monotonic_ns() - self._epoch_ns) // 1000,
                )
                snapshot = self._snapshot(current, observed_at_us)
                observed_at_us = max(
                    observed_at_us, snapshot.captured_at_us
                )
                refresh_helper(
                    current.request.request_id,
                    expected_ticket_id=current.ticket_id,
                    observed_at_us=observed_at_us,
                    snapshot=snapshot,
                )
            command = interpret_runtime_ticket(current)
            self._start_helper_preparation(current, command, payload)
            observation, execution_recovery = self._execute_once(
                current, command, payload, recovery_events
            )
            if execution_recovery is not None:
                recoveries.append(execution_recovery)
                if ticket.decision.reason == "STARTUP_PARENT_PRELOAD":
                    raise PhysicalAdapterError("startup exact parent verification failed")
                current = execution_recovery.fallback
                if current is None:
                    raise PhysicalAdapterError(
                        "physical execution failed without a fallback"
                    )
                continue
            if observation is None:
                raise PhysicalAdapterError("physical execution result is absent")
            receipt = execution_receipt_from_observation(
                command, observation
            )
            completion = self._scheduler.complete_automated_request(
                current.request.request_id,
                receipt,
                snapshot_provider=self._snapshot,
            )
            return PhysicalAdapterResult(
                ticket=current,
                command=command,
                observation=observation,
                completion=completion,
                attempt_ticket_ids=tuple(attempts),
                dispatch_receipts=tuple(dispatch_receipts),
                recoveries=tuple(recoveries),
                recovery_events=tuple(
                    MappingProxyType(row) for row in recovery_events
                ),
            )

    def execute(
        self, ticket: RuntimeRequestTicket, payload: object
    ) -> PhysicalAdapterResult:
        """Execute one ticket and close scheduler state on coordinator errors."""
        try:
            return self._execute(ticket, payload)
        except BaseException:
            try:
                if isinstance(ticket, RuntimeRequestTicket):
                    current = self._scheduler.runtime_ticket(
                        ticket.request.request_id
                    )
                    if current.dispatch_state not in {
                        "CANCELLED", "COMPLETED", "FAILED"
                    }:
                        at_us = max(
                            current.request.arrival_us,
                            (time.monotonic_ns() - self._epoch_ns) // 1000,
                        )
                        cancel = getattr(
                            self._scheduler,
                            "cancel_runtime_request_if_pending",
                            self._scheduler.cancel_runtime_request,
                        )
                        cancel(
                            current.request.request_id,
                            at_us,
                            "physical_adapter_abort",
                        )
            except BaseException:
                pass
            raise
