"""Arrival-time coordination for scheduler-selected physical execution."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import threading
import time
from types import MappingProxyType
from typing import Callable, Mapping, NoReturn

from .._internal.lifecycle import RequestShapeUnsupportedError
from .._internal.policy import Request
from .._internal.runtime_capabilities import HeterogeneousRuntimeSnapshot
from .._internal.runtime_controller import (
    RUNTIME_SELECTION_MODES,
    RuntimeReplanRetryRequired,
    RuntimeRequestTicket,
)
from .contracts import PhysicalAdapterError, PhysicalExecutionBackend
from .runtime import (
    CanonicalPhysicalAdapter,
    HelperPreparationFaultInjector,
    PhysicalAdapterResult,
)


SnapshotProvider = Callable[
    [RuntimeRequestTicket, int], HeterogeneousRuntimeSnapshot
]
BackendFactory = Callable[
    [RuntimeRequestTicket, object], PhysicalExecutionBackend
]


class _PooledPhysicalBackend:
    def __init__(
        self,
        backend: PhysicalExecutionBackend,
        pool: concurrent.futures.ThreadPoolExecutor,
    ) -> None:
        self._backend = backend
        self._pool = pool
        self.supports_staged_capacity_release = callable(getattr(
            backend, "execute_with_capacity_release", None
        ))

    def bind_scheduler(self, scheduler: object) -> None:
        bind = getattr(self._backend, "bind_scheduler", None)
        if callable(bind):
            bind(scheduler)

    def apply_transition(self, command, payload, control_check):
        return self._pool.submit(
            self._backend.apply_transition,
            command,
            payload,
            control_check,
        ).result()

    def rollback_transition(self, command):
        rollback = getattr(self._backend, "rollback_transition", None)
        if not callable(rollback):
            raise PhysicalAdapterError(
                "physical backend cannot roll back transitions"
            )
        return self._pool.submit(rollback, command).result()

    def execute(self, command, payload, control_check):
        return self._pool.submit(
            self._backend.execute,
            command,
            payload,
            control_check,
        ).result()

    def execute_with_capacity_release(
        self, command, payload, control_check, capacity_release
    ):
        staged = getattr(
            self._backend, "execute_with_capacity_release", None
        )
        if callable(staged):
            return self._pool.submit(
                staged,
                command,
                payload,
                control_check,
                capacity_release,
            ).result()
        observation = self.execute(command, payload, control_check)
        capacity_release(observation.finished_us)
        return observation


@dataclass(frozen=True)
class CanonicalRuntimeSubmission:
    """One observed arrival and its non-policy physical payload."""

    request: Request
    model_id: str
    snapshot: HeterogeneousRuntimeSnapshot
    payload: object
    selection_mode: str = "energy-aware"

    def __post_init__(self) -> None:
        if not isinstance(self.request, Request):
            raise PhysicalAdapterError("runtime submission request is invalid")
        self.request.validate()
        if (
            type(self.model_id) is not str
            or not self.model_id
            or not self.model_id.isascii()
        ):
            raise PhysicalAdapterError("runtime submission model id is invalid")
        if not isinstance(self.snapshot, HeterogeneousRuntimeSnapshot):
            raise PhysicalAdapterError("runtime submission snapshot is invalid")
        if self.selection_mode not in RUNTIME_SELECTION_MODES:
            raise PhysicalAdapterError(
                "runtime submission selection mode is invalid"
            )


@dataclass(frozen=True)
class RuntimeSubmissionRejection:
    """A request the scheduler refused at submission as permanently unsupported."""

    request_id: str
    model_id: str
    arrival_us: int
    observed_at_us: int
    reason: str
    details: Mapping[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "arrival_us": self.arrival_us,
            "details": dict(sorted(self.details.items())),
            "model_id": self.model_id,
            "observed_at_us": self.observed_at_us,
            "reason": self.reason,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class CanonicalRuntimeTraceResult:
    """Completed scheduler tickets in causal submission order."""

    request_ids: tuple[str, ...]
    tickets: Mapping[str, RuntimeRequestTicket]
    executions: Mapping[str, PhysicalAdapterResult]

    def __post_init__(self) -> None:
        request_ids = tuple(self.request_ids)
        tickets = dict(self.tickets)
        executions = dict(self.executions)
        if (
            not request_ids
            or len(request_ids) != len(set(request_ids))
            or set(request_ids) != set(tickets)
            or set(request_ids) != set(executions)
        ):
            raise PhysicalAdapterError("runtime trace result coverage differs")
        for request_id in request_ids:
            ticket = tickets[request_id]
            execution = executions[request_id]
            if (
                not isinstance(ticket, RuntimeRequestTicket)
                or not isinstance(execution, PhysicalAdapterResult)
                or ticket.request.request_id != request_id
                or execution.ticket.request.request_id != request_id
                or execution.command.executor_id
                    != execution.ticket.binding.executor_id
                or execution.command.route_id
                    != execution.ticket.decision.route_id
            ):
                raise PhysicalAdapterError(
                    "runtime trace result physical binding differs"
                )
        object.__setattr__(self, "request_ids", request_ids)
        object.__setattr__(
            self, "tickets", MappingProxyType(dict(tickets))
        )
        object.__setattr__(
            self, "executions", MappingProxyType(dict(executions))
        )


class CanonicalArrivalCoordinator:
    """Submit every arrival to one scheduler and execute only its ticket.

    With fail_fast (the default) the first failed lifecycle aborts the run
    at the next arrival wait or submission, exactly as drain() would abort
    it; fail_fast=False surfaces lifecycle failures only in drain().
    """

    def __init__(
        self,
        scheduler: object,
        backend: PhysicalExecutionBackend | None,
        *,
        epoch_ns: int,
        snapshot_provider: SnapshotProvider,
        max_workers: int,
        backend_factory: BackendFactory | None = None,
        lease_guard_us: int = 250_000,
        lease_quantum_us: int = 2_000_000,
        on_renewal: Callable[[object], None] | None = None,
        helper_preparation_fault_injection: str | None = None,
        fail_fast: bool = True,
    ) -> None:
        if type(max_workers) is not int or max_workers <= 0:
            raise PhysicalAdapterError(
                "runtime arrival worker count is invalid"
            )
        if type(epoch_ns) is not int or epoch_ns < 0:
            raise PhysicalAdapterError("runtime arrival epoch is invalid")
        if type(fail_fast) is not bool:
            raise PhysicalAdapterError(
                "runtime arrival fail-fast flag is invalid"
            )
        if (backend is None) == (backend_factory is None):
            raise PhysicalAdapterError(
                "runtime arrival requires one physical backend source"
            )
        if backend_factory is not None and not callable(backend_factory):
            raise PhysicalAdapterError(
                "runtime physical backend factory is invalid"
            )
        self._scheduler = scheduler
        self._epoch_ns = epoch_ns
        event_clock = getattr(scheduler, "set_phone_layout_event_clock", None)
        if callable(event_clock):
            event_clock(lambda: max(
                0, (time.monotonic_ns() - epoch_ns) // 1000
            ))
        self._backend = backend
        self._backend_factory = backend_factory
        self._snapshot_provider = snapshot_provider
        self._lease_guard_us = lease_guard_us
        self._lease_quantum_us = lease_quantum_us
        self._on_renewal = on_renewal
        self.helper_preparation_fault_injector = (
            HelperPreparationFaultInjector(
                helper_preparation_fault_injection
            )
        )
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="unified-physical",
        )
        self._lock = threading.Lock()
        self._closed = False
        self._request_ids: list[str] = []
        self._tickets: dict[str, RuntimeRequestTicket] = {}
        self._rejections: dict[str, RuntimeSubmissionRejection] = {}
        self._futures: dict[
            str, concurrent.futures.Future[PhysicalAdapterResult]
        ] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._fail_fast = fail_fast
        self._lifecycle_failed = threading.Event()
        self._fail_fast_primary: BaseException | None = None

    @staticmethod
    def _lifecycle_error(
        future: concurrent.futures.Future[PhysicalAdapterResult],
    ) -> BaseException | None:
        if not future.done() or future.cancelled():
            return None
        return future.exception()

    def _note_lifecycle_outcome(
        self, future: concurrent.futures.Future[PhysicalAdapterResult]
    ) -> None:
        if self._lifecycle_error(future) is not None:
            self._lifecycle_failed.set()

    def _raise_on_lifecycle_failure(self) -> None:
        """Abort as drain() does once any submitted lifecycle has failed."""
        if not self._lifecycle_failed.is_set():
            return
        with self._lock:
            request_ids = tuple(self._request_ids)
            futures = dict(self._futures)
        for request_id in request_ids:
            primary = self._lifecycle_error(futures[request_id])
            if primary is not None:
                break
        else:
            raise PhysicalAdapterError(
                "runtime lifecycle failure is not recorded"
            )
        if self._fail_fast_primary is not primary:
            self._fail_fast_primary = primary
            primary.add_note(
                "arrival coordinator stopped arrivals at "
                + str(self.observed_at_us())
                + " us: lifecycle of " + request_id + " failed"
            )
        self._raise_after_abort(
            primary, futures[request_id], request_ids, futures, None
        )

    @staticmethod
    def _run_lifecycle(
        future: concurrent.futures.Future[PhysicalAdapterResult],
        adapter: CanonicalPhysicalAdapter,
        ticket: RuntimeRequestTicket,
        payload: object,
    ) -> None:
        if not future.set_running_or_notify_cancel():
            return
        try:
            result = adapter.execute(ticket, payload)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    def observed_at_us(self) -> int:
        return max(0, (time.monotonic_ns() - self._epoch_ns) // 1000)

    def wait_for_arrival(self, arrival_us: int) -> int:
        if type(arrival_us) is not int or arrival_us < 0:
            raise PhysicalAdapterError("runtime arrival time is invalid")
        target_ns = self._epoch_ns + arrival_us * 1000
        while True:
            self._raise_on_lifecycle_failure()
            remaining_ns = target_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return self.observed_at_us()
            pause_s = min(remaining_ns / 1_000_000_000, 0.01)
            if self._fail_fast:
                self._lifecycle_failed.wait(pause_s)
            else:
                time.sleep(pause_s)

    def rejections(self) -> tuple[RuntimeSubmissionRejection, ...]:
        """Requests refused at submission, in arrival order."""
        with self._lock:
            return tuple(self._rejections.values())

    def submit(
        self,
        submission: CanonicalRuntimeSubmission,
        *,
        observed_at_us: int | None = None,
    ) -> RuntimeRequestTicket | None:
        """Schedule an observed arrival before starting physical execution.

        Returns None when the scheduler rejects the request as permanently
        unsupported (RequestShapeUnsupportedError); the rejection is kept in
        rejections() with its exact reason and every other request keeps
        running. Any other scheduling error still propagates, and with
        fail_fast a failed lifecycle is raised before this arrival is
        scheduled.
        """
        if not isinstance(submission, CanonicalRuntimeSubmission):
            raise PhysicalAdapterError("runtime arrival submission is invalid")
        self._raise_on_lifecycle_failure()
        observed_at_us = (
            self.observed_at_us()
            if observed_at_us is None else observed_at_us
        )
        if (
            type(observed_at_us) is not int
            or observed_at_us < submission.request.arrival_us
        ):
            raise PhysicalAdapterError(
                "runtime request was submitted before its arrival"
            )
        scheduling_observed_at_us = max(
            observed_at_us, submission.snapshot.captured_at_us
        )
        try:
            submission.snapshot.validate_at(scheduling_observed_at_us)
        except ValueError as exc:
            raise PhysicalAdapterError(str(exc)) from exc
        request_id = submission.request.request_id
        with self._lock:
            if self._closed:
                raise PhysicalAdapterError("runtime arrival coordinator is closed")
            if request_id in self._tickets or request_id in self._rejections:
                raise PhysicalAdapterError("runtime arrival is duplicated")
            snapshot = submission.snapshot
            snapshot_retries = 0
            while True:
                self._scheduler.observe_automated_runtime_snapshot(
                    snapshot,
                    observed_at_us=scheduling_observed_at_us,
                )
                try:
                    ticket = self._scheduler.submit_automated_request(
                        submission.request,
                        submission.model_id,
                        snapshot,
                        observed_at_us=scheduling_observed_at_us,
                        selection_mode=submission.selection_mode,
                    )
                except RequestShapeUnsupportedError as exc:
                    self._rejections[request_id] = RuntimeSubmissionRejection(
                        request_id=request_id,
                        model_id=submission.model_id,
                        arrival_us=submission.request.arrival_us,
                        observed_at_us=scheduling_observed_at_us,
                        reason=exc.reason,
                        details=MappingProxyType(dict(exc.details)),
                    )
                    return None
                except RuntimeReplanRetryRequired as exc:
                    snapshot_retries += 1
                    if snapshot_retries >= 4:
                        raise PhysicalAdapterError(
                            "runtime arrival fresh-snapshot retries exhausted"
                        ) from exc
                    predecessor = self._scheduler.runtime_ticket(
                        exc.request_id
                    )
                    sampled_at_us = max(
                        scheduling_observed_at_us,
                        self.observed_at_us(),
                    )
                    snapshot = self._snapshot_provider(
                        predecessor, sampled_at_us
                    )
                    if not isinstance(
                        snapshot, HeterogeneousRuntimeSnapshot
                    ):
                        raise PhysicalAdapterError(
                            "runtime snapshot provider returned an invalid "
                            "snapshot"
                        )
                    scheduling_observed_at_us = max(
                        submission.request.arrival_us,
                        snapshot.captured_at_us,
                    )
                    snapshot.validate_at(scheduling_observed_at_us)
                    continue
                break
            backend = (
                self._backend
                if self._backend_factory is None
                else self._backend_factory(ticket, submission.payload)
            )
            physical_backend = _PooledPhysicalBackend(
                backend, self._pool
            )
            adapter = CanonicalPhysicalAdapter(
                self._scheduler,
                physical_backend,
                epoch_ns=self._epoch_ns,
                snapshot_provider=self._snapshot_provider,
                lease_guard_us=self._lease_guard_us,
                lease_quantum_us=self._lease_quantum_us,
                on_renewal=self._on_renewal,
                fault_injector=self.helper_preparation_fault_injector,
            )
            future: concurrent.futures.Future[
                PhysicalAdapterResult
            ] = concurrent.futures.Future()
            if self._fail_fast:
                future.add_done_callback(self._note_lifecycle_outcome)
            thread = threading.Thread(
                target=self._run_lifecycle,
                args=(future, adapter, ticket, submission.payload),
                name="unified-runtime-" + request_id,
            )
            self._request_ids.append(request_id)
            self._tickets[request_id] = ticket
            self._futures[request_id] = future
            self._threads[request_id] = thread
            thread.start()
        return ticket

    def submit_at_arrival(
        self, submission: CanonicalRuntimeSubmission
    ) -> RuntimeRequestTicket:
        observed_at_us = self.wait_for_arrival(
            submission.request.arrival_us
        )
        return self.submit(submission, observed_at_us=observed_at_us)

    def _abort_outstanding(
        self,
        request_ids: tuple[str, ...],
        futures: Mapping[
            str, concurrent.futures.Future[PhysicalAdapterResult]
        ],
    ) -> tuple[tuple[BaseException, ...], frozenset[str]]:
        errors = []
        cancelled_request_ids = set()
        observed_at_us = self.observed_at_us()
        for request_id in request_ids:
            try:
                ticket = self._scheduler.runtime_ticket(request_id)
                if ticket.dispatch_state not in {
                    "CANCELLED", "COMPLETED", "FAILED"
                }:
                    cancel = getattr(
                        self._scheduler,
                        "cancel_runtime_request_if_pending",
                        self._scheduler.cancel_runtime_request,
                    )
                    cancel(
                        request_id,
                        max(ticket.request.arrival_us, observed_at_us),
                        "arrival_coordinator_abort",
                    )
                    cancelled_request_ids.add(request_id)
            except BaseException as error:
                errors.append(error)
        for request_id in request_ids:
            futures[request_id].cancel()
        return tuple(errors), frozenset(cancelled_request_ids)

    @staticmethod
    def _note_cleanup_errors(
        primary: BaseException,
        errors: tuple[BaseException, ...],
    ) -> None:
        for error in errors:
            primary.add_note(
                "arrival coordinator cleanup failed: "
                + type(error).__name__
                + ": "
                + str(error)
            )

    def _raise_after_abort(
        self,
        primary: BaseException,
        primary_future: (
            concurrent.futures.Future[PhysicalAdapterResult] | None
        ),
        request_ids: tuple[str, ...],
        futures: Mapping[
            str, concurrent.futures.Future[PhysicalAdapterResult]
        ],
        deadline: float | None,
    ) -> NoReturn:
        cleanup, cancelled_request_ids = self._abort_outstanding(
            request_ids, futures
        )
        cleanup_errors = list(cleanup)
        for request_id in request_ids:
            future = futures[request_id]
            if future is primary_future or future.cancelled():
                continue
            remaining = (
                None
                if deadline is None
                else max(0.0, deadline - time.monotonic())
            )
            try:
                future.result(timeout=remaining)
            except concurrent.futures.CancelledError:
                pass
            except BaseException as error:
                if request_id not in cancelled_request_ids:
                    cleanup_errors.append(error)
        self._note_cleanup_errors(primary, tuple(cleanup_errors))
        raise primary

    def drain(self, timeout_s: float | None = None) -> CanonicalRuntimeTraceResult:
        """Wait for every submitted physical lifecycle to become terminal."""
        if timeout_s is not None and timeout_s <= 0:
            raise PhysicalAdapterError("runtime drain timeout is invalid")
        with self._lock:
            request_ids = tuple(self._request_ids)
            tickets = dict(self._tickets)
            futures = dict(self._futures)
        deadline = (
            None if timeout_s is None else time.monotonic() + timeout_s
        )
        remaining = (
            None if deadline is None else max(0.0, deadline - time.monotonic())
        )
        done, pending = concurrent.futures.wait(
            tuple(futures.values()),
            timeout=remaining,
            return_when=concurrent.futures.FIRST_EXCEPTION,
        )
        primary = None
        primary_future = None
        for request_id in request_ids:
            future = futures[request_id]
            if future not in done:
                continue
            if future.cancelled():
                primary = concurrent.futures.CancelledError()
            else:
                primary = future.exception()
            if primary is not None:
                primary_future = future
                break
        if primary is None and pending:
            primary = concurrent.futures.TimeoutError()
        if primary is not None:
            self._raise_after_abort(
                primary, primary_future, request_ids, futures, deadline
            )
        executions = {
            request_id: futures[request_id].result()
            for request_id in request_ids
        }
        tickets = {
            request_id: self._scheduler.runtime_ticket(request_id)
            for request_id in request_ids
        }
        return CanonicalRuntimeTraceResult(
            request_ids=request_ids,
            tickets=tickets,
            executions=executions,
        )

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            request_ids = tuple(self._request_ids)
            futures = dict(self._futures)
            threads = tuple(self._threads.values())
        errors = ()
        if not wait:
            errors, _ = self._abort_outstanding(request_ids, futures)
        for thread in threads:
            thread.join()
        self._pool.shutdown(wait=True, cancel_futures=not wait)
        if errors:
            raise PhysicalAdapterError(
                "runtime arrival cleanup failed: "
                + "; ".join(
                    type(error).__name__ + ": " + str(error)
                    for error in errors
                )
            )

    def __enter__(self) -> "CanonicalArrivalCoordinator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(wait=exc_type is None)
