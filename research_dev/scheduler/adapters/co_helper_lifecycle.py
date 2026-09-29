"""Trace-lifetime ownership of static FFN co-helpers, separate from primary phone sessions."""

from __future__ import annotations

from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Protocol, Sequence

from .._internal.capability_contracts.snapshots import ModelResidencyObservation
from .._internal.model_manifest import ModelManifest
from .._internal.plan_contracts.co_helpers import RuntimeCoHelperDeclaration
from .contracts import PhysicalAdapterError


class CoHelperStopPolicy(Protocol):
    """Ends one trace-length co-helper worker and returns its stop receipt (JSON).

    A policy with ``releases_exited_workers = True`` also accepts ``release_exited=True`` (elastic
    phones' trace end): a worker that already exited is then released, not refused."""

    name: str

    def stop(self, session: object, *, served_calls: int | None) -> Mapping[str, object]: ...


class IdleCoHelperStopPolicy:
    """Stop an owned resident worker only after its server clients have disconnected."""

    name = "owned-idle-sigterm"
    # Elastic phones: ``end_trace(continue_past_failures=True)`` asks this policy to release a worker
    # that already exited (lost mid-call) instead of refusing it as busy.
    releases_exited_workers = True

    def stop(self, session, *, served_calls=None, release_exited=False):
        options = {"release_exited": True} if release_exited else {}
        return session.stop(served_calls=served_calls, allow_idle_signal=True, **options).to_json()


class CoHelperLifecycle:
    """Start, residency, proof routing and stop of every co-helper of one declaration."""

    def __init__(
        self,
        declaration: RuntimeCoHelperDeclaration,
        sessions: Mapping[str, object],
        stop_policy: CoHelperStopPolicy | None = None,
        identity_checks: Mapping[str, Callable[[], str]] | None = None,
    ) -> None:
        if not isinstance(declaration, RuntimeCoHelperDeclaration):
            raise PhysicalAdapterError("co-helper lifecycle declaration is invalid")
        if set(sessions) != set(declaration.device_ids):
            raise PhysicalAdapterError("co-helper lifecycle needs exactly one session per co-helper")
        for row in declaration.helpers:
            configuration = getattr(sessions[row.device_id], "configuration", None)
            if (
                configuration is None
                or configuration.device_id != row.device_id
                or configuration.serial != row.serial
                or configuration.layer_mask != row.layer_mask
                or configuration.column_quantum != row.column_quantum
                or configuration.max_tokens != row.max_tokens
                or configuration.phone_port != row.transport_parameters["phone_worker_port"]
                or configuration.forward_port != row.transport_parameters["ffn_worker_port"]
            ):
                raise PhysicalAdapterError(
                    f"co-helper {row.device_id} worker differs from its catalog declaration"
                )
        checks = dict(identity_checks or {})
        if set(checks) - set(declaration.device_ids) or any(not callable(row) for row in checks.values()):
            raise PhysicalAdapterError("co-helper identity checks are invalid")
        self.declaration = declaration
        self.sessions = dict(sessions)
        self.stop_policy = stop_policy
        # device -> check of its pinned USB/kernel identity; returns the qualified identity sha256
        self.identity_checks = checks
        self.started: list[str] = []
        # Elastic phones: declared co-helpers that are not running (absent at start, lost and
        # released), and lost workers whose release runs off the caller's thread.
        self.absent: set[str] = set()
        self._stopping: dict[str, threading.Thread] = {}
        self._background_receipts: list[Mapping[str, object]] = []
        self._lock = threading.RLock()

    def _helper(self, device_id: str):
        row = next((row for row in self.declaration.helpers if row.device_id == device_id), None)
        if row is None:
            raise PhysicalAdapterError("device is not a co-helper of this declaration")
        return row

    def _start(self, row, log_path: Path, *, elastic: bool = False) -> list[Mapping[str, object]]:
        session = self.sessions[row.device_id]
        receipts = self._elastic_preflight(session) if elastic else [session.preflight().to_json()]
        try:
            if elastic:
                # elastic phones: the launch time anchors the fault-injection tool's ``t=`` condition
                receipts.append({**session.start(log_path, release_failed_start=True).to_json(),
                                 "monotonic_ns": time.monotonic_ns()})
            else:
                receipts.append(session.start(log_path).to_json())
        finally:
            if getattr(session, "active", True):
                with self._lock:
                    self.started.append(row.device_id)
        live = dict(session.transport_parameters())
        if any(live.get(key) != value for key, value in row.transport_parameters.items()
               if key in live):
            raise PhysicalAdapterError(
                f"co-helper {row.device_id} forward differs from its catalog declaration"
            )
        return receipts

    @staticmethod
    def _elastic_preflight(session) -> list[Mapping[str, object]]:
        """Elastic phones: a preflight refused by a worker this session launched earlier (a failed
        start or a lost worker whose release could not reach the phone, same boot and pids) releases
        that worker (SIGTERM, wait) and preflights once more; any other refusal stands."""
        try:
            return [session.preflight().to_json()]
        except PhysicalAdapterError:
            release = getattr(session, "release_orphaned_worker", None)
            orphan = release() if callable(release) else None
            if orphan is None:
                raise
        return [orphan.to_json(), session.preflight().to_json()]

    @staticmethod
    def _event(kind: str, device_id: str, **details: object) -> dict[str, object]:
        return {"device_id": device_id, "kind": kind, "monotonic_ns": time.monotonic_ns(), **details}

    def start_trace(
        self, log_directory: Path, *, tolerate_absent: bool = False, elastic: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        """Start every co-helper. With ``tolerate_absent`` (elastic phones with join) a helper that
        cannot start is recorded as absent (``DEVICE_ABSENT_AT_START``) instead of ending the trace;
        a forward that differs from the declaration still fails. ``elastic`` (any elastic-phones run;
        implied by ``tolerate_absent``) releases the worker of a failed start instead of leaving it
        to the trace's stop policy."""
        elastic = elastic or tolerate_absent
        if self.stop_policy is None:
            raise PhysicalAdapterError(
                "co-helper phones cannot start without a stop policy (worker shutdown message "
                "or opt-in idle SIGTERM: user decision)"
            )
        if self.started or self.absent:
            raise PhysicalAdapterError("co-helper phones are already started")
        receipts = []
        for row in self.declaration.helpers:
            log_path = log_directory / (row.session_id + "-worker.log")
            if not tolerate_absent:
                receipts.extend(self._start(row, log_path, elastic=elastic))
                continue
            try:
                receipts.extend(self._start(row, log_path, elastic=True))
            except Exception as error:  # fail closed: whatever failed, the helper is absent
                if row.device_id in self.started:
                    raise
                self.absent.add(row.device_id)
                receipts.append(self._event("DEVICE_ABSENT_AT_START", row.device_id,
                                            reason=type(error).__name__ + ": " + str(error)))
        return tuple(receipts)

    def join(self, device_id: str, log_path: Path) -> tuple[Mapping[str, object], ...]:
        """Start an absent co-helper again: its pinned USB/kernel identity, then the pinned-hash
        preflight and the launch. On failure a worker this call started is released and the device
        stays absent. The last receipt (``JOINED``) carries the verified identity."""
        row = self._helper(device_id)
        check = self.identity_checks.get(device_id)
        if check is None:
            raise PhysicalAdapterError(f"co-helper {device_id} has no pinned identity to join with")
        with self._lock:
            if device_id not in self.absent or device_id in self._stopping:
                raise PhysicalAdapterError(f"co-helper {device_id} is not absent")
            self.absent.discard(device_id)
        try:
            identity = check()
            if (type(identity) is not str or not identity.startswith("sha256:") or len(identity) != 71):
                raise PhysicalAdapterError(f"co-helper {device_id} identity check returned no identity")
            receipts = self._start(row, log_path, elastic=True)
        except BaseException:
            with self._lock:
                started = device_id in self.started
                if started:
                    self.started.remove(device_id)
            try:
                if started:
                    self._release(device_id)
            finally:
                with self._lock:
                    self.absent.add(device_id)
            raise
        return (*receipts, self._event("JOINED", device_id, identity_sha256=identity))

    def _release(self, device_id: str) -> Mapping[str, object]:
        session = self.sessions[device_id]
        release = getattr(session, "release_lost", None)
        if release is None:
            raise PhysicalAdapterError(f"co-helper {device_id} session cannot release a lost worker")
        return release().to_json()

    def lose(self, device_id: str) -> threading.Thread | None:
        """A started co-helper stopped serving: it leaves ``started`` now (no residency) and its
        worker is released on a background thread (SIGTERM only; never blocks the caller)."""
        self._helper(device_id)
        with self._lock:
            if device_id not in self.started:
                return None
            self.started.remove(device_id)
            thread = threading.Thread(target=self._background_release, args=(device_id,),
                                      name="co-helper-release-" + device_id, daemon=True)
            self._stopping[device_id] = thread
            thread.start()
            return thread

    def _background_release(self, device_id: str) -> None:
        try:
            row = dict(self._release(device_id))
        except BaseException as error:  # recorded: a release never raises on its own thread
            row = self._event("RELEASE_FAILED", device_id, reason=type(error).__name__ + ": " + str(error))
        with self._lock:
            self._background_receipts.append(row)
            self.absent.add(device_id)
            self._stopping.pop(device_id, None)

    def wait_released(self, timeout_s: float | None = None) -> tuple[Mapping[str, object], ...]:
        """Wait for background releases; returns (and clears) their receipts."""
        with self._lock:
            threads = tuple(self._stopping.values())
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        for thread in threads:
            thread.join(None if deadline is None else max(0.0, deadline - time.monotonic()))
        with self._lock:
            if any(thread.is_alive() for thread in threads):
                raise PhysicalAdapterError("co-helper release is still running")
            receipts, self._background_receipts = tuple(self._background_receipts), []
            return receipts

    def residency_observations(self, manifest: ModelManifest) -> tuple[ModelResidencyObservation, ...]:
        """``hot`` rows for the started co-helpers of this model."""
        return tuple(
            ModelResidencyObservation(
                model_id=manifest.model_id,
                artifact_sha256=manifest.artifact_sha256,
                device_id=row.device_id,
                state="hot",
                resident_tensor_ids=tuple(sorted(
                    f"blk.{index}.ffn_{kind}.weight"
                    for index in row.layer_indices
                    for kind in ("down", "gate", "up")
                )),
                resident_bytes=row.resident_bytes,
                generation=1,
            )
            for row in self.declaration.helpers
            if row.device_id in self.started
        )

    def split_session_proofs(
        self, proofs: Sequence[object]
    ) -> tuple[tuple[object, ...], Mapping[str, tuple[object, ...]]]:
        """Primary-phone session proofs and, per co-helper device, its own proof rows."""
        by_session = {row.session_id: row.device_id for row in self.declaration.helpers}
        primary = tuple(row for row in proofs if row.session_id not in by_session)
        return primary, {
            device_id: tuple(row for row in proofs if by_session.get(row.session_id) == device_id)
            for device_id in self.declaration.device_ids
        }

    def end_trace(
        self,
        served_calls_by_device: Mapping[str, int | None],
        *,
        continue_past_failures: bool = False,
    ) -> tuple[Mapping[str, object], ...]:
        """Stop every started co-helper. With ``continue_past_failures`` (elastic phones) background
        releases finish first and every helper is attempted; a worker that already exited (lost
        mid-call, its loss not yet released) is released by a policy that supports it, so its
        forward is removed; the failures are raised together, carrying the successful receipts as
        ``error.receipts``."""
        if self.stop_policy is None:
            raise PhysicalAdapterError("co-helper phones have no stop policy")
        receipts, errors = [], []
        options = (
            {"release_exited": True}
            if continue_past_failures
            and getattr(self.stop_policy, "releases_exited_workers", False) is True
            else {}
        )
        if continue_past_failures:
            try:
                receipts.extend(self.wait_released(timeout_s=300))
            except PhysicalAdapterError as error:
                errors.append(str(error))
        for device_id in tuple(self.started):
            try:
                receipts.append(dict(self.stop_policy.stop(
                    self.sessions[device_id],
                    served_calls=served_calls_by_device.get(device_id),
                    **options,
                )))
            except BaseException as error:
                if not continue_past_failures:
                    raise
                errors.append(device_id + ": " + type(error).__name__ + ": " + str(error))
                continue
            self.started.remove(device_id)
        if errors:
            failure = PhysicalAdapterError("co-helper stop failed: " + "; ".join(errors))
            failure.receipts = tuple(receipts)
            raise failure
        return tuple(receipts)
