"""Physical rig: executor samples, protected-work and allocation observations, execution success accounting, backend view."""

from __future__ import annotations

from dataclasses import replace
import http.client
import json
import math
import re
import time
from typing import Mapping, Sequence
from ..._internal.adaptive_decode_contracts import (
    AdaptiveDecodeError,
    AdaptiveDecodeGroupedObservation,
)
from ..._internal.capability_contracts.executors import whole_phone_launch_parameters
from ..._internal.types import canonical_sha256
from ..._internal.runtime_execution import (
    HELPER_LOST_PHASE,
    SERVER_EXITED_PHASE,
)
from ..._internal.runtime_plan import RuntimeHelperExecutionEnvelope
from ..._internal.runtime_system_cost import RuntimeProtectedWorkObservation
from ..android_llama_server import ManagedAndroidLlamaServer
from ..contracts import (
    CompletionStreamError,
    LlamaServerExitedError,
    PhysicalAdapterError,
    PhysicalFailureClassification,
    PhysicalHelperLostError,
    RawEnergyMeasurement,
    StalePhysicalSlotError,
    elastic_drop_recovery_enabled,
)
from ..energy import RaplNvmlPhoneEnergyMeter
from ..http_backend import CanonicalHttpExecutionBackend, HttpEndpointFailure
from ..snapshot import EndpointRuntimeSample
from ..ticket import PhysicalExecutionCommand
from ..host_runtime import server_energy_summary
from .common import _LiveExecutorResidency as _LiveExecutorResidency


# ---- elastic phones: drop classification (SPEC 20260925-elastic-phones, slice 1) ----------------
# ``S41SERVERFFNERROR [helper=<label> ]detail=...`` (tools/server/server.cpp): printed per failed
# helper client; the label is omitted when the server has a single helper (the primary phone).
_FFN_ERROR_LINE = re.compile(
    r"S41SERVERFFNERROR (?:helper=([A-Za-z0-9_-]{1,32}) )?detail="
)
# Several helpers prefix every client error with ``helper <label>: `` (policy apply, context,
# connect); the prefix reaches both the stderr log and the SSE error of each processing slot.
_HELPER_ERROR = re.compile(r"(?:^|[^A-Za-z0-9_-])helper ([A-Za-z0-9_-]{1,32}): ")
_LIBUSB_ERROR = "LIBUSB_ERROR"
# llama_decode returned 2 (the FFN client aborted the graph): helper context only when the
# server has exactly one helper, otherwise the helper must be named by other evidence.
_COMPUTE_ABORTED = "Compute aborted"
# ``update_slots`` caught a failed (pre_/post_)decode, errored every processing slot and keeps
# serving (tools/server/server-context.cpp): no exit follows, so no later shutdown line either.
_SLOTS_ABORTED = "decode() failed: "
_EVIDENCE_LIMIT = 512
# ``S41SERVERFFNRESET`` (S2a server): a helper session the server closed (after a loss it already
# reported) or its shutdown summary; it repeats no fresh evidence, so it never names a lost helper.
_HELPER_RESET_PREFIX = "S41SERVERFFNRESET"
# Seconds between two liveness checks of the helpers of an unlabeled abort.
_LIVENESS_RECHECK_S = 0.25


def _server_aborted_slots(lines: Sequence[str]) -> bool:
    return any(_SLOTS_ABORTED in line for line in lines)


# One helper of a server: (label or None for the single unlabeled helper, device id when the rig
# knows the label, llama-server transport name).
ServerHelperRow = tuple[str | None, str | None, str | None]


def _ascii_evidence(value: str) -> str:
    text = value.encode("ascii", errors="backslashreplace").decode("ascii")
    return " ".join(text.split())[:_EVIDENCE_LIMIT]


def device_id_for_helper_label(rig: object, label: str | None) -> str | None:
    """The rig device id of one llama-server helper label (None: unknown).

    Labels come from the co-helper declaration (``RuntimeCoHelperPhone.label``; the declaration's
    ``primary_label`` names the primary phone); a single-helper server prints no label and names
    the primary phone (``configuration.phone_device_id``).
    """
    primary = getattr(getattr(rig, "configuration", None), "phone_device_id", None)
    if label is None:
        return primary if type(primary) is str and primary else None
    for lifecycle in getattr(rig, "_co_helper_lifecycles", {}).values():
        declaration = lifecycle.declaration
        if label == declaration.primary_label:
            return primary if type(primary) is str and primary else None
        for row in declaration.helpers:
            if row.label == label:
                return row.device_id
    return None


def server_helper_rows(rig: object, environment: Mapping[str, str]) -> tuple[ServerHelperRow, ...]:
    """The phone helpers one llama-server launch environment connects (empty: no FFN split)."""
    count = environment.get("S41_SERVER_FFN_HELPERS")
    if count is None:
        transport = environment.get("S41_SERVER_FFN_TRANSPORT")
        if transport is None:
            return ()
        return ((None, device_id_for_helper_label(rig, None), transport),)
    if type(count) is not str or not count.isdigit() or int(count) < 1:
        raise PhysicalAdapterError("llama-server FFN helper count is invalid")
    rows = []
    for index in range(int(count)):
        label = environment.get(f"S41_SERVER_FFN_HELPER{index}_LABEL")
        if type(label) is not str or not label:
            raise PhysicalAdapterError("llama-server FFN helper label is absent")
        rows.append((
            label,
            device_id_for_helper_label(rig, label),
            environment.get(f"S41_SERVER_FFN_HELPER{index}_TRANSPORT"),
        ))
    if len({row[0] for row in rows}) != len(rows):
        raise PhysicalAdapterError("llama-server FFN helper labels repeat")
    return tuple(rows)


def classify_helper_failure(
    lines: Sequence[str],
    stream_error_message: str | None,
    helpers: Sequence[ServerHelperRow],
) -> tuple[tuple[str, ...], str]:
    """Lost helper devices named by server stderr / SSE error text, with the first evidence line.

    Evidence naming an unknown label is ignored (fail closed): the failure then keeps today's
    classification. ``Compute aborted`` names the helper only for a single-helper server.
    """
    rows = tuple(helpers)
    if not rows:
        return (), ""
    by_label = {label: device for label, device, _ in rows}
    single = len(rows) == 1 and rows[0][0] is None
    found: dict[str, str] = {}

    def claim(label: str | None, line: str) -> None:
        if label is None and not single:
            return
        device = by_label.get(label)
        if device is not None:
            found.setdefault(device, line)

    texts = [line for line in lines if type(line) is str and not line.startswith(_HELPER_RESET_PREFIX)]
    if type(stream_error_message) is str and stream_error_message:
        texts.append(stream_error_message)
    for line in texts:
        match = _FFN_ERROR_LINE.search(line)
        if match is not None:
            claim(match.group(1), line)
            continue
        for match in _HELPER_ERROR.finditer(line):
            claim(match.group(1), line)
        if _LIBUSB_ERROR in line:
            for _label, device, transport in rows:
                if transport == "functionfs-usb" and device is not None:
                    found.setdefault(device, line)
    if not found and single:
        aborted = next((line for line in texts if _COMPUTE_ABORTED in line), None)
        if aborted is not None:
            claim(None, aborted)
    devices = tuple(sorted(found))
    return devices, ("" if not devices else _ascii_evidence(found[devices[0]]))


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    seen: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and all(current is not row for row in seen) and len(seen) < 16:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return tuple(seen)


def _server_side_failure(chain: Sequence[BaseException]) -> bool:
    """Whether a failed request looks like the server or its connection died (worth waiting).

    A dying server cuts the stream: an SSE error chunk, a closed or reset connection, or a
    truncated last line (JSON / UTF-8 decode error) before the final chunk. The adaptive-decode
    boundary sees the same failure first through a server control call of the request (FFN
    policy, context or window stats): the server answered it with an error, or with the
    request's slot already released (``StalePhysicalSlotError``) because it errored the request.
    """
    return any(
        isinstance(row, (
            CompletionStreamError, HttpEndpointFailure, OSError, http.client.HTTPException,
            json.JSONDecodeError, UnicodeDecodeError, StalePhysicalSlotError,
        ))
        or (isinstance(row, PhysicalAdapterError) and str(row) == "completion final chunk is absent")
        or _server_control_message(row) is not None
        for row in chain
    )


def _server_control_message(error: BaseException) -> str | None:
    """The server's own text of a failed server control call (``http_backend``), else None."""
    message = getattr(error, "server_control_message", None)
    return message if isinstance(error, PhysicalAdapterError) and type(message) is str else None


class RigObservationMixin:
    """Physical rig: executor samples, protected-work and allocation observations, execution success accounting, backend view."""

    def _helper_phone_power(self):
        return {device: (self.configuration.catalog.phone_power_profile_by_device[device], activity)
                for device, activity in getattr(self, "_co_helper_activity", {}).items()}

    def _begin_measurement_epoch(self, epoch_ns: int) -> None:
        if type(epoch_ns) is not int or epoch_ns <= 0:
            raise PhysicalAdapterError(
                "physical measurement epoch is invalid"
            )
        with self._lock:
            if self._transition_active:
                raise PhysicalAdapterError(
                    "physical measurement epoch changed during transition"
                )
            self._execution_backend = None
        self.epoch_ns = epoch_ns

    def trace_energy(
        self, start_ns: int, end_ns: int
    ) -> RawEnergyMeasurement:
        return RaplNvmlPhoneEnergyMeter(
            self._sampler.latest_rows,
            server_energy_summary,
            self._phone_sampler,
            energy_boundary_id=(
                self.configuration.catalog.placement_profile.energy_boundary_id
            ),
            attribution_kind=self.configuration.energy_attribution_kind,
            server_window_rows=self._sampler.rows_between,
            phone_power_profile=self._phone_power_profile,
            phone_activity=(
                self._phone_activity
                if self._phone_power_profile is not None else None
            ),
            helper_phone_power=self._helper_phone_power(),
        ).measure(start_ns, end_ns)

    def _execution_success(
        self,
        command: PhysicalExecutionCommand,
        payload: Mapping[str, object],
    ) -> Mapping[str, object]:
        with self._lock:
            execution = self._execution_markers.get(command.ticket_id)
        if execution is None:
            raise PhysicalAdapterError(
                "physical execution proof marker is absent"
            )
        server, manifest, marker, shape = execution
        adaptive = None
        raw_adaptive = payload.get("adaptive_decode_observation")
        if raw_adaptive is not None:
            try:
                adaptive = AdaptiveDecodeGroupedObservation.from_json(
                    raw_adaptive
                )
            except (TypeError, AdaptiveDecodeError) as error:
                raise PhysicalAdapterError(
                    "adaptive execution observation is invalid"
                ) from error
        if command.helper_envelope is not None and marker.phone_contract is None:
            marker = server.bind_ready_helper(marker, command, manifest)
        helper_envelopes = payload.get("_runtime_helper_envelopes", ())
        if (
            not isinstance(helper_envelopes, tuple)
            or any(
                not isinstance(row, RuntimeHelperExecutionEnvelope)
                for row in helper_envelopes
            )
        ):
            raise PhysicalAdapterError(
                "physical helper history is invalid"
            )
        execution_proof = server.finish_execution(
            marker,
            command,
            manifest,
            output_tokens=shape[1],
            adaptive_observation=adaptive,
            static_control_ack=payload.get("static_ffn_control_ack"),
            helper_envelopes=helper_envelopes,
        )
        if execution_proof.phone_calls_by_session:
            proofs = execution_proof.phone_calls_by_session
            lifecycle = getattr(self, "_co_helper_lifecycles", {}).get(command.artifact_sha256)
            if lifecycle is not None:
                proofs, helper_proofs = lifecycle.split_session_proofs(proofs)
                self._co_helper_receipts.append({"phase": "execution", "ticket_id": command.ticket_id,
                    "proofs_by_device": {device: [row.to_json() for row in rows]
                                         for device, rows in helper_proofs.items()}})
            direct_phone = self._direct_phone_session
            if not direct_phone.active:
                raise PhysicalAdapterError(
                    "phone execution proof has no active phone session"
                )
            direct_phone.record_execution_proof(
                command.ticket_id,
                command.artifact_sha256,
                proofs,
            )
        proof = execution_proof.to_json()
        with self._lock:
            if command.ticket_id in self._execution_proofs:
                raise PhysicalAdapterError(
                    "physical execution proof is duplicated"
                )
            self._execution_proofs[command.ticket_id] = proof
        return proof

    def backend(self) -> CanonicalHttpExecutionBackend:
        with self._lock:
            if self._execution_backend is None:
                energy_meter = RaplNvmlPhoneEnergyMeter(
                    self._sampler.latest_rows,
                    server_energy_summary,
                    self._phone_sampler,
                    energy_boundary_id=(
                        self.configuration.catalog.placement_profile
                            .energy_boundary_id
                    ),
                    attribution_kind=(
                        self.configuration.energy_attribution_kind
                    ),
                    server_window_rows=self._sampler.rows_between,
                    phone_power_profile=self._phone_power_profile,
                    phone_activity=self._phone_activity,
                    helper_phone_power=self._helper_phone_power(),
                )
                optional = (
                    {"failure_classifier": self._classify_execution_failure}
                    if self._elastic_drop_recovery() else {}
                )
                if self._helper_loss_recovery() == "mask_out":
                    # S2a: no FFN control may give a masked-out helper a layer on its server
                    optional["control_guard"] = self._helper_mask_guard
                self._execution_backend = CanonicalHttpExecutionBackend(
                    self._client,
                    energy_meter,
                    epoch_ns=self.epoch_ns,
                    prepare_transition=self._transition_registry.execute,
                    rollback_transition=self.rollback_helper_transition,
                    on_execution_start=self._execution_start,
                    on_execution_success=self._execution_success,
                    on_execution_finish=self._execution_finish,
                    on_scheduler_bound=self._bind_scheduler,
                    on_control_ack=self._dormant_control_ack,
                    before_prompt=self._dormant_before_prompt,
                    **optional,
                )
            return self._execution_backend

    # ---- elastic phones: drop recovery (slice 1) --------------------------------------------------
    # Seconds a failed request waits for its server to exit so the shutdown lines are read.
    _failure_evidence_timeout_s: float = 10.0

    def _elastic_drop_recovery(self) -> bool:
        """``elastic_phones.drop_recovery`` of this rig; absent keeps today's behaviour.

        Read from ``configuration.elastic_phones`` (HeterogeneousRigConfiguration, slice 2) or a
        rig attribute ``_elastic_phones``; tests build rigs without ``__init__``.
        """
        elastic = getattr(getattr(self, "configuration", None), "elastic_phones", None)
        if elastic is None:
            elastic = getattr(self, "_elastic_phones", None)
        return elastic_drop_recovery_enabled(elastic)

    def _server_helper_rows(self, server: object) -> tuple[ServerHelperRow, ...]:
        environment = getattr(server, "environment", None)
        return server_helper_rows(self, environment if isinstance(environment, Mapping) else {})

    def _dead_helper_devices(
        self,
        helpers: Sequence[ServerHelperRow],
        deadline: float | None = None,
    ) -> tuple[str, ...]:
        """Co-helpers of one server whose phone session reports itself dead (slice 2 ``alive()``).

        With a ``deadline`` (``time.monotonic()``) the check repeats until one helper is dead or
        the deadline passes: the session may report the loss a moment after the server did. A
        server whose helpers have no liveness source is answered at once.
        """
        sessions = getattr(self, "_co_helper_sessions", {})
        while True:
            dead, checked = [], False
            for _label, device, _transport in helpers:
                alive = getattr(sessions.get(device), "alive", None)
                if device is None or not callable(alive):
                    continue
                checked = True
                try:
                    if alive() is False:
                        dead.append(device)
                except Exception:
                    continue
            if dead or not checked or deadline is None or time.monotonic() >= deadline:
                return tuple(sorted(dead))
            time.sleep(min(_LIVENESS_RECHECK_S, max(0.0, deadline - time.monotonic())))

    def _retire_for_lost_helpers(
        self,
        command: PhysicalExecutionCommand,
        server: object,
        helpers: Sequence[ServerHelperRow],
        devices: tuple[str, ...],
        started_ns: int | None = None,
    ) -> tuple[int | None, bool]:
        """Retire the live server whose own FFN helper was lost: (exit code, masked out).

        ``helper_loss_recovery: mask_out`` (S2a) keeps a desktop-parent server that can mask the
        lost helpers out (``mask_out_lost_helpers``): (None, True). Otherwise, or when the mask-out
        is refused, the server is retired as before; (None, False) keeps a server that does not
        own any lost device.
        """
        own = tuple(sorted(set(devices) & {device for _label, device, _transport in helpers if device}))
        if not own:
            return None, False
        if self._helper_loss_recovery() == "mask_out" and self.mask_out_lost_helpers(
            command, expected_server=server, device_ids=own, attempt_started_ns=started_ns,
        ) is None:
            return None, True
        return self.retire_helper_lost_executor(
            command.executor_id, expected_server=server, device_ids=devices
        ), False

    def _classify_execution_failure(
        self,
        command: PhysicalExecutionCommand,
        error: BaseException,
        *,
        started_ns: int | None = None,
    ) -> PhysicalFailureClassification | None:
        """Read one failed execution as helper_lost / server_exited (None: today's failure).

        A dead server is reaped here, before the scheduler's recovery snapshot, so the recovery
        plans a load instead of routing to the dead endpoint; a live server that lost one of its
        own FFN helpers is retired (stopped, then reaped) for the same reason: llama-server
        survives the loss with the helper's client latched failed. Server stderr is shared by every
        slot of the server, so it is read as this request's evidence only when the failure itself
        is server-side (a cut stream, a closed connection) or the server has exited: a local
        failure (lease or control expiry, a bug) of a live server is never another slot's helper
        loss. The wait for the exit ends early once the server logged that it errored its slots
        (it keeps running then). ``started_ns`` (the attempt start) marks losses of a device
        readmitted since as stale.
        """
        chain = _exception_chain(error)
        exited = next((row for row in chain if isinstance(row, LlamaServerExitedError)), None)
        if exited is not None:
            executor_id = exited.executor_id or command.executor_id
            if executor_id == getattr(self.configuration, "resident_executor_id", None):
                return None
            reaped = self.reap_exited_executor(executor_id)
            self._await_retire(executor_id)
            return PhysicalFailureClassification(
                SERVER_EXITED_PHASE,
                executor_id=executor_id,
                returncode=exited.returncode if reaped is None else reaped["returncode"],
                evidence="llama-server exited before the request started",
            )
        lost = {row.device_id for row in chain if isinstance(row, PhysicalHelperLostError)}
        with self._lock:
            execution = self._execution_markers.get(command.ticket_id)
        server = None if execution is None else execution[0]
        reader = getattr(server, "failure_evidence", None)
        if not callable(reader):
            return None if not lost else self._helper_lost_classification(
                command, tuple(sorted(lost)), started_ns,
                evidence="phone session transport lost",
            )
        marker = execution[2]
        server_side = _server_side_failure(chain)
        timeout_s = self._failure_evidence_timeout_s if server_side else 0.0
        deadline = time.monotonic() + timeout_s
        lines, returncode = reader(
            marker.stderr_index,
            timeout_s=timeout_s,
            decisive=_server_aborted_slots,
        )
        if not server_side and returncode is None:
            if not lost:
                return None
            devices = tuple(sorted(lost))
            returncode, masked = self._retire_for_lost_helpers(
                command, server, self._server_helper_rows(server), devices, started_ns
            )
            return self._helper_lost_classification(
                command, devices, started_ns,
                executor_id=command.executor_id if returncode is not None else None,
                returncode=returncode,
                evidence="phone session transport lost",
                **({"masked_executor_id": command.executor_id} if masked else {}),
            )
        stream_message = next((
            row.server_error_message for row in chain
            if isinstance(row, CompletionStreamError) and row.server_error_message
        ), None)
        # a failed control call's server text ("helper <label>: ...") is this request's evidence
        lines = (*lines, *(
            message for message in map(_server_control_message, chain) if message
        ))
        helpers = self._server_helper_rows(server)
        devices, evidence = classify_helper_failure(lines, stream_message, helpers)
        if not devices and not lost and helpers and any(
            _COMPUTE_ABORTED in line for line in (*lines, stream_message or "")
        ):
            # Several helpers and an unlabeled abort: name the helper only when its own session
            # reports it dead (liveness from the co-helper session, slice 2), until the deadline
            # of the evidence wait.
            devices = self._dead_helper_devices(helpers, deadline)
            evidence = "" if not devices else "compute aborted; helper session not alive"
        devices = tuple(sorted(set(devices) | lost))
        if returncode is not None:
            # another request may be retiring this server: its classification ends with it
            self._await_retire(command.executor_id, server)
        if not devices and returncode is not None:
            # this rig stopped the server after one of its helpers was lost (a request that
            # started after the loss dies with the server, without helper lines of its own)
            devices = self._retired_lost_devices(server)
            evidence = "" if not devices else "llama-server retired after its helper was lost"
        masked = False
        if returncode is not None:
            self.reap_exited_executor(command.executor_id, expected_server=server)
        elif devices:
            returncode, masked = self._retire_for_lost_helpers(
                command, server, helpers, devices, started_ns
            )
        if devices:
            return self._helper_lost_classification(
                command, devices, started_ns,
                executor_id=command.executor_id if returncode is not None else None,
                returncode=returncode,
                evidence=evidence or "phone session transport lost",
                **({"masked_executor_id": command.executor_id} if masked else {}),
            )
        if returncode is not None and command.executor_id != getattr(
            self.configuration, "resident_executor_id", None
        ):
            # The resident server is not a dynamic endpoint: no transition reloads it.
            return PhysicalFailureClassification(
                SERVER_EXITED_PHASE,
                executor_id=command.executor_id,
                returncode=returncode,
                evidence="llama-server exited during the request",
            )
        return None

    def _helper_lost_classification(
        self,
        command: PhysicalExecutionCommand,
        devices: tuple[str, ...],
        started_ns: int | None,
        **fields: object,
    ) -> PhysicalFailureClassification:
        stale_losses = getattr(self, "_stale_helper_losses", None)
        stale = (
            () if not callable(stale_losses)
            else stale_losses(devices, started_ns, command.ticket_id)
        )
        primary = getattr(getattr(self, "configuration", None), "phone_device_id", None)
        if primary in devices and primary not in stale and self._helper_loss_recovery() == "mask_out":
            # S2a: the primary's FunctionFS worker session is never reused after its loss
            self._require_direct_phone_relaunch(primary, command)
        return PhysicalFailureClassification(
            HELPER_LOST_PHASE,
            failed_device_ids=devices,
            stale_device_ids=stale,
            **fields,
        )

    # Seconds a snapshot waits for the first GPU sample taken after a reaped server exited, and
    # the margin that orders a GPU sample (midpoint of its nvidia-smi call) against a reap.
    _reaped_gpu_sample_timeout_s: float = 3.0
    _reaped_gpu_sample_margin_ns: int = 250_000_000

    def _gpu_memory_sample(self) -> dict[str, object]:
        """The latest GPU sample; right after a reap, one taken after the server exited.

        Drop recovery only. NVML rows arrive periodically (HostEnergySampler), so the snapshot of
        a recovery taken right after a reap may still count the dead server's VRAM and reject the
        reload. Every snapshot until ``wait_until_ns`` of the reap (``_reaped_gpu_sample_timeout_s``
        after the exit: a stuck sampler costs one bounded wait, shared by concurrent recoveries)
        waits for a row sampled after the exit, which is authoritative and retires the reap's
        record; a row sampled while the server still held its allocation (``held_until_ns``, see
        ``_credit_reaped_gpu_memory``) is credited with the server's last NVML bytes; a row in
        between is used as it is (fail closed).
        """
        gpu = self._sampler.latest_gpu()
        with self._lock:
            pending = tuple(getattr(self, "_reaped_gpu_memory", ()))
        if not pending:
            return gpu
        margin = self._reaped_gpu_sample_margin_ns
        settled_ns = max(row["exit_ns"] for row in pending) + margin
        wait_until_ns = max(row["wait_until_ns"] for row in pending)
        while (
            type(gpu.get("sample_t_ns")) is int
            and gpu["sample_t_ns"] < settled_ns
            and time.monotonic_ns() < wait_until_ns
        ):
            time.sleep(0.02)
            gpu = self._sampler.latest_gpu()
        sample_ns = gpu.get("sample_t_ns")
        if type(sample_ns) is not int:
            return gpu
        with self._lock:
            log = self._reaped_gpu_memory_log()
            log[:] = [row for row in log if sample_ns < row["exit_ns"] + margin]
            credit = sum(
                row["bytes"] for row in log if sample_ns + margin < row["held_until_ns"]
            )
        used = int(gpu["memory_used_bytes"])
        credit = min(credit, used)
        if credit <= 0:
            return gpu
        return {
            **gpu,
            "memory_free_bytes": min(
                int(gpu["memory_total_bytes"]), int(gpu["memory_free_bytes"]) + credit
            ),
            "memory_used_bytes": used - credit,
        }

    def _reap_exited_live_executors(self) -> None:
        """Forget every dynamic llama-server that exited while idle (drop recovery only)."""
        with self._lock:
            if getattr(self, "_transition_active", False):
                # The transition owns the endpoint set; the next snapshot reaps.
                return
            live = tuple(self._live_executors.items())
        for executor_id, state in live:
            exit_code = getattr(state.server, "exit_code", None)
            if callable(exit_code) and state.server.process is not None and exit_code() is not None:
                self.reap_exited_executor(executor_id, expected_server=state.server)

    def _executor_samples(self) -> dict[str, EndpointRuntimeSample]:
        catalog = self.configuration.catalog
        if self._elastic_drop_recovery():
            self._reap_exited_live_executors()
        unavailable = EndpointRuntimeSample(
            "unavailable", "unavailable", 0
        )
        result = {}
        transition_executors = {
            row.executor_id for row in catalog.transitions
            if row.executor_id is not None
        }
        for row in catalog.executors:
            if getattr(row, "device_id", None) in getattr(self, "_co_helper_sessions", {}):
                result[row.executor_id] = unavailable
                continue
            observed = self._runtime_monitor.snapshot(
                "endpoint:" + row.executor_id
            )
            sample = (
                observed.value
                if not observed.stale
                and observed.error is None
                and isinstance(observed.value, EndpointRuntimeSample)
                else unavailable
            )
            result[row.executor_id] = (
                sample
                if sample.ready or row.executor_id not in transition_executors
                else replace(sample, transition_available=self._transition_control_available(row))
            )
        with self._lock:
            live = dict(self._live_executors)
            active_large = tuple(self._active_large.values())
            # drop recovery: a server being retired after a helper loss serves nothing more
            retiring = frozenset(getattr(self, "_retiring_executors", ()))
        for row in catalog.composite_executors:
            state = live.get(row.executor_id)
            if state is not None:
                retired = row.executor_id in retiring
                process_alive = (
                    state.server.process is not None
                    and state.server.process.poll() is None
                    and not retired
                )
                parallel = int(state.parameters["parallel"])
                active = sum(
                    command.executor_id == row.executor_id
                    for command in active_large
                )
                result[row.executor_id] = EndpointRuntimeSample(
                    "healthy" if process_alive else "unavailable",
                    "live" if process_alive else "unavailable",
                    max(0, parallel - active),
                    transition_available=not process_alive and not retired,
                )
            elif row.executor_id in transition_executors:
                result[row.executor_id] = EndpointRuntimeSample(
                    "unavailable",
                    "unavailable",
                    0,
                    transition_available=True,
                )
            else:
                result[row.executor_id] = EndpointRuntimeSample(
                    "unavailable", "unavailable", 0
                )
        return result

    @staticmethod
    def _phone_allocation_identity(state: _LiveExecutorResidency) -> dict[str, object]:
        server = state.server
        parameters = whole_phone_launch_parameters(state.parameters)
        if (not isinstance(server, ManagedAndroidLlamaServer) or state.generation <= 0
                or server.artifact_sha256 != state.manifest.artifact_sha256
                or server.endpoint != state.endpoint or server.launch_parameters != parameters):
            raise PhysicalAdapterError("Android allocation launch identity differs")
        return {
            "executor_id": state.executor_id, "endpoint": state.endpoint,
            "artifact_sha256": state.manifest.artifact_sha256, "generation": state.generation,
            "launch_parameters_sha256": canonical_sha256(parameters),
            "process_identity": server.process_identity.to_json(),
        }

    def _protected_work_observation(
        self, active, switching, captured_at_us, *, request_id=None,
    ):
        owned = tuple(row for row in active
                      if request_id is not None and row.request_id == request_id)
        active = tuple(row for row in active if row not in owned)
        scope = {"protected_work_own_ticket_count": len(owned),
                 "protected_work_peer_count": len(active)}
        if owned:
            with self._lock:
                unscoped = set(self._execution_markers) - {
                    row.ticket_id for row in (*owned, *active)}
            if unscoped:
                return RuntimeProtectedWorkObservation(
                    observation_id="PROTECTED_POWER_PEER_SCOPE_UNAVAILABLE",
                    critical_path_end_us=captured_at_us, phase_power_mw=0,
                    stranded_idle_power_mw=0, causal_tail_power_mw=0,
                    sample_count=1, measured=False,
                ), {**scope, "protected_work_unscoped_peer_count": len(unscoped),
                    "protected_power_valid": 0}
        if owned and not active and not switching:
            return None, {**scope, "protected_power_valid": 0}
        ticket_ids = tuple(row.ticket_id for row in active)
        horizon = (None if self._scheduler is None
                   else self._scheduler.runtime_protected_work_end_us(ticket_ids))
        features = {}
        power = 0
        sample_count = 1
        reason = "PROTECTED_POWER_HORIZON_UNAVAILABLE"
        try:
            if horizon is None or horizon <= captured_at_us:
                raise PhysicalAdapterError(reason)
            if switching:
                raise PhysicalAdapterError("PROTECTED_POWER_PHASE_CHANGING")
            with self._lock:
                starts = [self._active_large_since_ns.get(identity) for identity in ticket_ids]
            if not starts or any(value is None for value in starts):
                raise PhysicalAdapterError("PROTECTED_POWER_EXECUTION_START_UNAVAILABLE")
            now_ns = self.epoch_ns + captured_at_us * 1000
            rows = tuple(row for row in self._sampler.latest_rows()
                         if isinstance(row.get("gpu"), dict)
                         and isinstance(row.get("rapl_package"), dict))
            rows = tuple(row for row in rows if all(
                type(row[source].get("sample_t_ns")) is int
                and max(starts) <= row[source]["sample_t_ns"] <= now_ns
                for source in ("gpu", "rapl_package")))
            if len(rows) < 2:
                raise PhysicalAdapterError("PROTECTED_POWER_SAMPLES_UNAVAILABLE")
            for row in rows:
                if row["rapl_package"].get("name") != "package-0":
                    raise PhysicalAdapterError("PROTECTED_POWER_PACKAGE_IDENTITY_DIFFERS")
            start_ns = max(min(row[source]["sample_t_ns"] for row in rows)
                           for source in ("gpu", "rapl_package"))
            end_ns = min(max(row[source]["sample_t_ns"] for row in rows)
                         for source in ("gpu", "rapl_package"))
            if end_ns > now_ns or now_ns - end_ns > 2_500_000_000:
                raise PhysicalAdapterError("PROTECTED_POWER_SAMPLES_STALE")
            context = self._scheduler.runtime_protected_work_power_context(
                ticket_ids, (start_ns - self.epoch_ns) // 1000,
                (end_ns - self.epoch_ns + 999) // 1000)
            if context["reason"] != "PROTECTED_POWER_ISOLATED":
                raise PhysicalAdapterError(str(context["reason"]))
            summary = server_energy_summary(rows, start_ns, end_ns)
            powers = [float(summary[key]) * 1000 for key in (
                "cpu_package_average_power_w", "gpu_board_average_power_w")]
            if any(not math.isfinite(value) or value < 0 for value in powers):
                raise PhysicalAdapterError("PROTECTED_POWER_SAMPLE_MALFORMED")
            cpu_mw, gpu_mw = (math.ceil(value) for value in powers)
            power = cpu_mw + gpu_mw
            sample_count = len(rows)
            features = {
                "protected_cpu_power_mw": cpu_mw,
                "protected_gpu_power_mw": gpu_mw,
                "protected_power_sample_start_us": (start_ns - self.epoch_ns) // 1000,
                "protected_power_sample_end_us": (end_ns - self.epoch_ns) // 1000,
                "protected_power_sample_age_us": (now_ns - end_ns) // 1000,
            }
            reason = "RAPL_NVML_PROTECTED_POWER"
        except (PhysicalAdapterError, KeyError, TypeError, ValueError, OverflowError) as error:
            reason = str(error)
        measured = reason == "RAPL_NVML_PROTECTED_POWER"
        return RuntimeProtectedWorkObservation(
            observation_id=reason,
            critical_path_end_us=captured_at_us if horizon is None else horizon,
            phase_power_mw=power,
            # The alternative already pays its own execution/tail energy.
            stranded_idle_power_mw=0,
            causal_tail_power_mw=0,
            sample_count=sample_count,
            measured=measured,
        ), {**scope, **features, "protected_power_valid": int(measured)}
