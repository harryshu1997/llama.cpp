"""Physical rig: bridge start and qualification, executor and session stops, shutdown."""

from __future__ import annotations

import os
import threading
import time
from ..bridge import (
    parse_functionfs_bridge_qualification,
    probe_functionfs_usb_device,
    qualify_functionfs_bridge,
)
from ..contracts import LlamaServerExitedError, PhysicalAdapterError
from ..phone_session import DirectPhoneFfnSession
from ..ticket import PhysicalTransitionCommand
from ..host_runtime import CapturedProcess


class RigLifecycleMixin:
    """Physical rig: bridge start and qualification, executor and session stops, shutdown."""

    def _finish_direct_phone(
        self,
        direct_phone: DirectPhoneFfnSession,
        *,
        require_execution: bool = True,
    ) -> None:
        receipt = direct_phone.finish(require_execution=require_execution)
        with self._lock:
            row = receipt.to_json()
            self._direct_phone_receipts.append(row)
            self._usb_restore_receipts.append(dict(row["restoration"]))
            self._phone_session_active = False
            self._last_phone_parameters = None
            self._phone_terminal_sent = True
            self._phone_executor_id = None
            self._phone_parameters = None
            self._phone_residency = None

    def _abort_direct_phone(
        self, direct_phone: DirectPhoneFfnSession
    ) -> None:
        restoration = direct_phone.abort()
        with self._lock:
            restored = restoration.to_json()
            if restored not in self._usb_restore_receipts:
                self._usb_restore_receipts.append(restored)
            self._phone_session_active = False
            self._last_phone_parameters = None
            self._phone_terminal_sent = True
            self._phone_executor_id = None
            self._phone_parameters = None
            self._phone_residency = None

    def _stop_phone_session(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            direct_phone = self._current_direct_phone
            if terminate_phone_session and any(
                row.parameters.get("android_control_transport") == "adb-ncm"
                for row in getattr(self, "_live_executors", {}).values()
            ):
                raise PhysicalAdapterError("FunctionFS transport is still leased by a whole-phone endpoint")
        if direct_phone is not None:
            if not terminate_phone_session:
                return
            try:
                if allow_incomplete_direct_phone:
                    self._abort_direct_phone(direct_phone)
                else:
                    self._finish_direct_phone(
                        direct_phone,
                        require_execution=require_phone_execution,
                    )
            finally:
                with self._lock:
                    if self._current_direct_phone is direct_phone:
                        self._current_direct_phone = None
        else:
            self._close_bridge(
                terminate_phone_session=terminate_phone_session
            )

    def _stop_executor(
        self,
        executor_id: str,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            state = self._live_executors.pop(executor_id, None)
            owns_phone = (
                state.owns_phone_session
                if state is not None
                else self._phone_executor_id == executor_id
            )
        self._stop_popped_executor(
            executor_id,
            state,
            owns_phone,
            terminate_phone_session=terminate_phone_session,
            allow_incomplete_direct_phone=allow_incomplete_direct_phone,
            require_phone_execution=require_phone_execution,
        )

    def _server_exit_log(self) -> list[dict[str, object]]:
        with self._lock:
            log = getattr(self, "_server_exit_events", None)
            if log is None:
                log = []
                self._server_exit_events = log
            return log

    @property
    def server_exit_events(self) -> tuple[dict[str, object], ...]:
        """``SERVER_EXITED`` rows of every reaped llama-server (drop recovery)."""
        with self._lock:
            return tuple(dict(row) for row in getattr(self, "_server_exit_events", ()))

    def reap_exited_executor(
        self,
        executor_id: str,
        *,
        expected_server: object | None = None,
    ) -> dict[str, object] | None:
        """Forget one dynamic executor whose managed llama-server has exited.

        Elastic phones, drop recovery: the dead endpoint leaves the residency map, so the next
        snapshot reports it cold, route generation plans a load and ``_begin_transition_execution``
        relaunches it. The phone session is kept (``terminate_phone_session=False``). Exactly one
        caller reaps a given server (identity check under the rig lock), so each exit yields one
        ``SERVER_EXITED`` row; a live, replaced, retiring or unmanaged endpoint is left alone (None).
        """
        if type(executor_id) is not str or not executor_id:
            raise PhysicalAdapterError("reaped executor id is invalid")
        with self._lock:
            state = self._live_executors.get(executor_id)
            if state is None or (
                expected_server is not None and state.server is not expected_server
            ) or executor_id in getattr(self, "_retiring_executors", {}):
                return None
            exit_code = getattr(state.server, "exit_code", None)
            if not callable(exit_code) or getattr(state.server, "process", None) is None:
                return None
            returncode = exit_code()
            if returncode is None:
                return None
            del self._live_executors[executor_id]
            event = {
                "artifact_sha256": state.manifest.artifact_sha256,
                "at_us": max(0, (time.monotonic_ns() - self.epoch_ns) // 1000),
                "endpoint": state.endpoint,
                "executor_id": executor_id,
                "generation": state.generation,
                "kind": "SERVER_EXITED",
                "returncode": returncode,
            }
            self._server_exit_log().append(event)
            self._credit_reaped_gpu_memory(state, exit_ns=time.monotonic_ns())
        self._stop_popped_executor(
            executor_id,
            state,
            state.owns_phone_session,
            terminate_phone_session=False,
        )
        return dict(event)

    # Seconds a co-tenant request waits for another request's retire of their shared server.
    _retire_wait_timeout_s: float = 60.0

    def retire_helper_lost_executor(
        self,
        executor_id: str,
        *,
        expected_server: object,
        device_ids: tuple[str, ...],
    ) -> int | None:
        """Stop and forget a live dynamic llama-server that lost one of its FFN helpers.

        Elastic phones, drop recovery. llama-server survives a helper loss: ``update_slots``
        catches the aborted decode, errors every processing slot and keeps serving
        (tools/server/server-context.cpp), while the lost helper's FFN client latches its failure
        (examples/layersplit/ffn-split-client.cpp ``set_error``) and aborts every later graph that
        reaches it. The endpoint is therefore never reused: it is stopped while it stays in the
        residency map marked retiring (samples report it unavailable, no reaper or load touches
        it), then forgotten like an exited server (one ``SERVER_EXITED`` row naming the lost
        devices), so the recovery snapshot reports it cold and route generation plans a reload
        whose new process connects only the helpers its policy uses. Exactly one caller stops a
        given server; a co-tenant request of the same server waits for that stop. Returns the
        server's exit code, or None when this rig does not manage the endpoint (resident,
        replaced or unknown) or the stop did not end the process in time.
        """
        if type(executor_id) is not str or not executor_id:
            raise PhysicalAdapterError("retired executor id is invalid")
        if not isinstance(device_ids, tuple) or not device_ids or any(
            type(row) is not str or not row for row in device_ids
        ):
            raise PhysicalAdapterError("retired executor lost devices are invalid")
        with self._lock:
            retiring = self._retiring_executor_log()
            pending = retiring.get(executor_id)
            state = self._live_executors.get(executor_id)
            owner = (
                pending is None
                and state is not None
                and state.server is expected_server
                and callable(getattr(expected_server, "exit_code", None))
                and getattr(expected_server, "process", None) is not None
            )
            if owner:
                pending = (expected_server, threading.Event())
                retiring[executor_id] = pending
        if not owner:
            if pending is not None and pending[0] is expected_server:
                pending[1].wait(self._retire_wait_timeout_s)
            exit_code = getattr(expected_server, "exit_code", None)
            return exit_code() if callable(exit_code) else None
        try:
            stopping_ns = time.monotonic_ns()
            self._stop_popped_executor(
                executor_id,
                state,
                state.owns_phone_session,
                terminate_phone_session=False,
            )
            returncode = expected_server.exit_code()
            if returncode is None:
                raise PhysicalAdapterError("retired llama-server did not exit")
            with self._lock:
                if self._live_executors.get(executor_id) is state:
                    del self._live_executors[executor_id]
                event = {
                    "artifact_sha256": state.manifest.artifact_sha256,
                    "at_us": max(0, (time.monotonic_ns() - self.epoch_ns) // 1000),
                    "cause": "HELPER_LOST",
                    "endpoint": state.endpoint,
                    "executor_id": executor_id,
                    "failed_device_ids": list(device_ids),
                    "generation": state.generation,
                    "kind": "SERVER_EXITED",
                    "returncode": returncode,
                }
                self._server_exit_log().append(event)
                self._retired_helper_losses = (
                    *getattr(self, "_retired_helper_losses", ()), (expected_server, device_ids)
                )
                self._credit_reaped_gpu_memory(
                    state, exit_ns=time.monotonic_ns(), held_until_ns=stopping_ns
                )
            return returncode
        finally:
            with self._lock:
                if self._retiring_executor_log().get(executor_id) is pending:
                    del self._retiring_executor_log()[executor_id]
            pending[1].set()

    def _await_retire(self, executor_id: str, server: object | None = None) -> None:
        """Wait (bounded) until another request's retire of this executor's server finished, so a
        failure classified while the server stops reaches the scheduler only once the residency
        map no longer holds it (the recovery snapshot then reports it cold)."""
        with self._lock:
            pending = getattr(self, "_retiring_executors", {}).get(executor_id)
        if pending is not None and (server is None or pending[0] is server):
            pending[1].wait(self._retire_wait_timeout_s)

    def _require_unretired_endpoint(self, executor_id: str, state) -> None:
        """Refuse to start work on a dynamic server this rig is retiring or has forgotten.

        Drop recovery only. A ticket acquired on the hot route just before the retire (or reap)
        of its server finds the endpoint stopping or gone: that is the server's exit, not a
        mismatch of the ticket, so it raises ``LlamaServerExitedError`` and the attempt is
        recovered as ``server_exited`` before it started (through the reload). Without a
        ``SERVER_EXITED`` row for the executor nothing changes.
        """
        if not self._elastic_drop_recovery():
            return
        with self._lock:
            retiring = getattr(self, "_retiring_executors", {}).get(executor_id)
            exits = [row for row in getattr(self, "_server_exit_events", ())
                     if row["executor_id"] == executor_id]
        if retiring is not None and state is not None and state.server is retiring[0]:
            exit_code = getattr(state.server, "exit_code", None)
            returncode = exit_code() if callable(exit_code) else None
        elif state is None and exits:
            returncode = exits[-1]["returncode"]
        else:
            return
        raise LlamaServerExitedError(
            "llama-server execution endpoint is not active",
            executor_id=executor_id,
            returncode=returncode,
        )

    def _retired_lost_devices(self, server: object) -> tuple[str, ...]:
        """The lost helpers this rig retired ``server`` for (empty: never retired)."""
        with self._lock:
            return next((
                devices for retired, devices in getattr(self, "_retired_helper_losses", ())
                if retired is server
            ), ())

    def _retiring_executor_log(self) -> dict[str, tuple[object, threading.Event]]:
        with self._lock:
            log = getattr(self, "_retiring_executors", None)
            if log is None:
                log = {}
                self._retiring_executors = log
            return log

    def _credit_reaped_gpu_memory(self, state, *, exit_ns: int, held_until_ns: int | None = None) -> None:
        """Remember the VRAM a reaped server held until the GPU samples catch up with its exit.

        ``_snapshot_once`` records the last NVML per-process bytes of every live GPU server
        (drop recovery only). The server held them at least until that probe, and until
        ``held_until_ns`` when this rig stopped it itself; a later GPU sample is authoritative.
        """
        probed = getattr(self, "_executor_gpu_bytes", {}).get(state.executor_id)
        if probed is None or probed[0] is not state.server:
            return
        _server, amount, probed_ns = probed
        self._reaped_gpu_memory_log().append({
            "bytes": amount,
            "executor_id": state.executor_id,
            "exit_ns": exit_ns,
            "held_until_ns": (
                probed_ns if held_until_ns is None else max(probed_ns, held_until_ns)
            ),
            "wait_until_ns": exit_ns + int(self._reaped_gpu_sample_timeout_s * 1_000_000_000),
        })

    def _reaped_gpu_memory_log(self) -> list[dict[str, object]]:
        with self._lock:
            log = getattr(self, "_reaped_gpu_memory", None)
            if log is None:
                log = []
                self._reaped_gpu_memory = log
            return log

    def _stop_popped_executor(
        self,
        executor_id: str,
        state,
        owns_phone: bool,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        if state is None and not owns_phone:
            return
        if state is not None:
            try:
                state.server.stop()
            except BaseException:
                if state.parameters.get("android_control_transport") == "adb-ncm":
                    with self._lock:
                        self._live_executors.setdefault(executor_id, state)
                raise
            self._dormant_forget_server(state.endpoint)
            power = getattr(self, "_device_power", None)
            if power is not None:
                power.on_server_stopped(executor_id)
            if getattr(self, "_masked_helpers", None):
                self._end_helper_masks(executor_id, state.server, "SERVER_STOPPED")
        if owns_phone:
            self._stop_phone_session(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )

    def _stop_dynamic_executors(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        with self._lock:
            executor_ids = tuple(sorted(self._live_executors, key=lambda key:
                self._live_executors[key].parameters.get("android_control_transport") != "adb-ncm"
            )) if terminate_phone_session else tuple(self._live_executors)
            phone_executor_id = self._phone_executor_id
        for executor_id in executor_ids:
            self._stop_executor(
                executor_id,
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )
        with self._lock:
            phone_still_active = self._phone_executor_id is not None
        if phone_executor_id is not None and phone_still_active:
            self._stop_phone_session(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )

    def _stop_current(
        self,
        *,
        terminate_phone_session: bool = True,
        allow_incomplete_direct_phone: bool = False,
        require_phone_execution: bool = True,
    ) -> None:
        """Compatibility entry point for callers predating the residency map."""
        if hasattr(self, "_live_executors"):
            self._stop_dynamic_executors(
                terminate_phone_session=terminate_phone_session,
                allow_incomplete_direct_phone=(
                    allow_incomplete_direct_phone
                ),
                require_phone_execution=require_phone_execution,
            )
            return
        with self._lock:
            server = getattr(self, "_current_server", None)
            direct_phone = getattr(self, "_current_direct_phone", None)
        if server is not None:
            server.stop()
        if direct_phone is not None:
            if allow_incomplete_direct_phone:
                self._abort_direct_phone(direct_phone)
            else:
                self._finish_direct_phone(
                    direct_phone,
                    require_execution=require_phone_execution,
                )
            with self._lock:
                self._current_direct_phone = None
        else:
            self._close_bridge(
                terminate_phone_session=terminate_phone_session
            )
        with self._lock:
            self._current_executor_id = None
            self._current_manifest = None
            self._current_parameters = None
            self._current_operator_plan = None

    def _start_bridge(
        self,
        command: PhysicalTransitionCommand,
    ) -> CapturedProcess:
        parameters = command.adapter_parameters
        output = self.configuration.output_directory
        label = "bridge-" + str(self._launch_attempt)
        environment = os.environ.copy()
        environment["S42_FFN_BRIDGE_QUALIFICATION_CLIENTS"] = "1"
        environment["S42_FFN_BRIDGE_SHUTDOWN_CLIENTS"] = "1"
        process = CapturedProcess(
            [
                str(self.configuration.bridge_path),
                str(parameters["ffn_bridge_host"]),
                str(parameters["ffn_bridge_port"]),
                str(parameters["bridge_allocator"]),
            ],
            environment,
            output,
            label,
        )
        process.start()
        try:
            self._qualify_started_bridge(process, command)
        except BaseException:
            process.terminate()
            raise
        return process

    def _qualify_started_bridge(
        self,
        process: CapturedProcess,
        command: PhysicalTransitionCommand,
    ) -> None:
        parameters = command.adapter_parameters
        process.wait_stderr("[ffn-dmabuf-bridge] ready", 30)
        usb = probe_functionfs_usb_device(
            vendor_id=str(parameters["functionfs_vendor_id"]),
            product_id=str(parameters["functionfs_product_id"]),
        )
        client_receipt = qualify_functionfs_bridge(
            host=str(parameters["ffn_bridge_host"]),
            port=int(parameters["ffn_bridge_port"]),
            layer_mask=int(parameters["ffn_layer_mask"]),
            n_embd=int(parameters["ffn_n_embd"]),
            columns=int(parameters["ffn_columns"]),
            max_tokens=int(parameters["bridge_max_tokens"]),
            activation=str(parameters["ffn_activation"]),
            token_shapes=(1, min(32, int(parameters["bridge_max_tokens"]))),
            repeats=4,
            timeout_s=120,
        )
        process.wait_stderr("FFNDMABUFQUAL ", 120)
        bridge_receipt = parse_functionfs_bridge_qualification(
            tuple(process.stderr_lines)
        )
        client_values = client_receipt.to_json()
        if (
            bridge_receipt["calls"] != client_values["calls"]
            or bridge_receipt["upload_bytes"]
                != client_values["request_payload_bytes"]
            or bridge_receipt["download_bytes"]
                != client_values["response_payload_bytes"]
            or bridge_receipt["allocator"]
                != parameters["bridge_allocator"]
        ):
            raise PhysicalAdapterError(
                "FunctionFS qualification endpoints disagree"
            )
        h2d_bandwidth = int(
            float(bridge_receipt["h2d_payload_bytes_per_s"])
        )
        d2h_bandwidth = int(float(
            bridge_receipt[
                "d2h_conservative_payload_bytes_per_s"
            ]
        ))
        if h2d_bandwidth <= 0 or d2h_bandwidth <= 0:
            raise PhysicalAdapterError(
                "FunctionFS qualification bandwidth is invalid"
            )
        composite = self.configuration.catalog.composite_executor_by_id.get(
            command.participant.executor_id
        )
        used_resources = set(
            command.participant.resource_ids
            if composite is None else composite.resource_ids
        )
        identity = self._transport_identity(
            used_resources, command.adapter_parameters
        )
        link_bandwidth = {}
        for link in self.configuration.catalog.placement_profile.links:
            if "link:" + link.link_id not in used_resources:
                continue
            if link.target_device == self.configuration.phone_device_id:
                observed_bandwidth = h2d_bandwidth
            elif link.source_device == self.configuration.phone_device_id:
                observed_bandwidth = d2h_bandwidth
            else:
                continue
            link_bandwidth[link.link_id] = min(
                link.bandwidth_bytes_per_s, observed_bandwidth
            )
        if not link_bandwidth:
            raise PhysicalAdapterError(
                "FunctionFS qualification has no catalog transport link"
            )
        qualification = {
            "allocator": bridge_receipt["allocator"],
            "bridge": dict(bridge_receipt),
            "bridge_binary_sha256": self._bridge_sha256,
            "client": client_values,
            "cost_estimator_bandwidth_bytes_per_s": dict(sorted(
                link_bandwidth.items()
            )),
            "functionfs_identity": identity,
            "queue_depth": int(parameters["bridge_queue_depth"]),
            "usb": usb.to_json(),
        }
        with self._lock:
            self._transport_qualifications.append(qualification)
            self._link_bandwidth_samples.update(link_bandwidth)

    def close(self, *, require_phone_execution: bool = True) -> None:
        cleanup_errors = []
        try:
            if require_phone_execution:
                self._stop_current()
            else:
                self._stop_current(require_phone_execution=False)
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            self._stop_co_helpers()
        except BaseException as error:
            cleanup_errors.append(error)
        android_launcher = getattr(self, "_android_phone_launcher", None)
        if android_launcher is not None:
            try:
                if any(row.parameters.get("android_control_transport") == "adb-ncm"
                       for row in getattr(self, "_live_executors", {}).values()):
                    raise PhysicalAdapterError("Android control is still leased by a live endpoint")
                android_launcher.close_control()
            except BaseException as error:
                cleanup_errors.append(error)
        if self._resident_server is not None:
            try:
                self._resident_server.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            finally:
                self._resident_server = None
        if self._runtime_monitor_started:
            try:
                self._runtime_monitor.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._runtime_monitor_started = False
        power = getattr(self, "_device_power", None)
        if power is not None:
            # restore the clocks while the host sampler still runs so the restore is on record
            try:
                power.close()
            except BaseException as error:
                cleanup_errors.append(error)
        if self._phone_sampler_started:
            try:
                self._phone_sampler.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._phone_sampler_started = False
        if self._server_sampler_started:
            try:
                self._sampler.stop()
            except BaseException as error:
                cleanup_errors.append(error)
            self._server_sampler_started = False
        if cleanup_errors:
            raise PhysicalAdapterError(
                "physical cleanup failed: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
