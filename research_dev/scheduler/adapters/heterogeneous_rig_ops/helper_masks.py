"""Physical rig: mask a lost FFN helper out of a live llama-server (elastic phones S2a).

``elastic_phones.helper_loss_recovery: "mask_out"`` (drop recovery only). llama-server survives a
helper loss: it errors every processing slot and keeps serving (tools/server/server-context.cpp).
With the S2a server change (S2A_SERVER.diff, announced by ``S41SERVERFFNCAPS helper_mask_out=1``)
a failed helper client whose runtime policy owns none of its layers is idle, so the live server
keeps serving every policy without the lost helper, and a later policy that owns the helper again
reconnects it (TCP only: ``helper_reconnect_tcp=1``). The rig therefore keeps such a server instead of
retiring it: it records the lost helper as masked out on that server, the scheduler recovers the
errored requests on the same route, and a guard refuses every FFN control that would give the
masked helper a layer on that server (fail closed). A server without the capability, a split
server (its recovery is the paired desktop route elsewhere), or a masked helper lost again by an
attempt that started after the mask is retired as before (fix 3).

Events (``helper_mask_events``): HELPER_MASKED_OUT, HELPER_MASK_OUT_REFUSED, HELPER_POLICY_DROPPED,
HELPER_REATTACHED (live reconnect on readmission), HELPER_REATTACH_PENDING_RELAUNCH (a USB helper,
or a server that cannot reconnect: the helper stays masked until the server is stopped or
relaunched), HELPER_MASK_ENDED (the masked server stopped or was replaced) and
PRIMARY_SESSION_RELAUNCHED (the lost primary's FunctionFS session was started afresh by a transition
instead of being reused).
"""

from __future__ import annotations

import time
from typing import Mapping

from ..._internal.adaptive_decode_contracts import policy_device_set
from ..contracts import PhysicalAdapterError, elastic_helper_loss_recovery

_CAPABILITIES_PREFIX = "S41SERVERFFNCAPS "
MASK_OUT_CAPABILITY = "helper_mask_out"
# a TCP helper reconnects on the live server when a policy owns it again; a FunctionFS (USB) worker
# reads a new HELLO only after its session was relaunched, so a USB helper waits for a relaunch
RECONNECT_TCP_CAPABILITY = "helper_reconnect_tcp"


def server_helper_capabilities(server: object) -> frozenset[str]:
    """Helper-recovery capabilities a llama-server announced at startup (``name=1`` words)."""
    lines = getattr(server, "stderr_lines", None)
    if not isinstance(lines, (list, tuple)):
        return frozenset()
    for line in tuple(lines):
        if type(line) is str and line.startswith(_CAPABILITIES_PREFIX):
            return frozenset(
                word[:-2] for word in line[len(_CAPABILITIES_PREFIX):].split() if word.endswith("=1")
            )
    return frozenset()


def server_helper_layer_masks(rows, environment: Mapping[str, str]) -> dict[str, int | None]:
    """Launch layer mask of each known helper device of one server (None: not in the environment).

    ``rows`` are ``server_helper_rows`` (label, device, transport) in helper order; a single-helper
    server owns the whole ``S41_SERVER_FFN_LAYER_MASK``.
    """
    several = environment.get("S41_SERVER_FFN_HELPERS") is not None
    result: dict[str, int | None] = {}
    for index, (_label, device, _transport) in enumerate(rows):
        if device is None:
            continue
        value = environment.get(
            f"S41_SERVER_FFN_HELPER{index}_LAYER_MASK" if several else "S41_SERVER_FFN_LAYER_MASK"
        )
        result[device] = int(value) if type(value) is str and value.isdigit() and int(value) > 0 else None
    return result


class RigHelperMaskMixin:
    """Physical rig: masked-out helpers of live llama-servers (elastic phones S2a)."""

    def _helper_loss_recovery(self) -> str:
        """``elastic_phones.helper_loss_recovery`` of this rig (``retire`` when absent)."""
        elastic = getattr(getattr(self, "configuration", None), "elastic_phones", None)
        if elastic is None:
            elastic = getattr(self, "_elastic_phones", None)
        return elastic_helper_loss_recovery(elastic)

    @property
    def helper_mask_events(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            return tuple(dict(row) for row in getattr(self, "_helper_mask_log", ()))

    def _masked_helper_map(self) -> dict[str, dict[str, object]]:
        """executor id -> {"server": the masked process, "devices": {device: mask row}} (lock held)."""
        masks = getattr(self, "_masked_helpers", None)
        if masks is None:
            masks = self._masked_helpers = {}
        return masks

    def _record_helper_mask(self, kind: str, executor_id: str, device_id: str, **details: object) -> None:
        """Append one ``helper_mask_events`` row (lock held)."""
        log = getattr(self, "_helper_mask_log", None)
        if log is None:
            log = self._helper_mask_log = []
        log.append({
            "at_us": max(0, (time.monotonic_ns() - self.epoch_ns) // 1000),
            "device": device_id,
            "kind": kind,
            "server": executor_id,
            **details,
        })

    def masked_helper_devices(self, executor_id: str) -> tuple[str, ...]:
        """Helpers masked out of the live server of ``executor_id`` (empty: none or replaced)."""
        with self._lock:
            masks = getattr(self, "_masked_helpers", {}).get(executor_id)
            state = self._live_executors.get(executor_id)
            if masks is None or state is None or state.server is not masks["server"]:
                return ()
            return tuple(sorted(masks["devices"]))

    def mask_out_lost_helpers(
        self,
        command,
        *,
        expected_server: object,
        device_ids: tuple[str, ...],
        attempt_started_ns: int | None,
    ) -> str | None:
        """Keep a live desktop-parent server that lost ``device_ids`` and mask them out.

        Returns None when the server keeps serving without them, otherwise why it cannot (the
        caller retires it). Idempotent for the co-tenants of one loss: an attempt that started
        before the mask was recorded sees the same loss; a masked helper lost again by a later
        attempt means the server did not stay clean, so it is refused (retire, fail closed).
        """
        executor_id = command.executor_id
        if not isinstance(device_ids, tuple) or not device_ids or any(
            type(row) is not str or not row for row in device_ids
        ):
            raise PhysicalAdapterError("masked-out helper devices are invalid")
        contract = getattr(command, "execution_contract", None)
        environment = getattr(expected_server, "environment", None)
        rows = self._server_helper_rows(expected_server)
        layer_masks = server_helper_layer_masks(
            rows, environment if isinstance(environment, Mapping) else {},
        )
        transports = {device: transport for _label, device, transport in rows if device is not None}
        exit_code = getattr(expected_server, "exit_code", None)
        with self._lock:
            state = self._live_executors.get(executor_id)
            masks = self._masked_helper_map().get(executor_id)
            if masks is not None and (state is None or masks["server"] is not state.server):
                self._end_helper_masks_locked(executor_id, "SERVER_REPLACED")
                masks = None
            reason = None
            if getattr(contract, "execution_mode", None) != "desktop":
                # a split server's requests recover on the paired desktop route: nothing to keep
                reason = "NOT_A_DESKTOP_PARENT"
            elif state is None or state.server is not expected_server:
                reason = "SERVER_NOT_LIVE"
            elif executor_id in getattr(self, "_retiring_executors", {}):
                reason = "SERVER_RETIRING"
            elif not callable(exit_code) or exit_code() is not None:
                reason = "SERVER_EXITED"
            elif MASK_OUT_CAPABILITY not in server_helper_capabilities(expected_server):
                reason = "SERVER_LACKS_HELPER_MASK_OUT"
            elif any(device not in layer_masks for device in device_ids):
                reason = "LOST_DEVICE_IS_NOT_A_HELPER"
            elif masks is not None and any(
                device in masks["devices"] and (
                    attempt_started_ns is None
                    or attempt_started_ns >= masks["devices"][device]["masked_ns"]
                )
                for device in device_ids
            ):
                reason = "MASKED_HELPER_LOST_AGAIN"
            if reason is not None:
                for device in device_ids:
                    self._record_helper_mask(
                        "HELPER_MASK_OUT_REFUSED", executor_id, device, reason=reason,
                        ticket_id=getattr(command, "ticket_id", None))
                return reason
            if masks is None:
                masks = self._masked_helper_map()[executor_id] = {
                    "server": expected_server, "devices": {},
                }
            now = time.monotonic_ns()
            for device in device_ids:
                if device in masks["devices"]:
                    continue
                masks["devices"][device] = {"layer_mask": layer_masks[device], "masked_ns": now,
                                            "transport": transports.get(device)}
                self._record_helper_mask(
                    "HELPER_MASKED_OUT", executor_id, device,
                    endpoint=state.endpoint, generation=state.generation,
                    layer_mask=layer_masks[device], ticket_id=getattr(command, "ticket_id", None),
                    transport=transports.get(device))
            return None

    def _scheduler_quarantined_devices(self) -> frozenset[str]:
        reader = getattr(getattr(self, "_scheduler", None), "quarantined_devices", None)
        if not callable(reader):
            return frozenset()
        value = reader()
        return frozenset(value) if isinstance(value, (Mapping, set, frozenset, tuple, list)) else frozenset()

    def _helper_mask_guard(self, command, control) -> None:
        """Refuse an FFN control that gives a masked-out or quarantined helper a layer.

        The HTTP backend calls it (mask_out mode only) before any control reaches a server; an
        adaptive control refused here fails like a refused server control (the controller drops
        the device set and returns to the host policy). A policy is dropped when it names such a
        device as an owner or overlaps the device's launch layers on this server; a masked helper
        whose launch mask is unknown blocks every assisted policy of its server.
        """
        policy = control.policy
        if policy.baseline:
            return
        quarantined = self._scheduler_quarantined_devices()
        with self._lock:
            state = self._live_executors.get(command.executor_id)
            masks = getattr(self, "_masked_helpers", {}).get(command.executor_id)
            if masks is not None and (state is None or state.server is not masks["server"]):
                self._end_helper_masks_locked(command.executor_id, "SERVER_REPLACED")
                masks = None
            excluded = {} if masks is None else {
                device: ("MASKED_OUT", row["layer_mask"]) for device, row in masks["devices"].items()
            }
        if quarantined:
            environment = None if state is None else getattr(state.server, "environment", None)
            launch = {} if state is None else server_helper_layer_masks(
                self._server_helper_rows(state.server),
                environment if isinstance(environment, Mapping) else {},
            )
            for device in quarantined:
                # a quarantined device that is not a helper of this server can still own a policy
                excluded.setdefault(device, ("QUARANTINED", launch[device] if device in launch else 0))
        owners = set(policy_device_set(policy))
        blocked = tuple(sorted(
            device for device, (_reason, mask) in excluded.items()
            if device in owners or mask is None or policy.layer_mask & mask
        ))
        if not blocked:
            return
        with self._lock:
            for device in blocked:
                self._record_helper_mask(
                    "HELPER_POLICY_DROPPED", command.executor_id, device, reason=excluded[device][0],
                    policy_hash=policy.policy_hash, request_id=command.request_id,
                    layer_mask=policy.layer_mask)
        raise PhysicalAdapterError(
            "FFN control gives masked-out or quarantined helper " + ",".join(blocked)
            + " a layer on " + command.executor_id
        )

    def _reattach_masked_helpers(self, device_id: str) -> None:
        """A readmitted helper returns to every live server that masked it out.

        A TCP helper of a server that reconnects re-owned TCP helpers (``helper_reconnect_tcp``)
        comes back at the next policy that owns its layers. A USB (FunctionFS) helper, or any helper
        of a server without that capability, stays masked until the server is stopped or relaunched
        by a planned transition: the new process connects it afresh, after the transition path
        relaunched its phone session.
        """
        with self._lock:
            masks_by_server = getattr(self, "_masked_helpers", {})
            for executor_id, masks in tuple(masks_by_server.items()):
                row = masks["devices"].get(device_id)
                if row is None:
                    continue
                state = self._live_executors.get(executor_id)
                if state is None or state.server is not masks["server"]:
                    self._end_helper_masks_locked(executor_id, "SERVER_REPLACED")
                    continue
                if (row.get("transport") == "tcp"
                        and RECONNECT_TCP_CAPABILITY in server_helper_capabilities(masks["server"])):
                    del masks["devices"][device_id]
                    self._record_helper_mask(
                        "HELPER_REATTACHED", executor_id, device_id, mode="live_reconnect",
                        generation=state.generation)
                    if not masks["devices"]:
                        del masks_by_server[executor_id]
                elif not row.get("pending_relaunch"):
                    row["pending_relaunch"] = True
                    self._record_helper_mask(
                        "HELPER_REATTACH_PENDING_RELAUNCH", executor_id, device_id,
                        generation=state.generation,
                        reason="USB_SESSION" if row.get("transport") != "tcp" else "SERVER_CANNOT_RECONNECT")

    def _end_helper_masks(self, executor_id: str, server: object | None, cause: str) -> None:
        """The masked server stopped: its masks end with it (a relaunch connects afresh)."""
        with self._lock:
            masks = getattr(self, "_masked_helpers", {}).get(executor_id)
            if masks is not None and (server is None or masks["server"] is server):
                self._end_helper_masks_locked(executor_id, cause)

    def _end_helper_masks_locked(self, executor_id: str, cause: str) -> None:
        masks = getattr(self, "_masked_helpers", {}).pop(executor_id, None)
        if masks is None:
            return
        for device, row in sorted(masks["devices"].items()):
            self._record_helper_mask(
                "HELPER_MASK_ENDED", executor_id, device, cause=cause,
                pending_relaunch=bool(row.get("pending_relaunch")))

    def _require_direct_phone_relaunch(self, device_id: str, command) -> None:
        """The primary phone was lost: the next transition that needs its FunctionFS session
        relaunches it (stop, then start) instead of reusing the session the loss left behind."""
        with self._lock:
            if getattr(self, "_direct_phone_relaunch_required", None) is None:
                self._direct_phone_relaunch_required = device_id
                self._record_helper_mask(
                    "PRIMARY_SESSION_RELAUNCH_REQUIRED", command.executor_id, device_id,
                    ticket_id=getattr(command, "ticket_id", None))

    def _direct_phone_relaunch_pending(self) -> bool:
        with self._lock:
            return getattr(self, "_direct_phone_relaunch_required", None) is not None

    def _direct_phone_relaunched(self, executor_id: str) -> None:
        with self._lock:
            device_id = getattr(self, "_direct_phone_relaunch_required", None)
            if device_id is not None:
                self._direct_phone_relaunch_required = None
                self._record_helper_mask("PRIMARY_SESSION_RELAUNCHED", executor_id, device_id)
