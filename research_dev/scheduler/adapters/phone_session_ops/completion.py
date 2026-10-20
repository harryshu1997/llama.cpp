"""DirectPhoneFfnSession completion operations on its existing owner."""

from __future__ import annotations

from dataclasses import replace
import shlex

from ..bridge import AndroidUsbRestorationReceipt, verify_android_usb_restored
from ..contracts import PhysicalAdapterError
from ..phone_session_contracts.events import (
    parse_direct_phone_ffn_terminal,
    parse_phone_residency_phase_events,
    parse_phone_residency_call_events,
)
from ..phone_session_contracts.receipts import DirectPhoneFfnCloseReceipt


def _acknowledge_restoration(controller, remote_root: str | None) -> None:
    """Tell the phone's restore watchdog that adb works again (restore_android_usb.sh restarts adbd
    without it). Best effort: the receipt does not depend on it."""
    if not remote_root:
        return
    try:
        controller._adb(
            "su -c " + shlex.quote("touch " + shlex.quote(remote_root + "/host-ack"))
        )
    except Exception:  # noqa: BLE001 - the acknowledgement is advisory
        return


# 2026-10-06: seconds between close attempts while another host process (a live llama-server's FFN
# client) still claims the phone's FunctionFS interface
_USB_BUSY_RETRY_S = 1.0


def _close_until_restored(controller, timeout_s: float, *, sleep=None, clock=None):
    """Send the direct-phone USB close, retrying while the interface is busy, until the phone is back on adb.

    A whole relaunch of a phone (a one-session replacement) can start while a live llama-server's FFN
    client still claims the phone's FunctionFS interface (the desktop transition stops that server in
    parallel). The close tool then fails with ``LIBUSB_ERROR_BUSY``; before this the close was attempted
    once and the rest of the window only waited for a restoration that could not happen (pAlpf1pix2,
    pAlpc1pix2: one ~90 s SESSION_FAILED per run). Now the close is retried every second while it is
    busy, with a non-blocking adb check in between (the server's own exit may already have ended the
    worker). A close that succeeds or fails for another reason gets the rest of the window to restore.
    Returns the restoration receipt; raises with the last close error and the restoration error.
    """
    import time as _time  # noqa: PLC0415 - injectable for tests
    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    deadline = clock() + timeout_s
    while True:
        close_error = None
        try:
            controller._close_current_usb()
        except PhysicalAdapterError as error:
            close_error = error
        busy = close_error is not None and "LIBUSB_ERROR_BUSY" in str(close_error)
        retry_left = busy and deadline - clock() > _USB_BUSY_RETRY_S
        try:
            return verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
                timeout_s=0 if retry_left else max(0, int(deadline - clock())),
            )
        except PhysicalAdapterError as restoration_error:
            if not retry_left:
                details = (str(close_error) + "; " if close_error is not None else "") + str(restoration_error)
                raise PhysicalAdapterError(details) from restoration_error
        sleep(_USB_BUSY_RETRY_S)

def finish(
    controller, *, require_execution: bool = True
) -> DirectPhoneFfnCloseReceipt:
    launch = controller._launch
    remote_root = controller._remote_root
    if launch is None or remote_root is None:
        raise PhysicalAdapterError("direct phone session is not active")
    try:
        restoration = verify_android_usb_restored(
            serial=controller.configuration.serial,
            adb_port=controller.configuration.adb_port,
            minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
            timeout_s=0,
        )
    except PhysicalAdapterError as restoration_error:
        try:
            restoration = _close_until_restored(controller, 90)
        except PhysicalAdapterError as final_restoration_error:
            raise PhysicalAdapterError(
                str(restoration_error) + "; " + str(final_restoration_error)
            ) from final_restoration_error
    _acknowledge_restoration(controller, remote_root)
    try:
        worker_log = controller._adb(
            "su -c "
            + shlex.quote(
                "cat " + shlex.quote(remote_root + "/worker.log")
            )
        )
        terminal_log = worker_log
        residency_phase_events = parse_phone_residency_phase_events(
            worker_log.splitlines()
        )
        if launch.phone_shards:
            controller._validate_shard_loads(
                worker_log.splitlines(), tuple(controller._proof_shards)
            )
            launch = replace(
                launch,
                weight_sources=controller._phone_weight_sources(
                    launch.phone_shards, residency_phase_events
                ),
            )
            terminal_log = controller._adb(
                "su -c "
                + shlex.quote(
                    "cat " + shlex.quote(remote_root + "/router.log")
                )
            )
        terminal = parse_direct_phone_ffn_terminal(
            terminal_log.splitlines()
        )
        historical = tuple(
            getattr(controller, "_historical_execution_proofs", ())
        )
        if launch.phone_shards:
            executed_shards = tuple(controller._executed_proof_shards)
            if require_execution and not executed_shards and not historical:
                raise PhysicalAdapterError(
                    "phone session has no ticket-bound execution proof"
                )
            controller._validate_shard_terminal(
                terminal,
                tuple(controller._proof_shards),
                require_execution=require_execution,
                executed_shards=executed_shards,
                historical_shards=tuple(
                    row.shard
                    for row in controller._shard_residency_windows
                ),
            )
        return DirectPhoneFfnCloseReceipt(
            launch=launch,
            terminal=terminal,
            restoration=restoration,
            bound_ticket_ids=tuple(controller._bound_ticket_ids),
            historical_execution_proofs=historical,
            residency_phase_events=residency_phase_events,
            residency_call_events=parse_phone_residency_call_events(
                terminal_log.splitlines()
            ),
        )
    finally:
        controller._launch = None
        controller._remote_root = None
        controller._bound_ticket_ids = []
        controller._proof_shards = []
        controller._executed_proof_shards = []
        controller._residency_generation = 0
        controller._shard_residency_windows = []
        controller._ticket_bind_generations = {}
        controller._historical_execution_proofs = []
        controller._execution_by_artifact = {}
        controller._transport_by_artifact = {}
        controller._load_count_by_session = {}
        controller._column_quantum_by_session = {}
        controller._max_tokens_by_session = {}
        controller._multi_session_port_by_id = {}


def abort(controller) -> AndroidUsbRestorationReceipt:
    launch = controller._launch
    if launch is None or controller._remote_root is None:
        return verify_android_usb_restored(
            serial=controller.configuration.serial,
            adb_port=controller.configuration.adb_port,
            minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
            timeout_s=180,
        )
    try:
        try:
            receipt = verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=(
                    controller.configuration.minimum_usb_speed_mbps
                ),
                timeout_s=15,
            )
            _acknowledge_restoration(controller, controller._remote_root)
            return receipt
        except PhysicalAdapterError as initial_error:
            try:
                receipt = _close_until_restored(controller, 180)
                _acknowledge_restoration(controller, controller._remote_root)
                return receipt
            except PhysicalAdapterError as restoration_error:
                raise PhysicalAdapterError(
                    str(initial_error) + "; " + str(restoration_error)
                ) from restoration_error
    finally:
        controller._launch = None
        controller._remote_root = None
        controller._bound_ticket_ids = []
        controller._proof_shards = []
        controller._executed_proof_shards = []
        controller._residency_generation = 0
        controller._shard_residency_windows = []
        controller._ticket_bind_generations = {}
        controller._historical_execution_proofs = []
        controller._execution_by_artifact = {}
        controller._transport_by_artifact = {}
        controller._load_count_by_session = {}
        controller._column_quantum_by_session = {}
        controller._max_tokens_by_session = {}
        controller._multi_session_port_by_id = {}
