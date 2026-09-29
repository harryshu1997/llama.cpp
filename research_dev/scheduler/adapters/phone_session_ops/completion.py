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
        close_error = None
        try:
            controller._close_current_usb()
        except PhysicalAdapterError as error:
            close_error = error
        try:
            restoration = verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=(
                    controller.configuration.minimum_usb_speed_mbps
                ),
                timeout_s=90,
            )
        except PhysicalAdapterError as final_restoration_error:
            details = str(restoration_error)
            if close_error is not None:
                details += "; " + str(close_error)
            details += "; " + str(final_restoration_error)
            raise PhysicalAdapterError(details) from final_restoration_error
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
            return verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=(
                    controller.configuration.minimum_usb_speed_mbps
                ),
                timeout_s=15,
            )
        except PhysicalAdapterError as initial_error:
            try:
                controller._close_current_usb()
            except PhysicalAdapterError as close_error:
                close_note = str(close_error)
            else:
                close_note = ""
            try:
                return verify_android_usb_restored(
                    serial=controller.configuration.serial,
                    adb_port=controller.configuration.adb_port,
                    minimum_speed_mbps=(
                        controller.configuration.minimum_usb_speed_mbps
                    ),
                    timeout_s=180,
                )
            except PhysicalAdapterError as restoration_error:
                details = str(initial_error)
                if close_note:
                    details += "; " + close_note
                details += "; " + str(restoration_error)
                raise PhysicalAdapterError(
                    details
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
