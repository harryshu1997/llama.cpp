"""DirectPhoneFfnSession launch operations on its existing owner."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import time
from typing import Callable, Mapping

from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import RuntimePhoneShard
from ..bridge import (
    FunctionFsUsbObservation,
    probe_functionfs_usb_device,
    verify_android_usb_restored,
)
from ..contracts import PhysicalAdapterError
from ..llama_server import PhoneFfnExecutionContract
from ..phone_transport import PhoneTransportContract
from ..ticket import PhysicalTransitionCommand
from ..phone_session_contracts.receipts import DirectPhoneFfnLaunchReceipt
from .common import _ShardResidencyWindow


def _start_remote_command(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    shards: tuple[RuntimePhoneShard, ...],
    multi_session: bool,
    shard_manifest: str | None,
    worker: str,
    model: str,
    backend: str,
) -> tuple[str, str, str, str]:
    session_identity = (
        command.ticket_id + ":" + str(os.getpid()) + ":"
        + str(time.monotonic_ns())
    )
    token = hashlib.sha256(
        session_identity.encode("ascii")
    ).hexdigest()[:16]
    remote_root = controller.configuration.session_root.rstrip("/") + "/" + token
    column_quantum = command.adapter_parameters.get("ffn_column_quantum")
    if (
        type(column_quantum) is not int
        or column_quantum <= 0
        or execution.columns % column_quantum
    ):
        raise PhysicalAdapterError(
            "direct phone FFN column quantum is invalid"
        )
    partition_count = command.adapter_parameters.get(
        "ffn_runtime_partition_count"
    )
    maximum_partitions = command.adapter_parameters.get(
        "ffn_max_runtime_partitions"
    )
    if partition_count is not None and (
        type(partition_count) is not int
        or partition_count != execution.columns // column_quantum
        or (
            maximum_partitions is not None
            and (
                type(maximum_partitions) is not int
                or maximum_partitions <= 0
                or partition_count > maximum_partitions
            )
        )
    ):
        raise PhysicalAdapterError(
            "direct phone FFN partition capacity is invalid"
        )
    words = list(controller._worker_environment(
        execution, transport, column_quantum
    ))
    controller._append_start_environment(words)
    cpu_affinity = getattr(controller.configuration, "cpu_affinity", None)
    if cpu_affinity is not None:
        words.extend(("taskset", cpu_affinity))
    words.extend((
        "sh",
        controller.configuration.session_script,
        worker,
        model,
        execution.layers,
        str(execution.columns),
        backend,
        remote_root,
        controller.configuration.restore_script,
        str(controller.configuration.session_timeout_s),
        str(controller.configuration.max_requests),
        manifest.artifact_sha256,
    ))
    if multi_session:
        assert controller.configuration.resident_workers_path is not None
        assert controller.configuration.resident_router_path is not None
        assert shard_manifest is not None
        words.extend((
            controller.configuration.resident_workers_path,
            controller.configuration.resident_router_path,
            shard_manifest,
        ))
    worker_command = " ".join(shlex.quote(value) for value in words)
    launch_log = remote_root + "/launch.log"
    background = (
        "mkdir -p " + shlex.quote(remote_root)
        + " && nohup env " + worker_command
        + " > " + shlex.quote(launch_log)
        + " 2>&1 < /dev/null &"
    )
    return token, remote_root, launch_log, "su -c " + shlex.quote(background)


def _append_start_environment(controller, words: list[str]) -> None:
    call_log_period = os.environ.get("S42_RESIDENT_CALL_LOG_PERIOD")
    if call_log_period is not None:
        if not call_log_period.isdigit() or int(call_log_period) < 1:
            raise PhysicalAdapterError(
                "resident phone call log period is invalid"
            )
        words.append(
            "S42_RESIDENT_CALL_LOG_PERIOD=" + call_log_period
        )
    session_count = getattr(
        controller.configuration, "multi_session_device_count", None
    )
    if session_count is not None:
        words.append(
            "S42_PHONE_SESSION_COUNT="
            + str(session_count)
        )
    if controller.configuration.android_gadget_path is not None:
        assert controller.configuration.functionfs_gadget_path is not None
        assert controller.configuration.functionfs_root_path is not None
        assert controller.configuration.phone_usb_controller is not None
        words.extend((
            "SCHEDULER_ANDROID_GADGET="
            + controller.configuration.android_gadget_path,
            "SCHEDULER_FUNCTIONFS_GADGET="
            + controller.configuration.functionfs_gadget_path,
            "SCHEDULER_FUNCTIONFS_ROOT="
            + controller.configuration.functionfs_root_path,
            "SCHEDULER_PHONE_UDC="
            + controller.configuration.phone_usb_controller,
        ))
    if controller.configuration.diagnostic_port:
        assert controller.configuration.busybox_path is not None
        words.extend((
            "S42_USB_NCM=1",
            "S42_DIAGNOSTIC_PORT="
            + str(controller.configuration.diagnostic_port),
            "S42_BUSYBOX=" + controller.configuration.busybox_path,
        ))


def _spawn_phone_session(
    controller,
    remote_command: str,
    control_check: Callable[[], None],
) -> subprocess.Popen[str]:
    try:
        control_check()
        return subprocess.Popen(
            [
                str(controller.configuration.adb_path),
                "-P", str(controller.configuration.adb_port),
                "-s", controller.configuration.serial,
                "shell", remote_command,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="ascii",
            errors="backslashreplace",
        )
    except OSError as error:
        raise PhysicalAdapterError(
            "phone session ADB launch failed"
        ) from error


def _remote_start_failure_detail(
    controller, remote_root: str, launch_log: str
) -> str | None:
    try:
        failure = controller._remote_launch_failure(remote_root)
    except PhysicalAdapterError:
        return None
    if failure is None:
        return None
    try:
        launch_tail = controller._adb_root(
            "tail -n 20 " + shlex.quote(launch_log)
            + " 2>/dev/null || true",
            timeout_s=5,
        ).strip()
    except PhysicalAdapterError:
        launch_tail = ""
    return failure + (": " + launch_tail if launch_tail else "")


def _wait_for_phone_session(
    controller,
    process: subprocess.Popen[str],
    transport: PhoneTransportContract,
    shards: tuple[RuntimePhoneShard, ...],
    phone_kernel_release: str,
    phone_usb_controller: str,
    remote_root: str,
    launch_log: str,
    control_check: Callable[[], None],
) -> tuple[FunctionFsUsbObservation | None, str]:
    timeout_s = controller.configuration.launch_timeout_s * max(1, len(shards))
    deadline = time.monotonic() + timeout_s
    next_remote_probe_at = time.monotonic() + 1.0
    last_error = "FunctionFS USB did not enumerate"
    usb = None
    while time.monotonic() < deadline:
        control_check()
        returncode = process.poll()
        if returncode not in {None, 0}:
            stdout, stderr = process.communicate()
            detail = stderr.strip() or stdout.strip()
            last_error = "phone session launcher exited"
            if detail:
                last_error += ": " + detail
            break
        try:
            usb = probe_functionfs_usb_device(
                vendor_id=f"{transport.vendor_id:04x}",
                product_id=f"{transport.product_id:04x}",
            )
        except PhysicalAdapterError as error:
            last_error = str(error)
            now = time.monotonic()
            if now >= next_remote_probe_at:
                next_remote_probe_at = now + 1.0
                failure = controller._remote_start_failure_detail(
                    remote_root, launch_log
                )
                if failure is not None:
                    last_error = failure
                    break
            time.sleep(0.25)
            continue
        if usb.negotiated_speed_mbps < controller.configuration.minimum_usb_speed_mbps:
            last_error = "FunctionFS USB speed is below minimum"
            usb = None
            time.sleep(0.25)
            continue
        try:
            controller._validate_live_transport_identity(
                transport, usb, phone_kernel_release, phone_usb_controller
            )
        except PhysicalAdapterError as error:
            last_error = str(error)
            usb = None
        break
    return usb, last_error


def _stop_phone_session_launcher(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _fail_phone_session_start(
    controller,
    last_error: str,
    remote_root: str,
    launch_log: str,
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    artifact_sha256: str,
) -> None:
    failure = controller._remote_start_failure_detail(remote_root, launch_log)
    if failure is not None:
        last_error = failure
    try:
        verify_android_usb_restored(
            serial=controller.configuration.serial,
            adb_port=controller.configuration.adb_port,
            minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
            timeout_s=15,
        )
        cleanup_note = ""
    except PhysicalAdapterError:
        try:
            controller._close_direct_usb(execution, transport, artifact_sha256)
        except PhysicalAdapterError as cleanup_error:
            cleanup_note = str(cleanup_error)
        else:
            cleanup_note = ""
        try:
            verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
                timeout_s=90,
            )
        except PhysicalAdapterError as restoration_error:
            cleanup_note = (
                cleanup_note + "; " if cleanup_note else ""
            ) + str(restoration_error)
    if cleanup_note:
        last_error += "; cleanup: " + cleanup_note
    raise PhysicalAdapterError(last_error)


def _connect_start_diagnostic(
    controller,
    usb: FunctionFsUsbObservation,
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    artifact_sha256: str,
    control_check: Callable[[], None],
) -> str | None:
    try:
        control_check()
        return controller._connect_diagnostic_ncm(usb)
    except BaseException as error:
        try:
            controller._close_direct_usb(execution, transport, artifact_sha256)
        except PhysicalAdapterError as cleanup_error:
            error.add_note("direct NCM cleanup failed: " + str(cleanup_error))
        try:
            verify_android_usb_restored(
                serial=controller.configuration.serial,
                adb_port=controller.configuration.adb_port,
                minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
                timeout_s=180,
            )
        except PhysicalAdapterError as restoration_error:
            error.add_note(
                "direct NCM restoration failed: " + str(restoration_error)
            )
        raise


def _publish_phone_session_launch(
    controller,
    command: PhysicalTransitionCommand,
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    usb: FunctionFsUsbObservation,
    remote_hashes: Mapping[str, str],
    phone_kernel_release: str,
    diagnostic_interface: str | None,
    shards: tuple[RuntimePhoneShard, ...],
    multi_session: bool,
    shard_manifest_sha256: str | None,
    token: str,
    remote_root: str,
) -> DirectPhoneFfnLaunchReceipt:
    launch = DirectPhoneFfnLaunchReceipt(
        ticket_id=command.ticket_id,
        artifact_sha256=command.artifact_sha256,
        session_id=token,
        execution=execution,
        transport=transport,
        usb=usb,
        remote_hashes=remote_hashes,
        usb_close_sha256=controller._sha256(controller.configuration.usb_close_path),
        phone_kernel_release=phone_kernel_release,
        cpu_affinity=getattr(controller.configuration, "cpu_affinity", None),
        diagnostic_interface=diagnostic_interface,
        phone_shards=shards if multi_session else (),
        shard_manifest_sha256=shard_manifest_sha256,
        weight_sources=(
            controller._phone_weight_sources(shards)
            if multi_session else ()
        ),
    )
    controller._launch = launch
    controller._remote_root = remote_root
    controller._bound_ticket_ids = [command.ticket_id]
    controller._proof_shards = list(shards if multi_session else ())
    controller._executed_proof_shards = []
    controller._residency_generation = 1
    controller._shard_residency_windows = [
        _ShardResidencyWindow(shard=row, first_generation=1)
        for row in controller._proof_shards
    ]
    controller._ticket_bind_generations = {command.ticket_id: 1}
    controller._historical_execution_proofs = []
    controller._execution_by_artifact = {command.artifact_sha256: execution}
    controller._transport_by_artifact = {command.artifact_sha256: transport}
    controller._load_count_by_session = {row.session_id: 1 for row in shards}
    column_quantum = command.adapter_parameters.get(
        "ffn_column_quantum"
    )
    if type(column_quantum) is not int or column_quantum <= 0:
        raise PhysicalAdapterError(
            "direct phone FFN column quantum is invalid"
        )
    controller._column_quantum_by_session = {
        row.session_id: column_quantum for row in shards
    }
    controller._max_tokens_by_session = {
        row.session_id: execution.max_tokens for row in shards
    }
    return launch


def start(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
    control_check: Callable[[], None] | None = None,
) -> DirectPhoneFfnLaunchReceipt:
    (
        control_check,
        phone_kernel_release,
        execution,
        shards,
        multi_session,
        shard_manifest,
        shard_manifest_sha256,
        worker,
        model,
        backend,
    ) = controller._start_contract(
        command, manifest, transport, control_check
    )
    remote_hashes, phone_usb_controller = controller._start_remote_hashes(
        manifest,
        shards,
        worker,
        model,
        multi_session,
        control_check,
    )
    token, remote_root, launch_log, remote_command = (
        controller._start_remote_command(
            command,
            manifest,
            execution,
            transport,
            shards,
            multi_session,
            shard_manifest,
            worker,
            model,
            backend,
        )
    )
    process = controller._spawn_phone_session(remote_command, control_check)
    usb, last_error = controller._wait_for_phone_session(
        process,
        transport,
        shards,
        phone_kernel_release,
        phone_usb_controller,
        remote_root,
        launch_log,
        control_check,
    )
    controller._stop_phone_session_launcher(process)
    if usb is None:
        controller._fail_phone_session_start(
            last_error,
            remote_root,
            launch_log,
            execution,
            transport,
            command.artifact_sha256,
        )
        raise AssertionError("unreachable")
    diagnostic_interface = controller._connect_start_diagnostic(
        usb,
        execution,
        transport,
        command.artifact_sha256,
        control_check,
    )
    return controller._publish_phone_session_launch(
        command,
        execution,
        transport,
        usb,
        remote_hashes,
        phone_kernel_release,
        diagnostic_interface,
        shards,
        multi_session,
        shard_manifest_sha256,
        token,
        remote_root,
    )
