"""DirectPhoneFfnSession preflight operations on its existing owner."""

from __future__ import annotations

import re
import shlex
from typing import Callable, Iterable, Mapping

from ..._internal.model_manifest import ModelManifest
from ..._internal.runtime_plan import RuntimePhoneShard
from ..bridge import verify_android_usb_restored
from ..contracts import PhysicalAdapterError
from ..llama_server import (
    PhoneFfnExecutionContract,
    phone_ffn_resident_contract,
    primary_phone_ffn_contract,
)
from ..ffn_shards import (
    FfnShardIndexError,
    remote_hash_entries as ffn_shard_hash_entries,
    verify_remote_hashes as verify_ffn_shard_hashes,
)
from ..phone_transport import PhoneTransportContract
from ..ticket import PhysicalTransitionCommand
from ..phone_session_contracts.receipts import DirectPhoneFfnPreflightReceipt


def preflight(controller) -> DirectPhoneFfnPreflightReceipt:
    phone_kernel_release = controller._phone_kernel_release()
    paths = {
        "restore_script": controller.configuration.restore_script,
        "session_script": controller.configuration.session_script,
    }
    paths.update({
        "model:" + artifact: path
        for artifact, path in (
            controller.configuration.model_paths_by_artifact.items()
        )
    })
    paths.update({
        "worker:" + artifact: path
        for artifact, path in (
            controller.configuration.worker_paths_by_artifact.items()
        )
    })
    paths.update(
        ffn_shard_hash_entries(getattr(controller.configuration, "ffn_shards_by_artifact", {}))
    )
    if controller.configuration.busybox_path is not None:
        paths["busybox"] = controller.configuration.busybox_path
    if controller.configuration.resident_workers_path is not None:
        assert controller.configuration.resident_router_path is not None
        paths["resident_workers"] = (
            controller.configuration.resident_workers_path
        )
        paths["resident_router"] = controller.configuration.resident_router_path
    executable_paths = {
        controller.configuration.session_script,
        controller.configuration.restore_script,
        *controller.configuration.worker_paths_by_artifact.values(),
    }
    if controller.configuration.busybox_path is not None:
        executable_paths.add(controller.configuration.busybox_path)
    if controller.configuration.resident_workers_path is not None:
        assert controller.configuration.resident_router_path is not None
        executable_paths.add(controller.configuration.resident_workers_path)
        executable_paths.add(controller.configuration.resident_router_path)
    executable_checks = " && ".join(
        "test -x " + shlex.quote(path)
        for path in sorted(executable_paths)
    )
    controller._adb_root(
        executable_checks + " && echo DIRECT_PHONE_PREFLIGHT_OK"
    )
    controller._validate_worker_loadability(
        controller.configuration.worker_paths_by_artifact.values()
    )
    remote_hashes = controller._remote_hashes(
        paths,
        root=True,
        timeout_s=controller.configuration.launch_timeout_s,
    )
    controller._validate_static_transport_identity(remote_hashes)
    for artifact in controller.configuration.model_paths_by_artifact:
        if remote_hashes["model:" + artifact] != artifact:
            raise PhysicalAdapterError(
                "direct phone model artifact differs from registration"
            )
    try:
        verify_ffn_shard_hashes(
            getattr(controller.configuration, "ffn_shards_by_artifact", {}), remote_hashes
        )
    except FfnShardIndexError as error:
        raise PhysicalAdapterError(str(error)) from error
    restoration = verify_android_usb_restored(
        serial=controller.configuration.serial,
        adb_port=controller.configuration.adb_port,
        minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
    )
    return DirectPhoneFfnPreflightReceipt(
        remote_hashes=remote_hashes,
        usb_close_sha256=controller._sha256(
            controller.configuration.usb_close_path
        ),
        restoration=restoration,
        phone_kernel_release=phone_kernel_release,
        ffn_shard_indexes={
            artifact: index.index_sha256
            for artifact, index in controller.configuration
                .ffn_shards_by_artifact.items()
        },
    )


def _validate_worker_loadability(
    controller, workers: Iterable[str]
) -> None:
    paths = tuple(sorted(set(workers)))
    for worker in paths:
        worker_root = worker.rpartition("/")[0]
        command = (
            "worker_output=$(LD_LIBRARY_PATH="
            + shlex.quote(worker_root)
            + " ADSP_LIBRARY_PATH="
            + shlex.quote(worker_root)
            + " "
            + shlex.quote(worker)
            + " 2>&1); worker_status=$?; "
            + "if [ \"$worker_status\" -ne 2 ]; then "
            + "printf '%s\\n' \"$worker_output\" >&2; exit 1; fi; "
            + "case \"$worker_output\" in usage:*) ;; "
            + "*) printf '%s\\n' \"$worker_output\" >&2; "
            + "exit 1;; esac"
        )
        try:
            controller._adb_root(command)
        except PhysicalAdapterError as error:
            raise PhysicalAdapterError(
                "direct phone worker is not loadable: " + str(error)
            ) from error


def _start_contract(
    controller,
    command: PhysicalTransitionCommand,
    manifest: ModelManifest,
    transport: PhoneTransportContract,
    control_check: Callable[[], None] | None,
) -> tuple[
    Callable[[], None],
    str,
    PhoneFfnExecutionContract,
    tuple[RuntimePhoneShard, ...],
    bool,
    str | None,
    str | None,
    str,
    str,
    str,
]:
    if control_check is None:
        control_check = lambda: None
    if not callable(control_check):
        raise PhysicalAdapterError(
            "direct phone session control check is invalid"
        )
    control_check()
    if controller._launch is not None or transport.transport != "functionfs-usb":
        raise PhysicalAdapterError("direct phone session is already active")
    phone_kernel_release = controller._phone_kernel_release()
    execution = primary_phone_ffn_contract(
        command, phone_ffn_resident_contract(command, manifest)
    )
    shards = (
        command.transition.phone_shards
        or command.execution_contract.phone_shards
    )
    multi_session = bool(shards)
    shard_manifest = None
    shard_manifest_sha256 = None
    if multi_session:
        if (
            controller.configuration.resident_workers_path is None
            or controller.configuration.resident_router_path is None
        ):
            raise PhysicalAdapterError(
                "multi-session phone execution is not configured"
            )
        shard_manifest, shard_manifest_sha256 = (
            controller._multi_session_manifest(shards)
        )
    try:
        worker = controller.configuration.worker_paths_by_artifact[
            manifest.artifact_sha256
        ]
        model = controller.configuration.model_paths_by_artifact[
            manifest.artifact_sha256
        ]
        backend = (
            shards[0].session_id
            if shards
            else controller.configuration.backend_by_device[execution.device_id]
        )
    except KeyError as error:
        raise PhysicalAdapterError(
            "direct phone execution artifact is not deployed"
        ) from error
    if (
        type(backend) is not str
        or not re.fullmatch(r"[A-Za-z0-9._-]+", backend)
    ):
        raise PhysicalAdapterError(
            "direct phone session backend is not executable"
        )
    verify_android_usb_restored(
        serial=controller.configuration.serial,
        adb_port=controller.configuration.adb_port,
        minimum_speed_mbps=controller.configuration.minimum_usb_speed_mbps,
    )
    control_check()
    return (
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
    )


def _start_remote_hashes(
    controller,
    manifest: ModelManifest,
    shards: tuple[RuntimePhoneShard, ...],
    worker: str,
    model: str,
    multi_session: bool,
    control_check: Callable[[], None],
) -> tuple[Mapping[str, str], str]:
    remote_paths = {
        "model": model,
        "restore_script": controller.configuration.restore_script,
        "session_script": controller.configuration.session_script,
        "worker": worker,
    }
    if multi_session:
        assert controller.configuration.resident_workers_path is not None
        assert controller.configuration.resident_router_path is not None
        remote_paths["resident_workers"] = (
            controller.configuration.resident_workers_path
        )
        remote_paths["resident_router"] = controller.configuration.resident_router_path
    remote_paths.update({
        "model:" + str(shard.artifact_sha256): (
            controller.configuration.model_paths_by_artifact[
                str(shard.artifact_sha256)
            ]
        )
        for shard in shards
        if shard.artifact_sha256 is not None
    })
    controller._phone_weight_sources(shards)
    ffn_shard_records = tuple(
        record
        for shard in shards
        if shard.artifact_sha256 is not None
        for index in (
            controller.configuration.ffn_shards_by_artifact.get(
                shard.artifact_sha256
            ),
        )
        if index is not None
        for record in (
            index.resolve(
                shard.artifact_sha256,
                shard.layer_mask,
                shard.maximum_columns,
            ),
        )
        if record is not None
    )
    remote_paths.update({
        "ffn-shard:" + record.shard_sha256: record.remote_path
        for record in ffn_shard_records
    })
    remote_hashes = controller._remote_hashes(remote_paths)
    control_check()
    controller._validate_static_transport_identity(remote_hashes)
    for shard in shards:
        if (
            shard.artifact_sha256 is None
            or remote_hashes.get("model:" + shard.artifact_sha256)
                != shard.artifact_sha256
        ):
            raise PhysicalAdapterError(
                "multi-session phone model artifact differs"
            )
    try:
        verify_ffn_shard_hashes(
            getattr(controller.configuration, "ffn_shards_by_artifact", {}),
            remote_hashes,
            records=ffn_shard_records,
        )
    except FfnShardIndexError as error:
        raise PhysicalAdapterError(str(error)) from error
    return remote_hashes, controller._adb("getprop sys.usb.controller").strip()
