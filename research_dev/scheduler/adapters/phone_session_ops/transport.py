"""DirectPhoneFfnSession transport operations on its existing owner."""

from __future__ import annotations

import hashlib
import http.client
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from types import MappingProxyType
from typing import Mapping

from ..bridge import FunctionFsUsbObservation
from ..contracts import PhysicalAdapterError
from ..llama_server import PhoneFfnExecutionContract
from ..phone_transport import PhoneTransportContract
from ..remote_hash_cache import cached_remote_hashes, update_remote_hash_cache
from ..phone_session_contracts.events import (
    PhoneResidencyPhaseEvent,
    parse_phone_residency_phase_events,
    PhoneResidencyCallEvent,
    parse_phone_residency_call_events,
)
from .common import _MINIMUM_PERSISTENT_HASH_CACHE_BYTES


def _adb(controller, remote_command: str, timeout_s: int = 30) -> str:
    try:
        result = subprocess.run(
            [
                str(controller.configuration.adb_path),
                "-P",
                str(controller.configuration.adb_port),
                "-s",
                controller.configuration.serial,
                "shell",
                remote_command,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="ascii",
            errors="backslashreplace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise PhysicalAdapterError(
            "phone session ADB command timed out"
        ) from error
    if result.returncode != 0:
        raise PhysicalAdapterError(
            "phone session ADB command failed: "
            + result.stderr.strip()
        )
    return result.stdout


def _adb_root(controller, remote_command: str, timeout_s: int = 30) -> str:
    return controller._adb(
        "su -c " + shlex.quote(remote_command),
        timeout_s=timeout_s,
    )


def _remote_hashes(
    controller,
    paths: Mapping[str, str],
    *,
    root: bool = False,
    timeout_s: int = 30,
) -> Mapping[str, str]:
    unique_paths = set(paths.values())
    missing = sorted(
        unique_paths - set(controller._verified_remote_hash_by_path)
    )
    identities = None
    cache_path = controller.configuration.remote_hash_cache_path
    if missing and cache_path is not None:
        identities = controller._remote_file_identities(
            tuple(missing),
            root=root,
            timeout_s=timeout_s,
        )
        controller._verified_remote_hash_by_path.update(
            cached_remote_hashes(
                cache_path,
                serial=controller.configuration.serial,
                identities=MappingProxyType({
                    path: identity
                    for path, identity in identities.items()
                    if identity.size_bytes
                        >= _MINIMUM_PERSISTENT_HASH_CACHE_BYTES
                }),
            )
        )
        missing = sorted(
            unique_paths - set(controller._verified_remote_hash_by_path)
        )
    if missing:
        command = "sha256sum " + " ".join(
            shlex.quote(path) for path in missing
        )
        output = (
            controller._adb_root(command, timeout_s=timeout_s)
            if root else controller._adb(command, timeout_s=timeout_s)
        )
        by_path = {}
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 2 and len(fields[0]) == 64:
                by_path[fields[1]] = "sha256:" + fields[0]
        if set(by_path) != set(missing):
            raise PhysicalAdapterError(
                "phone session hashes are incomplete"
            )
        controller._verified_remote_hash_by_path.update(by_path)
        if cache_path is not None and identities is not None:
            cacheable_hashes = {
                path: sha256
                for path, sha256 in by_path.items()
                if identities[path].size_bytes
                    >= _MINIMUM_PERSISTENT_HASH_CACHE_BYTES
            }
            try:
                if cacheable_hashes:
                    update_remote_hash_cache(
                        cache_path,
                        serial=controller.configuration.serial,
                        identities=MappingProxyType({
                            path: identities[path]
                            for path in cacheable_hashes
                        }),
                        hashes=cacheable_hashes,
                    )
            except (OSError, ValueError):
                pass
    if unique_paths - set(controller._verified_remote_hash_by_path):
        raise PhysicalAdapterError("phone session hashes are incomplete")
    return MappingProxyType({
        name: controller._verified_remote_hash_by_path[path]
        for name, path in sorted(paths.items())
    })


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _close_direct_usb(
    controller,
    execution: PhoneFfnExecutionContract,
    transport: PhoneTransportContract,
    artifact_sha256: str = "sha256:" + "0" * 64,
) -> None:
    flags = 1 | (2 if execution.activation == "swiglu" else 0)
    try:
        result = subprocess.run(
            [
                str(controller.configuration.usb_close_path),
                hex(transport.vendor_id),
                hex(transport.product_id),
                hex(execution.layer_mask),
                str(execution.n_embd),
                str(execution.columns),
                str(execution.max_tokens),
                str(flags),
                str(transport.usbfs_available_bytes),
                transport.generation,
                artifact_sha256,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="ascii",
            errors="backslashreplace",
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        raise PhysicalAdapterError(
            "direct phone USB close timed out"
        ) from error
    if (
        result.returncode != 0
        or "FFNUSB_CLOSE status=ok" not in result.stdout
    ):
        raise PhysicalAdapterError(
            "direct phone USB close failed: "
            + (result.stderr.strip() or result.stdout.strip())
        )


def _remote_launch_failure(controller, remote_root: str) -> str | None:
    terminal_status = remote_root + "/terminal.status"
    worker_pid = remote_root + "/worker.pid"
    command = (
        "if [ -f " + shlex.quote(terminal_status) + " ]; then "
        "printf 'TERMINAL '; cat " + shlex.quote(terminal_status) + "; "
        "elif [ -f " + shlex.quote(worker_pid) + " ]; then "
        "worker_pid=$(cat " + shlex.quote(worker_pid) + "); "
        "if kill -0 \"$worker_pid\" 2>/dev/null; then "
        "printf 'RUNNING\\n'; else printf 'WORKER_EXITED\\n'; fi; "
        "else printf 'STARTING\\n'; fi"
    )
    output = controller._adb_root(command, timeout_s=5)
    lines = output.splitlines()
    if not lines or lines[0] in {"RUNNING", "STARTING"}:
        return None
    if lines[0].startswith("TERMINAL "):
        status = lines[0].removeprefix("TERMINAL ").strip()
        return "phone session terminated before USB enumeration: status=" + status
    if lines[0] == "WORKER_EXITED":
        return "phone worker exited before USB enumeration"
    return "phone session launch state is invalid"


def _phone_kernel_release(controller) -> str:
    release = controller._adb("uname -r").strip()
    if (
        not release
        or not release.isascii()
        or release != controller.configuration.required_kernel_release
    ):
        raise PhysicalAdapterError(
            "phone kernel is not qualified for direct DMA-BUF: "
            + (release or "absent")
        )
    return release


def _validate_static_transport_identity(
    controller,
    remote_hashes: Mapping[str, str],
) -> None:
    identity = controller.configuration.transport_qualification_identity
    if identity is None:
        return
    software = identity.software_identity
    host_path = controller.configuration.transport_host_binary_path
    assert host_path is not None
    if (
        controller._sha256(host_path) != software["host_binary_sha256"]
        or any(
            controller._sha256(path)
                != software["host_dependency_sha256:" + name]
            for name, path in (
                controller.configuration
                    .transport_host_dependency_paths.items()
            )
        )
        or remote_hashes.get("session_script")
            != software["phone_session_sha256"]
        or any(
            value != software["phone_worker_sha256"]
            for name, value in remote_hashes.items()
            if name == "worker" or name.startswith("worker:")
        )
        or controller.configuration.phone_boot_image_sha256
            != identity.hardware_identity["phone_boot_image_sha256"]
        or controller.configuration.serial
            != identity.hardware_identity["phone_usb_serial"]
    ):
        raise PhysicalAdapterError(
            "phone transport qualification software identity differs"
        )
    if "resident_workers" in remote_hashes or "resident_router" in (
        remote_hashes
    ):
        if (
            software.get("phone_resident_workers_sha256")
                != remote_hashes.get("resident_workers")
            or software.get("phone_resident_router_sha256")
                != remote_hashes.get("resident_router")
        ):
            raise PhysicalAdapterError(
                "multi-session phone transport is not qualified"
            )


def _validate_live_transport_identity(
    controller,
    transport: PhoneTransportContract,
    usb: FunctionFsUsbObservation,
    phone_kernel_release: str,
    phone_usb_controller: str,
) -> None:
    identity = controller.configuration.transport_qualification_identity
    if transport.qualification_identity_sha256 is None:
        if identity is not None:
            raise PhysicalAdapterError(
                "scheduler ticket lacks transport qualification identity"
            )
        return
    if (
        identity is None
        or transport.qualification_identity_sha256
            != identity.identity_sha256
    ):
        raise PhysicalAdapterError(
            "scheduler transport qualification identity differs"
        )
    hardware = identity.hardware_identity
    functionfs_identity = usb.vendor_id + ":" + usb.product_id
    if (
        phone_kernel_release != hardware["phone_kernel_release"]
        or phone_usb_controller != hardware["phone_usb_controller"]
        or usb.sysfs_device != hardware["phone_usb_sysfs_device"]
        or functionfs_identity != hardware["functionfs_identity"]
        or usb.negotiated_speed_mbps
            < identity.minimum_usb_speed_mbps
        or transport.allocator not in identity.qualified_allocators
        or transport.generation != identity.transport_generation
    ):
        raise PhysicalAdapterError(
            "live FunctionFS transport identity differs"
        )


def _ncm_interfaces(sysfs_device: str) -> tuple[str, ...]:
    result = []
    for interface in Path("/sys/class/net").iterdir():
        device = interface / "device"
        if not device.exists():
            continue
        target = os.path.realpath(device)
        if "/" + sysfs_device + ":" in target:
            result.append(interface.name)
    return tuple(sorted(result))


def _diagnostic_available(controller) -> bool:
    host = controller.configuration.diagnostic_host
    if host is None or not controller.configuration.diagnostic_port:
        return True
    connection = http.client.HTTPConnection(
        host, controller.configuration.diagnostic_port, timeout=1
    )
    try:
        connection.request("GET", "/power.json")
        response = connection.getresponse()
        response.read()
        return response.status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()


def _read_diagnostic_file(controller, name: str) -> str:
    host = controller.configuration.diagnostic_host
    if (
        host is None
        or not controller.configuration.diagnostic_port
        or type(name) is not str
        or not re.fullmatch(r"[A-Za-z0-9._-]+", name)
    ):
        raise PhysicalAdapterError(
            "direct phone diagnostic file is unavailable"
        )
    connection = http.client.HTTPConnection(
        host, controller.configuration.diagnostic_port, timeout=5
    )
    try:
        connection.request("GET", "/" + name)
        response = connection.getresponse()
        payload = response.read((16 << 20) + 1)
        if response.status != 200 or len(payload) > (16 << 20):
            raise PhysicalAdapterError(
                "direct phone diagnostic file is invalid"
            )
        return payload.decode("ascii", "strict")
    except (OSError, UnicodeDecodeError, http.client.HTTPException) as error:
        raise PhysicalAdapterError(
            "direct phone diagnostic file read failed"
        ) from error
    finally:
        connection.close()


def residency_phase_events(
    controller,
) -> tuple[PhoneResidencyPhaseEvent, ...]:
    if controller._launch is None or controller._remote_root is None:
        raise PhysicalAdapterError(
            "direct phone session is not active"
        )
    return parse_phone_residency_phase_events(
        controller._read_diagnostic_file("residency.log").splitlines()
    )


def residency_call_events(
    controller,
) -> tuple[PhoneResidencyCallEvent, ...]:
    if controller._launch is None or controller._remote_root is None:
        raise PhysicalAdapterError(
            "direct phone session is not active"
        )
    return parse_phone_residency_call_events(
        controller._read_diagnostic_file("router.log").splitlines()
    )


def _connect_diagnostic_ncm(
    controller, usb: FunctionFsUsbObservation
) -> str | None:
    manager = controller.configuration.network_manager_path
    if manager is None:
        return None
    deadline = time.monotonic() + controller.configuration.launch_timeout_s
    attempted: set[str] = set()
    while time.monotonic() < deadline:
        for interface in controller._ncm_interfaces(usb.sysfs_device):
            if interface not in attempted:
                try:
                    subprocess.run(
                        [str(manager), "device", "connect", interface],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=15,
                    )
                except subprocess.TimeoutExpired:
                    pass
                attempted.add(interface)
            if controller._diagnostic_available():
                return interface
        time.sleep(0.25)
    raise PhysicalAdapterError(
        "direct phone diagnostic NCM is unavailable"
    )
