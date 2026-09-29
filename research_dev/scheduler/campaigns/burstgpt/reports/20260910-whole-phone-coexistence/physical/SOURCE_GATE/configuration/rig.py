"""Typed scheduler configuration manifests: rig."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .common import (
    RIG_MANIFEST_SCHEMA,
    _SHA256,
    _integer,
    _object,
    _optional_text,
    _path,
    _path_map,
    _require,
    _sequence,
    _text,
    _text_map,
)


@dataclass(frozen=True)
class RigDeviceConfiguration:
    device_id: str
    kind: str
    memory_capacity_bytes: int

    @classmethod
    def from_json(cls, value: object) -> "RigDeviceConfiguration":
        row = _object(value, "rig device")
        return cls(
            device_id=_text(row.get("device_id"), "device id"),
            kind=_text(row.get("kind"), "device kind"),
            memory_capacity_bytes=_integer(
                row.get("memory_capacity_bytes"),
                "device memory capacity",
                minimum=1,
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "kind": self.kind,
            "memory_capacity_bytes": self.memory_capacity_bytes,
        }


@dataclass(frozen=True)
class RigResourceConfiguration:
    resource_id: str
    capacity: int
    identity: str

    @classmethod
    def from_json(cls, value: object) -> "RigResourceConfiguration":
        row = _object(value, "rig resource")
        return cls(
            resource_id=_text(row.get("resource_id"), "resource id"),
            capacity=_integer(row.get("capacity"), "resource capacity", minimum=1),
            identity=_text(row.get("identity"), "resource identity"),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "capacity": self.capacity,
            "identity": self.identity,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True)
class RigTopologyConfiguration:
    cpu_device_id: str
    gpu_device_id: str
    phone_device_id: str
    cpu_resource_id: str
    gpu_resource_id: str
    host_memory_resource_id: str
    gpu_memory_resource_id: str
    phone_memory_resource_id: str
    functionfs_resource_id: str
    phone_transport_resource_ids: tuple[str, ...]
    phone_compute_resource_ids: tuple[str, ...]
    gpu_exclusive_residency_resource_id: str
    phone_exclusive_residency_resource_id: str

    @classmethod
    def from_json(cls, value: object) -> "RigTopologyConfiguration":
        row = _object(value, "rig topology")

        def identifiers(name: str) -> tuple[str, ...]:
            result = tuple(
                _text(item, name)
                for item in _sequence(row.get(name), name)
            )
            _require(result and len(set(result)) == len(result), name)
            return result

        return cls(
            cpu_device_id=_text(row.get("cpu_device_id"), "CPU device id"),
            gpu_device_id=_text(row.get("gpu_device_id"), "GPU device id"),
            phone_device_id=_text(
                row.get("phone_device_id"), "phone device id"
            ),
            cpu_resource_id=_text(
                row.get("cpu_resource_id"), "CPU resource id"
            ),
            gpu_resource_id=_text(
                row.get("gpu_resource_id"), "GPU resource id"
            ),
            host_memory_resource_id=_text(
                row.get("host_memory_resource_id"),
                "host memory resource id",
            ),
            gpu_memory_resource_id=_text(
                row.get("gpu_memory_resource_id"),
                "GPU memory resource id",
            ),
            phone_memory_resource_id=_text(
                row.get("phone_memory_resource_id"),
                "phone memory resource id",
            ),
            functionfs_resource_id=_text(
                row.get("functionfs_resource_id"),
                "FunctionFS resource id",
            ),
            phone_transport_resource_ids=identifiers(
                "phone_transport_resource_ids"
            ),
            phone_compute_resource_ids=identifiers(
                "phone_compute_resource_ids"
            ),
            gpu_exclusive_residency_resource_id=_text(
                row.get("gpu_exclusive_residency_resource_id"),
                "GPU residency resource id",
            ),
            phone_exclusive_residency_resource_id=_text(
                row.get("phone_exclusive_residency_resource_id"),
                "phone residency resource id",
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "cpu_device_id": self.cpu_device_id,
            "cpu_resource_id": self.cpu_resource_id,
            "functionfs_resource_id": self.functionfs_resource_id,
            "gpu_device_id": self.gpu_device_id,
            "gpu_exclusive_residency_resource_id": (
                self.gpu_exclusive_residency_resource_id
            ),
            "gpu_memory_resource_id": self.gpu_memory_resource_id,
            "gpu_resource_id": self.gpu_resource_id,
            "host_memory_resource_id": self.host_memory_resource_id,
            "phone_compute_resource_ids": list(
                self.phone_compute_resource_ids
            ),
            "phone_device_id": self.phone_device_id,
            "phone_exclusive_residency_resource_id": (
                self.phone_exclusive_residency_resource_id
            ),
            "phone_memory_resource_id": self.phone_memory_resource_id,
            "phone_transport_resource_ids": list(
                self.phone_transport_resource_ids
            ),
        }


@dataclass(frozen=True)
class PhoneRigConfiguration:
    serial: str
    adb_port: int
    diagnostic_endpoint: str
    battery_ppm: int
    minimum_usb_speed_mbps: int
    kernel_release: str
    boot_image_sha256: str
    session_script: str
    restore_script: str
    worker_path: str
    resident_workers_path: str | None
    resident_router_path: str | None
    busybox_path: str
    session_root: str
    remote_hash_cache_path: Path
    android_gadget_path: str
    functionfs_gadget_path: str
    functionfs_root_path: str
    usb_controller: str
    multi_session_port_base: int | None
    whole_server_path: str
    whole_server_sha256: str
    whole_library_directory: str
    whole_state_directory: str
    whole_executable_device: str
    whole_forward_port: int
    whole_remote_port: int
    whole_control_transport: str = "adb-usb"
    whole_ncm_adb_endpoint: str | None = None

    @classmethod
    def from_json(
        cls, value: object, base: Path
    ) -> "PhoneRigConfiguration":
        row = _object(value, "phone rig")
        boot = _text(row.get("boot_image_sha256"), "phone boot image SHA-256")
        _require(_SHA256.fullmatch(boot) is not None, "phone boot image SHA-256")
        whole_server_sha256 = _text(
            row.get("whole_server_sha256"), "whole-phone server SHA-256"
        )
        _require(
            _SHA256.fullmatch(whole_server_sha256) is not None,
            "whole-phone server SHA-256",
        )
        resident_workers = _optional_text(
            row.get("resident_workers_path"), "resident workers path"
        )
        resident_router = _optional_text(
            row.get("resident_router_path"), "resident router path"
        )
        _require(
            (resident_workers is None) == (resident_router is None),
            "resident phone binaries must be supplied together",
        )
        port = row.get("multi_session_port_base")
        control_transport = row.get("whole_control_transport", "adb-usb")
        ncm_endpoint = _optional_text(row.get("whole_ncm_adb_endpoint"), "whole-phone NCM ADB endpoint")
        _require(control_transport in {"adb-usb", "adb-ncm"}
                 and (control_transport == "adb-ncm") == (ncm_endpoint is not None),
                 "whole-phone control transport")
        return cls(
            serial=_text(row.get("serial"), "phone serial"),
            adb_port=_integer(row.get("adb_port"), "ADB port", minimum=1, maximum=65535),
            diagnostic_endpoint=_text(
                row.get("diagnostic_endpoint"), "phone diagnostic endpoint"
            ),
            battery_ppm=_integer(
                row.get("battery_ppm"), "phone battery ppm", maximum=1_000_000
            ),
            minimum_usb_speed_mbps=_integer(
                row.get("minimum_usb_speed_mbps"),
                "minimum USB speed",
                minimum=1,
            ),
            kernel_release=_text(row.get("kernel_release"), "phone kernel release"),
            boot_image_sha256=(boot if boot.startswith("sha256:") else "sha256:" + boot),
            session_script=_text(row.get("session_script"), "phone session script"),
            restore_script=_text(row.get("restore_script"), "phone restore script"),
            worker_path=_text(row.get("worker_path"), "phone worker path"),
            resident_workers_path=resident_workers,
            resident_router_path=resident_router,
            busybox_path=_text(row.get("busybox_path"), "phone busybox path"),
            session_root=_text(row.get("session_root"), "phone session root"),
            remote_hash_cache_path=_path(
                row.get("remote_hash_cache_path"), base, "phone remote hash cache"
            ),
            android_gadget_path=_text(
                row.get("android_gadget_path"), "Android gadget path"
            ),
            functionfs_gadget_path=_text(
                row.get("functionfs_gadget_path"), "FunctionFS gadget path"
            ),
            functionfs_root_path=_text(
                row.get("functionfs_root_path"), "FunctionFS root path"
            ),
            usb_controller=_text(row.get("usb_controller"), "phone USB controller"),
            multi_session_port_base=(
                None
                if port is None
                else _integer(port, "multi-session port", minimum=1, maximum=65535)
            ),
            whole_server_path=_text(row.get("whole_server_path"), "whole-phone server"),
            whole_server_sha256=(
                whole_server_sha256
                if whole_server_sha256.startswith("sha256:")
                else "sha256:" + whole_server_sha256
            ),
            whole_library_directory=_text(
                row.get("whole_library_directory"), "whole-phone library directory"
            ),
            whole_state_directory=_text(
                row.get("whole_state_directory"), "whole-phone state directory"
            ),
            whole_executable_device=_text(
                row.get("whole_executable_device"), "whole-phone executable device"
            ),
            whole_forward_port=_integer(
                row.get("whole_forward_port"), "whole-phone forward port", minimum=1, maximum=65535
            ),
            whole_remote_port=_integer(
                row.get("whole_remote_port"), "whole-phone remote port", minimum=1, maximum=65535
            ),
            whole_control_transport=control_transport,
            whole_ncm_adb_endpoint=ncm_endpoint,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "adb_port": self.adb_port,
            "android_gadget_path": self.android_gadget_path,
            "battery_ppm": self.battery_ppm,
            "boot_image_sha256": self.boot_image_sha256,
            "busybox_path": self.busybox_path,
            "diagnostic_endpoint": self.diagnostic_endpoint,
            "functionfs_gadget_path": self.functionfs_gadget_path,
            "functionfs_root_path": self.functionfs_root_path,
            "kernel_release": self.kernel_release,
            "minimum_usb_speed_mbps": self.minimum_usb_speed_mbps,
            "multi_session_port_base": self.multi_session_port_base,
            "remote_hash_cache_path": str(self.remote_hash_cache_path),
            "resident_router_path": self.resident_router_path,
            "resident_workers_path": self.resident_workers_path,
            "restore_script": self.restore_script,
            "serial": self.serial,
            "session_root": self.session_root,
            "session_script": self.session_script,
            "usb_controller": self.usb_controller,
            "whole_executable_device": self.whole_executable_device,
            "whole_forward_port": self.whole_forward_port,
            "whole_library_directory": self.whole_library_directory,
            "whole_remote_port": self.whole_remote_port,
            "whole_server_path": self.whole_server_path,
            "whole_server_sha256": self.whole_server_sha256,
            "whole_state_directory": self.whole_state_directory,
            **({"whole_control_transport": self.whole_control_transport,
                "whole_ncm_adb_endpoint": self.whole_ncm_adb_endpoint}
               if self.whole_control_transport != "adb-usb" else {}),
            "worker_path": self.worker_path,
        }


@dataclass(frozen=True)
class RigManifest:
    rig_id: str
    repo_root: Path
    devices: tuple[RigDeviceConfiguration, ...]
    resources: tuple[RigResourceConfiguration, ...]
    topology: RigTopologyConfiguration
    endpoints: Mapping[str, str]
    binaries: Mapping[str, Path]
    library_directories: Mapping[str, Path]
    transport_host_dependencies: Mapping[str, Path]
    phone: PhoneRigConfiguration

    @classmethod
    def from_json(cls, value: object, base: Path) -> "RigManifest":
        row = _object(value, "rig manifest")
        _require(row.get("schema") == RIG_MANIFEST_SCHEMA, "rig manifest schema")
        devices = tuple(
            RigDeviceConfiguration.from_json(item)
            for item in _sequence(row.get("devices"), "rig devices")
        )
        resources = tuple(
            RigResourceConfiguration.from_json(item)
            for item in _sequence(row.get("resources"), "rig resources")
        )
        _require(
            devices
            and len({item.device_id for item in devices}) == len(devices),
            "rig device identities are not unique",
        )
        _require(
            resources
            and len({item.resource_id for item in resources}) == len(resources),
            "rig resource identities are not unique",
        )
        topology = RigTopologyConfiguration.from_json(row.get("topology"))
        device_ids = {item.device_id for item in devices}
        resource_ids = {item.resource_id for item in resources}
        _require(
            {
                topology.cpu_device_id,
                topology.gpu_device_id,
                topology.phone_device_id,
            } <= device_ids,
            "rig topology references an absent device",
        )
        _require(
            {
                topology.cpu_resource_id,
                topology.gpu_resource_id,
                topology.functionfs_resource_id,
                topology.gpu_exclusive_residency_resource_id,
                topology.phone_exclusive_residency_resource_id,
                *topology.phone_transport_resource_ids,
                *topology.phone_compute_resource_ids,
            } <= resource_ids,
            "rig topology references an absent resource",
        )
        return cls(
            rig_id=_text(row.get("rig_id"), "rig id"),
            repo_root=_path(row.get("repo_root"), base, "repository root"),
            devices=devices,
            resources=resources,
            topology=topology,
            endpoints=_text_map(row.get("endpoints"), "rig endpoints"),
            binaries=_path_map(row.get("binaries"), base, "rig binaries"),
            library_directories=_path_map(
                row.get("library_directories"), base, "library directories"
            ),
            transport_host_dependencies=_path_map(
                row.get("transport_host_dependencies"),
                base,
                "transport host dependencies",
            ),
            phone=PhoneRigConfiguration.from_json(row.get("phone"), base),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "binaries": {key: str(value) for key, value in self.binaries.items()},
            "devices": [row.to_json() for row in self.devices],
            "endpoints": dict(self.endpoints),
            "library_directories": {
                key: str(value) for key, value in self.library_directories.items()
            },
            "phone": self.phone.to_json(),
            "repo_root": str(self.repo_root),
            "resources": [row.to_json() for row in self.resources],
            "rig_id": self.rig_id,
            "schema": RIG_MANIFEST_SCHEMA,
            "topology": self.topology.to_json(),
            "transport_host_dependencies": {
                key: str(value)
                for key, value in self.transport_host_dependencies.items()
            },
        }

    @property
    def device_by_kind(self) -> Mapping[str, RigDeviceConfiguration]:
        result = {row.kind: row for row in self.devices}
        _require(len(result) == len(self.devices), "rig device kinds are ambiguous")
        return MappingProxyType(result)
