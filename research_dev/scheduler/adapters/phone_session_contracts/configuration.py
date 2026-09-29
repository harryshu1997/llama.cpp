"""Phone FFN configuration, physical events and receipts: configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..contracts import PhysicalAdapterError
from ..ffn_shards import FfnShardIndex
from ..transport_profiles import TransportQualificationIdentity
from .common import _android_path, _artifact_mapping


@dataclass(frozen=True)
class DirectPhoneFfnSessionConfiguration:
    adb_path: Path
    usb_close_path: Path
    serial: str
    adb_port: int
    session_script: str
    restore_script: str
    session_root: str
    worker_paths_by_artifact: Mapping[str, str]
    model_paths_by_artifact: Mapping[str, str]
    backend_by_device: Mapping[str, str]
    minimum_usb_speed_mbps: int
    required_kernel_release: str
    launch_timeout_s: int = 240
    session_timeout_s: int = 7200
    max_requests: int = 0
    diagnostic_port: int = 0
    diagnostic_host: str | None = None
    busybox_path: str | None = None
    network_manager_path: Path | None = None
    transport_qualification_identity: (
        TransportQualificationIdentity | None
    ) = None
    transport_host_binary_path: Path | None = None
    transport_host_dependency_paths: Mapping[str, Path] = field(
        default_factory=dict
    )
    phone_boot_image_sha256: str | None = None
    android_gadget_path: str | None = None
    functionfs_gadget_path: str | None = None
    functionfs_root_path: str | None = None
    phone_usb_controller: str | None = None
    remote_hash_cache_path: Path | None = None
    resident_workers_path: str | None = None
    resident_router_path: str | None = None
    # Optional offline FFN shard indexes (adapters/ffn_shards.py) keyed by
    # artifact sha256; when a stored shard covers an authorized session the
    # session opens the shard file instead of the complete model GGUF.
    ffn_shards_by_artifact: Mapping[str, FfnShardIndex] = field(
        default_factory=dict
    )
    multi_session_port_base: int | None = None
    multi_session_device_count: int | None = None
    cpu_affinity: str | None = None

    def __post_init__(self) -> None:
        if self.cpu_affinity is not None and (
            type(self.cpu_affinity) is not str
            or not 1 <= len(self.cpu_affinity) <= 16
            or any(value not in "0123456789abcdef" for value in self.cpu_affinity)
            or int(self.cpu_affinity, 16) == 0
        ):
            raise PhysicalAdapterError("phone CPU affinity is invalid")
        if (
            not isinstance(self.adb_path, Path)
            or not self.adb_path.is_file()
            or not os.access(self.adb_path, os.X_OK)
            or not isinstance(self.usb_close_path, Path)
            or not self.usb_close_path.is_file()
            or not os.access(self.usb_close_path, os.X_OK)
            or not self.serial
            or not self.serial.isascii()
            or type(self.adb_port) is not int
            or not 0 < self.adb_port <= 65535
            or type(self.minimum_usb_speed_mbps) is not int
            or self.minimum_usb_speed_mbps <= 0
            or type(self.required_kernel_release) is not str
            or not self.required_kernel_release
            or not self.required_kernel_release.isascii()
            or type(self.launch_timeout_s) is not int
            or self.launch_timeout_s <= 0
            or type(self.session_timeout_s) is not int
            or self.session_timeout_s <= 0
            or type(self.max_requests) is not int
            or self.max_requests < 0
            or type(self.diagnostic_port) is not int
            or not 0 <= self.diagnostic_port <= 65535
            or (self.diagnostic_port == 0) != (self.busybox_path is None)
            or (self.diagnostic_port == 0) != (self.diagnostic_host is None)
            or (self.diagnostic_port == 0)
                != (self.network_manager_path is None)
            or (
                self.remote_hash_cache_path is not None
                and (
                    not isinstance(self.remote_hash_cache_path, Path)
                    or not self.remote_hash_cache_path.is_absolute()
                )
            )
        ):
            raise PhysicalAdapterError(
                "direct phone session configuration is invalid"
            )
        object.__setattr__(
            self,
            "session_script",
            _android_path("phone session script", self.session_script),
        )
        object.__setattr__(
            self,
            "restore_script",
            _android_path("phone restore script", self.restore_script),
        )
        object.__setattr__(
            self,
            "session_root",
            _android_path("phone session root", self.session_root),
        )
        if self.busybox_path is not None:
            object.__setattr__(
                self,
                "busybox_path",
                _android_path("phone busybox", self.busybox_path),
            )
            if (
                type(self.diagnostic_host) is not str
                or not self.diagnostic_host
                or not self.diagnostic_host.isascii()
                or not isinstance(self.network_manager_path, Path)
                or not self.network_manager_path.is_file()
                or not os.access(self.network_manager_path, os.X_OK)
            ):
                raise PhysicalAdapterError(
                    "phone diagnostic network configuration is invalid"
                )
        identity = self.transport_qualification_identity
        dependency_paths = dict(self.transport_host_dependency_paths)
        if any(
            type(name) is not str
            or not name
            or not name.isascii()
            or ":" in name
            or not isinstance(path, Path)
            or not path.is_file()
            for name, path in dependency_paths.items()
        ):
            raise PhysicalAdapterError(
                "phone transport host dependency is invalid"
            )
        if identity is not None:
            expected_dependencies = {
                key.removeprefix("host_dependency_sha256:")
                for key in identity.software_identity
                if key.startswith("host_dependency_sha256:")
            }
            if (
                not isinstance(identity, TransportQualificationIdentity)
                or not isinstance(self.transport_host_binary_path, Path)
                or not self.transport_host_binary_path.is_file()
                or type(self.phone_boot_image_sha256) is not str
                or not self.phone_boot_image_sha256.startswith("sha256:")
                or len(self.phone_boot_image_sha256) != 71
                or any(
                    value not in "0123456789abcdef"
                    for value in self.phone_boot_image_sha256[7:]
                )
                or set(dependency_paths) != expected_dependencies
            ):
                raise PhysicalAdapterError(
                    "phone transport qualification configuration is invalid"
                )
            if identity.hardware_identity.get("phone_cpu_affinity") != self.cpu_affinity:
                raise PhysicalAdapterError("phone CPU affinity differs from qualification")
        elif (
            self.transport_host_binary_path is not None
            or dependency_paths
            or self.phone_boot_image_sha256 is not None
            or self.cpu_affinity is not None
        ):
            raise PhysicalAdapterError(
                "phone transport qualification identity is absent"
            )
        object.__setattr__(
            self,
            "transport_host_dependency_paths",
            MappingProxyType(dict(sorted(dependency_paths.items()))),
        )
        gadget_values = (
            self.android_gadget_path,
            self.functionfs_gadget_path,
            self.functionfs_root_path,
            self.phone_usb_controller,
        )
        if any(value is not None for value in gadget_values):
            if not all(value is not None for value in gadget_values):
                raise PhysicalAdapterError(
                    "phone gadget configuration is incomplete"
                )
            object.__setattr__(
                self,
                "android_gadget_path",
                _android_path(
                    "phone Android gadget", self.android_gadget_path
                ),
            )
            object.__setattr__(
                self,
                "functionfs_gadget_path",
                _android_path(
                    "phone FunctionFS gadget",
                    self.functionfs_gadget_path,
                ),
            )
            object.__setattr__(
                self,
                "functionfs_root_path",
                _android_path(
                    "phone FunctionFS root", self.functionfs_root_path
                ),
            )
            if (
                type(self.phone_usb_controller) is not str
                or not self.phone_usb_controller
                or not self.phone_usb_controller.isascii()
            ):
                raise PhysicalAdapterError(
                    "phone USB controller is invalid"
                )
        object.__setattr__(
            self,
            "worker_paths_by_artifact",
            _artifact_mapping(
                "phone worker", self.worker_paths_by_artifact
            ),
        )
        object.__setattr__(
            self,
            "model_paths_by_artifact",
            _artifact_mapping("phone model", self.model_paths_by_artifact),
        )
        ffn_indexes = dict(self.ffn_shards_by_artifact)
        if any(
            artifact not in self.model_paths_by_artifact
            or not isinstance(index, FfnShardIndex)
            or index.parent_sha256 != artifact
            for artifact, index in ffn_indexes.items()
        ):
            raise PhysicalAdapterError(
                "phone FFN shard indexes differ from deployed artifacts"
            )
        object.__setattr__(
            self,
            "ffn_shards_by_artifact",
            MappingProxyType(dict(sorted(ffn_indexes.items()))),
        )
        backends = {}
        for device_id, backend in self.backend_by_device.items():
            if (
                type(device_id) is not str
                or not device_id
                or not device_id.isascii()
                or type(backend) is not str
                or not backend
                or not backend.isascii()
            ):
                raise PhysicalAdapterError("phone backend mapping is invalid")
            backends[device_id] = backend
        if not backends:
            raise PhysicalAdapterError("phone backend mapping is empty")
        object.__setattr__(
            self,
            "backend_by_device",
            MappingProxyType(dict(sorted(backends.items()))),
        )
        multi_values = (
            self.resident_workers_path,
            self.resident_router_path,
            self.multi_session_port_base,
            self.multi_session_device_count,
        )
        if any(value is not None for value in multi_values):
            if not all(value is not None for value in multi_values):
                raise PhysicalAdapterError(
                    "multi-session phone configuration is incomplete"
                )
            object.__setattr__(
                self,
                "resident_workers_path",
                _android_path(
                    "phone resident workers", self.resident_workers_path
                ),
            )
            object.__setattr__(
                self,
                "resident_router_path",
                _android_path(
                    "phone resident router", self.resident_router_path
                ),
            )
            if (
                type(self.multi_session_port_base) is not int
                or self.multi_session_port_base <= 0
                or self.multi_session_port_base > 65535
                or type(self.multi_session_device_count) is not int
                or self.multi_session_device_count <= 0
                or self.multi_session_port_base
                    + self.multi_session_device_count > 65536
            ):
                raise PhysicalAdapterError(
                    "multi-session phone capacity is invalid"
                )
