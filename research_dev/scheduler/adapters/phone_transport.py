"""Interpret the exact phone transport selected in an execution ticket."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .contracts import PhysicalAdapterError


# ``tcp`` keeps its legacy meaning (a host tensor bridge owns the phone link); ``adb-tcp`` is a
# protocol-v6 worker on the phone reached through ``adb forward`` on that phone's own serial
ADB_TCP_TRANSPORT_GENERATION = "adb-tcp-worker-v6"


def _text(parameters: Mapping[str, int | str], name: str) -> str:
    value = parameters.get(name)
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(
            "phone transport parameter is invalid: " + name
        )
    return value


def _integer(parameters: Mapping[str, int | str], name: str) -> int:
    value = parameters.get(name)
    if type(value) is not int or value < 0:
        raise PhysicalAdapterError(
            "phone transport parameter is invalid: " + name
        )
    return value


@dataclass(frozen=True)
class PhoneTransportContract:
    transport: str
    allocator: str
    queue_depth: int
    concurrent_streams: int
    max_payload_bytes: int
    full_duplex: bool
    split_h2d: bool
    generation: str
    profile_id: str
    usbfs_available_bytes: int
    slot_safety_bytes: int
    vendor_id: int
    product_id: int
    control_host: str
    control_port: int
    batch_plan: str = "split-row"
    qualification_identity_sha256: str | None = None
    # adb-tcp only: the phone that owns the forward and the worker's phone-side port
    adb_serial: str | None = None
    adb_port: int = 0
    phone_worker_port: int = 0

    def __post_init__(self) -> None:
        if self.batch_plan not in {"coalesced-batch", "single", "split-row"}:
            raise PhysicalAdapterError("phone transport batch plan is invalid")
        identity = self.qualification_identity_sha256
        if identity is not None and (
            type(identity) is not str
            or not identity.startswith("sha256:")
            or len(identity) != 71
            or any(value not in "0123456789abcdef" for value in identity[7:])
        ):
            raise PhysicalAdapterError(
                "phone transport qualification identity is invalid"
            )
        if (self.transport == "adb-tcp") != (self.adb_serial is not None) or (
            self.transport == "adb-tcp" and (
                type(self.adb_serial) is not str or not self.adb_serial
                or not self.adb_serial.isascii()
                or not 0 < self.adb_port <= 65535
                or not 0 < self.phone_worker_port <= 65535
                or not 0 < self.control_port <= 65535
            )
        ):
            raise PhysicalAdapterError("adb-tcp phone transport identity is invalid")

    @property
    def uses_tensor_bridge(self) -> bool:
        return self.transport == "tcp"

    def shares_resident_session_with(
        self,
        other: "PhoneTransportContract",
    ) -> bool:
        """Return whether two contracts can use one resident phone router."""
        if not isinstance(other, PhoneTransportContract):
            return False
        return (
            self.transport == other.transport == "functionfs-usb"
            and self.allocator == other.allocator
            and self.queue_depth == other.queue_depth
            and self.concurrent_streams == other.concurrent_streams
            and self.full_duplex == other.full_duplex
            and self.split_h2d == other.split_h2d
            and self.generation == other.generation
            and self.usbfs_available_bytes == other.usbfs_available_bytes
            and self.slot_safety_bytes == other.slot_safety_bytes
            and self.vendor_id == other.vendor_id
            and self.product_id == other.product_id
            and self.control_host == other.control_host
            and self.control_port == other.control_port
            and self.batch_plan == other.batch_plan
            and self.qualification_identity_sha256
                == other.qualification_identity_sha256
        )

    def server_environment(self) -> Mapping[str, str]:
        if self.uses_tensor_bridge or self.transport == "adb-tcp":
            return MappingProxyType({
                "S41_SERVER_FFN_HOST": self.control_host,
                "S41_SERVER_FFN_PORT": str(self.control_port),
                "S41_SERVER_FFN_TRANSPORT": "tcp",
            })
        return MappingProxyType({
            "S41_SERVER_FFN_TRANSPORT": "functionfs-usb",
            "S41_SERVER_FFN_USB_ALLOCATOR": self.allocator,
            "S41_SERVER_FFN_USB_BATCH_PLAN": self.batch_plan,
            "S41_SERVER_FFN_USB_FULL_DUPLEX": str(int(self.full_duplex)),
            "S41_SERVER_FFN_USB_MAX_PAYLOAD_BYTES": str(
                self.max_payload_bytes
            ),
            "S41_SERVER_FFN_USB_PRODUCT_ID": str(self.product_id),
            "S41_SERVER_FFN_USB_QUEUE_DEPTH": str(self.queue_depth),
            "S41_SERVER_FFN_USB_SLOT_SAFETY_BYTES": str(
                self.slot_safety_bytes
            ),
            "S41_SERVER_FFN_USB_SPLIT_H2D": str(int(self.split_h2d)),
            "S41_SERVER_FFN_USB_TRANSPORT_GENERATION": self.generation,
            "S41_SERVER_FFN_USB_VENDOR_ID": str(self.vendor_id),
            "S41_SERVER_FFN_USBFS_AVAILABLE_BYTES": str(
                self.usbfs_available_bytes
            ),
        })


def phone_transport_contract(
    parameters: Mapping[str, int | str],
) -> PhoneTransportContract:
    """Validate transport facts carried by the selected physical binding."""
    transport = parameters.get("ffn_transport", "tcp")
    if transport == "tcp":
        allocator = _text(parameters, "bridge_allocator")
        if allocator not in {
            "malloc", "devmem", "malloc-split", "devmem-split"
        }:
            raise PhysicalAdapterError("phone bridge allocator is invalid")
        host = _text(parameters, "ffn_bridge_host")
        port = _integer(parameters, "ffn_bridge_port")
        if not 0 < port <= 65535:
            raise PhysicalAdapterError("phone bridge port is invalid")
        return PhoneTransportContract(
            transport="tcp",
            allocator=allocator,
            queue_depth=_integer(parameters, "bridge_queue_depth"),
            concurrent_streams=1,
            max_payload_bytes=_integer(parameters, "usb_max_payload_bytes")
                if "usb_max_payload_bytes" in parameters else 0,
            full_duplex=False,
            split_h2d=allocator.endswith("-split"),
            generation=_text(parameters, "usb_transport_generation")
                if "usb_transport_generation" in parameters
                else "functionfs-dmabuf-sync-v1",
            profile_id=_text(parameters, "usb_transport_profile_id")
                if "usb_transport_profile_id" in parameters
                else "legacy-unqualified",
            usbfs_available_bytes=0,
            slot_safety_bytes=0,
            vendor_id=0,
            product_id=0,
            control_host=host,
            control_port=port,
            qualification_identity_sha256=None,
        )
    if transport == "adb-tcp":
        return PhoneTransportContract(
            transport="adb-tcp",
            allocator="socket",
            queue_depth=1,
            concurrent_streams=1,
            max_payload_bytes=0,
            full_duplex=False,
            split_h2d=False,
            generation=ADB_TCP_TRANSPORT_GENERATION,
            profile_id=_text(parameters, "usb_transport_profile_id")
                if "usb_transport_profile_id" in parameters
                else "adb-tcp-unqualified",
            usbfs_available_bytes=0,
            slot_safety_bytes=0,
            vendor_id=0,
            product_id=0,
            control_host=_text(parameters, "ffn_worker_host"),
            control_port=_integer(parameters, "ffn_worker_port"),
            batch_plan="single",
            qualification_identity_sha256=(
                _text(parameters, "usb_transport_qualification_identity_sha256")
                if "usb_transport_qualification_identity_sha256" in parameters
                else None
            ),
            adb_serial=_text(parameters, "adb_serial"),
            adb_port=_integer(parameters, "adb_port"),
            phone_worker_port=_integer(parameters, "phone_worker_port"),
        )
    if transport != "functionfs-usb":
        raise PhysicalAdapterError("phone transport is invalid")
    allocator = _text(parameters, "usb_allocator")
    queue_depth = _integer(parameters, "usb_queue_depth")
    concurrent_streams = (
        _integer(parameters, "usb_concurrent_streams")
        if "usb_concurrent_streams" in parameters else 1
    )
    max_payload = _integer(parameters, "usb_max_payload_bytes")
    full_duplex = _integer(parameters, "usb_full_duplex")
    split_h2d = _integer(parameters, "usb_split_h2d")
    safety = _integer(parameters, "usb_slot_safety_bytes")
    vendor_id = _integer(parameters, "usb_vendor_id")
    product_id = _integer(parameters, "usb_product_id")
    if (
        allocator not in {"malloc", "devmem"}
        or not 0 < queue_depth <= 64
        or not 0 < concurrent_streams <= queue_depth
        or max_payload <= 0
        or full_duplex not in {0, 1}
        or split_h2d not in {0, 1}
        or safety <= 0
        or not 0 < vendor_id <= 0xFFFF
        or not 0 < product_id <= 0xFFFF
    ):
        raise PhysicalAdapterError("phone USB transport contract is invalid")
    return PhoneTransportContract(
        transport=transport,
        allocator=allocator,
        queue_depth=queue_depth,
        concurrent_streams=concurrent_streams,
        max_payload_bytes=max_payload,
        full_duplex=bool(full_duplex),
        split_h2d=bool(split_h2d),
        generation=_text(parameters, "usb_transport_generation"),
        profile_id=_text(parameters, "usb_transport_profile_id"),
        usbfs_available_bytes=_integer(
            parameters, "usbfs_available_bytes"
        ),
        slot_safety_bytes=safety,
        vendor_id=vendor_id,
        product_id=product_id,
        control_host="direct-functionfs",
        control_port=0,
        batch_plan=(
            parameters.get("usb_batch_plan", "split-row")
        ),
        qualification_identity_sha256=(
            _text(
                parameters,
                "usb_transport_qualification_identity_sha256",
            )
            if "usb_transport_qualification_identity_sha256" in parameters
            else None
        ),
    )
