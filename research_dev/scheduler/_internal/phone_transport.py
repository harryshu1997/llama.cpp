"""Phone transfer contracts used by certified offload routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .operator_split import OperatorSplitInvocation


__all__ = [
    "PHONE_TRANSPORT_SCHEMA",
    "DmaBufTransfer",
    "PhoneTransportContract",
    "PhoneTransportError",
]


PHONE_TRANSPORT_SCHEMA = "research-scheduler-phone-transport-v1"


class PhoneTransportError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PhoneTransportError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PhoneTransportError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhoneTransportError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DmaBufTransfer:
    tokens: int
    payload_bytes: int
    upload_wire_bytes: int
    download_wire_bytes: int
    aggregate_wire_bytes: int
    queue_depth: int

    def __post_init__(self) -> None:
        _integer("transfer tokens", self.tokens, 1)
        _integer("transfer payload_bytes", self.payload_bytes, 1)
        _integer("transfer upload_wire_bytes", self.upload_wire_bytes, 1)
        _integer("transfer download_wire_bytes", self.download_wire_bytes, 1)
        _integer("transfer aggregate_wire_bytes", self.aggregate_wire_bytes, 2)
        _integer("transfer queue_depth", self.queue_depth, 1)
        if self.aggregate_wire_bytes != (
            self.upload_wire_bytes + self.download_wire_bytes
        ):
            raise PhoneTransportError("aggregate wire bytes do not match")


@dataclass(frozen=True)
class PhoneTransportContract:
    transport_id: str
    protocol: str
    host_endpoint: str
    phone_endpoint: str
    allocator: str
    io_type: str
    payload_offset_bytes: int
    max_payload_bytes: int
    queue_depth: int
    usb_speed_mbps: int
    usb_vendor_product: str
    phone_resource_id: str
    transport_resource_id: str
    usb_root_resource_id: str
    bridge_residency_id: str
    worker_residency_id: str
    reset_generation: int
    max_reset_recoveries: int

    def __post_init__(self) -> None:
        for name in (
            "transport_id",
            "protocol",
            "host_endpoint",
            "phone_endpoint",
            "phone_resource_id",
            "transport_resource_id",
            "usb_root_resource_id",
        ):
            _text(name, getattr(self, name))
        if self.allocator not in {
            "malloc",
            "devmem",
            "malloc-split",
            "devmem-split",
        }:
            raise PhoneTransportError("unsupported transport allocator")
        if self.io_type not in {"f16", "f32"}:
            raise PhoneTransportError("transport io_type must be f16 or f32")
        _integer("payload_offset_bytes", self.payload_offset_bytes, 1)
        _integer("max_payload_bytes", self.max_payload_bytes, 1)
        _integer("queue_depth", self.queue_depth, 1)
        _integer("usb_speed_mbps", self.usb_speed_mbps, 1)
        vendor, separator, product = self.usb_vendor_product.partition(":")
        if (
            separator != ":"
            or len(vendor) != 4
            or len(product) != 4
            or any(ch not in "0123456789abcdef" for ch in vendor + product)
        ):
            raise PhoneTransportError("usb_vendor_product must be lowercase hex")
        _text("bridge_residency_id", self.bridge_residency_id)
        _text("worker_residency_id", self.worker_residency_id)
        _integer("reset_generation", self.reset_generation)
        _integer("max_reset_recoveries", self.max_reset_recoveries)

    @classmethod
    def from_json(cls, value: object) -> "PhoneTransportContract":
        if type(value) is not dict:
            raise PhoneTransportError("phone transport must be an object")
        row: Mapping[str, Any] = value
        if row.get("schema") != PHONE_TRANSPORT_SCHEMA:
            raise PhoneTransportError("phone transport schema mismatch")
        return cls(
            transport_id=row.get("transport_id"),
            protocol=row.get("protocol"),
            host_endpoint=row.get("host_endpoint"),
            phone_endpoint=row.get("phone_endpoint"),
            allocator=row.get("allocator"),
            io_type=row.get("io_type"),
            payload_offset_bytes=row.get("payload_offset_bytes"),
            max_payload_bytes=row.get("max_payload_bytes"),
            queue_depth=row.get("queue_depth"),
            usb_speed_mbps=row.get("usb_speed_mbps"),
            usb_vendor_product=row.get("usb_vendor_product"),
            phone_resource_id=row.get("phone_resource_id"),
            transport_resource_id=row.get("transport_resource_id"),
            usb_root_resource_id=row.get("usb_root_resource_id"),
            bridge_residency_id=row.get("bridge_residency_id"),
            worker_residency_id=row.get("worker_residency_id"),
            reset_generation=row.get("reset_generation"),
            max_reset_recoveries=row.get("max_reset_recoveries"),
        )

    @property
    def max_wire_bytes(self) -> int:
        return self.payload_offset_bytes + self.max_payload_bytes

    def transfer(self, invocation: OperatorSplitInvocation) -> DmaBufTransfer:
        if invocation.activation_bytes > self.max_payload_bytes:
            raise PhoneTransportError("activation exceeds transport payload")
        wire_bytes = self.payload_offset_bytes + invocation.activation_bytes
        return DmaBufTransfer(
            tokens=invocation.tokens,
            payload_bytes=invocation.activation_bytes,
            upload_wire_bytes=wire_bytes,
            download_wire_bytes=wire_bytes,
            aggregate_wire_bytes=2 * wire_bytes,
            queue_depth=self.queue_depth,
        )

    def required_resource_ids(self) -> tuple[str, ...]:
        return (
            self.phone_resource_id,
            self.transport_resource_id,
            self.usb_root_resource_id,
        )

    def required_residency_ids(self) -> tuple[str, ...]:
        return (self.bridge_residency_id, self.worker_residency_id)

    def to_json(self) -> dict[str, object]:
        return {
            "allocator": self.allocator,
            "bridge_residency_id": self.bridge_residency_id,
            "host_endpoint": self.host_endpoint,
            "io_type": self.io_type,
            "max_payload_bytes": self.max_payload_bytes,
            "max_reset_recoveries": self.max_reset_recoveries,
            "payload_offset_bytes": self.payload_offset_bytes,
            "phone_endpoint": self.phone_endpoint,
            "phone_resource_id": self.phone_resource_id,
            "protocol": self.protocol,
            "queue_depth": self.queue_depth,
            "reset_generation": self.reset_generation,
            "schema": PHONE_TRANSPORT_SCHEMA,
            "transport_id": self.transport_id,
            "transport_resource_id": self.transport_resource_id,
            "usb_root_resource_id": self.usb_root_resource_id,
            "usb_speed_mbps": self.usb_speed_mbps,
            "usb_vendor_product": self.usb_vendor_product,
            "worker_residency_id": self.worker_residency_id,
        }
