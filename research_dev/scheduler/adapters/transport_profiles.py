"""Identity-bound transport qualification profiles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.placement import TransferLink
from .._internal.types import canonical_json


TRANSPORT_QUALIFICATION_IDENTITY_SCHEMA = (
    "research-scheduler-transport-qualification-identity-v1"
)


# The server prints this line when its FFN split client is compiled in and the FFN environment is
# present. A server built without S41_SERVER_FFN_SPLIT ignores that environment silently, so the
# marker must exist somewhere in the host server stack before receipts can be bound to it.
HOST_FFN_CLIENT_MARKER = b"S41SERVERFFN ready"


class TransportProfileError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise TransportProfileError(name + " is invalid")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value)
    if (
        not result.startswith("sha256:")
        or len(result) != 71
        or any(character not in "0123456789abcdef" for character in result[7:])
    ):
        raise TransportProfileError(name + " must be SHA-256")
    return result


def _positive_integer(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise TransportProfileError(name + " is invalid")
    return value


def _string_mapping(name: str, value: object) -> Mapping[str, str]:
    if type(value) is not dict or not value:
        raise TransportProfileError(name + " is invalid")
    result = {
        _text(name + " key", key): _text(name + " value", item)
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class TransportQualificationIdentity:
    identity_id: str
    transport_generation: str
    hardware_identity: Mapping[str, str]
    software_identity: Mapping[str, str]
    qualified_allocators: tuple[str, ...]
    receipt_sha256s: tuple[str, ...]
    minimum_usb_speed_mbps: int

    def __post_init__(self) -> None:
        _text("transport qualification identity id", self.identity_id)
        _text(
            "transport qualification generation", self.transport_generation
        )
        hardware = _string_mapping(
            "transport qualification hardware identity",
            self.hardware_identity,
        )
        software = _string_mapping(
            "transport qualification software identity",
            self.software_identity,
        )
        required_hardware = {
            "functionfs_identity",
            "phone_boot_image_sha256",
            "phone_kernel_release",
            "phone_usb_controller",
            "phone_usb_serial",
            "phone_usb_sysfs_device",
        }
        required_software = {
            "host_binary_sha256",
            "phone_session_sha256",
            "phone_worker_sha256",
            "qualification_binary_sha256",
            "qualification_phone_session_sha256",
            "qualification_phone_worker_sha256",
            "transport_client_source_sha256",
        }
        if not required_hardware.issubset(hardware):
            raise TransportProfileError(
                "transport qualification hardware identity is incomplete"
            )
        if not required_software.issubset(software):
            raise TransportProfileError(
                "transport qualification software identity is incomplete"
            )
        for values, keys in (
            (hardware, ("phone_boot_image_sha256",)),
            (
                software,
                (
                    "host_binary_sha256",
                    "phone_session_sha256",
                    "phone_worker_sha256",
                    "qualification_binary_sha256",
                    "qualification_phone_session_sha256",
                    "qualification_phone_worker_sha256",
                    "transport_client_source_sha256",
                ),
            ),
        ):
            for key in keys:
                _sha256("transport qualification " + key, values[key])
        for key, value in software.items():
            if key.startswith("host_dependency_sha256:"):
                if key == "host_dependency_sha256:":
                    raise TransportProfileError(
                        "transport qualification host dependency is invalid"
                    )
                _sha256("transport qualification " + key, value)
        resident_keys = {
            "phone_resident_router_sha256",
            "phone_resident_workers_sha256",
        }
        present_resident_keys = resident_keys.intersection(software)
        if present_resident_keys and present_resident_keys != resident_keys:
            raise TransportProfileError(
                "transport qualification resident identity is incomplete"
            )
        for key in sorted(present_resident_keys):
            _sha256("transport qualification " + key, software[key])
        allocators = tuple(sorted({
            _text("transport qualification allocator", value)
            for value in self.qualified_allocators
        }))
        if not allocators or set(allocators) - {"devmem", "malloc"}:
            raise TransportProfileError(
                "transport qualification allocator is unsupported"
            )
        receipts = tuple(sorted({
            _sha256("transport qualification receipt", value)
            for value in self.receipt_sha256s
        }))
        if not receipts:
            raise TransportProfileError(
                "transport qualification receipts are absent"
            )
        speed = _positive_integer(
            "transport qualification USB speed",
            self.minimum_usb_speed_mbps,
        )
        object.__setattr__(self, "hardware_identity", hardware)
        object.__setattr__(self, "software_identity", software)
        object.__setattr__(self, "qualified_allocators", allocators)
        object.__setattr__(self, "receipt_sha256s", receipts)
        object.__setattr__(self, "minimum_usb_speed_mbps", speed)

    @property
    def identity_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json(self.to_json()).encode("ascii")
        ).hexdigest()

    def to_json(self) -> dict[str, object]:
        return {
            "hardware_identity": dict(self.hardware_identity),
            "identity_id": self.identity_id,
            "minimum_usb_speed_mbps": self.minimum_usb_speed_mbps,
            "qualified_allocators": list(self.qualified_allocators),
            "receipt_sha256s": list(self.receipt_sha256s),
            "schema": TRANSPORT_QUALIFICATION_IDENTITY_SCHEMA,
            "software_identity": dict(self.software_identity),
            "transport_generation": self.transport_generation,
        }

    @classmethod
    def from_json(cls, value: object) -> "TransportQualificationIdentity":
        if type(value) is not dict or value.get("schema") != (
            TRANSPORT_QUALIFICATION_IDENTITY_SCHEMA
        ):
            raise TransportProfileError(
                "transport qualification identity schema differs"
            )
        allocators = value.get("qualified_allocators")
        receipts = value.get("receipt_sha256s")
        if type(allocators) is not list or type(receipts) is not list:
            raise TransportProfileError(
                "transport qualification identity lists are invalid"
            )
        return cls(
            identity_id=value.get("identity_id"),
            transport_generation=value.get("transport_generation"),
            hardware_identity=value.get("hardware_identity"),
            software_identity=value.get("software_identity"),
            qualified_allocators=tuple(allocators),
            receipt_sha256s=tuple(receipts),
            minimum_usb_speed_mbps=value.get("minimum_usb_speed_mbps"),
        )


def load_transport_qualification_identity(
    path: Path,
) -> TransportQualificationIdentity:
    if not isinstance(path, Path) or not path.is_file():
        raise TransportProfileError(
            "transport qualification identity file is absent"
        )
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TransportProfileError(
            "transport qualification identity cannot be read"
        ) from exc
    return TransportQualificationIdentity.from_json(value)


def _host_stack_has_ffn_client(
    host_binary_path: Path, dependency_paths: Mapping[str, Path]
) -> bool:
    """True when the server or one of its libraries carries the FFN client."""
    for path in (host_binary_path, *dependency_paths.values()):
        with path.open("rb") as stream:
            previous = b""
            while True:
                chunk = stream.read(1 << 20)
                if not chunk:
                    break
                if HOST_FFN_CLIENT_MARKER in previous + chunk:
                    return True
                previous = chunk[-len(HOST_FFN_CLIENT_MARKER):]
    return False


def build_transport_qualification_identity(
    *,
    identity_id: str,
    transport_generation: str,
    hardware_identity: Mapping[str, str],
    phone_session_sha256: str,
    phone_worker_sha256: str,
    qualification_phone_session_sha256: str,
    qualification_phone_worker_sha256: str,
    host_binary_path: Path,
    qualification_binary_path: Path,
    transport_client_source_path: Path,
    qualified_allocators: Sequence[str],
    receipt_paths: Sequence[Path],
    minimum_usb_speed_mbps: int,
    host_dependency_paths: Mapping[str, Path] | None = None,
    phone_resident_workers_path: Path | None = None,
    phone_resident_router_path: Path | None = None,
) -> TransportQualificationIdentity:
    """Bind measured USB receipts to one exact physical software stack."""
    if any(
        not isinstance(path, Path) or not path.is_file()
        for path in (
            host_binary_path,
            qualification_binary_path,
            transport_client_source_path,
        )
    ):
        raise TransportProfileError(
            "transport qualification host binary is absent"
        )
    paths = tuple(receipt_paths)
    if not paths or any(
        not isinstance(path, Path) or not path.is_file() for path in paths
    ):
        raise TransportProfileError(
            "transport qualification receipt path is absent"
        )
    dependency_paths = dict(host_dependency_paths or {})
    resident_paths = (
        phone_resident_workers_path,
        phone_resident_router_path,
    )
    if any(path is not None for path in resident_paths) and (
        any(path is None for path in resident_paths)
        or any(not isinstance(path, Path) or not path.is_file()
               for path in resident_paths)
    ):
        raise TransportProfileError(
            "transport qualification resident binary is incomplete"
        )
    if any(
        type(name) is not str
        or not name
        or not name.isascii()
        or ":" in name
        or not isinstance(path, Path)
        or not path.is_file()
        for name, path in dependency_paths.items()
    ):
        raise TransportProfileError(
            "transport qualification host dependency is invalid"
        )
    if not _host_stack_has_ffn_client(host_binary_path, dependency_paths):
        raise TransportProfileError(
            "transport qualification host server lacks the FFN split client"
            " (built without S41_SERVER_FFN_SPLIT)"
        )
    identity = TransportQualificationIdentity(
        identity_id=identity_id,
        transport_generation=transport_generation,
        hardware_identity=hardware_identity,
        software_identity={
            "host_binary_sha256": _file_sha256(host_binary_path),
            "phone_session_sha256": phone_session_sha256,
            "phone_worker_sha256": phone_worker_sha256,
            "qualification_binary_sha256": _file_sha256(
                qualification_binary_path
            ),
            "qualification_phone_session_sha256": (
                qualification_phone_session_sha256
            ),
            "qualification_phone_worker_sha256": (
                qualification_phone_worker_sha256
            ),
            "transport_client_source_sha256": _file_sha256(
                transport_client_source_path
            ),
            **{
                "host_dependency_sha256:" + name: _file_sha256(path)
                for name, path in sorted(dependency_paths.items())
            },
            **(
                {}
                if phone_resident_workers_path is None
                else {
                    "phone_resident_workers_sha256": _file_sha256(
                        phone_resident_workers_path
                    ),
                    "phone_resident_router_sha256": _file_sha256(
                        phone_resident_router_path
                    ),
                }
            ),
        },
        qualified_allocators=tuple(qualified_allocators),
        receipt_sha256s=tuple(_file_sha256(path) for path in paths),
        minimum_usb_speed_mbps=minimum_usb_speed_mbps,
    )
    materialize_measured_usb_links(
        tuple(sorted({path.parent for path in paths})),
        identity,
        host_device_id="qualification-host",
        phone_device_id="qualification-phone",
    )
    return identity


def materialize_measured_usb_links(
    directories: Sequence[Path],
    identity: TransportQualificationIdentity,
    *,
    host_device_id: str,
    phone_device_id: str,
) -> tuple[TransferLink, ...]:
    """Convert exact-stack transport receipts into measured directed links."""
    if not isinstance(identity, TransportQualificationIdentity):
        raise TransportProfileError(
            "transport qualification identity is invalid"
        )
    host_device_id = _text("transport host device", host_device_id)
    phone_device_id = _text("transport phone device", phone_device_id)
    if host_device_id == phone_device_id:
        raise TransportProfileError("transport devices must differ")
    paths = []
    for directory in directories:
        if not isinstance(directory, Path) or not directory.is_dir():
            raise TransportProfileError(
                "transport qualification directory is absent"
            )
        paths.extend(sorted(directory.glob("*.json")))
    if not paths:
        raise TransportProfileError(
            "transport qualification receipts are absent"
        )
    allowed_receipts = set(identity.receipt_sha256s)
    receipts = []
    for path in paths:
        receipt_sha256 = _file_sha256(path)
        if receipt_sha256 not in allowed_receipts:
            continue
        try:
            value = json.loads(path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TransportProfileError(
                "transport qualification receipt cannot be read"
            ) from exc
        if (
            type(value) is not dict
            or value.get("schema") != "s41_ffs_dmabuf_transport_v2"
            or value.get("device_mode") != "dmabuf"
            or value.get("host_mode") != "async"
            or value.get("reset_recoveries") != 0
            or value.get("transport_generation")
                != identity.transport_generation
            or value.get("host_allocator")
                not in identity.qualified_allocators
        ):
            raise TransportProfileError(
                "transport qualification receipt identity differs"
            )
        for name in (
            "queue_depth",
            "configured_queue_depth",
            "request_bytes",
            "response_bytes",
            "usbfs_available_bytes",
            "slot_safety_bytes",
        ):
            _positive_integer("transport receipt " + name, value.get(name))
        if value["configured_queue_depth"] < value["queue_depth"]:
            raise TransportProfileError(
                "transport receipt queue depth exceeds configuration"
            )
        receipts.append((value, receipt_sha256))
    if not receipts:
        raise TransportProfileError(
            "identity-bound transport qualification receipts are absent"
        )

    def configuration(value: Mapping[str, object]) -> tuple[object, ...]:
        return (
            value["host_allocator"],
            value["configured_queue_depth"],
            value["queue_depth"],
            value["transport_generation"],
            value["usbfs_available_bytes"],
            value["slot_safety_bytes"],
        )

    grouped: dict[
        tuple[tuple[object, ...], int],
        list[tuple[Mapping[str, object], str]],
    ] = {}
    for value, evidence_id in receipts:
        for payload in {value["request_bytes"], value["response_bytes"]}:
            grouped.setdefault(
                (configuration(value), payload), []
            ).append((value, evidence_id))

    rows = []
    for (config, payload), values in sorted(grouped.items()):
        duplex = tuple(
            row for row in values
            if row[0]["request_bytes"] == payload
            and row[0]["response_bytes"] == payload
        )
        h2d = tuple(
            row for row in values
            if row[0]["request_bytes"] == payload
            and row[0]["response_bytes"] < payload
        )
        d2h = tuple(
            row for row in values
            if row[0]["response_bytes"] == payload
            and row[0]["request_bytes"] < payload
        )
        if not duplex or not h2d or not d2h:
            continue
        allocator, configured, active, generation, usbfs, safety = config
        bundle_id = (
            f"{generation}:{allocator}:configured-{configured}:"
            f"active-{active}:payload-{payload}"
        )
        evidence_ids = tuple(sorted({
            identity.identity_sha256,
            *(evidence_id for _, evidence_id in duplex + h2d + d2h),
        }))
        for direction, source, target, direction_rows, rate_name in (
            ("h2d", host_device_id, phone_device_id, h2d,
             "h2d_payload_MBps"),
            ("d2h", phone_device_id, host_device_id, d2h,
             "d2h_payload_MBps"),
        ):
            rates = tuple(row[0].get(rate_name) for row in direction_rows)
            if not rates or any(
                type(rate) not in {int, float} or rate < 1.0
                for rate in rates
            ):
                raise TransportProfileError(
                    "transport directional rate is invalid"
                )
            rows.append(TransferLink(
                link_id=(
                    "usb-" + direction + "-"
                    + hashlib.sha256(bundle_id.encode("ascii")).hexdigest()[:16]
                ),
                source_device=source,
                target_device=target,
                fixed_latency_us=0,
                bandwidth_bytes_per_s=max(1, int(min(rates) * 1_000_000)),
                fixed_dynamic_uj=0,
                dynamic_pj_per_byte=0,
                domain_active_power_mw={},
                status="measured",
                ready=True,
                evidence_ids=evidence_ids,
                minimum_payload_bytes=payload,
                maximum_payload_bytes=payload,
                queue_depth=active,
                concurrent_streams=active,
                allocator=allocator,
                full_duplex=True,
                transport_generation=generation,
                transport_profile_id=bundle_id + ":" + direction,
                usbfs_available_bytes=usbfs,
                slot_safety_bytes=safety,
                qualification_identity_sha256=identity.identity_sha256,
            ))
    if not rows:
        raise TransportProfileError(
            "complete identity-bound transport profiles are absent"
        )
    return tuple(rows)


__all__ = [
    "TRANSPORT_QUALIFICATION_IDENTITY_SCHEMA",
    "TransportProfileError",
    "TransportQualificationIdentity",
    "load_transport_qualification_identity",
    "materialize_measured_usb_links",
]
