"""Several FFN helper phones behind one llama-server: ownership, launch environment, accounting.

One desktop server drives one ``ffn_split::client`` per helper phone (``tools/server/server.cpp``,
``S41_SERVER_FFN_HELPERS``). Every helper owns whole layers, disjoint from the others; a decode
batch keeps one union layer mask and one column width and each client receives its owned subset.
This module holds the host-side contracts around that runtime:

* :class:`PhoneHelperBinding` / :func:`validate_disjoint_ownership` - which device owns which
  layers; overlaps, duplicates and uncovered layers fail closed;
* :func:`helper_server_environment` - the ``S41_SERVER_FFN_HELPER<k>_*`` launch environment; a
  single helper collapses to the legacy single-client environment byte for byte;
* :func:`attribute_ffn_calls` - per-device call accounting of ``S41SERVERFFNCALL`` proofs by the
  owner of each layer, cross-checked against the owner's request-id range;
* :class:`PhoneHelperTransportIdentity` - what one helper's transport qualification binds, and
  which receipt kinds are still missing;
* :func:`check_usb_topology` - read-only sysfs facts for the bus-sharing preflight.

Nothing here touches a phone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .._internal.types import canonical_sha256
from .contracts import PhysicalAdapterError
from .phone_transport import (
    ADB_TCP_TRANSPORT_GENERATION,
    AOA_BRIDGE_TRANSPORT_GENERATION,
    PhoneTransportContract,
    phone_transport_contract,
)


MAXIMUM_HELPERS = 8
# native request-id range of helper k: [1 + k * 2^24, 1 + (k + 1) * 2^24)
REQUEST_ID_RANGE = 1 << 24
HELPER_TRANSPORTS = ("functionfs-usb", "adb-tcp")
_TRANSPORT_KEYS = ("S41_SERVER_FFN_TRANSPORT", "S41_SERVER_FFN_HOST", "S41_SERVER_FFN_PORT")
_TRANSPORT_PREFIXES = ("S41_SERVER_FFN_USB_", "S41_SERVER_FFN_USBFS_")
# shared keys the native runtime refuses next to several helpers
_MULTI_HELPER_FORBIDDEN_KEYS = (
    "S41_SERVER_FFN_REMOTE_RESIDENT_LAYER_MASK",
    "S41_SERVER_FFN_TAIL_FENCE_SOCKET",
    "S41_SERVER_FFN_TAIL_FENCE_LAYER",
    "S41_SERVER_FFN_TAIL_FENCE_JOIN_LAYER",
    "S41_SERVER_FFN_ROW_DIAGNOSTIC_STEPS",
)


def layer_indices(layer_mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(64) if layer_mask >> index & 1)


def layer_spec(layer_mask: int) -> str:
    """``18-23`` / ``0-5,8`` form accepted by the worker's ``--layers``."""
    spans: list[list[int]] = []
    for index in layer_indices(layer_mask):
        if spans and spans[-1][1] == index - 1:
            spans[-1][1] = index
        else:
            spans.append([index, index])
    if not spans:
        raise PhysicalAdapterError("layer mask is empty")
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def _label(value: object) -> str:
    if (
        type(value) is not str or not 0 < len(value) <= 32 or not value.isascii()
        or any(not (character.isalnum() or character in "-_") for character in value)
    ):
        raise PhysicalAdapterError("phone helper label must be 1-32 [A-Za-z0-9_-] characters")
    return value


@dataclass(frozen=True)
class PhoneHelperBinding:
    """One helper phone of a model server.

    ``transport_parameters`` is the ``phone_transport_contract`` input of this helper. It is
    empty for the ticket's own phone (the first helper), which uses the ticket parameters.
    """

    device_id: str
    serial: str
    layer_mask: int
    label: str
    transport_parameters: Mapping[str, int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("device_id", "serial"):
            value = getattr(self, name)
            if type(value) is not str or not value or not value.isascii():
                raise PhysicalAdapterError("phone helper " + name + " is invalid")
        if type(self.layer_mask) is not int or not 0 < self.layer_mask < 1 << 64:
            raise PhysicalAdapterError("phone helper layer mask must be a nonzero 64-bit mask")
        _label(self.label)
        parameters = dict(self.transport_parameters)
        if any(type(key) is not str or type(value) not in (int, str) for key, value in parameters.items()):
            raise PhysicalAdapterError("phone helper transport parameters are invalid")
        object.__setattr__(self, "transport_parameters", MappingProxyType(dict(sorted(parameters.items()))))

    def to_json(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "label": self.label,
            "layer_mask": self.layer_mask,
            "serial": self.serial,
            "transport_parameters": dict(self.transport_parameters),
        }

    @classmethod
    def from_json(cls, value: object) -> "PhoneHelperBinding":
        if type(value) is not dict or type(value.get("transport_parameters", {})) is not dict:
            raise PhysicalAdapterError("phone helper binding is invalid")
        return cls(
            device_id=value.get("device_id"),
            serial=value.get("serial"),
            layer_mask=value.get("layer_mask"),
            label=value.get("label"),
            transport_parameters=value.get("transport_parameters", {}),
        )


def validate_disjoint_ownership(
    helpers: Sequence[PhoneHelperBinding],
    *,
    required_layer_mask: int | None = None,
) -> int:
    """Union of the helpers' layers; refuse duplicates, overlaps and uncovered layers."""
    rows = tuple(helpers)
    if not 0 < len(rows) <= MAXIMUM_HELPERS or any(
        not isinstance(row, PhoneHelperBinding) for row in rows
    ):
        raise PhysicalAdapterError(f"phone helpers must be 1 to {MAXIMUM_HELPERS} bindings")
    for name in ("device_id", "serial", "label"):
        values = [getattr(row, name) for row in rows]
        if len(set(values)) != len(values):
            raise PhysicalAdapterError("phone helper " + name + " values are not unique")
    union = 0
    for row in rows:
        if row.layer_mask & union:
            raise PhysicalAdapterError(
                "phone helper layer ownership overlaps at layers "
                + layer_spec(row.layer_mask & union)
            )
        union |= row.layer_mask
    if required_layer_mask is not None and union != required_layer_mask:
        raise PhysicalAdapterError(
            f"phone helper ownership {union} differs from the server layer mask {required_layer_mask}"
        )
    return union


def helper_layer_masks(policy_layer_mask: int, helpers: Sequence[PhoneHelperBinding]) -> Mapping[str, int]:
    """Owned subset of one union decode policy per device (devices with no active layer omitted)."""
    union = validate_disjoint_ownership(helpers)
    if type(policy_layer_mask) is not int or policy_layer_mask < 0 or policy_layer_mask & ~union:
        raise PhysicalAdapterError("decode policy layer mask exceeds the helpers' ownership")
    return MappingProxyType({
        row.device_id: policy_layer_mask & row.layer_mask
        for row in helpers if policy_layer_mask & row.layer_mask
    })


def helper_transport_contracts(
    helpers: Sequence[PhoneHelperBinding],
    primary_parameters: Mapping[str, int | str],
) -> tuple[PhoneTransportContract, ...]:
    contracts = []
    for index, row in enumerate(helpers):
        parameters = row.transport_parameters
        if not parameters:
            if index != 0:
                raise PhysicalAdapterError(f"phone helper {row.device_id} has no transport parameters")
            parameters = primary_parameters
        contract = phone_transport_contract(parameters)
        if contract.transport not in HELPER_TRANSPORTS:
            raise PhysicalAdapterError(
                f"phone helper {row.device_id} transport {contract.transport} is not a direct worker"
            )
        if contract.transport == "adb-tcp" and contract.adb_serial != row.serial:
            raise PhysicalAdapterError(f"phone helper {row.device_id} forward belongs to another serial")
        contracts.append(contract)
    if sum(row.transport == "functionfs-usb" for row in contracts) > 1:
        raise PhysicalAdapterError("at most one phone helper can use the functionfs-usb transport")
    return tuple(contracts)


def helper_server_environment(
    helpers: Sequence[PhoneHelperBinding],
    shared_environment: Mapping[str, str],
    contracts: Sequence[PhoneTransportContract],
) -> Mapping[str, str]:
    """llama-server FFN environment for one or several helpers.

    ``shared_environment`` holds the shape shared by every client (artifact, union layer mask,
    columns, runtime control, dormant host share ...). One helper yields the legacy environment;
    several yield ``S41_SERVER_FFN_HELPERS`` plus ``S41_SERVER_FFN_HELPER<k>_*``.
    """
    rows = tuple(helpers)
    shared = dict(shared_environment)
    for key, value in shared.items():
        if type(key) is not str or type(value) is not str or key.startswith("S41_SERVER_FFN_HELPER") \
                or key in _TRANSPORT_KEYS or key.startswith(_TRANSPORT_PREFIXES):
            raise PhysicalAdapterError("shared FFN environment carries a transport key: " + str(key))
    mask_text = shared.get("S41_SERVER_FFN_LAYER_MASK", "")
    if not mask_text.isdigit():
        raise PhysicalAdapterError("shared FFN environment lacks S41_SERVER_FFN_LAYER_MASK")
    validate_disjoint_ownership(rows, required_layer_mask=int(mask_text))
    if len(contracts) != len(rows):
        raise PhysicalAdapterError("every phone helper needs one transport contract")
    if len(rows) == 1:
        return MappingProxyType(dict(sorted({**shared, **contracts[0].server_environment()}.items())))
    forbidden = [key for key in _MULTI_HELPER_FORBIDDEN_KEYS if key in shared]
    if forbidden:
        raise PhysicalAdapterError("several phone helpers cannot combine with " + ",".join(forbidden))
    result = {**shared, "S41_SERVER_FFN_HELPERS": str(len(rows))}
    for index, (row, contract) in enumerate(zip(rows, contracts)):
        prefix = f"S41_SERVER_FFN_HELPER{index}_"
        result[prefix + "LABEL"] = row.label
        result[prefix + "LAYER_MASK"] = str(row.layer_mask)
        for key, value in contract.server_environment().items():
            result[prefix + key.removeprefix("S41_SERVER_FFN_")] = value
    return MappingProxyType(dict(sorted(result.items())))


def phone_helper_bindings_from_parameters(
    parameters: Mapping[str, object],
) -> tuple[PhoneHelperBinding, ...] | None:
    """Decode the ``phone_helpers`` adapter parameter (JSON text); the first is the ticket's phone."""
    encoded = parameters.get("phone_helpers")
    if encoded is None:
        return None
    try:
        rows = json.loads(encoded) if type(encoded) is str else None
    except ValueError:
        rows = None
    if type(rows) is not list or not rows:
        raise PhysicalAdapterError("phone_helpers adapter parameter must be a nonempty JSON array")
    helpers = tuple(PhoneHelperBinding.from_json(row) for row in rows)
    if helpers[0].device_id != parameters.get("phone_device_id") or helpers[0].transport_parameters:
        raise PhysicalAdapterError("the first phone helper must be the ticket's own phone")
    validate_disjoint_ownership(helpers)
    return helpers


def phone_helper_launch_environment(
    helpers: Sequence[PhoneHelperBinding],
    ffn_environment: Mapping[str, str],
    primary_parameters: Mapping[str, int | str],
) -> Mapping[str, str]:
    """Replace the single-phone transport of a launch environment by one client per helper."""
    shared = {
        key: value for key, value in ffn_environment.items()
        if key not in _TRANSPORT_KEYS and not key.startswith(_TRANSPORT_PREFIXES)
    }
    return helper_server_environment(
        helpers, shared, helper_transport_contracts(helpers, primary_parameters)
    )


@dataclass(frozen=True)
class PhoneHelperCallAccount:
    device_id: str
    label: str
    layer_mask: int
    calls: int
    rows: int
    payload_bytes: int
    layers_seen: tuple[int, ...]
    first_request_id: int | None
    last_request_id: int | None

    def to_json(self) -> dict[str, object]:
        return {
            "calls": self.calls,
            "device_id": self.device_id,
            "first_request_id": self.first_request_id,
            "label": self.label,
            "last_request_id": self.last_request_id,
            "layer_mask": self.layer_mask,
            "layers_seen": list(self.layers_seen),
            "payload_bytes": self.payload_bytes,
            "rows": self.rows,
        }


def attribute_ffn_calls(
    calls: Sequence[object],
    helpers: Sequence[PhoneHelperBinding],
) -> Mapping[str, PhoneHelperCallAccount]:
    """Attribute ``S41SERVERFFNCALL`` proofs to the device owning each layer.

    ``calls`` are ``LlamaServerFfnCall`` rows. A call on an unowned layer, a repeated request
    id, or a request id outside its owner's native range fails closed.
    """
    rows = tuple(helpers)
    validate_disjoint_ownership(rows)
    totals = {row.device_id: {"calls": 0, "rows": 0, "payload": 0, "layers": set(), "ids": []} for row in rows}
    seen: set[int] = set()
    for call in calls:
        request_id, layer = getattr(call, "request_id", None), getattr(call, "layer", None)
        if type(request_id) is not int or type(layer) is not int or request_id in seen:
            raise PhysicalAdapterError("llama-server FFN call is invalid or repeated")
        seen.add(request_id)
        owners = [index for index, row in enumerate(rows) if row.layer_mask >> layer & 1]
        if len(owners) != 1:
            raise PhysicalAdapterError(f"llama-server FFN call on layer {layer} has no owning phone")
        index = owners[0]
        if not 1 + index * REQUEST_ID_RANGE <= request_id < 1 + (index + 1) * REQUEST_ID_RANGE:
            raise PhysicalAdapterError(
                f"llama-server FFN call {request_id} on layer {layer} is outside helper {rows[index].label}'s ids"
            )
        row = totals[rows[index].device_id]
        row["calls"] += 1
        row["rows"] += call.tokens
        row["payload"] += call.payload_bytes
        row["layers"].add(layer)
        row["ids"].append(request_id)
    return MappingProxyType({
        row.device_id: PhoneHelperCallAccount(
            device_id=row.device_id, label=row.label, layer_mask=row.layer_mask,
            calls=totals[row.device_id]["calls"], rows=totals[row.device_id]["rows"],
            payload_bytes=totals[row.device_id]["payload"],
            layers_seen=tuple(sorted(totals[row.device_id]["layers"])),
            first_request_id=min(totals[row.device_id]["ids"], default=None),
            last_request_id=max(totals[row.device_id]["ids"], default=None),
        )
        for row in rows
    })


# --- transport qualification identity of one helper ---------------------------------------

PHONE_HELPER_TRANSPORT_IDENTITY_SCHEMA = "research-scheduler-phone-helper-transport-identity-v1"
IDENTITY_REQUIREMENTS = MappingProxyType({
    "functionfs-usb": MappingProxyType({
        # exactly the single-phone TransportQualificationIdentity fields
        "hardware": ("functionfs_identity", "phone_boot_image_sha256", "phone_kernel_release",
                     "phone_usb_controller", "phone_usb_serial", "phone_usb_sysfs_device"),
        "software": ("host_binary_sha256", "phone_session_sha256", "phone_worker_sha256",
                     "transport_client_source_sha256"),
        "receipts": ("usb-link-speed", "dmabuf-transport-h2d-d2h-duplex"),
    }),
    "adb-tcp": MappingProxyType({
        "hardware": ("adb_usb_identity", "host_usb_controller", "phone_kernel_release",
                     "phone_usb_serial", "phone_usb_sysfs_device"),
        "software": ("host_binary_sha256", "phone_shard_sha256", "phone_worker_sha256",
                     "transport_client_source_sha256", "worker_environment_sha256"),
        "receipts": ("usb-link-speed", "numerical-rows-1-2-4", "server-token-identity",
                     "adb-forward-round-trip", "scheduler-launched-session"),
    }),
    # opt-in WS10: the adb-tcp worker behind the host AOA bridge + phone relay (adb stays the control plane)
    "aoa-bridge": MappingProxyType({
        "hardware": ("adb_usb_identity", "aoa_usb_identity", "host_usb_controller", "phone_kernel_release",
                     "phone_usb_serial", "phone_usb_sysfs_device"),
        "software": ("aoa_bridge_options_sha256", "host_binary_sha256", "host_bridge_sha256", "phone_relay_sha256",
                     "phone_shard_sha256", "phone_worker_sha256", "transport_client_source_sha256",
                     "worker_environment_sha256"),
        "receipts": ("usb-link-speed", "numerical-rows-1-2-4", "server-token-identity", "aoa-bridge-round-trip",
                     "aoa-bridge-byte-identity", "scheduler-launched-session"),
    }),
})
_TRANSPORT_GENERATIONS = MappingProxyType({"adb-tcp": ADB_TCP_TRANSPORT_GENERATION,
                                           "aoa-bridge": AOA_BRIDGE_TRANSPORT_GENERATION})


def _sha256_text(name: str, value: object) -> str:
    if type(value) is not str or len(value) != 71 or not value.startswith("sha256:") \
            or any(character not in "0123456789abcdef" for character in value[7:]):
        raise PhysicalAdapterError(name + " must be sha256:<64 hex>")
    return value


@dataclass(frozen=True)
class PhoneHelperTransportIdentity:
    """Exact hardware, software and receipts one helper phone's transport was qualified with."""

    device_id: str
    transport: str
    transport_generation: str
    minimum_usb_speed_mbps: int
    hardware_identity: Mapping[str, str]
    software_identity: Mapping[str, str]
    receipts: Mapping[str, str]  # receipt kind -> sha256 of the receipt file

    def __post_init__(self) -> None:
        requirements = IDENTITY_REQUIREMENTS.get(self.transport)
        if requirements is None:
            raise PhysicalAdapterError("phone helper identity transport is invalid")
        if self.transport in _TRANSPORT_GENERATIONS and self.transport_generation != _TRANSPORT_GENERATIONS[self.transport]:
            raise PhysicalAdapterError(self.transport + " identity generation is " + _TRANSPORT_GENERATIONS[self.transport])
        if type(self.minimum_usb_speed_mbps) is not int or self.minimum_usb_speed_mbps <= 0:
            raise PhysicalAdapterError("phone helper identity USB speed floor is invalid")
        for name, required in (("hardware", requirements["hardware"]), ("software", requirements["software"])):
            values = dict(getattr(self, name + "_identity"))
            missing = sorted(set(required) - set(values))
            if missing or any(type(value) is not str or not value for value in values.values()):
                raise PhysicalAdapterError(f"phone helper {name} identity lacks " + ",".join(missing))
            object.__setattr__(self, name + "_identity", MappingProxyType(dict(sorted(values.items()))))
        for key, value in self.software_identity.items():
            if key.endswith("sha256") or ":" in key:
                _sha256_text("phone helper software " + key, value)
        if self.transport in ("adb-tcp", "aoa-bridge") and not any(
            key.startswith("phone_library_sha256:") for key in self.software_identity
        ):
            raise PhysicalAdapterError(self.transport + " identity needs phone_library_sha256:<name> entries")
        receipts = {kind: _sha256_text("phone helper receipt " + kind, value)
                    for kind, value in dict(self.receipts).items()}
        object.__setattr__(self, "receipts", MappingProxyType(dict(sorted(receipts.items()))))

    @property
    def missing_receipts(self) -> tuple[str, ...]:
        return tuple(kind for kind in IDENTITY_REQUIREMENTS[self.transport]["receipts"]
                     if kind not in self.receipts)

    @property
    def qualified(self) -> bool:
        return not self.missing_receipts

    def to_json(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "hardware_identity": dict(self.hardware_identity),
            "minimum_usb_speed_mbps": self.minimum_usb_speed_mbps,
            "receipts": dict(self.receipts),
            "schema": PHONE_HELPER_TRANSPORT_IDENTITY_SCHEMA,
            "software_identity": dict(self.software_identity),
            "transport": self.transport,
            "transport_generation": self.transport_generation,
        }

    @property
    def identity_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    @classmethod
    def from_json(cls, value: object) -> "PhoneHelperTransportIdentity":
        if type(value) is not dict or value.get("schema") != PHONE_HELPER_TRANSPORT_IDENTITY_SCHEMA:
            raise PhysicalAdapterError("phone helper transport identity schema differs")
        return cls(**{key: value.get(key) for key in (
            "device_id", "transport", "transport_generation", "minimum_usb_speed_mbps",
            "hardware_identity", "software_identity", "receipts")})


def validate_helper_identities(
    helpers: Sequence[PhoneHelperBinding],
    identities: Mapping[str, PhoneHelperTransportIdentity],
    observed_usb: Mapping[str, "UsbPortObservation"],
) -> None:
    """Every helper runs on the qualified serial, port and speed; none is qualified from another's receipts."""
    validate_disjoint_ownership(helpers)
    for row in helpers:
        identity = identities.get(row.device_id)
        if identity is None or identity.device_id != row.device_id:
            raise PhysicalAdapterError(f"phone helper {row.device_id} has no transport identity of its own")
        if not identity.qualified:
            raise PhysicalAdapterError(
                f"phone helper {row.device_id} lacks receipts: " + ",".join(identity.missing_receipts)
            )
        observed = observed_usb.get(row.device_id)
        if (
            observed is None or observed.serial != row.serial
            or identity.hardware_identity["phone_usb_serial"] != row.serial
            or identity.hardware_identity["phone_usb_sysfs_device"] != observed.sysfs_device
            or observed.negotiated_speed_mbps < identity.minimum_usb_speed_mbps
        ):
            raise PhysicalAdapterError(f"phone helper {row.device_id} USB link differs from its qualification")
    digests = [identities[row.device_id].identity_sha256 for row in helpers]
    if len(set(digests)) != len(digests):
        raise PhysicalAdapterError("phone helpers share one transport identity")


# --- host USB topology (sysfs, read-only) --------------------------------------------------

@dataclass(frozen=True)
class UsbPortObservation:
    sysfs_device: str
    serial: str
    vendor_product: str
    negotiated_speed_mbps: int
    host_controller: str
    root_port: str
    behind_hub: bool

    def to_json(self) -> dict[str, object]:
        return {
            "behind_hub": self.behind_hub,
            "host_controller": self.host_controller,
            "negotiated_speed_mbps": self.negotiated_speed_mbps,
            "root_port": self.root_port,
            "serial": self.serial,
            "sysfs_device": self.sysfs_device,
            "vendor_product": self.vendor_product,
        }


def observe_usb_port(sysfs_device: str, *, sysfs_root: Path = Path("/sys/bus/usb/devices")) -> UsbPortObservation:
    """Facts of one USB device (e.g. ``2-9.2``) from sysfs; works in ADB and FunctionFS modes."""
    bus, separator, path = sysfs_device.partition("-")
    device = sysfs_root / sysfs_device
    if not separator or not bus.isdigit() or "/" in sysfs_device or not device.is_dir():
        raise PhysicalAdapterError("USB sysfs device is absent: " + sysfs_device)
    try:
        read = lambda name: (device / name).read_text(encoding="ascii").strip()  # noqa: E731
        serial, vendor, product = read("serial"), read("idVendor"), read("idProduct")
        speed = int(float(read("speed")))
    except (OSError, ValueError) as error:
        raise PhysicalAdapterError("USB sysfs device is unreadable: " + sysfs_device) from error
    # /sys/devices/pci0000:00/0000:00:14.0/usb2/2-9/2-9.2 -> 0000:00:14.0
    controllers = [part for part in Path(os.path.realpath(device)).parts
                   if part.count(":") == 2 and "." in part]
    if not controllers:
        raise PhysicalAdapterError("USB host controller is unresolved for " + sysfs_device)
    return UsbPortObservation(
        sysfs_device=sysfs_device, serial=serial, vendor_product=f"{vendor}:{product}".lower(),
        negotiated_speed_mbps=speed, host_controller=controllers[-1],
        root_port=bus + "-" + path.split(".")[0], behind_hub="." in path,
    )


def find_usb_device_by_serial(serial: str, *, sysfs_root: Path = Path("/sys/bus/usb/devices")) -> str | None:
    """sysfs name (``2-2``) of the one USB device with this serial, or None."""
    matches = []
    for device in sorted(sysfs_root.glob("*-*")):
        try:
            if ":" not in device.name and (device / "serial").read_text(encoding="ascii").strip() == serial:
                matches.append(device.name)
        except (OSError, UnicodeDecodeError):
            continue
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class UsbTopologyCheck:
    name: str
    passed: bool
    detail: str

    def to_json(self) -> dict[str, object]:
        return {"detail": self.detail, "name": self.name, "passed": self.passed}


def check_usb_topology(
    phones: Sequence[tuple[str, str, str | None, int]],
    *,
    sysfs_root: Path = Path("/sys/bus/usb/devices"),
) -> tuple[tuple[UsbTopologyCheck, ...], Mapping[str, UsbPortObservation]]:
    """Preflight rows for ``(device_id, serial, sysfs_device, minimum_speed_mbps)`` phones.

    A phone without a declared port (the legacy primary) is found by its serial.

    Each phone must answer at its declared port with its serial and speed floor; two phones on
    one root port (a shared hub uplink) fail. A shared xHCI controller is reported, not refused:
    decode calls carry about 10-40 KiB each and are latency-bound.
    """
    checks, observed = [], {}
    for device_id, serial, sysfs_device, minimum_speed in phones:
        name = "phone-usb-port:" + device_id
        sysfs_device = sysfs_device or find_usb_device_by_serial(serial, sysfs_root=sysfs_root)
        try:
            if sysfs_device is None:
                raise PhysicalAdapterError("no unique USB device has serial " + serial)
            row = observe_usb_port(sysfs_device, sysfs_root=sysfs_root)
        except PhysicalAdapterError as error:
            checks.append(UsbTopologyCheck(name, False, str(error)))
            continue
        observed[device_id] = row
        checks.append(UsbTopologyCheck(
            name, row.serial == serial and row.negotiated_speed_mbps >= minimum_speed,
            f"{row.sysfs_device} serial={row.serial} id={row.vendor_product} {row.negotiated_speed_mbps}M "
            f"controller={row.host_controller} root_port={row.root_port} hub={'yes' if row.behind_hub else 'no'}",
        ))
    if len(phones) > 1:
        roots = [row.root_port for row in observed.values()]
        controllers = sorted({row.host_controller for row in observed.values()})
        checks.append(UsbTopologyCheck(
            "phone-usb-topology",
            len(observed) == len(phones) and len(set(roots)) == len(roots),
            "controllers " + ",".join(controllers) + "; root ports " + ",".join(roots),
        ))
    return tuple(checks), MappingProxyType(observed)
