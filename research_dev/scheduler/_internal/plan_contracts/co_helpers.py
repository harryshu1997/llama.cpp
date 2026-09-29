"""Execution-plan contracts grouped by responsibility: static FFN co-helper phones.

Stage A of two-phone dispatch. A phone-assisted route keeps one scheduled primary phone (the
composite's ``helper_device_id`` with its sessions, shards and re-provisioning) and may add
co-helper phones that own disjoint whole FFN layers of the same model for a whole trace. The
scheduler never prepares, replaces or evicts a co-helper; its worker lifecycle belongs to the rig.

Two adapter parameters carry it:

* ``phone_co_helpers_v1`` on a catalog composite: :class:`RuntimeCoHelperDeclaration`;
* ``phone_helpers`` on a plan and inside a dormant desktop contract: the llama-server launch
  binding (``adapters.phone_helpers.PhoneHelperBinding`` rows). The first row is the ticket's own
  phone without transport parameters; helper ``k`` numbers its calls from ``1 + k * 2^24``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lcm
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from .common import RuntimePlanError, _integer, _sha256, _text


PHONE_CO_HELPERS_PARAMETER = "phone_co_helpers_v1"
PHONE_HELPERS_PARAMETER = "phone_helpers"
PHONE_CO_HELPERS_SCHEMA = "research-scheduler-phone-co-helpers-v1"
CO_HELPER_REQUEST_ID_RANGE = 1 << 24
MAXIMUM_CO_HELPERS = 7
_TRANSPORT_KEYS = frozenset({
    "adb_port",
    "adb_serial",
    "ffn_transport",
    "ffn_worker_host",
    "ffn_worker_port",
    "phone_worker_port",
    "usb_transport_profile_id",
    "usb_transport_qualification_identity_sha256",
})
_BINDING_KEYS = frozenset({
    "device_id", "label", "layer_mask", "serial", "transport_parameters",
})


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _label(name: str, value: object) -> str:
    value = _text(name, value)
    if len(value) > 32 or any(
        not (character.isalnum() or character in "-_") for character in value
    ):
        raise RuntimePlanError(name + " must be 1-32 [A-Za-z0-9_-] characters")
    return value


def _layer_mask(name: str, value: object) -> int:
    if type(value) is not int or not 0 < value < 1 << 64:
        raise RuntimePlanError(name + " must be a nonzero 64-bit mask")
    return value


def _port(name: str, value: object) -> int:
    if type(value) is not int or not 0 < value <= 65535:
        raise RuntimePlanError(name + " must be a TCP port")
    return value


@dataclass(frozen=True)
class RuntimeCoHelperPhone:
    """One co-helper phone: a protocol-v6 FFN worker behind ``adb forward``."""

    device_id: str
    serial: str
    label: str
    session_id: str
    layer_mask: int
    column_quantum: int
    max_tokens: int
    shard_sha256: str
    resident_bytes: int
    transport_parameters: Mapping[str, int | str]

    def __post_init__(self) -> None:
        _text("co-helper device", self.device_id)
        _text("co-helper serial", self.serial)
        _label("co-helper label", self.label)
        _label("co-helper session", self.session_id)
        _layer_mask("co-helper layer mask", self.layer_mask)
        if _integer("co-helper column quantum", self.column_quantum, 32) % 32:
            raise RuntimePlanError("co-helper column quantum must be a multiple of 32")
        _integer("co-helper maximum tokens", self.max_tokens, 1)
        _sha256("co-helper shard", self.shard_sha256)
        _integer("co-helper resident bytes", self.resident_bytes, 1)
        parameters = dict(self.transport_parameters)
        if (
            set(parameters) - _TRANSPORT_KEYS
            or parameters.get("ffn_transport") != "adb-tcp"
            or parameters.get("adb_serial") != self.serial
            or any(type(key) is not str for key in parameters)
            or any(type(value) not in (int, str) for value in parameters.values())
        ):
            raise RuntimePlanError("co-helper transport must be its own adb-tcp forward")
        _port("co-helper adb port", parameters.get("adb_port"))
        _port("co-helper forward port", parameters.get("ffn_worker_port"))
        _port("co-helper phone port", parameters.get("phone_worker_port"))
        _text("co-helper forward host", parameters.get("ffn_worker_host"))
        if "usb_transport_qualification_identity_sha256" in parameters:
            _sha256(
                "co-helper transport identity",
                parameters["usb_transport_qualification_identity_sha256"],
            )
        object.__setattr__(
            self, "transport_parameters",
            MappingProxyType(dict(sorted(parameters.items()))),
        )

    @property
    def endpoint(self) -> str:
        return "adb-tcp://" + self.serial + "/" + self.session_id

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(index for index in range(64) if self.layer_mask >> index & 1)

    def to_json(self) -> dict[str, object]:
        return {
            "column_quantum": self.column_quantum,
            "device_id": self.device_id,
            "label": self.label,
            "layer_mask": self.layer_mask,
            "max_tokens": self.max_tokens,
            "resident_bytes": self.resident_bytes,
            "serial": self.serial,
            "session_id": self.session_id,
            "shard_sha256": self.shard_sha256,
            "transport_parameters": dict(self.transport_parameters),
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeCoHelperPhone":
        if type(value) is not dict or set(value) != {
            "column_quantum", "device_id", "label", "layer_mask", "max_tokens",
            "resident_bytes", "serial", "session_id", "shard_sha256",
            "transport_parameters",
        } or type(value["transport_parameters"]) is not dict:
            raise RuntimePlanError("co-helper phone JSON is invalid")
        return cls(**value)


@dataclass(frozen=True)
class RuntimeCoHelperDeclaration:
    """The primary phone's launch identity plus every static co-helper, in helper order."""

    primary_label: str
    primary_serial: str
    helpers: tuple[RuntimeCoHelperPhone, ...]

    def __post_init__(self) -> None:
        _label("primary helper label", self.primary_label)
        _text("primary helper serial", self.primary_serial)
        helpers = tuple(self.helpers)
        if not 0 < len(helpers) <= MAXIMUM_CO_HELPERS or any(
            not isinstance(row, RuntimeCoHelperPhone) for row in helpers
        ):
            raise RuntimePlanError("co-helper declaration needs 1 to 7 phones")
        for values in (
            [row.device_id for row in helpers],
            [self.primary_serial, *(row.serial for row in helpers)],
            [self.primary_label, *(row.label for row in helpers)],
            [row.session_id for row in helpers],
        ):
            if len(set(values)) != len(values):
                raise RuntimePlanError("co-helper identities are not unique")
        union = 0
        for row in helpers:
            if row.layer_mask & union:
                raise RuntimePlanError("co-helper layer ownership overlaps")
            union |= row.layer_mask
        object.__setattr__(self, "helpers", helpers)

    @property
    def layer_mask(self) -> int:
        result = 0
        for row in self.helpers:
            result |= row.layer_mask
        return result

    @property
    def device_ids(self) -> tuple[str, ...]:
        return tuple(row.device_id for row in self.helpers)

    def helper_for_layer(self, layer_index: int) -> RuntimeCoHelperPhone | None:
        return next(
            (row for row in self.helpers if row.layer_mask >> layer_index & 1),
            None,
        )

    def column_quantum(self, primary_quantum: int) -> int:
        """Smallest width every client accepts (OP15 2176 with a 4352 Pixel -> 4352)."""
        result = _integer("primary column quantum", primary_quantum, 1)
        for row in self.helpers:
            result = lcm(result, row.column_quantum)
        return result

    def phone_helpers(self, primary_device_id: str, primary_layer_mask: int) -> str:
        """Canonical ``phone_helpers`` launch binding for one primary layer set."""
        _text("primary helper device", primary_device_id)
        _layer_mask("primary helper layer mask", primary_layer_mask)
        if primary_device_id in self.device_ids or primary_layer_mask & self.layer_mask:
            raise RuntimePlanError("primary phone overlaps a co-helper")
        return _canonical([
            {
                "device_id": primary_device_id,
                "label": self.primary_label,
                "layer_mask": primary_layer_mask,
                "serial": self.primary_serial,
                "transport_parameters": {},
            },
            *(
                {
                    "device_id": row.device_id,
                    "label": row.label,
                    "layer_mask": row.layer_mask,
                    "serial": row.serial,
                    "transport_parameters": dict(row.transport_parameters),
                }
                for row in self.helpers
            ),
        ])

    def to_json(self) -> dict[str, object]:
        return {
            "helpers": [row.to_json() for row in self.helpers],
            "primary_label": self.primary_label,
            "primary_serial": self.primary_serial,
            "schema": PHONE_CO_HELPERS_SCHEMA,
        }

    def encode(self) -> str:
        return _canonical(self.to_json())

    @property
    def declaration_sha256(self) -> str:
        return "sha256:" + hashlib.sha256(self.encode().encode("ascii")).hexdigest()

    @classmethod
    def from_json(cls, value: object) -> "RuntimeCoHelperDeclaration":
        if (
            type(value) is not dict
            or value.get("schema") != PHONE_CO_HELPERS_SCHEMA
            or set(value) != {"helpers", "primary_label", "primary_serial", "schema"}
            or type(value["helpers"]) is not list
        ):
            raise RuntimePlanError("co-helper declaration JSON is invalid")
        return cls(
            primary_label=value["primary_label"],
            primary_serial=value["primary_serial"],
            helpers=tuple(RuntimeCoHelperPhone.from_json(row) for row in value["helpers"]),
        )


def co_helper_declaration(
    parameters: Mapping[str, object],
) -> RuntimeCoHelperDeclaration | None:
    """Decode the canonical ``phone_co_helpers_v1`` parameter, or None when absent."""
    raw = parameters.get(PHONE_CO_HELPERS_PARAMETER)
    if raw is None:
        return None
    try:
        decoded = json.loads(raw) if type(raw) is str else None
    except ValueError:
        decoded = None
    declaration = RuntimeCoHelperDeclaration.from_json(decoded)
    if declaration.encode() != raw:
        raise RuntimePlanError("co-helper declaration is not canonical")
    return declaration


def phone_helper_rows(value: object) -> tuple[Mapping[str, object], ...]:
    """Validate a ``phone_helpers`` launch binding: disjoint, unique, first without transport."""
    try:
        rows = json.loads(value) if type(value) is str else None
    except ValueError:
        rows = None
    if type(rows) is not list or not 0 < len(rows) <= MAXIMUM_CO_HELPERS + 1:
        raise RuntimePlanError("phone helper binding is invalid")
    union = 0
    for index, row in enumerate(rows):
        if (
            type(row) is not dict
            or set(row) != _BINDING_KEYS
            or type(row["transport_parameters"]) is not dict
            or (index == 0) != (not row["transport_parameters"])
        ):
            raise RuntimePlanError("phone helper binding row is invalid")
        _text("phone helper device", row["device_id"])
        _text("phone helper serial", row["serial"])
        _label("phone helper label", row["label"])
        mask = _layer_mask("phone helper layer mask", row["layer_mask"])
        if mask & union:
            raise RuntimePlanError("phone helper layer ownership overlaps")
        union |= mask
    for name in ("device_id", "label", "serial"):
        if len({row[name] for row in rows}) != len(rows):
            raise RuntimePlanError("phone helper identities are not unique")
    return tuple(MappingProxyType(row) for row in rows)


def phone_helper_layer_masks(value: object) -> Mapping[str, int]:
    """Owned layer mask per device of a launch binding, in helper order."""
    return MappingProxyType({
        row["device_id"]: row["layer_mask"] for row in phone_helper_rows(value)
    })


def phone_helpers_support(resident: object, requested: object) -> bool:
    """A live server's helpers can serve the requested binding.

    The same devices, labels, serials and transports in the same helper order (the request-id
    ranges depend on it), each requested owner a subset of the resident owner's layers.
    """
    try:
        resident_rows = phone_helper_rows(resident)
        requested_rows = phone_helper_rows(requested)
    except RuntimePlanError:
        return False
    return len(resident_rows) == len(requested_rows) and all(
        live["device_id"] == row["device_id"]
        and live["label"] == row["label"]
        and live["serial"] == row["serial"]
        and live["transport_parameters"] == row["transport_parameters"]
        and row["layer_mask"] & ~live["layer_mask"] == 0
        for live, row in zip(resident_rows, requested_rows)
    )


def phone_helpers_with_primary_layers(value: object, primary_layer_mask: int) -> str:
    """The same binding with the ticket's phone owning ``primary_layer_mask``."""
    rows = [dict(row) for row in phone_helper_rows(value)]
    rows[0]["layer_mask"] = primary_layer_mask
    result = _canonical(rows)
    phone_helper_rows(result)
    return result
