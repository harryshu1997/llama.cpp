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
# WS23 (2026-10-05): a helper-led (anchored) composite carries the model's co-helper declaration under this key ONLY
# so the campaign can build the helper worker lifecycle when no primary-anchored composite exists; route generation,
# costing and the launch read phone_co_helpers_v1 and must not see a co-helper on the led route (the helper is the route)
ANCHORED_CO_HELPERS_PARAMETER = "anchored_co_helpers_v1"
PHONE_HELPERS_PARAMETER = "phone_helpers"
# WS23 opt-in composite parameter: every helper phone of the route is its own resource (see route_generation.common)
PER_HELPER_OUTAGE_PARAMETER = "per_helper_outage"
# WS23 opt-in composite parameter: the route's ticket phone is a static co-helper's worker (a helper-anchored route)
STATIC_HELPER_ANCHOR_PARAMETER = "static_helper_anchor"
PHONE_CO_HELPERS_SCHEMA = "research-scheduler-phone-co-helpers-v1"
CO_HELPER_REQUEST_ID_RANGE = 1 << 24
MAXIMUM_CO_HELPERS = 7
_TRANSPORT_KEYS = frozenset({
    "adb_port",
    "adb_serial",
    "ffn_transport",
    "ffn_worker_host",
    "ffn_worker_port",
    "ffn_link_proxy_upstream_port",
    "ffn_link_transport",
    "phone_worker_port",
    "usb_transport_profile_id",
    "usb_transport_qualification_identity_sha256",
})
_BINDING_KEYS = frozenset({
    "device_id", "label", "layer_mask", "serial", "transport_parameters",
})
# WS15 optional binding keys, present only for a helper with fused attention-projection layers
_FUSED_BINDING_KEYS = frozenset({"fused_layer_mask", "fused_next_mask"})
# WS20 optional binding key (contract stub, no runtime support yet): per owned layer, the column share
# [offset, offset + width) this helper computes when another helper computes the rest of the same layer
_SHARE_BINDING_KEYS = frozenset({"column_shares"})
# WS21: the primary row (OP15 over FunctionFS) also carries the attention geometry that sizes its ring slots
_FUSED_GEOMETRY_KEYS = frozenset({"fused_n_q", "fused_n_kv"})
# remote-resident decode co-helpers (opt-in): a co-helper row capped below the server's maximum batch
# (S41_SERVER_FFN_HELPER<k>_MAX_TOKENS); its layers' prefill stays on the desktop
_MAX_TOKENS_BINDING_KEYS = frozenset({"max_tokens"})


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
    # WS15: layers whose full-width call also runs O(l) (+ layer l+1's Q/K/V for fused_next_mask)
    fused_layer_mask: int = 0
    fused_next_mask: int = 0
    # WS20 (contract stub): layer -> (column offset, column width) of this helper's share of a layer whose FFN
    # several helpers compute at once. Empty = whole layers (today's runtime). Offsets and widths are
    # multiples of the helper's column quantum; a shared layer must be an owned layer.
    column_shares: Mapping[int, tuple[int, int]] = MappingProxyType({})

    def __post_init__(self) -> None:
        _text("co-helper device", self.device_id)
        for name in ("fused_layer_mask", "fused_next_mask"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < 1 << 64:
                raise RuntimePlanError("co-helper " + name + " must be a 64-bit mask")
        if self.fused_layer_mask & ~self.layer_mask or self.fused_next_mask & ~self.fused_layer_mask:
            raise RuntimePlanError("co-helper fused layers must be its own layers (next within fused)")
        shares = {}
        for layer, share in dict(self.column_shares).items():
            if type(layer) is not int or not 0 <= layer < 64 or not self.layer_mask >> layer & 1:
                raise RuntimePlanError("co-helper column share names a layer it does not own")
            if type(share) not in (tuple, list) or len(share) != 2 or any(type(v) is not int for v in share):
                raise RuntimePlanError("co-helper column share must be (offset, width)")
            offset, width = share
            if offset < 0 or width <= 0 or offset % self.column_quantum or width % self.column_quantum:
                raise RuntimePlanError("co-helper column share must be quantum-aligned with a positive width")
            shares[layer] = (offset, width)
        object.__setattr__(self, "column_shares", MappingProxyType(dict(sorted(shares.items()))))
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
        if "ffn_link_proxy_upstream_port" in parameters:
            # opt-in link delay proxy: ffn_worker_port is the proxy, this the adb forward behind it
            if _port("co-helper link proxy upstream port", parameters["ffn_link_proxy_upstream_port"]) \
                    == parameters["ffn_worker_port"]:
                raise RuntimePlanError("co-helper link proxy must listen on its own port")
        if "ffn_link_transport" in parameters and (
            parameters["ffn_link_transport"] != "aoa-bridge" or "ffn_link_proxy_upstream_port" in parameters
        ):
            # opt-in WS10: the host AOA bridge serves ffn_worker_port; never combined with the delay proxy
            raise RuntimePlanError("co-helper link transport must be aoa-bridge without a link proxy")
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
            **self.fused_json(),
            **({"column_shares": {str(layer): list(share) for layer, share in self.column_shares.items()}}
               if self.column_shares else {}),
        }

    def fused_json(self) -> dict[str, int]:
        """The WS15 fused keys, only when this helper has fused layers (FFN-only JSON is unchanged)."""
        if not self.fused_layer_mask:
            return {}
        return {"fused_layer_mask": self.fused_layer_mask, "fused_next_mask": self.fused_next_mask}

    @classmethod
    def from_json(cls, value: object) -> "RuntimeCoHelperPhone":
        base = {
            "column_quantum", "device_id", "label", "layer_mask", "max_tokens",
            "resident_bytes", "serial", "session_id", "shard_sha256",
            "transport_parameters",
        }
        if type(value) is not dict or not base <= set(value) \
                or not set(value) - base <= _FUSED_BINDING_KEYS | _SHARE_BINDING_KEYS \
                or type(value["transport_parameters"]) is not dict:
            raise RuntimePlanError("co-helper phone JSON is invalid")
        row = dict(value)
        if "column_shares" in row:
            shares = row["column_shares"]
            if type(shares) is not dict or any(not str(k).isdigit() for k in shares):
                raise RuntimePlanError("co-helper column_shares must map layer -> [offset, width]")
            row["column_shares"] = {int(k): tuple(v) if type(v) is list else v for k, v in shares.items()}
        return cls(**row)


_PRIMARY_FUSED_KEYS = frozenset({"primary_fused_layer_mask", "primary_fused_next_mask",
                                 "primary_fused_n_q", "primary_fused_n_kv"})


@dataclass(frozen=True)
class RuntimeCoHelperDeclaration:
    """The primary phone's launch identity plus every static co-helper, in helper order."""

    primary_label: str
    primary_serial: str
    helpers: tuple[RuntimeCoHelperPhone, ...]
    # WS21: layers of the primary phone whose full-width call runs fused (from its shard index);
    # 0 keeps the declaration and every identity built on it byte-identical
    primary_fused_layer_mask: int = 0
    primary_fused_next_mask: int = 0
    primary_fused_n_q: int = 0
    primary_fused_n_kv: int = 0

    def __post_init__(self) -> None:
        _label("primary helper label", self.primary_label)
        _text("primary helper serial", self.primary_serial)
        for name in ("primary_fused_layer_mask", "primary_fused_next_mask", "primary_fused_n_q", "primary_fused_n_kv"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value >= 1 << 64:
                raise RuntimePlanError("co-helper declaration primary fused field is invalid")
        if self.primary_fused_next_mask & ~self.primary_fused_layer_mask:
            raise RuntimePlanError("co-helper declaration primary fused next mask exceeds the fused mask")
        if (self.primary_fused_layer_mask != 0) != (self.primary_fused_n_q > 0 and self.primary_fused_n_kv > 0):
            raise RuntimePlanError("co-helper declaration primary fused geometry is incomplete")
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

    def phone_helpers(
        self, primary_device_id: str, primary_layer_mask: int, *, maximum_tokens: int | None = None,
    ) -> str:
        """Canonical ``phone_helpers`` launch binding for one primary layer set.

        ``maximum_tokens`` (opt-in, remote-resident decode co-helpers): the server's maximum batch;
        a co-helper with fewer rows carries its own ``max_tokens`` cap in its row.
        """
        _text("primary helper device", primary_device_id)
        _layer_mask("primary helper layer mask", primary_layer_mask)
        if maximum_tokens is not None and (
            _integer("helper maximum batch", maximum_tokens, 1) < max(
                (row.max_tokens for row in self.helpers), default=1)
        ):
            raise RuntimePlanError("co-helper batch exceeds the server's maximum batch")
        if primary_device_id in self.device_ids or primary_layer_mask & self.layer_mask:
            raise RuntimePlanError("primary phone overlaps a co-helper")
        primary_fused = self.primary_fused_layer_mask & primary_layer_mask
        primary_next = self.primary_fused_next_mask & primary_fused
        return _canonical([
            {
                "device_id": primary_device_id,
                "label": self.primary_label,
                "layer_mask": primary_layer_mask,
                "serial": self.primary_serial,
                "transport_parameters": {},
                **({"fused_layer_mask": primary_fused, "fused_next_mask": primary_next,
                    "fused_n_q": self.primary_fused_n_q, "fused_n_kv": self.primary_fused_n_kv}
                   if primary_fused else {}),
            },
            *(
                {
                    "device_id": row.device_id,
                    "label": row.label,
                    "layer_mask": row.layer_mask,
                    "serial": row.serial,
                    "transport_parameters": dict(row.transport_parameters),
                    **row.fused_json(),
                    **({"max_tokens": row.max_tokens}
                       if maximum_tokens is not None and row.max_tokens < maximum_tokens else {}),
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
            **({"primary_fused_layer_mask": self.primary_fused_layer_mask,
                "primary_fused_next_mask": self.primary_fused_next_mask,
                "primary_fused_n_q": self.primary_fused_n_q,
                "primary_fused_n_kv": self.primary_fused_n_kv}
               if self.primary_fused_layer_mask else {}),
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
            or set(value) - _PRIMARY_FUSED_KEYS != {"helpers", "primary_label", "primary_serial", "schema"}
            or not (set(value) & _PRIMARY_FUSED_KEYS) in (set(), _PRIMARY_FUSED_KEYS)
            or type(value["helpers"]) is not list
        ):
            raise RuntimePlanError("co-helper declaration JSON is invalid")
        return cls(
            primary_label=value["primary_label"],
            primary_serial=value["primary_serial"],
            helpers=tuple(RuntimeCoHelperPhone.from_json(row) for row in value["helpers"]),
            primary_fused_layer_mask=value.get("primary_fused_layer_mask", 0),
            primary_fused_next_mask=value.get("primary_fused_next_mask", 0),
            primary_fused_n_q=value.get("primary_fused_n_q", 0),
            primary_fused_n_kv=value.get("primary_fused_n_kv", 0),
        )


def anchored_co_helper_declaration(
    parameters: Mapping[str, object],
) -> RuntimeCoHelperDeclaration | None:
    """Decode the ``anchored_co_helpers_v1`` carrier of a helper-led composite, or None when absent."""
    return co_helper_declaration(parameters, key=ANCHORED_CO_HELPERS_PARAMETER)


def co_helper_declaration(
    parameters: Mapping[str, object],
    key: str = PHONE_CO_HELPERS_PARAMETER,
) -> RuntimeCoHelperDeclaration | None:
    """Decode the canonical ``phone_co_helpers_v1`` parameter, or None when absent."""
    raw = parameters.get(key)
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
            or not _BINDING_KEYS <= set(row)
            or not set(row) - _BINDING_KEYS
                <= _FUSED_BINDING_KEYS | _FUSED_GEOMETRY_KEYS | _MAX_TOKENS_BINDING_KEYS
            or type(row["transport_parameters"]) is not dict
            or (index == 0) != (not row["transport_parameters"])
            or ("max_tokens" in row and (
                index == 0 or type(row["max_tokens"]) is not int or row["max_tokens"] < 1))
        ):
            raise RuntimePlanError("phone helper binding row is invalid")
        _text("phone helper device", row["device_id"])
        _text("phone helper serial", row["serial"])
        _label("phone helper label", row["label"])
        mask = _layer_mask("phone helper layer mask", row["layer_mask"])
        if set(row) & (_FUSED_BINDING_KEYS | _FUSED_GEOMETRY_KEYS):
            fused = row.get("fused_layer_mask")
            following = row.get("fused_next_mask")
            if set(row) & _FUSED_BINDING_KEYS != _FUSED_BINDING_KEYS \
                    or type(fused) is not int or type(following) is not int \
                    or not 0 < fused < 1 << 64 or fused & ~mask or following < 0 or following & ~fused:
                raise RuntimePlanError("phone helper fused layers are invalid")
            # WS21: the primary row needs n_q/n_kv (FunctionFS slot sizing); co-helper rows (TCP workers) carry none
            geometry = set(row) & _FUSED_GEOMETRY_KEYS
            if (index == 0) != (geometry == _FUSED_GEOMETRY_KEYS) or any(
                    type(row[key]) is not int or not 0 < row[key] < 1 << 32 for key in geometry):
                raise RuntimePlanError("phone helper fused geometry is invalid")
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
        and live.get("max_tokens") == row.get("max_tokens")
        and row["layer_mask"] & ~live["layer_mask"] == 0
        # WS21: fused-ness is fixed at launch; the request must ask exactly what the live server runs on its layers
        and row.get("fused_layer_mask", 0) == live.get("fused_layer_mask", 0) & row["layer_mask"]
        and row.get("fused_next_mask", 0) == live.get("fused_next_mask", 0) & row.get("fused_layer_mask", 0)
        for live, row in zip(resident_rows, requested_rows)
    )


def phone_helpers_with_primary_layers(value: object, primary_layer_mask: int) -> str:
    """The same binding with the ticket's phone owning ``primary_layer_mask``."""
    rows = [dict(row) for row in phone_helper_rows(value)]
    rows[0]["layer_mask"] = primary_layer_mask
    if rows[0].get("fused_layer_mask"):
        # WS21: the primary's fused layers follow its layers; none left -> an FFN-only row
        rows[0]["fused_layer_mask"] &= primary_layer_mask
        rows[0]["fused_next_mask"] &= rows[0]["fused_layer_mask"]
        if not rows[0]["fused_layer_mask"]:
            for key in _FUSED_BINDING_KEYS | _FUSED_GEOMETRY_KEYS:
                rows[0].pop(key, None)
    result = _canonical(rows)
    phone_helper_rows(result)
    return result
