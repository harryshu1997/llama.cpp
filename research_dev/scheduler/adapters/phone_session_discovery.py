"""Materialize phone residency sessions from a raw backend probe."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from types import MappingProxyType

from .._internal.placement import MemoryPoolProfile, PlacementHardwareProfile
from .._internal.runtime_capabilities import RuntimePhoneSessionCapability
from .contracts import PhysicalAdapterError


PHONE_SESSION_DISCOVERY_SCHEMA = (
    "research-scheduler-phone-session-discovery-v1"
)


@dataclass(frozen=True)
class PhoneSessionDiscoveryConfiguration:
    device_id: str
    endpoint_prefix: str
    worker_identity_sha256: str
    shared_compute_resource_id: str
    shared_transport_resource_ids: tuple[str, ...]
    phone_wide_memory_resource_id: str
    phone_wide_limit_bytes: int
    supported_layer_mask: int
    maximum_columns: int
    column_quantum: int
    supported_data_types: tuple[str, ...]
    batch_plans: tuple[str, ...]

    def __post_init__(self) -> None:
        text_values = (
            self.device_id,
            self.endpoint_prefix,
            self.shared_compute_resource_id,
            self.phone_wide_memory_resource_id,
            *self.shared_transport_resource_ids,
            *self.supported_data_types,
            *self.batch_plans,
        )
        if any(
            type(value) is not str or not value or not value.isascii()
            for value in text_values
        ):
            raise PhysicalAdapterError(
                "phone session discovery configuration is invalid"
            )
        if (
            not self.worker_identity_sha256.startswith("sha256:")
            or len(self.worker_identity_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.worker_identity_sha256[7:]
            )
            or type(self.phone_wide_limit_bytes) is not int
            or self.phone_wide_limit_bytes <= 0
            or type(self.supported_layer_mask) is not int
            or not 0 < self.supported_layer_mask < 1 << 64
            or type(self.maximum_columns) is not int
            or self.maximum_columns <= 0
            or type(self.column_quantum) is not int
            or self.column_quantum <= 0
            or self.maximum_columns % self.column_quantum
        ):
            raise PhysicalAdapterError(
                "phone session discovery limits are invalid"
            )
        object.__setattr__(
            self,
            "shared_transport_resource_ids",
            tuple(sorted(set(self.shared_transport_resource_ids))),
        )


def parse_phone_session_probe(
    output: str,
    configuration: PhoneSessionDiscoveryConfiguration,
) -> tuple[RuntimePhoneSessionCapability, ...]:
    """Convert successful retained allocations into session capabilities."""

    if type(output) is not str or not isinstance(
        configuration, PhoneSessionDiscoveryConfiguration
    ):
        raise PhysicalAdapterError("phone session probe input is invalid")
    session_rows = []
    resident = []
    complete = None
    for raw in output.splitlines():
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if type(row) is not dict:
            continue
        if row.get("event") == "resident":
            resident.append(row)
            session_rows.append(row)
        elif row.get("event") == "unavailable":
            session_rows.append(row)
        elif row.get("event") == "complete":
            complete = row
    if (
        complete is None
        or complete.get("status") != "PASS"
        or complete.get("resident_sessions") != len(resident)
        or not resident
    ):
        raise PhysicalAdapterError("phone session discovery failed closed")
    sessions = []
    resident_total = 0
    for index, row in enumerate(session_rows):
        session_id = row.get("backend")
        ready = row.get("event") == "resident"
        allocated_bytes = row.get(
            "allocated_bytes" if ready else "requested_bytes"
        )
        reason = None if ready else row.get("reason")
        if (
            type(session_id) is not str
            or not session_id
            or not session_id.isascii()
            or type(allocated_bytes) is not int
            or allocated_bytes <= 0
            or row.get("session_index", index) != index
            or (
                not ready
                and (
                    type(reason) is not str
                    or not reason
                    or not reason.isascii()
                )
            )
        ):
            raise PhysicalAdapterError(
                "phone session discovery row is invalid"
            )
        if ready:
            resident_total += allocated_bytes
        sessions.append(RuntimePhoneSessionCapability(
            session_id=session_id,
            device_id=configuration.device_id,
            endpoint=configuration.endpoint_prefix + session_id,
            worker_identity_sha256=(
                configuration.worker_identity_sha256
            ),
            memory_resource_id=(
                configuration.phone_wide_memory_resource_id
                + ":session:"
                + session_id
            ),
            resident_memory_limit_bytes=allocated_bytes,
            shared_compute_resource_id=(
                configuration.shared_compute_resource_id
            ),
            shared_transport_resource_ids=(
                configuration.shared_transport_resource_ids
            ),
            supported_layer_mask=configuration.supported_layer_mask,
            maximum_columns=configuration.maximum_columns,
            column_quantum=configuration.column_quantum,
            supported_data_types=configuration.supported_data_types,
            batch_plans=configuration.batch_plans,
            ready=ready,
            residency_state="cold",
            unavailable_reason=reason,
        ))
    if resident_total > configuration.phone_wide_limit_bytes:
        raise PhysicalAdapterError(
            "phone session allocations exceed the phone-wide limit"
        )
    return tuple(sessions)


def add_phone_session_memory_pools(
    profile: PlacementHardwareProfile,
    sessions: tuple[RuntimePhoneSessionCapability, ...],
) -> PlacementHardwareProfile:
    """Add per-session constraint pools without changing phone-wide memory."""

    if not isinstance(profile, PlacementHardwareProfile):
        raise PhysicalAdapterError("phone session placement profile is invalid")
    pools = dict(profile.memory_pools)
    for session in sessions:
        current = pools.get(session.memory_resource_id)
        expected = MemoryPoolProfile(
            pool_id=session.memory_resource_id,
            capacity_bytes=session.resident_memory_limit_bytes,
            reserved_bytes=0,
        )
        if current is not None and current != expected:
            raise PhysicalAdapterError(
                "phone session memory pool identity differs"
            )
        pools[session.memory_resource_id] = expected
    return replace(
        profile,
        memory_pools=MappingProxyType(dict(sorted(pools.items()))),
    )


def phone_session_discovery_json(
    sessions: tuple[RuntimePhoneSessionCapability, ...],
) -> dict[str, object]:
    if not sessions:
        raise PhysicalAdapterError("phone session discovery is empty")
    return {
        "schema": PHONE_SESSION_DISCOVERY_SCHEMA,
        "sessions": [row.to_json() for row in sessions],
    }


def phone_sessions_from_json(
    value: object,
) -> tuple[RuntimePhoneSessionCapability, ...]:
    if type(value) is not dict or value.get("schema") != (
        PHONE_SESSION_DISCOVERY_SCHEMA
    ) or type(value.get("sessions")) is not list:
        raise PhysicalAdapterError("phone session discovery schema differs")
    sessions = tuple(
        RuntimePhoneSessionCapability.from_json(row)
        for row in value["sessions"]
    )
    if not sessions:
        raise PhysicalAdapterError("phone session discovery is empty")
    return sessions
