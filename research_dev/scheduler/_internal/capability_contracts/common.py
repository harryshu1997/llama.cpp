"""Device, executor, catalog and snapshot capability contracts: common."""

from __future__ import annotations

from typing import Mapping

from ..placement import PLACEMENT_PROFILE_SCHEMA, PlacementHardwareProfile


RUNTIME_CAPABILITY_SCHEMA = "research-scheduler-runtime-capabilities-v1"


RUNTIME_SYSTEM_SNAPSHOT_SCHEMA = "research-scheduler-system-snapshot-v1"


RESIDENCY_STATES = frozenset({"cold", "hot", "warm"})


SPLIT_AXES = frozenset({"column", "row", "tensor"})


COMPOSITE_ROUTE_FAMILIES = frozenset({
    "layer_placement",
    "operator_offload",
    "operator_split",
})


GENERATED_ROUTE_FAMILIES = frozenset({
    "whole_model",
    *COMPOSITE_ROUTE_FAMILIES,
})


_MATURITY_RANK = {
    "QUARANTINED": 0,
    "PRIOR_ONLY": 1,
    "CALIBRATION_PENDING": 2,
    "SHADOW": 3,
    "QUALIFIED": 4,
}


PHONE_POWER_ESTIMATION_VERSION = "assumed-phone-power-v1"


PHONE_POWER_EVIDENCE_ASSUMED_4P5W = "ASSUMED_4P5W"


class RuntimeCapabilityError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeCapabilityError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeCapabilityError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RuntimeCapabilityError(f"{name} must be bool")
    return value


def _object(name: str, value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise RuntimeCapabilityError(f"{name} must be an object")
    return value


def _list(name: str, value: object) -> list[object]:
    if type(value) is not list:
        raise RuntimeCapabilityError(f"{name} must be a list")
    return value


def _placement_profile_json(
    profile: PlacementHardwareProfile,
) -> dict[str, object]:
    return {
        "devices": [
            {
                "allocation_limit_bytes": row.allocation_limit_bytes,
                "device_id": row.device_id,
                "kind": row.kind,
                "memory_pool_id": row.memory_pool_id,
                "ready": row.ready,
            }
            for row in sorted(
                profile.devices.values(), key=lambda item: item.device_id
            )
        ],
        "domains": [
            {
                "domain_id": row.domain.domain_id,
                "evidence_ids": list(row.evidence_ids),
                "idle_power_mw": row.domain.idle_power_mw,
                "status": row.status,
            }
            for row in sorted(
                profile.domains.values(),
                key=lambda item: item.domain.domain_id,
            )
        ],
        "energy_boundary_id": profile.energy_boundary_id,
        "idle_charge_domains": sorted(profile.idle_charge_domains),
        "kernels": [
            {
                "active_power_mw": row.kernel.active_power_mw,
                "device_id": row.device_id,
                "domain_id": row.kernel.domain_id,
                "effective_bytes_per_s": row.kernel.effective_bytes_per_s,
                "effective_ops_per_s": row.kernel.effective_ops_per_s,
                "evidence_ids": list(row.evidence_ids),
                "kernel_id": row.kernel.kernel_id,
                "launch_us": row.kernel.launch_us,
                "profile_id": row.profile_id,
                "status": row.status,
            }
            for row in sorted(
                profile.kernels.values(), key=lambda item: item.profile_id
            )
        ],
        "links": [
            {
                "bandwidth_bytes_per_s": row.bandwidth_bytes_per_s,
                "allocator": row.allocator,
                "concurrent_streams": row.concurrent_streams,
                "domain_active_power_mw": dict(row.domain_active_power_mw),
                "dynamic_pj_per_byte": row.dynamic_pj_per_byte,
                "evidence_ids": list(row.evidence_ids),
                "fixed_dynamic_uj": row.fixed_dynamic_uj,
                "fixed_latency_us": row.fixed_latency_us,
                "full_duplex": row.full_duplex,
                "link_id": row.link_id,
                "maximum_payload_bytes": row.maximum_payload_bytes,
                "minimum_payload_bytes": row.minimum_payload_bytes,
                "queue_depth": row.queue_depth,
                "ready": row.ready,
                "source_device": row.source_device,
                "status": row.status,
                "target_device": row.target_device,
                "transport_generation": row.transport_generation,
                "transport_profile_id": row.transport_profile_id,
                "usbfs_available_bytes": row.usbfs_available_bytes,
                "slot_safety_bytes": row.slot_safety_bytes,
                **(
                    {}
                    if row.qualification_identity_sha256 is None
                    else {
                        "qualification_identity_sha256": (
                            row.qualification_identity_sha256
                        )
                    }
                ),
            }
            for row in sorted(profile.links, key=lambda item: item.link_id)
        ],
        "memory_pools": [
            {
                "capacity_bytes": row.capacity_bytes,
                "pool_id": row.pool_id,
                "reserved_bytes": row.reserved_bytes,
            }
            for row in sorted(
                profile.memory_pools.values(), key=lambda item: item.pool_id
            )
        ],
        "profile_id": profile.profile_id,
        "schema": PLACEMENT_PROFILE_SCHEMA,
    }


def _texts(name: str, values: tuple[str, ...], *, allow_empty: bool = False):
    result = tuple(_text(name, value) for value in values)
    if (not result and not allow_empty) or len(result) != len(set(result)):
        raise RuntimeCapabilityError(f"{name} must be unique")
    return result


def _fractions(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
    result = tuple(values)
    if (
        len(result) != len(set(result))
        or any(type(value) is not int or not 0 < value < 1_000_000 for value in result)
    ):
        raise RuntimeCapabilityError(f"{name} must contain unique proper fractions")
    return tuple(sorted(result))
