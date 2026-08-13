"""Materialize measured kernel campaigns into scheduler profiles."""

from __future__ import annotations

from typing import Any, Mapping

from .policy import ResourceProfile
from .matmul import MatmulScheduleError, MatmulSystemProfile


GIB = 1024 * 1024 * 1024
KERNEL_CAMPAIGN_SCHEMA = "s42-kernel-energy-profile-v1"

__all__ = [
    "GIB",
    "KERNEL_CAMPAIGN_SCHEMA",
    "materialize_matmul_profile",
]


def _median(name: str, value: object) -> int:
    if type(value) is int:
        return value
    if type(value) is not dict or type(value.get("median")) is not int:
        raise MatmulScheduleError(f"{name} lacks an integer median")
    return value["median"]


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise MatmulScheduleError(f"{name} must be a positive integer")
    return value


def _rows(name: str, value: object) -> list[Mapping[str, Any]]:
    if type(value) is not list:
        raise MatmulScheduleError(f"{name} must be a list")
    rows: list[Mapping[str, Any]] = []
    for row in value:
        if type(row) is not dict:
            raise MatmulScheduleError(f"{name} contains a non-object row")
        rows.append(row)
    return rows


def _evidence_ids(value: object) -> list[str]:
    result: list[str] = []
    for row in _rows("campaign evidence", value):
        digest = row.get("sha256")
        if type(digest) is not str or not digest:
            raise MatmulScheduleError("campaign evidence lacks sha256")
        result.append(digest)
    if not result:
        raise MatmulScheduleError("campaign row has no evidence")
    return sorted(set(result))


def _resource_bindings(
    resource_ids: Mapping[str, str] | None,
    resource_profiles: Mapping[str, ResourceProfile] | None,
) -> dict[str, ResourceProfile]:
    default_ids = {
        "cpu": "compute:cpu",
        "gpu": "compute:gpu",
        "phone": "compute:phone",
        "pcie": "link:pcie",
        "usb": "link:usb",
    }
    supplied_ids = {} if resource_ids is None else dict(resource_ids)
    if supplied_ids:
        unknown = set(supplied_ids) - set(default_ids)
        if unknown:
            raise MatmulScheduleError(
                f"unknown matmul resource aliases: {sorted(unknown)}"
            )
        default_ids.update(supplied_ids)
    if any(type(item) is not str or not item for item in default_ids.values()):
        raise MatmulScheduleError("matmul resource ids must be non-empty strings")
    if len(set(default_ids.values())) != len(default_ids):
        raise MatmulScheduleError("matmul resource ids must be distinct")

    defaults = {
        "cpu": ("desktop_cpu", "cpu"),
        "gpu": ("desktop_gpu", "gpu"),
        "phone": ("phone_accelerator", "phone"),
        "pcie": ("transfer_link", "pcie"),
        "usb": ("transfer_link", "usb"),
    }
    result = {
        key: ResourceProfile(
            resource_id=default_ids[key],
            kind=kind,
            capacity=1,
            ready=True,
            identity=identity,
        )
        for key, (kind, identity) in defaults.items()
    }
    if resource_profiles is not None:
        unknown = set(resource_profiles) - set(result)
        if unknown:
            raise MatmulScheduleError(
                f"unknown shared resource profiles: {sorted(unknown)}"
            )
        for key, profile in resource_profiles.items():
            if not isinstance(profile, ResourceProfile):
                raise MatmulScheduleError(
                    "shared resources must contain ResourceProfile objects"
                )
            explicit_id = supplied_ids.get(key)
            if explicit_id is not None and explicit_id != profile.resource_id:
                raise MatmulScheduleError(
                    f"resource id and shared profile differ: {key}"
                )
            result[key] = profile
    if len({profile.resource_id for profile in result.values()}) != len(result):
        raise MatmulScheduleError("matmul resource bindings must be distinct")
    return result


def _kernel_range(backend: str) -> tuple[int, int]:
    if backend == "htp":
        return 512, 11_136
    if backend == "adreno":
        return 512, 512
    return 128, 0


def materialize_matmul_profile(
    source: Mapping[str, Any],
    *,
    generic_family: bool = False,
    gpu_capacity_bytes: int = 16 * GIB,
    phone_capacity_bytes: int = 10 * GIB,
    gpu_reserved_bytes: int = 0,
    phone_reserved_bytes: int = 0,
    resource_ids: Mapping[str, str] | None = None,
    resource_profiles: Mapping[str, ResourceProfile] | None = None,
) -> dict[str, object]:
    """Build a conservative shadow profile from a measured campaign.

    HTP and Adreno rows share one logical phone device. This prevents the
    planner from assuming that both accelerators, phone DRAM, and USB can be
    used independently before a concurrent-overlap profile is measured.
    """
    if type(source) is not dict or source.get("schema") != KERNEL_CAMPAIGN_SCHEMA:
        raise MatmulScheduleError("kernel campaign schema mismatch")
    resources = _resource_bindings(resource_ids, resource_profiles)

    capacities = (
        ("gpu capacity", gpu_capacity_bytes),
        ("phone capacity", phone_capacity_bytes),
    )
    reservations = (
        ("gpu reservation", gpu_reserved_bytes, gpu_capacity_bytes),
        ("phone reservation", phone_reserved_bytes, phone_capacity_bytes),
    )
    for name, value in capacities:
        _positive_int(name, value)
    for name, value, capacity in reservations:
        if type(value) is not int or value < 0 or value > capacity:
            raise MatmulScheduleError(
                f"{name} must be between zero and its capacity"
            )

    domains = [
        {
            "domain_id": row["domain_id"],
            "idle_power_mw": _median(
                f"domain {row['domain_id']} idle power", row.get("idle_power_mw")
            ),
        }
        for row in _rows("kernel campaign energy domains", source.get("energy_domains"))
    ]
    if not domains:
        raise MatmulScheduleError("kernel campaign has no energy domains")

    devices = [
        {
            "device_id": "cpu",
            "kind": "desktop_cpu",
            "resource_id": resources["cpu"].resource_id,
            "resource_kind": resources["cpu"].kind,
            "resource_capacity": resources["cpu"].capacity,
            "resource_identity": resources["cpu"].identity,
            "memory_capacity_bytes": 0,
            "reserved_bytes": 0,
            "ready": resources["cpu"].ready,
        },
        {
            "device_id": "gpu",
            "kind": "desktop_gpu",
            "resource_id": resources["gpu"].resource_id,
            "resource_kind": resources["gpu"].kind,
            "resource_capacity": resources["gpu"].capacity,
            "resource_identity": resources["gpu"].identity,
            "memory_capacity_bytes": gpu_capacity_bytes,
            "reserved_bytes": gpu_reserved_bytes,
            "ready": resources["gpu"].ready,
        },
        {
            "device_id": "phone",
            "kind": "phone_accelerator",
            "resource_id": resources["phone"].resource_id,
            "resource_kind": resources["phone"].kind,
            "resource_capacity": resources["phone"].capacity,
            "resource_identity": resources["phone"].identity,
            "memory_capacity_bytes": phone_capacity_bytes,
            "reserved_bytes": phone_reserved_bytes,
            "ready": resources["phone"].ready,
        },
    ]

    backend_device = {
        "cpu": "cpu",
        "cuda": "gpu",
        "htp": "phone",
        "adreno": "phone",
    }
    kernels = []
    included_backends: set[str] = set()
    for row in _rows("kernel campaign kernels", source.get("kernel_rows")):
        backend = row.get("backend")
        device_id = backend_device.get(backend)
        if device_id is None:
            continue
        included_backends.add(backend)
        shape = row.get("shape")
        if type(shape) is not dict:
            raise MatmulScheduleError("kernel campaign shape is invalid")
        m = _positive_int("kernel shape m", shape.get("m"))
        k = _positive_int("kernel shape k", shape.get("k"))
        n = _positive_int("kernel shape n", shape.get("n"))
        latency_us = _positive_int(
            "kernel latency",
            _median(f"kernel {row.get('profile_id')} latency", row.get("latency_us")),
        )
        resident_bytes = _positive_int(
            "kernel resident bytes", row.get("resident_bytes")
        )
        memory_bytes = resident_bytes + 4 * m * (k + n)
        power = {
            domain_id: _median(
                f"kernel {row.get('profile_id')} power {domain_id}", value
            )
            for domain_id, value in row.get("power_mw_by_domain", {}).items()
        }
        minimum_n, maximum_n = _kernel_range(backend)
        source_status = str(row.get("status", ""))
        measured = source_status.startswith("measured") and not generic_family
        kernels.append({
            "profile_id": (
                f"{row['profile_id']}-generic-shadow"
                if generic_family
                else row["profile_id"]
            ),
            "device_id": device_id,
            "kernel_family": "*" if generic_family else row["kernel_family"],
            "quantization": row["resident_type"],
            "shape": {"m": m, "k": 0 if generic_family else k, "n": n},
            "effective_ops_per_s": _positive_int(
                "kernel effective ops", row.get("effective_ops_per_s")
            ),
            "effective_bytes_per_s": max(
                1, memory_bytes * 1_000_000 // latency_us
            ),
            "launch_us": 0,
            "domain_power_mw": power,
            "minimum_n": minimum_n,
            "maximum_n": maximum_n,
            "n_quantum": 128,
            "status": "measured" if measured else "estimated",
            "evidence_ids": _evidence_ids(row.get("evidence")),
        })
    if not kernels:
        raise MatmulScheduleError("kernel campaign has no usable matmul rows")

    endpoint_device = {"cpu": "cpu", "cuda": "gpu", "phone-memory": "phone"}
    links = []
    for row in _rows(
        "kernel campaign derived links", source.get("derived_link_models")
    ):
        source_device = endpoint_device.get(row.get("source_device"))
        target_device = endpoint_device.get(row.get("target_device"))
        if source_device is None or target_device is None:
            continue
        resource_key = (
            "pcie"
            if "cuda" in {row["source_device"], row["target_device"]}
            else "usb"
        )
        resource = resources[resource_key]
        evidence = row.get("evidence_ids")
        if type(evidence) is not list or not evidence or any(
            type(item) is not str or not item for item in evidence
        ):
            raise MatmulScheduleError("derived link has invalid evidence ids")
        links.append({
            "link_id": row["model_id"],
            "source_device": source_device,
            "target_device": target_device,
            "resource_id": resource.resource_id,
            "resource_kind": resource.kind,
            "resource_capacity": resource.capacity,
            "resource_identity": resource.identity,
            "fixed_latency_us": row["fixed_latency_us"],
            "bandwidth_bytes_per_s": row["bandwidth_bytes_per_s"],
            "fixed_energy_uj": row["fixed_dynamic_uj"],
            "dynamic_pj_per_byte": row["dynamic_pj_per_byte"],
            "domain_power_mw": {},
            "minimum_bytes": row["min_payload_bytes"],
            "maximum_bytes": row["max_payload_bytes"],
            "status": "estimated",
            "ready": resource.ready,
            "evidence_ids": sorted(set(evidence)),
        })
    if len(links) != 4:
        raise MatmulScheduleError("kernel campaign lacks four directed links")

    result: dict[str, object] = {
        "schema": "s42-matmul-vq-profile-v1",
        "profile_id": f"{source['profile_id']}-matmul-vq-shadow",
        "energy_boundary_id": source["energy_boundary"]["id"],
        "source_profile_id": source["profile_id"],
        "domains": domains,
        "devices": devices,
        "kernels": kernels,
        "links": links,
        "policy": {
            "host_device_id": "cpu",
            "final_device_id": "cpu",
            "host_domain_id": "cpu-package",
            "host_active_power_mw": 115_000,
            "latency_limit_ppm": 1_000_000,
            "split_search_points": 16,
            "queue_limit": 4096,
            "require_measured": False,
        },
        "qualification": {
            "status": "shadow_only",
            "generic_family": generic_family,
            "included_backends": sorted(included_backends),
            "phone_resource_model": "shared_htp_adreno",
            "note": (
                "Generic rows reuse measured fused-FFN effective rates and are "
                "estimated until each model/operator shape is profiled."
                if generic_family
                else "Only exact source campaign shape buckets may retain measured status."
            ),
        },
    }
    MatmulSystemProfile.from_json(result)
    return result
