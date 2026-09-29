"""Materialize measured kernel campaigns into scheduler profiles."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from .policy import ResourceProfile
from .matmul import MatmulScheduleError, MatmulSystemProfile


GIB = 1024 * 1024 * 1024
KERNEL_CAMPAIGN_SCHEMA = "s42-kernel-energy-profile-v1"

__all__ = [
    "GIB",
    "KERNEL_CAMPAIGN_SCHEMA",
    "materialize_matmul_profile",
    "read_scheduler_inventory",
]


def read_scheduler_inventory(path: Path, *, start_us: int, end_us: int,
                             phase: str | None = None) -> dict[str, object]:
    """Read completed split groups; optional phase selection assumes single-slot decode."""
    if type(start_us) is not int or type(end_us) is not int or not 0 <= start_us < end_us:
        raise MatmulScheduleError("scheduler inventory interval is invalid")
    if phase not in (None, "prefill", "decode"):
        raise MatmulScheduleError("scheduler inventory phase is invalid")
    graphs, syncs = {}, []

    def integer(row, key, minimum=0):
        value = row.get(key)
        if type(value) is not int or value < minimum:
            raise MatmulScheduleError(f"scheduler trace {key} is invalid")
        return value

    def tensor(value):
        if not isinstance(value, dict) or any(type(value.get(k)) is not str or not value[k]
                                              for k in ("name", "op", "dtype")):
            raise MatmulScheduleError("scheduler trace tensor identity is invalid")
        shape = value.get("shape")
        if type(shape) is not list or len(shape) != 4 or any(type(n) is not int or n < 0 for n in shape):
            raise MatmulScheduleError("scheduler trace tensor shape is invalid")
        name = value["name"]
        match = re.search(r"blk\.(\d+)\.|-(\d+)(?:\D|$)", name)
        layer = int(next(g for g in match.groups() if g is not None)) if match else None
        family = ("ffn" if "ffn" in name else "attention" if any(
            part in name for part in ("attn", "rope", "kq", "cache_", "k_cache", "v_cache")) else
            "embedding" if "token_embd" in name or value["op"] == "GET_ROWS" else
            "output" if "output" in name else "other")
        return layer, family

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError as error:
                raise MatmulScheduleError("scheduler trace is not complete JSONL") from error
            if not isinstance(row, dict) or row.get("schema") != "ggml-sched-trace-v1":
                raise MatmulScheduleError("scheduler trace schema is unsupported")
            if type(row.get("scheduler")) is not str or not row["scheduler"]:
                raise MatmulScheduleError("scheduler trace owner is missing")
            key = row["scheduler"], integer(row, "graph", 1)
            event = row.get("event")
            if event == "sync":
                syncs.append(row)
                continue
            group = graphs.setdefault(key, {"splits": {}, "copies": {}})
            if event == "graph":
                if "total" in group:
                    raise MatmulScheduleError("scheduler trace has duplicate graph totals")
                group["total"] = row
            elif event == "split":
                index = integer(row, "split")
                if index in group["splits"]:
                    raise MatmulScheduleError("scheduler trace has duplicate splits")
                group["splits"][index] = row
            elif event == "copy":
                index = integer(row, "split"), integer(row, "input")
                if index in group["copies"]:
                    raise MatmulScheduleError("scheduler trace has duplicate copies")
                group["copies"][index] = row
            else:
                raise MatmulScheduleError("scheduler trace event is unsupported")

    selected = {}
    for key, group in graphs.items():
        if "total" not in group:
            raise MatmulScheduleError("scheduler trace has an unfinished graph")
        total = group["total"]
        started, finished = integer(total, "started_us"), integer(total, "finished_us")
        if finished < started or total.get("completion") != "async_submission":
            raise MatmulScheduleError("scheduler graph timing is invalid")
        if start_us <= started and finished <= end_us:
            tokens = integer(total, "tokens", 1)
            if phase is None or (tokens > 1) == (phase == "prefill"):
                selected[key] = group
        elif max(started, start_us) < min(finished, end_us):
            raise MatmulScheduleError("scheduler inventory cuts a graph")
    if not selected:
        raise MatmulScheduleError("scheduler inventory has no graphs")

    inventory, components, weight_copies = [], {}, {}
    copy_us = compute_us = submission_us = copied_bytes = weight_bytes = weight_us = wait_us = 0
    for key, group in selected.items():
        total = group["total"]
        count = integer(total, "split_count", 1)
        if sorted(group["splits"]) != list(range(count)):
            raise MatmulScheduleError("scheduler graph split coverage is incomplete")
        submission_us += integer(total, "host_wall_us")
        tokens = integer(total, "tokens", 1)
        for index, split in sorted(group["splits"].items()):
            nodes = split.get("nodes")
            if type(nodes) is not list or len(nodes) != integer(split, "node_count", 1):
                raise MatmulScheduleError("scheduler split node coverage is incomplete")
            identities = [tensor(node) for node in nodes]
            layers = tuple(sorted({layer for layer, _ in identities if layer is not None}))
            families = tuple(sorted({family for _, family in identities}))
            backend = split.get("backend")
            if type(backend) is not str or not backend:
                raise MatmulScheduleError("scheduler split backend is invalid")
            copies = [c for (s, _), c in sorted(group["copies"].items()) if s == index]
            if [c["input"] for c in copies] != list(range(len(copies))):
                raise MatmulScheduleError("scheduler split input coverage is incomplete")
            for copy in copies:
                layer, family = tensor(copy.get("tensor"))
                size = integer(copy, "bytes")
                duration, wait = integer(copy, "host_us"), integer(copy, "wait_us")
                ranges = copy.get("ranges")
                if type(ranges) is not list or any(type(r) is not list or len(r) != 2
                        or any(type(n) is not int or n < 0 for n in r) for r in ranges):
                    raise MatmulScheduleError("scheduler copy ranges are invalid")
                if sum(r[1] for r in ranges) != size or wait > duration:
                    raise MatmulScheduleError("scheduler copy accounting is inconsistent")
                if type(copy.get("weights")) is not bool or copy.get("destination") != backend:
                    raise MatmulScheduleError("scheduler copy placement is inconsistent")
                copy["layer"], copy["family"] = layer, family
                copy["kind"] = "weights" if copy["weights"] else "kv" if any(
                    p in copy["tensor"]["name"] for p in ("cache_", "k_cache", "v_cache")) else "activation"
                wait_us += wait
                if copy["weights"]:
                    weight_bytes += size
                    weight_us += duration - wait
                    weight_key = copy["source"], backend, copy["tensor"]["name"]
                    weight = weight_copies.setdefault(weight_key, {"source": copy["source"], "destination": backend,
                        "tensor": copy["tensor"]["name"], "layer": layer, "family": family, "calls": 0, "bytes": 0,
                        "copy_host_excluding_wait_us": 0})
                    weight["calls"] += 1
                    weight["bytes"] += size
                    weight["copy_host_excluding_wait_us"] += duration - wait
            nbytes = sum(c["bytes"] for c in copies)
            duration = sum(c["host_us"] for c in copies)
            compute = integer(split, "compute_host_us")
            if nbytes != integer(split, "copied_bytes") or duration != integer(split, "copy_host_us"):
                raise MatmulScheduleError("scheduler split copy coverage is incomplete")
            if duration + compute > integer(split, "wall_us"):
                raise MatmulScheduleError("scheduler split timing overlaps")
            copy_us += duration
            compute_us += compute
            copied_bytes += nbytes
            component_key = backend, layers, families
            component = components.setdefault(component_key, {"device": backend, "layers": list(layers),
                "families": list(families), "split_calls": 0, "copy_bytes": 0, "copy_host_us": 0, "compute_host_us": 0})
            component["split_calls"] += 1
            component["copy_bytes"] += nbytes
            component["copy_host_us"] += duration
            component["compute_host_us"] += compute
        if any(s >= count for s, _ in group["copies"]):
            raise MatmulScheduleError("scheduler copy has no split")
        inventory.append({"scheduler": key[0], "graph": key[1], "tokens": tokens,
                          "phase": "prefill" if tokens > 1 else "decode",
                          "started_us": total["started_us"], "finished_us": total["finished_us"],
                          "splits": list(group["splits"].values()), "copies": list(group["copies"].values())})

    completed, sync_us = set(), 0
    for sync in syncs:
        keys = {(sync["scheduler"], n) for n in range(integer(sync, "first_graph", 1), sync["graph"] + 1)}
        overlap = keys & selected.keys()
        if not overlap:
            continue
        if overlap != keys or completed & keys or not start_us <= integer(sync, "started_us") <= integer(sync, "finished_us") <= end_us:
            raise MatmulScheduleError("scheduler completion crosses the inventory boundary")
        sync_us += integer(sync, "host_wait_us")
        completed.update(keys)
    if completed != selected.keys():
        raise MatmulScheduleError("scheduler inventory lacks normal completion evidence")
    return {"schema": "scheduler-split-inventory-v1", "graph_count": len(inventory), "graphs": inventory,
        "components": list(components.values()), "weight_transfers": list(weight_copies.values()),
        "copy_bytes": copied_bytes, "weight_copy_bytes": weight_bytes, "copy_host_us": copy_us,
        "copy_wait_us": wait_us, "weight_copy_host_excluding_wait_us": weight_us, "compute_host_us": compute_us,
        "normal_sync_wait_us": sync_us, "submission_wall_us": submission_us,
        "accounted_host_us": copy_us + compute_us + sync_us,
        "timing_note": "Host copy/compute calls plus existing completion waits; GPU enqueue is not device compute. "
                       "Mixed split groups retain joint layer/family timing; no invented per-node allocation."}


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


def _validate_memory_bounds(
    gpu_capacity_bytes: int,
    phone_capacity_bytes: int,
    gpu_reserved_bytes: int,
    phone_reserved_bytes: int,
) -> None:
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


def _energy_domains(source: Mapping[str, Any]) -> list[dict[str, object]]:
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
    return domains


def _device_rows(
    resources: Mapping[str, ResourceProfile],
    *,
    gpu_capacity_bytes: int,
    phone_capacity_bytes: int,
    gpu_reserved_bytes: int,
    phone_reserved_bytes: int,
) -> list[dict[str, object]]:
    return [
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


def _kernel_row(
    row: Mapping[str, Any],
    backend: str,
    device_id: str,
    generic_family: bool,
) -> dict[str, object]:
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
    return {
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
    }


def _kernel_rows(
    source: Mapping[str, Any],
    generic_family: bool,
) -> tuple[list[dict[str, object]], set[str]]:
    backend_device = {
        "cpu": "cpu",
        "cuda": "gpu",
        "htp": "phone",
        "adreno": "phone",
    }
    kernels: list[dict[str, object]] = []
    included_backends: set[str] = set()
    for row in _rows("kernel campaign kernels", source.get("kernel_rows")):
        backend = row.get("backend")
        device_id = backend_device.get(backend)
        if device_id is None:
            continue
        included_backends.add(backend)
        kernels.append(_kernel_row(row, backend, device_id, generic_family))
    if not kernels:
        raise MatmulScheduleError("kernel campaign has no usable matmul rows")
    return kernels, included_backends


def _link_rows(
    source: Mapping[str, Any],
    resources: Mapping[str, ResourceProfile],
) -> list[dict[str, object]]:
    endpoint_device = {"cpu": "cpu", "cuda": "gpu", "phone-memory": "phone"}
    links: list[dict[str, object]] = []
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
    return links


def _profile_document(
    source: Mapping[str, Any],
    *,
    generic_family: bool,
    domains: list[dict[str, object]],
    devices: list[dict[str, object]],
    kernels: list[dict[str, object]],
    links: list[dict[str, object]],
    included_backends: set[str],
) -> dict[str, object]:
    return {
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
    _validate_memory_bounds(
        gpu_capacity_bytes,
        phone_capacity_bytes,
        gpu_reserved_bytes,
        phone_reserved_bytes,
    )
    domains = _energy_domains(source)
    devices = _device_rows(
        resources,
        gpu_capacity_bytes=gpu_capacity_bytes,
        phone_capacity_bytes=phone_capacity_bytes,
        gpu_reserved_bytes=gpu_reserved_bytes,
        phone_reserved_bytes=phone_reserved_bytes,
    )
    kernels, included_backends = _kernel_rows(source, generic_family)
    links = _link_rows(source, resources)
    result = _profile_document(
        source,
        generic_family=generic_family,
        domains=domains,
        devices=devices,
        kernels=kernels,
        links=links,
        included_backends=included_backends,
    )
    MatmulSystemProfile.from_json(result)
    return result
