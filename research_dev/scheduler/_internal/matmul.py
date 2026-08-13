#!/usr/bin/env python3
"""Causal virtual-queue scheduler for split matmul placement.

This module plans only dense matmul kernels. Shard-safe non-matmul work
inherits the preceding placement until the next matmul. The planner is a
shadow compiler: a selected route still needs an epoch-bound runtime
certificate before it can be executed by llama-server.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy import (
    LeaseDemand,
    LeasePreview,
    LeaseRecord,
    ResourceProfile,
    ResourceTimeline,
    SchedulerError,
)


PROFILE_SCHEMA = "s42-matmul-vq-profile-v1"
WORKLOAD_SCHEMA = "s42-matmul-vq-workload-v1"
RESULT_SCHEMA = "s42-matmul-vq-result-v1"
PROFILE_STATUSES = {"estimated", "measured"}
DEVICE_ORDER = ("cpu", "gpu", "phone")
QUEUE_TRANSITIONS = (
    "ANNOUNCED",
    "READY",
    "ASSIGNED",
    "RUNNING",
    "COMPLETE",
)

__all__ = [
    "DEVICE_ORDER",
    "PROFILE_SCHEMA",
    "PROFILE_STATUSES",
    "QUEUE_TRANSITIONS",
    "RESULT_SCHEMA",
    "WORKLOAD_SCHEMA",
    "ActivationState",
    "CandidatePlan",
    "EnergyDomainProfile",
    "FollowOp",
    "KernelCatalog",
    "KernelEstimate",
    "LinkCatalog",
    "MatmulDeviceProfile",
    "MatmulKernelProfile",
    "MatmulLinkProfile",
    "MatmulOp",
    "MatmulPolicy",
    "MatmulScheduleError",
    "MatmulSystemProfile",
    "MatmulPlanner",
    "MemoryLedger",
    "MemoryPreview",
    "ModelProgram",
    "Phase",
    "PowerInterval",
    "PowerTimeline",
    "ProgramOp",
    "TransferEstimate",
    "load_profile",
    "load_workload",
    "main",
    "run_workload",
]


class MatmulScheduleError(ValueError):
    pass


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise MatmulScheduleError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise MatmulScheduleError(f"{name} must be bool")
    return value


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise MatmulScheduleError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise MatmulScheduleError(f"{name} must be ASCII") from exc
    return value


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise MatmulScheduleError(f"{name} must be an object")
    return value


def _array(name: str, value: object, nonempty: bool = True) -> list[Any]:
    if type(value) is not list or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise MatmulScheduleError(f"{name} must be a {qualifier}list")
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise MatmulScheduleError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _portion(total: int, part: int, whole: int) -> int:
    _integer("portion total", total)
    _integer("portion part", part)
    _integer("portion whole", whole, 1)
    if part > whole:
        raise MatmulScheduleError("portion exceeds its whole")
    return _ceil_div(total * part, whole) if part else 0


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // math.gcd(left, right)


def _status(name: str, value: object) -> str:
    result = _string(name, value)
    if result not in PROFILE_STATUSES:
        raise MatmulScheduleError(f"{name} has an unknown status")
    return result


def _evidence(name: str, value: object) -> tuple[str, ...]:
    rows = _array(name, value)
    return tuple(_string(name, row) for row in rows)


@dataclass(frozen=True)
class EnergyDomainProfile:
    domain_id: str
    idle_power_mw: int

    @classmethod
    def from_json(cls, value: object) -> "EnergyDomainProfile":
        row = _object("energy domain", value)
        return cls(
            domain_id=_string("energy domain id", row.get("domain_id")),
            idle_power_mw=_integer(
                "energy domain idle_power_mw", row.get("idle_power_mw")
            ),
        )


@dataclass(frozen=True)
class MatmulDeviceProfile:
    device_id: str
    kind: str
    resource_id: str
    resource_kind: str
    resource_capacity: int
    resource_identity: str
    memory_capacity_bytes: int
    reserved_bytes: int
    ready: bool

    @classmethod
    def from_json(cls, value: object) -> "MatmulDeviceProfile":
        row = _object("matmul device", value)
        device_id = _string("device id", row.get("device_id"))
        kind = _string("device kind", row.get("kind"))
        result = cls(
            device_id=device_id,
            kind=kind,
            resource_id=_string("device resource_id", row.get("resource_id")),
            resource_kind=_string(
                "device resource_kind", row.get("resource_kind", kind)
            ),
            resource_capacity=_integer(
                "device resource_capacity",
                row.get("resource_capacity", 1),
                1,
            ),
            resource_identity=_string(
                "device resource_identity",
                row.get("resource_identity", device_id),
            ),
            memory_capacity_bytes=_integer(
                "device memory_capacity_bytes",
                row.get("memory_capacity_bytes", 0),
            ),
            reserved_bytes=_integer(
                "device reserved_bytes", row.get("reserved_bytes", 0)
            ),
            ready=_boolean("device ready", row.get("ready")),
        )
        if (
            result.memory_capacity_bytes
            and result.reserved_bytes > result.memory_capacity_bytes
        ):
            raise MatmulScheduleError("device reserve exceeds capacity")
        return result


@dataclass(frozen=True)
class MatmulKernelProfile:
    profile_id: str
    device_id: str
    kernel_family: str
    quantization: str
    shape_m: int
    shape_k: int
    shape_n: int
    effective_ops_per_s: int
    effective_bytes_per_s: int
    launch_us: int
    domain_power_mw: Mapping[str, int]
    minimum_n: int
    maximum_n: int
    n_quantum: int
    status: str
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "MatmulKernelProfile":
        row = _object("matmul kernel", value)
        shape = _object("matmul kernel shape", row.get("shape"))
        raw_power = _object(
            "matmul kernel domain_power_mw", row.get("domain_power_mw")
        )
        result = cls(
            profile_id=_string("matmul kernel profile_id", row.get("profile_id")),
            device_id=_string("matmul kernel device_id", row.get("device_id")),
            kernel_family=_string(
                "matmul kernel kernel_family", row.get("kernel_family")
            ),
            quantization=_string(
                "matmul kernel quantization", row.get("quantization")
            ),
            shape_m=_integer("matmul kernel shape.m", shape.get("m"), 1),
            shape_k=_integer("matmul kernel shape.k", shape.get("k")),
            shape_n=_integer("matmul kernel shape.n", shape.get("n"), 1),
            effective_ops_per_s=_integer(
                "matmul kernel effective_ops_per_s",
                row.get("effective_ops_per_s"),
                1,
            ),
            effective_bytes_per_s=_integer(
                "matmul kernel effective_bytes_per_s",
                row.get("effective_bytes_per_s"),
                1,
            ),
            launch_us=_integer(
                "matmul kernel launch_us", row.get("launch_us", 0)
            ),
            domain_power_mw={
                _string("kernel power domain", key): _integer(
                    f"kernel power {key}", power, 1
                )
                for key, power in raw_power.items()
            },
            minimum_n=_integer(
                "matmul kernel minimum_n", row.get("minimum_n", 1), 1
            ),
            maximum_n=_integer(
                "matmul kernel maximum_n", row.get("maximum_n", 0)
            ),
            n_quantum=_integer(
                "matmul kernel n_quantum", row.get("n_quantum", 1), 1
            ),
            status=_status("matmul kernel status", row.get("status")),
            evidence_ids=_evidence(
                "matmul kernel evidence_ids", row.get("evidence_ids")
            ),
        )
        if not result.domain_power_mw:
            raise MatmulScheduleError("matmul kernel has no power domain")
        if result.maximum_n and result.minimum_n > result.maximum_n:
            raise MatmulScheduleError("matmul kernel N range is invalid")
        return result

    def family_matches(self, op: "MatmulOp") -> bool:
        return (
            self.kernel_family in {"*", op.kernel_family}
            and self.quantization in {"*", op.quantization}
            and self.shape_k in {0, op.k}
        )

    def supports_columns(self, op: "MatmulOp", columns: int) -> bool:
        if not self.family_matches(op) or columns < self.minimum_n:
            return False
        if self.maximum_n and columns > self.maximum_n:
            return False
        return columns % _lcm(self.n_quantum, op.split_quantum_n) == 0


@dataclass(frozen=True)
class MatmulLinkProfile:
    link_id: str
    source_device: str
    target_device: str
    resource_id: str
    resource_kind: str
    resource_capacity: int
    resource_identity: str
    fixed_latency_us: int
    bandwidth_bytes_per_s: int
    fixed_energy_uj: int
    dynamic_pj_per_byte: int
    domain_power_mw: Mapping[str, int]
    minimum_bytes: int
    maximum_bytes: int
    status: str
    ready: bool
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "MatmulLinkProfile":
        row = _object("matmul link", value)
        raw_power = _object(
            "matmul link domain_power_mw", row.get("domain_power_mw", {})
        )
        link_id = _string("matmul link id", row.get("link_id"))
        resource_id = _string(
            "matmul link resource_id", row.get("resource_id")
        )
        result = cls(
            link_id=link_id,
            source_device=_string(
                "matmul link source_device", row.get("source_device")
            ),
            target_device=_string(
                "matmul link target_device", row.get("target_device")
            ),
            resource_id=resource_id,
            resource_kind=_string(
                "matmul link resource_kind",
                row.get("resource_kind", "transfer_link"),
            ),
            resource_capacity=_integer(
                "matmul link resource_capacity",
                row.get("resource_capacity", 1),
                1,
            ),
            resource_identity=_string(
                "matmul link resource_identity",
                row.get("resource_identity", resource_id),
            ),
            fixed_latency_us=_integer(
                "matmul link fixed_latency_us", row.get("fixed_latency_us")
            ),
            bandwidth_bytes_per_s=_integer(
                "matmul link bandwidth_bytes_per_s",
                row.get("bandwidth_bytes_per_s"),
                1,
            ),
            fixed_energy_uj=_integer(
                "matmul link fixed_energy_uj", row.get("fixed_energy_uj", 0)
            ),
            dynamic_pj_per_byte=_integer(
                "matmul link dynamic_pj_per_byte",
                row.get("dynamic_pj_per_byte", 0),
            ),
            domain_power_mw={
                _string("link power domain", key): _integer(
                    f"link power {key}", power, 1
                )
                for key, power in raw_power.items()
            },
            minimum_bytes=_integer(
                "matmul link minimum_bytes", row.get("minimum_bytes", 0)
            ),
            maximum_bytes=_integer(
                "matmul link maximum_bytes", row.get("maximum_bytes", 0)
            ),
            status=_status("matmul link status", row.get("status")),
            ready=_boolean("matmul link ready", row.get("ready")),
            evidence_ids=_evidence(
                "matmul link evidence_ids", row.get("evidence_ids")
            ),
        )
        if result.maximum_bytes and result.minimum_bytes > result.maximum_bytes:
            raise MatmulScheduleError("matmul link payload range is invalid")
        return result

    def estimate_status(self, payload_bytes: int) -> str:
        in_range = payload_bytes >= self.minimum_bytes and (
            self.maximum_bytes == 0 or payload_bytes <= self.maximum_bytes
        )
        return self.status if in_range else "estimated"


@dataclass(frozen=True)
class MatmulPolicy:
    host_device_id: str
    final_device_id: str
    host_domain_id: str
    host_active_power_mw: int
    latency_limit_ppm: int
    split_search_points: int
    queue_limit: int
    require_measured: bool

    @classmethod
    def from_json(cls, value: object) -> "MatmulPolicy":
        row = _object("matmul policy", value)
        result = cls(
            host_device_id=_string(
                "matmul policy host_device_id", row.get("host_device_id")
            ),
            final_device_id=_string(
                "matmul policy final_device_id", row.get("final_device_id")
            ),
            host_domain_id=_string(
                "matmul policy host_domain_id", row.get("host_domain_id")
            ),
            host_active_power_mw=_integer(
                "matmul policy host_active_power_mw",
                row.get("host_active_power_mw"),
                1,
            ),
            latency_limit_ppm=_integer(
                "matmul policy latency_limit_ppm",
                row.get("latency_limit_ppm", 1_000_000),
                1_000_000,
            ),
            split_search_points=_integer(
                "matmul policy split_search_points",
                row.get("split_search_points", 16),
                2,
            ),
            queue_limit=_integer(
                "matmul policy queue_limit", row.get("queue_limit", 4096), 1
            ),
            require_measured=_boolean(
                "matmul policy require_measured",
                row.get("require_measured", False),
            ),
        )
        return result


@dataclass(frozen=True)
class MatmulSystemProfile:
    profile_id: str
    energy_boundary_id: str
    domains: Mapping[str, EnergyDomainProfile]
    devices: Mapping[str, MatmulDeviceProfile]
    resources: Mapping[str, ResourceProfile]
    kernels: tuple[MatmulKernelProfile, ...]
    links: tuple[MatmulLinkProfile, ...]
    policy: MatmulPolicy

    @classmethod
    def from_json(cls, value: object) -> "MatmulSystemProfile":
        row = _object("matmul system profile", value)
        if row.get("schema") != PROFILE_SCHEMA:
            raise MatmulScheduleError("matmul system profile schema mismatch")

        domains: dict[str, EnergyDomainProfile] = {}
        for raw in _array("matmul profile domains", row.get("domains")):
            domain = EnergyDomainProfile.from_json(raw)
            if domain.domain_id in domains:
                raise MatmulScheduleError("duplicate energy domain id")
            domains[domain.domain_id] = domain

        devices: dict[str, MatmulDeviceProfile] = {}
        resources: dict[str, ResourceProfile] = {}
        for raw in _array("matmul profile devices", row.get("devices")):
            device = MatmulDeviceProfile.from_json(raw)
            if device.device_id in devices:
                raise MatmulScheduleError("duplicate device id")
            if device.resource_id in resources:
                raise MatmulScheduleError("duplicate resource id")
            devices[device.device_id] = device
            resources[device.resource_id] = ResourceProfile(
                resource_id=device.resource_id,
                kind=device.resource_kind,
                capacity=device.resource_capacity,
                ready=device.ready,
                identity=device.resource_identity,
            )

        kernels = tuple(
            MatmulKernelProfile.from_json(raw)
            for raw in _array("matmul profile kernels", row.get("kernels"))
        )
        kernel_ids: set[str] = set()
        for kernel in kernels:
            if kernel.profile_id in kernel_ids:
                raise MatmulScheduleError("duplicate matmul kernel profile id")
            kernel_ids.add(kernel.profile_id)
            if kernel.device_id not in devices:
                raise MatmulScheduleError("kernel references an unknown device")
            for domain_id, power in kernel.domain_power_mw.items():
                domain = domains.get(domain_id)
                if domain is None:
                    raise MatmulScheduleError("kernel references an unknown domain")
                if power < domain.idle_power_mw:
                    raise MatmulScheduleError("kernel power is below domain idle")

        links: list[MatmulLinkProfile] = []
        link_ids: set[str] = set()
        for raw in _array("matmul profile links", row.get("links"), False):
            link = MatmulLinkProfile.from_json(raw)
            if link.link_id in link_ids:
                raise MatmulScheduleError("duplicate matmul link id")
            link_ids.add(link.link_id)
            if (
                link.source_device not in devices
                or link.target_device not in devices
            ):
                raise MatmulScheduleError("link references an unknown device")
            for domain_id, power in link.domain_power_mw.items():
                domain = domains.get(domain_id)
                if domain is None:
                    raise MatmulScheduleError("link references an unknown domain")
                if power < domain.idle_power_mw:
                    raise MatmulScheduleError("link power is below domain idle")
            expected_resource = ResourceProfile(
                resource_id=link.resource_id,
                kind=link.resource_kind,
                capacity=link.resource_capacity,
                ready=link.ready,
                identity=link.resource_identity,
            )
            resource = resources.get(link.resource_id)
            if resource is None:
                resources[link.resource_id] = expected_resource
            elif resource != expected_resource:
                raise MatmulScheduleError(
                    "shared matmul link resource profile differs"
                )
            links.append(link)

        policy = MatmulPolicy.from_json(row.get("policy"))
        if policy.host_device_id not in devices:
            raise MatmulScheduleError("policy host device is unknown")
        if policy.final_device_id != policy.host_device_id:
            raise MatmulScheduleError("final matmul destination must be the host")
        host_domain = domains.get(policy.host_domain_id)
        if host_domain is None:
            raise MatmulScheduleError("policy host domain is unknown")
        if policy.host_active_power_mw < host_domain.idle_power_mw:
            raise MatmulScheduleError("host active power is below host idle")

        return cls(
            profile_id=_string("matmul profile id", row.get("profile_id")),
            energy_boundary_id=_string(
                "matmul energy_boundary_id", row.get("energy_boundary_id")
            ),
            domains=domains,
            devices=devices,
            resources=resources,
            kernels=kernels,
            links=tuple(links),
            policy=policy,
        )


@dataclass(frozen=True)
class MatmulOp:
    op_id: str
    layer_id: str
    weight_id: str
    kernel_family: str
    quantization: str
    m: int
    k: int
    n: int
    input_bytes: int
    output_bytes: int
    weight_bytes: int
    compute_ops: int
    split_quantum_n: int = 1
    allowed_devices: tuple[str, ...] = DEVICE_ORDER
    deadline_us: int = 0

    @classmethod
    def from_json(cls, value: object) -> "MatmulOp":
        row = _object("matmul op", value)
        shape = _object("matmul op shape", row.get("shape"))
        m = _integer("matmul M", shape.get("m"), 1)
        k = _integer("matmul K", shape.get("k"), 1)
        n = _integer("matmul N", shape.get("n"), 1)
        allowed = tuple(
            _string("matmul allowed device", item)
            for item in _array(
                "matmul allowed_devices",
                row.get("allowed_devices", list(DEVICE_ORDER)),
            )
        )
        result = cls(
            op_id=_string("matmul op id", row.get("op_id")),
            layer_id=_string("matmul layer id", row.get("layer_id")),
            weight_id=_string("matmul weight id", row.get("weight_id")),
            kernel_family=_string(
                "matmul kernel family", row.get("kernel_family")
            ),
            quantization=_string(
                "matmul quantization", row.get("quantization")
            ),
            m=m,
            k=k,
            n=n,
            input_bytes=_integer(
                "matmul input_bytes", row.get("input_bytes"), 1
            ),
            output_bytes=_integer(
                "matmul output_bytes", row.get("output_bytes"), 1
            ),
            weight_bytes=_integer(
                "matmul weight_bytes", row.get("weight_bytes"), 1
            ),
            compute_ops=_integer(
                "matmul compute_ops",
                row.get("compute_ops", 2 * m * k * n),
                1,
            ),
            split_quantum_n=_integer(
                "matmul split_quantum_n", row.get("split_quantum_n", 1), 1
            ),
            allowed_devices=allowed,
            deadline_us=_integer(
                "matmul deadline_us", row.get("deadline_us", 0)
            ),
        )
        if len(set(result.allowed_devices)) != len(result.allowed_devices):
            raise MatmulScheduleError("matmul allowed_devices contains duplicates")
        if set(result.allowed_devices) - set(DEVICE_ORDER):
            raise MatmulScheduleError("matmul allowed_devices is unknown")
        if result.compute_ops < 2 * result.m * result.k * result.n:
            raise MatmulScheduleError("matmul compute_ops is below 2*M*K*N")
        return result


@dataclass(frozen=True)
class FollowOp:
    op_id: str
    layer_id: str
    op_kind: str
    shard_safe: bool = True
    output_bytes: int = 0

    @classmethod
    def from_json(cls, value: object) -> "FollowOp":
        row = _object("follow op", value)
        return cls(
            op_id=_string("follow op id", row.get("op_id")),
            layer_id=_string("follow layer id", row.get("layer_id")),
            op_kind=_string("follow op kind", row.get("op_kind")),
            shard_safe=_boolean(
                "follow shard_safe", row.get("shard_safe", True)
            ),
            output_bytes=_integer(
                "follow output_bytes", row.get("output_bytes", 0)
            ),
        )


ProgramOp = MatmulOp | FollowOp


@dataclass(frozen=True)
class ModelProgram:
    program_id: str
    model_id: str
    arrival_us: int
    deadline_us: int
    ops: tuple[ProgramOp, ...]

    @classmethod
    def from_json(cls, value: object) -> "ModelProgram":
        row = _object("model program", value)
        raw_ops = _array("model program ops", row.get("ops"))
        ops: list[ProgramOp] = []
        for raw in raw_ops:
            op = _object("model program op", raw)
            kind = _string("model program op kind", op.get("kind"))
            if kind == "matmul":
                ops.append(MatmulOp.from_json(op))
            elif kind == "follow":
                ops.append(FollowOp.from_json(op))
            else:
                raise MatmulScheduleError("unknown model program op kind")
        result = cls(
            program_id=_string("model program id", row.get("program_id")),
            model_id=_string("model id", row.get("model_id")),
            arrival_us=_integer("model program arrival_us", row.get("arrival_us")),
            deadline_us=_integer(
                "model program deadline_us", row.get("deadline_us"), 1
            ),
            ops=tuple(ops),
        )
        if result.deadline_us <= result.arrival_us:
            raise MatmulScheduleError("model program deadline must follow arrival")
        ids = [op.op_id for op in result.ops]
        if len(ids) != len(set(ids)):
            raise MatmulScheduleError("model program contains duplicate op ids")
        if not any(isinstance(op, MatmulOp) for op in result.ops):
            raise MatmulScheduleError("model program contains no matmul")
        return result


@dataclass(frozen=True)
class ActivationState:
    total_bytes: int
    units: int
    shards: tuple[tuple[str, int, int], ...]

    @classmethod
    def full(cls, device_id: str, total_bytes: int) -> "ActivationState":
        return cls(total_bytes, 1, ((device_id, 1, total_bytes),))

    @classmethod
    def from_columns(
        cls,
        total_bytes: int,
        total_columns: int,
        columns: Mapping[str, int],
    ) -> "ActivationState":
        nonzero = [
            (device, columns.get(device, 0))
            for device in DEVICE_ORDER
            if columns.get(device, 0) > 0
        ]
        if sum(amount for _, amount in nonzero) != total_columns:
            raise MatmulScheduleError("activation columns do not sum to N")
        remaining = total_bytes
        shards: list[tuple[str, int, int]] = []
        for index, (device, amount) in enumerate(nonzero):
            size = (
                remaining
                if index == len(nonzero) - 1
                else total_bytes * amount // total_columns
            )
            remaining -= size
            shards.append((device, amount, size))
        return cls(total_bytes, total_columns, tuple(shards))

    def bytes_by_device(self, total_bytes: int | None = None) -> dict[str, int]:
        target = self.total_bytes if total_bytes is None else total_bytes
        _integer("activation rescale bytes", target)
        remaining = target
        result: dict[str, int] = {}
        for index, (device, units, _) in enumerate(self.shards):
            size = (
                remaining
                if index == len(self.shards) - 1
                else target * units // self.units
            )
            remaining -= size
            result[device] = size
        return result

    def rescale(self, total_bytes: int) -> "ActivationState":
        rows = self.bytes_by_device(total_bytes)
        return ActivationState(
            total_bytes,
            self.units,
            tuple(
                (device, units, rows[device])
                for device, units, _ in self.shards
            ),
        )

    def devices(self) -> tuple[str, ...]:
        return tuple(device for device, _, _ in self.shards)


def load_profile(path: Path) -> MatmulSystemProfile:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatmulScheduleError(f"cannot read profile: {exc}") from exc
    return MatmulSystemProfile.from_json(raw)


def load_workload(path: Path) -> tuple[ModelProgram, ...]:
    try:
        raw = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MatmulScheduleError(f"cannot read workload: {exc}") from exc
    row = _object("matmul workload", raw)
    if row.get("schema") != WORKLOAD_SCHEMA:
        raise MatmulScheduleError("matmul workload schema mismatch")
    programs = tuple(
        ModelProgram.from_json(item)
        for item in _array("matmul workload programs", row.get("programs"))
    )
    ids = [program.program_id for program in programs]
    if len(ids) != len(set(ids)):
        raise MatmulScheduleError("matmul workload contains duplicate programs")
    return programs


@dataclass(frozen=True)
class KernelEstimate:
    profile_id: str
    device_id: str
    columns: int
    compute_ops: int
    memory_bytes: int
    compute_us: int
    memory_us: int
    duration_us: int
    domain_power_mw: Mapping[str, int]
    status: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TransferEstimate:
    transfer_id: str
    link_id: str
    source_device: str
    target_device: str
    resource_id: str
    bytes: int
    duration_us: int
    scalar_energy_uj: int
    domain_power_mw: Mapping[str, int]
    status: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Phase:
    phase_id: str
    kind: str
    resource_id: str
    start_offset_us: int
    duration_us: int
    domain_power_mw: Mapping[str, int]
    device_id: str = ""
    transfer: TransferEstimate | None = None
    kernel: KernelEstimate | None = None


@dataclass(frozen=True)
class PowerInterval:
    domain_id: str
    start_us: int
    end_us: int
    power_mw: int


@dataclass(frozen=True)
class MemoryPreview:
    feasible: bool
    reason: str
    usage_before_bytes: Mapping[str, int]
    peak_usage_bytes: Mapping[str, int]
    usage_after_bytes: Mapping[str, int]
    available_after_bytes: Mapping[str, int]
    weight_required_bytes: Mapping[str, int]
    weight_growth_bytes: Mapping[str, int]
    output_activation_bytes: Mapping[str, int]


@dataclass(frozen=True)
class CandidatePlan:
    route_id: str
    columns: Mapping[str, int]
    output_state: ActivationState
    kernels: tuple[KernelEstimate, ...]
    transfers: tuple[TransferEstimate, ...]
    phases: tuple[Phase, ...]
    service_us: int
    scalar_energy_uj: int
    measured: bool
    evidence_ids: tuple[str, ...]
    memory: MemoryPreview
    lease_preview: LeasePreview | None = None
    incremental_energy_nj: int = 0
    objective_energy_nj: int = 0
    lookahead_finish_us: int = 0
    lookahead_transfers: tuple[TransferEstimate, ...] = ()


class KernelCatalog:
    def __init__(self, profile: MatmulSystemProfile) -> None:
        self.profile = profile
        self._by_device: dict[str, list[MatmulKernelProfile]] = {}
        for kernel in profile.kernels:
            self._by_device.setdefault(kernel.device_id, []).append(kernel)
        for rows in self._by_device.values():
            rows.sort(key=lambda item: item.profile_id)

    @staticmethod
    def _shape_distance(kernel: MatmulKernelProfile, op: MatmulOp) -> int:
        # Integer log distance keeps selection deterministic without floats.
        larger = max(kernel.shape_m, op.m)
        smaller = min(kernel.shape_m, op.m)
        m_ratio_ppm = _ceil_div(larger * 1_000_000, smaller)
        return abs(kernel.shape_n - op.n) + m_ratio_ppm

    def profiles_for(self, device_id: str, op: MatmulOp) -> tuple[MatmulKernelProfile, ...]:
        return tuple(
            row
            for row in self._by_device.get(device_id, ())
            if row.family_matches(op)
        )

    def estimate(
        self,
        device_id: str,
        op: MatmulOp,
        columns: int,
    ) -> KernelEstimate | None:
        choices = [
            row
            for row in self.profiles_for(device_id, op)
            if row.supports_columns(op, columns)
        ]
        if not choices:
            return None
        kernel = min(
            choices,
            key=lambda row: (self._shape_distance(row, op), row.profile_id),
        )
        compute_ops = _portion(op.compute_ops, columns, op.n)
        weight_bytes = _portion(op.weight_bytes, columns, op.n)
        output_bytes = _portion(op.output_bytes, columns, op.n)
        memory_bytes = op.input_bytes + weight_bytes + output_bytes
        compute_us = _ceil_div(
            compute_ops * 1_000_000, kernel.effective_ops_per_s
        )
        memory_us = _ceil_div(
            memory_bytes * 1_000_000, kernel.effective_bytes_per_s
        )
        duration_us = kernel.launch_us + max(compute_us, memory_us)
        exact_shape = (
            op.m == kernel.shape_m
            and op.k == kernel.shape_k
            and columns == kernel.shape_n
            and kernel.kernel_family == op.kernel_family
            and kernel.quantization == op.quantization
        )
        status = kernel.status if exact_shape else "estimated"
        return KernelEstimate(
            profile_id=kernel.profile_id,
            device_id=device_id,
            columns=columns,
            compute_ops=compute_ops,
            memory_bytes=memory_bytes,
            compute_us=compute_us,
            memory_us=memory_us,
            duration_us=duration_us,
            domain_power_mw=kernel.domain_power_mw,
            status=status,
            evidence_ids=kernel.evidence_ids,
        )

    def alignment(self, device_id: str, op: MatmulOp) -> int:
        rows = self.profiles_for(device_id, op)
        if not rows:
            return 0
        result = op.split_quantum_n
        for row in rows:
            result = _lcm(result, row.n_quantum)
        return result


class LinkCatalog:
    def __init__(self, profile: MatmulSystemProfile) -> None:
        self.profile = profile
        self._direct: dict[tuple[str, str], list[MatmulLinkProfile]] = {}
        self._resource_ready = {
            resource_id: resource.ready
            for resource_id, resource in profile.resources.items()
        }
        for link in profile.links:
            self._direct.setdefault(
                (link.source_device, link.target_device), []
            ).append(link)
        for rows in self._direct.values():
            rows.sort(key=lambda item: item.link_id)

    def set_resource_ready(self, resource_id: str, ready: bool) -> None:
        if resource_id in self._resource_ready:
            self._resource_ready[resource_id] = ready

    def estimate(
        self,
        transfer_id: str,
        source_device: str,
        target_device: str,
        payload_bytes: int,
        require_measured: bool,
    ) -> TransferEstimate | None:
        _integer("transfer payload", payload_bytes, 1)
        choices: list[TransferEstimate] = []
        for link in self._direct.get((source_device, target_device), ()):
            if not self._resource_ready.get(link.resource_id, False):
                continue
            status = link.estimate_status(payload_bytes)
            if require_measured and status != "measured":
                continue
            duration_us = link.fixed_latency_us + _ceil_div(
                payload_bytes * 1_000_000, link.bandwidth_bytes_per_s
            )
            scalar_energy_uj = link.fixed_energy_uj + _ceil_div(
                payload_bytes * link.dynamic_pj_per_byte, 1_000_000
            )
            choices.append(TransferEstimate(
                transfer_id=transfer_id,
                link_id=link.link_id,
                source_device=source_device,
                target_device=target_device,
                resource_id=link.resource_id,
                bytes=payload_bytes,
                duration_us=duration_us,
                scalar_energy_uj=scalar_energy_uj,
                domain_power_mw=link.domain_power_mw,
                status=status,
                evidence_ids=link.evidence_ids,
            ))
        if not choices:
            return None
        return min(
            choices,
            key=lambda row: (
                row.duration_us,
                row.scalar_energy_uj,
                row.link_id,
            ),
        )


class MemoryLedger:
    """Conservative persistent weights and activation reservations."""

    def __init__(self, profile: MatmulSystemProfile) -> None:
        self.profile = profile
        self._external: dict[tuple[str, str], int] = {}
        self._weights: dict[tuple[str, str], int] = {}
        self._activations: dict[tuple[str, str], int] = {}
        self._peak = {
            device_id: device.reserved_bytes
            for device_id, device in profile.devices.items()
        }

    def reserve_external(
        self, device_id: str, allocation_id: str, allocation_bytes: int
    ) -> None:
        _integer("external allocation bytes", allocation_bytes, 1)
        key = (device_id, allocation_id)
        old = self._external.get(key)
        if old is not None and old != allocation_bytes:
            raise MatmulScheduleError("external allocation size changed")
        self.set_external(device_id, allocation_id, allocation_bytes)

    def set_external(
        self, device_id: str, allocation_id: str, allocation_bytes: int
    ) -> None:
        device = self.profile.devices.get(device_id)
        if device is None:
            raise MatmulScheduleError("external allocation device is unknown")
        _string("external allocation id", allocation_id)
        _integer("external allocation bytes", allocation_bytes)
        key = (device_id, allocation_id)
        old_present = key in self._external
        old = self._external.get(key, 0)
        if allocation_bytes:
            self._external[key] = allocation_bytes
        else:
            self._external.pop(key, None)
        usage = self.usage_by_device()[device_id]
        if device.memory_capacity_bytes and usage > device.memory_capacity_bytes:
            if old_present:
                self._external[key] = old
            else:
                self._external.pop(key, None)
            raise MatmulScheduleError("external allocation exceeds device memory")
        self._peak[device_id] = max(self._peak[device_id], usage)

    def usage_by_device(self) -> dict[str, int]:
        result = {
            device_id: device.reserved_bytes
            for device_id, device in self.profile.devices.items()
        }
        for collection in (self._external, self._weights, self._activations):
            for (device_id, _), amount in collection.items():
                result[device_id] += amount
        return result

    def available_by_device(self) -> dict[str, int]:
        usage = self.usage_by_device()
        return {
            device_id: (
                0
                if device.memory_capacity_bytes == 0
                else max(0, device.memory_capacity_bytes - usage[device_id])
            )
            for device_id, device in self.profile.devices.items()
        }

    def _old_activation(self, program_id: str, device_id: str) -> int:
        return self._activations.get((device_id, program_id), 0)

    def preview_matmul(
        self,
        program: ModelProgram,
        op: MatmulOp,
        columns: Mapping[str, int],
        input_state: ActivationState,
        output_state: ActivationState,
    ) -> MemoryPreview:
        before = self.usage_by_device()
        source_bytes = input_state.bytes_by_device(op.input_bytes)
        output_bytes = output_state.bytes_by_device()
        peak = dict(before)
        after = dict(before)
        weight_required: dict[str, int] = {}
        weight_growth: dict[str, int] = {}
        output_activation: dict[str, int] = {}
        reason = ""

        for device_id, device in self.profile.devices.items():
            if device_id == self.profile.policy.host_device_id:
                continue
            old_activation = self._old_activation(program.program_id, device_id)
            if old_activation != source_bytes.get(device_id, 0):
                # The ledger can be more conservative after an earlier future
                # reservation, but it must never understate the live shard.
                old_activation = max(
                    old_activation, source_bytes.get(device_id, 0)
                )
            target_columns = columns.get(device_id, 0)
            required_weight = (
                _portion(op.weight_bytes, target_columns, op.n)
                if target_columns
                else 0
            )
            weight_key = (device_id, f"{program.model_id}:{op.weight_id}")
            existing_weight = self._weights.get(weight_key, 0)
            growth = max(0, required_weight - existing_weight)
            target_output = output_bytes.get(device_id, 0)
            if target_columns:
                input_peak = op.input_bytes
            else:
                input_peak = old_activation
            peak[device_id] = (
                before[device_id]
                + growth
                + max(0, input_peak - old_activation)
                + target_output
            )
            after[device_id] = (
                before[device_id]
                + growth
                - self._old_activation(program.program_id, device_id)
                + target_output
            )
            weight_required[device_id] = required_weight
            weight_growth[device_id] = growth
            output_activation[device_id] = target_output
            if device.memory_capacity_bytes and (
                peak[device_id] > device.memory_capacity_bytes
                or after[device_id] > device.memory_capacity_bytes
            ):
                reason = f"MEMORY_CAPACITY:{device_id}"

        available = {
            device_id: (
                0
                if device.memory_capacity_bytes == 0
                else max(0, device.memory_capacity_bytes - after[device_id])
            )
            for device_id, device in self.profile.devices.items()
        }
        return MemoryPreview(
            feasible=not reason,
            reason=reason,
            usage_before_bytes=before,
            peak_usage_bytes=peak,
            usage_after_bytes=after,
            available_after_bytes=available,
            weight_required_bytes=weight_required,
            weight_growth_bytes=weight_growth,
            output_activation_bytes=output_activation,
        )

    def commit_matmul(
        self,
        program: ModelProgram,
        op: MatmulOp,
        preview: MemoryPreview,
    ) -> None:
        if not preview.feasible:
            raise MatmulScheduleError("cannot commit infeasible memory preview")
        for device_id, required in preview.weight_required_bytes.items():
            if required:
                key = (device_id, f"{program.model_id}:{op.weight_id}")
                self._weights[key] = max(self._weights.get(key, 0), required)
        for device_id in self.profile.devices:
            if device_id == self.profile.policy.host_device_id:
                continue
            key = (device_id, program.program_id)
            output = preview.output_activation_bytes.get(device_id, 0)
            if output:
                self._activations[key] = output
            else:
                self._activations.pop(key, None)
            self._peak[device_id] = max(
                self._peak[device_id], preview.peak_usage_bytes[device_id]
            )

    def resize_activation(
        self, program_id: str, state: ActivationState
    ) -> None:
        sizes = state.bytes_by_device()
        usage = self.usage_by_device()
        next_usage = dict(usage)
        for device_id, device in self.profile.devices.items():
            if device_id == self.profile.policy.host_device_id:
                continue
            old = self._activations.get((device_id, program_id), 0)
            next_usage[device_id] = usage[device_id] - old + sizes.get(device_id, 0)
            if (
                device.memory_capacity_bytes
                and next_usage[device_id] > device.memory_capacity_bytes
            ):
                raise MatmulScheduleError(
                    f"non-matmul activation exceeds {device_id} memory"
                )
        for device_id in self.profile.devices:
            if device_id == self.profile.policy.host_device_id:
                continue
            key = (device_id, program_id)
            amount = sizes.get(device_id, 0)
            if amount:
                self._activations[key] = amount
            else:
                self._activations.pop(key, None)
            self._peak[device_id] = max(
                self._peak[device_id], next_usage[device_id]
            )

    def release_activation(self, program_id: str) -> None:
        for device_id in self.profile.devices:
            self._activations.pop((device_id, program_id), None)

    def weight_allocations(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for (device_id, allocation_id), amount in sorted(self._weights.items()):
            result.setdefault(device_id, {})[allocation_id] = amount
        return result

    def external_allocations(self) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for (device_id, allocation_id), amount in sorted(self._external.items()):
            result.setdefault(device_id, {})[allocation_id] = amount
        return result

    @property
    def peak_by_device(self) -> Mapping[str, int]:
        return dict(self._peak)


class PowerTimeline:
    """Integrate the max power envelope in each physical energy domain."""

    def __init__(
        self,
        domains: Mapping[str, EnergyDomainProfile],
        origin_us: int,
    ) -> None:
        self.domains = dict(domains)
        self.origin_us = origin_us
        self.makespan_us = origin_us
        self.intervals: list[PowerInterval] = []
        self.scalar_energy_nj = 0

    def _domain_energy_nj(
        self,
        domain_id: str,
        intervals: Sequence[PowerInterval],
        makespan_us: int,
    ) -> int:
        if makespan_us <= self.origin_us:
            return 0
        rows = [
            row
            for row in intervals
            if row.domain_id == domain_id
            and row.end_us > self.origin_us
            and row.start_us < makespan_us
        ]
        points = {self.origin_us, makespan_us}
        for row in rows:
            points.add(max(self.origin_us, row.start_us))
            points.add(min(makespan_us, row.end_us))
        ordered = sorted(points)
        idle = self.domains[domain_id].idle_power_mw
        energy_nj = 0
        for left, right in zip(ordered, ordered[1:]):
            if right <= left:
                continue
            power = max(
                [idle]
                + [
                    row.power_mw
                    for row in rows
                    if row.start_us < right and left < row.end_us
                ]
            )
            energy_nj += power * (right - left)
        return energy_nj

    def energy_by_domain_nj(
        self,
        extra: Sequence[PowerInterval] = (),
        makespan_us: int | None = None,
    ) -> dict[str, int]:
        finish = self.makespan_us if makespan_us is None else makespan_us
        intervals = self.intervals + list(extra)
        return {
            domain_id: self._domain_energy_nj(
                domain_id, intervals, finish
            )
            for domain_id in self.domains
        }

    def total_energy_nj(
        self,
        extra: Sequence[PowerInterval] = (),
        scalar_energy_uj: int = 0,
        makespan_us: int | None = None,
    ) -> int:
        return (
            sum(self.energy_by_domain_nj(extra, makespan_us).values())
            + self.scalar_energy_nj
            + scalar_energy_uj * 1000
        )

    def incremental_energy_nj(
        self,
        extra: Sequence[PowerInterval],
        scalar_energy_uj: int,
        makespan_us: int,
    ) -> int:
        before = self.total_energy_nj()
        after = self.total_energy_nj(
            extra,
            scalar_energy_uj,
            max(self.makespan_us, makespan_us),
        )
        return after - before

    def commit(
        self,
        intervals: Sequence[PowerInterval],
        scalar_energy_uj: int,
        makespan_us: int,
    ) -> None:
        self.intervals.extend(intervals)
        self.scalar_energy_nj += scalar_energy_uj * 1000
        self.makespan_us = max(self.makespan_us, makespan_us)


@dataclass
class _ProgramState:
    program: ModelProgram
    enqueue_sequence: int
    cursor: int
    ready_us: int
    activation: ActivationState
    finalized: bool = False


def _sample_aligned_values(
    total: int,
    alignment: int,
    minimum: int,
    maximum: int,
    search_points: int,
) -> tuple[int, ...]:
    if alignment <= 0 or maximum < minimum:
        return ()
    first = _ceil_div(minimum, alignment) * alignment
    last = maximum // alignment * alignment
    if first > last:
        return ()
    units = list(range(first // alignment, last // alignment + 1))
    if len(units) <= search_points:
        return tuple(unit * alignment for unit in units)
    selected = {units[0], units[-1]}
    for index in range(search_points):
        position = round(index * (len(units) - 1) / (search_points - 1))
        selected.add(units[position])
    return tuple(sorted(unit * alignment for unit in selected))


class MatmulPlanner:
    """Energy-first matmul placement over a finite causal virtual queue."""

    def __init__(
        self,
        profile: MatmulSystemProfile,
        timeline: ResourceTimeline | None = None,
    ) -> None:
        missing = set(DEVICE_ORDER) - set(profile.devices)
        if missing:
            raise MatmulScheduleError(
                "matmul VQ profile lacks devices: " + ",".join(sorted(missing))
            )
        if profile.policy.host_device_id != "cpu":
            raise MatmulScheduleError("matmul VQ host device must be cpu")
        self.profile = profile
        self.kernels = KernelCatalog(profile)
        self.links = LinkCatalog(profile)
        if timeline is None:
            self.resources = ResourceTimeline(profile.resources)
        else:
            try:
                timeline.require_compatible(profile.resources)
            except SchedulerError as exc:
                raise MatmulScheduleError(str(exc)) from exc
            self.resources = timeline
        self.memory = MemoryLedger(profile)
        self.power: PowerTimeline | None = None
        self._device_ready = {
            device_id: self.resources.is_ready(device.resource_id)
            for device_id, device in profile.devices.items()
        }
        self._programs: dict[str, _ProgramState] = {}
        self._enqueue_sequence = 0
        self._started = False
        self._decisions: list[dict[str, object]] = []
        self._rejections: dict[str, int] = {}
        self._origin_us: int | None = None
        self._runtime_events: list[dict[str, object]] = []
        self._invalidated_owners: set[str] = set()

    def reserve_external_memory(
        self, device_id: str, allocation_id: str, allocation_bytes: int
    ) -> None:
        if self._started:
            raise MatmulScheduleError(
                "external memory must be reserved before scheduling starts"
            )
        self.memory.reserve_external(device_id, allocation_id, allocation_bytes)

    def update_external_memory(
        self,
        device_id: str,
        allocation_id: str,
        allocation_bytes: int,
        at_us: int,
    ) -> None:
        _integer("external memory update at_us", at_us)
        self.memory.set_external(device_id, allocation_id, allocation_bytes)
        self._runtime_events.append({
            "kind": "external_memory_update",
            "at_us": at_us,
            "device_id": device_id,
            "allocation_id": allocation_id,
            "allocation_bytes": allocation_bytes,
        })

    def set_resource_ready(
        self, resource_id: str, ready: bool, at_us: int
    ) -> tuple[str, ...]:
        _string("runtime resource id", resource_id)
        _boolean("runtime resource ready", ready)
        _integer("runtime resource update at_us", at_us)
        if ready:
            self.resources.restore_resource(resource_id)
            affected: tuple[str, ...] = ()
        else:
            affected = self.resources.revoke_resource(resource_id, at_us)
            self._invalidated_owners.update(affected)
        self.links.set_resource_ready(resource_id, ready)
        for device_id, device in self.profile.devices.items():
            if device.resource_id == resource_id:
                self._device_ready[device_id] = ready
        self._runtime_events.append({
            "kind": "resource_readiness_update",
            "at_us": at_us,
            "resource_id": resource_id,
            "ready": ready,
            "invalidated_owners": list(affected),
        })
        return affected

    def set_device_ready(
        self, device_id: str, ready: bool, at_us: int
    ) -> tuple[str, ...]:
        device = self.profile.devices.get(device_id)
        if device is None:
            raise MatmulScheduleError("runtime device is unknown")
        return self.set_resource_ready(device.resource_id, ready, at_us)

    def reserve_external_resource(
        self,
        resource_id: str,
        reservation_id: str,
        start_us: int,
        finish_us: int,
        slots: int = 1,
    ) -> tuple[LeaseRecord, ...]:
        _string("external resource id", resource_id)
        _string("external resource reservation id", reservation_id)
        _integer("external resource start_us", start_us)
        _integer("external resource finish_us", finish_us, start_us + 1)
        _integer("external resource slots", slots, 1)
        duration_us = finish_us - start_us
        demand = LeaseDemand(
            lease_id=f"external:{reservation_id}",
            resource_id=resource_id,
            slots=slots,
            start_offset_us=0,
            duration_us=duration_us,
            duration_upper_us=duration_us,
        )
        try:
            preview = self.resources.preview_leases(
                (demand,), start_us, duration_us, duration_us
            )
        except SchedulerError as exc:
            raise MatmulScheduleError(
                f"cannot reserve external resource: {exc}"
            ) from exc
        if preview.start_us != start_us:
            raise MatmulScheduleError("external resource window is already busy")
        leases = self.resources.commit_leases(
            preview, f"external:{reservation_id}"
        )
        self._runtime_events.append({
            "kind": "external_resource_reservation",
            "at_us": start_us,
            "resource_id": resource_id,
            "reservation_id": reservation_id,
            "finish_us": finish_us,
            "leases": [_lease_to_json(row) for row in leases],
        })
        return leases

    def release_resource_lease(self, token: str, actual_end_us: int) -> None:
        _string("released resource lease token", token)
        _integer("released resource lease actual_end_us", actual_end_us)
        try:
            self.resources.release(token, actual_end_us)
        except SchedulerError as exc:
            raise MatmulScheduleError(f"cannot release resource lease: {exc}") from exc
        self._runtime_events.append({
            "kind": "resource_lease_release",
            "at_us": actual_end_us,
            "token": token,
        })

    def resource_forecast(self, at_us: int) -> Mapping[str, object]:
        _integer("resource forecast at_us", at_us)
        resources = self.resources.resource_snapshot(at_us)
        usage = self.memory.usage_by_device()
        available = self.memory.available_by_device()
        devices = {}
        for device_id, device in sorted(self.profile.devices.items()):
            resource = resources[device.resource_id]
            ready = self._device_ready[device_id] and resource["ready"]
            devices[device_id] = {
                "resource_id": device.resource_id,
                "ready": ready,
                "earliest_available_us": resource["next_free_us"],
                "predicted_free_us": (
                    resource["reserved_until_us"] if ready else None
                ),
                "active_until_us": resource["active_until_us"],
                "reserved_until_us": resource["reserved_until_us"],
                "active_owners": list(resource["active_owners"]),
                "queued_owners": list(resource["queued_owners"]),
                "memory_capacity_bytes": device.memory_capacity_bytes,
                "memory_usage_bytes": usage[device_id],
                "memory_available_bytes": available[device_id],
            }
        return {
            "at_us": at_us,
            "devices": devices,
            "resources": resources,
        }

    def enqueue(self, program: ModelProgram) -> None:
        if program.program_id in self._programs:
            raise MatmulScheduleError("duplicate virtual-queue program id")
        pending = sum(
            len(state.program.ops) - state.cursor
            for state in self._programs.values()
            if not state.finalized
        )
        if pending + len(program.ops) > self.profile.policy.queue_limit:
            raise MatmulScheduleError("virtual queue is full")
        if self._started and self._origin_us is not None:
            if program.arrival_us < self._origin_us:
                raise MatmulScheduleError(
                    "cannot enqueue an earlier arrival after scheduling starts"
                )
        self._enqueue_sequence += 1
        self._programs[program.program_id] = _ProgramState(
            program=program,
            enqueue_sequence=self._enqueue_sequence,
            cursor=0,
            ready_us=program.arrival_us,
            activation=ActivationState.full("cpu", 0),
        )
        self._origin_us = (
            program.arrival_us
            if self._origin_us is None
            else min(self._origin_us, program.arrival_us)
        )

    def _record_rejection(self, reason: str) -> None:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1

    def _kernel_for(
        self, device_id: str, op: MatmulOp, columns: int
    ) -> KernelEstimate | None:
        if columns == 0:
            return None
        if device_id not in op.allowed_devices:
            return None
        if not self._device_ready[device_id]:
            return None
        estimate = self.kernels.estimate(device_id, op, columns)
        if estimate is None:
            return None
        if self.profile.policy.require_measured and estimate.status != "measured":
            return None
        return estimate

    def _remote_values(self, device_id: str, op: MatmulOp) -> tuple[int, ...]:
        if device_id not in op.allowed_devices:
            return (0,)
        rows = self.kernels.profiles_for(device_id, op)
        if not rows or not self._device_ready[device_id]:
            return (0,)
        alignment = self.kernels.alignment(device_id, op)
        minimum = min(row.minimum_n for row in rows)
        maxima = [row.maximum_n for row in rows if row.maximum_n]
        maximum = min(op.n, max(maxima)) if maxima else op.n
        values = _sample_aligned_values(
            op.n,
            alignment,
            minimum,
            maximum,
            self.profile.policy.split_search_points,
        )
        return (0,) + values

    def _column_candidates(self, op: MatmulOp) -> tuple[Mapping[str, int], ...]:
        gpu_values = set(self._remote_values("gpu", op))
        phone_values = set(self._remote_values("phone", op))

        # Add exact complements so GPU plus phone full-offload cuts are not
        # missed by independent sampling.
        for value in tuple(gpu_values):
            complement = op.n - value
            if complement >= 0 and self._kernel_for("phone", op, complement):
                phone_values.add(complement)
        for value in tuple(phone_values):
            complement = op.n - value
            if complement >= 0 and self._kernel_for("gpu", op, complement):
                gpu_values.add(complement)

        candidates: list[Mapping[str, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for gpu_columns in sorted(gpu_values):
            for phone_columns in sorted(phone_values):
                cpu_columns = op.n - gpu_columns - phone_columns
                if cpu_columns < 0:
                    continue
                columns = {
                    "cpu": cpu_columns,
                    "gpu": gpu_columns,
                    "phone": phone_columns,
                }
                key = (cpu_columns, gpu_columns, phone_columns)
                if key in seen:
                    continue
                if any(
                    amount > 0
                    and self._kernel_for(device_id, op, amount) is None
                    for device_id, amount in columns.items()
                ):
                    continue
                seen.add(key)
                candidates.append(columns)
        candidates.sort(
            key=lambda row: (row["phone"], row["gpu"], row["cpu"])
        )
        return tuple(candidates)

    def _transfer(
        self,
        op_id: str,
        label: str,
        source_device: str,
        target_device: str,
        payload_bytes: int,
    ) -> TransferEstimate | None:
        return self.links.estimate(
            f"{op_id}:{label}",
            source_device,
            target_device,
            payload_bytes,
            self.profile.policy.require_measured,
        )

    @staticmethod
    def _phase_for_transfer(
        transfer: TransferEstimate, start_offset_us: int
    ) -> Phase:
        return Phase(
            phase_id=transfer.transfer_id,
            kind="transfer",
            resource_id=transfer.resource_id,
            start_offset_us=start_offset_us,
            duration_us=transfer.duration_us,
            domain_power_mw=transfer.domain_power_mw,
            transfer=transfer,
        )

    @staticmethod
    def _phase_for_kernel(
        op_id: str,
        kernel: KernelEstimate,
        resource_id: str,
        start_offset_us: int,
    ) -> Phase:
        return Phase(
            phase_id=f"{op_id}:compute:{kernel.device_id}",
            kind="compute",
            resource_id=resource_id,
            start_offset_us=start_offset_us,
            duration_us=kernel.duration_us,
            domain_power_mw=kernel.domain_power_mw,
            device_id=kernel.device_id,
            kernel=kernel,
        )

    def _build_candidate(
        self,
        state: _ProgramState,
        op: MatmulOp,
        columns: Mapping[str, int],
    ) -> CandidatePlan | None:
        output_state = ActivationState.from_columns(
            op.output_bytes, op.n, columns
        )
        memory = self.memory.preview_matmul(
            state.program,
            op,
            columns,
            state.activation,
            output_state,
        )
        if not memory.feasible:
            self._record_rejection(memory.reason)
            return None

        kernels: list[KernelEstimate] = []
        for device_id in DEVICE_ORDER:
            amount = columns.get(device_id, 0)
            if not amount:
                continue
            kernel = self._kernel_for(device_id, op, amount)
            if kernel is None:
                self._record_rejection(f"NO_KERNEL:{device_id}")
                return None
            kernels.append(kernel)

        source = state.activation.bytes_by_device(op.input_bytes)
        targets = {device for device, amount in columns.items() if amount}
        phases: list[Phase] = []
        transfers: list[TransferEstimate] = []
        gather_end: dict[str, int] = {}
        resource_end: dict[str, int] = {}

        for source_device in ("gpu", "phone"):
            payload = source.get(source_device, 0)
            if not payload:
                continue
            needed_at_host = "cpu" in targets
            needed_by_other = any(
                target not in {"cpu", source_device} for target in targets
            )
            if not (needed_at_host or needed_by_other):
                continue
            transfer = self._transfer(
                op.op_id,
                f"gather-{source_device}",
                source_device,
                "cpu",
                payload,
            )
            if transfer is None:
                self._record_rejection(f"NO_LINK:{source_device}->cpu")
                return None
            phase = self._phase_for_transfer(transfer, 0)
            phases.append(phase)
            transfers.append(transfer)
            gather_end[source_device] = phase.duration_us
            resource_end[phase.resource_id] = max(
                resource_end.get(phase.resource_id, 0), phase.duration_us
            )

        branch_ready: dict[str, int] = {}
        if "cpu" in targets:
            remote_needed = [
                gather_end.get(source_device, 0)
                for source_device in ("gpu", "phone")
                if source.get(source_device, 0)
            ]
            if any(
                source.get(device_id, 0) and device_id not in gather_end
                for device_id in ("gpu", "phone")
            ):
                self._record_rejection("CPU_INPUT_NOT_GATHERED")
                return None
            branch_ready["cpu"] = max(remote_needed, default=0)

        for target_device in ("gpu", "phone"):
            if target_device not in targets:
                continue
            missing_sources = [
                source_device
                for source_device, payload in source.items()
                if payload and source_device != target_device
            ]
            missing_bytes = sum(source[item] for item in missing_sources)
            if not missing_bytes:
                branch_ready[target_device] = 0
                continue
            data_ready = max(
                [
                    gather_end.get(source_device, 0)
                    for source_device in missing_sources
                    if source_device != "cpu"
                ]
                or [0]
            )
            if any(
                source_device != "cpu" and source_device not in gather_end
                for source_device in missing_sources
            ):
                self._record_rejection("REMOTE_INPUT_NOT_GATHERED")
                return None
            transfer = self._transfer(
                op.op_id,
                f"upload-{target_device}",
                "cpu",
                target_device,
                missing_bytes,
            )
            if transfer is None:
                self._record_rejection(f"NO_LINK:cpu->{target_device}")
                return None
            start_offset = max(
                data_ready, resource_end.get(transfer.resource_id, 0)
            )
            phase = self._phase_for_transfer(transfer, start_offset)
            phases.append(phase)
            transfers.append(transfer)
            resource_end[phase.resource_id] = max(
                resource_end.get(phase.resource_id, 0),
                start_offset + phase.duration_us,
            )
            branch_ready[target_device] = start_offset + phase.duration_us

        for kernel in kernels:
            phase = self._phase_for_kernel(
                op.op_id,
                kernel,
                self.profile.devices[kernel.device_id].resource_id,
                branch_ready.get(kernel.device_id, 0),
            )
            phases.append(phase)

        service_us = max(
            phase.start_offset_us + phase.duration_us for phase in phases
        )
        measured = all(kernel.status == "measured" for kernel in kernels) and all(
            transfer.status == "measured" for transfer in transfers
        )
        evidence = tuple(sorted({
            evidence_id
            for row in tuple(kernels) + tuple(transfers)
            for evidence_id in row.evidence_ids
        }))
        route_id = "n:" + ",".join(
            f"{device}={columns[device]}" for device in DEVICE_ORDER
        )
        return CandidatePlan(
            route_id=route_id,
            columns=dict(columns),
            output_state=output_state,
            kernels=tuple(kernels),
            transfers=tuple(transfers),
            phases=tuple(sorted(
                phases,
                key=lambda phase: (
                    phase.start_offset_us,
                    phase.resource_id,
                    phase.phase_id,
                ),
            )),
            service_us=service_us,
            scalar_energy_uj=sum(row.scalar_energy_uj for row in transfers),
            measured=measured,
            evidence_ids=evidence,
            memory=memory,
        )

    @staticmethod
    def _lease_demands(candidate: CandidatePlan) -> tuple[LeaseDemand, ...]:
        return tuple(
            LeaseDemand(
                lease_id=phase.phase_id,
                resource_id=phase.resource_id,
                slots=1,
                start_offset_us=phase.start_offset_us,
                duration_us=phase.duration_us,
                duration_upper_us=phase.duration_us,
            )
            for phase in candidate.phases
        )

    def _power_intervals(
        self,
        candidate: CandidatePlan,
        preview: LeasePreview,
    ) -> tuple[PowerInterval, ...]:
        intervals = [PowerInterval(
            domain_id=self.profile.policy.host_domain_id,
            start_us=preview.start_us,
            end_us=preview.finish_us,
            power_mw=self.profile.policy.host_active_power_mw,
        )]
        for phase in candidate.phases:
            start_us = preview.start_us + phase.start_offset_us
            end_us = start_us + phase.duration_us
            intervals.extend(
                PowerInterval(domain_id, start_us, end_us, power)
                for domain_id, power in phase.domain_power_mw.items()
            )
        return tuple(intervals)

    def _preview_candidate(
        self,
        state: _ProgramState,
        candidate: CandidatePlan,
    ) -> CandidatePlan | None:
        assert self.power is not None
        try:
            preview = self.resources.preview_leases(
                self._lease_demands(candidate),
                state.ready_us,
                candidate.service_us,
                candidate.service_us,
            )
        except SchedulerError:
            self._record_rejection("RESOURCE_NOT_READY")
            return None
        intervals = self._power_intervals(candidate, preview)
        incremental = self.power.incremental_energy_nj(
            intervals,
            candidate.scalar_energy_uj,
            preview.finish_us,
        )
        return replace(
            candidate,
            lease_preview=preview,
            incremental_energy_nj=incremental,
            objective_energy_nj=incremental,
            lookahead_finish_us=preview.finish_us,
        )

    @staticmethod
    def _mandatory_gather_bytes(
        state: _ProgramState,
        op: MatmulOp,
    ) -> int | None:
        output_bytes = op.output_bytes
        for following in state.program.ops[state.cursor + 1:]:
            if isinstance(following, MatmulOp):
                return None
            output_bytes = following.output_bytes or output_bytes
            if not following.shard_safe:
                return output_bytes
        return output_bytes

    def _add_gather_lookahead(
        self,
        state: _ProgramState,
        op: MatmulOp,
        candidate: CandidatePlan,
    ) -> CandidatePlan | None:
        assert candidate.lease_preview is not None
        assert self.power is not None
        output_bytes = self._mandatory_gather_bytes(state, op)
        if output_bytes is None:
            return candidate
        source = candidate.output_state.bytes_by_device(output_bytes)
        transfers: list[TransferEstimate] = []
        phases: list[Phase] = []
        resource_end: dict[str, int] = {}
        for source_device in ("gpu", "phone"):
            payload = source.get(source_device, 0)
            if not payload:
                continue
            transfer = self._transfer(
                op.op_id,
                f"lookahead-final-{source_device}",
                source_device,
                "cpu",
                payload,
            )
            if transfer is None:
                self._record_rejection(
                    f"NO_LOOKAHEAD_LINK:{source_device}->cpu"
                )
                return None
            start_offset = resource_end.get(transfer.resource_id, 0)
            phase = self._phase_for_transfer(transfer, start_offset)
            phases.append(phase)
            transfers.append(transfer)
            resource_end[transfer.resource_id] = (
                start_offset + transfer.duration_us
            )
        if not phases:
            return candidate
        service_us = max(
            phase.start_offset_us + phase.duration_us for phase in phases
        )
        demands = tuple(
            LeaseDemand(
                lease_id=phase.phase_id,
                resource_id=phase.resource_id,
                slots=1,
                start_offset_us=phase.start_offset_us,
                duration_us=phase.duration_us,
                duration_upper_us=phase.duration_us,
            )
            for phase in phases
        )
        try:
            preview = self.resources.preview_leases(
                demands,
                candidate.lease_preview.finish_us,
                service_us,
                service_us,
            )
        except SchedulerError:
            self._record_rejection("LOOKAHEAD_RESOURCE_NOT_READY")
            return None
        candidate_intervals = self._power_intervals(
            candidate, candidate.lease_preview
        )
        gather_intervals = [PowerInterval(
            domain_id=self.profile.policy.host_domain_id,
            start_us=preview.start_us,
            end_us=preview.finish_us,
            power_mw=self.profile.policy.host_active_power_mw,
        )]
        for phase in phases:
            start_us = preview.start_us + phase.start_offset_us
            gather_intervals.extend(
                PowerInterval(
                    domain_id,
                    start_us,
                    start_us + phase.duration_us,
                    power,
                )
                for domain_id, power in phase.domain_power_mw.items()
            )
        scalar_energy_uj = candidate.scalar_energy_uj + sum(
            row.scalar_energy_uj for row in transfers
        )
        objective = self.power.incremental_energy_nj(
            candidate_intervals + tuple(gather_intervals),
            scalar_energy_uj,
            preview.finish_us,
        )
        return replace(
            candidate,
            objective_energy_nj=objective,
            lookahead_finish_us=preview.finish_us,
            lookahead_transfers=tuple(transfers),
        )

    def _operator_deadline(
        self,
        state: _ProgramState,
        op: MatmulOp,
        baseline: CandidatePlan,
    ) -> int:
        assert baseline.lease_preview is not None
        baseline_elapsed = baseline.lookahead_finish_us - state.ready_us
        latency_limit = state.ready_us + _ceil_div(
            baseline_elapsed * self.profile.policy.latency_limit_ppm,
            1_000_000,
        )
        deadline = state.program.deadline_us
        if op.deadline_us:
            deadline = min(deadline, op.deadline_us)
        return min(deadline, latency_limit)

    def _commit_candidate(
        self,
        state: _ProgramState,
        op: MatmulOp,
        selected: CandidatePlan,
        baseline: CandidatePlan,
        operator_deadline_us: int,
    ) -> None:
        assert selected.lease_preview is not None
        assert baseline.lease_preview is not None
        assert self.power is not None
        original_ready_us = state.ready_us
        leases = self.resources.commit_leases(
            selected.lease_preview,
            f"{state.program.program_id}:{op.op_id}",
        )
        intervals = self._power_intervals(selected, selected.lease_preview)
        self.power.commit(
            intervals,
            selected.scalar_energy_uj,
            selected.lease_preview.finish_us,
        )
        self.memory.commit_matmul(state.program, op, selected.memory)
        input_state = state.activation.rescale(op.input_bytes)
        state.activation = selected.output_state
        state.ready_us = selected.lease_preview.finish_us
        state.cursor += 1

        self._decisions.append({
            "kind": "matmul",
            "program_id": state.program.program_id,
            "model_id": state.program.model_id,
            "op_id": op.op_id,
            "layer_id": op.layer_id,
            "queue_transitions": list(QUEUE_TRANSITIONS),
            "shape": {"m": op.m, "k": op.k, "n": op.n},
            "sizes": {
                "input_bytes": op.input_bytes,
                "output_bytes": op.output_bytes,
                "weight_bytes": op.weight_bytes,
                "compute_ops": op.compute_ops,
            },
            "input_residency": _activation_to_json(input_state),
            "output_residency": _activation_to_json(selected.output_state),
            "placement": dict(selected.columns),
            "split_axis": "output_columns_n",
            "route_id": selected.route_id,
            "ready_us": original_ready_us,
            "start_us": selected.lease_preview.start_us,
            "finish_us": selected.lease_preview.finish_us,
            "service_us": selected.service_us,
            "queue_us": selected.lease_preview.start_us - original_ready_us,
            "queue_by_resource_us": dict(
                selected.lease_preview.queue_by_resource_us
            ),
            "blocking_resources": list(
                selected.lease_preview.blocking_resources
            ),
            "cpu_baseline_finish_us": baseline.lease_preview.finish_us,
            "cpu_baseline_effective_finish_us": baseline.lookahead_finish_us,
            "operator_deadline_us": operator_deadline_us,
            "incremental_energy_uj": _ceil_div(
                selected.incremental_energy_nj, 1000
            ),
            "objective_energy_with_mandatory_gather_uj": _ceil_div(
                selected.objective_energy_nj, 1000
            ),
            "lookahead_finish_us": selected.lookahead_finish_us,
            "lookahead_transfers": [
                _transfer_to_json(row) for row in selected.lookahead_transfers
            ],
            "measured": selected.measured,
            "weight_prepare": {
                "source": "device_local_storage",
                "latency_us": 0,
                "energy_uj": 0,
                "assumption": "free_by_policy",
            },
            "kernels": [_kernel_to_json(row) for row in selected.kernels],
            "transfers": [_transfer_to_json(row) for row in selected.transfers],
            "leases": [_lease_to_json(row) for row in leases],
            "memory": _memory_to_json(selected.memory),
            "evidence_ids": list(selected.evidence_ids),
        })

    def _schedule_matmul(self, state: _ProgramState, op: MatmulOp) -> None:
        built = [
            candidate
            for columns in self._column_candidates(op)
            if (candidate := self._build_candidate(state, op, columns)) is not None
        ]
        if not built:
            raise MatmulScheduleError(f"no feasible matmul route for {op.op_id}")
        previewed = [
            candidate
            for row in built
            if (candidate := self._preview_candidate(state, row)) is not None
        ]
        previewed = [
            candidate
            for row in previewed
            if (candidate := self._add_gather_lookahead(state, op, row))
            is not None
        ]
        if not previewed:
            raise MatmulScheduleError(f"no ready matmul route for {op.op_id}")
        baseline = next(
            (
                row
                for row in previewed
                if row.columns == {"cpu": op.n, "gpu": 0, "phone": 0}
            ),
            None,
        )
        if baseline is None:
            raise MatmulScheduleError(f"CPU baseline is unavailable for {op.op_id}")
        operator_deadline = self._operator_deadline(state, op, baseline)
        feasible = [
            row
            for row in previewed
            if row.lookahead_finish_us <= operator_deadline
        ]
        for row in previewed:
            if row not in feasible:
                self._record_rejection("LATENCY_OR_DEADLINE")
        if not feasible:
            raise MatmulScheduleError(
                f"no matmul route meets deadline for {op.op_id}"
            )
        selected = min(
            feasible,
            key=lambda row: (
                row.objective_energy_nj,
                row.lookahead_finish_us,
                row.route_id,
            ),
        )
        self._commit_candidate(
            state, op, selected, baseline, operator_deadline
        )

    def _gather_activation(
        self,
        state: _ProgramState,
        op_id: str,
        layer_id: str,
        kind: str,
        output_bytes: int,
    ) -> None:
        assert self.power is not None
        source = state.activation.bytes_by_device(output_bytes)
        transfers: list[TransferEstimate] = []
        phases: list[Phase] = []
        for source_device in ("gpu", "phone"):
            payload = source.get(source_device, 0)
            if not payload:
                continue
            transfer = self._transfer(
                op_id,
                f"final-{source_device}",
                source_device,
                "cpu",
                payload,
            )
            if transfer is None:
                raise MatmulScheduleError(
                    f"cannot gather {source_device} activation"
                )
            transfers.append(transfer)
            phases.append(self._phase_for_transfer(transfer, 0))

        if not phases:
            old_state = state.activation
            state.activation = ActivationState.full("cpu", output_bytes)
            self.memory.release_activation(state.program.program_id)
            self._decisions.append({
                "kind": kind,
                "program_id": state.program.program_id,
                "model_id": state.program.model_id,
                "op_id": op_id,
                "layer_id": layer_id,
                "queue_transitions": list(QUEUE_TRANSITIONS),
                "input_residency": _activation_to_json(old_state),
                "output_residency": _activation_to_json(state.activation),
                "ready_us": state.ready_us,
                "start_us": state.ready_us,
                "finish_us": state.ready_us,
                "service_us": 0,
                "queue_us": 0,
                "incremental_energy_uj": 0,
                "transfers": [],
                "leases": [],
            })
            return

        service_us = max(phase.duration_us for phase in phases)
        demands = tuple(
            LeaseDemand(
                lease_id=phase.phase_id,
                resource_id=phase.resource_id,
                slots=1,
                start_offset_us=0,
                duration_us=phase.duration_us,
                duration_upper_us=phase.duration_us,
            )
            for phase in phases
        )
        try:
            preview = self.resources.preview_leases(
                demands, state.ready_us, service_us, service_us
            )
        except SchedulerError as exc:
            raise MatmulScheduleError(f"cannot reserve final gather: {exc}") from exc
        if preview.finish_upper_us > state.program.deadline_us:
            raise MatmulScheduleError("final CPU gather misses program deadline")
        candidate = CandidatePlan(
            route_id=f"{kind}:cpu",
            columns={"cpu": 1, "gpu": 0, "phone": 0},
            output_state=ActivationState.full("cpu", output_bytes),
            kernels=(),
            transfers=tuple(transfers),
            phases=tuple(phases),
            service_us=service_us,
            scalar_energy_uj=sum(row.scalar_energy_uj for row in transfers),
            measured=all(row.status == "measured" for row in transfers),
            evidence_ids=tuple(sorted({
                evidence_id
                for row in transfers
                for evidence_id in row.evidence_ids
            })),
            memory=MemoryPreview(
                True, "", {}, {}, {}, {}, {}, {}, {}
            ),
            lease_preview=preview,
        )
        intervals = self._power_intervals(candidate, preview)
        incremental_nj = self.power.incremental_energy_nj(
            intervals, candidate.scalar_energy_uj, preview.finish_us
        )
        leases = self.resources.commit_leases(
            preview, f"{state.program.program_id}:{op_id}"
        )
        self.power.commit(
            intervals, candidate.scalar_energy_uj, preview.finish_us
        )
        original_ready_us = state.ready_us
        old_state = state.activation.rescale(output_bytes)
        state.activation = candidate.output_state
        state.ready_us = preview.finish_us
        self.memory.release_activation(state.program.program_id)
        self._decisions.append({
            "kind": kind,
            "program_id": state.program.program_id,
            "model_id": state.program.model_id,
            "op_id": op_id,
            "layer_id": layer_id,
            "queue_transitions": list(QUEUE_TRANSITIONS),
            "input_residency": _activation_to_json(old_state),
            "output_residency": _activation_to_json(state.activation),
            "ready_us": original_ready_us,
            "start_us": preview.start_us,
            "finish_us": preview.finish_us,
            "service_us": service_us,
            "queue_us": preview.start_us - original_ready_us,
            "incremental_energy_uj": _ceil_div(incremental_nj, 1000),
            "transfers": [_transfer_to_json(row) for row in transfers],
            "leases": [_lease_to_json(row) for row in leases],
        })

    def _schedule_follow(self, state: _ProgramState, op: FollowOp) -> None:
        output_bytes = op.output_bytes or state.activation.total_bytes
        if not op.shard_safe:
            resized = state.activation.rescale(output_bytes)
            self.memory.resize_activation(state.program.program_id, resized)
            state.activation = resized
            self._gather_activation(
                state,
                op.op_id,
                op.layer_id,
                "non_matmul_barrier",
                output_bytes,
            )
            state.cursor += 1
            return
        input_state = state.activation
        resized = state.activation.rescale(output_bytes)
        self.memory.resize_activation(state.program.program_id, resized)
        state.activation = resized
        state.cursor += 1
        self._decisions.append({
            "kind": "non_matmul_follow",
            "program_id": state.program.program_id,
            "model_id": state.program.model_id,
            "op_id": op.op_id,
            "layer_id": op.layer_id,
            "op_kind": op.op_kind,
            "queue_transitions": list(QUEUE_TRANSITIONS),
            "placement": list(state.activation.devices()),
            "input_residency": _activation_to_json(input_state),
            "output_residency": _activation_to_json(state.activation),
            "ready_us": state.ready_us,
            "start_us": state.ready_us,
            "finish_us": state.ready_us,
            "service_us": 0,
            "queue_us": 0,
            "incremental_energy_uj": 0,
            "reason": "inherits_preceding_matmul_placement",
        })

    def _finalize(self, state: _ProgramState) -> None:
        self._gather_activation(
            state,
            f"{state.program.program_id}:final-output",
            "final",
            "final_cpu_gather",
            state.activation.total_bytes,
        )
        state.finalized = True

    def _next_state(self) -> _ProgramState | None:
        rows = [state for state in self._programs.values() if not state.finalized]
        if not rows:
            return None
        return min(
            rows,
            key=lambda state: (
                state.ready_us,
                state.program.deadline_us,
                state.enqueue_sequence,
            ),
        )

    def _ensure_power(self) -> None:
        if self.power is None:
            if self._origin_us is None:
                raise MatmulScheduleError("virtual queue is empty")
            self.power = PowerTimeline(self.profile.domains, self._origin_us)

    def _dispatch_state(
        self,
        now_us: int,
        program_id: str | None,
        allow_future: bool,
    ) -> _ProgramState | None:
        if program_id is not None:
            _string("dispatch program id", program_id)
            state = self._programs.get(program_id)
            if state is None:
                raise MatmulScheduleError("dispatch program is unknown")
            rows = [] if state.finalized else [state]
        else:
            rows = [
                state
                for state in self._programs.values()
                if not state.finalized
            ]
        if not allow_future:
            rows = [state for state in rows if state.ready_us <= now_us]
        if not rows:
            return None
        return min(
            rows,
            key=lambda state: (
                state.ready_us,
                state.program.deadline_us,
                state.enqueue_sequence,
            ),
        )

    def schedule_next(
        self,
        now_us: int,
        program_id: str | None = None,
        allow_future: bool = False,
    ) -> Mapping[str, object] | None:
        """Assign one READY queue entry against the current live calendars."""
        _integer("dispatch now_us", now_us)
        _boolean("dispatch allow_future", allow_future)
        state = self._dispatch_state(now_us, program_id, allow_future)
        if state is None:
            return None
        self._ensure_power()
        self._started = True
        dispatch_ready_us = max(now_us, state.ready_us)
        state.ready_us = dispatch_ready_us
        forecast = self.resource_forecast(dispatch_ready_us)
        decision_count = len(self._decisions)

        if state.cursor == len(state.program.ops):
            self._finalize(state)
        else:
            op = state.program.ops[state.cursor]
            if isinstance(op, MatmulOp):
                if state.activation.total_bytes == 0:
                    state.activation = ActivationState.full("cpu", op.input_bytes)
                self._schedule_matmul(state, op)
            else:
                self._schedule_follow(state, op)

        if len(self._decisions) != decision_count + 1:
            raise MatmulScheduleError("dispatch did not produce one decision")
        decision = self._decisions[-1]
        decision["dispatch_state"] = "ASSIGNED"
        decision["scheduled_at_us"] = now_us
        decision["dispatch_ready_us"] = dispatch_ready_us
        decision["device_free_before_us"] = {
            device_id: row["predicted_free_us"]
            for device_id, row in forecast["devices"].items()
        }
        return dict(decision)

    def enqueue_and_schedule(
        self, program: ModelProgram, now_us: int
    ) -> Mapping[str, object] | None:
        self.enqueue(program)
        return self.schedule_next(now_us)

    def run(self) -> Mapping[str, object]:
        if not self._programs:
            raise MatmulScheduleError("virtual queue is empty")
        while (state := self._next_state()) is not None:
            decision = self.schedule_next(
                state.ready_us,
                program_id=state.program.program_id,
                allow_future=True,
            )
            if decision is None:
                raise MatmulScheduleError("virtual queue stopped making progress")
        return self.result()

    def result(self) -> Mapping[str, object]:
        if self.power is None:
            raise MatmulScheduleError("scheduler has not run")
        domain_nj = self.power.energy_by_domain_nj()
        total_nj = self.power.total_energy_nj()
        programs = []
        all_finalized = all(state.finalized for state in self._programs.values())
        for state in sorted(
            self._programs.values(), key=lambda row: row.enqueue_sequence
        ):
            programs.append({
                "program_id": state.program.program_id,
                "model_id": state.program.model_id,
                "arrival_us": state.program.arrival_us,
                "deadline_us": state.program.deadline_us,
                "finish_us": state.ready_us if state.finalized else None,
                "predicted_ready_us": state.ready_us,
                "deadline_met": (
                    state.ready_us <= state.program.deadline_us
                    if state.finalized else None
                ),
                "op_count": len(state.program.ops),
                "scheduled_op_count": state.cursor,
                "finalized": state.finalized,
                "current_devices": list(state.activation.devices()),
                "final_device": "cpu" if state.finalized else None,
            })
        if self._invalidated_owners:
            status = "REPLAN_REQUIRED"
        elif all_finalized:
            status = "PLANNED_NOT_RUNTIME_CERTIFIED"
        else:
            status = "PARTIAL_PLANNED_NOT_RUNTIME_CERTIFIED"
        return {
            "schema": RESULT_SCHEMA,
            "profile_id": self.profile.profile_id,
            "energy_boundary_id": self.profile.energy_boundary_id,
            "status": status,
            "objective": "minimum_incremental_fleet_energy_under_cpu_latency_limit",
            "split_axis": "output_columns_n",
            "origin_us": self.power.origin_us,
            "makespan_us": self.power.makespan_us,
            "total_energy_uj": _ceil_div(total_nj, 1000),
            "energy_by_domain_uj": {
                domain_id: _ceil_div(amount, 1000)
                for domain_id, amount in domain_nj.items()
            },
            "unattributed_transfer_energy_uj": _ceil_div(
                self.power.scalar_energy_nj, 1000
            ),
            "programs": programs,
            "decisions": list(self._decisions),
            "invalidated_owners": sorted(self._invalidated_owners),
            "runtime_events": list(self._runtime_events),
            "resource_forecast": self.resource_forecast(
                self.power.makespan_us
            ),
            "memory": {
                "capacity_bytes": {
                    device_id: device.memory_capacity_bytes
                    for device_id, device in self.profile.devices.items()
                },
                "current_usage_bytes": self.memory.usage_by_device(),
                "current_available_bytes": self.memory.available_by_device(),
                "peak_usage_bytes": dict(self.memory.peak_by_device),
                "external_allocations": self.memory.external_allocations(),
                "resident_weight_allocations": self.memory.weight_allocations(),
            },
            "candidate_rejections": dict(sorted(self._rejections.items())),
            "assumptions": {
                "weight_acquisition_latency_us": 0,
                "weight_acquisition_energy_uj": 0,
                "non_matmul_cost": "not_profiled; shard-safe ops inherit placement",
                "host_power_during_offload_mw": (
                    self.profile.policy.host_active_power_mw
                ),
                "final_destination": "cpu",
                "physical_dispatch": "not_performed",
                "runtime_policy": (
                    "only unassigned queue entries use updated resources; "
                    "assigned work is not migrated"
                ),
            },
        }


def _activation_to_json(state: ActivationState) -> dict[str, object]:
    return {
        "total_bytes": state.total_bytes,
        "shards": [
            {
                "device_id": device_id,
                "fraction_units": units,
                "fraction_total": state.units,
                "bytes": amount,
            }
            for device_id, units, amount in state.shards
        ],
    }


def _kernel_to_json(row: KernelEstimate) -> dict[str, object]:
    return {
        "profile_id": row.profile_id,
        "device_id": row.device_id,
        "columns": row.columns,
        "compute_ops": row.compute_ops,
        "memory_bytes": row.memory_bytes,
        "compute_us": row.compute_us,
        "memory_us": row.memory_us,
        "duration_us": row.duration_us,
        "domain_power_mw": dict(row.domain_power_mw),
        "status": row.status,
        "evidence_ids": list(row.evidence_ids),
    }


def _transfer_to_json(row: TransferEstimate) -> dict[str, object]:
    return {
        "transfer_id": row.transfer_id,
        "link_id": row.link_id,
        "source_device": row.source_device,
        "target_device": row.target_device,
        "resource_id": row.resource_id,
        "bytes": row.bytes,
        "duration_us": row.duration_us,
        "scalar_energy_uj": row.scalar_energy_uj,
        "domain_power_mw": dict(row.domain_power_mw),
        "status": row.status,
        "evidence_ids": list(row.evidence_ids),
    }


def _lease_to_json(row: LeaseRecord) -> dict[str, object]:
    return {
        "token": row.token,
        "lease_id": row.lease_id,
        "resource_id": row.resource_id,
        "lanes": list(row.lanes),
        "start_us": row.start_us,
        "predicted_end_us": row.predicted_end_us,
        "reserved_until_us": row.reserved_until_us,
    }


def _memory_to_json(row: MemoryPreview) -> dict[str, object]:
    return {
        "usage_before_bytes": dict(row.usage_before_bytes),
        "peak_usage_bytes": dict(row.peak_usage_bytes),
        "usage_after_bytes": dict(row.usage_after_bytes),
        "available_after_bytes": dict(row.available_after_bytes),
        "weight_required_bytes": dict(row.weight_required_bytes),
        "weight_growth_bytes": dict(row.weight_growth_bytes),
        "output_activation_bytes": dict(row.output_activation_bytes),
    }


def run_workload(
    profile: MatmulSystemProfile,
    programs: Sequence[ModelProgram],
    external_reservations: Sequence[tuple[str, str, int]] = (),
) -> Mapping[str, object]:
    scheduler = MatmulPlanner(profile)
    for device_id, allocation_id, allocation_bytes in external_reservations:
        scheduler.reserve_external_memory(
            device_id, allocation_id, allocation_bytes
        )
    for program in programs:
        scheduler.enqueue(program)
    return scheduler.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan a finite virtual queue of dense matmul operators"
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile = load_profile(args.profile)
    programs = load_workload(args.workload)
    result = run_workload(profile, programs)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return 0
    if args.output.exists():
        raise MatmulScheduleError("refusing to overwrite output")
    args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
