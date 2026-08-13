#!/usr/bin/env python3
"""Energy-first task, layer, and operator placement compiler.

The compiler searches only profiled implementations supplied by the caller.
It does not create tensor cuts or certify a route. A selected plan still has
to pass RoutePolicy's measured route gates before runtime enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Any, Mapping, Sequence

from .policy import (
    QUALITY_RANK,
    EnergyDomain,
    KernelEnergyProfile,
    SchedulerError,
    estimate_kernel_energy,
)


PLACEMENT_PROFILE_SCHEMA = "s42-placement-hardware-profile-v1"
PLACEMENT_STATUSES = {"estimated", "measured"}
LOAD_STATUSES = {"not_applicable", "estimated", "measured"}

__all__ = [
    "LOAD_STATUSES",
    "PLACEMENT_PROFILE_SCHEMA",
    "PLACEMENT_STATUSES",
    "ComputeStep",
    "ExecutionBranch",
    "ExecutionStep",
    "HierarchicalPlacementPlanner",
    "LayerDecision",
    "MemoryPoolProfile",
    "OperatorCandidate",
    "OperatorDecision",
    "OperatorNode",
    "PlacementDevice",
    "PlacementError",
    "PlacementHardwareProfile",
    "PlacementPlan",
    "ProfiledEnergyDomain",
    "ProfiledKernel",
    "ResidentAllocation",
    "TaskPlacementDecision",
    "TaskRoute",
    "TransferDecision",
    "TransferLink",
    "TransferStep",
    "placement_plan_to_json",
    "task_placement_to_json",
]


class PlacementError(SchedulerError):
    pass


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise PlacementError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PlacementError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise PlacementError(f"{name} must be bool")
    return value


def _string(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PlacementError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PlacementError(f"{name} must be ASCII") from exc
    return value


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise PlacementError(f"{name} must be an object")
    return value


def _evidence(name: str, value: object) -> tuple[str, ...]:
    if type(value) is not list or not value:
        raise PlacementError(f"{name} must be a non-empty list")
    return tuple(_string(name, item) for item in value)


@dataclass(frozen=True)
class MemoryPoolProfile:
    pool_id: str
    capacity_bytes: int
    reserved_bytes: int

    @classmethod
    def from_json(cls, value: object) -> "MemoryPoolProfile":
        row = _object("memory pool", value)
        result = cls(
            pool_id=_string("memory pool id", row.get("pool_id")),
            capacity_bytes=_integer(
                "memory pool capacity_bytes", row.get("capacity_bytes"), 1
            ),
            reserved_bytes=_integer(
                "memory pool reserved_bytes", row.get("reserved_bytes", 0)
            ),
        )
        if result.reserved_bytes > result.capacity_bytes:
            raise PlacementError("memory pool reserve exceeds capacity")
        return result


@dataclass(frozen=True)
class PlacementDevice:
    device_id: str
    kind: str
    memory_pool_id: str
    allocation_limit_bytes: int
    ready: bool

    @classmethod
    def from_json(cls, value: object) -> "PlacementDevice":
        row = _object("placement device", value)
        return cls(
            device_id=_string("placement device id", row.get("device_id")),
            kind=_string("placement device kind", row.get("kind")),
            memory_pool_id=_string(
                "placement device memory_pool_id", row.get("memory_pool_id")
            ),
            allocation_limit_bytes=_integer(
                "placement device allocation_limit_bytes",
                row.get("allocation_limit_bytes", 0),
            ),
            ready=_boolean("placement device ready", row.get("ready")),
        )


@dataclass(frozen=True)
class ProfiledEnergyDomain:
    domain: EnergyDomain
    status: str
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "ProfiledEnergyDomain":
        row = _object("profiled energy domain", value)
        status = _string("profiled energy domain status", row.get("status"))
        if status not in PLACEMENT_STATUSES:
            raise PlacementError("unknown profiled energy domain status")
        return cls(
            domain=EnergyDomain.from_json(row),
            status=status,
            evidence_ids=_evidence(
                "profiled energy domain evidence_ids",
                row.get("evidence_ids"),
            ),
        )


@dataclass(frozen=True)
class ProfiledKernel:
    profile_id: str
    device_id: str
    status: str
    kernel: KernelEnergyProfile
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "ProfiledKernel":
        row = _object("profiled kernel", value)
        status = _string("profiled kernel status", row.get("status"))
        if status not in PLACEMENT_STATUSES:
            raise PlacementError("unknown profiled kernel status")
        return cls(
            profile_id=_string("profiled kernel id", row.get("profile_id")),
            device_id=_string("profiled kernel device_id", row.get("device_id")),
            status=status,
            kernel=KernelEnergyProfile.from_json(row),
            evidence_ids=_evidence(
                "profiled kernel evidence_ids", row.get("evidence_ids")
            ),
        )


@dataclass(frozen=True)
class TransferLink:
    link_id: str
    source_device: str
    target_device: str
    fixed_latency_us: int
    bandwidth_bytes_per_s: int
    fixed_dynamic_uj: int
    dynamic_pj_per_byte: int
    domain_active_power_mw: Mapping[str, int]
    status: str
    ready: bool
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_json(cls, value: object) -> "TransferLink":
        row = _object("transfer link", value)
        status = _string("transfer link status", row.get("status"))
        if status not in PLACEMENT_STATUSES:
            raise PlacementError("unknown transfer link status")
        raw_power = _object(
            "transfer link domain_active_power_mw",
            row.get("domain_active_power_mw", {}),
        )
        return cls(
            link_id=_string("transfer link id", row.get("link_id")),
            source_device=_string(
                "transfer link source_device", row.get("source_device")
            ),
            target_device=_string(
                "transfer link target_device", row.get("target_device")
            ),
            fixed_latency_us=_integer(
                "transfer link fixed_latency_us", row.get("fixed_latency_us")
            ),
            bandwidth_bytes_per_s=_integer(
                "transfer link bandwidth_bytes_per_s",
                row.get("bandwidth_bytes_per_s"),
                1,
            ),
            fixed_dynamic_uj=_integer(
                "transfer link fixed_dynamic_uj",
                row.get("fixed_dynamic_uj", 0),
            ),
            dynamic_pj_per_byte=_integer(
                "transfer link dynamic_pj_per_byte",
                row.get("dynamic_pj_per_byte", 0),
            ),
            domain_active_power_mw={
                _string("transfer energy domain", key): _integer(
                    f"transfer active power {key}", power, 1
                )
                for key, power in raw_power.items()
            },
            status=status,
            ready=_boolean("transfer link ready", row.get("ready")),
            evidence_ids=_evidence(
                "transfer link evidence_ids", row.get("evidence_ids")
            ),
        )


@dataclass(frozen=True)
class PlacementHardwareProfile:
    profile_id: str
    energy_boundary_id: str
    memory_pools: Mapping[str, MemoryPoolProfile]
    devices: Mapping[str, PlacementDevice]
    domains: Mapping[str, ProfiledEnergyDomain]
    idle_charge_domains: frozenset[str]
    kernels: Mapping[str, ProfiledKernel]
    links: tuple[TransferLink, ...]

    @classmethod
    def from_json(cls, value: object) -> "PlacementHardwareProfile":
        row = _object("placement hardware profile", value)
        if row.get("schema") != PLACEMENT_PROFILE_SCHEMA:
            raise PlacementError("placement hardware profile schema mismatch")

        raw_pools = row.get("memory_pools")
        if type(raw_pools) is not list or not raw_pools:
            raise PlacementError("memory_pools must be a non-empty list")
        pools: dict[str, MemoryPoolProfile] = {}
        for raw in raw_pools:
            pool = MemoryPoolProfile.from_json(raw)
            if pool.pool_id in pools:
                raise PlacementError("duplicate memory pool id")
            pools[pool.pool_id] = pool

        raw_devices = row.get("devices")
        if type(raw_devices) is not list or not raw_devices:
            raise PlacementError("devices must be a non-empty list")
        devices: dict[str, PlacementDevice] = {}
        for raw in raw_devices:
            device = PlacementDevice.from_json(raw)
            if device.device_id in devices:
                raise PlacementError("duplicate placement device id")
            if device.memory_pool_id not in pools:
                raise PlacementError("device references an unknown memory pool")
            devices[device.device_id] = device

        raw_domains = row.get("domains")
        if type(raw_domains) is not list or not raw_domains:
            raise PlacementError("domains must be a non-empty list")
        domains: dict[str, ProfiledEnergyDomain] = {}
        for raw in raw_domains:
            domain = ProfiledEnergyDomain.from_json(raw)
            if domain.domain.domain_id in domains:
                raise PlacementError("duplicate energy domain id")
            domains[domain.domain.domain_id] = domain
        raw_idle_charge = row.get("idle_charge_domains")
        if type(raw_idle_charge) is not list or not raw_idle_charge:
            raise PlacementError(
                "idle_charge_domains must be a non-empty list"
            )
        idle_charge_domains = frozenset(
            _string("idle charge domain", item) for item in raw_idle_charge
        )
        if idle_charge_domains - set(domains):
            raise PlacementError("idle charge references an unknown domain")

        raw_kernels = row.get("kernels")
        if type(raw_kernels) is not list or not raw_kernels:
            raise PlacementError("kernels must be a non-empty list")
        kernels: dict[str, ProfiledKernel] = {}
        for raw in raw_kernels:
            kernel = ProfiledKernel.from_json(raw)
            if kernel.profile_id in kernels:
                raise PlacementError("duplicate profiled kernel id")
            if kernel.device_id not in devices:
                raise PlacementError("kernel references an unknown device")
            domain = domains.get(kernel.kernel.domain_id)
            if domain is None:
                raise PlacementError("kernel references an unknown energy domain")
            if kernel.kernel.active_power_mw < domain.domain.idle_power_mw:
                raise PlacementError("kernel active power is below domain idle power")
            kernels[kernel.profile_id] = kernel

        raw_links = row.get("links")
        if type(raw_links) is not list:
            raise PlacementError("links must be a list")
        links: list[TransferLink] = []
        link_ids: set[str] = set()
        for raw in raw_links:
            link = TransferLink.from_json(raw)
            if link.link_id in link_ids:
                raise PlacementError("duplicate transfer link id")
            if (
                link.source_device not in devices
                or link.target_device not in devices
            ):
                raise PlacementError("link references an unknown device")
            for domain_id, active_power in link.domain_active_power_mw.items():
                domain = domains.get(domain_id)
                if domain is None:
                    raise PlacementError("link references an unknown energy domain")
                if active_power < domain.domain.idle_power_mw:
                    raise PlacementError(
                        "link active power is below domain idle power"
                    )
            links.append(link)
            link_ids.add(link.link_id)

        return cls(
            profile_id=_string("placement profile id", row.get("profile_id")),
            energy_boundary_id=_string(
                "placement energy_boundary_id", row.get("energy_boundary_id")
            ),
            memory_pools=pools,
            devices=devices,
            domains=domains,
            idle_charge_domains=idle_charge_domains,
            kernels=kernels,
            links=tuple(links),
        )


@dataclass(frozen=True)
class ComputeStep:
    step_id: str
    kernel_profile_id: str
    invocations: int
    compute_ops: int
    memory_bytes: int


@dataclass(frozen=True)
class TransferStep:
    step_id: str
    source_device: str
    target_device: str
    bytes: int


ExecutionStep = ComputeStep | TransferStep


@dataclass(frozen=True)
class ExecutionBranch:
    branch_id: str
    steps: tuple[ExecutionStep, ...]


@dataclass(frozen=True)
class ResidentAllocation:
    allocation_id: str
    device_id: str
    bytes: int


@dataclass(frozen=True)
class OperatorCandidate:
    candidate_id: str
    operator_id: str
    input_device: str
    output_device: str
    branches: tuple[ExecutionBranch, ...]
    tail_steps: tuple[ExecutionStep, ...] = ()
    resident_allocations: tuple[ResidentAllocation, ...] = ()
    workspace_bytes: Mapping[str, int] = field(default_factory=dict)
    quality_class: str = "exact"
    status: str = "estimated"
    placement_verified: bool = False
    evidence_ids: tuple[str, ...] = ()
    split_axis: str = "none"
    split_amount: int = 0
    split_total: int = 0


@dataclass(frozen=True)
class OperatorNode:
    operator_id: str
    layer_id: str
    input_bytes: int
    output_bytes: int
    candidates: tuple[OperatorCandidate, ...]


@dataclass(frozen=True)
class TransferDecision:
    step_id: str
    source_device: str
    target_device: str
    bytes: int
    link_ids: tuple[str, ...]
    latency_us: int
    dynamic_energy_uj: int


@dataclass(frozen=True)
class OperatorDecision:
    operator_id: str
    layer_id: str
    candidate_id: str
    input_device: str
    output_device: str
    compute_devices: tuple[str, ...]
    transition: TransferDecision | None
    internal_transfers: tuple[TransferDecision, ...]
    candidate_latency_us: int
    dynamic_energy_uj: int
    split_axis: str
    split_amount: int
    split_total: int


@dataclass(frozen=True)
class LayerDecision:
    layer_id: str
    candidate_ids: tuple[str, ...]
    devices: tuple[str, ...]
    latency_us: int
    dynamic_energy_uj: int


@dataclass(frozen=True)
class PlacementPlan:
    problem_id: str
    profile_id: str
    energy_boundary_id: str
    scope: str
    operator_decisions: tuple[OperatorDecision, ...]
    layer_decisions: tuple[LayerDecision, ...]
    final_transfer: TransferDecision | None
    latency_us: int
    dynamic_energy_uj: int
    idle_energy_uj: int
    total_energy_uj: int
    energy_by_domain_uj: Mapping[str, int]
    memory_by_pool_bytes: Mapping[str, int]
    memory_by_device_bytes: Mapping[str, int]
    resources: tuple[str, ...]
    deadline_met: bool
    measured: bool
    search_optimal: bool
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskRoute:
    route_id: str
    nodes: tuple[OperatorNode, ...]
    initial_device: str
    final_device: str
    ready: bool = True
    resident: bool = True
    quality_class: str = "exact"
    load_latency_us: int = 0
    load_energy_uj: int = 0
    load_status: str = "not_applicable"
    staged_allocation_bytes: int = 0
    staged_allocation_adoptable: bool = False
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlacementDecision:
    route_id: str
    placement: PlacementPlan
    latency_us: int
    energy_uj: int
    measured: bool
    evidence_ids: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _WorkCost:
    latency_us: int
    dynamic_uj: int
    dynamic_uj_by_domain: tuple[tuple[str, int], ...]
    unattributed_dynamic_uj: int
    active_us: tuple[tuple[str, int], ...]
    domains: frozenset[str]
    measured: bool
    evidence_ids: frozenset[str]
    transfers: tuple[TransferDecision, ...]
    compute_devices: frozenset[str]


def _active_map(cost: _WorkCost) -> dict[str, int]:
    return dict(cost.active_us)


def _dynamic_map(cost: _WorkCost) -> dict[str, int]:
    return dict(cost.dynamic_uj_by_domain)


def _make_cost(
    latency_us: int,
    dynamic_uj: int,
    active: Mapping[str, int],
    domains: frozenset[str],
    measured: bool,
    evidence_ids: frozenset[str],
    transfers: tuple[TransferDecision, ...],
    compute_devices: frozenset[str],
    dynamic_uj_by_domain: Mapping[str, int] | None = None,
    unattributed_dynamic_uj: int | None = None,
) -> _WorkCost:
    attributed = dict(dynamic_uj_by_domain or {})
    if unattributed_dynamic_uj is None:
        unattributed_dynamic_uj = dynamic_uj - sum(attributed.values())
    if (
        min(attributed.values(), default=0) < 0
        or unattributed_dynamic_uj < 0
        or sum(attributed.values()) + unattributed_dynamic_uj != dynamic_uj
    ):
        raise PlacementError("invalid dynamic-energy attribution")
    return _WorkCost(
        latency_us=latency_us,
        dynamic_uj=dynamic_uj,
        dynamic_uj_by_domain=tuple(sorted(attributed.items())),
        unattributed_dynamic_uj=unattributed_dynamic_uj,
        active_us=tuple(sorted((key, value) for key, value in active.items())),
        domains=domains,
        measured=measured,
        evidence_ids=evidence_ids,
        transfers=transfers,
        compute_devices=compute_devices,
    )


def _zero_cost() -> _WorkCost:
    return _make_cost(
        0, 0, {}, frozenset(), True, frozenset(), (), frozenset(), {}, 0
    )


def _serial_cost(left: _WorkCost, right: _WorkCost) -> _WorkCost:
    active = _active_map(left)
    for domain_id, value in right.active_us:
        active[domain_id] = active.get(domain_id, 0) + value
    dynamic = _dynamic_map(left)
    for domain_id, value in right.dynamic_uj_by_domain:
        dynamic[domain_id] = dynamic.get(domain_id, 0) + value
    return _make_cost(
        left.latency_us + right.latency_us,
        left.dynamic_uj + right.dynamic_uj,
        active,
        left.domains | right.domains,
        left.measured and right.measured,
        left.evidence_ids | right.evidence_ids,
        left.transfers + right.transfers,
        left.compute_devices | right.compute_devices,
        dynamic,
        left.unattributed_dynamic_uj + right.unattributed_dynamic_uj,
    )


def _cost_dominates(left: _WorkCost, right: _WorkCost) -> bool:
    if left.domains != right.domains or left.compute_devices != right.compute_devices:
        return False
    if left.measured != right.measured:
        return False
    left_active = _active_map(left)
    right_active = _active_map(right)
    if set(left_active) != set(right_active):
        return False
    no_worse = (
        left.latency_us <= right.latency_us
        and left.dynamic_uj <= right.dynamic_uj
        and all(left_active[key] <= right_active[key] for key in left_active)
    )
    strictly_better = (
        left.latency_us < right.latency_us
        or left.dynamic_uj < right.dynamic_uj
        or any(left_active[key] < right_active[key] for key in left_active)
    )
    return no_worse and strictly_better


def _prune_costs(costs: Sequence[_WorkCost]) -> list[_WorkCost]:
    unique: dict[tuple[object, ...], _WorkCost] = {}
    for cost in costs:
        key = (
            cost.latency_us,
            cost.dynamic_uj,
            cost.dynamic_uj_by_domain,
            cost.unattributed_dynamic_uj,
            cost.active_us,
            cost.domains,
            cost.measured,
            cost.transfers,
            cost.compute_devices,
        )
        unique[key] = cost
    rows = sorted(
        unique.values(),
        key=lambda item: (
            item.dynamic_uj,
            item.latency_us,
            tuple(
                (row.step_id, row.link_ids) for row in item.transfers
            ),
        ),
    )
    return [
        row for index, row in enumerate(rows)
        if not any(
            _cost_dominates(other, row)
            for other in rows[:index]
        )
    ]


class _TransferNetwork:
    def __init__(self, profile: PlacementHardwareProfile) -> None:
        self.profile = profile
        self._outgoing: dict[str, list[TransferLink]] = {}
        for link in profile.links:
            self._outgoing.setdefault(link.source_device, []).append(link)
        for links in self._outgoing.values():
            links.sort(key=lambda item: item.link_id)

    def _link_cost(
        self,
        step_id: str,
        link: TransferLink,
        transfer_bytes: int,
    ) -> _WorkCost:
        latency_us = link.fixed_latency_us + _ceil_div(
            transfer_bytes * 1_000_000,
            link.bandwidth_bytes_per_s,
        )
        dynamic_uj = link.fixed_dynamic_uj + _ceil_div(
            transfer_bytes * link.dynamic_pj_per_byte,
            1_000_000,
        )
        unattributed_dynamic_uj = dynamic_uj
        dynamic_by_domain: dict[str, int] = {}
        active: dict[str, int] = {}
        domain_evidence: set[str] = set()
        domains_measured = True
        for domain_id, active_power_mw in link.domain_active_power_mw.items():
            profiled_domain = self.profile.domains[domain_id]
            domain = profiled_domain.domain
            domains_measured = (
                domains_measured and profiled_domain.status == "measured"
            )
            domain_evidence.update(profiled_domain.evidence_ids)
            domain_dynamic_uj = _ceil_div(
                (active_power_mw - domain.idle_power_mw) * latency_us,
                1000,
            )
            dynamic_uj += domain_dynamic_uj
            dynamic_by_domain[domain_id] = domain_dynamic_uj
            active[domain_id] = latency_us
        transfer = TransferDecision(
            step_id=step_id,
            source_device=link.source_device,
            target_device=link.target_device,
            bytes=transfer_bytes,
            link_ids=(link.link_id,),
            latency_us=latency_us,
            dynamic_energy_uj=dynamic_uj,
        )
        return _make_cost(
            latency_us,
            dynamic_uj,
            active,
            frozenset(active),
            link.status == "measured" and domains_measured,
            frozenset(link.evidence_ids) | frozenset(domain_evidence),
            (transfer,),
            frozenset(),
            dynamic_by_domain,
            unattributed_dynamic_uj,
        )

    def choices(
        self,
        step_id: str,
        source_device: str,
        target_device: str,
        transfer_bytes: int,
        require_measured: bool,
    ) -> list[_WorkCost]:
        _integer("transfer bytes", transfer_bytes)
        if source_device not in self.profile.devices:
            raise PlacementError("transfer source device is unknown")
        if target_device not in self.profile.devices:
            raise PlacementError("transfer target device is unknown")
        if (
            not self.profile.devices[source_device].ready
            or not self.profile.devices[target_device].ready
        ):
            return []
        if source_device == target_device:
            return [_zero_cost()]

        paths: list[tuple[TransferLink, ...]] = []

        def visit(
            current: str,
            visited: frozenset[str],
            links: tuple[TransferLink, ...],
        ) -> None:
            if current == target_device:
                paths.append(links)
                return
            for link in self._outgoing.get(current, ()):
                if (
                    not link.ready
                    or not self.profile.devices[link.target_device].ready
                    or link.target_device in visited
                ):
                    continue
                if require_measured and link.status != "measured":
                    continue
                visit(
                    link.target_device,
                    visited | {link.target_device},
                    links + (link,),
                )

        visit(source_device, frozenset({source_device}), ())
        costs: list[_WorkCost] = []
        for path in paths:
            cost = _zero_cost()
            for index, link in enumerate(path):
                cost = _serial_cost(
                    cost,
                    self._link_cost(f"{step_id}:{index}", link, transfer_bytes),
                )
            if require_measured and not cost.measured:
                continue
            transfer = TransferDecision(
                step_id=step_id,
                source_device=source_device,
                target_device=target_device,
                bytes=transfer_bytes,
                link_ids=tuple(link.link_id for link in path),
                latency_us=cost.latency_us,
                dynamic_energy_uj=cost.dynamic_uj,
            )
            costs.append(_make_cost(
                cost.latency_us,
                cost.dynamic_uj,
                _active_map(cost),
                cost.domains,
                cost.measured,
                cost.evidence_ids,
                (transfer,),
                cost.compute_devices,
                _dynamic_map(cost),
                cost.unattributed_dynamic_uj,
            ))
        return _prune_costs(costs)


@dataclass(frozen=True)
class _CandidateCost:
    candidate: OperatorCandidate
    work: _WorkCost


@dataclass(frozen=True)
class _PartialPlan:
    location: str
    latency_us: int
    dynamic_uj: int
    dynamic_uj_by_domain: tuple[tuple[str, int], ...]
    unattributed_dynamic_uj: int
    active_us: tuple[tuple[str, int], ...]
    domains: frozenset[str]
    measured: bool
    evidence_ids: frozenset[str]
    allocations: frozenset[ResidentAllocation]
    workspace_by_pool: tuple[tuple[str, int], ...]
    workspace_by_device: tuple[tuple[str, int], ...]
    decisions: tuple[OperatorDecision, ...]


class HierarchicalPlacementPlanner:
    def __init__(
        self,
        profile: PlacementHardwareProfile,
        beam_width: int = 4096,
    ) -> None:
        self.profile = profile
        self.beam_width = _integer("beam_width", beam_width, 1)
        self.network = _TransferNetwork(profile)

    def _step_costs(
        self,
        step: ExecutionStep,
        require_measured: bool,
    ) -> list[_WorkCost]:
        if isinstance(step, TransferStep):
            _string("transfer step id", step.step_id)
            return self.network.choices(
                step.step_id,
                step.source_device,
                step.target_device,
                _integer("transfer step bytes", step.bytes),
                require_measured,
            )
        if not isinstance(step, ComputeStep):
            raise PlacementError("unknown execution step")
        _string("compute step id", step.step_id)
        kernel = self.profile.kernels.get(step.kernel_profile_id)
        if kernel is None:
            raise PlacementError("compute step references an unknown kernel")
        device = self.profile.devices[kernel.device_id]
        if not device.ready:
            return []
        if require_measured and kernel.status != "measured":
            return []
        profiled_domain = self.profile.domains[kernel.kernel.domain_id]
        if require_measured and profiled_domain.status != "measured":
            return []
        domain = profiled_domain.domain
        estimate = estimate_kernel_energy(
            domain,
            kernel.kernel,
            _integer("compute step invocations", step.invocations),
            _integer("compute step compute_ops", step.compute_ops),
            _integer("compute step memory_bytes", step.memory_bytes),
        )
        return [_make_cost(
            estimate.active_us,
            estimate.dynamic_uj,
            {domain.domain_id: estimate.active_us},
            frozenset({domain.domain_id}),
            (
                kernel.status == "measured"
                and profiled_domain.status == "measured"
            ),
            frozenset(kernel.evidence_ids) | frozenset(
                profiled_domain.evidence_ids
            ),
            (),
            frozenset({kernel.device_id}),
            {domain.domain_id: estimate.dynamic_uj},
            0,
        )]

    def _sequence_costs(
        self,
        steps: Sequence[ExecutionStep],
        require_measured: bool,
    ) -> list[_WorkCost]:
        costs = [_zero_cost()]
        for step in steps:
            options = self._step_costs(step, require_measured)
            costs = _prune_costs([
                _serial_cost(prefix, option)
                for prefix in costs
                for option in options
            ])
            if not costs:
                break
        return costs

    def _candidate_costs(
        self,
        candidate: OperatorCandidate,
        required_quality: str,
        require_measured: bool,
    ) -> list[_CandidateCost]:
        _string("candidate id", candidate.candidate_id)
        _string("candidate operator id", candidate.operator_id)
        _string("candidate split axis", candidate.split_axis)
        _integer("candidate split amount", candidate.split_amount)
        _integer("candidate split total", candidate.split_total)
        if candidate.split_axis == "none":
            if candidate.split_amount or candidate.split_total:
                raise PlacementError("unsplit candidate carries split dimensions")
        elif (
            candidate.split_amount == 0
            or candidate.split_total == 0
            or candidate.split_amount > candidate.split_total
        ):
            raise PlacementError("candidate split dimensions are invalid")
        if candidate.quality_class not in QUALITY_RANK:
            raise PlacementError("unknown candidate quality class")
        if QUALITY_RANK[candidate.quality_class] < QUALITY_RANK[required_quality]:
            return []
        if not candidate.placement_verified:
            return []
        if candidate.status not in PLACEMENT_STATUSES:
            raise PlacementError("unknown candidate status")
        if require_measured and candidate.status != "measured":
            return []
        if not candidate.branches:
            raise PlacementError("operator candidate has no execution branch")
        branch_ids = [
            _string("candidate branch id", branch.branch_id)
            for branch in candidate.branches
        ]
        if len(set(branch_ids)) != len(branch_ids):
            raise PlacementError("duplicate candidate branch id")
        if not candidate.evidence_ids:
            raise PlacementError("operator candidate has no evidence id")
        for device_id in (candidate.input_device, candidate.output_device):
            device = self.profile.devices.get(device_id)
            if device is None:
                raise PlacementError("candidate references an unknown device")
            if not device.ready:
                return []

        branch_options = [
            self._sequence_costs(branch.steps, require_measured)
            for branch in candidate.branches
        ]
        if any(not options for options in branch_options):
            return []
        tail_options = self._sequence_costs(
            candidate.tail_steps, require_measured
        )
        if not tail_options:
            return []

        results: list[_CandidateCost] = []
        for branches in itertools.product(*branch_options):
            domain_owners: dict[str, str] = {}
            conflict = False
            for index, branch in enumerate(branches):
                for domain_id in branch.domains:
                    owner = domain_owners.setdefault(domain_id, str(index))
                    if owner != str(index):
                        conflict = True
            if conflict:
                continue
            active: dict[str, int] = {}
            dynamic_by_domain: dict[str, int] = {}
            for branch in branches:
                for domain_id, value in branch.active_us:
                    active[domain_id] = active.get(domain_id, 0) + value
                for domain_id, value in branch.dynamic_uj_by_domain:
                    dynamic_by_domain[domain_id] = (
                        dynamic_by_domain.get(domain_id, 0) + value
                    )
            parallel = _make_cost(
                max(branch.latency_us for branch in branches),
                sum(branch.dynamic_uj for branch in branches),
                active,
                frozenset().union(*(branch.domains for branch in branches)),
                all(branch.measured for branch in branches),
                frozenset().union(
                    *(branch.evidence_ids for branch in branches)
                ) | frozenset(candidate.evidence_ids),
                tuple(
                    transfer
                    for branch in branches
                    for transfer in branch.transfers
                ),
                frozenset().union(
                    *(branch.compute_devices for branch in branches)
                ),
                dynamic_by_domain,
                sum(branch.unattributed_dynamic_uj for branch in branches),
            )
            for tail in tail_options:
                work = _serial_cost(parallel, tail)
                work = _make_cost(
                    work.latency_us,
                    work.dynamic_uj,
                    _active_map(work),
                    work.domains,
                    work.measured and candidate.status == "measured",
                    work.evidence_ids,
                    work.transfers,
                    work.compute_devices,
                    _dynamic_map(work),
                    work.unattributed_dynamic_uj,
                )
                results.append(_CandidateCost(candidate, work))
        pruned = _prune_costs([row.work for row in results])
        allowed = set(pruned)
        return [row for row in results if row.work in allowed]

    def _memory_state(
        self,
        allocations: frozenset[ResidentAllocation],
        workspace_by_pool: Mapping[str, int],
        workspace_by_device: Mapping[str, int],
    ) -> tuple[dict[str, int], dict[str, int], bool]:
        resident: dict[tuple[str, str], int] = {}
        resident_by_device: dict[tuple[str, str], int] = {}
        for allocation in allocations:
            _integer("resident allocation bytes", allocation.bytes, 1)
            device = self.profile.devices.get(allocation.device_id)
            if device is None:
                raise PlacementError("allocation references an unknown device")
            key = (allocation.allocation_id, device.memory_pool_id)
            previous = resident.get(key)
            if previous is not None and previous != allocation.bytes:
                raise PlacementError("shared allocation has inconsistent sizes")
            resident[key] = allocation.bytes
            resident_by_device[
                (allocation.allocation_id, allocation.device_id)
            ] = allocation.bytes
        totals = {
            pool_id: pool.reserved_bytes
            for pool_id, pool in self.profile.memory_pools.items()
        }
        for (_, pool_id), amount in resident.items():
            totals[pool_id] += amount
        for pool_id, amount in workspace_by_pool.items():
            if pool_id not in totals:
                raise PlacementError("workspace references an unknown memory pool")
            totals[pool_id] += amount
        device_totals = {device_id: 0 for device_id in self.profile.devices}
        for (_, device_id), amount in resident_by_device.items():
            device_totals[device_id] += amount
        for device_id, amount in workspace_by_device.items():
            if device_id not in device_totals:
                raise PlacementError("workspace references an unknown device")
            device_totals[device_id] += amount
        pools_fit = all(
            totals[pool_id] <= pool.capacity_bytes
            for pool_id, pool in self.profile.memory_pools.items()
        )
        devices_fit = all(
            device.allocation_limit_bytes == 0
            or device_totals[device_id] <= device.allocation_limit_bytes
            for device_id, device in self.profile.devices.items()
        )
        return totals, device_totals, pools_fit and devices_fit

    def _add_candidate_memory(
        self,
        partial: _PartialPlan,
        candidate: OperatorCandidate,
    ) -> tuple[
        frozenset[ResidentAllocation],
        tuple[tuple[str, int], ...],
        tuple[tuple[str, int], ...],
    ] | None:
        allocations = partial.allocations | frozenset(candidate.resident_allocations)
        workspace_by_pool = dict(partial.workspace_by_pool)
        workspace_by_device = dict(partial.workspace_by_device)
        current_by_pool: dict[str, int] = {}
        for device_id, amount in candidate.workspace_bytes.items():
            _integer("candidate workspace bytes", amount)
            device = self.profile.devices.get(device_id)
            if device is None:
                raise PlacementError("workspace references an unknown device")
            pool_id = device.memory_pool_id
            current_by_pool[pool_id] = current_by_pool.get(pool_id, 0) + amount
            workspace_by_device[device_id] = max(
                workspace_by_device.get(device_id, 0), amount
            )
        for pool_id, amount in current_by_pool.items():
            workspace_by_pool[pool_id] = max(
                workspace_by_pool.get(pool_id, 0), amount
            )
        _, _, fits = self._memory_state(
            allocations,
            workspace_by_pool,
            workspace_by_device,
        )
        if not fits:
            return None
        return (
            allocations,
            tuple(sorted(workspace_by_pool.items())),
            tuple(sorted(workspace_by_device.items())),
        )

    def _partial_energy(self, partial: _PartialPlan) -> int:
        return partial.dynamic_uj + sum(
            _ceil_div(
                self.profile.domains[domain_id].domain.idle_power_mw
                * partial.latency_us,
                1000,
            )
            for domain_id in partial.domains
        )

    def _prune_partials(
        self,
        partials: Sequence[_PartialPlan],
    ) -> tuple[list[_PartialPlan], bool]:
        groups: dict[tuple[object, ...], list[_PartialPlan]] = {}
        for partial in partials:
            key = (
                partial.location,
                partial.domains,
                partial.allocations,
                partial.workspace_by_pool,
                partial.workspace_by_device,
                partial.measured,
            )
            groups.setdefault(key, []).append(partial)
        kept: list[_PartialPlan] = []
        for rows in groups.values():
            rows.sort(key=lambda item: (
                self._partial_energy(item),
                item.latency_us,
                tuple(row.candidate_id for row in item.decisions),
            ))
            for row in rows:
                row_active = dict(row.active_us)
                dominated = False
                for other in kept:
                    if (
                        other.location != row.location
                        or other.domains != row.domains
                        or other.allocations != row.allocations
                        or other.workspace_by_pool != row.workspace_by_pool
                        or other.workspace_by_device != row.workspace_by_device
                        or other.measured != row.measured
                    ):
                        continue
                    other_active = dict(other.active_us)
                    if (
                        other.latency_us <= row.latency_us
                        and other.dynamic_uj <= row.dynamic_uj
                        and all(
                            other_active.get(key, 0)
                            <= row_active.get(key, 0)
                            for key in set(other_active) | set(row_active)
                        )
                    ):
                        dominated = True
                        break
                if not dominated:
                    kept.append(row)
        kept.sort(key=lambda item: (
            self._partial_energy(item), item.latency_us, item.location
        ))
        truncated = len(kept) > self.beam_width
        return kept[:self.beam_width], truncated

    def _validate_nodes(self, nodes: Sequence[OperatorNode]) -> None:
        if not nodes:
            raise PlacementError("placement problem has no operator nodes")
        seen: set[str] = set()
        for node in nodes:
            _string("operator id", node.operator_id)
            _string("layer id", node.layer_id)
            _integer("operator input bytes", node.input_bytes)
            _integer("operator output bytes", node.output_bytes)
            if node.operator_id in seen:
                raise PlacementError("duplicate operator id")
            if not node.candidates:
                raise PlacementError("operator node has no candidates")
            if any(
                candidate.operator_id != node.operator_id
                for candidate in node.candidates
            ):
                raise PlacementError("candidate and operator ids do not match")
            seen.add(node.operator_id)

    def plan_sequence(
        self,
        problem_id: str,
        nodes: Sequence[OperatorNode],
        initial_device: str,
        final_device: str,
        deadline_us: int,
        required_quality: str = "exact",
        require_measured: bool = True,
        initial_allocations: Sequence[ResidentAllocation] = (),
    ) -> PlacementPlan:
        _string("problem id", problem_id)
        self._validate_nodes(nodes)
        _integer("placement deadline_us", deadline_us, 1)
        if required_quality not in QUALITY_RANK:
            raise PlacementError("unknown required quality")
        for device_id in (initial_device, final_device):
            device = self.profile.devices.get(device_id)
            if device is None:
                raise PlacementError("placement endpoint device is unknown")
            if not device.ready:
                raise PlacementError("placement endpoint device is not ready")
        idle_profiles = [
            self.profile.domains[domain_id]
            for domain_id in self.profile.idle_charge_domains
        ]
        if require_measured and any(
            domain.status != "measured" for domain in idle_profiles
        ):
            raise PlacementError("idle energy domain is not measured")

        allocations = frozenset(initial_allocations)
        _, _, fits = self._memory_state(allocations, {}, {})
        if not fits:
            raise PlacementError("initial resident memory exceeds capacity")
        partials = [_PartialPlan(
            location=initial_device,
            latency_us=0,
            dynamic_uj=0,
            dynamic_uj_by_domain=(),
            unattributed_dynamic_uj=0,
            active_us=(),
            domains=self.profile.idle_charge_domains,
            measured=all(
                domain.status == "measured" for domain in idle_profiles
            ),
            evidence_ids=frozenset(
                evidence_id
                for domain in idle_profiles
                for evidence_id in domain.evidence_ids
            ),
            allocations=allocations,
            workspace_by_pool=(),
            workspace_by_device=(),
            decisions=(),
        )]
        rejection_counts = {
            "candidate_gate": 0,
            "memory": 0,
            "transfer": 0,
            "deadline": 0,
        }
        frontier_truncated = False

        for node in nodes:
            next_partials: list[_PartialPlan] = []
            for partial in partials:
                for candidate in node.candidates:
                    candidate_costs = self._candidate_costs(
                        candidate, required_quality, require_measured
                    )
                    if not candidate_costs:
                        rejection_counts["candidate_gate"] += 1
                        continue
                    transitions = self.network.choices(
                        f"{node.operator_id}:input",
                        partial.location,
                        candidate.input_device,
                        node.input_bytes,
                        require_measured,
                    )
                    if not transitions:
                        rejection_counts["transfer"] += 1
                        continue
                    memory = self._add_candidate_memory(partial, candidate)
                    if memory is None:
                        rejection_counts["memory"] += 1
                        continue
                    (
                        candidate_allocations,
                        workspace_by_pool,
                        workspace_by_device,
                    ) = memory
                    for transition in transitions:
                        for row in candidate_costs:
                            total = _serial_cost(transition, row.work)
                            latency_us = partial.latency_us + total.latency_us
                            if latency_us > deadline_us:
                                rejection_counts["deadline"] += 1
                                continue
                            active = dict(partial.active_us)
                            for domain_id, amount in total.active_us:
                                active[domain_id] = active.get(domain_id, 0) + amount
                            dynamic_by_domain = dict(partial.dynamic_uj_by_domain)
                            for domain_id, amount in total.dynamic_uj_by_domain:
                                dynamic_by_domain[domain_id] = (
                                    dynamic_by_domain.get(domain_id, 0) + amount
                                )
                            domains = partial.domains | total.domains
                            if any(active.get(domain_id, 0) > latency_us for domain_id in domains):
                                rejection_counts["candidate_gate"] += 1
                                continue
                            transition_decision = (
                                transition.transfers[0]
                                if transition.transfers
                                else None
                            )
                            decision = OperatorDecision(
                                operator_id=node.operator_id,
                                layer_id=node.layer_id,
                                candidate_id=candidate.candidate_id,
                                input_device=candidate.input_device,
                                output_device=candidate.output_device,
                                compute_devices=tuple(sorted(row.work.compute_devices)),
                                transition=transition_decision,
                                internal_transfers=row.work.transfers,
                                candidate_latency_us=row.work.latency_us,
                                dynamic_energy_uj=total.dynamic_uj,
                                split_axis=candidate.split_axis,
                                split_amount=candidate.split_amount,
                                split_total=candidate.split_total,
                            )
                            next_partials.append(_PartialPlan(
                                location=candidate.output_device,
                                latency_us=latency_us,
                                dynamic_uj=partial.dynamic_uj + total.dynamic_uj,
                                dynamic_uj_by_domain=tuple(
                                    sorted(dynamic_by_domain.items())
                                ),
                                unattributed_dynamic_uj=(
                                    partial.unattributed_dynamic_uj
                                    + total.unattributed_dynamic_uj
                                ),
                                active_us=tuple(sorted(active.items())),
                                domains=domains,
                                measured=(
                                    partial.measured
                                    and total.measured
                                    and candidate.status == "measured"
                                ),
                                evidence_ids=(
                                    partial.evidence_ids | total.evidence_ids
                                ),
                                allocations=candidate_allocations,
                                workspace_by_pool=workspace_by_pool,
                                workspace_by_device=workspace_by_device,
                                decisions=partial.decisions + (decision,),
                            ))
            partials, truncated = self._prune_partials(next_partials)
            frontier_truncated = frontier_truncated or truncated
            if not partials:
                detail = ", ".join(
                    f"{key}={value}" for key, value in rejection_counts.items()
                )
                raise PlacementError(
                    f"no feasible placement at operator {node.operator_id}: {detail}"
                )

        final_rows: list[tuple[_PartialPlan, _WorkCost]] = []
        final_bytes = nodes[-1].output_bytes
        for partial in partials:
            for transfer in self.network.choices(
                f"{problem_id}:output",
                partial.location,
                final_device,
                final_bytes,
                require_measured,
            ):
                latency_us = partial.latency_us + transfer.latency_us
                if latency_us > deadline_us:
                    continue
                active = dict(partial.active_us)
                for domain_id, amount in transfer.active_us:
                    active[domain_id] = active.get(domain_id, 0) + amount
                dynamic_by_domain = dict(partial.dynamic_uj_by_domain)
                for domain_id, amount in transfer.dynamic_uj_by_domain:
                    dynamic_by_domain[domain_id] = (
                        dynamic_by_domain.get(domain_id, 0) + amount
                    )
                domains = partial.domains | transfer.domains
                if any(active.get(domain_id, 0) > latency_us for domain_id in domains):
                    continue
                final_rows.append((_PartialPlan(
                    location=final_device,
                    latency_us=latency_us,
                    dynamic_uj=partial.dynamic_uj + transfer.dynamic_uj,
                    dynamic_uj_by_domain=tuple(sorted(dynamic_by_domain.items())),
                    unattributed_dynamic_uj=(
                        partial.unattributed_dynamic_uj
                        + transfer.unattributed_dynamic_uj
                    ),
                    active_us=tuple(sorted(active.items())),
                    domains=domains,
                    measured=partial.measured and transfer.measured,
                    evidence_ids=partial.evidence_ids | transfer.evidence_ids,
                    allocations=partial.allocations,
                    workspace_by_pool=partial.workspace_by_pool,
                    workspace_by_device=partial.workspace_by_device,
                    decisions=partial.decisions,
                ), transfer))
        if not final_rows:
            raise PlacementError("no placement meets the final transfer and deadline")

        def final_energy(row: _PartialPlan) -> int:
            return row.dynamic_uj + sum(
                _ceil_div(
                    self.profile.domains[domain_id].domain.idle_power_mw
                    * row.latency_us,
                    1000,
                )
                for domain_id in row.domains
            )

        selected, final_cost = min(
            final_rows,
            key=lambda item: (
                final_energy(item[0]),
                item[0].latency_us,
                tuple(decision.candidate_id for decision in item[0].decisions),
            ),
        )
        idle_by_domain = {
            domain_id: _ceil_div(
                self.profile.domains[domain_id].domain.idle_power_mw
                * selected.latency_us,
                1000,
            )
            for domain_id in selected.domains
        }
        energy_by_domain = {
            domain_id: (
                idle_by_domain[domain_id]
                + dict(selected.dynamic_uj_by_domain).get(domain_id, 0)
            )
            for domain_id in selected.domains
        }
        if selected.unattributed_dynamic_uj:
            energy_by_domain["cross-domain-transfer"] = (
                selected.unattributed_dynamic_uj
            )

        memory_by_pool, memory_by_device, _ = self._memory_state(
            selected.allocations,
            dict(selected.workspace_by_pool),
            dict(selected.workspace_by_device),
        )
        decisions_by_layer: dict[str, list[OperatorDecision]] = {}
        for decision in selected.decisions:
            decisions_by_layer.setdefault(decision.layer_id, []).append(decision)
        layers: list[LayerDecision] = []
        for layer_id, layer_rows in decisions_by_layer.items():
            rows = tuple(layer_rows)
            layers.append(LayerDecision(
                layer_id=layer_id,
                candidate_ids=tuple(row.candidate_id for row in rows),
                devices=tuple(sorted({
                    device
                    for row in rows
                    for device in row.compute_devices
                })),
                latency_us=sum(
                    row.candidate_latency_us
                    + (row.transition.latency_us if row.transition else 0)
                    for row in rows
                ),
                dynamic_energy_uj=sum(row.dynamic_energy_uj for row in rows),
            ))
        if any(len(layer.devices) > 1 for layer in layers) or any(
            len(row.compute_devices) > 1 for row in selected.decisions
        ):
            scope = "operator"
        elif len({layer.devices for layer in layers}) > 1:
            scope = "layer"
        else:
            scope = "task"
        final_transfer = final_cost.transfers[0] if final_cost.transfers else None
        resources = {
            f"device:{device_id}"
            for row in selected.decisions
            for device_id in row.compute_devices
        }
        for row in selected.decisions:
            transfers = row.internal_transfers
            if row.transition is not None:
                transfers = (row.transition,) + transfers
            for transfer in transfers:
                resources.update(
                    f"link:{link_id}" for link_id in transfer.link_ids
                )
        if final_transfer is not None:
            resources.update(
                f"link:{link_id}" for link_id in final_transfer.link_ids
            )
        idle_energy = sum(idle_by_domain.values())
        return PlacementPlan(
            problem_id=problem_id,
            profile_id=self.profile.profile_id,
            energy_boundary_id=self.profile.energy_boundary_id,
            scope=scope,
            operator_decisions=selected.decisions,
            layer_decisions=tuple(layers),
            final_transfer=final_transfer,
            latency_us=selected.latency_us,
            dynamic_energy_uj=selected.dynamic_uj,
            idle_energy_uj=idle_energy,
            total_energy_uj=selected.dynamic_uj + idle_energy,
            energy_by_domain_uj=energy_by_domain,
            memory_by_pool_bytes=memory_by_pool,
            memory_by_device_bytes=memory_by_device,
            resources=tuple(sorted(resources)),
            deadline_met=selected.latency_us <= deadline_us,
            measured=selected.measured,
            search_optimal=not frontier_truncated,
            evidence_ids=tuple(sorted(selected.evidence_ids)),
        )

    def plan_task(
        self,
        task_id: str,
        routes: Sequence[TaskRoute],
        deadline_us: int,
        required_quality: str = "exact",
        require_measured: bool = True,
    ) -> TaskPlacementDecision:
        _string("task id", task_id)
        _integer("task deadline_us", deadline_us, 1)
        if not routes:
            raise PlacementError("task has no route alternatives")
        choices: list[
            tuple[int, int, str, PlacementPlan, bool, tuple[str, ...]]
        ] = []
        rejected: list[tuple[str, str]] = []
        route_ids: set[str] = set()
        for route in routes:
            _string("task route id", route.route_id)
            if route.route_id in route_ids:
                raise PlacementError("duplicate task route id")
            route_ids.add(route.route_id)
            _integer("task route load_latency_us", route.load_latency_us)
            _integer("task route load_energy_uj", route.load_energy_uj)
            _integer(
                "task route staged_allocation_bytes",
                route.staged_allocation_bytes,
            )
            if not route.ready:
                rejected.append((route.route_id, "ROUTE_NOT_READY"))
                continue
            if (
                route.staged_allocation_bytes
                and not route.staged_allocation_adoptable
            ):
                rejected.append((
                    route.route_id,
                    "STAGED_ALLOCATION_NOT_ADOPTABLE",
                ))
                continue
            if (
                route.staged_allocation_adoptable
                and route.staged_allocation_bytes == 0
            ):
                raise PlacementError(
                    "adoptable task staging must declare allocation bytes"
                )
            if not route.resident and route.load_latency_us == 0:
                rejected.append((route.route_id, "WEIGHTS_NOT_RESIDENT"))
                continue
            if route.quality_class not in QUALITY_RANK:
                raise PlacementError("unknown task route quality class")
            if QUALITY_RANK[route.quality_class] < QUALITY_RANK[required_quality]:
                rejected.append((route.route_id, "QUALITY_INSUFFICIENT"))
                continue
            if route.load_status not in LOAD_STATUSES:
                raise PlacementError("unknown task route load status")
            if route.load_status == "not_applicable" and (
                route.load_latency_us or route.load_energy_uj
            ):
                raise PlacementError(
                    "not-applicable task load must have zero latency and energy"
                )
            if route.load_status != "not_applicable" and not route.evidence_ids:
                raise PlacementError("profiled task load has no evidence id")
            if require_measured and route.load_status == "estimated":
                rejected.append((route.route_id, "LOAD_ENERGY_NOT_MEASURED"))
                continue
            remaining = deadline_us - route.load_latency_us
            if remaining <= 0:
                rejected.append((route.route_id, "LOAD_EXCEEDS_DEADLINE"))
                continue
            try:
                placement = self.plan_sequence(
                    f"{task_id}:{route.route_id}",
                    route.nodes,
                    route.initial_device,
                    route.final_device,
                    remaining,
                    required_quality,
                    require_measured,
                )
            except PlacementError as exc:
                rejected.append((route.route_id, str(exc)))
                continue
            latency_us = route.load_latency_us + placement.latency_us
            energy_uj = route.load_energy_uj + placement.total_energy_uj
            measured = (
                placement.measured
                and route.load_status in {"not_applicable", "measured"}
            )
            choices.append((
                energy_uj,
                latency_us,
                route.route_id,
                placement,
                measured,
                route.evidence_ids,
            ))
        if not choices:
            raise PlacementError(
                "no feasible task route: "
                + "; ".join(f"{route}={reason}" for route, reason in rejected)
            )
        (
            energy_uj,
            latency_us,
            route_id,
            placement,
            measured,
            route_evidence_ids,
        ) = min(choices, key=lambda item: item[:3])
        return TaskPlacementDecision(
            route_id=route_id,
            placement=placement,
            latency_us=latency_us,
            energy_uj=energy_uj,
            measured=measured,
            evidence_ids=tuple(sorted(
                set(placement.evidence_ids) | set(route_evidence_ids)
            )),
            rejected=tuple(sorted(rejected)),
        )


def placement_plan_to_json(plan: PlacementPlan) -> dict[str, object]:
    return {
        "problem_id": plan.problem_id,
        "profile_id": plan.profile_id,
        "energy_boundary_id": plan.energy_boundary_id,
        "scope": plan.scope,
        "latency_us": plan.latency_us,
        "dynamic_energy_uj": plan.dynamic_energy_uj,
        "idle_energy_uj": plan.idle_energy_uj,
        "total_energy_uj": plan.total_energy_uj,
        "energy_by_domain_uj": dict(plan.energy_by_domain_uj),
        "memory_by_pool_bytes": dict(plan.memory_by_pool_bytes),
        "memory_by_device_bytes": dict(plan.memory_by_device_bytes),
        "resources": list(plan.resources),
        "deadline_met": plan.deadline_met,
        "measured": plan.measured,
        "search_optimal": plan.search_optimal,
        "evidence_ids": list(plan.evidence_ids),
        "operators": [
            {
                "operator_id": row.operator_id,
                "layer_id": row.layer_id,
                "candidate_id": row.candidate_id,
                "input_device": row.input_device,
                "output_device": row.output_device,
                "compute_devices": list(row.compute_devices),
                "candidate_latency_us": row.candidate_latency_us,
                "dynamic_energy_uj": row.dynamic_energy_uj,
                "split_axis": row.split_axis,
                "split_amount": row.split_amount,
                "split_total": row.split_total,
                "transition": (
                    None
                    if row.transition is None
                    else {
                        "bytes": row.transition.bytes,
                        "dynamic_energy_uj": row.transition.dynamic_energy_uj,
                        "latency_us": row.transition.latency_us,
                        "link_ids": list(row.transition.link_ids),
                        "source_device": row.transition.source_device,
                        "target_device": row.transition.target_device,
                    }
                ),
                "internal_transfers": [
                    {
                        "bytes": transfer.bytes,
                        "dynamic_energy_uj": transfer.dynamic_energy_uj,
                        "latency_us": transfer.latency_us,
                        "link_ids": list(transfer.link_ids),
                        "source_device": transfer.source_device,
                        "target_device": transfer.target_device,
                    }
                    for transfer in row.internal_transfers
                ],
            }
            for row in plan.operator_decisions
        ],
        "layers": [
            {
                "layer_id": row.layer_id,
                "candidate_ids": list(row.candidate_ids),
                "devices": list(row.devices),
                "latency_us": row.latency_us,
                "dynamic_energy_uj": row.dynamic_energy_uj,
            }
            for row in plan.layer_decisions
        ],
    }


def task_placement_to_json(
    decision: TaskPlacementDecision,
) -> dict[str, object]:
    return {
        "route_id": decision.route_id,
        "latency_us": decision.latency_us,
        "energy_uj": decision.energy_uj,
        "measured": decision.measured,
        "evidence_ids": list(decision.evidence_ids),
        "rejected": [
            {"route_id": route_id, "reason": reason}
            for route_id, reason in decision.rejected
        ],
        "placement": placement_plan_to_json(decision.placement),
    }
