#!/usr/bin/env python3
"""Immutable READY route registry for the S15 host dispatch plane.

Route snapshots are immutable and replaced atomically. A lease pins the exact
route, residency, lease, and device-boot epochs of the snapshot it was minted
from; any later snapshot that changes one of those epochs, drains the route, or
whose generation the caller did not observe fails closed. This module performs
no device I/O and makes no latency, energy, or throughput claim.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_S14 = Path(__file__).resolve().parent.parent / "s14_mixed_streaming_scheduler"
if str(_S14) not in sys.path:
    sys.path.insert(0, str(_S14))

from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import BatchRuntimeError, RouteConfig  # noqa: E402


ROUTE_STATES = ("READY", "DRAINING", "UNAVAILABLE")
COMPOUND_KINDS = ("SINGLE", "SERIAL_CHAIN", "SHARED_TAIL_PARALLEL_HEAD")

# The selected A6000 for the matched gate plus the two measured phone serials.
ALLOWED_DEVICES = (
    "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f",
    "op15:3C15AU002CL00000",
    "op12:5ae7a43d",
)


class RouteRegistryError(ValueError):
    pass


def _int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RouteRegistryError(f"{name} must be an integer >= {minimum}")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RouteRegistryError(f"{name} must be a non-empty string")
    return value


def route_content_digest(config: RouteConfig) -> str:
    """Deterministic digest over every field a mutation could touch."""
    if type(config) is not RouteConfig:
        raise RouteRegistryError("config must be a RouteConfig")
    points = sorted(config.points, key=lambda point: point.batch_size)
    lines = [
        f"route_id={config.route_id}",
        f"service_class={config.service_class}",
        f"model_id={config.model_id}",
        f"island_id={config.island_id}",
        f"profile_id={config.profile_id}",
        f"route_epoch={config.route_epoch}",
        f"roofline_class={config.roofline_class}",
        f"high_priority_max={config.high_priority_max}",
    ]
    for point in points:
        if type(point) is not CertifiedBatchPoint:
            raise RouteRegistryError("route profile contains an uncertified batch point")
        lines.append(
            "point="
            f"{point.batch_size}:{point.duration_us}:"
            f"{point.correctness_certificate_id}:{point.placement_certificate_id}"
        )
    return "sha256:" + hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class RouteSnapshot:
    config: RouteConfig
    content_digest: str
    device_id: str
    layer_range: tuple[int, int] | None
    state: str
    residency_epoch: int
    lease_epoch: int
    device_boot_epoch: int
    thermal_ceiling_millic: int
    thermal_observed_millic: int
    execution_credits: int
    compound_kind: str
    correctness_certified: bool

    @property
    def route_id(self) -> str:
        return self.config.route_id

    @property
    def route_epoch(self) -> int:
        return self.config.route_epoch

    @property
    def profile_id(self) -> str:
        return self.config.profile_id

    @property
    def batch_envelope(self) -> tuple[int, int]:
        sizes = [point.batch_size for point in self.config.points]
        return (min(sizes), max(sizes))

    def validate(self) -> None:
        if type(self.config) is not RouteConfig:
            raise RouteRegistryError("snapshot config must be a RouteConfig")
        if self.content_digest != route_content_digest(self.config):
            raise RouteRegistryError("route content digest does not match the declared route")
        try:
            self.config.validate()
        except BatchRuntimeError as exc:
            raise RouteRegistryError(str(exc)) from exc
        if self.device_id not in ALLOWED_DEVICES:
            raise RouteRegistryError(f"unknown device {self.device_id!r}")
        if self.state not in ROUTE_STATES:
            raise RouteRegistryError(f"unknown route state {self.state!r}")
        if self.compound_kind not in COMPOUND_KINDS:
            raise RouteRegistryError(f"unknown compound kind {self.compound_kind!r}")
        if type(self.correctness_certified) is not bool:
            raise RouteRegistryError("correctness_certified must be bool")
        _int("route_epoch", self.route_epoch, 1)
        _int("residency_epoch", self.residency_epoch, 1)
        _int("lease_epoch", self.lease_epoch, 1)
        _int("device_boot_epoch", self.device_boot_epoch, 1)
        _int("thermal_ceiling_millic", self.thermal_ceiling_millic, 1)
        _int("thermal_observed_millic", self.thermal_observed_millic, 0)
        _int("execution_credits", self.execution_credits, 0)
        if self.thermal_observed_millic > self.thermal_ceiling_millic:
            raise RouteRegistryError("route exceeds its thermal envelope")
        if self.layer_range is not None:
            if type(self.layer_range) is not tuple or len(self.layer_range) != 2 \
                    or type(self.layer_range[0]) is not int or type(self.layer_range[1]) is not int \
                    or self.layer_range[0] < 0 or self.layer_range[1] <= self.layer_range[0]:
                raise RouteRegistryError("invalid layer range")
        if self.state == "READY":
            if self.compound_kind != "SINGLE":
                raise RouteRegistryError("a compound route cannot be READY")
            if not self.correctness_certified:
                raise RouteRegistryError("an uncertified route cannot be READY")

    def dispatchable(self) -> bool:
        return (
            self.state == "READY"
            and self.compound_kind == "SINGLE"
            and self.correctness_certified
            and self.execution_credits > 0
        )


@dataclass(frozen=True)
class RegistrySnapshot:
    generation: int
    routes: tuple[RouteSnapshot, ...]

    def route(self, route_id: str) -> RouteSnapshot | None:
        for snapshot in self.routes:
            if snapshot.route_id == route_id:
                return snapshot
        return None


@dataclass(frozen=True)
class Lease:
    lease_serial: int
    route_id: str
    profile_id: str
    device_id: str
    route_epoch: int
    residency_epoch: int
    lease_epoch: int
    device_boot_epoch: int
    generation: int
    granted_us: int


class ReadyRouteRegistry:
    """Atomic, immutable route table with epoch- and credit-fenced leasing."""

    def __init__(self) -> None:
        self._generation = 0
        self._routes: dict[str, RouteSnapshot] = {}
        self._outstanding: dict[str, int] = {}
        self._active: dict[int, Lease] = {}
        self._lease_serial = 0

    @property
    def generation(self) -> int:
        return self._generation

    def install(self, snapshots: Sequence[RouteSnapshot]) -> int:
        """Atomically replace the whole route table and bump the generation."""
        if type(snapshots) not in (list, tuple):
            raise RouteRegistryError("snapshots must be a list or tuple")
        validated: dict[str, RouteSnapshot] = {}
        for snapshot in snapshots:
            if type(snapshot) is not RouteSnapshot:
                raise RouteRegistryError("snapshots must contain RouteSnapshot values")
            snapshot.validate()
            if snapshot.route_id in validated:
                raise RouteRegistryError(f"duplicate route_id {snapshot.route_id!r}")
            validated[snapshot.route_id] = snapshot
        self._routes = validated
        self._generation += 1
        return self._generation

    def snapshot(self) -> RegistrySnapshot:
        ordered = tuple(self._routes[key] for key in sorted(self._routes))
        return RegistrySnapshot(self._generation, ordered)

    def get(self, route_id: str) -> RouteSnapshot | None:
        _text("route_id", route_id)
        return self._routes.get(route_id)

    def outstanding(self, route_id: str) -> int:
        _text("route_id", route_id)
        return self._outstanding.get(route_id, 0)

    def acquire_lease(
        self,
        route_id: str,
        now_us: int,
        expected_generation: int | None = None,
    ) -> Lease:
        _text("route_id", route_id)
        _int("now_us", now_us)
        if expected_generation is not None:
            _int("expected_generation", expected_generation, 0)
            if expected_generation != self._generation:
                raise RouteRegistryError("stale registry snapshot")
        snapshot = self._routes.get(route_id)
        if snapshot is None:
            raise RouteRegistryError(f"unknown route_id {route_id!r}")
        if not snapshot.dispatchable():
            raise RouteRegistryError(f"route {route_id!r} is not dispatchable")
        held = self._outstanding.get(route_id, 0)
        if held >= snapshot.execution_credits:
            raise RouteRegistryError(f"route {route_id!r} has no free execution credit")
        self._lease_serial += 1
        lease = Lease(
            lease_serial=self._lease_serial,
            route_id=route_id,
            profile_id=snapshot.profile_id,
            device_id=snapshot.device_id,
            route_epoch=snapshot.route_epoch,
            residency_epoch=snapshot.residency_epoch,
            lease_epoch=snapshot.lease_epoch,
            device_boot_epoch=snapshot.device_boot_epoch,
            generation=self._generation,
            granted_us=now_us,
        )
        self._outstanding[route_id] = held + 1
        self._active[lease.lease_serial] = lease
        return lease

    def validate_lease(self, lease: Lease) -> bool:
        if type(lease) is not Lease:
            raise RouteRegistryError("lease must be a Lease")
        if self._active.get(lease.lease_serial) is not lease:
            return False
        snapshot = self._routes.get(lease.route_id)
        if snapshot is None or snapshot.state != "READY":
            return False
        return (
            lease.generation == self._generation
            and snapshot.profile_id == lease.profile_id
            and snapshot.device_id == lease.device_id
            and snapshot.route_epoch == lease.route_epoch
            and snapshot.residency_epoch == lease.residency_epoch
            and snapshot.lease_epoch == lease.lease_epoch
            and snapshot.device_boot_epoch == lease.device_boot_epoch
        )

    def release_lease(self, lease: Lease) -> None:
        """Return the credit. A stale lease may still be released for accounting."""
        if type(lease) is not Lease:
            raise RouteRegistryError("lease must be a Lease")
        if lease.lease_serial not in self._active:
            raise RouteRegistryError("lease is not active")
        del self._active[lease.lease_serial]
        held = self._outstanding.get(lease.route_id, 0)
        if held <= 0:
            raise RouteRegistryError("lease accounting underflow")
        self._outstanding[lease.route_id] = held - 1
