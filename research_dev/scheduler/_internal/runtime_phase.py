"""Scheduler-owned leases for externally observed physical phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .policy import LeaseRecord


class RuntimePhaseError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimePhaseError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object) -> int:
    if type(value) is not int or value < 0:
        raise RuntimePhaseError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RuntimePhaseObservation:
    source_id: str
    phase_id: str
    owner_id: str | None
    phase_start_us: int
    occupied_resource_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("runtime phase source_id", self.source_id)
        _text("runtime phase phase_id", self.phase_id)
        if self.owner_id is not None:
            _text("runtime phase owner_id", self.owner_id)
        _integer("runtime phase phase_start_us", self.phase_start_us)
        resources = tuple(
            _text("runtime phase resource_id", value)
            for value in self.occupied_resource_ids
        )
        if len(resources) != len(set(resources)):
            raise RuntimePhaseError(
                "runtime phase resource ids must be unique"
            )
        object.__setattr__(self, "occupied_resource_ids", resources)


class RuntimePhaseLeaseController:
    """Translate raw phase ownership into one composite resource lease."""

    def __init__(
        self,
        source_id: str,
        available_resource_ids: Sequence[str],
        renewal_us: int,
    ) -> None:
        self.source_id = _text("runtime phase source_id", source_id)
        self._available = frozenset(
            _text("runtime phase available resource", value)
            for value in available_resource_ids
        )
        self.renewal_us = _integer(
            "runtime phase renewal_us", renewal_us
        )
        if self.renewal_us < 1:
            raise RuntimePhaseError("runtime phase renewal_us must be positive")
        self.phase_id: str | None = None
        self.owner_id: str | None = None
        self.resource_ids: tuple[str, ...] = ()
        self.leases: tuple[LeaseRecord, ...] = ()
        self.reserved_until_us: int | None = None
        self.generation = 0
        self.renewal_count = 0
        self.history: list[dict[str, object]] = []
        self._record: dict[str, object] | None = None

    def _close(
        self,
        end_us: int,
        *,
        extend: Callable[[str, Sequence[LeaseRecord], int, int], object],
        release: Callable[[str, int], None],
    ) -> None:
        if not self.leases:
            return
        assert self.reserved_until_us is not None
        if end_us > self.reserved_until_us:
            extend(
                "external:" + self.source_id,
                self.leases,
                end_us,
                end_us,
            )
            self.reserved_until_us = end_us
        for lease in self.leases:
            release(lease.token, max(lease.start_us, end_us))
        assert self._record is not None
        self._record["released_at_us"] = end_us
        self._record["status"] = "RELEASED"
        self.history.append(self._record)
        self.leases = ()
        self.reserved_until_us = None
        self._record = None
        self.resource_ids = ()

    def observe(
        self,
        observation: RuntimePhaseObservation,
        now_us: int,
        *,
        reserve: Callable[[Sequence[str], str, int, int], tuple[LeaseRecord, ...]],
        extend: Callable[[str, Sequence[LeaseRecord], int, int], object],
        release: Callable[[str, int], None],
    ) -> Mapping[str, object]:
        if not isinstance(observation, RuntimePhaseObservation):
            raise RuntimePhaseError("runtime phase observation is invalid")
        if observation.source_id != self.source_id:
            raise RuntimePhaseError("runtime phase source identity differs")
        now_us = _integer("runtime phase now_us", now_us)
        resources = tuple(
            resource_id
            for resource_id in observation.occupied_resource_ids
            if resource_id in self._available
        )
        changed = (
            observation.phase_id != self.phase_id
            or observation.owner_id != self.owner_id
            or resources != self.resource_ids
        )
        if changed:
            self._close(
                observation.phase_start_us,
                extend=extend,
                release=release,
            )
            self.phase_id = observation.phase_id
            self.owner_id = observation.owner_id
            self.resource_ids = resources
            self.renewal_count = 0
            if observation.owner_id is not None:
                if not resources:
                    raise RuntimePhaseError(
                        "active runtime phase has no registered resource"
                    )
                self.generation += 1
                finish_us = max(
                    observation.phase_start_us + 1,
                    now_us + self.renewal_us,
                )
                reservation_id = (
                    self.source_id
                    + ":"
                    + str(self.generation)
                    + ":"
                    + observation.phase_id
                )
                self.leases = reserve(
                    resources,
                    reservation_id,
                    observation.phase_start_us,
                    finish_us,
                )
                self.reserved_until_us = finish_us
                self._record = {
                    "lease_tokens": [lease.token for lease in self.leases],
                    "owner_id": observation.owner_id,
                    "phase_id": observation.phase_id,
                    "release_estimate_us": finish_us,
                    "renewal_count": 0,
                    "resource_ids": list(resources),
                    "started_at_us": observation.phase_start_us,
                    "status": "ACTIVE",
                }
        elif self.leases:
            assert self.reserved_until_us is not None
            finish_us = now_us + self.renewal_us
            if finish_us > self.reserved_until_us:
                extend(
                    "external:" + self.source_id,
                    self.leases,
                    now_us,
                    finish_us,
                )
                self.reserved_until_us = finish_us
                self.renewal_count += 1
                assert self._record is not None
                self._record["release_estimate_us"] = finish_us
                self._record["renewal_count"] = self.renewal_count
        return self.snapshot()

    def snapshot(self) -> dict[str, object]:
        return {
            "owner_id": self.owner_id,
            "phase_id": self.phase_id,
            "release_estimate_us": self.reserved_until_us,
            "resource_ids": list(self.resource_ids),
            "status": "AVAILABLE" if not self.leases else "LEASED",
        }

    def finish(
        self,
        at_us: int,
        *,
        extend: Callable[[str, Sequence[LeaseRecord], int, int], object],
        release: Callable[[str, int], None],
    ) -> None:
        self._close(
            _integer("runtime phase finish at_us", at_us),
            extend=extend,
            release=release,
        )
