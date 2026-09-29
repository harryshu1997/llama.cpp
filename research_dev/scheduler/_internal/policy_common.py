"""Shared policy primitives: errors, strict coercions, profile constants, resource profile and lease demand."""

from __future__ import annotations

from typing import Any, Mapping

from dataclasses import dataclass


def _mapping(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise SchedulerError(f"{name} must be an object")
    return value



PROFILE_SCHEMA = "s42-general-scheduler-profile-v1"


QUALITY_RANK = {
    "unverified": 0,
    "approximate": 1,
    "semantic": 1,
    "bounded_numeric": 2,
    "exact": 3,
}


POLICY_MODES = {"control", "enforce", "shadow", "capacity", "adaptive"}


GRANULARITIES = {"task", "layer", "operator"}


ENERGY_STATUSES = {"unknown", "estimated", "measured"}


OVERLAP_STATUSES = {"unknown", "diagnostic", "measured", "not_applicable"}


class SchedulerError(ValueError):
    pass


def _strict_int(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise SchedulerError(f"{name} must be an integer >= {minimum}")
    return value


def _strict_bool(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise SchedulerError(f"{name} must be bool")
    return value


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise SchedulerError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SchedulerError(f"{name} must be ASCII") from exc
    return value


@dataclass(frozen=True)
class ResourceProfile:
    resource_id: str
    kind: str
    capacity: int
    ready: bool
    identity: str

    @classmethod
    def from_json(cls, value: object) -> "ResourceProfile":
        row = _mapping("resource", value)
        result = cls(
            resource_id=_text("resource_id", row.get("resource_id")),
            kind=_text("resource.kind", row.get("kind")),
            capacity=_strict_int("resource.capacity", row.get("capacity"), 1),
            ready=_strict_bool("resource.ready", row.get("ready")),
            identity=_text("resource.identity", row.get("identity")),
        )
        return result


@dataclass(frozen=True)
class LeaseDemand:
    lease_id: str
    resource_id: str
    slots: int
    start_offset_us: int
    duration_us: int
    duration_upper_us: int
