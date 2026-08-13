"""Lifecycle-specific scheduler contracts."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .policy import ProfileBundle


__all__ = [
    "LifecycleProfileSet",
    "LifecycleReceipt",
    "UnifiedScheduleError",
]


class UnifiedScheduleError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise UnifiedScheduleError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise UnifiedScheduleError(f"{name} must be ASCII") from exc
    return value


@dataclass(frozen=True)
class LifecycleReceipt:
    state: str
    tail_charge_id: str | None = None

    def validate(self, reuse_states: frozenset[str]) -> None:
        _text("lifecycle state", self.state)
        if self.state in reuse_states:
            _text("lifecycle tail charge id", self.tail_charge_id)
        elif self.tail_charge_id is not None:
            raise UnifiedScheduleError(
                "only a reused lifecycle state can carry a tail charge id"
            )


@dataclass(frozen=True)
class LifecycleProfileSet:
    profile_set_id: str
    state_key: str
    profiles: Mapping[str, ProfileBundle]
    default_state: str
    reuse_states: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        _text("profile set id", self.profile_set_id)
        _text("profile state key", self.state_key)
        profiles = {
            _text("profile lifecycle state", state): profile
            for state, profile in self.profiles.items()
        }
        if not profiles or any(
            not isinstance(profile, ProfileBundle)
            for profile in profiles.values()
        ):
            raise UnifiedScheduleError(
                "lifecycle profiles must contain ProfileBundle objects"
            )
        if self.default_state not in profiles:
            raise UnifiedScheduleError(
                "lifecycle default state has no profile"
            )
        reuse_states = frozenset(
            _text("reused lifecycle state", state)
            for state in self.reuse_states
        )
        if reuse_states - set(profiles):
            raise UnifiedScheduleError(
                "reused lifecycle state has no profile"
            )
        workloads = {
            route.workload_id
            for route in next(iter(profiles.values())).routes
        }
        if not workloads:
            raise UnifiedScheduleError("lifecycle profile has no workload")
        for profile in profiles.values():
            if {
                route.workload_id for route in profile.routes
            } != workloads:
                raise UnifiedScheduleError(
                    "lifecycle profile workloads differ by state"
                )
        object.__setattr__(
            self,
            "profiles",
            MappingProxyType(dict(sorted(profiles.items()))),
        )
        object.__setattr__(self, "reuse_states", reuse_states)

    @property
    def workloads(self) -> frozenset[str]:
        return frozenset(
            route.workload_id
            for route in next(iter(self.profiles.values())).routes
        )

    def select_state(self, receipt: LifecycleReceipt | None) -> str:
        if receipt is None:
            return self.default_state
        receipt.validate(self.reuse_states)
        return (
            receipt.state
            if receipt.state in self.profiles
            else self.default_state
        )
