"""Causal request-level placement receipts for the unified scheduler."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from .policy import Decision, Request, decision_to_json
from .runtime_cost import (
    RuntimeCostEstimateSet,
    RuntimeExecutorBinding,
    RuntimeModelArtifact,
)
from .runtime_placement import RuntimePlacementSnapshot


ONLINE_PLACEMENT_SCHEMA = "research-scheduler-online-placement-v2"
ONLINE_ROUTE_FAMILIES = (
    "cpu",
    "gpu",
    "phone",
    "gpu-cpu",
    "gpu-phone",
    "cpu-phone",
)

__all__ = [
    "ONLINE_PLACEMENT_SCHEMA",
    "ONLINE_ROUTE_FAMILIES",
    "OnlinePlacementError",
    "OnlinePlacementReceipt",
    "OnlinePlacementTracker",
    "OnlineRouteFamilyEstimate",
]


class OnlinePlacementError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise OnlinePlacementError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise OnlinePlacementError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise OnlinePlacementError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _request_json(request: Request) -> dict[str, object]:
    return {
        "arrival_us": request.arrival_us,
        "deadline_us": request.deadline_us,
        "features": dict(sorted(request.features.items())),
        "input_tokens": request.input_tokens,
        "output_tokens": request.output_tokens,
        "quality_requirement": request.quality_requirement,
        "request_id": request.request_id,
        "semantics": {
            "cancelled": request.semantics.cancelled,
            "cancellation_required": (
                request.semantics.cancellation_required
            ),
            "context_shift_required": (
                request.semantics.context_shift_required
            ),
            "full_logits_required": (
                request.semantics.full_logits_required
            ),
            "grammar_required": request.semantics.grammar_required,
            "kv_migration_required": (
                request.semantics.kv_migration_required
            ),
            "kv_owner": request.semantics.kv_owner,
            "sampler_location": request.semantics.sampler_location,
            "speculative_decode": request.semantics.speculative_decode,
        },
        "workload_id": request.workload_id,
    }


@dataclass(frozen=True)
class OnlineRouteFamilyEstimate:
    family: str
    route_id: str | None
    executor_id: str | None
    admitted: bool
    reason: str
    service_upper_us: int | None
    fleet_energy_upper_uj: int | None

    def __post_init__(self) -> None:
        if self.family not in ONLINE_ROUTE_FAMILIES:
            raise OnlinePlacementError("unknown online route family")
        if self.route_id is not None:
            _text("online route id", self.route_id)
        if self.executor_id is not None:
            _text("online executor id", self.executor_id)
        if type(self.admitted) is not bool:
            raise OnlinePlacementError("online route admitted must be bool")
        _text("online route reason", self.reason)
        if self.service_upper_us is not None:
            _integer(
                "online route service_upper_us",
                self.service_upper_us,
                1,
            )
        if self.fleet_energy_upper_uj is not None:
            _integer(
                "online route fleet_energy_upper_uj",
                self.fleet_energy_upper_uj,
                1,
            )

    def to_json(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "executor_id": self.executor_id,
            "family": self.family,
            "fleet_energy_upper_uj": self.fleet_energy_upper_uj,
            "reason": self.reason,
            "route_id": self.route_id,
            "service_upper_us": self.service_upper_us,
        }


@dataclass(frozen=True)
class OnlinePlacementReceipt:
    sequence_index: int
    observed_at_us: int
    previous_prefix_sha256: str
    causal_input_sha256: str
    causal_input: Mapping[str, object]
    prefix_sha256: str
    request_id: str
    workload_id: str
    snapshot_id: str
    selected_family: str
    selected_route_id: str
    family_estimates: tuple[OnlineRouteFamilyEstimate, ...]
    decision: Decision

    def __post_init__(self) -> None:
        _integer("online sequence_index", self.sequence_index)
        _integer("online observed_at_us", self.observed_at_us)
        for name in (
            "previous_prefix_sha256",
            "causal_input_sha256",
            "prefix_sha256",
        ):
            digest = _text(f"online {name}", getattr(self, name))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef"
                for character in digest
            ):
                raise OnlinePlacementError(
                    f"online {name} must be a lowercase SHA-256"
                )
        if type(self.causal_input) is not dict:
            raise OnlinePlacementError(
                "online causal_input must be an object"
            )
        if hashlib.sha256(_canonical(self.causal_input)).hexdigest() != (
            self.causal_input_sha256
        ):
            raise OnlinePlacementError("online causal input hash mismatch")
        _text("online request_id", self.request_id)
        _text("online workload_id", self.workload_id)
        _text("online snapshot_id", self.snapshot_id)
        if self.selected_family not in ONLINE_ROUTE_FAMILIES:
            raise OnlinePlacementError("unknown selected route family")
        _text("online selected_route_id", self.selected_route_id)
        rows = tuple(self.family_estimates)
        if (
            len(rows) != len(ONLINE_ROUTE_FAMILIES)
            or tuple(row.family for row in rows) != ONLINE_ROUTE_FAMILIES
        ):
            raise OnlinePlacementError(
                "online receipt must contain the six route families"
            )
        if not isinstance(self.decision, Decision):
            raise OnlinePlacementError("online decision is invalid")
        if (
            self.decision.request_id != self.request_id
            or self.decision.workload_id != self.workload_id
            or self.decision.route_id != self.selected_route_id
        ):
            raise OnlinePlacementError(
                "online decision identity differs from its receipt"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "causal_input": json.loads(_canonical(self.causal_input)),
            "causal_input_sha256": self.causal_input_sha256,
            "decision": decision_to_json(self.decision),
            "family_estimates": [
                row.to_json() for row in self.family_estimates
            ],
            "observed_at_us": self.observed_at_us,
            "prefix_sha256": self.prefix_sha256,
            "previous_prefix_sha256": self.previous_prefix_sha256,
            "request_id": self.request_id,
            "schema": ONLINE_PLACEMENT_SCHEMA,
            "selected_family": self.selected_family,
            "selected_route_id": self.selected_route_id,
            "sequence_index": self.sequence_index,
            "snapshot_id": self.snapshot_id,
            "workload_id": self.workload_id,
        }


class OnlinePlacementTracker:
    """Append-only proof that decisions use only an observed request prefix."""

    def __init__(self) -> None:
        self._next_sequence_index = 0
        self._last_observed_at_us = 0
        self._prefix_sha256 = "0" * 64
        self._request_ids: set[str] = set()

    @property
    def next_sequence_index(self) -> int:
        return self._next_sequence_index

    @property
    def prefix_sha256(self) -> str:
        return self._prefix_sha256

    @staticmethod
    def validate_family_routes(
        family_routes: Mapping[str, str | None],
    ) -> Mapping[str, str | None]:
        if not isinstance(family_routes, Mapping):
            raise OnlinePlacementError(
                "online family routes must be a mapping"
            )
        if tuple(family_routes) != ONLINE_ROUTE_FAMILIES:
            raise OnlinePlacementError(
                "online family routes must list the six families in order"
            )
        result: dict[str, str | None] = {}
        used: set[str] = set()
        for family in ONLINE_ROUTE_FAMILIES:
            route_id = family_routes[family]
            if route_id is not None:
                route_id = _text("online family route id", route_id)
                if route_id in used:
                    raise OnlinePlacementError(
                        "online route id belongs to multiple families"
                    )
                used.add(route_id)
            result[family] = route_id
        return MappingProxyType(result)

    def prepare(
        self,
        *,
        request: Request,
        model: RuntimeModelArtifact,
        bindings: Sequence[RuntimeExecutorBinding],
        snapshot: RuntimePlacementSnapshot,
        observed_at_us: int,
        family_routes: Mapping[str, str | None],
        scheduler_state: Mapping[str, object],
    ) -> tuple[Mapping[str, str | None], str, dict[str, object]]:
        if not isinstance(request, Request):
            raise OnlinePlacementError("online request is invalid")
        request.validate()
        if not isinstance(model, RuntimeModelArtifact):
            raise OnlinePlacementError("online model is invalid")
        if not isinstance(snapshot, RuntimePlacementSnapshot):
            raise OnlinePlacementError("online snapshot is invalid")
        _integer("online observed_at_us", observed_at_us)
        if observed_at_us < request.arrival_us:
            raise OnlinePlacementError(
                "online request was observed before its arrival"
            )
        if observed_at_us < self._last_observed_at_us:
            raise OnlinePlacementError(
                "online observation time moved backward"
            )
        if request.request_id in self._request_ids:
            raise OnlinePlacementError(
                "online request was already placed"
            )
        rows = tuple(bindings)
        if any(
            not isinstance(row, RuntimeExecutorBinding) for row in rows
        ):
            raise OnlinePlacementError("online executor binding is invalid")
        routes = self.validate_family_routes(family_routes)
        if type(scheduler_state) is not dict:
            raise OnlinePlacementError(
                "online scheduler state must be an object"
            )
        causal_input = {
            "bindings": [
                row.to_json()
                for row in sorted(rows, key=lambda item: item.route_id)
            ],
            "family_routes": dict(routes),
            "model": model.to_json(),
            "observed_at_us": observed_at_us,
            "previous_prefix_sha256": self._prefix_sha256,
            "request": _request_json(request),
            "scheduler_state": scheduler_state,
            "sequence_index": self._next_sequence_index,
            "snapshot": snapshot.to_json(),
        }
        digest = hashlib.sha256(_canonical(causal_input)).hexdigest()
        return routes, digest, causal_input

    def commit(
        self,
        *,
        request: Request,
        snapshot: RuntimePlacementSnapshot,
        observed_at_us: int,
        family_routes: Mapping[str, str | None],
        causal_input: Mapping[str, object],
        causal_input_sha256: str,
        estimates: RuntimeCostEstimateSet,
        decision: Decision,
    ) -> OnlinePlacementReceipt:
        routes = self.validate_family_routes(family_routes)
        by_route = {row.route_id: row for row in estimates.estimates}
        selected_family = next(
            (
                family
                for family, route_id in routes.items()
                if route_id == decision.route_id
            ),
            None,
        )
        if selected_family is None:
            raise OnlinePlacementError(
                "selected route has no online route family"
            )
        family_estimates = []
        for family in ONLINE_ROUTE_FAMILIES:
            route_id = routes[family]
            estimate = None if route_id is None else by_route.get(route_id)
            if route_id is None:
                family_estimates.append(OnlineRouteFamilyEstimate(
                    family=family,
                    route_id=None,
                    executor_id=None,
                    admitted=False,
                    reason="ROUTE_NOT_PROFILED",
                    service_upper_us=None,
                    fleet_energy_upper_uj=None,
                ))
            elif estimate is None:
                family_estimates.append(OnlineRouteFamilyEstimate(
                    family=family,
                    route_id=route_id,
                    executor_id=None,
                    admitted=False,
                    reason="RUNTIME_COST_ABSENT",
                    service_upper_us=None,
                    fleet_energy_upper_uj=None,
                ))
            else:
                family_estimates.append(OnlineRouteFamilyEstimate(
                    family=family,
                    route_id=route_id,
                    executor_id=estimate.executor_id,
                    admitted=estimate.admitted,
                    reason=estimate.reason,
                    service_upper_us=estimate.service_upper_us,
                    fleet_energy_upper_uj=estimate.fleet_energy_upper_uj,
                ))
        previous = self._prefix_sha256
        if type(causal_input) is not dict:
            raise OnlinePlacementError(
                "online causal input must be an object"
            )
        if hashlib.sha256(_canonical(causal_input)).hexdigest() != (
            causal_input_sha256
        ):
            raise OnlinePlacementError("online causal input hash mismatch")
        receipt_body = {
            "causal_input_sha256": causal_input_sha256,
            "decision": decision_to_json(decision),
            "family_estimates": [
                row.to_json() for row in family_estimates
            ],
            "observed_at_us": observed_at_us,
            "previous_prefix_sha256": previous,
            "request_id": request.request_id,
            "selected_family": selected_family,
            "selected_route_id": decision.route_id,
            "sequence_index": self._next_sequence_index,
            "snapshot_id": snapshot.snapshot_id,
            "workload_id": request.workload_id,
        }
        prefix = hashlib.sha256(_canonical(receipt_body)).hexdigest()
        receipt = OnlinePlacementReceipt(
            sequence_index=self._next_sequence_index,
            observed_at_us=observed_at_us,
            previous_prefix_sha256=previous,
            causal_input_sha256=causal_input_sha256,
            causal_input=dict(causal_input),
            prefix_sha256=prefix,
            request_id=request.request_id,
            workload_id=request.workload_id,
            snapshot_id=snapshot.snapshot_id,
            selected_family=selected_family,
            selected_route_id=decision.route_id,
            family_estimates=tuple(family_estimates),
            decision=decision,
        )
        self._next_sequence_index += 1
        self._last_observed_at_us = observed_at_us
        self._prefix_sha256 = prefix
        self._request_ids.add(request.request_id)
        return receipt
