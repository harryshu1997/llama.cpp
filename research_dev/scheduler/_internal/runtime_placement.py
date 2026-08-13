"""Runtime selection of measured heterogeneous placement candidates."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .capacity import DeviceMemoryCapacity
from .types import MetricEstimate, canonical_sha256


RUNTIME_PLACEMENT_SCHEMA = "research-scheduler-runtime-placement-v1"
RUNTIME_PLACEMENT_STATUSES = frozenset({"estimated", "measured"})

__all__ = [
    "RUNTIME_PLACEMENT_SCHEMA",
    "RuntimePlacementCandidate",
    "RuntimePlacementDecision",
    "RuntimePlacementError",
    "RuntimePlacementPlanner",
    "RuntimePlacementSnapshot",
]


class RuntimePlacementError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise RuntimePlacementError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimePlacementError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimePlacementError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise RuntimePlacementError(f"{name} must be bool")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RuntimePlacementError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise RuntimePlacementError(f"{name} must be an object")
    return value


def _metric(name: str, value: object) -> MetricEstimate:
    if not isinstance(value, MetricEstimate):
        raise RuntimePlacementError(f"{name} must be MetricEstimate")
    return value


def _metric_from_json(name: str, value: object) -> MetricEstimate:
    row = _object(name, value)
    try:
        return MetricEstimate(
            mean=row.get("mean"),
            upper=row.get("upper"),
            lower=row.get("lower"),
            sample_count=row.get("sample_count"),
            measured=row.get("measured"),
        )
    except (TypeError, ValueError) as exc:
        raise RuntimePlacementError(f"{name} is invalid") from exc


def _metric_json(value: MetricEstimate) -> dict[str, object]:
    return {
        "lower": value.lower,
        "mean": value.mean,
        "measured": value.measured,
        "sample_count": value.sample_count,
        "upper": value.upper,
    }


def _int_mapping(name: str, value: Mapping[str, int]) -> Mapping[str, int]:
    result = {
        _text(f"{name} resource", key): _integer(
            f"{name} bytes for {key}", item
        )
        for key, item in value.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def _binding_mapping(
    value: Mapping[str, int | str],
) -> Mapping[str, int | str]:
    result: dict[str, int | str] = {}
    for key, item in value.items():
        name = _text("runtime binding name", key)
        if type(item) is int:
            result[name] = _integer(f"runtime binding {name}", item)
        elif type(item) is str:
            result[name] = _text(f"runtime binding {name}", item)
        else:
            raise RuntimePlacementError(
                f"runtime binding {name} must be int or string"
            )
    if not result:
        raise RuntimePlacementError("runtime bindings must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class RuntimePlacementSnapshot:
    snapshot_id: str
    captured_at_us: int
    valid_until_us: int
    capacities: Mapping[str, DeviceMemoryCapacity]

    def __post_init__(self) -> None:
        _text("runtime placement snapshot_id", self.snapshot_id)
        _integer("runtime placement captured_at_us", self.captured_at_us)
        _integer("runtime placement valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.captured_at_us:
            raise RuntimePlacementError(
                "runtime placement snapshot validity is empty"
            )
        capacities = dict(self.capacities)
        if not capacities or any(
            type(resource_id) is not str
            or not isinstance(capacity, DeviceMemoryCapacity)
            or capacity.resource_id != resource_id
            for resource_id, capacity in capacities.items()
        ):
            raise RuntimePlacementError(
                "runtime placement capacities are invalid"
            )
        object.__setattr__(
            self,
            "capacities",
            MappingProxyType(dict(sorted(capacities.items()))),
        )

    @classmethod
    def from_json(cls, value: object) -> "RuntimePlacementSnapshot":
        row = _object("runtime placement snapshot", value)
        raw_capacities = row.get("capacities")
        if type(raw_capacities) is not list:
            raise RuntimePlacementError(
                "runtime placement capacities must be a list"
            )
        capacities = tuple(
            DeviceMemoryCapacity.from_json(item) for item in raw_capacities
        )
        return cls(
            snapshot_id=row.get("snapshot_id"),
            captured_at_us=row.get("captured_at_us"),
            valid_until_us=row.get("valid_until_us"),
            capacities={item.resource_id: item for item in capacities},
        )

    def to_json(self) -> dict[str, object]:
        return {
            "capacities": [
                capacity.to_json()
                for capacity in self.capacities.values()
            ],
            "captured_at_us": self.captured_at_us,
            "snapshot_id": self.snapshot_id,
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class RuntimePlacementCandidate:
    candidate_id: str
    workload_id: str
    work_set_sha256: str
    energy_boundary_id: str
    additional_bytes: Mapping[str, int]
    runtime_bindings: Mapping[str, int | str]
    latency_us: MetricEstimate
    fleet_energy_uj: MetricEstimate
    status: str
    placement_verified: bool
    workload_verified: bool
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("candidate_id", "workload_id", "energy_boundary_id"):
            _text(f"runtime placement {name}", getattr(self, name))
        object.__setattr__(
            self,
            "work_set_sha256",
            _sha256("runtime placement work_set_sha256", self.work_set_sha256),
        )
        object.__setattr__(
            self,
            "additional_bytes",
            _int_mapping("runtime placement additional", self.additional_bytes),
        )
        object.__setattr__(
            self,
            "runtime_bindings",
            _binding_mapping(self.runtime_bindings),
        )
        latency = _metric("runtime placement latency", self.latency_us)
        energy = _metric("runtime placement fleet energy", self.fleet_energy_uj)
        if self.status not in RUNTIME_PLACEMENT_STATUSES:
            raise RuntimePlacementError("unknown runtime placement status")
        if self.status == "measured" and (
            not latency.measured
            or not energy.measured
            or latency.sample_count == 0
            or energy.sample_count == 0
        ):
            raise RuntimePlacementError(
                "measured runtime placement requires measured samples"
            )
        _boolean(
            "runtime placement placement_verified", self.placement_verified
        )
        _boolean(
            "runtime placement workload_verified", self.workload_verified
        )
        evidence = tuple(
            _text("runtime placement evidence id", item)
            for item in self.evidence_ids
        )
        if not evidence or len(evidence) != len(set(evidence)):
            raise RuntimePlacementError(
                "runtime placement evidence must be non-empty and unique"
            )
        object.__setattr__(self, "evidence_ids", evidence)

    @classmethod
    def from_json(cls, value: object) -> "RuntimePlacementCandidate":
        row = _object("runtime placement candidate", value)
        additional = row.get("additional_bytes")
        bindings = row.get("runtime_bindings")
        evidence = row.get("evidence_ids")
        if type(additional) is not dict or type(bindings) is not dict:
            raise RuntimePlacementError(
                "runtime placement candidate mappings are invalid"
            )
        if type(evidence) is not list:
            raise RuntimePlacementError(
                "runtime placement evidence must be a list"
            )
        return cls(
            candidate_id=row.get("candidate_id"),
            workload_id=row.get("workload_id"),
            work_set_sha256=row.get("work_set_sha256"),
            energy_boundary_id=row.get("energy_boundary_id"),
            additional_bytes=additional,
            runtime_bindings=bindings,
            latency_us=_metric_from_json(
                "runtime placement latency", row.get("latency_us")
            ),
            fleet_energy_uj=_metric_from_json(
                "runtime placement fleet energy", row.get("fleet_energy_uj")
            ),
            status=row.get("status"),
            placement_verified=row.get("placement_verified"),
            workload_verified=row.get("workload_verified"),
            evidence_ids=tuple(evidence),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "additional_bytes": dict(self.additional_bytes),
            "candidate_id": self.candidate_id,
            "energy_boundary_id": self.energy_boundary_id,
            "evidence_ids": list(self.evidence_ids),
            "fleet_energy_uj": _metric_json(self.fleet_energy_uj),
            "latency_us": _metric_json(self.latency_us),
            "placement_verified": self.placement_verified,
            "runtime_bindings": dict(self.runtime_bindings),
            "status": self.status,
            "work_set_sha256": self.work_set_sha256,
            "workload_id": self.workload_id,
            "workload_verified": self.workload_verified,
        }


@dataclass(frozen=True)
class RuntimePlacementDecision:
    snapshot: RuntimePlacementSnapshot
    baseline: RuntimePlacementCandidate
    selected: RuntimePlacementCandidate
    minimum_energy_saving_ppm: int
    maximum_latency_ppm: int
    conservative_energy_saving_ppm: int
    mean_energy_saving_ppm: int
    conservative_latency_change_ppm: int
    mean_latency_change_ppm: int
    selection_reason: str
    rejected: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, RuntimePlacementSnapshot):
            raise RuntimePlacementError("runtime placement snapshot is invalid")
        if not isinstance(self.baseline, RuntimePlacementCandidate):
            raise RuntimePlacementError("runtime placement baseline is invalid")
        if not isinstance(self.selected, RuntimePlacementCandidate):
            raise RuntimePlacementError("selected runtime placement is invalid")
        baseline_selected = (
            self.selected.candidate_id == self.baseline.candidate_id
        )
        if baseline_selected and self.selected != self.baseline:
            raise RuntimePlacementError(
                "baseline fallback must select the baseline candidate"
            )
        _integer(
            "runtime placement minimum_energy_saving_ppm",
            self.minimum_energy_saving_ppm,
        )
        _integer(
            "runtime placement maximum_latency_ppm",
            self.maximum_latency_ppm,
            1,
        )
        if self.maximum_latency_ppm > 1_000_000:
            raise RuntimePlacementError(
                "runtime placement latency limit permits a regression"
            )
        for name in (
            "conservative_energy_saving_ppm",
            "mean_energy_saving_ppm",
            "conservative_latency_change_ppm",
            "mean_latency_change_ppm",
        ):
            if type(getattr(self, name)) is not int:
                raise RuntimePlacementError(f"{name} must be int")
        if baseline_selected:
            if any(
                value != 0
                for value in (
                    self.conservative_energy_saving_ppm,
                    self.mean_energy_saving_ppm,
                    self.conservative_latency_change_ppm,
                    self.mean_latency_change_ppm,
                )
            ):
                raise RuntimePlacementError(
                    "baseline fallback metrics must be zero"
                )
        else:
            if (
                self.conservative_energy_saving_ppm
                < self.minimum_energy_saving_ppm
            ):
                raise RuntimePlacementError(
                    "selected runtime placement misses the energy gate"
                )
            if self.conservative_latency_change_ppm > (
                self.maximum_latency_ppm - 1_000_000
            ):
                raise RuntimePlacementError(
                    "selected runtime placement misses the latency gate"
                )
        _text("runtime placement selection_reason", self.selection_reason)
        if baseline_selected and self.selection_reason != (
            "BASELINE_FALLBACK_NO_ADMISSIBLE_ALTERNATIVE"
        ):
            raise RuntimePlacementError(
                "baseline fallback selection reason is invalid"
            )
        rejected = tuple(self.rejected)
        if any(
            type(item) is not tuple
            or len(item) != 2
            or any(type(value) is not str or not value for value in item)
            for item in rejected
        ):
            raise RuntimePlacementError(
                "runtime placement rejections are invalid"
            )
        rejected_ids = [item[0] for item in rejected]
        if (
            len(rejected_ids) != len(set(rejected_ids))
            or self.selected.candidate_id in rejected_ids
        ):
            raise RuntimePlacementError(
                "runtime placement rejection ids are invalid"
            )
        object.__setattr__(self, "rejected", tuple(sorted(rejected)))

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(self._json_without_hash())

    def _json_without_hash(self) -> dict[str, object]:
        return {
            "baseline": self.baseline.to_json(),
            "conservative_energy_saving_ppm": (
                self.conservative_energy_saving_ppm
            ),
            "conservative_latency_change_ppm": (
                self.conservative_latency_change_ppm
            ),
            "maximum_latency_ppm": self.maximum_latency_ppm,
            "mean_energy_saving_ppm": self.mean_energy_saving_ppm,
            "mean_latency_change_ppm": self.mean_latency_change_ppm,
            "minimum_energy_saving_ppm": self.minimum_energy_saving_ppm,
            "rejected": [
                {"candidate_id": candidate_id, "reason": reason}
                for candidate_id, reason in self.rejected
            ],
            "schema": RUNTIME_PLACEMENT_SCHEMA,
            "selected": self.selected.to_json(),
            "selection_reason": self.selection_reason,
            "snapshot": self.snapshot.to_json(),
        }

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["decision_sha256"] = self.decision_sha256
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimePlacementDecision":
        row = _object("runtime placement decision", value)
        if row.get("schema") != RUNTIME_PLACEMENT_SCHEMA:
            raise RuntimePlacementError("runtime placement schema mismatch")
        raw_rejected = row.get("rejected")
        if type(raw_rejected) is not list:
            raise RuntimePlacementError(
                "runtime placement rejected must be a list"
            )
        rejected = []
        for value in raw_rejected:
            item = _object("runtime placement rejection", value)
            rejected.append((
                _text(
                    "runtime placement rejected candidate",
                    item.get("candidate_id"),
                ),
                _text("runtime placement rejected reason", item.get("reason")),
            ))
        result = cls(
            snapshot=RuntimePlacementSnapshot.from_json(row.get("snapshot")),
            baseline=RuntimePlacementCandidate.from_json(row.get("baseline")),
            selected=RuntimePlacementCandidate.from_json(row.get("selected")),
            minimum_energy_saving_ppm=row.get("minimum_energy_saving_ppm"),
            maximum_latency_ppm=row.get("maximum_latency_ppm"),
            conservative_energy_saving_ppm=row.get(
                "conservative_energy_saving_ppm"
            ),
            mean_energy_saving_ppm=row.get("mean_energy_saving_ppm"),
            conservative_latency_change_ppm=row.get(
                "conservative_latency_change_ppm"
            ),
            mean_latency_change_ppm=row.get("mean_latency_change_ppm"),
            selection_reason=row.get("selection_reason"),
            rejected=tuple(rejected),
        )
        if row.get("decision_sha256") != result.decision_sha256:
            raise RuntimePlacementError(
                "runtime placement decision hash mismatch"
            )
        return result


class RuntimePlacementPlanner:
    def __init__(self, minimum_samples: int = 2) -> None:
        self.minimum_samples = _integer(
            "runtime placement minimum_samples", minimum_samples, 1
        )

    @staticmethod
    def _capacity_reason(
        candidate: RuntimePlacementCandidate,
        snapshot: RuntimePlacementSnapshot,
    ) -> str | None:
        for resource_id, required_bytes in candidate.additional_bytes.items():
            capacity = snapshot.capacities.get(resource_id)
            if capacity is None:
                return "RESOURCE_ABSENT:" + resource_id
            if required_bytes > capacity.available_bytes:
                return "CAPACITY:" + resource_id
        return None

    def plan(
        self,
        *,
        candidates: Sequence[RuntimePlacementCandidate],
        baseline_candidate_id: str,
        snapshot: RuntimePlacementSnapshot,
        now_us: int,
        minimum_energy_saving_ppm: int,
        maximum_latency_ppm: int,
    ) -> RuntimePlacementDecision:
        baseline_candidate_id = _text(
            "runtime placement baseline_candidate_id", baseline_candidate_id
        )
        _integer("runtime placement now_us", now_us)
        minimum_energy_saving_ppm = _integer(
            "runtime placement minimum_energy_saving_ppm",
            minimum_energy_saving_ppm,
        )
        maximum_latency_ppm = _integer(
            "runtime placement maximum_latency_ppm", maximum_latency_ppm, 1
        )
        if maximum_latency_ppm > 1_000_000:
            raise RuntimePlacementError(
                "runtime placement latency limit permits a regression"
            )
        if not isinstance(snapshot, RuntimePlacementSnapshot):
            raise RuntimePlacementError("runtime placement snapshot is invalid")
        if not snapshot.captured_at_us <= now_us < snapshot.valid_until_us:
            raise RuntimePlacementError("runtime placement snapshot is stale")
        rows = tuple(candidates)
        if not rows or any(
            not isinstance(item, RuntimePlacementCandidate) for item in rows
        ):
            raise RuntimePlacementError(
                "runtime placement candidates are invalid"
            )
        by_id = {item.candidate_id: item for item in rows}
        if len(by_id) != len(rows):
            raise RuntimePlacementError(
                "runtime placement candidate ids are not unique"
            )
        baseline = by_id.get(baseline_candidate_id)
        if baseline is None:
            raise RuntimePlacementError("runtime placement baseline is absent")
        if (
            baseline.status != "measured"
            or not baseline.placement_verified
            or not baseline.workload_verified
            or not baseline.latency_us.measured
            or not baseline.fleet_energy_uj.measured
            or baseline.latency_us.sample_count < self.minimum_samples
            or baseline.fleet_energy_uj.sample_count < self.minimum_samples
            or baseline.latency_us.lower is None
            or baseline.fleet_energy_uj.lower is None
        ):
            raise RuntimePlacementError(
                "runtime placement baseline is not fully measured and verified"
            )
        baseline_capacity_reason = self._capacity_reason(baseline, snapshot)
        if baseline_capacity_reason is not None:
            raise RuntimePlacementError(
                "runtime placement baseline does not fit: "
                + baseline_capacity_reason
            )

        admitted: list[
            tuple[RuntimePlacementCandidate, int, int, int, int]
        ] = []
        rejected: list[tuple[str, str]] = []
        for candidate in rows:
            if candidate is baseline:
                continue
            reason = None
            if candidate.workload_id != baseline.workload_id:
                reason = "WORKLOAD_MISMATCH"
            elif candidate.work_set_sha256 != baseline.work_set_sha256:
                reason = "WORK_SET_MISMATCH"
            elif candidate.energy_boundary_id != baseline.energy_boundary_id:
                reason = "ENERGY_BOUNDARY_MISMATCH"
            if reason is None:
                reason = self._capacity_reason(candidate, snapshot)
            if reason is None:
                if not candidate.workload_verified:
                    reason = "WORKLOAD_UNVERIFIED"
                elif not candidate.placement_verified:
                    reason = "PLACEMENT_UNVERIFIED"
                elif candidate.status != "measured":
                    reason = "PLACEMENT_NOT_MEASURED"
                elif not (
                    candidate.latency_us.measured
                    and candidate.fleet_energy_uj.measured
                ):
                    reason = "METRICS_NOT_MEASURED"
                elif min(
                    candidate.latency_us.sample_count,
                    candidate.fleet_energy_uj.sample_count,
                ) < self.minimum_samples:
                    reason = "METRIC_SAMPLE_COUNT"
                elif candidate.latency_us.lower is None:
                    reason = "LATENCY_LOWER_MISSING"
                elif candidate.fleet_energy_uj.lower is None:
                    reason = "ENERGY_LOWER_MISSING"
            if reason is not None:
                rejected.append((candidate.candidate_id, reason))
                continue

            conservative_energy_saving_ppm = (
                (
                    baseline.fleet_energy_uj.lower
                    - candidate.fleet_energy_uj.upper
                )
                * 1_000_000
                // baseline.fleet_energy_uj.lower
            )
            mean_energy_saving_ppm = (
                (
                    baseline.fleet_energy_uj.mean
                    - candidate.fleet_energy_uj.mean
                )
                * 1_000_000
                // baseline.fleet_energy_uj.mean
            )
            conservative_latency_change_ppm = (
                (
                    candidate.latency_us.upper
                    - baseline.latency_us.lower
                )
                * 1_000_000
                // baseline.latency_us.lower
            )
            mean_latency_change_ppm = (
                (candidate.latency_us.mean - baseline.latency_us.mean)
                * 1_000_000
                // baseline.latency_us.mean
            )
            if (
                conservative_energy_saving_ppm
                < minimum_energy_saving_ppm
            ):
                rejected.append((candidate.candidate_id, "ENERGY_GATE"))
                continue
            if conservative_latency_change_ppm > (
                maximum_latency_ppm - 1_000_000
            ):
                rejected.append((candidate.candidate_id, "LATENCY_GATE"))
                continue
            admitted.append((
                candidate,
                conservative_energy_saving_ppm,
                mean_energy_saving_ppm,
                conservative_latency_change_ppm,
                mean_latency_change_ppm,
            ))

        if not admitted:
            return RuntimePlacementDecision(
                snapshot=snapshot,
                baseline=baseline,
                selected=baseline,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                maximum_latency_ppm=maximum_latency_ppm,
                conservative_energy_saving_ppm=0,
                mean_energy_saving_ppm=0,
                conservative_latency_change_ppm=0,
                mean_latency_change_ppm=0,
                selection_reason=(
                    "BASELINE_FALLBACK_NO_ADMISSIBLE_ALTERNATIVE"
                ),
                rejected=tuple(rejected),
            )
        selected = min(
            admitted,
            key=lambda item: (
                item[0].fleet_energy_uj.upper,
                item[0].latency_us.upper,
                item[0].candidate_id,
            ),
        )
        rejected.extend(
            (item[0].candidate_id, "HIGHER_FLEET_ENERGY")
            for item in admitted
            if item is not selected
        )
        return RuntimePlacementDecision(
            snapshot=snapshot,
            baseline=baseline,
            selected=selected[0],
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            maximum_latency_ppm=maximum_latency_ppm,
            conservative_energy_saving_ppm=selected[1],
            mean_energy_saving_ppm=selected[2],
            conservative_latency_change_ppm=selected[3],
            mean_latency_change_ppm=selected[4],
            selection_reason=(
                "LOWEST_MEASURED_FLEET_ENERGY_WITH_STRICT_LATENCY_GAIN"
            ),
            rejected=tuple(rejected),
        )
