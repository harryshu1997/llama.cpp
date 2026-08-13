"""Capacity-gated contiguous CPU-prefix and accelerator-suffix placement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


__all__ = [
    "LAYER_PLACEMENT_SCHEMA",
    "CapacityError",
    "CapacityPlanner",
    "DeviceMemoryCapacity",
    "LayerPlacementCandidate",
    "LayerPlacementContract",
]


LAYER_PLACEMENT_SCHEMA = "research-scheduler-layer-placement-v1"
PLACEMENT_STATUSES = {"estimated", "measured"}


class CapacityError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise CapacityError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise CapacityError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CapacityError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise CapacityError(f"{name} must be bool")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value).removeprefix("sha256:")
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise CapacityError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + result


def _object(name: str, value: object) -> Mapping[str, Any]:
    if type(value) is not dict:
        raise CapacityError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class DeviceMemoryCapacity:
    resource_id: str
    capacity_bytes: int
    occupied_bytes: int
    reserve_bytes: int

    def __post_init__(self) -> None:
        _text("capacity resource_id", self.resource_id)
        _integer("capacity capacity_bytes", self.capacity_bytes, 1)
        _integer("capacity occupied_bytes", self.occupied_bytes)
        _integer("capacity reserve_bytes", self.reserve_bytes)
        if self.occupied_bytes + self.reserve_bytes > self.capacity_bytes:
            raise CapacityError("occupied memory and reserve exceed capacity")

    @property
    def available_bytes(self) -> int:
        return self.capacity_bytes - self.occupied_bytes - self.reserve_bytes

    @classmethod
    def from_json(cls, value: object) -> "DeviceMemoryCapacity":
        row = _object("device memory capacity", value)
        return cls(
            resource_id=row.get("resource_id"),
            capacity_bytes=row.get("capacity_bytes"),
            occupied_bytes=row.get("occupied_bytes"),
            reserve_bytes=row.get("reserve_bytes"),
        )

    def to_json(self) -> dict[str, int | str]:
        return {
            "available_bytes": self.available_bytes,
            "capacity_bytes": self.capacity_bytes,
            "occupied_bytes": self.occupied_bytes,
            "reserve_bytes": self.reserve_bytes,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True)
class LayerPlacementCandidate:
    candidate_id: str
    model_id: str
    model_sha256: str
    total_layers: int
    cpu_prefix_layers: int
    gpu_suffix_layers: int
    runtime_gpu_layers: int
    gpu_weight_bytes: int
    gpu_kv_bytes: int
    gpu_compute_bytes: int
    cpu_weight_bytes: int
    cpu_kv_bytes: int
    cpu_compute_bytes: int
    latency_us: int | None
    latency_sample_count: int
    status: str
    placement_verified: bool
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("placement candidate_id", self.candidate_id)
        _text("placement model_id", self.model_id)
        object.__setattr__(
            self,
            "model_sha256",
            _sha256("placement model_sha256", self.model_sha256),
        )
        _integer("placement total_layers", self.total_layers, 1)
        _integer("placement cpu_prefix_layers", self.cpu_prefix_layers)
        _integer("placement gpu_suffix_layers", self.gpu_suffix_layers)
        _integer("placement runtime_gpu_layers", self.runtime_gpu_layers)
        if self.cpu_prefix_layers + self.gpu_suffix_layers != self.total_layers:
            raise CapacityError("CPU prefix and GPU suffix do not cover the model")
        if not (
            self.gpu_suffix_layers
            <= self.runtime_gpu_layers
            <= self.total_layers + 1
        ):
            raise CapacityError("runtime GPU layer count is inconsistent")
        for name in (
            "gpu_weight_bytes",
            "gpu_kv_bytes",
            "gpu_compute_bytes",
            "cpu_weight_bytes",
            "cpu_kv_bytes",
            "cpu_compute_bytes",
        ):
            _integer(f"placement {name}", getattr(self, name))
        if self.gpu_weight_bytes + self.cpu_weight_bytes == 0:
            raise CapacityError("placement has no model weights")
        if self.latency_us is not None:
            _integer("placement latency_us", self.latency_us, 1)
        _integer("placement latency_sample_count", self.latency_sample_count)
        if self.status not in PLACEMENT_STATUSES:
            raise CapacityError("unknown placement status")
        if self.status == "measured" and (
            self.latency_us is None or self.latency_sample_count == 0
        ):
            raise CapacityError("measured placement requires latency samples")
        _boolean("placement placement_verified", self.placement_verified)
        evidence = tuple(_text("placement evidence id", item) for item in self.evidence_ids)
        if not evidence or len(evidence) != len(set(evidence)):
            raise CapacityError("placement evidence must be non-empty and unique")
        object.__setattr__(self, "evidence_ids", evidence)

    @property
    def gpu_total_bytes(self) -> int:
        return self.gpu_weight_bytes + self.gpu_kv_bytes + self.gpu_compute_bytes

    @property
    def cpu_total_bytes(self) -> int:
        return self.cpu_weight_bytes + self.cpu_kv_bytes + self.cpu_compute_bytes

    @property
    def cpu_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.cpu_prefix_layers))

    @property
    def gpu_layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.cpu_prefix_layers, self.total_layers))

    @classmethod
    def from_json(cls, value: object) -> "LayerPlacementCandidate":
        row = _object("layer placement candidate", value)
        return cls(
            candidate_id=row.get("candidate_id"),
            model_id=row.get("model_id"),
            model_sha256=row.get("model_sha256"),
            total_layers=row.get("total_layers"),
            cpu_prefix_layers=row.get("cpu_prefix_layers"),
            gpu_suffix_layers=row.get("gpu_suffix_layers"),
            runtime_gpu_layers=row.get("runtime_gpu_layers"),
            gpu_weight_bytes=row.get("gpu_weight_bytes"),
            gpu_kv_bytes=row.get("gpu_kv_bytes"),
            gpu_compute_bytes=row.get("gpu_compute_bytes"),
            cpu_weight_bytes=row.get("cpu_weight_bytes"),
            cpu_kv_bytes=row.get("cpu_kv_bytes"),
            cpu_compute_bytes=row.get("cpu_compute_bytes"),
            latency_us=row.get("latency_us"),
            latency_sample_count=row.get("latency_sample_count"),
            status=row.get("status"),
            placement_verified=row.get("placement_verified"),
            evidence_ids=tuple(row.get("evidence_ids", ())),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "cpu_compute_bytes": self.cpu_compute_bytes,
            "cpu_kv_bytes": self.cpu_kv_bytes,
            "cpu_prefix_layers": self.cpu_prefix_layers,
            "cpu_total_bytes": self.cpu_total_bytes,
            "cpu_weight_bytes": self.cpu_weight_bytes,
            "evidence_ids": list(self.evidence_ids),
            "gpu_compute_bytes": self.gpu_compute_bytes,
            "gpu_kv_bytes": self.gpu_kv_bytes,
            "gpu_suffix_layers": self.gpu_suffix_layers,
            "gpu_total_bytes": self.gpu_total_bytes,
            "gpu_weight_bytes": self.gpu_weight_bytes,
            "latency_sample_count": self.latency_sample_count,
            "latency_us": self.latency_us,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "placement_verified": self.placement_verified,
            "runtime_gpu_layers": self.runtime_gpu_layers,
            "status": self.status,
            "total_layers": self.total_layers,
        }


@dataclass(frozen=True)
class LayerPlacementContract:
    route_id: str
    gpu_capacity: DeviceMemoryCapacity
    cpu_capacity: DeviceMemoryCapacity
    selected: LayerPlacementCandidate
    selection_reason: str
    rejected: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        _text("layer placement route_id", self.route_id)
        if not isinstance(self.gpu_capacity, DeviceMemoryCapacity):
            raise CapacityError("GPU capacity is invalid")
        if not isinstance(self.cpu_capacity, DeviceMemoryCapacity):
            raise CapacityError("CPU capacity is invalid")
        if self.gpu_capacity.resource_id == self.cpu_capacity.resource_id:
            raise CapacityError("GPU and CPU capacities share a resource")
        if not isinstance(self.selected, LayerPlacementCandidate):
            raise CapacityError("selected layer placement is invalid")
        if not self.selected.placement_verified or self.selected.status != "measured":
            raise CapacityError("selected layer placement is not verified and measured")
        if self.selected.gpu_total_bytes > self.gpu_capacity.available_bytes:
            raise CapacityError("selected layer placement exceeds GPU capacity")
        if self.selected.cpu_total_bytes > self.cpu_capacity.available_bytes:
            raise CapacityError("selected layer placement exceeds CPU capacity")
        _text("layer placement selection_reason", self.selection_reason)
        rejected = tuple(self.rejected)
        if any(
            type(row) is not tuple
            or len(row) != 2
            or any(type(item) is not str or not item for item in row)
            for row in rejected
        ):
            raise CapacityError("layer placement rejections are invalid")
        rejected_ids = [row[0] for row in rejected]
        if (
            len(rejected_ids) != len(set(rejected_ids))
            or self.selected.candidate_id in rejected_ids
        ):
            raise CapacityError("layer placement rejection ids are invalid")
        object.__setattr__(self, "rejected", tuple(sorted(rejected)))

    @property
    def cpu_layer_ids(self) -> tuple[int, ...]:
        return self.selected.cpu_layer_ids

    @property
    def gpu_layer_ids(self) -> tuple[int, ...]:
        return self.selected.gpu_layer_ids

    @staticmethod
    def _layer_spec(layers: tuple[int, ...]) -> str:
        return "none" if not layers else f"{layers[0]}-{layers[-1]}"

    @property
    def cpu_layer_spec(self) -> str:
        return self._layer_spec(self.cpu_layer_ids)

    @property
    def gpu_layer_spec(self) -> str:
        return self._layer_spec(self.gpu_layer_ids)

    @classmethod
    def from_json(cls, value: object) -> "LayerPlacementContract":
        row = _object("layer placement contract", value)
        if row.get("schema") != LAYER_PLACEMENT_SCHEMA:
            raise CapacityError("layer placement schema mismatch")
        raw_rejected = row.get("rejected")
        if type(raw_rejected) is not list:
            raise CapacityError("layer placement rejected must be a list")
        rejected = []
        for value in raw_rejected:
            item = _object("layer placement rejection", value)
            rejected.append((
                _text("layer placement rejected candidate", item.get("candidate_id")),
                _text("layer placement rejected reason", item.get("reason")),
            ))
        return cls(
            route_id=row.get("route_id"),
            gpu_capacity=DeviceMemoryCapacity.from_json(row.get("gpu_capacity")),
            cpu_capacity=DeviceMemoryCapacity.from_json(row.get("cpu_capacity")),
            selected=LayerPlacementCandidate.from_json(row.get("selected")),
            selection_reason=row.get("selection_reason"),
            rejected=tuple(rejected),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "cpu_capacity": self.cpu_capacity.to_json(),
            "cpu_layer_spec": self.cpu_layer_spec,
            "gpu_capacity": self.gpu_capacity.to_json(),
            "gpu_layer_spec": self.gpu_layer_spec,
            "rejected": [
                {"candidate_id": candidate_id, "reason": reason}
                for candidate_id, reason in self.rejected
            ],
            "route_id": self.route_id,
            "schema": LAYER_PLACEMENT_SCHEMA,
            "selected": self.selected.to_json(),
            "selection_reason": self.selection_reason,
        }


class CapacityPlanner:
    def __init__(self, minimum_samples: int = 1) -> None:
        self.minimum_samples = _integer("capacity minimum_samples", minimum_samples, 1)

    def plan(
        self,
        *,
        route_id: str,
        candidates: Sequence[LayerPlacementCandidate],
        gpu_capacity: DeviceMemoryCapacity,
        cpu_capacity: DeviceMemoryCapacity,
    ) -> LayerPlacementContract:
        _text("capacity route_id", route_id)
        rows = tuple(candidates)
        if not rows or any(not isinstance(row, LayerPlacementCandidate) for row in rows):
            raise CapacityError("capacity candidates are invalid")
        ids = [row.candidate_id for row in rows]
        if len(ids) != len(set(ids)):
            raise CapacityError("capacity candidate ids are not unique")
        identities = {
            (row.model_id, row.model_sha256, row.total_layers) for row in rows
        }
        if len(identities) != 1:
            raise CapacityError("capacity candidates describe different models")

        admitted: list[LayerPlacementCandidate] = []
        rejected: list[tuple[str, str]] = []
        for row in rows:
            reason = None
            if row.gpu_total_bytes > gpu_capacity.available_bytes:
                reason = "GPU_CAPACITY"
            elif row.cpu_total_bytes > cpu_capacity.available_bytes:
                reason = "CPU_CAPACITY"
            elif not row.placement_verified:
                reason = "PLACEMENT_UNVERIFIED"
            elif row.status != "measured":
                reason = "PLACEMENT_NOT_MEASURED"
            elif row.latency_us is None:
                reason = "LATENCY_MISSING"
            elif row.latency_sample_count < self.minimum_samples:
                reason = "LATENCY_SAMPLE_COUNT"
            if reason is None:
                admitted.append(row)
            else:
                rejected.append((row.candidate_id, reason))
        if not admitted:
            detail = ", ".join(f"{candidate_id}={reason}" for candidate_id, reason in rejected)
            raise CapacityError(f"no verified layer placement fits capacity: {detail}")
        selected = min(
            admitted,
            key=lambda row: (
                row.latency_us,
                -row.gpu_suffix_layers,
                row.candidate_id,
            ),
        )
        rejected.extend(
            (row.candidate_id, "SLOWER_VERIFIED_PLACEMENT")
            for row in admitted
            if row is not selected
        )
        return LayerPlacementContract(
            route_id=route_id,
            gpu_capacity=gpu_capacity,
            cpu_capacity=cpu_capacity,
            selected=selected,
            selection_reason="FASTEST_VERIFIED_CAPACITY_FIT",
            rejected=tuple(rejected),
        )
