"""Deadline-bounded GPU backfill using already-resident weight slices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, NamedTuple, Sequence

from .dynamic_residency import DynamicResidencySnapshot
from .types import MetricEstimate, canonical_sha256


GPU_BACKFILL_SCHEMA = "research-scheduler-gpu-backfill-v1"
GPU_WAVEFRONT_SCHEMA = "research-scheduler-gpu-wavefront-v1"
GPU_WAVEFRONT_OBJECTIVES = frozenset({
    "coverage_then_energy",
    "energy_then_coverage",
})

__all__ = [
    "GPU_BACKFILL_SCHEMA",
    "GPU_WAVEFRONT_OBJECTIVES",
    "GPU_WAVEFRONT_SCHEMA",
    "GpuBackfillCandidate",
    "GpuBackfillDecision",
    "GpuBackfillError",
    "GpuBubbleWindow",
    "GpuReadyChunk",
    "GpuWavefrontDecision",
    "GpuWavefrontSnapshot",
    "select_gpu_backfill",
    "select_gpu_wavefront_backfill",
]


class GpuBackfillError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise GpuBackfillError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise GpuBackfillError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise GpuBackfillError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise GpuBackfillError(f"{name} must be a lowercase SHA-256")
    return "sha256:" + digest


def _unique(
    name: str,
    values: Sequence[str],
    *,
    nonempty: bool = False,
) -> tuple[str, ...]:
    rows = tuple(_text(name, value) for value in values)
    if (nonempty and not rows) or len(rows) != len(set(rows)):
        qualifier = "non-empty and " if nonempty else ""
        raise GpuBackfillError(f"{name} must be {qualifier}unique")
    return rows


def _evidence(values: Sequence[str]) -> tuple[str, ...]:
    return _unique("GPU backfill evidence id", values, nonempty=True)


def _metric(name: str, value: MetricEstimate) -> None:
    if not isinstance(value, MetricEstimate):
        raise GpuBackfillError(f"{name} must be MetricEstimate")
    if type(value.measured) is not bool:
        raise GpuBackfillError(f"{name} measured must be bool")


def _metric_json(value: MetricEstimate) -> dict[str, object]:
    return {
        "lower": value.lower,
        "mean": value.mean,
        "measured": value.measured,
        "sample_count": value.sample_count,
        "upper": value.upper,
    }


@dataclass(frozen=True)
class GpuBubbleWindow:
    bubble_id: str
    fence_receipt_id: str
    source_snapshot_id: str
    source_snapshot_sha256: str
    source_generation: int
    source_epoch_key: str
    gpu_resource_id: str
    protected_owner_id: str
    protected_model_id: str
    captured_at_us: int
    valid_until_us: int
    protected_ready_lower_us: int
    guard_us: int
    runtime_verified: bool
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "bubble_id",
            "fence_receipt_id",
            "source_snapshot_id",
            "gpu_resource_id",
            "protected_owner_id",
            "protected_model_id",
        ):
            _text(f"GPU bubble {name}", getattr(self, name))
        _integer("GPU bubble source_generation", self.source_generation, 1)
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _sha256(
                "GPU bubble source_snapshot_sha256",
                self.source_snapshot_sha256,
            ),
        )
        object.__setattr__(
            self,
            "source_epoch_key",
            _sha256("GPU bubble source_epoch_key", self.source_epoch_key),
        )
        _integer("GPU bubble captured_at_us", self.captured_at_us)
        _integer("GPU bubble valid_until_us", self.valid_until_us, 1)
        _integer(
            "GPU bubble protected_ready_lower_us",
            self.protected_ready_lower_us,
            1,
        )
        _integer("GPU bubble guard_us", self.guard_us)
        if type(self.runtime_verified) is not bool:
            raise GpuBackfillError("GPU bubble runtime_verified must be bool")
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))
        if not (
            self.captured_at_us
            < self.valid_until_us
            <= self.protected_ready_lower_us
        ):
            raise GpuBackfillError("GPU bubble interval is invalid")

    def to_json(self) -> dict[str, object]:
        return {
            "bubble_id": self.bubble_id,
            "captured_at_us": self.captured_at_us,
            "evidence_ids": list(self.evidence_ids),
            "fence_receipt_id": self.fence_receipt_id,
            "gpu_resource_id": self.gpu_resource_id,
            "guard_us": self.guard_us,
            "protected_model_id": self.protected_model_id,
            "protected_owner_id": self.protected_owner_id,
            "protected_ready_lower_us": self.protected_ready_lower_us,
            "runtime_verified": self.runtime_verified,
            "schema": GPU_BACKFILL_SCHEMA,
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class GpuBackfillCandidate:
    candidate_id: str
    work_id: str
    model_id: str
    gpu_resource_id: str
    workspace_resource_id: str
    workspace_bytes: int
    required_placement_ids: tuple[str, ...]
    additional_resource_ids: tuple[str, ...]
    deadline_us: int
    energy_boundary_id: str
    accounting_scope: str
    service_latency_us: MetricEstimate
    restore_latency_us: MetricEstimate
    avoided_energy_uj: MetricEstimate
    backfill_energy_uj: MetricEstimate
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "work_id",
            "model_id",
            "gpu_resource_id",
            "workspace_resource_id",
            "energy_boundary_id",
            "accounting_scope",
        ):
            _text(f"GPU backfill {name}", getattr(self, name))
        _integer("GPU backfill workspace_bytes", self.workspace_bytes)
        object.__setattr__(
            self,
            "required_placement_ids",
            _unique(
                "GPU backfill required placement",
                self.required_placement_ids,
                nonempty=True,
            ),
        )
        additional = _unique(
            "GPU backfill additional resource", self.additional_resource_ids
        )
        if self.gpu_resource_id in additional:
            raise GpuBackfillError(
                "GPU backfill repeats its GPU resource"
            )
        object.__setattr__(self, "additional_resource_ids", additional)
        _integer("GPU backfill deadline_us", self.deadline_us, 1)
        for name in (
            "service_latency_us",
            "restore_latency_us",
            "avoided_energy_uj",
            "backfill_energy_uj",
        ):
            _metric(f"GPU backfill {name}", getattr(self, name))
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    @property
    def execution_resource_ids(self) -> tuple[str, ...]:
        return (self.gpu_resource_id, *self.additional_resource_ids)

    @property
    def lease_duration_mean_us(self) -> int:
        return self.service_latency_us.mean + self.restore_latency_us.mean

    @property
    def lease_duration_upper_us(self) -> int:
        return self.service_latency_us.upper + self.restore_latency_us.upper

    def to_json(self) -> dict[str, object]:
        return {
            "accounting_scope": self.accounting_scope,
            "additional_resource_ids": list(self.additional_resource_ids),
            "avoided_energy_uj": _metric_json(self.avoided_energy_uj),
            "backfill_energy_uj": _metric_json(self.backfill_energy_uj),
            "candidate_id": self.candidate_id,
            "deadline_us": self.deadline_us,
            "energy_boundary_id": self.energy_boundary_id,
            "evidence_ids": list(self.evidence_ids),
            "gpu_resource_id": self.gpu_resource_id,
            "model_id": self.model_id,
            "required_placement_ids": list(self.required_placement_ids),
            "restore_latency_us": _metric_json(self.restore_latency_us),
            "service_latency_us": _metric_json(self.service_latency_us),
            "work_id": self.work_id,
            "workspace_bytes": self.workspace_bytes,
            "workspace_resource_id": self.workspace_resource_id,
        }


@dataclass(frozen=True)
class GpuBackfillDecision:
    candidate_id: str | None
    reason: str
    bubble_id: str
    bubble_sha256: str
    source_snapshot_id: str
    source_generation: int
    source_epoch_key: str
    protected_owner_id: str
    protected_ready_lower_us: int
    start_us: int | None
    work_finish_upper_us: int | None
    restore_finish_upper_us: int | None
    slack_after_guard_us: int | None
    energy_saving_lower_uj: int | None
    energy_saving_ppm: int | None
    required_placement_ids: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        return {
            "bubble_id": self.bubble_id,
            "bubble_sha256": self.bubble_sha256,
            "candidate_id": self.candidate_id,
            "energy_saving_lower_uj": self.energy_saving_lower_uj,
            "energy_saving_ppm": self.energy_saving_ppm,
            "protected_owner_id": self.protected_owner_id,
            "protected_ready_lower_us": self.protected_ready_lower_us,
            "reason": self.reason,
            "rejected": [
                {"candidate_id": candidate_id, "reason": reason}
                for candidate_id, reason in self.rejected
            ],
            "required_placement_ids": list(self.required_placement_ids),
            "restore_finish_upper_us": self.restore_finish_upper_us,
            "slack_after_guard_us": self.slack_after_guard_us,
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "start_us": self.start_us,
            "work_finish_upper_us": self.work_finish_upper_us,
        }


@dataclass(frozen=True)
class GpuReadyChunk:
    chunk_id: str
    pipeline_id: str
    model_id: str
    sequence_index: int
    layer_start: int
    layer_end: int
    token_count: int
    ready_receipt_id: str
    input_buffer_id: str
    input_buffer_sha256: str
    predecessor_output_receipt_id: str | None
    ready_at_us: int
    valid_until_us: int
    producer_resource_ids: tuple[str, ...]
    runtime_verified: bool
    candidate: GpuBackfillCandidate
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "chunk_id",
            "pipeline_id",
            "model_id",
            "ready_receipt_id",
            "input_buffer_id",
        ):
            _text(f"GPU ready chunk {name}", getattr(self, name))
        _integer("GPU ready chunk sequence_index", self.sequence_index)
        _integer("GPU ready chunk layer_start", self.layer_start)
        _integer("GPU ready chunk layer_end", self.layer_end, 1)
        if self.layer_end <= self.layer_start:
            raise GpuBackfillError("GPU ready chunk layer range is empty")
        _integer("GPU ready chunk token_count", self.token_count, 1)
        object.__setattr__(
            self,
            "input_buffer_sha256",
            _sha256(
                "GPU ready chunk input_buffer_sha256",
                self.input_buffer_sha256,
            ),
        )
        predecessor = self.predecessor_output_receipt_id
        if predecessor is not None:
            _text("GPU ready chunk predecessor receipt", predecessor)
        if (self.sequence_index == 0) != (predecessor is None):
            raise GpuBackfillError(
                "GPU ready chunk predecessor does not match its sequence"
            )
        _integer("GPU ready chunk ready_at_us", self.ready_at_us)
        _integer("GPU ready chunk valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.ready_at_us:
            raise GpuBackfillError(
                "GPU ready chunk validity interval is empty"
            )
        object.__setattr__(
            self,
            "producer_resource_ids",
            _unique(
                "GPU ready chunk producer resource",
                self.producer_resource_ids,
                nonempty=True,
            ),
        )
        if not set(self.producer_resource_ids).issubset(
            self.candidate.execution_resource_ids
        ):
            raise GpuBackfillError(
                "GPU ready chunk producer resources are not leased by its candidate"
            )
        if type(self.runtime_verified) is not bool:
            raise GpuBackfillError(
                "GPU ready chunk runtime_verified must be bool"
            )
        if not isinstance(self.candidate, GpuBackfillCandidate):
            raise GpuBackfillError("GPU ready chunk candidate is invalid")
        if self.candidate.candidate_id != self.chunk_id:
            raise GpuBackfillError(
                "GPU ready chunk id differs from its candidate"
            )
        if self.candidate.model_id != self.model_id:
            raise GpuBackfillError(
                "GPU ready chunk model differs from its candidate"
            )
        object.__setattr__(self, "evidence_ids", _evidence(self.evidence_ids))

    def to_json(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_json(),
            "chunk_id": self.chunk_id,
            "evidence_ids": list(self.evidence_ids),
            "input_buffer_id": self.input_buffer_id,
            "input_buffer_sha256": self.input_buffer_sha256,
            "layer_end": self.layer_end,
            "layer_start": self.layer_start,
            "model_id": self.model_id,
            "pipeline_id": self.pipeline_id,
            "predecessor_output_receipt_id": (
                self.predecessor_output_receipt_id
            ),
            "producer_resource_ids": list(self.producer_resource_ids),
            "ready_at_us": self.ready_at_us,
            "ready_receipt_id": self.ready_receipt_id,
            "runtime_verified": self.runtime_verified,
            "sequence_index": self.sequence_index,
            "token_count": self.token_count,
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class GpuWavefrontSnapshot:
    wavefront_id: str
    source_snapshot_id: str
    source_snapshot_sha256: str
    source_generation: int
    source_epoch_key: str
    captured_at_us: int
    valid_until_us: int
    next_sequence_by_pipeline: Mapping[str, int]
    completed_output_receipt_ids: tuple[str, ...]
    ready_chunks: tuple[GpuReadyChunk, ...]

    def __post_init__(self) -> None:
        _text("GPU wavefront_id", self.wavefront_id)
        _text("GPU wavefront source_snapshot_id", self.source_snapshot_id)
        object.__setattr__(
            self,
            "source_snapshot_sha256",
            _sha256(
                "GPU wavefront source_snapshot_sha256",
                self.source_snapshot_sha256,
            ),
        )
        _integer("GPU wavefront source_generation", self.source_generation, 1)
        object.__setattr__(
            self,
            "source_epoch_key",
            _sha256("GPU wavefront source_epoch_key", self.source_epoch_key),
        )
        _integer("GPU wavefront captured_at_us", self.captured_at_us)
        _integer("GPU wavefront valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.captured_at_us:
            raise GpuBackfillError("GPU wavefront validity interval is empty")

        next_by_pipeline = dict(self.next_sequence_by_pipeline)
        if not next_by_pipeline:
            raise GpuBackfillError(
                "GPU wavefront pipeline sequence map is empty"
            )
        for pipeline_id, sequence_index in next_by_pipeline.items():
            _text("GPU wavefront pipeline_id", pipeline_id)
            _integer(
                "GPU wavefront next sequence_index", sequence_index
            )
        object.__setattr__(
            self,
            "next_sequence_by_pipeline",
            MappingProxyType(dict(sorted(next_by_pipeline.items()))),
        )
        completed = _unique(
            "GPU wavefront completed output receipt",
            self.completed_output_receipt_ids,
        )
        object.__setattr__(self, "completed_output_receipt_ids", completed)

        chunks = tuple(self.ready_chunks)
        if any(not isinstance(chunk, GpuReadyChunk) for chunk in chunks):
            raise GpuBackfillError("GPU wavefront ready chunk is invalid")
        identities = [
            (chunk.pipeline_id, chunk.sequence_index) for chunk in chunks
        ]
        for values, name in (
            ([chunk.chunk_id for chunk in chunks], "chunk id"),
            ([chunk.ready_receipt_id for chunk in chunks], "ready receipt"),
            ([chunk.input_buffer_id for chunk in chunks], "input buffer"),
            (identities, "pipeline sequence"),
        ):
            if len(values) != len(set(values)):
                raise GpuBackfillError(
                    f"duplicate GPU wavefront {name}"
                )
        completed_set = set(completed)
        for chunk in chunks:
            if chunk.pipeline_id not in next_by_pipeline:
                raise GpuBackfillError(
                    "GPU ready chunk pipeline is absent from wavefront"
                )
            predecessor = chunk.predecessor_output_receipt_id
            if predecessor is not None and predecessor not in completed_set:
                raise GpuBackfillError(
                    "GPU ready chunk predecessor is not completed"
                )
        object.__setattr__(self, "ready_chunks", chunks)

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_us": self.captured_at_us,
            "completed_output_receipt_ids": list(
                self.completed_output_receipt_ids
            ),
            "next_sequence_by_pipeline": dict(
                self.next_sequence_by_pipeline
            ),
            "ready_chunks": [chunk.to_json() for chunk in self.ready_chunks],
            "schema": GPU_WAVEFRONT_SCHEMA,
            "source_epoch_key": self.source_epoch_key,
            "source_generation": self.source_generation,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "valid_until_us": self.valid_until_us,
            "wavefront_id": self.wavefront_id,
        }


@dataclass(frozen=True)
class GpuWavefrontDecision:
    backfill: GpuBackfillDecision
    wavefront_id: str
    wavefront_sha256: str
    objective: str
    ready_queue_depth: int
    chunk_id: str | None
    pipeline_id: str | None
    sequence_index: int | None
    ready_receipt_id: str | None
    input_buffer_id: str | None
    input_buffer_sha256: str | None
    layer_start: int | None
    layer_end: int | None
    token_count: int | None
    bubble_usable_us: int | None
    gpu_work_coverage_ppm: int | None
    envelope_coverage_ppm: int | None

    @property
    def candidate_id(self) -> str | None:
        return self.backfill.candidate_id

    @property
    def reason(self) -> str:
        return self.backfill.reason

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        return {
            "backfill": self.backfill.to_json(),
            "bubble_usable_us": self.bubble_usable_us,
            "chunk_id": self.chunk_id,
            "envelope_coverage_ppm": self.envelope_coverage_ppm,
            "gpu_work_coverage_ppm": self.gpu_work_coverage_ppm,
            "input_buffer_id": self.input_buffer_id,
            "input_buffer_sha256": self.input_buffer_sha256,
            "layer_end": self.layer_end,
            "layer_start": self.layer_start,
            "objective": self.objective,
            "pipeline_id": self.pipeline_id,
            "ready_queue_depth": self.ready_queue_depth,
            "ready_receipt_id": self.ready_receipt_id,
            "schema": GPU_WAVEFRONT_SCHEMA,
            "sequence_index": self.sequence_index,
            "token_count": self.token_count,
            "wavefront_id": self.wavefront_id,
            "wavefront_sha256": self.wavefront_sha256,
        }


def _idle(
    bubble: GpuBubbleWindow,
    rejected: list[tuple[str, str]],
) -> GpuBackfillDecision:
    return GpuBackfillDecision(
        candidate_id=None,
        reason="LEAVE_GPU_IDLE",
        bubble_id=bubble.bubble_id,
        bubble_sha256=canonical_sha256(bubble.to_json()),
        source_snapshot_id=bubble.source_snapshot_id,
        source_generation=bubble.source_generation,
        source_epoch_key=bubble.source_epoch_key,
        protected_owner_id=bubble.protected_owner_id,
        protected_ready_lower_us=bubble.protected_ready_lower_us,
        start_us=None,
        work_finish_upper_us=None,
        restore_finish_upper_us=None,
        slack_after_guard_us=None,
        energy_saving_lower_uj=None,
        energy_saving_ppm=None,
        required_placement_ids=(),
        rejected=tuple(rejected),
    )


class _FeasibleBackfill(NamedTuple):
    negative_saving_lower: int
    restore_finish_upper_us: int
    candidate_id: str
    candidate: GpuBackfillCandidate
    start_us: int
    work_finish_upper_us: int
    slack: int
    saving_ppm: int


class _FeasibleWavefront(NamedTuple):
    score: tuple[int, int, int, str]
    chunk: GpuReadyChunk
    decision: GpuBackfillDecision
    bubble_usable_us: int
    work_coverage_ppm: int
    envelope_coverage_ppm: int


def _validate_backfill_inputs(
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    now_us: int,
    minimum_energy_saving_ppm: int,
) -> None:
    if not isinstance(snapshot, DynamicResidencySnapshot):
        raise GpuBackfillError("GPU backfill residency snapshot is invalid")
    if not isinstance(bubble, GpuBubbleWindow):
        raise GpuBackfillError("GPU bubble window is invalid")
    _integer("GPU backfill now_us", now_us)
    _integer(
        "GPU backfill minimum_energy_saving_ppm",
        minimum_energy_saving_ppm,
    )
    if minimum_energy_saving_ppm >= 1_000_000:
        raise GpuBackfillError(
            "GPU backfill energy saving gate reaches 100 percent"
        )
    if not (
        snapshot.captured_at_us <= now_us < snapshot.valid_until_us
        and bubble.captured_at_us <= now_us < bubble.valid_until_us
    ):
        raise GpuBackfillError("GPU bubble or residency snapshot is expired")
    source_identity = (
        snapshot.snapshot_id,
        snapshot.generation,
        snapshot.epoch_key,
    )
    if source_identity != (
        bubble.source_snapshot_id,
        bubble.source_generation,
        bubble.source_epoch_key,
    ):
        raise GpuBackfillError("GPU bubble residency epoch mismatch")
    if (
        bubble.source_snapshot_sha256
        != canonical_sha256(snapshot.to_json())
    ):
        raise GpuBackfillError("GPU bubble residency snapshot mismatch")


def _ready_by_candidate(
    resource_ready_us: int | Mapping[str, int | None],
    *,
    id_name: str,
    ready_name: str,
) -> Mapping[str, int | None] | None:
    if isinstance(resource_ready_us, Mapping):
        for candidate_id, ready_us in resource_ready_us.items():
            _text(id_name, candidate_id)
            if ready_us is not None:
                _integer(ready_name, ready_us)
        return resource_ready_us
    _integer(ready_name, resource_ready_us)
    return None


def _validate_backfill_candidates(
    rows: tuple[GpuBackfillCandidate, ...],
) -> None:
    if any(not isinstance(row, GpuBackfillCandidate) for row in rows):
        raise GpuBackfillError("GPU backfill candidate is invalid")
    candidate_ids = [row.candidate_id for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise GpuBackfillError("duplicate GPU backfill candidate id")
    boundary_scope = {
        (row.energy_boundary_id, row.accounting_scope) for row in rows
    }
    if len(boundary_scope) > 1:
        raise GpuBackfillError(
            "GPU fillers use different energy boundaries or scopes"
        )


def _residency_rejection(
    candidate: GpuBackfillCandidate,
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
) -> str | None:
    if candidate.gpu_resource_id != bubble.gpu_resource_id:
        return "GPU_RESOURCE_MISMATCH"
    placements = []
    for placement_id in candidate.required_placement_ids:
        placement = snapshot.placements.get(placement_id)
        if placement is None:
            return "RESIDENCY_MISSING"
        placements.append(placement)
    if any(
        placement.spec.model_id != candidate.model_id
        for placement in placements
    ):
        return "RESIDENCY_MODEL_MISMATCH"
    if any(
        placement.spec.resource_id != candidate.workspace_resource_id
        for placement in placements
    ):
        return "RESIDENCY_RESOURCE_MISMATCH"
    if any(
        candidate.gpu_resource_id
            not in placement.spec.execution_resource_ids
        for placement in placements
    ):
        return "RESIDENCY_EXECUTION_MISMATCH"
    workspace = snapshot.memory.get(candidate.workspace_resource_id)
    if workspace is None:
        return "WORKSPACE_RESOURCE_MISSING"
    if candidate.workspace_bytes > workspace.available_bytes:
        return "WORKSPACE_MEMORY"
    return None


def _evaluate_backfill_candidate(
    candidate: GpuBackfillCandidate,
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    *,
    now_us: int,
    resource_ready_us: int | Mapping[str, int | None],
    ready_by_candidate: Mapping[str, int | None] | None,
    minimum_energy_saving_ppm: int,
    require_measured: bool,
) -> str | _FeasibleBackfill:
    rejection = _residency_rejection(candidate, snapshot, bubble)
    if rejection is not None:
        return rejection
    candidate_ready_us = (
        ready_by_candidate.get(candidate.candidate_id)
        if ready_by_candidate is not None
        else resource_ready_us
    )
    if candidate_ready_us is None:
        return "RESOURCE_NOT_READY"
    metrics = (
        candidate.service_latency_us,
        candidate.restore_latency_us,
        candidate.avoided_energy_uj,
        candidate.backfill_energy_uj,
    )
    if require_measured and not all(
        row.measured and row.sample_count > 0 for row in metrics
    ):
        return "MEASUREMENT_REQUIRED"
    if candidate.avoided_energy_uj.lower is None:
        return "AVOIDED_ENERGY_LCB_MISSING"
    start_us = max(now_us, candidate_ready_us)
    work_finish_upper_us = start_us + candidate.service_latency_us.upper
    restore_finish_upper_us = (
        work_finish_upper_us + candidate.restore_latency_us.upper
    )
    if work_finish_upper_us > candidate.deadline_us:
        return "FILLER_DEADLINE"
    safe_end_us = min(
        snapshot.valid_until_us,
        bubble.valid_until_us,
        bubble.protected_ready_lower_us,
    )
    if restore_finish_upper_us + bubble.guard_us > safe_end_us:
        return "BUBBLE_TOO_SHORT"
    saving_lower = (
        candidate.avoided_energy_uj.lower
        - candidate.backfill_energy_uj.upper
    )
    if saving_lower <= 0:
        return "ENERGY_REGRESSION"
    saving_ppm = (
        saving_lower * 1_000_000 // candidate.avoided_energy_uj.lower
    )
    if saving_ppm < minimum_energy_saving_ppm:
        return "ENERGY_MARGIN"
    slack = safe_end_us - restore_finish_upper_us - bubble.guard_us
    return _FeasibleBackfill(
        -saving_lower,
        restore_finish_upper_us,
        candidate.candidate_id,
        candidate,
        start_us,
        work_finish_upper_us,
        slack,
        saving_ppm,
    )


def _selected_backfill(
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    row: _FeasibleBackfill,
    rejected: list[tuple[str, str]],
) -> GpuBackfillDecision:
    selected = row.candidate
    saving_lower = (
        selected.avoided_energy_uj.lower
        - selected.backfill_energy_uj.upper
    )
    return GpuBackfillDecision(
        candidate_id=selected.candidate_id,
        reason="ENERGY_POSITIVE_GPU_BACKFILL",
        bubble_id=bubble.bubble_id,
        bubble_sha256=canonical_sha256(bubble.to_json()),
        source_snapshot_id=snapshot.snapshot_id,
        source_generation=snapshot.generation,
        source_epoch_key=snapshot.epoch_key,
        protected_owner_id=bubble.protected_owner_id,
        protected_ready_lower_us=bubble.protected_ready_lower_us,
        start_us=row.start_us,
        work_finish_upper_us=row.work_finish_upper_us,
        restore_finish_upper_us=row.restore_finish_upper_us,
        slack_after_guard_us=row.slack,
        energy_saving_lower_uj=saving_lower,
        energy_saving_ppm=row.saving_ppm,
        required_placement_ids=selected.required_placement_ids,
        rejected=tuple(rejected),
    )


def select_gpu_backfill(
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    candidates: Sequence[GpuBackfillCandidate],
    *,
    now_us: int,
    resource_ready_us: int | Mapping[str, int | None],
    minimum_energy_saving_ppm: int = 50_000,
    require_measured: bool = True,
) -> GpuBackfillDecision:
    _validate_backfill_inputs(
        snapshot, bubble, now_us, minimum_energy_saving_ppm
    )
    ready_by_candidate = _ready_by_candidate(
        resource_ready_us,
        id_name="GPU ready candidate_id",
        ready_name="GPU backfill resource_ready_us",
    )
    rows = tuple(candidates)
    _validate_backfill_candidates(rows)

    if require_measured and not bubble.runtime_verified:
        return _idle(
            bubble,
            [
                (candidate.candidate_id, "BUBBLE_UNVERIFIED")
                for candidate in rows
            ],
        )

    rejected: list[tuple[str, str]] = []
    feasible: list[_FeasibleBackfill] = []
    for candidate in rows:
        outcome = _evaluate_backfill_candidate(
            candidate,
            snapshot,
            bubble,
            now_us=now_us,
            resource_ready_us=resource_ready_us,
            ready_by_candidate=ready_by_candidate,
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            require_measured=require_measured,
        )
        if isinstance(outcome, str):
            rejected.append((candidate.candidate_id, outcome))
            continue
        feasible.append(outcome)

    if not feasible:
        return _idle(bubble, rejected)
    return _selected_backfill(snapshot, bubble, min(feasible), rejected)


def _validate_wavefront_inputs(
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    wavefront: GpuWavefrontSnapshot,
    now_us: int,
    objective: str,
) -> None:
    if not isinstance(snapshot, DynamicResidencySnapshot):
        raise GpuBackfillError("GPU wavefront residency snapshot is invalid")
    if not isinstance(bubble, GpuBubbleWindow):
        raise GpuBackfillError("GPU wavefront bubble is invalid")
    if not isinstance(wavefront, GpuWavefrontSnapshot):
        raise GpuBackfillError("GPU wavefront snapshot is invalid")
    _integer("GPU wavefront now_us", now_us)
    if objective not in GPU_WAVEFRONT_OBJECTIVES:
        raise GpuBackfillError("unknown GPU wavefront objective")
    if not (
        snapshot.captured_at_us <= now_us < snapshot.valid_until_us
        and wavefront.captured_at_us <= now_us < wavefront.valid_until_us
    ):
        raise GpuBackfillError(
            "GPU wavefront or residency snapshot is expired"
        )
    if (
        wavefront.source_snapshot_id != snapshot.snapshot_id
        or wavefront.source_generation != snapshot.generation
        or wavefront.source_epoch_key != snapshot.epoch_key
    ):
        raise GpuBackfillError("GPU wavefront residency epoch mismatch")
    if (
        wavefront.source_snapshot_sha256
        != canonical_sha256(snapshot.to_json())
    ):
        raise GpuBackfillError("GPU wavefront residency snapshot mismatch")


def _chunk_admission_rejection(
    chunk: GpuReadyChunk,
    wavefront: GpuWavefrontSnapshot,
    bubble: GpuBubbleWindow,
    *,
    now_us: int,
    completed_receipts: set[str],
    require_measured: bool,
) -> str | None:
    candidate = chunk.candidate
    if candidate.model_id != chunk.model_id:
        return "CHUNK_MODEL_MISMATCH"
    if candidate.model_id == bubble.protected_model_id:
        return "PROTECTED_MODEL_CHUNK"
    expected_index = wavefront.next_sequence_by_pipeline[
        chunk.pipeline_id
    ]
    if chunk.sequence_index != expected_index:
        return "PIPELINE_NOT_HEAD"
    predecessor = chunk.predecessor_output_receipt_id
    if predecessor is not None and predecessor not in completed_receipts:
        return "PREDECESSOR_NOT_COMPLETE"
    if now_us < chunk.ready_at_us:
        return "INPUT_NOT_READY"
    if now_us >= chunk.valid_until_us:
        return "INPUT_EXPIRED"
    if require_measured and not chunk.runtime_verified:
        return "READY_CHUNK_UNVERIFIED"
    return None


def _wavefront_score(
    objective: str,
    chunk: GpuReadyChunk,
    decision: GpuBackfillDecision,
    work_coverage_ppm: int,
) -> tuple[int, int, int, str]:
    candidate = chunk.candidate
    if objective == "coverage_then_energy":
        return (
            -work_coverage_ppm,
            -decision.energy_saving_lower_uj,
            candidate.deadline_us,
            chunk.chunk_id,
        )
    return (
        -decision.energy_saving_lower_uj,
        -work_coverage_ppm,
        candidate.deadline_us,
        chunk.chunk_id,
    )


def _evaluate_wavefront_chunk(
    chunk: GpuReadyChunk,
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    *,
    now_us: int,
    candidate_ready_us: int | Mapping[str, int | None],
    safe_end_us: int,
    minimum_energy_saving_ppm: int,
    require_measured: bool,
    objective: str,
) -> str | _FeasibleWavefront:
    candidate = chunk.candidate
    decision = select_gpu_backfill(
        snapshot,
        bubble,
        (candidate,),
        now_us=now_us,
        resource_ready_us=candidate_ready_us,
        minimum_energy_saving_ppm=minimum_energy_saving_ppm,
        require_measured=require_measured,
    )
    if decision.candidate_id is None:
        return (
            decision.rejected[0][1]
            if decision.rejected
            else "BACKFILL_REJECTED"
        )
    if (
        decision.work_finish_upper_us is None
        or decision.start_us is None
        or decision.restore_finish_upper_us is None
        or decision.energy_saving_lower_uj is None
    ):
        raise GpuBackfillError(
            "selected GPU wavefront decision is incomplete"
        )
    if decision.work_finish_upper_us > chunk.valid_until_us:
        return "READY_CHUNK_EXPIRES"
    bubble_usable_us = (
        safe_end_us - decision.start_us - bubble.guard_us
    )
    if bubble_usable_us <= 0:
        return "BUBBLE_TOO_SHORT"
    work_coverage_ppm = min(
        1_000_000,
        candidate.service_latency_us.upper
        * 1_000_000
        // bubble_usable_us,
    )
    envelope_coverage_ppm = min(
        1_000_000,
        (decision.restore_finish_upper_us - decision.start_us)
        * 1_000_000
        // bubble_usable_us,
    )
    score = _wavefront_score(objective, chunk, decision, work_coverage_ppm)
    return _FeasibleWavefront(
        score,
        chunk,
        decision,
        bubble_usable_us,
        work_coverage_ppm,
        envelope_coverage_ppm,
    )


def _idle_wavefront(
    bubble: GpuBubbleWindow,
    wavefront: GpuWavefrontSnapshot,
    wavefront_sha256: str,
    objective: str,
    rejected: list[tuple[str, str]],
) -> GpuWavefrontDecision:
    return GpuWavefrontDecision(
        backfill=_idle(bubble, rejected),
        wavefront_id=wavefront.wavefront_id,
        wavefront_sha256=wavefront_sha256,
        objective=objective,
        ready_queue_depth=0,
        chunk_id=None,
        pipeline_id=None,
        sequence_index=None,
        ready_receipt_id=None,
        input_buffer_id=None,
        input_buffer_sha256=None,
        layer_start=None,
        layer_end=None,
        token_count=None,
        bubble_usable_us=None,
        gpu_work_coverage_ppm=None,
        envelope_coverage_ppm=None,
    )


def _selected_wavefront(
    feasible: list[_FeasibleWavefront],
    wavefront: GpuWavefrontSnapshot,
    wavefront_sha256: str,
    objective: str,
    rejected: list[tuple[str, str]],
) -> GpuWavefrontDecision:
    row = min(feasible, key=lambda row: row[0])
    selected = row.chunk
    rejected.extend(
        (chunk.chunk_id, "LOWER_WAVEFRONT_PRIORITY")
        for _, chunk, _, _, _, _ in feasible
        if chunk.chunk_id != selected.chunk_id
    )
    selected_decision = replace(
        row.decision,
        rejected=tuple(rejected),
    )
    return GpuWavefrontDecision(
        backfill=selected_decision,
        wavefront_id=wavefront.wavefront_id,
        wavefront_sha256=wavefront_sha256,
        objective=objective,
        ready_queue_depth=len(feasible),
        chunk_id=selected.chunk_id,
        pipeline_id=selected.pipeline_id,
        sequence_index=selected.sequence_index,
        ready_receipt_id=selected.ready_receipt_id,
        input_buffer_id=selected.input_buffer_id,
        input_buffer_sha256=selected.input_buffer_sha256,
        layer_start=selected.layer_start,
        layer_end=selected.layer_end,
        token_count=selected.token_count,
        bubble_usable_us=row.bubble_usable_us,
        gpu_work_coverage_ppm=row.work_coverage_ppm,
        envelope_coverage_ppm=row.envelope_coverage_ppm,
    )


def select_gpu_wavefront_backfill(
    snapshot: DynamicResidencySnapshot,
    bubble: GpuBubbleWindow,
    wavefront: GpuWavefrontSnapshot,
    *,
    now_us: int,
    resource_ready_us: int | Mapping[str, int | None],
    minimum_energy_saving_ppm: int = 50_000,
    require_measured: bool = True,
    objective: str = "coverage_then_energy",
) -> GpuWavefrontDecision:
    _validate_wavefront_inputs(snapshot, bubble, wavefront, now_us, objective)
    ready_by_candidate = _ready_by_candidate(
        resource_ready_us,
        id_name="GPU wavefront ready candidate_id",
        ready_name="GPU wavefront resource_ready_us",
    )

    rejected: list[tuple[str, str]] = []
    feasible: list[_FeasibleWavefront] = []
    safe_end_us = min(
        snapshot.valid_until_us,
        bubble.valid_until_us,
        bubble.protected_ready_lower_us,
    )
    completed_receipts = set(wavefront.completed_output_receipt_ids)
    for chunk in wavefront.ready_chunks:
        candidate = chunk.candidate
        rejection = _chunk_admission_rejection(
            chunk,
            wavefront,
            bubble,
            now_us=now_us,
            completed_receipts=completed_receipts,
            require_measured=require_measured,
        )
        if rejection is not None:
            rejected.append((chunk.chunk_id, rejection))
            continue

        candidate_ready_us: int | Mapping[str, int | None]
        if ready_by_candidate is None:
            candidate_ready_us = resource_ready_us
        else:
            candidate_ready_us = {
                candidate.candidate_id: ready_by_candidate.get(
                    candidate.candidate_id
                )
            }
        outcome = _evaluate_wavefront_chunk(
            chunk,
            snapshot,
            bubble,
            now_us=now_us,
            candidate_ready_us=candidate_ready_us,
            safe_end_us=safe_end_us,
            minimum_energy_saving_ppm=minimum_energy_saving_ppm,
            require_measured=require_measured,
            objective=objective,
        )
        if isinstance(outcome, str):
            rejected.append((chunk.chunk_id, outcome))
            continue
        feasible.append(outcome)

    wavefront_sha256 = canonical_sha256(wavefront.to_json())
    if not feasible:
        return _idle_wavefront(
            bubble, wavefront, wavefront_sha256, objective, rejected
        )
    return _selected_wavefront(
        feasible, wavefront, wavefront_sha256, objective, rejected
    )
