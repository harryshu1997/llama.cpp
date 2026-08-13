"""Priority and gap-filling arbitration for one shared phone accelerator."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .phone_residency import (
    PhoneArmGroup,
    PhoneArmSignal,
    PhoneOffloadCandidate,
    PhoneOffloadDecision,
    PhoneResidencyError,
    PhoneResidencyPlan,
    PhoneResidencySnapshot,
    select_energy_positive_offload,
)
from .types import canonical_sha256


PHONE_ARBITER_SCHEMA = "research-scheduler-phone-arbiter-v2"
PHONE_ARBITER_PRIORITIES = frozenset({"protected", "filler"})

__all__ = [
    "PHONE_ARBITER_PRIORITIES",
    "PHONE_ARBITER_SCHEMA",
    "PhoneArbiterDecision",
    "PhoneArbiterError",
    "PhoneArbiterQueue",
    "PhoneArbiterWindow",
    "PhoneReadyWork",
    "select_phone_arbiter_work",
]


class PhoneArbiterError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise PhoneArbiterError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise PhoneArbiterError(f"{name} must be ASCII") from exc
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhoneArbiterError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _text(name, value).removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PhoneArbiterError(f"{name} must be a lowercase SHA-256")
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
        raise PhoneArbiterError(f"{name} must be {qualifier}unique")
    return rows


def _arm_start(signal: PhoneArmSignal | PhoneArmGroup) -> int:
    if isinstance(signal, PhoneArmSignal):
        return signal.execute_not_before_us
    return signal.signals[0].execute_not_before_us


def _arm_json(
    signal: PhoneArmSignal | PhoneArmGroup | None,
) -> dict[str, object] | None:
    return None if signal is None else signal.to_json()


def _offload_json(decision: PhoneOffloadDecision) -> dict[str, object]:
    return {
        "arm_signal": _arm_json(decision.arm_signal),
        "candidate_id": decision.candidate_id,
        "energy_saving_ppm": decision.energy_saving_ppm,
        "exposed_join_wait_upper_us": decision.exposed_join_wait_upper_us,
        "offload_units": decision.offload_units,
        "reason": decision.reason,
        "rejected": [
            {"candidate_id": candidate_id, "reason": reason}
            for candidate_id, reason in decision.rejected
        ],
        "split_latency_upper_us": decision.split_latency_upper_us,
    }


@dataclass(frozen=True)
class PhoneArbiterWindow:
    window_id: str
    phone_snapshot_id: str
    phone_snapshot_sha256: str
    plan_id: str
    reset_generation: int
    protected_owner_id: str
    protected_model_id: str
    captured_at_us: int
    valid_until_us: int
    protected_ready_lower_us: int | None
    guard_us: int
    desktop_memory_available_bytes: int
    desktop_memory_reserve_bytes: int
    swap_in_delta_pages: int
    swap_out_delta_pages: int
    maximum_swap_io_pages: int
    duplicate_hot_slice_ids: tuple[str, ...]
    runtime_verified: bool
    evidence_ids: tuple[str, ...]
    protected_completion_receipt_id: str | None = None
    protected_completion_at_us: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "window_id",
            "phone_snapshot_id",
            "plan_id",
            "protected_owner_id",
            "protected_model_id",
        ):
            _text(f"phone arbiter {name}", getattr(self, name))
        object.__setattr__(
            self,
            "phone_snapshot_sha256",
            _sha256(
                "phone arbiter snapshot SHA-256",
                self.phone_snapshot_sha256,
            ),
        )
        for name in (
            "reset_generation",
            "captured_at_us",
            "guard_us",
            "desktop_memory_available_bytes",
            "desktop_memory_reserve_bytes",
            "swap_in_delta_pages",
            "swap_out_delta_pages",
            "maximum_swap_io_pages",
        ):
            _integer(f"phone arbiter {name}", getattr(self, name))
        _integer(
            "phone arbiter valid_until_us", self.valid_until_us, 1
        )
        if self.valid_until_us <= self.captured_at_us:
            raise PhoneArbiterError("phone arbiter window is invalid")
        completion_id = self.protected_completion_receipt_id
        completion_at_us = self.protected_completion_at_us
        if (completion_id is None) != (completion_at_us is None):
            raise PhoneArbiterError(
                "phone arbiter protected completion receipt is incomplete"
            )
        if completion_id is None:
            if self.protected_ready_lower_us is None:
                raise PhoneArbiterError(
                    "active phone arbiter window has no protected lower bound"
                )
            _integer(
                "phone arbiter protected_ready_lower_us",
                self.protected_ready_lower_us,
                1,
            )
            if self.valid_until_us > self.protected_ready_lower_us:
                raise PhoneArbiterError("phone arbiter window is invalid")
        else:
            if self.protected_ready_lower_us is not None:
                raise PhoneArbiterError(
                    "completed phone arbiter window retains a protected bound"
                )
            _text(
                "phone arbiter protected completion receipt",
                completion_id,
            )
            _integer(
                "phone arbiter protected completion time",
                completion_at_us,
            )
            if completion_at_us > self.captured_at_us:
                raise PhoneArbiterError(
                    "phone arbiter protected completion is after capture"
                )
        object.__setattr__(
            self,
            "duplicate_hot_slice_ids",
            _unique(
                "phone arbiter duplicate hot slice",
                self.duplicate_hot_slice_ids,
            ),
        )
        if type(self.runtime_verified) is not bool:
            raise PhoneArbiterError(
                "phone arbiter runtime_verified must be bool"
            )
        object.__setattr__(
            self,
            "evidence_ids",
            _unique(
                "phone arbiter evidence id",
                self.evidence_ids,
                nonempty=True,
            ),
        )

    def admission_failure(self, require_measured: bool) -> str | None:
        if self.duplicate_hot_slice_ids:
            return "DUPLICATE_HOT_WEIGHT_SLICE"
        if (
            self.desktop_memory_available_bytes
            < self.desktop_memory_reserve_bytes
        ):
            return "DESKTOP_MEMORY_RESERVE"
        if (
            self.swap_in_delta_pages + self.swap_out_delta_pages
            > self.maximum_swap_io_pages
        ):
            return "DESKTOP_SWAP_ACTIVITY"
        if require_measured and not self.runtime_verified:
            return "RUNTIME_WINDOW_UNVERIFIED"
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_us": self.captured_at_us,
            "desktop_memory_available_bytes": (
                self.desktop_memory_available_bytes
            ),
            "desktop_memory_reserve_bytes": self.desktop_memory_reserve_bytes,
            "duplicate_hot_slice_ids": list(self.duplicate_hot_slice_ids),
            "evidence_ids": list(self.evidence_ids),
            "guard_us": self.guard_us,
            "maximum_swap_io_pages": self.maximum_swap_io_pages,
            "phone_snapshot_id": self.phone_snapshot_id,
            "phone_snapshot_sha256": self.phone_snapshot_sha256,
            "plan_id": self.plan_id,
            "protected_completion_at_us": self.protected_completion_at_us,
            "protected_completion_receipt_id": (
                self.protected_completion_receipt_id
            ),
            "protected_model_id": self.protected_model_id,
            "protected_owner_id": self.protected_owner_id,
            "protected_ready_lower_us": self.protected_ready_lower_us,
            "reset_generation": self.reset_generation,
            "runtime_verified": self.runtime_verified,
            "schema": PHONE_ARBITER_SCHEMA,
            "swap_in_delta_pages": self.swap_in_delta_pages,
            "swap_out_delta_pages": self.swap_out_delta_pages,
            "valid_until_us": self.valid_until_us,
            "window_id": self.window_id,
        }


@dataclass(frozen=True)
class PhoneReadyWork:
    work_id: str
    pipeline_id: str
    request_id: str
    route_id: str
    model_id: str
    priority_class: str
    sequence_index: int
    predecessor_output_receipt_id: str | None
    ready_receipt_id: str
    ready_at_us: int
    valid_until_us: int
    deadline_us: int
    physical_m: int
    runtime_verified: bool
    candidate: PhoneOffloadCandidate
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "work_id",
            "pipeline_id",
            "request_id",
            "route_id",
            "model_id",
            "ready_receipt_id",
        ):
            _text(f"phone ready work {name}", getattr(self, name))
        if self.priority_class not in PHONE_ARBITER_PRIORITIES:
            raise PhoneArbiterError("unknown phone work priority class")
        _integer("phone ready work sequence_index", self.sequence_index)
        predecessor = self.predecessor_output_receipt_id
        if predecessor is not None:
            _text("phone ready work predecessor receipt", predecessor)
        if (self.sequence_index == 0) != (predecessor is None):
            raise PhoneArbiterError(
                "phone ready work predecessor does not match its sequence"
            )
        _integer("phone ready work ready_at_us", self.ready_at_us)
        _integer("phone ready work valid_until_us", self.valid_until_us, 1)
        _integer("phone ready work deadline_us", self.deadline_us, 1)
        _integer("phone ready work physical_m", self.physical_m, 1)
        if min(self.valid_until_us, self.deadline_us) <= self.ready_at_us:
            raise PhoneArbiterError(
                "phone ready work validity interval is empty"
            )
        if type(self.runtime_verified) is not bool:
            raise PhoneArbiterError(
                "phone ready work runtime_verified must be bool"
            )
        if not isinstance(self.candidate, PhoneOffloadCandidate):
            raise PhoneArbiterError("phone ready work candidate is invalid")
        if self.candidate.candidate_id != self.work_id:
            raise PhoneArbiterError(
                "phone ready work id differs from its candidate"
            )
        object.__setattr__(
            self,
            "evidence_ids",
            _unique(
                "phone ready work evidence id",
                self.evidence_ids,
                nonempty=True,
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate.candidate_id,
            "deadline_us": self.deadline_us,
            "evidence_ids": list(self.evidence_ids),
            "model_id": self.model_id,
            "physical_m": self.physical_m,
            "pipeline_id": self.pipeline_id,
            "predecessor_output_receipt_id": (
                self.predecessor_output_receipt_id
            ),
            "priority_class": self.priority_class,
            "ready_at_us": self.ready_at_us,
            "ready_receipt_id": self.ready_receipt_id,
            "request_id": self.request_id,
            "route_id": self.route_id,
            "runtime_verified": self.runtime_verified,
            "sequence_index": self.sequence_index,
            "valid_until_us": self.valid_until_us,
            "work_id": self.work_id,
        }


@dataclass(frozen=True)
class PhoneArbiterQueue:
    queue_id: str
    phone_snapshot_id: str
    phone_snapshot_sha256: str
    plan_id: str
    reset_generation: int
    captured_at_us: int
    valid_until_us: int
    next_sequence_by_pipeline: Mapping[str, int]
    completed_output_receipt_ids: tuple[str, ...]
    ready_work: tuple[PhoneReadyWork, ...]

    def __post_init__(self) -> None:
        for name in ("queue_id", "phone_snapshot_id", "plan_id"):
            _text(f"phone arbiter queue {name}", getattr(self, name))
        object.__setattr__(
            self,
            "phone_snapshot_sha256",
            _sha256(
                "phone arbiter queue snapshot SHA-256",
                self.phone_snapshot_sha256,
            ),
        )
        _integer("phone arbiter queue reset_generation", self.reset_generation)
        _integer("phone arbiter queue captured_at_us", self.captured_at_us)
        _integer("phone arbiter queue valid_until_us", self.valid_until_us, 1)
        if self.valid_until_us <= self.captured_at_us:
            raise PhoneArbiterError("phone arbiter queue interval is empty")
        next_by_pipeline = dict(self.next_sequence_by_pipeline)
        if not next_by_pipeline:
            raise PhoneArbiterError("phone arbiter pipeline map is empty")
        for pipeline_id, sequence_index in next_by_pipeline.items():
            _text("phone arbiter pipeline_id", pipeline_id)
            _integer("phone arbiter next sequence", sequence_index)
        object.__setattr__(
            self,
            "next_sequence_by_pipeline",
            MappingProxyType(dict(sorted(next_by_pipeline.items()))),
        )
        completed = _unique(
            "phone arbiter completed receipt",
            self.completed_output_receipt_ids,
        )
        object.__setattr__(self, "completed_output_receipt_ids", completed)
        rows = tuple(self.ready_work)
        if any(not isinstance(row, PhoneReadyWork) for row in rows):
            raise PhoneArbiterError("phone arbiter ready work is invalid")
        for values, name in (
            ([row.work_id for row in rows], "work id"),
            ([row.ready_receipt_id for row in rows], "ready receipt"),
            (
                [(row.pipeline_id, row.sequence_index) for row in rows],
                "pipeline sequence",
            ),
        ):
            if len(values) != len(set(values)):
                raise PhoneArbiterError(f"duplicate phone arbiter {name}")
        completed_set = set(completed)
        for row in rows:
            if row.pipeline_id not in next_by_pipeline:
                raise PhoneArbiterError(
                    "phone work pipeline is absent from queue"
                )
            predecessor = row.predecessor_output_receipt_id
            if predecessor is not None and predecessor not in completed_set:
                raise PhoneArbiterError(
                    "phone work predecessor is not completed"
                )
        object.__setattr__(self, "ready_work", rows)

    def to_json(self) -> dict[str, object]:
        return {
            "captured_at_us": self.captured_at_us,
            "completed_output_receipt_ids": list(
                self.completed_output_receipt_ids
            ),
            "next_sequence_by_pipeline": dict(
                self.next_sequence_by_pipeline
            ),
            "phone_snapshot_id": self.phone_snapshot_id,
            "phone_snapshot_sha256": self.phone_snapshot_sha256,
            "plan_id": self.plan_id,
            "queue_id": self.queue_id,
            "ready_work": [row.to_json() for row in self.ready_work],
            "reset_generation": self.reset_generation,
            "schema": PHONE_ARBITER_SCHEMA,
            "valid_until_us": self.valid_until_us,
        }


@dataclass(frozen=True)
class PhoneArbiterDecision:
    work_id: str | None
    pipeline_id: str | None
    sequence_index: int | None
    model_id: str | None
    priority_class: str | None
    reason: str
    queue_id: str
    queue_sha256: str
    window_id: str
    window_sha256: str
    phone_snapshot_id: str
    phone_snapshot_sha256: str
    protected_completion_receipt_id: str | None
    start_us: int | None
    phone_finish_upper_us: int | None
    safe_end_us: int | None
    slack_us: int | None
    offload: PhoneOffloadDecision
    rejected: tuple[tuple[str, str], ...]

    @property
    def decision_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    def to_json(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "offload": _offload_json(self.offload),
            "phone_finish_upper_us": self.phone_finish_upper_us,
            "phone_snapshot_id": self.phone_snapshot_id,
            "phone_snapshot_sha256": self.phone_snapshot_sha256,
            "pipeline_id": self.pipeline_id,
            "priority_class": self.priority_class,
            "protected_completion_receipt_id": (
                self.protected_completion_receipt_id
            ),
            "queue_id": self.queue_id,
            "queue_sha256": self.queue_sha256,
            "reason": self.reason,
            "rejected": [
                {"work_id": work_id, "reason": reason}
                for work_id, reason in self.rejected
            ],
            "safe_end_us": self.safe_end_us,
            "sequence_index": self.sequence_index,
            "slack_us": self.slack_us,
            "start_us": self.start_us,
            "window_id": self.window_id,
            "window_sha256": self.window_sha256,
            "work_id": self.work_id,
        }


def _idle(
    queue: PhoneArbiterQueue,
    window: PhoneArbiterWindow,
    reason: str,
    rejected: Sequence[tuple[str, str]],
) -> PhoneArbiterDecision:
    return PhoneArbiterDecision(
        work_id=None,
        pipeline_id=None,
        sequence_index=None,
        model_id=None,
        priority_class=None,
        reason=reason,
        queue_id=queue.queue_id,
        queue_sha256=canonical_sha256(queue.to_json()),
        window_id=window.window_id,
        window_sha256=canonical_sha256(window.to_json()),
        phone_snapshot_id=queue.phone_snapshot_id,
        phone_snapshot_sha256=queue.phone_snapshot_sha256,
        protected_completion_receipt_id=(
            window.protected_completion_receipt_id
        ),
        start_us=None,
        phone_finish_upper_us=None,
        safe_end_us=None,
        slack_us=None,
        offload=PhoneOffloadDecision(
            candidate_id=None,
            reason=reason,
            arm_signal=None,
            offload_units=0,
            split_latency_upper_us=None,
            exposed_join_wait_upper_us=None,
            energy_saving_ppm=None,
            rejected=(),
        ),
        rejected=tuple(rejected),
    )


def select_phone_arbiter_work(
    plan: PhoneResidencyPlan,
    snapshot: PhoneResidencySnapshot,
    window: PhoneArbiterWindow,
    queue: PhoneArbiterQueue,
    *,
    now_us: int,
    resource_ready_us: int | Mapping[str, int | None],
    minimum_energy_saving_ppm: int = 50_000,
    latency_limit_ppm: int = 1_000_000,
    maximum_join_wait_ppm: int = 50_000,
    require_measured: bool = True,
) -> PhoneArbiterDecision:
    _integer("phone arbiter now_us", now_us)
    try:
        snapshot.validate_against(plan)
    except PhoneResidencyError as exc:
        raise PhoneArbiterError(str(exc)) from exc
    snapshot_hash = canonical_sha256(snapshot.to_json())
    identity = (
        snapshot.snapshot_id,
        snapshot_hash,
        plan.plan_id,
        plan.reset_generation,
    )
    if (
        (
            queue.phone_snapshot_id,
            queue.phone_snapshot_sha256,
            queue.plan_id,
            queue.reset_generation,
        )
        != identity
        or (
            window.phone_snapshot_id,
            window.phone_snapshot_sha256,
            window.plan_id,
            window.reset_generation,
        )
        != identity
    ):
        raise PhoneArbiterError("phone arbiter snapshot identity mismatch")
    if not (
        window.captured_at_us <= now_us < window.valid_until_us
        and queue.captured_at_us <= now_us < queue.valid_until_us
    ):
        raise PhoneArbiterError("phone arbiter snapshot is stale")
    if isinstance(resource_ready_us, Mapping):
        ready_by_work = resource_ready_us
        for work_id, ready_us in ready_by_work.items():
            _text("phone arbiter ready work id", work_id)
            if ready_us is not None:
                _integer("phone arbiter resource ready time", ready_us)
    else:
        _integer("phone arbiter resource ready time", resource_ready_us)
        ready_by_work = None

    admission_failure = window.admission_failure(require_measured)
    if admission_failure is not None:
        return _idle(
            queue,
            window,
            "NO_ADMISSIBLE_PHONE_WORK",
            [(row.work_id, admission_failure) for row in queue.ready_work],
        )

    rejected: list[tuple[str, str]] = []
    feasible: list[
        tuple[
            tuple[int, int, int, str],
            PhoneReadyWork,
            PhoneOffloadDecision,
            int,
            int,
            int,
        ]
    ] = []
    completed = set(queue.completed_output_receipt_ids)
    for work in queue.ready_work:
        if work.sequence_index != queue.next_sequence_by_pipeline[
            work.pipeline_id
        ]:
            rejected.append((work.work_id, "OUT_OF_ORDER"))
            continue
        predecessor = work.predecessor_output_receipt_id
        if predecessor is not None and predecessor not in completed:
            rejected.append((work.work_id, "PREDECESSOR_NOT_COMPLETED"))
            continue
        if now_us < work.ready_at_us:
            rejected.append((work.work_id, "INPUT_NOT_READY"))
            continue
        if now_us >= min(work.valid_until_us, work.deadline_us):
            rejected.append((work.work_id, "INPUT_EXPIRED"))
            continue
        if require_measured and not work.runtime_verified:
            rejected.append((work.work_id, "READY_WORK_UNVERIFIED"))
            continue
        if (
            window.protected_completion_receipt_id is not None
            and work.priority_class == "protected"
        ):
            rejected.append((work.work_id, "PROTECTED_OWNER_COMPLETED"))
            continue
        if (
            work.priority_class == "filler"
            and work.model_id == window.protected_model_id
        ):
            rejected.append((work.work_id, "PROTECTED_MODEL_NOT_FILLER"))
            continue
        try:
            resident_models = {
                plan.slice_location(slice_id)[1].model_id
                for slice_id in work.candidate.slice_ids
            }
        except PhoneResidencyError:
            rejected.append((work.work_id, "RESIDENCY_UNKNOWN"))
            continue
        if resident_models != {work.model_id}:
            rejected.append((work.work_id, "RESIDENT_MODEL_MISMATCH"))
            continue
        safe_end_us = min(
            work.deadline_us,
            work.valid_until_us,
            queue.valid_until_us,
            window.valid_until_us,
        )
        if (
            work.priority_class == "filler"
            and window.protected_completion_receipt_id is None
        ):
            if window.protected_ready_lower_us is None:
                raise PhoneArbiterError(
                    "active phone arbiter window has no protected lower bound"
                )
            safe_end_us = min(
                safe_end_us,
                window.protected_ready_lower_us - window.guard_us,
            )
        if safe_end_us <= now_us:
            rejected.append((work.work_id, "PROTECTED_WORK_GUARD"))
            continue
        ready_us = (
            ready_by_work.get(work.work_id)
            if ready_by_work is not None
            else resource_ready_us
        )
        try:
            offload = select_energy_positive_offload(
                plan,
                snapshot,
                (work.candidate,),
                request_id=work.request_id,
                route_id=work.route_id,
                physical_m=work.physical_m,
                now_us=now_us,
                phone_resource_ready_us={work.work_id: ready_us},
                deadline_us=safe_end_us,
                minimum_energy_saving_ppm=minimum_energy_saving_ppm,
                latency_limit_ppm=latency_limit_ppm,
                maximum_join_wait_ppm=maximum_join_wait_ppm,
                require_measured=require_measured,
            )
        except PhoneResidencyError as exc:
            raise PhoneArbiterError(str(exc)) from exc
        if offload.candidate_id is None or offload.arm_signal is None:
            reason = (
                offload.rejected[0][1]
                if offload.rejected
                else "PHONE_ROUTE_REJECTED"
            )
            if (
                reason == "DEADLINE"
                and work.priority_class == "filler"
                and window.protected_completion_receipt_id is None
            ):
                reason = "PROTECTED_WORK_GUARD"
            rejected.append((work.work_id, reason))
            continue
        start_us = _arm_start(offload.arm_signal)
        phone_finish_upper_us = (
            start_us + work.candidate.phone_path_us.upper
        )
        if phone_finish_upper_us > safe_end_us:
            rejected.append((work.work_id, "PROTECTED_WORK_GUARD"))
            continue
        energy_saving_ppm = offload.energy_saving_ppm
        if energy_saving_ppm is None:
            raise PhoneArbiterError("selected phone energy decision is incomplete")
        priority_rank = 0 if work.priority_class == "protected" else 1
        score = (
            priority_rank,
            work.deadline_us,
            -energy_saving_ppm,
            work.work_id,
        )
        feasible.append((
            score,
            work,
            offload,
            start_us,
            phone_finish_upper_us,
            safe_end_us,
        ))

    if not feasible:
        return _idle(
            queue,
            window,
            "NO_ADMISSIBLE_PHONE_WORK",
            rejected,
        )
    (
        _,
        selected,
        offload,
        start_us,
        phone_finish_upper_us,
        safe_end_us,
    ) = min(feasible, key=lambda row: row[0])
    rejected.extend(
        (work.work_id, "LOWER_PHONE_PRIORITY")
        for _, work, _, _, _, _ in feasible
        if work.work_id != selected.work_id
    )
    return PhoneArbiterDecision(
        work_id=selected.work_id,
        pipeline_id=selected.pipeline_id,
        sequence_index=selected.sequence_index,
        model_id=selected.model_id,
        priority_class=selected.priority_class,
        reason=(
            "PROTECTED_PHONE_WORK"
            if selected.priority_class == "protected"
            else (
                "POST_PROTECTED_PHONE_WORK"
                if window.protected_completion_receipt_id is not None
                else "GAP_FILLING_PHONE_WORK"
            )
        ),
        queue_id=queue.queue_id,
        queue_sha256=canonical_sha256(queue.to_json()),
        window_id=window.window_id,
        window_sha256=canonical_sha256(window.to_json()),
        phone_snapshot_id=queue.phone_snapshot_id,
        phone_snapshot_sha256=queue.phone_snapshot_sha256,
        protected_completion_receipt_id=(
            window.protected_completion_receipt_id
        ),
        start_us=start_us,
        phone_finish_upper_us=phone_finish_upper_us,
        safe_end_us=safe_end_us,
        slack_us=safe_end_us - phone_finish_upper_us,
        offload=offload,
        rejected=tuple(rejected),
    )
