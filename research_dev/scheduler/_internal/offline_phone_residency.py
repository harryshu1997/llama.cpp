"""Scheduler-owned plans for phone FFN residency prepared before a trace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .model_placement_controller import ModelPhoneResidencyLayout
from .phone_shards import PhoneFfnResidencyLayout
from .policy import Request
from .runtime_plan import (
    RuntimeHelperExecutionEnvelope,
    RuntimeTransitionReceipt,
)
from .runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeExecutorState,
)
from .types import canonical_sha256


OFFLINE_PHONE_RESIDENCY_STATES = frozenset({
    "PLANNED", "LOADING", "PARTIAL", "READY", "FAILED",
})
OFFLINE_PHONE_RESIDENCY_STAGE_STATES = frozenset({
    "PROPOSED", "LOADING", "READY", "FAILED",
})


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ValueError(name + " is invalid")
    return value


def _sha256(name: str, value: object) -> str:
    value = _text(name, value)
    if (
        not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(name + " is invalid")
    return value


def offline_phone_workload_sha256(
    requests_by_model: Mapping[str, Sequence[Request]],
) -> str:
    """Hash the demand input without tying it to arrival wall-clock time."""

    rows = []
    for model_id, requests in sorted(requests_by_model.items()):
        _text("offline phone model", model_id)
        values = tuple(requests)
        if not values:
            raise ValueError("offline phone model request list is empty")
        for request in values:
            if not isinstance(request, Request):
                raise ValueError("offline phone request is invalid")
            request.validate()
            rows.append({
                "features": dict(sorted(request.features.items())),
                "input_tokens": request.input_tokens,
                "model_id": model_id,
                "output_tokens": request.output_tokens,
                "quality_requirement": request.quality_requirement,
                "request_id": request.request_id,
                "semantics": asdict(request.semantics),
                "workload_id": request.workload_id,
            })
    if not rows:
        raise ValueError("offline phone workload is empty")
    return canonical_sha256({
        "requests": rows,
        "schema": "research-offline-phone-workload-v1",
    })


def select_offline_resident_superset(
    layouts: Sequence[PhoneFfnResidencyLayout],
) -> PhoneFfnResidencyLayout:
    """Select the largest useful measured or learning layout that fits."""

    rows = tuple(layouts)
    if not rows or any(
        not isinstance(row, PhoneFfnResidencyLayout) for row in rows
    ):
        raise ValueError("offline phone residency candidates are invalid")
    useful = tuple(
        row for row in rows
        if row.objective_kind == "queue_rough_compute_ops"
        or row.queue_benefit > row.transition_cost
    )
    if not useful:
        raise ValueError("offline phone residency has no amortized candidate")
    return min(
        useful,
        key=lambda row: (
            -len(row.shards),
            -row.resident_bytes,
            row.objective,
            -row.queue_benefit,
            row.geometry_sha256,
        ),
    )


def verify_offline_phone_layout(
    layout: ModelPhoneResidencyLayout,
    snapshot: HeterogeneousRuntimeSnapshot,
    *,
    phone_device_id: str,
    executor_id: str,
) -> str:
    """Verify the complete physical session map without a desktop endpoint."""

    if (
        not isinstance(layout, ModelPhoneResidencyLayout)
        or not isinstance(snapshot, HeterogeneousRuntimeSnapshot)
    ):
        raise ValueError("offline phone verification input is invalid")
    _text("offline phone device", phone_device_id)
    _text("offline phone executor", executor_id)
    expected = {row.session_id: row for row in layout.layout.shards}
    observed = {
        row.session_id: row
        for row in snapshot.phone_session_residency
        if row.device_id == phone_device_id
        and row.executor_id == executor_id
    }
    if set(observed) != set(expected):
        raise ValueError("offline phone physical session map differs")
    proof_rows = []
    for session_id, shard in sorted(expected.items()):
        row = observed[session_id]
        generation = layout.layout.session_generation_by_id.get(session_id)
        if (
            row.state != "READY"
            or row.endpoint != shard.endpoint
            or row.artifact_sha256 != shard.artifact_sha256
            or row.resident_geometry_sha256
                != shard.resident_geometry_sha256
            or row.operator_plan_sha256 != shard.operator_plan_sha256
            or row.session_generation != generation
            or row.resident_bytes != shard.resident_bytes
        ):
            raise ValueError(
                "offline phone physical session identity differs: "
                + session_id
            )
        proof_rows.append(row.to_json())
    return canonical_sha256({
        "layout_generation": layout.generation,
        "layout_geometry_sha256": layout.layout.geometry_sha256,
        "phone_device_id": phone_device_id,
        "physical_sessions": proof_rows,
        "snapshot_id": snapshot.snapshot_id,
        "schema": "research-offline-phone-residency-verification-v1",
    })


def check_offline_transition_receipts(
    stage: "OfflinePhoneResidencyStage",
    receipts: Sequence[RuntimeTransitionReceipt],
) -> tuple[RuntimeTransitionReceipt, ...]:
    """Validate physical receipts against one exact offline load stage."""

    rows = tuple(receipts)
    if (
        not isinstance(stage, OfflinePhoneResidencyStage)
        or stage.state != "LOADING"
        or tuple(row.transition_id for row in rows)
            != stage.transition_ids
        or any(
            not isinstance(row, RuntimeTransitionReceipt)
            or row.ticket_id != stage.preparation_ticket_id
            or row.request_id != stage.request_id
            or row.artifact_sha256
                != stage.helper_envelope.artifact_sha256
            or row.operator_plan_sha256
                != stage.operator_plan_sha256
            or row.status != "COMPLETED"
            for row in rows
        )
    ):
        raise ValueError("offline phone transition receipt differs")
    return rows


@dataclass(frozen=True)
class OfflinePhoneResidencyStage:
    plan_id: str
    stage_id: str
    stage_index: int
    model_id: str
    request: Request
    layout: ModelPhoneResidencyLayout
    helper_envelope: RuntimeHelperExecutionEnvelope
    preparation_ticket_id: str
    transition_ids: tuple[str, ...]
    state: str = "PROPOSED"
    resource_lease_tokens: tuple[str, ...] = ()
    yielding_resource_ids: tuple[str, ...] = ()
    memory_owner_id: str = ""
    projection_token_sha256: str | None = None
    started_at_us: int | None = None
    ready_at_us: int | None = None
    verified_at_us: int | None = None
    transition_receipts: tuple[RuntimeTransitionReceipt, ...] = ()
    verification_sha256: str | None = None
    phone_safety_state: RuntimeExecutorState | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _sha256("offline phone plan", self.plan_id)
        _text("offline phone stage", self.stage_id)
        _text("offline phone stage model", self.model_id)
        if (
            type(self.stage_index) is not int
            or self.stage_index < 0
            or not isinstance(self.request, Request)
            or not isinstance(self.layout, ModelPhoneResidencyLayout)
            or not isinstance(
                self.helper_envelope, RuntimeHelperExecutionEnvelope
            )
            or self.state not in OFFLINE_PHONE_RESIDENCY_STAGE_STATES
            or self.helper_envelope.phone_layout_generation
                != self.layout.generation
            or self.helper_envelope.phone_layout_geometry_sha256
                != self.layout.layout.geometry_sha256
        ):
            raise ValueError("offline phone residency stage is invalid")
        self.request.validate()
        transitions = self.helper_envelope.preparation_transitions
        if (
            not transitions
            or tuple(row.transition_id for row in transitions)
                != tuple(self.transition_ids)
        ):
            raise ValueError("offline phone stage transition is invalid")
        _text(
            "offline phone preparation ticket", self.preparation_ticket_id
        )
        if self.memory_owner_id:
            _text("offline phone memory owner", self.memory_owner_id)
        if self.projection_token_sha256 is not None:
            _sha256(
                "offline phone projection", self.projection_token_sha256
            )
        if self.verification_sha256 is not None:
            _sha256(
                "offline phone verification", self.verification_sha256
            )
        if (
            self.phone_safety_state is not None
            and not isinstance(
                self.phone_safety_state, RuntimeExecutorState
            )
        ):
            raise ValueError("offline phone safety state is invalid")
        if any(
            value is not None and (type(value) is not int or value < 0)
            for value in (
                self.started_at_us, self.ready_at_us, self.verified_at_us,
            )
        ):
            raise ValueError("offline phone stage timestamp is invalid")
        if self.failure_reason is not None:
            _text("offline phone stage failure", self.failure_reason)

    @property
    def selected_session_id(self) -> str:
        changed = self.layout.layout.changed_session_ids
        if len(changed) != 1:
            raise ValueError("offline phone stage is not session-specific")
        return changed[0]

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def request_ticket_id(self) -> str:
        return self.stage_id

    @property
    def operator_plan_sha256(self) -> str:
        return self.helper_envelope.operator_plan_sha256

    def to_json(self) -> dict[str, object]:
        return {
            "failure_reason": self.failure_reason,
            "helper_envelope": self.helper_envelope.to_json(),
            "layout": self.layout.to_json(),
            "memory_owner_id": self.memory_owner_id,
            "model_id": self.model_id,
            "operator_plan_sha256": self.operator_plan_sha256,
            "plan_id": self.plan_id,
            "phone_safety_state": (
                None
                if self.phone_safety_state is None
                else self.phone_safety_state.to_json()
            ),
            "preparation_ticket_id": self.preparation_ticket_id,
            "projection_token_sha256": self.projection_token_sha256,
            "ready_at_us": self.ready_at_us,
            "request_id": self.request_id,
            "resource_lease_tokens": list(self.resource_lease_tokens),
            "schema": "research-offline-phone-residency-stage-v1",
            "selected_session_id": self.selected_session_id,
            "stage_id": self.stage_id,
            "stage_index": self.stage_index,
            "started_at_us": self.started_at_us,
            "state": self.state,
            "transition_ids": list(self.transition_ids),
            "transition_receipts": [
                row.to_json() for row in self.transition_receipts
            ],
            "verification_sha256": self.verification_sha256,
            "verified_at_us": self.verified_at_us,
            "yielding_resource_ids": list(self.yielding_resource_ids),
        }


@dataclass(frozen=True)
class OfflinePhoneResidencyPlan:
    plan_id: str
    workload_sha256: str
    target_layout: PhoneFfnResidencyLayout
    pending_layouts: tuple[PhoneFfnResidencyLayout, ...]
    request_by_artifact: Mapping[str, tuple[str, Request]]
    session_endpoints: Mapping[str, str]
    shared_compute_resource_id: str
    shared_transport_resource_ids: tuple[str, ...]
    workspace_bytes: int
    phone_wide_limit_bytes: int
    persistent_service_reserve_bytes: int
    created_at_us: int
    source_snapshot_id: str
    materialization_snapshot: HeterogeneousRuntimeSnapshot
    state: str = "PLANNED"
    stages: tuple[OfflinePhoneResidencyStage, ...] = ()
    unavailable_session_ids: tuple[str, ...] = ()
    finished_at_us: int | None = None
    adoption_started_at_us: int | None = None
    adoption_verified_at_us: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _sha256("offline phone plan", self.plan_id)
        _sha256("offline phone workload", self.workload_sha256)
        if (
            not isinstance(self.target_layout, PhoneFfnResidencyLayout)
            or self.state not in OFFLINE_PHONE_RESIDENCY_STATES
            or type(self.workspace_bytes) is not int
            or self.workspace_bytes < 0
            or type(self.phone_wide_limit_bytes) is not int
            or self.phone_wide_limit_bytes <= 0
            or self.target_layout.resident_bytes > self.phone_wide_limit_bytes
            or type(self.persistent_service_reserve_bytes) is not int
            or self.persistent_service_reserve_bytes < 0
            or type(self.created_at_us) is not int
            or self.created_at_us < 0
            or not isinstance(
                self.materialization_snapshot,
                HeterogeneousRuntimeSnapshot,
            )
        ):
            raise ValueError("offline phone residency plan is invalid")
        _text("offline phone snapshot", self.source_snapshot_id)
        _text(
            "offline phone shared compute", self.shared_compute_resource_id
        )
        if not self.pending_layouts and not self.stages:
            raise ValueError("offline phone residency plan has no stages")
        if any(
            not isinstance(row, PhoneFfnResidencyLayout)
            for row in self.pending_layouts
        ) or any(
            not isinstance(row, OfflinePhoneResidencyStage)
            for row in self.stages
        ):
            raise ValueError("offline phone residency plan stages are invalid")
        requests = dict(self.request_by_artifact)
        if not requests or any(
            type(value) is not tuple
            or len(value) != 2
            or type(value[0]) is not str
            or not isinstance(value[1], Request)
            for value in requests.values()
        ):
            raise ValueError("offline phone request map is invalid")
        object.__setattr__(
            self,
            "request_by_artifact",
            MappingProxyType(dict(sorted(requests.items()))),
        )
        object.__setattr__(
            self,
            "session_endpoints",
            MappingProxyType(dict(sorted(self.session_endpoints.items()))),
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(sorted(self.metadata.items()))),
        )

    @property
    def current_stage(self) -> OfflinePhoneResidencyStage | None:
        return None if not self.stages else self.stages[-1]

    @property
    def phone_safety_state(self) -> RuntimeExecutorState | None:
        return next((
            stage.phone_safety_state
            for stage in reversed(self.stages)
            if stage.phone_safety_state is not None
        ), None)

    @property
    def resident_bytes(self) -> int:
        ready = tuple(row for row in self.stages if row.state == "READY")
        return 0 if not ready else ready[-1].layout.layout.resident_bytes

    @property
    def preload_metrics(self) -> Mapping[str, object]:
        receipts = tuple(
            receipt
            for stage in self.stages
            for receipt in stage.transition_receipts
        )
        started_at_us = min(
            (row.started_us for row in receipts), default=None
        )
        finished_at_us = max(
            (row.finished_us for row in receipts), default=None
        )
        fleet_energy = {}
        transfer_energy = {}
        evidence_ids = set()
        for receipt in receipts:
            for domain_id, energy_uj in (
                receipt.fleet_energy_uj_by_domain.items()
            ):
                fleet_energy[domain_id] = (
                    fleet_energy.get(domain_id, 0) + energy_uj
                )
            for link_id, energy_uj in (
                receipt.transfer_energy_uj_by_link.items()
            ):
                transfer_energy[link_id] = (
                    transfer_energy.get(link_id, 0) + energy_uj
                )
            evidence_ids.update(receipt.measurement_evidence_ids)
        return MappingProxyType({
            "adoption_attachment_time_us": (
                None
                if self.adoption_started_at_us is None
                or self.adoption_verified_at_us is None
                else self.adoption_verified_at_us
                    - self.adoption_started_at_us
            ),
            "fleet_energy_uj": sum(fleet_energy.values()),
            "fleet_energy_uj_by_domain": dict(sorted(fleet_energy.items())),
            "measurement_evidence_ids": sorted(evidence_ids),
            "offline_preload_finished_at_us": finished_at_us,
            "offline_preload_started_at_us": started_at_us,
            "offline_preload_time_us": (
                None
                if started_at_us is None or finished_at_us is None
                else finished_at_us - started_at_us
            ),
            "physical_load_time_us": sum(
                row.finished_us - row.started_us for row in receipts
            ),
            "ready_session_count": sum(
                row.state == "READY" for row in self.stages
            ),
            "resident_bytes": self.resident_bytes,
            "transition_count": len(receipts),
            "transfer_energy_uj_by_link": dict(sorted(
                transfer_energy.items()
            )),
        })

    def to_json(self) -> dict[str, object]:
        return {
            "adoption_started_at_us": self.adoption_started_at_us,
            "adoption_verified_at_us": self.adoption_verified_at_us,
            "created_at_us": self.created_at_us,
            "finished_at_us": self.finished_at_us,
            "metadata": dict(self.metadata),
            "metrics": dict(self.preload_metrics),
            "pending_layouts": [
                row.to_json() for row in self.pending_layouts
            ],
            "persistent_service_reserve_bytes": (
                self.persistent_service_reserve_bytes
            ),
            "phone_wide_limit_bytes": self.phone_wide_limit_bytes,
            "plan_id": self.plan_id,
            "resident_bytes": self.resident_bytes,
            "schema": "research-offline-phone-residency-plan-v1",
            "session_endpoints": dict(self.session_endpoints),
            "shared_compute_resource_id": self.shared_compute_resource_id,
            "shared_transport_resource_ids": list(
                self.shared_transport_resource_ids
            ),
            "source_snapshot_id": self.source_snapshot_id,
            "stages": [row.to_json() for row in self.stages],
            "state": self.state,
            "target_layout": self.target_layout.to_json(),
            "unavailable_session_ids": list(
                self.unavailable_session_ids
            ),
            "workload_sha256": self.workload_sha256,
            "workspace_bytes": self.workspace_bytes,
        }
