"""Scheduler-owned cohorts for physically batched decode execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import threading
import time
from types import MappingProxyType
from typing import Mapping, Sequence

from .policy import LeaseRecord
from .runtime_cost import RuntimeExecutorBinding
from .runtime_plan import RuntimeExecutionPlan
from .types import canonical_sha256


RUNTIME_DECODE_COHORT_SCHEMA = "research-scheduler-decode-cohort-v2"


class RuntimeDecodeCohortError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeDecodeCohortError(
            f"{name} must be non-empty ASCII text"
        )
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RuntimeDecodeCohortError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


@dataclass(frozen=True)
class RuntimeDecodeCohortKey:
    artifact_sha256: str
    endpoint: str
    desktop_placement_sha256: str
    resident_geometry_sha256: str
    transport_geometry_sha256: str
    common_policy_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "artifact_sha256",
            "endpoint",
            "desktop_placement_sha256",
            "resident_geometry_sha256",
            "transport_geometry_sha256",
            "common_policy_sha256",
        ):
            _text("decode cohort " + name, getattr(self, name))

    @property
    def key_sha256(self) -> str:
        return canonical_sha256(self.to_json())

    def to_json(self) -> dict[str, str]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "common_policy_sha256": self.common_policy_sha256,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "endpoint": self.endpoint,
            "resident_geometry_sha256": self.resident_geometry_sha256,
            "transport_geometry_sha256": self.transport_geometry_sha256,
        }


@dataclass(frozen=True)
class RuntimeDecodeCohortBinding:
    cohort_id: str
    key_sha256: str
    leader_request_id: str
    member_request_ids: tuple[str, ...]
    common_policy_sha256: str
    shared_lease_tokens: tuple[str, ...]
    active_batch: int
    maximum_members: int
    sealed: bool

    def __post_init__(self) -> None:
        for name in (
            "cohort_id",
            "key_sha256",
            "leader_request_id",
            "common_policy_sha256",
        ):
            _text("decode cohort binding " + name, getattr(self, name))
        members = tuple(self.member_request_ids)
        leases = tuple(self.shared_lease_tokens)
        if (
            not members
            or len(members) != len(set(members))
            or self.leader_request_id != members[0]
            or any(type(row) is not str or not row or not row.isascii()
                   for row in members)
            or len(leases) != len(set(leases))
            or any(type(row) is not str or not row or not row.isascii()
                   for row in leases)
        ):
            raise RuntimeDecodeCohortError(
                "decode cohort binding members or leases are invalid"
            )
        _integer("decode cohort active batch", self.active_batch, 1)
        _integer("decode cohort maximum members", self.maximum_members, 1)
        if (
            self.maximum_members > 4
            or self.active_batch != len(members)
            or self.active_batch > self.maximum_members
            or type(self.sealed) is not bool
        ):
            raise RuntimeDecodeCohortError(
                "decode cohort binding capacity is invalid"
            )
        object.__setattr__(self, "member_request_ids", members)
        object.__setattr__(self, "shared_lease_tokens", leases)

    def to_json(self) -> dict[str, object]:
        return {
            "active_batch": self.active_batch,
            "cohort_id": self.cohort_id,
            "common_policy_sha256": self.common_policy_sha256,
            "key_sha256": self.key_sha256,
            "leader_request_id": self.leader_request_id,
            "maximum_members": self.maximum_members,
            "member_request_ids": list(self.member_request_ids),
            "schema": RUNTIME_DECODE_COHORT_SCHEMA,
            "sealed": self.sealed,
            "shared_lease_tokens": list(self.shared_lease_tokens),
        }


@dataclass(frozen=True)
class RuntimeDecodeCohortAdmission:
    binding: RuntimeDecodeCohortBinding
    shared_leases: tuple[LeaseRecord, ...]
    leader: bool
    required_reserved_until_us: int


@dataclass(frozen=True)
class RuntimeDecodeCohortReplanRelease:
    previous_binding: RuntimeDecodeCohortBinding
    remaining_binding: RuntimeDecodeCohortBinding | None
    previous_lease_owner_id: str
    transferred_leases: tuple[LeaseRecord, ...]


@dataclass(frozen=True)
class RuntimeDecodeCohortReceipt:
    cohort_id: str
    member_request_ids: tuple[str, ...]
    energy_owner_request_id: str
    common_policy_sha256: str
    active_batch: int
    started_at_us: int
    finished_at_us: int
    fleet_energy_uj_by_domain: Mapping[str, int]
    transfer_energy_uj_by_link: Mapping[str, int]
    measurement_evidence_ids: tuple[str, ...]
    attribution_kind: str
    energy_boundary_id: str
    total_input_tokens: int
    total_output_tokens: int
    energy_estimation_metadata: Mapping[
        str, int | str | bool
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "cohort_id",
            "energy_owner_request_id",
            "common_policy_sha256",
            "attribution_kind",
            "energy_boundary_id",
        ):
            _text("decode cohort receipt " + name, getattr(self, name))
        members = tuple(self.member_request_ids)
        evidence = tuple(sorted(self.measurement_evidence_ids))
        energy = {
            _text("decode cohort energy domain", key): _integer(
                "decode cohort energy", value
            )
            for key, value in self.fleet_energy_uj_by_domain.items()
        }
        transfer_energy = {
            _text("decode cohort transfer link", key): _integer(
                "decode cohort transfer energy", value
            )
            for key, value in self.transfer_energy_uj_by_link.items()
        }
        metadata: dict[str, int | str | bool] = {}
        for raw_name, raw_value in self.energy_estimation_metadata.items():
            name = _text(
                "decode cohort energy estimation metadata", raw_name
            )
            if type(raw_value) is bool:
                metadata[name] = raw_value
            elif type(raw_value) is int and raw_value >= 0:
                metadata[name] = raw_value
            elif type(raw_value) is str:
                metadata[name] = _text(
                    "decode cohort energy estimation metadata value",
                    raw_value,
                )
            else:
                raise RuntimeDecodeCohortError(
                    "decode cohort energy estimation metadata is invalid"
                )
        if (
            not members
            or len(members) != len(set(members))
            or self.energy_owner_request_id not in members
            or not evidence
            or len(evidence) != len(set(evidence))
            or not energy
            or self.attribution_kind not in {
                "diagnostic",
                "isolated",
                "matched_abba",
                "device_domain",
            }
        ):
            raise RuntimeDecodeCohortError(
                "decode cohort receipt identity or evidence is invalid"
            )
        _integer("decode cohort receipt batch", self.active_batch, 1)
        _integer("decode cohort receipt start", self.started_at_us)
        _integer("decode cohort receipt finish", self.finished_at_us, 1)
        _integer(
            "decode cohort receipt total input tokens",
            self.total_input_tokens,
            1,
        )
        _integer(
            "decode cohort receipt total output tokens",
            self.total_output_tokens,
            1,
        )
        if (
            self.active_batch != len(members)
            or self.finished_at_us <= self.started_at_us
        ):
            raise RuntimeDecodeCohortError(
                "decode cohort receipt interval is invalid"
            )
        object.__setattr__(self, "member_request_ids", members)
        object.__setattr__(self, "measurement_evidence_ids", evidence)
        object.__setattr__(
            self,
            "fleet_energy_uj_by_domain",
            MappingProxyType(dict(sorted(energy.items()))),
        )
        object.__setattr__(
            self,
            "energy_estimation_metadata",
            MappingProxyType(dict(sorted(metadata.items()))),
        )
        object.__setattr__(
            self,
            "transfer_energy_uj_by_link",
            MappingProxyType(dict(sorted(transfer_energy.items()))),
        )

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.to_json(include_hash=False))

    def to_json(self, *, include_hash: bool = True) -> dict[str, object]:
        result = {
            "active_batch": self.active_batch,
            "attribution_kind": self.attribution_kind,
            "cohort_id": self.cohort_id,
            "common_policy_sha256": self.common_policy_sha256,
            "energy_boundary_id": self.energy_boundary_id,
            "energy_estimation_metadata": dict(
                self.energy_estimation_metadata
            ),
            "energy_owner_request_id": self.energy_owner_request_id,
            "finished_at_us": self.finished_at_us,
            "fleet_energy_uj_by_domain": dict(
                self.fleet_energy_uj_by_domain
            ),
            "measurement_evidence_ids": list(
                self.measurement_evidence_ids
            ),
            "member_request_ids": list(self.member_request_ids),
            "schema": RUNTIME_DECODE_COHORT_SCHEMA,
            "started_at_us": self.started_at_us,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "transfer_energy_uj_by_link": dict(
                self.transfer_energy_uj_by_link
            ),
        }
        if include_hash:
            result["receipt_sha256"] = self.receipt_sha256
        return result


@dataclass
class _Cohort:
    cohort_id: str
    key: RuntimeDecodeCohortKey
    leader_request_id: str
    members: list[str]
    maximum_members: int
    formation_deadline_ns: int
    shared_leases: tuple[LeaseRecord, ...] = ()
    sealed: bool = False
    terminal_members: set[str] | None = None
    capacity_released_members: set[str] | None = None
    output_tokens_by_request: dict[str, int] | None = None
    singleton_owner_id: str | None = None

    def __post_init__(self) -> None:
        if self.terminal_members is None:
            self.terminal_members = set()
        if self.capacity_released_members is None:
            self.capacity_released_members = set()
        if self.output_tokens_by_request is None:
            self.output_tokens_by_request = {}


@dataclass(frozen=True)
class _CohortSnapshot:
    cohort_id: str
    key: RuntimeDecodeCohortKey
    leader_request_id: str
    members: tuple[str, ...]
    maximum_members: int
    formation_deadline_ns: int
    shared_leases: tuple[LeaseRecord, ...]
    sealed: bool
    terminal_members: tuple[str, ...]
    capacity_released_members: tuple[str, ...]
    output_tokens_by_request: tuple[tuple[str, int], ...]
    singleton_owner_id: str | None


@dataclass(frozen=True)
class _CohortCheckpoint:
    cohorts: tuple[_CohortSnapshot, ...]
    request_to_cohort: tuple[tuple[str, str], ...]
    receipts: tuple[tuple[str, RuntimeDecodeCohortReceipt], ...]
    estimator_ingested: tuple[str, ...]
    next_cohort: int


class RuntimeDecodeCohortManager:
    """Own compatible request batching and one shared physical lease."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._cohorts: dict[str, _Cohort] = {}
        self._request_to_cohort: dict[str, str] = {}
        self._next_cohort = 1
        self._receipts: dict[str, RuntimeDecodeCohortReceipt] = {}
        self._estimator_ingested: set[str] = set()

    @staticmethod
    def candidate_key(
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        *,
        quality_requirement: str,
    ) -> RuntimeDecodeCohortKey | None:
        if (
            not isinstance(plan, RuntimeExecutionPlan)
            or not isinstance(binding, RuntimeExecutorBinding)
        ):
            return None
        parameters = plan.adapter_parameters
        geometry = parameters.get("ffn_resident_geometry_sha256")
        if (
            parameters.get("ffn_assistance_phase") != "decode"
            or parameters.get("ffn_runtime_control_protocol")
                != "decode-boundary-v1"
            or parameters.get("ffn_weight_buffer_layout")
                != "resident-superset"
            or type(geometry) is not str
            or not geometry
            or plan.desktop_placement_sha256 is None
        ):
            return None
        quality_requirement = _text(
            "decode cohort quality requirement", quality_requirement
        )
        policy = canonical_sha256({
            "adaptive_policy_generation": parameters.get(
                "adaptive_policy_generation", "coarse-to-fine-v1"
            ),
            "column_quantum": parameters.get("ffn_column_quantum"),
            "layer_mask": parameters.get("ffn_resident_layer_mask"),
            "maximum_columns": parameters.get("ffn_resident_columns"),
            "minimum_column_quantum": parameters.get(
                "ffn_minimum_column_quantum"
            ),
            "protocol": parameters.get("ffn_runtime_control_protocol"),
            "quality_requirement": quality_requirement,
            "resident_geometry_sha256": geometry,
            "runtime_partition_count": parameters.get(
                "ffn_runtime_partition_count"
            ),
            "schema": "decode-cohort-common-policy-v1",
        })
        if plan.transitions and RuntimeDecodeCohortManager._shareable_preparation(plan):
            policy = canonical_sha256({
                "common_policy_sha256": policy,
                "adapter_parameters": dict(parameters),
                "participants": [row.to_json() for row in binding.participants],
                "transitions": [row.to_json() for row in plan.transitions],
            })
        transport = canonical_sha256({
            name: parameters.get(name)
            for name in (
                "ffn_max_tokens",
                "usb_allocator",
                "usb_batch_plan",
                "usb_capacity_d2h_transport_profile_id",
                "usb_capacity_h2d_transport_profile_id",
                "usb_concurrent_streams",
                "usb_full_duplex",
                "usb_max_payload_bytes",
                "usb_queue_depth",
                "usb_transport_generation",
                "usb_transport_profile_id",
                "usb_transport_qualification_identity_sha256",
            )
        })
        return RuntimeDecodeCohortKey(
            artifact_sha256=binding.artifact_sha256,
            endpoint=binding.endpoint,
            desktop_placement_sha256=plan.desktop_placement_sha256,
            resident_geometry_sha256=geometry,
            transport_geometry_sha256=transport,
            common_policy_sha256=policy,
        )

    @staticmethod
    def capacity(plan: RuntimeExecutionPlan) -> int:
        parameters = plan.adapter_parameters
        if parameters.get("usb_batch_plan") == "coalesced-batch":
            if plan.execution_contract.batch_plan != "coalesced-batch":
                return 1
            values = (parameters.get("parallel"), parameters.get("ffn_max_tokens"),
                      plan.execution_contract.maximum_batch_size)
            if any(type(value) is not int or value < 2 for value in values):
                return 1
            return min(4, *values)
        values = tuple(parameters.get(name) for name in (
            "parallel",
            "usb_concurrent_streams",
            "usb_queue_depth",
        ))
        if any(type(value) is not int or value < 2 for value in values):
            return 1
        return min(4, *values)

    @staticmethod
    def _shareable_preparation(plan: RuntimeExecutionPlan) -> bool:
        phone_id = plan.execution_contract.phone_device_id
        return all(
            row.device_id == phone_id
            and row.prepares_device_ids == (phone_id,)
            and row.target_state == "hot"
            for row in plan.transitions
        )

    @staticmethod
    def formation_us(
        plan: RuntimeExecutionPlan, service_upper_us: int
    ) -> int:
        _integer(
            "decode cohort service upper", service_upper_us, 1
        )
        value = plan.adapter_parameters.get("decode_cohort_formation_us")
        if value is None:
            value = min(
                1_000_000,
                max(50_000, service_upper_us // 100),
            )
        if type(value) is not int or value < 0:
            raise RuntimeDecodeCohortError(
                "decode cohort formation interval is invalid"
            )
        return value

    def can_join(
        self,
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        *,
        quality_requirement: str,
        now_ns: int | None = None,
    ) -> bool:
        key = self.candidate_key(
            plan,
            binding,
            quality_requirement=quality_requirement,
        )
        if key is None or self.capacity(plan) < 2 or not self._shareable_preparation(plan):
            return False
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        _integer("decode cohort clock", now_ns)
        resources = set(plan.resource_ids)
        with self._condition:
            return any(
                row.key == key
                and not row.sealed
                and bool(row.shared_leases)
                and len(row.members) < row.maximum_members
                and now_ns <= row.formation_deadline_ns
                and {
                    lease.resource_id for lease in row.shared_leases
                } == resources
                for row in self._cohorts.values()
            )

    def checkpoint(self) -> object:
        with self._condition:
            return _CohortCheckpoint(
                cohorts=tuple(
                    _CohortSnapshot(
                        cohort_id=row.cohort_id,
                        key=row.key,
                        leader_request_id=row.leader_request_id,
                        members=tuple(row.members),
                        maximum_members=row.maximum_members,
                        formation_deadline_ns=row.formation_deadline_ns,
                        shared_leases=row.shared_leases,
                        sealed=row.sealed,
                        terminal_members=tuple(sorted(row.terminal_members)),
                        capacity_released_members=tuple(sorted(
                            row.capacity_released_members
                        )),
                        output_tokens_by_request=tuple(sorted(
                            row.output_tokens_by_request.items()
                        )),
                        singleton_owner_id=row.singleton_owner_id,
                    )
                    for row in sorted(
                        self._cohorts.values(),
                        key=lambda value: value.cohort_id,
                    )
                ),
                request_to_cohort=tuple(sorted(
                    self._request_to_cohort.items()
                )),
                receipts=tuple(sorted(self._receipts.items())),
                estimator_ingested=tuple(sorted(
                    self._estimator_ingested
                )),
                next_cohort=self._next_cohort,
            )

    def restore(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _CohortCheckpoint):
            raise RuntimeDecodeCohortError(
                "decode cohort checkpoint is invalid"
            )
        cohorts = {
            row.cohort_id: _Cohort(
                cohort_id=row.cohort_id,
                key=row.key,
                leader_request_id=row.leader_request_id,
                members=list(row.members),
                maximum_members=row.maximum_members,
                formation_deadline_ns=row.formation_deadline_ns,
                shared_leases=row.shared_leases,
                sealed=row.sealed,
                terminal_members=set(row.terminal_members),
                capacity_released_members=set(
                    row.capacity_released_members
                ),
                output_tokens_by_request=dict(
                    row.output_tokens_by_request
                ),
                singleton_owner_id=row.singleton_owner_id,
            )
            for row in checkpoint.cohorts
        }
        with self._condition:
            self._cohorts = cohorts
            self._request_to_cohort = dict(checkpoint.request_to_cohort)
            self._receipts = dict(checkpoint.receipts)
            self._estimator_ingested = set(
                checkpoint.estimator_ingested
            )
            self._next_cohort = checkpoint.next_cohort
            self._condition.notify_all()

    def _binding(self, cohort: _Cohort) -> RuntimeDecodeCohortBinding:
        return RuntimeDecodeCohortBinding(
            cohort_id=cohort.cohort_id,
            key_sha256=cohort.key.key_sha256,
            leader_request_id=cohort.leader_request_id,
            member_request_ids=tuple(cohort.members),
            common_policy_sha256=cohort.key.common_policy_sha256,
            shared_lease_tokens=tuple(
                row.token for row in cohort.shared_leases
            ),
            active_batch=len(cohort.members),
            maximum_members=cohort.maximum_members,
            sealed=cohort.sealed,
        )

    @staticmethod
    def _seal(cohort: _Cohort) -> None:
        if cohort.sealed:
            return
        formation_order = {
            request_id: index
            for index, request_id in enumerate(cohort.members)
        }
        if all(
            cohort.output_tokens_by_request.get(request_id, 0) > 0
            for request_id in cohort.members
        ):
            cohort.members.sort(key=lambda request_id: (
                -cohort.output_tokens_by_request[request_id],
                formation_order[request_id],
            ))
            cohort.leader_request_id = cohort.members[0]
        cohort.sealed = True

    def admit(
        self,
        request_id: str,
        plan: RuntimeExecutionPlan,
        binding: RuntimeExecutorBinding,
        *,
        quality_requirement: str,
        observed_at_us: int,
        service_upper_us: int,
        output_tokens: int | None = None,
        now_ns: int | None = None,
    ) -> RuntimeDecodeCohortAdmission | None:
        request_id = _text("decode cohort request", request_id)
        key = self.candidate_key(
            plan,
            binding,
            quality_requirement=quality_requirement,
        )
        maximum_members = self.capacity(plan)
        if key is None or maximum_members < 2 or not self._shareable_preparation(plan):
            return None
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        _integer("decode cohort clock", now_ns)
        _integer("decode cohort observation", observed_at_us)
        _integer("decode cohort service upper", service_upper_us, 1)
        if output_tokens is not None:
            _integer("decode cohort output tokens", output_tokens, 1)
        with self._condition:
            if request_id in self._request_to_cohort:
                raise RuntimeDecodeCohortError(
                    "request already belongs to a decode cohort"
                )
            matches = tuple(
                row for row in self._cohorts.values()
                if row.key == key
                and not row.sealed
                and row.shared_leases
                and len(row.members) < row.maximum_members
                and now_ns <= row.formation_deadline_ns
                and {
                    lease.resource_id for lease in row.shared_leases
                } == set(plan.resource_ids)
            )
            if matches:
                cohort = min(matches, key=lambda row: row.cohort_id)
                cohort.members.append(request_id)
                if output_tokens is not None:
                    cohort.output_tokens_by_request[request_id] = output_tokens
                if len(cohort.members) == cohort.maximum_members:
                    self._seal(cohort)
                self._request_to_cohort[request_id] = cohort.cohort_id
                required_reserved_until_us = max(
                    max(row.reserved_until_us for row in cohort.shared_leases),
                    max(
                        observed_at_us,
                        min(
                            value.start_us for value in cohort.shared_leases
                        ),
                    ) + service_upper_us,
                )
                self._condition.notify_all()
                return RuntimeDecodeCohortAdmission(
                    binding=self._binding(cohort),
                    shared_leases=cohort.shared_leases,
                    leader=False,
                    required_reserved_until_us=(
                        required_reserved_until_us
                    ),
                )
            cohort_id = "decode-cohort-" + str(self._next_cohort)
            self._next_cohort += 1
            formation_us = self.formation_us(plan, service_upper_us)
            cohort = _Cohort(
                cohort_id=cohort_id,
                key=key,
                leader_request_id=request_id,
                members=[request_id],
                maximum_members=maximum_members,
                formation_deadline_ns=now_ns + formation_us * 1000,
                sealed=formation_us == 0,
                output_tokens_by_request=(
                    {} if output_tokens is None
                    else {request_id: output_tokens}
                ),
            )
            self._cohorts[cohort_id] = cohort
            self._request_to_cohort[request_id] = cohort_id
            return RuntimeDecodeCohortAdmission(
                binding=self._binding(cohort),
                shared_leases=(),
                leader=True,
                required_reserved_until_us=(
                    observed_at_us + service_upper_us
                ),
            )

    def bind_leases(
        self, cohort_id: str, leases: Sequence[LeaseRecord]
    ) -> RuntimeDecodeCohortBinding:
        cohort_id = _text("decode cohort id", cohort_id)
        rows = tuple(leases)
        if not rows or any(not isinstance(row, LeaseRecord) for row in rows):
            raise RuntimeDecodeCohortError(
                "decode cohort shared leases are invalid"
            )
        if any(row.owner_id != cohort_id for row in rows):
            raise RuntimeDecodeCohortError(
                "decode cohort lease owner differs"
            )
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            if cohort is None or cohort.shared_leases:
                raise RuntimeDecodeCohortError(
                    "decode cohort lease binding state is invalid"
                )
            cohort.shared_leases = rows
            self._condition.notify_all()
            return self._binding(cohort)

    def update_shared_leases(
        self,
        cohort_id: str,
        leases: Sequence[LeaseRecord],
    ) -> RuntimeDecodeCohortBinding:
        """Replace one cohort lease set after an atomic calendar update."""
        cohort_id = _text("decode cohort id", cohort_id)
        rows = tuple(leases)
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            if cohort is None or not cohort.shared_leases:
                raise RuntimeDecodeCohortError(
                    "decode cohort shared lease state is invalid"
                )
            previous = cohort.shared_leases
            expected_owner_id = (
                cohort.cohort_id
                if cohort.singleton_owner_id is None
                else cohort.singleton_owner_id
            )
            if (
                len(rows) != len(previous)
                or any(not isinstance(row, LeaseRecord) for row in rows)
                or tuple(row.token for row in rows)
                    != tuple(row.token for row in previous)
                or any(
                    row.owner_id != expected_owner_id
                    or old.owner_id != expected_owner_id
                    or row.resource_id != old.resource_id
                    or row.lanes != old.lanes
                    or row.start_us != old.start_us
                    or row.predicted_end_us != old.predicted_end_us
                    or row.reserved_until_us < old.reserved_until_us
                    for row, old in zip(rows, previous)
                )
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort shared lease update differs"
                )
            cohort.shared_leases = rows
            self._condition.notify_all()
            return self._binding(cohort)

    def lease_owner_id(self, cohort_id: str) -> str:
        cohort_id = _text("decode cohort id", cohort_id)
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            if cohort is None:
                raise RuntimeDecodeCohortError(
                    "decode cohort is absent"
                )
            return cohort.singleton_owner_id or cohort.cohort_id

    def pending_singleton_handoff(
        self, cohort_id: str
    ) -> str | None:
        cohort_id = _text("decode cohort id", cohort_id)
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            if cohort is None or cohort.singleton_owner_id is not None:
                return None
            active = tuple(
                request_id for request_id in cohort.members
                if request_id not in cohort.terminal_members
            )
            return active[0] if len(active) == 1 else None

    def transfer_singleton_ownership(
        self, cohort_id: str, request_id: str
    ) -> tuple[LeaseRecord, ...]:
        """Commit an in-flight cohort's shared leases to its survivor."""
        cohort_id = _text("decode cohort id", cohort_id)
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            active = () if cohort is None else tuple(
                value for value in cohort.members
                if value not in cohort.terminal_members
            )
            if (
                cohort is None
                or cohort.singleton_owner_id is not None
                or active != (request_id,)
                or self._request_to_cohort.get(request_id) != cohort_id
                or not cohort.shared_leases
                or any(
                    row.owner_id != cohort_id
                    for row in cohort.shared_leases
                )
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort singleton handoff state differs"
                )
            cohort.shared_leases = tuple(
                replace(row, owner_id=request_id)
                for row in cohort.shared_leases
            )
            cohort.singleton_owner_id = request_id
            self._condition.notify_all()
            return cohort.shared_leases

    def withdraw(self, request_id: str) -> None:
        """Undo one unsealed follower admission before ticket creation."""
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            cohort = None if cohort_id is None else self._cohorts.get(cohort_id)
            if (
                cohort is None
                or cohort.leader_request_id == request_id
                or request_id not in cohort.members
                or request_id in cohort.terminal_members
                or request_id in cohort.capacity_released_members
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort follower withdrawal is invalid"
                )
            cohort.members.remove(request_id)
            cohort.output_tokens_by_request.pop(request_id, None)
            cohort.sealed = False
            del self._request_to_cohort[request_id]
            self._condition.notify_all()

    def prepare_replan(
        self, request_id: str
    ) -> RuntimeDecodeCohortReplanRelease | None:
        """Detach one queued member before replacing its ticket."""
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            if cohort_id is None:
                return None
            cohort = self._cohorts.get(cohort_id)
            if (
                cohort is None
                or request_id not in cohort.members
                or request_id in cohort.terminal_members
                or request_id in cohort.capacity_released_members
                or cohort.terminal_members
                or cohort.capacity_released_members
                or not cohort.shared_leases
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort replan state is invalid"
                )
            previous = self._binding(cohort)
            previous_owner_id = (
                cohort.singleton_owner_id or cohort.cohort_id
            )
            if len(cohort.members) == 1:
                transferred = tuple(
                    replace(row, owner_id=request_id)
                    for row in cohort.shared_leases
                )
                del self._request_to_cohort[request_id]
                del self._cohorts[cohort_id]
                remaining = None
            else:
                cohort.members.remove(request_id)
                cohort.output_tokens_by_request.pop(request_id, None)
                if cohort.leader_request_id == request_id:
                    cohort.leader_request_id = cohort.members[0]
                del self._request_to_cohort[request_id]
                transferred = ()
                remaining = self._binding(cohort)
            self._condition.notify_all()
            return RuntimeDecodeCohortReplanRelease(
                previous_binding=previous,
                remaining_binding=remaining,
                previous_lease_owner_id=previous_owner_id,
                transferred_leases=transferred,
            )

    def dissolve_singleton(
        self, request_id: str
    ) -> tuple[LeaseRecord, ...]:
        """Remove a sealed prospective cohort that found no follower."""
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            cohort = None if cohort_id is None else self._cohorts.get(cohort_id)
            if (
                cohort is None
                or not cohort.sealed
                or cohort.members != [request_id]
                or cohort.leader_request_id != request_id
                or not cohort.shared_leases
                or cohort.terminal_members
                or cohort.capacity_released_members
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort singleton dissolution is invalid"
                )
            leases = tuple(
                replace(row, owner_id=request_id)
                for row in cohort.shared_leases
            )
            del self._request_to_cohort[request_id]
            del self._cohorts[cohort_id]
            self._condition.notify_all()
            return leases

    def binding(self, request_id: str) -> RuntimeDecodeCohortBinding | None:
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            if cohort_id is None:
                return None
            return self._binding(self._cohorts[cohort_id])

    def binding_by_id(
        self, cohort_id: str
    ) -> RuntimeDecodeCohortBinding:
        cohort_id = _text("decode cohort id", cohort_id)
        with self._condition:
            cohort = self._cohorts.get(cohort_id)
            if cohort is None:
                raise RuntimeDecodeCohortError(
                    "decode cohort is absent"
                )
            return self._binding(cohort)

    def wait_until_sealed(
        self, request_id: str
    ) -> RuntimeDecodeCohortBinding | None:
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            if cohort_id is None:
                return None
            while True:
                cohort = self._cohorts[cohort_id]
                if cohort.sealed:
                    return self._binding(cohort)
                remaining_ns = cohort.formation_deadline_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    self._seal(cohort)
                    self._condition.notify_all()
                    return self._binding(cohort)
                self._condition.wait(remaining_ns / 1_000_000_000)

    def mark_terminal(self, request_id: str) -> tuple[bool, str | None]:
        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            if cohort_id is None:
                return True, None
            cohort = self._cohorts[cohort_id]
            if request_id in cohort.terminal_members:
                raise RuntimeDecodeCohortError(
                    "decode cohort member is already terminal"
                )
            cohort.terminal_members.add(request_id)
            del self._request_to_cohort[request_id]
            last = len(cohort.terminal_members) == len(cohort.members)
            self._condition.notify_all()
            return last, cohort_id

    def mark_capacity_released(
        self, request_id: str
    ) -> tuple[bool, str | None]:
        """Record physical completion without consuming terminal state."""

        request_id = _text("decode cohort request", request_id)
        with self._condition:
            cohort_id = self._request_to_cohort.get(request_id)
            if cohort_id is None:
                return True, None
            cohort = self._cohorts[cohort_id]
            if request_id in cohort.capacity_released_members:
                raise RuntimeDecodeCohortError(
                    "decode cohort member capacity is already released"
                )
            cohort.capacity_released_members.add(request_id)
            last = (
                len(cohort.capacity_released_members)
                == len(cohort.members)
            )
            self._condition.notify_all()
            return last, cohort_id

    def record_receipt(
        self, receipt: RuntimeDecodeCohortReceipt
    ) -> None:
        if not isinstance(receipt, RuntimeDecodeCohortReceipt):
            raise RuntimeDecodeCohortError(
                "decode cohort execution receipt is invalid"
            )
        with self._condition:
            cohort = self._cohorts.get(receipt.cohort_id)
            if (
                cohort is None
                or receipt.cohort_id in self._receipts
                or tuple(cohort.members) != receipt.member_request_ids
                or cohort.leader_request_id
                    != receipt.energy_owner_request_id
                or cohort.key.common_policy_sha256
                    != receipt.common_policy_sha256
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort execution receipt differs"
                )
            self._receipts[receipt.cohort_id] = receipt

    def mark_estimator_ingested(self, cohort_id: str) -> None:
        cohort_id = _text("decode cohort id", cohort_id)
        with self._condition:
            if (
                cohort_id not in self._receipts
                or cohort_id in self._estimator_ingested
            ):
                raise RuntimeDecodeCohortError(
                    "decode cohort estimator ingestion differs"
                )
            self._estimator_ingested.add(cohort_id)

    def receipt(
        self, cohort_id: str
    ) -> RuntimeDecodeCohortReceipt | None:
        cohort_id = _text("decode cohort id", cohort_id)
        with self._condition:
            return self._receipts.get(cohort_id)

    def snapshot(self) -> Mapping[str, object]:
        with self._condition:
            return MappingProxyType({
                "cohorts": {
                    cohort_id: {
                        **self._binding(row).to_json(),
                        "active_member_request_ids": [
                            request_id for request_id in row.members
                            if request_id not in row.terminal_members
                        ],
                        "capacity_released_member_request_ids": sorted(
                            row.capacity_released_members
                        ),
                        "lease_owner_id": (
                            row.singleton_owner_id or row.cohort_id
                        ),
                    }
                    for cohort_id, row in sorted(self._cohorts.items())
                },
                "receipts": {
                    cohort_id: row.to_json()
                    for cohort_id, row in sorted(self._receipts.items())
                },
                "estimator_ingested": sorted(self._estimator_ingested),
                "schema": RUNTIME_DECODE_COHORT_SCHEMA,
            })
