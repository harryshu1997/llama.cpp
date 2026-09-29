"""Immutable contracts for token-boundary decode adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from .types import canonical_sha256


ADAPTIVE_DECODE_SCHEMA = "research-scheduler-adaptive-decode-v1"
ADAPTIVE_DECODE_STATES = frozenset({
    "BASELINE",
    "PREPARING",
    "PROBING",
    "EXPLOITING",
    "RECOVERING",
    "COMPLETED",
})


class AdaptiveDecodeError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise AdaptiveDecodeError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AdaptiveDecodeError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(name: str, value: object) -> str:
    result = _text(name, value)
    if (
        not result.startswith("sha256:")
        or len(result) != 71
        or any(character not in "0123456789abcdef" for character in result[7:])
    ):
        raise AdaptiveDecodeError(f"{name} must be SHA-256")
    return result


def _integer_mapping(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    result = {
        _text(f"{name} key", key): _integer(f"{name} value", value)
        for key, value in values.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True)
class AdaptiveDecodeConfig:
    minimum_remaining_tokens: int = 24
    minimum_window_tokens: int = 4
    maximum_window_tokens: int = 24
    maximum_probe_tokens: int = 80
    maximum_probe_candidates: int = 4
    maximum_probe_attempts_per_context: int = 2
    measurement_resolution_us: int = 200_000
    transition_cost_us: int = 2_000
    transition_energy_uj: int = 20_000
    minimum_energy_saving_ppm: int = 50_000
    maximum_latency_ppm: int = 1_000_000
    uncertainty_ppm: int = 100_000
    exploration_latency_budget_ppm: int = 150_000
    exploration_energy_budget_ppm: int = 150_000
    warmup_windows_per_policy: int = 1
    allow_assumed_phone_power_for_operational_selection: bool = False
    server_policy_coherence: bool = False
    coarse_probe_fractions_ppm: tuple[int, ...] = (
        1_000_000,
        750_000,
        500_000,
        250_000,
    )
    refinement_steps_ppm: tuple[int, ...] = (125_000, 62_500)

    def __post_init__(self) -> None:
        if type(self.server_policy_coherence) is not bool:
            raise AdaptiveDecodeError("adaptive server policy coherence is invalid")
        for name in (
            "minimum_remaining_tokens",
            "minimum_window_tokens",
            "maximum_window_tokens",
            "maximum_probe_tokens",
            "maximum_probe_candidates",
            "maximum_probe_attempts_per_context",
            "measurement_resolution_us",
            "transition_cost_us",
            "transition_energy_uj",
        ):
            _integer("adaptive config " + name, getattr(self, name), 1)
        _integer(
            "adaptive config warmup_windows_per_policy",
            self.warmup_windows_per_policy,
        )
        if type(
            self.allow_assumed_phone_power_for_operational_selection
        ) is not bool:
            raise AdaptiveDecodeError(
                "adaptive assumed phone power policy is invalid"
            )
        for name in (
            "minimum_energy_saving_ppm",
            "uncertainty_ppm",
            "exploration_latency_budget_ppm",
            "exploration_energy_budget_ppm",
        ):
            value = _integer("adaptive config " + name, getattr(self, name))
            if value > 1_000_000:
                raise AdaptiveDecodeError(
                    "adaptive configuration ratio exceeds one"
                )
        maximum_latency = _integer(
            "adaptive config maximum_latency_ppm",
            self.maximum_latency_ppm,
            1,
        )
        if maximum_latency > 10_000_000:
            raise AdaptiveDecodeError(
                "adaptive latency multiplier is too large"
            )
        if (
            self.minimum_window_tokens > self.maximum_window_tokens
            or self.maximum_window_tokens > self.maximum_probe_tokens
        ):
            raise AdaptiveDecodeError("adaptive window limits are invalid")
        coarse = tuple(self.coarse_probe_fractions_ppm)
        refinement = tuple(self.refinement_steps_ppm)
        if (
            not coarse
            or len(coarse) != len(set(coarse))
            or any(
                type(value) is not int
                or not 0 < value <= 1_000_000
                for value in coarse
            )
        ):
            raise AdaptiveDecodeError(
                "adaptive coarse probe fractions are invalid"
            )
        if (
            not refinement
            or len(refinement) != len(set(refinement))
            or any(
                type(value) is not int
                or not 0 < value < 1_000_000
                for value in refinement
            )
            or tuple(sorted(refinement, reverse=True)) != refinement
        ):
            raise AdaptiveDecodeError(
                "adaptive refinement steps are invalid"
            )
        object.__setattr__(self, "coarse_probe_fractions_ppm", coarse)
        object.__setattr__(self, "refinement_steps_ppm", refinement)


@dataclass(frozen=True)
class AdaptiveDecodePolicy:
    route_id: str
    executor_id: str
    operator_plan_sha256: str
    desktop_parent_route_id: str
    desktop_placement_sha256: str
    layer_indices: tuple[int, ...]
    layer_mask: int
    columns: int
    split_fraction_ppm: int
    resource_ids: tuple[str, ...]
    baseline: bool = False
    predicted_latency_per_token_us: int | None = None
    predicted_energy_per_token_uj: int | None = None
    policy_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("route_id", "executor_id", "desktop_parent_route_id"):
            _text("adaptive policy " + name, getattr(self, name))
        _sha256("adaptive policy operator plan", self.operator_plan_sha256)
        _sha256(
            "adaptive policy desktop placement", self.desktop_placement_sha256
        )
        layers = tuple(self.layer_indices)
        if (
            len(layers) != len(set(layers))
            or tuple(sorted(layers)) != layers
            or any(type(value) is not int or not 0 <= value < 64 for value in layers)
        ):
            raise AdaptiveDecodeError("adaptive policy layers are invalid")
        resources = tuple(sorted(
            _text("adaptive policy resource", value)
            for value in self.resource_ids
        ))
        if not resources or len(resources) != len(set(resources)):
            raise AdaptiveDecodeError("adaptive policy resources are invalid")
        _integer("adaptive policy layer mask", self.layer_mask)
        _integer("adaptive policy columns", self.columns)
        _integer("adaptive policy split fraction", self.split_fraction_ppm)
        if self.split_fraction_ppm > 1_000_000:
            raise AdaptiveDecodeError("adaptive split fraction exceeds one")
        if type(self.baseline) is not bool:
            raise AdaptiveDecodeError("adaptive baseline flag is invalid")
        if self.baseline:
            if layers or self.layer_mask or self.columns or self.split_fraction_ppm:
                raise AdaptiveDecodeError("adaptive baseline contains assistance")
            if self.route_id != self.desktop_parent_route_id:
                raise AdaptiveDecodeError("adaptive baseline parent differs")
        elif (
            not layers
            or self.layer_mask != sum(1 << value for value in layers)
            or self.columns <= 0
            or not 0 < self.split_fraction_ppm <= 1_000_000
        ):
            raise AdaptiveDecodeError("adaptive assistance geometry is invalid")
        for name in (
            "predicted_latency_per_token_us",
            "predicted_energy_per_token_uj",
        ):
            value = getattr(self, name)
            if value is not None:
                _integer("adaptive policy " + name, value, 1)
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "resource_ids", resources)
        payload = json.dumps(
            self._json_without_hash(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        object.__setattr__(
            self,
            "policy_hash",
            "sha256:" + hashlib.sha256(payload).hexdigest(),
        )

    def _json_without_hash(self) -> dict[str, object]:
        return {
            "baseline": self.baseline,
            "columns": self.columns,
            "desktop_parent_route_id": self.desktop_parent_route_id,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "executor_id": self.executor_id,
            "layer_indices": list(self.layer_indices),
            "layer_mask": self.layer_mask,
            "operator_plan_sha256": self.operator_plan_sha256,
            "predicted_energy_per_token_uj": self.predicted_energy_per_token_uj,
            "predicted_latency_per_token_us": self.predicted_latency_per_token_us,
            "resource_ids": list(self.resource_ids),
            "route_id": self.route_id,
            "schema": ADAPTIVE_DECODE_SCHEMA,
            "split_fraction_ppm": self.split_fraction_ppm,
        }

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["policy_hash"] = self.policy_hash
        return result

    @classmethod
    def from_json(cls, value: object) -> "AdaptiveDecodePolicy":
        if type(value) is not dict:
            raise AdaptiveDecodeError("adaptive policy JSON is invalid")
        row = dict(value)
        if row.get("schema") != ADAPTIVE_DECODE_SCHEMA:
            raise AdaptiveDecodeError("adaptive policy schema differs")
        result = cls(
            route_id=row.get("route_id"),
            executor_id=row.get("executor_id"),
            operator_plan_sha256=row.get("operator_plan_sha256"),
            desktop_parent_route_id=row.get("desktop_parent_route_id"),
            desktop_placement_sha256=row.get("desktop_placement_sha256"),
            layer_indices=tuple(row.get("layer_indices", ())),
            layer_mask=row.get("layer_mask"),
            columns=row.get("columns"),
            split_fraction_ppm=row.get("split_fraction_ppm"),
            resource_ids=tuple(row.get("resource_ids", ())),
            baseline=row.get("baseline"),
            predicted_latency_per_token_us=row.get(
                "predicted_latency_per_token_us"
            ),
            predicted_energy_per_token_uj=row.get(
                "predicted_energy_per_token_uj"
            ),
        )
        if row.get("policy_hash") != result.policy_hash:
            raise AdaptiveDecodeError("adaptive policy hash differs")
        return result


@dataclass(frozen=True)
class AdaptiveDecodeControl:
    request_id: str
    slot_id: int
    plan_generation: int
    policy: AdaptiveDecodePolicy

    def __post_init__(self) -> None:
        _text("adaptive control request", self.request_id)
        _integer("adaptive control slot", self.slot_id)
        _integer("adaptive control generation", self.plan_generation, 1)
        if not isinstance(self.policy, AdaptiveDecodePolicy):
            raise AdaptiveDecodeError("adaptive control policy is invalid")

    def to_server_json(self) -> dict[str, object]:
        return {
            "action": "ffn_split",
            "columns": self.policy.columns,
            "enabled": not self.policy.baseline,
            "layer_mask": self.policy.layer_mask,
            "plan_generation": self.plan_generation,
            "policy_hash": self.policy.policy_hash,
            "request_id": self.request_id,
            "slot_id": self.slot_id,
        }

    def to_json(self) -> dict[str, object]:
        return {
            **self.to_server_json(),
            "policy": self.policy.to_json(),
        }


@dataclass(frozen=True)
class AdaptiveDecodePolicyAck:
    request_id: str
    slot_id: int
    plan_generation: int
    applied_token_index: int
    applied_at_us: int
    policy_hash: str

    def __post_init__(self) -> None:
        _text("adaptive acknowledgement request", self.request_id)
        _integer("adaptive acknowledgement slot", self.slot_id)
        _integer("adaptive acknowledgement generation", self.plan_generation, 1)
        _integer("adaptive acknowledgement token", self.applied_token_index)
        _integer("adaptive acknowledgement time", self.applied_at_us)
        _sha256("adaptive acknowledgement policy", self.policy_hash)

    def to_json(self) -> dict[str, object]:
        return {
            "applied_at_us": self.applied_at_us,
            "applied_token_index": self.applied_token_index,
            "plan_generation": self.plan_generation,
            "policy_hash": self.policy_hash,
            "request_id": self.request_id,
            "slot_id": self.slot_id,
        }

    @classmethod
    def from_json(cls, value: object) -> "AdaptiveDecodePolicyAck":
        if type(value) is not dict:
            raise AdaptiveDecodeError("adaptive acknowledgement JSON is invalid")
        return cls(
            request_id=value.get("request_id"),
            slot_id=value.get("slot_id"),
            plan_generation=value.get("plan_generation"),
            applied_token_index=value.get("applied_token_index"),
            applied_at_us=value.get("applied_at_us"),
            policy_hash=value.get("policy_hash"),
        )


@dataclass(frozen=True)
class AdaptiveDecodeWindowBoundary:
    request_id: str
    slot_id: int
    window_index: int
    token_start: int
    token_end: int
    started_at_us: int
    finished_at_us: int
    policy: AdaptiveDecodePolicy
    applied_ack: AdaptiveDecodePolicyAck | None
    window_role: str = "legacy_unspecified"

    def __post_init__(self) -> None:
        _text("adaptive boundary request", self.request_id)
        _integer("adaptive boundary slot", self.slot_id)
        _integer("adaptive boundary index", self.window_index)
        _integer("adaptive boundary token start", self.token_start)
        _integer("adaptive boundary token end", self.token_end, 1)
        _integer("adaptive boundary start", self.started_at_us)
        _integer("adaptive boundary finish", self.finished_at_us, 1)
        if self.window_role not in {"legacy_unspecified", "exploration", "exploitation"}:
            raise AdaptiveDecodeError("adaptive boundary window role is invalid")
        if (
            self.token_end <= self.token_start
            or self.finished_at_us <= self.started_at_us
            or not isinstance(self.policy, AdaptiveDecodePolicy)
        ):
            raise AdaptiveDecodeError("adaptive boundary interval is invalid")
        if self.applied_ack is not None and (
            not isinstance(self.applied_ack, AdaptiveDecodePolicyAck)
            or self.applied_ack.request_id != self.request_id
            or self.applied_ack.slot_id != self.slot_id
            or self.applied_ack.applied_token_index != self.token_start
            or self.applied_ack.policy_hash != self.policy.policy_hash
        ):
            raise AdaptiveDecodeError("adaptive boundary acknowledgement differs")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class AdaptiveDecodeRawWindowObservation:
    fleet_energy_uj_by_domain: Mapping[str, int]
    phone_compute_us: int
    usb_transfer_us: int
    rpc_us: int
    exposed_tail_us: int
    output_valid: bool
    evidence_ids: tuple[str, ...]
    energy_boundary_id: str
    energy_attribution_kind: str
    failure_reason: str | None = None
    usb_upload_bytes: int = 0
    usb_download_bytes: int = 0
    desktop_compute_us: int = 0
    useful_overlap_us: int = 0
    request_queue_delay_us: int = 0
    protected_interference_us: int = 0
    active_batch: int | None = None
    next_active_batch: int | None = None
    membership_changed: bool = False
    execution_context_available: bool = True
    usb_h2d_us: int = 0
    usb_d2h_us: int = 0
    accounting_token_count: int | None = None
    cohort_id: str | None = None
    cohort_member_request_ids: tuple[str, ...] = ()
    energy_owner_request_id: str | None = None
    configured_queue_depth: int = 0
    maximum_active_slots: int = 0
    maximum_outstanding_transfers: int = 0
    batched_calls: int = 0
    transfer_subrequests: int = 0
    maximum_tokens: int = 0
    completed_phone_calls: int | None = None
    completed_phone_input_rows: int | None = None
    external_activity_sha256: str | None = None

    def __post_init__(self) -> None:
        energy = _integer_mapping(
            "adaptive raw window energy", self.fleet_energy_uj_by_domain
        )
        if not energy:
            raise AdaptiveDecodeError("adaptive raw window energy is empty")
        for name in (
            "phone_compute_us",
            "usb_transfer_us",
            "rpc_us",
            "exposed_tail_us",
            "usb_upload_bytes",
            "usb_download_bytes",
            "desktop_compute_us",
            "useful_overlap_us",
            "request_queue_delay_us",
            "protected_interference_us",
            "usb_h2d_us",
            "usb_d2h_us",
            "configured_queue_depth",
            "maximum_active_slots",
            "maximum_outstanding_transfers",
            "batched_calls",
            "transfer_subrequests",
            "maximum_tokens",
        ):
            _integer("adaptive raw window " + name, getattr(self, name))
        if (
            self.completed_phone_calls is None
            or self.completed_phone_input_rows is None
        ):
            if not (
                self.completed_phone_calls is None
                and self.completed_phone_input_rows is None
            ):
                raise AdaptiveDecodeError(
                    "adaptive completed phone counters are incomplete"
                )
        else:
            calls = _integer(
                "adaptive raw window completed phone calls",
                self.completed_phone_calls,
            )
            rows = _integer(
                "adaptive raw window completed phone input rows",
                self.completed_phone_input_rows,
            )
            if (calls == 0) != (rows == 0) or rows < calls:
                raise AdaptiveDecodeError(
                    "adaptive completed phone counters are invalid"
                )
        if self.active_batch is not None:
            _integer("adaptive raw window active batch", self.active_batch, 1)
        if self.next_active_batch is not None:
            _integer(
                "adaptive raw window next active batch",
                self.next_active_batch,
                1,
            )
        if type(self.membership_changed) is not bool:
            raise AdaptiveDecodeError("adaptive membership change is invalid")
        if type(self.execution_context_available) is not bool:
            raise AdaptiveDecodeError("adaptive execution context availability is invalid")
        if not self.execution_context_available and (
            self.membership_changed or self.next_active_batch is not None
        ):
            raise AdaptiveDecodeError("unavailable adaptive context cannot assert membership")
        if self.membership_changed and self.next_active_batch is None:
            raise AdaptiveDecodeError("adaptive membership change lacks live batch")
        if self.accounting_token_count is not None:
            _integer(
                "adaptive raw window accounting token count",
                self.accounting_token_count,
                1,
            )
        if type(self.output_valid) is not bool:
            raise AdaptiveDecodeError("adaptive output validation is invalid")
        _text("adaptive raw energy boundary", self.energy_boundary_id)
        if self.energy_attribution_kind not in {
            "diagnostic", "isolated", "matched_abba", "device_domain"
        }:
            raise AdaptiveDecodeError(
                "adaptive raw energy attribution is invalid"
            )
        evidence = tuple(sorted(
            _text("adaptive raw evidence", value) for value in self.evidence_ids
        ))
        if not evidence or len(evidence) != len(set(evidence)):
            raise AdaptiveDecodeError("adaptive raw evidence is invalid")
        if self.failure_reason is not None:
            _text("adaptive failure reason", self.failure_reason)
        if self.failure_reason is None and not self.output_valid:
            raise AdaptiveDecodeError("invalid output lacks a failure reason")
        if self.external_activity_sha256 is not None:
            _sha256("adaptive external activity identity", self.external_activity_sha256)
        members = tuple(self.cohort_member_request_ids)
        if self.cohort_id is None:
            if (
                members
                or self.energy_owner_request_id is not None
            ):
                raise AdaptiveDecodeError(
                    "adaptive non-cohort observation has cohort ownership"
                )
        else:
            _text("adaptive raw window cohort", self.cohort_id)
            if (
                len(members) < 2
                or len(members) != len(set(members))
                or any(
                    type(value) is not str
                    or not value
                    or not value.isascii()
                    for value in members
                )
                or self.energy_owner_request_id != members[0]
                or self.accounting_token_count is None
                or self.active_batch != len(members)
                or (
                    self.next_active_batch is not None
                    and not 1 <= self.next_active_batch < len(members)
                )
            ):
                raise AdaptiveDecodeError(
                    "adaptive cohort observation identity differs"
                )
        object.__setattr__(self, "fleet_energy_uj_by_domain", energy)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "cohort_member_request_ids", members)


@dataclass(frozen=True)
class AdaptiveDecodeWindowReceipt:
    request_id: str
    slot_id: int
    window_index: int
    token_start: int
    token_end: int
    context_length: int
    active_batch: int
    started_at_us: int
    finished_at_us: int
    policy: AdaptiveDecodePolicy
    applied_ack: AdaptiveDecodePolicyAck | None
    fleet_energy_uj_by_domain: Mapping[str, int]
    latency_per_token_us: int
    phone_compute_us: int
    usb_transfer_us: int
    rpc_us: int
    exposed_tail_us: int
    output_valid: bool
    evidence_ids: tuple[str, ...]
    energy_boundary_id: str
    energy_attribution_kind: str
    failure_reason: str | None
    previous_record_sha256: str
    measurement_eligible: bool = True
    usb_upload_bytes: int = 0
    usb_download_bytes: int = 0
    desktop_compute_us: int = 0
    useful_overlap_us: int = 0
    request_queue_delay_us: int = 0
    protected_interference_us: int = 0
    usb_h2d_us: int = 0
    usb_d2h_us: int = 0
    accounting_token_count: int | None = None
    cohort_id: str | None = None
    cohort_member_request_ids: tuple[str, ...] = ()
    energy_owner_request_id: str | None = None
    configured_queue_depth: int = 0
    maximum_active_slots: int = 0
    maximum_outstanding_transfers: int = 0
    batched_calls: int = 0
    transfer_subrequests: int = 0
    maximum_tokens: int = 0
    completed_phone_calls: int | None = None
    completed_phone_input_rows: int | None = None
    window_role: str = "legacy_unspecified"
    next_active_batch: int | None = None
    membership_changed: bool = False
    execution_context_available: bool = True
    external_activity_sha256: str | None = None
    external_activity_changed: bool = False
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        boundary = AdaptiveDecodeWindowBoundary(
            request_id=self.request_id,
            slot_id=self.slot_id,
            window_index=self.window_index,
            token_start=self.token_start,
            token_end=self.token_end,
            started_at_us=self.started_at_us,
            finished_at_us=self.finished_at_us,
            policy=self.policy,
            applied_ack=self.applied_ack,
        )
        _integer("adaptive window context", self.context_length, 1)
        _integer("adaptive window batch", self.active_batch, 1)
        _integer("adaptive window token latency", self.latency_per_token_us, 1)
        raw = AdaptiveDecodeRawWindowObservation(
            fleet_energy_uj_by_domain=self.fleet_energy_uj_by_domain,
            phone_compute_us=self.phone_compute_us,
            usb_transfer_us=self.usb_transfer_us,
            rpc_us=self.rpc_us,
            exposed_tail_us=self.exposed_tail_us,
            output_valid=self.output_valid,
            evidence_ids=self.evidence_ids,
            energy_boundary_id=self.energy_boundary_id,
            energy_attribution_kind=self.energy_attribution_kind,
            failure_reason=self.failure_reason,
            usb_upload_bytes=self.usb_upload_bytes,
            usb_download_bytes=self.usb_download_bytes,
            desktop_compute_us=self.desktop_compute_us,
            useful_overlap_us=self.useful_overlap_us,
            request_queue_delay_us=self.request_queue_delay_us,
            protected_interference_us=self.protected_interference_us,
            active_batch=self.active_batch,
            next_active_batch=self.next_active_batch,
            membership_changed=self.membership_changed,
            execution_context_available=self.execution_context_available,
            usb_h2d_us=self.usb_h2d_us,
            usb_d2h_us=self.usb_d2h_us,
            accounting_token_count=self.accounting_token_count,
            cohort_id=self.cohort_id,
            cohort_member_request_ids=self.cohort_member_request_ids,
            energy_owner_request_id=self.energy_owner_request_id,
            configured_queue_depth=self.configured_queue_depth,
            maximum_active_slots=self.maximum_active_slots,
            maximum_outstanding_transfers=(
                self.maximum_outstanding_transfers
            ),
            batched_calls=self.batched_calls,
            transfer_subrequests=self.transfer_subrequests,
            maximum_tokens=self.maximum_tokens,
            completed_phone_calls=self.completed_phone_calls,
            completed_phone_input_rows=self.completed_phone_input_rows,
            external_activity_sha256=self.external_activity_sha256,
        )
        if type(self.measurement_eligible) is not bool:
            raise AdaptiveDecodeError(
                "adaptive measurement eligibility is invalid"
            )
        if type(self.external_activity_changed) is not bool:
            raise AdaptiveDecodeError(
                "adaptive external activity change is invalid"
            )
        if self.window_role not in {
            "exploration", "exploitation", "legacy_unspecified"
        }:
            raise AdaptiveDecodeError("adaptive window role is invalid")
        if (
            self.accounting_token_count is not None
            and self.accounting_token_count < boundary.token_count
        ):
            raise AdaptiveDecodeError(
                "adaptive accounting token count is too small"
            )
        if self.cohort_id is not None and (
            self.request_id != self.energy_owner_request_id
            or not self.cohort_member_request_ids
            or self.request_id != self.cohort_member_request_ids[0]
        ):
            raise AdaptiveDecodeError(
                "adaptive cohort window owner differs"
            )
        previous = self.previous_record_sha256
        if previous != "0" * 64:
            _sha256("adaptive previous record", "sha256:" + previous)
        object.__setattr__(
            self, "fleet_energy_uj_by_domain", raw.fleet_energy_uj_by_domain
        )
        object.__setattr__(self, "evidence_ids", raw.evidence_ids)
        object.__setattr__(
            self,
            "cohort_member_request_ids",
            raw.cohort_member_request_ids,
        )
        object.__setattr__(self, "record_sha256", canonical_sha256(
            self._json_without_hash(boundary)
        ))

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start

    @property
    def energy_token_count(self) -> int:
        return (
            self.token_count
            if self.accounting_token_count is None
            else self.accounting_token_count
        )

    @property
    def whole_fleet_energy_uj(self) -> int:
        return sum(self.fleet_energy_uj_by_domain.values())

    @property
    def energy_measurement_eligible(self) -> bool:
        return (
            self.measurement_eligible
            and "ASSUMED_4P5W" not in self.evidence_ids
            and self.energy_attribution_kind in {
                "isolated", "matched_abba", "device_domain"
            }
        )

    @property
    def energy_per_token_uj(self) -> int:
        return max(1, self.whole_fleet_energy_uj // self.energy_token_count)

    @property
    def usb_payload_bandwidth_bytes_per_s(self) -> int:
        if self.usb_transfer_us <= 0:
            return 0
        return (
            (self.usb_upload_bytes + self.usb_download_bytes) * 1_000_000
            // self.usb_transfer_us
        )

    def _json_without_hash(
        self, boundary: AdaptiveDecodeWindowBoundary | None = None
    ) -> dict[str, object]:
        result = {
            "active_batch": self.active_batch,
            "applied_ack": (
                None if self.applied_ack is None else self.applied_ack.to_json()
            ),
            "context_length": self.context_length,
            "evidence_ids": list(self.evidence_ids),
            "energy_attribution_kind": self.energy_attribution_kind,
            "energy_boundary_id": self.energy_boundary_id,
            "exposed_tail_us": self.exposed_tail_us,
            "failure_reason": self.failure_reason,
            "finished_at_us": self.finished_at_us,
            "fleet_energy_uj_by_domain": dict(self.fleet_energy_uj_by_domain),
            "latency_per_token_us": self.latency_per_token_us,
            "output_valid": self.output_valid,
            "phone_compute_us": self.phone_compute_us,
            "policy": self.policy.to_json(),
            "previous_record_sha256": self.previous_record_sha256,
            "request_id": self.request_id,
            "rpc_us": self.rpc_us,
            "schema": ADAPTIVE_DECODE_SCHEMA,
            "slot_id": self.slot_id,
            "started_at_us": self.started_at_us,
            "token_end": self.token_end,
            "token_start": self.token_start,
            "usb_transfer_us": self.usb_transfer_us,
            "usb_h2d_us": self.usb_h2d_us,
            "usb_d2h_us": self.usb_d2h_us,
            "window_index": self.window_index,
        }
        if not self.measurement_eligible:
            result["measurement_eligible"] = False
        if self.window_role != "legacy_unspecified":
            result["window_role"] = self.window_role
        if self.next_active_batch is not None:
            result["next_active_batch"] = self.next_active_batch
        if self.membership_changed:
            result["membership_changed"] = True
        if not self.execution_context_available:
            result["execution_context_available"] = False
        if self.external_activity_sha256 is not None:
            result["external_activity_sha256"] = self.external_activity_sha256
        if self.external_activity_changed:
            result["external_activity_changed"] = True
        if self.accounting_token_count is not None:
            result["accounting_token_count"] = self.accounting_token_count
        if self.completed_phone_calls is not None:
            result["completed_phone_calls"] = self.completed_phone_calls
            result["completed_phone_input_rows"] = (
                self.completed_phone_input_rows
            )
        if self.cohort_id is not None:
            result.update({
                "cohort_id": self.cohort_id,
                "cohort_member_request_ids": list(
                    self.cohort_member_request_ids
                ),
                "energy_owner_request_id": self.energy_owner_request_id,
            })
        for name in (
            "batched_calls",
            "configured_queue_depth",
            "desktop_compute_us",
            "maximum_active_slots",
            "maximum_outstanding_transfers",
            "maximum_tokens",
            "protected_interference_us",
            "request_queue_delay_us",
            "transfer_subrequests",
            "usb_download_bytes",
            "usb_upload_bytes",
            "useful_overlap_us",
        ):
            value = getattr(self, name)
            if value:
                result[name] = value
        return result

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["energy_per_token_uj"] = self.energy_per_token_uj
        result["record_sha256"] = self.record_sha256
        result["usb_payload_bandwidth_bytes_per_s"] = (
            self.usb_payload_bandwidth_bytes_per_s
        )
        result["whole_fleet_energy_uj"] = self.whole_fleet_energy_uj
        return result

    @classmethod
    def from_json(cls, value: object) -> "AdaptiveDecodeWindowReceipt":
        if type(value) is not dict or value.get("schema") != ADAPTIVE_DECODE_SCHEMA:
            raise AdaptiveDecodeError("adaptive window JSON is invalid")
        acknowledgement = value.get("applied_ack")
        result = cls(
            request_id=value.get("request_id"),
            slot_id=value.get("slot_id"),
            window_index=value.get("window_index"),
            token_start=value.get("token_start"),
            token_end=value.get("token_end"),
            context_length=value.get("context_length"),
            active_batch=value.get("active_batch"),
            started_at_us=value.get("started_at_us"),
            finished_at_us=value.get("finished_at_us"),
            policy=AdaptiveDecodePolicy.from_json(value.get("policy")),
            applied_ack=(
                None if acknowledgement is None
                else AdaptiveDecodePolicyAck.from_json(acknowledgement)
            ),
            fleet_energy_uj_by_domain=dict(
                value.get("fleet_energy_uj_by_domain", {})
            ),
            latency_per_token_us=value.get("latency_per_token_us"),
            phone_compute_us=value.get("phone_compute_us"),
            usb_transfer_us=value.get("usb_transfer_us"),
            rpc_us=value.get("rpc_us"),
            exposed_tail_us=value.get("exposed_tail_us"),
            output_valid=value.get("output_valid"),
            evidence_ids=tuple(value.get("evidence_ids", ())),
            energy_boundary_id=value.get(
                "energy_boundary_id", "legacy-diagnostic-boundary"
            ),
            energy_attribution_kind=value.get(
                "energy_attribution_kind", "diagnostic"
            ),
            failure_reason=value.get("failure_reason"),
            previous_record_sha256=value.get("previous_record_sha256"),
            measurement_eligible=value.get("measurement_eligible", True),
            usb_upload_bytes=value.get("usb_upload_bytes", 0),
            usb_download_bytes=value.get("usb_download_bytes", 0),
            desktop_compute_us=value.get("desktop_compute_us", 0),
            useful_overlap_us=value.get("useful_overlap_us", 0),
            request_queue_delay_us=value.get("request_queue_delay_us", 0),
            protected_interference_us=value.get(
                "protected_interference_us", 0
            ),
            usb_h2d_us=value.get("usb_h2d_us", 0),
            usb_d2h_us=value.get("usb_d2h_us", 0),
            accounting_token_count=value.get("accounting_token_count"),
            cohort_id=value.get("cohort_id"),
            cohort_member_request_ids=tuple(
                value.get("cohort_member_request_ids", ())
            ),
            energy_owner_request_id=value.get("energy_owner_request_id"),
            configured_queue_depth=value.get("configured_queue_depth", 0),
            maximum_active_slots=value.get("maximum_active_slots", 0),
            maximum_outstanding_transfers=value.get(
                "maximum_outstanding_transfers", 0
            ),
            batched_calls=value.get("batched_calls", 0),
            transfer_subrequests=value.get("transfer_subrequests", 0),
            maximum_tokens=value.get("maximum_tokens", 0),
            completed_phone_calls=value.get("completed_phone_calls"),
            completed_phone_input_rows=value.get(
                "completed_phone_input_rows"
            ),
            window_role=value.get("window_role", "legacy_unspecified"),
            next_active_batch=value.get("next_active_batch"),
            membership_changed=value.get("membership_changed", False),
            execution_context_available=value.get("execution_context_available", True),
            external_activity_sha256=value.get("external_activity_sha256"),
            external_activity_changed=value.get("external_activity_changed", False),
        )
        if (
            value.get("record_sha256") != result.record_sha256
            or value.get("whole_fleet_energy_uj") !=
                result.whole_fleet_energy_uj
            or value.get("energy_per_token_uj") != result.energy_per_token_uj
            or value.get("usb_payload_bandwidth_bytes_per_s", 0) !=
                result.usb_payload_bandwidth_bytes_per_s
        ):
            raise AdaptiveDecodeError("adaptive window hash or totals differ")
        return result


@dataclass(frozen=True)
class AdaptiveDecodeDirective:
    state: str
    reason: str
    target_token_index: int | None
    control: AdaptiveDecodeControl | None = None
    boundary: AdaptiveDecodeWindowBoundary | None = None

    def __post_init__(self) -> None:
        if self.state not in ADAPTIVE_DECODE_STATES:
            raise AdaptiveDecodeError("adaptive directive state is invalid")
        _text("adaptive directive reason", self.reason)
        if self.target_token_index is not None:
            _integer("adaptive directive target token", self.target_token_index)
        if self.control is not None and not isinstance(
            self.control, AdaptiveDecodeControl
        ):
            raise AdaptiveDecodeError("adaptive directive control is invalid")
        if self.boundary is not None and not isinstance(
            self.boundary, AdaptiveDecodeWindowBoundary
        ):
            raise AdaptiveDecodeError("adaptive directive boundary is invalid")


@dataclass(frozen=True)
class AdaptiveDecodeGroupedObservation:
    request_id: str
    ticket_id: str
    model_artifact_sha256: str
    planning_profile_sha256: str
    desktop_placement_sha256: str
    windows: tuple[AdaptiveDecodeWindowReceipt, ...]
    final_policy: AdaptiveDecodePolicy
    terminal_status: str
    state_history: tuple[str, ...]
    terminal_reason: str | None = None
    unmeasured_tail_tokens: int = 0
    unmeasured_tail_reason: str | None = None
    helper_layout_geometry_sha256: str | None = None
    final_policy_ack: AdaptiveDecodePolicyAck | None = None
    grouped_observation_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        for name in ("request_id", "ticket_id"):
            _text("adaptive grouped " + name, getattr(self, name))
        _sha256("adaptive grouped artifact", self.model_artifact_sha256)
        _sha256(
            "adaptive grouped planning profile",
            self.planning_profile_sha256,
        )
        _sha256("adaptive grouped placement", self.desktop_placement_sha256)
        windows = tuple(self.windows)
        if not windows or any(
            not isinstance(row, AdaptiveDecodeWindowReceipt) for row in windows
        ):
            raise AdaptiveDecodeError("adaptive grouped windows are invalid")
        if tuple(row.window_index for row in windows) != tuple(range(len(windows))):
            raise AdaptiveDecodeError("adaptive grouped window order differs")
        if any(row.request_id != self.request_id for row in windows):
            raise AdaptiveDecodeError("adaptive grouped request differs")
        if any(
            previous.token_end != current.token_start
            or previous.finished_at_us > current.started_at_us
            for previous, current in zip(windows, windows[1:])
        ):
            raise AdaptiveDecodeError(
                "adaptive grouped window coverage differs"
            )
        if not isinstance(self.final_policy, AdaptiveDecodePolicy):
            raise AdaptiveDecodeError("adaptive final policy is invalid")
        if self.terminal_status not in {"COMPLETED", "FAILED", "CANCELLED"}:
            raise AdaptiveDecodeError("adaptive terminal status is invalid")
        if self.terminal_reason is not None:
            _text("adaptive terminal reason", self.terminal_reason)
        _integer(
            "adaptive unmeasured tail tokens", self.unmeasured_tail_tokens
        )
        if bool(self.unmeasured_tail_tokens) != bool(
            self.unmeasured_tail_reason
        ):
            raise AdaptiveDecodeError(
                "adaptive unmeasured tail reason differs"
            )
        if self.unmeasured_tail_reason is not None:
            _text(
                "adaptive unmeasured tail reason",
                self.unmeasured_tail_reason,
            )
        if self.helper_layout_geometry_sha256 is not None:
            _sha256(
                "adaptive grouped helper layout geometry",
                self.helper_layout_geometry_sha256,
            )
        ack = self.final_policy_ack
        if ack is not None and (
            not isinstance(ack, AdaptiveDecodePolicyAck)
            or not self.unmeasured_tail_tokens
            or self.final_policy.baseline
            or ack.request_id != self.request_id
            or ack.slot_id != windows[-1].slot_id
            or ack.policy_hash != self.final_policy.policy_hash
            or ack.applied_token_index != windows[-1].token_end
            or ack.applied_at_us < windows[-1].finished_at_us
            or any(
                row.applied_ack is not None
                and ack.plan_generation <= row.applied_ack.plan_generation
                for row in windows
            )
        ):
            raise AdaptiveDecodeError("adaptive final acknowledgement differs")
        history = tuple(self.state_history)
        if not history or history[-1] != "COMPLETED" or any(
            value not in ADAPTIVE_DECODE_STATES for value in history
        ):
            raise AdaptiveDecodeError("adaptive state history is invalid")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "state_history", history)
        object.__setattr__(self, "grouped_observation_sha256", canonical_sha256(
            self._json_without_hash()
        ))

    def _json_without_hash(self) -> dict[str, object]:
        result = {
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "final_policy": self.final_policy.to_json(),
            "model_artifact_sha256": self.model_artifact_sha256,
            "planning_profile_sha256": self.planning_profile_sha256,
            "request_id": self.request_id,
            "schema": ADAPTIVE_DECODE_SCHEMA,
            "state_history": list(self.state_history),
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "ticket_id": self.ticket_id,
            "windows": [row.to_json() for row in self.windows],
        }
        if self.unmeasured_tail_tokens:
            result["unmeasured_tail_tokens"] = self.unmeasured_tail_tokens
            result["unmeasured_tail_reason"] = self.unmeasured_tail_reason
        if self.helper_layout_geometry_sha256 is not None:
            result["helper_layout_geometry_sha256"] = (
                self.helper_layout_geometry_sha256
            )
        if self.final_policy_ack is not None:
            result["final_policy_ack"] = self.final_policy_ack.to_json()
        return result

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["grouped_observation_sha256"] = self.grouped_observation_sha256
        return result

    @classmethod
    def from_json(cls, value: object) -> "AdaptiveDecodeGroupedObservation":
        if type(value) is not dict or value.get("schema") != ADAPTIVE_DECODE_SCHEMA:
            raise AdaptiveDecodeError("adaptive grouped JSON is invalid")
        result = cls(
            request_id=value.get("request_id"),
            ticket_id=value.get("ticket_id"),
            model_artifact_sha256=value.get("model_artifact_sha256"),
            planning_profile_sha256=value.get("planning_profile_sha256"),
            desktop_placement_sha256=value.get(
                "desktop_placement_sha256"
            ),
            windows=tuple(
                AdaptiveDecodeWindowReceipt.from_json(row)
                for row in value.get("windows", ())
            ),
            final_policy=AdaptiveDecodePolicy.from_json(
                value.get("final_policy")
            ),
            terminal_status=value.get("terminal_status"),
            state_history=tuple(value.get("state_history", ())),
            terminal_reason=value.get("terminal_reason"),
            unmeasured_tail_tokens=value.get(
                "unmeasured_tail_tokens", 0
            ),
            unmeasured_tail_reason=value.get(
                "unmeasured_tail_reason"
            ),
            helper_layout_geometry_sha256=value.get(
                "helper_layout_geometry_sha256"
            ),
            final_policy_ack=(
                None if value.get("final_policy_ack") is None else
                AdaptiveDecodePolicyAck.from_json(value["final_policy_ack"])
            ),
        )
        if value.get("grouped_observation_sha256") != (
            result.grouped_observation_sha256
        ):
            raise AdaptiveDecodeError("adaptive grouped hash differs")
        return result


def validate_policy_set(
    baseline: AdaptiveDecodePolicy,
    candidates: Sequence[AdaptiveDecodePolicy],
) -> tuple[AdaptiveDecodePolicy, ...]:
    if not isinstance(baseline, AdaptiveDecodePolicy) or not baseline.baseline:
        raise AdaptiveDecodeError("adaptive baseline policy is invalid")
    rows = tuple(candidates)
    if any(not isinstance(row, AdaptiveDecodePolicy) for row in rows):
        raise AdaptiveDecodeError("adaptive candidate policy is invalid")
    if len({row.policy_hash for row in rows}) != len(rows):
        raise AdaptiveDecodeError("adaptive candidate policies are duplicated")
    for row in rows:
        if (
            row.baseline
            or row.desktop_parent_route_id != baseline.route_id
            or row.desktop_placement_sha256 != baseline.desktop_placement_sha256
        ):
            raise AdaptiveDecodeError("adaptive candidate parent differs")
    return tuple(sorted(rows, key=lambda row: (
        row.columns,
        len(row.layer_indices),
        row.route_id,
    )))
