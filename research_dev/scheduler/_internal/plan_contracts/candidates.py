"""Execution-plan contracts grouped by responsibility: candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

from ..runtime_capabilities import ROUTE_MATURITY_STATES
from ..runtime_cost import RuntimeExecutorBinding
from .common import (
    AUTOMATED_CANDIDATE_SET_SCHEMA,
    RuntimePlanError,
    _integer,
    _text,
    primary_rejection_reason,
)
from .costs import AutomatedRouteCost
from .execution import RuntimeExecutionPlan


@dataclass(frozen=True)
class AutomatedRouteCandidate:
    candidate_id: str
    plan: RuntimeExecutionPlan
    binding: RuntimeExecutorBinding
    cost: AutomatedRouteCost
    maturity: str
    admitted: bool
    rejection_reasons: tuple[str, ...]
    baseline: bool
    pareto_dominated: bool = False
    marginal_system_cost: Mapping[str, object] | None = None
    system_finish_upper_us: int | None = None
    paired_baseline_route_id: str | None = None
    residency_break_even: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _text("automated candidate id", self.candidate_id)
        if not isinstance(self.plan, RuntimeExecutionPlan):
            raise RuntimePlanError("automated candidate plan is invalid")
        if self.plan.route_id != self.candidate_id:
            raise RuntimePlanError("automated candidate and plan ids differ")
        if not isinstance(self.binding, RuntimeExecutorBinding):
            raise RuntimePlanError("automated candidate binding is invalid")
        if (
            self.binding.route_id != self.candidate_id
            or self.binding.operator_plan_sha256 != self.plan.plan_sha256
        ):
            raise RuntimePlanError("automated candidate binding differs")
        if not isinstance(self.cost, AutomatedRouteCost):
            raise RuntimePlanError("automated candidate cost is invalid")
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimePlanError("automated candidate maturity is invalid")
        if type(self.admitted) is not bool or type(self.baseline) is not bool:
            raise RuntimePlanError("automated candidate flags are invalid")
        reasons = tuple(self.rejection_reasons)
        if len(reasons) != len(set(reasons)):
            raise RuntimePlanError("automated rejection reasons are duplicated")
        for reason in reasons:
            _text("automated rejection reason", reason)
        if self.admitted == bool(reasons):
            raise RuntimePlanError("automated admission and reasons disagree")
        if (self.marginal_system_cost is None) != (
            self.system_finish_upper_us is None
        ):
            raise RuntimePlanError(
                "automated marginal cost fields are incomplete"
            )
        if self.marginal_system_cost is not None:
            if not isinstance(self.marginal_system_cost, Mapping):
                raise RuntimePlanError(
                    "automated marginal system cost is invalid"
                )
            system_finish = _integer(
                "automated system finish upper",
                self.system_finish_upper_us,
            )
            if system_finish < self.cost.finish_upper_us:
                raise RuntimePlanError(
                    "automated system finish precedes request finish"
                )
            object.__setattr__(
                self,
                "marginal_system_cost",
                MappingProxyType(dict(sorted(
                    self.marginal_system_cost.items()
                ))),
            )
        if self.paired_baseline_route_id is not None:
            _text(
                "automated paired baseline route",
                self.paired_baseline_route_id,
            )
            if self.plan.baseline_executor_id is None:
                raise RuntimePlanError(
                    "paired route lacks a baseline executor"
                )
        if self.residency_break_even is not None:
            if not isinstance(self.residency_break_even, Mapping):
                raise RuntimePlanError(
                    "automated residency break-even is invalid"
                )
            object.__setattr__(
                self,
                "residency_break_even",
                MappingProxyType(dict(sorted(
                    self.residency_break_even.items()
                ))),
            )
        object.__setattr__(self, "rejection_reasons", tuple(sorted(reasons)))

    @property
    def route_family(self) -> str:
        return self.plan.route_family

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self.plan.device_ids

    @property
    def assisted_operator_kind(self) -> str | None:
        return self.plan.assisted_operator_kind

    @property
    def split_axis(self) -> str:
        return self.plan.split_axis

    @property
    def split_fraction_ppm(self) -> int:
        return self.plan.split_fraction_ppm

    @property
    def residency_variant(self) -> str:
        return self.plan.residency_variant

    @property
    def primary_rejection_reason(self) -> str | None:
        return primary_rejection_reason(self.rejection_reasons)

    def to_json(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "baseline": self.baseline,
            "binding": self.binding.to_json(),
            "candidate_id": self.candidate_id,
            "cost": self.cost.to_json(),
            "maturity": self.maturity,
            "marginal_system_cost": (
                None
                if self.marginal_system_cost is None
                else dict(self.marginal_system_cost)
            ),
            "pareto_dominated": self.pareto_dominated,
            "paired_baseline_route_id": self.paired_baseline_route_id,
            "plan": self.plan.to_json(),
            "primary_rejection_reason": self.primary_rejection_reason,
            "rejection_reasons": list(self.rejection_reasons),
            "residency_break_even": (
                None
                if self.residency_break_even is None
                else dict(self.residency_break_even)
            ),
            "system_finish_upper_us": self.system_finish_upper_us,
        }


_CANDIDATE_METADATA_REQUIRED = {
    "cache_hit",
    "capability_generation_sha256",
    "evaluated_plan_count",
    "input_token_bucket",
    "output_token_bucket",
    "rough_plan_count",
    "search_budget",
    "search_kind",
    "visited_plan_ids",
}


_CANDIDATE_METADATA_LEGACY_EPOCH = {
    "model_placement_epoch_generation",
    "model_placement_epoch_sha256",
    "virtual_queue_request_count",
}


_CANDIDATE_METADATA_PUBLISHED_EPOCH = {
    "model_placement_epoch",
    "model_placement_epoch_fast_path",
    "model_placement_epoch_invalidation_reason",
    "route_template_identity_sha256",
    "selected_residency_component_identity_sha256",
}


_CANDIDATE_METADATA_CROSS_SHAPE = {
    "route_template_cross_shape_reuse",
    "route_template_request_shape_bucket",
    "route_template_source_shape_bucket",
}


_CANDIDATE_METADATA_INDEPENDENT = {
    "model_placement_resolution",
    "phone_residency",
    "route_template_audit_sha256",
    "route_template_exact_request_reuse",
    "route_template_live_route_ids",
} | _CANDIDATE_METADATA_CROSS_SHAPE


def _validate_placement_resolution(metadata: Mapping[str, object]) -> None:
    if "model_placement_resolution" not in metadata:
        return
    resolution = dict(metadata["model_placement_resolution"])
    required = {
        "invalidated_epoch_sha256",
        "outcome",
        "pass_count",
        "passes",
        "schema",
    }
    if (
        set(resolution) != required
        or resolution["schema"]
            != "runtime-model-placement-resolution-v1"
    ):
        raise RuntimePlanError(
            "candidate set placement resolution fields differ"
        )
    pass_count = _integer(
        "candidate set placement resolution pass count",
        resolution["pass_count"],
        1,
    )
    passes = resolution["passes"]
    if (
        pass_count > 2
        or type(passes) is not list
        or len(passes) != pass_count
    ):
        raise RuntimePlanError(
            "candidate set placement resolution passes differ"
        )
    invalidated = resolution["invalidated_epoch_sha256"]
    if invalidated is not None and (
        type(invalidated) is not str
        or not invalidated.startswith("sha256:")
        or len(invalidated) != 71
    ):
        raise RuntimePlanError(
            "candidate set invalidated epoch is invalid"
        )
    _text(
        "candidate set placement resolution outcome",
        resolution["outcome"],
    )
    required_pass = {
        "epoch_generation",
        "live_component_identity_sha256",
        "live_selected_route_id",
        "proposed_component_identity_sha256",
        "proposed_route_id",
        "rejection_reasons",
    }
    for row in passes:
        if type(row) is not dict or set(row) != required_pass:
            raise RuntimePlanError(
                "candidate set placement resolution pass differs"
            )
        _integer(
            "candidate set placement epoch generation",
            row["epoch_generation"],
            1,
        )
        for name in (
            "live_component_identity_sha256",
            "proposed_component_identity_sha256",
        ):
            value = row[name]
            if (
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
            ):
                raise RuntimePlanError(
                    "candidate set placement component is invalid"
                )
        for name in ("live_selected_route_id", "proposed_route_id"):
            _text("candidate set placement " + name, row[name])
        reasons = row["rejection_reasons"]
        if (
            type(reasons) is not list
            or len(reasons) != len(set(reasons))
            or any(
                type(reason) is not str
                or not reason
                or not reason.isascii()
                for reason in reasons
            )
        ):
            raise RuntimePlanError(
                "candidate set placement rejections are invalid"
            )


def _validate_route_template_shape_metadata(
    metadata: Mapping[str, object],
) -> None:
    if (
        "route_template_exact_request_reuse" in metadata
        and type(metadata[
            "route_template_exact_request_reuse"
        ]) is not bool
    ):
        raise RuntimePlanError(
            "candidate set exact route template flag is invalid"
        )
    if not _CANDIDATE_METADATA_CROSS_SHAPE.issubset(metadata):
        return
    if type(metadata["route_template_cross_shape_reuse"]) is not bool:
        raise RuntimePlanError(
            "candidate set route template shape flag is invalid"
        )
    for name in (
        "route_template_request_shape_bucket",
        "route_template_source_shape_bucket",
    ):
        bucket = metadata[name]
        if (
            type(bucket) not in {tuple, list}
            or len(bucket) != 2
            or any(
                type(value) is not int or value < 1
                for value in bucket
            )
        ):
            raise RuntimePlanError(
                "candidate set route template shape is invalid"
            )


def _validate_search_metadata_counts(
    metadata: Mapping[str, object],
    row_count: int,
) -> None:
    if type(metadata["cache_hit"]) is not bool:
        raise RuntimePlanError("candidate set cache flag is invalid")
    for name in (
        "evaluated_plan_count",
        "input_token_bucket",
        "output_token_bucket",
        "rough_plan_count",
        "search_budget",
    ):
        _integer("candidate set " + name, metadata[name], 1)
    if metadata["evaluated_plan_count"] != row_count:
        raise RuntimePlanError(
            "candidate set evaluated plan count differs"
        )
    if metadata["search_budget"] > 32:
        raise RuntimePlanError(
            "candidate set search budget exceeds 32"
        )
    _text("candidate set search kind", metadata["search_kind"])
    _text(
        "candidate set capability generation",
        metadata["capability_generation_sha256"],
    )


def _validate_published_epoch_identity(
    metadata: Mapping[str, object],
    epoch: Mapping[str, object],
) -> None:
    for name in (
        "demand_generation_sha256",
        "selected_component_identity_sha256",
        "selected_desktop_parent_plan_sha256",
        "selected_desktop_parent_placement_sha256",
        "selected_operator_plan_sha256",
    ):
        value = _text(
            "candidate set placement epoch " + name, epoch[name]
        )
        if not value.startswith("sha256:") or len(value) != 71:
            raise RuntimePlanError(
                "candidate set placement epoch hash is invalid"
            )
    for name in (
        "pressure_bucket",
        "selected_desktop_parent_route_id",
        "selected_route_id",
    ):
        _text("candidate set placement epoch " + name, epoch[name])
    for name in (
        "expected_reuse_count",
        "selected_transition_energy_uj",
        "selected_transition_latency_us",
    ):
        _integer("candidate set placement epoch " + name, epoch[name])
    break_even = epoch["break_even_use_count"]
    if break_even is not None:
        _integer(
            "candidate set placement epoch break_even_use_count",
            break_even,
        )
    old_component = epoch["old_component_identity_sha256"]
    if old_component is not None and (
        type(old_component) is not str
        or not old_component.startswith("sha256:")
        or len(old_component) != 71
    ):
        raise RuntimePlanError(
            "candidate set placement old component is invalid"
        )
    phone_generation = epoch["phone_layout_generation"]
    if phone_generation is not None:
        _integer(
            "candidate set placement phone layout generation",
            phone_generation,
            1,
        )


def _validate_published_epoch_collections(
    metadata: Mapping[str, object],
    epoch: Mapping[str, object],
) -> None:
    trigger_reasons = epoch["trigger_reasons"]
    if (
        type(trigger_reasons) is not list
        or not trigger_reasons
        or any(
            type(value) is not str
            or not value
            or not value.isascii()
            for value in trigger_reasons
        )
    ):
        raise RuntimePlanError(
            "candidate set placement triggers are invalid"
        )
    if type(metadata["model_placement_epoch_fast_path"]) is not bool:
        raise RuntimePlanError(
            "candidate set placement fast-path flag is invalid"
        )
    _text(
        "candidate set placement invalidation reason",
        metadata["model_placement_epoch_invalidation_reason"],
    )
    for name in (
        "route_template_identity_sha256",
        "selected_residency_component_identity_sha256",
    ):
        value = _text("candidate set " + name, metadata[name])
        if not value.startswith("sha256:") or len(value) != 71:
            raise RuntimePlanError(
                "candidate set placement hash is invalid"
            )
    for name in (
        "selected_resident_artifact_sha256s",
        "selected_resident_shard_geometry_sha256s",
        "selected_session_resource_ids",
    ):
        values = epoch[name]
        if (
            type(values) is not list
            or name == "selected_resident_artifact_sha256s"
                and not values
            or len(values) != len(set(values))
            or any(
                type(value) is not str
                or not value
                or not value.isascii()
                for value in values
            )
        ):
            raise RuntimePlanError(
                "candidate set placement component is invalid"
            )


def _validate_published_epoch_metadata(
    metadata: Mapping[str, object],
) -> None:
    if not _CANDIDATE_METADATA_PUBLISHED_EPOCH.issubset(metadata):
        return
    if type(metadata["model_placement_epoch"]) is not dict:
        raise RuntimePlanError(
            "candidate set placement epoch is invalid"
        )
    epoch = metadata["model_placement_epoch"]
    required = {
        "break_even_use_count",
        "demand_generation_sha256",
        "expected_reuse_count",
        "old_component_identity_sha256",
        "phone_layout_generation",
        "pressure_bucket",
        "selected_component_identity_sha256",
        "selected_desktop_parent_plan_sha256",
        "selected_desktop_parent_placement_sha256",
        "selected_desktop_parent_route_id",
        "selected_operator_plan_sha256",
        "selected_resident_artifact_sha256s",
        "selected_resident_shard_geometry_sha256s",
        "selected_route_id",
        "selected_session_resource_ids",
        "selected_transition_energy_uj",
        "selected_transition_latency_us",
        "trigger_reasons",
    }
    if not required.issubset(epoch):
        raise RuntimePlanError(
            "candidate set placement epoch metadata is incomplete"
        )
    _validate_published_epoch_identity(metadata, epoch)
    _validate_published_epoch_collections(metadata, epoch)


def _validate_optional_candidate_metadata(
    metadata: Mapping[str, object],
) -> None:
    if "route_template_audit_sha256" in metadata:
        value = _text(
            "candidate set route template audit",
            metadata["route_template_audit_sha256"],
        )
        if not value.startswith("sha256:") or len(value) != 71:
            raise RuntimePlanError(
                "candidate set route template audit is invalid"
            )
    if _CANDIDATE_METADATA_LEGACY_EPOCH.issubset(metadata):
        _integer(
            "candidate set model placement epoch generation",
            metadata["model_placement_epoch_generation"],
        )
        _integer(
            "candidate set virtual queue request count",
            metadata["virtual_queue_request_count"],
        )
        epoch_sha256 = _text(
            "candidate set model placement epoch",
            metadata["model_placement_epoch_sha256"],
        )
        if (
            not epoch_sha256.startswith("sha256:")
            or len(epoch_sha256) != 71
        ):
            raise RuntimePlanError(
                "candidate set model placement epoch is invalid"
            )


def _validate_candidate_search_metadata(
    metadata: dict[str, object],
    candidate_ids: set[str],
    row_count: int,
) -> None:
    allowed = (
        _CANDIDATE_METADATA_REQUIRED
        | _CANDIDATE_METADATA_LEGACY_EPOCH
        | _CANDIDATE_METADATA_PUBLISHED_EPOCH
        | _CANDIDATE_METADATA_INDEPENDENT
    )
    if (
        not _CANDIDATE_METADATA_REQUIRED.issubset(metadata)
        or not set(metadata).issubset(allowed)
        or bool(_CANDIDATE_METADATA_LEGACY_EPOCH.intersection(metadata))
            != _CANDIDATE_METADATA_LEGACY_EPOCH.issubset(metadata)
        or bool(_CANDIDATE_METADATA_PUBLISHED_EPOCH.intersection(metadata))
            != _CANDIDATE_METADATA_PUBLISHED_EPOCH.issubset(metadata)
    ):
        raise RuntimePlanError(
            "candidate set search metadata fields differ"
        )
    if bool(_CANDIDATE_METADATA_CROSS_SHAPE.intersection(metadata)) != (
        _CANDIDATE_METADATA_CROSS_SHAPE.issubset(metadata)
    ):
        raise RuntimePlanError(
            "candidate set route template shape fields differ"
        )
    live_route_ids = metadata.get("route_template_live_route_ids")
    if live_route_ids is not None and (
        type(live_route_ids) not in {list, tuple}
        or not live_route_ids
        or len(set(live_route_ids)) != len(live_route_ids)
        or any(
            type(route_id) is not str
            or route_id not in candidate_ids
            for route_id in live_route_ids
        )
    ):
        raise RuntimePlanError(
            "candidate set live route identities differ"
        )
    for name, message in (
        ("phone_residency",
         "candidate set phone residency metadata is invalid"),
        ("model_placement_resolution",
         "candidate set placement resolution is invalid"),
    ):
        if name in metadata and not isinstance(metadata[name], Mapping):
            raise RuntimePlanError(message)
    _validate_placement_resolution(metadata)
    _validate_route_template_shape_metadata(metadata)
    _validate_search_metadata_counts(metadata, row_count)
    _validate_published_epoch_metadata(metadata)
    _validate_optional_candidate_metadata(metadata)
    visited = tuple(metadata["visited_plan_ids"])
    if len(visited) != row_count or any(
        type(value) is not str or not value or not value.isascii()
        for value in visited
    ):
        raise RuntimePlanError(
            "candidate set visited plan ids are invalid"
        )
    metadata["visited_plan_ids"] = visited


def _candidate_set_generation_sha256(candidate_set) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        {
            "baseline_route_id": candidate_set.baseline_route_id,
            "candidates": [
                candidate_set._generation_candidate(row)
                for row in candidate_set.candidates
            ],
            "model_id": candidate_set.model_id,
            "recovery_fallback_route_id": (
                candidate_set.recovery_fallback_route_id
            ),
            "request_id": candidate_set.request_id,
            "schema": AUTOMATED_CANDIDATE_SET_SCHEMA,
            "search_metadata": {
                key: list(value) if type(value) is tuple else value
                for key, value in candidate_set.search_metadata.items()
            },
            "snapshot_id": candidate_set.snapshot_id,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class AutomatedCandidateSet:
    request_id: str
    model_id: str
    snapshot_id: str
    candidates: tuple[AutomatedRouteCandidate, ...]
    baseline_route_id: str
    recovery_fallback_route_id: str | None = None
    search_metadata: Mapping[str, object] = field(default_factory=dict)
    _generation_sha256: str = field(init=False, repr=False, compare=False)

    @staticmethod
    def _generation_candidate(
        row: AutomatedRouteCandidate,
    ) -> tuple[object, ...]:
        binding = row.binding
        cost = row.cost
        return (
            row.candidate_id,
            row.plan.plan_sha256,
            (
                binding.artifact_bytes,
                binding.artifact_sha256,
                binding.backend,
                binding.eligibility_reasons,
                binding.endpoint,
                binding.executor_id,
                binding.operator_plan_protocol,
                tuple(
                    (
                        participant.backend,
                        participant.device_id,
                        participant.endpoint,
                        participant.executor_id,
                        participant.resource_ids,
                    )
                    for participant in binding.participants
                ),
                binding.queueable,
                binding.ready,
                binding.resident,
                binding.route_id,
            ),
            (
                cost.start_us,
                cost.finish_us,
                cost.finish_upper_us,
                cost.service_us,
                cost.service_upper_us,
                cost.queue_delay_us,
                cost.compute_us,
                cost.memory_us,
                cost.transfer_us,
                cost.join_wait_us,
                cost.exposed_tail_us,
                cost.load_us,
                cost.eviction_us,
                cost.restore_us,
                cost.switching_us,
                cost.interference_us,
                cost.fleet_energy_uj,
                cost.fleet_energy_lower_uj,
                cost.fleet_energy_upper_uj,
                cost.component_service_us,
                cost.component_energy_uj,
                cost.warm_execution_energy_uj,
                cost.warm_execution_energy_lower_uj,
                cost.warm_execution_energy_upper_uj,
                cost.transition_energy_uj,
                cost.transition_energy_lower_uj,
                cost.transition_energy_upper_uj,
                cost.latency_evidence,
                cost.energy_evidence,
                tuple(cost.memory_by_resource_bytes.items()),
                cost.residency_hysteresis_us,
                tuple(
                    (
                        transfer.step_id,
                        transfer.source_device,
                        transfer.target_device,
                        transfer.payload_bytes,
                        transfer.invocations,
                        transfer.total_bytes,
                        transfer.queue_depth,
                        transfer.concurrent_streams,
                        transfer.message_waves,
                        transfer.fixed_latency_us,
                        transfer.latency_us,
                        transfer.dynamic_energy_uj,
                        transfer.link_ids,
                    )
                    for transfer in cost.transfer_costs
                ),
            ),
            row.maturity,
            row.admitted,
            row.rejection_reasons,
            row.baseline,
            row.pareto_dominated,
            (
                None
                if row.marginal_system_cost is None
                else tuple(row.marginal_system_cost.items())
            ),
            row.system_finish_upper_us,
            row.paired_baseline_route_id,
            (
                None
                if row.residency_break_even is None
                else tuple(row.residency_break_even.items())
            ),
        )

    def __post_init__(self) -> None:
        for name in (
            "request_id", "model_id", "snapshot_id", "baseline_route_id"
        ):
            _text(f"candidate set {name}", getattr(self, name))
        rows = tuple(self.candidates)
        if (
            not rows
            or any(
                not isinstance(row, AutomatedRouteCandidate)
                for row in rows
            )
        ):
            raise RuntimePlanError("candidate set rows are invalid")
        candidate_ids = {row.candidate_id for row in rows}
        if len(candidate_ids) != len(rows):
            raise RuntimePlanError("candidate ids are duplicated")
        baseline = next(
            (
                row for row in rows
                if row.candidate_id == self.baseline_route_id
            ),
            None,
        )
        if (
            baseline is None
            or not baseline.baseline
            or not baseline.admitted
        ):
            raise RuntimePlanError(
                "candidate set fallback is unavailable"
            )
        if sum(int(row.baseline) for row in rows) != 1:
            raise RuntimePlanError(
                "candidate set requires one baseline"
            )
        recovery_route_id = (
            self.baseline_route_id
            if self.recovery_fallback_route_id is None
            else _text(
                "candidate set recovery fallback route",
                self.recovery_fallback_route_id,
            )
        )
        if not any(
            row.candidate_id == recovery_route_id for row in rows
        ):
            raise RuntimePlanError(
                "candidate set recovery fallback is absent"
            )
        metadata = dict(self.search_metadata)
        if metadata:
            _validate_candidate_search_metadata(
                metadata, candidate_ids, len(rows)
            )
        object.__setattr__(
            self,
            "candidates",
            tuple(sorted(rows, key=lambda row: row.candidate_id)),
        )
        object.__setattr__(
            self, "recovery_fallback_route_id", recovery_route_id
        )
        object.__setattr__(
            self,
            "search_metadata",
            MappingProxyType(dict(sorted(metadata.items()))),
        )
        object.__setattr__(
            self,
            "_generation_sha256",
            _candidate_set_generation_sha256(self),
        )

    @property
    def baseline(self) -> AutomatedRouteCandidate:
        return next(
            row for row in self.candidates
            if row.candidate_id == self.baseline_route_id
        )

    @property
    def generation_sha256(self) -> str:
        return self._generation_sha256

    @property
    def recovery_fallback(self) -> AutomatedRouteCandidate:
        return next(
            row for row in self.candidates
            if row.candidate_id == self.recovery_fallback_route_id
        )

    def to_json(self) -> dict[str, object]:
        return {
            "baseline_route_id": self.baseline_route_id,
            "candidates": [row.to_json() for row in self.candidates],
            "model_id": self.model_id,
            "recovery_fallback_route_id": self.recovery_fallback_route_id,
            "request_id": self.request_id,
            "schema": AUTOMATED_CANDIDATE_SET_SCHEMA,
            "search_metadata": {
                key: list(value) if type(value) is tuple else value
                for key, value in self.search_metadata.items()
            },
            "snapshot_id": self.snapshot_id,
        }
