"""Arrival-prefix residency cohorts for runtime placement planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median_low
from types import MappingProxyType
from typing import Mapping, Sequence

from .runtime_cost import RuntimeExecutorBinding
from .runtime_capabilities import (
    HeterogeneousRuntimeSnapshot,
    RuntimeCapabilityCatalog,
)
from .runtime_controller import RuntimeRequestTicket
from .runtime_plan import RuntimeExecutionPlan, RuntimePhoneShard
from .types import canonical_sha256


class RuntimeResidencyCohortError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise RuntimeResidencyCohortError(
            f"{name} must be non-empty ASCII text"
        )
    return value


@dataclass(frozen=True)
class RuntimeResidencyCohortPolicy:
    history_limit: int = 8
    maximum_hysteresis_us: int = 30_000_000
    minimum_interarrival_samples: int = 2
    reuse_horizon_us: int = 30_000_000

    def __post_init__(self) -> None:
        if type(self.history_limit) is not int or self.history_limit < 3:
            raise RuntimeResidencyCohortError(
                "residency cohort history limit is invalid"
            )
        if (
            type(self.maximum_hysteresis_us) is not int
            or self.maximum_hysteresis_us < 0
            or type(self.reuse_horizon_us) is not int
            or self.reuse_horizon_us < 0
            or type(self.minimum_interarrival_samples) is not int
            or self.minimum_interarrival_samples < 1
            or self.minimum_interarrival_samples >= self.history_limit
        ):
            raise RuntimeResidencyCohortError(
                "residency cohort policy is invalid"
            )


@dataclass(frozen=True)
class RuntimeResidencyComponentIdentity:
    artifact_sha256: str
    resident_shard_geometry_sha256: tuple[str, ...]
    desktop_placement_sha256: str | None
    transport_generation: str | None
    operator_protocol: str | None
    session_resource_ids: tuple[str, ...]
    executor_id: str = field(compare=False)
    resident_artifact_sha256s: tuple[str, ...] = ()
    resident_endpoint: str | None = None
    identity_sha256: str = ""

    def __post_init__(self) -> None:
        artifact = _text(
            "residency component artifact", self.artifact_sha256
        )
        if not artifact.startswith("sha256:") or len(artifact) != 71:
            raise RuntimeResidencyCohortError(
                "residency component artifact is invalid"
            )
        geometries = tuple(sorted(
            _text("residency component geometry", value)
            for value in self.resident_shard_geometry_sha256
        ))
        if len(geometries) != len(set(geometries)) or any(
            not value.startswith("sha256:") or len(value) != 71
            for value in geometries
        ):
            raise RuntimeResidencyCohortError(
                "residency component geometry is invalid"
            )
        desktop = self.desktop_placement_sha256
        if desktop is not None and (
            not _text("residency component desktop placement", desktop)
                .startswith("sha256:")
            or len(desktop) != 71
        ):
            raise RuntimeResidencyCohortError(
                "residency component desktop placement is invalid"
            )
        for name in ("transport_generation", "operator_protocol"):
            value = getattr(self, name)
            if value is not None:
                _text("residency component " + name, value)
        resources = tuple(sorted(
            _text("residency component session resource", value)
            for value in self.session_resource_ids
        ))
        if len(resources) != len(set(resources)):
            raise RuntimeResidencyCohortError(
                "residency component session resources are duplicated"
            )
        _text("residency component executor", self.executor_id)
        resident_artifacts = self.resident_artifact_sha256s or (artifact,)
        resident_artifacts = tuple(sorted(
            _text("residency component resident artifact", value)
            for value in resident_artifacts
        ))
        if len(resident_artifacts) != len(set(resident_artifacts)) or any(
            not value.startswith("sha256:") or len(value) != 71
            for value in resident_artifacts
        ):
            raise RuntimeResidencyCohortError(
                "residency component resident artifacts are invalid"
            )
        endpoint = self.resident_endpoint
        if endpoint is None:
            endpoint = self.executor_id
        endpoint = _text("residency component endpoint", endpoint)
        object.__setattr__(self, "resident_endpoint", endpoint)
        object.__setattr__(
            self, "resident_shard_geometry_sha256", geometries
        )
        object.__setattr__(self, "session_resource_ids", resources)
        object.__setattr__(
            self, "resident_artifact_sha256s", resident_artifacts
        )
        expected = canonical_sha256(self._json_without_hash())
        if self.identity_sha256 and self.identity_sha256 != expected:
            raise RuntimeResidencyCohortError(
                "residency component identity hash differs"
            )
        object.__setattr__(self, "identity_sha256", expected)

    def _json_without_hash(self) -> dict[str, object]:
        result = {
            "artifact_sha256": self.artifact_sha256,
            "desktop_placement_sha256": self.desktop_placement_sha256,
            "operator_protocol": self.operator_protocol,
            "resident_endpoint": self.resident_endpoint,
            "resident_shard_geometry_sha256": list(
                self.resident_shard_geometry_sha256
            ),
            "schema": "research-scheduler-residency-component-v2",
            "session_resource_ids": list(self.session_resource_ids),
            "transport_generation": self.transport_generation,
        }
        if self.resident_artifact_sha256s != (self.artifact_sha256,):
            result["resident_artifact_sha256s"] = list(
                self.resident_artifact_sha256s
            )
        return result

    def covers_artifact(self, artifact_sha256: str) -> bool:
        return artifact_sha256 in self.resident_artifact_sha256s

    def to_json(self) -> dict[str, object]:
        result = self._json_without_hash()
        result["executor_id"] = self.executor_id
        result["identity_sha256"] = self.identity_sha256
        return result


def runtime_residency_component_identity_from_parts(
    artifact_sha256: str,
    resident_shard_geometry_sha256: tuple[str, ...],
    desktop_placement_sha256: str | None,
    transport_generation: str | None,
    operator_protocol: str | None,
    session_resource_ids: tuple[str, ...],
    executor_id: str,
    resident_endpoint: str | None,
    phone_shards: tuple[RuntimePhoneShard, ...] = (),
) -> RuntimeResidencyComponentIdentity:
    if any(row.artifact_sha256 is None for row in phone_shards):
        raise RuntimeResidencyCohortError(
            "phone residency shard lacks artifact identity"
        )
    resident_artifacts = tuple(sorted({
        str(row.artifact_sha256) for row in phone_shards
    }))
    residency_artifact_sha256 = artifact_sha256
    identity_desktop_placement_sha256 = desktop_placement_sha256
    if len(resident_artifacts) > 1:
        residency_artifact_sha256 = canonical_sha256({
            "shards": [
                {
                    "artifact_sha256": row.artifact_sha256,
                    "geometry_sha256": row.resident_geometry_sha256,
                    "session_id": row.session_id,
                }
                for row in phone_shards
            ],
        })
        identity_desktop_placement_sha256 = None
    return RuntimeResidencyComponentIdentity(
        artifact_sha256=residency_artifact_sha256,
        resident_shard_geometry_sha256=(
            resident_shard_geometry_sha256
        ),
        desktop_placement_sha256=(
            identity_desktop_placement_sha256
        ),
        transport_generation=transport_generation,
        operator_protocol=operator_protocol,
        session_resource_ids=session_resource_ids,
        executor_id=executor_id,
        resident_artifact_sha256s=resident_artifacts,
        resident_endpoint=resident_endpoint,
    )


def runtime_residency_component_identity(
    artifact_sha256: str,
    plan: RuntimeExecutionPlan,
    binding: RuntimeExecutorBinding,
) -> RuntimeResidencyComponentIdentity:
    if not isinstance(plan, RuntimeExecutionPlan):
        raise RuntimeResidencyCohortError(
            "residency component plan is invalid"
        )
    if not isinstance(binding, RuntimeExecutorBinding):
        raise RuntimeResidencyCohortError(
            "residency component binding is invalid"
        )
    if binding.operator_plan_sha256 != plan.plan_sha256:
        raise RuntimeResidencyCohortError(
            "residency component plan and binding differ"
        )
    parameters = plan.adapter_parameters
    transport = parameters.get("usb_transport_generation")
    if transport is None:
        transport = parameters.get("request_transport")
    if transport is not None and type(transport) is not str:
        raise RuntimeResidencyCohortError(
            "residency component transport is invalid"
        )
    session_resources = tuple(
        demand.resource_id for demand in plan.memory_demands
        if demand.kind == "session_residency_constraint"
    )
    shard_geometries = tuple(
        shard.resident_geometry_sha256
        for shard in plan.execution_contract.phone_shards
    )
    shard_set_geometry = parameters.get(
        "phone_shard_set_geometry_sha256"
    )
    if shard_set_geometry is None:
        shard_set_geometry = parameters.get(
            "ffn_resident_geometry_sha256"
        )
    if shard_set_geometry is not None:
        if type(shard_set_geometry) is not str:
            raise RuntimeResidencyCohortError(
                "residency component shard-set geometry is invalid"
            )
        shard_geometries += (shard_set_geometry,)
    phone_shards = plan.execution_contract.phone_shards
    return runtime_residency_component_identity_from_parts(
        artifact_sha256=artifact_sha256,
        resident_shard_geometry_sha256=shard_geometries,
        desktop_placement_sha256=plan.desktop_placement_sha256,
        transport_generation=transport,
        operator_protocol=binding.operator_plan_protocol,
        session_resource_ids=session_resources,
        executor_id=binding.executor_id,
        resident_endpoint=(
            binding.endpoint
            if shard_geometries or session_resources
            else binding.executor_id
        ),
        phone_shards=phone_shards,
    )


@dataclass(frozen=True)
class RuntimeResidencyCohortHold:
    resource_id: str
    model_id: str
    artifact_sha256: str
    hold_until_us: int
    active_request_count: int
    predicted_reuse_us: int | None
    replacement_cost_us: int
    component_identity_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "resource_id",
            "model_id",
            "artifact_sha256",
            "component_identity_sha256",
        ):
            _text("residency cohort " + name, getattr(self, name))
        if (
            not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or type(self.hold_until_us) is not int
            or self.hold_until_us < 0
            or type(self.active_request_count) is not int
            or self.active_request_count < 0
            or type(self.replacement_cost_us) is not int
            or self.replacement_cost_us < 0
            or (
                self.predicted_reuse_us is not None
                and (
                    type(self.predicted_reuse_us) is not int
                    or self.predicted_reuse_us < 0
                )
            )
        ):
            raise RuntimeResidencyCohortError(
                "residency cohort hold is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "active_request_count": self.active_request_count,
            "artifact_sha256": self.artifact_sha256,
            "component_identity_sha256": self.component_identity_sha256,
            "hold_until_us": self.hold_until_us,
            "model_id": self.model_id,
            "predicted_reuse_us": self.predicted_reuse_us,
            "replacement_cost_us": self.replacement_cost_us,
            "resource_id": self.resource_id,
        }


@dataclass(frozen=True)
class RuntimeResidencyReuseProjection:
    artifact_sha256: str
    component_identity_sha256: str
    observed_request_count: int
    predicted_additional_uses: int
    horizon_us: int

    def __post_init__(self) -> None:
        _text("residency reuse artifact", self.artifact_sha256)
        _text(
            "residency reuse component", self.component_identity_sha256
        )
        if (
            not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or not self.component_identity_sha256.startswith("sha256:")
            or len(self.component_identity_sha256) != 71
            or type(self.observed_request_count) is not int
            or self.observed_request_count < 1
            or type(self.predicted_additional_uses) is not int
            or self.predicted_additional_uses < 0
            or type(self.horizon_us) is not int
            or self.horizon_us < 0
        ):
            raise RuntimeResidencyCohortError(
                "residency reuse projection is invalid"
            )

    @property
    def expected_use_count(self) -> int:
        return self.observed_request_count + self.predicted_additional_uses

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "component_identity_sha256": (
                self.component_identity_sha256
            ),
            "expected_use_count": self.expected_use_count,
            "horizon_us": self.horizon_us,
            "observed_request_count": self.observed_request_count,
            "predicted_additional_uses": self.predicted_additional_uses,
        }


@dataclass(frozen=True)
class RuntimeModelPlacementEpoch:
    artifact_sha256: str
    generation: int
    observed_at_us: int
    active_request_count: int
    virtual_queue_request_count: int
    component_opportunity_counts: Mapping[str, int]
    reuse_projections: Mapping[str, RuntimeResidencyReuseProjection]
    selected_component_identity_sha256: str | None = None
    route_template_identity_sha256: str | None = None
    selected_route_id: str | None = None
    selected_executor_id: str | None = None
    selected_operator_plan_sha256: str | None = None
    capability_generation_sha256: str | None = None
    profile_generation_sha256: str | None = None
    transport_generation_sha256: str | None = None
    learning_generation_sha256: str | None = None
    input_token_bucket: int | None = None
    output_token_bucket: int | None = None
    quality_requirement: str | None = None
    objective: str | None = None
    maximum_latency_ppm: int | None = None
    expected_reuse_count: int | None = None
    valid_from_us: int | None = None
    valid_until_us: int | None = None
    demand_generation_sha256: str | None = None
    pressure_bucket: str | None = None
    trigger_reasons: tuple[str, ...] = ()
    break_even_use_count: int | None = None
    selected_transition_latency_us: int | None = None
    selected_transition_energy_uj: int | None = None
    old_component_identity_sha256: str | None = None
    selected_desktop_parent_route_id: str | None = None
    selected_desktop_parent_plan_sha256: str | None = None
    selected_desktop_parent_placement_sha256: str | None = None
    selected_resident_artifact_sha256s: tuple[str, ...] = ()
    selected_resident_shard_geometry_sha256s: tuple[str, ...] = ()
    selected_session_resource_ids: tuple[str, ...] = ()
    phone_layout_generation: int | None = None
    allowed_adaptive_fractions_ppm: tuple[int, ...] = ()
    invalidation_reason: str | None = None
    epoch_sha256: str = ""

    def __post_init__(self) -> None:
        artifact_sha256 = _text(
            "model placement epoch artifact", self.artifact_sha256
        )
        if (
            not artifact_sha256.startswith("sha256:")
            or len(artifact_sha256) != 71
            or type(self.generation) is not int
            or self.generation < 0
            or type(self.observed_at_us) is not int
            or self.observed_at_us < 0
            or type(self.active_request_count) is not int
            or self.active_request_count < 0
            or type(self.virtual_queue_request_count) is not int
            or self.virtual_queue_request_count < 0
        ):
            raise RuntimeResidencyCohortError(
                "model placement epoch is invalid"
            )
        counts = dict(self.component_opportunity_counts)
        projections = dict(self.reuse_projections)
        if any(
            not isinstance(component_id, str)
            or not component_id.startswith("sha256:")
            or len(component_id) != 71
            or type(count) is not int
            or count < 1
            for component_id, count in counts.items()
        ):
            raise RuntimeResidencyCohortError(
                "model placement epoch opportunity count is invalid"
            )
        if any(
            component_id != projection.component_identity_sha256
            or projection.artifact_sha256 != artifact_sha256
            for component_id, projection in projections.items()
        ):
            raise RuntimeResidencyCohortError(
                "model placement epoch projection is invalid"
            )
        counts = dict(sorted(counts.items()))
        projections = dict(sorted(projections.items()))
        publication_values = (
            self.selected_component_identity_sha256,
            self.route_template_identity_sha256,
            self.selected_route_id,
            self.selected_executor_id,
            self.selected_operator_plan_sha256,
            self.capability_generation_sha256,
            self.profile_generation_sha256,
            self.transport_generation_sha256,
            self.learning_generation_sha256,
            self.input_token_bucket,
            self.output_token_bucket,
            self.quality_requirement,
            self.objective,
            self.maximum_latency_ppm,
            self.expected_reuse_count,
            self.valid_from_us,
            self.valid_until_us,
            self.demand_generation_sha256,
            self.pressure_bucket,
            self.selected_transition_latency_us,
            self.selected_transition_energy_uj,
            self.selected_desktop_parent_route_id,
            self.selected_desktop_parent_plan_sha256,
            self.selected_desktop_parent_placement_sha256,
        )
        published = any(value is not None for value in publication_values)
        if published and any(value is None for value in publication_values):
            raise RuntimeResidencyCohortError(
                "published model placement epoch is incomplete"
            )
        fractions = tuple(self.allowed_adaptive_fractions_ppm)
        if published:
            for name in (
                "selected_component_identity_sha256",
                "route_template_identity_sha256",
                "selected_operator_plan_sha256",
                "capability_generation_sha256",
                "profile_generation_sha256",
                "transport_generation_sha256",
                "learning_generation_sha256",
                "demand_generation_sha256",
                "selected_desktop_parent_plan_sha256",
                "selected_desktop_parent_placement_sha256",
            ):
                value = _text(
                    "model placement epoch " + name,
                    getattr(self, name),
                )
                if not value.startswith("sha256:") or len(value) != 71:
                    raise RuntimeResidencyCohortError(
                        "model placement epoch hash is invalid"
                    )
            for name in (
                "selected_route_id",
                "selected_executor_id",
                "quality_requirement",
                "objective",
                "pressure_bucket",
                "selected_desktop_parent_route_id",
            ):
                _text(
                    "model placement epoch " + name,
                    getattr(self, name),
                )
            for name in (
                "input_token_bucket",
                "output_token_bucket",
                "maximum_latency_ppm",
                "expected_reuse_count",
                "valid_from_us",
                "valid_until_us",
                "selected_transition_latency_us",
                "selected_transition_energy_uj",
            ):
                value = getattr(self, name)
                if type(value) is not int or value < 0:
                    raise RuntimeResidencyCohortError(
                        "model placement epoch bound is invalid"
                    )
            if (
                self.break_even_use_count is not None
                and (
                    type(self.break_even_use_count) is not int
                    or self.break_even_use_count < 0
                )
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch break-even is invalid"
                )
            if (
                self.input_token_bucket == 0
                or self.output_token_bucket == 0
                or self.maximum_latency_ppm == 0
                or self.expected_reuse_count == 0
                or self.valid_until_us <= self.valid_from_us
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch validity is invalid"
                )
            if (
                not fractions
                or len(fractions) != len(set(fractions))
                or any(
                    type(value) is not int
                    or value < 0
                    or value > 1_000_000
                    for value in fractions
                )
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch adaptive fractions are invalid"
                )
            resident_artifacts = tuple(sorted(
                _text(
                    "model placement epoch resident artifact", value
                )
                for value in self.selected_resident_artifact_sha256s
            ))
            shard_geometries = tuple(sorted(
                _text(
                    "model placement epoch shard geometry", value
                )
                for value in self.selected_resident_shard_geometry_sha256s
            ))
            session_resources = tuple(sorted(
                _text(
                    "model placement epoch session resource", value
                )
                for value in self.selected_session_resource_ids
            ))
            if (
                not resident_artifacts
                or len(resident_artifacts) != len(set(resident_artifacts))
                or any(
                    not value.startswith("sha256:") or len(value) != 71
                    for value in resident_artifacts
                )
                or len(shard_geometries) != len(set(shard_geometries))
                or any(
                    not value.startswith("sha256:") or len(value) != 71
                    for value in shard_geometries
                )
                or len(session_resources) != len(set(session_resources))
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch resident component is invalid"
                )
            if self.phone_layout_generation is not None and (
                type(self.phone_layout_generation) is not int
                or self.phone_layout_generation < 1
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch phone layout generation is invalid"
                )
            if (
                self.phone_layout_generation is not None
                and not shard_geometries
            ):
                raise RuntimeResidencyCohortError(
                    "model placement epoch phone layout binding is incomplete"
                )
            fractions = tuple(sorted(fractions))
            reasons = tuple(sorted(
                _text("model placement epoch trigger reason", value)
                for value in self.trigger_reasons
            ))
            if not reasons or len(reasons) != len(set(reasons)):
                raise RuntimeResidencyCohortError(
                    "model placement epoch trigger reasons are invalid"
                )
            if self.old_component_identity_sha256 is not None:
                old_component = _text(
                    "model placement epoch old component",
                    self.old_component_identity_sha256,
                )
                if (
                    not old_component.startswith("sha256:")
                    or len(old_component) != 71
                ):
                    raise RuntimeResidencyCohortError(
                        "model placement epoch old component hash is invalid"
                    )
        elif fractions:
            raise RuntimeResidencyCohortError(
                "unpublished model placement epoch has adaptive fractions"
            )
        if self.invalidation_reason is not None:
            _text(
                "model placement epoch invalidation reason",
                self.invalidation_reason,
            )
        body = {
            "active_request_count": self.active_request_count,
            "artifact_sha256": artifact_sha256,
            "component_opportunity_counts": counts,
            "generation": self.generation,
            "schema": (
                "research-scheduler-model-placement-epoch-v4"
                if published
                else "research-scheduler-model-placement-epoch-v1"
            ),
            "virtual_queue_request_count": (
                self.virtual_queue_request_count
            ),
        }
        if published:
            body["publication"] = {
                "allowed_adaptive_fractions_ppm": list(fractions),
                "break_even_use_count": self.break_even_use_count,
                "capability_generation_sha256": (
                    self.capability_generation_sha256
                ),
                "expected_reuse_count": self.expected_reuse_count,
                "demand_generation_sha256": (
                    self.demand_generation_sha256
                ),
                "input_token_bucket": self.input_token_bucket,
                "invalidation_reason": self.invalidation_reason,
                "learning_generation_sha256": (
                    self.learning_generation_sha256
                ),
                "maximum_latency_ppm": self.maximum_latency_ppm,
                "objective": self.objective,
                "old_component_identity_sha256": (
                    self.old_component_identity_sha256
                ),
                "output_token_bucket": self.output_token_bucket,
                "profile_generation_sha256": (
                    self.profile_generation_sha256
                ),
                "phone_layout_generation": self.phone_layout_generation,
                "pressure_bucket": self.pressure_bucket,
                "quality_requirement": self.quality_requirement,
                "route_template_identity_sha256": (
                    self.route_template_identity_sha256
                ),
                "selected_component_identity_sha256": (
                    self.selected_component_identity_sha256
                ),
                "selected_desktop_parent_plan_sha256": (
                    self.selected_desktop_parent_plan_sha256
                ),
                "selected_desktop_parent_placement_sha256": (
                    self.selected_desktop_parent_placement_sha256
                ),
                "selected_desktop_parent_route_id": (
                    self.selected_desktop_parent_route_id
                ),
                "selected_executor_id": self.selected_executor_id,
                "selected_operator_plan_sha256": (
                    self.selected_operator_plan_sha256
                ),
                "selected_route_id": self.selected_route_id,
                "selected_resident_artifact_sha256s": list(
                    resident_artifacts
                ),
                "selected_resident_shard_geometry_sha256s": list(
                    shard_geometries
                ),
                "selected_session_resource_ids": list(
                    session_resources
                ),
                "selected_transition_energy_uj": (
                    self.selected_transition_energy_uj
                ),
                "selected_transition_latency_us": (
                    self.selected_transition_latency_us
                ),
                "transport_generation_sha256": (
                    self.transport_generation_sha256
                ),
                "trigger_reasons": list(reasons),
                "valid_from_us": self.valid_from_us,
                "valid_until_us": self.valid_until_us,
            }
        expected = canonical_sha256(body)
        if self.epoch_sha256 and self.epoch_sha256 != expected:
            raise RuntimeResidencyCohortError(
                "model placement epoch hash differs"
            )
        object.__setattr__(
            self,
            "component_opportunity_counts",
            MappingProxyType(counts),
        )
        object.__setattr__(
            self, "reuse_projections", MappingProxyType(projections)
        )
        object.__setattr__(
            self, "allowed_adaptive_fractions_ppm", fractions
        )
        if published:
            object.__setattr__(self, "trigger_reasons", reasons)
            object.__setattr__(
                self,
                "selected_resident_artifact_sha256s",
                resident_artifacts,
            )
            object.__setattr__(
                self,
                "selected_resident_shard_geometry_sha256s",
                shard_geometries,
            )
            object.__setattr__(
                self,
                "selected_session_resource_ids",
                session_resources,
            )
        object.__setattr__(self, "epoch_sha256", expected)

    @property
    def published(self) -> bool:
        return self.selected_route_id is not None

    def to_json(self) -> dict[str, object]:
        result = {
            "active_request_count": self.active_request_count,
            "artifact_sha256": self.artifact_sha256,
            "component_opportunity_counts": dict(
                self.component_opportunity_counts
            ),
            "epoch_sha256": self.epoch_sha256,
            "generation": self.generation,
            "observed_at_us": self.observed_at_us,
            "reuse_projections": {
                component_id: projection.to_json()
                for component_id, projection
                in self.reuse_projections.items()
            },
            "schema": (
                "research-scheduler-model-placement-epoch-v4"
                if self.published
                else "research-scheduler-model-placement-epoch-v1"
            ),
            "virtual_queue_request_count": (
                self.virtual_queue_request_count
            ),
        }
        if self.published:
            result.update({
                "allowed_adaptive_fractions_ppm": list(
                    self.allowed_adaptive_fractions_ppm
                ),
                "break_even_use_count": self.break_even_use_count,
                "capability_generation_sha256": (
                    self.capability_generation_sha256
                ),
                "expected_reuse_count": self.expected_reuse_count,
                "demand_generation_sha256": (
                    self.demand_generation_sha256
                ),
                "input_token_bucket": self.input_token_bucket,
                "invalidation_reason": self.invalidation_reason,
                "learning_generation_sha256": (
                    self.learning_generation_sha256
                ),
                "maximum_latency_ppm": self.maximum_latency_ppm,
                "objective": self.objective,
                "old_component_identity_sha256": (
                    self.old_component_identity_sha256
                ),
                "output_token_bucket": self.output_token_bucket,
                "profile_generation_sha256": (
                    self.profile_generation_sha256
                ),
                "phone_layout_generation": self.phone_layout_generation,
                "pressure_bucket": self.pressure_bucket,
                "quality_requirement": self.quality_requirement,
                "route_template_identity_sha256": (
                    self.route_template_identity_sha256
                ),
                "selected_component_identity_sha256": (
                    self.selected_component_identity_sha256
                ),
                "selected_desktop_parent_plan_sha256": (
                    self.selected_desktop_parent_plan_sha256
                ),
                "selected_desktop_parent_placement_sha256": (
                    self.selected_desktop_parent_placement_sha256
                ),
                "selected_desktop_parent_route_id": (
                    self.selected_desktop_parent_route_id
                ),
                "selected_executor_id": self.selected_executor_id,
                "selected_operator_plan_sha256": (
                    self.selected_operator_plan_sha256
                ),
                "selected_route_id": self.selected_route_id,
                "selected_resident_artifact_sha256s": list(
                    self.selected_resident_artifact_sha256s
                ),
                "selected_resident_shard_geometry_sha256s": list(
                    self.selected_resident_shard_geometry_sha256s
                ),
                "selected_session_resource_ids": list(
                    self.selected_session_resource_ids
                ),
                "selected_transition_energy_uj": (
                    self.selected_transition_energy_uj
                ),
                "selected_transition_latency_us": (
                    self.selected_transition_latency_us
                ),
                "transport_generation_sha256": (
                    self.transport_generation_sha256
                ),
                "trigger_reasons": list(self.trigger_reasons),
                "valid_from_us": self.valid_from_us,
                "valid_until_us": self.valid_until_us,
            })
        return result


@dataclass(frozen=True)
class _CohortCheckpoint:
    arrivals: tuple[tuple[str, tuple[int, ...]], ...]
    components: tuple[tuple[str, RuntimeResidencyComponentIdentity], ...]
    request_components: tuple[tuple[str, str], ...]
    request_artifacts: tuple[tuple[str, str], ...]
    planning_arrivals: tuple[tuple[str, tuple[int, ...]], ...]
    planning_request_components: tuple[
        tuple[str, tuple[str, ...]], ...
    ]
    planning_request_artifacts: tuple[tuple[str, str], ...]
    generation_by_artifact: tuple[tuple[str, int], ...]
    published_epochs: tuple[
        tuple[tuple[object, ...], RuntimeModelPlacementEpoch], ...
    ]
    published_generation_by_key: tuple[
        tuple[tuple[object, ...], int], ...
    ]
    confirmed_component_by_resource: tuple[tuple[str, str], ...]


class RuntimeResidencyCohortTracker:
    """Track only observed arrivals and derive bounded switch hysteresis."""

    def __init__(
        self, policy: RuntimeResidencyCohortPolicy | None = None
    ) -> None:
        self.policy = policy or RuntimeResidencyCohortPolicy()
        self._arrivals: dict[str, tuple[int, ...]] = {}
        self._components: dict[
            str, RuntimeResidencyComponentIdentity
        ] = {}
        self._request_components: dict[str, str] = {}
        self._request_artifacts: dict[str, str] = {}
        self._planning_arrivals: dict[str, tuple[int, ...]] = {}
        self._planning_request_components: dict[
            str, tuple[str, ...]
        ] = {}
        self._planning_request_artifacts: dict[str, str] = {}
        self._generation_by_artifact: dict[str, int] = {}
        self._published_epochs: dict[
            tuple[object, ...], RuntimeModelPlacementEpoch
        ] = {}
        self._published_generation_by_key: dict[
            tuple[object, ...], int
        ] = {}
        self._confirmed_component_by_resource: dict[str, str] = {}

    def checkpoint(self) -> object:
        return _CohortCheckpoint(
            arrivals=tuple(sorted(self._arrivals.items())),
            components=tuple(sorted(self._components.items())),
            request_components=tuple(sorted(
                self._request_components.items()
            )),
            request_artifacts=tuple(sorted(
                self._request_artifacts.items()
            )),
            planning_arrivals=tuple(sorted(
                self._planning_arrivals.items()
            )),
            planning_request_components=tuple(sorted(
                self._planning_request_components.items()
            )),
            planning_request_artifacts=tuple(sorted(
                self._planning_request_artifacts.items()
            )),
            generation_by_artifact=tuple(sorted(
                self._generation_by_artifact.items()
            )),
            published_epochs=tuple(sorted(
                self._published_epochs.items()
            )),
            published_generation_by_key=tuple(sorted(
                self._published_generation_by_key.items()
            )),
            confirmed_component_by_resource=tuple(sorted(
                self._confirmed_component_by_resource.items()
            )),
        )

    def restore(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, _CohortCheckpoint):
            raise RuntimeResidencyCohortError(
                "residency cohort checkpoint is invalid"
            )
        self._arrivals = dict(checkpoint.arrivals)
        self._components = dict(checkpoint.components)
        self._request_components = dict(checkpoint.request_components)
        self._request_artifacts = dict(checkpoint.request_artifacts)
        self._planning_arrivals = dict(checkpoint.planning_arrivals)
        self._planning_request_components = dict(
            checkpoint.planning_request_components
        )
        self._planning_request_artifacts = dict(
            checkpoint.planning_request_artifacts
        )
        self._generation_by_artifact = dict(
            checkpoint.generation_by_artifact
        )
        self._published_epochs = dict(checkpoint.published_epochs)
        self._published_generation_by_key = dict(
            checkpoint.published_generation_by_key
        )
        self._confirmed_component_by_resource = dict(
            checkpoint.confirmed_component_by_resource
        )

    def _advance_generation(self, artifact_sha256: str) -> None:
        self._generation_by_artifact[artifact_sha256] = (
            self._generation_by_artifact.get(artifact_sha256, 0) + 1
        )

    def record_planning_arrival(
        self,
        request_id: str,
        components: Sequence[RuntimeResidencyComponentIdentity],
        arrival_us: int,
    ) -> None:
        """Record feasible resident components without selecting one."""
        request_id = _text("placement epoch request", request_id)
        rows = tuple(components)
        if (
            not rows
            or any(
                not isinstance(row, RuntimeResidencyComponentIdentity)
                for row in rows
            )
            or type(arrival_us) is not int
            or arrival_us < 0
        ):
            raise RuntimeResidencyCohortError(
                "placement epoch arrival is invalid"
            )
        artifacts = set(rows[0].resident_artifact_sha256s)
        for row in rows[1:]:
            artifacts.intersection_update(
                row.resident_artifact_sha256s
            )
        if len(artifacts) != 1:
            raise RuntimeResidencyCohortError(
                "placement epoch spans multiple artifacts"
            )
        request_artifact = next(iter(artifacts))
        component_ids = tuple(sorted({
            row.identity_sha256 for row in rows
        }))
        previous = self._planning_request_components.get(request_id)
        if previous is not None:
            if (
                previous != component_ids
                or self._planning_request_artifacts.get(request_id)
                    != request_artifact
            ):
                raise RuntimeResidencyCohortError(
                    "placement epoch request identity differs"
                )
            return
        for component in rows:
            known = self._components.get(component.identity_sha256)
            if known is not None and known != component:
                raise RuntimeResidencyCohortError(
                    "placement epoch component hash collides"
                )
            self._components[component.identity_sha256] = component
        for component_id in component_ids:
            history = self._planning_arrivals.get(component_id, ())
            self._planning_arrivals[component_id] = tuple(sorted(
                history + (arrival_us,)
            ))[-self.policy.history_limit:]
        self._planning_request_components[request_id] = component_ids
        self._planning_request_artifacts[request_id] = request_artifact
        self._advance_generation(request_artifact)

    def record_arrival(
        self,
        request_id: str,
        component: RuntimeResidencyComponentIdentity,
        arrival_us: int,
        artifact_sha256: str | None = None,
    ) -> None:
        request_id = _text("residency cohort request", request_id)
        if not isinstance(component, RuntimeResidencyComponentIdentity):
            raise RuntimeResidencyCohortError(
                "residency cohort component is invalid"
            )
        if (
            type(arrival_us) is not int
            or arrival_us < 0
        ):
            raise RuntimeResidencyCohortError(
                "residency cohort arrival is invalid"
            )
        component_id = component.identity_sha256
        if artifact_sha256 is None:
            if len(component.resident_artifact_sha256s) != 1:
                raise RuntimeResidencyCohortError(
                    "residency cohort request artifact is ambiguous"
                )
            artifact_sha256 = component.resident_artifact_sha256s[0]
        artifact_sha256 = _text(
            "residency cohort request artifact", artifact_sha256
        )
        if not component.covers_artifact(artifact_sha256):
            raise RuntimeResidencyCohortError(
                "residency cohort component does not cover request"
            )
        previous = self._request_components.get(request_id)
        if previous is not None:
            if (
                previous != component_id
                or self._request_artifacts.get(request_id)
                    != artifact_sha256
            ):
                raise RuntimeResidencyCohortError(
                    "residency cohort request identity differs"
                )
            return
        known = self._components.get(component_id)
        if known is not None and known != component:
            raise RuntimeResidencyCohortError(
                "residency cohort component hash collides"
            )
        rows = self._arrivals.get(component_id, ())
        rows = tuple(sorted(rows + (arrival_us,)))[-self.policy.history_limit:]
        self._arrivals[component_id] = rows
        self._components[component_id] = component
        self._request_components[request_id] = component_id
        self._request_artifacts[request_id] = artifact_sha256
        self._advance_generation(artifact_sha256)

    @staticmethod
    def _session_replacement_resources(
        catalog: RuntimeCapabilityCatalog,
    ) -> tuple[dict[str, str], dict[str, object]]:
        group_by_memory_resource: dict[str, str] = {}
        session_by_memory_resource: dict[str, object] = {}
        for executor in catalog.executors:
            for session in executor.phone_sessions:
                replacement_resource = executor.exclusive_residency_resource_id
                if replacement_resource == "residency:whole:" + executor.executor_id:
                    replacement_resource = session.shared_compute_resource_id
                session_by_memory_resource[
                    session.memory_resource_id
                ] = session
                if replacement_resource is not None:
                    group_by_memory_resource[
                        session.memory_resource_id
                    ] = replacement_resource
        return group_by_memory_resource, session_by_memory_resource

    def confirm_resident_component(
        self,
        component: RuntimeResidencyComponentIdentity,
        catalog: RuntimeCapabilityCatalog,
    ) -> tuple[str, ...]:
        """Confirm a phone component only after successful physical execution."""
        if not isinstance(component, RuntimeResidencyComponentIdentity):
            raise RuntimeResidencyCohortError(
                "confirmed residency component is invalid"
            )
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise RuntimeResidencyCohortError(
                "confirmed residency catalog is invalid"
            )
        if not component.session_resource_ids:
            return ()
        group_by_memory, _ = self._session_replacement_resources(catalog)
        if any(
            resource_id not in group_by_memory
            for resource_id in component.session_resource_ids
        ):
            return ()
        resource_ids = tuple(sorted({
            group_by_memory[resource_id]
            for resource_id in component.session_resource_ids
        }))
        known = self._components.get(component.identity_sha256)
        if known is not None and known != component:
            raise RuntimeResidencyCohortError(
                "confirmed residency component hash collides"
            )
        self._components[component.identity_sha256] = component
        changed = False
        for resource_id in resource_ids:
            if self._confirmed_component_by_resource.get(resource_id) == (
                component.identity_sha256
            ):
                continue
            self._confirmed_component_by_resource[resource_id] = (
                component.identity_sha256
            )
            changed = True
        if changed:
            for artifact_sha256 in component.resident_artifact_sha256s:
                self._advance_generation(artifact_sha256)
        return resource_ids

    def confirmed_resident_components(
        self,
        catalog: RuntimeCapabilityCatalog,
        snapshot: HeterogeneousRuntimeSnapshot,
    ) -> Mapping[str, RuntimeResidencyComponentIdentity]:
        """Return physically confirmed components by replacement domain."""
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise RuntimeResidencyCohortError(
                "resident component catalog is invalid"
            )
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise RuntimeResidencyCohortError(
                "resident component snapshot is invalid"
            )
        group_by_memory, session_by_memory = (
            self._session_replacement_resources(catalog)
        )
        executor_devices_by_group: dict[str, set[str]] = {}
        for executor in catalog.executors:
            for group in {group_by_memory.get(session.memory_resource_id)
                          for session in executor.phone_sessions} - {None}:
                executor_devices_by_group.setdefault(group, set()).add(
                    executor.device_id
                )
        result: dict[str, RuntimeResidencyComponentIdentity] = {}
        for resource_id, component_id in (
            self._confirmed_component_by_resource.items()
        ):
            component = self._components.get(component_id)
            if component is None or any(
                group_by_memory.get(session_resource_id) != resource_id
                for session_resource_id in component.session_resource_ids
            ):
                continue
            sessions = tuple(
                session_by_memory[session_resource_id]
                for session_resource_id in component.session_resource_ids
                if session_resource_id in session_by_memory
            )
            if len(sessions) != len(component.session_resource_ids) or any(
                not session.ready for session in sessions
            ):
                continue
            explicit_artifacts = {
                session.resident_artifact_sha256
                for session in sessions
                if session.resident_artifact_sha256 is not None
            }
            if explicit_artifacts and not explicit_artifacts.issubset(
                component.resident_artifact_sha256s
            ):
                continue
            observed_artifacts = {
                row.artifact_sha256
                for row in snapshot.residency
                if row.device_id in executor_devices_by_group.get(
                    resource_id, set()
                )
                and catalog.residency_group(row.executor_id, row.device_id) == resource_id
                and row.state in {"hot", "warm"}
                and row.resident_bytes > 0
            }
            if observed_artifacts and not observed_artifacts.issubset(
                component.resident_artifact_sha256s
            ):
                continue
            result[resource_id] = component
        return MappingProxyType(dict(sorted(result.items())))

    def planned_component_identity_sha256s(
        self, request_id: str
    ) -> tuple[str, ...]:
        request_id = _text("placement epoch request", request_id)
        return self._planning_request_components.get(request_id, ())

    def record_terminal(self, request_id: str) -> None:
        request_id = _text("residency cohort request", request_id)
        self._request_components.pop(request_id, None)
        selected_artifact = self._request_artifacts.pop(
            request_id, None
        )
        self._planning_request_components.pop(request_id, ())
        planned_artifact = self._planning_request_artifacts.pop(
            request_id, None
        )
        if (
            selected_artifact is not None
            and planned_artifact is not None
            and selected_artifact != planned_artifact
        ):
            raise RuntimeResidencyCohortError(
                "residency terminal request artifact differs"
            )
        artifact_sha256 = selected_artifact or planned_artifact
        if artifact_sha256 is not None:
            self._advance_generation(artifact_sha256)

    def _predicted_reuse_us(
        self, component_identity_sha256: str, observed_at_us: int
    ) -> int | None:
        rows = self._arrivals.get(component_identity_sha256, ())
        intervals = tuple(
            right - left for left, right in zip(rows, rows[1:])
            if right > left
        )
        if len(intervals) < self.policy.minimum_interarrival_samples:
            return None
        predicted = rows[-1] + median_low(intervals)
        if not (
            observed_at_us < predicted
            <= observed_at_us + self.policy.reuse_horizon_us
        ):
            return None
        return predicted

    @staticmethod
    def _replacement_cost_us(
        catalog: RuntimeCapabilityCatalog,
        device_id: str,
        artifact_sha256: str,
        resident_bytes: int,
    ) -> int:
        values = []
        for transition in catalog.transitions:
            if (
                device_id not in transition.prepares_device_ids
                or transition.target_state != "hot"
                or transition.artifact_sha256
                    not in {None, artifact_sha256}
            ):
                continue
            latency_us, _ = transition.cost(resident_bytes)
            values.append(latency_us)
        return max(values, default=0)

    def holds(
        self,
        catalog: RuntimeCapabilityCatalog,
        snapshot: HeterogeneousRuntimeSnapshot,
        tickets: Sequence[RuntimeRequestTicket],
        observed_at_us: int,
    ) -> Mapping[str, RuntimeResidencyCohortHold]:
        if not isinstance(catalog, RuntimeCapabilityCatalog):
            raise RuntimeResidencyCohortError(
                "residency cohort catalog is invalid"
            )
        if not isinstance(snapshot, HeterogeneousRuntimeSnapshot):
            raise RuntimeResidencyCohortError(
                "residency cohort snapshot is invalid"
            )
        if (
            type(observed_at_us) is not int
            or observed_at_us < 0
            or any(not isinstance(row, RuntimeRequestTicket) for row in tickets)
        ):
            raise RuntimeResidencyCohortError(
                "residency cohort runtime input is invalid"
            )
        maximum_hysteresis_us = snapshot.cost_features.get(
            "residency_maximum_hysteresis_us",
            self.policy.maximum_hysteresis_us,
        )
        reuse_horizon_us = snapshot.cost_features.get(
            "residency_reuse_horizon_us", self.policy.reuse_horizon_us
        )
        if (
            type(maximum_hysteresis_us) is not int
            or maximum_hysteresis_us < 0
            or type(reuse_horizon_us) is not int
            or reuse_horizon_us < 0
        ):
            raise RuntimeResidencyCohortError(
                "residency cohort runtime policy is invalid"
            )
        result: dict[str, RuntimeResidencyCohortHold] = {}
        for residency in snapshot.residency:
            capability = catalog.executor_by_device.get(
                residency.device_id
            )
            replacement_resource = catalog.residency_group(residency.executor_id, residency.device_id)
            if (
                capability is None
                or replacement_resource is None
                or residency.state not in {"hot", "warm"}
                or residency.resident_bytes <= 0
            ):
                continue
            active = tuple(
                (
                    ticket,
                    runtime_residency_component_identity(
                        ticket.model.artifact_sha256,
                        ticket.execution_plan,
                        ticket.binding,
                    ),
                )
                for ticket in tickets
                if ticket.dispatch_state in {
                    "ACQUIRED", "QUEUED", "REPLAN_REQUIRED"
                }
                and ticket.lease_status != "RELEASED_PENDING_RECEIPT"
                and ticket.model.artifact_sha256
                    == residency.artifact_sha256
                and ticket.execution_plan is not None
                and residency.device_id in ticket.execution_plan.device_ids
            )
            active_by_component: dict[
                str,
                list[tuple[
                    RuntimeRequestTicket,
                    RuntimeResidencyComponentIdentity,
                ]],
            ] = {}
            for ticket, component in active:
                active_by_component.setdefault(
                    component.identity_sha256, []
                ).append((ticket, component))
            component_by_id = {
                component_id: component_active[0][1]
                for component_id, component_active
                in active_by_component.items()
            }
            expected_executor_id = (
                capability.executor_id
                if residency.executor_id is None
                else residency.executor_id
            )
            expected_composite = catalog.composite_executor_by_id.get(
                expected_executor_id
            )
            expected_endpoint = (
                expected_composite.endpoint
                if expected_composite is not None
                else capability.endpoint
            )
            for component_id, component in self._components.items():
                expected_residency_scope = (
                    expected_endpoint
                    if component.resident_shard_geometry_sha256
                    or component.session_resource_ids
                    else expected_executor_id
                )
                if (
                    not component.covers_artifact(
                        residency.artifact_sha256
                    )
                    or component.resident_endpoint
                        != expected_residency_scope
                    or (
                        residency.resident_geometry_sha256 is None
                        and component.resident_shard_geometry_sha256
                    )
                    or (
                        residency.resident_geometry_sha256 is not None
                        and residency.resident_geometry_sha256 not in (
                            component.resident_shard_geometry_sha256
                        )
                    )
                ):
                    continue
                component_by_id.setdefault(component_id, component)
                active_by_component.setdefault(component_id, [])
            for component_id, component in component_by_id.items():
                component_active = active_by_component[component_id]
                active_request_ids = {
                    ticket.request.request_id
                    for ticket, _active_component in component_active
                }
                component_active.extend(
                    (ticket, component)
                    for ticket in tickets
                    if ticket.dispatch_state in {
                        "QUEUED", "REPLAN_REQUIRED"
                    }
                    and ticket.model.artifact_sha256
                        == residency.artifact_sha256
                    and ticket.request.request_id
                        not in active_request_ids
                    and component_id in self._planning_request_components.get(
                        ticket.request.request_id, ()
                    )
                )
                predicted_reuse_us = self._predicted_reuse_us(
                    component_id, observed_at_us
                )
                if predicted_reuse_us is not None and (
                    predicted_reuse_us
                        > observed_at_us + reuse_horizon_us
                ):
                    predicted_reuse_us = None
                replacement_cost_us = min(
                    maximum_hysteresis_us,
                    self._replacement_cost_us(
                        catalog,
                        residency.device_id,
                        residency.artifact_sha256,
                        residency.resident_bytes,
                    ),
                )
                if (
                    predicted_reuse_us is None
                    and len(component_active) < 2
                ):
                    replacement_cost_us = 0
                active_until_us = max(
                    (
                        ticket.decision.finish_upper_us
                        for ticket, _component in component_active
                    ),
                    default=observed_at_us,
                )
                hold_until_us = min(
                    observed_at_us + reuse_horizon_us,
                    max(
                        active_until_us,
                        predicted_reuse_us or observed_at_us,
                    ) + replacement_cost_us,
                )
                if hold_until_us <= observed_at_us:
                    continue
                hold = RuntimeResidencyCohortHold(
                    resource_id=(
                        replacement_resource
                    ),
                    model_id=residency.model_id,
                    artifact_sha256=residency.artifact_sha256,
                    hold_until_us=hold_until_us,
                    active_request_count=len(component_active),
                    predicted_reuse_us=predicted_reuse_us,
                    replacement_cost_us=replacement_cost_us,
                    component_identity_sha256=(
                        component.identity_sha256
                    ),
                )
                previous = result.get(hold.resource_id)
                if previous is None or (
                    hold.hold_until_us,
                    hold.component_identity_sha256,
                ) > (
                    previous.hold_until_us,
                    previous.component_identity_sha256,
                ):
                    result[hold.resource_id] = hold
        return MappingProxyType(dict(sorted(result.items())))

    def reuse_projections(
        self,
        artifact_sha256: str,
        observed_at_us: int,
        *,
        request_id: str | None = None,
        queued_request_ids: Sequence[str] = (),
    ) -> Mapping[str, RuntimeResidencyReuseProjection]:
        artifact_sha256 = _text(
            "residency reuse artifact", artifact_sha256
        )
        if type(observed_at_us) is not int or observed_at_us < 0:
            raise RuntimeResidencyCohortError(
                "residency reuse observation time is invalid"
            )
        if request_id is not None:
            request_id = _text("residency reuse request", request_id)
            existing_id = self._request_components.get(request_id)
            existing = (
                None
                if existing_id is None
                else self._components[existing_id]
            )
            if (
                existing is not None
                and (
                    not existing.covers_artifact(artifact_sha256)
                    or self._request_artifacts.get(request_id)
                        != artifact_sha256
                )
            ):
                raise RuntimeResidencyCohortError(
                    "residency reuse request identity differs"
                )
        queued_ids = frozenset(
            _text("residency reuse queued request", value)
            for value in queued_request_ids
        )
        result = {}
        for component_id, component in self._components.items():
            if not component.covers_artifact(artifact_sha256):
                continue
            selected_request_ids = {
                active_request_id
                for active_request_id, value
                in self._request_components.items()
                if active_request_id != request_id
                and value == component_id
                and self._request_artifacts.get(active_request_id)
                    == artifact_sha256
            }
            queued_compatible_ids = {
                active_request_id
                for active_request_id in queued_ids
                if active_request_id != request_id
                and active_request_id not in selected_request_ids
                and component_id in self._planning_request_components.get(
                    active_request_id, ()
                )
                and self._planning_request_artifacts.get(
                    active_request_id
                ) == artifact_sha256
            }
            observed = (
                1
                + len(selected_request_ids)
                + len(queued_compatible_ids)
            )
            selected_prediction = self._predicted_reuse_us(
                component_id, observed_at_us
            )
            planning_rows = self._planning_arrivals.get(component_id, ())
            planning_intervals = tuple(
                right - left
                for left, right in zip(planning_rows, planning_rows[1:])
                if right > left
            )
            planning_prediction = None
            if (
                len(planning_intervals)
                    >= self.policy.minimum_interarrival_samples
            ):
                value = planning_rows[-1] + median_low(
                    planning_intervals
                )
                if (
                    observed_at_us < value
                    <= observed_at_us + self.policy.reuse_horizon_us
                ):
                    planning_prediction = value
            predicted_reuse_us = (
                planning_prediction
                if selected_prediction is None
                else selected_prediction
                if planning_prediction is None
                else min(selected_prediction, planning_prediction)
            )
            result[component_id] = RuntimeResidencyReuseProjection(
                artifact_sha256=artifact_sha256,
                component_identity_sha256=component_id,
                observed_request_count=observed,
                predicted_additional_uses=(
                    0 if predicted_reuse_us is None else 1
                ),
                horizon_us=(
                    0 if predicted_reuse_us is None
                    else predicted_reuse_us - observed_at_us
                ),
            )
        return MappingProxyType(dict(sorted(result.items())))

    def model_placement_epoch(
        self,
        artifact_sha256: str,
        observed_at_us: int,
        *,
        request_id: str | None = None,
        queued_request_ids: Sequence[str] = (),
    ) -> RuntimeModelPlacementEpoch:
        queued_ids = tuple(sorted({
            _text("placement epoch queued request", value)
            for value in queued_request_ids
        }))
        projections = self.reuse_projections(
            artifact_sha256,
            observed_at_us,
            request_id=request_id,
            queued_request_ids=queued_ids,
        )
        opportunity_counts = {
            component_id: len(arrivals)
            for component_id, arrivals in self._planning_arrivals.items()
            if self._components[component_id].covers_artifact(
                artifact_sha256
            )
        }
        return RuntimeModelPlacementEpoch(
            artifact_sha256=artifact_sha256,
            generation=self._generation_by_artifact.get(
                artifact_sha256, 0
            ),
            observed_at_us=observed_at_us,
            active_request_count=sum(
                self._request_artifacts.get(request_id)
                    == artifact_sha256
                for request_id in self._request_components
            ),
            virtual_queue_request_count=sum(
                self._planning_request_artifacts.get(queued_request_id)
                    == artifact_sha256
                for queued_request_id in queued_ids
            ),
            component_opportunity_counts=opportunity_counts,
            reuse_projections=projections,
        )

    @staticmethod
    def _published_epoch_key(
        artifact_sha256: str,
        input_token_bucket: int,
        output_token_bucket: int,
        quality_requirement: str,
        objective: str,
        maximum_latency_ppm: int,
    ) -> tuple[object, ...]:
        artifact_sha256 = _text(
            "published placement artifact", artifact_sha256
        )
        quality_requirement = _text(
            "published placement quality", quality_requirement
        )
        objective = _text("published placement objective", objective)
        if (
            type(input_token_bucket) is not int
            or input_token_bucket < 1
            or type(output_token_bucket) is not int
            or output_token_bucket < 1
            or type(maximum_latency_ppm) is not int
            or maximum_latency_ppm < 1
        ):
            raise RuntimeResidencyCohortError(
                "published placement key is invalid"
            )
        return (
            artifact_sha256,
            input_token_bucket,
            output_token_bucket,
            quality_requirement,
            objective,
            maximum_latency_ppm,
        )

    def next_published_generation(
        self,
        artifact_sha256: str,
        input_token_bucket: int,
        output_token_bucket: int,
        quality_requirement: str,
        objective: str,
        maximum_latency_ppm: int,
    ) -> int:
        key = self._published_epoch_key(
            artifact_sha256,
            input_token_bucket,
            output_token_bucket,
            quality_requirement,
            objective,
            maximum_latency_ppm,
        )
        return self._published_generation_by_key.get(key, 0) + 1

    def publish_model_placement_epoch(
        self, epoch: RuntimeModelPlacementEpoch
    ) -> None:
        if not isinstance(epoch, RuntimeModelPlacementEpoch) or not (
            epoch.published
        ):
            raise RuntimeResidencyCohortError(
                "published model placement epoch is invalid"
            )
        key = self._published_epoch_key(
            epoch.artifact_sha256,
            epoch.input_token_bucket,
            epoch.output_token_bucket,
            epoch.quality_requirement,
            epoch.objective,
            epoch.maximum_latency_ppm,
        )
        expected_generation = (
            self._published_generation_by_key.get(key, 0) + 1
        )
        if epoch.generation != expected_generation:
            raise RuntimeResidencyCohortError(
                "published model placement generation differs"
            )
        self._published_epochs[key] = epoch
        self._published_generation_by_key[key] = epoch.generation

    def published_model_placement_epoch(
        self,
        artifact_sha256: str,
        input_token_bucket: int,
        output_token_bucket: int,
        quality_requirement: str,
        objective: str,
        maximum_latency_ppm: int,
    ) -> RuntimeModelPlacementEpoch | None:
        key = self._published_epoch_key(
            artifact_sha256,
            input_token_bucket,
            output_token_bucket,
            quality_requirement,
            objective,
            maximum_latency_ppm,
        )
        return self._published_epochs.get(key)

    def invalidate_model_placement_epoch(
        self, epoch: RuntimeModelPlacementEpoch
    ) -> None:
        if not isinstance(epoch, RuntimeModelPlacementEpoch) or not (
            epoch.published
        ):
            raise RuntimeResidencyCohortError(
                "model placement invalidation epoch is invalid"
            )
        key = self._published_epoch_key(
            epoch.artifact_sha256,
            epoch.input_token_bucket,
            epoch.output_token_bucket,
            epoch.quality_requirement,
            epoch.objective,
            epoch.maximum_latency_ppm,
        )
        current = self._published_epochs.get(key)
        if current is None:
            return
        if current.epoch_sha256 != epoch.epoch_sha256:
            raise RuntimeResidencyCohortError(
                "model placement invalidation epoch differs"
            )
        del self._published_epochs[key]

    def invalidate_model_placement_epoch_sha256(
        self, epoch_sha256: str
    ) -> None:
        epoch_sha256 = _text(
            "model placement invalidation hash", epoch_sha256
        )
        matches = tuple(
            (key, epoch)
            for key, epoch in self._published_epochs.items()
            if epoch.epoch_sha256 == epoch_sha256
        )
        if len(matches) > 1:
            raise RuntimeResidencyCohortError(
                "model placement invalidation hash is ambiguous"
            )
        if matches:
            del self._published_epochs[matches[0][0]]

    def compatible_published_model_placement_epochs(
        self,
        artifact_sha256: str,
        quality_requirement: str,
        objective: str,
        maximum_latency_ppm: int,
    ) -> tuple[RuntimeModelPlacementEpoch, ...]:
        artifact_sha256 = _text(
            "published placement artifact", artifact_sha256
        )
        quality_requirement = _text(
            "published placement quality", quality_requirement
        )
        objective = _text("published placement objective", objective)
        if (
            type(maximum_latency_ppm) is not int
            or maximum_latency_ppm < 1
        ):
            raise RuntimeResidencyCohortError(
                "published placement key is invalid"
            )
        rows = (
            epoch for epoch in self._published_epochs.values()
            if epoch.artifact_sha256 == artifact_sha256
            and epoch.quality_requirement == quality_requirement
            and epoch.objective == objective
            and epoch.maximum_latency_ppm == maximum_latency_ppm
        )
        return tuple(sorted(
            rows,
            key=lambda epoch: (
                epoch.observed_at_us,
                epoch.generation,
                epoch.epoch_sha256,
            ),
            reverse=True,
        ))

    def published_model_placement_epoch_hashes(self) -> frozenset[str]:
        return frozenset(
            epoch.epoch_sha256 for epoch in self._published_epochs.values()
        )

    def snapshot(self) -> Mapping[str, object]:
        return MappingProxyType({
            "active_request_count": len(self._request_components),
            "arrival_count_by_artifact": {
                artifact_sha256: sum(
                    len(self._arrivals.get(component_id, ()))
                    for component_id, component in self._components.items()
                    if component.covers_artifact(artifact_sha256)
                )
                for artifact_sha256 in sorted({
                    artifact_sha256
                    for component in self._components.values()
                    for artifact_sha256
                    in component.resident_artifact_sha256s
                })
            },
            "arrival_count_by_component": {
                component_id: len(rows)
                for component_id, rows in sorted(self._arrivals.items())
            },
            "model_placement_generation": dict(sorted(
                self._generation_by_artifact.items()
            )),
            "published_model_placement_epochs": {
                epoch.epoch_sha256: epoch.to_json()
                for epoch in sorted(
                    self._published_epochs.values(),
                    key=lambda row: row.epoch_sha256,
                )
            },
            "planning_arrival_count_by_component": {
                component_id: len(rows)
                for component_id, rows
                in sorted(self._planning_arrivals.items())
            },
            "virtual_queue_request_count": len(
                self._planning_request_components
            ),
        })
