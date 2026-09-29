"""Disjoint FFN residency placement across phone sessions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
import threading
from types import MappingProxyType
from typing import Mapping, Sequence

from .model_manifest import (
    ModelManifest,
    ModelOperatorManifest,
    ModelTensorManifest,
)
from .runtime_capabilities import RuntimePhoneSessionCapability
from .types import canonical_sha256


class PhoneShardPlacementError(ValueError):
    pass


_SHARD_SET_CACHE_MAXIMUM = 512
_shard_set_cache: dict[str, tuple[PhoneFfnShardSet, ...]] = {}
_shard_set_cache_lock = threading.Lock()


def _sha256(name: str, value: object) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PhoneShardPlacementError(name + " is invalid")
    return value


@dataclass(frozen=True)
class PhoneFfnShardStorageMetadata:
    """Immutable storage coverage for one session's FFN shard file."""

    parent_artifact_sha256: str
    shard_sha256: str
    path: str
    layer_mask: int
    maximum_columns: int
    session_id: str

    def __post_init__(self) -> None:
        _sha256("phone FFN shard parent", self.parent_artifact_sha256)
        _sha256("phone FFN shard", self.shard_sha256)
        if (
            type(self.path) is not str
            or not self.path
            or not self.path.isascii()
            or type(self.layer_mask) is not int
            or not 0 < self.layer_mask < 1 << 64
            or type(self.maximum_columns) is not int
            or self.maximum_columns <= 0
            or type(self.session_id) is not str
            or not self.session_id
            or not self.session_id.isascii()
        ):
            raise PhoneShardPlacementError(
                "phone FFN shard storage metadata is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "layer_mask": self.layer_mask,
            "maximum_columns": self.maximum_columns,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "path": self.path,
            "session_id": self.session_id,
            "shard_sha256": self.shard_sha256,
        }


def phone_sessions_with_storage_coverage(
    sessions: Sequence[RuntimePhoneSessionCapability],
    stored_by_session: Mapping[str, PhoneFfnShardStorageMetadata] | None,
) -> tuple[RuntimePhoneSessionCapability, ...]:
    """Limit new loads to the slices stored for each physical session."""

    if stored_by_session is None:
        return tuple(sessions)
    rows = []
    for session in sessions:
        stored = stored_by_session.get(session.session_id)
        if stored is None:
            continue
        layer_mask = session.supported_layer_mask & stored.layer_mask
        columns = min(session.maximum_columns, stored.maximum_columns)
        if not layer_mask or columns % session.column_quantum:
            continue
        rows.append(replace(
            session,
            supported_layer_mask=layer_mask,
            maximum_columns=columns,
        ))
    return tuple(rows)


@dataclass(frozen=True)
class PhoneFfnSessionIdentity:
    """Exact execution identity for one physically verified HTP session."""

    session_id: str
    artifact_sha256: str
    resident_geometry_sha256: str
    operator_plan_sha256: str
    session_generation: int

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not str
            or not self.session_id
            or not self.session_id.isascii()
            or type(self.session_generation) is not int
            or self.session_generation < 1
        ):
            raise PhoneShardPlacementError(
                "phone session identity is invalid"
            )
        _sha256("phone session artifact", self.artifact_sha256)
        _sha256(
            "phone session resident geometry",
            self.resident_geometry_sha256,
        )
        _sha256(
            "phone session operator plan", self.operator_plan_sha256
        )

    @property
    def helper_key(self) -> tuple[str, str, str, int]:
        return (
            self.session_id,
            self.artifact_sha256,
            self.resident_geometry_sha256,
            self.session_generation,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "helper_key": list(self.helper_key),
            "operator_plan_sha256": self.operator_plan_sha256,
            "resident_geometry_sha256": (
                self.resident_geometry_sha256
            ),
            "session_generation": self.session_generation,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class PhoneFfnShardPlacement:
    artifact_sha256: str
    session_id: str
    endpoint: str
    memory_resource_id: str
    operator_ids: tuple[str, ...]
    layer_mask: int
    maximum_columns: int
    resident_bytes: int
    resident_geometry_sha256: str
    operator_plan_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "endpoint": self.endpoint,
            "layer_mask": self.layer_mask,
            "maximum_columns": self.maximum_columns,
            "operator_plan_sha256": self.operator_plan_sha256,
            "resident_bytes": self.resident_bytes,
            "resident_geometry_sha256": self.resident_geometry_sha256,
            "session_id": self.session_id,
        }


@dataclass(frozen=True)
class PhoneFfnShardSet:
    shards: tuple[PhoneFfnShardPlacement, ...]
    operator_ids: tuple[str, ...]
    resident_bytes: int
    geometry_sha256: str
    packing_value: int
    packing_value_kind: str
    unavailable_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PhoneFfnResidencyDemand:
    manifest: ModelManifest
    queued_work: int
    maximum_columns: int
    batch_plan: str
    benefit_by_operator: Mapping[str, int] = MappingProxyType({})
    benefit_value_kind: str | None = None
    allowed_operator_ids: tuple[str, ...] | None = None
    transition_energy_uj_by_session: Mapping[str, int] = (
        MappingProxyType({})
    )
    transition_energy_aggregation: str = "sum"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.manifest, ModelManifest)
            or type(self.queued_work) is not int
            or self.queued_work <= 0
            or type(self.maximum_columns) is not int
            or self.maximum_columns <= 0
            or type(self.batch_plan) is not str
            or not self.batch_plan
            or not self.batch_plan.isascii()
            or not isinstance(self.benefit_by_operator, Mapping)
            or not isinstance(
                self.transition_energy_uj_by_session, Mapping
            )
            or self.transition_energy_aggregation not in {
                "shared_phone_union", "sum",
            }
        ):
            raise PhoneShardPlacementError(
                "phone residency demand is invalid"
            )
        benefits = dict(self.benefit_by_operator)
        transition_energy = dict(
            self.transition_energy_uj_by_session
        )
        if any(
            type(operator_id) is not str
            or type(value) is not int
            or value <= 0
            for operator_id, value in benefits.items()
        ):
            raise PhoneShardPlacementError(
                "phone residency demand benefit is invalid"
            )
        if any(
            type(session_id) is not str
            or not session_id
            or not session_id.isascii()
            or type(value) is not int
            or value < 0
            for session_id, value in transition_energy.items()
        ):
            raise PhoneShardPlacementError(
                "phone residency demand transition energy is invalid"
            )
        benefit_value_kind = self.benefit_value_kind
        if benefit_value_kind is None:
            benefit_value_kind = (
                "measured_net_energy_uj"
                if benefits else "rough_compute_ops"
            )
        if (
            type(benefit_value_kind) is not str
            or not benefit_value_kind
            or not benefit_value_kind.isascii()
        ):
            raise PhoneShardPlacementError(
                "phone residency demand benefit identity is invalid"
            )
        allowed = self.allowed_operator_ids
        if allowed is not None:
            allowed = tuple(sorted(allowed))
            if (
                not allowed
                or len(allowed) != len(set(allowed))
                or any(type(value) is not str for value in allowed)
            ):
                raise PhoneShardPlacementError(
                    "phone residency demand operator set is invalid"
                )
        object.__setattr__(
            self,
            "benefit_by_operator",
            MappingProxyType(dict(sorted(benefits.items()))),
        )
        object.__setattr__(
            self, "benefit_value_kind", benefit_value_kind
        )
        object.__setattr__(self, "allowed_operator_ids", allowed)
        object.__setattr__(
            self,
            "transition_energy_uj_by_session",
            MappingProxyType(dict(sorted(transition_energy.items()))),
        )


@dataclass(frozen=True)
class PhoneFfnResidencyLayout:
    shards: tuple[PhoneFfnShardPlacement, ...]
    queued_work_by_artifact: Mapping[str, int]
    queue_benefit_by_artifact: Mapping[str, int]
    queue_benefit_by_session: Mapping[str, int]
    queue_benefit: int
    transition_cost: int
    transition_cost_by_session: Mapping[str, int]
    objective: int
    objective_kind: str
    changed_session_ids: tuple[str, ...]
    geometry_sha256: str
    transition_cost_aggregation: str = "sum"
    session_generation_by_id: Mapping[str, int] = MappingProxyType({})
    replacement_source_identities: tuple[PhoneFfnSessionIdentity, ...] = ()
    replacement_source_resident_bytes_by_session: Mapping[str, int] = (
        MappingProxyType({})
    )

    def __post_init__(self) -> None:
        _initialize_phone_residency_layout(self)

    @property
    def resident_bytes(self) -> int:
        return sum(row.resident_bytes for row in self.shards)

    def packing_value_for_artifact(self, artifact_sha256: str) -> int:
        if artifact_sha256 not in {
            row.artifact_sha256 for row in self.shards
        }:
            raise PhoneShardPlacementError(
                "phone residency artifact is absent"
            )
        return self.queue_benefit_by_artifact.get(artifact_sha256, 0)

    def with_session_generations(
        self, generation_by_id: Mapping[str, int]
    ) -> "PhoneFfnResidencyLayout":
        values = dict(generation_by_id)
        if set(values) != {row.session_id for row in self.shards}:
            raise PhoneShardPlacementError(
                "phone residency session generations are incomplete"
            )
        return replace(self, session_generation_by_id=values)

    def session_identity(
        self, session_id: str
    ) -> PhoneFfnSessionIdentity:
        shard = next((
            row for row in self.shards if row.session_id == session_id
        ), None)
        generation = self.session_generation_by_id.get(session_id)
        if shard is None or generation is None:
            raise PhoneShardPlacementError(
                "phone residency session identity is absent"
            )
        return PhoneFfnSessionIdentity(
            session_id=shard.session_id,
            artifact_sha256=shard.artifact_sha256,
            resident_geometry_sha256=shard.resident_geometry_sha256,
            operator_plan_sha256=shard.operator_plan_sha256,
            session_generation=generation,
        )

    @property
    def session_identities(self) -> tuple[PhoneFfnSessionIdentity, ...]:
        return tuple(
            self.session_identity(row.session_id)
            for row in self.shards
        )

    def to_json(self) -> dict[str, object]:
        return {
            "changed_session_ids": list(self.changed_session_ids),
            "geometry_sha256": self.geometry_sha256,
            "objective": self.objective,
            "objective_kind": self.objective_kind,
            "queue_benefit": self.queue_benefit,
            "queue_benefit_by_artifact": dict(
                self.queue_benefit_by_artifact
            ),
            "queue_benefit_by_session": dict(
                self.queue_benefit_by_session
            ),
            "queued_work_by_artifact": dict(
                self.queued_work_by_artifact
            ),
            "resident_bytes": self.resident_bytes,
            "replacement_source_identities": [
                row.to_json() for row in self.replacement_source_identities
            ],
            "replacement_source_resident_bytes_by_session": dict(
                self.replacement_source_resident_bytes_by_session
            ),
            "session_generation_by_id": dict(
                self.session_generation_by_id
            ),
            "shards": [
                {
                    "artifact_sha256": row.artifact_sha256,
                    "endpoint": row.endpoint,
                    "layer_mask": row.layer_mask,
                    "maximum_columns": row.maximum_columns,
                    "operator_plan_sha256": row.operator_plan_sha256,
                    "resident_bytes": row.resident_bytes,
                    "resident_geometry_sha256": (
                        row.resident_geometry_sha256
                    ),
                    "session_id": row.session_id,
                }
                for row in self.shards
            ],
            "transition_cost": self.transition_cost,
            "transition_cost_aggregation": (
                self.transition_cost_aggregation
            ),
            "transition_cost_by_session": dict(
                self.transition_cost_by_session
            ),
        }


def _residency_geometry_sha256(
    shards: Sequence[PhoneFfnShardPlacement],
) -> str:
    artifacts = {row.artifact_sha256 for row in shards}
    if len(artifacts) == 1:
        payload = {
            "artifact_sha256": next(iter(artifacts)),
            "shards": [
                {
                    "geometry_sha256": row.resident_geometry_sha256,
                    "session_id": row.session_id,
                }
                for row in shards
            ],
        }
    else:
        payload = {
            "shards": [
                {
                    "artifact_sha256": row.artifact_sha256,
                    "geometry_sha256": row.resident_geometry_sha256,
                    "session_id": row.session_id,
                }
                for row in shards
            ],
        }
    return canonical_sha256(payload)


def artifact_layout_identity_sha256(
    layout: PhoneFfnResidencyLayout, artifact_sha256: str,
) -> str:
    """Content identity of one model's resident shards in a layout.

    Independent of the layout generation and of the other models' sessions: a layer set
    that recurs keeps its identity, a different layer set or shard does not."""
    rows = sorted(
        (row for row in layout.shards if row.artifact_sha256 == artifact_sha256),
        key=lambda row: row.session_id,
    )
    if not rows:
        raise PhoneShardPlacementError("phone residency artifact is absent")
    layer_mask = 0
    for row in rows:
        layer_mask |= row.layer_mask
    return canonical_sha256({
        "artifact_sha256": artifact_sha256,
        "layer_mask": layer_mask,
        "schema": "phone-artifact-layout-identity-v1",
        "shards": [
            {
                "layer_mask": row.layer_mask,
                "maximum_columns": row.maximum_columns,
                "operator_plan_sha256": row.operator_plan_sha256,
                "resident_geometry_sha256": row.resident_geometry_sha256,
                "session_id": row.session_id,
            }
            for row in rows
        ],
    })


def _validate_layout_header(
    layout: PhoneFfnResidencyLayout,
    shards: tuple[PhoneFfnShardPlacement, ...],
) -> None:
    if (
        not shards
        or len({row.session_id for row in shards}) != len(shards)
        or any(not isinstance(row, PhoneFfnShardPlacement) for row in shards)
        or type(layout.queue_benefit) is not int
        or layout.queue_benefit < 0
        or type(layout.transition_cost) is not int
        or layout.transition_cost < 0
        or type(layout.objective) is not int
        or type(layout.objective_kind) is not str
        or not layout.objective_kind.isascii()
        or not layout.objective_kind
        or layout.transition_cost_aggregation not in {
            "hybrid", "shared_phone_union", "sum",
        }
    ):
        raise PhoneShardPlacementError("phone residency layout is invalid")


def _validate_layout_benefits(
    layout: PhoneFfnResidencyLayout,
    work: Mapping[str, int],
    benefit: Mapping[str, int],
    session_benefit: Mapping[str, int],
    session_ids: set[str],
) -> None:
    if any(
        type(artifact) is not str
        or not artifact.startswith("sha256:")
        or len(artifact) != 71
        or type(value) is not int
        or value <= 0
        for artifact, value in work.items()
    ):
        raise PhoneShardPlacementError(
            "phone residency layout queue work is invalid"
        )
    if (
        set(benefit) != set(work)
        or any(type(value) is not int or value <= 0
               for value in benefit.values())
        or sum(benefit.values()) != layout.queue_benefit
    ):
        raise PhoneShardPlacementError(
            "phone residency layout queue benefit is invalid"
        )
    if (
        any(
            type(session_id) is not str
            or session_id not in session_ids
            or type(value) is not int
            or value <= 0
            for session_id, value in session_benefit.items()
        )
        or sum(session_benefit.values()) != layout.queue_benefit
    ):
        raise PhoneShardPlacementError(
            "phone residency layout session benefit is invalid"
        )


def _validate_layout_identity_maps(
    layout: PhoneFfnResidencyLayout,
    session_ids: set[str],
    changed: tuple[str, ...],
    session_generations: Mapping[str, int],
    replacement_sources: tuple[PhoneFfnSessionIdentity, ...],
    replacement_source_bytes: Mapping[str, int],
    transition_by_session: Mapping[str, int],
) -> None:
    if (
        set(session_generations) - session_ids
        or any(
            type(session_id) is not str
            or type(generation) is not int
            or generation < 1
            for session_id, generation in session_generations.items()
        )
    ):
        raise PhoneShardPlacementError(
            "phone residency session generations are invalid"
        )
    if len(changed) != len(set(changed)):
        raise PhoneShardPlacementError(
            "phone residency changed sessions are duplicated"
        )
    if (
        any(not isinstance(row, PhoneFfnSessionIdentity)
            for row in replacement_sources)
        or len({row.session_id for row in replacement_sources})
            != len(replacement_sources)
        or {row.session_id for row in replacement_sources}
            != set(replacement_source_bytes)
        or set(replacement_source_bytes) - set(changed)
        or any(type(value) is not int or value <= 0
               for value in replacement_source_bytes.values())
    ):
        raise PhoneShardPlacementError(
            "phone residency replacement sources are invalid"
        )
    if (
        set(transition_by_session) != set(changed)
        or any(type(value) is not int or value < 0
               for value in transition_by_session.values())
        or sum(transition_by_session.values()) != layout.transition_cost
    ):
        raise PhoneShardPlacementError(
            "phone residency session transition cost is invalid"
        )


def _initialize_phone_residency_layout(
    layout: PhoneFfnResidencyLayout,
) -> None:
    shards = tuple(sorted(layout.shards, key=lambda row: row.session_id))
    _validate_layout_header(layout, shards)
    work = dict(layout.queued_work_by_artifact)
    benefit = dict(layout.queue_benefit_by_artifact)
    session_benefit = dict(layout.queue_benefit_by_session)
    transition_by_session = dict(layout.transition_cost_by_session)
    session_generations = dict(layout.session_generation_by_id)
    replacement_sources = tuple(sorted(
        layout.replacement_source_identities,
        key=lambda row: row.session_id,
    ))
    replacement_source_bytes = dict(
        layout.replacement_source_resident_bytes_by_session
    )
    session_ids = {row.session_id for row in shards}
    changed = tuple(sorted(layout.changed_session_ids))
    _validate_layout_benefits(
        layout, work, benefit, session_benefit, session_ids
    )
    _validate_layout_identity_maps(
        layout,
        session_ids,
        changed,
        session_generations,
        replacement_sources,
        replacement_source_bytes,
        transition_by_session,
    )
    if layout.geometry_sha256 != _residency_geometry_sha256(shards):
        raise PhoneShardPlacementError(
            "phone residency layout geometry differs"
        )
    values = {
        "shards": shards,
        "queued_work_by_artifact": work,
        "queue_benefit_by_artifact": benefit,
        "queue_benefit_by_session": session_benefit,
        "changed_session_ids": changed,
        "transition_cost_by_session": transition_by_session,
        "session_generation_by_id": session_generations,
        "replacement_source_identities": replacement_sources,
        "replacement_source_resident_bytes_by_session": (
            replacement_source_bytes
        ),
    }
    for name, value in values.items():
        if isinstance(value, dict):
            value = MappingProxyType(dict(sorted(value.items())))
        object.__setattr__(layout, name, value)


@dataclass(frozen=True)
class _PackingState:
    assignments: tuple[tuple[str, ...], ...]
    used_bytes: tuple[int, ...]
    value: int


def _layer_index(operator: ModelOperatorManifest) -> int:
    prefix, separator, raw_index = operator.layer_id.partition(":")
    try:
        index = int(raw_index)
    except ValueError as error:
        raise PhoneShardPlacementError(
            "FFN layer identity is invalid"
        ) from error
    if prefix != "layer" or separator != ":" or not 0 <= index < 64:
        raise PhoneShardPlacementError("FFN layer identity is invalid")
    return index


def _operator_score(
    manifest: ModelManifest,
    operator: ModelOperatorManifest,
    value: int,
) -> tuple[Fraction, int, str]:
    tensors = _resident_tensors(manifest, operator)
    resident_bytes = sum(row.nbytes for row in tensors)
    return (
        Fraction(value, resident_bytes),
        -_layer_index(operator),
        operator.operator_id,
    )


def _resident_tensors(
    manifest: ModelManifest,
    operator: ModelOperatorManifest,
) -> tuple[ModelTensorManifest, ...]:
    return tuple(
        manifest.tensor_by_id[tensor_id]
        for tensor_id in operator.tensor_ids
        if len(manifest.tensor_by_id[tensor_id].shape) >= 2
    )


def _packing_state_key(
    state: _PackingState,
    operator_by_id: Mapping[str, ModelOperatorManifest],
) -> tuple[object, ...]:
    layers = tuple(
        tuple(sorted(
            _layer_index(operator_by_id[operator_id])
            for operator_id in operator_ids
        ))
        for operator_ids in state.assignments
    )
    return (
        -state.value,
        sum(state.used_bytes),
        -sum(len(row) for row in state.assignments),
        tuple(0 if value > 0 else 1 for value in state.used_bytes),
        layers,
        state.used_bytes,
    )


def _prune_packing_states(
    states: Sequence[_PackingState],
    operator_by_id: Mapping[str, ModelOperatorManifest],
    maximum_states: int,
) -> tuple[_PackingState, ...]:
    by_usage: dict[tuple[int, ...], _PackingState] = {}
    for state in states:
        current = by_usage.get(state.used_bytes)
        if current is None or _packing_state_key(
            state, operator_by_id
        ) < _packing_state_key(current, operator_by_id):
            by_usage[state.used_bytes] = state
    ranked = sorted(
        by_usage.values(),
        key=lambda row: _packing_state_key(row, operator_by_id),
    )
    if len(ranked) <= maximum_states:
        return tuple(ranked)
    leaders: dict[int, _PackingState] = {}
    for state in ranked:
        active = sum(value > 0 for value in state.used_bytes)
        leaders.setdefault(active, state)
    selected = list(leaders.values())
    selected_ids = {id(row) for row in selected}
    selected.extend(
        row for row in ranked
        if id(row) not in selected_ids
    )
    return tuple(selected[:maximum_states])


@dataclass(frozen=True)
class _DisjointOperatorData:
    operators: tuple[ModelOperatorManifest, ...]
    ranked: tuple[ModelOperatorManifest, ...]
    operator_by_id: Mapping[str, ModelOperatorManifest]
    values: Mapping[str, int]
    resident_bytes: Mapping[str, int]
    data_types: Mapping[str, frozenset[str]]
    measured_benefits: bool


def _validate_disjoint_shard_request(
    manifest: ModelManifest,
    sessions: Sequence[RuntimePhoneSessionCapability],
    *,
    phone_wide_limit_bytes: int,
    maximum_columns: int,
    maximum_search_states: int,
    benefit_value_kind: str,
    include_unready: bool,
) -> tuple[RuntimePhoneSessionCapability, ...]:
    if not isinstance(manifest, ModelManifest):
        raise PhoneShardPlacementError("phone shard manifest is invalid")
    if type(phone_wide_limit_bytes) is not int or phone_wide_limit_bytes <= 0:
        raise PhoneShardPlacementError("phone-wide memory limit is invalid")
    if type(maximum_columns) is not int or maximum_columns <= 0:
        raise PhoneShardPlacementError("phone shard width is invalid")
    if type(maximum_search_states) is not int or maximum_search_states <= 0:
        raise PhoneShardPlacementError(
            "phone shard search-state limit is invalid"
        )
    if type(include_unready) is not bool:
        raise PhoneShardPlacementError(
            "phone shard readiness policy is invalid"
        )
    if (
        type(benefit_value_kind) is not str
        or not benefit_value_kind
        or not benefit_value_kind.isascii()
    ):
        raise PhoneShardPlacementError(
            "phone shard benefit identity is invalid"
        )
    session_rows = tuple(sessions)
    if any(
        not isinstance(row, RuntimePhoneSessionCapability)
        for row in session_rows
    ):
        raise PhoneShardPlacementError("phone session capability is invalid")
    if len({row.session_id for row in session_rows}) != len(session_rows):
        raise PhoneShardPlacementError("phone session IDs are duplicated")
    if session_rows and len({row.device_id for row in session_rows}) != 1:
        raise PhoneShardPlacementError("phone sessions span multiple devices")
    return session_rows


def _disjoint_operator_data(
    manifest: ModelManifest,
    benefit_by_operator: Mapping[str, int],
    allowed_operator_ids: Sequence[str] | None,
) -> _DisjointOperatorData:
    allowed = None if allowed_operator_ids is None else set(
        allowed_operator_ids
    )
    operators = tuple(
        row for row in manifest.operators
        if row.kind == "ffn"
        and (allowed is None or row.operator_id in allowed)
        and _resident_tensors(manifest, row)
    )
    all_ffn_ids = {
        row.operator_id for row in manifest.operators if row.kind == "ffn"
    }
    if not isinstance(benefit_by_operator, Mapping) or any(
        type(operator_id) is not str
        or operator_id not in all_ffn_ids
        or type(value) is not int
        for operator_id, value in benefit_by_operator.items()
    ):
        raise PhoneShardPlacementError(
            "phone shard operator benefit is invalid"
        )
    measured_benefits = bool(benefit_by_operator)
    values = {
        row.operator_id: (
            benefit_by_operator.get(row.operator_id, 0)
            if measured_benefits else
            sum(
                2 * tensor.elements
                for tensor in _resident_tensors(manifest, row)
            )
        )
        for row in operators
    }
    ranked = tuple(sorted(
        (
            row for row in operators
            if values[row.operator_id] > 0
        ),
        key=lambda row: _operator_score(
            manifest, row, values[row.operator_id]
        ),
        reverse=True,
    ))
    resident_bytes = {
        row.operator_id: sum(
            tensor.nbytes for tensor in _resident_tensors(manifest, row)
        )
        for row in operators
    }
    data_types = {
        row.operator_id: frozenset(
            tensor.quantization
            for tensor in _resident_tensors(manifest, row)
        )
        for row in operators
    }
    return _DisjointOperatorData(
        operators=operators,
        ranked=ranked,
        operator_by_id={row.operator_id: row for row in operators},
        values=values,
        resident_bytes=resident_bytes,
        data_types=data_types,
        measured_benefits=measured_benefits,
    )


def _session_supports_operator(
    session: RuntimePhoneSessionCapability,
    operator: ModelOperatorManifest,
    data_types: Mapping[str, frozenset[str]],
    maximum_columns: int,
    batch_plan: str,
) -> bool:
    return (
        bool(session.supported_layer_mask & (1 << _layer_index(operator)))
        and maximum_columns <= session.maximum_columns
        and maximum_columns % session.column_quantum == 0
        and batch_plan in session.batch_plans
        and (
            "*" in session.supported_data_types
            or data_types[operator.operator_id].issubset(
                session.supported_data_types
            )
        )
    )


def _rank_disjoint_sessions(
    sessions: Sequence[RuntimePhoneSessionCapability],
    data: _DisjointOperatorData,
    maximum_columns: int,
    batch_plan: str,
    include_unready: bool,
) -> tuple[RuntimePhoneSessionCapability, ...]:
    return tuple(sorted(
        (
            row for row in sessions
            if row.ready or include_unready
        ),
        key=lambda row: (
            not row.ready,
            -sum(
                1
                for operator in data.operators
                if _session_supports_operator(
                    row,
                    operator,
                    data.data_types,
                    maximum_columns,
                    batch_plan,
                )
            ),
            -row.resident_memory_limit_bytes,
            row.session_id,
        ),
    ))


def _search_disjoint_packing_states(
    rows: tuple[RuntimePhoneSessionCapability, ...],
    data: _DisjointOperatorData,
    phone_wide_limit_bytes: int,
    maximum_columns: int,
    batch_plan: str,
    maximum_search_states: int,
) -> tuple[_PackingState, ...]:
    compatible_sessions = {
        operator.operator_id: tuple(
            index for index, session in enumerate(rows)
            if _session_supports_operator(
                session,
                operator,
                data.data_types,
                maximum_columns,
                batch_plan,
            )
        )
        for operator in data.ranked
    }
    states = (_PackingState(
        assignments=tuple(() for _ in rows),
        used_bytes=tuple(0 for _ in rows),
        value=0,
    ),)
    for operator in data.ranked:
        required = data.resident_bytes[operator.operator_id]
        value = data.values[operator.operator_id]
        expanded = list(states)
        for state in states:
            total = sum(state.used_bytes)
            if total + required > phone_wide_limit_bytes:
                continue
            for index in compatible_sessions[operator.operator_id]:
                if (
                    state.used_bytes[index] + required
                    > rows[index].resident_memory_limit_bytes
                ):
                    continue
                assignments = list(state.assignments)
                assignments[index] = (
                    *assignments[index], operator.operator_id
                )
                used = list(state.used_bytes)
                used[index] += required
                expanded.append(_PackingState(
                    assignments=tuple(assignments),
                    used_bytes=tuple(used),
                    value=state.value + value,
                ))
        states = _prune_packing_states(
            expanded, data.operator_by_id, maximum_search_states
        )
    return states


def _disjoint_shard(
    manifest: ModelManifest,
    session: RuntimePhoneSessionCapability,
    operator_ids: tuple[str, ...],
    resident_bytes: int,
    maximum_columns: int,
) -> PhoneFfnShardPlacement:
    operator_by_id = manifest.operator_by_id
    assigned = tuple(sorted(
        (operator_by_id[operator_id] for operator_id in operator_ids),
        key=_layer_index,
    ))
    sorted_ids = tuple(row.operator_id for row in assigned)
    layer_mask = sum(1 << _layer_index(row) for row in assigned)
    geometry = canonical_sha256({
        "artifact_sha256": manifest.artifact_sha256,
        "columns": maximum_columns,
        "layer_mask": layer_mask,
        "resident_bytes": resident_bytes,
        "session_id": session.session_id,
        "worker_identity_sha256": session.worker_identity_sha256,
    })
    return PhoneFfnShardPlacement(
        artifact_sha256=manifest.artifact_sha256,
        session_id=session.session_id,
        endpoint=session.endpoint,
        memory_resource_id=session.memory_resource_id,
        operator_ids=sorted_ids,
        layer_mask=layer_mask,
        maximum_columns=maximum_columns,
        resident_bytes=resident_bytes,
        resident_geometry_sha256=geometry,
        operator_plan_sha256=canonical_sha256({
            "artifact_sha256": manifest.artifact_sha256,
            "columns": maximum_columns,
            "operator_ids": sorted_ids,
            "session_id": session.session_id,
        }),
    )


def _disjoint_shard_sets_from_states(
    manifest: ModelManifest,
    rows: tuple[RuntimePhoneSessionCapability, ...],
    states: tuple[_PackingState, ...],
    data: _DisjointOperatorData,
    maximum_columns: int,
    benefit_value_kind: str,
) -> tuple[PhoneFfnShardSet, ...]:
    results = []
    for count in range(1, len(rows) + 1):
        choices = tuple(
            state for state in states
            if sum(value > 0 for value in state.used_bytes) == count
        )
        if not choices:
            continue
        best = min(
            choices,
            key=lambda row: _packing_state_key(row, data.operator_by_id),
        )
        selected_indices = tuple(
            index for index, value in enumerate(best.used_bytes) if value > 0
        )
        selected = tuple(rows[index] for index in selected_indices)
        shards = tuple(
            _disjoint_shard(
                manifest,
                rows[index],
                best.assignments[index],
                best.used_bytes[index],
                maximum_columns,
            )
            for index in selected_indices
        )
        all_ids = tuple(
            operator.operator_id for operator in manifest.operators
            if any(
                operator.operator_id in shard.operator_ids
                for shard in shards
            )
        )
        results.append(PhoneFfnShardSet(
            shards=shards,
            operator_ids=all_ids,
            resident_bytes=sum(best.used_bytes),
            geometry_sha256=canonical_sha256({
                "artifact_sha256": manifest.artifact_sha256,
                "shards": [
                    {
                        "geometry_sha256": row.resident_geometry_sha256,
                        "session_id": row.session_id,
                    }
                    for row in shards
                ],
            }),
            packing_value=best.value,
            packing_value_kind=(
                benefit_value_kind
                if data.measured_benefits else "rough_compute_ops"
            ),
            unavailable_session_ids=tuple(sorted(
                row.session_id for row in selected if not row.ready
            )),
        ))
    return tuple(results)


def _generate_disjoint_ffn_shard_sets(
    manifest: ModelManifest,
    sessions: Sequence[RuntimePhoneSessionCapability],
    *,
    phone_wide_limit_bytes: int,
    maximum_columns: int,
    batch_plan: str,
    benefit_by_operator: Mapping[str, int] = MappingProxyType({}),
    benefit_value_kind: str = "measured_net_energy_uj",
    allowed_operator_ids: Sequence[str] | None = None,
    maximum_search_states: int = 4096,
    include_unready: bool = False,
) -> tuple[PhoneFfnShardSet, ...]:
    """Generate the best feasible placement for each session count 1..N."""

    session_rows = _validate_disjoint_shard_request(
        manifest,
        sessions,
        phone_wide_limit_bytes=phone_wide_limit_bytes,
        maximum_columns=maximum_columns,
        maximum_search_states=maximum_search_states,
        benefit_value_kind=benefit_value_kind,
        include_unready=include_unready,
    )
    data = _disjoint_operator_data(
        manifest, benefit_by_operator, allowed_operator_ids
    )
    rows = _rank_disjoint_sessions(
        session_rows,
        data,
        maximum_columns,
        batch_plan,
        include_unready,
    )
    if not rows or not data.ranked:
        return ()
    states = _search_disjoint_packing_states(
        rows,
        data,
        phone_wide_limit_bytes,
        maximum_columns,
        batch_plan,
        maximum_search_states,
    )
    return _disjoint_shard_sets_from_states(
        manifest,
        rows,
        states,
        data,
        maximum_columns,
        benefit_value_kind,
    )


def _session_packing_identity(
    session: RuntimePhoneSessionCapability,
) -> tuple[object, ...]:
    return (
        session.session_id,
        session.device_id,
        session.endpoint,
        session.worker_identity_sha256,
        session.memory_resource_id,
        session.resident_memory_limit_bytes,
        session.shared_compute_resource_id,
        session.shared_transport_resource_ids,
        session.supported_layer_mask,
        session.maximum_columns,
        session.column_quantum,
        session.supported_data_types,
        session.batch_plans,
        session.ready,
    )


def _cached_disjoint_ffn_shard_sets(
    manifest: ModelManifest,
    sessions: Sequence[RuntimePhoneSessionCapability],
    *,
    phone_wide_limit_bytes: int,
    maximum_columns: int,
    batch_plan: str,
    benefit_by_operator: Mapping[str, int],
    benefit_value_kind: str,
    allowed_operator_ids: Sequence[str] | None,
    maximum_search_states: int,
    include_unready: bool,
) -> tuple[PhoneFfnShardSet, ...]:
    session_rows = tuple(sessions)
    key = canonical_sha256({
        "allowed_operator_ids": (
            None
            if allowed_operator_ids is None
            else sorted(allowed_operator_ids)
        ),
        "artifact_sha256": manifest.artifact_sha256,
        "batch_plan": batch_plan,
        "benefit_by_operator": dict(sorted(
            benefit_by_operator.items()
        )),
        "benefit_value_kind": benefit_value_kind,
        "include_unready": include_unready,
        "maximum_columns": maximum_columns,
        "maximum_search_states": maximum_search_states,
        "phone_wide_limit_bytes": phone_wide_limit_bytes,
        "sessions": [
            list(_session_packing_identity(row)) for row in session_rows
        ],
    })
    with _shard_set_cache_lock:
        cached = _shard_set_cache.get(key)
    if cached is not None:
        return cached
    generated = _generate_disjoint_ffn_shard_sets(
        manifest,
        session_rows,
        phone_wide_limit_bytes=phone_wide_limit_bytes,
        maximum_columns=maximum_columns,
        batch_plan=batch_plan,
        benefit_by_operator=benefit_by_operator,
        benefit_value_kind=benefit_value_kind,
        allowed_operator_ids=allowed_operator_ids,
        maximum_search_states=maximum_search_states,
        include_unready=include_unready,
    )
    with _shard_set_cache_lock:
        existing = _shard_set_cache.setdefault(key, generated)
        while len(_shard_set_cache) > _SHARD_SET_CACHE_MAXIMUM:
            oldest = next(iter(_shard_set_cache))
            if oldest == key and len(_shard_set_cache) > 1:
                oldest = next(
                    row for row in _shard_set_cache if row != key
                )
            _shard_set_cache.pop(oldest)
    return existing


def generate_disjoint_ffn_shard_sets(
    manifest: ModelManifest,
    sessions: Sequence[RuntimePhoneSessionCapability],
    *,
    phone_wide_limit_bytes: int,
    maximum_columns: int,
    batch_plan: str,
    benefit_by_operator: Mapping[str, int] = MappingProxyType({}),
    benefit_value_kind: str = "measured_net_energy_uj",
    allowed_operator_ids: Sequence[str] | None = None,
    maximum_search_states: int = 4096,
    include_unready: bool = False,
) -> tuple[PhoneFfnShardSet, ...]:
    """Return an immutable packing cached by static model/device inputs."""

    return _cached_disjoint_ffn_shard_sets(
        manifest,
        sessions,
        phone_wide_limit_bytes=phone_wide_limit_bytes,
        maximum_columns=maximum_columns,
        batch_plan=batch_plan,
        benefit_by_operator=benefit_by_operator,
        benefit_value_kind=benefit_value_kind,
        allowed_operator_ids=allowed_operator_ids,
        maximum_search_states=maximum_search_states,
        include_unready=include_unready,
    )


@dataclass(frozen=True)
class _MixedResidencyInputs:
    demands: tuple[PhoneFfnResidencyDemand, ...]
    usable_sessions: tuple[RuntimePhoneSessionCapability, ...]
    demand_by_artifact: Mapping[str, PhoneFfnResidencyDemand]
    artifacts: tuple[str, ...]
    current_by_session: Mapping[str, PhoneFfnShardPlacement]
    transition_costs: Mapping[str, int]
    objective_kind: str
    phone_wide_limit_bytes: int
    include_unready: bool
    storage_by_artifact: Mapping[
        str, Mapping[str, PhoneFfnShardStorageMetadata]
    ]
    manifest_by_artifact: Mapping[str, ModelManifest]


@dataclass(frozen=True)
class _MixedArtifactPacking:
    new_shards: tuple[PhoneFfnShardPlacement, ...]
    artifact_benefit: int
    shard_benefits: Mapping[str, int]


@dataclass(frozen=True)
class _MixedAssignmentPacking:
    shards: tuple[PhoneFfnShardPlacement, ...]
    queue_benefit: int
    queue_benefit_by_artifact: Mapping[str, int]
    queue_benefit_by_session: Mapping[str, int]


def _mixed_objective_kind(
    demands: Sequence[PhoneFfnResidencyDemand],
) -> str:
    benefit_kinds = {str(row.benefit_value_kind) for row in demands}
    energy_benefit_kinds = {
        "assumed_net_energy_uj",
        "measured_net_energy_uj",
        "profile_prior_net_energy_uj",
    }
    if benefit_kinds <= energy_benefit_kinds:
        return "queue_energy_delta_uj"
    if benefit_kinds == {"rough_compute_ops"}:
        return "queue_rough_compute_ops"
    raise PhoneShardPlacementError("phone residency benefit units differ")


def _mixed_residency_inputs(
    demands: Sequence[PhoneFfnResidencyDemand],
    sessions: Sequence[RuntimePhoneSessionCapability],
    phone_wide_limit_bytes: int,
    current_shards: Sequence[PhoneFfnShardPlacement],
    transition_energy_uj_by_session: Mapping[str, int],
    maximum_layouts: int,
    include_unready: bool,
    shard_storage: Sequence[PhoneFfnShardStorageMetadata],
    resident_manifests: Sequence[ModelManifest] = (),
) -> _MixedResidencyInputs:
    demand_rows = tuple(demands)
    session_rows = tuple(sorted(sessions, key=lambda row: row.session_id))
    current_rows = tuple(current_shards)
    transition_costs = dict(transition_energy_uj_by_session)
    manifests = tuple(resident_manifests)
    if any(not isinstance(row, ModelManifest) for row in manifests):
        raise PhoneShardPlacementError("phone residency manifests are invalid")
    if (
        (not demand_rows and not (current_rows and manifests))
        or any(not isinstance(row, PhoneFfnResidencyDemand)
               for row in demand_rows)
        or len({row.manifest.artifact_sha256 for row in demand_rows})
            != len(demand_rows)
    ):
        raise PhoneShardPlacementError("phone residency demands are invalid")
    if (
        not session_rows
        or any(not isinstance(row, RuntimePhoneSessionCapability)
               for row in session_rows)
        or len({row.session_id for row in session_rows}) != len(session_rows)
        or len({row.device_id for row in session_rows}) != 1
    ):
        raise PhoneShardPlacementError("phone residency sessions are invalid")
    if (
        type(phone_wide_limit_bytes) is not int
        or phone_wide_limit_bytes <= 0
        or type(maximum_layouts) is not int
        or maximum_layouts <= 0
        or type(include_unready) is not bool
    ):
        raise PhoneShardPlacementError(
            "phone residency layout limits are invalid"
        )
    session_ids = {row.session_id for row in session_rows}
    if (
        any(not isinstance(row, PhoneFfnShardPlacement)
            for row in current_rows)
        or len({row.session_id for row in current_rows}) != len(current_rows)
        or any(
            session_id not in session_ids
            or type(value) is not int
            or value < 0
            for session_id, value in transition_costs.items()
        )
    ):
        raise PhoneShardPlacementError(
            "phone residency current layout is invalid"
        )
    usable_sessions = tuple(
        row for row in session_rows if row.ready or include_unready
    )
    demand_by_artifact = {
        row.manifest.artifact_sha256: row for row in demand_rows
    }
    storage_by_artifact: dict[str, dict[str, PhoneFfnShardStorageMetadata]] = {}
    for row in shard_storage:
        if not isinstance(row, PhoneFfnShardStorageMetadata):
            raise PhoneShardPlacementError(
                "phone FFN shard storage metadata is invalid"
            )
        stored = storage_by_artifact.setdefault(row.parent_artifact_sha256, {})
        if row.session_id in stored:
            raise PhoneShardPlacementError(
                "phone FFN shard storage sessions are duplicated"
            )
        stored[row.session_id] = row
    return _MixedResidencyInputs(
        demands=demand_rows,
        usable_sessions=usable_sessions,
        demand_by_artifact=demand_by_artifact,
        artifacts=tuple(sorted(demand_by_artifact)),
        current_by_session={row.session_id: row for row in current_rows},
        transition_costs=transition_costs,
        objective_kind=_mixed_objective_kind(demand_rows),
        phone_wide_limit_bytes=phone_wide_limit_bytes,
        include_unready=include_unready,
        storage_by_artifact=storage_by_artifact,
        manifest_by_artifact={
            row.artifact_sha256: row
            for row in (*manifests, *(demand.manifest for demand in demand_rows))
        },
    )


def _retained_mixed_shards(
    inputs: _MixedResidencyInputs,
    assignment: tuple[int, ...],
) -> dict[str, PhoneFfnShardPlacement]:
    retained: dict[str, PhoneFfnShardPlacement] = {}
    for session, selected in zip(inputs.usable_sessions, assignment):
        current = inputs.current_by_session.get(session.session_id)
        if current is None:
            continue
        compatible = (
            current.endpoint == session.endpoint
            and current.memory_resource_id == session.memory_resource_id
            and current.maximum_columns <= session.maximum_columns
            and current.maximum_columns % session.column_quantum == 0
            and current.layer_mask & ~session.supported_layer_mask == 0
            and current.resident_bytes <= session.resident_memory_limit_bytes
        )
        if not compatible:
            continue
        if selected < 0:
            if current.artifact_sha256 not in inputs.demand_by_artifact:
                retained[session.session_id] = current
            continue
        artifact = inputs.artifacts[selected]
        demand = inputs.demand_by_artifact[artifact]
        operator_by_id = demand.manifest.operator_by_id
        allowed = (
            set(operator_by_id)
            if demand.allowed_operator_ids is None
            else set(demand.allowed_operator_ids)
        )
        if (
            current.artifact_sha256 == artifact
            and current.maximum_columns == demand.maximum_columns
            and demand.batch_plan in session.batch_plans
            and set(current.operator_ids) <= allowed
            and set(current.operator_ids) <= set(operator_by_id)
        ):
            retained[session.session_id] = current
    return retained


def _mixed_shard_benefit(
    demand: PhoneFfnResidencyDemand,
    shard: PhoneFfnShardPlacement,
) -> int:
    if demand.benefit_by_operator:
        value = sum(
            demand.benefit_by_operator.get(operator_id, 0)
            for operator_id in shard.operator_ids
        )
    else:
        value = sum(
            2 * tensor.elements
            for operator_id in shard.operator_ids
            for tensor in _resident_tensors(
                demand.manifest,
                demand.manifest.operator_by_id[operator_id],
            )
        )
    return value * demand.queued_work


def _pack_mixed_artifact(
    inputs: _MixedResidencyInputs,
    demand: PhoneFfnResidencyDemand,
    assigned_sessions: tuple[RuntimePhoneSessionCapability, ...],
    retained_by_session: Mapping[str, PhoneFfnShardPlacement],
    fixed_retained_bytes: int,
    packed_new_bytes: int,
) -> _MixedArtifactPacking | None:
    retained = tuple(
        retained_by_session[session.session_id]
        for session in assigned_sessions
        if session.session_id in retained_by_session
    )
    retained_session_ids = {row.session_id for row in retained}
    retained_operator_ids = {
        operator_id for row in retained for operator_id in row.operator_ids
    }
    new_sessions = tuple(
        session for session in assigned_sessions
        if session.session_id not in retained_session_ids
    )
    new_shards: tuple[PhoneFfnShardPlacement, ...] = ()
    if new_sessions:
        operator_by_id = demand.manifest.operator_by_id
        allowed = (
            set(operator_by_id)
            if demand.allowed_operator_ids is None
            else set(demand.allowed_operator_ids)
        )
        remaining_operator_ids = tuple(sorted(
            allowed - retained_operator_ids
        ))
        remaining_capacity = (
            inputs.phone_wide_limit_bytes
            - fixed_retained_bytes
            - packed_new_bytes
        )
        if not remaining_operator_ids or remaining_capacity <= 0:
            return None
        shard_sets = _cached_disjoint_ffn_shard_sets(
            demand.manifest,
            phone_sessions_with_storage_coverage(
                new_sessions,
                inputs.storage_by_artifact.get(demand.manifest.artifact_sha256),
            ),
            phone_wide_limit_bytes=remaining_capacity,
            maximum_columns=demand.maximum_columns,
            batch_plan=demand.batch_plan,
            benefit_by_operator=demand.benefit_by_operator,
            benefit_value_kind=str(demand.benefit_value_kind),
            allowed_operator_ids=remaining_operator_ids,
            maximum_search_states=4096,
            include_unready=inputs.include_unready,
        )
        selected = next((
            row for row in shard_sets
            if len(row.shards) == len(new_sessions)
        ), None)
        if selected is None or {
            row.session_id for row in selected.shards
        } != {row.session_id for row in new_sessions}:
            return None
        new_shards = selected.shards
    selected_shards = tuple(sorted(
        (*retained, *new_shards), key=lambda row: row.session_id
    ))
    if {row.session_id for row in selected_shards} != {
        row.session_id for row in assigned_sessions
    }:
        return None
    shard_benefits = {
        shard.session_id: value
        for shard in selected_shards
        if (value := _mixed_shard_benefit(demand, shard)) > 0
    }
    artifact_benefit = sum(shard_benefits.values())
    selected_value = sum(
        _mixed_shard_benefit(demand, shard)
        for shard in selected_shards
    )
    if artifact_benefit != selected_value:
        raise PhoneShardPlacementError(
            "phone residency session benefit differs"
        )
    return _MixedArtifactPacking(
        new_shards=new_shards,
        artifact_benefit=artifact_benefit,
        shard_benefits=shard_benefits,
    )


def _pack_mixed_assignment(
    inputs: _MixedResidencyInputs,
    assignment: tuple[int, ...],
) -> _MixedAssignmentPacking | None:
    retained_by_session = _retained_mixed_shards(inputs, assignment)
    fixed_retained = tuple(retained_by_session.values())
    fixed_retained_bytes = sum(row.resident_bytes for row in fixed_retained)
    if fixed_retained_bytes > inputs.phone_wide_limit_bytes:
        return None
    shards = list(fixed_retained)
    packed_new_bytes = 0
    queue_benefit = 0
    benefits_by_artifact: dict[str, int] = {}
    benefits_by_session: dict[str, int] = {}
    for artifact_index, artifact in enumerate(inputs.artifacts):
        assigned_sessions = tuple(
            session for session, selected in zip(
                inputs.usable_sessions, assignment
            )
            if selected == artifact_index
        )
        if not assigned_sessions:
            continue
        packed = _pack_mixed_artifact(
            inputs,
            inputs.demand_by_artifact[artifact],
            assigned_sessions,
            retained_by_session,
            fixed_retained_bytes,
            packed_new_bytes,
        )
        if packed is None:
            return None
        shards.extend(packed.new_shards)
        packed_new_bytes += sum(
            row.resident_bytes for row in packed.new_shards
        )
        queue_benefit += packed.artifact_benefit
        if packed.artifact_benefit > 0:
            benefits_by_artifact[artifact] = packed.artifact_benefit
        benefits_by_session.update(packed.shard_benefits)
    if not shards or queue_benefit <= 0:
        return None
    if sum(row.resident_bytes for row in shards) > inputs.phone_wide_limit_bytes:
        return None
    return _MixedAssignmentPacking(
        shards=tuple(shards),
        queue_benefit=queue_benefit,
        queue_benefit_by_artifact=benefits_by_artifact,
        queue_benefit_by_session=benefits_by_session,
    )


def _mixed_changed_sessions(
    current_by_session: Mapping[str, PhoneFfnShardPlacement],
    shard_by_session: Mapping[str, PhoneFfnShardPlacement],
) -> tuple[str, ...]:
    return tuple(sorted(
        session_id
        for session_id in set(current_by_session) | set(shard_by_session)
        if (
            session_id not in current_by_session
            or session_id not in shard_by_session
            or current_by_session[session_id].artifact_sha256
                != shard_by_session[session_id].artifact_sha256
            or current_by_session[session_id].resident_geometry_sha256
                != shard_by_session[session_id].resident_geometry_sha256
        )
    ))


def _raw_mixed_transition_costs(
    inputs: _MixedResidencyInputs,
    changed_sessions: tuple[str, ...],
    shard_by_session: Mapping[str, PhoneFfnShardPlacement],
) -> tuple[dict[str, int], dict[str, str]]:
    raw_costs: dict[str, int] = {}
    aggregations: dict[str, str] = {}
    for session_id in changed_sessions:
        new_shard = shard_by_session.get(session_id)
        old_shard = inputs.current_by_session.get(session_id)
        demand = (
            None
            if new_shard is None
            else inputs.demand_by_artifact.get(new_shard.artifact_sha256)
        )
        if demand is None and old_shard is not None:
            demand = inputs.demand_by_artifact.get(old_shard.artifact_sha256)
        route_cost = (
            None
            if demand is None
            else demand.transition_energy_uj_by_session.get(session_id)
        )
        raw_costs[session_id] = (
            inputs.transition_costs.get(session_id, 0)
            if route_cost is None else route_cost
        )
        aggregations[session_id] = (
            "sum" if route_cost is None else demand.transition_energy_aggregation
        )
    return raw_costs, aggregations


def _mixed_transition_costs(
    inputs: _MixedResidencyInputs,
    changed_sessions: tuple[str, ...],
    shard_by_session: Mapping[str, PhoneFfnShardPlacement],
) -> tuple[dict[str, int], int, str]:
    raw_costs, aggregations = _raw_mixed_transition_costs(
        inputs, changed_sessions, shard_by_session
    )
    costs = {
        session_id: raw_costs[session_id]
        for session_id in changed_sessions
        if aggregations[session_id] == "sum"
    }
    shared_ids = tuple(
        session_id for session_id in changed_sessions
        if aggregations[session_id] == "shared_phone_union"
    )
    if shared_ids:
        shared_total = max(raw_costs[session_id] for session_id in shared_ids)
        raw_total = sum(raw_costs[session_id] for session_id in shared_ids)
        allocations = (
            {session_id: 0 for session_id in shared_ids}
            if raw_total == 0 else
            {
                session_id: shared_total * raw_costs[session_id] // raw_total
                for session_id in shared_ids
            }
        )
        remainder = shared_total - sum(allocations.values())
        for session_id in shared_ids:
            if remainder == 0:
                break
            allocations[session_id] += 1
            remainder -= 1
        costs.update(allocations)
    aggregation_kinds = set(aggregations.values())
    aggregation = (
        next(iter(aggregation_kinds))
        if len(aggregation_kinds) == 1 else "hybrid"
    )
    return costs, sum(costs.values()), aggregation


def _mixed_layout(
    inputs: _MixedResidencyInputs,
    packing: _MixedAssignmentPacking,
) -> PhoneFfnResidencyLayout:
    shard_by_session = {row.session_id: row for row in packing.shards}
    changed = _mixed_changed_sessions(
        inputs.current_by_session, shard_by_session
    )
    costs, transition_cost, aggregation = _mixed_transition_costs(
        inputs, changed, shard_by_session
    )
    geometry_sha256 = _residency_geometry_sha256(tuple(sorted(
        packing.shards, key=lambda row: row.session_id
    )))
    return PhoneFfnResidencyLayout(
        shards=packing.shards,
        queued_work_by_artifact={
            artifact: inputs.demand_by_artifact[artifact].queued_work
            for artifact in sorted(packing.queue_benefit_by_artifact)
        },
        queue_benefit_by_artifact=packing.queue_benefit_by_artifact,
        queue_benefit_by_session=packing.queue_benefit_by_session,
        queue_benefit=packing.queue_benefit,
        transition_cost=transition_cost,
        transition_cost_by_session=costs,
        objective=(
            transition_cost - packing.queue_benefit
            if inputs.objective_kind == "queue_energy_delta_uj"
            else -packing.queue_benefit
        ),
        objective_kind=inputs.objective_kind,
        changed_session_ids=changed,
        geometry_sha256=geometry_sha256,
        transition_cost_aggregation=aggregation,
    )


def _mixed_layout_rank(layout: PhoneFfnResidencyLayout) -> tuple[object, ...]:
    return (
        layout.objective,
        -layout.queue_benefit,
        layout.transition_cost,
        -len(layout.shards),
        tuple(sorted(shard.session_id for shard in layout.shards)),
        layout.geometry_sha256,
    )


def _smaller_resident_shards(inputs, session, source):
    """Layer-only resize choices; runtime fraction changes do not release RAM."""
    if (source.endpoint != session.endpoint
            or source.memory_resource_id != session.memory_resource_id
            or source.layer_mask & ~session.supported_layer_mask
            or source.maximum_columns > session.maximum_columns
            or source.maximum_columns % session.column_quantum):
        return ()
    stored = inputs.storage_by_artifact.get(source.artifact_sha256)
    if stored is not None and (
        session.session_id not in stored
        or source.layer_mask & ~stored[session.session_id].layer_mask
        or source.maximum_columns > stored[session.session_id].maximum_columns
    ):
        return ()
    manifest = inputs.manifest_by_artifact.get(source.artifact_sha256)
    unchanged = (
        (source,)
        if source.resident_bytes <= session.resident_memory_limit_bytes else ()
    )
    if manifest is None:
        return unchanged
    demand = inputs.demand_by_artifact.get(source.artifact_sha256)
    if demand is not None and demand.batch_plan not in session.batch_plans:
        return ()
    data = _disjoint_operator_data(
        manifest, {} if demand is None else demand.benefit_by_operator,
        source.operator_ids,
    )
    if any(types.difference(session.supported_data_types) for types in data.data_types.values()):
        return ()
    # Do not infer partial-width byte counts from whole-matrix metadata.
    if (set(data.operator_by_id) != set(source.operator_ids)
            or sum(data.resident_bytes.values()) != source.resident_bytes
            or source.maximum_columns != manifest.feed_forward_length):
        return unchanged
    ranked = sorted(data.operators, key=lambda row: _operator_score(
        manifest, row, data.values[row.operator_id]
    ), reverse=True)
    options = list(unchanged)
    for count in range(1, len(ranked)):
        ids = tuple(row.operator_id for row in ranked[:count])
        size = sum(data.resident_bytes[key] for key in ids)
        if size <= session.resident_memory_limit_bytes:
            options.append(_disjoint_shard(
                manifest, session, ids, size, source.maximum_columns,
            ))
    return tuple(options)


def _resident_layout_packing(inputs, shards):
    by_session, by_artifact = {}, {}
    for shard in shards:
        demand = inputs.demand_by_artifact.get(shard.artifact_sha256)
        benefit = 0 if demand is None else _mixed_shard_benefit(demand, shard)
        if benefit > 0:
            by_session[shard.session_id] = benefit
            by_artifact[shard.artifact_sha256] = (
                by_artifact.get(shard.artifact_sha256, 0) + benefit
            )
    return _MixedAssignmentPacking(shards, sum(by_session.values()), by_artifact, by_session)


def _memory_cap_growth_layouts(inputs):
    """Offer demanded same-artifact growth, retaining every other session."""
    current = inputs.current_by_session
    headroom = inputs.phone_wide_limit_bytes - sum(row.resident_bytes for row in current.values())
    if not current or headroom <= 0:
        return ()
    artifact_indices = {artifact: index for index, artifact in enumerate(inputs.artifacts)}
    assignment = tuple(
        artifact_indices.get(current[row.session_id].artifact_sha256, -1)
        if row.session_id in current else -1 for row in inputs.usable_sessions
    )
    if _retained_mixed_shards(inputs, assignment) != current:
        return ()
    layouts = []
    for session in inputs.usable_sessions:
        source = current.get(session.session_id)
        demand = None if source is None else inputs.demand_by_artifact.get(source.artifact_sha256)
        if (demand is None or source.maximum_columns != demand.maximum_columns
                or source.maximum_columns != demand.manifest.feed_forward_length
                or source.endpoint != session.endpoint
                or source.memory_resource_id != session.memory_resource_id):
            continue
        limit = min(headroom, session.resident_memory_limit_bytes - source.resident_bytes)
        if limit <= 0:
            continue
        covered_sessions = phone_sessions_with_storage_coverage(
            (session,), inputs.storage_by_artifact.get(source.artifact_sha256),
        )
        if not covered_sessions:
            continue
        session, = covered_sessions
        data = _disjoint_operator_data(demand.manifest, demand.benefit_by_operator,
                                       demand.allowed_operator_ids)
        if (not set(source.operator_ids).issubset(data.operator_by_id)
                or sum(data.resident_bytes[key] for key in source.operator_ids) != source.resident_bytes
                or any(not _session_supports_operator(
                    session, data.operator_by_id[key], data.data_types,
                    source.maximum_columns, demand.batch_plan,
                ) for key in source.operator_ids)):
            continue
        expected = _disjoint_shard(demand.manifest, session, source.operator_ids,
                                   source.resident_bytes, source.maximum_columns)
        if (expected.resident_geometry_sha256 != source.resident_geometry_sha256
                or expected.operator_plan_sha256 != source.operator_plan_sha256):
            continue
        covered_ids = {key for row in current.values()
                       if row.artifact_sha256 == source.artifact_sha256 for key in row.operator_ids}
        extras = _cached_disjoint_ffn_shard_sets(
            demand.manifest, (session,), phone_wide_limit_bytes=limit,
            maximum_columns=source.maximum_columns, batch_plan=demand.batch_plan,
            benefit_by_operator=demand.benefit_by_operator,
            benefit_value_kind=str(demand.benefit_value_kind),
            allowed_operator_ids=tuple(sorted(set(data.operator_by_id) - covered_ids)),
            maximum_search_states=4096, include_unready=inputs.include_unready,
        )
        for extra_set in extras:
            extra, = extra_set.shards
            grown = _disjoint_shard(
                demand.manifest, session, (*source.operator_ids, *extra.operator_ids),
                source.resident_bytes + extra.resident_bytes, source.maximum_columns,
            )
            shards = tuple(grown if key == source.session_id else current[key] for key in sorted(current))
            layouts.append(_mixed_layout(inputs, _resident_layout_packing(inputs, shards)))
    return tuple(layouts)


def _memory_cap_shrink_layouts(inputs):
    """Fit resident subsets without changing artifacts or emptying sessions."""
    current = inputs.current_by_session
    if not current or sum(row.resident_bytes for row in current.values()) <= inputs.phone_wide_limit_bytes:
        return ()
    sessions = {row.session_id: row for row in inputs.usable_sessions}
    if not set(current).issubset(sessions):
        return ()
    options = tuple(_smaller_resident_shards(inputs, sessions[key], current[key])
                    for key in sorted(current))
    if any(not rows for rows in options):
        return ()

    def rank(shards):
        changed = tuple(row.session_id for row in shards if row != current[row.session_id])
        _, cost, _ = _mixed_transition_costs(inputs, changed, {row.session_id: row for row in shards})
        benefit = _resident_layout_packing(inputs, shards).queue_benefit
        return (cost - benefit if inputs.objective_kind == "queue_energy_delta_uj" else -benefit,
                len(changed), -sum(row.resident_bytes for row in shards),
                tuple(row.resident_geometry_sha256 for row in shards))

    states = ((),)
    for index, choices in enumerate(options):
        remaining_minimum = sum(min(row.resident_bytes for row in rows) for rows in options[index + 1:])
        by_size = {}
        for previous in states:
            for shard in choices:
                rows = (*previous, shard)
                size = sum(row.resident_bytes for row in rows)
                if size + remaining_minimum > inputs.phone_wide_limit_bytes:
                    continue
                changed = sum(row != current[row.session_id] for row in rows)
                key = (size, changed)
                if key not in by_size or rank(rows) < rank(by_size[key]):
                    by_size[key] = rows
        states = tuple(sorted(by_size.values(), key=rank)[:4096])
    return tuple(_mixed_layout(inputs, _resident_layout_packing(inputs, rows)) for rows in states)


def generate_mixed_ffn_residency_layouts(
    demands: Sequence[PhoneFfnResidencyDemand],
    sessions: Sequence[RuntimePhoneSessionCapability],
    *,
    phone_wide_limit_bytes: int,
    current_shards: Sequence[PhoneFfnShardPlacement] = (),
    transition_energy_uj_by_session: Mapping[str, int] = MappingProxyType({}),
    maximum_layouts: int = 128,
    include_unready: bool = False,
    shard_storage: Sequence[PhoneFfnShardStorageMetadata] = (),
    resident_manifests: Sequence[ModelManifest] = (),
    allow_resident_shrink: bool = False,
) -> tuple[PhoneFfnResidencyLayout, ...]:
    """Generate bounded portfolios; opt-in shrinking also permits demanded regrowth."""

    if type(allow_resident_shrink) is not bool:
        raise PhoneShardPlacementError("resident shrink permission must be boolean")
    inputs = _mixed_residency_inputs(
        demands,
        sessions,
        phone_wide_limit_bytes,
        current_shards,
        transition_energy_uj_by_session,
        maximum_layouts,
        include_unready,
        shard_storage,
        resident_manifests,
    )
    if not inputs.usable_sessions:
        return ()
    results: dict[str, PhoneFfnResidencyLayout] = {}
    if allow_resident_shrink:
        for layout in (*_memory_cap_shrink_layouts(inputs), *_memory_cap_growth_layouts(inputs)):
            results[layout.geometry_sha256] = layout

    # A value of -1 retains an inactive artifact already resident in that
    # session. Otherwise it leaves the discovered session unallocated.
    for assignment in product(
        range(-1, len(inputs.artifacts)),
        repeat=len(inputs.usable_sessions),
    ):
        if all(value < 0 for value in assignment):
            continue
        packing = _pack_mixed_assignment(inputs, assignment)
        if packing is None:
            continue
        layout = _mixed_layout(inputs, packing)
        previous = results.get(layout.geometry_sha256)
        if previous is None or (
            layout.objective,
            -layout.queue_benefit,
            layout.changed_session_ids,
        ) < (
            previous.objective,
            -previous.queue_benefit,
            previous.changed_session_ids,
        ):
            results[layout.geometry_sha256] = layout
    return tuple(sorted(results.values(), key=_mixed_layout_rank)[:maximum_layouts])


def progressive_ffn_residency_layouts(
    target: PhoneFfnResidencyLayout,
    *,
    current_shards: Sequence[PhoneFfnShardPlacement] = (),
) -> tuple[PhoneFfnResidencyLayout, ...]:
    """Expand one feasible target into cumulative one-session publications."""

    if not isinstance(target, PhoneFfnResidencyLayout):
        raise PhoneShardPlacementError(
            "progressive phone residency target is invalid"
        )
    current_rows = tuple(current_shards)
    current = {row.session_id: row for row in current_rows}
    target_by_session = {
        row.session_id: row for row in target.shards
    }
    if (
        len(current) != len(current_rows)
        or set(current) - set(target_by_session)
        or any(
            not isinstance(row, PhoneFfnShardPlacement)
            for row in current.values()
        )
    ):
        raise PhoneShardPlacementError(
            "progressive phone residency source is invalid"
        )
    changed = tuple(
        session_id
        for session_id in target_by_session
        if current.get(session_id) != target_by_session[session_id]
    )
    if not changed:
        return ()

    benefit_by_session = dict(target.queue_benefit_by_session)
    order = tuple(sorted(
        changed,
        key=lambda session_id: (
            -benefit_by_session.get(session_id, 0),
            session_id,
        ),
    ))
    working = dict(current)
    stages = []
    for session_id in order:
        working[session_id] = target_by_session[session_id]
        shards = tuple(working[key] for key in sorted(working))
        stage_session_benefit = {
            key: benefit_by_session[key]
            for key in sorted(working)
            if key in benefit_by_session
        }
        stage_artifact_benefit: dict[str, int] = {}
        for key, value in stage_session_benefit.items():
            artifact = working[key].artifact_sha256
            stage_artifact_benefit[artifact] = (
                stage_artifact_benefit.get(artifact, 0) + value
            )
        stage_work = {
            artifact: target.queued_work_by_artifact[artifact]
            for artifact in sorted(stage_artifact_benefit)
        }
        transition_cost = target.transition_cost_by_session.get(
            session_id, 0
        )
        queue_benefit = sum(stage_session_benefit.values())
        stages.append(PhoneFfnResidencyLayout(
            shards=shards,
            queued_work_by_artifact=stage_work,
            queue_benefit_by_artifact=stage_artifact_benefit,
            queue_benefit_by_session=stage_session_benefit,
            queue_benefit=queue_benefit,
            transition_cost=transition_cost,
            transition_cost_by_session={session_id: transition_cost},
            objective=(
                transition_cost - queue_benefit
                if target.objective_kind == "queue_energy_delta_uj"
                else -queue_benefit
            ),
            objective_kind=target.objective_kind,
            changed_session_ids=(session_id,),
            geometry_sha256=_residency_geometry_sha256(shards),
            transition_cost_aggregation=(
                target.transition_cost_aggregation
            ),
        ))
    return tuple(stages)


def select_mixed_ffn_residency_layout(
    layouts: Sequence[PhoneFfnResidencyLayout],
    *,
    current_geometry_sha256: str | None,
    switching_margin_uj: int,
) -> tuple[PhoneFfnResidencyLayout | None, str]:
    """Select the lowest queue objective without reacting to small changes."""

    rows = tuple(layouts)
    if (
        any(not isinstance(row, PhoneFfnResidencyLayout) for row in rows)
        or type(switching_margin_uj) is not int
        or switching_margin_uj < 0
        or (
            current_geometry_sha256 is not None
            and (
                not current_geometry_sha256.startswith("sha256:")
                or len(current_geometry_sha256) != 71
            )
        )
    ):
        raise PhoneShardPlacementError(
            "phone residency selection input is invalid"
        )
    if not rows:
        return None, "NO_FEASIBLE_PHONE_RESIDENCY"
    current = next(
        (
            row for row in rows
            if row.geometry_sha256 == current_geometry_sha256
        ),
        None,
    )
    energy_rows = tuple(
        row for row in rows
        if row.objective_kind == "queue_energy_delta_uj"
    )
    if not energy_rows:
        if current is not None:
            return current, "PHONE_RESIDENCY_ENERGY_UNKNOWN"
        return None, "PHONE_RESIDENCY_ENERGY_UNKNOWN"
    ranked = sorted(energy_rows, key=lambda row: (
        row.objective,
        -row.queue_benefit,
        row.transition_cost,
        tuple(sorted(
            shard.session_id for shard in row.shards
        )),
        row.geometry_sha256,
    ))
    best = ranked[0]
    current = next(
        (
            row for row in energy_rows
            if row.geometry_sha256 == current_geometry_sha256
        ),
        None,
    )
    if current is None:
        if -best.objective <= switching_margin_uj:
            return None, "PHONE_RESIDENCY_SWITCH_MARGIN"
        return best, "PHONE_RESIDENCY_QUEUE_BENEFIT"
    if current.objective - best.objective <= switching_margin_uj:
        return current, "PHONE_RESIDENCY_HYSTERESIS"
    return best, "PHONE_RESIDENCY_QUEUE_BENEFIT"
