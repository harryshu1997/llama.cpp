"""Execution-plan contracts grouped by responsibility: phone."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .common import RuntimePlanError, _integer, _sha256, _text


@dataclass(frozen=True)
class RuntimePhoneShard:
    """One immutable layer shard placed in a phone residency session."""

    session_id: str
    endpoint: str
    layer_mask: int
    maximum_columns: int
    resident_bytes: int
    resident_geometry_sha256: str
    operator_plan_sha256: str
    artifact_sha256: str | None = None
    session_generation: int = 0

    def __post_init__(self) -> None:
        _text("phone shard session id", self.session_id)
        _text("phone shard endpoint", self.endpoint)
        mask = _integer("phone shard layer mask", self.layer_mask, 1)
        if mask >= 1 << 64:
            raise RuntimePlanError("phone shard layer mask exceeds 64 layers")
        _integer("phone shard maximum columns", self.maximum_columns, 1)
        _integer("phone shard resident bytes", self.resident_bytes, 1)
        _sha256(
            "phone shard resident geometry", self.resident_geometry_sha256
        )
        _sha256("phone shard operator plan", self.operator_plan_sha256)
        if self.artifact_sha256 is not None:
            _sha256("phone shard artifact", self.artifact_sha256)
        _integer(
            "phone shard session generation", self.session_generation
        )

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index in range(64)
            if self.layer_mask & (1 << index)
        )

    def to_json(self) -> dict[str, object]:
        return {
            **({} if self.artifact_sha256 is None else {
                "artifact_sha256": self.artifact_sha256,
            }),
            "endpoint": self.endpoint,
            "layer_mask": self.layer_mask,
            "maximum_columns": self.maximum_columns,
            "operator_plan_sha256": self.operator_plan_sha256,
            "resident_bytes": self.resident_bytes,
            "resident_geometry_sha256": self.resident_geometry_sha256,
            "session_id": self.session_id,
            "session_generation": self.session_generation,
        }


def phone_session_map_sha256(
    shards: Sequence[object],
    session_generation_by_id: Mapping[str, int] | None = None,
) -> str:
    """Hash one exact physical phone-session map for transition fencing."""

    generations = dict(session_generation_by_id or {})
    rows = []
    for shard in shards:
        session_id = _text(
            "phone session map session", getattr(shard, "session_id", None)
        )
        generation = generations.get(
            session_id, getattr(shard, "session_generation", None)
        )
        rows.append({
            "artifact_sha256": _sha256(
                "phone session map artifact",
                getattr(shard, "artifact_sha256", None),
            ),
            "endpoint": _text(
                "phone session map endpoint", getattr(shard, "endpoint", None)
            ),
            "layer_mask": _integer(
                "phone session map layer mask",
                getattr(shard, "layer_mask", None),
                1,
            ),
            "maximum_columns": _integer(
                "phone session map columns",
                getattr(shard, "maximum_columns", None),
                1,
            ),
            "operator_plan_sha256": _sha256(
                "phone session map operator plan",
                getattr(shard, "operator_plan_sha256", None),
            ),
            "resident_bytes": _integer(
                "phone session map resident bytes",
                getattr(shard, "resident_bytes", None),
                1,
            ),
            "resident_geometry_sha256": _sha256(
                "phone session map geometry",
                getattr(shard, "resident_geometry_sha256", None),
            ),
            "session_generation": _integer(
                "phone session map generation", generation, 1
            ),
            "session_id": session_id,
        })
    rows.sort(key=lambda row: row["session_id"])
    if not rows or len({row["session_id"] for row in rows}) != len(rows):
        raise RuntimePlanError("phone session map is invalid")
    payload = json.dumps(
        {
            "schema": "research-scheduler-phone-session-map-v1",
            "sessions": rows,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def phone_session_assignment_sha256(
    selected_session_id: str,
    source_layout_hash: str,
    target_layout_hash: str,
    source_generation: int,
) -> str:
    """Hash the immutable one-session assignment selected by the scheduler."""

    body = {
        "schema": "research-scheduler-phone-session-assignment-v1",
        "selected_session_id": _text(
            "phone session assignment session", selected_session_id
        ),
        "source_generation": _integer(
            "phone session assignment source generation", source_generation
        ),
        "source_layout_hash": _sha256(
            "phone session assignment source layout", source_layout_hash
        ),
        "target_generation": source_generation + 1,
        "target_layout_hash": _sha256(
            "phone session assignment target layout", target_layout_hash
        ),
    }
    payload = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PhoneSessionReplacementAuthorization:
    """Immutable authority for one copy-on-write physical replacement."""

    selected_session_id: str
    source_layout_hash: str
    target_layout_hash: str
    source_generation: int
    assignment_hash: str

    def __post_init__(self) -> None:
        selected = _text(
            "phone replacement selected session", self.selected_session_id
        )
        source_layout = _sha256(
            "phone replacement source layout", self.source_layout_hash
        )
        target_layout = _sha256(
            "phone replacement target layout", self.target_layout_hash
        )
        source_generation = _integer(
            "phone replacement source generation", self.source_generation
        )
        assignment = _sha256(
            "phone replacement assignment", self.assignment_hash
        )
        if assignment != phone_session_assignment_sha256(
            selected,
            source_layout,
            target_layout,
            source_generation,
        ):
            raise RuntimePlanError(
                "phone replacement assignment hash differs"
            )

    @property
    def target_generation(self) -> int:
        return self.source_generation + 1

    @classmethod
    def create(
        cls,
        *,
        selected_session_id: str,
        source_shards: Sequence[object],
        target_shards: Sequence[object],
        source_generation_by_id: Mapping[str, int] | None = None,
        target_generation_by_id: Mapping[str, int] | None = None,
    ) -> "PhoneSessionReplacementAuthorization":
        source_layout_hash = phone_session_map_sha256(
            source_shards, source_generation_by_id
        )
        target_layout_hash = phone_session_map_sha256(
            target_shards, target_generation_by_id
        )
        source_by_id = dict(source_generation_by_id or {})
        source_generation = source_by_id.get(selected_session_id)
        if source_generation is None:
            source = next((
                row for row in source_shards
                if getattr(row, "session_id", None) == selected_session_id
            ), None)
            source_generation = getattr(source, "session_generation", 0)
        source_generation = _integer(
            "phone replacement source generation", source_generation
        )
        return cls(
            selected_session_id=selected_session_id,
            source_layout_hash=source_layout_hash,
            target_layout_hash=target_layout_hash,
            source_generation=source_generation,
            assignment_hash=phone_session_assignment_sha256(
                selected_session_id,
                source_layout_hash,
                target_layout_hash,
                source_generation,
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "assignment_hash": self.assignment_hash,
            "selected_session_id": self.selected_session_id,
            "source_generation": self.source_generation,
            "source_layout_hash": self.source_layout_hash,
            "target_generation": self.target_generation,
            "target_layout_hash": self.target_layout_hash,
        }
