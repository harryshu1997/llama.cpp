"""Execution-plan contracts grouped by responsibility: remote-resident FFN weights.

An *assisted copy* (``RuntimePhoneShard`` in a helper plan) keeps the complete weights on the
desktop and lets a phone session compute a suffix of the FFN columns. A *remote-resident* group
is different: the desktop process never allocates or loads the dense gate/up/down weights of
the listed layers, so the owning phone sessions are a hard execution dependency of the desktop
parent. The group records the parent artifact, the tensor dependencies, the geometry, the dtype,
the shard hashes, the device sessions with their generations and the local backing files, and it
participates in the desktop placement identity, the plan hash and the physical launch contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Mapping

from .common import RuntimePlanError, _integer, _sha256, _text


REMOTE_RESIDENT_FFN_SCHEMA = "research-scheduler-remote-resident-ffn-v1"
REMOTE_RESIDENT_FFN_DTYPES = frozenset({"f16"})
REMOTE_RESIDENT_FFN_TENSOR_KINDS = ("ffn_down", "ffn_gate", "ffn_up")


def _layer_indices(mask: int) -> tuple[int, ...]:
    return tuple(index for index in range(64) if mask & (1 << index))


def remote_resident_tensor_ids(layer_mask: int) -> tuple[str, ...]:
    """The exact dense FFN weight tensors omitted from the desktop for ``layer_mask``."""
    return tuple(sorted(
        f"blk.{index}.{kind}.weight"
        for index in _layer_indices(layer_mask)
        for kind in REMOTE_RESIDENT_FFN_TENSOR_KINDS
    ))


def _canonical_sha256(payload: object) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class RuntimeRemoteResidentSession:
    """One phone session that owns the only copy of a layer group's FFN weights."""

    session_id: str
    endpoint: str
    layer_mask: int
    shard_sha256: str
    resident_geometry_sha256: str
    resident_bytes: int
    remote_path: str
    session_generation: int = 0
    operator_plan_sha256: str | None = None

    def __post_init__(self) -> None:
        _text("remote-resident session id", self.session_id)
        _text("remote-resident session endpoint", self.endpoint)
        mask = _integer("remote-resident session layer mask", self.layer_mask, 1)
        if mask >= 1 << 64:
            raise RuntimePlanError("remote-resident session layer mask exceeds 64 layers")
        _sha256("remote-resident shard hash", self.shard_sha256)
        _sha256("remote-resident geometry", self.resident_geometry_sha256)
        _integer("remote-resident session resident bytes", self.resident_bytes, 1)
        path = _text("remote-resident backing file", self.remote_path)
        if not path.startswith("/"):
            raise RuntimePlanError("remote-resident backing file must be an absolute device path")
        _integer("remote-resident session generation", self.session_generation)
        if self.operator_plan_sha256 is not None:
            _sha256("remote-resident session operator plan", self.operator_plan_sha256)

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return _layer_indices(self.layer_mask)

    def placement_payload(self) -> dict[str, object]:
        """Generation-free identity used by the desktop placement hash."""
        return {
            "endpoint": self.endpoint,
            "layer_mask": self.layer_mask,
            "remote_path": self.remote_path,
            "resident_bytes": self.resident_bytes,
            "resident_geometry_sha256": self.resident_geometry_sha256,
            "session_id": self.session_id,
            "shard_sha256": self.shard_sha256,
        }

    def to_json(self) -> dict[str, object]:
        result = {**self.placement_payload(), "session_generation": self.session_generation}
        if self.operator_plan_sha256 is not None:
            result["operator_plan_sha256"] = self.operator_plan_sha256
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimeRemoteResidentSession":
        if not isinstance(value, Mapping):
            raise RuntimePlanError("remote-resident session row is invalid")
        return cls(
            session_id=value.get("session_id"),
            endpoint=value.get("endpoint"),
            layer_mask=value.get("layer_mask"),
            shard_sha256=value.get("shard_sha256"),
            resident_geometry_sha256=value.get("resident_geometry_sha256"),
            resident_bytes=value.get("resident_bytes"),
            remote_path=value.get("remote_path"),
            session_generation=value.get("session_generation", 0),
            operator_plan_sha256=value.get("operator_plan_sha256"),
        )


@dataclass(frozen=True)
class RuntimeRemoteResidentFfn:
    """Dense FFN layers whose weights exist only on verified phone sessions."""

    parent_artifact_sha256: str
    layer_mask: int
    dtype: str
    omitted_bytes: int
    tensor_ids: tuple[str, ...]
    shard_index_sha256: str
    sessions: tuple[RuntimeRemoteResidentSession, ...]
    _geometry_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _sha256("remote-resident parent artifact", self.parent_artifact_sha256)
        mask = _integer("remote-resident layer mask", self.layer_mask, 1)
        if mask >= 1 << 64:
            raise RuntimePlanError("remote-resident layer mask exceeds 64 layers")
        if self.dtype not in REMOTE_RESIDENT_FFN_DTYPES:
            raise RuntimePlanError("remote-resident dtype is unsupported")
        _integer("remote-resident omitted bytes", self.omitted_bytes, 1)
        _sha256("remote-resident shard index", self.shard_index_sha256)
        sessions = tuple(self.sessions)
        if not sessions or any(
            not isinstance(row, RuntimeRemoteResidentSession) for row in sessions
        ):
            raise RuntimePlanError("remote-resident group requires phone sessions")
        if (
            len({row.session_id for row in sessions}) != len(sessions)
            or len({row.endpoint for row in sessions}) != len(sessions)
        ):
            raise RuntimePlanError("remote-resident session identity is duplicated")
        covered = 0
        for row in sessions:
            if covered & row.layer_mask:
                raise RuntimePlanError("remote-resident session layers overlap")
            covered |= row.layer_mask
        if covered != mask:
            raise RuntimePlanError(
                "remote-resident sessions do not cover the layer mask exactly"
            )
        tensor_ids = tuple(self.tensor_ids)
        if tensor_ids != remote_resident_tensor_ids(mask):
            raise RuntimePlanError(
                "remote-resident tensor dependencies must be the complete dense "
                "gate/up/down groups of the masked layers"
            )
        sessions = tuple(sorted(sessions, key=lambda row: row.session_id))
        object.__setattr__(self, "sessions", sessions)
        object.__setattr__(self, "tensor_ids", tensor_ids)
        object.__setattr__(
            self, "_geometry_sha256", _canonical_sha256(self.placement_payload())
        )

    @property
    def geometry_sha256(self) -> str:
        """Generation-free identity of the remote group (artifact, tensors, shards, sessions)."""
        return self._geometry_sha256

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return _layer_indices(self.layer_mask)

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(row.session_id for row in self.sessions)

    @property
    def session_generation_by_id(self) -> dict[str, int]:
        return {row.session_id: row.session_generation for row in self.sessions}

    @property
    def bound(self) -> bool:
        """True when every owning session carries a physical generation."""
        return all(row.session_generation >= 1 for row in self.sessions)

    def with_session_generations(
        self, generation_by_id: Mapping[str, int], *,
        operator_plan_by_id: Mapping[str, str] | None = None,
    ) -> "RuntimeRemoteResidentFfn":
        """Bind the physical session generations; every session must be covered."""
        missing = set(self.session_ids) - set(generation_by_id)
        if missing:
            raise RuntimePlanError(
                "remote-resident generation binding lacks sessions: "
                + ",".join(sorted(missing))
            )
        rows = []
        for row in self.sessions:
            generation = _integer(
                "remote-resident bound generation", generation_by_id[row.session_id], 1
            )
            rows.append(RuntimeRemoteResidentSession(
                session_id=row.session_id,
                endpoint=row.endpoint,
                layer_mask=row.layer_mask,
                shard_sha256=row.shard_sha256,
                resident_geometry_sha256=row.resident_geometry_sha256,
                resident_bytes=row.resident_bytes,
                remote_path=row.remote_path,
                session_generation=generation,
                operator_plan_sha256=(
                    row.operator_plan_sha256 if operator_plan_by_id is None
                    else operator_plan_by_id.get(row.session_id)
                ),
            ))
        return RuntimeRemoteResidentFfn(
            parent_artifact_sha256=self.parent_artifact_sha256,
            layer_mask=self.layer_mask,
            dtype=self.dtype,
            omitted_bytes=self.omitted_bytes,
            tensor_ids=self.tensor_ids,
            shard_index_sha256=self.shard_index_sha256,
            sessions=tuple(rows),
        )

    def placement_payload(self) -> dict[str, object]:
        """Generation-free payload folded into the desktop placement identity."""
        return {
            "dtype": self.dtype,
            "layer_mask": self.layer_mask,
            "omitted_bytes": self.omitted_bytes,
            "parent_artifact_sha256": self.parent_artifact_sha256,
            "schema": REMOTE_RESIDENT_FFN_SCHEMA,
            "sessions": [row.placement_payload() for row in self.sessions],
            "shard_index_sha256": self.shard_index_sha256,
            "tensor_ids": list(self.tensor_ids),
        }

    def to_json(self) -> dict[str, object]:
        return {
            **self.placement_payload(),
            "geometry_sha256": self.geometry_sha256,
            "sessions": [row.to_json() for row in self.sessions],
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeRemoteResidentFfn":
        if not isinstance(value, Mapping):
            raise RuntimePlanError("remote-resident group is invalid")
        if value.get("schema", REMOTE_RESIDENT_FFN_SCHEMA) != REMOTE_RESIDENT_FFN_SCHEMA:
            raise RuntimePlanError("remote-resident group schema is unsupported")
        sessions = value.get("sessions")
        if not isinstance(sessions, (list, tuple)):
            raise RuntimePlanError("remote-resident group sessions are invalid")
        tensor_ids = value.get("tensor_ids")
        if not isinstance(tensor_ids, (list, tuple)):
            raise RuntimePlanError("remote-resident group tensors are invalid")
        result = cls(
            parent_artifact_sha256=value.get("parent_artifact_sha256"),
            layer_mask=value.get("layer_mask"),
            dtype=value.get("dtype"),
            omitted_bytes=value.get("omitted_bytes"),
            tensor_ids=tuple(tensor_ids),
            shard_index_sha256=value.get("shard_index_sha256"),
            sessions=tuple(
                RuntimeRemoteResidentSession.from_json(row) for row in sessions
            ),
        )
        recorded = value.get("geometry_sha256")
        if recorded is not None and recorded != result.geometry_sha256:
            raise RuntimePlanError("remote-resident geometry hash differs")
        return result
