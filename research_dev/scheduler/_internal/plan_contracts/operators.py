"""Execution-plan contracts grouped by responsibility: operators."""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading

from ..runtime_capabilities import SPLIT_AXES
from .common import RUNTIME_BATCH_PLANS, RUNTIME_EXECUTION_MODES, RuntimePlanError, _integer, _text
from .phone import RuntimePhoneShard
from .remote_resident import RuntimeRemoteResidentFfn


_OPERATOR_JSON_CACHE_LIMIT = 4096


_OPERATOR_JSON_PLACEHOLDER = "__runtime_operator_rows__"


_operator_json_cache: dict[
    int, tuple[tuple["RuntimeOperatorAssignment", ...], bytes]
] = {}


_operator_json_cache_lock = threading.Lock()


@dataclass(frozen=True)
class RuntimeExecutionContract:
    """Hash-bound data-plane instructions selected with one route."""

    execution_mode: str
    initial_split_fraction_ppm: int
    allowed_adaptive_fractions_ppm: tuple[int, ...]
    batch_plan: str
    maximum_batch_size: int
    queue_depth: int
    phone_device_id: str | None = None
    phone_endpoint: str | None = None
    operator_kind: str | None = None
    phone_shards: tuple[RuntimePhoneShard, ...] = ()
    # dense FFN layers whose weights exist only on phone sessions (desktop parents only)
    remote_resident_ffn: RuntimeRemoteResidentFfn | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in RUNTIME_EXECUTION_MODES:
            raise RuntimePlanError("execution contract mode is invalid")
        if self.remote_resident_ffn is not None and (
            not isinstance(self.remote_resident_ffn, RuntimeRemoteResidentFfn)
            or self.execution_mode != "desktop"
        ):
            raise RuntimePlanError(
                "remote-resident FFN weights require the desktop parent contract"
            )
        if self.batch_plan not in RUNTIME_BATCH_PLANS:
            raise RuntimePlanError("execution contract batch plan is invalid")
        initial = _integer(
            "execution contract initial split",
            self.initial_split_fraction_ppm,
        )
        if initial > 1_000_000:
            raise RuntimePlanError(
                "execution contract initial split exceeds one"
            )
        fractions = tuple(sorted(
            _integer("execution contract adaptive fraction", value)
            for value in self.allowed_adaptive_fractions_ppm
        ))
        if (
            len(fractions) != len(set(fractions))
            or any(value > 1_000_000 for value in fractions)
        ):
            raise RuntimePlanError(
                "execution contract adaptive fractions are invalid"
            )
        _integer(
            "execution contract maximum batch size",
            self.maximum_batch_size,
            1,
        )
        _integer("execution contract queue depth", self.queue_depth, 1)
        phone_values = (
            self.phone_device_id,
            self.phone_endpoint,
            self.operator_kind,
        )
        shards = tuple(self.phone_shards)
        if any(not isinstance(row, RuntimePhoneShard) for row in shards):
            raise RuntimePlanError("execution contract phone shard is invalid")
        if (
            len({row.session_id for row in shards}) != len(shards)
            or len({row.endpoint for row in shards}) != len(shards)
        ):
            raise RuntimePlanError(
                "execution contract phone shard identity is duplicated"
            )
        covered_layers_by_artifact: dict[str, int] = {}
        for shard in shards:
            artifact = shard.artifact_sha256 or "legacy-single-artifact"
            covered_layers = covered_layers_by_artifact.get(artifact, 0)
            if covered_layers & shard.layer_mask:
                raise RuntimePlanError(
                    "execution contract phone shard layers overlap"
                )
            covered_layers_by_artifact[artifact] = (
                covered_layers | shard.layer_mask
            )
        if self.execution_mode == "desktop":
            if (
                initial != 0
                or fractions
                or self.batch_plan != "none"
                or self.maximum_batch_size != 1
                or self.queue_depth != 1
                or any(value is not None for value in phone_values)
                or shards
            ):
                raise RuntimePlanError(
                    "desktop execution contract carries phone work"
                )
        else:
            for name, value in zip(
                ("phone device", "phone endpoint", "operator kind"),
                phone_values,
            ):
                _text("execution contract " + name, value)
            if self.batch_plan == "none":
                raise RuntimePlanError(
                    "phone execution contract lacks a batch plan"
                )
            if self.batch_plan == "single" and self.maximum_batch_size != 1:
                raise RuntimePlanError(
                    "single execution contract has a batched capacity"
                )
            if self.execution_mode == "static-split":
                if initial <= 0 or fractions:
                    raise RuntimePlanError(
                        "static split execution contract is invalid"
                    )
            elif (
                initial not in fractions
                or not fractions
                or fractions[0] != 0
                or not any(value > 0 for value in fractions)
            ):
                raise RuntimePlanError(
                    "adaptive split execution contract is invalid"
                )
        object.__setattr__(
            self, "allowed_adaptive_fractions_ppm", fractions
        )
        object.__setattr__(
            self,
            "phone_shards",
            tuple(sorted(shards, key=lambda row: row.session_id)),
        )

    @classmethod
    def desktop(
        cls, remote_resident_ffn: RuntimeRemoteResidentFfn | None = None
    ) -> "RuntimeExecutionContract":
        return cls(
            execution_mode="desktop",
            initial_split_fraction_ppm=0,
            allowed_adaptive_fractions_ppm=(),
            batch_plan="none",
            maximum_batch_size=1,
            queue_depth=1,
            remote_resident_ffn=remote_resident_ffn,
        )

    def resident_phone_shards(self, maximum_columns: int) -> tuple[RuntimePhoneShard, ...]:
        """Resolve the same residency identity for helpers and remote weight owners."""
        remote = self.remote_resident_ffn
        if remote is None:
            return self.phone_shards
        if not remote.bound or any(row.operator_plan_sha256 is None for row in remote.sessions):
            raise RuntimePlanError("remote-resident owners are not physically bound")
        return tuple(RuntimePhoneShard(
            session_id=row.session_id,
            endpoint=row.endpoint,
            layer_mask=row.layer_mask,
            maximum_columns=maximum_columns,
            resident_bytes=row.resident_bytes,
            resident_geometry_sha256=row.resident_geometry_sha256,
            operator_plan_sha256=row.operator_plan_sha256,
            artifact_sha256=remote.parent_artifact_sha256,
            session_generation=row.session_generation,
        ) for row in remote.sessions)

    def to_json(self) -> dict[str, object]:
        result = {
            "allowed_adaptive_fractions_ppm": list(
                self.allowed_adaptive_fractions_ppm
            ),
            "batch_plan": self.batch_plan,
            "execution_mode": self.execution_mode,
            "initial_split_fraction_ppm": (
                self.initial_split_fraction_ppm
            ),
            "maximum_batch_size": self.maximum_batch_size,
            "queue_depth": self.queue_depth,
        }
        if self.phone_device_id is not None:
            result.update({
                "operator_kind": self.operator_kind,
                "phone_device_id": self.phone_device_id,
                "phone_endpoint": self.phone_endpoint,
            })
        if self.phone_shards:
            result["phone_shards"] = [
                row.to_json() for row in self.phone_shards
            ]
        if self.remote_resident_ffn is not None:
            # omitted when absent so every existing plan hash is unchanged
            result["remote_resident_ffn"] = self.remote_resident_ffn.to_json()
        return result


@dataclass(frozen=True)
class RuntimeOperatorAssignment:
    operator_id: str
    operator_kind: str
    candidate_id: str
    device_ids: tuple[str, ...]
    split_axis: str
    split_fraction_ppm: int
    kernel_profile_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("operator_id", "operator_kind", "candidate_id"):
            _text(f"operator assignment {name}", getattr(self, name))
        devices = tuple(self.device_ids)
        if (
            not devices
            or len(devices) != len(set(devices))
            or any(type(value) is not str or not value or not value.isascii() for value in devices)
        ):
            raise RuntimePlanError("operator assignment devices are invalid")
        if self.split_axis != "none" and self.split_axis not in SPLIT_AXES:
            raise RuntimePlanError("operator assignment split axis is invalid")
        _integer("operator assignment split fraction", self.split_fraction_ppm)
        if self.split_axis == "none" and self.split_fraction_ppm != 0:
            raise RuntimePlanError("unsplit operator has a split fraction")
        if self.split_axis != "none" and not 0 < self.split_fraction_ppm < 1_000_000:
            raise RuntimePlanError("split operator fraction is invalid")
        object.__setattr__(self, "device_ids", tuple(sorted(devices)))
        profiles = tuple(
            _text("operator assignment kernel profile", value)
            for value in self.kernel_profile_ids
        )
        if len(profiles) != len(set(profiles)):
            raise RuntimePlanError(
                "operator assignment kernel profiles are duplicated"
            )
        object.__setattr__(self, "kernel_profile_ids", tuple(sorted(profiles)))

    def to_json(self) -> dict[str, object]:
        result = {
            "candidate_id": self.candidate_id,
            "device_ids": list(self.device_ids),
            "operator_id": self.operator_id,
            "operator_kind": self.operator_kind,
            "split_axis": self.split_axis,
            "split_fraction_ppm": self.split_fraction_ppm,
        }
        if self.kernel_profile_ids:
            result["kernel_profile_ids"] = list(self.kernel_profile_ids)
        return result


def _canonical_operator_json(
    operators: tuple[RuntimeOperatorAssignment, ...],
) -> bytes:
    cache_key = id(operators)
    with _operator_json_cache_lock:
        cached = _operator_json_cache.get(cache_key)
        if cached is not None and cached[0] is operators:
            return cached[1]
    encoded = json.dumps(
        [row.to_json() for row in operators],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    with _operator_json_cache_lock:
        if len(_operator_json_cache) >= _OPERATOR_JSON_CACHE_LIMIT:
            _operator_json_cache.pop(next(iter(_operator_json_cache)))
        _operator_json_cache[cache_key] = (operators, encoded)
    return encoded
