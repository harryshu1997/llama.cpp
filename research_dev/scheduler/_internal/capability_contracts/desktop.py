"""Device, executor, catalog and snapshot capability contracts: desktop."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Sequence

from ..runtime_system_cost import ROUTE_MATURITY_STATES
from ..plan_contracts.remote_resident import RuntimeRemoteResidentFfn
from .common import (
    RuntimeCapabilityError,
    SPLIT_AXES,
    _boolean,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)


@dataclass(frozen=True)
class RuntimeCompositeOperatorPlacement:
    """One exact operator assignment implemented by a coordinator."""

    operator_id: str
    primary_device_id: str
    helper_device_id: str | None
    split_axis: str
    split_fraction_ppm: int
    assisted: bool = False

    def __post_init__(self) -> None:
        _text("composite operator id", self.operator_id)
        _text("composite operator primary device", self.primary_device_id)
        _boolean("composite operator assisted", self.assisted)
        if self.helper_device_id is not None:
            _text("composite operator helper device", self.helper_device_id)
            if self.helper_device_id == self.primary_device_id:
                raise RuntimeCapabilityError(
                    "composite operator devices must differ"
                )
        if self.split_axis != "none" and self.split_axis not in SPLIT_AXES:
            raise RuntimeCapabilityError(
                "composite operator split axis is unsupported"
            )
        split = _integer(
            "composite operator split fraction", self.split_fraction_ppm
        )
        if self.helper_device_id is None:
            if self.split_axis != "none" or split:
                raise RuntimeCapabilityError(
                    "single-device operator carries a split"
                )
        elif self.split_axis == "none" or not 0 < split < 1_000_000:
            raise RuntimeCapabilityError(
                "assisted operator requires a proper split"
            )

    def to_json(self) -> dict[str, object]:
        result = {
            "helper_device_id": self.helper_device_id,
            "operator_id": self.operator_id,
            "primary_device_id": self.primary_device_id,
            "split_axis": self.split_axis,
            "split_fraction_ppm": self.split_fraction_ppm,
        }
        if self.assisted:
            result["assisted"] = True
        return result

    @classmethod
    def from_json(
        cls, value: object
    ) -> "RuntimeCompositeOperatorPlacement":
        row = _object("runtime composite operator placement", value)
        return cls(
            operator_id=row.get("operator_id"),
            primary_device_id=row.get("primary_device_id"),
            helper_device_id=row.get("helper_device_id"),
            split_axis=row.get("split_axis"),
            split_fraction_ppm=row.get("split_fraction_ppm"),
            assisted=row.get("assisted", False),
        )


def desktop_control_placement_payload(
    artifact_sha256: str,
    placements: Sequence[RuntimeCompositeOperatorPlacement],
    cuda_graph_mode: str = "default",
    remote_resident_ffn: RuntimeRemoteResidentFfn | None = None,
) -> dict[str, object]:
    if cuda_graph_mode not in ("default", "disabled"):
        raise RuntimeCapabilityError("desktop control CUDA graph mode is invalid")
    payload = {
        "artifact_sha256": artifact_sha256,
        "operator_placements": [
            row.to_json() for row in sorted(placements, key=lambda row: row.operator_id)
        ],
    }
    # Omitted mode in historical profiles means default, never disabled.
    if cuda_graph_mode != "default":
        payload["cuda_graph_mode"] = cuda_graph_mode
    if remote_resident_ffn is not None:
        # A desktop parent without these weights is a different placement: it is qualified,
        # launched and identified separately. Omitted when absent so historical hashes hold.
        if not isinstance(remote_resident_ffn, RuntimeRemoteResidentFfn):
            raise RuntimeCapabilityError("desktop control remote-resident group is invalid")
        if remote_resident_ffn.parent_artifact_sha256 != artifact_sha256:
            raise RuntimeCapabilityError(
                "desktop control remote-resident group belongs to another artifact"
            )
        payload["remote_resident_ffn"] = remote_resident_ffn.placement_payload()
    return payload


@dataclass(frozen=True)
class RuntimeDesktopControlProfile:
    """A measured, frozen all-desktop placement for one model artifact."""

    profile_id: str
    artifact_sha256: str
    executor_id: str
    operator_placements: tuple[RuntimeCompositeOperatorPlacement, ...]
    maturity: str
    evidence_ids: tuple[str, ...]
    cuda_graph_mode: str = "default"
    remote_resident_ffn: RuntimeRemoteResidentFfn | None = None
    _placement_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("profile_id", "executor_id"):
            _text("desktop control " + name, getattr(self, name))
        artifact = _text(
            "desktop control artifact hash", self.artifact_sha256
        )
        if (
            not artifact.startswith("sha256:")
            or len(artifact) != 71
            or any(value not in "0123456789abcdef" for value in artifact[7:])
        ):
            raise RuntimeCapabilityError(
                "desktop control artifact hash must be SHA-256"
            )
        placements = tuple(self.operator_placements)
        if (
            not placements
            or any(
                not isinstance(row, RuntimeCompositeOperatorPlacement)
                or row.helper_device_id is not None
                or row.split_axis != "none"
                or row.split_fraction_ppm != 0
                or row.assisted
                for row in placements
            )
            or len({row.operator_id for row in placements}) != len(placements)
        ):
            raise RuntimeCapabilityError(
                "desktop control operator placements are invalid"
            )
        if self.maturity not in ROUTE_MATURITY_STATES:
            raise RuntimeCapabilityError(
                "desktop control maturity is invalid"
            )
        evidence = _texts(
            "desktop control evidence id", self.evidence_ids
        )
        placements = tuple(sorted(
            placements, key=lambda row: row.operator_id
        ))
        payload = desktop_control_placement_payload(
            artifact, placements, self.cuda_graph_mode, self.remote_resident_ffn
        )
        placement_sha256 = "sha256:" + hashlib.sha256(json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")).hexdigest()
        object.__setattr__(self, "operator_placements", placements)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "_placement_sha256", placement_sha256)

    @property
    def placement_sha256(self) -> str:
        return self._placement_sha256

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "evidence_ids": list(self.evidence_ids),
            "executor_id": self.executor_id,
            "maturity": self.maturity,
            "operator_placements": [
                row.to_json() for row in self.operator_placements
            ],
            "placement_sha256": self.placement_sha256,
            "profile_id": self.profile_id,
            **({"cuda_graph_mode": self.cuda_graph_mode}
               if self.cuda_graph_mode != "default" else {}),
            **({"remote_resident_ffn": self.remote_resident_ffn.to_json()}
               if self.remote_resident_ffn is not None else {}),
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeDesktopControlProfile":
        row = _object("runtime desktop control profile", value)
        result = cls(
            profile_id=row.get("profile_id"),
            artifact_sha256=row.get("artifact_sha256"),
            executor_id=row.get("executor_id"),
            cuda_graph_mode=row.get("cuda_graph_mode", "default"),
            remote_resident_ffn=(
                None if row.get("remote_resident_ffn") is None
                else RuntimeRemoteResidentFfn.from_json(row["remote_resident_ffn"])
            ),
            operator_placements=tuple(
                RuntimeCompositeOperatorPlacement.from_json(item)
                for item in _list(
                    "runtime desktop control operator placements",
                    row.get("operator_placements"),
                )
            ),
            maturity=row.get("maturity"),
            evidence_ids=tuple(_list(
                "runtime desktop control evidence ids",
                row.get("evidence_ids"),
            )),
        )
        if row.get("placement_sha256") != result.placement_sha256:
            raise RuntimeCapabilityError(
                "desktop control placement hash differs"
            )
        return result
