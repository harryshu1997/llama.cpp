"""Typed scheduler configuration manifests: campaign."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .._internal.types import canonical_sha256
from .common import (
    CAMPAIGN_MANIFEST_SCHEMA,
    _SHA256,
    _boolean,
    _integer,
    _object,
    _optional_path,
    _optional_text,
    _path,
    _path_json,
    _require,
    _sequence,
    _text,
)


@dataclass(frozen=True)
class TraceConfiguration:
    large_requests_path: Path
    overlay_requests_path: Path
    trace_manifest_path: Path
    replay_schedule_path: Path | None
    request_indices: tuple[int, ...]
    arrival_scale: int | None

    @classmethod
    def from_json(cls, value: object, base: Path) -> "TraceConfiguration":
        row = _object(value, "campaign trace")
        request_indices = tuple(
            _integer(item, "request index")
            for item in _sequence(row.get("request_indices", []), "request indices")
        )
        _require(
            len(request_indices) == len(set(request_indices)),
            "request indices are not unique",
        )
        replay = _optional_path(row.get("replay_schedule_path"), base, "replay schedule")
        scale_value = row.get("arrival_scale")
        scale = None if scale_value is None else _integer(
            scale_value, "arrival scale", minimum=1
        )
        _require(
            replay is None or not request_indices and scale is None,
            "replay schedule cannot be combined with replay transforms",
        )
        return cls(
            large_requests_path=_path(row.get("large_requests_path"), base, "large trace"),
            overlay_requests_path=_path(row.get("overlay_requests_path"), base, "overlay trace"),
            trace_manifest_path=_path(row.get("trace_manifest_path"), base, "trace manifest"),
            replay_schedule_path=replay,
            request_indices=request_indices,
            arrival_scale=scale,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "arrival_scale": self.arrival_scale,
            "large_requests_path": str(self.large_requests_path),
            "overlay_requests_path": str(self.overlay_requests_path),
            "replay_schedule_path": _path_json(self.replay_schedule_path),
            "request_indices": list(self.request_indices),
            "trace_manifest_path": str(self.trace_manifest_path),
        }


@dataclass(frozen=True)
class FixedPhoneResidencyConfiguration:
    """An experimental assignment, independent of arrived-work policy."""

    reference_id: str
    assignments: tuple[tuple[str, str, int, int], ...]

    def __post_init__(self) -> None:
        _text(self.reference_id, "fixed residency reference")
        _require(type(self.assignments) is tuple, "fixed residency assignments must be a tuple")
        rows = self.assignments
        _require(bool(rows), "fixed residency assignments are empty")
        sessions = set()
        masks = {}
        for row in rows:
            _require(type(row) is tuple and len(row) == 4, "fixed residency assignment is invalid")
            session_id, artifact, mask, columns = row
            _text(session_id, "fixed residency session")
            _require(session_id not in sessions, "fixed residency session is duplicated")
            sessions.add(session_id)
            _require(
                type(artifact) is str and artifact.startswith("sha256:")
                and _SHA256.fullmatch(artifact) is not None,
                "fixed residency artifact hash is invalid",
            )
            _integer(mask, "fixed residency layer mask", minimum=1, maximum=(1 << 64) - 1)
            _integer(columns, "fixed residency columns", minimum=1)
            _require(not masks.get(artifact, 0) & mask,
                     "fixed residency layer masks overlap")
            masks[artifact] = masks.get(artifact, 0) | mask
        object.__setattr__(self, "assignments", tuple(sorted(rows)))

    @classmethod
    def from_json(cls, value: object) -> "FixedPhoneResidencyConfiguration":
        row = _object(value, "fixed phone residency")
        _require(row.get("mode") == "fixed", "fixed residency mode is invalid")
        assignments = []
        for value in _sequence(row.get("assignments"), "fixed residency assignments"):
            shard = _object(value, "fixed residency assignment")
            assignments.append((
                shard.get("session_id"), shard.get("artifact_sha256"),
                shard.get("layer_mask"), shard.get("maximum_columns"),
            ))
        return cls(row.get("reference_id"), tuple(assignments))

    def to_json(self) -> dict[str, object]:
        return {
            "mode": "fixed",
            "reference_id": self.reference_id,
            "assignments": [
                {"session_id": session, "artifact_sha256": artifact,
                 "layer_mask": mask, "maximum_columns": columns}
                for session, artifact, mask, columns in self.assignments
            ],
        }

    @property
    def assignment_sha256(self) -> str:
        return canonical_sha256(self.to_json())


@dataclass(frozen=True)
class PhoneHtpMemoryCapConfiguration:
    phone_device_id: str
    cap_bytes: int
    workspace_bytes: int = 0

    def __post_init__(self) -> None:
        _text(self.phone_device_id, "phone HTP memory cap device")
        _integer(self.cap_bytes, "phone HTP memory cap")
        _integer(self.workspace_bytes, "phone HTP workspace")
        _require(self.workspace_bytes <= self.cap_bytes, "phone HTP workspace exceeds cap")

    @classmethod
    def from_json(cls, value: object) -> "PhoneHtpMemoryCapConfiguration":
        row = _object(value, "phone HTP memory cap")
        return cls(row.get("phone_device_id"), row.get("cap_bytes"), row.get("workspace_bytes", 0))

    def to_json(self) -> dict[str, object]:
        return {"phone_device_id": self.phone_device_id, "cap_bytes": self.cap_bytes,
                "workspace_bytes": self.workspace_bytes}


@dataclass(frozen=True)
class CampaignManifest:
    campaign_id: str
    rig_manifest_path: Path
    models_manifest_path: Path
    evidence_manifest_path: Path
    trace: TraceConfiguration
    selection_mode: str
    maximum_latency_ppm: int
    minimum_energy_saving_ppm: int
    energy_attribution_kind: str
    max_workers: int
    adaptive_minimum_remaining_tokens: int | None
    maximum_phone_sessions: int | None
    reference_catalog_path: Path | None
    reference_replay_schedule_path: Path | None
    reference_runner_command_path: Path | None
    helper_preparation_fault_injection: str | None = None
    fixed_phone_residency: FixedPhoneResidencyConfiguration | None = None
    include_startup_preparation: bool = False
    phone_htp_memory_caps: tuple[PhoneHtpMemoryCapConfiguration, ...] = ()
    protected_work_policy: str = "strict"

    @classmethod
    def from_json(cls, value: object, base: Path) -> "CampaignManifest":
        row = _object(value, "campaign manifest")
        _require(
            row.get("schema") == CAMPAIGN_MANIFEST_SCHEMA,
            "campaign manifest schema",
        )
        selection_mode = _text(row.get("selection_mode"), "selection mode")
        protected_work_policy = row.get("protected_work_policy", "strict")
        _require(protected_work_policy in {"strict", "energy-budgeted"}, "protected work policy")
        _require(
            selection_mode in {
                "adaptive-decode",
                "calibration",
                "deadline-first",
                "desktop-baseline",
                "energy-aware",
            },
            "selection mode is invalid",
        )
        attribution = _text(
            row.get("energy_attribution_kind"), "energy attribution kind"
        )
        _require(
            attribution in {"diagnostic", "isolated", "matched_abba"},
            "energy attribution kind is invalid",
        )
        adaptive = row.get("adaptive_minimum_remaining_tokens")
        maximum_phone_sessions = row.get("maximum_phone_sessions")
        fault_injection = _optional_text(
            row.get("helper_preparation_fault_injection"),
            "helper preparation fault injection",
        )
        _require(
            fault_injection in {None, "post-load-once"},
            "helper preparation fault injection is invalid",
        )
        reference_values = (
            row.get("reference_catalog_path"),
            row.get("reference_replay_schedule_path"),
            row.get("reference_runner_command_path"),
        )
        memory_caps = tuple(PhoneHtpMemoryCapConfiguration.from_json(value)
                            for value in _sequence(row.get("phone_htp_memory_caps", []), "phone HTP memory caps"))
        _require(len({cap.phone_device_id for cap in memory_caps}) == len(memory_caps),
                 "phone HTP memory cap devices are not unique")
        _require(not memory_caps or row.get("fixed_phone_residency") is None,
                 "fixed phone residency cannot change its memory cap")
        _require(
            all(value is None for value in reference_values)
            or all(value is not None for value in reference_values),
            "resolved-contract references must be supplied together",
        )
        return cls(
            campaign_id=_text(row.get("campaign_id"), "campaign id"),
            rig_manifest_path=_path(row.get("rig_manifest_path"), base, "rig manifest"),
            models_manifest_path=_path(row.get("models_manifest_path"), base, "models manifest"),
            evidence_manifest_path=_path(row.get("evidence_manifest_path"), base, "evidence manifest"),
            trace=TraceConfiguration.from_json(row.get("trace"), base),
            selection_mode=selection_mode,
            protected_work_policy=protected_work_policy,
            maximum_latency_ppm=_integer(
                row.get("maximum_latency_ppm"),
                "maximum latency ppm",
                minimum=1_000_000,
            ),
            minimum_energy_saving_ppm=_integer(
                row.get("minimum_energy_saving_ppm", 10_000),
                "minimum energy saving ppm",
                maximum=999_999,
            ),
            energy_attribution_kind=attribution,
            max_workers=_integer(row.get("max_workers", 32), "maximum workers", minimum=1),
            adaptive_minimum_remaining_tokens=(
                None
                if adaptive is None
                else _integer(adaptive, "adaptive minimum remaining tokens", minimum=1)
            ),
            maximum_phone_sessions=(
                None
                if maximum_phone_sessions is None
                else _integer(
                    maximum_phone_sessions,
                    "maximum phone sessions",
                    minimum=1,
                )
            ),
            reference_catalog_path=_optional_path(
                row.get("reference_catalog_path"),
                base,
                "reference catalog",
            ),
            reference_replay_schedule_path=_optional_path(
                row.get("reference_replay_schedule_path"),
                base,
                "reference replay schedule",
            ),
            reference_runner_command_path=_optional_path(
                row.get("reference_runner_command_path"),
                base,
                "reference runner command",
            ),
            helper_preparation_fault_injection=fault_injection,
            fixed_phone_residency=(
                None if row.get("fixed_phone_residency") is None else
                FixedPhoneResidencyConfiguration.from_json(row["fixed_phone_residency"])
            ),
            include_startup_preparation=_boolean(
                row.get("include_startup_preparation", False), "include startup preparation",
            ),
            phone_htp_memory_caps=tuple(sorted(memory_caps, key=lambda cap: cap.phone_device_id)),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "adaptive_minimum_remaining_tokens": self.adaptive_minimum_remaining_tokens,
            **({} if self.fixed_phone_residency is None else {
                "fixed_phone_residency": self.fixed_phone_residency.to_json(),
            }),
            **({"include_startup_preparation": True} if self.include_startup_preparation else {}),
            **({"phone_htp_memory_caps": [cap.to_json() for cap in self.phone_htp_memory_caps]}
               if self.phone_htp_memory_caps else {}),
            "campaign_id": self.campaign_id,
            "energy_attribution_kind": self.energy_attribution_kind,
            "evidence_manifest_path": str(self.evidence_manifest_path),
            "helper_preparation_fault_injection": (
                self.helper_preparation_fault_injection
            ),
            "max_workers": self.max_workers,
            "maximum_latency_ppm": self.maximum_latency_ppm,
            "maximum_phone_sessions": self.maximum_phone_sessions,
            "minimum_energy_saving_ppm": self.minimum_energy_saving_ppm,
            "models_manifest_path": str(self.models_manifest_path),
            "reference_catalog_path": _path_json(
                self.reference_catalog_path
            ),
            "reference_replay_schedule_path": _path_json(
                self.reference_replay_schedule_path
            ),
            "reference_runner_command_path": _path_json(
                self.reference_runner_command_path
            ),
            "rig_manifest_path": str(self.rig_manifest_path),
            "schema": CAMPAIGN_MANIFEST_SCHEMA,
            "selection_mode": self.selection_mode,
            "protected_work_policy": self.protected_work_policy,
            "trace": self.trace.to_json(),
        }
