"""Typed scheduler configuration manifests: evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .common import (
    EVIDENCE_MANIFEST_SCHEMA,
    _boolean,
    _integer,
    _object,
    _optional_path,
    _path,
    _path_json,
    _require,
    _sequence,
    _text,
)


@dataclass(frozen=True)
class PhonePowerEvidenceConfiguration:
    evidence_kind: str
    active_power_mw: int
    idle_power_mw: int
    allow_assumed_for_scheduling: bool
    minimum_battery_ppm: int

    @classmethod
    def from_json(cls, value: object) -> "PhonePowerEvidenceConfiguration":
        row = _object(value, "phone power evidence")
        return cls(
            evidence_kind=_text(row.get("evidence_kind"), "phone power evidence kind"),
            active_power_mw=_integer(row.get("active_power_mw"), "active phone power", minimum=1),
            idle_power_mw=_integer(row.get("idle_power_mw"), "idle phone power", minimum=1),
            allow_assumed_for_scheduling=_boolean(
                row.get("allow_assumed_for_scheduling"),
                "allow assumed phone power",
            ),
            minimum_battery_ppm=_integer(
                row.get("minimum_battery_ppm"),
                "minimum phone battery",
                maximum=1_000_000,
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "active_power_mw": self.active_power_mw,
            "allow_assumed_for_scheduling": self.allow_assumed_for_scheduling,
            "evidence_kind": self.evidence_kind,
            "idle_power_mw": self.idle_power_mw,
            "minimum_battery_ppm": self.minimum_battery_ppm,
        }


@dataclass(frozen=True)
class EvidenceManifest:
    kernel_profile_path: Path
    calibration_directory: Path
    calibration_runs: Mapping[str, tuple[str, ...]]
    desktop_calibration_arm: str
    assisted_calibration_arm: str
    overlay_catalog_path: Path | None
    prematerialized_catalog_path: Path | None
    prematerialized_source_manifest_path: Path | None
    phone_session_discovery_path: Path
    transport_qualification_identity_path: Path
    transport_qualification_directories: tuple[Path, ...]
    observation_store_path: Path
    observation_source_catalog_path: Path | None
    adaptive_observation_store_path: Path
    adaptive_observation_source_catalog_path: Path | None
    phone_power: PhonePowerEvidenceConfiguration
    helper_phone_evidence_paths: Mapping[str, Path] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: object, base: Path) -> "EvidenceManifest":
        row = _object(value, "evidence manifest")
        _require(
            row.get("schema") == EVIDENCE_MANIFEST_SCHEMA,
            "evidence manifest schema",
        )
        directories = tuple(
            _path(item, base, "transport qualification directory")
            for item in _sequence(
                row.get("transport_qualification_directories"),
                "transport qualification directories",
            )
        )
        _require(directories, "transport qualification directories are empty")
        raw_runs = _object(row.get("calibration_runs"), "calibration runs")
        calibration_runs = {
            _text(key, "calibration arm"): tuple(
                _text(item, "calibration run")
                for item in _sequence(value, "calibration run list")
            )
            for key, value in raw_runs.items()
        }
        desktop_arm = _text(
            row.get("desktop_calibration_arm"), "desktop calibration arm"
        )
        assisted_arm = _text(
            row.get("assisted_calibration_arm"), "assisted calibration arm"
        )
        _require(
            desktop_arm != assisted_arm
            and desktop_arm in calibration_runs
            and assisted_arm in calibration_runs
            and all(calibration_runs.values()),
            "calibration arm configuration",
        )
        return cls(
            kernel_profile_path=_path(row.get("kernel_profile_path"), base, "kernel profile"),
            calibration_directory=_path(row.get("calibration_directory"), base, "calibration directory"),
            calibration_runs=MappingProxyType(dict(sorted(
                calibration_runs.items()
            ))),
            desktop_calibration_arm=desktop_arm,
            assisted_calibration_arm=assisted_arm,
            overlay_catalog_path=_optional_path(
                row.get("overlay_catalog_path"), base, "overlay catalog"
            ),
            prematerialized_catalog_path=_optional_path(
                row.get("prematerialized_catalog_path"), base, "prematerialized catalog"
            ),
            prematerialized_source_manifest_path=_optional_path(
                row.get("prematerialized_source_manifest_path"),
                base,
                "prematerialized source manifest",
            ),
            phone_session_discovery_path=_path(
                row.get("phone_session_discovery_path"), base, "phone session discovery"
            ),
            transport_qualification_identity_path=_path(
                row.get("transport_qualification_identity_path"),
                base,
                "transport qualification identity",
            ),
            transport_qualification_directories=directories,
            observation_store_path=_path(
                row.get("observation_store_path"), base, "observation store"
            ),
            observation_source_catalog_path=_optional_path(
                row.get("observation_source_catalog_path"),
                base,
                "observation source catalog",
            ),
            adaptive_observation_store_path=_path(
                row.get("adaptive_observation_store_path"),
                base,
                "adaptive observation store",
            ),
            adaptive_observation_source_catalog_path=_optional_path(
                row.get("adaptive_observation_source_catalog_path"),
                base,
                "adaptive observation source catalog",
            ),
            phone_power=PhonePowerEvidenceConfiguration.from_json(
                row.get("phone_power")
            ),
            helper_phone_evidence_paths=MappingProxyType({
                _text(key, "helper phone evidence device"): _path(value, base, "helper phone evidence")
                for key, value in _object(row.get("helper_phone_evidence_paths", {}),
                                          "helper phone evidence").items()
            }),
        )

    def to_json(self) -> dict[str, object]:
        return {
            **({"helper_phone_evidence_paths": {key: str(value)
                for key, value in self.helper_phone_evidence_paths.items()}}
               if self.helper_phone_evidence_paths else {}),
            "adaptive_observation_source_catalog_path": _path_json(
                self.adaptive_observation_source_catalog_path
            ),
            "adaptive_observation_store_path": str(
                self.adaptive_observation_store_path
            ),
            "calibration_directory": str(self.calibration_directory),
            "calibration_runs": {
                key: list(value)
                for key, value in self.calibration_runs.items()
            },
            "desktop_calibration_arm": self.desktop_calibration_arm,
            "assisted_calibration_arm": self.assisted_calibration_arm,
            "kernel_profile_path": str(self.kernel_profile_path),
            "observation_source_catalog_path": _path_json(
                self.observation_source_catalog_path
            ),
            "observation_store_path": str(self.observation_store_path),
            "overlay_catalog_path": _path_json(self.overlay_catalog_path),
            "phone_power": self.phone_power.to_json(),
            "phone_session_discovery_path": str(
                self.phone_session_discovery_path
            ),
            "prematerialized_catalog_path": _path_json(
                self.prematerialized_catalog_path
            ),
            "prematerialized_source_manifest_path": _path_json(
                self.prematerialized_source_manifest_path
            ),
            "schema": EVIDENCE_MANIFEST_SCHEMA,
            "transport_qualification_directories": [
                str(path) for path in self.transport_qualification_directories
            ],
            "transport_qualification_identity_path": str(
                self.transport_qualification_identity_path
            ),
        }
