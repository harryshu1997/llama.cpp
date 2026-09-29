"""Device, executor, catalog and snapshot capability contracts: phone."""

from __future__ import annotations

from dataclasses import dataclass

from .common import (
    PHONE_POWER_ESTIMATION_VERSION,
    PHONE_POWER_EVIDENCE_ASSUMED_4P5W,
    RESIDENCY_STATES,
    RuntimeCapabilityError,
    _boolean,
    _integer,
    _list,
    _object,
    _text,
    _texts,
)


@dataclass(frozen=True)
class RuntimePhonePowerProfile:
    """Operational whole-phone power used when charging hides discharge."""

    device_id: str
    domain_id: str
    active_power_mw: int
    idle_power_mw: int
    evidence_kind: str
    estimation_version: str
    allow_assumed_for_scheduling: bool
    minimum_battery_ppm: int

    def __post_init__(self) -> None:
        _text("phone power device", self.device_id)
        _text("phone power domain", self.domain_id)
        _integer("phone active power", self.active_power_mw, 1)
        _integer("phone idle power", self.idle_power_mw, 1)
        if self.active_power_mw < self.idle_power_mw:
            raise RuntimeCapabilityError(
                "phone active power is below idle power"
            )
        if self.evidence_kind != PHONE_POWER_EVIDENCE_ASSUMED_4P5W:
            raise RuntimeCapabilityError(
                "phone power evidence kind is unsupported"
            )
        if (
            self.active_power_mw != 4_500
            or self.idle_power_mw != 875
        ):
            raise RuntimeCapabilityError(
                "ASSUMED_4P5W requires the declared power model"
            )
        _text("phone power estimation version", self.estimation_version)
        _boolean(
            "phone assumed power scheduling policy",
            self.allow_assumed_for_scheduling,
        )
        battery = _integer(
            "phone minimum battery", self.minimum_battery_ppm
        )
        if battery > 1_000_000:
            raise RuntimeCapabilityError(
                "phone minimum battery exceeds one"
            )

    @classmethod
    def assumed_4p5w(
        cls,
        *,
        device_id: str,
        domain_id: str,
        allow_assumed_for_scheduling: bool = False,
        minimum_battery_ppm: int = 50_000,
    ) -> "RuntimePhonePowerProfile":
        return cls(
            device_id=device_id,
            domain_id=domain_id,
            active_power_mw=4_500,
            idle_power_mw=875,
            evidence_kind=PHONE_POWER_EVIDENCE_ASSUMED_4P5W,
            estimation_version=PHONE_POWER_ESTIMATION_VERSION,
            allow_assumed_for_scheduling=allow_assumed_for_scheduling,
            minimum_battery_ppm=minimum_battery_ppm,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "active_power_mw": self.active_power_mw,
            "allow_assumed_for_scheduling": (
                self.allow_assumed_for_scheduling
            ),
            "device_id": self.device_id,
            "domain_id": self.domain_id,
            "evidence_kind": self.evidence_kind,
            "estimation_version": self.estimation_version,
            "idle_power_mw": self.idle_power_mw,
            "minimum_battery_ppm": self.minimum_battery_ppm,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimePhonePowerProfile":
        row = _object("runtime phone power profile", value)
        return cls(
            device_id=row.get("device_id"),
            domain_id=row.get("domain_id"),
            active_power_mw=row.get("active_power_mw"),
            idle_power_mw=row.get("idle_power_mw"),
            evidence_kind=row.get("evidence_kind"),
            estimation_version=row.get("estimation_version"),
            allow_assumed_for_scheduling=row.get(
                "allow_assumed_for_scheduling"
            ),
            minimum_battery_ppm=row.get("minimum_battery_ppm"),
        )


@dataclass(frozen=True)
class RuntimePhoneSessionCapability:
    """One phone residency arena backed by shared execution resources."""

    session_id: str
    device_id: str
    endpoint: str
    worker_identity_sha256: str
    memory_resource_id: str
    resident_memory_limit_bytes: int
    shared_compute_resource_id: str
    shared_transport_resource_ids: tuple[str, ...]
    supported_layer_mask: int
    maximum_columns: int
    column_quantum: int
    supported_data_types: tuple[str, ...]
    batch_plans: tuple[str, ...]
    ready: bool
    residency_state: str
    resident_artifact_sha256: str | None = None
    resident_geometry_sha256: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "session_id",
            "device_id",
            "endpoint",
            "memory_resource_id",
            "shared_compute_resource_id",
        ):
            _text("phone session " + name, getattr(self, name))
        worker = _text(
            "phone session worker identity", self.worker_identity_sha256
        )
        if (
            not worker.startswith("sha256:")
            or len(worker) != 71
            or any(value not in "0123456789abcdef" for value in worker[7:])
        ):
            raise RuntimeCapabilityError(
                "phone session worker identity must be SHA-256"
            )
        _integer(
            "phone session resident memory limit",
            self.resident_memory_limit_bytes,
            1,
        )
        transports = _texts(
            "phone session shared transport",
            self.shared_transport_resource_ids,
        )
        if self.shared_compute_resource_id in transports:
            raise RuntimeCapabilityError(
                "phone session compute and transport resources overlap"
            )
        layer_mask = _integer(
            "phone session supported layer mask",
            self.supported_layer_mask,
            1,
        )
        if layer_mask >= 1 << 64:
            raise RuntimeCapabilityError(
                "phone session layer mask exceeds 64 layers"
            )
        columns = _integer(
            "phone session maximum columns", self.maximum_columns, 1
        )
        quantum = _integer(
            "phone session column quantum", self.column_quantum, 1
        )
        if columns % quantum:
            raise RuntimeCapabilityError(
                "phone session maximum columns are not quantized"
            )
        data_types = _texts(
            "phone session data type", self.supported_data_types
        )
        batch_plans = _texts("phone session batch plan", self.batch_plans)
        if set(batch_plans) - {
            "coalesced-batch", "single", "split-row"
        }:
            raise RuntimeCapabilityError(
                "phone session batch plan is unsupported"
            )
        _boolean("phone session readiness", self.ready)
        if self.unavailable_reason is not None:
            _text(
                "phone session unavailable reason",
                self.unavailable_reason,
            )
            if self.ready:
                raise RuntimeCapabilityError(
                    "ready phone session carries an unavailable reason"
                )
        if self.residency_state not in RESIDENCY_STATES:
            raise RuntimeCapabilityError(
                "phone session residency state is invalid"
            )
        resident_values = (
            self.resident_artifact_sha256,
            self.resident_geometry_sha256,
        )
        if self.residency_state == "cold":
            if any(value is not None for value in resident_values):
                raise RuntimeCapabilityError(
                    "cold phone session carries residency identity"
                )
        elif any(value is None for value in resident_values):
            raise RuntimeCapabilityError(
                "resident phone session lacks residency identity"
            )
        for value in resident_values:
            if value is None:
                continue
            digest = _text("phone session residency hash", value)
            if (
                not digest.startswith("sha256:")
                or len(digest) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in digest[7:]
                )
            ):
                raise RuntimeCapabilityError(
                    "phone session residency identity must be SHA-256"
                )
        object.__setattr__(self, "shared_transport_resource_ids", transports)
        object.__setattr__(self, "supported_data_types", data_types)
        object.__setattr__(self, "batch_plans", batch_plans)

    def supports(
        self,
        *,
        layer_index: int,
        columns: int,
        data_type: str,
        batch_plan: str,
    ) -> bool:
        return (
            self.ready
            and 0 <= layer_index < 64
            and bool(self.supported_layer_mask & (1 << layer_index))
            and 0 < columns <= self.maximum_columns
            and columns % self.column_quantum == 0
            and ("*" in self.supported_data_types
                 or data_type in self.supported_data_types)
            and batch_plan in self.batch_plans
        )

    def to_json(self) -> dict[str, object]:
        result = {
            "batch_plans": list(self.batch_plans),
            "column_quantum": self.column_quantum,
            "device_id": self.device_id,
            "endpoint": self.endpoint,
            "maximum_columns": self.maximum_columns,
            "memory_resource_id": self.memory_resource_id,
            "ready": self.ready,
            "residency_state": self.residency_state,
            "resident_memory_limit_bytes": self.resident_memory_limit_bytes,
            "session_id": self.session_id,
            "shared_compute_resource_id": self.shared_compute_resource_id,
            "shared_transport_resource_ids": list(
                self.shared_transport_resource_ids
            ),
            "supported_data_types": list(self.supported_data_types),
            "supported_layer_mask": self.supported_layer_mask,
            "worker_identity_sha256": self.worker_identity_sha256,
        }
        if self.resident_artifact_sha256 is not None:
            result["resident_artifact_sha256"] = (
                self.resident_artifact_sha256
            )
            result["resident_geometry_sha256"] = (
                self.resident_geometry_sha256
            )
        if self.unavailable_reason is not None:
            result["unavailable_reason"] = self.unavailable_reason
        return result

    @classmethod
    def from_json(cls, value: object) -> "RuntimePhoneSessionCapability":
        row = _object("runtime phone session capability", value)
        return cls(
            session_id=row.get("session_id"),
            device_id=row.get("device_id"),
            endpoint=row.get("endpoint"),
            worker_identity_sha256=row.get("worker_identity_sha256"),
            memory_resource_id=row.get("memory_resource_id"),
            resident_memory_limit_bytes=row.get(
                "resident_memory_limit_bytes"
            ),
            shared_compute_resource_id=row.get(
                "shared_compute_resource_id"
            ),
            shared_transport_resource_ids=tuple(_list(
                "phone session shared transports",
                row.get("shared_transport_resource_ids"),
            )),
            supported_layer_mask=row.get("supported_layer_mask"),
            maximum_columns=row.get("maximum_columns"),
            column_quantum=row.get("column_quantum"),
            supported_data_types=tuple(_list(
                "phone session supported data types",
                row.get("supported_data_types"),
            )),
            batch_plans=tuple(_list(
                "phone session batch plans", row.get("batch_plans")
            )),
            ready=row.get("ready"),
            residency_state=row.get("residency_state"),
            resident_artifact_sha256=row.get("resident_artifact_sha256"),
            resident_geometry_sha256=row.get("resident_geometry_sha256"),
            unavailable_reason=row.get("unavailable_reason"),
        )
