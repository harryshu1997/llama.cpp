"""Phone FFN configuration, physical events and receipts: events."""

from __future__ import annotations

from dataclasses import dataclass
import json

from ..contracts import PhysicalAdapterError
from .common import (
    _MULTI_TERMINAL_PREFIX,
    _RESIDENCY_CALL_PREFIX,
    _RESIDENCY_PHASE_PREFIX,
    _TERMINAL_PATTERN,
)


@dataclass(frozen=True)
class DirectPhoneFfnSessionProof:
    session_id: str
    endpoint_sha256: str
    resident_geometry_sha256: str
    operator_plan_sha256: str
    layer_mask: int
    calls: int
    rows: int
    h2d_bytes: int
    d2h_bytes: int
    h2d_us: int
    d2h_us: int
    compute_us: int
    rpc_us: int
    h2d_active_us: int | None = None
    idle_receive_wait_us: int | None = None
    h2d_timing_scope: str = "legacy-idle-inclusive"
    artifact_sha256: str | None = None
    session_generation: int | None = None

    def __post_init__(self) -> None:
        hashes = (
            self.endpoint_sha256,
            self.resident_geometry_sha256,
            self.operator_plan_sha256,
        )
        if (
            type(self.session_id) is not str
            or not self.session_id
            or not self.session_id.isascii()
            or (
                self.artifact_sha256 is not None
                and (
                    not self.artifact_sha256.startswith("sha256:")
                    or len(self.artifact_sha256) != 71
                )
            )
            or any(
                type(value) is not str
                or not value.startswith("sha256:")
                or len(value) != 71
                or any(
                    character not in "0123456789abcdef"
                    for character in value[7:]
                )
                for value in hashes
            )
            or type(self.layer_mask) is not int
            or not 0 < self.layer_mask < 1 << 64
            or (
                self.session_generation is not None
                and (
                    type(self.session_generation) is not int
                    or self.session_generation < 1
                )
            )
            or any(
                type(value) is not int or value < 0
                for value in (
                    self.calls,
                    self.rows,
                    self.h2d_bytes,
                    self.d2h_bytes,
                    self.h2d_us,
                    self.d2h_us,
                    self.compute_us,
                    self.rpc_us,
                )
            )
            or self.h2d_timing_scope not in {
                "legacy-idle-inclusive",
                "payload-ready-read-v1",
                "host-completion-required-v2",
            }
            or (
                self.h2d_timing_scope == "legacy-idle-inclusive"
                and (
                    self.h2d_active_us is not None
                    or self.idle_receive_wait_us is not None
                )
            )
            or (
                self.h2d_timing_scope == "payload-ready-read-v1"
                and (
                    type(self.h2d_active_us) is not int
                    or self.h2d_active_us < 0
                    or type(self.idle_receive_wait_us) is not int
                    or self.idle_receive_wait_us < 0
                    or self.h2d_us != self.h2d_active_us
                )
            )
            or (
                self.h2d_timing_scope == "host-completion-required-v2"
                and (
                    self.h2d_us != 0
                    or self.h2d_active_us != 0
                    or type(self.idle_receive_wait_us) is not int
                    or self.idle_receive_wait_us < 0
                )
            )
        ):
            raise PhysicalAdapterError(
                "direct phone session proof is invalid"
            )

    def to_json(self) -> dict[str, object]:
        result = {
            "calls": self.calls,
            "compute_us": self.compute_us,
            "d2h_bytes": self.d2h_bytes,
            "d2h_us": self.d2h_us,
            "endpoint_sha256": self.endpoint_sha256,
            "h2d_bytes": self.h2d_bytes,
            "h2d_us": self.h2d_us,
            "layer_mask": self.layer_mask,
            "operator_plan_sha256": self.operator_plan_sha256,
            "resident_geometry_sha256": (
                self.resident_geometry_sha256
            ),
            "rows": self.rows,
            "rpc_us": self.rpc_us,
            "session_id": self.session_id,
        }
        if self.artifact_sha256 is not None:
            result["artifact_sha256"] = self.artifact_sha256
        if self.session_generation is not None:
            result["session_generation"] = self.session_generation
        if self.h2d_timing_scope != "legacy-idle-inclusive":
            result.update({
                "h2d_active_us": self.h2d_active_us,
                "h2d_timing_scope": self.h2d_timing_scope,
                "idle_receive_wait_us": self.idle_receive_wait_us,
            })
        return result

    @property
    def h2d_estimator_eligible(self) -> bool:
        return False


@dataclass(frozen=True)
class DirectPhoneFfnTerminalReceipt:
    transport: str
    requests: int
    reset_recoveries: int
    status: int
    queue_depth: int = 1
    maximum_pending_outputs: int = 0
    phone_payload_copies: int = 0
    d2h_completions: int = 0
    d2h_queue_us: int = 0
    d2h_queue_max_us: int = 0
    session_proofs: tuple[DirectPhoneFfnSessionProof, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.transport not in {"direct", "session-router"}
            or type(self.requests) is not int
            or self.requests < 0
            or type(self.reset_recoveries) is not int
            or self.reset_recoveries < 0
            or self.status != 0
            or type(self.queue_depth) is not int
            or self.queue_depth <= 0
            or type(self.maximum_pending_outputs) is not int
            or self.maximum_pending_outputs < 0
            or self.maximum_pending_outputs > self.queue_depth
            or type(self.phone_payload_copies) is not int
            or self.phone_payload_copies < 0
            or type(self.d2h_completions) is not int
            or self.d2h_completions < 0
            or type(self.d2h_queue_us) is not int
            or self.d2h_queue_us < 0
            or type(self.d2h_queue_max_us) is not int
            or self.d2h_queue_max_us < 0
            or self.d2h_queue_max_us > self.d2h_queue_us
            or any(
                not isinstance(row, DirectPhoneFfnSessionProof)
                for row in self.session_proofs
            )
            or (
                self.transport == "direct" and self.session_proofs
            )
            or (
                self.transport == "session-router"
                and (
                    not self.session_proofs
                    or len({
                        (
                            row.session_id,
                            row.artifact_sha256,
                            row.resident_geometry_sha256,
                            row.session_generation,
                        )
                        for row in self.session_proofs
                    }) != len(self.session_proofs)
                    or sum(
                        row.calls for row in self.session_proofs
                    ) != self.requests
                )
            )
        ):
            raise PhysicalAdapterError(
                "direct phone FFN terminal receipt is invalid"
            )

    def to_json(self) -> dict[str, object]:
        result = {
            "requests": self.requests,
            "queue_depth": self.queue_depth,
            "maximum_pending_outputs": self.maximum_pending_outputs,
            "phone_payload_copies": self.phone_payload_copies,
            "d2h_completions": self.d2h_completions,
            "d2h_queue_us": self.d2h_queue_us,
            "d2h_queue_max_us": self.d2h_queue_max_us,
            "reset_recoveries": self.reset_recoveries,
            "status": self.status,
            "transport": self.transport,
        }
        if self.session_proofs:
            result["session_proofs"] = [
                row.to_json() for row in self.session_proofs
            ]
        return result


def parse_direct_phone_ffn_terminal(
    lines: tuple[str, ...] | list[str],
) -> DirectPhoneFfnTerminalReceipt:
    matches = []
    multi_rows = []
    for raw in lines:
        stripped = raw.strip()
        match = _TERMINAL_PATTERN.fullmatch(stripped)
        if match is not None:
            matches.append(match.groups())
        if stripped.startswith(_MULTI_TERMINAL_PREFIX):
            try:
                multi_rows.append(json.loads(
                    stripped.removeprefix(_MULTI_TERMINAL_PREFIX)
                ))
            except json.JSONDecodeError as error:
                raise PhysicalAdapterError(
                    "multi-session phone terminal JSON is invalid"
                ) from error
    if len(matches) + len(multi_rows) != 1:
        raise PhysicalAdapterError(
            "direct phone FFN terminal receipt is not unique"
        )
    if multi_rows:
        value = multi_rows[0]
        proofs = value.get("shards") if type(value) is dict else None
        if type(proofs) is not list:
            raise PhysicalAdapterError(
                "multi-session phone terminal receipt is invalid"
            )
        try:
            return DirectPhoneFfnTerminalReceipt(
                transport=value.get("transport"),
                requests=value.get("requests"),
                reset_recoveries=value.get("recoveries"),
                status=value.get("status"),
                queue_depth=value.get("queue_depth"),
                maximum_pending_outputs=value.get(
                    "maximum_pending_outputs"
                ),
                phone_payload_copies=value.get("phone_payload_copies"),
                d2h_completions=value.get("d2h_completions"),
                d2h_queue_us=value.get("d2h_queue_us"),
                d2h_queue_max_us=value.get("d2h_queue_max_us"),
                session_proofs=tuple(
                    DirectPhoneFfnSessionProof(**row) for row in proofs
                ),
            )
        except (TypeError, ValueError) as error:
            raise PhysicalAdapterError(
                "multi-session phone terminal receipt is invalid"
            ) from error
    (
        transport,
        requests,
        queue_depth,
        pending_outputs,
        phone_payload_copies,
        d2h_completions,
        d2h_queue_us,
        d2h_queue_max_us,
        recoveries,
        status,
    ) = matches[0]
    return DirectPhoneFfnTerminalReceipt(
        transport=transport,
        requests=int(requests),
        reset_recoveries=int(recoveries),
        status=int(status),
        queue_depth=int(queue_depth or 1),
        maximum_pending_outputs=int(pending_outputs or 0),
        phone_payload_copies=int(phone_payload_copies or 0),
        d2h_completions=int(d2h_completions or 0),
        d2h_queue_us=int(d2h_queue_us or 0),
        d2h_queue_max_us=int(d2h_queue_max_us or 0),
    )


@dataclass(frozen=True)
class PhoneResidencyPhaseEvent:
    component: str
    phase: str
    session_id: str
    artifact_sha256: str
    session_generation: int
    monotonic_us: int
    epoch_us: int

    def __post_init__(self) -> None:
        values = (self.component, self.phase, self.session_id)
        if (
            any(
                type(value) is not str
                or not value
                or not value.isascii()
                for value in values
            )
            or not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.artifact_sha256[7:]
            )
            or type(self.session_generation) is not int
            or self.session_generation < 1
            or type(self.monotonic_us) is not int
            or self.monotonic_us < 0
            or type(self.epoch_us) is not int
            or self.epoch_us < 0
        ):
            raise PhysicalAdapterError(
                "phone residency phase event is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "component": self.component,
            "epoch_us": self.epoch_us,
            "monotonic_us": self.monotonic_us,
            "phase": self.phase,
            "schema": "s42-phone-residency-phase-v1",
            "session_generation": self.session_generation,
            "session_id": self.session_id,
        }


def parse_phone_residency_phase_events(
    lines: list[str] | tuple[str, ...],
) -> tuple[PhoneResidencyPhaseEvent, ...]:
    result = []
    for line in lines:
        if not line.startswith(_RESIDENCY_PHASE_PREFIX):
            continue
        try:
            body = json.loads(line[len(_RESIDENCY_PHASE_PREFIX):])
            if (
                not isinstance(body, dict)
                or body.get("schema")
                    != "s42-phone-residency-phase-v1"
            ):
                raise ValueError("schema")
            result.append(PhoneResidencyPhaseEvent(
                component=body.get("component"),
                phase=body.get("phase"),
                session_id=body.get("session_id"),
                artifact_sha256=body.get("artifact_sha256"),
                session_generation=body.get("session_generation"),
                monotonic_us=body.get("monotonic_us"),
                epoch_us=body.get("epoch_us"),
            ))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise PhysicalAdapterError(
                "phone residency phase event is malformed"
            ) from error
    return tuple(result)


@dataclass(frozen=True)
class PhoneResidencyCallEvent:
    session_id: str
    artifact_sha256: str
    session_generation: int
    calls: int
    monotonic_us: int
    epoch_us: int

    def __post_init__(self) -> None:
        if (
            type(self.session_id) is not str
            or not self.session_id
            or not self.session_id.isascii()
            or not self.artifact_sha256.startswith("sha256:")
            or len(self.artifact_sha256) != 71
            or any(
                value not in "0123456789abcdef"
                for value in self.artifact_sha256[7:]
            )
            or type(self.session_generation) is not int
            or self.session_generation < 1
            or type(self.calls) is not int
            or self.calls < 1
            or type(self.monotonic_us) is not int
            or self.monotonic_us < 0
            or type(self.epoch_us) is not int
            or self.epoch_us < 0
        ):
            raise PhysicalAdapterError(
                "phone residency call event is invalid"
            )

    def to_json(self) -> dict[str, object]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "calls": self.calls,
            "epoch_us": self.epoch_us,
            "monotonic_us": self.monotonic_us,
            "schema": "s42-phone-residency-call-v1",
            "session_generation": self.session_generation,
            "session_id": self.session_id,
        }


def parse_phone_residency_call_events(
    lines: list[str] | tuple[str, ...],
) -> tuple[PhoneResidencyCallEvent, ...]:
    result = []
    for line in lines:
        if not line.startswith(_RESIDENCY_CALL_PREFIX):
            continue
        try:
            body = json.loads(line[len(_RESIDENCY_CALL_PREFIX):])
            if (
                not isinstance(body, dict)
                or body.get("schema")
                    != "s42-phone-residency-call-v1"
            ):
                raise ValueError("schema")
            result.append(PhoneResidencyCallEvent(
                session_id=body.get("session_id"),
                artifact_sha256=body.get("artifact_sha256"),
                session_generation=body.get("session_generation"),
                calls=body.get("calls"),
                monotonic_us=body.get("monotonic_us"),
                epoch_us=body.get("epoch_us"),
            ))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise PhysicalAdapterError(
                "phone residency call event is malformed"
            ) from error
    return tuple(result)
