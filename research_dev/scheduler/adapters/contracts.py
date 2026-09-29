"""Physical adapter contracts with no placement policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from .._internal.plan_contracts.co_helpers import phone_helper_layer_masks
from .._internal.plan_contracts.common import RuntimePlanError
from .._internal.runtime_execution import (
    HELPER_LOST_PHASE,
    SERVER_EXITED_PHASE,
)


class PhysicalAdapterError(RuntimeError):
    pass


class StalePhysicalSlotError(PhysicalAdapterError):
    """A request-scoped directive refers to a released server slot."""


DORMANT_PHONE_FFN_RUNTIME_PARAMETER = "dormant_phone_ffn_runtime_v1"
DORMANT_PHONE_FFN_RUNTIME_KEYS = frozenset({
    "bridge_allocator",
    "bridge_queue_depth",
    "ffn_activation",
    "ffn_assistance_phase",
    "ffn_bridge_host",
    "ffn_bridge_port",
    "ffn_host_share_release",
    "ffn_host_share_drop_cache",
    "ffn_host_share_populate",
    "scheduler_trace_path",
    "ffn_max_tokens",
    "ffn_n_embd",
    "ffn_resident_columns",
    "ffn_resident_layer_mask",
    "ffn_runtime_control_protocol",
    "ffn_timeout_ms",
    "ffn_transport",
    "phone_device_id",
    "phone_helpers",
    "usb_allocator",
    "usb_batch_plan",
    "usb_concurrent_streams",
    "usb_full_duplex",
    "usb_max_payload_bytes",
    "usb_product_id",
    "usb_queue_depth",
    "usb_slot_safety_bytes",
    "usb_split_h2d",
    "usb_transport_generation",
    "usb_transport_profile_id",
    "usb_transport_qualification_identity_sha256",
    "usb_vendor_id",
    "usbfs_available_bytes",
})
_DORMANT_PHONE_FFN_REQUIRED_KEYS = frozenset({
    "ffn_activation",
    "ffn_assistance_phase",
    "ffn_max_tokens",
    "ffn_n_embd",
    "ffn_resident_columns",
    "ffn_resident_layer_mask",
    "ffn_runtime_control_protocol",
    "ffn_timeout_ms",
    "ffn_transport",
    "phone_device_id",
})
_DORMANT_PHONE_FFN_FUNCTIONFS_KEYS = frozenset({
    "usb_allocator",
    "usb_batch_plan",
    "usb_full_duplex",
    "usb_max_payload_bytes",
    "usb_product_id",
    "usb_queue_depth",
    "usb_slot_safety_bytes",
    "usb_split_h2d",
    "usb_transport_generation",
    "usb_transport_profile_id",
    "usb_vendor_id",
    "usbfs_available_bytes",
})
_DORMANT_PHONE_FFN_TCP_KEYS = frozenset({
    "bridge_allocator",
    "bridge_queue_depth",
    "ffn_bridge_host",
    "ffn_bridge_port",
})


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise PhysicalAdapterError(f"{name} must be non-empty ASCII text")
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhysicalAdapterError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    value = _text(name, value)
    if value.startswith("sha256:"):
        digest = value[7:]
    else:
        digest = value
        value = "sha256:" + value
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise PhysicalAdapterError(f"{name} must be SHA-256")
    return value


def _integer_mapping(name: str, values: Mapping[str, int]) -> Mapping[str, int]:
    result = {
        _text(f"{name} key", key): _integer(f"{name} value", value)
        for key, value in values.items()
    }
    return MappingProxyType(dict(sorted(result.items())))


def dormant_phone_ffn_parameters(
    parameters: Mapping[str, int | str],
) -> Mapping[str, int | str] | None:
    """Decode and validate a dormant request-scoped FFN contract."""

    raw = parameters.get(DORMANT_PHONE_FFN_RUNTIME_PARAMETER)
    if raw is None:
        return None
    if type(raw) is not str or not raw or not raw.isascii():
        raise PhysicalAdapterError(
            "dormant phone FFN runtime contract is invalid"
        )
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise PhysicalAdapterError(
            "dormant phone FFN runtime contract is invalid"
        ) from exc
    if type(decoded) is not dict or any(
        type(name) is not str
        or not name
        or not name.isascii()
        or name not in DORMANT_PHONE_FFN_RUNTIME_KEYS
        or type(value) not in {int, str}
        or (type(value) is int and value < 0)
        or (type(value) is str and (not value or not value.isascii()))
        for name, value in decoded.items()
    ):
        raise PhysicalAdapterError(
            "dormant phone FFN runtime contract is invalid"
        )
    required = set(_DORMANT_PHONE_FFN_REQUIRED_KEYS)
    transport = decoded.get("ffn_transport")
    if transport == "functionfs-usb":
        required.update(_DORMANT_PHONE_FFN_FUNCTIONFS_KEYS)
    elif transport == "tcp":
        required.update(_DORMANT_PHONE_FFN_TCP_KEYS)
    else:
        raise PhysicalAdapterError(
            "dormant phone FFN runtime transport is invalid"
        )
    if (
        not required.issubset(decoded)
        or decoded["ffn_assistance_phase"] != "decode"
        or decoded["ffn_runtime_control_protocol"]
            != "decode-boundary-v1"
    ):
        raise PhysicalAdapterError(
            "dormant phone FFN runtime contract is incomplete"
        )
    if "phone_helpers" in decoded:
        try:
            masks = phone_helper_layer_masks(decoded["phone_helpers"])
        except RuntimePlanError as exc:
            raise PhysicalAdapterError(
                "dormant phone FFN helpers are invalid"
            ) from exc
        if (
            next(iter(masks)) != decoded["phone_device_id"]
            or sum(masks.values()) != decoded["ffn_resident_layer_mask"]
        ):
            raise PhysicalAdapterError(
                "dormant phone FFN helpers differ from the resident layers"
            )
    return MappingProxyType(dict(sorted(decoded.items())))


@dataclass(frozen=True)
class RawEnergyMeasurement:
    """Raw measurements collected under one physical accounting boundary."""

    energy_boundary_id: str
    fleet_energy_uj_by_domain: Mapping[str, int]
    measurement_evidence_ids: tuple[str, ...]
    transfer_energy_uj_by_link: Mapping[str, int] = field(default_factory=dict)
    attribution_kind: str = "diagnostic"
    estimation_metadata: Mapping[str, int | str | bool] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _text("raw energy boundary", self.energy_boundary_id)
        fleet = _integer_mapping(
            "raw fleet energy", self.fleet_energy_uj_by_domain
        )
        if not fleet:
            raise PhysicalAdapterError("raw fleet energy is empty")
        transfer = _integer_mapping(
            "raw transfer energy", self.transfer_energy_uj_by_link
        )
        evidence = tuple(sorted(
            _text("raw energy evidence", value)
            for value in self.measurement_evidence_ids
        ))
        if not evidence or len(evidence) != len(set(evidence)):
            raise PhysicalAdapterError("raw energy evidence is invalid")
        if self.attribution_kind not in {
            "diagnostic", "isolated", "matched_abba", "device_domain"
        }:
            raise PhysicalAdapterError(
                "raw energy attribution kind is invalid"
            )
        metadata: dict[str, int | str | bool] = {}
        for raw_name, raw_value in self.estimation_metadata.items():
            name = _text("raw energy estimation metadata", raw_name)
            if type(raw_value) is bool:
                metadata[name] = raw_value
            elif type(raw_value) is int and raw_value >= 0:
                metadata[name] = raw_value
            elif type(raw_value) is str:
                metadata[name] = _text(
                    "raw energy estimation metadata value", raw_value
                )
            else:
                raise PhysicalAdapterError(
                    "raw energy estimation metadata is invalid"
                )
        object.__setattr__(self, "fleet_energy_uj_by_domain", fleet)
        object.__setattr__(self, "transfer_energy_uj_by_link", transfer)
        object.__setattr__(self, "measurement_evidence_ids", evidence)
        object.__setattr__(
            self,
            "estimation_metadata",
            MappingProxyType(dict(sorted(metadata.items()))),
        )


@dataclass(frozen=True)
class RawTransitionObservation:
    """Facts returned by a physical residency-transition callback."""

    started_us: int
    finished_us: int
    status: str
    evicted_artifact_sha256s: tuple[str, ...] = ()
    energy: RawEnergyMeasurement | None = None

    def __post_init__(self) -> None:
        _integer("raw transition start", self.started_us)
        _integer("raw transition finish", self.finished_us)
        if self.finished_us < self.started_us:
            raise PhysicalAdapterError(
                "raw transition finishes before it starts"
            )
        if self.status not in {"COMPLETED", "FAILED"}:
            raise PhysicalAdapterError("raw transition status is invalid")
        evicted = tuple(sorted(self.evicted_artifact_sha256s))
        if len(evicted) != len(set(evicted)) or any(
            not value.startswith("sha256:")
            or len(value) != 71
            or any(
                character not in "0123456789abcdef"
                for character in value[7:]
            )
            for value in evicted
        ):
            raise PhysicalAdapterError(
                "raw transition eviction evidence is invalid"
            )
        object.__setattr__(
            self, "evicted_artifact_sha256s", evicted
        )
        if self.energy is not None and not isinstance(
            self.energy, RawEnergyMeasurement
        ):
            raise PhysicalAdapterError("raw transition energy is invalid")


@dataclass(frozen=True)
class RawExecutionObservation:
    """Raw endpoint result and synchronized measurement facts."""

    started_us: int
    finished_us: int
    output_sha256: str
    payload: object = None
    energy: RawEnergyMeasurement | None = None
    energy_scope: str = "route_total"

    def __post_init__(self) -> None:
        _integer("raw execution start", self.started_us)
        _integer("raw execution finish", self.finished_us)
        if self.finished_us < self.started_us:
            raise PhysicalAdapterError(
                "raw execution finishes before it starts"
            )
        object.__setattr__(
            self,
            "output_sha256",
            _sha256("raw execution output", self.output_sha256),
        )
        if self.energy is not None and not isinstance(
            self.energy, RawEnergyMeasurement
        ):
            raise PhysicalAdapterError("raw execution energy is invalid")
        if self.energy_scope not in {"route_total", "warm_execution"}:
            raise PhysicalAdapterError(
                "raw execution energy scope is invalid"
            )


class LlamaServerExitedError(PhysicalAdapterError):
    """A managed llama-server process is gone when work is bound to it.

    The message is the historical one; only a rig running with
    ``elastic_phones.drop_recovery`` turns it into a structured
    ``server_exited`` :class:`PhysicalBackendFailure`.
    """

    def __init__(
        self,
        message: str,
        *,
        executor_id: str | None,
        returncode: int | None,
    ) -> None:
        super().__init__(message)
        if executor_id is not None:
            _text("exited llama-server executor", executor_id)
        if returncode is not None and type(returncode) is not int:
            raise PhysicalAdapterError(
                "exited llama-server return code is invalid"
            )
        self.executor_id = executor_id
        self.returncode = returncode


class PhysicalHelperLostError(PhysicalAdapterError):
    """A transport-level loss of one phone helper, raised by phone sessions."""

    def __init__(self, message: str, *, device_id: str) -> None:
        super().__init__(message)
        self.device_id = _text("lost helper device", device_id)


class CompletionStreamError(PhysicalAdapterError):
    """An invalid completion stream chunk; keeps the server's error message."""

    def __init__(
        self, message: str, *, server_error_message: str | None = None
    ) -> None:
        super().__init__(message)
        if server_error_message is not None and type(
            server_error_message
        ) is not str:
            raise PhysicalAdapterError(
                "completion stream error message is invalid"
            )
        self.server_error_message = server_error_message


@dataclass(frozen=True)
class PhysicalFailureClassification:
    """A rig's structured reading of one failed execution (drop recovery)."""

    phase: str
    failed_device_ids: tuple[str, ...] = ()
    executor_id: str | None = None
    returncode: int | None = None
    evidence: str = ""
    # lost devices readmitted after the failed attempt started: no new quarantine
    stale_device_ids: tuple[str, ...] = ()
    # helper_lost only (elastic phones S2a): the live server that masked the lost helpers out
    # and keeps serving; the failed attempt's route stays usable without them
    masked_executor_id: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in {HELPER_LOST_PHASE, SERVER_EXITED_PHASE}:
            raise PhysicalAdapterError(
                "physical failure classification phase is invalid"
            )
        devices = tuple(sorted(
            _text("classified failed device", value)
            for value in self.failed_device_ids
        ))
        if len(devices) != len(set(devices)) or (
            (self.phase == HELPER_LOST_PHASE) != bool(devices)
        ):
            raise PhysicalAdapterError(
                "physical failure classification devices are invalid"
            )
        stale = _stale_devices(self.stale_device_ids, devices)
        object.__setattr__(self, "stale_device_ids", stale)
        if self.phase == SERVER_EXITED_PHASE and self.executor_id is None:
            raise PhysicalAdapterError(
                "server_exited classification lacks its executor"
            )
        if self.executor_id is not None:
            _text("classified failed executor", self.executor_id)
        if self.returncode is not None and type(self.returncode) is not int:
            raise PhysicalAdapterError(
                "physical failure classification return code is invalid"
            )
        if type(self.evidence) is not str or not self.evidence.isascii():
            raise PhysicalAdapterError(
                "physical failure classification evidence is invalid"
            )
        _masked_executor(self.masked_executor_id, self.phase, self.executor_id)
        object.__setattr__(self, "failed_device_ids", devices)

    def to_json(self) -> dict[str, object]:
        return {
            "evidence": self.evidence,
            "executor_id": self.executor_id,
            "failed_device_ids": list(self.failed_device_ids),
            "phase": self.phase,
            "returncode": self.returncode,
            **({"stale_device_ids": list(self.stale_device_ids)}
               if self.stale_device_ids else {}),
            **({"masked_executor_id": self.masked_executor_id}
               if self.masked_executor_id is not None else {}),
        }


def _masked_executor(value: object, phase: str, exited_executor_id: str | None) -> None:
    """A masked-out server is a helper_lost fact of a server that did not exit."""
    if value is None:
        return
    _text("masked-out executor", value)
    if phase != HELPER_LOST_PHASE or exited_executor_id is not None:
        raise PhysicalAdapterError(
            "a masked-out executor requires a helper_lost failure of a live server"
        )


def _stale_devices(
    values: object, failed_device_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise PhysicalAdapterError("stale lost devices are invalid")
    stale = tuple(sorted(
        _text("stale lost device", value) for value in values
    ))
    if len(stale) != len(set(stale)) or not set(stale) <= set(
        failed_device_ids
    ):
        raise PhysicalAdapterError(
            "stale lost devices must be unique lost devices"
        )
    return stale


def elastic_helper_loss_recovery(elastic_phones: object) -> str:
    """``elastic_phones.helper_loss_recovery`` under drop recovery: ``mask_out`` keeps a live
    server that lost a helper and masks the helper out, ``retire`` (absent, the default) stops
    and reloads it; no drop recovery means ``retire`` (today's behaviour)."""
    if not elastic_drop_recovery_enabled(elastic_phones):
        return "retire"
    value = elastic_phones.get("helper_loss_recovery", "retire")
    if value not in {"mask_out", "retire"}:
        raise PhysicalAdapterError("elastic phones helper loss recovery is invalid")
    return value


def elastic_drop_recovery_enabled(elastic_phones: object) -> bool:
    """Whether an ``elastic_phones`` campaign field enables drop recovery.

    Absent (None) keeps today's behaviour; any other value must be a mapping
    (``Mapping.get``) whose truthy ``drop_recovery`` enables it.
    """

    if elastic_phones is None:
        return False
    reader = getattr(elastic_phones, "get", None)
    if not callable(reader):
        raise PhysicalAdapterError("elastic phones configuration is invalid")
    return bool(reader("drop_recovery"))


class PhysicalBackendFailure(RuntimeError):
    """Physical failure facts used by the scheduler's recovery policy."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        retry_safe: bool,
        execution_started: bool,
        started_us: int,
        finished_us: int,
        failed_resource_ids: tuple[str, ...] = (),
        failed_device_ids: tuple[str, ...] = (),
        executor_id: str | None = None,
        returncode: int | None = None,
        stale_device_ids: tuple[str, ...] = (),
        masked_executor_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.phase = _text("physical failure phase", phase)
        if type(retry_safe) is not bool or type(execution_started) is not bool:
            raise PhysicalAdapterError("physical failure flags are invalid")
        self.retry_safe = retry_safe
        self.execution_started = execution_started
        self.started_us = _integer("physical failure start", started_us)
        self.finished_us = _integer("physical failure finish", finished_us)
        if self.finished_us < self.started_us:
            raise PhysicalAdapterError(
                "physical failure finishes before it starts"
            )
        resources = tuple(sorted(
            _text("physical failed resource", value)
            for value in failed_resource_ids
        ))
        if len(resources) != len(set(resources)):
            raise PhysicalAdapterError(
                "physical failed resources are not unique"
            )
        self.failed_resource_ids = resources
        devices = tuple(sorted(
            _text("physical failed device", value)
            for value in failed_device_ids
        ))
        if len(devices) != len(set(devices)) or (
            (self.phase == HELPER_LOST_PHASE) != bool(devices)
        ):
            raise PhysicalAdapterError(
                "physical failed devices require one helper_lost failure"
            )
        self.failed_device_ids = devices
        self.stale_device_ids = _stale_devices(stale_device_ids, devices)
        if executor_id is not None:
            _text("physical failed executor", executor_id)
        if self.phase == SERVER_EXITED_PHASE and executor_id is None:
            raise PhysicalAdapterError(
                "server_exited failure lacks its executor"
            )
        if returncode is not None and type(returncode) is not int:
            raise PhysicalAdapterError(
                "physical failure return code is invalid"
            )
        self.executor_id = executor_id
        self.returncode = returncode
        _masked_executor(masked_executor_id, self.phase, executor_id)
        self.masked_executor_id = masked_executor_id

    @property
    def failed_device_id(self) -> str | None:
        """The first lost device (sorted), None unless helper_lost."""
        return self.failed_device_ids[0] if self.failed_device_ids else None


class PhysicalExecutionBackend(Protocol):
    """Rig callback surface; commands contain all placement decisions."""

    def apply_transition(
        self,
        command: object,
        payload: object,
        control_check: Callable[[], None],
    ) -> RawTransitionObservation:
        ...

    def execute(
        self,
        command: object,
        payload: object,
        control_check: Callable[[], None],
    ) -> RawExecutionObservation:
        ...

    def execute_with_capacity_release(
        self,
        command: object,
        payload: object,
        control_check: Callable[[], None],
        capacity_release: Callable[[int], None],
    ) -> RawExecutionObservation:
        ...
