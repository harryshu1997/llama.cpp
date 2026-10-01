"""Typed scheduler configuration manifests: campaign."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from dataclasses import dataclass
from pathlib import Path

from .._internal.joint_planner_active import joint_planner_config_from_json
from .._internal.joint_planner_shadow import JointPlannerShadowError
from .._internal.layer_placement import LayerPlacementError
from .._internal.layer_placement_control import MeasuredPlacementConfig
from .._internal.types import canonical_sha256
from .common import (
    CAMPAIGN_MANIFEST_SCHEMA,
    SchedulerConfigurationError,
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


# Android PowerManager thermal status scale: NONE (0) .. SHUTDOWN (6).
ANDROID_THERMAL_STATUS_SHUTDOWN = 6


@dataclass(frozen=True)
class PhoneThermalStatusLimitConfiguration:
    """Opt-in per-phone thermal policy: the phone stays thermally qualified while its
    Android thermal status is at most ``maximum_thermal_status`` (0..6). Absent keeps
    the platform rule (status 0 only), byte-identical to today's artifacts."""

    phone_device_id: str
    maximum_thermal_status: int

    def __post_init__(self) -> None:
        _text(self.phone_device_id, "phone thermal status limit device")
        _integer(self.maximum_thermal_status, "phone maximum thermal status",
                 maximum=ANDROID_THERMAL_STATUS_SHUTDOWN)

    @classmethod
    def from_json(cls, value: object) -> "PhoneThermalStatusLimitConfiguration":
        row = _object(value, "phone thermal status limit")
        _require(set(row) == {"phone_device_id", "maximum_thermal_status"},
                 "phone thermal status limit fields are invalid")
        return cls(row["phone_device_id"], row["maximum_thermal_status"])

    def to_json(self) -> dict[str, object]:
        return {"phone_device_id": self.phone_device_id,
                "maximum_thermal_status": self.maximum_thermal_status}


@dataclass(frozen=True)
class PhoneResidentModelReprovisioningConfiguration:
    """Re-provision the phone FFN sessions for the model(s) the desktop serves.

    ``load_bytes_per_second`` is the session load prior (flash read + HTP upload) used
    until ``minimum_learned_samples`` complete SESSION_LOADING -> SESSION_VERIFIED
    windows have been observed. A decode boundary re-evaluates the layout when the
    scheduler state changed, else at most once per ``boundary_reevaluation_interval_us``
    (0: at every boundary).

    ``early_on_transition`` (opt-in) re-evaluates the layout when a desktop residency
    change is decided (a queued plan carrying a desktop load is committed) and at every
    request release while such a change is decided or pending, not only at the dispatch
    of the loading request. ``count_queued_demand`` (opt-in) follows the model whose
    queued requests are next in dispatch order (its arrived decode work counts as demand),
    not only the models the desktop is already loading or executing."""

    mode: str = "resident-model"
    load_bytes_per_second: int = 200_000_000
    minimum_learned_samples: int = 2
    boundary_reevaluation_interval_us: int = 10_000_000
    early_on_transition: bool = False
    count_queued_demand: bool = False

    def __post_init__(self) -> None:
        _require(self.mode == "resident-model", "phone resident-model reprovisioning mode is invalid")
        _integer(self.load_bytes_per_second, "phone reprovisioning load rate", minimum=1)
        _integer(self.minimum_learned_samples, "phone reprovisioning learned samples", minimum=1)
        _integer(self.boundary_reevaluation_interval_us, "phone reprovisioning boundary interval", minimum=0)
        for name in ("early_on_transition", "count_queued_demand"):
            _require(type(getattr(self, name)) is bool, "phone reprovisioning " + name + " must be a boolean")

    @classmethod
    def from_json(cls, value: object) -> "PhoneResidentModelReprovisioningConfiguration":
        row = _object(value, "phone resident-model reprovisioning")
        _require(set(row) <= {"mode", "load_bytes_per_second", "minimum_learned_samples",
                              "boundary_reevaluation_interval_us", "early_on_transition",
                              "count_queued_demand"},
                 "phone resident-model reprovisioning fields are invalid")
        return cls(row.get("mode", "resident-model"), row.get("load_bytes_per_second", 200_000_000),
                   row.get("minimum_learned_samples", 2),
                   row.get("boundary_reevaluation_interval_us", 10_000_000),
                   row.get("early_on_transition", False), row.get("count_queued_demand", False))

    def to_json(self) -> dict[str, object]:
        # The opt-in flags appear only when set, so earlier configurations serialize unchanged.
        return {"mode": self.mode, "load_bytes_per_second": self.load_bytes_per_second,
                "minimum_learned_samples": self.minimum_learned_samples,
                "boundary_reevaluation_interval_us": self.boundary_reevaluation_interval_us,
                **({"early_on_transition": True} if self.early_on_transition else {}),
                **({"count_queued_demand": True} if self.count_queued_demand else {})}


@dataclass(frozen=True)
class StartupDesktopParentConfiguration:
    model_id: str
    executor_id: str
    desktop_placement_sha256: str

    def __post_init__(self) -> None:
        _text(self.model_id, "startup model")
        _text(self.executor_id, "startup executor")
        value = _text(self.desktop_placement_sha256, "startup desktop placement")
        _require(value.startswith("sha256:") and len(value) == 71
                 and all(c in "0123456789abcdef" for c in value[7:]),
                 "startup desktop placement hash is invalid")

    @classmethod
    def from_json(cls, value: object) -> "StartupDesktopParentConfiguration":
        row = _object(value, "startup desktop parent")
        _require(set(row) == {"model_id", "executor_id", "desktop_placement_sha256"},
                 "startup desktop parent fields are invalid")
        return cls(**row)

    def to_json(self) -> dict[str, str]:
        return {"model_id": self.model_id, "executor_id": self.executor_id,
                "desktop_placement_sha256": self.desktop_placement_sha256}



def _adaptive_decode_overrides(value: object) -> Mapping[str, object] | None:
    """Field-by-field overrides of the adaptive decode controller configuration.

    Keys are `AdaptiveDecodeConfig` field names; values are positive integers, booleans or
    non-empty lists of positive integers (probe fraction ladders). Unknown names are rejected by
    the controller configuration itself when the runner builds it, so a typo fails the run at
    resolve time instead of silently keeping the default."""
    if value is None:
        return None
    row = _object(value, "adaptive decode overrides")
    _require(bool(row), "adaptive decode overrides are empty")
    checked: dict[str, object] = {}
    for name, item in row.items():
        _require(type(name) is str and name.isidentifier(), "adaptive decode override name is invalid")
        if type(item) is bool:
            checked[name] = item
        elif type(item) is list:
            checked[name] = [_integer(entry, "adaptive decode override " + name, minimum=1) for entry in item]
            _require(bool(checked[name]), "adaptive decode override " + name + " is empty")
        else:
            checked[name] = _integer(item, "adaptive decode override " + name, minimum=1)
    return MappingProxyType(checked)


_DISPATCH_POLICY_FLAGS = ("work_conserving_admission", "model_affinity", "continuous_join",
                         "event_replanning")
_DISPATCH_POLICY_BOUNDS = ("affinity_maximum_bypasses", "affinity_maximum_wait_us",
                           "max_barrier_extension_s", "residency_hysteresis_s")
_DISPATCH_POLICY_PPM = ("residency_hysteresis_min_probability_ppm",)


def _dispatch_policy(value: object) -> Mapping[str, object] | None:
    """Opt-in dispatch ordering (runtime `RuntimeDispatchPolicy` fields); absent keeps the legacy order."""
    if value is None:
        return None
    row = _object(value, "dispatch policy")
    _require(bool(row), "dispatch policy is empty")
    _require(set(row) <= {*_DISPATCH_POLICY_FLAGS, *_DISPATCH_POLICY_BOUNDS, *_DISPATCH_POLICY_PPM,
                          "joint_planner", "measured_placement"},
             "dispatch policy has unknown fields")
    checked = {name: _boolean(row[name], "dispatch policy " + name)
               for name in _DISPATCH_POLICY_FLAGS if name in row}
    checked.update({name: _integer(row[name], "dispatch policy " + name, minimum=0)
                    for name in _DISPATCH_POLICY_BOUNDS if name in row})
    checked.update({name: _integer(row[name], "dispatch policy " + name, minimum=0, maximum=1_000_000)
                    for name in _DISPATCH_POLICY_PPM if name in row})
    _require(not checked.get("model_affinity") or checked.get("work_conserving_admission") is True,
             "dispatch policy model_affinity requires work_conserving_admission")
    _require(not checked.get("continuous_join") or checked.get("work_conserving_admission") is True,
             "dispatch policy continuous_join requires work_conserving_admission")
    _require(not checked.get("residency_hysteresis_s")
             or checked.get("work_conserving_admission") is True,
             "dispatch policy residency_hysteresis_s requires work_conserving_admission")
    _require("residency_hysteresis_min_probability_ppm" not in checked
             or bool(checked.get("residency_hysteresis_s")),
             "dispatch policy residency_hysteresis_min_probability_ppm requires residency_hysteresis_s")
    if "joint_planner" in row:
        try:
            joint_planner_config_from_json(row["joint_planner"])
        except JointPlannerShadowError as exc:
            raise SchedulerConfigurationError("dispatch policy " + str(exc)) from exc
        checked["joint_planner"] = dict(sorted(row["joint_planner"].items()))
    if "measured_placement" in row:
        # opt-in WS11 measured layer placement (shadow): validated here, split off by the runner
        try:
            MeasuredPlacementConfig.from_json(row["measured_placement"])
        except (LayerPlacementError, TypeError) as exc:
            raise SchedulerConfigurationError("dispatch policy measured_placement: " + str(exc)) from exc
        checked["measured_placement"] = dict(sorted(row["measured_placement"].items()))
    return MappingProxyType(dict(sorted(checked.items())))


ELASTIC_PHONES_DEFAULTS = MappingProxyType({
    "drop_recovery": False,
    "join": False,
    "join_probe_interval_s": 10,
    "readmission_cooldown_s": 60,
    "max_readmissions_per_device": 3,
})


# How a live llama-server survives the loss of one of its FFN helpers (drop recovery only):
# "retire" stops it and reloads (the default), "mask_out" keeps it and masks the helper out.
HELPER_LOSS_RECOVERY_MODES = ("mask_out", "retire")


def elastic_phones_configuration(value: object) -> Mapping[str, object] | None:
    """Opt-in phone membership changes at runtime; absent keeps today's static phone set.

    ``drop_recovery`` recovers requests of a lost helper and reloads a dead server; ``join``
    re-admits a lost or absent co-helper after an identity-pinned restart. The normalized form
    spells out every field; the optional ``helper_loss_recovery`` (``mask_out`` or ``retire``,
    requires ``drop_recovery``) appears only when given, absent means ``retire``."""
    if value is None:
        return None
    row = _object(value, "elastic phones")
    _require(set(row) <= {*ELASTIC_PHONES_DEFAULTS, "helper_loss_recovery"},
             "elastic phones has unknown fields")
    checked = dict(ELASTIC_PHONES_DEFAULTS)
    for name in ("drop_recovery", "join"):
        if name in row:
            checked[name] = _boolean(row[name], "elastic phones " + name)
    _require(checked["drop_recovery"] or checked["join"], "elastic phones enables no behaviour")
    minimums = {"join_probe_interval_s": 1, "readmission_cooldown_s": 0, "max_readmissions_per_device": 0}
    for name, minimum in minimums.items():
        if name in row:
            checked[name] = _integer(row[name], "elastic phones " + name, minimum=minimum)
    if "helper_loss_recovery" in row:
        _require(row["helper_loss_recovery"] in HELPER_LOSS_RECOVERY_MODES,
                 "elastic phones helper_loss_recovery must be mask_out or retire")
        _require(checked["drop_recovery"], "elastic phones helper_loss_recovery requires drop_recovery")
        checked["helper_loss_recovery"] = row["helper_loss_recovery"]
    return MappingProxyType(dict(sorted(checked.items())))


# intel_pstate energy_performance_preference values (``/sys/devices/system/cpu/cpu*/cpufreq``)
CPU_EPP_VALUES = ("default", "performance", "balance_performance", "balance_power", "power")


@dataclass(frozen=True)
class DeviceIdlePowerConfiguration:
    """Idle sub-policy of ``device_power``: while the scheduler knows the GPU has no work for at
    least ``min_gap_s`` the SM clock is locked at ``gpu_min_clocks_mhz`` (and the CPU EPP set to
    ``cpu_epp`` when given); the lock is lifted ``lead_ms`` before the next known GPU work."""

    min_gap_s: int
    lead_ms: int
    gpu_min_clocks_mhz: int
    cpu_epp: str | None

    def __post_init__(self) -> None:
        _integer(self.min_gap_s, "device power idle min_gap_s", minimum=1)
        _integer(self.lead_ms, "device power idle lead_ms", minimum=1)
        _integer(self.gpu_min_clocks_mhz, "device power idle gpu_min_clocks_mhz", minimum=1)
        _require(self.lead_ms < self.min_gap_s * 1000,
                 "device power idle lead_ms must be shorter than min_gap_s")
        _require(self.cpu_epp is None or self.cpu_epp in CPU_EPP_VALUES,
                 "device power idle cpu_epp is not an intel_pstate EPP value")

    @classmethod
    def from_json(cls, value: object) -> "DeviceIdlePowerConfiguration":
        row = _object(value, "device power idle")
        _require(set(row) == {"min_gap_s", "lead_ms", "gpu_min_clocks_mhz", "cpu_epp"},
                 "device power idle fields are invalid")
        return cls(row["min_gap_s"], row["lead_ms"], row["gpu_min_clocks_mhz"], row["cpu_epp"])

    def to_json(self) -> dict[str, object]:
        return {"cpu_epp": self.cpu_epp, "gpu_min_clocks_mhz": self.gpu_min_clocks_mhz,
                "lead_ms": self.lead_ms, "min_gap_s": self.min_gap_s}


@dataclass(frozen=True)
class DeviceDecodeCapConfiguration:
    """Decode sub-policy of ``device_power``: while a GPU execution is active the SM clock range is
    ``[idle.gpu_min_clocks_mhz, sm_max_mhz]`` (memory clocks are never touched). With
    ``protect_prefill`` the cap holds only while every active execution is past its first token:
    an execution start restores full clocks synchronously before its prompt is processed."""

    sm_max_mhz: int
    protect_prefill: bool = False

    def __post_init__(self) -> None:
        _integer(self.sm_max_mhz, "device power decode_cap sm_max_mhz", minimum=1)
        _boolean(self.protect_prefill, "device power decode_cap protect_prefill")

    @classmethod
    def from_json(cls, value: object) -> "DeviceDecodeCapConfiguration":
        row = _object(value, "device power decode_cap")
        _require({"sm_max_mhz"} <= set(row) <= {"sm_max_mhz", "protect_prefill"},
                 "device power decode_cap fields are invalid")
        return cls(row["sm_max_mhz"], _boolean(row.get("protect_prefill", False),
                                                "device power decode_cap protect_prefill"))

    def to_json(self) -> dict[str, object]:
        return {"sm_max_mhz": self.sm_max_mhz, **({"protect_prefill": True} if self.protect_prefill else {})}


# ``device_power.arrival_information``: "oracle" is fed the next trace arrival before it happens (the
# behaviour of a policy without the key); "online" only ever sees arrivals that already happened.
DEVICE_POWER_ARRIVAL_INFORMATION = ("oracle", "online")
DEVICE_POWER_ONLINE_PREDICTORS = ("global", "per_model")


@dataclass(frozen=True)
class DeviceOnlineIdleConfiguration:
    """Online idle rule (``arrival_information: "online"``). The controller learns inter-arrival
    gaps from observed arrivals only (``predictor``: one global sequence or one per model) and,
    while the GPU is idle, enters IDLE_MIN when the pessimistic estimate of an arrival within
    ``idle.min_gap_s`` is at most ``max_arrival_probability_ppm``, or once the GPU has been idle
    for ``fallback_idle_ms`` (ski-rental timeout; null = predictor only). It restores on every
    observed arrival, queued ticket, transition, load and (synchronously) execution start."""

    predictor: str
    max_arrival_probability_ppm: int
    fallback_idle_ms: int | None

    def __post_init__(self) -> None:
        _require(self.predictor in DEVICE_POWER_ONLINE_PREDICTORS,
                 "device power online_idle predictor must be global or per_model")
        _integer(self.max_arrival_probability_ppm, "device power online_idle max_arrival_probability_ppm",
                 minimum=1, maximum=999_999)
        if self.fallback_idle_ms is not None:
            _integer(self.fallback_idle_ms, "device power online_idle fallback_idle_ms", minimum=1)

    @classmethod
    def from_json(cls, value: object) -> "DeviceOnlineIdleConfiguration":
        row = _object(value, "device power online_idle")
        _require(set(row) == {"predictor", "max_arrival_probability_ppm", "fallback_idle_ms"},
                 "device power online_idle fields are invalid")
        return cls(row["predictor"], row["max_arrival_probability_ppm"], row["fallback_idle_ms"])

    def to_json(self) -> dict[str, object]:
        return {"fallback_idle_ms": self.fallback_idle_ms,
                "max_arrival_probability_ppm": self.max_arrival_probability_ppm, "predictor": self.predictor}


@dataclass(frozen=True)
class DevicePowerConfiguration:
    """Opt-in scheduler-driven power control of one GPU device (campaign ``device_power``).

    ``gpu_uuid`` pins every ``nvidia-smi`` command to that board. ``idle`` is the base sub-policy
    and carries the device's lowest SM clock, which is also the floor of ``decode_cap`` and the
    lock of ``load_min`` (clocks at the floor while a model loads from disk), so both require it.
    ``arrival_information`` absent keeps the calendar-fed controller exactly as before; set, the
    controller also records ``DEVICE_POWER_TELEMETRY.json`` (idle intervals, restore waits) and
    "online" (with ``online_idle``) never receives a future arrival.
    Absent (None) keeps today's behaviour and every RESULT byte-identical."""

    device: str
    gpu_uuid: str
    idle: DeviceIdlePowerConfiguration | None = None
    decode_cap: DeviceDecodeCapConfiguration | None = None
    load_min: bool = False
    arrival_information: str | None = None
    online_idle: DeviceOnlineIdleConfiguration | None = None

    @property
    def online(self) -> bool:
        return self.arrival_information == "online"

    def __post_init__(self) -> None:
        _require(self.arrival_information is None or self.arrival_information in DEVICE_POWER_ARRIVAL_INFORMATION,
                 "device power arrival_information must be oracle or online")
        _require(self.online_idle is None or isinstance(self.online_idle, DeviceOnlineIdleConfiguration),
                 "device power online_idle is invalid")
        _require((self.online_idle is not None) == (self.arrival_information == "online"),
                 "device power online_idle is required exactly when arrival_information is online")
        _require(self.arrival_information != "online" or self.idle is not None,
                 "device power online arrival information requires idle")
        _text(self.device, "device power device")
        _text(self.gpu_uuid, "device power gpu_uuid")
        _require(self.idle is None or isinstance(self.idle, DeviceIdlePowerConfiguration),
                 "device power idle is invalid")
        _require(self.decode_cap is None or isinstance(self.decode_cap, DeviceDecodeCapConfiguration),
                 "device power decode_cap is invalid")
        _boolean(self.load_min, "device power load_min")
        _require(self.idle is not None or self.decode_cap is not None or self.load_min,
                 "device power enables no behaviour")
        _require(self.idle is not None or (self.decode_cap is None and not self.load_min),
                 "device power decode_cap and load_min require idle (its gpu_min_clocks_mhz is the floor)")
        _require(self.decode_cap is None or self.decode_cap.sm_max_mhz > self.idle.gpu_min_clocks_mhz,
                 "device power decode_cap sm_max_mhz must exceed idle gpu_min_clocks_mhz")

    @classmethod
    def from_json(cls, value: object) -> "DevicePowerConfiguration":
        row = _object(value, "device power")
        _require({"device", "gpu_uuid"} <= set(row) <= {"device", "gpu_uuid", "idle", "decode_cap", "load_min",
                                                          "arrival_information", "online_idle"},
                 "device power fields are invalid")
        return cls(
            device=row["device"],
            gpu_uuid=row["gpu_uuid"],
            idle=None if row.get("idle") is None else DeviceIdlePowerConfiguration.from_json(row["idle"]),
            decode_cap=(None if row.get("decode_cap") is None
                        else DeviceDecodeCapConfiguration.from_json(row["decode_cap"])),
            load_min=_boolean(row.get("load_min", False), "device power load_min"),
            arrival_information=row.get("arrival_information"),
            online_idle=(None if row.get("online_idle") is None
                         else DeviceOnlineIdleConfiguration.from_json(row["online_idle"])),
        )

    def to_json(self) -> dict[str, object]:
        return {
            **({} if self.arrival_information is None else {"arrival_information": self.arrival_information}),
            **({} if self.decode_cap is None else {"decode_cap": self.decode_cap.to_json()}),
            "device": self.device,
            "gpu_uuid": self.gpu_uuid,
            **({} if self.idle is None else {"idle": self.idle.to_json()}),
            **({"load_min": True} if self.load_min else {}),
            **({} if self.online_idle is None else {"online_idle": self.online_idle.to_json()}),
        }


SPECULATIVE_ROWS_DRAFT_MAX_LIMIT = 3
SPECULATIVE_ROWS_FIELDS = frozenset({
    "draft_model_path", "draft_max", "draft_min", "row_budget", "qualified_rows",
    "draft_gpu_layers", "patched_server_sha256",
})


@dataclass(frozen=True)
class SpeculativeRowsModelConfiguration:
    """Opt-in draft-model speculation of one target model (campaign ``speculative_rows``).

    ``draft_model_path`` is the draft GGUF (same vocabulary as the target, checked when the
    catalog is materialized); ``draft_max`` (1..3) and ``draft_min`` bound the draft of one
    verification step; ``row_budget`` caps the rows one phone FFN call may carry (None: the
    phone contract's ``max_tokens``); ``qualified_rows`` lists the per-call row counts qualified
    on the helpers (None: every count up to the budget); ``draft_gpu_layers`` places the draft
    (0 keeps it on the CPU: the draft context copies the target's context size, so its KV cache
    is large); ``patched_server_sha256`` pins a llama-server build that isolates the draft from
    the phone FFN split and honours per-request ``speculative.n_max`` (``server_speculative.diff``).
    Without the pin only desktop-only launches carry the draft."""

    draft_model_path: Path
    draft_max: int
    draft_min: int = 0
    row_budget: int | None = None
    qualified_rows: tuple[int, ...] | None = None
    draft_gpu_layers: int = 0
    patched_server_sha256: str | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.draft_model_path, Path) and self.draft_model_path.is_absolute(),
                 "speculative rows draft_model_path must be an absolute path")
        _integer(self.draft_max, "speculative rows draft_max", minimum=1,
                 maximum=SPECULATIVE_ROWS_DRAFT_MAX_LIMIT)
        _integer(self.draft_min, "speculative rows draft_min", maximum=self.draft_max)
        _require(self.row_budget is None or _integer(self.row_budget, "speculative rows row_budget", minimum=1) > 0,
                 "speculative rows row_budget is invalid")
        if self.qualified_rows is not None:
            rows = tuple(self.qualified_rows)
            _require(bool(rows) and all(type(row) is int and row >= 1 for row in rows)
                     and len(set(rows)) == len(rows), "speculative rows qualified_rows are invalid")
            object.__setattr__(self, "qualified_rows", tuple(sorted(rows)))
        _integer(self.draft_gpu_layers, "speculative rows draft_gpu_layers")
        if self.patched_server_sha256 is not None:
            pin = _text(self.patched_server_sha256, "speculative rows patched_server_sha256")
            _require(pin.startswith("sha256:") and _SHA256.fullmatch(pin) is not None,
                     "speculative rows patched_server_sha256 must be a sha256: digest")

    @classmethod
    def from_json(cls, value: object, base: Path) -> "SpeculativeRowsModelConfiguration":
        row = _object(value, "speculative rows model")
        _require({"draft_model_path", "draft_max"} <= set(row) <= SPECULATIVE_ROWS_FIELDS,
                 "speculative rows model fields are invalid")
        rows = row.get("qualified_rows")
        return cls(
            draft_model_path=_path(row.get("draft_model_path"), base, "speculative rows draft_model_path"),
            draft_max=row["draft_max"],
            draft_min=row.get("draft_min", 0),
            row_budget=row.get("row_budget"),
            qualified_rows=None if rows is None else tuple(
                _integer(item, "speculative rows qualified row", minimum=1)
                for item in _sequence(rows, "speculative rows qualified_rows")),
            draft_gpu_layers=row.get("draft_gpu_layers", 0),
            patched_server_sha256=row.get("patched_server_sha256"),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "draft_gpu_layers": self.draft_gpu_layers,
            "draft_max": self.draft_max,
            "draft_min": self.draft_min,
            "draft_model_path": str(self.draft_model_path),
            **({} if self.patched_server_sha256 is None else {"patched_server_sha256": self.patched_server_sha256}),
            **({} if self.qualified_rows is None else {"qualified_rows": list(self.qualified_rows)}),
            **({} if self.row_budget is None else {"row_budget": self.row_budget}),
        }


def speculative_rows_configuration(
    value: object, base: Path
) -> Mapping[str, SpeculativeRowsModelConfiguration] | None:
    """Opt-in speculative rows keyed by target model id; absent keeps every launch draft-free."""
    if value is None:
        return None
    rows = _object(value, "speculative rows")
    _require(bool(rows), "speculative rows names no model")
    return MappingProxyType({
        _text(model_id, "speculative rows model id"): SpeculativeRowsModelConfiguration.from_json(row, base)
        for model_id, row in sorted(rows.items())
    })


def speculative_rows_json(
    configuration: Mapping[str, SpeculativeRowsModelConfiguration] | None,
) -> dict[str, object] | None:
    return None if configuration is None else {
        model_id: row.to_json() for model_id, row in sorted(configuration.items())
    }


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
    # decode-only relocation admission: host budget every server and its released FFN share is booked against
    host_memory_budget_bytes: int | None = None
    adaptive_maximum_probe_attempts_per_context: int | None = None
    adaptive_decode_overrides: Mapping[str, object] | None = None
    fixed_phone_residency: FixedPhoneResidencyConfiguration | None = None
    include_startup_preparation: bool = False
    phone_htp_memory_caps: tuple[PhoneHtpMemoryCapConfiguration, ...] = ()
    phone_thermal_status_limits: tuple[PhoneThermalStatusLimitConfiguration, ...] = ()
    protected_work_policy: str = "strict"
    startup_desktop_parents: tuple[StartupDesktopParentConfiguration, ...] = ()
    phone_resident_model_reprovisioning: PhoneResidentModelReprovisioningConfiguration | None = None
    dispatch_policy: Mapping[str, object] | None = None
    elastic_phones: Mapping[str, object] | None = None
    device_power: DevicePowerConfiguration | None = None
    speculative_rows: Mapping[str, SpeculativeRowsModelConfiguration] | None = None

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
        probe_attempts = row.get("adaptive_maximum_probe_attempts_per_context")
        overrides = _adaptive_decode_overrides(row.get("adaptive_decode_overrides"))
        maximum_phone_sessions = row.get("maximum_phone_sessions")
        fault_injection = _optional_text(
            row.get("helper_preparation_fault_injection"),
            "helper preparation fault injection",
        )
        host_budget = row.get("host_memory_budget_bytes")
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
        thermal_limits = tuple(PhoneThermalStatusLimitConfiguration.from_json(value)
                               for value in _sequence(row.get("phone_thermal_status_limits", []),
                                                      "phone thermal status limits"))
        _require(len({limit.phone_device_id for limit in thermal_limits}) == len(thermal_limits),
                 "phone thermal status limit devices are not unique")
        startup = tuple(StartupDesktopParentConfiguration.from_json(value)
                        for value in _sequence(row.get("startup_desktop_parents", []), "startup desktop parents"))
        _require(len({value.executor_id for value in startup}) == len(startup),
                 "startup desktop executors are duplicated")
        _require(not startup or row.get("include_startup_preparation") is True,
                 "startup desktop preload requires paid preparation accounting")
        reprovisioning = row.get("phone_resident_model_reprovisioning")
        _require(reprovisioning is None or row.get("fixed_phone_residency") is None,
                 "fixed phone residency cannot be re-provisioned")
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
            adaptive_maximum_probe_attempts_per_context=(
                None if probe_attempts is None
                else _integer(probe_attempts, "adaptive maximum probe attempts per context", minimum=1)
            ),
            adaptive_decode_overrides=overrides,
            host_memory_budget_bytes=(
                None if host_budget is None
                else _integer(host_budget, "host memory budget bytes", minimum=1)
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
            phone_thermal_status_limits=tuple(
                sorted(thermal_limits, key=lambda limit: limit.phone_device_id)),
            startup_desktop_parents=startup,
            phone_resident_model_reprovisioning=(
                None if reprovisioning is None else
                PhoneResidentModelReprovisioningConfiguration.from_json(reprovisioning)
            ),
            dispatch_policy=_dispatch_policy(row.get("dispatch_policy")),
            elastic_phones=elastic_phones_configuration(row.get("elastic_phones")),
            device_power=(None if row.get("device_power") is None
                          else DevicePowerConfiguration.from_json(row["device_power"])),
            speculative_rows=speculative_rows_configuration(row.get("speculative_rows"), base),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "adaptive_decode_overrides": (
                None if self.adaptive_decode_overrides is None else dict(self.adaptive_decode_overrides)
            ),
            "adaptive_maximum_probe_attempts_per_context": self.adaptive_maximum_probe_attempts_per_context,
            "adaptive_minimum_remaining_tokens": self.adaptive_minimum_remaining_tokens,
            "host_memory_budget_bytes": self.host_memory_budget_bytes,
            **({} if self.fixed_phone_residency is None else {
                "fixed_phone_residency": self.fixed_phone_residency.to_json(),
            }),
            **({"include_startup_preparation": True} if self.include_startup_preparation else {}),
            **({} if self.dispatch_policy is None else {"dispatch_policy": dict(self.dispatch_policy)}),
            **({} if self.elastic_phones is None else {"elastic_phones": dict(self.elastic_phones)}),
            **({} if self.device_power is None else {"device_power": self.device_power.to_json()}),
            **({} if self.speculative_rows is None
               else {"speculative_rows": speculative_rows_json(self.speculative_rows)}),
            **({"startup_desktop_parents": [row.to_json() for row in self.startup_desktop_parents]}
               if self.startup_desktop_parents else {}),
            **({"phone_htp_memory_caps": [cap.to_json() for cap in self.phone_htp_memory_caps]}
               if self.phone_htp_memory_caps else {}),
            **({"phone_thermal_status_limits": [
                    row.to_json() for row in self.phone_thermal_status_limits]}
               if self.phone_thermal_status_limits else {}),
            **({} if self.phone_resident_model_reprovisioning is None else {
                "phone_resident_model_reprovisioning": self.phone_resident_model_reprovisioning.to_json(),
            }),
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
