"""Shared helpers, exception types, and schedule records for the unified scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Mapping, TypeVar
from .._internal.policy import LeaseRecord
from .._internal.lifecycle import UnifiedScheduleError
from .._internal.runtime_capabilities import RuntimeExecutorState
from .._internal.runtime_cost import RuntimeModelArtifact
from .._internal.runtime_plan import (
    AutomatedRouteCandidate,
    RuntimeHelperExecutionEnvelope,
    RuntimeTransitionReceipt,
)
from .._internal.runtime_residency_cohorts import RuntimeResidencyComponentIdentity
from .._internal.adaptive_decode_contracts import AdaptiveDecodePolicy
from .._internal.dynamic_residency import DynamicResidencyDecision
from .._internal.gpu_backfill import GpuBackfillDecision, GpuWavefrontDecision
from .._internal.phone_residency import PhoneOffloadDecision
from .._internal.phone_arbiter import PhoneArbiterDecision

if TYPE_CHECKING:
    from ..scheduler import UnifiedScheduler


class _ReadyHelperSafetyDeferred(UnifiedScheduleError):
    pass


class _StalePhoneSessionAssignment(UnifiedScheduleError):
    pass


_DORMANT_PHONE_FFN_RUNTIME_PARAMETER = "dormant_phone_ffn_runtime_v1"


_RECOVERABLE_ERRORS = (ValueError,)


_DORMANT_PHONE_FFN_RUNTIME_KEYS = frozenset({
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


_RuntimeResult = TypeVar("_RuntimeResult")


@dataclass(frozen=True)
class _ModelPlacementCompatibility:
    compatible: bool
    component: RuntimeResidencyComponentIdentity
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _RequestHelperPreparation:
    request_id: str
    request_ticket_id: str
    # Physical completion survives request re-admission; no request leases are kept.
    model: RuntimeModelArtifact
    base_executor_id: str
    preparation_ticket_id: str
    phone_layout_generation: int
    phone_layout_geometry_sha256: str
    projection_token_sha256: str
    operator_plan_sha256: str
    helper_envelope: RuntimeHelperExecutionEnvelope
    transition_ids: tuple[str, ...]
    resource_lease_tokens: tuple[str, ...]
    yielding_resource_ids: tuple[str, ...]
    memory_owner_id: str
    started_at_us: int
    ready_at_us: int
    state: str
    phone_safety_state: RuntimeExecutorState | None = None
    transition_receipts: tuple[RuntimeTransitionReceipt, ...] = ()
    verification_sha256: str | None = None


@dataclass(frozen=True)
class _RequestHelperAttachAttempt:
    attached: bool
    retryable: bool
    reason: str


@dataclass(frozen=True)
class _LateRequestHelperContext:
    helper: RuntimeHelperExecutionEnvelope
    baseline: AdaptiveDecodePolicy
    candidates: tuple[AdaptiveDecodePolicy, ...]
    ticket_policy: AdaptiveDecodePolicy | None
    component: RuntimeResidencyComponentIdentity
    evidence_state: str


def _event_replanning_active(controller: object) -> bool:
    """Whether ``dispatch_policy.event_replanning`` is configured (off for any stand-in)."""
    policy = getattr(getattr(controller, "_runtime_controller", None), "dispatch_policy", None)
    return getattr(policy, "event_replanning", False) is True


def _runtime_serialized(
    method: Callable[..., _RuntimeResult],
) -> Callable[..., _RuntimeResult]:
    @wraps(method)
    def wrapped(self: "UnifiedScheduler", *args: object, **kwargs: object):
        with self._runtime_lock:
            if not _event_replanning_active(self):
                return method(self, *args, **kwargs)
            # Under dispatch_policy.event_replanning the events a call observed are
            # handled once, when the outermost serialized call returns.
            depth = self.__dict__.get("_event_replanning_depth", 0)
            self._event_replanning_depth = depth + 1
            try:
                result = method(self, *args, **kwargs)
            finally:
                self._event_replanning_depth = depth
            if depth == 0:
                from .automated_requests_ops.event_replanning import process_pending_events
                process_pending_events(self)
            return result

    return wrapped


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value:
        raise UnifiedScheduleError(f"{name} must be a non-empty string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise UnifiedScheduleError(f"{name} must be ASCII") from exc
    return value


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise UnifiedScheduleError("invalid ceiling division")
    return (numerator + denominator - 1) // denominator


def _phone_shard_structure(row: object) -> tuple[object, ...]:
    return (
        getattr(row, "artifact_sha256"),
        getattr(row, "session_id"),
        getattr(row, "endpoint"),
        getattr(row, "layer_mask"),
        getattr(row, "maximum_columns"),
        getattr(row, "resident_bytes"),
        getattr(row, "resident_geometry_sha256"),
        getattr(row, "operator_plan_sha256"),
    )


def _apportion_measured_route_energy(
    lower_uj: int,
    energy_uj: int,
    upper_uj: int,
    warm_energy_uj: int,
    *,
    has_transition: bool,
) -> tuple[int, int, int, int, int, int] | None:
    """Preserve a measured route total while exposing its attribution."""
    if not has_transition:
        return lower_uj, energy_uj, upper_uj, 0, 0, 0
    if not (
        0 < lower_uj <= energy_uj <= upper_uj
        and 0 < warm_energy_uj < energy_uj
    ):
        return None
    warm_ppm = max(
        1,
        min(999_999, warm_energy_uj * 1_000_000 // energy_uj),
    )
    warm_lower = max(1, lower_uj * warm_ppm // 1_000_000)
    warm = max(warm_lower, energy_uj * warm_ppm // 1_000_000)
    warm_upper = max(warm, upper_uj * warm_ppm // 1_000_000)
    transition = (
        lower_uj - warm_lower,
        energy_uj - warm,
        upper_uj - warm_upper,
    )
    if not (
        0 <= transition[0] <= transition[1] <= transition[2]
    ):
        return None
    return warm_lower, warm, warm_upper, *transition


def _paired_energy_upper_uj(
    candidate: AutomatedRouteCandidate,
    parent: AutomatedRouteCandidate,
    energy_boundary_id: str,
    *,
    warm: bool,
) -> int | None:
    """Return a conditional upper bound from matched component evidence."""
    evidence = candidate.residency_break_even
    evidence_kind = (
        None if evidence is None
        else evidence.get("paired_energy_evidence")
    )
    if (
        evidence is None
        or evidence_kind not in {"MEASURED", "ASSUMED_4P5W"}
        or evidence.get("paired_energy_boundary_id") != energy_boundary_id
        or evidence.get("paired_energy_parent_route_id")
            != parent.candidate_id
        or evidence.get("paired_energy_parent_placement_sha256")
            != parent.plan.desktop_placement_sha256
        or candidate.paired_baseline_route_id != parent.candidate_id
        or candidate.plan.desktop_placement_sha256 is None
        or candidate.plan.desktop_placement_sha256
            != parent.plan.desktop_placement_sha256
        or (
            evidence_kind == "MEASURED"
            and (
                candidate.cost.energy_evidence != "MEASURED"
                or parent.cost.energy_evidence != "MEASURED"
            )
        )
        or (
            evidence_kind == "ASSUMED_4P5W"
            and (
                candidate.cost.energy_evidence != "ASSUMED"
                or parent.cost.energy_evidence != "ASSUMED"
                or evidence.get("phone_energy_evidence")
                    != "ASSUMED_4P5W"
            )
        )
    ):
        return None
    baseline_groups = evidence.get(
        "adaptive_history_baseline_group_count"
    )
    candidate_groups = evidence.get(
        "adaptive_history_selected_group_count"
    )
    if (
        type(baseline_groups) is not int
        or baseline_groups < 2
        or type(candidate_groups) is not int
        or candidate_groups < 2
    ):
        return None
    if not warm:
        portfolio_upper = evidence.get(
            "portfolio_effective_paired_energy_upper_uj"
        )
        portfolio = evidence.get(
            "phone_residency_portfolio_authorization"
        )
        if portfolio_upper is not None:
            if (
                type(portfolio_upper) is not int
                or portfolio_upper <= 0
                or not isinstance(portfolio, Mapping)
                or type(portfolio.get("net_benefit_share_uj")) is not int
                or portfolio["net_benefit_share_uj"] <= 0
            ):
                return None
            return portfolio_upper
    prefix = "paired_warm_energy_delta" if warm else "paired_energy_delta"
    deltas = tuple(
        evidence.get(prefix + suffix)
        for suffix in ("_lower_uj", "_uj", "_upper_uj")
    )
    if (
        any(type(value) is not int for value in deltas)
        or not deltas[0] <= deltas[1] <= deltas[2]
    ):
        return None
    parent_lower = (
        parent.cost.warm_execution_energy_lower_uj
        if warm else parent.cost.fleet_energy_lower_uj
    )
    if parent_lower is None:
        return None
    upper = parent_lower + deltas[2]
    return upper if upper > 0 else None


def _residency_break_even_warm_upper_uj(
    evidence: Mapping[str, object] | None,
    *,
    paired_upper_uj: int | None = None,
) -> int | None:
    if paired_upper_uj is not None:
        if type(paired_upper_uj) is not int or paired_upper_uj < 0:
            raise UnifiedScheduleError(
                "paired warm route energy is invalid"
            )
        return paired_upper_uj
    if evidence is None:
        return None
    value = evidence.get("paired_warm_candidate_upper_uj")
    if value is None:
        value = evidence.get("warm_route_upper_uj")
    if value is not None:
        if type(value) is not int or value < 0:
            raise UnifiedScheduleError(
                "residency break-even warm energy is invalid"
            )
        return value
    if any(
        key in evidence
        for key in (
            "candidate_cohort_upper_uj",
            "desktop_cohort_lower_uj",
            "expected_use_count",
            "incremental_transition_energy_uj",
            "passed",
            "transition_energy_uj",
        )
    ):
        raise UnifiedScheduleError(
            "residency break-even warm energy is absent"
        )
    return None


@dataclass(frozen=True)
class DynamicPlacementLease:
    lease_id: str
    owner_id: str
    placement_ids: tuple[str, ...]
    source_snapshot_id: str
    source_generation: int
    source_epoch_key: str
    acquired_at_us: int


@dataclass(frozen=True)
class PhoneOffloadSchedule:
    decision: PhoneOffloadDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]
    placement_ids: tuple[str, ...]
    source_snapshot_id: str | None
    source_generation: int | None
    source_epoch_key: str | None


@dataclass(frozen=True)
class PhoneArbiterSchedule:
    decision: PhoneArbiterDecision
    phone_schedule: PhoneOffloadSchedule | None

    @property
    def owner_id(self) -> str | None:
        return (
            None
            if self.phone_schedule is None
            else self.phone_schedule.owner_id
        )

    @property
    def leases(self) -> tuple[LeaseRecord, ...]:
        return () if self.phone_schedule is None else self.phone_schedule.leases

    @property
    def queue_by_resource_us(self) -> Mapping[str, int]:
        return (
            MappingProxyType({})
            if self.phone_schedule is None
            else self.phone_schedule.queue_by_resource_us
        )

    @property
    def blocking_resources(self) -> tuple[str, ...]:
        return (
            ()
            if self.phone_schedule is None
            else self.phone_schedule.blocking_resources
        )


@dataclass(frozen=True)
class DynamicResidencySchedule:
    decision: DynamicResidencyDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]


@dataclass(frozen=True)
class GpuBackfillSchedule:
    decision: GpuBackfillDecision
    owner_id: str | None
    leases: tuple[LeaseRecord, ...]
    queue_by_resource_us: Mapping[str, int]
    blocking_resources: tuple[str, ...]


@dataclass(frozen=True)
class GpuWavefrontSchedule:
    decision: GpuWavefrontDecision
    backfill_schedule: GpuBackfillSchedule

    @property
    def owner_id(self) -> str | None:
        return self.backfill_schedule.owner_id

    @property
    def leases(self) -> tuple[LeaseRecord, ...]:
        return self.backfill_schedule.leases

    @property
    def queue_by_resource_us(self) -> Mapping[str, int]:
        return self.backfill_schedule.queue_by_resource_us

    @property
    def blocking_resources(self) -> tuple[str, ...]:
        return self.backfill_schedule.blocking_resources
