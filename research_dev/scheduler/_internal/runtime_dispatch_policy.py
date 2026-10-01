"""Opt-in dispatch ordering: work-conserving admission, model affinity, continuous join,
residency hysteresis, event re-planning."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Mapping, Sequence


RUNTIME_DISPATCH_POLICY_SCHEMA = "research-scheduler-dispatch-policy-v1"


class RuntimeDispatchPolicyError(ValueError):
    pass


_CONTINUOUS_JOIN_FIELDS = frozenset({"continuous_join", "max_barrier_extension_s"})
_RESIDENCY_HYSTERESIS_FIELDS = frozenset({
    "residency_hysteresis_s", "residency_hysteresis_min_probability_ppm",
})
_EVENT_REPLANNING_FIELDS = frozenset({"event_replanning"})
DEFAULT_RESIDENCY_HYSTERESIS_MIN_PROBABILITY_PPM = 500_000
# Serialized only when set, so policies written before them serialize unchanged.
_OPTIONAL_FIELDS = (
    _CONTINUOUS_JOIN_FIELDS | _RESIDENCY_HYSTERESIS_FIELDS | _EVENT_REPLANNING_FIELDS
)


@dataclass(frozen=True)
class RuntimeDispatchPolicy:
    """Every behaviour defaults off, so baseline arms keep the legacy order.

    ``work_conserving_admission``: only plans that may change desktop
    residency (any transition on an exclusive residency device) keep the
    arrival-order barrier. Work on the resident model is
    ordered by its reserved lanes and runs ahead of a queued residency change
    whose lanes it frees in time; a queued attempt whose only transition is
    the executor publication another attempt just completed is replanned at
    once instead of after that attempt.

    ``model_affinity`` (requires work-conserving admission): an arriving
    request of the resident model that would be reserved behind a queued
    residency change of another model displaces that change and the work
    queued behind it (they are replanned after it). A queued request that
    arrived while its model was loading is treated the same way once the
    model is published: it is replanned and displaces the not-started
    residency changes of other models it still waits on. A displaced request
    is protected once it has been displaced ``affinity_maximum_bypasses``
    times or has waited ``affinity_maximum_wait_us`` since its arrival; a
    protected request blocks every further displacement.

    ``continuous_join`` (requires work-conserving admission): an arriving
    request of a model already decoding on a server joins that server as a
    desktop parent in a free slot instead of reserving an assisted route
    behind the running request that holds the phone lanes; phone assistance
    reaches it through the shared helper window under server policy
    coherence. A joiner ordered behind a queued residency change of another
    model precedes that change when its predicted finish extends the
    server's committed busy window by at most ``max_barrier_extension_s``
    seconds (0: the joiner never extends the window). Co-tenants of one
    server are batch composition, not external desktop activity, for the
    adaptive measurement context.

    ``residency_hysteresis_s`` (requires work-conserving admission; 0 off):
    a queued residency change of another model on the released resource may
    wait up to this many seconds after the exclusive residency resource was
    last released by a request of the model resident then, but only when the
    wait can pay: a same-model request is queued and can still run first
    (held until it is dispatched or the window ends), or the learned
    same-model inter-arrival gaps predict an arrival inside the window with
    probability at least ``residency_hysteresis_min_probability_ppm``. A
    change whose own model has queued work that already waited longer than
    the window is never held. A same-model request queued or arriving inside
    the window runs first (its lanes free before the held change, or model
    affinity displaces the change); the change is never held more than the
    window beyond its earliest possible start.

    ``event_replanning`` (independent of the others): queued and deferred
    decisions are revisited when the runtime event that made them wait
    happens, not at the next arrival or dispatch. A release of phone helper
    sessions (decode completion, lease release, request completion,
    cancellation or failure) re-evaluates the phone residency re-provisioning
    when sessions an earlier decision counted as in use are free now; a phone
    that becomes admissible again (thermal gate cleared, telemetry recovered,
    readmitted after quarantine) re-evaluates the phone layout and replans the
    not-started attempts decided while it was out; a proposed phone layout
    that a live request cannot prepare records why
    (``PREPARATION_BLOCKED``/``PREPARATION_UNBLOCKED``). The events are
    handled once, when the outermost scheduler call that observed them
    returns.
    """

    work_conserving_admission: bool = False
    model_affinity: bool = False
    affinity_maximum_bypasses: int = 10
    affinity_maximum_wait_us: int = 1_200_000_000
    continuous_join: bool = False
    max_barrier_extension_s: int = 0
    residency_hysteresis_s: int = 0
    residency_hysteresis_min_probability_ppm: int = (
        DEFAULT_RESIDENCY_HYSTERESIS_MIN_PROBABILITY_PPM
    )
    event_replanning: bool = False

    def __post_init__(self) -> None:
        for name in (
            "work_conserving_admission", "model_affinity", "continuous_join",
            "event_replanning",
        ):
            if type(getattr(self, name)) is not bool:
                raise RuntimeDispatchPolicyError(
                    "dispatch policy " + name + " must be a boolean"
                )
        for name in (
            "affinity_maximum_bypasses",
            "affinity_maximum_wait_us",
            "max_barrier_extension_s",
            "residency_hysteresis_s",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise RuntimeDispatchPolicyError(
                    "dispatch policy " + name + " must be a nonnegative integer"
                )
        for name in ("model_affinity", "continuous_join"):
            if getattr(self, name) and not self.work_conserving_admission:
                raise RuntimeDispatchPolicyError(
                    "dispatch policy " + name + " requires "
                    "work_conserving_admission"
                )
        if self.residency_hysteresis_s and not self.work_conserving_admission:
            raise RuntimeDispatchPolicyError(
                "dispatch policy residency_hysteresis_s requires "
                "work_conserving_admission"
            )
        self._validate_hysteresis_probability()

    def _validate_hysteresis_probability(self) -> None:
        """The arrival threshold is a probability in ppm, set only with a window."""
        value = self.residency_hysteresis_min_probability_ppm
        if type(value) is not int or not 0 <= value <= 1_000_000:
            raise RuntimeDispatchPolicyError(
                "dispatch policy residency_hysteresis_min_probability_ppm "
                "must be an integer in [0, 1000000]"
            )
        if (
            value != DEFAULT_RESIDENCY_HYSTERESIS_MIN_PROBABILITY_PPM
            and not self.residency_hysteresis_s
        ):
            raise RuntimeDispatchPolicyError(
                "dispatch policy residency_hysteresis_min_probability_ppm "
                "requires residency_hysteresis_s"
            )

    @property
    def enabled(self) -> bool:
        return (
            self.work_conserving_admission
            or self.model_affinity
            or self.continuous_join
        )

    @property
    def precedence_enabled(self) -> bool:
        """Whether an admission may be ordered ahead of queued followers."""
        return self.model_affinity or self.continuous_join

    @property
    def residency_hysteresis_us(self) -> int:
        return self.residency_hysteresis_s * 1_000_000

    def to_json(self) -> dict[str, object]:
        return {
            **{
                row.name: getattr(self, row.name) for row in fields(self)
                if row.name not in _OPTIONAL_FIELDS
                or getattr(self, row.name) != row.default
            },
            "schema": RUNTIME_DISPATCH_POLICY_SCHEMA,
        }

    @classmethod
    def from_json(cls, value: object) -> "RuntimeDispatchPolicy":
        if not isinstance(value, Mapping):
            raise RuntimeDispatchPolicyError("dispatch policy must be an object")
        names = {row.name for row in fields(cls)}
        rows = {key: item for key, item in value.items() if key != "schema"}
        if value.get("schema", RUNTIME_DISPATCH_POLICY_SCHEMA) != (
            RUNTIME_DISPATCH_POLICY_SCHEMA
        ):
            raise RuntimeDispatchPolicyError("dispatch policy schema differs")
        unknown = sorted(set(rows) - names)
        if unknown:
            raise RuntimeDispatchPolicyError(
                "dispatch policy has unknown fields: " + ", ".join(unknown)
            )
        return cls(**rows)


DEFAULT_RUNTIME_DISPATCH_POLICY = RuntimeDispatchPolicy()


def plan_changes_residency(
    transitions: Sequence[object], exclusive_by_device: Mapping[object, str]
) -> bool:
    """Whether a plan prepares an exclusive residency device.

    Any such transition (a load, an eviction, or the publication of an
    executor whose model is only projected to be resident) may replace the
    resident model when it runs, so only plans without one keep residency.
    """
    return any(
        device_id in exclusive_by_device
        for transition in transitions
        for device_id in transition.prepares_device_ids
    )
