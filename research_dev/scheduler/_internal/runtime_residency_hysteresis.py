"""Residency hysteresis decisions: hold a residency change only when the wait can pay.

The dispatch queue asks this module, once per queued residency change and
release window, whether the change should wait for the released model:

- a change on another residency resource than the released one is never held;
- a change whose own model has queued work that already waited longer than
  the window is never held (fairness);
- a change is held while a same-model request is queued and can still run
  before it (it is not waiting on the change), until that request is
  dispatched or the window ends;
- otherwise it is held only when the same-model inter-arrival gaps learned
  online from admissions (an EWMA, no trace look-ahead) predict an arrival
  inside the window with at least the configured probability, modelling the
  arrivals as memoryless with that mean gap.

A threshold of 0 keeps the unconditional speculative hold.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


REASON_SAME_MODEL_QUEUED = "SAME_MODEL_QUEUED"
REASON_ARRIVAL_LIKELY = "SAME_MODEL_ARRIVAL_LIKELY"
REASON_SPECULATIVE = "SPECULATIVE"
REASON_OTHER_RESOURCE = "OTHER_RESIDENCY_RESOURCE"
REASON_OWN_MODEL_WAITED = "OWN_MODEL_WAITED_LONGER_THAN_HYSTERESIS"
REASON_HISTORY_INSUFFICIENT = "ARRIVAL_HISTORY_INSUFFICIENT"
REASON_ARRIVAL_UNLIKELY = "ARRIVAL_UNLIKELY"
HOLD_REASONS = frozenset({
    REASON_SAME_MODEL_QUEUED, REASON_ARRIVAL_LIKELY, REASON_SPECULATIVE,
})
SKIP_REASONS = frozenset({
    REASON_OTHER_RESOURCE, REASON_OWN_MODEL_WAITED,
    REASON_HISTORY_INSUFFICIENT, REASON_ARRIVAL_UNLIKELY,
})
# Gaps needed before a prediction; the EWMA keeps 3/4 of the old mean.
MINIMUM_ARRIVAL_GAPS = 2
_EWMA_KEEP = 3
_EWMA_DENOMINATOR = 4
DECISION_LOG_LIMIT = 512


class ResidencyHysteresisError(ValueError):
    pass


def _nonnegative(value: object) -> bool:
    return type(value) is int and value >= 0


def _key(value: object) -> bool:
    return type(value) is str and bool(value)


@dataclass(frozen=True)
class ResidencyArrivalHistory:
    """Admissions of one model: last admission, EWMA gap (None before a gap), gap count."""

    last_admitted_at_us: int
    mean_gap_us: int | None = None
    gap_count: int = 0

    def __post_init__(self) -> None:
        if (
            not _nonnegative(self.last_admitted_at_us)
            or not _nonnegative(self.gap_count)
            or (self.mean_gap_us is None) != (self.gap_count == 0)
            or self.mean_gap_us is not None and not _nonnegative(self.mean_gap_us)
        ):
            raise ResidencyHysteresisError("residency arrival history is invalid")

    def admitted(self, admitted_at_us: int) -> "ResidencyArrivalHistory":
        """This history after one more admission of the model."""
        if not _nonnegative(admitted_at_us):
            raise ResidencyHysteresisError("residency arrival time is invalid")
        gap_us = max(0, admitted_at_us - self.last_admitted_at_us)
        mean_us = gap_us if self.mean_gap_us is None else (
            (_EWMA_KEEP * self.mean_gap_us + gap_us) // _EWMA_DENOMINATOR
        )
        return ResidencyArrivalHistory(
            max(admitted_at_us, self.last_admitted_at_us), mean_us, self.gap_count + 1,
        )

    def arrival_probability_ppm(self, window_us: int) -> int | None:
        """P(an arrival within ``window_us``) for memoryless arrivals, or None without history."""
        if self.gap_count < MINIMUM_ARRIVAL_GAPS or self.mean_gap_us is None:
            return None
        if self.mean_gap_us == 0:
            return 1_000_000
        return int(round(-math.expm1(-window_us / self.mean_gap_us) * 1_000_000))


@dataclass(frozen=True)
class ResidencyHysteresisDecision:
    """Whether one queued residency change is held in one release window, and why."""

    request_id: str
    key: str
    released_at_us: int
    released_key: str
    held_until_us: int | None
    reason: str
    arrival_probability_ppm: int | None = None

    def __post_init__(self) -> None:
        probability = self.arrival_probability_ppm
        if (
            not _key(self.request_id) or not _key(self.key)
            or not _key(self.released_key) or self.key == self.released_key
            or not _nonnegative(self.released_at_us)
            or self.reason not in HOLD_REASONS | SKIP_REASONS
            or (self.held_until_us is None) != (self.reason in SKIP_REASONS)
            or self.held_until_us is not None and (
                type(self.held_until_us) is not int
                or self.held_until_us < self.released_at_us
            )
            or probability is not None and (
                type(probability) is not int or not 0 <= probability <= 1_000_000
            )
        ):
            raise ResidencyHysteresisError("residency hysteresis decision is invalid")

    @property
    def held(self) -> bool:
        return self.held_until_us is not None

    def to_json(self) -> dict[str, object]:
        return {
            "barrier_request_id": self.request_id,
            "held": self.held,
            "reason": self.reason,
            "released_at_us": self.released_at_us,
            **({} if self.held_until_us is None else {"held_until_us": self.held_until_us}),
            **({} if self.arrival_probability_ppm is None
               else {"arrival_probability_ppm": self.arrival_probability_ppm}),
        }


def decide_residency_hysteresis(
    *,
    request_id: str,
    key: str,
    release: tuple[int, str, tuple[str, ...]],
    change_resources: frozenset[str],
    own_model_waited_us: int,
    same_model_pending: bool,
    arrivals: ResidencyArrivalHistory | None,
    window_us: int,
    minimum_probability_ppm: int,
) -> ResidencyHysteresisDecision:
    """Decide one hold from the observed queue state at the release (see the module doc)."""
    released_at_us, released_key, released_resources = release

    def decided(reason: str, probability: int | None = None) -> ResidencyHysteresisDecision:
        return ResidencyHysteresisDecision(
            request_id, key, released_at_us, released_key,
            None if reason in SKIP_REASONS else released_at_us + window_us,
            reason, probability,
        )

    if not change_resources.intersection(released_resources):
        return decided(REASON_OTHER_RESOURCE)
    if own_model_waited_us > window_us:
        return decided(REASON_OWN_MODEL_WAITED)
    if same_model_pending:
        return decided(REASON_SAME_MODEL_QUEUED)
    probability = None if arrivals is None else arrivals.arrival_probability_ppm(window_us)
    if minimum_probability_ppm == 0:
        return decided(REASON_SPECULATIVE, probability)
    if probability is None:
        return decided(REASON_HISTORY_INSUFFICIENT)
    if probability < minimum_probability_ppm:
        return decided(REASON_ARRIVAL_UNLIKELY, probability)
    return decided(REASON_ARRIVAL_LIKELY, probability)


def validated_release(value: object) -> tuple[int, str, tuple[str, ...]] | None:
    """A checkpointed release (completed_at_us, key, sorted lane resource ids), or None."""
    if value is None:
        return None
    if (
        type(value) is not tuple or len(value) != 3
        or not _nonnegative(value[0]) or not _key(value[1])
        or type(value[2]) is not tuple
        or any(not _key(row) for row in value[2])
        or tuple(sorted(set(value[2]))) != value[2]
    ):
        raise ResidencyHysteresisError("residency release is invalid")
    return value


def validated_arrivals(value: object) -> dict[str, ResidencyArrivalHistory]:
    """Checkpointed per-model arrival histories, as ((key, history), ...)."""
    if type(value) is not tuple:
        raise ResidencyHysteresisError("residency arrival histories are invalid")
    result: dict[str, ResidencyArrivalHistory] = {}
    for row in value:
        if (
            type(row) is not tuple or len(row) != 2 or not _key(row[0])
            or row[0] in result or not isinstance(row[1], ResidencyArrivalHistory)
        ):
            raise ResidencyHysteresisError("residency arrival histories are invalid")
        result[row[0]] = row[1]
    return result


def validated_decisions(value: object) -> tuple[ResidencyHysteresisDecision, ...]:
    """Checkpointed hysteresis decisions (the decision objects validate themselves)."""
    if type(value) is not tuple or len(value) > DECISION_LOG_LIMIT or any(
        not isinstance(row, ResidencyHysteresisDecision) for row in value
    ):
        raise ResidencyHysteresisError("residency hysteresis decisions are invalid")
    return value


def validated_current_decisions(value: object) -> dict[str, ResidencyHysteresisDecision]:
    """Checkpointed current decision per request, as a tuple of decisions."""
    if type(value) is not tuple:
        raise ResidencyHysteresisError("residency hysteresis decisions are invalid")
    result: dict[str, ResidencyHysteresisDecision] = {}
    for row in value:
        if not isinstance(row, ResidencyHysteresisDecision) or row.request_id in result:
            raise ResidencyHysteresisError("residency hysteresis decisions are invalid")
        result[row.request_id] = row
    return result
