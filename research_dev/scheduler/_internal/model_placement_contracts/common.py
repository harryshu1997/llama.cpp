"""Model demand, layout and request-binding records: common."""

from __future__ import annotations

from typing import Sequence


MODEL_PLACEMENT_ACTIONS = frozenset({
    "KEEP_EPOCH",
    "REFRESH_IN_BACKGROUND",
    "RECOMPUTE_NOW",
    "FALLBACK",
})


PHONE_RESIDENCY_LAYOUT_STATES = frozenset({
    "PROPOSED",
    "PREPARING",
    "READY",
    "DRAINING",
})


PHONE_SESSION_RESIDENCY_STATES = frozenset({
    "EMPTY",
    "READY",
    "DRAINING",
    "LOADING",
    "VERIFIED",
    "FAILED",
    "UNAVAILABLE",
})


REQUEST_HELPER_REBIND_STATES = frozenset({"REQUESTED", "QUIESCED"})


class ModelPlacementControllerError(ValueError):
    pass


def _text(name: str, value: object) -> str:
    if type(value) is not str or not value or not value.isascii():
        raise ModelPlacementControllerError(
            f"{name} must be non-empty ASCII text"
        )
    return value


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ModelPlacementControllerError(
            f"{name} must be an integer >= {minimum}"
        )
    return value


def _sha256(name: str, value: object) -> str:
    value = _text(name, value)
    if not value.startswith("sha256:") or len(value) != 71:
        raise ModelPlacementControllerError(f"{name} is invalid")
    return value


def _optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _sha256(name, value)


def _identities(name: str, values: Sequence[str]) -> tuple[str, ...]:
    rows = tuple(sorted(_text(name, value) for value in values))
    if len(rows) != len(set(rows)):
        raise ModelPlacementControllerError(f"{name} values are duplicated")
    return rows


def _material_bucket(value: int) -> int:
    _integer("model demand bucket value", value)
    return 0 if value == 0 else 1 << (value.bit_length() - 1)
