"""Validated scheduler profile loading and lifecycle catalogs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .lifecycle import LifecycleProfileSet
from .policy import ProfileBundle, SchedulerError


__all__ = [
    "ProfileCatalogError",
    "canonical_profile_bytes",
    "load_lifecycle_profile_set",
    "load_profile_bundle",
]


class ProfileCatalogError(ValueError):
    pass


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileCatalogError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_profile_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def load_profile_bundle(path: Path) -> tuple[dict[str, Any], ProfileBundle]:
    try:
        with path.open("r", encoding="ascii") as source:
            raw = json.load(source, object_pairs_hook=_no_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileCatalogError(f"cannot read profile: {exc}") from exc
    if type(raw) is not dict:
        raise ProfileCatalogError("profile must contain an object")
    supplied_hash = raw.get("profile_hash")
    if type(supplied_hash) is not str or not supplied_hash.startswith("sha256:"):
        raise ProfileCatalogError("profile hash is missing")
    unhashed = dict(raw)
    del unhashed["profile_hash"]
    expected_hash = "sha256:" + hashlib.sha256(
        canonical_profile_bytes(unhashed)
    ).hexdigest()
    if supplied_hash != expected_hash:
        raise ProfileCatalogError("profile hash mismatch")
    try:
        profile = ProfileBundle.from_json(raw)
    except SchedulerError as exc:
        raise ProfileCatalogError(str(exc)) from exc
    return raw, profile


def load_lifecycle_profile_set(
    profile_set_id: str,
    state_key: str,
    paths: Mapping[str, Path],
    default_state: str,
    reuse_states: frozenset[str] = frozenset(),
) -> tuple[Mapping[str, dict[str, Any]], LifecycleProfileSet]:
    raw: dict[str, dict[str, Any]] = {}
    profiles: dict[str, ProfileBundle] = {}
    for state, path in paths.items():
        raw[state], profiles[state] = load_profile_bundle(path)
    return raw, LifecycleProfileSet(
        profile_set_id=profile_set_id,
        state_key=state_key,
        profiles=profiles,
        default_state=default_state,
        reuse_states=reuse_states,
    )
