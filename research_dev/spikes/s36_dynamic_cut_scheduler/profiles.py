#!/usr/bin/env python3
"""Fail-closed profile bundle contract for S36."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dynamic_cut_policy import PolicyError, RouteProfile


SCHEMA = "s36-dynamic-cut-profiles-v1"
EXPECTED_ROUTES = {
    "cuda-c4": ("cuda", 4),
    "op12-c4": ("op12", 4),
    "op12-c8": ("op12", 8),
    "op15-c4": ("op15", 4),
    "op15-c8": ("op15", 8),
}


class ProfileError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def with_profile_hash(bundle: dict[str, Any]) -> dict[str, Any]:
    if "profile_hash" in bundle:
        raise ProfileError("profile hash already exists")
    result = dict(bundle)
    result["profile_hash"] = "sha256:" + hashlib.sha256(
        canonical_bytes(bundle)
    ).hexdigest()
    return result


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProfileError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_profile_bundle(path: Path) -> tuple[dict[str, Any], tuple[RouteProfile, ...]]:
    try:
        bundle = json.loads(
            path.read_text(encoding="ascii"), object_pairs_hook=_no_duplicates,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot load profile bundle: {exc}") from exc
    profiles = validate_profile_bundle(bundle)
    return bundle, profiles


def validate_profile_bundle(bundle: object) -> tuple[RouteProfile, ...]:
    if type(bundle) is not dict or bundle.get("schema") != SCHEMA:
        raise ProfileError("profile bundle schema mismatch")
    supplied_hash = bundle.get("profile_hash")
    if type(supplied_hash) is not str or not supplied_hash.startswith("sha256:"):
        raise ProfileError("profile bundle hash is missing")
    unhashed = dict(bundle)
    del unhashed["profile_hash"]
    expected_hash = "sha256:" + hashlib.sha256(canonical_bytes(unhashed)).hexdigest()
    if supplied_hash != expected_hash:
        raise ProfileError("profile bundle hash mismatch")
    if bundle.get("physical") is not True:
        raise ProfileError("profile bundle is not physical")
    if bundle.get("model_scope") != "GEMMA4_12B_F16_LOGICAL_MODEL":
        raise ProfileError("profile model scope mismatch")
    rows = bundle.get("routes")
    if type(rows) is not list or len(rows) != len(EXPECTED_ROUTES):
        raise ProfileError("profile route set is incomplete")

    result = []
    seen: set[str] = set()
    for row in rows:
        if type(row) is not dict:
            raise ProfileError("profile route must be an object")
        route_id = row.get("route_id")
        if type(route_id) is not str or route_id in seen or route_id not in EXPECTED_ROUTES:
            raise ProfileError("profile route id is invalid")
        seen.add(route_id)
        device, cut = EXPECTED_ROUTES[route_id]
        if row.get("device") != device or row.get("cut") != cut:
            raise ProfileError("profile route identity mismatch")
        points = row.get("batch_points")
        if type(points) is not list or {point.get("batch_size") for point in points if type(point) is dict} != {8, 32}:
            raise ProfileError("profile route lacks B8/B32 points")
        for point in points:
            if (
                type(point) is not dict
                or type(point.get("batch_size")) is not int
                or type(point.get("repetitions")) is not int
                or point["repetitions"] < 2
                or type(point.get("p95_latency_us")) is not int
                or point["p95_latency_us"] <= 0
                or type(point.get("max_latency_us")) is not int
                or point["max_latency_us"] < point["p95_latency_us"]
                or point.get("token_consistent") is not True
            ):
                raise ProfileError("profile batch point is invalid")
        profile = RouteProfile(
            route_id=route_id,
            device=device,
            cut=cut,
            predicted_p95_us=row.get("predicted_p95_us"),
            safety_margin_us=row.get("safety_margin_us"),
            measured=row.get("measured"),
            eligible=row.get("eligible"),
        )
        try:
            profile.validate()
        except PolicyError as exc:
            raise ProfileError(f"profile route is invalid: {exc}") from exc
        if not profile.measured or not profile.eligible:
            raise ProfileError("S36 runtime requires five eligible measured routes")
        if profile.predicted_p95_us != max(
            point["max_latency_us"] for point in points
        ):
            raise ProfileError("route prediction is not the measured maximum")
        if profile.safety_margin_us != (
            profile.predicted_p95_us * 20 + 99
        ) // 100:
            raise ProfileError("route safety margin is not frozen at 20 percent")
        result.append(profile)
    if seen != set(EXPECTED_ROUTES):
        raise ProfileError("profile route set differs from S36")
    result.sort(key=lambda profile: profile.route_id)
    return tuple(result)
