#!/usr/bin/env python3
"""Bind the independently validated persistent OP15 B32 profile."""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
PERSISTENT = HERE.parent / "s15_persistent_b32"
for path in (S14, PERSISTENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402
import validate_report as persistent  # noqa: E402


class PersistentB32ProfileError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def validated_profile() -> tuple[dict, str]:
    report = persistent.validate()
    return report, persistent.digest(persistent.REPORT)


def with_persistent_b32_profile(base: RouteConfig) -> RouteConfig:
    if type(base) is not RouteConfig:
        raise PersistentB32ProfileError("base must be a RouteConfig")
    base.validate()
    if any(point.batch_size == 32 for point in base.points):
        raise PersistentB32ProfileError("route already contains a B32 point")
    report, report_digest = validated_profile()
    duration_us = max(value["elapsed_us"] for value in report["sessions"])
    if type(duration_us) is not int or not 0 < duration_us <= 4_000_000:
        raise PersistentB32ProfileError("persistent B32 duration is not deadline-eligible")
    point = CertifiedBatchPoint(
        32,
        duration_us,
        report_digest + "#seven-session-exact-same-batch-cuda",
        report_digest + "#seven-session-htp-placement-reset",
    )
    identity = "\n".join((
        base.profile_id,
        report_digest,
        report["artifacts"]["android_worker"],
        report["artifacts"]["host_worker"],
        f"duration_us={duration_us}",
        "session_protocol=ls-stagenet-session-v2",
    )).encode("ascii")
    profile_id = "sha256:" + hashlib.sha256(identity).hexdigest()
    return RouteConfig(
        route_id=base.route_id,
        service_class=base.service_class,
        model_id=base.model_id,
        island_id=base.island_id,
        profile_id=profile_id,
        route_epoch=base.route_epoch,
        roofline_class=base.roofline_class,
        points=tuple(sorted((*base.points, point), key=lambda value: value.batch_size)),
        high_priority_max=base.high_priority_max,
    )
