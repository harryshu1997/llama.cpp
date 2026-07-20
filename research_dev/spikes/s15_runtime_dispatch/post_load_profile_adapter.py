#!/usr/bin/env python3
"""Bind the independently validated post-load OP15 B32 profile."""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
RECERT = HERE.parent / "s15_arrival_faithful_b32"
for path in (S14, RECERT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402
import validate_recertification as recert  # noqa: E402


class PostLoadProfileError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def validated_profile() -> tuple[dict, str]:
    report = recert.validate()
    report_digest = recert.digest(recert.REPORT)
    return report, report_digest


def with_post_load_b32_profile(base: RouteConfig) -> RouteConfig:
    if type(base) is not RouteConfig:
        raise PostLoadProfileError("base must be a RouteConfig")
    base.validate()
    if any(point.batch_size == 32 for point in base.points):
        raise PostLoadProfileError("route already contains a B32 point")
    report, report_digest = validated_profile()
    duration_us = report["completion_conservative_us"]
    if type(duration_us) is not int or duration_us <= 0 or duration_us > 4_000_000:
        raise PostLoadProfileError("post-load B32 duration is not deadline-eligible")
    evidence = report_digest
    point = CertifiedBatchPoint(
        32,
        duration_us,
        evidence + "#post-load-b32-exact-same-batch-cuda",
        evidence + "#post-load-b32-htp-placement",
    )
    profile_payload = "\n".join((
        base.profile_id,
        report["profile_id"],
        evidence,
        f"duration_us={duration_us}",
    )).encode("ascii")
    profile_id = "sha256:" + hashlib.sha256(profile_payload).hexdigest()
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
