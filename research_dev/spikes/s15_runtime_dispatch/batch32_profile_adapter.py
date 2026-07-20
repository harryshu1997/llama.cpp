#!/usr/bin/env python3
"""Bind the independently validated S15 B32 phone profiles to route configs."""

from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
S14 = HERE.parent / "s14_mixed_streaming_scheduler"
B32_GATE = HERE.parent / "s15_batch32_gate"
for path in (S14, B32_GATE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402
import validate_gate as gate  # noqa: E402


class Batch32ProfileError(ValueError):
    pass


@functools.lru_cache(maxsize=1)
def _validated_gate() -> tuple[dict[str, tuple[int, ...]], str]:
    walls = gate.validate_report()
    frozen = {name: tuple(values) for name, values in walls.items()}
    report_digest = gate.sha256(gate.REPORT)
    return frozen, report_digest


def with_b32_profile(base: RouteConfig, device_name: str) -> RouteConfig:
    if type(base) is not RouteConfig:
        raise Batch32ProfileError("base must be a RouteConfig")
    base.validate()
    if device_name not in ("op15", "op12"):
        raise Batch32ProfileError("device_name must be op15 or op12")
    if any(point.batch_size == 32 for point in base.points):
        raise Batch32ProfileError("route already contains a B32 point")
    walls, report_digest = _validated_gate()
    samples = walls[device_name]
    if len(samples) != 7:
        raise Batch32ProfileError("B32 profile does not contain seven processes")
    duration_us = max(samples)
    evidence = "sha256:" + report_digest
    point = CertifiedBatchPoint(
        32,
        duration_us,
        evidence + "#b32-exact-same-batch-cuda",
        evidence + "#b32-htp-placement",
    )
    profile_payload = "\n".join((
        base.profile_id,
        device_name,
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
