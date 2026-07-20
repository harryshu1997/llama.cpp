#!/usr/bin/env python3
"""Build S15 route snapshots from the frozen S14 measured profiles.

The OP15 [0,8) and OP12 [0,6) routes are the only measured, dispatch-eligible
routes; they are loaded through the existing S14 adapters. The BGE server route
here is a structural mechanics placeholder for the selected-A6000 lane and is
NOT a measured BGE latency profile. The compound routes exist only so the plane
can prove it refuses to dispatch them.
"""

from __future__ import annotations

import sys
from pathlib import Path

_S14 = Path(__file__).resolve().parent.parent / "s14_mixed_streaming_scheduler"
if str(_S14) not in sys.path:
    sys.path.insert(0, str(_S14))

from live_profile_adapter import load_op15_head_route  # noqa: E402
from op12_profile_adapter import load_op12_head_route  # noqa: E402
from power_frontier_policy import CertifiedBatchPoint  # noqa: E402
from priority_batch_runtime import RouteConfig  # noqa: E402

from route_registry import RouteSnapshot, route_content_digest  # noqa: E402
from batch32_profile_adapter import with_b32_profile  # noqa: E402
from post_load_profile_adapter import with_post_load_b32_profile  # noqa: E402
from persistent_b32_profile_adapter import with_persistent_b32_profile  # noqa: E402


ENERGY = _S14 / "energy"
OP15_DEVICE = "op15:3C15AU002CL00000"
OP12_DEVICE = "op12:5ae7a43d"
GPU_DEVICE = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"


def op15_b1_paths() -> list[Path]:
    return [ENERGY / f"stageb_op15_k8_b1_r{repeat}.json" for repeat in range(7)]


def op12_b1_paths() -> list[Path]:
    return [ENERGY / f"op12_k6_b1_v2_r{repeat}.json" for repeat in range(7)]


def _ready(config: RouteConfig, device_id: str, layer_range: tuple[int, int],
           *, residency_epoch: int = 1, lease_epoch: int = 1,
           device_boot_epoch: int = 1, credits: int = 1,
           thermal_ceiling: int = 95_000, thermal_observed: int = 60_000) -> RouteSnapshot:
    return RouteSnapshot(
        config=config,
        content_digest=route_content_digest(config),
        device_id=device_id,
        layer_range=layer_range,
        state="READY",
        residency_epoch=residency_epoch,
        lease_epoch=lease_epoch,
        device_boot_epoch=device_boot_epoch,
        thermal_ceiling_millic=thermal_ceiling,
        thermal_observed_millic=thermal_observed,
        execution_credits=credits,
        compound_kind="SINGLE",
        correctness_certified=True,
    )


def op15_head_snapshot(route_epoch: int = 11, **kwargs) -> RouteSnapshot:
    config = load_op15_head_route(op15_b1_paths(), route_epoch)
    return _ready(config, OP15_DEVICE, (0, 8), **kwargs)


def op12_head_snapshot(route_epoch: int = 11, **kwargs) -> RouteSnapshot:
    config = load_op12_head_route(op12_b1_paths(), route_epoch)
    return _ready(config, OP12_DEVICE, (0, 6), **kwargs)


def op15_b32_snapshot(route_epoch: int = 12, **kwargs) -> RouteSnapshot:
    config = with_b32_profile(load_op15_head_route(op15_b1_paths(), route_epoch), "op15")
    return _ready(config, OP15_DEVICE, (0, 8), **kwargs)


def op15_post_load_b32_snapshot(route_epoch: int = 13, **kwargs) -> RouteSnapshot:
    config = with_post_load_b32_profile(load_op15_head_route(op15_b1_paths(), route_epoch))
    return _ready(config, OP15_DEVICE, (0, 8), **kwargs)


def op15_persistent_b32_snapshot(route_epoch: int = 14, **kwargs) -> RouteSnapshot:
    config = with_persistent_b32_profile(load_op15_head_route(op15_b1_paths(), route_epoch))
    return _ready(config, OP15_DEVICE, (0, 8), **kwargs)


def op12_b32_snapshot(route_epoch: int = 12, **kwargs) -> RouteSnapshot:
    config = with_b32_profile(load_op12_head_route(op12_b1_paths(), route_epoch), "op12")
    return _ready(config, OP12_DEVICE, (0, 6), **kwargs)


def synthetic_bge_route(route_epoch: int = 3) -> RouteConfig:
    """Structural mechanics route for the selected-A6000 BGE lane. Not measured."""
    digest = "sha256:" + "b9" * 32
    points = (
        CertifiedBatchPoint(1, 4000, f"{digest}#mechanics-correct", f"{digest}#mechanics-placement"),
        CertifiedBatchPoint(2, 4200, f"{digest}#mechanics-correct", f"{digest}#mechanics-placement"),
        CertifiedBatchPoint(4, 5200, f"{digest}#mechanics-correct", f"{digest}#mechanics-placement"),
    )
    return RouteConfig(
        route_id="server-bge-encoder",
        service_class="embedding",
        model_id="bge-small-en-v1.5-f16",
        island_id="bge-encoder-0-12",
        profile_id=digest,
        route_epoch=route_epoch,
        roofline_class="compute_bound",
        points=points,
        high_priority_max=0,
    )


def synthetic_bge_snapshot(route_epoch: int = 3, **kwargs) -> RouteSnapshot:
    config = synthetic_bge_route(route_epoch)
    return _ready(config, GPU_DEVICE, None, **kwargs)


def _compound_config(route_id: str, route_epoch: int = 5) -> RouteConfig:
    base = load_op15_head_route(op15_b1_paths(), route_epoch)
    return RouteConfig(
        route_id=route_id,
        service_class=base.service_class,
        model_id=base.model_id,
        island_id="gemma-head-0-12",
        profile_id=base.profile_id,
        route_epoch=route_epoch,
        roofline_class=base.roofline_class,
        points=base.points,
        high_priority_max=base.high_priority_max,
    )


def shared_tail_snapshot(state: str = "UNAVAILABLE", route_epoch: int = 5) -> RouteSnapshot:
    config = _compound_config("twophone-sharedtail-0-12", route_epoch)
    return RouteSnapshot(
        config=config,
        content_digest=route_content_digest(config),
        device_id=OP15_DEVICE,
        layer_range=(0, 12),
        state=state,
        residency_epoch=1,
        lease_epoch=1,
        device_boot_epoch=1,
        thermal_ceiling_millic=95_000,
        thermal_observed_millic=60_000,
        execution_credits=1,
        compound_kind="SHARED_TAIL_PARALLEL_HEAD",
        correctness_certified=False,
    )


def serial_chain_snapshot(state: str = "UNAVAILABLE", route_epoch: int = 5) -> RouteSnapshot:
    config = _compound_config("op15-then-op12-serial-0-12", route_epoch)
    return RouteSnapshot(
        config=config,
        content_digest=route_content_digest(config),
        device_id=OP12_DEVICE,
        layer_range=(0, 12),
        state=state,
        residency_epoch=1,
        lease_epoch=1,
        device_boot_epoch=1,
        thermal_ceiling_millic=95_000,
        thermal_observed_millic=60_000,
        execution_credits=1,
        compound_kind="SERIAL_CHAIN",
        correctness_certified=False,
    )
