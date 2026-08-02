#!/usr/bin/env python3
"""Physical topology exposing every jointly resident handoff cut."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping


HERE = Path(__file__).resolve().parent
S36 = HERE.parent / "s36_dynamic_cut_scheduler"
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
for dependency in (S36, S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from runtime_support import SequenceSlotPool  # noqa: E402
from stage_v3_client import Hello, StageV3Client  # noqa: E402

from cut_batcher import CutBatcher, LockedStageClient  # noqa: E402
from dynamic_route_runtime import (  # noqa: E402
    DynamicRoute,
    DynamicRouteRunner,
    DynamicStage,
)
from physical_topology import (  # noqa: E402
    WORKERS,
    PhysicalTopology,
    _validate_hellos,
)


CUTS = (4, 5, 6, 7, 8)
PHONE_WORKERS = ("op12", "op15")


def route_specs() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (f"{phone}-c{cut}", phone, cut)
        for phone in PHONE_WORKERS
        for cut in CUTS
    )


def build_arbitrary_topology(
    endpoints: Mapping[str, tuple[str, int]],
    timeout_s: float,
    gather_us: int,
    queue_depth: int,
    knee: int,
) -> PhysicalTopology:
    if set(endpoints) != set(WORKERS):
        raise ValueError("exactly four worker endpoints are required")
    if type(knee) is not int or knee <= 0:
        raise ValueError("batch knee must be positive")

    raw_clients: dict[str, StageV3Client] = {}
    clients: dict[str, LockedStageClient] = {}
    hellos: dict[str, Hello] = {}
    batchers: dict[str, CutBatcher] = {}
    try:
        for name in WORKERS:
            raw = StageV3Client.connect(*endpoints[name], timeout_s)
            raw_clients[name] = raw
            clients[name] = LockedStageClient(raw)
            hello = clients[name].hello()
            if not isinstance(hello, Hello):
                raise RuntimeError(f"{name} returned an invalid hello")
            hellos[name] = hello
        _validate_hellos(hellos)

        ranges = {
            "cuda": {4: (0, 4)},
            "op12": {cut: (0, cut) for cut in CUTS},
            "op15": {cut: (0, cut) for cut in CUTS},
            "tail": {cut: (cut, 48) for cut in CUTS},
        }
        for name in WORKERS:
            max_rows = min(hellos[name].n_batch, hellos[name].n_ubatch, 64)
            effective_knee = min(knee, max_rows)
            batchers[name] = CutBatcher(
                name,
                clients[name],
                ranges[name],
                {cut: effective_knee for cut in ranges[name]},
                max_rows,
                gather_us,
                queue_depth,
            )

        stages = {
            name: DynamicStage(
                name,
                clients[name],
                hellos[name],
                SequenceSlotPool(hellos[name].max_streams),
                batchers[name],
                name == "tail",
            )
            for name in WORKERS
        }
        routes = tuple(
            DynamicRoute(route_id, stages[phone], stages["tail"], cut)
            for route_id, phone, cut in route_specs()
        )
        return PhysicalTopology(
            raw_clients=raw_clients,
            clients=clients,
            hellos=hellos,
            stages=stages,
            batchers=batchers,
            routes=routes,
            runner=DynamicRouteRunner(routes),
        )
    except BaseException:
        for batcher in batchers.values():
            try:
                batcher.stop(timeout_s)
            except BaseException:
                pass
        for client in clients.values():
            try:
                client.close()
            except BaseException:
                pass
        raise
