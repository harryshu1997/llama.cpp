#!/usr/bin/env python3
"""Build the measured R0/R2 topology with one shared CUDA-tail queue."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S24 = HERE.parent / "s24_overlap_handoff_poc"
S26 = HERE.parent / "s26_priority_scheduler"
for dependency in (S26, S24, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from physical_adapter import PhysicalRuntimeError, PhysicalTopology, WORKER_NAMES
from route_runtime import (
    FiniteRoute,
    ResidentStage,
    RouteDeviceBatcher,
    RouteRunner,
    StageEndpoint,
    validate_stage_hello,
)
from runtime_support import SequenceSlotPool, SerializedStageClient
from stage_v3_client import StageV3Client


EXPECTED_ROUTES = {
    "R0": (
        ("cuda-prefix", 0, 8),
        ("cuda-mid", 8, 16),
        ("cuda-tail", 16, 48),
    ),
    "R2": (
        ("op12-prefix", 0, 8),
        ("op15-mid", 8, 16),
        ("cuda-tail", 16, 48),
    ),
}


def validate_shared_pair(routes: tuple[FiniteRoute, ...]) -> None:
    by_id = {route.route_id: route for route in routes}
    if len(by_id) != len(routes) or set(by_id) != set(EXPECTED_ROUTES):
        raise PhysicalRuntimeError("topology must define exactly R0 and R2")
    for route_id, expected in EXPECTED_ROUTES.items():
        actual = tuple(
            (stage.worker_name, stage.layer_start, stage.layer_end)
            for stage in by_id[route_id].stages
        )
        if actual != expected:
            raise PhysicalRuntimeError(f"{route_id} differs from the measured route")
    r0_tail = by_id["R0"].stages[-1]
    r2_tail = by_id["R2"].stages[-1]
    if (
        r0_tail.client is not r2_tail.client
        or r0_tail.slots is not r2_tail.slots
        or r0_tail.batcher is not r2_tail.batcher
    ):
        raise PhysicalRuntimeError("R0 and R2 must share one CUDA-tail queue")


def build_topology(args: Any) -> PhysicalTopology:
    endpoint_values = {
        "cuda-prefix": args.cuda_prefix,
        "cuda-mid": args.cuda_mid,
        "op12-prefix": args.op12,
        "op15-mid": args.op15,
        "cuda-tail": args.cuda_tail,
    }
    knees = {
        "cuda-prefix": 4,
        "cuda-mid": 4,
        "op12-prefix": 4,
        "op15-mid": 4,
        "cuda-tail": 8,
    }
    raw_clients: dict[str, StageV3Client] = {}
    clients: dict[str, SerializedStageClient] = {}
    hellos: dict[str, Any] = {}
    slots: dict[str, SequenceSlotPool] = {}
    batchers: dict[str, RouteDeviceBatcher] = {}
    try:
        for name in WORKER_NAMES:
            raw_clients[name] = StageV3Client.connect(
                *endpoint_values[name], args.timeout,
            )
            clients[name] = SerializedStageClient(raw_clients[name])
            hellos[name] = clients[name].hello()
            slots[name] = SequenceSlotPool(hellos[name].max_streams)
        for name in WORKER_NAMES:
            batchers[name] = RouteDeviceBatcher(
                name,
                clients[name],
                knees[name],
                5000,
                args.queue_depth,
            )

        def stage(
            worker: str,
            start: int,
            end: int,
            terminal: bool,
        ) -> ResidentStage:
            resident = ResidentStage(
                worker_name=worker,
                layer_start=start,
                layer_end=end,
                endpoint=StageEndpoint(*endpoint_values[worker]),
                client=clients[worker],
                slots=slots[worker],
                batcher=batchers[worker],
                terminal=terminal,
            )
            validate_stage_hello(resident, hellos[worker])
            return resident

        cuda_prefix = stage("cuda-prefix", 0, 8, False)
        cuda_mid = stage("cuda-mid", 8, 16, False)
        op12_prefix = stage("op12-prefix", 0, 8, False)
        op15_mid = stage("op15-mid", 8, 16, False)
        cuda_tail = stage("cuda-tail", 16, 48, True)
        routes = (
            FiniteRoute("R0", (cuda_prefix, cuda_mid, cuda_tail)),
            FiniteRoute("R2", (op12_prefix, op15_mid, cuda_tail)),
        )
        validate_shared_pair(routes)
        return PhysicalTopology(
            raw_clients,
            clients,
            hellos,
            slots,
            batchers,
            routes,
            RouteRunner(routes),
        )
    except BaseException:
        for batcher in batchers.values():
            try:
                batcher.stop(args.timeout)
            except BaseException:
                pass
        for client in clients.values():
            try:
                client.close()
            except BaseException:
                pass
        raise
