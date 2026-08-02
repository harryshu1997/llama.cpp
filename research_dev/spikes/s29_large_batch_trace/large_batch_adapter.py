#!/usr/bin/env python3
"""Build the S29 R0/R2 topology with large device-local batch queues."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S24 = HERE.parent / "s24_overlap_handoff_poc"
S26 = HERE.parent / "s26_priority_scheduler"
S28 = HERE.parent / "s28_priority_shared_tail"
for dependency in (S28, S26, S24, S22):
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


DEFAULT_KNEES = {
    "cuda-prefix": 32,
    "cuda-mid": 32,
    "op12-prefix": 32,
    "op15-mid": 32,
    "cuda-tail": 32,
}

EXPECTED_ROUTES = {
    "R0": (
        ("cuda-prefix", 0, 6),
        ("cuda-mid", 6, 8),
        ("cuda-tail", 8, 48),
    ),
    "R2": (
        ("op12-prefix", 0, 6),
        ("op15-mid", 6, 8),
        ("cuda-tail", 8, 48),
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
            raise PhysicalRuntimeError(f"{route_id} differs from the S29 route")
    r0_tail = by_id["R0"].stages[-1]
    r2_tail = by_id["R2"].stages[-1]
    if (
        r0_tail.client is not r2_tail.client
        or r0_tail.slots is not r2_tail.slots
        or r0_tail.batcher is not r2_tail.batcher
    ):
        raise PhysicalRuntimeError("R0 and R2 must share one CUDA-tail queue")


def _knees(args: Any) -> dict[str, int]:
    values = {
        name: int(getattr(args, name.replace("-", "_") + "_knee", default))
        for name, default in DEFAULT_KNEES.items()
    }
    if any(value <= 0 for value in values.values()):
        raise PhysicalRuntimeError("batch knees must be positive")
    return values


def build_topology(args: Any) -> PhysicalTopology:
    endpoint_values = {
        "cuda-prefix": args.cuda_prefix,
        "cuda-mid": args.cuda_mid,
        "op12-prefix": args.op12,
        "op15-mid": args.op15,
        "cuda-tail": args.cuda_tail,
    }
    knees = _knees(args)
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
            if knees[name] > hellos[name].max_streams:
                raise PhysicalRuntimeError(
                    f"{name} knee exceeds its stream capacity"
                )
        for name in WORKER_NAMES:
            batchers[name] = RouteDeviceBatcher(
                name,
                clients[name],
                knees[name],
                int(getattr(args, "gather_us", 5000)),
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

        cuda_prefix = stage("cuda-prefix", 0, 6, False)
        cuda_mid = stage("cuda-mid", 6, 8, False)
        op12_prefix = stage("op12-prefix", 0, 6, False)
        op15_mid = stage("op15-mid", 6, 8, False)
        cuda_tail = stage("cuda-tail", 8, 48, True)
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


def knees_from_args(args: Any) -> dict[str, int]:
    return _knees(args)


def expected_routes() -> dict[str, tuple[tuple[str, int, int], ...]]:
    return dict(EXPECTED_ROUTES)
