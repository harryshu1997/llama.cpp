#!/usr/bin/env python3
"""Build the four-worker physical topology for S36."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
for dependency in (S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from runtime_support import SequenceSlotPool  # noqa: E402
from stage_v3_client import (  # noqa: E402
    Hello,
    ProtocolError,
    STAGE_V3_CAP_RANGE,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)

from cut_batcher import CutBatcher, LockedStageClient  # noqa: E402
from dynamic_route_runtime import (  # noqa: E402
    DynamicRoute,
    DynamicRouteRunner,
    DynamicStage,
)


WORKERS = ("cuda", "op12", "op15", "tail")
EXPECTED_RESIDENCY = {
    "cuda": (0, 4),
    "op12": (0, 8),
    "op15": (0, 8),
    "tail": (4, 48),
}


class TopologyError(RuntimeError):
    pass


@dataclass
class PhysicalTopology:
    raw_clients: dict[str, StageV3Client]
    clients: dict[str, LockedStageClient]
    hellos: dict[str, Hello]
    stages: dict[str, DynamicStage]
    batchers: dict[str, CutBatcher]
    routes: tuple[DynamicRoute, ...]
    runner: DynamicRouteRunner
    batchers_stopped: bool = False
    session_ended: bool = False

    def stop_batchers(self, timeout_s: float) -> None:
        if self.batchers_stopped:
            return
        errors = []
        for name in WORKERS:
            try:
                self.batchers[name].stop(timeout_s)
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self.batchers_stopped = True
        if errors:
            raise TopologyError("batcher stop failed: " + "; ".join(errors))

    def end_sessions(self, session_end: str) -> dict[str, object]:
        if not self.batchers_stopped:
            raise TopologyError("batchers must stop before worker sessions")
        if self.session_ended:
            raise TopologyError("worker sessions already ended")
        if session_end not in ("detach", "stop"):
            raise ValueError("session end must be detach or stop")
        final: dict[str, object] = {}
        for name in WORKERS:
            status = self.clients[name].status()
            if status.active_sequences != 0:
                raise ProtocolError(f"{name} retained live worker KV")
            drained = self.clients[name].drain()
            if drained.active_sequences != 0 or not drained.draining:
                raise ProtocolError(f"{name} drain failed")
            final[name] = {
                "before_drain": asdict(status),
                "after_drain": asdict(drained),
            }
        for name in WORKERS:
            if session_end == "detach":
                self.clients[name].detach()
            else:
                self.clients[name].stop()
        self.session_ended = True
        return final

    def close(self) -> None:
        for client in self.clients.values():
            try:
                client.close()
            except BaseException:
                pass


def _validate_hellos(hellos: Mapping[str, Hello]) -> None:
    if set(hellos) != set(WORKERS):
        raise TopologyError("physical topology is incomplete")
    dimensions = {(hello.n_layer, hello.n_embd) for hello in hellos.values()}
    if dimensions != {(48, 3840)}:
        raise TopologyError("workers do not expose the Gemma 4 12B dimensions")
    file_types = {hello.file_type for hello in hellos.values()}
    if len(file_types) != 1 or None in file_types:
        raise TopologyError("worker GGUF file types differ")
    for name, hello in hellos.items():
        if (hello.layer_start, hello.layer_end) != EXPECTED_RESIDENCY[name]:
            raise TopologyError(f"{name} resident layer range differs from S36")
        if not hello.capabilities & STAGE_V3_CAP_RANGE:
            raise TopologyError(f"{name} omitted dynamic-cut capability")
        terminal = bool(hello.capabilities & STAGE_V3_CAP_TERMINAL)
        if terminal != (name == "tail"):
            raise TopologyError(f"{name} terminal capability differs from S36")


def build_topology(
    endpoints: Mapping[str, tuple[str, int]],
    timeout_s: float,
    gather_us: int,
    queue_depth: int,
    phone_knee: int,
    cuda_knee: int,
    tail_knee: int,
) -> PhysicalTopology:
    if set(endpoints) != set(WORKERS):
        raise ValueError("exactly four worker endpoints are required")
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
                raise TopologyError(f"{name} returned an invalid hello")
            hellos[name] = hello
        _validate_hellos(hellos)

        ranges = {
            "cuda": {4: (0, 4)},
            "op12": {4: (0, 4), 8: (0, 8)},
            "op15": {4: (0, 4), 8: (0, 8)},
            "tail": {4: (4, 48), 8: (8, 48)},
        }
        requested_knees = {
            "cuda": cuda_knee,
            "op12": phone_knee,
            "op15": phone_knee,
            "tail": tail_knee,
        }
        for name in WORKERS:
            capacity = min(
                hellos[name].n_batch,
                hellos[name].n_ubatch,
                64,
            )
            knee = min(requested_knees[name], capacity)
            batchers[name] = CutBatcher(
                name,
                clients[name],
                ranges[name],
                {cut: knee for cut in ranges[name]},
                capacity,
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
        routes = (
            DynamicRoute("cuda-c4", stages["cuda"], stages["tail"], 4),
            DynamicRoute("op12-c4", stages["op12"], stages["tail"], 4),
            DynamicRoute("op12-c8", stages["op12"], stages["tail"], 8),
            DynamicRoute("op15-c4", stages["op15"], stages["tail"], 4),
            DynamicRoute("op15-c8", stages["op15"], stages["tail"], 8),
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

