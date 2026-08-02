#!/usr/bin/env python3
"""Physical R0/R2 topology adapter built from the existing S24 route runtime."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
S24 = HERE.parent / "s24_overlap_handoff_poc"
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
import sys
for dependency in (S24, S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from async_pipeline import parse_endpoint
from route_runtime import (
    FiniteRoute,
    ResidentStage,
    RouteDeviceBatcher,
    RouteOutcome,
    RouteRunner,
    StageEndpoint,
    validate_stage_hello,
)
from runtime_support import SequenceSlotPool, SerializedStageClient
from stage_v3_client import ProtocolError, StageV3Client
from workloads import validate as validate_trace


WORKER_NAMES = (
    "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
)


class PhysicalRuntimeError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalRuntimeError(f"cannot load trace: {exc}") from exc
    if type(value) is not dict:
        raise PhysicalRuntimeError("trace must be an object")
    validate_trace(value)
    return value


@dataclass
class PhysicalTopology:
    raw_clients: dict[str, StageV3Client]
    clients: dict[str, SerializedStageClient]
    hellos: dict[str, Any]
    slots: dict[str, SequenceSlotPool]
    batchers: dict[str, RouteDeviceBatcher]
    routes: tuple[FiniteRoute, ...]
    runner: RouteRunner
    ended: bool = False
    batchers_stopped: bool = False

    def stop_batchers(self, timeout_s: float) -> None:
        if self.batchers_stopped:
            return
        errors = []
        for name, batcher in self.batchers.items():
            try:
                batcher.stop(timeout_s)
            except BaseException as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        self.batchers_stopped = True
        if errors:
            raise PhysicalRuntimeError("batcher stop failed: " + "; ".join(errors))

    def end_sessions(self, session_end: str) -> dict[str, Any]:
        if self.ended:
            raise PhysicalRuntimeError("StageNet sessions already ended")
        final = {}
        for name in WORKER_NAMES:
            status = self.clients[name].status()
            if status.active_sequences != 0:
                raise ProtocolError(f"{name} has live worker KV before drain")
            drained = self.clients[name].drain()
            if drained.active_sequences != 0 or not drained.draining:
                raise ProtocolError(f"{name} drain certificate failed")
            final[name] = {
                "before_drain": asdict(status),
                "after_drain": asdict(drained),
            }
        for name in WORKER_NAMES:
            if session_end == "detach":
                self.clients[name].detach()
            else:
                self.clients[name].stop()
        self.ended = True
        return final

    def close(self) -> None:
        for client in self.clients.values():
            try:
                client.close()
            except BaseException:
                pass


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
            if name == "cuda-tail":
                continue
            batchers[name] = RouteDeviceBatcher(
                name,
                clients[name],
                knees[name],
                5000,
                args.queue_depth,
            )
        batchers["cuda-tail-r0"] = RouteDeviceBatcher(
            "cuda-tail", clients["cuda-tail"], 4, 5000, args.queue_depth,
        )
        batchers["cuda-tail-r2"] = RouteDeviceBatcher(
            "cuda-tail", clients["cuda-tail"], 4, 5000, args.queue_depth,
        )

        def stage(
            worker: str,
            start: int,
            end: int,
            terminal: bool,
            batcher_name: str | None = None,
        ) -> ResidentStage:
            resident = ResidentStage(
                worker_name=worker,
                layer_start=start,
                layer_end=end,
                endpoint=StageEndpoint(*endpoint_values[worker]),
                client=clients[worker],
                slots=slots[worker],
                batcher=batchers[batcher_name or worker],
                terminal=terminal,
            )
            validate_stage_hello(resident, hellos[worker])
            return resident

        cuda_prefix = stage("cuda-prefix", 0, 8, False)
        cuda_mid = stage("cuda-mid", 8, 16, False)
        op12_prefix = stage("op12-prefix", 0, 8, False)
        op15_mid = stage("op15-mid", 8, 16, False)
        cuda_tail_r0 = stage("cuda-tail", 16, 48, True, "cuda-tail-r0")
        cuda_tail_r2 = stage("cuda-tail", 16, 48, True, "cuda-tail-r2")
        routes = (
            FiniteRoute("R0", (cuda_prefix, cuda_mid, cuda_tail_r0)),
            FiniteRoute("R2", (op12_prefix, op15_mid, cuda_tail_r2)),
        )
        if (
            routes[0].stages[-1].client is not routes[1].stages[-1].client
            or routes[0].stages[-1].slots is not routes[1].stages[-1].slots
            or routes[0].stages[-1].batcher is routes[1].stages[-1].batcher
        ):
            raise PhysicalRuntimeError("CUDA tail must share state but isolate priority queues")
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


def physical_capacities(topology: PhysicalTopology) -> dict[str, int]:
    return {name: topology.hellos[name].max_streams for name in WORKER_NAMES}


def outcome_record(
    outcome: RouteOutcome,
    source: dict[str, Any],
    origin_ns: int,
    admitted_ns: int,
) -> dict[str, Any]:
    return {
        "request_id": outcome.request_id,
        "route_epoch": outcome.route_epoch,
        "route_id": outcome.route_id,
        "priority": outcome.priority,
        "prompt_length": outcome.prompt_length,
        "output_steps": len(outcome.output_tokens),
        "output_tokens": list(outcome.output_tokens),
        "scheduled_arrival_ns": outcome.scheduled_arrival_ns - origin_ns,
        "admitted_ns": admitted_ns - origin_ns,
        "call_ns": outcome.call_ns - origin_ns,
        "lease_ready_ns": outcome.lease_ready_ns - origin_ns,
        "first_token_ns": outcome.first_token_ns - origin_ns,
        "completed_ns": outcome.completed_ns - origin_ns,
        "admission_queue_us": (admitted_ns - outcome.scheduled_arrival_ns) // 1000,
        "lease_queue_us": outcome.lease_queue_us,
        "ttft_us": outcome.ttft_us,
        "latency_us": outcome.latency_us,
        "slo_us": outcome.slo_us,
        "slo_met": outcome.slo_met,
        "observed_input_tokens": source["observed_input_tokens"],
        "observed_output_tokens": source["observed_output_tokens"],
    }


def _nearest_rank(values: Sequence[int], numerator: int, denominator: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (numerator * len(ordered) + denominator - 1) // denominator
    return ordered[max(0, rank - 1)]


def _summary(values: Sequence[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": _nearest_rank(values, 95, 100),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def _batch_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(event["batch_size"]) for event in events]
    compute = [
        int(event["compute_us"]) for event in events if event.get("status") == "OK"
    ]
    return {
        "batch_count": len(sizes),
        "batch_sizes": sizes,
        "mean_batch": statistics.fmean(sizes) if sizes else 0.0,
        "max_batch": max(sizes, default=0),
        "compute_us": _summary(compute),
        "summed_compute_us": sum(compute),
        "dispatch_reasons": dict(sorted(Counter(
            str(event["dispatch_reason"]) for event in events
        ).items())),
        "failed_batches": sum(event.get("status") != "OK" for event in events),
    }


def summarize_run(
    runtime: dict[str, Any], batch_events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    requests = runtime["requests"]
    priority = {}
    for value in sorted({row["priority"] for row in requests}):
        rows = [row for row in requests if row["priority"] == value]
        priority[str(value)] = {
            "completed": len(rows),
            "slo_misses": sum(not row["slo_met"] for row in rows),
            "ttft_us": _summary([row["ttft_us"] for row in rows]),
            "latency_us": _summary([row["latency_us"] for row in rows]),
            "admission_queue_us": _summary([row["admission_queue_us"] for row in rows]),
            "lease_queue_us": _summary([row["lease_queue_us"] for row in rows]),
        }
    physical_events = {name: [] for name in WORKER_NAMES}
    for name, events in batch_events.items():
        physical = "cuda-tail" if name.startswith("cuda-tail-") else name
        if physical not in physical_events:
            raise PhysicalRuntimeError(f"unknown batch event stream: {name}")
        physical_events[physical].extend(events)
    for events in physical_events.values():
        events.sort(key=lambda event: int(event["dispatch_ns"]))
    workers = {name: _batch_summary(events) for name, events in physical_events.items()}
    cuda_compute = {
        name: workers[name]["summed_compute_us"]
        for name in ("cuda-prefix", "cuda-mid", "cuda-tail")
    }
    return {
        "completed_requests": len(requests),
        "rejected_requests": len(runtime["rejected"]),
        "slo_misses": sum(not row["slo_met"] for row in requests),
        "makespan_us": runtime["duration_ns"] // 1000,
        "route_distribution": dict(sorted(Counter(
            row["route_id"] for row in requests
        ).items())),
        "priority": priority,
        "workers": workers,
        "cuda_island_compute_us": cuda_compute,
        "summed_cuda_island_compute_us": sum(cuda_compute.values()),
        "gpu_board_energy": None,
    }
