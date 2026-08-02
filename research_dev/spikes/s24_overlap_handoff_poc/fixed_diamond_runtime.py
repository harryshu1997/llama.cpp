#!/usr/bin/env python3
"""Run the physical S24 fixed diamond over resident StageNet V3 workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import statistics
import sys
import threading
import time
from array import array
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
S21 = HERE.parent / "s21_4060ti_server_trace"
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
for dependency in (S21, S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from async_pipeline import parse_endpoint
from fixed_policy import (
    FixedRouteProfile,
    FixedSloRouter,
    NoFeasibleRoute,
    SloWork,
)
from replay import Nvml, NvmlSampler
from route_runtime import (
    FiniteRoute,
    ResidentStage,
    RouteDeviceBatcher,
    RouteOutcome,
    RouteRequest,
    RouteRunner,
    StageEndpoint,
    validate_fixed_routes,
    validate_shared_treatment,
    validate_stage_hello,
)
from runtime_support import SequenceSlotPool, SerializedStageClient
from stage_v3_client import ProtocolError, StageV3Client
from workloads import validate as validate_trace


SCHEMA = "s24-fixed-diamond-physical-v1"
PROFILE_SCHEMA = "s24-fixed-route-profiles-v1"
CONTROL_IDS = ("C0", "C1", "C2", "C3")
ROUTE_IDS = ("R0", "R1", "R2")
WORKER_NAMES = (
    "cuda-prefix", "cuda-mid", "op12-prefix", "op15-mid", "cuda-tail",
)
ROUTE_RESOURCES = {
    "R0": ("cuda-prefix", "cuda-mid", "cuda-tail"),
    "R1": ("cuda-prefix", "op15-mid", "cuda-tail"),
    "R2": ("op12-prefix", "op15-mid", "cuda-tail"),
}


class PhysicalRuntimeError(RuntimeError):
    pass


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0,1]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_values(values: Sequence[int]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": nearest_rank(values, 0.95),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
    }


def summarize_batch_events(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    sizes = [int(event["batch_size"]) for event in events]
    compute = [
        int(event["compute_us"])
        for event in events
        if event.get("status") == "OK"
    ]
    queue_us = [int(event["max_queue_us"]) for event in events]
    reasons = Counter(str(event["dispatch_reason"]) for event in events)
    return {
        "batch_count": len(sizes),
        "batch_sizes": sizes,
        "mean_batch": statistics.fmean(sizes) if sizes else 0.0,
        "max_batch": max(sizes, default=0),
        "compute_us": summarize_values(compute),
        "summed_compute_us": sum(compute),
        "max_queue_us": summarize_values(queue_us),
        "dispatch_reasons": dict(sorted(reasons.items())),
        "mixed_route_batches": sum(
            len(event.get("contributing_routes", [])) > 1
            for event in events
        ),
        "mixed_upstream_batches": sum(
            len(event.get("contributing_upstreams", [])) > 1
            for event in events
        ),
        "failed_batches": sum(event.get("status") != "OK" for event in events),
    }


def integrate_nvml(
    samples: Sequence[dict[str, Any]], duration_ns: int,
) -> dict[str, Any]:
    if duration_ns <= 0:
        raise ValueError("NVML duration must be positive")
    ordered = sorted(samples, key=lambda row: int(row["time_ns"]))
    if len(ordered) < 2:
        raise PhysicalRuntimeError("fewer than two NVML samples")
    times = [int(row["time_ns"]) for row in ordered]
    powers = [float(row["power_w"]) for row in ordered]
    if (
        times[0] < 0
        or any(right <= left for left, right in zip(times, times[1:]))
        or any(not math.isfinite(power) or power < 0 for power in powers)
    ):
        raise PhysicalRuntimeError("invalid NVML sample timeline")
    clipped_end = min(times[-1], duration_ns)
    energy_j = powers[0] * min(times[0], duration_ns) / 1e9
    for index in range(1, len(ordered)):
        left = times[index - 1]
        right = min(times[index], duration_ns)
        if left >= duration_ns:
            break
        energy_j += (
            (powers[index - 1] + powers[index]) * 0.5
            * (right - left) / 1e9
        )
    if clipped_end < duration_ns:
        energy_j += powers[-1] * (duration_ns - clipped_end) / 1e9
    gaps = [right - left for left, right in zip(times, times[1:])]
    return {
        "scope": "RTX_4060_TI_GPU_BOARD_ONLY",
        "method": "NVML_SIDECAR_TRAPEZOID_WITH_CONSTANT_EDGE_FILL",
        "duration_ns": duration_ns,
        "sample_count": len(ordered),
        "first_sample_ns": times[0],
        "last_sample_ns": times[-1],
        "max_sample_gap_ns": max(gaps),
        "mean_power_w": statistics.fmean(powers),
        "min_power_w": min(powers),
        "max_power_w": max(powers),
        "energy_j": energy_j,
    }


def load_trace(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalRuntimeError(f"cannot load trace: {exc}") from exc
    if not isinstance(value, dict):
        raise PhysicalRuntimeError("trace must be an object")
    validate_trace(value)
    return value


def parse_route_delays(values: Sequence[str]) -> dict[str, int]:
    result = {route_id: 0 for route_id in ROUTE_IDS}
    for value in values:
        route_id, separator, delay_text = value.partition(":")
        if not separator or route_id not in ROUTE_IDS:
            raise argparse.ArgumentTypeError("route delay must be R0:US, R1:US, or R2:US")
        try:
            delay = int(delay_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("route delay must be an integer") from exc
        if delay < 0:
            raise argparse.ArgumentTypeError("route delay cannot be negative")
        result[route_id] = delay
    return result


def load_profiles(
    path: Path,
) -> tuple[list[FixedRouteProfile], dict[str, int], list[dict[str, Any]]]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PhysicalRuntimeError(f"cannot load route profiles: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema", "profiles", "resource_capacities", "source_reports",
    } or value.get("schema") != PROFILE_SCHEMA:
        raise PhysicalRuntimeError("route profile contract mismatch")
    source_reports = value["source_reports"]
    if not isinstance(source_reports, list) or not source_reports:
        raise PhysicalRuntimeError("route profiles require source reports")
    evidence_digests = set()
    verified_reports = []
    for row in source_reports:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise PhysicalRuntimeError("profile source report contract mismatch")
        source = Path(row["path"])
        if not source.is_absolute():
            source = path.parent / source
        observed = sha256_file(source)
        expected = str(row["sha256"])
        if expected != f"sha256:{observed}":
            raise PhysicalRuntimeError(f"profile evidence hash mismatch: {source}")
        evidence_digests.add(expected)
        verified_reports.append({"path": str(source), "sha256": expected})
    try:
        profiles = [
            FixedRouteProfile(**{
                **row,
                "resources": tuple(row["resources"]),
            })
            for row in value["profiles"]
        ]
        capacities = {
            str(name): int(capacity)
            for name, capacity in value["resource_capacities"].items()
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise PhysicalRuntimeError(f"invalid route profile value: {exc}") from exc
    if {profile.route_id for profile in profiles} != set(ROUTE_IDS):
        raise PhysicalRuntimeError("route profiles must define R0, R1, and R2")
    if any(profile.evidence_sha256 not in evidence_digests for profile in profiles):
        raise PhysicalRuntimeError("route profile references unverified evidence")
    return profiles, capacities, verified_reports


class CapacityLedger:
    def __init__(self, capacities: dict[str, int]):
        if set(capacities) != set(WORKER_NAMES) or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in capacities.values()
        ):
            raise ValueError("capacity ledger must define every fixed worker")
        self.capacities = dict(capacities)
        self.active = {name: 0 for name in capacities}
        self.pinned: dict[int, tuple[int, str]] = {}
        self._lock = threading.Lock()

    def try_reserve(self, request_id: int, route_epoch: int, route_id: str) -> bool:
        resources = ROUTE_RESOURCES[route_id]
        with self._lock:
            if request_id in self.pinned:
                raise PhysicalRuntimeError("capacity request is already pinned")
            if any(
                self.active[resource] >= self.capacities[resource]
                for resource in resources
            ):
                return False
            for resource in resources:
                self.active[resource] += 1
            self.pinned[request_id] = (route_epoch, route_id)
            return True

    def release(self, request_id: int, route_epoch: int) -> None:
        with self._lock:
            pinned = self.pinned.get(request_id)
            if pinned is None or pinned[0] != route_epoch:
                raise PhysicalRuntimeError("capacity release identity mismatch")
            for resource in ROUTE_RESOURCES[pinned[1]]:
                if self.active[resource] <= 0:
                    raise PhysicalRuntimeError("capacity resource underflow")
                self.active[resource] -= 1
            del self.pinned[request_id]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacities": dict(self.capacities),
                "active": dict(self.active),
                "pinned": {
                    str(request_id): [epoch, route_id]
                    for request_id, (epoch, route_id) in self.pinned.items()
                },
            }


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


def build_topology(args: argparse.Namespace) -> PhysicalTopology:
    endpoint_values = {
        "cuda-prefix": args.cuda_prefix,
        "cuda-mid": args.cuda_mid,
        "op12-prefix": args.op12,
        "op15-mid": args.op15,
        "cuda-tail": args.cuda_tail,
    }
    knees = {
        "cuda-prefix": args.cuda_prefix_knee,
        "cuda-mid": args.cuda_mid_knee,
        "op12-prefix": args.op12_knee,
        "op15-mid": args.op15_knee,
        "cuda-tail": args.cuda_tail_knee,
    }
    gathers = {
        "cuda-prefix": args.cuda_prefix_gather_us,
        "cuda-mid": args.cuda_mid_gather_us,
        "op12-prefix": args.op12_gather_us,
        "op15-mid": args.op15_gather_us,
        "cuda-tail": args.cuda_tail_gather_us,
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
            if name == "op15-mid" and args.control == "C1":
                continue
            batchers[name] = RouteDeviceBatcher(
                name,
                clients[name],
                knees[name],
                gathers[name],
                args.queue_depth,
            )
        if args.control == "C1":
            for route_id in ("R1", "R2"):
                label = f"op15-mid-{route_id.lower()}"
                batchers[label] = RouteDeviceBatcher(
                    label,
                    clients["op15-mid"],
                    knees["op15-mid"],
                    gathers["op15-mid"],
                    args.queue_depth,
                )

        def stage(
            worker_name: str,
            start: int,
            end: int,
            terminal: bool,
            batcher_label: str | None = None,
        ) -> ResidentStage:
            endpoint = endpoint_values[worker_name]
            resident = ResidentStage(
                worker_name=worker_name,
                layer_start=start,
                layer_end=end,
                endpoint=StageEndpoint(*endpoint),
                client=clients[worker_name],
                slots=slots[worker_name],
                batcher=batchers[batcher_label or worker_name],
                terminal=terminal,
            )
            validate_stage_hello(resident, hellos[worker_name])
            return resident

        cuda_prefix = stage("cuda-prefix", 0, 8, False)
        cuda_mid = stage("cuda-mid", 8, 16, False)
        op12_prefix = stage("op12-prefix", 0, 8, False)
        op15_r1 = stage(
            "op15-mid", 8, 16, False,
            "op15-mid-r1" if args.control == "C1" else None,
        )
        op15_r2 = op15_r1 if args.control != "C1" else stage(
            "op15-mid", 8, 16, False, "op15-mid-r2",
        )
        cuda_tail = stage("cuda-tail", 16, 48, True)
        routes = (
            FiniteRoute("R0", (cuda_prefix, cuda_mid, cuda_tail)),
            FiniteRoute("R1", (cuda_prefix, op15_r1, cuda_tail)),
            FiniteRoute("R2", (op12_prefix, op15_r2, cuda_tail)),
        )
        validate_fixed_routes(routes)
        if args.control != "C1":
            validate_shared_treatment(routes)
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
    return {
        name: topology.hellos[name].max_streams for name in WORKER_NAMES
    }


def validate_profile_capacities(
    profiles: Sequence[FixedRouteProfile],
    capacities: dict[str, int],
    topology: PhysicalTopology,
) -> None:
    physical = physical_capacities(topology)
    if set(capacities) != set(physical):
        raise PhysicalRuntimeError("profile capacities differ from fixed workers")
    for name, capacity in capacities.items():
        if capacity > physical[name]:
            raise PhysicalRuntimeError(f"profile exceeds {name} physical capacity")
    for profile in profiles:
        if profile.resources != ROUTE_RESOURCES[profile.route_id]:
            raise PhysicalRuntimeError(f"{profile.route_id} profile resources changed")
        if profile.max_active > min(capacities[name] for name in profile.resources):
            raise PhysicalRuntimeError(f"{profile.route_id} active limit exceeds resources")


@dataclass(frozen=True)
class ScheduledRow:
    source: dict[str, Any]
    arrival_us: int

    @property
    def request_id(self) -> int:
        return int(self.source["request_id"])

    @property
    def priority(self) -> int:
        return int(self.source["priority"])


def select_rows(
    trace: dict[str, Any],
    route_filter: Sequence[str],
    limit_per_route: int | None,
    arrival_scale: float,
    route_delays: dict[str, int],
) -> list[ScheduledRow]:
    allowed = set(route_filter or ROUTE_IDS)
    counts = Counter()
    result = []
    for source in trace["requests"]:
        route_id = source["route_hint"]
        if route_id not in allowed:
            continue
        if limit_per_route is not None and counts[route_id] >= limit_per_route:
            continue
        counts[route_id] += 1
        arrival_us = int(round(source["arrival_us"] * arrival_scale))
        arrival_us += route_delays[route_id]
        result.append(ScheduledRow(source, arrival_us))
    if not result:
        raise PhysicalRuntimeError("workload selection is empty")
    result.sort(key=lambda row: (row.arrival_us, row.request_id))
    return result


def outcome_record(
    outcome: RouteOutcome,
    source: dict[str, Any],
    origin_ns: int,
    admitted_ns: int,
    boundary_records: list[dict[str, Any]],
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
        "admission_queue_us": (
            admitted_ns - outcome.scheduled_arrival_ns
        ) // 1000,
        "lease_queue_us": outcome.lease_queue_us,
        "ttft_us": outcome.ttft_us,
        "latency_us": outcome.latency_us,
        "slo_us": outcome.slo_us,
        "slo_met": outcome.slo_met,
        "observed_input_tokens": source["observed_input_tokens"],
        "observed_output_tokens": source["observed_output_tokens"],
        "boundary_activations": boundary_records,
    }


def write_boundaries(
    outcome: RouteOutcome,
    activation_dir: Path | None,
) -> list[dict[str, Any]]:
    if not outcome.boundaries:
        return []
    if activation_dir is None:
        raise PhysicalRuntimeError("captured boundaries require an activation directory")
    records = []
    for boundary in outcome.boundaries:
        values = array("f", boundary.values)
        if sys.byteorder != "little":
            values.byteswap()
        payload = values.tobytes()
        name = (
            f"request-{outcome.request_id}-epoch-{outcome.route_epoch}-"
            f"{boundary.worker_name}-layer-{boundary.layer_end}-"
            f"position-{boundary.position}.f32"
        )
        path = activation_dir / name
        path.write_bytes(payload)
        norm = math.sqrt(sum(float(value) * float(value) for value in boundary.values))
        records.append({
            "worker": boundary.worker_name,
            "layer_end": boundary.layer_end,
            "position": boundary.position,
            "elements": len(boundary.values),
            "bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            "l2_norm": norm,
            "minimum": min(boundary.values),
            "maximum": max(boundary.values),
            "path": str(path),
        })
    return records


def run_workload(
    topology: PhysicalTopology,
    trace: dict[str, Any],
    args: argparse.Namespace,
    profiles: list[FixedRouteProfile] | None,
    profile_capacities: dict[str, int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    route_delays = parse_route_delays(args.route_delay_us)
    rows = select_rows(
        trace,
        args.route_filter,
        args.limit_per_route,
        args.arrival_scale,
        route_delays,
    )
    if args.capture_boundaries:
        if args.activation_dir is None:
            raise PhysicalRuntimeError("--capture-boundaries requires --activation-dir")
        if args.activation_dir.exists():
            raise PhysicalRuntimeError(f"activation directory exists: {args.activation_dir}")
        args.activation_dir.mkdir(parents=True)

    capacities = physical_capacities(topology)
    ledger = CapacityLedger(capacities) if args.control != "C3" else None
    policy = None
    if args.control == "C3":
        if profiles is None or profile_capacities is None:
            raise PhysicalRuntimeError("C3 requires verified route profiles")
        validate_profile_capacities(profiles, profile_capacities, topology)
        policy = FixedSloRouter(profiles, profile_capacities)

    origin_ns = time.monotonic_ns()
    nvml = None
    sampler = None
    if args.measure_energy:
        nvml = Nvml(args.gpu_index)
        if args.gpu_uuid is not None and nvml.uuid != args.gpu_uuid:
            nvml.close()
            raise PhysicalRuntimeError(
                f"NVML selected {nvml.uuid}, expected {args.gpu_uuid}"
            )
        sampler = NvmlSampler(
            nvml,
            origin_ns,
            args.nvml_interval_ms / 1000.0,
        )
        sampler.start()

    next_arrival = 0
    pending: list[ScheduledRow] = []
    completion_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    active: dict[int, threading.Thread] = {}
    completed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    fatal: BaseException | None = None
    next_epoch = 1

    profile_by_route = {
        profile.route_id: profile for profile in profiles or []
    }

    def elapsed_us() -> int:
        return max(0, (time.monotonic_ns() - origin_ns) // 1000)

    def execute(
        scheduled: ScheduledRow,
        route_id: str,
        route_epoch: int,
        batch_wait_us: int | None,
        admitted_ns: int,
    ) -> None:
        source = scheduled.source
        request = RouteRequest(
            request_id=scheduled.request_id,
            route_epoch=route_epoch,
            route_id=route_id,
            prompt_tokens=tuple(
                [source["synthetic_token"]] * source["input_tokens"]
            ),
            output_steps=source["output_steps"],
            slo_us=source["slo_us"],
            priority=source["priority"],
            batch_wait_us=batch_wait_us,
            prefill_chunk=args.prefill_chunk,
        )
        clean = False
        try:
            outcome = topology.runner.run(
                request,
                args.request_timeout,
                scheduled_arrival_ns=origin_ns + scheduled.arrival_us * 1000,
                capture_boundaries=args.capture_boundaries,
            )
            boundary_records = write_boundaries(outcome, args.activation_dir)
            clean = True
            if policy is not None:
                policy.complete(request.request_id, route_epoch)
            elif ledger is not None:
                ledger.release(request.request_id, route_epoch)
            completion_queue.put({
                "kind": "completed",
                "request_id": request.request_id,
                "record": outcome_record(
                    outcome,
                    source,
                    origin_ns,
                    admitted_ns,
                    boundary_records,
                ),
            })
        except BaseException as exc:
            if request.request_id not in topology.runner.pinned():
                try:
                    if policy is not None and policy.pinned(request.request_id) is not None:
                        policy.complete(request.request_id, route_epoch)
                    elif ledger is not None and request.request_id in ledger.pinned:
                        ledger.release(request.request_id, route_epoch)
                    clean = True
                except BaseException as release_exc:
                    exc = PhysicalRuntimeError(
                        f"request failure plus reservation release failure: {release_exc}"
                    )
            completion_queue.put({
                "kind": "fatal",
                "request_id": request.request_id,
                "clean": clean,
                "error": exc,
            })

    try:
        while len(completed) + len(rejected) < len(rows):
            now_us = elapsed_us()
            while next_arrival < len(rows) and rows[next_arrival].arrival_us <= now_us:
                pending.append(rows[next_arrival])
                next_arrival += 1

            while True:
                try:
                    item = completion_queue.get_nowait()
                except queue.Empty:
                    break
                active.pop(item["request_id"], None)
                if item["kind"] == "fatal":
                    fatal = item["error"]
                else:
                    completed.append(item["record"])
            if fatal is not None:
                break

            pending.sort(key=lambda row: (
                row.priority,
                row.arrival_us + int(row.source["slo_us"]),
                row.arrival_us,
                row.request_id,
            ))
            admitted_any = False
            index = 0
            while index < len(pending):
                scheduled = pending[index]
                source = scheduled.source
                now_us = elapsed_us()
                route_id: str
                route_epoch: int
                batch_wait_us = args.batch_wait_us
                reason: str
                if policy is not None:
                    work = SloWork(
                        scheduled.request_id,
                        scheduled.arrival_us,
                        source["slo_us"],
                        source["input_tokens"],
                        source["output_steps"],
                        source["priority"],
                    )
                    try:
                        decision = policy.admit(work, now_us)
                    except NoFeasibleRoute:
                        unconstrained = any(
                            now_us + profile.service_us(work) <= work.deadline_us
                            for profile in profile_by_route.values()
                        )
                        if not unconstrained and not active:
                            rejected.append({
                                "request_id": scheduled.request_id,
                                "arrival_us": scheduled.arrival_us,
                                "rejected_us": now_us,
                                "priority": scheduled.priority,
                                "reason": "NO_PROFILED_FIXED_ROUTE_CAN_MEET_SLO",
                            })
                            pending.pop(index)
                            continue
                        index += 1
                        continue
                    route_id = decision.route_id
                    route_epoch = decision.route_epoch
                    batch_wait_us = decision.batch_wait_us
                    reason = decision.reason
                else:
                    route_id = "R0" if args.control == "C0" else source["route_hint"]
                    route_epoch = next_epoch
                    if ledger is None or not ledger.try_reserve(
                        scheduled.request_id, route_epoch, route_id,
                    ):
                        index += 1
                        continue
                    next_epoch += 1
                    reason = (
                        "ALL_CUDA_EQUAL_WORK_CONTROL"
                        if args.control == "C0"
                        else "PINNED_TRACE_ROUTE_EQUAL_WORK_CONTROL"
                    )

                admitted_ns = time.monotonic_ns()
                decisions.append({
                    "request_id": scheduled.request_id,
                    "route_epoch": route_epoch,
                    "route_id": route_id,
                    "route_hint": source["route_hint"],
                    "priority": source["priority"],
                    "scheduled_arrival_us": scheduled.arrival_us,
                    "admitted_us": (admitted_ns - origin_ns) // 1000,
                    "admission_queue_us": (
                        admitted_ns - (origin_ns + scheduled.arrival_us * 1000)
                    ) // 1000,
                    "batch_wait_us": batch_wait_us,
                    "reason": reason,
                })
                pending.pop(index)
                thread = threading.Thread(
                    target=execute,
                    args=(
                        scheduled,
                        route_id,
                        route_epoch,
                        batch_wait_us,
                        admitted_ns,
                    ),
                    name=f"s24-request-{scheduled.request_id}",
                )
                active[scheduled.request_id] = thread
                thread.start()
                admitted_any = True

            if len(completed) + len(rejected) == len(rows):
                break
            if not admitted_any:
                wait_s = 0.01
                if next_arrival < len(rows):
                    wait_s = min(
                        wait_s,
                        max(0.0, (rows[next_arrival].arrival_us - elapsed_us()) / 1e6),
                    )
                try:
                    item = completion_queue.get(timeout=wait_s)
                    active.pop(item["request_id"], None)
                    if item["kind"] == "fatal":
                        fatal = item["error"]
                    else:
                        completed.append(item["record"])
                except queue.Empty:
                    pass
                if fatal is not None:
                    break

        for thread in list(active.values()):
            thread.join(timeout=args.request_timeout)
        if any(thread.is_alive() for thread in active.values()):
            raise TimeoutError("request thread did not terminate")
        while True:
            try:
                item = completion_queue.get_nowait()
            except queue.Empty:
                break
            if item["kind"] == "fatal" and fatal is None:
                fatal = item["error"]
            elif item["kind"] == "completed":
                completed.append(item["record"])
        if fatal is not None:
            raise PhysicalRuntimeError("physical request failed") from fatal
        if len(completed) + len(rejected) != len(rows):
            raise PhysicalRuntimeError("request conservation failed")
    finally:
        duration_ns = time.monotonic_ns() - origin_ns
        if sampler is not None:
            sampler.stop()
        if nvml is not None:
            nvml.close()

    completed.sort(key=lambda row: row["request_id"])
    rejected.sort(key=lambda row: row["request_id"])
    if topology.runner.pinned():
        raise PhysicalRuntimeError("route runner retained a request pin")
    if any(pool.leased() for pool in topology.slots.values()):
        raise PhysicalRuntimeError("software sequence leases remain live")
    if ledger is not None and (ledger.pinned or any(ledger.active.values())):
        raise PhysicalRuntimeError("capacity ledger did not drain")
    if policy is not None:
        counts = policy.active_counts()
        if any(counts["routes"].values()) or any(counts["resources"].values()):
            raise PhysicalRuntimeError("SLO policy did not drain")

    energy = None
    nvml_samples: list[dict[str, Any]] = []
    nvml_errors: list[str] = []
    if sampler is not None:
        nvml_samples = sampler.samples
        nvml_errors = sampler.errors
        if nvml_errors:
            raise PhysicalRuntimeError(f"NVML sampling failed: {nvml_errors}")
        energy = integrate_nvml(nvml_samples, duration_ns)
        energy["requested_sample_interval_ms"] = args.nvml_interval_ms
        expected_pids = set(args.expected_gpu_pid)
        observed_compute = {
            int(process["pid"])
            for sample in nvml_samples
            for process in sample["processes"]
            if process["kind"] == "compute"
        }
        if expected_pids and observed_compute != expected_pids:
            raise PhysicalRuntimeError(
                f"NVML compute PID set {sorted(observed_compute)} differs from "
                f"expected {sorted(expected_pids)}"
            )
        energy["gpu_uuid"] = args.gpu_uuid
        energy["expected_compute_pids"] = sorted(expected_pids)
        energy["observed_compute_pids"] = sorted(observed_compute)

    runtime = {
        "origin_monotonic_ns": origin_ns,
        "duration_ns": duration_ns,
        "selected_request_count": len(rows),
        "completed_count": len(completed),
        "rejected_count": len(rejected),
        "route_delays_us": route_delays,
        "decisions": decisions,
        "requests": completed,
        "rejected": rejected,
        "energy": energy,
        "nvml_samples": nvml_samples,
        "nvml_errors": nvml_errors,
    }
    state = {
        "runner_pins": topology.runner.pinned(),
        "software_leases": {
            name: pool.leased() for name, pool in topology.slots.items()
        },
        "capacity_ledger": ledger.snapshot() if ledger is not None else None,
        "policy_active": policy.active_counts() if policy is not None else None,
    }
    return runtime, completed, state


def summarize_run(
    runtime: dict[str, Any],
    batch_events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    requests = runtime["requests"]
    route_distribution = Counter(row["route_id"] for row in requests)
    priority = {}
    for value in sorted({row["priority"] for row in requests}):
        rows = [row for row in requests if row["priority"] == value]
        priority[str(value)] = {
            "completed": len(rows),
            "slo_misses": sum(not row["slo_met"] for row in rows),
            "ttft_us": summarize_values([row["ttft_us"] for row in rows]),
            "latency_us": summarize_values([row["latency_us"] for row in rows]),
            "admission_queue_us": summarize_values([
                row["admission_queue_us"] for row in rows
            ]),
            "lease_queue_us": summarize_values([
                row["lease_queue_us"] for row in rows
            ]),
        }
    batch_summaries = {
        name: summarize_batch_events(events)
        for name, events in batch_events.items()
    }
    op15_events = [
        event
        for name, events in batch_events.items()
        if name == "op15-mid" or name.startswith("op15-mid-r")
        for event in events
    ]
    cuda_names = ("cuda-prefix", "cuda-mid", "cuda-tail")
    cuda_compute = {
        name: sum(
            int(event["compute_us"])
            for event in batch_events.get(name, [])
            if event.get("status") == "OK"
        )
        for name in cuda_names
    }
    return {
        "completed_requests": len(requests),
        "rejected_requests": len(runtime["rejected"]),
        "slo_misses": sum(not row["slo_met"] for row in requests),
        "makespan_us": runtime["duration_ns"] // 1000,
        "route_distribution": dict(sorted(route_distribution.items())),
        "priority": priority,
        "workers": batch_summaries,
        "op15_pooled": summarize_batch_events(op15_events),
        "cuda_tail": batch_summaries.get("cuda-tail", summarize_batch_events([])),
        "cuda_island_compute_us": cuda_compute,
        "summed_cuda_island_compute_us": sum(cuda_compute.values()),
        "gpu_board_energy": runtime["energy"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--cuda-prefix", type=parse_endpoint, required=True)
    result.add_argument("--cuda-mid", type=parse_endpoint, required=True)
    result.add_argument("--op12", type=parse_endpoint, required=True)
    result.add_argument("--op15", type=parse_endpoint, required=True)
    result.add_argument("--cuda-tail", type=parse_endpoint, required=True)
    result.add_argument("--trace", type=Path, required=True)
    result.add_argument("--control", choices=CONTROL_IDS, required=True)
    result.add_argument("--profiles", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--activation-dir", type=Path)
    result.add_argument("--capture-boundaries", action="store_true")
    result.add_argument("--allow-numeric-uncertified", action="store_true")
    result.add_argument("--session-end", choices=("detach", "stop"), default="detach")
    result.add_argument("--route-filter", action="append", choices=ROUTE_IDS, default=[])
    result.add_argument("--limit-per-route", type=int)
    result.add_argument("--route-delay-us", action="append", default=[])
    result.add_argument("--arrival-scale", type=float, default=1.0)
    result.add_argument("--prefill-chunk", type=int)
    result.add_argument("--batch-wait-us", type=int)
    result.add_argument("--queue-depth", type=int, default=1024)
    result.add_argument("--timeout", type=float, default=600.0)
    result.add_argument("--request-timeout", type=float, default=3600.0)
    result.add_argument("--cuda-prefix-knee", type=int, default=4)
    result.add_argument("--cuda-mid-knee", type=int, default=4)
    result.add_argument("--op12-knee", type=int, required=True)
    result.add_argument("--op15-knee", type=int, required=True)
    result.add_argument("--cuda-tail-knee", type=int, default=8)
    result.add_argument("--cuda-prefix-gather-us", type=int, default=5000)
    result.add_argument("--cuda-mid-gather-us", type=int, default=5000)
    result.add_argument("--op12-gather-us", type=int, default=5000)
    result.add_argument("--op15-gather-us", type=int, default=5000)
    result.add_argument("--cuda-tail-gather-us", type=int, default=5000)
    result.add_argument("--measure-energy", action="store_true")
    result.add_argument("--gpu-index", type=int, default=0)
    result.add_argument("--gpu-uuid")
    result.add_argument("--nvml-interval-ms", type=float, default=10.0)
    result.add_argument("--expected-gpu-pid", type=int, action="append", default=[])
    return result


def validate_args(args: argparse.Namespace, arg_parser: argparse.ArgumentParser) -> None:
    positive = (
        args.queue_depth,
        args.timeout,
        args.request_timeout,
        args.cuda_prefix_knee,
        args.cuda_mid_knee,
        args.op12_knee,
        args.op15_knee,
        args.cuda_tail_knee,
        args.arrival_scale,
        args.nvml_interval_ms,
    )
    if any(value <= 0 for value in positive):
        arg_parser.error("runtime bounds must be positive")
    for value in (
        args.cuda_prefix_gather_us,
        args.cuda_mid_gather_us,
        args.op12_gather_us,
        args.op15_gather_us,
        args.cuda_tail_gather_us,
    ):
        if value < 0:
            arg_parser.error("gather bounds cannot be negative")
    if args.limit_per_route is not None and args.limit_per_route <= 0:
        arg_parser.error("route limit must be positive")
    if args.prefill_chunk is not None and args.prefill_chunk <= 0:
        arg_parser.error("prefill chunk must be positive")
    if args.batch_wait_us is not None and args.batch_wait_us < 0:
        arg_parser.error("batch wait cannot be negative")
    if args.control == "C3" and args.profiles is None:
        arg_parser.error("C3 requires --profiles")
    if args.control != "C3" and args.profiles is not None:
        arg_parser.error("profiles are accepted only for C3")
    if args.measure_energy and not args.gpu_uuid:
        arg_parser.error("energy measurement requires --gpu-uuid")
    selected_routes = (
        {"R0"}
        if args.control == "C0"
        else set(args.route_filter or ROUTE_IDS)
    )
    if args.control == "C3":
        selected_routes = set(ROUTE_IDS)
    if selected_routes & {"R1", "R2"} and not args.allow_numeric_uncertified:
        arg_parser.error("phone routes require --allow-numeric-uncertified")
    if args.output.exists():
        arg_parser.error(f"output already exists: {args.output}")


def main() -> int:
    arg_parser = parser()
    args = arg_parser.parse_args()
    validate_args(args, arg_parser)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    trace = load_trace(args.trace)
    profiles = None
    profile_capacities = None
    profile_sources: list[dict[str, Any]] = []
    if args.profiles is not None:
        profiles, profile_capacities, profile_sources = load_profiles(args.profiles)

    topology = None
    failure: BaseException | None = None
    try:
        topology = build_topology(args)
        runtime, _completed, software_state = run_workload(
            topology,
            trace,
            args,
            profiles,
            profile_capacities,
        )
        topology.stop_batchers(args.timeout)
        batch_events = {
            name: list(batcher.events)
            for name, batcher in topology.batchers.items()
        }
        final_workers = topology.end_sessions(args.session_end)
        report = {
            "schema": SCHEMA,
            "status": "RUN_COMPLETE",
            "control": args.control,
            "numeric_scope": (
                "MECHANICS_ONLY_F16_PHONE_Q8_SERVER_NUMERICALLY_UNCERTIFIED"
                if args.allow_numeric_uncertified
                else "Q8_CUDA_ONLY"
            ),
            "trace": {
                "path": str(args.trace),
                "sha256": "sha256:" + sha256_file(args.trace),
                "trace_hash": trace["trace_hash"],
                "scope": trace["scope"],
            },
            "configuration": {
                "session_end": args.session_end,
                "route_filter": args.route_filter or list(ROUTE_IDS),
                "limit_per_route": args.limit_per_route,
                "arrival_scale": args.arrival_scale,
                "prefill_chunk": args.prefill_chunk,
                "fixed_batch_wait_us": args.batch_wait_us,
                "queue_depth": args.queue_depth,
                "knees": {
                    "cuda-prefix": args.cuda_prefix_knee,
                    "cuda-mid": args.cuda_mid_knee,
                    "op12-prefix": args.op12_knee,
                    "op15-mid": args.op15_knee,
                    "cuda-tail": args.cuda_tail_knee,
                },
                "gather_us": {
                    "cuda-prefix": args.cuda_prefix_gather_us,
                    "cuda-mid": args.cuda_mid_gather_us,
                    "op12-prefix": args.op12_gather_us,
                    "op15-mid": args.op15_gather_us,
                    "cuda-tail": args.cuda_tail_gather_us,
                },
                "endpoints": {
                    "cuda-prefix": list(args.cuda_prefix),
                    "cuda-mid": list(args.cuda_mid),
                    "op12-prefix": list(args.op12),
                    "op15-mid": list(args.op15),
                    "cuda-tail": list(args.cuda_tail),
                },
            },
            "profile_sources": profile_sources,
            "workers": {
                name: asdict(hello) for name, hello in topology.hellos.items()
            },
            "runtime": runtime,
            "batch_events": batch_events,
            "summary": summarize_run(runtime, batch_events),
            "final_workers": final_workers,
            "final_software_state": software_state,
            "energy_exclusions": {
                "phone_energy": "UNKNOWN",
                "network_energy": "UNKNOWN",
                "a6000_host_energy": "UNKNOWN",
                "total_system_energy": "UNKNOWN",
            },
        }
        args.output.write_bytes(canonical_bytes(report))
        print(json.dumps({
            "status": report["status"],
            "control": args.control,
            "completed": report["summary"]["completed_requests"],
            "slo_misses": report["summary"]["slo_misses"],
            "output": str(args.output),
        }, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException as exc:
        failure = exc
        failure_report = {
            "schema": SCHEMA,
            "status": "RUN_FAILED",
            "control": args.control,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            args.output.write_bytes(canonical_bytes(failure_report))
        except OSError:
            pass
        print(json.dumps(failure_report, sort_keys=True, separators=(",", ":")))
        return 2
    finally:
        if topology is not None:
            if not topology.batchers_stopped:
                try:
                    topology.stop_batchers(args.timeout)
                except BaseException:
                    pass
            if not topology.ended:
                try:
                    if all(
                        topology.clients[name].status().active_sequences == 0
                        for name in WORKER_NAMES
                    ):
                        topology.end_sessions("detach")
                except BaseException:
                    pass
            topology.close()
        if failure is not None:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
