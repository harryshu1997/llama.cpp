#!/usr/bin/env python3
"""Replay a dense observed-arrival trace through live StageNet V3 workers."""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
S22 = ROOT / "research_dev/spikes/s22_slo_overlap_pipeline"
sys.path.insert(0, str(S22))

from async_pipeline import DeviceBatcher, RequestSpec, parse_endpoint, run_request, summarize_batches
from mixed_slo_pipeline import load_profiles, require_numeric_override
from slo_policy import NoFeasibleRoute, RouteProfile, SloRouter, WorkRequest
from stage_v3_client import ProtocolError, STAGE_V3_CAP_TERMINAL, StageV3Client

from dense_trace import TraceError, strict_object, validate
from runtime_support import SequenceSlotPool, SerializedStageClient, SlotError


def load_requests(path: Path) -> tuple[dict[str, Any], list[WorkRequest]]:
    trace = strict_object(path)
    validate(trace)
    requests = [
        WorkRequest(
            row["request_id"], row["arrival_us"], row["slo_us"],
            row["execution_steps"], row["priority"],
        )
        for row in trace["requests"]
    ]
    return trace, requests


def fastest_service_us(profiles: list[RouteProfile], request: WorkRequest) -> int:
    return min(profile.service_us(request.steps) for profile in profiles)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=parse_endpoint, required=True)
    parser.add_argument("--op15", type=parse_endpoint, required=True)
    parser.add_argument("--op12", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--queue-depth", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--allow-numeric-uncertified", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.queue_depth <= 0 or args.timeout <= 0 or args.token < 0:
        parser.error("queue, timeout, and token bounds are invalid")

    profiles, correctness_scopes = load_profiles(args.profiles)
    trace, requests = load_requests(args.trace)
    require_numeric_override(
        correctness_scopes, [profile.route_id for profile in profiles],
        args.allow_numeric_uncertified,
    )
    profile_by_head = {profile.head_name: profile for profile in profiles}
    endpoint_names = {"cuda", "op15", "op12"}
    if set(profile_by_head) != endpoint_names:
        raise ValueError("profiles must define exactly cuda, op15, and op12 heads")

    raw_clients: dict[str, StageV3Client] = {}
    clients: dict[str, SerializedStageClient] = {}
    batchers: dict[str, DeviceBatcher] = {}
    threads: list[threading.Thread] = []
    try:
        for name in sorted(endpoint_names):
            raw_clients[name] = StageV3Client.connect(*getattr(args, name), args.timeout)
        raw_clients["tail"] = StageV3Client.connect(*args.tail, args.timeout)
        clients = {name: SerializedStageClient(client) for name, client in raw_clients.items()}
        hellos = {name: client.hello() for name, client in clients.items()}
        tail_hello = hellos["tail"]
        if not tail_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("dense runtime tail is not terminal")
        for name in endpoint_names:
            hello = hellos[name]
            profile = profile_by_head[name]
            if hello.layer_start != 0 or hello.capabilities & STAGE_V3_CAP_TERMINAL:
                raise ProtocolError(f"{name} is not a prefix worker")
            if hello.layer_end != tail_hello.layer_start:
                raise ProtocolError(f"{name} join boundary differs from the tail")
            if profile.max_active > hello.max_streams:
                raise ProtocolError(f"{name} profile exceeds physical stream capacity")

        pools = {
            name: SequenceSlotPool(profile_by_head[name].max_active)
            for name in endpoint_names
        }
        pools["tail"] = SequenceSlotPool(tail_hello.max_streams)
        gather_by_head = {
            profile.head_name: profile.gather_cap_us for profile in profiles
        }
        tail_gather_us = min(profile.gather_cap_us for profile in profiles)
        for name, client in clients.items():
            hello = hellos[name]
            gather_us = tail_gather_us if name == "tail" else gather_by_head[name]
            batchers[name] = DeviceBatcher(
                name, client, min(hello.n_batch, hello.n_ubatch),
                gather_us, args.queue_depth,
            )

        router = SloRouter(profiles)
        trace_rows = {row["request_id"]: row for row in trace["requests"]}
        ordered = sorted(requests, key=lambda request: (
            request.arrival_us, request.priority, request.request_id,
        ))
        pending: list[WorkRequest] = []
        next_arrival = 0
        completions: queue.Queue[dict[str, Any]] = queue.Queue()
        completed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        fatal_errors: list[BaseException] = []
        start_ns = time.monotonic_ns()

        def elapsed_us() -> int:
            return max(0, (time.monotonic_ns() - start_ns) // 1000)

        def execute(
            request: WorkRequest, decision, head_seq: int, tail_seq: int,
            admitted_us: int,
        ) -> None:
            head_name = decision.head_name
            source_row = trace_rows[request.request_id]
            spec = RequestSpec(
                request.request_id, decision.route_epoch, head_name,
                head_seq, tail_seq, request.steps, request.slo_us / 1000.0,
                decision.batch_wait_us,
                tuple([args.token] * source_row["execution_input_tokens"]),
                source_row["execution_input_tokens"],
            )
            try:
                outcome = run_request(
                    spec, batchers[head_name], batchers["tail"], args.token,
                    threading.Barrier(1), args.timeout,
                )
                finished_us = elapsed_us()
                clients[head_name].remove(head_seq, request.request_id, decision.route_epoch)
                clients["tail"].remove(tail_seq, request.request_id, decision.route_epoch)
                router.complete(request.request_id, decision.route_epoch)
                pools[head_name].release(head_seq, request.request_id)
                pools["tail"].release(tail_seq, request.request_id)
                outcome.update({
                    "arrival_us": request.arrival_us,
                    "admitted_us": admitted_us,
                    "finished_us": finished_us,
                    "elapsed_ms": (finished_us - request.arrival_us) / 1000.0,
                    "queue_ms": (admitted_us - request.arrival_us) / 1000.0,
                    "slo_met": finished_us <= request.deadline_us,
                    "priority": request.priority,
                    "route_id": decision.route_id,
                    "event_id": source_row["event_id"],
                    "observed_input_tokens": source_row["observed_input_tokens"],
                    "observed_output_tokens": source_row["observed_output_tokens"],
                })
                completions.put({"kind": "completed", "outcome": outcome})
            except BaseException as exc:
                completions.put({"kind": "fatal", "error": exc})

        while len(completed) + len(rejected) < len(ordered):
            now_us = elapsed_us()
            while next_arrival < len(ordered) and ordered[next_arrival].arrival_us <= now_us:
                pending.append(ordered[next_arrival])
                next_arrival += 1
            while True:
                try:
                    item = completions.get_nowait()
                except queue.Empty:
                    break
                if item["kind"] == "fatal":
                    fatal_errors.append(item["error"])
                else:
                    completed.append(item["outcome"])
            if fatal_errors:
                raise RuntimeError("dense physical request failed") from fatal_errors[0]

            pending.sort(key=lambda request: (
                request.priority, request.deadline_us, request.arrival_us,
                request.request_id,
            ))
            admitted_any = False
            index = 0
            while index < len(pending) and pools["tail"].available() > 0:
                request = pending[index]
                now_us = elapsed_us()
                try:
                    decision = router.admit(request, now_us)
                except NoFeasibleRoute:
                    if now_us + fastest_service_us(profiles, request) > request.deadline_us:
                        rejected.append({
                            "request_id": request.request_id,
                            "event_id": trace_rows[request.request_id]["event_id"],
                            "arrival_us": request.arrival_us,
                            "rejected_us": now_us,
                            "priority": request.priority,
                            "reason": "NO_PROFILED_ROUTE_CAN_MEET_DEADLINE",
                        })
                        pending.pop(index)
                        continue
                    index += 1
                    continue

                head_pool = pools[decision.head_name]
                head_seq = head_pool.try_acquire(request.request_id)
                tail_seq = pools["tail"].try_acquire(request.request_id)
                if head_seq is None or tail_seq is None:
                    if head_seq is not None:
                        head_pool.release(head_seq, request.request_id)
                    if tail_seq is not None:
                        pools["tail"].release(tail_seq, request.request_id)
                    router.complete(request.request_id, decision.route_epoch)
                    raise RuntimeError("router and physical slot ledgers disagree")

                admitted_us = elapsed_us()
                decisions.append({
                    **asdict(decision),
                    "arrival_us": request.arrival_us,
                    "admitted_us": admitted_us,
                    "queue_us": admitted_us - request.arrival_us,
                    "head_seq": head_seq,
                    "tail_seq": tail_seq,
                })
                pending.pop(index)
                thread = threading.Thread(
                    target=execute,
                    args=(request, decision, head_seq, tail_seq, admitted_us),
                    name=f"dense-{request.request_id}",
                )
                threads.append(thread)
                thread.start()
                admitted_any = True

            if len(completed) + len(rejected) == len(ordered):
                break
            if not admitted_any:
                wait_s = 0.01
                if next_arrival < len(ordered):
                    wait_s = min(
                        wait_s,
                        max(0.0, (ordered[next_arrival].arrival_us - elapsed_us()) / 1e6),
                    )
                try:
                    item = completions.get(timeout=wait_s)
                    if item["kind"] == "fatal":
                        fatal_errors.append(item["error"])
                    else:
                        completed.append(item["outcome"])
                except queue.Empty:
                    pass

        for thread in threads:
            thread.join(timeout=args.timeout)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("dense request thread did not terminate")
        if fatal_errors:
            raise RuntimeError("dense physical request failed") from fatal_errors[0]

        for batcher in batchers.values():
            batcher.stop(args.timeout)
        events = {name: list(batcher.events) for name, batcher in batchers.items()}
        batchers.clear()
        statuses = {name: client.drain() for name, client in clients.items()}
        if any(status.active_sequences != 0 for status in statuses.values()):
            raise ProtocolError("dense runtime left live worker sequences")
        if any(router.active_counts().values()) or any(pool.leased() for pool in pools.values()):
            raise RuntimeError("dense runtime left live software leases")

        completed.sort(key=lambda row: row["request_id"])
        rejected.sort(key=lambda row: row["request_id"])
        all_slo_met = not rejected and len(completed) == len(ordered) \
            and all(row["slo_met"] for row in completed)
        verdict = (
            "DENSE_ARRIVAL_MECHANICS_PASS_NUMERICALLY_UNCERTIFIED"
            if all_slo_met else "DENSE_ARRIVAL_SLO_OR_ADMISSION_FAIL"
        )
        report = {
            "schema": "s23-dense-physical-v1",
            "verdict": verdict,
            "mechanics_pass": all_slo_met,
            "trace_hash": trace["trace_hash"],
            "trace_scope": trace["scope"],
            "correctness_scopes": correctness_scopes,
            "numeric_uncertified_override": args.allow_numeric_uncertified,
            "request_count": len(ordered),
            "completed_count": len(completed),
            "rejected_count": len(rejected),
            "decisions": decisions,
            "requests": completed,
            "rejected": rejected,
            "workers": {name: asdict(hello) for name, hello in hellos.items()},
            "batches": {name: summarize_batches(rows) for name, rows in events.items()},
            "batch_events": events,
        }
        for client in clients.values():
            if args.session_end == "stop":
                client.stop()
            else:
                client.detach()
        data = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="ascii")
        print(data, end="")
        return 0 if all_slo_met else 2
    finally:
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


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, ProtocolError, RuntimeError, SlotError,
            TimeoutError, TraceError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
