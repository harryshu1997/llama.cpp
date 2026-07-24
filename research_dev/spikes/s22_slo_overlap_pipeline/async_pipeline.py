#!/usr/bin/env python3
"""Asynchronous two-head, one-tail StageNet V3 proof on real devices."""

from __future__ import annotations

import argparse
import json
import queue
import statistics
import threading
import time
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from stage_v3_client import (
    BatchResult,
    BatchRow,
    ProtocolError,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    route_epoch: int
    head_name: str
    head_seq: int
    tail_seq: int
    steps: int
    slo_ms: float
    batch_wait_us: int | None = None
    prompt_tokens: tuple[int, ...] | None = None
    prefill_chunk: int | None = None


@dataclass
class PendingRow:
    row: BatchRow
    future: Future[BatchResult]
    enqueued_ns: int
    latest_dispatch_ns: int
    priority: int
    order: int


class DeviceBatcher:
    MAX_PRIORITY = (1 << 31) - 1

    def __init__(
        self,
        name: str,
        client: StageV3Client,
        max_batch: int,
        gather_us: int,
        queue_depth: int,
    ):
        if max_batch <= 0 or gather_us < 0 or queue_depth <= 0:
            raise ValueError("invalid batcher bounds")
        self.name = name
        self.client = client
        self.max_batch = max_batch
        self.gather_us = gather_us
        self.events: list[dict] = []
        self._queue: queue.PriorityQueue[
            tuple[int, int, int, PendingRow | None]
        ] = queue.PriorityQueue(queue_depth)
        self._error: BaseException | None = None
        self._stopping = False
        self._next_order = 0
        self._state_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"batch-{name}")
        self._thread.start()

    def submit(
        self,
        row: BatchRow,
        timeout_s: float,
        batch_wait_us: int | None = None,
        priority: int = 0,
    ) -> Future[BatchResult]:
        if batch_wait_us is not None and batch_wait_us < 0:
            raise ValueError("batch wait budget cannot be negative")
        if (
            isinstance(priority, bool)
            or not isinstance(priority, int)
            or not 0 <= priority <= self.MAX_PRIORITY
        ):
            raise ValueError("priority is out of range")
        future: Future[BatchResult] = Future()
        enqueued_ns = time.monotonic_ns()
        wait_us = self.gather_us if batch_wait_us is None else min(
            self.gather_us, batch_wait_us,
        )
        with self._state_lock:
            if self._error is not None:
                raise RuntimeError(f"{self.name} batcher failed") from self._error
            if self._stopping:
                raise RuntimeError(f"{self.name} batcher is stopping")
            order = self._next_order
            self._next_order += 1
            pending = PendingRow(
                row,
                future,
                enqueued_ns,
                enqueued_ns + wait_us * 1000,
                priority,
                order,
            )
            try:
                self._queue.put(self._queue_item(pending), timeout=timeout_s)
            except queue.Full as exc:
                raise TimeoutError(f"{self.name} admission queue is full") from exc
        return future

    @staticmethod
    def _priority_band(priority: int) -> int:
        return 0 if priority == 0 else 1

    @staticmethod
    def _queue_item(
        pending: PendingRow,
    ) -> tuple[int, int, int, PendingRow]:
        return (
            pending.priority,
            pending.latest_dispatch_ns,
            pending.order,
            pending,
        )

    def _fail_pending(self, error: BaseException) -> None:
        while True:
            try:
                _priority, _deadline, _order, pending = self._queue.get_nowait()
            except queue.Empty:
                return
            if pending is not None and not pending.future.done():
                pending.future.set_exception(error)

    def _run(self) -> None:
        try:
            while True:
                _priority, _deadline, _order, first = self._queue.get()
                if first is None:
                    return
                pending = [first]
                deadline_ns = first.latest_dispatch_ns
                priority_band = self._priority_band(first.priority)
                while len(pending) < self.max_batch:
                    try:
                        queue_item = self._queue.get_nowait()
                    except queue.Empty:
                        remaining_s = (deadline_ns - time.monotonic_ns()) / 1e9
                        if remaining_s <= 0:
                            break
                        try:
                            queue_item = self._queue.get(timeout=remaining_s)
                        except queue.Empty:
                            break
                    item = queue_item[3]
                    if item is None:
                        self._queue.put_nowait(queue_item)
                        break
                    item_band = self._priority_band(item.priority)
                    if item_band != priority_band:
                        if item_band < priority_band:
                            for queued in pending:
                                self._queue.put_nowait(self._queue_item(queued))
                            pending = [item]
                            deadline_ns = item.latest_dispatch_ns
                            priority_band = item_band
                            continue
                        self._queue.put_nowait(queue_item)
                        break
                    pending.append(item)
                    deadline_ns = min(deadline_ns, item.latest_dispatch_ns)

                compute_start_ns = time.monotonic_ns()
                results = self.client.batch([item.row for item in pending])
                compute_end_ns = time.monotonic_ns()
                if len(results) != len(pending):
                    raise ProtocolError(f"{self.name} result count mismatch")
                self.events.append({
                    "batch_size": len(pending),
                    "compute_us": (compute_end_ns - compute_start_ns) // 1000,
                    "max_queue_us": max(
                        (compute_start_ns - item.enqueued_ns) // 1000 for item in pending
                    ),
                    "request_ids": [item.row.request_id for item in pending],
                    "priorities": [item.priority for item in pending],
                })
                for item, result in zip(pending, results):
                    item.future.set_result(result)
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
            for item in locals().get("pending", []):
                if not item.future.done():
                    item.future.set_exception(exc)
            self._fail_pending(exc)

    def stop(self, timeout_s: float) -> None:
        with self._state_lock:
            if self._stopping:
                raise RuntimeError(f"{self.name} batcher stop repeated")
            self._stopping = True
        self._queue.put((
            self.MAX_PRIORITY + 1,
            (1 << 63) - 1,
            (1 << 63) - 1,
            None,
        ), timeout=timeout_s)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            raise TimeoutError(f"{self.name} batcher did not stop")
        if self._error is not None:
            raise RuntimeError(f"{self.name} batcher failed") from self._error


def parse_endpoint(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("endpoint port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("endpoint port is out of range")
    return host, port


def parse_tokens(value: str) -> tuple[int, ...]:
    try:
        tokens = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("tokens must be comma-separated integers") from exc
    if not tokens or any(token < 0 for token in tokens):
        raise argparse.ArgumentTypeError("tokens must be nonnegative and nonempty")
    return tokens


def run_request(
    spec: RequestSpec,
    head: DeviceBatcher,
    tail: DeviceBatcher,
    token: int,
    start_barrier: threading.Barrier,
    operation_timeout_s: float,
) -> dict:
    start_barrier.wait(timeout=operation_timeout_s)
    start_ns = time.monotonic_ns()
    prompt_tokens = spec.prompt_tokens or (token,)
    if not prompt_tokens:
        raise ValueError("request prompt cannot be empty")

    prefill_chunk = spec.prefill_chunk or len(prompt_tokens)
    if prefill_chunk <= 0:
        raise ValueError("prefill chunk must be positive")
    head_results: list[BatchResult] = []
    for start in range(0, len(prompt_tokens), prefill_chunk):
        stop = min(start + prefill_chunk, len(prompt_tokens))
        head_futures = [
            head.submit(
                BatchRow(
                    spec.request_id, spec.route_epoch, spec.head_seq,
                    position, prompt_tokens[position],
                ),
                operation_timeout_s, spec.batch_wait_us,
            )
            for position in range(start, stop)
        ]
        head_results.extend(
            future.result(timeout=operation_timeout_s) for future in head_futures
        )
    if any(result.hidden is None or result.token is not None for result in head_results):
        raise ProtocolError("head stage returned a terminal result")
    tail_results: list[BatchResult] = []
    for start in range(0, len(prompt_tokens), prefill_chunk):
        stop = min(start + prefill_chunk, len(prompt_tokens))
        tail_futures = [
            tail.submit(
                BatchRow(
                    spec.request_id, spec.route_epoch, spec.tail_seq,
                    position, prompt_tokens[position], head_results[position].hidden,
                ),
                operation_timeout_s, spec.batch_wait_us,
            )
            for position in range(start, stop)
        ]
        tail_results.extend(
            future.result(timeout=operation_timeout_s) for future in tail_futures
        )
    if any(result.token is None or result.hidden is not None for result in tail_results):
        raise ProtocolError("tail stage returned an activation")
    next_token = tail_results[-1].token
    if next_token is None:
        raise ProtocolError("tail omitted the prompt result token")
    generated = [next_token]

    for output_index in range(1, spec.steps):
        position = len(prompt_tokens) + output_index - 1
        head_result = head.submit(
            BatchRow(
                spec.request_id, spec.route_epoch, spec.head_seq,
                position, next_token,
            ),
            operation_timeout_s, spec.batch_wait_us,
        ).result(timeout=operation_timeout_s)
        if head_result.hidden is None or head_result.token is not None:
            raise ProtocolError("head stage returned a terminal result")
        tail_result = tail.submit(
            BatchRow(
                spec.request_id, spec.route_epoch, spec.tail_seq,
                position, next_token, head_result.hidden,
            ),
            operation_timeout_s, spec.batch_wait_us,
        ).result(timeout=operation_timeout_s)
        if tail_result.token is None or tail_result.hidden is not None:
            raise ProtocolError("tail stage returned an activation")
        next_token = tail_result.token
        generated.append(next_token)

    elapsed_ms = (time.monotonic_ns() - start_ns) / 1e6
    return {
        "elapsed_ms": elapsed_ms,
        "head": spec.head_name,
        "prompt_length": len(prompt_tokens),
        "prefill_chunk": prefill_chunk,
        "request_id": spec.request_id,
        "route_epoch": spec.route_epoch,
        "slo_ms": spec.slo_ms,
        "slo_met": elapsed_ms <= spec.slo_ms,
        "tokens": generated,
    }


def summarize_batches(events: Sequence[dict]) -> dict:
    sizes = [event["batch_size"] for event in events]
    return {
        "batch_count": len(sizes),
        "batch_sizes": sizes,
        "max_batch": max(sizes, default=0),
        "mean_batch": statistics.fmean(sizes) if sizes else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op15", type=parse_endpoint, required=True)
    parser.add_argument("--op12", type=parse_endpoint, required=True)
    parser.add_argument("--tail", type=parse_endpoint, required=True)
    parser.add_argument("--requests-per-phone", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--gather-us", type=int, default=5000)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--slo-ms", type=float, default=30000.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--token", type=int, default=2)
    parser.add_argument("--prompt-tokens", type=parse_tokens)
    parser.add_argument("--prefill-chunk", type=int)
    parser.add_argument("--session-end", choices=("stop", "detach"), default="stop")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests_per_phone <= 0 or args.steps <= 0 or args.gather_us < 0:
        parser.error("request, step, and gather bounds are invalid")
    if args.queue_depth <= 0 or args.slo_ms <= 0 or args.timeout <= 0:
        parser.error("queue, SLO, and timeout bounds are invalid")
    if args.prefill_chunk is not None and args.prefill_chunk <= 0:
        parser.error("prefill chunk must be positive")

    clients: dict[str, StageV3Client] = {}
    batchers: dict[str, DeviceBatcher] = {}
    try:
        clients["op15"] = StageV3Client.connect(*args.op15, args.timeout)
        clients["op12"] = StageV3Client.connect(*args.op12, args.timeout)
        clients["tail"] = StageV3Client.connect(*args.tail, args.timeout)
        hellos = {name: client.hello() for name, client in clients.items()}
        for name in ("op15", "op12"):
            hello = hellos[name]
            if hello.layer_start != 0 or hello.capabilities & STAGE_V3_CAP_TERMINAL:
                raise ProtocolError(f"{name} is not a head stage")
            if hello.max_streams < args.requests_per_phone:
                raise ProtocolError(f"{name} lacks sequence capacity")
        if hellos["op15"].layer_end != hellos["op12"].layer_end:
            raise ProtocolError("phone head join boundaries differ")
        tail_hello = hellos["tail"]
        if not tail_hello.capabilities & STAGE_V3_CAP_TERMINAL:
            raise ProtocolError("desktop worker is not terminal")
        if tail_hello.layer_start != hellos["op15"].layer_end:
            raise ProtocolError("phone-to-tail boundary mismatch")
        request_count = 2 * args.requests_per_phone
        if tail_hello.max_streams < request_count:
            raise ProtocolError("tail lacks sequence capacity")

        for name, client in clients.items():
            hello = hellos[name]
            batchers[name] = DeviceBatcher(
                name, client,
                min(hello.n_batch, hello.n_ubatch),
                args.gather_us, args.queue_depth,
            )

        specs: list[RequestSpec] = []
        request_id = 1001
        tail_seq = 0
        for head_name in ("op15", "op12"):
            for head_seq in range(args.requests_per_phone):
                specs.append(RequestSpec(
                    request_id, 1, head_name, head_seq, tail_seq,
                    args.steps, args.slo_ms, None, args.prompt_tokens,
                    args.prefill_chunk,
                ))
                request_id += 1
                tail_seq += 1

        barrier = threading.Barrier(len(specs))
        outcomes: list[dict | None] = [None] * len(specs)
        errors: list[BaseException] = []

        def target(index: int, spec: RequestSpec) -> None:
            try:
                outcomes[index] = run_request(
                    spec, batchers[spec.head_name], batchers["tail"],
                    args.token, barrier, args.timeout,
                )
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=target, args=(index, spec), name=f"request-{spec.request_id}")
            for index, spec in enumerate(specs)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.timeout * args.steps)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("request threads did not finish")
        if errors:
            raise RuntimeError("request pipeline failed") from errors[0]
        completed = [outcome for outcome in outcomes if outcome is not None]
        if len(completed) != len(specs):
            raise RuntimeError("request conservation failed")

        for batcher in batchers.values():
            batcher.stop(args.timeout)
        batch_events = {
            name: list(batcher.events) for name, batcher in batchers.items()
        }
        batchers.clear()

        for spec in specs:
            clients[spec.head_name].remove(
                spec.head_seq, spec.request_id, spec.route_epoch,
            )
            clients["tail"].remove(
                spec.tail_seq, spec.request_id, spec.route_epoch,
            )
        final_status = {name: client.drain() for name, client in clients.items()}
        if any(status.active_sequences != 0 for status in final_status.values()):
            raise ProtocolError("live sequences remain after removal")

        token_sets = {tuple(outcome["tokens"]) for outcome in completed}
        all_slo_met = all(outcome["slo_met"] for outcome in completed)
        tokens_equal = len(token_sets) == 1
        verdict = "PASS" if all_slo_met and tokens_equal else (
            "CORRECTNESS_FAIL" if not tokens_equal else "SLO_FAIL"
        )
        report = {
            "schema": "s22-async-three-device-v1",
            "verdict": verdict,
            "boundary_layer": tail_hello.layer_start,
            "configuration": {
                "gather_us": args.gather_us,
                "queue_depth": args.queue_depth,
                "requests_per_phone": args.requests_per_phone,
                "session_end": args.session_end,
                "slo_ms": args.slo_ms,
                "steps": args.steps,
                "token": args.token,
                "prompt_tokens": list(args.prompt_tokens) if args.prompt_tokens else None,
                "prefill_chunk": args.prefill_chunk,
            },
            "workers": {
                name: asdict(hello) for name, hello in hellos.items()
            },
            "requests": completed,
            "batches": {
                name: summarize_batches(events)
                for name, events in batch_events.items()
            },
            "batch_events": batch_events,
            "cross_phone_tokens_equal": tokens_equal,
        }
        for client in clients.values():
            if args.session_end == "stop":
                client.stop()
            else:
                client.detach()
        report_bytes = json.dumps(
            report, sort_keys=True, separators=(",", ":"),
        ) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(report_bytes, encoding="ascii")
        print(report_bytes, end="")
        return 0 if report["verdict"] == "PASS" else 2
    finally:
        for batcher in batchers.values():
            try:
                batcher.stop(args.timeout)
            except BaseException:
                pass
        for client in clients.values():
            client.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ProtocolError, RuntimeError, TimeoutError, ValueError) as exc:
        print(json.dumps({"verdict": "FAIL", "error": str(exc)}, sort_keys=True))
        raise SystemExit(2)
