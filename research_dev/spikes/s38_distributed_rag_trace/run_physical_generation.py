#!/usr/bin/env python3
"""Run matched F16 RAG generation through CUDA and OP15 heads."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
S22 = HERE.parent / "s22_slo_overlap_pipeline"
S23 = HERE.parent / "s23_dense_trace_runtime"
S36 = HERE.parent / "s36_dynamic_cut_scheduler"
for dependency in (HERE, S36, S23, S22):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from cut_batcher import CutBatcher, LockedStageClient  # noqa: E402
from dynamic_route_runtime import (  # noqa: E402
    DynamicOutcome,
    DynamicRequest,
    DynamicRoute,
    DynamicRouteRunner,
    DynamicStage,
)
from runtime_support import SequenceSlotPool  # noqa: E402
from stage_v3_client import (  # noqa: E402
    Hello,
    STAGE_V3_CAP_RANGE,
    STAGE_V3_CAP_TERMINAL,
    StageV3Client,
)


SCHEMA = "s38-f16-physical-mixed-generation-v1"
SUBSET_SCHEMA = "s38-physical-generation-subset-v1"
HOST_MODEL_SHA = "bac4e29337273d1cb41aa36f4209ebcf63baa1162e6ab4dc55f6034b3d69b35a"
PHONE_MODEL_SHA = "a74845294bc1a3cba6ac26c747615d64d9127c904aae04c43c7ea2ebc1ea25c8"
WORKERS = ("cuda", "op15", "tail")
EXPECTED_RANGES = {"cuda": (0, 8), "op15": (0, 8), "tail": (8, 48)}


class PhysicalRunError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PhysicalRunError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_subset(path: Path, selected_ids: tuple[int, ...]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="ascii"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(value, dict) or value.get("schema") != SUBSET_SCHEMA:
        raise PhysicalRunError("unexpected physical subset schema")
    model = value.get("model")
    protocol = value.get("protocol")
    requests = value.get("requests")
    if (
        not isinstance(model, dict)
        or model.get("sha256") != HOST_MODEL_SHA
        or model.get("file_type") != 1
        or not isinstance(protocol, dict)
        or protocol.get("max_context") != 3072
        or not isinstance(requests, list)
        or not requests
    ):
        raise PhysicalRunError("physical subset identity or protocol mismatch")
    by_id: dict[int, dict[str, Any]] = {}
    for request in requests:
        request_id = request.get("request_id") if isinstance(request, dict) else None
        prompt = request.get("prompt_tokens") if isinstance(request, dict) else None
        stop_tokens = request.get("stop_tokens") if isinstance(request, dict) else None
        if (
            type(request_id) is not int
            or request_id <= 0
            or request_id in by_id
            or not isinstance(prompt, list)
            or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)
            or request.get("prompt_token_count") != len(prompt)
            or type(request.get("output_steps")) is not int
            or request["output_steps"] <= 0
            or len(prompt) + request["output_steps"] > protocol["max_context"]
            or not isinstance(stop_tokens, list)
            or not stop_tokens
            or any(type(token) is not int or token < 0 for token in stop_tokens)
            or len(set(stop_tokens)) != len(stop_tokens)
            or not isinstance(request.get("event_id"), str)
        ):
            raise PhysicalRunError("invalid request in physical subset")
        by_id[request_id] = request
    if not selected_ids:
        selected_ids = tuple(sorted(by_id))
    if len(set(selected_ids)) != len(selected_ids) or any(request_id not in by_id for request_id in selected_ids):
        raise PhysicalRunError("selected request IDs differ from the subset")
    return value, [by_id[request_id] for request_id in selected_ids]


def validate_hellos(hellos: Mapping[str, Hello]) -> None:
    if set(hellos) != set(WORKERS):
        raise PhysicalRunError("physical topology is incomplete")
    expected_hashes = {
        "cuda": HOST_MODEL_SHA,
        "tail": HOST_MODEL_SHA,
        "op15": PHONE_MODEL_SHA,
    }
    for name, hello in hellos.items():
        if (
            (hello.layer_start, hello.layer_end) != EXPECTED_RANGES[name]
            or (hello.n_layer, hello.n_embd) != (48, 3840)
            or hello.max_streams != 2
            or hello.n_ctx_seq != 6144
            or hello.file_type != 1
            or hello.model_sha256 != expected_hashes[name]
            or min(hello.n_batch, hello.n_ubatch, 64) < 64
            or not hello.capabilities & STAGE_V3_CAP_RANGE
            or bool(hello.capabilities & STAGE_V3_CAP_TERMINAL) != (name == "tail")
        ):
            raise PhysicalRunError(
                f"{name} hello differs from the frozen route: {asdict(hello)}"
            )


class PhysicalTopology:
    def __init__(
        self,
        endpoints: Mapping[str, tuple[str, int]],
        timeout_s: float,
        gather_us: int,
        knee: int,
        queue_depth: int,
    ) -> None:
        self.raw_clients: dict[str, StageV3Client] = {}
        self.clients: dict[str, LockedStageClient] = {}
        self.hellos: dict[str, Hello] = {}
        self.batchers: dict[str, CutBatcher] = {}
        self.stages: dict[str, DynamicStage] = {}
        self.routes: tuple[DynamicRoute, ...] = ()
        self.runner: DynamicRouteRunner | None = None
        try:
            for name in WORKERS:
                raw = StageV3Client.connect(*endpoints[name], timeout_s)
                self.raw_clients[name] = raw
                client = LockedStageClient(raw)
                self.clients[name] = client
                hello = client.hello()
                if not isinstance(hello, Hello):
                    raise PhysicalRunError(f"{name} returned an invalid hello")
                self.hellos[name] = hello
            validate_hellos(self.hellos)
            ranges = {
                "cuda": {8: (0, 8)},
                "op15": {8: (0, 8)},
                "tail": {8: (8, 48)},
            }
            for name in WORKERS:
                capacity = min(self.hellos[name].n_batch, self.hellos[name].n_ubatch, 64)
                self.batchers[name] = CutBatcher(
                    name,
                    self.clients[name],
                    ranges[name],
                    {8: min(knee, capacity)},
                    capacity,
                    gather_us,
                    queue_depth,
                )
                self.stages[name] = DynamicStage(
                    name,
                    self.clients[name],
                    self.hellos[name],
                    SequenceSlotPool(self.hellos[name].max_streams),
                    self.batchers[name],
                    name == "tail",
                )
            self.routes = (
                DynamicRoute("cuda-c8", self.stages["cuda"], self.stages["tail"], 8),
                DynamicRoute("op15-c8", self.stages["op15"], self.stages["tail"], 8),
            )
            self.runner = DynamicRouteRunner(self.routes)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for batcher in self.batchers.values():
            try:
                batcher.stop(30.0)
            except BaseException:
                pass
        for client in self.clients.values():
            try:
                client.close()
            except BaseException:
                pass

    def finish(self) -> dict[str, Any]:
        for name in WORKERS:
            self.batchers[name].stop(60.0)
        lifecycle = {}
        for name in WORKERS:
            status = self.clients[name].status()
            if status.active_sequences != 0:
                raise PhysicalRunError(f"{name} retained live sequences")
            drained = self.clients[name].drain()
            if drained.active_sequences != 0 or not drained.draining:
                raise PhysicalRunError(f"{name} drain failed")
            lifecycle[name] = {
                "before_drain": asdict(status),
                "after_drain": asdict(drained),
            }
        for name in WORKERS:
            self.clients[name].stop()
        return lifecycle


def parse_request_ids(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    try:
        request_ids = tuple(int(item, 10) for item in value.split(","))
    except ValueError as error:
        raise PhysicalRunError("request IDs must be comma-separated integers") from error
    if any(request_id <= 0 for request_id in request_ids) or len(set(request_ids)) != len(request_ids):
        raise PhysicalRunError("request IDs must be unique positive integers")
    return request_ids


def run_cohort(
    topology: PhysicalTopology,
    route_id: str,
    requests: Sequence[dict[str, Any]],
    id_base: int,
    timeout_s: float,
    gather_us: int,
    stagger_after_decode: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, object]]]]:
    if topology.runner is None:
        raise PhysicalRunError("physical runner is unavailable")
    before = {name: len(topology.batchers[name].events) for name in WORKERS}
    outcomes: dict[int, DynamicOutcome] = {}
    errors: list[BaseException] = []
    lock = threading.Lock()

    def execute(source: dict[str, Any]) -> None:
        try:
            request = DynamicRequest(
                request_id=id_base + source["request_id"],
                route_epoch=id_base + source["request_id"],
                route_id=route_id,
                prompt_tokens=tuple(source["prompt_tokens"]),
                output_steps=source["output_steps"],
                priority=1,
                slo_us=1 << 60,
                batch_wait_us=gather_us,
                prefill_quantum=64,
                stop_tokens=tuple(source["stop_tokens"]),
            )
            outcome = topology.runner.run(request, timeout_s)
            with lock:
                outcomes[source["request_id"]] = outcome
        except BaseException as error:
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=execute, args=(source,)) for source in requests]
    started_ns = time.monotonic_ns()
    if stagger_after_decode and len(threads) > 1:
        threads[0].start()
        head_name = route_id.split("-", 1)[0]
        trigger_deadline = time.monotonic() + min(timeout_s, 600.0)
        while time.monotonic() < trigger_deadline:
            head_events = topology.batchers[head_name].events[before[head_name]:]
            if any("decode" in event["phases"] for event in head_events):
                break
            if not threads[0].is_alive():
                raise PhysicalRunError(
                    f"{route_id} ended before the mixed-phase admission trigger"
                )
            time.sleep(0.001)
        else:
            raise PhysicalRunError(f"{route_id} decode admission trigger timed out")
        for thread in threads[1:]:
            thread.start()
    else:
        for thread in threads:
            thread.start()
    for thread in threads:
        thread.join(timeout_s + 30.0)
    if any(thread.is_alive() for thread in threads):
        raise PhysicalRunError(f"{route_id} cohort thread did not terminate")
    if errors:
        raise PhysicalRunError(f"{route_id} cohort failed") from errors[0]
    if set(outcomes) != {source["request_id"] for source in requests}:
        raise PhysicalRunError(f"{route_id} cohort lost an outcome")
    elapsed_us = (time.monotonic_ns() - started_ns) // 1000
    records = []
    for source in requests:
        outcome = outcomes[source["request_id"]]
        records.append({
            "cut": outcome.cut,
            "device": outcome.device,
            "event_id": source["event_id"],
            "finish_reason": outcome.finish_reason,
            "latency_us": outcome.latency_us,
            "output_tokens": list(outcome.output_tokens),
            "prompt_tokens": outcome.prompt_length,
            "request_id": source["request_id"],
            "route_id": outcome.route_id,
            "ttft_us": outcome.ttft_us,
        })
    records.sort(key=lambda item: item["request_id"])
    events = {
        name: topology.batchers[name].events[before[name]:]
        for name in WORKERS
    }
    records.append({"cohort_elapsed_us": elapsed_us})
    return records, events


def summarize_events(events: Mapping[str, Sequence[dict[str, object]]]) -> dict[str, Any]:
    result = {}
    for name in WORKERS:
        rows = list(events[name])
        sizes = [int(row["batch_size"]) for row in rows]
        result[name] = {
            "batches": len(rows),
            "compute_us": sum(int(row["compute_us"]) for row in rows),
            "max_batch": max(sizes, default=0),
            "mean_batch": statistics.fmean(sizes) if sizes else 0.0,
            "mixed_batches": sum(bool(row["mixed_phase"]) for row in rows),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-ids", default="1,4")
    parser.add_argument("--cuda", default="127.0.0.1:26420")
    parser.add_argument("--op15", default="127.0.0.1:26423")
    parser.add_argument("--tail", default="127.0.0.1:26421")
    parser.add_argument("--gather-us", type=int, default=500000)
    parser.add_argument("--knee", type=int, default=8)
    parser.add_argument("--queue-depth", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--allow-no-mixed", action="store_true")
    parser.add_argument("--require-eog", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise PhysicalRunError(f"output already exists: {args.output}")
    if not 0 <= args.gather_us <= 1_000_000 or not 1 <= args.knee <= 64:
        raise PhysicalRunError("invalid batching configuration")
    if not 1 <= args.queue_depth <= 65536 or not 1.0 <= args.timeout_s <= 7200.0:
        raise PhysicalRunError("invalid runtime bounds")

    def endpoint(value: str) -> tuple[str, int]:
        host, separator, raw_port = value.rpartition(":")
        if not separator or not host:
            raise PhysicalRunError(f"invalid endpoint: {value}")
        try:
            port = int(raw_port, 10)
        except ValueError as error:
            raise PhysicalRunError(f"invalid endpoint: {value}") from error
        if not 1 <= port <= 65535:
            raise PhysicalRunError(f"invalid endpoint: {value}")
        return host, port

    selected_ids = parse_request_ids(args.request_ids)
    subset_path = args.subset.resolve()
    subset, requests = load_subset(subset_path, selected_ids)
    endpoints = {
        "cuda": endpoint(args.cuda),
        "op15": endpoint(args.op15),
        "tail": endpoint(args.tail),
    }
    topology = PhysicalTopology(
        endpoints,
        min(args.timeout_s, 300.0),
        args.gather_us,
        args.knee,
        args.queue_depth,
    )
    lifecycle: dict[str, Any] | None = None
    try:
        control, control_events = run_cohort(
            topology,
            "cuda-c8",
            requests,
            100000,
            args.timeout_s,
            args.gather_us,
            len(requests) > 1,
        )
        treatment, treatment_events = run_cohort(
            topology,
            "op15-c8",
            requests,
            200000,
            args.timeout_s,
            args.gather_us,
            len(requests) > 1,
        )
        control_rows = {row["event_id"]: row for row in control if "event_id" in row}
        treatment_rows = {row["event_id"]: row for row in treatment if "event_id" in row}
        if set(control_rows) != set(treatment_rows):
            raise PhysicalRunError("control and treatment event sets differ")
        token_match = not any(
            control_rows[event_id]["output_tokens"] != treatment_rows[event_id]["output_tokens"]
            for event_id in control_rows
        )
        mixed_pass = any(
            bool(event["mixed_phase"]) for event in treatment_events["op15"]
        )
        source_by_event = {request["event_id"]: request for request in requests}
        eog_pass = all(
            row["finish_reason"] == "stop"
            and len(row["output_tokens"]) < source_by_event[event_id]["output_steps"]
            and row["output_tokens"][-1] in source_by_event[event_id]["stop_tokens"]
            for event_id, row in treatment_rows.items()
        )
        control_summary = summarize_events(control_events)
        treatment_summary = summarize_events(treatment_events)
        control_cuda_us = (
            control_summary["cuda"]["compute_us"]
            + control_summary["tail"]["compute_us"]
        )
        treatment_cuda_us = treatment_summary["tail"]["compute_us"]
        if control_cuda_us <= 0 or treatment_cuda_us <= 0:
            raise PhysicalRunError("CUDA compute evidence is empty")
        lifecycle = topology.finish()
        problems = []
        if not token_match:
            problems.append("OP15 route tokens differ from CUDA control")
        if not mixed_pass and not args.allow_no_mixed:
            problems.append("OP15 emitted no mixed prefill/decode batch")
        if args.require_eog and not eog_pass:
            problems.append("OP15 route did not terminate on a declared EOG token")
        status = "PASS" if not problems else "FAIL_MECHANICS_GATE"
        result = {
            "claims": {
                "bounded_generation_only": True,
                "eog_termination": eog_pass,
                "long_context_rag_prompts": True,
                "matched_f16_tokens": token_match,
                "phone_energy": "UNKNOWN",
                "phone_mixed_prefill_decode": mixed_pass,
                "rag_answer_quality": "NOT_MEASURED",
                "server_gpu_energy": "NOT_MEASURED",
            },
            "config": {
                "endpoints": {
                    "cuda": args.cuda,
                    "op15": args.op15,
                    "tail": args.tail,
                },
                "flash_attention": "disabled_by_GGML_DECODE_NO_FA",
                "gather_us": args.gather_us,
                "knee": args.knee,
                "queue_depth": args.queue_depth,
                "request_ids": list(selected_ids),
                "shared_kv_cells": 6144,
                "start_rule": (
                    "admit_long_prefill_after_first_head_decode_batch"
                    if len(requests) > 1 else "single_request_immediate"
                ),
                "stream_planning_context": 3072,
                "timeout_s": args.timeout_s,
            },
            "control": {
                "events": control_events,
                "outcomes": control,
                "summary": control_summary,
            },
            "hellos": {name: asdict(topology.hellos[name]) for name in WORKERS},
            "lifecycle": lifecycle,
            "metrics": {
                "control_selected_cuda_compute_us": control_cuda_us,
                "selected_cuda_compute_time_relief": 1.0 - treatment_cuda_us / control_cuda_us,
                "treatment_selected_cuda_compute_us": treatment_cuda_us,
            },
            "schema": SCHEMA,
            "status": status,
            "problems": problems,
            "subset": {
                "path": str(subset_path),
                "sha256": file_sha256(subset_path),
                "source_schema": subset["schema"],
            },
            "treatment": {
                "events": treatment_events,
                "outcomes": treatment,
                "summary": treatment_summary,
            },
        }
        content = canonical_json(result) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(args.output.resolve(), content)
        print(canonical_json({
            "output": str(args.output.resolve()),
            "selected_cuda_compute_time_relief": result["metrics"]["selected_cuda_compute_time_relief"],
            "sha256": hashlib.sha256(content.encode("ascii")).hexdigest(),
            "status": result["status"],
        }))
        return 0 if not problems else 2
    finally:
        if lifecycle is None:
            topology.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, PhysicalRunError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
