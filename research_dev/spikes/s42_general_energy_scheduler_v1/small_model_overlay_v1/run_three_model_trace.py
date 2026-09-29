#!/usr/bin/env python3
"""Run the focused BurstGPT plus Llama 1B trace on the physical desktop."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
import urllib.error


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
MIXED_DIR = S42_ROOT / "mixed_model_trace_v1"
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT), str(MIXED_DIR), str(HERE)]

import run_six_model_trace as physical  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    Decision,
    DeviceMemoryCapacity,
    ProfileBundle,
    Request,
    RuntimeExecutorBinding,
    RuntimeModelArtifact,
    RuntimePlacementSnapshot,
    UnifiedScheduler,
    decision_to_json,
)
from verify_small_model_overlay import validate as validate_trace  # noqa: E402


CONFIRMATION = "RUN_THREE_MODEL_MIXED_TRACE"
TRACE_SCHEMA = "s42-three-model-burstgpt-small-overlay-v1"
RESULT_SCHEMA = "s42-three-model-physical-result-v1"
EVENT_SCHEMA = "s42-three-model-event-v1"

QWEN14 = "qwen3-14b-q4_k_m"
GEMMA12 = "gemma-4-12b-it-q4_0"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
WORKLOAD_ID = "llama-1b-resident-task"
RUNTIME_PROFILE = (
    S42_ROOT
    / "whole_task_phone_v1/results/4060ti_op15_20260807"
    / "SCHEDULER_PROFILE_TAIL_REUSED.json"
)
GIB = 1024**3


def build_specs(args: argparse.Namespace) -> dict[str, physical.ModelSpec]:
    return {
        QWEN14: physical.ModelSpec(
            QWEN14, args.qwen14_model, "cuda", 18480, 4, 24576, 2048, 512
        ),
        GEMMA12: physical.ModelSpec(
            GEMMA12, args.gemma12_model, "cuda", 18487, 8, 32768, 4096, 512
        ),
        LLAMA1: physical.ModelSpec(
            LLAMA1,
            args.llama1_model,
            "cpu",
            18484,
            4,
            8192,
            2048,
            512,
            threads=4,
            cpus=args.llama1_cpus,
        ),
    }


def meminfo(text: str) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, raw = line.partition(":")
        if not separator or name not in {"MemTotal", "MemAvailable"}:
            continue
        fields = raw.strip().split()
        physical.run_trace.require(
            len(fields) == 2 and fields[1] == "kB",
            "memory snapshot units",
        )
        values[name] = int(fields[0]) * 1024
    physical.run_trace.require(
        set(values) == {"MemTotal", "MemAvailable"},
        "memory snapshot fields",
    )
    return values["MemTotal"], values["MemAvailable"]


def endpoint_ready(port: int) -> bool:
    try:
        return physical.run_trace.http_json(
            f"http://127.0.0.1:{port}/health", 1
        ).get("status") == "ok"
    except (OSError, ValueError, urllib.error.URLError):
        return False


def runtime_snapshot(
    args: argparse.Namespace,
    now_us: int,
    index: int,
    phone_available: bool,
) -> RuntimePlacementSnapshot:
    host_total, host_available = meminfo(
        Path("/proc/meminfo").read_text(encoding="ascii")
    )
    gpu = physical.run_trace.gpu_snapshot()
    capacities = {
        "host-ram": DeviceMemoryCapacity(
            "host-ram",
            host_total,
            host_total - host_available,
            min(2 * GIB, host_available),
        ),
        "cuda0-vram": DeviceMemoryCapacity(
            "cuda0-vram",
            gpu["memory_total_bytes"],
            gpu["memory_used_bytes"],
            min(512 * 1024**2, gpu["memory_free_bytes"]),
        ),
    }
    if phone_available:
        phone_total, phone_available_bytes = meminfo(
            physical.adb_su(args, "cat /proc/meminfo").stdout
        )
        capacities["op15-ram"] = DeviceMemoryCapacity(
            "op15-ram",
            phone_total,
            phone_total - phone_available_bytes,
            min(2 * GIB, phone_available_bytes),
        )
    return RuntimePlacementSnapshot(
        snapshot_id=f"three-model-live-{index}-{now_us}",
        captured_at_us=now_us,
        valid_until_us=now_us + 2_000_000,
        capacities=capacities,
    )


def runtime_request(row: dict[str, Any]) -> Request:
    return Request(
        request_id=f"mixed-{row['mixed_request_index']}",
        workload_id=WORKLOAD_ID,
        arrival_us=row["arrival_us"],
        deadline_us=row["arrival_us"] + row["slo_us"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        quality_requirement="bounded_numeric",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--server-cuda", type=Path, required=True)
    parser.add_argument("--server-cpu", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cpu-lib-dir", type=Path, required=True)
    parser.add_argument("--qwen14-model", type=Path, required=True)
    parser.add_argument("--gemma12-model", type=Path, required=True)
    parser.add_argument("--llama1-model", type=Path, required=True)
    parser.add_argument("--llama1-cpus", default="20-23")
    parser.add_argument("--runtime-profile", type=Path, default=RUNTIME_PROFILE)
    parser.add_argument("--adb-port", type=int, default=5037)
    parser.add_argument("--phone-serial", default="3C15AU002CL00000")
    parser.add_argument(
        "--phone-task-bin-dir", default="/data/local/tmp/llama-ubatch-op15/bin"
    )
    parser.add_argument(
        "--phone-task-model",
        default=(
            "/data/local/tmp/unifer/llamacpp/"
            "Llama-3.2-1B-Instruct-Q4_0.gguf"
        ),
    )
    parser.add_argument("--phone-task-port", type=int, default=18382)
    parser.add_argument("--phone-task-forward-port", type=int, default=29382)
    parser.add_argument(
        "--mode",
        choices=("server-baseline", "runtime-scheduler"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    physical.run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    physical.run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    validate_trace(args.requests, args.manifest)
    rows = physical.run_trace.read_jsonl(args.requests)
    physical.run_trace.require(
        len(rows) == 84
        and all(row.get("schema") == TRACE_SCHEMA for row in rows),
        "trace geometry",
    )
    manifest = physical.read_manifest(args.manifest)
    specs = build_specs(args)
    model_identities = physical.validate_model_files(specs, manifest)
    for path in (
        args.server_cuda,
        args.server_cpu,
        args.cuda_lib_dir,
        args.cpu_lib_dir,
    ):
        physical.run_trace.require(path.exists(), f"missing runtime: {path}")
    runtime_profile = None
    runtime_model = None
    if args.mode == "runtime-scheduler":
        physical.run_trace.require(
            args.runtime_profile.is_file(), "runtime cost profile"
        )
        runtime_profile = ProfileBundle.from_json(json.loads(
            args.runtime_profile.read_text(encoding="ascii")
        ))
        physical.run_trace.require(
            runtime_profile.trace_workload_map.get(LLAMA1) == WORKLOAD_ID,
            "runtime profile workload map",
        )
        inventory = manifest["model_inventory"][LLAMA1]
        runtime_model = RuntimeModelArtifact(
            LLAMA1,
            inventory["artifact_sha256"],
            inventory["artifact_bytes"],
        )

    args.output.mkdir(parents=True)
    events = physical.run_trace.EventWriter(args.output / "events.jsonl")
    processes: dict[str, physical.run_trace.CapturedProcess] = {}
    sampler: physical.DynamicSampler | None = None
    phone_process: physical.run_trace.CapturedProcess | None = None
    phone_remote_pid: int | None = None
    phone_spec: physical.ModelSpec | None = None
    phone_start_error: str | None = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    server_records: list[dict[str, Any]] = []
    live_scheduler: UnifiedScheduler | None = None
    scheduler_lock = threading.Lock()
    runtime_estimates: dict[int, dict[str, object]] = {}
    runtime_decisions: dict[int, Decision] = {}
    runtime_release_receipts: dict[int, dict[str, object]] = {}

    try:
        if args.mode == "runtime-scheduler":
            try:
                (
                    phone_process,
                    phone_load_ms,
                    phone_props,
                    phone_remote_pid,
                ) = physical.start_phone_task_server(
                    args, manifest, args.output
                )
                phone_spec = physical.ModelSpec(
                    LLAMA1,
                    args.llama1_model,
                    "phone-adreno",
                    args.phone_task_forward_port,
                    1,
                    4096,
                    1024,
                    256,
                )
                server_records.append({
                    "backend": "phone-adreno",
                    "load_ms": phone_load_ms,
                    "model_id": LLAMA1,
                    "props": phone_props,
                    "stage": "runtime_discovered_resident_executor",
                })
            except BaseException as error:
                phone_start_error = f"{type(error).__name__}: {error}"
                phone_process = None
                phone_remote_pid = None
                phone_spec = None

        for model_id in (LLAMA1, QWEN14):
            spec = specs[model_id]
            process, load_ms, props = physical.start_server(
                spec,
                args.server_cuda,
                args.server_cpu,
                args.cuda_lib_dir,
                args.cpu_lib_dir,
                args.output,
                f"server-{model_id}",
            )
            processes[model_id] = process
            record = {
                "backend": spec.backend,
                "load_ms": load_ms,
                "model_id": model_id,
                "props": props,
                "stage": "initial",
            }
            server_records.append(record)

        for model_id in (LLAMA1, QWEN14):
            candidates = [
                row for row in rows
                if row["execution_model_id"] == model_id
            ]
            warm = dict(min(candidates, key=lambda row: row["input_tokens"]))
            warm["output_tokens"] = 2
            started_ns = time.monotonic_ns()
            physical.text_completion(
                specs[model_id],
                warm,
                args.output / f"warm-{model_id}.raw",
                lambda _: None,
            )
            next(
                record for record in server_records
                if record["model_id"] == model_id
                and record["stage"] == "initial"
            )["warm_ms"] = (time.monotonic_ns() - started_ns) / 1e6

        if phone_spec is not None:
            candidates = [
                row for row in rows
                if row["execution_model_id"] == LLAMA1
            ]
            warm = dict(min(candidates, key=lambda row: row["input_tokens"]))
            warm["output_tokens"] = 2
            started_ns = time.monotonic_ns()
            physical.text_completion(
                phone_spec,
                warm,
                args.output / f"warm-{LLAMA1}-phone.raw",
                lambda _: None,
            )
            server_records[0]["warm_ms"] = (
                time.monotonic_ns() - started_ns
            ) / 1e6

        if runtime_profile is not None:
            live_scheduler = UnifiedScheduler((runtime_profile,), "enforce")
            live_scheduler.set_resource_ready("cuda0", False, 0)
            for resource_id in ("op15-adreno", "usb-token-rpc"):
                live_scheduler.set_resource_ready(
                    resource_id, phone_spec is not None, 0
                )

        sampler = physical.DynamicSampler(args.output)
        for model_id, process in processes.items():
            sampler.set_pid(model_id, process.pid)
        if phone_process is not None:
            sampler.set_pid("phone-adreno", phone_process.pid)
        sampler.start()
        time.sleep(0.5)
        paid_start_ns = time.monotonic_ns()
        events.write({
            "kind": "trace_start",
            "policy": args.mode,
            "schema": EVENT_SCHEMA,
            "t_ns": paid_start_ns,
        })

        def execute_request(
            row: dict[str, Any],
            spec: physical.ModelSpec,
            route: str,
            scheduler_decision: Decision | None = None,
        ) -> None:
            first: list[int] = []
            index = row["mixed_request_index"]
            try:
                if scheduler_decision is not None:
                    target_dispatch_ns = (
                        paid_start_ns + scheduler_decision.start_us * 1000
                    )
                    while time.monotonic_ns() < target_dispatch_ns:
                        time.sleep(min(
                            (
                                target_dispatch_ns - time.monotonic_ns()
                            ) / 1e9,
                            0.01,
                        ))
                dispatch_ns = time.monotonic_ns()
                value = physical.text_completion(
                    spec,
                    row,
                    args.output / f"stream-{route}-{index:03d}.raw",
                    first.append,
                )
                physical.run_trace.require(
                    len(first) == 1, "first-token accounting"
                )
                completion_ns = time.monotonic_ns()
                scheduled_ns = paid_start_ns + row["arrival_us"] * 1000
                release_receipt = None
                if scheduler_decision is not None:
                    actual_end_us = (
                        completion_ns - paid_start_ns
                    ) // 1000
                    expired_phases = []
                    late = []
                    released = []
                    with scheduler_lock:
                        physical.run_trace.require(
                            live_scheduler is not None,
                            "runtime scheduler disappeared",
                        )
                        for lease in scheduler_decision.leases:
                            release_us = max(
                                lease.start_us,
                                min(actual_end_us, lease.reserved_until_us),
                            )
                            live_scheduler.release(lease.token, release_us)
                            released.append(lease.token)
                            if actual_end_us > lease.reserved_until_us:
                                if (
                                    lease.predicted_end_us
                                    < scheduler_decision.finish_us
                                ):
                                    expired_phases.append(lease.token)
                                else:
                                    late.append(lease.token)
                    release_receipt = {
                        "actual_end_us": actual_end_us,
                        "expired_phase_tokens": expired_phases,
                        "late_tokens": late,
                        "released_tokens": released,
                        "status": "late" if late else "released",
                    }
                    with result_lock:
                        runtime_release_receipts[index] = release_receipt
                record = {
                    "completion_ns": completion_ns,
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "execution_model_id": row["execution_model_id"],
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "mixed_request_index": index,
                    "output_text": value["output_text"],
                    "output_tokens": row["output_tokens"],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "prompt_transport": row["prompt_transport"],
                    "quality_pass": None,
                    "route": route,
                    "runtime_cost_estimates": runtime_estimates.get(index),
                    "runtime_prompt_tokens": value["runtime_prompt_tokens"],
                    "scheduler_decision": (
                        None
                        if scheduler_decision is None
                        else decision_to_json(scheduler_decision)
                    ),
                    "scheduler_release": release_receipt,
                    "scheduled_arrival_ns": scheduled_ns,
                    "slo_met": (
                        completion_ns
                        <= scheduled_ns + row["slo_us"] * 1000
                    ),
                    "slo_us": row["slo_us"],
                    "stream_request_index": row["stream_request_index"],
                    "stream_token_ids_complete": value[
                        "stream_token_ids_complete"
                    ],
                    "tokens": value["tokens"],
                }
                with result_lock:
                    results.append(record)
                events.write({"kind": "request_complete", **record})
            except BaseException as error:
                if scheduler_decision is not None and live_scheduler is not None:
                    with scheduler_lock:
                        live_scheduler.cancel(
                            scheduler_decision.request_id,
                            max(
                                row["arrival_us"],
                                (
                                    time.monotonic_ns() - paid_start_ns
                                ) // 1000,
                            ),
                        )
                with result_lock:
                    errors.append(
                        f"{index}: {type(error).__name__}: {error}"
                    )

        held_gemma: list[dict[str, Any]] = []
        futures: dict[str, list[concurrent.futures.Future[None]]] = {
            QWEN14: [],
            GEMMA12: [],
            LLAMA1: [],
        }
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=84, thread_name_prefix="three-model"
        ) as pool:
            for row in rows:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while time.monotonic_ns() < target_ns:
                    time.sleep(min(
                        (target_ns - time.monotonic_ns()) / 1e9,
                        0.01,
                    ))
                model_id = row["execution_model_id"]
                events.write({
                    "execution_model_id": model_id,
                    "kind": "request_arrival",
                    "mixed_request_index": row["mixed_request_index"],
                    "schema": EVENT_SCHEMA,
                    "t_ns": time.monotonic_ns(),
                })
                if model_id == GEMMA12:
                    held_gemma.append(row)
                    events.write({
                        "execution_model_id": model_id,
                        "kind": "request_held_for_residency",
                        "mixed_request_index": row["mixed_request_index"],
                        "schema": EVENT_SCHEMA,
                        "t_ns": time.monotonic_ns(),
                    })
                    continue
                if model_id == LLAMA1 and live_scheduler is not None:
                    physical.run_trace.require(
                        runtime_model is not None,
                        "runtime model identity",
                    )
                    index = row["mixed_request_index"]
                    snapshot_now_us = max(
                        row["arrival_us"],
                        (
                            time.monotonic_ns() - paid_start_ns
                        ) // 1000,
                    )
                    cpu_ready = endpoint_ready(specs[LLAMA1].port)
                    phone_ready = (
                        phone_spec is not None
                        and endpoint_ready(phone_spec.port)
                    )
                    snapshot = runtime_snapshot(
                        args,
                        snapshot_now_us,
                        index,
                        phone_spec is not None,
                    )
                    bindings = [RuntimeExecutorBinding(
                        executor_id=f"http://127.0.0.1:{specs[LLAMA1].port}",
                        route_id="desktop-cpu",
                        model_id=LLAMA1,
                        artifact_sha256=runtime_model.artifact_sha256,
                        artifact_bytes=runtime_model.artifact_bytes,
                        backend="cpu",
                        resource_ids=("desktop-cpu",),
                        memory_resource_id="host-ram",
                        resident=True,
                        ready=cpu_ready,
                    )]
                    if phone_spec is not None:
                        bindings.append(RuntimeExecutorBinding(
                            executor_id=(
                                "http://127.0.0.1:"
                                f"{phone_spec.port}"
                            ),
                            route_id="phone-adreno",
                            model_id=LLAMA1,
                            artifact_sha256=runtime_model.artifact_sha256,
                            artifact_bytes=runtime_model.artifact_bytes,
                            backend="phone-adreno",
                            resource_ids=(
                                "op15-adreno", "usb-token-rpc"
                            ),
                            memory_resource_id="op15-ram",
                            resident=True,
                            ready=phone_ready,
                        ))
                    decision_now_us = max(
                        snapshot_now_us,
                        (
                            time.monotonic_ns() - paid_start_ns
                        ) // 1000,
                    )
                    with scheduler_lock:
                        for resource_id in (
                            "op15-adreno", "usb-token-rpc"
                        ):
                            live_scheduler.set_resource_ready(
                                resource_id, phone_ready, decision_now_us
                            )
                        estimate_set = live_scheduler.estimate_runtime_costs(
                            runtime_request(row),
                            runtime_model,
                            tuple(bindings),
                            snapshot=snapshot,
                            now_us=decision_now_us,
                        )
                        decision = live_scheduler.schedule_runtime_costs(
                            runtime_request(row),
                            estimate_set,
                            runtime_now_us=decision_now_us,
                        )
                        runtime_decisions[index] = decision
                    estimate_json = estimate_set.to_json()
                    selected_estimate = next(
                        estimate for estimate in estimate_set.estimates
                        if estimate.route_id == decision.route_id
                    )
                    physical.run_trace.require(
                        selected_estimate.admitted,
                        "selected runtime estimate admission",
                    )
                    runtime_estimates[index] = estimate_json
                    events.write({
                        "cost_estimates": estimate_json,
                        "decision": decision_to_json(decision),
                        "kind": "runtime_scheduler_decision",
                        "mixed_request_index": index,
                        "schema": EVENT_SCHEMA,
                        "t_ns": time.monotonic_ns(),
                    })
                    if decision.route_id == "phone-adreno":
                        physical.run_trace.require(
                            phone_spec is not None,
                            "selected phone executor",
                        )
                        selected_spec = phone_spec
                    else:
                        physical.run_trace.require(
                            decision.route_id == "desktop-cpu",
                            "runtime executor binding",
                        )
                        selected_spec = specs[LLAMA1]
                    futures[model_id].append(pool.submit(
                        execute_request,
                        row,
                        selected_spec,
                        f"{model_id}-{decision.route_id}",
                        decision,
                    ))
                    continue
                spec = specs[model_id]
                futures[model_id].append(pool.submit(
                    execute_request,
                    row,
                    spec,
                    f"{model_id}-{spec.backend}",
                ))

            for future in futures[QWEN14]:
                future.result(timeout=3600)
            physical.run_trace.require(
                any(
                    row["execution_model_id"] == QWEN14
                    for row in results
                ),
                "Qwen completion",
            )
            events.write({
                "kind": "cuda_model_drained",
                "model_id": QWEN14,
                "schema": EVENT_SCHEMA,
                "t_ns": time.monotonic_ns(),
            })
            processes.pop(QWEN14).terminate()
            sampler.set_pid(QWEN14, 0)

            spec = specs[GEMMA12]
            process, load_ms, props = physical.start_server(
                spec,
                args.server_cuda,
                args.server_cpu,
                args.cuda_lib_dir,
                args.cpu_lib_dir,
                args.output,
                f"server-{GEMMA12}",
            )
            processes[GEMMA12] = process
            sampler.set_pid(GEMMA12, process.pid)
            warm = dict(min(
                held_gemma, key=lambda row: row["input_tokens"]
            ))
            warm["output_tokens"] = 2
            warm_started_ns = time.monotonic_ns()
            physical.text_completion(
                spec,
                warm,
                args.output / f"warm-{GEMMA12}.raw",
                lambda _: None,
            )
            ready_ns = time.monotonic_ns()
            warm_ms = (ready_ns - warm_started_ns) / 1e6
            server_records.append({
                "backend": spec.backend,
                "load_ms": load_ms,
                "model_id": GEMMA12,
                "props": props,
                "ready_s": (ready_ns - paid_start_ns) / 1e9,
                "stage": "promotion",
                "warm_ms": warm_ms,
            })
            events.write({
                "kind": "cuda_model_ready",
                "load_ms": load_ms,
                "model_id": GEMMA12,
                "schema": EVENT_SCHEMA,
                "t_ns": ready_ns,
                "warm_ms": warm_ms,
            })
            for row in held_gemma:
                futures[GEMMA12].append(pool.submit(
                    execute_request,
                    row,
                    spec,
                    f"{GEMMA12}-cuda",
                ))
            for model_id in (GEMMA12, LLAMA1):
                for future in futures[model_id]:
                    future.result(timeout=3600)

        physical.run_trace.require(
            not errors, "request errors: " + "; ".join(errors)
        )
        physical.run_trace.require(len(results) == 84, "request conservation")
        physical.run_trace.require(
            sorted(row["mixed_request_index"] for row in results)
            == list(range(84)),
            "mixed request identity conservation",
        )
        llama_indices = {
            row["mixed_request_index"] for row in rows
            if row["execution_model_id"] == LLAMA1
        }
        if live_scheduler is not None:
            physical.run_trace.require(
                set(runtime_estimates) == llama_indices
                and set(runtime_decisions) == llama_indices
                and set(runtime_release_receipts) == llama_indices,
                "runtime scheduler coverage",
            )
        paid_end_ns = max(row["completion_ns"] for row in results)
        time.sleep(0.5)
        sampler.stop()
        samples = list(sampler.rows)
        sampler = None
        for process in processes.values():
            process.terminate()
        processes.clear()
        if phone_process is not None:
            phone_process.terminate()
            phone_process = None
        if phone_remote_pid is not None:
            physical.stop_phone_task_server(args, phone_remote_pid)
            phone_remote_pid = None

        route_counts: dict[str, int] = {}
        for decision in runtime_decisions.values():
            route_counts[decision.route_id] = (
                route_counts.get(decision.route_id, 0) + 1
            )

        result = {
            "manifest_sha256": physical.digest_file(args.manifest),
            "metrics": physical.metrics(results, paid_start_ns),
            "model_identities": model_identities,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "policy": {
                "cpu_resident_models": [LLAMA1],
                "gpu_sequence": [QWEN14, GEMMA12],
                "phone_enabled": phone_spec is not None,
                "policy_id": args.mode,
            },
            "request_results": sorted(
                results, key=lambda row: row["mixed_request_index"]
            ),
            "resources": {
                "gpu_memory_used_max_bytes": max(
                    row["gpu"]["memory_used_bytes"] for row in samples
                ),
                "gpu_power_w": physical.run_trace.stats([
                    row["gpu"]["power_mw"] / 1000 for row in samples
                ]),
                "gpu_utilization_pct": physical.run_trace.stats([
                    row["gpu"]["utilization_pct"] for row in samples
                ]),
                "process_swap_max_bytes": {
                    label: max(
                        row["pids"].get(label, {}).get("swap_bytes", 0)
                        for row in samples
                    )
                    for label in sorted({
                        label for row in samples for label in row["pids"]
                    })
                },
                "system_available_min_bytes": min(
                    row["system"]["available_bytes"] for row in samples
                ),
                "system_swap_free_min_bytes": min(
                    row["system"]["swap_free_bytes"] for row in samples
                ),
            },
            "schema": RESULT_SCHEMA,
            "scheduler_runtime": {
                "cost_estimates": {
                    str(index): runtime_estimates[index]
                    for index in sorted(runtime_estimates)
                },
                "decisions": [
                    {
                        **decision_to_json(runtime_decisions[index]),
                        "mixed_request_index": index,
                    }
                    for index in sorted(runtime_decisions)
                ],
                "enabled": live_scheduler is not None,
                "phone_start_error": phone_start_error,
                "profile_sha256": (
                    None
                    if runtime_profile is None
                    else physical.digest_file(args.runtime_profile)
                ),
                "release_receipts": {
                    str(index): runtime_release_receipts[index]
                    for index in sorted(runtime_release_receipts)
                },
                "resource_snapshot_at_end": (
                    None
                    if live_scheduler is None
                    else live_scheduler.resource_snapshot(
                        (paid_end_ns - paid_start_ns) // 1000
                    )
                ),
                "route_counts": dict(sorted(route_counts.items())),
            },
            "server_records": server_records,
            "server_energy": physical.run_trace.server_energy_summary(
                samples, paid_start_ns, paid_end_ns
            ),
            "status": "PASS",
            "trace_sha256": physical.digest_file(args.requests),
        }
        physical.run_trace.write_json(args.output / "RESULT.json", result)
        print(json.dumps({
            "duration_s": result["metrics"]["duration_s"],
            "server_energy_j": result["server_energy"][
                "server_compute_device_energy_j"
            ],
            "slo_met": result["metrics"]["slo_met"],
            "status": "PASS",
        }, sort_keys=True))
        return 0
    except BaseException as error:
        physical.run_trace.write_json(args.output / "FAILURE.json", {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s42-three-model-physical-failure-v1",
            "status": "FAIL",
        })
        raise
    finally:
        events.close()
        if sampler is not None:
            sampler.stop()
        for process in processes.values():
            process.terminate()
        if phone_process is not None:
            phone_process.terminate()
        if phone_remote_pid is not None:
            physical.stop_phone_task_server(args, phone_remote_pid)


if __name__ == "__main__":
    raise SystemExit(main())
