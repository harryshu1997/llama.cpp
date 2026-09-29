#!/usr/bin/env python3
"""Run the immutable six-model trace with explicit model residency phases."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parent
REPO_ROOT = HERE.parents[3]
DEFAULT_BURST_DIR = (
    REPO_ROOT
    / "research_dev/spikes/s41_gemma_qwen_continuous_baseline"
    / "tp_operator_split_v1/burstgpt_gpu_cpu_op15_v1"
)
BURST_DIR = Path(os.environ.get("S41_BURST_DIR", DEFAULT_BURST_DIR))
S39_DIR = REPO_ROOT / "research_dev/spikes/s39_desktop_swap_baseline"
sys.path.insert(0, str(BURST_DIR))
sys.path.insert(0, str(S39_DIR))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(S42_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import cache_control  # noqa: E402
import plan_live_vq_trace  # noqa: E402
import run_trace  # noqa: E402
import run_server_trace  # noqa: E402
from research_dev.scheduler import (  # noqa: E402
    Decision,
    ProfileBundle,
    UnifiedScheduler,
    decision_to_json,
    residency_sequence_to_json,
    solve_residency_problem,
)
from verify_mixed_trace import validate as validate_trace  # noqa: E402


CONFIRMATION = "RUN_SIX_MODEL_MIXED_TRACE"
TRACE_SCHEMA = "s42-six-model-burstgpt-mixed-v1"
RESULT_SCHEMA = "s42-six-model-physical-result-v1"
PLAN_SCHEMA = "s42-six-model-residency-plan-v1"
LIVE_PLAN_SCHEMA = plan_live_vq_trace.SCHEMA

QWEN14 = "qwen3-14b-q4_k_m"
GEMMA12 = "gemma-4-12b-it-q4_0"
QWEN06 = "qwen3-0.6b-q8_0"
LLAMA1 = "llama-3.2-1b-instruct-q4_0"
QWEN8 = "qwen3-8b-q8_0"
GEMMA_E2B = "gemma-4-e2b-it-q8_0-vlm"
PHONE_ROUTE = f"{GEMMA12}-cpu-op15-ffn"
TASK_PHONE_ROUTE = plan_live_vq_trace.PHONE_ROUTE
UNIFIED_ROUTE = plan_live_vq_trace.UNIFIED_ROUTE


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_path: Path
    backend: str
    port: int
    parallel: int
    ctx_size: int
    batch_size: int
    ubatch_size: int
    threads: int = 0
    cpus: str | None = None
    mmproj_path: Path | None = None


@dataclass(frozen=True)
class ImageTransport:
    source_bytes: int
    source_media_type: str
    source_sha256: str
    transport_media_type: str
    transport_payload: bytes
    transport_sha256: str
    transcoded: bool


class DynamicSampler:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.pids: dict[str, int] = {}
        self.lock = threading.Lock()
        self.rows: list[dict[str, Any]] = []
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def set_pid(self, label: str, pid: int) -> None:
        with self.lock:
            self.pids[label] = pid

    def start(self) -> None:
        self.thread.start()

    def latest_gpu_snapshot(self, max_age_ns: int) -> dict[str, Any]:
        run_trace.require(max_age_ns > 0, "GPU sampler maximum age")
        with self.lock:
            run_trace.require(self.rows, "GPU sampler has no snapshot")
            row = self.rows[-1]
            gpu = dict(row["gpu"])
            captured_at_ns = int(gpu["sample_t_ns"])
        run_trace.require(
            time.monotonic_ns() - captured_at_ns <= max_age_ns,
            "GPU sampler snapshot is stale",
        )
        return gpu

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)
        run_trace.require(not self.thread.is_alive(), "sampler stop")
        run_trace.require(self.error is None, f"sampler: {self.error}")

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    pids = dict(self.pids)
                before_ns = time.monotonic_ns()
                gpu = run_trace.gpu_snapshot()
                after_ns = time.monotonic_ns()
                gpu["sample_t_ns"] = (before_ns + after_ns) // 2
                try:
                    rapl = run_trace.rapl_package_snapshot()
                except (
                    FileNotFoundError,
                    PermissionError,
                    run_trace.RunError,
                    ValueError,
                ):
                    rapl = None
                row = {
                    "gpu": gpu,
                    "pids": {
                        label: run_trace.proc_status_or_zero(pid)
                        for label, pid in pids.items()
                    },
                    "rapl_package": rapl,
                    "schema": "s42-six-model-resource-sample-v1",
                    "system": run_trace.system_memory(),
                    "t_ns": time.monotonic_ns(),
                }
                with self.lock:
                    self.rows.append(row)
                self.stop_event.wait(0.2)
            with (self.output / "resource-samples.jsonl").open("xb") as stream:
                for row in self.rows:
                    stream.write(run_trace.canonical(row))
        except BaseException as error:
            self.error = f"{type(error).__name__}: {error}"


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def completed_prefixed_json(
    lines: list[str], prefix: str
) -> dict[str, Any]:
    matches = [line for line in lines if line.startswith(prefix + "{")]
    run_trace.require(
        len(matches) == 1,
        f"completed summary count for {prefix.strip()}",
    )
    value = json.loads(matches[0][len(prefix):])
    run_trace.require(type(value) is dict, "completed summary object")
    return value


def read_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    run_trace.require(type(value) is dict, "manifest object")
    return value


def validate_live_plan(
    plan: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    routes = plan.get("request_routes")
    run_trace.require(
        plan.get("mode") in {"control", "adaptive"}
        and type(routes) is dict
        and set(routes) == {str(index) for index in range(len(rows))},
        "live scheduler plan coverage",
    )
    run_trace.require(
        set(routes.values()) == {UNIFIED_ROUTE},
        "live scheduler route compatibility",
    )
    phone = plan.get("phone")
    scheduler = plan.get("scheduler")
    scheduled_model_ids = list(plan_live_vq_trace.trace_model_ids(rows))
    gpu = plan.get("gpu")
    sequence = None if type(gpu) is not dict else gpu.get("sequence")
    residency_plan = (
        None if type(gpu) is not dict else gpu.get("residency_plan")
    )
    residency_problem = (
        None if type(gpu) is not dict else gpu.get("residency_problem")
    )
    try:
        recomputed_residency = (
            None
            if type(residency_problem) is not dict
            else residency_sequence_to_json(
                solve_residency_problem(residency_problem)
            )
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise run_trace.RunError(
            f"live GPU residency problem: {exc}"
        ) from exc
    phases = (
        None if type(residency_plan) is not dict
        else residency_plan.get("phases")
    )
    run_trace.require(
        type(phone) is dict
        and type(scheduler) is dict
        and scheduler.get("enabled") is True
        and scheduler.get("policy_mode") == plan["mode"]
        and scheduler.get("scheduled_model_ids") == scheduled_model_ids
        and scheduler.get("alternative_model_ids") == [LLAMA1]
        and scheduler.get("single_route_model_ids")
            == [model_id for model_id in scheduled_model_ids if model_id != LLAMA1]
        and type(sequence) is list
        and set(sequence) == {QWEN14, QWEN8, GEMMA12}
        and scheduler.get("deferred_until_ready_model_ids")
            == sequence[1:]
        and type(residency_plan) is dict
        and residency_plan == recomputed_residency
        and residency_plan.get("model_order") == sequence
        and type(phases) is list
        and [phase.get("model_id") for phase in phases] == sequence
        and phases[0].get("switch_latency_us") == 0
        and phases[0].get("residency_id")
            == residency_plan.get("initial_residency_id"),
        "live scheduler configuration",
    )
    selected = phone.get("predicted_selected_mixed_request_indices")
    raw_profile = scheduler.get("profile")
    executors = scheduler.get("route_executors")
    run_trace.require(
        type(raw_profile) is dict and type(executors) is dict,
        "live scheduler profile",
    )
    profile = ProfileBundle.from_json(raw_profile)
    run_trace.require(
        set(executors) == {route.route_id for route in profile.routes},
        "live scheduler route executor coverage",
    )
    expected_workloads = {
        plan_live_vq_trace.request_workload_id(
            row["mixed_request_index"]
        )
        for row in rows
    }
    run_trace.require(
        {route.workload_id for route in profile.routes} == expected_workloads,
        "live scheduler workload coverage",
    )
    fixed_executors = {
        QWEN14: {f"{QWEN14}-cuda"},
        QWEN8: {f"{QWEN8}-cuda"},
        GEMMA12: {f"{GEMMA12}-cuda"},
        QWEN06: {f"{QWEN06}-cpu"},
        GEMMA_E2B: {f"{GEMMA_E2B}-cpu"},
    }
    for row in rows:
        model_id = row["execution_model_id"]
        workload_id = plan_live_vq_trace.request_workload_id(
            row["mixed_request_index"]
        )
        actual = {
            executors[route.route_id]
            for route in profile.routes
            if route.workload_id == workload_id
        }
        expected = fixed_executors.get(model_id)
        if model_id == LLAMA1:
            expected = {f"{LLAMA1}-cpu"}
            if plan["mode"] == "adaptive":
                expected.add(TASK_PHONE_ROUTE)
        run_trace.require(
            actual == expected,
            "live scheduler executor compatibility",
        )
    if plan["mode"] == "control":
        run_trace.require(
            phone.get("route_kind") == "none"
            and selected == []
            and TASK_PHONE_ROUTE not in set(executors.values()),
            "control live scheduler plan",
        )
        return

    run_trace.require(
        phone.get("route_kind") == "whole-task-adreno"
        and phone.get("model_id") == LLAMA1
        and phone.get("route_id") == TASK_PHONE_ROUTE
        and phone.get("preloaded_before_paid_start") is True
        and phone.get("scheduler_effective_lanes") == 1
        and type(selected) is list
        and selected == sorted(selected),
        "adaptive live scheduler plan",
    )
    run_trace.require(
        f"{LLAMA1}-cuda" not in set(executors.values()),
        "adaptive route executor map",
    )


def read_scheduler_plan(
    path: Path,
    rows: list[dict[str, Any]],
    trace_sha256: str,
) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="ascii"))
    run_trace.require(
        type(plan) is dict
        and plan.get("schema") in {PLAN_SCHEMA, LIVE_PLAN_SCHEMA}
        and plan.get("status") == "PASS",
        "scheduler plan status",
    )
    expected_hash = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    run_trace.require(
        type(expected_hash) is str
        and hashlib.sha256(run_trace.canonical(unsigned)).hexdigest()
            == expected_hash,
        "scheduler plan identity",
    )
    run_trace.require(plan.get("trace_sha256") == trace_sha256, "plan trace")
    if plan["schema"] == LIVE_PLAN_SCHEMA:
        validate_live_plan(plan, rows)
        return plan
    run_trace.require(
        plan.get("gpu", {}).get("sequence") == [QWEN14, QWEN8, GEMMA12],
        "plan GPU sequence",
    )
    routes = plan.get("request_routes")
    run_trace.require(
        type(routes) is dict
        and set(routes) == {str(index) for index in range(len(rows))},
        "plan request coverage",
    )
    allowed = {
        QWEN14: {f"{QWEN14}-cuda"},
        QWEN8: {f"{QWEN8}-cuda"},
        GEMMA12: {f"{GEMMA12}-cuda", PHONE_ROUTE},
        QWEN06: {f"{QWEN06}-cpu"},
        LLAMA1: {f"{LLAMA1}-cpu"},
        GEMMA_E2B: {f"{GEMMA_E2B}-cpu"},
    }
    for row in rows:
        route = routes[str(row["mixed_request_index"])]
        run_trace.require(
            route in allowed[row["execution_model_id"]],
            "plan route compatibility",
        )
    phone = plan.get("phone", {})
    selected = phone.get("selected_mixed_request_indices")
    run_trace.require(
        phone.get("model_id") == GEMMA12
        and phone.get("route_id") == PHONE_ROUTE
        and phone.get("preloaded_before_paid_start") is True
        and phone.get("desktop_companion_retire_when_idle_before_cuda")
            == GEMMA12
        and phone.get("scheduler_effective_lanes") == 2
        and type(selected) is list
        and selected
        and selected == sorted(selected)
        and all(type(index) is int for index in selected),
        "plan phone residency",
    )
    run_trace.require(
        {
            int(index) for index, route in routes.items()
            if route == PHONE_ROUTE
        } == set(selected),
        "plan phone assignment",
    )
    split = phone.get("split_policy", {})
    run_trace.require(
        split.get("id") == "i3-hidden-wait"
        and split.get("io") == "f16"
        and split.get("layer_mask") == "0x0000ffffffffffff"
        and split.get("max_columns") == 11136
        and split.get("timeout_ms") == 35000
        and split.get("table")
            == "1:9664,3:8192,8:4096,128:8192,512:11136",
        "plan phone split policy",
    )
    return plan


def phone_server_args(
    args: argparse.Namespace, plan: dict[str, Any]
) -> argparse.Namespace:
    split = plan["phone"]["split_policy"]
    return SimpleNamespace(
        bridge_port=args.bridge_port,
        cold_batch_size=4096,
        cold_cpus=args.gemma_phone_cpus,
        cold_ctx_size=32768,
        cold_lib_dir=args.gemma_phone_lib_dir,
        cold_model=args.gemma12_model,
        cold_n_gpu_layers=0,
        cold_parallel=8,
        cold_port=args.gemma_phone_port,
        cold_repack="default",
        cold_server=args.gemma_phone_server,
        cold_threads=-1,
        cold_ubatch_size=512,
        ffn_layer_mask=split["layer_mask"],
        ffn_timeout_ms=split["timeout_ms"],
        max_columns=split["max_columns"],
        mode="op15",
        split_io=split["io"],
        split_policy=split["table"],
    )


def phone_route_kind(plan: dict[str, Any]) -> str:
    return plan["phone"].get("route_kind", "cpu-op15-ffn")


def adb_run(
    args: argparse.Namespace,
    *command: str,
    check: bool = True,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "adb", "-P", str(args.adb_port), "-s", args.phone_serial,
            *command,
        ],
        capture_output=True,
        check=check,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=timeout,
    )


def adb_su(
    args: argparse.Namespace,
    command: str,
    *,
    check: bool = True,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    return adb_run(
        args,
        "shell",
        f"su -c {shlex.quote(command)}",
        check=check,
        timeout=timeout,
    )


def phone_task_server_command(args: argparse.Namespace) -> list[str]:
    values = [
        f"{args.phone_task_bin_dir}/llama-server",
        "--model", args.phone_task_model,
        "--alias", LLAMA1,
        "--ctx-size", "4096",
        "--parallel", "1",
        "--batch-size", "1024",
        "--ubatch-size", "256",
        "--cont-batching",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--host", "127.0.0.1",
        "--port", str(args.phone_task_port),
        "--n-gpu-layers", "99",
        "--device", "GPUOpenCL",
        "--flash-attn", "off",
        "--no-webui",
        "--log-colors", "off",
    ]
    body = " ".join(shlex.quote(value) for value in values)
    shell = (
        f"cd {shlex.quote(args.phone_task_bin_dir)} && "
        f"export LD_LIBRARY_PATH=. && exec {body}"
    )
    return [
        "adb", "-P", str(args.adb_port), "-s", args.phone_serial,
        "shell", f"su -c {shlex.quote(shell)}",
    ]


def start_phone_task_server(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    output: Path,
) -> tuple[run_trace.CapturedProcess, float, dict[str, Any], int]:
    run_trace.require(
        adb_run(args, "get-state").stdout.strip() == "device",
        "whole-task phone unavailable",
    )
    check = adb_su(
        args,
        f"test -x {shlex.quote(args.phone_task_bin_dir + '/llama-server')} "
        f"-a -f {shlex.quote(args.phone_task_model)}",
        check=False,
    )
    run_trace.require(check.returncode == 0, "whole-task phone runtime")
    remote_hash = adb_su(
        args, f"sha256sum {shlex.quote(args.phone_task_model)}"
    ).stdout.split()[0]
    run_trace.require(
        remote_hash == manifest["model_inventory"][LLAMA1]["artifact_sha256"],
        "whole-task phone model identity",
    )
    run_trace.require(
        adb_su(args, "pidof llama-server", check=False).returncode != 0,
        "foreign phone llama-server",
    )
    adb_run(
        args,
        "forward",
        f"tcp:{args.phone_task_forward_port}",
        f"tcp:{args.phone_task_port}",
    )
    process = run_trace.CapturedProcess(
        phone_task_server_command(args),
        os.environ.copy(),
        output,
        "phone-task-server",
    )
    started_ns = time.monotonic_ns()
    process.start()
    succeeded = False
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            run_trace.require(
                process.process is not None
                and process.process.poll() is None,
                "whole-task phone server exited during load",
            )
            try:
                health = run_trace.http_json(
                    f"http://127.0.0.1:{args.phone_task_forward_port}/health",
                    1,
                )
                if health.get("status") == "ok":
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise run_trace.RunError("whole-task phone readiness timeout")
        props = run_trace.http_json(
            f"http://127.0.0.1:{args.phone_task_forward_port}/props", 10
        )
        run_trace.require(
            props.get("model_alias") == LLAMA1,
            "whole-task phone model alias",
        )
        pids = adb_su(args, "pidof llama-server").stdout.split()
        run_trace.require(
            len(pids) == 1 and pids[0].isdigit(),
            "whole-task phone server pid",
        )
        succeeded = True
        return (
            process,
            (time.monotonic_ns() - started_ns) / 1e6,
            props,
            int(pids[0]),
        )
    finally:
        if not succeeded:
            process.terminate()
            adb_run(
                args,
                "forward", "--remove",
                f"tcp:{args.phone_task_forward_port}",
                check=False,
            )


def stop_phone_task_server(
    args: argparse.Namespace, remote_pid: int | None
) -> None:
    if remote_pid is not None:
        adb_su(args, f"kill -INT {remote_pid}", check=False)
        time.sleep(0.2)
        adb_su(args, f"kill -TERM {remote_pid}", check=False)
    adb_run(
        args,
        "forward", "--remove", f"tcp:{args.phone_task_forward_port}",
        check=False,
    )


def validate_model_files(
    specs: dict[str, ModelSpec], manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    inventory = manifest["model_inventory"]
    result = {}
    for model_id, spec in sorted(specs.items()):
        expected = inventory[model_id]
        run_trace.require(spec.model_path.is_file(), f"missing model: {model_id}")
        size = spec.model_path.stat().st_size
        digest = digest_file(spec.model_path)
        run_trace.require(
            size == expected["artifact_bytes"]
            and digest == expected["artifact_sha256"],
            f"model identity: {model_id}",
        )
        row = {
            "artifact_bytes": size,
            "artifact_sha256": digest,
            "model_path": str(spec.model_path),
        }
        if spec.mmproj_path is not None:
            run_trace.require(spec.mmproj_path.is_file(), "missing projector")
            projector_size = spec.mmproj_path.stat().st_size
            projector_digest = digest_file(spec.mmproj_path)
            run_trace.require(
                projector_size == expected["projector_bytes"]
                and projector_digest == expected["projector_sha256"],
                "projector identity",
            )
            row.update({
                "projector_bytes": projector_size,
                "projector_path": str(spec.mmproj_path),
                "projector_sha256": projector_digest,
            })
        result[model_id] = row
    return result


def server_command(
    spec: ModelSpec,
    server_cuda: Path,
    server_cpu: Path,
) -> tuple[list[str], dict[str, str]]:
    server = server_cuda if spec.backend == "cuda" else server_cpu
    command = [
        str(server),
        "--model", str(spec.model_path),
        "--alias", spec.model_id,
        "--fit", "off",
        "--ctx-size", str(spec.ctx_size),
        "--parallel", str(spec.parallel),
        "--batch-size", str(spec.batch_size),
        "--ubatch-size", str(spec.ubatch_size),
        "--cont-batching",
        "--kv-unified",
        "--no-cache-idle-slots",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--host", "127.0.0.1",
        "--port", str(spec.port),
        "--metrics",
        "--slots",
        "--no-webui",
        "--log-colors", "off",
        "--log-verbosity", "4",
        "--log-timestamps",
    ]
    if spec.backend == "cuda":
        command.extend([
            "--flash-attn", "on",
            "--split-mode", "none",
            "--n-gpu-layers", "all",
            "--main-gpu", "0",
            "--device", "CUDA0",
        ])
    else:
        command.extend([
            "--n-gpu-layers", "0",
            "--threads", str(spec.threads),
            "--threads-batch", str(spec.threads),
        ])
    if spec.mmproj_path is not None:
        command.extend(["--mmproj", str(spec.mmproj_path), "--jinja"])
        if spec.backend == "cpu":
            command.append("--no-mmproj-offload")
    if spec.cpus:
        command = ["taskset", "--cpu-list", spec.cpus, *command]
    environment = os.environ.copy()
    return command, environment


def start_server(
    spec: ModelSpec,
    server_cuda: Path,
    server_cpu: Path,
    cuda_lib_dir: Path,
    cpu_lib_dir: Path,
    output: Path,
    label: str,
) -> tuple[run_trace.CapturedProcess, float, dict[str, Any]]:
    command, environment = server_command(spec, server_cuda, server_cpu)
    lib_dir = cuda_lib_dir if spec.backend == "cuda" else cpu_lib_dir
    server = server_cuda if spec.backend == "cuda" else server_cpu
    environment["LD_LIBRARY_PATH"] = (
        f"{lib_dir}:{server.parent}:" + environment.get("LD_LIBRARY_PATH", "")
    )
    if spec.backend == "cuda":
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    process = run_trace.CapturedProcess(command, environment, output, label)
    started_ns = time.monotonic_ns()
    process.start()
    succeeded = False
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            run_trace.require(
                process.process is not None and process.process.poll() is None,
                f"{label}: exited during load",
            )
            try:
                health = run_trace.http_json(
                    f"http://127.0.0.1:{spec.port}/health", 1
                )
                if health.get("status") == "ok":
                    break
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.1)
        else:
            raise run_trace.RunError(f"{label}: readiness timeout")
        load_ms = (time.monotonic_ns() - started_ns) / 1e6
        props = run_trace.http_json(
            f"http://127.0.0.1:{spec.port}/props", 10
        )
        run_trace.require(
            props.get("model_alias") == spec.model_id,
            f"{label}: model alias",
        )
        if spec.backend == "cuda":
            matches = re.findall(
                r"offloaded ([0-9]+)/([0-9]+) layers to GPU",
                "\n".join(process.stderr_lines),
            )
            run_trace.require(
                any(
                    int(loaded) == int(total) and int(total) > 0
                    for loaded, total in matches
                ),
                f"{label}: full CUDA placement",
            )
        if spec.mmproj_path is not None:
            modalities = props.get("modalities")
            run_trace.require(
                type(modalities) is dict and modalities.get("vision") is True,
                f"{label}: vision capability",
            )
        succeeded = True
        return process, load_ms, props
    finally:
        if not succeeded:
            process.terminate()


def text_completion(
    spec: ModelSpec,
    row: dict[str, Any],
    stream_path: Path,
    on_first: Callable[[int], None],
) -> dict[str, Any]:
    body = {
        "cache_prompt": False,
        "ignore_eos": True,
        "n_predict": row["output_tokens"],
        "prompt": row["prompt_tokens"],
        "return_tokens": True,
        "seed": row["mixed_request_index"],
        "stream": True,
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{spec.port}/completion",
        data=run_trace.canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final = None
    tokens: list[int] = []
    first_reported = False
    with stream_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                payload_line = raw_line.decode("utf-8").strip()
                if not payload_line.startswith("data:"):
                    continue
                payload = payload_line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                value = json.loads(payload)
                run_trace.require(
                    type(value) is dict and "error" not in value,
                    "text completion chunk",
                )
                chunk = value.get("tokens", [])
                run_trace.require(
                    type(chunk) is list
                    and all(type(token) is int for token in chunk),
                    "text completion tokens",
                )
                predicted = value.get("tokens_predicted")
                if (
                    not first_reported
                    and type(predicted) is int
                    and predicted > 0
                    and not value.get("stop", False)
                ):
                    on_first(time.monotonic_ns())
                    first_reported = True
                tokens.extend(chunk)
                if value.get("stop", False):
                    final = value
    run_trace.require(final is not None, "text completion final")
    timings = final.get("timings")
    run_trace.require(
        type(timings) is dict
        and timings.get("prompt_n") == row["input_tokens"]
        and timings.get("predicted_n") == row["output_tokens"]
        and len(tokens) <= row["output_tokens"],
        "text completion accounting",
    )
    return {
        "output_text": None,
        "predicted_ms": timings["predicted_ms"],
        "prompt_ms": timings["prompt_ms"],
        "runtime_prompt_tokens": timings["prompt_n"],
        "stream_token_ids_complete": len(tokens) == row["output_tokens"],
        "tokens": tokens,
    }


def build_image_transports(
    rows: list[dict[str, Any]],
) -> dict[str, ImageTransport]:
    result = {}
    for row in rows:
        if row["prompt_transport"] != "multimodal_message":
            continue
        image = row["image"]
        source_sha256 = image["sha256"]
        if source_sha256 in result:
            continue
        payload = (REPO_ROOT / image["path"]).read_bytes()
        run_trace.require(
            len(payload) == image["bytes"]
            and hashlib.sha256(payload).hexdigest() == source_sha256,
            "VLM image identity",
        )
        media_type = image["media_type"]
        transcoded = media_type == "image/webp"
        if transcoded:
            try:
                from PIL import Image
            except ImportError as error:
                raise run_trace.RunError(
                    "Pillow is required for WebP transport"
                ) from error
            target = io.BytesIO()
            with Image.open(io.BytesIO(payload)) as bitmap:
                run_trace.require(
                    bitmap.size == (image["width"], image["height"]),
                    "VLM decoded image geometry",
                )
                bitmap.convert("RGB").save(target, format="PNG")
            transport_payload = target.getvalue()
            transport_media_type = "image/png"
        else:
            transport_payload = payload
            transport_media_type = media_type
        result[source_sha256] = ImageTransport(
            source_bytes=len(payload),
            source_media_type=media_type,
            source_sha256=source_sha256,
            transport_media_type=transport_media_type,
            transport_payload=transport_payload,
            transport_sha256=hashlib.sha256(transport_payload).hexdigest(),
            transcoded=transcoded,
        )
    return result


def image_data_url(
    row: dict[str, Any], transports: dict[str, ImageTransport]
) -> str:
    transport = transports[row["image"]["sha256"]]
    encoded = base64.b64encode(transport.transport_payload).decode("ascii")
    return f"data:{transport.transport_media_type};base64,{encoded}"


def vlm_completion(
    spec: ModelSpec,
    row: dict[str, Any],
    image_transports: dict[str, ImageTransport],
    stream_path: Path,
    on_first: Callable[[int], None],
) -> dict[str, Any]:
    body = {
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
        "max_tokens": row["output_tokens"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": row["prompt_text"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url(row, image_transports),
                    },
                },
            ],
        }],
        "model": spec.model_id,
        "seed": row["mixed_request_index"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{spec.port}/v1/chat/completions",
        data=run_trace.canonical(body),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    final: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    content_pieces: list[str] = []
    emitted_pieces: list[str] = []
    reasoning_pieces: list[str] = []
    first_seen = False
    with stream_path.open("xb") as raw_stream:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                raw_stream.write(raw_line)
                payload_line = raw_line.decode("utf-8").strip()
                if not payload_line.startswith("data:"):
                    continue
                payload = payload_line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                value = json.loads(payload)
                run_trace.require(
                    type(value) is dict and "error" not in value,
                    "VLM completion chunk",
                )
                if type(value.get("usage")) is dict:
                    usage = value["usage"]
                choices = value.get("choices", [])
                run_trace.require(type(choices) is list, "VLM choices")
                for choice in choices:
                    delta = choice.get("delta", {})
                    if type(delta) is dict:
                        for key, destination in (
                            ("reasoning_content", reasoning_pieces),
                            ("content", content_pieces),
                        ):
                            piece = delta.get(key)
                            if type(piece) is str and piece:
                                if not first_seen:
                                    on_first(time.monotonic_ns())
                                    first_seen = True
                                destination.append(piece)
                                emitted_pieces.append(piece)
                    if choice.get("finish_reason") is not None:
                        final = value
                if type(value.get("timings")) is dict:
                    final = value
    run_trace.require(final is not None, "VLM completion final")
    timings = final.get("timings")
    if type(timings) is not dict and usage is not None:
        timings = usage.get("timings")
    run_trace.require(type(timings) is dict, "VLM completion timings")
    run_trace.require(
        timings.get("predicted_n") == row["output_tokens"],
        "VLM output token accounting",
    )
    prompt_n = timings.get("prompt_n")
    run_trace.require(
        type(prompt_n) is int and prompt_n > row["input_tokens"],
        "VLM runtime image tokens",
    )
    return {
        "content_text": "".join(content_pieces),
        "output_text": "".join(emitted_pieces),
        "predicted_ms": timings["predicted_ms"],
        "prompt_ms": timings["prompt_ms"],
        "reasoning_text": "".join(reasoning_pieces),
        "runtime_prompt_tokens": prompt_n,
        "stream_token_ids_complete": None,
        "tokens": None,
    }


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def score_vlm(row: dict[str, Any], output_text: str) -> bool:
    normalized = normalize_text(output_text)
    return all(
        any(normalize_text(answer) in normalized for answer in group)
        for group in row["quality_case"]["answer_groups"]
    )


def metrics(
    rows: list[dict[str, Any]], paid_start_ns: int
) -> dict[str, Any]:
    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "completed": len(group),
            "completion_s": run_trace.stats([
                (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in group
            ]),
            "end_s": max(
                (row["completion_ns"] - paid_start_ns) / 1e9 for row in group
            ),
            "output_tokens": sum(row["output_tokens"] for row in group),
            "queue_s": run_trace.stats([
                (row["dispatch_ns"] - row["scheduled_arrival_ns"]) / 1e9
                for row in group
            ]),
            "service_s": run_trace.stats([
                (row["completion_ns"] - row["dispatch_ns"]) / 1e9
                for row in group
            ]),
            "slo_met": sum(row["slo_met"] for row in group),
            "ttft_s": run_trace.stats([
                (row["first_token_ns"] - row["dispatch_ns"]) / 1e9
                for row in group
            ]),
        }

    by_model = {
        model_id: one([row for row in rows if row["execution_model_id"] == model_id])
        for model_id in sorted({row["execution_model_id"] for row in rows})
    }
    by_route = {
        route: one([row for row in rows if row["route"] == route])
        for route in sorted({row["route"] for row in rows})
    }
    duration_s = (max(row["completion_ns"] for row in rows) - paid_start_ns) / 1e9
    return {
        "by_model": by_model,
        "by_route": by_route,
        "completed": len(rows),
        "duration_s": duration_s,
        "output_throughput_tokens_s": (
            sum(row["output_tokens"] for row in rows) / duration_s
        ),
        "output_tokens": sum(row["output_tokens"] for row in rows),
        "slo_met": sum(row["slo_met"] for row in rows),
    }


def build_specs(args: argparse.Namespace) -> dict[str, ModelSpec]:
    return {
        QWEN14: ModelSpec(
            QWEN14, args.qwen14_model, "cuda", 18480, 4, 24576, 2048, 512
        ),
        QWEN06: ModelSpec(
            QWEN06, args.qwen06_model, "cpu", 18483, 4, 8192, 2048, 512,
            threads=4, cpus=args.qwen06_cpus,
        ),
        LLAMA1: ModelSpec(
            LLAMA1, args.llama1_model, "cpu", 18484, 4, 8192, 2048, 512,
            threads=4, cpus=args.llama1_cpus,
        ),
        GEMMA_E2B: ModelSpec(
            GEMMA_E2B, args.gemma_e2b_model, "cpu", 18485, 2, 8192, 2048,
            512, threads=8, cpus=args.vlm_cpus, mmproj_path=args.mmproj,
        ),
        QWEN8: ModelSpec(
            QWEN8, args.qwen8_model, "cuda", 18486, 4, 8192, 2048, 512
        ),
        GEMMA12: ModelSpec(
            GEMMA12, args.gemma12_model, "cuda", 18487, 8, 32768, 4096, 512
        ),
    }


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
    parser.add_argument("--qwen06-model", type=Path, required=True)
    parser.add_argument("--llama1-model", type=Path, required=True)
    parser.add_argument("--qwen8-model", type=Path, required=True)
    parser.add_argument("--gemma-e2b-model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path, required=True)
    parser.add_argument("--scheduler-plan", type=Path, required=True)
    parser.add_argument("--gemma-phone-server", type=Path)
    parser.add_argument("--gemma-phone-lib-dir", type=Path)
    parser.add_argument("--gemma-phone-port", type=int, default=18488)
    parser.add_argument("--gemma-phone-cpus")
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--bridge-cpus")
    parser.add_argument(
        "--bridge-allocator", choices=("devmem", "heap"), default="devmem"
    )
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen06-cpus", default="16-19")
    parser.add_argument("--llama1-cpus", default="20-23")
    parser.add_argument("--vlm-cpus", default="0,2,4,6,8,10,12,14")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    validate_trace(args.requests, args.manifest)
    rows = run_trace.read_jsonl(args.requests)
    run_trace.require(
        len(rows) == 114
        and all(row.get("schema") == TRACE_SCHEMA for row in rows),
        "trace geometry",
    )
    manifest = read_manifest(args.manifest)
    scheduler_plan = read_scheduler_plan(
        args.scheduler_plan, rows, digest_file(args.requests)
    )
    phone_kind = phone_route_kind(scheduler_plan)
    if phone_kind == "cpu-op15-ffn":
        for path in (
            args.gemma_phone_server,
            args.gemma_phone_lib_dir,
            args.bridge,
        ):
            run_trace.require(
                path is not None and path.exists(),
                f"missing phone route path: {path}",
            )
    specs = build_specs(args)
    model_identities = validate_model_files(specs, manifest)
    image_transports = build_image_transports(rows)

    args.output.mkdir(parents=True)
    events = run_trace.EventWriter(args.output / "events.jsonl")
    processes: dict[str, run_trace.CapturedProcess] = {}
    bridge: run_trace.CapturedProcess | None = None
    phone_process: run_trace.CapturedProcess | None = None
    phone_remote_pid: int | None = None
    phone_spec: ModelSpec | None = None
    if phone_kind == "cpu-op15-ffn":
        phone_spec = ModelSpec(
            GEMMA12,
            args.gemma12_model,
            "cpu-op15",
            args.gemma_phone_port,
            8,
            32768,
            4096,
            512,
        )
    elif phone_kind == "whole-task-adreno":
        phone_spec = ModelSpec(
            LLAMA1,
            args.llama1_model,
            "phone-adreno",
            args.phone_task_forward_port,
            1,
            4096,
            1024,
            256,
        )
    sampler: DynamicSampler | None = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    server_records: list[dict[str, Any]] = []
    phone_retired_before_cuda = False
    phone_retirement: dict[str, Any] | None = None
    live_scheduler: UnifiedScheduler | None = None
    live_scheduler_lock = threading.Lock()
    live_decisions: dict[int, Decision] = {}
    live_release_receipts: dict[int, dict[str, Any]] = {}
    try:
        phone_indices = set(
            scheduler_plan["phone"].get(
                "predicted_selected_mixed_request_indices",
                scheduler_plan["phone"].get(
                    "selected_mixed_request_indices", []
                ),
            )
        )
        if phone_kind == "cpu-op15-ffn":
            bridge = run_trace.start_bridge(args, args.output)
            phone_started_ns = time.monotonic_ns()
            phone_process = run_server_trace.start_cold_server(
                phone_server_args(args, scheduler_plan), args.output
            )
            phone_load_ms = (time.monotonic_ns() - phone_started_ns) / 1e6
            phone_props = run_trace.http_json(
                f"http://127.0.0.1:{args.gemma_phone_port}/props", 10
            )
            phone_backend = "cpu-op15"
            phone_model_id = GEMMA12
        elif phone_kind == "whole-task-adreno":
            (
                phone_process,
                phone_load_ms,
                phone_props,
                phone_remote_pid,
            ) = start_phone_task_server(args, manifest, args.output)
            phone_backend = "phone-adreno"
            phone_model_id = LLAMA1
        else:
            run_trace.require(phone_kind == "none", "phone route kind")

        if phone_spec is not None:
            run_trace.require(
                phone_props.get("model_alias") == phone_model_id,
                "phone route model alias",
            )
            phone_record = {
                "backend": phone_backend,
                "load_ms": phone_load_ms,
                "model_id": phone_model_id,
                "props": phone_props,
                "stage": "preloaded_parallel_route",
            }
            server_records.append(phone_record)
            phone_warm = dict(min(
                (
                    row for row in rows
                    if (
                        row["mixed_request_index"] in phone_indices
                        or (
                            not phone_indices
                            and row["execution_model_id"] == phone_model_id
                        )
                    )
                ),
                key=lambda row: row["input_tokens"] + row["output_tokens"],
            ))
            phone_warm["output_tokens"] = 2
            phone_warm_started_ns = time.monotonic_ns()
            text_completion(
                phone_spec,
                phone_warm,
                args.output / f"warm-{phone_model_id}-phone.raw",
                lambda _: None,
            )
            phone_record["warm_ms"] = (
                time.monotonic_ns() - phone_warm_started_ns
            ) / 1e6

        if scheduler_plan.get("scheduler", {}).get("enabled") is True:
            live_scheduler = UnifiedScheduler(
                (ProfileBundle.from_json(
                    scheduler_plan["scheduler"]["profile"]
                ),),
                scheduler_plan["scheduler"]["policy_mode"],
            )

        gpu_sequence = scheduler_plan["gpu"]["sequence"]
        initial_gpu_model = gpu_sequence[0]
        initial_ids = (QWEN06, LLAMA1, GEMMA_E2B, initial_gpu_model)
        for model_id in initial_ids:
            spec = specs[model_id]
            process, load_ms, props = start_server(
                spec,
                args.server_cuda,
                args.server_cpu,
                args.cuda_lib_dir,
                args.cpu_lib_dir,
                args.output,
                f"server-{model_id}",
            )
            processes[model_id] = process
            server_records.append({
                "backend": spec.backend,
                "load_ms": load_ms,
                "model_id": model_id,
                "props": props,
                "stage": "initial",
            })

        for model_id in (QWEN06, LLAMA1, initial_gpu_model):
            candidates = [row for row in rows if row["execution_model_id"] == model_id]
            warm = dict(min(candidates, key=lambda row: row["input_tokens"]))
            warm["output_tokens"] = 2
            text_completion(
                specs[model_id],
                warm,
                args.output / f"warm-{model_id}.raw",
                lambda _: None,
            )

        sampler = DynamicSampler(args.output)
        for model_id, process in processes.items():
            sampler.set_pid(model_id, process.pid)
        if phone_process is not None:
            sampler.set_pid(scheduler_plan["phone"]["route_id"], phone_process.pid)
        if bridge is not None:
            sampler.set_pid("op15-dmabuf-bridge", bridge.pid)
        sampler.start()
        time.sleep(0.5)
        paid_start_ns = time.monotonic_ns()
        events.write({
            "kind": "trace_start",
            "plan_sha256": scheduler_plan["plan_sha256"],
            "policy": f"live-vq-{scheduler_plan['mode']}-v3",
            "schema": "s42-six-model-event-v1",
            "t_ns": paid_start_ns,
        })

        def execute_request(
            row: dict[str, Any],
            spec: ModelSpec,
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
                            (target_dispatch_ns - time.monotonic_ns()) / 1e9,
                            0.01,
                        ))
                dispatch_ns = time.monotonic_ns()
                stream_path = args.output / f"stream-{route}-{index:03d}.raw"
                if row["prompt_transport"] == "tokens":
                    value = text_completion(spec, row, stream_path, first.append)
                    quality_pass = None
                else:
                    run_trace.require(
                        row["prompt_transport"] == "multimodal_message"
                        and spec.mmproj_path is not None,
                        "multimodal route",
                    )
                    value = vlm_completion(
                        spec,
                        row,
                        image_transports,
                        stream_path,
                        first.append,
                    )
                    quality_pass = score_vlm(row, value["output_text"])
                run_trace.require(len(first) == 1, "first-token accounting")
                completion_ns = time.monotonic_ns()
                scheduled_ns = paid_start_ns + row["arrival_us"] * 1000
                release_receipt = None
                if scheduler_decision is not None:
                    actual_end_us = (completion_ns - paid_start_ns) // 1000
                    released: list[str] = []
                    expired_phases: list[str] = []
                    late: list[str] = []
                    with live_scheduler_lock:
                        run_trace.require(
                            live_scheduler is not None,
                            "live scheduler disappeared",
                        )
                        for lease in scheduler_decision.leases:
                            if actual_end_us <= lease.reserved_until_us:
                                live_scheduler.release(
                                    lease.token, actual_end_us
                                )
                                released.append(lease.token)
                            elif (
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
                        live_release_receipts[index] = release_receipt
                record = {
                    "completion_ns": completion_ns,
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "execution_model_id": row["execution_model_id"],
                    "first_token_ns": first[0],
                    "input_tokens": row["input_tokens"],
                    "mixed_request_index": index,
                    "content_text": value.get("content_text"),
                    "output_text": value["output_text"],
                    "output_tokens": row["output_tokens"],
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "prompt_transport": row["prompt_transport"],
                    "quality_pass": quality_pass,
                    "reasoning_text": value.get("reasoning_text"),
                    "route": route,
                    "scheduler_decision": (
                        None
                        if scheduler_decision is None
                        else decision_to_json(scheduler_decision)
                    ),
                    "scheduler_release": release_receipt,
                    "runtime_prompt_tokens": value["runtime_prompt_tokens"],
                    "scheduled_arrival_ns": scheduled_ns,
                    "slo_met": completion_ns <= scheduled_ns + row["slo_us"] * 1000,
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
                    with live_scheduler_lock:
                        live_scheduler.cancel(
                            scheduler_decision.request_id,
                            max(
                                row["arrival_us"],
                                (time.monotonic_ns() - paid_start_ns) // 1000,
                            ),
                        )
                with result_lock:
                    errors.append(
                        f"{index}: {type(error).__name__}: {error}"
                    )

        held: dict[str, list[dict[str, Any]]] = {
            model_id: [] for model_id in gpu_sequence[1:]
        }
        futures_by_model: dict[str, list[concurrent.futures.Future[None]]] = {
            model_id: [] for model_id in specs
        }
        phone_futures: list[concurrent.futures.Future[None]] = []
        phone_cache_future: concurrent.futures.Future[
            dict[str, Any]
        ] | None = None
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=114, thread_name_prefix="six-model"
        ) as pool:
            def submit_scheduled(
                row: dict[str, Any], runtime_now_us: int
            ) -> concurrent.futures.Future[None]:
                run_trace.require(
                    live_scheduler is not None,
                    "scheduled request without live scheduler",
                )
                index = row["mixed_request_index"]
                with live_scheduler_lock:
                    decision = live_scheduler.schedule(
                        plan_live_vq_trace.request_from_row(row),
                        runtime_now_us=runtime_now_us,
                    )
                    live_decisions[index] = decision
                route = scheduler_plan["scheduler"]["route_executors"][
                    decision.route_id
                ]
                events.write({
                    "decision": decision_to_json(decision),
                    "executor_route": route,
                    "kind": "scheduler_decision",
                    "mixed_request_index": index,
                    "schema": "s42-six-model-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
                if route == TASK_PHONE_ROUTE:
                    run_trace.require(
                        phone_spec is not None,
                        "task phone executor unavailable",
                    )
                    future = pool.submit(
                        execute_request,
                        row,
                        phone_spec,
                        TASK_PHONE_ROUTE,
                        decision,
                    )
                    phone_futures.append(future)
                    return future
                model_id = row["execution_model_id"]
                spec = specs[model_id]
                run_trace.require(
                    route == f"{model_id}-{spec.backend}",
                    "scheduler executor binding",
                )
                future = pool.submit(
                    execute_request, row, spec, route, decision
                )
                futures_by_model[model_id].append(future)
                return future

            for row in rows:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while time.monotonic_ns() < target_ns:
                    time.sleep(min((target_ns - time.monotonic_ns()) / 1e9, 0.01))
                model_id = row["execution_model_id"]
                events.write({
                    "execution_model_id": model_id,
                    "kind": "request_arrival",
                    "mixed_request_index": row["mixed_request_index"],
                    "prompt_transport": row["prompt_transport"],
                    "schema": "s42-six-model-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
                route = scheduler_plan["request_routes"][
                    str(row["mixed_request_index"])
                ]
                if route == UNIFIED_ROUTE:
                    if model_id in held:
                        held[model_id].append(row)
                        events.write({
                            "execution_model_id": model_id,
                            "kind": "request_held_for_residency",
                            "mixed_request_index": row["mixed_request_index"],
                            "schema": "s42-six-model-event-v1",
                            "t_ns": time.monotonic_ns(),
                        })
                    else:
                        submit_scheduled(
                            row,
                            max(
                                row["arrival_us"],
                                (time.monotonic_ns() - paid_start_ns) // 1000,
                            ),
                        )
                    continue
                if route == PHONE_ROUTE:
                    run_trace.require(
                        phone_spec is not None,
                        "HTP phone executor unavailable",
                    )
                    future = pool.submit(
                        execute_request, row, phone_spec, PHONE_ROUTE
                    )
                    phone_futures.append(future)
                    continue
                if model_id in held:
                    held[model_id].append(row)
                    events.write({
                        "execution_model_id": model_id,
                        "kind": "request_held",
                        "mixed_request_index": row["mixed_request_index"],
                        "schema": "s42-six-model-event-v1",
                        "t_ns": time.monotonic_ns(),
                    })
                    continue
                spec = specs[model_id]
                futures_by_model[model_id].append(
                    pool.submit(execute_request, row, spec, route)
                )

            def retire_phone_route() -> dict[str, Any]:
                for future in phone_futures:
                    future.result(timeout=3600)
                retire_started_ns = time.monotonic_ns()
                events.write({
                    "kind": "phone_route_retire_started",
                    "model_id": GEMMA12,
                    "reason": "queue_idle",
                    "schema": "s42-six-model-event-v1",
                    "t_ns": retire_started_ns,
                })
                phone_process.terminate()
                bridge.terminate()
                sampler.set_pid(PHONE_ROUTE, 0)
                sampler.set_pid("op15-dmabuf-bridge", 0)
                retired_ns = time.monotonic_ns()
                events.write({
                    "kind": "phone_route_retired",
                    "model_id": GEMMA12,
                    "reason": "queue_idle",
                    "retire_ms": (retired_ns - retire_started_ns) / 1e6,
                    "schema": "s42-six-model-event-v1",
                    "t_ns": retired_ns,
                })
                return {
                    "retire_started_ns": retire_started_ns,
                    "retired_ns": retired_ns,
                }

            def prepare_gemma_cache() -> dict[str, Any]:
                retirement = (
                    phone_retirement_future.result(timeout=3600)
                    if phone_retirement_future is not None
                    else {}
                )
                cache_started_ns = time.monotonic_ns()
                cache = cache_control.warm_file(args.gemma12_model)
                run_trace.require(
                    cache["resident_ppm"] >= 950_000,
                    "Gemma cache prewarm residency",
                )
                cache_ready_ns = time.monotonic_ns()
                events.write({
                    **cache,
                    "kind": "model_cache_prepared",
                    "model_id": GEMMA12,
                    "schema": "s42-six-model-event-v1",
                    "started_ns": cache_started_ns,
                    "t_ns": cache_ready_ns,
                })
                return {
                    **retirement,
                    "cache": cache,
                    "cache_ready_ns": cache_ready_ns,
                }

            phone_retirement_future = (
                pool.submit(retire_phone_route)
                if phone_kind == "cpu-op15-ffn"
                else None
            )

            for future in futures_by_model[initial_gpu_model]:
                future.result(timeout=3600)
            initial_gpu_end_ns = max(
                row["completion_ns"]
                for row in results
                if row["execution_model_id"] == initial_gpu_model
            )
            events.write({
                "kind": "protected_cuda_release",
                "model_id": initial_gpu_model,
                "schema": "s42-six-model-event-v1",
                "t_ns": initial_gpu_end_ns,
            })
            processes.pop(initial_gpu_model).terminate()
            sampler.set_pid(initial_gpu_model, 0)

            for model_id in gpu_sequence[1:]:
                if model_id == GEMMA12 and phone_kind == "cpu-op15-ffn":
                    run_trace.require(
                        phone_cache_future is not None,
                        "Gemma cache prewarm launch",
                    )
                    phone_retirement = phone_cache_future.result(timeout=3600)
                    phone_retired_before_cuda = (
                        phone_kind == "cpu-op15-ffn"
                    )
                    cache_probe = cache_control.resident_pages(
                        args.gemma12_model
                    )
                    phone_retirement["cache_at_switch"] = cache_probe
                    events.write({
                        **cache_probe,
                        "kind": "model_cache_verified",
                        "model_id": GEMMA12,
                        "schema": "s42-six-model-event-v1",
                        "t_ns": time.monotonic_ns(),
                    })
                spec = specs[model_id]
                process, load_ms, props = start_server(
                    spec,
                    args.server_cuda,
                    args.server_cpu,
                    args.cuda_lib_dir,
                    args.cpu_lib_dir,
                    args.output,
                    f"server-{model_id}",
                )
                processes[model_id] = process
                sampler.set_pid(model_id, process.pid)
                warm = dict(min(held[model_id], key=lambda row: row["input_tokens"]))
                warm["output_tokens"] = 2
                warm_started_ns = time.monotonic_ns()
                text_completion(
                    spec,
                    warm,
                    args.output / f"warm-{model_id}.raw",
                    lambda _: None,
                )
                warm_ms = (time.monotonic_ns() - warm_started_ns) / 1e6
                ready_ns = time.monotonic_ns()
                server_records.append({
                    "backend": spec.backend,
                    "load_ms": load_ms,
                    "model_id": model_id,
                    "props": props,
                    "ready_s": (ready_ns - paid_start_ns) / 1e9,
                    "stage": "promotion",
                    "warm_ms": warm_ms,
                })
                events.write({
                    "kind": "cuda_model_ready",
                    "load_ms": load_ms,
                    "model_id": model_id,
                    "schema": "s42-six-model-event-v1",
                    "t_ns": ready_ns,
                    "warm_ms": warm_ms,
                })
                if model_id == QWEN8 and phone_kind == "cpu-op15-ffn":
                    phone_cache_future = pool.submit(prepare_gemma_cache)
                if live_scheduler is None:
                    route = f"{model_id}-cuda"
                    futures = [
                        pool.submit(execute_request, row, spec, route)
                        for row in held[model_id]
                    ]
                    futures_by_model[model_id].extend(futures)
                else:
                    runtime_now_us = max(
                        0, (ready_ns - paid_start_ns) // 1000
                    )
                    futures = [
                        submit_scheduled(row, runtime_now_us)
                        for row in held[model_id]
                    ]
                for future in futures:
                    future.result(timeout=3600)
                events.write({
                    "kind": "cuda_model_drained",
                    "model_id": model_id,
                    "schema": "s42-six-model-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
                processes.pop(model_id).terminate()
                sampler.set_pid(model_id, 0)

            for model_id in (QWEN06, LLAMA1, GEMMA_E2B):
                for future in futures_by_model[model_id]:
                    future.result(timeout=3600)
            for future in phone_futures:
                future.result(timeout=3600)
            if phone_cache_future is not None:
                phone_retirement = phone_cache_future.result(timeout=3600)

        run_trace.require(not errors, "request errors: " + "; ".join(errors))
        run_trace.require(len(results) == 114, "request conservation")
        run_trace.require(
            sorted(row["mixed_request_index"] for row in results) == list(range(114)),
            "mixed request identity conservation",
        )
        if live_scheduler is not None:
            run_trace.require(
                set(live_decisions) == set(range(114))
                and set(live_release_receipts) == set(range(114)),
                "live scheduler decision and release conservation",
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
        if bridge is not None:
            bridge.terminate()
        if phone_kind == "whole-task-adreno" and phone_remote_pid is not None:
            stop_phone_task_server(args, phone_remote_pid)
        if phone_kind == "whole-task-adreno":
            stop_phone_task_server(args, phone_remote_pid)
            phone_remote_pid = None

        if phone_kind == "cpu-op15-ffn":
            phone_log_lines = (
                args.output / "cold-server.stderr"
            ).read_text(
                encoding="utf-8", errors="backslashreplace"
            ).splitlines()
            bridge_log_lines = (
                args.output / "dmabuf-bridge.stderr"
            ).read_text(
                encoding="utf-8", errors="backslashreplace"
            ).splitlines()
            ffn_summary = completed_prefixed_json(
                phone_log_lines, "S41SERVERFFN "
            )
            shape_summaries = [
                json.loads(line[len("S41SERVERFFNSHAPE "):])
                for line in phone_log_lines
                if line.startswith("S41SERVERFFNSHAPE ")
            ]
            bridge_summary = run_server_trace.prefixed_json(
                bridge_log_lines, "FFNDMABUF ", True
            )
            expected_columns = {
                columns for _, columns in run_trace.parse_prefill_policy(
                    scheduler_plan["phone"]["split_policy"]["table"]
                )
                if columns > 0
            }
            run_trace.require(
                ffn_summary is not None
                and bridge_summary is not None
                and ffn_summary.get("status") == "ok"
                and bridge_summary.get("status") == "ok"
                and ffn_summary.get("calls") == bridge_summary.get("calls")
                and ffn_summary.get("calls", 0) > 0
                and bridge_summary.get("reset_recoveries") == 0
                and shape_summaries
                and {
                    shape["columns"] for shape in shape_summaries
                } <= expected_columns,
                "phone split summary",
            )
            phone_summary = {
                "bridge": bridge_summary,
                "ffn": ffn_summary,
                "residency": scheduler_plan["phone"],
                "shapes": shape_summaries,
            }
        else:
            phone_summary = {
                "residency": scheduler_plan["phone"],
                "task_request_indices": sorted(
                    row["mixed_request_index"]
                    for row in results
                    if row["route"] == TASK_PHONE_ROUTE
                ),
            }

        result = {
            "image_transports": {
                source_sha256: {
                    "source_bytes": transport.source_bytes,
                    "source_media_type": transport.source_media_type,
                    "source_sha256": transport.source_sha256,
                    "transport_bytes": len(transport.transport_payload),
                    "transport_media_type": transport.transport_media_type,
                    "transport_sha256": transport.transport_sha256,
                    "transcoded": transport.transcoded,
                }
                for source_sha256, transport in sorted(image_transports.items())
            },
            "manifest_sha256": digest_file(args.manifest),
            "metrics": metrics(results, paid_start_ns),
            "model_identities": model_identities,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "policy": {
                "cpu_resident_models": [QWEN06, LLAMA1, GEMMA_E2B],
                "gpu_sequence": gpu_sequence,
                "phone_preloaded_model": scheduler_plan["phone"]["model_id"],
                "phone_request_indices": sorted(
                    row["mixed_request_index"]
                    for row in results
                    if row["route"] in {PHONE_ROUTE, TASK_PHONE_ROUTE}
                ),
                "phone_retirement": phone_retirement,
                "phone_retired_before_cuda": phone_retired_before_cuda,
                "policy_id": f"live-vq-{scheduler_plan['mode']}-v3",
                "scheduler_alternative_model_ids": scheduler_plan.get(
                    "scheduler", {}
                ).get("alternative_model_ids", []),
                "scheduler_scheduled_model_ids": scheduler_plan.get(
                    "scheduler", {}
                ).get("scheduled_model_ids", []),
                "scheduler_single_route_model_ids": scheduler_plan.get(
                    "scheduler", {}
                ).get("single_route_model_ids", []),
                "scheduler_plan_sha256": scheduler_plan["plan_sha256"],
            },
            "phone": phone_summary,
            "scheduler_runtime": {
                "coverage": {
                    "alternative_model_ids": scheduler_plan.get(
                        "scheduler", {}
                    ).get("alternative_model_ids", []),
                    "scheduled_model_ids": scheduler_plan.get(
                        "scheduler", {}
                    ).get("scheduled_model_ids", []),
                    "single_route_model_ids": scheduler_plan.get(
                        "scheduler", {}
                    ).get("single_route_model_ids", []),
                },
                "decisions": [
                    {
                        **decision_to_json(live_decisions[index]),
                        "mixed_request_index": index,
                        "executor_route": next(
                            row["route"] for row in results
                            if row["mixed_request_index"] == index
                        ),
                    }
                    for index in sorted(live_decisions)
                ],
                "enabled": live_scheduler is not None,
                "release_receipts": {
                    str(index): live_release_receipts[index]
                    for index in sorted(live_release_receipts)
                },
                "resource_snapshot_at_end": (
                    None
                    if live_scheduler is None
                    else live_scheduler.resource_snapshot(
                        (paid_end_ns - paid_start_ns) // 1000
                    )
                ),
            },
            "request_results": sorted(
                results, key=lambda row: row["mixed_request_index"]
            ),
            "resources": {
                "gpu_memory_used_max_bytes": max(
                    row["gpu"]["memory_used_bytes"] for row in samples
                ),
                "gpu_power_w": run_trace.stats([
                    row["gpu"]["power_mw"] / 1000 for row in samples
                ]),
                "gpu_utilization_pct": run_trace.stats([
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
            "scheduler_plan_file_sha256": digest_file(args.scheduler_plan),
            "server_records": server_records,
            "server_energy": run_trace.server_energy_summary(
                samples, paid_start_ns, paid_end_ns
            ),
            "status": "PASS",
            "trace_sha256": digest_file(args.requests),
            "vlm": {
                "quality_pass": sum(
                    row["quality_pass"] is True
                    for row in results
                    if row["execution_model_id"] == GEMMA_E2B
                ),
                "request_count": 10,
                "runtime_prompt_tokens": [
                    row["runtime_prompt_tokens"]
                    for row in results
                    if row["execution_model_id"] == GEMMA_E2B
                ],
            },
        }
        run_trace.write_json(args.output / "RESULT.json", result)
        print(json.dumps({
            "duration_s": result["metrics"]["duration_s"],
            "server_energy_j": result["server_energy"]
                ["server_compute_device_energy_j"],
            "slo_met": result["metrics"]["slo_met"],
            "status": "PASS",
            "vlm_quality_pass": result["vlm"]["quality_pass"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        run_trace.write_json(args.output / "FAILURE.json", {
            "error": f"{type(error).__name__}: {error}",
            "schema": "s42-six-model-physical-failure-v1",
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
        if bridge is not None:
            bridge.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
