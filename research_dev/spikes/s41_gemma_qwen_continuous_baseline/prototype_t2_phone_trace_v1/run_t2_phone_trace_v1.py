#!/usr/bin/env python3
"""Run one non-qualification S41 T2 BurstGPT trace on CUDA and phones."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from typing import Any


GEMMA_ID = "gemma-4-12b-it-q8_0"
QWEN_ID = "qwen3-14b-q4_k_m"
QWEN_SHA256 = (
    "500a8806e85ee9c83f3ae084202955924"
    "51379b4f8cf2d0f41c15dffeb6b81f0"
)
PHONE_SERIALS = {
    "op12": "5ae7a43d",
    "op15": "3C15AU002CL00000",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True)):
        require(raw.endswith(b"\n"), f"requests[{index}]: framing")
        row = json.loads(raw)
        require(type(row) is dict, f"requests[{index}]: object")
        require(canonical_bytes(row) == raw, f"requests[{index}]: canonical")
        rows.append(row)
    return rows


def write_new(path: Path, value: Any) -> None:
    raw = canonical_bytes(value)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EventWriter:
    def __init__(self, path: Path):
        self.stream = path.open("xb", buffering=0)
        self.lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.stream.write(canonical_bytes(row))

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


class SerializedStageClient:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()

    def batch(self, rows):
        with self.lock:
            return self.client.batch(rows)

    def remove(self, seq_id, request_id, route_epoch):
        with self.lock:
            return self.client.remove(seq_id, request_id, route_epoch)

    def drain(self):
        with self.lock:
            return self.client.drain()

    def stop(self):
        with self.lock:
            return self.client.stop()

    def close(self):
        with self.lock:
            return self.client.close()


class PrototypeSupervisor:
    def __init__(
        self,
        phone_module,
        stage_module,
        gateway_module,
        plan: dict[str, Any],
        output: Path,
    ):
        self.phone = phone_module
        self.stage = stage_module
        self.gateway = gateway_module
        self.plan = plan
        self.output = output
        self.processes = {}

    def open(self, _spec):
        self.processes = self.phone.start_processes(self.plan, self.output)
        relay = self.processes["op15_direct_relay"].process
        require(relay is not None, "relay process")
        deadline = time.monotonic() + (
            self.plan["processes"]["op15_direct_relay"]["startup_timeout_ms"]
            / 1000
        )
        client = None
        last_error = None
        while time.monotonic() < deadline:
            require(relay.poll() is None, "relay exited")
            try:
                client = self.stage.StageV3Client.connect(
                    self.plan["relay_host"],
                    self.plan["relay_port"],
                    min(5.0, max(0.1, deadline - time.monotonic())),
                )
                hello = client.hello()
                return self.gateway.RouteConnection(
                    SerializedStageClient(client),
                    hello,
                    "prototype-qwen-route-v1",
                )
            except (OSError, self.stage.ProtocolError) as error:
                last_error = error
                if client is not None:
                    client.close()
                time.sleep(0.1)
        raise RuntimeError(f"phone route connect: {last_error}")

    def close(self, _spec, client) -> None:
        stop_error = None
        try:
            client.stop()
        except BaseException as error:
            stop_error = error
        finally:
            client.close()
        try:
            self.phone.stop_processes(self.processes)
        finally:
            self.processes = {}
        if stop_error is not None:
            raise stop_error

    def kill(self) -> None:
        for process in self.processes.values():
            process.kill()
        self.processes = {}


def phone_memory(prototype, adb_path: str) -> dict[str, str]:
    script = (
        "set -eu; "
        "for pid in $(pidof llama-layersplit 2>/dev/null); do "
        "printf 'PID=%s\\n' \"$pid\"; "
        "grep -E '^(VmRSS|VmSwap):' /proc/$pid/status; "
        "done; "
        "grep -E '^(MemAvailable|SwapTotal|SwapFree):' /proc/meminfo"
    )
    return {
        name: prototype.run_checked(
            prototype.adb(adb_path, serial, "shell", script)
        )
        for name, serial in PHONE_SERIALS.items()
    }


def phone_command(
    gateway,
    command_id: int,
    row: dict[str, Any],
    committed: list[int],
    kind: int,
) -> dict[str, Any]:
    request_id = f"s41-{row['request_index']:03d}"
    request = {
        "committed_output_tokens": list(committed),
        "model_id": QWEN_ID,
        "owner_id": "phone-qwen",
        "ownership_epoch": 1,
        "position": len(row["prompt_tokens"]) + len(committed),
        "prompt_tokens": list(row["prompt_tokens"]),
        "publication_index": len(committed),
        "request_id": request_id,
        "state": 1,
    }
    return {
        "command_id": command_id,
        "controller_epoch": 1,
        "executor_id": "phone-qwen",
        "executor_instance_id": "prototype-t2",
        "kind": kind,
        "max_output_tokens": 1 if kind == gateway.COMMAND_EXECUTE else 0,
        "model_id": QWEN_ID,
        "request": request,
        "request_id": request_id,
        "schema": "llama-server-warm-tier-command-v3",
        "total_output_tokens": row["output_tokens"] if kind == 0 else 0,
    }


def make_route_spec(gateway):
    return gateway.RouteSpec(
        model_id=QWEN_ID,
        model_sha256=QWEN_SHA256,
        artifact_certificate_sha256="0" * 64,
        readiness_lock_sha256="0" * 64,
        readiness_phase_id="prototype",
        relay_host="prototype",
        relay_port=1,
        file_type=15,
        layer_start=0,
        layer_end=40,
        n_layer=40,
        n_embd=5120,
        max_streams=8,
        n_batch=64,
        n_ubatch=64,
        batch_knee=8,
        gather_us=1000,
        queue_depth=128,
        prefill_chunk=64,
        phase="A_ONLY",
        phase_lock_sha256="0" * 64,
        qualification={},
        qualification_sha256="0" * 64,
        slot="A",
        load_argv=(),
        unload_argv=(),
        a6000_identity="prototype",
        op15_boot_id="prototype",
        op12_boot_id="prototype",
        op15_shard_sha256="0" * 64,
        op12_shard_sha256="0" * 64,
        worker_sha256="0" * 64,
    )


def result_metrics(
    results: list[dict[str, Any]],
    paid_start_ns: int,
) -> dict[str, Any]:
    by_model = {}
    for model_id in (GEMMA_ID, QWEN_ID):
        rows = [row for row in results if row["model_id"] == model_id]
        ttft = sorted(
            (row["first_token_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        )
        completion = sorted(
            (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
            for row in rows
        )
        p95_index = max(0, (95 * len(rows) + 99) // 100 - 1)
        by_model[model_id] = {
            "completed": len(rows),
            "completion_p95_s": completion[p95_index],
            "slo_met": sum(
                row["completion_ns"] - row["scheduled_arrival_ns"]
                <= row["slo_us"] * 1000
                for row in rows
            ),
            "ttft_p95_s": ttft[p95_index],
        }
    end_ns = max(row["completion_ns"] for row in results)
    return {
        "by_model": by_model,
        "completed": len(results),
        "duration_s": (end_ns - paid_start_ns) / 1e9,
        "slo_met": sum(
            row["completion_ns"] - row["scheduled_arrival_ns"]
            <= row["slo_us"] * 1000
            for row in results
        ),
        "throughput_tokens_s": (
            sum(len(row["tokens"]) for row in results)
            / ((end_ns - paid_start_ns) / 1e9)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--switches", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--phone-plan", type=Path, required=True)
    parser.add_argument("--phone-producer", type=Path, required=True)
    parser.add_argument("--usb-launcher", type=Path, required=True)
    parser.add_argument("--prototype-helpers", type=Path, required=True)
    parser.add_argument("--phone-gateway", type=Path, required=True)
    parser.add_argument("--stage-client", type=Path, required=True)
    parser.add_argument("--s41-runner", type=Path, required=True)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adb-path",
        default="/usr/lib/android-sdk/platform-tools/adb",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    require(
        args.execute and args.confirm == "RUN_S41_T2_PHONE_TRACE_PROTOTYPE",
        "confirmation",
    )
    require(args.output.is_absolute() and not args.output.exists(), "output")

    for directory in (
        args.phone_gateway.parent,
        args.stage_client.parent,
        args.phone_gateway.parent.parents[1] / "s39_phone_model_switch_trace",
    ):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    s41 = load_module("s41_t2_runner", args.s41_runner)
    prototype = load_module("s41_t2_helpers", args.prototype_helpers)
    phone = load_module("s41_t2_phone_producer", args.phone_producer)
    gateway = load_module("s41_t2_phone_gateway", args.phone_gateway)
    stage = load_module("s41_t2_stage_client", args.stage_client)

    runner_args = argparse.Namespace(
        server=args.server,
        cuda_lib_dir=args.cuda_lib_dir,
        output=args.output,
        gpu_index=0,
        server_timeout_s=240.0,
        request_timeout_s=600.0,
        port=18460,
    )
    runner = s41.Runner(runner_args)
    server = None
    executor = None
    supervisor = None
    event_writer = None
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    result_lock = threading.Lock()
    command_lock = threading.Lock()
    command_id = 0
    try:
        preflight = runner.preflight()
        requests = read_jsonl(args.requests)
        switches = read_jsonl(args.switches)
        require(
            len(requests) == 74
            and len(switches) == 9
            and sum(row["model_id"] == GEMMA_ID for row in requests) == 57
            and sum(row["model_id"] == QWEN_ID for row in requests) == 17,
            "trace geometry",
        )
        manifest = json.loads(args.input_manifest.read_bytes())
        require(
            manifest["files"]["REQUESTS.jsonl"]["sha256"]
            == digest_file(args.requests)
            and manifest["files"]["SWITCHES.jsonl"]["sha256"]
            == digest_file(args.switches),
            "trace manifest",
        )

        phone_source, _ = prototype.read_json(args.phone_plan)
        states = {
            name: prototype.phone_live_state(args.adb_path, serial)
            for name, serial in PHONE_SERIALS.items()
        }
        component_paths = {"op12": set(), "op15": set()}
        for process in phone_source["processes"].values():
            argv = process["argv"]
            inline = json.loads(argv[argv.index("--plan-json") + 1])
            component_paths[inline["endpoint"]].update(
                component["path"] for component in inline["components"]
            )
        component_stats = {
            endpoint: prototype.live_component_stats(
                args.adb_path,
                PHONE_SERIALS[endpoint],
                sorted(paths),
            )
            for endpoint, paths in component_paths.items()
        }
        phone_plan = prototype.patch_phone_plan(
            phone_source,
            states["op12"],
            states["op15"],
            component_stats,
        )
        prototype.direct_android_processes(
            phone_plan,
            args.usb_launcher,
            args.adb_path,
            args.output,
        )
        write_new(args.output / "phone-route-prototype.json", phone_plan)

        supervisor = PrototypeSupervisor(
            phone,
            stage,
            gateway,
            phone_plan,
            args.output,
        )
        spec = make_route_spec(gateway)
        executor = gateway.PhoneRouteExecutor(
            "phone-qwen",
            {QWEN_ID: spec},
            supervisor,
            QWEN_ID,
            timeout_s=600.0,
            wire_evidence_path=args.output / "phone-wire.jsonl",
        )
        memory_before = phone_memory(prototype, args.adb_path)

        server = s41.legacy.ServerProcess(
            runner,
            GEMMA_ID,
            runner_args.port,
            "t2-gemma-cuda",
        )
        runner.set_server(server)
        gemma_ready = server.start()
        event_writer = EventWriter(args.output / "trace-events.jsonl")
        paid_start_ns = time.monotonic_ns()
        event_writer.write({
            "kind": "trace_start",
            "mode": "T2_PHONE_NO_PROMOTION_PROTOTYPE",
            "requests_sha256": digest_file(args.requests),
            "schema": "s41-t2-phone-trace-event-v1",
            "switches_sha256": digest_file(args.switches),
            "t_ns": paid_start_ns,
        })

        def next_command_id() -> int:
            nonlocal command_id
            with command_lock:
                command_id += 1
                return command_id

        def run_phone(row: dict[str, Any]) -> None:
            dispatch_ns = time.monotonic_ns()
            committed: list[int] = []
            first_ns = 0
            try:
                for _ in range(row["output_tokens"]):
                    command = phone_command(
                        gateway,
                        next_command_id(),
                        row,
                        committed,
                        gateway.COMMAND_EXECUTE,
                    )
                    value = executor.handle(command)
                    publications = value["publications"]
                    require(
                        value["success"] is True
                        and len(publications) == 1,
                        "phone publication",
                    )
                    if not committed:
                        first_ns = time.monotonic_ns()
                    committed.append(publications[0]["token"])
                cleanup = phone_command(
                    gateway,
                    next_command_id(),
                    row,
                    committed,
                    gateway.COMMAND_CLEANUP,
                )
                require(executor.handle(cleanup)["success"] is True, "cleanup")
                completed_ns = time.monotonic_ns()
                record = {
                    "completion_ns": completed_ns,
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "first_token_ns": first_ns,
                    "model_id": row["model_id"],
                    "request_index": row["request_index"],
                    "route": "OP15_OP12_OPENCL",
                    "scheduled_arrival_ns":
                        paid_start_ns + row["arrival_us"] * 1000,
                    "schema": "s41-t2-phone-request-result-v1",
                    "slo_us": row["slo_us"],
                    "tokens": committed,
                }
                with result_lock:
                    results.append(record)
                    event_writer.write({"kind": "request_complete", **record})
            except BaseException as error:
                with result_lock:
                    errors.append(f"{row['event_id']}: {error}")

        def run_gemma(row: dict[str, Any]) -> None:
            dispatch_ns = time.monotonic_ns()
            first: list[int] = []
            try:
                value = s41.legacy.stream_completion(
                    server,
                    row,
                    args.output / f"stream-{row['request_index']:03d}.raw",
                    600.0,
                    first.append,
                )
                require(len(first) == 1, "Gemma first token")
                completed_ns = time.monotonic_ns()
                record = {
                    "completion_ns": completed_ns,
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "first_token_ns": first[0],
                    "model_id": row["model_id"],
                    "request_index": row["request_index"],
                    "route": "RTX4060TI_CUDA0",
                    "scheduled_arrival_ns":
                        paid_start_ns + row["arrival_us"] * 1000,
                    "schema": "s41-t2-phone-request-result-v1",
                    "slo_us": row["slo_us"],
                    "tokens": value["tokens"],
                }
                with result_lock:
                    results.append(record)
                    event_writer.write({"kind": "request_complete", **record})
            except BaseException as error:
                with result_lock:
                    errors.append(f"{row['event_id']}: {error}")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="gemma",
        ) as gemma_pool, concurrent.futures.ThreadPoolExecutor(
            max_workers=8,
            thread_name_prefix="qwen-phone",
        ) as phone_pool:
            futures = []
            for row in requests:
                target_ns = paid_start_ns + row["arrival_us"] * 1000
                while time.monotonic_ns() < target_ns:
                    time.sleep(min((target_ns - time.monotonic_ns()) / 1e9, 0.01))
                event_writer.write({
                    "event_id": row["event_id"],
                    "kind": "request_arrival",
                    "model_id": row["model_id"],
                    "request_index": row["request_index"],
                    "scheduled_t_ns": target_ns,
                    "schema": "s41-t2-phone-trace-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
                pool = phone_pool if row["model_id"] == QWEN_ID else gemma_pool
                function = run_phone if row["model_id"] == QWEN_ID else run_gemma
                futures.append(pool.submit(function, row))
            for future in futures:
                future.result(timeout=900)

        require(not errors, "request errors: " + "; ".join(errors))
        require(len(results) == 74, "request conservation")
        memory_after = phone_memory(prototype, args.adb_path)
        metrics = result_metrics(results, paid_start_ns)
        live_server_memory = s41.legacy.proc_status(server.pid)
        require(live_server_memory["process_swap_bytes"] == 0, "Gemma swap")
        event_writer.write({
            "kind": "trace_end",
            "schema": "s41-t2-phone-trace-event-v1",
            "t_ns": time.monotonic_ns(),
        })
        result = {
            "gemma_ready": gemma_ready,
            "input_manifest_sha256": digest_file(args.input_manifest),
            "metrics": metrics,
            "mode": "T2_PHONE_NO_PROMOTION_PROTOTYPE",
            "phone_memory_after": memory_after,
            "phone_memory_before": memory_before,
            "phone_plan_sha256": digest_file(
                args.output / "phone-route-prototype.json"
            ),
            "preflight": preflight,
            "prototype_only": True,
            "requests_sha256": digest_file(args.requests),
            "server_memory": live_server_memory,
            "status": "BURSTGPT_T2_PHONE_TRACE_PROTOTYPE_PASS",
            "switches": {
                "bound_count": len(switches),
                "executed_count": 0,
                "reason": "T2_PHONE_NO_PROMOTION",
                "sha256": digest_file(args.switches),
            },
        }
        write_new(args.output / "RESULT.json", result)
        return 0
    except BaseException as error:
        if args.output.exists():
            try:
                write_new(args.output / "FAILURE.json", {
                    "error": f"{type(error).__name__}: {error}",
                    "prototype_only": True,
                    "status": "BURSTGPT_T2_PHONE_TRACE_PROTOTYPE_FAILED",
                    "traceback": traceback.format_exc(),
                })
            except FileExistsError:
                pass
        return 2
    finally:
        if event_writer is not None:
            event_writer.close()
        if executor is not None:
            try:
                executor.close()
            except BaseException:
                if supervisor is not None:
                    supervisor.kill()
        elif supervisor is not None:
            supervisor.kill()
        if hasattr(runner, "events"):
            runner.close()
        prototype.cleanup_remote(args.adb_path)


if __name__ == "__main__":
    raise SystemExit(main())
