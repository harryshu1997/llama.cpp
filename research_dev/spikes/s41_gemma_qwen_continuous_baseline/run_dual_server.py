#!/usr/bin/env python3
"""Run the S41 dual-ready GPU plus CPU/RAM server control."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import threading
import time
from typing import Any

import run_gpu_switch


baseline = run_gpu_switch.legacy
GEMMA = "gemma-4-12b-it-q8_0"
QWEN = "qwen3-14b-q4_k_m"


class DualServerProcess(baseline.ServerProcess):
    def __init__(
        self,
        runner: run_gpu_switch.Runner,
        model_id: str,
        port: int,
        label: str,
        placement: str,
        cpu_threads: int,
    ) -> None:
        super().__init__(runner, model_id, port, label)
        if placement not in {"CPU_RAM", "FULL_CUDA"}:
            raise baseline.RunError("invalid dual placement")
        self.placement = placement
        self.cpu_threads = cpu_threads

    def build_command(self) -> list[str]:
        model = self.runner.models[self.model_id]
        command = [
            str(self.runner.server),
            "--model", str(model["path"]),
            "--alias", self.model_id,
            "--fit", "off",
            "--ctx-size", "4096",
            "--parallel", "8",
            "--batch-size", "2048",
            "--ubatch-size", "512",
            "--flash-attn", "on",
            "--cont-batching",
            "--kv-unified",
            "--no-cache-idle-slots",
            "--cache-type-k", "f16",
            "--cache-type-v", "f16",
            "--split-mode", "none",
        ]
        if self.placement == "FULL_CUDA":
            command.extend([
                "--n-gpu-layers", "all",
                "--main-gpu", "0",
                "--device", "CUDA0",
            ])
        else:
            command.extend([
                "--n-gpu-layers", "0",
                "--device", "none",
                "--no-kv-offload",
                "--threads", str(self.cpu_threads),
                "--threads-batch", str(self.cpu_threads),
            ])
        command.extend([
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--metrics",
            "--slots",
            "--no-webui",
            "--log-colors", "off",
            "--log-timestamps",
            "--verbose",
        ])
        return command

    def start(self, require_headroom: bool = True) -> dict[str, Any]:
        record = super().start(
            require_headroom=self.placement == "FULL_CUDA")
        matches = record["full_cuda_offload_matches"]
        if self.placement == "FULL_CUDA":
            valid = any(
                int(loaded) == int(total) and int(total) > 0
                for loaded, total in matches
            )
        else:
            valid = any(
                int(loaded) == 0 and int(total) > 0
                for loaded, total in matches
            )
        if not valid:
            raise baseline.RunError(
                f"{self.label}: placement does not match {self.placement}")
        record["declared_placement"] = self.placement
        return record


def write_manifest(path: Path) -> None:
    files = sorted(
        item for item in path.iterdir()
        if item.is_file() and item.name != "SHA256SUMS.txt"
    )
    (path / "SHA256SUMS.txt").write_text(
        "".join(
            f"{baseline.digest_file(item)}  {item.name}\n"
            for item in files
        ),
        encoding="ascii",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    runner_args = argparse.Namespace(
        cuda_lib_dir=args.cuda_lib_dir,
        gpu_index=args.gpu_index,
        output=args.output,
        port=args.port,
        request_timeout_s=args.request_timeout_s,
        server=args.server,
        server_timeout_s=args.server_timeout_s,
        serving_profile="fixed_full_cuda",
    )
    runner = run_gpu_switch.Runner(runner_args)
    runner.serving_profile = "dual_explicit_gpu_cpu"
    placements = {
        args.gpu_model_id: "FULL_CUDA",
        args.cpu_model_id: "CPU_RAM",
    }
    servers: dict[str, DualServerProcess] = {}
    sampler: baseline.PowerSampler | None = None
    try:
        preflight = runner.preflight()
        warm_records = runner.warm_both()
        ready: dict[str, dict[str, Any]] = {}
        for offset, model_id in enumerate(
                (args.gpu_model_id, args.cpu_model_id)):
            server = DualServerProcess(
                runner,
                model_id,
                args.port + offset,
                f"dual-{placements[model_id].lower()}-{model_id}",
                placements[model_id],
                args.cpu_threads,
            )
            servers[model_id] = server
            ready[model_id] = server.start()
        if not all(server.healthy() for server in servers.values()):
            raise baseline.RunError("a dual server is not healthy")
        capacities = {
            model_id: record["props"].get("total_slots")
            for model_id, record in ready.items()
        }
        if capacities != {
                args.gpu_model_id: 8, args.cpu_model_id: 8}:
            raise baseline.RunError(f"dual slot mismatch: {capacities}")
        compute_pids = {
            row["pid"] for row in baseline.compute_processes()
            if row["used_memory_bytes"] > 128 * 1024 * 1024
        }
        gpu_pid = servers[args.gpu_model_id].pid
        cpu_pid = servers[args.cpu_model_id].pid
        if gpu_pid not in compute_pids or cpu_pid in compute_pids:
            raise baseline.RunError(
                "dual process placement does not match CUDA process list")

        semaphores = {
            model_id: threading.Semaphore(8)
            for model_id in servers
        }
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        lock = threading.Lock()
        sampler = baseline.PowerSampler(
            runner.output / "resource_samples.jsonl",
            runner.gpu_index,
            lambda: gpu_pid,
        )
        sampler.start()
        time.sleep(0.3)
        paid_start_ns = time.monotonic_ns()
        runner.events.write({
            "control_id": "S41_GPU_CPU_DUAL_READY",
            "kind": "replay_start",
            "repeat_index": args.repeat_index,
            "schema": "s41-dual-server-event-v1",
            "t_ns": paid_start_ns,
        })

        def request_worker(row: dict[str, Any]) -> None:
            target_ns = paid_start_ns + row["arrival_us"] * 1000
            while True:
                remaining = target_ns - time.monotonic_ns()
                if remaining <= 0:
                    break
                time.sleep(min(remaining / 1e9, 0.01))
            arrival_ns = time.monotonic_ns()
            runner.events.write({
                "actual_t_ns": arrival_ns,
                "event_id": row["event_id"],
                "kind": "request_arrival",
                "model_id": row["model_id"],
                "request_index": row["request_index"],
                "scheduled_t_ns": target_ns,
                "schema": "s41-dual-server-event-v1",
            })
            semaphore = semaphores[row["model_id"]]
            semaphore.acquire()
            dispatch_ns = time.monotonic_ns()
            runner.events.write({
                "event_id": row["event_id"],
                "kind": "request_dispatched",
                "model_id": row["model_id"],
                "request_index": row["request_index"],
                "schema": "s41-dual-server-event-v1",
                "t_ns": dispatch_ns,
            })
            first_ns: list[int] = []
            try:
                result = baseline.stream_completion(
                    servers[row["model_id"]],
                    row,
                    runner.output / f"stream-{row['request_index']:03d}.raw",
                    args.request_timeout_s,
                    first_ns.append,
                )
                if len(first_ns) != 1:
                    raise baseline.RunError("first-token count mismatch")
                record = {
                    "completion_ns": time.monotonic_ns(),
                    "dispatch_model_id": row["model_id"],
                    "dispatch_ns": dispatch_ns,
                    "event_id": row["event_id"],
                    "first_token_ns": first_ns[0],
                    "model_id": row["model_id"],
                    "request_index": row["request_index"],
                    "scheduled_arrival_ns": target_ns,
                    "schema": "s41-dual-server-result-v1",
                    "slo_us": row["slo_us"],
                    **result,
                }
                with lock:
                    results.append(record)
                runner.events.write({
                    **record,
                    "kind": "request_complete",
                    "schema": "s41-dual-server-event-v1",
                })
            except Exception as exc:
                with lock:
                    errors.append(f"{row['event_id']}: {exc}")
                runner.events.write({
                    "error": str(exc),
                    "event_id": row["event_id"],
                    "kind": "request_error",
                    "request_index": row["request_index"],
                    "schema": "s41-dual-server-event-v1",
                    "t_ns": time.monotonic_ns(),
                })
            finally:
                semaphore.release()

        threads = [
            threading.Thread(
                target=request_worker,
                args=(row,),
                name=f"dual-request-{row['request_index']:03d}",
            )
            for row in runner.requests
        ]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + args.replay_timeout_s
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            errors.append("dual replay timeout")
        if errors or len(results) != len(runner.requests):
            raise baseline.RunError(
                f"dual request failure: completed={len(results)} "
                f"errors={errors}")
        paid_end_ns = max(row["completion_ns"] for row in results)
        runner.events.write({
            "kind": "replay_end",
            "schema": "s41-dual-server-event-v1",
            "t_ns": paid_end_ns,
        })
        time.sleep(0.3)
        sampler.stop()
        energy = sampler.integrate(paid_start_ns, paid_end_ns)
        sampler = None
        process_after = {
            model_id: baseline.proc_status(server.pid or -1)
            for model_id, server in servers.items()
        }
        system_after = baseline.system_memory()
        swap_before = preflight["system_memory"]["system_swap_free_bytes"]
        system_swap_growth_bytes = max(
            0, swap_before - system_after["system_swap_free_bytes"])
        process_swap_bytes = sum(
            row["process_swap_bytes"] for row in process_after.values())
        resource_gate_pass = (
            system_swap_growth_bytes == 0 and process_swap_bytes == 0)
        counts = Counter(row["model_id"] for row in results)
        report = {
            "capacities": capacities,
            "cpu_threads": args.cpu_threads,
            "energy": energy,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "placements": placements,
            "preflight": preflight,
            "process_after": process_after,
            "ready": ready,
            "repeat_index": args.repeat_index,
            "request_counts": dict(counts),
            "request_results": sorted(
                results, key=lambda row: row["request_index"]),
            "resource_gate": {
                "maximum_process_swap_bytes": 0,
                "maximum_system_swap_growth_bytes": 0,
                "process_swap_bytes": process_swap_bytes,
                "status": (
                    "PASS" if resource_gate_pass
                    else "FAIL_SWAP_GROWTH"
                ),
                "system_swap_growth_bytes": system_swap_growth_bytes,
            },
            "schema": "s41-dual-server-v1",
            "status": (
                "S41_GPU_CPU_DUAL_READY_PASS"
                if resource_gate_pass
                else "S41_GPU_CPU_DUAL_READY_RESOURCE_FAIL"
            ),
            "system_memory_after": system_after,
            "warm_records": warm_records,
        }
        (runner.output / "dual.json").write_bytes(
            baseline.canonical(report))
        return report
    except Exception as exc:
        if hasattr(runner, "output") and runner.output.exists():
            failure = {
                "error": str(exc),
                "repeat_index": args.repeat_index,
                "schema": "s41-dual-server-failure-v1",
                "status": "S41_GPU_CPU_DUAL_READY_FAILED",
            }
            (runner.output / "failure.json").write_bytes(
                baseline.canonical(failure))
        raise
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except Exception:
                pass
        for server in reversed(list(servers.values())):
            try:
                server.stop()
            except Exception:
                pass
        if hasattr(runner, "events"):
            runner.close()
        if args.output.exists():
            write_manifest(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--gpu-model-id", choices=[GEMMA, QWEN], required=True)
    parser.add_argument("--cpu-model-id", choices=[GEMMA, QWEN], required=True)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18350)
    parser.add_argument("--server-timeout-s", type=float, default=300.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--replay-timeout-s", type=float, default=1200.0)
    args = parser.parse_args()
    if args.gpu_model_id == args.cpu_model_id:
        parser.error("GPU and CPU models must differ")
    if args.cpu_threads < 1 or args.cpu_threads > 24:
        parser.error("--cpu-threads must be in [1,24]")
    return args


if __name__ == "__main__":
    try:
        result = run(parse_args())
        print(json.dumps({"status": result["status"]}, sort_keys=True))
        if result["status"] != "S41_GPU_CPU_DUAL_READY_PASS":
            raise SystemExit(2)
    except (
        baseline.RunError,
        OSError,
        ValueError,
        run_gpu_switch.validate_inputs.ValidationError,
    ) as exc:
        print(f"S41_DUAL_SERVER_ERROR: {exc}")
        raise SystemExit(2)
