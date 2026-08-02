#!/usr/bin/env python3
"""Run the two-resident stock-default control on one GPU."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "s39_desktop_swap_baseline"
sys.path.insert(0, str(BASE))

import run_desktop_baseline as baseline  # noqa: E402
import validate_contract  # noqa: E402


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
    validate_contract.validate()
    runner_args = argparse.Namespace(
        cuda_lib_dir=args.cuda_lib_dir,
        gpu_index=args.gpu_index,
        output=args.output,
        port=args.port,
        request_timeout_s=args.request_timeout_s,
        server=args.server,
        server_timeout_s=args.server_timeout_s,
        serving_profile="stock_default",
    )
    runner = baseline.Runner(runner_args)
    servers: dict[str, baseline.ServerProcess] = {}
    sampler: baseline.PowerSampler | None = None
    report: dict[str, Any] = {}
    try:
        preflight = runner.preflight()
        warm_records = runner.warm_both()
        ready_records: dict[str, dict[str, Any]] = {}
        load_error: str | None = None
        for offset, model_id in enumerate((
                "qwen3-8b-q8_0", "qwen3-14b-q4_k_m")):
            server = baseline.ServerProcess(
                runner, model_id, args.port + offset, f"dual-{model_id}"
            )
            servers[model_id] = server
            try:
                ready_records[model_id] = server.start()
            except Exception as exc:
                load_error = f"{model_id}: {exc}"
                break
        if load_error is not None:
            report = {
                "load_error": load_error,
                "preflight": preflight,
                "ready": ready_records,
                "repeat_index": args.repeat_index,
                "schema": "s39-stock-default-dual-v1",
                "status": "STOCK_DEFAULT_DUAL_LOAD_FAILED",
                "warm_records": warm_records,
            }
            (runner.output / "dual.json").write_bytes(
                baseline.canonical(report)
            )
            return report
        if not all(server.healthy() for server in servers.values()):
            raise baseline.RunError("a dual server is not healthy before replay")
        capacities = {
            model_id: ready["props"].get("total_slots")
            for model_id, ready in ready_records.items()
        }
        if any(type(value) is not int or value < 1
               for value in capacities.values()):
            raise baseline.RunError(f"invalid default slot counts: {capacities}")
        semaphores = {
            model_id: threading.Semaphore(capacity)
            for model_id, capacity in capacities.items()
        }
        results: list[dict[str, Any]] = []
        errors: list[str] = []
        lock = threading.Lock()
        sampler = baseline.PowerSampler(
            runner.output / "resource_samples.jsonl",
            runner.gpu_index,
            lambda: None,
        )
        sampler.start()
        time.sleep(0.3)
        paid_start_ns = time.monotonic_ns()
        runner.events.write({
            "control_id": "STOCK_DEFAULT_DUAL_WARM",
            "kind": "replay_start",
            "repeat_index": args.repeat_index,
            "schema": "s39-stock-default-dual-event-v1",
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
                "schema": "s39-stock-default-dual-event-v1",
            })
            semaphore = semaphores[row["model_id"]]
            semaphore.acquire()
            dispatch_ns = time.monotonic_ns()
            runner.events.write({
                "event_id": row["event_id"],
                "kind": "request_dispatched",
                "model_id": row["model_id"],
                "request_index": row["request_index"],
                "schema": "s39-stock-default-dual-event-v1",
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
                    "schema": "s39-stock-default-dual-result-v1",
                    "slo_us": row["slo_us"],
                    **result,
                }
                with lock:
                    results.append(record)
                runner.events.write({
                    **record,
                    "kind": "request_complete",
                    "schema": "s39-stock-default-dual-event-v1",
                })
            except Exception as exc:
                with lock:
                    errors.append(f"{row['event_id']}: {exc}")
                runner.events.write({
                    "error": str(exc),
                    "event_id": row["event_id"],
                    "kind": "request_error",
                    "request_index": row["request_index"],
                    "schema": "s39-stock-default-dual-event-v1",
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
                f"dual request failure: completed={len(results)} errors={errors}"
            )
        paid_end_ns = max(row["completion_ns"] for row in results)
        runner.events.write({
            "kind": "replay_end",
            "schema": "s39-stock-default-dual-event-v1",
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
        if system_after["system_swap_free_bytes"] < swap_before \
                or any(row["process_swap_bytes"] != 0
                       for row in process_after.values()):
            raise baseline.RunError("dual replay grew swap")
        counts = Counter(row["model_id"] for row in results)
        report = {
            "capacities": capacities,
            "energy": energy,
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "preflight": preflight,
            "process_after": process_after,
            "ready": ready_records,
            "repeat_index": args.repeat_index,
            "request_counts": dict(counts),
            "request_results": sorted(
                results, key=lambda row: row["request_index"]
            ),
            "schema": "s39-stock-default-dual-v1",
            "status": "STOCK_DEFAULT_DUAL_REPLAY_PASS",
            "system_memory_after": system_after,
            "warm_records": warm_records,
        }
        (runner.output / "dual.json").write_bytes(baseline.canonical(report))
        return report
    except Exception as exc:
        if hasattr(runner, "output") and runner.output.exists():
            failure = {
                "error": str(exc),
                "repeat_index": args.repeat_index,
                "schema": "s39-stock-default-dual-failure-v1",
                "status": "STOCK_DEFAULT_DUAL_REPLAY_FAILED",
            }
            (runner.output / "failure.json").write_bytes(
                baseline.canonical(failure)
            )
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
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--port", type=int, default=18250)
    parser.add_argument("--server-timeout-s", type=float, default=180.0)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--replay-timeout-s", type=float, default=900.0)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
        print(json.dumps({
            "status": result["status"],
        }, sort_keys=True))
    except (baseline.RunError, OSError, ValueError) as exc:
        print(f"STOCK_DEFAULT_DUAL_ERROR: {exc}")
        raise SystemExit(2)
