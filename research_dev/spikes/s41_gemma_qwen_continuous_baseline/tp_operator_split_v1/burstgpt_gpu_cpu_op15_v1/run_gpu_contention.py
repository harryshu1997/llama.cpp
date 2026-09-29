#!/usr/bin/env python3
"""Measure CPU plus OP15 cold inference with a resident or busy GPU model."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import run_trace


CONFIRMATION = "RUN_GPU_CONTENTION_CPU_OP15"


def parse_cpu_list(text: str) -> set[int]:
    result: set[int] = set()
    for item in text.split(","):
        fields = item.split("-", 1)
        try:
            first = int(fields[0])
            last = int(fields[-1])
        except ValueError as error:
            raise run_trace.RunError("CPU list integer") from error
        run_trace.require(
            first >= 0 and last >= first,
            "CPU list bounds",
        )
        result.update(range(first, last + 1))
    run_trace.require(result, "empty CPU list")
    return result


def format_cpu_list(cpus: set[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def process_affinity(pid: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    task_root = Path(f"/proc/{pid}/task")
    for task in sorted(task_root.iterdir(), key=lambda path: int(path.name)):
        try:
            cpus = set(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        key = format_cpu_list(cpus)
        counts[key] = counts.get(key, 0) + 1
    run_trace.require(counts, f"no threads for PID {pid}")
    return {
        "cpu_sets": counts,
        "thread_count": sum(counts.values()),
    }


def require_process_affinity(
    pid: int,
    expected: set[int],
    label: str,
) -> dict[str, Any]:
    record = process_affinity(pid)
    run_trace.require(
        all(
            parse_cpu_list(cpu_text) <= expected
            for cpu_text in record["cpu_sets"]
        ),
        f"{label} thread affinity",
    )
    return record


def require_process_nice(pid: int, expected: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    task_root = Path(f"/proc/{pid}/task")
    for task in sorted(task_root.iterdir(), key=lambda path: int(path.name)):
        try:
            value = os.getpriority(os.PRIO_PROCESS, int(task.name))
        except ProcessLookupError:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    run_trace.require(
        counts and set(counts) == {str(expected)},
        "hot thread nice value",
    )
    return {
        "nice_values": counts,
        "thread_count": sum(counts.values()),
    }


def frequency_policy() -> dict[str, Any]:
    root = Path("/sys/devices/system/cpu/intel_pstate")
    result: dict[str, Any] = {}
    for name in ("max_perf_pct", "min_perf_pct", "no_turbo"):
        path = root / name
        if path.exists():
            result[name] = int(path.read_text().strip())

    policies = []
    for path in sorted(Path("/sys/devices/system/cpu/cpufreq").glob("policy*")):
        record: dict[str, Any] = {"name": path.name}
        for name in (
            "scaling_governor",
            "scaling_max_freq",
            "scaling_min_freq",
        ):
            value_path = path / name
            if value_path.exists():
                value = value_path.read_text().strip()
                record[name] = int(value) if value.isdigit() else value
        policies.append(record)
    result["policies"] = policies
    return result


class HotSaturator:
    def __init__(
        self,
        port: int,
        rows: list[dict[str, Any]],
        output: Path,
        workers: int,
    ):
        self.port = port
        self.rows = rows
        self.output = output
        self.workers = workers
        self.stop_event = threading.Event()
        self.condition = threading.Condition()
        self.records: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.futures: list[concurrent.futures.Future[None]] = []

    def start(self) -> None:
        self.output.mkdir()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.workers,
            thread_name_prefix="gpu-load",
        )
        self.futures = [
            self.executor.submit(self._run, slot)
            for slot in range(self.workers)
        ]

    def _run(self, slot: int) -> None:
        sequence = 0
        while not self.stop_event.is_set():
            row = self.rows[(slot + sequence * self.workers) % len(self.rows)]
            start_ns = time.monotonic_ns()
            try:
                value = run_trace.hot_completion(
                    self.port,
                    row,
                    self.output / f"slot-{slot:02d}-{sequence:05d}.raw",
                    lambda _: None,
                )
                completion_ns = time.monotonic_ns()
                record = {
                    "completion_ns": completion_ns,
                    "input_tokens": row["input_tokens"],
                    "output_tokens": len(value["tokens"]),
                    "predicted_ms": value["predicted_ms"],
                    "prompt_ms": value["prompt_ms"],
                    "start_ns": start_ns,
                }
                with self.condition:
                    self.records.append(record)
                    self.condition.notify_all()
            except BaseException as error:
                if not self.stop_event.is_set():
                    with self.condition:
                        self.errors.append(
                            f"slot {slot}: {type(error).__name__}: {error}"
                        )
                        self.condition.notify_all()
                return
            sequence += 1

    def wait_until_loaded(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        with self.condition:
            while len(self.records) < self.workers and not self.errors:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise run_trace.RunError("GPU load readiness timeout")
                self.condition.wait(min(remaining, 0.2))
        run_trace.require(not self.errors, "; ".join(self.errors))

    def stop(self) -> None:
        self.stop_event.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        for future in self.futures:
            future.result()
        run_trace.require(not self.errors, "; ".join(self.errors))

    def metrics(self, paid_start_ns: int, paid_end_ns: int) -> dict[str, Any]:
        paid = [
            row for row in self.records
            if paid_start_ns <= row["completion_ns"] <= paid_end_ns
        ]
        duration_s = (paid_end_ns - paid_start_ns) / 1e9
        tokens = sum(row["output_tokens"] for row in paid)
        return {
            "completed_requests": len(paid),
            "completed_tokens": tokens,
            "output_throughput_tokens_s": tokens / duration_s,
            "worker_count": self.workers,
        }


def cold_metrics(
    rows: list[dict[str, Any]],
    paid_start_ns: int,
) -> dict[str, Any]:
    run_trace.require(len(rows) == 17, "cold request count")
    paid_end_ns = max(row["completion_ns"] for row in rows)
    duration_s = (paid_end_ns - paid_start_ns) / 1e9
    completion_s = [
        (row["completion_ns"] - row["scheduled_arrival_ns"]) / 1e9
        for row in rows
    ]
    queue_s = [
        (row["dispatch_ns"] - row["scheduled_arrival_ns"]) / 1e9
        for row in rows
    ]
    service_s = [
        (row["completion_ns"] - row["dispatch_ns"]) / 1e9
        for row in rows
    ]
    output_tokens = sum(len(row["tokens"]) for row in rows)
    return {
        "completed": len(rows),
        "completion_s": run_trace.stats(completion_s),
        "decode_s": run_trace.stats([row["decode_us"] / 1e6 for row in rows]),
        "duration_s": duration_s,
        "output_throughput_tokens_s": output_tokens / duration_s,
        "output_tokens": output_tokens,
        "prefill_s": run_trace.stats([
            row["prefill_us"] / 1e6 for row in rows
        ]),
        "queue_s": run_trace.stats(queue_s),
        "route_wall_s": run_trace.stats([
            row["route_wall_us"] / 1e6 for row in rows
        ]),
        "service_s": run_trace.stats(service_s),
        "slo_met": sum(
            row["completion_ns"] - row["scheduled_arrival_ns"]
            <= row["slo_us"] * 1000
            for row in rows
        ),
    }


def sample_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    run_trace.require(rows, "resource samples")
    utilization = [row["gpu"]["utilization_pct"] for row in rows]
    power_w = [row["gpu"]["power_mw"] / 1000 for row in rows]
    return {
        "cold_rss_max_bytes": max(row["cold"]["rss_bytes"] for row in rows),
        "cold_swap_max_bytes": max(row["cold"]["swap_bytes"] for row in rows),
        "gpu_power_w": run_trace.stats(power_w),
        "gpu_utilization_pct": run_trace.stats(utilization),
        "hot_rss_max_bytes": max(row["hot"]["rss_bytes"] for row in rows),
        "hot_swap_max_bytes": max(row["hot"]["swap_bytes"] for row in rows),
        "sample_count": len(rows),
        "system_available_min_bytes": min(
            row["system"]["available_bytes"] for row in rows
        ),
        "system_swap_free_min_bytes": min(
            row["system"]["swap_free_bytes"] for row in rows
        ),
    }


def validate(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]], list[tuple[int, int]], dict[str, set[int]] | None
]:
    run_trace.require(
        args.execute and args.confirm == CONFIRMATION,
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(),
        "output",
    )
    for path in (
        args.requests,
        args.hot_server,
        args.hot_model,
        args.cuda_lib_dir,
        args.cold_driver,
        args.cold_model,
        args.cold_lib_dir,
        args.bridge,
    ):
        run_trace.require(path.exists(), f"missing path: {path}")
    run_trace.require(
        run_trace.digest_file(args.requests) == run_trace.REQUESTS_SHA256,
        "trace identity",
    )
    run_trace.require(
        args.hot_model.stat().st_size == 9_001_752_960
        and run_trace.digest_file(args.hot_model) == run_trace.HOT_MODEL_SHA256,
        "hot model identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176
        and run_trace.digest_file(args.cold_model) == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )
    run_trace.require(
        args.max_columns in (9664, 10240, 11136)
        and args.decode_columns in (8192, 9664, args.max_columns)
        and args.decode_columns <= args.max_columns,
        "split width",
    )
    run_trace.require(1 <= args.hot_workers <= 8, "hot worker count")
    run_trace.require(0 <= args.hot_nice <= 19, "hot nice value")
    policy = run_trace.parse_prefill_policy(args.prefill_policy)
    run_trace.require(
        policy == [
            (64, 8192),
            (128, 8192),
            (320, args.max_columns),
            (512, args.max_columns),
        ],
        "prefill policy",
    )
    affinity_specs = (
        args.control_cpus,
        args.hot_cpus,
        args.bridge_cpus,
        args.cold_cpus,
    )
    run_trace.require(
        not any(affinity_specs) or all(affinity_specs),
        "affinity arguments must be supplied together",
    )
    affinity = None
    if all(affinity_specs):
        affinity = {
            "control": parse_cpu_list(args.control_cpus),
            "hot": parse_cpu_list(args.hot_cpus),
            "bridge": parse_cpu_list(args.bridge_cpus),
            "cold": parse_cpu_list(args.cold_cpus),
        }
        available = set(os.sched_getaffinity(0))
        run_trace.require(
            all(cpus <= available for cpus in affinity.values()),
            "requested CPU is unavailable",
        )
        run_trace.require(
            affinity["cold"].isdisjoint(affinity["control"])
            and affinity["cold"].isdisjoint(affinity["bridge"]),
            "cold CPUs must be isolated from control and bridge",
        )
        run_trace.require(
            args.allow_hot_cold_overlap
            or affinity["cold"].isdisjoint(affinity["hot"]),
            "cold CPUs overlap hot CPUs without opt-in",
        )
    requests = run_trace.read_jsonl(args.requests)
    run_trace.require(
        len(requests) == 74
        and sum(run_trace.role(row) == "hot" for row in requests) == 57
        and sum(run_trace.role(row) == "cold" for row in requests) == 17,
        "trace geometry",
    )
    return requests, policy, affinity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-load",
        choices=("resident-idle", "saturated"),
        required=True,
    )
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--hot-server", type=Path, required=True)
    parser.add_argument("--hot-model", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cold-driver", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--hot-port", type=int, default=18480)
    parser.add_argument("--hot-workers", type=int, default=8)
    parser.add_argument("--hot-nice", type=int, default=0)
    parser.add_argument("--control-cpus")
    parser.add_argument("--hot-cpus")
    parser.add_argument("--bridge-cpus")
    parser.add_argument("--cold-cpus")
    parser.add_argument("--allow-hot-cold-overlap", action="store_true")
    parser.add_argument("--max-columns", type=int, default=9664)
    parser.add_argument("--decode-columns", type=int, default=9664)
    parser.add_argument(
        "--prefill-policy",
        default="64:8192,128:8192,320:9664,512:9664",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    requests, policy, affinity = validate(args)
    if affinity is not None:
        os.sched_setaffinity(0, affinity["control"])
    cold_requests = [row for row in requests if run_trace.role(row) == "cold"]
    hot_requests = [row for row in requests if run_trace.role(row) == "hot"]
    args.mode = "op15"
    args.output.mkdir(parents=True)

    hot = None
    bridge = None
    cold = None
    sampler = None
    saturator = None
    try:
        preflight = {
            "cold_driver_sha256": run_trace.digest_file(args.cold_driver),
            "cold_model_sha256": run_trace.COLD_MODEL_SHA256,
            "cpu_frequency_policy": frequency_policy(),
            "gpu": run_trace.gpu_snapshot(),
            "gpu_load": args.gpu_load,
            "allow_hot_cold_overlap": args.allow_hot_cold_overlap,
            "hot_nice_requested": args.hot_nice,
            "affinity_requested": (
                {
                    label: sorted(cpus)
                    for label, cpus in affinity.items()
                }
                if affinity is not None else None
            ),
            "hot_model_sha256": run_trace.HOT_MODEL_SHA256,
            "hot_server_sha256": run_trace.digest_file(args.hot_server),
            "offload": {
                "decode_columns": args.decode_columns,
                "max_columns": args.max_columns,
                "prefill_policy": policy,
            },
            "repeat_index": args.repeat_index,
            "requests_sha256": run_trace.REQUESTS_SHA256,
            "schema": "s41-gpu-contention-preflight-v1",
            "system_memory": run_trace.system_memory(),
        }
        run_trace.write_json(args.output / "preflight.json", preflight)

        hot = run_trace.start_hot(args, args.output)
        bridge = run_trace.start_bridge(args, args.output)
        cold = run_trace.start_cold(args, args.output)
        affinity_at_ready = None
        if affinity is not None:
            affinity_at_ready = {
                "bridge": require_process_affinity(
                    bridge.pid, affinity["bridge"], "bridge"
                ),
                "cold": require_process_affinity(
                    cold.pid, affinity["cold"], "cold"
                ),
                "control": require_process_affinity(
                    os.getpid(), affinity["control"], "control"
                ),
                "hot": require_process_affinity(
                    hot.pid, affinity["hot"], "hot"
                ),
            }
        hot_nice_at_ready = require_process_nice(hot.pid, args.hot_nice)
        run_trace.require(
            run_trace.proc_status(hot.pid)["swap_bytes"] == 0
            and run_trace.proc_status(cold.pid)["swap_bytes"] == 0,
            "process swap before warmup",
        )

        hot_warm = hot_requests[0]
        cold_warm = cold_requests[0]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            hot_future = pool.submit(
                run_trace.hot_completion,
                args.hot_port,
                hot_warm,
                args.output / "warm-hot.raw",
                lambda _: None,
            )
            cold_future = pool.submit(
                cold.exchange,
                1,
                cold_warm,
                "DETACH",
                run_trace.policy_columns(policy, cold_warm["input_tokens"]),
                args.decode_columns,
            )
            hot_future.result(timeout=600)
            cold_future.result(timeout=600)

        if args.gpu_load == "saturated":
            saturator = HotSaturator(
                args.hot_port,
                hot_requests,
                args.output / "hot-load",
                args.hot_workers,
            )
            saturator.start()
            saturator.wait_until_loaded(120)
        else:
            time.sleep(1.0)

        affinity_at_paid_start = affinity_at_ready
        if affinity is not None:
            affinity_at_paid_start = {
                "bridge": require_process_affinity(
                    bridge.pid, affinity["bridge"], "bridge"
                ),
                "cold": require_process_affinity(
                    cold.pid, affinity["cold"], "cold"
                ),
                "control": require_process_affinity(
                    os.getpid(), affinity["control"], "control"
                ),
                "hot": require_process_affinity(
                    hot.pid, affinity["hot"], "hot"
                ),
            }
        hot_nice_at_paid_start = require_process_nice(
            hot.pid, args.hot_nice
        )

        sampler = run_trace.ResourceSampler(args.output, hot.pid, cold.pid)
        sampler.start()
        time.sleep(0.6)
        paid_start_ns = time.monotonic_ns()
        results = []
        launch_id = 1
        for index, row in enumerate(cold_requests):
            target_ns = paid_start_ns + row["arrival_us"] * 1000
            while True:
                remaining_ns = target_ns - time.monotonic_ns()
                if remaining_ns <= 0:
                    break
                time.sleep(min(remaining_ns / 1e9, 0.01))
            dispatch_ns = time.monotonic_ns()
            launch_id += 1
            value = cold.exchange(
                launch_id,
                row,
                "STOP" if index + 1 == len(cold_requests) else "DETACH",
                run_trace.policy_columns(policy, row["input_tokens"]),
                args.decode_columns,
            )
            completion_ns = time.monotonic_ns()
            results.append({
                "completion_ns": completion_ns,
                "decode_us": value["decode_us"],
                "dispatch_ns": dispatch_ns,
                "event_id": row["event_id"],
                "ffn_decode_columns": args.decode_columns,
                "ffn_prefill_columns": run_trace.policy_columns(
                    policy, row["input_tokens"]
                ),
                "prefill_us": value["prefill_us"],
                "request_index": row["request_index"],
                "route_wall_us": value["route_wall_us"],
                "scheduled_arrival_ns": target_ns,
                "slo_us": row["slo_us"],
                "tokens": value["token_ids"][0],
            })
        paid_end_ns = max(row["completion_ns"] for row in results)

        sampler.stop()
        resource_rows = list(sampler.rows)
        sampler = None
        if saturator is not None:
            saturator.stop()

        run_trace.require(cold.process is not None, "cold process")
        cold.process.wait(timeout=60)
        run_trace.require(cold.process.returncode == 0, "cold process status")
        run_trace.require(bridge.process is not None, "bridge process")
        bridge.process.wait(timeout=60)
        run_trace.require(bridge.process.returncode == 0, "bridge status")

        ffn_lines = [
            line for line in cold.stderr_lines if line.startswith("FFNSPLIT ")
        ]
        bridge_lines = [
            line for line in bridge.stderr_lines
            if line.startswith("FFNDMABUF ")
        ]
        run_trace.require(
            len(ffn_lines) == 1 and len(bridge_lines) == 1,
            "offload summaries",
        )
        result = {
            "affinity_at_paid_start": affinity_at_paid_start,
            "affinity_at_ready": affinity_at_ready,
            "cold": cold_metrics(results, paid_start_ns),
            "cpu_frequency_policy_after": frequency_policy(),
            "gpu_load": args.gpu_load,
            "hot_nice_at_paid_start": hot_nice_at_paid_start,
            "hot_nice_at_ready": hot_nice_at_ready,
            "hot_load": (
                saturator.metrics(paid_start_ns, paid_end_ns)
                if saturator is not None else {
                    "completed_requests": 0,
                    "completed_tokens": 0,
                    "output_throughput_tokens_s": 0.0,
                    "worker_count": 0,
                }
            ),
            "offload": {
                "dmabuf": json.loads(bridge_lines[0].split(" ", 1)[1]),
                "ffn": json.loads(ffn_lines[0].split(" ", 1)[1]),
            },
            "paid_end_ns": paid_end_ns,
            "paid_start_ns": paid_start_ns,
            "preflight": preflight,
            "repeat_index": args.repeat_index,
            "request_results": results,
            "resources": sample_metrics(resource_rows),
            "schema": "s41-gpu-contention-result-v1",
            "status": "PASS",
        }
        run_trace.require(
            result["cpu_frequency_policy_after"]
            == preflight["cpu_frequency_policy"],
            "CPU frequency policy changed",
        )
        run_trace.require(
            result["cold"]["output_tokens"] == 505
            and result["resources"]["cold_swap_max_bytes"] == 0
            and result["resources"]["hot_swap_max_bytes"] == 0,
            "result gates",
        )
        run_trace.write_json(args.output / "RESULT.json", result)
        return 0
    except BaseException as error:
        if args.output.exists():
            run_trace.write_json(args.output / "FAILURE.json", {
                "error": f"{type(error).__name__}: {error}",
                "gpu_load": args.gpu_load,
                "repeat_index": args.repeat_index,
                "schema": "s41-gpu-contention-failure-v1",
                "status": "FAIL",
            })
        return 2
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except BaseException:
                pass
        if saturator is not None and saturator.executor is not None:
            try:
                saturator.stop()
            except BaseException:
                pass
        if cold is not None:
            cold.terminate()
        if bridge is not None:
            bridge.terminate()
        if hot is not None:
            hot.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
