#!/usr/bin/env python3
"""Run exact Gemma BurstGPT rows through a persistent wavefront route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any


SCHEMA = "s42-gemma-wavefront-driver-v1"
COMMAND_SCHEMA_V2 = "layersplit-persistent-command-v2"
COMMAND_SCHEMA_V3 = "layersplit-persistent-command-v3"
PREFILL_MODES = ("async_stream", "sync_chunked")


class DriverError(RuntimeError):
    pass


def parse_cpu_list(text: str) -> tuple[int, ...]:
    values: set[int] = set()
    for item in text.split(","):
        fields = item.split("-", 1)
        try:
            first = int(fields[0])
            last = int(fields[-1])
        except (IndexError, ValueError) as exc:
            raise argparse.ArgumentTypeError("CPU list must contain integers") from exc
        if first < 0 or last < first:
            raise argparse.ArgumentTypeError("CPU list has invalid bounds")
        values.update(range(first, last + 1))
    if not values:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    return tuple(sorted(values))


def format_cpu_list(cpus: tuple[int, ...] | set[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))


def process_affinity(pid: int, expected: tuple[int, ...]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for task in sorted(
        Path(f"/proc/{pid}/task").iterdir(),
        key=lambda path: int(path.name),
    ):
        try:
            cpus = set(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:
            continue
        key = format_cpu_list(cpus)
        counts[key] = counts.get(key, 0) + 1
    if not counts or any(
        not set(parse_cpu_list(cpu_text)) <= set(expected)
        for cpu_text in counts
    ):
        raise DriverError("persistent Gemma driver affinity is invalid")
    return {
        "cpu_sets": counts,
        "thread_count": sum(counts.values()),
    }


def set_process_affinity(
    pid: int, expected: tuple[int, ...]
) -> dict[str, object]:
    target = set(expected)
    for _ in range(16):
        try:
            tasks_before = {
                int(task.name) for task in Path(f"/proc/{pid}/task").iterdir()
            }
        except FileNotFoundError as exc:
            raise DriverError("persistent Gemma driver ended before affinity switch") from exc
        for task_id in sorted(tasks_before):
            try:
                os.sched_setaffinity(task_id, target)
            except ProcessLookupError:
                continue
        try:
            tasks_after = {
                int(task.name) for task in Path(f"/proc/{pid}/task").iterdir()
            }
            receipt = process_affinity(pid, expected)
        except (DriverError, FileNotFoundError):
            time.sleep(0.001)
            continue
        if tasks_before == tasks_after:
            return receipt
        time.sleep(0.001)
    raise DriverError("persistent Gemma driver affinity did not stabilize")


def switch_affinity_after_receipt(
    process: subprocess.Popen[str],
    receipt_path: Path,
    protected_cpus: tuple[int, ...],
    target_cpus: tuple[int, ...],
    timeout_s: float,
    result: dict[str, object],
) -> None:
    try:
        protected_done_ns = wait_for(
            receipt_path, process, timeout_s, "protected completion"
        )
        before = process_affinity(process.pid, protected_cpus)
        switch_started_ns = time.monotonic_ns()
        after = set_process_affinity(process.pid, target_cpus)
        switch_completed_ns = time.monotonic_ns()
        result.update({
            "after": after,
            "before": before,
            "protected_done_ns": protected_done_ns,
            "switch_completed_ns": switch_completed_ns,
            "switch_started_ns": switch_started_ns,
            "status": "PASS",
        })
    except BaseException as exc:
        result.update({
            "error": f"{type(exc).__name__}: {exc}",
            "status": "FAIL",
        })


def parse_indices(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value) for value in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be integers") from exc
    if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("indices must be non-negative and unique")
    return values


def load_requests(path: Path, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            index = row.get("request_index")
            if index in indices:
                selected[index] = row
    if set(selected) != set(indices):
        raise DriverError("requested Gemma rows are missing")
    result = [selected[index] for index in indices]
    for row in result:
        tokens = row.get("prompt_tokens")
        if (
            type(tokens) is not list
            or len(tokens) != row.get("input_tokens")
            or any(type(token) is not int or token < 0 for token in tokens)
            or type(row.get("output_tokens")) is not int
            or row["output_tokens"] <= 0
        ):
            raise DriverError("Gemma row has invalid exact token semantics")
    return result


def command(
    launch_id: int,
    prompt_tokens: list[int],
    n_gen: int,
    session_end: str,
    ffn_prefill_columns: int | None = None,
    ffn_decode_columns: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "launch_id": launch_id,
        "n_gen": n_gen,
        "prompt_tokens": prompt_tokens,
        "request_count": 1,
        "schema": (
            COMMAND_SCHEMA_V3
            if ffn_prefill_columns is not None
            else COMMAND_SCHEMA_V2
        ),
        "session_end": session_end,
    }
    if ffn_prefill_columns is not None:
        if ffn_decode_columns is None:
            raise DriverError("dynamic FFN widths are incomplete")
        value["ffn_prefill_columns"] = ffn_prefill_columns
        value["ffn_decode_columns"] = ffn_decode_columns
    return value


def write_command(process: subprocess.Popen[str], value: dict[str, object]) -> None:
    if process.stdin is None:
        raise DriverError("persistent driver stdin is absent")
    process.stdin.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
    process.stdin.flush()


def read_reply(
    process: subprocess.Popen[str],
    launch_id: int,
    dynamic_ffn: bool = False,
    prefill_mode: str | None = None,
) -> dict[str, object]:
    if process.stdout is None:
        raise DriverError("persistent driver stdout is absent")
    line = process.stdout.readline()
    if not line:
        raise DriverError("persistent driver ended before its reply")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DriverError("persistent driver emitted invalid JSON") from exc
    if (
        type(value) is not dict
        or value.get("schema") != (
            "layersplit-persistent-result-v3"
            if dynamic_ffn
            else "layersplit-persistent-result-v2"
        )
        or value.get("launch_id") != launch_id
        or value.get("outcome") != "completed"
        or value.get("request_count") != 1
        or value.get("batch_size") != 1
    ):
        raise DriverError("persistent driver reply is invalid")
    if prefill_mode is not None:
        if prefill_mode not in PREFILL_MODES:
            raise DriverError("unknown persistent driver prefill mode")
        integer_fields = (
            "prefill_chunks",
            "prefill_chunk_tokens",
            "prefill_stage_us",
            "prefill_tail_us",
            "prefill_pipeline_wall_us",
            "prefill_overlap_us",
            "prefill_max_ready_depth",
        )
        if any(type(value.get(name)) is not int for name in integer_fields):
            raise DriverError("persistent driver chunked prefill receipt is invalid")
        if (
            value["prefill_chunks"] <= 0
            or value.get("prefill_mode") != prefill_mode
            or value["prefill_chunk_tokens"] <= 0
            or value["prefill_stage_us"] <= 0
            or value["prefill_tail_us"] <= 0
            or value["prefill_pipeline_wall_us"] <= 0
            or value["prefill_overlap_us"] < 0
            or value["prefill_max_ready_depth"] <= 0
            or value["prefill_max_ready_depth"] > value["prefill_chunks"]
            or value.get("prefill_us")
            != value["prefill_stage_us"] + value["prefill_tail_us"]
            or value["prefill_overlap_us"]
            != max(
                0,
                value["prefill_stage_us"]
                + value["prefill_tail_us"]
                - value["prefill_pipeline_wall_us"],
            )
        ):
            raise DriverError("persistent driver chunked prefill receipt is invalid")
    return value


def wait_for(path: Path, process: subprocess.Popen[str], timeout_s: float, name: str) -> int:
    deadline = time.monotonic() + timeout_s
    while True:
        if process.poll() is not None:
            raise DriverError(f"persistent driver ended before {name}")
        if time.monotonic() >= deadline:
            raise DriverError(f"timed out waiting for {name}")
        try:
            text = path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            text = ""
        except (OSError, UnicodeError) as exc:
            raise DriverError(f"invalid {name} receipt") from exc
        if text:
            try:
                value = int(text)
            except ValueError as exc:
                raise DriverError(f"invalid {name} receipt") from exc
            break
        time.sleep(0.01)
    if value <= 0 or value > time.monotonic_ns():
        raise DriverError(f"{name} receipt is outside monotonic time")
    return value


def write_monotonic_receipt(path: Path, value_ns: int) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(f"{value_ns}\n", encoding="ascii")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def wait_ready(
    stderr_path: Path,
    process: subprocess.Popen[str],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            text = stderr_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            text = ""
        if "PERSISTENT_DRIVER_READY " in text:
            return
        if process.poll() is not None:
            raise DriverError("persistent driver failed during model load")
        if time.monotonic() >= deadline:
            raise DriverError("timed out loading the persistent Gemma driver")
        time.sleep(0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layersplit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--indices", type=parse_indices, required=True)
    parser.add_argument("--arm-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--route-kind", choices=("lm-head", "gpu-prefix"), default="lm-head"
    )
    parser.add_argument("--gate-host", default="127.0.0.1")
    parser.add_argument("--gate-port", type=int, required=True)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--gpu-prefix-layers", type=int)
    parser.add_argument("--ffn-host")
    parser.add_argument("--ffn-port", type=int)
    parser.add_argument("--ffn-layers")
    parser.add_argument("--ffn-columns", type=int)
    parser.add_argument("--ffn-prefill-columns", type=int)
    parser.add_argument("--ffn-decode-columns", type=int)
    parser.add_argument("--ffn-timeout-ms", type=int, default=120000)
    parser.add_argument("--ffn-f16-io", action="store_true")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-list", type=parse_cpu_list, required=True)
    parser.add_argument("--protected-cpu-list", type=parse_cpu_list)
    parser.add_argument("--protected-done-file", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--max-prefill", type=int, default=512)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--prefill-mode", choices=PREFILL_MODES)
    parser.add_argument("--maximum-output", type=int, default=2048)
    parser.add_argument("--library-path", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-timeout-s", type=float, default=300.0)
    parser.add_argument("--arm-timeout-s", type=float, default=3600.0)
    args = parser.parse_args()

    for path in (args.layersplit, args.model, args.requests):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if not args.library_path.is_dir():
        parser.error("library path is missing")
    for path in (args.arm_file, args.ready_file, args.stderr, args.output):
        if not path.is_absolute():
            parser.error("runtime paths must be absolute")
    if (
        args.arm_file.exists()
        or args.ready_file.exists()
        or args.stderr.exists()
        or args.output.exists()
        or not args.ready_file.parent.is_dir()
        or not args.stderr.parent.is_dir()
        or not args.output.parent.is_dir()
    ):
        parser.error("runtime paths must be unused")
    if not (0 < args.gate_port <= 65535):
        parser.error("invalid gate port")
    ffn_values = (
        args.ffn_host,
        args.ffn_port,
        args.ffn_layers,
        args.ffn_columns,
        args.ffn_prefill_columns,
        args.ffn_decode_columns,
    )
    dynamic_ffn = any(value is not None for value in ffn_values)
    if dynamic_ffn and any(value is None for value in ffn_values):
        parser.error("FFN route arguments must be provided together")
    if dynamic_ffn and (
        not args.ffn_host
        or not args.ffn_layers
        or not (0 < args.ffn_port <= 65535)
        or args.ffn_columns <= 0
        or not (0 <= args.ffn_prefill_columns <= args.ffn_columns)
        or not (0 <= args.ffn_decode_columns <= args.ffn_columns)
        or not args.ffn_f16_io
    ):
        parser.error("invalid dynamic FFN route")
    if not dynamic_ffn and args.ffn_f16_io:
        parser.error("--ffn-f16-io requires a dynamic FFN route")
    if not 1 <= args.ffn_timeout_ms <= 600000:
        parser.error("invalid FFN timeout")
    protected_cpus = args.protected_cpu_list or args.cpu_list
    phase_switch = protected_cpus != args.cpu_list
    if phase_switch != (args.protected_done_file is not None):
        parser.error(
            "a protected completion receipt is required exactly when CPU affinity changes"
        )
    if args.protected_done_file is not None and (
        not args.protected_done_file.is_absolute()
        or not args.protected_done_file.parent.is_dir()
    ):
        parser.error("protected completion receipt path is invalid")
    if args.route_kind == "lm-head" and (
        args.rows is None
        or args.top_k is None
        or args.rows <= 0
        or args.top_k <= 0
        or args.gpu_prefix_layers is not None
    ):
        parser.error("LM-head route requires positive rows and top-k only")
    if args.route_kind == "gpu-prefix" and (
        args.rows is not None
        or args.top_k is not None
        or args.gpu_prefix_layers is None
        or not 1 <= args.gpu_prefix_layers < 48
    ):
        parser.error("GPU-prefix route requires a layer count from 1 through 47")
    if args.route_kind == "lm-head" and args.prefill_mode is not None:
        parser.error("--prefill-mode requires the GPU-prefix route")
    prefill_mode = (
        args.prefill_mode or "async_stream"
        if args.route_kind == "gpu-prefix"
        else None
    )
    if min(
        args.threads,
        args.context,
        args.max_prefill,
        args.ubatch,
        args.maximum_output,
    ) <= 0:
        parser.error("invalid driver geometry")
    if args.ubatch > 512:
        parser.error("ubatch must not exceed 512")
    available_cpus = set(os.sched_getaffinity(0))
    if (
        not set(args.cpu_list) <= available_cpus
        or not set(protected_cpus) <= available_cpus
    ):
        parser.error("requested CPU is unavailable")

    rows = load_requests(args.requests, args.indices)
    if any(
        len(row["prompt_tokens"]) > args.max_prefill
        or len(row["prompt_tokens"]) + row["output_tokens"] > args.context
        or row["output_tokens"] > args.maximum_output
        for row in rows
    ):
        parser.error("selected request exceeds driver bounds")

    process_command = [
        "taskset", "--cpu-list", format_cpu_list(protected_cpus),
        str(args.layersplit),
        "-m", str(args.model),
        "-n", str(args.maximum_output),
        "--driver-requests", "1",
        "--driver-batch", "1",
        "--driver-context", str(args.context),
        "--driver-max-prefill", str(args.max_prefill),
        "--driver-ubatch", str(args.ubatch),
        "--persistent-jsonl",
        "-ngl", "0",
        "-t", str(args.threads),
        "-tb", str(args.threads),
    ]
    if args.route_kind == "lm-head":
        process_command.extend([
            "--mode", "overlapdriver",
            "--host", args.ffn_host if dynamic_ffn else "127.0.0.1",
            "--lm-head-host", args.gate_host,
            "--lm-head-port", str(args.gate_port),
            "--lm-head-rows", str(args.rows),
            "--lm-head-top-k", str(args.top_k),
            "--lm-head-f16-io",
        ])
    else:
        process_command.extend([
            "--mode", "pipedriver",
            "--host", args.gate_host,
            "--port", str(args.gate_port),
            (
                "--driver-stream-prefill"
                if prefill_mode == "async_stream"
                else "--driver-sync-prefill"
            ),
        ])
    if dynamic_ffn:
        if args.route_kind == "gpu-prefix":
            process_command.extend([
                "--ffn-host", args.ffn_host,
                "--ffn-port", str(args.ffn_port),
            ])
        else:
            process_command.extend(["--port", str(args.ffn_port)])
        process_command.extend([
            "--ffn-layers", args.ffn_layers,
            "--ffn-columns", str(args.ffn_columns),
            "--ffn-timeout-ms", str(args.ffn_timeout_ms),
            "--ffn-f16-io",
        ])
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment.pop("CUDA_MPS_PIPE_DIRECTORY", None)
    environment.pop("CUDA_MPS_LOG_DIRECTORY", None)
    environment.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    environment.pop("CUDA_MPS_CLIENT_PRIORITY", None)
    environment["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    if args.route_kind == "gpu-prefix":
        environment["LLAMA_LAYER_START"] = str(args.gpu_prefix_layers)
    environment["LD_LIBRARY_PATH"] = (
        str(args.library_path)
        + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    )

    started_ns = time.monotonic_ns()
    request_results: list[dict[str, object]] = []
    affinity_switch: dict[str, object] | None = None
    affinity_thread: threading.Thread | None = None
    with args.stderr.open("x", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            process_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=environment,
        )
        try:
            wait_ready(args.stderr, process, args.load_timeout_s)
            affinity_at_ready = process_affinity(process.pid, protected_cpus)
            warm_tokens = rows[0]["prompt_tokens"][: min(
                args.ubatch, len(rows[0]["prompt_tokens"])
            )]
            write_command(process, command(
                1,
                warm_tokens,
                1,
                "DETACH",
                0 if dynamic_ffn else None,
                0 if dynamic_ffn else None,
            ))
            warmup = read_reply(
                process,
                1,
                dynamic_ffn,
                prefill_mode,
            )
            ready_ns = time.monotonic_ns()
            write_monotonic_receipt(args.ready_file, ready_ns)
            arm_ns = wait_for(args.arm_file, process, args.arm_timeout_s, "paid arm")
            affinity_at_paid_start = process_affinity(process.pid, protected_cpus)
            if phase_switch:
                assert args.protected_done_file is not None
                affinity_switch = {}
                affinity_thread = threading.Thread(
                    target=switch_affinity_after_receipt,
                    args=(
                        process,
                        args.protected_done_file,
                        protected_cpus,
                        args.cpu_list,
                        args.arm_timeout_s,
                        affinity_switch,
                    ),
                    name="gemma-affinity-switch",
                )
                affinity_thread.start()

            for offset, row in enumerate(rows):
                launch_id = offset + 2
                session_end = "STOP" if offset + 1 == len(rows) else "DETACH"
                dispatch_ns = time.monotonic_ns()
                write_command(process, command(
                    launch_id,
                    row["prompt_tokens"],
                    row["output_tokens"],
                    session_end,
                    args.ffn_prefill_columns if dynamic_ffn else None,
                    args.ffn_decode_columns if dynamic_ffn else None,
                ))
                reply = read_reply(
                    process,
                    launch_id,
                    dynamic_ffn,
                    prefill_mode,
                )
                completion_ns = time.monotonic_ns()
                token_ids = reply.get("token_ids")
                if (
                    type(token_ids) is not list
                    or len(token_ids) != 1
                    or type(token_ids[0]) is not list
                    or len(token_ids[0]) != row["output_tokens"]
                ):
                    raise DriverError("persistent driver output length is not exact")
                request_results.append({
                    "completion_ns": completion_ns,
                    "decode_us": reply.get("decode_us"),
                    "dispatch_ns": dispatch_ns,
                    "input_tokens": row["input_tokens"],
                    "launch_id": launch_id,
                    "output_tokens": row["output_tokens"],
                    "prefill_chunk_tokens": reply.get("prefill_chunk_tokens"),
                    "prefill_chunks": reply.get("prefill_chunks"),
                    "prefill_max_ready_depth": reply.get(
                        "prefill_max_ready_depth"
                    ),
                    "prefill_overlap_us": reply.get("prefill_overlap_us"),
                    "prefill_mode": reply.get("prefill_mode"),
                    "prefill_pipeline_wall_us": reply.get(
                        "prefill_pipeline_wall_us"
                    ),
                    "prefill_stage_us": reply.get("prefill_stage_us"),
                    "prefill_tail_us": reply.get("prefill_tail_us"),
                    "prefill_us": reply.get("prefill_us"),
                    "request_index": row["request_index"],
                    "route_wall_us": reply.get("route_wall_us"),
                    "token_ids": token_ids[0],
                })
            if process.stdin is not None:
                process.stdin.close()
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise DriverError("persistent Gemma driver failed")
            if affinity_thread is not None:
                affinity_thread.join(timeout=1)
                if affinity_thread.is_alive():
                    raise DriverError("Gemma affinity switch did not finish")
                if affinity_switch is None or affinity_switch.get("status") != "PASS":
                    raise DriverError("Gemma affinity switch failed")
        except BaseException:
            process.kill()
            process.wait()
            if affinity_thread is not None:
                affinity_thread.join(timeout=1)
            raise

    result = {
        "arm_ns": arm_ns,
        "driver_command": process_command,
        "finished_ns": time.monotonic_ns(),
        "indices": list(args.indices),
        "route_kind": args.route_kind,
        "gpu_prefix_layers": args.gpu_prefix_layers,
        "prefill_mode": prefill_mode,
        "stream_prefill": prefill_mode == "async_stream",
        "ubatch": args.ubatch,
        "ffn_route": ({
            "columns": args.ffn_columns,
            "decode_columns": args.ffn_decode_columns,
            "f16_io": args.ffn_f16_io,
            "host": args.ffn_host,
            "layers": args.ffn_layers,
            "port": args.ffn_port,
            "prefill_columns": args.ffn_prefill_columns,
            "timeout_ms": args.ffn_timeout_ms,
        } if dynamic_ffn else None),
        "producer_affinity": {
            "at_paid_start": affinity_at_paid_start,
            "at_ready": affinity_at_ready,
            "post_protected_switch": affinity_switch,
            "protected_requested_cpus": list(protected_cpus),
            "requested_cpus": list(args.cpu_list),
        },
        "producer_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "LLAMA_LAYER_START": environment.get("LLAMA_LAYER_START"),
        },
        "request_results": request_results,
        "ready_ns": ready_ns,
        "schema": SCHEMA,
        "started_ns": started_ns,
        "status": "PASS",
        "warmup": warmup,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
