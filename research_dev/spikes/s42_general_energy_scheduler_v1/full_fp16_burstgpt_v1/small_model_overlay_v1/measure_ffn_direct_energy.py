#!/usr/bin/env python3
"""Measure isolated Llama CPU and CPU/phone FFN routes in ABBA order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
S42_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
MIXED_ROOT = S42_ROOT / "mixed_model_trace_v1"
sys.path[:0] = [str(REPO_ROOT), str(S42_ROOT), str(MIXED_ROOT), str(HERE)]

import run_fp16_small_overlay as overlay  # noqa: E402


CONFIRMATION = "RUN_S42_LLAMA_FFN_DIRECT_ENERGY"
SCHEMA = "s42-llama1b-ffn-direct-energy-campaign-v1"
CPU_ROUTE = "desktop-cpu"
SPLIT_ROUTE = "cpu-phone-ffn-split"
MODEL_ID = "llama-3.2-1b-instruct-q4_0"


class MeasurementError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementError(message)


def canonical(value: object) -> bytes:
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


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def selected_requests(path: Path, indices: tuple[int, ...]) -> list[dict[str, Any]]:
    rows = overlay.load_rows(path)
    by_index = {row.get("overlay_request_index"): row for row in rows}
    require(len(by_index) == len(rows), "unique overlay request indices")
    require(set(indices).issubset(by_index), "request index coverage")
    selected = [by_index[index] for index in indices]
    require(
        all(
            row.get("execution_model_id") == MODEL_ID
            and type(row.get("input_tokens")) is int
            and row["input_tokens"] > 0
            and type(row.get("output_tokens")) is int
            and row["output_tokens"] > 0
            and type(row.get("prompt_tokens")) is list
            and len(row["prompt_tokens"]) == row["input_tokens"]
            for row in selected
        ),
        "Llama request identity",
    )
    require(
        len({(row["input_tokens"], row["output_tokens"]) for row in selected})
        >= 3,
        "energy fit geometry",
    )
    return selected


def compact_response(response: dict[str, Any]) -> dict[str, Any]:
    tokens = response.pop("tokens")
    output_text = response.pop("output_text")
    return {
        **response,
        "output_text_sha256": hashlib.sha256(
            output_text.encode("utf-8")
        ).hexdigest(),
        "token_count": len(tokens),
        "tokens_sha256": hashlib.sha256(canonical(tokens)).hexdigest(),
    }


def idle_case(
    case_id: str,
    batch_id: str,
    position: str,
    duration_s: float,
) -> dict[str, Any]:
    started_ns = time.monotonic_ns()
    time.sleep(duration_s)
    completed_ns = time.monotonic_ns()
    return {
        "batch_id": batch_id,
        "case_id": case_id,
        "duration_s": (completed_ns - started_ns) / 1e9,
        "paid_end_monotonic_ns": completed_ns,
        "paid_start_monotonic_ns": started_ns,
        "position": position,
        "route": "idle",
    }


def request_case(
    case_id: str,
    batch_id: str,
    cycle: int,
    pair_id: int,
    route: str,
    row: dict[str, Any],
    port: int,
    output: Path,
) -> dict[str, Any]:
    first_token_ns: list[int] = []
    started_ns = time.monotonic_ns()
    response = overlay.endpoint_completion(
        "127.0.0.1",
        port,
        row,
        output / f"{case_id}.raw",
        first_token_ns.append,
    )
    completed_ns = time.monotonic_ns()
    require(len(first_token_ns) == 1, f"first token receipt: {case_id}")
    return {
        "batch_id": batch_id,
        "case_id": case_id,
        "cycle": cycle,
        "duration_s": (completed_ns - started_ns) / 1e9,
        "first_token_monotonic_ns": first_token_ns[0],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "overlay_request_index": row["overlay_request_index"],
        "paid_end_monotonic_ns": completed_ns,
        "paid_start_monotonic_ns": started_ns,
        "pair_id": pair_id,
        "response": compact_response(response),
        "route": route,
    }


def batch_plan(abba_cycles: int) -> list[tuple[int, int, str]]:
    result = []
    for cycle in range(1, abba_cycles + 1):
        first_pair = 2 * cycle - 1
        second_pair = 2 * cycle
        result.extend((
            (cycle, first_pair, CPU_ROUTE),
            (cycle, first_pair, SPLIT_ROUTE),
            (cycle, second_pair, SPLIT_ROUTE),
            (cycle, second_pair, CPU_ROUTE),
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-cpu", type=Path, required=True)
    parser.add_argument("--cpu-lib-dir", type=Path, required=True)
    parser.add_argument("--ffn-manifest", type=Path, required=True)
    parser.add_argument("--ffn-policy", type=Path, required=True)
    parser.add_argument("--phone-host", required=True)
    parser.add_argument("--phone-ffn-port", type=int, default=18384)
    parser.add_argument("--cpu-port", type=int, default=18484)
    parser.add_argument("--split-port", type=int, default=18486)
    parser.add_argument("--cpus", default="20-23")
    parser.add_argument("--request-indices", default="1,4,5,8,9")
    parser.add_argument("--abba-cycles", type=int, default=2)
    parser.add_argument("--idle-duration-s", type=float, default=5.0)
    parser.add_argument("--split-prewarm-timeout-s", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    require(args.execute and args.confirm == CONFIRMATION, "confirmation")
    require(
        args.output.is_absolute() and not args.output.exists(),
        "new absolute output",
    )
    require(args.abba_cycles >= 2, "at least two ABBA cycles")
    require(args.idle_duration_s >= 3.0, "idle measurement duration")
    require(args.split_prewarm_timeout_s > 0, "split prewarm timeout")
    for path in (
        args.requests,
        args.model,
        args.server_cpu,
        args.cpu_lib_dir,
        args.ffn_manifest,
        args.ffn_policy,
    ):
        require(path.exists(), f"missing input: {path}")
    indices = tuple(int(value) for value in args.request_indices.split(","))
    require(indices and len(indices) == len(set(indices)), "request indices")
    rows = selected_requests(args.requests, indices)
    manifest = overlay.verified_record(
        args.ffn_manifest, overlay.FFN_MANIFEST_SCHEMA
    )
    policy = overlay.verified_record(
        args.ffn_policy, overlay.FFN_POLICY_SCHEMA
    )
    require(
        manifest["model"]["id"] == MODEL_ID
        and manifest["model"]["size_bytes"] == args.model.stat().st_size
        and manifest["model"]["sha256"] == digest(args.model),
        "model and manifest identity",
    )
    require(
        policy.get("evidence", {}).get("manifest_record_sha256")
        == manifest["record_sha256"]
        and policy.get("qualification", {}).get("physical_shape_calibrated")
        is True,
        "physically calibrated policy binding",
    )

    args.output.mkdir(parents=True)
    cpu_spec = overlay.physical.ModelSpec(
        MODEL_ID,
        args.model,
        "cpu",
        args.cpu_port,
        1,
        4096,
        1024,
        512,
        threads=4,
        cpus=args.cpus,
    )
    split_spec = overlay.physical.ModelSpec(
        MODEL_ID,
        args.model,
        "cpu",
        args.split_port,
        1,
        4096,
        1024,
        512,
        threads=4,
        cpus=args.cpus,
    )
    processes = []
    sampler = None
    cases: list[dict[str, Any]] = []
    failure = None
    measurement_start_ns = None
    measurement_end_ns = None
    split_prewarm: dict[str, Any] | None = None
    try:
        cpu_process, cpu_load_ms, cpu_props = overlay.physical.start_server(
            cpu_spec,
            args.server_cpu,
            args.server_cpu,
            args.cpu_lib_dir,
            args.cpu_lib_dir,
            args.output,
            "llama1-cpu",
        )
        processes.append(cpu_process)
        split_process, split_load_ms, split_props = overlay.start_split_server(
            split_spec,
            args.server_cpu,
            args.cpu_lib_dir,
            args.output,
            args.phone_host,
            args.phone_ffn_port,
            manifest,
            policy,
        )
        processes.append(split_process)
        warm = dict(min(rows, key=lambda row: row["input_tokens"]))
        warm["output_tokens"] = 2
        overlay.endpoint_completion(
            "127.0.0.1",
            args.cpu_port,
            warm,
            args.output / f"warm-{CPU_ROUTE}.raw",
            lambda _: None,
        )
        split_warm_row, split_warm_shape = overlay.split_accelerator_prewarm(
            rows, policy
        )
        split_warm_started_ns = time.monotonic_ns()
        split_warm_response = overlay.endpoint_completion(
            "127.0.0.1",
            args.split_port,
            split_warm_row,
            args.output / f"warm-{SPLIT_ROUTE}.raw",
            lambda _: None,
            args.split_prewarm_timeout_s,
        )
        split_prewarm = {
            "duration_s": (
                time.monotonic_ns() - split_warm_started_ns
            ) / 1e9,
            "response": compact_response(split_warm_response),
            "shape": split_warm_shape,
            "status": "PASS",
        }

        sampler = overlay.physical.DynamicSampler(args.output)
        sampler.set_pid("llama1-cpu", cpu_process.pid)
        sampler.set_pid("llama1-cpu-phone-ffn", split_process.pid)
        sampler.start()
        time.sleep(1.0)
        measurement_start_ns = time.monotonic_ns()
        plan = batch_plan(args.abba_cycles)
        for batch_number, (cycle, pair_id, route) in enumerate(plan, 1):
            batch_id = f"c{cycle}-b{batch_number}-{route}"
            cases.append(idle_case(
                f"idle-before-{batch_id}",
                batch_id,
                "before",
                args.idle_duration_s,
            ))
            ordered = rows if batch_number % 2 else list(reversed(rows))
            port = args.cpu_port if route == CPU_ROUTE else args.split_port
            for row in ordered:
                case_id = (
                    f"{route}-q{row['overlay_request_index']}-"
                    f"c{cycle}-b{batch_number}"
                )
                cases.append(request_case(
                    case_id,
                    batch_id,
                    cycle,
                    pair_id,
                    route,
                    row,
                    port,
                    args.output,
                ))
            cases.append(idle_case(
                f"idle-after-{batch_id}",
                batch_id,
                "after",
                args.idle_duration_s,
            ))
        measurement_end_ns = time.monotonic_ns()
        time.sleep(1.0)
        sampler.stop()
        samples = list(sampler.rows)
        sampler = None
        require(
            samples
            and samples[0]["t_ns"] < cases[0]["paid_start_monotonic_ns"]
            and samples[-1]["t_ns"] > cases[-1]["paid_end_monotonic_ns"],
            "host sample coverage",
        )
        paid_rows = [
            sample
            for sample in samples
            if measurement_start_ns <= sample["t_ns"] <= measurement_end_ns
        ]
        require(
            paid_rows
            and max(sample["gpu"]["utilization_pct"] for sample in paid_rows)
            <= 5,
            "GPU was not idle",
        )
        for process in reversed(processes):
            process.terminate()
        processes.clear()
        result = {
            "abba_cycles": args.abba_cycles,
            "batch_plan": [
                {"cycle": cycle, "pair_id": pair_id, "route": route}
                for cycle, pair_id, route in plan
            ],
            "cases": cases,
            "execution": {
                "cpu": {
                    "load_ms": cpu_load_ms,
                    "model_alias": cpu_props.get("model_alias"),
                    "parallel": 1,
                    "threads": 4,
                },
                "split": {
                    "load_ms": split_load_ms,
                    "model_alias": split_props.get("model_alias"),
                    "parallel": 1,
                    "policy_text": policy["policy_text"],
                    "threads": 4,
                },
            },
            "input_sha256": {
                "ffn_manifest": digest(args.ffn_manifest),
                "ffn_policy": digest(args.ffn_policy),
                "model": digest(args.model),
                "requests": digest(args.requests),
            },
            "measurement_end_monotonic_ns": measurement_end_ns,
            "measurement_start_monotonic_ns": measurement_start_ns,
            "model_id": MODEL_ID,
            "request_indices": list(indices),
            "resource_samples": {
                "count": len(samples),
                "path": "resource-samples.jsonl",
                "sha256": digest(args.output / "resource-samples.jsonl"),
            },
            "routes": [CPU_ROUTE, SPLIT_ROUTE],
            "schema": SCHEMA,
            "split_accelerator_prewarm": split_prewarm,
            "status": "PASS",
        }
        (args.output / "RESULT.json").write_bytes(canonical(result))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if sampler is not None:
            try:
                sampler.stop()
            except BaseException:
                pass
        for process in reversed(processes):
            process.terminate()
    if failure is not None:
        (args.output / "FAILURE.json").write_bytes(canonical({
            "error": failure,
            "schema": "s42-llama1b-ffn-direct-energy-failure-v1",
            "status": "FAIL",
        }))
        raise MeasurementError(failure)
    print(json.dumps({
        "cases": len(cases),
        "output": str(args.output / "RESULT.json"),
        "status": "PASS",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MeasurementError as error:
        print(f"S42_FFN_DIRECT_MEASUREMENT_ERROR: {error}")
        raise SystemExit(2)
