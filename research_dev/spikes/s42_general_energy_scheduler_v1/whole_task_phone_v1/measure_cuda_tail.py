#!/usr/bin/env python3
"""Measure the RTX 4060 Ti post-response return to resident P8 idle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


HERE = Path(__file__).resolve().parent
KERNEL_ENERGY = HERE.parent / "kernel_energy_v1"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(KERNEL_ENERGY) not in sys.path:
    sys.path.insert(0, str(KERNEL_ENERGY))

import energy_common  # noqa: E402
import measure_desktop  # noqa: E402
import run_resident_route_campaign as campaign  # noqa: E402


CONFIRMATION = "RUN_S42_CUDA_TAIL_V1"
SCHEMA = "s42-cuda-resident-tail-result-v1"


def integrate_gpu(
    rows: list[dict[str, Any]], start_ns: int, end_ns: int
) -> float:
    return energy_common.integrate_power(
        [
            (int(row["monotonic_ns"]), row["gpu_power_mw"] / 1000.0)
            for row in rows
        ],
        start_ns,
        end_ns,
    )


def measure(
    row: dict[str, Any], repetition: int, cuda_port: int
) -> dict[str, Any]:
    idle_snapshot = campaign.wait_gpu_idle()
    sampler = measure_desktop.Sampler(0.1)
    sampler.start()
    time.sleep(0.7)
    request_start_ns = time.monotonic_ns()
    response = campaign.completion(cuda_port, row)
    response_end_ns = time.monotonic_ns()
    settled_snapshot = campaign.wait_gpu_idle()
    settled_ns = time.monotonic_ns()
    time.sleep(0.3)
    sampler.stop()

    idle_rows = [
        sample
        for sample in sampler.rows
        if sample["monotonic_ns"] < request_start_ns
    ]
    campaign.require(len(idle_rows) >= 3, "CUDA tail idle samples")
    idle_gpu_power_w = sum(
        sample["gpu_power_mw"] / 1000.0 for sample in idle_rows
    ) / len(idle_rows)
    tail_duration_s = (settled_ns - response_end_ns) / 1e9
    tail_gpu_energy_j = integrate_gpu(
        sampler.rows, response_end_ns, settled_ns
    )
    tail_idle_energy_j = idle_gpu_power_w * tail_duration_s
    return {
        "active_cpu_package_energy_j": energy_common.integrate_rapl(
            sampler.rows,
            request_start_ns,
            response_end_ns,
            "monotonic_ns",
        ),
        "active_duration_s": (response_end_ns - request_start_ns) / 1e9,
        "active_gpu_board_energy_j": integrate_gpu(
            sampler.rows, request_start_ns, response_end_ns
        ),
        "idle_gpu_power_w": idle_gpu_power_w,
        "idle_snapshot": idle_snapshot,
        "input_tokens": row["input_tokens"],
        "mixed_request_index": row["mixed_request_index"],
        "output_tokens": row["output_tokens"],
        "repetition": repetition,
        "response": response,
        "settled_snapshot": settled_snapshot,
        "tail_duration_s": tail_duration_s,
        "tail_gpu_board_energy_j": tail_gpu_energy_j,
        "tail_gpu_dynamic_energy_j": max(
            0.0, tail_gpu_energy_j - tail_idle_energy_j
        ),
        "tail_gpu_idle_energy_j": tail_idle_energy_j,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--desktop-model", type=Path, required=True)
    parser.add_argument("--desktop-cuda-server", type=Path, required=True)
    parser.add_argument("--desktop-cpu-server", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path, required=True)
    parser.add_argument("--cpu-lib-dir", type=Path, required=True)
    parser.add_argument(
        "--phone-bin-dir", default="/data/local/tmp/llama-ubatch-op15/bin"
    )
    parser.add_argument(
        "--phone-model",
        default=(
            "/data/local/tmp/unifer/llamacpp/"
            "Llama-3.2-1B-Instruct-Q4_0.gguf"
        ),
    )
    parser.add_argument("--request-index", type=int, default=50)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cuda-port", type=int, default=19480)
    parser.add_argument("--cpu-port", type=int, default=19481)
    parser.add_argument("--phone-port", type=int, default=18482)
    parser.add_argument("--phone-forward-port", type=int, default=29482)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    campaign.require(
        args.execute and args.confirm == CONFIRMATION, "confirmation"
    )
    campaign.require(args.repetitions >= 1, "repetitions")
    row = campaign.read_trace(args.trace, (args.request_index,))[0]
    identity = campaign.preflight(
        argparse.Namespace(**{
            **vars(args),
            "confirm": campaign.CONFIRMATION,
        })
    )
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_bytes(campaign.canonical(identity))

    processes: list[campaign.ManagedProcess] = []
    phone_pid: int | None = None
    failure: str | None = None
    cases: list[dict[str, Any]] = []
    try:
        processes, ports, phone_pid = campaign.start_servers(args, args.output)
        warm = dict(row)
        warm["output_tokens"] = 2
        for route in campaign.ROUTES:
            campaign.completion(ports[route], warm)
        for repetition in range(1, args.repetitions + 1):
            cases.append(measure(row, repetition, args.cuda_port))
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        campaign.stop_servers(
            processes, phone_pid, args.phone_forward_port
        )

    if failure is not None:
        (args.output / "FAILURE.json").write_bytes(campaign.canonical({
            "error": failure,
            "schema": "s42-cuda-resident-tail-failure-v1",
            "status": "FAIL",
        }))
        raise campaign.CampaignError(failure)
    result = {
        "cases": cases,
        "identity": identity,
        "model_id": campaign.MODEL_ID,
        "schema": SCHEMA,
        "status": "PASS",
    }
    (args.output / "RESULT.json").write_bytes(campaign.canonical(result))
    print(json.dumps({
        "output": str(args.output / "RESULT.json"),
        "status": "PASS",
        "tail_dynamic_energy_j_mean": sum(
            case["tail_gpu_dynamic_energy_j"] for case in cases
        ) / len(cases),
        "tail_duration_s_mean": sum(
            case["tail_duration_s"] for case in cases
        ) / len(cases),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except campaign.CampaignError as error:
        print(f"S42_CUDA_TAIL_ERROR: {error}")
        raise SystemExit(2)
