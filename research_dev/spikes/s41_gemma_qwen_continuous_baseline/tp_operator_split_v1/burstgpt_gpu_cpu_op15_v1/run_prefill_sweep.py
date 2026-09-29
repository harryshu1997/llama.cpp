#!/usr/bin/env python3
"""Calibrate dynamic phone FFN width for representative cold prefill shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

import run_trace


SHAPES = (53, 76, 303, 512)
WIDTHS = (0, 4096, 6144, 8192, 9664)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--cold-driver", type=Path, required=True)
    parser.add_argument("--cold-model", type=Path, required=True)
    parser.add_argument("--cold-lib-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--bridge-port", type=int, default=25660)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    run_trace.require(
        args.execute and args.confirm == "RUN_DYNAMIC_PREFILL_SWEEP",
        "confirmation",
    )
    run_trace.require(
        args.output.is_absolute() and not args.output.exists(), "output"
    )
    for path in (
        args.requests,
        args.cold_driver,
        args.cold_model,
        args.cold_lib_dir,
        args.bridge,
    ):
        run_trace.require(path.exists(), f"missing path: {path}")
    run_trace.require(
        run_trace.digest_file(args.requests) == run_trace.REQUESTS_SHA256,
        "request identity",
    )
    run_trace.require(
        args.cold_model.stat().st_size == 6_975_878_176
        and run_trace.digest_file(args.cold_model)
            == run_trace.COLD_MODEL_SHA256,
        "cold model identity",
    )

    source = run_trace.read_jsonl(args.requests)
    cold = [row for row in source if run_trace.role(row) == "cold"]
    prompts = {
        shape: next(row["prompt_tokens"] for row in cold
                    if row["input_tokens"] == shape)
        for shape in SHAPES
    }

    args.mode = "op15"
    args.max_columns = 9664
    args.decode_columns = 0
    args.output.mkdir(parents=True)
    bridge: run_trace.CapturedProcess | None = None
    driver: run_trace.ColdDriver | None = None
    rows: list[dict[str, Any]] = []
    try:
        bridge = run_trace.start_bridge(args, args.output)
        driver = run_trace.start_cold(args, args.output)
        launch_id = 0

        def execute(shape: int, width: int, measured: bool,
                    session_end: str = "DETACH") -> None:
            nonlocal launch_id
            launch_id += 1
            request = {
                "output_tokens": 2,
                "prompt_tokens": prompts[shape],
            }
            result = driver.exchange(
                launch_id,
                request,
                session_end,
                width,
                0,
            )
            if measured:
                rows.append({
                    "decode_us": result["decode_us"],
                    "launch_id": launch_id,
                    "prefill_tokens": shape,
                    "prefill_us": result["prefill_us"],
                    "route_wall_us": result["route_wall_us"],
                    "width": width,
                })

        for shape in SHAPES:
            execute(shape, 9664, False)

        ordered_cases: list[tuple[int, int]] = []
        for width_order in (WIDTHS, tuple(reversed(WIDTHS))):
            for shape_index, shape in enumerate(SHAPES):
                rotated = (
                    width_order[shape_index:] + width_order[:shape_index]
                )
                ordered_cases.extend((shape, width) for width in rotated)
        for index, (shape, width) in enumerate(ordered_cases):
            execute(
                shape,
                width,
                True,
                "STOP" if index + 1 == len(ordered_cases) else "DETACH",
            )

        assert driver.process is not None and bridge.process is not None
        driver.process.wait(timeout=60)
        bridge.process.wait(timeout=60)
        run_trace.require(driver.process.returncode == 0, "driver status")
        run_trace.require(bridge.process.returncode == 0, "bridge status")

        grouped: list[dict[str, Any]] = []
        for shape in SHAPES:
            for width in WIDTHS:
                samples = [
                    row["prefill_us"] for row in rows
                    if row["prefill_tokens"] == shape
                    and row["width"] == width
                ]
                run_trace.require(len(samples) == 2, "sample conservation")
                grouped.append({
                    "prefill_tokens": shape,
                    "prefill_us_median": round(statistics.median(samples)),
                    "prefill_us_samples": samples,
                    "width": width,
                })

        ffn_lines = [
            line for line in driver.stderr_lines if line.startswith("FFNSPLIT ")
        ]
        bridge_lines = [
            line for line in bridge.stderr_lines if line.startswith("FFNDMABUF ")
        ]
        run_trace.require(
            len(ffn_lines) == 1 and len(bridge_lines) == 1,
            "offload summaries",
        )
        result = {
            "cases": rows,
            "cold_driver_sha256": run_trace.digest_file(args.cold_driver),
            "cold_model_sha256": run_trace.COLD_MODEL_SHA256,
            "dmabuf": json.loads(bridge_lines[0].split(" ", 1)[1]),
            "ffn": json.loads(ffn_lines[0].split(" ", 1)[1]),
            "grouped": grouped,
            "schema": "s41-dynamic-prefill-sweep-v1",
            "status": "PASS",
        }
        run_trace.write_json(args.output / "RESULT.json", result)
        return 0
    except BaseException as error:
        if args.output.exists():
            run_trace.write_json(args.output / "FAILURE.json", {
                "error": f"{type(error).__name__}: {error}",
                "schema": "s41-dynamic-prefill-sweep-failure-v1",
                "status": "FAIL",
            })
        return 2
    finally:
        if driver is not None:
            driver.terminate()
        if bridge is not None:
            bridge.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
