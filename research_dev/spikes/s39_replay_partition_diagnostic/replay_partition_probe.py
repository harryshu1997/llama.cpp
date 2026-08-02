#!/usr/bin/env python3
"""Capture one raw CUDA replay-partition run without comparing outputs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import replay_partition as rp
from async_pipeline import parse_endpoint
from stage_v3_client import ProtocolError, StageV3Client


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cuda-route", type=parse_endpoint, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.timeout <= 0 or args.output.exists():
        parser.error("invalid timeout or existing output")

    raw_client: StageV3Client | None = None
    recording: rp.RecordingClient | None = None
    finished = False
    try:
        contract, contract_sha256 = rp.load_contract(args.contract)
        inputs = rp.load_inputs(args.inputs, contract)
        matches = [
            run
            for run in contract["execution"]["runs"]
            if run["name"] == args.run_name
        ]
        rp.require(len(matches) == 1, "run name is not in the frozen contract")
        run_spec = matches[0]
        started_ns = time.monotonic_ns()
        raw_client = StageV3Client.connect(*args.cuda_route, args.timeout)
        hello = raw_client.hello()
        rp.validate_hello(hello, contract)
        recording = rp.RecordingClient(raw_client)
        initial_status = recording.status()
        rp.require(initial_status.active_sequences == 0, "route starts with live state")
        paths = [
            rp.execute_path(
                recording,
                path_spec,
                inputs,
                contract["execution"]["continuation_tokens"],
            )
            for path_spec in run_spec["paths"]
        ]
        final_status = recording.status()
        rp.require(final_status.active_sequences == 0, "route ends with live state")
        report = {
            "calls": recording.calls,
            "capture_order": [
                "INPUT_ROWS",
                "POSITIONS",
                "CALL_SHAPES",
                "CONTINUATION_VECTORS",
                "STATE_COUNTS",
                "POST_RUN_COMPARISON_BY_SEPARATE_VALIDATOR",
            ],
            "contract_sha256": contract_sha256,
            "ended_ns": time.monotonic_ns(),
            "hello": asdict(hello),
            "inputs_sha256": contract["inputs"]["sha256"],
            "paths": paths,
            "probe": {
                "host_boot_id": Path(
                    "/proc/sys/kernel/random/boot_id"
                ).read_text(encoding="ascii").strip(),
                "pid": os.getpid(),
            },
            "run_name": args.run_name,
            "schema": rp.RAW_SCHEMA,
            "scope": "CUDA_ONLY_DIAGNOSTIC",
            "started_ns": started_ns,
            "state_counts": {
                "after_all_paths": final_status.active_sequences,
                "before_all_paths": initial_status.active_sequences,
            },
            "status": "RAW_CAPTURE_COMPLETE_NO_EQUALITY_EVALUATED",
            "top2_logit_margins": {
                "captured": False,
                "reason": (
                    "StageNet V3 terminal responses expose selected token IDs "
                    "but not logits"
                ),
            },
        }
        rp.write_atomic(args.output, report)
        rp.w5.finish(recording, "stop")
        finished = True
        return 0
    except (OSError, ProtocolError, rp.DiagnosticError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if recording is not None and not finished:
            for path_spec in (
                matches[0]["paths"] if "matches" in locals() and matches else []
            ):
                try:
                    status = recording.status()
                    if status.active_sequences:
                        rp.w5.remove_group(
                            recording,
                            contract["execution"]["batch"],
                            path_spec["identity_base"],
                        )
                except BaseException:
                    pass
            try:
                recording.stop()
            except BaseException:
                pass
        if raw_client is not None:
            raw_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
