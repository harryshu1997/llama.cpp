#!/usr/bin/env python3
"""Certify exact token correctness for the live three-device route by batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cp_d_priority as CPD
import cp_e_live_priority as CPE
import pipe3_device as P3


HERE = Path(__file__).resolve().parent


class BatchCertError(RuntimeError):
    pass


def thermal_pair(limit_s: int) -> dict[str, Any]:
    return {
        "op15": CPE.wait_for_phone_thermal(P3.OP15_SERIAL, limit_s),
        "op12": CPE.wait_for_phone_thermal(P3.OP12_SERIAL, limit_s),
    }


def thermal_end() -> dict[str, Any]:
    result = {
        "op15": CPE.CPB.thermal_snapshot(P3.OP15_SERIAL),
        "op12": CPE.CPB.thermal_snapshot(P3.OP12_SERIAL),
    }
    CPE.validate_end_thermal(result["op15"], P3.OP15_SERIAL)
    CPE.validate_end_thermal(result["op12"], P3.OP12_SERIAL)
    return result


def run_batch(
    batch: int,
    index: int,
    n_gen: int,
    prompt: str,
    timeout_s: int,
    output_dir: Path,
) -> dict[str, Any]:
    before = thermal_pair(timeout_s)
    port_a = 16500 + index * 2
    port_b = port_a + 1
    result_path = output_dir / f"pipe3_b{batch}.json"
    command = [
        sys.executable,
        str(Path(P3.__file__)),
        "--gpu-uuid", CPE.SELECTED_GPU_UUID,
        "--k1", str(CPE.K1),
        "--k2", str(CPE.K2),
        "--portA", str(port_a),
        "--portB", str(port_b),
        "--batch", str(batch),
        "--n-gen", str(n_gen),
        "--requests", str(batch),
        "--warmups", "1",
        "--driver-context", "512",
        "--driver-max-prefill", "64",
        "--prompt", prompt,
        "--remote-dir", CPE.REMOTE_DIR,
        "--remote-bin", CPE.REMOTE_BIN,
        "--timeout", str(timeout_s),
        "--output", str(result_path),
    ]
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s * 2)
    (output_dir / f"pipe3_b{batch}.stdout").write_text(process.stdout, encoding="utf-8")
    (output_dir / f"pipe3_b{batch}.stderr").write_text(process.stderr, encoding="utf-8")
    after = thermal_end()
    result = None
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result = None
    certified = process.returncode == 0 and type(result) is dict \
        and result.get("certified") is True \
        and result.get("status") == "CERTIFIED_3DEVICE_TOKEN_CORRECT_MECHANICS"
    return {
        "batch": batch,
        "returncode": process.returncode,
        "certified": certified,
        "result_path": str(result_path),
        "result_sha256": CPD.sha256_file(result_path) if result_path.exists() else None,
        "result_status": result.get("status") if type(result) is dict else None,
        "checks": result.get("checks") if type(result) is dict else None,
        "request_wall_us_p50": result.get("request_wall_us_p50") if type(result) is dict else None,
        "thermal_start": before,
        "thermal_end": after,
    }


def write_result(path: Path, artifacts: dict[str, Any], rows: list[dict[str, Any]], status: str) -> None:
    certified = [row["batch"] for row in rows if row["certified"]]
    result = {
        "schema": "s14-cp-e-batch-cert-v1",
        "status": status,
        "scope": "three-device token correctness and placement only; no latency, throughput, or energy claim",
        "route": {"op15": [0, CPE.K1], "op12": [CPE.K1, CPE.K2], "server": [CPE.K2, P3.N_LAYER]},
        "batches_requested": [1, 4, 8, 16, 32, 64],
        "certified_batches": certified,
        "largest_certified_batch": max(certified) if certified else None,
        "artifacts": artifacts,
        "rows": rows,
        "limits": [
            "A nonpassing batch is ineligible for the live priority-energy gate.",
            "Exact token matching is stricter than semantic coherence and is not an activation-error metric.",
            "Phone and total-system energy are unknown.",
        ],
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,8,16,32,64")
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--prompt", default="Explain batching.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=HERE / "logs_batch_cert")
    parser.add_argument("--output", type=Path, default=HERE / "cp_e_batch_cert_result.json")
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if batches != [1, 4, 8, 16, 32, 64] or args.n_gen <= 0 or args.timeout <= 0:
        raise BatchCertError("the v1 screen requires batches 1,4,8,16,32,64 and positive limits")
    if not CPD.second_gpu_idle():
        raise BatchCertError("second GPU is active before the batch screen")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = CPE.capture_artifacts()
    artifacts["host"]["cp_e_batch_cert.py"] = {
        "path": str(Path(__file__)),
        "sha256": CPD.sha256_file(Path(__file__)),
    }
    rows = []
    write_result(args.output, artifacts, rows, "IN_PROGRESS")
    for index, batch in enumerate(batches):
        rows.append(run_batch(batch, index, args.n_gen, args.prompt, args.timeout, args.output_dir))
        write_result(args.output, artifacts, rows, "IN_PROGRESS")
        time.sleep(1)
    if not CPD.second_gpu_idle():
        raise BatchCertError("second GPU is active after the batch screen")
    write_result(args.output, artifacts, rows, "BATCH_CERT_SCREEN_COMPLETE")
    print(json.dumps({
        "status": "BATCH_CERT_SCREEN_COMPLETE",
        "certified_batches": [row["batch"] for row in rows if row["certified"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
