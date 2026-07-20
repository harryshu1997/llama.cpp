#!/usr/bin/env python3
"""Live one-GPU priority screen with an independent OP15 head island."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import bge_server_prof as BGE
import cp_d_priority as CPD
import cp_e_live_priority as CPE
import pipe3_device as P3
import stage_a_gpu_board as A
import stageb_headcert as SB
from live_profile_adapter import load_op15_head_route


HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
SELECTED_GPU_UUID = CPD.SELECTED_GPU_UUID
K = 8
REMOTE_DIR = CPE.REMOTE_DIR
REMOTE_BIN = CPE.REMOTE_BIN


def capture_artifacts(profile_paths: list[Path]) -> dict[str, Any]:
    host_paths = {
        "cp_f_live_op15.py": Path(__file__),
        "cp_e_live_priority.py": Path(CPE.__file__),
        "live_profile_adapter.py": SPIKE / "live_profile_adapter.py",
        "priority_batch_runtime.py": SPIKE / "priority_batch_runtime.py",
        "power_frontier_policy.py": SPIKE / "power_frontier_policy.py",
        "stageb_headcert.py": Path(SB.__file__),
        "bge_server_prof.py": Path(BGE.__file__),
        "stage_a_gpu_board.py": Path(A.__file__),
        "llama_layersplit_host": Path(SB.HOST_BIN),
        "llama_embedding_host": Path(BGE.CUDA_BIN),
        "llama_layersplit_android": CPE.ANDROID_LAYERSPLIT,
        "bge_server_profile": CPD.BGE_PROFILE,
    }
    host = {
        name: {"path": str(path), "sha256": CPD.sha256_file(path)}
        for name, path in host_paths.items()
    }
    profiles = {
        path.name: {"path": str(path), "sha256": CPD.sha256_file(path)}
        for path in profile_paths
    }
    phone_binary = CPE.remote_sha256(P3.OP15_SERIAL, f"{REMOTE_DIR}/{REMOTE_BIN}")
    if phone_binary != host["llama_layersplit_android"]["sha256"]:
        raise CPE.LiveGateError("OP15 binary differs from the current Android build")
    return {
        "host": host,
        "op15_profile_processes": profiles,
        "phone_binary": phone_binary,
        "phone_shard": CPE.remote_sha256(P3.OP15_SERIAL, P3.HEAD_SHARD.format(k1=K)),
    }


def low_command(label: str, requests: int, n_gen: int, prompt: str, port: int) -> CPE.ReadyProcess:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU_UUID
    env["LD_LIBRARY_PATH"] = str(Path(SB.HOST_BIN).parent) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["LAYERSPLIT_PLACEMENT_CERT"] = "1"
    env.pop("LLAMA_LAYER_END", None)
    command = [SB.HOST_BIN, "-m", SB.FULL_MODEL, "-ngl", "99"]
    if label == "P0":
        env.pop("LLAMA_LAYER_START", None)
        command += ["--mode", "monodriver"]
    elif label == "P2":
        env["LLAMA_LAYER_START"] = str(K)
        command += ["--mode", "pipedriver", "--host", "127.0.0.1", "--port", str(port)]
    else:
        raise CPE.LiveGateError(f"unsupported label {label}")
    command += [
        "-p", prompt, "-n", str(n_gen), "--driver-requests", str(requests),
        "--driver-warmup", "1", "--driver-batch", "1", "--driver-context", "512",
        "--driver-max-prefill", "64", "--wait-for-go",
    ]
    return CPE.ReadyProcess(command, env, "DRIVER_READY ", "DRIVER_DONE ")


def parse_low(
    label: str,
    process: CPE.ReadyProcess,
    requests: int,
    reference: list[int],
    phone_cert: dict | None,
) -> dict[str, Any]:
    rows = SB.parse_routejson(process.stderr_text())
    if process.done_time_s is None or len(rows) != requests:
        raise CPE.LiveGateError("incomplete Gemma result set")
    if any(row.get("batch_size") != 1 or row.get("token_ids") != reference
           or row.get("generated_tokens") != len(reference) for row in rows):
        raise CPE.LiveGateError("Gemma batch or all-request correctness failed")
    host_cert = SB.parse_placement_cert(process.stderr_text())
    CPE.validate_host_cert(host_cert, 0 if label == "P0" else K, SB.N_LAYER)
    if label == "P2":
        reasons = P3.validate_stage_cert(phone_cert, 0, K)
        if reasons:
            raise CPE.LiveGateError(f"OP15 placement failed: {reasons}")
    wall = [float(row["request_wall_us"]) for row in rows]
    return {
        "batch": 1,
        "requests": requests,
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "lat_us_p50": CPE.percentile(wall, 0.50),
        "lat_us_p95": CPE.percentile(wall, 0.95),
        "lat_us_p99": CPE.percentile(wall, 0.99),
        "done_time_s": process.done_time_s,
        "all_tokens_match": True,
        "phone_live": label == "P2",
        "op15_cert": phone_cert,
        "op12_cert": None,
        "host_cert": host_cert,
    }


def run_cohort(
    label: str,
    repeat: int,
    requests: int,
    n_gen: int,
    prompt: str,
    reference: list[int],
    bge_batch: int,
    bge_words: int,
    bge_exact: int,
    bge_reps: int,
    ready_timeout_s: int,
    execution_timeout_s: int,
) -> dict[str, Any]:
    log_dir = HERE / "logs_live_op15" / f"{label.lower()}_r{repeat}"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("high.stdout", "high.stderr", "low.stdout", "low.stderr"):
        (log_dir / name).unlink(missing_ok=True)
    port = 15920 + repeat * 2 + (0 if label == "P0" else 1)
    stage = handle = high = low = None
    thermal_start = thermal_end = None
    sampler = A.Sampler(CPD.GPU_INDEX)
    try:
        if label == "P2":
            thermal_start = CPE.wait_for_phone_thermal(P3.OP15_SERIAL, ready_timeout_s)
            stage, handle = P3.start_stage(
                P3.OP15_SERIAL, P3.HEAD_SHARD.format(k1=K), 0, K, port, 1,
                n_gen, 512, 64, SB.mbuf_for_k(K), log_dir / "op15.log",
                REMOTE_DIR, REMOTE_BIN, ready_timeout_s,
            )
        low = low_command(label, requests, n_gen, prompt, port)
        low.start()
        low.wait_ready(ready_timeout_s)
        high = CPE.bge_command(bge_batch, bge_words, bge_exact, bge_reps)
        high.start()
        high.wait_ready(ready_timeout_s)
        if not CPD.second_gpu_idle():
            raise CPE.LiveGateError("second GPU active before GO")
        sampler.start()
        time.sleep(1.0)
        go_time_s = time.time()
        high.go()
        low.go()
        high_rc = high.wait(execution_timeout_s)
        low_rc = low.wait(execution_timeout_s)
        if high_rc != 0 or low_rc != 0:
            raise CPE.LiveGateError(f"process failure high={high_rc} low={low_rc}")
        time.sleep(0.4)
        bge = CPE.parse_bge(high, bge_batch, bge_exact, bge_reps)
        cohort_end_s = max(float(bge["paid_end_s"]), float(low.done_time_s or 0.0))
        energy = CPD.persisted_energy_window(sampler, go_time_s, cohort_end_s)
    finally:
        sampler.stop()
        if high is not None:
            high.terminate()
        if low is not None:
            low.terminate()
        CPE.persist_process_logs(log_dir, high, low)
        if stage is not None:
            P3.stop_stage(P3.OP15_SERIAL, stage, handle, port, REMOTE_BIN)

    phone_cert = None
    if label == "P2":
        thermal_end = SB.thermal_snapshot(P3.OP15_SERIAL)
        CPE.validate_end_thermal(thermal_end, P3.OP15_SERIAL)
        phone_cert = SB.parse_placement_cert(
            (log_dir / "op15.log").read_text(encoding="utf-8", errors="replace")
        )
    gemma = parse_low(label, low, requests, reference, phone_cert)
    overlap_start = max(go_time_s, float(bge["paid_start_s"]))
    overlap_end = min(float(bge["paid_end_s"]), float(gemma["done_time_s"]))
    overlap_s = max(0.0, overlap_end - overlap_start)
    shorter = min(
        float(bge["paid_end_s"]) - float(bge["paid_start_s"]),
        float(gemma["done_time_s"]) - go_time_s,
    )
    return {
        "label": label,
        "repeat": repeat,
        "go_time_s": go_time_s,
        "second_gpu_idle_before_go": True,
        "bge": bge,
        "gemma": gemma,
        "concurrent_overlap_s": overlap_s,
        "concurrent_overlap_fraction_of_shorter": overlap_s / shorter if shorter > 0 else 0.0,
        "phone_thermal_start": {"op15": thermal_start} if thermal_start else None,
        "phone_thermal_end": {"op15": thermal_end} if thermal_end else None,
        "selected_gpu_cohort": energy,
    }


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--n-gen", type=int, default=8)
    parser.add_argument("--bge-batch", type=int, default=16)
    parser.add_argument("--bge-words", type=int, default=25)
    parser.add_argument("--bge-window-s", type=float, default=12.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--ready-timeout", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--prompt", default="Explain why batching improves accelerator utilization.")
    parser.add_argument("--output", default=str(HERE / "cp_f_live_op15_result.json"))
    args = parser.parse_args()
    if args.repeats != 3 or min(args.requests, args.n_gen, args.bge_batch,
                                args.ready_timeout, args.timeout) <= 0 \
            or not math.isfinite(args.bge_window_s) or args.bge_window_s <= 0:
        raise CPE.LiveGateError("invalid fixed experiment shape")
    bge_points, knee, exact, atlas_digest = CPD.load_bge_profile(CPD.BGE_PROFILE, 32)
    if knee != args.bge_batch:
        raise CPE.LiveGateError("BGE batch differs from the measured knee")
    p50 = next(point.duration_us for point in bge_points if point.batch_size == knee)
    bge_reps = math.ceil(args.bge_window_s * 1_000_000 / p50)
    profile_paths = [HERE / f"stageb_op15_k8_b1_r{repeat}.json" for repeat in range(7)]
    route_profile = load_op15_head_route(profile_paths, 1)
    artifacts = capture_artifacts(profile_paths)
    reference = SB.run_mono_reference(
        SELECTED_GPU_UUID, args.prompt, args.n_gen, args.timeout,
        batch=1, ctx=512, max_prefill=64,
    )
    rows = []
    failed_label = failed_repeat = None
    try:
        for repeat in range(3):
            order = ("P0", "P2") if repeat % 2 == 0 else ("P2", "P0")
            for label in order:
                failed_label, failed_repeat = label, repeat
                rows.append(run_cohort(
                    label, repeat, args.requests, args.n_gen, args.prompt, reference,
                    args.bge_batch, args.bge_words, exact, bge_reps,
                    args.ready_timeout, args.timeout,
                ))
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": "s14-cp-f-live-op15-v1",
            "status": "LIVE_OP15_FAIL_MEASUREMENT",
            "error": str(exc),
            "failed_label": failed_label,
            "failed_repeat": failed_repeat,
            "completed_rows": rows,
            "artifacts": artifacts,
        }
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2
    summary = CPE.summarize(rows, 1.05, 2.0, 0.1, 0.90)
    passed = summary["high_priority_gate"] and summary["low_priority_gate"] \
        and summary["overlap_gate"] and summary["server_board_relief_gate"]
    result = {
        "schema": "s14-cp-f-live-op15-v1",
        "status": "LIVE_OP15_PASS" if passed else "LIVE_OP15_FAIL_GATE",
        "scope": "one selected A6000 GPU_BOARD plus live OP15 mechanics; phone/USB/total energy unknown",
        "selected_gpu_uuid": SELECTED_GPU_UUID,
        "second_gpu_uuid": CPD.SECOND_GPU_UUID,
        "second_gpu_idle_at_endpoints": CPD.second_gpu_idle(),
        "priority_provenance": "synthetic",
        "slo_provenance": "synthetic relative p95 gates frozen at 1.05/2.0",
        "profile": {
            "bge": atlas_digest,
            "op15": route_profile.profile_id,
            "op15_batch": route_profile.points[0].batch_size,
            "op15_predicted_duration_us": route_profile.points[0].duration_us,
        },
        "artifacts": artifacts,
        "route": {"op15": [0, 8], "server": [8, 48]},
        "workload": {
            "bge_batch": args.bge_batch,
            "bge_reps": bge_reps,
            "bge_window_s_requested": args.bge_window_s,
            "gemma_batch": 1,
            "gemma_requests": args.requests,
            "gemma_n_gen": args.n_gen,
            "prompt": args.prompt,
        },
        "reference_tokens": reference,
        "summary": summary,
        "rows": rows,
        "limits": [
            "Only selected-GPU board energy is measured.",
            "Phone, USB, host-wall, and total-system energy remain unknown.",
            "B1 is the only exact certified OP15 batch; batched phone decode remains unproven.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], **summary}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
