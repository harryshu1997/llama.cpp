#!/usr/bin/env python3
"""S14 CP-D selected-A6000 P0/P2 compute-window energy diagnostic.

Each plan performs the same high-priority BGE batch first, followed by the same
number of low-priority Gemma decode tokens. P0 runs Gemma [0,48); P2 runs only
[8,48), importing the separately certified phone-head feasibility. Phone work
is not live in this diagnostic, and phone/USB/total energy remain unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SPIKE))

import bge_server_prof as BGE  # noqa: E402
import stage_a_gpu_board as A  # noqa: E402
from power_frontier_policy import BatchPoint, WorkItem, choose_batch, throughput_knee  # noqa: E402

GPU_INDEX = 0
SELECTED_GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
SECOND_GPU_UUID = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"
HEAD_K = 8
RESULT_SCHEMA = "s14-cp-d-priority-v3"
RESULT_STATUS = "GPU_BOARD_COMPUTE_WINDOW_P0_P2_DIAGNOSTIC_PASS"
BGE_PROFILE = HERE / "bge_server_result_repaired.json"


class CPDError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    if size == 0:
        raise CPDError(f"empty artifact: {path}")
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CPDError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=no_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(CPDError(f"invalid JSON constant {value!r}")),
    )
    if type(value) is not dict:
        raise CPDError(f"expected JSON object: {path}")
    return value


def second_gpu_idle() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise CPDError("failed to query second-GPU activity")
    return SECOND_GPU_UUID not in result.stdout.splitlines()


def load_bge_profile(path: Path, seq_target: int) -> tuple[list[BatchPoint], int, int, str]:
    atlas = load_json(path)
    if atlas.get("n_failures") != 0 or atlas.get("failures") != []:
        raise CPDError("BGE server atlas contains failures")
    if atlas.get("selected_gpu_uuid") != SELECTED_GPU_UUID:
        raise CPDError("BGE atlas selected-GPU mismatch")
    if atlas.get("second_gpu_idle_at_endpoints") is not True:
        raise CPDError("BGE atlas did not exclude the second GPU at acquisition endpoints")
    expected_artifacts = {
        "bge_server_prof.py": sha256_file(Path(BGE.__file__)).removeprefix("sha256:"),
        "embedding.cpp": sha256_file(HERE.parents[3] / "examples/embedding/embedding.cpp").removeprefix("sha256:"),
        "llama_embedding_cuda": sha256_file(Path(BGE.CUDA_BIN)).removeprefix("sha256:"),
        "llama_embedding_cpu": sha256_file(Path(BGE.CPU_BIN)).removeprefix("sha256:"),
    }
    if atlas.get("artifacts") != expected_artifacts:
        raise CPDError("BGE atlas artifact bundle differs from the current acquisition stack")
    model_digest = sha256_file(Path(BGE.GGUF)).removeprefix("sha256:")
    if atlas.get("model_sha256") != model_digest:
        raise CPDError("BGE atlas model digest differs from the current model")

    rows = [row for row in atlas.get("shapes", []) if row.get("seq_len_target") == seq_target]
    if not rows:
        raise CPDError(f"BGE atlas has no sequence target {seq_target}")
    points: list[BatchPoint] = []
    exact_lengths = set()
    for row in rows:
        batch = row.get("batch")
        p50 = row.get("lat_us_p50")
        if type(batch) is not int or isinstance(p50, bool) or not isinstance(p50, (int, float)):
            raise CPDError("BGE atlas row has invalid batch or latency")
        if not math.isfinite(float(p50)) or p50 <= 0:
            raise CPDError("BGE atlas row has non-positive latency")
        procs = atlas.get("bench", {}).get("procs")
        reps = atlas.get("bench", {}).get("reps_per_proc")
        if type(procs) is not int or procs < 7 or type(reps) is not int or reps <= 0 \
                or row.get("ok_procs") != procs or row.get("n_samples") != procs * reps:
            raise CPDError("BGE atlas row lacks the frozen process/sample coverage")
        if row.get("cov", 1.0) > 0.05 or row.get("placement_status") != "SCHEDULED_PLACEMENT_OK" \
                or row.get("placement_non_cuda_ops"):
            raise CPDError("BGE atlas variability or placement gate failed")
        by_buffer = row.get("placement_compute_by_buffer")
        if not isinstance(by_buffer, dict) or not by_buffer \
                or any("CUDA" not in name for name in by_buffer):
            raise CPDError("BGE atlas compute escaped CUDA")
        exact = row.get("seq_len_exact")
        if type(exact) is not int or exact <= 0:
            raise CPDError("BGE atlas row has invalid exact sequence length")
        exact_lengths.add(exact)
        points.append(BatchPoint(batch, int(round(float(p50)))))
    if len(exact_lengths) != 1:
        raise CPDError("BGE atlas sequence length changes across batches")
    measured_knee = atlas.get("throughput_knee_measured", {}).get(str(seq_target), {}).get("knee_batch_95pct")
    recomputed_knee = throughput_knee(points, 95, 100)
    if measured_knee != recomputed_knee:
        raise CPDError("BGE measured knee does not reproduce")
    cosine = atlas.get("correctness_cuda_vs_cpu", {}).get(str(seq_target), {}).get("cuda_vs_cpu_cosine")
    gate = atlas.get("cosine_gate")
    if not isinstance(cosine, (int, float)) or not isinstance(gate, (int, float)) or cosine < gate:
        raise CPDError("BGE correctness gate failed")
    return sorted(points, key=lambda point: point.batch_size), recomputed_knee, exact_lengths.pop(), sha256_file(path)


def load_gemma_profile(path: Path, max_batch: int) -> tuple[list[BatchPoint], str]:
    artifact = load_json(path)
    device = artifact.get("device", {})
    if artifact.get("head") != "[0,8)" or artifact.get("flash_attn") != "on":
        raise CPDError("Gemma batch profile has the wrong route")
    if device.get("serial") != "3C15AU002CL00000" or device.get("backend") != "HTP0":
        raise CPDError("Gemma batch profile has the wrong device/backend")
    resolution = artifact.get("correctness_resolution", {})
    if "CORRECT" not in str(resolution.get("verdict", "")):
        raise CPDError("Gemma batch correctness is not certified")

    points = []
    for row in artifact.get("measured", []):
        batch = row.get("batch")
        duration_ms = row.get("forward_ms")
        if type(batch) is not int or batch > max_batch or batch == 2:
            continue
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
            raise CPDError("Gemma batch row has invalid duration")
        points.append(BatchPoint(batch, int(round(float(duration_ms) * 1000))))
    points.sort(key=lambda point: point.batch_size)
    if not points or points[0].batch_size != 1 or points[-1].batch_size != max_batch:
        raise CPDError("Gemma profile does not cover batch 1 through the requested maximum")
    return points, sha256_file(path)


def policy_decisions(
    bge_points: list[BatchPoint],
    gemma_points: list[BatchPoint],
    now_us: int = 0,
) -> dict[str, Any]:
    bge_ready = max(point.batch_size for point in bge_points)
    gemma_ready = max(point.batch_size for point in gemma_points)
    hi = [
        WorkItem(
            f"bge-{index}", "encoder", "bge-small-en-v1.5-f16", "bge_encoder_0_12",
            "bge|cls|l32", now_us, now_us + 10_000_000, 0,
        )
        for index in range(bge_ready)
    ]
    lo = [
        WorkItem(
            f"gemma-{index}", "generation", "gemma-4-12b-it-f16", "gemma_head_0_8",
            "gemma|decode|c512", now_us, now_us + 10_000_000, 1,
        )
        for index in range(gemma_ready)
    ]
    high = choose_batch(now_us, hi, bge_points, "compute_bound", high_priority_max=0)
    low = choose_batch(now_us, lo, gemma_points, "memory_bound", high_priority_max=0)
    if high.action != "LAUNCH" or low.action != "LAUNCH":
        raise CPDError("measured profiles did not produce launch decisions")
    isolation_ok = False
    try:
        choose_batch(
            now_us,
            [hi[0], WorkItem("collision", "generation", "gemma-4-12b-it-f16", "gemma_head_0_8",
                             hi[0].compatibility_key, now_us, now_us + 10_000_000, 0)],
            bge_points,
            "compute_bound",
        )
    except Exception:
        isolation_ok = True
    if not isolation_ok:
        raise CPDError("compatibility collision was not rejected")
    return {
        "high": {"action": high.action, "batch": high.batch_size, "reason": high.reason},
        "low": {"action": low.action, "batch": low.batch_size, "reason": low.reason},
        "compatibility_isolation_enforced": True,
    }


def required_reps(p50_us: int, requested: int, min_window_s: float) -> int:
    if type(p50_us) is not int or p50_us <= 0 or type(requested) is not int or requested <= 0 \
            or not math.isfinite(min_window_s) or min_window_s <= 0:
        raise CPDError("invalid BGE repetition/window request")
    return max(requested, math.ceil(min_window_s * 1_000_000 / p50_us))


def parse_placement(stderr: str) -> dict[str, Any]:
    lines = [line for line in stderr.splitlines() if line.startswith("PLACEMENTCERT ")]
    if len(lines) != 1:
        raise CPDError(f"expected one BGE placement certificate, found {len(lines)}")
    cert = json.loads(lines[0][len("PLACEMENTCERT "):])
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" or cert.get("missing_buffer_compute_nodes") != 0:
        raise CPDError("BGE placement certificate failed")
    by_buffer = cert.get("by_buffer", {})
    if sum(value for key, value in by_buffer.items() if "CUDA" in key) <= 0:
        raise CPDError("BGE placement has no CUDA compute")
    non_cuda = cert.get("non_htp_ops", [])
    if any(not value.startswith("GET_ROWS@CPU:") for value in non_cuda):
        raise CPDError(f"BGE placement contains undeclared CPU work: {non_cuda}")
    return cert


def persisted_energy_window(sampler: A.Sampler, start_s: float, end_s: float) -> dict[str, Any]:
    start_us = int(round(start_s * 1_000_000))
    end_us = int(round(end_s * 1_000_000))
    if end_us <= start_us:
        raise CPDError("invalid paid power window")
    samples = [
        {
            "t_us": int(round(t * 1_000_000)),
            "power_mw": int(round(power * 1000)),
            "util_milli_pct": int(round(util * 1000)),
            "pstate": pstate,
        }
        for t, power, util, pstate in list(sampler.samples)
        if start_s - 1.0 <= t <= end_s + 1.0
    ]
    samples.sort(key=lambda row: row["t_us"])
    if len(samples) < 2 or samples[0]["t_us"] > start_us or samples[-1]["t_us"] < end_us:
        raise CPDError("power samples do not bracket the paid window")
    in_window = [row for row in samples if start_us <= row["t_us"] <= end_us]
    if len(in_window) < 5:
        raise CPDError("too few paid-window power samples")
    energy_nj = 0
    max_gap_us = 0
    for left, right in zip(samples, samples[1:]):
        lo = max(left["t_us"], start_us)
        hi = min(right["t_us"], end_us)
        if hi > lo:
            energy_nj += left["power_mw"] * (hi - lo)
            max_gap_us = max(max_gap_us, right["t_us"] - left["t_us"])
    if energy_nj <= 0 or max_gap_us > 250_000:
        raise CPDError("power integration or sample-gap gate failed")
    duration_us = end_us - start_us
    return {
        "energy_nj": energy_nj,
        "energy_j": energy_nj / 1_000_000_000,
        "duration_us": duration_us,
        "duration_s": duration_us / 1_000_000,
        "avg_power_w": energy_nj / duration_us / 1000,
        "min_power_w": min(row["power_mw"] for row in in_window) / 1000,
        "max_power_w": max(row["power_mw"] for row in in_window) / 1000,
        "n_samples": len(in_window),
        "max_sample_gap_us": max_gap_us,
        "max_sample_gap_s": max_gap_us / 1_000_000,
        "avg_util_pct": (
            sum(row["util_milli_pct"] for row in in_window) / len(in_window) / 1000
        ),
        "paid_window_us": {"start": start_us, "end": end_us},
        "raw_power_samples": samples,
    }


def measure_bge(
    sampler: A.Sampler,
    batch: int,
    exact_tokens: int,
    words: int,
    reps: int,
) -> dict[str, Any]:
    corpus = HERE / "logs_bge_server" / f"cpd_bge_words{words}_b{batch}.txt"
    BGE.write_corpus(corpus, BGE.make_prompt(words), batch)
    ctx = BGE.ctx_for(batch, exact_tokens)
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": SELECTED_GPU_UUID,
            "LD_LIBRARY_PATH": BGE.CUDA_LIB,
            "BGEPROF_REPS": str(reps),
            "BGEPROF_WARMUP": "5",
        }
    )
    command = [
        BGE.CUDA_BIN, "-m", BGE.GGUF, "-ngl", "99", "--pooling", "cls",
        "--embd-normalize", "2", "-fa", "off", "-f", str(corpus),
        "--parallel", str(max(batch, 2)), "-c", str(ctx), "-b", str(ctx),
        "--embd-output-format", "array",
    ]
    process = subprocess.run(command, capture_output=True, text=True, env=env, timeout=900)
    lines = [line for line in process.stdout.splitlines() if line.startswith("BGEPROF ")]
    if process.returncode != 0 or len(lines) != 1:
        raise CPDError(f"BGE process failed: rc={process.returncode}, records={len(lines)}")
    parsed = json.loads(lines[0][len("BGEPROF "):])
    if parsed.get("batch") != batch or parsed.get("total_tokens") != batch * exact_tokens:
        raise CPDError("BGE result shape differs from the measured profile")
    if parsed.get("reps") != reps or parsed.get("finite") is not True or len(parsed.get("us", [])) != reps:
        raise CPDError("BGE result failed repetition or finite-output checks")
    paid_start = parsed.get("paid_start_s")
    paid_end = parsed.get("paid_end_s")
    if not isinstance(paid_start, (int, float)) or not isinstance(paid_end, (int, float)) or paid_end <= paid_start:
        raise CPDError("BGE binary did not emit a valid paid window")
    time.sleep(0.3)
    energy = persisted_energy_window(sampler, float(paid_start), float(paid_end))
    placement = parse_placement(process.stderr)
    values = sorted(float(value) for value in parsed["us"])
    return {
        **energy,
        "batch": batch,
        "reps": reps,
        "encodes": batch * reps,
        "lat_us_p50": values[len(values) // 2],
        "placement": placement,
    }


def measure_gemma(sampler: A.Sampler, k: int, batch: int, steps: int) -> dict[str, Any]:
    result = A.run_tailbench(k, batch, steps, GPU_INDEX)
    if result["returncode"] != 0 or result["tokens"] != batch * steps:
        raise CPDError(f"Gemma tailbench k={k} failed equal-work checks")
    time.sleep(0.2)
    energy = persisted_energy_window(sampler, result["bench_start"], result["bench_end"])
    return {
        **result,
        **energy,
        "energy_per_token_mj": 1000.0 * energy["energy_j"] / result["tokens"],
    }


def median(rows: list[dict[str, Any]], key: str) -> float:
    values = sorted(float(row[key]) for row in rows)
    return values[len(values) // 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-target", type=int, default=32)
    parser.add_argument("--seq-words", type=int, default=25)
    parser.add_argument("--gemma-batch", type=int, default=16)
    parser.add_argument("--gemma-steps", type=int, default=400)
    parser.add_argument("--bge-reps", type=int, default=30)
    parser.add_argument("--bge-min-window-s", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default=str(HERE / "cp_d_result.json"))
    args = parser.parse_args()

    output = Path(args.output)
    atlas_path = BGE_PROFILE
    gemma_path = HERE / "stage_d_batch_scaling.json"
    try:
        bge_points, bge_knee, exact_tokens, atlas_digest = load_bge_profile(atlas_path, args.seq_target)
        gemma_points, gemma_digest = load_gemma_profile(gemma_path, args.gemma_batch)
        decisions = policy_decisions(bge_points, gemma_points)
        if decisions["high"]["batch"] != bge_knee or decisions["low"]["batch"] != args.gemma_batch:
            raise CPDError("policy decisions differ from the measured knee/max-batch contract")
        p50_us = next(point.duration_us for point in bge_points if point.batch_size == bge_knee)
        reps = required_reps(p50_us, args.bge_reps, args.bge_min_window_s)
    except Exception as exc:  # noqa: BLE001
        output.write_text(json.dumps({"schema": RESULT_SCHEMA, "status": "FAIL_PROFILE",
                                      "error": str(exc)}, indent=2), encoding="utf-8")
        print(f"[CP-D FAIL_PROFILE] {exc}", file=sys.stderr)
        return 2

    sampler = A.Sampler(GPU_INDEX)
    plan_rows: list[dict[str, Any]] = []
    try:
        sampler.start()
        time.sleep(1.0)
        for repeat in range(args.repeats):
            order = ("P0", "P2") if repeat % 2 == 0 else ("P2", "P0")
            for label in order:
                if not second_gpu_idle():
                    raise CPDError("second GPU became active during the cohort")
                bge = measure_bge(sampler, bge_knee, exact_tokens, args.seq_words, reps)
                gemma = measure_gemma(sampler, 0 if label == "P0" else HEAD_K,
                                       args.gemma_batch, args.gemma_steps)
                plan_rows.append(
                    {
                        "label": label,
                        "repeat": repeat,
                        "execution_order": ["high_priority_bge", "low_priority_gemma"],
                        "bge": bge,
                        "gemma": gemma,
                        "selected_gpu_energy_j": bge["energy_j"] + gemma["energy_j"],
                        "selected_gpu_active_s": bge["duration_s"] + gemma["duration_s"],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        result = {
            "schema": RESULT_SCHEMA,
            "status": "FAIL_MEASUREMENT",
            "error": str(exc),
            "completed_plan_rows": plan_rows,
        }
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[CP-D FAIL_MEASUREMENT] {exc}", file=sys.stderr)
        return 2
    finally:
        sampler.stop()

    if not second_gpu_idle():
        result = {"schema": RESULT_SCHEMA, "status": "FAIL_SECOND_GPU_ACTIVE"}
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 2

    p0 = [row for row in plan_rows if row["label"] == "P0"]
    p2 = [row for row in plan_rows if row["label"] == "P2"]
    p0_energy = median(p0, "selected_gpu_energy_j")
    p2_energy = median(p2, "selected_gpu_energy_j")
    saving = 1.0 - p2_energy / p0_energy
    result = {
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "scope": (
            "selected-A6000 paid compute windows only; phone head feasibility imported, not live; "
            "phone/USB/relay/idle/total-wall energy unknown"
        ),
        "selected_gpu_uuid": SELECTED_GPU_UUID,
        "second_gpu_uuid": SECOND_GPU_UUID,
        "second_gpu_idle": True,
        "priority_provenance": "synthetic",
        "slo_provenance": "synthetic",
        "policy_decisions": decisions,
        "profiles": {
            "bge_server_atlas": atlas_digest,
            "gemma_phone_batch": gemma_digest,
            "bge_seq_target": args.seq_target,
            "bge_seq_exact": exact_tokens,
            "bge_batch": bge_knee,
            "gemma_batch": args.gemma_batch,
            "excluded_gemma_batch": 2,
            "excluded_reason": "historical artifact contains contradictory B2 correctness prose",
        },
        "artifacts": {
            "cp_d_priority.py": {"path": str(Path(__file__)), "sha256": sha256_file(Path(__file__))},
            "bge_server_prof.py": {"path": str(Path(BGE.__file__)), "sha256": sha256_file(Path(BGE.__file__))},
            "stage_a_gpu_board.py": {"path": str(Path(A.__file__)), "sha256": sha256_file(Path(A.__file__))},
            "llama_embedding": {"path": str(Path(BGE.CUDA_BIN)), "sha256": sha256_file(Path(BGE.CUDA_BIN))},
            "llama_layersplit": {"path": str(Path(A.BIN)), "sha256": sha256_file(Path(A.BIN))},
        },
        "matched_work_per_plan": {
            "bge_encodes": bge_knee * reps,
            "gemma_decode_tokens": args.gemma_batch * args.gemma_steps,
        },
        "measured": {
            "p0_selected_gpu_energy_j_median": p0_energy,
            "p2_selected_gpu_energy_j_median": p2_energy,
            "selected_gpu_energy_saving_frac": saving,
            "p0_gemma_mj_per_token_median": median([row["gemma"] for row in p0], "energy_per_token_mj"),
            "p2_gemma_mj_per_token_median": median([row["gemma"] for row in p2], "energy_per_token_mj"),
            "p0_bge_lat_us_p50_median": median([row["bge"] for row in p0], "lat_us_p50"),
            "p2_bge_lat_us_p50_median": median([row["bge"] for row in p2], "lat_us_p50"),
            "plan_rows": plan_rows,
        },
        "limits": [
            "P2 does not execute the phone head in this GPU-only diagnostic; CP-C supplies that physical route.",
            "The two services execute in strict priority order, not concurrently on the selected GPU.",
            "P1/P3 are unmeasured because no controllable lower A6000 state is available.",
            "This result cannot establish total-system energy saving.",
        ],
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[CP-D] selected-GPU paid-window saving P0->P2: {100.0 * saving:.2f}% "
        f"({p0_energy:.1f} -> {p2_energy:.1f} J)",
        flush=True,
    )
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
