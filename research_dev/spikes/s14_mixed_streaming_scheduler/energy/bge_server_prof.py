#!/usr/bin/env python3
"""S14 Checkpoint A: A6000 BGE encode latency/throughput atlas (server_control).

Repaired measurement. Uses the env-gated BGEPROF bench mode of llama-embedding
(examples/embedding/embedding.cpp): model load, graph reserve, tokenize and warmup
are OUTSIDE the paid interval; only repeated single-batch encodes of B independent
sequences are timed, inside one loaded process. batch = number of independent
sequences (explicit sequence parallelism via --parallel B and enough token capacity).

Fail-closed on: CPU fallback (any op off the selected GPU), wrong/again-visible GPU,
missing BGEPROF line or wrong output count, non-finite embeddings, or CUDA-vs-CPU
cosine below the frozen 0.999 gate.

Reports p50/p95/p99, throughput, CoV, exact token lengths, sample/failure counts.
Roofline item 7: the measured throughput knee (via power_frontier_policy.throughput_knee)
is reported SEPARATELY from an analytic upper bound for the executed BERT graph.
The bound is not a compute/memory classification because it omits activation and
mask traffic. The measured knee is the scheduler input.

Correctness is additionally certified end-to-end (CUDA array vs CPU array cosine).
Phone energy and phone latency are out of scope here (Checkpoint B).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIKE = HERE.parent
sys.path.insert(0, str(SPIKE))
from bge_corpus import make_prompt, write_corpus  # noqa: E402
from power_frontier_policy import BatchPoint, throughput_knee  # noqa: E402

REPO = HERE.parents[3]
GGUF = str(REPO / "models/bge-small-en-v1.5-f16.gguf")
CUDA_BIN = str(REPO / "build-cuda/bin/llama-embedding")
CPU_BIN = str(REPO / "build-cpu/bin/llama-embedding")
CUDA_LIB = str(REPO / "build-cuda/bin")
CPU_LIB = str(REPO / "build-cpu/bin")

# selected A6000 (GPU0). CUDA_VISIBLE_DEVICES is pinned to this UUID so the process
# can only touch this board; the second GPU performs zero experiment work by construction.
SELECTED_GPU_UUID = "GPU-45b611d6-7c5d-9e48-f260-4fbe5f8ef69f"
SECOND_GPU_UUID = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"

# BGE-small-en-v1.5 dims (BERT), for the analytic roofline (embeddings on CPU excluded)
N_LAYER, D, FF, N_HEAD = 12, 384, 1536, 12
WEIGHT_BYTES = 12 * (4 * D * D + 2 * D * FF) * 2  # f16 transformer weights (no embd/vocab)
# A6000 (Ampere GA102): fp16-tensor ~155 TFLOP/s, HBM ~768 GB/s -> ridge ~ 202 FLOP/byte
A6000_PEAK_FLOPS = 155e12
A6000_BW_BYTES = 768e9
A6000_RIDGE = A6000_PEAK_FLOPS / A6000_BW_BYTES

MAX_POS = 512  # bge-small max_position_embeddings; a sequence longer than this is out of range


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bgeprof(binary: str, lib: str, corpus: Path, batch: int, ctx: int, reps: int,
             warmup: int, gpu: bool) -> tuple[dict | None, str, int]:
    # --parallel must be >= 2: n_parallel==1 triggers an auto unified-KV/mem-fit path that asserts.
    env = {"LD_LIBRARY_PATH": lib, "PATH": "/usr/bin:/bin",
           "BGEPROF_REPS": str(reps), "BGEPROF_WARMUP": str(warmup)}
    cmd = [binary, "-m", GGUF, "--pooling", "cls", "--embd-normalize", "2", "-fa", "off",
           "-f", str(corpus), "--parallel", str(max(batch, 2)), "-c", str(ctx), "-b", str(ctx),
           "--embd-output-format", "array"]
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = SELECTED_GPU_UUID
        cmd[1:1] = ["-ngl", "99"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("BGEPROF ")), None)
    parsed = json.loads(line[len("BGEPROF "):]) if line else None
    return parsed, p.stderr, p.returncode


def ctx_for(batch: int, seq_exact: int) -> int:
    return (((max(batch, 2) * seq_exact) + 64 + 31) // 32) * 32


def run_bench(corpus: Path, batch: int, ctx: int, reps: int, warmup: int) -> tuple[dict, str, int]:
    return _bgeprof(CUDA_BIN, CUDA_LIB, corpus, batch, ctx, reps, warmup, gpu=True)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def pctl(xs: list[float], q: float) -> float:
    s = sorted(xs); k = (len(s) - 1) * q; lo = int(k)
    return s[lo] if lo + 1 >= len(s) else s[lo] + (k - lo) * (s[lo + 1] - s[lo])


def second_gpu_idle() -> bool:
    process = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if process.returncode != 0:
        raise RuntimeError("failed to query second-GPU activity")
    return SECOND_GPU_UUID not in process.stdout.splitlines()


def parse_cuda_placement(stderr: str) -> dict:
    lines = [line for line in stderr.splitlines() if line.startswith("PLACEMENTCERT ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected one CUDA placement certificate, found {len(lines)}")
    cert = json.loads(lines[0][len("PLACEMENTCERT "):])
    if cert.get("status") != "SCHEDULED_PLACEMENT_OK" or cert.get("missing_buffer_compute_nodes") != 0:
        raise RuntimeError("CUDA placement status failed")
    by_buffer = cert.get("by_buffer")
    if not isinstance(by_buffer, dict) or not by_buffer \
            or any("CUDA" not in name for name in by_buffer):
        raise RuntimeError("CUDA placement escaped the selected backend")
    if sum(value for value in by_buffer.values() if type(value) is int and value > 0) <= 0:
        raise RuntimeError("CUDA placement contains no compute")
    if cert.get("non_htp_ops"):
        raise RuntimeError(f"CUDA placement contains CPU work: {cert['non_htp_ops']}")
    return cert


def executed_graph_bound(
    seq_exact: dict[int, int],
    batches: list[int],
) -> dict:
    """Analytic FLOP/weight-byte bound for the current FA-off no-cache graph.

    The graph builds attention over all tokens in the physical batch. Its mask
    removes cross-sequence values, but not the dense QK/AV work.
    """
    roofline = {
        "a6000_ridge_flop_per_byte": round(A6000_RIDGE, 1),
        "weight_bytes_f16": WEIGHT_BYTES,
        "geometry": "fa_off_no_cache_attention_over_total_batch_tokens",
        "classification": "UNCLASSIFIED_USE_MEASURED_KNEE",
        "note": (
            "weight-only arithmetic intensity is an optimistic upper bound; it omits "
            "activation, mask, and intermediate traffic and cannot prove compute-bound. "
            "The executed graph performs masked cross-sequence attention work."
        ),
        "points": [],
    }
    dense_macs_per_token = N_LAYER * (4 * D * D + 2 * D * FF)
    for target_len in sorted(seq_exact):
        exact_len = seq_exact[target_len]
        for batch in batches:
            total_tokens = exact_len * batch
            dense_flops = 2 * dense_macs_per_token * total_tokens
            useful_attn_flops = 4 * N_LAYER * D * exact_len * exact_len * batch
            executed_attn_flops = 4 * N_LAYER * D * total_tokens * total_tokens
            executed_flops = dense_flops + executed_attn_flops
            useful_flops = dense_flops + useful_attn_flops
            roofline["points"].append(
                {
                    "seq_len_target": target_len,
                    "seq_len_exact": exact_len,
                    "batch": batch,
                    "total_tokens": total_tokens,
                    "useful_fwd_gflops": round(useful_flops / 1e9, 2),
                    "executed_fwd_gflops": round(executed_flops / 1e9, 2),
                    "masked_excess_gflops": round((executed_flops - useful_flops) / 1e9, 2),
                    "weight_only_ai_upper_bound": round(executed_flops / WEIGHT_BYTES, 1),
                    "regime": "UNCLASSIFIED_USE_MEASURED_KNEE",
                }
            )
    return roofline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="32:25,128:105,512:400", help="L:words pairs (tokens must stay <=512)")
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--procs", type=int, default=7)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--cosine-gate", type=float, default=0.999)
    ap.add_argument("--output", default=str(HERE / "bge_server_result.json"))
    args = ap.parse_args()
    if not second_gpu_idle():
        raise RuntimeError("second GPU is active before CP-A acquisition")

    seqs = [(int(a), int(b)) for a, b in (p.split(":") for p in args.seqs.split(","))]
    batches = [int(x) for x in args.batches.split(",")]
    logdir = HERE / "logs_bge_server"; logdir.mkdir(exist_ok=True)

    shapes, failures = [], []
    correctness = {}
    seq_exact: dict[int, int] = {}
    for L, words in seqs:
        # probe exact tokenized length + correctness gate (CUDA vs CPU) once per L
        c1 = logdir / f"corr_{L}.txt"; write_corpus(c1, make_prompt(words), 1)
        cu_p, cu_stderr, cu_rc = _bgeprof(CUDA_BIN, CUDA_LIB, c1, 1, 1024, 1, 1, gpu=True)
        cp_p, _, cp_rc = _bgeprof(CPU_BIN, CPU_LIB, c1, 1, 1024, 1, 1, gpu=False)
        if cu_p is None or cp_p is None:
            failures.append(f"L={L} correctness run failed (cuda_rc={cu_rc} cpu_rc={cp_rc})")
            correctness[L] = {"status": "RUN_FAILED"}
            continue
        try:
            parse_cuda_placement(cu_stderr)
        except (RuntimeError, json.JSONDecodeError) as exc:
            failures.append(f"L={L} correctness CUDA placement failed: {exc}")
            correctness[L] = {"status": "PLACEMENT_FAILED"}
            continue
        exact = cu_p["total_tokens"]
        seq_exact[L] = exact
        if exact > MAX_POS:
            failures.append(f"L={L}: exact tokens {exact} exceed model max_pos {MAX_POS}")
        cos = cosine(cu_p["emb0"], cp_p["emb0"])
        correctness[L] = {"cuda_vs_cpu_cosine": round(cos, 6), "n_embd": cu_p["n_embd"],
                          "seq_len_exact": exact}
        if cos < args.cosine_gate:
            failures.append(f"L={L} cosine {cos:.6f} < gate {args.cosine_gate}")
        print(f"[correctness] L~{L} (exact {exact} tok): cosine={cos:.6f} (gate {args.cosine_gate})", flush=True)

    for L, words in seqs:
        exact = seq_exact.get(L)
        if exact is None or exact > MAX_POS:
            continue
        for B in batches:
            ctx = ctx_for(B, exact)
            corpus = logdir / f"c_L{L}_B{B}.txt"; write_corpus(corpus, make_prompt(words), B)
            all_us, exact_tokens, ok_procs = [], None, 0
            placement_last = None
            for pi in range(args.procs):
                try:
                    parsed, stderr, rc = run_bench(corpus, B, ctx, args.reps, args.warmup)
                except Exception as e:  # noqa: BLE001
                    failures.append(f"L{L}B{B} proc{pi}: exception {str(e)[:120]}")
                    continue
                if rc != 0 or parsed is None:
                    failures.append(f"L{L}B{B} proc{pi}: rc={rc} parsed={parsed is not None}")
                    continue
                try:
                    placement = parse_cuda_placement(stderr)
                except (RuntimeError, json.JSONDecodeError) as exc:
                    failures.append(f"L{L}B{B} proc{pi}: placement {exc}")
                    continue
                if parsed["batch"] != B or not parsed["finite"]:
                    failures.append(f"L{L}B{B} proc{pi}: batch={parsed['batch']} finite={parsed['finite']}")
                    continue
                all_us.extend(parsed["us"])
                exact_tokens = parsed["total_tokens"] // B
                placement_last = placement
                ok_procs += 1
            if not all_us:
                shapes.append({"seq_len": L, "batch": B, "status": "NO_SAMPLES"})
                continue
            p50 = pctl(all_us, 0.50)
            row = {
                "seq_len_target": L, "seq_len_exact": exact_tokens, "batch": B,
                "ok_procs": ok_procs, "n_samples": len(all_us),
                "lat_us_p50": round(p50, 2), "lat_us_p95": round(pctl(all_us, 0.95), 2),
                "lat_us_p99": round(pctl(all_us, 0.99), 2),
                "lat_us_mean": round(sum(all_us) / len(all_us), 2),
                "cov": round((sum((x - sum(all_us) / len(all_us)) ** 2 for x in all_us) / len(all_us)) ** 0.5
                             / (sum(all_us) / len(all_us)), 4),
                "throughput_enc_s": round(B / (p50 / 1e6), 1),
                "placement_status": placement_last["status"],
                "placement_compute_by_buffer": placement_last["by_buffer"],
                "placement_non_cuda_ops": placement_last["non_htp_ops"],
            }
            shapes.append(row)
            print(f"  L~{L} B={B:>2}: p50={row['lat_us_p50']:.0f}us p99={row['lat_us_p99']:.0f}us "
                  f"CoV={row['cov']:.3f} tput={row['throughput_enc_s']:.0f} enc/s "
                  f"({ok_procs}/{args.procs} procs, {len(all_us)} samples)", flush=True)

    # throughput knee per seq length (policy contract) -- MEASURED, separate from roofline
    knees = {}
    for L, _ in seqs:
        pts = [BatchPoint(r["batch"], int(round(r["lat_us_p50"])))
               for r in shapes if r.get("seq_len_target") == L and "lat_us_p50" in r]
        if len(pts) >= 2 and any(p.batch_size == 1 for p in pts):
            try:
                knees[L] = {"knee_batch_95pct": throughput_knee(pts, 95, 100)}
            except Exception as e:  # noqa: BLE001
                knees[L] = {"error": str(e)[:120]}

    roofline = executed_graph_bound(seq_exact, batches)
    second_gpu_idle_end = second_gpu_idle()
    if not second_gpu_idle_end:
        failures.append("second GPU is active after CP-A acquisition")

    result = {
        "schema": "s14-bge-server-control-v3",
        "checkpoint": "A",
        "scope": "A6000_GPU_BOARD BGE encode latency/throughput; phone latency+energy OUT OF SCOPE",
        "selected_gpu_uuid": SELECTED_GPU_UUID, "second_gpu_uuid": SECOND_GPU_UUID,
        "second_gpu_idle_at_endpoints": second_gpu_idle_end,
        "model": "bge-small-en-v1.5-f16", "model_sha256": "4cd429b83d2805e4028d96f6174b153bc87657808ba9f3b3c4f84d374b481e03",
        "flash_attn": "off (explicit softmax; no CPU fallback)",
        "bench": {"procs": args.procs, "reps_per_proc": args.reps, "warmup": args.warmup,
                  "method": "env-gated BGEPROF loop; load/tokenize/warmup excluded"},
        "correctness_cuda_vs_cpu": correctness, "cosine_gate": args.cosine_gate,
        "artifacts": {
            "bge_server_prof.py": sha256_file(Path(__file__)),
            "embedding.cpp": sha256_file(REPO / "examples/embedding/embedding.cpp"),
            "llama_embedding_cuda": sha256_file(Path(CUDA_BIN)),
            "llama_embedding_cpu": sha256_file(Path(CPU_BIN)),
        },
        "shapes": shapes,
        "throughput_knee_measured": knees,
        "roofline_analytic": roofline,
        "failures": failures, "n_failures": len(failures),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nfailures={len(failures)}  wrote {args.output}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
