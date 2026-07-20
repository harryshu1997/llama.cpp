#!/usr/bin/env python3
"""S14 energy Stage D: measured phone||server concurrency (the overlap the design needs).

Stages A/B/C established: the A6000 saves 15.1% GPU-board decode energy when it runs
tail [8,48) instead of full (A); a phone can actually run the head [0,8) no-fallback,
token-correct (B); folding that onto mix-v1 gives a 13.3% ACCOUNTING ceiling (C). All
of that ASSUMES the phone runs concurrently with a busy server at no cost. Stage D
MEASURES that assumption with two live processes + NVML:

  process S (GPU0):  a saturated A6000 decode backlog = "server busy on other task".
  process P (phone): OP15 runs [0,8) heads continuously, tail on GPU1 (so GPU0's OWN
                     workload is byte-identical between control and treatment -> the
                     only difference is the concurrent USB relay + host CPU + PCIe).

Measures:
  (1) INTERFERENCE: GPU0 board energy + decode throughput, SOLO vs with process P live.
      If ~0, the phone runs alongside the server for free.
  (2) R_phone: the phone's SUSTAINED [0,8) head throughput (tok/s) under concurrency.
  (3) The honest REALISED saving = (offloadable rate the phone can carry) * s(8),
      which the phone throughput BOUNDS -- one phone at single-stream is ~8-10 tok/s
      vs the A6000's ~347 tok/s, so this exposes the throughput mismatch that the
      Stage C accounting ceiling ignores.

Scope: A6000 GPU_BOARD energy + throughput ONLY. Phone energy UNKNOWN. Honest either
way -- this is the measurement that says whether the overlap saving is real or the
phone simply cannot carry enough of the load.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_a_gpu_board as A  # noqa: E402

GPU0_INDEX = 0                                   # server backlog board (measured)
GPU1_UUID = "GPU-431a4567-fa90-7a73-625a-2ee6e7b5eaaf"  # phone-tail board (isolates GPU0 work)
STAGEB = str(HERE / "stageb_headcert.py")
N_GEN = 8


def phone_head_tok_s(result_path: Path, k: int, n_gen: int) -> float:
    """Sustained single-stream [0,k) head rate from the phone pipeline's measured
    per-request latency (stageb consumes ROUTEJSON internally, so read its result)."""
    if not result_path.exists():
        return 0.0
    d = json.loads(result_path.read_text())
    for r in d.get("per_k", []):
        if r.get("k") == k and r.get("placement_status") == "SCHEDULED_PLACEMENT_OK":
            sa = r.get("stage_a_us_p50")
            if sa:
                return n_gen / (sa / 1e6)
    return 0.0


def measure_gpu0(sampler: A.Sampler, batch: int, steps: int, label: str) -> dict[str, Any]:
    """One saturated full-model tailbench on GPU0, bracketed by NVML."""
    r = A.run_tailbench(0, batch, steps, GPU0_INDEX)   # k=0 => full [0,48), the busy backlog
    e = sampler.energy_j(r["bench_start"], r["bench_end"])
    return {
        "label": label,
        "energy_j": e["energy_j"],
        "duration_s": e["duration_s"],
        "avg_power_w": e["avg_power_w"],
        "avg_util_pct": e["avg_util_pct"],
        "tokens": r["tokens"],
        "throughput_tok_s": r["tokens"] / e["duration_s"],
        "energy_per_token_mj": 1000.0 * e["energy_j"] / r["tokens"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default="3C15AU002CL00000")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--phone-requests", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--e-full-mj", type=float, default=859.916)   # Stage A measured
    ap.add_argument("--e-tail-mj", type=float, default=730.022)   # Stage A measured [8,48)
    ap.add_argument("--output", default=str(HERE / "stage_d_result.json"))
    args = ap.parse_args()

    log_dir = HERE / "stage_d_logs"
    log_dir.mkdir(exist_ok=True)
    phone_pipe_log = log_dir / "phone_pipeline.log"

    sampler = A.Sampler(GPU0_INDEX)
    sampler.start()
    time.sleep(2.0)

    # warmup GPU0 clocks
    print("warmup GPU0 ...", flush=True)
    try:
        A.run_tailbench(0, args.batch, min(args.steps, 200), GPU0_INDEX)
    except Exception as e:  # noqa: BLE001
        print(f"warmup failed: {e}", file=sys.stderr)

    # (1) CONTROL: GPU0 saturated backlog, phone idle
    print("control: GPU0 saturated backlog, phone IDLE ...", flush=True)
    control = measure_gpu0(sampler, args.batch, args.steps, "control_phone_idle")
    print(f"  E={control['energy_j']:.1f}J  {control['throughput_tok_s']:.0f} tok/s  "
          f"P={control['avg_power_w']:.0f}W  util={control['avg_util_pct']:.0f}%", flush=True)

    # start process P: phone [0,8) head pipeline, tail on GPU1 (keeps GPU0 work identical)
    print("launching phone [0,8) pipeline (tail on GPU1) ...", flush=True)
    phone_env = dict(os.environ)
    phone_cmd = [sys.executable, STAGEB, "--k-list", str(args.k), "--batch", "1",
                 "--n-gen", str(N_GEN), "--requests", str(args.phone_requests), "--warmups", "2",
                 "--gpu-uuid", GPU1_UUID, "--serial", args.serial, "--timeout", "1200",
                 "--output", str(log_dir / "phone_pipeline_result.json")]
    with open(phone_pipe_log, "w") as pl:
        phone = subprocess.Popen(phone_cmd, stdout=pl, stderr=subprocess.STDOUT, env=phone_env)

    # wait until the phone is actually DECODING [0,8), not just loading. stageb's phone
    # stage log flips to steady decode after "[stagenet] client connected"; wait for that
    # plus a margin so the treatment window overlaps real concurrent head compute.
    phone_stage_log = HERE / "stageb_logs" / f"phone_k{args.k}.log"
    deadline = time.monotonic() + 400
    decoding = False
    while time.monotonic() < deadline:
        if phone_stage_log.exists() and "[stagenet] client connected" in phone_stage_log.read_text(errors="replace"):
            decoding = True
            break
        if phone.poll() is not None:
            print("phone pipeline exited early:\n" + phone_pipe_log.read_text()[-1500:], file=sys.stderr)
            break
        time.sleep(1.0)
    time.sleep(3.0)  # let the phone reach steady per-token decode
    print(f"  phone pipeline decoding={decoding}; starting overlap window", flush=True)

    # (2) TREATMENT: identical GPU0 backlog, phone pipeline concurrently DECODING
    treatment = measure_gpu0(sampler, args.batch, args.steps, "treatment_phone_concurrent")
    phone_alive_during_window = phone.poll() is None
    print(f"  E={treatment['energy_j']:.1f}J  {treatment['throughput_tok_s']:.0f} tok/s  "
          f"P={treatment['avg_power_w']:.0f}W  util={treatment['avg_util_pct']:.0f}%  "
          f"phone_alive={phone_alive_during_window}", flush=True)

    sampler.stop()
    # let the phone pipeline finish so its measured per-request latency is available
    try:
        phone.wait(timeout=300)
    except subprocess.TimeoutExpired:
        phone.terminate()
        try:
            phone.wait(timeout=30)
        except subprocess.TimeoutExpired:
            subprocess.run(["adb", "-s", args.serial, "shell", "pkill -9 -f llama-layersplit"], check=False)
            phone.kill()

    # phone sustained single-stream head rate from its measured per-request latency
    r_phone_tok_s = phone_head_tok_s(log_dir / "phone_pipeline_result.json", args.k, N_GEN)

    # (1) interference
    dtput = (treatment["throughput_tok_s"] - control["throughput_tok_s"]) / control["throughput_tok_s"]
    denergy = (treatment["energy_per_token_mj"] - control["energy_per_token_mj"]) / control["energy_per_token_mj"]

    # (3) honest realised saving bound: the phone can carry R_phone tok/s of head work;
    # each such token saves the A6000 (e_full - e_tail). As a fraction of the A6000's
    # own decode rate that is R_phone / tput_gpu0.
    s_k = 1.0 - args.e_tail_mj / args.e_full_mj
    a6000_tput = control["throughput_tok_s"]
    f_sat = r_phone_tok_s / a6000_tput if a6000_tput else 0.0
    realised_saving_at_saturation = f_sat * s_k

    result = {
        "schema": "s14-stage-d-overlap-v1",
        "scope": "A6000_GPU0_BOARD_ENERGY_AND_THROUGHPUT_PLUS_PHONE_HEAD_RATE_ONLY_PHONE_ENERGY_UNKNOWN",
        "formal_claim": "MEASURED_CONCURRENT_OVERLAP_INTERFERENCE_AND_REALISED_SAVING_BOUND",
        "gpu0_index": GPU0_INDEX, "phone_tail_gpu": GPU1_UUID, "phone_serial": args.serial,
        "offload_depth_k": args.k, "batch": args.batch, "steps": args.steps,
        "measured_s_k": s_k,
        "control_phone_idle": control,
        "treatment_phone_concurrent": treatment,
        "interference": {
            "throughput_delta_frac": dtput,
            "energy_per_token_delta_frac": denergy,
            "clean_overlap": abs(dtput) < 0.03 and abs(denergy) < 0.03,
            "note": "GPU0 workload is byte-identical in both; delta is pure concurrency cost "
                    "(USB relay + host CPU + PCIe from the phone pipeline).",
        },
        "phone_head": {
            "phone_alive_during_window": phone_alive_during_window,
            "sustained_head_tok_s": r_phone_tok_s,
            "source": "phone_pipeline_result.json stage_a_us_p50 (single-stream [0,k) over adb/USB)",
        },
        "realised_saving": {
            "a6000_decode_tok_s": a6000_tput,
            "phone_offloadable_fraction_at_saturation": f_sat,
            "realised_gpu_board_saving_at_saturation": realised_saving_at_saturation,
            "note": "One single-stream phone carries f_sat of the A6000's decode rate; the "
                    "realised board saving is f_sat * s(k). This is the throughput-bounded "
                    "reality the Stage C accounting ceiling (f up to 0.879) ignores.",
        },
        "caveats": [
            "A6000 GPU0 board energy + throughput ONLY. Phone energy UNKNOWN.",
            "GPU0 runs the SAME full-model backlog in control and treatment; the phone tail runs on GPU1 to isolate pure concurrency interference.",
            "R_phone is single-stream (batched multi-seq HTP decode hangs, S1); more phones or a working batch would raise it.",
            "realised_saving_at_saturation assumes the A6000 is the bottleneck (busy). At the light mix-v1 rate the A6000 is idle-dominated, a different regime.",
            "Stage A/B energies are reused as measured constants (e_full, e_tail); NVML board-average +/-5W.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print("\n=== Stage D: measured phone||server overlap ===", flush=True)
    print(f"  interference: throughput {100*dtput:+.1f}%  energy/tok {100*denergy:+.1f}%  "
          f"clean={result['interference']['clean_overlap']}", flush=True)
    print(f"  phone sustained head rate: {r_phone_tok_s:.1f} tok/s  "
          f"(vs A6000 {a6000_tput:.0f} tok/s)", flush=True)
    print(f"  => one phone carries {100*f_sat:.1f}% of A6000 decode; realised board saving "
          f"at saturation = {100*realised_saving_at_saturation:.2f}%  (s(k)={100*s_k:.1f}%)", flush=True)
    print(f"  wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
