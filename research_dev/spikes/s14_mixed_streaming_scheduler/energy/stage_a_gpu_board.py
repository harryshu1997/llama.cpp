#!/usr/bin/env python3
"""S14 energy Stage A: A6000 GPU-board energy decomposition.

Measures, via NVML (nvidia-smi), the selected-A6000 board energy per decoded
token when the GPU runs the FULL model [0,48) versus only the TAIL [k,48) -- i.e.
when the phone has offloaded the head [0,k). The tail-vs-full energy gap is the
GPU-board benefit of offloading [0,k), at equal decoded work.

Scope (per NEXT_PLAN section 11 / S14 CP5): selected-A6000 GPU_BOARD energy ONLY.
Phone energy, USB/relay, and total-wall are UNKNOWN and out of scope. This is a
mechanism DIAGNOSTIC, not a total-system energy claim, and it does not reuse the
frozen S11-E0 cohort.

Method:
  - one continuous NVML power sampler on the selected GPU (arrival-stamped);
  - layersplit --mode tailbench (LLAMA_LAYER_START=k) prints BENCH_START/BENCH_END
    epochs (system_clock) bracketing the timed decode window; activation values
    are irrelevant to energy, so a fixed dummy is injected;
  - energy = ZOH integral of power over [BENCH_START, BENCH_END];
  - matched work (same batch x steps) across all k; rotated repeats cancel drift;
  - power.limit is required invariant across the whole campaign.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
MODEL = "/home/myid/zs89458/Documents/models/gemma-4-12B-it-f16.gguf"
BIN = str(REPO_ROOT / "build-cuda/bin/llama-layersplit")
N_LAYER = 48


class Sampler:
    """Continuous NVML power sampler: streams nvidia-smi and arrival-stamps rows."""

    def __init__(self, gpu_index: int, period_ms: int = 100) -> None:
        self.gpu_index = gpu_index
        self.period_ms = period_ms
        self.samples: list[tuple[float, float, float, str]] = []  # (t, power_w, util, pstate)
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = False
        self.power_limit_w: float | None = None

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop:
                break
            t = time.time()
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                power = float(parts[0])
                limit = float(parts[1])
                util = float(parts[2])
            except ValueError:
                continue
            pstate = parts[3]
            self.power_limit_w = limit
            self.samples.append((t, power, util, pstate))

    def start(self) -> None:
        self._proc = subprocess.Popen(
            ["nvidia-smi", "-i", str(self.gpu_index),
             "--query-gpu=power.draw,power.limit,utilization.gpu,pstate",
             "--format=csv,noheader,nounits", f"-lms", str(self.period_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def energy_j(self, t0: float, t1: float) -> dict[str, Any]:
        """ZOH energy integral of power over [t0, t1] plus sample-quality gates."""
        rows = [(t, p, u, ps) for (t, p, u, ps) in self.samples if t0 - 1.0 <= t <= t1 + 1.0]
        rows.sort(key=lambda r: r[0])
        window = [(t, p) for (t, p, _, _) in rows if t0 <= t <= t1]
        if len(window) < 5:
            raise RuntimeError(f"too few in-window samples ({len(window)}) for [{t0},{t1}]")
        energy = 0.0
        max_gap = 0.0
        for i in range(len(rows) - 1):
            ta, pa = rows[i][0], rows[i][1]
            tb = rows[i + 1][0]
            lo, hi = max(ta, t0), min(tb, t1)
            if hi > lo:
                energy += pa * (hi - lo)
                max_gap = max(max_gap, tb - ta)
        powers = [p for _, p in window]
        util_vals = [u for (t, p, u, ps) in rows if t0 <= t <= t1]
        return {
            "energy_j": energy,
            "duration_s": t1 - t0,
            "avg_power_w": energy / (t1 - t0),
            "min_power_w": min(powers),
            "max_power_w": max(powers),
            "n_samples": len(window),
            "max_sample_gap_s": max_gap,
            "avg_util_pct": sum(util_vals) / len(util_vals) if util_vals else None,
        }


def run_tailbench(k: int, batch: int, steps: int, gpu_index: int) -> dict[str, Any]:
    """Run one tailbench decode over [k,48); return BENCH markers + tokens."""
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_index), "LLAMA_LAYER_START": str(k)}
    import os
    full_env = dict(os.environ)
    full_env.update(env)
    proc = subprocess.run(
        [BIN, "-m", MODEL, "-ngl", "99", "--mode", "tailbench", "-b", str(batch), "-n", str(steps)],
        capture_output=True, text=True, env=full_env, timeout=600,
    )
    start = end = tokens = None
    model_buf_mib = None
    for line in (proc.stdout + "\n" + proc.stderr).splitlines():
        if line.startswith("BENCH_START"):
            start = float(line.split()[1])
        elif line.startswith("BENCH_END"):
            end = float(line.split()[1])
            for tok in line.split():
                if tok.startswith("tokens="):
                    tokens = int(tok.split("=")[1])
        elif "CUDA0 model buffer size" in line:
            model_buf_mib = float(line.split("=")[1].strip().split()[0])
    if start is None or end is None or tokens is None:
        raise RuntimeError(f"tailbench k={k} did not emit BENCH markers (rc={proc.returncode})")
    return {"k": k, "bench_start": start, "bench_end": end, "tokens": tokens,
            "model_buffer_mib": model_buf_mib, "returncode": proc.returncode}


def rotated_schedule(k_list: list[int], repeats: int) -> list[int]:
    """Interleaved rotation so each k is spread across the campaign (drift cancel)."""
    sched = []
    for r in range(repeats):
        order = k_list if r % 2 == 0 else list(reversed(k_list))
        sched.extend(order)
    return sched


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-list", default="0,6,12")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--output", default=str(HERE / "stage_a_result.json"))
    args = ap.parse_args()

    k_list = [int(x) for x in args.k_list.split(",")]
    sched = rotated_schedule(k_list, args.repeats)

    sampler = Sampler(args.gpu)
    sampler.start()
    time.sleep(2.0)  # let the sampler stream a few rows

    runs: list[dict[str, Any]] = []
    # one untimed warmup to ramp clocks
    print(f"warmup (k={k_list[-1]}) ...", flush=True)
    try:
        run_tailbench(k_list[-1], args.batch, min(args.steps, 300), args.gpu)
    except Exception as e:  # noqa: BLE001
        print(f"warmup failed: {e}", file=sys.stderr)

    for i, k in enumerate(sched):
        print(f"[{i+1}/{len(sched)}] tailbench k={k} tail=[{k},{N_LAYER}) ...", flush=True)
        r = run_tailbench(k, args.batch, args.steps, args.gpu)
        e = sampler.energy_j(r["bench_start"], r["bench_end"])
        r.update(e)
        r["energy_per_token_mj"] = 1000.0 * e["energy_j"] / r["tokens"]
        r["throughput_tok_s"] = r["tokens"] / e["duration_s"]
        runs.append(r)
        print(f"    E={e['energy_j']:.1f} J  {r['energy_per_token_mj']:.3f} mJ/tok  "
              f"avgP={e['avg_power_w']:.1f}W  util={e['avg_util_pct']:.0f}%  "
              f"tput={r['throughput_tok_s']:.0f} tok/s  buf={r['model_buffer_mib']}MiB", flush=True)

    sampler.stop()

    # aggregate per k
    per_k: dict[int, dict[str, Any]] = {}
    for k in k_list:
        rk = [r for r in runs if r["k"] == k]
        ept = sorted(r["energy_per_token_mj"] for r in rk)
        per_k[k] = {
            "n": len(rk),
            "energy_per_token_mj_median": ept[len(ept) // 2],
            "energy_per_token_mj_min": ept[0],
            "energy_per_token_mj_max": ept[-1],
            "model_buffer_mib": rk[0]["model_buffer_mib"],
            "avg_power_w": sum(r["avg_power_w"] for r in rk) / len(rk),
            "throughput_tok_s_median": sorted(r["throughput_tok_s"] for r in rk)[len(rk) // 2],
        }
    base = per_k[0]["energy_per_token_mj_median"] if 0 in per_k else None
    savings = {}
    for k in k_list:
        if base and k != 0:
            savings[k] = {
                "gpu_energy_saved_frac": 1.0 - per_k[k]["energy_per_token_mj_median"] / base,
                "layer_fraction_offloaded": k / N_LAYER,
                "hbm_freed_mib": per_k[0]["model_buffer_mib"] - per_k[k]["model_buffer_mib"],
            }

    result = {
        "schema": "s14-stage-a-gpu-board-energy-v1",
        "scope": "SELECTED_A6000_GPU_BOARD_ENERGY_ONLY_PHONE_UNKNOWN",
        "formal_claim": "GPU_BOARD_DIAGNOSTIC",
        "gpu_index": args.gpu,
        "power_limit_w": sampler.power_limit_w,
        "batch": args.batch, "steps": args.steps, "n_layer": N_LAYER,
        "matched_work_tokens": args.batch * args.steps,
        "schedule": sched,
        "runs": runs,
        "per_k": {str(k): v for k, v in per_k.items()},
        "gpu_board_savings_vs_full": {str(k): v for k, v in savings.items()},
        "caveats": [
            "Selected-A6000 GPU_BOARD energy only. Phone/USB/total-wall UNKNOWN.",
            "Batched decode, saturated (gap_ms=0): assumes the server stays busy (overlap). Idle-wait energy is NOT modelled here.",
            "Diagnostic, not a total-system energy claim. Does not reuse the S11-E0 frozen cohort.",
            "NVML power.draw is a ~1 Hz board-average sensor (+/-5 W); windows are sized for many samples.",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.output}")
    print(f"power_limit={sampler.power_limit_w}W  matched_work={args.batch*args.steps} tok/run")
    for k in k_list:
        v = per_k[k]
        s = savings.get(k, {})
        print(f"  tail[{k:2d},48): {v['energy_per_token_mj_median']:.3f} mJ/tok  "
              f"buf={v['model_buffer_mib']:.0f}MiB  P={v['avg_power_w']:.0f}W"
              + (f"  -> GPU energy saved {100*s['gpu_energy_saved_frac']:.1f}% "
                 f"(offload {100*s['layer_fraction_offloaded']:.0f}% layers, HBM -{s['hbm_freed_mib']:.0f}MiB)" if s else "  (baseline)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
