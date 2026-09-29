# Pixel 10 Pro CPU+GPU concurrent FFN worker - progress log

Newest last. UTC timestamps.

## 2026-09-25 00:40 UTC - start

- Read AGENTS.md, NEXT_AGENT_PROMPT_SCHEDULER.md, pixel-integration-2, pixel-second-phone, fast-path-M3
  (PIXEL_CPU_GPU_*, PIXEL_PACKED_CPU/GPU_RESULTS), dual-engine bench/worker/integration READMEs.
- Prior facts that shape the plan:
  - Production Pixel worker = `software/pixel10pro-packed-cpu-v1` (sha 64133753): original Q4_K/Q6_K
    weights, residual-corrected Q8_K activations, paired SDOT, dynamic 64-row chunks, 6 pinned threads (mask fc).
    Phone-local back-to-back replay: 6.3 ms/layer full width B1, 10.7 B2, 16.0 B4 (non-root, nc burst).
  - Same worker at server cadence (packed-server qualification, root via su, TCP from desktop):
    compute 28.6 ms/layer full, 16.2 ms half (5x slower than phone-local); campaign: 29.3 B1 / 70.6 B4.
    => the dominant production loss is not the kernel; cadence (DVFS / wake-up) or root cgroup.
  - Packed GPU (native Vulkan q4_K) ~20 ms full at B1, same as F16 GPU; F16 CPU+GPU split (Pixel agent,
    09-23) 18.3 -> 13.1 ms at B1, never tried with the packed CPU path (asserted off).
  - OP15 lesson: bench 1.21-1.24x did not carry to the campaign (GPU wake-up after between-token idle).
- Plan: (1) diagnose root/cadence gap; (2) bench packed CPU + Vulkan GPU split under back-to-back AND
  production-like cadence at M=1/2/4; (3) integrate only if it pays.

## 2026-09-25 01:35 UTC - tooling + D1 (phone-local, CPU cadence + GPU-only) DONE

- New worker (copy of the production Pixel packed-CPU source + env-gated additions, `software/pixel-cpugpu-v1/`,
  patch `CPUGPU_V1.patch`): `S43_PIXEL_GPU_TRAILING_COLUMNS` (GPU owns trailing columns as whole blocks, tuned
  packed CPU allowed alongside), `S43_PIXEL_CPU_BATCH_PAIR` (decode each weight row once for all rows),
  `S43_PIXEL_UCLAMP_MIN` (per-process sched_setattr util_min, inherited by all threads), `S43_PIXEL_CPU_POLL`
  (ggml threadpool poll level). Phone-side replay client `pixel-ffn-replay` (cadence emulation + in-call clock sampler).
- Staged in NEW dirs only: phone `/data/local/tmp/s43-pixel-cpugpu-20260925-v1` (libs/shard/prod worker copied from
  the Pixel agent's dirs, sha256 = GATE_CONFIG), desktop `/mnt/storage/s43-pixel-cpugpu-20260925-v1`.
- Smoke: Vulkan `mul_mat_vec_q4_k_f32_f32` for >1 column FAILS to compile on PowerVR (ErrorUnknown) -> packed GPU
  weights only usable at M=1; GPU side uses F16 (decoded Q4_K, `S42_PIXEL_GPU_EXPAND_F16`) for M>1.
- D1 (`physical/d1/SUMMARY.md`), compute ms/layer p50, full width, 6 layers streamed:

  | arm | burst M1/M2/M4 | prod-cadence M1/M2/M4 (10 ms gaps, 250 ms/token) |
  | --- | --- | --- |
  | production binary (control, x2) | 7.0 / 9.3 / 15.0 | **28.3 / 44.6 / 67.3** (campaign: 29.3 / 46.1 / 70.6) |
  | new binary, flags off | 6.8 / 9.3 / 15.1 | 28.3 / 44.4 / 68.2 (byte-identical outputs) |
  | + uclamp 1024 | 5.9 / 7.7 / 15.0 | 15.2 / 16.7 / 20.9 |
  | + uclamp + poll 100 | 5.4 / 7.6 / 14.9 | **6.8 / 9.3 / 17.3** (first layer of a token ~17 ms) |
  | + uclamp + batch pair | 5.9 / 6.8 / 13.1 | 14.9 / 16.5 / 20.4 |
  | GPU only F16 (+uclamp) | 14.5 / 30.2 / 86.8 | 27.7 / 37.0 / 100 |

  Clocks sampled during calls: prod cadence mid cores 400-550 MHz and DSU ~400-530 MHz vs 1.7-3.0 GHz / 1.4-1.7 GHz
  in burst -> the production 4.5x gap is DVFS (sched_pixel response_time 75 ms mid / 300 ms big).
  All CPU variants byte-identical to the production binary at M=1/2/4; GPU F16 rel-L2 4.4e-4 vs CPU.

## 2026-09-25 03:05 UTC - D2 CPU+GPU sweep, 3 reps: STEP-1 GATE FAIL (full table BENCH_TABLE.md)

GPU side = decoded F16 (Q4_K multi-column Vulkan pipeline does not compile), trailing whole-block ownership,
CPU = packed tuned path; all dual arms with uclamp 1024 + poll 100 (the no-boost dual is 0.24-0.54x).
Median over 3 reps (s10/s15 2 reps), x = speedup vs same-rep boosted CPU-only:

| GPU share | burst M1 / M2 / M4 | prod-cadence M1 / M2 / M4 |
| --- | --- | --- |
| 0 (CPU only, ms) | 6.03 / 8.03 / 17.24 | 7.27 / 9.71 / 20.60 |
| 1.0 (GPU only, ms) | 14.51 / 30.21 / 86.77 | 27.67 / 36.97 / 99.97 |
| 0.10 (5 thr) | 0.94x / 0.96x / 0.93x | 0.99x / 0.75x / 0.79x |
| 0.15 (5 thr) | 0.95x / 0.97x / 0.98x | 0.88x / 0.63x / 0.71x |
| 0.25 (6 thr) | 0.67x / 0.43x / 0.73x | 0.53x / 0.47x / 0.56x |
| 0.25 (5 thr, best) | 0.91x / 1.01x / 0.78x | 0.58x / 0.48x / 0.57x |
| 0.33 | 0.35x / 0.39x / 0.45x | 0.39x / 0.36x / 0.40x |
| 0.40 | 0.40x / 0.40x / 0.37x | 0.36x / 0.34x / 0.37x |
| 0.50 | 0.56x / 0.41x / 0.33x | 0.31x / 0.32x / 0.34x |
| 0.60 | 0.34x / 0.39x / 0.26x | 0.28x / 0.29x / 0.27x |

- Why: GPU F16 is 2.4x (M1) to 5x (M4) slower than the boosted packed CPU, so the ideal share is <= 20-29 %
  and the ideal bound is 1.41x (M1) / 1.25x (M2) / 1.17x (M4) with zero interference; measured interference
  is large (CPU leg ~25 % slower while the GPU runs; GPU leg 1.5-2.5x its solo share time, GPU clock drops
  to 470-970 MHz). Row split (GPU rows with full weights) is bounded by GPU-only M1 = 14.5 ms >= CPU-only M4,
  and loses at M2 (14.5 vs 8.0 ms): not implemented.
- rel-L2 dual vs CPU-only 2.2e-4..3.8e-4 (max over reps); GPU-only 4.4e-4.
- Decision: do NOT integrate CPU+GPU. The measured lever is the per-process DVFS boost (+ batch pair):
  CPU-only boosted vs production binary at prod cadence 28.3/44.6/67.3 -> 7.3/9.7/20.6 ms (3.9x/4.6x/3.3x),
  batch pair a further 1.12x (M2) / 1.26x (M4), byte-identical. Over-TCP qualification of that running.

## 2026-09-25 03:40 UTC - Step 2 (boost integrated env-gated) over-TCP qualification PASS; report written

- tcp1 (production AdbTcpPhoneWorkerSession, rooted, phone lock, adb forward, finite budget, A/B/C/A/B):
  per-layer RPC production 38.4 / 61.4 / 88.4 ms -> boost+batch 13.6 / 20.2 / 32.5 ms (B1/B2/B4);
  compute 27.8 / 45.5 / 72.0 -> 6.5 / 9.7 / 18.7 ms; outputs byte-identical to the production worker;
  every stop exit 0, forward removed, boot unchanged.
- tcp2 sustained boost+batch 3,954 calls (~4.5 min): no drift per quarter, battery 36.5 -> 35.5 C.
- d3: rows=3 byte-identical; prod cadence 58.5 -> 14.0 (boost) -> 11.3 ms (boost+batch).
- Final state: no Pixel worker, no adb forward, battery 100 % 36.0 C, lock free. README.md written.
