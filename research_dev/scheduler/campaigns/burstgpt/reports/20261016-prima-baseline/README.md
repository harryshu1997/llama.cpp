# prima.cpp baseline on the rig (2026-10-16)

Three metered runs of prima.cpp (OpenCPIL/prima.cpp, commit 288df6a, ICLR 2026) on the
Qwen3-32B Q8_0 single-model trace with the desktop (RTX 4060 Ti 16 GB, i9-12900K) as ring head
and the OnePlus 15 and Pixel 10 Pro as ring workers over USB (adb tunnels). Same trace, plan
template, meters and accountants as the paper's 32B Q8_0 arms (`template-q8two-ts3-anchor-hfix-v2`,
14 requests, 13 on the 32B, one 1B overlay request at t = 1,497 s; span 1,676 s).

## Result

| system (32B Q8_0 single-model trace) | runs | fleet kJ (mean) | range | vs desktop | host CPU / GPU kJ | OP15 kJ (W) | Pixel kJ (W) | run s | TTFT p50 s | TPOT p50 s | E2E p50 / p90 s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| desktop only (pQ8S1nph/b/c) | 3 | **233.0** | 232.6–233.4 | – | 181.8 / 48.4 | 1.9 (1.0) | 0.9 (0.5) | 1,873 | 31–38 | 1.15–1.21 | 372–384 / 1,064 |
| **prima.cpp, 3 devices, Halda (pPR1/2/3)** | 3 | **158.6** | 155.8–160.4 | **−31.9 %** | 62.2 / 77.9 | 8.1 (2.9)* | 10.4 (3.7) | 2,789 | 267–292 | 2.22–2.47 | 748–856 / 1,527 |
| ours, two-phone controller (pA1q8va/vb/vc) | 3 | **118.2** | 116.9–120.4 | −49.3 % | 56.3 / 47.8 | 9.2 (4.9) | 4.9 (2.6) | 1,880 | 48–53 | 1.20–1.30 | 373–380 / 1,081 |
| ours, OnePlus owns FFN 0–20, A (pB1q8od/oa/ob) | 3 | 166.8 | 166.4–167.1 | −28.4 % | 100.6 / 52.9 | 12.4 (6.0) | 0.9 (0.4) | 2,081 | 95–140 | 0.95–0.98 | 387–410 / 1,092 |
| ours, A + Pixel decodes 21–40, B (pC1q8pc/pa/pb) | 3 | 112.1 | 107.8–116.2 | −51.9 % | 25.9 / 62.9 | 14.1 (5.5) | 9.2 (3.6) | 2,579 | 109–534 | 1.34–1.35 | 767–880 / 1,526 |

\* the OnePlus meter flagged CHARGING_IN_WINDOW (all three runs) and COUNTER_JUMP (runs 1–2): its
energy is an upper bound. Even at 0 kJ for the OnePlus, prima.cpp stays at 150.5 kJ.

Relative to prima.cpp: controller −25.5 % energy at half its TPOT, a fifth of its TTFT and the
desktop's E2E; B −29.3 % at a similar E2E; A +5.2 % energy but desktop-class latency.
Latency percentiles are over all 14 requests (the paper's convention); Qwen-only TTFT p50 is
361 / 455 / 464 s for the three prima.cpp runs.

## What prima.cpp did

- Halda (HiGHS) profiled every device at startup (see `artifacts/pPR*-profiler-table.txt`): desktop
  CPU Q8 ~201–220 GFLOPS, 22.9 GiB RAM available, 12.8 GiB free VRAM, CUDA Q8 55 TFLOPS, NVMe
  2.7 GB/s; OnePlus 283–364 GFLOPS, 7.8 GiB available, 129 GB/s RAM, 3.6 GB/s disk; Pixel
  46–60 GFLOPS, 8.8 GiB available, 23.5 GB/s RAM, 1.4 GB/s disk.
- Split: desktop 40 layers (18 on the GPU) / OnePlus 11 / Pixel 13 in runs 1–2; 39 / 12 / 13
  in run 3. All phone layers run on the phone CPUs (whole layers: attention, KV cache, FFN,
  prefill). No device was removed as "weak".
- Server: rank-0 `llama-server -c 16384 -np 4 -cb -b 2048 -ub 512 --prefetch` (per slot 4,096 =
  our arms' context), token-id prompts, `-a qwen3-32b-q8_0`. The 1B overlay request ran on a
  separate single-node prima.cpp server on the GPU (as in our arms).
- Per request (run 1): decode 1.99–3.32 s/token, prefill 96–200 ms/token (5–10 tokens/s), queue
  grew 0 → 511 s over the trace; total queue time 3,394 / 3,926 / 4,134 s per run (ours 1,493).

## Why ours is cheaper and faster

1. NPU vs CPU: our OnePlus layers run on the Hexagon NPU; prima.cpp's run on the phone CPUs.
2. FFN only: we ship the FFN input and keep attention, KV state and the residual stream on the
   desktop; prima.cpp moves whole layers, so every token crosses three USB hops through two phone
   CPUs with attention on the phones.
3. Prefill placement: our controller keeps prefill on the desktop GPU (TTFT 48–53 s vs 267–292 s);
   prima.cpp pushes 2k-token prompts through the phone CPUs.
4. Per-step decisions vs a fixed split: Halda sets layer windows once from a latency model; our
   controller chooses the phone set per decode step (one or two requests on the OnePlus alone,
   three on both phones), which keeps E2E at the desktop's.

Energy mechanics: prima.cpp cuts the desktop CPU from 181.8 to 62.2 kJ (like us: 56.3) but the GPU
rises 48.4 → 77.9 kJ because the run lasts 2,789 s instead of 1,873 (idle board power), and the
phones draw 2.9 + 3.7 W for the whole run.

## Setup (short; full procedure in the memory note `prima-cpp-baseline-procedure-2026-10-16`)

- Fork patched for Qwen3 (`claude-scratch/prima/patches/port_qwen3.py`, profiler fix
  `fix_profiler_qwen3.py`; 40 greedy tokens identical to upstream llama.cpp; ring == single node),
  server streams token ids (`send_partial_response` "tokens"). Builds: desktop Makefile CUDA
  (`artifacts/build_cuda.sh`), Android NDK r27c static libc++ with libzmq; fio `--disable-shm`.
- Harness `tools/prima_baseline.py` (this repo) + desktop `$R/rig/launch_prima_baseline.sh`,
  `prima_chain.sh` (meter_phones.sh --charging controlled), `prima_series.sh` (copies in
  `artifacts/`). Phones: `/data/local/tmp/s43-prima-v1/` (llama-cli, fio, 0.6B smoke model;
  Pixel also the 32B Q8_0; OnePlus 32B at `/data/local/tmp/s43-q8/`).
- Series: `PRIMA_EXTRA=--prefetch prima_series.sh pPR 3` (2026-10-16 15:22–18:04 UTC), cool-down
  600 s, runs `$R/inputs-prima-pPR{1,2,3}`, meters `$R/meter/pPR{1,2,3}-prima`.
- Not tuned beyond prima.cpp's defaults (`--gpu-mem` left to Halda; `-lw` fixed windows available
  via `PRIMA_LAYER_WINDOW` but unused). Gemma 4 cannot run on the fork (architecture missing).

## Artifacts

`artifacts/fleet-prima-final.{json,md}`, `latency-prima-final.{json,md}` (all 15 runs above),
`pPR*-PRIMA_RECEIPT.json`, `pPR*-RESULT-summary.json`, `pPR*-profiler-table.txt`,
`pPR*-assignment.txt`, `PRIMA-SERIES-pPR.log`, `PHONES-q8-two.json`, rig scripts.
Full run directories stay on the desktop (`$R/inputs-prima-pPR*/run-eval/run/`, with server,
overlay and worker logs and the 14 token streams per run).
