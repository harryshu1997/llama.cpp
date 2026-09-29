# Pixel 10 Pro FFN worker: CPU+GPU concurrency (no) and a DVFS boost (yes), 2026-09-25

Task: make the Pixel's phone FFN worker (Qwen3-14B layers 18-23, full width 17,408) fast enough to keep up
at batch 2-4, first by running CPU and GPU concurrently on disjoint column slices (the OP15 dual-engine
mechanism). Nothing is committed. No campaign trace was run. The OP15 was not touched. Progress log:
`PROGRESS.md`.

## Headline

- **CPU+GPU concurrency does not pay on the Pixel, so it is not integrated.** The Step-1 gate was >= 1.4x at
  M=2-4. The best measured cell is 1.01x (burst, M=2, GPU share 0.25, 5 CPU threads). At server-like
  cadence, every dual configuration is 0.26-0.99x of CPU-only. Two causes:
  - the Vulkan F16 GPU path is 2.4x (M=1) to 5x (M=4) slower than the packed CPU;
  - running both engines at once slows both.

  Even with zero interference, the bound would be 1.42x / 1.27x / 1.20x at M=1 / 2 / 4.
- **The Pixel's production slowness is DVFS, not compute.** At server cadence the qualified worker's CPU
  clusters sit at 400-550 MHz and the DSU/L3 at ~400-530 MHz. That makes one layer 28 / 45 / 72 ms at
  B1 / B2 / B4, against 6-15 ms in back-to-back replay. The `sched_pixel` governor needs 75 ms (mid cores)
  or 300 ms (big core) of busy time to ramp, and the worker's calls are 6-30 ms bursts.
- **Fix, integrated env-gated into the Pixel worker:** a per-process uclamp floor, plus thread-pool
  polling between calls, plus a batched pair-dot. Qualified over ADB TCP with the production lifecycle
  class:
  - outputs are byte-identical to the qualified worker at rows 1, 2, 3 and 4;
  - per-layer RPC: 38.4 -> 13.6 ms (B1), 61.4 -> 20.2 ms (B2), 88.4 -> 32.5 ms (B4), i.e. 2.8x / 3.0x / 2.7x;
  - worker compute: 4.3x / 4.7x / 3.9x faster.
- **Campaign meaning.** The Pixel's six layers per decode step go from 231 / 369 / 530 ms to
  81 / 121 / 195 ms at B1 / B2 / B4. OP15's eighteen layers take ~175 ms (B1) and ~220 ms (B4). The Pixel is
  therefore no longer the longer of the two phones at B4. Per layer it is still ~2.7x slower than OP15 at B4.

## 1. Setup

| item | value |
| --- | --- |
| phone | Pixel 10 Pro (Tensor G5, PowerVR DXT-48-1536), rooted, `adb -P 5037 -s 5A040DLCH004ES`, USB powered, battery 100 %, 27-37 C |
| weights | the Pixel agent's `QWEN_PACKED.ffn.gguf` (original Q4_K/Q6_K, sha256 `940f5f1f...`), layers 18-23, 6 layers streamed per step (948 MB, not cache resident) |
| CPU path | qualified packed path: Q8_K residual-corrected activations, paired SDOT, dynamic 64-row chunks, 6 pinned threads on cores 2-7 (`fc`) |
| GPU path | Vulkan0 F16. The weights are decoded Q4_K (`S42_PIXEL_GPU_EXPAND_F16`), with the tuned `vec4_u1` shader (WG 128, rows 8, SG 128). **Packed Q4_K on the GPU fails for M>1**: `mul_mat_vec_q4_k_f32_f32` pipeline creation returns ErrorUnknown on the PowerVR driver (`physical/s0/.../dual25/worker.log`). |
| split | the GPU owns the trailing `S43_PIXEL_GPU_TRAILING_COLUMNS` of every layer as whole blocks, and the CPU owns the rest. Every runtime width (4,352 quantum) stays served. Per-call join and F32 merge use the existing secondary-backend mechanism. |
| method | the real worker binary plus a phone-side replay client (`pixel-ffn-replay`) over 127.0.0.1. "burst" = back-to-back calls. "prod" = 6 layer calls with 10 ms gaps, then 250 ms idle per token, emulating the server cadence. A sampler thread records CPU/GPU/DSU clocks during each call. |
| statistics | per-call worker compute, median of 3 repetitions (reps 2-3 in reversed arm order, separate worker processes), p90 pooled |

The bench is the worker itself rather than `ffn-dual-bench.cpp`, for two reasons: the packed int8 CPU
kernels live only in the Pixel worker, and the integration would otherwise have to be re-measured.

## 2. Step 1 bench: CPU+GPU split (3 reps; full table `BENCH_TABLE.md`, raw `physical/d2r1`, `physical/d2r23`)

All dual arms have the DVFS boost on, because without it everything is clock-bound (`s25-noboost`: 0.24-0.54x).
Wall time is the per-layer worker compute p50 in ms. `x` is the speedup vs the same repetition's CPU-only arm.

| GPU share (cols) | burst M1 | burst M2 | burst M4 | prod M1 | prod M2 | prod M4 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 = CPU only | 6.03 ms | 8.03 ms | 17.24 ms | 7.27 ms | 9.71 ms | 20.60 ms |
| 1.0 = GPU only | 14.51 ms (0.42x) | 30.21 ms (0.27x) | 86.77 ms (0.20x) | 27.67 ms | 36.97 ms | 99.97 ms |
| 0.10 (1,792), 5 thr | 0.94x | 0.96x | 0.93x | 0.99x | 0.75x | 0.79x |
| 0.15 (2,560), 5 thr | 0.95x | 0.97x | 0.98x | 0.88x | 0.63x | 0.71x |
| 0.25 (4,352), 6 thr | 0.67x | 0.43x | 0.73x | 0.53x | 0.47x | 0.56x |
| 0.25, 5 thr (best) | 0.91x | **1.01x** [0.95-1.16] | 0.78x | 0.58x | 0.48x | 0.57x |
| 0.25, 4 thr (1 rep) | 0.74x | 0.78x | 0.67x | 0.47x | 0.46x | 0.49x |
| 0.33 (5,632) | 0.35x | 0.39x | 0.45x | 0.39x | 0.36x | 0.40x |
| 0.40 (6,912) | 0.40x | 0.40x | 0.37x | 0.36x | 0.34x | 0.37x |
| 0.50 (8,704) | 0.56x | 0.41x | 0.33x | 0.31x | 0.32x | 0.34x |
| 0.60 (10,496) | 0.34x | 0.39x | 0.26x | 0.28x | 0.29x | 0.27x |

Engine legs (s25, 6 threads, burst):

| M | CPU leg (75 % of columns) | GPU leg (25 %) | CPU-only, 100 % of columns |
| --- | ---: | ---: | ---: |
| 1 | 5.9 ms | 8.9 ms | 6.0 ms |
| 2 | 6.9 ms | 17.9 ms | 8.0 ms |
| 4 | 12.1 ms | 22.8 ms | 17.2 ms |

So the GPU leg is always the long one. Logical (F16-equivalent) throughput: CPU-only 89 / 67 / 31 GB/s
(burst M1 / M2 / M4); physically that is 26 / 20 / 9 GB/s of packed bytes. The best dual reaches 81 / 67 / 24
logical and 38 / 32 / 11 physical GB/s. rel-L2 vs CPU-only is 2.2e-4 to 3.8e-4 for dual and 4.4e-4 for
GPU-only; this is not bit-exact, because the GPU runs F16 weights x F32 activations.

Why it cannot win:

1. **Throughput ratio.** The GPU F16 GEMV is 14.5 / 30 / 87 ms at M=1 / 2 / 4, against 6.0 / 8.0 / 17.2 ms for
   the CPU. The ideal share is therefore t_cpu / (t_cpu + t_gpu) = 29 / 21 / 17 %, and the zero-interference
   bound (1 + t_cpu / t_gpu) is 1.42x / 1.27x / 1.20x. The Vulkan multi-column F16 path scales almost linearly with M: it does
   not reuse weights. The packed Q4_K multi-column pipeline does not compile at all.
2. **Interference.**
   - With the GPU running, the CPU leg is 15-30 % slower per column at M=1-2: 5.9 ms for 75 % of the
     columns, vs 6.0 ms for 100 % alone at M=1.
   - The GPU leg takes 1.05-2.5x its solo share time: 8.9 vs 3.6 ms at M=1, 17.9 vs 7.6 ms at M=2.
   - The sampled GPU clock falls from 1,094 MHz (solo) to 470-970 MHz in dual mode.
   - The PowerVR driver threads (`vk_*`) are unpinned (cores 0-7) and preempt the pinned compute threads.
   - The bus is not the limit: dual combined traffic is 10-38 GB/s physical.
3. **Row split** (GPU rows with full weights) is bounded by GPU-only M=1 (14.5 ms). That already equals
   CPU-only M=4 and is 1.8x CPU-only M=2, so it was not implemented.

Decision: **FAIL, not integrated.** The trailing-split code stays in the worker, but is off unless
`S43_PIXEL_GPU_TRAILING_COLUMNS` is set. Using it is not recommended.

## 3. What actually limits the Pixel: DVFS at server cadence (`physical/d1`, `physical/d3`)

CPU-only, per-layer worker compute p50 in ms. "prod" is the emulated server cadence. The production-binary
rows are the mean of two controls.

| worker | burst M1 / M2 / M4 | prod M1 / M2 / M3 / M4 | mid-core MHz / DSU MHz during prod calls |
| --- | --- | --- | --- |
| production binary (sha `64133753`) | 7.0 / 9.3 / 15.4 | **28.5 / 44.4 / 58.5 / 67.8** | 404-850 / 405-530 |
| campaign, measured at the server (for reference) | | 29.3 / 46.1 / - / 70.6 | |
| new binary, all flags off | 6.8 / 9.3 / 15.1 | 28.3 / 44.4 / - / 68.2 (byte-identical) | same |
| + `S43_PIXEL_UCLAMP_MIN=1024` | 5.9 / 7.7 / 15.0 | 15.2 / 16.7 / - / 20.9 | 2,570-2,800 / 740-1,000 |
| + `S43_PIXEL_CPU_POLL=100` only | 6.1 / 8.7 / 16.8 | 15.6 / 16.1 / - / 31.3 | |
| + uclamp + poll (**boost**) | 5.4 / 7.6 / 14.9 | **6.8 / 9.3 / 14.0 / 17.3** | 2,870-2,980 / 1,300-1,450 |
| + boost + `S43_PIXEL_CPU_BATCH_PAIR=1` | 0.96x / 1.11x / 1.18x vs boost (same rep, 2 reps) | 1.02x / 1.12x / - / 1.26x vs boost; M3 11.3 ms | |

Governor facts (`sched_pixel`, read as root):

- `response_time_ms` is 75 on the mid clusters (policy2/5) and 300 on the big core (policy7);
  `up_rate_limit_us` is 500.
- A PMU limiter is enabled (`limit_frequency` 1.785 / 2.457 GHz).
- The DSU clock (`200c0780.dsufreq`) follows CPU activity.

How the three flags act:

- **uclamp** makes the governor pick high CPU clocks as soon as a compute thread is runnable.
- **poll** keeps the pool threads spinning for ~13M rounds (~25-65 ms) after each graph. That covers the
  ~10 ms gaps between layer calls, so the cores and the DSU stay up.
- **Batch pair** decodes each Q4_K/Q6_K weight row once for all rows of a call (`pixel_q4_pair<2B>`),
  instead of once per row. The arithmetic per output is unchanged, so outputs are byte-identical.

The first call of each token still takes ~15-17 ms at B1, because the 250 ms token gap is longer than the
poll window.

## 4. Step 2: integration and over-TCP qualification (`physical/tcp1`, `physical/tcp2`)

`tools/qualify_tcp.py` drives the scheduler's own `AdbTcpPhoneWorkerSession`: a rooted launch under
`/data/local/tmp/.s42-pixel-ffn-kernels.lock`, sha256 preflight, adb forward, and an exact finite budget.
The stop is the normal budget exit, with the forward removed and the boot id checked. The desktop sends real
layer 18-23 activations: 6 layer calls per token with a 1 ms host gap, then 250 ms idle. Arms ran in the
order A/B/C/A/B.

| worker (env) | rows | compute p50 / p90 ms | RPC p50 / p90 ms | non-compute ms | first layer / others ms | 6-layer RPC ms | outputs |
| --- | ---: | --- | --- | ---: | --- | ---: | --- |
| production (2 arms) | 1 | 27.8 / 31.7 | 38.4 / 43.0 | 10.2 | 27.9 / 27.7 | 231 | reference |
| | 2 | 45.5 / 51.7 | 61.4 / 67.5 | 15.2 | 47.8 / 43.7 | 369 | reference |
| | 4 | 72.0 / 81.7 | 88.4 / 98.1 | 17.4 | 82.0 / 67.2 | 530 | reference |
| boost (2 arms) | 1 | 6.4 / 15.4 | 13.4 / 25.5 | 7.0 | 15.6 / 6.4 | 80 | byte-identical |
| | 2 | 10.3 / 18.5 | 21.2 / 32.0 | 10.9 | 18.5 / 9.9 | 127 | byte-identical |
| | 4 | 20.6 / 24.0 | 34.9 / 39.4 | 14.0 | 21.9 / 20.3 | 210 | byte-identical |
| **boost + batch pair** | 1 | 6.5 / 14.6 | **13.6** / 23.2 | 7.1 | 15.2 / 6.4 | **81** | byte-identical |
| | 2 | 9.7 / 17.1 | **20.2** / 29.7 | 10.8 | 16.9 / 9.5 | **121** | byte-identical |
| | 4 | 18.7 / 22.1 | **32.5** / 37.7 | 13.7 | 20.5 / 18.3 | **195** | byte-identical |

- The production arm reproduces the campaign's per-layer RPC: 38.4 / 53.8 / 81.6 ms at B1 / B2 / B4.
- The boost also lowers the non-compute part of the RPC by 20-30 %, because the phone's network path runs at
  higher clocks.
- **Sustained run** (`tcp2`): boost + batch pair, 3,954 calls, ~4.5 min at load. Per-quarter RPC p50 stays at
  13.6-13.8 ms (B1), 20.4-21.0 ms (B2) and 33.8-34.5 ms (B4), with no drift. Outputs were deterministic, the
  battery went 36.5 -> 35.5 C, and the worker exited 0 with the forward removed and the boot unchanged.
- **B3** (phone-local, `physical/d3`): byte-identical. Production cadence: 58.5 -> 14.0 (boost) -> 11.3 ms
  (boost + batch pair).
- **Lifecycle**: every worker in this report exited on its finite budget. No worker was signalled or killed,
  none was left, no forward was left, and the boot id never changed.

## 5. How to enable (exact)

The new worker is staged in its own phone dir, with byte-identical copies of the qualified libraries and the
shard:

| phone path (`/data/local/tmp/s43-pixel-cpugpu-20260925-v1/`) | sha256 |
| --- | --- |
| `llama-ffn-split-worker` | `5d824455d10961fda0fc6369ee8b885195107f68f450198ac2e6d94549174128` |
| `libggml.so` | `601b8a7c7d14ab951ee85e6680bea3ed428b73ed63fc8a1fcb55d4999760c6f2` |
| `libggml-base.so` | `ec8396655c0b24bf828702a6e83371fe6bd3f24b57149e3be7d666813ff9f49d` |
| `libggml-cpu.so` | `b911532d756cad93e74391e86ed4d0e8e6f66773ec0dab79f8aa893021b0589d` |
| `libggml-vulkan.so` | `892bf36afff1c3b964afe421a25b9b6cbf0e55de949bee093c91038114cd4fa1` |
| `QWEN_PACKED.ffn.gguf` | `940f5f1f2ce0c68d726713e0b1ec86808334c7ca769feac07cd3fa8581c4eae9` |

The source is `software/pixel-cpugpu-v1/`. `CPUGPU_V1.patch` is its diff against the production worker source
`20260922-fast-path-M3/software/pixel10pro-packed-cpu-v1`. The build line is the Pixel agent's recipe; see
`software/pixel-cpugpu-v1/SHA256SUMS`. With none of the new variables set, the binary behaves like the
production worker (byte-identical outputs, D1).

In the helper-phone config (`GATE_CONFIG.json` `helper_phone`):

- `worker_path` = the path above;
- `library_directories` = `["/data/local/tmp/s43-pixel-cpugpu-20260925-v1"]`;
- `shard_path` = that dir's `QWEN_PACKED.ffn.gguf`;
- `expected_sha256_by_path` = the six rows above;
- `worker_environment` = the existing nine `S42_PIXEL_*` variables plus:

```
S43_PIXEL_UCLAMP_MIN=1024      # needs root (as_root=true, already the case)
S43_PIXEL_CPU_POLL=100
S43_PIXEL_CPU_BATCH_PAIR=1     # requires S42_PIXEL_CPU_PAIR_DOT=1 and --max-tokens <= 4 (both already true)
```

Everything else stays as it is: `as_root` true, the phone lock, port, columns, quantum, max tokens and
idle-TERM lifecycle.

## 6. What a campaign needs (identity / receipts); not done here

`prepare_campaign_int2.py` builds the Pixel evidence bundle (`PIXEL_EVIDENCE.json`). With the new worker:

- **Software identity changes.**
  - `phone_worker_sha256` -> `sha256:5d824455...`.
  - `worker_environment_sha256` is new (three extra variables).
  - The `phone_library_sha256:<path>` keys change, because the paths change; the hashes are the same.
  - The shard hash is unchanged. The server, its libraries and the OP15 identity are unchanged, so no server
    identity re-materialization is needed.
- **`numerical-rows-1-2-4`.** The script asserts the Pixel agent's suite (arm `04-sdot-pair-dynamic64`) and
  that its `EXPECTED_PHONE_HASHES` contain the worker hash. That fails for the new hash. The replacement
  evidence is `physical/tcp1` plus `physical/d1` / `physical/d3`: outputs byte-identical to the qualified
  worker at rows 1-4. The receipt builder needs a small change to accept this.
- **`COST_CALIBRATION.json`.** This is the one that changes decisions. `kernel_compute_us` came from the old
  worker's server-cadence log (~29 ms). The new server-cadence value is B1 compute p50 6.5 ms, i.e. about
  83 GB/s effective logical. The link receipt (`adb-forward-round-trip`) is still valid but is now
  conservative: non-compute overhead measured 7.1 / 10.8 / 13.7 ms at B1 / B2 / B4, against the calibrated
  11.8 / 16.0 / 20.5 ms. The per-device-policy work should use the new numbers.
- **`server-token-identity`.** This used the old worker's mechanism runs (OP15+Pixel). The FFN outputs are
  byte-identical, so tokens are identical by construction. A fresh receipt needs a short server run; see
  section 7.
- **`scheduler-launched-session`.** The lifecycle code is unchanged. The idle-TERM stop was not re-run with
  the new binary; every run here used finite budgets.

## 7. Caveats and what needs you

- **Phone power is not measured.** It is still the assumed 4.5 W active / 0.875 W idle model. The boost raises
  clocks during calls, and poll spins 5 threads for ~25-65 ms after each call. So the real active power is
  higher than before, while per-call busy time is 3-4x shorter. No per-joule claim is made. The sustained run
  showed no thermal throttling (36 C, USB powered, 100 % battery).
- **DVFS state is uncontrolled.** Clocks are unlocked; the only change is per-process uclamp. Repetitions
  vary by up to ~20 % (CPU-only burst M4: 14.9-18.1 ms), which is why arms are interleaved and repeated. The
  phone-local "prod" cadence (10 ms / 250 ms gaps) is an emulation. The TCP runs use the real adb path, but
  the host side is a synthetic driver, not llama-server.
- **First layer of every token** is still ~15 ms at B1 (vs 6.4 ms for the others). A poll level above 100
  would cover the 250 ms token gap, at the cost of spinning through it; this was not tested.
- **The GPU remains unusable for M>1** on this driver stack. The packed q4_K multi-column pipeline fails to
  compile, and the F16 multi-column GEMV does not reuse weights. A custom tiled multi-row F16 or Q4_K shader
  would be needed before CPU+GPU could be reconsidered.
- **Decisions for you:**
  - Adopt the boosted worker for the next two-phone arms? This needs the identity/receipt work in section 6
    and a fresh cost calibration.
  - Is a short server-level token-identity run acceptable? A Pixel-only server run avoids the OP15; the old
    mechanism receipt used OP15+Pixel.
  - Should the CPU+GPU trailing-split code be removed from the worker, or kept off?

## 8. Files

| path | content |
| --- | --- |
| `BENCH_TABLE.md` / `.json` | Step-1 table, all configs x segments, 3 reps |
| `software/pixel-cpugpu-v1/` | worker source (+ `base/` production source), `CPUGPU_V1.patch`, `pixel-ffn-replay.cpp`, binaries, `SHA256SUMS` |
| `tools/` | `make_suite.py` (phone suite generator), `run_suite.sh`, `stage_phone.sh`, `update_bins.sh` (desktop side, under the rig lock), `analyze_suite.py`, `combine_bench.py`, `qualify_tcp.py`, `analyze_tcp.py` |
| `suites/` | suite configs and generated phone scripts (s0 smoke, d1, d2r1, d2r23, d3, tcp1, tcp2) |
| `physical/` | all raw runs: per-arm worker logs, replay CSVs with per-call clocks, output dumps, battery/thermal/clock snapshots, thread affinity + uclamp, SUMMARY.{md,json}; `tcp1/TCP_TABLE.md` |

Phone dir `/data/local/tmp/s43-pixel-cpugpu-20260925-v1` (1.0 GB) and desktop
`/mnt/storage/s43-pixel-cpugpu-20260925-v1` (59 MB) were created by this work. Both are left in place,
because the phone dir holds the binaries an enabled campaign would use. The Pixel agent's dirs were only
read from.
