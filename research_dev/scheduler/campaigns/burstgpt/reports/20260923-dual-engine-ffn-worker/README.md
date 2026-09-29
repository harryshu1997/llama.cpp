# NPU+GPU dual-engine mode in llama-ffn-split-worker (OP15), 2026-09-23

Contents: code/env summary, build, then **CHARGED** results (current code, authoritative), then the earlier LOW-BATTERY results (per-block GPU layout, kept for the record).

## What changed (examples/layersplit/ffn-split-worker.cpp, diff: ffn-split-worker.dual.diff vs ffn-split-worker.base.cpp)

Opt-in through env vars only (no protocol, CLI or HELLO change; weight_hash identical because it is computed on the unsplit suffix):

| env | meaning |
|---|---|
| `S43_FFN_SECONDARY_BACKEND` | secondary device, e.g. `GPUOpenCL`; unset = mode off (the code path is the old one). `none` = control: dual code path (F32 NPU output + CPU cast) without a secondary |
| `S43_FFN_SECONDARY_FRACTION` | 0..0.5, share of every column block moved to the secondary (the block's trailing columns) |
| `S43_FFN_SECONDARY_ALIGN` | split granularity in columns, default 64 (per block; the effective fraction is logged) |
| `S43_FFN_SECONDARY_MAX_TOKENS` | default 1: requests with tokens > N run NPU-only on an NPU copy of the secondary columns (costs +fraction x weights on the NPU). 0 = always split, no copy |
| `S43_FFN_DUAL_LOG_PERIOD` | per-call `S43DUALFFN` line every N calls (default 1, 0 = off) |
| `S43_FFN_DUAL_WARMUP_ROUNDS` | concurrent full-width tokens=1 dual calls per layer at load (default 3, 0 = off); added with the campaign integration, see [20260923-dual-engine-integration](../20260923-dual-engine-integration/README.md) |

Mechanics: at load every block is split into primary [0, c-s) and secondary [c-s, c) columns; the secondary columns of all blocks of a layer are concatenated in block order into one gate/up/down tensor triple on the secondary backend (disjoint from the NPU copy when MAX_TOKENS=0), so the GPU graph is one gate/up/swiglu/down chain per call; a trailing block selection is a trailing column range (row view for gate/up, strided view for down). (The first version, used for the LOW-BATTERY data below, kept one GPU sub-graph per block.) Per request the secondary graph (own context + gallocr, F32 input converted on the CPU) runs on a persistent helper thread while the main thread runs the NPU graph, whose output stays F32 (no cast node). After both finish the CPU adds the partials and casts to F16 (or F32) into the response; in the DMA-BUF path the merged result is copied into the output buffer inside the existing cpu_start/cpu_end window, so cache sync is unchanged. Works for TCP, direct and staged DMA-BUF paths; runtime column selection (trailing blocks) is honoured by both halves. Log: `S43DUALFFN request layer tokens columns primary_columns secondary_columns primary_us secondary_us wait_us merge_us total_us` and a `[ffn-worker] dual requests=... primary_p50_us secondary_p50_us total_p50_us` line every 32 calls.

## Build

```
docker run --rm -u $(id -u):$(id -g) -v $PWD:/workspace -w /workspace ghcr.io/snapdragon-toolchain/arm64-android:v0.3 bash -c '
cmake -S . -B build-dual-ffn-worker-android -G Ninja -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r28b/build/cmake/android.toolchain.cmake \
 -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_HEXAGON=ON -DGGML_OPENCL=ON \
 -DGGML_OPENCL_EMBED_KERNELS=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENMP=ON -DGGML_NATIVE=OFF -DGGML_CPU_REPACK=ON \
 -DHEXAGON_SDK_ROOT=/opt/hexagon/6.4.0.2 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.4.0.2/tools/HEXAGON_Tools/19.0.04 -DPREBUILT_LIB_DIR=android_aarch64 \
 -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build-dual-ffn-worker-android --target llama-ffn-split-worker htp-v81 -j 24'
```
Phone files: bin/libggml{,-base,-cpu,-hexagon,-opencl}.so, ggml/src/ggml-hexagon/libggml-htp-v81.so, llama-ffn-split-worker, plus the NDK libomp.so (`/opt/android-ndk-r28b/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/19/lib/linux/aarch64/libomp.so`; /odm's libomp is not mappable from shell). Driver: `ffn_dual_driver.cpp` (NDK clang++ -static-libstdc++, `-Iexamples/layersplit`).

## CHARGED (battery 80 % held by the OPLUS limit, 4.18-4.23 V, 28-32 C, notify_code 0 before/after every config)

Same setup as below (4 Qwen layers streamed, 8704 cols / quantum 2176, fresh worker per config, 80 timed calls, tokens=1), orders rotated per rep. Per-config battery before/after, 0.2 s (effectively ~0.27 s) GPU clock / DDR+LLCC bus_dcvs votes via su are in raw*/c_*/battery_*.txt, clocks_*.txt; aggregates in raw*/charged_summary.md and charged_runs.json (`analyze_charged.py`). Level stayed at 80 % for all 88 configs (voltage drifts 4.23 -> 4.18 V).

### 1. Old per-block GPU layout (raw/c_r1..c_r4, 4 rotated reps), per-layer p50 ms

| fraction | p50 per rep | mean | speedup (paired) | NPU / GPU leg ms |
|---|---|---|---|---|
| off | 5.08, 5.03, 5.00, 4.98 | 5.02 | 1 | - |
| none | 5.16, 5.08, 5.16, 5.09 | 5.12 | 0.97-0.99 | 5.0-5.1 / - |
| 0.1 | 4.49, 4.43, 4.37, 4.40 | 4.42 | 1.13-1.14 | 4.3-4.4 / 3.3-3.5 |
| 0.15 | 4.86, **4.15**, 4.71, 4.80 | 4.63 | 1.04-1.21 | 4.4 / 4.4-4.5 (fast run 4.1 / 3.7) |
| 0.2 | 4.78, 4.77, 4.83, 4.76 | 4.78 | 1.04-1.06 | 4.0-4.1 / 4.5 |
| 0.25 | 4.80, 4.34, 4.87, 4.86 | 4.72 | 1.02-1.16 | 3.7-3.9 / 4.3-4.6 |
| 0.3 | 4.83, 4.35, 4.83, 4.85 | 4.72 | 1.03-1.16 | 3.5-3.7 / 4.3-4.6 |

So charging did NOT make 1.22x at 0.15 consistent: 1 of 4 processes. Only 0.1 was consistent (1.13-1.14x). A longer-window diagnostic (raw_diag, 400 calls, 3 reps) gave 0.15 -> 1.00x in 3/3, 0.1 -> 1.12-1.13x.

### What differs in the no-gain processes

- GPU clock: 1200 MHz (max) in every dual run, 222 MHz idle in off/none -> not a GPU DVFS issue.
- CPU-side DDR vote (bus_dcvs/DDR cur_freq, and the bwmon/memlat voters) sits at the floor 547 MHz nearly always; it was higher (median 1867 MHz, LLCC 544 MHz) in exactly the two fast r2 runs (0.15, 0.25) but the sampling is too coarse (~3 samples per 0.7 s window) to call it the cause, and the NPU/GPU vote through their own ICC paths is not visible there. No DDR bwmon counters were readable beyond these votes.
- NPU timing: the NPU leg is 4.4 ms (slow) vs 4.1 ms (fast) at 0.15; the GPU leg is the decisive difference: 4.4-4.6 ms (slow) vs 3.7 ms (fast), and with the per-block layout it is ~4.3-4.6 ms almost regardless of its column count (1280..2560) -> GPU work is not bandwidth-proportional; it is ~20 small kernels per call (5 per block x 4 blocks) that get starved while the NPU streams, then finish after the NPU.
- Test: one block per layer (quantum 8704, raw_q1, 3 reps): off 4.93, 0.1 4.26 (1.15-1.16x), 0.15 4.29 (1.14-1.16x), 0.2-0.3 4.6 (1.05-1.09x), spread within reps < 0.05 ms -> bimodality gone. Fix implemented: fused GPU layout (see Mechanics).

### 2. Fused GPU layout (current code; raw_fused/c_fused_r1..r4 + partial), per-layer p50 ms

| fraction (eff.) | p50 per rep | mean | p10-p90 (typ.) | speedup (paired) | NPU leg | GPU leg | merge | NPU / GPU / total GB/s |
|---|---|---|---|---|---|---|---|---|
| off | 5.15, 5.07, 5.06, 4.99 | 5.07 | 4.94-5.17 | 1 | 5.07 | - | - | 52.8 / - / 52.8 |
| none | 5.11, 5.02, 5.04, 5.10 | 5.07 | 4.94-5.19 | 0.98-1.01 | 5.0 | - | 17-36 us | 53 / - / 53 |
| 0.1 (8.8 %) | 4.53, 4.29, 4.32, 4.32 | 4.36 | 4.24-4.36 | 1.14-1.18 | 4.27 | 2.34 | 18 us | 57 / 10 / 62 |
| **0.15 (14.7 %)** | 4.26, 4.07, 4.07, 4.06 | **4.11** | 4.01-4.13 | **1.21-1.24** | 4.03 | 3.33 | 17 us | 56.5 / 11.8 / 65.7 |
| 0.2 (20.6 %) | 4.65, 4.17, 4.18, 4.16 | 4.29 | 4.12-4.36 | 1.11-1.21 | 3.96 | 4.07 | 18 us | 53.6 / 13.5 / 64.1 |
| 0.25 (26.5 %) | 4.21, 4.18, 4.18, 4.19 | 4.19 | 4.15-4.42 | 1.19-1.22 | 3.70 | 4.10 | 18 us | 53.3 / 17.3 / 63.9 |

Partial runtime columns (6528 = 3 of 4 blocks, strided GPU down view): off 3.78 -> 0.15 3.12 ms (1.21x), rel L2 1.8e-4.
The only weaker values are the first dual configs of rep 1 (0.1/0.15/0.2 at 4.26-4.65 ms), i.e. the first minute of GPU use; every other process is within +-0.02 ms.

tokens=4 (MAX_TOKENS default 1, NPU fallback with copy columns; raw/c_t4_r1..r2): off 5.81 / 5.79 ms, 0.15 6.75 / 6.80 ms (0.85-0.86x) -> multi-token calls pay ~16 % for the extra NPU sub-graphs; unchanged by the fused layout.

Correctness (charged, fused): rel L2 vs NPU-only 1.36e-4 (0.1), 1.61e-4 (0.15), 1.82e-4 (0.2), 1.94e-4 (0.25); none control bit-exact; T4 fallback 7.7e-6. Host CPU+CPU check of the fused layout incl. partial selections: rel L2 <= 5.6e-6.

### CHARGED verdict

With the fused GPU layout, f = 0.15 is a consistent 1.21-1.24x per layer at tokens=1 (5.07 -> 4.07-4.11 ms for 50 % of Qwen3-14B columns on OP15; combined ~66 GB/s vs 53 GB/s NPU-only), 0.25 is a flatter alternative (1.19-1.22x); 0.1 is the safe low setting (1.14-1.18x). Keep tokens >= 2 NPU-only (default). Remaining costs: +14.7 % NPU weight memory for the NPU copy used by multi-token calls (set MAX_TOKENS=0 to avoid the copy, at the price of slow GPU multi-token calls), and multi-token calls ~16 % slower. The first minute after start may be slower (warm-up), so a warmup of a few dual calls at load would be worth adding before integration.

---

# LOW-BATTERY runs (earlier, per-block GPU layout)

## Test setup

OP15 standalone TCP worker, Qwen3-14B F16 (`/data/local/tmp/s41-opoffload-dmabuf-v1/Qwen3-14B-Q4KM-dequant-f16.gguf`), layers 0-3, columns 8704 of 17408 (50 %), quantum 2176 (4 blocks/layer, 1020 MiB NPU weights), f16 io, env `GGML_HEXAGON_MBUF=4192 NHVX=4 NDEV=1 VMEM=3328 S41_DISABLE_GRAPH_CACHE=1`, `LD_LIBRARY_PATH=ADSP_LIBRARY_PATH=<dir>`. `ffn_dual_driver` on the phone (127.0.0.1) streams layers 0,1,2,3 round-robin (267 MB/layer, never cache resident), 2 warmup + 20 iterations = 80 timed calls, identical deterministic inputs per (iter, layer) for every config. Fresh worker per config. Scripts: phone_sweep.sh, run_all*.sh; raw per-config CSV/logs/summary.md in raw/. Latency = worker compute_us (graph build + both engines + merge), per layer call.

## Results, tokens=1 (per-layer ms, p50 of 80 calls; off = NPU only)

Effective secondary share (align 64 per 2176 block): 0.05->5.9 %, 0.1->8.8 %, 0.15->14.7 %, 0.2->20.6 %, 0.25->26.5 %, 0.3->29.4 %, 0.4->41 %.

Phase A (battery 8-9 %, earlier runs: sweep, sweep2a/b/c, sweep3a/b):

| config | off | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 |
|---|---|---|---|---|---|---|---|
| sweep  | 5.00 | | 4.41 | | 4.75 | 4.32 | 4.94 |
| sweep2a | 5.02 | 4.37 | 4.43 | 4.67 | 4.27 | 4.80 | |
| sweep2b | 5.04 | 4.54 | 4.41 | **4.10** | 4.27 | 4.74 | |
| sweep2c | 5.03 | | | **4.09** | | | |
| sweep3a | 5.01 | | 4.42 | **4.10** | 4.71 | | |
| sweep3b | 4.97 | | 4.99 | 5.06 | 4.25 | | |

Control `none` (dual code path, no GPU): 5.08 / 5.06 ms vs off 5.01 / 4.97 -> the F32-output + CPU cast changes nothing; gains are from the GPU.

Phase B (battery 5 %, discharging on USB; sweep4_r1-4, sweep5a/b/c):

| config | off | 0.15 | 0.2 | 0.25 |
|---|---|---|---|---|
| sweep4 r1..r4 | 5.00-5.04 | 5.36-5.44 | 5.16-5.59 | |
| sweep5a | 4.92 | 5.40 | 5.62 | 5.66 |
| sweep5b | 5.02 | 5.33 | 5.16 | 5.63 |
| sweep5c (MAX_TOKENS=0) | 4.97 | 4.94 | | |

Breakdown (typical): good state f=0.15: primary 4.05 ms (85 % cols), secondary 3.69 ms, wait 0, merge 17-36 us -> 1.22x. Bad state f=0.15: primary 5.2 ms (NPU slower on 85 % of the columns than on 100 % alone), secondary 5.0 ms. Combined DRAM bandwidth: good state ~65 GB/s (NPU 55 + GPU 10.6) vs NPU-only 53 GB/s; bad state ~52 GB/s total, i.e. the GPU just steals NPU bandwidth. The GPU leg alone is ~1 ms for 1792 columns (bench agent's gpu_solo), but 3.5-5 ms when concurrent.

## tokens=4

GPU secondary is 2-3x slower than the whole NPU call (Adreno f16 x f32 M=4 kernel): T4 off 5.78 ms vs split 12.8-18.6 ms (sweep). Hence the default gate `S43_FFN_SECONDARY_MAX_TOKENS=1`. The NPU fallback with the copy columns as extra sub-blocks costs 6.70/7.88 ms vs off 5.72/6.72 ms (+17 %), because each block becomes two smaller NPU sub-graphs (same bytes, more ops). T4 off itself moved 5.72 -> 6.72 ms between phases.

## Correctness (vs NPU-only, same inputs, all 80 calls x 4 layers)

rel L2 1.2e-4 (f 0.05) .. 2.2e-4 (f 0.4), max abs 0.0625-0.125 on outputs with max |y| = 788 (F16 output rounding + GPU accumulation order). T4 NPU-only fallback rel L2 7.7e-6. `none` control bit-exact. Host CPU+CPU check (tiny synthetic Qwen GGUF, partial runtime columns, tokens 1-4): rel L2 <= 6e-6, weight_hash unchanged.

## Verdict (low battery, per-block layout; superseded by CHARGED)

- Mode works and is numerically fine. Best fraction 0.15 (effective 14.7 %): per-layer 5.0 -> 4.1 ms (1.22x) at 50 % columns, matching the bench agent (1.25x at secondary 0.15-0.25). 0.2 is close (4.25-4.27 ms, 1.17x) and was the more stable of the two in phase A.
- Not robust: the win appears only in some worker processes/phone states (sweep3b 0.1/0.15 gave nothing; all of phase B, at battery 5 %, lost 0-13 %). The mechanism is shared DRAM bandwidth; when the SoC does not grant more than the NPU-solo bandwidth, the split is a loss. I would not enable it in production without a runtime guard (e.g. compare S43DUALFFN total_us against an NPU-only probe and fall back) and a re-check on a charged phone.
- Keep tokens>=2 NPU-only (default). With MAX_TOKENS=1 the NPU holds a copy of the secondary columns (+15 % NPU memory at f=0.15) and multi-token calls pay ~17 %; MAX_TOKENS=0 avoids the copy but multi-token calls then go to the slow GPU path.

## Caveats

- Only the TCP path was run on the phone; DMA-BUF (FunctionFS) direct/staged paths compile and share run_dual but were not exercised physically. Resident workers exec the same binary, so the env passes through.
- Battery was at 5-9 % and discharging for all runs; DVFS/thermal state is uncontrolled and bimodal per process. No energy measured.
- Single model/shape (Qwen3-14B, 4 layers, 8704 cols, quantum 2176). With quantum 512 blocks the per-block alignment makes fractions coarser (64/512 = 12.5 % steps) and adds GPU kernel launches.
- Split granularity 64 columns; HTP F16 matmul and Adreno kernels had no alignment requirement beyond that.
