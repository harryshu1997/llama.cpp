# Pixel 10 Pro GPU GEMV over OpenCL (2026-10-27)

Device: PowerVR D-Series DXT-48-1536, OpenCL 3.0 (driver 25.1@6794074), 6 CUs at 1094 MHz, cl_khr_fp16, cl_khr_integer_dot_product,
images (RGBA32UI), image2d_from_buffer, 256 KB local, 1 GB __constant. Loader: /vendor/lib64/libOpenCL.so (dlopen from a shell binary).
Toolchain: docker snapdragon-toolchain-hostgcc:v0.3, NDK r28b clang++ (CL headers from the NDK sysroot). Kernels compile on the phone.

## Kernels (clgemv.cl)
- `f16_rows`: F16 row-major weights, f32 activations in __constant, L=WG/R lanes per row with 16 B loads, local reduction.
- `q4_global`: Q4 (32-weight blocks, f16 scale) in a tile layout [row-group g][block b][R rows] x 16 B so a work-group's loads are contiguous;
  activations int8 per block (Q8 style) in __constant; `dot_acc_sat` packed int8 dot products; `q4_image` = same via a 2D RGBA32UI image;
  `q4_global_interleaved` = scales interleaved; `q4_global_m` = M activation rows share the weight stream.

## Qwen FFN shapes, device time per matrix (median of 30), GPU at 1094 MHz, R=8 WG=128 unless noted
| kernel | gate/up 17408x5120 | GB/s | down 5120x17408 | GB/s | error vs quantized reference |
|---|---|---|---|---|---|
| f16_rows | 6.33 ms | 28.2 | 7.01 ms | 25.4 | 3.8e-4 (vs exact) |
| q4_global | 1.37 ms | 36.6 | 1.40 ms | 35.8 | 1.0e-7 |
| q4_image | 1.37 ms | 36.6 | 1.39 ms | 36.2 | 1.0e-7 |
| q4_global_interleaved | 1.46 ms | 34.3 | 1.46 ms | 34.4 | 1.0e-7 |
| q4_global R=32 WG=256 | 1.29 ms | 38.7 | - | - | 1.1e-7 |

Sweep (gate/up): R=4/128 34.1, R=8/128 35.5, R=16/128 37.8, R=8/64 37.2, R=8/256 36.3, R=16/256 35.6, R=32/256 38.7 GB/s.
Batch rows (q4_global_m, R=32 WG=256): M=1 1.29 ms, M=2 1.92 ms, M=4 3.09 ms per matrix (weights read once; per-row 52 / 65 GB/s-equivalent).

## Per-layer wall (cllat: gate + up + down, 3 dispatches)
| condition | per layer with clFinish | with event wait | streamed (40 layers, one finish) | write 5 KB + run + read 20 KB | tiny kernel round trip |
|---|---|---|---|---|---|
| process unpinned | 16.6 ms | 17.7 | 5.1 | 18.4 | 1.1 ms |
| process pinned to core 7 (`taskset 80`) | **4.6 ms** | 4.5 | 4.4 | **5.1** | 0.44 ms |
| pinned + GPU min_freq 1094 MHz | 4.6 ms | 4.6 | 4.35 | 4.85 | 0.40 ms |
The 16 ms was the OpenCL submission path running on a little core (same disease as the CPU worker's DVFS fix); the GPU clock was already at 1094 MHz.

## Power and thermal
Battery + USB sysfs at 5 Hz: idle 0.65 W; sustained q4_global 5.8 W (busy-window mean); sustained f16_rows 3.9 W.
Energy per Qwen FFN layer on the GPU ~ 5.8 W x 3.9 ms = 23 mJ (20 mJ above idle) vs the CPU packed path ~ 3.9 W x 6.25 = 24 mJ (estimated from the 2.2 W campaign average at 47 % duty).
60 s continuous at 100 % duty: GPU stays at 1094 MHz, battery 25.4 -> 27.0 C, no throttling.

## Comparison with the production Pixel path (CPU packed int8 kernel, 20260925-pixel-cpu-gpu)
| | CPU packed (production) | GPU OpenCL q4 (this) |
|---|---|---|
| per Qwen FFN layer, M=1 | 6.0-6.25 ms compute | 3.9 ms device, 4.6 ms wall incl. submit + sync, 5.1 with activation write and result read |
| M=2 / M=4 | 8.0 / 17.2 ms | 5.8 / 9.3 ms |
| effective weight bandwidth | 26 GB/s | 38.7 GB/s |
| old Vulkan GPU path (ggml) | - | 19.8 ms F16 or Q4_K (compute-bound shader) |

## What integration needs
Q4_K-layout kernel (the Pixel shards are Q4_K/Q6_K with 6-bit sub-block scales; this kernel assumes Q4_0-style 32-blocks with an f16 scale) or a re-quantized shard with a quality check; the worker's GPU backend replaced by an OpenCL path owning whole layers with the submitting thread pinned to a big core and all three dispatches batched per call; token-identity receipts and evidence re-qualification; phone power model update (5.8 W busy).

## Bandwidth ceiling study (same day, later)
- Raw read ceiling of the GPU under OpenCL (`stream_read`, uint4 xor): **46.1 GB/s** device time. The Q4 kernel is therefore at 84-90 % of the port.
- Pinning gmc_freq (4224 MHz), memss_freq (932), fabhbw_freq (711) and the GPU (1094) to their maxima with root: no change (46.0 / 38.8 GB/s);
  the devfreq cur_freq nodes read 409/178/172 MHz during load, so they are vote proxies, not the granted clocks. Restored afterwards.
- `q4_v2` (unsigned nibbles with the -8 folded into a per-block activation sum, 16 B-aligned in-stream scales, next-block loads issued before the
  current block's math): gate/up 17408x5120 **1.21 ms = 41.4 GB/s** (R=32 WG=256; R=16/128 40.1); down 5120x17408 1.52-1.59 ms (31.5-32.9),
  worse than `q4_global` there. Best down: `q4_global` R=16 WG=128 **1.38 ms = 36.3 GB/s** (R=8/128 1.39, R=4/128 1.48, R=8/64 1.52).
  Best Qwen FFN layer = 2 x 1.21 + 1.38 = **3.80 ms device** (was 3.9).
- Concurrent CPU + GPU (cpustream.cpp NEON xor on cores 6..2 + the GPU loop on core 7): CPU alone 26 (1 thr) / 32 GB/s (3-6 thr);
  concurrent GPU 30.4 / 26.6 / 26.1 + CPU 15.7 / 20.1 / 20.4 = **aggregate 46.0-46.8 GB/s** for 1/3/5 CPU threads, identical with pinned
  clocks. The SoC delivers ~46-47 GB/s in total to any mix of consumers; a CPU+GPU column split could take a layer from 3.8 to ~3.2 ms
  (+13 %) at roughly 9-10 W, which is not worth its interference and thermal cost.
- Verdict: 41 GB/s on the wide projections and 36 on the down projection is within 10-20 % of the Pixel's physical ceiling; "50 GB/s" is
  not available on this SoC (theoretical 67.6 GB/s at 8,448 MT/s x 64 bit, 69 % realised).

## "DMA-like" read paths on the PowerVR (same day)
| path | GB/s | note |
|---|---|---|
| plain compute loads (`stream_read`, uint4) | 46.1 | the fast path |
| `async_work_group_copy` into local memory, 16 KB chunks double-buffered (`stream_async`) | 3.2-4.9 | 10x slower: the driver runs it as a slow per-work-group copy |
| `clEnqueueCopyBuffer` (driver copy engine) | 0.54 read, 1.08 read+write | a host-side memcpy or worse |
Verdict: the GPU has no DMA path that beats its own load units; the HTP's DMA advantage on the OnePlus is a different IP block, and the Pixel SoC's
46-47 GB/s cap applies to every consumer anyway.

## Instruction and command study (same day): what else the driver offers, measured on 17408x5120 (R=32 WG=256 unless noted)
| item | result | keep? |
|---|---|---|
| driver dot-product acceleration flags (clGetDeviceInfo 0x1074/0x1075) | only unsigned x unsigned reported accelerated (vector and packed); signed and mixed "not accelerated" | informational: the signed/mixed vector forms still run at full speed below |
| `dot_acc_sat(char4,char4)` signed vector (q4_global) | 1.29 ms, 38.9 GB/s | baseline |
| `dot_acc_sat(uchar4,char4)` mixed + per-block activation sum (q4_v2) | **1.21 ms, 41.6 GB/s** | best for the wide shapes |
| packed `dot_acc_sat_4x8packed_uu_uint` with precomputed nibble sums (q4_uu, 4 B more per row-block) | 1.32 ms (42.3 GB/s on 11 % more bytes) | no: slower in time |
| packed `_ss_int` with a per-byte subtract (q4_dotpacked) | 1.17 ms but WRONG (byte borrow when a nibble < 8) | no |
| no dot: short16 multiply-add (q4_nodot) | 1.87 ms, 26.9 GB/s | no: the dot instructions are worth 1.55x |
| half-precision math (q4_half) | 1.93 ms, 26.1 GB/s | no |
| `sub_group_reduce_add` with lanes-of-a-row contiguous (q4_subgroup) | 5.0-5.3 ms, 9-10 GB/s (loads become strided across rows) | no |
| weight buffer CL_MEM_COPY_HOST_PTR vs CL_MEM_USE_HOST_PTR | 46.1 / 45.9 GB/s stream, 41.6 / 41.5 kernel | either (USE_HOST_PTR = zero-copy from the worker's mmap) |
| weight buffer CL_MEM_ALLOC_HOST_PTR | 10.6 GB/s stream, 10.5 kernel | NEVER for weights (5x slower) |
| queue priority hint (cl_khr_priority_hints high/low) | single-dispatch round trip 1.73-1.89 ms either way | no effect |
| `-cl-fast-relaxed-math`, `-cl-unsafe-math-optimizations` | identical; `-cl-opt-disable` -> clCreateKernel fails | none |
| SVM | coarse-grain buffer only (caps=1): no fine-grained hand-off | map/unmap or write/read stays |
| `async_work_group_copy` / `clEnqueueCopyBuffer` | 3-5 GB/s / 0.54 GB/s | no |
Best per shape remains: gate/up q4_v2 R=32/256 1.21 ms; down q4_global R=16/128 1.38-1.40 ms -> layer 3.8 ms device.

## Other precisions (same day): per Qwen FFN matrix on the Pixel GPU, exact against each format's quantized reference
| format | bits/weight | gate/up 17408x5120 | down 5120x17408 | layer (2 gate/up + down) | kernel notes |
|---|---|---|---|---|---|
| F16 (`f16_rows`) | 16 | 6.28 ms, 28.4 GB/s | 7.0 ms, 25.4 | 19.6 ms | convert + fma; 535 MB per layer |
| Q8_0 (`q8_tile`) | 8.5 | 2.18 ms, 43.5 GB/s | 2.27 ms, 41.8 | 6.6 ms | int8 dots, no unpack: bandwidth-bound at the ceiling |
| Q4_K, the shard format (`q4k_tile2`) | 4.5 | 1.34 ms, 37.3 GB/s (R=16/128) | 1.46 ms, 34.4 | 4.1 ms | super-block 256, 6-bit sub-scales/mins, ggml packing; nibbles interleaved in 16 B pieces across rows, meta as one uint4 |
| Q4_K, naive row layout (`q4k_tile`) | 4.5 | 2.07 ms, 24.2 GB/s | 2.41 ms, 20.8 | 6.6 ms | 128 B per row contiguous: uncoalesced per instruction, 12 byte loads for scales |
| Q4_0 (`q4_v2` / `q4_global`) | 4.5 | 1.21 ms, 41.6 GB/s | 1.38 ms, 36.3 | 3.8 ms | reference point |
Reading: the real shard format costs 8 % more than Q4_0 on the GPU once its nibbles are interleaved like the Q4_0 tiles; the first
attempt with 128 B per row was 1.6x slower, so the layout matters more than the format. Q8_0 reaches the ceiling (43.5 GB/s) because
there is no unpacking. Q6_K (the shard's other tensor type, 6.56 bits) is not implemented; at Q4_K efficiency it would be ~5.5-6 ms per layer.
Comparison with the CPU packed path (Q4_K/Q6_K shard, 6.0-6.25 ms per layer): GPU Q4_K 4.1 ms = 1.5x; the integration can keep the shard as is.

## Low-level study: what sets the Pixel GPU's 47 GB/s (same day, rooted phone)
Clock domains (devfreq + /sys/kernel/debug/google_pm_qos + opp): `gmc_freq` / `irm_gmc_freq` (DRAM controller, OPPs 409..4224 MHz), `memss_freq`
(memory subsystem, 178..932 MHz), `fabhbw_freq` (high-bandwidth fabric, 172..711 MHz), `34f00000.gpu0` (198..1094 MHz). The devfreq
`cur_freq`/`trans_stat` of gmc_freq only reflect that node's own vote (powersave governor, 409 MHz forever); the DRAM clock is the aggregate of
PM QoS votes: `gs_memlat_governor` (CPU memory latency) votes 2688 MHz at idle, 672 MHz during a GPU-only stream (no CPU misses) and 3686 MHz
under a CPU stream; no GPU vote appears in the lists or in the `atrace freq` clock_set_rate events (only cpuNgmc/cpuNdsu/dsulat/lb_adpf).

| experiment | GPU stream (uint16 loads, WG 512) | reading |
|---|---|---|
| governor clocks | 47.0-47.4 GB/s | baseline |
| gmc/memss/fabhbw/GPU min pinned to max | 47.2 | the pins apply (trans_stat shows the node at 4224) but nothing improves |
| pointer-chase on core 6 (0.3 GB/s, raises the CPU vote to 3686 MHz) | 46.8 | a faster DRAM vote does not help the GPU |
| CPU stream on core 6 at the same time | GPU 33.6 + CPU 20.1 = **53.7 aggregate** | the SoC delivers > 47; the 47 is the GPU's path |
| fabhbw capped 480 / 343 MHz | 46.9 / 41.1 | fabric binds only below ~400 MHz |
| **memss capped 233 / 466 / 666 / 800 / 932 MHz** | **14.5 / 28.6 / 39.6 / 44.9 / 47.3** | linear in the memory-subsystem clock: the GPU's path is 50.8 GB/s per GHz of memss, 47 at its 932 MHz maximum |
| gmc devfreq capped 1536 MHz | 22.0 | the DRAM clock only binds when forced far below the GPU's need |
| CPU stream (6 thr) with memss 466 | 31.6 -> 25.9 | the CPU shares the memss path but is core-limited at 32 |
| wider loads: uint4 / uint16 / uint16 WG 512 | 45.8 / 46.7 / 47.4 | +3 % from more bytes in flight |
| cl_img allocation flags: uncached-CPU, cached-CPU, GPU-local | 46.6 / 46.6 / 46.6 (kernel 41.6 / 41.6 / 41.4) | no effect; GPU-cached / write-combine / CPU-local / relaxed are rejected (-64) |
| `async_work_group_copy` / `clEnqueueCopyBuffer` | 3-5 / 0.5 | no DMA path |
Conclusion: the GPU's streaming ceiling is the memory-subsystem clock (memss, 932 MHz max) times the GPU's share of that path, 47 GB/s
(~79 % of 64 B/cycle). It is not DRAM speed (the SoC gives 54 GB/s to CPU+GPU), not the fabric clock, not allocation or access width,
and no vote or pin changes it. The whole-SoC ceiling is ~54 GB/s with both engines streaming.

## Pre-warming (same day): no cache to warm, but the clocks must be
- Cache sweep (`cache_sweep`: re-read a 0.5-44 MB prefix 60x back-to-back): never above 47 GB/s at any size; 0.5-4 MB prefixes read at
  9-15 GB/s because such dispatches are too small to fill the GPU. No GPU-side cache retains weights between dispatches -> nothing to pre-load.
- Production cadence (`cllat ... 2.5 150 N`: 23 layer calls per token, 2.5 ms host gap between calls, 150 ms between tokens, submitter on core 7):

| clocks | layer wall ms (med / p90) | first layer of a token |
|---|---|---|
| governor (GPU sits at 396 MHz 60 % of the time) | 14.8 / 15.6 | 10.2 |
| GPU min pinned 1094 MHz | 5.8 / 6.2 | 8.9 |
| memss min pinned 932 MHz only | 14.6 / 15.3 | 9.3 |
| both pinned | 5.35 / 5.49 | 8.0 |
| both + keep-alive dispatch (1 work-group) every 20 ms in the token gap | 5.33 / 5.44 | 5.2 |
| both + keep-alive every 50 ms | 5.33 / 5.45 | 4.8 |
  Same at 8 ms gaps (Gemma-like cadence): 15.4 -> 5.4 ms. Idle power with the pins and no work: 0.64 W vs 0.66 W (none).
  PowerVR module parameters (AutoVzGPUPowerdown, EnableRDPowerIsland...) are read-only (444), so the wake-up after an idle gap is handled
  by the keep-alive dispatch, not by the driver.
- Integration rules: (1) hold the GPU devfreq min at 1094 MHz for the session (root; free at idle), memss optional (+8 %); (2) submit from a
  big core; (3) during gaps longer than ~20 ms issue a one-work-group dispatch every 20-50 ms. Per Qwen token on the Pixel: 23 x 5.3 ms =
  122 ms of GPU chain vs 23 x 6.25 = 144 ms CPU packed (hand-offs equal), and the GPU batch advantage on top (M=4: 9.3 vs 17.2 ms per layer).

## Worker integration (same day, later): the kernels inside the FFN worker against the production CPU path

Worker source `reports/20260922-fast-path-M3/software/pixel10pro-opencl-v1/` (`pixel_opencl_ffn.h`, `pixel_opencl_kernels.cl`,
`BUILD.sh`; env `S43_PIXEL_OPENCL=1` on top of the packed-CPU env). Staged on the Pixel at `/data/local/tmp/s43-pixel-opencl-20261027-v1/`
(`single/` = single-dot variant). A/B over TCP loopback on layers 18-23 of the real 1-23 shard with `pixel-ffn-replay` (recorded
activations, 5 segments: 1/2/4 rows, burst and production cadence = 2.5 ms between calls, 150 ms between tokens); scripts and the
per-call CSVs are in `worker-ab/`. Mean / p90 compute_us per call, ms:

| path | 1 row burst | 1 row prod | 2 rows burst | 4 rows burst | 4 rows prod | rel-L2 vs f32 | vs CPU path |
|---|---|---|---|---|---|---|---|
| production CPU packed int8 (reference) | 5.83 / 6.43 | 7.66 / 14.56 | 7.18 / 7.98 | 13.71 / 15.14 | 14.87 / 18.07 | 4e-4 | - |
| GPU, first build (as integrated) | 6.6 / 7.6 | 6.8 / 7.9 | 8.2 / 9.7 | 12.8 / 14.8 | 13.0 / 14.7 | 1.1e-2 | 1.1e-2 |
| GPU single-dot + CPU floor + keep-alive 1 ms | 5.34 / 5.91 | 5.51 / 6.26 | 7.37 / 8.78 | 12.14 / 13.32 | 11.84 / 13.34 | 1.1e-2 | 1.1e-2 (fails the 1e-3 receipt) |
| GPU residual two-dot + CPU floor + keep-alive 1 ms | 6.60 / 7.37 | 6.70 / 7.31 | 11.29 / 12.47 | 20.55 / 21.83 | 20.00 / 21.32 | 4e-4 | 2e-4 (passes) |

What was found on the way (each a separate measurement, all in this directory's scripts):
- The worker's kernel body is the microbench kernel (1.35 ms per matrix in the harness, `worker_kernels_test.cl`), but inside the
  worker a gate+up pair took 4.5 ms instead of 2.7. Excluded one by one: thread placement (whole process on core 7), kernel
  arithmetic, weight buffer flags (USE_HOST_PTR -> COPY_HOST_PTR, host copy freed: RSS 1.95 -> 1.0 GB for 6 layers), input
  transfer mode (write / none / blocking / map), read-only activation buffers, zero global offset, keep-alive queue, per-role kernel
  objects, profiling queue, 18 x 50 MB resident buffers, the ggml Vulkan instance (stub `libggml-vulkan.so`), root vs shell,
  max_tokens buffer sizes, pool threads, uclamp.
- Cause 1: the submitting core (7) sits at its 533 MHz idle clock at the worker's call cadence (sampled), and the driver's CPU-side
  work per launch is ~0.9 ms there vs ~0.3 ms at >= 2 GHz. `sched_setattr` uclamp.min=1024 is accepted but ignored by sched_pixel;
  `policy7/scaling_min_freq` works (sweep 533 -> 1036 -> 2073 -> 3168 MHz: 5.02 / 4.78 / 4.66 / 4.62 ms per layer in `cllaunch`).
  The worker now floors it (`S43_PIXEL_CL_CPU_MIN_KHZ`, default 2073000, restored at exit).
- Cause 2: at the call cadence the GPU idles ~1.5 ms between calls and pays a wake-up on the next launch; keep-alive every 1 ms
  (instead of 20) removes it (-0.7 ms per call). Idle power with the 1 ms keep-alive not yet measured.
- At max CPU clock launches pipeline (gate,up,silu,down + finish = device sum + 0.3 ms), so kernel fusion would gain little.
- Accuracy: int8 blocks of 32 give 1.1e-2 rel-L2 vs f32 (the same class as the desktop's CUDA Q8_1 activations); the CPU packed path
  reaches 4e-4 with its residual second dot. The same residual on the GPU (x ~= xs*q + xsl*ql, second 8-bit dot per weight nibble,
  bsum/bsum16 float side terms) reaches 4e-4 vs f32 / 2e-4 vs the CPU path, but the kernel becomes ALU-bound: 2.0 ms per matrix
  batched (25 GB/s), +25 % at 1 row and +50-60 % at 2-4 rows (M=4 does 16 dots per 16 B piece).
- Numerics reference of the receipt (`pixel_experiment_suite.py`): the Pixel CPU worker itself (`reference-cpu`), tolerance 1e-3.

Verdict: the GPU route beats the production CPU path only without the residual (1 row -8 % burst / -28 % at production cadence,
4 rows -12 %, 2 rows parity, p90 at production cadence 6.3 vs 14.6 ms), which needs the numerics receipt relaxed to the desktop's own
int8-activation class (~1.1e-2). Gate-compliant, the residual variant is -13 % at 1 row at production cadence and +50 % at 2-4 rows,
i.e. usable only as a hybrid (GPU for 1-row calls, packed CPU otherwise) with an expected per-token effect inside run-to-run noise.

## Qualification of the int8 route (2026-10-28)

Decision: the int8-activation variant is the production path. Its activation rounding was aligned with ggml's
`quantize_row_q8_0` (blocks of 32, scale amax/127 stored as f16, values rounded half to even), on the host for x and in the
`silu_mul_quant` kernel for h. Against the official receipt (`native/ffn_numerics.py`: Q8_0-rounded activations, float64 products
with the f16 parent's weights) the worst rel-L2 over 23 layers x rows 1/2/4 is 2.36e-3 (f32 distance 3.2e-2 worst, 1.1e-2 typical).
`worker-ab/dump_check.py` locates the 2.3e-3: x codes identical, gate/up 2.1e-4 apart (f16-rounded parent weights vs exact Q4_K),
h 3e-4 apart, then the reference re-quantises h to int8 and 28 % of the codes land one step apart. Receipt policy
`int8-activation-class` (rel-L2 <= 5e-3, recorded in the receipt); server token identity PASS (4 x 64 tokens, 2852 calls).
Evidence bundle, config, receipts and the run scripts are copied to `evidence/`.

## Trace result (2026-10-28, 1x fleet trace, `trace-g16/`)

Same-night A/B on the new desktop, identical derived placement (only the Pixel worker block differs), full OnePlus Gemma
assistance in both arms (46-49 k Gemma calls), Qwen phone calls identical (33.7-34.4 k):

| arm | host kJ | vs desktop arm g161nph | fleet kJ (phones metered) | Pixel kJ metered | Qwen TPOT ms | E2E mean s | queue wait s |
|---|---|---|---|---|---|---|---|
| g16c1new CPU-Pixel (production worker) | 159.1 | -8.4 % | 170.3 | 3.9 (2.30 W) | 331 | 124 | 418 |
| g16o1new OpenCL-Pixel (int8 kernels) | 158.3 | -8.8 % | 168.1 | 2.7 (1.56 W) | 311 | 119 | 394 |

The GPU route leaves the desktop-side saving where it was (-0.8 kJ host, within noise), cuts the Pixel's own metered energy by
31 % (shorter busy time outweighs the higher busy power) and the Qwen decode by 6 % per token. The earlier pair member
g16r1new (host 165.6 kJ) lost its Gemma assistance to an OnePlus residency effect (234 Gemma calls, 51 decisions during the
shard load) and is not a kernel comparison. Baseline pair g62 (10-26): desktop 179.6 / CPU-Pixel 155.3 kJ. Pointer
`W/DERIVE_AOA_EVIDENCE` (and the --helper-evidence line of `W/DERIVE_MP_ARGS`) now name the OpenCL r2 bundle; the CPU bundle
pointers are kept as `*.cpu-path` (`rig/switch_pixel_evidence.sh` swaps them).
