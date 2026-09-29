# OP15 dual-engine FFN split (HTP NPU || Adreno OpenCL GPU), 2026-09-23

Question: does splitting a phone FFN layer's intermediate columns between the
Hexagon HTP and the Adreno GPU, both running at the same time, make the
per-layer FFN faster than the NPU alone? What NPU:GPU column ratio is best?

## Verdict

- **M=1 (decode): yes, 1.25x.** The best split is f = 0.85 (NPU 14848 / GPU
  2560 columns): 7.78 ms per layer vs 9.74 ms for NPU-only, which gives
  68.7 GB/s aggregate. Results are flat from f = 0.75 to 0.85 (1.24-1.25x).
  This is the ~70 GB/s single-DRAM-master ceiling measured in July. The two
  engines together do not go past it.
- **M=4: effectively no.** Best is f = 0.956 at 1.04x (9.76 vs 10.15 ms).
  For M >= 2 the Adreno F16 path uses the `l4_lm` tiled GEMM, which is
  5-7x slower than the GEMV. Every larger GPU share loses (f = 0.8 gives 0.79x).
- **M=8: yes, but mostly because the NPU default path is slow at M=8.** With
  the default HMX on, NPU-only takes 27.9 ms (HMX F16 path, 19 GB/s). Dual at
  f = 0.7 takes 19.6 ms (1.42x). HVX-only NPU (`GGML_HEXAGON_NHMX=0`) takes
  20.1 ms by itself, and HVX-only dual at f = 0.9 takes 17.8 ms. That is
  1.13x over HVX-only NPU and 1.57x over the default NPU path.

Projection (Qwen3-14B FFN, 40 layers, batch 1): 9.74 -> 7.78 ms/layer, i.e.
~390 -> ~311 ms of FFN per token (-20 %). Weights stay disjoint, so memory
does not grow: this only splits the same 535 MB/layer between two buffers.
This is a projection from a 6-layer standalone benchmark, not a production
worker measurement.

## Main table (Qwen3-14B F16, layers 10-15, 3 reps x 50 sweeps x 6 layers each)

`wall p50` = per-layer wall time from the start of both legs to the end of
the CPU merge. Each leg includes the input `tensor_set` + `graph_compute` +
output `tensor_get`. [min-max] is the spread over the 3 reps (each rep is its
own process). `agg GB/s` = full-layer bytes (534.8 MB) / wall p50. In `npu_solo`
and `gpu_solo` rows at f < 1, that same leg runs alone in the same process.
Those rows are the no-contention reference. The full table with mean/p90 is in
[TABLE_main.md](TABLE_main.md).

| M | f act | NPU/GPU cols | wall p50 ms [spread] | NPU leg (dual / solo) | GPU leg (dual / solo) | sync p50 | agg GB/s | vs NPU-only |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 17408/0 | 9.739 [9.682-9.842] | 9.74 | - | - | 54.9 | 1.000x |
| 1 | 0.956 | 16640/768 | 8.784 [8.768-8.808] | 8.78 / 9.08 | 2.74 / 0.99 | 1 us | 60.9 | 1.109x |
| 1 | 0.897 | 15616/1792 | 8.269 [8.252-8.287] | 8.27 / 8.47 | 4.74 / 1.01 | 1 us | 64.7 | 1.178x |
| 1 | **0.853** | 14848/2560 | **7.779 [7.750-7.827]** | 7.77 / 8.06 | 6.40 / 1.78 | 1 us | **68.7** | **1.252x** |
| 1 | 0.794 | 13824/3584 | 7.852 [7.787-7.898] | 7.45 / 7.53 | 7.84 / 2.34 | 2 us | 68.1 | 1.240x |
| 1 | 0.750 | 13056/4352 | 7.833 [7.798-7.901] | 7.08 / 7.14 | 7.83 / 2.62 | 2 us | 68.3 | 1.243x |
| 1 | 0.706 | 12288/5120 | 7.925 [7.846-8.082] | 6.66 / 6.74 | 7.92 / 2.91 | 2 us | 67.5 | 1.229x |
| 1 | 0.603 | 10496/6912 | 8.012 [7.992-8.041] | 5.66 / 5.74 | 8.00 / 3.95 | 2 us | 66.7 | 1.216x |
| 1 | 0.500 | 8704/8704 | 8.108 [8.069-8.137] | 4.73 / 4.82 | 8.10 / 4.86 | 3 us | 66.0 | 1.201x |
| 1 | 0.000 | 0/17408 | 9.455 [9.371-9.552] | - | 9.45 | - | 56.6 | 1.030x |
| 4 | 1.000 | 17408/0 | 10.152 [10.076-10.286] | 10.15 | - | - | 52.7 | 1.000x |
| 4 | **0.956** | 16640/768 | **9.763 [9.732-9.810]** | 9.74 / 9.70 | 4.90 / 4.12 | 1 us | 54.8 | **1.040x** |
| 4 | 0.794 | 13824/3584 | 12.925 [12.882-12.965] | 8.05 / 8.06 | 12.90 / 11.98 | 2 us | 41.4 | 0.785x |
| 4 | 0.706 | 12288/5120 | 19.572 [17.257-24.193] | 7.23 / 7.19 | 19.55 / 15.99 | 2 us | 27.3 | 0.519x |
| 4 | 0.500 | 8704/8704 | 27.807 [27.654-27.954] | 5.20 / 5.17 | 27.77 / 26.99 | 2 us | 19.2 | 0.365x |
| 4 | 0.000 | 0/17408 | 66.958 [61.916-73.829] | - | 66.96 | - | 8.0 | 0.152x |
| 8 | 1.000 | 17408/0 | 27.904 [26.036-30.885] | 27.90 | - | - | 19.2 | 1.000x |
| 8 | 0.956 | 16640/768 | 25.926 [25.264-26.847] | 25.90 / 26.23 | 4.14 / 4.03 | 1 us | 20.6 | 1.076x |
| 8 | 0.794 | 13824/3584 | 21.856 [21.597-22.218] | 21.81 / 21.53 | 12.38 / 11.96 | 1 us | 24.5 | 1.277x |
| 8 | **0.706** | 12288/5120 | **19.596 [18.576-20.894]** | 19.50 / 18.95 | 16.81 / 16.00 | 1 us | 27.3 | **1.424x** |
| 8 | 0.500 | 8704/8704 | 27.519 [27.378-27.663] | 13.89 / 13.32 | 27.49 / 26.89 | 2 us | 19.4 | 1.014x |
| 8 | 0.000 | 0/17408 | 67.659 [62.481-74.484] | - | 67.66 | - | 7.9 | 0.412x |

The requested f are rounded to 256-column multiples (f=0.95 -> 0.956, 0.9 ->
0.897, 0.85 -> 0.853, 0.8 -> 0.794, 0.7 -> 0.706, 0.6 -> 0.603). f = 0.95 is an
extra cell I added. M=4/8 were run for f in {1.0, 0.95, 0.8, 0.7, 0.5, 0.0}.

### HVX-only NPU diagnostic (`GGML_HEXAGON_NHMX=0`, 2 reps, M=4/8)

The M=8 NPU-only number is poor, so I checked whether HMX is the cause.
Full table: [TABLE_hvx_only.md](TABLE_hvx_only.md).

| M | f act | wall p50 ms [spread] | NPU leg dual/solo | GPU leg dual/solo | vs HVX NPU-only |
|---|---|---|---|---|---|
| 4 | 1.000 | 10.578 [10.572-10.583] | 10.58 | - | 1.000x |
| 4 | 0.897 | 10.018 [9.132-10.905] | 9.07 / 9.11 | 9.65 / 9.22 | 1.056x |
| 4 | 0.794 | 17.870 | 8.12 / 8.14 | 17.85 / 14.89 | 0.592x |
| 8 | 1.000 | 20.129 [20.122-20.135] | 20.13 | - | 1.000x |
| 8 | **0.897** | **17.800 [17.784-17.817]** | 17.74 / 17.69 | 10.69 / 8.94 | **1.131x** |
| 8 | 0.794 | 18.035 | 15.85 / 15.80 | 18.00 / 15.33 | 1.116x |
| 8 | 0.706 | 24.559 | 14.08 / 14.03 | 24.48 / 18.18 | 0.820x |

At M=8, HVX-only NPU (20.1 ms) beats the default HMX path (27.9 ms). The
production worker env does not set NHMX, so HMX is on there. This is a
separate, cheap win for M=8 that does not need the GPU.

## Best f and speedup vs NPU-only

| M | best f (actual) | wall | NPU-only | speedup | note |
|---|---|---|---|---|---|
| 1 | 0.85 (0.75-0.85 plateau) | 7.78 ms | 9.74 ms | 1.25x | aggregate 68.7 GB/s, at DRAM ceiling |
| 4 | 0.956 | 9.76 ms | 10.15 ms | 1.04x | within noise of "don't split" |
| 8 (HMX default) | 0.706 | 19.60 ms | 27.90 ms | 1.42x | NPU HMX F16 path is the problem |
| 8 (NHMX=0) | 0.897 | 17.80 ms | 20.13 ms | 1.13x | 1.57x vs default NPU-only |

## Mechanism (M=1)

- The NPU leg is unaffected by contention. Its dual time is equal to or
  slightly below its solo time, e.g. at f=0.85: 7.77 dual vs 8.06 solo. It
  keeps its ~55-57 GB/s. It is capped near 55 GB/s by itself, so it does not
  saturate DRAM.
- The GPU is the one that gets squeezed. Alone, the Adreno F16 GEMV
  (`kernel_mul_mat_f16_f32_l4_dr`) hits 56.6 GB/s for a full layer. With the
  NPU running at the same time it only gets the leftover ~12-14 GB/s. For
  example, at f=0.8 the GPU leg goes from 2.34 ms solo to 7.84 ms dual.
- The best f is where the GPU slice finishes in the leftover bandwidth by the
  time the NPU finishes: f ~ 55/(55+14) ~ 0.8-0.85. That matches the measured
  plateau.
- Aggregate peaks at 68.7 GB/s. Two masters do not exceed the ~70 GB/s
  single-master ceiling here. July measured 73-86 GB/s for two masters, but not
  at this operating point.
- Sync overhead (spin barrier, one host thread per backend) is 1-3 us and CPU
  merge is 3-9 us at M=1. Launch skew is < 1 us. Neither matters.

## GPU-only at M=1 and the OpenCL path used

- GPU-only per layer: 9.455 ms p50 [9.371-9.552] = **56.6 GB/s**. This is on
  par with NPU-only (54.9 GB/s). Mean is 9.90 ms and p90 is 11.5 ms, so the GPU
  has more jitter than the NPU.
- Kernel selection (from `ggml_cl_mul_mat` in ggml-opencl.cpp): for M=1 the
  F16 x F32 matmuls take `kernel_mul_mat_f16_f32_l4_dr` (buffer GEMV, not an
  image kernel). For M >= 2 they take `kernel_mul_mm_f16_f32_l4_lm` (tiled
  GEMM, 64-wide N tile underfilled at M=4/8, 7-8 GB/s). The Adreno KQ/KQV image
  path needs `ne1 >= 32` and does not apply.
- Xmem was **not used**. `GGML_OPENCL_ADRENO_XMEM_GEMM` was unset (default
  off), and `ggml_cl_can_use_adreno_xmem_gemm_f16_f32` also requires N >= 16
  tokens, so it could not trigger at M <= 8 anyway.

## Correctness

Output check: layer 10, fixed seeded input U(-1,1), compared against an fp64
CPU reference computed from the same F16 weights (rep 1). Also compared
against the NPU-only output of the same rep
([TABLE_vs_npu_only.md](TABLE_vs_npu_only.md)).

- NPU-only vs CPU: rel L2 3.2e-4 (M=1), 4.9e-4 (M=8).
- GPU-only vs CPU: rel L2 3.9e-7 (M=1). The GPU accumulates in fp32; HTP does
  not.
- Dual vs NPU-only: rel L2 6e-5 to 4.9e-4, max abs <= 3.6e-3 (ref L2 89-227).
  Every dual config is closer to the CPU reference than NPU-only (2.2e-4 to
  3.1e-4 at M=1), because the GPU share is more exact. Outputs are
  bit-identical across reps.
- Every graph node passed `ggml_backend_supports_op` on its own backend, so
  there was no CPU fallback.

## Caveats

- **DVFS / thermal.** DDR, GPU and NPU clocks are demand-scaled and cannot be
  pinned without root. The NPU-solo leg often runs slightly slower than the
  same leg under dual (e.g. 8.06 vs 7.77 ms). That is consistent with GPU
  activity raising the DDR clock. GPU-solo times drift run to run (M=4 full
  GPU 61.9-73.8 ms). The f order was rotated per rep, so drift is spread
  across cells.
- Battery was 8-14 % and charging during the whole campaign. thermal_zone0
  read 31-41 C.
- M=1 rep spreads are tight (<= 3 %).
- L = 6 layers, not 8. The HTP op-batch VA budget is `GGML_HEXAGON_VMEM=3328`
  MiB. 8 NPU-only layers (4.28 GB) would exceed it and force DSP mmap churn
  (`prep_op_bufs` drops and remaps every layer), which would bias the NPU-only
  baseline. 6 layers is 3.21 GB, still far beyond the LLCC.
- The NPU-only M=1 result (9.74 ms, 54.9 GB/s) reproduces production
  (9.72 ms, 55.0 GB/s).
- Real weights: Qwen3-14B-Q4KM-dequant-f16.gguf, blk.10-15 ffn_gate/up
  [5120,17408] and ffn_down [17408,5120], all F16. Not synthetic.
- Standalone benchmark, not the production worker. The phone CPU does the
  merge in-process. There is no USB transport, no hidden-state wire, and no
  dmabuf sharing (the leg inputs/outputs are small `tensor_set`/`tensor_get`
  calls, included in the leg times). Production integration would also need an
  OpenCL leg in `ffn-split-worker` plus a second weight residency path.
- The build is the current dirty tree (`5f89a2d9d-dirty`), not the Aug-29
  production libs. Their hashes differ, but the NPU-only baseline matches.
- Gemma-4-12B (GeGLU) was not measured. The bench supports it: `gemma4` arch
  -> geglu.
- Energy was not measured. The dual split keeps both engines busy, so J/token
  is unknown and may be worse.
- M=4/8 for f in {0.9, 0.85, 0.75, 0.6} were not run with HMX on. In the
  HVX-only diagnostic, M=4 f=0.9 had a noisy spread (9.13-10.91).

## Commands / env

Build (snapdragon docker, new dir `build-dual-ffn-bench-android`):

```sh
docker run --rm -u $(id -u):$(id -g) -v "$PWD:/workspace" -w /workspace snapdragon-toolchain-hostgcc:v0.3 bash -lc '
cmake -S . -B build-dual-ffn-bench-android -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/opt/android-ndk-r28b/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-31 \
  -DHEXAGON_SDK_ROOT=/opt/hexagon/6.4.0.2 -DHEXAGON_TOOLS_ROOT=/opt/hexagon/6.4.0.2/tools/HEXAGON_Tools/19.0.04 \
  -DPREBUILT_LIB_DIR=android_aarch64 -DGGML_HEXAGON=ON -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON -DGGML_OPENCL_PROFILING=OFF -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=ON &&
cmake --build build-dual-ffn-bench-android --target llama-ffn-dual-bench ggml-hexagon ggml-opencl htp-v81 -j 16'
```

Phone run (one process per f; see `scripts/phone_run.sh`, `scripts/campaign.sh`,
`scripts/diag.sh`). All runs were under
`flock -w 1800 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`
on the desktop, via `adb -P 5037 -s 3C15AU002CL00000`. Before each run the
script checked that no `ffn-split` process was running.

```sh
cd /data/local/tmp/s43-dual-ffn-bench-20260923 && \
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_NDEV=1 GGML_HEXAGON_MBUF=4192 \
GGML_HEXAGON_VMEM=3328 GGML_HEXAGON_NHVX=4 [GGML_HEXAGON_NHMX=0] \
./llama-ffn-dual-bench --model /data/local/tmp/s41-opoffload-dmabuf-v1/Qwen3-14B-Q4KM-dequant-f16.gguf \
  --frac <f> --layers 6 --layer-start 10 --sweeps 50 --warmup 3 --batches 1,4,8 \
  [--no-cpu-ref] --samples out/<tag>.csv --dump out/<tag>
```

## Files

- Source: `examples/layersplit/ffn-dual-bench.cpp`, target `llama-ffn-dual-bench`
  in `examples/layersplit/CMakeLists.txt`.
- Binary: `build-dual-ffn-bench-android/bin/llama-ffn-dual-bench` (+ libs,
  `ggml/src/ggml-hexagon/libggml-htp-v81.so`). Desktop copy:
  `/home/zhihao/s43-dual-ffn-bench-20260923/`.
- `logs/` main campaign (30 runs), `logs_hvx_only/` NHMX=0 diagnostic (8 runs).
  `samples/` has per-layer CSV samples and `dumps/` has layer-10 outputs.
- `analyze.py` builds the tables, `compare_dumps.py` does the dual-vs-NPU-only
  check.
- Phone directory `/data/local/tmp/s43-dual-ffn-bench-20260923` was removed
  after the run.
