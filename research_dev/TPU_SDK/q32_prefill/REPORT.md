# Pixel 10 Pro TPU prefill screen: one Qwen3-32B Q4_0 FFN layer

2026-10-08. Latency feasibility screen only (no production integration).
Device: Pixel 10 Pro (Tensor G5, serial 5A040DLCH004ES, rooted; the same boot `dbff3b27…` for every run).
SDK v2.0-20260902, LiteRT 2.2.0, vendor TPU runtime `libedgetpu_litert.so` (SouthBound 0.14).

## Verdict

**No.** The Pixel TPU cannot run a 512-token prefill of one 32B FFN layer in about 40–60 ms.

Best Q4_0-faithful result: the exactly dequantized Q4_0 weights in fp16 take **241 ms per invoke at M = 512**.
- TPU hardware time is 238 ms, which is 1.67 TFLOPS.
- That is 4.0–6.0× the budget, 5.3× slower than the OnePlus 15 HTP (45.5 ms), and 6× slower than the desktop's ~40 ms PCIe streaming.

The limit is compute, not configuration:
- The array saturates at ~1.7 TFLOPS from M ≈ 128 upward.
- A gate+up-only graph peaks at 2.0 TFLOPS, so even a perfect graph would need ≥ 200 ms.
- Sharding makes it slower.
- The runtime performance/burst modes are ignored on this phone (SouthBound 0.14 < 0.18).
- bf16 runs at the same speed with 9× the error.

There is no fast, Q4_0-preserving 4-bit path in this SDK (section 4). int8 DRQ is a different numeric format: 212 ms (1.94 TOPS) with 4.4 % error.

Where the TPU wins:
- **Pixel CPU (ggml i8mm):** the TPU is faster for M ≳ 48 tokens (2.1× at 128, 2.2× at 512). The CPU wins at M ≤ 32 and for decode.
- **Desktop ~40 ms per-micro-batch streaming:** never faster. The TPU floor is 47–50 ms invoke for M ≤ 64, close only for micro-batches of ≤ 64 tokens. It is 1.6× slower at 128 and 6× slower at 512.
- **OnePlus HTP:** the TPU loses at every M, by 4.6–12×.

| M (tokens) | Pixel TPU fp16 invoke | Pixel TPU hw | Pixel CPU (i8mm) | OnePlus 15 HTP | TPU vs HTP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.0 ms | 40.7 ms | 9.4 ms | 3.9 ms | 12.3× slower |
| 32 | 47.1 ms | 41.0 ms | 39 ms | 9.6 ms | 4.9× slower |
| 128 | 65.7 ms | 58.5 ms | 138 ms | 14.2 ms | 4.6× slower |
| 512 | 241.1 ms | 237.9 ms | 531 ms | 45.5 ms | 5.3× slower |

The CPU and HTP columns are earlier measurements (phone prefill screen, 2026-10-08) and were not re-measured here. Closing the gap at M = 512 would need ≈ 9 TFLOPS sustained; the TPU delivers 1.7.

## 1. Variant × M

Every row is one full layer: `y = (silu(x@gate.T) * (x@up.T)) @ down.T`, with x and y FP32 `[M, 5120]` and n_ff 25600.

- **Execution:** every model compiled to a single `DISPATCH_OP` (one partition, all ops offloaded). The probe rejects any non-dispatch op and requests NPU-only execution.
- **invoke p50:** median wall time of `LiteRtRunCompiledModel` over 30 calls (M ≤ 32), 20 calls (M = 64/128) or 12 calls (M = 512), after warm-up.
- **hw:** vendor `hardware_execution_time_us` divided by the execution count.
- **TFLOPS:** 6·M·5120·25600 / time.
- **rel-L2:** maximum over 2 input cases of ‖y_TPU − y_ref‖ / ‖y_ref‖. The reference is fp32 NumPy with the exactly dequantized Q4_0 weights.
- **Repeats:** all repeated outputs were bit-identical and finite.

| Variant (what reached the TPU) | M | invoke p50 ms | hw ms | TFLOPS (hw) | rel-L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| (a) fp16: dequantized Q4_0 as FLOAT16 constants, `half` truncation | 1 | 47.95 | 40.67 | 0.019 | 4.6e-4 |
| | 32 | 47.12 | 41.02 | 0.61 | 4.4e-4 |
| | 64 | 49.78 | 42.69 | 1.18 | 4.4e-4 |
| | 128 | 65.70 | 58.55 | 1.72 | 4.4e-4 |
| | **512** | **241.14** | **237.91** | **1.69** | 4.4e-4 |
| (b) bf16: same source, `bfloat16` truncation | 1 | 47.89 | 40.89 | 0.019 | 4.2e-3 |
| | 32 | 44.20 | 41.32 | 0.61 | 4.0e-3 |
| | 128 | 65.95 | 58.34 | 1.73 | 4.0e-3 |
| | 512 | 241.35 | 236.88 | 1.70 | 4.0e-3 |
| (c) int4 exact Q4_0, "emulated sub-channel" (INT4 q verbatim + signed fp32 d, RESHAPE→BATCH_MATMUL→MUL→SUM→RESHAPE) | 1 | 99.53 | 92.72 | 0.008 | 7.3e-4 |
| | 32 | 866.2 | 859.6 | 0.029 | 7.1e-4 |
| | 64 | 1850.3 | 1842.6 | 0.027 | 7.1e-4 |
| | 128, 256 | compiles; TPU firmware error code 10 on the first invoke | | | |
| | 512 | compiler `INTERNAL` error | | | |
| (d) int8 DRQ, a different format (per-channel int8 requant, `enable_dynamic_range_quantization`) | 1 | 49.45 | 42.69 | 0.018 | 5.1e-2 |
| | 32 | 45.18 | 41.94 | 0.60 | 4.4e-2 |
| | 128 | 63.64 | 56.21 | 1.79 | 4.4e-2 |
| | 512 | 211.72 | 207.41 | 1.94 | 4.4e-2 |
| (ref) int4 per-channel DRQ, not Q4_0 | 1 | 27.75 | 20.88 | 0.038 | 0.32 |
| | 128 | 101.52 | 94.13 | 1.07 | 0.30 |
| | 512 | 410.30 | 405.98 | 0.99 | 0.30 |

Tuning at M = 512 with fp16 weights:

| Change | invoke p50 ms | hw ms | Note |
| --- | ---: | ---: | --- |
| baseline (`sharding_intensity=minimal`) | 241.1 | 237.9 | min/max over 12 calls: 239.9 / 242.4 |
| `sharding_intensity=moderate` | 332.6 | 329.8 | slower |
| `sharding_intensity=maximum` | 661.4 | 654.1 | slower |
| runtime `performance_mode=5` (burst) | 243.4 | 236.7 | ignored: SouthBound 0.14 < 0.18 |
| runtime `performance_mode=3` (high perf) | 245.3 | 238.2 | ignored |
| compile-time `performance_mode=high_performance` | – | – | compiled file byte-identical to the baseline |
| gate+up only (2/3 of the FLOPs) | 141.2 | 134.0 | 2.00 TFLOPS, the highest rate seen |
| down only (1/3 of the FLOPs) | 133.7 | 126.7 | 1.06 TFLOPS; its input is 52 MB of FP32 |

The probe's FP32 copies into and out of uncached AHardwareBuffers add 49 / 71 / 93 / 146 / 390 ms of worker time at M = 1 / 32 / 64 / 128 / 512. That cost is excluded above; production would need f16 or zero-copy I/O.

## 2. Memory

There is one ION/dma-buf "system" heap buffer per compiled layer; `dmabuf_dump` and `/proc/self/fdinfo` agree.

| Variant | Compiled .tflite | Weight dma-buf | Activation scratch | MemAvailable drop per layer |
| --- | ---: | ---: | ---: | ---: |
| fp16 / bf16 | 787 MB | 750 MiB | 0 / 9 / 18 / 75 MiB at M = 1 / 64 / 128 / 512 | 0.76–0.89 GB |
| int4 exact (emulated) | 223–253 MB | 212–221 MiB | 382 MiB (M = 1), 751 MiB (M = 32), 1253 MiB (M = 64) | 0.48–1.23 GB |
| int8 DRQ | 394–530 MB | 375–505 MiB | 0–102 MiB | 0.40–0.57 GB |
| int4 per-channel DRQ | 197–215 MB | 188–204 MiB | 0–127 MiB | 0.20–0.42 GB |

- **Residency across calls: yes.** One process loaded 9 fp16 layers (7.35 GB of dma-buf; MemAvailable went from 11.4 to 4.2 GB). Round-robin invokes took 62–70 ms each, the same as a single layer, with bit-identical outputs. A 3.5 GB free-memory guard stopped the 10th load.
- **10 fp16 layers:** ≈ 7.9–8.9 GB. That fits only marginally, and not next to the 2.2 GB Q4_0 CPU copy. fp16 needs 0.79 GB per layer against 0.22 GB for Q4_0.
- **Load time:** with a warm page cache, 0.36–0.51 s per layer; the first invoke takes 0.18–0.33 s. While a layer loads, the model file is mmapped as well, so peak memory is about 2× the weights.

## 3. Energy (ODPM, M = 512, net of idle, per invoke)

| Variant | VDD_TPU + VDD_TPU_M | DRAM + GMC rails | TPU rail power |
| --- | ---: | ---: | ---: |
| fp16 | 0.47 J (0.92 mJ/token) | 0.23 J | 1.09 W + 0.09 W |
| int8 DRQ | 0.39 J | 0.21 J | 0.99 W + 0.09 W |
| int4 per-channel DRQ | 0.57 J | 0.40 J | 0.90 W + 0.12 W |

A root sampler read the rails every ~0.1 s, aligned to the timed loop by CLOCK_BOOTTIME. The TPU rails read ~0 W at idle. VBATT is not total power, because the phone is USB-powered.

## 4. 4-bit representations

The effective TPU weights were read column by column through one-hot inputs (`results/INT4_DIAG.json`).

- **TFLite `BlockwiseQuantization`** (INT4 + fp16 per-32-block scales, the ai-edge-quantizer format) compiles fully and runs fast, but **the scales are silently ignored**.
  - The effective weight equals raw q, so outputs are wrong by 1e2–1e6. This is a dangerous silent failure.
  - `enable_dynamic_range_quantization` and `enable_4bit_compilation` change nothing.
  - Positive scales give a compiler `INTERNAL` error.
- **INT4 → DEQUANTIZE → FC**, with or without a scales MUL: compiler `INTERNAL` error.
- **Emulated sub-channel (variant c):**
  - It is exact: one-hot rel-L2 = 0, including signed scales.
  - It is not fused: the `[M, K/32, 1, N]` partial products are materialized in DRAM. That causes the huge scratch, a 2–35× slowdown, and failures at M ≥ 128.
- **Per-channel INT4:** runs with its scales honoured. Requantizing Q4_0 to it costs 14 % weight error and 30 % layer error, and it is slower than fp16 at M ≥ 128.
- **Conclusion:** there is no way to keep Q4_0-sized weights on this TPU at usable speed.

## 5. Compile options

All variants use `compile_q32.py` → `ai_edge_litert.aot.aot_compile(..., target=[Tensor_G5], keep_going=False, **flags)`. Plugin sha256 is `d74bea45…`; per-compile records are in `results/compile/*.RESULT.json`.

| Variant | Flags |
| --- | --- |
| fp16 | `google_tensor_truncation_type=half`, `google_tensor_sharding_intensity=minimal` |
| bf16 | `google_tensor_truncation_type=bfloat16`, `minimal` |
| int4 exact (emulated) | `half`, `minimal` (4-bit / DRQ flags give the same program) |
| int8 DRQ | `half`, `minimal`, `google_tensor_enable_dynamic_range_quantization=true` |
| int4 per-channel DRQ | as int8 DRQ |
| tuning | `sharding_intensity=moderate/extensive/maximum`; `performance_mode=high_performance` |

- **Source graphs:** FLOAT16 constants → DEQUANTIZE → FULLY_CONNECTED. The full layer had no FlatBuffer or compile-memory problem: compilation took 1.5–4 min with ~11 GB host RSS.
- **Probe:** `q32_tpu_probe` (sha256 `b2a0363b…`), NDK r28b.
- **Libraries:** `libLiteRt.so` (sha256 `97355a36…`) and `libLiteRtDispatch_GoogleTensor.so` (sha256 `35b59265…`), the same binaries as the M3 runs.

## 6. Inputs, weights, reference

- **Weights:** layer 11 gate/up/down Q4_0 bytes from `q32-phone/pixel-11-20/PIXEL10PRO0.ffn.gguf`, byte-identical to `Qwen3-32B-Q4_0.gguf`. `q4_0.py` is bit-exact against gguf-py `dequantize`; half of all blocks have negative d.
- **Inputs:** deterministic synthetic activations (no real ones were captured):
  - `x = rmsnorm(h) · ffn_norm.weight`, with layer 11's real norm weight;
  - h is Gaussian with per-channel lognormal(0, 0.5) scales and 4 outlier channels ×20;
  - x RMS ≈ 0.37.
- **Reference:** fp32 NumPy; fp32 and fp64 agree to ≤ 6e-7.

## 7. Blockers and notes

- The TPU dispatch runtime does not work as root ("No usable Dispatch runtime found"). The probe runs as `shell`, with a separate root energy sampler.
- Performance hints need SouthBound ≥ 0.18; this phone has 0.14.
- **Device hygiene:**
  - The Pixel lock `/data/local/tmp/.s42-pixel-ffn-kernels.lock` was held for every phone run.
  - Phone files were only under `/data/local/tmp/s43-q32-tpu`; models and inputs were deleted afterwards.
  - There was no reboot, no adb server restart, no OnePlus or rig-lock use, and no sysfs change.
- **Raw outputs:** `/mnt/data_h16t/zs89458/claude-scratch/q32-tpu/runs` on the laptop and `/mnt/storage/s43-ws17-quantized-20261003/q32-phone/tpu/runs` on the desktop.
- **Not measured:** thermal behaviour, cold load from flash, multi-layer graphs, f16 or zero-copy I/O.

## Files

- **Data prep:** `gguf_index.py`, `q4_0.py`, `make_vectors.py`.
- **Model build:** `export_q32_ffn.py`, `compile_q32.py`, `build_model.sh`, `cpu_check.py`.
- **Phone side:** `q32_tpu_probe.cpp` (the probe), `run1.sh` (phone wrapper with the lock), `phone_jobs.sh` (desktop job runner), `odpm_sampler.sh`.
- **Analysis:** `analyze.py`, `odpm_energy.py`, `int4_diag.py`.
- **`results/`:**
  - `LATENCY_NUMERICS_MEMORY.json`, `INT4_DIAG.json`, `VECTORS.json`;
  - `odpm/*.json`;
  - `compile/*.json` (every compile, including failures);
  - `phone_logs/`.
