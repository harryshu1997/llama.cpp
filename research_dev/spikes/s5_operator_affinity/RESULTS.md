# S5-V0 Operator-Affinity and Additive-Capacity Screen -- RESULTS

Status: COMPLETE. Perf sweep (7 rounds x 5 device-backends), supplementary projection sweep,
and USB transport measurement all done 2026-07-11. **Verdict: FAIL** (section 9). Energy
BLOCKED; decode-attention C-sweep BLOCKED pending a reviewed measurement-only enabler.

The question: do OP12/OP15 add **useful concurrent capacity** to an A6000 when treated as
extra backends running **whole, unsplit** Gemma-4 12B FP16 operator instances? This is NOT
S3 (output-row split, failed) nor S4 (Adreno attention offload, failed); those negatives are
not re-litigated. No Gemma graph / KV / scheduler / serving-path code was modified. The
pre-existing uncommitted edits in `layersplit.cpp`, `microop.cpp`, `oplayerprof.cpp`,
`ggml-hexagon.cpp` were preserved untouched.

## 1. Build, revision, device/backend identity

```text
Git revision:        933c722f6 (+ preserved uncommitted FA-toggle / per-tensor-share /
                     fused-FA edits in layersplit.cpp + ggml-hexagon.cpp; NOT modified here)
Host build:          build-cuda/ (CUDA), test-export-graph-ops + test-backend-ops rebuilt from
                     current source 2026-07-11.
Android build:       build-snapdragon/, docker snapdragon-toolchain-hostgcc:v0.3, NDK r28b,
                     arm64-v8a android-31, Ninja; GGML_HEXAGON=ON GGML_OPENCL=ON
                     GGML_OPENCL_PROFILING=ON LLAMA_BUILD_TESTS=ON. test-backend-ops rebuilt
                     from current source 2026-07-11 (bin/test-backend-ops).
Model:               /home/.../models/gemma-4-12B-it-f16.gguf (F16, 48 layers, dense, tied lm_head)
```

| Tag | Device | ADB serial | Backend dev | SoC / accel | HTP hwinfo |
|---|---|---|---|---|---|
| cuda0 | RTX A6000 | (host) | CUDA0 | GA102, driver 580.159.03, 48 GB | - |
| op15_HTP0 | OnePlus 15 CPH2749 | 3C15AU002CL00000 | HTP0 | SM8850, Hexagon v81 | 8 thr / 8 hvx / 1 hmx / 8 MB vtcm; skel libggml-htp-v81.so |
| op15_GPUOpenCL | OnePlus 15 | 3C15AU002CL00000 | GPUOpenCL | Adreno 840, 7556 MB | - |
| op12_HTP0 | OnePlus 12 CPH2583 | 5ae7a43d | HTP0 | SM8650, Hexagon v75 | 4 thr / 4 hvx / 1 hmx / 8 MB vtcm; skel libggml-htp-v75.so |
| op12_GPUOpenCL | OnePlus 12 | 5ae7a43d | GPUOpenCL | Adreno 750 | - |

CPU backend is the correctness reference only (harness backend2).

## 2. Exact commands

```sh
# (host) export exact real-graph ops, F16 model, sweeping M and B
build-cuda/bin/test-export-graph-ops -m gemma-4-12B-it-f16.gguf -c 4096 -b M -ub M -np 1 -o msweep_mM.txt   # M in {1,5,16,32,128,512}
build-cuda/bin/test-export-graph-ops -m gemma-4-12B-it-f16.gguf -c 2048 -b 2048 -ub 2048 -np B -o attn_bB.txt  # B in {16,32}
# curate exact matrix cells (build_curated.py): cur_gemm.txt(24) cur_light.txt(8) cur_attn.txt(4) cur_supp.txt(12)

# (host) CUDA0
CUDA_VISIBLE_DEVICES=0 build-cuda/bin/test-backend-ops {support|test|perf} --test-file cur_all.txt -b CUDA0 --output sql

# (device) HTP0 / GPUOpenCL  (LD_LIBRARY_PATH+ADSP_LIBRARY_PATH -> /data/local/tmp/s5tbo)
adb -s <serial> shell 'cd /data/local/tmp/s5tbo && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 \
   ./test-backend-ops {support|test|perf} --test-file <file> -b {HTP0|GPUOpenCL} --output sql'
# perf: 7 processes/backend, rotating device order, discard round 0, median+range over rounds 1-6.
```

## 3. Correctness and fallback table (mode=test vs CPU reference; support = ggml_backend_supports_op)

The harness computes each op **directly on the chosen backend** (no per-op scheduler), so
"not supported" is the only fallback signal -- there is no silent CPU fallback path.

| Cell | cuda0 | op15_HTP0 | op15_GPU | op12_HTP0 | op12_GPU |
|---|---|---|---|---|---|
| MUL_MAT SWA_Q/SWA_OUT/FFN_gate_up/FFN_down (all M) | PASS | PASS | PASS | PASS | PASS |
| RMS_NORM (M32,512) | PASS | PASS | PASS | **CRASH** (v75 DSP-queue 0x2e, standalone) | PASS |
| ROPE (M32,512) | PASS | PASS | PASS | **CRASH** (v75 DSP-queue 0x2e, standalone) | PASS |
| ADD (M32,512) | PASS | PASS | PASS | PASS | PASS |
| GEGLU (M32,512) | PASS | PASS | PASS | PASS | PASS |
| FLASH_ATTN_EXT SWA hd256 (B16,32) | PASS | PASS | PASS | **FAIL** (correctness vs CPU) | PASS |
| FLASH_ATTN_EXT FULL hd512 (B16,32) | PASS | PASS | **UNSUPPORTED** (no hd512 kernel) | **FAIL** (correctness) | **UNSUPPORTED** |

Findings that gate the screen:
- **GPU (both Adreno) cannot run FULL-class attention** (head_dim 512 absent from the OpenCL
  supported_dims table) -> GPU attention covers at most the 40/48 SWA layers. Confirms S4.
- **op12 HTP0 (v75) fails FA CPU-correctness on both classes** while op15 HTP0 (v81) passes ->
  op12's attention path is rejected; op12 as a full-decoder-layer backend is INVALID.
- **op12 HTP0 (v75) crashes on standalone RMS_NORM and ROPE** (DSP-queue fault) -- these ops
  work inside the fused decode graph (layersplit proves this) but not in the isolated single-op
  test path. A test-backend-ops x v75 limitation, not a model limitation. Excluded from op12
  HTP measurement (norm/rope substituted from op15_HTP0 in island estimates, flagged).

## 4. Median primitive timing

7 processes per (backend,cell), round 0 discarded, median + CoV over rounds 1-6. Rotated
device order, thermal captured (op15 tmax sensor pinned 95 C throughout = a fixed/disabled
zone, not a real reading; cpu7 883 MHz; op12 tmax 47-58 C, cpu7 672 MHz; A6000 41-42 C idle
clocks, boosts under load). Full raw: `scratchpad/s5_operator_affinity/sql/perf_*.sql`.

**Median us (CoV%). HTP/CUDA = op15_HTP0 vs cuda0 slowdown.**

### Dense MUL_MAT (F16 weight x F32 act)
| cell | M | cuda0 | op15_HTP0 | op12_HTP0 | op15_GPU | op12_GPU | HTP/CUDA |
|---|--:|--:|--:|--:|--:|--:|--:|
| SWA_Q [3840,4096] | 1 | 47 (0.0) | 456 (2.1) | 581 (8.1) | 459 (0.1) | 525 (2.9) | 10x |
|  | 5 | 47 | 994 | 1376 | 3314 | - | 21x |
|  | 16 | 50 | 1041 | 1401 | 3305 (16.1) | 5732 | 21x |
|  | 32 | 55 | 1047 | 1412 | 3361 (15.5) | 5800 | 19x |
|  | 128 | 68 | 1050 | 1376 | 6191 | 10524 | 15x |
|  | 512 | 217 | 1435 | 2313 | 23591 | - | 7x |
| SWA_OUT [4096,3840] | 1 | 46 | 459 | 570 | 455 | 527 | 10x |
|  | 512 | 200 | 2432 | 3247 | 23464 | - | 12x |
| FFN_gate/up [3840,15360] | 1 | 168 | 1807 (5.8) | 2089 | 1702 | 1901 | 11x |
|  | 32 | 199 | 3756 | 5160 | 10808 (15.6) | 18389 | 19x |
|  | 128 | 216 | 4055 | 5254 | 22014 | 37202 | 19x |
|  | 512 | 643 | 5163 | 8701 | 87622 (17.7) | - | 8x |
| FFN_down [15360,3840] | 1 | 168 | 1802 (5.2) | 2199 | 1729 | 1993 | 11x |
|  | 32 | 182 | 3857 | 5204 | 36909 (22.7) | 32677 (5.3) | 21x |
|  | 128 | 222 | 3992 | 5202 | 92162 (15.6) | 72022 (7.1) | 18x |
|  | 512 | 614 | 16260 | 21389 | 467960 (5.4) | - | 26x |

(M=5,16,128 rows for SWA_OUT/FFN elided for space; full matrix in `perf_summary.json`.)

### Attention FLASH_ATTN_EXT (decode, n_kv=256 -- see limitation 1)
| class | B | cuda0 | op15_HTP0 | op12_HTP0 | op15_GPU | op12_GPU |
|---|--:|--:|--:|--:|--:|--:|
| SWA hd256 | 16 | 56 | 4359 | 97 | 7877 | (n/s) |
| SWA hd256 | 32 | 111 | 8684 | 168 | 18644 | (n/s) |
| FULL hd512 | 16 | 29 | 894 | 163 | (unsupp) | (unsupp) |
| FULL hd512 | 32 | 78 | 1746 | 308 | (unsupp) | (unsupp) |

op12_HTP0 FA times are perf-only and **FAIL CPU-correctness (rejected)**. op15_HTP0 FA is
correct but the isolated-op time is anomalous (op15 8.7 ms vs op12 0.17 ms for the same op,
both CoV<0.3%) -- **isolated FLASH_ATTN_EXT on HTP is not representative of the fused decode
graph** (S3 H0 measured op15/op12 HTP whole-layer B32 decode at 34/40 ms, both correct). Use
these HTP FA numbers only as an isolation artifact, not a fused-graph attention cost.

### Lightweight controls
| op | M | cuda0 | op15_HTP0 | op12_HTP0 | op15_GPU |
|---|--:|--:|--:|--:|--:|
| RMS_NORM | 32/512 | 3 / 29 | 29 / 265 | crash / crash | 21 / 303 |
| ROPE | 32/512 | 2 / 27 | 37 / 279 | crash / crash | 22 / 232 |
| ADD | 32/512 | 2 / 38 | 37 / 371 | 38 / 494 | 26 / 343 |
| GEGLU | 32/512 | 4 / 139 | 143 / 1624 | 244 / 2911 | 60 / 1338 |

**CoV gate (<=5%):** MET for cuda0 (all <1%) and both HTP GEMMs at M>=5 (<=3%); marginal at
M=1 (op15 FFN 5.2-5.8%, op12 SWA_Q 8.1%). **FAILED for Adreno GPU** at mid-M (15-23% on
FFN/SWA GEMMs). GPU also non-viable on magnitude (up to 468 ms/op) and op12_GPU perf is
incomplete (300 s process cap on large-M Adreno GEMMs).

Reliability note: the GEMM numbers are trustworthy -- decode-GEMV M=1 matches DRAM bandwidth
(FFN gate/up 118 MB / 168 us = 700 GB/s on A6000; / 1807 us = 65 GB/s on op15 HTP), low CoV,
and the perf harness keeps weights resident and re-reads them per op (correct model).

**All OpenCL results are STOCK** (xmem GEMM `GGML_OPENCL_ADRENO_XMEM_GEMM` NOT set). The
env-gated xmem image-GEMM path (per prior work ~2.35x faster on op12 GPU decode) was **not
tested here** -- it would not close the 20-179x HTP-vs-A6000 gap and carries the known
stale-prepack-cache correctness hazard (S3). Any future xmem run must enable both required env
vars, run correctness first, and be reported separately.

## 5. Inferred FFN / attention / full-layer island estimates

Sums of independently measured single-op medians. **NOT a fused-graph measurement.**
`T_FFN = T_norm + 2*T_gate_up + T_GEGLU + T_down + T_residual`. K/V-proj measured via the
supplementary sweep (`cur_supp.txt`); KV-store (SET_ROWS) omitted (small). op12_HTP0 norm/rope
substituted from op15_HTP0 (v75 standalone crash), flagged.

FFN island (us/layer):

| backend | FFN M32 | FFN M512 |
|---|--:|--:|
| cuda0 | 588 | 2108 |
| op15_HTP0 | 11579 | 28846 |
| op12_HTP0 | 15837 (norm sub) | 42462 (norm sub) |
| op15_GPUOpenCL | 58632 | 645187 |

Attention island (SWA, us/layer, M=32, FA B=32) and full SWA decoder layer + throughput:

| backend | ATTN-SWA M32 | FULL LAYER M32 | R = 32/T (layer-tok/s) | R/R_cuda |
|---|--:|--:|--:|--:|
| cuda0 | 299 | 887 | 36086 | 1.000 |
| op15_HTP0 | 12031 | 23610 | 1355 | 0.038 |
| op12_HTP0 | 4586 (FA invalid) | 20422 (FA invalid) | 1567 | 0.043 |
| op15_GPUOpenCL | 28586 | 87218 | 367 | 0.010 |

The attention island inherits the isolated-FA artifact (limitation 2); the FFN island (reliable
GEMMs only) already shows the phone is **20-27x slower per layer** than the A6000. op12's full
layer is INVALID (FA fails correctness). At the A6000's efficient batch the gap widens: A6000
FFN throughput at M=512 = 242,934 layer-tok/s vs op15 HTP ~1,355 (179x).

## 6. Transfer measurements (residual `[3840,M]` over USB adb; weights resident, excluded)

The real pipeline ships the `[n_embd=3840, n_tokens]` residual over USB via `adb push`/`pull`
(phones cannot peer; all hops relay through the host). Median of 9 trials, weights excluded.

| dev | dtype | M | bytes | push ms | pull ms | round-trip ms | eff MB/s |
|---|---|--:|--:|--:|--:|--:|--:|
| op15 | F32 | 16 | 245760 | 17 | 16 | 31 | 15 |
| op15 | F32 | 32 | 491520 | 24 | 22 | 44 | 22 |
| op15 | F32 | 512 | 7864320 | 205 | 199 | 401 | 39 |
| op15 | F16 | 16 | 122880 | 14 | 13 | 24 | 10 |
| op15 | F16 | 32 | 245760 | 17 | 16 | 32 | 15 |
| op15 | F16 | 512 | 3932160 | 109 | 104 | 214 | 37 |
| op12 | F32 | 512 | 7864320 | 211 | 212 | 424 | 37 |
| op12 | F16 | 32 | 245760 | 20 | 19 | 37 | 13 |
| op12 | F16 | 512 | 3932160 | 116 | 114 | 226 | 35 |

Effective goodput is only 10-39 MB/s (adb per-call + protocol overhead dominates; a fixed
~14-20 ms floor even at M=16). F16 halves the payload but the small-M floor limits the win.
**A single residual round-trip (24-49 ms at M=16-32, 214-424 ms at M=512) EXCEEDS the phone's
per-layer compute:** transport/compute per single SWA layer = 1.36 (op15) / 1.81 (op12). This
amortizes only if a phone holds many layers per visit, but the phone's ~26x throughput deficit
is the binding limit regardless.

## 7. Additive-backend capacity bound

`R_ideal = R_CUDA + R_OP15_HTP + R_OP12_HTP`, `ideal_gain = R_ideal / R_CUDA` (layer-tokens/s).

Compute-only ideal gain (before transport):

| comparison basis | R_CUDA | +R_op15 | +R_op12 | ideal_gain |
|---|--:|--:|--:|--:|
| same batch M=32 (handicaps A6000) | 36086 | 1355 | 1567* | **1.038x** (CUDA+op15) / 1.081x (both*) |
| A6000 efficient batch (M=512 FFN proxy) | 242934 | 1355 | 1567* | **1.012x** |

*op12 HTP full-layer throughput uses its (invalid) FA; even counting it, the gate fails.
Both phones together are **1.2% of fleet capacity** at the A6000's operating batch.

**B_total sweep** (A6000 caps its batch at 256; overflow assigned to the two phones
concurrently; compared to CUDA queuing the identical total as sequential sub-batches). Ratio =
CUDA-only-time / 3-device-makespan per layer; >1 means the phones help.

| B_total | A6000-only /layer | 3-device makespan /layer | ratio | outcome |
|--:|--:|--:|--:|---|
| 128 | (fits in one A6000 batch) | same | 1.00 | phones idle, no gain |
| 256 | (fits in one A6000 batch) | same | 1.00 | phones idle, no gain |
| 320 | 1.3 ms (oper) / 8.9 ms (M32) | 21.9 ms (phones do 64) | **0.06 / 0.41** | 3-device 2.5-17x SLOWER |
| 384 | 1.6 ms (oper) / 10.6 ms (M32) | 43.8 ms (phones do 128) | **0.04 / 0.24** | 3-device 4-28x SLOWER |

For B_total>256 the phones are net-negative: their combined ~2,922 layer-tok/s cannot absorb
the overflow, while the A6000 clears 384 as ~1.5 sub-batches at near-flat per-layer cost
(588 us @ M32 -> 2108 us @ M512). The A6000's batch-384 limit is a **memory (OOM)** limit, not a
throughput wall -- it is trivially worked around by sub-batching, so there is no throughput
capacity for the phones to usefully backfill.

## 8. Explicit limitations of test-backend-ops (and this screen)

1. **Decode `n_kv` pinned to the 256-cell FA pad.** `test-export-graph-ops` reserves the
   decode graph on an empty KV cache, so decode (n_q=1) FLASH_ATTN_EXT is only reachable at
   logical C~=256. The requested C={32,512,1024} sweep is **NOT deliverable** with the
   unmodified tool (C=32 below the pad floor; C=512/1024 need a filled KV; larger n_kv only
   appears in prefill graphs with n_q>1). The C-parameterized decode-attention sweep is
   **BLOCKED pending a reviewed measurement-only enabler** (smallest options: a `--kv-fill C`
   decode-reserve knob, or a test-file generator that rewrites only the K/V/mask KV-length of
   the real decode FA op). All attention numbers here are at C~=256.
2. **Isolated single-op execution.** Each op runs alone (own alloc, own DSP session flush).
   This is stricter than production (no operator fusion, no persistent session) and exposes
   backend fragility that the fused graph hides (op12 v75 RMS_NORM/ROPE crash; op12 v75 FA
   correctness). Island sums therefore also miss intra-layer fusion and overlap.
3. **op12 GPUOpenCL perf is incomplete** -- large-M Adreno-750 GEMMs exceed the 300 s
   per-process cap; only low-M rows complete. op12 GPU is directionally the slowest backend.
4. **flops/bandwidth columns are 0** for MUL_MAT in this harness (generic-op path); FLOPS and
   effective bytes/s are computed offline from shapes x median time.
5. **Energy: BLOCKED.** USB-powered phone telemetry cannot measure phone J/op (input-current
   cap clips the rail; battery coulomb reads 0 on Full). All energy conclusions are BLOCKED;
   the next gate requires unplugged WiFi-ADB or calibrated physical power measurement.

## 9. Verdict

**S5-V0: FAIL.** No Gemma-4 12B FP16 operator island qualifies OP12/OP15 as a useful additive
backend for the A6000 under the decision gates. Do NOT build an FFN-island harness or scheduler.

Gate-by-gate:

| Gate | Result |
|---|---|
| correctness, no CPU fallback | **PARTIAL** -- GEMMs/lightweight PASS on HTP; GPU cannot run FULL-class attention (hd512 unsupported, both phones); op12 HTP0 attention FAILS correctness; op12 HTP0 standalone RMS_NORM/ROPE crash |
| timing CoV <= 5% | HTP GEMMs MET (M>=5); marginal at M=1; **Adreno GPU FAILS** (15-23%) |
| transport-adjusted 3-device goodput >= 1.10x | **FAIL** -- compute-only ideal is 1.012-1.081x; residual RTT exceeds per-layer compute (1.36-1.81x) |
| positive for B_total > 256 | **FAIL** -- 3-device is 2.5-28x SLOWER than CUDA-only (ratio 0.04-0.41); A6000's batch limit is memory, not throughput |
| >= 15% phone RAM free | not binding -- phones have RAM; the failure is throughput, not capacity |
| energy | **BLOCKED** -- USB-powered; cannot measure phone J/op |

Why it fails: the phone's fastest correct engine (HTP) is **8-35x slower per GEMM** and
**~20-27x slower per full decoder layer** than the A6000. Two phones supply ~1.2% of the fleet's
layer-token capacity at the A6000's operating batch. Because the A6000 is throughput-flat in M
(per-layer time 588 us @ M32 -> 2108 us @ M512), server overflow above 256 is cleared by
sub-batching at trivial cost, leaving no throughput deficit for the phones to backfill; routing
overflow to phones instead makes the fleet several-fold slower. Transport (USB adb, 10-39 MB/s
effective; a single residual round-trip 24-424 ms) compounds this. The Adreno GPU is
categorically non-viable (up to 468 ms/op, 15-23% CoV, no FULL-class attention). This is
consistent with -- not a repeat of -- S3 (row-split) and S4 (Adreno attention): the phones are
simply too slow, per operator, to add concurrent server-scale capacity.

**BLOCKED (needs a reviewed measurement-only enabler, then re-run):**
- Decode-attention C={32,512,1024} sweep -- `test-export-graph-ops` pins decode `n_kv` to the
  256 FA pad (limitation 1). This does not change the verdict (GEMMs alone already fail the
  gate), but the C-scaling of HTP attention -- the term that most helps or hurts the phone
  layer -- is unmeasured. Smallest enabler: a `--kv-fill C` decode-reserve knob, or a test-file
  generator that rewrites only the K/V/mask KV-length of the real decode FA op. **STOP for
  review before applying.**
- HTP fused-graph attention cost -- isolated FLASH_ATTN_EXT via test-backend-ops is not
  representative (limitation 2); a fused single-layer probe (e.g. `oplayerprof`) would give the
  real HTP attention term. Not required to overturn a FAIL this decisive.
- **Energy: BLOCKED.** Next energy gate requires unplugged WiFi-ADB or calibrated physical
  power measurement; no phone J/op is claimable from USB-powered telemetry.

STOP here per the spike contract. The measured offline additive-capacity bound does not pass;
no island is shortlisted; no scheduler or island-harness work is authorized.
