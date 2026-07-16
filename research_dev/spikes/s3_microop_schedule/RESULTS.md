# S3 Heterogeneous Micro-Operator Results

Status: H0 real batch-decode baseline measured. H1 produced no complete-operator
pass and is archived without integration. Audit correction: OP12 xmem interior
rows are invalid because stale prepacked slices were reused.

Repository cleanup note: the standalone measurement harness and sweep driver
were removed from the live examples after a fail-open code audit. This document
retains the measured negative result; it is not a production implementation.

Contracts:

- [PLAN.md](PLAN.md)
- [RELATED_WORK.md](RELATED_WORK.md)

## Build and Device Identity

```text
Git revision:            933c722f6 (+ uncommitted Fused-FA, FA toggle, per-tensor sharing; preserved, no edits)
Dirty worktree:          yes
Android preset/toolchain: arm64-android-snapdragon; on-device llama-layersplit built 2026-07-09 18:38
                          (Fused-FA present: decode ctx reports flash_attn=enabled)
OpenCL profiling build:  OFF (GGML_OPENCL_PROFILING not set; H0 timing = host steady_clock ms/round)
HTP skel version:        libggml-htp-v75.so (OP12), libggml-htp-v81.so (OP15)
OpenCL driver OP12:      Adreno 750 (SM8650) -- flash_attn kernel FAILS to compile (sub_group_shuffle_xor, err -11) -> GPU decode run FA-off
OpenCL driver OP15:      Adreno 840 (SM8850) -- GPU decode run FA-off for parity
A6000 reference:         RTX A6000 48GB, CUDA driver 580.159.03, build-cuda/bin/llama-layersplit
```

## Measurement Provenance

| Device | Source | Counter/tool | Units/semantics | Access | Raw artifact | Valid |
|---|---|---|---|---|---|---|
| OP12 | timing | host steady_clock (decode-leg ms/round, dualengine section B) | ms/round | shell, no root | h0_op12_*.jsonl + h0_logs/ | timing valid; DDR = none (no root) -> effective_min_weight_read only (H1) |
| OP15 | timing | host steady_clock (decode-leg ms/round) | ms/round | shell | h0_op15_*.jsonl + h0_logs/ | timing valid; DDR ddr_counter available via dcvs/bw_hwmon_meas (root, Magisk) for H1 |
| A6000 | timing | host steady_clock | ms/round | host | h0_a6000.jsonl | timing valid |

Use only `ddr_counter`, `effective_min_weight_read`, or `none` as the bandwidth
source. Do not put an effective byte-rate value in the direct-counter column.
No DDR counters were read in H0 (H0 is a decode-timing control; DDR study is H1).

## Gate Summary

| Gate | OP12 | OP15 | Verdict |
|---|---|---|---|
| H0: trustworthy measurement | PASS | PASS | PASS -- real B-way decode, correctness holds, no cross-seq bleed, zero-interference confirmed |
| H1: batch-shaped output-row projection | INVALID/archived | FAIL | no integration -- no complete row passed; OP12 xmem correctness is invalid |
| H2: row vs branch vs streams | not run | not run | archived |
| H3: ready-task replay | not run | not run | archived |
| Mutable activation justification | not run | not run | archived |
| Production: full layer and energy | not run | not run | archived |

## H0 Solo and Co-Run Controls

The template's caveat ("dualengine leg times are collected while both workers run")
was tested directly, not assumed. Interference control on OP12 HTP0, B=8, 1-layer
shard: vary the concurrent GPU-prefill load and watch the HTP decode leg.

| Device | B | Decode | Concurrent prefill leg | HTP decode-leg ms/round |
|---|---:|---|---|---:|
| OP12 | 8 | HTP0 | GPUOpenCL, 1 token/round (~32 ms/round) | 26.2 |
| OP12 | 8 | HTP0 | GPUOpenCL, 64 tokens/round (~133 ms/round, 4x heavier) | 24.9 |

The HTP decode leg is load-insensitive (26.2 vs 24.9 ms, ~5%, within run-to-run
noise) while the co-scheduled GPU prefill quadruples. => cross-engine interference
is ~0 for NPU-decode || GPU-prefill (consistent with the prior dualengine 1.76x/1.92x
overlap result), so the dualengine decode-leg time IS a valid per-engine number for
this workload. A true single-worker solo intact-decode is not available in the
existing modes (tailbench needs an lm_head shard and inherits the 256k-vocab lm_head
cost; on OP12 its GPU path also hits the FA compile break). The H1 standalone harness
will provide clean single-worker solo/co-run/overlap for the isolated projection.
3 untimed warmup rounds precede every timed run (equal warmups across configs).

## H0 Real Batch Decode Baseline

One measured round is one `llama_decode` call with B distinct sequence IDs
(`seq_id[j]=j`), one token per sequence, private KV histories, advancing positions.
Shard `12b-f16-mid-2-3.gguf` = 1 real transformer layer (blk.2), head-less (nextn).
Timing = dualengine decode-leg mean ms/round (dualengine reports aggregate mean, not a
per-round distribution -> no p50/p95 in H0; the H1 harness will report percentiles).
Rel-L2 = batched B-way output vs each sequence replayed alone on the same engine
(dualengine section A, decode engine solo). Rel-L2 is FLAT across B=1..64 (does not
grow with B) => no cross-sequence bleed. `1-row isolation` = each serial replay matches
its batched row. Raw: `scratchpad/h0_{op12,op15,a6000}_*.jsonl`, logs `scratchpad/h0_logs/`.

| Device | Backend | B | KV len | Rounds | mean ms/round | tokens/s | Rel-L2 vs serial | 1-row isolation | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| OP12 | HTP | 1 | 30 | 30 | 10.7 | 93 | 0.000e+00 | PASS | PASS |
| OP12 | HTP | 2 | 30 | 30 | 11.3 | 177 | 0.000e+00 | PASS | PASS |
| OP12 | HTP | 4 | 30 | 30 | 17.2 | 233 | 0.000e+00 | PASS | PASS |
| OP12 | HTP | 5 | 30 | 30 | 25.6 | 196 | 5.351e-04 | PASS | PASS |
| OP12 | HTP | 8 | 30 | 30 | 27.6 | 290 | 5.000e-04 | PASS | PASS |
| OP12 | HTP | 16 | 30 | 30 | 31.8 | 504 | 5.274e-04 | PASS | PASS |
| OP12 | HTP | 32 | 30 | 30 | 39.6 | 808 | 5.232e-04 | PASS | PASS |
| OP12 | HTP | 64 | 30 | 30 | OOM | - | - | - | FAIL (2GB HTP buf map; KV alloc-all-48-layers) |
| OP12 | GPU | 1 | 30 | 30 | 18.8 | 53 | 0.000e+00 | PASS | PASS |
| OP12 | GPU | 2 | 30 | 30 | 86.7 | 23 | 1.110e-05 | PASS | PASS |
| OP12 | GPU | 4 | 30 | 30 | 87.5 | 46 | 1.385e-05 | PASS | PASS |
| OP12 | GPU | 5 | 30 | 30 | 88.6 | 56 | 1.762e-05 | PASS | PASS |
| OP12 | GPU | 8 | 30 | 30 | 89.3 | 90 | 1.806e-05 | PASS | PASS |
| OP12 | GPU | 16 | 30 | 30 | 94.3 | 170 | 2.158e-05 | PASS | PASS |
| OP12 | GPU | 32 | 30 | 30 | 102.5 | 312 | 2.099e-05 | PASS | PASS |
| OP12 | GPU | 64 | 30 | 30 | 111.6 | 574 | 2.082e-05 | PASS | PASS |
| OP15 | HTP | 1 | 30 | 30 | 11.1 | 90 | 0.000e+00 | PASS | PASS |
| OP15 | HTP | 2 | 30 | 30 | 10.3 | 194 | 0.000e+00 | PASS | PASS |
| OP15 | HTP | 4 | 30 | 30 | 12.2 | 327 | 0.000e+00 | PASS | PASS |
| OP15 | HTP | 5 | 30 | 30 | 26.5 | 189 | 4.978e-04 | PASS | PASS |
| OP15 | HTP | 8 | 30 | 30 | 24.8 | 323 | 4.985e-04 | PASS | PASS |
| OP15 | HTP | 16 | 30 | 30 | 35.8 | 447 | 5.148e-04 | PASS | PASS |
| OP15 | HTP | 32 | 30 | 30 | 34.3 | 933 | 5.102e-04 | PASS | PASS |
| OP15 | HTP | 64 | 30 | 30 | 49.1 | 1303 | 5.116e-04 | PASS | PASS |
| OP15 | GPU | 1 | 30 | 30 | 16.9 | 59 | 0.000e+00 | PASS | PASS |
| OP15 | GPU | 2 | 30 | 30 | 83.8 | 24 | 1.110e-05 | PASS | PASS |
| OP15 | GPU | 4 | 30 | 30 | 85.0 | 47 | 1.385e-05 | PASS | PASS |
| OP15 | GPU | 5 | 30 | 30 | 84.0 | 60 | 1.762e-05 | PASS | PASS |
| OP15 | GPU | 8 | 30 | 30 | 88.3 | 91 | 1.806e-05 | PASS | PASS |
| OP15 | GPU | 16 | 30 | 30 | 96.2 | 166 | 2.158e-05 | PASS | PASS |
| OP15 | GPU | 32 | 30 | 30 | 111.4 | 287 | 2.099e-05 | PASS | PASS |
| OP15 | GPU | 64 | 30 | 30 | 132.2 | 484 | 2.082e-05 | PASS | PASS |
| A6000 | CUDA0 | 1 | 30 | 30 | 1.3 | 758 | 0.000e+00 | PASS | PASS |
| A6000 | CUDA0 | 5 | 30 | 30 | 1.5 | 3356 | 7.406e-04 | PASS | PASS |
| A6000 | CUDA0 | 8 | 30 | 30 | 1.5 | 5195 | 7.335e-04 | PASS | PASS |
| A6000 | CUDA0 | 16 | 30 | 30 | 1.6 | 10127 | 7.844e-04 | PASS | PASS |
| A6000 | CUDA0 | 32 | 30 | 30 | 1.8 | 18286 | 4.615e-03 | PASS | PASS |
| A6000 | CUDA0 | 64 | 30 | 30 | 2.1 | 29907 | 3.593e-03 | PASS | PASS |

Row-permutation + one-row-perturbation cross-bleed test: not run as a dedicated pass
in H0 (existing modes have no permute/perturb path without editing layersplit.cpp).
The batched-vs-serial replay is the equivalent bleed detector here (flat-low rel_L2);
a dedicated permute/perturb pass will run in the H1 harness. Do not put an isolated
`[K,B]` projection in this table.

Findings: HTP dominates GPU decode 2.6-8x on both phones (GPU is attention-bound; the
GPU B=1->2 jump is a kernel-path switch). HMX selection boundary at B=5 confirmed on
both phones (B=4 HVX -> B=5 HMX step). Phones are 8-23x slower per layer than the
A6000, which is near-flat over batch (1.3->2.1 ms) -> phones cannot win on throughput,
only energy. Implication for H1: GPU intact-decode is ~3x slower but that includes
explicit attention; H1 isolates the FFN gate GEMV (the fair complementary-engine test),
and any useful output-row split will be heavily HTP-weighted.

## H1 Output-Row Split

`workload=batch_shaped_projection`, `M_kernel=B`, tensor `blk.2.ffn_gate.weight`
[K=3840,N=15360] F16, split along `ne1=N` (HTP rows [0:n_h], GPU rows [n_h:N]),
host row-merge, no reduction. `complete p50` = fanout + concurrent-proj wall + merge.
Storage=private (1x total, zero duplication -- satisfies "no weight duplication").
Binary `llama-phone-microop`; raw `scratchpad/h1_*.jsonl`, stderr `h1_logs/`.
These are NOT batch-decode results (isolated projection screen only).

Best interior split per M (HTP-share that minimizes complete p50; it is 75% at every
useful M -- the GPU is 2-3x slower so completion balances near 75/25):

| Device | xmem | M | HTP% | intact best p50 ms | split complete p50 ms | speedup | imbal | eff GBps split/intact | proj rel-L2 | gate |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| OP12 | off | 5 | 75 | 5.93 | 6.34 | 0.94x | 0.23 | 18.6/19.9 | 2.8e-4 | FAIL |
| OP12 | off | 8 | 75 | 6.17 | 6.48 | 0.95x | 0.27 | 18.2/19.1 | 2.8e-4 | FAIL |
| OP12 | off | 32 | 75 | 6.13 | 7.46 | 0.82x | 0.27 | 15.8/19.3 | 2.8e-4 | FAIL |
| OP12 | on  | 16 | 75 | 5.77 | 5.60 | 1.03x | 0.08 | 21.1/20.5 | 9.0e-3 | FAIL (rel-L2>5e-3) |
| OP12 | on  | 32 | 75 | 6.18 | 6.34 | 0.98x | 0.02 | 18.6/19.1 | 9.1e-3 | FAIL |
| OP15 | off | 8 | 75 | 5.98 | 8.23 | 0.73x | 0.41 | 14.3/19.7 | 2.8e-4 | FAIL |
| OP15 | off | 64 | 75 | 6.15 | 15.71 | 0.39x | 0.65 | 7.5/19.2 | 2.8e-4 | FAIL |
| OP15 | on  | 16 | 75 | 6.20 | 7.02 | 0.88x | 0.24 | 16.8/19.0 | 9.0e-3 | FAIL |
| OP15 | on  | 32 | 75 | 6.38 | 7.84 | 0.81x | 0.27 | 15.0/18.5 | 9.1e-3 | FAIL |

Full 8-M x 5-split matrix x {xmem off,on} x 2 phones (160 rows) is in the JSONL.
Zero complete rows meet the H1 latency and p95 gate. Correct xmem-off rows lose,
and OP15 also loses at the raw compute boundary.

Audit correction for xmem-on: the interior merged `split_rel_L2` values at
M>=16 are approximately 1.219, 0.995, and 0.707 for 25/50/75 percent HTP. The
static xmem prepack cache is keyed only by a recyclable OpenCL allocation handle
and offset, while the harness frees and reallocates private slices between
configurations. These values match reuse of the wrong packed weight rows. The
reported ~9e-3 value is only intact GPU-versus-HTP projection error; it does not
validate the split.

OP12 xmem raw concurrent wall was directionally faster than intact HTP at M=16
(1.166x) and M=32 (1.223x), but host fanout/merge erased the complete result and
the corresponding outputs are invalid. This is neither a pass nor a conclusive
platform fail. The fullcopy/shared-parent controls, derived prepack bytes, and
resident-memory sub-gate were not measured.

## H2 Equal-Work Comparison

| Device | B | Mode | HTP work | GPU work | Complete p50 ms | p95 ms | Useful rows/s | Traffic amp | Peak RSS MiB | Verdict |
|---|---:|---|---|---|---:|---:|---:|---:|---:|---|

Modes are `single`, `row`, `branch`, and `streams`. Keep total B/useful output
constant across a comparison block.

## Correctness

H0 (intact decode; no split, so projection/split columns n/a). Decode rel-L2 =
batched B-way vs per-sequence serial replay on the same engine. Sequence isolation =
flat-low rel-L2 across B (bleed would grow with B) + 0/ B argmax mismatch at every point.

| Device | Backend | B range | Mode | Decode rel-L2 (min..max over B) | Argmax mismatch | Sequence isolation | Result |
|---|---|---|---|---|---|---|---|
| OP12 | HTP | 1..32 | intact FA | 0 .. 5.35e-4 | 0 / B all B | PASS (flat, no bleed) | PASS |
| OP12 | GPU | 1..64 | intact no-FA | 0 .. 2.16e-5 | 0 / B all B | PASS | PASS |
| OP15 | HTP | 1..64 | intact FA | 0 .. 5.15e-4 | 0 / B all B | PASS | PASS |
| OP15 | GPU | 1..64 | intact no-FA | 0 .. 2.16e-5 | 0 / B all B | PASS | PASS |
| A6000 | CUDA0 | 1..64 | intact | 0 .. 4.6e-3 | 0 / B all B | PASS | PASS |

H1 xmem-off split/repeat/sentinel checks pass. H1 xmem-on interior split
correctness is invalid for the stale-cache reason documented above.

## H3 Scheduling Replay

| Device | Workload | Policy | Makespan ms | Critical idle % | Rows/s | TTFT p95 | TPOT p95 | Max queue | Result |
|---|---|---|---:|---:|---:|---:|---:|---:|---|

## Memory, Thermal, and Energy

Governor `uag` on both phones; DDR devfreq at idle floor 547 MHz (`bus_dcvs/DDR/cur_freq`)
throughout -- a single-layer decode does not push the coarse DDR-freq proxy off its floor.
Peak RSS / free RAM not sampled per-run in H0 (deferred to H1, which needs them for the
storage-control byte accounting). J/token intentionally empty (USB-powered; see NEXT_PLAN).

| Device | Mode | B | Note |
|---|---|---:|---|
| OP12 | HTP intact | 64 | FAIL: `HTP0 buffer mapping failed domain_id 3 size 2147487744` (2 GiB). KV cache allocates cells for all 48 model layers regardless of the 1-layer partial load (known KV-over-allocation bug); MBUF=3072 did not help. |
| OP15 | HTP intact | 64 | OK (49.1 ms/round) -- more RAM absorbs the 48-layer KV over-allocation. |

USB-powered battery telemetry is diagnostic only. Leave J/token empty until the
physical measurement boundary in `NEXT_PLAN.md` is available.

## Raw Artifacts

H0 (all under `scratchpad/`, i.e. the session scratchpad, not committed):

```text
h0_op12_htp.jsonl  h0_op12_gpu.jsonl  h0_op15_htp.jsonl  h0_op15_gpu.jsonl  h0_a6000.jsonl
h0_all.jsonl                          (concatenation of the above)
h0_logs/h0_<tag>_B<N>.log             (full dualengine stderr per B, incl. correctness lines + n_past)
h0_run.sh                             (runner; no repo edits)
h0_op12_console.txt  h0_op15_console.txt
```

Run command (per point): on-device
`timeout 200 env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. LLAMA_LAYER_START=2 LLAMA_LAYER_END=3
GGML_HEXAGON_MBUF=3072 [GGML_DECODE_NO_FA=1 for GPU] ./llama-layersplit -m 12b-f16-mid-2-3.gguf
--mode dualengine --dev-decode <HTP0|GPUOpenCL> --dev-prefill <other> --prompt-len 1 -b B -n 30`.
A6000: `build-cuda/bin/llama-layersplit ... --dev-decode CUDA0 --dev-prefill CPU`.

H1 (llama-phone-microop; source `examples/layersplit/microop.cpp`, target added to
`examples/layersplit/CMakeLists.txt`, driver `spikes/s3_microop_schedule/sweep.sh`):

```text
h1_op12_private_xmemoff.jsonl  h1_op12_private_xmemon.jsonl
h1_op15_private_xmemoff.jsonl  h1_op15_private_xmemon.jsonl   h1_all.jsonl (160 rows)
h1_logs/h1_<tag>_stderr.log
```

Host build: `cmake -B build-microop -DLLAMA_BUILD_EXAMPLES=ON -DGGML_HEXAGON=OFF
-DGGML_OPENCL=OFF && cmake --build build-microop --target llama-phone-microop` (CPU smoke:
projL2=0, splitL2=0, sentinel ok). Android build: docker `snapdragon-toolchain-hostgcc:v0.3`,
`cmake --preset arm64-android-snapdragon-release -B /build-out && cmake --build /build-out
--target llama-phone-microop`. Device run (per storage x xmem):
`LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_MBUF=3072 [GGML_OPENCL_ADRENO_XMEM_GEMM=1
GGML_OPENCL_XMEM_PREPACK_CACHE=1] ./llama-phone-microop --model 12b-f16-mid-2-3.gguf
--backends HTP0,GPUOpenCL --M 1,2,4,5,8,16,32,64 --split 0,25,50,75,100 --storage private
--warmups 5 --iters 20 --min-secs 1.5 --out h1.jsonl`.

H1 correctness: xmem-off projection and merged split rel-L2 are below 3e-4,
repeat rel-L2 is zero, and useful-B sentinels pass. Xmem-on interior merged
outputs at M>=16 are invalid because of stale prepack-cache reuse.

## Decision

H0 provides the static batch baseline. H1 has no passing complete row and no
integration is authorized. Disposition by path:

```text
OP15 output-row split:       FAIL
OP12 xmem-off row split:     FAIL
OP12 xmem-on interior split: INVALID / inconclusive
```

The output-row mechanism is archived rather than repaired because it is no
longer the research target. S4 asks a distinct operator-type, multi-stream
decode and energy question. Do not use H1 to claim that every form of
HTP/OpenCL operator pipelining is non-viable.
