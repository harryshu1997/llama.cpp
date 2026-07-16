# S5-V0 Operator-Affinity and Additive-Capacity Screen — PLAN

Status: EXECUTING. Inspect-only w.r.t. production; no Gemma graph / KV / scheduler /
serving-path edits. Existing uncommitted edits in `layersplit.cpp`, `microop.cpp`,
`oplayerprof.cpp`, `ggml-hexagon.cpp` are preserved untouched.

## Question

Which Gemma-4 12B **FP16-weight / FP32-activation** operator islands are useful on
OP12 / OP15 when the phones are treated as **additional backends** for an A6000
server, under continuous multi-stream work and server batches near or above the
A6000 capacity limit? This is an **operator-affinity + additive-capacity** screen,
not scheduler integration and not intra-operator splitting.

This is a distinct question from S3 and S4, whose negatives are NOT re-litigated:

- S3 output-row (N) splitting FAILED on both phones -> not retested.
- S4 Adreno attention offload FAILED at realistic KV -> not retested.
- No K/column split, QKV-branch split, or isolated lightweight-op offload.

S5 asks the opposite of S3/S4: not "can we split one operator across two engines on
one phone" but "does a phone running **whole, unsplit** operator instances from
independent ready streams add usable concurrent throughput to the A6000 fleet."

## Method (existing tools, no source changes)

1. **Export exact graph ops.** `test-export-graph-ops` (built from current source)
   on the real `gemma-4-12B-it-f16.gguf` emits every unique op node of the real
   decode/prefill graph (op, dst type, ne, op_params, sources incl. F16 weight
   types, name). Driven at several `(-np, -ub, -c)` to sweep M and B.
2. **Consume via `--test-file`.** `test-backend-ops` (current source) rebuilds each
   op verbatim as an isolated single-op graph and runs it **directly on the chosen
   backend** (`ggml_backend_graph_compute(backend, gf)`); there is **no per-op
   scheduler**, so "not supported" is the only fallback signal (no silent CPU
   fallback path exists in this harness).
3. Three modes per case: `support` (`ggml_backend_supports_op`), `test`
   (correctness vs CPU reference via `ggml_backend_compare_graph_backend`), `perf`
   (warm-up, replicate op to ~fixed FLOP/byte budget, loop >=1 s, report per-op
   `time_us`). Output `--output sql` (CSV omits timing).

## Backends / devices

| Tag | Device | Backend dev name | Identity |
|---|---|---|---|
| cuda0 | RTX A6000 (host) | CUDA0 | driver 580.159.03, 48 GB |
| op15 HTP0 | OnePlus 15 (CPH2749/SM8850) | HTP0 | Hexagon v81, 8 thr/8 hvx/1 hmx/8MB vtcm, skel libggml-htp-v81.so |
| op15 GPU | OnePlus 15 | GPUOpenCL | Adreno 840 |
| op12 HTP0 | OnePlus 12 (CPH2583/SM8650) | HTP0 | Hexagon v75, 4 thr/4 hvx/1 hmx/8MB vtcm, skel libggml-htp-v75.so |
| op12 GPU | OnePlus 12 | GPUOpenCL | Adreno 750 |

CPU is the correctness reference only (harness auto-uses CPU backend as backend2).

## Operator matrix (all F16 weight / F32 act, from the real graph)

Dense MUL_MAT cells (keyed by weight ne `[K,N]`), M = {1,5,16,32,128,512}:

| Cell | weight [K,N] | out [N,M] |
|---|---|---|
| SWA Q | [3840,4096] | [4096,M] |
| SWA output | [4096,3840] | [3840,M] |
| FFN gate/up | [3840,15360] | [15360,M] |
| FFN down | [15360,3840] | [3840,M] |

Attention FLASH_ATTN_EXT, decode (n_q=1), n_seq(B)={16,32}:

| Class | head_dim | n_head:kv | seed |
|---|---|---|---|
| SWA | 256 | 16:8 GQA | blk.2 |
| FULL | 512 | 16:1 MQA (V-less) | blk.5 |

Lightweight controls, M={32,512}: RMS_NORM `[3840,M]`, ROPE `[256,16,M]`,
ADD `[3840,M]`, GEGLU(GLU op) `[15360,M]`.

## KNOWN TOOL LIMITATION (documented; C-sweep gated for review)

`test-export-graph-ops` reserves the decode graph on an **empty** KV cache, so the
FA op's `n_kv` is pinned to the 256-cell FA pad **regardless of `-c`**. Decode
attention (n_q=1) is therefore only reachable at logical **C~=256**; C={32,512,1024}
are NOT reachable with the unmodified tool (C=32 is below the pad floor; C=512/1024
need a filled KV the tool never creates; larger n_kv only appears in prefill graphs
where n_q>1, a different op class). Per the spike rule, the C-parameterized decode
attention sweep is **BLOCKED pending a reviewed measurement-only enabler** (smallest
options: (a) a `--kv-fill C` decode-reserve knob in test-export-graph-ops, or (b) a
test-file generator that rewrites only the K/V/mask KV-length of the real decode FA
op). Everything else runs on the unmodified tool.

## Perf protocol

7 separate processes per device-backend; **discard round 0**; median + range over
rounds 1-6. Device order rotated per round (interleave). Thermal/clocks snapshot per
run. op12 HTP0 excludes standalone RMS_NORM/ROPE (v75 DSP-queue crash, see RESULTS).

## Offline estimates (from measured primitives; NOT fused-graph measurements)

- FFN island: `T_FFN = T_norm + 2*T_gate_up + T_GEGLU + T_down + T_residual`
- Attention island: norm + Q/K/V + RoPE + KV-store + FA + output-proj + residual
- Full layer: attention island + FFN island
- Per backend: island latency, ops/s, effective weight bytes/s, boundary bytes,
  phone/A6000 ratio.

## Additive-capacity bound

`R_ideal = R_CUDA + R_OP15_HTP + R_OP12_HTP`; `ideal_gain = R_ideal / R_CUDA`.
Evaluate B_total = {128,256,320,384}. For B_total<=256 compare to the A6000's best
combined batch; for B_total>256 keep A6000 at 256 and assign overflow to phones;
compare to CUDA queuing identical total work. Add a transport-adjusted bound using
residual `[3840,M]` F32 and F16 payloads with measured USB/TCP one-way + round-trip
(weights resident, excluded from per-op traffic).

## Decision gates (shortlist an island only if ALL hold)

correctness passes, no CPU fallback; timing CoV <= 5%; transport-adjusted 3-device
goodput >= 1.10x CUDA-only for equal work; positive for B_total>256; >=15% phone RAM
free. Energy: **BLOCKED** (USB-powered; cannot measure phone J/op) -> next gate needs
unplugged WiFi-ADB or calibrated physical power.

## Outputs

`PLAN.md`, `RESULTS.md`, raw artifacts under `scratchpad/s5_operator_affinity/`.
STOP after the S5-V0 verdict; no FFN-island harness or scheduler until the measured
offline bound passes review.
