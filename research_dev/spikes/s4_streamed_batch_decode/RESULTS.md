# S4 Multi-Stream Batch Decode Results

Status: V0 Checkpoint 1 (graph-cut + capability + metadata inventory) COMPLETE and
source-verified, inspect-only, nothing edited. Checkpoint 2a (GPU fused-attention
veto probe) run on BOTH phones: both PASS for the SWA decode class (FA compiles +
runs on-GPU, no CPU fallback) — the historical OP12 veto is STALE. Per-op profiling
(rest of Checkpoint 2) pending review. See "V0 GPU Fused-Attention Veto" and
"V0 Checkpoint 1 Inventory" below.

Contracts:

- [PLAN.md](PLAN.md)
- [RELATED_WORK.md](RELATED_WORK.md)

## Build and Device Identity

```text
Git revision:
Dirty worktree:
Host/Android toolchains:
HTP skel OP12/OP15:
OpenCL driver OP12/OP15:
OpenCL profiling build:
Model and shard hashes:
Physical power instruments/calibration:
```

## Gate Summary

| Gate | OP12 | OP15 | Verdict |
|---|---|---|---|
| V0 GPU attention (SWA decode FA compile+place) | PASS on-GPU (non-split; split fails non-fatal) | PASS on-GPU (both variants) | no CPU fallback either phone |
| V0 offline >=1.20x ideal schedule (SWA) | expected FAIL (older GPU) not run | **FAIL** at C>=512 (0.82x/0.47x); pass only at C=32 | Adreno OpenCL decode-FA too slow + scales ~linearly with KV -> STOP |
| V1 attention, prefill, and handoff coherency | not run | not run | pending |
| V2 one-layer reentrant pipeline | not run | not run | pending |
| V2E whole-phone J/layer-token | not run | not run | blocked until physical boundary |
| V3-A SLO-constrained capacity | not run | not run | pending |
| V3-B matched service | not run | not run | pending |
| V4 whole-phone service J/completed-output-token | not run | not run | blocked until physical boundary |
| V4 gross fleet energy | not run | not run | blocked until physical boundary |

## V0 Checkpoint 1 Inventory (inspect-only, no device runs)

Source-read on git worktree; nothing edited. Two load-bearing claims verified
directly in source (not just via agent report): the `wo` straddle and the
head_dim=512 fallback.

### Graph cut (dense gemma-4 12B; MoE/per-layer-embd paths inactive)

```text
HTP-A(s,l): gemma4.cpp:296 retain inpL (residual); :299 input RMS norm;
            :312 Q proj; :317 Q norm; :320 Q RoPE;
            :327 K proj; :338 K norm; :344 K RoPE;
            :330-332 V proj (or reuse Kcur); :339 V rms_norm (weightless);
            export {Qcur, Kcur, Vcur}.
GPU-B(s,l): INTERIOR of the single build_attn(iswa) overload, llama-graph.cpp:2869:
            :2913/:2928 select base/SWA KV ctx + mask; :2919 cpy_k, :2925 cpy_v
            into GPU-owned KV; :2934 build_attn_mha -> ggml_flash_attn_ext (:2426);
            ends at kqv_out marker :2935.
HTP-C(s,l): wo output proj (CURRENTLY FUSED at llama-graph.cpp:2941-2942, one line
            past the kqv_out marker) -> must be relocated to HTP-C; then
            gemma4.cpp:365-367 attn post-norm; :370 residual; :429-439 FFN
            (gate/up/GELU/down, LLM_FFN_PAR); :442-444 FFN post-norm; :448 residual;
            :484 inpL=cur -> HTP-A(s,l+1).
```

CRITICAL STRADDLE (single, as PLAN.md:133-138 anticipated): `wo` is fused inside
build_attn at llama-graph.cpp:2941-2942, past the `kqv_out` marker at :2935
(VERIFIED in source). Realizing the S4 cut = return `kqv_out` (:2935) as GPU-B
output and move the `wo` mul-mat (:2942) into HTP-C. Also VERIFY `v_rot`
(llama-graph.cpp:2937-2939) and `k_rot` are null in steady decode (KV-shift only).

Mask + KV-cell indices are built HOST-side (build_attn_inp_kq_mask
llama-graph.cpp:26-43; filled in set_input :580-621 -> llama-kv-cache.cpp
set_input_k_idxs:1469 / set_input_v_idxs:1485 / set_input_kq_mask:1735, each
asserting a host buffer). Their build+upload time is X_META and is currently
unmeasured (see BLOCKED).

### Metadata (gemma-4-12B-it-f16, raw dump: gemma4_12b_meta.txt)

48 layers, PLAIN dense (embedding_length_per_layer_input=0, shared_kv_layers=0,
no MoE, tied lm_head). Two attention classes in a 5 SWA : 1 FULL repeat:

| Class | Layers | is_swa | window | n_head:kv | head_dim | V path | Seed |
|---|---|---|---|---|---:|---|---|
| SWA / sliding-window | 40 (0-4,6-10,...) | true | 1024 | 16:8 (GQA) | 256 | normal attn_v [3840,2048] | blk.2 |
| FULL / global | 8 (5,11,...,47) | false | full | 16:1 (MQA) | 512 | V-LESS: reuse K-proj, weightless rms_norm, no V-RoPE | blk.5 |

FFN identical on both: n_embd=3840, n_ff=15360, F16. A V0 pass on one class does
NOT extrapolate to the other (PLAN.md:219-220).

### Capability map (file:line in source)

- HTP (HTP-A/HTP-C ops): RMS_NORM (ggml-hexagon.cpp:2879, F32 only), MUL_MAT
  F16-weight x F32-act (:2706, dst F32, no repack for F16), ROPE (:3077, F32
  NEOX only), ADD (:2818) all SUPPORTED. HTP also has native fused FLASH_ATTN_EXT
  (:1970) for F16 K/V — but attention is assigned to GPU-B in S4, so this is only
  the intact-HTP baseline path.
- OpenCL (GPU-B): F16-weight MUL_MAT supported (ggml-opencl.cpp:5730);
  FLASH_ATTN_EXT gated by a head-dim table (:5836-5840) and type combo (:5853-5875).
  gemma Q-f32/K-f16/V-f16 matches is_f32_f16 (:5857).

### NEW FINDING — S4 attention offload covers at most the 40 SWA layers

The OpenCL FA `supported_dims` table (ggml-opencl.cpp:5836-5840, VERIFIED) is
`{40,64,80,96,112,128,(192,128),192,256}`. head_dim **256 is present** (SWA
class OK) but **512 is ABSENT** (FULL/global class) -> `dims_supported=false` ->
GPU-B fused attention **silently CPU-falls-back on all 8 global layers on BOTH
phones**. Consequences:
- S4 GPU-owned attention is viable only for the 40 SWA layers; the 8 global
  layers must stay on the intact-HTP path even where S4 passes.
- Any throughput/energy accounting must attribute 8/48 layers to intact-HTP.
- This is a hardware/kernel-table limit independent of the OP12 shuffle issue.

### Veto predictions (to confirm with device build logs at Checkpoint 2)

- OP12 (Adreno-750): PREDICTED VETO. (a) historical FA compile failure
  `sub_group_shuffle_xor` (talks.md:168, AGENT_HANDOFF.md:132-134) — in-tree
  mitigations exist (non-fatal FA compile ggml-opencl.cpp:3992-3997, subgroup
  probe :4625-4638) so the gemma F16 path *may* now compile, but a missing
  required variant HARD-ABORTS at GGML_ASSERT(kernel!=NULL) :12785 (not a silent
  fallback). Plus (b) head_dim=512 fallback on the FULL class. Confirm via raw CL
  build log; if FA fails or falls back, mark UNSUPPORTED and stop (no CPU
  substitution).
- OP15 (Adreno-840): proceeds, but limited to the 40 SWA layers by the same
  head_dim=512 table; no documented compile veto.

### V0 gate deliverability

- Ideal compute-only >=1.20x gate (uses only H_A, H_C, G_B, A_H): FULLY
  DELIVERABLE at Checkpoint 2 with no source edit. Steady-state binding
  inequality reduces to `G_B < H_A + A_H + H_C` (since A_H>=0).
- Bounded >=1.15x gate (adds measured handoff + metadata time): BLOCKED — see below.

### BLOCKED (each needs a reviewed measurement-only edit; none applied)

- X_AG (A->B QKV) and X_GC (B->C kqv_out) copy+fence TIME: no A/B boundary in the
  intact graph. Smallest edit: a standalone handoff microbench (new file under
  examples/layersplit, touches no gemma4.cpp/llama-graph.cpp/KV/sched) that
  rpcmem-allocates the exact byte sizes, imports into OpenCL (QCOM ext-host-ptr),
  and brackets import+fence with host clocks. Bytes are derivable now.
- X_META upload TIME: bracket set_input (llama-graph.cpp:580-621) with two host
  timestamps — but this touches llama-graph.cpp, which PLAN.md:137-138 forbids in
  gates 1-2 -> PROPOSE-and-STOP, do not apply.
- KV-store vs fused-FA split of G_B: BLOCKED if the OpenCL profiler cannot
  separate cpy_k/cpy_v from FLASH_ATTN_EXT events; report aggregate G_B if so.

Checkpoint 1 STOPS here (PLAN.md:695-697). No S4 runtime code and no measurement-only
edit proceeds until reviewed. The GPU-FA veto probe below was the one device slice
explicitly authorized ("FA-veto check first") and uses only existing binaries.

## V0 GPU Fused-Attention Veto (device, Checkpoint 2a)

Question (PLAN.md:283-296, 324-326): does GPU fused attention compile and run for
the SWA class (head_dim 256) with NO CPU fallback on each phone? Method: existing
`llama-layersplit` binary, no new code. Force the batched decode onto the Adreno GPU
with FA enabled and watch scheduler placement + the OpenCL build log:

```text
LLAMA_LAYER_START=2 LLAMA_LAYER_END=3 GGML_SCHED_DEBUG=2 \
./llama-layersplit --mode dualengine --dev-decode GPUOpenCL --dev-prefill CPU \
  -m 12b-f16-mid-2-3.gguf --prompt-len 4 -b 8 -n 2
```

Shard `12b-f16-mid-2-3.gguf` = blk.2 (SWA, GQA 16:8, head_dim 256). Q is F32,
K/V F16 -> OpenCL is_f32_f16 FA path. Result:

| Phone (SoC/GPU) | FA kernel compile | FLASH_ATTN_EXT placement | graph splits | Correctness (batched B=8 vs serial) | Verdict |
|---|---|---|---|---|---|
| OP12 CPH2583 / Adreno 750 | `fa f32_f16` OK; **`fa f32_f16 split` FAILS** (`sub_group_shuffle_xor` implicit-decl, flash_attn_f32_f16.cl:239) -- **NON-FATAL** | node #27 FLASH_ATTN on `[OpenC]`, all inputs `[OpenC]` | **1** (no CPU split) | PASS rel_L2 1.806e-5, max|d| 2.7e-4, argmax 0/8 | **PASS (SWA decode)** |
| OP15 CPH2749 / Adreno 840 | `fa f32_f16` OK **and** `fa f32_f16 split` OK (no error) | node #27 FLASH_ATTN on `[OpenC]` | **1** (no CPU split) | PASS rel_L2 1.806e-5, argmax 0/8 | **PASS (clean)** |

Both EXIT=0, no `GGML_ASSERT(kernel!=NULL)` abort. Raw logs:
`scratchpad/fa_veto_op12_swa_decode.log`, `scratchpad/fa_veto_op15_swa_decode.log`.

Findings:
- **The historical OP12 veto (talks.md:168, AGENT_HANDOFF.md:132-134) is STALE for
  the decode-FA path.** In-tree mitigations (non-fatal FA compile
  ggml-opencl.cpp:3992-3997) skip the failing split variant; the non-split
  `f32_f16` kernel runs the whole SWA decode attention on the Adreno 750 GPU with
  no CPU fallback and correct output. Neither phone is vetoed for the SWA class.
- **Caveat (OP12 only) -- RESOLVED to 512-token context.** The split (`N_SPLIT>1`)
  FA variant still fails to compile, but a follow-up probe put BOTH a 512-token GPU
  prefill AND a 100-round decode on the Adreno 750
  (`--dev-decode GPUOpenCL --dev-prefill GPUOpenCL --prompt-len 512 -b 8 -n 100`,
  raw log `scratchpad/fa_veto_op12_prefill512_decode100.log`): 880 FLASH_ATTN ops
  ALL on `[OpenC]`, 0 on CPU, `graph splits = 1` on both contexts, correctness PASS
  (rel_L2 1.8e-5), EXIT=0. The failing split variant is **never actually required** --
  the non-split kernel serves prefill (n_q=512) and decode alike. So OP12 GPU-B is
  viable up to 512-token context in both phases. Only C=1024 (top of the SWA window)
  remains unconfirmed; no cliff is expected since non-split served 512 cleanly.
- **FULL/global class (head_dim 512) NOT runtime-tested** (no blk.5 shard on device;
  all on-device shards start at blk.2 SWA). Source is conclusive: head_dim 512 is
  absent from the OpenCL supported_dims table (ggml-opencl.cpp:5836-5840, VERIFIED)
  -> GPU-B fused attention CPU-falls-back on all 8 global layers on BOTH phones.
  S4 GPU-owned attention therefore covers at most the 40 SWA layers.

## V0 Offline Schedule Bound -- RESULT (op15 SWA, blk.2)

Measured with a new inspect-only harness `examples/layersplit/oplayerprof.cpp`
(intact single-layer (B,C) decode via the public llama API; per-op decomposition
from GGML_HEXAGON_PROFILE and a GGML_OPENCL_PROFILING build; no graph/KV/scheduler
edits). H_A/A_H/H_C from the HTP per-op profiler; G_B (GPU KV-store + fused FA) from
cl_profiling.csv decode rows. Ideal speedup = wall / max(H_A+H_C, G_B), i.e. HTP does
A+C while the Adreno owns attention. Gate: >=1.20x at B=16 AND B=32 for the same C.

Raw: scratchpad/v0/op15_htp_swa_{off.jsonl,prof.log}, gb_op15_B*_C*.csv,
GATE_op15_swa.txt.

| B | C | intact HTP wall ms | attn% (A_H) | ceiling | H_A+H_C ms | G_B (Adreno FA) ms | ideal speedup | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 32 | 20.0 | 22.8% | 1.30 | 15.5 | 7.7 | **1.30** | pass (degenerate C) |
| 16 | 512 | 20.7 | 23.3% | 1.30 | 15.9 | 25.4 | **0.82** | FAIL |
| 16 | 1024 | 23.3 | 32.2% | 1.48 | 15.8 | 44.2 | **0.53** | FAIL |
| 32 | 32 | 25.6 | 37.0% | 1.59 | 16.1 | 17.2 | **1.49** | pass (degenerate C) |
| 32 | 512 | 26.5 | 37.8% | 1.61 | 16.5 | 56.7 | **0.47** | FAIL |

**VERDICT: FAIL at realistic decode context (op15 SWA).** The operator-type split
clears >=1.20x only at C=32 (near-empty KV). Root cause: the Adreno OpenCL decode
flash-attention kernel is slow and scales ~linearly with KV length -- G_B at B=16
goes 7.7 -> 25.4 -> 44.2 ms as C goes 32 -> 512 -> 1024 -- whereas the Hexagon HMX
flash-attention handles the same growth in ~1.6x (A_H 4.6 -> 7.5 ms) and HTP does the
ENTIRE rest of the layer in ~15-16 ms flat. So moving attention off HMX onto Adreno
replaces a cheap, well-scaling op with an expensive, poorly-scaling one; at any C the
real decode operates in (>=256-512), GPU attention becomes the bottleneck and the
"pipeline" is SLOWER than intact HTP (down to 0.47x). Both required batches (16 and 32)
fail at C>=512.

Generality: op15 is the Adreno 840 (newer/faster GPU); op12/Adreno 750 is expected no
better (older GPU, and its FA split-variant does not even compile). The FULL class
(8 layers) cannot run GPU attention at all (head_dim 512 fallback). So the S4
operator-type split is non-viable on both layer classes.

Per PLAN.md stop rule ("stop before implementation if the same S>=2 schedule is below
1.20x ideal at B=16 or B=32 for the same class and C"): **STOP. Do not proceed to
S4-V1/V2 or any graph integration.** op12 confirmation and the co-run/handoff terms are
moot once the solo ideal bound already fails.

## V0 Per-Operator Profiles

| Run ID | Device | Layer/type | B | C/C_eff | Backend | Mode | Profile on/off | FA/HMX path | H_A ms | attn/KV ms | H_C ms | Layer ms | Correctness | Free RAM MiB | Result |
|---|---|---|---:|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|

## V0 Handoff, Metadata, and Interference

| Run ID | Device | Layer/type | B/C | S/D | QKV bytes/ms | kqv_out bytes/ms | Mask+descriptor bytes/setup/upload ms | HTP slowdown | GPU slowdown | Profile wall perturbation | Result |
|---|---|---|---|---|---|---|---|---:|---:|---:|---|

## V0 Bandwidth Provenance

| Run ID | Device | Source | Counter/tool | Domain | Units/semantics | Access | Sampling/alignment | Raw artifact | Valid |
|---|---|---|---|---|---|---|---|---|---|

Use only `ddr_counter`, `effective_min_weight_read`, or `none` as Source. Never
place a useful-byte/time estimate in a physical DDR-counter column.

## V0 Offline Schedule Bound

| Run ID | Device | Layer/type | B | C/C_eff | S/D | Groups | Policy | Ideal ms | Bounded ms | HTP ms | Ideal speedup | Bounded speedup | All dense HMX | Logical traffic amp | Result |
|---|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|

## V1 GPU Attention and Prefill Correctness

| Run ID | Device | Phase | Tokens/B | C/C_eff | FA path | kqv_out rel-L2 | KV rel-L2 | Old KV unchanged | Next decode | Permutation/isolation | Repeat rel-L2 | Result |
|---|---|---|---|---|---|---:|---:|---|---|---|---:|---|

## V1 Activation Handoff

| Run ID | Device | Direction | Mode | Bytes | Offset/slot cases | p50 ms | p95 ms | p99 ms | Errors/iterations | Peak RSS MiB | Result |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---|

## V2 Equal-Work One-Layer Results

| Run ID | Device | Layer/type | B | C/C_eff | S/D | Mode | Groups | Layer-tokens/s | Round p50/p95/p99 ms | Group p95 ms | Fill/drain ms | HTP/GPU slowdown | Rel-L2 | All dense HMX | Free RAM MiB | Result |
|---|---|---|---:|---|---|---|---|---:|---|---:|---:|---|---:|---|---:|---|

Paired block-level ratios, not per-round samples, are the confidence-interval
units.

## V2 Gate Calculations

| Pair set ID | Device | Layer/type | B/C/S/D | Candidate | Control | Paired blocks | Geomean rate ratio | Ratio CI lower | p95 round ratio | Result |
|---|---|---|---|---|---|---:|---:|---:|---:|---|

## V2 Prefill-to-Decode Lifecycle

| Run ID | Device | Layer/type | B | Prompt length | Decode rounds | Mode | Prefill ms | Decode ms | Lifecycle layer-tokens/s | Output/KV correctness | Ratio vs HTP | Result |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|---|

## V2E Local Phone Energy

| Pair/Run ID | Device | Layer/type | B/C/S/D | Mode | Duration s | Layer-tokens | Whole-phone J | J/layer-token | Paired ratio | Ratio CI upper | Correctness pass | Result |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|

## Scheduler Trace

| Run ID | Task | Layer | Group | Slot/gen | Backend | Ready ms | Start ms | End ms | Queue age ms | HMX/HVX | Handoff/metadata bytes |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|---:|

## V3 Service Results

| Run ID | Experiment | Topology/mode | Offered rate | Max sustainable rate | B target | Completed/failed | Completed output tokens/s | TTFT p50/p95 | TPOT p50/p95 | Queue p95 | Thermal decay | Result |
|---|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---|

V3-A records SLO-constrained capacity. V3-B uses one matched sub-capacity trace;
equal completed throughput is expected and is not a gain gate.

## V3-A Capacity Gate Calculations

| Pair set ID | S4 mode | Primary non-S4 control | Paired sweeps | S4 max rate | Control max rate | Rate ratio | Ratio CI lower | Both SLO-valid | Result |
|---|---|---|---:|---:|---:|---:|---:|---|---|

## V4 Service Energy Results

| Pair/Run ID | Boundary | Topology/mode | Phone state delta | Completed output tokens | Duration s | Gross J | Gross J/completed token | Idle-adjusted J/token | V3-B service pass | Paired ratio | Ratio CI upper | Result |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|

Gross fleet energy is primary. Idle-adjusted and NVML/backend attribution are
diagnostics. Record phones-disconnected A6000-only and powered-idle-phone
deployment controls separately.

## Memory and Thermal

| Run ID | Device | Mode | B/S/D/C | Native weights MiB | Derived prepack MiB | GPU KV MiB | HTP KV MiB | Activations MiB | Peak RSS MiB | Free RAM MiB | Start/end C | Clock drift | Result |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|

Any nonzero persistent HTP KV for an S4 layer must be explained and fails the
one-KV-allocation gate unless it is an explicitly measured control.

## Raw Artifacts

List stable paths and checksums for task JSONL, per-op profiles, backend logs,
clock/thermal traces, raw power samples, request trace, exact commands, and
build output. Do not rely only on an ephemeral agent scratchpad.

## Decision

Pending V0 data.
