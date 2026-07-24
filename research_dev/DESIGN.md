# Historical Design A - 3-Stage Pipeline-Parallel Phone-Offload LLM Serving

> **STATUS: HISTORICAL SUBSTRATE, NOT THE CURRENT IMPLEMENTATION CONTRACT.**
> The primary target is now the executable phone warm tier in
> [ACTIVE_WARM_TIER_DESIGN.md](ACTIVE_WARM_TIER_DESIGN.md). Read that file,
> [NEXT_PLAN.md](NEXT_PLAN.md), and the top of [talks.md](talks.md). Design A is
> retained as route `A0`: a proven Gemma pipeline and source of transport,
> sharding, local-KV, persistent-worker, and continuous-batch mechanisms. The
> timestamped material below is preserved for design intent and experimental
> provenance; do not execute its old next-step statements as the live roadmap.

*System design plan · Gemma-4 12B (fp16) · A6000 → OP15 → OP12 → A6000 · drafted 2026-07-06*

> **How to read this.** This document is the technical design. The phased build plan, de-risk spikes, and repo layout live in [MILESTONES.md](MILESTONES.md); a one-page orientation is in [README.md](README.md). It was synthesized from an 8-subsystem design pass with an adversarial feasibility review — several subsystem agents read the actual llama.cpp source (`gemma4.cpp`, `llama-model.cpp`, `ggml-rpc`, the hexagon/opencl backends), so the constraints below are code-grounded, not assumed.

> **Measurement correction (2026-07-10).** Treat every "zero interference"
> statement below as the original design hypothesis. `dualengine` proves that
> HTP and OpenCL work can overlap in wall time, but it did not record equivalent
> solo leg times and therefore does not isolate co-run slowdown. S3-H0 later
> measured that limitation and is retained in the archived spike results. Static
> HTP decode at B=16/B=32 was proven; dynamic service is tracked by the current
> mixed-workload plan rather than this historical document.

> **Two corrections that override the intuitive picture — read first:**
>
> 1. **Phones hold the *first* layers, the A6000 is the *terminal* stage (R10).** llama.cpp pins `lm_head`+sampler (`dev_output`) to the last device in the order, so to keep the 2 GB head + 256k-vocab softmax + sampler on the A6000 we order devices `[op15, op12, A6000]`. The phones therefore own layers `0..2` and the server owns `embed + layers 3..47 + norm + lm_head + sampler`. The device *visit order* is still `server(embed) → OP15 → OP12 → server(rest+head+sample)` as drawn — the server simply appears at both ends of the loop.
> 2. **The one genuinely unproven risk is batched decode on the phone NPU (R1).** Op-level support exists, but `n_parallel=2` was recorded to *hang* on op15. We therefore take **GPU-decode / NPU-prefill-only** as the baseline and treat batched NPU decode as upside to be earned in spike **S1**. Everything else (transport, partitioning, provisioning) checked out as feasible or feasible-with-caveats.

> **Honest framing.** A prior review preferred hub-spoke *Design B* (replication + router). We build **A per direction** and surface its costs rather than bury them — see the Executive Summary and the Risk Register.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Architecture at a Glance](#architecture-at-a-glance)
- [1. Topology, Layer Partitioning & Dataflow](#1-topology-layer-partitioning-dataflow)
- [2. Interconnect, Transport & Wire Protocol](#2-interconnect-transport-wire-protocol)
- [3. Weight Provisioning — Pre-Downloaded Shards](#3-weight-provisioning-pre-downloaded-shards)
- [4. On-Phone Single-Copy Weight Sharing (mmap + fastRPC + OpenCL)](#4-on-phone-single-copy-weight-sharing-mmap-fastrpc-opencl)
- [5. Continuous Batching & Prefill/Decode Scheduling](#5-continuous-batching-prefilldecode-scheduling)
- [6. Distributed KV Cache & Intra-Phone NPU/GPU Execution](#6-distributed-kv-cache-intra-phone-npugpu-execution)
- [7. Orchestration, Energy Accounting, Observability & Failure](#7-orchestration-energy-accounting-observability-failure)
- [Risk Register](#risk-register)
- [Consolidated Open Questions](#consolidated-open-questions)

---

## Executive Summary

**What we are building.** A layer-partitioned pipeline that offloads a slice of Gemma-4 12B (fp16) transformer layers from an RTX A6000 onto two Android phones (OnePlus 15 / Hexagon v81 + Adreno 840; OnePlus 12 / Hexagon v75 + Adreno 750), to save fleet energy per token. The A6000 holds token embeddings, the bulk of the layers, final norm, `lm_head`, the **sampler**, and the continuous-batch **scheduler/orchestrator**. Each phone holds a small contiguous block of layers and owns the KV cache for *its own* layers — **KV never crosses the wire**; only the `[n_embd, n_tokens]` residual stream does. Prefill (compute-bound) targets the phone **NPU/HMX**; batched decode (memory-bound GEMV) targets the phone **GPU/Adreno**; they run concurrently (measured zero interference).

**Baseline config (the plumbing bring-up, not the energy win).** OP15 = 2 layers, OP12 = 1 layer, A6000 = the remaining ~45 of ~48 layers. Weights are **pre-provisioned as local mmap'd shard files on each phone** (no runtime weight RPC — a hard constraint that rules out stock `ggml-rpc` weight upload; RPC is reused as the *activation* transport only). Wire dtype is the stock **F32** residual unless we add an explicit `ggml_cast`-to-F16 at each boundary. Device strings confirmed: NPU = `HTP0`, GPU = `GPUOpenCL`.

**What the baseline WILL achieve:** an end-to-end, correct 3-stage pipeline with distributed KV, proving the transport, the shard/loader path, the loop-back-to-server for `lm_head`+sampling, and the intra-phone NPU/GPU split. **What it will NOT achieve:** meaningful compute offload or energy savings — 3 of ~48 layers is ~6% of the model, and per-phone active energy may sit at or below the coulomb noise floor. Energy wins only appear after later milestones rebalance many SWA (window-capped) layers onto the phones, *and* only if batched on-phone decode is proven.

**The honest A-vs-B note.** A prior review preferred a hub-spoke "Design B" (replication + router). We are building **A per direction** — a chained pipeline — and we surface, rather than bury, its structural costs: (1) a well-batched A6000 already floors ~**0.15 J/tok** (fp16) / ~0.137 (q4) at batch 256; a 300 W always-hot server plus ~5 W of host-drawn USB rail per phone means the fleet J/tok almost certainly **does not beat the pure-server baseline at the 3-layer split**, and may not beat it until a large fraction of layers move to phones. (2) The single biggest threat is not transport or partitioning — both are code-confirmed feasible — but that **multi-sequence continuous-batched decode is UNPROVEN on the experimental Hexagon backend, with a recorded `NPU n_parallel=2 HANGS` on op15.** We therefore adopt **GPU-decode / NPU-prefill-only as the *baseline assumption* for phone decode**, and treat batched NPU decode as upside to be earned in an early de-risk spike, not as a premise. Build order is sequenced so we discover a fatal "no" (batched phone decode, single-copy mmap import) in week 1–2, before investing in the orchestrator.

---

## Architecture at a Glance

```text
PLACEMENT (who holds what)
  A6000 (terminal device, ~300W): token_embd (host CPU-forced) | layers 3..47 + their KV |
                                  final_norm | lm_head (~2GB) | SAMPLER | scheduler/orchestrator
  OP15 (v81, HTP0+GPUOpenCL):     layers 0..1 (mmap shard) + their KV (SWA-preferred, window-capped)
  OP12 (v75, HTP0+GPUOpenCL):     layer 2      (mmap shard) + its  KV
  NOTE: phones own the FIRST layers, and A6000 is ordered LAST, so dev_output pins
        lm_head+sampler back on the A6000 (llama-model.cpp:1303 pins output to terminal dev).

WIRE: only the residual stream [n_embd=3840, n_tokens] crosses each hop.
      Stock dtype = F32 -> 7.86 MB /512-tok-prefill, 1.97 MB /128-seq-decode.
      Add ggml_cast->F16 at each boundary to halve (3.93 MB / 0.94 MB). KV NEVER crosses.
      Out-of-band each step (tiny, mandatory): inp_pos + batch composition + seq->KV-slot map.

PREFILL (per request, streamed at 8 fps; compute-bound -> NPU/HMX)
  frame -> [A6000] embed + layers 3..47  --residual-->  [OP15 NPU] layers 0..1
        (physically OP15 -> HOST -> OP12; phones can't peer over USB)
             [OP15 NPU] --residual(via host relay)--> [OP12 NPU] layer 2
             [OP12] --residual--> [A6000] final_norm + lm_head + SAMPLE -> first token + KV built at every stage locally
  Result per request: distributed KV (each stage owns its layers') + first sampled token. Accumulate ready seqs.

DECODE (fire when ~64-128 seqs ready; continuous batching; memory-bound -> GPU/Adreno)
  ┌──────────────────────────────── loop per step ────────────────────────────────┐
  │ [A6000] embed(next tok ids) + layers 3..47  --residual[3840,B]-->                │
  │ [OP15 GPU] layers 0..1  --(host relay)-->  [OP12 GPU] layer 2  --residual-->     │
  │ [A6000] final_norm + lm_head(256k vocab) + SAMPLE  --> next-token ids  ──────────┤
  └── unified KV on each phone (kv_unified=TRUE, ne[3]==1, block-diagonal F16 mask) ─┘
  Cross-stage OVERLAP is NOT automatic (RPC/hexagon/opencl fail the async&&events gate)
  -> orchestrator drives k in-flight microbatches over thread-parallel BLOCKING RPC,
     separate prefill/decode sockets to avoid head-of-line blocking.

INTRA-PHONE (one physical weight copy, two engines):
  mmap'd fp16 shard --ION/dmabuf fd (rpcmem)--> NPU/HMX (prefill)  AND
                                    --dmabuf import--> Adreno GPU (decode)  [NEW OpenCL code]
  KV pool lives in phone-local DDR (ideally shared dma-buf for zero-copy prefill->decode handoff).
```

---

## 1. Topology, Layer Partitioning & Dataflow

### 0. Scope and ground rules

This section fixes **where every tensor of Gemma-4 12B lives** and **exactly what crosses each wire** in the 3-stage pipeline `server -> OP15 -> OP12 -> server`. It is deliberately conservative: the baseline offloads only **3 of ~48 transformer layers (~6%)** and is a *plumbing/correctness* milestone, **not** an energy win. Every layer count, hidden size, and head count below is either grounded in the llama.cpp `gemma4` implementation (`src/models/gemma4.cpp`, `src/llama-hparams.cpp`) or marked **confirm from gguf** — I did not have the 12B weights on disk (only `ggml-vocab-gemma-4.gguf`), so arch scalars must be re-read with `gguf_dump.py` on the real shard before you freeze the config.

### 1. Model facts that drive the partition (from the gemma4 backend)

Reading `src/models/gemma4.cpp` and `src/llama-hparams.cpp`, Gemma-4 is **not** a plain LLaMA block. Three features change how you are allowed to cut the pipeline:

1. **The residual stream is the only thing that flows layer-to-layer.** `res->t_layer_inp[il] = inpL` (gemma4.cpp:213) — the input to layer `il` is exactly the previous layer's output, a single `[n_embd, n_tokens]` tensor. **This is the pipeline payload.** Cut between layer `k` and `k+1` = ship `inpL` (the output of layer `k`). Nothing else about the transformer stack crosses a stage boundary.

2. **Shared-KV tail layers (the main partitioning trap).** `has_kv(il)` (llama-hparams.cpp:259) returns `true` only for `il < n_layer_kv_from_start`; the **last `n_kv_shared_layers` layers reuse the KV cache of earlier layers** (gemma4.cpp:269-274, the `else` branch runs attention with `Kcur=Vcur=nullptr`). `n_layer_kv_from_start = n_layer_all - n_kv_shared_layers` (gemma4.cpp:10). If the gguf has `n_kv_shared_layers > 0`, the **tail layers we are handing to the phones are exactly the ones that do not own their KV** — their attention needs K/V that physically lives in an earlier layer on the *server*. That silently violates the "each stage owns its layers' KV, KV never crosses the wire" invariant. **Mitigation is in §7 — this must be checked first.**

3. **Optional per-layer token embeddings & MoE.** The gemma4 graph conditionally builds `inp_per_layer = project_per_layer_inputs(inpL, ...)` when `per_layer_tok_embd` exists (gemma4.cpp:194-200), and layers may be MoE (`ffn_gate_inp != nullptr`, gemma4.cpp:291). These are features of the *effective/MatFormer* variants (E2B=35L, E4B=42L in the `n_layer()` switch at gemma4.cpp:22-28). A **dense 12B** shard almost certainly has `n_embd_per_layer == 0` and no experts, but **confirm from gguf** — if `n_embd_per_layer > 0`, every phone stage additionally needs the per-layer embedding input for its layers (see §5, "per-layer embedding wrinkle").

The `n_layer()` switch has **no 48-layer case** (30/35/42/60 map to 26B-A4B/E2B/E4B/31B); a 48-layer 12B loads as `LLM_TYPE_UNKNOWN`, which is cosmetic but tells you the exact "12B" arch scalars are **confirm from gguf** — treat `n_layer≈48`, `n_embd≈3840`, `n_head≈16`, `n_head_kv≈8`, `head_dim≈256`, `n_ff≈15360`, `n_vocab≈262144` as *working assumptions* below.

### 2. Sizing (working numbers, fp16)

Per-layer dense weight (formula, then value at the assumed scalars):

```
attn  = wq + wk + wv + wo
      = n_embd*n_head*head_dim + 2*(n_embd*n_head_kv*head_dim) + n_head*head_dim*n_embd
      = 3840*4096 + 2*(3840*2048) + 4096*3840                   ≈ 47.2 M params
ffn   = 3 * n_embd * n_ff = 3 * 3840 * 15360                    ≈ 176.9 M params
per-layer ≈ 224.1 M params  ->  ×2 bytes  ≈ 0.448 GB / layer (fp16)   [matches the "0.4-0.5 GB/layer" brief]
```

| Item | Size (fp16) | Notes |
|---|---|---|
| Per transformer layer | **~0.448 GB** | confirm `n_ff` from gguf; MoE layers differ |
| 48 layers total | ~21.5 GB | |
| tok_embd / lm_head | **~2.01 GB** | `3840×262144×2`; **tied** in gemma4 (output=NULL -> duplicates tok_embd, gemma4.cpp:44-50) so one physical 2 GB copy |
| Whole model | ~23.5 GB (~24 GB) | consistent with brief |
| **OP15 share (2 layers)** | **~0.90 GB** | trivially fits phone RAM |
| **OP12 share (1 layer)** | **~0.45 GB** | trivially fits phone RAM |

Phone weight shares are **~1%** of phone RAM (12-16 GB class) — the baseline is nowhere near a RAM limit; the limiter for rebalancing is thermal/throughput (§6), not capacity.

### 3. The concrete placement (baseline, L=48; every boundary is a config knob)

```
┌─────────────────────────── A6000 (server) ──────────────────────────┐
│ tok_embd (2 GB)                                                      │
│ transformer layers  0 .. 44   (45 layers, ~20.2 GB)                  │
│ output_norm  +  lm_head (tied, 2 GB)  +  final_logit_softcap         │
│ SAMPLER  +  continuous-batch scheduler/orchestrator                  │
│ [optional] vision encoder + mmproj  (§8)                             │
└─────────────────────────────────────────────────────────────────────┘
        │  ship inpL = output of layer 44   ([n_embd, n_tok] fp16)
        ▼
┌─────────────── OP15 (Hexagon v81 HMX / Adreno) ─────────────┐
│ transformer layers  45 .. 46   (2 layers, ~0.90 GB, mmap'd) │
│ owns KV for layers 45,46 (local, never shipped)             │
└──────────────────────────────────────────────────────────────┘
        │  ship inpL = output of layer 46
        ▼
┌─────────────── OP12 (Hexagon v75 / Adreno 750) ────────────┐
│ transformer layer   47          (1 layer, ~0.45 GB, mmap'd) │
│ owns KV for layer 47 (local, never shipped)                 │
└──────────────────────────────────────────────────────────────┘
        │  ship inpL = output of layer 47   (the final hidden state)
        ▼
   back to A6000:  output_norm -> lm_head -> softcap -> sample
```

**Placement rationale.** The lm_head (2 GB, `n_vocab≈262k`) and tok_embd are far too large for a phone and are needed *together with the sampler* at the loop-back point, so both embedding endpoints and the sampler stay server-side; that also makes the loop close naturally (the server owns *both* the front — embeddings + layers 0..44 — *and* the tail — norm + head + sampler), so the next-token embedding never has to be shipped anywhere (§4). The phones hold a contiguous tail block so there is exactly **one cut in (server->OP15) and one cut out (OP12->server)** per pipeline pass — minimum hop count, minimum bubble surface.

**Partition config schema** (drive the whole system from this; nothing about the split is hard-coded):

```jsonc
{
  "stages": [
    { "node": "a6000", "role": "head+tail", "layers": [0, 44],
      "has": ["tok_embd", "layers", "output_norm", "lm_head", "sampler", "scheduler"] },
    { "node": "op15",  "role": "mid", "layers": [45, 46] },
    { "node": "op12",  "role": "mid", "layers": [47, 47] }
  ],
  "hidden_dtype": "f16",          // pipeline payload dtype
  "n_layer": 48,                  // confirm from gguf
  "assert_kv_local": true         // fail-fast if a stage's layers reuse non-local KV (§7)
}
```

Rebalancing = edit `layers` ranges + re-run the offline sharder (§9). No code change.

### 4. Dataflow — PREFILL (per request, streaming at 8 fps)

Each incoming request (prompt of `T` tokens, or a video frame's image embeddings, §8) is prefilled through the pipeline as one forward pass:

1. **Server**: embed tokens (`tok_embd`, scaled by `sqrt(n_embd)` for text tokens; **not** scaled for raw image embeddings — gemma4.cpp:181-182), run layers 0..44 on CUDA (compute-bound, GPU is ideal). Build KV for layers 0..44 locally.
2. **Server -> OP15**: ship `inpL` = output of layer 44, shape `[n_embd, T]`. For `T=512`: `512×3840×2 = 3.93 MB` (matches brief). Also ship `inp_pos` (positions, `T×4 B`, negligible) so the phone can RoPE + build its SWA mask.
3. **OP15**: run layers 45,46 on the **NPU/HMX** (prefill is compute-bound; op12 measured 7.12 TFLOPS at M=512). Build & keep KV for 45,46 locally.
4. **OP15 -> OP12**: ship `inpL` = output of layer 46 (`[n_embd, T]`).
5. **OP12**: run layer 47 on NPU/HMX, build & keep KV for 47 locally.
6. **OP12 -> Server**: ship final hidden `[n_embd, T]`. Server applies `output_norm`, `lm_head`, final-logit softcap, samples **token 1** for this sequence. Sequence's distributed KV (server:0-44, OP15:45-46, OP12:47) is now warm; the sequence joins the "ready" pool.

Prefill activations are **hundreds of × under link capacity** (3.93 MB vs USB ~600 MB/s / WiFi ~100 MB/s): comms is a *latency* cost, not a bandwidth cost, exactly as measured. KV never crosses the wire.

### 5. Dataflow — DECODE (continuous batch of B≈64-128 ready sequences)

When ~64-128 sequences are ready, decode runs the *same* pipeline, but now each pass advances **all B sequences by one token** (each contributes 1 query row; the batch is over the token dim `ne[1]`, which the hexagon MUL_MAT path batches over and which the OpenCL tiled GEMM handles for batch≥2).

One decode step:

1. **Server**: has the current B next-token *embeddings* already (from the previous step's sampler, or step-0 from prefill). Run layers 0..44 (CUDA), update server KV for the B rows.
2. **Server -> OP15**: ship `inpL` `[n_embd, B]`. For `B=128`: `128×3840×2 = 0.94 MB` (matches brief). Ship the B positions too.
3. **OP15**: layers 45,46 on the **Adreno GPU** (decode is memory-bound GEMV; GPU is the right engine intra-phone, and it runs concurrently with any NPU prefill work — measured zero interference). Update local KV.
4. **OP15 -> OP12**: `inpL` `[n_embd, B]` (~0.94 MB at B=128).
5. **OP12**: layer 47 on Adreno GPU, update local KV.
6. **OP12 -> Server (the loop-back)**: ship final hidden `[n_embd, B]`. Server:
   - `output_norm` -> `lm_head` -> logits `[n_vocab, B]` (`262144×128×2 ≈ 67 MB`, **stays on the GPU, never shipped**) -> softcap -> **sampler** -> B new token IDs.
   - **Re-entry at the front (server-local, no wire):** the server looks up `tok_embd[new_id]`, applies the `sqrt(n_embd)` scale, and that becomes `inpL` for layer 0 of the **next** step. Because the server owns both the sampler and the embedding table and layer 0, **the next-token embedding is produced and consumed on the same node** — nothing about the loop-back or re-injection crosses the network. Only the two `[n_embd, B]` hidden tensors (server->OP15, OP12->server) and one OP15->OP12 hidden tensor traverse links per step.
7. Continuous batching: the scheduler (server) admits newly-ready prefilled sequences and evicts finished ones between steps, keeping B topped up. Admit/evict only changes which rows are in the `[n_embd, B]` tensor and each stage's active KV set — no weight movement, no re-shard.

**UNPROVEN — de-risk before trusting this loop.** Multi-sequence continuous-batched decode *on the phone backends* is not validated: the project recorded **"NPU n_parallel=2 HANGS" on op15**, and the 7.12 TFLOPS roofline is a *single synthetic matmul*, not B sequences each with its own KV + a block-diagonal / SWA attention mask. The decode dataflow above **assumes** the phone Adreno path (and/or NPU) correctly handles B separate KV streams with per-sequence masks. This must be proven with `llama-batched-bench` on op15/op12 (one hosted layer, real B) **before** any milestone depends on batched phone decode. If only B=1 works on the phone, the pipeline degrades to single-stream (~11 tok/s, energy-poor) and the whole energy thesis is moot — so this is the first bring-up gate, owned jointly with the batching subsystem.

**Per-layer embedding wrinkle (only if `n_embd_per_layer > 0`).** The gemma4 graph derives `inp_per_layer` from the **token IDs** at the front (gemma4.cpp:194-200) and each layer consumes its slice. In a pipeline split the phones don't run the front embedding code, so if the shard has per-layer embeddings you must **also ship, per stage, that stage's layers' per-layer embedding inputs** (shape `[n_embd_per_layer, n_tok, n_local_layers]`) alongside `inpL`, or pre-provision the `per_layer_tok_embd` slice on the phone and ship the token IDs. Confirm from gguf; for a dense 12B this term is expected to be zero and the wrinkle disappears.

### 6. KV-cache sizing per stage (why SWA layer choice matters for phones)

Per full-attention layer, KV per token = `2×n_head_kv×head_dim×2 B = 2×8×256×2 = 8 KB/token/layer`. For **one full-attention layer** at `B=128, ctx=4096`: `8KB×128×4096 ≈ 4.3 GB` — *not* negligible on a phone. Gemma-4 interleaves **sliding-window (SWA) and full-attention** layers (`is_swa_impl` pattern, gemma4.cpp:4-5; typical Gemma cadence ~5 SWA : 1 full — **confirm pattern from gguf**). SWA layers cap their KV at `n_swa` positions (e.g. 512-1024) regardless of context, so their KV is ~`8KB×128×1024 ≈ 1 GB` and bounded. **Design implication:** when you pick which tail layers the phones host (and when rebalancing), **prefer SWA layers** for the phones to keep phone KV small and flat; a phone holding a *full*-attention layer at long context + large B can blow its RAM budget from KV alone even though the *weights* are only 0.45 GB. Record each hosted layer's `is_swa(il)` in the shard manifest.

### 7. The shared-KV-tail trap and its mitigation (must resolve before freezing the split)

As established in §1.2, if `n_kv_shared_layers > 0` the **highest-indexed layers reuse earlier layers' KV**. The baseline hands layers 45-47 (the tail) to the phones — precisely the candidates for `has_kv==false`. If any hosted layer has `has_kv(il)==false`, its attention must read K/V that lives on the server -> **KV would have to cross the wire every step**, destroying both the locality invariant and the latency budget.

**Procedure (fail-fast, encoded as `assert_kv_local`):**
1. From the gguf read `n_kv_shared_layers`. If **0**, the tail split is safe — proceed.
2. If **> 0**, for each phone-hosted layer check `has_kv(il)`. For any `il` with `has_kv==false`, resolve which earlier layer it reuses and require that source layer to be **co-located** on the same node.
3. If co-location is impossible with a pure tail cut, **move the phone block off the tail**: host a contiguous *interior* block whose layers all own their KV (all `has_kv==true`) and are SWA (§6). Dataflow becomes `server(0..a) -> OP15 -> OP12 -> server(b..47 + norm + head + sample)` — one extra server segment, same three hops for the phones, loop-back unchanged. This is strictly more flexible than the tail cut and should be the fallback the sharder emits automatically.

Surfacing this now is the point of building Design A honestly: the naive "phones take the last 3 layers" is only correct when `n_kv_shared_layers==0`, which we must verify rather than assume.

### 8. Vision / mmproj attachment (video at 8 fps)

If inputs are frames, a **SigLIP-class vision encoder + multimodal projector (mmproj)** precede the LLM. Their output is a sequence of image embeddings already in `n_embd` width that enter at the **pipeline front as "raw embeddings input"** — the gemma4 graph explicitly special-cases this: raw image embeddings are injected as `inpL` **without** the `sqrt(n_embd)` scale (gemma4.cpp:181-182, `ubatch.token ? sqrtf(n_embd) : 1.0f`). **Baseline placement: vision encoder + mmproj on the server**, in front of layer 0. Rationale: the encoder is compute-heavy (favor the A6000), it must sit where embeddings are injected (server front), and it keeps the phones doing *pure transformer layers* so the pipeline plumbing we are validating stays clean. Each 8-fps frame's image-embedding sequence is then prefilled through the exact §4 path. (A later milestone could offload the vision encoder's prefill to a phone NPU — it is compute-bound and HMX-friendly — but that is out of scope for the bring-up and would add a fourth partition boundary.)

### 9. Rebalancing knob (what "more layers on phones" costs)

The split is a config edit (§3) consumed by an **offline sharder** that writes each node's local shard file (pre-provisioned, mmap'd — **no runtime RPC weight push**, per the hard constraint). Bounds on how many layers a phone *could* hold:

| Bound | OP15 (v81, ~70-80 GB/s) | OP12 (v75, ~68 GB/s) |
|---|---|---|
| **RAM** at 0.448 GB/layer | ~15-25 layers before crowding OS (12-16 GB class) | ~15-25 layers |
| **KV RAM** (if full-attn, B=128, 4k ctx) | ~4.3 GB/full-layer -> prefer SWA layers | same |
| **Thermal/throughput (the real limit)** | decode is memory-bound; per step the phone streams `L×0.448 GB` of weights at ~42-48 GB/s effective (measured HVX saturation), so step time ≈ `L×0.448/0.045 s ≈ L×10 ms` just for weight reads at B where the read is amortized | slightly slower bus |

So **RAM allows ~15-25 layers but sustained ~5-12 W thermal + the per-step weight-read time set the practical ceiling far lower** — the phone's layer count is bounded by "can it finish its layers within the pipeline step budget without throttling," not by capacity. Concretely: at the baseline (OP15=2, OP12=1) the phones add ~20-30 ms of memory-bound work per decode step — already the slow stages, and already a bubble source (§ heterogeneous DVFS). Rebalancing should proceed **one layer at a time**, each step re-measured with `llama-batched-bench`, watching for (a) the phone becoming the pipeline bottleneck, (b) thermal throttle collapsing sustained clocks, and (c) KV RAM if any added layer is full-attention. **Give the phones SWA layers first**, keep them contiguous, and keep the split off any shared-KV group (§7).

### 10. Honest status

- **Correct/grounded:** payload = single `[n_embd, n_tok]` residual per hop; loop-back and re-embed are server-local (no wire); sizes (0.448 GB/layer, 0.94 MB @B128, 3.93 MB @512-tok, 2 GB tied lm_head) all reconcile with the brief and the code.
- **Confirm from gguf:** `n_layer`, `n_embd`, `n_head(_kv)`, `head_dim`, `n_ff`, `n_vocab`, `n_kv_shared_layers`, `n_embd_per_layer`, `is_swa` pattern, MoE-or-dense.
- **Unproven / gating:** batched multi-sequence decode on the phone backends (NPU hang on op15 at n_parallel=2) — the entire decode dataflow assumes it; prove with `llama-batched-bench` before any energy claim. Baseline = ~6% of layers offloaded = plumbing only, **no energy win expected or claimed**.

---

## 2. Interconnect, Transport & Wire Protocol

Design for the physical + logical link that carries **hidden-state activations** (and control) around the logical ring `A6000 → OP15 → OP12 → A6000`. Every transformer stage runs its own local `llama.cpp` process holding its own pre-provisioned, mmap'd layer shard and its own KV cache. **Nothing but activations and small control frames ever cross the wire — no weights, no KV, no token-ids to the phones.**

### 0. What actually crosses the wire (and what never does)

| Data | Size (per step) | Crosses wire? |
|---|---|---|
| Layer weights (fp16 shard) | ~0.4–0.5 GB/layer (confirm from gguf) | **No** — pre-downloaded, mmap'd, shared NPU↔GPU via fastRPC |
| KV cache (per stage, per seq) | grows with context | **No** — each stage owns the KV for *its* layers only; referenced by `seq_id` |
| Token-ids / embeddings | vocab ~256k (confirm from gguf) | **No** — embedding table + sampler live on A6000; only the resulting **hidden state** is sent to OP15 |
| Hidden-state activation (fp16) | decode B=128: **0.94 MB**; prefill 512-tok: **3.9 MB** | **Yes** — the only bulk payload |
| Control (session/seq-lifecycle/credits/heartbeat) | tens of bytes | Yes |

The sampler on the A6000 turns the returned hidden state into a token, looks the token's **embedding** up locally, and sends the *next* input hidden state to OP15. Phones therefore only ever see fp16 hidden vectors — this is what keeps token-ids and the 256k lm_head off the wire. (Byte figures: `128 × 3840 × 2 = 983 040 B ≈ 0.94 MB`; `512 × 3840 × 2 = 3 932 160 B ≈ 3.9 MB`. Hidden = 3840, layers ≈ 48 — **confirm from gguf**; the 12B gguf is not staged on this host, only E2B/E4B + vocab files are.)

### 1. Physical topology: solving the host-centric-USB trap

Over USB, an Android phone is a **device**, not a host: with USB tethering it exposes one network interface and forms a **point-to-point IP subnet with the PC only**. Two phones on two USB ports get two independent subnets and **cannot peer** — so the `OP15 → OP12` hop has no direct path.

**Recommended (primary): USB tethering + host IP forwarding.**
- Enable tethering on each phone (RNDIS on older / **NCM** on newer Android). Host sees `usb0`/`rndis0` per phone; assign static /30s, e.g. `A6000=10.0.15.1 / OP15=10.0.15.2`, `A6000=10.0.12.1 / OP12=10.0.12.2`.
- `sysctl net.ipv4.ip_forward=1` on the A6000; add routes so OP15's subnet reaches OP12's subnet. The `OP15→OP12` hop then **hairpins through the A6000 kernel** (two USB traversals, no compute). The server is idle during the phone stages, so this is free capacity.
- Set `TCP_NODELAY` (already done in `transport.cpp:578`) and, on Linux, `TCP_QUICKACK`; raise the `usb0` MTU where the gadget allows, to cut segment count for the 0.94/3.9 MB payloads.

**Why not "all nodes on one WiFi LAN" (the alternative):** it removes the peering problem (any-to-any routing, real `OP15→OP12`) and is the right **fallback for an untethered demo**, but WiFi adds 2–10 ms RTT + jitter + shared-medium contention → pipeline bubbles across already-heterogeneous v81/v75 stages. Keep it as a config flag, not the default.

**Design note (honest):** whether kernel-forwarded or app-relayed, the middle hop traverses the server twice. An alternative is to make the server an **application-level relay** (server reads OP15's output socket, forwards to OP12) — it costs the same two USB traversals but gives the continuous-batch scheduler direct visibility/control of the activation stream (reorder, drop, re-batch). Recommend **kernel IP-forward for the data path** + **separate control sockets** from the server to each phone for lifecycle; revisit app-relay if the scheduler needs to mutate in-flight microbatches.

### 2. Transport choice: raw TCP (reuse the ggml-rpc socket layer)

**Recommendation: raw TCP, one persistent pre-established connection per directed hop, length-prefix framed** — exactly the mechanism `ggml-rpc/transport.cpp` already ships (`socket_t`, `send_data`/`recv_data`, `send_msg` = `[u64 LE length][body]`, `TCP_NODELAY` on).

- **vs gRPC:** HTTP/2 framing + protobuf serialize/parse would copy the fp16 blob and add per-message CPU + head-of-line coupling for sub-ms-sensitive hops. No benefit here — we have exactly 3 static peers and one message shape.
- **vs ZeroMQ:** nice socket patterns but an extra dependency and its internal queuing/copying adds latency; our framing + backpressure is ~200 lines on top of the existing `socket_t`.
- Payloads are already **contiguous fp16 in the ggml compute buffer** → serialization is a header write + one `send` of the raw buffer (**zero-copy**); receive reads straight into the stage-input tensor's data. All nodes are little-endian (ARM64 + x86-64) → **no byte-swap**.

The existing `ggml-rpc` *backend* (op-granular `SET_TENSOR`/`GRAPH_COMPUTE` round-trips) is the wrong layer — it would ship graphs/weights and do per-op RTTs across three nodes. We reuse its **socket/framing primitives**, not its RPC semantics, and add a purpose-built **stage-granular activation protocol**.

### 3. Wire protocol

**Outer framing:** reuse `send_msg`/`recv_msg` — `[uint64 LE total_len][body]`. Body = fixed header + optional seq-table + payload. Header is 8-byte aligned so the fp16 payload starts aligned for `memcpy` into the ggml buffer.

```c
enum pipe_msg_type : uint16_t {
    MSG_SESSION_INIT = 1,  // model hash, assigned layer range [a,b), dtype, hidden, max_batch, kv_pool
    MSG_SEQ_ALLOC    = 2,  // list of seq_id to create local KV slots for
    MSG_SEQ_FREE     = 3,  // evict seq_id (continuous-batching evict)
    MSG_PREFILL_ACT  = 4,  // ragged: per-seq prompt hidden states
    MSG_DECODE_ACT   = 5,  // one hidden row per active seq (the hot path)
    MSG_CREDIT       = 6,  // flow-control window update (backpressure)
    MSG_HEARTBEAT    = 7,  // liveness / DVFS-throttle signal
    MSG_ERROR        = 8,  // fail-fast + resync
};

struct pipe_hdr {          // 32 bytes, 8-byte aligned
    uint32_t magic;        // 'PIPE'
    uint16_t version;
    uint16_t msg_type;
    uint32_t batch_id;     // pipeline microbatch / decode step id (monotonic; drop-detect)
    uint32_t n_seq;        // sequences in this batch
    uint32_t n_tokens;     // decode: == n_seq;  prefill: sum of prompt lengths
    uint32_t hidden_dim;   // 3840 (confirm from gguf)
    uint8_t  dtype;        // 0=fp16,1=fp32,2=bf16  (confirm gemma native dtype from gguf)
    uint8_t  flags;        // bit0=last-microbatch-of-step
    uint16_t _pad;
    uint32_t payload_bytes;
};

struct pipe_seq_ent { uint32_t seq_id; uint32_t pos; uint32_t n_tok; }; // n_seq of these
```

- **Payload layout:** row-major `[token][hidden]`, contiguous fp16, `n_tokens × hidden_dim × 2` bytes — a straight copy of the stage's output tensor (`ne[0]=hidden` contiguous, `ne[1]=n_tokens`, packed `nb`). No per-row padding.
- **seq-table** carries the `seq_id ↔ position` map so each stage indexes its **local** KV slot; decode `n_tok=1`, prefill `n_tok=prompt_len`. This is how KV stays local yet stays consistent across stages without ever shipping KV.
- **Control** frames (`SESSION_INIT`/`SEQ_ALLOC`/`SEQ_FREE`/`CREDIT`) have header + a small typed body, no payload. `SEQ_ALLOC`/`FREE` drive per-stage KV-slot lifecycle in lock-step with the server's continuous-batch admit/evict.
- **Ordering/loss:** one TCP connection per hop → in-order, reliable. `batch_id` is monotonic; a gap ⇒ `MSG_ERROR` + resync (fail-fast; TCP won't reorder, so a gap means a dropped connection).

### 4. Per-hop byte + latency budget

| Payload | Bytes | USB xfer @~600 MB/s* | WiFi xfer @~100 MB/s |
|---|---|---|---|
| Decode B=128 | 0.94 MB | ~1.6 ms | ~9.4 ms |
| Decode B=64  | 0.47 MB | ~0.8 ms | ~4.7 ms |
| Prefill 512-tok | 3.9 MB | ~6.5 ms | ~39 ms |
| Control frame | ~tens B | ~RTT | ~RTT |

RTT: **USB ~0.1–1 ms**, **WiFi ~2–10 ms**.

*\*The ~600 MB/s USB figure is USB3 raw; **RNDIS/NCM tethering goodput is frequently far lower (~30–300 MB/s) — MUST MEASURE with iperf3** and treat the table's USB column as a ceiling.*

**Per decode step (3 activation hops, ~90 ms step):**
- **USB:** `3 × (xfer + RTT) + middle-hop hairpin ≈ 3×~2 ms + ~2 ms ≈ 8 ms` added end-to-end ≈ **~9 % of the step**.
- **WiFi:** `≈ 3×~10 ms + hairpin ≈ 35–50 ms` ≈ **40–55 % of the step**, plus jitter → bubbles. → **USB is the clear default.**

### 5. How pipelining/microbatching hides the latency

Throughput is decoupled from per-token latency. Split each decode batch (64–128 seqs) into **k microbatches** streamed through the ring so that while OP12 works microbatch *i*, OP15 works *i+1*, and the A6000 works *i+2*. With in-flight depth ≥ (#stages + comms slots), steady-state throughput = `1 / max(stage_compute, hop_time)`; the ~8 ms USB comms **overlaps** compute and only shows up in pipeline fill/drain, not throughput.

**Honest caveat for the baseline (3 of ~48 layers on phones):** the phone excursion `server→OP15→OP12→server` sits in the per-token critical path, but the offloaded compute (3 layers) is only a few ms while the excursion's comms is ~8 ms (USB) / ~40 ms (WiFi). So the baseline **plumbing** can be *net-negative* on latency — which is expected: the tiny share is to validate the transport end-to-end, not to save energy. Later milestones amortize the fixed comms by moving more layers onto the phones (bounded by RAM + sustained thermal power); the comms cost per step is ~constant while offloaded compute grows.

### 6. Backpressure at 8 fps

- **Credit-based flow control** (`MSG_CREDIT`): each stage advertises N buffer slots; upstream sends a microbatch only against a credit. A phone that DVFS/thermal-throttles drains its credits → upstream naturally stalls → the server's scheduler sees credit starvation and sheds load (reduce phone layer share / fall back to CPU / drop stale frames). The credit window is sized to pipeline depth to keep stages full without unbounded queueing.
- **Prefill at 8 fps** = one request / 125 ms; a 3.9 MB prefill hop is ~6.5 ms over USB — **>15× under budget**, so prefill comms is never the bottleneck (the phone NPU prefill compute is). Incoming frames queue in a **bounded** server buffer; on overflow, shed (drop frame or CPU-only path).
- **Liveness:** `MSG_HEARTBEAT` doubles as a throttle signal; missed heartbeats → mark stage degraded, rebalance.

### 7. Confirmations
- **KV never crosses the wire** — asserted structurally: no message type serializes KV; stages exchange only `seq_id`/`pos` references into their **local** KV. Add a debug assert on the send path that the only bulk buffer is the hidden-state tensor.
- **Token-ids never reach phones** — embedding lookup + sampler are A6000-only; phones exchange fp16 hidden states exclusively.

---

## 3. Weight Provisioning — Pre-Downloaded Shards

### 1. Goal and hard constraints

Each node in the pipeline (A6000 server, OP15, OP12) must have **its own layer weights present on local storage before inference starts**. No weight tensor ever crosses the wire at runtime. On each phone the weight file is `mmap`'d once so the Hexagon NPU and the Adreno GPU share **one physical copy** through fastRPC. Only per-layer activations (the hidden state at a stage boundary) move between nodes at runtime.

This subsystem delivers: (a) a **shard extractor** that slices the source gguf into per-device standalone files; (b) a **manifest** (model id, layer ranges, dtype, hashes, version); (c) an **on-device cache** with integrity check, cold-start, and re-shard-on-rebalance flows; (d) the **llama.cpp loader mapping** so each stage loads only its layers and runs as a hidden-state → hidden-state block transform.

### 2. Model facts (confirm from gguf)

Run `python3 gguf-py/gguf/scripts/gguf_dump.py gemma-4-12B-f16.gguf` on the real weights and fill these in; values below are the design's working assumptions, flagged where unverified:

| Field | Working value | Source |
|---|---|---|
| arch | `gemma4` (LLM_ARCH_GEMMA4, llama-model.cpp:141) | confirm from gguf |
| `gemma4.block_count` (n_layer) | ~48 | confirm from gguf |
| hidden size `d` | ~3840 | confirm from gguf |
| attention | GQA | confirm from gguf |
| vocab | ~256k | confirm from gguf |
| per-layer fp16 weight size | ~0.4–0.5 GB | given / confirm from gguf |
| `token_embd.weight` | ~256k × 3840 × 2 B ≈ **1.97 GB** | confirm from gguf |
| `output.weight` (lm_head) | present, or **tied** to `token_embd` (Gemma ties embeddings) | confirm from gguf — decides whether a separate lm_head tensor exists |
| **SWA pattern period `n_pattern`** | Gemma interleaves sliding-window and global attention by layer index | **confirm from gguf** — load-bearing for sharding (see §5) |

### 3. Partition plan

Baseline (validates plumbing; tiny phone share). Assign the phones a **contiguous ascending tail range** so the pipeline order server → OP15 → OP12 → server matches increasing global layer index, and so we can keep the SWA/RoPE-by-index semantics intact.

| Node | Global layers owned | Also owns | Approx. weight size (fp16) |
|---|---|---|---|
| A6000 server | `blk.0 … blk.44` (45 layers) | `token_embd`, final `output_norm`, `output`/lm_head, sampler, scheduler | ~22 GB (45×0.45 + ~2–4 GB heads) |
| OP15 (v81) | `blk.45`, `blk.46` (2 layers) | — | ~0.9 GB |
| OP12 (v75) | `blk.47` (1 layer) | — | ~0.45 GB |

Phone shards are ~0.5–1 GB — far under phone RAM, and small enough that a full re-provision over WiFi (~100 MB/s) is ~5–10 s. The phones hold **no embeddings, no lm_head, no sampler**; they only apply transformer blocks. KV for a stage's own layers lives on that stage and never ships.

### 4. Shard file format

A shard is a **standalone gguf** containing only the tensors that node owns, plus enough metadata to construct the graph for exactly its layer range. It is a real gguf, so integrity, dtype, and shapes are self-describing.

Two candidate encodings; we recommend **B** for Gemma-4:

**Path A — renumbered partial gguf (simple, but unsafe for Gemma-4).** Rewrite `block_count` to the local count and renumber surviving `blk.{g}` → `blk.{0..n-1}`. Produces a fully valid small model that vanilla llama.cpp loads with **zero loader changes**. Trap: llama.cpp derives per-layer behavior from the *global* index — `is_swa(il)` = `il % n_pattern` (llama-hparams.cpp:8–16) and RoPE base switches on SWA per layer (llama-model.cpp:1988). Renumbering `blk.45→blk.0` changes `il % n_pattern` and silently misassigns attention type and RoPE base. **Safe only for uniform-layer models, or when the local start index is a multiple of `n_pattern`.** Do not use for Gemma-4 unless the range is pattern-aligned and verified.

**Path B — global-index-preserving stage shard (recommended).** Keep the true global metadata (`block_count` = 48, the SWA pattern, RoPE bases) and ship only the assigned `blk.{g}` tensors **without renumbering**. Add two KV keys:

```
pipeline.layer_start = 45      # first global layer this shard owns
pipeline.layer_count = 2       # number of contiguous layers
pipeline.is_first    = false   # if true, stage embeds tokens
pipeline.is_last     = false   # if true, stage runs final norm + lm_head + sample
```

Every per-layer tensor keeps its global name (`blk.45.*`, `blk.46.*`), so `is_swa(45)`, `is_swa(46)` and their RoPE bases are computed exactly as in the full model. This costs a small, localized loader/graph patch (§6) but eliminates the renumbering hazard entirely.

**Vocab in phone shards.** A full-model load path expects tokenizer metadata. Phones never tokenize, embed, or sample, but the simplest zero-fork option is to copy the *vocab KV strings* (a few MB of token text — not the 1.97 GB embedding tensor) into phone shards so the loader is happy. Alternative: patch the loader to permit a headless stage with no vocab (part of the Path B patch). Decide once `pipeline.is_first/is_last` handling lands.

### 5. Manifest schema

One `manifest.json` per model version, produced by the extractor, published next to the shard files. Phones verify against it before mmap.

```json
{
  "manifest_version": 1,
  "model_id": "gemma-4-12b-f16",
  "source_gguf_sha256": "…",
  "arch": "gemma4",
  "n_layer_total": 48,
  "dtype": "f16",
  "swa_pattern": { "n_pattern": 6, "dense_first": false },   // confirm from gguf
  "shard_format": "B-global-index",
  "version": "2026-07-06T00:00:00Z+r3",   // bumped on every re-shard/rebalance
  "shards": [
    { "shard_id": "server", "node": "a6000",
      "layers": [0,44], "has_embd": true, "has_lm_head": true,
      "files": [ {"name":"server.gguf","bytes":23622320128,"sha256":"…"} ] },
    { "shard_id": "op15", "node": "op15",
      "layers": [45,46], "has_embd": false, "has_lm_head": false,
      "files": [ {"name":"op15.gguf","bytes":966367641,"sha256":"…"} ] },
    { "shard_id": "op12", "node": "op12",
      "layers": [47,47], "has_embd": false, "has_lm_head": false,
      "files": [ {"name":"op12.gguf","bytes":483183820,"sha256":"…"} ] }
  ]
}
```

`sha256` is over each file's bytes; `source_gguf_sha256` ties every shard to one parent so a stage can never silently run mismatched weights. `version` (monotonic) is the rebalance key.

### 6. The shard extractor

A standalone Python tool over `gguf-py` (`gguf_reader` + `gguf_writer`) — no build, runs on the server before provisioning.

```
extract_shards.py --in gemma-4-12b-f16.gguf --plan plan.json --out ./shards/
  plan.json: {"server":[0,44], "op15":[45,46], "op12":[47,47]}

for each shard S in plan:
    w = GGUFWriter(out/S.gguf, arch)
    copy all general.* and <arch>.* hparam KV verbatim   # keeps block_count=48, SWA, RoPE
    add KV pipeline.layer_start, pipeline.layer_count, pipeline.is_first, pipeline.is_last
    copy vocab KV strings (token list/scores/types)       # small; phones stay loader-compatible
    for tensor t in source:
        g = parse_layer_index(t.name)                     # blk.{g}.*  → g, else non-layer
        if g in S.range: add_tensor(t)                    # Path B: keep name blk.{g}.*
        if t in {token_embd, output_norm, output} and S.has_embd/has_lm_head: add_tensor(t)
    write; record sha256, byte size
emit manifest.json (source sha, per-file sha, version)
```

Reuse note: `tools/gguf-split` already walks tensors and writes new gguf files with split KV — the extractor is the same shape but partitions **by layer name** instead of by tensor count, and writes standalone (not `split.count`) files. Start from its tensor-copy loop.

**Layer-granular option (rebalance-friendly).** Instead of one gguf per stage, emit **one gguf per layer** (`blk.45.gguf`, …) plus a per-stage head gguf. A stage then loads a *list* of gguf files — llama.cpp already merges multiple gguf files into one model via `llama_get_list_splits` / the additional-GGUF path (llama-model-loader.cpp:78, 589–667). Rebalancing a single layer then transfers only that ~0.45 GB file, not the whole stage shard. Recommended once baseline works.

### 7. On-device cache, integrity, cold start

**Cache path (Android).**
- Dev/bring-up (adb): `/data/local/tmp/llmshard/<model_id>/<version>/` — writable via `adb push`, survives app restarts.
- Productionish: app-private `getExternalFilesDir()/llmshard/<model_id>/<version>/`.

Keep versions in separate directories so a rebalance never overwrites the running shard; a symlink/pointer file `current → <version>/` selects the active one for atomic swap.

**Fetch.** One-time, before inference: `adb push` (USB bring-up), or `scp`/HTTP GET from the server over the same LAN used for activations. Phone writes to `<version>/.staging/`, fsync, then verifies.

**Integrity check (cold start).** On process start, before mmap:
1. Read `manifest.json`; check `model_id`, `arch`, `n_layer_total`, `swa_pattern` match the binary's expectations and that this node's `shard_id` is present.
2. For each file, recompute SHA-256; compare to manifest. Mismatch → refuse to start, re-fetch.
3. Check `source_gguf_sha256` equals the server's advertised parent hash (the server sends it in the pipeline handshake) so all three stages provably came from the same parent model. Mismatch → refuse.
4. `mmap` the verified file(s) read-only; hand the same mapping to both the Hexagon and OpenCL backends (single physical copy).

Refusing on any mismatch is deliberate: a stage running stale or mismatched layers corrupts the whole pipeline output silently.

### 8. Re-shard on rebalance

Later milestones move more layers onto phones (bounded by phone RAM and sustained thermal power). Flow:
1. Server runs `extract_shards.py` with a new `plan.json`, producing shards under a new `version`.
2. Server updates `manifest.json` (new version, new hashes, new layer ranges).
3. Each phone fetches **only the files whose sha changed** (with the layer-granular option, just the moved layer gguf(s)) into `<new_version>/`, verifies (§7).
4. **Atomic cutover:** drain the pipeline (finish in-flight microbatches), each phone reloads from `<new_version>/`, re-mmaps, rebuilds its stage graph for the new range; server flips `current →` new version; resume. KV caches are per-stage and per-layer-range, so a moved layer's KV is rebuilt on that layer's new owner (it is regenerated on next prefill; no KV migration over the wire).
5. Old `<version>/` dirs are GC'd after cutover.

Because ranges stay contiguous and global indices are preserved, a rebalance is just "different `pipeline.layer_start/count` + the corresponding layer files" — no re-numbering, no metadata surgery.

### 9. Mapping to llama.cpp model loading

Each node runs a llama.cpp process that loads **its shard as a model** and executes only its layer range as a hidden-state transform. Two loader-side pieces:

1. **Load only assigned layers.** With Path B, the shard's gguf physically contains only `blk.{start..start+count-1}`. `block_count` still reads 48, so the create_tensor loop (llama-model.cpp:1305+) would try to create all 48 layers and throw on the missing ones. Patch: gate the per-layer `create_tensor` calls on `il ∈ [pipeline.layer_start, +count)`; skip others. This keeps `hparams` (SWA pattern, RoPE bases, head dims) fully populated so `is_swa(il)` / `rope_freq_base(il)` remain correct for the owned layers.
2. **Stage run-mode graph.** A middle stage must (a) take an **input hidden-state tensor** at the pipeline input instead of `token_embd`→embedding lookup, (b) run only its layers, (c) emit the hidden state (no final norm/lm_head/logits). Gate on `pipeline.is_first` (do the embedding) and `pipeline.is_last` (do final norm + lm_head + sample). This is a new "pipeline stage" build path in the graph builder; it is the main code investment of this subsystem and it localizes all the risk.

Path A needs neither patch but is unsafe for Gemma-4 (§4). Recommendation: **Path B + the two patches above.** Confirm `n_pattern` from gguf and add a build-time assert that each stage's `[start, start+count)` produces the same `is_swa`/RoPE-base sequence as the full model (guards against a future non-contiguous or misaligned plan).

### 10. Relationship to ggml-rpc (weights: NO; activations: maybe)

**Why ggml-rpc does not satisfy the weight constraint.** The RPC backend is a "remote dumb backend": the client builds the graph and **uploads every weight tensor at runtime** via `RPC_CMD_SET_TENSOR` / `SET_TENSOR_HASH` into a device buffer on the remote (ggml-rpc.cpp:465–487). Even with the optional hash-keyed disk cache (`rpc-server -c`, rpc-server.cpp:234; `get_cached_file` by FNV hash), the first run still pushes the weights over the wire, and the cached copy lands in an **RPC device buffer**, not the model's `mmap`'d region — so it is neither "no runtime weight RPC" nor "one physical copy shared by NPU+GPU." It fails both hard constraints. We therefore do **not** use ggml-rpc for weights.

**Activations only — possible, but not via the stock ggml-rpc backend.** ggml-rpc's model is "remote executes ops the client graphs," which does not match "each phone independently owns and runs its layers from local mmap'd weights and only exchanges a boundary hidden state." The clean fit is a **thin activation transport** (the pipeline-transport subsystem): each stage is a full local model (§9); at a stage boundary we send just the hidden-state tensor (batch-128 × d3840 fp16 ≈ **0.94 MB/hop**; 512-token prefill ≈ **3.9 MB**), which is 100s× under USB/WiFi capacity — a latency cost, not a bandwidth one. ggml-rpc's `SET_TENSOR`/`GET_TENSOR` wire framing can be *borrowed* as the serialization for that one activation tensor, but the weight-upload and remote-graph machinery is bypassed. Net: **ggml-rpc backend = not used for weights; may be cannibalized only as an activation-tensor transport, and even then a purpose-built stage transport is cleaner.**

### 11. Honest unknowns / risks

- **Gemma-4 `n_pattern` and tied-embedding status are unconfirmed** (confirm from gguf). Both directly shape the shard format and whether a separate lm_head tensor exists on the server shard.
- **The Path B loader/graph patch (stage run-mode, hidden-state in/out, per-range create_tensor) is unbuilt.** It is the real engineering surface of this subsystem; everything else is tooling.
- **Headless (no-embd/no-lm_head/no-sampler) model load is not a first-class llama.cpp mode.** We rely on copying vocab strings into phone shards to stay loader-compatible until the headless path is patched.
- **This subsystem does not prove the phone can *run* continuous-batched multi-sequence decode** — that is the separate, flagged n_parallel=2 hang risk. Weight provisioning only guarantees the right bytes are present and mmap-shared; de-risk decode with `llama-batched-bench` on the phone shard early.

---

## 4. On-Phone Single-Copy Weight Sharing (mmap + fastRPC + OpenCL)

### Goal and scope

Each phone (OP15, OP12) is pre-provisioned with the fp16 weight tensors for its assigned Gemma-4 12B layers (baseline: OP15 = 2 layers, OP12 = 1 layer; per-layer fp16 weight ≈ 0.4–0.5 GB — confirm exact per-tensor shapes from gguf). Within a phone the pipeline uses **both** engines on the **same** layer weights: prefill (compute-bound) runs on the Hexagon HMX NPU, decode (memory-bound GEMV) runs on the Adreno GPU, concurrently, with the project's measured zero-interference co-schedule. This subsystem guarantees there is **one physical copy** of each weight tensor in DRAM, mmap'd once, that both engines read.

The honest headline: **a single shared copy is feasible and byte-compatible for F16/F32 weights (our fp16 model) on the default kernel paths, and it is NOT feasible for quantized weights** because the two backends repack quantized data into mutually incompatible layouts. Since the target model is fp16, the shared-copy design is the primary path; the quantized case is documented as a fallback (per-engine disjoint copies).

### What the two backends do today (ground truth from the tree)

**Hexagon (`ggml/src/ggml-hexagon/ggml-hexagon.cpp`)** already allocates weights in an ION/dmabuf and shares them CPU↔CDSP. `ggml_hexagon_shared_buffer::alloc()` (lines 296–315) does exactly the primitive we need:
- `rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, size)` → CPU-mmap'd ION buffer (`base`).
- `rpcmem_to_fd(base)` → the **dmabuf fd** (`fd`).
- `fastrpc_mmap(domain_id, fd, base, 0, size, flags)` (line 266) → registers the buffer with the CDSP/HTP so NPU kernels read it by the same virtual address.

Weight *layout* in that buffer depends on type (`set_tensor`, lines 853–898):
- **F16 / F32**: `default: memcpy(...)` (line 896) — stored **plain row-major**, no repack. HMX tiles into 32×32 blocks at *runtime* in VTCM (`htp/matmul-ops.h`: `HTP_MM_HMX_TILE_N_COLS/ROWS 32`), streaming from plain DDR rows — so the *stored* bytes are ordinary row-major f16.
- **Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4**: `repack_*_tiled(...)` — stored in a Hexagon-specific **tiled** layout (`ggml_hexagon_is_repack_type`, line 156). A separate `repack_buffer_type` exists for these.

**OpenCL (`ggml/src/ggml-opencl/ggml-opencl.cpp`)** today allocates a **separate** device buffer — `clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL)` in `..._buffer_type_alloc_buffer` (line 8815; falls back to `CL_LARGE_BUFFER_QCOM` = `0x41A6` at line 8821). There is **no** ION/dmabuf import path in the file (grep for `EXT_HOST_PTR_QCOM`/`clImportMemory` = 0 hits). Weight *layout*:
- **F16**: default `set_tensor` path writes plain bytes via `clEnqueueWriteBuffer(extra->data_device, ...)` (lines 7725–7727). The default GEMM `mul_mm_f16_f32_l4_lm` (batch ≥ 2) and the GEMV kernels (`kernel_mul_mat_f16_f32*`) read this plain row-major f16 directly. **Same byte layout as Hexagon f16.**
- **Quantized**: `set_tensor` does an SoA split (scales/quants into sub-buffers), builds `clCreateImage` image1d objects, and for Adreno additionally `transpose_2d_as_16b/8b` — a layout with no relation to Hexagon's tiling.
- **xmem GEMM** (`ggml_cl_mul_mat_f16_f32_adreno_xmem`, env-gated `GGML_OPENCL_ADRENO_XMEM_GEMM`, batch ≥ 16): even for f16 it **re-packs the base weight** into a transposed image via `kernel_adreno_xmem_prepack_weight_f16` into a *derived* `cl_mem` (the working-tree diff adds a prepack cache for it). This derived buffer is a per-engine artifact regardless of sharing.

### Layout compatibility verdict

| Weight type | Hexagon stored layout | OpenCL consumed layout (default path) | Single shared copy? |
|---|---|---|---|
| **F16 / F32** | plain row-major (memcpy) | plain row-major (l4_lm GEMM / GEMV) | **YES — byte-identical** |
| F16 + xmem GEMM (batch ≥ 16) | plain row-major | transposed prepacked image (derived) | Shared base + GPU-private derived image |
| Q4_0/Q4_1/Q8_0/MXFP4/IQ4_NL | Hexagon tiled | SoA + image + Adreno transpose | **NO — incompatible** |

For the fp16 Gemma-4 workload the default engine paths (HMX prefill, l4_lm/GEMV decode) both consume **plain row-major f16**, so one ION buffer serves both. This is the design we build. xmem is off by default; if later enabled for batched decode it needs a GPU-private transposed image derived from (not replacing) the shared base — call that out as a +~0.4–0.5 GB/layer GPU-side cost, only if xmem is turned on.

### Mechanism: one ION/dmabuf, two engine handles

We introduce a new buffer type, `ggml_backend_phone_shared_buffer_type`, that owns the single allocation and hands each engine its own handle onto the same physical pages.

```
                 rpcmem_alloc2(RPCMEM_HEAP_ID_SYSTEM, size)   [ION/dmabuf]
                          |            |             |
                 CPU mmap (base)   dmabuf fd     (page-aligned, padded)
                          |            |
        +-----------------+            +------------------------+
        | fastrpc_mmap(domain,fd,base) |  clImportMemory / cl_qcom_*_host_ptr(fd)
        v                              v
   HTP/CDSP reads base            Adreno reads imported cl_mem
   (NPU, prefill/HMX)             (GPU, decode/GEMV)
                    ONE physical copy in DRAM
```

**Allocation (reuse Hexagon's primitive).** The shared buffer's `alloc` is the existing `rpcmem_alloc2 + rpcmem_to_fd + fastrpc_mmap` sequence, factored so the fd/base/size are exposed to the OpenCL importer. Keep `RPCMEM_HEAP_ID_SYSTEM` (system ION heap → dmabuf) so both fastRPC and OpenCL can wrap it.

**OpenCL import (the new code).** Wrap the same dmabuf/ION pages as a `cl_mem` using a Qualcomm extension — no copy:
- Preferred: `cl_qcom_dmabuf_host_ptr` — `cl_mem_dmabuf_host_ptr { cl_mem_ext_host_ptr ext; void* dmabuf_hostptr; int dmabuf_filedesc; }` with `ext.allocation_type = CL_MEM_DMABUF_HOST_PTR_QCOM`, passed to `clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM, size, &dmabuf_hostptr_struct, &err)`. `dmabuf_filedesc` = `rpcmem_to_fd(base)`.
- Fallback: `cl_qcom_ion_host_ptr` — `cl_mem_ion_host_ptr { ext; void* ion_hostptr; int ion_filedesc; }`, `ext.allocation_type = CL_MEM_ION_HOST_PTR_QCOM`, `ext.host_cache_policy = CL_MEM_HOST_WRITEBACK_QCOM`.
- Probe `clGetDeviceInfo(CL_DEVICE_EXTENSIONS)` at init for `cl_qcom_dmabuf_host_ptr` / `cl_qcom_ion_host_ptr`; pick whichever the Adreno driver on that phone advertises (v81/Adreno-840 vs v75/Adreno-750 may differ — confirm by measurement on each device).

The imported `cl_mem` covers the whole buffer; per-tensor `cl_mem` are `clCreateSubBuffer` regions at each tensor's byte offset (the same sub-buffer mechanism already used for SoA quants, e.g. line 6409), and each tensor's `ggml_tensor_extra_cl.data_device` points at the shared base `cl_mem` with `extra->offset` = tensor offset. **No `clEnqueueWriteBuffer` for weights** — the bytes are already there from the CPU-side `memcpy` in `set_tensor`.

### Alignment and padding

- rpcmem/ION allocations are page-aligned (≥ 4 KB). fastRPC is satisfied by that.
- OpenCL ION/dmabuf import requires: host_ptr aligned to `CL_DEVICE_PAGE_SIZE_QCOM`, and the allocation size **padded** by `CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM` (query both at init). Round the `rpcmem_alloc2` size up by the padding and align base to the device page size.
- Per-tensor sub-buffer offsets must satisfy `CL_DEVICE_MEM_BASE_ADDR_ALIGN` (the code already `align_to(..., backend_ctx->alignment)`, typically 1024 B). Set the **phone-shared buffer type alignment** = `max(4 KB page, EXT_MEM_PADDING, OpenCL base-addr-align)` and let ggml-alloc (galloc) place tensors on that boundary. For plain f16 this only wastes a few hundred bytes per tensor.
- HMX has no *storage* alignment requirement for f16 beyond contiguity (tiling is a runtime VTCM operation); rows are contiguous, so shared placement is fine.

### Cache coherence (made trivial by read-only weights)

Weights are **written once** by the CPU (`set_tensor` memcpy) at provisioning/load, then **only read** by NPU and GPU at runtime. There are no concurrent writers, so the hard bidirectional-coherence problem does not arise. The only coherence action is a **one-time clean of the CPU write** so the DSP and GPU see the bytes:
- CDSP side: fastRPC performs cache maintenance on registered rpcmem buffers around remote invocations; the first HTP matmul call sees clean data. (Belt-and-suspenders: an explicit rpcmem cache-flush ioctl after load.)
- GPU side: choose `host_cache_policy = CL_MEM_HOST_WRITEBACK_QCOM` and flush the CPU cache once after the final weight write (e.g. `msync`/rpcmem flush) before the first GPU enqueue. Because the buffer is `CL_MEM_READ_ONLY` to OpenCL and never rewritten, no per-step maintenance is needed — which is *why* the measured NPU-compute + GPU-memory co-schedule shows zero interference: two readers of the same clean pages, no coherence traffic.
- Do **not** use uncached ION (`RPCMEM_FLAG_UNCACHED`): HVX/HMX and Adreno both want cached DDR reads for bandwidth; uncached would wreck the decode GEMV that is already bandwidth-bound (~42–48 GB/s measured).

### llama.cpp / ggml integration

The real integration friction is ggml's model: **a weight tensor belongs to exactly one buffer, owned by one backend**, but here two backends must consume it. Two ways to resolve, in order of preference:

1. **Fused "phone" backend (recommended).** A thin backend (`ggml-phone`) that owns *both* a Hexagon session and an OpenCL context and exposes the single `phone_shared` buffer type. Its `supports_op`/graph-compute dispatches each op to NPU or GPU by op-type and batch size (prefill/large-M MUL_MAT → HTP; decode GEMV → Adreno), reading weights from the shared buffer via the engine-appropriate handle (base+offset for HTP; sub-buffer `cl_mem` for OpenCL). This keeps ggml's "one tensor, one buffer" invariant intact — the buffer is the phone's, and the phone internally routes. Weight tensors carry both handles: HTP uses `base`/`fd`; a parallel `extra_cl` (imported sub-buffer) is attached for the OpenCL kernels. This is the cleanest fit and localizes all sharing logic.

2. **Shared buffer type consumed by two backends.** Register `phone_shared` buft with `supports_backend` true for both hexagon and opencl, and have each backend's kernels look up their engine handle from a side-table keyed by dmabuf fd. This fights ggml-sched's residency assumptions (it will try to copy a tensor when the consuming backend differs from the buffer's backend) and needs sched patches; higher risk. Keep as fallback.

Concrete code changes:
- **New module** `ggml-phone-shared` (or a shared header used by both backends): the ION allocator + fastRPC registration + OpenCL importer, exposing `{void* base, int dmabuf_fd, size_t size, cl_mem base_mem}`.
- **ggml-hexagon**: factor `ggml_hexagon_shared_buffer::alloc` so the fd/base are retrievable (they already are as members `base`/`fd`, lines 257–259); add an accessor. No change to f16 `set_tensor` (already plain memcpy).
- **ggml-opencl**: add the import branch (new `alloc_buffer` variant for `phone_shared`): instead of `clCreateBuffer(CL_MEM_READ_WRITE)` (line 8815), call `clCreateBuffer(CL_MEM_READ_ONLY | CL_MEM_EXT_HOST_PTR_QCOM, size, &host_ptr_struct)`. Add device-extension probing at init. For f16 weight tensors, `init_tensor`/`set_tensor` must **skip** the `clEnqueueWriteBuffer` and instead create the sub-buffer view over the imported base. Guard the whole path behind capability detection so non-Adreno OpenCL is unaffected.
- **Buffer-type alignment**: expose `get_alignment` = max of the three constraints above.

### Fallback if a single copy is impossible (quantized, or import unsupported)

If the phone ships a quantized shard, or the Adreno driver lacks the ION/dmabuf-host-ptr extension:
- **Per-engine disjoint copies of divergent tensors only.** Keep the shared ION buffer for everything byte-compatible (f16 norms, embeddings-adjacent f16, any f16 weights) and allocate a **second, engine-native** buffer for the quantized matmul weights (Hexagon tiled in ION; OpenCL SoA/image in its own device buffer). Memory cost = one extra copy of only the quantized matmul weights per layer.
- **Split the layer by matmul ownership.** Since prefill→NPU and decode→GPU, and both need the *same* projection weights, you cannot cleanly split a single layer's weights by engine without duplicating them — so for quantized this collapses to the disjoint-copy fallback. Document honestly: **true single-copy sharing for quantized weights is infeasible with the current backends.** The fp16 target avoids this.

### Risks / unproven

- The ION/dmabuf → OpenCL import extension availability on **each** phone's Adreno driver is unverified; must be probed on OP15 (Adreno 840) and OP12 (Adreno 750). If only the older `cl_qcom_ion_host_ptr` exists, use it; rpcmem's fd is a dmabuf, and the ION variant also accepts an ion fd on these stacks — confirm by a smoke test that reads back a known pattern through both engines.
- Sub-buffer alignment vs galloc packing: if ggml packs f16 tensors tighter than the OpenCL base-addr alignment, sub-buffer creation fails; enforce via the buffer-type alignment. Verify no per-tensor overlap.
- The fused-backend approach is a non-trivial ggml addition; de-risk with a standalone unit test (allocate one ION buffer, write an f16 weight, run a matmul on HTP and the same on Adreno via the imported cl_mem, compare against CPU reference) before wiring into the pipeline.

---

## 5. Continuous Batching & Prefill/Decode Scheduling

### 0. Where the scheduler lives

The scheduler is a single **orchestrator process on the A6000 host**. It is the only stateful controller; the two phones are stateless compute stages that execute an assigned layer sub-graph on an activation tensor and return an activation tensor. The orchestrator owns:

- the **ready-queue** and admission/eviction logic (continuous batching),
- the **batch table** (slot -> sequence mapping, the KV *routing* metadata — not the KV itself),
- the **sampler**, the token-embedding lookup, the final norm + `lm_head`,
- the **microbatch pipeliner** that keeps all three physical stages busy.

Pipeline order per step: `server-body (layers 0..44) -> OP15 (2 layers) -> OP12 (1 layer) -> server-head (final norm + lm_head + sample) -> embed next token -> repeat`. Layer counts are the baseline split; **confirm layer total and per-layer tensor shapes from gguf_dump** (spec says ~48 layers, hidden ~3840, GQA, vocab ~256k, per-layer fp16 ~0.4–0.5 GB).

**KV is distributed and never crosses the wire.** Each stage owns the KV for *its* layers. The orchestrator's batch table therefore stores only `{seq_id -> slot, position, length, phase}`; the actual K/V tensors for OP15's 2 layers live in OP15's paged KV pool, OP12's in OP12's, server's 45 layers in A6000 VRAM. Admit/evict is a control-plane broadcast of "slot s now belongs to seq q at position p", executed identically on every stage so the three KV pools stay index-aligned.

### 1. Critical structural finding (drives the whole design)

llama.cpp's **in-tree `ggml_backend_sched` pipeline parallelism cannot span this pipeline.** `src/llama-context.cpp:366-390` enables it only when *every* non-CPU device advertises `caps.async && caps.events`. Verified in this tree:

- `ggml-rpc`: `set_tensor_async = NULL`, comment "we don't have any async operations" (`ggml-rpc.cpp:654,726-729`);
- `ggml-hexagon`: `caps.events = false` (`ggml-hexagon.cpp:3640`);
- `ggml-opencl`: `caps.async = false, caps.events = false` (`ggml-opencl.cpp:8903-8906`).

So `pipeline_parallel` auto-disables and the scheduler serializes stage-by-stage. **Cross-stage overlap must be built at the orchestrator level**, by issuing multiple in-flight microbatches over (thread-parallel) blocking RPC calls — not by relying on `sched`'s `n_copies` double-buffering. This is the single most important buildability fact in this subsystem and it is confirmed from source, not assumed.

### 2. Ready-queue and batch-formation trigger

Two queues on the orchestrator:

```
prefill_queue : arrivals (8/s), FIFO, each carries {seq_id, prompt_tokens, arrival_ts}
ready_set     : sequences that have KV populated + first token sampled, awaiting decode
active_batch  : up to S = 64..128 slots currently in the decode wave (S = LLAMA_MAX_SEQ-bounded; branch raised LLAMA_MAX_SEQ 256->1024, so slot count is not the limit)
```

**Trigger (hysteresis, not a hard gate):** start/continue a decode wave when `|ready_set| + |active_batch| >= LOW=64`; top the batch up toward `HIGH=128` from `ready_set` at every step; if `ready_set` drains and `active_batch < LOW`, keep decoding the residual (do not stall live sequences) but flag under-occupancy to the autoscaler. A **max-wait timer** (e.g. 150–250 ms, ~1–2 arrival intervals) forces a wave even below LOW so tail latency is bounded when arrival is bursty. This is the classic occupancy-OR-timeout formation rule; the timeout value should be tuned to the measured phone stage time (Section 6) so a partly-full wave is never cheaper to *not* run.

### 3. Prefill/decode disaggregation and how they share the 3 devices

Prefill and decode contend for the same three physical engines but want **opposite** engines intra-phone (measured: prefill -> NPU/HMX compute-bound; decode -> GPU memory-bound; zero interference co-scheduling). Two-level policy:

**Intra-phone (free win):** run the prefill sub-graph on the phone **NPU** and the decode sub-graph on the phone **GPU concurrently**, sharing one mmap'd weight copy via fastRPC. The project measured zero interference for compute(NPU)+memory(GPU) on one phone, so a phone can host a prefill microbatch and a decode microbatch in the same wall-clock window. This is the main lever that lets prefill@8fps and the decode wave coexist without time-slicing the phone.

**Cross-stage schedule — chunked prefill (Sarathi-Serve style), server-side:** the server body (45 layers, the heavy stage) *cannot* run prefill-NPU and decode-GPU tricks (it is one CUDA context), so there we interleave in the token dimension. Each pipeline step the orchestrator builds **one fused ubatch** = `[decode tokens: 1 per active seq] ++ [a bounded prefill chunk of C tokens from the head prefill_queue seq]`. Choose `C` so the fused ubatch's server-body time stays ~constant (prefill tokens are cheap on A6000 which is compute-rich; the cap exists to protect decode-step latency). This piggybacks prefill onto decode steps, avoiding convoy stalls where a long prompt blocks the decode wave. hexagon's quantized `MUL_MAT` already batches over the token dim `ne[1]` (spec), so the phones accept a fused `[decode ++ prefill-chunk]` ubatch natively as long as `ne[2]/ne[3]` are not broadcast.

**Attention correctness for the fused ubatch:** decode rows attend to their own full KV history; prefill-chunk rows attend causally within the chunk + already-committed prefix. This needs a **block-diagonal / per-row attention mask** — llama.cpp's unified-KV masked attention already expresses this on CPU/CUDA, but **on the phone backends it is exactly the unproven path** (Section 7).

### 4. Pipeline microbatching (hiding hops and phone latency)

With `sched` overlap unavailable (Section 1), the orchestrator runs a **software pipeline of K in-flight microbatches** (K = number of stages = 3, +1 for hop slack ≈ 4). Split the S-way decode batch into K microbatches of ~S/K sequences. At each tick, stage boundaries pass activation tensors forward while the next microbatch enters stage 0:

```
t0: MB0@server-body
t1: MB0@OP15         MB1@server-body
t2: MB0@OP12         MB1@OP15          MB2@server-body
t3: MB0@server-head  MB1@OP12          MB2@OP15          MB0-next@server-body
... steady state: all 3 stages + both phone engines busy every tick
```

Steady-state throughput = `S / max(stage_time)`. Each stage handoff is one RPC of an activation tensor; **comms is latency- not bandwidth-bound** (batch-128 hidden-3840 fp16 = 0.94 MB/hop; 512-tok prefill = 3.9 MB — both are 100s× under USB ~600 MB/s / WiFi ~100 MB/s). Microbatching hides both the per-hop RTT and the phone stage time behind other microbatches' compute. Implement handoffs as **async (thread-pool) blocking RPC** calls — one worker thread per stage edge — since the RPC backend itself is synchronous.

**USB no-peer trap folded into the schedule:** OP15->OP12 cannot peer over USB; the activation must route through the host (USB tethering + host IP-forward, or a shared WiFi LAN). So the physical transfer graph is `host<->OP15`, `host<->OP12` (4 host<->phone transfers/step), not a clean ring. Over **WiFi** at 0.94 MB that is ~4 × 9.4 ms ≈ 38 ms/step of comms — comparable to the phone compute stage and a real bottleneck. Over **USB** it is ~4 × 1.6 ms ≈ 6 ms — negligible. **Recommendation: USB tethering with host IP-forwarding**, WiFi only as fallback. (Confirm effective USB throughput and RTT by measurement.)

### 5. Server owns sampler + batch table

The final hidden state loops back to the server every step; server does final norm + `lm_head` (huge, ~256k vocab — stays on A6000) + sampling, then embeds the sampled token and re-injects it at stage 0. Centralizing the sampler means: (a) no logits ever cross the wire (only 0.94 MB hidden states do), (b) admission/eviction and stop-condition checks are single-writer on the batch table, (c) grammar/penalty/logit-bias state stays in one place. The batch table is the authority; each decode tick it emits the per-stage slot map so all three KV pools evict finished slots and admit new ready sequences in lock-step.

### 6. Throughput / latency budget (honest, numbers flagged)

Per-layer fp16 weights ≈ 0.45 GB (**confirm from gguf**). Roofline knee for phone GPU decode ≈ M=64–128 (measured); at M=128 the phone is **compute-bound**, so a phone stage's time is ~`FLOP / TFLOPS`, not `bytes / BW`.

Per-token-per-layer decode FLOP ≈ `2 × params_per_layer ≈ 2 × 225M = 0.45 GFLOP` (params/layer ≈ 0.45 GB / 2 B). Estimated stage times at M=128 (**all confirm-by-measurement**):

| Stage | Layers | Compute @ M=128 | Bound | Est. stage time |
|---|---|---|---|---|
| server-body (A6000) | 45 | 45×128×0.45G = 2.6 TFLOP; also mem 20 GB/768 GB/s | memory-bound (M=128 < A6000 knee ~200) | ~26–30 ms |
| OP15 (Adreno 840) | 2 | 2×128×0.45G = 115 GFLOP @ ~0.95 TFLOPS (xmem) | compute-bound | ~60–120 ms |
| OP12 (Adreno 750) | 1 | 1×128×0.45G = 58 GFLOP @ ~0.4–0.9 TFLOPS | compute-bound | ~65–145 ms |
| server-head (lm_head) | — | 128×256k×3840×2 ≈ 250 GFLOP | memory-bound | ~5–10 ms |

`max(stage) ≈ 120 ms` (a phone). Aggregate decode ≈ `128 / 0.120 s ≈ ~1000 tok/s`; per-sequence ≈ 8 tok/s.

**Sustainability check:** need `aggregate_decode >= 8 req/s × L` (L = mean output length). ~1000 tok/s supports **L ≈ 120 tokens**. Prefill@8fps is cheap and hides on the A6000 (a 512-tok prompt = 2.6 TFLOP/layer of compute A6000 eats easily; phone prefill runs on NPU/HMX at 7.12 TFLOPS). So the **binding constraint is the decode phone stage**, and only marginally.

**Brutal honesty about the baseline:** moving 3 of ~48 layers to phones *reduces* system throughput. Server-only 48-layer decode at batch 256 is memory-bound at ~24 GB/768 GB/s ≈ 31 ms/step ⇒ ~8000 tok/s and floors ~0.15 J/tok. The 3-phone pipeline is ~1000 tok/s — roughly **8× worse throughput** — because two slow, poorly-balanced compute-bound stages (120 ms) are inserted behind a fast 30 ms stage, leaving the server idle ~75% of each pipeline period. **The baseline saves no energy and lowers throughput; it exists only to validate plumbing.** Energy/throughput wins require the *later* milestones that move many layers to phones **and** shrink the server stage proportionally so stages balance — at which point per-stage time equalizes and phone batched decode (~0.05–0.08 J/tok, unverified) can undercut the server. State this to the team up front.

### 7. THE risk: batched decode on phone is UNPROVEN — de-risk before anything else

The entire decode-batching value of this scheduler rests on a capability that is **not proven on the phone backends**:

- **Proven:** "batch M rows in one matmul" — a single `MUL_MAT` with `ne[1]=M` tokens. The 7.12 TFLOPS roofline and the xmem GEMM knee are exactly this. hexagon quantized `MUL_MAT` batches over `ne[1]`.
- **Unproven / recorded FAILURE:** "**N sequences with separate KV pools + a block-diagonal masked attention** decoding concurrently." The project recorded **"NPU n_parallel=2 HANGS" on op15**. A synthetic M-sweep matmul is *not* N-sequence continuous batching: it has no per-sequence KV, no paged-attention gather, no block-diagonal mask, no admit/evict churn. Section 3's fused `[decode ++ prefill-chunk]` ubatch and Section 4's per-microbatch masked attention **are** this unproven path.

**De-risk plan (must pass before building the pipeliner):**

1. On each phone in isolation, `llama-batched-bench` / `llama-batched` with `-np 2,4,8,16,32` on a small staged model (gemma-4-E2B-Q4 already on op12), Hexagon backend, and confirm it does not hang and that per-step time scales sub-linearly with `-np` (the batching win). Repeat on op15 (v81) where the hang was seen.
2. If NPU hangs persist, fall back to **decode-on-GPU (OpenCL) with npl>1** on the phone (decode is a GPU workload in this design anyway; NPU is for prefill). Prove multi-sequence masked attention on `ggml-opencl` first — it is the actual decode engine.
3. Validate the **masked block-diagonal attention op** on-device via `test-backend-ops` for `FLASH_ATTN_EXT` (spec: requires dst `ne[3]==1`) and `SOFT_MAX` with a real per-row mask, at N=2..32, before trusting the fused ubatch.
4. Only after 1–3 pass on-device do the software pipeliner (Section 4) and chunked-prefill fusion (Section 3) become real; until then the scheduler runs **single-microbatch, single-sequence-per-stage** as a correctness harness.

Treat Sections 3–4 as *contingent* on this de-risk. The scheduler is buildable today in a degraded serialized single-sequence mode (proven ops only); the batched/pipelined mode is gated on the phone-decode-batching experiment.

### 8. Backpressure / overload

At 8 fps sustained, if `ready_set` grows faster than the decode wave drains (L too large, or a phone throttles via DVFS), apply backpressure: (a) shed/queue new arrivals with a bounded admission queue, (b) shrink the phone layer share back toward server (rebalance), (c) lower `HIGH` occupancy to cut stage time. Heterogeneous DVFS/thermal throttling of v81 vs v75 causes pipeline bubbles; the microbatch pipeliner should **measure each stage's rolling latency and size microbatches per-stage** (smaller microbatch to the slower phone) rather than assuming a static S/K split.

---

## 6. Distributed KV Cache & Intra-Phone NPU/GPU Execution

This subsystem owns two things: (1) how KV cache is partitioned so **each pipeline stage stores KV only for its own layers** and never ships KV over the wire, and (2) how each phone runs **prefill on the Hexagon NPU/HMX and decode on the Adreno GPU concurrently** out of one mmap'd weight copy. It ends with the single most important de-risking experiment for Design A: proving multi-sequence continuous-batched decode (FA + masked softmax for N sequences) actually runs on the phone backend.

### 0. Model facts (confirm from gguf via `gguf_dump` on the real 12B shard)

The repo only ships `models/ggml-vocab-gemma-4.gguf` (vocab-only, no arch metadata), so the numbers below are the working assumptions from the project brief and must be confirmed from the real weight gguf before sizing buffers.

| Field | Working value | Source |
|---|---|---|
| `block_count` (n_layer) | ~48 | brief — confirm from gguf |
| `embedding_length` (d_model) | ~3840 | brief — confirm from gguf |
| n_head / n_kv_head | 16 / 8 (GQA) | assumption — confirm from gguf |
| head_dim | 256 (Gemma family) | assumption — confirm from gguf |
| KV width/layer = n_kv_head·head_dim | 2048 per K, 2048 per V | derived |
| vocab | ~256k | brief — confirm from gguf |
| per-layer fp16 weight | ~0.4–0.5 GB | brief — confirm from gguf |
| local/global attention pattern | Gemma-3/4 uses interleaved **sliding-window (local)** + **global** layers, ~5:1 | assumption — **confirm from gguf** (`*.attention.sliding_window`, per-layer type). This is the single biggest KV lever; see §2.3 |

**KV cost per token per layer (fp16, global layer):**
`kv_bytes = 2 (K,V) · n_kv_head · head_dim · 2 B = 2·8·256·2 = 8192 B = 8 KiB/token/layer` (confirm from gguf).

### 1. KV ownership: each stage stores only its layers' KV

llama.cpp already builds one `llama_kv_cache` sized to the **local model shard** (the layers actually instantiated on that node). Because each node loads only its assigned layers, its KV cache is automatically scoped to those layers — no code change is needed to "partition" KV; it falls out of loading a layer subset per node. The design contract:

- **Server (A6000):** owns embeddings + ~45 layers + final norm + lm_head + sampler → holds KV for ~45 layers.
- **OP15:** owns 2 layers (baseline) → holds KV for 2 layers.
- **OP12:** owns 1 layer (baseline) → holds KV for 1 layer.
- **KV never crosses the interconnect.** Only the hidden-state activation crosses each hop (batch-128 × d3840 × fp16 ≈ 0.94 MB/hop; 512-token prefill ≈ 3.9 MB — both 100s× under USB/WiFi capacity, per measured findings). Each hop carries `[d_model, n_tokens_in_batch]`, not KV.
- **Sequence identity must be stable across stages.** The server scheduler assigns a global `seq_id` (llama.cpp `LLAMA_MAX_SEQ = 1024`, ample for batch 64–128) and every stage keys its KV pages by that same `seq_id`. Admit/evict decisions are made **only on the server** (it owns the scheduler) and broadcast as control messages; phones apply the same `seq_id → KV-slot` mapping so pages stay aligned. A per-step batch descriptor (list of active `seq_id`s + their positions) travels with the activation so each stage writes KV into the right slots.

### 2. Paged KV, per-device budget, growth & eviction

#### 2.1 Paging model
Use llama.cpp's existing PagedAttention-style block KV **per stage**, unified across sequences (`llama_cparams.kv_unified = true` — the flag exists in `src/llama-cparams.h`). Unified KV is not just an optimization here; it is a **hard requirement** for the phone attention path (see §5): all N sequences share one physically-contiguous K/V buffer per layer and are separated by a **block-diagonal mask**, which keeps the attention op's batch dim `ne[3] == 1` — the exact thing the Hexagon FLASH_ATTN_EXT and SOFT_MAX kernels require.

Block size: default 256 tokens/block is fine on the server; on phones use a **small block (e.g. 32–64)** so that partially-filled sequences waste less LPDDR (phone RAM is the binding constraint, §2.4).

#### 2.2 Per-device KV budget (fp16, all-global-layer worst case, 8 KiB/tok/layer)

| Config | Layers | KV/token | KV @ 128 seq × 4096 ctx | KV @ 64 seq × 2048 ctx |
|---|---|---|---|---|
| Server (~45 L) | 45 | 360 KiB | ~180 GB ❌ (won't fit 48 GB) | ~45 GB (tight) |
| OP15 (2 L) | 2 | 16 KiB | ~8.0 GB ❌ | ~2.0 GB |
| OP12 (1 L) | 1 | 8 KiB | ~4.0 GB ⚠ | ~1.0 GB |

The server row shows Design A's real wall is **server KV**, not the phones: measured findings already record the A6000 OOMs at batch 320 for 12B fp16 on the KV-memory wall. The 45-layer server share at 128×4096 is far past that — so **either context is capped (~2k), or KV is quantized (q8_0/q4 KV), or sliding-window layers cut most of it (§2.3), or more layers move to phones**. This is the dominant sizing constraint and must be resolved before the batched-decode milestone.

#### 2.3 Sliding-window layers are the primary KV reducer (confirm from gguf)
If Gemma-4 keeps the Gemma-3 pattern (5 local : 1 global, local window ~1024), then ~5/6 of layers cap KV at the window regardless of context. Effective KV/token collapses from 8 KiB×n_layer to roughly `(n_global·8KiB) + (n_local·8KiB·min(ctx,window)/ctx)`. At 4096 ctx with window 1024 this is ~a 3–4× reduction. **Action: read the per-layer attention type + `sliding_window` from the gguf and size local-layer KV to the window, not to n_ctx.** Whether OP15's 2 baseline layers are local or global materially changes its phone budget (2 global = 16 KiB/tok; 2 local@1024 = fixed 4 MiB/seq).

#### 2.4 Phone RAM budget after weights
Per phone: `RAM_total − OS/runtime (~2–3 GB) − weights (mmap, shared) − Adreno/Hexagon scratch − KV`.
- OP15 weights: 2 layers × ~0.45 GB ≈ **0.9 GB** (mmap, one physical copy — §4).
- OP12 weights: 1 layer × ~0.45 GB ≈ **0.45 GB**.
- Weights are small; **KV dominates the phone footprint.** With a 12–16 GB phone and ~10 GB usable, OP15 at 2 global layers can hold ~128 seq only up to ~2–2.5k ctx before it competes with the GPU decode scratch. Enforce a **per-device admission cap** `max_active_seq × max_ctx` derived from measured free RAM, and have the server scheduler respect the *minimum* cap across all stages (the pipeline can only admit what the tightest phone can hold).

#### 2.5 Growth, eviction, preemption
- **Growth:** paged — allocate blocks on demand as positions advance; no giant pre-alloc.
- **Eviction/preemption:** driven by the **server scheduler only** (it owns continuous batching). When the server evicts/preempts a `seq_id`, it emits a control message; every stage frees that seq's KV blocks. Because KV is stage-local, eviction is a local free on each node — no cross-node KV migration. A preempted sequence's KV is dropped everywhere and recomputed on resume (recompute, not swap, to avoid shipping KV over the wire — which the constraints forbid anyway).
- **Backpressure:** if any phone hits its RAM cap, it NAKs admission; the server holds the sequence in the waiting queue. This makes the tightest phone the flow-control point (a Design-A trap: heterogeneous stages, §Open questions).

### 3. Intra-phone graph split: which ops to which backend

On each phone the layer subgraph is split across **two ggml backends registered in one `ggml_backend_sched`**: `ggml-hexagon` (NPU) and `ggml-opencl` (Adreno). The split is chosen by *phase*, not by op type:

- **Prefill ubatch (M = 512-ish tokens, compute-bound):** route the whole layer graph to **Hexagon/HMX**. Confirmed supported ops on the backend cover the full layer: `MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT`, `SOFT_MAX`, `RMS_NORM`/`NORM`, `ROPE`, `ADD`/`MUL`/activations (verified in `ggml-hexagon.cpp` op dispatch, lines ~3186–3210). HMX is eligible only for `M > HTP_MM_HMX_MIN_NROWS` (the diff adds `GGML_HEXAGON_HMX_MIN1` to force M=1..4 onto HMX for measurement only — do **not** use it in production; prior memory shows the fake-batch K-split for M=1 is infeasible). Prefill's M=512 is squarely in HMX's sweet spot (measured 7.12 TFLOPS @ M=512 on op12).
- **Decode step (M = N sequences, one token each, memory-bound GEMV):** route the layer graph to **Adreno/OpenCL**. Decode is bandwidth-bound; the phone GPU is the right engine (measured ~0.4 TFLOPS stock, ~0.95 with xmem at batch ≥16, knee M≈64–128). For batch ≥16 the f16 path uses `mul_mm_f16_f32_l4_lm`; the xmem image2d GEMM (env `GGML_OPENCL_ADRENO_XMEM_GEMM`, batch ≥16) is the fast path to enable once correctness is proven.

**Concurrency:** prefill (NPU) and decode (GPU) run **at the same time** on one phone — the project measured **zero interference** for compute(NPU)+memory(GPU) co-scheduling. Implementation: two `ggml_backend_sched` instances (or two graph streams) with independent command queues — NPU drains prefill ubatches for newly-arrived requests while GPU drains decode steps for the ready batch. They contend only on LPDDR bandwidth, and the measurement says HMX-compute vs GPU-memory don't collide.

### 4. Shared mmap weight buffer consumed by both engines

The sharing subsystem provides **one mmap'd, fastRPC-registered physical copy** of each phone's layer weights. This subsystem consumes it as follows:

- Weight tensors live in an **ION/dma-buf** region mmap'd into the app, registered with fastRPC so the **Hexagon** side sees it via a shared handle, and imported into **OpenCL** as an external/`cl_mem` (host-ptr or dma-buf import) so the **Adreno** side reads the *same physical pages*. No duplicate weight buffers (a hard constraint).
- Quantized weights on the Hexagon path must be **repacked** (`ggml_backend_buffer_is_hexagon_repack`, enforced in `ggml_hexagon_supported_mul_mat`). For fp16 (this project's weights) no repack is required, so the *same* fp16 bytes feed both the HMX prefill matmul and the Adreno decode GEMV — clean sharing. **If a q-format is later used, repack layout diverges between NPU and GPU and the "one physical copy" claim breaks** — flag to the sharing subsystem.
- Because both engines read weights from the same LPDDR pages, the concurrent NPU-prefill + GPU-decode of §3 is exactly two readers of one buffer; that is what the zero-interference measurement covers.

#### 4.1 Prefill→decode handoff needs ZERO interconnect transfer
When a request finishes prefill on the NPU, its **KV blocks are already written into the stage-local paged KV in LPDDR**. Decode on the GPU reads those same KV pages directly — **no copy, no interconnect hop, no NPU→GPU DMA** — because both engines address the same LPDDR and (per constraint) KV lives in shared/importable buffers. The handoff is purely a scheduler event ("seq X is now in decode set"): the GPU decode graph simply starts including seq X's KV blocks. This is the intra-phone analogue of "KV never crosses the wire." **Requirement:** the KV cache buffer must itself be allocated in the shared dma-buf region (not a Hexagon-private VTCM/DDR buffer) so both backends can address it; verify the KV buffer type is the shared LPDDR buffer, not an engine-private one.

### 5. UNPROVEN: multi-sequence batched decode; FA + masked softmax for N sequences on phone

This is the **critical open risk of Design A** and this subsystem's top de-risk item. The roofline M-sweep (7.12 TFLOPS @ M=512) is a **single synthetic matmul**, not N sequences with separate KV and a block-diagonal mask — it does **not** prove continuous batching runs on the phone backend. The project already recorded **"NPU n_parallel=2 HANGS" on op15.** Decode is on the GPU here, but the same N-sequence attention correctness question applies to whichever engine runs it.

**Why the mapping is delicate — the `ne[3] == 1` constraint:**
`ggml_hexagon_supported_flash_attn_ext` **rejects any FA op with `dst->ne[3] != 1`** (ggml-hexagon.cpp line ~1911), and `ggml_hexagon_supported_softmax` requires F32 src/dst and forbids sinks. In llama.cpp, **unified KV (`kv_unified=true`) packs all N sequences into one K/V buffer and one FA call with batch dim `ne[3]==1`**, using a **block-diagonal mask (src3, F16 — matches the backend's mask-type check)** to prevent cross-sequence attention. So:
- **Unified KV path → FA `ne[3]==1` → phone-supported.** This is why §2.1 mandates `kv_unified=true`.
- **Non-unified / per-sequence path → FA `ne[3]==N>1` → the phone backend refuses the op** and it silently falls back (to CPU or fails), destroying the win. Any code path that produces `ne[3]>1` on the phone breaks the design.

**Still unproven even with unified KV:** (a) that the block-diagonal F16 mask + masked softmax produce *correct* per-sequence outputs on the Hexagon/Adreno kernels for N≥2; (b) that it doesn't hang as n_parallel=2 did; (c) that GQA broadcast (n_head=16 attending 8 kv-heads) is handled — note `ggml_hexagon_supported_softmax` requires `src0->ne[2] % src1->ne[2] == 0` (mask head-broadcast) and MUL_MAT forbids ne[2]/ne[3] broadcast for quantized, so shapes must be arranged as GQA-contiguous, not broadcast. Verify on the OpenCL decode kernels too.

#### 5.1 Exact validation experiment (do this FIRST, before any rebalancing)
Use the already-staged `llama-batched-bench` on a **small model already on op12** (gemma-4-E2B-Q4 or Llama-3.2-1B-Q4) to prove N-sequence batched decode runs and is correct on each phone backend **in isolation**, before wiring the 3-stage pipeline.

**Step A — does batched decode even run on the NPU backend (the hang repro)?**
```
# On op12 and op15, NPU backend, escalate parallelism:
llama-batched-bench -m gemma-4-E2B-Q4.gguf \
  --device HEXAGON0 -ngl 99 \
  -c 4096 -npp 512 -ntg 128 \
  -npl 1,2,4,8,16,32,64 \
  -fa on            # force FLASH_ATTN_EXT path (unified KV)
# PASS: completes for npl>=2 without hanging; FAIL: reproduces "n_parallel=2 HANGS".
```
**Step B — same on the Adreno decode backend (this is the real decode engine):**
```
llama-batched-bench -m gemma-4-E2B-Q4.gguf \
  --device GPUOpenCL -ngl 99 \
  -c 4096 -npp 512 -ntg 128 \
  -npl 16,32,64,128 -fa on
# Then re-run with GGML_OPENCL_ADRENO_XMEM_GEMM=1 (batch>=16 fast path).
```
**Step C — correctness, not just "it ran".** Compare per-sequence decoded token streams against a single-sequence run and against CPU/CUDA reference (greedy, temp=0):
```
# Reference (trusted backend):
llama-batched -m gemma-4-E2B-Q4.gguf -ngl 0 -np 1 --temp 0 -p "<fixed prompt>" -n 64
# Phone batched, same prompt replicated across np sequences:
llama-batched -m gemma-4-E2B-Q4.gguf --device GPUOpenCL -np 8 --temp 0 -p "<fixed prompt>" -n 64
# PASS: all 8 sequences produce IDENTICAL tokens to the reference (proves the
#       block-diagonal mask isolates sequences and masked softmax is correct).
# Then use DISTINCT prompts per sequence to catch cross-sequence KV bleed.
```
**Step D — targeted op test.** Extend `tests/test-backend-ops.cpp` (already modified in this branch) with a FLASH_ATTN_EXT + SOFT_MAX case shaped like N-sequence unified-KV decode: `Q=[head_dim, n_head, N, 1]`, unified `K/V`, F16 block-diagonal mask, GQA ratio 16:8, assert `dst->ne[3]==1`, and diff against the CPU backend. This isolates the kernel from the full model.

**Gate:** Design A does not proceed to rebalancing layers onto phones until Steps A–D pass on both op12 and op15 for at least npl=16 (the batching threshold below which phone decode is energy-poor per measured findings). If the NPU hangs (Step A) but the GPU passes (Step B), that's acceptable — decode is a GPU job here; but it must be documented, because it means the NPU is prefill-only and cannot be a decode fallback.

### 6. Summary of buildable contracts
- KV is stage-local by construction (load layer subset → KV scoped to it); never transmitted.
- `kv_unified=true` is mandatory (satisfies phone FA/softmax `ne[3]==1`).
- Paged KV, small block on phones; server scheduler is the sole admit/evict/preempt authority; phones NAK on RAM cap.
- Phone RAM is bounded by KV, not weights; sliding-window layers (confirm from gguf) are the main KV reducer; server KV is the true OOM wall.
- Prefill→NPU, decode→GPU, concurrent, zero measured interference; handoff is a scheduler event with **zero copy** because KV + weights live in shared LPDDR.
- The N-sequence batched-decode correctness/hang risk is gated behind the §5.1 `llama-batched-bench` experiment, run before any layer rebalancing.

---

## 7. Orchestration, Energy Accounting, Observability & Failure

This section designs the **control plane** for Design A: the 3-stage pipeline `A6000 -> OP15 (2 layers) -> OP12 (1 layer) -> A6000 (final norm + lm_head + sample)`. It covers request lifecycle and admission control at 8 fps, health/heartbeat, **energy accounting as a first-class deliverable** (unified J/tok against the ~0.15 J/tok A6000 baseline), observability, thermal/DVFS policy, and failure recovery.

Design A was chosen deliberately over the hub-spoke Design B that a prior review preferred. The control plane's job is therefore to **surface and instrument Design A's structural risks so they are measurable, not hidden** - specifically the inter-phone USB routing bounce, heterogeneous-stage bubbles, and the unproven phone continuous-batched decode. We build A and make its costs legible; we do not silently substitute B.

**The single most important thing this control plane must do first** is answer an open feasibility question before any energy claim is meaningful: *does multi-sequence continuous-batched decode even run on the phone NPU?* The project already recorded `NPU n_parallel=2 HANGS` on op15 (deadlock, spins forever; npl=4/8/16/... are fine). Milestone 0 below is a hard gate on this.

---

### 1. Control-plane topology

`ggml-rpc` is a **data plane only**. Its command set (confirmed in `ggml/src/ggml-rpc/ggml-rpc.cpp`, enum `rpc_cmd`) is `ALLOC_BUFFER, GET_ALIGNMENT, GET_MAX_SIZE, BUFFER_GET_BASE, FREE_BUFFER, BUFFER_CLEAR, SET_TENSOR, SET_TENSOR_HASH, GET_TENSOR, COPY_TENSOR, GRAPH_COMPUTE, GET_DEVICE_MEMORY, INIT_TENSOR, GET_ALLOC_SIZE, HELLO, DEVICE_COUNT, GRAPH_RECOMPUTE`. There is **no heartbeat, no health, no deadline/timeout, and no cancel command**. `GRAPH_COMPUTE` is a synchronous blocking round-trip (`send_rpc_cmd(... output ...)`); a wedged phone backend blocks the caller indefinitely with nothing in the protocol to break it. Therefore **liveness, deadlines, admission, energy, and failover must all live in a separate out-of-band control plane the orchestrator owns.** We do not modify the RPC wire format (keeps us mergeable and avoids the "new subsystem" objection in AGENTS.md).

```
                 +-------------------- A6000 host (Linux) ----------------------+
                 |  llama.cpp server (embeddings + ~45 layers + norm + lm_head  |
                 |  + sampler + continuous-batch scheduler)  <-- the ORCHESTRATOR|
                 |  control-plane daemon (this subsystem):                       |
                 |    - admission / backpressure (8 fps)                         |
                 |    - per-hop deadline watchdog + failover                     |
                 |    - energy collector (nvidia-smi)  + aggregator              |
                 |    - metrics/trace sink (Prometheus text + JSONL)             |
                 +----+-------------------------------+------------------------+
                      | ggml-rpc data (TCP)           | control (TCP, separate port)
        USB tether    v                               v
   host-forwarded  OP15 (v81, 2 layers)          OP12 (v75, 1 layer)
     IP route      phone-agent daemon            phone-agent daemon
        \______ OP15->OP12 hop MUST route via host IP (USB is host-centric) _____/
```

**Key Design-A plumbing fact to instrument, not wish away:** over USB the phones are host-centric *devices* and cannot peer. The `OP15 -> OP12` activation handoff physically transits the host (USB tether + host IP-forwarding, or all nodes on one WiFi/LAN). The control plane must (a) make that route explicit config, and (b) *measure* the extra half-round-trip so the pipeline bubble budget is honest. Activation sizes are tiny (batch-128 hidden-3840 fp16 ~= 0.94 MB/hop; 512-token prefill ~= 3.9 MB), hundreds of times under USB ~600 MB/s / WiFi ~100 MB/s, so this is a **latency** line item, not bandwidth - budget ~0.1-1 ms/hop over USB, more over WiFi, and confirm by measurement.

### 1.1 Phone-agent daemon (new, small, per phone)

A tiny userspace daemon on each phone, launched over `adb shell`, colocated with the existing `llama.cpp` RPC server binary. Responsibilities:
- Own the **control TCP socket** back to the host (separate port from RPC data).
- Emit **heartbeat + telemetry** every `T_hb` (default 500 ms): monotonic seq no, last-completed-graph timestamp, thermal zones, DVFS clocks, `charge_counter`, `voltage_now`, engine in use (HTP0 vs GPUOpenCL).
- Execute **control verbs** from host: `set_governor`, `pin_or_release_clocks (best-effort)`, `disable_charging`, `drain` (stop accepting new microbatches), `reprovision (verify shard hash)`.
- It does **not** touch weights at runtime. Per the hard constraint, each phone's assigned layer tensors are pre-provisioned shard files on local storage and **mmap'd** so the Hexagon NPU and Adreno GPU share one physical copy via fastRPC. The agent's only job around weights is to **verify the shard file hash at startup** (reuse `gguf_hash.py` / the SET_TENSOR_HASH machinery) and refuse to start on mismatch.

---

### 2. Orchestration: request lifecycle and admission control

### 2.1 Two regimes, one pipeline

- **Prefill (compute-bound):** each incoming request (8/s) is prefilled through the pipeline as it arrives. On phones this maps to the **NPU/HMX** (op12 measured 7.12 TFLOPS at M=512; op15 knee ~M=512, peak ~11.66 TFLOPS). Race-to-idle is the right energy policy here.
- **Decode (memory-bound GEMV):** once ~64-128 sequences are ready, batched decode loops the pipeline under **continuous batching** (dynamic admit/evict). On phones this maps to the **Adreno GPU**, *specifically to free the NPU for the next request's prefill* - this is the measured zero-interference co-schedule win (compute on NPU || memory on GPU), not a decode speedup (both engines share the same ~68-70 GB/s LPDDR bus).

The **intra-phone concurrency** the co-schedule result licenses is exactly: while stage N is prefilling request R on the NPU, its GPU services the decode microbatch for already-admitted sequences. The orchestrator schedules the two engines as one logical stage with two lanes.

### 2.2 Request lifecycle (state machine)

```
NEW --admit--> PREFILL_S0(A6000) -> PREFILL_S1(OP15,NPU) -> PREFILL_S2(OP12,NPU)
    -> PREFILL_HEAD(A6000: norm+lm_head+sample) -> READY(first token, KV resident per-stage)
READY --(batch >= B_lo)--> DECODE_ADMITTED
DECODE loop step: S0(A6000) -> S1(OP15,GPU) -> S2(OP12,GPU) -> HEAD(A6000) -> emit token -> next-embed
DECODE_ADMITTED --EOS/maxlen--> EVICT (free per-stage KV pages) --> DONE
any state --deadline/health fail--> DEGRADE (see S7)
```

**Distributed KV invariant (instrument it):** each stage owns the KV for its own layers; KV **never crosses the wire**. The orchestrator tracks per-stage KV page occupancy but transfers only token IDs / activations. Evictions must be broadcast to all stages so every stage frees the same sequence's pages in lockstep; a lost eviction = per-stage KV leak. The control plane reconciles KV occupancy in every heartbeat and alarms on divergence.

### 2.3 Admission control and backpressure at 8 fps

The pipeline's slowest stage sets the sustainable admit rate. With heterogeneous stages (v81 vs v75) and independent DVFS, the bottleneck **moves at runtime**, so admission is closed-loop, not a fixed rate.

```
每 tick (e.g. 100 ms):
  bottleneck_stage = argmax_s ewma(service_time[s])       # measured, not assumed
  drain_rate = 1 / service_time[bottleneck_stage]
  if inflight_prefill >= K_pipe (== pipeline depth headroom):
        DEFER new frame  (backpressure)
  if any stage.queue_depth > Q_max OR any stage.thermal == THROTTLING:
        shed: drop frame OR route its prefill to server-only fallback (S7)
  else admit frame
```

- **8 fps is the arrival rate, not a guarantee.** If the phone stages cannot drain 8/s of prefill (thermal throttle, DVFS dip), the control plane must *shed to the server-only path* (S7) rather than let the queue grow unbounded - an unbounded queue silently inflates latency and, worse for our thesis, inflates J/tok because devices sit hot while backed up.
- Backpressure is expressed as a **credit window** per stage: the orchestrator holds at most `K_pipe` in-flight prefills (pipeline depth + small slack). This bounds tail latency and keeps the loopback-to-server lm_head from becoming a hidden serialization point.
- **Decode admit/evict (continuous batching):** keep the batch topped to `B_target` in `[64,128]`. Admit READY sequences when `batch < B_target` and all stages report `queue_depth==0` for the decode lane; evict on EOS/maxlen. Never let decode batch drop into the energy-poor small-batch regime: measured op15 fp16 floor is ~98 mJ/tok at B>=64 (GPU) / B>=128 (NPU), rising ~8.8x toward B=1. Batch occupancy is thus an **energy control variable**, tracked in S5.

---

### 3. Health and heartbeat

Because RPC has no liveness, we layer three timescales:

| Mechanism | Period | Detects | Action |
|---|---|---|---|
| Control heartbeat | 500 ms | agent/process death, thermal event, clock dip | mark stage SUSPECT -> DEGRADE if 3 missed |
| Per-hop deadline watchdog | per graph_compute | wedged/slow backend (the npl=2-style hang) | cancel-by-abandon the hop, failover |
| Provisioning check | at start + on reconnect | wrong/corrupt shard, wrong SoC lib (v75 vs v81) | refuse to admit that stage |

**Deadline watchdog detail (this is the load-bearing failure guard).** A hung phone `GRAPH_COMPUTE` blocks the RPC call with no protocol timeout. We wrap every phone hop in a host-side deadline = `p99_service_time[stage] * slack (e.g. 3x)`. On expiry the orchestrator does **not** try to cancel the RPC (there is no cancel verb); it **abandons the connection** (close socket -> the blocking `send_rpc_cmd` returns error), marks the stage DOWN, and fails that sequence over to the server-only path. The phone agent, seeing its data socket dropped, resets its RPC server (kills/reforks the llama.cpp RPC process) to clear the wedged HTP0 state - this is the concrete recovery for the observed NPU deadlock class.

Heartbeat payload is also the **telemetry transport** (S4/S5): folding metrics into the heartbeat avoids a second polling loop competing with `adb`.

---

### 4. Energy accounting (first-class)

**Goal:** report end-to-end **J/tok in the same units as the A6000 baseline table** (net decode J/tok, ~0.15 for 12B fp16 / ~0.137 for q4_0 at batch 256) so offload savings are provable, not asserted. "Net" here = active power above idle, integrated over the decode window, divided by tokens generated - we match that convention exactly on every device.

### 4.1 Per-device energy sources

**A6000 (host GPU):** `nvidia-smi --query-gpu=power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv,noheader,nounits -lms 200`. Integrate `power.draw` over the decode window; subtract measured idle (~25-32 W on these cards). This reproduces the baseline methodology already used to produce the 0.137-0.152 J/tok table (200 ms sampling, idle ~25 W). Attribute to tokens the host actually samples (it owns final norm+lm_head+sample, so it counts every emitted token anyway).

**Phones (coulomb / charge-counter method - this is the project's validated ground truth):**
1. Disable charging: `echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable` (needs `su`; the `/sys/class/power_supply/battery/` mmi path is denied on these units). With charging disabled and USB plugged, USB is capped ~2.5 W (500 mA @ 5 V) and covers idle, so at idle the battery draws **zero** and the `charge_counter` discharge slope measures **compute power above idle** directly.
2. Sample `/sys/class/power_supply/battery/charge_counter` (uAh) and `voltage_now` (uV) at ~1 Hz over the decode window. `P_batt = -d(charge_counter)/dt * voltage`. `P_whole_device = P_batt + P_usb_rail`.
3. **Do not trust `current_now`** - it is lagged/unreliable on these devices; use the `charge_counter` slope. (Both facts are recorded measured findings.)
4. `J/tok_phone_stage = P_decode_window / (tokens_in_window)`; decode window = last `T_TG` seconds of the run, matching the batched-bench convention.

### 4.2 Unifying to one number

For a decode step the pipeline is serial across stages but the *devices run concurrently* across steps (pipelining). We report **two** honest views, never conflating them:

- **Fleet J/tok (headline, comparable to baseline):** `sum over devices of (P_device_active_net integrated over the run) / total_tokens_emitted`. This is the apples-to-apples number vs the A6000's ~0.15 J/tok, because the A6000 baseline is *also* whole-device net energy / tokens. This is the number that proves or disproves offload savings.
- **Per-stage attribution (diagnostic):** each device's net energy / tokens, so we can see whether OP15's 2 layers and OP12's 1 layer actually pull their weight.

**Brutal honesty flag (must be shown in the report, not buried):** the baseline phone share is 3 of ~48 layers - deliberately tiny. That means each phone's *active* energy contribution is a thin sliver over idle, and it may sit **at or below the coulomb noise floor** (op15 method has ~10% run-to-run variance; whole-device idle dominates). At the 3-layer milestone the correct, honest claim is *"pipeline plumbing + energy accounting validated end-to-end,"* **not** *"energy saved."* Real savings require the later rebalancing milestones that move many layers onto phones (bounded by phone RAM and sustained thermal power). The energy harness is built now precisely so that when layers move, the savings/regressions are measured the same way from day one.

**A second honesty flag - the A6000 never leaves the loop.** Design A keeps embeddings + ~45 layers + lm_head + sampler + scheduler on the 300 W A6000, which stays hot the entire time. The prior review's identified killer applies: *keeping the server hot to schedule can erase a thin phone win.* Our fleet-J/tok number captures this automatically (server power is in the sum), so the accounting will tell the truth even when the architecture is unfavorable. That is the point of making it first-class.

### 4.3 Energy aggregator

A host daemon collects: `nvidia-smi` stream (local), phone `charge_counter`/`voltage` slopes (via heartbeat), and the token counter (from the sampler, authoritative). Every reporting window it writes a row:

```
window, tokens, batch_occ, P_a6000_net, P_op15_net, P_op12_net,
J_tok_fleet, J_tok_a6000_share, J_tok_op15_share, J_tok_op12_share,
J_tok_baseline_ref(0.15), delta_vs_baseline
```

---

### 5. Observability

**Metrics (Prometheus text exposition on the host + JSONL trace for offline):**

| Metric | Source | Why it matters here |
|---|---|---|
| `stage_service_ms{stage}` p50/p95/p99 | host timestamps around each hop | finds the moving bottleneck; drives admission |
| `pipeline_bubble_ms{stage}` | idle gap = step_wall - busy_time per stage | **the Design-A tax:** heterogeneous v81/v75 + DVFS + USB bounce show up here |
| `hop_latency_ms{s->s+1}` | host send/recv | quantifies the OP15->OP12-via-host bounce |
| `batch_occupancy` | scheduler | energy control variable (keep >=64) |
| `tokens_per_s` (prefill/decode split) | sampler | matches batched-bench speed_pp/speed_tg |
| `kv_pages{stage}` / divergence | heartbeat reconcile | catches distributed-KV eviction leaks |
| `temp_c{stage,zone}`, `clock_mhz{stage,engine}` | phone agent / nvidia-smi | thermal + DVFS attribution |
| `J_per_tok_*` | energy aggregator (S4) | the deliverable |

**Bubble accounting is the headline observability output for Design A.** A pipeline stage's bubble = wall time it sat idle waiting on an upstream/downstream stage. With a fast server front (45 layers) feeding two slow, differently-throttling phones, bubbles are expected and are *the* structural cost of the pipeline choice. We render a per-step Gantt (server / OP15-NPU / OP15-GPU / OP12-NPU / OP12-GPU / head) so bubbles, the USB bounce, and DVFS dips are visible. This is exactly the evidence needed to decide later milestones (rebalance layers, or concede toward a Design-B-style topology).

**Cross-check harness.** Reuse `llama-batched-bench` per phone as the *ground-truth microbenchmark* the live pipeline is validated against: it already emits JSON (`t_pp, speed_pp, t_tg, speed_tg, pl, n_kv`, seen in `tools/batched-bench/batched-bench.cpp`). If live per-stage decode t/s diverges from the standalone batched-bench number for the same `pl`, something in the pipeline (bubbles, comms, DVFS) is eating it - the delta is itself a metric.

---

### 6. Thermal / DVFS policy

Policy is **per phase**, and the two phases want opposite things:

- **Prefill = compute-bound -> race-to-idle.** Push NPU/DSP + DDR clocks up, finish the HMX GEMM fast, return to idle. Energy per useful FLOP is lowest at high clock when compute-bound; time-at-power is short. Set the phone to a performance governor for the prefill lane.
- **Decode = memory-bound GEMV -> do NOT down-clock.** Down-clocking a bandwidth-bound op stretches the weight-read time at barely-lower power, which **raises J/tok**. This is a measured, load-bearing rule. Keep DDR/GPU clocks up during decode; the win is not speed but avoiding the energy penalty of a stretched memory read.

**Enforcement is best-effort and must be honest about it.** The project measured that on op15 the **DDR clock cannot be pinned**: `/sys/devices/system/cpu/bus_dcvs/DDR/` exposes no writable `min_freq`/governor (permission denied), no settable devfreq node; DDR floats to demand (seen at 60% under light load). Consequences the control plane must encode:
1. We can set **CPU/GPU/NPU governors** (`set_governor`) but **cannot guarantee DDR frequency.** The policy nudges demand (keep the engine busy so the governor votes clocks up) rather than commanding clocks.
2. **DVFS confounds absolute energy comparisons.** Concurrent load boosts clocks; a light co-runner does not. So the energy report must always log `clock_mhz` alongside `J/tok`, and comparisons must be **within-run relative** (the DVFS-robust protocol the co-schedule study converged on), not cross-run absolute. The observability layer therefore tags every energy window with the clock trace so a reviewer can see whether a delta is real or a DVFS artifact.

**Thermal throttle handling:** each heartbeat carries thermal-zone temps. On crossing a warn threshold: stop admitting *new* prefill to that phone (shed to server-only), let in-flight drain, let it cool. On a hard throttle or `THROTTLING` state: DEGRADE that stage (S7). We never fight the vendor thermal governor; we route around it.

---

### 7. Failure modes and recovery

| Failure | Detection | Recovery |
|---|---|---|
| Phone NPU decode **hang** (the recorded npl=2 deadlock) | per-hop deadline watchdog | abandon socket, DEGRADE stage to server-only, agent reforks RPC process to clear HTP0 |
| Phone thermal throttle / DVFS collapse | heartbeat temps/clocks | stop admitting to stage, drain, cool, re-add when recovered |
| Phone drop-out (USB unplug / process death) | 3 missed heartbeats | DEGRADE: server absorbs that phone's layers |
| USB inter-phone route breaks (host forwarding down) | hop_latency timeout on OP15->OP12 | reroute both phones as leaves off host; if impossible, DEGRADE OP12 |
| Distributed KV divergence (missed eviction) | heartbeat KV reconcile | forced re-sync of eviction list; if unrecoverable, evict+rerun affected seqs |
| Shard corruption / wrong SoC lib (v75 vs v81) | startup hash + lib check | refuse to admit stage; alarm |
| Server sampler/scheduler crash | local supervisor | it is the SPOF (holds lm_head+sampler); restart, resume from per-stage KV where possible |

**DEGRADE = the universal fallback: "fall back to server-only for those layers."** Because the A6000 already holds ~45 of ~48 layers and the full lm_head, it can transiently host the 1-3 phone layers with negligible added VRAM/compute (measured headroom: 12B fp16 fits to ~batch 288 before OOM on one 48 GB card; +3 layers is minor). The orchestrator keeps a **server-resident copy of the phone-assigned layers loadable on demand** so DEGRADE is a routing change, not a reload. This directly neutralizes Design A's biggest availability risk: a flaky heterogeneous phone stage cannot take the pipeline down, it just stops saving (whatever) energy until it recovers. The energy accounting will show the DEGRADE window as a J/tok bump back toward the pure-server baseline - which is the correct, visible signal.

**SPOF note (honest):** the server is a single point of failure for lm_head+sampler+scheduler. Design A inherently centralizes these; we do not solve HA here, we just supervise/restart. Flag for the team as an accepted limitation of the chosen topology.

---

### 8. Build milestones (control plane)

- **M0 (hard gate, do FIRST):** on each phone, run `llama-batched-bench` with `-npl 1,2,4,8,16,32,64,128` on the fp16 gemma path and confirm multi-seq **continuous-batched decode actually runs** with separate KV + block-diagonal mask. The roofline M-sweep does NOT prove this; the npl=2 hang is a live bug. If phone continuous batching is unusable, the whole decode-offload premise (and its energy math) is void - escalate before building further.
- **M1:** control channel + phone-agent + heartbeat/telemetry; deadline watchdog + DEGRADE-to-server. Prove failover with a killed RPC process.
- **M2:** energy harness end-to-end (nvidia-smi + coulomb) producing fleet J/tok in baseline units on the 3-layer pipeline. Deliverable: a J/tok number with error bars and a clock trace, honestly labeled "plumbing validated, not yet saving."
- **M3:** observability (bubble Gantt, batch occupancy, hop latency) + admission/backpressure closed loop at 8 fps.
- **M4:** per-phase DVFS policy + thermal-throttle routing; validate race-to-idle prefill and no-downclock decode against measured J/tok.

---

## Risk Register

Most-severe first. Verdict tags fold in the adversarial feasibility checks.

| # | Risk (verdict) | Why it bites | Mitigation | De-risked by |
|---|---|---|---|---|
| R1 | **Multi-seq continuous-batched decode UNPROVEN on phone NPU; op15 `n_parallel=2` HANGS** (risky-unproven) | Hexagon `supports_op` returns true for N-seq decode (batched over token dim, ne[3]==1), so the graph dispatches and **deadlocks** at runtime (padded block-diagonal mask in FA/SOFTMAX, VTCM overrun, or fastRPC/dspqueue stall). The synthetic 7.12 TFLOPS roofline only exercised the weight GEMM, not masked multi-seq attention + paged-KV writes. This gates the entire decode-batching value of Design A. | **Adopt GPU-decode / NPU-prefill-only as the baseline phone-decode path**; treat NPU batched decode as upside. Root-cause the hang with `GGML_HEXAGON` verbose/opfilter. Gate go/no-go on the **weaker** device (v75/op12), not just op15. | **S1 spike (M0)** + M3 |
| R2 | **Fleet J/tok may never beat the pure-server baseline at the 3-layer split** (feasible-with-caveats) | A6000 floors ~0.15 J/tok (fp16, batch 256); it stays a 300 W always-hot node holding embeddings + ~45 layers + lm_head + sampler + scheduler. Phones add ~5 W host-drawn USB rail that is attributed to nobody. Batched phone decode (~0.05–0.08 J/tok) is **unverified**. | Be explicit that the baseline is *plumbing only*. Drive the layer-rebalance milestones toward SWA/window-capped layers; only claim energy wins after S1 proves batched phone decode **and** the coulomb measurement clears the noise floor with error bars. | M4 (energy), M5 (rebalance) |
| R3 | **Single physical weight copy (mmap + fastRPC NPU + OpenCL GPU) is not provided by stock code and may fail on-device** (feasible-with-caveats) | f16/f32 layouts are byte-identical (Hexagon plain memcpy; OpenCL default GEMV/l4_lm read row-major), BUT ggml-opencl allocates its own `cl_mem` and copies — **no dmabuf/ION import path exists** (verified zero hits). Import is extension-gated (`cl_qcom_dmabuf_host_ptr` vs `ion_host_ptr` on kernel-5.x), alignment/padding-gated (`CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM`), and **breaks for the fast xmem image2d GEMM** (non-row-major prepack). | Write the OpenCL dmabuf-import buffer-type + scheduler glue; pad the rpcmem alloc. Budget a **two-copy fallback** (~0.4–0.5 GB/phone layer set). If batched GPU decode must be energy-competitive, accept a derived xmem prepacked image (second copy) — call out the tension with the single-copy constraint. | **S2 spike (M0/M1)** |
| R4 | **No pipeline overlap from in-tree scheduler** (feasible, confirmed) | `pipeline_parallel` auto-disables unless every non-CPU device advertises `caps.async && caps.events`; RPC=async/events false, hexagon events=false, opencl both false. Triple-locked by the RPC transport. Stages run **serially** (correct but bubbled). | Build cross-stage overlap at the **orchestrator** level: k in-flight microbatches over thread-parallel blocking RPC, per-stage worker+queue, separate connections. | M2 |
| R5 | **KV-sharing group straddling a pipeline cut breaks the "KV never crosses" invariant** (feasible-with-caveats) | Gemma-4 has `shared_kv_layers>0` (E4B=18; confirm 12B): no-KV tail layers read a boundary layer's KV via `build_attn`. If producer and consumer land on opposite sides of a cut, KV must cross the wire or the group must be co-located. Also `is_swa` comes from a **per-layer gguf array** (not `il%n_pattern`), so renumbering a shard misassigns attention type + RoPE base + KV head dims. | Assign phones a **KV-self-contained** block (own the reuse-source boundary layers, or an interior all-has_kv range). Keep **global** `blk.{g}.*` names; carry per owned-layer `sliding_window_pattern` + `rope_freq_base_swa` + `n_embd_head_k_swa`. Drive ownership via device/layer assignment, **not** truncated ggufs or gated `create_tensor` (build_gemma4 null-derefs otherwise). | M1 |
| R6 | **Phone KV can blow the RAM budget for global (non-SWA) layers** (feasible-with-caveats) | KV, not weights, dominates. 2 **full** layers @128seq×4096 ≈ 4 GiB; 2 **SWA** layers @128seq×512-window ≈ 256 MiB — an 8–16× swing on *which* layers land on the phone. (Numbers from E4B geometry: head_count_kv=2, key_length 512/256-swa; **confirm from 12B gguf**.) Server KV is the true OOM wall (OOM past batch ~320 at 12B fp16). | Assign **SWA/window-capped** layers to phones; cap context (~2k) for any global layer; verify KV-quant support on the hexagon FA path before relying on it. Resolve server KV wall (context cap / KV-quant / move more layers) before batched-decode milestone. | M1, M3 |
| R7 | **No control plane: a wedged phone hangs the host forever; any RPC error GGML_ABORTs the whole orchestrator** (feasible-with-caveats) | `ggml-rpc` has no heartbeat/deadline/cancel. In this fork GRAPH_COMPUTE is fire-and-forget; the blocking round-trip is **GET_TENSOR** (host pulling stage output). `recv_data` has no `SO_RCVTIMEO`/`SO_KEEPALIVE`; a live-but-hung phone never FINs. Client errors run `RPC_STATUS_ASSERT==GGML_ABORT`, killing all 64–128 in-flight sequences. | Build an **out-of-band control plane**: `SO_RCVTIMEO`+`TCP_USER_TIMEOUT` bounding GET_TENSOR, per-hop deadlines, replace the aborts with drop-socket + evict-sequences + agent reforks the phone `rpc-server`. Test via mid-step SIGSTOP (hang) and SIGKILL (crash). | M4 |
| R8 | **USB phones cannot peer; OP15→OP12 must transit the host, with a default subnet collision** (feasible-with-caveats) | Both phones default to `192.168.42.0/24` with gateway `.129` (duplicate subnet+gateway); a tethered phone installs no route to a foreign subnet, so "just enable IPv4 forward" needs root `ip route` on each phone. Handily, **stock ggml-rpc already routes cross-device copies via host** (returns false on different sockets → host get+set), so the hairpin is free. | Prefer a **server-mediated app-level relay** (independent server↔OP15 and server↔OP12 RPC connections; matches the loop-back-to-server for lm_head). Alternative: put all 3 nodes on **one WiFi/LAN** (peers directly, 1 hop). Bandwidth is a non-issue either way. | M2 |
| R9 | **Interconnect is latency- (not bandwidth-) bound; RNDIS/NCM goodput is far below the USB3 ceiling** (feasible-with-caveats) | The "1.6 ms/hop" assumed ~587 MB/s line rate; realistic RNDIS is ~12–37 MB/s → ~5–35 ms/hop, and the logical 3-hop decode is **4 physical USB segments**. Stock RPC lockstep loop + single connection serializes and head-of-line-blocks a decode frame behind a prefill frame. Overlap hides comms only if p99 per-hop ≤ per-stage compute. | Replace lockstep with a **streaming, double-buffered stage server**; separate prefill/decode sockets. Measure real goodput+p99 (iperf3 + 983,040-byte ping-pong) before trusting any latency budget. Accept the interconnect only if p99/hop ≤ measured per-stage compute. | M2 |
| R10 | **`lm_head`+sampler land on op12 by default; wire dtype is F32 (2× the assumed payload)** (feasible-with-caveats) | `dev_output` pins to the *terminal* device; default local-first order (CUDA0, op15, op12) drops the ~2 GB lm_head + 256k-vocab softmax onto op12 (infeasible). Residual is **F32** (extract_layer_inputs divides nbytes by sizeof(float)), so stock payloads are 7.86/1.97 MB, not 3.93/0.94. | Order devices **[op15, op12, A6000]** so A6000 is terminal; phones own the first 3 layers; `token_embd` is already host-CPU-forced. Insert `ggml_cast`→F16 at each boundary to halve bytes (harmless post-RMSNorm). | M0/M1 |
| R11 | **12B arch scalars unconfirmed; code doesn't recognize a 48-layer Gemma-4 (loads as LLM_TYPE_UNKNOWN)** (open across all subsystems) | Every stage-time, KV, and RAM number depends on `block_count`, `head_count_kv`, `key_length`/`_swa`, `sliding_window(_pattern)`, `shared_kv_layers`, `embedding_length_per_layer_input`. One verdict reports 48/3840 from a real gguf; others say the 12B isn't staged here. `n_embd_per_layer=0` (no PLE) was reported for 12B but must be reconfirmed. | **First-week `gguf_dump`** of the actual staged 12B; patch the n_layer switch so it loads as a known type; write "confirm from gguf" until then. | M0 |

---

## Consolidated Open Questions

Collected from every subsystem; resolve as milestones close.

**Topology, Layer Partitioning & Dataflow**
- What are the real Gemma-4 12B arch scalars from the actual gguf (n_layer, n_embd, n_head, n_head_kv, head_dim, n_ff, n_vocab)? The 12B config is not in the gemma4 n_layer switch and loads as LLM_TYPE_UNKNOWN, so all scalars used here are working assumptions.
- Is n_kv_shared_layers > 0 for the 12B shard? If yes, which earlier layer do the tail layers reuse, and does that force the interior-block fallback split in §7?
- Does the 12B shard have per-layer token embeddings (n_embd_per_layer > 0) and/or MoE layers? If so, each phone stage must additionally receive its layers' per-layer embedding inputs, adding a second per-hop payload.
- What is the actual is_swa interleave pattern, and can we guarantee the phone-hosted layers are SWA so their KV stays bounded at large batch and long context?
- Does the phone backend (Adreno OpenCL and/or Hexagon NPU) actually run B separate KV streams with per-sequence block-diagonal + SWA masks, given the recorded op15 NPU n_parallel=2 hang? This gates the entire decode dataflow.

**Interconnect, Transport & Wire Protocol**
- Actual USB tethering mode (RNDIS vs NCM) and achievable goodput/RTT on OP15 and OP12 — the ~600 MB/s USB3 figure is a ceiling; RNDIS often caps at ~30–300 MB/s. Must measure with iperf3 and a small-message ping-pong before trusting the latency budget.
- Does the current llama.cpp pipeline path pass raw hidden-state tensors between independent per-stage model instances, or must we add the stage-granular activation-relay described here? The ggml-rpc backend is op-granular and ships weights/graphs, so a new relay path is likely required.
- Kernel IP-forward vs application-level relay for the OP15→OP12 middle hop — the latter costs the same two USB traversals but gives the continuous-batch scheduler in-flight visibility; decide based on whether the scheduler needs to mutate/re-batch activations mid-stream.
- Gemma-4 12B native wire dtype (fp16 vs bf16) and exact hidden_dim / layer count / vocab — confirm from gguf_dump (12B not staged on this host); bf16→fp16 conversion cost on the A6000 send path if native is bf16.
- Correct microbatch count k and credit-window depth to keep all three heterogeneous (v81/v75) stages full without over-buffering — needs measurement of real per-stage decode time, gated on the UNPROVEN phone continuous-batched decode (llama-batched-bench de-risk).

**Weight Provisioning — Pre-Downloaded Shards**
- Exact Gemma-4 12B gguf metadata: block_count, hidden size, SWA n_pattern/dense_first, and whether output (lm_head) is a separate tensor or tied to token_embd — all needed to finalize the partition and shard format.
- Will the phone-side llama.cpp accept a headless stage (no embeddings/lm_head/sampler), or must we copy vocab strings into phone shards until a no-vocab load path is patched?
- Does the team prefer one-gguf-per-stage (simplest) or one-gguf-per-layer merged via the existing multi-GGUF split loader (rebalance transfers only the moved layer)?
- Should the activation boundary reuse ggml-rpc's SET_TENSOR wire framing, or does the separate pipeline-transport subsystem define its own protocol that this subsystem should target?

**On-Phone Single-Copy Weight Sharing (mmap + fastRPC + OpenCL)**
- Which ION/dmabuf-host-ptr OpenCL extension does each phone's Adreno driver actually advertise (cl_qcom_dmabuf_host_ptr vs cl_qcom_ion_host_ptr) on OP15 (Adreno 840 / v81) and OP12 (Adreno 750 / v75)? Must be probed on-device.
- Exact per-tensor fp16 shapes and per-layer weight byte totals for Gemma-4 12B (confirm from gguf_dump) to size the ION allocation and verify sub-buffer alignment does not overflow the phone RAM budget.
- Does ggml-alloc (galloc) place f16 weight tensors on offsets satisfying CL_DEVICE_MEM_BASE_ADDR_ALIGN once the phone-shared buffer-type alignment is set, or is per-tensor padding needed that inflates the shard size?
- If batched decode later enables the xmem GEMM (batch >= 16), is the extra GPU-private transposed prepacked image (~0.4-0.5 GB/layer) affordable within phone RAM alongside the shared base, or does that force keeping decode on the non-xmem l4_lm path?
- Is fastRPC's implicit cache maintenance on registered rpcmem buffers sufficient, or is an explicit one-time rpcmem cache-flush ioctl required after weight load before the first GPU/DSP read?

**Continuous Batching & Prefill/Decode Scheduling**
- Exact Gemma-4 12B arch from gguf_dump: total layer count, per-layer fp16 tensor bytes, GQA group count, intermediate/MLP width, and lm_head/vocab size — all stage-time and memory numbers depend on these (currently 'confirm from gguf').
- Does multi-sequence (npl>1) decode with separate KV + block-diagonal mask run at all on ggml-hexagon (v81/v75) and on ggml-opencl, and does per-step time scale sub-linearly with npl? The op15 npl=2 hang must be root-caused or the design must commit decode to OpenCL only.
- Measured phone decode stage time at M=64/128 for 1-2 layers (compute-bound estimate 60-145 ms is unverified), and whether xmem GEMM (batch>=16, env-gated) is usable for the masked-attention decode path or only for plain GEMM.
- Effective host<->phone RPC throughput and RTT over USB tethering vs WiFi, and whether OP15->OP12 host-forwarding adds a serializing hop that defeats microbatch overlap.
- Whether the fused [decode ++ prefill-chunk] ubatch (Sarathi chunked prefill) is accepted by the hexagon MUL_MAT token-dim batching together with a correct per-row causal mask, or whether prefill and decode must be time-sliced on-phone after all.
- Batched phone decode energy (estimated 0.05-0.08 J/tok) is unverified and is the entire later-milestone justification for Design A; must be measured with the coulomb method once npl>1 works.

**Distributed KV Cache & Intra-Phone NPU/GPU Execution**
- Confirm from the real 12B gguf: n_layer, d_model, n_head/n_kv_head, head_dim, and especially the per-layer local(sliding-window)/global attention pattern + window size — this determines phone KV budget and whether OP15's 2 baseline layers are local or global.
- Does N-sequence batched decode actually run without hanging on the phone backends? The project recorded 'NPU n_parallel=2 HANGS' on op15; §5.1 Steps A/B must resolve whether the hang is NPU-only (acceptable, decode is GPU) or also affects the Adreno decode path (blocking).
- Is the stage-local KV cache buffer allocatable in the shared dma-buf/ION region so both Hexagon and Adreno can address it for the zero-copy prefill->decode handoff, or does the Hexagon KV live in an engine-private DDR/VTCM buffer that would force a copy?
- Server KV is the true OOM wall (~45 layers OOMs past batch 320 at 12B fp16 per measured findings): resolve the mitigation (context cap, KV quantization q8_0/q4, sliding-window reliance, or moving more layers to phones) before the batched-decode milestone.
- For GQA (16 heads over 8 kv-heads), confirm the phone FA/softmax kernels handle head broadcast via contiguous GQA layout rather than ne[2]/ne[3] broadcast (which quantized MUL_MAT forbids) — verify shapes in the test-backend-ops case.
- If any layer is later served as a quantized format, does the Hexagon repack layout diverge from the Adreno layout, breaking the single-physical-copy mmap sharing? (fp16 weights avoid this; q-format needs a decision.)

**Orchestration, Energy Accounting, Observability & Failure**
- M0 gate: does multi-sequence continuous-batched decode (separate KV + block-diagonal mask) actually run on the phone NPU, given the recorded npl=2 deadlock on op15? If not, phone decode offload and its energy model are void.
- At only 3 of ~48 layers, is each phone's active energy contribution above the coulomb noise floor (~10% variance, whole-device idle dominates), or is per-stage energy attribution unmeasurable until later rebalancing milestones?
- DDR clock cannot be pinned on op15 (no writable devfreq node); can we make bubble-time and J/tok reproducible enough for cross-run comparison, or must all energy claims stay within-run relative with a logged clock trace?
- The OP15->OP12 activation handoff must bounce through the host over USB (host-centric devices cannot peer); what is the measured added latency per decode step and does host IP-forwarding stay reliable under sustained 8 fps admission?
- Does the whole-device fleet J/tok including the always-hot 300 W A6000 (embeddings + ~45 layers + lm_head + sampler + scheduler) ever beat the pure-server baseline at the 3-layer split, or only after many layers move to phones?
