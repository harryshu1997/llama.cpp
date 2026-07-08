# 📓 Project Log — Phone-Offload LLM Serving

> Running progress record. **Current status** is at the top and kept up to date.
> The **log** below is newest-first; every entry is timestamped.
> Goal: offload LLM work from an A6000 server to phones (OP15, OP12) to save energy.

---

## 📍 Current status — `2026-07-08 EDT`

**Phase: M4 in progress — DUAL-ENGINE (one session, two backends) live on BOTH real phones.** ✅ New `dualengine` mode in `llama-layersplit`: ONE process loads the shard on TWO devices and runs **NPU decode (HTP0) ∥ GPU prefill (GPUOpenCL)** on two threads concurrently. On the real phones the wall time == the *longer* single-engine leg → **zero-interference overlap** (op12 **1.76×**, op15 **1.92×** vs serial). Static **B=16 batched decode** on the NPU is bit-correct vs single-seq (rel_L2 ~5e-4, 0/16 argmax mismatch) — **no cross-seq bleed** — and the recorded **op15 S1 hang did NOT reproduce** (a static lockstep batch avoids the continuous-batching `n_parallel` path that hung). This delivers the user's requirements **(1) one session/two backends** and **(2) static batch (16/32) decode ∥ one-by-one prefill**. Still 2× weight (one shard copy per engine); **(3) one weight copy** is next (Build 3: rpcmem dmabuf + `clImportMemoryARM`, gated on the S2 probe). *xmem changes untouched.* Tracker: [PORT.md](PORT.md).

*(prior)* **M2 done — gemma-4 12B fp16 live on the actual phones over USB.** `op15[0,2) → op12[2,3) → A6000[3,48)` answers *"…capital of France?"* → **"The capital of France is Paris."** Each phone stores ONLY its shard (op15 2.72 GB, op12 0.43 GB) via partial-load (`c615983dd`) + `shard_gguf.py`. Single-engine pipeline: **NPU 194 / CPU 219 / GPU 264 ms/tok**.

```
 EXPLORE ✅ ─── DESIGN ✅ ─── M0 🔵 ─── M1 ✅ ─── M2 🔵 ─── M3 ⬜ ─── M4 ⬜ ─── M5 ⬜
 (benchmarks   (research_dev/  (ground-truth   static    inter-    batched   energy   layer
  + reviews)    DESIGN.md…)     + S1/S2 spikes) pipeline  connect   decode    +recover rebalance)
                                              ▲ live 3-dev USB pipeline here
```

| Milestone | State | Exit gate |
|---|---|---|
| Explore (benchmarks, reviews) | ✅ done | — |
| Design (`research_dev/`) | ✅ done | DESIGN.md + MILESTONES.md + README.md |
| **M0 — ground truth + vetoes** | 🔵 next | `gguf_dump` 12B; **S1** batched-decode; **S2** shared-weights |
| **M1 — static pipeline correct** | ✅ done | 3-way head→mid→tail (cuts 2,3) **bit-exact** vs mono on a 12-tok prompt; k≤13 ceiling N-invariant |
| M2 — interconnect + overlap | ⬜ | throughput = max(stage, hop), not sum |
| M3 — batched continuous decode | ⬜ | 64–128 seqs stable, no hang, no CPU fallback |
| M4 — reliability + energy | ⬜ | defensible fleet J/tok; wedged phone recovers |
| M5 — rebalance to a real win | ⬜ | find split where J/tok < server, or prove it can't |

**⛔ Next:** (1) ✅ **Pipeline on NPU/GPU** — done (see log below). (2) **Throughput at real cut ratios** — the current split is only 2/3 layers on phones; at that size the accelerator is dispatch-bound and loses to CPU. Rebalance many layers onto the phones so compute amortizes the per-forward overhead — that's the real test of whether NPU/GPU pays off (M5 crossover). (3) Port MTMD embd-inject + VQ, then the hard **dma-buf** last. (4) Still open: **S1** batched-decode veto (`n_parallel=2` hung on op15) — batching would amortize the same NPU dispatch cost. (5) op12 GPU: split flash-attn kernel fails to compile (`sub_group_shuffle_xor` unsupported on Adreno v75-era) — ran via fallback; revisit if op12 GPU becomes load-bearing.

---

## 🧭 Decisions locked

- **Build Design A** (server→OP15→OP12→server pipeline) per direction. *(A review preferred hub-spoke B; we build A and mitigate its costs.)*
- **Phones own the FIRST layers, A6000 is the terminal stage** (llama.cpp pins `lm_head`+sampler to the last device). Baseline: OP15 = layers 0–1, OP12 = layer 2, A6000 = embed + 3–47 + head + sampler.
- **Baseline phone decode = GPU-only, NPU = prefill-only** (batched NPU decode unproven).
- **Weights: pre-downloaded mmap shards**, one copy shared by NPU (fastRPC/dmabuf) + GPU (OpenCL import). No runtime weight RPC.
- **Only the residual `[n_embd, n_tokens]` crosses the wire; KV stays on each stage.**

## 🏗️ Architecture at a glance

```
 8 fps in ─► [A6000] embed + layers 3..47 ─residual─► [OP15] L0-1 ─►(via host)─► [OP12] L2
                     ▲                                                                │
                     │  next-token ids                                     residual   │
                     └──────────── [A6000] norm + lm_head + SAMPLE ◄───────────────────┘
   PREFILL → phone NPU/HMX   |   DECODE → phone GPU/Adreno   |   KV: local to each stage
```

## 📊 Key numbers so far

| Thing | Value | Source |
|---|---|---|
| op12 NPU prefill peak | **7.12 TFLOPS** @ batch 512 | roofline sweep |
| op12 GPU decode (stock) | ~0.40 TFLOPS | roofline sweep |
| op12 GPU decode (xmem+cache) | **0.95 TFLOPS** @ 128, 2.35× stock | xmem re-run |
| A6000 12B fp16 decode floor | **~0.15 J/tok** (net) @ batch 256 | user's table |
| phone decode, single stream | ~11 tok/s, ~0.7–0.9 J/tok | estimate |
| phone decode, batched ≥16 | ~0.05–0.08 J/tok *(if it runs — R1)* | estimate, **unverified** |

---

## 🗒️ Log

### `2026-07-08 EDT` — ⚡ DUAL-ENGINE: one session, NPU decode ∥ GPU prefill, on BOTH real phones ✅
Built the `dualengine` mode (`examples/layersplit/layersplit.cpp`) — the user's design requirements (1)+(2), realized in ONE process:
- **Two `llama_context` over two `llama_model`** (one per device), two `std::thread` workers. Decode engine pinned `--dev-decode HTP0`, prefill engine `--dev-prefill GPUOpenCL`. Agent-confirmed safe: `llama_model` weights are read-only during decode (`build_graph` is `const`), each context owns its own KV/sched/backends, and distinct devices don't contend. The `events=false` pipeline-parallel gate is orthogonal (it only governs intra-context micro-batch overlap) — the two-thread approach sidesteps it.
- **Static B-way batched decode** (accumulate B, one `llama_decode` with distinct `seq_id`s) — clears the HMX B≥5 gate; prefill stays one-request-at-a-time.
- **Self-validating**: (A) batched decode vs B single decodes on the SAME engine → L2-relative diff (fp noise ~5e-4, argmax 0/16) proves no cross-seq bleed; (B) times each engine alone vs overlapped wall.

**Real-hardware numbers (12B fp16 shard, B=16, 8 rounds):**

| Phone | decode HTP0 alone | prefill GPUOpenCL alone | wall (overlapped) | speedup | correctness |
|---|---|---|---|---|---|
| op12 (v75, [2,3) mid/inject) | 914 ms | 1194 ms | **1197 ms** ≈ max(·) | **1.76×** | rel_L2 5.3e-4, 0/16 |
| op15 (v81, [0,2) head/token) | 1077 ms | 993 ms | **1078 ms** ≈ max(·) | **1.92×** | rel_L2 5.0e-4, 0/16 |

Wall == the *longer* leg on both → **zero-interference concurrent execution** (reproduces [[cross-engine-coschedule-npu-gpu]] inside one process). **op15 did NOT hang** at B=16 — the static lockstep batch avoids the continuous-batching path that hung at `n_parallel=2` ([[s1-npu-batch-decode-hang-localized]]); the user's "static batch first" call was right. Still 2× weight (one shard per engine); requirement (3) one-copy is next.

**Research settled two internals (2 Explore agents):** (i) two contexts genuinely share one read-only model's weights; the blocker to one-copy across HTP0+GPUOpenCL is `supports_buft` (session/context identity) + private repack layouts — but that repack is only for QUANTIZED types. (ii) **F16 weights are stored NATIVE-LINEAR on BOTH backends** (Hexagon skips repack for F16/F32; OpenCL f16 write is plain-linear) → the linear bytes ARE shareable. Hexagon already exports a dmabuf fd (`rpcmem_alloc2`→`rpcmem_to_fd`, `ggml_hexagon_shared_buffer{base,fd,size}`); OpenCL can import it via `clImportMemoryARM(CL_IMPORT_TYPE_DMA_BUF_ARM, fd)` (declared in the linked NDK headers, unused today). The xmem prepack reads that linear f16 `cl_mem`, so an import-alias feeds it directly → one shared linear copy + a small derived os8 tile. Gate = does the Adreno driver honor the ARM import (S2 probe).

### `2026-07-08 EDT` — 🚀 gemma-4 **12B** deployed to the ACTUAL PHONES over USB — correct coherent output ✅
The real target model, sharded across the fleet, generating correct text end-to-end:

```
 "What is the capital of France?"  --chat (channel template)
   [op15] L0-1 (shard 2.72 GB) ─USB─► [op12] L2 (shard 0.43 GB) ─USB─► [A6000] L3-47 + lm_head
   → "The capital of France is Paris."   ✓ (matches full model via llama-cli)
```

Each phone stores **ONLY its slice** (op15 2.72 GB, op12 **0.43 GB** — not the 24 GB model). Ran on all three phone engines, all correct:

| phones engine | 12B ms/tok (incl prefill+USB) |
|---|---|
| **NPU** (HTP0)   | **194** |
| CPU              | 219 |
| GPU (GPUOpenCL)  | 264 |

**What it took:** `--chat` (apply the model's channel Jinja template — raw prompts degenerate; missing BOS was a gotcha), the plain-arch injection fix (`3f7784540`), partial-load + shard tool (`c615983dd`), f16 conversion, and a phone-lib rebuild with the injection fix. The injection fix holds on op12's **real NPU** (middle stage, no segfault). Phones cleaned up after (0 procs, 0 forwards). *xmem files untouched.*

### `2026-07-08 EDT` — 12B split validated end-to-end on host; fixed a plain-arch injection bug 🐛✅ (`3f7784540`)
Ran the **12B f16** pipeline on host CUDA from real shards: `op15[0,2)` + `op12[2,3)` (shards) → server `[3,48)` (full f16, partial load). Caught + fixed a real bug **before** phone deploy (the point of host validation):
- **Bug:** a 12B *middle* stage segfaulted — the `ls>0` injection path always did `ggml_get_rows(model.tok_embd, inj_tokens)` (an E2B per-layer-rebuild leftover), but 12B has no per-layer embd and a middle stage doesn't load `tok_embd` → null deref.
- **Fix:** branch the injection on `model.per_layer_tok_embd`. Plain arch (12B) builds only the injected-residual input (`inj_h`) and uses it as `inpL` — no token, no `tok_embd`, no orphaned input. Guarded `llm_graph_input_embd_h::set_input` for the null token/embd tensors; loader keeps `tok_embd` when `n_embd_per_layer>0` (E2B).

**Correctness: the split reproduces the full f16 model's argmax bit-for-bit** (verified on several raw prompts — split and mono-full both give the same token id). Per-hop timing (12B): op15 1.8 ms, op12 1.1 ms, A6000 tail 35.6 ms.

**Gotcha found:** raw-prompt output is **degenerate** (`a a a…`) — but so is the *full* bf16 AND f16 model (identical), because **gemma-4-12B-it is instruction-tuned + "any-to-any"** with a complex **channel-based Jinja chat template** (`<|channel>`, `<|"|>`, function-calling). Raw completion prompts are out-of-distribution. Not a pipeline bug — coherent output needs the model's chat template applied (separate task). Also: 12B loads as `LLM_TYPE_UNKNOWN` (48 not in the gemma4 n_layer switch) — cosmetic, inference unaffected (E2B path identical).

**Also:** phone lib set rebuilt with the partial-load loader; f16 12B shards staged (op15 2.72 GB, op12 0.43 GB). Ready for phone deploy.

### `2026-07-08 EDT` — Phone stores ONLY its layer slice (partial load + gguf shard tool) ✅ (`c615983dd`)
Requirement: 12B fp16 (~24 GB) can't fit a phone → each stage must hold only its layers. Built two pieces in our repo:
1. **`gemma4.cpp` partial load** — `load_arch_tensors` now reads the same `LLAMA_LAYER_START/END` as the graph and creates ONLY layers `[ls,le)` (+ `tok_embd` for head/terminal, `output`+`output_norm` for terminal). Out-of-range tensors are never created → never allocated, never loaded. `llama-model.cpp` passes `done_getting_tensors(partial=true)` when the env is set so a full-gguf partial load doesn't trip the tensor-count check.
2. **`research_dev/shard_gguf.py`** — extracts a layer slice into a per-stage gguf, keeping **original block indices + all metadata** (block_count, SWA pattern, rope, tokenizer) so per-layer SWA/rope indexing is bit-identical.

**Validated:** E2B shards load (312/601 tensors for tail; `is_swa` correct per *absolute* index). **12B shard sizes:** op12 `[2,3)` = **0.43 GB** (1.9% of 22 GB), op15 `[0,2)` = 2.72 GB (2 layers ~0.9 GB + `tok_embd` ~1.9 GB). Also: **f16 12B conversion done** (bf16→f16, mandatory per the kernel audit).

**Note:** op15's 1.9 GB is the `tok_embd` table, kept because the head currently EMBEDS (`ls==0`). The intended design has the **server embed** and send the residual to op15 → then op15 needs no `tok_embd` (~0.9 GB). That's a driver/topology change (next). E2B shards stay large because the MatFormer `per_layer_token_embd` table (~1.3 GB) is global — a 12B non-issue.

### `2026-07-07 EDT` — Model = gemma-4 12B fp16; kernel-shape audit + S1 hang did NOT reproduce 🔎
On-device (op15, before the user reclaimed it): the recorded **S1 hang did NOT reproduce**. Lockstep batched decode (our `tailbench`, full model) ran B=1→**64** (27→51 tok/s); continuous-batching `batched-bench` (Q4_0) ran npl=1→**8** (TG 21→31 t/s) — all clean. The old "npl=2 hangs on op15" was the **fp16** sweep; fp16 npl=2 was the one run in progress when op15 was reclaimed (no verdict). So the hang is at worst fp16-specific, not multi-seq-attention-general. **#4 (static lockstep batch) is proven feasible on the NPU today.** ([[s1-npu-batch-decode-hang-localized]])

**Model locked: gemma-4 12B fp16.** Dumped the local gguf: n_layer=48, n_embd=3840, n_ff=15360, n_head=16/kv=8, head_dim=256, vocab=262144. **12B is the PLAIN arch** — `per_layer_token_embd=0` (no E2B token-relay hack) + `shared_kv_layers=0` (cut ANY layer). Simpler to split than E2B.

**Kernel-shape audit (code-grounded) — user's "be careful about shapes" concern resolved:**
- **bf16→f16 MANDATORY.** NPU `supports_mul_mat` has no BF16 case → whole matmul → CPU (`ggml-hexagon.cpp:2672`). The local file is bf16; must convert to f16 (~24 GB) before deploy.
- **All 12B GEMMs are kernel-clean** (every K,N ÷32; xmem K%16 + out≥64). Prefill→**xmem** (M≥16), decode→**HMX** (B≥5). Only lm_head falls off xmem (harmless, terminal).
- **HMX gate: decode B≥5** (`m≤4→HVX`). Static batch 32/64 is well clear. ✓
- **xmem triple-gated**: compile `-DGGML_OPENCL_USE_ADRENO_KERNELS` + env `GGML_OPENCL_ADRENO_XMEM_GEMM` + Adreno. Default OFF → l4_lm (~3–4× slower).
- **CORRECTION to earlier claim:** peak-kernels vs one-shared-weight is NOT strictly either/or. HMX reads the **native-linear f16 in place**; xmem **prepacks from that same linear copy**. With S2 dmabuf-import + **uncached** xmem you keep **both peak kernels at 1× persistent RAM** (small per-call repack tax); **cached** os8 = 2× RAM, zero tax (fine at a 2–3-layer split, ~2–3 GB). Only degrade GPU→l4_lm if the tax bites; HMX never lost. ([[gemma-4-12b-arch-kernel-shapes]], [[npu-decode-gpu-prefill-conditional]])

### `2026-07-07 EDT` — Design check: NPU-batch-decode ∥ GPU-prefill — verdict + S1 hang localized 🔎
Question raised: *"can we form a decode batch, run it on the phone NPU, and prefill on the GPU concurrently?"* Ran a 6-agent code+roofline review (4 readers of the actual backends + synthesis + adversary). **Verdict: partly on the same page — right goal, wrong as a static rule, currently unbuildable.**

**On the same page (correct):** batched decode is graph-feasible (gemma-4 graph is `n_seqs`-general; `tailbench` already assembles a real B-way batch, `llama_batch_init(n_streams,…)`, `seq_id[j]=j`, one `llama_decode`). Batching is exactly what pushes decode to high-M where the NPU/HMX wins. The utilization idea (NPU holds sustained decode, GPU absorbs bursty prefill) is legitimate.

**Corrections (why it's not a fixed rule):**
1. **It inverts the locked baseline** (decode→GPU, prefill→NPU) and is right *only* at sustained **B>4** (HMX gate) inside an NPU-favorable band — energy is **non-monotone**: NPU B≤16, **GPU B32–64**, NPU B128. Below B=4 it loses on *both* phases (NPU decode → HVX/no-win; prefill stranded on the 17× weaker GPU, 403 GFLOPS vs 7.12 TFLOPS). ⇒ must be an **adaptive router on batch occupancy**, baseline as low-load default.
2. **"Simultaneously" ⇒ two processes, not one.** One `ggml_backend_sched` serializes splits; pipeline-parallel overlap is force-disabled because OpenCL & Hexagon both report `events=false` ([opencl:8906](../ggml/src/ggml-opencl/ggml-opencl.cpp#L8906), [hexagon:3640](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L3640), gate [llama-context:385](../src/llama-context.cpp#L385)). Two pinned contexts ⇒ **2× weight RAM** (no cross-engine dmabuf import = the S2 veto). 12B likely infeasible at 2× without S2.
3. **Zero-interference may not transfer** — it was NPU-*compute* ∥ GPU-*memory*; this pairing is likely *memory ∥ memory* (M=1 decode is bandwidth-bound), which our data says **contends** on the bus. Re-measure.

**S1 hang localized (the blocker).** `n_parallel=2` hung on op15 — *not* a llama deadlock (batching is lock-free, `n_seqs`-general). It's an **HTP backend bug**: host `flush_pending()` infinite-retries on the 1 s DSP timeout with no watchdog (`AEE_EEXPIRED → continue`, [hexagon:1516-1550](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1516)); DSP worker-pool busy-spins `while(atomic_load(&n_pending))` (`worker-pool.c:216`); `FLASH_ATTN_EXT` `supports_op` accepts a 2-seq attention op with no mask/KV validation ([hexagon:1883-1916](../ggml/src/ggml-hexagon/ggml-hexagon.cpp#L1883)). **Fixable, and the floor** — nothing else builds until real B>4 decode completes on the NPU.

**Also found:** the serving path isn't ready — `stagenet`/`pipedriver` are strictly single-seq (`llama_batch_init(1,…)`, `seq_id 0`); a B-way NPU decode needs `tailbench`'s batch-assembly ported into the persistent socket loop (protocol carrying B tokens+hidden, `n_seq_max=B`, B residual replies).

### `2026-07-07 EDT` — Pipeline on phone NPU + GPU, A6000 CUDA terminal — per-hop timing ✅
Ran the persistent pipeline with the phone stages on each engine, host tail on the **A6000/CUDA** (was silently `-ngl 0`=CPU; added `HNGL` to `pipeline_persistent.sh`, default 99). Added per-hop timing to `pipedriver` (times only the 32 generation-phase steps). All three engines emit the same correct text.

Split: **op15=[0,2) → op12=[2,3) → A6000=[3,35)**, gemma-4-E2B Q4_0, 32-token decode.

| phones engine | stageA op15 (2L) | stageB op12 (1L) | tail A6000 (32L) | Σ decode/tok | full ms/tok¹ |
|---|---|---|---|---|---|
| **NPU** (HTP0)   | **34.2** ms | 7.4 ms  | 6.4 ms | 48.0 ms | 164 |
| **GPU** (OpenCL) | 16.5 ms | 13.1 ms | 6.4 ms | 36.0 ms | 111 |
| **CPU**          | 18.1 ms | 11.3 ms | 6.3 ms | 35.7 ms |  67 |

¹ full pipeline wall-clock/tok incl. prompt prefill + first-forward warmup (NPU graph / OpenCL kernel compile) — NPU pays the most here.

```
 per-token decode path (steady state), NPU config:
   op15 NPU 2L ──34ms──►  op12 NPU 1L ──7ms──►  A6000 CUDA 32L ──6.4ms──► sample
   └────────────── phone stages dominate ──────────────┘   └─ tail is cheap ─┘
   (USB RTT is tiny: residual = 1536×f32 = 6 KB/hop, <2 ms — the cost is phone dispatch+compute)
```

**Findings.**
- **The A6000 tail is not the bottleneck** — 32 of 35 layers decode in **6.4 ms** on CUDA. The 2–3 phone layers cost 5–6× more.
- **At this split the accelerators LOSE to CPU.** Single-token decode of 2 layers has too little compute to amortize the NPU's fastRPC dispatch (~34 ms on op15) or the GPU's kernel-enqueue overhead. Phone CPU is fastest end-to-end (67 ms/tok).
- **op15 (head) ≫ op12** disproportionately — the head also builds the per-layer token-embedding projection for all 35 layers, not just its 2.
- **op12 GPU** can't compile the split flash-attn kernel (`sub_group_shuffle_xor` unsupported on Adreno v75-era) → ran via fallback. Portability caveat.
- **Implication (R2/M5):** a 3-layer split is plumbing, not a win. The NPU/GPU only pays off after rebalancing many layers onto the phones (compute amortizes dispatch) and/or batching decode (S1). Backends confirmed live: op15 Hexagon v81 HTP0 (hvx 8, hmx 1, vtcm 8 MB) + Adreno 840 OpenCL.

### `2026-07-06 17:30 EDT` — Persistent pipeline: KV-resident incremental decode over USB ✅ (`a50edb599`)
Replaced the stateless act-file relay with **persistent stages** — each phone runs a long-lived server holding its model + KV; only the residual+token cross USB per step.

```
 host pipedriver (tail [3,35), drives)
   │  adb forward tcp:15555 (USB)          │  adb forward tcp:15556 (USB)
   ▼                                       ▼
 [op15] stagenet [0,2)  ──residual+tok──► [op12] stagenet [2,3)  ──► back to host tail ──► sample ─┐
   ▲ KV-resident                            ▲ KV-resident                                          │
   └────────────────── next token (feeds op15 for pos+1) ◄──────────────────────────────────────┘
```

- **op15[0,2) → op12[2,3) → server[3,35)** generated `"The quick brown fox jumps over the lazy dog and then runs away."` at **~320 ms/tok** (CPU) — vs ~6 s/tok for the re-prefill version (**~18×**).
- New driver modes `stagenet` + `pipedriver` (+ `connect_to`, `--port2`); frame `{pos,tok,nh,hidden}` carries the token id so each stage injects a **DUAL batch** (the per-layer-embd fix). Runner: [research_dev/pipeline_persistent.sh](pipeline_persistent.sh).
- Gotchas: killing the host-side `adb shell` doesn't kill the on-device process (use `adb shell pkill -9 -f layersplit`); **device port 5555 is adbd's wireless-adb listener** → use 15555/15556.
- Next: swap phones to NPU/GPU in the pipeline; measure per-stage USB RTT vs compute; then throughput at real cut ratios.

### `2026-07-06 16:45 EDT` — 3-device USB pipeline BUILT + generating text ✅
All phones are USB-connected to the server (they can't peer → server-mediated relay). Built an orchestrator ([research_dev/pipeline_3dev.sh](pipeline_3dev.sh) + `_gen.sh`) that chains the stages over **adb (USB)** with no new C++ transport — reuses head/mid/tail + a tiny `--tokens-file` for exact token feedback.

```
 input ─► [op15] head layers [0,2) ─residual(USB/adb)─► server ─► [op12] mid [2,3)
                                                                        │ residual (USB)
                                            next-token ◄── [server] tail [3,35) ◄┘
```

- **Single forward:** pipeline next-token == whole-model `mono` → **top-1 MATCH ✅** (heterogeneous op15-arm64 → op12-arm64 → server-x86; residuals cross as fp32, ~73 KB/hop, negligible).
- **Generation:** greedy decode over the pipeline produced coherent text —
  `"The quick brown fox jumps over the lazy dog and then runs away."`
- Stateless stages (no persistent KV) → each step re-prefills the full seq (O(N²) + a model reload per stage; ~2 s/stage). **This is the correctness/plumbing build.** Throughput build = persistent stages holding KV over `adb forward` TCP (the socket modes still need the dual-batch + token-relay fix + a `midnet`).
- ⚠️ paused mid-experiment: op15 in use by another agent.

### `2026-07-06 16:15 EDT` — On-device validation: op15 NPU + GPU + CPU ✅
Built the phone lib set from **this checkout** via the npu-harness snapdragon-docker (`scripts/build_npu_op12.sh --force` — builds all htp-vNN + opencl + libllama-with-fix, ABI-matched). Deployed to op15 (Hexagon v81 / Adreno 840) and ran the oracle on a 12-token prompt:

```
 op15 engine        mono         2-way k=13    3-way (2,3)     verdict
 NPU (Hexagon v81)  8784 @10.076 8784 @10.076  8784 @10.076    ✅ BIT-EXACT
 CPU                8784 @9.743  8784 @9.743   8784 @9.743     ✅ BIT-EXACT
 GPU (Adreno 840)   8784 @9.267  8784 @9.267   8784 @9.162     top-1 exact (3way logit = FP-noise)
```

- **All three engines predict the same next token** (8784 ' runs'); NPU + CPU reproduce the whole model to the bit through the 3-stage split → the injection/extraction survives the real **CPU↔NPU offload boundary**.
- GPU: 2-way is bit-exact; the 3-way logit drifts ~1% (extra CPU↔GPU residual round-trip at the mid stage = OpenCL reduction-order non-determinism, not a fix bug).
- Also ran on **op12** (Hexagon v75) earlier: NPU bit-exact too. Both phones covered; **op15 is the focus device.**
- Build recipe note: docker build compiled all of `libggml-htp-{v73,v75,v79,v81}.so` — deploy the arch matching the SoC (op15→v81, op12→v75); the arm64 `llama-layersplit` + `libllama`(fix) + `libggml-opencl` are shared.

### `2026-07-06 15:40 EDT` — Multi-token + 3-way pipeline bit-exact ✅ (commit `99224d3a7`)
Generalized the driver to N-token prefill and added a **middle stage** — the full op15→op12→server chain now validates end-to-end. A parallel 4-agent read-workflow first confirmed the **graph needs zero changes** (already N-token-general; already supports `ls>0 && le<n_layer`); all work was driver-only.

```
 12-tok prompt "The quick brown fox jumps over the lazy dog and then"
 MONO (whole model, last-token) ............ id=8784  logit=11.035674   ← reference

 2-way head[0,k)→tail[k,35):  k=3 ✅  k=10 ✅  k=13 ✅ | k=14 ✗   (k≤13 ceiling holds for N tokens)
 3-way head[0,2)→ mid[2,3) →tail[3,35): ...  id=8784  logit=11.035674   ✅ BIT-EXACT
        (= op15 → op12 → server baseline, through a real middle stage)
 single-token back-compat (--tok) ................................... ✅
```

- **act-file v2**: `{n_embd, N, tokens[N], residual[N*n_embd]}` — relays N cut residuals + the N token ids (needed at every hop to rebuild per-layer embeddings). New `mid` mode injects + runs head-less + relays onward.
- **Caveat for real transport:** the *socket* relay modes (tailnet/headnet/*stream) still inject `token==NULL` = the old residual-only path → would resurrect the per-layer-embd bug for a layer-split tail. They need the dual-batch + token-relay before use across devices. (Next.)

### `2026-07-06 15:05 EDT` — M1 FIXED → the split is bit-exact ✅ (commit `48b020120`)
Made the gemma-4 cross-device cut numerically correct. **The tail now decodes a DUAL batch** — the relayed input token rebuilds the per-layer embeddings exactly, the injected residual becomes `inpL`:

```
 HEAD [0,k)  --(act-file: n_embd, TOKEN_ID, residual[n_embd])-->  TAIL [k,35)
   token batch                                                     DUAL batch {token+embd}
   dumps residual @cut                                             token → per-layer embd (exact)
                                                                   embd  → inpL for layers [k,35)
```

- **3 files:** `gemma4.cpp` (tail builds token-embd manually — `build_inp_embd` prunes its embd tensor on a dual batch — then swaps `inpL` to the injected residual after the per-layer projection); `llama-graph.cpp` (null-guard `llm_graph_input_embd::set_input` so the token-only per-layer input tolerates `ubatch.embd`; no-op elsewhere); `layersplit.cpp` (relay token id + dual batch).
- **Reused existing infra:** the injected residual rides the MTP `embd_h.h` hidden-state input — **no new core class/API**.
- **Result — full cut sweep, gemma-4-E2B, 3 tokens:**

```
 cut k :  1  2  3  4  5  6  7  8  9 10 11 12 13 | 14 ...........34
 match : ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ ✅ |  ✗  (wrong)
                     bit-exact id+logit          ^ shared-KV boundary
```

- **Second finding — shared-KV cut ceiling.** `shared_kv_layers=20` ⇒ layers 0–14 own KV, 15–34 **reuse** it (last SWA owner = L13). So a cut is valid only for **k≤13**; k≥14 leaves the tail's KV cache missing reused entries. **Baseline cuts k=2 & k=3 are exact** → phones-hold-first-layers is safe; just don't cut inside the shared-KV tail. Memory: `gemma3n-perlayer-breaks-layersplit`.

### `2026-07-06 14:40 EDT` — M1 validation FAILED → found an architectural blocker
Ran the oracle on `gemma-4-E2B-it-Q4_0` (35 layers, CPU): `mono` vs `head→tail`.

```
 mono  (tok=1000)                  ARGMAX id=1000  logit=-12.27   ← reference
 head[0,k) → tail[k,35) :
   k=10  id=107   k=17  id=140   k=25  id=2   k=30  id=1000(logit -5.35!)   k=34  id=105
                                                    └ top-1 matches by luck, logit still wrong
```

**Every cut point is numerically wrong.** Root cause (proven in code, not just empirically):

```
 gemma-4 = Gemma-3n style. Each layer adds a PER-LAYER TOKEN EMBEDDING:
   inp_per_layer = per_layer_token_embd[ token_id ]          ← needs the INPUT TOKEN
   inp_per_layer = project(inp_per_layer, scaled_tok_embd)   ← needs the TOKEN EMBEDDING
   ...added at layer il for every il.
 A residual-only cut carries neither. The token-less TAIL stage silently hits the
 else-branch → uses the PADDING token (id 0) embedding for ALL its layers, and
 mis-projects the deep residual (±53) as if it were the token embedding. → garbage.
```

- **Not a port defect:** the fork's `gemma4.cpp` has byte-identical logic (`project_per_layer_inputs(inpL,…)` + padding fallback) → the fork's LayerSplit was **never numerically validated on a per-layer-embd model**.
- **This is exactly what M1 is for.** Caught before we built the batched/energy layers on top of a wrong pipeline.
- **Fix options** (in [PORT.md](PORT.md)): ① relay token IDs + inject residual (cheapest, recommended) · ② relay the projected per-layer tensor · ③ plain-arch fallback model. Memory: `gemma3n-perlayer-breaks-layersplit`.

### `2026-07-06 14:19 EDT` — Ported the LayerSplit driver → functional pipeline
Copied `examples/layersplit/` (1009-line driver) from the fork. **Built + ran with zero code changes** — this repo already ships `llama-ext.h` + the `embeddings_nextn` C API it needs. Committed `1f1b3f60a`.
- Modes: `mono` (reference) · `head`/`tail` (cut-activation correctness oracle) · `tailnet`/`headnet` (**raw-TCP** cross-device stage relay) · `tailbench` (batched tail decode).
- **Milestone:** with the LayerSplit hooks (`0a577ab99`) + this driver, the **single-model cross-device pipeline is buildable and functional** (M1 skeleton). Next: validate logits (mono vs head→tail) on a real model.

### `2026-07-06 14:03 EDT` — Started porting reuse code → branch `plan-a-port`
Bringing the Unifer fork's Plan-A code into this repo as clean ports (fork is at a different base — b9531 vs our b9850, so replay feature diffs, not copy). Tracker: [PORT.md](PORT.md).
- ✅ **RPC `tensor_extras`** (`ggml-rpc.cpp`, 5 hunks) — carries Adreno/HTP repack metadata across the RPC round-trip.
- ✅ **LayerSplit hooks** (`gemma4.cpp`, 5 edits) — `LLAMA_LAYER_START/END` bounds + head-less cut via `res->t_h_nextn`. **De-risked:** the `t_h_nextn`/`embeddings_nextn` plumbing already exists upstream here, so no new API needed; dropped the bundled fused-QKV optimization.
- **Difficulty map (PORT.md):** dma-buf zero-copy (our S2) is the hard one — its file drifted ~3500 lines and holds our xmem changes → do it **last**. RPC/LayerSplit were easy/moderate.
- ✅ **Build-verified + committed** on `plan-a-port`: native CPU+RPC build compiled both TUs clean (0 errors/warnings); commits `3f42a8504` (RPC) + `0a577ab99` (LayerSplit). xmem changes untouched.
- **Next:** the `layersplit` driver (`examples/layersplit/`), then transport, then dma-buf.

### `2026-07-06 13:45 EDT` — Reuse audit: most of Plan A already exists

Read Unifer's `PLAN_A_REUSE.md`; verified every component in the fork `~/Documents/llama.cpp @ route2-b9531` (= Unifer's `third_party/llama.cpp`).

- **Reuse — committed:** GPU↔NPU **dma-buf zero-copy** *(≈ our S2, already done!)*, VQ admission gate, HTP vision encoder (op15 only), MTMD embd inject; continuous batching is upstream.
- **Reuse — ⚠️ uncommitted (commit first):** layer-split (`gemma4.cpp` LAYER_START/END + `examples/layersplit/`), RPC `tensor_extras` patch, `examples/cdsd/` transport, split-K GEMV kernel.
- **Corrections:** op12 NPU fragile (vision-unusable; LLM matmul works — we measured it); **op15 is shared** (no broadcast pkill); USB-2 ~40 MB/s, RTT <2 ms; the doc independently confirms Plan A = **throughput** play, not an energy story at large batch.
- **Heads-up:** design docs live in `llama.cpp-release/research_dev`, but code + `project_*` memories live in **Unifer** → consolidate.

### `2026-07-06 13:33 EDT` — Design A system plan written → `research_dev/`
Ran an 8-subsystem design + adversarial feasibility workflow (agents read the real llama.cpp source). Output: **DESIGN.md**, **MILESTONES.md**, **README.md**.
- Feasibility: 2 feasible, 13 feasible-with-caveats, **1 risky-unproven** (batched phone decode).
- Code-grounded surprises: phones must hold the *first* layers (device-order pins `lm_head` to terminal); Gemma-4 shared-KV tail layers constrain the cut; llama.cpp pipeline overlap auto-disables; no OpenCL dmabuf-import path exists (needs new code).

### `2026-07-06` — Is batched decode even possible on the phone?
Checked the Hexagon backend: ops (MUL_MAT, FLASH_ATTN_EXT, softmax) *support* batched decode, **but** real multi-sequence continuous batching is unproven and `n_parallel=2` was recorded to **hang** on op15. The roofline "batch M" is one synthetic matmul — **not** N sequences with separate KV + masked attention. ⇒ This is the #1 risk; de-risk with `llama-batched-bench` (spike S1).

### `2026-07-06` — A6000 J/tok corrected; overflow scenario
Corrected my earlier over-optimistic server number: real A6000 12B decode floor is **~0.15 J/tok**, not ~0.03. In the *server-saturated / spillover* framing, phones become attractive because the alternative is lighting up another under-utilized 300 W box. Energy win is real but conditional on batched phone decode working.

### `2026-07-06` — Design review: A vs B
5-lens adversarial review. **B (hub-spoke replication + router) preferred**; A (pipeline) has real costs (host-relayed phone↔phone hop, bubbles, always-hot server). Prefill offload = robust energy win (~1.3–1.8×); decode offload = capacity/KV-relief, energy only if batched. User chose to build A anyway.

### `2026-07-06` — GPU efficient-kernel check + xmem re-run
Confirmed op12 GPU already uses the efficient `l4_lm` tiled GEMM (not a slow path); the ~400 GFLOPS ceiling is real, small-batch weakness is 64-wide tile underfill. Enabling **xmem image-GEMM + prepack cache** lifted GPU peak to **0.95 TFLOPS** (2.35× stock), engaging at batch ≥16.

### `2026-07-06` — Batch-size roofline sweep (op12)
Swept GEMM batch on NPU vs GPU. **NPU knee M=512 → 7.12 TFLOPS; GPU knee ~M=64–128 → ~0.4 TFLOPS.** NPU wins every batch except M=1. Produced a line-graph artifact. This is the empirical basis for "prefill→NPU, decode→GPU".
