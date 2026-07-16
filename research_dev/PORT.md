# Plan A — Porting the Unifer reuse code into this repo

*Bringing the reusable Plan-A components from the `route2-b9531` fork into a clean tree here.*
Companion to [DESIGN.md](DESIGN.md) · reuse source: `~/Documents/Unifer/research_dev/services/lazyvlm/PLAN_A_REUSE.md`.

> **CURRENT-DIRECTION NOTE:** this is a provenance and reuse inventory, not the
> live roadmap. The primary target is
> [mixed-workload reverse offload](MIXED_WORKLOAD_DESIGN.md), executed through
> [NEXT_PLAN.md](NEXT_PLAN.md) and S8-V0. A component is not authorized for port
> merely because it appears below.

## Relevance to the current target

| Historical component | Current role | Decision |
|---|---|---|
| LayerSplit hooks/driver | Route `A0`, island executor and correctness control | reuse |
| Raw TCP/CDSD | Data-plane reference after framed protocol hardening | adapt |
| Per-tensor HTP/OpenCL sharing | Resident phone weights | reuse after version/generation hardening |
| VQ byte table | Process-local occupancy and CONWIP instrumentation | audit/minimally extract; not distributed status |
| `mtmd-sched` HEFT | Offline baseline and policy reference | reference only in S8-V0 |
| 16-byte remote roster | Fast telemetry design reference | adapt separately from model readiness |
| HTP vision encoder | Candidate island only if MW1 profile selects it | unauthorized until measured |

## Setup

- **Target (this repo):** `llama.cpp-release` @ `release-b9850`, work branch **`plan-a-port`**.
- **Source fork:** `~/Documents/llama.cpp` @ `route2-b9531` (HEAD `6b6942985`), = Unifer's `third_party/llama.cpp`.
- **Version gap:** target contains the fork's merge-base `6effcecd0` in its history, so ports = replay each fork *feature diff* (relative to that base) onto b9850. **Not** clean copies — the trees drifted.
- **Provenance:** every ported hunk is tagged `// [plan-a port]` so it greps out cleanly.
- **Do NOT** clobber the pre-existing uncommitted work here (xmem prepack cache in `ggml-opencl.cpp` etc.).

## Component status

| # | Component | Files | Drift | Difficulty | Approach | Status |
|---|---|---|--:|---|---|---|
| 5a | **RPC `tensor_extras`** (carry OpenCL/HTP repack metadata across RPC) | `ggml-rpc.cpp` | ~28 ln | easy | port | ✅ **done · built · committed** `3f42a8504` |
| 4 | **LayerSplit hooks** (`LLAMA_LAYER_START/END`, head-less cut via `t_h_nextn`) | `gemma4.cpp` | ~100 ln | moderate | port (LayerSplit only) | ✅ **done · built · committed** `0a577ab99` |
| 5b | LayerSplit driver (mono/head/tail + raw-TCP tailnet/headnet + tailbench) | `examples/layersplit/` | new | easy³ | copy | ✅ **done · built · committed** `1f1b3f60a` |
| 5c | CDSD adb-TCP transport | `examples/cdsd/` | new | moderate | copy+adapt | ⬜ |
| 5d | `MTMD_EMBD_DUMP/REPLACE` (cross-device embd inject) | `mtmd.cpp` | ~900 ln | moderate¹ | port | ⬜ |
| 2 | **GPU↔NPU dma-buf zero-copy** (our spike S2) | `ggml-opencl.cpp` | ~3500 ln | **hard**² | port carefully | ⬜ |
| 1 | VQ byte-table admission gate | `ggml-backend.cpp` | ~730 ln | moderate-hard | port | ⬜ |
| 3 | HTP vision encoder (op15-only) | `ggml-hexagon/htp/*` | 9 files | hard | port if vision needed | ⬜ defer |
| 8 | HEFT `mtmd-sched` (optional) | `mtmd-sched.cpp` | — | easy | copy | ⬜ defer |
| — | split-K GEMV kernel, FGDN fallback | `*.cl`, `llama-context.cpp` | — | — | port | ⬜ supporting |

¹ tiny 29-line feature but the file drifted a lot — port the hunk, not the file.
² the feature is ~577 lines across 3 commits, but the file differs by ~3500 lines **and** already carries our uncommitted xmem changes → do this **last**, carefully.
³ ported with **zero code changes** — this repo already ships `llama-ext.h` + the `embeddings_nextn` C API the driver needs.

## Key finding that de-risked the LayerSplit port

The target **already has the `t_h_nextn` / `embeddings_nextn` mechanism upstream** (`llm_graph_result::t_h_nextn`, `get_h_nextn()`, `cparams.embeddings_nextn`, context-side read; eagle3/step35/qwen35moe already use it for MTP). So the head-less-stage output needed **no new plumbing** — we reuse it. We also **dropped the fused-QKV** optimization that was bundled in the fork's gemma4 diff (needs model-loading changes, not core to the pipeline).

## What's done (branch `plan-a-port`)

- **5a — RPC `tensor_extras`:** 5 hunks in `ggml-rpc.cpp` — persist backend-set `tensor->extra` server-side keyed by data addr, re-attach on `deserialize_tensor`. Lets Adreno/HTP repack metadata survive the RPC round-trip.
- **4 — LayerSplit:** 5 edits in `gemma4.cpp` — `[ls,le)` loop bounds from env, conditional `inp_out_ids`, head-less early-return exposing the cut hidden via `res->t_h_nextn`. Driver enables it with `cparams.embeddings_nextn=true` + the env knobs.

## Where we are

**3 components in, all built + committed on `plan-a-port` (native CPU+RPC build, 0 errors).** Together these give a **buildable, functional single-model cross-device pipeline**: the LayerSplit graph hooks (4) + the driver (5b) whose `head`/`tail` modes are the correctness oracle and `tailnet`/`headnet` are a raw-TCP stage relay. The RPC patch (5a) is for the *weights-over-RPC* path (a separate, later transport option).

## ✅ M1 validation result (2026-07-06) — FIXED, the cut is bit-exact (commit `48b020120`)

Ran `mono` vs `head→tail` on `gemma-4-E2B-it-Q4_0` (35 L, CPU). Initially **failed at every cut** (residual-only relay lost gemma-3n's per-layer token embeddings; the token-less tail fell back to the padding token — the fork's `gemma4.cpp` had the identical, never-validated bug). **Fixed by chosen option ① (relay token IDs).** Now `head→tail` == `mono` **bit-exact (id + logit)** for all valid cuts, verified on 3 tokens.

**How the fix works:** the tail decodes a **DUAL batch** = relayed input token + injected residual. The token rebuilds the per-layer token embeddings and scaled token embedding exactly (`build_inp_per_layer` / `project_per_layer_inputs` read `ubatch.token`); the injected residual becomes `inpL` for the layer loop. Implementation reused the existing MTP hidden-injection input (`llm_graph_input_embd_h::h`) — **no new core class or API**. 3 files: `gemma4.cpp`, `llama-graph.cpp` (null-guard), `layersplit.cpp`.

**Second finding — shared-KV cut ceiling.** `shared_kv_layers=20` ⇒ layers 0–14 own KV, 15–34 reuse it (last KV-owning SWA layer = 13). Valid cuts are **k=1..13**; k≥14 breaks (tail's KV cache missing the reused entries). **Baseline cuts k=2 & k=3 are exact** → phones-hold-first-layers is safe. Memory: `gemma3n-perlayer-breaks-layersplit`.

## ✅ Multi-token + 3-way pipeline (2026-07-06, commit `99224d3a7`)

Driver generalized to N-token prefill + a `mid` stage. On gemma-4-E2B, a 12-token prompt through **head[0,2)→mid[2,3)→tail[3,35)** (the op15→op12→server baseline) is **bit-exact** vs `mono` (id=8784, logit=11.035674); 2-way exact for k≤13, breaks at k=14 (N-invariant). **The graph needed zero changes** — it was already N-token-general and already supported the middle chunk (`ls>0 && le<n_layer`); confirmed by a 4-agent read-workflow. act-file v2 = `{n_embd, N, tokens[N], residual[N*n_embd]}`.

## Historical next steps (superseded)

The numbered list below records what the Design A port expected at the time. Do
not execute it without an explicit selection from the current S8/MW plan.

1. **Real cross-device transport.** The socket relay modes (`tailnet`/`headnet`/`*stream`) still inject `token==NULL` (the pre-`48b020120` residual-only path) → they would resurrect the per-layer-embd bug for a layer-split tail. Give them the dual-batch + token-relay (like `run_tail`/`run_mid`) and add a `midnet`, then run head→mid→tail over TCP across real op15/op12/server.
2. **Android/Hexagon build** of the fix (cross-compile check on the phones).
3. Port **5d** (MTMD embd inject) and **1** (VQ admission) — moderate.
4. Port the hard **#2 dma-buf** last (its file drifted ~3500 ln + holds xmem).
5. `5c` cdsd is an *alternative* transport — the driver already has raw-TCP relay, so it's lower priority.
