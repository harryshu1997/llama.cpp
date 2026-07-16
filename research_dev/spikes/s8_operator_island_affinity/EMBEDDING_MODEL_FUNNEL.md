# Second-Service Support-First Funnel (S8-V0a-R2)

Status: frozen selection + measurement funnel for the RAG embedding/rerank
service class. Closes review finding 7 (the second service was not executable).
Specification only; no model is downloaded and no harness is built in V0a-R.

Why this exists: `oplayerprof` measures a Gemma DECODER layer only. It cannot
produce RAG or vision rows. Gate B needs an eligible island in a SECOND service
class, so the RAG embed/rerank class needs an EXACT model and a support-first
funnel that tests support BEFORE any latency number is recorded.

## 1. Exact model selection

Primary embedding model:

```text
name:        bge-small-en-v1.5
arch:        BERT (llama.cpp LLM_ARCH_BERT)
layers:      12   hidden: 384   heads: 12   ctx: 512
params:      ~33M
pooling:     model-default CLS pooling ([CLS] token) + L2 normalization of the
             output embedding (BGE uses CLS, not mean; the pinned pooling/
             normalization is part of the graph_hash and MUST match the HF model)
gguf dtype:  f16 (convert via convert_hf_to_gguf.py)
driver:      llama-embedding / llama-server --embedding
```

Reranker (RAG rerank stage):

```text
name:        bge-reranker-base
arch:        XLM-RoBERTa / BERT-class (llama.cpp reranking path)
pooling:     rank (llama.cpp --reranking / --pooling rank)
driver:      llama-server --reranking
```

Smaller fallback if RAM/support is tight: `all-MiniLM-L6-v2` (BERT, 6 layers,
hidden 384, ~22M params). Exactly one model is pinned per atlas run; its
`model_version` (GGUF SHA-256) and `graph_hash` go in the profile row.

## 2. The op-support risk (state it up front)

The phone HTP (Hexagon) backend in this tree was built for Gemma DECODE:
RMSNorm, causal + local-SWA attention, RoPE, HMX matmul, HVX softmax. A BERT
embedding graph needs a DIFFERENT op set:

- LayerNorm (not RMSNorm);
- bidirectional attention (NO causal mask);
- GELU FFN activation;
- learned absolute / typed position embeddings (not RoPE);
- CLS pooling + L2 normalization over the sequence.

Whether `ggml-hexagon` supports LayerNorm, bidirectional attention, GELU, and
pooling with NO CPU fallback is UNKNOWN and is the likely blocker. This funnel is
designed so that this is discovered at Gate 1 (support), before any latency
claim. HONEST EXPECTATION: HTP may not support a BERT graph; the embedding island
may be OpenCL/GPU-only, or unsupported on phones. If so, the second-service-class
claim rests on OpenCL embedding or Gate B fails for RAG -- that is a valid,
reportable outcome, not a reason to force the kernel.

## 3. Support-first funnel (gates in order; a gate failure stops the row)

Gate 0 -- reference chain (HF FP32 -> llama.cpp CPU):
- FIRST compare the pinned Hugging Face / SentenceTransformers FP32 model against
  the llama.cpp CPU GGUF on a fixed committed input set. This proves the GGUF
  conversion + CPU path reproduce the ORIGINAL model (pooling, normalization,
  tokenization) before any phone backend is trusted. Embedding: cosine similarity
  >= a committed threshold (e.g. 0.999) between HF-FP32 and llama.cpp-CPU vectors,
  and identical retrieval top-k on a committed corpus. Reranker: score error
  within a committed epsilon and identical ranking order. Only after HF-FP32 vs
  CPU passes is the llama.cpp CPU output used as the reference for Gate 2.

Gate 1 -- op support / no fallback on the phone backend:
- Place the graph on the target backend (HTP0 or GPUOpenCL) and use a cb_eval
  placement proof (same technique as `oplayerprof`) to record the backend of
  EVERY compute node. If ANY node runs on CPU (fallback), the island is
  `UNSUPPORTED` on that backend and the row stops here with
  `fallback=cpu`/`supported=false`. Enumerate exactly which op class forced the
  fallback (LayerNorm / bidir-attn / GELU / pooling) so the finding is actionable.

Gate 2 -- correctness vs CPU (task-appropriate metrics, not argmax):
- Embedding island: compare the phone-backend embedding to the Gate-0 CPU
  reference with COSINE similarity and per-dimension vector error, AND retrieval
  top-k agreement (identical top-k document set on a committed corpus). A decode-
  style argmax check is NOT meaningful for a dense embedding vector.
- Reranker island: compare rerank SCORE error to CPU within a committed epsilon
  AND require RANKING agreement (identical ordering, or a committed rank-
  correlation threshold) on a committed candidate set.
- Committed thresholds; `blocked`/`fail` stops the row. Cover multiple sequence
  lengths (section 3.1).

Gate 3 -- boundary + hot latency (only after 0-2 pass):
- 7 independent processes, discard first, rotate order, persisted per-process
  artifacts + SHA-256. Record `layer_range` (0..12 for full BERT),
  `attention_class=bidirectional_none`, `graph_hash`, `shape_envelope`
  (batch/seq range), boundary bytes (input token ids in, embedding vector out),
  resident/scratch bytes, p50/p95/p99, CoV, temperature, hashes. Add the matched
  A6000 control + `post_transfer_slo_feasible` + measured `server_relief` (Gate B
  fields, SCHEMA_CONTRACT section 4).

### 3.1 Context / truncation coverage (required)

Gates 0, 2, and 3 MUST each cover a committed set of sequence lengths spanning
short, typical, and at/over the model's `ctx=512` limit, including the exact
truncation boundary. The truncation policy (right-truncate to 512, committed) is
part of the `graph_hash` and MUST match HF. A row is scoped by its
`shape_envelope` (seq range) and does not certify lengths outside it; behavior at
and beyond 512 (truncation) is explicitly measured, not assumed.

### 3.2 RAGPulse payloads are synthetic (no source text)

RAGPulse supplies arrival timestamps, input/output lengths, and privacy-safe hash
locality ONLY -- it does NOT expose query or document TEXT. The embedding/rerank
correctness fixtures therefore use SEPARATELY LABELED SYNTHETIC payloads (a
committed synthetic corpus + queries), never presented as RAGPulse content.
RAGPulse drives arrival/length/locality in the trace; the synthetic corpus drives
the kernel correctness/latency. The two provenances are never conflated.

## 4. Harness (specify, do not build in V0a-R)

MW1 needs an `embprof` harness analogous to `oplayerprof` but driving
`llama-embedding` / `llama-server --embedding` / `--reranking`:
- select backend via the existing GGML backend-placement env;
- cb_eval placement proof (Gate 1);
- resdiff correctness vs CPU (Gate 2);
- 7-process latency with persisted artifacts (Gate 3).

It reuses `resdiff.py` and the cb_eval pattern from `oplayerprof`. It is NOT
`oplayerprof` (which is Gemma-decoder-specific). Building `embprof` is an MW1
task, gated on human approval after Gate A.

## 5. Outcome mapping

- Gate 1 UNSUPPORTED on both HTP and GPU -> RAG embed/rerank is not a phone
  island; report the narrower scope; Gate B for RAG cannot pass; the
  mixed-workload thesis rests on whatever second class (e.g. vision) does pass,
  or narrows to generation-only.
- Gate 1 supported on GPU only -> embedding is a GPU-lane island; still a valid
  second class if Gates 2-3 pass; co-run with a decode island on the same phone
  is a separate interference measurement.
- All gates pass on a phone backend -> a genuine second-service eligible island;
  populate the atlas row and attempt Gate B.

No latency or capacity number is recorded before Gate 1 and Gate 2 pass.
