# S38 Distributed RAG Trace and Model Bundle

Status: `F16_LONG_CONTEXT_MIXED_MECHANICS_PASS; Q8_RAG_C2_C3_BLOCKED`

## 1. Research question

Can two phones act as data-local RAG workers and LLM accelerators for one busy
server while preserving request SLOs?

Each phone owns a disjoint document shard. A query fans out to both phones:

```text
RAG request
    |
    +-------------------------+
    |                         |
    v                         v
OP12 local shard          OP15 local shard
BGE query embed           BGE query embed
local retrieval           local retrieval
local rerank              local rerank
    |                         |
    +-----------+-------------+
                v
       merge ranked evidence
                |
                v
      SLO-aware Gemma-4 route
 phone layer islands + server tail, or server-only
```

Both phones run useful data-local work even when a query's ground-truth evidence
is on only one shard. The global merge does not know that location before local
retrieval. The large reasoning model remains collaborative because it does not
fit the target latency/capacity envelope as a phone-only service.

## 2. Frozen workload sources

### Timing and cache locality

RAGPulse commit `99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8` supplies
7,106 real requests from a university Q&A deployment. It provides arrival
timestamps, token demand, sessions, retrieved-passage IDs, and cache locality,
but no text.

### Executable content and quality oracle

MultiHop-RAG revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`
supplies 2,556 questions, answers, a 609-document news corpus, and 0 or 2-4
ground-truth evidence documents per query.

The merged workload is `semi_synthetic`, not real. RAGPulse supplies arrival and
demand; MultiHop-RAG supplies executable text and quality labels. Each source is
retained independently.

## 3. Deterministic composition

`build_trace.py` performs these frozen transforms:

1. Verify all three input SHA-256 digests before parsing.
2. Reject duplicate JSON keys and malformed source records.
3. Stable-sort RAGPulse by `(integer timestamp, source line)` per the S8 rule.
4. Select 2,556 arrivals and map a SHA-256-ordered permutation of all 2,556
   MultiHop-RAG questions one-to-one. No payload is repeated.
5. Identify documents by SHA-256 of their URL.
6. Assign a document to `OP12` or `OP15` using digest modulo two.
7. Emit S8-compatible request records separately from executable payloads.
8. Select the densest 128-request interval and accelerate every arrival gap by
   exactly 100x for the bounded real-device replay. This derived trace remains
   labeled `semi_synthetic`.
9. For the matched server-only baseline, first build the densest 512-request
   interval, then select a deterministic 320-request cohort proportionally by
   `(question_type, evidence_count)`. Within each stratum, selection is evenly
   spaced over requested output and input lengths. This keeps both measured
   server runs in the 10-20 minute range without selecting an easy prefix.

Priority and deadline are null. Public sources do not provide real SLO labels.
An SLO sidecar is deferred until the measured stage profiles can define feasible
budgets without circularly manufacturing a scheduler win.

`validate_bundle.py` is independent of the builder. It reopens the output
bytes, enforces canonical ASCII JSON, checks exact source and output bindings,
validates every request against the existing S8 schema plus strict integer
types, reconstructs document shards and evidence bindings, and verifies that
the dense replay is the declared contiguous time transform.

## 4. Frozen model set

| Stage | Model | Artifact status |
|---|---|---|
| Query/document embedding | BAAI bge-small-en-v1.5 F16 GGUF | Existing, phone HTP correctness and placement previously measured |
| Candidate reranking | BAAI bge-reranker-base F16 GGUF | Downloaded, converted, and executing on both server GPUs; phone certification pending |
| Optional local rewrite/compression | Qwen3-0.6B Q8_0 GGUF | Existing; not required for the first pipeline |
| Final reasoning | Gemma-4 12B IT Q8_0 GGUF | Exact artifact fits both servers and is the matched baseline; phone routes must use this quantization for later comparisons |

`MODEL_MANIFEST.json` binds exact byte counts and SHA-256 values. The reranker
now executes in the server-only baselines, but it is not a phone island until
CPU-reference scores, HTP placement, numerical agreement, memory, and latency
pass on each phone.

## 5. Execution DAG

For every non-null query:

```text
fanout query
  -> embed on each phone
  -> search each local index
  -> rerank local top-k
  -> merge global top-k
  -> context assembly
  -> SLO-selected Gemma prefill route
  -> sticky decode route with continuous batching
  -> answer and evidence-quality evaluation
```

Null-evidence queries remain in the trace as a retrieval-negative control. The
scheduler may skip retrieval only after a measured routing policy identifies
them; the ground-truth label is not available to the online scheduler.

## 6. Experimental controls

- `C0`: all stages and all data on the server.
- `C1`: distributed phone retrieval/reranking, server-only Gemma.
- `C2`: C1 plus a static phone-prefix Gemma route.
- `C3`: C1 plus SLO-aware dynamic phone/server layer placement and batching.
- `C4`: C3 with one phone disabled, exposing whether two data shards help.

Report retrieval recall@k, rerank MRR/NDCG, answer exact match/F1, SLO
attainment, TTFT, TPOT, makespan, network bytes, realized batch sizes, server GPU
time, and selected-GPU board energy. Phone and total-system energy remain
unknown until a valid meter exists.

## 7. Next checkpoints

- [x] CP0: pin and download source datasets and model artifacts.
- [x] CP1: build and independently validate the deterministic trace bundle.
- [x] CP2a: chunk all 609 documents at exact BGE token boundaries and build the
  all-server control index (7,008 chunks, 256 content tokens, 48-token overlap).
- [ ] CP2b: partition the frozen index into the two existing document shards and
  deploy one local vector index per phone.
- [ ] CP3: certify bge-reranker-base CPU -> HTP correctness and measure its
  batch/sequence envelope on both phones.
- [ ] CP4: implement parallel query fanout, local top-k, and deterministic
  global merge.
- [x] CP5a: add the fail-closed S38 adapter for llama-server chat templating,
  exact tokenization, S36 mixed prefill/decode batching, and arbitrary-cut
  request pinning. The adapter falls back before dispatch when model identity,
  correctness, placement, latency, measured shape, context, or SLO evidence is
  missing.
- [x] CP5a-R: release long prompts to StageNet in bounded 64-token prefill
  quanta and stop generation on declared EOG tokens. Route admission now also
  rejects every positive reasoning budget because the terminal StageNet worker
  returns argmax tokens and does not implement llama-server's reasoning-budget
  sampler.
- [ ] CP5b: certify a matched Gemma route for the S38 generation artifact. The
  current Q8_0 phone route remains ineligible because S33 measured a quality
  failure; the earlier F16 phone route does not match the Q8_0 baseline.
- [x] CP5c: exercise an auxiliary F16 phone route at the 1,507- and
  2,532-token endpoints selected from the real S38 prompt envelope and provision a context/stream
  capacity that fits it. Two 3,072-token planning slots use one 6,144-cell
  unified KV pool. This closes the mechanics and capacity point only; F16 does
  not match the frozen Q8_0 C0 artifact.
- [x] CP5d-M: prove on OP15 that one physical F16 batch contains both a live
  decode row and real RAG prefill rows. Six B64 HTP batches each contained one
  decode row plus 63 prefill rows, with exact tokens against the split-CUDA
  control for the bounded eight-token run.
- [ ] CP5d: run matched-Q8_0 C2/C3 through the adapter and prove at least one
  physical batch contains both RAG prefill rows and live decode rows. S33's intra-phone
  GPU/NPU wavefront is a separate Qwen-only optimization until Gemma SWA KV,
  dual residency, and placement pass their own gate.
- [ ] CP5e: match generation termination semantics before a quality comparison.
  EOG detection is implemented and unit-tested, but a 256-token physical run
  proved that raw StageNet argmax does not reproduce `/v1/chat/completions`.
  The latter stops at 70 tokens because llama-server forces the reasoning-end
  sequence when the 64-token reasoning budget expires. Add a versioned sampler
  contract to the terminal worker or return sufficient logits to a host sampler
  before allowing reasoning-budget work onto a phone route.
- [ ] CP6: profile C0-C4 on the frozen cohort-320 trace before adding synthetic
  SLOs. C0 is complete on the matched A6000 and RTX 4060 Ti; C1-C4 remain.
- [ ] CP7: freeze feasible SLO classes from CP6 measurements and run the
  SLO-aware policy.

No energy-saving or end-to-end RAG claim is authorized before CP6.
