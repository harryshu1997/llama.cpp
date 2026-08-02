# S38 Trace-Bundle Results

Verdict: `F16_LONG_CONTEXT_MIXED_MECHANICS_PASS; Q8_RAG_C2_C3_BLOCKED`

## Downloaded sources

| Artifact | Revision | Records | SHA-256 |
|---|---:|---:|---|
| MultiHop-RAG questions | `71ac0d...` | 2,556 | `03cfb492...15bff` |
| MultiHop-RAG corpus | `71ac0d...` | 609 | `20b61b5...e4d28f` |
| RAGPulse primary trace | `99a6276...` | 7,106 | `cd371571...1e65` |

All six RAGPulse component files were downloaded at the same pinned commit. The
sources live outside git under `/home/myid/zs89458/Documents/rag_assets/`.

## Generated bundle

The canonical bundle is
`/home/myid/zs89458/Documents/rag_assets/composed/s38-v1/`.

| Output | Records | SHA-256 |
|---|---:|---|
| `op12_documents.jsonl` | 321 | `cfc1c4c...c5bd` |
| `op15_documents.jsonl` | 288 | `b09af589...afb2` |
| `payloads.jsonl` | 2,556 | `b7cb9966...47a` |
| `requests.jsonl` | 2,556 | `b51a1375...d8df` |
| `dense128_x100.jsonl` | 128 | `ea89f298...8563` |

The dense interval spans 2,683 source seconds and 26.83 replay seconds after the
frozen 100x acceleration.

## Calibrated server cohort

The final server-only control uses the frozen 320-request cohort bound by
`SERVER_COHORT_320.json`.

| Property | Value |
|---|---:|
| Requests | 320 |
| Offered-arrival span | 151.55 s |
| Requested input tokens | 943,165 |
| Requested output tokens | 79,685 |
| Trace SHA-256 | `c18c5d2d...87b3a0` |

Selection is proportional over all eight `(question_type, evidence_count)`
strata and evenly covers requested token demand within each stratum. It is not
the first 320 requests. A byte-for-byte independent rebuild matched the frozen
cohort and manifest.

Calibration was performance-driven and is disclosed: 192 requests took 6.12
minutes on the A6000; 384 took 12.40 minutes there but projected above the
20-minute bound on the RTX 4060 Ti. Its partial 22-request desktop attempt was
stopped and is not a result. Cohort 320 was then frozen before the two matched
final acquisitions.

## Workload composition

| Property | Count |
|---|---:|
| Comparison questions | 856 |
| Inference questions | 816 |
| Temporal questions | 583 |
| Null-evidence questions | 301 |
| Evidence only on OP12 | 352 |
| Evidence only on OP15 | 231 |
| Evidence on both phones | 1,672 |

Cross-phone evidence is required by 65.4% of all questions and 74.1% of
questions with evidence. This behavior follows the frozen hash split; it was not
inserted per query.

## Validation

- 2,684 request records (full plus dense) pass the existing S8 request schema.
- Document shards are disjoint and their union contains all 609 documents.
- Every ground-truth evidence reference resolves to one document and shard.
- Every payload is bound exactly once to a request.
- Full and dense request timestamps are nondecreasing.
- Every generated artifact is ASCII canonical JSON/JSONL and matches its
  manifest digest.
- Two independent builds produced byte-identical directories.
- `validate_bundle.py` reports `S38_BUNDLE_VALID` on the canonical bundle.
- Removing one byte from a copied request artifact is rejected with a byte
  count mismatch and exit status 1.

The generalized validator also accepts independently generated dense-128,
dense-256, and dense-512 bundles while deriving the exact dense filename and
record count from each manifest. All three report `S38_BUNDLE_VALID`.

## Server index

The all-server control index contains all 609 documents and 7,008 chunks. Each
chunk contains at most 256 BGE content tokens with 48-token overlap, plus the
exact CLS/SEP tokens. Chunking uses the serving model's `/tokenize` result, not
an approximate text splitter. The normalized float32 embedding matrix and
chunk metadata are hash-bound by index manifest
`7d62c859...c4c37`. Offline index construction is excluded from run time.

## Matched server-only control

Both hosts ran the same llama.cpp version `9874 (99449bafa)`, trace, index,
payloads, and exact model artifacts. Each used one GPU with BGE-small F16 in
eight embedding slots, bge-reranker-base F16 in one rerank slot, and Gemma-4
12B IT Q8_0 in two continuously batched generation slots. Sixteen clients
replayed the offered arrivals.

| Metric | RTX A6000 GPU 1 | RTX 4060 Ti |
|---|---:|---:|
| Wall time | 612.25 s (10.20 min) | 1,062.10 s (17.70 min) |
| Requests/s | 0.523 | 0.301 |
| Realized output tokens/s | 38.45 | 22.24 |
| Service p50 / p95 | 30.27 / 32.73 s | 52.95 / 56.75 s |
| Client queue p50 / p95 | 217.74 / 414.03 s | 414.57 / 816.81 s |
| Response p50 / p95 | 246.12 / 443.86 s | 464.33 / 869.22 s |
| Answer exact match | 55.94% | 55.94% |
| Answer token F1 | 57.59% | 57.54% |
| Retrieval mean recall | 91.08% | 91.08% |
| Rerank mean recall | 68.27% | 68.27% |

The offered rate is 2.112 requests/s, 4.04x measured A6000 throughput and 7.01x
measured RTX 4060 Ti throughput. Queueing is therefore intentional and must be
included in future SLO evaluation. Generation accounts for 99.7% and 99.8% of
median service time, respectively; retrieval offload alone cannot materially
relieve this control.

All 320 answers are nonempty. Gemma stopped normally for 307 A6000 and 306
desktop requests; 13 and 14 requests reached their trace-derived output limit.
RAGPulse token counts are demand labels, so the runner separately records
652,332/652,501 realized prompt tokens and 23,540/23,620 completion tokens.

`compare_server_baselines.py` reopens both manifests and proves the timestamp
equations, request conservation, and identical trace, payload, index, and model
bindings. The A6000 is 1.735x faster in requests/s and 1.729x in realized output
tokens/s. Cross-host floating-point differences are visible but do not change
aggregate quality: retrieved chunk sets agree for 97.81% of requests, reranked
top-6 sets for 97.50%, raw answers for 84.38%, and normalized answers for
86.56%. The persisted comparison is `SERVER_BASELINE_COMPARISON.json`.

| Bound artifact | SHA-256 |
|---|---|
| A6000 result manifest | `125b1203...65b25` |
| RTX 4060 Ti result manifest | `21624fce...6a49` |
| Matched comparison record | `175ad0cb...16520` |
| Frozen cohort record | `11e2bcb2...c6c92d` |

## Model status

`bge-reranker-base` revision `2cfc18c...` was downloaded from BAAI and converted
to a 563,949,984-byte F16 GGUF with 201 tensors. GGUF metadata reports BERT,
12 layers, hidden size 768, 12 heads, context 512. It now executes throughout
both server-only baselines. Phone HTP correctness, placement, memory, and
latency remain unmeasured.

This server-control acquisition does not include a distributed local index,
phone reranker, SLO result, or energy result. The later auxiliary F16 mechanics
run is reported separately below. The A6000 control was isolated to physical
GPU 1, but an unrelated workload occupied physical GPU 0; host-level
interference was not controlled. These are latency, throughput, and quality
controls only.

## Mixed prefill/decode integration

`mixed_generation.py` now connects S38 request semantics to the existing S36
StageNet runtime. It uses llama-server's own `/apply-template`, `/tokenize`,
and `/detokenize` endpoints, constructs one request-pinned `DynamicRequest`,
and hands it to `DynamicRouteRunner`. That runner already places prefill and
decode rows in the same cut-homogeneous `CutBatcher`; S36 physically measured
one B35 HTP call containing 32 prefill rows and three decode rows.

The integration is fail-closed. A route is considered only when its model
digest and GGUF file type match, correctness/placement/latency evidence all
pass, the real prompt and output lie inside the measured shape envelope, the
context fits, and the predicted bound fits the SLO for dynamic policy. Static
controls may omit an SLO but cannot bypass the other gates. Runtime failure
after phone dispatch is surfaced instead of silently replaying the request on
the server.

The matched S38 Q8_0 route is currently blocked, not passed. The frozen A6000
cohort has prompt lengths from 1,507 to 2,532 tokens (mean 2,038.54). The S36
phone proof used four-token prompts, an eight-token per-sequence context, and
F16 weights. Separately, S33 found that the same-Q8_0 phone route failed its
natural-prompt quality gate. Therefore the adapter correctly selects the
server for the current frozen baseline until CP5b and CP5c produce eligible
evidence.

The StageNet result now stops on declared EOG tokens and releases long prompts
in bounded 64-token prefill quanta. It still cannot reproduce the baseline
reasoning-budget behavior: llama-server changes sampling when the budget
expires, while the terminal StageNet worker returns only an argmax token.
`mixed_generation.py` therefore rejects every positive reasoning budget before
phone dispatch. A future route remains mechanics-only until CP5e closes that
sampler gap.

S33's layer-wavefront path was not copied into this adapter. It currently
requires Qwen2, rejects SWA KV, loads two complete model contexts, and has CPU
compute fallbacks. Treating that path as Gemma-ready would be incorrect.

## Long-context F16 physical mechanics

An auxiliary F16 run exercised the adapter's underlying runtime with real S38
RAG prompts. This is not a matched C2/C3 comparison: the frozen C0 baseline is
Q8_0, while this route uses F16 on the A6000, OP15, and shared CUDA tail.

The physical topology was:

```text
split-CUDA control: CUDA [0,8) -> CUDA [8,48)
phone treatment:    OP15 [0,8) -> CUDA [8,48)
```

Both routes used two persistent sequence slots, one unified 6,144-cell KV
pool, 64-row prefill quanta, and flash attention disabled consistently. The
two requests contain 1,507 and 2,532 prompt tokens. The second request was
admitted only after the first request produced a real decode batch.

| Physical observation | Control | OP15 treatment |
|---|---:|---:|
| Output decisions equal | 16/16 | 16/16 |
| Head batches | 78 | 78 |
| Mean / maximum head batch | 51.96 / 64 | 51.96 / 64 |
| Mixed head batches | 6 | 6 |
| Mixed-batch composition | 1 decode + 63 prefill | 1 decode + 63 prefill |
| Request latencies | 9.54, 17.34 s | 14.90, 25.86 s |
| Selected CUDA compute time | 6.401 s | 4.928 s |

The OP15 placement log reports 19,422 HTP compute nodes, 78 declared CPU
`GET_ROWS` nodes, zero missing compute buffers, and
`SCHEDULED_PLACEMENT_OK`. Selected-CUDA compute-time relief is 23.01%, but it
is not an energy measurement. The deliberately slack 500 ms gather window
also makes both treatment requests slower, so this run proves mixed-phase
batching and CUDA work displacement, not an SLO or latency win.

The first physical attempt used the default fused flash-attention route and a
100 ms gather window. It produced no mixed OP15 batch and matched only 15 of 16
token decisions. That run remains a frozen failure. Disabling flash attention
on all three stages produced the exact passing run above; no broader fused-FA
correctness claim is made.

## Termination and sampler boundary

A separate one-request run extended generation to 256 raw greedy decisions.
The split-CUDA control and OP15 route matched all 256 decisions, but neither
emitted a declared Gemma EOG token. The run therefore failed its required-EOG
gate. Its selected-CUDA compute-time relief of 17.96% remains informational.

The same messages sent through llama-server `/v1/chat/completions` stop after
70 tokens with the answer `Yes.`. A raw `/completion` request using the exact
pretokenized prompt also fails to stop within 256 tokens. The difference is
llama-server's reasoning-budget sampler, not chat templating alone. StageNet
does not yet implement that sampler or return logits sufficient for a host
sampler. This is why positive reasoning-budget requests now take the server
fallback before any phone work begins.

| Frozen physical artifact | SHA-256 |
|---|---|
| Bounded prompt subset | `2f69760a...08a44` |
| Initial failed run | `b7b69cfb...baa5` |
| Exact FA-off mechanics run | `a8de88f1...a05a` |
| FA-off OP15 placement log | `eb5a564c...15a` |
| EOG probe subset | `3e79ab6d...d8ec` |
| Failed required-EOG run | `77cede6b...eebc` |

The correct current verdict is therefore
`F16_LONG_CONTEXT_MIXED_MECHANICS_PASS; Q8_RAG_C2_C3_BLOCKED`. No phone energy,
server-board energy, answer-quality, or end-to-end distributed RAG benefit is
claimed from these physical mechanics runs.
