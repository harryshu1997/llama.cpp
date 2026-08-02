# S38 Matched Server-Only Baseline

## Purpose

This is control `C0`: embedding, retrieval, reranking, and generation execute on
one server GPU with no phone work. It establishes the load, quality, and latency
that later phone-assisted controls must match or improve.

## Frozen cohort

`SERVER_COHORT_320.json` binds the selected trace:

- 320 requests from the densest 512-request, 100x-accelerated RAGPulse window.
- 151.55 seconds of offered arrivals.
- 943,165 requested input tokens and 79,685 requested output tokens.
- proportional coverage of all eight `(question_type, evidence_count)` strata.
- SHA-256 `c18c5d2d518fe4e4bcff7bd9fa1cf5a3d66f5e396e8b8e25ff18e3fc4a87b3a0`.

The selector does not take the first 320 records. It assigns proportional
stratum quotas, covers the requested token ranges within each stratum, then
restores arrival order. The workload remains `semi_synthetic` because timing
comes from RAGPulse while text and quality labels come from MultiHop-RAG.
RAGPulse input/output token counts remain demand labels; the runner separately
records the actual composed Gemma prompt and completion token counts.

## Matched pipeline

Each host runs three resident `llama-server` processes on one GPU:

| Stage | Model | Runtime configuration |
|---|---|---|
| query and document embedding | BGE-small-en-v1.5 F16 | CLS pooling, 8 slots |
| candidate reranking | bge-reranker-base F16 | rank pooling, 1 slot |
| final answer | Gemma-4 12B IT Q8_0 | 2 slots, continuous batching |

Every request performs query embedding, cosine search over 7,008 exact-token
chunks, unique-document top-20 selection, top-6 reranking, context assembly,
and greedy Gemma generation. Gemma receives a bounded thinking budget of
`min(64, requested_output_tokens / 2)` and an answer-only prompt. The offline
document index build is outside the paid window. Sixteen client workers
preserve the trace arrivals and bound the number of admitted requests.

The three model hashes, trace hash, payload hash, and index-manifest hash must
match. The runner also hashes the actual model path reported by each endpoint;
a command-line digest alone cannot certify the model.

## Hosts

| Host | Selected GPU | VRAM | Board power limit |
|---|---|---:|---:|
| A6000 server | NVIDIA RTX A6000, physical GPU 1 | 49,140 MiB | 300 W |
| desktop | NVIDIA GeForce RTX 4060 Ti | 16,380 MiB | 165 W |

Both `llama-server` binaries report version 9874 at source commit `99449bafa`.
They are separate native builds for their installed CUDA runtime and compiler,
so their executable hashes are not expected to match.

The A6000 host's physical GPU 0 is outside this experiment. The server commands
select physical GPU 1, which appears as logical CUDA device 0 inside those
processes. Physical GPU 0 had an unrelated user workload during acquisition;
selected-GPU placement is isolated, but host-level CPU and memory interference
was not controlled. The desktop uses its only GPU.

## Timing semantics

All request timestamps share one monotonic origin:

- `client_queue_us = worker_start_us - arrival_due_us`
- `service_us = finish_us - worker_start_us`, subject to at most 1 us of
  independent integer rounding
- `response_us = finish_us - arrival_due_us`

Only `response_us` is arrival-to-answer latency for an SLO. Service latency is
reported separately because it excludes client-executor queueing. Wall time
includes the offered-arrival window and final drain.

## Scope

This run measures latency, throughput, retrieval recall, and answer quality. It
does not measure energy. It does not demonstrate distributed retrieval, phone
execution, an SLO policy, or total-system savings.
