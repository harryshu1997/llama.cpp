# S8-V0 Mixed-Workload Operator-Island Atlas and Offline Oracle

Status: V0b-P0 complete; Gate A normalization and structural replay are NOT RUN.
Gates B and C remain blocked. `RESULTS.md` is the live checkpoint record.

## Question

Before implementing a distributed scheduler, do measured server/phone profiles
and real workload traces contain any operator islands whose resident phone
execution can improve SLO-valid server capacity?

This is a latency, correctness, residency, and simulation screen. Energy remains
disabled until the physical boundary is certified.

## Inputs

Read:

1. `AGENTS.md`
2. `research_dev/MIXED_WORKLOAD_DESIGN.md`
3. `research_dev/TWO_LEVEL_SCHEDULER.md`
4. `research_dev/WORKLOAD_TRACES.md`
5. `research_dev/NEXT_PLAN.md`
6. this plan and `RELATED_WORK.md`

Preserve the dirty worktree. Do not port the historical VQ, edit a model graph,
or build a production scheduler during V0.

## Workload set

Freeze at least three service classes:

- generation: prefill plus stateful batched decode;
- RAG or search: query embedding and reranking plus generation; and
- one encoder/background class: multimodal vision, OCR, audio, or equivalent.

Use at least two real trace scenarios and one deterministic mixed composition.
The initial preferred sources are BurstGPT/Azure LLM, Azure LMM, and RAGPulse.

## Candidate islands

For every service, include these controls where meaningful:

- server whole request;
- phone whole request;
- one contiguous resident stage;
- Design A route `A0` for Gemma generation;
- same-weight batched or merged island;
- independent phone HTP/GPU co-run pair; and
- server-only execution with identical request ordering.

Do not include arbitrary network micro-ops. An island records boundary tensor
bytes and state ownership.

## Existing substrate audit

V0 must document, with file/line references:

- current server request queue and slot lifecycle;
- current raw TCP and ggml-rpc data-plane behavior;
- current GGUF sharding, downloading/cache helpers, and partial model loading;
- current per-tensor HTP/OpenCL weight sharing and reload hazards;
- current dual HTP/GPU worker implementation and unresolved failures;
- historical VQ byte-table, CONWIP, singleton, and HEFT prototypes in
  `/home/myid/zs89458/Documents/llama.cpp`; and
- historical 16-byte remote telemetry design and Python roster in Unifer.

Classify each item as reuse, adapt, reference-only, or reject. Do not equate
process-local VQ depth with distributed status.

## Profile atlas schema

Every measured row must include:

```text
service, island_id, model/version/hash, graph hash
device, backend, shape/dtype, state class
supported, fallback, kernel provenance, correctness result
weights resident/cold, resident bytes, scratch/state bytes
boundary input/output bytes and measured transfer
warmup/load latency, p50/p95/p99, CoV, completed work
solo and co-run partner/slowdown
temperature, clocks, RAM, exact build and artifact hashes
```

No placeholder number may enter a passing route. Missing measurements produce
`UNKNOWN` and make that route ineligible.

## Trace and simulator contract

Build a standalone trace normalizer and offline simulator under
`research_dev/mixed_workload/`. It must not depend on a live phone.

Required policies:

1. server only;
2. whole-request offload;
3. fixed route `A0` where applicable;
4. fastest-device greedy;
5. measured-latency HEFT/EFT without residency;
6. residency-aware capacity oracle; and
7. a deployable rolling heuristic with hysteresis.

The oracle uses measured profiles and may optimize capacity only. Energy fields
remain absent/disabled. It must model:

- one queue per physical backend;
- weight download/load/warmup and minimum lease horizon;
- phone RAM and mutually resident model sets;
- transfer and host-relay time;
- state affinity and no hot KV migration;
- co-run interference rather than free overlap;
- bounded worker queues and failure fallback; and
- server GPU compute, HBM, and admission constraints.

## Gates

### Gate A: trace

- deterministic byte-identical normalization;
- source URL, release, license, checksum, row/window manifest;
- no silent clipping or invented deadlines; and
- at least two real scenarios plus one labeled mixed composition.

### Gate B: atlas

- at least one island from two distinct service classes is supported and
  correct on a phone efficient kernel;
- every shortlisted route has measured boundary, hot latency, memory, and
  fallback status;
- p95 and CoV are based on at least seven independent processes, discard first;
- no hidden CPU fallback, duplicate state, or unexplained weight copy; and
- hot residency is stable for 30 minutes or is explicitly deferred to MW2.

### Gate C: offline opportunity

Against the best server-only/fixed baseline at equal SLO, the oracle must show
at least one of:

- 10 percent more admitted/completed work;
- 10 percent less server GPU-ms;
- 10 percent less peak/occupied HBM; or
- avoidance of an otherwise required extra server capacity unit.

The gain must appear in at least two trace scenarios and survive documented
profile-error sensitivity. This is a capacity gate, not an energy gate.

## Stop rules

- If Gate A fails, repair trace provenance before profiling.
- If Gate B finds no islands from two workload classes, stop the general
  mixed-workload thesis and report the supported narrower scope.
- If Gate C fails even for the oracle, do not build a live scheduler.
- A grouped different-weight kernel, S7 ragged attention, or FFN merge is an
  independent optional mechanism. Its failure does not invalidate coarse
  islands, and its success does not bypass Gates A-C.
- Stop before any core scheduler, graph, KV, or kernel edit. Present V0 for
  human review.

## Deliverables

- `research_dev/mixed_workload/README.md`
- canonical trace schema, normalizers, manifests, and tests
- versioned profile schema and measured/UNKNOWN atlas
- deterministic simulator, baselines, and negative tests
- `RESULTS.md` with exact gate tables
- raw artifacts outside ephemeral storage with checksums
- current git status/diff and confirmation that nothing was committed or pushed
