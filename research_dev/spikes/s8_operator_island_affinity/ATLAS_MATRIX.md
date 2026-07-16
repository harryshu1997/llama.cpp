# First Atlas Experiment Matrix (S8-V0a-R2)

Status: revised after the V0a review. Defines WHICH islands x routes x controls
the MW1 atlas must fill and the current state of each cell. No new measurement is
run. V0a-R changes: every row now carries an EXACT `layer_range`,
`attention_class`, `graph_hash`, and `shape_envelope` (review finding 6); the
blk.2 one-layer point is downgraded to a narrow correctness point (not a decode
island); the second service class is bound to a concrete model + funnel
(EMBEDDING_MODEL_FUNNEL.md); Gate B is amended (matched server control +
post-transfer SLO + measured server relief).

Row schema: `schemas/profile_row.schema.json`. A PASSING row needs support + no
fallback + kernel provenance + correctness at the SAME layer/attention/shape +
p50/p95/p99 + CoV over >=7 processes with PERSISTED artifacts + boundary/memory
bytes + temperature + build/device hashes + the Gate-B fields. Energy columns are
absent (DEFERRED).

## 0. Every row is scoped (review finding 6)

A profile row certifies ONLY its exact scope. The four scoping fields are
mandatory and a request is covered by a row only if all match / contain it:

- `layer_range` [start,end) of the absolute layers the island runs. A point at
  (2,3) does NOT certify (0,48).
- `attention_class`: `causal_local_swa` (gemma-4 SWA layers),
  `causal_global` (gemma-4 global layers), `bidirectional_none` (BERT embed),
  etc. A SWA-layer point does not certify a global-attention layer.
- `graph_hash`: exact compiled-graph identity. A different layer range or
  attention class is a different graph_hash and a different row.
- `shape_envelope`: validated batch/context/prompt range + dtype. A request
  outside the envelope is not covered.

## 1. Devices/backends (routes)

`A6000-CUDA` (server reference); `OP15-HTP`, `OP15-GPU` (Hexagon v81 + Adreno
840); `OP12-HTP`, `OP12-GPU` (Hexagon v75 + Adreno 750).

## 2. Controls per service (PLAN.md)

C1 server whole request; C2 phone whole request; C3 one contiguous resident
island; C4 Design A `A0`; C5 same-weight batch/merge (optional mechanism); C6
independent HTP/GPU co-run pair (only if certified); C7 server-only with
identical ordering. Arbitrary network micro-ops are rejected.

## 3. Generation (Gemma-4 12B, prefill + stateful decode)

n_layer_total=48. SWA and global attention layers are DISTINCT graphs. A row is
HOMOGENEOUS (one `attention_class`) OR, for a heterogeneous island, sets
`attention_class=mixed` and carries an exact per-layer `attention_class_by_layer`
vector (`schemas/profile_row.schema.json`). Two attention classes are NEVER put
into a single-enum cell. The two representative rows below are split into
homogeneous SWA-only and global-only islands.

| control | route | layer_range | attn_class | correctness | latency (7-proc, persisted) | Gate-B fields | status |
|---|---|---|---|---|---|---|---|
| C3 decode | OP15-HTP | (2,3) point | causal_local_swa | rel_L2 on B1/C512 dump (device-ambiguous, see S6) | NONE (no persisted record) | none | CORRECTNESS POINT only, NOT an island row |
| C3 decode | OP12-HTP | (2,3) point | causal_local_swa | v75 AUTO explicit, B1/C512 dump | NONE | none | CORRECTNESS POINT only |
| C3 decode SWA island | OP15-HTP | homogeneous SWA layers | causal_local_swa | UNKNOWN | UNKNOWN | UNKNOWN | NOT RUN -- required for an island row |
| C3 decode global island | OP15-HTP | homogeneous global layers | causal_global | UNKNOWN | UNKNOWN | UNKNOWN | NOT RUN -- required for an island row |
| C3 decode (fused-FA) | OP15-HTP | any | (n/a) | v81 5.02e-3 marginal (unpersisted) | -- | -- | INELIGIBLE (fused-FA path; human review) |
| C3 decode (fused-FA) | OP12-HTP | any | (n/a) | v75 BROKEN 0.5-0.8 | -- | -- | INELIGIBLE (v75 fused-FA broken) |
| C3 prefill | OP15-GPU | (2,3) point | causal_local_swa | stock OpenCL 2.9e-3 (prior) | UNKNOWN | UNKNOWN | POINT only |
| C4 A0 pipeline | 3-device | (0,48) split | mixed (+ per-layer vector) | proven route | UNKNOWN (atlas form) | UNKNOWN | route PROVEN; not an atlas row |
| C5 FFN merge | OP15-HTP | 1 layer | not_applicable (FFN) | 2.18e-3 (prior, unpersisted, UNEQUAL boundary) | NONE | none | MECHANISM SIGNAL, not a row |
| C6 HTP-dec \|\| GPU-pre | OP15 | -- | -- | single-stream signal only | NONE | none | INELIGIBLE (co-run not certified; S6 evidence unpersisted) |
| C1/C7 server | A6000-CUDA | (0,48) | mixed (+ per-layer vector) | n/a (ref) | UNKNOWN | n/a | UNKNOWN -- must measure baseline |

Notes: the certified decode path is FA-off / explicit attention; fused-FA is
INELIGIBLE. The blk.2 points are correctness POINTS at (2,3) causal_local_swa,
B1/C512 -- they do not certify a decode island, and their DEVICE is not encoded
in the persisted artifact (see S6_EVIDENCE_AUDIT.md; no OP15 decode artifact is on
disk). An island row requires a
representative layer set spanning BOTH SWA and global attention classes, a
declared shape envelope, and persisted 7-process latency. See S6_EVIDENCE_AUDIT.md.

## 4. RAG / embedding-rerank

Bound to EMBEDDING_MODEL_FUNNEL.md: `bge-small-en-v1.5` (BERT, 12 layers,
`bidirectional_none`) for embed; `bge-reranker-base` for rerank. Retrieval and
context-assembly are CPU-only DAG stages.

| control | route | layer_range | attn_class | funnel gate | status |
|---|---|---|---|---|---|
| C3 query_embed | OP15-HTP | (0,12) | bidirectional_none | Gate 1 (support) UNKNOWN | NOT RUN -- HTP BERT-op support is the likely blocker |
| C3 query_embed | OP15-GPU | (0,12) | bidirectional_none | Gate 1 UNKNOWN | NOT RUN |
| C3 rerank | OP15-HTP/GPU | (0,12) | bidirectional_none | Gate 1 UNKNOWN | NOT RUN |
| C6 embed(HTP) \|\| rerank(GPU) | OP15 | -- | -- | interference UNKNOWN | NOT RUN |
| C1/C7 server | A6000-CUDA | (0,12) | bidirectional_none | reference | UNKNOWN |

Top MW1 gap: get one embed/rerank island through funnel Gates 1-3 to attempt a
SECOND eligible service class. `oplayerprof` cannot measure this; an `embprof`
harness is required (EMBEDDING_MODEL_FUNNEL.md section 4).

## 5. Encoder / vision (Azure LMM vision-encoder)

A concrete vision encoder (e.g. a SigLIP/CLIP-class encoder, or the Gemma-4
vision tower if separable) must be selected and put through the SAME support-first
funnel (Gate 0 CPU ref -> Gate 1 op-support/no-fallback -> Gate 2 correctness ->
Gate 3 latency). All cells UNKNOWN / NOT RUN. If no encoder is supported with a
correct efficient kernel on a phone backend, the encoder class is documented as
unsupported and Gate B rests on generation + RAG.

## 6. Co-run interference sub-matrix (C6 detail)

A co-run pair is eligible only if its MEASURED pairwise interference passes p95 +
correctness at the REAL island shapes (not a single-stream microtrace).

| pair (same phone) | status |
|---|---|
| HTP-decode(B16/C512) \|\| GPU-prefill | UNKNOWN (prior S6 evidence single-stream + unpersisted) |
| HTP-embed \|\| GPU-rerank | UNKNOWN |
| HTP-vision \|\| GPU-decode | UNKNOWN |
| HTP-decode \|\| GPU-decode | UNKNOWN (likely bus-bound) |

## 7. Amended Gate B (review requirement)

Gate B passes only if at least one island in TWO distinct service classes is,
per route, ALL of:

1. supported with a correct efficient kernel, no CPU fallback (funnel Gate 1-2);
2. measured over >=7 persisted-artifact processes with p50/p95/p99 + CoV;
3. MATCHED SERVER CONTROL present: the SAME island/shape measured on A6000
   (`profile_row.server_control`), so relief is comparative not absolute;
4. POST-TRANSFER SLO FEASIBLE: p95 + measured boundary transfer + measured
   interference still meets the island SLO (`post_transfer_slo_feasible=true`);
5. MEASURED SERVER RELIEF present: actual freed GPU-ms / HBM bytes / HBM
   bandwidth (`profile_row.server_relief`), not phone latency.

Current standing: Gate B is NOT met. Generation has only blk.2 correctness POINTS
(no island latency row, no server control, no relief). RAG and vision have no
funnel-passing island. A second eligible service class is the top gap and none of
the five conditions above is currently satisfied for any phone island.

## 8. Measurement protocol for filling a cell

Authorized fail-closed harnesses only (S6_EVIDENCE_AUDIT.md section 6):
`oplayerprof` (Gemma decode/prefill latency + cb_eval placement), `resdiff.py`
(correctness), `layersplit` dualengine (co-run/service, AFTER the C5 lifetime
repairs in SUBSTRATE_AUDIT.md), `ffnmerge` (optional, AFTER equal-boundary
repair), and a NEW `embprof` for RAG/vision. Protocol: 7 processes, discard
first, rotate order, PERSIST every per-process artifact + SHA-256, prove
efficient-kernel + no-fallback, record layer_range + attention_class + graph_hash
+ shape_envelope + build/device hashes. Any missing field keeps the cell UNKNOWN.
