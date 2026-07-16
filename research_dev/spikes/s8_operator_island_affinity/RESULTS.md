# S8-V0 Results

Status: V0b-P0 (source pinning + final Gate-A input contract) COMPLETE. The
normalizer and server-only replay are NOT implemented (P0 stops before them). MW1
atlas and the oracle remain BLOCKED until Gate A passes. No mixed-workload
capacity or energy claim. Nothing committed or pushed.

## Verdict

`PENDING` -- Gates A, B, C are NOT RUN. Progression: V0a draft -> V0a-R (10
findings) -> V0a-R2 (executable machine-validated schemas) -> V0b-P0 (real
sources pinned + inspected, per-source configs frozen, mix/quantile/hash semantics
frozen, server-only replay defined). The DECISION_CONTRACT + its schemas stay
DRAFT-BLOCKED before V0c.

## V0b-P0 deliverables

- `SOURCE_PINS.md`: both sources fetched OUTSIDE git, inspected by a full pass,
  pinned. BurstGPT v2.0 `BurstGPT_3.csv` = 231682327 bytes, sha256
  `2299986a...a43b8f`, 5344021 rows, CC-BY-4.0 (zero-response failures 387963/
  7.26% KEPT in the primary trace). RAGPulse `data/0_trace.jsonl` @ commit
  `99a62769a91d5ebd17a2d4ddbbc88c1d16edc0e8` = 1923473 bytes, sha256
  `cd371571...801e65`, 7106 records, MIT.
- `schemas/source_config.schema.json` + `configs/{burstgpt,ragpulse}.config.json`:
  exact header/keys, column/value maps, session-blank->null, retrieved_chunks
  derivation, namespaced cache_keys order, modality defaults, source_fields, and a
  const `gate_a` block (deadline/priority always null, provenance none). Both
  configs validate under BOTH validators.
- Per-source timestamp policy (discovered by inspection): BurstGPT
  `require_nondecreasing` (0 inversions); RAGPulse `sort_stable` (1 inversion --
  a global strict rule would have wrongly rejected it).
- `NORMALIZATION_SPEC.md`: decimal quantiles replaced by rational-integer
  (`r=(q_num*N+q_den-1)//q_den`, low/median/high = 1/10, 1/2, 9/10); mix fully
  frozen (`t_mix=floor(t*scale_num/scale_den)+offset_us`, checked arithmetic,
  stretch/compress stated, unique ranks, rank-serialized streams, provenance
  rewriting, rank-namespaced event IDs, duplicate-ID rejection); server-only
  replay defined as a strictly structural reader/order/DAG pass (not inference).
- `SCHEMA_CONTRACT.md`: serialization authority for Gate-A records
  (schemas+configs > NORMALIZATION_SPEC > this file > informative prose); the
  "WORKLOAD_TRACES.md wins" cycle removed; a
  hash-preimage section stating exactly which bytes each hash covers; artifact
  `kind=normalize` now requires nonempty outputs each binding a sidecar hash.
- `validate_manifests.py`: semantic cross-field validator (bin arithmetic,
  first<=last, quantile-rank bounds, nonmonotonic<->policy, mix unique/sorted
  ranks + component binding, safe-integer bounds, normalize binding) + 15
  fixtures.
- `run_schema_tests.py` hardened: pins+verifies jsonschema 4.10.3 and
  ajv-cli 5.0.0, prechecks existence/regular-file/JSON-parse/index-completeness,
  and a `--index` override that PROVES a missing expected-invalid fixture fails.

## V0b-P0 test outputs

```
run_schema_tests.py : validators pinned OK; prechecks OK (51 entries);
                      15/15 schemas compile+load; fixtures 51 (21 valid +
                      30 invalid) failures 0; exit 0
--index /tmp/bad_idx.json (phantom missing invalid fixture): exit 1
                      ("ERR fixture missing" -> proves the guard)
validate_manifests.py --selftest : semantic fixtures 15 (3 valid + 12 invalid)
                      failures 0; exit 0
configs re-validate  : burstgpt + ragpulse VALID under jsonschema AND ajv
```

## V0a-R2 deliverables

| deliverable | file(s) | state |
|---|---|---|
| self-contained JSON Schemas (no preloaded refs) | `schemas/*.schema.json` (15) | validate standalone under both validators |
| fail-closed request/trace/profile/action schemas | `request`, `trace_manifest`, `profile_row`, `route_action` | if/then + bidirectional + PASS gating |
| positive + adversarial fixtures | `fixtures/` (19 valid + 24 invalid) | all covered |
| repeatable schema test runner (both validators) | `run_schema_tests.py` | 0 failures, exit 0 |
| frozen Gate-A normalization | `NORMALIZATION_SPEC.md` | load windows, parsing, offsets, hash binding |
| canonical bytes (portable) | `SCHEMA_CONTRACT.md` section 2 | no float/NaN/empty in trace records |
| single version/provenance authority | `WORKLOAD_TRACES.md`, `TRACE_SOURCE_AUDIT.md`, all schemas | `schema_version` + `semi_synthetic` only |
| DRAFT-BLOCKED decision contract | `DECISION_CONTRACT.md` + 4 schemas | status banner + 6 unresolved items |
| corrected embedding funnel | `EMBEDDING_MODEL_FUNNEL.md` | CLS+L2, HF-FP32 ref, cosine+topk, truncation |
| homogeneous attention rows | `ATLAS_MATRIX.md` | SWA/global split + per-layer vector |
| corrected S6 + substrate audits | `S6_EVIDENCE_AUDIT.md`, `SUBSTRATE_AUDIT.md` | see below |

## Schema validation (exact)

Validators: `/usr/bin/jsonschema` 4.10.3 and `npx --yes ajv-cli@5 validate
--spec=draft2020`. `run_schema_tests.py` output:
- compile/load: 15/15 schemas compile (ajv) AND load (jsonschema) from their own
  filesystem path, no preloaded refs.
- fixtures: 43 total = 19 expected-valid + 24 expected-invalid; failures = 0;
  RESULT ALL PASS; runner exit 0. (The runner returns nonzero on any unexpected
  result -- demonstrated during development when 4 mismatches gave exit 1.)

Adversarial rejections proven by fixtures: missing canonical field; deadline/
priority null-vs-provenance mismatch (both directions); unlabeled synthetic
deadline/priority; timestamp overflow; negative tokens; unknown field; float in
`source_fields`; mix without `streams`; real manifest carrying `streams`;
zero/negative scaling; profile PASS with unknown correctness / cpu fallback /
n_proc<7 / missing timing / missing server_control / missing server_relief /
empty artifacts; invalid layer range; invalid shape range; route action
inconsistent with `action_kind` (uncertified corun / two-assignment single /
one-request merged); mixed attention without per-layer vector.

## How each review-2 finding was addressed

- C1 (schema bundle): every schema is self-contained (local `#/$defs` only) and
  validates from its path under both validators with no `-r` preload;
  `request.schema.json` requires all canonical keys, applies bidirectional
  deadline/priority rules, integer upper bounds, scalar-only `source_fields`, and
  rejects unknown fields; `trace_manifest.schema.json` uses if/then so real
  rejects `streams` and mixed requires `streams[]` (each stream binds component
  trace + sidecar hashes + rank + rational scale + explicit offset);
  `profile_row.schema.json` makes `verdict:PASS` fail-closed. Fixtures + runner
  added.
- C2 (Gate-A normalization): single authority (`schema_version`,
  `semi_synthetic`) across all live docs; row-index quantiles replaced by aligned
  15-minute offered-token load windows (nearest-rank, full tie-break); frozen
  BurstGPT/RAGPulse parsing (encoding/BOM/newline/dialect/columns/grammar/
  monotonicity/exclusions); mix offsets are explicit committed integers (SplitMix64
  removed); canonical bytes forbid floats/NaN/empty in trace records with checked
  integer arithmetic; Gate-A v1 keeps `deadline_us`/`priority_class` null (synthetic
  SLO deferred -- no MW1 dependency); full hash binding (source+revision+config+
  code+output+sidecar; mixed binds every component).
- C3 (consistency): DECISION_CONTRACT + decision/action/lease schemas marked
  DRAFT-BLOCKED-BEFORE-V0c with 6 unresolved items; embedding funnel corrected
  (CLS+L2 pooling, HF-FP32->CPU reference chain, cosine + retrieval top-k for
  embeddings, score + ranking agreement for rerank, RAGPulse supplies no text so
  payloads are labeled synthetic, context/truncation coverage); Gemma atlas rows
  split into homogeneous SWA/global islands (or an exact per-layer attention
  vector); stale audit statements fixed -- no OP15 decode artifact exists on disk
  (dumps are device-ambiguous / OP12-consistent), oplayerprof/resdiff/dualengine/
  ffnmerge are component tools PENDING known fail-closed repairs, FFN boundary
  equality is separated from physical weight-copy count, SUBSTRATE summaries now
  match the downgraded detail (sockets ADAPT, OverlapLeg REFERENCE_ONLY, VQ ring
  ADAPT-concept/REFERENCE_ONLY-code), and per-tensor sharing is noted as existing
  and working single-load but unsafe across reloads.

## Gates

- Gate A (trace): NOT RUN (the normalizer + server-only replay are P1, not P0).
  The Gate-A INPUT contract is now executable: both sources are pinned + inspected
  (SOURCE_PINS.md), parsing/mappings/timestamp-policy are frozen in validated
  configs, windows/quantiles/mix/hash-preimages are frozen in NORMALIZATION_SPEC +
  SCHEMA_CONTRACT, and schema + semantic validators pass. Datasets are held
  OUTSIDE git.
- Gate B (atlas): NOT MET. Only device-ambiguous blk.2 B1/C512 correctness POINTS
  exist; no island has a 7-process persisted latency row, matched server control,
  post-transfer SLO feasibility, or measured server relief; RAG/vision have no
  funnel-passing island.
- Gate C (oracle): NOT RUN.

## Claim boundary

- P0 pinned + inspected the two real sources and froze the input contract. It did
  NOT implement the normalizer, server-only replay, simulator, downloader daemon,
  embprof, MW1 measurement, or oracle. No runtime/model/graph/KV/scheduler/backend/
  kernel/S6/S7 file was edited.
- No energy inferred from latency; energy DEFERRED.
- Missing measurements are UNKNOWN/ineligible, never estimated as passing.
- No S6/S7 verdict or support policy changed.
- Datasets are held outside the git worktree (`/home/myid/zs89458/Documents/
  s8_sources/`); only small configs/fixtures are in the repo.

## Next (human review before proceeding)

V0b-P1 / Gate A ONLY: implement the normalizer against the frozen configs +
NORMALIZATION_SPEC, add the parser-level negative tests, run the structural
server-only replay (section 13), prove byte-identical `output_sha256`
re-normalization, compute Gate A, then stop. MW1 profiling (including `embprof`)
and the oracle stay blocked until Gate A passes; the DECISION_CONTRACT stays
DRAFT-BLOCKED until its 6 unresolved items are settled at the start of V0c.

## Remaining blockers

- Gate A not yet computed (needs the P1 normalizer + replay).
- Gate B not met (no persisted phone island row; second service class unproven --
  EMBEDDING_MODEL_FUNNEL Gate 1 is the likely blocker).
- Gate C not run.
- DECISION_CONTRACT DRAFT-BLOCKED (6 unresolved V0c items).
- RAGPulse redistribution: MIT-licensed repo; source bytes are held outside git
  and not redistributed here.
