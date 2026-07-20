# S8-V0 Results

Status: V0b-P1 real-component normalization and structural replay COMPLETE.
Deterministic mixed-trace composition is NOT implemented, so the mixed-workload
Gate A remains BLOCKED. MW1 atlas and the oracle remain BLOCKED. No capacity,
latency, or energy claim. Nothing committed or pushed.

## Gate-A structural replay result (2026-07-16)

`structural_replay.py` implements the strictly structural consumer in
`NORMALIZATION_SPEC.md` section 13 for real component traces. It reads immutable
byte snapshots, verifies the raw source against its pinned config, validates
every request/sidecar/artifact/DAG/result schema, checks window semantics and
canonical arrival order, verifies every sibling output plus the artifact replay
digest, and accounts only observed demand. Before a certifying CLI result is
emitted, it reruns the pinned normalizer from the raw source in isolated Python
mode and byte-compares the complete JSONL/sidecar/artifact bundle. It does not
read a model profile or emit latency, capacity, power, or energy.

Real replays pass for the current-code BurstGPT median and all four RAGPulse
windows under `scratchpad/s8_gate_a_p1_20260716_r4/`:

| source/scenario | rows | input tokens | output tokens | retrieved chunks |
|---|---:|---:|---:|---:|
| BurstGPT median | 165 | 79095 | 18989 | 0 |
| RAGPulse low | 2 | 6976 | 761 | 12 |
| RAGPulse median | 12 | 33816 | 2606 | 60 |
| RAGPulse high | 27 | 65891 | 6994 | 97 |
| RAGPulse burst | 50 | 142115 | 12304 | 229 |

The Burst median contains 152 API requests and 13 conversation requests. The
RAG median structural output reproduced byte-for-byte from the independently
re-normalized component bundle. Structural replay tests pass `32/32`;
normalizer tests pass `24/24`; all 17 schemas compile under jsonschema and AJV;
the existing 52 schema fixtures and 15 semantic fixtures still pass. The final
structural reader is
`sha256:4ac6c29a5927e36612109364a91271e3eb9eb112992a7b7791124e99bb784761`;
the Burst and RAG median structural outputs are
`sha256:083d985c6692b6cf6c3f8ec081acc61bec555def072ca7aabb1c58994f0568ab`
and
`sha256:c86e6b25a6f0228040272dce8f1a2fa1e3d72e918805e8fd4efb49bf8a995c5e`.

The reader explicitly rejects `semi_synthetic` inputs because the frozen mix
composition transform has not been implemented. Therefore the honest verdict
is:

```text
REAL_COMPONENT_GATE_A_PASS
MIX_COMPOSITION_NOT_IMPLEMENTED
MIXED_WORKLOAD_GATE_A_BLOCKED
```

## V0b-P1 normalizer result (2026-07-16)

`normalize_trace.py` now implements the frozen BurstGPT CSV and RAGPulse JSONL
contracts. It verifies a temporary immutable source snapshot, selects exact
integer low/median/high/burst windows, emits canonical ASCII JSONL, persists
hash-bound trace and artifact manifests, and atomically publishes the complete
run directory. Certifying runs require direct CLI source execution; imported
execution fails closed. The pinned RAGPulse source's exact final `\n\n` required
one narrow contract repair: permit its single terminal blank line only, while
still rejecting interior or multiple blank lines.

Current-code RAG and Burst artifacts are under
`scratchpad/s8_gate_a_p1_20260716_r4/`. Both component normalizations were run
twice and reproduced byte-for-byte.

| source/scenario | selected bin | rows | output SHA-256 |
|---|---:|---:|---|
| BurstGPT median | 22706 | 165 | `3c57cdcb0ec79fcd3f64bacfc17c7acd8da68a1448724660aa7f0c69d5eecf46` |
| RAGPulse low | 629 | 2 | `c9edd08aaccdbfbb7f40de1b57223606d8fd595ca47150c37706cb870e0878af` |
| RAGPulse median | 146 | 12 | `ce4add9579ce2f172df58157c4535a38d59bae2fb6fe5b108cfcea9ebd420d1e` |
| RAGPulse high | 10 | 27 | `e68034324558761d408687ace7a1a96cd343b861e199dd6f0decb576b98e7548` |
| RAGPulse burst | 5 | 50 | `01439f5c16c5fe320289286e1481ed2fbe406629908bac8f016a94380767d83d` |

The full scans verified BurstGPT `5,344,021` records / `9,874` non-empty
15-minute bins / zero timestamp inversions, and RAGPulse `7,106` records / `583`
bins / one inversion / one permitted terminal blank. The current normalizer code
bundle is `sha256:75eb9df302b3fddbb14fe9028be2331d57e69d32153f52e3ca61549749d91e6b`.
RAGPulse's complete current-code run reproduced byte-for-byte in a second direct
CLI run.

Verification: normalizer `24/24`; schema fixtures `52/52` under jsonschema and
AJV; semantic fixtures `15/15`; every final request, trace sidecar, and artifact
manifest validates; stored output hashes recompute exactly. Adversarial
regressions cover immutable-source use, partial-stage failure, existing-output
preflight, persisted artifact binding, JSON `-0`, unsupported JSON source-field
mapping, direct-source execution, and mid-run code changes.

The real-component portion of Gate A is complete. Mixed composition remains
unimplemented and fail-closed. No runtime, scheduler, model, backend, or kernel
file was changed by this checkpoint.

## Verdict

`PARTIAL` -- real-component Gate A passes; mixed Gate A, B, and C do not.
Progression: V0a draft -> V0a-R (10
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

- Gate A (trace): REAL COMPONENTS PASS; MIX BLOCKED. Both pinned sources
  normalize deterministically and replay structurally from raw source through
  validated DAG demand accounting. The mix transform remains specified but is
  not implemented; `semi_synthetic` replay fails closed.
- Gate B (atlas): NOT MET. Only device-ambiguous blk.2 B1/C512 correctness POINTS
  exist; no island has a 7-process persisted latency row, matched server control,
  post-transfer SLO feasibility, or measured server relief; RAG/vision have no
  funnel-passing island.
- Gate C (oracle): NOT RUN.

## Claim boundary

- P0 pinned and inspected the sources; P1 now implements real-component
  normalization and structural replay. It does not implement mixed composition,
  inference replay, a downloader daemon, embprof, MW1 measurement, or an oracle.
  No runtime/model/graph/KV/scheduler/backend/kernel/S6/S7 file was edited.
- No energy inferred from latency; energy DEFERRED.
- Missing measurements are UNKNOWN/ineligible, never estimated as passing.
- No S6/S7 verdict or support policy changed.
- Datasets are held outside the git worktree (`/home/myid/zs89458/Documents/
  s8_sources/`); only small configs/fixtures are in the repo.

## Next (human review before proceeding)

Implement the deterministic mixed-component transform already frozen in
`NORMALIZATION_SPEC`, bind every input trace and sidecar, and run the same
structural replay. In parallel, the S12 exact-profile audit may define the
minimum varied-payload measurement atlas, but no real trace may receive a
latency until an exact profile row exists. The DECISION_CONTRACT remains
DRAFT-BLOCKED until its six V0c items are resolved.

## Remaining blockers

- Mixed Gate A is blocked on the unimplemented composition transform.
- Gate B not met (no persisted phone island row; second service class unproven --
  EMBEDDING_MODEL_FUNNEL Gate 1 is the likely blocker).
- Gate C not run.
- DECISION_CONTRACT DRAFT-BLOCKED (6 unresolved V0c items).
- RAGPulse redistribution: MIT-licensed repo; source bytes are held outside git
  and not redistributed here.
