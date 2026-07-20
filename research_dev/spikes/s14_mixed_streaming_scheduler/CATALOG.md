# S14 CP0-c: frozen candidate island catalog

Status: FROZEN 2026-07-17, before any CP1/CP2 scheduler result is viewed.

    catalog_id    s14-cp0c-island-catalog-v1
    catalog_hash  sha256:3cf13792185f41919af3a6ee47fdb41eb236b93e0967b31082ab062dd3d4b3d5
    scope         CP0C_ISLAND_CATALOG_MECHANICS_NO_ENERGY   (energy NOT_RUN)
    git_head      5e92adf00

This closes the second CP0 deliverable in PLAN.md section 7 ("Measure candidate
`[0,k)` and stateless-island rows ... Freeze the finite candidate catalog before
scheduler results are viewed"). It instantiates the predeclared `ATLAS_MATRIX.md`
row contract (S8-V0a-R2) as a machine-checked, digest-bound, fail-closed
artifact and freezes the finite candidate GEOMETRY the slow solver may later
choose among. It runs NO fresh device work: every row binds already-frozen
S11 / Gate-1 evidence or a structural (deterministic) quantity, and every gap is
labelled and machine-enforced.

## 1. What "frozen" means here

The catalog is content-addressed by `catalog_hash` (sha256 over its canonical
serialization with `catalog_hash` removed). Freezing it now fixes the finite set
of candidate islands and each candidate's CURRENT evidence status BEFORE the
scheduler runs, so CP1/CP2 cannot (a) invent a new graph cut at dispatch time or
(b) tune the candidate set to flatter a scheduler result. It is a research
artifact, not a git commit; nothing is committed or pushed.

Freezing does NOT assert that every candidate is fully measured. It asserts that
the SET is committed and each row's evidence is honestly labelled and pinned. An
unmeasured candidate is frozen as an explicit, ineligible placeholder.

## 2. Frozen candidates (4 islands, 5 rows, 2 models)

| island | model | layers | attn | route | verdict | eligibility |
|---|---|---|---|---|---|---|
| gemma_head_0_2 | gemma-4-12b-it-f16 | [0,2) | causal_local_swa | OP15/HTP0 | LOWER_BOUND | INELIGIBLE (fallback unknown) |
| gemma_layer_2_3 | gemma-4-12b-it-f16 | [2,3) | causal_local_swa | OP12/HTP0 | LOWER_BOUND | INELIGIBLE (fallback unknown) |
| gemma_head_0_3 | gemma-4-12b-it-f16 | [0,3) | causal_local_swa | OP15/HTP0 | UNKNOWN | INELIGIBLE_UNMEASURED |
| bge_encoder_0_12 | bge-small-en-v1.5-f16 | [0,12) | bidirectional_none | OP15/HTP0 | LOWER_BOUND | INELIGIBLE_NO_LATENCY |
| bge_encoder_0_12 | bge-small-en-v1.5-f16 | [0,12) | bidirectional_none | OP12/HTP0 | LOWER_BOUND | INELIGIBLE_NO_LATENCY |

ZERO rows are dispatch-eligible. No single measured run yields BOTH a coherent
per-request island latency AND a same-run HTP no-fallback certificate:

- `gemma_head_0_2` has a clean B=1 stage latency (154962 us) from the batched
  sweep, but that sweep explicitly disclaims a placement certificate, and exact
  tokens do not prove HTP execution (a CPU fallback yields identical tokens).
  The E0 Long Readiness run DID certify HTP0 no-fallback for this exact route
  ("the phone's layer compute is assigned to HTP0, with only GET_ROWS assigned
  to CPU"), but on a different binary/run whose latency is a B=8 group, so it is
  not bound to this row. Its `fallback`/`kernel_provenance` are therefore UNKNOWN
  and it is INELIGIBLE.
- BGE is correctness- and no-fallback-certified (Gate-1) but its per-encode
  latency is unmeasured.

This is the true state. It was NOT the first draft: the first freeze labelled
`gemma_head_0_2` ELIGIBLE_COARSE on `fallback=none`. A 5-lens adversarial review
(see section 8) confirmed that `fallback=none` was unbound (its two cited
artifacts do not certify HTP placement, and one disclaims it) and that the row
stamped a binary that produced neither of its numbers. Both were fixed by
downgrading to the honest UNKNOWN and pinning per-role binaries; the headline
"one eligible row" did not survive contact with its own evidence.

### Evidence per row

- `gemma_head_0_2 @ OP15/HTP0` (LOWER_BOUND): stage latency p50 154962 us,
  resident 2925710752 B (includes tok_embd, carried by the head), KV 4 MiB,
  boundary in/out 476160 B at B=1 (31 x 3840 x f32), exact-token correctness vs
  SERVER_ONLY. Server-only control p50 156951 us, board HBM 24580 MiB; measured
  selected-board relief 860 MiB. `fallback`/`kernel_provenance`/`supported` are
  UNKNOWN, not `none`/`hmx_gemm`: the two bound artifacts certify exact tokens
  and the numbers, but neither certifies HTP no-fallback placement, and the
  batched sweep's RESULTS.md explicitly disclaims a per-node placement
  certificate. `build_hash`/`device_binary_hash` pin the sweep binary
  (786defa7/9472537) that actually produced the 154962 stage latency, NOT the
  repaired-checkpoint binary. `post_transfer_slo_feasible=false` because S11-E0
  proved the SERIAL route grew equal-work runtime 2.31x; S14's overlap thesis is
  precisely the mechanism that aims to change that, and the row records the
  measured serial fact, not the thesis. Bound to `s11_batched_route.json` and
  `FUNCTIONAL_RESULT.json`. Separate ROUTE-feasibility evidence that HTP0
  no-fallback IS achievable for [0,2) (E0 Long Readiness, different run/binary)
  is bound at catalog level as `S11_E0_PLACEMENT_CERT`.
- `bge_encoder_0_12 @ OP15/HTP0` and `@ OP12/HTP0` (LOWER_BOUND): pooled-CLS
  cosine 0.997331 / 0.997328 vs the CPU f32 reference, no fallback, explicit
  (fa=off) attention, on both v81 and v75 (Gate-1 `GATE1_HTP_BGE_FUSED_PASS`).
  Per-encode latency is UNMEASURED: Gate-1 captured only load-dominated
  whole-invocation walls (OP15 1244 ms vs CPU 179 ms), which are not a clean
  per-request island latency, so `p50_us` is null and both rows are
  INELIGIBLE_NO_LATENCY. `get_rows` on the f16 embedding table is a declared
  host CPU pre-stage OUTSIDE the island, so the island itself is fallback=none
  and `boundary_in` is the F32 token embeddings.
- `gemma_layer_2_3 @ OP12/HTP0` (LOWER_BOUND): exact-token correctness from the
  S11-B two-phone run, KV 2 MiB. `fallback`/`kernel_provenance`/`supported` are
  UNKNOWN: there is no in-tree OP12/v75 scheduled-buffer placement certificate
  for [2,3) (ATLAS_MATRIX has none). It has no isolated per-island latency, and
  its only evidence comes from a SERIAL two-phone chain, which PLAN.md section 2
  excludes from the initial catalog. Kept as a declared independent-placement
  candidate but INELIGIBLE.
- `gemma_head_0_3 @ OP15/HTP0` (UNKNOWN): declared, unmeasured. It freezes a
  larger homogeneous-SWA head candidate (layers 0,1,2 are all `causal_local_swa`
  for any sliding-window pattern >= 4, confirmed against ATLAS_MATRIX) so the
  solver's geometry frontier is fixed now, not chosen after results.

## 3. Eligibility rule (enforced by validate_catalog.py)

A row is SCHEDULER_ELIGIBLE iff `verdict in {PASS, LOWER_BOUND}` and
`correctness==pass`, `fallback==none`, `supported==true`, `p50_us` is non-null,
both boundary byte counts are non-null, and
`post_transfer_slo_feasible==true`. If it also has `n_proc<7` it is
ELIGIBLE_COARSE (provisional). Missing latency is INELIGIBLE_NO_LATENCY;
`post_transfer_slo_feasible!=true` is INELIGIBLE_SLO; `verdict==UNKNOWN` is
INELIGIBLE_UNMEASURED. The validator prints per-row eligibility; CP1 may
dispatch only eligible rows.

INVARIANT (enforced by test, honored by the builder): a row may assert
`fallback=none` only where a no-fallback certificate is actually bound among its
artifacts. For a stateless island that is the Gate-1 op-support gate (BGE rows
carry it). For a stateful head island that is a same-run scheduled-buffer
`PLACEMENTCERT`; no gemma row has one bound to the run that produced its latency,
so every gemma row is `fallback=unknown` and no row is currently eligible.
Exact-token equivalence is a correctness gate, NOT a placement gate.

## 4. Known DERIVED / provisional fields (honest gaps)

- `graph_hash` and `weight_set_id` are v0 DERIVED identities: deterministic
  functions of `(model_version, layer_range, attention_class)`. They satisfy the
  atlas requirement that a different range or attention class is a different
  identity, but they are NOT yet the compiled ggml graph digest
  (`test-export-graph-ops`) or the S9 prepared-image digest. CP1 replaces them
  with the real digests when it exports graphs and stages prepared images.
- `gemma-4-12b` `[0,2)/[2,3)/[0,3)` are homogeneous `causal_local_swa`. Heads
  that cross the first global-attention layer are a DECLARED FRONTIER not frozen
  as `mixed` candidates yet: the exact SWA/global boundary is the gguf
  `sliding_window_pattern`, which is read from the model at CP1 measurement time
  and added then (avoids asserting an unverified per-layer vector in a frozen
  artifact).
- `bge-small-en-v1.5-f16.gguf` (digest pinned) was not retained on host after
  Gate-1; `host_model_path` is the intended path.

## 5. Measurement gaps enumerated for the CP1 measured-atlas pass

These require fresh device work and are intentionally NOT run here (freeze
first, measure second; and per AGENTS.md STOP before energy):

1. A single COHERENT gemma_head_0_2 run that emits a clean per-request latency
   AND a same-run scheduled-buffer `PLACEMENTCERT` on ONE binary. Today the
   B=1 latency (sweep) and the HTP0 placement cert (E0 readiness) are on
   different binaries/runs, which is exactly why the row is fallback=unknown.
   This is the single highest-value gap: it is what makes any gemma island
   dispatch-eligible at all.
2. BGE per-encode latency (clean, repeated) on OP15/HTP0 and OP12/HTP0.
3. A6000-CUDA BGE server-only baseline: no CUDA `llama-embedding` binary exists
   in the local build tree, so `server_control` is null for the BGE rows.
4. >=7-process CoV / p95 / p99 for the coherent gemma run (gap 1) to reach a
   profile_row PASS rather than LOWER_BOUND.
5. Real `test-export-graph-ops` `graph_hash` and S9 `weight_set_id` (section 4).
6. Optional larger homogeneous-SWA heads and the first mixed head, once the gguf
   pattern is read; and an OP12/v75 [2,3) placement certificate.

None of these blocks the freeze; they are the CP1 work-list.

## 6. Reproduce and verify

    cd research_dev/spikes/s14_mixed_streaming_scheduler
    /usr/bin/python3 build_catalog.py            # byte-identical rerun
    /usr/bin/python3 validate_catalog.py --verify-artifacts
    /usr/bin/python3 tests/test_island_catalog.py   # 22/22

`build_catalog.py` and `validate_catalog.py` share only primitive helpers
(`catalog_common.py`); the validator re-derives and re-checks every value the
builder asserts (content-address hashes, cross-refs, boundary bounds, the
fail-closed PASS predicate independent of the schema, and on-disk digest
bindings). Schemas: `../s8_operator_island_affinity/schemas/`
(`island_catalog`, reusing `island_descriptor` and `profile_row`). jsonschema is
required and present in `/usr/bin/python3` (absent in the npu-harness venv).

## 7. Files

    island_catalog.json      the frozen artifact (canonical; catalog_hash above)
    build_catalog.py         deterministic builder
    validate_catalog.py      independent fail-closed validator + eligibility
    catalog_common.py        shared primitives (canonical json, sha256, derivations)
    tests/test_island_catalog.py   22 fail-closed tests
    CATALOG.md               this document
    SHA256SUMS.txt           digests of the above

## 8. Adversarial review (2026-07-17)

A 5-lens find-and-verify workflow (13 agents: validator fail-open, factual
accuracy vs bound sources, evidence binding, determinism/independence, and
contract/scope) reviewed the first freeze. It surfaced 8 candidate findings; 2
were confirmed on independent verification, both in the binding lens, both on the
gemma rows:

1. `gemma_head_0_2` asserted `fallback=none` (the sole predicate that made it the
   only eligible row), but neither cited artifact certifies HTP no-fallback
   placement and the batched sweep explicitly disclaims it. FIXED: downgraded to
   `fallback=unknown` (row now INELIGIBLE); bound the real E0 route-feasibility
   certificate at catalog level; documented that exact tokens are not a placement
   proof.
2. The row stamped the repaired-checkpoint binary as "the exact artifacts" while
   its `p50` came from the sweep binary and its `server_control` from a third
   fixed-route binary. FIXED: pinned per-role binaries (sweep binary on the row
   that carries the sweep latency; repaired binary on the two-phone correctness
   row) and dropped the "exact artifacts" framing.

Three regression tests lock both fixes (`test_fallback_none_only_where_bound`,
`test_gemma_row_binary_matches_its_measurement`, and the zero-eligible assertion
in `test_eligibility_statuses`). A follow-up validator audit found two additional
fail-open cases: a forged no-fallback row could bypass the post-transfer SLO gate,
and Python's JSON loader accepted duplicate keys or non-finite numeric constants.
The eligibility predicate now requires `post_transfer_slo_feasible==true`, and
the shared loader rejects duplicate keys and NaN/Infinity; two regression tests
cover each class. The suite is 22/22, and the manifest hashes below were refreshed.
The net effect remains an honest "zero eligible rows", with the single device
measurement that would change that (gap 1, section 5) named precisely.
