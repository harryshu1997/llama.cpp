# S6 Evidence Audit for the S8 Atlas (V0a-R2)

Status: revised down after the V0a review. This file decides which S6-L / S7
results may enter the profile atlas. The V0a version OVER-CERTIFIED S6: it cited
raw v2 JSON records that do not exist on disk, treated a single-stream microtrace
as a B16/C512 result, and treated an unequal-boundary mechanism signal as a
quantitative entry. Corrected below.

Governing rule (MIXED_WORKLOAD_DESIGN.md section 3, PLAN.md, SCHEMA_CONTRACT.md
section 3): a passing atlas row needs support, no hidden fallback, kernel
provenance, a correctness certificate at the SAME shape/layer/attention, hot/cold
latency, resident/scratch/state bytes, boundary bytes + measured transfer, co-run
slowdown, temperature, exact hashes, p50/p95/p99 + CoV over >=7 processes with
PERSISTED per-process artifacts, and (for Gate B) a matched server control +
post-transfer SLO feasibility + measured server relief. Any missing field is
UNKNOWN and the route is ineligible.

## 0. What actually persists on disk

Audited `scratchpad/s6_latency_repair_v2/`:

- `dumps/res12_htp_decode_B1_C512.f32`, `dumps/res12_cpu_decode_B1_C512.f32`:
  residual tensors (3840 f32 = 15360 B, one decode vector) for a decode at **B=1,
  C=512, blk.2** (HTP vs CPU). These are the only artifact-backed decode
  correctness dumps. IMPORTANT: the DEVICE is NOT encoded in the artifact; the
  `res12` naming is consistent with OP12, and there is **no artifact attributable
  to OP15 decode** in this directory. So the OP15 2.95e-3 decode figure is
  UNPERSISTED; only a device-ambiguous (OP12-consistent) decode dump exists.
- `dumps/res_htp_prefill_B1_C64.f32`, `dumps/res_cpu_prefill_B1_C64.f32`: a
  prefill correctness point at B1/C64/blk.2 (also device-ambiguous).
- `sweep_dryrun/run_{0,1}_16_64.rec`: DRY-RUN sweep records (shape 16_64), not
  real device measurements.
- `worker_stress`, `parse_dual.py`, `resdiff_selftest/*`, `dual_sweep.sh`:
  host-side harness proofs and self-tests.
- ABSENT: any dualengine DUALJSON record for B16/C512 (SATURATED, SERVICE,
  fixedpair) and any ffnmerge FFNJSON record. The V0a "raw JSON under
  scratchpad/s6_latency_repair_v2/" citation was wrong.

## 1. Corrected eligibility verdicts

| S6/S7 result | prior V0a claim | corrected status | reason |
|---|---|---|---|
| oplayerprof / resdiff harnesses | tool (not a row) | TOOL, eligible for MW1 | fail-closed; proves placement via cb_eval |
| blk.2 B1/C512 decode correctness, device-ambiguous (OP12-consistent dump) | "OP15 2.95e-3 PASS decode island" | NARROW correctness POINT; OP15 attribution WITHDRAWN | the persisted dump has no device tag and is OP12-consistent; no OP15 decode artifact exists, so the OP15 2.95e-3 figure is unpersisted; the dump itself is B=1, one shape, ONE layer (blk.2), causal_local_swa -- does NOT certify a 48-layer decode island |
| blk.2 B1/C512 decode correctness OP12 (v75 AUTO) | "3.61e-3 PASS decode island" | NARROW correctness POINT only | B=1, one shape, one layer; reproducible; explicit-attention path |
| SATURATED throughput 1.93x (B16/C512) | "PROVISIONAL row" | NOT AN ATLAS ENTRY | no persisted raw record; 1 rep; CoV 0.060 > 0.05 |
| SERVICE / request latency +19% | "FAIL row (interference)" | NOT AN ATLAS ENTRY; weak qualitative signal only | the SERVICE leg is a SINGLE-STREAM (B=1) decode advancing positions 0..63 (layersplit.cpp:1678,1693-1694), has NO correctness check, and no persisted record; it is not a B16/C512 decode-island interference measurement |
| complete-FFN merge 1.91x | "LOWER_BOUND row" | MECHANISM SIGNAL, not an entry | TWO SEPARATE problems (see note): (a) the merged-vs-control comparison is not boundary-equal, and (b) the affine control holds a 2nd PHYSICAL weight copy (single_copy=0); plus no persisted record and 1 rep |
| v75 fused FLASH_ATTN_EXT | BROKEN | NEGATIVE evidence (keep) | rel_L2 0.5-0.8; gate returns false for opt_arch==75 |
| v81 fused FLASH_ATTN_EXT | 5.02e-3 marginal FAIL | UNVERIFIED marginal signal (keep, human review) | no persisted record; characterize before any policy |
| energy (all) | DEFERRED | DEFERRED | no valid physical J |

## 2. Not eligible as passing atlas rows (corrected)

1. SATURATED 1.93x, SERVICE +19%, FFN merge 1.91x, v81 5.02e-3: NONE has a
   persisted per-process artifact. Under the artifact rule they are not atlas
   entries of any kind (not even negative rows) until re-run with saved records.
2. The SERVICE number specifically is a single-stream B=1 decode over positions
   0-63 with no correctness gate. It cannot stand in for a B16/C512 batched
   decode-island co-run measurement. At most it hints that concurrent prefill
   perturbs a single decode stream; that hint is not quantitative.
3. FFN merge has TWO INDEPENDENT problems that must not be conflated: (a)
   BOUNDARY EQUALITY -- the merged path and its control do not compare the same
   boundary work (in/out tensor scope), so the speedup is not apples-to-apples;
   and (b) PHYSICAL WEIGHT-COPY COUNT -- the affine control holds a second
   physical weight copy (single_copy=0). Fixing one does not fix the other:
   boundary equality is a graph/measurement-scope issue; the weight-copy count is
   a memory-provisioning issue (a qualified per-tensor share could make
   single_copy=1 while the boundary comparison stays unequal, or vice versa). It
   is a mechanism SIGNAL, not a certified island, and needs BOTH an equal-boundary
   comparison AND a single physical weight copy before re-measurement.
4. The blk.2 correctness points certify ONE layer at ONE shape on a single
   stream. Per review finding 6, a blk.2 causal_local_swa point does not certify
   a 48-layer decode island. It is recorded as a correctness POINT with exact
   `layer_range=(2,3)`, `attention_class=causal_local_swa`, `shape B=1 C=512`.
5. Any energy value: DEFERRED; energy fields stay null across the atlas.

## 3. Valid negative evidence to preserve

Unchanged from V0a (these constrain candidate generation; MIXED_WORKLOAD_DESIGN
section 13): S3 output-row split FAIL; S4 GPU attention at realistic context
FAIL; S5 isolated phone whole operators vs A6000 FAIL; S6 concurrent prefill
perturbs a single decode stream (qualitative); v75 fused FA BROKEN; v81 fused FA
marginal. None is deleted or hidden.

## 4. Reproducible result to carry forward (scoped down)

OP12 (v75) AUTO explicit-attention decode: placement proven `[HTP0]`, no CPU
fallback, vs OP12 CPU rel_L2 3.61e-3 at **B=1, C=512, blk.2, causal_local_swa**.
This is a reproducible CORRECTNESS POINT for the v75 explicit-attention kernel on
one layer/shape. It is NOT a decode-island latency row and NOT a full-depth
certificate. To become an atlas row it needs: full or representative layer range,
the shape envelope, 7-process p50/p95/p99 + CoV with persisted artifacts,
boundary/resident bytes, a matched A6000 control, and measured server relief.

## 5. Requirements to promote any S6 result to a passing atlas row

| target | requirement |
|---|---|
| decode island (either phone) | measure across a representative layer set (not just blk.2) at a declared shape envelope; 7 processes, discard first, rotate order; PERSIST every per-process JSON + hash; record `layer_range`, `attention_class`, `graph_hash`, boundary/resident bytes; add A6000 matched control + `server_relief`; correctness vs CPU at the SAME shape |
| overlap SATURATED / co-run | 7-process persisted records; CoV <= 5% both legs; a co-run row is eligible only if the pairwise interference passes p95 + correctness at the REAL island shape (B16/C512), not a single-stream microtrace |
| SERVICE / request latency | a real advancing-KV service measurement at the island's batch shape with a correctness gate and persisted artifacts; current evidence is single-stream and unpersisted |
| FFN merge | BOTH independent fixes: (a) an equal-boundary merged-vs-control comparison, AND (b) a single physical weight copy (single_copy=1 via a version/generation-qualified per-tensor share); plus 7 processes, persisted records, 512-token real residual; then an OPTIONAL-mechanism row, not a coarse-island gate |
| v81 fused FA | characterize FA-on/off/CPU at B={1,32} C={32,1024} with persisted records; human review of the policy; certify decode islands only on the FA-off path meanwhile |
| energy | BLOCKED until MW5 |

## 6. Harnesses -- component tools PENDING known fail-closed repairs

These are candidate MW1 tools, NOT certified measurement instruments yet. Each
has known repairs outstanding and must be re-verified before it produces a
passing atlas row:

- `oplayerprof`: decode/prefill latency + cb_eval placement proof + real residual
  dump. Pending: independent re-verification of the fail-closed behaviors; it
  measures a Gemma DECODER layer ONLY and cannot fill RAG/vision rows (see
  EMBEDDING_MODEL_FUNNEL.md and the separate `embprof` spec).
- `resdiff.py`: cross-backend-vs-CPU correctness (rel_L2 + argmax + hash),
  fail-closed. Pending: task-appropriate metrics for non-decode islands
  (embeddings need cosine + top-k, not argmax; EMBEDDING_MODEL_FUNNEL section 3).
- `layersplit` dualengine (co-run/service): pending the lane-lifetime repairs in
  SUBSTRATE_AUDIT.md C5 (fault isolation, sticky error latching, bounded queue);
  its current SERVICE leg is single-stream (not a B16/C512 island measurement).
- `ffnmerge`: pending BOTH an equal-boundary comparison AND a single physical
  weight copy (two independent fixes, section 2 item 3).

Every future run MUST persist per-process artifacts and their SHA-256 so the atlas
row's `artifact_paths` / `artifact_hashes` are real.

Nothing in this file changes an S6 verdict or a support policy. It maps S6
evidence onto atlas eligibility and corrects the V0a over-certification.
