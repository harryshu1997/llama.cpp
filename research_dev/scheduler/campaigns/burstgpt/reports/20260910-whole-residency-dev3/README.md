# Whole-phone residency ownership and energy evidence recovery

This is a bounded repair and an unchanged three-request adaptive retest. No
baseline, long trace, native rebuild, commit, push, USB reset, GDM change or
unrelated process termination is authorized or performed.

## Diagnosis

The previous result's first Llama decision at 61.136337 s admitted the whole-phone
candidate but excluded it with `BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED`.
`MODEL_EPOCH_AUDIT_ONLY` appeared later, after the desktop epoch was selected;
it was not the original cause. Both cold-start energy paths were SHADOW.

The frozen initial automated observation store contains zero execution,
transition or template rows. The previous three-request run recorded diagnostic
energy only, with zero qualified transitions/templates. These are not qualified
cold-start measurements that can safely be copied into the current NCM route.

Three code defects were repaired:

- Automated evidence recovery was nested inside adaptive FFN helper discovery.
  Whole-model routes now recover compatible evidence independently, and selection
  sees the recovered costs immediately. All identity and attribution checks remain.
- Whole-phone and HTP residency shared one exclusive ownership group. The whole
  endpoint now has its own ownership resource; transitions and queued projection
  preserve unrelated HTP residency without claiming its bytes as freed. Compute,
  transport, memory limits and physical proof checks remain shared/strict.
- Whole-model admission only enforced the shared portfolio when an explicit HTP
  cap had been configured. It now checks the declared pool, current READY shards,
  workspace, whole-service peaks and reserve even without that optional cap.
  A larger Android MemTotal does not increase the declared allocation budget.

An already-resident unqualified service also retains its peak reservation;
energy qualification is not a prerequisite for memory accounting.

## Scope and validation

Production changes:

- `_unified/automated_candidates_ops/generation.py`
- `_internal/route_generation/identity.py`
- `adapters/catalog_materialization.py`
- `_internal/capability_contracts/catalog.py`
- `_internal/route_generation/feasibility.py`
- `_internal/route_generation/costing.py`
- `_internal/runtime_resources.py`
- `_internal/runtime_residency_projection.py`
- `_internal/runtime_residency_cohorts.py`
- `_unified/automated_selection.py`
- `_unified/automated_selection_ops/resources.py`
- `_unified/helper_preparation_ops/authorization.py`
- `_unified/helper_preparation_ops/memory.py`
- `_unified/placement_epochs_ops/frontier.py`
- `_unified/phone_residency_ops/publication.py`

Tests: `test_whole_phone_ownership.py`, `test_phone_memory_cap.py`, and
`test_automated_runtime.py`, plus the capability stub in
`test_session_cow_transaction.py`. Documentation: this report, `ARCHITECTURE.md`, and
`research_dev/talks.md`. Existing native, session transaction, helper attachment,
adaptive fraction, energy-measurement and cleanup implementations are unchanged.

The physical test uses normal energy-aware scheduling, requests 36/37/50,
arrivals 1/61/91 s and outputs 292/292/71. It reuses the existing NCM experiment
wrapper with a fresh root:
`/mnt/storage/s42-whole-residency-dev3-20260910-v3` (deployment adds `-deploy`).
Initial evidence, model/server/worker binaries, FFN shard files, CUDA graph mode,
desktop parents and preparation/cleanup accounting are unchanged.

221 focused/related tests passed in 102.492 s after the final ledger repair.
Both replay goldens remain unchanged. No complete scheduler harness was run.
The v3 physical run and cleanup audit passed. Successful execution is not proof
that Llama offloads or that dynamic placement beats the frozen references.

Replay goldens, unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

## Physical v3 outcome

PASS: 3/3 requests, 655/655 output tokens, accepted semantic-sanity checks,
exact terminal proofs, zero execution recoveries and zero USB reset recoveries.
Paid duration including runtime preparation and cleanup: 276.056278 s.
Post-run audit verified all 386 deployed source files, normal Android USB,
no remaining phone workers and the untouched GDM process. The copied archive's
787 artifact hashes all match. No baseline or longer trace was run.

| Request | Execution start/end (s) | Service latency (s) | Phone calls | Assisted tokens | Fraction-weighted/all tokens |
| --- | --- | ---: | ---: | ---: | ---: |
| Gemma 36 | 17.731 / 148.455 | 130.725 | 6,488 | 280/292, 95.89% | 88.10% |
| Llama 37 | 151.767 / 153.337 | 1.570 | 0 | 0/292 | 0% |
| Qwen 50 | 208.552 / 269.842 | 61.290 | 66 | 11/71, 15.49% | 15.49% |

Gemma's fractions were 100% for 229 tokens, 75% for 22, 50% for 18, 25% for 11,
and 0% for 12. Initial phone execution covered layers 0-15; expansion added
layers 16-23. All are CPU-resident FFNs under the unchanged desktop parent.
These percentages do not mean whole-model or all-layer offload. Gemma's
unassisted tail did not recur. Qwen used layers 12-17 at 17,408 columns and
100% for 11 tokens, then returned to desktop after
`VERIFICATION_INCONCLUSIVE / BOUND_RESOLUTION_UNAFFORDABLE` at token 20 with
51 tokens remaining. This existing sustained-Qwen limitation is not a measured
negative-energy result and was not changed in this repair.

### Residency and attachment

| Layout generation | Change | PREPARING (s) | READY (s) | Native load-to-READY (s) |
| --- | --- | ---: | ---: | ---: |
| 1 | Gemma HTP0, session gen1 | 5.585 | 27.352 | 11.281 |
| 2 | Add Gemma HTP1, session gen1 | 27.791 | 37.625 | 9.386 |
| 3 | Add Gemma HTP2, session gen1 | 38.420 | 48.839 | 9.856 |
| 4 | Replace only HTP2 with Qwen, session gen2 | 149.887 | 164.479 | 14.149 |

Native intervals run from LOAD_AUTHORIZED to READY. Scheduler intervals also
include transport/setup and publication; they are not interchangeable. Every
physical load has weight-read, HTP-init, upload, endpoint, VERIFIED and READY
timestamps. The first session is usable at 27.352 s and all three at 48.839 s.
All four loads used `weight_source=ffn_shard` with exact stored/executed masks,
index/shard/parent hashes and bytes in the physical result.

Gemma started on desktop at 17.731 s, before phone readiness. ATTACHED was
recorded at 27.595 s, with expansions at 37.902 and 49.236 s. Native HTP0/HTP1
calls continued during initial HTP2 loading (26 retained-session calls).
Per-session Gemma totals: HTP0 2,240; HTP1 2,240; HTP2 2,008. Qwen made 66
HTP2/gen2 calls. Load counts were 1/1/2, with retained generations 1/1.
Replacement occurred after Gemma completed; this run does not prove active
retained-call continuity during that replacement, reverse replacement or rollback.
Those paths retain focused software coverage, not a new physical claim here.

502 direct helper events were exported. All 370 saved phone telemetry samples
were VALID, with maximum age 2.627 s. The before/after analyzer files are
[BEFORE_PHONE_WAIT_TIMELINE.json](BEFORE_PHONE_WAIT_TIMELINE.json) and
[AFTER_PHONE_WAIT_TIMELINE.json](AFTER_PHONE_WAIT_TIMELINE.json): both have four
READY layouts, zero failed and zero never-prepared. Maximum pre-inference delay
fell from 157.1 to 117.6 s; first Qwen residency followed its arrival by 73.5 s,
still longer than one load time. The analyzer uses its recorded diagnostic
thresholds, not a substitute for controller eligibility.

### Why Llama still stayed on desktop

At 61.132233 s, the exact whole-phone candidate was rejected with
`PHONE_SERVICE_MEMORY_REBALANCE_PENDING`. It requested 3,000,000,000 additional
bytes, with zero unrelated HTP eviction credit. Three live Gemma shards use
8,493,465,600 bytes. Adding 4,951,552 bytes HTP workspace, the configured 3 GB
whole-service peak and the 805,306,368-byte reserve requires 12,303,723,520 bytes,
exceeding the declared 10 GB pool. Live Android MemTotal of 15.846 GB does not
override that budget. The 3 GB service peak is a conservative reservation, not
a newly measured OpenCL high-water mark.

Automatic whole-phone execution is therefore not complete. It still needs a
verified capacity-valid HTP resize (not credit for a proposed shrink), plus
qualified current-mode cold-start energy for the desktop/phone comparison.
The frozen initial automated store is empty; both cold transition energy
profiles remain SHADOW. The existing compatible-observation recovery fix cannot
manufacture those missing measurements. No phone route or session was forced.

### Energy, with historical comparisons only

Measured CPU package energy: 12.325001 kJ; measured GPU board energy: 8.330818 kJ.
Phone energy is assumed, with the same 0.875 W idle treatment as the references.
All preparation, exploration and cleanup remain inside the paid trace total.

| Active phone power | Fleet energy (kJ) | Below frozen matched desktop | Below frozen fixed GGG |
| --- | ---: | ---: | ---: |
| 3 W | 21.108 | 33.41% | 5.56% |
| 4.5 W | 21.257 | 32.94% | 5.49% |
| 6 W | 21.406 | 32.47% | 5.41% |

At 4.5 W this is 33.29% below frozen clean upstream desktop, 16.10% below GGQ,
33.49% below GQQ and 36.21% below QQQ. These are historical-reference differences,
not a fresh matched A/B or isolated proof of scheduler improvement. The strict
validator still rejects `A/B identity differs: catalog_sha256`; decoded source
and catalog differences and all compatibility checks are in [SUMMARY.json](SUMMARY.json).
Only final references-source-v7 results were read and their hashes are unchanged.

Relative to the previous successful v1 retest, duration changed from 316.883 to
276.056 s and fleet energy from 22.402 to 21.257 kJ at 4.5 W. Desktop preparation
changed from Gemma/Llama/Qwen 40.876/7.965/55.019 s to 11.480/2.482/51.130 s.
No cache flush or prefetch policy was introduced; repeated runs and artifact
checks can affect storage cache state. Do not attribute the entire speed/energy
difference to this code repair. CUDA evidence contains 2,386 actual graph launches
and 25 executable-reusing recaptures under the existing metric definition.

### Immutable physical artifacts

Remote: `/mnt/storage/s42-whole-residency-dev3-20260910-v3`.
Local: [physical](physical/), including commands, journals, native traces,
power samples, snapshots, helper events and source manifest.

- RESULT.json: `d7c80fb71a7cd00502d7b2e40ca8e6381452d73ee7ddb09d156988f948774c24`
- SOURCE_MANIFEST_EXECUTION.json: `71b6b417c3e108f74a9f7e31a6ad5e482205adccb1df204f5a10030e5a9ac128`
- CATALOG.json: `6d7d5c27679c1308ef1050aaf9c92644c85249a196a70df8cdae924afcbebf18`
- ARTIFACTS.json: `061d177d82dd09e21a8ef9bb0a93a15bf10384f1a22b12b0e6b0135909dab82a`

## Preserved failed attempt v2

The initial retest failed at Qwen submission with
`associated eviction lacks an exact exclusive anchor: op15-phone`.
Separating the catalog's whole-phone owner exposed the memory ledger's remaining
single-owner-per-device assumption. This was a repair regression, not a hardware
failure. The ledger now resolves exact executor/device anchors from the catalog
in generation, memory preview, commit and helper preparation. Associated desktop
allocations still require a matching anchor; cross-owner credit and stale
artifact, executor, generation or byte identities remain rejected.

The failed physical snapshot now generates all 27 candidates without the error
in a decision-only compiler check using the saved Qwen snapshot and manifest,
277 input tokens, 71 output tokens, original request identity, arrival and SLO.
This is a candidate-generation check, not a replay of physical timing.

Remote attempt: `/mnt/storage/s42-whole-residency-dev3-20260910-v2`.
Local preserved copy: [failed-v2](failed-v2/).
FAILURE.json SHA-256:
`f81a22a6547501888a5291f687fc69ddfc7edca1961f343ec6b8c22b07d87d49`.
Post-failure read-only checks found no desktop campaign/server process and
Android USB was restored to `ptp,adb`. No reset or manual process termination
was performed. Llama's initial rejection was still
`BASELINE_ENERGY_EVIDENCE_NOT_QUALIFIED`; that snapshot contained one READY Gemma
shard, not the previous run's three, so memory was not its rejecting condition.
