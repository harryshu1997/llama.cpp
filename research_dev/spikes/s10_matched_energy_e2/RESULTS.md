# S10-V0-R-E2 results

## VERDICT: E2_MATCHED_TIMELINE_MECHANICS_PASS_MEASUREMENT_NOT_RUN

The integer timeline integrator, artifact recomputation, typed measurement
boundaries, pair matching, uncertainty gates, and repetition-plan validation are
implemented and adversarially tested. **No physical measurement was run, and none
is authorized by this checkpoint.** No server relief and no energy saving is
claimed.

Two facts make the verdict tighter than "the mechanics work":

1. **No SERVER_WALL instrument exists on this host.**
   `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is not merely unmeasured; it is
   UNREACHABLE without new hardware. `ServerWallCapability` v1 is also
   mechanics-only: its proof is a hashed declaration, so every `MEASURED` v1
   capability is rejected with `E_CAPABILITY_UNCERTIFIED`.
2. **A physical label is unreachable by construction here.** The pair comparator
   is diagnostic only: even a valid MEASURED pair with a real 20% conservative win
   returns `MEASUREMENT_INVALID / PAIR_ONLY_NO_AGGREGATE_CLAIM`. A label requires
   `SUM_ALL_PAIRS_V1` over a complete, independently anchored repetition plan and
   every resolved run record. That evaluator and ledger are deliberately not
   implemented in this checkpoint.

## Instrument capabilities actually observed

First-hand on the live host (INSTRUMENT_AUDIT.md), inspection only, no sustained
measurement:

| instrument | observed | classification |
|---|---|---|
| NVML `power.draw` | present, driver 580.159.03. **TWO A6000 boards** (`GPU-45b611d6...` bus 51:00.0, `GPU-431a4567...` bus 9C:00.0). Vendor doc: "entire board... average power draw over 1 sec... accurate to within +/- 5 watts" | `GPU_BOARD` only |
| NVML `power.draw.instant` | present (25.12 W vs 25.93 W average); **cadence UNMEASURED** | `GPU_BOARD`; unusable until measured |
| RAPL | `intel-rapl:0`=`package-0`, `intel-rapl:0:0`=`core`. `energy_uj` is `-r-------- root root`, sudo needs a password. **No `dram`, no `psys`.** CPU is an AMD Threadripper PRO 5995WX | `COMPONENT_PACKAGE`; unusable twice over |
| IPMI / BMC / PDU / clamp | `ipmitool`/`ipmi-sensors`/`freeipmi`/`racadm` absent; `/dev/ipmi*` absent; hwmon only nvme/enp2s0/k10temp/dell_smm | `NONE` |
| USB / VBUS | none | `NONE`; phone energy stays UNKNOWN |
| Clock | `CLOCK_MONOTONIC` res 1e-09, `monotonic_ns` available | adequate |

The A6000 is Ampere, so `power.draw` is a **1-second average** with a **+/-5 W**
stated accuracy. That accuracy is now the enforced `uncertainty` floor for any
NVML timeline, charged per measured board.

## Existing-trace rejection reason (CP5)

`cp2_a6000_power_trace.csv`, re-measured for this audit and run through the real
importer:

```
rows=323  independent_updates=57  span_us=32262000  max_gap_us=106000
observed change rate ~1.77 Hz (busy window 1.64 Hz, 39 changes in 23.84 s)
poll rate 10.01 Hz
```

Rejected on **four independent grounds**, none relaxed:

- `E_UPDATES`: 57 value changes behind 323 rows against
  `MIN_INDEPENDENT_UPDATES=100`. The sampler polled a 1 Hz sensor at 10 Hz; 265
  rows are duplicates. Oversampling does not create information.
- `E_SCOPE`: NVML cannot be promoted to `SERVER_WALL`.
- `E_STATUS_CHANGE`: the trace spans P0, P2, P3, P8.
- `E_PAIRS`: a single timeline is not a matched comparison.

The gate was frozen before the trace was read and was **not lowered** to admit it.

## Review repairs carried into this checkpoint

- E1 now rejects target/reference payload aliases, phone-owned server-capacity
  evidence, stale route/correctness/thermal revisions, `NONE` transport carrying
  bytes, and duplicate node IDs before projection.
- The standalone temporal oracle and checker enforce the checked-in instance
  schema at runtime. Valid-input schedules, both frozen optima, and all 1187
  differential comparisons are unchanged.
- A GPU-board delta is named `boundary_delta_nj`. Its `server_wall_delta_nj` and
  phone-plus-external break-even budget are null. Only a complete `SERVER_WALL`
  boundary can populate those fields. (This corrected a real conceptual error: a
  board sensor cannot establish how CPU, DRAM, storage, fans, or PSU losses moved,
  so its delta is neither a server delta nor a phone-energy budget.)
- Window integration clips the first and last hold intervals exactly to the
  declared window.
- NVML uncertainty is charged per measured board, so a two-board sum carries twice
  the one-board worst-case floor.
- Any failed measurement pair invalidates the repetition set. It cannot be omitted
  while eight favorable pairs support a claim.
- Warmups are recorded separately, execution order has exact length, IDs and
  record digests cannot be reused globally, and a repetition set cannot
  self-declare an aggregate result.
- Work, outcomes, markers, and build/device identities are now rechecked against a
  hashed `e2.execution.v2` artifact. It cross-binds the `e2.raw.v2` artifact,
  timeline, run nonce, provenance, and paid-payload digest. The paid window is the
  complete marker range.
- Wall capability records resolve and parse their hashed proof, match its
  provenance, and apply its uncertainty floor. A measured v1 record is still
  rejected because that proof does not bind calibration or acquisition evidence.
- Control/treatment matching now rejects cross-device/build/source pairs and any
  reuse of raw-power, execution, run nonce, or paid power/status payload.

## What was built

- `CONTRACT.md` - frozen semantics: four labels, `SYSTEM_ENERGY_SAVING`
  inexpressible, typed capability, left-edge ZOH integration, matching rules,
  conservative decision, repetition set, sample gates, honest limits.
- `schemas/` - draft-2020, `additionalProperties:false`, local `$ref` only,
  integer-only: `realized_timeline.v1`, `matched_comparison.v1`,
  `repetition_set.v1`, `server_wall_capability.v1`.
- `src/` - `e2_canon` (reuses E1's canonical JSON/SHA-256 read-only, no E1
  decision logic), `e2_schema_gate` (runs under `/usr/bin/python3 -I`),
  `integrator`, `comparator`, `import_nvml_trace`.
- `fixtures/` - one concrete synthetic MECHANICS_ONLY pair whose integral is
  hand-checkable (300000 mW x 20 s = 6e12 nJ; 240000 mW x 20 s = 4.8e12 nJ,
  exactly -20%), plus a structural repetition-set fixture. Only pair 0 resolves to
  checked-in timelines; the other seven pairs are shape-only placeholders and are
  not aggregate evidence.

The synthetic pair computes `relief: true` and `meets_ten_percent_gate: true` and
is still labelled `MEASUREMENT_INVALID / SYNTHETIC_NO_PHYSICAL_CLAIM`. Correct
arithmetic, no physical claim.

## Fail-open paths discovered, and closed

**One I found myself before the audit.** `compare()` took `samples=None` and
skipped recomputation unless a caller opted in - which no caller, including the
CLI, ever did. A forged `energy_nj=1` validated clean on the strength of a correct
artifact hash. Recomputation is now mandatory; the parameter is gone and a test
asserts the signature.

**An independent red team then found nine more. All real, all closed, each with a
regression:**

| # | severity | hole | why it mattered |
|---|---|---|---|
| F1 | MAJOR (CRITICAL once the aggregate lands) | `build_comparison()` accepted a label, checking only set membership | importing the module stamped a digest-valid, schema-valid `SERVER_RELIEF_PASS` onto junk whose treatment burned 1e15 nJ MORE, on a host with no wall instrument. The next checkpoint's aggregate evaluator calls exactly this function |
| F2 | MAJOR | quality gates counted the WHOLE artifact; energy integrates only the window | padding busy samples outside the paid window cost zero energy and bought unlimited `independent_updates`. It admitted the exact 57-update trace the contract says it rejects |
| F3 | MAJOR | TOCTOU: hash opened the path, parse re-opened it | two opens, two files. The red team won the race **74/400** with no privileges, producing a record whose declared energy described bytes other than the ones it pinned |
| F4 | MAJOR | a timeline's own `status` was never read | `FAILED` and `INELIGIBLE` runs reached `decide()` with `relief=True` |
| F5 | MAJOR | `E_STATUS_CHANGE` was opt-in | `pstate` was optional and scalar, so the check was **dead code across all 104 tests**; the real trace spans four pstates |
| F6 | MINOR | additive guard keyed on `RealizedTimeline` only | `MatchedComparison` and `RepetitionSet` - more aggregated, not less - passed through |
| F7 | MINOR | forbidden-label scan contiguous and case-sensitive | lowercase evaded it |
| F8 | MINOR | `included_rails: []` accepted | a timeline measuring no rail was valid |
| F9 | MINOR | `normalizer_digest` schema-required, checked nowhere | a timeline could announce any normalizer |

Fixes: the label is derived, never supplied (`build_comparison` calls `compare`
internally; its signature no longer has `label`); quality is measured by
`window_quality(...)`; `read_verified_artifact` reads ONCE and returns the bytes the
parser must consume; `status != OK` is `E_PAIRS`; `pstates` is required, one per
sample, typed, and checked over the hold intervals that intersect the paid window;
the additive guard covers all three
aggregate kinds; the label scan is normalized; `included_rails` needs
`minItems: 1`; `normalizer_digest` is pinned to the frozen rule.

**A final review found six more pre-aggregate binding gaps.** Capability proof IDs
and uncertainty floors were unused; matching omitted device/source/build identity;
one raw trace could serve both roles; work/markers/outcomes were self-declared;
repetition indices and warmups could hide/reuse attempts; and CLI `--out` filled
repetition/order provenance with empty-list hashes. These now fail closed. Pair
records carry timeline digests, indices are contiguous, warmups are disjoint,
execution metadata has a hashed artifact, and `--out` remains disabled until the
all-pairs evaluator exists.

**The latest repair closes the remaining evidence-join and diagnostic-record
gaps.** Raw and execution evidence now use v2 formats joined by run nonce,
provenance, artifact identities, and the digest of the exact paid power/status
intervals. P-states are bounded printable strings. The gap gate includes the age
of both real bracketing observations, while status changes use the same half-open
hold intervals as integration. Repetition validation rejects timeline-ID and
record-digest reuse across every pair and warm-up, and validates the typed
instrument/scope relation. `build_comparison` derives its repetition and order
bindings from an exact pair in a validated set. `MatchedComparison` v1 is
diagnostic-only in both schema and runtime, even after resealing. These scenarios
are persisted as unit regressions; no unpersisted external kill-chain script is
claimed as evidence. The builder also rejects a repetition set whose
scope/instrument differs from the timelines, and converts malformed timeline input
to `ComparisonError` without a traceback or partial record.

**A post-verdict live audit found four more executable defects; all are now
closed.** A hostile `sitecustomize` could preload a fake bare `canon` module and
bypass the integer gate; E2 now loads the pinned E1 file under a private absolute
module identity, with a hostile-preload regression. A resealed comparison could
change copied energies, uncertainty arithmetic, gate booleans, or its reason;
`validate_comparison` now requires and resolves all source records and recomputes
every field and returns stable failure lists rather than leaking validation
exceptions. Equivalent power functions split into redundant equal samples now
coalesce before paid-payload hashing. The unused `measured_update_period_us` field
was removed, and the NVML importer now parses timestamps and decimal milliwatts
with exact integer arithmetic, rejecting excess precision and hidden columns.

The same audit identified two nonlocal proof gaps that are intentionally not
papered over. Aggregate counts and opaque digests do not prove per-request model,
output-token, terminal, or correctness equivalence. Markers do not prove that
prefetch/warm-up occurred after the start or that transfer/cleanup and all queued
actions completed before the end. These are next-gate blockers, not E2 passing
claims.

**One red-team claim I could not fix, and did not pretend to.** F7 also reported
that splitting the phrase across two free-form fields evades the scan. It does,
and no contiguous substring scan can catch it: canonical JSON sorts keys, so the
fields are not adjacent. The scan is defence in depth; what actually blocks the
claim is that `result_label` is a closed enum in both the schema and
`ALLOWED_LABELS`. That limit is now an executable test
(`test_a_split_label_is_NOT_caught_by_the_scan_and_need_not_be`), not a footnote.

**What the red team could NOT break**, worth as much as what it did: instrument
promotion (NVML/RAPL -> `SERVER_WALL` both `E_SCOPE`, including relabelling the
free-form string to "Yokogawa WT310 whole-server wall meter"); the E1 float/bool
kill shot (blocked at `validate_timeline` AND independently inside `decide()`);
the 10% integer cross-multiplication at the exact boundary (10.0% passes, 9.9999%
fails); `control_lower <= 0`; path traversal, absolute paths, symlink files, and
symlink directory components; and `compare()`'s label discipline.

## Commands and counts

```
bash research_dev/spikes/s10_matched_energy_e2/scripts/run_tests.sh   # exit 0
  -> E1 baseline byte-identical (before): 45 files
  -> S10_E2_FIXTURES_OK                    (deterministic regeneration)
  -> Ran 152 tests ... OK
  -> synthetic pair: relief arithmetic true, label MEASUREMENT_INVALID
                     - SYNTHETIC_NO_PHYSICAL_CLAIM
  -> existing A6000 trace: rows 323 independent_updates 57 -> REJECTED
  -> S10_E2_NEGATIVE_PASS (30 cases)
  -> E2 determinism across 5 processes/seeds:
     50b68e886fb9936eee9c7a0763e9bc20474babfe4329b6e0c57002144006f812
  -> E1 baseline byte-identical (after): 45 files
  -> S10_E2_MATCHED_TIMELINE_TESTS_PASS

bash research_dev/spikes/s10_power_frontier_repair/scripts/run_tests.sh  # exit 0
  -> Ran 201 tests ... OK (28 foundation + 173 evidence)
     S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS, S10_V0R_TYPED_EVIDENCE_TESTS_PASS,
     S10_V0R_EVIDENCE_NEGATIVE_PASS (39 cases),
     S10_V0R_CLI_NEGATIVE_PASS (18 cases)
     differential: 1187 compared, 13 jointly infeasible, 0 mismatches

git diff --check                        # clean
ASCII scan of the E2 directory          # all ASCII
find ... -name '__pycache__' -o '*.pyc' # 0
```

152 unit tests and 30 negative CLI cases pass. Each negative exits nonzero with a
stable `E2_FAIL`/`E2_IMPORT_REJECTED` prefix and no traceback. They cover the
required list: missing/truncated/modified artifact, traversal, absolute path,
symlink, hostile PYTHONPATH, duplicate keys, float/bool as integer, nonmonotonic
and duplicate timestamps, missing marker, excessive gap, too few updates,
negative/overflow values, mismatched workload/trace/policy/SLO digests, different
work, different terminal outcomes, different windows, different scope/rails, NVML
relabelled SERVER_WALL, RAPL relabelled SERVER_WALL, phone energy as zero,
uncertainty omitted, roles swapped after the fact, dropped repetition, duplicate
run, unbalanced order, synthetic promoted to MEASURED, SYSTEM_ENERGY_SAVING
injected, and an aggregate offered to the additive solver.

## E1 manifest result

`E1_PRE_REPAIR_MANIFEST.txt` preserves the original reviewed snapshot.
`E1_BASELINE_MANIFEST.txt` pins the final repaired 45-file baseline. The E2 runner
checks that pin before and after its own suite; the E1 run passes 201 tests (28
foundation + 173 evidence), 39 evidence negatives, 18 CLI negatives, and 1187
differential comparisons with zero mismatches.

## Remaining blockers before the first controlled physical A/B

1. **Resolved same-work and lifecycle closure.** Bind a workload manifest, model
   identity, per-request terminal/output/correctness ledger, initial/final cache,
   residency and thermal state, and a drain proof for all queued actions. Every
   prefetch, warm-up, transfer, execution, and cleanup cost must be inside the paid
   boundary or covered by a frozen steady-state amortization rule.
2. **The aggregate evaluator and anchored attempt ledger.** A pair is diagnostic
   only. A physical label needs `SUM_ALL_PAIRS_V1` over every resolved record in a
   complete, independently precommitted repetition plan. The current validator is
   structural and the fixture is shape-only beyond pair 0. No aggregate is built.
3. **A GPU_BOARD A/B is possible TODAY**, designed around two hard facts: at the
   observed ~1.64-1.77 Hz change rate a run needs roughly 60 s of steady state per
   timeline to reach 100 changes, and the +/-5 W accuracy must enter the decision
   as uncertainty rather than be dropped. `power.draw.instant`'s cadence should be
   measured first; it may shorten that. Needs separate authorization.
4. **SERVER_WALL needs new hardware and a stronger capability record.** An inline
   wall meter, or a BMC/PDU with readable input power, is absent. In addition,
   `ServerWallCapability` v1 is only a hashed declaration. A later version must
   bind topology, calibration validity, and raw acquisition evidence before a
   measured wall claim is eligible.
5. **TOTAL_WALL additionally needs the phone/USB/charger boundary**, unchanged
   from E1 and outside E2's scope by construction.

After blocker 1 is closed, the honest ceiling with this host's present
instrumentation is `GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED` - relief on one
boundary, with the host CPU, DRAM, and PSU losses outside it and possibly moving
the other way. Phone, USB supply, and charger energy remain UNKNOWN, never zero.

## Files changed

Added, all under `research_dev/spikes/s10_matched_energy_e2/`: `PLAN.md`,
`CONTRACT.md`, `INSTRUMENT_AUDIT.md`, `RESULTS.md`, `E1_BASELINE_MANIFEST.txt`,
`E1_PRE_REPAIR_MANIFEST.txt`, `schemas/` (4), `src/` (5), `tests/` (2),
`scripts/` (2), `fixtures/` (4 records + 2 raw artifacts).

Updated: `research_dev/NEXT_PLAN.md` (E1 complete, E2 the current gate),
`research_dev/talks.md` (status block + one newest-first entry).

**Not touched**: everything under `research_dev/spikes/s10_power_frontier_repair/`
(verified byte-identical before and after), tools/server, llama-server, model
graphs, KV internals, backend schedulers, phone-PIM runtime/protocol, and
CUDA/HTP/OpenCL kernels. No commit, stage, push, or PR. No physical measurement.
ASCII only.

Stop for human review.
