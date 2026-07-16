# S10-V0-R-E2 plan: matched control/treatment timeline gate

Status: E2_MATCHED_TIMELINE_MECHANICS_PASS_MEASUREMENT_NOT_RUN.

## Why E2 is a separate spike

E1 ends at `TYPED_EVIDENCE_INTEGRITY_PASS_PHYSICAL_CLAIMS_BLOCKED`. It binds
per-device, per-route terms and ADDS them. A `SERVER_WALL` or `GPU_BOARD`
measurement is an AGGREGATE timeline of a whole boundary: feeding it back into an
additive solver double-counts shared power and attributes it to individual routes.
E1 therefore rejects every `MEASURED` instance, and that rejection is correct
rather than a gap to be patched.

E2 is the other shape. It compares two REALIZED timelines POST HOC:

    optimized server-only control   vs   Q-PIM treatment

at the same boundary, over the same closed work, with the same SLO outcomes. E2
v1 proves only aggregate accounting closure and exact equality of opaque workload
digests; it does not yet resolve a per-request work ledger or prove that lifecycle
work cannot move outside the paid window. Those are explicit blockers for the next
gate. E2 is not a solver and never becomes an input to one.

## Scope

Authorized in this checkpoint:

1. an instrument capability audit of the live host (inspection only);
2. versioned E2 records and schemas;
3. a deterministic integer integrator and conservative comparator;
4. adversarial verification; and
5. a negative import of the one existing A6000 trace.

NOT authorized, and not done: any physical energy experiment, PF1, C0-C5, a
runtime, a scheduler, or any change to E1 or to protected runtime code.

## The four labels, and the one that does not exist

    GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED
    SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED
    RELIEF_FAIL
    MEASUREMENT_INVALID

`SYSTEM_ENERGY_SAVING` is not a value in any E2 schema or enum. It is not
"disallowed by policy"; it is inexpressible. Phone, USB supply, charger, and
external-device energy stay `UNKNOWN` and are never zero. A positive server-side
result is a break-even BUDGET for those excluded terms, not a total-system saving.

## What the host can actually support

From INSTRUMENT_AUDIT.md, first-hand:

- NVML is the only usable instrument, and it is `GPU_BOARD` only. Two A6000
  boards exist, so a timeline must name its board(s). `power.draw` on Ampere is a
  1-second average with a vendor-stated +/-5 W accuracy.
- RAPL is unusable twice over: `energy_uj` is root-only (no sudo), and only
  `package-0`/`core` domains exist (no dram, no psys). It is a component counter
  and can never be a server wall.
- No BMC, no IPMI, no PDU, no external meter exists.

So `SERVER_RELIEF_PASS_TOTAL_ENERGY_BLOCKED` is currently UNREACHABLE on this
host: not unmeasured, but impossible without new hardware. It is also blocked by
the current `ServerWallCapability` v1 contract, whose proof is only a hashed
declaration and is deliberately rejected for `MEASURED` provenance. The strongest
future label this host's present instrumentation could support is
`GPU_BOARD_RELIEF_ONLY_TOTAL_ENERGY_BLOCKED`, after an all-pairs aggregate exists.

## Checkpoint status

- [x] CP0 reproduce, repair, and freeze E1. The final 45-file baseline passes 201
      tests (28 foundation + 173 evidence), 39 evidence negatives, 18 CLI
      negatives, 1187 differential comparisons with zero mismatches, and both
      frozen optima.
      `E1_PRE_REPAIR_MANIFEST.txt` preserves the reviewed 185-test snapshot;
      `E1_BASELINE_MANIFEST.txt` pins the repaired 45-file tree and is re-checked
      before and after the E2 suite.
- [x] CP1 instrument capability audit (`INSTRUMENT_AUDIT.md`).
- [x] CP2 versioned records and schemas (`schemas/`), one strict validator.
- [x] CP3 deterministic integer integrator and comparator (`src/`).
- [x] CP4 adversarial verification (`tests/test_e2.py`, `scripts/e2_negative.sh`).
- [x] CP5 negative import of the existing A6000 trace (`src/import_nvml_trace.py`).
- [ ] A physical matched A/B experiment. NOT authorized here, and blocked by the
      instrument audit for anything above `GPU_BOARD`.

## Deliberate limits

- A single pair is DIAGNOSTIC ONLY and emits `MEASUREMENT_INVALID` with
  `PAIR_ONLY_NO_AGGREGATE_CLAIM`. A physical label requires the complete
  predeclared repetition set and the frozen `SUM_ALL_PAIRS_V1` aggregate over
  every pair. That aggregate evaluator is deliberately NOT implemented in this
  checkpoint, so physical labels are unreachable by construction, not by promise.
- Raw and execution artifacts use the v2 cross-binding: one `run_nonce`, typed
  per-sample P-states, and one digest over the exact paid power/status intervals.
  Reformatting a trace or changing out-of-window padding cannot manufacture a
  second run.
- `RepetitionSet` v1 validates structure and global ID/digest non-reuse only. It
  does not resolve every referenced record or prove that the plan predated the
  runs. The checked-in fixture is shape-only beyond its first concrete pair.
- `MatchedComparison` v1 is diagnostic-only in both schema and runtime. Its
  builder derives the repetition-set and order bindings from an exact referenced
  pair; callers cannot provide those digests or a result label. Its validator
  resolves the source timelines and repetition set and recomputes every copied
  identity, energy, uncertainty, arithmetic, and reason field.
- Aggregate work counts do not prove identical realized work. The current
  workload, trace, and SLO digests are opaque; no resolved per-request terminal,
  model/output, or correctness ledger exists yet.
- Marker equality does not prove lifecycle closure. The current execution record
  has no initial/final cache, residency, thermal, or outstanding-action state, so
  uncharged prefetch/warm-up before the window or transfer/cleanup after it must
  be excluded by the next contract before any aggregate claim.
- E2 validates structure and arithmetic. It cannot detect a FABRICATED raw
  artifact: bytes that hash correctly and parse cleanly are accepted. The gate
  makes a lie recorded, digest-pinned, and attributable, not impossible.
- Synthetic timelines exercise mechanics and emit no physical result, whatever
  their arithmetic says.
- Gross energy only. No idle baseline is invented or subtracted.

## Next gate

Before collection, the next gate must bind a separately anchored pre-run plan,
append-only attempt ledger, per-request realized-work proof, and paid lifecycle
closure, then resolve every run in `SUM_ALL_PAIRS_V1`. A later controlled
GPU_BOARD A/B can be designed around the 1.64-1.77 Hz observed change rate and the
+/-5 W accuracy floor. Everything above `GPU_BOARD` also needs hardware that does
not exist on this host and a later wall-capability version that binds calibration,
topology, validity, and acquisition evidence.
