# S10-V0-R results

## VERDICT: TEMPORAL_FOUNDATION_PASS

The temporal oracle and a structurally independent optimum verifier agree on every
frozen fixture and on all 1187 compared generated cases, and every adversarial,
mutation, determinism, and CLI gate passes.

TEMPORAL_FOUNDATION_PASS does NOT authorize C0-C5. Typed evidence binding is the
next separate gate. Nothing was committed or pushed.

The historical S10-V0 `FAIL` remains invalid/inconclusive. Its oracle did not search
the schedule space required by the plan, and its checker and verdict path were fail
open. No opportunity, mechanism, energy, or system verdict is authorized.

## What changed in this checkpoint

The two exactness defects that blocked the foundation are closed, and the optimality
claim is now independently proved instead of asserted from search counters.

1. **Temporal enumeration (CP1/CP2).** The solver has two exact modes, chosen by the
   instance and never by convenience. The frozen contract is in `PLAN.md`
   ("Exact temporal domain").
   - EARLIEST mode (`wake_us == idle_entry_us == transition_nj == 0` and every
     `output_bytes == 0`): earliest-start is provably exact, because server energy
     reduces to `p8*H + (p0-p8)*sum(server durations)` independent of start times,
     phone energy never depends on start times, the activation peak is 0, and
     earliest-start simultaneously minimises every finish time of a fixed order.
   - TEMPORAL mode (anything else): every legal integer start time is enumerated in
     addition to routes, compatible batch partitions, and per-device action orders.
     Feasibility windows come from releases, DAG/lane precedence, the wake ramp and
     the HORIZON only -- never from deadlines, because TARDY is a legal outcome and a
     deadline-derived bound would silently discard feasible schedules.
2. **The frozen transition counterexample is now solved, not refused.** The solver
   selects the delayed placement at **147250000 nJ** instead of the earliest
   **162000000 nJ**, at identical zero-miss/zero-lateness outcomes:
   `n0 [850,950]`, `n1 [1000,1100]`, one merged P0 window `[800,1150]` rather than
   two. The signed certificate records `complete=true`; internal work counters are
   deliberately not part of the proof artifact.
3. **New frozen activation counterexample** (`fixtures/activation_delay_counterexample.json`):
   the earliest placement has activation peak **200 > bound 150** and is infeasible,
   while a delayed placement (`p1 [10,14]`) reaches peak **100**. The oracle finds it
   without weakening the producer-through-last-consumer allocation lifetime.
4. **Independent optimum verification (CP3).** `checker/reference.py` is a
   checker-owned exhaustive recomputation. Structural inspection shows that it
   imports only the standard library and has separate configuration, start-time,
   legality, and objective enumeration; it imports no oracle/checker scheduling
   code and performs no branch-and-bound.
   Default checker mode now **accepts an independently proven optimum** and
   **rejects a feasible-but-suboptimal certificate**. The certificate's
   `search.complete` marker is an assertion, never proof; extra search metadata is
   rejected.
   `--feasibility-only` stays visibly separate and always reports
   `optimality_verified=false`.

## Evidence

### Frozen fixtures (all three independently verified)

| fixture | mode | oracle optimum | independent verifier | agrees |
|---|---|---|---|---|
| `transition_delay_counterexample` | TEMPORAL | `[0,0,-2,147250000]` | `checker/reference.py` -> `(0,0,-2,147250000)` | yes |
| `activation_delay_counterexample` | TEMPORAL | `[0,0,-2,2974000]` | `checker/reference.py` -> `(0,0,-2,2974000)` | yes |
| `partial_partition` | EARLIEST | `[0,0,-3,439600000]` | `tests/slow_reference.py` batch reference | yes |

`partial_partition` has `horizon_us: 5000`, which needs about 19.6M start-time
combinations against the reference's 8M bound, so the checker's default mode fails
closed on it and certifies nothing. That is the contract's intended behaviour
("within frozen tiny bounds"), and its optimum is still independently proved by the
separately written batch reference, so no frozen fixture rests on the oracle's own
word.

### Differential corpus (CP4)

`tests/gen_cases.py` was frozen before results were read. 1200 deterministic seeds,
run as **four separate processes** under `PYTHONHASHSEED` 0, 1, 42, 12345:

```
slice 0   : compared 294, agreed_infeasible 6, out_of_domain 0
slice 300 : compared 298, agreed_infeasible 2, out_of_domain 0
slice 600 : compared 297, agreed_infeasible 3, out_of_domain 0
slice 900 : compared 298, agreed_infeasible 2, out_of_domain 0
TOTAL     : compared 1187, agreed_infeasible 13, out_of_domain 0, mismatches 0
```

Shapes cover unbatched and compatible-batch cases, multiple routes, DAG chains and a
fork/join diamond, nonzero wake/idle-entry/transition, and nonzero `output_bytes`
with a binding activation bound. Each compared case also has its oracle certificate
re-validated by the standalone checker. **Determinism**: the same slice re-run under
`PYTHONHASHSEED=99` and `PYTHONHASHSEED=random` reproduces a byte-identical digest
over every `(seed, objective, certificate_sha256)`.

### Required regressions and mutations (all present, all failing closed)

- 162000000 -> 147250000 transition-delay selection, including the exact placement
  and the single merged P0 window.
- Earliest-infeasible / delayed-feasible activation placement.
- Equal busy time with different gap energy (retained), plus wake/idle-entry
  **boundary equality**: windows that touch exactly at 250 merge into `[50,450]`;
  one microsecond later they split. Total active time is identical (400 us) on both
  sides, so the energy gap is exactly one `transition_nj` charge.
- Documented B&B rules R1/R2 verified against an unpruned search (`prune=False`) on
  the frozen activation fixture and generated cases; the independent reference
  uses no incumbent/objective pruning and agrees across the whole corpus.
- Horizon, release, deadline, precedence, lane-overlap, duration, batch-identity,
  activation-peak, terminal-DAG, energy, and objective mutations.
- Resealed feasible suboptimal schedule carrying the genuine optimum's completeness
  marker -> rejected with a stable `SUBOPTIMAL` diagnostic; the same schedule still
  passes `--feasibility-only`, which claims nothing.
- Forged search metadata and empty device/route identifiers -> rejected by schemas,
  semantic validation, or both.
- Differential coverage is non-vacuous: every requested seed must land in exactly
  one terminal bucket, skipped-domain must be zero for the frozen corpus, and paired
  unexpected RuntimeErrors are mismatches rather than agreed infeasibility.
- Incomplete search, state-cap exhaustion (`--max-states 10`), and out-of-domain
  temporal instances -> explicit error, never `complete=true`, never a best-so-far.
- Duplicate keys, NaN/Infinity, booleans in integer fields, unknown fields, wrong
  evidence scope, alternate server-kind devices, malformed profiles.
- 18 invalid CLI cases: every one exits nonzero with a stable `ORACLE_FAIL` /
  `CHECK_FAIL` prefix and **no traceback**.

## Commands, counts, exit codes

```
bash research_dev/spikes/s10_power_frontier_repair/scripts/run_tests.sh   # exit 0
  -> Ran 26 tests ... OK                      (was 19 at baseline)
  -> 3 instance schemas + 3 certificate schemas validated
  -> partial_partition: feasibility-only pass; default mode fails closed with the
     SPECIFIC "not independently verifiable inside the reference domain" reason
  -> transition fixture : optimality_verified=true, objective [0,0,-2,147250000]
  -> activation fixture : optimality_verified=true, objective [0,0,-2,2974000]
  -> suboptimal certificate rejected (SUBOPTIMAL), no traceback
  -> S10_V0R_CLI_NEGATIVE_PASS (18 cases)
  -> all 1200 differential seeds accounted for: 1187 compared, 13 jointly
     infeasible, 0 skipped, 0 mismatches; digests deterministic
  -> S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS
git diff --check                              # clean
ASCII scan of the repair directory            # all ASCII
find ... -name '__pycache__' -o -name '*.pyc' # 0 remaining
```

Baseline before editing: `run_tests.sh` -> 19 tests, exit 0, marker
`S10_V0R_CURRENT_MECHANICS_TESTS_PASS`. The marker is now
`S10_V0R_TEMPORAL_FOUNDATION_TESTS_PASS`.

## Changed and added files

Changed: `oracle/exact.py` (temporal mode, declared domain, documented B&B),
`checker/checker.py` (independent optimality proof; strict signed fields),
`schemas/{instance,certificate}.schema.json` (nonempty identifiers; no unverifiable
search counters),
`tests/test_foundation.py` (inverted the two refusal tests, added temporal and
mutation regressions), `scripts/run_tests.sh` (temporal CLI coverage, suboptimal
rejection, CLI negatives, 1200-case differential, determinism), `PLAN.md`
(frozen exact-temporal-domain contract; removed the now-contradictory tail),
`RESULTS.md` (this file).

Added: `checker/reference.py`, `fixtures/activation_delay_counterexample.json`,
`tests/gen_cases.py`, `tests/differential.py`, `tests/make_suboptimal.py`,
`scripts/cli_negative.sh`.

## Deviations and honest limits

- The exact temporal domain is deliberately far smaller than the schema maximum:
  `TEMPORAL_MAX_NODES=6`, `TEMPORAL_MAX_ACTIONS=6`, and an unpruned start-time
  product bound `TEMPORAL_MAX_WINDOW_PRODUCT=8000000`. The cumulative product is
  checked before recursion for each device order. Earlier layouts may already have
  been visited, but crossing the bound raises and emits no certificate.
- Default-mode optimality can only be certified inside `checker/reference.py`'s
  declared domain (`REFERENCE_MAX_ENUM=8000000`). `partial_partition` is outside it
  and is certified by no one in default mode, by design.
- EARLIEST mode is exact by a written proof plus differential tests against
  `tests/slow_reference.py`; it is not covered by start-time enumeration, because
  start times provably cannot change its objective.
- The causal helper is still only a prefix snapshot test, not a rolling C5 policy.
- `evidence.scope=MECHANICS_ONLY` remains a safety label, not typed measured
  evidence. Phone energy is a labelled profile input, not physical wall energy.

## Still not implemented or run

- Typed artifact bindings for route, power, correctness, thermal, and boundary
  inputs (the next separate gate).
- C0-C5 policies, PF1, CP-SAT, residency policy, a live scheduler.
- Real mixed-model traces, a measured second service route, new device
  measurements, and synchronized complete-wall energy.

## Integrity

- HEAD `933c722f6`, unchanged. No reset, checkout, revert, stage, commit, push, or PR.
- The historical `../s10_power_frontier/` tree was not intentionally edited and
  still contains 101 files. The earlier exact recursive-hash claim is withdrawn:
  no byte-exact hashing recipe and pre-run manifest were persisted, so that value
  cannot be independently reproduced from the tree.
- No edits to tools/server, llama-server, model graphs, KV internals,
  ggml_backend_sched, the phone-PIM protocol/runtime, or any HTP/OpenCL/CUDA kernel.
  No phones were used and no measurements were added.
- Pre-existing S6-S10 dirty files preserved; the only tracked file this task touches
  is `research_dev/talks.md` (top status block plus one newest-first entry).
- ASCII only; generated bytecode removed.

Stop for human review.
