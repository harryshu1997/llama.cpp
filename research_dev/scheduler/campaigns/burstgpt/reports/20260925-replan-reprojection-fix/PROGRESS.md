# Replan fix: memory-rejected baseline in the replan loop (2026-09-25)

Task: fix the scheduler crash that aborted the OP15 all-on arm of `longtail_v1`
(`../20260925-two-phone-eval/README.md`, section 4), in an isolated copy, then merge into main.
No hardware was run. The rig and the phones were only read (`ssh zhihao@172.20.74.85`, read-only
copies of the run artifacts). The deploy is NOT synced. Nothing is committed.

Deliverable: `REPLAN_REPROJECTION_FIX.diff` (562 lines, 7 files, base = main at 08:10 UTC).

## 1. Root cause (decision-log evidence)

### 1.1 The run did not die at 1,583 s: request 007 failed at 431.4 s

Decision log `inputs-op15-lt1/run-eval/run/FAILURE_SCHEDULER_DECISION_LOG.json` (144 records;
compact timeline in `data/op15_lt1_decision_timeline.txt`):

| rec | t (s) | event | request (model) | ticket window (s) | note |
| ---: | ---: | --- | --- | --- | --- |
| 23 | 430.031 | DECISION | 007 (Qwen, hot) | 489.4-769.8 | `MODEL_AFFINITY_DISPLACEMENT`: 007 displaces the queued Gemma switch 004 |
| 24 | 430.976 | REPLAN | 004 (Gemma, cold load) | 769.8-1,433.4 | displaced behind 007 |
| 27 | 431.149 | COMPLETED | 005 (Qwen) | | early -> 007 woken `capacity_released_early`, its lease cancelled |
| 25/26 | 431.18-431.38 | REPLAN, ACQUIRED | 006 (Qwen) | 431.2-667.2 | |
| 28 | 431.405 | REPLAN | 004 (Gemma, cold load) | **667.2**-1,329.9 | placed while 007 held nothing |
| 29 | 431.406 | **FAILED** | **007** (Qwen) | | `scheduler_replan_failed:UnifiedScheduleError:qualified desktop baseline is not available: MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram` |
| 129 | 1,580.201 | DECISION | 027 (Qwen, cold load) | 2,081.0-2,526.1 | last arrival of the trace |
| 130-134, 136-143 | 1,582.055 | CANCELLED | 004, 01, 014, 015, 016, 021-026, 02, 027 | | `failure_reason: arrival_coordinator_abort` |
| 135 | 1,583.085 | FAILED | 018 (Gemma, hot) | | same message as 007 |

- 007's lifecycle thread raised at 431.4 s, but `CanonicalArrivalCoordinator` only looks at lifecycle
  futures in `drain()`, which starts after the last arrival (027, 1,580.2 s).
  `wait(FIRST_EXCEPTION)` then returned at once with 007's exception, and `_abort_outstanding`
  cancelled everything at 1,582.055 s (the CANCELLED rows).
- The cancellations woke 018 (`predecessor_cancelled`, snapshot 5830 at 1,583.029 s).
  - Its replan hit the same bug.
  - Its error was swallowed as cleanup: 018 was already FAILED when the abort loop reached it.
- `FAILURE.json` has the same message and call path for both, so the two-phone README attributed the
  crash to 018 at 1,583 s. 018 is a Gemma request (the README calls it Qwen); 027 is the Qwen load.
- The arm therefore lost 007 at 431 s and ran 1,150 s more before the drain noticed.

### 1.2 Mechanism: a causal dependent's queued load inside the replanned baseline's window

At 007's replan (snapshots 1184-1197 of the run):

- `causal_predecessors[004] = [003, 006, 007]`: after the affinity displacement, 004 depends on 007.
  004 still has the earlier sequence (it arrived at 248 s).
- 004 is not a priority-compaction follower of 007, because followers must have a later sequence.
  So `replan_automated_request` took `_replan_priority_compaction_without_followers`, and
  `_execute_automated_replan` ran without compaction. The rig traceback shows this path.
- The candidates were generated. The only exclusive-replacement reservations in the ledger then belong
  to 007's causal dependents:
  - 004: Gemma load from 667.2 s;
  - 01: llama load from 1,329.9 s, which depends on 004.

  007's hot slot (about 280 s of service) therefore ran into 004's replacement window. This is
  inferred: the candidates of a failed replan are not logged.
  - The memory ledger rejects the baseline: `MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram`.
  - No phone route was admissible either.
  - `_pick_objective_candidate` raised.
- The loop already had the right remedy, but applied it too late. It defers causal dependents
  reserved before the new attempt's start (`defer_causal_dependents_before`, the 09-23 fix-2), and it
  does so only after selection has succeeded. Selection raised first.
- 004 cannot dispatch before 007 completes (`_causal_ready`), so its reservation was stale.

018 at 1,583.029 s has the same shape:
- In snapshot 5830 every ticket except 027 is CANCELLED, FAILED or lease-cancelled.
- 027 (QUEUED, Qwen cold load from 2,081.0 s) is the only reservation holder.
- `causal_predecessors[027]` contains 018.

### 1.3 Why an arrival-style re-projection alone would not have fixed it

The replay diagnostics (`data/REPLAY_FIX_EVENTS.txt`, `tools/debug_replan_hook.py`) print three
things at the rejected baseline:
- every ledger reservation that overlaps the baseline window;
- whether the owner of that reservation is a causal dependent;
- whether re-projecting residency at the baseline's start changes the snapshot.

In both replay crashes, the only overlapping owner was a queued causal dependent: 027 for 016, and 022
for 009. Without it the baseline is admitted.

Re-projection does change the snapshot, because the dependent's load starts before the baseline.
But it would plan the request as a reload behind its own dependent, and that dependent cannot dispatch
until the request completes. The result is causally inverted: an idle GPU and a wasted reload. So the
fix defers the dependents first. It re-projects residency only when no queued dependent precedes the
projection point.

## 2. Offline reproduction

- **An exact replay of the rig timeline is not feasible.** The phone-assisted service times,
  adaptive-decode learning and phone-helper events cannot be rebuilt from the artifacts, so the
  replayed costs diverge from the first decisions on.
- **Instead, the recorded run itself was replayed** through the real scheduler, built by the
  runner's `_build_scheduler` from this run's own arguments:
  - arrivals at the recorded times;
  - the recorded per-request execution times and load times;
  - WC + affinity, as on the rig.
  The tool is `tools/simulate_dispatch.py`, the 09-24 dispatch-policy replay. I made two changes:
  - a `--selection-mode` option;
  - `replay_common.pinned_clock`: `time.monotonic_ns` is pinned to the simulated time during
    dispatch probes. Before this, receipts carried wall-clock jitter, which caused the "physical
    execution receipt differs" race that the 09-24 agent reported as "ltbase WC flaky".
- **Results** (`data/REPLAY_NEUTRALITY.txt`: 5 recorded runs x 3 policies x 2 selection modes, base
  vs root):

| replay | base (main) | root (fix) |
| --- | --- | --- |
| this run (`op15lt1`), aff, energy-aware | crash at 1,585.9 s, 016 (Gemma) replan: `qualified desktop baseline is not available: MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram`; overlapping owner 027 (queued Qwen load, dependent) | 31/31; 027 deferred once, then replanned behind the Gemma work |
| same, desktop-baseline mode | crash, same point: `frozen desktop control is not currently feasible` | 31/31 |
| 09-24 longtail desktop run (`ltbase`), wc, both modes | crash at 1,017.2 s, 009 (Qwen) replan; owner 022 (queued Gemma load, dependent) | 31/31 |
| the other 26 run/policy/mode cells | complete | **replay logs byte-identical to base** (same makespans, loads, decisions) |

- **The replayed crash is the bug class of the rig's 018 failure.** A follower of a finished load
  replans a hot baseline behind the queued cold load of the other model, which arrived at 1,580.2 s.
- **The rig's primary path (007) is reproduced as a unit test** (section 4). There, the dependent is
  an affinity-displaced switch with an earlier sequence, and the replan is `capacity_released_early`
  without followers.
- **Determinism.**
  - Uncontended energy-aware replays repeated three times are byte-identical:
    - `op15lt1` root aff;
    - `op15lt1` base and root wc;
    - `ltbase` base aff.
  - The desktop-baseline reruns of the fix cases match their first run: `ltbase` wc and `op15lt1` aff.
  - Two energy-aware + aff replays run under 60-way CPU contention diverged from the uncontended ones
    at arrival predictions (from 661 s), with the same makespans. This happened on base as well and
    before any fix event, so it is a pre-existing timing sensitivity of the energy-aware path, not
    the fix.
  - The fix path itself is covered by a pinned-clock decision-log determinism test.

## 3. Fix (`REPLAN_REPROJECTION_FIX.diff`)

`_unified/automated_requests_ops/replan_commit.py`:
- New `_clear_rejected_replan_baseline`. `_execute_automated_replan` calls it right after the memory
  admission of each candidate set, before selection. When the baseline is memory-rejected for any
  reason other than `RESOURCE_CALENDAR_CURRENT`:
  1. **Defer dependents** (not under priority compaction). It defers queued causal dependents
     reserved before the end of the rejected memory window, which is the window the ledger checked.
     It reuses the existing `defer_causal_dependents_before`: the same deferral that the loop
     already applies after selection with the attempt's start. If anything was deferred, the
     candidates are regenerated.
  2. **Otherwise re-project residency** at the baseline's earliest start. That start is computed from
     the live barriers, the plan's own barriers and the causal observation time, as
     `selection._reproject_rejected_baseline` does, and is never before the end of the predecessors
     the replan waits for. The projection works from the replan's input snapshot, not from the
     already projected `context.snapshot`, so no transition is applied twice. It is skipped when a
     queued dependent is reserved before that start. If the snapshot changes: the live barriers are
     updated, `context.snapshot` is replaced, and the candidates are regenerated.
  3. **Otherwise nothing changes.** Selection runs the same checks and still raises
     `qualified desktop baseline is not available: <code>` on a real infeasibility (fail-closed).
- **One bounded outer loop.** The existing loop's budget went from `len(current_tickets()) + 1` to
  `+ 2`. It covers both kinds of retry: each deferral retires at least one queued dependent, and each
  re-projection needs a changed snapshot. On exhaustion it raises
  `runtime replan residency planning did not converge` (was `... dependent deferral ...`; no caller
  matches on the text).
- The post-selection preview folds in the live barriers of a re-projection. They are empty otherwise,
  so behaviour is unchanged.

Other files:
- `replan.py` / `common.py`: `_AutomatedReplanPreparation` carries `observed_snapshot` (the replan's
  input snapshot) and `projection_request_ids` (the predecessors projected into `context.snapshot`).
- `_internal/runtime_controller.py` + `runtime_controller_ops/replan.py`: read-only
  `queued_causal_dependents(request_id, before_us)`, a wrapper over the existing queue query.

Unchanged: the arrival path, priority compaction, followers, the ledger, the calendar and every
selection rule. The fix acts only where the old code raised, or where a memory-rejected baseline
forced another route.

## 4. Tests

New tests, fail on base (main) and pass on root:

| test | base | root |
| --- | --- | --- |
| `test_dispatch_policy.py::test_early_capacity_replan_defers_the_displaced_switch_in_its_window`: rig path of 007 (affinity-displaced switch, earlier sequence, `capacity_released_early`, no followers) | ERROR `qualified desktop baseline is not available: MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory` | ok |
| `test_automated_runtime_residency.py::test_replan_defers_dependent_replacement_inside_baseline_window`: two-lane GPU, late wake runs the hot root into its dependent's queued load; the dependent is then replanned behind the root | ERROR (same message) | ok |
| `...::test_replan_dependent_deferral_decision_log_is_deterministic`: pinned clock, two fresh schedulers, equal decision-log head hashes | ERROR (same message) | ok |
| `...::test_replan_reprojects_baseline_behind_unprojected_replacement`: re-projection branch (predecessor projection hidden, as the 09-23 arrival test hides the epoch); planned as a reload after the queued load | ERROR (same message) | ok |
| `...::test_compaction_replan_keeps_dependents_and_fails_closed`: under priority compaction nothing is deferred and it still raises with the reason | ok (unchanged behaviour) | ok |

The test probe (`tools/debug_replan_hook.install_fix_probe`) confirms which branch each test takes: deferral for the
first three, re-projection for the fourth.

- pyflakes: clean on root and base (445 files, package minus reports), and on the merged main files.

## 5. Suite, merge (08:10-08:43 UTC)

- **Full suite in the copy** (`tests/run_all.py`, 137 modules, 1,972 tests ran):
  - Every `research_dev/scheduler/tests` module passes, with two exceptions:
    - the known-flaky `test_cached_synthetic_refinement_is_below_ten_milliseconds`, which passes when
      its module is rerun alone (39/39);
    - `test_resident_router_subset`, a known copy failure (setUpClass).
  - The S42 spike modules error with `FileNotFoundError`: the copy's symlinked `spikes/` resolves
    to a scratch copy without the sibling spike data (`data/SUITE_COPY_FAILED_MODULES.txt`). The
    same error appears on the base copy (`test_mixed_model_trace`: errors=1 on both), so it is the
    environment, not the fix. The main-tree run below covers these modules.
- **Merge into main.**
  - Main was byte-identical to the base copy for the whole scheduler tree (minus reports and
    baselines) at merge time.
  - Pre-merge backups of the 7 touched files and `talks.md`:
    `$SCRATCH/replanfix/premerge-backup/` (sha256 list: `$SCRATCH/replanfix/premerge-backup.sha256`).
  - `git apply --check` then `git apply`, both with stdin closed. Afterwards the 7 files equal the
    tested root copy.
  - `git apply --check -R` passes, so the merge can be reverted with `git apply -R`.
- **Full suite on main**, `/usr/bin/python3 research_dev/scheduler/tests/run_all.py`
  (`SUITE_MAIN_AFTER.log`):
  - **141 modules / 2,026 tests** (2,021 before + 5 new).
  - The only failure is the known-flaky
    `test_cached_synthetic_refinement_is_below_ten_milliseconds` (10.25 ms against a 10 ms bound).
  - Its module rerun alone passes twice (39/39).
  - After the suite, one test helper in `test_automated_runtime_residency.py` was made explicit
    (`compaction=` instead of passing kwargs through). That module was rerun on main (52/52) and the
    diff regenerated: `git apply --check` passes against the pre-merge files and `-R` against main.
- **Deploy NOT synced** (per the task). Nothing on the rig was written.

`$SCRATCH` = `/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad`.
The isolated copies are `$SCRATCH/replanfix/{base,root}`. The replay run dirs are
`$SCRATCH/replanfix/runs/` (`op15lt1` built from the rig artifacts; the others are symlinks to
`$SCRATCH/dispatch-fixes2/runs/`). The replay outputs are `$SCRATCH/replanfix/sim/`.

Re-run a replay:
```sh
cd $SCRATCH/replanfix/tools   # tools/ here are copies of the ones in this report dir
PYTHONPATH=$SCRATCH/evidence-fixes/shared/gguf-py python3 simulate_dispatch.py ../root aff \
  --run op15lt1 --selection-mode energy-aware --json OUT.json   # SIM_DEBUG_REPLAN=1 prints the diagnostics
```

## 6. Caveats

- **The deferral defers more than the conflicting dependent.** It covers every queued causal dependent
  reserved before the rejected window ends, not only the one whose reservation conflicts. Those
  dependents cannot dispatch before the request completes anyway, and the existing post-selection rule
  defers the ones before the start. It runs only on the memory-rejected path: all 26 replay cells
  without a rejection are byte-identical.
- **The re-projection branch has only a white-box test.** The test hides the predecessor projection.
  No replay or rig run hit that branch: every observed instance was a dependent.
- **Under priority compaction nothing new happens.** No deferral is made, which protects the follower
  order. A rejection there still fails closed, as before.
- **A timing field is still wrong.** The replan timing record still reports
  `residency_fixed_point_iterations: 1`. That is diagnostic only and not in the decision log.
- **Not fixed: late detection of lifecycle failures.** The coordinator surfaces a lifecycle failure
  only at `drain()`, so 007's failure went unnoticed for 1,150 s of rig time. Checking the futures
  while arrivals are submitted, and failing fast, would stop such arms early. It needs a decision.
- **The deploy is not synced.** The next rig runs need the parent's sync under the lock.
