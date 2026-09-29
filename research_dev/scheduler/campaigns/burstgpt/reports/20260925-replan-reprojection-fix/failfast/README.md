# Fail fast on the first request lifecycle failure (2026-09-25)

Follow-up to `../PROGRESS.md` section 6 ("Not fixed: late detection of lifecycle failures"). No hardware or adb
was used, and the deploy is NOT synced. Nothing is committed. The change is merged into main (09:11 UTC).

## Problem

`CanonicalArrivalCoordinator` checked the lifecycle futures only in `drain()`, and the runner calls `drain()`
only after the last arrival. A failed request therefore left the arm running until the trace ended:

| run | lifecycle failed | noticed | rig time lost |
| --- | --- | --- | --- |
| 09-25 OP15 all-on `longtail_v1` | 007 at 431.4 s (decision record 29) | drain after arrival 027 (1,580.2 s), abort at 1,582.1 s | about 1,150 s |
| 09-23 run-treatment-4 | 003 at 330.9 s | at drain | about 27 min |

## Change (`FAILFAST.diff`, 5 files, base = main at 08:53 UTC)

`adapters/coordinator.py`:
- New keyword `fail_fast: bool = True` (the default). It must be a real bool.
- Every lifecycle future gets a done callback. The callback sets an event when the lifecycle raised.
- `wait_for_arrival()` checks the event on every tick. With fail-fast on, the 10 ms tick sleeps on the event,
  so a failure wakes the waiter at once. `submit()` checks the event before it schedules the arrival.
- A recorded failure goes through the same abort as `drain()`. Its failure branch moved verbatim into
  `_raise_after_abort`, which both paths call:
  - the primary is the first failed lifecycle in submission order;
  - pending tickets are cancelled with `arrival_coordinator_abort`;
  - the other lifecycles are awaited, and their errors are added as cleanup notes;
  - then the primary exception itself is raised.
- The exception gets one extra note: `arrival coordinator stopped arrivals at <t> us: lifecycle of <id> failed`.
  A failed coordinator keeps refusing later arrivals with the same exception.
- `fail_fast=False` restores the old behaviour exactly: no callback, a plain `time.sleep` tick, and failures
  surface only in `drain()`.

`campaigns/burstgpt/arguments.py` + `runner.py`:
- New flag `--lifecycle-failure-mode {fail-fast,drain}`, default `fail-fast`. The runner passes it as `fail_fast`.
- `launch.py` and the campaign config do not expose the flag, so every launched campaign gets the default.

**Unchanged failure path.** The exception now leaves `_submit_arrivals` instead of `drain()`, and it reaches
the same handler in `main()`:
- `FAILURE.json` keeps its schema. Its `error` is the lifecycle's own message; the note is in `traceback`.
- The `FAILURE_*` dumps are written as before.
- `_close_rig` runs as before: `coordinator.close(wait=False)`, `rig.close(require_phone_execution=False)` (phone
  restore), and `CLEANUP_FAILURE.json`.

The only difference in the failure artifacts: arrivals after the failure are never submitted, so the failure
decision log ends at the failure and its abort. `RESULT.json` is unchanged.

The replay tools (`../tools/`) drive the scheduler directly and never use the coordinator, so they are unaffected.

## Tests

| test | base (main before) | root / main after |
| --- | --- | --- |
| `test_arrival_coordinator.py::ArrivalFailFastTests::test_lifecycle_failure_aborts_before_the_next_arrival_is_submitted`: real scheduler, the backend fails arrival-1's lifecycle, arrivals 2-3 are due at 3.0-3.05 s | **FAIL** `PhysicalAdapterError not raised`: base waited and submitted arrivals 2 and 3 | ok: raised before 3 s; only arrivals 0-1 are in the decision log; 1 is FAILED; the note names arrival-1 |
| `...::test_submit_refuses_arrivals_after_a_recorded_lifecycle_failure`: direct `submit()` after a recorded failure, twice | **FAIL** (both arrivals scheduled) | ok: nothing scheduled, the note is added once |
| `...::test_drain_mode_keeps_serving_arrivals_until_drain`: `fail_fast=False`, plus the flag's type check | ERROR (no flag) | ok: the next arrival is scheduled; `drain()` raises the failure |
| `...::test_non_failing_decision_log_is_unchanged_by_fail_fast` (**digest guard**): pinned clock, 4 sequential arrivals, head `03c92218...` computed on base | default ok; `fail_fast=False` subtest ERROR (no flag) | ok in both modes |
| `test_campaign_inputs.py::test_runner_fails_fast_on_lifecycle_failures_unless_drain_is_requested`: flag default and choices, runner wiring (AST) | ERROR | ok |

- **Digest-guard clock.** `time.monotonic_ns` returns the arrival on the submitting thread and arrival + 0.5 s
  on lifecycle threads, and lease renewal is stubbed. The head was byte-identical three times on each of base,
  root (default) and root (`fail_fast=False`).
- **Stability.** The four coordinator tests passed 8 times in a row on root while the copy suite ran in parallel.
- **Base run log.** `NEW_TESTS_ON_BASE.log`: the new test files were run against the base copy.
- **pyflakes.** Clean on root and base: 445 files, the package minus reports and baselines. Also clean on the
  5 merged files in main.

## Suite and merge

- **Copy suite** (`SUITE_COPY_ROOT.log`, `run_all.py` in the root copy): 141 modules / 2,030 tests. There were two
  failures, both known:
  - the flaky `test_cached_synthetic_refinement_is_below_ten_milliseconds`, which passes when its module is
    rerun alone (39/39);
  - `test_resident_router_subset`, whose setUpClass needs `examples/layersplit/`, which the copy lacks.
  The S42 spike modules passed in the copy this time, because `spikes/` was symlinked to main's.
- **Merge.**
  - Main was byte-identical to the base copy at merge time (scheduler tree minus reports and baselines).
  - Backups of the 5 files, `talks.md` and the two-phone README are in `$SCRATCH/failfast/premerge-backup/`,
    with sha256 in `$SCRATCH/failfast/premerge-backup.sha256`.
  - `git apply --check`, then `git apply`, both with stdin closed. The merged files equal the root copy.
  - `git apply --check -R` passes, so `git apply -R FAILFAST.diff` reverts the merge.
- **Main suite after merge** (`SUITE_MAIN_AFTER.log`, `/usr/bin/python3 research_dev/scheduler/tests/run_all.py`):
  **141 modules / 2,031 tests, exit 0.** That is 2,026 + 5 new tests. Per module, the counts match the replan-fix
  run except `test_arrival_coordinator` (6 -> 10) and `test_campaign_inputs` (9 -> 10).

`$SCRATCH` = `/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad`.
The copies are `$SCRATCH/failfast/{base,root}`.

## Caveats

- **The abort still waits for in-flight lifecycles.** Like `drain()`, it waits for the other lifecycles to
  settle (cancelled or finished) before raising, so `FAILURE.json` appears after that wait.
  - `drain()` bounded that wait by its remaining deadline (the runner passes 14,400 s). The fail-fast path has
    no deadline.
  - This matches what happens next anyway: `close(wait=False)` joins the lifecycle threads without a bound.
- **At most the arrival in flight is still scheduled.** A failure that lands while an arrival's `submit()` is
  already scheduling it still lets that one arrival through. The next tick then stops the run.
- **The rig time saved is not measured.** No hardware ran. On the 09-25 timeline, the abort would have started
  at about 431 s instead of 1,582 s, plus the time the in-flight lifecycles take to settle.
- **The deploy is not synced.** The next rig runs need the parent's sync under the lock.
