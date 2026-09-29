# Progress: fail-fast lifecycle failures + two-phone README correction (2026-09-25, UTC)

No hardware, no adb, no commits. The deploy is not synced.

| time | step |
| --- | --- |
| 08:50 | Read `AGENTS.md`, `NEXT_AGENT_PROMPT_SCHEDULER.md` and `../PROGRESS.md`. Located the drain-only check in `adapters/coordinator.py` (`drain()`: `wait(FIRST_EXCEPTION)`, then `_abort_outstanding`). The runner's `_submit_arrivals` loops `wait_for_arrival` / `rig.snapshot` / `submit` and calls `drain()` only after the last arrival. |
| 08:53 | Created the copies `$SCRATCH/failfast/{base,root}` with rsync, excluding `reports/` and `baselines/`, which are symlinked to main. A first `cp -a` filled `/tmp` (28 GB of reports) and was removed. Main == base, verified with `diff -rq`. |
| 08:55 | Probes on base (`$SCRATCH/failfast/explore/`): a `RuntimeError` from `execute` fails the lifecycle with `physical_execution_control_failed`, and base keeps serving the later arrivals. With the real clock, the decision log is not deterministic (ACQUIRED times). A per-thread pinned `monotonic_ns` gives a stable head. |
| 08:58 | Implemented fail-fast in the root copy: the coordinator's `fail_fast` flag, done callback plus event, checks in `wait_for_arrival`/`submit`, and `drain()`'s failure branch shared as `_raise_after_abort`. Added the runner flag `--lifecycle-failure-mode`. |
| 09:00 | Tests: 4 in `test_arrival_coordinator.py` and 1 in `test_campaign_inputs.py`. The digest-guard head `03c92218...` was computed on base, and is byte-identical 3x on base, root default and root `fail_fast=False`. On base, the new tests give 4 failures and 2 errors (`NEW_TESTS_ON_BASE.log`). On root, all pass, and 8 repeats were all OK. pyflakes is clean on both trees (445 files). |
| 09:02-09:10 | Copy suite (`SUITE_COPY_ROOT.log`): 141 modules / 2,030 tests. The only failures are the known flaky 10 ms test (passes alone) and `test_resident_router_subset` (no `examples/` in the copy). |
| 09:11 | Merge into main: main == base, backups plus sha256 in `$SCRATCH/failfast/premerge-backup*`, `git apply --check` + `git apply` with stdin closed, files == root, `-R --check` OK. |
| 09:11-09:19 | Main suite (`SUITE_MAIN_AFTER.log`): **141 modules / 2,031 tests, exit 0**. |
| 09:12 | Task B: added a "Correction" paragraph under the status table of `../../20260925-two-phone-eval/README.md` and a pointer at its section 4 crash bullet. The primary failure was Qwen 007 at 431.4 s (record 29). 018 is Gemma and failed at 1,583 s only after the drain-time abort (records 130-143). 027 is the Qwen load. The paragraph notes that both fixes are merged. The original text is left as it was. |
| 09:25 | Wrote this directory (`FAILFAST.diff`, `README.md`, logs) and a `talks.md` entry. |

Noticed, not changed: `../../20260925-two-phone-eval/README.md` contains a duplicated copy of its status section,
prefixed `PHONE_POWER_## Status and headline` (just before section 4). It looks like an editing artifact from the
earlier session. It was left alone.
