# Change #4 (phone re-provisioning per resident desktop model) - progress

## M0 2026-09-24: state assessment of the previous agent's work

- `base/` == current main `research_dev/scheduler` (only a new report dir differs upstream).
- Wiring (`patch_root.py`) is applied in `root/`: config knob + manifest field + campaign
  plumbing, `common.py` fields, portfolio/economics hooks, mixin methods, dispatch + release
  triggers in `runtime_requests.py`. `reprovision.py` exists (582 lines, complete draft).
- Tests: 2 new files; 2 failures + 1 error on root:
  - portfolio release test: SimpleNamespace runtime controller lacks `checkpoint` (test double).
  - swap latency expected 8 s, got 1 (learned-rate prior not used / sample handling).
  - RAM cap test: staged first stage keeps an over-limit current (768 > 640) - test design issue.
- Problems found by reading the code:
  1. `phone_shards.py` edit changes stage benefits for the memory-cap arm (flag-off behaviour
     change) -> revert; build the single-session stage inside `reprovision.py`.
  2. Learned load rate sums per-session durations; sessions of one transition share the wall
     window -> group by layout generation (bytes / wall time). Uses `published_at_us`; use the
     trace clock `observed_at_us` instead.
  3. Release hook runs before `release_request`, so the completing request still pins its
     sessions (active_helper_references) -> move after release.
  4. Reprovision path bypasses the base `force` handling for unavailable sessions -> defer to
     the base selector when a current session is not ready.
  5. Any desktop commitment -> all sessions to one model; design wants a proportional split when
     two phone-capable models are served concurrently -> commitment is a set.
  6. In-use predicate differs from `phone_layout_transition_blockers` -> align.
- Measured from run-5 (desktop, read-only): SESSION_LOADING->VERIFIED 19.4/12.3/14.3/12.1 s for
  3.21/3.21/2.67/2.83 GB -> 205 MB/s aggregate; live phone limit 9.27-9.67 GB; session limit
  3,208,646,656 B; objective `queue_rough_compute_ops` (learning selector).
- Shards: Qwen 0-17 (HTP0 0-5, HTP1 6-11, HTP2 12-17) present on OP15; Gemma v2/gemma24 has
  0-23 (8/session); layers 24-25 missing.

## M1 2026-09-24: reprovision.py rewritten, hooks fixed, tests green on root / red on base

- `phone_shards.py` reverted to base (no shared-code change).
- `reprovision.py`: commitment = set (loading > executing > desktop hot > warm > last followed);
  modes FOLLOW / PROPORTIONAL / HOLD; staging picks the best *generated* single-session layout
  (memory-feasible in the intermediate state, no private helpers); in-use predicate mirrors
  `phone_layout_transition_blockers`; defers to the base selector on unavailable sessions;
  request-impact revalidation fail-closed; learned rate per layout generation (wall window).
- `portfolio.py`: `_phone_reprovision_demand` (identity when off), confirmation via the
  existing `force` path (`choice.force or choice.confirmed`).
- `runtime_requests.py`: release hook moved after `release_request`/`notify`.
- Default prior 200 MB/s (run-5 aggregate 205 MB/s).
- Tests: test_phone_resident_model_reprovision.py 28 OK; test_phone_reprovision_portfolio.py 6 OK.
  On base: portfolio 5/6 fail (knob-off passes by design), unit file ImportError.

## M2 2026-09-24: physical fail-closed wait, docs, estimates, diff

- `helper_preparation_ops/start.py`: for reprovision proposals with blockers, return
  `DEFERRED WAITING_FOR_HELPER_RELEASE` (event deduplicated) instead of draining; other
  proposals keep `_defer_preparation_for_blockers`.
- Learned rate: per-session latest LOADING restarts the window (retries).
- Scheduler README section "Re-provisioning for the desktop model".
- Estimates: `estimate/estimate_reprovision.py` (+ .json) via the report's hypotheticals.py.
- `REPROVISION.diff` (16 files) passes `git apply --check` on the main tree.
- Full suite run 1 (pre-final code): only test_resident_router_subset failed (copy-only).

## M3 2026-09-24: final verification and delivery

- Added a scheduler-level test: `wait_runtime_request` fires the dispatch hook only when configured
  (7 portfolio tests, 6 red on base; 30 unit tests).
- Final full suite on root: 102 files ok, only test_resident_router_subset (copy-only) fails;
  phone/runtime/campaign subset rerun after the last edit: all ok (known flaky timing test
  passed on rerun). pyflakes: 0 findings for the whole package (base also 0).
- Main tree re-diffed: only new report files differ from base; REPROVISION.diff passes
  `git apply --check`. README.md written.
