# dispatch-fixes2 progress (affinity follow-up to #2/#5)

Layout: base/ = pristine copy of CURRENT main research_dev/scheduler (minus __pycache__, reports/ and baselines/;
reports symlinked read-only), root/ = edited copy. runs/{dp,allon} = local read-only copies of the rig artifacts.

## M0 (2026-09-24 15:11) setup
- base/ and root/ copied from main (identical). Read dispatch-policy README, RIG_RESULTS_ALLON.
- NEXT: pull decision logs (read-only scp) for dev2baseDP and dev2allon; reconstruct 005/006 decisions.

## M1 (15:45) root cause reconstructed (decision logs + snapshots + replay)
- Rig artifacts copied read-only to rig/{dp,allon}; replay run dirs runs/{dp,allon} built (RUN_COMMAND minus the
  quoted --dispatch-policy-json; policy set by the tool).
- dp arm: 006 arrived 112.0 s while 002's Qwen load (dispatched 79.0, published 128.8) was in flight -> Qwen not hot
  / executor not ready -> `_model_is_resident` False -> affinity returns None before any refusal accounting (stats 0).
  006 attempt 0 = cold->hot Qwen load (barrier) ordered by arrival after 005 (Gemma load barrier, 109.6 s).
- At publication 129.2 s WC replanned 006 (residency_observation_changed) to a transition-free Qwen plan on lane 3
  (start 129.2), but `_rebind_causal_predecessors` keeps existing edges between still-conflicting entries: the
  attempt-0 arrival-order edge 006->005 survived (005 cancelled plan holds all cuda0 lanes). 006 sat QUEUED/DEFERRED
  behind 005 until 004 completed (273 s) -> 005 loaded Gemma -> 006 needed a reload (625 s).
- Replay of the dp arm (tools/simulate_dispatch.py --run dp, base tree, aff): 863 s vs 865 s real, same 6 loads,
  same 006 trace (QUEUED barrier False preds ['005'] at 128.8 s). So the replay tool is faithful; the earlier
  prediction differed only because it used dev2base (legacy) durations: 000 held Gemma until 161 s there, so 005's
  arrival (109 s) saw Gemma resident and displaced the queued Qwen switch. On the rig 000/001 finished at 73 s.
- allon arm: same arrival-during-load pattern (002 load 102.8-153.6 s); at 153.9 s 005/006/007 were replanned as
  priority-compaction followers of 003/004; 006's follower replan projected 005's QUEUED Gemma switch first -> a
  Qwen reload plan again (not transition-free).
- Decision: extend affinity to replans (a queued request replanned while its model is resident is treated like an
  arrival): displace not-started other-model changes it causally waits on, same bounds/refusals, pre-selection.

## M2 (16:20) fix implemented in root/, dp replay passes
- affinity.py: `model_affinity_replan_displacement` (replan of a resident-model request that causally waits on a
  not-started other-model residency change -> displace it + the work queued behind it; same bounds/refusals);
  closure/bounds shared with the arrival path via `_close_displacement` (arrival behaviour unchanged).
- replan_commit.py: `_replan_with_model_affinity` runs first in `_replan_automated_request_once` (any replan, incl.
  compaction followers): nested transaction {replan_queued_now(displaced); dispatch_precedence; prepare+execute
  replan with require_resident_plan=True (guard raises before commit if the plan changes residency)}; any error ->
  rollback, count refusal, ordinary replan on unchanged state.
- runtime_queue.admit: precedence also accepted on a REPLANNING re-admission (drop owner->displaced edges, add
  displaced->owner, then rebind).
- replan.py: a displaced replan (model_affinity_displaced) does not invalidate its sequence frontier. Found by the
  replay: without it 005's displaced replan deferred 006 (later sequence, now its predecessor) and 006's re-replan
  hit the memory horizon of 005's new Gemma reservation -> "frozen desktop control is not currently feasible" ->
  terminal FAILED (latent for arrival displacements too).
- dp replay root/aff: 624.5 s (was 863), loads G, L, Q, G (3 large vs 5); 1 displacement (005, 007), 0 refusals.
NEXT: flag-off/WC identity replays, dev2base/ltbase/allon replays, unit tests, full suite, pricing, diff.

## M3 (17:05) publication wake + replays
- Unit exploration showed a second shape (= the allon arm): at publication the waiting request's load plan is NOT
  obsolete (projection still has the QUEUED switch before it) and its predecessors are not all running, so nothing
  replans it. Added an affinity wake in observations.py: under model_affinity a QUEUED attempt whose transitions
  are realized by the live snapshot is replanned now when `model_affinity_replan_displacement(record_refusal=False)`
  says its replan may displace (pre-check, no stat change). Churn bound: observations are event-driven.
- Replays (tools/run_sims2.sh, sim/*.json, tools/summarize2.py): flag off and WC-only logs identical base vs root on
  dp, dev2base, allon, ltbase (ltbase WC flaky on BOTH trees: replay-tool wall-clock race, "physical execution
  receipt differs"; identical when both pass). aff: dp 863->624.5 s, loads 5->3; allon(desktop-baseline approx.
  with allon durations) 802->594.9 s, 5->3; dev2base unchanged 663 s (arrival path); ltbase 3172->3182 s (+0.3%),
  large loads 5->4, mean/max latency 910/2037 -> 746/1924 s, 13 displacements, 13 refusals (9 replan-path fairness
  bounds, 4 arrival no-gain; no exception fallbacks). aff replays deterministic (2 runs identical).
NEXT: unit tests, full suite, pricing, diff/README.

## M4 (17:40) tests + rebase
- tests/test_dispatch_policy.py: +7 tests (29 total). root: 29/29 OK. base + new test file: 7 new fail
  (6 FAIL, 1 ERROR), 22 old pass (fail_before.log). Ablation: dropping the replan.py exemption fails
  test_displaced_switch_replans_behind_without_deferring_the_resident_work and makes the dp replay FAIL 006
  ("frozen desktop control is not currently feasible").
- Full suite (tools/run_scheduler_tests.py, pre-rebase trees): root and base both all OK except
  test_resident_router_subset.py (needs examples/layersplit C++, env). Known flaky tests passed.
- Coordinator: #4 follow-ups merged into main (17 files). Rebased: base2 = fresh main copy, root2 = base2 +
  own.patch (clean; only README.md overlapped, applied with offset). The 8 .py files are byte-identical to root.
- DISPATCH_AFFINITY_FIX.diff (9 files, +478/-20) vs CURRENT main: git apply --check OK (stdin closed); applying to
  a copy of main reproduces root2 exactly.
NEXT: rerun dispatch/runtime tests + full suite on root2, fail-before on base2, replays on base2/root2, README.

## M5 (18:20) DONE
- root2 full suite: all OK except test_resident_router_subset.py (env, same on base2) and one run of the timing-flaky
  test_cached_synthetic_refinement_is_below_ten_milliseconds (10.6 ms under concurrent load; 3/3 OK in isolation on
  both trees). base2 full suite: all OK except test_resident_router_subset.py.
- Rebased replays (sim2/): root2 logs identical to pre-rebase root for every run/policy; off/WC identical to base2.
- Pricing (sim2/estimates_dev2.txt, sim2/estimates_lt.txt). README.md written.
- Delivered: affinity-followup/{DISPATCH_AFFINITY_FIX.diff, README.md, fail_before_base2.log, tools/, sim2/} in
  reports/20260924-dispatch-policy/; git apply --check from the report copy OK.
