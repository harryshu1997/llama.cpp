# dispatch-fixes progress (changes #2 work-conserving admission, #5 model affinity)

Layout: base/ = pristine copy of main research_dev/scheduler (minus reports/, baselines/); root/ = edited copy.
runs/dev2base/ = local copy of dev2base run artifacts (decision log, RESULT) for analysis.

## M0 (2026-09-24 ~10:50) assessment of the previous agent's partial work
- New `_internal/runtime_dispatch_policy.py` (RuntimeDispatchPolicy: work_conserving_admission, model_affinity,
  affinity_maximum_bypasses=3, affinity_maximum_wait_us=300 s; from_json/to_json; default all off).
- runtime_queue.py: policy in ctor; time-aware ordering `_conflicting_order_is_other_first` (one barrier + follower
  leases end before barrier leases start on shared lanes -> follower first); skip stale (cancelled) others when WC;
  preparation predecessors (follower waits for owner publication, not completion); precede_request_ids (affinity);
  flag off == base ordering (checked by reading).
- runtime_controller.py / admission.py: policy plumbing, bypass bookkeeping + stats, checkpoint/restore extended.
- NOT written yet: any caller in _unified (selection) that uses preparation predecessors / precede ids; the
  prepare-completion wake; plumbing campaign.json -> launch -> arguments -> runner -> scheduler; tests.

Root cause confirmed on dev2base decision log:
- every desktop plan uses cuda0/desktop-cpu (exclusive residency resources) -> residency_transition_barrier=True;
  queue conflicts are lane-based and time-agnostic -> all large-model entries ordered by arrival.
- 001 (arrived 2 s during 000's Gemma load) planned WITH its own load at 431 s (000's predicted end) because
  the live snapshot had Gemma cold; woken only at 000 completion (161 s).
- 005 (Gemma, 109 s) planned immediately on free lane 1 (no transition), but causally behind 001..004
  (older barrier entries whose prepare leases cover all lanes at 431..1017 s) -> ran at 565 s.

## M1 (~12:10) redesign + core implemented in root (previous agent's 3 files reset to base; backups prev_*.bak)
Dropped: preparation predecessors, stale-skip. Kept: policy module (affinity now REQUIRES WC), time-aware
ordering, precedence edges (new admissions only).
Core pieces (all gated by policy; flag off == base, verified by identical simulation log):
- resources.py `_runtime_residency_order_barrier`: WC -> barrier only for residency-changing plans
  (source!=target or evictions on exclusive devices).
- runtime_queue.py: `_other_dispatches_first` (+`_frees_lanes_before`: live QUEUED barrier -> lane windows;
  cancelled barrier -> bound = reserved end of its ACTIVE predecessors); precede ids in admit/bind;
  `_promote_deferred_behind_published_work` at prepare completion; early-completion frontier also takes
  non-follower same-residency queued work (`_backfills_early_capacity`) + promotes such DEFERRED replans;
  `dispatch_order_view()`; snapshot adds dispatch_policy only when enabled.
- runtime_controller(+ops/dispatch.py, admission.py): configure_dispatch_policy, dispatch_precedence ctx
  manager consumed by admit, bypass counts/notes/stats (checkpointed), displace_queued_attempts.
- observations.py: WC wake for queued attempts whose transitions are already realized (executor published,
  model hot, evictions gone) and whose predecessors are running.
- affinity.py + selection.py `_submit_with_model_affinity`: displacement at arrival, full re-resolution in a
  rollback transaction, accepted only if start earlier and no residency change; fairness bound.
- runtime_requests.py: configure_runtime_dispatch_policy, runtime_dispatch_policy_state, decision-log note.
Tools: tools/replay_common.py, tools/simulate_dispatch.py (discrete-event replay through the real scheduler;
base sim reproduces dev2base: makespan 1147 s vs 1149 s real, same 6 loads).
Sim results dev2 (sim/*.json): base/off 1147 s 6 loads; wc 959 s 5 loads; aff 663 s 3 loads.
TODO: plumbing (campaign/launch/arguments/runner + RESULT), unit tests, full test run, estimates, README, diff.

## M2 (~13:30) fixes found by simulation + tests + estimates
- Barrier definition corrected: ANY transition on an exclusive device (incl. hot->hot publication of a model only
  projected resident) is a residency change; only transition-free plans are "keepers" (else Gemma 'publication'
  plans overtook Qwen ones by dispatch key -> extra switches on long-tail).
- Deferred promotion at publication also calls _promote_deferred_frontier (followers whose edge to the owner was
  dropped by rebind); counted as published_work_promotions.
- Stale replan receipt bug in priority compaction (follower ticket REPLAN_REQUIRED with old queue generation ->
  REPLAN_WAKE_ROLLED_BACK forever -> runner retries exhausted): WC-gated refresh_replan_receipt in replan.py.
- tests/test_dispatch_policy.py: 22 tests, all pass on root; on base (+ read-only API shim tools/run_fail_before.py)
  15 fail/error, 7 pass (the 7 are invariants/flag-off/determinism) -> testruns/fail_before.log.
- Full suite: base = all OK except test_resident_router_subset (C++ include, env); root run (pre-final) same.
  Needs rerun after final changes. Trees need reports/ symlinks (done) + spikes symlink (done).
- Sims (sim/*.json; tools/summarize_sim.py; tools/compare_real.py):
  dev2: base 1147 s 5 large loads; wc 959 s 4; aff(any bound) 663 s 2.
  long-tail v1 baseline arm (ltbase): base 4835 s 12 loads (real 4901 s, same order); wc 4156 s 12;
  aff 3/300s 4342 s 12 (too tight); 3/900 4088; 5/1800 3966; 10/900 3364 6; 10/1200 = 10/1800 = unbounded 3172 s 5.
- Estimates (tools/estimate.py = accounting hypotheticals.py replay of simulated orders):
  long-tail none: base 4761s/509.6kJ, wc 4066/421.4, aff10/1200 3141/349.0; current phone: 4682/402.8 ->
  4204/369.9 (wc) -> 3334/326.5 (aff); coherent: 4557/379.6 -> 3916/315.7 -> 3050/264.4.
  dev2 none: 1256/101.5 -> 1057/86.3 (wc) -> 719/63.9 (aff).
TODO: defaults 10 bypasses/1200 s? final full test run; rebase on CURRENT main (reprovision + two-phone merged);
diff + README.

## M3 (~14:00) defaults, README section, rebase onto CURRENT main
- Defaults now affinity_maximum_bypasses=10, affinity_maximum_wait_us=1200 s (sweep on long-tail; 3/300 s is worse
  than WC alone there). Scheduler README gained "Dispatch ordering" subsection.
- own.patch = base->root for files.txt (18 files). Rebased: newbase = snapshot of main at ~11:56 (includes
  reprovision #4 + two-phone merges); newroot = newbase + own.patch (campaign.py 2 hunks resolved by hand next to
  phone_resident_model_reprovisioning). newroot: pyflakes clean, test_dispatch_policy 22/22 OK; fail-before on
  newbase (+shim) 15 fail/error, 7 pass (invariants) -> testruns/fail_before_newbase.log.
- Running: full suites newroot/newbase (testruns/new*_all.*), final sims tools/run_final_sims.sh -> sim/final/.
NEXT: check results, write README.md, produce DISPATCH_FIXES.diff vs CURRENT main + git apply --check.

## M4 (~14:40) DONE
- newroot full suite: all OK except test_resident_router_subset (env, same on newbase) and the known timing-flaky
  refinement test (flakes on both trees; 60-iteration benchmark equal 8.5-8.75 ms).
- Final sims on rebased trees (sim/final): flag off identical to newbase for dev2 and long-tail;
  dev2 1147 -> 959 (wc) -> 663 s (aff); long-tail 4835 -> 4156 (wc) -> 3172 s (aff default 10/1200), 3/300 s 4342.
- DISPATCH_FIXES.diff generated against CURRENT main (18 files, +1837/-27); git apply --check OK from repo root;
  applying it to a fresh main copy reproduces newroot exactly. README.md written.
