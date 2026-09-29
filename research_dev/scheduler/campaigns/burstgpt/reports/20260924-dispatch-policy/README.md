# Scheduler changes #2 (work-conserving admission) and #5 (model affinity), 2026-09-24

Deliverable: `DISPATCH_FIXES.diff` (against the CURRENT main tree, i.e. after the #4 re-provisioning and the
two-phone merges; `git apply --check` from the repo root passes). Opt-in: with no `dispatch_policy` in
campaign.json nothing changes (verified: identical decision sequence in the dispatcher replay, full test suite).
No hardware was used; estimates come from a discrete-event replay of the recorded runs through the real
scheduler code plus the offload accounting's calibrated pass model (`hypotheticals.py`).

## 1. Why same-model requests serialized (verified on the dev2base decision log)

1. **Every desktop plan was a residency barrier.** `_runtime_residency_order_barrier` marked any plan that uses an
   exclusive residency resource (`cuda0`, `desktop-cpu`), i.e. every large-model plan, and the queue orders
   barrier pairs by arrival. Conflicts are lane-based and time-agnostic. Gemma 005 (arrived 109 s) was planned
   at 109 s on a free lane with no transition, but became a causal follower of 001..004 (older plans whose load
   leases cover all lanes at 431..1017 s); it was dispatched only at 492 s, by then with a Gemma reload (done at
   564 s). Qwen 006 then waited behind 005.
2. **A request arriving during a load plans its own load.** 001 (arrived with 000 at 1 s) got a hot->hot
   *publication* transition (Gemma executor not yet ready) whose prepare lease needs all lanes, so the calendar put
   it after 000's predicted end (431 s). When 000's load finished (98 s) the preparation-window commit deferred 001
   behind the running 000, and nothing woke it until 000 completed (161 s).
3. **Early completion only woke causal followers.** Work bound to a later lane (007 behind 005) stayed asleep when a
   different lane was freed early; predicted decode times are off by 0.2x..9.7x (section 6).

## 2. Change set (all behaviour gated by `RuntimeDispatchPolicy`; default off)

| file | change |
| --- | --- |
| `_internal/runtime_dispatch_policy.py` (new) | `RuntimeDispatchPolicy(work_conserving_admission, model_affinity, affinity_maximum_bypasses=10, affinity_maximum_wait_us=1_200_000_000)`, validation (affinity requires WC), `to_json/from_json`; `plan_changes_residency(transitions, exclusive_by_device)` = any transition that prepares an exclusive residency device (load, eviction, or publication of a model only projected resident) |
| `_unified/automated_selection_ops/resources.py` | WC: the queue barrier flag is `plan_changes_residency` instead of "uses an exclusive resource"; transition-free work on the resident model is a "keeper" |
| `_internal/runtime_queue.py` | policy (set only while empty); `_other_dispatches_first`: a keeper runs before a barrier when its lanes are free before the barrier can start (live QUEUED barrier: its lane windows; cancelled barrier awaiting replan: the reserved ends of its ACTIVE predecessors bound it; no running predecessor -> arrival order); barrier pairs keep arrival order; WC rebind uses the same rule (off: arrival order as before); `precede_request_ids` on new admissions (affinity edges); at preparation completion promote deferred replans whose predecessors are all running and published, plus the causal-ready deferred frontier (except `residency_projection_invalid` deferrals); early completion also offers freed lanes to QUEUED keepers that do not depend on the completed owner, and early replans that only wait on running work replan now; `dispatch_order_view()`, `policy_events()` (checkpointed); snapshot adds `dispatch_policy` only when enabled |
| `_internal/runtime_controller.py`, `runtime_controller_ops/dispatch.py` (new), `admission.py` | `configure_dispatch_policy` (only before the first admission), `dispatch_precedence(...)` context consumed by exactly one admission (edges + bypass accounting only after the queue accepted it), bypass counts / decision notes / statistics (checkpointed with the controller), `replan_queued_now`, `refresh_replan_receipt` |
| `_unified/automated_requests_ops/observations.py` | WC: a queued attempt whose transitions are already realized in the live snapshot (executor ready, model hot on every prepared device, evictions gone) and whose predecessors are all running is replanned now (`residency_observation_changed`, counted as `publication_replans`) |
| `_unified/automated_requests_ops/affinity.py` (new), `selection.py` | affinity at arrival: if the arrival's model is resident with a ready executor and a residency change of another model would run first (a QUEUED change reserved at/before the arrival's start, or a cancelled one not held back past the arrival by running predecessors), displace those changes and every attempt queued behind them (cancel reservations + memory, replan after the arrival), re-run the full selection inside a rollback transaction, and keep it only if the arrival now starts earlier on a transition-free plan; refused when any displaced other-model attempt reached the bypass or wait bound, holds a decode cohort or phone layout token, or another model's change is running/replanning |
| `_unified/automated_requests_ops/replan.py` | WC: priority-compaction followers whose ticket holds a stale REPLAN receipt are rebound to the current queue generation (see caveat 4) |
| `_unified/runtime_requests.py` | `configure_runtime_dispatch_policy`, `runtime_dispatch_policy_state`; decision-log `selected.dispatch_policy` note on displacing admissions |
| `__init__.py` | export `RuntimeDispatchPolicy`, `RuntimeDispatchPolicyError` |
| `configuration/campaign.py`, `campaigns/burstgpt/{launch,arguments,runner}.py` | campaign field `dispatch_policy` (validated; omitted from `to_json` when absent so existing resolved-configuration hashes are unchanged) -> `--dispatch-policy-json` -> `scheduler.configure_runtime_dispatch_policy`; RESULT.json gets `dispatch_policy` (policy, bypass counts, statistics) only when set. One small hunk per plumbing file, next to (not inside) the #4 field |
| `README.md` | "Dispatch ordering" subsection |
| `tests/test_dispatch_policy.py` (new) | 22 tests (section 4) |

Fail-closed properties kept: every admission and every displaced/woken request goes through the normal selection
(route certification, memory ledger preview/reserve, residency projection, exclusive-transition barrier, calendar,
KV/slot lanes); residency changes stay arrival-ordered among themselves and still wait for every authoritative
lease on the exclusive resource; decisions are deterministic (sorted iteration, no clocks), and every wake uses an
existing reason string except `model_affinity_displaced`.

## 3. How to enable (campaign.json)

```json
"dispatch_policy": {"work_conserving_admission": true}
```

WC plus affinity (recommended bounds shown; they are also the defaults):

```json
"dispatch_policy": {"work_conserving_admission": true, "model_affinity": true,
                    "affinity_maximum_bypasses": 10, "affinity_maximum_wait_us": 1200000000}
```

Rules: `model_affinity` requires `work_conserving_admission`; unknown keys, non-boolean flags and negative bounds are
rejected at resolve time. Absent field = legacy dispatcher. Statistics in RESULT.json `dispatch_policy.statistics`:
`affinity_displacements`, `affinity_displaced_attempts`, `affinity_refusals`, `publication_replans`,
`published_work_promotions`, `early_capacity_promotions`; `bypass_counts` per request.

## 4. Tests

`tests/test_dispatch_policy.py` (22): policy validation; queue order (keeper before a later residency change,
overlap keeps arrival order, residency changes keep arrival order, cancelled change bounded by running
predecessors, publication promotes deferred followers but not projection-invalid deferrals, early completion
offers freed lanes to non-follower keepers, precedence validation + checkpoint round trip); scheduler level on two
models sharing one exclusive two-lane GPU (flag off keeps the same-model arrival behind the queued switch; WC: the
same-model arrival is dispatchable immediately and the switch waits for it; the switch still waits for every
exclusive lease; a queued request joins at publication instead of after its predecessor; affinity admits the
resident model before a queued switch and logs the note; protection after the bypass bound and after the wait
bound; the displaced switch replans after the resident work; slot capacity never exceeded; identical decision logs
over two runs); campaign field validation and runner/argument plumbing.

- Pass after: 22/22 on the rebased tree.
- Fail before: against the current main tree with a read-only API shim that only adds the policy type, a no-op
  configure call and queue views (`tools/run_fail_before.py`): 15 fail/error, 7 pass; the 7 are the invariants
  that must hold on both (flag off, arrival order for residency changes and overlaps, determinism, slot capacity,
  policy value validation).
- Full suite, every `tests/test_*.py` in its own process (run_all semantics, S42 spikes excluded):
  base (current main snapshot) and patched tree both: everything OK except `test_resident_router_subset.py`
  (compiles `examples/layersplit/*.cpp`, not present in the copied tree; environment). The known timing-flaky
  `test_cached_synthetic_refinement_is_below_ten_milliseconds` (10 ms bound) failed once in the patched full run
  (10.7 ms, run concurrently with the unpatched suite and the replays) and flakes on both trees in isolation
  (unpatched 1 of 4, patched 1 of 3 reruns); a 60-iteration benchmark of the measured call gives 8.5-8.75 ms on
  both trees (the call does not reach the changed code). `test_kv_touch_occupies_exactly_the_cache_pages` passed.
- `pyflakes` on all 17 changed Python files: clean.

## 5. Effect estimate (no hardware)

Method: `tools/simulate_dispatch.py` replays a recorded run through the real scheduler (runner-built catalog and
models, recorded arrival times and request snapshots, synthesized residency snapshots per resident model, recorded
per-model load durations and per-request execution durations, lease renewal like the runner, replan retries like
the runner). Validation on the legacy dispatcher: dev2base 1147 s vs 1149 s real, same order and loads; long-tail
v1 baseline arm 4835 s vs 4901 s real, same service order and 12 large-model loads. Flag off on the patched tree
reproduces the unpatched log line for line. The simulated service orders and overlaps are then priced with the
accounting's calibrated pass model (`tools/estimate.py` -> `hypotheticals.Sim` replay mode; it reproduces the
measured long-tail arms within 2-5 %). Durations are the recorded solo durations (batching slowdown ignored in the
dispatcher replay; the pass model adds it back).

### dev_v2 (dev2base inputs, 9 requests)

Dispatcher replay (makespan = last completion; start = dispatch, includes the load for loaders):

| request | model | arrival s | real done s | legacy (sim) | WC | WC + affinity |
|---|---|---:|---:|---:|---:|---:|
| 000 | Gemma | 1 | 161 | 1-161 | 1-161 | 1-161 |
| 001 | Gemma | 2 | 178 | 161-178 | 98-115 | 98-115 |
| llama overlay 00 | Llama | 4 | 181 | 178-181 | 161-164 | 429-432 |
| 002 | Qwen | 79 | 350 | 181-349 | 164-332 | 432-600 |
| 003 | Qwen | 86 | 434 | 349-433 | 254-338 | 522-606 |
| 004 | Qwen | 99 | 492 | 349-490 | 254-395 | 522-663 |
| 005 | Gemma | 109 | 878 | 490-879 | 395-784 | 115-429 |
| 006 | Qwen | 112 | 1053 | 879-1054 | 784-959 | 522-613 |
| 007 | Gemma | 128 | 1149 | 1054-1147 | 128-147 | 161-180 |

| | legacy (sim) | WC | WC + affinity |
|---|---:|---:|---:|
| makespan s | 1147 (real 1149) | 959 (-16 %) | 663 (-42 %) |
| large-model loads | 5 | 4 | 2 |
| mean / max latency s | 510 / 1019 | 327 / 847 | 344 / 564 |
| requests decoded with a partner / alone while same-model work was queued | 2 / 4 of 8 | 6 / 0 | 8 / 0 |
| pass model, desktop only: s / host kJ | 1256 / 101.5 | 1057 / 86.3 (-16 % / -15 %) | 719 / 63.9 (-43 % / -37 %) |
| pass model, today's phone policy | 1251 / 85.1 | 1072 / 74.9 | 762 / 60.6 (-39 % / -29 %) |
| pass model, coherent coalesced phone (#1) | 1223 / 80.1 | 1032 / 68.3 | 705 / 50.7 (-42 % / -37 %) |

What changed: 001 joins at 000's publication (98 s instead of 161 s), 007 backfills the free Gemma slot at
128 s, 003/004 join 002's Qwen server when its load publishes (254 s instead of 349 s); with affinity Gemma 005
displaces the queued Qwen switches (one displacement: 002, 003, 004 and the Llama overlay bypassed once) and 006
joins the Qwen server at load completion, so the Gemma -> Qwen -> Gemma -> Qwen -> Gemma sequence becomes
Gemma -> (Llama) -> Qwen. dev_v2 prices use the long-tail calibration with dev_v2 shapes (prompt time from the
long-tail per-token rate), so the absolute pass-model numbers are ~9 % above the measured 1154 s / 96.75 kJ;
use the relative changes. Any bound between 3/300 s and unbounded gives the same dev_v2 result.

### Long-tail v1 (run-5 trace; dispatcher replayed on the desktop-only baseline arm)

| | legacy (sim) | WC | WC + affinity 10 / 1200 s (default) | WC + affinity 3 / 300 s |
|---|---:|---:|---:|---:|
| makespan s (dispatcher replay) | 4835 (real 4901) | 4156 (-14 %) | 3172 (-34 %) | 4342 (-10 %) |
| large-model loads | 12 | 12 | 5 | 12 |
| mean / max latency s | 1754 / 3363 | 1511 / 2684 | 900 / 2037 | 1628 / 2892 |
| mean / max queue wait s | 1562 / 3177 | 1318 / 2554 | 724 / 1831 | 1436 / 2792 |
| decoded with a partner / alone while same-model work was queued (of 28) | 12 / 14 | 23 / 4 | 27 / 0 | - |
| displacements (displaced attempts), refusals | - | - | 10 (65), 3 | 3 (8), 11 |
| pass model, desktop only: s / host kJ | 4761 / 509.6 | 4066 / 421.4 (-15 % / -17 %) | 3141 / 349.0 (-34 % / -32 %) | 4278 / 448.5 |
| pass model, today's phone policy (run-5 conditions) | 4682 / 402.8 | 4204 / 369.9 (-10 % / -8 %) | 3334 / 326.5 (-29 % / -19 %) | 4287 / 376.7 |
| pass model, coherent coalesced phone (#1) | 4557 / 379.6 | 3916 / 315.7 (-14 % / -17 %) | 3050 / 264.4 (-33 % / -30 %) | 4147 / 339.7 |

Bound sweep (dispatcher replay makespan, large loads): 3/300 s 4342 s 12; 3/900 s and 3/1800 s 4088 s 10;
5/1800 s 3966 s 12; 10/900 s 3364 s 6; 10/1200 s = 10/1800 s = unbounded 3172 s 5. The bypass count binds first
on this trace: a Gemma request is displaced by many Qwen arrivals, so 3 bypasses stops affinity after the first
burst and the late displacements only add replans. The run-5 measurement (12 loads = 766 s at the 27.6 W floor,
16/28 requests alone with same-model work queued) corresponds to the legacy column: the dispatcher replay gives 12
loads and 12/28 paired for the legacy dispatcher, 5 loads and 27/28 paired with WC + affinity. With today's
per-request phone policy the phone share drops (80 % -> 39 %) because pairs decode mixed (the accounting's #1
enabler); with #1 the phone share stays at 92 %. The accounting's own hypotheticals for comparison: back-fill
<= 2 with today's phone 3244 s / 344 kJ, affinity (1800 s, wave admission) 4046 s / 390 kJ.

## 6. Goal 3 (tighter lease decode estimates): not implemented

Measured predicted/actual execution time of the ACQUIRED plans: dev2base median 1.45 (range 0.30-9.71), long-tail
baseline median 1.04 (range 0.20-6.88). The error is a wide spread, not a bias: scaling estimates down would turn
the 5x under-predictions (e.g. Gemma 005: 95 s predicted, 314 s actual) into more lease overruns and replan
cascades. The WC changes make dispatch insensitive to over-prediction instead (publication promotions and
early-capacity replans pull work in when a lane frees, whatever the prediction said), and under-prediction is
already handled by lease renewal. A safe version would need per-model/per-batch measured per-token latency with an
upper quantile, applied only to the lease (not the memory reservation horizon); left for a follow-up.

## 7. Caveats

1. The estimate is a replay: solo execution durations, the four recorded residency snapshots, no helper/phone work
   in the dispatcher replay (the phone effect is only in the pass-model pricing). Loads keep their recorded
   durations (first load of a model slower).
2. Affinity trades the other model's latency for fewer switches. The wait bound limits when a request may still be
   displaced, not its total wait (running work of the resident model is not preempted). On long-tail the default
   bound lowers the maximum latency too (all requests wait less than under FIFO with 12 switches), but a trace with
   a steady stream of one model would push the other model to the bound every time. 3 bypasses / 300 s was worse
   than WC alone on long-tail (section 5), 10 / >=1200 s was best; dev_v2 is insensitive.
3. WC backfill uses predicted lease ends. A keeper that overruns its prediction delays the residency change it was
   ordered before (standard backfilling risk); lease renewal then replans the change.
4. Latent issue found by the replay (present in the legacy code, reachable more often with WC): a
   priority-compaction follower whose ticket still holds an older REPLAN receipt (it was deferred and promoted again)
   made the owner's replan fail with `REPLAN_WAKE_ROLLED_BACK` on every retry, which the runner turns into a run
   failure after 4 retries. Fixed only under WC (`refresh_replan_receipt`) to keep the flag-off path identical.
5. `published_work_promotions` also fire for deferred work behind any running transition-free request at an
   unrelated load completion; that is an earlier replan, not an earlier dispatch, but it adds replan work.
6. Transient "frozen desktop control is not currently feasible" during one priority compaction in the affinity
   replay resolved on the runner-style retry.
7. Not a continuous-batching change: requests still join via separate server slots; the slot limit is the
   coordinator lane count (Gemma 2, Qwen 4 in these catalogs).

## 8. Files

- `DISPATCH_FIXES.diff`: patch against the current main tree (18 files).
- `newroot/` = current main + patch (tested), `newbase/` = the main snapshot it was built on. `base/`, `root/`: the
  pre-rebase pair. `prev_*.bak`: the previous agent's partial edits (superseded).
- `tools/`: `simulate_dispatch.py` + `replay_common.py` (dispatcher replay), `estimate.py` (pricing),
  `summarize_sim.py`, `timeline_table.py`, `compare_real.py`, `run_scheduler_tests.py`, `run_fail_before.py`.
- `sim/final/*.json` (replays of the rebased tree), `sim/*.json` (sweeps), `testruns/*` (test logs),
  `runs/{dev2base,ltbase}` (local copies of the recorded artifacts), `PROGRESS.md`.
