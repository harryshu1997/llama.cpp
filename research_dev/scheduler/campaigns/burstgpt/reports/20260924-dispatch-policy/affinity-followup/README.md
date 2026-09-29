# Model affinity follow-up (#5): why it never fired on the dev_v2 rig arms, and the fix (2026-09-24)

Deliverable: `DISPATCH_AFFINITY_FIX.diff` (9 files, +478/-20) against the CURRENT main tree (after the #4
follow-up merge). `git apply --check` from the repo root passes with stdin closed, and applying it to a copy of
main reproduces the tested tree byte for byte. Only `model_affinity: true` changes behaviour: with the flag off, or
with work-conserving admission alone, the decision sequence is unchanged (replays below, full test suite). No
hardware was used and nothing was run on the desktop; the rig artifacts were copied read-only.

## 1. Root cause (decision logs of both measured arms)

### dev_v2 desktop + dispatcher (`run-dev2baseDP-1`)

| t (s) | record | what the log shows |
|---:|---|---|
| 73.2 / 75.4 | 9, 11 | Gemma 000 done, Llama overlay done |
| 79.0 | 12 | Qwen 002 planned with a cold->hot Qwen load (evicts Gemma), dispatched at once; load publishes at 128.8 |
| 86.4 / 99.0 | 13, 14 | Qwen 003 / 004 planned as hot->hot publications behind 002 (Qwen not ready yet) |
| 109.6 | 15 | Gemma 005: cold->hot Gemma load (evicts Qwen), start 681 s, barrier after 002-004 |
| 112.0 | 16 | Qwen 006: cold->hot Qwen load (evicts Gemma), start 843.8 s, barrier after 005 (arrival order) |
| 127.9 | 17 | Gemma 007: Gemma load, barrier after 006 |
| 129.2 | 22 | Qwen 006 REPLAN (wake `residency_observation_changed`): transition-free Qwen plan on coordinator lane 3, start 129.2, predicted end 253.9 |
| 273.3 | 26, 27 | 004 completes; 005 replans (Gemma load 273-317) |
| 317.0 / 625.6 | 29, 30 | 006 replans twice, now with a Qwen reload; runs 711-805 |

Queue snapshots: at 129.2 s (`runtime-008`, just before 006's replan) 006 has predecessors
{002, 003, 004, 005} and 005 is `DEFERRED_REPLAN` (`residency_projection_invalid`, reservation cancelled). At
273.3 s (`runtime-009`) 006 is `DEFERRED_REPLAN` (`predecessor_replan`), `residency_transition_barrier: false`,
predecessors `['005']`. RESULT `dispatch_policy.statistics`: displacements 0, displaced attempts 0, refusals 0,
bypass counts `{}`.

Answers to the candidate explanations:

1. **Code path never reached (the cause).** Affinity was evaluated only at arrival, in
   `_submit_automated_request_once`. `model_affinity_displacement` first requires the arrival's model to be hot
   with a ready executor (`_model_is_resident`). 006 arrived at 112.0 s, inside 002's Qwen load (79.0-128.8 s),
   so the check failed and the function returned None before any bound or refusal accounting (hence all
   statistics 0). 003 and 004 arrived during the same load, and 005 and 007 arrived after Gemma had been evicted
   and before Qwen was ready. No arrival of this trace fell in a window where its model was resident, so no
   displacement was ever possible.
2. **The replan at publication kept the arrival-order edge.** At publication, WC replanned 006 onto a
   transition-free plan on the free fourth Qwen slot (start 129.2 s). `_rebind_causal_predecessors` keeps
   existing edges between entries that still conflict, and 005's cancelled Gemma plan holds all eight `cuda0`
   lanes. So the attempt-0 edge 006 -> 005 (both were residency barriers; 005 arrived 2.4 s earlier) survived.
   006 stayed QUEUED and then DEFERRED behind 005 until 004 finished at 273 s, 005 loaded Gemma, and 006 needed a
   reload.
3. **Not a slot limit.** Qwen has 4 lanes: 002, 003 and 004 used lanes 0-2, and 006's replanned plan got lane 3
   at 129.2 s.
4. **Not the fairness bound.** The bounds were never evaluated: bypass counts are empty.
5. **The WC keeper rule is timing-fragile and was not re-evaluated.** Even if the edge had been re-derived, the
   WC keeper-before-barrier rule (`_frees_lanes_before`) orders 006 first only when its predicted end (253.9 s)
   is at or before the latest reserved end of 005's running predecessors. On the rig, 004 was ACTIVE (336 s), so
   006 would have gone first. In the replay, 003/004 were not yet ACTIVE at that moment, only 002 (245.5 s), so
   006 would not. Only affinity is meant to let the resident model delay a switch.

### dev_v2 OP15 all-on (`run-dev2allon-1`)

The mechanism is the same, with a second shape. 002's Qwen load ran 102.8-153.6 s, and 005 (109.4 s), 006
(111.8 s) and 007 (127.8 s) arrived during it. At 153.9 s and 154.9 s, 005, 006 and 007 were replanned as
priority-compaction followers of 003 and 004 (`capacity_released_early`, records 24-32). 006's follower replan
projected 005's still-QUEUED Gemma switch first, so it got a Qwen reload plan again (start 435.5 s, then 524.2 s)
and stayed a barrier behind 005. Its load plan was never "obsolete" in that projection, and its predecessors were
not all running, so neither publication wake applied. The statistics are again 0/0/0.

### Why the replay predicted 663 s and 2 loads

The replay tool is faithful. Replaying the dp arm with its own recorded durations through the merged code gives
**863 s vs 865 s measured**, the same 5 large-model loads and the same 006 trace (`QUEUED barrier False preds
['005']` at 128.8 s; `sim/dp_base_aff_trace.log`). The 663 s prediction came from replaying **dev2base (legacy
arm) durations**. There, Gemma 000's cold load took 97 s (acquired 98.1 s) and 000 ran until 161 s. At 109 s,
005's arrival therefore saw Gemma resident and ready while the Qwen switch (002-004) was still queued behind 000,
so arrival affinity fired. On the rig the page cache was pre-populated (`populate=1`): Gemma's load took about
6 s, 000 and 001 finished at 73 s, and the Qwen switch was dispatched at 79 s. Both 005 and 006 then arrived
inside a transition window where neither model is "resident with a ready executor". The difference is when the
switch starts, not solo versus batched decoding.

## 2. Change set (all behind `model_affinity`; flag off and WC-only unchanged)

| file | change |
|---|---|
| `_unified/automated_requests_ops/affinity.py` | `model_affinity_replan_displacement`: a replan whose model is resident with a ready executor, and that causally waits (transitively) on not-started residency changes of other models, displaces those changes plus every attempt queued behind them, excluding itself. It refuses when another model's change is running or replanning, and applies the same bypass/wait bounds and cohort/projection-token refusals as the arrival path. The closure and bounds are shared with the arrival path through `_close_displacement`, so arrival behaviour is unchanged. `record_refusal=False` gives a side-effect-free pre-check. |
| `_unified/automated_requests_ops/replan_commit.py` | `_replan_with_model_affinity` runs first in `_replan_automated_request_once`, including for compaction followers. Inside a nested transaction it runs `replan_queued_now(displaced, model_affinity_displaced)`, sets `dispatch_precedence`, and runs the normal prepare + execute replan with `require_resident_plan=True`. That guard raises before the memory preview and commit when the new plan still changes residency. Any error rolls the whole attempt back, counts `affinity_refusals`, and runs the ordinary replan on the unchanged state, which handles the error as before. |
| `_unified/automated_requests.py` | the mixin wrapper passes `require_resident_plan` through (default False) |
| `_internal/runtime_queue.py` | `admit` accepts `precede_request_ids` on a REPLANNING re-admission. It drops the owner's edges to the displaced entries, adds displaced -> owner edges (a cycle raises `RuntimeQueueError`), then rebinds. |
| `_unified/automated_requests_ops/observations.py` | Affinity wake (the all-on shape): a QUEUED attempt whose transitions are already realized by the live snapshot (its load is published) is replanned now when the pre-check says its replan may displace, even though its predecessors are not all running. The wake is counted in `publication_replans`. |
| `_unified/automated_requests_ops/replan.py` | A `model_affinity_displaced` replan no longer invalidates its sequence frontier. The work queued behind it was displaced with it. Without this, 005's displaced replan deferred 006 (a later sequence, but now its predecessor), and 006's re-replan hit the memory horizon of 005's new Gemma reservation: "frozen desktop control is not currently feasible", terminal FAILED (ablation below). This also closes the same latent hazard for arrival displacements. |
| `_internal/runtime_dispatch_policy.py`, `README.md` | docs |
| `tests/test_dispatch_policy.py` | +7 tests (29 total) |

Fail-closed properties kept:
- Every replan still passes route certification, the memory ledger preview/reserve, residency projection, the
  exclusive-transition barrier, the calendar and the KV/slot lanes.
- A displacement is kept only when the replanned plan preserves residency. Otherwise the transaction restores
  the queue, the tickets, the reservations and the decision log.
- Bounds and refusal conditions are the arrival ones.
- Decisions are deterministic (sorted iteration, no clocks).
- No new wake reason: the wake uses `residency_observation_changed` and displaced attempts use
  `model_affinity_displaced`.

The decision log carries `selected.dispatch_policy` on the displacing REPLAN record with the same fields as at
arrival plus `replan_reason`. No new statistics.

## 3. Tests

- Scheduler level (two models share an exclusive 2-lane accelerator; a1 loads A; b1 (B) and a2 (A) arrive during
  the load):
  - publication replans a2 ahead of the queued switch, and a2 is dispatchable at publication;
  - the same happens when a2 is woken by the obsolete-transition rule instead of the affinity wake;
  - the displaced switch replans behind the resident work without deferring it;
  - with WC only, or past the bypass bound (refusal counted), the switch stays first;
  - the all-on shape (a2 replanned before its executor was ready, then woken at the next observation);
  - determinism.
- Queue level: precedence on a replan re-admission, cycle rejection, checkpoint round trip.
- **Pass after:** 29/29 on the rebased tree.
- **Fail before:** current main plus the new test file: the 7 new tests fail (6 FAIL, 1 ERROR) and the 22 existing
  tests pass (`testruns/fail_before_base2.log`).
- **Ablation:** without the `replan.py` exemption, `test_displaced_switch_replans_behind_without_deferring_the_resident_work`
  fails and the dp replay ends with 006 FAILED.
- **Full suite** (every `tests/test_*.py` in its own process, `tools/run_scheduler_tests.py`), rebased tree and
  current main run concurrently: everything OK except `test_resident_router_subset.py`, which needs the
  `examples/layersplit` C++ sources and fails the same way on main. On the patched tree the known timing-flaky
  `test_cached_synthetic_refinement_is_below_ten_milliseconds` failed once (10.6 ms under load) and passed 3/3 in
  isolation on both trees. `test_kv_touch_occupies_exactly_the_cache_pages`, `test_two_phone_server_native` and
  `test_split_kv_attention` passed. The pre-rebase pair gave the same result.
- `pyflakes` on the 8 changed Python files: clean.

## 4. Effect estimate (dispatcher replay through the real scheduler, then the calibrated pass model)

The dispatcher replay (`tools/simulate_dispatch.py`) uses each arm's own recorded load and execution durations.
Makespan = last completion. Large-model loads exclude the Llama overlay.

| arm (durations) | legacy | WC + affinity (main) | WC + affinity (fix) |
|---|---:|---:|---:|
| dev_v2 desktop + dispatcher: makespan s / large loads | 966 / 5 | 863 / 5 (real 865) | **624.5 / 3** |
| dev_v2 all-on durations (desktop-baseline approximation) | 902.5 / 5 | 802.1 / 5 (real 812) | **594.9 / 3** |
| dev_v2 dev2base durations (earlier prediction) | 1147 / 5 | 663.3 / 2 | 663.3 / 2 (unchanged) |
| long-tail v1 baseline arm | 4835 / 12 | 3172 / 5 | 3182 / 4 |

Fixed dev_v2 service orders:
- dp durations: `g000[1-73] g001[7-31] l00[73-75] q002[79-211] q003[129-211] q004[129-273] q006[129-223]
  g005[273-625] g007[316-332]`. Qwen 006 joins the Qwen server at publication and Gemma 007 joins Gemma 005.
- all-on durations: the same shape, with `q006[154-230]` and `g007[326-344]`.
- Every large request decodes with a same-model partner (8/8, was 5/8). One displacement (005 and 007 bypassed
  once), 0 refusals.
- Mean / max latency: dp 283 / 735 -> 159 / 515 s; all-on 278 / 674 -> 171 / 486 s.

Pass model (`tools/estimate.py --dev2`: long-tail calibration with dev_v2 shapes and dev2base load times, so the
absolute values are high; use the ratios):

| order | desktop only s / host kJ | today's phone policy | coherent coalesced phone |
|---|---:|---:|---:|
| main WC + affinity (dp and all-on orders are the same) | 1169 / 90.7 | 1184 / 79.0 | 1144 / 72.3 |
| fix | 872 / 73.6 (-25 % / -19 %) | 906 / 66.9 (-23 % / -15 %) | 856 / 58.7 (-25 % / -19 %) |

Scaled to the measured arms (rough):
- desktop + dispatcher: 865 s / 82.3 kJ -> about 645 s / 67 kJ;
- all-on: 812 s / 45.7 kJ -> about 620 s / 39 kJ (today's phone ratio) or 607 s / 37 kJ (coherent ratio).

The dispatcher replay alone gives -28 % (desktop + dispatcher) and -26 % (all-on) makespan.

Long-tail v1 (regression check), main vs fix:
- Dispatcher replay: 3172 -> 3182 s (+0.3 %), large loads 5 -> 4.
- Mean / max latency 910 / 2037 -> 746 / 1924 s.
- 13 displacements (66 attempts) and 13 refusals: 9 replan-path fairness bounds, 4 arrival no-gain. No refusal
  came from the error fallback (`tools/sim_trace_refusals.py`).
- Pass model: desktop only 3141 / 349.0 -> 3201 / 362.5 (+1.9 % / +3.9 %); today's phone 3334 / 326.5 ->
  3352 / 325.9; coherent 3050 / 264.4 -> 3090 / 271.9. Mean queue wait 683 -> 589 s.

Long-tail is therefore neutral on time and energy and better on latency. Qwen 005 and 009 now join the first Qwen
run, but Qwen 027 (arrived 1580 s) waits behind the 13-request Gemma block and decodes alone at the end.

## 5. Caveats

1. These are replays, not measurements. They use recorded solo-or-as-measured durations and the recorded
   residency snapshots; the all-on arm is replayed in desktop-baseline selection mode (its plans were desktop
   routes, `QUALIFIED_FALLBACK_NO_SAFE_ALTERNATIVE`), without phone helpers.
2. Batching slowdown is ignored in the dispatcher replay. With the fix, Qwen 006 decodes as the fourth Qwen
   request. If batching pushes 004 and 006 past about 273 s, Gemma 005's switch starts later; the pass model
   includes that slowdown.
3. The replay of the all-on arm did not reproduce its exact replan sequence. On the rig, 006 was replanned as a
   compaction follower at 153.9 s; in the replay it is reached by the new publication wake. Both paths are
   covered by tests, but only the dp shape is validated against a recording.
4. Affinity trades the other model's latency for fewer switches, under the same bounds as at arrival (defaults 10
   bypasses / 1200 s). A replan displacement counts toward the displaced request's bypass count.
5. The publication wake re-evaluates on every observation. Observations are event-driven (transition completion,
   arrival), so a pre-check that passes while the replan then refuses (the plan still changes residency) costs one
   extra replan per event and counts one refusal. Error fallbacks are also counted as `affinity_refusals` and are
   not separated from bound refusals in RESULT.json.
6. Latent hazard, now reduced: a queued keeper whose memory horizon (the reservation end, which follows the upper
   finish bound) overlaps a switch reserved behind it cannot be replanned again without the desktop control being
   rejected (terminal FAILED). The displaced-replan trigger is closed here. Other triggers of a second keeper
   replan before it dispatches (lease or capacity replans) are theoretically possible; none occurred in the
   replays.
7. The long-tail WC-only replay flakes on both trees ("runtime physical execution receipt differs from the
   decision"). This is a wall-clock race in the replay tool's epoch arithmetic. When both runs pass they give the
   same makespan and loads (4155.7 s, 12).

## 6. Files

- `DISPATCH_AFFINITY_FIX.diff`, `README.md` (this file); copies in
  `research_dev/scheduler/campaigns/burstgpt/reports/20260924-dispatch-policy/affinity-followup/`.
- The work dir (scratchpad `dispatch-fixes2/`) holds:
  - the trees: `base2/` (current main snapshot) and `root2/` (+ fix, tested); `base/` and `root/` are the
    pre-rebase pair;
  - `rig/{dp,allon}` (read-only copies of the rig artifacts) and `runs/{dp,allon}` (replay inputs; RUN_COMMAND
    re-tokenized with shlex, `--dispatch-policy-json` dropped, JSON values compacted);
  - `sim2/` (replays of the rebased trees) and `sim/` (pre-rebase, traces);
  - `testruns/`, and `tools/` (`run_sims3.sh`, `summarize3.py`, `sim_trace_refusals.py`, plus the earlier
    replay/pricing tools).
