# Change #4 follow-up: keep server verdicts across re-provisioning, stop the per-token re-evaluation (2026-09-24)

Fixes defects 1 and 2 of `../RIG_RESULTS.md` section 6. Built in an isolated copy of the current main
`research_dev/scheduler` (`base/` pristine, `root/` edited). Nothing was committed, no hardware was used, and
the desktop run directories were only listed and streamed (`ssh cat`) to local copies.

| file | content |
| --- | --- |
| `REPROVISION_FIXES.diff` | 17 files, +686 / -13, paths `a/research_dev/scheduler/...`. `git apply --check` passes from the repo root against the current main (stdin closed). |
| `PROGRESS.md` | checkpoints |
| `tools/` | the scratch scripts behind the numbers below (run from a tree root with `PYTHONPATH=.:gguf-py`) |

`REPROVISION_FIXES.diff` supersedes the previous agent's `REPROVISION_FIXES_PARTIAL.diff`. That diff was redone
on a fresh copy for three reasons:

- It contained an unrelated `measure_pixel_aoa.py` hunk.
- It made `server_policy_key` a 3-tuple, which `AdaptiveDecodeController.restore` rejects (`len(key) != 4`), so
  any checkpoint restore with a coherence group would have raised.
- Its boundary gate was untested and missed the actual trigger.

## 1. Problem 1: every re-provisioning stage reset the server's verdicts

**Cause (confirmed on the coherentRP run data).**

- `coherence.server_policy_key` was (model, desktop placement, layout generation, whole-layout geometry).
- Gemma 005 (generation 9) and 007 (generation 15) had the same desktop placement (`5bfd230a`), the same
  24-layer Gemma shards and the same P100 policy identity. Only the generation differed.
- So 007 got a fresh group (`verdicts {}`). At 29 tokens it was `INSUFFICIENT_OPPORTUNITY` x7, and all 26 of
  its window tokens ran on the host.
- Qwen 006 (generation 12) likewise re-decided the verdict that 002/004 had reached on generation 6 with the
  same 17 layers.
- `run-dev2allon-1` shows the same resets.

**Fix.**

- `phone_shards.artifact_layout_identity_sha256(layout, artifact)` is a content hash of one model's own
  shards: per session, the session id, layer mask, maximum columns, resident geometry (artifact, layers,
  columns, bytes, session, worker identity) and operator plan, plus the union layer mask.
- It does not depend on the generation or on the other models' sessions.
- Applied to the real RP layouts (`tools/identity_check.py`), Gemma 24 on generations 3/9/15 gives one
  identity and Qwen 17 on generations 6/12 gives another. The recurring intermediate stages match as well
  (Gemma 16 on generations 4/8/10, Gemma 8 on 5/7/11). Different layer sets get different identities.

The controller (`_internal/adaptive_decode*`):

- `start` takes an optional `helper_layout_identity_sha256`, and `helper_ready` / `helper_rebound` take
  `phone_layout_identity_sha256`. The identity must be a digest, and on `start` it needs a helper generation.
- Like the generation, the identity is fixed when the helper attaches. A rebind to new shards moves the
  session to the new shards' group.
- `server_policy_key` is `(model, placement, None, identity)` when the identity is set. Otherwise it is the
  old `(model, placement, generation, geometry)`, so there is no sharing across generations without an
  identity (fail-closed). The key stays a 4-tuple, which the checkpoint check requires.
- Everything the group holds is kept for a recurring layer set: the per-batch verdicts, reasons, attempts,
  shared records and any unfinished probe. A different layer set starts an empty group.

The unified layer (`_unified/adaptive_decode_control.py`):

- `_adaptive_layout_identity_sha256(ticket, generation)` derives the identity of the ticket's model from
  `model_placement_controller.phone_layout(generation)`.
- It does so only with `server_policy_coherence`. It returns None when the generation or the model is not in
  the layout.
- It is passed at adaptive start, at the boundary where the helper becomes ready (not per token), and at
  helper (re)materialization (`helper_envelopes_ops/materialization.py`).
- `helper_ready` / `helper_rebound` get the keyword only when it is set, so the calls with coherence off are
  unchanged.

Decision records:

- The controller snapshot carries `helper_layout_identity_sha256`.
- `server_policy` gains `layout_identity_sha256`.
- `ASSISTANCE_DECISION` carries `helper_layout_identity_sha256` only when it is set, so records with coherence
  off are byte-identical.

The per-request history lookup (`history._group_matches_session`, `candidates._operational_verification_policy`)
has no generation in its key. It matches the whole-layout geometry, which is identical for 005 and 007.
007's `historical_groups 0` has other causes: its context bucket was 8, 005's windows are in bucket 12, and
001's windows are ineligible. It is left unchanged. It is stricter than the per-model identity: a change to
another model's session hides the history. Relaxing it would need the identity in the persisted
grouped-observation schema.

## 2. Problem 2: per-token re-evaluation while the other model was queued

**Trigger (confirmed).**

- `adaptive_decode_boundary` -> `_reevaluate_pending_phone_layout_at_boundary` runs on every decode token.
- With the other model queued and not on the phone, it has an `uncovered` artifact with cached online-learning
  demand.
- Its dedup compares the previous record's `route_evidence_by_artifact[x].source_route_id` against
  `compiler.phone_residency_evidence_status(x)`.
- For a learning model the record holds the learning status (`LEARNING_EXPLORATION_READY` with a source
  route), while the compiler returns `ROUTE_EVIDENCE_UNUSABLE` with no source route. The two never compare
  equal.
- The result is a full portfolio evaluation per token:
  - coherentRP: 1,202 `REPROVISION_RETAINED`, 34.6 s of summed publication delay, 19.9 MB of RESULT.json;
  - allon: 1,224 RETAINED.
- The harness reproduces it through the real hook (`tools/storm_repro.py`): on base, 60 boundaries give 60
  RETAINED records.

**Fix (knob on only; `_unified/phone_residency_ops/reprovision.py`, two call sites in `portfolio.py`).**

1. **Gate, placed before the dedup** (`_boundary_reevaluation_due`). It can only remove evaluations relative to
   base. A boundary re-evaluates when this state changed since the last boundary evaluation:
   - the non-terminal tickets with their dispatch state, transition (desktop load) status and model;
   - the READY/target layout generations;
   - each session's state, generation, resident model and helper-reference count;
   - the in-use sessions;
   - arrived-work buckets (powers of two, as the dedup uses);
   - the route evidence (reason, source route, benefit) of the models with work;
   - the uncovered set.

   Otherwise it re-evaluates at most once per `boundary_reevaluation_interval_us`. This is a new optional knob
   field: default 10 s, 0 = every boundary.
2. **Record coalescing** (`_coalesce_unchanged_decision`). A RETAINED or HOLD decision is not recorded when it
   is identical to the last EVALUATED record. Identical means the same decision fields (arrived work compared
   by bucket), the same current, planning and selected layout, generation and state, the same confirmation
   and the same memory limit.
3. **Counters.** The next record carries `boundary_evaluations_skipped` and `unchanged_decisions_coalesced`
   since the previous record, plus their `_total`s.

These re-evaluations are not gated, so swap-at-dispatch and staging are unchanged:

- dispatch (`_reevaluate_phone_layout_for_desktop_load`);
- release;
- session-ready (`_reevaluate_ready_layout_portfolio`);
- the pending-candidate branch of the boundary hook.

Knob off: the gate returns True and no coalescing runs, which is the base path.

## 3. Tests

"base" means the new test file run against the pristine `base/` copy.

| test | base | root |
| --- | --- | --- |
| `test_adaptive_layout_identity.py` (new, 10): recurring layout keeps verdict across a generation bump; 007-like 3-token follower starts on `P100` with `SERVER_POLICY_COHERENCE`; key/snapshot carry the identity | ERROR (no identity API) | ok |
| different layer set starts fresh (empty group, baseline) | ERROR | ok |
| without an identity a generation bump still starts fresh; same generation still shares | pass | ok |
| checkpoint/restore of an identity-keyed group (the partial diff's 3-tuple would raise here) | ERROR | ok |
| rebind keeps the group only for unchanged shards; invalid identity rejected | ERROR | ok |
| ASSISTANCE_DECISION: coherence off has no new field and `server_policy None`; coherence on carries the identity | pass / ERROR | ok |
| identity function: other model's session change keeps it, different A layer set changes it; scheduler derives it per generation only with coherence, None for unknown generation or absent model | ERROR | ok |
| `test_phone_reprovision_boundary.py` (new, 6, base names only): 60 unchanged boundaries -> 3 evaluations (0, 10.08, 20.16 s), 1 record, counters 57 / 2 | FAIL (60 / 60) | ok |
| a state change re-evaluates at the next boundary; the record carries the counts (9 skipped) | FAIL | ok |
| each of arrival, completion, dispatch, desktop-load finish, session generation change, route-evidence change triggers exactly one re-evaluation | FAIL (all 6) | ok |
| swap at dispatch still proposes the one-session first stage (FOLLOW, `loading`, confirmed); the record carries 28 skipped / 1 coalesced | FAIL (storm part) | ok |
| interval 0: every boundary evaluates, identical RETAINED recorded once | FAIL | ok |
| knob off: 20 boundaries -> 20 evaluations and 20 `LEARNING_RETAINED` records, no gate state (the base behaviour) | pass | ok |
| `test_phone_resident_model_reprovision.py` config test updated for the new field (round trip, 0, rejects -1 / 1.5) | FAIL | ok |

**Option-off digest guard** (`tools/option_off_digest.py`, run on base and on root):

- Knob off: the 40-boundary storm scenario plus the knob-off portfolio evaluation, 65 phone-layout events.
- Coherence off: lead, follow, measured phone window, completion and next requests, as 6 ASSISTANCE_DECISION
  payloads and their directives.
- The digests are identical on base and root: `ea96f194...` and `954f5fd9...`.

**Full suite and pyflakes:** see section 6.

## 4. Expected effect (coherentRP, estimates, not measured)

**Problem 1.** Host-policy window tokens spent before the first phone window, from `tools/reset_cost.py`, with
the section-5 fleet J/token of RIG_RESULTS:

| request | host tokens | why they would move to the phone | estimated saving |
| --- | ---: | --- | ---: |
| 007 | 26 | the whole request inherits 005's `verdict[1] = P100`; saves 57.9 - 24.0 J per token | about 0.88 kJ |
| 005 | 11 | inherits 001's verdict (generation 3, same identity) | at most about 0.37 kJ |
| 006 | 11 | inherits 002/004's `verdict[1]` and `verdict[2]` | at most about 0.39 kJ |

- Total: at most about 1.6 kJ fleet, about 3 % of the run's 53.7 kJ host energy. The first window of a
  follower may still be a transition window, so this is an upper bound.
- allon is similar, and 001's 27 host tokens there would also inherit 000's verdict.
- More phone tokens can flip near-tie outputs (RIG section 2).

**Problem 2.** Applying the gate and the coalescing offline to the recorded EVALUATED streams
(`tools/gate_replay.py`, an approximation of the state signature from what the records hold):

| run | RETAINED records now | evaluations with the fix | records with the fix |
| --- | ---: | ---: | ---: |
| coherentRP | 1,202 | about 73 | about 33 |
| allon | 1,224 | about 68 | about 32 |

- Evaluation time drops from about 35 s to about 0.6 s (7.7 ms each).
- RESULT.json should drop from 41 MB to about 21 MB.
- In the harness, a skipped boundary costs 0.11 ms against 1.84 ms for a base evaluation (600 boundaries, 501
  base records against 10).
- Whether the extra `TOKEN_STREAM_CATCH_UP` windows disappear is not shown. The stalls were at most one
  evaluation per token, and those evaluations are now mostly gone.

## 5. Caveats

1. **Not scoped to the knob.** Problem 1 applies to every arm with `server_policy_coherence`, with or without
   re-provisioning. A coherentEF-type arm also keeps its verdicts across a startup generation that leaves the
   model's shards unchanged. Arms with coherence off are byte-identical (the guard above).
2. **Failure verdicts last longer.** A host verdict from `SERVER_PHONE_POLICY_FAILED` (a failed phone window
   or control) now lasts until the model's shards change, not until the next generation. A swap-back to the
   same shards no longer clears it. This is deliberately not special-cased (fail-closed). No failures occurred
   in these runs.
3. **What the identity leaves out.** It does not include the executor or batch plan (the group key never
   did), the column quantum (a static session property, covered by the columns and the worker identity) or
   the stored shard file hash. For a given parent artifact, the resident geometry and the operator plan
   determine the shard content.
4. **Time-dependent decisions wait.** When nothing else changes, the gate delays them by up to the interval
   (10 s). Examples are a stage deferred by the 30 s minimum residency and an expiring lease reservation.
5. **Pending candidates are not gated.** The boundary branch for an unconfirmed pending candidate still runs
   at every boundary, as on base. It was not involved in the storm: all 1,202 records were confirmed, with
   planning equal to current.
6. **Trailing counts are not exported.** Counts after the last record are not written anywhere (there is no
   final flush), so the `_total` values in the last record are a lower bound.
7. **The knob-off path has the same dedup defect.** It is left as is so that knob-off stays identical. Whenever
   the learning selector keeps a queued model off the phone, the base dedup records `LEARNING_RETAINED` per
   token; the harness reproduces this on base and root. The one-line cause is the `source_route_id`
   comparison in `portfolio._reevaluate_pending_phone_layout_at_boundary`. coherentEF logged only 35
   EVALUATED events because learning swapped on arrival.
8. **Configuration output changes.** The configuration `to_json()` now includes
   `boundary_reevaluation_interval_us`, so re-provisioning manifests and the `..._CONFIGURED` event carry it.
   The `launch.py` -> `runner.py` plumbing passes the JSON verbatim.
9. **Not validated on hardware.** The first rig check should look at three things:
   - `desktop_reprovision` counters and EVALUATED counts;
   - 007/006-type requests starting on `SERVER_POLICY_COHERENCE` with `server_policy.layout_identity_sha256`
     set;
   - swap-at-dispatch timing unchanged against RIG section 3.

## 6. Full suite and pyflakes

The runner is `run_scheduler_tests.py root tests_root.json`. It runs every `tests/test_*.py` in its own
`/usr/bin/python3` process, with module mode for files with relative imports, `gguf-py` and
`research_dev/spikes` linked in, and `PYTHONDONTWRITEBYTECODE=1`.

- **Final run:** 110 files, 1,654 tests. The only failing file is `test_resident_router_subset.py`, which is
  environmental: it compiles `examples/layersplit/ffn-split-resident-router.cpp`, and that file is absent in a
  copy.
- **Known flaky and special files:**
  - `test_automated_runtime_admission.py` (including the 10 ms test) passed.
  - `test_kv_lazy_backing.py` (`test_kv_touch_occupies_exactly_the_cache_pages`) passed.
  - `test_split_kv_attention.py` passed in module mode (5 tests).
  - `test_two_phone_server_native.py` exited 0 with its native test skipped.
- **Earlier run** (`tests_root_run1.log`, before the last fix): `test_session_cow_transaction.py` had 2
  errors. Its tests use tickets without `model`, a bare `object.__new__(UnifiedScheduler)` without an adaptive
  config, and exact `assert_called_once_with` on `helper_ready` / `helper_rebound`. Fixed by checking
  coherence before reading the ticket's model and by passing the identity keyword only when it is set.
- **pyflakes** on the package without `campaigns/burstgpt/reports` (`_internal _unified adapters
  configuration tests scheduler.py config.py __init__.py campaigns/burstgpt/*.py`) is clean on root, as on
  base.
- The diff is all ASCII.

## 7. Reproduce

```sh
W=/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/reprovision-fixes2
cd /home/myid/zs89458/Documents/llama.cpp-release && git apply --check $W/REPROVISION_FIXES.diff < /dev/null
cd $W/root && PYTHONPATH=$PWD:$PWD/gguf-py python3 -m unittest research_dev.scheduler.tests.test_adaptive_layout_identity \
    research_dev.scheduler.tests.test_phone_reprovision_boundary
cd $W && /usr/bin/python3 run_scheduler_tests.py root tests_root.json      # one process per test file
for t in base root; do (cd $W/$t && PYTHONPATH=$PWD:$PWD/gguf-py python3 ../tools/option_off_digest.py); done
cd $W/root && PYTHONPATH=$PWD:$PWD/gguf-py python3 ../tools/identity_check.py <run dir>   # RESULT.json copy
python3 $W/tools/gate_replay.py <run dir>; python3 $W/tools/reset_cost.py <run dir>
$W/make_diff.sh    # rebuilds REPROVISION_FIXES.diff from base/ and root/
```
