# Adaptive-decode evidence fixes F1a, F1b, F2, F3, F4 (2026-09-24)

These fixes implement F1a, F1b, F2, F3 and F4 from
`research_dev/scheduler/campaigns/burstgpt/reports/20260924-phone-rejection-diagnosis/README.md`.
They were built in a private copy of `research_dev/scheduler`. The main repo was not touched: no
edits, no commits, no hardware, no ssh/adb.

- **Patch:** `EVIDENCE_FIXES.diff` (894 lines, paths `a/research_dev/scheduler/...`, apply from the repo
  root with `git apply` or `patch -p1`).
  - 10 files modified plus 1 new test file.
  - Code: +262/-4 lines. The largest changes are in `sequencing.py` (+102), `windows.py` (+60) and
    `_unified/adaptive_decode_control.py` (+24). The new test file adds 328 lines.
  - `git apply --check` against the current main tree passes. The 10 target files are byte-identical to
    the snapshot the patch was built from.
- **Snapshot:** `base/` is the pristine copy of the main-repo tree taken at 00:34, and it already
  contains the coherent-phone-policy change. `root/` is the working copy with the fixes.
- **Merge overlap:** the patch does not touch `coherence.py`. It does touch
  `_unified/adaptive_decode_control.py` (4 small hunks), which the coherence work also edits. If that
  file changes before the merge, re-check with `git apply --check`.
- **F5 and F6 are not implemented** (see the last section).

## Results in short

| check | result |
| --- | --- |
| New unit tests (`tests/test_adaptive_evidence_fixes.py`, 12 tests) | Pristine tree: 8 fail and the 4 regression guards pass. Fixed tree: 12/12 pass. |
| Full scheduler suite (run_all.py semantics without the S42 spike tests) | Pristine: 100 files, 1484 tests. Fixed: 101 files, 1496 tests. Per-file exits and counts are identical except for the new file. Both trees fail `test_resident_router_subset.py` (environmental: it compiles `examples/layersplit/*.cpp`). The known-flaky 10 ms timing test fails intermittently on both trees. |
| pyflakes on the 11 changed files | clean. All changed files are ASCII-only. |
| Diagnosis replay `replay_qualification.py` (000, 001, 002, 004) on both trees | Byte-identical output, and equal to the diagnosis's `data/replay_*_as_run.txt`. `_qualifies`, the bounds and the latency limits are unchanged. |
| Closed-loop decision replay, dev run (new harness, validated on the pristine tree) | 000: host becomes **P100** (phone 25 -> 89 of 140 tokens). 001: host becomes **P100** (phone 51 -> 551 of 577 tokens). 002 and 004: identical on both trees. |
| Run-5 batch-2 rejections | No phone candidate passes `_qualifies` at any batch-2 window, in any of the 27 run-5 requests that measured a phone policy (023 never did), even with every elimination cleared and the F3/F4 rules applied. No fix can flip them (argument below). |

## What changed

The shared invariants hold for every fix:

- Promotion and keep decisions still go through the unchanged `_qualifies` (the same bounds, latency
  limit and assumed-power flag).
- The strict `energy_measurement_eligible` property is unchanged.
- The new rules only remove evidence, or delay or re-test a negative decision.

### F1a: an inconclusive incumbent re-check measures more; it does not demote

Code: `sequencing._next_after_window` at the incumbent re-check, and `sequencing._incumbent_inconclusive`.

- **Condition.** When `_qualifies(incumbent)` fails, both sides have current operational windows, and
  `_qualification_needs_more_evidence(..., check_budget=False)` is True (the means favor the incumbent
  but the bounds overlap), the controller calls the existing `_reserve_comparable_measurements`. That
  function uses the same `comparable_measurement_targets` and `_measurement_pair_budget` as
  `_select_probe_winner`, then `_finish_verification`.
- **Unaffordable.** If the extra windows cannot be afforded, the existing path returns INCONCLUSIVE and
  demotes. That is still fail-closed.
- **Decisive re-checks demote as before** (`INCUMBENT_NO_LONGER_BENEFICIAL`): means against, latency
  elimination, or no evidence on one side. The last case keeps run-5's coherence followers unchanged.

### F1b: eligible host windows re-qualify a measured candidate

Code: `sequencing._candidate_requalified` in the exploitation branch.

- **Trigger.** While EXPLOITING on the host, each new eligible host window re-checks the non-eliminated
  probe candidates that have current windows of their own.
- **Switch.** If one passes `_qualifies`, `_continue_best("CANDIDATE_REQUALIFIED")` switches to it
  (`_best_valid_policy` -> `_consider_incumbent`).
- **No new eliminations.** Host windows do not add negative decisions.

### F2: one host window cannot permanently eliminate a probe candidate

Code: `sequencing._single_reference_rejection`, the new stage `reference_baseline`, and
`promotion._single_reference_probe` / `_host_evidence_count`.

- **Trigger.** In the probe `candidate` stage, a rejection can rest on a single host window (host
  evidence count, current windows plus historical groups, below 2). The rejection is either
  `LATENCY_BOUND_EXCEEDED`, or, in LEARNING, the paired gate failing while the energy bounds still
  overlap.
- **Re-test.** The controller first takes one more host reference window, if `_measurement_pair_budget`
  (1 host window, 0 candidate windows) affords it. It then re-applies the unchanged
  `_update_elimination` and paired gate, now decisive, and continues the sweep. A decisive rejection is
  permanent, with the same reason strings.
- **What is not deferred:**
  - bound-resolved single pairs (ENERGY_DOMINATED: candidate lower >= host upper);
  - incumbents and coherence verdicts (their latency guard is their keep gate);
  - ticket fallbacks and anything outside PROBING.
- **Unaffordable re-test.** The old behaviour applies.

### F3: stall/catch-up guard (`TOKEN_STREAM_CATCH_UP`)

Code: `windows.token_stream_caught_up`, `boundary()` and `record_window`.

- **Input.** `boundary()` already receives every token's observation time. It now keeps
  `(token, at_us)` for the open window.
- **Rule** (derived from 000 W0/W1 and run-5 027 W4-W6). A window "ended in catch-up" when both hold:
  - its last >= 3 tokens were observed within 1/8 of the window's mean token interval of each other;
  - that burst follows a gap larger than 4x the window's median gap.
- **Both windows sharing the late token are removed:**
  - the window that ended in catch-up;
  - the next window, unless it starts at a control acknowledgement.
- **Evidence only.** The window keeps its warm-up role and accounting, and the receipt names the reason.
- **Why this shape.** A burst means the stream reader was behind the server, so that boundary token's
  time is late. A stall that is fully absorbed inside one window, followed by a normal last token (for
  example dev 000 W4, gaps 1101/0.8/513/504 ms), is kept.

### F4: session load on the helper's phone (`HELPER_PHONE_SESSION_LOAD`)

Code: new `AdaptiveDecodeController.helper_disturbance(request_id, reason=...)`,
`sequencing.hold_disturbed_helper`, and `_unified._report_helper_phone_disturbance`.

- **Detection (`_unified`).** At start, at each window record and at each acknowledgement, `_unified`
  checks whether any phone session sharing the helper's phone (the `session://<phone>/` endpoint prefix
  of the helper envelope's `phone_session_ids`) is in placement state LOADING. LOADING covers
  SESSION_LOADING to SESSION_VERIFIED.
- **Phone windows** open while that holds become ineligible with that reason. Host windows stay evidence.
- **Probe hold.** While it holds, `_next_after_window` starts no probe or verification:
  - an open verification is deferred;
  - a running probe ends as PROBE_INCOMPLETE and retries after the load;
  - otherwise `_continue_best(reason)` runs.
- **Coherence mode.** The hold sits after the coherence server directive, so it applies when the
  server-policy coherence path does not decide.

### Supporting changes

- **Receipt.** `AdaptiveDecodeWindowReceipt.measurement_ineligible_reason: str | None`.
  - Set only when a guard removed an otherwise eligible window.
  - It must be ASCII and implies `measurement_eligible=False`.
  - It is serialized and hashed only when set, so every existing record and store keeps its hash.
  - `from_json` reads it.
- **Control-cost estimate.** `_verification_control_cost` ignores guard-removed windows, so a removed
  window followed by a zero-token control is not mistaken for a transition.
- **Snapshot fields:** `measurement_guard_reasons`, `helper_disturbance`, `reference_policy_hash`.
  `ASSISTANCE_DECISION` also records `helper_disturbance`.
- **New reason strings:**
  - directive reasons `CANDIDATE_REQUALIFIED` and `HELPER_PHONE_SESSION_LOAD`;
  - zero-assistance reason `REFERENCE_BASELINE`;
  - stage `reference_baseline`.

## Tests

The new file is `research_dev/scheduler/tests/test_adaptive_evidence_fixes.py`. It uses a token-level
driver that calls `boundary()` for every token. The "Pristine" column comes from running the same file
against `baserun/` (the pristine tree).

| test | fix | pristine | fixed |
| --- | --- | --- | --- |
| `test_overlapping_bounds_resolve_with_more_windows_and_keep_the_phone` | F1a | FAIL (`INCUMBENT_NO_LONGER_BENEFICIAL`) | pass |
| `test_decisive_recheck_still_demotes` | F1a guard | pass | pass |
| `test_host_windows_requalify_a_measured_candidate` | F1b | FAIL (`WINDOW_OPENED` != `CANDIDATE_REQUALIFIED`) | pass |
| `test_host_windows_never_requalify_a_worse_candidate` | F1b guard | pass | pass |
| `test_one_host_window_cannot_eliminate_a_learning_candidate` | F2 | FAIL (P100 eliminated LEARNING_NO_PAIRED_IMPROVEMENT) | pass |
| `test_confirmed_rejection_is_permanent_after_the_second_reference` | F2 guard | pass | pass |
| `test_bound_resolved_single_pair_is_still_eliminated_at_once` | F2 guard | pass | pass |
| `test_catch_up_window_and_the_window_it_starts_are_not_evidence` | F3 | FAIL (window eligible) | pass |
| `test_catch_up_warmup_still_counts_as_warmup` | F3 | ERROR (no reason field) | pass |
| `test_guard_reason_is_serialized_only_when_set` | F3 contract | ERROR (unknown field) | pass |
| `test_session_load_removes_phone_evidence_and_defers_the_probe` | F4 | ERROR (no `helper_disturbance`) | pass |
| `test_load_on_the_helper_phone_is_reported_and_other_phones_are_not` | F4 `_unified` | ERROR (no `_report_helper_phone_disturbance`) | pass |

### Full suite

| tree | files | tests run | failing files |
| --- | --- | --- | --- |
| pristine (`baserun/`, `tests_base.json`) | 100 | 1484 | `test_resident_router_subset.py` (environmental) |
| fixed (`root/`, `tests_root.json`, final code) | 101 | 1496 (= 1484 + 12 new) | `test_resident_router_subset.py` (environmental), and in that run the known-flaky `test_automated_runtime_admission.py::test_cached_synthetic_refinement_is_below_ten_milliseconds` |

- **Environmental failure.** `test_resident_router_subset.py` fails identically on both trees: it
  compiles `examples/layersplit/ffn-split-resident-router.cpp`, which is outside `research_dev/scheduler`.
- **Same results per file.** Every other file has the same exit and the same test count on both trees.
- **The flaky timing test is not caused by this patch.** The test asserts that a mean of 5
  `generate_automated_candidates` calls stays below 10 ms, and on this machine it sits at the bound:
  - Direct measurement of the timed quantity, interleaved on both trees
    (`analysis/time_refinement.py`): medians 9.0-9.5 ms on both trees, maxima 10.1-10.5 ms on both.
  - It failed 5 of 8 reruns on the pristine tree (`tests_base_admission_rerun*.json`) and 2 of 3 on the
    fixed tree.
  - It passed in the first full fixed-tree run.
- **The other named flaky test,** `test_kv_touch_occupies_exactly_the_cache_pages`, passed in every
  run.

**How it was run:** `run_scheduler_tests.py ROOT OUT.json`, using `/usr/bin/python3` with cwd = ROOT.
Each file runs in its own process, the same way as `run_all.py`, but without the S42 spike list.

**Environment:**
- The copies need llama.cpp's `gguf-py` on `PYTHONPATH` (32 files import `gguf`) and read-only
  `research_dev/spikes` data (5 files). Both are copied into `shared/` and linked into each tree, so the
  main repo is never read at test time.
- The same environment was used for both trees.

## Replay results

### 1. The diagnosis's function-level replay (`scripts/replay_qualification.py --eliminated-from-events`)

Output is byte-identical between the two trees for 000, 001, 002 and 004 (`replay/out/qual_*`).
Both equal `data/replay_*_as_run.txt`. That script evaluates `_bounds`, `_qualifies`,
`_learning_probe_improves`, `_qualification_needs_more_evidence`, the targets and `_update_elimination`
on a copy of the session, so this confirms the fixes did not touch the qualification maths. It cannot
show end states, because it does not run the state machine. That is what the closed-loop harness
below is for.

### 2. Closed-loop decision replay of the dev run (`replay/replay_decisions.py`)

**How it drives the controller.** The harness drives the real `AdaptiveDecodeController` of a tree
through its public API: `start`, `boundary` for every token (with recorded token times), `record_window`,
`acknowledge`, `seal_tail`, and `helper_disturbance` when present. It seeds the history the way the run
did: the input store plus this run's groups that finished before the request started, as
`replay_seeding.py` does.

**Replay and synthetic modes:**
- While the controller's decisions match the recording, it feeds the recorded windows, token times and
  acknowledgements verbatim.
- After the first different decision, each window is synthesized. Its energy per token, domain split,
  evidence ids and token-gap pattern come from recorded windows of the same policy in the same request,
  in recorded order and then cycling the eligible ones.
- A phone window that overlapped a helper-phone load is never used as a synthetic source. When a request
  has no undisturbed window of a policy, the same model and fraction is borrowed from the run's other
  requests. This happens once: 000's P100 is taken from 002 and 004, because 000's only P100 windows
  fell inside the HTP2 load.
- Transitions use the request's median recorded transition length.

**Validation.** On the pristine tree it reproduces the recording:
- 001 and 004 match exactly;
- 000 and 002 match up to the final-token tail handling (divergence at token 137 of 141 and 48 of 51).

| request | tree | first divergence (token) | final | host / phone tokens (recorded) | window energy J (recorded) | guard-removed windows |
| --- | --- | --- | --- | --- | --- | --- |
| 000 | pristine | 137 (tail) | B | 115 / 25 (113 / 25) | 10502 (10493) | - |
| 000 | fixed | 9 | **P100** | 51 / 89 | **8335** | W1, W2 `TOKEN_STREAM_CATCH_UP` |
| 001 | pristine | none | B | 526 / 51 (526 / 51) | 33660 (33835) | - |
| 001 | fixed | 75 | **P100** | 26 / 551 | **28110** | - |
| 002 | pristine = fixed | 48 (tail) | P100 | 11 / 39 (11 / 36) | 2672 (2518) | - |
| 004 | pristine = fixed | none | P100 | 14 / 181 (14 / 179) | 9412 (9412) | - |

Energies are whole-fleet diagnostic energies with ASSUMED_4P5W phone power, summed over the simulated
windows. They are not certified energy.

**Fixed path for 000:**

1. W1 ends in a catch-up burst and W2 starts at W1's late token (F3), so neither is evidence.
2. HTP2 on the same op15 phone is LOADING from 63.9 s to 88.0 s (publication times), so no probe starts
   (F4, tokens 9-45).
3. After the load, P100 is probed at token 49 against 9 steady host windows. P50 is probed at 60.
4. P100 is exploited from token 71 to the end.

**Fixed path for 001:**

1. At token 60 (W15), P100's re-check is inconclusive (targets 4/4, as in the diagnosis), so it goes to
   `resolving_evidence` instead of `INCUMBENT_NO_LONGER_BENEFICIAL`.
2. Three host windows (63-75) and two P100 windows follow.
3. `_finish_verification` qualifies P100 at token 86, and P100 runs to the end.

**Ablation.** Each fix is switched off in-process with `--disable` (`replay/out/ablation_*`):

| request | fixes off | final | host / phone | energy J |
| --- | --- | --- | --- | --- |
| 000 | none | P100 | 51 / 89 | 8335 |
| 000 | F4 | P100 | 19 / 121 | 7425 |
| 000 | F2, F4 (F3 alone of these) | P100 | 19 / 121 | 7425 |
| 000 | F3, F4 (F2 alone of these) | P100 | 26 / 114 | 7447 |
| 000 | F2, F3, F4 | B | 115 / 25 | 10502 (= recording) |
| 001 | F1a or F1b (either alone) | P100 | 26 / 551 | 28110 |
| 001 | F1a, F1b | B | 526 / 51 | 33660 (= recording) |

- **000:** each of F2, F3 and F4 is sufficient on its own.
- **001:** F1a and F1b are each sufficient on their own and switch at the same token (75).
- **Cost of F4 in 000:** holding through the whole HTP2 load costs 000 about 32 phone tokens compared with
  F4 off. This "F4 off" row is optimistic: its synthetic phone windows during the load are undisturbed,
  whereas the real P50 warm-up during the weight upload stalled for 14.8 s.

### 3. Run-5 batch-2 rejections (`replay/replay_no_flip.py`)

**Why a sufficient condition is enough.** The fixes have only these ways to put a request on the phone:
- F1b re-qualification;
- F1a resolution, which ends in `_finish_verification`;
- F2-deferred candidates, which later reach `_select_probe_winner` or `_best_valid_policy`.

Every one of them ends in the unchanged `_qualifies`. The script rebuilds each run-5 session window by
window, including context resets at batch, membership and external-activity changes. It applies F3 (the
tree's own `token_stream_caught_up` on the recorded token times) and F4, clears every elimination, and
evaluates `_qualifies` for every measured candidate at every window.

**Result.** No candidate qualifies at any batch-2 window in any of the 27 requests that measured a
phone policy (000, 001, 002, 003, 008, 009, 015, 016, 022, 024, 025 have batch-2 windows). Run-5 used no historical groups.
The rejections cannot flip.

**Other direction.** F3/F4 could in principle remove phone qualifications. Comparing the same scan with
recorded eligibility (`--no-guards`, `replay/out/no_flip_run5_noguards.json`) and with the guards, every
request keeps its qualifying points. The only changes:
- 006: 38 -> 41.
- 027: 0 -> 61, from token 62. This is the diagnosis's stall artifact inside a batch-1 verification:
  without host W5/W6 (24.7 and 37.9 J/token), P100 qualifies. The run reached P100 only through the
  prior monitor at token 65.

### Windows the guards remove (`replay/out/guard_stats.txt`)

- **dev:** 2 of 224 eligible windows.
  - 000 W1 (F3, -45.8% from its 25 host peers);
  - 000 W4 (F4: P100 during HTP2's weight-read phase; it was *not* slowed, 529.7 ms/token, so this
    removal is conservative).
- **run-5:** 31 of 1803 eligible windows (25 catch-up ends, 6 late starts, 0 F4).
  - 29 of the 31 deviate more than 10% from their peer median (median |deviation| 49%). These are
    006 W9-W15, 007 W29-W40, 021 W37-W60, 027 W5-W6 and 004 W70-W71.
  - The two within 10%: 027 W6 (its "median" of 3 includes the other artifact) and 021 W60 (a late
    start, -7.6%).

## Caveats

- **The replay is a model.** After divergence, windows are resampled from the same policy's recorded
  windows. Transition lengths are medians, and helper availability starts at the request's final layout
  (000's gen-1 to gen-2 helper expansion is not replayed). Energies are ASSUMED_4P5W diagnostics.
- **F3's thresholds (>= 3 burst tokens, burst gap < mean/8, stall > 4x median)** come from two runs.
  Token times are when the stream reader saw each token, not server times. A stream that is bursty by
  design, such as speculative decoding, would lose windows (fail-closed). The better long-term fix
  remains server-side token times.
- **F4 is conservative.** It holds for the whole LOADING state, including weight reads that did not slow
  P100, and marks such phone windows ineligible. It is sampled at window boundaries (start, record,
  acknowledgement), so a load shorter than one window can be missed. A stuck LOADING session keeps probes
  held; that is fail-closed. In coherence mode only the evidence marking applies while the server
  directive decides.
- **F1b can switch host -> phone repeatedly in principle** if evidence oscillates around the threshold.
  Every switch is `_qualifies`-gated and needs new eligible host evidence. No flapping appears in the
  replays.
- **F2 needs the probe budget for its re-test,** including the per-context attempt cap. When it cannot
  be afforded, the old immediate elimination applies.
- **Thin-evidence rejections can still happen through `CURRENT_PAIR_NOT_IMPROVED`** in
  `_finish_verification`, for example a verification pair with one host window. That path was out of
  scope and is unchanged.

## Not implemented

- **F5** (charge exploration per batch context and keep per-batch evidence partitions) overlaps the
  concurrent per-batch-verdict coherence change. The run-5 batch-1 tails (about 372 host tokens) still
  need it.
- **F6** (monotone `sqrt(n)` bands, count historical windows) needs sign-off. It would change the
  qualification maths that this patch deliberately leaves byte-identical.

## Reproduce

Run everything from `$EF`, which is this directory
(`/tmp/claude-1761612022/-home-myid-zs89458-Documents-llama-cpp-release/6fd7e39c-bd31-4e92-b800-5784f94be902/scratchpad/evidence-fixes`).
`S` is the scratchpad directory that holds the `dev/` and `run5/` artifact copies. Use `/usr/bin/python3` for everything.

```sh
# unit tests (fixed tree / pristine tree)
cd root && PYTHONPATH=$PWD:$EF/shared/gguf-py python3 research_dev/scheduler/tests/test_adaptive_evidence_fixes.py -v
cd baserun && PYTHONPATH=$PWD:$EF/shared/gguf-py python3 $EF/basecheck/test_adaptive_evidence_fixes.py -v
# full suite
python3 run_scheduler_tests.py root tests_root.json
# closed-loop replay, ablation and no-flip check
python3 replay/replay_decisions.py --source root/research_dev $S/dev burstgpt_longtail_dev_v1:001 [--disable F1a,F1b]
python3 replay/replay_no_flip.py --source root/research_dev $S/run5
python3 analysis/guard_stats.py root/research_dev $S/dev $S/run5
(cd root && PYTHONPATH=$PWD:$EF/shared/gguf-py python3 $EF/analysis/time_refinement.py $PWD 8)   # flaky 10 ms test
./make_diff.sh    # rebuilds EVIDENCE_FIXES.diff from base/ and root/
```

## Directory contents

- `base/`: the pristine snapshot, code only.
- `root/`: the full working copy with the fixes. It includes the 27 GB of reports and baselines, copied
  at idle I/O priority.
- `baserun/`: base plus links to the same large data, used for pristine test and replay runs.
- `shared/`: `gguf-py` and `spikes`.
- `replay/`: the replay harnesses, with their outputs under `replay/out/`.
  - They import the diagnosis's `scripts/timeline.py` from the main repo, read-only.
  - They are not in the patch. Copy them into the report's `scripts/` if you want them kept.
- `analysis/`: `token_gaps.py` (F3 rule exploration), `guard_stats.py` and `time_refinement.py`.
- Test logs: `tests_base.*`, `tests_root.*`.
