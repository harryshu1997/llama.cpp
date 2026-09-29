# Stable-demand confirmation and bounded replacement retest

Final status: the V2 bounded two-request campaign is PASS 2/2. Stable demand
now confirms without another arrival. The additional READY-poll correctness
regression is fixed. This run does not establish retained-call interruption
bounds: the replacement finished before the first Qwen phone call.

## Scope

Advance demanded residency confirmation on fresh observations without changing
the replacement objective, three-snapshot requirement, minimum residency,
qualification, memory admission, session authorization, or fraction policy.
The existing preparation watcher performs the refresh, including while its
desktop request is queued. No additional watcher or runner policy was added.

The controller checkpoints the last accepted sample timestamp. Repeated,
out-of-order, and debounce-interval samples cannot advance confirmation.
New observations may have identical decision contents. A confirmed candidate
survives minimum-residency deferral until it is proposed or the current layout
wins again. The READY layout remains authoritative until physical verification.

Changed production files:

- `_internal/model_placement_controller.py`
- `_unified/phone_residency.py`
- `_unified/helper_envelopes.py`
- `_internal/adaptive_decode.py` (the physical V1 failure below)

Tests: new `tests/test_phone_layout_confirmation.py` and an added regression
in `tests/test_adaptive_decode.py`. No native, format, loader, catalog, or
qualification implementation changed. No files were removed.

## Software evidence

132 controller/COW/telemetry tests passed. A further 47 confirmation, offline
residency, adapter and replay tests passed. After the physical failure, 90
adaptive/runtime/replay tests passed. These are focused runs, not a new complete
scheduler-suite run; the two replay tests occur in more than one invocation.
There are 267 distinct focused tests (269 executions across these three runs).
Ten regressions were added: nine confirmation tests and one adaptive test.

Both existing golden hashes remain unchanged:

- v3: `ea5b30c9e4a3f705c194188d6e5788c64990953aad73ff3c4b7573bc69eca27d`
- v8: `965f218bb5f81a57dbb51167624798f02eb79354b6cc042683b2b1ed9b1a868d`

The additional stable-demand replay uses the saved v8 catalog, evidence and
request 49. After one submission it introduces no arrival or decode update.

| Elapsed after submission | Before | After |
| --- | --- | --- |
| 0 s | Confirmation 1; background refresh denied | Confirmation 1 |
| 1 s | Still confirmation 1 | Confirmation 2 |
| 2 s | Still confirmation 1 | One-session generation-2 proposal |
| 3 s | Still confirmation 1 | Same proposal; no duplicate |

The generated fixture selects HTP0; production does not name that session.
The READY source and desktop ticket remain unchanged. This decision replay
does not claim physical execution or qualification of its legacy partial route.
The two repeated after-replay files are byte-identical.

- Before JSON SHA256: `618901d227994fb0ac8afad10c572ef564ed2ee37a4c3e44e1c57a19560791a7`
- After JSON SHA256: `540354d77b8c87bcdb2f5eccc4ac925eb44c785aa90002cd085db5e2f2910caa`

Artifacts, commands, source-before copies and focused logs are preserved under
`/home/zhihao/s42-confirmation-liveness-20260907-v1-inputs`.

## Physical V1: confirmation works, request verification fails

Run: `/home/zhihao/s42-confirmation-liveness-20260907-v1-gate/run`.
Qwen 43 arrives at 1 s and Gemma 49 at 91 s. This is a bounded two-request
gate, not another 24-request trace. The original failed run is retained.

Gemma preparation is materialized at 92.836471 s. The Qwen retained-mask drain
is requested then and acknowledged at 94.181185 s. Loading starts at
94.328536 s; PREPARATION_READY is recorded at 135.711488 s. Thus first Gemma
readiness is 44.711488 s after arrival; host load/publication takes 41.382952 s.
This removes the earlier roughly 93 s confirmation wait, not the physical load.
Only HTP1 becomes Gemma generation 2; HTP0 and HTP2 remain Qwen generation 1.

Qwen completes 341 tokens with 4,284 native phone calls. Gemma produces 480
native calls, then fails at received token 84 with
`adaptive verification candidate differs`. There is no successful Gemma
terminal proof, and the run is not PASS or usable for a savings claim.

Cause reproduced with four synthetic windows: after paired verification
rejects a candidate, the controller chooses the baseline and clears the
verification candidate. An unchanged READY poll incorrectly changes EXPLOITING
back to PROBING, retaining the now-finished verification stage. The next
baseline window raises the exact error. READY polling now rearms probing only
when the helper actually becomes available. Identity checks still execute on
every poll, and the original verification check is unchanged.

- FAILURE.json SHA256: `f35a3ba0012d0a8858a94a5d55ce35032efc67ea278832f83287a4f082c829ce`
- FAILURE_REQUEST_HELPER_EVENTS.json SHA256: `27a54e36eae8dedf4b78013031802c5db201dd5685d84ebbf0f2a6885e95730a`
- Source manifest SHA256: `a21c4dc28cbcb8bce890c679a2852decf9a7fa41b5890a5f542aef8655816288`

## Physical V2: bounded campaign PASS, interruption not exercised

Run: `/home/zhihao/s42-confirmation-liveness-20260907-v2-gate/run`.
Inputs, preflight, commands, deployed source manifest, read-only audit scripts
and outputs: `/home/zhihao/s42-confirmation-liveness-20260907-v2-inputs`.
Fresh deployment: `/home/zhihao/s42-confirmation-liveness-20260907-v2-deploy`.
The requests and arrivals are identical to V1. No source changes were made
during either physical run. All original V1 artifacts remain in place.

Preflight PASS with the existing qualified Qwen GPU16 and Gemma GPU22 parents,
unchanged graph-disabled runtime and actual FFN shard indexes. No desktop
requalification, memory relaxation, native rebuild or forced route was used.

### Confirmation and publication

The same candidate receives three distinct, fresh phone observations at host
observation times 91.136445, 94.048604 and 95.011080 s. The confirmation event
is published at 95.111169 s, 4.111169 s after Gemma's arrival. The target is
proposed at 95.111316 s. There are no additional workload arrivals between
submission and confirmation. Confirmation count and debounce remain unchanged.

The earlier matched V6 run took about 93 s from Gemma arrival to confirmation.
This is a diagnostic timing comparison across workloads/revisions, not a
matched performance or energy experiment. The saved-demand software replay
isolates the liveness change without changing demand or execution progress.

| Session operation | Logical layout generation | Physical generation | Native load-to-READY | Host READY publication |
| --- | --- | --- | --- | --- |
| Initial Qwen HTP0 | 1 | 1 | 11.865561 s | 23.354231 s |
| Add Qwen HTP1 | 2 | 1 | 16.321970 s | 40.239642 s |
| Add Qwen HTP2 | 3 | 1 | 12.417935 s | 53.226689 s |
| Selected HTP1: Qwen to Gemma | 4 | 2 | 13.804886 s | 109.670722 s |

Host times are relative to campaign start. Native durations use the phone's
monotonic clock; clocks are not mixed. Replacement DRAINING/LOADING/PREPARING
events publish at 95.297255 / 95.297280 / 95.297389 s. Verification is observed
at 109.399656 s and SESSION_VERIFIED/SESSION_READY publish at
109.670403 / 109.670580 s. Host preparation-start observation to global READY
publication is 14.659558 s; first Gemma readiness is 18.670722 s after arrival.

HTP1 was selected by the scheduler, not named in policy. Its physical epoch
advances 1 to 2. HTP0 and HTP2 retain the exact Qwen artifact, endpoint,
geometry, operator plan and generation 1. There are four physical loads:
three initial shards and one replacement. All use `weight_source=ffn_shard`.
No fraction change reloads weights. The exact one-session authorization is:

- Assignment: `710b1d7e9e797551b0eabe401f2fc7eba8ccea992a5d44f62a522d7b2e60bb65`.
- Source layout: `fb135ecd5cc7f674735af059253b4dd13c0ad95dd2400f16f81c4db0d91f4528`.
- Target layout: `75fc6aebd3b3e93a5ab0b0f41d989eaa8e4b43577ac378771d29c271a6129de4`.

### Requests, coverage and limitations

| Request | Execution start / finish | Output tokens | Phone calls | Weighted eligible coverage | Tokens with positive assistance |
| --- | --- | --- | --- | --- | --- |
| Qwen 43 | 87.730936 / 311.203015 s | 341 | 576 | 9.56% | 14.12% |
| Gemma 49 | 379.431116 / 649.275771 s | 491 | 3,768 | 92.96% | 96.12% |

Both requests have exact execution-ticket/terminal/adaptive-observation proof
matches and pass `semantic-sanity-v1`. Total native, request and per-session
proof counts agree at 4,344. Qwen contributes 288 calls on each retained
session; Gemma contributes 3,768 on selected HTP1/gen2 after READY. There are
zero request recoveries, phone reset recoveries or stale-generation failures.
No fallback or unexpected endpoint restart is observed. The normal desktop
model swap and final campaign teardown are not counted as phone restarts.

Qwen's first token is at 99.906840 s, while Gemma's shard is still loading.
Desktop decode therefore continues during replacement. Gemma's 288.431116 s
arrival-to-execution delay is desktop GPU queuing and model preparation; its
phone shard was already READY 269.760394 s before execution. Initial phone
publication also overlaps Qwen desktop preparation. This does not imply that
desktop model loading or GPU queuing have been eliminated.

Qwen's first positive fraction is applied at 111.021423 s, after replacement.
There are zero retained phone calls before or during that replacement and
288 per retained session afterward. The unchanged
`s42-retained-session-call-gap-v2` audit returns INSUFFICIENT for both sessions,
with no equivalent transition intervals. This is neither a measured bound
violation nor a passing retained-serving interruption test. No new reverse
replacement or injected physical rollback was run in this confirmation gate;
the focused COW regressions remain green.

Coverage uses current exact observation-group hashes, not older groups with
the same request ID. Both first eligible tokens are token 1. Denominators are
340 and 490 tokens; weighted numerators are 32.5 and 455.5 tokens. Positive
policy token counts (48 and 471) also match native calls divided by the active
12-layer Qwen / 8-layer Gemma masks. Fraction is relative to the supported
shard mask, not the whole model's compute. The final two-token server-release
guard tails use the final applied policy and are marked unmeasured for window
qualification; terminal phone calls still cover their execution.

Qwen explores all nonzero fractions and finishes with the existing baseline
selection; Gemma explores and exploits 100%. No fraction-policy optimization
was included. Low Qwen useful coverage and the previously reported attached
requests that never probe remain follow-up work. Do not interpret Gemma's
high coverage as a measured fleet-energy saving.

All 5,640 saved planning observations are VALID, maximum age 2.599079 s:
5,638 HTTP fallback snapshots during FunctionFS and two initial ADB snapshots.
Four load admissions are fresh and valid. No sampled telemetry outage is
observed in this rerun. No 24-/84-request trace or new baseline was run, and
there is no new savings claim from this unmatched two-request experiment.

### Hashes and changed files

- RESULT.json: `8c93cf2262526928e3befae24a27e74a5148b36b22444991b624527a383e9feb`.
- SOURCE_MANIFEST.json file: `e723644b9f1cd6df3a3aabbdaae717208a758c7637ece1592e662d531ec25132`.
- Source manifest canonical identity: `a8b1650519781fd32b3b575bbe5ffef9efd0f6c1fdca520ba4fe6d471ce372e7`.
- Capability catalog: `667499be3aa638ff93f818f55618341ce161f4b6459af45a42f35449d661f75d`.
- ADAPTIVE_DECODE_OBSERVATIONS.json: `5e1d3f698a2a22bf34f7c059045408035aea3e75bff5684efc9bc7f3b296df95`.
- RUNTIME_AUDIT.json: `3a3a2995104603b8f583b5002b38450c6685a7645264cac7d0171a8fd71f2ce8`.
- EXACT_AUDIT.json: `63d1b0c33d5d770cdb6fd5f9fd8bf9c327f3eaf1d30b9852f05196c156560183`.
- CONFIRMATION_AUDIT_V2.json: `f5f7b4726adcf802cdd00c1cf05d9616bb5f48727fe55c1bb6162516e4c472c0`.
- TELEMETRY_AUDIT.json: `e0659b6cb9b07c0cf0b36d28bfa1e30c985441cae15818e855232d01306353ba`.
- RUN_ARTIFACT_HASHES.json: `6a894eb65bbc66ad2b51a36fa36e712c5ab2468f10e4720caa7cfac3b8615def`;
  covers 11,298 files, 758,728,791 bytes in the completed run.

Exact repository files changed in this task, excluding pre-existing changes:

1. `research_dev/scheduler/_internal/model_placement_controller.py`
2. `research_dev/scheduler/_unified/phone_residency.py`
3. `research_dev/scheduler/_unified/helper_envelopes.py`
4. `research_dev/scheduler/_internal/adaptive_decode.py`
5. `research_dev/scheduler/tests/test_phone_layout_confirmation.py` (new)
6. `research_dev/scheduler/tests/test_adaptive_decode.py`
7. `research_dev/scheduler/campaigns/burstgpt/reports/20260907-confirmation-liveness/README.md` (new)
8. `research_dev/talks.md`

Analysis scripts and commands in the fresh artifact directories are not
scheduling policy. No unrelated dirty-worktree files were removed or edited.
No commit, push, PR, GDM stop, sudo or unrelated process interference occurred.
