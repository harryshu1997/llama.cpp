# Pixel second-phone integration, 2026-09-24

User requested the current scheduler results, Pixel integration and a matched
two-phone trace. This supersedes the earlier qualification-only scope. No
commits or pushes. Existing private work and pending scratch patches retained.

Latest: **v7 completion PASS, Pixel activation FAIL**. 9/9 requests completed,
863.663s, host61.905kJ, Pixel0calls. Two completed activation failures; retries
stopped and cleanup PASS. Final logs identify a primary-phone transition mask
mismatch; independent candidate-pruning reproducer is retained. See the final
checkpoint below. No incremental Pixel trace saving is verified.

## Current scheduler result

Single `longtail_dev_v2` pair: desktop with dispatcher 82.323 kJ / 865.004 s;
OP15 all-on 45.686 kJ / 811.949 s. Host energy reduction 44.504%, time reduction
6.133%. Completion PASS, strict saved-token identity FAIL 7/9. Phone energy
is assumed, separately reported. See
`../20260924-coherent-policy-coalesced/RIG_RESULTS_ALLON.md`.

The earlier legacy-dispatcher desktop reference was 96.75 kJ. The all-on
45.69 kJ arm is 52.8% below that older reference, combining dispatcher and
offload changes. The best exact-output development arm remains coherentEF2
at 69.64 kJ, about 28.0% below the legacy reference, with 9/9 outputs identical.
These are nine-request development measurements; they are not the full
31-request evaluation trace.

## Packed Pixel server prerequisite

Weight identity PASS: all 1,604,321,280 stored values in 18 tensors match the
desktop F16 parent after dequantization and rounding. This is exhaustive;
the earlier packed-weight manifest sampled 32 rows per tensor. Execution
arithmetic remains a separate qualification.

The first server attempt FAIL: the private AOA microbenchmark worker's
60-second idle timeout expired during desktop startup (78.65 s). No Pixel
FFN call completed. After the failed request ended, the server was idle and
stopped with SIGINT; its own ADB forward was removed. No active worker was
killed. Evidence is retained as `run1/FAILURE.json` and
`run1/POSTFAIL_CLEANUP.json`.

Retry PASS using the preserved packed CPU worker without that microbenchmark
idle timeout. Six pinned CPU threads, paired SDOT, fused residual correction,
dynamic 64-row scheduling; layers 18-23, columns 8,704 or 17,408, ADB TCP.
The worker held the Pixel lock and exited normally at its finite call budget.
All four 64-token outputs are identical; 744 served calls plus 24 cleanup
calls outside measurement. Server and worker exits are both zero.

| Arm | Request s | Decode s | Host request J | Saving vs mean control |
| --- | ---: | ---: | ---: | ---: |
| Desktop before | 41.088 | 38.362 | 4792.491 | reference |
| Pixel half width | 44.419 | 41.747 | 4308.755 | 10.887% |
| Pixel full width | 49.127 | 46.385 | 4324.380 | 10.563% |
| Desktop after | 41.064 | 38.326 | 4877.785 | reference |

These are one request per split, not a trace or an OP15 + Pixel comparison.
Half/full decode is 8.875%/20.971% slower than the mean control. Pixel energy
was not measured. The packed server check predates the multi-helper rebuild;
the mechanism gate below must qualify the rebuilt binary.

Evidence: `../20260922-fast-path-M3/physical/pixel10pro-packed-server-20260924-v1/`;
desktop `/mnt/storage/s42-pixel10pro-packed-server-20260924-v1`.

## Integration work in progress

- Existing native multi-helper code was in main but absent from the deployed
  binary. Rebuilt `llama-server`: PASS, 17 S41SERVERFFN strings and the helper
  environment/marker present. Prior binaries and sources preserved in
  `/mnt/storage/s42-two-phone-pixel-20260924-v1/before-native-build`.
- Refreshed base transport identity and generated
  `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json`. Old
  coalesced identities in existing input directories are stale after this
  rebuild and must not be used for new runs.
- Optional rooted ADB worker launch and phone lock: 58 targeted tests PASS
  in an isolated tree. Full suite: 1,955 tests across 136 modules passed after
  one missing scratch fixture was restored and that module rerun. The original
  suite exit 1 and successful fixture rerun are both retained. This preserves the default command
  and retains the existing finite-budget drain and idle-only stop rules.
- `GATE_CONFIG.json`: OP15 layers 0-17, Pixel packed CPU layers 18-23.
  Planned arms: desktop, OP15 full width, two phones at 75% (matched total
  columns x layers), two phones full width (additional capacity).
- AOA is not wired into the native server. Stock accessory uses contiguous
  v6 frames; FunctionFS uses padded DMA-BUF frames and shutdown semantics.
  The initial combined qualification uses the existing direct ADB TCP path.
- Campaign lifecycle, Pixel receipts/cost evidence and preflight now pass.
  Two trace attempts now completed but failed Pixel activation; the latest
  checkpoint below records the blockers. No incremental trace saving yet.

## Combined mechanism gate

Correctness and cleanup PASS. All five single-request 64-token outputs match.
The four-request desktop, OP15 and two-phone arms also match all four outputs
(256 output tokens per arm), with prompts 256/320/384/448 tokens. OP15 owns
layers 0-17; Pixel owns 18-23. Each combined arm has 1,098 OP15 calls and
366 Pixel calls; B4 has 4,338 OP15 rows and 1,446 Pixel rows, proving coalescing.
Every Pixel worker exhausted its finite budget and exited 0, with forwards
removed and boot unchanged. Draining calls are outside energy measurement.

| Active requests | Arm | Request s | Host request kJ | Saving vs desktop | Pixel increment vs OP15 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Desktop, mean of before/after | 41.611 | 4.847 | reference | |
| 1 | OP15 full | 34.895 | 2.414 | 50.2% | reference |
| 1 | Both, matched total FFN work | 35.962 | 2.627 | 45.8% | -8.8% |
| 1 | Both, full width | 42.259 | 2.048 | 57.8% | +15.2% |
| 4 | Desktop | 49.629 | 6.669 | reference | |
| 4 | OP15 full | 47.755 | 3.214 | 51.8% | reference |
| 4 | Both, full width | 71.202 | 3.452 | 48.2% | -7.4% |

Performance: additional host saving PASS at B1 full width, FAIL at B4 and
matched-work B1. Full-width Pixel adds 21.1% request latency at B1 and 49.1%
at B4 relative to OP15. These are individual bounded requests, not trace
savings; startup costs differ and phone energy is not measured. The table
uses complete request intervals. Decode-only windows begin at the control
acknowledgement in assisted arms and first token in controls, so they are not
the headline comparison. Full raw paid/startup spans remain in RESULT.json.

Evidence: MECHANISM_R1_COMPARISON.json, MECHANISM_B4_COMPARISON.json and
physical/mechanism*. Campaign lifecycle and qualification wiring remains in the isolated
/tmp/s42-pixel-campaign-20260924/root tree. Completed trace attempts and their
activation failures are recorded below; this integration is not merged.

## Campaign qualification checkpoint, 2026-09-24 19:01 UTC

- TCP calibration r1 FAIL: B2 execute header counted one row. Fixed elements
  to 5120 x rows. Worker drained its remaining finite budget, exited 0.
  Fresh r2 PASS: 216 calls, rows 1/2/4, half/full width, FNV framing checked.
  Warm median non-compute RPC overhead: 11.836/15.987/20.528 ms for B1/B2/B4.
  This timing check does not replace independent numerical qualification.
- Resident lifecycle r1 FAIL: rooted Android ps hid the executable directory;
  the old PID matcher missed the worker. Manual TERM was sent only after both
  clients disconnected, no established socket, and root readlink verified
  the exact executable. Raw r1 RESULT says PASS after external cleanup; its
  ASSESSMENT.json overrides that verdict. It is not qualification evidence.
- PID matcher now verifies /proc/PID/exe for rooted basename entries. Fresh
  resident lifecycle r2 PASS: two HELLO connections, stop refused while each
  was connected, automatic idle TERM, exit 0, forward removed, boot unchanged,
  no worker remaining. Local real-worker lifecycle tests: 7 PASS.
- Scratch campaign catalog resolves with separately pinned helper evidence
  and power. Full module suite: 1,788 tests reported, one timing assertion
  FAIL (10.011 ms vs 10 ms); isolated module rerun 10.376 ms. The unchanged
  main tree also fails that assertion at 10.376 ms. No threshold changed.
  Latest overlay/launch checks: 51 PASS with one skip; pyflakes 18 files PASS.
  Full trace and energy comparison remain pending.

Physical campaign preflight r1 FAIL before inference: duplicate phone-state
check IDs when iterating two phone executors. Fixed per-device check IDs for
multi-phone catalogs, preserving legacy single-phone IDs. The endpoint
validator also now accepts a phone split-helper's exact physical placeholder;
other non-HTTP endpoints remain rejected. Regression suite for these checks:
27 PASS; all 19 changed Python files pass pyflakes. Fresh r2 preflight is
running. Separate transport-admission check PASS: Qwen 40,960-byte/four-row
and Gemma 61,440-byte/eight-row qualified capacity, one coalesced helper per
desktop parent, and Pixel declared only for Qwen.

Physical campaign preflight r2 PASS at 2026-09-24 19:19:12 UTC. First real
nine-request `longtail_dev_v2` two-phone campaign started with fresh v3 inputs,
output `inputs-two-phone-v3/run-r2` under the remote root. Completion, energy,
strict tokens and matched new controls are pending.

Campaign `v3/run-r2` FAIL at 19:21:50 UTC before Pixel FFN execution: the new
energy callback sliced a missing co-helper binding on an OP15-only Gemma
window. Corrected the absent-binding fallback and added coverage for both
OP15-only and multi-phone windows: 108 targeted tests PASS, pyflakes PASS.
Pixel cleanup PASS: idle TERM, exit 0, boot unchanged, forward removed, no
worker. Raw failure and logs are under `physical/campaign-run-r2`. Fresh v4
inputs were generated for the retry and both controls; no completed trace
energy or strict-token comparison exists yet.

Campaign `v4/run-r1` FAIL at 2026-09-24 19:32:13 UTC after preflight PASS:
priority compaction attempted to release a replanned reservation after the root
was legitimately deferred by residency projection. Gemma completed 2,304 OP15
calls, including 456 multi-row calls; Pixel still had zero FFN calls. The preceding
optional-helper callback failure did not recur. Pixel cleanup PASS: idle TERM,
exit 0, no worker/forward, unchanged boot. Raw output is
`physical/campaign-v4-run-r1`. A focused regression reproduces the exact error
before a three-line guard; after it, the deferred root and follower both complete.
Related suite: 147 tests PASS; changed files pass pyflakes. Fresh v5 retry is
preparing. These are two different trace failures, with no trace energy result.

Campaign `v5/run-r1` preflight PASS19:46:13UTC, trace FAIL19:49:13UTC.
Qwen003 failed with "qualified desktop baseline is not available" after Qwen002
preparation; saved snapshot shows Qwen resident/healthy and no quarantine.
Pixel made zero FFN calls. Cleanup PASS, exit0, no worker/forward, unchanged boot.
Full output is `physical/campaign-v5-run-r1`. An offline six-request fixture
reproduced long-request replanning rejection behind later model replacements
(`MEMORY_REPLACEMENT_CONFLICT_CURRENT:gpu-memory`). Using existing priority
compaction for `preparation_phase_completed` as well as early completion lets
the long same-model request join at load completion and keeps the later switch
after it. Regression PASS; targeted suite 147/148 PASS, known timing assertion
FAIL10.040539ms vs10ms. Failure messages now retain the baseline rejection reason.
Fresh v6 inputs and preflight started19:57:08UTC. The hardware cause still needs
confirmation; stop if this same baseline-rejection failure repeats.

## First completed campaign and activation repair, 20:20 UTC

`v6/run-r1`: completion PASS9/9, 924.684964s, measured host61.351856146kJ.
Coverage FAIL: Qwen0/4 assisted and Pixel0calls. OP15 assisted only Gemma005
(14,520 calls) and007 (600). Both scheduler crash fixes survived this run.
Pixel cleanup PASS exit0, no worker/forward, unchanged boot. Raw output:
`physical/campaign-v6-run-r1`. Exact-token comparison and fresh controls pending.

The actual catalog combined OP15's 512-column minimum with Pixel's fixed
4,352-column block using LCM, producing8,704. Requiring quarter and three-quarter
widths then eliminated the entire split family. All four saved Qwen admissions
have zero operator_split candidates. The correction preserves the primary
minimum and tests helper divisibility on the chosen runtime quantum. For Qwen,
the intended grid is4,352 (four partitions). A non-divisor-minimum regression
fails before and passes after. Validation:167 helper/route/adaptive tests PASS,
9 catalog tests PASS, pyflakes26 changed files PASS. An initial suite invocation
included nonexistent test_runtime_catalog; its loader error remains in the log,
and the correct test_catalog_materialization passed separately. Deployment and
fresh v7 preflight began20:19:40UTC. Pixel trace benefit remains unverified.

Energy audit: v6 OP15=1.458508215kJ and Pixel=0.824508303kJ, both **assumed**.
The historical comparator includes only the OP15 domain in its fleet total.
`compare_campaigns.py` retains its token checks and host comparison while summing
all phone domains, with the assumption explicit. Host energy is unchanged.

`v7/run-r1` preflight PASS at 20:26:38 UTC; the trace started immediately.
Actual Qwen catalog verification PASS: four split coordinators each have
17,408 columns, quantum 4,352, four partitions (`ACTUAL_QWEN_V7_GRID.json`).
The newer model-affinity changes in the main tree are absent from this isolated
run; matched controls will use the same isolated snapshot. Main-tree integration
will need a rebase of `replan.py` and its dispatcher tests to preserve that work.

## Latest checkpoint, 2026-09-24 20:48 UTC: activation FAIL, hardware retries stopped

| Gate | Result | Evidence |
| --- | --- | --- |
| v7 preflight | PASS | Completed 20:26:38 UTC |
| v7 trace completion | PASS | 9/9 requests, 20 attempts, 863.662723 s |
| v7 measured host energy | Measured | CPU 36.086450 + GPU 25.818873 = 61.905323 kJ |
| Pixel trace activation | FAIL | Zero calls; Qwen 0/4 requests assisted |
| OP15 execution | PASS | Gemma 3/4 requests assisted, 15,936 calls |
| Exact tokens vs previous desktop+dispatcher | FAIL | 7/9; Gemma000 first differs at token19, Qwen004 at111 (zero-based) |
| Pixel cleanup | PASS | Idle TERM, exit0, no worker, forward removed, boot unchanged |
| Incremental Pixel saving | NOT VERIFIED | No fresh OP15 or desktop control launched |

Raw output: `physical/campaign-v7-run-r1`; inputs and preflight are alongside it.
`V7_RESULT_ASSESSMENT.json` records the checks. OP15 battery notify code was0;
no charger intervention was needed. Both completed integration attempts (v6/v7)
failed the same zero-Pixel activation gate. Their underlying defects differ;
we conservatively stopped further hardware retries under the handoff limit.
The rig run ended normally; no campaign or phone worker was force-killed.

The older desktop+dispatcher arm used82.322534kJ/865.004s. Its request inputs
match v7, but its build/code differ. The diagnostic reduction is24.801%, with
7/9 outputs identical. This is **not a matched Pixel benefit**: Pixel executed
nothing, and the better prior OP15-only all-on arm remains45.685598kJ.
`V7_VS_PREVIOUS_DESKTOP_DIAGNOSTIC.json` explicitly marks that limitation.
Both phone energy domains are assumed: OP15 1.417301kJ, Pixel0.771857kJ;
64.094481kJ is a modeled fleet total, not measured fleet energy.

### Remaining blockers and next validation

1. The final live log contains225 `PREPARATION_FAILED` events with
   `physical_helper_preparation_failed:exact partial phone residency transition is unavailable`.
   It is attempting an OP15 Gemma-to-Qwen shard replacement; the first attempt
   targets Qwen layers0-5 on HTP0 while retaining Gemma on HTP1/HTP2.
   Code review finds that `phone_session_ops/replacement.py` compares the full
   `phone_ffn_resident_contract` mask (including Pixel layers18-23) with the
   OP15-only shard mask. This ownership mismatch explains the refusal; a focused
   transition regression is still required before changing the check. Preserve
   authorization, generation, shard identity and transport validation. Apply the
   primary-phone contract consistently to startup/reuse/replacement and stored
   direct-phone state; keep the combined mask for server dispatch and proofs.
   The earlier projection in `_persistent_phone_residency_state` happens too late
   to fix this guard.
2. Independently, bounded route selection can keep unusable `operator_offload`
   representatives and remove the `operator_split` representatives needed for
   adaptive probing. `diagnose_frontier.py` on the saved catalog/snapshot gives
   zero policies with current selection and four with a diagnostic split
   preference (4,352/8,704/13,056/17,408 columns; OP15 0-17, Pixel18-23).
   This is a fresh-scheduler counterfactual, not a full queue replay or hardware
   qualification. The actual admission log does contain two one-session split
   candidates, so candidate pruning alone is **not** the live run's root cause.
   Large split envelopes are displaced by offload representatives in the
   mandatory groups. Rough costing also applies phone links during prefill and
   produces missing-link sentinel costs; audit it against decode-only execution.
3. After regression tests, run a short Gemma-to-Qwen transition smoke with a
   retained OP15 session and actual Pixel FFN calls. Require ownership proofs,
   token checks and cleanup before paying for another trace. Then run fresh
   OP15-only, OP15+Pixel and desktop arms on the same code/build/trace. Expand to
   the31-request evaluation only after the development comparison establishes
   a benefit and reports its output-correctness result.

The26-file integration remains isolated. `STAGEA_V7_REBASED_REVIEW.diff`
three-way merges the other session's newer affinity changes in `replan.py` and
its tests, and `git apply --check` passes against main. It has not been applied.
Rebased validation:245 targeted tests,244 passed initially and one lacked a
report fixture; that test passed after restoring the fixture. An earlier launch
lacked the gguf-py symlink and is also retained. All26 changed Python files pass
pyflakes. The original full suite's known10ms performance-threshold failure
remains reported in `STAGEA_TEST_SUMMARY.json`. The rebased tree has no hardware
measurement; v7 used the original isolated snapshot.
