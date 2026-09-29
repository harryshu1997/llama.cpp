# Pixel 10 Pro second phone, integration 2 (2026-09-24)

This work takes over from the Pixel agent. Its report is `../20260924-pixel-second-phone/`. That agent
stopped at 20:48 UTC after two trace runs in which the Pixel made zero FFN calls. In this round:

- the Pixel agent's Stage A integration is merged into main;
- both activation blockers are fixed, each with fail-before/pass-after tests;
- two phones were run on real traces.

Nothing is committed or pushed. Both phones were used, with the user's authorization. The progress
log is `PROGRESS.md`.

## Headline

- **Activation works.** The OP15+Pixel route now runs end to end on real traces:
  - the Pixel served 1,998 (dev_v1), 1,926 and 1,104 (dev_v2) Qwen FFN calls;
  - every Qwen request was assisted by both phones;
  - there were 0 `PREPARATION_FAILED` events (v7 had 225);
  - OP15 re-provisioned Qwen -> Gemma -> Qwen, including the v7 partial case (HTP1 on Qwen while HTP0
    and HTP2 still hold Gemma), and every session reached SESSION_VERIFIED.
- **The Pixel does not help (energy or time).** Two matched dev_v2 pairs from the same deploy and tree:
  | | OP15+Pixel all-on | OP15-only all-on | Pixel effect |
  | --- | ---: | ---: | ---: |
  | Mean host energy | 41.34 kJ | 39.83 kJ | +3.8 % |
  | Mean duration | 744 s | 635 s | +17 % |
  | Pairwise host energy (pair 1 / pair 2) | | | +4.2 % / +3.3 % |

  The cleanest view is the Qwen phase. In the OP15-only run that assisted all 4 Qwen requests, the
  window took 10.3 kJ / 135 s. The two-phone runs took 12.8-13.8 kJ / 161-189 s.
- **Why:** per layer, the Pixel's packed-CPU worker over ADB TCP takes 38 ms at B1 and 82 ms at B4.
  OP15 takes 9.7-12.7 ms, and serves 18 layers to the Pixel's 6. Layers run in sequence, so at B4 the
  Pixel's six layers take about 490 ms per step against OP15's eighteen at about 218 ms.
- **The longtail_v1 pair (iii) was not run.** Step (iii) was conditional on the Pixel helping.
- **Phone energy is an assumed model** (4.5 W active / 0.875 W idle, never measured) and is reported
  separately from the host energy. Host energy is measured: RAPL package + NVML board.

## 3. Hardware (step 3)

All runs came from the merged main tree, synced to `/mnt/storage/s42-trace-v2-20260921-prep/source`:

- scheduler digest `sha256:5e351e70...`;
- server `libllama-server-impl` a87e7772;
- identity `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json`.

The Pixel worker and binaries are the Pixel agent's qualified packed-CPU worker over ADB TCP (rooted, 6
pinned threads, automatic idle-TERM), with the evidence re-materialized by `prepare_campaign_int2.py`.
Its receipts are sha256-verified copies of the Pixel agent's.

Every phone arm used:

- coherence (`server_policy_coherence`);
- coalesced batch plans for both models;
- #4 re-provisioning (`phone_resident_model_reprovisioning: {}`);
- the dispatch policy (`work_conserving_admission` + `model_affinity`).

The desktop arm is `desktop-baseline` with the same dispatch policy.

Each chain ran under one `flock -w 7200` on the rig lock, held across sync, prepare, preflight and run
(`run_chain_int2.py`; status files in `physical/chains/`). Batteries were logged before and after every
preflight and every run. OP15 `battery_notify_code` stayed 0 throughout (level 80 -> 77 %); the Pixel
stayed at 100 %. The Pixel worker always stopped cleanly: idle TERM, exit 0, forward removed, boot id
unchanged, no worker left. After the last run: no Pixel worker, no adb forwards, lock free.

### (i) Gemma->Qwen transition test, `longtail_dev_v1` (6 requests)

| arm | dur s | host kJ (meas.) | vs desktop | assumed OP15 / Pixel kJ | FFN calls OP15 / Pixel | identical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| desktop + dispatcher | 714.5 | 64.78 | ref | 0.63 idle | - | ref |
| OP15 + Pixel all-on | 785.8 | 32.48 | -49.9 % (dur +10.0 %) | 1.94 / 1.41 | 20,541 / **1,998** | 5/6 |

What the two-phone run showed:

- **Assistance.** Qwen: 3/3 requests (000, 002, 004) assisted by both phones. Gemma: 2/2, on OP15 only.
  All at full width.
- **Proofs.** The Pixel's calls are under session `PIXEL10PRO0`, with request ids from 16,777,217
  (helper-1 range), and they cover layers 18-23. The server registered `helpers=2`: op15 mask 262143,
  pixel10pro mask 16515072.
- **OP15 layout timeline.**
  | Time | Change |
  | --- | --- |
  | 5-103 s | Qwen loaded on HTP0-2 |
  | 193-229 s | HTP1, HTP2, HTP0 switched to Gemma |
  | 488-533 s | HTP1, HTP0, HTP2 switched back to Qwen; this is the v7 failing pattern, and it now passes |
- **Outputs.** Qwen 004 first differs at token 149 of 196, a late near-tie flip. The other 5 outputs are identical.
- **Duration.** The +10 % is mostly the second Qwen load. It took 149.8 s against 78.7 s, and request
  004 waited 157 s for its start while OP15 re-provisioned (`DEV1_ALLON_STYLE.md`).

### (ii) Matched `longtail_dev_v2` arms (9 requests)

The two phone arms were run twice, with the order counterbalanced: ii1 ran desktop -> OP15 -> two-phone,
ii2 ran two-phone -> OP15.

| arm | dur s | CPU kJ | GPU kJ | host kJ (meas.) | vs desktop | assumed phones kJ | identical vs desktop | Qwen / Gemma assisted | FFN calls OP15 / Pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| desktop + dispatcher | 682.2 | 47.00 | 21.04 | 68.04 | ref | 0.60 (OP15 idle) | ref | - | - |
| OP15 all-on r1 | 617.4 | 21.35 | 19.47 | 40.82 | -40.0 % | 1.38 | 7/9 | 1/4, 3/4 | 17,147 / 0 |
| OP15 all-on r2 | 652.1 | 18.47 | 20.37 | 38.84 | -42.9 % | 1.54 | 8/9 | 4/4, 2/4 | 23,665 / 0 |
| OP15+Pixel all-on r1 | 806.2 | 18.51 | 24.03 | 42.54 | -37.5 % | 1.86 + 1.14 | 6/9 | 4/4, 3/4 | 21,513 / 1,926 |
| OP15+Pixel all-on r2 | 681.5 | 19.29 | 20.85 | 40.13 | -41.0 % | 1.64 + 0.88 | 5/9 | 4/4, 3/4 | 19,088 / 1,104 |

Pixel increment over OP15-only (host = measured; fleet = host + assumed phones):

| comparison | host | duration | modeled fleet |
| --- | ---: | ---: | ---: |
| pair r1 | +4.2 % | +30.6 % | +7.9 % |
| pair r2 | +3.3 % | +4.5 % | +5.6 % |
| means | +3.8 % | +17.2 % | +6.8 % |

Qwen execution window (`DEV2_MODEL_WINDOW_ENERGY.json`: host energy from the first Qwen execution start
to the last Qwen end; no other large model runs inside it):

| arm | window s | host kJ | host W | Qwen phone assistance |
| --- | ---: | ---: | ---: | --- |
| desktop | 145.2 | 17.70 | 121.9 | none |
| OP15 r1 | 140.5 | 15.44 | 109.9 | 1/4 (the controller kept 002/003/006 on the host) |
| OP15 r2 | 135.4 | **10.31** | 76.2 | 4/4 (B4 1,734 OP15 calls) |
| OP15+Pixel r1 | 189.1 | 12.84 | 67.9 | 4/4 |
| OP15+Pixel r2 | 160.6 | 13.76 | 85.6 | 4/4 |

Per-layer RPC in ms (Qwen server `S41SERVERFFNSHAPE`, full width 17,408):

| helper | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: |
| op15 (layers 0-16/17, FunctionFS) | 9.7-9.9 | 10.0-10.1 | 10.6-10.9 | 12.1-12.7 |
| pixel10pro (layers 18-23, adb-tcp) | 38.1-38.4 | 53.8 | 70.5-71.5 | 81.6-82.3 |

Pixel compute alone is 29.3 ms at B1 and 70.6 ms at B4.

How to read these numbers:

- **The Pixel costs energy as well as time.** With the Pixel, the host draws less power (CPU energy
  drops), but the Qwen batch lasts longer. Against the OP15-only run that also assisted all 4 Qwen
  requests (r2), the two-phone Qwen phase used 25-33 % more host energy and 19-40 % more time.
- **Most of r1's extra time was model loads, not the Pixel.** Two-phone r1 lost about 100 s there: the
  Qwen hot load took 114 s against 46.6 s, and the Gemma reload 90.7 s against 62.8 s. The r2 loads were
  65.0 / 67.1 s, so this is page-cache noise.
- **The controller spends probe budget on slow policies.** At B4 the two-phone arm logged 109
  `SERVER_PROBE_BUDGET_EXHAUSTED` verdicts: the controller keeps probing the slow two-phone policy. Stage
  A cannot choose "OP15 only" for Qwen: every phone route of a model with a co-helper uses both phones
  (GAPS_README "Known properties").
- **Token differences follow near ties, not the Pixel.** The differences sit at fixed positions across
  runs: Qwen 004 @57 (OP15) or @111 (two-phone), Gemma 000 @19, Gemma 005 @41, Qwen 003 @74. In
  two-phone r1, Gemma 007 differs from token 1, and Gemma is never Pixel-served. Every output keeps its
  full length.

### (iii) `longtail_v1` pair: not run

The condition was "only if the Pixel arm helps". It did not: +3.3 % / +4.2 % host energy in both matched
pairs.

## 4. Deliverables in this directory

| Path | Content |
| --- | --- |
| `PIXEL_STAGEA_ON_MAIN.diff` | Pixel Stage A v7 rebased on main (merged) |
| `ACTIVATION_FIXES.diff` | fixes (a) + (b) + `tests/test_two_phone_activation.py` (merged) |
| `ACTIVATION_TESTS_{BEFORE,AFTER}.log`, `SUITE_*.log` | test evidence |
| `frontier/` | v7 frontier before/after (`diagnose_frontier.py` output), single-phone all-on replay before/after, `frontier_groups.py`, `replay_frontier.py` |
| `prepare_campaign_int2.py`, `run_chain_int2.py` | input derivation (from the Pixel agent's script) and the locked chain runner |
| `analyze_int2.py`, `interval_energy_int2.py`, `dev2_post.sh` | per-arm accounting (energy, calls per device and session, layout timeline, token identity) and per-model window energy; the last two run on the desktop |
| `DEV1_*`, `DEV2_*` | analysis outputs (`*_ALLON_STYLE.md` = loads, dispatch and pairs via `../20260924-coherent-policy-coalesced/analyze_allon.py`) |
| `physical/` | inputs, preflight, battery logs, RESULT.json, streams and Pixel worker logs of all 7 runs; chain status files |

The desktop root is `/mnt/storage/s43-pixel-int2-20260924/` (templates, inputs, full run dirs incl.
server stderr and decision logs). The Pixel agent's directories were not modified.

## 5. Caveats and open items for the user

- **Single runs only.** The dev_v2 phone arms have 2 runs each; dev_v1 and the desktop control have 1.
  Run-to-run variance is large. The two two-phone dev_v2 runs differ by 125 s and 2.4 kJ, mainly because
  of model loads.
- **Phone energy is assumed, not measured.** Both phones use the 4.5 W / 0.875 W model, and the Pixel
  idle charge applies for the whole trace.
- **Rough costing still uses prefill payloads for decode-only envelopes.**
  (`costing_rough._rough_transfer_cost`: 2^40 us sentinels when a prefill payload exceeds a link's
  maximum.) Fix (b) makes adaptive routes survive regardless. Correct decode-only rough costing, which
  would mirror `_decode_split_candidate`, would change single-phone rankings and was left out.
  Exact costing also treats a co-helper's offload assignment as local to the Pixel for all phases; that
  only affects the unused offload candidates.
- **No backoff for repeated preparation failures.** v7 retried the same failing preparation 225 times.
  The failures are fixed, but a retry backoff for identical failures does not exist.
- **Your decisions:**
  - For a Pixel benefit, the options are (1) a faster Pixel worker at B >= 2 (current B4 is 82 ms per
    layer), (2) Stage B per-device policies, so the controller can drop the Pixel at B >= 2 and keep it
    at B1, where the mechanism gate showed +15 % host saving, or (3) more or other layers on the Pixel.
  - Whether to still run the `longtail_v1` pair (about 3 h for desktop / OP15 / two-phone) despite the
    dev_v2 result.
  - A review/commit checkpoint. Nothing is committed; main now contains both merged diffs.

## 1. One tree (step 1)

| Item | Result |
| --- | --- |
| Pixel isolated tree | `/tmp/s42-pixel-campaign-20260924/{base,root}`, with root = base + `STAGEA_V7_INTEGRATION.diff` (checked with `git apply -R --check`). V7 is cumulative: BEFORE_PREFLIGHT, STAGEA, V5, V6 and REPLAN_CORRECTIONS are all inside it. |
| `ROOTED_WORKER_LIFECYCLE.diff` | Already in main: all 3 files hash to its `after`. |
| Rebase | 26 files. 24 were unchanged in main since the Pixel base and were taken as-is. `replan.py` and `test_dispatch_policy.py` changed on both sides. Both went through a `git merge-file` three-way merge and merged cleanly: the hunks are disjoint (main's affinity follow-up `AFFINITY_DISPLACEMENT_REASON`, and the Pixel side's `preparation_phase_completed` compaction plus its `REPLAN_REQUIRED` early return). The result is byte-identical to the Pixel agent's `rebased-v7` review tree. |
| Deliverable | `PIXEL_STAGEA_ON_MAIN.diff`: 26 files, 2 of them new (`campaigns/burstgpt/helper_phone_evidence.py`, `tests/test_static_helper_campaign.py`). `git apply --check` passes with stdin closed. |
| Scratch suite before merge | `run_all.py`: 136 modules / 1,988 tests, exit 0 (`SUITE_SCRATCH_STEP1.log`) |
| Merged into main | 21:05 UTC. Backups are in `$SCRATCH/pixel-int2/premerge-backup-step1/` (24 files + SHA256SUMS). |
| Main suite after merge | 136 modules / 1,988 tests, exit 0 (`SUITE_MAIN_AFTER_MERGE1.log`) |

## 2. The two activation fixes (step 2)

The combined deliverable is `ACTIVATION_FIXES.diff` (8 files, merged into main at 21:33 UTC). Backups
are in `$SCRATCH/pixel-int2/premerge-backup-step2/`.

### (a) OP15 residency checks compared the union mask with OP15's own shards

In v7 this produced 225 `PREPARATION_FAILED` events: `exact partial phone residency transition is
unavailable`.

A two-phone Qwen plan has resident slice `ffn_resident_layer_mask = OP15 layers | Pixel layers 18-23`.
The server really does serve that union. OP15's direct phone session, however, holds only its own
shards. `supports_partial_reconfiguration` required `execution.layer_mask == sum(OP15 shard masks)`, so
it returned False for every Gemma->Qwen HTP replacement. The rig then raised the guard in
`heterogeneous_rig.py` (`_transition_conflicting_executors`). The Pixel agent's projection in
`_persistent_phone_residency_state` ran only after the transition, which was too late.

The fix:

- **New helper, `adapters/llama_server_contracts.py:920` `primary_phone_ffn_contract(command, contract)`.**
  When `phone_helpers` is present, it narrows the slice to the first binding row, which is the ticket's
  own phone. It refuses a binding whose first row is another device, or whose union does not cover the
  slice. Without `phone_helpers` it returns the identity.
- **Applied at every direct-phone (OP15) use:**
  - `adapters/phone_session_ops/replacement.py:167` (`supports_partial_reconfiguration`) and `:237` (`reconfigure`);
  - `adapters/phone_session_ops/identity.py:201` (reuse `supports`);
  - `adapters/phone_session_ops/preflight.py:171` (`_start_contract`: launch layers, close contract, `_execution_by_artifact`);
  - `adapters/heterogeneous_rig_ops/residency.py:181` (stored persistent state; replaces the inline projection).
- **Unchanged, and still the union:** the server launch environment (`S41_SERVER_FFN_LAYER_MASK`), the
  dormant contract, ticket validation and the proofs (`phone_ffn_resident_contract` itself).

### (b) The bounded frontier dropped every `operator_split` envelope

The change is in `_internal/route_generation/costing_rough.py:705/760/794` (`_rough_visits`). The mandatory
representative of each resident-envelope group, under the desktop-control parent, now prefers the
`operator_split` FFN visit. That is the only family `adaptive_decode_policies` probes. The offload visit
of the same envelope stays an ordinary group candidate.

Why the splits were being dropped, found by dumping the v7 frontier:

- decode-only resident envelopes are rough-costed with prefill link payloads;
- those payloads exceed the Pixel link's 40,960 B maximum, and every exceeded link costs a 2^40 us sentinel;
- split envelopes carry twice as many sentinels as offload ones, because offload assignments of co-helper layers charge no link;
- so offload won all 6 mandatory slots, and the splits fell out of the 24-visit refinement budget.

The sentinel costing itself is not changed; see section 5.

Effect, and the check that single-phone behaviour did not change:

- v7 saved catalog + snapshot (the Pixel agent's `diagnose_frontier.py`): adaptive policies go from
  **0 to 4**, at 4352 / 8704 / 13056 / 17408 columns with OP15 layers 0-17 and Pixel layers 18-23.
- The all-on dev_v2 single-phone run: all 9 saved request snapshots, replayed for 2 models, gave
  identical candidates, admitted sets, policies and envelopes before and after the change.

### Tests

`tests/test_two_phone_activation.py` has 11 tests:

| Class | Checks |
| --- | --- |
| PrimaryPhoneContractTests | partial replacement, reuse, session start, **rig-level v7 Gemma->Qwen HTP0 replacement**, stored residency, single-phone unchanged |
| PrimaryPhoneTicketTests | a real two-phone harness ticket narrows 0b111111 to 0b1111; a single-phone ticket stays identical |
| AdaptiveSplitFrontierTests | the split is the mandatory representative even though offload is cheaper; with a budget of only the mandatory visits, adaptive policies still exist (two-phone and single-phone) |
| ReprovisioningExclusionTests | follow layouts built from the family's `operator_ids` never cover the Pixel layers |

Results:

| Tree | Result | Log |
| --- | --- | --- |
| Post-merge main, before the fixes | 6 FAIL + 2 ERROR (8 of 11; the remaining 3 are guards) | `ACTIVATION_TESTS_BEFORE.log` |
| With the fixes | 11/11 PASS | `ACTIVATION_TESTS_AFTER.log` |
| Scratch suite (137 modules / 1,999 tests) | only the known timing assertion `test_cached_synthetic_refinement_is_below_ten_milliseconds` fails (10.55 ms vs 10 ms); isolated reruns 3/3 OK | `SUITE_SCRATCH_STEP2.log` |
| Main after the merge | the same result | `SUITE_MAIN_AFTER_MERGE2.log` |

One test double lacked `route_family`, so the new preference reads `route_family` with `getattr`.

### Re-provisioning (#4) and the dispatch policy with two phones

- **Re-provisioning layouts cannot include the Pixel layers.** The layouts come from demand rows. The
  rows' allowed operators are the phone family's `operator_ids`, and those already exclude co-helper
  layers (`catalog_materialization.py`).
- **The shard index on the rig confirms it.** OP15's Qwen shard index covers layers 0-17 only (HTP0 0x3f,
  HTP1 0xfc0, HTP2 0x3f000). The campaign resolve step refuses any overlap with the Pixel layers.
- **Both ran on hardware.** Re-provisioning and the dispatch policy (`work_conserving_admission` +
  `model_affinity`) were active in every phone arm below.
