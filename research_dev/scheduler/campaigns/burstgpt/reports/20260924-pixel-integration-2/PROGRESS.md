# Pixel integration 2: progress log (newest last)

Takes over from the Pixel agent (`../20260924-pixel-second-phone/`, stopped 20:48 UTC after two
zero-Pixel activation failures). Scratch trees: `$SCRATCH/pixel-int2/{base,root}` (base = main at
20:54 UTC, reports/ and non-scheduler research_dev entries symlinked read-only).

## 2026-09-24 20:54 UTC - start
- Read AGENTS.md, NEXT_AGENT_PROMPT_SCHEDULER.md, Pixel README (full), its diffs/scripts, two-phone
  readiness README + GAPS_README.
- Findings before any edit:
  - `ROOTED_WORKER_LIFECYCLE.diff` is ALREADY in main (all 3 files hash to its `after`).
  - `STAGEA_V7_INTEGRATION.diff` is cumulative (subsumes BEFORE_PREFLIGHT, STAGEA, V5, V6,
    REPLAN_CORRECTIONS). `STAGEA_V7_REBASED_REVIEW.diff` = V7 rebased on main; all 26 target files
    in main still hash to the rebase manifest's `main_sha256`, and `git apply --check` passes.
  - Rig: lock free, both phones listed on adb 5037 (OP15 usb:2-2, Pixel usb:2-9.2).

## 2026-09-24 21:05 UTC - STEP 1 one tree: merged
- Rebase method: for each of the 26 files the Pixel agent changed (its `/tmp/s42-pixel-campaign-20260924/{base,root}`,
  root = base + `STAGEA_V7_INTEGRATION.diff`, verified with `git apply -R --check`; REPLAN_CORRECTIONS
  and BEFORE_PREFLIGHT are subsumed), 24 files were unchanged in main since its base -> taken as-is;
  `replan.py` and `test_dispatch_policy.py` changed on both sides -> `git merge-file` three-way, both
  clean (main's affinity follow-up `AFFINITY_DISPLACEMENT_REASON` + its `preparation_phase_completed`
  compaction and `REPLAN_REQUIRED` early return are disjoint hunks). Result is byte-identical to the
  Pixel agent's `rebased-v7` tree for all 26 files.
- Scratch root full suite (`run_all.py`): 136 modules / 1,988 tests, exit 0.
- `PIXEL_STAGEA_ON_MAIN.diff` (26 files, 2 new) `git apply --check` OK -> APPLIED to main.
  Pre-merge backups: `$SCRATCH/pixel-int2/premerge-backup-step1/` (24 files + SHA256SUMS; 2 files absent in main).
- Main full suite re-run: running.
- 21:14 UTC main full suite after the merge: 136 modules / 1,988 tests, exit 0.

## 2026-09-24 21:12 UTC - STEP 2 activation fixes (scratch root, base2 = post-merge main)
- (a) new `adapters/llama_server_contracts.py::primary_phone_ffn_contract(command, contract)`: with
  `phone_helpers` it narrows the resident slice to the first (ticket's own) phone's layers; refuses a
  binding whose union does not cover the slice or whose first row is another device. Applied to every
  direct-phone (OP15) use: `phone_session_ops/replacement.py` (supports_partial_reconfiguration +
  reconfigure), `phone_session_ops/identity.py` (reuse `supports`), `phone_session_ops/preflight.py`
  (`_start_contract`, i.e. launch + stored `_execution_by_artifact`), `heterogeneous_rig_ops/residency.py`
  (replaces the Pixel agent's inline projection). Server launch env, dormant contract and proofs keep
  the union (`phone_ffn_resident_contract` unchanged).
- (b) `_internal/route_generation/costing_rough.py::_rough_visits`: the mandatory representative of a
  resident-envelope group (desktop-control parent) now prefers the `operator_split` FFN visit (the only
  family `adaptive_decode_policies` probes); the offload visit stays a regular group candidate. Root
  cause seen in the v7 frontier: decode-only resident envelopes are rough-costed with prefill link
  payloads, which exceed the Pixel link's 40,960 B maximum -> 2^40 us sentinels per layer; split
  envelopes carry twice the sentinels of offload ones (offload of co-helper layers charges no link),
  so offload won all 6 mandatory slots and splits fell out of the 24-visit refinement budget.
- Tests `tests/test_two_phone_activation.py` (11): on base2 6 FAIL + 2 ERROR (8 of 11), on root 11/11 PASS.
  v7 saved catalog+snapshot (`diagnose_frontier.py`): adaptive policies 0 -> 4 (4352/8704/13056/17408
  columns, OP15 0-17 + Pixel 18-23). Single-phone all-on dev_v2 replay (9 saved request snapshots x
  2 models): candidates, admitted sets, policies and envelopes identical before/after.
- Re-provisioning (#4): layouts come from demand rows whose allowed operators are the phone family's
  `operator_ids`, which already exclude co-helper layers (catalog_materialization.py); OP15's Qwen
  shard index on the rig covers layers 0-17 only (HTP0 0x3f, HTP1 0xfc0, HTP2 0x3f000) and the campaign
  resolve refuses overlap. Regression test added (follow layouts never cover the Pixel layers).
- First root suite run hit a test double without `route_family` (test_multi_session_phone) -> use
  getattr in the new preference; rerun running.
- 21:32 UTC scratch root full suite (both fixes): 137 modules / 1,999 tests; only failure the known
  timing assertion `test_cached_synthetic_refinement_is_below_ten_milliseconds` (10.55 ms vs 10 ms);
  isolated module rerun 3/3 OK. `ACTIVATION_FIXES.diff` (8 files) `git apply --check` OK -> APPLIED to
  main (backups `$SCRATCH/pixel-int2/premerge-backup-step2/`). Main suite rerun running.

## 2026-09-24 21:36 UTC - rig deploy prepared (no phone touched yet)
- New root `/mnt/storage/s43-pixel-int2-20260924/`: copies (sha256-verified) of the Pixel agent's
  GATE_CONFIG.json, qualification/numerical, mechanism-r1, mechanism-b4-r1, tcp-calibration-r2,
  idle-stop-r2 (inputs to its receipts), `prepare_campaign_int2.py` (its prepare_campaign.py with the
  template dir + trace tag as arguments), `run_chain_int2.py` (one outer flock: sync stage -> deploy
  source, prepare, per arm battery log + preflight + battery log + run + battery log; stops on the first
  failure, on OP15 notify 512, or on zero Pixel calls when asked).
- Templates via `prepare_trace_inputs_v2.py` from the all-on dev_v2 inputs (same deploy):
  `template-dev1` (longtail_dev_v1 trace) and `template-dev2`; models/rig identical to all-on.
- Read-only state: Pixel no worker, no adb forwards; OP15 notify 0.

## 2026-09-24 21:37 UTC - hardware (i) launched
Chain `CHAIN-i1` under one `flock -w 7200`: sync stage -> deploy source, prepare attempt i1 from
`template-dev1`, then two-phone dev_v1 (stop if 0 Pixel calls), then desktop-only dev_v1 (+dispatch policy).
- 21:42 UTC sync PASS (deploy scheduler digest a63d73ee -> 5e351e70 = staged main), prepare i1 OK.
- 21:45 UTC two-phone dev_v1 preflight PASS; run started 21:45:58. Batteries: OP15 80 % notify 0, Pixel 100 %.
- Main full suite after the step-2 merge: 137 modules / 1,999 tests; only the known 10 ms timing
  assertion failed (isolated rerun OK).
- ~21:52 UTC live: Qwen server registered `helpers=2` (op15 mask 262143 = layers 0-17 FunctionFS,
  pixel10pro mask 16515072 = layers 18-23 tcp:26991); Pixel worker ready (packed CPU, 6 layers,
  quantum 4352). First Pixel FFN calls observed (request ids from 16777217 = helper-1 range, layers
  18/19, rows=2: Qwen 000 and 002 batched, 17,408 columns).
- 21:59 UTC two-phone dev_v1 run PASS: 6/6 requests, 785.8 s, measured host 32.477 kJ; assumed phone
  energy OP15 1.942 kJ + Pixel 1.409 kJ. FFN calls: Gemma OP15 14,880; Qwen OP15 5,661 + **Pixel
  1,998 (PIXEL10PRO0)**; Qwen 3/3 and Gemma 2/2 assisted, all at full width (1,000,000 ppm).
  **0 PREPARATION_FAILED** (v7: 225). OP15 layout timeline: Qwen HTP0-2 (5-103 s) -> Gemma HTP1,
  HTP2, HTP0 (193-229 s) -> **Gemma->Qwen partial replacement HTP1 then HTP0 then HTP2 (488-533 s)**,
  every SESSION_VERIFIED. Pixel lifecycle: idle TERM, exit 0, forward removed, boot unchanged.
  Batteries after: OP15 79 % notify 0, Pixel 100 %. Desktop-only dev_v1 control running.
- 22:17 UTC desktop-only dev_v1 control PASS: 714.5 s, host 64.784 kJ. Token identity two-phone vs
  desktop: **5/6 identical**; Qwen 004 first differs at token 149 of 196 (late near-tie flip).
  Two-phone all-on vs desktop+dispatcher: host -49.9 %, duration +10.0 %. (Not the incremental Pixel
  effect; that is step (ii).) `DEV1_TRANSITION_ANALYSIS.json`, run copies in `physical/dev1-*`.
- 22:20 UTC chain `CHAIN-ii1` launched: dev_v2 desktop+dispatcher -> OP15 all-on -> OP15+Pixel all-on.
- dev_v1 two-phone per-helper server summaries (S41SERVERFFN/SHAPE, per-layer RPC, full width 17,408):

  | Qwen launch | helper | layers | calls | rpc mean ms (B1 / B2) | phone compute mean ms (B1 / B2) |
  | --- | --- | --- | ---: | --- | --- |
  | #2 (000+002) | op15 FunctionFS | 0-17 | 1,870 | 9.71 / 9.89 | 8.96 / 8.85 |
  | #2 | pixel10pro adb-tcp | 18-23 | 660 | 37.5 / 55.9 | 28.9 / 46.1 |
  | #10 (004) | op15 | 0-17 | 3,264 | 9.65 | 8.93 |
  | #10 | pixel10pro | 18-23 | 1,152 | 38.4 | 29.3 |

  Layers run in order, so per decode token the Pixel adds 6 x ~38 ms ~ 230 ms next to OP15's
  18 x ~9.7 ms ~ 175 ms: the Pixel's six layers cost more wall time than OP15's eighteen.
- dev_v1 loads/dispatch (`DEV1_ALLON_STYLE.md`, analyze_allon.py run on the desktop): both arms
  qwen -> gemma -> qwen, 4 reloads, 2 affinity displacements each. The two-phone arm's second Qwen
  launch took 149.8 s to load (desktop 78.7 s) and Qwen 004 waited 157 s from acquire to start
  (desktop 81.6 s): it waited for the OP15 Gemma->Qwen re-provisioning (488-533 s). That load wait
  explains most of the +10 % duration.
- 22:36 UTC dev_v2 desktop+dispatcher PASS: 682.2 s, host 68.035 kJ (earlier baseDP before the
  affinity follow-up: 865.0 s / 82.3 kJ). 22:40 OP15 all-on preflight PASS, run started.
- 22:51 UTC dev_v2 OP15 all-on PASS: 617.4 s (-9.5 %), host 40.817 kJ (-40.0 % vs desktop+dispatcher),
  7/9 identical (Gemma 000 @19, Qwen 004 @57); Gemma 3/4 + Qwen 1/4 assisted (16,008 + 1,139 calls).
  OP15+Pixel preflight started.

## 2026-09-24 23:12 UTC - (ii) matched dev_v2 arms (same deploy/tree, single runs)

| arm | dur s | CPU kJ | GPU kJ | host kJ (meas.) | vs desktop | assumed phone kJ | identical | Qwen / Gemma assisted | calls OP15 / Pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| desktop + dispatcher | 682.2 | 47.00 | 21.04 | 68.035 | ref | 0.60 (OP15 idle) | ref | - | - |
| OP15 all-on | 617.4 | 21.35 | 19.47 | 40.817 | -40.0 % / -9.5 % dur | 1.38 | 7/9 | 1/4, 3/4 | 17,147 / 0 |
| OP15+Pixel all-on | 806.2 | 18.51 | 24.03 | 42.541 | -37.5 % / +18.2 % dur | 1.86 + 1.14 | 6/9 | 4/4, 3/4 | 21,513 / 1,926 |

- Pixel increment vs OP15 all-on: host +4.2 % (+1.72 kJ), duration +30.6 %, modeled fleet +7.9 %.
  **The Pixel arm does not help at trace level** in this single run.
- Where the time goes (DEV2_ALLON_STYLE.md): Qwen hot load 114.0 s vs 46.6 s and Gemma cold reload
  90.7 vs 62.8 s (model loads, not Pixel work; page-cache state is the likely cause), then the 4-way Qwen
  batch runs 189 s vs 140 s. Gemma 005 decode is identical (269.6 s both).
- Qwen window energy (DEV2_MODEL_WINDOW_ENERGY.json): two-phone 12.84 kJ / 189.1 s (67.9 W) vs OP15
  15.44 kJ / 140.5 s (109.9 W) vs desktop 17.70 kJ / 145.2 s: -16.8 % host inside the Qwen batch, but
  the OP15 arm assisted only 1 of the 4 Qwen requests (controller chose host for 002/003/006), so this
  mixes Pixel layers with a different assistance choice.
- Pixel per-layer RPC by batch (Qwen server SHAPE): B1 38.4 ms, B2 53.8, B3 70.5, B4 81.6 (compute
  29.4 -> 70.6 ms); OP15 B1 9.7 -> B4 12.1 ms for its 18 layers. At B4 the Pixel's 6 layers take ~490 ms
  per step vs OP15's 18 layers ~218 ms.
- Token differences two-phone: Qwen 004 @111, Gemma 005 @41, Gemma 007 @1 (Gemma is not Pixel-served);
  OP15 arm: Gemma 000 @19, Qwen 004 @57.
- Pixel lifecycle clean (idle TERM exit 0, forward removed, boot unchanged); OP15 notify 0 throughout.
- Because the +4.2 % is close to the known repeat noise (3.1 % energy / 13 % duration) and the loads
  differ, a counterbalanced repeat pair (two-phone first, then OP15) was launched at 23:13 UTC
  (`CHAIN-ii2`) before deciding on the longtail pair.
- 23:32 UTC repeat two-phone (ii2, run first this time) PASS: 681.5 s, host 40.132 kJ (CPU 19.29 /
  GPU 20.85), assumed phones 2.52 kJ, Qwen 4/4 + Gemma 3/4 assisted, Pixel 1,104 calls, 5/9 identical
  vs the ii1 desktop control (Gemma 000 @19, Qwen 003 @74, Qwen 004 @111, Gemma 005 @41). The two
  two-phone runs differ by 125 s and 2.4 kJ. OP15 repeat running.

## 2026-09-24 23:50 UTC - repeat pair done; conclusion
- OP15 repeat (ii2, run second) PASS: 652.1 s, host 38.843 kJ, 8/9 identical, Qwen 4/4 + Gemma 2/4
  assisted (B4 1,734 OP15 calls).
- Pixel increment over OP15 all-on: pair r1 +4.2 % host / +30.6 % time; pair r2 +3.3 % / +4.5 %;
  means 41.34 vs 39.83 kJ (+3.8 %), 744 vs 635 s (+17 %). Qwen window: OP15 r2 (4/4 assisted)
  10.31 kJ / 135 s vs two-phone 12.84-13.76 kJ / 161-189 s. The r1 two-phone load slowness (114 s Qwen)
  did not recur in r2 (65 s): it was page-cache noise.
- (iii) longtail_v1 pair NOT run: the Pixel arm does not help. Cleanup verified: no Pixel worker, no adb
  forwards, OP15 notify 0 (77 %), Pixel 100 %, lock free.
- README.md written; talks.md updated.
