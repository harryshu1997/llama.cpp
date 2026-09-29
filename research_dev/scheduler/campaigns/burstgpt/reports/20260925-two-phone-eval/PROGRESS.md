# Two-phone evaluation with the DVFS-fixed Pixel worker: progress log (newest last)

Task: adopt the fixed Pixel worker (`../20260925-pixel-cpu-gpu/`, sha256 5d824455...), re-derive its
evidence, run the matched dev_v2 arms (gate: per-device >= OP15 all-on within noise), then the full
longtail_v1 evaluation against the all-desktop baseline. User authorized hardware on both phones.
Nothing is committed or pushed.

## 2026-09-25 01:45 UTC - start
- Read AGENTS.md, NEXT_AGENT_PROMPT_SCHEDULER.md, `../20260925-pixel-cpu-gpu/README.md` (full),
  `../20260924-per-device-policies/README.md`, `../20260924-pixel-integration-2/README.md` + its
  prepare/run-chain scripts and PROGRESS.md.
- Rig idle at start: lock free, no worker/forward, both phones on adb 5037 (OP15 usb:2-2, Pixel usb:2-9.2).
  Pixel staged dir `/data/local/tmp/s43-pixel-cpugpu-20260925-v1/` hashes verified read-only (worker
  5d824455, libs + shard as README section 5).
- Main suite before any change: 141 modules / 2,021 tests, exit 0 (`SUITE_MAIN_BEFORE.log`).
- Deploy scheduler digest `sha256:5e351e70...` = the int-2 sync; main/staged digest
  `sha256:062ce583...` (per-device merge included). The deploy is NOT synced yet -> first locked step.

## 2026-09-25 02:20 UTC - step 1 tooling (no hardware yet)
New desktop root `/mnt/storage/s43-two-phone-eval-20260925/` (sha256-verified copies):
int-2 GATE_CONFIG + numerical suite + mechanism-r1/b4-r1 + tcp-calibration-r2 + idle-stop-r2
(`COPIED_FROM_INT2.sha256`, 160 files), cpugpu tcp1/tcp2/d1/d3 raw runs + suites
(`cpugpu-evidence/COPIED_FROM_CPUGPU.sha256`, 325 files + suites), the unchanged Pixel-only server
harness (`tools/qualify_pixel_server.py` + `qualify_op11_tcp.py`, sha = the Pixel agent's) and
`tools/qualify_idle_stop.py`.

| file (this dir) | what |
| --- | --- |
| `make_gate_config.py` | GATE_CONFIG.json from the int-2 one: helper_phone -> staged fixed worker dir, 6 pins, 9 S42 vars + 3 S43 flags, as_root; old helper kept as `helper_phone_qualified_reference`; hash cache/prompt copied into the root |
| `make_server_identity_config.py` | Pixel-only server identity config from the Pixel agent's CONFIG2.json, worker/libs/shard/env/pins replaced |
| `prepare_campaign_eval.py` | copy of prepare_campaign_int2.py; changes 1-5 in its docstring (byte-identity bridge, fresh server identity, idle stop of the configured worker, recalibrated kernel, campaign ids) |
| `test_prepare_campaign_eval.py` | 20 tests (13 bridge accept/reject, 5 server-identity/idle, 2 on the real cpugpu evidence): all pass |
| `run_chain_eval.py` | copy of run_chain_int2.py; ordered steps (server identity, prepare, idle stop, arms), phone-side power samplers during runs, leftover listing |
| `tools/power_sampler.sh` | phone-side ~1 Hz read-only sysfs sampler, stops on a stop file (never killed), self-limits to 5 h |

Fixed-worker calibration from tcp1 `c-boost-batch` (72 measured calls per row count, mean incl. the
slow first layer of each token): compute 7.69 / 10.73 / 18.41 ms (B1/B2/B4; p50 6.46 / 9.67 / 18.66),
-> bytes rate 69.5 GB/s logical (old 18.9), ops rate 99.7 G/s (slower of B2/B4); modeled B4 21.5 ms.
Link unchanged (r2: 11.8/16.0/20.5 ms overhead vs 7.3/11.0/13.9 ms measured with the fixed worker).

## 2026-09-25 02:13 UTC - HW 1: sync + fresh Pixel-only server identity run PASS
Chain `chains/CHAIN-adopt1.jsonl` (one outer `flock -w 7200`):
- sync PASS: deploy scheduler `sha256:5e351e70...` -> `sha256:062ce583...` = staged main (per-device merge now on the rig).
- `server-identity-r1` (unchanged `qualify_pixel_server.py`, config from `make_server_identity_config.py`;
  deploy llama-server, single adb-tcp helper = the fixed worker, rooted, layers 18-23): 4/4 outputs
  identical (64 tokens each), 744 Pixel calls (372 at 8,704 + 372 at 17,408 columns), 24 cleanup calls
  outside measurement, server exit 0, worker exit 0 (finite budget), no worker / forward left.

  | request | columns | request s | decode s | host J | vs desktop mean |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | desktop | 0 | 43.73 | 38.68 | 4,860 | |
  | Pixel 50 % | 8,704 | 38.74 | 36.12 | 4,468 | -8.5 % |
  | Pixel 100 % | 17,408 | 39.15 | 36.50 | 4,036 | -17.3 % |
  | desktop | 0 | 40.92 | 38.24 | 4,909 | |

  Old worker, same harness (09-24): Pixel 100 % 49.1 s (+19 %) / -10.6 %. The fixed worker is now faster
  than the desktop CPU on its 6 layers at B1. Worker compute p50 at server cadence 5.4-6.0 ms per layer.
- Power diagnostic (1 Hz sysfs, `analysis/SERVER_IDENTITY_POWER.json`): Pixel ~0.55 W idle (desktop
  requests), ~1.9-2.0 W mean during the Pixel-served requests (USB input capped at 900 mA = 4.6 W; bursts
  above it come from the battery: charge counter -2.5 mAh over the run; battery current is negative when
  discharging on this driver). OP15 idle 0.18 W USB, battery current_now reads a constant 0 (only the
  charge counter is usable there).
- Batteries: OP15 80 % notify 0 before/after; Pixel 100 %.

## 2026-09-25 02:17 UTC - HW 2: idle-stop of the fixed worker PASS; final Pixel bundle; dev_v2 chain started
Chain `chains/CHAIN-dev2a.jsonl` (one outer flock across all of it): sync (no-op, digest 062ce583),
`prepare pre` (bundle only, previous worker's idle receipt) -> `idle-stop-fixed` (unchanged
`qualify_idle_stop.py`: rooted fixed worker, 2 HELLO connections, connected stop refused twice, automatic
idle TERM, exit 0, forward removed, boot unchanged, no worker left) -> `prepare dev2a` (final bundle
`qualification-dev2a/`) -> arms desktop+DP, OP15 all-on, OP15+Pixel per-device.

Final bundle (`qualification-dev2a/PIXEL_EVIDENCE.json`):
- software identity: worker `sha256:5d824455...`, env `sha256:38b7ae12...` (9 S42 + 3 S43 vars), libraries
  at the staged paths (same hashes), shard 940f5f1f, host server ca975ce2 / impl a87e7772 (unchanged, so
  `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json` stays valid; content-equal copy per arm).
- numerical-rows-1-2-4: suite checks unchanged + `byte_identity_bridge` (tcp1 c-boost-batch == a-prod/d-prod
  at m1/m2/m4; d3 b-boost-batch == a-prod at rows 3; d1 all 6 CPU arms identical).
- server-token-identity: `server-identity-r1` + the three int-2 mechanism checks.
- scheduler-launched-session: `idle-stop-fixed` bound to the configured worker/env.
- COST_CALIBRATION: kernel 7,694 us B1 (was 28,302), bytes 69.5 GB/s, ops 99.7 G/s; link unchanged.
- Input diff vs int-2 ii2: only campaign ids, evidence/shard-index paths, and in the two-phone rig the
  worker path, library dir and the 3 flags.

## 2026-09-25 03:07 UTC - HW 3: dev_v2 pair 1 (desktop+DP -> OP15 all-on -> OP15+Pixel per-device), all PASS
Chain `chains/CHAIN-dev2a.jsonl`; analysis `analysis/DEV2A_*` (post_eval.sh on the desktop).

| arm | dur s | CPU kJ | GPU kJ | host kJ | vs desktop | assisted Qwen / Gemma | FFN calls OP15 / Pixel | identical |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| desktop + DP | 627.1 | 44.56 | 19.41 | 63.97 | ref | - | - | ref |
| OP15 all-on | 634.6 | 15.04 | 19.45 | 34.49 | -46.1 % | 4/4, 4/4 | 27,552 / 0 | 5/9 |
| OP15+Pixel per-device | 729.2 | 26.81 | 21.57 | 48.37 | -24.4 % | 4/4, 2/4 | 17,705 / 1,608 | 4/9 |

Per-model execution windows (`DEV2A_MODEL_WINDOW_ENERGY.json`):

| arm | Qwen window kJ / s | Gemma window kJ / s |
| --- | --- | --- |
| desktop + DP | 18.05 / 145.7 | 42.05 / 348.1 |
| OP15 all-on | 9.99 / 133.8 | 32.84 / 584.1 |
| OP15+Pixel | 10.01 / 141.5 | 47.49 / 708.3 |

- The trace-level gap is Gemma, which the Pixel never serves. In the two-phone run the Gemma model was still in
  the page cache from the previous run (ended with Gemma 005): its load took 3.0 s, so Gemma 000/001 started
  at 10.4 s, before the OP15 Gemma sessions were ready, and ran unassisted (ASSISTANCE_DECISION
  HELPER_PHONE_SESSION_LOAD -> INSUFFICIENT_OPPORTUNITY / PHONE_HELPER_UNAVAILABLE). In the OP15 run (after the
  desktop arm, which ended with Qwen) the Gemma load took 33 s and 000/001 started at 41 s, assisted. The
  two-phone Gemma reload before 005 then took 93.7 s (OP15 run 36.5 s), and 005 (617 tokens) waited 110 s.
  Order/page-cache confound, not a Pixel effect; the counterbalanced pair 2 (two-phone first, both after a
  Gemma-ending run) is the matched comparison.
- Qwen (the only Pixel-served model): equal host energy (10.01 vs 9.99 kJ), +6 % time.
- Per-device decisions (`DEV2A_PER_DEVICE_twophone_r1.txt`): cold start OP15 alone at B4 (verdict at 202 s),
  OP15+Pixel challenged and won B4 at 241 s; at B1 OP15 first (289 s), OP15+Pixel won at 295 s. No drops.
  Window fleet J/token (host measured + assumed phones): B1 OP15 38.8 vs OP15+Pixel 28.0; B4 45.9 vs 40.0;
  ms/token B4 612 vs 706 (within the 1.25x bound).
- Per-layer RPC in the campaign server (SHAPE lines): Pixel B1 13.4 / B3 19.4 / B4 23.3 ms (int-2 old worker
  38.4 / 70.5 / 81.6); OP15 9.8 / 10.7 / 12.1 ms.
- Phone power diagnostic (`DEV2A_PHONE_POWER.json`, 1 Hz sysfs): OP15 idle 0.41 W; OP15 assisting sits at its
  2.5 W USB cap for the whole run and drains the battery (charge counter -128 mAh in the OP15 run), i.e.
  ~5.2-5.9 W inside the assisted windows (assumed 4.5 W). Pixel idle 0.49 W; 1.3-1.45 W mean over the Qwen window
  of the two-phone run (assumed 4.5 W active).
- Pixel lifecycle: idle TERM stop, no worker or forward left after every step. Batteries: OP15 80 % -> see
  chain battery rows; notify 0 throughout; Pixel 100 %.
- Counterbalanced pair 2 (`CHAIN-dev2b`: two-phone -> OP15) launched 03:08 UTC.

## 2026-09-25 03:45 UTC - HW 4: counterbalanced pair 2 (two-phone -> OP15), all PASS; GATE decision
Analysis of all five arms: `analysis/DEV2_*`.

| arm | dur s | CPU kJ | GPU kJ | host kJ | vs desktop | assisted Qwen / Gemma | calls OP15 / Pixel | Qwen window kJ / s | Gemma window kJ / s |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| desktop + DP | 627.1 | 44.56 | 19.41 | 63.97 | ref | - | - | 18.05 / 145.7 | 42.05 / 348.1 |
| OP15 r1 | 634.6 | 15.04 | 19.45 | 34.49 | -46.1 % | 4/4, 4/4 | 27,552 / 0 | 9.99 / 133.8 | 32.84 / 584.1 |
| OP15 r2 | 630.5 | 22.47 | 19.40 | 41.87 | -34.6 % | **0/4**, 4/4 | 18,072 / 0 | 18.56 / 150.8 | 19.00 / 325.8 |
| OP15+Pixel r1 | 729.2 | 26.81 | 21.57 | 48.37 | -24.4 % | 4/4, **2/4** | 17,705 / 1,608 | 10.01 / 141.5 | 47.49 / 708.3 |
| OP15+Pixel r2 | 622.1 | 15.84 | 19.80 | 35.64 | -44.3 % | 4/4, 3/4 | 25,020 / 1,842 | **9.11** / 140.0 | 34.81 / 602.8 |

- Means: OP15 38.18 kJ / 632.6 s; OP15+Pixel 42.00 kJ / 675.7 s (+10.0 % / +6.8 %). Pairwise: +40.3 % (pair 1,
  Gemma page-cache start) and -14.9 % (pair 2). Identical-configuration OP15 repeats differ by 21 % (34.5 vs
  41.9 kJ): trace-level noise here is dominated by which requests get assisted, not by the phones.
- OP15 r2 lost all Qwen assistance for a reason unrelated to the Pixel: the dispatcher ran all Gemma first
  (000/001/005/007, 67-393 s), Qwen at 474 s, and the OP15 Gemma->Qwen re-provisioning never became ready
  (`PREPARATION_ENVELOPE_REJECTED` "phone helper replacement source is not ready" x119 per Qwen request,
  `HELPER_REMATERIALIZATION_FAILED` "ready layout produced no helper opportunity") -> PHONE_HELPER_UNAVAILABLE.
- Pixel-attributable component (Qwen window, all 4 Qwen assisted): OP15 9.99 kJ vs OP15+Pixel 10.01 / 9.11 kJ
  (mean -4.4 %), time 133.8 vs 141.5 / 140.0 s (+5 %).
- Per-device decisions, both two-phone runs: OP15 alone first at every composition, then OP15+Pixel challenged
  and won B4 and B1 (r1 at 241/295 s, r2 at 201/244 s); no drops, no failures. Window fleet J/token (host
  measured + assumed phones) r2: B1 36.4 -> 26.4, B4 42.8 -> 37.1; ms/token B4 601 -> 688 (bound 1.25x).
- Pixel per-layer RPC in the server: B1 12.8-13.4, B2 15.5, B3 17.8-19.4, B4 22.7-23.3 ms.
- Exactness vs desktop: 4-5/9 per arm; the same fixed positions in every phone arm AND in the unassisted
  OP15 r2 Qwen outputs (Gemma 000 @19, Qwen 003 @103/@74, 004 @111/@155, Gemma 005 @41/@59) -> batching
  near-ties, full lengths everywhere.
- Phone power diagnostic (whole run / Qwen window, W): OP15 assisting 4.0-5.3 / 3.5-6.1 (2.5 W USB cap +
  battery); Pixel 0.49 idle, 1.3-1.5 (usb+counter) / 1.45-1.8 (usb+battery current) in the Qwen window.
- **GATE: PASS (proceed to longtail_v1).** The per-device two-phone arm is within the measured noise of OP15
  all-on at trace level (+10 % on means vs a 21 % spread between identical OP15 runs), and at least as good on
  the only model the Pixel serves (Qwen window -4.4 % energy, +5 % time); the policy never dropped or failed a
  device set. Caveat carried forward: single longtail runs will show the same assistance-availability noise.

## 2026-09-25 03:44 UTC - longtail_v1 inputs derived; 4-arm chain started
- `CHAIN-lt-prep`: sync (no-op) + `prepare lt1` from `template-longtail` (derived with
  prepare_trace_inputs_v2.py from the int-2 dev_v2 template; diff = the four longtail_v1 trace paths + ids).
  Fresh bundle `qualification-lt1/` (same receipts as dev2a).
- Legacy all-desktop arm: `prepare_trace_inputs_v2.py --source inputs-desktop-lt1 --drop-dispatch-policy`
  -> `inputs-desktop-legacy-lt1` (diff = `dispatch_policy` removed + ids/paths), identity copied.
- `CHAIN-lt` launched 03:44:45 UTC, order: legacy desktop -> desktop+DP -> OP15 all-on -> OP15+Pixel per-device.

## 2026-09-25 05:14 UTC - HW 5: longtail_v1 arm 1/4 (legacy all-desktop) PASS
- 4,984 s, 31/31 requests, CPU 377.0 + GPU 159.3 = **536.3 kJ** host (09-23 on older code: 523.7 kJ / 4,901 s;
  +2.4 % / +1.7 %). Desktop + dispatch policy preflight started 05:13 UTC.

## 2026-09-25 06:16 UTC - HW 6: longtail_v1 arm 2/4 (desktop + dispatch policy) PASS
- 3,381 s, 31/31, CPU 279.6 + GPU 113.5 = **393.0 kJ** host: -26.7 % energy / -32.2 % duration vs legacy.
  OP15 all-on preflight started 06:15 UTC.

## 2026-09-25 06:47 UTC - HW 7: longtail_v1 arm 3/4 (OP15 all-on) FAILED at 1,583 s (scheduler crash, fail-closed)
- `inputs-op15-lt1/run-eval/run/FAILURE.json`: `UnifiedScheduleError: qualified desktop baseline is not available:
  MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram`, raised in `_pick_objective_candidate`
  (`_unified/automated_selection_ops/objectives.py:271`) during a REPLAN of Qwen request 018
  (`replan_automated_request` -> `_replan_priority_compaction_without_followers` -> `_execute_automated_replan`).
- Sequence (decision log): at 1,580 s request 027 was admitted with a cold model load (31.0 GB: 19.2 GB host RAM +
  11.9 GB VRAM); at 1,583 s the replan of 018 found its desktop baseline overlapping 027's exclusive replacement
  on host RAM, and selection raised instead of re-projecting.
- Same bug class as the arrival-path case fixed on 09-23 (`selection.py::_reproject_rejected_baseline`, re-projects
  residency at a memory-rejected baseline's start); the replan loop in `replan_commit._execute_automated_replan`
  has no such re-projection. Not Pixel-related (OP15-only arm). The chain stopped at the first failure (the
  two-phone arm did not run); no worker or forward left; OP15 79 % notify 0.
- My waiter missed the failure for ~40 min (it waited for a result_status line; the failed run has none).
- 07:29 UTC: retry chain `CHAIN-lt2` (fresh inputs `lt2`, unchanged tree): OP15 all-on, then OP15+Pixel. In
  parallel, a replan re-projection fix is prepared in an isolated copy (not merged) in case the crash repeats.

## 2026-09-25 07:31 UTC - STOP: OP15 battery_notify_code 512 (charger latched off)
- Retry chain `CHAIN-lt2`: sync PASS, `prepare lt2` OK (inputs-{desktop,op15,two-phone}-lt2, qualification-lt2),
  then the battery check before the OP15 preflight read notify **512** (level 80 %) -> the chain raised and stopped;
  no preflight, no run. Re-read at 07:33 UTC: still 512. Rig idle: no scheduler/server process, no adb forward,
  no Pixel worker, lock free.
- Longtail desktop arms analysed (`analysis/LT_*`): legacy 15 reloads / 10 switches / 630 s loads; desktop+DP
  8 / 4 / 225 s, 12 affinity displacements; 14/31 outputs identical (host-only near-ties, full lengths).
- README written; resume command in its status section. Needs the user: replug the OP15; decide on a
  replan-path fix before re-running the phone arms (the crash can recur); commit checkpoint.
