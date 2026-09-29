# Two-phone evaluation with the DVFS-fixed Pixel worker (2026-09-25)

Task: adopt the fixed Pixel worker from `../20260925-pixel-cpu-gpu/` into the two-phone campaign, run the
matched dev_v2 arms, and, if the gate passes, the full `longtail_v1` evaluation against the all-desktop
baseline. The user authorized hardware on both phones. Nothing is committed or pushed. No scheduler or
server code was changed: every new file is tooling in this directory. The progress log is `PROGRESS.md`.

## 2026-09-25 16:38 UTC - Current eval_v2 pair, independently audited

The earlier stopped/replug/resume notes below are historical. The authorized OP15 RAM boot and
replan/fail-fast deployment preceded today's `longtail_eval_v2` chain. Its baseline and two-phone
arms completed PASS, 14/14 each, on matching source files/binaries and matched request inputs:

| Arm | CPU kJ | GPU kJ | Host kJ | Duration s | Exact outputs |
| --- | ---: | ---: | ---: | ---: | --- |
| Legacy desktop | 159.872 | 68.663 | 228.535 | 2244.348 | reference |
| Dispatcher + OP15 + Pixel | 38.285 | 56.129 | 94.414 | 1832.951 | 13/14, strict FAIL |

Measured host saving **58.687%**; duration **18.330% shorter**. Qwen 7/7 and Gemma 6/6 assisted;
OP15 63,852 / Pixel 7,176 proof calls. Qwen 004 first differs at zero-based token 137; no logits
justify an exactness exception. Phone power remains assumed, not included in the host percentage.
One pair, not a repeated estimate. Clean exit 0 and no leftover Pixel worker/forward.

At 16:34:07 UTC, **after** this arm's PASS record, the outer chain stopped on `CANCEL requested`.
Desktop+dispatcher and OP15-only controls on eval_v2 were not run; additional Pixel-only benefit
is therefore unverified. The reporting session did not request this cancellation or run hardware.
See [audited tables, graphs and PDF](../20260925-progress-report/README.md).

## Status and headline (2026-09-25 07:35 UTC)

**Stopped: the OP15 reported `battery_notify_code` 512 (charger latched off) at 07:31 UTC, before the retry of the
two phone longtail arms started.** Per the rules, no further hardware ran. The user has to replug the OP15. At that
point the OP15 was at 80 %, the Pixel at 100 %, the rig lock was free, and no worker or adb forward was left.

| step | result |
| --- | --- |
| 1. adoption of the fixed Pixel worker | **done**. New Pixel evidence bundle: worker 5d824455, new environment sha. The receipt check accepts the new hash only through a byte-identity bridge to the qualified worker; 20 unit tests. Fresh Pixel-only server token identity PASS: 4/4 identical, and the Pixel at 100 % saves 17.3 % host energy at B1 against 10.6 % for the old worker. Idle-TERM stop re-qualified. Cost prior 7.7 ms/layer at B1 (was 28.3 ms). |
| 2. dev_v2 matched arms (5 runs) | **gate PASS**. OP15+Pixel per-device is within the measured noise of OP15 all-on: means 42.0 vs 38.2 kJ, while identical OP15 runs spread by 21 %. It is at least as good where the Pixel acts: Qwen window 9.11 / 10.01 vs 9.99 kJ. The policy picked OP15+Pixel at B1 and B4 from measured windows, with no drops. |
| 3. longtail_v1 | **2 of 4 arms done**: legacy all-desktop **536.3 kJ / 4,984 s**; desktop + dispatcher **393.0 kJ / 3,381 s (-26.7 %)**. **OP15 all-on crashed** at 1,583 s: a scheduler bug in a replan, where the baseline was rejected behind a queued host-RAM replacement (section 4). The retry (`inputs-*-lt2`, prepared) was blocked by the OP15 notify 512. OP15+Pixel was not run. |

**Correction (2026-09-25, after `../20260925-replan-reprojection-fix/`).** Row 3 and section 4 misattribute the OP15
all-on crash. The arm failed at **431.4 s, not 1,583 s**: Qwen request **007** failed in a replan with the same
`qualified desktop baseline is not available: MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram` message (decision-log
record 29). The coordinator surfaced lifecycle failures only in `drain()`, after the last arrival (027, the Qwen cold
load, at 1,580.2 s), so the arm ran 1,150 s more. Request 018 is a **Gemma** request, not Qwen. It failed at 1,583 s only
as a side effect: the drain-time `arrival_coordinator_abort` (records 130-143) woke it and its replan hit the same bug.
FAILURE.json carries the same message for both, which is how the misattribution happened. Root cause, replays and tests
are in `../20260925-replan-reprojection-fix/`. The replan fix is merged into main (08:43 UTC). A fail-fast abort on the
first lifecycle failure is also merged (`../20260925-replan-reprojection-fix/failfast/`). So item 2 below ("the fix is
not written yet") is out of date. The deploy is not synced yet.

**What needs the user:**

1. **Replug the OP15** (notify 512).
2. **Decide how to handle the replan crash.** It is not Pixel-related and it can recur in both phone arms.
   Either retry the two phone arms unchanged, accepting the risk of losing about 30 min again, or first get
   a replan-path fix: re-project residency at a memory-rejected baseline's start, as the 09-23 arrival fix
   does. The fix is not written yet.
3. **Commit checkpoint.** Nothing is committed. This round changed no scheduler code; all new files are in
   this directory.

**Resume** (after the replug, under one outer lock; a fresh prepare re-checks the USB identities):
```sh
R=/mnt/storage/s43-two-phone-eval-20260925
flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
  python3 $R/run_chain_eval.py --status $R/chains/CHAIN-lt3.jsonl --stage $R/stage \
    --prepare-script $R/prepare_campaign_eval.py --prepare $R lt3 $R/template-longtail lt \
    --arm $R/inputs-op15-lt3 --arm $R/inputs-two-phone-lt3
bash $R/analysis/post_eval.sh $R/analysis/LTFULL legacy legacy=$R/inputs-desktop-legacy-lt1 \
  desktopDP=$R/inputs-desktop-lt1 op15=$R/inputs-op15-lt3 twophone=$R/inputs-two-phone-lt3
```

## 1. Adoption of the fixed worker (step 1)

### What changed and where

| File (this directory) | Change |
| --- | --- |
| `make_gate_config.py` | Derives `GATE_CONFIG.json` from the int-2 config. `helper_phone` gets the staged dir `/data/local/tmp/s43-pixel-cpugpu-20260925-v1` (worker, library dir, shard), its six sha256 pins, the nine qualified `S42_PIXEL_*` variables plus `S43_PIXEL_UCLAMP_MIN=1024`, `S43_PIXEL_CPU_POLL=100`, `S43_PIXEL_CPU_BATCH_PAIR=1`, and `as_root` true. The old block is kept as `helper_phone_qualified_reference`. The script refuses libraries or a shard whose bytes differ from the qualified ones, and refuses batch pair without pair-dot / with max tokens > 4. |
| `make_server_identity_config.py` | Derives the Pixel-only server identity config from the Pixel agent's qualified `CONFIG2.json`. Only the phone worker, libraries, shard, environment and pins change. |
| `prepare_campaign_eval.py` | Copy of `../20260924-pixel-integration-2/prepare_campaign_int2.py`, with changes 1-5 below. |
| `test_prepare_campaign_eval.py` | 20 unit tests: 13 bridge accept/reject cases, 5 server-identity/idle cases, and 2 on the real cpugpu evidence. All pass. |
| `run_chain_eval.py` | Copy of `run_chain_int2.py`. Adds ordered steps (server identity, prepare, idle stop, arms), a phone-side power sampler during every run, and a leftover worker/forward listing after every step. |
| `tools/power_sampler.sh` | Phone-side, read-only, about 1 Hz sysfs sampler. It stops on a stop file (it is never signalled) and limits itself to 5 h. |
| `analyze_phone_power.py`, `post_eval.sh` | Phone power diagnostic, and the per-arm-set analysis driver that runs the existing analysis tools on the desktop. |
| `tools/qualify_pixel_server.py`, `qualify_op11_tcp.py`, `qualify_idle_stop.py` | Unchanged copies of the Pixel agent's harnesses (sha256 equal to its desktop copies). |

Changes in `prepare_campaign_eval.py` against `prepare_campaign_int2.py`:

1. **numerical-rows-1-2-4.** The libraries and the shard must still be in the numerical suite's hash set,
   as before. A worker hash outside the set is accepted only through `byte_identity_bridge()`, which
   requires all of the following:
   - **TCP rows 1/2/4 (hash-bound).** In cpugpu `tcp1`, the configured worker with its exact environment
     (`c-boost-batch`) and the qualified worker 64133753 with the qualified environment (`a-prod`,
     `d-prod`) produce identical output dumps at m1, m2 and m4. The dumps are re-hashed from the files.
     Every one of those arms:
     - passed with a clean finite-budget stop (exit 0, not signalled, boot unchanged, forward removed, no
       pid left);
     - had its preflight observe exactly the configured pins on the configured paths;
     - has a launch command that runs exactly that worker, library dir, shard and environment.
     The reference worker's hash must be in the suite.
   - **Phone-local rows 3 (path-bound).** In cpugpu `d3`, `b-boost-batch` output equals `a-prod` output.
   - **Flag matrix.** In cpugpu `d1`, all six CPU arms (each flag alone and combined) are identical.

   The unit tests show that each broken link is rejected: a different dump, a dump that differs even when
   its recorded hash was updated to match, another environment, a launch without a flag, an unclean stop,
   a preflight pin mismatch, missing rows, a row-3 mismatch, a flag-matrix mismatch, a nonzero exit, a
   library or shard outside the suite, and a reference worker outside the suite.
2. **server-token-identity.** The fresh Pixel-only server run (`server-identity-r1`) is required. It must
   have:
   - 4/4 outputs identical at 64 tokens each;
   - Pixel calls at both widths;
   - server exit 0 and worker exit 0;
   - the configured worker, environment and pins;
   - the current server and server library.

   The int-2 OP15+Pixel mechanism checks are kept unchanged.
3. **scheduler-launched-session.** The idle-TERM qualification must have launched the configured worker
   with the configured environment and pins. ATTEMPT `pre` builds a bundle from the previous worker's
   idle receipt, only so that this qualification can run, and materializes no inputs.
4. **COST_CALIBRATION.json.** The kernel now comes from the fixed worker's TCP calls, in 72 measured calls
   per row count. The mean per layer includes the slow first layer of each token:
   - compute 7.69 / 10.73 / 18.41 ms at B1 / B2 / B4;
   - bytes rate 69.5 GB/s logical, set by B1 (was 18.9 GB/s, from a 28.3 ms kernel);
   - ops rate 99.7 G/s, set by the slower of B2/B4. The modeled B4 is 21.5 ms, which is conservative.

   The link keeps the int-2 derivation from `tcp-calibration-r2`. It is conservative: its overhead is
   11.8/16.0/20.5 ms, against 7.3/11.0/13.9 ms measured with the fixed worker. The receipt lists both.
5. **Campaign ids** are `s43-two-phone-eval-<tag>-<arm>-<attempt>`.

### Hardware checks of the adoption

All ran under one outer `flock -w 7200` per chain (`chains/CHAIN-adopt1`, `CHAIN-dev2a`).

- **Sync.** The deploy scheduler went from `sha256:5e351e70...` (the int-2 state, without the per-device
  merge) to `sha256:062ce583...`, which equals the staged main tree. The main suite before any run:
  141 modules / 2,021 tests, exit 0 (`SUITE_MAIN_BEFORE.log`). There was no server rebuild: the library
  a87e7772 is unchanged, so `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json` stays valid.
- **Fresh token identity** (`server-identity-r1`: unchanged Pixel-only llama-server harness, rooted fixed
  worker, single adb-tcp helper, finite budget). Outputs were 4/4 identical. There were 744 Pixel calls
  and 24 cleanup calls outside measurement. Both exits were clean.

  | request (64 tokens, B1) | request s | host J | vs desktop mean |
  | --- | ---: | ---: | ---: |
  | desktop before / after | 43.73 / 40.92 | 4,860 / 4,909 | ref |
  | Pixel 50 % of layers 18-23 | 38.74 | 4,468 | -8.5 % |
  | Pixel 100 % | 39.15 | 4,036 | **-17.3 %** |
  | old worker, same harness (09-24), Pixel 100 % | 49.13 | 4,324 | -10.6 % |

  The fixed worker is now faster than the desktop CPU on its six layers at B1.
- **Idle-TERM stop with the fixed worker** (`idle-stop-fixed`, unchanged `qualify_idle_stop.py`): PASS.
  - It made two HELLO connections, and the connected stop was refused both times.
  - Then came the automatic idle TERM: exit 0, forward removed, boot unchanged, no worker left.
- **Final bundle** (`qualification-dev2a/`, and the same again as `-dev2b` and `-lt1`):
  - worker `sha256:5d824455...`;
  - environment `sha256:38b7ae12...`;
  - library keys at the staged paths, with the same hashes;
  - shard 940f5f1f;
  - the three new receipts described above.

  The input diff against int-2 is only the campaign ids, the evidence and shard-index paths, and, in the
  two-phone rig, the worker path, the library dir and the three flags.

## 2. dev_v2 matched arms (step 2)

Order: pair 1 ran desktop+DP, then OP15 all-on, then OP15+Pixel. Pair 2 was counterbalanced: OP15+Pixel,
then OP15. All five runs passed.
- Host energy is measured: RAPL package plus NVML board.
- The "assumed" phone energy (4.5 W active / 0.875 W idle) is not used in any comparison below.
- Analysis: `analysis/DEV2_*`, produced by `post_eval.sh` with the existing tools (`compare_trace_energy.py`,
  `analyze_int2.py` / `analyze_longdecode_pair.py`, `analyze_allon.py`, `interval_energy_int2.py`,
  `analyze_per_device.py`, `analyze_phone_power.py`).

| arm | dur s | CPU kJ | GPU kJ | host kJ | vs desktop | assisted Qwen / Gemma | FFN calls OP15 / Pixel | identical |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| desktop + dispatch policy | 627.1 | 44.56 | 19.41 | 63.97 | ref | - | - | ref |
| OP15 all-on r1 | 634.6 | 15.04 | 19.45 | 34.49 | -46.1 % | 4/4, 4/4 | 27,552 / 0 | 5/9 |
| OP15 all-on r2 | 630.5 | 22.47 | 19.40 | 41.87 | -34.6 % | **0/4**, 4/4 | 18,072 / 0 | 5/9 |
| OP15+Pixel per-device r1 | 729.2 | 26.81 | 21.57 | 48.37 | -24.4 % | 4/4, **2/4** | 17,705 / 1,608 | 4/9 |
| OP15+Pixel per-device r2 | 622.1 | 15.84 | 19.80 | 35.64 | -44.3 % | 4/4, 3/4 | 25,020 / 1,842 | 5/9 |

Per-model execution windows (first start to last end of a model's requests; host kJ / s):

| arm | Qwen window | Gemma window |
| --- | --- | --- |
| desktop + dispatch policy | 18.05 / 145.7 | 42.05 / 348.1 |
| OP15 r1 | 9.99 / 133.8 | 32.84 / 584.1 |
| OP15 r2 | 18.56 / 150.8 (unassisted) | 19.00 / 325.8 |
| OP15+Pixel r1 | 10.01 / 141.5 | 47.49 / 708.3 |
| OP15+Pixel r2 | **9.11** / 140.0 | 34.81 / 602.8 |

What the numbers say:

- **The trace-level spread is assistance availability, not the Pixel.** Two OP15 runs with the same
  configuration differ by 21 % (34.5 vs 41.9 kJ). The two runs that lost the most have causes unrelated
  to the Pixel:
  - **OP15 r2 lost all Qwen assistance.** The dispatcher ran all of Gemma first (67-393 s), and Qwen at 474 s.
    - The Gemma->Qwen re-provisioning proposal (generation 4, HTP1) was published at 395.6 s, but it
      started preparing only at 554 s and became ready at 597 s. By then the Qwen requests were nearly done.
    - In the meantime every Qwen preparation was rejected with `phone helper replacement source is not
      ready` (about 119 per request): `_unified/helper_envelopes_ops/templates.py:255`, where the ready
      layout's state is not READY.
    - The result was `PHONE_HELPER_UNAVAILABLE`.
  - **OP15+Pixel r1 lost Gemma 000/001.** Gemma was still in the page cache from the previous run, so its
    load took 3.0 s. Gemma 000/001 therefore started at 10.4 s, before the OP15 Gemma sessions were ready,
    and ran on the host (`HELPER_PHONE_SESSION_LOAD` -> `INSUFFICIENT_OPPORTUNITY`). The following
    Gemma reload took 93.7 s (36.5 s in OP15 r1).
- **Where the Pixel acts (Qwen, all four assisted), it is at least neutral.** Qwen window energy: OP15 9.99 kJ,
  OP15+Pixel 10.01 / 9.11 kJ (mean -4.4 %); time 133.8 s vs 141.5 / 140.0 s (+5 %).
- **Per-device decisions** (`analysis/DEV2_PER_DEVICE_twophone_r*.txt`). Both two-phone runs:
  - started with OP15 alone at every batch composition;
  - then OP15+Pixel challenged and won B4 (r1 at 241 s, r2 at 201 s) and B1 (295 s / 244 s);
  - had no device-set drop and no failure.

  Measured windows (fleet J/token = host measured + assumed phones; ms per token per slot):

  | Qwen composition | OP15 alone r1 / r2 | OP15+Pixel r1 / r2 | ms/token OP15 -> OP15+Pixel (r2) |
  | --- | --- | --- | --- |
  | B1 | 38.8 / 36.4 J | 28.0 / 26.4 J | 515 -> 505 |
  | B4 | 45.9 / 42.8 J | 40.0 / 37.1 J | 601 -> 688 (within the 1.25x bound) |

  B2 and B3 had too few windows to reach a challenger verdict.
- **Per-layer RPC in the campaign server** (`S41SERVERFFNSHAPE`):

  | helper | B1 | B2 | B3 | B4 |
  | --- | ---: | ---: | ---: | ---: |
  | Pixel, fixed worker | 12.8-13.4 ms | 15.5 ms | 17.8-19.4 ms | 22.7-23.3 ms |
  | Pixel, old worker (int-2) | 38.4 ms | 53.8 ms | 70.5 ms | 81.6 ms |
  | OP15 | 9.8 ms | 10.2-10.3 ms | 10.7 ms | 12.1 ms |

  Six Pixel layers at B4 now take about 140 ms, against OP15's eighteen at about 218 ms.
- **Exactness** against the desktop arm: 4-5/9 per arm, with full lengths everywhere. The differences sit at
  the same fixed positions in every phone arm and also in the unassisted OP15 r2 Qwen outputs:
  - Gemma 000 @19;
  - Qwen 003 @103 or @74;
  - Qwen 004 @111 or @155;
  - Gemma 005 @41 or @59.

  So they are batching near-ties, not phone arithmetic.
- **Gate** (per-device >= OP15 all-on within the measured noise): **PASS.**
  - On the means, OP15+Pixel is +10.0 % host energy and +6.8 % time. The spread between identical OP15
    runs is 21 %, and the pairwise results are +40.3 % (pair 1) and -14.9 % (pair 2).
  - On the only model the Pixel serves, OP15+Pixel is at least as good: -4.4 %.
  - The policy made no drop and had no failure.

## 3. Phone power diagnostic (not a meter)

`tools/power_sampler.sh` sampled both phones at about 1 Hz during every run: USB and battery voltage and
current, and the charge counter. Details are in `analyze_phone_power.py`.
- **OP15.** Its `battery/current_now` reads a constant 0, so its battery term comes only from the charge
  counter slope.
- **Pixel.** Its battery current is negative when discharging. The Pixel was checked two ways: USB plus
  battery current, and USB plus counter. The two agree within 0.1-0.4 W.

| phone / state | measured-ish W | assumed W |
| --- | ---: | ---: |
| OP15 idle (desktop arm) | 0.41 | 0.875 |
| OP15 assisting (whole run / Qwen window) | 4.0-5.3 / 3.5-6.1: pinned at its 2.5 W USB cap (500 mA SDP port) for the whole run, with the rest drawn from the battery (charge counter -128 mAh in OP15 r1) | 4.5 |
| Pixel idle | 0.49 | 0.875 |
| Pixel, Qwen window of the two-phone runs (on and off by composition) | 1.3-1.8 | 4.5 |
| Pixel, Pixel-only server requests (continuously serving at B1) | 1.9-2.0 | 4.5 |

So the 4.5 W model overstates the Pixel by a factor of 2-3 and understates the OP15 while assisting.
- Both phones were USB-powered with the charge limit on, and the readings are instantaneous or
  firmware-averaged values.
- Treat this as a diagnostic, not an energy measurement.

PHONE_POWER_## Status and headline (2026-09-25 07:35 UTC)

**Stopped: the OP15 reported `battery_notify_code` 512 (charger latched off) at 07:31 UTC, before the retry of the
two phone longtail arms started.** Per the rules, no further hardware ran. The user has to replug the OP15. At that
point the OP15 was at 80 %, the Pixel at 100 %, the rig lock was free, and no worker or adb forward was left.

| step | result |
| --- | --- |
| 1. adoption of the fixed Pixel worker | **done**. New Pixel evidence bundle: worker 5d824455, new environment sha. The receipt check accepts the new hash only through a byte-identity bridge to the qualified worker; 20 unit tests. Fresh Pixel-only server token identity PASS: 4/4 identical, and the Pixel at 100 % saves 17.3 % host energy at B1 against 10.6 % for the old worker. Idle-TERM stop re-qualified. Cost prior 7.7 ms/layer at B1 (was 28.3 ms). |
| 2. dev_v2 matched arms (5 runs) | **gate PASS**. OP15+Pixel per-device is within the measured noise of OP15 all-on: means 42.0 vs 38.2 kJ, while identical OP15 runs spread by 21 %. It is at least as good where the Pixel acts: Qwen window 9.11 / 10.01 vs 9.99 kJ. The policy picked OP15+Pixel at B1 and B4 from measured windows, with no drops. |
| 3. longtail_v1 | **2 of 4 arms done**: legacy all-desktop **536.3 kJ / 4,984 s**; desktop + dispatcher **393.0 kJ / 3,381 s (-26.7 %)**. **OP15 all-on crashed** at 1,583 s: a scheduler bug in a replan, where the baseline was rejected behind a queued host-RAM replacement (section 4). The retry (`inputs-*-lt2`, prepared) was blocked by the OP15 notify 512. OP15+Pixel was not run. |

**What needs the user:**

1. **Replug the OP15** (notify 512).
2. **Decide how to handle the replan crash.** It is not Pixel-related and it can recur in both phone arms.
   Either retry the two phone arms unchanged, accepting the risk of losing about 30 min again, or first get
   a replan-path fix: re-project residency at a memory-rejected baseline's start, as the 09-23 arrival fix
   does. The fix is not written yet.
3. **Commit checkpoint.** Nothing is committed. This round changed no scheduler code; all new files are in
   this directory.

**Resume** (after the replug, under one outer lock; a fresh prepare re-checks the USB identities):
```sh
R=/mnt/storage/s43-two-phone-eval-20260925
flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
  python3 $R/run_chain_eval.py --status $R/chains/CHAIN-lt3.jsonl --stage $R/stage \
    --prepare-script $R/prepare_campaign_eval.py --prepare $R lt3 $R/template-longtail lt \
    --arm $R/inputs-op15-lt3 --arm $R/inputs-two-phone-lt3
bash $R/analysis/post_eval.sh $R/analysis/LTFULL legacy legacy=$R/inputs-desktop-legacy-lt1 \
  desktopDP=$R/inputs-desktop-lt1 op15=$R/inputs-op15-lt3 twophone=$R/inputs-two-phone-lt3
```

## 4. longtail_v1 (step 3; 2 of 4 arms)

All arms came from the same deploy and tree: scheduler 062ce583, server a87e7772. They ran in one locked chain
(`chains/CHAIN-lt.jsonl`), in the prescribed order. The legacy arm was derived with
`prepare_trace_inputs_v2.py --drop-dispatch-policy` from the desktop arm; the only diff is `dispatch_policy`
plus ids and paths. Analysis: `analysis/LT_*`.

| longtail_v1 arm (31 requests) | dur s | CPU kJ | GPU kJ | host kJ | vs legacy | model reloads / switches | load s | identical vs legacy |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| desktop, legacy dispatcher (all-desktop baseline) | 4,984 | 377.0 | 159.3 | **536.3** | ref | 15 / 10 | 630 | ref |
| (same arm on 09-23, older code) | 4,901 | | | 523.7 | | | | |
| desktop + dispatch policy | 3,381 | 279.6 | 113.5 | **393.0** | **-26.7 %** (dur -32.2 %) | 8 / 4 | 225 | 14/31 |
| OP15 all-on | crashed at 1,583 s (FAILURE.json) | | | | | | | |
| OP15+Pixel per-device | not run (chain stops at the first failure; retry blocked by OP15 notify 512) | | | | | | | |

About these rows:

- **Dispatcher.** The dispatcher alone (work-conserving admission + model affinity) runs the same 31 requests
  with 4 model switches instead of 10. It made 12 affinity displacements.
- **Exactness.** The 17 differing outputs are host-only in both arms and keep their full lengths. They come from
  batch-composition near-ties: the dispatcher batches same-model requests differently. Examples: Gemma 021 @0,
  015 @2, 024 @1; Qwen 013 @30, 001 @43.
- **The OP15 all-on crash** (`physical/op15-lt1-failure/`):
  - Correction: the primary failure was Qwen 007 at 431.4 s (see the correction under the status table).
    The points below describe the later, secondary failure of 018, which is a Gemma request.
  - Error: `UnifiedScheduleError: qualified desktop baseline is not available:
    MEMORY_REPLACEMENT_CONFLICT_CURRENT:host-ram`, raised by `_pick_objective_candidate`
    (`_unified/automated_selection_ops/objectives.py:271`).
  - Where: while replanning Qwen 018 (`replan_automated_request` -> `_replan_priority_compaction_without_followers`
    -> `replan_commit._execute_automated_replan`).
  - Trigger: 3 s earlier, request 027 had been admitted with a cold model load (31.0 GB: 19.2 GB host RAM +
    11.9 GB VRAM). 018's hot baseline slot overlapped that exclusive replacement, and the memory ledger rejected it.
  - Why it is fatal: the replan loop has no re-projection, unlike the arrival path's
    `selection._reproject_rejected_baseline` (the 09-23 fix). So selection raises.
    `_handle_automated_replan_failure` then publishes a terminal failure, and the run aborts.
  - Before the crash: the arm was healthy, and 14 requests had been phone-assisted.
  - The same code path exists in every arm with phones. Neither desktop arm hit it.

## 5. Other findings and open items

- **OP15 Gemma->Qwen re-provisioning stalled in OP15 r2** (dev_v2, OP15-only). This cost the whole Qwen
  assistance of that run.
  - The proposal (generation 4) was published at 395.6 s, but it started PREPARING only at 554 s.
  - Every Qwen helper preparation in between was rejected with `phone helper replacement source is not ready`
    (`_unified/helper_envelopes_ops/templates.py:255`; the ready layout's state was not READY).
  - This is a #4 re-provisioning readiness gap, not investigated further.
- **Page-cache start effect.** When the next run starts with the first model still cached (3 s load), its first
  requests start before the OP15 sessions are ready, and they run unassisted. This happened to Gemma 000/001
  in two-phone r1. Arms that follow a run ending with the same model are biased against phone assistance.
- **OP15 power and battery.** While assisting, the OP15 sits at its 2.5 W USB cap for the whole run and drains
  the battery: 128 mAh in one dev_v2 run. The charger then latched off (notify 512) after about 5.5 h of this
  campaign.
- **Waiter bug (mine).** My first longtail waiter looked for a `result_status` line. A failed run has none, so
  the OP15 crash went unnoticed for about 40 min, which the rig then spent idle. `wait` checks now also match
  failures.

## 6. Files

| path | content |
| --- | --- |
| `PROGRESS.md` | timestamped log of every step |
| `make_gate_config.py`, `make_server_identity_config.py`, `prepare_campaign_eval.py`, `test_prepare_campaign_eval.py`, `run_chain_eval.py`, `analyze_phone_power.py`, `post_eval.sh`, `tools/` | tooling (section 1) |
| `SUITE_MAIN_BEFORE.log` | main suite before the runs: 141 modules / 2,021 tests, exit 0 |
| `analysis/` | `DEV2A_*`, `DEV2_*` (dev_v2, all five arms), `LT_*` (longtail desktop arms), `SERVER_IDENTITY_POWER.json` |
| `physical/` | chain status files, `GATE_CONFIG.json`, the final Pixel bundle and receipts (`qualification-dev2a/`), server identity and idle-stop results, the OP15 longtail failure |

The desktop root is `/mnt/storage/s43-two-phone-eval-20260925/`: the full run dirs, inputs dev2a/dev2b/lt1/lt2,
the templates, and the stage copy of main. The phone-side power samples are in
`/data/local/tmp/s43-two-phone-eval-20260925/` on both phones, and copies sit next to each run. The earlier
Pixel dirs and the int-2 root were only read.
