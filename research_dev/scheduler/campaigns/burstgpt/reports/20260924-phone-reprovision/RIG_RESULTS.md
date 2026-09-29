# Change #4 on the rig: dev_v2 coherentRP vs coherentEF, plus a coherentEF repeat (2026-09-24)

First hardware run of phone re-provisioning (`README.md` in this directory). Checkpoints: `PROGRESS_RP.md`. Result
files: `data-rp/` (desktop copies in `/home/zhihao/s42-reprovision-20260924-rig/results/`).

**Headline.** With re-provisioning on, the coherentEF configuration used 53.69 kJ of host energy in 898.1 s, against
71.86 kJ (coherentEF) and 69.64 kJ (coherentEF2, the repeat): -25.3 % and -22.9 %, and -44.5 % against dev2base
(96.75 kJ). The phone followed each of the 4 desktop model switches. It held 17 Qwen layers (6 before) and 24 Gemma
layers (16 before). Every swap was verified before the desktop server started executing. There were 0
WAITING_FOR_HELPER_RELEASE deferrals and 0 preparation or transition failures. The repeat run changed host energy by
3.1 %, so the gain is about 8 times the run-to-run difference. Exactness is 6/9 against 9/9 for both coherent runs:
the flips are at two token positions known from other arms plus one new position, and all three continuations read
as fluent near-ties.

## 1. Setup

- Code: the current main tree, synced into the deploy source under the rig lock with `run_arms_rp.py`. The sync
  changed exactly 28 files (`data-rp/SYNC_MANIFEST_RP.sha256`):
  - change #4;
  - the inert two-phone code, gated on `phone_helpers` / `helper_phones` / `adb-tcp`, none of which these inputs set;
  - the new `prepare_trace_inputs_v2.py --phone-resident-model-reprovisioning-json` flag and its test.

  A second rsync dry run found no difference between the staged tree and the deploy. The C++ server binary was not
  rebuilt. One lock hold covered the sync, both preflights and both runs (16:00:39 to 16:46:22 UTC).
- Inputs: `derive_inputs_rp.sh`, from the 09-23 long-tail treatment inputs and the dev_v2 trace. Both arms use the
  COALESCED_BOTH identity (`s43-coalesced-both-20260924`, 15 receipts, file sha256 `e95f0b94...`, boot `f13c7c03`),
  `server_policy_coherence`, and `coalesced-batch` qualified for both models. Diffs against
  `...-coherentEF-20260924-inputs`, with the arm name masked:
  - `dev2coherentRP` (`/home/zhihao/s42-trace-longtaildev2-coherentRP-20260924-inputs`) adds only the campaign field
    `"phone_resident_model_reprovisioning": {}`. The defaults are a 200 MB/s prior and 2 learned samples. The runner
    command carried `--phone-resident-model-reprovisioning-json {"load_bytes_per_second":200000000,
    "minimum_learned_samples":2,"mode":"resident-model"}`.
  - `dev2coherentEF2` (`...-coherentEF2-20260924-inputs`) has no difference other than the campaign id.
- Admission (`check_admission_both.py` on each arm's own physical preflight catalog): PASS for both arms and
  identical to coherentEF. The catalog link identity is `sha256:866f628e...`. Each desktop parent of each model has
  one QUALIFIED `operator_split:coalesced-batch` helper, and the split-row variants are SHADOW. Qwen resolves to
  payload-40960, Gemma to payload-61440, and the Gemma catalog parallel=2 to payload-38400.
- Order and outcome:
  - RP preflight 16:01-16:07 PASS, run 16:07:42-16:23:07, exit 0, RESULT PASS 9/9.
  - EF2 preflight 16:23-16:28 PASS, run 16:28:29-16:46:22, exit 0, RESULT PASS 9/9.
  - Battery at every check: notify code 0 (never 512), level 80 -> 78 %, USB 500 mA, status 4 until the RP run,
    then 2 (charging), temperature 26-34 C.

## 2. Energy, duration, exactness

Host = RAPL package + NVML board. Outputs are compared token by token with dev2base (`analyze_longdecode_pair.py`).

| arm | duration s | host kJ (CPU + GPU) | vs dev2base kJ / s | vs coherentEF kJ / s | outputs identical to dev2base |
| --- | ---: | ---: | ---: | ---: | --- |
| dev2base | 1,154.3 | 96.75 (64.90 + 31.85) | - | - | - |
| dev2plainEF | 1,035.1 | 76.31 (46.37 + 29.93) | -21.1 % / -10.3 % | +6.2 % / +12.0 % | 6/9 |
| dev2coherentEF | 924.5 | 71.86 (43.50 + 28.36) | -25.7 % / -19.9 % | - | 9/9 |
| **dev2coherentEF2** (repeat) | 1,042.8 | 69.64 (39.91 + 29.73) | -28.0 % / -9.7 % | **-3.1 % / +12.8 %** | 9/9 (and 9/9 vs coherentEF) |
| **dev2coherentRP** (#4) | **898.1** | **53.69 (26.03 + 27.65)** | **-44.5 % / -22.2 %** | **-25.3 % / -2.9 %** (vs EF2 -22.9 % / -13.9 %) | 6/9 |

The mean of the two coherentEF runs is 70.75 kJ, so coherentRP is 24.1 % below it.

The three coherentRP mismatches:

| request | first differing token | dev2base continues | coherentRP continues | seen before? |
| --- | ---: | --- | --- | --- |
| Qwen 003 | 103 | "a neural processing unit (N" | "with a dedicated NPU or" | yes: plainEF, coherentsr |
| Qwen 004 | 57 | ", which might be a mistake" | ". Maybe they want a thorough" | new position (other arms flipped at 210) |
| Gemma 005 | 146 | "large matrix multiplications much faster." | "the dense matrix multiplications that characterize" | yes: dev2plain, dev2coherent, plainEF |

Both sides of every mismatch are fluent, like the near-tie flips of section 4.4 of the coherent-policy report. The
streams carry no logprobs, so "near tie" is an inference. The likely cause is that more FFN layers run on the phone:
17 instead of 6 for Qwen and 24 instead of 16 for Gemma, which means more HTP f16 arithmetic.

**Where the energy went** (`phase_energy.py`: host energy integrated over the union of each model's adaptive windows;
totals reproduce RESULT to 0.01 kJ):

| arm | Qwen decode | Gemma decode | all decode windows | outside decode windows |
| --- | --- | --- | --- | --- |
| coherentEF | 283.3 s, 32.80 kJ | 338.4 s, 27.25 kJ | 621.7 s, 60.04 kJ | 302.8 s, 11.82 kJ |
| coherentEF2 | 276.0 s, 29.78 kJ | 337.4 s, 25.28 kJ | 613.3 s, 55.06 kJ | 429.5 s, 14.58 kJ |
| coherentRP | **245.9 s, 18.23 kJ** | **337.7 s, 22.51 kJ** | **583.6 s, 40.74 kJ** | 314.5 s, 12.95 kJ |

coherentRP against coherentEF (-18.17 kJ):

| component | change |
| --- | ---: |
| Qwen decode | -14.57 kJ |
| Gemma decode | -4.74 kJ |
| outside decode windows | +1.13 kJ |

coherentRP against coherentEF2 (-15.95 kJ):

| component | change |
| --- | ---: |
| Qwen decode | -11.55 kJ |
| Gemma decode | -2.77 kJ |
| outside decode windows | -1.63 kJ |

The gain is almost all decode energy, and most of it is Qwen.

## 3. Re-provisioning timeline (coherentRP)

`analyze_reprovision.py`; full tables in `data-rp/REPROVISION_RP.md`. Times are seconds from the start of the paid
interval.

| READY s | gen | changed | phone layers | why |
| ---: | ---: | --- | --- | --- |
| 24.3 / 43.4 / 56.1 | 1-3 | HTP0, HTP1, HTP2 | Gemma 8 -> 16 -> 24 | startup PROPORTIONAL (no desktop knowledge yet; only Gemma arrived) |
| 105.0 / 120.8 / 133.1 | 4-6 | HTP1, HTP0, HTP2 | Gemma 16 + Qwen 6 -> Gemma 8 + Qwen 12 -> **Qwen 17** | FOLLOW Qwen, source `loading`, leader 002 |
| 369.0 / 380.5 / 391.3 | 7-9 | HTP2, HTP0, HTP1 | Gemma 8 + Qwen 12 -> Gemma 16 + Qwen 6 -> **Gemma 24** | PROPORTIONAL (Qwen still hot on the desktop) then FOLLOW Gemma, leader 005 |
| 690.8 / 705.2 / 721.2 | 10-12 | HTP1, HTP0, HTP2 | -> **Qwen 17** | PROPORTIONAL then FOLLOW Qwen, leader 006 |
| 849.0 / 859.5 / 870.2 | 13-15 | HTP1, HTP2, HTP0 | -> **Gemma 24** | FOLLOW Qwen+Gemma then FOLLOW Gemma, leader 007 |

Each stage changes exactly one session, and every intermediate layout is feasible. Qwen reaches 17 layers because the
live phone limit at the second-stage decision was 9.46 GB (at 121 s) and 9.30 GB (at 691 s), below the 9.626 GB that
18 layers need. HTP2 therefore got the 5-layer Qwen shard (2.67 GB), as the README caveat predicted.

Session loads (15; SESSION_LOADING -> SESSION_VERIFIED):

| load group | range |
| --- | --- |
| first three (startup, Gemma) | 12.2-22.3 s (127-233 MB/s) |
| the other twelve | 10.1-15.6 s per session (201-281 MB/s) |

The learned rate went from 161 MB/s (3 samples) to 217 MB/s (15 samples).

Swap against desktop load:

| switch | dispatch (window start) | predicted window end | phone target layers verified | desktop server executing | first decode window |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 002 | 89.4 | 144.4 | 133.1 (43.7 s) | 162.4 | 165.7 |
| Gemma 005 | 356.2 | 423.9 | 391.3 (35.1 s) | 403.1 | 413.8 |
| Qwen 006 | 675.5 | 730.5 | 721.2 (45.7 s) | 759.8 | 768.4 |
| Gemma 007 | 838.1 | 905.7 | 870.2 (32.1 s) | 879.6 | 883.3 |

All 4 swaps finished inside the predicted desktop-load window and 9-39 s before the desktop executed. Every request
after a switch (002-007) started decoding with its model's full phone layer count (17 Qwen / 24 Gemma). The
`fits_load_window` values:

- The first FOLLOW stage of the first swap (89.4 s) recorded `fits_load_window = false`: the target swap was
  estimated at 59.8 s at the early learned rate of 161 MB/s, longer than the 55 s window. The actual swap took
  43.7 s.
- Every later stage recorded `true`.

`desktop_reprovision` blocks:

| count | what |
| ---: | --- |
| 1,225 | EVALUATED events with `desktop_reprovision` |
| 10 | DESKTOP_REPROVISION stages |
| 3 | PROPORTIONAL_SPLIT stages |
| 4 | HOLD_IDLE_MODEL |
| 1,202 | REPROVISION_RETAINED |
| 0 | `blocked_session_ids` |
| 1,167 | evaluations with all three sessions in use (while decoding): 1,166 RETAINED, 1 HOLD_IDLE_MODEL |
| 0 | PREPARATION_DEFERRED / WAITING_FOR_HELPER_RELEASE |

Phone layers of the decoding model, time-weighted over that model's decode windows:

| arm | Qwen | Gemma |
| --- | ---: | ---: |
| plainEF | 6.0 | 16.0 |
| coherentEF | 6.0 | 16.3 |
| coherentEF2 | 6.0 | 17.4 |
| coherentRP | **17.0** | **22.2** |

In the coherentEF runs the layout settled on Gemma 16 + Qwen 6 at t = 98.5 s (coherentEF) and 174.9 s (coherentEF2)
and never changed; the other model's layers sat idle while one model decoded. In coherentRP, Gemma averages 22.2 rather than 24 only because 000 started
decoding at 15.5 s, before the startup loads finished.

## 4. Phone share, calls, mixed passes

Phone share = decode-window tokens under a phone policy. Calls come from `S41SERVERFFNSHAPE`, over all processes of a
server role.

| arm | Qwen phone share | Gemma phone share | Qwen calls by rows | Gemma calls by rows | Qwen mixed passes |
| --- | ---: | ---: | --- | --- | ---: |
| plainEF | 278/595 (46.7 %) | 692/797 (86.8 %) | 1,692 x 1 | 11,760 x 1 | 121 |
| coherentEF | 234/595 (39.3 %) | 696/797 (87.3 %) | 96 x 1 + 660 x 2 | 11,856 x 1 | 1 |
| coherentEF2 | 342/595 (57.5 %) | 756/797 (94.9 %) | 1,962 x 1 + 66 x 2 | 13,320 x 1 | 5 |
| coherentRP | **522/595 (87.7 %)** | 653/797 (81.9 %) | 5,423 x 1 + **1,802 x 2** | 15,816 x 1 | 7 |

In coherentRP the Qwen pair 003/004 overlapped for 68.7 s. It ran 212 phone slot tokens at batch 2, which is
106 two-slot passes x 17 layers = 1,802 two-row calls. Gemma had no pair in any arm, so every Gemma call is 1-row.

Gemma's share fell from 87-95 % to 81.9 % for two reasons:

- **Gemma 000:** 24/116 phone tokens against 44/116 in coherentEF. The startup loads were slower (HTP0 22.3 s against
  18.0 s), and an ADB dropout hit at 24-44 s (see caveats). This is startup noise, not a change-#4 effect.
- **Gemma 007:** 0 of its 26 window tokens ran on the phone (`INSUFFICIENT_OPPORTUNITY` x7). See defect 1 in
  section 6.

## 5. J/token by batch composition

**Window receipts.** Each cell gives the fleet J per produced token (the assumed 4.5 W phone included, divided by
`active_batch`), the ms per slot token, and the token count in brackets. All valid windows are counted. Eligible-only
figures are in `data-rp/TABLES_RP.md` and give the same picture.

| arm | Qwen b1 host | Qwen b1 phone | Qwen b2 host | Qwen b2 phone | Gemma host | Gemma phone |
| --- | --- | --- | --- | --- | --- | --- |
| coherentEF | 75.0 J, 617 ms (321) | 60.5 J, 591 ms (18) | 37.0 J, 592 ms (26) | 30.9 J, 596 ms (216) | 57.0 J, 456 ms (89) | 32.9 J, 428 ms (696) |
| coherentEF2 | 73.2 J, 610 ms (25) | 60.1 J, 582 ms (318) | 38.7 J, 633 ms (215) | 37.7 J, 700 ms (24) | 52.7 J, 432 ms (28) | 32.3 J, 430 ms (756) |
| coherentRP | 72.5 J, 625 ms (26) | **36.6 J, 518 ms (310)** | 39.1 J, 670 ms (34) | **19.7 J, 542 ms (212)** | 57.9 J, 464 ms (132) | **24.0 J, 424 ms (653)** |

The phone policy got both cheaper and faster with more layers:

| model and batch | J/token, coherentEF runs -> coherentRP | ms/token, coherentEF runs -> coherentRP |
| --- | --- | --- |
| Qwen batch 1 | 60 -> 37 | 582-591 -> 518 |
| Qwen batch 2 | 31 -> 20 | 596 -> 542 |
| Gemma | 33 -> 24 | about 424-430 (unchanged) |

Against the host, the Qwen saving is now about 50 % at both batch sizes, far above the 19 % that the n = 1 server
probe needs.

**Wall-clock host energy by phase** (`phase_energy_batch.py`: host only, over the union of that model's windows at
that batch size, divided by the window tokens):

| arm | Qwen batch-1 phase | Qwen batch-2 phase | Gemma phase |
| --- | --- | --- | --- |
| plainEF | 199.8 s, 21.91 kJ, 64.6 J/token | 113.0 s, 13.36 kJ, 54.3 | 337.1 s, 28.04 kJ, 35.7 |
| coherentEF | 211.8 s, 25.34 kJ, 73.9 | 72.7 s, 7.62 kJ, 31.5 | 338.4 s, 27.25 kJ, 34.7 |
| coherentEF2 | 200.2 s, 20.71 kJ, 60.4 | 77.1 s, 9.24 kJ, 38.7 | 337.4 s, 25.28 kJ, 32.2 |
| coherentRP | **176.9 s, 12.90 kJ, 38.4** | **70.3 s, 5.50 kJ, 22.4** | **337.7 s, 22.51 kJ, 28.7** |

**Server verdicts (Qwen)** and decision reasons:

| arm | verdict[1] | verdict[2] |
| --- | --- | --- |
| coherentEF | host, `SERVER_PAIR_NOT_IMPROVED` at 153.5 s | phone |
| coherentEF2 | phone | host, `SERVER_PAIR_NOT_IMPROVED` at 334.9 s |
| coherentRP | phone | phone |

- The repeat inverted both of coherentEF's verdicts. This confirms the section 4.4 diagnosis: at n = 1 the 19 %
  threshold makes each verdict noise-limited in both directions when the true saving is 15-20 % (6 Qwen layers).
  With 17 layers, coherentRP decided phone at both batch sizes.
- coherentRP's only Qwen host decisions are the 8 `SERVER_COMPARISON_HOST_WINDOW`. Qwen 006 re-decided verdict[1]
  (phone at 774.7 s) because its layout generation (12) was new.
- Evidence-fix markers:

  | marker | coherentEF | coherentEF2 | coherentRP |
  | --- | ---: | ---: | ---: |
  | `HELPER_PHONE_SESSION_LOAD` decisions (Gemma 000) | 15 | 0 (000 started after its sessions were ready) | 15 |
  | `TOKEN_STREAM_CATCH_UP` windows | 2 | 0 | 10 (8 Gemma, 2 Qwen) |
  | `CANDIDATE_REQUALIFIED` | 0 | 0 | 0 |

## 6. Findings on change #4 (not fixed here)

1. **Every swap resets the server's verdicts, even for a geometry decided before.**
   - The coherence group key contains the phone layout generation, and each re-provision stage bumps it.
   - Gemma 007 arrived at generation 15. That generation has the same geometry (`0249915b...`, 24 layers) as
     generation 9, on which 005 had run 603 phone tokens. 007 nevertheless found an empty server policy
     (`verdicts {}`, owner 007) and no historical evidence (every candidate: `historical_groups 0`). With 29 tokens it
     could not afford a probe (`INSUFFICIENT_OPPORTUNITY`), so it ran all on the host: about 26 x (57.9 - 24.0) J
     = about 0.9 kJ.
   - In both coherentEF runs, 007 inherited 005's phone verdict because the layout never changed.
   - The same reset made Qwen 006 re-probe (a few host tokens).
   - Suggested fix: carry a model's verdicts across layout generations when that model's shards (geometry,
     operator plan) are unchanged, or key the group by the model's own geometry instead of the global generation.
2. **Per-token portfolio re-evaluation while the other model is queued.**
   - `_reevaluate_pending_phone_layout_at_boundary` did not take its early return, so it logged 1,202
     REPROVISION_RETAINED evaluations, about one per decode token: 615 during Gemma 005 while Qwen 006 was queued
     and not on the phone. coherentEF logged 35 EVALUATED events in total, coherentEF2 165.
   - The queue-work buckets and route evidence were unchanged between those events, so the trigger is probably the
     pending/confirmed-selection path. Not root-caused.
   - Cost:
     - median publication delay 7.8 ms per evaluation, 41 s summed over the run;
     - RESULT.json 41 MB against 20 MB;
     - possibly the extra `TOKEN_STREAM_CATCH_UP` windows (542-563 s in 005, 812 s in 006, both in such periods).
   - Decode latency was not worse (Gemma phone 424 ms/token against 428-430 ms).
3. **The first-swap estimate was pessimistic.** The rate learned from the three startup loads (161 MB/s, including a
   127 MB/s first load under an ADB dropout) marked the first Qwen swap `fits_load_window = false`, though it fitted
   with 11 s to spare. It had no effect on behaviour: the swap was not deferred.
4. **17 Qwen layers, not 18.** The live limit at the second-stage decisions was 9.30-9.46 GB. Once a session was
   packed short, the third Qwen session kept the 5-layer shard at 705.6 s, although the live limit had risen to
   9.69 GB by then (selected limit 9.626 GB). This is the "no regrowth" caveat of the README; one more Qwen layer is
   worth at most about 1/17 of the Qwen phone saving.
5. **Exactness 6/9.** See section 2. It should be watched on the 24-request trace; nothing here points to an error in
   the re-provisioning path, which does not touch the numerics.

## 7. Noise estimate (coherentEF vs coherentEF2, identical inputs and code except the 28-file sync)

| component | coherentEF2 minus coherentEF |
| --- | ---: |
| whole-run host energy | -2.22 kJ (-3.1 %) |
| decode windows | -4.98 kJ |
| outside decode windows | +2.76 kJ |
| duration | +118.3 s (+12.8 %) |

Where the difference comes from:

- **Qwen verdicts.** Qwen decode was -3.0 kJ, because the verdict inversion (section 5) traded batch-1 host for
  batch-2 host, and batch 1 is the larger phase.
- **Gemma.** Gemma decode was -2.0 kJ, because 000 ran almost all on the phone.
- **Duration: a cold page cache.** The first Gemma desktop load took 68.8 s against 4.3 s in coherentEF (desktop
  llama-server log: 63 s between reading the metadata and `load_tensors`, i.e. the 23.8 GB GGUF came from disk, not
  from the page cache). 000 started decoding at 81.6 s instead of 16 s.
- **Desktop load times per switch** vary between runs (seconds, first to last):

  | arm | Gemma | Qwen | Gemma | Qwen | Gemma |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | coherentEF | 4.3 | 42.8 | 31.1 | 75.0 | 54.7 |
  | coherentEF2 | 68.8 | 70.0 | 34.5 | 84.2 | 63.6 |
  | coherentRP | 4.2 | 66.4 | 41.0 | 72.3 | 35.5 |

  This is the known page-cache confound: Qwen at 29.5 GB and Gemma at 23.8 GB do not both stay cached.
- **Exactness** is 9/9 in both runs, and the two runs are 9/9 identical to each other.

Conclusion: whole-run host energy repeats within about 3 % (2.2 kJ apart), and whole-run duration does not repeat (13 %;
pre-decode phases, mostly desktop load). The coherentRP gain of 16-18 kJ is 7-8 times the observed difference. It
sits in the decode windows (-14 to -19 kJ), where the page-cache confound does not reach.

## 8. Caveats

- One run of coherentRP and two of coherentEF. The noise estimate comes from one pair.
- Host energy only. Phone energy is the assumed 4.5 W active / 0.875 W idle model (in the window J/token figures,
  not in host kJ). The phone worked more (Qwen calls x 9.6, Gemma calls +33 %), so fleet gains are slightly smaller
  than host gains. From `analyze_longdecode_pair`:

  | arm | phone active s | assumed phone kJ | fleet kJ (host + assumed phone) |
  | --- | ---: | ---: | ---: |
  | coherentEF | 143.7 | 1.33 | 73.19 |
  | coherentEF2 | 174.1 | 1.54 | 71.18 |
  | coherentRP | 387.8 | 2.19 | 55.88 (-23.6 % vs coherentEF) |

  The assumption is weakest for re-provisioning: the 15 session loads (4 in the other runs) are phone flash + HTP
  upload work that the model does not price separately.
- ADB dropouts:

  | arm | ADB device missing (`PHONE_TELEMETRY_DEFERRED ... device not found`) |
  | --- | --- |
  | coherentRP | 24.4-43.9 s |
  | coherentEF2 | 21.2-32.0 s |
  | plainEF | 2 events |

  All fell during the startup session loads. They recovered without failures but slow the startup loads.
- dev_v2 has 4 desktop switches, 9 requests and one same-model pair. The swap overhead is small against this trace's
  decode phases (each switch is 32-46 s of phone loading, fully hidden behind 35-72 s desktop loads here). Short
  requests after a swap pay the verdict reset (finding 1).
- The deployed tree is the main tree as of 11:55 EDT. Another agent has since added a `--dispatch-policy` flag and test
  to `prepare_trace_inputs_v2.py` in the main tree; that change is not in the deploy and plays no part in these runs.

## 9. Reproduce

```sh
# desktop, rig dir /home/zhihao/s42-reprovision-20260924-rig (copies of this directory's scripts)
bash derive_inputs_rp.sh dev2coherentRP dev2coherentEF2        # outside the lock
nohup python3 -u run_arms_rp.py dev2coherentRP dev2coherentEF2 > RUN_ARMS_RP-1.log 2>&1 < /dev/null &
# analysis, off the desktop on copies (snapshots excluded); ../20260924-coherent-policy-coalesced/ has analyze_ef.py etc.
python3 analyze_ef.py --arm base=<dev2base run> --arm plainEF=... --arm coherentEF=... --arm coherentEF2=... \
    --arm coherentRP=... --out ANALYSIS_RP.json && python3 tables_ef.py ANALYSIS_RP.json > TABLES_RP.md
python3 analyze_reprovision.py --arm coherentEF=... --arm coherentEF2=... --arm coherentRP=... --out REPROVISION_RP.json --md REPROVISION_RP.md
python3 ../20260924-coherent-policy-coalesced/phase_energy.py label=<run> ...; python3 phase_energy_batch.py label=<run> ...
python3 analyze_longdecode_pair.py --baseline <dev2base run> --treatment <run> --output PAIR_x.json   # rig dir copy
```

Run dirs: `<inputs>/run-dev2coherentRP-1/run`, `<inputs>/run-dev2coherentEF2-1/run`.
