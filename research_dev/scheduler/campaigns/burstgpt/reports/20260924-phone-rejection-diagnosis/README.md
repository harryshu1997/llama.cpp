# Why the dev-trace run went back to the desktop when the phone measured better (2026-09-24)

This is a read-only diagnosis. No scheduler code, hardware, adb or commits were touched. Runs analysed:

- **dev**: `/home/zhihao/s42-trace-longtaildev-treatment-20260923-inputs/run-treatment-1/run` (6 requests: 3 Qwen, 2 Gemma, 1 Llama overlay; one decode slot at a time, `active_slots_peak = 1`).
- **run-5**: `/home/zhihao/s42-trace-longtail-treatment-20260923-inputs/run-treatment-5/run` (31 requests, 28 of them large).

Labels used below: **[M]** measured, i.e. read straight from the run artifacts. **[R]** replayed: the run-time controller functions were re-run on the recorded windows (`scripts/replay_*.py`). This is deterministic; the only assumption is the rebuilt session state. **[I]** inferred: a causal or counterfactual claim, such as the energy a fix would have saved.

## Source identity

Both runs used the same source: `SOURCE_MANIFEST.json` head `347e63bbd`, with identical hashes for all 449 files. Six adaptive files have changed in the working tree since then, because another agent is editing them. `run-source/` keeps the run-time copies, and `run-source/MANIFEST.txt` lists the sha256 of every file cited here.

All `file:line` references are to the run-time versions under `research_dev/scheduler/`. They are identical to today's working tree for `sequencing.py`, `promotion.py`, `bounds.py`, `candidates.py`, `budgeting.py`, `history.py`, `helpers.py`, `adaptive_decode_contracts.py` and `_unified/runtime_requests.py`. `windows.py`, `coherence.py`, `completion.py`, `reporting.py`, `adaptive_decode_state.py` and `_unified/adaptive_decode_control.py` differ; use `run-source/`. In today's `windows.py`, lines after 612 shift by +1.

Session parameters [M] (ASSISTANCE_DECISION events and `RESOLVED_CONFIGURATION.json`):

- `minimum_energy_saving_ppm = 10000` and `maximum_latency_ppm = 1250000`, from the campaign through `_adaptive_start_config`.
- `uncertainty_ppm = 100000`, `warmup_windows_per_policy = 1`, 4-token windows, `server_policy_coherence = false`.
- `allow_assumed_phone_power_for_operational_selection = true`. It comes from the phone power profile (`allow_assumed_for_scheduling: true`) via `_unified/adaptive_decode_control.py:371-399,451-472`. The top-level `adaptive_controller_configuration` in `RESULT.json` shows `false`, `1000000` and `50000`; those are the controller defaults, which each session replaces.
- `helper_evidence_state = LEARNING` for every request in both runs.

## TL;DR

1. **The windows were not ineligible.** `measurement_eligible` is written to JSON only when it is false (`adaptive_decode_contracts.py:808-809`); a missing key means true (`:904`).
   - In dev, 224 of 258 windows are eligible [M]. The 34 ineligible ones are ineligible by design: 13 warm-up windows, 17 transition windows, 3 context-unavailable end windows and 1 released tail.
   - What is false for every window is the strict certification property `energy_measurement_eligible` (`contracts.py:752-759`). The campaign runs with `energy_attribution_kind = diagnostic` and the phone power is `ASSUMED_4P5W`.
   - The controller does not use that strict property here. It uses the operational rule (`bounds.py:12-20`: the assumed-power flag plus `measurement_eligible` plus the `ASSUMED_4P5W` evidence id), which admits these windows.
   - Eligibility is required to promote or keep a phone policy, but it was not what blocked 000 or 001.
2. **Two different mechanisms ended 000 and 001 on the desktop.** Neither was a history or prior test.
   - **001 (Gemma).** P100 qualified with one phone window vs one host window, by 0.11 J/token. The second phone window moved the mean slightly (47.06 to 47.95 J/token) but did not narrow the +/-10 % uncertainty band, because `isqrt(2) = 1`. The incumbent re-check `_qualifies` then failed at promotion.py:160 (phone upper bound 52.75 > required 51.87 J/token). `_continue_best("INCUMBENT_NO_LONGER_BENEFICIAL")` fell back to the host, and **host windows never re-evaluate the candidates**, so it stayed on the host for 515 tokens. Replay shows P100 would have re-qualified after 3 more host windows, at token 75.
   - **000 (Qwen).** The only host reference window (W1) is an observation artifact: 40.9 J/token and 316 ms/token, against a steady 75.6 J/token and 608 ms/token. It is the catch-up window after a 3.4 s stall in the scheduler's token stream. The LEARNING paired gate compared P100's single window (47.2 J/token) against W1 and **permanently eliminated P100** at token 21. `_update_elimination` would also have recorded `LATENCY_BOUND_EXCEEDED`, against the artifact's 347 ms latency upper bound. P50 was eliminated later against a host mean that still contained W1. Replay: without the permanent elimination, P100 qualifies at token 35.
   - The P50 probe during the phone's HTP2 session load did not remove P100; P100 was already eliminated. That probe's stalled window was a warm-up window, so it never became evidence. The controller has no notion of phone-side residency transitions.
3. **Why 002 and 004 were different.** Qwen's phone saving is about 37 %, twice the roughly 19 % that the +/-10 %/+/-10 % bands require when each side has 1-3 windows.
   - 002 passed on its first pair, with no history.
   - 004 was seeded from 002's completed group. Replay: the seed disappears if 002's group is dropped.
   - Gemma's saving is about 19-22 %, right at that threshold, so a Gemma request's outcome depends on +/-2-3 % noise in its single host reference window.
4. **Run-5 did not end any request on the host through the dev pattern.** The 10 requests that ended on the host were all rejected at batch 2, where the phone was measured *worse* (alternating passes next to a host co-tenant), or were never measured.
   - Related losses [I]: 4 requests returned to batch 1 and never re-probed, because exploration energy spent at batch 2 had exhausted the per-request budget. That is 372 host tokens, about 8.2 kJ.
   - One stall artifact during a batch-1 verification (027) cost about 37 extra host tokens, about 1.1 kJ.
   - In the dev run, the same two mechanisms cost about 8.5 kJ [I], 11.7 % of the run's 72.9 kJ fleet energy.
5. **Proposed fixes** (section 5). Each keeps promotion gated by the same bounds. Each changes only when a *negative* decision becomes permanent, or allows a positive decision to be re-checked.
   - F1: re-qualify candidates from host windows, and do not demote while the evidence is only inconclusive.
   - F2: no permanent elimination from a one-window LEARNING pair.
   - F3: a stall/catch-up guard on window eligibility.
   - F4: treat phone session loads on the helper device as a disturbance.
   - F5: charge exploration per batch context, and keep per-batch evidence.
   - F6 (needs sign-off): make the uncertainty band monotone in n.

## 1. Window eligibility (question 1)

| dev windows | count | why | code |
| --- | ---: | --- | --- |
| eligible (key absent in JSON) | 224 | stable window after the policy's warm-up | `windows.py:302-311` |
| warm-up (first stable window of a policy) | 13 | `warmup_seen < warmup_windows_per_policy` | `windows.py:307-311` |
| transition (control pending, closed at ack) | 17 | `_append_window(..., stable_window=False)` | `windows.py:149-166` |
| context unavailable (last window of 000, 003, 004) | 3 | `observation.execution_context_available=False` ("live decode membership is empty") | `windows.py:302,500-501` |
| released-slot tail (001) | 1 | `discard_stale_window` | `windows.py:684-750` |

Across the whole file (35 groups, 30 of them from the input store) 1479 of 1790 windows are eligible. That matches `RESULT.adaptive_observation_store_state.valid_windows = 1479`; its `energy_valid_windows = 1123` are all input-store windows with isolated or matched-ABBA attribution.

Every one of the 258 dev windows has `energy_attribution_kind = "diagnostic"` and the `ASSUMED_4P5W` evidence id [M], so the strict `energy_measurement_eligible` property (`contracts.py:752-759`) is false for all of them. That is intended: no certified energy claim is possible from this run, and nothing below should change that.

The decisions use the operational rule instead. `_operational_energy_eligible` (`bounds.py:12-20`) accepts a row when three things hold: the flag `allow_assumed_phone_power_for_operational_selection` is set, `measurement_eligible` is true, and the row carries `ASSUMED_4P5W`.

**Is eligibility required to promote or keep a phone policy?** Yes.

- `_valid_records` and `_current_valid_records` (`bounds.py:23-74`) keep only rows that are operationally eligible, are at the session's current `active_batch`, have valid output and have no failure.
- `_qualifies` returns false unless both the host and the candidate have such rows (`promotion.py:117-149`).
- The latency bounds use only `measurement_eligible` rows (`bounds.py:77-167`).
- `_learning_probe_improves` needs eligible rows on both sides (`promotion.py:198-235`).
- The incumbent is re-checked with `_qualifies` at every exploitation boundary (`sequencing.py:644-647`).

**Which condition failed?**

- Batch: no. Every dev window is at batch 1.
- Overlapping desktop activity: no. `external_activity_sha256` is `b95f9ad...` on every window, and it never changed.
- Isolation, i.e. the strict energy eligibility: yes, it fails for every window, by construction. It is not what the operational decisions use.
- The real problem is the quality of the eligible evidence (section 2).

## 2. Decision paths (question 2)

### 2.1 Request 001 (Gemma, 578 tokens): an inconclusive band flip, then no re-evaluation

Windows [M] (J/token, ms/token; `*` marks eligible):

```
B   W1*  58.21 462.9 | W0 warm-up, W2 transition
P100 W4* 47.06 441.5 | W3 warm-up, W5 transition     tok 20: incumbent := P100
P75  W7* 51.72 430.4 | P50 W10* 54.64 434.9 | P25 W13* 56.14 448.6
P100 W15* 48.85 453.9 (exploitation)                  tok 60: INCUMBENT_NO_LONGER_BENEFICIAL
B    W17..W145* 59.3 468 (129 eligible windows, tokens 63-578)
```

Call path at token 60 (t = 244.9 s) [M] (events) and [R] (`data/replay_001_as_run.txt`):

1. `record_window` takes its normal branch for the phone window W15 (`windows.py:658-672`):
   - `_update_elimination(P100)`: latency passes, and the ENERGY_DOMINATED check is skipped because P100 is the incumbent (`promotion.py:96`).
   - `_consider_incumbent(P100)` sees `_qualifies` false and changes nothing.
2. `_next_after_window` (`sequencing.py:471`) finds the state EXPLOITING and the stage still `"candidate"`, so the probe-restart branch at `sequencing.py:640` does not fire. The incumbent check at `sequencing.py:644-647` calls `_qualifies(P100)`, which returns false.
3. Inside `_qualifies` (`promotion.py:104-188`, no coherence, so it compares current records only):
   - The LEARNING gate passes (47.95 < 58.21 J/token; 447.7 <= 462.9 x 1.25 ms/token).
   - Host `_bounds` = (58.21, **52.39**, 64.03, 462.9, 509.2) and P100 `_bounds` = (47.95, 43.16, **52.75**, 447.7, 492.5). The band is `max(20000, 100000 // isqrt(n))` = 10 % for n = 1..3 (`bounds.py:191-196`).
   - `required_upper` = 52.39 x 0.99 = **51.87** J/token (`promotion.py:155-159`), and 52.75 > 51.87, so it returns False at **`promotion.py:160`**.
   - This is the confidence/sample-size rule: phone upper bound vs host lower bound, with +/-10 % on each side.
   - The paired test, the latency bound and history played no part. There was no history (section 3), and nothing was eliminated (`eliminated_policy_reasons = {}` for the whole request).
4. `_continue_best` (`promotion.py:47-54`) calls `_best_valid_policy` (`promotion.py:31-44`). The incumbent fails. P100, P75, P50 and P25 all fail `_qualifies`. `_ticket_fallback` returns None in LEARNING (`candidates.py:213-214`), so the host is chosen.
5. Every later host window takes the path `record_window`, then `_follow_coherent_policy` (None: no co-tenant), then `_next_after_window`, which reaches `sequencing.py:648-654` and simply reopens the host.
   - `_update_elimination` and `_consider_incumbent` run only for **phone** windows (`windows.py:659-665`). The 129 host windows that tightened the host band were never used to re-check the candidates.
   - 001 logs 130 `INCUMBENT_NO_LONGER_BENEFICIAL` events. Only the one at token 60 is a decision. The other 129 are the sticky `zero_assistance_reason` attached to host `WINDOW_OPENED` decisions (`_unified/adaptive_decode_control.py:934-943`).

The same arithmetic passed at token 20, with W4 alone: phone upper 51.76 <= required 51.87, a margin of 0.11 J/token [R]. So the second phone window, 1.8 J/token worse, flipped a qualified incumbent to "not qualified", because `isqrt(1) = isqrt(2) = isqrt(3)`.

At W15, `_qualification_needs_more_evidence(P100)` is True and `comparable_measurement_targets` = (4, 4) [R]. The existing "resolve with more windows" path (`promotion.py:317-354`) would have been used by `_finish_verification` or `_select_probe_winner`. The incumbent-drop path at `sequencing.py:645-647` bypasses it.

Counterfactuals [R]:

- With the recorded host windows, P100 **re-qualifies after W19, at token 75**: host n = 4, band 5 %, required upper 55.72 J/token >= 52.75.
- With continuous `sqrt` bands, P100 never drops at W15 (upper 51.34 <= 51.87; `data/replay_001_sqrt_bands.txt`).

### 2.2 Request 000 (Qwen, 141 tokens): an artifact reference, then permanent LEARNING eliminations

Windows [M]:

```
B    W0  132.5 J 1114.7 ms (warm-up)  W1* 40.92 315.6   W2 38.91 308.0 (transition)
P100 W3  47.92 529.1 (warm-up)        W4* 47.23 529.7   W5 transition            tok 21: P100 eliminated
P50  W6 195.70 4150.4 (warm-up, 71.2-87.8 s)            W7 transition            tok 28: PROBE_INCOMPLETE
B    W8* 78.46 709.5                                    W9 transition            tok 35: retry P50
P50  W10* 62.65 517.0                                   W11 transition           tok 42: P50 eliminated
B    W12..W34* 74.7-81.1 (steady 75.6 J, 608 ms)                                 tok 45: MEASURED_REJECTION
```

**Why W1 is an artifact** [M] timing, [I] cause.

- The window boundaries are the scheduler's token observation times (`DECODE_BOUNDARY_OBSERVED.token_observed_at_us`). Token 2 was read at 59.127 s but its boundary was processed at 62.538 s, a 3.41 s lag. Tokens 3-5 then arrived within 17 ms at 62.66 s, and tokens 7-9 within 2 ms at 63.94 s.
- No scheduler event was recorded between `HELPER_EXPANDED` (59.127 s) and the phone-residency `SELECTION_OBSERVED` (62.625 s). The serialized runtime was presumably busy with the phone-layout selection that proposed generation 3 at 63.18 s.
- W0 (1115 ms/token) absorbed the stall, and W1 is the catch-up window: 4 tokens in 1.26 s, when the server's steady rate is about 0.6 s/token.
- W0-W2 together average 70.8 J/token and 579 ms/token, which matches the steady state. W1 alone is -46 % in both energy and latency vs the request's later host median [M] (`data/window_noise.txt`). Among the 215 dev windows that have at least two others of the same request, policy and batch, it is the only one that deviates more than 10 % from their median; the median deviation is 0.9 %.

**Call path at token 21 (t = 69.4 s)** [M] + [R]:

1. `record_window` calls `_update_elimination(P100)` (`windows.py:660`). The host latency upper bound is 347.2 ms/token (W1 alone), so the limit is 434 ms/token, while P100's upper bound is 582.7. The replay shows it records **`LATENCY_BOUND_EXCEEDED`** (`promotion.py:80-85`).
2. `_next_after_window` reaches stage `candidate`. Both sides have rows, so the LEARNING check at `sequencing.py:681-693` runs: `_learning_probe_improves(P100)` compares 47.23 against 40.92 J/token (`promotion.py:255-260`) and returns False. It then overwrites the reason with **`LEARNING_NO_PAIRED_IMPROVEMENT`**, the reason the event log shows.
3. So both the energy test and the latency test were decided by W1. The comparison is of plain means with no uncertainty band; it uses one host window and one phone window. The elimination lasts until a context change (`windows.py:544`), which never happened.

**P50 during the phone's HTP2 load** [M] timing, [I] cause.

- The phone phase events, aligned on HTP2's VERIFIED event, put HTP2's WEIGHT_READ at 64.0-72.7 s and its WEIGHT_UPLOAD at 72.8-87.6 s. These are upper-bound times: the scheduler records SESSION_VERIFIED with some lag.
- The P50 warm-up window W6 stalled 14.8 s (token 24 at 71.43 s, token 25 at 86.25 s), i.e. during the upload. P100's windows during WEIGHT_READ were not slowed (529.7 ms/token, vs about 545 for P100 in 002 and 004).
- W6 overran the P50 probe budget (deadline 76.51 s), so the probe went `_probe_admitted` false, then `_incomplete_probe`, then host (`sequencing.py:624-632`, `26-36`), and was retried after the load (`sequencing.py:606-614`).
- The retried P50 window W10 (62.65 J/token) was then compared against the host mean of W1 and W8, (40.92 + 78.46) / 2 = 59.69 J/token. That still contains the artifact, so P50 was eliminated as well. `_select_probe_winner` found no candidate left, which produced `MEASURED_REJECTION`, and the host ran to the end (94 tokens).

**Answers to the sub-questions.**

- The P50 probe did not eliminate P100; P100 was eliminated at token 21, before P50 started.
- The controller neither discards nor penalizes windows measured during phone residency transitions.
  - `record_window` treats a window as unstable only for a drain-shortened window, a batch or membership change, a change in desktop external activity, a recovered context, or (in coherence mode only) a non-comparable server window (`windows.py:477-510`).
  - `runtime_external_desktop_activity` deliberately ignores phone-only work (`_unified/runtime_requests.py:131-158`).
  - The stalled window was harmless only because it happened to be the policy's warm-up window.

Counterfactuals [R] (`data/replay_000_*`):

- Without the permanent elimination, P100 qualifies as soon as the first real host window W8 arrives, at token 35.
- With W1 excluded, P100 has no host reference at token 21, so the controller's `verification_baseline` path (`sequencing.py:676-680`) takes a fresh host window first, and P100 qualifies against it.

## 3. Why 002 and 004 promoted the phone (question 3)

- **002.** There was no history: no matching group at its start, and `hist = 0` in every event. Replay (`data/replay_seeding_dev.txt`): the 30 input-store groups (v8, 2026-09-02) never match this rig, because `_group_matches_session` (`history.py:248-276`) requires the same planning profile or component binding, desktop placement and helper geometry. Its first pair was P100 51.28 J/token vs host 78.42 J/token.
  - That host window is itself high: it contains a 1.6 s token-delivery gap, and runs about 10 % slow in latency.
  - Phone upper 56.4 <= required 69.9 J/token, so P100 qualifies at the first eligible phone window and stays qualified for all 8 of its windows [R] (`data/replay_002_as_run.txt`).
- **004.** Operational verification was seeded by `_operational_verification_policy` (`candidates.py:77-112`) from **002's completed group, in the same run**.
  - Replay: with 002's group dropped, the seed disappears.
  - The complete pair (P100 46.87 vs host 76.0 J/token) then gave VERIFIED `CURRENT_PAIR_IMPROVES` (`promotion.py:377-397`).
- **The model asymmetry.** With 1-3 windows per side, the bands require phone mean / host mean <= 0.9 x 0.99 / 1.1 = **0.81** (a saving of at least 19 %). Measured ratios [M], run-wide medians:

  | model | run | phone / host |
  | --- | --- | ---: |
  | Qwen | dev | 0.62 |
  | Qwen | run-5 | 0.63 |
  | Gemma | run-5 | 0.775 |
  | Gemma | dev | 0.81 |

  Qwen fails only when an artifact halves the host reference (000). Gemma is a coin flip on +/-2-3 % noise in the host reference (001 lost; run-5's Gemma requests 004, 011, 021 and 026 won).
- History does not help once it exists. 003 (the second Gemma request) matched 001's group, but that group holds one P100 window in the context bucket, and historical evidence counts **groups**, not windows, in the band (`bounds.py:188-196`, `candidates.py:103-106`). So 001's 129 host windows give the same +/-10 % as one window, and no seed results [R].

## 4. Run-5 (question 4)

Run-5 had 28 large requests with 7993 window tokens, 5960 of them on the phone (74.6 %). Ten requests ended on the host [M] (`data/run5_request_summary.txt`, `data/run5_compact_timeline.txt`).

| cause | requests | evidence |
| --- | --- | --- |
| rejected at batch 2, where the phone was measured worse (alternating passes next to a host co-tenant, `coherence.py:3-9`) | 000, 001, 002, 003, 024, 025 | run-wide medians [M]: Qwen P100@b2 120.3 J/1196 ms vs host@b2 80.7 J/634 ms; Gemma P100@b2 106.5 J/917 ms vs host@b2 61.5 J/486 ms. Eliminations: `LEARNING_NO_PAIRED_IMPROVEMENT` and `ENERGY_DOMINATED` at batch 2 |
| batch-2 latency elimination | 008 (and 025) | `LATENCY_BOUND_EXCEEDED` for all fractions at batch 2 |
| never measured (short request or budget; followers dropped straight after `SERVER_POLICY_COHERENCE` because they had no evidence of their own) | 009, 016, 023 | `INSUFFICIENT_OPPORTUNITY`, `PROBE_INCOMPLETE`, `INCUMBENT_NO_LONGER_BENEFICIAL` after the request-local follow rule (`coherence.py:267-295`) |

**The dev pattern** (phone better at the same batch, then rejected by a band flip or an artifact reference) ended no run-5 request on the host. The only batch-1 eliminations in run-5 were P25 `ENERGY_DOMINATED` by P100 (in 004 and 022), which is correct.

Related losses [M] tokens, [I] cost (cost = tokens x run-wide host-minus-P100 median at batch 1: Qwen 29.2 J/token, Gemma 13.6 J/token):

- **Batch-1 tails never re-probed after a batch-2 rejection:**

  | request | host tokens at b1 | cost |
  | --- | ---: | ---: |
  | 000 | 23 | 0.67 kJ |
  | 002 | 51 | 1.49 kJ |
  | 008 | 124 | 3.62 kJ |
  | 025 | 174 | 2.37 kJ |
  | **total** | **372** | **8.2 kJ** (1.9 % of the run's 437 kJ) |

  - Replay of `_measurement_pair_budget` (`scripts/replay_budget.py`): the exploration energy already spent at batch 2 was 1.28 kJ (008), 1.73 kJ (025) and 1.90 kJ (002). It is charged against the request's remaining 15 % allowance (`budgeting.py:260-267`).
  - Right after the batch change, while the host estimate is still the catalog prediction, the replayed allowance is at most 0.28 kJ. It is 0 from the first eligible batch-1 host window onward (008 from token 140, 025 from token 184), and 0 at every batch-1 boundary for 002. With a zero allowance no probe can be reserved (`_can_probe`, used at `sequencing.py:640-643,655-657`).
  - 000 had 23 tokens left, below `minimum_remaining_tokens = 24`.
  - 000's own batch-1 evidence from before the batch change (P100 48.2 vs host 77.3 J/token) had been discarded at the context reset (`windows.py:527-545`).
- **027, a stall artifact inside a batch-1 verification.**
  - The token stream stalled 23.5 s right after the phone-to-host switch (token 12 at 4661.7 s, tokens 13-20 at 4685.28 s).
  - The following eligible host windows read 24.7 J/token at 198 ms and 37.9 J/token at 304 ms, against a median of 78.9 J/token.
  - The verification ended INCOMPLETE twice (`RESERVATION_EXHAUSTED`, `ACK_LEFT_INSUFFICIENT_MEASUREMENT_TIME`), which saved it from a false rejection by luck. The prior monitor then took over at token 65.
  - That cost about 37 extra host tokens, about 1.1 kJ [I].

## 5. Minimal fixes (question 5)

Invariants to keep:

- `energy_measurement_eligible` stays false for diagnostic or ASSUMED windows (no certified energy claim).
- Operational selection stays behind `allow_assumed_phone_power_for_operational_selection`.
- Warm-up and transition windows stay ineligible.
- Promotion remains gated by `_qualifies` with its latency limit.
- Any new eligibility rule may only *remove* evidence.

The fixes below change when negative decisions become permanent and when positive decisions are re-checked.

| # | change | code | expected effect |
| --- | --- | --- | --- |
| F1a | Do not demote an incumbent while `_qualification_needs_more_evidence(...)` is True. Route to `_reserve_comparable_measurements` (existing) or keep it running; demote only when the mean evidence turns against it (`needs_more` false) or the latency limit is exceeded. | `sequencing.py:645-647` | 001: no drop at token 60 (replay: `more=True`, targets (4,4)). About 515 host tokens stay on P100, saving about 5.8 kJ and about 10 s [I] |
| F1b | On each eligible host window while exploiting the host after a qualification loss, re-run `_consider_incumbent` for the non-eliminated probe candidates, and switch if one qualifies. Host evidence already tightens the host band; it is just never used. | `windows.py:658-668` (host branch), `sequencing.py:648-654` | 001: P100 re-qualifies at token 75 [R], about 500 phone tokens, about 5.6 kJ [I]. Also covers any future band flip |
| F2 | Make the LEARNING pair gate non-permanent. Eliminate only when the difference is resolved by bounds (as `ENERGY_DOMINATED` does, `promotion.py:92-101`) or with >= 2 host windows; otherwise mark the candidate not-yet-improved and re-test it as host windows arrive (with F1b). Apply the same rule to `LATENCY_BOUND_EXCEEDED` computed from a single host window. | `sequencing.py:681-693`, `promotion.py:72-85` | 000: P100 qualifies at token 35 [R], about 94 more phone tokens, about 2.7 kJ and 6 s [I] |
| F3 | Add a stall/catch-up guard. Mark a window unstable when its start follows a token-observation gap or boundary lag above a threshold (for example 4x the policy's per-token estimate or 2 s), or when it drains a burst of >= 3 tokens observed within 50 ms. Better long-term: take token times from the server rather than the stream reader. | window builder in `_unified/adaptive_decode_control.py` / `stable_window` in `windows.py:505-510` | Removes the 000 W1 (-46 %) and run-5 027 W5/W6 (-69 %, -52 %) references. Evidence is only removed, never added |
| F4 | Treat a PREPARING or SESSION_LOADING session on the helper's own phone as a disturbance: mark overlapping windows unstable, and defer probe reservations until the load is READY (like `hold_unknown_context`, `sequencing.py:39-49`). The phase events already give the intervals. | observation assembly + `record_window` | 000: avoids the 16.6 s P50 window (about 0.8 kJ vs about 0.25 kJ normal) and the `PROBE_INCOMPLETE` retry. Prevents such a stall from becoming evidence when it does not land on a warm-up window |
| F5 | Charge exploration spend to the batch context in which it was spent (reset or scale it at context change), and keep per-batch evidence partitions inside a request instead of discarding them when the batch returns (`context_record_start`). | `budgeting.py:260-267`, `windows.py:527-545` | run-5: re-probing becomes possible on the 372 batch-1 tail tokens, up to about 8.2 kJ [I]. Coordinate with the concurrent per-batch-verdict change in `coherence.py` |
| F6 (needs sign-off) | Make the band monotone in n: continuous `sqrt(n)` (integer-safe, e.g. `isqrt(n * 10**6)`), and count historical windows, not groups. This loosens n = 2, 3 and 5..8 slightly; `bound_kind` `heuristic_integer_sqrt` is recorded in `qualification_measurement_plan` (`promotion.py:327`). | `bounds.py:105-108,158-163,191-196`, `candidates.py:103-106` | 001 would not have flipped at W15 [R]. Removes the "more evidence can only hurt" behaviour behind the Gemma coin flip |

The user's suggestion to "promote on consistent paired evidence when isolation is impossible" is already how the controller works: the operational path promotes on paired ASSUMED-power windows. What failed was how that paired evidence was judged:

- one-window references,
- a plateaued band,
- permanent eliminations,
- no re-evaluation from the host.

Making the eligibility rules "match multi-request reality" did not matter in dev, which had one slot. It matters in run-5, through F5 and the concurrent coherence work.

Expected dev-run effect of F1 + F2 (+F3) [I]: about 609 more phone tokens, taking the phone share from 29 % to about 89 %, and saving about 8.5 kJ, 11.7 % of the dev run's 72.9 kJ fleet energy. The absolute kJ use ASSUMED phone power and diagnostic attribution; they are not a certified energy claim.

## 6. Reproduce

The scripts only read data. They need Python 3; the replay scripts also need the run-time source.

```sh
cd research_dev/scheduler/campaigns/burstgpt/reports/20260924-phone-rejection-diagnosis/scripts
export PYTHONDONTWRITEBYTECODE=1
# copy from the desktop (read-only): ADAPTIVE_DECODE_OBSERVATIONS.json RESULT.json SCHEDULER_DECISION_LOG.json SOURCE_MANIFEST.json
python3 assemble_run_source.py /tmp/runsrc --manifest RUN/SOURCE_MANIFEST.json   # working tree + run-source/ overlay
python3 timeline.py RUN [REQUEST]            # windows + ASSISTANCE_DECISION events
python3 compact_timeline.py RUN              # policy@batch segments + reasons
python3 summarize_requests.py RUN            # per-request phone vs host, pattern flag
python3 host_token_breakdown.py RUN          # host tails and inferred cost
python3 window_noise.py RUN...               # per-window dispersion, first host window vs later median
python3 replay_qualification.py --source /tmp/runsrc/research_dev RUN REQUEST --eliminated-from-events [--sqrt-bands] [--exclude-windows 1]
python3 replay_seeding.py --source /tmp/runsrc/research_dev RUN REQUEST [--drop OTHER_REQUEST]
python3 replay_budget.py --source /tmp/runsrc/research_dev RUN REQUEST TOKEN
```

`RESULT.json` keeps only the last 4096 helper events. `timeline.load` merges each request's complete events from the COMPLETED records of `SCHEDULER_DECISION_LOG.json`; run-5 needs this.

`data/` holds the outputs quoted above: the `replay_*` tables, the per-request summaries (`.txt`, `.json` and `.csv`), `window_noise`, the host-token breakdowns and the compact timelines of both runs.
