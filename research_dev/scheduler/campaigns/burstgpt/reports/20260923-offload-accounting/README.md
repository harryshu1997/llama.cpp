# Where the phone offload went: long-tail trace, phone-assisted run vs desktop baseline (2026-09-23)

Offline analysis of the completed pair (read-only artifacts, no hardware runs):

| run | dir (desktop 172.20.74.85) | status | duration | CPU kJ | GPU kJ | host kJ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline (desktop only) | `s42-trace-longtail-baseline-20260923-inputs/run-baseline-1/run` | PASS 31/31 | 4,901 s | 367.6 | 156.1 | 523.7 |
| treatment (energy-aware, OP15) | `s42-trace-longtail-treatment-20260923-inputs/run-treatment-5/run` | PASS 31/31 | 4,826 s | 275.8 | 155.2 | 431.0 (-17.7 %) |

Trace `burstgpt_longtail_v1`: 31 requests (16 Qwen3-14B "hot", 12 Gemma-4-12B "cold", 3 Llama-1B overlay),
arrivals over 1,579 s, 8,207 output tokens. Desktop: i9-12900K (31 GB) + RTX 4060 Ti (16 GB); one large
model resident at a time (Qwen: 16/40 layers on GPU, 17.2 GB mapped on host; Gemma: 22/48 on GPU). Phone
OP15 held 12 Qwen FFN layers (6.42 GB) + 8 Gemma FFN layers (2.83 GB) after t = 264 s.

Legend: **M** = measured from the run artifacts (RESULT.json, ADAPTIVE_DECODE_OBSERVATIONS.json,
resource-samples.jsonl, server stderr, SCHEDULER_DECISION_LOG.json); **E** = estimated with the calibrated
pass model in `hypotheticals.py` (validated against both runs, see section 9). Scripts and data: section 10.

## 0. Summary

The phone was active for **72.6 % of decode tokens** (5,960 of 8,207; proof-derived 5,996) and cut host power
during decode from 124 W to 95 W, but the run still spent **31 % of its 431 kJ at the 27.6 W idle floor**,
**16 % of wall time in model loads and switch overhead** (766 s at floor power), and its remaining decode energy
was dominated by the choice *not* to batch: a batch-2 desktop-only pass already beats a batch-1 phone-assisted
pass in J/token (Qwen 38.5 vs 48.8; Gemma 30.2 vs 46.5), while today's phone policy collapses whenever two
requests share a server. Ranked by estimated gain (section 8): make the phone policy batch-coherent and
batch-aware (enabler), admit work-conservingly up to 4 same-model slots, re-provision the phone per resident
desktop model (18 Qwen / 26 Gemma layers), then trim exploration. Combined estimate: 2,170 s / 115 kJ vs the
validated replay 4,734 s / 410 kJ (-54 % time, -72 % host energy; E).

## 1. Decode tokens by policy state (M, treatment)

Attribution: each adaptive window's tokens are labelled by its policy (baseline / phone fraction), role and
`active_batch`; a batch-2 baseline window is "mixed" when the co-tenant slot ran a phone policy at the same
time. Proof cross-check: phone calls per released layer (mean over layers) = 5,996 assisted tokens vs 5,960
phone-policy window tokens (transition tokens between control and ack account for the difference).

| category | Qwen | Gemma | Llama | all | share |
| --- | ---: | ---: | ---: | ---: | ---: |
| phone policy, batch 1, exploitation (100 % columns) | 2,009 | 3,424 | 0 | 5,433 | 66.2 % |
| phone policy, batch 1, probe windows (100/75/50/25 %) | 90 | 165 | 0 | 255 | 3.1 % |
| phone policy, batch 2, mixed with a baseline co-tenant (2x latency) | 169 | 103 | 0 | 272 | 3.3 % |
| baseline, batch 2, both slots baseline (after elimination) | 655 | 446 | 0 | 1,101 | 13.4 % |
| baseline, batch 2, co-tenant on phone (mixed, 2x latency) | 176 | 117 | 0 | 293 | 3.6 % |
| baseline, batch 1, exploitation (fallback after elimination / probe budget spent) | 263 | 172 | 0 | 435 | 5.3 % |
| baseline, batch 1, probe / verification windows | 98 | 106 | 0 | 204 | 2.5 % |
| first token + unmeasured tail (`server_release_guard`, 2-3 tokens/request) | 48 | 37 | 0 | 85 | 1.0 % |
| no phone candidates (overlay model) | 0 | 0 | 129 | 129 | 1.6 % |
| **total** | 3,508 | 4,570 | 129 | 8,207 | |
| **phone policy share** | **64.7 %** | **80.8 %** | 0 % | **72.6 %** | |

Where the 2,247 non-assisted tokens come from:

1. **Batch-2 pairs: 1,394 tokens (62 %)**. Whenever two requests shared a server (000/001, 002/003, 008/009,
   015/016, 022/023, 024/025) every phone probe measured 1.10-1.20 s/token against a 0.63 s baseline (Qwen)
   or 0.87-0.92 vs 0.48 s (Gemma), i.e. 1.8-1.9x, above the 1.25x latency bound -> `LATENCY_BOUND_EXCEEDED`
   for all four fractions, both sessions fall back to baseline for the rest of the pair, and the 80-token
   per-request probe budget is spent, so the survivor stays baseline after the partner leaves (002: 51
   tokens, 008: 124, 025: 158). Mechanism: `server_slot::can_batch_with` (tools/server/server-context.cpp:407)
   refuses to batch slots with different `ffn_split_policy`, so a mixed pair runs two alternating forward
   passes per token step; the probes of the two sessions are staggered, so the pair is mixed almost the
   whole time it probes. The server counted **547 mixed forward passes** ("release skipped: mixed slot
   policies"), the windows show **565 slot-tokens** decoded under mixed policies. The same pairs in the
   baseline arm decoded at 0.64-0.67 s/token per slot; in the treatment they took 0.70-0.85 s/token
   (**25-37 % slower than the baseline arm**, anomalies in section 7).
2. **Baseline fallback at batch 1: 435 tokens (19 %)**: post-pair tails above, plus the 7-token
   verification stretches (tokens 15-26/30) in 9 requests, plus 4 tokens in 009.
3. **Exploration: 204 baseline + 255 phone probe tokens (5.6 %)**: every request re-qualifies from scratch
   (first baseline window pair, then 4 fractions x (warm-up + measured window)); 12 requests probed all of
   100/75/50/25 %; the 75/50/25 % fractions were never better than 100 % (Gemma 51.6/53.7/56.8 vs 46.5
   J/token; Qwen all >= 59 J/pass-token at batch 2).
4. **Overlay 129 + first/tail 85 tokens (2.6 %)**: structural.
5. `INSUFFICIENT_OPPORTUNITY` fired only twice (helper events); it is not where tokens were lost.

Measured per-token cost by state (batch-1 windows are attributable; batch-2 figures are per pass token):

| model | state | ms/token | host J/token | CPU W | GPU W |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | batch 1 baseline | 624 | 76.4 | 90 | 32 |
| Qwen | batch 1 phone 100 % (12 layers) | 572 | 48.8 (-36 %) | 52 | 33 |
| Qwen | batch 2 both baseline (per pass token) | 320 | 38.5 | 94 | 32 |
| Qwen | batch 2 mixed, baseline slot / phone slot (per pass token) | 565 / 595 | 65.1 / 59.2 | 83 / 68 | 32 |
| Gemma | batch 1 baseline | 474 | 60.0 | 92 | 34 |
| Gemma | batch 1 phone 100 % (8 layers) | 456 | 46.5 (-22 %) | 68 | 34 |
| Gemma | batch 2 both baseline (per pass token) | 243 | 30.2 | 92 | 33 |
| Gemma | batch 2 mixed, baseline slot / phone slot (per pass token) | 436 / 459 | 52.6 / 52.6 | 87 / 81 | 34 |

## 2. Phone utilization timeline (M)

| | Qwen layers (6.42 GB) | Gemma layers (2.83 GB) | phone |
| --- | ---: | ---: | ---: |
| phone RPC busy (calls x mean rpc; windows agree within 1 %) | 281 s (5.8 % of run) | 210 s (4.4 %) | 491 s (10.2 %); RESULT `phone_active_time` incl. USB 546 s (11.3 %) |
| desktop serving this model (its server alive) | 2,430 s | 2,327 s | |
| **layers idle because the other model was resident** | **2,327 s (48.2 %)** | **2,430 s (50.4 %)** | |
| decode time of this model | 2,287 s | 2,194 s | |
| ... with a phone policy active | 1,400 s (61 %) | 1,733 s (79 %) | |
| ... without (pairs, probes, fallback) | 887 s | 462 s | |
| time-weighted share of resident bytes that were useful | | | 49.7 % |

Per call: Qwen 17,408-column call rpc 10.27 ms (compute 9.72 ms), Gemma 15,360-column call 7.06 ms (6.59 ms);
host waits for the call (`host_mean_ms` ~0.01, `wait_mean_ms` = rpc). 26,010 full Qwen calls + 28,456 full
Gemma calls; the fractional-column probe shapes account for only 3,070 calls.

## 3. Phone RAM and residency (M + E)

- **One residency change in the whole run (M)**: 17 Qwen layers (HTP0 0-5, HTP1 6-11, HTP2 12-16) until
  t = 251 s, then HTP2 drained and reloaded with Gemma layers 16-23 (12.1 s), giving 12 Qwen + 8 Gemma
  = 9.25 GB for the remaining 4,570 s. The change was triggered by the *arrival* of the first Gemma
  request (004, t = 248 s) although Gemma did not run until t = 472 s; 002/003 ran with 12 instead of
  17 Qwen layers meanwhile (moot, they were a mixed pair anyway). 126 residency events afterwards: 72
  `LEARNING_RETAINED`, 14 `EXPLORATION`, 5 `EXPANSION`, always the same 26 candidates, objective
  `queue_rough_compute_ops` summed over *all* queued work of both models, switching margin 1e16 uJ.
- **Session load cost (M)**: 4 HTP session loads, 12-20 s each for ~3.2 GB (weight read from phone flash
  10.3 s + HTP upload 1.1-2.4 s; ~270 MB/s), shards pre-staged at `/data/local/tmp/s42-ffn-shards-20260904-v1/`.
  No ADB/USB transfer is needed at swap time. The scheduler's transition cost prior is not learned from
  these (economics.py uses a fixed prior; unqualified transitions default to 1e18 uJ, section 8 #4).
- **Hypothetical demand-driven re-provisioning (E)**: because the desktop runs one model at a time and each
  switch costs 47 s of load (+16.7 s extra), the phone can be re-provisioned *inside* the switch window:
  18 Qwen layers (3 sessions x 6 = 9.63 GB, ~36 s) while Qwen is resident, 26 Gemma layers (9.2 GB, ~34 s)
  while Gemma is resident. Per-layer gains are linear in the measured windows (Qwen 4.4 ms and 2.3 J per
  token per layer; Gemma 2.2 ms and 1.69 J), which the calibrated pass model reproduces (section 9).
  With today's admission and phone policy this alone is worth **-37 % host energy (410 -> 257 kJ) and
  -4.7 % time**; with the coherent policy (#1) **-47 % (219 kJ)**. Caveats: assumes all 26 Gemma CPU FFN
  layers may be offloaded (the stored shard mask 16777215 covers layers 0-23 only, so shards for 24-25
  would have to be added), that 3 HTP sessions of ~3.2 GB each are the binding limit (18 Qwen layers is
  exactly 3 x 6), and that per-layer savings stay linear at 26 layers (Gemma host energy would drop to
  ~16 J/token, i.e. the host does only attention + waiting).

## 4. Queue and ordering (M + E)

| | baseline arm | treatment |
| --- | ---: | ---: |
| sum of queue waits, 28 large requests | 44,209 s | 43,406 s |
| mean / max wait | 1,579 / 3,195 s | 1,550 / 3,152 s |
| Qwen mean wait / Gemma mean wait | 1,097 / 2,222 s | 1,081 / 2,175 s |
| requests that ever decoded with a partner | 12 / 28 | 12 / 28 |
| requests that ran alone while same-model work was queued | 12 | 12 |
| model switches in service order + Llama-forced reload | 10 + 1 (12 loads) | 10 + 1 (12 loads) |
| load time (sum) / extra switch overhead beyond load+prompt | 653 s / 9.8 s mean | 565 s / 16.7 s mean (3.7-35 s) |
| switch dead time total | 770 s (15.7 %) | 766 s (15.9 %) |

Why the dispatcher serialized (M, from `SCHEDULER_DECISION_LOG.json` leases): a route that carries a model
transition takes an exclusive load lease on all 8 `cuda0` and all 4 `desktop-cpu` lanes for the predicted
load (~55 s) and a residency barrier orders every later request behind it by arrival; followers are placed
on the calendar after the leader's *predicted* end, and the predicted decode duration is ~1.8x too long
(010: planned 1,467 -> 2,364 s, actual 489 s), so followers wake only on `predecessor_completion` /
`capacity_released_early` replans. Result: after every switch the first request runs alone (005, 007, 011,
012, 014, 017, 018, 019, 021), and pairs form only when several requests are replanned at the same
completion instant. Service order is strict FIFO by reserved start; the queue held 4-11 Qwen and up to 11
Gemma requests for most of the run.

Estimates (E; `hypotheticals.py`, vs the validated replay 4,734 s / 410 kJ unless noted):

| change | duration | host kJ | phone share | comment |
| --- | ---: | ---: | ---: | --- |
| work-conserving back-fill, batch <= 2, phone policy as today | 3,244 s (-31 %) | 344 (-16 %) | 21 % | pairs everywhere -> the mixed penalty eats the phone |
| back-fill <= 2 + coherent phone policy | 2,737 s (-42 %) | 237 (-42 %) | 90 % | |
| back-fill <= 4, no phone | 2,230 s (-53 %) | 238 (-42 %) | 0 % | batching alone matches phone+batch-2 |
| back-fill <= 4 + coherent phone policy | 2,187 s (-54 %) | 181 (-56 %) | 87 % | Qwen KV 4,096 limits batch to 2-3 |
| model-affinity order, wave admission, starvation bound 1,800 s (vs wave reference 4,328 s / 393 kJ) | 4,046 s (-6.5 %) | 390 (-0.8 %) | 70 % | 8 -> 2 switches; a 900 s bound flip-flops (12 switches, slower) |

Switch count is a *time* lever, not an energy lever: loads run at the 27 W floor, so 6 avoided switches save
~380 s but only ~10 kJ (2.5 %). Co-batching is both: a pass costs the same energy for 1, 2 or 4 rows
(measured batch-2 pass 77.1 J vs batch-1 76.4 J for Qwen), so J/token halves per doubling.

## 5. Release / restore cycles (M)

| server | releases | mean / max elapsed | bytes | restores | mean / max elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen (hot) | 20 | 44 / 184 ms | 6.3 GB | 13 | 128 / 372 ms |
| Gemma (cold) | 23 | 31 / 62 ms | 2.2 GB | 7 | 36 / 40 ms |
| total | 43 | | | 20 | **3.51 s over the run (0.07 %)** |

Releases are `madvise(MADV_DONTNEED)` of the phone-owned FFN slices; restores `MADV_POPULATE_READ` from the
page cache (drop_cache=0), which is why they are cheap. Restores are needed whenever a prompt is processed
(prefill needs the host weights) or policies mix; 23 of the 43 releases are re-releases after a fraction
change during probing (each probe fraction = restore + release, the 4,352-column release took 184 ms).
Not a cost worth optimizing; cutting the probes removes about half the cycles as a side effect.

## 6. Energy split and the idle floor (M)

Resource samples (4 Hz RAPL package + NVML board) labelled by request/window state:

| state (treatment) | seconds | host kJ | host W | CPU W | GPU W |
| --- | ---: | ---: | ---: | ---: | ---: |
| decode, phone policy (Qwen b1 85.5 W, Gemma b1 102.4 W) | 2,838 | 270.3 | 95.2 | 52-68 | 33-34 |
| decode, baseline (Qwen b1 119 W, Gemma b1 126.5 W, batch 2 both baseline 126 W) | 684 | 84.5 | 123.6 | 85-94 | 32-35 |
| decode, mixed policies | 293 | 34.2 | 116.7 | 83-87 | 32-34 |
| decode, other (transition tokens) | 48 | 5.2 | 108 | | |
| prefill | 380 | 20.9 | 55.0 | 20 | 33-37 |
| model load (12 loads) | 567 | 15.3 | 27.0 | 6-7 | 19-21 |
| idle (no request) | 33 | 0.9 | 27.5 | 8 | 19 |
| **total sampled** | 4,842 | 431.4 | 89.1 | | |

Baseline arm: decode 3,926 s at 124.1 W (487 kJ), prefill 302 s at 62 W (18.7), load 653 s at 26 W (17.0),
idle 42 s at 26 W. Idle floor **27.6 W** (treatment; 25.9 W baseline; corroborated by the load state at
26-27 W, which is disk-bound with the CPU at 6-8 W and the GPU at 19-21 W).

- Floor energy over the run: 27.6 W x 4,826 s = **133 kJ = 31 % of 431 kJ**. Only a shorter run removes it
  (every second saved is 27.6 J); the batching variants above shorten the run by 2,000-2,600 s.
- Decode above the floor: 394 - 107 = 287 kJ (67 %). This is what offload and batching act on.
- Prefill above the floor: 10 kJ; loads: 0 (they run at the floor).
- The phone-assisted saving in the run came entirely from CPU package power during decode (-28 W over
  2,838 s = -81 kJ), consistent with the 92 kJ CPU delta between arms.

## 7. Anomalies (M)

1. **Mixed-policy pairs decode slower than the baseline arm**: 000/001/002/003 at 0.80-0.85 s/token vs the
   baseline arm's 0.64-0.68 for the same pairs; 009 0.78 vs 0.65; 016 0.79 vs 0.48; 024 0.63 vs 0.48.
2. **Model loads 15.5-106 s, disk-bound at the floor**: treatment 39, 31, 44, 15.5, 34, 78, 69, 42, 36, 45, 35,
   96 s; three loads > 60 s. 15.5 s is the page-cache-warm case (Qwen unloaded 20 s earlier for the Llama
   overlay). Time to first token reached 103-116 s for 012, 014 and 027.
3. **Switch overhead beyond load + prompt: 16.7 s mean, 35 s for 014** (warm-up request, server booking,
   helper attach); ~200 s in total.
4. **The Llama overlay evicts the resident large model** (`server_forgotten` 18571 before `server_booked`
   18485 at t = 1,058-1,061 s) although the 0.69 GB Llama server fits the 30 GB host budget; cost one Qwen
   reload (15.5 s) plus ~6 s of teardown/boot per overlay request.
5. **Four unexplained decode stalls at batch 1**: 006 tokens 34-38 (2.7 s/token, 8.6 s excess), 007 134-138
   (3.4 s/token, 11.3 s), 021 206-210 (4.6 s/token, 16.7 s), 027 16-20 (5.4 s/token, 19.1 s); each followed
   by a burst of fast windows, so part is buffered observation, but 021 loses ~10 s net. No residency or
   helper event coincides. Separately, three `DECODE_CONTEXT_UNAVAILABLE: TimeoutError` observer timeouts
   (002 at 280 s, 006 at 1,037 s, 009 at 1,238 s) with recovery 4-12 s later.
6. **Planner service-time estimates ~1.8x pessimistic** (010 planned 896 s decode, actual 489 s) -> 13
   `capacity_released_early` replans and calendar serialization (section 4).
7. **023 (Gemma, 120 tokens) was never assisted** while paired with 022 (helper window lease went to 022,
   `stale_slot_stats_discarded` tail); 009 got 5 assisted tokens, 016 got 5.
8. **Probe fractions 75/50/25 % never won** yet were probed in 12 requests (section 1).
9. Both arms diverged in 7 outputs mid-stream (known batch-shape float near-ties), not investigated here.

## 8. Ranked scheduler changes (E = estimated from the calibrated replay; code locations from a read of the current tree)

| # | change | est. gain vs validated replay (4,734 s / 410 kJ) | where |
| --- | --- | --- | --- |
| 1 | **One phone policy per server + batch-aware phone calls.** All slots of a server run the same (mask, columns); a joining request adopts the running policy instead of resetting the context and re-probing; phone calls carry all rows of the pass (coalesced) instead of one call per row. | alone: -3 % time, -7 % kJ, phone share 79 -> 94 %. It is the *enabler*: with back-fill the phone share is 87-96 % instead of 13-21 %. | `_internal/adaptive_decode_ops/coherence.py` (`server_policy_key`:27, `server_directive`:122, `_follow_coherent_policy`:267) behind `server_policy_coherence` (`_internal/adaptive_decode_contracts.py:78`, default False -> on); `_internal/adaptive_decode_ops/windows.py:483-594` (batch-membership context reset forces baseline); `_unified/helper_preparation_ops/common.py:shared_helper_attachments`:17 + `attachment.py:181-198` (share the HTP window lease instead of `arbitration.py:_select_helper_window_owner`:37 deferring the co-tenant: 34 `WINDOW_LEASE_DEFERRED`); `_internal/runtime_decode_cohort.py:candidate_key`:368 (returns None for desktop plans, so one controller never drives both slots; `adapters/decode_cohort.py:DecodeCohortPolicyCoordinator.register`:412); transport `usb_batch_plan` "split-row" -> "coalesced-batch" in the dormant runtime parameters (`_unified/automated_selection_ops/dormant.py:_complete_dormant_phone_ffn_parameters`:89; `ffn_max_tokens` 4/2 already allow it). Server side needs nothing new: `apply_dormant_host_share` (tools/server/server-context.cpp:3932) already releases when all slots agree. |
| 2 | **Work-conserving admission.** Let same-model followers start when the load ends and a slot + KV room are free, instead of behind the leader's predicted end; fix the 1.8x pessimistic decode estimate. | batch <= 2: -31 % time, -16 % kJ (phone collapses to 21 % without #1); with #1: **-42 % time, -42 % kJ** | `_internal/runtime_queue.py:_bind_causal_predecessors`:442-484 (residency barrier -> arrival-order serialization), `wait_ready`:1318; `_unified/automated_selection_ops/resources.py:289` (barrier condition); `_internal/resource_timeline.py:preview_leases`:433-517 (transition lease = all `cuda0`/`desktop-cpu` lanes for the load, followers placed after predicted end); decode-time estimate from history in `_internal/adaptive_decode_ops/estimates.py:historical_route_estimate`:17 / `_internal/runtime_cost.py`. |
| 3 | **Batch up to 4 same-model requests** (Qwen `parallel` is already 4, Gemma 2 -> 4); raise Qwen KV budget 4,096 -> 8,192 so 4 x ~1.5k-token requests fit. | with #1: **-54 % time, -56 % kJ** (2,187 s / 181 kJ); no phone at all: -53 % / -42 % | `_internal/runtime_decode_cohort.py:capacity`:447 = min(4, parallel, usb_concurrent_streams, usb_queue_depth); lanes in `campaigns/burstgpt/configs/v7/rig.json:105-150` (`desktop-cpu` 4) and `adapters/catalog_materialization.py:590-660`; Qwen `context_size`/`parallel` in the model adapter parameters (`configs/v4/models.json`). Bandwidth-bound decode: a pass costs the same energy for 1-4 rows (M). |
| 4 | **Re-provision the phone for the resident desktop model**: 18 Qwen layers while Qwen is resident, 26 Gemma while Gemma is; swap during the desktop model load (36 s of a 47 s load). | **-37 % kJ, -4.7 % time** with today's admission; **-47 % kJ** with #1; stacks with #2/#3 (F+A'+D: 2,171 s / 126 kJ) | `_unified/phone_residency_ops/portfolio.py:_update_phone_residency_portfolio`:307-525 and `demand.py:_phone_queue_demand`:489 (demand is summed over all queued models -> weight by the desktop's resident / next model, i.e. couple to `_internal/runtime_residency_cohorts.py:holds`:1317 and the placement epoch in `_unified/placement_epochs.py`); `_internal/phone_shards.py:generate_mixed_ffn_residency_layouts`:1884 (allow one artifact on all 3 sessions; today each layout keeps both models); `_internal/model_placement_ops/economics.py:evidence_for`:261-387 (transition cost is a fixed prior, unqualified = 1e18 uJ -> learn the measured 12 s / 3.2 GB from `phone_residency_phase_events`); `planning.py:propose_phone_layout`:176 (30 s minimum residency is fine). Gemma shards for layers 24-25 must be produced (`research_dev/shard_gguf.py`). |
| 5 | **Model-affinity ordering** with a starvation bound >= 1,800 s (a 900 s bound flip-flops and is slower). | -6.5 % time, -0.8 % kJ (8 -> 2 switches, wave admission); irrelevant once #2/#3 drain the queue | `_internal/runtime_queue.py:_dispatch_key`:293 (key (model != resident, start_us, arrival)); `_internal/runtime_residency_cohorts.py:33-56` (reuse horizon 30 s -> queue-aware hold); `adapters/coordinator.py:submit`:287. |
| 6 | **Trim exploration**: probe only the 100 % fraction when history shows it dominates; trust cross-request history instead of re-qualifying every request; no verification baseline stretch when the pair (baseline, 100 %) is already known for the context bucket. | -2.7 % time, -3.8 % kJ, +3 pp phone share; also removes ~half of the release/restore cycles | `_internal/adaptive_decode_ops/candidates.py:_sample_candidates`:150 (coarse fractions 1.0/0.75/0.5/0.25); `promotion.py:_qualifies`:104 (requires current-context rows; `history.py:_historical_policy_records`:306 has the cross-request data); `budgeting.py:_measurement_pair_budget`:183 (80-token cap that leaves post-pair survivors on baseline); `sequencing.py:676-680` (verification_baseline stage). |
| 7 | **Do not evict the large model for the overlay; cut switch extras.** | ~20 s + ~200 s of floor time (~6 kJ, 4.5 % time) | `adapters/heterogeneous_rig.py:575-601` (server booking), memory ledger budget 30 GB in `dormant_share_admission` (Llama 0.69 GB fits), runner transition warm-up in `campaigns/burstgpt/runner.py`. |
| 8 | Release/restore | nothing to gain (3.5 s) | - |

Combined (#1-#6, "ALL" in `hypotheticals.json`): **2,170 s / 115 kJ**, phone share 96 %, 3 switches, mean
queue wait 258 s (E). With Qwen KV 8,192: 2,191 s / 108 kJ.

## 9. Method and validation

- `parse_server_logs.py`: per server process (one per model load): load duration (first line -> "model
  loaded"), `print_timing` blocks (prompt / eval ms and tokens, mapped to requests by token counts),
  `dormant_host_share` release/restore lines, "release skipped: mixed slot policies" counters,
  `S41SERVERFFNSHAPE` and `FFNCONTROL` lines. Processes are aligned to the trace clock by file mtime and then
  re-anchored to the dispatch of their first request (mtime alignment was off by +10 s in the baseline run).
- `offload_accounting.py`: request timelines (arrival, reserved start, first token, end, partners), window
  labelling (section 1), phone utilization, queue statistics, release/restore, power states (each 0.25 s
  sample interval labelled idle / load / prefill / decode by model, batch and policy mix), anomalies.
- `hypotheticals.py`: a pass-level model `pass_ms = T_other(B) + n_cpu_ffn * t_cpu_ffn + phone_ms(B)`,
  `pass_J = T_other*(P_cpu_other+P_gpu) + cpu_ffn_ms*(P_cpu_ffn+P_gpu) + phone_ms*(P_cpu_wait+P_gpu)`,
  solved from the measured batch-1 baseline and phone windows of each model (Qwen: T_other 273 ms, 14.65 ms
  and 153 W per CPU FFN layer, 7.8 W CPU otherwise; Gemma: 233 ms, 9.27 ms, 180 W, 3.6 W), batch increment
  from the measured batch-2 windows, rpc per layer from the shape statistics, loads/prefill/idle at measured
  powers, measured mean load (47.1 s) and switch extra (16.7 s). The trace is replayed with the *observed*
  admission sets: **treatment 4,734 s / 410 kJ vs measured 4,826 s / 431 kJ (-1.9 % / -4.9 %); baseline
  4,792 s / 513.5 kJ vs 4,901 / 523.7 (-2.2 % / -1.9 %)**. The replay is slightly optimistic for the
  treatment because it models solo requests as fully assisted after 16 probe tokens (real share 73 % vs 79 %).
  Admission/ordering variants cannot reuse the observed admission, so they are also compared with the
  simulator's own "wave" rule (4,328 s / 393 kJ), which is itself 8 % faster than the real dispatcher.
- Estimate-only assumptions: coalesced multi-row phone calls cost +15 % per extra row (no multi-row call
  was ever issued in this run; `ffn_max_tokens` is 4/2); per-layer savings extrapolate linearly to 18/26
  layers; phone re-provisioning reads shards from phone flash at the measured 270 MB/s; batch-4 KV/attention
  cost grows by the measured batch-2 increment per extra row.

## 10. Files

- `parse_server_logs.py`, `offload_accounting.py`, `hypotheticals.py` (this directory).
- `data/accounting.json` (all measured tables), `data/hypotheticals.json` (calibration, validation, variants,
  pass table), `data/requests_{baseline,treatment}.csv`, `data/windows_treatment.csv` (1 row per adaptive
  window with category and co-tenant policy), `data/window_latency_energy_treatment.csv`,
  `data/request_windows_treatment.csv`, `data/power_states_{baseline,treatment}.csv`,
  `data/server_logs_{baseline,treatment}.json`.
- Reproduce (artifacts copied from the desktop, snapshots and the decision log excluded):
  `python3 parse_server_logs.py <run> --out server_logs_<arm>.json` for both arms, then
  `python3 offload_accounting.py --treatment <run> --baseline <run> --treatment-logs ... --baseline-logs ... --out-dir data`,
  then `python3 hypotheticals.py --accounting data/accounting.json --out data/hypotheticals.json`.
- Attribution rule reused from `../20260921-fast-path-trace-v2a/coverage_energy_data.py` (proof-derived
  assisted tokens; window `completed_phone_calls` is phone-global and not used for attribution).
