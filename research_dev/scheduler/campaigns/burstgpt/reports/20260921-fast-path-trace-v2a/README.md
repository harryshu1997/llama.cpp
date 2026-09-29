# 20260921 fast-path trace rerun (v2a = reduced 24-request trace, v2r = realistic BurstGPT window)

Question: does the current design (decode-only FFN relocation with the cohort co-dispatch fix, keep-cache
restore policy, energy-aware selection) reproduce or beat run7's saving on the reduced 24-request trace
(run7 reference: 125.1 kJ host / 127.0 kJ fleet / 1,577 s; matched saving 25.12 % on the v16 pair)?

Deploy: `/mnt/storage/s42-trace-v2-20260921-prep` (desktop `zhihao@172.20.74.85`, RTX 4060 Ti + OP15).
Inputs: `/home/zhihao/s42-trace-v2a-20260921-inputs` (treatment), `s42-trace-v2a-baseline-20260921-inputs`
(desktop-baseline selection), `s42-trace-v2r-20260921-inputs` (realistic trace, treatment).
Comparison: `compare_trace_energy.py <treatment RESULT.json> <baseline RESULT.json>`.

## Run ledger

| Run | Started (EDT) | Outcome |
| --- | --- | --- |
| treatment 1 | 2026-09-21 ~10:30 | FAIL: "did not confirm the dormant host share policy" (see root cause) |
| treatment 2 | 10:55 | FAIL: same + Gemma "policy requires release" |
| treatment 3 | 11:24 | FAIL: same; abort "runtime stale projection repair made no progress" |
| treatment 4 | 12:03 | PASS, 24/24, 1,468 s, 117.5 kJ host / 119.3 kJ fleet (server rebuilt with `S41_SERVER_FFN_SPLIT=ON`) |
| baseline 1 | 12:33 | PASS, 24/24, 1,381 s, 132.3 kJ host / 133.5 kJ fleet (desktop-baseline selection, same deploy, same trace) |

## Root cause of runs 1-3

The deploy's `build.sh` configured cmake without `-DS41_SERVER_FFN_SPLIT=ON`. The option defaults OFF, so
`tools/server/server.cpp` compiled the FFN split client out; the server ignored the entire `S41_SERVER_FFN_*`
environment and never printed `S41SERVERFFN ready`. Every phone-assisted Qwen launch therefore came up as a
plain desktop server, and the launcher's confirmation check refused it. Run7's library carried 12
`S41SERVERFFN` format strings, the new one 4 (server-context policy lines only).

Fixes kept from the three attempts (all still correct, none was the cause):
- launch contract defaults the cache-policy flags whenever no FFN environment is attached;
- the input patcher sets `ffn_host_share_drop_cache/populate` only on models with `ffn_host_share_release == 1`;
- launcher requires the `S41SERVERFFN dormant_policy` line only when the environment enables the host share.

New guard: `research_dev/scheduler/adapters/transport_profiles.py` refuses to materialize a transport
qualification identity when neither the host binary nor any host dependency contains `S41SERVERFFN ready`
(test `test_identity_builder_rejects_server_without_ffn_client`). Identity re-materialized after the rebuild;
the FFN-off identity is kept beside each inputs dir as `TRANSPORT_QUALIFICATION_IDENTITY.ffn_off.json`.

## Results

`compare_trace_energy.py` (host = RAPL cpu-package + NVML gpu-board, measured; phone assumed, own column):

| run | dur s | CPU kJ | GPU kJ | host kJ | host W | phone kJ* | fleet kJ | host vs 09-09 | fleet vs 09-09 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 2026-09-09 (old bundle, no release) | 1,689 | 98.8 | 52.0 | 150.8 | 89.3 | 1.84 | 152.6 | 0 | 0 |
| run7 (2026-09-17 bundle) | 1,577 | 76.9 | 48.2 | 125.1 | 79.3 | 1.90 | 127.0 | -17.0 % | -16.8 % |
| treatment 4 (this deploy) | 1,468 | 74.7 | 42.9 | 117.5 | 80.0 | 1.80 | 119.3 | -22.0 % | -21.8 % |

Treatment 4 beats run7 by 6.1 % host energy and 6.9 % duration at equal average host power. Against the
2026-09-09 baseline the saving is 22 %, below the 25 % target; that baseline is an old bundle, so the
matched pair on this deploy (baseline 1, desktop-baseline selection) is the number that counts. The
25.12 % figure quoted for v16 was a matched pair of that kind (v15 vs v15c).

## Matched pair on this deploy (the number that counts)

| arm | dur s | CPU kJ | GPU kJ | host kJ | host W | fleet kJ | host saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 1 (desktop-baseline) | 1,381 | 89.5 | 42.8 | 132.3 | 95.8 | 133.5 | 0 |
| treatment 4 (energy-aware, phone) | 1,468 | 74.7 | 42.9 | 117.5 | 80.0 | 119.3 | **-11.1 %** (fleet -10.6 %) |

- Correctness: all 24 generated token sequences are identical between the arms (parsed from the SSE streams).
- The whole saving is CPU package energy (89.5 -> 74.7 kJ); GPU energy is equal. Time-weighted CPU power
  64.3 -> 50.6 W, floors equal (p5 3.5 W). Treatment is 87 s (6 %) longer; Qwen request latencies rose
  (sum 602 -> 664 s), Gemma fell slightly (576 -> 556 s).
- Phone assistance was thin: 1,716 phone FFN layer calls, every one with `tokens=1` (no decode cohort ever
  formed, `decode_cohort_state` empty), 12- or 6-layer masks -> about 150 of the 830 Qwen output tokens
  (~18 %) were decoded with FFN on the phone. The v16 pair that reached 25.12 % had 76 % of Qwen tokens
  assisted, and only 5 of the 15 Qwen requests saw any phone-executed fraction (88118, 88123, 88127 with a
  25->100 % ramp, 88133, 88134). Servers flip-flopped: 45 releases and 45 restores; on the busiest server
  (36 cycles) every release served exactly 12 layer calls = one decode token before the restore, while the
  quiet server kept two releases open for 55 and 43 tokens.
- The 2026-09-09 baseline (150.8 kJ, 1,689 s) is no longer a fair control: the current bundle's baseline
  alone is 12 % cheaper and 18 % faster (ubatch 512, page-granular KV, fused work), which is why the
  treatment's 22 % against the old control shrinks to 11 % against the matched one.

Verdict: the target (>= 25 % matched saving) is NOT met on this trace. The mechanism works and is
bit-exact; the lever is release coverage (tokens decoded while the share is released), not per-call speed.

## Why coverage is low: the dormant share is server-wide, the trace is concurrent

On the busiest Qwen server (36 release/restore cycles) the log shows 36 tokens streamed while the share was
released, all from slot 0, against 406 tokens streamed while local from slots 0-3. The runtime releases the
host FFN only when every active slot of the server is in a release-eligible decode; any other slot doing a
prefill or a non-eligible decode restores the weights (about 110 ms per restore, 45 ms per release). With four
slots and BurstGPT-style interleaved arrivals the server is almost never in that state, so the 45 cycles
mostly bought one token each. The quiet server (three long releases, 98 tokens) shows the mechanism working
whenever the server is single-tenant.

Levers, in order of expected coverage gain:
1. Per-slot (per-sequence) release instead of server-wide: keep the released layers' FFN on the phone for
   eligible sequences while the host serves the others (host copy stays mapped; releases only the
   decode-side compute, which is the CPU-energy lever anyway).
2. Admission hysteresis (plan M4): do not admit a new prefill to a server whose share is released while
   an eligible decode is running; route it to the other parent or delay it within its SLO.
3. Decode cohorts (plan M2) so the phone call carries N<=4 tokens; today every call is `tokens=1`.

## v2r realistic trace (burstgpt_realistic30_v2: 20 requests, 10 Qwen / 8 Gemma / 2 Llama, 1,599 s of arrivals)

| run | outcome |
| --- | --- |
| treatment 1 | FAIL at replay start: trace manifest lacked `model_inventory` (builder gap, fixed) |
| treatment 2 | FAIL at request 012: stale Gemma cache flags in models.json -> Gemma cold loads refused -> route quarantined (validator added) |
| treatment 3 | PASS 20/20, 3,555 s, host 281.2 kJ (CPU 178.9 / GPU 102.3), 79.1 W, phone* 4.4 kJ |
| baseline 1 | PASS 20/20, 3,898 s, host 340.0 kJ (CPU 234.1 / GPU 105.9), 87.2 W |

Coverage in treatment 3 is far better than on the 24-request trace because outputs are long (median ~350
tokens): all 10 Qwen requests were phone-assisted, 20,154 phone layer calls at 12 layers = ~1,680 of 2,734 Qwen
output tokens (~61 %), 90 releases (87 with the 12-layer mask, 3 with 6 layers). Still one token per call and
still 12 of 24 CPU layers, so the layout and cohort levers apply unchanged.

### v2r matched pair

| arm | dur s | CPU kJ | GPU kJ | host kJ | host W | fleet kJ | host saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 1 (desktop-baseline) | 3,898 | 234.1 | 105.9 | 340.0 | 87.2 | 343.4 | 0 |
| treatment 3 (energy-aware, phone) | 3,555 | 178.9 | 102.3 | 281.2 | 79.1 | 285.6 | **-17.3 %** (fleet -16.8 %) |

Per-model summed request latency is equal between arms (Qwen 1,719 vs 1,718 s, Gemma 1,535 vs 1,544 s); the
treatment finished the whole trace 9 % sooner. Saving again comes from CPU energy (234 -> 179 kJ, -24 %) with
GPU within 3 %. Coverage 61 % of Qwen tokens (vs 18 % on the 24-request trace) turns 11 % into 17 %; the
remaining gap to 25 % is the 12-of-24 layer mask and single-token phone calls.

Output equality on v2r: 14 of 20 token sequences identical, 6 differ (first differing token at positions 1, 19,
35, 54, 110, 323 of 349-512-token outputs). One of the six (request 005) had no phone execution at all, so the
divergence is host-side: with different slot co-tenancy between arms the batched CPU/CUDA accumulation order
changes and greedy decode flips at near-tie tokens. On the 24-request trace (outputs <= 341 tokens) all 24 were
identical. Claim to make: phone FFN is bit-exact against the host for the same batch (M2 near-tie rule, v2a
24/24); long-output traces are not bit-reproducible across arms for reasons unrelated to the phone.

## Layout variants on the 24-request trace (lever 1)

| arm | phone layout | dur s | CPU kJ | GPU kJ | host kJ | host W | host saving vs baseline 1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline 1 | none | 1,381 | 89.5 | 42.8 | 132.3 | 95.8 | 0 |
| treatment 4 | Qwen HTP0+HTP2 (0-5, 12-17), Gemma HTP1 | 1,468 | 74.7 | 42.9 | 117.5 | 80.0 | -11.1 % |
| variant A (`s42-trace-v2a-qwenlayout-20260921-inputs`, Gemma `phone_resident_limit_bytes=1`) | Qwen HTP0+HTP1 (0-11), Gemma HTP2 | 1,457 | 68.8 | 42.5 | 111.2 | 76.4 | **-15.9 %** |
| variant B (`s42-trace-v2a-fixedlayout-20260921-inputs`, campaign `fixed_phone_residency` = Qwen on all three sessions) | none (see note) | 24/24 ran desktop-only, then FAIL | | | | | not comparable |
| variant C (`s42-trace-v2a-noshardgemma-20260921-inputs`, Gemma without phone shards, plans kept) | Qwen 0-16 briefly, then 0-5 + 12-16 (Gemma took HTP1 anyway, 9 layers) | 1,437 | 69.6 | 43.1 | 112.7 | 78.4 | -14.8 % |

Variant A did not do what it was meant to (the one-byte cap is only forwarded as
`maximum_helper_resident_weight_bytes` and nothing consumes it; Gemma still took a session), but it changed the
session order so Qwen's two sessions became contiguous layers 0-11 (mask 4095) and phone coverage rose from
~150 to ~413 Qwen tokens (4,962 layer calls, 47 releases) with the same 5 of 15 requests assisted. Same
12-of-24 layers, better coverage -> 15.9 %.

Variant B note: `fixed_phone_residency` is an evaluation mode, not a production layout pin. The runner preloads
the assignment in a side thread, arrived work is scheduled independently of it, and `end_trace` does not require
phone execution. The preload never engaged the phone (one preparation snapshot, no sessions) and the run ended
with "fixed residency assignment is infeasible; capacity or shard coverage differs". Do not use it for energy
arms. Variant C uses the production planner: a model without a shard index has no phone demand, so the planner
cannot place Gemma and the three OP15 sessions can only hold Qwen.

Variant C result: -14.8 %, statistically the same as variant A (-15.9 %); 4,232 calls (~385 tokens) over 51
releases, 5 of 15 Qwen requests assisted, as in every arm. Removing Gemma's shard index did not remove Gemma
from the phone: shards are built from the plan's execution contract (the full artifacts are on the phone) and
the planner's "learning" demand keeps preparing residency for Gemma phone routes that are never selected
(`_learning_phone_demand`). Conclusion for the 24-request trace: at 5/15 requests assisted the arm saturates
near 15-16 % regardless of 11, 12 or 17 released layers; the next lever is which requests attach (10 short
Qwen requests never do), then release thrash, then cohorts. Variant D (Gemma release on) tests whether the
second model's sessions can be turned into energy instead of waste.

## Why 10 of 15 Qwen requests never got the phone (variant C adaptive observations)

Measured per-window energy in variant C (RAPL + NVML, phone assumed), Qwen decode windows:

| window kind | n | median J/token | mean J/token |
| --- | ---: | ---: | ---: |
| baseline (host FFN) | 147 | 75.3 | 81.4 |
| phone-assisted | 244 | 40.6 | 50.6 |

The phone path is worth ~45 % per token when it is on. The adaptive decode controller still ended every
unassisted request in `EXPLOITING` the baseline (history BASELINE -> EXPLOITING -> PREPARING -> EXPLOITING) with no
`CONTROL_ISSUED`: it exploits the context's cached winner or the ticket fallback (baseline) and only probes when
`probe_candidates` exist, the helper is attached, remaining tokens exceed `minimum_remaining_tokens` (24) and the
per-context probe budget (`maximum_probe_attempts_per_context` = 2) is not spent. Probes that did run were
poisoned by the server-wide release thrash: request 88123's phone windows measured 116 and 202 J/token while a
co-tenant kept restoring the share, so the pair was judged not improved and the baseline became the cached
winner for that context. Requests whose probe landed in a quiet moment (88133: 46-48 J/token) stayed on the phone.

So the levers compose: per-sequence release removes the thrash, which makes probe measurements truthful, which
lets the controller exploit the phone for the whole context; a larger probe budget and a lower
`minimum_remaining_tokens` then cover the short requests. Layer count is not the constraint on this trace.

## Variant D: both models release (dynamic layout)

`s42-trace-v2a-bothrelease-20260921-inputs`: Gemma `ffn_host_share_release=1` (drop_cache 0, populate 1), layout
left to the planner. Result: 1,421 s, CPU 68.8 / GPU 42.8 / host **111.5 kJ (-15.7 %)**, 23/24 outputs identical
to the baseline. The Gemma cold-desktop servers now release too (mask 16711680 = layers 16-23, 15 releases,
8,064 phone calls = ~1,008 of 1,114 Gemma tokens, 4 of 6 Gemma requests assisted); Qwen kept 12 layers
(4,950 calls, ~412 tokens, 4 of 15 requests). Energy is the same as variants A and C within noise: Gemma's
released share is 8 of its 25 CPU-resident layers at ~354 MB each and its decode is GPU-heavy, so ~1,000
assisted Gemma tokens are worth only a few kJ, inside the run-to-run band.

Where the missing 10 points are: baseline CPU 89.5 kJ -> D 68.8 kJ (-20.7 kJ). The ~418 Qwen tokens still decoded
on the host cost ~(75-41) J each = ~14 kJ, which alone would put the arm at ~97.5 kJ = -26 %. That is the M4a
coverage work (per-sequence release, truthful probes, probe budget, `minimum_remaining_tokens`). Variant E
(both release + `adaptive_minimum_remaining_tokens` 24 -> 8) tests the cheapest of those knobs next.

## Variant E: D + `adaptive_minimum_remaining_tokens` 24 -> 8

112.4 kJ (-15.0 %), 1,458 s; 5 of 15 Qwen and 4 of 6 Gemma requests assisted; 30 controls issued, on the same
requests as before. The remaining-tokens threshold is not what keeps the short Qwen requests off the phone; the
per-context probe budget and the thrash-poisoned cached winner are (M4a items 1-2, code). Configuration knobs are
exhausted on this trace at 15-16 %.

### Ledger of the 24-request arms (host kJ, single runs, baseline 132.3)

| arm | host kJ | saving | what changed |
| --- | ---: | ---: | --- |
| treatment 4 | 117.5 | -11.1 % | as run7 design, keep-cache restore |
| A | 111.2 | -15.9 % | Qwen sessions contiguous 0-11 |
| C | 112.7 | -14.8 % | Qwen 0-16 then 0-5+12-16 |
| D | 111.5 | -15.7 % | Gemma release on, dynamic layout |
| E | 112.4 | -15.0 % | D + min remaining tokens 8 |

### v2r with both models releasing (`s42-trace-v2r-bothrelease-20260921-inputs`)

282.8 kJ, 3,411 s, -16.8 % host (treatment 3 was -17.3 %): all 10 Qwen requests assisted (Qwen held layers 6-16
after Gemma took HTP0 = layers 0-7), 4 of 8 Gemma requests assisted (8-layer mask), trace 12 % shorter than the
baseline. The planner's split moved with demand and the total did not change: Gemma's phone work is cheap
energy, and the arm is still bounded by the release/restore thrash on the busy Qwen server (74 releases on one
server).

## Cohort formation on the trace

`decode_cohort_state` is `{"cohorts": {}, "receipts": {}, "estimator_ingested": []}` in every measured arm and every
phone call carries `tokens=1`. The M2 cohort machinery is intact; no request on this trace ever reaches its key path.
Evidence from the decision log of the `treatment-2` run (`physical/treatment-2/FAILURE_SCHEDULER_DECISION_LOG.json`,
24 DECISION + 2 REPLAN records, same scheduler code as the passing arms):

1. Every selected route is a desktop route: `auto:coordinated:physical:hot:desktop:residency:{cold,hot}` (17 records),
   `...:cold:desktop:...` (6), `...:desktop-control:4b90b1d7ae732467:...` (3). All 514 phone-assisted candidates in
   those records have `admitted: false` (MODEL_EPOCH_AUDIT_ONLY 376, PLACEMENT_INFEASIBLE 281, EXECUTOR_NOT_READY 278,
   ROUTE_NOT_QUALIFIED 230, COLD_RESIDENCY_BREAK_EVEN 200, PHONE_RESIDENCY_LAYOUT_NOT_SELECTED 166). Phone assistance
   on this trace arrives later, through the adaptive helper attachment on the committed desktop ticket, never as the
   selected plan.
2. `_commit_decode_cohort_leases` (`_unified/automated_requests_ops/commit.py:43`) passes `selected.plan` to
   `RuntimeDecodeCohortManager.admit` (`_internal/runtime_decode_cohort.py:633`). `candidate_key` (line 368) requires
   `ffn_assistance_phase == "decode"`, `ffn_runtime_control_protocol == "decode-boundary-v1"`,
   `ffn_weight_buffer_layout == "resident-superset"` and a `ffn_resident_geometry_sha256` in `plan.adapter_parameters`.
   The selected desktop plans carry none of the four (`None` in all 26 records); they do carry `parallel: 4` and
   `decode_cohort_formation_us: 2500000`, which `formation_us` (line 477) reads only after the key test. So
   `key is None` -> `admit` returns `None` (line 652) -> `commit.py:53` commits request-owned leases with
   `decode_cohort=None` -> no cohort is ever created -> the snapshot stays empty. Two further gates would fail even with
   a key: `capacity` (line 447) needs `usb_concurrent_streams` and `usb_queue_depth` >= 2, both absent on the desktop
   plan (capacity 1), and `_shareable_preparation` (line 467) needs every transition to prepare the phone, while the
   desktop plans' one transition prepares `desktop-cpu`/`desktop-cuda`.
3. The server behaves consistently with that: `server_slot::can_batch_with` (`tools/server/server-context.cpp:407`)
   batches decode slots only when `(layer_mask, columns)` are equal (`server_ffn_split_policy::operator==`, line 81),
   and `apply_ffn_split_ubatch_context` (line 3990) hands the phone one row per token of the policy-equal slots in the
   ubatch, so `tokens` is the number of assisted slots decoding together. The `ffn_split_cohort` control (line 2543)
   is issued only through `DecodeCohortPolicyView`, which the adapter builds from `ticket.decode_cohort`
   (`adapters/decode_cohort.py:58`), i.e. never. With 5 of 15 Qwen requests assisted and the queue serving one
   assisted request at a time (the `_bind_causal_predecessors` note in the plan), every batch that reached the phone
   held one token.

Smallest correct fix: not a few lines in `research_dev/scheduler`, so no code was changed here. The cohort key has to
be derived from the helper attachment, not from `selected.plan` at commit: the attached `HelperOpportunity`'s
`helper_operator_plan` carries the four parameters and the phone-only transitions that `candidate_key` and
`_shareable_preparation` look for, `ticket.execution_plan` does not. Admit the request to a cohort keyed on
(artifact, endpoint, desktop placement, helper geometry) when the adaptive controller attaches or issues the first phone
control, bind the shared lease then, and make the members' adaptive controllers converge on one `(layer_mask, columns)`
per (artifact, layout generation) rather than one cached winner per context, otherwise `can_batch_with` still splits
them. The cheaper route once per-sequence release lands: skip scheduler cohorts and let `update_slots` batch the
policy-equal assisted slots, which already yields `tokens = assisted slots` with no cohort control; it needs the same
cross-request agreement on the policy (M4a item 2). Either way the co-dispatch blocker recorded in the plan
(`test_adaptive_runtime.test_cold_phone_cohort_shares_transition_identity`) must be lifted first, or the second
assisted request waits for the first.

## M4a run 1 (server release rule + probe budget 4 + learning-demand decay + both release)

`s42-trace-v2a-m4a1-20260921-inputs/run-treatment-1`: 113.3 kJ (-14.3 %), 1,418 s, 22/24 outputs identical to
the baseline. The thrash is gone: the busy Qwen server made 4 releases, 1 restore and skipped 73 releases under
mixed slot policies (variant D: 39 releases / 39 restores). Coverage did not move: Qwen 4 of 15 requests,
3,322 calls on the busy server, 32 controls; Gemma 4 of 6. Energy equal to variants A/C/D within noise, so the
release/restore cost itself was small; what limits the arm is the controller not assisting the short requests.

Reason codes in the result: INSUFFICIENT_OPPORTUNITY 37, PRIOR_UNDER_MONITOR 42, VERIFICATION_MONITORING 6,
PROBE_CANDIDATE_REJECTED 12, EXECUTION_DEFERRED 6, CURRENT_PAIR_NOT_IMPROVED 0. INSUFFICIENT_OPPORTUNITY is
raised when a helper and probe candidates exist but the probe is unaffordable for the remaining tokens
(`maximum_probe_tokens` 80, four coarse fractions) and no compatible cached winner exists for the context. The
seeded store (2026-09-02, other bundle) holds Qwen phone winners, but its contexts do not match this bundle's
placement identity, so every context starts cold and only the long requests (>= 80 tokens) can afford the sweep.

## M4a run 2 (cheap probe: one 100 % candidate, 16-token probe, 8-token windows, min remaining 8, attempts 4)

`s42-trace-v2a-m4a2-20260921-inputs/run-treatment-1`: **125.3 kJ (-5.3 %)**, worse than every earlier arm. 6 of 15
Qwen requests probed (up from 4-5) but most probes ended with the baseline as final policy; Qwen phone calls fell
to ~1,500 (m4a1: ~5,200), CPU energy rose to 82.4 kJ. 24/24 outputs identical. Per-window medians still 39 J/token
phone vs 77 J/token host, so the phone path did not get worse; the controller's acceptance of 8-token windows
did (analysis below). Do not use this override set.

### The measured root cause of low coverage: mixed slot policies serialize decode and confound the window energy

m4a2 adaptive store, Qwen decode windows grouped by policy and concurrent batch size:

| window | active_batch | n | median J/token | median ms/token |
| --- | ---: | ---: | ---: | ---: |
| host (baseline policy) | 1 | 147 | 76.9 | 611 |
| host (baseline policy) | 2 | 64 | 78.0 | 620 |
| phone policy | 1 | 168 | 38.9 | 406 |
| phone policy | 2 | 10 | 128.1 | 1,175 |

Two host slots batch into one forward pass, so their per-token cost does not change with concurrency. A phone-policy
slot cannot batch with a host-policy slot (`can_batch_with` requires equal split policies), so the server
alternates batches: the assisted request waits ~610 ms for the co-tenant's host step and then spends ~400 ms on its
own phone step, its measured latency doubles, and the window's fleet energy contains the co-tenant's host decode.
Every probe that ran next to a host-policy co-tenant therefore read 108-204 J/token and was rejected against a
77 J/token baseline, although the phone path alone is 39 J/token. This, not the release thrash and not the probe
budget, is why 10 of 15 Qwen requests never kept the phone on this 4-slot trace, and why the realistic trace
(more single-tenant windows) reaches 17 %.

Fix: policy coherence per server. When one decode slot of a server switches to the phone policy, the other decode
slots of the same model on that server must follow (they then batch into one forward and one phone call with N
tokens, which is also the cohort M2 accepted), or the controller must measure and compare only windows with the
same batch composition. The first is the energy fix; the second only stops the wrong rejections.

## M4a run 3 (per-server policy coherence, both release, probe attempts 4, new server release rule)

`s42-trace-v2a-m4a3-20260921-inputs/run-treatment-2` (run 1 aborted on a coverage gap from a mid-window follow,
since fixed): **113.9 kJ (-13.8 %)**, 1,488 s. Qwen 6 of 15 assisted, 35 controls, only 8 `SERVER_POLICY_COHERENCE`
follows, and not one multi-token phone call (4,170 Qwen calls, all `tokens=1`). Phone windows next to a host
co-tenant still read 138 J/token (n=14) against 40 J/token alone (n=229); host windows 72-79 J/token at 1-3
slots. So the follow rule as implemented rarely applies: at a follower's start the helper is usually still
preparing (no `helper_available`), and by the time its first baseline window is recorded most short requests
have fewer than two windows left or the leader is gone. The co-tenant batching that would make the phone policy
pay under concurrency was never observed on this trace.

Where this leaves the 24-request trace: every arm today sits at 14-16 % host saving (baseline 132.3 kJ):

| arm | host kJ | saving |
| --- | ---: | ---: |
| treatment 4 | 117.5 | -11.1 % |
| variant A (contiguous Qwen sessions) | 111.2 | -15.9 % |
| variant D (Gemma release on) | 111.5 | -15.7 % |
| m4a1 (server release rule) | 113.3 | -14.3 % |
| m4a2 (cheap probe) | 125.3 | -5.3 % |
| m4a3 (coherence) | 113.9 | -13.8 % |

The realistic trace stands at -17.3 % (both-release -16.8 %). The remaining lever is unchanged and now precisely
located: assisted and host-policy slots on one server cannot share a forward pass, so under concurrency the
phone request is serialized and its measurement poisoned. Making co-tenants batch requires (a) followers to
attach before their first window (helper attach latency, `PREPARING`), (b) the server to co-batch policy-equal
slots (verify with `tokens>1` in `S41SERVERFFNUSB`), and (c) either per-row FFN placement inside one ubatch or a
policy that is decided per server rather than per request. That is a redesign item for the implementing agent,
not a configuration change.

## 2026-09-22 task 1 preparation: server policy ownership and shared helper leases

Local checks PASS: 240 tests across adaptive coherence/decode/runtime, campaign inputs, sustained assistance,
and runtime controller; pyflakes clean on changed files. `server_policy_coherence` defaults off and is enabled
through the campaign's typed adaptive overrides for the new arm. One controller owns the policy for an
artifact/desktop placement/layout generation; short followers can inherit at their first decode boundary.
Followers preserve per-request acknowledgements and contiguous token receipts. Compatible helper attachments
share one reservation, including renewal and last-member release. Mixed policies cannot supply comparison
evidence, and historical windows must match batch size and external-work context. Native dormant release and
batching are unchanged; the deployed binary contains 16 FFN markers. No physical PASS is claimed yet.

## 2026-09-22 task 2 trace construction

Build PASS with the handoff command, default `--min-output 4`, prompt/output caps 2048/1100,
duration 1800 s, request count range 12..20. Artifacts: `/mnt/storage/burstgpt-source/longdecode_v1`.
19 requests: 13 Qwen / 4 Gemma / 2 Llama; 10578 input / 3267 output tokens; arrivals span 1625 s.
No inputs or outputs are clipped. Output p50/p90/max is 152/446/472: the first eligible real window contains
no outputs above 512, so this window does not demonstrate the benefit of removing the old output cap.
Matched treatment and desktop-baseline energy arms remain pending. No filtered best-case trace was built.


## 2026-09-22 03:51 UTC: M4a4 shared server policy, first physical attempt - FAIL

All 24 requests completed, zero rejected, in 1340.004 s. Measured host energy
was 109.561 kJ (68.606 CPU + 40.955 GPU), a 17.158% saving against the matched
132.253 kJ baseline. Phone energy is separately assumed at 1.831 kJ.

| Handoff criterion | Measured result | Status |
| --- | --- | --- |
| Qwen USB calls with multiple rows | 0; all 3636 USB calls have one row | FAIL |
| Phone windows at active_batch >= 2, median <= 55 J/token | 17 windows, median fleet 57.141 J/token; host 55.468 J/token; 6 measurement-eligible windows | FAIL |
| At least 12 of 15 Qwen requests assisted | 5 of 15 | FAIL |
| Host saving at least 22% | 17.158% | FAIL |
| All 24 output streams token-identical | 22 of 24 | FAIL |

Exact token differences: Gemma request 88132 at zero-based token 2, request
88139 at token 36. Request 88139 made no phone calls. No logits were captured
to establish a cause or a near-tie exception; the strict correctness gate fails.
All 15 Qwen streams match the baseline.

The server does batch: 352 `S41SERVERFFNCALL` lines carry two rows, alongside
2932 one-row lines. Its dormant startup contract uses `usb_batch_plan=split-row`,
so each two-row forward is split into two one-row USB calls. Equal slot policies
and shared leases are necessary but do not change the immutable transport mode.
The second attempt must launch the already-qualified coalesced transport.

The first probe also switches back after one phone token (`PROBE_INCOMPLETE`).
Request-local probe ownership/budget checks still terminate a decision intended
to belong to the server. The next revision retains a bounded shared probe across
request completion and context changes, shares only evidence with matching batch
size and external activity, and honors that decision during helper arbitration.
244 targeted tests and pyflakes pass. Physical acceptance of this revision is
pending. No native rebuild, kernel change, or second-phone action was performed.

Evidence: [acceptance](physical/m4a4/ACCEPTANCE.json),
[raw result](physical/m4a4/RESULT.json), and saved native stderr / SSE streams
in `physical/m4a4/`. Remote inputs are
`/home/zhihao/s42-trace-v2a-m4a4-20260922-inputs`.


## 2026-09-22 03:57 UTC: M4a5 preparation - local PASS, physical pending

245 targeted tests pass; pyflakes is clean. The additional late-ack regression
ensures the request-local measurement deadline cannot cancel an admitted shared
server probe. The shared token budget still bounds unqualified execution.
The deployed source matches the 16-file [snapshot](physical/m4a5-source/SOURCE_SHA256.json).
The native library is unchanged and still contains 16 `S41SERVERFFN` markers.

Fresh inputs: `/home/zhihao/s42-trace-v2a-m4a5-coalesced-20260922-inputs`.
Relative to m4a3, adaptive overrides contain only `server_policy_coherence=true`;
Qwen additionally launches with `usb_batch_plan=coalesced-batch`, using the
accepted M2 envelope N<=4. Its qualified batch-plan list includes coalesced.
Gemma retains split-row. Both models retain the mixed-policy dormant-release
rule and cache-preserving release settings. The earlier m4a5 input directory
received preflight only and has no physical trace run.

The helper joins at the first decode callback, before its first recorded window
when the shared policy is already known. The HTTP control still needs a native
acknowledgement; assistance from the literal first generated token is not yet
verified. No first-token guarantee is claimed from unit tests.


## 2026-09-22 04:11 UTC: additional owner-election diagnostic - FAIL

A local reproduction outside the 245-test suite found a remaining case: an
unprobed short owner completes while a long follower remains active. Ownership
moves to the follower and its stage becomes `initial_baseline`, but its state
remains `EXPLOITING`; `_can_probe` is true and the next directive still only
opens another host window. Unfinished *existing* probes survive handoff, but
an unstarted probe does not begin on this path.
[Reproduction state](physical/m4a5-source/OWNER_HANDOFF_REPRO.json).

The second physical attempt has completed its first five Qwen requests with no
phone calls, so the 12/15 coverage target is unreachable. This diagnostic is a
confirmed local defect, not yet a verified attribution of the physical failure.
The active run is being allowed to finish; no third Task 1 run will be launched.
The independent Task 2 baseline/treatment pair is queued after its result and
uses `server_policy_coherence=false`. The production source remains the measured
m4a5 snapshot.


## 2026-09-22 04:29 UTC: M4a5 final physical acceptance - FAIL; Task 1 stopped

All 24 requests completed, zero rejected, in 1489.329 s (7.88% longer than
the saved baseline). Measured host energy is 94.909 kJ (52.483 CPU + 42.425 GPU),
a 28.237% saving. Separately assumed phone energy is 2.182 kJ.

| Handoff criterion | Measured result | Status |
| --- | --- | --- |
| Qwen USB calls with multiple rows | No Qwen phone calls | FAIL |
| Phone windows at active_batch >= 2, median <= 55 J/token | No Qwen phone windows; median unmeasured | FAIL |
| At least 12 of 15 Qwen requests assisted | 0 of 15 | FAIL |
| Host saving at least 22% | 28.237% | PASS |
| All 24 output streams token-identical | 22 of 24 | FAIL |

The **same strict correctness failure recurred**: Gemma 88132 first differs at
zero-based token 2 and 88139 at token 36, exactly as in m4a4. No logits justify
a correctness exception. Task 1 stops here under the two-failure rule; no third
Task 1 run is scheduled.

The Qwen failure is now attributable to admission, not an executed shared-probe
decision: every Qwen request has coalesced operator-split candidates rejected
with `TRANSPORT_PROFILE_INCOMPLETE`, no dormant FFN runtime in its terminal
parent plan, and no helper events. Setting a qualified batch-plan label does
not supply the campaign's missing matching transport profile. Coalesced Qwen
execution was never exercised, despite successful whole-rig preflight. The
separate local owner-handoff diagnostic is not the cause of these Qwen results.
See [route rejections](physical/m4a5/QWEN_ROUTE_REJECTIONS.json).

The saving occurred with **Gemma-only phone assistance**: 25368 Gemma phone calls
versus 7576 in m4a4. Desktop placements are unchanged (Qwen 16 GPU layers; Gemma
22). This is an observed full-trace saving, not evidence that Task 1's shared
Qwen path passed. First-token assistance remains unverified.

The production source remains the m4a5 snapshot. The feature is off by default.
The independent Task 2 pair uses the original split-row inputs with the feature
off; its baseline preflight started at 04:24:50 UTC after the final Task 1 run
released the rig lock. No native rebuild, second-phone action, commit or push
was performed.

Evidence: [acceptance](physical/m4a5/ACCEPTANCE.json),
[result](physical/m4a5/RESULT.json), [inputs](physical/m4a5/inputs/campaign.json),
and [source manifest](physical/m4a5-source/SOURCE_SHA256.json).

## 2026-09-22 05:04 UTC: Task 2 desktop baseline - PASS

The requested untruncated window completed all 19 requests, zero rejected,
with all 3267 output tokens present in the saved SSE streams. Duration was
2057.388 s; measured host energy was 203.348 kJ (139.943 CPU + 63.405 GPU).
Phone energy is separately assumed at 1.800 kJ, with zero active phone time.
This is a single baseline measurement; the matched saving and exact output
comparison remain pending. Treatment preflight started at 05:04:28 UTC under
the same rig lock and unchanged deployment.

Evidence: [summary](physical/longdecode-baseline/SUMMARY.json),
[result](physical/longdecode-baseline/RESULT.json), and saved native logs and
SSE streams under `physical/longdecode-baseline/`.

## 2026-09-22 05:13 UTC - Task 2 first output equality check FAIL; measurement continues

The first completed Gemma request, burstgpt_longdecode_v1:000, has all 446
output tokens but first differs from this pair's baseline at zero-based token
52. No logits establish the cause. The unchanged treatment continues to finish
the requested energy measurement; exact equality for the full pair cannot pass.

Evidence: [first output check](physical/longdecode-treatment/FIRST_OUTPUT_CHECK.json).

## 2026-09-22 05:40 UTC - Task 2 treatment execution FAIL; proof parser correction

The launcher exited 1 at 05:36:18 UTC, with 17 complete streams and request
017 interrupted after 81 saved tokens. No RESULT.json or matched host saving
exists. The primary error belongs to earlier Gemma request 002: expected 2496
FFN rows, parsed 2495. All 2496 USB and FFNCALL records exist. At native log
line 6333, an interleaved timestamp prefixes call 1610 / layer 1 without the
log severity letter; the strict parser discarded it. Allowing the existing
exact timestamp prefix with an optional severity restores all 2496 rows, 104
per layer. Separate cleanup error: actual lease completion outside reservation.
The rig lock is free and no desktop llama-server process remains.

Of 17 complete streams, 13 match exactly. Differences: Gemma 000/token 52;
Qwen 008/token 115, 013/token 7, 014/token 36 (zero-based). The log parser
correction cannot resolve these numerical differences. A fresh Task 2 treatment
retry is being prepared; Task 1 remains stopped.

Evidence: [failure](physical/longdecode-treatment/FAILURE.json),
[proof parser reproduction](physical/longdecode-treatment/PROOF_LOG_REPRO.json),
and saved failure observations, native logs and streams.

## 2026-09-22 05:45 UTC - Task 2 proof parser fix local PASS; retry preflight running

79 adapter/KV/reference tests PASS and pyflakes PASS. The two deployed files
match local SHA-256 hashes. Saved failed log now yields 2496/2496 calls, 104
per layer. Retry input manifests match attempt 1 after path/campaign-ID
normalization; no native rebuild or model/transport/controller tuning. Preflight
started 05:41:49 UTC under the rig lock. Reuse the 203.348 kJ baseline, which
made no phone calls and is unaffected by the phone-proof parsing correction.

Retry source snapshot: [manifest](physical/longdecode-parser-fix/SOURCE_SHA256.json).
Attempt 1: [failed arm summary](physical/longdecode-treatment/FAILED_ARM_SUMMARY.json).

## 2026-09-22 06:25 UTC: Task 2 completed pair - PASS; long-tail benefit unverified

The fresh treatment retry completed all 19 requests with zero rejected. The
[exact comparison](physical/LONGDECODE_PAIR_COMPARISON.json) checks identical
request IDs, models, prompt hashes, input/output lengths, seeds, source arrivals
and SLO fields, then compares all 3267 output token IDs. All 19 streams match.
Both arms met all 19 of the scheduler's recorded latency upper bounds.

| Arm | CPU kJ | GPU kJ | Measured host kJ | Assumed phone kJ | Host + assumed phone kJ | Duration s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop baseline | 139.943 | 63.405 | 203.348 | 1.800 | 205.148 | 2057.388 |
| Treatment retry | 100.838 | 63.452 | 164.290 | 2.773 | 167.063 | 2094.451 |

Measured host saving: **19.208%**. Including the
separately stated phone assumption gives 18.565%. Duration increases
1.801%. Phone power remains an estimate,
0.875 W idle and 4.5 W active; treatment active time is 259.446 s.
This is one successful pair, not a repeatability or statistical estimate.

| Model | Requests | Requests with verified phone calls | Output tokens |
| --- | ---: | ---: | ---: |
| Qwen | 13 | 11 | 1931 |
| Gemma | 4 | 4 | 1323 |
| Llama | 2 | 0 | 13 |

Both arms retain m4a3's split-row configuration and
`server_policy_coherence=false`. Desktop GPU layer counts are Qwen 16, Gemma 22
and Llama 16. The retry changes only the FFN proof parser and paths/campaign ID
relative to attempt 1. The baseline has zero phone calls and is unaffected by
that parser correction. No native rebuild or new transport identity was needed.

The builder used exactly the requested caps 2048/1100 and default min-output.
It selected a real, untruncated window with maximum output **472**, so there are
**zero requests above 512**. The requested trace construction and matched
measurement pass; a benefit from removing the old output cap was not exercised.
No filtered best-case trace was substituted.

Attempt 1 remains a failed arm, with no RESULT.json and no matched energy
saving. Its 17 complete outputs include four mismatches (one Gemma, three Qwen).
The successful retry's 19 exact outputs do not explain those earlier numerical
differences. The parser correction fixes log recognition; it is not established
as a numerical fix. Repeatability remains unverified.

The retry has no FAILURE.json or CLEANUP_FAILURE.json; the rig lock was available
after completion. Task 1 remains stopped after its repeated strict output
failure. Task 3 / OP11 was not started under that stop condition. No commit,
push, phone kernel change, or manual phone-worker kill was performed.

Evidence: [energy comparison](physical/LONGDECODE_ENERGY_COMPARISON.json),
[baseline result](physical/longdecode-baseline/RESULT.json),
[treatment result](physical/longdecode-treatment-r2/RESULT.json),
[retry inputs](physical/longdecode-treatment-r2-inputs/campaign.json),
[parser source snapshot](physical/longdecode-parser-fix/SOURCE_SHA256.json),
and saved native stderr / SSE streams in both physical arm directories.

Reproduce the comparison from this report directory:

```sh
python3 compare_trace_energy.py --run baseline=physical/longdecode-baseline --run treatment=physical/longdecode-treatment-r2
python3 analyze_longdecode_pair.py --baseline physical/longdecode-baseline --treatment physical/longdecode-treatment-r2 --output /tmp/longdecode-pair-review.json
```

## Task 1 resumed (2026-09-22 12:48 UTC)

The user reopened Task 1. Prior m4a4/m4a5 acceptance results remain FAIL.

**Owner handoff regression: PASS, 19/19 coherence tests; pyflakes clean.**
The historical `OWNER_HANDOFF_REPRO.json` omitted the real batch-membership
update after the short owner completed. It left two declared members with only
one live session, so the comparability guard correctly rejected every window.
The added regression reports next_active_batch=1: that transition window is
ineligible and the next stable window issues the phone control. The existing
production controller passes; the earlier claim of an owner-handoff defect is
withdrawn. This correction does not change either physical result.

M4a5's configured receipt set covers only 7,680- and 10,240-byte transfers;
coalesced Qwen with four worker rows requires 40,960 bytes. The larger September
16 receipts identify another qualification stack. Fresh current-stack transport
measurements are required before another coalesced acceptance arm.

### Fresh transport and admission (2026-09-22 12:58 UTC)

**PASS:** 9/9 measurements, payloads 7,680 / 10,240 / 40,960 bytes in both
directions and duplex, queue depth 4. Android USB restored after every case.
The initial precheck stopped on an unrelated retained Gemma service before
changing USB; a read-only inspection confirmed that its failed endpoints are
separate and g2 is unbound. It was left untouched. No kernel change or rebuild.
[Raw receipts and commands](physical/task1-transport-v2/RESULT.json).

Current kernel notes/BTF match verified candidate image f13c7c03..., whereas
the previous identity named 26e8d418... despite the same release string. Fresh
inputs bind the verified image and new receipts. The actual materialized Qwen
coalesced transport admission passes at 40,960 bytes and depth 4. The native
library retains 16 FFN markers. **PASS:** 154 local coherence, controller,
runtime and transport tests. The new physical trace result remains pending.

### M4a6 startup (2026-09-22 13:07 UTC): FAIL before inference

The input builder updated the qualification's boot hash but omitted the matching
rig field. The runtime rejected `phone transport qualification software identity
differs` before any request or energy measurement. Preflight had passed, but no
RESULT.json or streams were produced. [Preserved input and logs](physical/m4a6-startup-failure/RUN.log).

The builder now checks the verified image file and updates both manifests, then
asserts matching boot hash, kernel release and serial. New m4a6-r2 inputs are in
preflight. No kernel boot/flash was performed. Pyflakes passes for all changed
Python files. The five physical acceptance checks remain unverified.

### Controller and helper regressions (2026-09-22 13:30 UTC): PASS locally

256 tests PASS across coherence, adaptive controller/runtime, session replacement,
offline residency and automated admission. Pyflakes passes all seven changed
Python files. Regression tests reproduce and cover three barriers:

- An unfinished probe's temporary host fallback previously qualified the host
  without a completed comparison. The shared proposal and budget now survive
  an execution-context outage or incomplete probe.
- A short request can materialize a READY helper under server policy coherence;
  it still requires the existing physical, layout, safety and lease checks.
- Missing marginal contention cost can be measured by an adaptive probe. The
  route remains unqualified for direct selection, its evidence is LEARNING,
  and memory/transport rejection still prevents probe admission.

No native code changed. These corrections are local while m4a6-r2 finishes;
physical verification remains pending.

### M4a6-r2 (2026-09-22 13:52 UTC): Task 1 FAIL

All 24 requests completed and all 24 outputs match baseline. Host energy
132.253 -> 138.748 kJ (+4.911%); duration 1380.566 -> 1519.066 s.
Assumed phone energy 1.533 kJ. Qwen assistance is 2/15; 763 one-row USB
calls and zero multi-row calls. Eleven phone windows had active_batch>=2,
zero eligible, median fleet 111.602 J/token (host 109.930). Thus only the
output-identity check passes. [Acceptance](physical/m4a6-r2/ACCEPTANCE.json).

Final records confirm helper rejection for missing marginal contention cost,
short-request preparation gates, and different helper transport variants
selected for one desktop. Additional local corrections pin READY helpers to
the server's USB batch plan and admit probes against the longest remaining
co-tenant decode in the same group, retaining its shared token cap. An already
available helper refresh uses helper_rebound rather than helper_ready.
260 broad tests PASS; after the final rebind correction, 18 focused checks and
the multi-session integration test PASS. Pyflakes is clean. M4a7 is pending.

### M4a7 (2026-09-22 14:00 UTC): preflight/admission PASS; acceptance pending

The 11 deployed files match the tested [source snapshot](physical/task1-controller-fixes-v2/VALIDATION.json).
Physical preflight PASS and actual coalesced capacity admission PASS at
40,960 bytes / depth 4. The fresh 24-request arm launched at 13:59:58 UTC
under the shared lock. All five physical acceptance checks remain pending.
No native rebuild or phone change. Previous rig files are backed up at
/mnt/storage/s42-task1-before-m4a7-20260922.

#### M4a7 live shared-call check (2026-09-22 14:08 UTC): PASS

272 Qwen USB calls carry two rows; 485 carry one row. Matching forward and USB
request/layer IDs show requests 88119 and 88121 in the same call, payload
20,480 bytes. [Matched native evidence](physical/MULTI_ROW_MATCHED_PROOF.json).
All first three Qwen requests used the phone and all five completed outputs
match baseline. Final energy, coverage, window efficiency and 24/24 identity
are pending; these partial results do not establish Task 1 acceptance.

#### M4a7 live coverage check (2026-09-22 14:14 UTC): FAIL

Seven Qwen requests completed; only 88118, 88119 and 88121 have native phone
calls. Completed 88122, 88123, 88125 and 88126 have none, so final coverage
cannot exceed 11/15. The required 12/15 threshold is unreachable in this arm.
The run remains active for final energy, windows, identity and failure diagnosis.

### M4a7 ended (2026-09-22 14:35 UTC): Task 1 FAIL; stop condition met

The launcher exited 1 at 14:19:45 UTC. This arm has no RESULT.json and cannot
supply a matched energy saving. [Partial acceptance](physical/m4a7/PARTIAL_ACCEPTANCE.json),
[failure](physical/m4a7/run-treatment-1/run/FAILURE.json).

| Acceptance check | Result | Evidence |
| --- | --- | --- |
| Qwen USB calls with >1 row | PASS | 272 two-row calls; 485 one-row calls |
| Batch>=2 phone median <=55 J/token | Not verified | Failed request's terminal group was not committed; no complete arm |
| At least 12/15 Qwen assisted | FAIL | 3/15 scheduled; only 10 Qwen streams completed |
| Host saving >=22% | Not verified | No final trace energy result |
| All 24 outputs identical | FAIL | 15 complete streams, all 15 identical; nine outputs incomplete or absent |

The scheduler records 14 COMPLETED, nine CANCELLED and one FAILED. Stream
completion alone is not terminal proof success. Qwen 88121 finished its output
but its proof failed; the long Gemma 88132 stream was later cancelled at 345
of 491 tokens. No token difference was found in the 15 complete streams.

The **same helper identity mismatch recurred**, so hardware retries stopped under
the handoff rule. M4a6-r2 selected different helper executor IDs for 88126/88127;
m4a7 did the same for 88119/88121 versus 88123. These requests share desktop
placement 42a306..., layout generation 3, geometry 1ff4e78..., and mask 131071.
[Extracted helper envelopes from both journals](physical/REPEATED_HELPER_VARIANT_MISMATCH.json).
**Correction at 15:16 UTC:** the route names suggested split-row versus coalesced,
but both executor IDs actually declared coalesced batching. All three m4a7
desktop tickets also contain the coalesced dormant-runtime hint. The filter
was not bypassed: it accepted both duplicate identities. The earlier missing-hint
hypothesis was incorrect. The catalog replay below establishes the cause.

The terminal proof failure is separate: the checker expected 476 Qwen layer
rows and saw 493, exactly one forward per each of 17 layers. Native control
applied at token 16 of 45; all 29 forwards are present. The tail checker
unconditionally deducted one row to account for a statistics read running
ahead of a requested boundary. In this case no row was counted ahead.

**Local correction PASS:** use the last policy acknowledgement and completed
row counters to calculate that deduction. Historical records without counters
keep their existing treatment. The new regression failed before the correction;
180 tests pass across server adapter, adaptive controller/coherence and decode
cohort; pyflakes passes both changed files. Missing/extra rows and incorrect
generations are still rejected. [Validation and source hashes](physical/task1-proof-tail-fix/VALIDATION.json),
[failed-arm accounting replay, 493/493](physical/task1-proof-tail-fix/M4A7_REPLAY.json),
[previous-arm replay, 66/66 and 697/697](physical/task1-proof-tail-fix/M4A6_REPLAY.json).
This two-file correction is local only and has not been physically verified.
The eleven earlier controller/helper changes remain deployed as the m4a7 snapshot.

The failure artifact also reports a cleanup transition failure without fallback.
A read-only end check found the lock available, both phones visible on ADB 5037,
and no owned desktop server or bridge process. No worker was manually killed,
no phone/kernel change was made and no native build was performed. Task 1
remains stopped/FAIL. Task 2's earlier matched result is unchanged; OP11 was
not used.

### Helper identity correction (2026-09-22 15:16 UTC): LOCAL PASS; Task 1 physical FAIL

The Task 1 input patch set the common `phone_adapter_parameters.usb_batch_plan`
to `coalesced-batch` while declaring both batch-plan variants. Catalog
materialization copied that override into the unsuffixed executor, then emitted
the explicit coalesced executor as well. In both failed archives, each Qwen
parent has two helper capabilities whose fields are identical except for
`executor_id`. The adaptive policy identity includes that ID, so co-tenants
selecting opposite aliases cannot follow the same server policy. Checking the
batch-plan value alone could not reject either alias.

The materializer now emits exactly the declared batch modes, assigns each mode
its own parameters and qualification, and permits a coalesced-only declaration.
Shared coordinator resources remain present when the split-row variant is absent.
The Task 1 input builder now declares only qualified coalesced Qwen helpers and
removes the conflicting common override. The admission checker requires exactly
one qualified coalesced helper per Qwen parent. Adaptive policy identity and
proof checks remain strict. The dormant host-release rule is unchanged.

| Local check | Result |
| --- | --- |
| Regression before correction | FAIL as expected: duplicate/missing batch identity, coalesced-only declaration rejected, and incorrect qualification |
| Catalog, configuration, transport, adaptive runtime/coherence, cohort, adapter tests | PASS, 153 tests in 13.657 s |
| Multi-session and offline residency tests | PASS, 59 tests in 54.652 s |
| Pyflakes | PASS, all seven checked Python files |
| Archived m4a6-r2 and m4a7 catalog replays | PASS, 2/2; old catalogs rejected, corrected catalogs accepted |
| Corrected Qwen helper count | One per parent, both CPU and desktop; each retained capability exactly matches the archived qualified coalesced capability |
| Hardware rerun, >=12/15 coverage, <=55 J/token, >=22% saving, 24/24 outputs | Not verified; no new hardware arm |

[Reproducible catalog replay](physical/task1-helper-identity-fix/REPLAY.py),
[replay results](physical/task1-helper-identity-fix/REPLAY.json),
[validation and source hashes](physical/task1-helper-identity-fix/VALIDATION.json).
The earlier tail-proof correction also remains local: 493/493 failed-arm rows
reconcile offline. Both corrections need physical validation together. Do not
reuse previous derived input directories after the input-builder change.
No source was deployed, no run was launched, and the repeated-run stop remains
in effect. This fixes two reproduced code defects, not the measured Task 1 result.

### M4a8b retest active (2026-09-22 15:40 UTC): shared-forward check PASS

The user reopened physical retesting and confirmed another session owns this
same retest. This session monitors it read-only. All 15 deployed source/test
hashes match [the retest manifest](TASK1_RETEST_SOURCE_HASHES.json), including
the helper identity and tail-proof corrections.

M4a8 failed before physical execution: the coalesced-only declaration removed
an executor still named by an existing calibration route profile. The local
catalog replay had filtered those profiles; the full campaign materializer
had not. M4a8b instead retains split-row as SHADOW and qualifies only coalesced.
The two modes now have distinct runtime settings, with one qualified coalesced
helper per Qwen parent. Physical preflight PASS. This differs from the stricter
coalesced-only inputs described above; that input patch still needs the campaign
profile references handled before it can be used unchanged.

At 15:39:31 UTC, Qwen has 322 one-row, 221 two-row and 221 three-row USB calls.
The shared-forward criterion is PASS with 442 multi-row calls. Four of five
completed Qwen requests used the phone, including previously unassisted 88123.
All five complete outputs match baseline exactly. The latest scheduler snapshot
marks previously failing 88121 COMPLETED. The remaining acceptance checks await
the full trace; early success is not Task 1 acceptance.

Rig input directory: `/home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs`.
[Latest read-only monitor](physical/m4a8b-monitor/LIVE.json). The run is active
under the shared lock; do not deploy source or start a competing run.

#### M4a8b coverage check (2026-09-22 15:52 UTC): FAIL

Ten Qwen streams have completed; only 88118, 88119, 88121 and 88123 have native
phone calls. The second Qwen burst completed entirely on the host. Coverage
therefore cannot exceed 9/15, below the required 12/15. All 15 completed outputs
match baseline exactly, and the scheduler marks them COMPLETED. The run has
no failure artifact and continues for final energy and full-trace analysis.
The catalog identity correction enables real shared calls, but is insufficient
for Task 1 acceptance. The cause of the later unassisted requests awaits final
controller records.

### Task 1 reopened 2026-09-22 15:30 UTC: helper-identity fix deployed, one gap in the input builder

The user reopened Task 1. Both local corrections were deployed to
`/mnt/storage/s42-trace-v2-20260921-prep/source` and re-verified there
(`test_catalog_materialization`, `test_adaptive_coherence` PASS; 118 tests PASS on the controller host
across catalog, campaign inputs, coherence, adaptive runtime, decode cohort and transport profiles).

`PREPARE_TASK1_INPUTS.py` as written declares Qwen with `phone_batch_plans = ["coalesced-batch"]` only.
With the corrected materializer, which now emits exactly the declared modes, that removes the
`operator_split` and `operator_offload` executors that the measured route profiles still reference, and
catalog materialization fails with `RuntimeCapabilityError: runtime route profile executor is absent`
(arm `s42-trace-v2a-m4a8-20260922-inputs`, preflight, no rig time spent).

Minimal correction used for the live arm `s42-trace-v2a-m4a8b-20260922-inputs`: declare both batch modes
and qualify only the coalesced one, with the conflicting common `usb_batch_plan` override still removed.
The catalog then materializes and carries 35 Qwen phone capabilities with **zero groups identical except
`executor_id`**, so the duplicate-identity defect the helper-identity fix targets is gone in this
configuration while every measured route profile still resolves. Either the builder should adopt this
shape, or the route-profile set needs pruning to match a coalesced-only declaration.

### M4a8b final assessment (2026-09-22 16:06 UTC): FAIL

The other session's launcher exited 1 at 15:58:47 UTC. FAILURE.json reports
`PhysicalAdapterError: physical transition failed without a fallback`.
There is no RESULT.json and no completed-arm energy saving to compare.
Read-only archive: [partial acceptance](physical/m4a8b-monitor/PARTIAL_ACCEPTANCE.json),
[failure](physical/m4a8b-monitor/run-treatment-1/run/FAILURE.json),
[terminal monitor](physical/m4a8b-monitor/LIVE.json).

| Task 1 check | Measured result | Status |
| --- | --- | --- |
| Shared Qwen phone calls | 221 two-row and 221 three-row calls; 322 one-row calls | PASS |
| Observed phone windows at batch >= 2 | 19 windows; median host + assumed phone 38.134 J/token, host 36.856; 3 measurement-eligible windows | PASS for observed windows |
| At least 12/15 Qwen requests assisted | 4/15 scheduled; 4/10 completed Qwen requests | FAIL |
| At least 22% host saving | No completed-arm energy result | Unverified |
| All 24 outputs token-identical | 19 complete and identical, 5 absent | FAIL, incomplete |

Window statistics select bound groups for the four requests with native phone
calls in this arm, excluding historical observations. These are diagnostic
windows from an incomplete run, with phone power assumed. Qwen 88121 completed,
so its previous terminal tail-proof abort did not recur. The five requests in
the second Qwen burst attached helpers but made no phone calls: 120 assistance
decisions selected fraction zero with `SERVER_POLICY_COHERENCE`. That identifies
the remaining coverage symptom; its controller cause is not yet established.
The final transition failed on Qwen 88133; four further Qwen requests were
cancelled. The underlying transition error is not established by FAILURE.json.
No deployment, second run, phone change or worker kill was performed by this
monitoring session.

### Energy and output audit (2026-09-22 16:06 UTC): current >25% target FAIL

A fresh read-only comparison of all 12 completed v2a arm results against the
saved baseline also checks every raw output token stream. It corrects any
interpretation that the earlier 14-16% energy arms all passed strict output
identity. [Full audit](physical/CURRENT_ENERGY_AUDIT.json).

| Comparison | Baseline host kJ | Treatment host kJ | Host saving | Exact outputs |
| --- | ---: | ---: | ---: | ---: |
| Best current 24-request arm satisfying output identity, treatment 4 | 132.253 | 117.533 | 11.130% | 24/24 |
| Layout A, previously quoted as 15.9% | 132.253 | 111.245 | 15.885% | 22/24, FAIL |
| Lowest measured current energy, m4a5 | 132.253 | 94.909 | 28.237% | 22/24, FAIL |
| Current Task 2 matched pair | 203.348 | 164.290 | 19.208% | 19/19 |
| Historical v16/v16c matched pair, older deployment | 145.041 | 107.582 | 25.827% | 24/24 |

The historical pair did exceed 25%: measured host saving 25.827%, or 25.118%
including separately assumed phone energy (146.353 -> 109.592 kJ). Its original
comparator reports equal source files and a valid matched comparison; this
audit independently rechecked all 24 token streams. It is the v16/v16c pair
from the September 12 admission report, not run7 and not the current deployment.
The introduction's attribution of the 25.12% figure to v15/v15c is incorrect.
[Historical audit](physical/HISTORICAL_V16_ENERGY_AUDIT.json).

The best recent completed, token-exact matched result remains Task 2's 19.208%
host saving (18.565% including assumed phone energy), one pair with maximum
output 472. The current setup has not reproduced the historical >25% result.
Comparing layout A to the older 150.8 kJ control gives approximately 26.2%,
but changes the baseline and still fails token identity; it is not a qualified
current >25% result. The earlier rounded 26.3% estimate used rounded energies.

## 2026-09-22 16:40 UTC - OP11 qualification only, independent of trace energy arms

OP11 OpenCL/TCP functional **PASS**: 48 calls, 112 rows, max relative L2
3.274e-4 against the CPU reference; matching identities and normal exits.
The 2/4-row latency is a deployment blocker: 4751.604 / 4749.101 ms per
layer call versus CPU 18.850 / 19.653 ms. Shared-decode suitability **FAIL**
on this measured comparison. No new energy arm or two-phone integration
was run. The parent/shard identity check passed for all 18 stored tensors.
User authorized qualification only; full integration stays deferred.
See [M3 qualification report](../20260922-fast-path-M3/README.md).

### 2026-09-22 18:15 UTC - OP11 NPU qualification update

The isolated OP11 NPU repair is **PASS** for Qwen layers 18-21 at 1/2/4 rows:
48 calls, 112 rows, max relative L2 0.000167056. Root cause was the DSP's
16 mapping slots despite the host's 64-buffer batch limit; normal DMA works
after fixing the mapping capacity/reuse/eviction. Median NPU round trips
59.622/63.301/77.097 ms replace OpenCL's 68.652/4751.604/4749.101 ms.
This is an FFN qualification, not another trace/energy arm. No new energy
saving, full-model token comparison or two-phone integration is claimed.
Wi-Fi unmeasured pending an OP11 network connection. Shared deployment and
OP15 unchanged. Details: [M3 NPU report](../20260922-fast-path-M3/README.md).

### 2026-09-22 19:26 UTC - Pixel 10 Pro qualification, no new energy arm

Pixel Vulkan functional **PASS** on Qwen layer18: 12 calls/28 rows, maximum
relative L2 0.000311882. USB negotiated5000M. Median round trips at rows1/2/4
103.110/142.184/201.093 ms, slower than the matching OP11 NPU one-layer test
20.673/63.005/77.991 ms: latency improvement **FAIL**. Tensor TPU, Wi-Fi,
full-model tokens, energy and integration unverified. No change to the trace
energy results above. [Pixel qualification report](../20260922-fast-path-M3/README.md).

### 2026-09-22 19:58 UTC - Pixel TPU microbenchmark, no new energy arm

Real TensorG5 TPU add round-trip **PASS**:180/180 exact TCP results, plus20
standalone smoke calls. Reply framing A/B/A produces median50.003/4.666/50.070ms;
best p905.203ms. This is128-element add, not Qwen FFN. No compiler SDK for
matched FFN AOT test; no change to energy/full-token claims. Coalescing production
FFN replies is an unverified follow-up candidate; production code unchanged.
[TPU qualification report](../20260922-fast-path-M3/README.md).

### 2026-09-22 21:09 UTC - Pixel GPU real-server FFN, separate short energy test

Pixel Vulkan server assistance **PASS** for Qwen layers18-23:744 decode calls,
all four64-token outputs exact. After larger FFN blocks and coalesced TCP replies,
50%/100% column splits use4329.807/4305.035J host request energy against bracketing
controls4800.048/4892.714J:10.659%/11.170% saving against their mean. Decode is
9.3%/18.1% slower. Raw-token, per-layer proof and raw-power audits PASS.
One synthetic prompt, one sample per split, one active slot; Pixel energy
unmeasured. This is not a rerun of either trace above and does not change the
current trace savings or establish25%. Automatic scheduler and two-phone
integration remain deferred. Cleanup PASS; OP15/shared server unchanged.
[Pixel GPU server report](../20260922-fast-path-M3/README.md).
