# Qwen3-32B two-phone trace arm (2026-10-08)

User direction: both phones on one big quantized model, single-model trace, phones in prefill too (frees desktop RAM).
Approved 2026-10-08 19:25 UTC ("ok work as you suggested"): scheduler work for a real 32B trace run with both phones.

## Strategy

Layers run in sequence, so splitting them between phones adds no speed; the measured placement planner gives each
layer to the cheapest device with room.

- **Q4_0**: 22 CPU layers x 221 MB = 4.9 GB. The OnePlus holds them all (two HTP sessions), and it is 2.5x faster
  per layer than the Pixel, so the planner is expected to use the OnePlus alone.
- **Q8_0**: 42 CPU layers x 418 MB = 17.5 GB, more than either phone holds (OnePlus about 21 layers in three
  sessions, Pixel about 20). Capacity forces both phones in, which is the user's reason for Q8.

So the two-phone arm targets Q8_0 if the per-layer screen shows both phones are fast enough; Q4_0 validates the
pipeline end to end first (smaller files, server tests already PASS).

Phase 1: the existing measured-placement arm (`derived-mpaoa`): OnePlus primary shards over FunctionFS, Pixel static
co-helper over AOA, adaptive decode, desktop keeps all weights, prefill on the desktop.
Phase 2: OnePlus as remote-resident owner (desktop drops its layers; phone prefill with the fast call path), with the
Pixel co-helper beside it. Blockers in the 2026-10-08 scheduler-agent report (decode planner never drives the Pixel on
a remote-resident route; proof rejects co-helper calls; gate driver F16-only; no catalog materialization).

## Phase 1 status (2026-10-08 21:45 UTC)

| model | step | status |
|---|---|---|
| Q4_0 | measured plan (`layer_placement plan`, profile `--desktop-layer` + 32B shapes, hand-written inventory) | ideal = OnePlus 0-20, 7.6 vs 23.0 J/token CPU-only; Pixel unused |
| Q4_0 | OnePlus shards HTP0 0-10 / HTP1 11-20 + Q4_0 discovery + template-q32op15-ts3 | done (q32_op15_onboard.sh) |
| Q4_0 | preflight (KIND op15) | PASS; catalog: OnePlus composites, runtime column block 3,200 |
| Q4_0 | 3 metered repeats pQ32O1{opc,op,opb} | DONE 21:40 UTC, all PASS 14/14: fleet 88.0 / 89.8 / 89.0 kJ, mean 88.9 vs 124.4 desktop-only (-28.5 %); CPU 84.3 -> 36.3 kJ, GPU 37.9 -> 43.0, OnePlus 1.4 -> 8.9 kJ (5.1 W); TPOT p50 259-261 vs 276-277 ms; TTFT p50 11.9-13.2 vs 1.7 s, E2E mean 105 vs 77 s (active_slots_peak 1: same-model arrivals queue behind the assisted leader) |
| Q8_0 | per-layer screen | CPU 15.2 ms, Pixel 17.2 ms, OnePlus 7.6 ms (production worker needs column blocks <= 12,800: VTCM) |
| Q8_0 | measured plan | OnePlus 0-20 (3 sessions, 8.8 GB) + Pixel 21-40 (8.4 GB): 24.7 vs 40.3 J/token |
| Q8_0 | Pixel evidence for a new model: `prepare_aoa_evidence retarget-model` (fresh server identity + plain FFN numerics, link/session receipts carried with an annex), `native/ffn_numerics.py` (ggml-cpu reference), identity-tool `ffn_numerics` hook | code + tests done (retarget 2/2, AOA/WS12/WS15 41/41) |
| Q8_0 | shards, pushes, discovery, AOA identity, evidence, template-q8two-ts3, derive_mpq8.sh, preflight | DONE 22:13 UTC: identity 4/4, numerics 2.6e-4, plan OnePlus 0-20 + Pixel 21-40 (24.7 J/token), preflight pQ8T0 PASS |
| Q8_0 | 3 metered repeats vs 233.0 kJ | pA1q8e FAILED 01:45 UTC 10-09: OnePlus NPU session HTP2 hung at call 2,112 (not USB / memory / heat; Q4_0 ran 50-60k calls hang-free); queue paused. ROOT CAUSE (not Q8-specific): `htp_packet_callback` yields VTCM on a release request before reading a packet already signalled -> host waits forever; reproduced on the phone alone (5 hangs / 74k calls, all at a session handoff); 5-line fix in `htp/main.c`, fixed skel d3af8783 0 hangs / 126,700 calls. Restarted 08:15 UTC 10-09 as pA1q8{d,f,g} on `template-q8two-ts3-anchor-hfix` (`anchor_q8h_go.sh`: preflight pA0q8h, then the queue) |
| Q8_0 | Pixel identity: FFN numerics | first run FAIL 2.9e-3: my reference rounded Q8_0 activations with roundf, ggml's NEON/AVX quantizers round half to even (60/60 calls explained by input ties); fixed (np.rint), rerun PASS worst 2.6e-4 (f16 wire floor) |

## Server batching fix ("prompt anchor", 2026-10-08 22:00 UTC)

The Q4_0 runs served one request at a time (active_slots_peak 1, TTFT p50 12-13 s vs 1.7 s). The scheduler dispatched
every request within 0.6 s; the wait was inside llama-server (server_queue_s 36-94 s). Slots batch only with an equal
FFN policy, a new prompt has the default policy (FFN control needs decode state), and the anti-starvation anchor
rotation considered decoding slots only, so a prompt waited for every phone-policy decoder to finish. The same bug
cost single 14B requests 43-217 s (p7f41, p7g1).

| step | status |
|---|---|
| fix: `server-context.cpp`, pending-prompt slots take a turn as the anchor | done; native test `tests/test_server_prompt_anchor_native.py` FAILS on the old server, PASSES on the new (5/5); s2a / two-phone / remote-resident native suites pass |
| desktop build `$Q/anchor-build` (overlay of production; cuda-build untouched) | done, impl 0d444833 (production 52d697db), llama-server binary unchanged |
| `anchor_repin.sh`: transport identity ANCHOR, Pixel server identity r6a, Q8 identity + retarget on the anchor server, template-*-anchor, preflights pA0q4 / pA0q8 | DONE 22:46 UTC, both PASS (fixes: `S43_GATE_SERVER` in the rig prepare script; `retarget-model --host-software`) |
| metered: Q4_0 OnePlus + Q8_0 two-phone on the anchor server, 3 repeats each | first Q8 arm pA1q8a ran DESKTOP-ONLY (excluded): Pixel evidence column_quantum 12800 left no hot phone route (needs one quantum >= lcm(6400, 12800) dividing the split widths 6400/12800/19200); co-helper offload routes are transfer-infeasible as in the 14B runs. Pixel re-qualified at quantum 6400 (identity 4/4, numerics 2.6e-4), preflight PASS, all hot phone-assisted split routes plan OK (laptop check). Queue restarted 23:36 UTC: q4a/q4b/q4c + q8d/q8e/q8f |
| Q4_0 OnePlus, anchor server, 3 repeats pA1q4{a,b,c} | DONE 01:29 UTC 10-09, all PASS 14/14: fleet 85.4 / 85.2 / 84.4 kJ, mean 85.0 = -31.7 % vs desktop-only (old server 88.9, -28.5 %); TTFT p50 4.5-4.8 s (old 12-13 s, desktop-only 1.7 s), queue 34-42 s (old 352-375); TPOT mean 293-308 ms (old 260, desktop-only 266): a 2-row decode step costs 358 ms with the phone vs 280 ms desktop-only (1 row 282 vs 276). pA1q4c loaded the model cold (157 s, cause not found): its first three requests waited behind the load (TTFT p90 124 s) |
| Q8_0 two-phone, anchor server, 3 repeats pA1q8{d,f,g} | DONE 10:24 UTC 10-09 on the hfix template: 214.0 / 210.8 / 215.1 = 213.3 kJ, -8.5 % vs 233.0; TPOT p50 1.37-1.43 s vs 1.17 s; TTFT p50 168-222 s vs 31-38 s (Pixel 2-row steps +43 %, OnePlus +15 %). Paper filled |
| preflight guard: a phone arm must have a phone route that plans | deployed 06:56 UTC 10-09 (2 files only), on for the Q8 restart: `PhysicalPreflightReport.phone_route_plannable_models` (generated phone routes with no placement-class rejection: PLACEMENT_INFEASIBLE / TRANSFER_PLAN_ABSENT / MEMORY_CAPACITY / KERNEL_SUPPORT_ABSENT / SLO_UPPER_BOUND); `run_chain_eval.py` opt-in `S43_REQUIRE_PHONE_ROUTE=MODEL[,MODEL]` -> preflight_status NO_PHONE_ROUTE (desktop arms exempt). On the real reports: pA1q8a lacks qwen3-32b-q8_0, pA1q8e / pA1q4b have their 32B. `phone_assistance_ready` cannot do this: it is false in every arm (phones not resident at preflight). Test `tests/test_preflight_phone_route_plannable.py` + `test_physical_preflight` 11/11 |


## Facts

- Q4_0 parent sha256:4b3aaa12...; Q8_0 parent sha256:2c50eb8a... (/home/zhihao/models/Qwen3-32B-Q8_0.gguf).
- Real CPU layers (llama.cpp first GPU layer = n_layer + 1 - ngl): Q4_0 0-21, Q8_0 0-41; the scheduler believes one
  fewer (kept for the 32B arms).
- Per layer, Q4_0 (WS16 screen, 2026-10-08): OnePlus HTP 3.9 ms / 0.077 J, desktop CPU 10.8 ms / 0.151 J, Pixel
  9.4 ms pinned (aff-7c-t5: libs /data/local/tmp/s43-q32-pixel-aff, worker 57577797).
- OnePlus FunctionFS decode call (32B Q4_0, production bundle): rpc p50 5.1 ms, compute 4.0 ms.
- Transport identity FUSED3 is keyed by payload bytes (n_embd 5120 x rows x 2): reusable for the 32B.

## Phase 2 status (2026-10-10 01:10 UTC): Q8_0, OnePlus owns layers 0-20 (prefill + decode), user: Q8 is the capacity test

Scheduler code (laptop, opt-in, uncommitted): target A (remote-resident owner) + target B (Pixel static co-helper),
`PHASE2_TARGET_A.md`; Q8 Phase 1 fixes `device_set_subset_challenge` + `live_membership_probe_timeout_ms`
(the /slots probe timed out 584x at 1 s on Q8 steps -> 76 % of tokens on the desktop).

| # | step | where | status |
|---|---|---|---|
| 1 | deploy scheduler (69 files, rsync to $R/stage) | rig, after the yield A/B pZ | done 22:47 UTC (yield A/B: er 84.3 vs pa 84.0 kJ, neutral) |
| 2 | Q8 Phase 1 v2 runs (`template-q8two-ts3-anchor-hfix-v2`, both fixes) x3 vs 213.3 / 233.0 kJ | rig | **done 00:49: 120.4 / 117.3 / 116.9 = 118.2 kJ = -49.3 % vs desktop-only** (phone-assisted tokens 24 % -> 94 %: OnePlus-only 37 % at batch 1-2, both 57 % at batch 3) |
| 3 | phone bundle: 10-08 fast-call bundle + fixed skel 52012757 (`$Q/p2/p2_push.sh` -> /data/local/tmp/s43-p2-owner-v1) | rig | done 00:54 (PUSH_OK) |
| 4 | desktop server: link anchor-build server-context + fastcall-build server.cpp/client objects (`$Q/p2/p2_build.sh` -> `$Q/p2-build`) | rig, after 2 | done 00:54 (BUILD_OK; first try hit a pipefail/SIGPIPE in the string check) |
| 4b | unmetered owner smoke, Q8 0-20 owner, quantum 6400, 512 + 2k prompts phone vs desktop (`$Q/p2/p2_smoke.sh`, log `$R/chains/P2-SMOKE.log`) | rig, after 2 | done 01:07. 512: tokens identical, prefill 7.5 vs 7.95 s, decode 684 vs 776 ms/token, file RSS 12.3 vs 20.3 GiB; 2k: identical, 23.8 vs 21.8 s, 749 vs 815 ms/token, 12.8 vs 20.3 GiB |
| 5 | **user:** raise desktop usbfs_memory_mb to >= 96 (128 suggested): (5,242,880 x 4 + 65,536) x 4 = 84 MB | user | done 07:33 (128); OP15 charging latch (notify 512) cleared by the RAM-boot reset 07:38 (user OK, `$Q/p2/op15_latch_reset.sh`) |
| 6 | FunctionFS qualification at 5,242,880 B (`qualify_q8owner_5mb.py`, written; refuses below 84 MB usbfs), identity Q8-OWNER | rig | after 5 |
| 7 | `template-q8owner-ts3` (`$Q/p2/compose_q8owner.py`), Pixel server identity r7p2 on the p2 server, transport identity, preflight (KIND op15) | rig | done 08:24: 5.24 MB receipts 07:39; pB0q8o failed (receipt dir missing from evidence -> catalog saw 61,440 B); pB0q8p BLOCKED TRANSPORT_PROFILE_INCOMPLETE (16 MB anchor receipts + 128 MB new receipts: route needs one usbfs value); anchor payloads re-measured at 128 MB (`p2_requal128.sh`, identity Q8OWNER128, 18 receipts); pB0q8q PASS (remote-resident route blocked by nothing) |
| 8 | Phase 2 A runs x3 (queue pB), then B (+ Pixel 21-40 static co-helper), RAM ledger | rig | DONE 2026-10-11 01:46. A pB1q8od/oa/ob 166.4/167.1/166.9 kJ = -28.4 % (TPOT p50 0.95-0.98 s, TTFT 95-141 s, E2E 387-410 s). B pC1q8pc/pa/pb 112.4/107.8/116.2 = 112.1 kJ = -51.9 % (CPU 25-27 kJ, TPOT 1.34-1.35 s, E2E 767-880 s, 70.7k Pixel calls/run). Ledger -8,773,632,000 B every launch. Fixes: preload contract (template), pinned owner admits MARGINAL_SYSTEM_COST_UNKNOWN, pinned latency miss -> recalibration, scheduler co-tenancy (share_key), full owner server = scheduler-owned wait, co_helper_capacity (decode co-helper cap), static control timeout = ffn_timeout_ms |
