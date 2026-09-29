# Handoff: scheduler adaptivity / phone offload (paused 2026-09-24 ~17:20 UTC)

Copy the block below as the first message to a fresh agent. Everything it references is in the repo
(`research_dev/talks.md` newest-first, `research_dev/scheduler/campaigns/burstgpt/reports/2026092{3,4}-*`)
or on the desktop rig.

---

You are picking up a paused research effort: **energy-aware LLM serving where a desktop
(RTX 4060 Ti + CPU) offloads dense FFN layers of Qwen3-14B / Gemma-4-12B (F16) to a OnePlus 15
phone (Hexagon NPU over FunctionFS DMA-BUF USB), with a Pixel 10 Pro as a second phone in
preparation.** Read `AGENTS.md` first, then `research_dev/talks.md` (newest-first, read the
2026-09-23/24 entries) and this file. Ask the user before any hardware run.

## Rules (non-negotiable, from the user)
- Never commit or push; no AI-written commit messages. The whole tree is UNCOMMITTED (~340 files):
  ask the user for a review/commit checkpoint.
- Desktop rig `ssh zhihao@172.20.74.85` (zsh: wrap globs in `bash -c`). Desktop NVMe nearly full:
  write large files to `/mnt/storage`.
- OP15 = adb serial `3C15AU002CL00000` via `adb -P 5037` ONLY. Never start another adb server, never
  kill one. Never touch the Pixel `5A040DLCH004ES`, its dirs (`/mnt/storage/s42-pixel10pro-*`,
  `reports/20260922-fast-path-M3/physical`) or its phone-side code: another (external) agent owns
  the Pixel execution path and is building the Pixel USB transfer (the Pixel is now rooted).
- Every rig action under the lock: `flock -w 7200 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock <cmd>`,
  held across sync + preflight + runs. Never break it. Never force-kill an in-flight phone worker.
  Never `pkill -f` a pattern your own command line matches; kill only PIDs you started; run
  `adb shell` with stdin closed (`< /dev/null`) or from pushed scripts (an open heredoc once
  hung a worker for 1.5 h holding the lock).
- Wait on `RESULT.json` / `FAILURE.json` files, not process patterns. Derive inputs with
  `campaigns/burstgpt/prepare_trace_inputs_v2.py`; never hand-edit generated inputs. Re-materialize
  the transport identity after any server rebuild (`MATERIALIZE_TRANSPORT*.sh` in the deploy).
- NO battery/charging guard (user rule: "we need max performance; tell me and I'll replug").
  Before/after long OP15 runs read `dumpsys battery` and `/sys/class/oplus_chg/battery/battery_notify_code`
  (via `su -c`): notify 512 = charger latched off (`chg_over_time`) -> STOP and tell the user. The
  desktop port is a 2.5 W SDP port; the phone's own charge limit holds it at 80 %.
- Develop scheduler changes in an isolated copy (`cp -a research_dev/scheduler` to a scratch root,
  run tests there, deliver a diff that `git apply --check`s against main with stdin closed); merge
  into main only when no rig chain will re-sync mid-A/B; keep pre-merge backups. Rig A/Bs run only
  from the merged main tree, synced under the lock to
  `/mnt/storage/s42-trace-v2-20260921-prep/source/research_dev/scheduler/`.
- Keep `research_dev/talks.md` updated (newest-first, UTC timestamp, tables) on every result.

## State of the code (main tree, all opt-in unless noted; full suite 136 files / 1,954 tests pass)
1. Scheduler crash fixes (4) + adaptive evidence fixes F1a/F1b/F2/F3/F4 (always on):
   `reports/20260924-phone-rejection-diagnosis/`.
2. #1 coherent phone policy per server (per-batch-size verdicts) + coalesced multi-row phone calls,
   both models (`--adaptive-decode-overrides-json '{"server_policy_coherence": true}'` +
   `--qualify-phone-batch-plan hot=coalesced-batch --qualify-phone-batch-plan cold=coalesced-batch`;
   identity `TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json`): `reports/20260924-coherent-policy-coalesced/`,
   `reports/20260924-gemma-coalesced-qualification/`. Server-probe thin-evidence fix merged
   (`reports/20260924-server-probe-fix/`; its C++ counter fix needs a server rebuild — deferred).
3. #4 phone re-provisioning (`"phone_resident_model_reprovisioning": {}`): `reports/20260924-phone-reprovision/`.
4. #2/#5 dispatcher (`"dispatch_policy": {"work_conserving_admission": true, "model_affinity": true}`):
   `reports/20260924-dispatch-policy/` (replay est. dev_v2 makespan 1,147 -> 663 s; latent
   stale-receipt replan bug fixed only under the policy — consider making it unconditional).
5. Two-phone groundwork (server multi-helper `S41_SERVER_FFN_HELPERS=N`, scheduler contracts,
   gaps 1-3 closed; a real two-phone run is fail-closed until Pixel receipts + cost evidence exist):
   `reports/20260924-two-phone-readiness/` (README, GAPS_README).
6. Trace builder rules (`--long-tail-*`, `--max-output-tokens`, `--min-same-model-overlaps`,
   `--min-requests-per-model`); repaired `tests/run_all.py`.
7. Parked: NPU+GPU dual-engine phone worker (1.25x in bench, ~1.0x over real USB;
   `reports/20260923-dual-engine-*`); phone-weight quantization (user: more phones instead).

## Traces (desktop `/mnt/storage/burstgpt-source/`)
- `longtail_v1` (31 req, ~80 min): evaluation trace. Measured: desktop 523.7 kJ; phone plain
  (before today's changes) 431.0 kJ (-17.7 %, 24/31 identical).
- `longtail_dev_v1` (6 req, no concurrency, ~15 min) and `longtail_dev_v2` (9 req, pairs form,
  ~17 min): development traces.

## Measured on dev_v2 (single runs; repeat noise 3.1 % energy / 13 % duration)
| arm | dur | host kJ | vs desktop | identical |
| desktop only (legacy dispatcher) | 1,154 s | 96.75 | | |
| phone plain + evidence fixes | 1,035 s | 76.31 | -21.1 % | 6/9 |
| coherentEF (#1 + fixes) | 925 / 1,043 s | 71.86 / 69.64 | -25.7 / -28.0 % | 9/9, 9/9 |
| coherentRP (#1 + #4) | 898 s | 53.69 | **-44.5 %** | 6/9 (near-ties) |
| desktop only + dispatch_policy (#2/#5) | 865 s | 82.3 | -14.9 % | 7/9 |
| **allon** (#1 + #4 + #2/#5 + all fixes) | 812 s | **45.7** | **-52.8 %** (-44.5 % vs desktop+dispatcher) | 7/9 |
Dispatcher observations (both arms with the policy): 4 same-model overlapping executions (legacy: 1),
loads 5 -> 3, `affinity_displacements` = 0 (the reorder came from publication replans / early
capacity promotions; Gemma 005 still ran alone 317-625 s while Qwen 006 waited until 715 s — check
why affinity never displaced; replay had predicted 2 loads / 663 s). Analysis of these two arms was
NOT written up (the agent was cut off): run dirs `run-dev2baseDP-1`, `run-dev2allon-1`, status
`/home/zhihao/s42-trace-longtaildev2-ALLON-20260924-STATUS.jsonl`, tooling
`/home/zhihao/s42-allon-20260924-rig/`. Gemma 26-layer shards WERE generated and pushed by that chain
(`/mnt/storage/s42-ffn-shards-20260924-v3-gemma26/gemma26/FFN_SHARDS.json`, phone dir
`/data/local/tmp/s43-ffn-shards-20260924-v3/gemma26`) — no arm uses them yet.
Run dirs: `/home/zhihao/s42-trace-longtaildev2-*-20260924-inputs/run-*/run`; tooling in
`/home/zhihao/s42-{coherent,reprovision}-20260924-rig/` and the report dirs
(`derive_inputs_*.sh`, `run_arms_*.py`, `analyze_*.py`, `compare_trace_energy.py`,
`analyze_longdecode_pair.py`).

## Coordination with the Pixel agent (read before touching the rig) — updated 2026-09-24 19:15 UTC
- The Pixel agent (external session; report `reports/20260924-pixel-second-phone/`, entries in
  talks.md 17:03-19:01 UTC) REBUILT the shared deploy's server library at 17:59 UTC
  (`/mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/libllama-server-impl.so` sha a87e7772…,
  old 521f679f…) to include the merged multi-helper code, and re-materialized identities. Consequence:
  `TRANSPORT_QUALIFICATION_IDENTITY_COALESCED_BOTH.json` (ours, 14:26 UTC) is STALE (binds the old
  library). Use `TRANSPORT_QUALIFICATION_IDENTITY_PIXEL_STAGEA_COALESCED_BOTH.json`
  (id `s42-pixel-stagea-coalesced-both-20260924`, binds a87e7772…) for any new arm, and verify its
  receipts cover the coalesced payloads (Qwen 40,960 B, Gemma 15,360 B) with the admission checker
  before running. `TRANSPORT_QUALIFICATION_IDENTITY.json` (base, split-row) was also refreshed.
- Pixel path chosen by that agent: packed-CPU worker over ADB TCP (rooted, 6 pinned threads),
  lifecycle = automatic idle TERM, connected stop refused (this settles the "how to stop the Pixel
  worker" decision). Rooted AOA USB transport measured but idle-robustness FAIL; Pixel FunctionFS
  NOT done -> our two-FFS host work stays parked. Two-phone mechanism PASS (OP15 layers 0-17 +
  Pixel 18-23, identical outputs); Pixel helps at B1 but at B4 costs +7.4 % energy / +49 % latency
  vs OP15 alone. Its rooted-lifecycle patch (`ROOTED_WORKER_LIFECYCLE.diff`) lives only in its
  `stagea-source` copy, NOT in main — coordinate before merging anything touching
  adapters/{co_helper_lifecycle,energy,heterogeneous_rig*,http_backend,phone_tcp_session}.py.
- It holds the rig lock for long stretches (two-phone v3 preflight/campaign iterations). Queue
  behind it with `flock -w`; never break the lock; do not touch `/mnt/storage/s42-two-phone-pixel-*`.
- It already wrote our all-on write-up: `reports/20260924-coherent-policy-coalesced/RIG_RESULTS_ALLON.md`.

## Unfinished work preserved at pause (all three agents were cut off by a usage limit)
- Rig chain finished (see table above) but its analysis/write-up was not done: write
  `reports/20260924-coherent-policy-coalesced/RIG_RESULTS_ALLON.md` from the run dirs (energy,
  exactness 7/9, loads/switches, pairs, dispatch_policy stats, re-provisioning timeline, server-probe
  reasons SERVER_PAIR_INCONCLUSIVE/REFERENCE_BASELINE).
- #4 follow-up fixes (coherence group keyed by layout generation -> every swap wipes verdicts;
  per-token re-evaluation storm -> 1,202 REPROVISION_RETAINED events, RESULT.json 41 MB): only a
  PARTIAL diff exists — `reports/20260924-phone-reprovision/followup-fixes/REPROVISION_FIXES_PARTIAL.diff`
  (670 lines, base->root of an isolated copy; unfinished, untested). Treat as a starting point.
- Two-FunctionFS-phone host support (USB client selects gadget by serial / bus path instead of first
  18d1:2d00 VID:PID; fake-libusb unit test; per-device identity; `functionfs-dmabuf` helper transport in
  rig.json; phone_ffs_session adapter): PARTIAL diff
  `reports/20260924-two-phone-readiness/two-ffs/TWO_FFS_PARTIAL.diff` (2,604 lines; its last note: server
  build and fake-libusb test finished, whitelist contexts for the two new selector keys still to do;
  unmerged, needs the scheduler tests + `git apply --check`). Do NOT merge blindly: review, finish, test.


## Update 2026-09-24 ~21:40 UTC — two-phone activation WORKS; Pixel not yet beneficial
- Merged into main: the Pixel agent's Stage A v7 integration (rebased) + ACTIVATION_FIXES (OP15
  shard-replacement check narrowed to OP15's own layers via `primary_phone_ffn_contract`; split
  routes kept as mandatory frontier choices in `route_generation/costing_rough.py`). Full suite 1,999
  tests pass. Report `reports/20260924-pixel-integration-2/`.
- Measured dev_v2 (matched, same tree): desktop+dispatcher 682 s / 68.0 kJ; OP15 all-on 617-652 s /
  39.8 kJ mean (-41.5 %); OP15+Pixel all-on 682-806 s / 41.3 kJ (-39.2 %) -> the Pixel costs
  +3-4 % energy and time vs OP15 alone: its layers take 38 ms (B1) / 82 ms (B4) vs 10-13 ms on the
  OP15 and layers run in sequence; Stage A puts it on every Qwen phone route. dev_v1 transition
  test: both phones active, 1,998 Pixel calls, 0 preparation failures, -49.9 % vs desktop+dispatcher.
- Today's fixes measured: desktop+dispatcher 865 -> 682 s / 82.3 -> 68.0 kJ; OP15 all-on 812 -> ~635 s.
- Open (user decisions): per-device policies (Pixel only at B1) vs faster Pixel worker vs different
  Pixel layers; run the longtail_v1 pair (desktop+dispatcher vs OP15 all-on, ~3 h); commit checkpoint.
- Left unfixed: rough cost charges prefill payloads for decode-only routes (worked around); no backoff
  for repeatedly failing preparations.

## Update 2026-09-25 ~01:10 UTC - per-device phone policies MERGED (code only)
- Two-phone envelopes now offer every device set as its own adaptive policy: OP15 / OP15+Pixel /
  Pixel x the column grid. The coherent server probe decides per batch composition:
  - OP15 alone first;
  - OP15+Pixel must beat OP15 by the unchanged bounds, otherwise it is dropped for that composition;
  - a failure drops the failed set and its supersets everywhere, with fallback OP15, then Pixel alone,
    then host.
- There is no server change: the union-mask control already switches a helper off. Single-phone
  behaviour is byte-identical (digest guards). Main suite: 141 modules / 2,021 tests pass.
- Report and arms to run later: `reports/20260924-per-device-policies/README.md` (section 6).
- The deploy is NOT synced; no hardware was run (user: Pixel CPU+GPU worker first, `reports/20260925-pixel-cpu-gpu/`).
- Stage A's "every Qwen phone route uses both phones" no longer holds in main. Pre-merge backups are in
  `$SCRATCH/perdev/premerge-backup/`.


## Update 2026-09-25 ~09:30 UTC — half-size evaluation trace built; fixes merged; waiting on OP15 replug
- `longtail_eval_v2` (/mnt/storage/burstgpt-source/longtail_eval_v2): first 30-min window with 14-18
  requests, long-tail tol 0.05, <= 4,000 capped output tokens, >= 4 requests per model -> 14 requests
  (7 Qwen, 6 Gemma, 1 Llama), 3 long (max 614), 3,604 output tokens (44 % of longtail_v1), real arrivals
  over 1,675 s; long tail 21.4 % / 49.5 % (log 18.6 % / 49.3 %). Purpose: ~25-35 min per arm under
  the new dispatcher so every arm can be run TWICE; keep longtail_v1 for a final confirmation.
- Merged since the last update: replan-path crash fix (reports/20260925-replan-reprojection-fix/),
  coordinator fail-fast (default; `--lifecycle-failure-mode drain` = old behaviour); main = 141
  modules / 2,031 tests. Deploy source synced to main at ~09:30 UTC (rig idle, verified).
- longtail_v1 so far: all-desktop legacy 536.3 kJ / 4,984 s; desktop+dispatcher 393.0 kJ / 3,381 s
  (-26.7 %); OP15 all-on crashed (now fixed); OP15+Pixel not run. dev_v2 gate with the fixed Pixel
  worker PASSED; run-to-run spread 21 % (which requests got assisted) -> robustness items:
  re-provisioning "replacement source is not ready" waits; Gemma starting before OP15 sessions ready.
- Phone power (1 Hz sysfs): OP15 4-6 W assisting (battery-fed above the 2.5 W port -> latch 512),
  Pixel 1.3-2.0 W (assumed 4.5 W overstates 2x).
- BLOCKED on: OP15 replug (notify 512). Then: eval_v2 arms x2 each (all-desktop legacy via
  `--drop-dispatch-policy`, desktop+dispatcher, OP15 all-on, OP15+Pixel per-device) or the remaining
  longtail_v1 phone arms — user to choose.

## Next steps, in order
0. (Resumed 2026-09-24 19:10 UTC) #4 follow-up fixes are MERGED (20:02 UTC; layout-identity
   coherence keys + boundary re-evaluation gate, `boundary_reevaluation_interval_us` default 10 s;
   `reports/20260924-phone-reprovision/followup-fixes/`). The affinity follow-up is also MERGED (affinity
   now fires when a queued request's model load is published; `reports/20260924-dispatch-policy/affinity-followup/`;
   replay: dev_v2 desktop+dispatcher 863 -> 624 s, 5 -> 3 loads; longtail neutral). Nothing is left in
   isolated copies except the parked two-FFS partial diff. RIG COORDINATION: the Pixel agent's chains use `flock -w 900` — if we
   hold the lock for a 20-40 min arm between its steps, its step times out. Agree a rig window with the
   user (e.g. after its two-phone campaign) before queueing our arms.
1. (Done by the Pixel agent) allon/baseDP write-up. Then re-run the full suite after merges
   (`/usr/bin/python3 research_dev/scheduler/tests/run_all.py`, expect ~1,954+ tests).
2. dev_v2 allon A/B including the #4 follow-ups (and optionally the Gemma 26-layer shards via
   `--gemma-ffn-shards` pointing at the new FFN_SHARDS.json); then ONE full `longtail_v1` pair
   (desktop-only WITH the new dispatcher vs everything-on) for the headline number; report
   token-identity (near-tie flips are expected: 7/9 on dev_v2, differing at fixed positions) and the
   phone-energy caveat (phone power is an assumed 4.5 W model, unmeasured).
2b. Investigate why `affinity_displacements` stayed 0 in both dispatch arms (Gemma 005 still ran
   alone while Qwen 006 waited) — the replay predicted 2 loads / 663 s, the run had 3 / 812-865 s.
3. Open decisions the user has NOT made yet: F6 (continuous-sqrt confidence band, loosens bounds —
   recommended OFF); Pixel worker stop method (moot if the Pixel uses the FunctionFS path);
   scheduling the two-phone smoke test (needs the Pixel agent's transport, receipts, VID:PID/serial,
   ffs root, transport generation, session-script contract); review/commit checkpoint.
4. Known smaller items: make the stale-receipt replan fix unconditional; C++ client `prefill_calls`
   classification (needs rebuild + identity); Gemma 26-layer shards (`native/ffn_shard_gguf.py`
   command in `reports/20260924-phone-reprovision/README.md`, out-dir under `/mnt/storage`).

## Why the saving was capped before today (for context)
Run-5 accounting (`reports/20260923-offload-accounting/`): idle floor 27.6 W = 31 % of energy, GPU
board 36 % (untouched by FFN offload), phone RAM-bound at 12 Qwen + 8 Gemma layers, 27 % of tokens
lost to mixed-policy pairs and thin-evidence drops, one-model-at-a-time dispatcher (16 % of time in
loads). Today's changes address each: #1 (pairs), F1-F4 (evidence), #4 (RAM use), #2/#5 (loads/idle).
