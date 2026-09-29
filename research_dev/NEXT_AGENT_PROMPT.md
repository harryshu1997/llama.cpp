# Latest update: Pixel GPU server FFN PASS, 2026-09-22 21:09 UTC

User selected Pixel GPU for server assistance. Existing TCP server client and
Vulkan worker work without Tensor SDK. Tuned column-quantum512->4352 (34->4
blocks/layer), and locally changed ffn-split-worker.cpp to send its TCP header
and payload together. New isolated Pixel binary only; shared server and OP15
unchanged. Single-layer round trip103.110->35.008ms; six-layer numerical
qualification72calls/168rows PASS, max relative L2 0.000326951, repeats exact.

Actual Qwen server layers18-23,one slot,256prompt/64output,host/half/full/host:
all4 outputs exact,744 verified phone calls,372 per split, ack token2. Host
request energy4800.048/4329.807/4305.035/4892.714J. Versus control mean:
half10.659%,full11.170% saved; decode slower9.3%/18.1%. One sample per split,
not full trace, no25% claim; Pixel energy unmeasured. Half is the better
observed latency tradeoff. Automatic scheduler/multi-phone integration remains
deferred. Tensor TPU custom FFN is still blocked by compiler SDK; no need for
SDK on this tested GPU path.

All test processes exited normally; no owned forwards/workers/servers, boot
unchanged. Rig root /mnt/storage/s42-pixel10pro-server-20260922-v1; new Pixel
worker /data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1. Do not overwrite
existing run directories. Run5 prelaunch port-guard failure retained; corrected
LISTEN-only check and fresh26982 port run6 PASS. Read M3 report final GPU
section, configs, source snapshots, raw data and ONE_LAYER/SIX_LAYER_AUDIT.json
under physical/pixel10pro-server-1/. No runs remain queued.

# Latest update: Pixel FFN TPU path, 2026-09-22 20:38 UTC

User wants Pixel to assist server FFN as OP15 does. Existing protocol/column
split is suitable; a new LiteRT TPU worker and converted Qwen graph still
need implementation and qualification. Immediate prerequisite is Tensor SDK
compiler access (user has no installation). Official Tensor G5 path is AOT;
on-device JIT unavailable. Do not equate public runtime availability or the
4.666ms tiny add result with custom Qwen compilation/performance.
M3 README's Pixel FFN TPU path specifies layer18/tail512 columns/one row first,
then full17408 columns and all1..4 rows before server tests. No new code or
physical arm in this review. Multi-phone integration remains deferred.

# Latest update: Pixel Tensor TPU round trip, 2026-09-22 19:58 UTC

User authorized TPU round-trip testing and confirmed no Tensor SDK installed.
Public LiteRT2.2.0 + Google precompiled P25 add: PASS20 standalone calls and
180 TCP calls, whole1/1 graph delegated to Tensor TPU, all exact outputs.
This is128-element float32 ADD, 1024 input/512 output payload bytes, not Qwen
FFN. Real FFN TPU conversion/compilation remains blocked by missing Tensor SDK.

A/B/A60calls each, ten warmups: split response medianRT50.003ms;
one response write4.666ms (p905.203ms), split again50.070ms. Best arm mean
TPU invoke1.499 + buffers0.332 + outside-worker2.863 =4.694ms. A/B/A supports
reply framing as cause of delay; exact TCP/ADB buffering mechanism untraced.
Production FFN TCP worker splits response writes nearline2107; coalescing it
is a concrete next candidate for40-46ms overhead. NOT APPLIED/VERIFIED for FFN,
OP11/OP15 or larger payloads. No production/scheduler change in this task.

Rig /mnt/storage/s42-pixel10pro-tpu-20260922-v1; phone same basename under
/data/local/tmp, plus coalesced/ for second binary. All workers/forwards clean,
boot unchanged, no root/kernel/settings change. OP11 disconnected; OP15 and
Pixel present. Energy, Wi-Fi, FFN TPU, full tokens and integration unverified.
Evidence: reports/20260922-fast-path-M3/PIXEL10PRO_TPU_ASSESSMENT.json,
README.md, physical/pixel10pro-tpu-1/, software/pixel10pro-tpu/.
Earlier entries below are historical; TPU sample is now qualified, Qwen TPU is not.

# Latest update: Pixel 10 Pro qualification, 2026-09-22 19:26 UTC

User connected Pixel10Pro serial5A040DLCH004ES over USB. OP11 is no longer
connected; OP15 remains available. Pixel negotiated5000M USB, TensorG5,
Android16, unprivileged shell, PowerVR Vulkan. No wlan0 IPv4/route.
Existing Vulkan FFN worker was built/deployed in isolation. Qwen layer18,
17408 columns, quantum512, rows1/2/4: PASS12calls/28rows, max relative L2
0.000311882, exact repeats. Median worker59.600/95.484/188.701ms;
round-trip103.110/142.184/201.093ms. Latency improvement over matching OP11
NPU FAIL (20.673/63.005/77.991ms). Three warm samples per row count only.
No TPU, full-model token, energy, Wi-Fi or integration proof. Current energy
records unchanged. Google Tensor SDK supports G5, but our TPU path would need
separate LiteRT AOT conversion/backend work; not implemented or authorized
as a large change. Full M3 integration remains deferred.

Cleanup PASS: finite workers exited, owned forwards removed, boot unchanged.
Rig root /mnt/storage/s42-pixel10pro-qualification-20260922-v1.
Phone root /data/local/tmp/s42-pixel10pro-qualification-20260922-v1.
Evidence: reports/20260922-fast-path-M3/README.md,
PIXEL10PRO_QUALIFICATION_ASSESSMENT.json, physical/pixel10pro-vulkan-1/,
software/pixel10pro-vulkan/. Preserve the failed tiny caller attempt as evidence;
it omitted sha256: and failed CPU startup before a Pixel worker ran. Corrected
run2-tiny and real run3-layer18 PASS. No production GPU code was changed.

## OP11 NPU REPAIR QUALIFIED 2026-09-22 18:15 UTC

User requested an NPU fix and Wi-Fi assessment; full M3 integration remains deferred.
NPU PASS: normal user DMA works after fixing 16 DSP mapping slots versus the
host's 64-buffer limit. Current changes: htp-ctx.h capacity follows
HTP_OP_MAX_BUFS; main.c uses a 64-bit reuse mask, eviction on slot pressure
and fail-fast guards; hex-dma.c checks allocations before memset and frees
partial allocations; main.c rejects NULL queues. The initial HVX-copy
workaround failed at Bad VA 0 and was removed; evidence is retained.

One-layer 12-call/28-row and four-layer 48-call/112-row runs pass against CPU.
Four-layer max relative L2 0.000167056, repeats exact, identity/hash checks PASS.
Median NPU round trips 59.622/63.301/77.097 ms at 1/2/4 rows; worker intervals
15.803/18.223/31.440 ms. Both workers exited normally with status 0. HMX was
explicitly disabled. New isolated phone bin:
/data/local/tmp/s42-op11-v73-mmap-20260922-v1. DSP SHA-256:
0025a0e4c500e3f8f1d62dd08009651ce7681444a8f4df0d5e4dc65e345ea374.
Rig results: /mnt/storage/s42-op11-v73-mmap-20260922-v1.

Old bins, shared scheduler deployment and OP15 are unchanged. No test worker,
forward or job remains. Wi-Fi is unmeasured: OP11 wlan0 has no IPv4 or route;
the user was asked to connect to a reachable network. Do not claim Wi-Fi
savings or equate outside-worker time to wire time. Full-model tokens,
all-six-layer residency, layers 22-23 numerics, energy and integration remain
unverified. Report: scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/
README.md, NPU_QUALIFICATION_ASSESSMENT.json and physical/op11-v73-mmap-1/.
Older statements that NPU remains unusable are superseded for this tested scope.

## OP11 QUALIFICATION COMPLETE 2026-09-22 16:40 UTC

User chose qualification only. Do not implement full M3 integration without
further instruction. OP11 TCP/OpenCL functional PASS: layers 18-21, rows 1/2/4,
48 calls, 112 rows, max relative L2 3.274e-4 (limit 0.01), exact repeats,
identity rejection/response hashes PASS. Parent/shard verification PASS
(all 18 stored tensors, layers 18-23). Both workers exited 0; forward removed;
no worker or job remains active/queued from this qualification.

Performance blocks shared decode: OP11 68.652 / 4751.604 / 4749.101 ms per
1/2/4-row layer call, CPU 18.254 / 18.850 / 19.653 ms. Multi-row worker compute
is about 4.74 s. Investigate small-batch F16 OpenCL matmul dispatch before
scheduler integration; source switches to local-memory GEMM at N>1, but
exact deployed-kernel causality is not verified. No production/backend code,
phone binaries, OP15 transport or kernel changed. Full-model tokens, six-layer
OpenCL capacity, two-phone execution and energy saving remain unverified.
See scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/README.md and
QUALIFICATION_ASSESSMENT.json. Receipts copied from
/mnt/storage/s42-op11-qualification-20260922-v1/ to physical/op11-tcp-1/.
Older queued-status checkpoint below is superseded.

## OP11 SCOPE UPDATE 2026-09-22 16:28 UTC

User chose "Qualify OP11 only for now" after the full M3 design review.
Do not implement the proposed scheduler/runtime integration. See
scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/README.md.
Local protocol harness PASS (48 calls/worker, 112 rows, max relative L2 0).
Physical test is queued under the shared lock in
/mnt/storage/s42-op11-qualification-20260922-v1 (RUN.log and run1/RESULT.json
or FAILURE.json). Other session currently owns m4a8b/run-treatment-2.
Do not start a duplicate qualification, touch OP15, or kill any worker.

## FINAL MONITOR CHECKPOINT 2026-09-22 16:06 UTC

M4a8b ended FAIL at 15:58:47 UTC, exit 1: physical transition failed without
fallback. No RESULT.json or final energy saving. Shared Qwen calls PASS,
442 multi-row. Observed batch>=2 phone-window median 38.134 J/token including
assumed phone, n=19, three measurement-eligible. Coverage FAIL, 4/15 scheduled.
Only 19/24 outputs completed, all exact; 88133 failed and four later Qwen
cancelled. Prior tail-proof failure on 88121 did not recur. Second Qwen burst
attached helpers but 120 SERVER_POLICY_COHERENCE decisions selected zero phone
fraction. Underlying coverage and transition causes need diagnosis.
The user assigned this session read-only monitoring of their other agent's
retest. Do not infer authorization for an independent competing run/deployment.

Energy audit correction: best current 24-request arm with 24 exact outputs is
117.533 vs 132.253 kJ, 11.130% saving. Earlier 14-16% variants have output
mismatches; 28.237% m4a5 also has two. Best recent exact matched pair remains
Task 2, 19.208% host saving, 19/19 exact. Historical v16/v16c reached 25.827%
host / 25.118% including assumed phone with 24/24 exact, but used an older
deployment and baseline. It is not a current >25% result or the run7 pair.
Report evidence: physical/CURRENT_ENERGY_AUDIT.json,
HISTORICAL_V16_ENERGY_AUDIT.json, m4a8b-monitor/PARTIAL_ACCEPTANCE.json.
Lower ACTIVE checkpoints are historical. No OP11 work or new run started.

## LIVE ACCEPTANCE UPDATE 2026-09-22 15:52 UTC

M4a8b coverage is FAIL: 10 Qwen requests complete, only four phone-assisted,
so final coverage is at most 9/15. The entire second Qwen burst ran on the host.
All 15 complete outputs exactly match baseline and are COMPLETED in scheduler
snapshots. No FAILURE.json; the other session's run remains active. Continue
read-only monitoring for final energy/window measurements and all 24 outputs.
Do not sync source or start a competing run. The identity and tail fixes alone
are insufficient for Task 1. Final controller records are needed for diagnosis.

## ACTIVE RETEST CHECKPOINT 2026-09-22 15:40 UTC

The user authorized fixing and retesting and confirmed that another session
owns the same m4a8b retest. This session is assigned to monitor and assess it.
Do not launch another run, deploy source, or alter the other session's inputs.
Input root: /home/zhihao/s42-trace-v2a-m4a8b-20260922-inputs.
Watch run-treatment-1/run/RESULT.json and FAILURE.json plus CHAIN.log.
All 15 deployed hashes match the report's TASK1_RETEST_SOURCE_HASHES.json.

M4a8 preflight FAIL: coalesced-only removal left calibration route profiles
referencing an absent split-row executor. M4a8b retains split-row as SHADOW
and qualifies coalesced only; the catalog duplicate aliases are gone. Physical
preflight PASS. At 15:39:31 UTC: 322 one-row, 221 two-row, 221 three-row Qwen
USB calls; shared-forward check PASS. Four of five completed Qwen streams
assisted, including 88123. All five outputs exact. Scheduler marks 88121
COMPLETED, unlike its previous proof failure. Full acceptance remains pending.
The coalesced-only input patch must not be reused unchanged: its full campaign
profile references still need correction; the offline replay filtered them.

Read-only monitor script: report MONITOR_TASK1.py; output physical/m4a8b-monitor.
Baseline RESULT and raw streams saved at physical/task1-baseline-reference.
This session's staging script exited before source mutation due to existing
m4a8 inputs; its isolated rsync was stopped. Stage directory is incomplete and
must not be used for deployment. Older stopped/local-only statements below are
historical. No commits, pushes, native rebuilds, phone changes or manual kills.

## CURRENT CHECKPOINT 2026-09-22 15:16 UTC

Task 1 physical result remains STOPPED/FAIL under the two-runs/same-reason rule.
The user asked why it still fails; two code defects now have local corrections.
No new hardware run, deployment, rebuild, phone change, commit or push occurred.

CORRECTED DIAGNOSIS: the helper route names were misleading. In both m4a6-r2
and m4a7, the unsuffixed and explicit coalesced helper executor IDs BOTH declare
coalesced batching; every capability field except executor_id is identical.
This affects both Qwen CPU and GPU desktop parents. All m4a7 tickets for
88119, 88121 and 88123 contain the coalesced dormant-runtime hint. The READY
batch-plan filter accepted both aliases, and policy identity includes executor_id.
The earlier split-row/missing-hint/bypassed-filter explanation was incorrect.

Local fix in adapters/catalog_materialization.py: emit each declared batch mode
once with matching parameters and qualification; allow a coalesced-only list;
retain shared coordinator resources when split-row is absent. Report scripts
PREPARE_TASK1_INPUTS.py and CHECK_TASK1_ADMISSION.py now declare only qualified
coalesced Qwen helpers and reject multiple helper identities per parent.
Regenerate fresh derived inputs after this patch; do not reuse prior attempts.
PASS: regression reproduced before correction (2 failures, 1 error); 153 main
and 59 multi-session/offline-residency tests (212 total); pyflakes on seven files;
2/2 archived catalog replays reject the old aliases and accept one corrected
coalesced helper per parent. Each remaining capability matches the archived
qualified coalesced capability exactly. Source before/after, REPLAY.py,
REPLAY.json and VALIDATION.json: physical/task1-helper-identity-fix.

The separate tail-proof correction in adapters/llama_server_ops/proofs.py also
remains LOCAL ONLY. Qwen 88121 expected 476 layer rows but observed 493
(28 vs 29 for each of 17 layers). Native control at token 16 of 45 requires 29.
The correction derives lookahead from completed counters and the last policy
ack instead of always subtracting a row. Earlier validation: 180 tests PASS,
493/493 offline replay; prior-arm 66/66 and 697/697 unchanged. Source before/after
and evidence: physical/task1-proof-tail-fix. Both fixes need physical validation
together; do not weaken policy identities, proof checks or the dormant release rule.

M4a7 ended 14:19:45 UTC, exit 1. Shared Qwen USB calls PASS: 272 two-row and
485 one-row calls. Qwen 3/15 scheduled assisted, 10 Qwen streams completed.
All 15 complete streams match baseline; nine outputs incomplete/absent.
No RESULT.json, so final host energy, matched saving and full batch-window
median are not verified. Scheduler: 14 COMPLETED, 9 CANCELLED, 1 FAILED.
M4a6-r2 complete-arm FAIL: 138.748 kJ (+4.911%), Qwen 2/15, no multi-row USB,
111.602 J/token at batch>=2, 24/24 exact.

Report: scheduler/campaigns/burstgpt/reports/20260921-fast-path-trace-v2a/README.md.
M4a7 archive: physical/m4a7; rig inputs:
/home/zhihao/s42-trace-v2a-m4a7-20260922-inputs.
Deployed source remains physical/task1-controller-fixes-v2 (11 files), backup
/mnt/storage/s42-task1-before-m4a7-20260922. Fresh transport receipts passed
9/9 and actual coalesced admission passed at 40960 bytes / queue depth 4.
The 14:35 read-only end check found the lock free, both devices on ADB 5037,
and no owned desktop server/bridge. A cleanup transition failure without
fallback also exists in FAILURE.json; no manual kill/reset was attempted.
Task 2 already has a matched result. OP11 is out of scope and was not used.
Never commit, push, write commit messages or PR text. Lower checkpoints are history.

## Historical execution checkpoint before Task 1 was reopened (2026-09-22)

Task 1 is STOPPED/FAIL after two repeated strict correctness failures. Task 2 completed: 19.208% measured host saving and 19/19 exact outputs in one successful pair; the selected window has no outputs above 512. OP11 was not started under the stop condition. Before acting on the historical prompt below, read [the completed-work report](scheduler/campaigns/burstgpt/reports/20260921-fast-path-trace-v2a/README.md). No runs remain active; do not repeat Task 1 without renewed direction.

# Handoff prompt: phone-assisted decode energy work

Paste the block below to the next agent. It is also embedded in `FAST_PATH_UTILIZATION_PLAN.md`; keep the two in sync.

```
You are picking up the phone-assisted decode-energy work in `/home/myid/zs89458/Documents/llama.cpp-release`,
branch `wip/unified-scheduler-cleanup-20260813`. The tree is deliberately dirty. Never commit, never push,
never write commit messages or PR text (`AGENTS.md`). Ask the user before anything destructive; the desktop
rig and both phones are shared hardware.

READ FIRST, in this order, before writing code:
1. `research_dev/FAST_PATH_UTILIZATION_PLAN.md` in full (M0-M5, M4a, this prompt).
2. `research_dev/scheduler/campaigns/burstgpt/reports/20260921-fast-path-trace-v2a/README.md` in full: every
   measured arm of 2026-09-21 and the root-cause analysis.
3. `research_dev/talks.md`, the newest 20 entries (newest first, timestamped; keep adding to it).
4. `research_dev/scheduler/campaigns/burstgpt/M3_SECOND_PHONE_CHECKLIST.md` sections 0, 7, 8, only if you touch
   the second phone.

MEASURED FACTS. Do not re-derive these; build on them.
- Rig: desktop `zhihao@172.20.74.85` (RTX 4060 Ti, i9-12900K), OnePlus 15 `3C15AU002CL00000` on usb 2-2 and
  OnePlus 11 `832358d4` on USB 2.0, both on adb port 5037 only. Deploy `/mnt/storage/s42-trace-v2-20260921-prep`
  with `source/` (scheduler + C++) and `cuda-build/`.
- A Qwen decode window costs 40 J/token with its FFN on the phone and 72-79 J/token on the host. Beside a
  host-policy co-tenant it costs 128-138 J/token, because `server_slot::can_batch_with` refuses to batch slots
  with different `ffn_split_policy`: the assisted request runs in alternating batches, its latency doubles and
  its window is charged the co-tenant's energy, so the adaptive controller rejects the phone policy. That is why
  only 4-6 of 15 Qwen requests ever attach.
- 24-request trace, host energy against the 132.3 kJ / 1,381 s desktop baseline: as-designed 117.5 kJ (-11.1 %),
  contiguous Qwen sessions 111.2 (-15.9 %), Gemma release on 111.5 (-15.7 %), server release rule 113.3 (-14.3 %),
  cheap probe 125.3 (-5.3 %, worse), per-server coherence 113.9 (-13.8 %). Configuration and controller tuning
  are exhausted at 14-16 %.
- Realistic BurstGPT trace: baseline 340.0 kJ / 3,898 s against treatment 281.2 kJ / 3,555 s, -17.3 %, with 61 %
  of Qwen tokens phone-assisted. Long decodes pay more because attach cost amortizes and the mix shifts to decode.
- The real BurstGPT conversation log has median output 190 tokens, p90 671, p99 1,068. Requests above 512 output
  tokens are 18.2 % of requests and carry 49.1 % of all output tokens, and our builder caps output at 512.

UNCOMMITTED WORK ALREADY IN THE TREE. Keep it, extend it, or remove it with a stated reason.
- `tools/server/server-context.cpp`: `apply_dormant_host_share` releases the host FFN only when every processing
  slot decodes under the same phone policy; under a policy mix it stays local and counts
  `ffn_dormant_release_skipped_mixed`.
- `ggml/src/ggml-hexagon/htp/htp-ops.h`: `HTP_OP_MAX_BUFS` 16 -> 64 (host-side batch limit; the DSP reads the
  buffer list by count).
- `research_dev/scheduler/_internal/adaptive_decode_ops/coherence.py` and `tests/test_adaptive_coherence.py`:
  a session exploiting the baseline adopts a co-tenant's phone policy. It fired 8 times in a run and never
  produced a multi-token phone call, so it is necessary but not sufficient.
- Campaign fields `adaptive_maximum_probe_attempts_per_context` and `adaptive_decode_overrides` (any
  `AdaptiveDecodeConfig` field), and learning-demand decay in `_unified/phone_residency_ops/demand.py`.

TASK 1, highest value: make co-tenants share one forward pass.
Decide the FFN split policy per (model artifact, desktop parent placement, layout generation) instead of per
request, so every decode slot of that model on that server runs the same policy from its first token; attach the
helper at admission for a server that is already assisted, instead of after the first recorded window; confirm
the server batches policy-equal slots and sends all their rows in one phone call; and compare only windows of
equal batch composition, or normalize, so a probe is never rejected against a baseline measured at another
concurrency. Files: `_internal/adaptive_decode*.py`, `_unified/adaptive_decode_control.py`,
`_unified/helper_preparation*.py`, `_unified/automated_requests_ops/`, and `tools/server/server-context.cpp`
around `update_slots` and `apply_ffn_split_ubatch_context`. Keep the dormant release rule above.
PASS requires all of: `S41SERVERFFNUSB` lines with `tokens=` greater than 1 on the busy Qwen server; phone
windows at `active_batch` 2 or more with median at most 55 J/token; at least 12 of 15 Qwen requests
phone-assisted; host saving at least 22 % against the 132.3 kJ baseline; all 24 outputs token-identical to the
baseline arm. Stop and report if two runs fail for the same reason.

TASK 2, independent of task 1: a long-decode trace, which is the design's honest best case.
Our 512-token output cap truncates half the real decode work. Rebuild the window with the real distribution:

    cd /mnt/storage/s42-trace-v2-20260921-prep/source
    LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64 \
    python3 -m research_dev.scheduler.campaigns.burstgpt.build_realistic_trace \
      --burstgpt-csv /mnt/storage/burstgpt-source/burstgpt_3.csv \
      --output-dir /mnt/storage/burstgpt-source/longdecode_v1 --trace-name burstgpt_longdecode_v1 \
      --codec /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin/llama-token-codec \
      --library-dir /mnt/storage/s42-trace-v2-20260921-prep/cuda-build/bin \
      --qwen-tokenizer-model /home/zhihao/models/Qwen3-14B-Q4_K_M.gguf \
      --gemma-tokenizer-model /home/zhihao/models/gemma-4-12B-it-Q4_0-op15-exact.gguf \
      --llama-tokenizer-model /home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf \
      --prompt-cap 2048 --output-cap 1100 --duration-s 1800 --min-requests 12 --max-requests 20 \
      --execution-artifact qwen3-14b-q4km-dequant-f16=/home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf=text_decoder_f16_proxy \
      --execution-artifact gemma-4-12b-q40-dequant-f16=/home/zhihao/models/gemma-4-12B-Q40-dequant-f16.gguf=text_decoder_f16_proxy \
      --execution-artifact llama-3.2-1b-instruct-q4_0=/home/zhihao/models/Llama-3.2-1B-Instruct-Q4_0.gguf

1,100 is the log's p99, and 2,048 plus 1,100 stays inside the 4,096-token per-sequence context of both desktop
parents, so nothing needs re-qualifying. Keep `--min-output` at its default for the headline trace: that is the
untruncated real window. A second trace with `--min-output 256` is allowed only as an explicitly labelled
best case, never as the headline. Derive inputs with
`campaigns/burstgpt/prepare_trace_inputs_v2.py`, then run treatment and desktop-baseline arms and report the
matched saving. Expect the run to take longer than the 3,555 s realistic trace; keep `--min-requests` low.

TASK 3, second phone, only after tasks 1 and 2 have run: OnePlus 11 integration. Its v73 NPU crashes in the
user-DMA `dmpoll` inside `hvx_mv_2d` with Bad VA 0, and `dma_queue_create` returning NULL is silently tolerated
in `htp/main.c` around line 428. Fix candidates and the working OpenCL fallback are in the checklist section 8.
Binaries, the OpenMP runtime and the Qwen shard for layers 18-23 are already on the phone. The rig manifest and
about 300 single-phone assumptions across 55 files are the real engineering cost; the brief is checklist
section 7.

RIG PROCEDURE.
- One run at a time under `flock -w 900 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock`.
- Sync scheduler changes with `rsync -rlptc --exclude __pycache__ --exclude .venv --exclude campaigns/burstgpt/reports
  research_dev/scheduler/ zhihao@172.20.74.85:/mnt/storage/s42-trace-v2-20260921-prep/source/research_dev/scheduler/`.
- C++ changes: rsync the files, then
  `/mnt/storage/s21_deps/cmake-4.2.3-linux-x86_64/bin/cmake --build /mnt/storage/s42-trace-v2-20260921-prep/cuda-build
  --target llama-server` with `LD_LIBRARY_PATH=/mnt/storage/s21_deps/cuda-13.2.1/lib:/mnt/storage/s21_deps/cuda-13.2.1/lib64`.
  The build must keep `-DS41_SERVER_FFN_SPLIT=ON`; verify with
  `strings cuda-build/bin/libllama-server-impl.so | grep -c S41SERVERFFN`, which must be 12 or more, or the server
  silently ignores every `S41_SERVER_FFN_*` variable. After any rebuild, `rm -rf phone-identities`, move the old
  `TRANSPORT_QUALIFICATION_IDENTITY.json` aside, run `bash /mnt/storage/s42-trace-v2-20260921-prep/MATERIALIZE_TRANSPORT.sh`,
  and copy the new identity into every inputs dir you use.
- New inputs per attempt: copy `campaign.json`, `rig.json`, `models.json`, `evidence.json` and the identity from
  `/home/zhihao/s42-trace-v2a-m4a3-20260921-inputs` (both models release, probe attempts 4), rewrite the paths and
  `campaign_id`. Launch from `/mnt/storage/s42-trace-v2-20260921-prep/source` with `LANG=C.UTF-8` and
  `S42_UNIFIED_REPO_ROOT=/mnt/storage/s42-trace-v2-20260921-prep/source`:
  `python3 research_dev/scheduler/campaigns/burstgpt/launch.py <campaign.json> <new dir> --preflight-only`, then the
  same without the flag. Output directories must not already exist.
- Compare with
  `python3 /home/zhihao/s42-trace-v2a-20260921-inputs/compare_trace_energy.py --run baseline=/home/zhihao/s42-trace-v2a-baseline-20260921-inputs/run-baseline-1/run/RESULT.json --run yours=<RESULT.json>`.
- Tests: `PYTHONPATH=.:research_dev/scheduler/tests python3 -m unittest <module>` from the repo root, plus pyflakes
  on every changed file.

HAZARDS THAT HAVE ALREADY COST RUNS.
- Never `pkill -f` a pattern that your own ssh or adb command line contains; it kills your shell. Use bracketed
  patterns in their own `adb shell` call, or kill by PID.
- Wait on `RESULT.json` and `FAILURE.json` files, not on process patterns, and never put `set -e` before the run
  line in a chain script: a failed launcher then skips its completion marker and stalls every follower.
- Regenerate every derived inputs dir after changing an input patcher. Cache-policy flags on a model without
  `ffn_host_share_release == 1` are refused at parse time and used to abort a whole run through route quarantine.
- `fixed_phone_residency` is an evaluation-only preload that arrived work ignores; `phone_resident_limit_bytes` is
  forwarded but never consumed. Neither can pin or evict a model from the phone. Use the shard index for that.
- Never force-kill an in-flight phone worker. Phone kernel changes only as a user-authorized temporary
  `fastboot boot`. Never start an adb server on a port other than 5037.

REPORTING. Report every milestone as PASS or FAIL with the measured numbers, never as a claim, and say plainly
what you did not verify. Append a timestamped entry to `research_dev/talks.md` newest-first for each result,
extend the report README with the new arms, and keep the plan status line current.
```
