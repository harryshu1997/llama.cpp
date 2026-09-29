# Dual-engine (NPU+GPU, zero-copy) phone FFN worker: campaign integration + dev-trace smoke pair, 2026-09-23/24

Integrates the env-gated dual-engine mode of `llama-ffn-split-worker` (report
[20260923-dual-engine-ffn-worker](../20260923-dual-engine-ffn-worker/README.md)) into a runnable OP15 campaign
deployment with its own transport identity, then runs the `burstgpt_longtail_dev_v1` pair (desktop baseline,
then energy-aware treatment on the dual worker). Server side (cuda-build, source) unchanged. Nothing committed.

**Summary.**
- **Integration works.** New identity, new phone dirs, zero-copy confirmed on the phone (no NPU copy). Both arms PASS, and outputs are 6/6 token-identical to the desktop baseline. The dual worker served 3252 real calls over the FunctionFS DMA-BUF transport, all tokens=1.
- **Energy.** Host -6.4 % vs desktop-only (76.7 -> 71.8 kJ), duration +0.9 %.
- **The bench's 1.21-1.24x per-layer speedup does NOT carry over to the campaign.**
  - Qwen full-width: 9.607 ms vs run-5's 9.72 ms (1.01x). Gemma full-width: 7.233 ms vs 6.59 ms (0.91x). Partial widths: 0.54-0.86x.
  - Cause 1: the first call after each ~0.4 s between-token idle pays +6 ms on the GPU leg (GPU wake-up).
  - Cause 2: within a burst, total DRAM bandwidth rises only ~3 %, versus +24 % in the continuous bench.
  - Measured phone power is about 0.35 W higher while active.
  - As deployed, dual is no better than NPU-only. Details in section 4.

## 1. What the transport identity allows, and what was done

`research-scheduler-transport-qualification-identity-v1` (adapters/transport_profiles.py) binds two different phone
stacks in `software_identity`:

| field | meaning | production value | DUAL value |
|---|---|---|---|
| `phone_worker_sha256` | the FFN worker the campaign runs | `43adcb75...` (s42-ffn-shards-20260904-v1-bin) | **`be638f7e...`** (s43 dual) |
| `phone_session_sha256` | the session script the campaign runs | `1118eb90...` (s42-hal-runtime-probe-20260906-v4) | **`db82632c...`** (s43 dual) |
| `qualification_phone_worker_sha256` | phone binary the USB receipts were MEASURED with (ffs_dmabuf_phone.android) | `e2c66e6b...` | same |
| `qualification_phone_session_sha256` | session script of that measurement (phone_gadget_session.sh) | `be192c4a...` | same |
| `phone_resident_workers_sha256` / `phone_resident_router_sha256` | optional pair (both or neither) | `0e7736c0...` / `25d27ff1...` | same (unchanged binaries) |
| host binary + 7 host dependencies, qualification binary, transport client source, receipts, hardware (kernel, boot image, USB controller/sysfs/serial, FunctionFS VID:PID) | | | all identical |

So the CURRENT worker/session are allowed to differ from the QUALIFIED ones by design: the production identity
already pairs worker `43adcb75` with qualification worker `e2c66e6b`. The receipts are raw FunctionFS DMA-BUF
transfer measurements (`s41_ffs_dmabuf_transport_v2`, devmem, async, queue depth, payload sizes); the dual mode
changes neither the wire format (no protocol/HELLO/CLI change, same `weight_hash`) nor payload sizes, so **no
re-qualification is required** and none was run.

What the runtime checks (adapters/phone_session_ops/transport.py, phone_session_contracts/configuration.py):
static - host server + every host dependency sha256, remote `session_script` == `phone_session_sha256`, every remote
`worker`/`worker:<artifact>` == `phone_worker_sha256`, boot image + serial; if resident binaries are in play both must
match. Live - ticket's `qualification_identity_sha256` == identity digest, kernel release, USB controller, sysfs
device, FunctionFS identity, negotiated speed >= 5000, allocator, transport generation. Configuration - host
dependency set, CPU affinity equal to the identity's (absent in both). Nothing was bypassed or weakened.

Not covered by the identity schema: the worker's shared libraries (libggml*.so, libggml-htp-v81.so, libomp.so).
They are recorded beside the identity in `data/identity/DUAL_PHONE_STACK.json` and asserted by the materialize
script against the built bundle; the scheduler itself would not notice a library swap.

New materialize script `/mnt/storage/s42-trace-v2-20260921-prep/MATERIALIZE_TRANSPORT_DUAL.sh` (copy:
`data/identity/`), under the rig lock:
1. pulls the PRODUCTION phone binaries and aborts unless they still match the production identity;
2. pulls the DUAL worker + session + all 7 worker libraries and aborts unless they match the workstation bundle
   (`EXPECTED_SHA256.json`); asserts resident workers/router equal production and the dual worker differs from it;
3. asserts qualification binary, receipt set, host server, 7 host dependencies and transport client source are
   byte-identical to the production identity (server side unchanged);
4. runs `python3 -m research_dev.scheduler.adapters.materialize_transport_qualification` with the new current
   digests and the unchanged qualification digests -> `TRANSPORT_QUALIFICATION_IDENTITY_DUAL.json`
   (identity digest `sha256:d332137d0af76cb3417afb6345b69cd42a9a2e57d11b864ea7a30821560d1bf6`, file sha256
   `4188c988...`). The production `TRANSPORT_QUALIFICATION_IDENTITY.json` is untouched (sha256 `82617a5c...`).

Diff vs production identity: `identity_id`, `phone_session_sha256`, `phone_worker_sha256`; 25 of 28 fields identical.

## 2. Phone deployment (new dirs only; no s41-/s42- dir modified)

| path | content | sha256 |
|---|---|---|
| `/data/local/tmp/s43-dual-ffn-worker-20260923-v1-bin/llama-ffn-split-worker` | dual worker (+ warm-up, below) | `be638f7e1f74963dfb4bba1083021d8caeef8b0241a6800e403b88b87d1da32c` |
| `.../libggml.so` | | `21b4f1f7e3cd095d070bf7b842178219df504c357242bed3e55e679ab653bb93` |
| `.../libggml-base.so` | | `a537bbd94ff4f27501a1e1f78abd16c3232dae81425a52f03803e6419e8b0182` |
| `.../libggml-cpu.so` | needs libomp.so | `52118e6e5f02c4842eddebcef09fda42ccedd2f3fb06400059cafd4eb30d2e03` |
| `.../libggml-hexagon.so` | | `fa5ed71dd0d1ab071b7aff63aa1b57c8de023986780cb22fc1b1d68bc706b2e6` |
| `.../libggml-opencl.so` | | `aaa3589f0a8c2753b8ce721edfab7ed06d68626a66b35acf6678badf7b397e3d` |
| `.../libggml-htp-v81.so` | DSP skel (ADSP_LIBRARY_PATH = worker dir) | `533d57c1b6dd1b970e14a5a03e131463da67f32aa5ce0ae212a0d6ff0356a6db` |
| `.../libomp.so` | NDK r28b aarch64 | `75d0611ca77c6aa25e669d37127697a63e88a626e26bf5ab9172c4c5893d63b1` |
| `/data/local/tmp/s43-dual-session-20260923-v1/direct_phone_ffn_session.sh` | production script + S43 env | `db82632c3608d2d8d2d2a9a94d782fd1fda1acddbfa5d3f39358d89bad91bc8d` |
| `/data/local/tmp/s43-dual-session-root-20260923-v1/` | campaign session root (new, per-session logs) | |
| `/data/local/tmp/s43-dual-smoke-20260923/worker.log` | standalone smoke log | |

Reused unchanged: resident workers `/data/local/tmp/s42-per-session-correctness-20260903-v2-bin/llama-ffn-split-resident-workers`
(`0e7736c0...`) and router `/data/local/tmp/s42-ready-subset-router-20260905-v2/llama-ffn-split-resident-router`
(`25d27ff1...`). Neither needs a rebuild: both link only libc/libm/libdl (no ggml), and the resident-workers binary
does `fork()` + `execv(--worker path)` (source `launch_worker()`, binary carries `exec failed` and no `[ffn-worker]`
strings), inheriting the environment, so the S43 variables exported by the session script reach every exec'd
worker. `ffn-split-worker-entry.h` is included only by ffn-split-worker.cpp in the current tree.

Session script diff (production -> dual; `data/phone-stack/direct_phone_ffn_session.sh.diff`):

```diff
@@ -265,6 +265,16 @@
 export GGML_HEXAGON_NHVX=${GGML_HEXAGON_NHVX:-4}
 export S41_DISABLE_GRAPH_CACHE=1
 
+# S43 dual-engine FFN split (NPU primary + Adreno secondary), zero-copy: the
+# qualified batch plan is split-row, so every phone call is tokens=1 and no NPU
+# copy of the GPU columns is kept (MAX_TOKENS=0). Overridable per launch.
+export S43_FFN_SECONDARY_BACKEND=${S43_FFN_SECONDARY_BACKEND:-GPUOpenCL}
+export S43_FFN_SECONDARY_FRACTION=${S43_FFN_SECONDARY_FRACTION:-0.15}
+export S43_FFN_SECONDARY_ALIGN=${S43_FFN_SECONDARY_ALIGN:-64}
+export S43_FFN_SECONDARY_MAX_TOKENS=${S43_FFN_SECONDARY_MAX_TOKENS:-0}
+export S43_FFN_DUAL_LOG_PERIOD=${S43_FFN_DUAL_LOG_PERIOD:-1}
+export S43_FFN_DUAL_WARMUP_ROUNDS=${S43_FFN_DUAL_WARMUP_ROUNDS:-3}
+
 io_flag=
 case "${S41_FFN_F16_IO:-0}" in
```

The adapter passes no S43 variables (its env list is fixed: S41_FFN_*, gadget, NCM), so the defaults apply.
LD_LIBRARY_PATH/ADSP_LIBRARY_PATH are the worker dir (script derives `worker_root` from the worker path).

### Worker change: concurrent warm-up at load

`S43_FFN_DUAL_WARMUP_ROUNDS` (default 3, 0 = off): after the secondary graph is reserved and the helper thread
started, run `rounds x layers` full-width tokens=1 dual calls on a zero input, then clear the p50 sample vectors.
Log: `[ffn-worker] dual warmup rounds=R layers=L calls=N first_total_us=.. last_total_us=.. elapsed_us=..`.
Refreshed diff: `../20260923-dual-engine-ffn-worker/ffn-split-worker.dual.diff` (vs the same base). Built with the
README cmake line in `ghcr.io/snapdragon-toolchain/arm64-android:v0.3` (only the worker target was relinked).

### Standalone smoke (TCP, OP15, rig lock)

Qwen HTP0 shard (layers 0-5, 17408 cols) with the session env, `--column-quantum 256` (my choice for the smoke; the
campaign uses 2176/1280, see below):

```
[ffn-worker] dual secondary=OpenCL fraction=0.150 align=64 secondary_columns=4352/17408 max_tokens=0 weights=765.00 MiB
[ffn-worker] weight buffers=HTP0 count=68
[ffn-worker] dual warmup rounds=3 layers=6 calls=18 first_total_us=21360 last_total_us=20474 elapsed_us=373855
[ffn-worker] ready backend=Hexagon layers=6 ... max_tokens=1 quantum=256 ... blocks=68 weights=2295.27 MiB hash=c8dfbd19d741a902
```

Zero-copy confirmed: `max_tokens=0` in the dual line (the `max_tokens=1` in the ready line is the per-call token
limit `--max-tokens`, unrelated), NPU 2295.27 MiB + GPU 765.00 MiB = 3060.27 MiB = the whole shard
(3,208,644,448 B), i.e. no NPU copy of the GPU columns. Host staging copies are freed after upload (primary and
fused secondary `swap()` in the load path). The 20 ms/layer warm-up calls are NOT representative: quantum 256 gives
68 tiny blocks per layer and, with 64-column alignment, a 25 % GPU share (4352/17408). With the campaign quanta the
effective share is 14.7 % (Qwen 2176: 320 of 2176 per block) and 15.0 % (Gemma 1280: 192 of 1280).
Smoke incident: in `adb shell "nohup worker > log 2>&1 &"` the background worker inherited the adb session's stdin,
so that adb call did not return while the worker lived and my wrapper never reached its stop step and held the rig lock idle for ~1.5 h until the coordinator flagged it; I
killed only my worker (PID 21468). All later phone steps run from pushed/desktop scripts with stdin closed.

## 3. Inputs

`/home/zhihao/s42-trace-longtaildev-{baseline,treatment}-20260923-inputs`, derived with `prepare_trace_inputs_v2.py`
from `s42-trace-longtail-{baseline,treatment}-20260923-inputs` (baseline `desktop-baseline`, treatment
`energy-aware`) with the dev trace `/mnt/storage/burstgpt-source/longtail_dev_v1/` (replay schedule, manifest,
semantic source, overlay), then in rig.json only: `worker_path`, `session_script`, `session_root` (new dirs),
`remote_hash_cache_path` -> `PHONE_HASH_CACHE_DUAL.json` (copy of the production cache; production cache untouched),
`rig_id`; `TRANSPORT_QUALIFICATION_IDENTITY.json` = the DUAL identity. Everything else (models, shards, resident
binaries, evidence, calibration) identical to the longtail inputs (`data/inputs/`, `rig.dev-vs-longtail.diff`).
Both preflights: `phone_assistance_ready=False` with all 62 phone routes SHADOW at preflight, exactly as in the
longtail baseline-1 and treatment-5 preflights (routes are learned during the run).

Launcher: `/home/zhihao/s43-trace-longtaildev-20260923-run_pair.py` (copy in `data/launcher/`): one
`flock -w 3600` on the rig lock for the whole pair; baseline preflight, treatment preflight, baseline run, treatment
run; battery + notify code logged before/after each run; phone session root tarred into each run dir.

## 4. Results (pair 1: baseline 03:14:56-03:30:21 UTC, treatment 03:30:22-03:45:57 UTC, 2026-09-24)

Run dirs: `/home/zhihao/s42-trace-longtaildev-baseline-20260923-inputs/run-baseline-1`,
`/home/zhihao/s42-trace-longtaildev-treatment-20260923-inputs/run-treatment-1` (phone session logs:
`run-treatment-1/phone-session-root.tar`). Copies: `data/runs/*` (RESULT.json.gz, logs, preflights),
`data/results/*`, `data/phone-session/*.tgz`. Analysis: `ANALYZE_PAIR.sh 1 1` on the desktop
(`/mnt/storage/s43-dual-prep/analysis`, log `data/results/ANALYZE_PAIR-1-1.log`).

### Energy (compare_trace_energy.py, exact output)

```
run                          status    dur s  req   CPU kJ   GPU kJ  host kJ  host W phone kJ*  fleet kJ host vs first fleet vs first
baseline                     PASS        898    6     51.5     25.2     76.7    85.4      0.79      77.5         +0.0%          +0.0%
dual                         PASS        905    6     45.6     26.2     71.8    79.3      1.19      72.9         -6.4%          -5.8%
* phone energy is assumed (4.5 W active / 0.875 W idle), not measured
```

This is phone-assisted (dual worker) vs desktop-only on the dev trace. It does NOT isolate dual vs NPU-only: there is
no NPU-only arm on the dev trace. For context only (different trace, 31 requests): the NPU-only longtail pair
(baseline-1 vs run-5) was host 523.7 -> 431.0 kJ, -17.7 %.

### Exactness (analyze_longdecode_pair.py): PASS

`checks`: both_runs_completed, same_requests, matched_request_inputs, all_outputs_token_identical = true; 6/6
identical, 0 differences, 1,071 output tokens (max 578). Host saving 6.40 %, duration +0.88 %. Treatment use of
the phone: Qwen 3/3 requests assisted, 388 output tokens, 2,940 phone calls (245 tokens x 12 phone layers); Gemma
2/2 assisted, 636 tokens, 312 phone calls (52 tokens x 6 layers); Llama overlay no phone.

### The dual path ran over the real USB DMA-BUF transport

- **Server side** (large-model-2 stderr):
  `S41SERVERFFN {"status":"ok","calls":300,"transport":"functionfs-usb","allocator":"devmem","transport_generation":"functionfs-dmabuf-async-ring-v2","batch_plan":"split-row","queue_depth":4,...}`.
  3,252 `S41SERVERFFNUSB` calls, all tokens=1.
- **Transport overhead** is unchanged: USB round trip minus compute is ~0.7 ms, as in run-5.
- **Phone side:** the session root holds one resident session. Its worker.log has one dual config line per exec'd worker:
```
[ffn-worker] dual secondary=OpenCL fraction=0.150 align=64 secondary_columns=2560/17408 max_tokens=0 weights=450.00 MiB   (Qwen HTP0, layers 0-5)
[ffn-worker] ready ... max_tokens=4 quantum=2176 ... blocks=8 weights=2610.03 MiB
[ffn-worker] dual secondary=OpenCL fraction=0.150 align=64 secondary_columns=2560/17408 max_tokens=0 weights=450.00 MiB   (Qwen HTP1, layers 6-11)
[ffn-worker] dual secondary=OpenCL fraction=0.150 align=64 secondary_columns=2304/15360 max_tokens=0 weights=303.75 MiB   (Gemma, layers 16-21)
[ffn-worker] ready ... max_tokens=2 quantum=1280 ... blocks=12 weights=1721.30 MiB
```
- **Zero-copy in the campaign:**
  - The worker received `S43_FFN_SECONDARY_MAX_TOKENS=0`.
  - Qwen: 2610.03 + 450.00 = 3060.03 MiB, exactly the shard.
  - Gemma: 1721.30 + 303.75 = 2025.05 MiB, exactly the shard.
  - `max_tokens=4/2` in the ready lines is the per-call `--max-tokens` from `S41_FFN_MAX_TOKENS`. Every call was tokens=1 (split-row), so the slow GPU M>1 path was never used.
- **Effective GPU share:** 14.7 % (Qwen), 15.0 % (Gemma).
- **Warm-up (3 x 6 calls per worker):**
  - HTP0: 10.8 -> 10.6 ms. HTP1: 9.0 -> 8.8 ms. Gemma: 5.7 -> 6.0 ms per call.
  - It takes 0.11-0.19 s per worker at load.
  - It cannot help against the per-token GPU wake-up below.

### Per-layer phone compute vs run-5 (NPU-only), host-side S41SERVERFFNSHAPE, call-weighted

| shape | dual calls | dual mean ms | run-5 ms | speedup |
|---|---|---|---|---|
| Qwen full, 17408 cols | 2,772 | **9.607** | **9.72** (user ref; call-weighted 9.736) | **1.012x** |
| Gemma full, 15360 cols | 114 | **7.233** | **6.59** (user ref; call-weighted 6.550) | **0.911x** |
| Qwen 8704 | 168 | 5.128 | 4.404 | 0.859x |
| Gemma 11520 | 66 | 5.714 | 4.915 | 0.860x |
| Gemma 7680 | 66 | 4.324 | 3.309 | 0.765x |
| Gemma 3840 | 66 | 3.152 | 1.694 | 0.538x |

Percentiles (usb_call_stats): Qwen full p10/p50/p90 8.04/9.71/10.34 ms (run-5 9.52/9.73/10.02), Gemma full
5.49/6.67/11.46 (run-5 6.42/6.57/6.75). Full tables: `data/results/treatment_timings.md`, `usb_call_stats_*.md`.

### Why: first call of every token burst, and no bandwidth gain within the burst

Per layer (phone-side S43DUALFFN p50 us, and host-side compute mean per layer vs run-5; `data/results/gap_analysis.md`):

| layer (position) | dual total | NPU leg | GPU leg | NPU waits | dual mean ms | run-5 mean ms |
|---|---|---|---|---|---|---|
| Qwen 0 (first after ~0.4 s idle) | 13,612 | 8,580 | **13,298** | 4,960 | **13.67** | 9.84 |
| Qwen 1-11 (5-20 ms gaps) | 9,010-9,896 | 8,912-9,877 | 7,144-7,864 | 0 | 9.05-9.49 | 9.55-10.06 |
| Gemma 16 (first) | 11,478 | 5,958 | **11,287** | 5,803 | **11.48** | 6.75 |
| Gemma 17-21 | 5,701-6,739 | 5,690-6,729 | 4,682-5,225 | 0 | 6.06-6.56 | 6.47-6.67 |

- **GPU wake-up.**
  - After the between-token idle, the GPU leg of the first call takes +5.9 ms (Qwen) and +6.3 ms (Gemma) longer than within the burst, and the NPU waits for it.
  - The NPU-only path has essentially no such penalty: run-5 layer 0 is 9.84 ms vs 9.73 ms for the others. Calls after 100-1000 ms gaps average 9.84 ms in run-5 vs 13.67 ms dual.
  - Layer 6 is the first call of the second Qwen process (another OpenCL context) and is not slow. The cause is idle time, not the process switch.
- **Within the burst the gain is only ~5 %.**
  - The NPU leg on 85 % of the columns takes ~9.4 ms. That is slower per column than NPU-only on 100 % (0.63 vs 0.56 us/col), and slower than the same leg on the first call, when the GPU is still waking (8.58 ms).
  - The GPU leg runs at ~2.9 us/col.
  - Combined DRAM rate: ~57 GB/s vs 55 GB/s NPU-only (+3 %). The bench reached 65.7 vs 52.8 GB/s (+24 %).
  - Both engines are slower per column than in the bench (NPU leg +17 %, GPU leg +11 %). The GPU/NPU rate ratio (~0.21) is the same.
  - This is the bench's "bad state" (the GPU steals NPU bandwidth), which the fused GPU layout on a charged phone had removed under the bench's continuous load.
- **Per token.**
  - Qwen, 12 phone layers: 115.3 ms dual vs 116.9 ms NPU-only (1.014x); 1.055x if the first call were like the rest.
  - Gemma, 6 layers: 42.8 vs 39.5 ms (0.92x); 1.05x without the first-call penalty.
- **Partial widths.** Small calls are GPU-latency bound. At 3840 cols the GPU leg p50 is 1.88 ms vs the NPU leg's 1.55 ms, and the NPU waits.
- **Likely mechanism (not measured in this run, no clock sampling in the campaign).** The campaign's GPU duty cycle is low and bursty: ~12 calls of ~7 ms, then ~0.4 s idle. So the Adreno power-collapses between tokens, and its governor never raises the GPU clock or bus/DDR vote the way the continuous bench load did. The bench saw 1200 MHz in every dual run.

### Measured phone power (diagnostic; phone_power_windows.py; phone USB-powered at 500 mA with the OPLUS hold)

| run | FFN-active samples (call within +-150 ms) | idle samples | active - idle |
|---|---|---|---|
| dual (dev trace) | 3,603 mW (n=489) | 3,033 mW (n=2,569) | **+570 mW** |
| run-5 NPU-only (longtail) | 2,905 mW (n=9,511) | 2,692 mW (n=7,681) | +213 mW |

Different trace and time of day, and the idle levels differ by 341 mW, so this is indicative only. The dual worker
appears to draw ~0.35 W more while active, and it buys no phone-latency gain. The energy table above uses the
assumed 4.5 W phone model and cannot see this.

### Phone state (dumpsys battery + oplus battery_notify_code via su; no guard applied)

| point (UTC) | level | status | voltage | temp | charge counter | notify |
|---|---|---|---|---|---|---|
| before smoke / setup (~01:20) | 80 % | 4 (not charging, OPLUS hold) | 4208 mV | 29.1 C | 5,346,000 | 0 |
| after smoke (~02:55) | 80 % | 4 | 4206 mV | 29.2 C | | 0 |
| before baseline 03:14:56 | 80 % | 4 | 4203 mV | 28.8 C | 5,340,000 | 0 |
| after baseline 03:30:22 | 80 % | 4 | 4203 mV | 29.1 C | 5,340,000 | 0 |
| before treatment 03:30:22 | 80 % | 4 | 4203 mV | 29.1 C | 5,340,000 | 0 |
| after treatment 03:45:57 | 79 % | 2 (charging, 287) | 4157 mV | 34.6 C | 5,268,000 | 0 |

No 512 at any point. During the treatment the battery supplied the difference beyond the 500 mA USB input: -72 mAh,
and then charging resumed at 79 %.

## 5. Verdict and next steps

- **Deployment is correct and runnable.**
  - Identity-bound, zero-copy and token-exact.
  - Resident workers exec the dual worker with the S43 environment over the qualified FunctionFS DMA-BUF transport.
  - Dev-trace host energy -6.4 % vs desktop-only.
- **It is not a speedup in the campaign.**
  - Per layer: 1.01x (Qwen) and 0.91x (Gemma) vs run-5.
  - Per token: 1.014x and 0.92x.
  - It costs ~0.35 W more phone power while active.
  - Recommendation: keep the production worker (NPU-only) for energy arms until the GPU leg is fixed.
- **Candidate fixes (none implemented; each needs a measurement):**
  1. Reproduce first. Run the standalone bench with the campaign's pattern (12 back-to-back calls, ~450 ms idle) while sampling the GPU clock and bus_dcvs, to confirm the DVFS/power-collapse explanation. Phone-only; minutes of lock time.
  2. GPU keep-alive while a session is active. The helper thread would submit a trivial kernel every ~40 ms between calls. That should remove the ~6 ms first-call penalty (worth ~4 % per Qwen token), at some GPU power.
     - A root alternative is kgsl `idle_timer` / `min_pwrlevel`. That is a system DVFS change and needs the user's go-ahead.
  3. The within-burst bandwidth gain needs the DDR/GPU vote up under bursty load. That may not be reachable without pinning clocks, which costs power, so the energy case must be re-checked with measured phone power.
  4. For attribution, run an NPU-only arm on the dev trace. Inputs = the longtaildev treatment inputs with the production worker/session paths and the production identity.
- **Zero-copy tradeoff.** A size gate (skip the GPU for small partial widths) would need the NPU copy of the GPU columns. Under zero-copy, those columns exist only on the GPU.

## 6. Caveats

- One pair, one phone, dev trace (6 requests, ~900 s). The per-call reference is run-5 (longtail trace, earlier
  day), not an NPU-only run on this trace.
- Phone energy in the energy table is assumed (4.5 W / 0.875 W); the measured-power numbers are diagnostic only.
- `S43_FFN_DUAL_LOG_PERIOD=1` writes one ~210-byte line per call into worker.log. That file is also residency.log,
  which the adapter reads over NCM with a 16 MiB cap, so a very long session could exceed the cap. The dev trace wrote 0.59 MB.
  For long traces, raise the period in the session script. That changes the script, so a new identity is needed.
- Two worker processes write to one worker.log. 8 of 3,252 S43DUALFFN lines are interleaved with other output
  (the host-side counts are exact).
- The identity schema does not cover the worker's shared libraries (recorded separately).
- The dev inputs point `remote_hash_cache_path` at a copy (`PHONE_HASH_CACHE_DUAL.json`); the production cache was not touched.
- **Smoke incident.** My first smoke command held the rig lock idle for ~1.5 h, for the reason in section 2. I stopped
  only my own worker, PID 21468, without the lock, because my own stuck command held it. Nothing else was killed.
- Nothing committed. Local tree changes: `examples/layersplit/ffn-split-worker.cpp` (warm-up), the refreshed
  `../20260923-dual-engine-ffn-worker/ffn-split-worker.dual.diff`, and this report.
