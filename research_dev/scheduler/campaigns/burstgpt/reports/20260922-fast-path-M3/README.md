# M3: OP11 integration with OP15

2026-09-23 00:29 UTC. Pixel per-operation profile **PASS**: 384 exact phone
outputs; separate gate/up/SwiGLU/down/cast intervals, plus an unfused add arm.
Normal full-width worker averages 32.628 ms; profiled worker 48.527-48.567 ms.
Timings include profiler barriers and scheduling. Earlier invalid graph-slice
attempt is retained as FAIL; isolated retry PASS has large overhead. No new
energy claim. See the final section for numbers and limitations.

2026-09-22 22:20 UTC. Pixel kernel-rate arithmetic **PASS** from existing
profiling: full-width matrix-vector intervals average 18.150 GFLOP/s;
matrix work divided by the stage-only worker interval is 16.123 GFLOP/s.
These are achieved workload rates, not a hardware peak measurement.
Optimization candidates below remain untested; no new physical run.

2026-09-22 21:58 UTC. Pixel phone-internal latency profile **PASS**: 384
numerically qualified calls, all outputs identical across diagnostic/control
arms. Execution plus synchronization is 97.6%/98.4% of phone worker time at
half/full FFN width; GPU matrix-vector operations dominate timestamp intervals.
This isolated microbenchmark adds no server/energy claim. See final section.

2026-09-22 21:09 UTC. Pixel Vulkan real-server FFN assistance **PASS** for Qwen
layers18-23:744 verified decode calls and all four64-token outputs exact. With
50%/100% of those layers' columns on Pixel, measured host request energy saves
10.659%/11.170% against the mean of two controls; decode is9.3%/18.1% slower.
One sample per split; Pixel energy and full-trace saving unmeasured. Automated
multi-phone scheduler integration remains deferred. Worker/server cleanup PASS.
See the final six-layer result below. No Tensor SDK is used for this GPU path.

2026-09-22 20:55 UTC. Pixel GPU server trial started at the user's request.
FFN block-size tuning and one-write TCP replies PASS numerical qualification;
median one-row round trip improved from103.110 to35.008ms. The bounded server
comparison is pending; no new energy saving or full-token PASS yet. See the
Pixel Vulkan server section below.

2026-09-22 20:38 UTC. Pixel FFN TPU feasibility review: the existing column-split
protocol can carry the workload, but custom Qwen TPU compilation remains blocked
by the missing Tensor SDK compiler. A bounded first test is specified in the
Pixel FFN TPU section below. No new hardware arm or implementation in this review;
the add and Vulkan results remain the measured evidence.

2026-09-22 19:58 UTC. Latest: Pixel Tensor TPU add round-trip qualification PASS.
Public LiteRT2.2.0 + Google precompiled P25 model, 20 standalone and 180 TCP
calls, exact numerical results. A/B/A split/single-send/split replies gave
median50.003/4.666/50.070 ms. Best p90 5.203 ms, phone worker median1.689 ms.
This is a 128-element add, not Qwen FFN; Qwen TPU, energy and integration
remain unverified. User has no Tensor SDK installed, so custom FFN AOT
compilation is blocked. Existing production worker unchanged; cleanup PASS.
See the TPU round-trip section below.

2026-09-22 19:26 UTC. Latest: Pixel 10 Pro Vulkan qualification PASS for Qwen
layer18, 12 calls/28 rows, max relative L2 0.000311882. Latency improvement
over repaired OP11 NPU FAIL: Pixel median round trips 103.110/142.184/201.093 ms
at rows1/2/4 versus matching OP11 one-layer 20.673/63.005/77.991 ms.
Pixel USB is 5000 Mbit/s; Tensor TPU, Wi-Fi, energy and integration NOT VERIFIED.
See the Pixel qualification section below. OP11 is now disconnected; OP15 and
Pixel are visible. All qualification workers exited normally; owned forwards removed.

2026-09-22 18:15 UTC: OP11 v73 NPU repair/FFN qualification PASS
after correcting the DSP mapping limit. Normal DMA works with HMX disabled:
48 calls, 112 rows, max relative L2 1.671e-4. Two/four-row median round trips
are 63.301/77.097 ms, replacing OpenCL's 4.75 s. Transport overhead remains
large; Wi-Fi is NOT MEASURED because OP11 has no network address or route.
User authorized NPU fixes and qualification; full M3 integration remains deferred.
Two-phone execution, full-model exact tokens and energy acceptance NOT VERIFIED.

The user requested OP11 integration after Tasks 1 and 2 ran. The starting design
is M3_SECOND_PHONE_CHECKLIST.md sections 7 and 8: OP11 OpenCL over TCP/ADB 5037,
OP15 on its existing FunctionFS transport, disjoint FFN layer ownership.
The original integration proposal excluded NPU and kernel changes. The user
subsequently requested an NPU fix and Wi-Fi assessment; results are below.

## Current read-only inventory

- OP11 serial 832358d4 is available through ADB 5037 at USB 480 Mbit/s.
  /data has 19 GB available. The prepared worker, OpenCL runtime, OpenMP runtime
  and Qwen layers 18-23 shard are already present. No FFN worker was running on
  OP11 at inspection.
- The saved OpenCL smoke log only proves worker readiness for layers 18-21,
  maximum four rows, 2040 MiB weights. It does not prove a completed TCP call,
  numerical correctness or an energy benefit. The new qualification below
  supplies completed-call and numerical evidence.
- At initial inspection the other user session owned the rig lock for m4a8b/run-treatment-2.
  OP15 is visible as an experiment USB device at 5000 Mbit/s. No shared source
  deployment or hardware experiment from this session overlapped that run.

## Proposed integration scope (only item 1 authorized)

1. Qualify OP11's existing worker through an ADB forward using protocol v6.
   Check artifact and layer identities, complete calls at one/two/four rows,
   returned payload hashes and numerical output against the desktop CPU worker
   using the same shard and input rows. Measure compute and round-trip latency.
   A finite request count lets each test worker exit normally. Keep its receipt
   separate from OP15's FunctionFS qualification.
2. Extend configuration/rig.py with device-keyed phone configurations and
   topology entries, preserving the existing single-phone serialized form.
   Each phone binds its serial, transport, USB speed floor, worker paths,
   resource IDs, endpoint and qualification. Reject duplicate device/session
   identities and ambiguous host forward ports.
3. Extend the existing phone session lifecycle for a direct TCP worker. The
   current adapters/phone_transport.py TCP mode denotes a tensor bridge and
   must retain that meaning for legacy campaigns. Direct ADB TCP must explicitly
   select its worker lifecycle and must never invoke FunctionFS gadget setup.
   Keep all process, forward and cleanup ownership tied to the selected device.
4. Extend catalog materialization, helper preparation and phone residency to
   keep device affinity on every shard, lease and receipt. OP11 capacity, cost
   evidence and session generations must be independent of OP15. Start with
   existing Qwen shards and disjoint masks; refuse overlaps or uncovered layers.
   Do not mark OP11 routes qualified from OP15 receipts or extrapolated costs.
5. Extend tools/server/server.cpp's existing FFN runtime to dispatch the shared
   decode policy to one client per phone. Reuse ffn_split::client for each
   transport. A decode batch keeps one union layer mask and column width; each
   client receives its owned subset. Require all owners ready before accepting
   the policy. Keep apply_dormant_host_share's mixed-policy guard. Add per-device
   call accounting while retaining legacy single-client proof compatibility.
6. Extend launch/preflight/runner and the proof and energy collectors to carry
   both device bindings end to end. Use fresh isolated M3 inputs/deployment;
   never silently run only the first phone in a multi-phone manifest. Rebuild
   and requalify the native transport identity before using changed binaries.

The central code areas are configuration/{rig,models,evidence}.py,
adapters/{catalog_materialization,heterogeneous_rig,phone_transport,phone_session}.py
and their operation modules, _internal/plan_contracts/phone.py,
_unified/{helper_preparation,phone_residency}*, campaigns/burstgpt/{catalog,launch,runner}.py,
tools/server/server.cpp and adapters/llama_server_ops/proofs.py. This crosses the
native runtime and scheduler contracts; it is not a second rig.json entry alone.

## Acceptance and reporting

- Local: legacy single-phone configuration and behavior retained; duplicate or
  overlapping ownership, wrong-device receipts and stale generations rejected;
  independent phone failure/cleanup; one/two/four-row dispatch and accounting.
- Physical bring-up: OP11 direct TCP completes with correct numerical output
  and normal worker exit. No force-kill of an in-flight phone worker.
- M3: desktop-only, OP15-only, and two-phone arms with matched prompts, output
  lengths and equal total assisted fraction where possible. Report exact token
  comparison, per-device call proofs, host decode watts, ms/token and released
  bytes. Keep measured host and assumed phone energy separate. A larger released
  mask is an additional capacity arm, not the equal-fraction comparison.
- Run the 24-request and realistic traces only after the mechanism gate passes.
  Do not attribute a two-phone energy win before a completed matched pair.

No production code or rig configuration has changed in this checkpoint.

## Qualification method and local result

`qualify_op11_tcp.py` uses the existing protocol-v6 worker. It sends the same
seeded, nonzero F16 inputs to a desktop CPU worker and the OP11 OpenCL worker.
The initial physical check covers layers 18-21, all 17408 FFN columns,
512-column blocks, K=5120, F16 input/output, SwiGLU, and 1/2/4 rows.
There are four repeats per layer and shape: 48 calls and 112 returned rows
per worker. The first repeat is excluded from the latency summary. Identical
inputs across repeats also check output repeatability; distinct rows expose
row reordering. CPU and phone workers run sequentially.

The numerical limit is relative L2 <= 0.01 for every returned row, matching
the earlier M2 FFN row diagnostic. This synthetic-input check does not replace
full-model exact-token acceptance. The report also retains NMSE, absolute
error, per-call hashes, raw inputs/outputs and worker compute/round-trip time.
Round-trip minus compute includes protocol handling and scheduling; it is
not an isolated USB transfer measurement.

Both workers must reject wrong-artifact and missing-layer HELLO messages,
return matching geometry/weight identities, pass every response payload hash,
and exit normally after their fixed request count. The script never kills a
worker. It refuses an occupied OP11 worker or port and creates its own ADB
forward on port 5037. It removes that forward after normal completion.

Local harness check: **PASS**, `physical/local-check-2/RESULT.json`: 48 calls
per worker, 112 rows, maximum relative L2 0, repeat outputs bit-identical,
both workers exited 0. Both identity rejection cases passed for both workers.
Pyflakes and `bash -n RUN_OP11.sh`: **PASS**. This local result uses a tiny
Llama fixture and CPU workers; it is not an OP11 qualification result.

Physical job queued at 2026-09-22 16:27:49 UTC under `flock -w 900` behind
the other session's m4a8b/run-treatment-2. Isolated receipt root:
`/mnt/storage/s42-op11-qualification-20260922-v1`. The shared source, scheduler
inputs, OP15 transport identity and phone kernel remain unchanged.

## Physical OP11 qualification, 2026-09-22 16:36 UTC

**Transport and numerical correctness: PASS.** All 48 calls completed at
1/2/4 rows, covering Qwen layers 18-21. All 112 rows passed relative L2 <= 0.01;
maximum relative L2 was 3.273897e-4 (0.032739%), maximum NMSE 1.071840e-7,
and maximum absolute difference 0.0078125. Every repeated input returned
bit-identical phone output. Both workers rejected the wrong-artifact and
missing-layer handshakes and returned matching weight identity
`f8647098761d88b7`. All response hashes passed. Both workers exited 0 after
their finite request count. [Raw result](physical/op11-tcp-1/run1/RESULT.json)
and [row comparisons](physical/op11-tcp-1/run1/ROW_COMPARISON.json).

| Rows per layer call | OP11 median round trip, ms | OP11 median compute, ms | CPU median round trip, ms | Phone/CPU round-trip ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.652 | 24.389 | 18.254 | 3.76 |
| 2 | 4751.604 | 4746.867 | 18.850 | 252.07 |
| 4 | 4749.101 | 4741.751 | 19.653 | 241.65 |

Each median uses 12 calls (four layers x three measured repeats), excluding
the first repeat for each layer/shape. The phone held 2040 MiB of weights in
34 buffers. Backend was `QUALCOMM Adreno(TM)` / Adreno 740 on CPH2451,
stock kernel `5.15.189-g306c8fd4beb0`, serial 832358d4, USB 480 Mbit/s,
direct TCP through ADB 5037.

**Suitability for shared decode: FAIL on observed latency.** This is a
performance judgment from the measured comparison, not a predeclared numeric
SLO gate. The 2/4-row compute path is the dominant blocker: worker-reported
compute accounts for about 4.74 of 4.75 seconds. Single-row noncompute time
is also high (median 44.425 ms, versus 4.963/6.573 ms for 2/4 rows).
Do not attribute the multi-row cliff to USB bandwidth or claim an energy win.

The current source selects `kernel_mul_mm_f16_f32_l4_lm` when `ne11 > 1`
for these F16 weights; the optional Adreno xmem path requires N >= 16.
That branch is a candidate explanation for the sharp small-batch cliff.
The deployed binary was not instrumented, so exact kernel-level causality
is NOT VERIFIED. No backend or worker implementation was changed. The next
technical investigation is the 2/4-row OpenCL matmul dispatch, before M3
scheduler integration.

**Shard provenance: PASS.** Independently hashed the complete Qwen parent and
the desktop shard, and compared all 18 stored tensors (layers 18-23, gate/up/down)
using raw tensor SHA-256 values plus shape/type checks. Parent:
`sha256:d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718`.
Shard:
`sha256:dd705a75a3047ead41dbc01984afebc23440e2ea575b25af5c5b22252c69f4c3`.
The actual phone shard has the same SHA-256. The existing index had
`parent_verified: false`; it was left unchanged, with new evidence retained in
[SHARD_PARENT_IDENTITY.json](physical/op11-tcp-1/SHARD_PARENT_IDENTITY.json).
`verify_shard_identity.py` also passed a local 12-tensor fixture check.

**Cleanup: PASS.** Normal CPU and phone exit status 0, 48 calls each; OP11 FFN
worker absent; owned ADB forward removed; phone boot ID unchanged. OP15 was
not used by this qualification. [Postflight](physical/op11-tcp-1/POSTFLIGHT.json).
No binaries, production source/configuration or phone kernel changed.

Not verified: simultaneous six-layer OpenCL residency, real decode activations,
full-model token equality, scheduler routing/ownership, simultaneous use of both
phones, or host/phone energy savings. The synthetic row test is not a production
transport-qualification identity and must not be inserted into current rig inputs.

Reproduce the call test with `RUN_OP11.sh` after choosing a new output root.
The provenance command, also under the rig lock, was:

```sh
flock -w 900 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock \
  python3 /mnt/storage/s42-op11-qualification-20260922-v1/verify_shard_identity.py \
  --source /mnt/storage/s42-trace-v2-20260921-prep/source \
  --parent /home/zhihao/models/Qwen3-14B-Q4KM-dequant-f16.gguf \
  --index /home/zhihao/s42-op11-qwen-shards-20260921-v1/qwen/FFN_SHARDS.json \
  --output /mnt/storage/s42-op11-qualification-20260922-v1/SHARD_PARENT_IDENTITY.json
```

## Phone-path latency breakdown from the same qualification

These are arithmetic means of the same 12 warm calls per shape, so the
components add to the total. Earlier tables report medians. No new hardware
run was needed. These are per-layer FFN calls with weights already resident.

| Rows | Worker interval, ms | Outside worker, ms | Total, ms | Worker share |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 24.349 | 44.735 | 69.084 | 35.246% |
| 2 | 4746.182 | 4.954 | 4751.136 | 99.896% |
| 4 | 4742.482 | 7.074 | 4749.556 | 99.851% |

The worker interval starts at layer lookup and includes graph construction/
allocation, input tensor upload, OpenCL graph execution and output tensor
download (ffn-split-worker.cpp:2040-2078). It is not a separate GPU-kernel
measurement. Graphs contain 34 column blocks, each with gate/up/down matmuls,
SwiGLU and partial-output accumulation; individual operation times are absent.

Outside-worker time includes both directions of ADB/TCP/USB transport,
socket handling, phone input/output validation, copies and hashes, host
header handling, and scheduling. Its mean is the per-call difference between
host round-trip and worker interval. Upload versus download, USB versus ADB,
and GPU copies versus kernels were not instrumented.

Each direction carries 10/20/40 KiB at 1/2/4 rows, plus request/response
headers of 36/48 bytes. At the nominal 480 Mbit/s link rate, an idealized
serialization-only floor is 0.343/0.684/1.367 ms total, ignoring ADB/TCP/USB
framing and scheduling. These calculated floors are not transfer measurements.

Single-row outside-worker time is consistently 43.255-46.130 ms; its exact
cause is unverified. Multi-row latency is more than 99.8% inside the worker
interval. One-time loading/startup, input generation and host payload hash
preparation, post-receive numerical checks, scheduler waiting, attention and
other model layers are excluded. Further decomposition needs new timing
instrumentation and a separate locked qualification run.

[Machine-readable breakdown](LATENCY_BREAKDOWN.json).

## OP11 v73 NPU repair and retest (2026-09-22 18:15 UTC)

**PASS: the tested NPU crash is fixed.** The host had been raised to 64 buffers
per batch, but `htp-ctx.h` still reserved only 16 DSP mappings. `mmap_buf`
silently returned when all slots were occupied, leaving later buffer bases at
zero. The worker allocates 34 weight buffers even for one Qwen layer. This
explains a zero-address weight read and the earlier fault reported at `dmpoll`.

The first diagnostic binary bypassed user DMA with synchronous HVX copies.
It still failed in startup, exit 134, zero measured phone calls: logcat reports
Bad VA 0 at `dma_queue_push_sync+0xc0`, and disassembly shows a vector source
load. That attempt is **FAIL**; it is retained under `software/op11-v73-sync`
and `physical/op11-v73-sync-1`. The workaround was removed from the working
source, since avoiding DMA did not address the unmapped input.

The retained fix makes the mapping capacity follow `HTP_OP_MAX_BUFS`, widens
the reuse mask to 64 bits, and evicts unused entries when mapping slots or
mapped-byte capacity are exhausted. Oversized batches, exhausted slots and
null mapping results now fail explicitly. Independent DMA allocation bugs
were also corrected: check either failed allocation before memset, free
partial allocations, and reject a failed queue during session start.
Normal DMA, the original unsigned PD and the stock phone kernel are retained.
HMX is explicitly disabled with `GGML_HEXAGON_NHMX=0`; the old checklist's
claim that the host automatically disables it below v75 was not accurate for
this tree. No signed-PD or kernel experiment was needed.

| Arm | Result | Phone calls / checked rows | Max relative L2 | Normal exit |
| --- | --- | ---: | ---: | --- |
| Synchronous-copy diagnostic, layer 18 | FAIL | 0 / 0 | unavailable | startup abort 134 |
| Mapping fix, layer 18 | PASS | 12 / 28 | 0.000083630 | CPU and phone 0 |
| Mapping fix, layers 18-21 | PASS | 48 / 112 | 0.000167056 | CPU and phone 0 |

Both successful runs cover 1/2/4 rows, full 17408 columns, 512-column blocks,
F16 I/O and the same seeded inputs as the earlier OpenCL check. All rows pass
the predeclared relative L2 <= 0.01 limit. Repeated outputs are bit-identical.
Artifact/layer rejection, HELLO weight identity, response hashes and unchanged
phone boot ID pass. The four-layer worker reports 2040.13 MiB of weights.

Unified timing below uses arithmetic means of 12 warm calls per shape across
four layers. Units are ms per layer call; components add to the total.

| Rows | Previous OpenCL total | NPU worker | NPU outside worker | NPU total | CPU total | Total speedup vs OpenCL |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 69.084 | 15.892 | 34.017 | 49.909 | 18.185 | 1.38x |
| 2 | 4751.136 | 18.268 | 44.856 | 63.124 | 18.860 | 75.27x |
| 4 | 4749.556 | 31.451 | 39.234 | 70.685 | 19.737 | 67.19x |

NPU median round trips are 59.622/63.301/77.097 ms; median worker intervals
15.803/18.223/31.440 ms. Worker time includes graph preparation, input/output
copies and execution. Outside-worker time includes ADB/TCP/USB, protocol
handling, hashing and scheduling; it is not pure link serialization. Some
calls avoid the roughly 40 ms extra delay, hence means below medians.
Exact attribution of that delay has not been verified. TCP_NODELAY already
exists on the current worker's accepted socket; do not assume adding it again
fixes the ADB forwarding path.

**Wi-Fi: NOT VERIFIED.** Read-only checks before and after the runs found no
OP11 wlan0 IPv4 address and no route. The user was asked to connect it to a
network reachable from 172.20.74.85. Direct Wi-Fi TCP could remove ADB overhead,
but the actual latency and throughput depend on that connection; no speedup
is claimed. Both endpoints would currently use Wi-Fi (desktop wlp3s0). The
NPU fix changes the bottleneck enough that the old conclusion of 99.8% worker
time no longer applies. A direct Wi-Fi comparison must use the same binary,
shard, payloads and finite call count, under the shared rig lock.

Local checks **PASS**: v73 and v81 DSP builds; mapping regression compiled with
undefined-behavior sanitizer (64 entries, full reuse, high bit 63, slot/byte
pressure eviction and fail-fast guards); the same test fails on the old 16-slot
source at an unmapped buffer. The extended protocol harness passed a native
CPU fixture (12 calls, 28 rows, zero error), pyflakes and shell syntax checks.
Both builds emit the existing LTO advisory in `core_dot_chunk_fp16`; physical
v73 numerics pass, but v81 was compile-tested only and not deployed to OP15.
DMA allocation-error injection was not tested.

Deployment is isolated at `/mnt/storage/s42-op11-v73-mmap-20260922-v1` on the rig
and `/data/local/tmp/s42-op11-v73-mmap-20260922-v1` on OP11. DSP SHA-256:
`0025a0e4c500e3f8f1d62dd08009651ce7681444a8f4df0d5e4dc65e345ea374`.
Other phone libraries/worker were copied from the existing qualification bin
directory. Existing bins, OP15 deployment and scheduler inputs were not
replaced. Source snapshots, build logs/hashes and exact launch scripts are in
`software/op11-v73-mmap/`; reproduce with new rig/phone/output roots, not by
rerunning scripts against completed directories. All hardware operations ran
under the shared lock. Final cleanup PASS: no OP11 FFN worker, no ADB forward,
no phone reboot and no job left queued. No force-kill was used.

Still unverified: six-layer simultaneous residency, layers 22-23 numerical
execution in this retest, real decode activation/token equality, two-phone
integration, sustained thermal behavior, host/phone energy and Wi-Fi speed.
These qualification results are not production transport receipts.

Evidence: [NPU assessment](NPU_QUALIFICATION_ASSESSMENT.json),
[latency breakdown](NPU_LATENCY_BREAKDOWN.json),
[one-layer result](physical/op11-v73-mmap-1/run1-layer18/RESULT.json),
[four-layer result](physical/op11-v73-mmap-1/run2-layers18-21/RESULT.json),
[cleanup/network check](physical/op11-v73-mmap-1/CLEANUP_AND_WIFI.json),
[mapping regression](software/op11-v73-mmap/MAPPING_REGRESSION.json).

## 2026-09-22 19:26 UTC - Pixel 10 Pro qualification

The user connected Pixel 10 Pro, serial `5A040DLCH004ES`, as a candidate after
asking about newer Google phones. This is qualification of the existing Vulkan
worker only. No new Tensor backend or multi-phone scheduler integration was made.

Inventory **PASS**: Tensor G5, Android16/API36, approximately 15.2 GiB RAM,
USB negotiated 5000 Mbit/s through port2-9.2. OP11 is no longer connected;
OP15 remains connected on its existing path. Pixel has no wlan0 IPv4/route,
so Wi-Fi is **NOT MEASURED**. The shell is unprivileged; no root, reboot or
kernel change was used. PowerVR D-Series DXT-48-1536 MC1 exposes Vulkan1.4.303,
FP16 and 16-bit storage, with a 128 MiB maximum storage-buffer range.

The existing OpenCL backend accepts Adreno/Intel, so the presence of PowerVR
OpenCL libraries alone does not qualify it. Built the existing FFN worker for
Android arm64/API28 with Vulkan enabled and OpenMP/OpenCL/Hexagon disabled,
using NDKr28b and static libc++. Build **PASS** after adding the missing `pocs/`
source subtree and extracted SPIR-V headers include path to the isolated build.
These were build setup failures, not Pixel runtime failures. No Vulkan or FFN
production code was changed. `qualify_op11_tcp.py` now accepts a phone label,
Vulkan0 and an explicit runtime-library list, preserving OP11 defaults.
Local CPU harness check **PASS**, 12 calls/28 rows with zero error; pyflakes PASS.

The rig and phone deployments are isolated at:

- Rig: `/mnt/storage/s42-pixel10pro-qualification-20260922-v1`
- Pixel: `/data/local/tmp/s42-pixel10pro-qualification-20260922-v1`

All hardware work ran under the shared execution flock and ADB5037. The first
tiny launch **FAIL** was a caller error: the artifact argument omitted the
required `sha256:` prefix. CPU startup rejected it before a Pixel worker ran.
The corrected tiny fixture **PASS** at 19:24:35 UTC: 12 calls, 28 rows,
max relative L2 0.000329620, exact repeats and normal exits.

Real Qwen layer18 **PASS** at 19:25:39 UTC: full 17408 FFN columns, quantum512,
F16 I/O, rows1/2/4, 34 Vulkan weight buffers and 510 MiB weights. All 12 calls
completed, all 28 rows were below the 1% relative-L2 limit; maximum relative
L2 0.000311882 (0.031188%), max NMSE 9.727e-8, max absolute error 0.0078125.
Repeated inputs returned identical bytes. Artifact/layer rejection, geometry,
weight identity, phone/host shard hashes and payload hashes passed. Logs show
Vulkan weight allocation and the PowerVR backend. This does not qualify the TPU.

Median milliseconds per single layer call; three warm samples per cell,
excluding the first of four repeats at each row count:

| Rows | Pixel GPU worker | Pixel outside worker | Pixel round trip | OP11 NPU worker | OP11 round trip | Desktop CPU round trip |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 59.600 | 43.510 | 103.110 | 16.308 | 20.673 | 18.440 |
| 2 | 95.484 | 44.119 | 142.184 | 18.517 | 63.005 | 18.991 |
| 4 | 188.701 | 11.879 | 201.093 | 32.054 | 77.991 | 19.766 |

The OP11 comparator is its **one-layer** `run1-layer18`, not the four-layer
aggregate. Input hashes and complete HELLO identities match between the runs.
These are separate short runs with no thermal/power controls. Independent
medians do not necessarily sum. Arithmetic means and raw receipts are retained
in `PIXEL10PRO_QUALIFICATION_ASSESSMENT.json` and the physical run directories.

Latency improvement over the repaired OP11 NPU **FAIL** on this comparison.
Pixel GPU worker time alone exceeds the OP11 NPU round trip at two/four rows.
The faster USB link does not resolve the GPU execution cost. Outside-worker
time includes protocol framing, copies, hashing, TCP/ADB/USB and scheduling;
it is not a measurement of raw USB transit. Worker time includes graph setup,
upload, execution and download. No pure-kernel or network-only timing is claimed.

The Tensor TPU remains a separate candidate. Google's current
[Tensor SDK documentation](https://developers.google.com/edge/litert/next/tensor-sdk)
lists Tensor G5 support and beta access registration. Its
[LiteRT NPU path](https://developers.google.com/edge/litert/next/npu) supports
AOT execution through CompiledModel, with no Google Tensor JIT yet. Using it
for our shard requires model conversion, compilation and a separate worker
backend, plus proof of actual TPU execution and numerical accuracy. Vendor
library presence is not qualification. No such implementation was attempted.

Cleanup **PASS** at 19:26:25 UTC: no Pixel FFN worker or owned ADB forward,
boot ID unchanged; the isolated files remain for reproducibility. Full-model
token identity, sustained/thermal behavior, other Qwen layers, Tensor TPU,
Wi-Fi, energy saving and multi-phone integration **NOT VERIFIED**.
No new trace/energy arm was run; current energy records remain unchanged.

Evidence: [assessment](PIXEL10PRO_QUALIFICATION_ASSESSMENT.json),
[Qwen result](physical/pixel10pro-vulkan-1/run3-layer18/RESULT.json),
[tiny result](physical/pixel10pro-vulkan-1/run2-tiny/RESULT.json),
[cleanup](physical/pixel10pro-vulkan-1/CLEANUP_AND_NETWORK.json),
[inventory](software/pixel10pro-vulkan/INVENTORY.json),
[Vulkan capabilities](software/pixel10pro-vulkan/VKJSON.json),
[build provenance](software/pixel10pro-vulkan/BUILD_PROVENANCE.json).

## 2026-09-22 19:58 UTC - Pixel Tensor TPU round-trip qualification

User requested a TPU round-trip test and confirmed no Tensor SDK is installed.
The public LiteRT2.2.0 runtime and GoogleTensor dispatch library can run Google's
precompiled P25 add sample without the compiler package. This qualifies a real
TPU request/response path. It is not a Qwen FFN benchmark and cannot establish
Qwen TPU speed, full-model correctness or energy savings.

### Setup and execution proof

- NNAPI inventory **PASS** at 19:48:09 UTC: `google-edgetpu` type4 accelerator,
  version2.0, feature1000008; CPU reference also present. Standard FP32 add
  support on that TPU driver **FAIL** (`fp32_add_supported=0`). No NNAPI inference
  was substituted for a TPU result. Other NNAPI operators were not qualified.
- Built small isolated `litert_tpu_probe.cpp` and `nnapi_inventory.cpp` with
  NDKr28b, Android arm64/API31, `-O2 -Wall -Wextra -Werror`; **PASS**. Runtime
  from official LiteRT2.2.0 Maven AAR and dispatch from its GitHub release;
  downloads and hashes retained in `software/pixel10pro-tpu/`.
- Model: Google `simple_add_op_google_tensor_p25_precompiled.tflite`, two
  float32 tensors of shape `[1,128,1]`, elementwise add, one same-shaped output.
  Application payload is 1024 bytes input and 512 bytes output. The dedicated
  probe uses an 8-byte request header and 24-byte response header; this is not
  the FFN protocol-v6 geometry or payload.
- NPU-only acceleration selected. Model inspection requires custom dispatch
  ops; the log shows `libLiteRtDispatch_GoogleTensor.so` loading the phone's
  `libedgetpu_litert.so`, a SouthBound context, and all1/1 graph nodes delegated
  to DispatchDelegate. Buffers are AHWB. CPU registration in the environment
  log is not CPU execution: the entire model is TPU dispatch, with no CPU or
  GPU fallback selected.
- Phone remains unprivileged, stock kernel, no reboot or system-setting change.
  All runs use the shared flock and ADB5037. Isolated rig root:
  `/mnt/storage/s42-pixel10pro-tpu-20260922-v1`; phone root:
  `/data/local/tmp/s42-pixel10pro-tpu-20260922-v1` with a `coalesced/` subdirectory
  for the second binary. Each worker has a finite request count and exits normally.

Standalone smoke **PASS**, 19:52:19 UTC: 20/20 calls correct, normal exit.
Excluding its first call, median runtime invocation1.625 ms and phone worker
1.836 ms. These timings exclude USB/ADB entirely.

### Round-trip results and response-framing experiment

All three arms **PASS**: 60 calls each, ten warmups excluded from timing,
50 measured calls each, mathematically exact outputs and exact repeated
outputs. The same ten input pairs cycle six times in each arm. All180 saved
input/output pairs and all deployed binary/runtime/model hashes were audited.
Both endpoints use TCP_NODELAY and one persistent TCP connection through ADB.
Only the response packing changes between binaries: separate header/payload
writes versus one preassembled response write. Restoring the original binary
reproduces the large delay.

| Arm | Finished UTC | Median invocation ms | Median phone worker ms | Median outside worker ms | Median round trip ms | p90 round trip ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A: separate header/payload writes | 19:54:54 | 3.140 | 3.746 | 46.334 | 50.003 | 53.008 |
| B: one response write | 19:56:36 | 1.356 | 1.689 | 2.870 | 4.666 | 5.203 |
| A: original binary again | 19:57:20 | 3.082 | 3.649 | 46.258 | 50.070 | 52.957 |

B improves median round trip by10.72x relative to initial A on this sample.
The A/B/A experiment supports response framing as a cause of the large delay.
A TCP/ADB buffering or delayed-ACK interaction is plausible but **NOT VERIFIED**
without packet-level evidence. Invocation times also change with request cadence;
no fixed-frequency or thermal controls were imposed.

Additive mean latency breakdown for arm B:

| Stage | Mean ms |
| --- | ---: |
| LiteRT TPU invocation | 1.499 |
| Input/output buffer access and copies | 0.332 |
| Outside the phone worker | 2.863 |
| Complete desktop round trip | 4.694 |

Invocation includes runtime dispatch and synchronization, not just TPU kernel
execution. Buffer stage is worker minus invocation. Outside-worker time includes
request/response framing, validation, host scheduling and TCP/ADB/USB both ways;
it is not pure wire time. Independently calculated medians above need not sum.

### Limits and next work

The Qwen FFN test is **NOT RUN**: its weights/graph have not been compiled for
Tensor G5. The public runtime is sufficient for the precompiled sample; Google's
[Tensor SDK setup](https://developers.google.com/edge/litert/next/tensor-sdk)
requires beta access for the compiler package used by the
[Google Tensor AOT backend](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/ci/tools/python/vendor_sdk/google_tensor/setup.py).
User confirmed no SDK installation. A matched FFN comparison needs that compiler,
conversion of the same layer/weights, complete TPU delegation, and numerical
comparison with the existing CPU receipts before a performance claim.

The production TCP FFN worker also writes its response header and payload
separately (`examples/layersplit/ffn-split-worker.cpp`, around2107). Coalescing
that response is a concrete follow-up candidate for the previous40-46 ms
outside-worker delay. It was **NOT APPLIED OR VERIFIED** for FFN, OP11, OP15 or
larger1/2/4-row payloads in this task. No production/scheduler source changed.
Wi-Fi, FFN TPU latency, sustained behavior, full-model tokens, energy and
multi-phone scheduling remain **NOT VERIFIED**. Current energy records unchanged.

Cleanup **PASS**, 19:58:34 UTC: no qualification worker or owned forward,
boot unchanged, no force kills. Isolated files retained for reproduction.
Evidence: [assessment](PIXEL10PRO_TPU_ASSESSMENT.json),
[best arm](physical/pixel10pro-tpu-1/run2-tcp-coalesced/RESULT.json),
[initial arm](physical/pixel10pro-tpu-1/run1-tcp/RESULT.json),
[original-binary repeat](physical/pixel10pro-tpu-1/run3-tcp-split-repeat/RESULT.json),
[cleanup](physical/pixel10pro-tpu-1/CLEANUP.json),
[build provenance](software/pixel10pro-tpu/BUILD_PROVENANCE.json),
[download provenance](software/pixel10pro-tpu/DOWNLOADS.json).

## 2026-09-22 20:38 UTC - Pixel FFN TPU path

The user asked whether Pixel can help the server execute FFN work as OP15 does.
The same mathematical split applies: retain the selected FFN weights on the
phone, send each decode activation batch, and return the phone's partial result
for addition to the desktop's complementary columns. Attention and KV ownership
stay with the desktop. Existing Vulkan qualification proves the real Qwen FFN
can use this protocol on Pixel, but does not qualify its TPU implementation.

| Requirement | Evidence and current status |
| --- | --- |
| Real Qwen FFN request/response on Pixel | Vulkan functional PASS: layer18, 12 calls/28 rows, maximum relative L2 0.000311882. Median round trips 103.110/142.184/201.093 ms for 1/2/4 rows. |
| Real Tensor TPU execution and USB/ADB round trip | Add microbenchmark PASS: 180/180 exact TCP outputs; best median 4.666 ms, p90 5.203 ms. Different graph and payload from Qwen. |
| Qwen graph compiled for Tensor G5 | NOT RUN; compiler SDK absent. Operator coverage and compilation success are unverified. |
| Server decode using Pixel TPU; exact output tokens; energy saving | NOT RUN. Neither existing qualification establishes these results. |

The native reference is `examples/layersplit/ffn-split-worker.cpp`'s `build_graph`:
gate/up matrix multiplies, SwiGLU, down projection, and a sum of 512-column
block results. Partial widths select the tail blocks of the resident shard.
Conversion must preserve those weights, column selection and accumulation
semantics; selecting a different prefix or silently quantizing weights would
change the comparison. Compiler precision and any arithmetic differences need
explicit numerical qualification.

The first implementation and test should be bounded as follows:

1. Export Qwen layer18's final 512 FFN columns from the verified F16 shard,
   K=5120, starting with one activation row. This has 15 MiB of source weights
   and 10 KiB of F16 input/output payload each way. These are calculated sizes,
   not measured TPU memory requirements. Preserve parent/shard hashes and record
   the converted graph, compiler version/options and compiled model hashes.
2. Compile for Tensor G5 and run it through the already qualified LiteRT runtime.
   Require complete TPU execution evidence and compare each returned row with
   the matching CPU worker. Record per-row relative L2, NMSE, absolute error and
   repeatability. The existing diagnostic gate is relative L2 <= 0.01; passing
   it is not a substitute for exact full-model output tokens.
3. Expose the compiled graph through the existing FFN protocol-v6 HELLO/EXEC
   contract, including artifact/layer/width/row checks, identities and payload
   hashes. Keep weights and compiled state resident. Start with max_tokens=1
   and one qualified width. Use one assembled response write and measure its
   behavior at real FFN payload sizes; the add experiment alone does not prove
   the larger-payload improvement. Reuse existing clients and qualification
   tooling rather than adding a scheduler or changing the wire protocol.
4. After the narrow graph passes, qualify the full 17408-column layer and all
   row counts 1 through 4 before advertising max_tokens=4. A profile qualified
   only at rows1/2/4 must not silently accept untested row3. The full layer has
   510 MiB of F16 source weights; compiled residency remains unmeasured.
5. Only then test one server using Pixel as its helper, with matched baseline
   tokens, latency and host energy. Preserve the dormant host-release guard
   and obtain new transport/runtime receipts. Simultaneous OP15+Pixel ownership
   remains the separate multi-phone integration described earlier in this report.

Google's [Tensor SDK documentation](https://developers.google.com/edge/litert/next/tensor-sdk)
lists Tensor G5 support and beta access for model compilation. Its
[NPU documentation](https://developers.google.com/edge/litert/next/npu)
specifies AOT execution and says Google Tensor does not yet support on-device
JIT compilation. The pinned LiteRT2.2.0
[SDK installer](https://github.com/google-ai-edge/LiteRT/blob/v2.2.0/ci/tools/python/vendor_sdk/google_tensor/setup.py)
requires a supplied SDK archive or download URL. The public dispatch/runtime
libraries used by the add test do not supply that compiler. SDK access can be
requested using Google's [beta form](https://services.google.com/fb/forms/tensor_ml_sdk_experimental_access/).
No form was submitted or SDK terms accepted on the user's behalf.

This review made documentation changes only. It did not run a new benchmark,
change server/scheduler behavior, or establish a Pixel TPU FFN performance or
energy result. Compilation is the immediate prerequisite; the arithmetic,
protocol and server acceptance gates remain outstanding.

## 2026-09-22 20:55 UTC - Pixel Vulkan FFN server trial

The user selected Pixel GPU assistance. Native `S41_SERVER_FFN_TRANSPORT=tcp`
already permits the Vulkan worker as a helper. This first test uses one server
and Pixel alone; it does not implement device ownership in the multi-phone
scheduler. Desktop attention/KV and the mixed-policy dormant-release guard
remain unchanged. All physical work is serialized under the shared flock,
using ADB5037 and isolated files under
`/mnt/storage/s42-pixel10pro-server-20260922-v1`.

### Worker tuning and qualification

All arms use the verified Qwen layer18 F16 shard, K5120,17408 columns,
rows1/2/4. The existing `--column-quantum 4352` option reduces the graph from
34 weight blocks to4, retaining510MiB of weights. Each qualification has
12calls/28rows and three warm timing samples per row count.

| Arm | Numerical result | Median RPC ms, rows1/2/4 | Median worker ms, rows1/2/4 |
| --- | --- | --- | --- |
| Earlier quantum512, split response | PASS, max relative L2 0.000311882 | 103.110 / 142.184 / 201.093 | 59.600 / 95.484 / 188.701 |
| Quantum4352, split response,20:48 UTC | PASS, max relative L2 0.000311957 | 67.663 / 49.669 / 73.835 | 27.007 / 43.289 / 65.360 |
| Quantum4352, coalesced response,20:51 UTC | PASS, max relative L2 0.000311957 | 35.008 / 44.600 / 78.942 | 30.736 / 39.113 / 70.054 |

The TCP worker now places its unchanged response header and payload in a reused
buffer for one `write_exact` call. All12 saved outputs are bit-identical between
the two4352-column arms, with identical inputs. One-row median outside-worker
time falls43.701->4.293ms. This extends the earlier add experiment to a real
10KiB FFN response. Multi-row timing varies, and no fixed-frequency/thermal
controls were applied; these short arms do not establish sustained performance.
CPU round-trip medians in the coalesced arm are18.239/22.013/19.007ms, so isolated
phone calls remain slower than the desktop reference.

Build **PASS** with NDKr28b/API28 and `-Wall -Wextra -Werror`; existing Vulkan
libraries reused. The source change is in `examples/layersplit/ffn-split-worker.cpp`,
deployed only to `/data/local/tmp/s42-pixel10pro-ffn-coalesced-20260922-v1`.
Old binaries and original source snapshot are retained. Both qualifications
exit0 normally, remove their owned forwards and preserve the phone boot ID.
No shared server rebuild or scheduler deployment was needed; the new standalone
receipts bind the changed worker. Evidence:
[block-size result](physical/pixel10pro-server-1/run1-quantum4352/RESULT.json),
[coalesced result](physical/pixel10pro-server-1/run2-coalesced4352/RESULT.json),
[build provenance](software/pixel10pro-ffn-coalesced/BUILD_PROVENANCE.json).

### Server comparison method

`qualify_pixel_server.py` reuses the existing HTTP client, decode-boundary
control and RAPL/NVML energy sampler. The unchanged server binary retains
`S41_SERVER_FFN_SPLIT=ON` and16 FFN markers. Qwen3-14B uses16 desktop GPU layers,
8 decode threads,24 prefill threads,4096 context,512 batch/ubatch, one slot and
flash attention disabled. Pixel owns only layer18, with a4352-column quantum.
One256-token prompt is generated for64 greedy tokens with seed17 and prompt
caching disabled in each arm: desktop,50% of layer18 columns,100%,desktop.
Both helper policies start through the first-output callback, so the recorded
acknowledgement determines actual assisted coverage.

The worker has a finite128-call limit. After all requests complete, the idle
server shuts down; bounded synthetic calls consume any remaining worker budget
outside the measurement interval so it exits normally. No worker is killed.
Host request/decode energy and output tokens are saved separately for every arm.
Pixel energy is not measured or silently estimated. Startup is excluded from
request energy; this measures resident-helper operation, not campaign amortization.
Server correctness, latency, host saving and cleanup are pending at this entry.

### One-layer server result,20:58 UTC: PASS

All four64-token streams match exactly, independently checked against raw SSE.
Both controls have zero phone calls. The two phone policies each acknowledge at
token2 and execute62 layer18 calls,124 total, matching every expected row.
The server records only decode calls, no phone prefill. Raw RAPL/NVML samples
independently reproduce every request/decode energy value; sampler diagnostics
contain no errors. This is a functioning llama.cpp server using Pixel's Vulkan
GPU for a real FFN slice, with the rest of Qwen executing on the desktop.

| Arm | Request s | Decode s | Measured host request J | Measured host decode J |
| --- | ---: | ---: | ---: | ---: |
| Desktop control before | 43.103 | 38.798 | 4803.106 | 4577.036 |
| Pixel,50% of layer18 FFN columns | 41.977 | 39.345 | 4739.356 | 4559.249 |
| Pixel,100% of layer18 FFN columns | 42.232 | 39.598 | 4628.738 | 4448.164 |
| Desktop control after | 41.163 | 38.523 | 4870.426 | 4690.393 |

Full-width request saving is3.630%/4.962% against the individual controls,
4.301% against their mean. Decode saving is2.816%/5.164%,mean4.004%, while
decode duration increases2.062%/2.791%. Half-width mean savings are2.014%
request and1.607% decode, with1.410%/2.134% longer decode. These percentages
apply to one selected layer, not all FFNs. The first control prefill is colder:
4.305s versus2.632-2.640s for subsequent requests. Decode-only comparison avoids
that prefill imbalance, but neither comparison establishes repeatability.
Pixel energy remains unmeasured; no fleet or full-trace saving is claimed.

Server shape statistics show mean RPC25.090ms for8704 columns and33.108ms for
17408. Mean overlapping desktop FFN branches are9.275ms and0.006ms respectively.
Host dormant-release proof is retained in the native log. Cleanup **PASS**:
server and worker exit0, four zero-input filler calls after measurement complete
the finite128-call worker budget, owned forward removed, boot unchanged.

Evidence: [server result](physical/pixel10pro-server-1/run3-server/RESULT.json),
[independent audit](physical/pixel10pro-server-1/ONE_LAYER_AUDIT.json),
[native log](physical/pixel10pro-server-1/run3-server/server.log),
[configuration](PIXEL_SERVER_CONFIG.json),
[analysis script](analyze_pixel_server.py).

### Six-layer follow-up,21:04 UTC: qualification PASS; server retry pending

Qwen layers18-23 all fit together:3060MiB F16 weights. Qualification **PASS**,
72calls/168rows, maximum relative L2 0.000326951, exact repeats and normal exits.
Median round trips33.673/48.526/77.804ms at rows1/2/4,18 warm samples per shape
across the six layers. [Qualification result](physical/pixel10pro-server-1/run4-six-layer-qualification/RESULT.json).

Run5 server startup **FAIL** before either process launch: its port guard
reported26978 occupied immediately after qualification. That guard checked all
TCP states, including closing/TIME_WAIT entries. The exact state at rejection
was not archived, so the TIME_WAIT explanation is plausible rather than proven;
the later inventory finds no matching socket or FFN worker. The guard now checks
LISTEN sockets and saves process/socket inventory. Run6 uses a new output
directory and fresh phone port26982, with the same request method,6-layer mask
and finite768-call budget. It is pending under the shared lock.

### Six-layer server result,21:09 UTC: PASS

Run6 finished21:07:50 UTC with all four64-token outputs identical. Independent
raw-SSE audit **PASS**; per-layer call coverage **PASS**; raw-sample energy
recomputation **PASS**. The half/full policies each acknowledge at token2 and
produce62 calls on each of layers18-23,372 per policy and744 total. Both controls
make zero phone calls. The same desktop server/library hashes are used as in
the one-layer test. Pixel's Vulkan worker and all library hashes match the
archived isolated build and numerical qualification.

| Arm | Request s | Decode s | Measured host request J | Measured host decode J |
| --- | ---: | ---: | ---: | ---: |
| Desktop control before | 41.214 | 38.509 | 4800.048 | 4615.058 |
| Pixel,50% of each selected FFN | 44.702 | 42.068 | 4329.807 | 4149.684 |
| Pixel,100% of each selected FFN | 48.164 | 45.478 | 4305.035 | 4119.223 |
| Desktop control after | 41.162 | 38.480 | 4892.714 | 4707.501 |

Percentages refer to columns in six selected layers, not a fraction of the
whole model. Baseline mean host request/decode energy is4846.381/4661.279J.

| Pixel split | Request saving vs each control | Request saving vs control mean | Decode saving vs control mean | Decode slowdown vs each control |
| --- | --- | ---: | ---: | --- |
| 50% | 9.797% / 11.505% | 10.659% | 10.975% | 9.242% / 9.324% |
| 100% | 10.313% / 12.011% | 11.170% | 11.629% | 18.099% / 18.187% |

For this short measurement,50% provides almost the full-width host saving
with less latency cost. This is one sample per split bracketed by two controls,
one synthetic prompt and one active server slot. It is not a statistical estimate,
a24-request trace, a25% saving result, a multi-slot server qualification or an
OP15+Pixel experiment. Phone energy is unmeasured, so fleet saving is unknown.
Model/worker startup is outside the request measurement; the result measures
resident-helper operation. Full scheduler admission, cost calibration, device
ownership and two-phone dispatch remain unimplemented for Pixel.

Cleanup **PASS**,21:09:03 UTC under the shared lock: server/worker exited0,
24 zero-input calls outside the measured intervals completed the finite768-call
worker budget. No owned phone worker, desktop server or ADB forward remains.
Boot unchanged, Pixel USB still5000M. OP15, shared server deployment and phone
kernel/settings were not changed. Run5's pre-launch failure is retained.

Evidence: [server result](physical/pixel10pro-server-1/run6-six-layer-server/RESULT.json),
[independent audit](physical/pixel10pro-server-1/SIX_LAYER_AUDIT.json),
[native log](physical/pixel10pro-server-1/run6-six-layer-server/server.log),
[cleanup](physical/pixel10pro-server-1/CLEANUP.json),
[configuration](PIXEL_SERVER_SIX_LAYER_RETRY_CONFIG.json),
[run command](physical/pixel10pro-server-1/RUN_SIX_RETRY.sh).

Recompute the numerical/energy audit from the repository root:

```sh
PYTHONPATH=. python3 research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/analyze_pixel_server.py research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/physical/pixel10pro-server-1/run6-six-layer-server
```

Final source checks: Android worker build with warnings as errors **PASS**;
pyflakes on both new Python scripts **PASS**; native before/after diff contains
only the TCP response-frame packing change. Existing FunctionFS execution
code and the dormant-release rule are retained. No commit or push.

### Six-layer latency accounting, 2026-09-22 21:26 UTC: PASS

Derived from the saved run6 logs and request timestamps; no hardware rerun.
Qwen3-14B layers 18-23, one active slot, 256 prompt tokens, 64 output tokens,
Pixel Vulkan over TCP through an ADB USB forward. Each offload setting has
372 single-row calls: six layers for each of 62 assisted tokens. The first
two output tokens are local. Percentages refer to columns in those layers.
All values below are means, so the component arithmetic is valid.

| Per FFN call | Pixel 50% | Pixel 100% |
| --- | ---: | ---: |
| Phone worker execution | 21.058 ms | 29.381 ms |
| Transport and runtime outside worker timer | 4.968 ms | 4.605 ms |
| Total RPC round trip | 26.026 ms | 33.987 ms |
| Outer client handling, derived as launch-to-join minus RPC | 0.191 ms | 0.180 ms |
| Host branch running concurrently | 9.580 ms | 0.014 ms |
| Server wait after host branch | 16.637 ms | 34.153 ms |
| Helper launch to join | 26.217 ms | 34.167 ms |

Phone worker plus outside-worker time equals RPC time. Host branch plus wait
equals launch-to-join, within 0.000001 ms log rounding. These are overlapping
views of the same interval: do not add the host branch to the RPC total.
Summed across six selected layers, launch-to-join is 157.300/205.003 ms per
assisted token at half/full columns; exposed server wait is 99.823/204.920 ms.

| Whole request | Desktop control mean | Pixel 50% | Pixel 100% |
| --- | ---: | ---: | ---: |
| Time to first token | 2.694 s | 2.634 s | 2.686 s |
| Measured decode interval | 38.494 s | 42.068 s | 45.478 s |
| Total request | 41.188 s | 44.702 s | 48.164 s |
| Decode interval / 64 output tokens | 601.475 ms | 657.310 ms | 710.601 ms |
| Decode slowdown vs control mean | 0% | 9.283% | 18.143% |

The worker timer includes layer lookup, graph construction, GPU input upload,
graph execution and output download. Pure GPU kernel time is unmeasured.
Outside-worker time combines transport, framing, copies, checks and scheduling;
upload/download wire latency is not separately measured. Launch-to-join excludes
initial host input tensor extraction and final output tensor publication;
whole-request timing includes them. Model/worker startup is outside the run.
These are one sample per split and two bracketing controls, not a repeatability
estimate. Phone-side execution accounts for 80.9%/86.4% of RPC time.

Evidence: [derived timings and source hashes](PIXEL_SERVER_LATENCY_BREAKDOWN.json),
[native timer records](physical/pixel10pro-server-1/run6-six-layer-server/server.log).

### Server time, overlap and Pixel execution rate, 2026-09-22 21:34 UTC: PASS

Existing-run arithmetic and source inspection, no new hardware test.

| Decode component, 64 outputs | Pixel 50% | Pixel 100% |
| --- | ---: | ---: |
| Server decode outside measured phone join waits | 35.879 s | 32.773 s |
| Measured phone join waits | 6.189 s | 12.705 s |
| Total decode | 42.068 s | 45.478 s |

The server component is wall time minus explicit phone waits. It includes CPU,
GPU, synchronization, tensor copies and other overhead; those components were
not separately timed. It is 560.606/512.085 ms per output token, compared with
96.704/198.516 ms per output token in measured phone waits.

Pixel uses the existing asynchronous FFN client thread, as OP15 does. The
phone job starts at ffn_norm; the desktop computes its column share before
joining at ffn_phone_partial and adding the partial outputs. At 50%, the host
branch lasts 9.580 ms and the phone RPC 26.026 ms, leaving exposed wait. At
100%, build_dense_ffn_split removes the host FFN branch. Qwen's next layer
depends on the combined FFN output, so its later-layer computation cannot run
ahead to cover that wait in this one-slot run. Source inspection confirms the
overlap mechanism; these data do not provide a matched OP15 speed comparison.

| Execution rate | Pixel 50% | Pixel 100% |
| --- | ---: | ---: |
| Worker time per FFN layer | 21.058 ms | 29.381 ms |
| Worker time for all six selected FFNs per assisted token | 126.345 ms | 176.288 ms |
| Worker layer calls per second | 47.489 | 34.035 |
| Effective matrix throughput | 12.698 GFLOP/s | 18.201 GFLOP/s |
| Whole-model output rate during decode | 1.521 tokens/s | 1.407 tokens/s |

Desktop control mean is 1.663 output tokens/s. Matrix throughput counts the
three dense products as 6 * 5120 * columns FLOPs and divides by the worker
interval. It excludes non-matrix operation counts but includes graph setup
and copies in elapsed time; it is not a GPU peak rating or pure-kernel result.
One sample per split; runtime/model startup remains outside these intervals.

### Pixel phone-internal latency profile, 2026-09-22 21:58 UTC: PASS

The earlier server worker timer combined graph construction, GPU buffer access
and synchronous graph execution. A bounded diagnostic now separates them.
The run finished at 21:55:50 UTC under the shared rig lock. It keeps Qwen
layers 18-23 resident together with the same F16 weights, 4352-column blocks,
one input row and 8704/17408 selected columns, alternating the width order.
Inputs are seeded synthetic activations, not saved server activations.

Sequence: CPU reference, original Pixel worker, isolated stage-timer worker,
stage-timer worker with existing Vulkan per-operation profiler, original worker.
Every arm executes 96 calls. Each mean below uses 36 calls per width across
six layers after discarding the first two of eight repeats per layer/width.
All 384 Pixel outputs are bit-identical across arms and repeats. Every CPU
comparison passes the existing 1% relative-L2 limit; maximum 0.000325483.

**Wall-clock stages without the per-operation GPU profiler:**

| Phone stage, one FFN layer / one token | 50% columns | 100% columns |
| --- | ---: | ---: |
| Layer lookup, graph construction and allocation | 0.158 ms | 0.240 ms |
| Input buffer write | 0.007 ms | 0.007 ms |
| Vulkan execution, dispatch and synchronization | 18.755 ms | 32.646 ms |
| Output buffer read | 0.287 ms | 0.275 ms |
| Phone worker total | 19.207 ms | 33.168 ms |
| Execution share of worker total | 97.644% | 98.425% |
| Transport/runtime outside worker timer | 4.348 ms | 4.359 ms |
| Complete RPC | 23.555 ms | 37.527 ms |

The stages sum exactly in integer microseconds for all 96 calls and match the
worker interval returned in every protocol response. ggml_backend_graph_compute
includes backend synchronization. Buffer access timers include relevant mapping,
flush/invalidate or copy work but do not distinguish those operations.

**Separate GPU timestamp diagnostic:**

| GPU timestamp category | 50% columns | 100% columns |
| --- | ---: | ---: |
| Gate/up matrix-vector projections | 14.482 ms | 22.413 ms |
| Down projection and fused partial-result addition | 0.484 ms | 7.051 ms |
| All matrix-vector intervals | 14.966 ms | 29.464 ms |
| SwiGLU activation | 3.058 ms | 2.654 ms |
| Tensor conversions/copies inside graph | 0.712 ms | 0.694 ms |
| Total GPU timestamp intervals | 18.736 ms | 32.812 ms |
| Matrix-vector share of GPU timestamp intervals | 79.878% | 89.797% |

These are the backend profiler's operation intervals, which can include barriers
and scheduling effects, not independent pure shader arithmetic measurements.
The profiler inserts timestamps/barriers and prints timings inside execution.
Its complete phone worker time rises to 24.184/38.207 ms, 29.952%/15.700%
above the mean of the bracketing original-worker controls. Therefore this
operator table is a diagnostic view and must not be added to, or substituted
into, the unprofiled wall-stage table as an exact decomposition.

| Worker mean, same new workload | 50% columns | 100% columns |
| --- | ---: | ---: |
| Original before | 18.224 ms | 33.164 ms |
| Stage timers only | 19.207 ms | 33.168 ms |
| Stage timers plus GPU profiler | 24.184 ms | 38.207 ms |
| Original after | 18.995 ms | 32.882 ms |

Stage timers differ +3.211%/+0.440% from control means. Clocks, thermals and
background activity were not controlled, so these differences are observed
changes, not precise causal overhead estimates. The earlier server's
21.058/29.381 ms worker means remain its result. These new measurements use
different inputs and call cadence and do not replace those server timings.
The matrix-vector work covers 255/510 MiB of resident F16 weights per FFN;
activation I/O is 10 KiB each way. GPU memory stalls versus arithmetic or
driver scheduling were not separately measured. No phone energy measurement.

Only an isolated worker source snapshot adds timing calls and one log line;
the production worker, scheduler and shared server are unchanged. Warnings-as-
errors build and pyflakes **PASS**. Runtime hashes match the prior server run.
All five workers exit normally after finite budgets, owned forwards are removed,
no Pixel FFN worker remains, and boot ID is unchanged: cleanup **PASS**.

Evidence: [derived stage/operator table and source hashes](PIXEL_PHONE_STAGE_BREAKDOWN.json),
[physical result](physical/pixel10pro-profile-1/run1/RESULT.json),
[run command](physical/pixel10pro-profile-1/RUN.sh),
[stage log](physical/pixel10pro-profile-1/run1/stage-timers/worker.log),
[GPU timestamp log](physical/pixel10pro-profile-1/run1/gpu-profile/worker.log),
[isolated timing diff](software/pixel10pro-profile/STAGE_TIMERS.patch),
[build identity](software/pixel10pro-profile/BUILD_PROVENANCE.json).

Recompute from repository root:

```sh
python3 research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/analyze_pixel_profile.py research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/physical/pixel10pro-profile-1/run1 --output /tmp/pixel-phone-stage-breakdown.json
```

### Pixel kernel throughput and next experiments, 2026-09-22 22:20 UTC

Arithmetic **PASS**, using the existing 36 warm calls per width. Count two
FLOPs per multiply-add: one row of the three dense FFN products costs
6 * 5120 * selected_columns FLOPs. Activation, conversion and partial-add
operation counts are excluded. Full-width results:

| Matrix operation group | GFLOP per layer/token | GPU timestamp mean | Effective GFLOP/s |
| --- | ---: | ---: | ---: |
| Gate and up | 0.356516 | 22.413 ms | 15.907 |
| Down, including fused partial-add interval | 0.178258 | 7.051 ms | 25.281 |
| All three products | 0.534774 | 29.464 ms | 18.150 |

The half-width aggregate is 17.867 GFLOP/s. Dividing matrix FLOPs by the
separate stage-only worker interval gives 13.921/16.123 GFLOP/s at half/full
width, including graph construction and buffer access in the denominator.
The timestamp profiler changes execution (observed full-width worker +15.7%);
these are diagnostic achieved rates, not isolated shader or hardware peak
measurements. The unusually short half-width down interval should not be used
as a peak-performance claim. [Reproducible arithmetic](PIXEL_KERNEL_GFLOPS.json).

Candidate order for reducing one-row latency, all **UNVERIFIED**:

1. Test `--column-quantum 8704` against 4352 with the existing worker. This
   reduces full-width blocks from four to two. Each F16 matrix is 85 MiB,
   below the captured 128 MiB Vulkan storage-buffer range; a single full-width
   matrix would be 170 MiB. Weight volume is unchanged, so a 2x speedup is not
   implied. Changed partial-sum grouping requires numerical and token checks.
2. Assess use of the existing backend graph optimizer. The worker calls
   `ggml_backend_graph_compute` directly; only the backend scheduler invokes
   the graph-optimization hook. Vulkan's hook can reorder independent nodes
   while preserving fusion patterns. This is a source finding, not evidence
   of a measured speedup. Allocation and dependency ordering must remain valid.
3. Tune the F16 matrix-vector workgroup and output rows per group on PowerVR.
   The current path already uses aligned vector loads; do not treat the older
   scalar-load issue as an unfixed local bug. The captured default subgroup is
   128. PowerVR's [occupancy guidance](https://docs.imgtec.com/performance-guides/compute-recommendations/html/topics/utilisation/maximise-utilisation.html)
   recommends 128 for Volcanic and testing workgroup variants while controlling
   register/shared-memory pressure. Smaller subgroups are not inherently faster.
4. Quantized phone weights or further projection/activation fusion are larger
   experiments. A full FFN uses 510 MiB of F16 weights for only 0.535 GFLOP,
   so reducing weight traffic is plausible; memory bandwidth versus driver or
   execution stalls has not been established with counters. Numerical behavior
   would need requalification.

For throughput, batching is already measured in the separate six-layer
qualification: median worker time is 29.807 ms for one row, 43.651 ms for two,
and 70.100 ms for four. Four rows cost 17.525 ms per row, 41.204% less than
the one-row call. This improves throughput while increasing call latency;
real-server Pixel co-tenant dispatch is still unverified.

No production changes, hardware run, energy result or new token-equivalence
claim in this analysis. Live backend/shader hashes match the source archived
with the tested Vulkan library.

### Pixel matrix shapes and precision audit, 2026-09-22 22:27 UTC: PASS

The profiled operation is W[M,K] * X[K,N] with N=1 decode row. The full
FFN width 17408 is split into four blocks of 4352 columns; half width uses
two blocks. These are the actual per-block dispatches:

| Projection | Weight shape | Activation shape | Output shape | Full / half calls |
| --- | --- | --- | --- | ---: |
| Gate | [4352,5120] | [5120,1] | [4352,1] | 4 / 2 |
| Up | [4352,5120] | [5120,1] | [4352,1] | 4 / 2 |
| Down | [5120,4352] | [4352,1] | [5120,1] | 4 / 2 |

The four full-width down outputs are summed. Each block has SwiGLU between
gate/up and down. Logical unsplit gate/up weights would be [17408,5120]
and down [5120,17408]. Six resident layers are evaluated separately, not
as six rows of the same matrix multiplication. GGML stores its innermost
dimension first, so tensor ne[0],ne[1] reverses these displayed weight axes.

Weights are F16, input wire data is cast to F32, intermediates are F32, and
the GEMV shader uses float/vec4 arithmetic and accumulation. The output is
cast back to F16 for the wire. The Vulkan device log reports matrix cores
none; no TPU/matrix-core path is used. Source: worker build_graph and weight
creation, vulkan-shaders-gen.cpp base_dict and mul_mat_vec.comp aligned path.

Each block projection has 44,564,480 FLOPs and 44,564,480 bytes (42.5 MiB)
of weights: approximately 1 FLOP per weight byte at N=1. Full-width work is
0.534774 GFLOP over 510 MiB of weights. Dividing this nominal weight volume
by the 29.464 ms summed matrix timestamps gives 18.150 GB/s effective weight
throughput, not a hardware DRAM counter or measured bandwidth ceiling.
The [matrix arithmetic-intensity model](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
explains why single-vector products have little data reuse and cannot be
judged against large-GEMM compute peaks. It does not establish the cause
of this Pixel implementation's current rate: dispatch, barriers, reduction,
driver/clock behavior and memory traffic are not separately resolved.
Standalone GEMV and sustained GPU-memory measurements remain unverified.
Increasing column quantum still leaves N=1; batching increases N and permits
weight reuse. This audit adds no new hardware run or performance result.

### Complete Pixel FFN operation-shape audit, 2026-09-22 22:33 UTC: PASS

Checked all 96 recorded GPU-profile calls (48 per selected width), including
warmup requests after worker readiness. Every matrix dimension and operation
count matches the source graph. Non-matrix shapes follow ggml construction;
no new runtime tensor dump or physical run was added. B=1 in this profile.

| Operation | Input(s), mathematical axes | Output | Full-width logical count |
| --- | --- | --- | ---: |
| Input cast | F16 [5120,B] | F32 [5120,B] | 1 |
| Gate | F16 W[4352,5120] * F32 X[5120,B] | F32 [4352,B] | 4 |
| Up | F16 W[4352,5120] * F32 X[5120,B] | F32 [4352,B] | 4 |
| SwiGLU | silu(gate) * up, two F32 [4352,B] | F32 [4352,B] | 4 |
| Down | F16 W[5120,4352] * F32 H[4352,B] | F32 [5120,B] | 4 |
| Partial-result addition | two F32 [5120,B] | F32 [5120,B] | 3 |
| Output cast | F32 [5120,B] | F16 [5120,B] | 1 |

SwiGLU uses separate gate/up inputs and preserves width; it does not halve
4352 again. Each partial addition is fused into a down dispatch in this log,
so physical counts are 2 casts, 8 gate/up, 4 GLU, 1 plain down, 3 down-plus-add.
Half width has 2 blocks: 2 casts, 4 gate/up, 2 GLU, 1 plain down, 1 down-plus-add.
There are no bias terms on this split path. FFN RMSNorm (scale [5120]),
host/phone output combination for partial assistance, and final residual
addition remain on the server; their activation shape is [5120,B].

Evidence: [complete shape manifest and source hashes](PIXEL_FFN_SHAPES.json),
worker build_graph, ggml_mul_mat/ggml_glu_impl, and Qwen3 server graph.

### Pixel per-operation speed and latency, 2026-09-23 00:29 UTC: PASS

The final intact-graph run finished at 00:29:16 UTC under the shared rig lock.
All 384 phone outputs are bit-identical across four arms and match the CPU
reference within the existing 1% relative-L2 limit; maximum 0.000325483.
Six resident Qwen layers 18-23, one row, 4352-column blocks, half/full widths
interleaved. Each reported width has 36 warm FFN calls across six layers;
the first two of eight repeats per layer/width are excluded. Projection and
SwiGLU rows therefore have 144 samples at full width. Weight loading and
startup are excluded. These are synthetic activations, not server replay.

A one-line change in an isolated Vulkan library appends tensor names to the
existing timestamp logger. All 132 original shader objects are reused.
The named worker executes the intact graph for these arms; single-operation
mode is disabled. No production source or shared rig deployment changes.

**Intact graph with normal down/add fusion:** GPU timestamp intervals below
include profiler barriers and queue/driver scheduling effects. They are not
independent pure shader arithmetic measurements.

| Operation | Calls / FFN | Mean / call | p90 / call | Total / FFN | Effective rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input F16 -> F32 cast | 1 | 0.638 ms | 0.953 ms | 0.638 ms | 8.031 Melement/s |
| Gate projection | 4 | 2.848 ms | 4.805 ms | 11.392 ms | 15.648 GFLOP/s |
| Up projection | 4 | 2.731 ms | 4.774 ms | 10.925 ms | 16.317 GFLOP/s |
| SwiGLU | 4 | 0.609 ms | 1.537 ms | 2.434 ms | 7.151 Melement/s |
| Down projection | 1 | 4.370 ms | 4.930 ms | 4.370 ms | 10.198 GFLOP/s |
| Down plus partial addition | 3 | 0.930 ms | 1.446 ms | 2.790 ms | 47.915 GFLOP/s |
| Output F32 -> F16 cast | 1 | 0.045 ms | 0.047 ms | 0.045 ms | 112.558 Melement/s |

Total GPU timestamp intervals: 32.594 ms. The single plain down is the first
block; the other three down calls include partial addition. Different block
positions and dependencies affect the intervals. Do not interpret their rate
difference as a standalone kernel speedup.

**Intact graph with fusion disabled**, to expose addition separately:

| Operation | Calls / FFN | Mean / call | p90 / call | Total / FFN | Effective rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Input F16 -> F32 cast | 1 | 0.665 ms | 1.198 ms | 0.665 ms | 7.705 Melement/s |
| Gate projection | 4 | 2.491 ms | 4.812 ms | 9.965 ms | 17.888 GFLOP/s |
| Up projection | 4 | 2.644 ms | 4.746 ms | 10.576 ms | 16.855 GFLOP/s |
| SwiGLU | 4 | 0.550 ms | 1.548 ms | 2.201 ms | 7.909 Melement/s |
| Down projection | 4 | 1.808 ms | 4.883 ms | 7.232 ms | 24.650 GFLOP/s |
| Partial-result addition | 3 | 0.478 ms | 1.470 ms | 1.433 ms | 10.722 Melement/s |
| Output F32 -> F16 cast | 1 | 0.018 ms | 0.024 ms | 0.018 ms | 284.340 Melement/s |

Total GPU timestamp intervals: 32.089 ms. Matrix rates count two FLOPs per
multiply-add, 44,564,480 FLOPs per projection/block. Non-matrix rates count
output elements, because assigning a hardware FLOP count to SwiGLU's exp/div
would be misleading. F16 weights, F32 intermediates and arithmetic. Casts
process 5120 elements, SwiGLU 4352, and addition 5120 per call.

| Worker interval | Half width | Full width |
| --- | ---: | ---: |
| Normal control before | 18.960 ms | 32.659 ms |
| Normal control after | 18.014 ms | 32.596 ms |
| Mean of normal controls | 18.487 ms | 32.628 ms |
| Named timestamp profile, fused | 32.277 ms | 48.567 ms |
| Named timestamp profile, unfused | 33.430 ms | 48.527 ms |

Observed profile overhead versus normal controls is +74.593%/+48.853% at
half/full width with fusion, and +80.828%/+48.730% without fusion. The named
logger prints more categories than the earlier grouped logger. Clocks and
thermals are uncontrolled, so these changes are not a causal decomposition
of instrumentation cost. Do not substitute the profiled categories for an
exact partition of normal-worker latency. Very short half-width down
intervals also produce rates inconsistent with isolated measurements; they
remain in the raw archive, but are not reported as GPU peak performance.

Two earlier diagnostic attempts are retained:

- **00:17 FAIL:** executing each graph node with a separate ordinary backend
  graph-compute call failed all 96 numerical comparisons, maximum relative L2
  46.418. Both preceding normal controls passed. Those timings are rejected.
  The backend resets inter-node dependency tracking during synchronization;
  the exact contribution of cross-graph visibility versus fence handling was
  not isolated. The worker exited normally; its owned forward was removed
  under the shared lock and cleanup passed at 00:18.
- **00:23 PASS:** enabling the existing Vulkan timestamp/barrier/fence path at
  each graph boundary restored bit-identical outputs across all 384 phone
  calls. All 1536 operation records and shapes passed the offline audit.
  However, separate completion per operation raised full-width worker time
  from 32.869 ms (control mean) to 126.614 ms (+285.209%). Isolated gate/up/down
  GPU intervals were 5.180/5.269/5.403 ms, demonstrating that this method changes
  scheduling substantially. It is a cross-check, not the normal-path table.

All workers in the successful runs exited normally with finite budgets;
owned ADB forwards are removed, boot ID is unchanged, and no Pixel FFN worker
remains. Cleanup PASS. Worker Werror build, logger-library build, pyflakes
on all four new Python files, source/runtime identities, numerical comparison,
operation counts, and timestamp sums PASS. No phone energy, full-server
output-token rerun, controlled clock study or hardware memory-counter proof.

Evidence: [intact-graph analysis](PIXEL_FFN_NAMED_PROFILE.json),
[isolated per-operation analysis](PIXEL_FFN_OP_PROFILE.json),
[final raw run](physical/pixel10pro-named-profile-1/run1/RESULT.json),
[fused profile log](physical/pixel10pro-named-profile-1/run1/named-profile/worker.log),
[unfused profile log](physical/pixel10pro-named-profile-1/run1/unfused-profile/worker.log),
[isolated retry](physical/pixel10pro-op-profile-2/run1/RESULT.json),
[failed attempt assessment](physical/pixel10pro-op-profile-1/ASSESSMENT.json),
[logger-only patch](software/pixel10pro-named-profile/NAMED_PROFILE.patch),
[worker instrumentation](software/pixel10pro-op-profile/OP_PROFILE.patch).
All three physical directories have ARCHIVE_SHA256.json receipts.

Recompute from repository root:

```sh
python3 research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/analyze_pixel_named.py research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/physical/pixel10pro-named-profile-1/run1 --output /tmp/pixel-named-profile.json
python3 research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/analyze_pixel_ops.py research_dev/scheduler/campaigns/burstgpt/reports/20260922-fast-path-M3/physical/pixel10pro-op-profile-2/run1 --output /tmp/pixel-isolated-ops.json
```


## 2026-09-23 01:09 UTC: Pixel GEMV tuning, build PASS, physical sweep pending

The user's kernel-tuning request is being tested with the existing F16 GEMV
shader specializations. The private build changes only the one-input-column
F16-weight/F32-input pipeline when explicitly enabled on PowerVR. Normal
pipelines retain their defaults. Production sources and phone runtime remain
unchanged. The sweep retains FP32 accumulation and uses no GPU timestamps or
stage instrumentation during timing.

- Workgroups128/256/512, output rows1/2/4/8, six selected combinations.
- Original128-thread/2-row controls bracket the sweep three times. A rebuilt
  default must produce exactly the same output bytes.
- Original kernel with quantum8704 is an independent graph-size candidate:
  two85MiB weight blocks per projection instead of four42.5MiB blocks. Each
  matrix stays below the Pixel's128MiB storage-buffer limit.
- Planned11 phone arms x120 calls, six layers18-23, half/full FFN widths,48 warm
  calls per width/arm after two warmup repeats. Each output is checked against
  CPU with relative-L2 limit0.01. A changed reduction order may change output
  bytes, so full-model token identity requires a later server check.

Build PASS: no compiler warnings;132 reused shader-object hashes match the
qualified snapshot. Python pyflakes and harness --help PASS. Library SHA256
`89a04a4ad91ea51a1448524eeb6a41ee9febac132ad719ed1a69826ab9976865`.
Reproduction: [builder](build_pixel_gemv_tune.py),
[harness](tune_pixel_gemv.py), [config](PIXEL_GEMV_SWEEP_CONFIG.json),
[source patch](software/pixel10pro-gemv-tune/GEMV_TUNE.patch), and
[build provenance](software/pixel10pro-gemv-tune/BUILD_PROVENANCE.json).

Physical status: waiting for the shared lock at
`/mnt/storage/s42-pixel10pro-gemv-tune-20260923-v1`. The other session is running
the long-tail baseline/treatment pair. No tuning phone worker has started.
Performance, Pixel numerical correctness, server-token identity and energy
are NOT VERIFIED for these candidates. No speedup claim yet.

### 2026-09-23 01:24 UTC: physical launch FAIL, zero Pixel calls

The900-second shared-lock wait exited75. No `run1/`, `RUN.log`, or `DONE.json`
was created. The long-tail campaign still owns the rig and remains in its
baseline arm. No tuning worker started, no phone file was deployed by this
attempt, and no tuning process remains queued. This is an execution-blocking
result, not a measured kernel regression. Correctness, speedup, full-model
tokens and energy remain **NOT VERIFIED**. Evidence:
[lock attempt](physical/pixel10pro-gemv-tune-1/LOCK_ATTEMPT.json).
The staged kernel command can be rerun once the lock becomes available:

```sh
ssh zhihao@172.20.74.85 'flock -w 900 -E 75 /home/zhihao/moe-resident-routing-4060ti-op15/.recent_moe_execution.lock bash /mnt/storage/s42-pixel10pro-gemv-tune-20260923-v1/RUN.sh'
```

A separate graph-order candidate also built PASS with `-Werror`, no warnings.
Its [four-line patch](software/pixel10pro-graph-tune/GRAPH_OPTIMIZE.patch)
calls the existing backend graph optimizer before allocation. The ordinary
worker directly computes its graph and bypasses the scheduler's optimizer
call. Vulkan's optimizer can place independent projections from separate
FFN blocks next to each other; whether that helps this GPU is unmeasured.
This private worker remains local, SHA256
`6190b96d65a10cbf4b336c11eeeb19eddb61a03bb6773db96a7ca46540f1fd6a`.
[Build provenance](software/pixel10pro-graph-tune/BUILD_PROVENANCE.json),
[graph-only sweep config](PIXEL_GRAPH_SWEEP_CONFIG.json).

The shared harness now supports choosing the original or graph-order worker,
independently of the Vulkan library and block size. The staged first sweep
retains its earlier immutable harness snapshot. The
[result auditor](analyze_pixel_gemv.py) recomputes every CPU/output comparison,
checks repeat determinism and timing summaries, and reports each candidate
against its surrounding controls. The server qualification harness accepts
`phone_environment` for later verification of a winning specialization.
All four changed Python files pass pyflakes. None of these build checks is
physical evidence of an improvement. The last measured original full-width
worker mean remains32.628ms from the prior named-profile controls.


## 2026-09-23 02:39 UTC: Pixel GEMV physical sweep numerical PASS

The first physical sweep completed11 phone arms x120 calls, all1320 numerical
comparisons PASS against CPU (max relative L2 0.000325482692).
Raw-output hashes, repeat determinism and recomputed timings PASS. Each width
has48 warm calls per arm, across six layers, with profiling disabled.
Controls full-width worker means32.856/33.448/33.287ms.

| Candidate (threads/output rows) | Full worker mean ms | Reduction vs surrounding controls | Exact calls vs original |
| --- | ---: | ---: | ---: |
| 01-rebuilt-default | 33.161 | -0.028% | 120/120 |
| 02-w128-r1 | 35.135 | -5.982% | 120/120 |
| 03-w256-r2 | 38.347 | -15.672% | 0/120 |
| 04-w128-r4 | 32.320 | 2.511% | 120/120 |
| 06-w512-r2 | 56.604 | -69.637% | 0/120 |
| 07-w128-r8 | 32.322 | 3.135% | 120/120 |
| 08-w512-r4 | 55.790 | -67.198% | 0/120 |
| 09-q8704-original | 33.855 | -1.461% | 0/120 |

The best exploratory setting is128 threads/8 output rows:32.322ms vs33.368ms
for its surrounding controls,3.135% lower. Row4 is similarly close. Larger
workgroups and quantum8704 failed to improve performance. These small gains
need repetition before selecting a default. All workers exited normally,
owned forwards were removed, and the phone boot remained unchanged.
[Audit](PIXEL_GEMV_SWEEP.json), [raw run](physical/pixel10pro-gemv-tune-1/run1/RESULT.json).
No full-model token, energy, or production deployment claim.


## 2026-09-23 02:42 UTC: graph reordering numerical PASS, performance FAIL

All600 phone calls passed CPU checks and repeat-determinism audit; maximum
relative L2 0.000325483. Graph reordering at quantum4352 averaged33.321ms per
full FFN against33.055ms surrounding controls,0.804% slower. At quantum8704,
original/reordered graphs averaged33.882/33.886ms,2.502/2.514% slower than the
controls. This candidate is rejected for performance. The isolated source and
binary are retained to reproduce the negative result. Cleanup PASS, normal
finite exits and unchanged phone boot. [Audit](PIXEL_GRAPH_SWEEP.json),
[raw run](physical/pixel10pro-graph-tune-1/run1/RESULT.json).

The next kernel experiment extends only the F16/F32 one-column specialization
to test32/64/128-lane subgroups, selecting the existing subgroup or hybrid
reduction shader as appropriate. Requested sizes are checked against device
limits. This is independent of graph ordering. The128-lane/128-thread/8-row
candidate is repeated in the new build. Precision remains FP32 accumulation.
[Config](PIXEL_SUBGROUP_SWEEP_CONFIG.json),
[patch](software/pixel10pro-gemv-tune-subgroup/GEMV_TUNE.patch).


## 2026-09-23 02:51 UTC: subgroup sweep numerical PASS;32/64-lane candidates regress

All1320 calls passed CPU checks (max relative L2 0.000325483), raw-output and
repeat-determinism audits. All workers exited normally, owned forwards were
removed, and the boot stayed unchanged. Controls full means33.409/33.236/
33.006ms. Timing remains unprofiled. [Audit](PIXEL_SUBGROUP_SWEEP.json).

| Threads/subgroup/output rows | Full worker mean ms | Reduction vs surrounding controls |
| --- | ---: | ---: |
| 128/128/8 | 32.063 | 3.780% |
| 32/32/2 | 39.861 | -19.622% |
| 64/64/2 | 36.563 | -9.726% |
| 64/64/4 | 37.442 | -12.362% |
| 128/64/4 | 38.160 | -15.213% |
| 128/32/4 | 38.206 | -15.354% |
| 128/32/8 | 38.463 | -16.129% |
| 64/64/8 | 36.692 | -10.783% |

Smaller subgroups failed to improve latency.128-thread/128-lane/8-row repeats
at32.063ms with exact output bytes vs the original. A20-repeat reversed-order
comparison of4 and8 rows is the confirmation step. No server or energy result
is inferred from these microbenchmarks.


## 2026-09-23 02:55 UTC: reversed-order kernel confirmation PASS

The longer sequence was original, row8, row4, original, row4, row8, original;
all candidates use128 threads and a128-lane subgroup.20 repeats per arm,
first2 excluded from timing, give108 warm calls per width per arm across
layers18-23. All1680 outputs are byte-identical to the original, and all CPU
checks pass (max relative L2 0.000325483). Raw-output, call-count,
repeat-determinism and timing recomputation audit PASS.

| Setting | Half FFN worker ms | Change vs matched controls | Full FFN worker ms | Change vs matched controls |
| --- | ---: | ---: | ---: | ---: |
| Original controls, matched mean | 18.535 | - | 33.098 | - |
|128 threads /128 lanes /4 output rows | 18.711 | +0.950% | 32.601 | -1.499% |
|128 threads /128 lanes /8 output rows | 18.397 | -0.743% | 32.399 | -2.111% |

Each candidate replicate is compared with the mean of the original controls
immediately surrounding its group. The table averages the two candidate
replicates and their matched controls. Individual row8 full means32.368 and
32.430ms; surrounding original means33.096 and33.099ms. This is a modest
improvement. The two row8 half-width replicates do not both beat controls,
so no robust half-width speedup claim. Full FFN effective matrix throughput
is16.506 GFLOP/s when divided by whole-worker time; this is not a peak-GPU
or isolated-shader rate. Precision remains FP16 weights/FP32 arithmetic.

Selected isolated setting: `S42_PIXEL_F16_WG=128`, `S42_PIXEL_F16_ROWS=8`,
`S42_PIXEL_F16_SUBGROUP=128`, quantum4352. Tuned library SHA256
`a97cb05dcac71b8adf826559e27ee62b9286c3c4a571523d5583976b60856b22`.
No production default has been changed. This setting now proceeds to a
four-request real-server token check at half/full widths with desktop
controls before/after. Server/energy conclusions remain unverified here.

Across the initial, graph-order, subgroup and confirmation sweeps, all4920
phone calls passed the CPU tolerance checks. Graph order, larger workgroups,
smaller subgroups and quantum8704 were rejected for performance. All finite
workers exited normally, owned forwards were removed, and phone boot was
unchanged. [Confirmation audit](PIXEL_GEMV_CONFIRM.json),
[config](PIXEL_GEMV_CONFIRM_CONFIG.json),
[raw results](physical/pixel10pro-gemv-confirm-1/run1/RESULT.json).


## 2026-09-23 03:08 UTC: selected Pixel kernel server qualification PASS

The selected 128-thread / 128-lane / 8-output specialization completed a real
Qwen server check: **all four 64-token outputs identical**, 744 measured phone
calls across layers 18-23, with 372 calls at each half/full FFN width. Raw SSE
output audit, per-layer call coverage and sampled-energy recomputation PASS.
The worker log proves the selected specialization was active. Both processes
exited 0; 24 finite-budget completion calls were outside measurement. Owned
forwards were removed, the phone boot remained unchanged, and independent
postrun cleanup PASS at 03:09 UTC. No tuning job remains queued.

| Arm | Phone worker mean ms/call | RPC mean ms/call | Decode s | CPU+GPU request energy J |
| --- | ---: | ---: | ---: | ---: |
| Desktop controls, mean | - | - | 39.360 | 4935.632 |
| Tuned Pixel, half-width | 19.848 | 24.707 | 42.637 | 4364.318 |
| Tuned Pixel, full-width | 28.642 | 33.239 | 45.398 | 4322.180 |

Half/full decode is 8.326%/15.340% slower than mean desktop control. Host request
energy is 11.575%/12.429% lower. These are one short request per split; phone
energy is unmeasured. The server check did not include a matched original-phone
arm, so it does not establish a kernel-caused server improvement. The kernel
claim remains the confirmed 33.098 -> 32.399 ms full-worker reduction, 2.111%.

The winning isolated library and explicit configuration are qualified for
these shapes; production defaults remain unchanged. All 4,920 synthetic calls
passed CPU comparisons, all 1,680 confirmation outputs were byte-identical,
and the real-server token check passed. Other models, multi-token performance,
full-trace savings and multi-phone integration remain unverified.

[Tuning process and complete results](PIXEL_KERNEL_TUNING.md),
[selected settings and hashes](PIXEL_GEMV_SELECTED.json),
[server audit](PIXEL_GEMV_SERVER.json),
[raw server result](physical/pixel10pro-gemv-server-1/run1/RESULT.json),
[postrun cleanup](physical/pixel10pro-gemv-server-1/POSTRUN_CLEANUP.json).
Changed Python files pass pyflakes; isolated C++ builds have no warnings.


## 2026-09-23 03:30 UTC: Pixel shader-body tuning build PASS

Nine actual shader-body variants are built: wider F16 vector loads,
unroll1/2/4/8, independent FP32 accumulator chains, eight elements per lane,
and striped lane-to-column mapping. They reuse existing GEMV bindings,
reduction and fusion handling. Weight format and arithmetic precision remain
unchanged. All27 SPIR-V modules pass Vulkan1.2 validation; arithmetic remains
FP32, and no Float16 arithmetic capability is required. C++/GLSL build and
Python pyflakes PASS, no compiler warnings. The extended result auditor
reproduces all four previous raw-output reports exactly.

A15-arm,1800-call comparison with current-best row8 controls is staged at
`/mnt/storage/s42-pixel10pro-dense-gemv-20260923-v1` and waiting under the
shared lock since03:30 UTC. No candidate has executed on Pixel yet. Numerical
correctness, speedup and server-token identity are NOT VERIFIED.
[Design, variants and protocol](PIXEL_DENSE_GEMV_PLAN.md),
[shader code](pixel_dense_gemv.glsl),
[build proof](software/pixel10pro-dense-gemv-v1/BUILD_PROVENANCE.json).


### 2026-09-23 03:46 UTC: shader-body physical launch FAIL, zero calls

The900-second shared-lock wait exited75. The longtail treatment retest still
owns the rig. No `run1/`, `RUN.log` or `DONE.json` exists for this attempt;
no Pixel file was deployed or shader called, and no tuning job remains queued.
The code/build/SPIR-V validation results above remain PASS, while numerical
correctness, speedup and server tokens are NOT VERIFIED. The current-best
qualified setting remains32.399ms full-worker mean from the prior confirmation.

[Lock evidence](physical/pixel10pro-dense-gemv-1/LOCK_ATTEMPT.json),
[resumable protocol and variant descriptions](PIXEL_DENSE_GEMV_PLAN.md),
[actual shader loop](pixel_dense_gemv.glsl). This is an execution blocker,
not evidence that any new shader is slower or incorrect.

### 2026-09-23 04:34 UTC: persistent Pixel test queue

The second900-second lock attempt also exited75 with zero phone calls. The
user asked to keep the tests queued. The longtail treatment-2 campaign remains
untouched and resumed request progress around04:25 UTC. Queue PID2282252 at
`/mnt/storage/s42-pixel10pro-dense-gemv-20260923-v1` now retries the prescribed
lock wait, executes the finite1800-call sweep once, and audits the raw outputs.
It stops after the sweep, including on failure. `QUEUE_EVENTS.jsonl` records
waits/completion; `DONE.json` records the physical run exit; `SWEEP_AUDIT.json`
will hold the successful audit. New correctness/speedup/server tokens remain
NOT VERIFIED. [Queue source](physical/pixel10pro-dense-gemv-1/QUEUE.py),
[second wait evidence](physical/pixel10pro-dense-gemv-1/LOCK_ATTEMPT_2.json).

### 2026-09-23 05:13 UTC: Pixel-only kernel run, no whole-rig wait

The user clarified the scope as kernel testing on the Pixel. Our earlier
whole-rig queue was unnecessarily broad; it is cancelled and launch-guarded.
The15-arm sweep now replays identical requests locally on Pixel and reuses
120 archived, hash-checked CPU reference outputs. Each worker exits normally
after120 calls. Internal worker time is measured; host/phone energy and USB
round-trip latency are not. All arms use this same local execution procedure.
An initial Android flock fd-inheritance error stopped before any calls; the
explicit fd redirect fixed it. Seven arms have finished; numerical/performance
qualification is still pending the complete raw audit.
[Local runner and auditor](pixel_local_sweep.py),
[reference provenance](physical/pixel10pro-dense-local-1/run1/REFERENCE_PROVENANCE.json).

### 2026-09-23 05:22 UTC: Pixel-only shader tests complete

Numerical audit **PASS** for3,960 calls. All2,160 confirmation outputs are
byte-identical to the original Pixel kernel; CPU max relative L2=0.000325482692.
The best repeated average is vector loads with unroll1: full20.455ms against
21.244ms matched controls (**3.714% lower**), half11.696ms against12.277ms
(**4.728% lower**). Unroll2 repeats at2.661% lower full-width latency, less than
its initial6.772% result. Larger unroll, extra accumulators and striped mapping
regress. Controls vary20.489-21.753ms; the small gain is provisional and neither
candidate beats every individual control. No production selection is changed.

The sweep and confirmation execute locally on Pixel, using archived CPU
references. The earlier whole-rig wait was unnecessary for this narrower scope;
our waiting job is cancelled. No desktop model work or energy/USB/server measurement
is part of these tests. The earlier32.399ms USB-driven result is not a matched
baseline for this local replay. Cleanup and pyflakes PASS; all four previous
sweep audits reproduce exactly. Initial shell-lock failure remains recorded
as FAIL with zero kernel calls.

[Full table and limitations](PIXEL_DENSE_LOCAL_RESULTS.md),
[confirmation summary](PIXEL_DENSE_LOCAL_CONFIRM_SUMMARY.json),
[candidate settings](PIXEL_DENSE_LOCAL_CANDIDATE.json).

### 2026-09-23 16:40 UTC: FFN partition-concurrency review

Source/shape review **PASS**: hidden-channel partitioning with a shared input
and summed partial-down outputs is valid and already implemented as four4352
channel chunks. Concurrent execution and any speedup are **NOT VERIFIED**.
Current row8 kernels already launch544/640 workgroups per gate-up/down dispatch;
full one-token F16 weight traffic is510MiB and is unchanged by partitioning.
A proposed phone-local1/2/4-part overlap comparison is documented, with separate
FP32 partial buffers and a fixed-order GPU sum. No kernel change or new run.
[Reasoning, shapes, limitations and test scope](PIXEL_FFN_CONCURRENCY.md).

### 2026-09-23 16:59 UTC: private up-matvec/SwiGLU fusion build

Build/pyflakes and27-module SPIR-V validation **PASS**. Fusion removes one
activation dispatch per4352-wide block and its up intermediate write/read,
using existing Vulkan alias/dependency checks and FP32 arithmetic. The
840-call phone-local on/off test is running; numerical/speedup results are
**NOT VERIFIED**. Other fusions remain enabled in both arms.
[Patch generator](pixel_swiglu_fusion.py),
[build provenance](software/pixel10pro-swiglu-fusion-v1/BUILD_PROVENANCE.json),
[test configuration](PIXEL_SWIGLU_SWEEP_CONFIG.json).

### 2026-09-23 17:07 UTC: first fusion measurement and coverage correction

Numerical **PASS**:840/840 outputs exact, max relative L2=0.000325483.
Full fusion coverage **FAIL**:264/384 dispatches per fused arm. An additional
12-call exact diagnostic leaves one GLU separate per request after the existing
alias check. Initial full-width reductions0.616%/3.010% are exploratory.
A private one-line worker change retains the FP32 input buffer. Build **PASS**;
2160-call confirmation with identical on/off workers and prior-best controls
is running, so the fix's coverage and speedup are **NOT VERIFIED**.
[Initial audit](physical/pixel10pro-swiglu-local-1/run1/SWEEP_AUDIT.json),
[coverage failure](physical/pixel10pro-swiglu-local-1/run1/FUSION_COVERAGE.json),
[diagnostic](physical/pixel10pro-swiglu-coverage-1/run1/DIAGNOSTIC_AUDIT.json),
[input lifetime patch](software/pixel10pro-swiglu-input-v1/PRESERVE_INPUT.patch).

### 2026-09-23 17:11 UTC: complete fusion works but does not improve speed

Numerical and full fusion coverage **PASS**:2160/2160 exact outputs, max
relative L2=0.000325483,744/744 fused dispatches in each treatment arm. The
input-retention change fixes the initial alias rejection without disabling
safety checks. Performance **FAIL**: full21.001ms vs20.352ms matched off
controls (**3.191% slower**); half11.819ms vs11.652ms (**1.431% slower**).
Both full-width repeats lose. The complete candidate also loses to the
previous vec4_u1 library/worker controls. It is rejected for promotion.

All3012 outputs across the initial sweep, diagnostic and confirmation are
exact. Builds, SPIR-V validation, pyflakes and cleanup **PASS**. No server,
USB latency, energy or full-model token measurement; previous selection and
production defaults unchanged.
[Process, unified table and limitations](PIXEL_SWIGLU_RESULTS.md),
[measured summary](PIXEL_SWIGLU_RESULTS.json),
[confirmation audit](physical/pixel10pro-swiglu-local-confirm-1/run1/SWEEP_AUDIT.json).

### 2026-09-23 17:54 UTC: Pixel CPU versus GPU comparison

Comparison and numerical qualification **PASS**,600 calls. Current generic
four-thread CPU: half23.975ms,full45.849ms. Matched vec4_u1 GPU controls:
half11.487ms,full19.929ms, giving **2.09x/2.30x GPU speedups**. Both repeats
agree. CPU/GPU outputs differ in low bits; all pass CPU-reference tolerance,
max relative L2=0.000325483 overall and0.0000988841 for CPU. Cleanup/pyflakes
**PASS**. No CPU tuning/fusion,energy,USB/server latency or token test.
Archived prior CPU references ran on the desktop and are explicitly separate.
[Unified table and limitations](PIXEL_CPU_GPU_RESULTS.md),
[measured summary](PIXEL_CPU_GPU_RESULTS.json),
[raw audit](physical/pixel10pro-cpu-local-1/run1/SWEEP_AUDIT.json).

### 2026-09-23 17:59 UTC: bandwidth and concurrent CPU/GPU assessment

Arithmetic/source review **PASS**. GPU26.834GB/s and CPU11.664GB/s are logical
weight bytes divided by full worker time; physical DRAM bandwidth and its
ceiling are **NOT VERIFIED**. Shared-memory concurrency might raise aggregate
throughput or introduce contention. A no-contention linear model gives
13.891ms at~70% GPU/~30% CPU versus19.929ms GPU-only, but combined execution
has **NOT RUN**. No peak-bandwidth or speedup claim is made.
[Mechanisms, estimates and separating experiments](PIXEL_BANDWIDTH_ASSESSMENT.md).

### 2026-09-23 18:22 UTC: measured Pixel streaming bandwidth sweep

24-arm numerical and cleanup **PASS**.512MiB reads: best CPU26.737GB/s,
GPU38.694GB/s wall (44.163GB/s device timestamps), joint44.218GB/s with
99.686% overlap. Both engines slow under concurrent load. GPU scalar27.967
vs vector38.694GB/s; aggressive unroll and more CPU threads do not win.
Copy24.348GB/s is read-plus-write traffic. No hardware DRAM counters, energy
or theoretical-peak proof. These are exploratory; confirmation is pending.
A1200-call persistent-CPU-thread/affinity FFN sweep is running.
[Bandwidth raw audit](physical/pixel10pro-bandwidth-1/run1/RESULT.json),
[CPU FFN test](PIXEL_CPU_TUNE_CONFIG.json).

### 2026-09-23 18:31 UTC: CPU FFN tuning exposes ineffective Android affinity

Numerical/cleanup **PASS**,1200 calls. Persistent6 CPU threads18.267ms vs
original43.950-50.227ms and GPU20.581-20.841ms; confirmation pending.
Requested affinity **FAIL**: all actual thread masks remain0-7. The CPU
library's platform guard silently selects a no-op on Android. A private
one-line guard fix builds successfully; arithmetic objects are unchanged.
The run remains useful as an unpinned threadpool sweep.
[Affinity evidence](physical/pixel10pro-cpu-tune-1/run1/CPU_AFFINITY.json),
[numerical/timing audit](physical/pixel10pro-cpu-tune-1/run1/SWEEP_AUDIT.json),
[private fix](software/pixel10pro-cpu-tune-v3/ANDROID_AFFINITY.patch).

### 2026-09-23 18:32 UTC: repeated streaming bandwidth

17-arm checksum/cleanup **PASS**. CPU1/core7 repeats26.77-26.97GB/s;
GPU vec4 repeats37.25-38.04GB/s versus scalar27.33-27.76GB/s.
Concurrent CPU1+GPU **46.47-46.58GB/s**; each engine slows but aggregate
throughput rises. A dynamic-chunk CPU experiment is pending because static
equal slices can leave fast cores waiting for slow cores.
[Repeated measurements](physical/pixel10pro-bandwidth-confirm-1/run1/RESULT.json).

### 2026-09-23 18:37 UTC: repeated CPU FFN speedup and affinity correction

Numerical/performance/cleanup **PASS**:1320 confirmation calls,2520 total.
Six persistent unpinned threads: full18.291ms vs
44.291ms original CPU matched controls
(**58.70% less latency,2.421x throughput**);
GPU references20.610ms. All2040 CPU outputs exact to original CPU.
Actual affinity after the private guard fix **PASS**; strict pinning does not
beat unbound6. No energy/server/USB/token proof or production deployment.
[Process, table, limits](PIXEL_CPU_TUNE_RESULTS.md),
[private candidate](PIXEL_CPU_TUNED_CANDIDATE.json).

### 2026-09-23 18:42 UTC: bandwidth task complete

55-arm checksum/cleanup **PASS**,8594 measured engine passes. Dynamic CPU
work assignment raises sustained reads24.059->30.421/30.565GB/s. Repeated
GPU37.25-38.04GB/s wall (43.19-43.71 device); joint CPU1+GPU46.47-46.58GB/s,
**22.91% above matched GPU-only**. Logical streams, not physical DRAM
counters/theoretical peak. The actual FFN gain is CPU persistent threads:
44.291->18.291ms; concurrent FFN and energy remain unverified.
[Complete process, unified tables and limits](PIXEL_BANDWIDTH_RESULTS.md),
[machine-readable summary](PIXEL_BANDWIDTH_RESULTS.json).

### 2026-09-23 18:54 UTC: CPU/server overlap budget

Source/arithmetic **PASS**, physical retest **NOT RUN** because the shared
rig lock is busy with OP15 sweeps. Existing overlap works by computing the
host FFN share while the phone RPC is pending. At50%,9.552ms host versus
9.162ms local CPU plus5.041ms historical overhead projects **4.651ms wait
per layer**;100% projects23.066ms and has no local FFN share to hide it.
These are mixed-source estimates, not measured new server timings.
No process interrupted or test queued.
[Calculation, mechanism and limits](PIXEL_CPU_SERVER_OVERLAP_ASSESSMENT.md).

### 2026-09-23 19:10 UTC: CPU/GPU FFN split feasibility

Evidence/arithmetic **PASS**; concurrent FFN speedup **NOT VERIFIED**.
Full CPU18.291ms versus GPU20.610ms. The measured stream contention suggests
that a naive equal split may provide little gain: a conditional sensitivity
model gives17.763ms before extra merge/synchronization. This is not a
physical FFN result. Existing dual-backend code was inspected only; no test
was started or queued.
[Measured inputs, split design and limits](PIXEL_CPU_GPU_FFN_ASSESSMENT.md).

### 2026-09-23 19:26 UTC: concurrent Pixel CPU/GPU FFN smoke

Private build and numerical **PASS**,180 calls,max relative L2=0.000325483.
New CPU-only mode is36/36 byte-exact to the qualified CPU worker.36 dual
records prove disjoint block coverage and overlapping execution branches.
Initial CPU6 50/50 speed **FAIL**: full22.128 vs18.280ms CPU controls;
half14.647 vs9.207ms. Six warm samples/width, exploratory. Merge0.0085ms
full; concurrent branch slowdowns dominate. Cleanup **PASS**.
19-arm ratio/thread sweep pending. No new server, energy or token result.
[Smoke audit](physical/pixel10pro-cpu-gpu-smoke-1/run1/SWEEP_AUDIT.json),
[partition and overlap proof](physical/pixel10pro-cpu-gpu-smoke-1/run1/DUAL_COVERAGE.json).

### 2026-09-23 19:35 UTC: CPU/GPU ratio and thread sweep

Numerical/overlap/cleanup **PASS**:2280 calls,1440 dual records,max relative
L2=0.000325483. Speed versus tuned CPU **FAIL** for all12
combinations. Best CPU4 with50/50 full19.349ms versus matched CPU18.214ms
(+6.23%); half10.062 versus9.162ms (+9.82%). GPU controls20.652-21.326ms.
2640-call confirmation now tests unbound4 and pinned4/6 with explicit GPU
host-thread affinity. No server or energy result.
[Sweep audit](physical/pixel10pro-cpu-gpu-sweep-1/run1/SWEEP_AUDIT.json),
[branch proof](physical/pixel10pro-cpu-gpu-sweep-1/run1/DUAL_COVERAGE.json).

### 2026-09-23 19:45 UTC: concurrent CPU/GPU FFN completed

Functionality, overlap, actual affinity and cleanup **PASS**:35 arms,5100
numerical checks,2916 dual calls. Selected50/50 uses four persistent CPU
threads on cores4-7 and the GPU submission thread on CPU cores0-3.

| Phone workload | Matched CPU ms | GPU reference ms | CPU+GPU ms | Mean saving vs CPU |
| --- | ---: | ---: | ---: | ---: |
| Half FFN | 9.169 | 12.057 | 8.086 | 11.81% |
| Full FFN | 18.210 | 21.014 | 15.334 | 15.79% |

Mean speed **PASS** in both repeats; p99 versus CPU **FAIL** (full25.294
versus18.716ms). Selected max relativeL2=0.000279321,all-arm max0.000325483.
Both selected arms240/240 exact to one another and to unbound4. Full merge
is0.0081ms; branch overlap14.046ms. Private benchmark qualification only;
USB,server tokens,energy,multi-row and scheduler integration unverified.
[Complete process, unified tables and limits](PIXEL_CPU_GPU_SPLIT_RESULTS.md),
[selected settings/hashes](PIXEL_CPU_GPU_SPLIT_CANDIDATE.json),
[cleanup](PIXEL_CPU_GPU_SPLIT_CLEANUP.json).

### 2026-09-23 20:12 UTC: unequal CPU/GPU ratios

Private worker/shader build **PASS**,27 SPIR-V modules validated; Python
static checks **PASS**. Ratios now use64-channel steps with fixed CPU4
affinity4-7/GPU-host0-3. Smoke checks new50/50 against the qualified old
path plus44.85/55.15 percent CPU shares. Physical correctness/speed
**NOT VERIFIED**,180-call smoke staging.
[Build evidence](PIXEL_CPU_GPU_RATIO_LOCAL_CHECKS.json).

### 2026-09-23 20:14 UTC: fine ratio smoke

Numerical/partition/overlap/affinity/cleanup **PASS**,180 calls,max
relativeL2=0.000281345. Fine50/50 matches old50/50 in36/36 outputs.
Unequal44.85/55.15 percent CPU shares also pass. Old50 full controls
25.979/24.149ms vary substantially versus prior15.334ms qualification;
only6 warm samples/width, so ratio speedup **NOT VERIFIED**.
9-ratio15-arm1800-call coarse sweep staging with repeated controls.
[Smoke audit](physical/pixel10pro-cpu-gpu-ratio-smoke-1/run1/SWEEP_AUDIT.json).

### 2026-09-23 20:22 UTC: coarse unequal ratio sweep

Numerical/affinity/cleanup **PASS**,1800 calls,maxL2=0.000299222.
Partition/timing arithmetic **PASS**,1560 dual records; strict every-call
overlap **FAIL** on1/1560 (35.294%CPU arm,half request64,GPU starts10us
after CPU completion). The outlier remains in latency statistics.
Full25%CPU15.135ms vs matched50/50 17.541ms (-13.72%);35.294%CPU
14.583 vs16.036ms (-9.06%). Controls drifted; candidates remain exploratory.
17-arm4080-call refinement covers12.5-38.235%CPU.
[Coarse audit](physical/pixel10pro-cpu-gpu-ratio-coarse-1/run1/SWEEP_AUDIT.json),
[explicit overlap failure](physical/pixel10pro-cpu-gpu-ratio-coarse-1/run1/DUAL_COVERAGE.json).

### 2026-09-23 20:29 UTC: ratio refinement

Numerical/partition/overlap/affinity/cleanup **PASS**,4080 calls,3600 dual
records,maxL2=0.000310862. After ten warmups,38.235%CPU has the lowest
observed mean: full13.145ms,half7.084ms. Matched50 controls15.765/8.330ms,
but full50 controls vary13.564..16.497ms,so speed promotion remains
**NOT VERIFIED**.22-arm5280-call reversed fine confirmation now staging.
[Warmup2 audit](physical/pixel10pro-cpu-gpu-ratio-refine-1/run1/SWEEP_AUDIT.json),
[prespecified warmup10 sensitivity](physical/pixel10pro-cpu-gpu-ratio-refine-1/run1/STEADY_STATE_AUDIT.json).

### 2026-09-23 20:42 UTC: unequal CPU/GPU ratio tuning completed

Numerical/selected mean/selected observed p99/affinity/cleanup **PASS**.
20 mixed ratios,59 arms,11340 numerical checks,maxL2=0.000325483.
Choose **39.706%CPU /60.294%GPU**,four pinned CPU threads with the GPU
submission thread on separate CPU cores.

| Phone FFN | CPU-only ms | GPU-only ms | Matched50/50 ms | Tuned ratio ms | Gain vs50/50 | Gain vsCPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half |9.233|12.015|8.166|7.077|13.34%|23.35%|
| Full |18.330|21.135|15.379|13.058|15.09%|28.76%|

Selected full repeats13.062/13.053ms,half7.125/7.028ms. Observed p99
full14.029 vsCPU19.034ms;half8.104 vs9.582ms. Final confirmation uses
ten warmups and120 measured samples per width per ratio across reversed
repeats.42.647%CPU gives the lowest full mean13.021ms,only0.037ms
faster,but half p99 fails against CPU (12.404 vs9.582ms).
Final4080/4080 dual calls overlap **PASS**; whole-exploration strict
overlap **FAIL**1/9420 (coarse outlier retained). Selected numerical
maxL2=0.000286656;240/240 byte-exact across repeats.
Private phone-local qualification only; no new USB/server/energy/full-token
or multi-row/integration result.
[All ratios, process, tails and limits](PIXEL_CPU_GPU_RATIO_RESULTS.md),
[selected configuration](PIXEL_CPU_GPU_RATIO_CANDIDATE.json),
[cleanup](PIXEL_CPU_GPU_RATIO_CLEANUP.json).

### 2026-09-23 20:51 UTC: bandwidth and compute headroom assessment

Evidence/arithmetic **PASS**; true DRAM and sustained compute peaks
**NOT ESTABLISHED**, no new hardware run. Full effective weight bandwidth
is40.955 GB/s,88.05% of the46.515 GB/s best measured concurrent stream.
The conditional memory-only reference is11.497 ms versus13.058 ms measured,
about11.95% latency headroom under that rate assumption. Batch1 has nominal
intensity1 FLOP/byte,so the low GFLOP/s is consistent with memory pressure.
DXT vendor1536 FP32 FLOPs/clock implies conditional1.680 TFLOP/s at1.094GHz;
no compute-only benchmark or continuous clock/counter proof exists.
[Calculations, limits and next experiments](PIXEL_CPU_GPU_ROOFLINE_ASSESSMENT.md).

## 2026-09-23 21:11 UTC - Pixel four-direction optimization: joint sweep and coalescing smoke PASS

Joint1/2/3/4/6-thread,15-ratio-point sweep:21 arms5040 numerical,
partition/overlap and affinity checks PASS,max relativeL2=0.00030946652.
ExistingCPU4/c3456 controls full mean13.181039ms; best non-four-thread
armCPU6/c3968 full13.193633ms. No replacement established by this sweep.
Private v4 supports mixed per-tensor Q4_K/Q6_K and full-block coalescing;
coalesced full/half use6 GEMVs with1.5x resident weights.144-call smoke
numerical PASS,maxL2=0.000286656; new inactive mode byte-exact to controls.
Only6 warm samples/width and controls19.403/20.055ms full: no smoke speed
claim. New batch shader reuses each F16 weight across2/4/8 independent
rows;27 SPIR-V modules validate PASS. Batch/packed physical tests pending.
Original packed tensors extracted losslessly; F16 rounding sampled32rows
per tensor PASS,exhaustive proxy comparison not done. No server/energy test.
Evidence: M3 physical/pixel10pro-cpu-gpu-joint-1/run1,
physical/pixel10pro-ffn-coalesce-smoke-1/run1,software/pixel10pro-cpu-gpu-v4,
software/pixel10pro-dense-batch-v1.


## 2026-09-23 21:19 UTC - Pixel batch smoke1 FAIL: client element count; graceful finite drain PASS

ReferenceCPU96 single-row calls completed; first mixed arm accepted12
single-row calls then rejected request13 because new harness sent elements5120
for tokens2. Worker validation behaved correctly. Worker was idle after
client disconnect; drained remaining132 valid single-row requests to its
finite limit,without signals or force-kill. Original malformed packets
preserved; remaining3 arms used corrected headers for diagnostics only.
Whole run excluded from performance comparisons and FAILURE.json recorded.
Fixed request and response audit elements to5120*tokens,added independent
serialized packet extent validation. Fresh672-call retry prepared with
new dispatch counters;27 SPIR-V modules PASS. No multi-row win claimed.
Evidence: M3 physical/pixel10pro-ffn-batch-smoke-1/run1/FAILURE.json,
recovery/,pixel_experiment_suite.py,software/pixel10pro-dense-batch-v2.


## 2026-09-23 21:26 UTC - Pixel multi-row smoke2 numerical/dispatch PASS; packed CPU buffer failure fixed for retest

Fresh batch1/2/4/8 suite672 calls PASS,max row relativeL2=0.000417331.
96 independent phoneCPU single-row references; row0 cross-check against
archived desktopCPU passes(maxL2=9.88841e-5),not byte-identical across
architectures.576 mixed calls overlap/partition checked. Both new shaders
count162 tiled GPU projections at each batch2/4/8,full coverage PASS.
Short timings not promoted: tiles8/4 perform poorly at8 rows.16-arm14496-call
sweep prepared,tiles1/2,subgroups32/64/128 and5 unequal CPU shares.
Packed GPU smoke36 calls numerical PASS,maxL2=0.000514351; CPU startup
FAIL optional repacking buffer rejects mixed Q4_K/Q6_K tensor allocation.
Private v5 uses standard CPU buffer for packed mode,build PASS.13-arm3120-call
packed/coalescing confirmation running,20reps/10warmups. No signal/force-kill,
server integration,energy or full-model token measurement.
Evidence: M3 physical/pixel10pro-ffn-batch-smoke-2/run1/SUITE_RESULT.json,
physical/pixel10pro-ffn-packed-smoke-1/run1/FAILURE.json,
software/pixel10pro-cpu-gpu-v5,PIXEL_FFN_BATCH_SWEEP_CONFIG.json.


## 2026-09-23 21:32 UTC - Pixel packed/coalescing confirmation: accuracy FAIL for native packed CPU; coalescing small gain only

13 arms3120 calls completed normally; all metadata/partition/dispatch checks
PASS. F16 selected controls full13.063823/half7.046517ms.
Coalesced full12.943708ms (-0.919%),half7.376308ms
(+4.680% slower),maxL2=0.000286559;738GPU
projection dispatches per arm verifies6 total GEMVs/full request.1.5x resident
weights; not promoted for both widths. Packed GPU passes(maxL2=0.000514351),
full19.926/20.607ms,slower than F16 mixed. Packed CPU6 full5.844/5.900ms,
but maxL2=0.018877 exceeds pre-existing0.01 limit: numerical FAIL.
Packed mixed/coalesced full22.104/21.626ms,maxL2=0.012111,also FAIL.
Do not promote low precision timings. Residual correction under test:
reuse native Q8_K converter to compute input-minus-rounded-input,then add
a second packed dot per CPU projection; keep original packed weights.
Batch14,496-call sweep running. No energy/full-model token verification.
Evidence: M3 physical/pixel10pro-ffn-packed-coalesce-1/run1/SUITE_RESULT.json,
pixel_quant_residual.py,software/pixel10pro-cpu-gpu-v6.


## 2026-09-23 21:52 UTC - Pixel batch sweep14496 numerical/dispatch PASS; CPU-only leads8-row throughput

16 arms14496 calls PASS,max rowL2=0.000417706509,12480/12480
mixed backend intervals overlap. Tiled GPU dispatch counts cover all expected
projections at2/4/8 rows. Subgroups64/32 regress;128 remains best. Among mixed
arms,tile2 CPUc4096 full batch2=26.495ms;CPUc4864 batch4=34.899,batch8=63.232ms.
CPU-only6 threads leads batches2/4/8 at23.467/25.182/38.442ms;GPU-only tile2
is29.040/54.220/115.800ms. Shape-interleaved sweep has B1 controls29-32ms,
so it cannot replace the steady single-row13.06ms baseline. Confirmation
will group repeats by batch and warm up each batch separately.
A shared-memory tile (8output rows,16 K lanes/row,256-wide K tile,
12KiB shared at8 inputs) now builds; fewer accumulators/thread. Numerical
qualification pending. Residual correction18-arm4320-call run completed
normally; numerical audit pending. No energy/full-model tokens.
Evidence: M3 physical/pixel10pro-ffn-batch-sweep-1/run1/SUITE_RESULT.json,
PIXEL_FFN_BATCH_SWEEP_COMPARISONS.json,pixel_dense_batch_shared.glsl.


## 2026-09-23 21:59 UTC - Pixel residual correction4320 numerical PASS; shared-memory smoke672 PASS, speed not established

Packed CPU residual correction reduces max relativeL2 from0.018877 to
0.0005194;18 arms4320 numerical checks PASS. CPU6 full10.240 then15.523ms,
so speedup is not repeatable yet. CPU8 full19.365 then69.518ms; pairedCPU8
272.161/247.317ms: severe tail/performance FAIL,excluded from candidates.
CPU paired blocks also fail CPU6 speed13.209/15.364ms. Mixed corrected
ratios58.8/70.6/76.5 percent CPU take21.901/21.107/20.773ms full.
Fresh2160-call pinnedCPU6/CPU4 versus unboundCPU6 confirmation completed,
audit pending. Shared-memory8x256 tile672-call smoke numerical/dispatch
PASS,maxL2=0.000417672; full8-row GPU114.295/mixed82.638ms,not a speed win.
Final13536-call batch comparison prepared:20repeats and10warmups per batch,
CPU6 endpoint and mixed70.6/76.5/82.4 percent ratios,shared tile and packed
corrected CPU6; key modes repeated. No production change or energy/tokens.
Evidence: M3 physical/pixel10pro-ffn-residual-2/run1,
physical/pixel10pro-ffn-shared-smoke-1/run1,PIXEL_FFN_BATCH_CONFIRM_CONFIG.json.


## 2026-09-23 22:01 UTC - Pixel packed CPU residual correction confirmation PASS: six pinned threads selected

2160-call reversed confirmation numerical PASS,max relativeL2=0.000519396.
CPU6 cores2-7,persistent poll0; original Q4_K/Q6_K weights,4352-column blocks,
second packed dot corrects Q8_K activation residual. Repeats full10.3628/11.2208ms,
half5.2685/5.7021ms. Aggregate full10.791833 vs matched
13.694537ms (-21.196%),half5.485275 vs
7.442371ms (-26.297%). Full p99
12.578810 vs22.173970ms;half6.435170 vs
11.604590ms. Affinity PASS,repeats240/240 exact.
Against historical best13.05775/7.076942ms,full17.35% andhalf22.49% faster;
use matched controls above for primary comparison. CPU4 slower;unboundCPU6
close but slightly slower. No CPU8/paired blocks selected. Private candidate
only; energy,full-model tokens and server/USB integration NOT VERIFIED.
Final13536-call steady-by-batch comparison running.
Evidence: M3 PIXEL_PACKED_CPU_CANDIDATE.json,
physical/pixel10pro-ffn-residual-confirm-1/run1,software/pixel10pro-cpu-gpu-v8.


## 2026-09-23 22:20 UTC - Pixel final batch audit: numerical PASS, CPU endpoint stability and strict overlap FAIL

15 arms / 13,536 calls pass numerical checks; maximum row relative L2
0.000562194. Packed CPU6 repeats are byte-exact 960/960; full batch1/2
means 12.094/21.187 ms, matched reductions 20.24%/24.20%. Full batch8
104.662 ms regresses 27.16%. Repeated F16 CPU6/GPU split with CPU76.471%
and register tile2 gives full batch4/8 28.662/50.146 ms, matched reductions
39.07%/38.98%; both repeats pass mean and p99 comparisons. The unbound
F16 CPU6 endpoint is unstable: full B1 18.382 then121.389 ms, B2 18.064
then301.126 ms. No stable CPU-only multi-row winner declared.
Partition/timing consistency and dispatch coverage PASS; strict all-call
positive overlap FAIL 8639/8640. One measured B2 half call in04-reg6-c6656
started GPU after CPU finished; selected B4/8 intervals all overlap.
Cleanup PASS, boot unchanged, no worker or forward. A bounded reversed
6816-call follow-up pins the F16 CPU-only endpoint to cores2-7 and repeats
the mixed candidate; no kernel change or broad new sweep. Energy, server
latency and full-model tokens remain unverified.
Evidence: M3 PIXEL_FFN_BATCH_CONFIRM_COMPARISONS.json,
PIXEL_FFN_BATCH_CONFIRM_AGGREGATES.json, PIXEL_FFN_OPTIMIZATIONS_CLEANUP.json,
PIXEL_FFN_PINNED_ENDPOINT_CONFIG.json.


## 2026-09-23 22:35 UTC - Pixel four-direction tuning COMPLETE: selected candidates PASS; failures retained

Implemented and tested original packed weights, independent multi-row GPU
kernels, joint CPU/core/channel ratios, and full-block coalescing in private
variants. Ten completed suites:50,976 calls; native packed CPU/mixed arms
fail accuracy on920 calls (960 calls in four rejected arms retained).
Corrected packed CPU6 cores2-7 wins one row: full10.792 vs13.695 ms
(-21.20%), half5.485 vs7.442 ms (-26.30%), maximum relative L2=0.000519396.
Full/half p99 12.579/6.435 ms; both reversed repeats improve mean and p99.

Final pinned F16 CPU endpoint follow-up PASS6816 calls,maxL2=0.000417678,
960/960 repeated outputs exact,4800/4800 mixed intervals overlap. Full
batch2/4/8 CPU-only19.829/21.697/35.423 ms versus matched old controls
28.771/46.546/81.873 ms (-31.08%/-53.39%/-56.73%). Half9.907/10.873/
18.106 ms (-38.99%/-58.67%/-63.47%). CPU-only beats the tested mixed
CPU76.471%/GPU23.529% path in both comparisons; mixed full22.927/30.760/
48.166 ms. Absolute timings still vary: CPU batch8 repeats27.809/43.037 ms;
no sustained thermal guarantee. Batch2 packed versus pinned F16 CPU was
measured in different suites, so their close ranking lacks a direct pair.

Coalescing numerical PASS but speed goal FAIL: full0.92% gain,half4.68%
regression,50% more resident weights. Packed GPU and uncorrected packed CPU
not selected. Shared-memory mixed batch8 single39.576 ms remains exploratory.
Prior broad batch strict overlap FAIL1/8640 is retained; final follow-up PASS.
Selected-path max row error across these confirmations0.000562194 <0.01.
Worker-Werror,27 shader modules per batch build,ten Python pyflakes/parse
checks PASS. Cleanup PASS22:29 UTC: no worker,free Pixel lock,no forwards,
same boot. No job queued,production changes,commit or push. Energy,full-model
tokens,server latency,automatic batch switching and physical peaks NOT VERIFIED.
Evidence: M3 [full report](PIXEL_FFN_OPTIMIZATIONS_RESULTS.md),
PIXEL_FFN_OPTIMIZATIONS_RESULTS.json,PIXEL_PACKED_CPU_CANDIDATE.json,
PIXEL_FFN_BATCH_CPU_CANDIDATE.json,PIXEL_FFN_OPTIMIZATIONS_CLEANUP.json.


## 2026-09-24 01:30 UTC - Pixel SDK installation and real FFN TPU qualification complete

Installation PASS, private Tensor SDKv2.0/Python3.12/LiteRT2.2.0; compiler
and physical dispatch cover all6/6 FFN ops on TensorG5. 528 FFN calls /
1056 rows plus24 exact ADD calls PASS. Full FP16 pooled warm B1/B4 medians:
invoke33.500/31.806ms, worker34.671/35.998ms, USB40.646/47.213ms per batch.
Maximum FP16 relativeL2=0.000571125 including tail512; full=0.000547222.
FP16 repeated outputs exact across processes. BF16 accuracyPASS0.004686623
but speed improvementFAIL (worker34.952ms); FP16 remains preferable.
Initial speed goalFAIL against earlier CPU measurements; fresh matched CPU,
full-model tokens, server overlap, energy and integration NOT VERIFIED.
CleanupPASS, same boot, no worker/forward/pending job, Pixel lockfree.
Private probes only; compiler/probe/source hashes and failed prelaunch retained.
See [complete report](PIXEL_TPU_SDK_RESULTS.md),
[aggregate measurements](PIXEL_TPU_SDK_RESULTS.json), and
[SDK installation](../../../../../TPU_SDK/README.md).

## 2026-09-24 01:43 UTC - Pixel TPU runtime audit and vendor hardware counters PASS

Checked pinned LiteRT2.2.0 source and actual compiled FFN files: one
DISPATCH_OP per graph, resident executable/context reused, NPU-only execution;
no LiteRT CPU fallback. Compiler sharding=minimal, runtime performance hint
unset (vendor default/clocks unknown). One partition does not prove one
hardware kernel or ideal tiling.

Two finite instrumented arms PASS, 48 calls / 120 rows, max relative
L2=0.000547222. Vendor metrics count16 measured invocations after8 warmup.
B1/B4 hardware counter totals437395/438707us, means27.337/27.419ms.
Invocation remainder5.680/5.625ms; probe buffer handling1.221/3.873ms;
USB/ADB/socket/host remainder6.566/12.862ms; total40.805/49.779ms.
Hardware-time equality for4x useful FLOPs supports weight movement or fixed
tile work as bottleneck candidates; actual DRAM stalls/MAC occupancy/per-op
attribution NOT VERIFIED. No new CPU comparison or energy/token claim.
CleanupPASS: same boot, no worker/forward, lockfree, no pending job.
Evidence: M3 [runtime audit](PIXEL_TPU_RUNTIME_AUDIT.md),
PIXEL_TPU_RUNTIME_AUDIT.json, physical/pixel10pro-tpu-runtime-audit-1.


## 2026-09-24 03:24 UTC - Pixel F16 layout tuning

Private lossless-layout smoke numerical PASS:936 calls,max relativeL2=0.000519396,216/216 mixed intervals overlap. CPU19.314->19.167ms and GPU20.768->20.398ms are exploratory means from three warm repetitions; no speedup promotion. Wider GPU tile regresses22.125ms. Existing packed CPU12.347ms remains ahead. FMLAL and device-specific formats under test. No new server/energy/token result. [Raw suite](physical/pixel10pro-f16-layout-smoke-1/run1/SUITE_RESULT.json).

### 2026-09-24 03:26 UTC - NEON FMLAL smoke

Numerical PASS1152calls,newFMLAL maxL2=0.0001207. Full17.946ms best versus native controls18.251/18.533ms is exploratory; explicit prefetch loses. PackedCPU11.635ms leads. Device-specific CPU packed/GPU F16 sweep3456calls active. [Raw results](physical/pixel10pro-neon-fmlal-smoke-1/run1/SUITE_RESULT.json).

### 2026-09-24 03:34 UTC - Per-device formats

Numerical PASS3456calls;2880/2880 mixed intervals overlap. BestCPU6/82.353percent CPU full10.830ms fails against packedCPU10.298/10.227ms controls. CPU branch10.801ms,GPU8.939ms,merge0.008ms. CPU residual-pass fusion under test. [Raw suite](physical/pixel10pro-device-format-sweep-1/run1/SUITE_RESULT.json).

### 2026-09-24 03:39 UTC - Residual fusion smoke PASS

864 numerical checks PASS; fusedCPU288/288 outputs exactly match old packedCPU. Six threads9.772ms full vs11.654/11.663ms controls; preliminary16percent gain. Mixed94.118percent CPU9.745ms is a tie pending4560-call confirmation. [Investigation report](PIXEL_DEVICE_LAYOUT_RESULTS.md).

### 2026-09-24 03:48 UTC - Device-layout confirmation and shape check

PASS 4560 confirmation calls and 816 batch-shape calls. Mixed CPU 94.118 percent / GPU 5.882 percent gives full 7.844/8.098 ms (-22.71/-22.13 percent against matched old packed CPU). Fused CPU 8.095/10.217 ms shows large variation; a focused three-repeat endpoint check is active. B1/2/4 fused CPU outputs 144/144 exact to old packed CPU; max row relative L2 0.000536187. [Results](PIXEL_DEVICE_LAYOUT_RESULTS.md).

### 2026-09-24 03:56 UTC - Pixel NEON/GPU/layout tuning complete

Numerical/dispatch PASS: 86 arms, 13944 calls, max row relative L2 0.000536187; 5304/5304 mixed overlaps; 1632/1632 fused CPU outputs exact to old corrected CPU. Final three-repeat comparison: old CPU full/half 10.179/5.189 ms; fused CPU 8.122/4.135 ms (-20.21/-20.31 percent), full p99 8.873 vs 11.949 ms. CPU latency PASS. Mixed 94.118 percent CPU / 5.882 percent GPU full 7.928 ms (-22.11 percent vs old CPU, -2.39 percent vs fused CPU); full p99 9.712 ms and max 20.993 ms, so mixed tail improvement vs fused CPU FAIL. CPU fusion selected privately; mixed retained experimentally. F16 NEON/GPU-only layout gains remain small. Cleanup and build checks PASS. Peak utilization, energy, server latency and token identity unverified. [Complete report](PIXEL_DEVICE_LAYOUT_RESULTS.md), [measurements](PIXEL_DEVICE_LAYOUT_RESULTS.json), [CPU candidate](PIXEL_FUSED_CPU_CANDIDATE.json), [mixed candidate](PIXEL_DEVICE_MIXED_CANDIDATE.json).

### 2026-09-24 04:15 UTC - Pixel direct-packed GPU build

Private Q4_K/Q6_K vec4/pair8 shader build PASS, four SPIR-V modules validated. Original packed weights and F32 activations/accumulation. 960-call smoke active; numerical and speed pending. Failed predeployment GLSL builds retained. [Shader](pixel_packed_gemv.comp), [config](PIXEL_PACKED_GPU_SMOKE_CONFIG.json).

### 2026-09-24 04:29 UTC - Pixel packed GPU sweeps: numerical PASS, speed FAIL

Three finite suites complete: 3648 calls, 38 arms, all output/dispatch checks PASS.
Vec4/pair8 best25.045ms full against native packed19.995/20.832ms. Block16
scale reuse and direct Q6 scale loads improve the new shader to roughly20ms,
but control drift prevents a gain claim. Third sweep matches native pipeline
robustness and tests subgroup shuffles: best shared block16 full20.531ms,
shuffle20.659ms, native controls19.885-20.687ms, fusedCPU9.830ms.
Maximum relativeL2=0.000519396 including CPU. Longer2880-call confirmation
with repeated native packedGPU, F16GPU and CPU endpoints is active.
No energy, physical peak or server token result. [Report](PIXEL_PACKED_GPU_RESULTS.md).

### 2026-09-24 04:37 UTC - Pixel custom packed GPU tuning complete

Numerical/dispatch PASS: 50 arms, 6528 calls; 3360 custom calls; max row
relativeL2=0.000519396. Final repeated full/half means: native packedGPU
20.200/11.157ms, custom block16 WG256/rows8/SG128 19.854/11.088ms,
F16GPU20.085/11.558ms, fusedCPU8.122/4.120ms. Custom -1.71/-0.62percent
against pooled nativeGPU is small and provisional across sweeps. GPU-over-CPU
speed goal FAIL (2.44xCPU); no custom kernel promoted. Subgroup shuffles and
four/eight-weight variants rejected for speed. All original weights preserved.
Eight final SPIR-V modules and build/lint/hash checks PASS. Cleanup PASS,
no queued job. New batch kernels, server latency/tokens, energy and physical
GPU limits unverified. [Full tuning process and every arm](PIXEL_PACKED_GPU_RESULTS.md),
[measurements](PIXEL_PACKED_GPU_RESULTS.json), [shader](pixel_packed_gemv.comp).

### 2026-09-24 05:12 UTC - Pixel CPU row scheduling and paired SDOT complete

Numerical/dispatch PASS: 41 arms,7008 calls,9648 input rows; all5376 optimized
outputs byte-identical to native, max relativeL2=0.000562194. Independent B1/2/4/8
correctness PASS. Matched B1 full/half: native8.094/4.119ms, pairedSDOT+dynamic64
6.325/3.239ms (-21.855/-21.365percent). Full p998.778->7.422ms PASS; half maximum
worsens4.511->5.274ms despite improved p99. Native scheduling alone FAIL8.186ms;
SDOT single6.670ms and paired static6.565ms isolate the main gains. Row profiles
confirm redistributed work, but include startup/cold executions. Original weights
and correction formula preserved. Build/lint/hash/cleanup PASS, no queued job.
New private B1 candidate only; server/USB, energy, full tokens, physical peak and
new mixed CPU/GPU qualification remain unverified.
[All arms and implementation](PIXEL_PACKED_CPU_RESULTS.md),
[measurements](PIXEL_PACKED_CPU_RESULTS.json), [candidate](PIXEL_PACKED_CPU_CANDIDATE.json),
[kernels](pixel_packed_cpu.h).


## 2026-09-24 17:18 UTC - Rooted Pixel AOA qualification: PASS; idle robustness: FAIL

[Complete AOA report and all 53 arms](PIXEL_AOA_RESULTS.md).
Root/transport/numerical/build/cleanup PASS: stock-kernel Pixel at port 2-9.2,
accessory+adb 18d1:2d01, 5000 Mb/s; 9,936 FFN calls / 14,256 rows exact to the
archived packed CPU candidate, plus 6,820 validated echo exchanges. This uses
/dev/usb_accessory, not OP15's FunctionFS implementation. Private transport-only
worker changes; no production scheduler/server change or OP15-targeted operation.

Pooled matched A/B/B/A medians (ms): B1 full FFN ADB 10.565 -> AOA 6.152 (-41.77%),
B1 half 7.858 -> 3.918 (-50.14%), B2 full 13.882 -> 8.859 (-36.19%), B4 full
22.091 -> 15.872 (-28.15%). B1 continuous echo 1.970 -> 0.213 (-89.17%). Full B1
outside-compute RPC time 1.234 -> 0.349 ms. Compute also changes with call cadence;
the full FFN reduction is not an isolated wire speedup.

Retaining this advantage with 5 ms pauses FAIL: full FFN 29.824 -> 28.179 ms
(-5.51%); echo 3.090 -> 4.529 ms (+46.60%, regression). Temporary timed wake-lock
remedy FAIL: full FFN still 29.597 -> 28.724 ms (-2.95%). Android Dozing was observed,
but successful-suspend counters stayed 0 in the wake-lock arms; the CPU/USB/scheduling
cause remains unisolated. Initial Python-audit-between-calls arms are preserved and
excluded from headline timing; three startup-only harness failures reached zero FFN
calls. No kernel, CPU clock, governor, SELinux or persistent power setting changed.

Cleanup PASS: Pixel restored to 18d1:4ee7 and sys.usb.config=adb, root/boot ID unchanged,
no worker/forward/wake lock, phone lock available. A reset return of -5 reflected
re-enumeration; final USB/ADB identity verified. Server overlap, full generated tokens,
energy and multi-phone integration were not tested; no energy-saving claim.

## 2026-09-24 18:03 UTC: rooted Pixel integration requested

Packed CPU server prerequisite PASS: 744 calls and four64-token outputs exact;
host request saving10.887%/10.563% at half/full width, with slower decode.
First idle-timeout attempt retained as FAIL. Native two-helper binary rebuilt
and transport identity refreshed. Combined mechanism gate and trace pending.
Details and evidence: [second-phone report](../20260924-pixel-second-phone/README.md).
