# S11 Results

## E0 GPU-Board Acquisition - `2026-07-17 EDT`

Verdict: `GPU_BOARD_DIAGNOSTIC_RELIEF_FAIL`

The frozen eight-pair ABBA cohort completed without rerun or selection. All 16
slots passed exact-work, scheduled-placement, process-contamination, power
quality, power-limit, and p95-SLO gates. All eight treatment thermal artifacts
passed. `formal_claim` remains `NONE` because this is selected-A6000 NVML board
energy only.

| Metric | Server only | OP15 `[0,2)` + A6000 tail |
|---|---:|---:|
| Requests / generated tokens | 4,096 / 131,072 | 4,096 / 131,072 |
| Aggregate paid window | 683.16 s | 1,576.04 s |
| Average selected-board power | 288.97 W | 191.83 W |
| Selected-board energy | 197.42 kJ | 302.33 kJ |
| Maximum p95 B=8 group latency | 1.339 s | 3.298 s |
| Ready-memory relief | 0 | 888 MiB |

The treatment lowers average selected-A6000 power by about one third, but takes
2.31x as long. Its selected-board energy is 53.15 percent higher. Gross
control-minus-treatment relief is -104.92 kJ; after the frozen uncertainty
terms, the conservative lower bound is -125.82 kJ. This is not a marginal miss.

Evidence audit: controls carry 168-171 independent power updates and treatments
390-396; maximum power gap is 108,332 us and maximum process gap is 230,200 us.
All slots remained at P2. Eight thermal artifacts pass with zero logger errors.
The raw plan and summary hashes and exact integer results are pinned in
`ACQUISITION_RESULT.json`.

Decision: reject the fixed serial OP15 `[0,2)` route as an energy-saving
primitive. Do not sweep B=16 or another layer boundary. The validated resident
execution and memory-relief mechanisms may be reused only in a new experiment
that overlaps phone work with useful A6000 work across concurrent batches or
mixed jobs. Phone and total-system energy remain unknown.

## E0 Long Readiness - `2026-07-17 EDT`

Status: `FUNCTIONAL_EXACT_PASS; ACQUISITION_FROZEN; ENERGY_NOT_MEASURED`

The repaired Android binary was built with the recovered Snapdragon toolchain,
deployed to OP15 with a matching SHA-256, and exercised against the selected
A6000. Flash attention is disabled only for the phone stage because the v81
fused path caused an exact-token divergence in the first 32-token screen. The
explicit attention path is exact and remains scheduled on HTP.

| Readiness item | Server only | OP15 `[0,2)` + A6000 tail |
|---|---:|---:|
| Requests / generated tokens | 512 / 16,384 | 512 / 16,384 |
| Window | 85.084 s | 198.724 s |
| Median B=8 group latency | 1.329 s | 3.103 s |
| p95 B=8 group latency | 1.332 s | 3.236 s |
| Useful throughput | 6.02 req/s | 2.58 req/s |
| Selected-A6000 ready memory | 24,306 MiB | 23,418 MiB |

All generated token IDs match. Scheduled-buffer evidence passes: the phone's
layer compute is assigned to HTP0, with only `GET_ROWS` assigned to CPU; the
explicit attention graph uses HTP0 `SOFT_MAX`. Continuous phone thermal evidence
contains 304 samples, no logger errors, status 0, and a maximum 49.2 C HMX
reading. First-eight and last-eight treatment median group latencies are 3.132 s
and 3.086 s, so this run shows no late thermal slowdown.

The route releases 888 MiB of selected-A6000 memory but is about 2.34x slower at
the median. It is therefore not a latency or throughput win. This checkpoint
froze the request count, 3.5 s p95 SLO, identities, binary hashes, and readiness
artifact hashes in `ACQUISITION_FREEZE.json`; the subsequent acquisition and
failure verdict are recorded in the section above. Phone energy and total-system
energy remain unknown, and `formal_claim` remains `NONE`.

Raw readiness artifacts are under
`scratchpad/s11_e0_readiness_512_faoff_20260717/`.

## Historical Harness Hardening + CP1.3 / CP1.5 / CP2 - `2026-07-16 EDT`

Status: `HARNESS_HARDENED; HOST_BUILDS_PASS; ANDROID_BUILD_BLOCKED;
ON_DEVICE_READINESS_BLOCKED; NO_MEASUREMENT_RUN`

`formal_claim` remains `NONE`. `PHONE_ENERGY_UNKNOWN` and
`TOTAL_SYSTEM_ENERGY_UNKNOWN` are still mandatory. No `--measure` run, commit, or
push was performed. The dirty worktree from prior sessions is preserved; the S11
spike remains git-untracked.

### Fail-closed evidence pipeline

A measured timeline is counted valid only if every piece of evidence, recomputed
from its hashed on-disk bytes, is valid. `reverify_pairs` reopens each artifact
once, verifies its SHA-256 and record count, recomputes validity, and CONJOINS it
with the runner's stored per-slot validity before overwriting `measurement_valid`;
`aggregate_result` refuses any un-reintegrated pair and consumes the recomputed
validity. The four evidence types and their fail-closed triggers:

| Evidence | Recomputed from bytes | Invalidates on |
|---|---|---|
| Power (+ limit) | ZOH energy; >=100 in-window updates; <=250 ms gap; power-limit set + invariant; uncertainty from the recomputed limit | metadata-only limit forgery; mixed limit across the 16 slots |
| Process (CP1.2/1.4) | continuous selected-GPU compute-app telemetry; bounded probe; persisted probe errors | any non-driver PID; coverage gap; monitor error; missing bracket |
| Thermal (CP1.3, treatment only) | on-device OP15 log; boot id; monotonic uptime; bracketed continuous coverage | status != 0; empty sensors; gap; logger error; wrong phone; control is `NOT_APPLICABLE` |
| Placement (CP1.5) | executed backend-placement certificate | missing; duplicate; unexpected role; zero-compute; CPU fallback; wrong backend |

### CP1.3 - on-device phone thermal logger

The logger runs entirely inside one OP15 shell and issues no periodic ADB calls
during the paid window (the host only touches a stop flag at the boundary). It
records boot identity, a monotonic `/proc/uptime` microsecond stamp, the Android
thermal status, and every sysfs thermal-zone temperature. The host strictly
hashes, reopens, and revalidates it and requires bracketed continuous coverage,
bounded gaps, no logger errors, thermal status 0, and non-empty sensors. It binds
to every treatment timeline; the control is explicitly `NOT_APPLICABLE`.

### CP1.5 - scheduled-buffer placement certificate

The smallest instrumentation change is confined to
`examples/layersplit/layersplit.cpp`: an OBSERVE-ONLY `ggml_backend_sched` eval
callback wired through the public `cb_eval` seam. It does NOT edit gemma4.cpp,
llama-graph.cpp, or ggml_backend_sched. Returning false at `ask` keeps the
scheduler batching each split. For every graph node it reads the scheduled output
buffer (`t->buffer`) and tallies all compute ops separately from copy and metadata
nodes. This is placement evidence, not proof that an individual kernel completed.
Process status and exact output are independent completion/correctness gates.
Each run emits one machine-readable `PLACEMENTCERT` with route, role, layer
range, op/buffer counts, and status. The runner strictly parses, hash-binds, and
reintegrates it; missing, duplicate, mismatched, zero-compute, or wrong-buffer
evidence invalidates.

Host validation (non-measured): a monodriver control on the selected A6000 emitted

~~~text
PLACEMENTCERT {... role:monodriver, observed_backends:["CUDA0"],
compute_nodes:31584, cpu_fallback_nodes:0, cpu_fallback:false,
status:PLACEMENT_OK}
~~~

and the runner parsed, evaluated, and reintegrated the real certificate as valid.
The phone roles (`host_tail`, `phone_stage`) use the identical mechanism but need
the blocked Android build to exercise on device.

### CP2 - builds and readiness

| Target | Config | Result |
|---|---|---|
| Host CUDA release | `build-cuda` (GGML_CUDA=ON) | PASS rc=0, sha256 `1e76972e...b46cb5` |
| Host CPU release | `build-cpu` | PASS rc=0, sha256 `2626e5a2...7000fd` |
| Host ASan/UBSan | `build-s11-asan` (Debug, ASAN+UBSAN) | PASS rc=0, sha256 `0efbadf8...2565c7` |
| Android release | `build-s11-android` (arm64, Hexagon+OpenCL) | **BLOCKED** |
| OP15 offline suite | `test_fixed_route.py` | 93/93 PASS |

Android is blocked because the toolchain was removed from this host: NDK
`/opt/android-ndk-r28b` is gone, Hexagon SDK `/opt/hexagon/6.4.0.2` is gone, and
`build-s11-android/bin` is root-owned. The phone binary already staged on OP15
predates CP1.5, so a non-measured treatment readiness pair cannot be run and no
90-120 second request count or p95 SLO has been frozen. This blocker is reported,
not worked around.

### Adversarial review

A 5-dimension find-and-verify review workflow (25 agents) confirmed 6 findings and
0 defects in the C++ placement instrumentation. Fixed the two code findings
(`reverify_pairs` now conjoins the runner's stored validity so runtime-only vetoes
survive the byte recompute and a benign dropped sample invalidates rather than
aborting; a `headers[0]` IndexError was guarded) and added the four missing
load-bearing tests. Every gate has a red-before/green-after test; the offline
suite grew 34 -> 93.

## V3 Board-Energy Checkpoint

Status: `GPU_BOARD_ACQUISITION_NOT_RUN`

The v3 exploratory runner now requires eight complete pairs, at least 32
realized generated tokens per request, a predeclared p95 SLO, hashed raw power
artifacts from one GPU UUID, conservative one-second NVML boundary uncertainty,
and a 10 percent sum-all-pairs gate. Its offline suite passes 34 tests.

No v3 physical energy acquisition has run. The existing four-token functional
artifacts are ineligible. The remaining pre-acquisition harness gaps are
continuous selected-GPU process and phone-thermal monitoring, aggregate
reverification and reintegration of the hashed raw power stream, unchanged
power-limit binding, and a machine-readable no-fallback placement certificate.
The frozen OP15 B=8 measurement route is now enforced in the CLI;
current thermal/process coverage is boundary-only. See
`GPU_BOARD_ENERGY_PLAN.md`.

Any future result remains selected-A6000 `GPU_BOARD` steady-state energy only.
Phone energy and total-system energy are unknown.

Status:

`FUNCTIONAL_EXACT_PASS; GPU_MEMORY_RELIEF_OBSERVED; LATENCY_FAIL; ENERGY_NOT_RUN`

## Final Functional Run

Date: 2026-07-16 EDT

Route:

- control: Gemma-4 12B F16, all 48 layers on A6000 GPU 1;
- treatment: OP15 HTP owns layers `[0,2)`, the same A6000 owns `[2,48)`;
- OP12 was disconnected, so the compiled two-phone route was not run.

Workload:

- ten warmup requests before `DRIVER_READY`;
- seven measured requests;
- identical chat-formatted prompt;
- one batched 28-token prefill followed by single-token decode;
- four greedy tokens per request;
- persistent model/KV contexts reset between requests.

| Metric | Server only | OP15 `[0,2)` + server | Verdict |
|---|---:|---:|---|
| Generated token IDs, all 7 requests | `100,45518,107,236829` | identical | PASS |
| A6000 memory at ready | 24,580 MiB | 23,724 MiB | 856 MiB released |
| Request wall, median | 156.95 ms | 380.50 ms | treatment 2.42x slower |
| OP15 stage time, median | 0 | 223.56 ms | current bottleneck |
| Host-tail time, median | 156.93 ms | 156.74 ms | two layers removed |

The result proves the PIM-style route mechanics: the phone can own a contiguous
weight island, execute it from resident local memory, return only activations,
release 856 MiB of A6000 memory, and preserve exact greedy output across resets.

It does not prove a latency benefit. Batched prefill reduced the treatment from
about 1.96 seconds in the initial sequential plumbing run to 0.35 seconds, but
at decode batch 1 the two-layer HTP stage plus USB is still slower than leaving
the full request on the A6000. A production scheduler must therefore select
this route only when its memory/capacity or measured power benefit exceeds the
latency cost, or after multiple streams can batch the phone decode work.

The phone leg is thermally/DVFS non-stationary. Two otherwise identical final
runs produced treatment medians of 345.50 and 380.50 ms; the final OP15-stage
CoV was 9.23 percent. This fails a 5 percent stability gate and is another reason
not to make a latency or energy claim from this screen.

The logs also show a remaining capacity defect: both partial contexts allocate
KV buffers for attention layers outside their owned layer range. The OP15
`[0,2)` stage allocated a 1,280 MiB HTP KV buffer. Layer-window-aware KV sizing
should release additional phone memory and may reduce setup overhead, but it is
not required for the exactness result above.

## Measurement Boundary

No NVML energy run was performed. The historical runner required eight ABBA
pairs, at least 100 independent sensor updates per timeline, and the
vendor-stated +/-5 W uncertainty floor. V3 records p-state transitions instead
of rejecting them. Even a future passing result is limited to a diagnostic
selected-GPU board observation.

`GPU_BOARD_DIAGNOSTIC_*`

Phone, USB, CPU, DRAM, PSU, and charger energy remain outside that boundary.

## Corrections Before Final Run

Three benchmark artifacts were found and fixed before the result above:

1. the paid window originally ended at process exit and would have charged model
   teardown asymmetrically; it now ends at `DRIVER_DONE`;
2. the decode loop scanned the 262K vocabulary after every prompt token; it now
   requests logits only at the final prompt position and generation positions;
3. an absent second phone stage reported a few microseconds of clock overhead;
   it now reports exactly zero.
4. the original pipeline sent prefill as repeated B=1 decode calls; the final
   protocol sends the complete prompt as one batched phone and host-tail prefill.

## Verification

- CUDA release target: PASS.
- Android Snapdragon release target: PASS.
- Deterministic harness unit tests: 8/8 PASS.
- Real OP15 HTP route: 7/7 request outputs exact.
- Server request-wall CoV: 0.41 percent; OP15 stage CoV: 9.23 percent.
- No phone process or adb forward remained after the run.

Condensed provenance is in `FUNCTIONAL_RESULT.json`. Full logs are under
`scratchpad/s11_fixed_route_batched_r9/`.

No physical result is recorded until the host and Android builds pass, the
available OP15 route completes, and its token IDs are compared against the
resident server-only control.
