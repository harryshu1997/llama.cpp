# Pixel 10 Pro measured bandwidth and ways to use it

2026-09-23. **PASS**:55 bounded bandwidth arms,8594 measured engine passes,
all CPU slice/chunk and every GPU-output checksum/copy checks. Cleanup PASS.
These are sustained logical streaming rates, not a measured theoretical DRAM
peak or hardware memory-controller counters. Use decimal GB/s throughout.

| Repeated configuration | Sustained wall rate | GPU device-timestamp rate |
| --- | ---: | ---: |
|CPU, six cores2-7, dynamically assigned4MiB chunks|30.42-30.57GB/s|n/a|
|GPU, vector4,128-thread groups,16384 invocations|37.25-38.04GB/s|43.19-43.71GB/s|
|CPU core7 plus GPU concurrently|46.47-46.58GB/s aggregate|not additive|

Highest exploratory GPU observation:38.694GB/s wall,44.163GB/s device.
Repeated measurements are the headline. The separate Vulkan copy arm reaches
24.348GB/s **read plus write**; it copies12.174GB of payload per second and
must not be compared directly with the read-only numerator above.

## What improved bandwidth

1. **Balance CPU work dynamically.** Equal slices across six cores give
   24.059GB/s. A shared queue of4MiB chunks gives
   30.421/30.565GB/s, **26.74% higher**.
   In the last measured pass, core7 reads220MiB while cores2-6 read
   44/76/84/44/44MiB. This replaces the fixed85.33MiB/core partition that
   makes the fast core wait.1MiB chunks give30.552GB/s with six cores and
   30.527 with eight; eight cores alone do not improve over the repeated
   six-core choice. Eight static equal slices previously fell to13.199GB/s.
2. **Use contiguous vector GPU loads.** Scalar reads repeat at
   27.335/27.762GB/s; vector4 repeats average
   37.696GB/s (**36.84% higher**).
   The FFN GPU candidate already uses vector loads, so this raw-stream gain
   is not a new FFN speedup. More unrolling, larger workgroups, more total
   invocations and explicit CPU prefetch did not establish further gains.
3. **Overlap CPU and GPU reads.** One fast CPU core plus GPU repeats at
   46.515GB/s, **22.91% above matched GPU-only controls**.
   CPU and GPU read disjoint512MiB allocations. They slow individually to
   about16.6-16.9 and29.7-29.9GB/s; their isolated rates cannot be added.
   Six/eight dynamic CPU threads plus GPU give45.537/45.078GB/s, below the
   one-core combination. Both measurement intervals overlap at least99.47%.

| Concurrent repeat | CPU GB/s | GPU GB/s | Aggregate GB/s | Matched GPU-only GB/s | Aggregate gain |
| --- | ---: | ---: | ---: | ---: | ---: |
|pixel10pro-bandwidth-confirm-1/07-joint-cpu1|16.889|29.786|46.577|37.917|22.84%|
|pixel10pro-bandwidth-confirm-1/13-joint-cpu1|16.796|29.738|46.474|37.648|23.44%|
|pixel10pro-bandwidth-dynamic-1/10-joint-cpu1-static|16.639|29.928|46.495|37.968|22.46%|

## Improvement in the actual FFN path

Persistent CPU threads make a larger practical difference than the raw-stream
GPU knobs. The repeated six-thread CPU worker drops from44.291 to18.291ms
for a full one-token FFN, **58.70% less latency /2.421x throughput**. GPU
references average20.610ms; the tuned CPU is11.25% lower latency here.
Logical weight rate rises12.074 to29.237GB/s. This is close to the30.4-30.6
CPU streaming result, but the different workloads and unlocked clocks mean
that this ratio is not a measurement of physical bus utilization.

A private Android platform-guard fix also makes CPU affinity take effect.
Before it, ggml silently accepts masks but leaves threads unpinned. Actual
thread masks now pass; unpinned six-thread execution remains slightly faster
than strict six-core pinning, so the selected candidate uses the original CPU
library with a persistent pool. Arithmetic is unchanged. Across2520 FFN calls,
all saved-reference checks pass and all2040 CPU outputs are byte-identical to
the original Pixel CPU. No production defaults are changed.

[Full FFN tuning process, arms and limits](PIXEL_CPU_TUNE_RESULTS.md),
[private qualified CPU settings](PIXEL_CPU_TUNED_CANDIDATE.json).

## Measurement method and boundaries

- 512MiB working set per engine; combined reads use1GiB total. Allocation,
 initialization, shader creation and correctness checks are outside timing.
 Eight complete warmup passes precede2s exploratory or3s confirmation loops.
 All byte counts cover complete passes, not selected fastest samples.
- CPU NEON reads XOR data into observable outputs using four accumulators.
 Fixed slices are64-byte aligned; dynamic chunks have independent expected
 checksums, unique atomic queue indices and exact per-pass read coverage.
 Only benchmark threads are pinned. No process priority, clock or phone
 system setting is changed.
- GPU Vulkan reads use four128MiB descriptors because the device reports a
 128MiB maximum storage-buffer range. All four dispatches are measured.
 Memory flags7 mean device-local, host-visible, coherent; no timed staging
 or host copy. Scalar/vector variants cover identical input byte counts.
 Read bandwidth excludes small checksum writes:256KiB per512MiB pass for
 16384 invocations,1MiB for65536,16MiB for1048576. Every checksum is checked.
- GPU wall time includes submit, fence wait and timestamp retrieval. Device
 timestamps include the recorded dispatches and barriers, without host
 submission/wakeup overhead. Do not mix the two columns as one measurement.
- Combined loops start at a barrier and run for the same nominal duration;
 aggregate bytes divide by the full union of their measured intervals.
 The reported overlap fraction includes the final partial scheduling skew.
- Run-local locks and existing-worker guards serialize Pixel experiments.
 The desktop only stages files and transfers logs; OP15/OP11 and desktop
 model workloads are untouched. Phone-local measurements exclude USB/RPC.
- Battery temperature spans28.2-35.5C across the bandwidth runs. Clocks are
 not locked; before/after CPU/GPU frequency snapshots are archived, not
 continuous telemetry. Phone wall date is wrong; launch records use host UTC.
- The first native build failed on an unavailable pthread-affinity symbol
 and a Vulkan flag type mismatch; both were fixed before hardware execution.
 Builds with warnings treated as errors and all four SPIR-V modules pass.
 No raw bandwidth arm failed correctness. All phone executables exit normally;
 boot ID is unchanged and no worker remains.
- No physical DRAM counters, official theoretical peak, energy measurement,
 concurrent CPU/GPU FFN split/merge, server timing or full-model token test.
 The concurrent-stream result establishes shared read capacity only.

## Evidence

[Measured summary](PIXEL_BANDWIDTH_RESULTS.json),
[24-arm exploration](physical/pixel10pro-bandwidth-1/run1/RESULT.json),
[17-arm confirmation](physical/pixel10pro-bandwidth-confirm-1/run1/RESULT.json),
[14-arm dynamic partition test](physical/pixel10pro-bandwidth-dynamic-1/run1/RESULT.json).
The run directories retain configs, exact commands, binaries/shaders, hashes,
per-pass timing arrays, checks, raw logs and temperature/frequency snapshots.
[Native benchmark](pixel_bandwidth.cpp), [shader](pixel_bandwidth.comp),
[build tool](build_pixel_bandwidth.py), [runner/auditor](pixel_bandwidth_run.py),
[dynamic build provenance](software/pixel10pro-bandwidth-v3/BUILD_PROVENANCE.json).
