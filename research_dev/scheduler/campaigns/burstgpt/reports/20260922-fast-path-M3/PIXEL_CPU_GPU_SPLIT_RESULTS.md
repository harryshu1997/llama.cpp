# Pixel concurrent CPU/GPU FFN split: implementation and measurements

2026-09-23 19:44 UTC. Functionality, branch overlap, actual affinity and cleanup **PASS**.
Mean worker latency improvement **PASS** in two repeats. P99 versus CPU-only
**FAIL**: the combined path retains occasional long calls.

| One-token phone workload | Tuned CPU6, matched controls | GPU references | CPU4 + GPU, pinned | Mean saving vs CPU | Mean saving vs GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Half FFN,8704 channels | 9.169 ms | 12.057 ms | **8.086 ms** | 11.81% | 32.93% |
| Full FFN,17408 channels | 18.210 ms | 21.014 ms | **15.334 ms** | 15.79% | 27.03% |

These are phone worker times, including graph setup, input handling inside the
worker interval, execution, result retrieval and the dual-path merge. They are
not GPU kernel timestamps or USB round trips. Both backends operate on the same
FFN layer concurrently. The server and phone energy were not measured.

## Implementation

A private variant reuses the existing secondary-backend worker and its helper
thread. It assigns existing4352-channel weight blocks to CPU or GPU instead
of splitting every block into smaller matrices. This preserves the tuned GPU
kernel shapes. Gate/up output rows and corresponding down input columns stay
resident on their assigned backend; the input is shared logically and copied
into each backend's input tensor. The phone sums both F32 partial outputs and
converts once to F16 for the existing single response.

The selected full split gives CPU blocks0/3 and GPU blocks1/2,8704 channels each.
For the half-width suffix, CPU block3 and GPU block2 supply4352 channels each.
CPU uses four persistent threads, polling disabled, pinned individually to
cores4-7. The GPU submission thread runs on CPU cores0-3. Other Vulkan driver
threads retain their existing affinity. The GPU shader is unchanged:vec4_u1,
workgroup128,subgroup128,rows8. The previously qualified private Android CPU
library guard fix is required for CPU affinity to take effect.

The canonical worker belongs to another ongoing experiment and was not edited.
The build captures that source plus a reviewable patch and reuses the qualified
runtime libraries. The worker's CPU-only mode matches all36 smoke outputs from
the prior qualified CPU worker. No production default or scheduler contract was
changed. Qualification covers only the recorded Qwen shard, one row, layers18-23
and widths8704/17408.

## Tuning process

1. Private v1 build with warnings treated as errors: **PASS**.180-call smoke:
   numerical **PASS**,36 partition/overlap records **PASS**. The first CPU6
   50/50 arm is slower:full22.128 versus18.280ms CPU,half14.647 versus9.207ms.
2. Sweep1/2/4/6 CPU threads and full CPU/GPU ratios75/25,50/50,25/75:
   19 arms,2280 calls **PASS** numerically,1440 overlapping dual calls. Every
   candidate loses to its surrounding CPU6 controls. Best unbound CPU4 50/50:
   full19.349 versus18.214ms (+6.23% latency),half10.062 versus9.162ms (+9.82%).
3. Affinity correction: CPU compute threads use their qualified pinned masks.
   A secondary helper created after a pinned CPU warmup can inherit the main
   CPU thread's restricted mask. Private v2 explicitly assigns the GPU host
   thread its own mask. Its actual mask and every CPU pool thread are checked
   in `/proc`; build and all eight recorded affinity checks **PASS**.
4. Reversed-order confirmation:11 arms,2640 calls,20 repetitions per layer and
   width. The first two repetitions are warmups;108 warm samples/width/arm.
   CPU4 pinned wins in both orders against both surrounding CPU controls.
   CPU6 pinned improves means but is less stable. Unbound4 varies materially.

| Confirmation arm | Half mean ms | Full mean ms | Full p90 ms |
| --- | ---: | ---: | ---: |
| 00-gpu | 12.315 | 21.257 | 24.229 |
| 01-cpu6 | 9.146 | 18.207 | 18.292 |
| 02-dual4-unbound | 13.135 | 23.111 | 29.859 |
| 03-dual4-pinned | 8.070 | 15.274 | 16.774 |
| 04-dual6-pinned | 8.469 | 15.790 | 21.892 |
| 05-cpu6 | 9.204 | 18.245 | 18.354 |
| 06-dual6-pinned | 9.031 | 17.585 | 24.794 |
| 07-dual4-pinned | 8.103 | 15.395 | 18.543 |
| 08-dual4-unbound | 8.885 | 16.474 | 20.708 |
| 09-cpu6 | 9.120 | 18.141 | 18.206 |
| 10-gpu | 11.799 | 20.771 | 23.292 |

CPU matched means use the nearest controls before and after each selected arm,
then average the two pairs. GPU references bracket the confirmation. No fastest
individual call or selectively trimmed sample supplies the headline.

## Where the time goes

Mean milliseconds across both selected pinned4 arms. Branch intervals overlap;
do not add CPU and GPU time. CPU elapsed includes the small dispatch interval.
"Worker total" additionally includes the outer layer lookup, logging and final
output-vector copy. Response hashing/framing is outside that interval.
Host-thread intervals do not establish exact
hardware kernel overlap duration.

| Interval | Half FFN | Full FFN |
| --- | ---: | ---: |
| CPU branch elapsed | 7.405 | 15.029 |
| GPU branch elapsed | 7.811 | 14.311 |
| Observed branch overlap | 7.209 | 14.046 |
| Wait after CPU branch | 0.646 | 0.277 |
| Merge and F16 conversion | 0.013 | 0.008 |
| Dual execution total | 8.065 | 15.314 |
| Worker total | 8.086 | 15.334 |

Full effective matrix rate is34.874GFLOP/s per
worker second. This divides the534.774MFLOP FFN matrix work by complete worker
latency; it is not pure-kernel throughput or a physical memory-counter result.
Merge is only0.0081ms. Placement of CPU compute and
GPU submission work, together with fewer CPU threads, establishes the gain;
this experiment does not isolate the contribution of each placement change.

## Correctness, tails and limits

- **5100/5100** saved desktop-reference checks PASS across smoke, sweep and
 confirmation; maximum relativeL2=0.000325483.
 Selected candidate maximum relativeL2=0.000279321.
 All2916 dual calls verify exact column coverage and positive branch overlap.
 The two selected arms have240/240 byte-identical outputs, also exact to the
 corresponding unbound4 outputs. CPU/GPU partial-sum results differ in low bits
 from CPU-only; full-model token equality was not tested.
- Selected full p50/p90/p99=14.904/
 17.816/25.294ms;
 CPU controls p90/p99=18.297/
 18.716ms. Selected maximum29.978ms versus
 CPU maximum19.425ms. Half p99=11.009ms versus
 CPU9.514ms. Mean improvement does not
 establish a better tail-latency guarantee. Candidate quantiles pool216 warm
 samples/width; CPU quantiles pool324 distinct warm control samples/width.
- The same stored inputs repeat for each layer. Clocks are unlocked; battery
 temperature during confirmation spans31.7-33.7C. Before/after frequency
 snapshots are archived but are not continuous telemetry. The experiment
 does not establish behavior at sustained server cadence, multi-row batches,
 different inputs/models, USB transport or phone/server energy.
- Private Pixel lock only. No desktop model execution, OP15/OP11 action,
 clock change or phone reboot. All35 arms exit normally; cleanup PASS at
 2026-09-23T19:40:56.687149+00:00. No Pixel worker or forwarding entry remains.
- Python static checks, raw-output audits and prior1320-call CPU audit PASS.
 Missing dual records and incorrect column coverage are rejected by the audit.

## Artifacts and reproduction

[Selected settings and hashes](PIXEL_CPU_GPU_SPLIT_CANDIDATE.json),
[measured summary](PIXEL_CPU_GPU_SPLIT_RESULTS.json),
[worker builder](build_pixel_cpu_gpu.py),
[private native patch](software/pixel10pro-cpu-gpu-v2/CPU_GPU.patch),
[phone replay and audit](pixel_local_sweep.py),
[confirmation config](PIXEL_CPU_GPU_SPLIT_CONFIRM_CONFIG.json),
[final cleanup](PIXEL_CPU_GPU_SPLIT_CLEANUP.json).

Raw arms: [smoke](physical/pixel10pro-cpu-gpu-smoke-1/run1/RESULT.json),
[sweep](physical/pixel10pro-cpu-gpu-sweep-1/run1/RESULT.json),
[confirmation](physical/pixel10pro-cpu-gpu-confirm-1/run1/RESULT.json).
Each retains request packets, input/reference hashes, F16 outputs, worker logs,
thread masks, timing records and launch commands. The isolated source snapshot
in v1/BASE_SOURCE.cpp is the build input; later canonical-worker edits are not
silently incorporated into this result.

From this report directory, reproduce the output/timing audit without hardware:

```sh
python3 analyze_pixel_gemv.py physical/pixel10pro-cpu-gpu-confirm-1/run1
```

Rebuild the private worker into a new, absent output directory:

```sh
python3 build_pixel_cpu_gpu.py software/pixel10pro-cpu-gpu-v1/BASE_SOURCE.cpp /tmp/pixel-cpu-gpu-rebuild
```

For a hardware repeat, prepare a fresh directory with `pixel_local_sweep.py`
and the saved config/candidate libraries; use its generated `RUN_PHONE.sh`.
That script takes the Pixel lock and refuses an existing FFN worker. Exact
staging and launch argv are in each run's `LAUNCH.json`. Reuse the archived
reference inputs; a phone-only replay does not require the desktop rig lock.
