# Pixel CPU/GPU unequal ratio tuning - 2026-09-23 20:40 UTC

Numerical, selected mean latency, selected observed p99, affinity and cleanup: **PASS**.
Recommend **39.706% CPU / 60.294% GPU** for the tested half/full FFNs. This is a tested operating point, not proof of a global optimum.

| Workload | CPU-only ms | GPU-only ms | Matched 50/50 ms | Selected 39.7/60.3 ms | Reduction vs 50/50 | Reduction vsCPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half 8704 | 9.233 | 12.015 | 8.166 | 7.077 | 13.34% | 23.35% |
| Full 17408 | 18.330 | 21.135 | 15.379 | 13.058 | 15.09% | 28.76% |

The selected full repeats are 13.06245 and 13.05305 ms; half repeats 7.12540 and 7.02848 ms. Both beat their bracketing 50/50 and CPU-only controls.
The lowest full mean is 42.647% CPU / 57.353% GPU at 13.02075 ms, only 0.037 ms (0.28%) below the recommendation. Its half p99 is 12.404 ms, worse than CPU 9.582 ms. The 39.706% choice passes both widths and is the fastest full-mean candidate among those passing both widths' mean/p99 checks.

## Measured process

| Stage | Arms | Calls | Numerical | Overlapping backend intervals | Timing treatment |
| --- | ---: | ---: | --- | ---: | --- |
| smoke | 5 | 180 | PASS | 180/180 | 2 warm-up repetitions |
| coarse | 15 | 1800 | PASS | 1559/1560 | 2 warm-up repetitions |
| refine | 17 | 4080 | PASS | 3600/3600 | 2 warm-up repetitions |
| confirm | 22 | 5280 | PASS | 4080/4080 | 10 warm-up repetitions |

Twenty mixed ratios from 12.5% through 75% CPU were tested, with CPU-only and GPU-only references. The coarse sweep favored 25-35%; refinement moved the fastest observed point to 38.235%; the reversed fine comparison establishes the recommendation above.
Initial controls ramped from 26 ms toward 14-15 ms over several repetitions. The warm-up policy was changed before refinement results were fetched: retain the original two-warm-up report, inspect repetitions 10-19 separately, and use ten warm-ups in final confirmation. Raw captures and every slow measured call are retained.
The final run uses 20 repetitions across six layers and two widths:240 calls per arm,60 warm calls per width per arm. Each fine ratio appears twice in opposite order, giving 120 warm samples per width.

## Final ratio table

| CPU share | GPU share | CPU columns per half | Half ms | Full ms | Half reduction vs 50 | Full reduction vs 50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 36.765% | 63.235% | 3200 | 7.156 | 13.264 | 10.72% | 12.64% |
| 37.500% | 62.500% | 3264 | 7.110 | 13.222 | 11.29% | 12.92% |
| 38.235% | 61.765% | 3328 | 7.320 | 13.592 | 8.67% | 10.48% |
| 38.971% | 61.029% | 3392 | 7.080 | 13.189 | 13.31% | 14.24% |
| 39.706% | 60.294% | 3456 | 7.077 | 13.058 | 13.34% | 15.09% |
| 42.647% | 57.353% | 3712 | 7.252 | 13.021 | 11.20% | 15.33% |

The fast region is broad; small differences among nearby ratios are not evidence of a precise universal optimum. The recommendation keeps one ratio for both request widths.

## Tails and branch timing

| Workload | Selected p50 ms | Selected p90 ms | Selected p99 ms | CPU p99 ms |50/50 p99 ms | Selected max ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half | 7.033 | 7.192 | 8.104 | 9.582 | 13.915 | 8.672 |
| Full | 13.048 | 13.303 | 14.029 | 19.034 | 27.189 | 14.750 |

| Workload | CPU branch ms | GPU branch ms | Overlap ms | Wait after CPU ms | Merge/conversion ms | Whole worker ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Half | 6.283 | 6.943 | 6.237 | 0.757 | 0.0145 | 7.077 |
| Full | 12.475 | 12.900 | 12.398 | 0.547 | 0.0121 | 13.058 |

Branch times overlap and must not be added. These are host/backend execution intervals, not correlated hardware kernel timestamps. CPU branch timing includes the small initial input/dispatch interval; whole-worker timing additionally includes outer work and logging.
Full matrix throughput is 40.955 GFLOP/s per worker-second; nominal weight bytes divided by worker time is 40.955 GB/s. These coincide for three F16 GEMVs at batch 1 (one matrix FLOP per nominal weight byte). This is not a DRAM-counter or hardware-peak measurement.

## Implementation and correctness

The private worker accepts `S42_PIXEL_CPU_HALF_COLUMNS` in 64-channel steps. For CPU count c and GPU count g=8704-c, the four resident blocks are [c, g, g, c], with CPU owning the first/last and GPU owning the middle two. The half request selects the last two blocks. Full requests retain two blocks per branch, and half requests one per branch across all ratios.
At the recommendation, the full FFN assigns 6912 channels to CPU and 10496 to GPU; half assigns 3456/5248. Each branch receives the same input, computes its own gate/up/SwiGLU/down, then the phone sums F32 outputs and returns one F16 result. CPU uses four persistent threads, poll 0, individually pinned to cores 4-7; the GPU submission thread runs on cores 0-3. The GPU retains `vec4_u1`, WG 128, SG 128, rows 8, unfused.
The dense shader guard now optionally accepts 64-aligned K through 8704 so uneven down projections use the selected shader. All 27 SPIR-V modules validate; the Android worker/backend builds and Python static checks pass. The new 50/50 path is byte-exact to the previously qualified path. Native code changes remain in isolated generated sources, with reviewable patches.
All 11,340 numerical checks PASS; maximum relative L2=0.000325483. Selected maximum=0.000286656; the selected two arms are 240/240 byte-exact to each other. 5,028 control outputs are byte-exact to the previous qualified CPU/GPU/50-path captures. Mixed ratios are not required to match CPU-only bytes.
Strict every-call overlap across the entire exploration is **FAIL** on 1/9420: coarse 35.294% half-width request 64 had CPUdone 5864 us and GPUstart 5874 us, zero overlap, worker 15205 us. It remains in all numerical/timing data. The failed partial audit is archived; the revised auditor records overlap failure separately from arithmetic/numerical checks. Final confirmation is 4080/4080 overlap PASS, including 480/480 at the recommendation.

## Limits and evidence

Qualification is phone-local, one token, layers 18-23, request widths 8704/17408, one deterministic input per layer. No new USB/server/energy/full-model-token/multi-row measurement or scheduler integration. Fine mode rejects other widths despite retaining 4352 in the existing protocol quantum. Clocks remain unlocked; final battery readings 33.2-36.4 C, with no continuous clock or DRAM counter capture. Empirical p99 from 120 samples per width is not a long-run tail guarantee.
Pixel cleanup PASS: normal finite worker exits, unchanged boot, free Pixel lock, no worker or Pixel forwarding. No desktop campaign was interrupted.

- [Machine-readable results and all comparisons](PIXEL_CPU_GPU_RATIO_RESULTS.json)
- [Selected configuration, qualified shapes and hashes](PIXEL_CPU_GPU_RATIO_CANDIDATE.json)
- [Final raw audit](physical/pixel10pro-cpu-gpu-ratio-confirm-1/run1/SWEEP_AUDIT.json)
- [Partition and overlap evidence](physical/pixel10pro-cpu-gpu-ratio-confirm-1/run1/DUAL_COVERAGE.json)
- [Warm-up policy](PIXEL_CPU_GPU_RATIO_ANALYSIS_POLICY.json)
- [Worker delta from the previous qualification](software/pixel10pro-cpu-gpu-v3/V2_TO_V3.patch)
- [Shader patch](software/pixel10pro-dense-gemv-ratio-v1/DENSE_SHADER.patch)
- [Build/static checks](PIXEL_CPU_GPU_RATIO_LOCAL_CHECKS.json)
- [Cleanup evidence](PIXEL_CPU_GPU_RATIO_CLEANUP.json)
- [Recompute this summary](analyze_pixel_cpu_gpu_ratios.py)
