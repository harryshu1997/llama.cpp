# Pixel direct-packed GPU kernel results

Status: numerical and dispatch PASS, 6528 calls / 50 arms. GPU-over-CPU speed goal FAIL. The best repeated custom kernel improves the native GPU mean by 1.71% in the final suite; earlier suites do not establish a consistent gain. No custom kernel is promoted.

## Final repeated comparison

One input token, Qwen layers 18-23, K=5120. Full FFN width is 17408; half is 8704. Times are warm phone-local worker milliseconds per layer/FFN call, pooled equally across the six layers. They include graph execution and synchronization, but exclude USB/network/server time, model loading and shader compilation. They are not six-layer totals or per-token generation latency.

Final suite: 20 repetitions, first 10 excluded as warmup, all remaining samples retained. Each arm contributes 60 full and 60 half calls. Native GPU has four controls, CPU/F16/custom256x8 have two runs each; other candidates have one.

| Path | Half mean ms | Full mean ms | Full p99 ms | Full max ms |
| --- | ---: | ---: | ---: | ---: |
| Native packed GPU | 11.157 | 20.200 | 27.223 | 27.431 |
| Previously tuned F16 GPU 8x4 | 11.558 | 20.085 | 29.225 | 31.602 |
| New block16, WG256/rows8/SG128 | 11.088 | 19.854 | 26.284 | 26.870 |
| New block16, WG128/rows4/SG128 | 11.335 | 20.577 | 25.055 | 25.223 |
| New block16 shuffle, WG128/rows8/SG128 | 11.603 | 20.752 | 24.358 | 24.742 |
| Existing fused CPU, six pinned threads | 4.120 | 8.122 | 8.799 | 10.313 |

Best custom full/half mean changes against the pooled native GPU controls are -1.71%/-0.62%. It still takes 2.44x the tuned CPU full-FFN time. This is a new matched CPU observation; the earlier CPU result is not used as the control.

| Repeated custom256x8 arm | Full custom / surrounding controls ms | Full saving | Half saving |
| --- | ---: | ---: | ---: |
| 03-block16-w256-r8 | 19.580 / 20.144 | 2.80% | 1.02% |
| 08-block16-w256-r8 | 20.128 / 20.255 | 0.63% | 0.23% |

Both final repetitions have small positive mean differences, but absolute control drift and the preceding sweep, where the same variant was slower, prevent a stable improvement claim. The final p99 also improves slightly; the run is too short to establish sustained tail behavior.

## Kernel changes and tuning process

All changes are private extensions of the existing qualified Vulkan worker/backend. Original Q4_K/Q6_K bytes are retained; there is no new weight quantization, precision reduction, offline repacking or additional expanded weight copy. Inputs and accumulation are F32, protocol inputs/outputs remain F16. The FFN still uses existing 4352-channel blocks, four for full width and two for half.

1. Vec4 assigns independent lane groups to output rows and decodes four adjacent weights per step. Pair8 pairs groups 32 positions apart to reuse packed nibble loads. Both are slower than the native GPU.
2. Block16 handles 16 logical weights per lane, reuses scale metadata and moves scaling outside the inner dots. Direct Q6 scale loads remove the native shared-scale barriers. This recovers most of the initial regression.
3. Swept 128/256-thread workgroups, 2/4/8 output rows and 32/64/128-lane subgroups. Smaller subgroups do not produce a clear win.
4. Tested subgroup-shuffle row reductions to replace shared-memory reduction. They are slower here, especially 64-lane row reductions.
5. Build v5 matches the native quantized pipeline robustness setting. Separate shared-memory, vec4 and pair8 arms distinguish that change from shuffles. Earlier v3/v4 used default robustness, so those runs are exploratory comparisons.
6. Repeated the closest candidates with native packed, tuned F16 and fused CPU controls, reversing the CPU/F16 endpoint order. No CPU+GPU ratio sweep was added because the standalone GPU change is small and inconsistent.

## All measured arms

The first three suites use eight repetitions/four warmup, or 24 warm calls per width/arm. Their p99 values are descriptive only. All startup/shape hashes, finite exits, reference norms and actual custom Q4/Q6 dispatch counts are checked by the suite audit.

### smoke: 960 calls / 10 arms, numerical PASS

[Raw audit/result](physical/pixel10pro-packed-gpu-smoke-1/run1/SUITE_RESULT.json). Repetitions 8, warmup 4.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-cpu | 9.731 | 5.032 | 12.364 |
| 01-f16-gpu | 20.332 | 11.420 | 24.535 |
| 02-packed-control | 20.832 | 11.185 | 25.832 |
| 03-vec4-r2 | 27.947 | 15.216 | 31.822 |
| 04-vec4-r4 | 25.476 | 14.192 | 27.180 |
| 05-vec4-r8 | 29.953 | 16.278 | 32.385 |
| 06-pair8-r2 | 25.822 | 14.373 | 27.881 |
| 07-pair8-r4 | 25.045 | 13.655 | 27.081 |
| 08-pair8-r8 | 29.229 | 15.956 | 31.152 |
| 09-packed-control | 19.995 | 11.052 | 26.250 |

### block16: 1152 calls / 12 arms, numerical PASS

[Raw audit/result](physical/pixel10pro-packed-gpu-block16-1/run1/SUITE_RESULT.json). Repetitions 8, warmup 4.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-packed-control | 22.778 | 12.478 | 27.772 |
| 01-cpu | 9.745 | 5.024 | 12.170 |
| 02-block16-w128-r2 | 21.823 | 11.714 | 25.577 |
| 03-block16-w128-r4 | 20.438 | 11.134 | 23.384 |
| 04-block16-w128-r8 | 21.095 | 11.940 | 24.911 |
| 05-block16-w256-r2 | 20.544 | 11.702 | 24.157 |
| 06-block16-w256-r4 | 19.992 | 11.181 | 24.770 |
| 07-block16-w256-r8 | 19.757 | 10.963 | 25.987 |
| 08-packed-control | 19.626 | 10.749 | 24.858 |
| 09-block16-sg64 | 19.921 | 11.200 | 24.107 |
| 10-block16-sg32 | 19.857 | 11.014 | 23.926 |
| 11-packed-control | 19.910 | 10.802 | 29.245 |

### shuffle: 1536 calls / 16 arms, numerical PASS

[Raw audit/result](physical/pixel10pro-packed-gpu-shuffle-1/run1/SUITE_RESULT.json). Repetitions 8, warmup 4.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-packed-control | 20.392 | 11.352 | 25.325 |
| 01-cpu | 9.830 | 5.035 | 12.303 |
| 02-block16-w128-r4 | 20.793 | 11.456 | 26.570 |
| 03-block16-w256-r8 | 20.531 | 11.468 | 26.977 |
| 04-pair8-w128-r4 | 24.955 | 13.410 | 29.727 |
| 05-vec4-w128-r4 | 25.434 | 14.247 | 27.469 |
| 06-packed-control | 20.231 | 11.041 | 26.427 |
| 07-shuffle-w128-r2 | 40.155 | 21.164 | 41.094 |
| 08-shuffle-w128-r4 | 23.554 | 12.566 | 41.560 |
| 09-shuffle-w128-r8 | 20.659 | 11.669 | 23.377 |
| 10-shuffle-w256-r4 | 39.917 | 21.012 | 40.857 |
| 11-shuffle-w256-r8 | 22.832 | 12.648 | 25.362 |
| 12-packed-control | 19.885 | 10.938 | 24.180 |
| 13-shuffle-sg64 | 24.503 | 13.166 | 30.039 |
| 14-shuffle-sg32 | 24.933 | 13.259 | 31.035 |
| 15-packed-control | 20.687 | 12.055 | 34.592 |

### confirm: 2880 calls / 12 arms, numerical PASS

[Raw audit/result](physical/pixel10pro-packed-gpu-confirm-1/run1/SUITE_RESULT.json). Repetitions 20, warmup 10.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-packed-control | 19.837 | 10.852 | 26.456 |
| 01-cpu | 8.083 | 4.113 | 8.755 |
| 02-f16 | 19.594 | 11.218 | 24.346 |
| 03-block16-w256-r8 | 19.580 | 10.975 | 25.571 |
| 04-packed-control | 20.451 | 11.324 | 26.411 |
| 05-block16-w128-r4 | 20.577 | 11.335 | 25.055 |
| 06-shuffle-w128-r8 | 20.752 | 11.603 | 24.358 |
| 07-packed-control | 20.401 | 11.240 | 27.036 |
| 08-block16-w256-r8 | 20.128 | 11.201 | 26.533 |
| 09-f16 | 20.576 | 11.897 | 30.333 |
| 10-cpu | 8.161 | 4.126 | 9.409 |
| 11-packed-control | 20.110 | 11.213 | 26.951 |

## Validation, interpretation and limits

Numerical PASS: 6528 calls, max per-row relative L2 0.000519396 against archived independent CPU references (threshold 0.01). The 3360 custom-GPU calls have max relative L2 0.000514465. Calls reuse archived inputs for each layer/width; they are not thousands of independent prompts. Tolerance checks do not establish bit-identical full-model tokens.

Custom dispatch PASS: 28736 Q4_K and 3592 Q6_K projection dispatches, including startup, across 29 custom arms. All six layers execute the selected kernels, including Q6 down projections on layers19/22. Batches2/4/8 have compiled pipelines but were not physically qualified in this experiment; no candidate was promoted, so qualification was not broadened.

Build PASS: v3/v4/v5 validate 4/6/8 SPIR-V modules respectively. The final eight modules target Vulkan1.2. Shader compilation uses -Werror; C++ uses the inherited compiler warning flags. Three changed Python files pass pyflakes and parsing. Source/binary snapshot hashes and ASCII/whitespace checks pass. Builds v1/v2 failed before deployment on an integer ternary and reserved GLSL identifier; their logs remain available.

Actual runtime reports PowerVR D-Series DXT-48-1536 MC1, native warp size128, 32KiB shared memory, int-dot0 and no matrix cores exposed by this backend. This does not establish the complete hardware capability. Packed and F16 GPU FFNs both remain near20ms despite the smaller packed weights. That observation suggests unpacking/arithmetic, occupancy or dispatch/synchronization can offset the traffic reduction; none is isolated by these worker timings. DRAM transactions, register pressure, stall counters and physical compute/bandwidth utilization were not measured. This is not proof that the GPU has reached its hardware limit.

Cleanup PASS: all finite workers exit normally, same boot, no remaining worker or Pixel ADB forward, Pixel lock reacquired nonblocking and released. Initial cleanup lock probe omitted explicit fd inheritance and returned1; the corrected 9>&9 probe succeeds. No job remains queued. Hardware work used the Pixel lock and ADB5037 only. No reboot, root, clock/power change, desktop model run or OP11/OP15 change. Production dispatch and defaults remain unchanged. No new USB/server latency, energy, full-model token, batch or sustained thermal qualification.

## Artifacts

- [Shader](pixel_packed_gemv.comp) and [private builder](build_pixel_packed_gemv.py).
- [Final immutable build](software/pixel10pro-packed-gemv-v5/BUILD_PROVENANCE.json), [backend patch](software/pixel10pro-packed-gemv-v5/PACKED_BACKEND.patch), [SPIR-V checks](software/pixel10pro-packed-gemv-v5/SPIRV_VALIDATION.json).
- [Confirmation configuration](PIXEL_PACKED_GPU_CONFIRM_CONFIG.json), [aggregate measurements](PIXEL_PACKED_GPU_RESULTS.json), [build checks](PIXEL_PACKED_GPU_BUILD_CHECKS.json), [cleanup](PIXEL_PACKED_GPU_CLEANUP.json).

Final GPU library SHA256: `892bf36afff1c3b964afe421a25b9b6cbf0e55de949bee093c91038114cd4fa1`. Worker remains the previously qualified layout-worker-v5, SHA256 `8604d14703ba64b8f62f63b901e8809ffb80e2a65cc758ef756cfc00cca5612d`. The best experimental GPU settings are `S42_PIXEL_Q_SHADER=block16`, `S42_PIXEL_Q_WG=256`, `S42_PIXEL_Q_ROWS=8`, `S42_PIXEL_Q_SUBGROUP=128`; F32 input and original packed weights are required. CPU remains the preferred measured B1 path.
