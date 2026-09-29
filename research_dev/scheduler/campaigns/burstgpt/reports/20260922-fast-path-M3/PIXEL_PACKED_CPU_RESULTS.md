# Pixel dynamic CPU scheduling and paired NEON results

Status: COMPLETE. Numerical/dispatch PASS; combined mean latency and full p99 PASS. Native-kernel scheduling alone FAIL. The paired SDOT kernel with 64-row scheduling is the new private B1 CPU candidate. Production defaults are unchanged.

## Repeated result

Times are warm phone-local milliseconds per FFN layer and one input token, averaged equally over Qwen layers18-23. K=5120; half/full hidden widths8704/17408. The worker includes graph construction/execution, activation, reductions and synchronization. Model loading, shader creation, USB and server time are excluded. These are not six-layer totals or token-generation latency.

The confirmation has20 repetitions per arm and excludes the first10. No warm outliers are removed. Four native controls contribute240 samples per width; each repeated candidate contributes120. The second candidate sequence reverses the first.

| Configuration | Half mean ms | Full mean ms | Full p99 ms | Full reduction vs native |
| --- | ---: | ---: | ---: | ---: |
| Existing fused CPU, native dots | 4.119 | 8.094 | 8.778 | 0.00% |
| Native dots + dynamic64 | 4.156 | 8.186 | 8.876 | -1.13% |
| New SDOT single dots, static rows | 3.387 | 6.670 | 7.494 | 17.60% |
| New paired SDOT, static rows | 3.323 | 6.565 | 8.761 | 18.89% |
| SDOT single dots + dynamic64 | 3.355 | 6.482 | 7.772 | 19.92% |
| Paired SDOT + dynamic64 | 3.239 | 6.325 | 7.422 | 21.86% |

Native full controls8.086/8.099/8.092/8.100ms and selected repeats6.317/6.334ms are stable within this run. Combined full/half reductions21.855/21.365percent; full p99 improves15.451percent. Scheduling alone is1.13percent slower at full width, correcting the promising first smoke result. The selected half-width maximum5.274ms exceeds the native maximum4.511ms; half p99 improves4.464->4.004ms. These descriptive tails are not a sustained latency guarantee.

Paired decoding adds about1.56percent over the new SDOT single-dot static path. Dynamic64 adds about3.66percent over paired static. The largest measured difference is the SDOT-enabled single-dot rewrite versus the existing native dot routines (17.60percent). That comparison also changes the loop implementation; it does not isolate instruction selection from every compiler/code-generation difference.

Useful mathematical FFN throughput is 84.55GFLOP/s, excluding extra residual-correction arithmetic. Unique packed weight bytes per worker second are 24.99GB/s (158064640 bytes per layer on average). Neither is a physical utilization counter or hardware peak measurement.

## Implementation and instruction audit

- Dynamic mode assigns contiguous16/32/64/128/256-row chunks using an atomic queue. Each projection has its own state, owned by its graph instance. The last worker resets its queue before the backend graph barrier; there is no global shared work queue. The original static path remains available.
- Paired kernels unpack original Q4_K/Q6_K weight blocks and scales once, then update separate primary and residual accumulators. Both activation quantizers and the formula W*Q(x)+W*Q(x-Q(x)) remain unchanged. No model rewrite, new weight quantization or expanded resident weight copy.
- ARM function target attributes enable SDOT only in the new routines. The rest of the worker retains the previous compiler flags; the existing CPU library is reused unchanged. getauxval checks ASIMDDP before selecting either new mode. The Pixel reports ASIMDDP on all eight cores.
- Mode0 uses the existing native dot twice; mode2 calls the new SDOT single-dot function twice; mode1 uses the new paired function. Static/dynamic settings are independently selectable. These controls separate scheduling and decode-sharing effects.
- Actual old Q4/Q6 disassembly contains0 SDOT,14 SMULL and1 SMLAL instructions. The new worker disassembly contains96 SDOT instructions. These are static instruction counts, not executed counts.
- New op counters prove actual Q4/Q6 selection. The six persistent CPU threads retain the qualified affinity maskfc. GPU execution is excluded by the private mode guard.

## Per-thread diagnostic

The separate profile arms time each dot callback and record its row count. Slack is the interval between that thread finishing and the last thread finishing the same projection, not a hardware stall counter. These aggregates include startup and cold calls; they are not a warm8ms timing breakdown. Normal timing arms disable profiling.

| Thread | Static row share | Static work/slack ms | Dynamic32 row share | Dynamic32 work/slack ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16.66% | 471.22 / 845.02 | 37.34% | 1018.81 / 47.42 |
| 1 | 16.66% | 1301.53 / 14.67 | 10.94% | 1046.47 / 19.71 |
| 2 | 16.68% | 1303.40 / 12.60 | 10.96% | 1049.44 / 16.49 |
| 3 | 16.66% | 1300.83 / 15.13 | 10.92% | 1046.87 / 19.10 |
| 4 | 16.66% | 992.93 / 323.20 | 14.93% | 1036.07 / 30.03 |
| 5 | 16.68% | 994.18 / 321.90 | 14.92% | 1036.59 / 29.48 |

Dynamic scheduling demonstrably redistributes work toward the faster thread and reduces early-completion slack. It still fails to speed the native kernel in the warmed comparison. Queue overhead, shared-resource contention and frequency behavior are not separately measured; reduced slack alone is not a latency-win proof.

## Every measured arm

### smoke: 1632 calls, 17 arms

[Raw suite](physical/pixel10pro-packed-cpu-smoke-1/run1/SUITE_RESULT.json). 8 repetitions, 4 warmup. Numerical and exact-output PASS.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-control | 9.825 | 5.055 | 12.302 |
| 01-dynamic16 | 10.172 | 5.191 | 12.077 |
| 02-dynamic32 | 10.172 | 5.196 | 12.081 |
| 03-dynamic64 | 8.281 | 4.253 | 9.524 |
| 04-dynamic128 | 9.203 | 4.704 | 10.968 |
| 05-dynamic256 | 10.439 | 5.350 | 12.457 |
| 06-control | 9.866 | 5.039 | 12.314 |
| 07-sdot-single | 8.045 | 4.117 | 10.372 |
| 08-sdot-pair | 7.752 | 3.955 | 9.346 |
| 09-sdot-single-dynamic32 | 8.205 | 4.188 | 9.798 |
| 10-sdot-pair-dynamic32 | 7.411 | 3.783 | 8.845 |
| 11-sdot-pair-dynamic64 | 7.148 | 3.709 | 8.481 |
| 12-control | 9.576 | 4.880 | 12.026 |
| 13-profile-native | 9.694 | 4.999 | 12.140 |
| 14-profile-dynamic32 | 10.213 | 5.190 | 12.046 |
| 15-profile-pair-dynamic32 | 7.418 | 3.797 | 8.739 |
| 16-control | 9.698 | 4.990 | 12.086 |

### confirm: 4320 calls, 18 arms

[Raw suite](physical/pixel10pro-packed-cpu-confirm-1/run1/SUITE_RESULT.json). 20 repetitions, 10 warmup. Numerical and exact-output PASS.

| Arm | Full mean ms | Half mean ms | Full p99 ms |
| --- | ---: | ---: | ---: |
| 00-control | 8.086 | 4.121 | 8.785 |
| 01-dynamic64 | 8.187 | 4.154 | 8.887 |
| 02-sdot-single | 6.667 | 3.391 | 7.576 |
| 03-sdot-pair | 6.577 | 3.333 | 8.362 |
| 04-sdot-single-dynamic64 | 6.468 | 3.316 | 7.654 |
| 05-sdot-pair-dynamic64 | 6.317 | 3.242 | 7.446 |
| 06-control | 8.099 | 4.120 | 8.735 |
| 07-pair-dynamic16 | 6.456 | 3.238 | 8.236 |
| 08-pair-dynamic128 | 6.375 | 3.219 | 8.332 |
| 09-pair-dynamic256 | 6.294 | 3.237 | 7.642 |
| 10-pair-dynamic32 | 6.298 | 3.191 | 7.566 |
| 11-control | 8.092 | 4.111 | 8.889 |
| 12-sdot-pair-dynamic64 | 6.334 | 3.236 | 7.381 |
| 13-sdot-single-dynamic64 | 6.495 | 3.393 | 8.031 |
| 14-sdot-pair | 6.554 | 3.312 | 8.415 |
| 15-sdot-single | 6.672 | 3.383 | 7.412 |
| 16-dynamic64 | 8.185 | 4.158 | 8.819 |
| 17-control | 8.100 | 4.124 | 8.751 |

Chunks16/32/128/256 with paired SDOT were also measured in the longer suite. Some single-arm means are slightly below64, but64 has two confirming repeats and good full p99. No global optimum is claimed. The first smoke made native dynamic64 look faster against less-warmed controls; the longer repeat supersedes that speed interpretation and preserves its raw results.

### Independent-row batch qualification

[Raw batch suite](physical/pixel10pro-packed-cpu-batch-1/run1/SUITE_RESULT.json):1056 calls,3696 input rows. Builds eight distinct rotated/scaled inputs for each layer and derives96 independent F16 CPU reference outputs. Original row0 is cross-checked against archived desktop CPU references. Batches1/2/4/8 are tested on native, dynamic-native, paired-static, paired-dynamic64 and single-SDOT-dynamic64. All960 packed outputs, including768 optimized outputs, are byte-identical to native. Maximum per-row relativeL2=0.000562194. These short arms qualify correctness, not batch throughput selection.

| Arm | B1 full ms | B2 full ms | B4 full ms | B8 full ms |
| --- | ---: | ---: | ---: | ---: |
| 01-control | 13.824 | 15.839 | 31.284 | 62.763 |
| 02-dynamic64 | 11.372 | 15.149 | 29.612 | 61.037 |
| 03-sdot-pair | 9.163 | 10.748 | 16.278 | 36.875 |
| 04-sdot-pair-dynamic64 | 8.356 | 10.742 | 15.964 | 32.771 |
| 05-sdot-single-dynamic64 | 9.505 | 11.984 | 19.382 | 40.235 |

## Validation, candidate and limits

PASS: 7008 calls / 9648 input rows / 41 arms. All5376 optimized outputs are byte-identical to native; all6912 packed outputs meet the independent-reference tolerance. The remaining96 calls generate independent F16 references. Native control240/240 also matches preserved worker-v5. Maximum relativeL2=0.000562194 against the existing0.01 threshold. Repetitions reuse qualification inputs, not thousands of independent prompts.

Actual tuned projection dispatch PASS: Q4_K=44992, Q6_K=5624, including startup. Optional per-thread row and call totals also match the expected graph coverage.

Build PASS on the first attempt with inherited -Wall -Wextra -Werror flags. Three changed Python files pass pyflakes/parse; source ASCII/whitespace and all build artifact hashes pass. Header and builder snapshots match the tested sources. [Build checks](PIXEL_PACKED_CPU_BUILD_CHECKS.json), [build provenance](software/pixel10pro-packed-cpu-v1/BUILD_PROVENANCE.json), [instruction audit](PIXEL_PACKED_CPU_INSTRUCTIONS.json).

[Selected private candidate](PIXEL_PACKED_CPU_CANDIDATE.json): six persistent threads, maskfc, original packed weights, fused residual correction, S42_PIXEL_CPU_PAIR_DOT=1 and S42_PIXEL_CPU_ROW_CHUNK=64. Worker SHA25664133753f47160c418b1c251cd6cb056b3bbe17fbd76490e229b81433d7bdcf5. Existing CPU runtime SHA256b911532d756cad93e74391e86ed4d0e8e6f66773ec0dab79f8aa893021b0589d. The previous candidate remains archived.

Cleanup PASS05:07UTC: all finite workers exit normally, same boot, no remaining FFN worker or Pixel ADB forward, Pixel lock reacquired nonblocking and released, no queued job. Only ADB5037 was used. No root, reboot, clock/power setting, desktop model, other-phone or production change. [Cleanup receipt](PIXEL_PACKED_CPU_CLEANUP.json).

Not verified: current USB/server latency, energy, full-model token identity, physical DRAM traffic or compute utilization, sustained thermal behavior, batch performance selection, and CPU/GPU split tuning with this faster CPU. The new CPU knobs currently require CPU-only operation. No commit, push or PR.

Sources: [paired kernels and queue](pixel_packed_cpu.h), [private builder](build_pixel_packed_cpu.py), [worker/header patch](software/pixel10pro-packed-cpu-v1/PACKED_CPU.patch). [Aggregate measurements](PIXEL_PACKED_CPU_RESULTS.json) and [pooled repeated comparison](PIXEL_PACKED_CPU_CONFIRM_COMPARISON.json).
