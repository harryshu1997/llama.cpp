# GPU contention against CPU plus OP15 cold inference

Date: 2026-08-04 EDT.

Verdict: `PASS; SATURATED_GPU_SLOWS_COLD_MAKESPAN_28.80_PERCENT; COLD_THROUGHPUT_MINUS_22.36_PERCENT; PREFILL_IS_THE_BOTTLENECK; PHONE_PATH_STABLE; DEFAULT_FREQUENCY_PRESERVED; STOCK_RESTORED`.

## Comparison graph

![Resident-idle versus saturated GPU contention](BURSTGPT_GPU_CONTENTION_COMPARISON_V1.png)

The left panel compares cold-route latency directly. The right panel
normalizes operator time to the resident-idle control and shows that the
prefill FFN overlap grows by 48.9%, while phone RPC, HTP compute, and USB each
change by less than 1%. The editable SVG is
`BURSTGPT_GPU_CONTENTION_COMPARISON_V1.svg`; its generator is
`burstgpt_gpu_cpu_op15_v1/plot_gpu_contention.py`.

## Result

A fully GPU-offloaded Qwen3-14B Q4_K_M server remained loaded in the RTX
4060 Ti in both arms. In the control it completed one warmup request and then
remained resident but idle. In the treatment its eight continuous-batching
slots were refilled back-to-back for the full paid interval. The treatment
therefore measures one saturated hot model, not model loading or VRAM
residency.

The cold route was identical in both arms: one persistent Gemma4-12B Q4_0
process split every dense FFN between eight desktop CPU threads and the OP15
HTP. The 17 cold requests retained their original BurstGPT arrival times and
generated 505 output tokens. Values are medians of three physical runs;
brackets show the run range.

| cold-route metric | GPU resident-idle | GPU saturated | change |
| --- | ---: | ---: | ---: |
| makespan | 187.085 s [185.006, 187.528] | 240.974 s [240.755, 241.247] | +28.80% |
| output throughput | 2.699 token/s [2.693, 2.730] | 2.096 token/s [2.093, 2.098] | -22.36% |
| prefill p50 | 8.500 s [8.398, 8.660] | 12.931 s [12.905, 12.990] | +52.13% |
| decode p50 | 5.121 s [5.039, 5.135] | 5.341 s [5.329, 5.374] | +4.29% |
| service p50 | 13.681 s [13.558, 13.808] | 18.345 s [18.317, 18.444] | +34.09% |
| queue p50 | 72.050 s [71.505, 73.506] | 101.129 s [101.045, 101.480] | +40.36% |
| completion p50 | 79.716 s [78.903, 81.253] | 111.363 s [111.289, 111.668] | +39.70% |
| cold SLO met | 3/17 [3, 3] | 2/17 [2, 2] | -1 request |

Cold output throughput uses only the 505 cold tokens and the interval from
paid start to the last cold completion. It includes the frozen arrival
schedule, whose last arrival is at 57.7 seconds. It is not the combined hot
plus cold system throughput reported by the earlier mixed trace.

The saturated Qwen route produced a median 50.238 output token/s [49.885,
50.446]. Its sampled GPU utilization was 83% at p50 and 100% at p95; the
resident-idle control was 0% at both percentiles. GPU power sampled through
`nvidia-smi` rose from 9.36 W to 107.62 W at p50. These are GPU readings, not
desktop or total-system energy measurements.

## Bottleneck

The phone path did not cause the 28.80% makespan loss. Median-of-run medians:

| split-operator component | resident-idle | saturated | change |
| --- | ---: | ---: | ---: |
| prefill overlapped FFN | 90.310 ms | 134.462 ms | +48.89% |
| prefill phone RPC | 63.053 ms | 63.332 ms | +0.44% |
| prefill HTP compute | 10.106 ms | 10.163 ms | +0.56% |
| prefill direct-DMA USB interval | 61.252 ms | 61.539 ms | +0.47% |
| decode overlapped FFN | 1.568 ms | 1.594 ms | +1.64% |
| decode desktop branch | 1.478 ms | 1.515 ms | +2.47% |
| decode phone RPC | 1.529 ms | 1.548 ms | +1.27% |
| decode HTP compute | 1.244 ms | 1.254 ms | +0.80% |
| decode direct-DMA USB interval | 1.489 ms | 1.509 ms | +1.37% |

For prefill, the desktop branch already exceeded the phone RPC in the idle
profile. Saturating the GPU leaves phone compute, USB, and the complete phone
RPC almost unchanged while increasing overlapped FFN time by 48.89%. The
desktop prefix is therefore the critical branch. Desktop-only attention,
normalization, graph work, and the vocabulary path account for the remaining
full-prefill increase.

Decode remains close to its prior balance: the desktop branch is 1.515 ms and
the phone RPC is 1.548 ms under load. This is why complete decode changes only
4.29%, while multi-token prefill changes 52.13%.

Live scheduler spot checks found the unpinned cold workers primarily on P-core
logical CPUs while Qwen was idle, but also on E-core logical CPUs while Qwen
was active. The hot server had 31 host threads and consumed about one CPU core
while saturated. This supports scheduler interference as one contributor.
The experiment did not separately apportion scheduler placement, cache
pressure, and DRAM traffic, so the broader supported conclusion is
desktop-side contention.

## Configuration

The phone stored the same 9,664-column suffix for all 48 dense FFNs and used
the same runtime policy in every run:

    prefill M <= 128: phone 8192 columns, desktop 7168 columns
    prefill M > 128:  phone 9664 columns, desktop 5696 columns
    decode M = 1:     phone 9664 columns, desktop 5696 columns

Both arms used `--no-repack`. No CPU affinity was applied because this test
was intended to measure the default desktop scheduler. Every run recorded the
same Intel policy before and after the paid interval:

    intel_pstate max_perf_pct: 100
    intel_pstate min_perf_pct: 16
    turbo disabled:            0
    governor:                  powersave
    policy0 scaling maximum:   5100000 kHz

No CPU or GPU clock was set, capped, or locked. The NVIDIA driver selected its
normal dynamic P-states.

## Correctness and safety

- All six runs completed 17/17 cold requests and 505/505 cold output tokens.
- All 17 token sequences were identical across all six runs.
- Every run completed 25,776 verified phone calls, with bridge `status=ok`,
  `FFN_OVERLAP_OK`, and phone `status=0`.
- Neither model process swapped. Minimum available system memory was at least
  25.70 GiB in the saturated runs.
- The temporary-kernel scan contained no DMA-BUF, SMMU, IOMMU, panic, BUG, or
  call-trace match.
- A normal reboot restored OP15 kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, USB `ptp,adb`, device
  `22d9:2772` at 5 Gb/s, with no userspace transport worker.

One preliminary phone worker launch used the display family name instead of
the registered backend device name and was rejected before descriptors were
published. It restored USB and is excluded. Its receipt is preserved as
`phone-worker.failed-backend.log` beside resident-idle run 1.

## Recommendation

Do not use the idle split calibration unchanged when the GPU server is
saturated. Two bounded follow-ups are justified:

1. Pin the eight cold threads to one logical thread on each P-core and place
   hot-server control work on E-cores or sibling logical CPUs. Repeat this
   campaign at default frequency to measure how much scheduler isolation
   recovers.
2. Re-run the prefill width solver under hot load. The phone RPC has about
   71 ms of slack against the 134 ms overlapped prefill median, so a larger
   phone suffix should help multi-token prefill. Decode is already balanced
   and should remain near 9,664 phone columns.

The second step may require the previously measured 11,136-column phone
resident maximum and its fresh memory-allocation gate. The planner should use
separate idle and GPU-busy desktop profiles rather than a single static CPU
throughput estimate.

## Evidence

Reproducible harnesses:

    burstgpt_gpu_cpu_op15_v1/run_gpu_contention.py
    burstgpt_gpu_cpu_op15_v1/analyze_gpu_contention.py

Raw result root on the physical desktop:

    /home/zhihao/s41-dynamic-ffn-v1/campaign/raw/

Analysis receipt:

    800eef0d497f4dc5018ba08c74e0ff58f518cf3e8225b620e2ac7c6bfeea4a59  GPU_CONTENTION_ANALYSIS.json

Result receipts, resident-idle runs 1 through 3:

    7f639ef6836aeb53c2ab2460d56be5b1b4dcbd9cfae4bbeb3008af28d52ec673
    656c58d87a22fc442167f67d18b6fd9fbbc115a4307347f67622714f8eb8f88a
    eca6acb619bf5297d7bfb13a70a8eb694af21e1956a68ba3dc7fff3f290c8287

Result receipts, saturated runs 1 through 3:

    9bfed72f48c831d73dabd09786943a1982cffdebbb0a923331442ef46e4f7a9d
    765b82ae30ecf6c2c9f6a965d1467393e24d5f80b7ce3f8e3785b619a9ed913e
    c7b0df42a2576045c97253018b5a10a008cfeddc9a0359412eb6ad0cd38a5895

The empty kernel-fault scan has SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Limitations

This is one desktop and one phone with three repetitions per arm. Saturating
all eight Qwen slots is a contention ceiling, not the original sparse
BurstGPT hot arrival pattern. CPU affinity was intentionally left at its
default, CPU memory-controller counters were not available, and the result
does not isolate scheduler interference from cache or DRAM contention. The
cold comparison also retains the existing `--no-repack` limitation.
