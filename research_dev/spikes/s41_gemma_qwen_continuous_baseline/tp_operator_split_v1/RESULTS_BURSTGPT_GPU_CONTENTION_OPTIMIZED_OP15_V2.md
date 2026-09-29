# Optimized CPU plus OP15 inference under a busy desktop GPU

Date: 2026-08-04 EDT.

Verdict: `PASS; THROUGHPUT_PRIORITY_PRESERVES_QWEN_AND_IMPROVES_GEMMA_11.61_PERCENT; COLD_PRIORITY_IMPROVES_GEMMA_24.55_PERCENT; DYNAMIC_OPERATOR_SPLIT; DEFAULT_FREQUENCY_PRESERVED; STOCK_PHONE_RESTORED`.

## Comparison graph

![Busy-GPU baseline and optimized operating points](BURSTGPT_GPU_CONTENTION_OPTIMIZED_V2.png)

The upper-left panel compares cold makespan directly. The lower-left panel
separates prefill and decode. The right panel shows the Qwen versus Gemma
throughput tradeoff, where the top-right direction is better. The editable
SVG is `BURSTGPT_GPU_CONTENTION_OPTIMIZED_V2.svg`; its generator is
`burstgpt_gpu_cpu_op15_v1/plot_gpu_contention_optimized.py`.

## Result

The original saturated-GPU configuration slowed the Gemma cold trace from
187.085 seconds to 240.974 seconds. Two optimized policies recover different
parts of that loss:

| metric | GPU idle reference | busy baseline | keep Qwen speed | minimum cold latency |
| --- | ---: | ---: | ---: | ---: |
| Gemma makespan | 187.085 s | 240.974 s | 215.905 s | 193.468 s |
| Gemma output throughput | 2.699 token/s | 2.096 token/s | 2.339 token/s | 2.610 token/s |
| Qwen output throughput | inactive | 50.238 token/s | 50.230 token/s | 48.269 token/s |
| prefill p50 | 8.500 s | 12.931 s | 10.827 s | 9.166 s |
| decode p50 | 5.121 s | 5.341 s | 5.344 s | 5.197 s |
| service p50 | 13.681 s | 18.345 s | 16.138 s | 14.398 s |
| queue p50 | 72.050 s | 101.129 s | 87.707 s | 76.364 s |
| completion p50 | 79.716 s | 111.363 s | 96.555 s | 84.141 s |
| cold SLO met | 3/17 | 2/17 | 2/17 | 3/17 |

All values are medians of three physical runs. The full ranges for the main
outcomes were:

| policy | Gemma makespan range | Gemma throughput range | Qwen throughput range |
| --- | ---: | ---: | ---: |
| keep Qwen speed | 215.190-216.061 s | 2.337-2.347 token/s | 50.221-50.342 token/s |
| minimum cold latency | 193.233-193.727 s | 2.607-2.613 token/s | 47.916-48.416 token/s |

Relative to the busy baseline, the keep-Qwen policy reduces cold makespan by
10.40% and raises cold throughput by 11.61%, while Qwen changes by -0.015%.
The cold-priority policy reduces cold makespan by 19.71% and raises cold
throughput by 24.55%, while Qwen changes by -3.92%.

The sum of active output token rates changes from 52.333 token/s in the busy
baseline to 52.569 token/s in the keep-Qwen policy and 50.876 token/s in the
cold-priority policy. This sum is only a scheduling count: Qwen and Gemma
tokens do not represent equal work or equal user value.

The idle result is a historical reference using the earlier 9,664-column
worker and default scheduler. Optimization deltas above use the busy baseline,
not the idle reference, because that is the controlled predecessor.

## What changed

### Dynamic operator split

The phone retains the largest Q4_0 suffix for every dense FFN and selects one
of three widths for each request without reloading weights:

    short prefill, M <= 128: phone 8192 columns, desktop 7168 columns
    long prefill,  M > 128:  phone 11136 columns, desktop 4224 columns
    decode,        M = 1:    phone 9664 columns, desktop 5696 columns

The FFN width is 15,360 columns. The phone therefore executes 53.3% for short
prefill, 72.5% for long prefill, and 62.9% for decode. This remains an
operator-level suffix split in every one of the 48 layers. It is not a layer
handoff. The desktop prefix and phone suffix execute concurrently, and their
partial outputs are added on the desktop.

Protocol v4 advertises `alternate_columns=9664`. With an 8,192-column quantum,
the resident 11,136-column suffix is stored as only three blocks:

    1472 + 1472 + 8192 = 11136 columns

This supports 8,192, 9,664, and 11,136 columns without creating many small
blocks. The phone holds 3,303.29 MiB of Q4_0 FFN weights. Desktop and phone use
the same Gemma4-12B Q4_0 model weights; the worker reported weight hash
`4ada42f7ae1d721e` in every run.

### HTP buffer layout

The first flexible worker placed all three resident weight blocks in one HTP
buffer. Although a 9,664-column decode used only a suffix of those weights, its
graph was backed by the full 3,303 MiB allocation. The measured decode overlap
increased to about 2.03 ms and full decode p50 to 5.83 seconds, making the
monolithic allocation the leading suspected source of the regression.

The final worker allocates one resident HTP buffer per block group. A decode
graph references only the 8,192 and one 1,472-column buffers, while long
prefill references all three. No weight is copied per request. This restored
the cold-priority decode balance:

| decode component | busy baseline | keep Qwen speed | minimum cold latency |
| --- | ---: | ---: | ---: |
| desktop branch | 1.515 ms | 1.551 ms | 1.518 ms |
| phone RPC | 1.548 ms | 1.531 ms | 1.525 ms |
| HTP compute | 1.254 ms | 1.235 ms | 1.233 ms |
| direct-DMA USB interval | 1.509 ms | 1.492 ms | 1.487 ms |
| complete overlapped FFN | 1.594 ms | 1.568 ms | 1.554 ms |

The minimum-latency decode is nearly balanced: desktop 1.518 ms versus phone
RPC 1.525 ms. Moving more decode columns in either direction is not justified
by the measured medians.

For long prefill, the larger suffix intentionally increases phone work while
reducing the critical desktop prefix:

| prefill component | busy baseline, 9664 cols | keep Qwen speed, 11136 cols | minimum cold latency, 11136 cols |
| --- | ---: | ---: | ---: |
| complete overlapped FFN | 134.462 ms | 97.857 ms | 87.318 ms |
| phone RPC | 63.332 ms | 65.869 ms | 65.878 ms |
| HTP compute | 10.163 ms | 13.056 ms | 13.085 ms |
| direct-DMA USB interval | 61.539 ms | 63.656 ms | 63.681 ms |

The extra phone compute and transfer are useful because they remove more work
from the desktop branch. Every optimized trace transferred 2,567,946,240 bytes
in each direction across 25,776 DMA-BUF calls.

### CPU placement policies

Both policies pin the eight Gemma threads to one primary logical CPU on each
P-core:

    Gemma:             0,2,4,6,8,10,12,14
    bridge/controller: 16-23 (E-cores)

The keep-Qwen policy lets the 31 Qwen server threads use logical CPUs 0-23.
It preserves GPU feeding and shares P-core primaries when needed. The
cold-priority policy limits Qwen to the unused P-core siblings plus E-cores:

    Qwen cold-priority: 1,3,5,7,9,11,13,15,16-23

The harness recorded and gated every process thread's allowed CPU set at
readiness and paid start. A diagnostic run allowed Qwen on all CPUs at nice
value +5. All 31 server threads inherited the requested value, but Gemma was
216.321 seconds and Qwen was 50.138 token/s. It did not improve the shared-CPU
point and is rejected.

## Optimization path

Single-run candidates were used only to choose the final policies:

| candidate | Gemma makespan | prefill p50 | decode p50 |
| --- | ---: | ---: | ---: |
| pinned, max/decode 9664 | 215.541 s | 11.513 s | 5.139 s |
| pinned, max/decode 10240 | 209.016 s | 10.994 s | 5.182 s |
| pinned, max/decode 11136 | 208.608 s | 9.538 s | 6.085 s |
| flexible widths, one HTP weight buffer | 204.924 s | 9.566 s | 5.833 s |
| flexible widths, separate HTP block buffers | 193.453 s | 9.525 s | 5.152 s |

The final rows in the main result table use three fresh repetitions, not these
selection runs.

## Correctness and safety

- All 12 main baseline and optimized runs completed 17/17 cold requests and
  their expected output-token counts. The six optimized runs each returned
  505 cold tokens.
- All six optimized runs returned the same 17 token sequences. Their canonical
  token-array SHA-256 is
  `9966a8fab2bd42e73669765d8986c11e231b0de7f1b55ac14d462e28a007286d`.
- The optimized tokens differ from the earlier split-width baseline, as
  expected from quantized partial-sum boundaries. Detokenizing all 17 sequences
  produced grammatical continuations. Sixteen directly discussed distributed
  neural inference; the final short response asked for the missing context of
  `Request 73`. Short outputs end mid-sentence because the trace preserves each
  request's fixed output-token budget.
- Every optimized run completed 25,776 calls with bridge `status=ok`,
  `FFN_OVERLAP_OK`, and phone worker status 0.
- Neither model process swapped. GPU utilization p95 was 100% in every busy
  run. No CPU or GPU frequency was locked or capped.
- The final phone scan found no DMA-BUF, SMMU/IOMMU, panic, BUG, or call-trace
  match.
- OP15 was normally rebooted to stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, USB `ptp,adb`, device
  `22d9:2772` at 5 Gb/s, with no transport worker. The desktop GPU was restored
  to 0% utilization, and Intel policy remained max 100, min 16, turbo enabled,
  governor `powersave`.

## Recommendation

Use the keep-Qwen policy when the hot service has a throughput SLO. It is the
default balanced policy because it preserves the measured hot rate and still
improves the cold route by 11.61%.

Use the cold-priority policy when cold completion time matters more than about
4% of hot throughput. It recovers 24.55% cold throughput, restores 3/17 cold
SLOs, and comes within 3.4% of the historical GPU-idle cold makespan even while
Qwen remains active.

Deployment can select either CPU-affinity policy without changing phone
weights or FFN graphs. The tensor-width policy remains the same in both.

## Evidence

Analysis receipt:

    94f77eeaa650ad9120f41ab9134708e992d39cc63475045ecafc083a1df8b666  burstgpt_gpu_cpu_op15_v1/GPU_CONTENTION_OPTIMIZATION_ANALYSIS_V2.json

Baseline analysis bound by the optimized receipt:

    800eef0d497f4dc5018ba08c74e0ff58f518cf3e8225b620e2ac7c6bfeea4a59  GPU_CONTENTION_ANALYSIS.json

Optimized result receipts, keep-Qwen runs 1 through 3:

    dfbfc6c08a63f57b607e4311fbef303b739b69a9fbccc965103e73ec246114ee
    384b61ad09bfb5fc9ba8fbe8e020c33d89c0aee11292193c8eee3ebb25c2f0e6
    36542fae8e9850f8358765ae253fa2b1a1b2f76acf04f7f711b8cc5b03f3f4e3

Optimized result receipts, cold-priority runs 1 through 3:

    e57f72ec5bf93974f5728b7942ea2a49e232272f5cc50f58fd2ff6f12489dc45
    10da0d05f2c56fd67e4265124b7ad15fa3c7bdc11efaf70f0858c89affbbb8b6
    9d59bf15ecbc89d9b1c5374139b9f7b25c212cbfeca1686bd0fcbd4cc725676b

Graph receipts:

    42eb9cc7ba50db20133bee653fdfb98f6df75fe59dbfcf5b94f96c72165ab80d  BURSTGPT_GPU_CONTENTION_OPTIMIZED_V2.svg
    63db415a8f849ee5827012767b2c6c119665f8a89339abe7df1aabc30f4ba0e3  BURSTGPT_GPU_CONTENTION_OPTIMIZED_V2.png

Reproducible analysis and plotting tools:

    burstgpt_gpu_cpu_op15_v1/analyze_gpu_contention_optimized.py
    burstgpt_gpu_cpu_op15_v1/plot_gpu_contention_optimized.py
    burstgpt_gpu_cpu_op15_v1/run_gpu_contention.py
    burstgpt_gpu_cpu_op15_v1/run_trace.py

Raw physical results remain under:

    /home/zhihao/s41-dynamic-ffn-v1/campaign/raw/

## Limitations

This is one desktop and one phone with three repetitions per final policy.
The Qwen treatment continuously refills eight server slots, so it is a busy
ceiling rather than the original sparse hot arrival trace. The experiment did
not measure wall-plug energy or memory-controller counters. The idle reference
predates the optimized worker and affinity policy. The result establishes two
measured operating points, not a universal CPU-affinity rule for other desktop
topologies.
