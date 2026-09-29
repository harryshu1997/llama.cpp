# BurstGPT dynamic FFN operator offload to OP15

Date: 2026-08-03 EDT.

Verdict: `FULL_TRACE_PASS; OP15_SPEEDUP; THROUGHPUT_PLUS_66.08_PERCENT; MAKESPAN_MINUS_39.79_PERCENT; COLD_SERVICE_MINUS_42.04_PERCENT; STOCK_RESTORED`.

This is a single paired physical run, not a multi-run statistical result.

## Timeline

![Full CPU versus CPU plus OP15 request timeline](BURSTGPT_DYNAMIC_FFN_TIMELINE_V2.png)

The top plot shows cumulative completed output tokens for one Gemma model
under two execution modes. Blue is full desktop CPU and orange is CPU plus
OP15. The curves step at exact request completion. CPU plus OP15 completes all
505 tokens 130.4 seconds earlier; full CPU has completed only 249 tokens at
that time. The middle strips show measured request execution spans, while the
bottom lane shows shared arrivals and both sets of completions. The editable SVG is
`BURSTGPT_DYNAMIC_FFN_TIMELINE_V2.svg`; its generator is
`burstgpt_gpu_cpu_op15_v1/plot_timeline.py`.

## Result

The RTX 4060 Ti served the hot Qwen3-14B Q4_K_M route while one persistent
Gemma4-12B Q4_0 desktop process served the cold route. In the treatment, that
same cold process split FFN operators between the i9-12900K and the OP15 HTP.
There was no second cold-model process and no layer handoff.

Both control and treatment used `--no-repack`. This makes the operator split
comparison layout-equivalent, but it does not establish a win over the fastest
repacked CPU-only configuration.

| metric | CPU control | CPU plus OP15 | change |
| --- | ---: | ---: | ---: |
| trace makespan | 327.705 s | 197.321 s | -39.79% (1.661x faster) |
| output throughput | 6.637 token/s | 11.023 token/s | +66.08% |
| cold prefill p50 | 16.565 s | 8.876 s | -46.42% (1.866x faster) |
| cold decode p50 | 7.675 s | 5.110 s | -33.43% (1.502x faster) |
| cold route wall p50 | 24.240 s | 14.044 s | -42.07% |
| cold service p50 | 24.253 s | 14.056 s | -42.04% |
| cold queue p50 | 149.296 s | 81.370 s | -45.50% |
| cold completion p50 | 162.906 s | 88.648 s | -45.58% |
| hot completion p50 | 2.825 s | 2.815 s | -0.35% |
| all-request SLO | 57/74 | 59/74 | +2 requests |

All 74 requests completed in both modes. Neither hot nor cold process swapped.
The treatment cold-process peak RSS was 7,539,953,664 bytes and minimum system
available memory was 28,162,433,024 bytes.

## Workload

The semantic trace keeps the original BurstGPT arrival schedule and contains
57 hot and 17 cold requests. Prompt and output lengths are capped at 512 and
32 tokens, respectively, instead of the old 128-prompt and 8-output test.
The trace requests 2,175 output tokens. The final scheduled arrival is at 57.7
seconds.

The cold prompt shapes are nine 512-token requests plus 23, 47, 53, 76, 245,
271, 303, and 309-token shapes. Fifteen cold requests generate 32 tokens; the
other two generate 11 and 14. Prompt tokens were produced with the tokenizer
and chat template of the model that actually served each route.

Trace SHA-256:

    ccde6e3e53dee4547e4eb80f9f090032afb04b1d3fec3fd07f0f961bed60cf8f

## Placement and dynamic split

The phone stores the largest selected suffix once for every one of Gemma's 48
dense FFNs:

    full FFN width:       15360 columns
    resident OP15 suffix: [5696, 15360), 9664 columns
    resident weights:     gate, up, and down, Q4_0, 2866.64 MiB
    maximum token rows:   512

The resident suffix is partitioned into two blocks of 1,472 and 8,192 columns.
At runtime the graph selects the final 8,192-column block or both blocks; it
does not reload or duplicate phone weights.

The calibrated policy used by the trace is:

    prefill M <= 128:  phone 8192 columns, desktop 7168 columns
    prefill M > 128:   phone 9664 columns, desktop 5696 columns
    decode M = 1:      phone 9664 columns, desktop 5696 columns

For each layer, the normalized K x M input with K=3840 is published once. The
phone and desktop then execute concurrently:

    gate = W_gate * input
    up = W_up * input
    hidden = GeGLU(gate, up)
    partial = W_down * hidden

The desktop computes the FFN prefix while the phone computes the suffix. The
two K x M partial results are summed before the residual path continues. All
attention, normalization, residual, vocabulary-head, and sampling work stays
on the desktop.

The width sweep selected this policy from physical measurements:

| prefill rows M | CPU only | best phone width | split time | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 53 | 1.822 s | 8192 | 1.119 s | 1.63x |
| 76 | 2.454 s | 9664 | 1.651 s | 1.49x |
| 303 | 9.858 s | 9664 | 5.685 s | 1.73x |
| 512 | 16.383 s | 9664 | 9.255 s | 1.77x |

The 8,192 and 9,664 points at M=76 were effectively tied, so the policy uses
8,192 through M=128 to retain margin for short shapes.

## Latency breakdown

The treatment completed 25,776 verified FFN calls: 24,912 decode calls and
864 multi-token prefill calls. It transferred 2,567,946,240 bytes in each
direction. The largest activation message was 3,932,288 bytes.

| split-operator component | decode p50 | prefill p50 |
| --- | ---: | ---: |
| phone HTP compute | 1.245 ms | 10.114 ms |
| complete phone RPC | 1.528 ms | 62.605 ms |
| overlapped FFN | 1.568 ms | 90.365 ms |
| direct-DMA USB interval | 1.491 ms | 60.832 ms |

The USB interval includes waiting for HTP output and must not be added to HTP
compute. For decode, the desktop branch is 1.480 ms versus a 1.528 ms phone
RPC, and the median join wait is only 0.073 ms. Decode is close to the desired
branch balance.

For a 512-token prefill layer, phone RPC is 64.690 ms while the overlapped FFN
is 95.293 ms. About 30.6 ms of the phone path is hidden behind desktop work.
The desktop FFN prefix, plus the desktop-only attention and graph work, is now
the prefill critical path. More USB tuning alone cannot remove that time.

At full-trace level, serialized cold-route queueing remains the largest
latency term. Lowering cold service p50 by 42.04% lowers queue p50 by 45.50%
because every completed cold request advances the following requests.

## Correctness and receipts

- Both runs returned all requested tokens and passed model, trace, process,
  protocol, payload-hash, and no-swap checks.
- The bridge ended with `status=ok`; the phone session ended with
  `worker_status=0`.
- Eight of 17 cold token sequences were exactly equal. The median common
  prefix was 31 of 32 tokens. The divergence is expected from FP16 activation
  exchange. All treatment outputs decoded to nonempty valid UTF-8 and the
  inspected outputs were coherent English.
- The hot route remained effectively unchanged in this paired run.
- The phone was rebooted to stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, normal `ptp,adb` mode,
  USB device `22d9:2772` at 5 Gb/s, with no worker left running.

Raw result identities on the physical desktop:

    CPU RESULT.json:
      1b036034936597037f3d7f3db0a28557122d3b2791e2d2d73414faec516a2c57
    OP15 RESULT.json:
      d1c43fd66efe43e518b7b8beedee917d74f79395e0ec09cc95a71754982df4ba
    OP15 phone-worker.log:
      8a47962491330b9aae07ccbc4aaaa90641b92beeb13d5300d8b61be1ba8bb936
    prefill sweep RESULT.json:
      862c3f0cee8aa4fd3773137835bb9993d81c109795059520ea6b3caa4d6714db

Raw directories:

    /home/zhihao/s41-dynamic-ffn-v1/campaign/raw/cpu-semantic-r1/
    /home/zhihao/s41-dynamic-ffn-v1/campaign/raw/op15-semantic-r2-coarse/
    /home/zhihao/s41-dynamic-ffn-v1/campaign/raw/prefill-sweep-v1/

## Proposed DVFS energy follow-up

The current decode split is already balanced: the 5,696-column desktop branch
takes 1.480 ms and the 9,664-column phone RPC takes 1.528 ms. Lowering desktop
frequency without widening the phone suffix would therefore move the desktop
branch onto the critical path.

Use this as the first bounded calibration matrix. The widths are estimates
from the measured F16 width slope and must be replaced by physical results:

| P-core maximum | initial phone width | desktop width | neighboring widths |
| ---: | ---: | ---: | --- |
| stock 5.1-5.2 GHz | 9664 | 5696 | current control |
| 4.6 GHz | 10048 | 5312 | 9792, 10240 |
| 4.1 GHz | 10560 | 4800 | 10240, 10752 |
| 3.6 GHz | 11072 | 4288 | 10752, 11136 |

Pin the eight cold inference threads to one logical thread on each P-core and
cap only those core policies. Keep the hot server affinity and every other
trace setting identical. Use an 8,192-column suffix for prefill shapes up to
128 tokens and the calibrated maximum for larger prefill and decode. The
11,136-column point is the measured phone-memory ceiling and requires a fresh
allocation gate before every full trace.

Run three alternating repetitions per surviving point and accept a profile
only when all of these hold:

- makespan at most 207.19 seconds, no more than 5% above the current treatment;
- throughput at least 10.50 output token/s;
- cold service p50 at most 14.76 seconds;
- all protocol, output-quality, no-swap, and kernel-fault gates pass;
- measured desktop package joules per completed token improve materially.

The i9 exposes `power/energy-pkg/` and `power/energy-cores/` through the perf
PMU, but reading them and changing `intel_pstate` limits requires administrator
access on the desktop. These counters establish desktop package energy only.
Phone energy must be synchronized separately; battery readings while USB is
charging are not a defensible total-system energy measurement.

## Remaining limitation

This result compares the same generic Q4_0 CPU layout in both arms. The phone
split currently relies on FFN submatrix views that are incompatible with the
existing shape-dependent CPU repack buffer. Therefore this run proves a large
gain over no-repack CPU execution, but not over the best repacked desktop-only
execution. A future implementation should make independently packed host FFN
blocks so the CPU prefix can retain its optimized kernel while the suffix is
offloaded.
