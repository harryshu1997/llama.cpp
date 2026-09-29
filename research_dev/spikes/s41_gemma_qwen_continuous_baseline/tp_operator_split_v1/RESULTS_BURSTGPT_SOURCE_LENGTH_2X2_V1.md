# BurstGPT source-length CPU and OP15 2x2

## Result

CPU + OP15 improved the complete cold-model run by 17.1% in makespan and
20.6% in cold output-token throughput relative to full CPU. The gain was much
larger in prefill than decode:

| Comparison, hot GPU absent | Full CPU | CPU + OP15 | Change |
| --- | ---: | ---: | ---: |
| Cold service sum | 2128.416 s | 1764.512 s | -17.1% |
| Prefill sum | 370.632 s | 216.023 s | -41.7% |
| Decode sum | 1757.775 s | 1548.479 s | -11.9% |
| Paid makespan | 2131.200 s | 1767.293 s | -17.1% |
| Cold output rate | 3.247 tok/s | 3.915 tok/s | +20.6% |

The median paired per-request speedup was 1.246x. The range was 1.084x to
1.612x over the 17 cold requests.

## Workload

The input is the untruncated BurstGPT source-length slice, with the original
arrival order compressed into a 55.75 s arrival span. The frozen input SHA-256
is `b20a9ba66ee3558d835a0e19ed3cfa4c31a4a9e8b4f9c085b29a14f80250a0ff`.

| Route | Requests | Prompt sum | Prompt p50 / p95 / max | Output sum | Output p50 / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold Gemma | 17 | 11,476 | 579 / 1,766 / 1,884 | 6,919 | 423 / 781 / 837 |
| Hot Qwen | 57 | 22,367 | 148 / 775 / 8,685 | 4,686 | 36 / 217 / 1,234 |

The cold path used Gemma 4 12B Q4_0 on eight physical i9-12900K P cores.
The offload path split every layer's FFN operator between the same CPU cores
and OP15 HTP. It used 8,192 columns for up to 128 prefill tokens and 9,664
columns otherwise, including decode. The phone worker held both layouts and
accepted up to 512 rows per call. Long prompts were internally chunked into
512-row ubatches.

The hot path used Qwen3 14B Q4_K_M fully resident on the RTX 4060 Ti 16 GB.
The GPU server and phone bridge used E cores. CPU and GPU frequencies remained
at their default policies.

## Complete runs

| Cold route | Qwen GPU trace | Prefill | Decode | Cold service | Makespan | Cold output rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Full CPU | Absent | 370.632 s | 1757.775 s | 2128.416 s | 2131.200 s | 3.247 tok/s |
| Full CPU | Present | 371.905 s | 1762.112 s | 2134.027 s | 2136.819 s | 3.238 tok/s |
| CPU + OP15 | Absent | 216.023 s | 1548.479 s | 1764.512 s | 1767.293 s | 3.915 tok/s |
| CPU + OP15 | Present | Censored | Censored | Censored | Censored | Censored |

For full CPU, adding the real Qwen trace changed cold service by +0.264%.
This is noise-scale for a one-repeat experiment.

## Four-way matched prefix

The first hot CPU + OP15 attempt completed 14 cold requests before USB reset.
The worker completed 248,544 expected calls through request 65, then reset 380
calls into request 66. This gives a valid matched prefix for all four cases:

| Cold route | Qwen GPU trace | Prefill | Decode | Cold service |
| --- | --- | ---: | ---: | ---: |
| Full CPU | Absent | 310.529 s | 1306.340 s | 1616.876 s |
| Full CPU | Present | 311.546 s | 1313.204 s | 1624.758 s |
| CPU + OP15 | Absent | 180.831 s | 1128.597 s | 1309.436 s |
| CPU + OP15 | Present | 184.607 s | 1125.695 s | 1310.309 s |

Across this valid prefix, the hot trace changed CPU + OP15 service by only
+0.067%. The aggregate hides one localized overlap effect:

| OP15 cold subset | Hot overlap | Prefill delta | Decode delta | Service delta |
| --- | ---: | ---: | ---: | ---: |
| Request 0, 857 + 192 tokens | 92% | +18.41% | +0.03% | +5.37% |
| Request 1, 245 + 423 tokens | 53% | +7.87% | -0.00% | +0.40% |
| Requests 2 through 65 | 0% | +0.31% | -0.29% | -0.21% |

Active GPU work affected the large prefill transfer and desktop branch, but
not decode. Keeping Qwen resident in 13.1 GiB of VRAM after its queue drained
did not affect the cold path.

## Phone and transport breakdown

The no-hot CPU + OP15 run completed 334,800 physical FFN RPCs without fallback:

| Shape class | Calls | Phone compute p50 | USB p50 | Complete RPC p50 | Overlap p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decode, one row | 333,216 | 1.789 ms | 2.210 ms | 2.307 ms | 2.334 ms |
| Prefill, multiple rows | 1,584 | 22.058 ms | 62.569 ms | 64.508 ms | 85.237 ms |

For the decode-dominated population, USB IN was the main transport component:
2.117 ms p50 versus 0.092 ms for USB OUT. Decode improves less than prefill
because every token still synchronizes 48 FFN splits and growing KV attention
stays on the desktop CPU.

Prefill has larger transfers, but enough CPU computation overlaps them to
produce the larger model-level gain. In the complete run, prefill fell 41.7%
while decode fell 11.9%.

## USB reliability finding

No full hot GPU + OP15 result is reported because both physical attempts
failed closed at the measurement level:

1. Attempt 1 reached 248,924 worker calls. The host logged `reset SuperSpeed
   USB device`, the bridge returned `LIBUSB_ERROR_NO_DEVICE`, and FunctionFS
   reported endpoint shutdown.
2. After a full phone reboot and temporary DMA-kernel boot, attempt 2 failed at
   3,169 calls. The host again reset the SuperSpeed device, the bridge returned
   `LIBUSB_ERROR_TIMEOUT`, and FunctionFS input requeue saw endpoint shutdown.

The current driver continued its local partial FFN branch after bridge loss,
so post-reset request times and tokens are invalid and are excluded. The next
transport change should make bridge loss immediately fatal to the driver, then
add session restart and protocol resynchronization if reset recovery is needed.
Average DMA bandwidth is not the blocker in these two failures.

## Artifacts

- Timeline graph: `burstgpt_gpu_cpu_op15_v1/results/source_length_2x2_v1/SOURCE_LENGTH_2X2_TIMELINE_V1.svg`
- Comparison graph: `burstgpt_gpu_cpu_op15_v1/results/source_length_2x2_v1/SOURCE_LENGTH_2X2_COMPARISON_V1.svg`
- Reduced analysis: `burstgpt_gpu_cpu_op15_v1/results/source_length_2x2_v1/ANALYSIS.json`
- Reducer: `burstgpt_gpu_cpu_op15_v1/analyze_source_2x2.py`
- Timeline plotter: `burstgpt_gpu_cpu_op15_v1/plot_source_2x2_timeline.py`
- Plotter: `burstgpt_gpu_cpu_op15_v1/plot_source_2x2.py`
