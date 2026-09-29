# Gemma4 FFN DMA-BUF width sweep

Date: 2026-08-03 EDT.

Verdict: `WIDER_FFN_OFFLOAD_SPEED_PASS; 5632_COLUMNS_STABLE; EXACT_TOKEN_GATE_IS_PROMPT_DEPENDENT`.

## Result

Increasing the OP15 suffix from 1792 to 5632 columns reduced median Gemma4
12B CPU decode time by 23.66% and increased decode throughput by 30.99%.
The wider phone branch remained fully hidden behind the desktop FFN prefix.

The widest tested split is not bit-identical to CPU-only greedy decoding for
this prompt. The 2720-column split matched all 16 CPU tokens in every paid
request and still reduced decode time by 10.43%. The next Q8-aligned test point,
2752 columns, changed token 10 deterministically. This boundary is only a
receipt for this model, prompt, and token sequence. It is not a general
bit-exact guarantee.

## Configuration

- Model: Gemma4 12B Q8_0, K=3840, NFF=15360, 48 dense layers.
- Desktop: Intel Core i9-12900K, 8 inference threads, `-ngl 0`.
- Phone: OnePlus 15 HTP0.
- Transport: direct FunctionFS DMA-BUF with persistent libusb host memory.
- Request: 28 prompt tokens followed by 16 greedy output tokens.
- Each long process: one warmup plus five paid requests.
- CPU baseline: two processes and ten paid samples.
- The 5632-column result: two processes and ten paid samples.
- Wire size per FFN call: 15,488 bytes OUT plus 15,488 bytes IN. It does not
  grow with the split width because both boundary tensors have K=3840 F32
  elements.

## Full-model timing

The CPU baseline median across ten samples was 6416.152 ms for the 15 decode
steps and 7273.252 ms for the complete 16-token request.

| phone columns | FFN share | resident weights | paid samples | decode | decode change | request wall | wall change | CPU token match |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 2720 | 17.71% | 1524.03 MiB | 5 | 5746.852 ms | -10.43% | 6661.456 ms | -8.41% | yes |
| 3072 | 20.00% | 1721.25 MiB | 5 | 5644.174 ms | -12.03% | 6498.405 ms | -10.65% | no |
| 4096 | 26.67% | 2295.00 MiB | 5 | 5347.673 ms | -16.65% | 6217.991 ms | -14.51% | no |
| 5120 | 33.33% | 2868.75 MiB | 5 | 5045.438 ms | -21.36% | 5921.785 ms | -18.58% | no |
| 5632 | 36.67% | 3155.63 MiB | 10 | 4898.145 ms | -23.66% | 5757.293 ms | -20.84% | no |

The two useful operating points are:

| profile | decode throughput | gain | request throughput | gain |
| --- | ---: | ---: | ---: | ---: |
| CPU only | 2.338 token/s | - | 2.200 token/s | - |
| 2720 columns | 2.610 token/s | +11.65% | 2.402 token/s | +9.18% |
| 5632 columns | 3.062 token/s | +30.99% | 2.779 token/s | +26.33% |

The 5632-column percentage is a performance result, not a bit-exact quality
result. Its generated path diverges after token 9, although every request is
deterministic and every transport hash and protocol check passes. A corpus-level
logit or perplexity tolerance test is required before using this profile as a
quality-certified default.

## Overlap timing

| phone columns | model RPC p50 | model RPC p90 | HTP compute p50 | desktop prefix p50 | desktop wait p50 | USB round trip p50 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2720 | 1.201 ms | 1.429 ms | 0.690 ms | 5.349 ms | 0.00017 ms | 1.134 ms |
| 3072 | 1.318 ms | 1.534 ms | 0.774 ms | 5.209 ms | 0.00020 ms | 1.234 ms |
| 4096 | 1.594 ms | 1.727 ms | 1.018 ms | 4.796 ms | 0.00018 ms | 1.512 ms |
| 5120 | 1.816 ms | 1.929 ms | 1.249 ms | 4.383 ms | 0.00017 ms | 1.741 ms |
| 5632, run 1 | 1.938 ms | 2.074 ms | 1.361 ms | 4.177 ms | 0.00018 ms | 1.854 ms |
| 5632, run 2 | 1.928 ms | 2.050 ms | 1.364 ms | 4.176 ms | 0.00016 ms | 1.852 ms |

At 5632 columns, the phone RPC still finishes about 2.24 ms before the desktop
prefix at the median. The split is limited by HTP resident-weight memory, not
by the overlap balance. Its 3155.63 MiB weight buffer consumes 98.6% of the
configured 3200 MiB HTP virtual-memory arena. A 5696-column weight buffer would
leave too little room for graph tensors and allocator overhead, so it was not
used as the stable endpoint.

For one layer and one token, the 2720-column phone suffix evaluates about
62.67 MFLOP. The 5632-column suffix evaluates about 129.76 MFLOP. Across all
48 layers, those values are about 3.01 and 6.23 GFLOP per decoded token.

## Correctness and safety receipts

- Every long treatment completed 4320 request/response/HTP calls with status 0.
- Both 5632-column processes produced the same token sequence and nearly
  identical timing.
- The 2720-column process matched the CPU token sequence in all five paid
  requests.
- 2720 columns matched a 16-token gate; 2752 columns changed token 10.
- All protocol metadata, activation hashes, response hashes, and finite-value
  checks passed.
- The fixed-kernel fault filter was empty after the campaign.
- The phone was restored by a normal reboot to stock kernel
  `6.12.23-android16-5-gb3b66ace21e0-ab14672634-4k`, `ptp,adb`, SuperSpeed,
  the original gadget, and no FFN worker.

Raw logs and SHA-256 receipts are stored on the desktop at:

```
/home/zhihao/s41-opoffload-dmabuf-v1/wider-v1/
```

## Recommendation

For a Q8_0 phone suffix, use 2720 columns for the current strict regression
gate and 5632 columns for performance experiments. The hybrid Q8_0/Q4_0 sweep
in `RESULTS_FFN_DMABUF_Q4_BALANCE_V1.md` is an exploratory hardware ceiling,
not a same-model replacement. The corrected byte-identical Q4_0 profile is in
`RESULTS_FFN_DMABUF_Q4_SAME_MODEL_V1.md`: 9088 phone columns reduced decode
time by 33.58% and matched all 16 tokens in 15 paid requests. The runtime
planner should select from a validated width table, not infer output
equivalence from transport integrity or from one greedy token sequence.
