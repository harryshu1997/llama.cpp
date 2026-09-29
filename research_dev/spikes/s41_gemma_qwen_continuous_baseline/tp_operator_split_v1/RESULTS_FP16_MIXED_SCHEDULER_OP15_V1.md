# FP16 mixed scheduler with RTX 4060 Ti and OP15

## Result

The FP16-weight experiment passed. Once Gemma is loaded, adding the OP15 HTP
split consistently improves the 17-request cold BurstGPT backlog.

| Cold Gemma route | Paid makespan | Output rate | Result |
| --- | ---: | ---: | --- |
| CUDA 25/49 + CPU + OP15 | 712.861 s | 9.706 tok/s | PASS |
| CUDA 25/49 + CPU control 1 | 768.272 s | 9.005 tok/s | inference complete, reporting failure |
| CUDA 25/49 + CPU control 2 | 781.483 s | 8.854 tok/s | inference complete, reporting failure |
| CUDA 25/49 + CPU control 3 | 789.753 s | 8.761 tok/s | PASS |

The three control makespans average 779.834 s. Relative to that mean, OP15:

- reduces cold execution time by 8.59%;
- raises cold output throughput by 9.39%;
- reduces time by 7.21% to 9.74% against every individual control.

The first two controls completed all 17 requests and 6,919 output tokens. They
were rejected only after paid inference, when optional energy integration did
not have a sample outside both timing boundaries. Control 3 includes the
corrected sampling guards and is the formal control result.

The full physical phone-assisted trace completed all 74 requests and all
11,605 requested output tokens in 2,646.528 s. Its phases were:

| Phase | Time |
| --- | ---: |
| Protected Qwen CUDA + CPU | 1,885.665 s |
| Qwen unload + Gemma load + warm | 48.002 s |
| Gemma CUDA + CPU + OP15 | 712.861 s |

Repeating the 31-minute Qwen phase does not change the cold-route comparison.
Using the same measured Qwen phase and each no-phone control gives reconstructed
whole-trace times of 2,673.273 s, 2,686.216 s, and 2,694.454 s. The mean is
2,684.648 s. The phone therefore reduces reconstructed full-trace time by
1.42% on average, with a 1.00% to 1.80% range.

The full-trace gain is smaller than the steady cold gain because phone-compatible
CPU weight storage adds about 29 seconds to the one-time Gemma switch.

## Artifacts and capacity

The remote performance artifacts use true F16 tensor storage, but their values
were dequantized from the existing Q4 checkpoints. This preserves tensor sizes,
placement, memory traffic, and compute timing. It does not reproduce the
numerical quality of the original FP16 checkpoints.

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3 14B F16-storage proxy | 29,543,423,360 | `d89e9e823744222e595e0b3c8fd5436ce5d3a6a446fa42492ebce6064dfa9718` |
| Gemma4 12B F16-storage proxy | 23,832,065,056 | `ed76f2183d2d1d65091986033023e6c78d27f6276c1b0c5826cc92acf73538cf` |

Together the files occupy 53.4 GB, which exceeds the desktop's 32.9 GB RAM.
Neither model fits the 16 GB RTX 4060 Ti by itself. The scheduler therefore
keeps one model active at a time and uses a CPU prefix plus CUDA suffix.

Qwen used 18/41 CUDA loads:

- CPU-mapped model buffer: 15,974.67 MiB;
- CUDA model buffer: 12,194.45 MiB;
- CPU KV: 2,208 MiB;
- CUDA KV: 1,632 MiB.

Gemma used 25/49 CUDA loads:

- CPU-mapped model buffer: 12,316.46 MiB;
- CUDA model buffer: 12,316.48 MiB;
- CPU KV buffers: 1,616 MiB total;
- CUDA KV buffers: 1,616 MiB total;
- CUDA compute buffer: 277.00 MiB.

Every accepted run used `MemorySwapMax=0`. Peak process swap was zero.

## Operator placement

The phone is not assigned whole layers. It runs a parallel slice of each dense
gated FFN in Gemma layers 0 through 22, which are the CPU-resident prefix.
Gemma layers 23 through 47 remain on CUDA.

For each eligible CPU FFN:

```text
intermediate columns: 15,360
desktop CPU branch:    columns 0 through 9,215   (9,216 columns, 60%)
OP15 HTP branch:       columns 9,216 through 15,359 (6,144 columns, 40%)
merge:                 desktop after both branches complete
```

The phone holds 3,105.05 MiB of F16 FFN weights in HTP0. The runtime policy is:

```text
M <= 16: offload 6,144 columns
M > 16:  keep the complete operator local
```

This policy lets large prefill matmuls avoid the staged USB path while decode
and small mixed prefill/decode graphs use the phone. The desktop and phone use
the same Gemma artifact and F16 weights. Activations also use F16 on the wire.

## Continuous-batch overlap

The physical trace issued 27,485 phone FFN calls. All completed with zero reset
recoveries. The bridge moved 1,214,223,360 bytes in each direction.

| Graph M | Calls | RPC mean | Phone compute mean | Host branch mean | Host wait mean | Overlapped mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5,014 | 3.722 ms | 2.745 ms | 7.362 ms | 0.002 ms | 7.363 ms |
| 2 | 3,473 | 4.342 ms | 3.050 ms | 7.417 ms | 0.002 ms | 7.420 ms |
| 4 | 506 | 5.638 ms | 4.141 ms | 7.486 ms | 0.010 ms | 7.496 ms |
| 5 | 460 | 8.560 ms | 6.991 ms | 7.493 ms | 1.201 ms | 8.694 ms |
| 7 | 1,380 | 8.921 ms | 7.043 ms | 7.595 ms | 1.508 ms | 9.103 ms |
| 8 | 16,100 | 9.045 ms | 7.065 ms | 7.640 ms | 1.604 ms | 9.244 ms |
| 11 | 184 | 10.097 ms | 7.671 ms | 7.817 ms | 2.542 ms | 10.359 ms |

At M=1 through M=4, phone work is almost completely hidden by the CPU branch.
At M=5 through M=11, the fixed 6,144-column phone branch becomes critical and
adds 1.2 to 2.5 ms of wait per eligible layer. It still beats computing the
full 15,360-column FFN on the CPU, but it is the clearest remaining operator
policy optimization target.

The FunctionFS DMA-BUF bridge reported:

- decode USB p50: 3.563 ms;
- multi-token USB p50: 8.834 ms;
- all-call USB p90: 9.018 ms;
- reset recoveries: 0.

The worker's all-call compute p50 was 7.026 ms. The phone stayed on the patched
FunctionFS kernel after the run, restored ADB normally, and logged no kernel
fault.

## Latency and energy

The formal comparison is control 3 against the phone-assisted cold phase.

| Metric | CUDA + CPU | CUDA + CPU + OP15 | Change |
| --- | ---: | ---: | ---: |
| Makespan | 789.753 s | 712.861 s | -9.74% |
| Output rate | 8.761 tok/s | 9.706 tok/s | +10.79% |
| Mean prompt time | 50.882 s | 38.704 s | -23.93% |
| P50 prompt time | 33.446 s | 33.132 s | -0.94% |
| Mean decode time | 258.003 s | 233.589 s | -9.46% |
| P50 decode time | 289.168 s | 267.224 s | -7.59% |
| Maximum decode time | 552.977 s | 510.283 s | -7.72% |
| Server compute-device energy | 93.097 kJ | 79.656 kJ | -14.44% |

The energy values include Intel package RAPL and RTX board power. They exclude
the phone, motherboard, DRAM outside package RAPL, fans, storage, and AC losses.
Using the prior conservative 4.5 W whole-phone screen for the entire 712.861 s
cold phase adds 3.208 kJ. The resulting 82.864 kJ estimate is still 10.99%
below the measured 93.097 kJ server control, but this is an estimate rather
than a phone power measurement from this run.

GPU utilization during the phone-assisted cold phase averaged 39.85%, with a
16% p50 and 100% p95. This wide range is expected: each layer alternates among
the CPU prefix, its overlapped phone FFN branch, and the CUDA suffix.

## Scheduler rule

For this exact placement, use the phone only when the expected saved cold
execution exceeds its additional load cost:

```text
use_phone = (T_cpu_gpu - T_cpu_gpu_phone) >
            (L_phone_compatible - L_standard)
```

Measured here:

```text
steady cold time reduction: 8.59% versus the three-control mean
additional load and warm cost: about 29.0 s
break-even no-phone cold work: about 29.0 / 0.0859 = 338 s
```

The 779.8 s control backlog clears this gate. A short cold burst below roughly
338 s should use CUDA plus CPU without the phone unless the view-safe weights
are already resident.

## Bottlenecks and next optimization

The route is qualified, but it is not globally optimal.

1. The main one-time cost is view-safe CPU weight storage. The standard control
   loads and warms in about 18.2 s; the phone route takes 47.1 s after Qwen
   unload. A mmap-compatible split view or persistent prepared CPU weights
   would expose most of the 8.6% steady-state gain at full-trace level.
2. M=5 through M=11 should use fewer than 6,144 phone columns. The current
   measurements indicate that a shape-dependent split near 5,120 to 5,632
   columns may balance the phone and host branches better. This needs a physical
   calibration before changing the qualified policy.
3. Large prefill remains local. Direct HTP-buffer USB previously faulted the
   phone, while the stable staged path makes large activation RPCs too costly.
4. Attention, lm_head, norms, embeddings, and the CUDA suffix remain local.
   No integrated phone route for those operators has beaten their parent path.

## Correctness and raw evidence

All phone and control runs produced the requested token counts. Four of 17 cold
token sequences exactly matched the control; the median common prefix was 55
tokens. Sampled long phone outputs were coherent. This experiment qualifies
performance and placement, not bit-exact equivalence.

Primary remote evidence:

- phone-assisted trace:
  `/home/zhihao/s41-dynamic-ffn-v1/server-traces/fp16-hierarchical-op15-v2-20260806T2038Z`
- formal control:
  `/home/zhihao/s41-dynamic-ffn-v1/server-traces/fp16-gemma-cuda-cpu-control-v3-20260807T0158Z`
- completed control replicates:
  `fp16-gemma-cuda-cpu-control-v1-20260807T0128Z` and
  `fp16-gemma-cuda-cpu-control-v2-20260807T0142Z`

Result SHA-256 values:

- phone `RESULT.json`: `a307eb6cb4bc80042e07ab79ca4fc9b5b38298120d9d92603fa4e1e31d489887`
- control 3 `RESULT.json`: `97f701102d2136316b4c557dbf811fd9ee59fd0e4501c839b32a8fd507696147`

The scheduler policy suite passes 12 of 12 tests.
