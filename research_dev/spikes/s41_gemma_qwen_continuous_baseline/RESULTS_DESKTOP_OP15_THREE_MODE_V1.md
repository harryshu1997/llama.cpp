# Desktop plus OP15 three-mode result

Date: 2026-08-01

## Verdict

`C1_FULL_TRACE_PASS; C2_RAW_TRACE_COMPLETE_RESOURCE_FAIL; C3_OPERATOR_PROTOTYPE_LATENCY_PASS_STRICT_CORRECTNESS_FAIL`

This experiment compares the three requested configurations on the physical
RTX 4060 Ti desktop. C1 and C2 execute both complete model routes over the
74-request S41 BurstGPT-derived trace. C3 is not yet a complete second-model
route: it runs a real Qwen GPU workload concurrently with a synthetic,
Gemma-shaped CPU layer whose FFN suffix is offloaded to OP15.

## Bound hardware and workload

- Desktop: `zhihao-Z690-C-ac`, i9-12900K, 30 GiB RAM.
- GPU: RTX 4060 Ti 16 GiB,
  `GPU-3d43c513-5a75-7fd0-a503-da920e0ffa08`.
- Phone: OP15 `3C15AU002CL00000`, HTP v81, direct AOA over a measured
  5,000 Mbit/s USB link.
- Gemma route: `gemma-4-12B-it-Q8_0-7b56.gguf`, 12,669,645,856 bytes.
- Qwen route: `Qwen3-14B-Q4_K_M.gguf`, 9,001,752,960 bytes.
- Trace: 74 requests, 57 Gemma and 17 Qwen, 592 generated tokens, nine model
  switches, eight-token outputs, and a 30-second request SLO.

The complete raw evidence is stored on the desktop under:

```text
/home/zhihao/s41_results/desktop_op15_v1_20260801
```

## Result summary

| Mode | Scope | Throughput | P95 TTFT | P95 completion | Resource verdict |
| --- | --- | ---: | ---: | ---: | --- |
| C1: one GPU, warm model switching | full 74-request trace | 9.803 tok/s | 8.365 s | 8.829 s | PASS |
| C2: Gemma GPU, Qwen CPU | full 74-request trace | 9.644 tok/s | 7.369 s | 16.629 s | FAIL_SWAP_GROWTH |
| C3: Qwen GPU plus CPU/OP15 split | real concurrent operator probe | CPU layer 7.102 ms | not applicable | not applicable | prototype only |

C1 completed 74/74 requests and met 74/74 SLOs. Its selected-GPU-board
energy was 3,611.716 J, or 6.101 J per generated token. This excludes CPU,
DRAM, storage, phone, and PSU losses. The maximum publication gap at a model
switch was 3.390 seconds.

C2 also completed 74/74 requests, but it is not a valid baseline result. The
two server processes accumulated 444,252,160 bytes of swap and system swap
grew by 1,372,205,056 bytes. Its latency and 3,252.704 J GPU-board reading are
diagnostic only. A paper-eligible repetition requires swap to be disabled
before both servers start.

## C3 physical concurrent operator result

Three paired repetitions kept eight Qwen3-14B requests active on all eight
llama-server slots throughout each measured operator process. Each GPU wave
generated 3,072 tokens. The second path ran a Gemma-shaped Q8_0 layer on the
i9 with eight CPU threads and a cold 128 MiB cache eviction. OP15 owned a
1,792-column FFN suffix and returned one 3,840-element f16 residual over AOA.

| Metric | CPU-only | CPU plus OP15 | Change |
| --- | ---: | ---: | ---: |
| median of run medians | 7.983 ms | 7.102 ms | -11.03% |
| median of run p90 values | 8.424 ms | 7.700 ms | -8.60% |
| concurrent Qwen GPU throughput | 134.217 tok/s | 141.003 tok/s | +5.06% |
| mean GPU board power | 162.715 W | 163.560 W | +0.52% |

The OP15 leg was 1.241 ms median and the late CPU merge was 0.0156 ms median.
The phone leg therefore fit inside the 5.29 ms CPU overlap span rather than
extending the critical path. The apparent Qwen throughput increase is an
exploratory co-scheduling observation, not a claimed phone speedup: the
paired sample count is three and the operator processes do not run for the
entire GPU response window.

The fixed-input path check had relative L2 error 0.001642 and exact hidden
argmax. The separate 30-input changing-activation gate had no non-finite
values and maximum relative L2 0.001812, but one hidden argmax differed.
Therefore C3 passes the latency mechanism test and fails the frozen strict
correctness gate. It must not be presented as a full Gemma route or a full
BurstGPT result.

## Interpretation

On this desktop, keeping the second large dense model on CPU does not increase
trace throughput relative to warm GPU switching and makes p95 completion much
worse. The useful phone result is narrower: a resident, high-arithmetic FFN
slice can reduce the CPU route's layer time while the GPU independently serves
the other model. This supports integrating the existing split into a real
decoder layer, but it does not yet establish end-to-end phone benefit.

The next bounded experiment is:

1. Disable desktop swap and repeat C2 for a valid full-model control.
2. Integrate one path-matched FFN suffix into the actual CPU decoder without
   changing scheduling policy or adding another offload type.
3. Enforce a publication-level numerical gate, then replay the same 74
   requests for C3.

## Evidence identifiers

- C1 raw run: `c1_gpu_switch_warm_r104`.
- C1 manifest SHA256: `71f60566ddcc68154bc3e6cf9365ce9a4a8a6859dbbaa990e0e00b7b899e7697`.
- C2 raw run: `c2_gemma_gpu_qwen_cpu_r104`.
- C2 manifest SHA256: `5b0ac5e4d13de7ae46d0f8877a6108d94238de098ff09d072a35a81d452a82b4`.
- Normalized reduction SHA256: `cc390469a972bd9670316ef235432de3ec5c8577a4bb498515bbdef41f620f3c`.
- C3 paired runs: `gpu_corun_r101`, `gpu_corun_r102`, and
  `gpu_corun_r103`.
- C3 manifest SHA256 values, in run order: `030303cd11c8a4d96275ad71e79eedd451a144cef48f1545aab4606dbc555f6e`,
  `71704b0b7c2bc5684ef0782d49bbd8f129b44a8a85440d9b070f5c240383bada`,
  and `27be222c81e7b8990aa73234bfe1a36b005f5101fd798dd419baa63859da3c37`.

To make the full-model runs possible, two unused tmpfs environments were
moved without deletion to:

```text
/home/zhihao/s41_tmpfs_archive_20260801/tierkv-baseline-eval
/home/zhihao/s41_tmpfs_archive_20260801/tierkv-length-eval-venv
```
