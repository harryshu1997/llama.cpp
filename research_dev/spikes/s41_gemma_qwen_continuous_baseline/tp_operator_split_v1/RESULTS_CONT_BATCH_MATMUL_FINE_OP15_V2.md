# Fine continuous-batch matmul crossover on OP15

Date: 2026-08-06 EDT

## Verdict

For the i9-12900K desktop CPU and OP15 HTP combination, route by the physical
matmul row count `M`, not by the number of active requests:

- Q/K/V projection: first median win at `M=10`; safe initial gate at `M>=12`.
- Attention output projection: first raw median win at `M=36`; safe initial
  gate at `M>=48` after reserving 0.150 ms for the output join and integration
  overhead.
- Existing FFN split: remains useful at decode and prefill shapes using its
  separately measured dynamic-width policy.
- Vocabulary head: the prior operator sweep found useful splits for `M=1..8`,
  but this path still requires a compatible sampler contract.

The Q/K/V and output-projection values are physical CPU and HTP kernel
measurements combined with the previously measured 465 MB/s direct DMA path
and 0.235 ms RPC floor. They are not yet end-to-end implementations in
llama-server.

## Fine crossover sweep

The CPU and phone sweeps ran concurrently on the 4060 Ti host and OP15. The
desktop GPU was idle because these cold-model operators execute on the CPU and
phone. The CPU used eight pinned P-core threads at the default frequency.
Each shape used three warmups and eleven measured iterations.

### Q/K/V projection

`Combined` is the slower of the desktop remainder and the complete modeled
phone branch. The gate requires at least 10 percent gain and 0.100 ms saving
at both p50 and p90.

| M | CPU p50 | selected split p50 | p50 gain | p90 gain | safe |
| ---: | ---: | --- | ---: | ---: | --- |
| 8 | 0.772 ms | CPU, 0.772 ms | 0.0% | 0.0% | no |
| 10 | 0.945 ms | phone half-Q, 0.813 ms | 14.0% | 6.9% | no |
| 12 | 1.142 ms | phone half-Q, 0.915 ms | 19.9% | 18.3% | yes |
| 16 | 1.473 ms | phone half-Q, 1.140 ms | 22.6% | 23.6% | yes |
| 24 | 2.272 ms | phone Q, 1.632 ms | 28.2% | 27.8% | yes |
| 32 | 2.949 ms | phone K+V, 1.923 ms | 34.8% | 35.0% | yes |
| 48 | 4.432 ms | phone Q, 2.454 ms | 44.6% | 44.6% | yes |
| 64 | 5.874 ms | phone Q, 3.016 ms | 48.7% | 48.8% | yes |

At `M=32`, phone Q and phone K+V are effectively tied: K+V wins p50 while Q
wins p90. Phone Q is the more stable initial route from `M=24` through the
fine-sweep range. The prior coarse `M=128` and `M=512` measurements favor
phone K+V with desktop Q.

### Attention output projection

The phone and CPU each evaluate a 2048-wide K slice. `Reserved` adds 0.150 ms
to the raw overlapped result for the final desktop sum and unmeasured
integration overhead. The same 10 percent and 0.100 ms p50/p90 gate is used.

| M | CPU p50 | raw split p50 | reserved p50 | reserved p50 gain | safe |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 1.468 ms | 1.493 ms | 1.643 ms | -11.9% | no |
| 36 | 1.663 ms | 1.509 ms | 1.659 ms | 0.3% | no |
| 40 | 1.834 ms | 1.643 ms | 1.793 ms | 2.3% | no |
| 44 | 2.013 ms | 1.777 ms | 1.927 ms | 4.3% | no |
| 48 | 2.285 ms | 1.837 ms | 1.987 ms | 13.1% | yes |
| 64 | 3.086 ms | 2.243 ms | 2.393 ms | 22.5% | yes |
| 96 | 4.660 ms | 3.088 ms | 3.238 ms | 30.5% | yes |
| 128 | 6.046 ms | 3.937 ms | 4.087 ms | 32.4% | yes |

## Real mixed prefill and decode probe

A controlled llama-server run verified the physical batch semantics with the
real FunctionFS DMA-BUF and HTP path:

- `--parallel 2`, `--batch-size 256`, `--ubatch-size 64`
- continuous batching and unified KV enabled
- one FFN layer offloaded to isolate physical graph shapes
- request A: 8 prompt tokens and 32 decode tokens
- request B: 56 prompt tokens and 2 decode tokens
- request B was dispatched after request A streamed its first token; request A
  was still active

The allocator log records the first combined graph as:

```text
n_tokens = 53, n_seqs_unq = 2
sequence 0 positions = [0, 51]
sequence 1 positions = [9, 9]
```

This is 52 prefill rows plus one decode row in the same physical matmul. The
next graph was `M=5`: four remaining prefill rows plus one decode row. The
phone summary independently recorded one `M=53` call and one `M=5` call.

The probe completed 35 server, bridge, and phone calls with matching counts,
zero reset recoveries, and status `ok`. The phone restored its normal
`ptp,adb` USB configuration automatically.

This confirms that low request concurrency does not imply a small matmul. For
a physical graph containing `D` decode rows and `P` prompt rows:

```text
M = D + P
QKV offload when M >= 12
output-projection offload when M >= 48
```

For one active decoder, Q/K/V needs at least 11 prompt rows in that graph and
the output projection needs at least 47. The scheduler must use the actual
physical graph `n_tokens`, since prompt chunking and `n_ubatch` can change `P`.

## Initial routing policy

```text
M < 12:       CPU QKV
12 <= M < 24: phone half-Q, CPU other half-Q + K + V
24 <= M < 128: phone Q, CPU K + V
M >= 128:    phone K + V, CPU Q

M < 48:      CPU output projection
M >= 48:     split output projection 50/50 along K
```

Use the existing dynamic FFN policy independently. Every route is selected
for each physical graph, so a request can use a large prefill route and then
return to the decode route without moving or repacking resident weights.

## Evidence

Reproducible analysis:

- `continuous_matmul_sweep_v1/analyze_fine_cross.py`

Raw results on the 4060 Ti host:

- `/home/zhihao/s41-cont-matmul-v1/results/cpu_fine_cross.jsonl`
- `/home/zhihao/s41-cont-matmul-v1/results/phone_fine_cross.jsonl`
- `/home/zhihao/s41-cont-matmul-v1/results/ANALYSIS_FINE_CROSS.json`

Mixed-batch server artifacts on the 4060 Ti host:

- `/home/zhihao/s41-cont-matmul-v1/mixed-batch-m56-v1/server.stderr`
- `/home/zhihao/s41-cont-matmul-v1/mixed-batch-m56-v1/bridge.stderr`
- phone session:
  `/data/local/tmp/s41-opoffload-dmabuf-v1/mixed-batch-m56-v3/worker.log`
