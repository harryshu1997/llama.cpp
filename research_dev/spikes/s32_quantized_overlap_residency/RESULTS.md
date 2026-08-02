# S32 Quantized Overlap Residency Results

## Verdict

`CAPACITY_GAIN_PASS_NUMERIC_ROUTE_FAIL`

Q8_0 lets both phones hold deeper B32 stages than the selected S31 F16 cut,
but it does not qualify an exact scheduler route. Q8_0 and Q4_0 both miss the
frozen HTP-versus-CPU relative-L2 limit, and a diverse end-to-end token test
shows output divergence. No quantized phone row is scheduler-eligible.

## Frozen artifacts

All three Q8_0 copies matched SHA-256
`7b56cbd0e0d96d5c8d7df9c21b39d67264e58cbabf7a96d4681eff8b3d492848`:

- host: `/home/myid/zs89458/Documents/models/gemma-4-12B-it-Q8_0.gguf`
- OP12 and OP15: `/data/local/tmp/ls-s32/gemma-4-12B-it-Q8_0.gguf`

The Android worker SHA-256 was
`01aab37abaf3af658edb1856f333636d7b2f2ac14661b7b6cc953f4b6ac8600f`.
The one-GPU CUDA worker SHA-256 was
`803672c04104f1a354789a1ded5169a4047dd5e474d247290fcb30dba15f0f36`.

## Numerical screen

The frozen gate was relative L2 at most 0.005 and cosine at least 0.999 for
every tested position, using the same GGUF on CPU and HTP.

| Format | Device and range | Relative-L2 range | Cosine range | Verdict |
| --- | --- | ---: | ---: | --- |
| Q8_0 | OP15 `[0,2)` | 0.01149-0.01356 | 0.999908-0.999934 | FAIL |
| Q4_0 | OP15 `[0,2)` | 0.01151-0.01928 | 0.999814-0.999934 | FAIL |

Both phone paths were deterministic across repeats. The failure is the
cross-backend numerical difference, not nondeterminism. Per the frozen plan,
Q4_0 stopped after this screen and Q8_0 continued only as `PERF_ONLY` capacity
evidence.

## B32 capacity screen

Each complete row contains seven measured cohorts after two discarded
warmups. Placement is HTP0 for all compute except the declared stage-zero
`GET_ROWS` CPU operation. Larger intervals are excluded at the first nonzero
process-swap or compute failure.

| Device | Q8_0 interval | B32 median/step | p95 | Median/layer | VmHWM | Swap | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OP12 | `[0,2)` | 391.606 ms | 395.270 ms | 195.803 ms | 5.82 GiB | 0 | capacity pass |
| OP12 | `[0,4)` | 762.129 ms | 767.581 ms | 190.532 ms | 5.63 GiB | 0 | largest clean tested |
| OP12 | `[0,5)` | 945.288 ms | 950.952 ms | 189.058 ms | 5.12 GiB | 85.6 MiB | excluded |
| OP15 | `[0,2)` | 58.297 ms | 63.647 ms | 29.149 ms | 5.14 GiB | 0 | capacity pass |
| OP15 | `[4,16)` | 222.800 ms | 227.735 ms | 18.567 ms | 8.70 GiB | 0 | largest clean tested |
| OP15 | `[4,17)` | 264.006 ms | 267.724 ms | 20.308 ms | 10.08 GiB | 13.5 MiB | excluded |

OP12 `[0,6)` and `[0,8)` also completed but used about 85 MiB swap and are
excluded. OP15 `[4,20)` with a 6144 MiB Hexagon VMEM setting and `[4,24)`
both loaded, then aborted in DSP execution. Increasing the VMEM reservation
did not recover a clean larger interval.

One OP15 `[4,16)` attempt overlapped an unrelated HTP process and timed out.
It is retained as contamination evidence and excluded. The table uses the
subsequent exclusive run.

## End-to-end token gate

The real topology was:

```
32 requests -> OP12 Q8 [0,4) -> OP15 Q8 [4,16) -> A6000 Q8 [16,48)
```

The reference used the same Q8_0 GGUF and one physical A6000:

```
32 requests -> A6000 Q8 [0,16) -> A6000 Q8 [16,48)
```

The first diagnostic cloned BOS token 2 across all 32 rows. It matched 32/32
four-token sequences, but that result is not general evidence. The strengthened
v2 run used distinct initial tokens 2 through 33:

| Metric | Physical phone chain | One-GPU reference |
| --- | ---: | ---: |
| Median B32 decode step | 1088.570 ms | 54.957 ms |
| Fully matching four-token requests | 18/32 | reference |
| Matching token decisions | 101/128 | reference |
| Placement certificates | 4/4 pass | included |

The physical route is about 19.81x slower than the one-GPU reference at this
unbalanced cut. OP12 is the dominant stage. The test is a correctness and
plumbing gate, not a proposed production route or a throughput result.

The independent `validate_chain.py` binder validates the model hashes,
topology, timing, output digests, process identities, and four placement
certificates, then records `TOKEN_EXACT_FAIL` and exits nonzero. The run also
observed 69,852 KiB OP15 process swap, so it has an independent resource
failure. The earlier exclusive OP15 capacity row remains the only zero-swap
evidence for `[4,16)`.

## Interpretation

Quantization creates more resident layer capacity, but capacity alone does not
create valid scheduling freedom. Under the project's exact-route contract,
the Q8_0 and Q4_0 phone kernels cannot enter the route catalog. The next
prototype should therefore use overlapping F16 resident windows and select
among a finite set of measured entry and exit points without reloading. If the
project later accepts approximate inference, it needs a new, predeclared
natural-workload task-quality gate rather than a retroactive relaxation of the
0.005 activation gate.

No phone, network, GPU-board, or total-system energy claim is made here.
